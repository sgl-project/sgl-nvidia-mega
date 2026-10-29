# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 SGLang Team
"""Stock MLP step body (MlpSteps) and MoE override steps (MoeRecords, MoeSteps)."""

from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass
from typing import Any, Optional

import torch

from sgl_nvidia_mega.glm_dsa.attention.step import StepContext


class MlpSteps:
    """Stock dense MLP / MoE of a layer, leaving the output's sum to the next step."""

    def forward(self, layer, x, buffer: Optional[torch.Tensor], step: StepContext):
        """``buffer``: this layer's input storage, dead once the attention input is
        read (the MoE may write its output into it)."""
        from sglang.srt.layers.moe.moe_runner.base import moe_output_buffer_ctx
        from sglang.srt.models.deepseek_v2 import DeepseekV2MoE
        from sglang.srt.runtime_context import get_forward

        mlp = layer.mlp
        if (
            isinstance(mlp, DeepseekV2MoE)
            and not mlp.experts.moe_runner_config.inplace
            and buffer is not None
        ):
            buffer_ctx = moe_output_buffer_ctx(buffer)
        else:
            buffer_ctx = nullcontext()
        with get_forward().scoped(defer_moe_finalize=False), buffer_ctx:
            return mlp(x, step.forward_batch, None)


# The steps of each group, in call order (describe, rank agreement, logs).

class MoeRecords:
    """The engine's per-expert outputs of one MoE layer, for a hop that finalizes them
    inside its all-reduce (row stride ``rows``): ``pq`` [4, rows, 8, 6144] fp32 (the four
    quarters' FC2 outputs per top-k slot), ``shpartials`` [4, rows, 6144] fp32 (shared expert),
    ``ids_fin`` / ``w_fin`` / ``g2s`` [rows, 8] (slot expert id or -1, routing weight, FC2
    scale)."""

    pq: torch.Tensor
    shpartials: torch.Tensor
    ids_fin: torch.Tensor
    w_fin: torch.Tensor
    g2s: torch.Tensor


def _process_like_server(gate, up, gate_sf, up_sf, w2, w2_sf):
    """The server's NVFP4 TRT-LLM weight processing on a synthetic layer: the loader's
    w13 order for this backend (up rows first) and the same processing call."""
    from sglang.srt.layers.moe.moe_runner import flashinfer_trtllm as trtllm

    e = gate.shape[0]
    f8 = torch.float8_e4m3fn
    p = lambda t: torch.nn.Parameter(t, requires_grad=False)  # noqa: E731
    layer = torch.nn.Module()
    layer.w13_weight = p(torch.cat([up, gate], dim=1).contiguous())
    layer.w2_weight = p(w2.contiguous())
    layer.w13_weight_scale = p(torch.cat([up_sf, gate_sf], dim=1).view(f8).contiguous())
    layer.w2_weight_scale = p(w2_sf.view(f8).contiguous())
    ones = torch.ones(e, dtype=torch.float32, device=gate.device)
    layer.w2_input_scale_quant = p(ones.clone())
    layer.g1_alphas = p(ones.clone())
    layer.g1_alphas_up = p(ones.clone())
    layer.moe_runner_config = types.SimpleNamespace(is_gated=True, activation="silu")
    trtllm.align_fp4_moe_weights_for_flashinfer_trtllm(layer)
    return (
        layer.w13_weight.data,
        layer.w13_weight_scale.data,
        layer.w2_weight.data,
        layer.w2_weight_scale.data,
    )


def _u8(t: torch.Tensor) -> torch.Tensor:
    return t if t.dtype == torch.uint8 else t.view(torch.uint8)


def _bad(t, dtype, shape=None, numel=None, contiguous=True) -> bool:
    return (
        t is None
        or t.dtype != dtype
        or (shape is not None and tuple(t.shape) != shape)
        or (numel is not None and t.numel() != numel)
        or (contiguous and not t.is_contiguous())
    )


def _layer_reason(mlp) -> Optional[str]:
    """Why this MoE layer cannot run on the engine (None if it can)."""
    ex = mlp.experts
    for name, shape in (
        ("w13_weight", (EXPERTS, 2 * INTER, HIDDEN // 2)),
        ("w2_weight", (EXPERTS, HIDDEN, INTER // 2)),
    ):
        if _bad(getattr(ex, name, None), torch.uint8, shape):
            return f"{name} is not contiguous uint8 {shape}"
    for name, n in (
        ("w13_weight_scale", EXPERTS * 2 * INTER * HIDDEN // 16),
        ("w2_weight_scale", EXPERTS * HIDDEN * INTER // 16),
    ):
        if _bad(getattr(ex, name, None), torch.float8_e4m3fn, numel=n):
            return f"{name} is not contiguous e4m3 with {n} elements"
    for name in ("g1_alphas", "g1_alphas_up", "g2_alphas"):
        if _bad(getattr(ex, name, None), torch.float32, (EXPERTS,), contiguous=False):
            return f"{name} is not fp32 [{EXPERTS}]"
    if not torch.equal(ex.g1_alphas, ex.g1_alphas_up):
        return "gate and up GEMM1 scales differ (the engine applies one per expert)"
    for name in ("w13_input_scale_quant", "w2_input_scale_quant"):
        t = getattr(ex, name, None)
        if t is None or t.numel() != 1:
            return f"{name} is not a single scale"
    cfg = ex.moe_runner_config
    if (
        getattr(cfg, "activation", "silu") != "silu"
        or getattr(cfg, "gemm1_clamp_limit", None)
        or getattr(cfg, "swiglu_limit", None)
        or getattr(cfg, "gemm1_alpha", None) is not None
    ):
        return "only an unclamped SiLU-gated MLP is supported"
    gu, dn = mlp.shared_experts.gate_up_proj.weight, mlp.shared_experts.down_proj.weight
    if _bad(gu, torch.bfloat16, (2 * INTER, HIDDEN), contiguous=False):
        return "shared expert gate_up_proj is not bf16 [512, 6144]"
    if _bad(dn, torch.bfloat16, (HIDDEN, INTER), contiguous=False):
        return "shared expert down_proj is not bf16 [6144, 256]"
    bias = mlp.gate.e_score_correction_bias
    if _bad(bias, torch.float32, (EXPERTS,), contiguous=False):
        return "router correction bias is not fp32 [256]"
    return None


class MoeSteps(MlpSteps):
    """The MoE layers of one verify row count on the engine; dense layers stay stock."""

    def __init__(self, glue: MoeGlue, rows: int) -> None:
        self.glue, self.rows = glue, rows
        self.split = (VERIFY_TOKENS_PER_REQUEST,) * (rows // VERIFY_TOKENS_PER_REQUEST)

    def forward(self, layer, x, buffer: Optional[torch.Tensor], step):
        glue = self.glue
        lt = glue.layers.get(layer.layer_id)
        if lt is None:
            return super().forward(layer, x, buffer, step)
        m = x.shape[0]
        if m != self.rows or x.dtype != torch.bfloat16 or not x.is_contiguous():
            raise RuntimeError(
                f"fused verify MoE: unexpected input {tuple(x.shape)} {x.dtype} at {self.rows} rows"
            )
        routing, step.routing = step.routing, None
        if routing is not None:
            # routed by this layer's o_proj hop; its MLP-output hop finalizes the records
            ids, weights = routing
            glue.engine(lt, x, ids, weights, self.split, what=1)
            return glue.records(m)
        ids, weights = glue.ids[:m], glue.weights[:m]
        gate = layer.mlp.gate
        scale = layer.mlp.routed_scaling_factor
        glue.route(gate(x), gate.e_score_correction_bias, scale, ids, weights)
        if (
            isinstance(buffer, torch.Tensor)
            and buffer.dtype == torch.bfloat16
            and buffer.dim() == 2
            and buffer.shape[0] >= m
            and buffer.shape[1] == HIDDEN
            and buffer.is_contiguous()
        ):
            out = buffer[:m]
        else:
            out = torch.empty(m, HIDDEN, dtype=torch.bfloat16, device=x.device)
        glue.engine(lt, x, ids, weights, self.split, out=out, what=3)
        return out


