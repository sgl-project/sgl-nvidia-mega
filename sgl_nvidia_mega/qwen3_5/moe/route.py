# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# Adapted from SGLang (Apache-2.0, SGLang Team):
# python/sglang/srt/layers/moe/fused_moe_triton/layer.py (FusedMoE.forward_deferred_finalize).
"""Slot ``route``: a MoE layer's router GEMM and the NVFP4 quantization of its routed
experts' input in one launch (``sm103/moe/router.py``), then SGLang's trtllm routed
experts on that input, with the deferred finalize.

The logits equal the gate's cuBLAS GEMM bit for bit and the quantized input equals the
trtllm runner's own quantizer, so the step's output is the stock one's. The runner
takes the quantized input through its pre-quantized (``hidden_states_scale``) path,
or slot ``mkm`` (``mkm.py``) takes the router front's outputs in its place.

Slot ``route_shared`` (numeric) also computes the shared expert's static-FP8 gate/up
projection in the router front's launch (SGLANG_Q35MF_ROUTE_SHARED on SGLang branch
nvidia/qwen35-agentx-gb300), so the shared expert on the alternate stream starts at its
activation, after that launch only; the gate/up sums in another order than the FP8
linear's GEMM.
"""

from __future__ import annotations

import logging

import msgspec
import torch

logger = logging.getLogger(__name__)

# The SGLang names this module reads (test/unit/test_contract.py pins them)
SGLANG_READS = {
    "srt/models/qwen2_moe.py": (
        "Qwen2MoeSparseMoeBlock",
        "gate",
        "topk",
        "experts",
        "supports_deferred_finalize",
        "layer_id",
        # the shared expert after its gate/up projection (slot route_shared)
        "Qwen2MoeMLP",
        "shared_expert",
        "shared_expert_gate",
        "gate_up_proj",
        "down_proj",
        "act_fn",
        "_enable_silu_fp4_quant_fusion",
        "_silu_fp4_quant_fused",
    ),
    "srt/models/qwen3_5.py": ("layers", "start_layer", "end_layer", "mlp"),
    "srt/layers/moe/topk.py": ("TopKOutputChecker", "format_is_bypassed"),
    "srt/layers/moe/fused_moe_triton/layer.py": (
        "dispatcher",
        "run_moe_core",
        "run_moe_core(dispatch_output)",
        "quant_method",
    ),
    "srt/layers/moe/token_dispatcher/standard.py": (
        "dispatch(hidden_states)",
        "dispatch(topk_output)",
        "combine(combine_input)",
        "hidden_states_scale",
    ),
    "srt/layers/moe/moe_runner/flashinfer_trtllm.py": (
        "flashinfer_trtllm_deferred_finalize_context",
        "trtllm_padded_hidden_size",
    ),
    "srt/layers/quantization/modelopt_quant.py": (
        '"w13_input_scale_quant"',
        "quant_config",
        "use_per_token_activation",
        "ModelOptFp8LinearMethod",
        "use_marlin",
        "use_sm120_fp8",
        '"input_scale"',
        '"weight_scale"',
    ),
    "srt/layers/activation.py": ("SiluAndMul",),
}

# The SGLang functions whose body this module mirrors, by the digest of their code
# (test/unit/test_contract.py recomputes them on the installed SGLang)
SGLANG_MIRRORS = {
    "srt/layers/moe/fused_moe_triton/layer.py:FusedMoE.forward_deferred_finalize": "af4b474cbaf33493",
    "srt/layers/moe/fused_moe_triton/layer.py:FusedMoE._dispatch_with_pre_quant": "dcdefc28cf643081",
    "srt/models/qwen2_moe.py:Qwen2MoeMLP.forward": "6091ffb613568a10",
}


# the (slot, rows) that have run (each logged once, as engagement evidence)
_ENGAGED: set = set()


class Bound(msgspec.Struct, frozen=True):
    """The MoE package's build result (``st.bound["moe"]``)."""

    route: bool = False  # the router front stands in for every layer's gate
    mkm: dict = msgspec.field(default_factory=dict)  # layer id -> its weight TMA maps
    mkm_wide: bool = False  # slot mkm_wide: mkm's 16-token engine built
    # [160] int32 0, 1, ...: slot mkm's rows are in expert order
    identity: object = None
    shared: bool = False  # slot route_shared: the shared gate/up in the router launch
    reason: str = ""  # why a requested slot is not built


def build(
    model, max_rows: int, fused: bool, shared: bool = False, wide: bool = False
) -> Bound:
    """The router front for every MoE layer, compiled for 1..max_rows rows, with
    ``fused`` slot mkm's kernel and weight maps (with ``wide`` its 16-token engine too,
    for the rows past 7 up to max_rows), and with ``shared`` slot route_shared's launch;
    or why a layer cannot take them."""
    from sgl_nvidia_mega.qwen3_5.moe import mkm
    from sgl_nvidia_mega.qwen3_5.sm103.moe import mkm as mkm_kernel
    from sgl_nvidia_mega.qwen3_5.sm103.moe import router

    m = model.model
    layers = [m.layers[i].mlp for i in range(m.start_layer, m.end_layer)]
    for i, mlp in zip(range(m.start_layer, m.end_layer), layers):
        reason = unsupported(mlp)
        if reason:
            return Bound(reason=f"layer {i}: {reason}")
    if shared:
        for i, mlp in zip(range(m.start_layer, m.end_layer), layers):
            reason = shared_unsupported(mlp)
            if reason:
                return Bound(reason=f"route_shared: layer {i}: {reason}")
    device = torch.cuda.current_device()
    router.build(device, max_rows, shared=shared)
    if not fused:
        return Bound(route=True, shared=shared)
    for i, mlp in zip(range(m.start_layer, m.end_layer), layers):
        reason = mkm.unsupported(mlp)
        if reason:
            return Bound(route=True, reason=f"mkm: layer {i}: {reason}")
    intermediate = layers[0].experts.w2_weight.shape[-1] * 2
    # the 16-token engine serves the rows past 7 where this rank's intermediate has one
    wide = (
        wide
        and max_rows > mkm_kernel.MAX_TOKENS
        and intermediate in mkm_kernel.WIDE_ENGINES
    )
    mkm_kernel.build(device, intermediate, max_rows if wide else mkm_kernel.MAX_TOKENS)
    return Bound(
        route=True,
        mkm={
            mlp.layer_id: mkm_kernel.weight_maps(
                mlp.experts.w13_weight, mlp.experts.w2_weight
            )
            for mlp in layers
        },
        mkm_wide=wide,
        identity=torch.arange(
            mkm_kernel.M16_MAX_TOKENS * mkm_kernel.TOP_K,
            dtype=torch.int32,
            device="cuda",
        ),
        shared=shared,
    )


def unsupported(mlp) -> str:
    """Why the router front cannot stand in for ``mlp``'s gate and the runner's input
    quantizer ("" when it can)."""
    from sglang.srt.models.qwen2_moe import Qwen2MoeSparseMoeBlock

    from sgl_nvidia_mega.qwen3_5.sm103.moe import router

    if not isinstance(mlp, Qwen2MoeSparseMoeBlock):
        return f"{type(mlp).__name__} is not a Qwen2MoeSparseMoeBlock"
    w = mlp.gate.weight
    if w.dtype != torch.bfloat16 or tuple(w.shape) != (router.EXPERTS, router.HIDDEN):
        return f"router gate {w.dtype} {tuple(w.shape)}"
    experts = mlp.experts
    scale = experts.w13_input_scale_quant
    if scale.dtype != torch.float32 or scale.numel() != 1:
        return f"routed input scale {scale.dtype} {tuple(scale.shape)}"
    if experts.trtllm_padded_hidden_size is not None:
        return "the routed experts' hidden dim is padded"
    if experts.quant_method.quant_config.use_per_token_activation:
        return "per-token activation scales"
    if not mlp.supports_deferred_finalize:
        return "no deferred finalize"
    return ""


def shared_unsupported(mlp) -> str:
    """Why the router front cannot compute ``mlp``'s shared gate/up ("" when it can):
    a static per-tensor FP8 projection of at most 512 rows, then SiLU-and-mul."""
    from sglang.srt.layers.activation import SiluAndMul
    from sglang.srt.layers.quantization.modelopt_quant import ModelOptFp8LinearMethod
    from sglang.srt.models.qwen2_moe import Qwen2MoeMLP

    from sgl_nvidia_mega.qwen3_5.sm103.moe import router

    expert = mlp.shared_expert
    if not isinstance(expert, Qwen2MoeMLP) or mlp.shared_expert_gate is None:
        return "no gated Qwen2MoeMLP shared expert"
    gu = expert.gate_up_proj
    method = gu.quant_method
    if (
        not isinstance(method, ModelOptFp8LinearMethod)
        or method.use_marlin
        or method.use_sm120_fp8
        or gu.bias is not None
        or gu.weight.dtype != torch.float8_e4m3fn
        or gu.weight.shape[0] != router.HIDDEN
        or gu.weight.shape[1] % 8
        or gu.weight.shape[1] > router.SHARED_MAX_N
        or not gu.weight.t().is_contiguous()
        or gu.input_scale.numel() != 1
        or gu.weight_scale.numel() != 1
    ):
        return "the shared gate/up is not a static per-tensor FP8 linear of <= 512 rows"
    if not isinstance(expert.act_fn, SiluAndMul):
        return f"shared expert activation {type(expert.act_fn).__name__}"
    return ""


def shared_expert_tail(expert, gate_up: torch.Tensor) -> torch.Tensor:
    """``Qwen2MoeMLP.forward`` after its gate_up projection."""
    if expert._enable_silu_fp4_quant_fusion and not isinstance(gate_up, tuple):
        x, _ = expert.down_proj(expert._silu_fp4_quant_fused(gate_up))
        return x
    x = expert.act_fn(gate_up)
    x, _ = expert.down_proj(x)
    return x


def routed(mlp, x: torch.Tensor, fused: Bound | None = None, shared: bool = False):
    """``FusedMoE.forward_deferred_finalize`` of ``mlp``'s routed experts on ``x``
    [rows, hidden], with the router front's logits and quantized input (``fused``: on
    slot mkm's kernel). With ``shared`` (at most 8 rows) the router launch also computes
    the shared expert's gate/up: returns (routed, gate_up, the event recorded on the
    current stream right after that launch)."""
    from sgl_nvidia_mega.qwen3_5.sm103.moe import router

    experts = mlp.experts
    for slot in ("route" if fused is None else "mkm",) + (
        ("route_shared",) if shared else ()
    ):
        if (slot, x.shape[0]) not in _ENGAGED:
            _ENGAGED.add((slot, x.shape[0]))
            logger.info(
                "NVIDIA megakernel: Qwen3.5 slot %s runs at %d rows", slot, x.shape[0]
            )
    if shared:
        gu = mlp.shared_expert.gate_up_proj
        logits, x_fp4, x_sf, gate_up = router.router_gemm_quant_shared(
            x,
            mlp.gate.weight,
            experts.w13_input_scale_quant.reshape(1),
            gu.weight.t(),
            gu.input_scale.reshape(1),
            gu.weight_scale.reshape(1),
        )
        ready = torch.cuda.Event()
        ready.record()
        out = _routed_experts(mlp, x, fused, logits, x_fp4, x_sf)
        return out, gate_up, ready
    logits, x_fp4, x_sf = router.router_gemm_quant(
        x, mlp.gate.weight, experts.w13_input_scale_quant.reshape(1)
    )
    return _routed_experts(mlp, x, fused, logits, x_fp4, x_sf)


def _routed_experts(mlp, x, fused, logits, x_fp4, x_sf):
    """The routed experts on the router front's outputs (``fused``: slot mkm's kernel),
    deferred."""
    from sglang.srt.layers.moe.moe_runner.flashinfer_trtllm import (
        flashinfer_trtllm_deferred_finalize_context,
    )
    from sglang.srt.layers.moe.topk import TopKOutputChecker

    from sgl_nvidia_mega.qwen3_5.moe import mkm

    experts = mlp.experts
    if fused is not None:
        maps = fused.mkm[mlp.layer_id]
        return mkm.routed(mlp, maps, fused.identity, logits, x_fp4, x_sf)
    topk_output = mlp.topk(x, logits)
    if not TopKOutputChecker.format_is_bypassed(topk_output):
        raise RuntimeError("the router front requires logits-based bypassed TopK")
    dispatch_output = experts.dispatcher.dispatch(
        hidden_states=x, topk_output=topk_output
    )._replace(hidden_states=x_fp4, hidden_states_scale=x_sf.view(torch.float8_e4m3fn))
    with flashinfer_trtllm_deferred_finalize_context():
        combine_input = experts.run_moe_core(dispatch_output=dispatch_output)
    return experts.dispatcher.combine(combine_input=combine_input)
