# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# Adapted from SGLang (Apache-2.0): Qwen3_5AttentionDecoderLayer.forward,
# Qwen3_5AttentionDecoderLayer.self_attention, Qwen2MoeMLP.forward,
# GemmaRMSNorm._forward_impl and ColumnParallelLinear.forward.

from __future__ import annotations

import contextlib
import functools
import logging
from typing import Optional

import torch

logger = logging.getLogger(__name__)

# The prefill rows this module runs, in the order the build checks them
ROWS: tuple[str, ...] = (
    "prefill_gate_fp8",
    "prefill_act_fp8",
    "gdn_prefill_norm_fp8",
    "prefill_norm_fp8",
)

SGLANG_READS = {
    "srt/layers/quantization/modelopt_quant.py": (
        "ModelOptFp8LinearMethod",
        "use_marlin",
        "use_sm120_fp8",
        "input_scale",
    ),
    # the linear's prequantized-input path: an E4M3 input is not quantized again
    "srt/layers/quantization/fp8_utils.py": (
        "apply_fp8_linear(pre_quant_output_dtype)",
        "apply_fp8_linear_bmm_flashinfer(pre_quant_output_dtype)",
        "input_prequantized",
    ),
    "srt/models/qwen3_5.py": (
        "Qwen3_5AttentionDecoderLayer",
        "_prepare_qkv_gate",
        "attn_output_gate",
        "o_proj",
        "attn",
        "attn_boundary",
        "ffn_boundary",
        "mlp",
        "fused_sigmoid_mul",
        "linear_attn",
        "norm",
        "out_proj",
        "input_layernorm",
        "in_proj_qkvz",
        "qkv_proj",
    ),
    "srt/models/qwen2_moe.py": (
        "Qwen2MoeSparseMoeBlock",
        "Qwen2MoeMLP",
        "shared_expert",
        "gate_up_proj",
        "down_proj",
        "act_fn",
        "can_run_silu_fp4_quant_fusion",
        "silu_and_mul_fp4_quant",
    ),
    "srt/layers/activation.py": ("SiluAndMul", "forward_cuda"),
    "kernels/fused_op.py": ("_forward_method",),
    "srt/layers/linear.py": (
        "input_size_per_partition",
        "ColumnParallelLinear",
        "forward",
    ),
    "srt/layers/layernorm.py": (
        "GemmaRMSNorm",
        "forward_cuda",
        "variance_epsilon",
        "weight",
    ),
    "srt/layers/layer_boundary/stage.py": ("prepare", "finish", "exit"),
    "srt/layers/layer_boundary/exit.py": ("defer_moe_finalize", "finish"),
    "srt/model_executor/forward_mode.py|srt/model_executor/forward_batch_info.py": (
        "is_idle",
    ),
}

SGLANG_MIRRORS = {
    "srt/layers/layernorm.py:GemmaRMSNorm._forward_impl": "8e9b3a51e75af28c",
    "srt/layers/layer_boundary/residual/add_norm.py:NormQuantReadout.update_and_read": (
        "a606319fdf5a2091"
    ),
    "srt/layers/layer_boundary/residual/add_norm.py:_norm_quant_kernel": (
        "211e37c00fbe7bb2"
    ),
    "srt/layers/layer_boundary/residual/add_norm.py:_update_and_read_residual_plain": (
        "4d97834b2f09f7c1"
    ),
    "srt/models/qwen3_5.py:Qwen3_5GatedDeltaNet._forward_input_proj": "5269f43d28e6f303",
    "srt/models/qwen3_5.py:Qwen3_5AttentionDecoderLayer.forward_prepare_cuda_fused": (
        "d10e23f5651990d0"
    ),
    "srt/layers/layernorm.py:GemmaRMSNorm.forward_cuda": "ef46f7b180a41cfd",
    "srt/layers/linear.py:ColumnParallelLinear.forward": "7538aefbf1cabf51",
    "srt/models/qwen3_5.py:Qwen3_5AttentionDecoderLayer.forward": "6473f673a1a0796b",
    "srt/models/qwen3_5.py:Qwen3_5AttentionDecoderLayer.self_attention": "db0f4d98e0fb1abf",
    "srt/models/qwen2_moe.py:Qwen2MoeMLP.forward": "60b6d1e897d94ed9",
    "srt/layers/activation.py:SiluAndMul.forward_cuda": "6e7a35a12d39720a",
}


def static_fp8_scale(linear) -> Optional[torch.Tensor]:
    from sglang.srt.layers.quantization.modelopt_quant import ModelOptFp8LinearMethod

    method = getattr(linear, "quant_method", None)
    if not isinstance(method, ModelOptFp8LinearMethod) or hasattr(linear, "base_layer"):
        return None
    if getattr(method, "use_marlin", False) or getattr(method, "use_sm120_fp8", False):
        return None
    scale = getattr(linear, "input_scale", None)
    if (
        not isinstance(scale, torch.Tensor)
        or scale.numel() != 1
        or scale.dtype != torch.float32
        or not scale.is_cuda
    ):
        return None
    return scale


class PrefillFp8:
    """The static FP8 hand-offs the build found each layer able to take."""

    def __init__(self):
        self.gate: dict[int, torch.Tensor] = {}
        self.act: dict = {}
        self.active = False
        self.norm: dict = {}
        self.norm_in: dict = {}
        self._stash = None
        self._engaged: set = set()

    @classmethod
    def build(cls, model, rows, linear) -> PrefillFp8:
        from sglang.srt.models.qwen3_5 import Qwen3_5AttentionDecoderLayer

        built = cls()
        layers = model.model.layers
        stage = range(model.model.start_layer, model.model.end_layer)
        if "prefill_gate_fp8" in rows:
            why = ""
            for i in stage:
                layer = layers[i]
                if i in linear or not isinstance(layer, Qwen3_5AttentionDecoderLayer):
                    continue
                scale = static_fp8_scale(layer.o_proj)
                if not layer.attn_output_gate:
                    why = why or f"layer {i}: no output gate"
                elif scale is None:
                    why = (
                        why
                        or f"layer {i}: o_proj is not a static per-tensor FP8 linear"
                    )
                else:
                    built.gate[i] = scale
            _log_build("prefill_gate_fp8", built.gate, why)
        if "prefill_act_fp8" in rows:
            built._build_act([(i, layers[i]) for i in stage])
        if "gdn_prefill_norm_fp8" in rows:
            from sgl_nvidia_mega.qwen3_5.gdn import norm_fp8

            why = ""
            for i in sorted(linear):
                gdn = layers[i].linear_attn
                scale = static_fp8_scale(gdn.out_proj)
                reason = norm_fp8.unsupported(gdn.norm) or (
                    "out_proj is not a static per-tensor FP8 linear"
                    if scale is None
                    else ""
                )
                if reason:
                    why = why or f"layer {i}: {reason}"
                else:
                    built.norm[gdn] = scale
            _log_build("gdn_prefill_norm_fp8", built.norm, why)
        if "prefill_norm_fp8" in rows:
            built._build_norm_in([(i, layers[i], i in linear) for i in stage])
        return built

    def _build_act(self, layers):
        from sglang.srt.models.qwen2_moe import Qwen2MoeMLP, Qwen2MoeSparseMoeBlock

        why = ""
        for i, layer in layers:
            mlp = getattr(layer, "mlp", None)
            if isinstance(mlp, Qwen2MoeSparseMoeBlock):
                mlp = getattr(mlp, "shared_expert", None)
            if not isinstance(mlp, Qwen2MoeMLP):
                continue
            reason = _act_unsupported(mlp)
            if reason:
                why = why or f"layer {i}: {reason}"
            else:
                self.act[mlp] = static_fp8_scale(mlp.down_proj)
        _log_build("prefill_act_fp8", self.act, why)
        if self.act:
            from sgl_nvidia_mega.qwen3_5.sm103.moe import prefill_act_fp8

            prefill_act_fp8.build(torch.cuda.current_device())
        for mlp in self.act:
            # SGLang's MoE block calls the shared expert: give it this forward
            mlp.forward = functools.partial(self.mlp_forward, mlp)

    def _build_norm_in(self, layers):
        why = ""
        for i, layer, gdn in layers:
            norm = getattr(layer, "input_layernorm", None)
            if gdn:
                linear = getattr(layer.linear_attn, "in_proj_qkvz", None)
            else:
                linear = getattr(layer, "qkv_proj", None)
            reason = _norm_in_unsupported(norm, linear)
            if reason:
                why = why or f"layer {i}: {reason}"
            else:
                self.norm_in[norm] = linear
        if self.norm_in:
            # the CuTe DSL kernel, imported only where a layer takes it
            from sgl_nvidia_mega.qwen3_5.sm103 import gemma_norm_fp8

            if not gemma_norm_fp8.cute_route(torch.cuda.current_device()):
                why, self.norm_in = "FlashInfer runs its CUDA norm kernel here", {}
            for hidden in sorted({norm.weight.shape[0] for norm in self.norm_in}):
                gemma_norm_fp8.build(torch.cuda.current_device(), hidden)
        _log_build("prefill_norm_fp8", self.norm_in, why)
        for norm, linear in self.norm_in.items():
            # SGLang's boundary calls the norm and the layer the linear
            norm.forward = functools.partial(self.norm_forward, norm)
            linear.forward = functools.partial(self.linear_forward, linear)

    @contextlib.contextmanager
    def body(self):
        """The prefill body's forward: the MLPs' fused path runs inside it only."""
        self.active = True
        try:
            yield
        finally:
            self.active, self._stash = False, None

    def rows(self) -> list:
        return [
            row
            for row, taken in zip(ROWS, (self.gate, self.act, self.norm, self.norm_in))
            if taken
        ]

    def _engage(self, row, rows):
        if row not in self._engaged:
            self._engaged.add(row)
            logger.info("NVIDIA megakernel: Qwen3.5 slot %s runs at %d rows", row, rows)

    def attention_layer(self, i, layer, positions, hidden, fb, capture):
        from sglang.srt.models.qwen2_moe import Qwen2MoeSparseMoeBlock

        hidden = layer.attn_boundary.prepare(hidden, fb, capture_gathered=capture)
        # fused AR+quant hands down a (fp8, scale) / (bf16, fp8, scale) tuple
        hs = hidden[0] if isinstance(hidden, tuple) else hidden
        if not fb.forward_mode.is_idle() and hs.shape[0] > 0:
            hidden = self.self_attention(i, layer, positions, hidden, fb)
        hidden = layer.attn_boundary.finish(hidden, fb)
        hidden = layer.ffn_boundary.prepare(hidden, fb)
        with layer.ffn_boundary.exit(fb) as ffn_exit:
            if isinstance(layer.mlp, Qwen2MoeSparseMoeBlock):
                hidden = layer.mlp(
                    hidden, fb, defer_finalize=ffn_exit.defer_moe_finalize
                )
            else:
                hidden = layer.mlp(hidden)
        return ffn_exit.finish(hidden)

    def self_attention(self, i, layer, positions, hidden, fb):
        q, k, v, gate = layer._prepare_qkv_gate(
            positions=positions, hidden_states=hidden, forward_batch=fb
        )
        attn_output = layer.attn(q, k, v, fb)
        output, _ = layer.o_proj(self.gate_input(i, attn_output, gate))
        return output

    def gate_input(self, i, attn_output, gate):
        from sgl_nvidia_mega.qwen3_5.sm103.attention import gate_quant

        if gate_quant.supported(attn_output, gate):
            self._engage("prefill_gate_fp8", attn_output.shape[0])
            return gate_quant.fused_sigmoid_mul_fp8(attn_output, gate, self.gate[i])
        from sglang.srt.models.qwen3_5 import fused_sigmoid_mul

        return fused_sigmoid_mul(attn_output, gate, inplace=True)

    def mlp_forward(self, mlp, x):
        from sglang.srt.models.qwen2_moe import (
            can_run_silu_fp4_quant_fusion,
            silu_and_mul_fp4_quant,
        )

        from sgl_nvidia_mega.qwen3_5.sm103.moe import prefill_act_fp8

        gate_up, _ = mlp.gate_up_proj(x)
        if can_run_silu_fp4_quant_fusion(mlp, gate_up):
            x, _ = mlp.down_proj(silu_and_mul_fp4_quant(mlp, gate_up))
            return x
        if self.active and prefill_act_fp8.supported(gate_up):
            self._engage("prefill_act_fp8", gate_up.shape[0])
            q = prefill_act_fp8.silu_and_mul_static_fp8(gate_up, self.act[mlp])
            x, _ = mlp.down_proj(q)
            return x
        x = mlp.act_fn(gate_up)
        x, _ = mlp.down_proj(x)
        return x

    def gdn_norm(self, gdn, core, z):
        from sgl_nvidia_mega.qwen3_5.gdn import norm_fp8

        scale = self.norm.get(gdn)
        if scale is None or not norm_fp8.supported(core, z):
            return gdn.norm(core, z)
        self._engage("gdn_prefill_norm_fp8", core.shape[0])
        return norm_fp8.prefill_rms_norm_gated_fp8(gdn.norm, core, z, scale)

    def norm_forward(self, norm, *args, **kwargs):
        from sgl_nvidia_mega.qwen3_5.sm103 import gemma_norm_fp8

        self._stash = None
        call = _fused_add_args(args, kwargs) if self.active else None
        if call is not None:
            x, residual = call
            linear = self.norm_in[norm]
            scale = static_fp8_scale(linear)
            if (
                scale is not None
                and gemma_norm_fp8.supported(x, residual)
                # the hand-off checks the version counter, which inference tensors lack
                and not x.is_inference()
            ):
                self._engage("prefill_norm_fp8", x.shape[0])
                q = gemma_norm_fp8.gemma_fused_add_rmsnorm_fp8(
                    x, residual, norm.weight.data, norm.variance_epsilon, scale
                )
                self._stash = (linear, x, q, x._version)
                return x, residual
        return type(norm).forward(norm, *args, **kwargs)

    def norm_scale(self, norm, x) -> Optional[torch.Tensor]:
        self._stash = None
        linear = self.norm_in.get(norm) if self.active else None
        # the hand-off checks the version counter, which inference tensors lack
        if linear is None or x.is_inference():
            return None
        return static_fp8_scale(linear)

    def norm_hand_off(self, norm, x, q):
        self._engage("prefill_norm_fp8", x.shape[0])
        self._stash = (self.norm_in[norm], x, q, x._version)

    def linear_forward(self, linear, x):
        stash, self._stash = self._stash, None
        if stash is not None and stash[0] is linear:
            _, y, q, version = stash
            if (
                x.dtype == y.dtype
                and x.data_ptr() == y.data_ptr()
                and x.shape == y.shape
                and x.stride() == y.stride()
                and x._version == version
            ):
                x = q
        elif stash is not None:
            self._stash = stash
        return type(linear).forward(linear, x)


def _forward_impl_args(x, residual=None, post_residual_addition=None, **other):
    return x, residual, post_residual_addition, other


def _fused_add_args(args, kwargs):
    try:
        x, residual, post_residual_addition, other = _forward_impl_args(*args, **kwargs)
    except TypeError:
        return None
    if (
        other
        or post_residual_addition is not None
        or not isinstance(x, torch.Tensor)
        or not isinstance(residual, torch.Tensor)
    ):
        return None
    return x, residual


def _norm_in_unsupported(norm, linear) -> str:
    from sglang.srt.layers.layernorm import GemmaRMSNorm
    from sglang.srt.layers.linear import ColumnParallelLinear

    method = getattr(norm, "_forward_method", None)
    if (
        type(norm) is not GemmaRMSNorm
        or "forward" in vars(norm)
        or (
            method is not None
            and getattr(method, "__func__", None) is not GemmaRMSNorm.forward_cuda
        )
    ):
        return "the input norm is not SGLang's CUDA GemmaRMSNorm"
    if norm.weight.shape[0] % 8:
        return "the hidden size is not a multiple of 8"
    if (
        not isinstance(linear, ColumnParallelLinear)
        or type(linear).forward is not ColumnParallelLinear.forward
        or "forward" in vars(linear)
    ):
        return "the norm's reader is not a ColumnParallelLinear"
    if static_fp8_scale(linear) is None:
        return "the norm's reader is not a static per-tensor FP8 linear"
    return ""


def _act_unsupported(mlp) -> str:
    from sglang.srt.layers.activation import SiluAndMul

    from sgl_nvidia_mega.qwen3_5.sm103.moe import prefill_act_fp8

    if "forward" in vars(mlp):
        return "the MLP's forward is not SGLang's"
    act = mlp.act_fn
    # BaseFusedOp's resolved forward, where one is pinned (the CUDA forward otherwise)
    method = getattr(act, "_forward_method", None)
    if type(act) is not SiluAndMul or (
        method is not None
        and getattr(method, "__func__", None) is not SiluAndMul.forward_cuda
    ):
        return "the MLP's activation is not SGLang's CUDA SiluAndMul"
    if static_fp8_scale(mlp.down_proj) is None:
        return "down_proj is not a static per-tensor FP8 linear"
    if mlp.down_proj.input_size_per_partition % prefill_act_fp8.VEC_ELEMENTS:
        return "down_proj's input width is not a multiple of the kernel's vector"
    return ""


def _log_build(row, taken, why):
    if why or not taken:
        logger.info(
            "NVIDIA megakernel: Qwen3.5 slot %s uses stock on %s (%s)",
            row,
            "some layers" if taken else "every layer",
            why or "no layer on this stage",
        )
