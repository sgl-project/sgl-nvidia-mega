# SPDX-License-Identifier: Apache-2.0
"""Native FP8 tensor references, validated once during kernel binding."""

from __future__ import annotations

from dataclasses import dataclass, fields
from typing import TYPE_CHECKING

from sgl_nvidia_mega.glm_5 import BLOCK, EXPERTS, HIDDEN, KV_LORA, Q_LORA
from sgl_nvidia_mega.glm_5.plan import Variant

if TYPE_CHECKING:
    import torch


def _check(tensor, shape, dtype, name, device, *, row_view=False):
    if tuple(tensor.shape) != shape:
        raise ValueError(f"{name}: expected shape {shape}, got {tuple(tensor.shape)}")
    if tensor.dtype != dtype or tensor.device != device:
        raise ValueError(f"{name}: expected {dtype} on {device}")
    if row_view:
        if tensor.stride(-1) != 1 or tensor.stride(-2) < shape[-1]:
            raise ValueError(
                f"{name}: expected contiguous columns and a valid row stride"
            )
    elif not tensor.is_contiguous():
        raise ValueError(f"{name} must be contiguous")


def _check_scale(tensor, rows, cols, name, device, *, row_view=False):
    import torch

    if tensor.dtype == torch.int32:
        shape = (rows, (cols // BLOCK + 3) // 4)
        if (
            tuple(tensor.shape) != shape
            or tensor.device != device
            or tensor.stride(0) != 1
            or tensor.stride(1) < rows
        ):
            raise ValueError(f"{name}: invalid native packed UE8M0 scale layout")
    else:
        _check(
            tensor,
            ((rows + BLOCK - 1) // BLOCK, cols // BLOCK),
            torch.float32,
            name,
            device,
            row_view=row_view,
        )


@dataclass(frozen=True)
class MlaWeights:
    x_gamma: torch.Tensor
    q_a_weight: torch.Tensor
    q_a_scale_inv: torch.Tensor
    kv_a_with_mqa_weight: torch.Tensor
    kv_a_with_mqa_scale_inv: torch.Tensor
    q_gamma: torch.Tensor
    q_b_weight: torch.Tensor
    q_b_scale_inv: torch.Tensor
    kv_gamma: torch.Tensor
    kv_b_weight: torch.Tensor
    kv_b_scale_inv: torch.Tensor
    o_proj_weight: torch.Tensor
    o_proj_scale_inv: torch.Tensor

    def kernel_args(self) -> tuple:
        return tuple(getattr(self, field.name) for field in fields(self))

    def validate(self, variant: Variant, device, *, norm_dtype="float32") -> None:
        import torch

        if variant.op != "mla":
            raise ValueError("MLA weights require an MLA variant")
        heads = 64 // variant.tp_size
        shapes = (
            (HIDDEN,),
            (Q_LORA, HIDDEN),
            (16, 48),
            (576, HIDDEN),
            (5, 48),
            (Q_LORA,),
            (heads * 256, Q_LORA),
            (heads * 2, 16),
            (KV_LORA,),
            (heads * 448, KV_LORA),
            (heads * 7 // 2, 4),
            (HIDDEN, heads * 256),
            (48, heads * 2),
        )
        fp8_fields = {1, 3, 6, 9, 11}
        for i, (field, tensor, shape) in enumerate(
            zip(fields(self), self.kernel_args(), shapes)
        ):
            if i in (2, 4, 7, 10, 12):
                matrix = self.kernel_args()[i - 1]
                _check_scale(
                    tensor, *matrix.shape, field.name, device, row_view=i == 12
                )
            else:
                dtype = (
                    torch.float8_e4m3fn
                    if i in fp8_fields
                    else getattr(torch, norm_dtype)
                )
                _check(tensor, shape, dtype, field.name, device, row_view=i == 11)


@dataclass(frozen=True)
class MoeWeights:
    rms_gamma: torch.Tensor
    router_weight: torch.Tensor
    router_bias: torch.Tensor
    up_gate_weight: torch.Tensor
    up_gate_scale: torch.Tensor
    down_weight: torch.Tensor
    down_scale: torch.Tensor

    def kernel_args(self) -> tuple:
        return tuple(getattr(self, field.name) for field in fields(self))

    def validate(self, variant: Variant, device, *, norm_dtype="float32") -> None:
        import torch

        if variant.op != "moe":
            raise ValueError("MoE weights require a MoE variant")
        inter = 2048 // variant.tp_size
        checks = (
            ("rms_gamma", (HIDDEN,), getattr(torch, norm_dtype)),
            ("router_weight", (256, HIDDEN), torch.bfloat16),
            ("router_bias", (256,), torch.float32),
            ("up_gate_weight", (EXPERTS, 2 * inter, HIDDEN), torch.float8_e4m3fn),
            ("up_gate_scale", (EXPERTS, 2 * inter // BLOCK, 48), torch.float32),
            ("down_weight", (EXPERTS, HIDDEN, inter), torch.float8_e4m3fn),
            ("down_scale", (EXPERTS, 48, inter // BLOCK), torch.float32),
        )
        for name, shape, dtype in checks:
            _check(getattr(self, name), shape, dtype, name, device)


def mla_weights_from_layer(layer) -> MlaWeights:
    """Views of the current SGLang fused projection; no repack or dtype cast."""
    attn = layer.self_attn
    fused = attn.fused_qkv_a_proj_with_mqa
    scales = fused.weight_scale_inv
    # FP32 scales have one row per block. Packed UE8M0 retains one row per
    # output channel in SGLang's native transposed int32 view.
    import torch

    split = Q_LORA if scales.dtype == torch.int32 else Q_LORA // BLOCK
    return MlaWeights(
        layer.input_layernorm.weight,
        fused.weight[:Q_LORA],
        scales[:split],
        fused.weight[Q_LORA:],
        scales[split:],
        attn.q_a_layernorm.weight,
        attn.q_b_proj.weight,
        attn.q_b_proj.weight_scale_inv,
        attn.kv_a_layernorm.weight,
        attn.kv_b_proj.weight,
        attn.kv_b_proj.weight_scale_inv,
        attn.o_proj.weight,
        attn.o_proj.weight_scale_inv,
    )


def moe_weights_from_layer(layer) -> MoeWeights:
    """The framework's fused routed+shared expert storage, kept by reference."""
    mlp = layer.mlp
    experts = mlp.experts
    return MoeWeights(
        layer.post_attention_layernorm.weight,
        mlp.gate.weight,
        mlp.gate.e_score_correction_bias,
        experts.w13_weight,
        experts.w13_weight_scale_inv,
        experts.w2_weight,
        experts.w2_weight_scale_inv,
    )


def norm_dtype(weights) -> str:
    import torch

    gains = (
        (weights.x_gamma, weights.q_gamma, weights.kv_gamma)
        if isinstance(weights, MlaWeights)
        else (weights.rms_gamma,)
    )
    dtype = gains[0].dtype
    if dtype not in (torch.float32, torch.bfloat16) or any(
        gain.dtype != dtype for gain in gains
    ):
        raise ValueError("GLM5 norm gains must share float32 or bfloat16 dtype")
    return "bfloat16" if dtype == torch.bfloat16 else "float32"
