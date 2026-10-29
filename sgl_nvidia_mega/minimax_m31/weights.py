# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""SGLang's loaded MiniMax-M3.1 parameters as the kernels' operands.

The attention, router and shared-expert operands are SGLang's own tensors, read in
place (MXFP8 weights in checkpoint layout with row-major UE8M0 scales), except the
attention's two scale tables, repacked into the 128x4 tiles MegaAttention reads,
and the router, widened to FP32. The routed experts are copied out of the TensorRT-LLM
layout ``flashinfer_trtllm_routed`` keeps them in: SGLang's shuffle moves every
expert's bytes the same way, a permutation recovered once by running it on position
indices. The copy (about 1 GiB a layer) leaves SGLang's experts to its own path."""

from __future__ import annotations

from functools import cache

import torch

from sgl_nvidia_mega.minimax_m31.sm103.moe import token_parallel as tp_moe

# The SGLang names this module reads (test/unit/test_contract.py pins them)
SGLANG_READS = {
    "srt/models/minimax_m3.py": (
        "self_attn",
        "mlp",
        '"fused_qkv_index_proj"',  # added as a submodule
        "o_proj",
        "q_norm",
        "k_norm",
        "index_q_norm",
        "index_k_norm",
        "input_layernorm",
        "post_attention_layernorm",
        "gate",
        "e_score_correction_bias",
        "shared_experts",
        "gate_up_proj",
        "down_proj",
        "experts",
        "routed_scaling_factor",
        "_qm",
    ),
    "srt/layers/quantization/utils.py": ("prepare_static_weights_for_trtllm_fp4_moe",),
    # the routed experts' NVFP4 scheme, as minimax_m3_nvfp4 sets it on the layer
    "srt/layers/quantization/minimax_m3_nvfp4.py": ("scheme", "use_flashinfer_trtllm"),
    "srt/layers/quantization/compressed_tensors/schemes/"
    "compressed_tensors_w4a4_nvfp4_moe.py": (
        "w13_weight",
        "w13_weight_scale",
        "w2_weight",
        "w2_weight_scale",
        "g1_alphas",
        "g2_alphas",
        "reorder_w1w3_to_w3w1",
    ),
    "srt/layers/quantization/fp8.py": ("use_mxfp8", "mxfp8_dense_backend"),
    "srt/layers/quantization/fp8_utils.py": ("is_flashinfer_trtllm",),
}

H, I, E = tp_moe.H, tp_moe.I, tp_moe.E
SCALE_BLOCK, NVFP4_GROUP = 32, 16
PROJECTION_ROWS, O_PROJ_K = 2560, 2048


def quant_method(module: torch.nn.Module):
    # SGLang's fused [qkv | index] holder keeps its method as ``_qm``
    return getattr(module, "quant_method", None) or getattr(module, "_qm", None)


def mxfp8(
    module: torch.nn.Module, rows: int, k: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """A linear's checkpoint-layout MXFP8 weight and row-major UE8M0 scales, checked."""
    method = quant_method(module)
    backend = getattr(method, "mxfp8_dense_backend", None)
    if backend is None or not getattr(method, "use_mxfp8", False):
        raise ValueError(f"{type(module).__name__} is not an MXFP8 linear")
    if backend.is_flashinfer_trtllm():
        raise ValueError(
            "--fp8-gemm-runner-backend flashinfer_trtllm shuffles MXFP8 weights in "
            "place; the kernels read checkpoint layout"
        )
    weight, scale = module.weight, getattr(module, "weight_scale_inv", None)
    if (
        getattr(module, "bias", None) is not None
        or weight.dtype != torch.float8_e4m3fn
        or scale is None
        or scale.dtype != torch.uint8
        or tuple(weight.shape) != (rows, k)
        or tuple(scale.shape) != (rows, k // SCALE_BLOCK)
        or not weight.is_contiguous()
        or not scale.is_contiguous()
    ):
        got = None if scale is None else (scale.dtype, tuple(scale.shape))
        raise ValueError(
            f"{type(module).__name__}: weight {weight.dtype} {tuple(weight.shape)}, "
            f"scales {got}: not a contiguous e4m3 [{rows}, {k}] with uint8 "
            f"[{rows}, {k // SCALE_BLOCK}] and no bias"
        )
    return weight.data, scale.data


def tile_scales(scale: torch.Tensor) -> torch.Tensor:
    """Row-major [R, C] UE8M0 scales as MegaAttention reads them: 128x4 tiles,
    [R / 128][C / 4][32][4][4]."""
    rows, cols = scale.shape
    return (
        scale.reshape(rows // 128, 4, 32, cols // 4, 4)
        .transpose(1, 3)
        .contiguous()
        .reshape(-1)
    )


def attention_operands(layer: torch.nn.Module) -> dict:
    """``AttentionWeights`` fields of one sparse decoder layer."""
    attn = layer.self_attn
    projection, projection_sf = mxfp8(attn.fused_qkv_index_proj, PROJECTION_ROWS, H)
    output, output_sf = mxfp8(attn.o_proj, H, O_PROJ_K)
    return dict(
        projection=projection,
        projection_sf=tile_scales(projection_sf),
        output_projection=output,
        output_projection_sf=tile_scales(output_sf),
        # raw Gemma weights: the kernel applies 1 + w
        input_norm=layer.input_layernorm.weight.data,
        q_norm=attn.q_norm.weight.data,
        k_norm=attn.k_norm.weight.data,
        qi_norm=attn.index_q_norm.weight.data,
        ki_norm=attn.index_k_norm.weight.data,
        post_attention_norm=layer.post_attention_layernorm.weight.data,
    )


def moe_operands(layer: torch.nn.Module, routed_scaling_factor: float) -> dict:
    """``TokenParallelMoe`` operands of one MoE decoder layer: SGLang's router and
    shared expert, and this package's copy of the routed experts."""
    mlp = layer.mlp
    gate_up, gate_up_sf = mxfp8(mlp.shared_experts.gate_up_proj, 2 * I, H)
    down, down_sf = mxfp8(mlp.shared_experts.down_proj, H, I)
    bias = mlp.e_score_correction_bias
    experts = mlp.experts
    device = gate_up.device
    return dict(
        # SGLang's router weights (BF16 unless SGLANG_OPT_USE_BF16_ROUTER_GEMM=0),
        # widened exactly
        router_weight=mlp.gate.weight.data.to(torch.float32).contiguous(),
        router_correction_bias=(
            torch.zeros(E, dtype=torch.float32, device=device)
            if bias is None
            else bias.data
        ),
        shared_fc1_weight=gate_up,
        shared_fc1_weight_sf=gate_up_sf,
        shared_fc2_weight=down,
        shared_fc2_weight_sf=down_sf,
        **routed_experts(experts),
        # SGLang's dequant scales: 1 / weight global scale (the input global scales
        # are 1), the routing multiplier folded into FC2
        routed_fc1_alpha=experts.g1_alphas.data.to(torch.float32).clone(),
        routed_fc2_alpha=experts.g2_alphas.data.to(torch.float32)
        * routed_scaling_factor,
    )


# SGLang's routed expert tensor, the kernel's operand, its rows and K bytes an expert
ROUTED = (
    ("w13_weight", "routed_fc1_weight", 2 * I, H // 2),
    ("w13_weight_scale", "routed_fc1_weight_sf", 2 * I, H // NVFP4_GROUP),
    ("w2_weight", "routed_fc2_weight", H, I // 2),
    ("w2_weight_scale", "routed_fc2_weight_sf", H, I // NVFP4_GROUP),
)


@cache
def trtllm_sources(device_index: int) -> tuple[torch.Tensor, ...]:
    """For each expert tensor of ``ROUTED``, the kernel-layout byte that every byte of
    SGLang's shuffled tensor holds (one expert; every expert shuffles alike).

    SGLang stores FC1 as [up | gate] rows (``reorder_w1w3_to_w3w1``), then permutes the
    rows of every tensor and interleaves the scales
    (``prepare_static_weights_for_trtllm_fp4_moe``): byte moves only. Each byte of a
    pre-shuffle position index runs through that shuffle as an expert's bytes."""
    from sglang.srt.layers.quantization.utils import (
        prepare_static_weights_for_trtllm_fp4_moe,
    )

    device = torch.device("cuda", device_index)
    indices = [
        torch.arange(rows * k, dtype=torch.int32, device=device).view(1, rows, k)
        for _, _, rows, k in ROUTED
    ]
    if max(t.numel() for t in indices) >= 1 << 24:
        raise AssertionError("expert tensors outgrow three index bytes")
    moved = []
    for byte in range(3):
        planes = [((t >> (8 * byte)) & 0xFF).to(torch.uint8) for t in indices]
        w13, w13_sf, w2, w2_sf = prepare_static_weights_for_trtllm_fp4_moe(
            planes[0], planes[2], planes[1], planes[3], H, I, 1
        )
        moved.append(
            [t.view(torch.uint8).reshape(-1).int() for t in (w13, w13_sf, w2, w2_sf)]
        )
    sources = []
    for t, (_, _, rows, k) in enumerate(ROUTED):
        source = (moved[0][t] | (moved[1][t] << 8) | (moved[2][t] << 16)).long()
        if t < 2:  # [up | gate] rows -> the kernel's [gate | up]
            row = source // k
            source = (row + rows // 2) % rows * k + source % k
        if not torch.equal(
            torch.sort(source).values, torch.arange(rows * k, device=device)
        ):
            raise RuntimeError(
                f"SGLang's TensorRT-LLM expert shuffle of {ROUTED[t][0]} is not a "
                "byte permutation"
            )
        sources.append(source)
    return tuple(sources)


def routed_experts(experts: torch.nn.Module) -> dict:
    """The layer's routed experts in the kernel's layout: per expert, FC1 gate rows
    then up rows and FC2, this rank's 768 of every expert's intermediate, row-major
    weights and scales."""
    sources = trtllm_sources(experts.w13_weight.device.index)
    out = {}
    for (name, operand, rows, k), source in zip(ROUTED, sources):
        shuffled = getattr(experts, name).data
        kernel = torch.empty(
            (E, rows, k),
            dtype=torch.uint8 if name.endswith("weight") else torch.float8_e4m3fn,
            device=shuffled.device,
        )
        kernel.view(torch.uint8).view(E, -1).index_copy_(
            1, source, shuffled.view(torch.uint8).reshape(E, -1)
        )
        out[operand] = kernel
    return out


def layer_reasons(layer: torch.nn.Module, index: int) -> list[str]:
    """Why one MoE decoder layer's operands do not fit the kernels."""
    attn, mlp = layer.self_attn, layer.mlp
    what = f"layer {index}"
    reasons = []
    if not getattr(attn, "is_sparse_attention_layer", False):
        return [f"{what} is not an MSA sparse attention layer"]
    if getattr(attn, "fused_qkv_index_proj", None) is None:
        return [f"{what}: SGLang did not fuse the qkv and index projections"]
    if attn.index_k_norm is None or not attn.disable_index_value:
        reasons.append(f"{what}: the kernel reads a K-only index with its own norm")
    if attn.idx_head_dim != attn.head_dim or not attn.rotary_emb.is_neox_style:
        reasons.append(f"{what}: the kernel applies one NeoX RoPE to K and index K")
    for module, rows, k in (
        (attn.fused_qkv_index_proj, PROJECTION_ROWS, H),
        (attn.o_proj, H, O_PROJ_K),
    ):
        try:
            mxfp8(module, rows, k)
        except ValueError as error:
            reasons.append(f"{what}: {error}")
    shared = getattr(mlp, "shared_experts", None)
    if shared is None or getattr(mlp, "num_fused_shared_experts", 0):
        reasons.append(f"{what}: the shared expert is not a separate MLP")
    else:
        for module, rows, k in (
            (shared.gate_up_proj, 2 * I, H),
            (shared.down_proj, H, I),
        ):
            try:
                mxfp8(module, rows, k)
            except ValueError as error:
                reasons.append(f"{what} shared expert: {error}")
    experts = mlp.experts
    scheme = getattr(experts, "scheme", None)
    if not getattr(scheme, "use_flashinfer_trtllm", False):
        reasons.append(
            f"{what}: the routed experts ({type(scheme).__name__}) are not in the "
            "flashinfer_trtllm_routed layout"
        )
        return reasons
    for name, _, rows, k in ROUTED:
        tensor = getattr(experts, name, None)
        dtype = torch.uint8 if name.endswith("weight") else torch.float8_e4m3fn
        if (
            tensor is None
            or tensor.dtype != dtype
            or tuple(tensor.shape) != (E, rows, k)
            or not tensor.is_contiguous()
        ):
            got = None if tensor is None else (tensor.dtype, tuple(tensor.shape))
            reasons.append(
                f"{what}: experts.{name} {got} is not {dtype} [{E}, {rows}, {k}] "
                "(128 experts, a 768 slice each on TP4)"
            )
    for name in ("g1_alphas", "g2_alphas"):
        alpha = getattr(experts, name, None)
        if alpha is None or tuple(alpha.shape) != (E,):
            reasons.append(f"{what}: experts.{name} is not one scale an expert")
    if mlp.gate.weight.dtype not in (torch.bfloat16, torch.float32):
        reasons.append(f"{what}: router weights are {mlp.gate.weight.dtype}")
    return reasons
