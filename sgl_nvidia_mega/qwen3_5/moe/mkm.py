# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Slot ``mkm``, on slot ``route``'s outputs: the top-10, FC1, SwiGLU, the NVFP4
requantization and FC2 of up to 7 rows in one launch (``sm103/moe/mkm.py``), in place of
the trtllm runner's routing and its two GEMMs. Its rows and expert weights reach
SGLang's deferred finalize as the runner's would, in expert order (an identity
permutation).
"""

from __future__ import annotations

import torch

# The SGLang names this module reads (test/unit/test_contract.py pins them)
SGLANG_READS = {
    "srt/models/qwen2_moe.py": ("topk", "experts", "layer_id"),
    "srt/layers/moe/topk.py": (
        "topk_config",
        "top_k",
        "renormalize",
        "scoring_func",
        "correction_bias",
        "use_grouped_topk",
    ),
    "srt/layers/moe/fused_moe_triton/layer.py": (
        "num_experts",
        "num_local_experts",
        "moe_ep_rank",
        "top_k",
        "routing_method_type",
        "moe_runner_config",
        "dispatcher",
    ),
    "srt/layers/moe/moe_runner/base.py": (
        "activation",
        "is_gated",
        "routed_scaling_factor",
        "gemm1_alpha",
        "gemm1_clamp_limit",
        "swiglu_limit",
    ),
    "srt/layers/moe/utils.py": ("RoutingMethodType", "RenormalizeNaive"),
    "srt/layers/quantization/modelopt_quant.py": (
        '"w13_weight"',
        '"w2_weight"',
        '"w13_weight_scale"',
        '"w2_weight_scale"',
        '"g1_alphas"',
        '"g2_alphas"',
    ),
    "srt/layers/moe/moe_runner/flashinfer_trtllm.py": (
        '"g1_scale_c"',
        "FlashInferTrtllmDeferredFinalizeOutput",
        "gemm2_out",
        "expert_weights",
        "expanded_idx_to_permuted_idx",
        "top_k",
    ),
    "srt/layers/moe/token_dispatcher/standard.py": (
        "StandardCombineInput",
        "combine(combine_input)",
    ),
}


def shared_topk_slots(config) -> int:
    """The trailing top-k columns SGLang's router fills with fused shared experts:
    ``TopKConfig.num_shared_topk_slots``, named ``num_fused_shared_experts`` before
    SGLang split its two meanings; read by name so both serve."""
    slots = getattr(config, "num_shared_topk_slots", None)
    return config.num_fused_shared_experts if slots is None else slots


def unsupported(mlp) -> str:
    """Why the fused experts cannot stand in for ``mlp``'s routed experts ("" when
    they can): the TP8 or TP4 shapes on the trtllm-gen weight layout, no expert
    parallelism, softmax top-10 renormalized, plain SwiGLU."""
    from sglang.srt.layers.moe.utils import RoutingMethodType

    from sgl_nvidia_mega.qwen3_5.sm103.moe import mkm

    experts, cfg = mlp.experts, mlp.topk.topk_config
    e, h, i = mkm.EXPERTS, mkm.HIDDEN, experts.w2_weight.shape[-1] * 2
    if i not in mkm.INTERMEDIATES:
        return f"intermediate {i} per rank (engines: {list(mkm.INTERMEDIATES)})"
    for name, t, dtype, shape in (
        ("w13_weight", experts.w13_weight, torch.uint8, (e, 2 * i, h // 2)),
        ("w2_weight", experts.w2_weight, torch.uint8, (e, h, i // 2)),
        (
            "w13_weight_scale",
            experts.w13_weight_scale,
            torch.float8_e4m3fn,
            (e, 2 * i, h // 16),
        ),
        (
            "w2_weight_scale",
            experts.w2_weight_scale,
            torch.float8_e4m3fn,
            (e, h, i // 16),
        ),
        ("g1_alphas", experts.g1_alphas, torch.float32, (e,)),
        ("g1_scale_c", experts.g1_scale_c, torch.float32, (e,)),
        ("g2_alphas", experts.g2_alphas, torch.float32, (e,)),
    ):
        if t.dtype != dtype or tuple(t.shape) != shape:
            return f"{name} {t.dtype} {tuple(t.shape)}"
    if not (
        experts.num_experts == experts.num_local_experts == e
        and experts.moe_ep_rank == 0
        and experts.top_k == cfg.top_k == mkm.TOP_K
        and cfg.renormalize
        and cfg.scoring_func == "softmax"
        and cfg.correction_bias is None
        and not cfg.use_grouped_topk
        and shared_topk_slots(cfg) == 0
        and experts.routing_method_type == RoutingMethodType.RenormalizeNaive
    ):
        return (
            "routing other than a softmax top-10, renormalized, over 512 local experts"
        )
    run = experts.moe_runner_config
    if not (
        run.activation == "silu"
        and run.is_gated
        and run.routed_scaling_factor in (None, 1.0)
        and run.gemm1_alpha is None
        and run.gemm1_clamp_limit is None
        and run.swiglu_limit is None
    ):
        return "an expert activation other than a plain SwiGLU"
    return ""


def routed(mlp, maps: tuple, identity: torch.Tensor, logits, x_fp4, x_sf):
    """The deferred-finalize output of ``mlp``'s routed experts from the router front's
    logits and quantized input (``maps``: the layer's weight TMA maps; ``identity``:
    0, 1, ... int32)."""
    from sglang.srt.layers.moe.moe_runner.flashinfer_trtllm import (
        FlashInferTrtllmDeferredFinalizeOutput,
    )
    from sglang.srt.layers.moe.token_dispatcher.standard import StandardCombineInput

    from sgl_nvidia_mega.qwen3_5.sm103.moe import mkm

    experts = mlp.experts
    gemm2_out, expert_weights = mkm.mkm_moe(
        maps,
        logits,
        x_fp4,
        x_sf,
        experts.w13_weight_scale.view(torch.uint8),
        experts.w2_weight_scale.view(torch.uint8),
        experts.g1_alphas,
        experts.g1_scale_c,
        experts.g2_alphas,
    )
    deferred = FlashInferTrtllmDeferredFinalizeOutput(
        gemm2_out=gemm2_out,
        expert_weights=expert_weights,
        expanded_idx_to_permuted_idx=identity[: logits.shape[0] * mkm.TOP_K],
        top_k=mkm.TOP_K,
    )
    return experts.dispatcher.combine(
        combine_input=StandardCombineInput(hidden_states=deferred)
    )
