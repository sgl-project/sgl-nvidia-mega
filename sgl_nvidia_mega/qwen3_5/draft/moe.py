# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# Adapted from SGLang (Apache-2.0): python/sglang/srt/layers/moe/single_token_draft_moe.py
# (SingleTokenDraftMoe: its eligibility checks and single-launch forward), on branch
# nvidia/qwen35-agentx-gb300.
"""The MTP draft layer's MoE as one launch per token (``sm103/draft/moe.py``): router
GEMV + softmax top-k, the routed experts and the gated shared expert, from the bf16
weights SGLang loaded for the draft's ``Qwen2MoeSparseMoeBlock``.

``DraftMoe(x)`` returns this rank's unreduced output (routed and gated shared summed),
whose TP sum the draft boundary completes. Against SGLang's stock draft MoE only the
fp32 dot-product order differs.
"""

from __future__ import annotations

import logging
from typing import Optional

import torch

from sgl_nvidia_mega.qwen3_5.moe.mkm import shared_topk_slots
from sgl_nvidia_mega.qwen3_5.sm103.draft import moe as kernels

logger = logging.getLogger(__name__)

# The SGLang names this module reads (test/unit/test_contract.py pins them)
SGLANG_READS = {
    "srt/models/qwen2_moe.py": (
        "gate",
        "experts",
        "shared_expert",
        "shared_expert_gate",
        "topk",
        "gate_up_proj",
        "down_proj",
        "layer_id",
    ),
    "srt/layers/moe/fused_moe_triton/layer.py": ("quant_method",),
    "srt/layers/quantization/unquant.py": (
        "UnquantizedFusedMoEMethod",
        "use_flashinfer_trtllm_moe",
        "w13_swiglu_interleaved",
        '"w13_weight"',
        '"w2_weight"',
    ),
    "srt/layers/moe/topk.py": (
        "topk_config",
        "top_k",
        "enable_waterfill",
        "scoring_func",
        "renormalize",
        "use_grouped_topk",
        "correction_bias",
        "custom_routing_function",
        "routed_scaling_factor",
    ),
    "srt/runtime_context.py": ("get_parallel", '"moe_ep_size"'),
}


def _router_reason(mlp) -> Optional[str]:
    cfg = mlp.topk.topk_config
    if (
        cfg.scoring_func != "softmax"
        or not cfg.renormalize
        or cfg.use_grouped_topk
        or cfg.correction_bias is not None
        or cfg.custom_routing_function is not None
        or shared_topk_slots(cfg) != 0
        or cfg.routed_scaling_factor not in (None, 1.0)
        or mlp.topk.enable_waterfill
    ):
        return "router is not plain softmax top-k with renormalization"
    w = mlp.gate.weight
    if w.dtype != torch.bfloat16 or w.shape[0] % 32 != 0 or not w.is_contiguous():
        return f"router weight {w.dtype} {tuple(w.shape)}"
    return None


def unsupported_reason(mlp) -> Optional[str]:
    """Why the single launch cannot serve this block (None when it can): the checks of
    SingleTokenDraftMoe._unsupported_reason and _router_unsupported_reason."""
    from sglang.srt.layers.quantization.unquant import UnquantizedFusedMoEMethod
    from sglang.srt.runtime_context import get_parallel

    experts, shared = mlp.experts, mlp.shared_expert
    if shared is None or mlp.shared_expert_gate is None:
        return "no separate shared expert"
    # The kernel reads W13 as [gate rows | up rows] in checkpoint order.
    method = experts.quant_method
    if not isinstance(method, UnquantizedFusedMoEMethod):
        return f"expert weights handled by {type(method).__name__}"
    if method.use_flashinfer_trtllm_moe or method.w13_swiglu_interleaved:
        return "expert weights in a kernel-specific layout"
    if get_parallel().moe_ep_size != 1:
        return "expert parallelism"
    w13, w2 = experts.w13_weight, experts.w2_weight
    ws13, ws2 = shared.gate_up_proj.weight, shared.down_proj.weight
    hidden, inter = w13.shape[-1], w2.shape[-1]
    tensors = (w13, w2, ws13, ws2, mlp.shared_expert_gate.weight)
    if any(t.dtype != torch.bfloat16 for t in tensors):
        return "non-bf16 operands"
    if w13.dim() != 3 or tuple(w13.shape[1:]) != (2 * inter, hidden):
        return f"w13 shape {tuple(w13.shape)}"
    if tuple(ws13.shape) != (2 * inter, hidden) or tuple(ws2.shape) != (hidden, inter):
        return f"shared expert shapes {tuple(ws13.shape)} {tuple(ws2.shape)}"
    if hidden % 1024 != 0 or inter % 32 != 0:
        return f"hidden {hidden} / intermediate {inter} not tileable"
    if not all(t.is_contiguous() for t in tensors):
        return "non-contiguous operands"
    reason = _router_reason(mlp)
    if reason is not None:
        return reason
    # the single launch's own tiling (static_asserts of src/draft_moe.cu)
    num_experts, topk = mlp.gate.weight.shape[0], mlp.topk.topk_config.top_k
    per_lane = num_experts // 32
    if inter % 64 != 0 or per_lane & (per_lane - 1) or not 1 <= topk <= kernels.STAGES:
        return f"{num_experts} experts / top-{topk} / intermediate {inter} not tileable"
    return None


class DraftMoe:
    """The single-launch draft MoE of one block: its scratch and the per-token launch."""

    MAX_TOKENS = 2

    def __init__(self, mlp, device: torch.device):
        self.mlp = mlp
        num_experts = mlp.gate.weight.shape[0]
        topk = mlp.topk.topk_config.top_k
        inter = mlp.experts.w2_weight.shape[-1]
        self.scratch = kernels.draft_moe_fused_scratch(num_experts, topk, inter, device)
        self.topk_weights = torch.empty(1, topk, dtype=torch.float32, device=device)
        self.topk_ids = torch.empty(1, topk, dtype=torch.int32, device=device)

    @classmethod
    def build(cls, mlp, device_index: int) -> Optional[DraftMoe]:
        """The block's single-launch MoE, compiled and checked to fit this device, or
        None (logged) when it cannot serve the block."""
        reason = unsupported_reason(mlp)
        if reason is None:
            w13 = mlp.experts.w13_weight
            reason = "; ".join(
                kernels.build(
                    device_index,
                    hidden=w13.shape[-1],
                    inter=mlp.experts.w2_weight.shape[-1],
                    experts=mlp.gate.weight.shape[0],
                    topk=mlp.topk.topk_config.top_k,
                )
            )
        if reason:
            logger.warning(
                "Single-launch draft MoE off for layer %d (%s); SGLang's MoE runs.",
                mlp.layer_id,
                reason,
            )
            return None
        logger.info("Single-launch draft MoE active for layer %d.", mlp.layer_id)
        return cls(mlp, torch.device("cuda", device_index))

    def ready(self, num_tokens: int) -> bool:
        """Whether it runs a batch of this many tokens."""
        return 0 < num_tokens <= self.MAX_TOKENS

    def single_launch(self, num_tokens: int) -> bool:
        """Whether that batch runs as single launches (always, once ready)."""
        return self.ready(num_tokens)

    def __call__(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """[m, H] -> this rank's MoE output [m, H] (m <= MAX_TOKENS), unreduced; the
        tokens run one launch each, in order."""
        mlp = self.mlp
        out = torch.empty_like(hidden_states)
        for m in range(hidden_states.shape[0]):
            kernels.draft_moe_fused(
                hidden_states[m : m + 1],
                mlp.gate.weight,
                mlp.experts.w13_weight,
                mlp.experts.w2_weight,
                mlp.shared_expert.gate_up_proj.weight,
                mlp.shared_expert.down_proj.weight,
                mlp.shared_expert_gate.weight.view(-1),
                self.scratch,
                self.topk_weights,
                self.topk_ids,
                out[m : m + 1],
            )
        return out
