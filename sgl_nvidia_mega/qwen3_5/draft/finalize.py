# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Plugin-owned eligibility for the Qwen3.5 MTP draft's deferred MoE finalize.

SGLang owns the model-independent deferred-finalize protocol.  Slot ``draft_fin``
only selects that protocol for the one Qwen3.5 draft shape whose producer and
consumer have been validated.  Keeping the selection here avoids a Qwen-specific
policy branch in SGLang's stock MoE layer.
"""

from __future__ import annotations

import torch

# The SGLang names this module reads (test/unit/test_contract.py pins them).
SGLANG_READS = {
    "srt/layers/quantization/unquant.py": ("UnquantizedFusedMoEMethod",),
    "srt/layers/moe/fused_moe_triton/layer.py": (
        "is_draft",
        "quant_method",
        "params_dtype",
        "hidden_size",
        "num_experts",
        "top_k",
        "runner_backend",
        "supports_deferred_finalize",
    ),
    "srt/models/qwen2_moe.py": (
        "experts",
        "shared_expert",
    ),
}


def enable(mlp) -> str:
    """Enable the generic producer on this exact draft block.

    Returns an empty string when the block can serve ``draft_fin`` and a stable
    refusal reason otherwise.  The mutation is local to the loaded draft model's
    routed-expert module; it changes no process-wide or stock-layer policy.
    """
    from sglang.srt.layers.quantization.unquant import UnquantizedFusedMoEMethod

    experts = mlp.experts
    runner = getattr(experts, "runner_backend", None)
    checks = (
        (not getattr(experts, "is_draft", False), "not a draft MoE"),
        (
            not isinstance(experts.quant_method, UnquantizedFusedMoEMethod),
            f"quant method {type(experts.quant_method).__name__}",
        ),
        (
            experts.params_dtype != torch.bfloat16,
            f"params dtype {experts.params_dtype}",
        ),
        (experts.hidden_size != 4096, f"hidden size {experts.hidden_size}"),
        (experts.num_experts != 512, f"experts {experts.num_experts}"),
        (experts.top_k != 10, f"top-k {experts.top_k}"),
        (mlp.shared_expert is None, "no separate shared expert"),
        (
            runner is None or not runner.is_flashinfer_trtllm(),
            f"MoE runner {runner}",
        ),
    )
    for refused, reason in checks:
        if refused:
            return reason
    experts.supports_deferred_finalize = True
    return ""
