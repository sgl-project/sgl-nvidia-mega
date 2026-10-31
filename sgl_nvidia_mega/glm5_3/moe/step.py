# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# Adapted from SGLang (Apache-2.0, SGLang Team):
# python/sglang/srt/models/deepseek_v2.py (DeepseekV2DecoderLayer.forward: the MLP / MoE call and its output buffer).
"""The stock dense MLP / MoE of a verify step's layer."""

from __future__ import annotations

from contextlib import nullcontext
from typing import Optional

import torch

from sgl_nvidia_mega.glm5_3.attention.step import StepContext

# The SGLang names this module reads (test/unit/test_contract.py pins them)
SGLANG_READS = {
    "srt/layers/moe/moe_runner/base.py": ("moe_output_buffer_ctx", "inplace"),
    "srt/layers/moe/fused_moe_triton/layer.py": ("moe_runner_config",),
    "srt/models/deepseek_v2.py": ("DeepseekV2MoE", "mlp", "experts"),
    "srt/runtime_context.py": ("get_forward", "scoped", '"defer_moe_finalize"'),
}

# The SGLang function bodies this module's stock steps copy, by the digest of their
# code (test/unit/test_contract.py pins them)
SGLANG_MIRRORS = {
    "srt/models/deepseek_v2.py:DeepseekV2DecoderLayer.forward": "228c19b34f667210",
}


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
