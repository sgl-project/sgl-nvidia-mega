# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# Adapted from SGLang (Apache-2.0, SGLang Team):
# python/sglang/srt/models/deepseek_v2.py (DeepseekV2Model.forward and DeepseekV2DecoderLayer.forward: the residual steps between the sublayers).
"""The stock all-reduce + residual + RMSNorm steps between the sublayers of a verify
step, as SGLang's layer boundaries run them at verify sizes."""

from __future__ import annotations

import torch

from sgl_nvidia_mega.glm5_3.attention.step import StepContext

# The SGLang names this module reads (test/unit/test_contract.py pins them)
SGLANG_READS = {
    "srt/layers/layernorm.py": (
        "forward_with_allreduce_fusion",
        "forward_with_allreduce_fusion(use_attn_tp_group)",
    ),
    "srt/layers/moe/utils.py": ("sum_post_experts_output",),
    # the replaced model's modules
    "srt/models/deepseek_v2.py": (
        "input_layernorm",
        "post_attention_layernorm",
        "self_attn",
        "o_proj",
        "norm",
        "model",
    ),
}

# The SGLang function bodies this module's stock steps copy, by the digest of their
# code (test/unit/test_contract.py pins them)
SGLANG_MIRRORS = {
    "srt/models/deepseek_v2.py:DeepseekV2DecoderLayer.forward": "228c19b34f667210",
}


class HopSteps:
    """Stock all-reduce + residual + norm steps between the stages."""

    def pre_attention(self, layer, mlp_out, residual, step: StepContext):
        """The attention input and the updated residual (first layer: ``mlp_out``
        is the embedding and ``residual`` None)."""
        norm = layer.input_layernorm
        if residual is None:
            return norm(mlp_out), mlp_out
        # The MLP output owes its sum over the MoE output group (the TP group).
        return norm.forward_with_allreduce_fusion(
            mlp_out, residual, use_attn_tp_group=False
        )

    def post_attention(self, layer, o_input, residual, step: StepContext):
        """``o_proj`` + attention-TP all-reduce + residual add + norm."""
        partial, _ = layer.self_attn.o_proj(o_input)
        return layer.post_attention_layernorm.forward_with_allreduce_fusion(
            partial, residual, use_attn_tp_group=True
        )

    def final(self, model, mlp_out, residual, step: StepContext) -> torch.Tensor:
        """The last layer's MLP output completed, added and normed."""
        from sglang.srt.layers.moe.utils import sum_post_experts_output

        hidden = sum_post_experts_output(mlp_out)
        return model.model.norm(hidden, residual)[0]
