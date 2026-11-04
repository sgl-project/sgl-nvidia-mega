# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# Adapted from SGLang (Apache-2.0, SGLang Team):
# python/sglang/srt/models/deepseek_v2.py (the tail of DeepseekV2ForCausalLM.forward).
"""The logits of a covered verify step: SGLang's logits processor on the final-normed
hidden states, as the tail of ``DeepseekV2ForCausalLM.forward`` runs it (no auxiliary
hidden states: the spec refuses their capture)."""

from __future__ import annotations

# The SGLang names this module reads (test/unit/test_contract.py pins them)
SGLANG_READS = {
    "srt/models/deepseek_v2.py": ("logits_processor", "lm_head"),
}

# The SGLang function bodies this module's stock steps copy, by the digest of their
# code (test/unit/test_contract.py pins them)
SGLANG_MIRRORS = {
    "srt/models/deepseek_v2.py:DeepseekV2ForCausalLM.forward": "bbdd93599bb4ff3d",
}


def logits(lm, input_ids, hidden, forward_batch):
    """SGLang's ``LogitsProcessorOutput`` for the step."""
    return lm.logits_processor(input_ids, hidden, lm.lm_head, forward_batch, None)
