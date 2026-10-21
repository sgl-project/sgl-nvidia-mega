# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""The logits of a covered step. Stock: SGLang's logits processor on the final hidden
rows, which also hands the speculative worker every row's hidden state."""

from __future__ import annotations

# The SGLang names this module reads (test/unit/test_contract.py pins them)
SGLANG_READS = {"srt/models/qwen3_vl.py": ("logits_processor", "lm_head")}


def logits(st, hidden, aux):
    """SGLang's ``LogitsProcessorOutput`` for the step (``aux``: the captured auxiliary
    hidden states, or None)."""
    lm = st.lm
    return lm.logits_processor(st.input_ids, hidden, lm.lm_head, st.fb, aux)
