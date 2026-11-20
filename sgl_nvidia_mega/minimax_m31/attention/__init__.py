# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# Adapted from SGLang (Apache-2.0, SGLang Team):
# python/sglang/srt/models/minimax_m3.py (MiniMaxM3DecoderLayer.forward).
"""The attention sublayer of a covered step the stock executor runs: the fused
q/k/v/index projection, the Gemma q/k norms + RoPE (+ FP8 q), the NVFP4 KV + index-K
store, the MSA-v1 indexer and sparse attention, and the output projection (an
unreduced partial).

Stock: SGLang's ``MiniMaxM3Attention`` module, called as the decoder layer calls it.
(Slot mega replaces the whole layer with MegaAttention at its width: ``runtime.mega``.)
"""

from __future__ import annotations

from typing import Any

# The SGLang names this module reads (test/unit/test_contract.py pins them)
SGLANG_READS = {
    "srt/models/minimax_m3.py": (
        "self_attn",
        "forward(positions)",
        "forward(hidden_states)",
        "forward(forward_batch)",
    ),
}


def step(layer, x, st) -> Any:
    """``layer``'s attention partial [rows, hidden] from its normed input ``x``."""
    return layer.self_attn(positions=st.positions, hidden_states=x, forward_batch=st.fb)
