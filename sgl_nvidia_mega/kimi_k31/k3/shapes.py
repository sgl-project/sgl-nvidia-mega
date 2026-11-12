# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Per-rank Kimi K3 / K3.1 attention shapes by TP size (TP8: 12 heads, TP16: 6 heads).

TP16 is admitted only with ``K31MK_TP16=1`` (default off: the shipped TP8 build is unchanged). Every
width below equals the TP8 constant it replaces at TP8 (checked by test/unit/test_kimi_k31_tp16_shapes.py).
"""

from __future__ import annotations

import os

TOTAL_HEADS = 96  # MLA and KDA heads of the full model
HEAD_DIM = 128
QKV_A = 2112  # replicated q_a (1536) + kv_a (576) rows of the packed MLA projection
INDEX_K = 128  # K3.1 folded index-K rows (indexer.wk)
ALIGN = 16  # packed projection rows are padded to whole 16-row tiles


def enabled_tp_sizes() -> frozenset[int]:
    return frozenset({8, 16}) if os.environ.get("K31MK_TP16", "0") == "1" else frozenset({8})


def heads(tp: int) -> int:
    if tp not in (8, 16):
        raise ValueError(f"Kimi K3 megakernel shapes exist for TP8 and TP16, not TP{tp}")
    return TOTAL_HEADS // tp


def kda_dim(tp: int) -> int:
    return heads(tp) * HEAD_DIM


def kda_rows(tp: int) -> tuple[int, int, int, int]:
    """(fused qkvg, f_a, beta, padding) rows of the packed KDA projection: TP8 (6144, 128, 12, 4) = 6288,
    TP16 (3072, 128, 6, 10) = 3216."""
    raw = 4 * kda_dim(tp) + 128 + heads(tp)
    return 4 * kda_dim(tp), 128, heads(tp), -raw % ALIGN


def kda_pitch(tp: int) -> int:
    return sum(kda_rows(tp))


def mla_gate(tp: int) -> int:
    """The per-rank output-gate width (g_proj rows), also the MLA attention output width."""
    return heads(tp) * HEAD_DIM


def mla_pitch(tp: int) -> int:
    """K3 packed MLA rows: q_a + kv_a, then the gate (TP8 3648)."""
    return QKV_A + mla_gate(tp)


def k31_index_col(tp: int) -> int:
    """K3.1: first folded index-K row of the packed projection (TP8 3648, TP16 2880)."""
    return QKV_A + mla_gate(tp)


def k31_pitch(tp: int) -> int:
    """K3.1 packed MLA/index-K rows (TP8 3776, TP16 3008)."""
    return k31_index_col(tp) + INDEX_K


def tp_of_heads(local_heads: int) -> int:
    tp = TOTAL_HEADS // local_heads
    if tp * local_heads != TOTAL_HEADS or tp not in (8, 16):
        raise ValueError(f"{local_heads} local heads is neither TP8 nor TP16")
    return tp
