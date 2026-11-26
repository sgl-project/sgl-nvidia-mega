# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Slot claim of the persistent MoE kernels (``csrc/slot_claim.cuh``): host tallies.

The kernels take their schedule slot by an exact atomic claim; the SM a CTA runs on only picks the preferred slot. They
also publish an FC1 tile only once each of the four FC1 epilogue warps counted it stored.
"""

from __future__ import annotations

BITMAP_WORDS = 8  # 256 slots
WORDS = (
    BITMAP_WORDS + 3
)  # bitmap, claims of the current launch, hinted-slot misses, verified launches


def tally(ctl, first_word: int) -> dict:
    """Host view (tests; stream-ordered, no launch in flight) of a claim area that starts at int32 word ``first_word``."""
    w = [
        int(x) & 0xFFFFFFFF
        for x in ctl.view(-1)[first_word : first_word + WORDS].tolist()
    ]
    return {
        "bitmap_clear": all(x == 0 for x in w[:BITMAP_WORDS]),
        "pending_claims": w[BITMAP_WORDS],
        "hint_misses": w[BITMAP_WORDS + 1],
        "verified_launches": w[BITMAP_WORDS + 2],
    }
