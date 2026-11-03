# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Slot claim of the persistent MoE kernels (``csrc/slot_claim.cuh``): build flag, install record and host tallies.

The kernels take their schedule slot by an exact atomic claim; the SM a CTA runs on only picks the preferred slot. They
also publish an FC1 tile only once each of the four FC1 epilogue warps counted it stored. The previous forms (slot taken
directly from the SM id, which the PTX ISA does not guarantee to stay fixed; one CTA-wide stored count, which assumes the
epilogue warps finish tiles in step) are built only with ``MUSE_VR_MOELD_SLOT_UNSAFE=1`` (default off; A/B captures).
Both forms pass an explicit define, so their builds never share a build key with each other or with older builds.
"""

from __future__ import annotations

import os

UNSAFE_ENV = "MUSE_VR_MOELD_SLOT_UNSAFE"
BITMAP_WORDS = 8  # 256 slots
WORDS = (
    BITMAP_WORDS + 3
)  # bitmap, claims of the current launch, hinted-slot misses, verified launches

_recorded: set[str] = set()


def unsafe() -> bool:
    return os.environ.get(UNSAFE_ENV, "0").strip() == "1"


def build_flags(kernel: str) -> tuple[str, ...]:
    """Compiler define for ``kernel``'s slot claim; records the form once per process and kernel."""
    form = "unsafe" if unsafe() else "fixed"
    if kernel not in _recorded:
        _recorded.add(kernel)
        from . import bed

        bed.report(
            f"slot_claim_{kernel}",
            True,
            form=form,
            fc1_publish="cta_count" if form == "unsafe" else "per_warp",
            env=f"{UNSAFE_ENV}={'1' if form == 'unsafe' else '0'}",
        )
    return (f"-DMUSE_VR_SLOT_UNSAFE={'1' if form == 'unsafe' else '0'}",)


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
