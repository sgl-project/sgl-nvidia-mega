# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""MMA-K form of the plugin's NVFP4 MoE kernels (moe_ld, MoE-LD-W, moe_ldtp, KM2): build flag and install record.

Every kernel accumulates each output element over the full K in K order with block-scaled MMAs. By default each MMA
takes one 64-wide K chunk (the form whose routed rows are bitwise equal to FlashInfer 0.7.1's routed NVFP4 MoE in the
unit tests). ``MUSE_VR_MOE_MMA_K=128`` builds the kernels with one MMA per two chunks (K = 128 per instruction), the form
whose rows are bitwise equal to FlashInfer 0.7.2's in the unit tests; ``auto`` picks the form listed for the installed
FlashInfer version and refuses an unlisted version. The default (``64``) adds no compiler define, so default builds keep
their build keys. Every kernel records its form once per process (``mma_k_<kernel>``) with the FlashInfer version and
whether that form is the one listed for it (``stock_order``: match / differs / unverified).
"""

from __future__ import annotations

import os

ENV = "MUSE_VR_MOE_MMA_K"
# the form whose rows the unit tests read bitwise equal to each FlashInfer version's routed NVFP4 MoE
LISTED = {"0.7.1": 64, "0.7.2": 128}
DEFINES = {
    "moe_ld": "MUSE_VR_MOELD_K128",
    "moe_ldtp": "MUSE_VR_MOELD_K128",
    "moe_ldw": "MUSE_VR_LDW_K128",
    "lb_km2": "MUSE_VR_KM2_K128",
}

_recorded: set[str] = set()


def flashinfer_version() -> str:
    try:
        import flashinfer

        return str(getattr(flashinfer, "__version__", "unknown"))
    except Exception:  # noqa: BLE001 - recorded as unverified
        return "unknown"


def form() -> int:
    """64 or 128 per ``MUSE_VR_MOE_MMA_K``; raises for an unknown value, or for ``auto`` with an unlisted FlashInfer."""
    v = os.environ.get(ENV, "64").strip().lower()
    if v in ("64", ""):
        return 64
    if v == "128":
        return 128
    if v == "auto":
        fi = flashinfer_version()
        if fi not in LISTED:
            raise RuntimeError(
                f"{ENV}=auto: FlashInfer {fi} has no listed MMA-K form (listed: {sorted(LISTED)})"
            )
        return LISTED[fi]
    raise RuntimeError(f"{ENV}={v!r}: expected 64, 128 or auto")


def build_flags(kernel: str) -> tuple[str, ...]:
    """Compiler define for ``kernel``'s MMA-K form (none for 64); records the form once per process and kernel."""
    k = form()
    if kernel not in _recorded:
        _recorded.add(kernel)
        from . import bed

        fi = flashinfer_version()
        order = (
            "unverified"
            if fi not in LISTED
            else ("match" if LISTED[fi] == k else "differs")
        )
        bed.report(
            f"mma_k_{kernel}",
            True,
            k=k,
            flashinfer=fi,
            stock_order=order,
            env=f"{ENV}={os.environ.get(ENV, '64')}",
        )
    return (f"-D{DEFINES[kernel]}=1",) if k == 128 else ()
