# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 SGLang Team
"""Runtime package: SLOTS mechanism, entry point, re-exports of executor/compile."""

from __future__ import annotations

import os
from typing import List

# Kernel slot names shipped in this build.
SHIPPED_SLOTS = (
    "attention",
    "moe",
    "hops",
    "qkv_a_small_m_shared",
    "dnorm",
    "draft",
)

DRAFT_SLOTS = frozenset({"draft"})

SLOTS_ENV = "SGLANG_NVIDIA_MEGAKERNEL_SLOTS"

# (slot, prerequisite): a slot is only active if its prerequisite is also active.
SLOT_NEEDS = (
    ("qkv_a_small_m_shared", "attention"),
    ("dnorm", "qkv_a_small_m_shared"),
    ("dnorm", "hops"),
)

# SGLang symbols this package reads (validated by test/unit/test_contract.py)
SGLANG_READS = {
    "kernels/plugin_kernels/nvidia/spec.py|srt/megakernels/spec.py": (
        "MegakernelSpec",
        "MegakernelBinding",
        "ForwardView",
        "KvPoolView",
    ),
    "srt/runtime_context.py": (
        "get_parallel",
        "get_forward",
    ),
}


def _requested_slots() -> frozenset:
    raw = os.environ.get(SLOTS_ENV)
    if raw is None:
        return frozenset(SHIPPED_SLOTS)
    return frozenset(name.strip() for name in raw.split(",")) - {""}


SLOTS = _requested_slots()


def slot_reasons() -> List[str]:
    """Reasons a slot was not activated (for startup logs / spec.reasons())."""
    reasons = []
    active = set(SLOTS)
    for slot, prereq in SLOT_NEEDS:
        if slot in active and prereq not in active:
            active.discard(slot)
            reasons.append(
                f"slot {slot!r} needs {prereq!r} which is not active"
            )
    if SLOTS != frozenset(SHIPPED_SLOTS):
        requested = os.environ.get(SLOTS_ENV, "")
        skipped = frozenset(SHIPPED_SLOTS) - SLOTS
        if skipped:
            reasons.append(
                f"{SLOTS_ENV}={requested!r}: skipped slots: "
                + ", ".join(sorted(skipped))
            )
    return reasons
