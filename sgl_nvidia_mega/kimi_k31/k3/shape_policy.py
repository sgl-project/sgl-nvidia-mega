# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the SGLang project
"""Hardware admission for the K3.1 copy of the Kimi K3 megakernel; no runtime dependencies."""

import os

SLOTS_ENV = "SGLANG_NVIDIA_MEGAKERNEL_SLOTS"


def resident_blocks() -> int:
    """GB300's 152 SMs; with "sm148" listed in SGLANG_NVIDIA_MEGAKERNEL_SLOTS, the 148 of a
    B200/GB200 (SM100) or a B300, the same opt-in as kimi_k3 (#65) and deepseek_v41 (#35)."""
    slots = {
        name.strip()
        for name in os.environ.get(SLOTS_ENV, "").split(",")
        if name.strip()
    }
    return 148 if "sm148" in slots else 152
