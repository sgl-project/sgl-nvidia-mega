# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""The GPUs the TP4 kernel units serve, and where their plans come from.

Every ``ll.*`` plan (GEMM ranges, row caps, hop caps, the shared expert's stream gate)
was measured on GB300 (SM103, 152 SMs). Rubin (VR200, SM107, 212 SMs) is in the same
SM100 family: the units' kernels are the same sources, JIT-compiled for the device
(``ll.jit``: SGLang targets ``sm_107a``, or ``sm_100f`` when its CUTLASS lacks 10.7)
or CuTe DSL compiled for it. On Rubin a unit runs GB300's plan, re-derived where it
depends on the SM count (occupancy queried at build, one-wave checks against 212 SMs),
unless ``RUBIN`` lists a measured entry for it; each unit says once in the build log
which of its plans are derived. Any other GPU declines every ``ll.*`` unit.

Coverage depends on the profile's compute capability only (every rank decides alike);
a unit whose plan needs an exact SM count checks ``Device.sms`` itself.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional, Sequence, Tuple

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Device:
    """A GPU the ``ll.*`` units serve."""

    name: str
    sm: Tuple[int, int]  # compute capability
    sms: int  # SMs of the part the plans are for
    # True: the plans were measured on this GPU; False: derived from GB300's
    measured: bool
    # GPCs assumed by the split-K one-wave checks (a cluster of c CTAs leaves up to
    # c - 1 SMs idle per GPC); not queried
    gpcs: int


# B300 / GB300: every plan's measurements (8 GPCs, the B200 / B300 layout).
GB300 = Device("GB300", (10, 3), 152, measured=True, gpcs=8)
# VR200 (two dies): 212 SMs. The GPC count is not published; 16 (8 per die) is the
# conservative assumption for the one-wave checks.
RUBIN = Device("Rubin (VR200)", (10, 7), 212, measured=False, gpcs=16)
DEVICES: Tuple[Device, ...] = (GB300, RUBIN)


def lookup(sm: Sequence[int]) -> Optional[Device]:
    """The served device of compute capability ``sm``, or None."""
    sm = tuple(int(v) for v in sm)
    for d in DEVICES:
        if d.sm == sm:
            return d
    return None


def served_names() -> str:
    """The served devices, for decline reasons."""
    return ", ".join(
        f"sm_{d.sm[0]}{d.sm[1]} ({d.name}, {'measured' if d.measured else 'derived'})"
        for d in DEVICES
    )


def reason(sm: Sequence[int]) -> Optional[str]:
    """None when the ``ll.*`` units serve compute capability ``sm``, else the decline
    reason (the units' ``covers``)."""
    if lookup(sm) is not None:
        return None
    major, minor = (int(v) for v in sm)
    return f"sm_{major}{minor}: the TP4 kernel units serve {served_names()}"


def derived(sm: Sequence[int]) -> bool:
    """Whether the units serve ``sm`` on plans derived from GB300's."""
    d = lookup(sm)
    return d is not None and not d.measured


# units that logged note_derived already
_noted: set = set()


def note_derived(unit: str, sm: Sequence[int], what: str = "") -> None:
    """Log once per unit (rank 0) that it serves a device whose plans are derived from
    GB300's, not measured there."""
    if not derived(sm) or unit in _noted:
        return
    _noted.add(unit)
    import torch

    dist = torch.distributed
    if dist.is_available() and dist.is_initialized() and dist.get_rank() != 0:
        return
    d = lookup(sm)
    logger.info(
        "%s: serving %s (sm_%d%d) on plans derived from GB300's, not measured on it%s",
        unit,
        d.name,
        d.sm[0],
        d.sm[1],
        f": {what}" if what else "",
    )
