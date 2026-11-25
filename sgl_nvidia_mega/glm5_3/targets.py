# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""The plans the TP4 kernel units (``glm5_3/ll``) run with, picked from the GPU's
architecture: GB300's (``ll/target.py``) on SM103, Rubin's (``sm107/tp4.py``) on SM107,
none elsewhere (every ``ll.*`` unit then declines).

``select`` is the only place that maps an architecture to a target. A profile calls it
once when it is resolved (``Profile.from_args``) and carries the result
(``Profile.target``); the units read their plans from the profile.
"""

from __future__ import annotations

from typing import Optional, Sequence

from sgl_nvidia_mega.glm5_3.ll.target import GB300, Target

SM103, SM107 = (10, 3), (10, 7)


def select(sm: Sequence[int]) -> Optional[Target]:
    """The units' target on compute capability ``sm``, or None."""
    sm = tuple(int(v) for v in sm)
    if sm == SM103:
        return GB300
    if sm == SM107:
        from sgl_nvidia_mega.glm5_3.sm107 import tp4

        return tp4.RUBIN
    return None
