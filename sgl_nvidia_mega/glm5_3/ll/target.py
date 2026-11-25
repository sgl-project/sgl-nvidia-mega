# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""The GPU the ``ll.*`` units run on and the plans they run there (``Target``), and
GB300's: the plans measured on it.

The units never check the GPU's architecture. The model code picks one target per
build from the architecture (``glm5_3/targets.py``) and hands it in the profile
(``Profile.target``); a unit reads its plans from it and declines when the profile has
none. Another architecture's target lives in that architecture's directory.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Mapping, Optional, Tuple

from sgl_nvidia_mega.glm5_3.ll import gemm_plan

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Target:
    """A GPU the ``ll.*`` units serve, with the plans they run on it."""

    name: str
    sm: Tuple[int, int]  # compute capability (named in log lines)
    # SMs the plans are for: ll.gemm engages at this count only; ll.sparse_mla_fd too
    # when ``measured``
    sms: int
    # GPCs the split-K one-wave check assumes (gemm_plan.splitk_fits_one_wave)
    gpcs: int
    # ll.gemm's plan entries on this GPU
    gemm_plan: Tuple[gemm_plan.Entry, ...] = field(repr=False)
    # whether FlashInfer's explicit split-K tactics run here (ll.gemm's ``auto`` source
    # takes FlashInfer's kernel only where they do, else the vendored copy)
    flashinfer_splitk: bool
    # True: the plans were measured on this GPU, and a sparse-MLA width it cannot
    # launch in one wave fails the build. False: they are derived from another GPU's,
    # the sparse-MLA row caps are upper bounds cut by the occupancy queried at build,
    # and a width the device cannot launch in one wave declines to SGLang's kernel
    measured: bool
    # unit -> what of its plan is derived, logged once (rank 0) at its build; empty
    # where ``measured``
    notes: Mapping[str, str] = field(
        default_factory=dict, repr=False, hash=False, compare=False
    )

    def note(self, unit: str) -> None:
        """Log once per unit (rank 0) what of its plan is derived on this GPU."""
        what = self.notes.get(unit)
        if what is None or (self.name, unit) in _noted:
            return
        _noted.add((self.name, unit))
        import torch

        dist = torch.distributed
        if dist.is_available() and dist.is_initialized() and dist.get_rank() != 0:
            return
        logger.info(
            "%s: serving %s (sm_%d%d) on plans derived from GB300's, not measured on "
            "it: %s",
            unit,
            self.name,
            self.sm[0],
            self.sm[1],
            what,
        )


# (target name, unit) pairs that logged their note already
_noted: set = set()


# B300 / GB300 (SM103): every plan of ll/ was measured on it (152 SMs, 8 GPCs: the
# B200 / B300 layout, assumed, not queried).
GB300 = Target(
    name="GB300",
    sm=gemm_plan.MEASURED_DEVICE[0],
    sms=gemm_plan.MEASURED_DEVICE[1],
    gpcs=gemm_plan.GPCS,
    gemm_plan=gemm_plan.PLAN,
    flashinfer_splitk=True,
    measured=True,
)


def reason(profile) -> Optional[str]:
    """None when ``profile`` names a target for the units, else their decline reason."""
    if getattr(profile, "target", None) is not None:
        return None
    major, minor = (int(v) for v in profile.sm)
    return f"sm_{major}{minor}: no plans for the TP4 kernel units on this GPU"
