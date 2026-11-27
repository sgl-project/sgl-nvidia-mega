# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the SGLang project
"""The plans the Kimi K3.1 kernels (``kimi_k31/sm103``) run with, picked from the GPU's
architecture: the package's own (``GB300``, what ``sm103/`` holds) on SM103 and SM100,
Rubin's (``sm107/tp8.py``) on SM107, none elsewhere (the build stops).

``select`` is the only place that maps an architecture to a target. The spec refuses
a TP size the target does not serve (``tp_reasons``); ``build_resources``
(``k3/providers.py``) selects once per build, carries the result
(``ResourceSet.target``) and hands the kernels' hosts plain values read from it.

Not every per-architecture value is here: ``sm103/`` still looks the architecture up
for its SM103 KDA verify table (``k3/kda/verify.py``) and for the o-proj producer's
staged K blocks, which have SM107 values (``k3/oproj/prepared.py``).

This module imports without a GPU, SGLang or CuTe DSL.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Mapping, Optional, Sequence, Tuple

SM100, SM103, SM107 = (10, 0), (10, 3), (10, 7)

# (early, lanes per state row, minimum CTAs per SM, CTAs per head), the schedule of
# ``sm103/k3/kda/kernel.py`` ``KdaSplit``
KdaSchedule = Tuple[int, int, int, int]

_NONE: Mapping = MappingProxyType({})


@dataclass(frozen=True)
class Target:
    """What one architecture changes in the plans ``sm103/`` holds. An empty field
    changes nothing, so the kernels run exactly as ``sm103/`` plans them."""

    name: str
    # KDA verify schedules by heads per rank, then by (requests, queries). A key that is
    # absent keeps ``verify_schedule`` (``sm103/k3/kda/verify.py``).
    kda_verify: Mapping[int, Mapping[Tuple[int, int], KdaSchedule]] = field(
        default_factory=dict, hash=False
    )
    # The TP sizes served on this architecture; empty: every size the spec admits.
    tp_sizes: Tuple[int, ...] = ()

    def __post_init__(self):
        frozen = {
            int(heads): MappingProxyType(dict(table))
            for heads, table in self.kda_verify.items()
        }
        object.__setattr__(self, "kda_verify", MappingProxyType(frozen))

    def kda_verify_schedules(self, heads: int) -> Mapping[Tuple[int, int], KdaSchedule]:
        """The KDA verify schedules of a rank with ``heads`` heads."""
        return self.kda_verify.get(int(heads), _NONE)


# SM103 (GB300, B300) and SM100 (B200, GB200): the plans as ``sm103/`` holds them.
GB300 = Target(name="GB300")


def select(sm: Sequence[int]) -> Optional[Target]:
    """The target on compute capability ``sm``, or None."""
    sm = tuple(int(v) for v in sm)
    if sm in (SM100, SM103):
        return GB300
    if sm == SM107:
        from sgl_nvidia_mega.kimi_k31.sm107 import tp8

        return tp8.RUBIN
    return None


def tp_reasons(sm: Sequence[int], tp: int) -> list[str]:
    """Why the target on compute capability ``sm`` does not serve TP ``tp``; empty when
    it does, or when ``sm`` has no target (the spec's architectures refuse that)."""
    target = select(sm)
    if target is None or not target.tp_sizes or int(tp) in target.tp_sizes:
        return []
    return [
        f"Kimi K3.1 megakernel on {target.name} serves TP {list(target.tp_sizes)}, "
        f"not TP{int(tp)}"
    ]
