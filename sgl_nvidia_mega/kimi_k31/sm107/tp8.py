# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the SGLang project
"""Rubin's plans for Kimi K3.1 at TP8: the ``targets.Target`` that
``kimi_k31/targets.py`` picks on SM107.

Rubin (VR200, SM107, 212 SMs) is in the SM100 family and runs the kernels of
``kimi_k31/sm103``, CuTe DSL compiled for the device. Their plans were measured on
GB300; this module holds the ones Rubin replaces. The o-proj producer's staged K
blocks have SM107 values in ``sm103/k3/oproj/prepared.py``; every other plan is
GB300's, not tuned for 212 SMs.
"""

from __future__ import annotations

from sgl_nvidia_mega.kimi_k31.targets import Target

# TP8: 96 KDA heads over 8 ranks
KDA_HEADS = 12

# KDA verify schedules (early, lanes per state row, minimum CTAs per SM, CTAs per head)
# by (requests, queries) at 5 rows a request (Q5), from a sweep of the KDA core alone on
# one GPU. 1 to 3 requests keep the package's (0, 8, 2, 8). From 4 requests that
# schedule faults at Q5 on SM107: CuTe DSL attaches a preferred shared-memory carve-out
# to a kernel launched with min_blocks_per_mp > 1, and the kernel faults under the one
# it sets for this shape. 4 CTAs per head avoid the shape. At 8 requests 2 CTAs per SM
# take a second wave and 3 do not.
KDA_VERIFY = {
    (4, 5): (0, 8, 2, 4),
    (5, 5): (0, 8, 2, 4),
    (6, 5): (0, 8, 2, 4),
    (7, 5): (0, 8, 2, 4),
    (8, 5): (0, 8, 3, 4),
}

# Rubin serves TP8 only: TP16 has no plans here.
RUBIN = Target(name="Rubin (VR200)", kda_verify={KDA_HEADS: KDA_VERIFY}, tp_sizes=(8,))
