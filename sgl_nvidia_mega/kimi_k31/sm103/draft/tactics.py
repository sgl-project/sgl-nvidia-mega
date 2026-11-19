# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the SGLang project
"""Static split-K tactics of the draft's projections, per device.

(M tile, N tile, split K, stages) per projection and compute capability, from a draft
tactic sweep (cold-L2 graph timing, projection gate) on GB300 (10, 3) and VR200 (10, 7).
The table is draft-scoped (its own
keys; the target's projection tactics and compile caches are not touched) and static (no
runtime autotune). Unknown
devices and projections fail closed to ``DEFAULT`` (the dense-MLP gate/up tactic every
draft projection used before the sweep)."""

import os

import torch

DEFAULT = (64, 8, 4, 8)
TACTICS = {
    # GB300, 152 SMs (best vs DEFAULT per call): conv / gate_up / fused_a keep DEFAULT
    # (within noise); kv_stack, fc_slice and head are faster with their own tactics.
    (10, 3): {
        "kv_stack": (128, 8, 4, 6),
        "fc_slice": (64, 8, 8, 11),
        "head": (128, 8, 1, 6),
    },
    # VR200, 212 SMs: conv / gate_up / fused_a / kv_stack keep DEFAULT; fc_slice and head
    # are faster with their own tactics.
    (10, 7): {
        "fc_slice": (64, 8, 8, 11),
        "head": (128, 8, 2, 6),
    },
}
# Experiment overrides: K31DRAFT_PROJ_TACTIC (conv / gate_up / fused_a) and
# K31DRAFT_HEAD_TACTIC (head), "m,n,split,stages".
OVERRIDES = {
    "conv": "K31DRAFT_PROJ_TACTIC",
    "gate_up": "K31DRAFT_PROJ_TACTIC",
    "fused_a": "K31DRAFT_PROJ_TACTIC",
    "head": "K31DRAFT_HEAD_TACTIC",
}


def draft_tactic(name, device):
    """The tactic of draft projection ``name`` on ``device``."""
    override = os.environ.get(OVERRIDES.get(name, ""), "")
    if override:
        return tuple(map(int, override.split(",")))
    capability = torch.cuda.get_device_capability(device)
    return TACTICS.get(capability, {}).get(name, DEFAULT)
