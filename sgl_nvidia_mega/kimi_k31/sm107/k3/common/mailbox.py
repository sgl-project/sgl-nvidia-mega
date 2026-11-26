# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the SGLang project
"""Lamport mailboxes: BF16 rows published in 16-byte fragments; a reader polls until no
fragment holds the -0.0 sentinel, then re-arms."""

import cutlass.cute as cute
from cutlass import Uint32

from .primitives import (
    store_relaxed,
)

SENTINEL = 0x80000000


# Re-arm ordering (Rubin port, 2026-10-29): on VR200 under interleaved HBM
# streaming a reader's relaxed re-arm of a slot was overtaken by the next remote publish into that slot, which then
# read as armed (deadline trap). One fence.acq_rel.gpu per thread after its re-arms, before its later stores,
# launch_dependents and exit, orders the re-arm first. Not a strict-PTX fix (gpu scope, multicast alias);
# a mailbox microbenchmark measures a small cost per hop.
# Where a hop computes between its re-arms and its first remote-visible store (AttnRes's exchange, RouteSitu's
# logits after its top16), the fence is at the end of the kernel (``rearm_fence_end``: after every output, before
# the closing griddepcontrol.wait), so the compute hides the re-arm's latency. The re-arm still lands before the
# kernel completes, and the slot's next remote write follows that completion through the successors'
# griddepcontrol.wait chain; this holds for every batch and TP. Measured against the fence right after the
# re-arms: Rubin TP8 c1 +1.24 %, GB300 TP16 c1 mean TPOT -0.39 %.
# Slot reuse is not always two layers apart: layers 92 and 0 share o_proj parity 0, and AttnRes's
# tail_box.out is a single slot.


def rearm(pointer):
    store_relaxed(pointer, (Uint32(SENTINEL),) * 4)


@cute.jit
def rearm_fence_end():
    """The fence of a hop that re-arms early, once per thread at the end of the kernel."""
    cute.arch.fence_acq_rel_gpu()
