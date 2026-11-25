# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the SGLang project
"""Lamport mailboxes: BF16 rows published in 16-byte fragments; a reader polls until no
fragment holds the -0.0 sentinel, then re-arms."""

import cutlass
import cutlass.cute as cute
from cutlass import Uint32

from .primitives import (
    load_words,
    published,
    store_relaxed,
    wait_deadline_check,
    wait_deadline_start,
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
def rearm_fence():
    """One gpu-scope acq_rel fence after this thread's re-arms."""
    cute.arch.fence_acq_rel_gpu()


@cute.jit
def rearm_fence_end():
    """The fence of a hop that re-arms early, once per thread at the end of the kernel."""
    cute.arch.fence_acq_rel_gpu()


@cute.jit
def fetch(values, i: cutlass.Constexpr[int], pointer):
    """Load fragment ``i`` into ``values[i]``; returns its dirty flag at bit ``i``."""
    got = load_words(pointer, count=4, volatile=True)
    for k in cutlass.range_constexpr(4):
        values[i, k] = got[k]
    return Uint32(not published(got)) << i


@cute.jit
def poll(
    base,
    stride,
    count: cutlass.Constexpr[int],
    deadline_ns: cutlass.Constexpr[int],
):
    """Lamport-poll ``count`` 16-byte fragments at base + i*stride (elements); all
    loads issue before any check and only dirty fragments re-poll."""
    values = cute.make_rmem_tensor((count, 4), Uint32)
    pending = Uint32(0)
    for i in cutlass.range_constexpr(count):
        pending = pending | fetch(values, i, base + i * stride)
    started, polls = cutlass.Int64(0), cutlass.Int32(0)
    if pending != 0:
        started = wait_deadline_start(deadline_ns)
    while pending != 0:
        polls = wait_deadline_check(started, polls, deadline_ns)
        again = Uint32(0)
        for i in cutlass.range_constexpr(count):
            if (pending >> i) & 1:
                again = again | fetch(values, i, base + i * stride)
        pending = again
    return values
