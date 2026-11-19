# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the SGLang project
"""Lamport mailboxes: BF16 rows published in 16-byte fragments; a reader polls until no
fragment holds the -0.0 sentinel, then re-arms."""

import cutlass
import cutlass.cute as cute
from cutlass import Uint32

from sgl_nvidia_mega.kimi_k31.k3.default_scope import best_default

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
# a mailbox microbenchmark measures a small cost per hop. On in every mode but 0, which removes it (A/B only).
# K31MK_REARM_FENCE=2 keeps every fence but, where a hop computes between its re-arms and its first
# remote-visible store, re-arms right after the poll and fences just before that store (the o_proj
# consumer's norm exchange, AttnRes's exchange): the compute hides the re-arm's latency.
# K31MK_REARM_FENCE=3 (the default in the GB300 BEST scope, default_scope.py; 1 elsewhere) moves those two hops' fences to the end of
# their kernels (after every output, before the closing griddepcontrol.wait); modes 2 and 3 also
# fence RouteSitu's logits re-arm after its top16 instead of before it. The re-arm still lands before
# the kernel completes, and the slot's next remote write follows that completion through the
# successors' griddepcontrol.wait chain.
# Slot reuse is not always two layers apart: layers 92 and 0 share o_proj parity 0, and AttnRes's
# tail_box.out is a single slot (A/B).
REARM_FENCE = int(best_default("K31MK_REARM_FENCE", "3", "1"))


def rearm(pointer):
    store_relaxed(pointer, (Uint32(SENTINEL),) * 4)


@cute.jit
def rearm_fence():
    """``REARM_FENCE``: one gpu-scope acq_rel fence after this thread's re-arms."""
    if cutlass.const_expr(REARM_FENCE != 0):
        cute.arch.fence_acq_rel_gpu()


@cute.jit
def rearm_fence_eager():
    """The fence right after the re-arms (``REARM_FENCE`` 1); mode 2 places it later."""
    if cutlass.const_expr(REARM_FENCE == 1):
        cute.arch.fence_acq_rel_gpu()


@cute.jit
def rearm_fence_end():
    """Mode 3's fence, once per thread at the end of the kernel."""
    if cutlass.const_expr(REARM_FENCE == 3):
        cute.arch.fence_acq_rel_gpu()


@cute.jit
def rearm_fence_late():
    """Mode 2's fence, after the early re-arms and before the first remote-visible store."""
    if cutlass.const_expr(REARM_FENCE == 2):
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
