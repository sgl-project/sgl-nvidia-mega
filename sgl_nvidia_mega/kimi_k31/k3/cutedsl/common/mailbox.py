# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the SGLang project
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Lamport mailboxes: BF16 rows published in 16-byte fragments; a reader polls until no
fragment holds the -0.0 sentinel, then re-arms."""

import os

import cutlass
import cutlass.cute as cute
from cutlass import Uint32
from cutlass._mlir.dialects import nvvm

from .primitives import (
    asm,
    load_words,
    published,
    store_relaxed,
    wait_deadline_check,
    wait_deadline_start,
)

SENTINEL = 0x80000000


# Re-arm ordering ("mc_rf"): under interleaved HBM
# streaming a reader's relaxed re-arm of a slot was overtaken by the next remote publish into that slot, which then
# read as armed (deadline trap). One fence.acq_rel.gpu per thread after its re-arms, before its later stores,
# launch_dependents and exit, orders the re-arm first. Not a strict-PTX fix (gpu scope, multicast alias);
# measured cost per hop in a stress probe. Default on; K31MK_REARM_FENCE=0 removes it (A/B only).
# K31MK_REARM_FENCE modes: "1" / "gpu" (default, the fence above), "0" (none: a cost
# control only, unsafe), "sys" (fence.acq_rel.sys), "sys_alias" (fence.proxy.alias, then fence.acq_rel.sys: a
# Dat3M litmus test forbids the lost publish with sys fences on both ranks and
# the alias fence on the re-arming rank; every poller here re-arms, so each rank runs both sides). Read at import;
# a test may set ``REARM_FENCE`` before it compiles kernels.
_MODES = {"1": "gpu", "gpu": "gpu", "0": "", "sys": "sys", "sys_alias": "sys_alias"}
REARM_FENCE = _MODES[os.environ.get("K31MK_REARM_FENCE", "1")]

# K-23: an mbarrier that
# peer CTAs complete with remote st.async (ClusterAttnRes / OprojConsumer arm_exchanges, MoeTail bar + 1 / bar + 2,
# Fp32FrontGemm `owned`) was armed at the default .release.cta scope, which is not morally strong with a peer's
# complete_tx (PTX 8.7): when a peer's bytes land before the arm, the waiter's try_wait.acquire.cluster does not order
# the peer's data (8.9.2, 8.9.4). Arming at .relaxed.cluster closes it (litmus-tested: forbidden; same SASS on CuTe DSL
# 4.8.0). Always on. Pass as ``**K23_ARM`` to cute.arch.mbarrier_arrive_and_expect_tx at every arm that peers complete.
K23_ARM = {"relaxed": True, "scope": nvvm.MemScopeKind.CLUSTER}


def rearm(pointer):
    store_relaxed(pointer, (Uint32(SENTINEL),) * 4)


@asm()
def _fence_acq_rel_sys():
    return "fence.acq_rel.sys;", "~{memory}", ()


@asm()
def _fence_proxy_alias():
    return "fence.proxy.alias;", "~{memory}", ()


@cute.jit
def rearm_fence(force: cutlass.Constexpr[bool] = False):
    """``REARM_FENCE``: one acq_rel fence (gpu scope by default) after this thread's
    re-arms; ``force``: a gpu fence even when the mode is "0" (a caller that needs it
    for its own hand-off, e.g. the B1 x_moe publish)."""
    if cutlass.const_expr(REARM_FENCE == "gpu" or (force and REARM_FENCE == "")):
        cute.arch.fence_acq_rel_gpu()
    if cutlass.const_expr(REARM_FENCE == "sys_alias"):
        _fence_proxy_alias()
    if cutlass.const_expr(REARM_FENCE in ("sys", "sys_alias")):
        _fence_acq_rel_sys()


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
