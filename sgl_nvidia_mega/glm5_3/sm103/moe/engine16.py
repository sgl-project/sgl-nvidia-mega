# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Small-M NVFP4 routed-expert engine, four-request form: one launch for 2..4 requests
of 1..4 rows each (M <= 16, request-major), with the BF16 shared expert in-engine.

The unit and its arithmetic are the one-request engine's (``engine.py``); what changes
is which token columns a unit owns:

* the token image has 16 rows (MMA N = 16): image row ``4 q + t`` is token t of request
  q (rows ``t >= rows(q)`` are zero), request q's tokens are accumulator columns ``4 q
  .. 4 q + 3``; the second 8-row group of every K-major image sits one group stride
  after the first;
* the expert list is the union of the requests' experts, ordered by class (experts of
  three or more requests, then two, then one; ascending ids within a class) so that the
  round-robin schedule spreads multi-request units over the clusters; each entry carries
  a group mask (bit q: request q routes to it), so an expert several requests use
  streams once;
* per unit and group: one K-split exchange and one 4-token h chain; FC2 reads the
  group's four columns;
* the shared-expert clusters use a 3-slot ring and keep their 16-row BF16 token image in
  the space of the fourth slot (routed clusters never write it).

Warp roles (256 threads): 0 expert union + producer, 1 MMA issue + TMEM owner, 2..5
epilogue (TMEM lane quadrant ``warp % 4``; warp ``2 + q`` builds request q's per-unit
token data), 6 scale-factor copies, 7 receive-buffer release.

Outputs as the one-request engine, by global row (row stride M)."""

from __future__ import annotations

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
from cutlass import Float32, Int32, Int64, Uint32, Uint64
from cutlass.experimental.cuda import (
    TensorMap,
    TensorMapL2Promotion,
    TensorMapSwizzle,
    create_tensor_map_tiled,
)
from cutlass.memory import SmemAllocator

from ..primitives import ptx_core as C
from ..primitives import ptx_tc as T
from . import _core_ptx as P
from .constants import HIDDEN, NUM_EXPERTS, TOP_K
from .engine import (
    BIG,
    CLUSTER,
    ERR_BAD_ID,
    ERR_UNIT_CAP,
    FC1_CPS,
    FC1_MMAS,
    FC2_CPS,
    FC2_RECORD,
    FC2_TILES,
    H_BUFS,
    H_CPS,
    K_CTA,
    LOOKAHEAD,
    RELEASE_WARP,
    REQ_ROWS,
    RING,
    SF13_PER_EXPERT,
    SF_WARP,
    SH_CHUNK,
    SH_CHUNKS,
    SH_CLUSTERS,
    SH_FC1_CHUNKS,
    SH_FC1_MMAS,
    SH_RECORD,
    SLOT,
    _align16,
)

GROUPS = 4
N = GROUPS * REQ_ROWS  # 16 image rows
RING_SH = 3
RECV_BUFS = 5
RECV_BUF = 3 * 128 * 16  # [3 peer slots][128 rows][4 FP32]
X_GRP = 6144  # NVFP4 image, one 8-row group: [48 K cores][8 rows][16 B]
XBF_GRP = 24576  # BF16 image, one 8-row group: [192 K cores][8 rows][16 B]
XSF_BYTES = (
    24 * 256 + 256
)  # per K step a 16-row scale window; +256 for the last 32-row copy
H_BUF = 1024  # NVFP4 h: [2 groups][2 K cores][128 B] = 512, scale window at +512
HBF_BYTES = 2048  # BF16 h: [2 groups][8 K cores][128 B]

OFF_RING = 0
OFF_XBF = RING_SH * BIG
OFF_X = max(RING * BIG, OFF_XBF + 2 * XBF_GRP)
OFF_XSF = OFF_X + 2 * X_GRP
OFF_RECV = OFF_XSF + XSF_BYTES
OFF_H = OFF_RECV + RECV_BUFS * RECV_BUF
OFF_HBF = OFF_H + H_BUFS * H_BUF
OFF_MISC = OFF_HBF + HBF_BYTES
SMEM = 223664  # one CTA per SM

MAX_UNITS = 17
MAX_LIST = 128
SCHED_ROW = 18
SCHED_BYTES = 136 * SCHED_ROW * 2

# tensor memory columns
T_ACC1 = 0  # 2 x 16
T_ACC2 = 32  # 12 x 16
T_SFB_X = 224  # 24 x 4
T_SFB_H = 320  # H_BUFS x 4
T_SFA = 384  # RING x 32

IDESC_NVF4 = T.idesc_nvf4(128, N)
IDESC_BF16 = T.idesc_bf16(128, N)

FC2_MMAS = tuple(
    (N * (4 * h2 + t), SLOT * h2 + 4096 * t, 0, 16 * h2 + 4 * t, 0, 0, h2)
    for h2 in range(2)
    for t in range(4)
)
SH_FC2_MMAS = tuple(
    (N * h2, 16384 * h2 + 256 * kk, 256 * kk, 0, 0, 0 if kk == 0 else 1, 0)
    for h2 in range(2)
    for kk in range(4)
)
X_CPS = tuple((4 * q, 256 * q, 0) for q in range(4))

# misc block (offsets from OFF_MISC)
B_FULL = 0
B_EMPTY = 32
B_XREADY = 64
B_XBAR = 72
B_TREADY = 80
B_SBAR = 88
B_ACC1_FULL = 96
B_ACC1_EMPTY = 112
B_ACC2_FULL = 128
B_ACC2_EMPTY = 136
B_HREADY = 144
B_SFA_FULL = B_HREADY + 8 * H_BUFS
B_SFX_READY = B_SFA_FULL + 8 * RING
B_PRECV = B_SFX_READY + 8
B_PFREE = B_PRECV + 8 * RECV_BUFS
B_CONSUMED = B_PFREE + 8 * RECV_BUFS
M_TMEM = B_CONSUMED + 8 * RECV_BUFS
M_NUNITS = M_TMEM + 4
M_NEX = M_TMEM + 8
M_BM = _align16(M_TMEM + 12)  # [4 requests][8 words] expert bitmaps
M_ELIST = M_BM + 4 * 8 * 4
M_EGM = M_ELIST + 4 * MAX_LIST  # one byte per entry
M_STAB = _align16(M_EGM + MAX_LIST)
M_UNITS = _align16(M_STAB + SCHED_BYTES)
# unit record: expert, quarter, shared flag, group mask, g1 alpha, token scale [16]
# (FP32), slot [16] (s8)
U_E, U_J, U_SH, U_GM, U_G1, U_TOK = 0, 4, 8, 12, 16, 20
U_SLOT = U_TOK + 4 * N
UNIT_BYTES = U_SLOT + N
assert OFF_MISC + M_UNITS + MAX_UNITS * UNIT_BYTES + 1024 <= SMEM


class Nvfp4MoeSmallM16:
    """``shared``: run the shared expert's four quarters in-engine (needs >= 8
    clusters). The expert list is the class-ordered union (module docstring).

    Call: as ``engine.Nvfp4MoeSmallM`` with ``split`` replaced by ``rows_per_request``:
    the rows of request q in byte q (1..4 for the present requests, 0 after the last).
    """

    def __init__(self, shared: bool = True):
        self.shared = shared

    @cute.jit
    def __call__(
        self,
        w13: Int64,
        x: Int64,
        ids: Int64,
        topw: Int64,
        shpack: Int64,
        g1a: Int64,
        g2a: Int64,
        gs1: Float32,
        gs2: Float32,
        rows: Int32,
        mqs: Int32,
        err: Int64,
        sched: Int64,
        shpartials: Int64,
        pq: Int64,
        ids_fin: Int64,
        w_fin: Int64,
        g2s: Int64,
        sf13: Int64,
        fc2pack: Int64,
        nclusters: Int32,
        stream: cuda.CUstream,
    ):
        tmap = create_tensor_map_tiled(
            w13,
            cutlass.Uint8,
            [128, NUM_EXPERTS * 512, 24],
            [3072 // 16, 128 // 16],
            [128, 128, 2],
            swizzle=TensorMapSwizzle.s128b,
            l2_promotion=TensorMapL2Promotion.l2_256b,
        )
        self.kernel(
            tmap,
            x,
            ids,
            topw,
            shpack,
            g1a,
            g2a,
            gs1,
            gs2,
            rows,
            mqs,
            err,
            sched,
            shpartials,
            pq,
            ids_fin,
            w_fin,
            g2s,
            sf13,
            fc2pack,
            nclusters,
        ).launch(
            grid=[CLUSTER * nclusters, 1, 1],
            block=[256, 1, 1],
            cluster=[CLUSTER, 1, 1],
            max_number_threads=[256, 1, 1],
            min_blocks_per_mp=1,
            stream=stream,
            use_pdl=True,
        )

    # ------------------------------------------------------------ protocol hooks
    # As in engine.Nvfp4MoeSmallM: each protocol step is one method here.
    @cute.jit
    def wait_schedule(self, misc):
        C.mbarrier_wait_parity(misc + B_SBAR, 0)

    @cute.jit
    def wait_inputs(self):
        C.pdl_wait()

    @cute.jit
    def wait_scales(self, bar, parity):
        C.mbarrier_wait_parity(bar, parity)

    @cute.jit
    def hready_slot(self, i):
        return i % H_BUFS

    @cute.jit
    def hready_phase(self, i):
        return (i // H_BUFS) & 1

    @cute.jit
    def order_scale_rows(self):
        """The zero fill of the x scale windows and the quantization's scale-byte stores
        (other threads, same 16-byte rows) are ordered."""
        cute.arch.barrier(barrier_id=2, number_of_threads=128)

    @cute.jit
    def ring_tail(self, misc, it, nring):
        first = it - nring
        if first < 0:
            first = Int32(0)
        for c in cutlass.range(first, it, unroll=1):
            C.mbarrier_wait_parity(misc + B_EMPTY + 8 * (c % nring), (c // nring) & 1)

    # ------------------------------------------------------------ helpers
    @cute.jit
    def ufield(self, misc, i, field: cutlass.Constexpr[int]):
        return C.ld_shared_u32(misc + M_UNITS + i * UNIT_BYTES + field)

    @cute.jit
    def rows_of(self, mqs, q):
        """Rows of request q (q may be dynamic)."""
        return (mqs >> (8 * q)) & 0xFF

    @cute.jit
    def first_row_of(self, rbs, q):
        return (rbs >> (8 * q)) & 0xFF

    @cute.jit
    def fc1_chunk(self, tmap_ptr, sf13, dst, expert, quarter, crank, c, bar, policy):
        T.tma_load_3d(
            dst,
            tmap_ptr,
            0,
            expert * 512 + 128 * quarter,
            crank * 6 + 2 * c,
            bar,
            policy,
        )
        src = (
            sf13
            + Int64(expert) * SF13_PER_EXPERT
            + Int64(96 * quarter + 24 * crank + 8 * c) * 512
        )
        C.bulk_copy_g2s(dst + 32768, src, 4096, bar, policy)

    @cute.jit
    def scan8(self, cnt, lane):
        incl = cnt
        for o in cutlass.range_constexpr(3):
            t = cute.arch.shuffle_sync_up(incl, 1 << o)
            if lane >= (1 << o):
                incl = incl + t
        return incl

    @cute.jit
    def stream(
        self,
        tmap_ptr,
        shpack,
        sf13,
        fc2pack,
        base,
        misc,
        crank,
        nunits,
        nring,
        pre,
        pre0,
        policy,
    ):
        """Producer lane: stream order FC1(0), FC1(1), FC2(0), FC1(2), FC2(1), ..., on a
        ring of ``nring`` slots, then wait for the MMA warp's last commit on every slot.
        """
        it = pre + pre0
        for kk in cutlass.range(nunits + LOOKAHEAD, unroll=1):
            for part in cutlass.range_constexpr(2):
                i = kk if part == 0 else kk - LOOKAHEAD
                if (i >= 0) & (i < nunits):
                    ush = self.ufield(misc, i, U_SH)
                    if ush != Uint32(0):
                        uj = Int32(self.ufield(misc, i, U_J))
                        src = shpack + (Int64(uj) * 4 + Int64(crank)) * SH_RECORD
                        c0 = Int32(SH_FC1_CHUNKS)
                        c1 = Int32(SH_CHUNKS)
                        if cutlass.const_expr(part == 0):
                            c0 = Int32(0)
                            if i == 0:
                                c0 = pre
                            c1 = Int32(SH_FC1_CHUNKS)
                        for c in cutlass.range(c0, c1, unroll=1):
                            s = it % nring
                            if it >= nring:
                                C.mbarrier_wait_parity(
                                    misc + B_EMPTY + 8 * s, ((it // nring) - 1) & 1
                                )
                            C.mbarrier_arrive_expect_tx(misc + B_FULL + 8 * s, SH_CHUNK)
                            C.bulk_copy_g2s(
                                base + OFF_RING + s * BIG,
                                src + Int64(c) * SH_CHUNK,
                                SH_CHUNK,
                                misc + B_FULL + 8 * s,
                                policy,
                            )
                            it = it + 1
                    else:
                        u_e_idx = Int32(self.ufield(misc, i, U_E))
                        uj = Int32(self.ufield(misc, i, U_J))
                        src2 = (
                            fc2pack
                            + ((Int64(u_e_idx) * 4 + Int64(uj)) * 4 + Int64(crank))
                            * FC2_RECORD
                        )
                        c0 = Int32(3)
                        c1 = Int32(5)
                        if cutlass.const_expr(part == 0):
                            c0 = Int32(0)
                            if i == 0:
                                c0 = pre0
                            c1 = Int32(3)
                        for c in cutlass.range(c0, c1, unroll=1):
                            s = it % nring
                            if it >= nring:
                                C.mbarrier_wait_parity(
                                    misc + B_EMPTY + 8 * s, ((it // nring) - 1) & 1
                                )
                            nbytes = Int32(BIG)
                            if c == 4:
                                nbytes = Int32(SLOT)
                            C.mbarrier_arrive_expect_tx(misc + B_FULL + 8 * s, nbytes)
                            if c < 3:
                                # expert / quarter are read from the unit table again
                                # for each FC1 chunk
                                ce = Int32(self.ufield(misc, i, U_E))
                                cj = Int32(self.ufield(misc, i, U_J))
                                self.fc1_chunk(
                                    tmap_ptr,
                                    sf13,
                                    base + OFF_RING + s * BIG,
                                    ce,
                                    cj,
                                    crank,
                                    c,
                                    misc + B_FULL + 8 * s,
                                    policy,
                                )
                            else:
                                C.bulk_copy_g2s(
                                    base + OFF_RING + s * BIG,
                                    src2 + Int64(2 * c - 6) * SLOT,
                                    nbytes,
                                    misc + B_FULL + 8 * s,
                                    policy,
                                )
                            it = it + 1
        self.ring_tail(misc, it, nring)

    # ------------------------------------------------ warp 0: union, schedule, producer
    @cute.jit
    def producer(
        self,
        tmap_ptr,
        ids,
        shpack,
        mqs,
        rbs,
        gall,
        err,
        sched,
        sf13,
        fc2pack,
        base,
        misc,
        crank,
        cl,
        my_sh,
        nring,
        lane,
    ):
        policy = C.l2_policy("evict_first")
        pre = Int32(0)
        if cutlass.const_expr(self.shared):  # noqa: SIM102
            if my_sh != 0:
                src = shpack + (Int64(cl) * 4 + Int64(crank)) * SH_RECORD
                if lane == 0:
                    for c in cutlass.range_constexpr(RING_SH):
                        C.mbarrier_arrive_expect_tx(misc + B_FULL + 8 * c, SH_CHUNK)
                        C.bulk_copy_g2s(
                            base + OFF_RING + c * BIG,
                            src + c * SH_CHUNK,
                            SH_CHUNK,
                            misc + B_FULL + 8 * c,
                            policy,
                        )
                pre = Int32(RING_SH)
                c = RING_SH + lane
                if c < SH_CHUNKS:
                    C.bulk_prefetch_l2(src + Int64(c) * SH_CHUNK, SH_CHUNK)
        if lane == 0:
            C.mbarrier_arrive_expect_tx(misc + B_SBAR, SCHED_BYTES)
            C.bulk_copy_g2s(
                misc + M_STAB,
                sched + Int64(cl) * SCHED_BYTES,
                SCHED_BYTES,
                misc + B_SBAR,
                C.l2_policy("evict_last"),
            )
        C.st_shared_u32(misc + M_BM + 4 * lane, Uint32(0))
        cute.arch.sync_warp()
        C.pdl_wait()
        # lane l holds (token l >> 3, slot l & 7) of each request; per-request bitmaps
        # by shared atomics
        badid = Int32(0)
        for q in cutlass.range_constexpr(GROUPS):
            mq = self.rows_of(mqs, q)
            inside = lane < TOP_K * mq
            eq = Int32(-1)
            if inside:
                eq = P.ld_cg_s32(
                    ids + Int64(TOP_K * self.first_row_of(rbs, q) + lane) * 4
                )
            ok = Int32(0)
            if inside & (Uint32(eq) < Uint32(NUM_EXPERTS)):
                ok = Int32(1)
            if inside & (ok == 0):
                badid = Int32(1)
            if ok != 0:
                P.atom_or_shared(
                    misc + M_BM + 4 * (8 * q + (eq >> 5)), Uint32(1) << Uint32(eq & 31)
                )
        if badid != 0:
            P.atom_or_global(err, ERR_BAD_ID)
        cute.arch.sync_warp()
        wq0 = Uint32(0)
        wq1 = Uint32(0)
        wq2 = Uint32(0)
        wq3 = Uint32(0)
        if lane < NUM_EXPERTS // 32:
            wq0 = C.ld_shared_u32(misc + M_BM + 4 * lane)
            wq1 = C.ld_shared_u32(misc + M_BM + 4 * (8 + lane))
            wq2 = C.ld_shared_u32(misc + M_BM + 4 * (16 + lane))
            wq3 = C.ld_shared_u32(misc + M_BM + 4 * (24 + lane))
        wq = [wq0, wq1, wq2, wq3]
        # classes: experts of >= 3 requests, of exactly 2, of 1 (placement only)
        w0 = wq[0] | wq[1] | wq[2] | wq[3]
        s1 = wq[0] ^ wq[1]
        c1 = wq[0] & wq[1]
        s2 = s1 ^ wq[2]
        c2 = s1 & wq[2]
        lo = s2 ^ wq[3]
        c3 = s2 & wq[3]
        ge2 = c1 | c2 | c3
        ge3 = (ge2 & lo) | (wq[0] & wq[1] & wq[2] & wq[3])
        wcls = [ge3, ge2 & ~ge3, w0 & ~ge2]
        ccnt = [P.popc(wcls[c]) for c in range(3)]
        cincl = [self.scan8(ccnt[c], lane) for c in range(3)]
        ctot = [cute.arch.shuffle_sync(cincl[c], 7) for c in range(3)]
        cbase1 = ctot[0]
        cbase2 = ctot[0] + ctot[1]
        d0 = cbase2 + ctot[2]
        # unit 0 resolved from the schedule row and the class bitmaps; its first copies
        # issue now
        pre0 = Int32(0)
        self.wait_schedule(misc)
        row0 = misc + M_STAB + 2 * SCHED_ROW * min(d0, MAX_LIST)
        code0 = Int32(-1)
        if P.ld_shared_s16(row0) > 0:
            code0 = P.ld_shared_s16(row0 + 2)
        if code0 >= 0:
            idx = code0 >> 2
            r0 = idx
            ci = cincl[0]
            cc = ccnt[0]
            cw = wcls[0]
            if idx >= cbase2:
                r0 = idx - cbase2
                ci = cincl[2]
                cc = ccnt[2]
                cw = wcls[2]
            elif idx >= cbase1:
                r0 = idx - cbase1
                ci = cincl[1]
                cc = ccnt[1]
                cw = wcls[1]
            mine = (r0 >= ci - cc) & (r0 < ci)
            who = Uint32(cute.arch.vote_ballot_sync(mine))
            e0 = Int32(0)
            if mine:
                bits = cw
                for _ in cutlass.range(r0 - (ci - cc), unroll=1):
                    bits = bits & (bits - Uint32(1))
                e0 = 32 * lane + P.ctz(bits)
            src_lane = Int32(0)
            if who != Uint32(0):
                src_lane = P.ctz(who)
            e0 = cute.arch.shuffle_sync(e0, src_lane)
            if (lane == 0) & (who != Uint32(0)):
                C.mbarrier_arrive_expect_tx(misc + B_FULL, BIG)
                self.fc1_chunk(
                    tmap_ptr,
                    sf13,
                    base + OFF_RING,
                    e0,
                    code0 & 3,
                    crank,
                    0,
                    misc + B_FULL,
                    policy,
                )
                pre0 = Int32(1)
            pre0 = cute.arch.shuffle_sync(pre0, 0)
        # the union list in class order, with group masks
        for c in cutlass.range_constexpr(3):
            bits = wcls[c]
            k = cincl[c] - ccnt[c]
            if cutlass.const_expr(c == 1):
                k = k + cbase1
            if cutlass.const_expr(c == 2):
                k = k + cbase2
            while bits != Uint32(0):
                b = P.ctz(bits)
                if k < MAX_LIST:
                    C.st_shared_u32(misc + M_ELIST + 4 * k, 32 * lane + b)
                    gm = Uint32(0)
                    for q in cutlass.range_constexpr(GROUPS):
                        gm = gm | (((wq[q] >> Uint32(b)) & Uint32(1)) << Uint32(q))
                    P.st_shared_b8(misc + M_EGM + k, gm)
                k = k + 1
                bits = bits & (bits - Uint32(1))
        cute.arch.sync_warp()
        d = min(d0, MAX_LIST)
        self.wait_schedule(misc)
        row = misc + M_STAB + 2 * SCHED_ROW * d
        nraw = P.ld_shared_s16(row)
        nunits = min(nraw, MAX_UNITS)
        if (lane == 0) & (nraw > MAX_UNITS):
            P.atom_or_global(err, ERR_UNIT_CAP)
        gmu = Int32(0)
        if lane < nunits:
            code = P.ld_shared_s16(row + 2 + 2 * lane)
            unit = misc + M_UNITS + lane * UNIT_BYTES
            if code < 0:
                C.st_shared_u32(unit + U_E, Int32(-1))
                C.st_shared_u32(unit + U_J, cl)
                C.st_shared_u32(unit + U_SH, Int32(1))
                gmu = gall
            else:
                C.st_shared_u32(
                    unit + U_E, C.ld_shared_u32(misc + M_ELIST + 4 * (code >> 2))
                )
                C.st_shared_u32(unit + U_J, code & 3)
                C.st_shared_u32(unit + U_SH, Int32(0))
                gmu = P.ld_shared_u8(misc + M_EGM + (code >> 2))
            C.st_shared_u32(unit + U_GM, gmu)
        # exchanges of this cluster (one per (unit, group)), the same in its four CTAs
        nx = P.popc(gmu)
        for o in (16, 8, 4, 2, 1):
            nx = nx + cute.arch.shuffle_sync_bfly(nx, o)
        if lane == 0:
            C.st_shared_u32(misc + M_NEX, nx)
        cute.arch.sync_warp()
        if lane == 0:
            if (pre0 == 0) & (nunits > 0):  # noqa: SIM102
                if self.ufield(misc, 0, U_SH) == Uint32(0):
                    C.mbarrier_arrive_expect_tx(misc + B_FULL, BIG)
                    e0 = Int32(self.ufield(misc, 0, U_E))
                    j0 = Int32(self.ufield(misc, 0, U_J))
                    self.fc1_chunk(
                        tmap_ptr,
                        sf13,
                        base + OFF_RING,
                        e0,
                        j0,
                        crank,
                        0,
                        misc + B_FULL,
                        policy,
                    )
                    pre0 = Int32(1)
            C.st_shared_u32(misc + M_NUNITS, nunits)
        # Every lane arrives (count 32): each lane's own unit-record stores are
        # released by its own arrive (see the init of B_TREADY).
        C.mbarrier_arrive(misc + B_TREADY)
        cute.arch.sync_warp()
        if lane == 0:
            self.stream(
                tmap_ptr,
                shpack,
                sf13,
                fc2pack,
                base,
                misc,
                crank,
                nunits,
                nring,
                pre,
                pre0,
                policy,
            )
        cute.arch.sync_warp()

    # ------------------------------------------------------------ warp 1: MMA issue
    @cute.jit
    def mma_warp(self, base, misc, nring):
        lead = T.elect_leader()
        C.mbarrier_wait_parity(misc + B_TREADY, 0)
        nunits = cute.arch.shuffle_sync(Int32(C.ld_shared_u32(misc + M_NUNITS)), 0)
        tm = cute.arch.shuffle_sync(Int32(C.ld_shared_u32(misc + M_TMEM)), 0)
        if nunits > 0:
            C.mbarrier_wait_parity(misc + B_XREADY, 0)
            self.wait_scales(misc + B_SFX_READY, 0)
            T.tc_fence_after()
        s_ring = base + OFF_RING
        it = Int32(0)
        for k in cutlass.range(nunits + LOOKAHEAD, unroll=1):
            shk = Int32(0)
            if k < nunits:
                shk = cute.arch.shuffle_sync(Int32(self.ufield(misc, k, U_SH)), 0)
            if k < nunits:
                b = k & 1
                if k >= 2:
                    C.mbarrier_wait_parity(
                        misc + B_ACC1_EMPTY + 8 * b, ((k >> 1) - 1) & 1
                    )
                    T.tc_fence_after()
                if shk != 0:
                    for c in cutlass.range(SH_FC1_CHUNKS, unroll=1):
                        s = it % nring
                        C.mbarrier_wait_parity(misc + B_FULL + 8 * s, (it // nring) & 1)
                        self.wait_scales(misc + B_SFA_FULL + 8 * s, (it // nring) & 1)
                        T.tc_fence_after()
                        a0 = T.smem_desc(s_ring + s * BIG, 128, 1024, 0)
                        b0 = T.smem_desc(base + OFF_XBF + 2048 * c, 128, XBF_GRP, 0)
                        T.tc_burst(
                            lead,
                            tm + T_ACC1 + N * b,
                            a0,
                            b0,
                            0,
                            0,
                            Int32(c != 0),
                            0,
                            misc + B_EMPTY + 8 * s,
                            0,
                            0,
                            mmas=SH_FC1_MMAS,
                            block_scaled=False,
                            idesc=IDESC_BF16,
                        )
                        cute.arch.sync_warp()
                        it = it + 1
                else:
                    for c in cutlass.range(3, unroll=1):
                        s = it % nring
                        C.mbarrier_wait_parity(misc + B_FULL + 8 * s, (it // nring) & 1)
                        self.wait_scales(misc + B_SFA_FULL + 8 * s, (it // nring) & 1)
                        T.tc_fence_after()
                        a0 = T.smem_desc(s_ring + s * BIG, 16, 1024, 2)
                        b0 = T.smem_desc(base + OFF_X + 2048 * c, 128, X_GRP, 0)
                        T.tc_burst(
                            lead,
                            tm + T_ACC1 + N * b,
                            a0,
                            b0,
                            tm + T_SFA + 32 * s,
                            tm + T_SFB_X + 32 * c,
                            Int32(c != 0),
                            0,
                            misc + B_EMPTY + 8 * s,
                            0,
                            0,
                            mmas=FC1_MMAS,
                            block_scaled=True,
                            idesc=IDESC_NVF4,
                        )
                        cute.arch.sync_warp()
                        it = it + 1
                T.tc_commit(lead, misc + B_ACC1_FULL + 8 * b)
                cute.arch.sync_warp()
            if (k >= LOOKAHEAD) & (k - LOOKAHEAD < nunits):
                i = k - LOOKAHEAD
                shi = cute.arch.shuffle_sync(Int32(self.ufield(misc, i, U_SH)), 0)
                C.mbarrier_wait_parity(
                    misc + B_HREADY + 8 * self.hready_slot(i), self.hready_phase(i)
                )
                T.tc_fence_after()
                if i > 0:
                    C.mbarrier_wait_parity(misc + B_ACC2_EMPTY, (i - 1) & 1)
                    T.tc_fence_after()
                if shi != 0:
                    bh = T.smem_desc(base + OFF_HBF, 128, 1024, 0)
                    for c in cutlass.range(SH_FC1_CHUNKS, SH_CHUNKS, unroll=1):
                        s = it % nring
                        C.mbarrier_wait_parity(misc + B_FULL + 8 * s, (it // nring) & 1)
                        self.wait_scales(misc + B_SFA_FULL + 8 * s, (it // nring) & 1)
                        T.tc_fence_after()
                        a0 = T.smem_desc(s_ring + s * BIG, 128, 1024, 0)
                        T.tc_burst(
                            lead,
                            tm + T_ACC2 + 2 * N * (c - SH_FC1_CHUNKS),
                            a0,
                            bh,
                            0,
                            0,
                            0,
                            0,
                            misc + B_EMPTY + 8 * s,
                            0,
                            0,
                            mmas=SH_FC2_MMAS,
                            block_scaled=False,
                            idesc=IDESC_BF16,
                        )
                        cute.arch.sync_warp()
                        it = it + 1
                else:
                    hb = i % H_BUFS
                    bh = T.smem_desc(base + OFF_H + H_BUF * hb, 128, 256, 0)
                    for c in cutlass.range(3, 5, unroll=1):
                        s = it % nring
                        C.mbarrier_wait_parity(misc + B_FULL + 8 * s, (it // nring) & 1)
                        self.wait_scales(misc + B_SFA_FULL + 8 * s, (it // nring) & 1)
                        T.tc_fence_after()
                        a0 = T.smem_desc(s_ring + s * BIG, 128, 256, 0)
                        T.tc_burst(
                            lead,
                            tm + T_ACC2 + 4 * N * (2 * c - 6),
                            a0,
                            bh,
                            tm + T_SFA + 32 * s,
                            tm + T_SFB_H + 4 * hb,
                            0,
                            Int32(c == 3),
                            misc + B_EMPTY + 8 * s,
                            0,
                            0,
                            mmas=FC2_MMAS,
                            block_scaled=True,
                            idesc=IDESC_NVF4,
                        )
                        cute.arch.sync_warp()
                        it = it + 1
                T.tc_commit(lead, misc + B_ACC2_FULL)
                cute.arch.sync_warp()
        cute.arch.sync_warp()

    # -------------------------------------------------- warp 6: scale factors into TMEM
    @cute.jit
    def sf_warp(self, base, misc, nring):
        lead = T.elect_leader()
        C.mbarrier_wait_parity(misc + B_TREADY, 0)
        nunits = cute.arch.shuffle_sync(Int32(C.ld_shared_u32(misc + M_NUNITS)), 0)
        tm = cute.arch.shuffle_sync(Int32(C.ld_shared_u32(misc + M_TMEM)), 0)
        if nunits > 0:
            C.mbarrier_wait_parity(misc + B_XREADY, 0)
            T.tc_fence_after()
            d0 = T.smem_desc(base + OFF_XSF, 0, 128, 0)
            for q in cutlass.range(6, unroll=1):
                T.tc_burst(
                    lead,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    tm + T_SFB_X + 16 * q,
                    d0 + Uint64(64 * q),
                    cps=X_CPS,
                    commit=False,
                )
            T.tc_commit(lead, misc + B_SFX_READY)
            cute.arch.sync_warp()
        it = Int32(0)
        for k in cutlass.range(nunits + LOOKAHEAD, unroll=1):
            for part in cutlass.range_constexpr(2):
                i = k if part == 0 else k - LOOKAHEAD
                if (i >= 0) & (i < nunits):
                    ush = cute.arch.shuffle_sync(Int32(self.ufield(misc, i, U_SH)), 0)
                    c0 = Int32(0)
                    c1 = Int32(3)
                    if cutlass.const_expr(part == 0):
                        if ush != 0:
                            c1 = Int32(SH_FC1_CHUNKS)
                    else:
                        c0 = Int32(3)
                        c1 = Int32(5)
                        if ush != 0:
                            c0 = Int32(SH_FC1_CHUNKS)
                            c1 = Int32(SH_CHUNKS)
                        else:
                            C.mbarrier_wait_parity(
                                misc + B_HREADY + 8 * self.hready_slot(i),
                                self.hready_phase(i),
                            )
                            T.tc_fence_after()
                            hb = i % H_BUFS
                            T.tc_burst(
                                lead,
                                0,
                                0,
                                0,
                                0,
                                0,
                                0,
                                0,
                                0,
                                tm + T_SFB_H + 4 * hb,
                                T.smem_desc(base + OFF_H + H_BUF * hb + 512, 0, 128, 0),
                                cps=H_CPS,
                                commit=False,
                            )
                            cute.arch.sync_warp()
                    for c in cutlass.range(c0, c1, unroll=1):
                        s = it % nring
                        C.mbarrier_wait_parity(misc + B_FULL + 8 * s, (it // nring) & 1)
                        T.tc_fence_after()
                        if ush != 0:
                            T.tc_commit(lead, misc + B_SFA_FULL + 8 * s)
                        else:
                            if cutlass.const_expr(part == 0):
                                d = T.smem_desc(
                                    base + OFF_RING + s * BIG + 32768, 0, 128, 0
                                )
                                T.tc_burst(
                                    lead,
                                    0,
                                    0,
                                    0,
                                    0,
                                    0,
                                    0,
                                    0,
                                    misc + B_SFA_FULL + 8 * s,
                                    tm + T_SFA + 32 * s,
                                    d,
                                    cps=FC1_CPS,
                                )
                            else:
                                d = T.smem_desc(
                                    base + OFF_RING + s * BIG + 16384, 0, 128, 0
                                )
                                T.tc_burst(
                                    lead,
                                    0,
                                    0,
                                    0,
                                    0,
                                    0,
                                    0,
                                    Int32(c == 3),
                                    misc + B_SFA_FULL + 8 * s,
                                    tm + T_SFA + 32 * s,
                                    d,
                                    cps=FC2_CPS,
                                )
                        cute.arch.sync_warp()
                        it = it + 1
        cute.arch.sync_warp()

    # --------------------------------------------------- warp 7: receive-buffer release
    @cute.jit
    def release_warp(self, misc, crank, lane):
        C.mbarrier_wait_parity(misc + B_TREADY, 0)
        nex = cute.arch.shuffle_sync(Int32(C.ld_shared_u32(misc + M_NEX)), 0)
        cute.arch.cluster_wait()
        if lane == 0:
            for e in cutlass.range(nex - RECV_BUFS, unroll=1):
                p = e % RECV_BUFS
                C.mbarrier_wait_parity(misc + B_CONSUMED + 8 * p, (e // RECV_BUFS) & 1)
                for pr in cutlass.range_constexpr(CLUSTER):
                    if crank != pr:
                        P.mbarrier_arrive_remote(
                            C.mapa_shared_cluster(misc + B_PFREE + 8 * p, pr)
                        )
        cute.arch.sync_warp()

    # ------------------------------------------------------------ warps 2..5
    @cute.jit
    def quantize_x(self, x, gs1, mqs, rbs, base, misc, crank, my_sh, et, lane):
        """The activations x (BF16) of this CTA's K range into the NVFP4 image +
        scale windows (routed operand) and, on the shared-expert clusters, the BF16
        image. Image row n = 4 q + t: request q row t (valid t < rows(q)); invalid
        rows are zero."""
        xbf = base + OFF_XBF
        for hf in cutlass.range(2, unroll=1):
            raw = cute.make_rmem_tensor(cute.make_layout(48), Uint32)
            for ii in cutlass.range_constexpr(6):
                b = et + 128 * (hf * 6 + ii)
                n = b // 96
                kb = b % 96
                for w in cutlass.range_constexpr(8):
                    raw[8 * ii + w] = Uint32(0)
                if (n & 3) < self.rows_of(mqs, n >> 2):
                    grow = self.first_row_of(rbs, n >> 2) + (n & 3)
                    src = x + (Int64(grow) * HIDDEN + K_CTA * crank + 16 * kb) * 2
                    lo = P.ld_cg_v4_u32(src)
                    hi = P.ld_cg_v4_u32(src + 16)
                    for w in cutlass.range_constexpr(4):
                        raw[8 * ii + w] = lo[w]
                        raw[8 * ii + 4 + w] = hi[w]
            for ii in cutlass.range_constexpr(6):
                b = et + 128 * (hf * 6 + ii)
                n = b // 96
                kb = b % 96
                if cutlass.const_expr(self.shared):  # noqa: SIM102
                    if my_sh != 0:
                        gb = xbf + (n >> 3) * XBF_GRP + (n & 7) * 16
                        C.st_shared_v4(
                            gb + (2 * kb) * 128,
                            tuple(raw[8 * ii + w] for w in range(4)),
                        )
                        C.st_shared_v4(
                            gb + (2 * kb + 1) * 128,
                            tuple(raw[8 * ii + 4 + w] for w in range(4)),
                        )
                w0 = Uint32(0)
                w1 = Uint32(0)
                sfb = Uint32(0)
                if (n & 3) < self.rows_of(mqs, n >> 2):
                    pairs = [P.bf16x2_unpack(raw[8 * ii + w]) for w in range(8)]
                    v = [f for pr in pairs for f in pr]
                    amax = Float32(0.0)
                    for q in cutlass.range_constexpr(16):
                        amax = T.max_f32(amax, T.abs_f32(v[q]))
                    osc, sfb = T.nvfp4_block_scale(amax, gs1)
                    q0 = [P.mul_f32(v[q], osc) for q in range(8)]
                    q1 = [P.mul_f32(v[8 + q], osc) for q in range(8)]
                    w0 = T.e2m1x8(*q0)
                    w1 = T.e2m1x8(*q1)
                P.st_shared_v2_u32(
                    base
                    + OFF_X
                    + (n >> 3) * X_GRP
                    + (kb >> 1) * 128
                    + (n & 7) * 16
                    + (kb & 1) * 8,
                    w0,
                    w1,
                )
                P.st_shared_b8(
                    base + OFF_XSF + (kb >> 2) * 256 + n * 16 + (kb & 3), sfb
                )
        C.fence_proxy_async_shared_cta()
        cute.arch.sync_warp()
        # bar.sync (PTX 8.9.4): all lanes' stores precede the release
        cute.arch.barrier(barrier_id=3 + (((et + 64) >> 5) & 3), number_of_threads=32)
        if lane == 0:
            C.mbarrier_arrive(misc + B_XREADY)

    @cute.jit
    def token_data(
        self,
        ids,
        topw,
        g1a,
        g2a,
        mqs,
        rbs,
        ids_fin,
        w_fin,
        g2s,
        misc,
        block,
        nunits,
        qw,
        lane,
    ):
        """Request qw's rows 4 qw .. 4 qw + 3 of every unit: token scale (routing weight
        x FC2 alpha; shared: 1 for a valid row) and top-k slot; warp 2 (qw 0) also
        writes g1 alpha. Block 0 writes the slot records of request qw."""
        mqq = self.rows_of(mqs, qw)
        rbq = self.first_row_of(rbs, qw)
        eid = Int32(-1)
        wt = Float32(0.0)
        if lane < TOP_K * mqq:
            eid = P.ld_cg_s32(ids + Int64(TOP_K * rbq + lane) * 4)
            wt = P.ld_cg_f32(topw + Int64(TOP_K * rbq + lane) * 4)
        mysh = Int32(0)
        my_unit = Int32(-1)
        mygm = Int32(0)
        if lane < nunits:
            mysh = Int32(self.ufield(misc, lane, U_SH))
            mygm = Int32(self.ufield(misc, lane, U_GM))
            if mysh == 0:
                my_unit = Int32(self.ufield(misc, lane, U_E))
        g2 = Float32(0.0)
        if my_unit >= 0:
            g2 = P.ld_cg_f32(g2a + Int64(my_unit) * 4)
        tk = cute.make_rmem_tensor(cute.make_layout(REQ_ROWS), Float32)
        sl = cute.make_rmem_tensor(cute.make_layout(REQ_ROWS), Int32)
        for t in cutlass.range_constexpr(REQ_ROWS):
            tk[t] = Float32(0.0)
            sl[t] = Int32(-1)
        for u in cutlass.range(nunits, unroll=1):
            e = Int32(self.ufield(misc, u, U_E))
            msk = Uint32(cute.arch.vote_ballot_sync(eid == e))
            for t in cutlass.range_constexpr(REQ_ROWS):
                bits = (msk >> Uint32(8 * t)) & Uint32(0xFF)
                src = Int32(0)
                if bits != Uint32(0):
                    src = 8 * t + P.ctz(bits)
                w = cute.arch.shuffle_sync(wt, src)
                if lane == u:
                    if bits != Uint32(0):
                        tk[t] = w
                        sl[t] = P.ctz(bits)
                    else:
                        tk[t] = Float32(0.0)
                        sl[t] = Int32(-1)
        if (block == 0) & (lane < TOP_K * mqq):
            at = TOP_K * rbq + lane
            ok = Uint32(eid) < Uint32(NUM_EXPERTS)
            g2v = Float32(0.0)
            sid = Int32(-1)
            if ok:
                g2v = P.ld_cg_f32(g2a + Int64(eid) * 4)
                sid = eid
            P.st_global_b32(ids_fin + Int64(at) * 4, sid)
            P.st_global_b32f(w_fin + Int64(at) * 4, wt)
            P.st_global_b32f(g2s + Int64(at) * 4, g2v)
        if lane < nunits:
            unit = misc + M_UNITS + lane * UNIT_BYTES
            if qw == 0:
                g1 = Float32(1.0)
                if mysh == 0:
                    g1 = Float32(0.0)
                    if my_unit >= 0:
                        g1 = P.ld_cg_f32(g1a + Int64(my_unit) * 4)
                C.st_shared_f32(unit + U_G1, g1)
            own = (mygm >> qw) & 1
            for t in cutlass.range_constexpr(REQ_ROWS):
                r = REQ_ROWS * qw + t
                tok = Float32(0.0)
                slot = Int32(-1)
                if mysh != 0:
                    if (own != 0) & (t < mqq):
                        tok = Float32(1.0)
                else:
                    if own != 0:
                        tok = P.mul_f32(tk[t], g2)
                        slot = sl[t]
                C.st_shared_f32(unit + U_TOK + 4 * r, tok)
                P.st_shared_b8(unit + U_SLOT + r, Uint32(slot))

    @cute.jit
    def routed_h(self, acc, g1, gs2, nrows, r0, hbuf, qd, lane):
        """Routed unit, one group: h = SiLU(g1 gate) * (g1 up), NVFP4 into h image rows
        r0 .. r0 + nrows - 1 (lane i of a quadrant: gate iff i & 8, up partner i ^ 8);
        words packed with three shuffles (lanes 8 / 12 hold them), the scale byte at
        lane 8."""
        gate = (lane & 8) != 0
        hv = []
        am = []
        for n in cutlass.range_constexpr(REQ_ROWS):
            mine = P.mul_f32(acc[n], g1)
            up = cute.arch.shuffle_sync_bfly(mine, 8)
            h = P.mul_f32(T.silu_fast(mine), up)
            hv.append(h)
            a = Float32(0.0)
            if gate:
                a = T.abs_f32(h)
            am.append(a)
        for n in cutlass.range_constexpr(REQ_ROWS):
            for off in (16, 4, 2, 1):
                am[n] = T.max_f32(am[n], cute.arch.shuffle_sync_bfly(am[n], off))
        hq = []
        sfb = []
        for n in cutlass.range_constexpr(REQ_ROWS):
            osc, sf = T.nvfp4_block_scale(am[n], gs2)
            hq.append(P.mul_f32(hv[n], osc))
            sfb.append(sf)
        words = []
        for n in cutlass.range_constexpr(REQ_ROWS):
            by = T.e2m1x2(hq[n], cute.arch.shuffle_sync_down(hq[n], 16))
            hw = by | (cute.arch.shuffle_sync_down(by, 1) << Uint32(8))
            words.append(hw | (cute.arch.shuffle_sync_down(hw, 2) << Uint32(16)))
        for n in cutlass.range_constexpr(REQ_ROWS):
            if n < nrows:
                r = r0 + n
                if (lane == 8) | (lane == 12):
                    off = Int32(0)
                    if lane == 12:
                        off = Int32(4)
                    C.st_shared_u32(
                        hbuf
                        + (r >> 3) * 256
                        + (qd >> 1) * 128
                        + (r & 7) * 16
                        + (qd & 1) * 8
                        + off,
                        words[n],
                    )
                if lane == 8:
                    P.st_shared_b8(hbuf + 512 + r * 16 + qd, sfb[n])

    @cute.jit
    def shared_h(self, acc, nrows, r0, hbf, qd, lane):
        """Shared quarter, one group: stock BF16 rounding points into BF16 h rows r0 ..
        r0 + nrows - 1."""
        for n in cutlass.range_constexpr(REQ_ROWS):
            gb = P.round_bf16(acc[n])
            ub = cute.arch.shuffle_sync_down(gb, 16)
            sv = P.round_bf16(T.silu_accurate(gb))
            hb = Uint32(C.cvt_rn_bf16_f32(P.mul_f32(sv, ub)))
            hn = cute.arch.shuffle_sync_down(hb, 1)
            if (n < nrows) & (lane < 16) & ((lane & 1) == 0):
                kc = 2 * qd + (lane >> 3)
                r = r0 + n
                C.st_shared_u32(
                    hbf + (r >> 3) * 1024 + kc * 128 + (r & 7) * 16 + (lane & 7) * 2,
                    hb | (hn << Uint32(16)),
                )

    @cute.jit
    def readout(
        self, i, rows, mqs, rbs, shpartials, pq, misc, crank, lane_base, row, lane
    ):
        """FC2(i): per group, the 12 tiles' 4 columns -> partials slots (routed) or
        shared_partials (shared quarter); release the accumulator after the last group's
        loads."""
        C.mbarrier_wait_parity(misc + B_ACC2_FULL, i & 1)
        T.tc_fence_after()
        C.fence_proxy_async_shared_cta()
        gm = Int32(self.ufield(misc, i, U_GM))
        glast = 31 - P.clz(gm)
        for g in cutlass.range(GROUPS, unroll=1):
            if ((gm >> g) & 1) != 0:
                mg = self.rows_of(mqs, g)
                rbg = self.first_row_of(rbs, g)
                d = T.tmem_load4x12(lane_base + T_ACC2 + 4 * g, stride=N)
                if g == glast:
                    T.tc_fence_before()
                    cute.arch.sync_warp()
                    if lane == 0:
                        C.mbarrier_arrive(misc + B_ACC2_EMPTY)
                # unit fields are read per group, where the readout uses them
                if self.ufield(misc, i, U_SH) == Uint32(0):
                    for n in cutlass.range_constexpr(REQ_ROWS):
                        sl = P.ld_shared_s8(
                            misc + M_UNITS + i * UNIT_BYTES + U_SLOT + REQ_ROWS * g + n
                        )
                        if (n < mg) & (sl >= 0):
                            uj = Int32(self.ufield(misc, i, U_J))
                            q = (
                                pq
                                + (
                                    (
                                        ((Int64(uj) * rows + rbg + n) * TOP_K + sl)
                                        * HIDDEN
                                    )
                                    + K_CTA * crank
                                    + row
                                )
                                * 4
                            )
                            for t in cutlass.range_constexpr(FC2_TILES):
                                P.st_global_f32(q + 512 * t, d[4 * t + n])
                else:
                    if cutlass.const_expr(self.shared):
                        for n in cutlass.range_constexpr(REQ_ROWS):
                            if n < mg:
                                s = C.ld_shared_f32(
                                    misc
                                    + M_UNITS
                                    + i * UNIT_BYTES
                                    + U_TOK
                                    + 4 * (REQ_ROWS * g + n)
                                )
                                uj = Int32(self.ufield(misc, i, U_J))
                                q = (
                                    shpartials
                                    + (
                                        (Int64(uj) * rows + rbg + n) * HIDDEN
                                        + K_CTA * crank
                                        + row
                                    )
                                    * 4
                                )
                                for t in cutlass.range_constexpr(FC2_TILES):
                                    P.st_global_f32(
                                        q + 512 * t,
                                        C.fma_rn(s, d[4 * t + n], Float32(0.0)),
                                    )

    @cute.jit
    def epilogue(
        self,
        x,
        ids,
        topw,
        g1a,
        g2a,
        gs1,
        gs2,
        rows,
        mqs,
        rbs,
        shpartials,
        pq,
        ids_fin,
        w_fin,
        g2s,
        base,
        misc,
        tm,
        crank,
        my_sh,
        block,
        warp,
        lane,
        tid,
    ):
        qd = warp & 3
        et = tid - 64
        row = 32 * qd + lane
        qw = warp - 2
        zero = (Uint32(0), Uint32(0), Uint32(0), Uint32(0))
        for i in cutlass.range(et, XSF_BYTES // 16, 128, unroll=1):
            C.st_shared_v4(base + OFF_XSF + 16 * i, zero)
        for i in cutlass.range(et, H_BUFS * H_BUF // 16, 128, unroll=1):
            C.st_shared_v4(base + OFF_H + 16 * i, zero)
        for i in cutlass.range(et, HBF_BYTES // 16, 128, unroll=1):
            C.st_shared_v4(base + OFF_HBF + 16 * i, zero)
        # the zeroing and the quantization's scale-byte stores touch the same 16-byte
        # rows
        self.order_scale_rows()
        self.wait_inputs()
        self.quantize_x(x, gs1, mqs, rbs, base, misc, crank, my_sh, et, lane)
        C.mbarrier_wait_parity(misc + B_TREADY, 0)
        nunits = Int32(C.ld_shared_u32(misc + M_NUNITS))
        nex = Int32(C.ld_shared_u32(misc + M_NEX))
        self.token_data(
            ids,
            topw,
            g1a,
            g2a,
            mqs,
            rbs,
            ids_fin,
            w_fin,
            g2s,
            misc,
            block,
            nunits,
            qw,
            lane,
        )
        cute.arch.barrier(barrier_id=1, number_of_threads=128)
        cute.arch.cluster_wait()
        lane_base = tm + Int32(32 * qd) * 65536
        recv = base + OFF_RECV
        ex = Int32(0)
        for k in cutlass.range(nunits + LOOKAHEAD, unroll=1):
            if k < nunits:
                i = k
                b = i & 1
                gm = Int32(self.ufield(misc, i, U_GM))
                ng = P.popc(gm)
                if et == 0:
                    for gi in cutlass.range(ng, unroll=1):
                        C.mbarrier_arrive_expect_tx(
                            misc + B_PRECV + 8 * ((ex + gi) % RECV_BUFS),
                            3 * 128 * 16,
                            ".relaxed.cluster",
                        )
                C.mbarrier_wait_parity(misc + B_ACC1_FULL + 8 * b, (i >> 1) & 1)
                T.tc_fence_after()
                va = T.tmem_load4_masked(lane_base + T_ACC1 + N * b, gm)
                T.tc_fence_before()
                cute.arch.sync_warp()
                if lane == 0:
                    C.mbarrier_arrive(misc + B_ACC1_EMPTY + 8 * b)
                g1 = C.ld_shared_f32(misc + M_UNITS + i * UNIT_BYTES + U_G1)
                ush = self.ufield(misc, i, U_SH)
                gi = Int32(0)
                for g in cutlass.range_constexpr(GROUPS):
                    if ((gm >> g) & 1) != 0:
                        mg = self.rows_of(mqs, g)
                        v0, v1, v2, v3 = va[g]
                        e = ex + gi
                        p = e % RECV_BUFS
                        use = e // RECV_BUFS
                        gi = gi + 1
                        if use >= 1:
                            C.mbarrier_wait_parity(
                                misc + B_PFREE + 8 * p,
                                (use - 1) & 1,
                                ".acquire.cluster",
                            )
                        lbar = misc + B_PRECV + 8 * p
                        for pr in cutlass.range_constexpr(CLUSTER):
                            if crank != pr:
                                # our slot in peer pr's buffer: our rank, skipping the
                                # peer's own
                                sslot = crank
                                if crank > pr:
                                    sslot = crank - 1
                                laddr = recv + ((p * 3 + sslot) * 128 + row) * 16
                                P.st_async_v4_f32(
                                    C.mapa_shared_cluster(laddr, pr),
                                    C.mapa_shared_cluster(lbar, pr),
                                    v0,
                                    v1,
                                    v2,
                                    v3,
                                )
                        C.mbarrier_wait_parity(
                            misc + B_PRECV + 8 * p, use & 1, ".acquire.cluster"
                        )
                        acc = [Float32(0.0), Float32(0.0), Float32(0.0), Float32(0.0)]
                        for src in cutlass.range_constexpr(CLUSTER):
                            p0, p1, p2, p3 = v0, v1, v2, v3
                            if crank != src:
                                rslot = Int32(src)
                                if src > crank:
                                    rslot = Int32(src - 1)
                                p0, p1, p2, p3 = P.ld_shared_v4_f32(
                                    recv + ((p * 3 + rslot) * 128 + row) * 16
                                )
                            acc = [
                                P.add_f32(acc[0], p0),
                                P.add_f32(acc[1], p1),
                                P.add_f32(acc[2], p2),
                                P.add_f32(acc[3], p3),
                            ]
                        cute.arch.sync_warp()
                        # bar.sync (PTX 8.9.4): all lanes' reads precede the release
                        cute.arch.barrier(barrier_id=3 + qd, number_of_threads=32)
                        if (lane == 0) & (e + RECV_BUFS < nex):
                            C.mbarrier_arrive(misc + B_CONSUMED + 8 * p)
                        if ush != Uint32(0):
                            self.shared_h(
                                acc, mg, REQ_ROWS * g, base + OFF_HBF, qd, lane
                            )
                        else:
                            self.routed_h(
                                acc,
                                g1,
                                gs2,
                                mg,
                                REQ_ROWS * g,
                                base + OFF_H + H_BUF * (i % H_BUFS),
                                qd,
                                lane,
                            )
                ex = ex + ng
                C.fence_proxy_async_shared_cta()
                cute.arch.sync_warp()
                # bar.sync (PTX 8.9.4): all lanes' stores precede the release
                cute.arch.barrier(barrier_id=3 + qd, number_of_threads=32)
                if lane == 0:
                    C.mbarrier_arrive(misc + B_HREADY + 8 * self.hready_slot(i))
            if (k >= LOOKAHEAD) & (k - LOOKAHEAD < nunits):
                self.readout(
                    k - LOOKAHEAD,
                    rows,
                    mqs,
                    rbs,
                    shpartials,
                    pq,
                    misc,
                    crank,
                    lane_base,
                    row,
                    lane,
                )

    # ------------------------------------------------------------ kernel
    @cute.kernel
    def kernel(
        self,
        tmap: cutlass.GridConstant[TensorMap],
        x: Int64,
        ids: Int64,
        topw: Int64,
        shpack: Int64,
        g1a: Int64,
        g2a: Int64,
        gs1: Float32,
        gs2: Float32,
        rows: Int32,
        mqs: Int32,
        err: Int64,
        sched: Int64,
        shpartials: Int64,
        pq: Int64,
        ids_fin: Int64,
        w_fin: Int64,
        g2s: Int64,
        sf13: Int64,
        fc2pack: Int64,
        nclusters: Int32,
    ):
        tid, _, _ = cute.arch.thread_idx()
        block, _, _ = cute.arch.block_idx()
        warp = tid // 32
        lane = tid % 32
        sm = SmemAllocator().allocate_tensor(
            cutlass.Uint8, cute.make_layout(SMEM), 1024
        )
        base = Int32(sm.iterator.toint())
        misc = base + OFF_MISC
        crank = Int32(cute.arch.block_idx_in_cluster())
        cl = block // CLUSTER
        # the kernel declares no cluster shape (the launch sets it): refuse any
        # other shape
        P.trap_if(Int32(cute.arch.cluster_size() != CLUSTER))
        C.pdl_launch_dependents()
        if tid == 0:
            for s in cutlass.range_constexpr(RING):
                C.mbarrier_init(misc + B_FULL + 8 * s, 1)
                C.mbarrier_init(misc + B_EMPTY + 8 * s, 1)
            C.mbarrier_init(misc + B_XREADY, 4)
            C.mbarrier_init(misc + B_XBAR, 1)
            # one arrive per lane of the table-building warp: each lane publishes its
            # own unit-record stores (PTX memory model; no reliance on the warp barrier)
            C.mbarrier_init(misc + B_TREADY, 32)
            C.mbarrier_init(misc + B_SBAR, 1)
            for b in cutlass.range_constexpr(2):
                C.mbarrier_init(misc + B_ACC1_FULL + 8 * b, 1)
                C.mbarrier_init(misc + B_ACC1_EMPTY + 8 * b, 4)
            C.mbarrier_init(misc + B_ACC2_FULL, 1)
            C.mbarrier_init(misc + B_ACC2_EMPTY, 4)
            for s in cutlass.range_constexpr(RING):
                C.mbarrier_init(misc + B_SFA_FULL + 8 * s, 1)
            C.mbarrier_init(misc + B_SFX_READY, 1)
            for b in cutlass.range_constexpr(H_BUFS):
                C.mbarrier_init(misc + B_HREADY + 8 * b, 4)
            for p in cutlass.range_constexpr(RECV_BUFS):
                C.mbarrier_init(misc + B_PRECV + 8 * p, 1)
                C.mbarrier_init(misc + B_PFREE + 8 * p, 3)
                C.mbarrier_init(misc + B_CONSUMED + 8 * p, 4)
            C.fence_mbarrier_init_release_cluster()
        if warp == 1:
            T.tmem_alloc(misc + M_TMEM, 512)
        T.tc_fence_before()
        cute.arch.sync_threads()
        T.tc_fence_after()
        cute.arch.cluster_arrive_relaxed(aligned=True)
        # TMEM base, read by every thread once the allocation is visible
        tm = Int32(C.ld_shared_u32(misc + M_TMEM))
        # per-request rows (byte q of mqs) and first global rows (byte q of rbs = sum of
        # the earlier rows)
        rbs = (mqs << 8) * 0x01010101
        gall = Int32(0)
        for q in cutlass.range_constexpr(GROUPS):
            if self.rows_of(mqs, q) > 0:
                gall = gall | (1 << q)
        my_sh = Int32(0)
        nring = Int32(RING)
        if cutlass.const_expr(self.shared):  # noqa: SIM102
            if (nclusters >= 2 * SH_CLUSTERS) & (cl < SH_CLUSTERS):
                my_sh = Int32(1)
                nring = Int32(RING_SH)
        if warp == 0:
            self.producer(
                tmap.get_ptr().toint(),
                ids,
                shpack,
                mqs,
                rbs,
                gall,
                err,
                sched,
                sf13,
                fc2pack,
                base,
                misc,
                crank,
                cl,
                my_sh,
                nring,
                lane,
            )
        elif warp == 1:
            self.mma_warp(base, misc, nring)
        elif warp == SF_WARP:
            self.sf_warp(base, misc, nring)
        elif warp == RELEASE_WARP:
            self.release_warp(misc, crank, lane)
        else:
            self.epilogue(
                x,
                ids,
                topw,
                g1a,
                g2a,
                gs1,
                gs2,
                rows,
                mqs,
                rbs,
                shpartials,
                pq,
                ids_fin,
                w_fin,
                g2s,
                base,
                misc,
                tm,
                crank,
                my_sh,
                block,
                warp,
                lane,
                tid,
            )
        if (warp < 2) | (warp == SF_WARP):
            cute.arch.cluster_wait()
        T.tc_fence_before()
        cute.arch.sync_threads()
        T.tc_fence_after()
        if warp == 1:
            T.tmem_free(tm, 512)
        cute.arch.cluster_arrive(aligned=True)
        cute.arch.cluster_wait()
