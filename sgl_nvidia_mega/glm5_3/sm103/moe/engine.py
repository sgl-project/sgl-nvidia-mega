# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Small-M NVFP4 routed-expert engine (W4A4, tcgen05 ``kind::mxf4nvf4`` block16) with
the BF16 shared expert in-engine.

Modes: ``g1`` one request (M <= 4 rows); ``g2`` two requests of 1..4 rows each (M <= 8).

Work unit = (expert e, quarter j): gate and up rows ``64 j .. 64 j + 63`` of FC1 (128
MMA rows -> 64 intermediate values h) and the matching 64 K columns of FC2 (all 6144
output rows). One cluster of four CTAs runs a unit; CTA rank c owns K range ``1536 c ..
+1536`` of FC1 and output rows ``1536 c .. +1536`` of FC2.

* FC1: 3 ring chunks per unit, each a 3D tensor-map box of the engine's processed w13
  (two 128 x 128 B slots, 128-byte swizzle, rows in the processing's in-block order) and
  4 KiB of its block scales; 24 MMAs 128 x 8 x 64 with A = weights, B = the tokens.
* K-split reduction: every CTA pushes its 128 x 4 partial to the three peers through
  distributed shared memory and sums the four partials in rank order, so all four hold
  the same sums; then g1 alpha, SiLU(gate) * up, NVFP4 quantization of h (global scale
  ``1 / a2``) into the FC2 operand image.
* FC2: 12 tiles of 128 x 64 from compact records, one MMA each; every routed unit stores
  its raw FC2 partial into the token's top-k slot of ``partials``; the shared quarter
  goes to ``shared_partials``.
* Pipeline: stream / MMA / epilogue order FC1(0), FC1(1), FC2(0), FC1(2), FC2(1), ... so
  the h chain of unit k overlaps the FC1 weight stream of unit k + 1.

Mode g2: image rows 0..3 hold request A, rows 4..7 request B (MMA N stays 8). The expert
list is the union of both requests' experts, each entry with a group mask (bit 0: used
by A, bit 1: by B); per unit and group one exchange and one h chain, so an expert both
requests use streams once. A release warp returns receive buffers to the peers, the FC2
accumulator is double-buffered, unit 0's first copies issue as soon as the union size is
known, and the shared quarter's remaining chunks are prefetched into L2 before the
dependency wait.

Warp roles: 0 expert dedup + producer, 1 MMA issue + TMEM owner, 6 scale-factor copies,
2..5 epilogue (TMEM lane quadrant ``warp % 4``), 7 release (g2).

Outputs (rows n < M by global row): ``partials`` [4][M][8][6144] FP32,
``shared_partials`` [4][M][6144] FP32, ``ids`` / ``weights`` / ``scales`` [M][8]
(written by CTA 0: the slot's expert id or -1, its routing weight, its FC2 alpha).
``moe_finalize`` reads exactly this."""

from __future__ import annotations

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
from cutlass import Float32, Int32, Int64, Uint8, Uint32, Uint64
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

# --------------------------------------------------------- geometry shared by the modes
CLUSTER = 4
K_CTA = HIDDEN // CLUSTER  # 1536
SLOT = 18432  # 16 KiB weights + 2 KiB scales
BIG = 2 * SLOT  # one ring chunk
RING = 4
LOOKAHEAD = 2  # FC2(k) issues after FC1(k + 2)
H_BUFS = LOOKAHEAD + 1
FC2_TILES = 12
SH_CHUNK = 32768
SH_FC1_CHUNKS = 12
SH_CHUNKS = 18
SH_CLUSTERS = 4
SH_RECORD = SH_CHUNKS * SH_CHUNK
FC2_RECORD = 3 * SLOT
SF13_PER_EXPERT = 512 * 384
REQ_ROWS = 4  # image rows per request
SF_WARP = 6
RELEASE_WARP = 7
ERR_BAD_ID = 1
ERR_UNIT_CAP = 2

# shared memory (offsets from the 1024-aligned base)
OFF_RING = 0
OFF_XBF = OFF_RING + RING * BIG  # BF16 x image [192 K cores][8 rows][16 B]
OFF_X = OFF_XBF + 24576  # NVFP4 x image [48 K cores][8 rows][16 B]
OFF_XSF = OFF_X + 6144  # x scale atoms, 24 x 128 B + 384
OFF_RECV = OFF_XSF + 3456  # receive buffers (per mode)
H_BUF = 768  # h image 256 B + its scale atom 512 B

# tensor memory columns
T_ACC1 = 0  # 2 x 8
T_ACC2 = 16  # 12 x 8
T_ACC2_B = 384  # second FC2 accumulator (g2)
T_SFB_X = 128  # 24 x 4
T_SFB_H = 224  # H_BUFS x 4
T_SFA = 256  # RING x 32

IDESC_NVF4 = T.idesc_nvf4(128, 8)
IDESC_BF16 = T.idesc_bf16(128, 8)

# burst tables: (d column, A byte offset, B byte offset, SFA column, SFB column,
# accumulate, group)
FC1_MMAS = tuple(
    (
        0,
        16384 * h2 + 32 * kk,
        256 * (4 * h2 + kk),
        16 * h2 + 4 * kk,
        4 * (4 * h2 + kk),
        "first" if h2 == kk == 0 else 1,
        0,
    )
    for h2 in range(2)
    for kk in range(4)
)
FC2_MMAS = tuple(
    (8 * (4 * h2 + t), SLOT * h2 + 4096 * t, 0, 16 * h2 + 4 * t, 0, 0, h2)
    for h2 in range(2)
    for t in range(4)
)
SH_FC1_MMAS = tuple(
    (
        0,
        16384 * h2 + 256 * kk,
        256 * (4 * h2 + kk),
        0,
        0,
        "first" if h2 == kk == 0 else 1,
        0,
    )
    for h2 in range(2)
    for kk in range(4)
)
SH_FC2_MMAS = tuple(
    (8 * h2, 16384 * h2 + 256 * kk, 256 * kk, 0, 0, 0 if kk == 0 else 1, 0)
    for h2 in range(2)
    for kk in range(4)
)
# scale-factor copies: (TMEM column, source byte offset, group)
FC1_CPS = tuple(
    (16 * h2 + 4 * kk, 2048 * h2 + 512 * kk, 0) for h2 in range(2) for kk in range(4)
)
FC2_CPS = tuple(
    (16 * h2 + 4 * kk, SLOT * h2 + 512 * kk, h2) for h2 in range(2) for kk in range(4)
)
X_CPS = tuple((4 * q, 128 * q, 0) for q in range(4))
H_CPS = ((0, 0, 0),)


def _align16(v: int) -> int:
    return (v + 15) // 16 * 16


class _Geometry:
    """Per-mode constants: thread count, receive buffers, unit table, schedule rows,
    shared memory map (each total leaves room for one CTA per SM)."""

    def __init__(self, mode: str):
        if mode not in ("g1", "g2"):
            raise ValueError(f"unknown mode {mode!r}")
        two = mode == "g2"
        self.two = two
        self.threads = 256 if two else 224
        self.recv_bufs = 5 if two else 4
        self.recv_slots = (
            3 if two else 4
        )  # g2: three peer slots per buffer; g1: one slot per rank
        self.recv_buf = self.recv_slots * 128 * 16
        self.off_h = OFF_RECV + self.recv_bufs * self.recv_buf
        self.off_hbf = self.off_h + H_BUFS * H_BUF
        self.off_misc = self.off_hbf + 1024
        self.acc2_bufs = 2 if two else 1
        self.max_units = 16 if two else 32
        self.max_list = 64 if two else 32
        self.sched_row = 16 if two else 34
        self.sched_bytes = 2080 if two else 2256
        self.rows = 2 * REQ_ROWS if two else REQ_ROWS
        self.pfree_count = 3 if two else 12
        self.smem = 229968 if two else 227440
        # barriers
        self.b_full = 0
        self.b_empty = 32
        self.b_xready = 64
        self.b_tready = 72
        self.b_sbar = 80
        self.b_acc1_full = 88
        self.b_acc1_empty = 104
        self.b_acc2_full = 120
        self.b_acc2_empty = self.b_acc2_full + 8 * self.acc2_bufs
        self.b_hready = self.b_acc2_empty + 8 * self.acc2_bufs
        self.b_sfa_full = self.b_hready + 8 * H_BUFS
        self.b_sfx_ready = self.b_sfa_full + 8 * RING
        self.b_precv = self.b_sfx_ready + 8
        self.b_pfree = self.b_precv + 8 * self.recv_bufs
        self.b_consumed = self.b_pfree + 8 * self.recv_bufs
        end = self.b_consumed + (8 * self.recv_bufs if two else 0)
        self.m_tmem = end
        self.m_nunits = end + 4
        self.m_nex = end + 8
        self.m_elist = _align16(end + 12)
        self.m_egm = self.m_elist + 4 * self.max_list
        self.m_stab = _align16(self.m_egm + 4 * self.max_list)
        self.m_units = _align16(self.m_stab + self.sched_bytes)
        # unit record: expert, quarter, shared flag, group mask, g1 alpha, token scale
        # [rows], slot [rows]
        self.u_e, self.u_j, self.u_sh, self.u_gm, self.u_g1 = 0, 4, 8, 12, 16
        self.u_tok = 20
        self.u_slot = 20 + 4 * self.rows
        self.unit_bytes = 20 + 8 * self.rows
        used = self.off_misc + self.m_units + self.max_units * self.unit_bytes + 1024
        assert used <= self.smem, (mode, used, self.smem)


class Nvfp4MoeSmallM:
    """``mode``: "g1" or "g2"; ``shared``: run the shared expert's four quarters
    in-engine (the served form); ``merge`` (g2): one union expert list for both requests
    (the served form; False lists A's experts then B's, test-only).

    Call: ``(w13, x, ids, weights, shared_pack, g1_alpha, g2_alpha, gs1, gs2, rows,
    split, err, schedule, shared_partials, partials, slot_ids, slot_weights,
    slot_scales, w13_scales, fc2_records, nclusters, stream)``; addresses are device
    pointers, ``gs1`` / ``gs2`` the FC1 / FC2 activation global scales (1 / input
    scale), ``split`` the rows of the first request (g2; ignored by g1)."""

    def __init__(self, mode: str = "g1", shared: bool = True, merge: bool = True):
        self.mode = mode
        self.g = _Geometry(mode)
        self.shared = shared
        self.merge = merge

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
        split: Int32,
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
        # processed w13 viewed as {128 B, E * 512 rows, 24 column blocks}: one box = one
        # FC1 chunk (two 128 x 128 slots)
        tmap = create_tensor_map_tiled(
            w13,
            Uint8,
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
            split,
            err,
            sched,
            shpartials,
            pq,
            ids_fin,
            w_fin,
            g2s,
            sf13,
            fc2pack,
        ).launch(
            grid=[CLUSTER * nclusters, 1, 1],
            block=[self.g.threads, 1, 1],
            cluster=[CLUSTER, 1, 1],
            max_number_threads=[self.g.threads, 1, 1],
            min_blocks_per_mp=1,
            stream=stream,
            use_pdl=True,
        )

    # ------------------------------------------------------------ protocol hooks
    # Each step of the inter-warp / inter-CTA protocol is one method here.
    @cute.jit
    def wait_schedule(self, misc):
        """The staged schedule table is complete."""
        C.mbarrier_wait_parity(misc + self.g.b_sbar, 0)

    @cute.jit
    def wait_inputs(self):
        """The epilogue's activations are written by the preceding kernel (dependent
        launch)."""
        C.pdl_wait()

    @cute.jit
    def wait_scales(self, bar, parity):
        """The scale-factor warp's copies into TMEM are complete."""
        C.mbarrier_wait_parity(bar, parity)

    @cute.jit
    def hready_slot(self, i):
        return i % H_BUFS

    @cute.jit
    def hready_phase(self, i):
        return (i // H_BUFS) & 1

    @cute.jit
    def ring_tail(self, misc, it):
        """Producer tail: the MMA warp's last commit on every ring slot has arrived."""
        first = it - RING
        if first < 0:
            first = Int32(0)
        for c in cutlass.range(first, it, unroll=1):
            C.mbarrier_wait_parity(
                misc + self.g.b_empty + 8 * (c % RING), (c // RING) & 1
            )

    # ------------------------------------------------------------ helpers
    @cute.jit
    def unit_field(self, misc, i, field: cutlass.Constexpr[int]):
        return C.ld_shared_u32(misc + self.g.m_units + i * self.g.unit_bytes + field)

    @cute.jit
    def fc1_chunk(self, tmap_ptr, sf13, dst, expert, quarter, crank, c, bar, policy):
        """FC1 chunk c of unit (expert, quarter) on rank crank: one tensor-map box (two
        slots) and the two slots' 4 KiB of scales."""
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
    def pre_issue(self, shpack, base, misc, crank, cl, my_sh, lane, policy):
        """Shared clusters: the shared quarter's first ring chunks (static weights)
        before the dependency wait; g2 also prefetches the remaining chunks into L2 (one
        per lane)."""
        pre = Int32(0)
        if cutlass.const_expr(self.shared):  # noqa: SIM102
            if my_sh != 0:
                src = shpack + (Int64(cl) * 4 + Int64(crank)) * SH_RECORD
                if lane == 0:
                    for c in cutlass.range_constexpr(RING):
                        C.mbarrier_arrive_expect_tx(
                            misc + self.g.b_full + 8 * c, SH_CHUNK
                        )
                        C.bulk_copy_g2s(
                            base + OFF_RING + c * BIG,
                            src + c * SH_CHUNK,
                            SH_CHUNK,
                            misc + self.g.b_full + 8 * c,
                            policy,
                        )
                pre = Int32(RING)
                if cutlass.const_expr(self.g.two):
                    c = RING + lane
                    if c < SH_CHUNKS:
                        C.bulk_prefetch_l2(src + Int64(c) * SH_CHUNK, SH_CHUNK)
        return pre

    @cute.jit
    def write_elist(
        self, misc, bits, k, lane, word_a, word_b, group: cutlass.Constexpr[int]
    ):
        """Set bits of ``bits`` (this lane's 32 experts) into the expert list from entry
        k, with their group masks (``group`` 0: from the two request words; 1 / 2: that
        value)."""
        while bits != Uint32(0):
            b = P.ctz(bits)
            if k < self.g.max_list:
                C.st_shared_u32(misc + self.g.m_elist + 4 * k, 32 * lane + b)
                if cutlass.const_expr(self.g.two):
                    gm = Uint32(group)
                    if cutlass.const_expr(group == 0):
                        gm = ((word_a >> Uint32(b)) & Uint32(1)) | (
                            ((word_b >> Uint32(b)) & Uint32(1)) << Uint32(1)
                        )
                    C.st_shared_u32(misc + self.g.m_egm + 4 * k, gm)
            k = k + 1
            bits = bits & (bits - Uint32(1))

    @cute.jit
    def scan8(self, cnt, lane):
        """Inclusive prefix sum of ``cnt`` over lanes 0..7."""
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
        pre,
        pre0,
        policy,
    ):
        """Producer lane: stream order FC1(0), FC1(1), FC2(0), FC1(2), FC2(1), ..., then
        wait for the MMA warp's last commit on every ring slot."""
        it = pre + pre0
        for kk in cutlass.range(nunits + LOOKAHEAD, unroll=1):
            for part in cutlass.range_constexpr(2):
                i = kk if part == 0 else kk - LOOKAHEAD
                if (i >= 0) & (i < nunits):
                    ush = self.unit_field(misc, i, self.g.u_sh)
                    if ush != Uint32(0):
                        uj = Int32(self.unit_field(misc, i, self.g.u_j))
                        src = shpack + (Int64(uj) * 4 + Int64(crank)) * SH_RECORD
                        c0 = Int32(SH_FC1_CHUNKS)
                        c1 = Int32(SH_CHUNKS)
                        if cutlass.const_expr(part == 0):
                            c0 = Int32(0)
                            if i == 0:
                                c0 = pre
                            c1 = Int32(SH_FC1_CHUNKS)
                        for c in cutlass.range(c0, c1, unroll=1):
                            s = it % RING
                            if it >= RING:
                                C.mbarrier_wait_parity(
                                    misc + self.g.b_empty + 8 * s,
                                    ((it // RING) - 1) & 1,
                                )
                            C.mbarrier_arrive_expect_tx(
                                misc + self.g.b_full + 8 * s, SH_CHUNK
                            )
                            C.bulk_copy_g2s(
                                base + OFF_RING + s * BIG,
                                src + Int64(c) * SH_CHUNK,
                                SH_CHUNK,
                                misc + self.g.b_full + 8 * s,
                                policy,
                            )
                            it = it + 1
                    else:
                        u_e_idx = Int32(self.unit_field(misc, i, self.g.u_e))
                        uj = Int32(self.unit_field(misc, i, self.g.u_j))
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
                            s = it % RING
                            if it >= RING:
                                C.mbarrier_wait_parity(
                                    misc + self.g.b_empty + 8 * s,
                                    ((it // RING) - 1) & 1,
                                )
                            nbytes = Int32(BIG)
                            if c == 4:
                                nbytes = Int32(SLOT)
                            C.mbarrier_arrive_expect_tx(
                                misc + self.g.b_full + 8 * s, nbytes
                            )
                            if c < 3:
                                # expert / quarter are read from the unit table again
                                # for each FC1 chunk
                                ce = Int32(self.unit_field(misc, i, self.g.u_e))
                                cj = Int32(self.unit_field(misc, i, self.g.u_j))
                                self.fc1_chunk(
                                    tmap_ptr,
                                    sf13,
                                    base + OFF_RING + s * BIG,
                                    ce,
                                    cj,
                                    crank,
                                    c,
                                    misc + self.g.b_full + 8 * s,
                                    policy,
                                )
                            else:
                                C.bulk_copy_g2s(
                                    base + OFF_RING + s * BIG,
                                    src2 + Int64(2 * c - 6) * SLOT,
                                    nbytes,
                                    misc + self.g.b_full + 8 * s,
                                    policy,
                                )
                            it = it + 1
        self.ring_tail(misc, it)

    # ------------------------------------------------ warp 0: dedup, schedule, producer
    @cute.jit
    def producer_g1(
        self,
        tmap_ptr,
        ids,
        shpack,
        rows,
        err,
        sched,
        sf13,
        fc2pack,
        base,
        misc,
        crank,
        cl,
        my_sh,
        lane,
    ):
        g = self.g
        policy = C.l2_policy("evict_first")
        pre = self.pre_issue(shpack, base, misc, crank, cl, my_sh, lane, policy)
        if lane == 0:
            C.mbarrier_arrive_expect_tx(misc + g.b_sbar, g.sched_bytes)
            C.bulk_copy_g2s(
                misc + g.m_stab,
                sched + Int64(cl) * g.sched_bytes,
                g.sched_bytes,
                misc + g.b_sbar,
                C.l2_policy("evict_last"),
            )
        C.pdl_wait()
        nid = rows * TOP_K
        e = Int32(0x7FFFFFFF)
        if lane < nid:
            e = P.ld_cg_s32(ids + Int64(lane) * 4)
        valid = Int32(0)
        if (lane < nid) & (Uint32(e) < Uint32(NUM_EXPERTS)):
            valid = Int32(1)
        if (lane < nid) & (valid == 0):
            P.atom_or_global(err, ERR_BAD_ID)
        if valid == 0:
            e = Int32(0x7FFFFFFF)
        # 256-bit set of the distinct experts: lane w < 8 holds word w
        word = Uint32(0)
        for w in cutlass.range_constexpr(NUM_EXPERTS // 32):
            bit = Uint32(0)
            if (valid != 0) & ((e >> 5) == w):
                bit = Uint32(1) << Uint32(e & 31)
            ow = cute.arch.warp_redux_sync(bit, "or")
            if lane == w:
                word = ow
        cnt = P.popc(word)
        incl = self.scan8(cnt, lane)
        d_raw = cute.arch.shuffle_sync(incl, 7)
        if (lane == 0) & (d_raw > TOP_K * rows):
            P.atom_or_global(err, ERR_BAD_ID)
        d = min(d_raw, TOP_K * rows)
        self.write_elist(misc, word, incl - cnt, lane, word, word, 1)
        cute.arch.sync_warp()
        self.wait_schedule(misc)
        row = misc + g.m_stab + 2 * g.sched_row * d
        nunits = P.ld_shared_s16(row)
        if lane < nunits:
            code = P.ld_shared_s16(row + 2 + 2 * lane)
            unit = misc + g.m_units + lane * g.unit_bytes
            if code < 0:
                C.st_shared_u32(unit + g.u_e, Int32(-1))
                C.st_shared_u32(unit + g.u_j, cl)
                C.st_shared_u32(unit + g.u_sh, Int32(1))
            else:
                C.st_shared_u32(
                    unit + g.u_e, C.ld_shared_u32(misc + g.m_elist + 4 * (code >> 2))
                )
                C.st_shared_u32(unit + g.u_j, code & 3)
                C.st_shared_u32(unit + g.u_sh, Int32(0))
        cute.arch.sync_warp()
        pre0 = Int32(0)
        if lane == 0:
            if nunits > 0:  # noqa: SIM102
                if self.unit_field(misc, 0, g.u_sh) == Uint32(0):
                    C.mbarrier_arrive_expect_tx(misc + g.b_full, BIG)
                    e0 = Int32(self.unit_field(misc, 0, g.u_e))
                    j0 = Int32(self.unit_field(misc, 0, g.u_j))
                    self.fc1_chunk(
                        tmap_ptr,
                        sf13,
                        base + OFF_RING,
                        e0,
                        j0,
                        crank,
                        0,
                        misc + g.b_full,
                        policy,
                    )
                    pre0 = Int32(1)
            C.st_shared_u32(misc + g.m_nunits, nunits)
        # Every lane arrives (count 32): each lane's own unit-record stores are
        # released by its own arrive (see the init of b_tready).
        C.mbarrier_arrive(misc + g.b_tready)
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
                pre,
                pre0,
                policy,
            )
        cute.arch.sync_warp()

    @cute.jit
    def producer_g2(
        self,
        tmap_ptr,
        ids,
        shpack,
        rows,
        ma,
        err,
        sched,
        sf13,
        fc2pack,
        base,
        misc,
        crank,
        cl,
        my_sh,
        lane,
    ):
        g = self.g
        mb = rows - ma
        policy = C.l2_policy("evict_first")
        pre = self.pre_issue(shpack, base, misc, crank, cl, my_sh, lane, policy)
        if lane == 0:
            C.mbarrier_arrive_expect_tx(misc + g.b_sbar, g.sched_bytes)
            C.bulk_copy_g2s(
                misc + g.m_stab,
                sched + Int64(cl) * g.sched_bytes,
                g.sched_bytes,
                misc + g.b_sbar,
                C.l2_policy("evict_last"),
            )
        C.pdl_wait()
        # lane l holds (token l >> 3, slot l & 7) of request A and of request B
        ea = Int32(0x7FFFFFFF)
        eb = Int32(0x7FFFFFFF)
        if lane < TOP_K * ma:
            ea = P.ld_cg_s32(ids + Int64(lane) * 4)
        if lane < TOP_K * mb:
            eb = P.ld_cg_s32(ids + Int64(TOP_K * ma + lane) * 4)
        va = Int32(0)
        vb = Int32(0)
        if (lane < TOP_K * ma) & (Uint32(ea) < Uint32(NUM_EXPERTS)):
            va = Int32(1)
        if (lane < TOP_K * mb) & (Uint32(eb) < Uint32(NUM_EXPERTS)):
            vb = Int32(1)
        if ((lane < TOP_K * ma) & (va == 0)) | ((lane < TOP_K * mb) & (vb == 0)):
            P.atom_or_global(err, ERR_BAD_ID)
        if va == 0:
            ea = Int32(0x7FFFFFFF)
        if vb == 0:
            eb = Int32(0x7FFFFFFF)
        wa = Uint32(0)
        wb = Uint32(0)
        for w in cutlass.range_constexpr(NUM_EXPERTS // 32):
            ba = Uint32(0)
            bb = Uint32(0)
            if (va != 0) & ((ea >> 5) == w):
                ba = Uint32(1) << Uint32(ea & 31)
            if (vb != 0) & ((eb >> 5) == w):
                bb = Uint32(1) << Uint32(eb & 31)
            oa = cute.arch.warp_redux_sync(ba, "or")
            ob = cute.arch.warp_redux_sync(bb, "or")
            if lane == w:
                wa = oa
                wb = ob
        w0 = wa
        if cutlass.const_expr(self.merge):
            w0 = wa | wb
        cnt = P.popc(w0)
        incl = self.scan8(cnt, lane)
        d0 = cute.arch.shuffle_sync(incl, 7)
        pre0 = Int32(0)
        if cutlass.const_expr(self.merge):
            # unit 0 resolved from the schedule row and the union bitmap; its first
            # copies issue now
            self.wait_schedule(misc)
            row0 = misc + g.m_stab + 2 * g.sched_row * min(d0, g.max_list)
            code0 = Int32(-1)
            if P.ld_shared_s16(row0) > 0:
                code0 = P.ld_shared_s16(row0 + 2)
            if code0 >= 0:
                idx = code0 >> 2
                mine = (idx >= incl - cnt) & (idx < incl)
                who = Uint32(cute.arch.vote_ballot_sync(mine))
                e0 = Int32(0)
                if mine:
                    bits = w0
                    for _ in cutlass.range(idx - (incl - cnt), unroll=1):
                        bits = bits & (bits - Uint32(1))
                    e0 = 32 * lane + P.ctz(bits)
                src_lane = Int32(0)
                if who != Uint32(0):
                    src_lane = P.ctz(who)
                e0 = cute.arch.shuffle_sync(e0, src_lane)
                if (lane == 0) & (who != Uint32(0)):
                    C.mbarrier_arrive_expect_tx(misc + g.b_full, BIG)
                    self.fc1_chunk(
                        tmap_ptr,
                        sf13,
                        base + OFF_RING,
                        e0,
                        code0 & 3,
                        crank,
                        0,
                        misc + g.b_full,
                        policy,
                    )
                    pre0 = Int32(1)
                pre0 = cute.arch.shuffle_sync(pre0, 0)
        self.write_elist(misc, w0, incl - cnt, lane, wa, wb, 0 if self.merge else 1)
        d = d0
        if cutlass.const_expr(not self.merge):
            cb = P.popc(wb)
            ib = self.scan8(cb, lane)
            db = cute.arch.shuffle_sync(ib, 7)
            self.write_elist(misc, wb, d0 + ib - cb, lane, wa, wb, 2)
            d = d0 + db
        if (lane == 0) & (d > g.max_list):
            P.atom_or_global(err, ERR_BAD_ID)
        d = min(d, g.max_list)
        cute.arch.sync_warp()
        self.wait_schedule(misc)
        row = misc + g.m_stab + 2 * g.sched_row * d
        nraw = P.ld_shared_s16(row)
        nunits = min(nraw, g.max_units)
        if (lane == 0) & (nraw > g.max_units):
            P.atom_or_global(err, ERR_UNIT_CAP)
        gmu = Int32(0)
        if lane < nunits:
            code = P.ld_shared_s16(row + 2 + 2 * lane)
            unit = misc + g.m_units + lane * g.unit_bytes
            if code < 0:
                C.st_shared_u32(unit + g.u_e, Int32(-1))
                C.st_shared_u32(unit + g.u_j, cl)
                C.st_shared_u32(unit + g.u_sh, Int32(1))
                gmu = Int32(1)
                if mb > 0:
                    gmu = Int32(3)
            else:
                C.st_shared_u32(
                    unit + g.u_e, C.ld_shared_u32(misc + g.m_elist + 4 * (code >> 2))
                )
                C.st_shared_u32(unit + g.u_j, code & 3)
                C.st_shared_u32(unit + g.u_sh, Int32(0))
                gmu = Int32(C.ld_shared_u32(misc + g.m_egm + 4 * (code >> 2)))
            C.st_shared_u32(unit + g.u_gm, gmu)
        # exchanges of this cluster (one per (unit, group)), the same in its four CTAs
        nx = P.popc(gmu)
        for o in (16, 8, 4, 2, 1):
            nx = nx + cute.arch.shuffle_sync_bfly(nx, o)
        if lane == 0:
            C.st_shared_u32(misc + g.m_nex, nx)
        cute.arch.sync_warp()
        if lane == 0:
            if (pre0 == 0) & (nunits > 0):  # noqa: SIM102
                if self.unit_field(misc, 0, g.u_sh) == Uint32(0):
                    C.mbarrier_arrive_expect_tx(misc + g.b_full, BIG)
                    e0 = Int32(self.unit_field(misc, 0, g.u_e))
                    j0 = Int32(self.unit_field(misc, 0, g.u_j))
                    self.fc1_chunk(
                        tmap_ptr,
                        sf13,
                        base + OFF_RING,
                        e0,
                        j0,
                        crank,
                        0,
                        misc + g.b_full,
                        policy,
                    )
                    pre0 = Int32(1)
            C.st_shared_u32(misc + g.m_nunits, nunits)
        # Every lane arrives (count 32): each lane's own unit-record stores are
        # released by its own arrive (see the init of b_tready).
        C.mbarrier_arrive(misc + g.b_tready)
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
                pre,
                pre0,
                policy,
            )
        cute.arch.sync_warp()

    # ------------------------------------------------------------ warp 1: MMA issue
    @cute.jit
    def mma_warp(self, base, misc):
        g = self.g
        lead = T.elect_leader()
        C.mbarrier_wait_parity(misc + g.b_tready, 0)
        nunits = cute.arch.shuffle_sync(Int32(C.ld_shared_u32(misc + g.m_nunits)), 0)
        tm = cute.arch.shuffle_sync(Int32(C.ld_shared_u32(misc + g.m_tmem)), 0)
        if nunits > 0:
            C.mbarrier_wait_parity(misc + g.b_xready, 0)
            self.wait_scales(misc + g.b_sfx_ready, 0)
            T.tc_fence_after()
        s_ring = base + OFF_RING
        it = Int32(0)
        for k in cutlass.range(nunits + LOOKAHEAD, unroll=1):
            shk = Int32(0)
            if k < nunits:
                shk = cute.arch.shuffle_sync(Int32(self.unit_field(misc, k, g.u_sh)), 0)
            if k < nunits:
                b = k & 1
                if k >= 2:
                    C.mbarrier_wait_parity(
                        misc + g.b_acc1_empty + 8 * b, ((k >> 1) - 1) & 1
                    )
                    T.tc_fence_after()
                if shk != 0:
                    for c in cutlass.range(SH_FC1_CHUNKS, unroll=1):
                        s = it % RING
                        C.mbarrier_wait_parity(
                            misc + g.b_full + 8 * s, (it // RING) & 1
                        )
                        self.wait_scales(misc + g.b_sfa_full + 8 * s, (it // RING) & 1)
                        T.tc_fence_after()
                        a0 = T.smem_desc(s_ring + s * BIG, 128, 1024, 0)
                        b0 = T.smem_desc(base + OFF_XBF + 2048 * c, 128, 1024, 0)
                        T.tc_burst(
                            lead,
                            tm + T_ACC1 + 8 * b,
                            a0,
                            b0,
                            0,
                            0,
                            Int32(c != 0),
                            0,
                            misc + g.b_empty + 8 * s,
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
                        s = it % RING
                        C.mbarrier_wait_parity(
                            misc + g.b_full + 8 * s, (it // RING) & 1
                        )
                        self.wait_scales(misc + g.b_sfa_full + 8 * s, (it // RING) & 1)
                        T.tc_fence_after()
                        a0 = T.smem_desc(s_ring + s * BIG, 16, 1024, 2)
                        b0 = T.smem_desc(base + OFF_X + 2048 * c, 128, 1024, 0)
                        T.tc_burst(
                            lead,
                            tm + T_ACC1 + 8 * b,
                            a0,
                            b0,
                            tm + T_SFA + 32 * s,
                            tm + T_SFB_X + 32 * c,
                            Int32(c != 0),
                            0,
                            misc + g.b_empty + 8 * s,
                            0,
                            0,
                            mmas=FC1_MMAS,
                            block_scaled=True,
                            idesc=IDESC_NVF4,
                        )
                        cute.arch.sync_warp()
                        it = it + 1
                T.tc_commit(lead, misc + g.b_acc1_full + 8 * b)
                cute.arch.sync_warp()
            if (k >= LOOKAHEAD) & (k - LOOKAHEAD < nunits):
                i = k - LOOKAHEAD
                shi = cute.arch.shuffle_sync(Int32(self.unit_field(misc, i, g.u_sh)), 0)
                ab = i % g.acc2_bufs
                acc2 = tm + T_ACC2
                if cutlass.const_expr(g.acc2_bufs == 2):  # noqa: SIM102
                    if ab == 1:
                        acc2 = tm + T_ACC2_B
                C.mbarrier_wait_parity(
                    misc + g.b_hready + 8 * self.hready_slot(i), self.hready_phase(i)
                )
                T.tc_fence_after()
                if i >= g.acc2_bufs:
                    C.mbarrier_wait_parity(
                        misc + g.b_acc2_empty + 8 * ab,
                        ((i - g.acc2_bufs) // g.acc2_bufs) & 1,
                    )
                    T.tc_fence_after()
                if shi != 0:
                    bh = T.smem_desc(base + g.off_hbf, 128, 1024, 0)
                    for c in cutlass.range(SH_FC1_CHUNKS, SH_CHUNKS, unroll=1):
                        s = it % RING
                        C.mbarrier_wait_parity(
                            misc + g.b_full + 8 * s, (it // RING) & 1
                        )
                        self.wait_scales(misc + g.b_sfa_full + 8 * s, (it // RING) & 1)
                        T.tc_fence_after()
                        a0 = T.smem_desc(s_ring + s * BIG, 128, 1024, 0)
                        T.tc_burst(
                            lead,
                            acc2 + 16 * (c - SH_FC1_CHUNKS),
                            a0,
                            bh,
                            0,
                            0,
                            0,
                            0,
                            misc + g.b_empty + 8 * s,
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
                    bh = T.smem_desc(base + g.off_h + H_BUF * hb, 128, 256, 0)
                    for c in cutlass.range(3, 5, unroll=1):
                        s = it % RING
                        C.mbarrier_wait_parity(
                            misc + g.b_full + 8 * s, (it // RING) & 1
                        )
                        self.wait_scales(misc + g.b_sfa_full + 8 * s, (it // RING) & 1)
                        T.tc_fence_after()
                        a0 = T.smem_desc(s_ring + s * BIG, 128, 256, 0)
                        T.tc_burst(
                            lead,
                            acc2 + 32 * (2 * c - 6),
                            a0,
                            bh,
                            tm + T_SFA + 32 * s,
                            tm + T_SFB_H + 4 * hb,
                            0,
                            Int32(c == 3),
                            misc + g.b_empty + 8 * s,
                            0,
                            0,
                            mmas=FC2_MMAS,
                            block_scaled=True,
                            idesc=IDESC_NVF4,
                        )
                        cute.arch.sync_warp()
                        it = it + 1
                T.tc_commit(lead, misc + g.b_acc2_full + 8 * ab)
                cute.arch.sync_warp()
        cute.arch.sync_warp()

    # -------------------------------------------------- warp 6: scale factors into TMEM
    @cute.jit
    def sf_warp(self, base, misc):
        g = self.g
        lead = T.elect_leader()
        C.mbarrier_wait_parity(misc + g.b_tready, 0)
        nunits = cute.arch.shuffle_sync(Int32(C.ld_shared_u32(misc + g.m_nunits)), 0)
        tm = cute.arch.shuffle_sync(Int32(C.ld_shared_u32(misc + g.m_tmem)), 0)
        if nunits > 0:
            C.mbarrier_wait_parity(misc + g.b_xready, 0)
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
                    d0 + Uint64(32 * q),
                    cps=X_CPS,
                    commit=False,
                )
            T.tc_commit(lead, misc + g.b_sfx_ready)
            cute.arch.sync_warp()
        it = Int32(0)
        for k in cutlass.range(nunits + LOOKAHEAD, unroll=1):
            for part in cutlass.range_constexpr(2):
                i = k if part == 0 else k - LOOKAHEAD
                if (i >= 0) & (i < nunits):
                    ush = cute.arch.shuffle_sync(
                        Int32(self.unit_field(misc, i, g.u_sh)), 0
                    )
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
                                misc + g.b_hready + 8 * self.hready_slot(i),
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
                                T.smem_desc(
                                    base + g.off_h + H_BUF * hb + 256, 0, 128, 0
                                ),
                                cps=H_CPS,
                                commit=False,
                            )
                            cute.arch.sync_warp()
                    for c in cutlass.range(c0, c1, unroll=1):
                        s = it % RING
                        C.mbarrier_wait_parity(
                            misc + g.b_full + 8 * s, (it // RING) & 1
                        )
                        T.tc_fence_after()
                        if ush != 0:
                            T.tc_commit(lead, misc + g.b_sfa_full + 8 * s)
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
                                    misc + g.b_sfa_full + 8 * s,
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
                                    misc + g.b_sfa_full + 8 * s,
                                    tm + T_SFA + 32 * s,
                                    d,
                                    cps=FC2_CPS,
                                )
                        cute.arch.sync_warp()
                        it = it + 1
        cute.arch.sync_warp()

    # ---------------------------------------------- warp 7 (g2): receive-buffer release
    @cute.jit
    def release_warp(self, misc, crank, lane):
        g = self.g
        C.mbarrier_wait_parity(misc + g.b_tready, 0)
        nex = cute.arch.shuffle_sync(Int32(C.ld_shared_u32(misc + g.m_nex)), 0)
        cute.arch.cluster_wait()
        if lane == 0:
            for e in cutlass.range(nex - g.recv_bufs, unroll=1):
                p = e % g.recv_bufs
                C.mbarrier_wait_parity(
                    misc + g.b_consumed + 8 * p, (e // g.recv_bufs) & 1
                )
                for pr in cutlass.range_constexpr(CLUSTER):
                    if crank != pr:
                        P.mbarrier_arrive_remote(
                            C.mapa_shared_cluster(misc + g.b_pfree + 8 * p, pr)
                        )
        cute.arch.sync_warp()

    # --------------------------------------- warps 2..5: x images, h chain, FC2 readout
    @cute.jit
    def zero_fills(self, base, et):
        zero = (Uint32(0), Uint32(0), Uint32(0), Uint32(0))
        for i in cutlass.range(et, 3456 // 16, 128, unroll=1):
            C.st_shared_v4(base + OFF_XSF + 16 * i, zero)
        for i in cutlass.range(et, H_BUFS * H_BUF // 16, 128, unroll=1):
            C.st_shared_v4(base + self.g.off_h + 16 * i, zero)
        for i in cutlass.range(et, 1024 // 16, 128, unroll=1):
            C.st_shared_v4(base + self.g.off_hbf + 16 * i, zero)

    @cute.jit
    def quantize_x(self, x, gs1, ma, mb, base, misc, crank, et, lane):
        """The activations x (BF16) of this CTA's K range into the BF16 image
        (shared-expert operand) and the NVFP4 image + scale atoms (routed operand).
        Image row n: request A row n (n < 4, valid n < ma) or request B row n - 4
        (valid n - 4 < mb); invalid rows are zero."""
        xbf = base + OFF_XBF
        raw = cute.make_rmem_tensor(cute.make_layout(48), Uint32)
        for i in cutlass.range_constexpr(6):
            b = et + 128 * i
            n = b // 96
            kb = b % 96
            for w in cutlass.range_constexpr(8):
                raw[8 * i + w] = Uint32(0)
            ok = n < ma
            grow = n
            if n >= REQ_ROWS:
                ok = n - REQ_ROWS < mb
                grow = ma + n - REQ_ROWS
            if ok:
                src = x + (Int64(grow) * HIDDEN + K_CTA * crank + 16 * kb) * 2
                lo = P.ld_cg_v4_u32(src)
                hi = P.ld_cg_v4_u32(src + 16)
                for w in cutlass.range_constexpr(4):
                    raw[8 * i + w] = lo[w]
                    raw[8 * i + 4 + w] = hi[w]
        for i in cutlass.range_constexpr(6):
            b = et + 128 * i
            n = b // 96
            kb = b % 96
            C.st_shared_v4(
                xbf + (2 * kb) * 128 + n * 16, tuple(raw[8 * i + w] for w in range(4))
            )
            C.st_shared_v4(
                xbf + (2 * kb + 1) * 128 + n * 16,
                tuple(raw[8 * i + 4 + w] for w in range(4)),
            )
        cute.arch.barrier(barrier_id=1, number_of_threads=128)
        for i in cutlass.range(6, unroll=1):
            b = et + 128 * i
            n = b // 96
            kb = b % 96
            w0 = Uint32(0)
            w1 = Uint32(0)
            sfb = Uint32(0)
            ok = n < ma
            if n >= REQ_ROWS:
                ok = n - REQ_ROWS < mb
            if ok:
                u0 = C.ld_shared_v4(xbf + (2 * kb) * 128 + n * 16)
                u1 = C.ld_shared_v4(xbf + (2 * kb + 1) * 128 + n * 16)
                v = []
                for word in (*u0, *u1):
                    lo, hi = P.bf16x2_unpack(word)
                    v.append(lo)
                    v.append(hi)
                amax = Float32(0.0)
                for q in cutlass.range_constexpr(16):
                    amax = T.max_f32(amax, T.abs_f32(v[q]))
                osc, sfb = T.nvfp4_block_scale(amax, gs1)
                q0 = [P.mul_f32(v[q], osc) for q in range(8)]
                q1 = [P.mul_f32(v[8 + q], osc) for q in range(8)]
                w0 = T.e2m1x8(*q0)
                w1 = T.e2m1x8(*q1)
            P.st_shared_v2_u32(
                base + OFF_X + (kb >> 1) * 128 + n * 16 + (kb & 1) * 8, w0, w1
            )
            P.st_shared_b8(base + OFF_XSF + (kb >> 2) * 128 + n * 16 + (kb & 3), sfb)
        C.fence_proxy_async_shared_cta()
        cute.arch.sync_warp()
        # bar.sync (PTX 8.9.4): all lanes' stores precede the release
        cute.arch.barrier(barrier_id=3 + (((et + 64) >> 5) & 3), number_of_threads=32)
        if lane == 0:
            C.mbarrier_arrive(misc + self.g.b_xready)

    @cute.jit
    def token_scales_g1(
        self, ids, topw, g1a, g2a, rows, ids_fin, w_fin, g2s, misc, block, nunits, lane
    ):
        g = self.g
        nid = rows * TOP_K
        eid = Int32(-1)
        wt = Float32(0.0)
        if lane < nid:
            eid = P.ld_cg_s32(ids + Int64(lane) * 4)
            wt = P.ld_cg_f32(topw + Int64(lane) * 4)
        mysh = Int32(0)
        my_unit = Int32(-1)
        if lane < nunits:
            mysh = Int32(self.unit_field(misc, lane, g.u_sh))
            if mysh == 0:
                my_unit = Int32(self.unit_field(misc, lane, g.u_e))
        g1 = Float32(0.0)
        g2 = Float32(0.0)
        if my_unit >= 0:
            g1 = P.ld_cg_f32(g1a + Int64(my_unit) * 4)
            g2 = P.ld_cg_f32(g2a + Int64(my_unit) * 4)
        tk = cute.make_rmem_tensor(cute.make_layout(REQ_ROWS), Float32)
        sl = cute.make_rmem_tensor(cute.make_layout(REQ_ROWS), Int32)
        for n in cutlass.range_constexpr(REQ_ROWS):
            tk[n] = Float32(0.0)
            sl[n] = Int32(-1)
        for u in cutlass.range(nunits, unroll=1):
            e = Int32(self.unit_field(misc, u, g.u_e))
            msk = Uint32(cute.arch.vote_ballot_sync(eid == e))
            for n in cutlass.range_constexpr(REQ_ROWS):
                bits = (msk >> Uint32(8 * n)) & Uint32(0xFF)
                src = Int32(0)
                if bits != Uint32(0):
                    src = 8 * n + P.ctz(bits)
                w = cute.arch.shuffle_sync(wt, src)
                if lane == u:
                    if bits != Uint32(0):
                        tk[n] = w
                        sl[n] = P.ctz(bits)
                    else:
                        tk[n] = Float32(0.0)
                        sl[n] = Int32(-1)
        if (block == 0) & (lane < nid):
            self.slot_record(ids_fin, w_fin, g2s, g2a, lane, eid, wt)
        if lane < nunits:
            unit = misc + g.m_units + lane * g.unit_bytes
            if mysh != 0:
                C.st_shared_f32(unit + g.u_g1, Float32(1.0))
                for n in cutlass.range_constexpr(REQ_ROWS):
                    one = Float32(0.0)
                    if n < rows:
                        one = Float32(1.0)
                    C.st_shared_f32(unit + g.u_tok + 4 * n, one)
                    C.st_shared_u32(unit + g.u_slot + 4 * n, Int32(-1))
            else:
                C.st_shared_f32(unit + g.u_g1, g1)
                for n in cutlass.range_constexpr(REQ_ROWS):
                    C.st_shared_f32(unit + g.u_tok + 4 * n, P.mul_f32(tk[n], g2))
                    C.st_shared_u32(unit + g.u_slot + 4 * n, sl[n])

    @cute.jit
    def slot_record(self, ids_fin, w_fin, g2s, g2a, at, eid, wt):
        """Slot ``at``: its expert id (-1 if out of range), routing weight and FC2
        alpha."""
        ok = Uint32(eid) < Uint32(NUM_EXPERTS)
        g2v = Float32(0.0)
        sid = Int32(-1)
        if ok:
            g2v = P.ld_cg_f32(g2a + Int64(eid) * 4)
            sid = eid
        P.st_global_b32(ids_fin + Int64(at) * 4, sid)
        P.st_global_b32f(w_fin + Int64(at) * 4, wt)
        P.st_global_b32f(g2s + Int64(at) * 4, g2v)

    @cute.jit
    def token_scales_g2(
        self,
        ids,
        topw,
        g1a,
        g2a,
        ma,
        mb,
        ids_fin,
        w_fin,
        g2s,
        misc,
        block,
        nunits,
        lane,
    ):
        g = self.g
        ea = Int32(-1)
        wa = Float32(0.0)
        eb = Int32(-1)
        wb = Float32(0.0)
        if lane < TOP_K * ma:
            ea = P.ld_cg_s32(ids + Int64(lane) * 4)
            wa = P.ld_cg_f32(topw + Int64(lane) * 4)
        if lane < TOP_K * mb:
            eb = P.ld_cg_s32(ids + Int64(TOP_K * ma + lane) * 4)
            wb = P.ld_cg_f32(topw + Int64(TOP_K * ma + lane) * 4)
        mysh = Int32(0)
        my_unit = Int32(-1)
        mygm = Int32(0)
        if lane < nunits:
            mysh = Int32(self.unit_field(misc, lane, g.u_sh))
            # only the two group bits are used: the mask's low byte
            mygm = P.ld_shared_u8(misc + g.m_units + lane * g.unit_bytes + g.u_gm)
            if mysh == 0:
                my_unit = Int32(self.unit_field(misc, lane, g.u_e))
        g1 = Float32(0.0)
        g2 = Float32(0.0)
        if my_unit >= 0:
            g1 = P.ld_cg_f32(g1a + Int64(my_unit) * 4)
            g2 = P.ld_cg_f32(g2a + Int64(my_unit) * 4)
        tk = cute.make_rmem_tensor(cute.make_layout(2 * REQ_ROWS), Float32)
        sl = cute.make_rmem_tensor(cute.make_layout(2 * REQ_ROWS), Int32)
        for r in cutlass.range_constexpr(2 * REQ_ROWS):
            tk[r] = Float32(0.0)
            sl[r] = Int32(-1)
        for u in cutlass.range(nunits, unroll=1):
            e = Int32(self.unit_field(misc, u, g.u_e))
            mka = Uint32(cute.arch.vote_ballot_sync(ea == e))
            mkb = Uint32(cute.arch.vote_ballot_sync(eb == e))
            for t in cutlass.range_constexpr(REQ_ROWS):
                ba = (mka >> Uint32(8 * t)) & Uint32(0xFF)
                bb = (mkb >> Uint32(8 * t)) & Uint32(0xFF)
                sa = Int32(0)
                sb = Int32(0)
                if ba != Uint32(0):
                    sa = 8 * t + P.ctz(ba)
                if bb != Uint32(0):
                    sb = 8 * t + P.ctz(bb)
                xa = cute.arch.shuffle_sync(wa, sa)
                xb = cute.arch.shuffle_sync(wb, sb)
                if lane == u:
                    if ba != Uint32(0):
                        tk[t] = xa
                        sl[t] = P.ctz(ba)
                    else:
                        tk[t] = Float32(0.0)
                        sl[t] = Int32(-1)
                    if bb != Uint32(0):
                        tk[REQ_ROWS + t] = xb
                        sl[REQ_ROWS + t] = P.ctz(bb)
                    else:
                        tk[REQ_ROWS + t] = Float32(0.0)
                        sl[REQ_ROWS + t] = Int32(-1)
        if block == 0:
            if lane < TOP_K * ma:
                self.slot_record(ids_fin, w_fin, g2s, g2a, lane, ea, wa)
            if lane < TOP_K * mb:
                self.slot_record(ids_fin, w_fin, g2s, g2a, TOP_K * ma + lane, eb, wb)
        if lane < nunits:
            unit = misc + g.m_units + lane * g.unit_bytes
            if mysh != 0:
                C.st_shared_f32(unit + g.u_g1, Float32(1.0))
            else:
                C.st_shared_f32(unit + g.u_g1, g1)
            for r in cutlass.range_constexpr(2 * REQ_ROWS):
                own = (mygm >> (r // REQ_ROWS)) & 1
                tok = Float32(0.0)
                slot = Int32(-1)
                if mysh != 0:
                    valid = r < ma
                    if cutlass.const_expr(r >= REQ_ROWS):
                        valid = r - REQ_ROWS < mb
                    if (own != 0) & valid:
                        tok = Float32(1.0)
                else:
                    if own != 0:
                        tok = P.mul_f32(tk[r], g2)
                        slot = sl[r]
                C.st_shared_f32(unit + g.u_tok + 4 * r, tok)
                C.st_shared_u32(unit + g.u_slot + 4 * r, slot)

    @cute.jit
    def routed_h(self, acc, g1, gs2, nrows, r0, hbuf, qd, lane):
        """Routed unit: h = SiLU(g1 gate) * (g1 up) of one group's 4 columns, NVFP4 into
        image rows r0 .. r0 + nrows - 1 (lane i of a quadrant: gate iff i & 8, its up
        partner i ^ 8)."""
        gate = (lane & 8) != 0
        hv = []
        am = []
        for n in cutlass.range_constexpr(REQ_ROWS):
            mine = P.mul_f32(g1, acc[n])
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
        wbase = (lane >> 2) & 1
        words = []
        for n in cutlass.range_constexpr(REQ_ROWS):
            q = [
                cute.arch.shuffle_sync(hq[n], 8 + 4 * wbase + (kk >> 1) + 16 * (kk & 1))
                for kk in range(8)
            ]
            words.append(T.e2m1x8(*q))
        for n in cutlass.range_constexpr(REQ_ROWS):
            if n < nrows:
                r = r0 + n
                if (lane == 8) | (lane == 12):
                    off = Int32(0)
                    if lane == 12:
                        off = Int32(4)
                    C.st_shared_u32(
                        hbuf + (qd >> 1) * 128 + r * 16 + (qd & 1) * 8 + off, words[n]
                    )
                if lane == 8:
                    P.st_shared_b8(hbuf + 256 + r * 16 + qd, sfb[n])

    @cute.jit
    def shared_h(self, acc, nrows, r0, hbf, qd, lane):
        """Shared quarter: stock BF16 rounding points (gate / up to BF16, SiLU to BF16,
        product to BF16) into BF16 image rows r0 .. r0 + nrows - 1."""
        for n in cutlass.range_constexpr(REQ_ROWS):
            gb = P.round_bf16(acc[n])
            ub = cute.arch.shuffle_sync_down(gb, 16)
            sv = P.round_bf16(T.silu_accurate(gb))
            hb = Uint32(C.cvt_rn_bf16_f32(P.mul_f32(sv, ub)))
            hn = cute.arch.shuffle_sync_down(hb, 1)
            if (n < nrows) & (lane < 16) & ((lane & 1) == 0):
                kc = 2 * qd + (lane >> 3)
                C.st_shared_u32(
                    hbf + kc * 128 + (r0 + n) * 16 + (lane & 7) * 2,
                    hb | (hn << Uint32(16)),
                )

    @cute.jit
    def epilogue_g1(
        self,
        x,
        ids,
        topw,
        g1a,
        g2a,
        gs1,
        gs2,
        rows,
        shpartials,
        pq,
        ids_fin,
        w_fin,
        g2s,
        base,
        misc,
        tm,
        crank,
        block,
        warp,
        lane,
        tid,
    ):
        g = self.g
        qd = warp & 3
        et = tid - 64
        row = 32 * qd + lane
        self.zero_fills(base, et)
        self.wait_inputs()
        self.quantize_x(x, gs1, rows, 0, base, misc, crank, et, lane)
        C.mbarrier_wait_parity(misc + g.b_tready, 0)
        nunits = Int32(C.ld_shared_u32(misc + g.m_nunits))
        if warp == 2:
            self.token_scales_g1(
                ids,
                topw,
                g1a,
                g2a,
                rows,
                ids_fin,
                w_fin,
                g2s,
                misc,
                block,
                nunits,
                lane,
            )
        cute.arch.barrier(barrier_id=1, number_of_threads=128)
        cute.arch.cluster_wait()
        lane_base = tm + Int32(32 * qd) * 65536
        racc = cute.make_rmem_tensor(cute.make_layout(FC2_TILES * REQ_ROWS), Float32)
        for t in cutlass.range_constexpr(FC2_TILES * REQ_ROWS):
            racc[t] = Float32(0.0)
        recv = base + OFF_RECV
        for k in cutlass.range(nunits + LOOKAHEAD, unroll=1):
            if k < nunits:
                i = k
                p = i % g.recv_bufs
                b = i & 1
                use = i // g.recv_bufs
                if et == 0:
                    C.mbarrier_arrive_expect_tx(
                        misc + g.b_precv + 8 * p, 3 * 128 * 16, ".relaxed.cluster"
                    )
                C.mbarrier_wait_parity(misc + g.b_acc1_full + 8 * b, (i >> 1) & 1)
                T.tc_fence_after()
                v = T.tmem_load4(lane_base + T_ACC1 + 8 * b)
                T.tc_fence_before()
                cute.arch.sync_warp()
                if lane == 0:
                    C.mbarrier_arrive(misc + g.b_acc1_empty + 8 * b)
                if use >= 1:
                    C.mbarrier_wait_parity(
                        misc + g.b_pfree + 8 * p, (use - 1) & 1, ".acquire.cluster"
                    )
                laddr = recv + ((p * 4 + crank) * 128 + row) * 16
                lbar = misc + g.b_precv + 8 * p
                for pr in cutlass.range_constexpr(CLUSTER):
                    if crank != pr:
                        P.st_async_v4_f32(
                            C.mapa_shared_cluster(laddr, pr),
                            C.mapa_shared_cluster(lbar, pr),
                            *v,
                        )
                C.mbarrier_wait_parity(
                    misc + g.b_precv + 8 * p, use & 1, ".acquire.cluster"
                )
                acc = [Float32(0.0), Float32(0.0), Float32(0.0), Float32(0.0)]
                for src in cutlass.range_constexpr(CLUSTER):
                    p0, p1, p2, p3 = v
                    if crank != src:
                        p0, p1, p2, p3 = P.ld_shared_v4_f32(
                            recv + ((p * 4 + src) * 128 + row) * 16
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
                if (lane == 0) & (i + g.recv_bufs < nunits):
                    for pr in cutlass.range_constexpr(CLUSTER):
                        if crank != pr:
                            P.mbarrier_arrive_remote(
                                C.mapa_shared_cluster(misc + g.b_pfree + 8 * p, pr)
                            )
                if self.unit_field(misc, i, g.u_sh) != Uint32(0):
                    self.shared_h(acc, rows, 0, base + g.off_hbf, qd, lane)
                else:
                    self.routed_h(
                        acc,
                        C.ld_shared_f32(misc + g.m_units + i * g.unit_bytes + g.u_g1),
                        gs2,
                        rows,
                        0,
                        base + g.off_h + H_BUF * (i % H_BUFS),
                        qd,
                        lane,
                    )
                C.fence_proxy_async_shared_cta()
                cute.arch.sync_warp()
                # bar.sync (PTX 8.9.4): all lanes' stores precede the release
                cute.arch.barrier(barrier_id=3 + qd, number_of_threads=32)
                if lane == 0:
                    C.mbarrier_arrive(misc + g.b_hready + 8 * self.hready_slot(i))
            if (k >= LOOKAHEAD) & (k - LOOKAHEAD < nunits):
                i = k - LOOKAHEAD
                C.mbarrier_wait_parity(misc + g.b_acc2_full, i & 1)
                T.tc_fence_after()
                C.fence_proxy_async_shared_cta()
                d = T.tmem_load4x12(lane_base + T_ACC2)
                T.tc_fence_before()
                cute.arch.sync_warp()
                if lane == 0:
                    C.mbarrier_arrive(misc + g.b_acc2_empty)
                if self.unit_field(misc, i, g.u_sh) == Uint32(0):
                    for n in cutlass.range_constexpr(REQ_ROWS):
                        sl = Int32(self.unit_field(misc, i, g.u_slot + 4 * n))
                        if (n < rows) & (sl >= 0):
                            uj = Int32(self.unit_field(misc, i, g.u_j))
                            q = (
                                pq
                                + (
                                    (((Int64(uj) * rows + n) * TOP_K + sl) * HIDDEN)
                                    + K_CTA * crank
                                    + row
                                )
                                * 4
                            )
                            for t in cutlass.range_constexpr(FC2_TILES):
                                P.st_global_f32(q + 512 * t, d[4 * t + n])
                else:
                    for n in cutlass.range_constexpr(REQ_ROWS):
                        s = C.ld_shared_f32(
                            misc + g.m_units + i * g.unit_bytes + g.u_tok + 4 * n
                        )
                        for t in cutlass.range_constexpr(FC2_TILES):
                            racc[4 * t + n] = C.fma_rn(s, d[4 * t + n], racc[4 * t + n])
                if cutlass.const_expr(self.shared):  # noqa: SIM102
                    # the shared quarter's partial is stored apart (stock adds it after
                    # the routed finalize), then its accumulator restarts
                    if self.unit_field(misc, i, g.u_sh) != Uint32(0):
                        uj = Int32(self.unit_field(misc, i, g.u_j))
                        q = (
                            shpartials
                            + ((Int64(uj) * rows) * HIDDEN + K_CTA * crank + row) * 4
                        )
                        for n in cutlass.range_constexpr(REQ_ROWS):
                            if n < rows:
                                for t in cutlass.range_constexpr(FC2_TILES):
                                    P.st_global_f32(
                                        q + (Int64(n) * HIDDEN + 128 * t) * 4,
                                        racc[4 * t + n],
                                    )
                        for t in cutlass.range_constexpr(FC2_TILES * REQ_ROWS):
                            racc[t] = Float32(0.0)

    @cute.jit
    def epilogue_g2(
        self,
        x,
        ids,
        topw,
        g1a,
        g2a,
        gs1,
        gs2,
        rows,
        ma,
        shpartials,
        pq,
        ids_fin,
        w_fin,
        g2s,
        base,
        misc,
        tm,
        crank,
        block,
        warp,
        lane,
        tid,
    ):
        g = self.g
        mb = rows - ma
        qd = warp & 3
        et = tid - 64
        row = 32 * qd + lane
        self.zero_fills(base, et)
        self.wait_inputs()
        self.quantize_x(x, gs1, ma, mb, base, misc, crank, et, lane)
        C.mbarrier_wait_parity(misc + g.b_tready, 0)
        nunits = Int32(C.ld_shared_u32(misc + g.m_nunits))
        nex = Int32(C.ld_shared_u32(misc + g.m_nex))
        if warp == 2:
            self.token_scales_g2(
                ids,
                topw,
                g1a,
                g2a,
                ma,
                mb,
                ids_fin,
                w_fin,
                g2s,
                misc,
                block,
                nunits,
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
                gm = Int32(self.unit_field(misc, i, g.u_gm))
                ng = P.popc(gm)
                g0 = Int32(1)
                if (gm & 1) != 0:
                    g0 = Int32(0)
                if et == 0:
                    C.mbarrier_arrive_expect_tx(
                        misc + g.b_precv + 8 * (ex % g.recv_bufs),
                        3 * 128 * 16,
                        ".relaxed.cluster",
                    )
                    if ng == 2:
                        C.mbarrier_arrive_expect_tx(
                            misc + g.b_precv + 8 * ((ex + 1) % g.recv_bufs),
                            3 * 128 * 16,
                            ".relaxed.cluster",
                        )
                C.mbarrier_wait_parity(misc + g.b_acc1_full + 8 * b, (i >> 1) & 1)
                T.tc_fence_after()
                va, vb = T.tmem_load4_pair(
                    lane_base + T_ACC1 + 8 * b + 4 * g0,
                    lane_base + T_ACC1 + 8 * b + 4,
                    Int32(ng == 2),
                )
                T.tc_fence_before()
                cute.arch.sync_warp()
                if lane == 0:
                    C.mbarrier_arrive(misc + g.b_acc1_empty + 8 * b)
                g1 = C.ld_shared_f32(misc + g.m_units + i * g.unit_bytes + g.u_g1)
                ush = self.unit_field(misc, i, g.u_sh)
                for gi in cutlass.range(ng, unroll=1):
                    grp = g0
                    if gi != 0:
                        grp = Int32(1)
                    mg = ma
                    if grp != 0:
                        mg = mb
                    v0, v1, v2, v3 = va
                    if gi != 0:
                        v0, v1, v2, v3 = vb
                    e = ex + gi
                    p = e % g.recv_bufs
                    use = e // g.recv_bufs
                    if use >= 1:
                        C.mbarrier_wait_parity(
                            misc + g.b_pfree + 8 * p, (use - 1) & 1, ".acquire.cluster"
                        )
                    lbar = misc + g.b_precv + 8 * p
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
                        misc + g.b_precv + 8 * p, use & 1, ".acquire.cluster"
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
                    if (lane == 0) & (e + g.recv_bufs < nex):
                        C.mbarrier_arrive(misc + g.b_consumed + 8 * p)
                    if ush != Uint32(0):
                        self.shared_h(
                            acc, mg, REQ_ROWS * grp, base + g.off_hbf, qd, lane
                        )
                    else:
                        self.routed_h(
                            acc,
                            g1,
                            gs2,
                            mg,
                            REQ_ROWS * grp,
                            base + g.off_h + H_BUF * (i % H_BUFS),
                            qd,
                            lane,
                        )
                ex = ex + ng
                C.fence_proxy_async_shared_cta()
                cute.arch.sync_warp()
                # bar.sync (PTX 8.9.4): all lanes' stores precede the release
                cute.arch.barrier(barrier_id=3 + qd, number_of_threads=32)
                if lane == 0:
                    C.mbarrier_arrive(misc + g.b_hready + 8 * self.hready_slot(i))
            if (k >= LOOKAHEAD) & (k - LOOKAHEAD < nunits):
                i = k - LOOKAHEAD
                ab = i % 2
                C.mbarrier_wait_parity(misc + g.b_acc2_full + 8 * ab, (i // 2) & 1)
                T.tc_fence_after()
                C.fence_proxy_async_shared_cta()
                acc2 = lane_base + T_ACC2
                if ab == 1:
                    acc2 = lane_base + T_ACC2_B
                gm = Int32(self.unit_field(misc, i, g.u_gm))
                ng = P.popc(gm)
                g0 = Int32(1)
                if (gm & 1) != 0:
                    g0 = Int32(0)
                for gi in cutlass.range(ng, unroll=1):
                    grp = g0
                    if gi != 0:
                        grp = Int32(1)
                    mg = ma
                    if grp != 0:
                        mg = mb
                    d = T.tmem_load4x12(acc2 + 4 * grp)
                    if gi == ng - 1:
                        T.tc_fence_before()
                        cute.arch.sync_warp()
                        if lane == 0:
                            C.mbarrier_arrive(misc + g.b_acc2_empty + 8 * ab)
                    # unit fields are read per group, where the readout uses them
                    if self.unit_field(misc, i, g.u_sh) == Uint32(0):
                        for n in cutlass.range_constexpr(REQ_ROWS):
                            r = REQ_ROWS * grp + n
                            sl = Int32(
                                C.ld_shared_u32(
                                    misc
                                    + g.m_units
                                    + i * g.unit_bytes
                                    + g.u_slot
                                    + 4 * r
                                )
                            )
                            if (n < mg) & (sl >= 0):
                                uj = Int32(self.unit_field(misc, i, g.u_j))
                                grow = r
                                if r >= REQ_ROWS:
                                    grow = ma + r - REQ_ROWS
                                q = (
                                    pq
                                    + (
                                        (
                                            ((Int64(uj) * rows + grow) * TOP_K + sl)
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
                                    r = REQ_ROWS * grp + n
                                    s = C.ld_shared_f32(
                                        misc
                                        + g.m_units
                                        + i * g.unit_bytes
                                        + g.u_tok
                                        + 4 * r
                                    )
                                    uj = Int32(self.unit_field(misc, i, g.u_j))
                                    grow = r
                                    if r >= REQ_ROWS:
                                        grow = ma + r - REQ_ROWS
                                    q = (
                                        shpartials
                                        + (
                                            (Int64(uj) * rows + grow) * HIDDEN
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
        split: Int32,
        err: Int64,
        sched: Int64,
        shpartials: Int64,
        pq: Int64,
        ids_fin: Int64,
        w_fin: Int64,
        g2s: Int64,
        sf13: Int64,
        fc2pack: Int64,
    ):
        g = self.g
        tid, _, _ = cute.arch.thread_idx()
        block, _, _ = cute.arch.block_idx()
        warp = tid // 32
        lane = tid % 32
        sm = SmemAllocator().allocate_tensor(Uint8, cute.make_layout(g.smem), 1024)
        base = Int32(sm.iterator.toint())
        misc = base + g.off_misc
        crank = Int32(cute.arch.block_idx_in_cluster())
        cl = block // CLUSTER
        # the kernel declares no cluster shape (the launch sets it): refuse any
        # other shape
        P.trap_if(Int32(cute.arch.cluster_size() != CLUSTER))
        C.pdl_launch_dependents()
        if tid == 0:
            for s in cutlass.range_constexpr(RING):
                C.mbarrier_init(misc + g.b_full + 8 * s, 1)
                C.mbarrier_init(misc + g.b_empty + 8 * s, 1)
            C.mbarrier_init(misc + g.b_xready, 4)
            # one arrive per lane of the table-building warp: each lane publishes its
            # own unit-record stores (PTX memory model; no reliance on the warp barrier)
            C.mbarrier_init(misc + g.b_tready, 32)
            C.mbarrier_init(misc + g.b_sbar, 1)
            for b in cutlass.range_constexpr(2):
                C.mbarrier_init(misc + g.b_acc1_full + 8 * b, 1)
                C.mbarrier_init(misc + g.b_acc1_empty + 8 * b, 4)
            for b in cutlass.range_constexpr(g.acc2_bufs):
                C.mbarrier_init(misc + g.b_acc2_full + 8 * b, 1)
                C.mbarrier_init(misc + g.b_acc2_empty + 8 * b, 4)
            for s in cutlass.range_constexpr(RING):
                C.mbarrier_init(misc + g.b_sfa_full + 8 * s, 1)
            C.mbarrier_init(misc + g.b_sfx_ready, 1)
            for b in cutlass.range_constexpr(H_BUFS):
                C.mbarrier_init(misc + g.b_hready + 8 * b, 4)
            for p in cutlass.range_constexpr(g.recv_bufs):
                C.mbarrier_init(misc + g.b_precv + 8 * p, 1)
                C.mbarrier_init(misc + g.b_pfree + 8 * p, g.pfree_count)
                if cutlass.const_expr(g.two):
                    C.mbarrier_init(misc + g.b_consumed + 8 * p, 4)
            C.fence_mbarrier_init_release_cluster()
        if warp == 1:
            T.tmem_alloc(misc + g.m_tmem, 512)
        T.tc_fence_before()
        cute.arch.sync_threads()
        T.tc_fence_after()
        cute.arch.cluster_arrive_relaxed(aligned=True)
        # TMEM base, read by every thread once the allocation is visible
        tm = Int32(C.ld_shared_u32(misc + g.m_tmem))
        my_sh = Int32(0)
        if cl < SH_CLUSTERS:
            my_sh = Int32(1)
        if warp == 0:
            if cutlass.const_expr(g.two):
                self.producer_g2(
                    tmap.get_ptr().toint(),
                    ids,
                    shpack,
                    rows,
                    split,
                    err,
                    sched,
                    sf13,
                    fc2pack,
                    base,
                    misc,
                    crank,
                    cl,
                    my_sh,
                    lane,
                )
            else:
                self.producer_g1(
                    tmap.get_ptr().toint(),
                    ids,
                    shpack,
                    rows,
                    err,
                    sched,
                    sf13,
                    fc2pack,
                    base,
                    misc,
                    crank,
                    cl,
                    my_sh,
                    lane,
                )
        elif warp == 1:
            self.mma_warp(base, misc)
        elif warp == SF_WARP:
            self.sf_warp(base, misc)
        elif warp == RELEASE_WARP:
            if cutlass.const_expr(g.two):
                self.release_warp(misc, crank, lane)
        else:
            if cutlass.const_expr(g.two):
                self.epilogue_g2(
                    x,
                    ids,
                    topw,
                    g1a,
                    g2a,
                    gs1,
                    gs2,
                    rows,
                    split,
                    shpartials,
                    pq,
                    ids_fin,
                    w_fin,
                    g2s,
                    base,
                    misc,
                    tm,
                    crank,
                    block,
                    warp,
                    lane,
                    tid,
                )
            else:
                self.epilogue_g1(
                    x,
                    ids,
                    topw,
                    g1a,
                    g2a,
                    gs1,
                    gs2,
                    rows,
                    shpartials,
                    pq,
                    ids_fin,
                    w_fin,
                    g2s,
                    base,
                    misc,
                    tm,
                    crank,
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
