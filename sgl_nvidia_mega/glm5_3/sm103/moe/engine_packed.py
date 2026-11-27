# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Packed-token (N=8 per active expert tile) SM103 persistent NVFP4 routed-expert
engine for GLM-5.3 TP4 mid-batch (M <= 32, quarters=8, INTER=512, atomic_out FP32).

Consumes ``ids[M, 8]`` + ``topw[M, 8]`` directly from ``MoeRouteTop8(multi_cta=True)``
and packs active experts into dense N=8 tiles in SMEM in ~0.5 us at kernel entry
(overlapping TMEM alloc, mbarrier init, and XSF/H zeroing behind routing via PDL).

Supports ``q_acc`` in {1, 2, 4, 8}: each cluster processes ``q_acc`` consecutive
quarters of the same packed tile back-to-back, reusing ``X`` + ``T_SFB_X`` and
accumulating FC2 directly in ``T_ACC2`` before flushing once via ``red.global.add.f32``.
"""

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
from .engine import (
    BIG,
    CLUSTER,
    FC1_CPS,
    FC1_MMAS,
    FC2_CPS,
    FC2_RECORD,
    FC2_TILES,
    H_BUF,
    H_BUFS,
    H_CPS,
    IDESC_NVF4,
    K_CTA,
    LOOKAHEAD,
    REQ_ROWS,
    RING,
    SLOT,
    X_CPS,
)

# FC2 MMAs with dynamic accumulate ("first" -> 0 on iql==0, 1 on iql>0)
FC2_MMAS_ACC = tuple(
    (8 * (4 * h2 + t), SLOT * h2 + 4096 * t, 0, 16 * h2 + 4 * t, 0, "first", h2)
    for h2 in range(2)
    for t in range(4)
)

# ------------------------------------------------------------------ geometry & memory map
THREADS = 288  # 9 warps
RECV_BUFS = 4
X_SLOTS = 2
X_BYTES = 6144
XSF_BYTES = 3456
X_SLOT_STRIDE = X_BYTES + XSF_BYTES  # 9600 B (128-byte aligned)
MAX_TILES = 256

# Shared memory layout (1024-byte aligned base, total ~212 KiB <= 227 KiB)
OFF_RING = 0  # 4 * 36864 = 147456 B (also scratch for in-cluster pack at startup)
OFF_X = OFF_RING + RING * BIG  # 2 * 9600 = 19200 B
OFF_RECV = OFF_X + X_SLOTS * X_SLOT_STRIDE  # 24576 B
OFF_H = OFF_RECV + RECV_BUFS * 3 * 128 * 16  # 2304 B
# Persistent packed tile table in SMEM (17408 B)
OFF_TILE_E = OFF_H + H_BUFS * H_BUF  # 256 * 4 = 1024 B
OFF_TILE_TOK = OFF_TILE_E + MAX_TILES * 4  # 256 * 8 * 4 = 8192 B
OFF_TILE_W = OFF_TILE_TOK + MAX_TILES * 32  # 256 * 8 * 4 = 8192 B
OFF_MISC = OFF_TILE_W + MAX_TILES * 32

# Startup pack scratch overlaid inside OFF_RING (before TMA starts!)
PACK_MASK = 0  # 256 * 4 = 1024 B
PACK_W = 1024  # 256 * 32 * 4 = 32768 B
PACK_WSUM = 1024 + 32768  # 32 B
PACK_WBASE = PACK_WSUM + 32  # 32 B

B_FULL = 0
B_EMPTY = B_FULL + RING * 8
B_SFA_FULL = B_EMPTY + RING * 8
B_XREADY = B_SFA_FULL + RING * 8
B_XFREE = B_XREADY + X_SLOTS * 8
B_SFX_FREE = B_XFREE + X_SLOTS * 8
B_SFX_READY = B_SFX_FREE + 8
B_ACC1_FULL = B_SFX_READY + X_SLOTS * 8
B_ACC1_EMPTY = B_ACC1_FULL + 2 * 8
B_HREADY = B_ACC1_EMPTY + 2 * 8
B_SFH_READY = B_HREADY + H_BUFS * 8
B_ACC2_FULL = B_SFH_READY + H_BUFS * 8
B_ACC2_EMPTY = B_ACC2_FULL + 2 * 8
B_PRECV = B_ACC2_EMPTY + 2 * 8
B_CONSUMED = B_PRECV + RECV_BUFS * 8
B_PFREE = B_CONSUMED + RECV_BUFS * 8
M_TMEM = B_PFREE + RECV_BUFS * 8
M_NTILES = M_TMEM + 4
M_NBLOCKS = M_NTILES + 4
M_NUNITS = M_NBLOCKS + 4
MISC_BYTES = ((M_NUNITS + 4 + 127) // 128) * 128
SMEM_BYTES = OFF_MISC + MISC_BYTES  # ~211.6 KiB <= 227 KiB

# Tensor memory columns (all 16-col/32-col aligned! total 448 <= 512)
T_ACC1 = 0  # 2 * 8 = 16 (0..15)
T_ACC2_A = 16  # 12 * 8 = 96 (16..111)
T_ACC2_B = 112  # 12 * 8 = 96 (112..207)
T_SFB_X = 208  # 24 * 4 = 96 (208..303)
T_SFB_H = 304  # 3 * 4 = 12 (304..315, padded to 320)
T_SFA = 320  # 4 * 32 = 128 (320..447)


class Nvfp4MoePacked:
    """Packed N=8 SM103 persistent NVFP4 MoE for M <= 32, quarters=8 (INTER=512)."""

    quarters = 8
    q_acc = 4
    pre = 3

    def __init__(self, quarters: int = 8, q_acc: int = 4, pre: int = 3):
        if quarters not in (4, 8):
            raise ValueError(f"Nvfp4MoePacked: quarters {quarters} (4 or 8)")
        if q_acc not in (1, 2, 4, 8) or (quarters % q_acc) != 0:
            raise ValueError(f"Nvfp4MoePacked: q_acc {q_acc} must divide quarters {quarters}")
        if pre not in (1, 3):
            raise ValueError(f"Nvfp4MoePacked: pre {pre} (1 or 3)")
        self.quarters = int(quarters)
        self.q_acc = int(q_acc)
        self.q_groups = self.quarters // self.q_acc
        self.pre = int(pre)
        self.fc1_rows = 128 * self.quarters
        self.sf13_per_expert = self.fc1_rows * 384

    def knobs(self) -> dict:
        return {"quarters": self.quarters, "q_acc": self.q_acc, "pre": self.pre}

    @cute.jit
    def unit_tile_q(self, cl, nclusters, k):
        blk = k // self.q_acc
        ql = k % self.q_acc
        bg = cl + blk * nclusters
        ti = bg // self.q_groups
        qi = (bg % self.q_groups) * self.q_acc + ql
        return ti, qi, blk, ql

    @cute.jit
    def pack_in_smem(self, ids, topw, rows, base, misc, cl, nclusters, tid, warp, lane):
        """Pack ids[rows, 8] + topw[rows, 8] into SMEM OFF_TILE_* in ~0.5 us."""
        if tid < NUM_EXPERTS:
            C.st_shared_u32(base + PACK_MASK + tid * 4, Uint32(0))
        cute.arch.sync_threads()

        # Up to 32 * 8 = 256 pairs -> 1 thread per (row, slot)
        total_pairs = rows * TOP_K
        if tid < total_pairs:
            r = tid >> 3
            e = P.ld_cg_s32(ids + Int64(tid) * 4)
            w = P.ld_cg_f32(topw + Int64(tid) * 4)
            if Uint32(e) < Uint32(NUM_EXPERTS):
                P.atom_or_shared(base + PACK_MASK + e * 4, Uint32(1) << Uint32(r))
                C.st_shared_f32(base + PACK_W + (e * 32 + r) * 4, w)
        cute.arch.sync_threads()

        val = Int32(0)
        nt = Int32(0)
        cnt = Int32(0)
        m = Uint32(0)
        if tid < NUM_EXPERTS:
            m = C.ld_shared_u32(base + PACK_MASK + tid * 4)
            cnt = P.popc(m)
            nt = (cnt + 7) >> 3
            val = nt
            for off in cutlass.range_constexpr(5):
                d = 1 << off
                up = cute.arch.shuffle_sync_up(val, d)
                if lane >= d:
                    val = val + up
            if lane == 31:
                C.st_shared_u32(base + PACK_WSUM + warp * 4, Uint32(val))
        cute.arch.sync_threads()

        if warp == 0:
            ws = Int32(0)
            if lane < 8:
                ws = Int32(C.ld_shared_u32(base + PACK_WSUM + lane * 4))
            winc = ws
            for off in cutlass.range_constexpr(3):
                d = 1 << off
                up = cute.arch.shuffle_sync_up(winc, d)
                if lane >= d:
                    winc = winc + up
            if lane < 8:
                C.st_shared_u32(base + PACK_WBASE + lane * 4, Uint32(winc - ws))
            if lane == 7:
                ntiles = winc
                total_b = ntiles * self.q_groups
                nblocks = Int32(0)
                if cl < total_b:
                    nblocks = (total_b - cl + nclusters - 1) // nclusters
                C.st_shared_u32(misc + M_NTILES, Uint32(ntiles))
                C.st_shared_u32(misc + M_NBLOCKS, Uint32(nblocks))
                C.st_shared_u32(misc + M_NUNITS, Uint32(nblocks * self.q_acc))
        cute.arch.sync_threads()

        if tid < NUM_EXPERTS:
            wbase = Int32(C.ld_shared_u32(base + PACK_WBASE + warp * 4))
            t0 = wbase + val - nt
            for ti in cutlass.range(nt, unroll=1):
                t = t0 + ti
                if t < MAX_TILES:
                    C.st_shared_u32(base + OFF_TILE_E + t * 4, Uint32(tid))
                    tok_base = base + OFF_TILE_TOK + t * 32
                    w_base = base + OFF_TILE_W + t * 32
                    for k in cutlass.range_constexpr(8):
                        idx = ti * 8 + k
                        if idx < cnt:
                            rr = P.fns(m, Uint32(0), Int32(idx + 1))
                            wv = C.ld_shared_f32(base + PACK_W + (tid * 32 + rr) * 4)
                            C.st_shared_u32(tok_base + k * 4, Uint32(rr))
                            C.st_shared_f32(w_base + k * 4, wv)
                        else:
                            C.st_shared_u32(tok_base + k * 4, Uint32(0xFFFFFFFF))
                            C.st_shared_f32(w_base + k * 4, Float32(0.0))
        cute.arch.sync_threads()

    @cute.jit
    def fc1_chunk(self, tmap_ptr, sf13, dst, expert, quarter, crank, c, bar, policy):
        T.tma_load_3d(
            dst,
            tmap_ptr,
            0,
            expert * self.fc1_rows + 128 * quarter,
            crank * 6 + 2 * c,
            bar,
            policy,
        )
        src = (
            sf13
            + Int64(expert) * self.sf13_per_expert
            + Int64(96 * quarter + 24 * crank + 8 * c) * 512
        )
        C.bulk_copy_g2s(dst + 32768, src, 4096, bar, policy)

    @cute.jit
    def ring_tail(self, misc, it):
        first = it - RING
        if first < 0:
            first = Int32(0)
        for c in cutlass.range(first, it, unroll=1):
            C.mbarrier_wait_parity(misc + B_EMPTY + 8 * (c % RING), (c // RING) & 1)

    @cute.jit
    def routed_h(self, acc, g1, gs2, r0, hbuf, qd, lane):
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

    # ------------------------------------------------------------ warp 0: TMA producer
    @cute.jit
    def producer_warp(
        self,
        tmap_ptr,
        sf13,
        fc2pack,
        base,
        misc,
        crank,
        cl,
        nclusters,
        nunits,
        lane,
    ):
        policy = C.l2_policy("evict_first")
        if (lane == 0) & (nunits > 0):
            t0, q0, _, _ = self.unit_tile_q(cl, nclusters, Int32(0))
            e0 = P.ld_shared_s32(base + OFF_TILE_E + t0 * 4)
            npre = 3 if cutlass.const_expr(self.pre == 3) else 1
            for c in cutlass.range_constexpr(npre):
                C.mbarrier_arrive_expect_tx(misc + B_FULL + 8 * c, BIG)
                self.fc1_chunk(
                    tmap_ptr,
                    sf13,
                    base + OFF_RING + c * BIG,
                    e0,
                    q0,
                    crank,
                    c,
                    misc + B_FULL + 8 * c,
                    policy,
                )
            pre0 = Int32(npre)
            it = Int32(npre)

            for k in cutlass.range(nunits + LOOKAHEAD, unroll=1):
                for part in cutlass.range_constexpr(2):
                    i = k if part == 0 else k - LOOKAHEAD
                    if (i >= 0) & (i < nunits):
                        ti, qi, _, _ = self.unit_tile_q(cl, nclusters, i)
                        ei = P.ld_shared_s32(base + OFF_TILE_E + ti * 4)
                        if cutlass.const_expr(part == 0):
                            c0 = Int32(0)
                            if i == 0:
                                c0 = pre0
                            for c in cutlass.range(c0, 3, unroll=1):
                                s = it % RING
                                if it >= RING:
                                    C.mbarrier_wait_parity(
                                        misc + B_EMPTY + 8 * s, ((it // RING) - 1) & 1
                                    )
                                C.mbarrier_arrive_expect_tx(misc + B_FULL + 8 * s, BIG)
                                self.fc1_chunk(
                                    tmap_ptr,
                                    sf13,
                                    base + OFF_RING + s * BIG,
                                    ei,
                                    qi,
                                    crank,
                                    c,
                                    misc + B_FULL + 8 * s,
                                    policy,
                                )
                                it = it + 1
                        else:
                            src2 = (
                                fc2pack
                                + ((Int64(ei) * self.quarters + Int64(qi)) * 4 + Int64(crank))
                                * FC2_RECORD
                            )
                            for c in cutlass.range_constexpr(2):
                                s = it % RING
                                if it >= RING:
                                    C.mbarrier_wait_parity(
                                        misc + B_EMPTY + 8 * s, ((it // RING) - 1) & 1
                                    )
                                nbytes = BIG if c == 0 else SLOT
                                C.mbarrier_arrive_expect_tx(misc + B_FULL + 8 * s, nbytes)
                                C.bulk_copy_g2s(
                                    base + OFF_RING + s * BIG,
                                    src2 + Int64(2 * c) * SLOT,
                                    nbytes,
                                    misc + B_FULL + 8 * s,
                                    policy,
                                )
                                it = it + 1
            self.ring_tail(misc, it)
        cute.arch.sync_warp()

    # ------------------------------------------------------------ warp 1: MMA issue
    @cute.jit
    def mma_warp(self, base, misc, tm, nunits):
        lead = T.elect_leader()
        s_ring = base + OFF_RING
        it = Int32(0)

        for k in cutlass.range(nunits + LOOKAHEAD, unroll=1):
            if k < nunits:
                b = k & 1
                blk = k // self.q_acc
                ql = k % self.q_acc
                xb = blk & 1
                if k >= 2:
                    C.mbarrier_wait_parity(
                        misc + B_ACC1_EMPTY + 8 * b, ((k >> 1) - 1) & 1
                    )
                    T.tc_fence_after()
                if ql == 0:
                    C.mbarrier_wait_parity(
                        misc + B_SFX_READY + 8 * xb, (blk >> 1) & 1
                    )
                    T.tc_fence_after()
                x_base = base + OFF_X + xb * X_SLOT_STRIDE
                for c in cutlass.range(3, unroll=1):
                    s = it % RING
                    C.mbarrier_wait_parity(misc + B_FULL + 8 * s, (it // RING) & 1)
                    C.mbarrier_wait_parity(misc + B_SFA_FULL + 8 * s, (it // RING) & 1)
                    T.tc_fence_after()
                    a0 = T.smem_desc(s_ring + s * BIG, 16, 1024, 2)
                    b0 = T.smem_desc(x_base + 2048 * c, 128, 1024, 0)
                    T.tc_burst(
                        lead,
                        tm + T_ACC1 + 8 * b,
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
                if ql == self.q_acc - 1:
                    T.tc_commit(lead, misc + B_SFX_FREE)
                    T.tc_commit(lead, misc + B_XFREE + 8 * xb)
                cute.arch.sync_warp()

            if (k >= LOOKAHEAD) & (k - LOOKAHEAD < nunits):
                i = k - LOOKAHEAD
                iblk = i // self.q_acc
                iql = i % self.q_acc
                ab = iblk & 1
                if (iql == 0) & (iblk >= 2):
                    C.mbarrier_wait_parity(
                        misc + B_ACC2_EMPTY + 8 * ab, ((iblk >> 1) - 1) & 1
                    )
                    T.tc_fence_after()
                hb = i % H_BUFS
                C.mbarrier_wait_parity(misc + B_SFH_READY + 8 * hb, (i // H_BUFS) & 1)
                T.tc_fence_after()
                acc2_tm = tm + T_ACC2_A
                if ab != 0:
                    acc2_tm = tm + T_ACC2_B
                hb_desc = T.smem_desc(base + OFF_H + H_BUF * hb, 128, 256, 0)
                acc_flag = Int32(iql != 0)
                for c in cutlass.range_constexpr(2):
                    s = it % RING
                    C.mbarrier_wait_parity(misc + B_FULL + 8 * s, (it // RING) & 1)
                    C.mbarrier_wait_parity(misc + B_SFA_FULL + 8 * s, (it // RING) & 1)
                    T.tc_fence_after()
                    a0 = T.smem_desc(s_ring + s * BIG, 128, 256, 0)
                    T.tc_burst(
                        lead,
                        acc2_tm + 64 * c,
                        a0,
                        hb_desc,
                        tm + T_SFA + 32 * s,
                        tm + T_SFB_H + 4 * hb,
                        acc_flag,
                        Int32(c == 0),
                        misc + B_EMPTY + 8 * s,
                        0,
                        0,
                        mmas=FC2_MMAS_ACC,
                        block_scaled=True,
                        idesc=IDESC_NVF4,
                    )
                    cute.arch.sync_warp()
                    it = it + 1
                if iql == self.q_acc - 1:
                    T.tc_commit(lead, misc + B_ACC2_FULL + 8 * ab)
                    cute.arch.sync_warp()

    # ------------------------------------------------------------ warp 6: scale factors
    @cute.jit
    def sf_warp(self, base, misc, tm, nunits):
        lead = T.elect_leader()
        s_ring = base + OFF_RING
        it = Int32(0)

        for k in cutlass.range(nunits + LOOKAHEAD, unroll=1):
            if k < nunits:
                blk = k // self.q_acc
                ql = k % self.q_acc
                xb = blk & 1
                if ql == 0:
                    if blk >= 1:
                        C.mbarrier_wait_parity(misc + B_SFX_FREE, (blk - 1) & 1)
                        T.tc_fence_after()
                    C.mbarrier_wait_parity(misc + B_XREADY + 8 * xb, (blk >> 1) & 1)
                    T.tc_fence_after()
                    xsf_base = base + OFF_X + xb * X_SLOT_STRIDE + X_BYTES
                    d0 = T.smem_desc(xsf_base, 0, 128, 0)
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
                    T.tc_commit(lead, misc + B_SFX_READY + 8 * xb)
                    cute.arch.sync_warp()

                for c in cutlass.range(3, unroll=1):
                    s = it % RING
                    C.mbarrier_wait_parity(misc + B_FULL + 8 * s, (it // RING) & 1)
                    T.tc_fence_after()
                    sd = T.smem_desc(s_ring + s * BIG + 32768, 0, 128, 0)
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
                        sd,
                        cps=FC1_CPS,
                        commit=True,
                    )
                    cute.arch.sync_warp()
                    it = it + 1

            if (k >= LOOKAHEAD) & (k - LOOKAHEAD < nunits):
                i = k - LOOKAHEAD
                hb = i % H_BUFS
                C.mbarrier_wait_parity(misc + B_HREADY + 8 * hb, (i // H_BUFS) & 1)
                T.tc_fence_after()
                hd = T.smem_desc(base + OFF_H + H_BUF * hb + 256, 0, 128, 0)
                T.tc_burst(
                    lead,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    misc + B_SFH_READY + 8 * hb,
                    tm + T_SFB_H + 4 * hb,
                    hd,
                    cps=H_CPS,
                    commit=True,
                )
                cute.arch.sync_warp()

                for c in cutlass.range_constexpr(2):
                    s = it % RING
                    C.mbarrier_wait_parity(misc + B_FULL + 8 * s, (it // RING) & 1)
                    T.tc_fence_after()
                    sd = T.smem_desc(s_ring + s * BIG + 16384, 0, 128, 0)
                    T.tc_burst(
                        lead,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        Int32(c == 0),
                        misc + B_SFA_FULL + 8 * s,
                        tm + T_SFA + 32 * s,
                        sd,
                        cps=FC2_CPS,
                        commit=True,
                    )
                    cute.arch.sync_warp()
                    it = it + 1

    # ------------------------------------------------------------ warp 7: DSMEM release
    @cute.jit
    def release_warp(self, base, misc, crank, cl, nclusters, nblocks, lane):
        nex = Int32(0)
        for blk in cutlass.range(nblocks, unroll=1):
            ti = (cl + blk * nclusters) // self.q_groups
            t4 = P.ld_shared_s32(base + OFF_TILE_TOK + (ti * 8 + 4) * 4)
            per_q = Int32(1)
            if t4 >= 0:
                per_q = Int32(2)
            nex = nex + per_q * self.q_acc
        if nex > RECV_BUFS:
            for e in cutlass.range(nex - RECV_BUFS, unroll=1):
                p = e % RECV_BUFS
                C.mbarrier_wait_parity(misc + B_CONSUMED + 8 * p, (e // RECV_BUFS) & 1)
                if (lane < CLUSTER) & (lane != crank):
                    P.mbarrier_arrive_remote(
                        C.mapa_shared_cluster(misc + B_PFREE + 8 * p, lane)
                    )
        cute.arch.sync_warp()

    # ------------------------------------------------------------ warp 8: L2 X gather
    @cute.jit
    def x_gather_warp(
        self, x_fp4, x_sf, base, misc, crank, cl, nclusters, nblocks, lane
    ):
        n = lane & 7
        g4 = lane >> 3
        zero4 = (Uint32(0), Uint32(0), Uint32(0), Uint32(0))

        for blk in cutlass.range(nblocks, unroll=1):
            xb = blk & 1
            if blk >= 2:
                C.mbarrier_wait_parity(misc + B_XFREE + 8 * xb, ((blk >> 1) - 1) & 1)
            ti = (cl + blk * nclusters) // self.q_groups
            tok = P.ld_shared_s32(base + OFF_TILE_TOK + (ti * 8 + n) * 4)
            x_dst = base + OFF_X + xb * X_SLOT_STRIDE
            xsf_dst = x_dst + X_BYTES

            if tok >= 0:
                fp4_row = x_fp4 + Int64(tok) * (HIDDEN // 2) + Int64(crank) * (K_CTA // 2)
                sf_row = x_sf + Int64(tok) * (HIDDEN // 16) + Int64(crank) * (K_CTA // 16)
                for j in cutlass.range_constexpr(12):
                    k2 = g4 + 4 * j
                    v4 = P.ld_cg_v4_u32(fp4_row + Int64(k2) * 16)
                    C.st_shared_v4(x_dst + k2 * 128 + n * 16, v4)
                for j in cutlass.range_constexpr(6):
                    k4 = g4 + 4 * j
                    w = Uint32(P.ld_cg_s32(sf_row + Int64(k4) * 4))
                    C.st_shared_u32(xsf_dst + k4 * 128 + n * 16, w)
            else:
                for j in cutlass.range_constexpr(12):
                    k2 = g4 + 4 * j
                    C.st_shared_v4(x_dst + k2 * 128 + n * 16, zero4)
                for j in cutlass.range_constexpr(6):
                    k4 = g4 + 4 * j
                    C.st_shared_u32(xsf_dst + k4 * 128 + n * 16, Uint32(0))

            C.fence_proxy_async_shared_cta()
            cute.arch.sync_warp()
            if lane == 0:
                C.mbarrier_arrive(misc + B_XREADY + 8 * xb)

    # ------------------------------------------------------------ warps 2..5: epilogue
    @cute.jit
    def epilogue_warps(
        self,
        g1a,
        g2a,
        gs2,
        out_fp32,
        base,
        misc,
        tm,
        crank,
        cl,
        nclusters,
        nblocks,
        nunits,
        et,
        lane,
    ):
        qd = (et >> 5) ^ 2
        row = qd * 32 + lane
        lane_base = tm + Int32(32 * qd) * 65536
        recv = base + OFF_RECV

        nex = Int32(0)
        for blk in cutlass.range(nblocks, unroll=1):
            ti = (cl + blk * nclusters) // self.q_groups
            t4 = P.ld_shared_s32(base + OFF_TILE_TOK + (ti * 8 + 4) * 4)
            per_q = Int32(1)
            if t4 >= 0:
                per_q = Int32(2)
            nex = nex + per_q * self.q_acc

        ex = Int32(0)
        for k in cutlass.range(nunits + LOOKAHEAD, unroll=1):
            if k < nunits:
                b = k & 1
                tk, _, _, _ = self.unit_tile_q(cl, nclusters, k)
                ek = P.ld_shared_s32(base + OFF_TILE_E + tk * 4)
                t4 = P.ld_shared_s32(base + OFF_TILE_TOK + (tk * 8 + 4) * 4)
                ng = Int32(1)
                if t4 >= 0:
                    ng = Int32(2)
                if et == 0:
                    C.mbarrier_arrive_expect_tx(
                        misc + B_PRECV + 8 * (ex % RECV_BUFS),
                        3 * 128 * 16,
                        ".relaxed.cluster",
                    )
                    if ng == 2:
                        C.mbarrier_arrive_expect_tx(
                            misc + B_PRECV + 8 * ((ex + 1) % RECV_BUFS),
                            3 * 128 * 16,
                            ".relaxed.cluster",
                        )
                C.mbarrier_wait_parity(misc + B_ACC1_FULL + 8 * b, (k >> 1) & 1)
                T.tc_fence_after()
                va, vb = T.tmem_load4_pair(
                    lane_base + T_ACC1 + 8 * b,
                    lane_base + T_ACC1 + 8 * b + 4,
                    Int32(ng == 2),
                )
                T.tc_fence_before()
                cute.arch.sync_warp()
                if lane == 0:
                    C.mbarrier_arrive(misc + B_ACC1_EMPTY + 8 * b)

                g1 = P.ld_cg_f32(g1a + Int64(ek) * 4)
                for gi in cutlass.range(ng, unroll=1):
                    v0, v1, v2, v3 = va
                    if gi != 0:
                        v0, v1, v2, v3 = vb
                    e = ex + gi
                    p = e % RECV_BUFS
                    use = e // RECV_BUFS
                    if use >= 1:
                        C.mbarrier_wait_parity(
                            misc + B_PFREE + 8 * p, (use - 1) & 1, ".acquire.cluster"
                        )
                    lbar = misc + B_PRECV + 8 * p
                    for pr in cutlass.range_constexpr(CLUSTER):
                        if crank != pr:
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
                    cute.arch.barrier(barrier_id=3 + qd, number_of_threads=32)
                    if (lane == 0) & (e + RECV_BUFS < nex):
                        C.mbarrier_arrive(misc + B_CONSUMED + 8 * p)
                    self.routed_h(
                        acc,
                        g1,
                        gs2,
                        REQ_ROWS * gi,
                        base + OFF_H + H_BUF * (k % H_BUFS),
                        qd,
                        lane,
                    )
                ex = ex + ng
                C.fence_proxy_async_shared_cta()
                cute.arch.sync_warp()
                cute.arch.barrier(barrier_id=3 + qd, number_of_threads=32)
                if lane == 0:
                    C.mbarrier_arrive(misc + B_HREADY + 8 * (k % H_BUFS))

            if (k >= LOOKAHEAD) & (k - LOOKAHEAD < nunits):
                i = k - LOOKAHEAD
                iblk = i // self.q_acc
                iql = i % self.q_acc
                if iql == self.q_acc - 1:
                    ab = iblk & 1
                    C.mbarrier_wait_parity(misc + B_ACC2_FULL + 8 * ab, (iblk >> 1) & 1)
                    T.tc_fence_after()
                    C.fence_proxy_async_shared_cta()
                    acc2 = lane_base + T_ACC2_A
                    if ab != 0:
                        acc2 = lane_base + T_ACC2_B
                    ti, _, _, _ = self.unit_tile_q(cl, nclusters, i)
                    ei = P.ld_shared_s32(base + OFF_TILE_E + ti * 4)
                    g2 = P.ld_cg_f32(g2a + Int64(ei) * 4)
                    t4 = P.ld_shared_s32(base + OFF_TILE_TOK + (ti * 8 + 4) * 4)
                    ng = Int32(1)
                    if t4 >= 0:
                        ng = Int32(2)
                    for gi in cutlass.range(ng, unroll=1):
                        d = T.tmem_load4x12(acc2 + 4 * gi)
                        if gi == ng - 1:
                            T.tc_fence_before()
                            cute.arch.sync_warp()
                            if lane == 0:
                                C.mbarrier_arrive(misc + B_ACC2_EMPTY + 8 * ab)
                        for n in cutlass.range_constexpr(REQ_ROWS):
                            idx = ti * 8 + 4 * gi + n
                            tok = P.ld_shared_s32(base + OFF_TILE_TOK + idx * 4)
                            if tok >= 0:
                                w = C.ld_shared_f32(base + OFF_TILE_W + idx * 4)
                                s = P.mul_f32(w, g2)
                                q = (
                                    out_fp32
                                    + (Int64(tok) * HIDDEN + K_CTA * crank + row) * 4
                                )
                                for t in cutlass.range_constexpr(FC2_TILES):
                                    P.red_global_add_f32(
                                        q + 512 * t,
                                        P.mul_f32(s, d[4 * t + n]),
                                    )

    # ---------------------------------------------------------------------- entry points
    @cute.jit
    def __call__(
        self,
        w13: Int64,
        x_fp4: Int64,
        x_sf: Int64,
        ids: Int64,
        topw: Int64,
        g1a: Int64,
        g2a: Int64,
        gs2: Float32,
        rows: Int32,
        out_fp32: Int64,
        sf13: Int64,
        fc2pack: Int64,
        nclusters: Int32,
        stream: cuda.CUstream,
    ):
        tmap = create_tensor_map_tiled(
            w13,
            Uint8,
            [128, NUM_EXPERTS * self.fc1_rows, 24],
            [3072 // 16, 128 // 16],
            [128, 128, 2],
            swizzle=TensorMapSwizzle.s128b,
            l2_promotion=TensorMapL2Promotion.l2_256b,
        )
        self.kernel(
            tmap,
            x_fp4,
            x_sf,
            ids,
            topw,
            g1a,
            g2a,
            gs2,
            rows,
            out_fp32,
            sf13,
            fc2pack,
            nclusters,
        ).launch(
            grid=[CLUSTER * nclusters, 1, 1],
            block=[THREADS, 1, 1],
            cluster=[CLUSTER, 1, 1],
            max_number_threads=[THREADS, 1, 1],
            min_blocks_per_mp=1,
            stream=stream,
            use_pdl=True,
        )

    @cute.kernel
    def kernel(
        self,
        tmap: cutlass.GridConstant[TensorMap],
        x_fp4: Int64,
        x_sf: Int64,
        ids: Int64,
        topw: Int64,
        g1a: Int64,
        g2a: Int64,
        gs2: Float32,
        rows: Int32,
        out_fp32: Int64,
        sf13: Int64,
        fc2pack: Int64,
        nclusters: Int32,
    ):
        tid, _, _ = cute.arch.thread_idx()
        block, _, _ = cute.arch.block_idx()
        warp = tid >> 5
        lane = tid & 31
        crank = Int32(cute.arch.block_idx_in_cluster())
        cl = block // CLUSTER
        P.trap_if(Int32(cute.arch.cluster_size() != CLUSTER))
        C.pdl_launch_dependents()

        sm = SmemAllocator().allocate_tensor(Uint8, cute.make_layout(SMEM_BYTES), 1024)
        base = Int32(sm.iterator.toint())
        misc = base + OFF_MISC

        # Init barriers, TMEM, zero XSF & H BEFORE pdl_wait -> overlaps with routing!
        if tid == 0:
            for s in cutlass.range_constexpr(RING):
                C.mbarrier_init(misc + B_FULL + 8 * s, 1)
                C.mbarrier_init(misc + B_EMPTY + 8 * s, 1)
                C.mbarrier_init(misc + B_SFA_FULL + 8 * s, 1)
            for s in cutlass.range_constexpr(X_SLOTS):
                C.mbarrier_init(misc + B_XREADY + 8 * s, 1)
                C.mbarrier_init(misc + B_XFREE + 8 * s, 1)
                C.mbarrier_init(misc + B_SFX_READY + 8 * s, 1)
            C.mbarrier_init(misc + B_SFX_FREE, 1)
            for s in cutlass.range_constexpr(2):
                C.mbarrier_init(misc + B_ACC1_FULL + 8 * s, 1)
                C.mbarrier_init(misc + B_ACC1_EMPTY + 8 * s, 4)
                C.mbarrier_init(misc + B_ACC2_FULL + 8 * s, 1)
                C.mbarrier_init(misc + B_ACC2_EMPTY + 8 * s, 4)
            for s in cutlass.range_constexpr(H_BUFS):
                C.mbarrier_init(misc + B_HREADY + 8 * s, 4)
                C.mbarrier_init(misc + B_SFH_READY + 8 * s, 1)
            for s in cutlass.range_constexpr(RECV_BUFS):
                C.mbarrier_init(misc + B_PRECV + 8 * s, 1)
                C.mbarrier_init(misc + B_CONSUMED + 8 * s, 4)
                C.mbarrier_init(misc + B_PFREE + 8 * s, CLUSTER - 1)
            C.fence_mbarrier_init_release_cluster()
        if warp == 1:
            T.tmem_alloc(misc + M_TMEM, 512)
        if warp == 8:
            zero = (Uint32(0), Uint32(0), Uint32(0), Uint32(0))
            for s in cutlass.range_constexpr(X_SLOTS):
                xsf_base = base + OFF_X + s * X_SLOT_STRIDE + X_BYTES
                for i in cutlass.range(lane, XSF_BYTES // 16, 32, unroll=1):
                    C.st_shared_v4(xsf_base + 16 * i, zero)
        if (warp >= 2) & (warp <= 5):
            zero = (Uint32(0), Uint32(0), Uint32(0), Uint32(0))
            et = tid - 64
            for i in cutlass.range(et, H_BUFS * H_BUF // 16, 128, unroll=1):
                C.st_shared_v4(base + OFF_H + 16 * i, zero)

        T.tc_fence_before()
        cute.arch.sync_threads()
        T.tc_fence_after()

        # Wait for MoeRouteTop8(multi_cta=True) + pack tiles directly into SMEM
        C.pdl_wait()
        self.pack_in_smem(ids, topw, rows, base, misc, cl, nclusters, tid, warp, lane)

        cute.arch.cluster_arrive_relaxed(aligned=True)
        cute.arch.cluster_wait()

        tm = Int32(C.ld_shared_u32(misc + M_TMEM))
        nblocks = Int32(C.ld_shared_u32(misc + M_NBLOCKS))
        nunits = Int32(C.ld_shared_u32(misc + M_NUNITS))

        if warp == 0:
            self.producer_warp(
                tmap.get_ptr().toint(),
                sf13,
                fc2pack,
                base,
                misc,
                crank,
                cl,
                nclusters,
                nunits,
                lane,
            )
        elif warp == 1:
            self.mma_warp(base, misc, tm, nunits)
        elif (warp >= 2) & (warp <= 5):
            self.epilogue_warps(
                g1a,
                g2a,
                gs2,
                out_fp32,
                base,
                misc,
                tm,
                crank,
                cl,
                nclusters,
                nblocks,
                nunits,
                tid - 64,
                lane,
            )
        elif warp == 6:
            self.sf_warp(base, misc, tm, nunits)
        elif warp == 7:
            self.release_warp(base, misc, crank, cl, nclusters, nblocks, lane)
        elif warp == 8:
            self.x_gather_warp(
                x_fp4, x_sf, base, misc, crank, cl, nclusters, nblocks, lane
            )

        T.tc_fence_before()
        cute.arch.sync_threads()
        T.tc_fence_after()
        if warp == 1:
            T.tmem_free(tm, 512)
        cute.arch.cluster_arrive(aligned=True)
        cute.arch.cluster_wait()
