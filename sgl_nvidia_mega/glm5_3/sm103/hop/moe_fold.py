# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""MoE finalize folded into the next all-reduce hop (+ residual + RMSNorm).

The MoE engine leaves its routed outputs unreduced (finalize deferred):
``pq[4][rows][8][H]`` fp32 quarter partials per (row, slot), ``shp[4][rows][H]`` fp32
shared-expert quarters, and ``ids`` / ``w`` / ``g2s`` ``[rows][8]`` (``ids < 0``:
dropped slot). This hop computes, per element, exactly the finalize of the engine and
publishes the result as its partial:

    y_k = bf16(((q0 + q1) + q2) + q3) * g2s_k)        (0 for a dropped slot)
    s   = fma(w_k, y_k, s) over k = 0..7 ascending      (dropped slots skipped)
    out = bf16(bf16(s) + bf16(((sh0 + sh1) + sh2) + sh3))

then all-reduces it like ``allreduce_norm_hop`` (residual + RMSNorm). Part 1 (the 512
polling threads, before the epoch barrier) loads every input of its (slot, 16-byte
chunk) pairs before using any and writes ``y`` to shared memory; part 2 (the 128
publishers, after the barrier) runs the ordered fma. The epoch / acknowledgement control
loads are issued by the epoch warp so they overlap the fp32 loads.

``split`` form: the grid adds one helper CTA per (row, cluster rank) that folds and
publishes the second half of the slice's columns (the main CTA the first half), so each
CTA streams half the fp32 inputs; the main CTA still consumes the whole slice. Every CTA
must be co-resident (the main CTAs wait on records written by helper CTAs of the same
grid), so the split form is used only where the cooperative occupancy allows 2 * rows +
1 clusters.
"""

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import cutlass.utils
from cutlass import Float32, Int32, Int64, Uint32

from ..primitives import ptx_core as C
from ..primitives import ptx_sys as sys_ops
from . import _ptx_local as P
from .constants import (
    BF16_NAN,
    CONSUMER_THREADS,
    ELEMS_PER_CTA,
    HIDDEN,
    POLL_THREADS,
    TAG_MASK,
    TOPK,
)
from .transport import ConsSmem, Consumer, Mbox

CHUNKS = ELEMS_PER_CTA // 4  # 16-byte fp32 chunks of one CTA's 768 elements (192)


class MoeFoldArNorm(Consumer):
    """Fold of the MoE finalize + all-reduce + residual + RMSNorm for up to ``max_rows``
    rows."""

    def __init__(self, *, split: bool, max_rows: int = 8):
        if max_rows != 8:
            raise ValueError("this fold is provided for up to 8 rows")
        self.split, self.max_rows = split, max_rows

    @cute.jit
    def __call__(
        self,
        pq: Int64,
        shp: Int64,
        ids: Int64,
        wfin: Int64,
        g2s: Int64,
        residual: Int64,
        gamma: Int64,
        out_res: Int64,
        out_norm: Int64,
        eps: Float32,
        rows: Int32,
        p0: Int64,
        p1: Int64,
        p2: Int64,
        p3: Int64,
        p4: Int64,
        p5: Int64,
        p6: Int64,
        p7: Int64,
        rank: Int32,
        err_host: Int64,
        site: Int32,
        stream: cuda.CUstream,
    ):
        nct = rows * 8 + 8
        if cutlass.const_expr(self.split):
            nct = rows * 16 + 8
        self.kernel(
            pq,
            shp,
            ids,
            wfin,
            g2s,
            residual,
            gamma,
            out_res,
            out_norm,
            eps,
            rows,
            p0,
            p1,
            p2,
            p3,
            p4,
            p5,
            p6,
            p7,
            rank,
            err_host,
            site,
        ).launch(
            grid=[nct, 1, 1],
            block=[CONSUMER_THREADS, 1, 1],
            max_number_threads=[CONSUMER_THREADS, 1, 1],
            cluster=[8, 1, 1],
            cooperative=True,
            use_pdl=True,
            stream=stream,
        )

    def slot_order(self, kk):
        """Slot of the ``kk``-th term of the ordered fma (ascending)."""
        return kk

    @cute.kernel
    def kernel(
        self,
        pq: Int64,
        shp: Int64,
        ids: Int64,
        wfin: Int64,
        g2s: Int64,
        residual: Int64,
        gamma: Int64,
        out_res: Int64,
        out_norm: Int64,
        eps: Float32,
        rows: Int32,
        p0: Int64,
        p1: Int64,
        p2: Int64,
        p3: Int64,
        p4: Int64,
        p5: Int64,
        p6: Int64,
        p7: Int64,
        rank: Int32,
        err_host: Int64,
        site: Int32,
    ):
        self.fold_body(
            pq,
            shp,
            ids,
            wfin,
            g2s,
            residual,
            gamma,
            out_res,
            out_norm,
            eps,
            rows,
            (p0, p1, p2, p3, p4, p5, p6, p7),
            rank,
            err_host,
            site,
        )

    @cute.jit
    def fold_body(
        self,
        pq,
        shp,
        ids,
        wfin,
        g2s,
        residual,
        gamma,
        out_res,
        out_norm,
        eps,
        rows,
        peers,
        rank,
        err_host,
        site,
        dn_bsum=None,
    ):
        """The fold kernel's body; ``dn_bsum`` selects the deferred-norm consumer
        epilogue (``Consumer.consume_cta``)."""
        mb = Mbox(peers, rank, self.max_rows, err_host, site)
        t = P.thread_index()
        bx = P.block_index()
        smem = cutlass.utils.SmemAllocator()
        cs = ConsSmem.allocate(smem, self.max_rows)
        fy = smem.allocate_tensor(
            Float32,
            cute.make_layout((TOPK, ELEMS_PER_CTA), stride=(ELEMS_PER_CTA, 1)),
            16,
        )
        fsh = smem.allocate_tensor(Float32, cute.make_layout(ELEMS_PER_CTA), 16)
        nmain = rows * 8
        is_ack = (bx >= nmain) and (bx < nmain + 8)
        is_helper = Int32(0)
        if cutlass.const_expr(self.split):  # noqa: SIM102
            if bx >= nmain + 8:
                is_helper = Int32(1)
        if bx < nmain:
            self.cons_prologue(cs)
        cute.arch.griddepcontrol_wait()
        cute.arch.griddepcontrol_launch_dependents()
        # the engine's outputs may come from an async-proxy writer
        sys_ops.fence_proxy_async_global()
        if is_ack:
            e = self.epoch_begin(mb, cs.es, False)
            if t == 0:
                self.epoch_arrive(mb, cs.es, rows, P.grid_size())
                if bx == nmain:
                    self.ack_cta(mb, e)
            sys_ops.exit_thread()
        bxm = bx
        if is_helper != 0:
            bxm = bx - nmain - 8
        m = bxm // 8
        c = bxm % 8
        h0 = c * 768 + 6 * t
        ru = cute.make_rmem_tensor((3,), Uint32)
        gu = cute.make_rmem_tensor((3,), Uint32)
        ru.fill(0)
        gu.fill(0)
        if t < 128 and is_helper == 0:
            off = Int64(2) * (Int64(m) * HIDDEN + Int64(h0))
            for k in cutlass.range_constexpr(3):
                ru[k] = sys_ops.ld_cg_u32(residual + off + Int64(4 * k))
            for k in cutlass.range_constexpr(3):
                gu[k] = sys_ops.ld_cg_u32(gamma + Int64(2) * Int64(h0) + Int64(4 * k))
        fik = cute.make_rmem_tensor((TOPK,), Int32)
        fwk = cute.make_rmem_tensor((TOPK,), Float32)
        fik.fill(0)
        fwk.fill(0.0)
        col = c * ELEMS_PER_CTA
        if cutlass.const_expr(self.split):
            self.fold_part1_half(pq, shp, ids, g2s, fy, fsh, rows, m, col, is_helper)
            npub = 64
        else:
            self.fold_part1(pq, shp, ids, g2s, fy, fsh, rows, m, col)
            npub = 128
        if t < npub:
            for k in cutlass.range_constexpr(TOPK):
                fik[k] = sys_ops.ld_cg_s32(ids + Int64(4 * (m * TOPK + k)))
                fwk[k] = P.bits_f32(sys_ops.ld_cg_u32(wfin + Int64(4 * (m * TOPK + k))))
        # the epoch warp reads the controls
        self.epoch_begin(mb, cs.es, True, POLL_THREADS)
        pz = cs.es.pz[0]
        em = cs.es.erow[m]
        buf = self.buffer_of(em)
        tag = em & Uint32(TAG_MASK)
        if t < npub:
            tp = t
            if cutlass.const_expr(self.split):
                tp = t + 64 * is_helper
            w = self.fold_part2(fy, fsh, fik, fwk, tp)
            for k in cutlass.range_constexpr(3):
                if pz != 0:
                    w[k] = Uint32(BF16_NAN | (BF16_NAN << 16))
                else:
                    w[k] = P.no_neg_zero(w[k])
            r0, r1 = self.pack_records(w[0], w[1], w[2], tag)
            self.delay(mb, 1)
            self.publish_records(mb, buf, m, c * 256 + 2 * tp, r0, r1)
        if cutlass.const_expr(self.split):  # noqa: SIM102
            if is_helper != 0:  # helper: publish only
                if t == POLL_THREADS:
                    self.epoch_arrive(mb, cs.es, rows, P.grid_size())
                sys_ops.exit_thread()
        if t >= POLL_THREADS:  # the epoch warp arrives, then leaves the CTA
            if t == POLL_THREADS:
                self.epoch_arrive(mb, cs.es, rows, P.grid_size())
            sys_ops.exit_thread()
        self.consume_cta(
            mb, cs, m, c, em, pz, ru, gu, out_res, out_norm, eps, True, True, dn_bsum
        )

    @cute.jit
    def fold_part1(
        self, pq: Int64, shp: Int64, ids: Int64, g2s: Int64, fy, fsh, rows, m, col
    ):
        """Part 1, plain form: thread t handles (slot, chunk) pairs p = t + 512 i, i <
        3; the last 192 polling threads also sum the shared-expert quarters. Every load
        before any use."""
        t = P.thread_index()
        if t < POLL_THREADS:
            v = cute.make_rmem_tensor((3, 4, 4), Float32)
            ik = cute.make_rmem_tensor((3,), Int32)
            gk = cute.make_rmem_tensor((3,), Float32)
            for i in cutlass.range_constexpr(3):
                p = t + POLL_THREADS * i
                k = p // CHUNKS
                ch = p - k * CHUNKS
                ik[i] = sys_ops.ld_cg_s32(ids + Int64(4 * (m * TOPK + k)))
                gk[i] = P.bits_f32(sys_ops.ld_cg_u32(g2s + Int64(4 * (m * TOPK + k))))
                for q in cutlass.range_constexpr(4):
                    a = pq + Int64(4) * (
                        ((Int64(q) * Int64(rows) + Int64(m)) * TOPK + Int64(k)) * HIDDEN
                        + Int64(col + 4 * ch)
                    )
                    x0, x1, x2, x3 = sys_ops.ld_cg_v4_f32(a)
                    v[i, q, 0] = x0
                    v[i, q, 1] = x1
                    v[i, q, 2] = x2
                    v[i, q, 3] = x3
            sv = cute.make_rmem_tensor((4, 4), Float32)
            sv.fill(0.0)
            if t >= POLL_THREADS - CHUNKS:
                ch = t - (POLL_THREADS - CHUNKS)
                for q in cutlass.range_constexpr(4):
                    a = shp + Int64(4) * (
                        (Int64(q) * Int64(rows) + Int64(m)) * HIDDEN
                        + Int64(col + 4 * ch)
                    )
                    x0, x1, x2, x3 = sys_ops.ld_cg_v4_f32(a)
                    sv[q, 0] = x0
                    sv[q, 1] = x1
                    sv[q, 2] = x2
                    sv[q, 3] = x3
            for i in cutlass.range_constexpr(3):
                p = t + POLL_THREADS * i
                k = p // CHUNKS
                ch = p - k * CHUNKS
                self.store_y(fy, v, i, ik[i], gk[i], k, ch)
            if t >= POLL_THREADS - CHUNKS:
                self.store_sh(fsh, sv, t - (POLL_THREADS - CHUNKS))

    @cute.jit
    def fold_part1_half(
        self, pq: Int64, shp: Int64, ids: Int64, g2s: Int64, fy, fsh, rows, m, col, half
    ):
        """Part 1, split form: half ``half`` of the slice (chunks [96 half, 96 half +
        96)); pairs p = t + 512 i, i < 2, p < 768; the last 96 polling threads sum the
        shared quarters."""
        t = P.thread_index()
        hch: cutlass.Constexpr = CHUNKS // 2
        if t < POLL_THREADS:
            v = cute.make_rmem_tensor((2, 4, 4), Float32)
            ik = cute.make_rmem_tensor((2,), Int32)
            gk = cute.make_rmem_tensor((2,), Float32)
            v.fill(0.0)
            ik.fill(0)
            gk.fill(0.0)
            for i in cutlass.range_constexpr(2):
                p = t + POLL_THREADS * i
                if p < TOPK * hch:
                    k = p // hch
                    ch = hch * half + (p - k * hch)
                    ik[i] = sys_ops.ld_cg_s32(ids + Int64(4 * (m * TOPK + k)))
                    gk[i] = P.bits_f32(
                        sys_ops.ld_cg_u32(g2s + Int64(4 * (m * TOPK + k)))
                    )
                    for q in cutlass.range_constexpr(4):
                        a = pq + Int64(4) * (
                            ((Int64(q) * Int64(rows) + Int64(m)) * TOPK + Int64(k))
                            * HIDDEN
                            + Int64(col + 4 * ch)
                        )
                        x0, x1, x2, x3 = sys_ops.ld_cg_v4_f32(a)
                        v[i, q, 0] = x0
                        v[i, q, 1] = x1
                        v[i, q, 2] = x2
                        v[i, q, 3] = x3
            sv = cute.make_rmem_tensor((4, 4), Float32)
            sv.fill(0.0)
            if t >= POLL_THREADS - hch:
                ch = hch * half + (t - (POLL_THREADS - hch))
                for q in cutlass.range_constexpr(4):
                    a = shp + Int64(4) * (
                        (Int64(q) * Int64(rows) + Int64(m)) * HIDDEN
                        + Int64(col + 4 * ch)
                    )
                    x0, x1, x2, x3 = sys_ops.ld_cg_v4_f32(a)
                    sv[q, 0] = x0
                    sv[q, 1] = x1
                    sv[q, 2] = x2
                    sv[q, 3] = x3
            for i in cutlass.range_constexpr(2):
                p = t + POLL_THREADS * i
                if p < TOPK * hch:
                    k = p // hch
                    ch = hch * half + (p - k * hch)
                    self.store_y(fy, v, i, ik[i], gk[i], k, ch)
            if t >= POLL_THREADS - hch:
                self.store_sh(fsh, sv, hch * half + (t - (POLL_THREADS - hch)))

    @cute.jit
    def store_y(self, fy, v, i, ik, gk, k, ch):
        """``y = bf16((((q0 + q1) + q2) + q3) * g)`` per lane (0 for a dropped slot)
        into ``fy[k]``."""
        px = C.mul_rn(
            C.add_rn(
                C.add_rn(C.add_rn(v[i, 0, 0], v[i, 1, 0]), v[i, 2, 0]), v[i, 3, 0]
            ),
            gk,
        )
        py = C.mul_rn(
            C.add_rn(
                C.add_rn(C.add_rn(v[i, 0, 1], v[i, 1, 1]), v[i, 2, 1]), v[i, 3, 1]
            ),
            gk,
        )
        pz = C.mul_rn(
            C.add_rn(
                C.add_rn(C.add_rn(v[i, 0, 2], v[i, 1, 2]), v[i, 2, 2]), v[i, 3, 2]
            ),
            gk,
        )
        pw = C.mul_rn(
            C.add_rn(
                C.add_rn(C.add_rn(v[i, 0, 3], v[i, 1, 3]), v[i, 2, 3]), v[i, 3, 3]
            ),
            gk,
        )
        w0 = C.cvt_rn_bf16x2_f32(px, py)
        w1 = C.cvt_rn_bf16x2_f32(pz, pw)
        y0 = P.bf16_lo(w0)
        y1 = P.bf16_hi(w0)
        y2 = P.bf16_lo(w1)
        y3 = P.bf16_hi(w1)
        if ik < 0:
            y0 = Float32(0.0)
            y1 = Float32(0.0)
            y2 = Float32(0.0)
            y3 = Float32(0.0)
        P.st_shared_v4_u32(
            Uint32((fy.iterator + (k * ELEMS_PER_CTA + 4 * ch)).toint()),
            P.f32_bits(y0),
            P.f32_bits(y1),
            P.f32_bits(y2),
            P.f32_bits(y3),
        )

    @cute.jit
    def store_sh(self, fsh, sv, ch):
        """Shared-expert chunk ``((sh0 + sh1) + sh2) + sh3`` (fp32, before its bf16
        rounding)."""
        s0 = C.add_rn(C.add_rn(C.add_rn(sv[0, 0], sv[1, 0]), sv[2, 0]), sv[3, 0])
        s1 = C.add_rn(C.add_rn(C.add_rn(sv[0, 1], sv[1, 1]), sv[2, 1]), sv[3, 1])
        s2 = C.add_rn(C.add_rn(C.add_rn(sv[0, 2], sv[1, 2]), sv[2, 2]), sv[3, 2])
        s3 = C.add_rn(C.add_rn(C.add_rn(sv[0, 3], sv[1, 3]), sv[2, 3]), sv[3, 3])
        P.st_shared_v4_u32(
            Uint32((fsh.iterator + 4 * ch).toint()),
            P.f32_bits(s0),
            P.f32_bits(s1),
            P.f32_bits(s2),
            P.f32_bits(s3),
        )

    @cute.jit
    def fold_part2(self, fy, fsh, fik, fwk, tp):
        """Part 2 for publisher slot ``tp`` (6 elements): the ordered fma over the
        slots, then ``bf16(bf16(routed) + bf16(shared))``; returns three bf16x2
        words."""
        sa = cute.make_rmem_tensor((6,), Float32)
        sh = cute.make_rmem_tensor((6,), Float32)
        for x in cutlass.range_constexpr(6):
            acc = Float32(0.0)
            for kk in cutlass.range_constexpr(TOPK):
                k: cutlass.Constexpr = self.slot_order(kk)
                if fik[k] >= 0:
                    acc = C.fma_rn(fwk[k], fy[k, 6 * tp + x], acc)
            sa[x] = acc
            sh[x] = fsh[6 * tp + x]
        w = cute.make_rmem_tensor((3,), Uint32)
        for k in cutlass.range_constexpr(3):
            ra = C.cvt_rn_bf16x2_f32(sa[2 * k], sa[2 * k + 1])
            rs = C.cvt_rn_bf16x2_f32(sh[2 * k], sh[2 * k + 1])
            o0 = C.add_rn(P.bf16_lo(ra), P.bf16_lo(rs))
            o1 = C.add_rn(P.bf16_hi(ra), P.bf16_hi(rs))
            w[k] = C.cvt_rn_bf16x2_f32(o0, o1)
        return w


class MoeFoldArNormDn(MoeFoldArNorm):
    """The fold for up to 8 rows with the attention input's RMSNorm deferred to the
    consumer of its projection: fold + all-reduce + residual as ``MoeFoldArNorm``, then
    ``u = bf16(x * (0 + gamma))`` in place of the normed output and each CTA's block sum
    of squares to ``bsum[rows][8]`` (the norm's own value and order); the cluster does
    not exchange the block sums. ``inv = rsqrt(fma(sum_c bsum[m][c], rcp(HIDDEN), eps))``
    (sum from 0 in ``c`` order) gives the normed row as ``(x * inv) * gamma``."""

    @cute.jit
    def __call__(
        self,
        pq: Int64,
        shp: Int64,
        ids: Int64,
        wfin: Int64,
        g2s: Int64,
        residual: Int64,
        gamma: Int64,
        out_res: Int64,
        out_u: Int64,
        bsum: Int64,
        rows: Int32,
        p0: Int64,
        p1: Int64,
        p2: Int64,
        p3: Int64,
        p4: Int64,
        p5: Int64,
        p6: Int64,
        p7: Int64,
        rank: Int32,
        err_host: Int64,
        site: Int32,
        stream: cuda.CUstream,
    ):
        nct = rows * 8 + 8
        if cutlass.const_expr(self.split):
            nct = rows * 16 + 8
        self.kernel_dn(
            pq,
            shp,
            ids,
            wfin,
            g2s,
            residual,
            gamma,
            out_res,
            out_u,
            bsum,
            rows,
            p0,
            p1,
            p2,
            p3,
            p4,
            p5,
            p6,
            p7,
            rank,
            err_host,
            site,
        ).launch(
            grid=[nct, 1, 1],
            block=[CONSUMER_THREADS, 1, 1],
            max_number_threads=[CONSUMER_THREADS, 1, 1],
            cluster=[8, 1, 1],
            cooperative=True,
            use_pdl=True,
            stream=stream,
        )

    @cute.kernel
    def kernel_dn(
        self,
        pq: Int64,
        shp: Int64,
        ids: Int64,
        wfin: Int64,
        g2s: Int64,
        residual: Int64,
        gamma: Int64,
        out_res: Int64,
        out_u: Int64,
        bsum: Int64,
        rows: Int32,
        p0: Int64,
        p1: Int64,
        p2: Int64,
        p3: Int64,
        p4: Int64,
        p5: Int64,
        p6: Int64,
        p7: Int64,
        rank: Int32,
        err_host: Int64,
        site: Int32,
    ):
        self.fold_body(
            pq,
            shp,
            ids,
            wfin,
            g2s,
            residual,
            gamma,
            out_res,
            out_u,
            Float32(0.0),
            rows,
            (p0, p1, p2, p3, p4, p5, p6, p7),
            rank,
            err_host,
            site,
            bsum,
        )


class MoeFoldArBsum(MoeFoldArNorm):
    """The fold for 9..16 rows with the RMSNorm deferred: fold + all-reduce + residual,
    then each CTA's block sum of squares to ``bsum[rows][8]``; ``RmsnormFromBlockSums``
    applies the norm. Together they equal the normed fold bitwise. With no cluster
    exchange there is no cluster launch: one cooperative grid of ``rows * 8`` CTAs plus
    one acknowledgement CTA."""

    def __init__(self, *, max_rows: int = 16):
        if max_rows != 16:
            raise ValueError("the block-sum fold is provided for up to 16 rows")
        self.split, self.max_rows = False, max_rows

    @cute.jit
    def __call__(
        self,
        pq: Int64,
        shp: Int64,
        ids: Int64,
        wfin: Int64,
        g2s: Int64,
        residual: Int64,
        out_res: Int64,
        bsum: Int64,
        rows: Int32,
        p0: Int64,
        p1: Int64,
        p2: Int64,
        p3: Int64,
        p4: Int64,
        p5: Int64,
        p6: Int64,
        p7: Int64,
        rank: Int32,
        err_host: Int64,
        site: Int32,
        stream: cuda.CUstream,
    ):
        self.kernel_bsum(
            pq,
            shp,
            ids,
            wfin,
            g2s,
            residual,
            out_res,
            bsum,
            rows,
            p0,
            p1,
            p2,
            p3,
            p4,
            p5,
            p6,
            p7,
            rank,
            err_host,
            site,
        ).launch(
            grid=[rows * 8 + 1, 1, 1],
            block=[CONSUMER_THREADS, 1, 1],
            max_number_threads=[CONSUMER_THREADS, 1, 1],
            cooperative=True,
            use_pdl=True,
            stream=stream,
        )

    @cute.kernel
    def kernel_bsum(
        self,
        pq: Int64,
        shp: Int64,
        ids: Int64,
        wfin: Int64,
        g2s: Int64,
        residual: Int64,
        out_res: Int64,
        bsum: Int64,
        rows: Int32,
        p0: Int64,
        p1: Int64,
        p2: Int64,
        p3: Int64,
        p4: Int64,
        p5: Int64,
        p6: Int64,
        p7: Int64,
        rank: Int32,
        err_host: Int64,
        site: Int32,
    ):
        mb = Mbox((p0, p1, p2, p3, p4, p5, p6, p7), rank, self.max_rows, err_host, site)
        t = P.thread_index()
        bx = P.block_index()
        smem = cutlass.utils.SmemAllocator()
        cs = ConsSmem.allocate(smem, self.max_rows)
        fy = smem.allocate_tensor(
            Float32,
            cute.make_layout((TOPK, ELEMS_PER_CTA), stride=(ELEMS_PER_CTA, 1)),
            16,
        )
        fsh = smem.allocate_tensor(Float32, cute.make_layout(ELEMS_PER_CTA), 16)
        nmain = rows * 8
        cute.arch.griddepcontrol_wait()
        cute.arch.griddepcontrol_launch_dependents()
        sys_ops.fence_proxy_async_global()
        if bx >= nmain:
            e = self.epoch_begin(mb, cs.es, False)
            if t == 0:
                self.epoch_arrive(mb, cs.es, rows, P.grid_size())
                if bx == nmain:
                    self.ack_cta(mb, e)
            sys_ops.exit_thread()
        m = bx // 8
        c = bx % 8
        h0 = c * 768 + 6 * t
        ru = cute.make_rmem_tensor((3,), Uint32)
        ru.fill(0)
        if t < 128:
            off = Int64(2) * (Int64(m) * HIDDEN + Int64(h0))
            for k in cutlass.range_constexpr(3):
                ru[k] = sys_ops.ld_cg_u32(residual + off + Int64(4 * k))
        fik = cute.make_rmem_tensor((TOPK,), Int32)
        fwk = cute.make_rmem_tensor((TOPK,), Float32)
        fik.fill(0)
        fwk.fill(0.0)
        self.fold_part1(pq, shp, ids, g2s, fy, fsh, rows, m, c * ELEMS_PER_CTA)
        if t < 128:
            for k in cutlass.range_constexpr(TOPK):
                fik[k] = sys_ops.ld_cg_s32(ids + Int64(4 * (m * TOPK + k)))
                fwk[k] = P.bits_f32(sys_ops.ld_cg_u32(wfin + Int64(4 * (m * TOPK + k))))
        self.epoch_begin(mb, cs.es, True, POLL_THREADS)
        pz = cs.es.pz[0]
        em = cs.es.erow[m]
        tag = em & Uint32(TAG_MASK)
        if t < 128:
            w = self.fold_part2(fy, fsh, fik, fwk, t)
            for k in cutlass.range_constexpr(3):
                if pz != 0:
                    w[k] = Uint32(BF16_NAN | (BF16_NAN << 16))
                else:
                    w[k] = P.no_neg_zero(w[k])
            r0, r1 = self.pack_records(w[0], w[1], w[2], tag)
            self.delay(mb, 1)
            self.publish_records(mb, self.buffer_of(em), m, c * 256 + 2 * t, r0, r1)
        if t >= POLL_THREADS:
            if t == POLL_THREADS:
                self.epoch_arrive(mb, cs.es, rows, P.grid_size())
            sys_ops.exit_thread()
        self.consume_cta_bsum(mb, cs, m, c, em, pz, ru, out_res, bsum)


class RmsnormFromBlockSums(Consumer):
    """``y[m] = bf16((x[m] * inv_m) * (0 + gamma))`` with ``inv_m = rsqrt(fma(sum_c
    bsum[m][c], rcp(HIDDEN), eps))``, the sum from 0 in ``c`` order: exactly the normed
    fold's norm. One CTA of 768 threads per row; programmatic dependent of the block-sum
    fold. ``x`` and ``bsum`` come from that fold, which may still run while this grid is
    resident, so they are read with coherent loads; ``gamma`` is static."""

    def __init__(self, *, max_rows: int = 16):
        self.max_rows = max_rows

    @cute.jit
    def __call__(
        self,
        x: Int64,
        bsum: Int64,
        gamma: Int64,
        y: Int64,
        eps: Float32,
        rows: Int32,
        stream: cuda.CUstream,
    ):
        self.kernel(x, bsum, gamma, y, eps).launch(
            grid=[rows, 1, 1],
            block=[768, 1, 1],
            max_number_threads=[768, 1, 1],
            use_pdl=True,
            stream=stream,
        )

    @cute.kernel
    def kernel(self, x: Int64, bsum: Int64, gamma: Int64, y: Int64, eps: Float32):
        cute.arch.griddepcontrol_wait()
        cute.arch.griddepcontrol_launch_dependents()
        sys_ops.fence_proxy_async_global()
        m = P.block_index()
        t = P.thread_index()
        tot = Float32(0.0)
        for cc in cutlass.range_constexpr(8):
            tot = C.add_rn(
                tot,
                P.bits_f32(
                    sys_ops.ld_u32(bsum + Int64(4) * (Int64(m) * 8 + Int64(cc)))
                ),
            )
        inv = C.rsqrt_approx(C.fma_rn(tot, C.rcp_rn(Float32(6144.0)), eps))
        xo = x + Int64(2) * (Int64(m) * HIDDEN + Int64(8 * t))
        go = gamma + Int64(2) * Int64(8 * t)
        yo = y + Int64(2) * (Int64(m) * HIDDEN + Int64(8 * t))
        for k in cutlass.range_constexpr(4):
            xv = sys_ops.ld_u32(xo + Int64(4 * k))
            gv = sys_ops.ld_nc_u32(go + Int64(4 * k))  # static weight
            g0 = C.add_rn(Float32(0.0), P.bf16_lo(gv))
            g1 = C.add_rn(Float32(0.0), P.bf16_hi(gv))
            y0 = C.cvt_rn_bf16_f32(C.mul_rn(C.mul_rn(P.bf16_lo(xv), inv), g0))
            y1 = C.cvt_rn_bf16_f32(C.mul_rn(C.mul_rn(P.bf16_hi(xv), inv), g1))
            sys_ops.st_global_u32(yo + Int64(4 * k), Uint32(y0) | (Uint32(y1) << 16))
        sys_ops.fence_proxy_async_global()
