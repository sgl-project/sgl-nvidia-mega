# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""o_proj all-reduce hop: a producer grid computes this rank's o_proj partial and
publishes it as mailbox records; a consumer grid (its programmatic dependent)
all-reduces the records with the residual add and RMSNorm.

Producer: one synchronization CTA plus 128 GEMV CTAs of 384 threads. GEMV CTA ``c`` owns
output rows ``[48c, 48c + 48)`` (12 warps = 3 row tiles x 4 K quarters); its weight
slice is loaded into registers before ``griddepcontrol.wait`` (static weights only),
then the activations are staged, ``mma.sync.m16n8k16`` accumulates per K quarter (tokens
on n8), and the four quarters are summed in a fixed order.

Router (``router=True``, MoE layers): the router logits are computed through the o_proj
hop. With ``W' = W_router * diag(gamma)`` and ``Q_r = W' W_o^(r)`` (fp32, split into
bf16 hi + lo), ``logits[m] = rstd_m * sum_r (Q_r a_r + W'[:, slice r] residual_m[slice
r])``. Sixteen more producer CTAs compute this rank's term for 16 experts each and
publish ``{fp32, tag32}`` words to every rank's router-partial mailbox; the consumer's
cluster rank 0 of each row gathers the eight terms in rank order, multiplies by the
row's RMSNorm rstd and routes the row (sigmoid scores + bias, top-8, normalized scaled
weights) in one warp.

Start edge (PTX ISA 9.7.15.14): the synchronization CTA executes ``griddepcontrol.wait;
fence.acq_rel.gpu; griddepcontrol.launch_dependents``, so the consumer grid's start is
ordered after every grid this producer depends on, and the consumer reads its residual,
gamma and epochs without waiting at entry. The consumer's last arriving CTA waits for
the producer grid (``griddepcontrol.wait``) before advancing the epoch (a slow producer
CTA may still be reading it), and thread 0 waits at exit, so the consumer completes
after the producer.
"""

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import cutlass.utils
from cutlass import Float32, Int32, Int64, Uint32, Uint64

from ..primitives import ptx_core as C
from ..primitives import ptx_sys as sys_ops
from . import _ptx_local as P
from .constants import (
    BF16_NAN,
    CONSUMER_THREADS,
    ERR_ROUTER_POLL,
    HIDDEN,
    NBUF,
    O_PROJ_IN,
    POLL_THREADS,
    POLL_TIMER_STRIDE,
    TAG_MASK,
    TOPK,
    WORLD,
)
from .transport import ConsSmem, Consumer, EpochSmem, Mbox, Transport

GEMV_ROWS = 48  # output rows per GEMV CTA
GEMV_CTAS = HIDDEN // GEMV_ROWS  # 128
GEMV_THREADS = 384  # 12 warps: 3 row tiles x 4 K quarters
GEMV_STEPS = 16  # 32-wide K steps per quarter (512 / 32)
XS_STRIDE = O_PROJ_IN + 32  # staged activation row stride (bf16)
EXPERTS = 256
ROUTER_CTAS = 16  # 16 experts (one m16 tile) per router CTA
ROUTER_K = O_PROJ_IN + HIDDEN // WORLD  # 2816 = [o_proj input | residual slice]
ROUTER_STEPS = 8  # max 32-wide K steps per warp (warps 0..3: 8, 4..11: 7)
RXS_STRIDE = ROUTER_K + 32
# the model's routed scaling factor (the glue passes the checkpoint value)
ROUTED_SCALE = 2.5


def router_partial_bytes(max_rows: int) -> int:
    """Router-partial mailbox ``[NBUF][8 src][max_rows][256]`` of ``{fp32, tag32}``
    words."""
    return NBUF * WORLD * max_rows * EXPERTS * 8


def _view(base, offset, dtype, shape, stride=None):
    ptr = cute.recast_ptr(base + offset, dtype=dtype)
    return cute.make_tensor(ptr, cute.make_layout(shape, stride=stride))


def _epoch_view(base, offset, max_rows):
    return EpochSmem(
        _view(base, offset, Uint32, 1),
        _view(base, offset + 4, Int32, 1),
        _view(base, offset + 8, Uint32, max_rows),
    )


class OProjArProduce(Transport):
    """Producer of an o_proj hop for up to ``max_rows`` rows (``router``: + the router
    terms)."""

    sync_ctas = 1  # the start-edge synchronization CTA

    def __init__(self, *, router: bool = False, max_rows: int = 8):
        if max_rows not in (8, 16) or (max_rows == 16 and not router):
            raise ValueError(
                "provided for up to 8 rows, or up to 16 rows with the router"
            )
        self.router, self.max_rows = router, max_rows
        # rows 9..16: a second n8 token tile per mma (same A fragments, same k order),
        # and the
        # activations staged by 16-byte cp.async copies (one wait before the epoch
        # barrier)
        self.two_tiles = max_rows == 16
        self.async_stage = max_rows == 16

    def smem_bytes(self):
        gemv = (
            self.max_rows * XS_STRIDE * 2
            + 4 * 3 * 16 * self.max_rows * 4
            + 8
            + 4 * self.max_rows
        )
        if not self.router:
            return gemv
        rt = (
            self.max_rows * RXS_STRIDE * 2
            + 12 * 16 * self.max_rows * 4
            + 8
            + 4 * self.max_rows
        )
        return max(gemv, rt)

    @cute.jit
    def stage16(self, smem_addr: Uint32, src: Int64):
        """Stage 16 bytes of activations into shared memory."""
        if cutlass.const_expr(self.async_stage):
            sys_ops.cp_async_cg_16(smem_addr, src)
        else:
            v = sys_ops.ld_cg_v4_u32(src)
            P.st_shared_v4_u32(smem_addr, v[0], v[1], v[2], v[3])

    @cute.jit
    def stage_done(self):
        if cutlass.const_expr(self.async_stage):
            # this thread's copies landed; the epoch barrier publishes them
            sys_ops.cp_async_wait_all()

    @cute.jit
    def __call__(
        self,
        x: Int64,
        w: Int64,
        qh: Int64,
        ql: Int64,
        residual: Int64,
        rows: Int32,
        p0: Int64,
        p1: Int64,
        p2: Int64,
        p3: Int64,
        p4: Int64,
        p5: Int64,
        p6: Int64,
        p7: Int64,
        q0: Int64,
        q1: Int64,
        q2: Int64,
        q3: Int64,
        q4: Int64,
        q5: Int64,
        q6: Int64,
        q7: Int64,
        rank: Int32,
        err_host: Int64,
        site: Int32,
        stream: cuda.CUstream,
    ):
        """``x`` [rows, 2048] bf16, ``w`` [6144, 2048] bf16 (this rank's o_proj shard).
        Router: ``qh`` / ``ql`` [256, 2816] bf16, ``residual`` [rows, 6144] bf16, ``q0
        .. q7`` the router-partial mailbox bases rotated like ``p0 .. p7`` (unused
        without the router)."""
        nct = (
            GEMV_CTAS
            + self.sync_ctas
            + (ROUTER_CTAS if cutlass.const_expr(self.router) else 0)
        )
        self.kernel(
            x,
            w,
            qh,
            ql,
            residual,
            rows,
            p0,
            p1,
            p2,
            p3,
            p4,
            p5,
            p6,
            p7,
            q0,
            q1,
            q2,
            q3,
            q4,
            q5,
            q6,
            q7,
            rank,
            err_host,
            site,
        ).launch(
            grid=[nct, 1, 1],
            block=[GEMV_THREADS, 1, 1],
            max_number_threads=[GEMV_THREADS, 1, 1],
            min_blocks_per_mp=1,
            use_pdl=True,
            stream=stream,
        )

    @cute.jit
    def sync_cta(self):
        """Start edge: every earlier grid (covered by this thread's wait) happens before
        the dependent grid's start (release fence, then the trigger; PTX ISA
        9.7.15.14)."""
        if P.thread_index() == 0:
            cute.arch.griddepcontrol_wait()
            sys_ops.fence_acq_rel_gpu()
            cute.arch.griddepcontrol_launch_dependents()

    @cute.kernel
    def kernel(
        self,
        x: Int64,
        w: Int64,
        qh: Int64,
        ql: Int64,
        residual: Int64,
        rows: Int32,
        p0: Int64,
        p1: Int64,
        p2: Int64,
        p3: Int64,
        p4: Int64,
        p5: Int64,
        p6: Int64,
        p7: Int64,
        q0: Int64,
        q1: Int64,
        q2: Int64,
        q3: Int64,
        q4: Int64,
        q5: Int64,
        q6: Int64,
        q7: Int64,
        rank: Int32,
        err_host: Int64,
        site: Int32,
    ):
        mb = Mbox((p0, p1, p2, p3, p4, p5, p6, p7), rank, self.max_rows, err_host, site)
        bx = P.block_index()
        smem = cutlass.utils.SmemAllocator()
        raw = smem.allocate_tensor(
            cutlass.Uint8, cute.make_layout(self.smem_bytes()), 16
        )
        base = raw.iterator
        if cutlass.const_expr(self.sync_ctas > 0):  # noqa: SIM102
            if bx < self.sync_ctas:
                self.sync_cta()
                sys_ops.exit_thread()
        cta = bx - self.sync_ctas
        pol = sys_ops.create_policy_evict_first()
        if cutlass.const_expr(self.router):  # noqa: SIM102
            if cta >= GEMV_CTAS:
                self.router_cta(
                    mb,
                    base,
                    cta - GEMV_CTAS,
                    x,
                    qh,
                    ql,
                    residual,
                    rows,
                    (q0, q1, q2, q3, q4, q5, q6, q7),
                    pol,
                )
                sys_ops.exit_thread()
        self.gemv_cta(mb, base, cta, x, w, rows, pol)

    @cute.jit
    def pdl_edge(self, rows: Int32):
        """``griddepcontrol.wait`` then ``launch_dependents`` of a GEMV / router CTA,
        inside a branch on a runtime value (``rows`` >= 1 in every launch)."""
        if rows > Int32(0):
            cute.arch.griddepcontrol_wait()
            cute.arch.griddepcontrol_launch_dependents()

    @cute.jit
    def gemv_cta(
        self, mb: Mbox, base, cta, x: Int64, w: Int64, rows: Int32, pol: Uint64
    ):
        t = P.thread_index()
        xs_bytes: cutlass.Constexpr = self.max_rows * XS_STRIDE * 2
        mx: cutlass.Constexpr = self.max_rows
        red = _view(
            base, xs_bytes, Float32, (4, 3, 16, mx), (3 * 16 * mx, 16 * mx, mx, 1)
        )
        es = _epoch_view(base, xs_bytes + 4 * 3 * 16 * mx * 4, self.max_rows)
        warp = t // 32
        lane = t % 32
        g = lane // 4
        q = lane % 4
        tile = warp % 3
        kq = warp // 3
        row0 = cta * GEMV_ROWS + tile * 16
        wa = w + Int64(2) * (Int64(row0 + g) * O_PROJ_IN + Int64(kq * 512 + 8 * q))
        wb = wa + Int64(2 * 8 * O_PROJ_IN)
        a = cute.make_rmem_tensor((GEMV_STEPS, 4), Uint32)
        b = cute.make_rmem_tensor((GEMV_STEPS, 4), Uint32)
        # static weights: issued before the wait
        for st in cutlass.range_constexpr(GEMV_STEPS):
            v = sys_ops.ld_nc_hint_v4_u32(wa + Int64(64 * st), pol)
            u = sys_ops.ld_nc_hint_v4_u32(wb + Int64(64 * st), pol)
            for k in cutlass.range_constexpr(4):
                a[st, k] = v[k]
                b[st, k] = u[k]
        self.pdl_edge(rows)
        sys_ops.fence_proxy_async_global()  # x may come from an async-proxy writer
        xs_base = Uint32(base.toint())
        i = t
        nchunk = rows * (O_PROJ_IN // 8)
        while i < nchunk:
            mm = i // (O_PROJ_IN // 8)
            kc = i % (O_PROJ_IN // 8)
            self.stage16(
                xs_base + Uint32(2 * (mm * XS_STRIDE + kc * 8)),
                x + Int64(2) * (Int64(mm) * O_PROJ_IN + Int64(kc * 8)),
            )
            i = i + GEMV_THREADS
        self.stage_done()
        if cutlass.const_expr(not self.router):  # noqa: SIM102
            if t < 8:
                # a slow GEMV CTA between its trigger and its read of E
                self.delay(mb, 6)
        self.epoch_begin(mb, es, True)  # its CTA barrier also covers the staging
        pz = es.pz[0]
        c0 = Float32(0.0)
        c1 = Float32(0.0)
        c2 = Float32(0.0)
        c3 = Float32(0.0)
        d0 = Float32(0.0)
        d1 = Float32(0.0)
        d2 = Float32(0.0)
        d3 = Float32(0.0)
        tok = g < rows
        tok2 = g + 8 < rows
        two = rows > 8
        xr = xs_base + Uint32(2 * (g * XS_STRIDE + kq * 512 + 8 * q))
        xr2 = xr + Uint32(2 * 8 * XS_STRIDE)
        for st in cutlass.range_constexpr(GEMV_STEPS):
            x0 = Uint32(0)
            x1 = Uint32(0)
            x2 = Uint32(0)
            x3 = Uint32(0)
            if tok:
                x0, x1, x2, x3 = P.ld_shared_v4_u32(xr + Uint32(2 * 32 * st))
            c0, c1, c2, c3 = P.mma_bf16_m16n8k16(
                (c0, c1, c2, c3), (a[st, 0], b[st, 0], a[st, 1], b[st, 1]), (x0, x1)
            )
            c0, c1, c2, c3 = P.mma_bf16_m16n8k16(
                (c0, c1, c2, c3), (a[st, 2], b[st, 2], a[st, 3], b[st, 3]), (x2, x3)
            )
            if cutlass.const_expr(self.two_tiles):  # noqa: SIM102
                if two:
                    y0 = Uint32(0)
                    y1 = Uint32(0)
                    y2 = Uint32(0)
                    y3 = Uint32(0)
                    if tok2:
                        y0, y1, y2, y3 = P.ld_shared_v4_u32(xr2 + Uint32(2 * 32 * st))
                    d0, d1, d2, d3 = P.mma_bf16_m16n8k16(
                        (d0, d1, d2, d3),
                        (a[st, 0], b[st, 0], a[st, 1], b[st, 1]),
                        (y0, y1),
                    )
                    d0, d1, d2, d3 = P.mma_bf16_m16n8k16(
                        (d0, d1, d2, d3),
                        (a[st, 2], b[st, 2], a[st, 3], b[st, 3]),
                        (y2, y3),
                    )
        red[kq, tile, g, 2 * q] = c0
        red[kq, tile, g, 2 * q + 1] = c1
        red[kq, tile, g + 8, 2 * q] = c2
        red[kq, tile, g + 8, 2 * q + 1] = c3
        if cutlass.const_expr(self.two_tiles):  # noqa: SIM102
            if two:
                red[kq, tile, g, 8 + 2 * q] = d0
                red[kq, tile, g, 8 + 2 * q + 1] = d1
                red[kq, tile, g + 8, 8 + 2 * q] = d2
                red[kq, tile, g + 8, 8 + 2 * q + 1] = d3
        cute.arch.sync_threads()
        # 8 threads per row, 6 output rows (2 records) each
        if t < rows * (GEMV_ROWS // 6):
            m = t // (GEMV_ROWS // 6)
            k6 = t % (GEMV_ROWS // 6)
            wv = cute.make_rmem_tensor((3,), Uint32)
            if pz != 0:  # poisoned: NaN records, the partial sums are not read
                for p in cutlass.range_constexpr(3):
                    wv[p] = Uint32(BF16_NAN | (BF16_NAN << 16))
            else:
                for p in cutlass.range_constexpr(3):
                    h0 = self.quarter_sum(red, 6 * k6 + 2 * p, m)
                    h1 = self.quarter_sum(red, 6 * k6 + 2 * p + 1, m)
                    wv[p] = P.no_neg_zero(self.round_pair(h0, h1))
            em = es.erow[m]
            r0, r1 = self.pack_records(wv[0], wv[1], wv[2], em & Uint32(TAG_MASK))
            if cutlass.const_expr(not self.router):
                self.delay(mb, 1)
            self.publish_records(
                mb, self.buffer_of(em), m, (cta * GEMV_ROWS + 6 * k6) // 3, r0, r1
            )

    def round_pair(self, h0, h1):
        if cutlass.const_expr(self.router):
            return C.cvt_rn_bf16x2_f32(h0, h1)
        return Uint32(C.cvt_rn_bf16_f32(h0)) | (Uint32(C.cvt_rn_bf16_f32(h1)) << 16)

    @cute.jit
    def quarter_sum(self, red, row, m):
        """Fixed-order K-quarter sum ``((q0 + q1) + q2) + q3`` of output row ``row``,
        token ``m``."""
        tl = row // 16
        rr = row % 16
        v = red[0, tl, rr, m]
        v = C.add_rn(v, red[1, tl, rr, m])
        v = C.add_rn(v, red[2, tl, rr, m])
        v = C.add_rn(v, red[3, tl, rr, m])
        return v

    @cute.jit
    def router_cta(
        self,
        mb: Mbox,
        base,
        h,
        x: Int64,
        qh: Int64,
        ql: Int64,
        residual: Int64,
        rows: Int32,
        rp,
        pol,
    ):
        """Router CTA ``h``: rows ``[16h, 16h + 16)`` of ``[Q_r | W'_slice]`` (hi and
        lo) in registers before the wait; the rank's term for 16 experts of every row,
        published as ``{fp32, tag32 = E_row[m]}`` words to every rank's router-partial
        mailbox."""
        t = P.thread_index()
        xs_bytes: cutlass.Constexpr = self.max_rows * RXS_STRIDE * 2
        mx: cutlass.Constexpr = self.max_rows
        red = _view(base, xs_bytes, Float32, (12, 16, mx), (16 * mx, mx, 1))
        es = _epoch_view(base, xs_bytes + 12 * 16 * mx * 4, self.max_rows)
        warp = t // 32
        lane = t % 32
        g = lane // 4
        q = lane % 4
        s0 = 8 * warp
        ns = 8
        if warp >= 4:
            s0 = 32 + 7 * (warp - 4)
            ns = 7
        ra = Int64(16 * h + g) * ROUTER_K + Int64(8 * q)
        rb = ra + Int64(8 * ROUTER_K)
        ah = cute.make_rmem_tensor((ROUTER_STEPS, 4), Uint32)
        bh = cute.make_rmem_tensor((ROUTER_STEPS, 4), Uint32)
        al = cute.make_rmem_tensor((ROUTER_STEPS, 4), Uint32)
        bl = cute.make_rmem_tensor((ROUTER_STEPS, 4), Uint32)
        ah.fill(0)
        bh.fill(0)
        al.fill(0)
        bl.fill(0)
        for i in cutlass.range_constexpr(ROUTER_STEPS):
            if i < ns:
                k0 = Int64(32 * (s0 + i))
                v0 = sys_ops.ld_nc_hint_v4_u32(qh + Int64(2) * (ra + k0), pol)
                v1 = sys_ops.ld_nc_hint_v4_u32(qh + Int64(2) * (rb + k0), pol)
                v2 = sys_ops.ld_nc_hint_v4_u32(ql + Int64(2) * (ra + k0), pol)
                v3 = sys_ops.ld_nc_hint_v4_u32(ql + Int64(2) * (rb + k0), pol)
                for k in cutlass.range_constexpr(4):
                    ah[i, k] = v0[k]
                    bh[i, k] = v1[k]
                    al[i, k] = v2[k]
                    bl[i, k] = v3[k]
        self.pdl_edge(rows)
        sys_ops.fence_proxy_async_global()
        xs_base = Uint32(base.toint())
        i = t
        nchunk = rows * (ROUTER_K // 8)
        while i < nchunk:  # B' = [x | residual slice of this rank]
            mm = i // (ROUTER_K // 8)
            kc = i % (ROUTER_K // 8)
            src = x + Int64(2) * (Int64(mm) * O_PROJ_IN + Int64(kc * 8))
            if kc >= O_PROJ_IN // 8:
                src = residual + Int64(2) * (
                    Int64(mm) * HIDDEN
                    + Int64(mb.rank) * (HIDDEN // WORLD)
                    + Int64((kc - O_PROJ_IN // 8) * 8)
                )
            self.stage16(xs_base + Uint32(2 * (mm * RXS_STRIDE + kc * 8)), src)
            i = i + GEMV_THREADS
        self.stage_done()
        self.epoch_begin(mb, es, True)
        hi0 = Float32(0.0)
        hi1 = Float32(0.0)
        hi2 = Float32(0.0)
        hi3 = Float32(0.0)
        lo0 = Float32(0.0)
        lo1 = Float32(0.0)
        lo2 = Float32(0.0)
        lo3 = Float32(0.0)
        e0 = Float32(0.0)
        e1 = Float32(0.0)
        e2 = Float32(0.0)
        e3 = Float32(0.0)
        f0 = Float32(0.0)
        f1 = Float32(0.0)
        f2 = Float32(0.0)
        f3 = Float32(0.0)
        tok = g < rows
        tok2 = g + 8 < rows
        two = rows > 8
        for i in cutlass.range_constexpr(ROUTER_STEPS):
            if i < ns:
                k0 = 32 * (s0 + i) + 8 * q
                x0 = Uint32(0)
                x1 = Uint32(0)
                x2 = Uint32(0)
                x3 = Uint32(0)
                if tok:
                    x0, x1, x2, x3 = P.ld_shared_v4_u32(
                        xs_base + Uint32(2 * (g * RXS_STRIDE + k0))
                    )
                hi0, hi1, hi2, hi3 = P.mma_bf16_m16n8k16(
                    (hi0, hi1, hi2, hi3),
                    (ah[i, 0], bh[i, 0], ah[i, 1], bh[i, 1]),
                    (x0, x1),
                )
                hi0, hi1, hi2, hi3 = P.mma_bf16_m16n8k16(
                    (hi0, hi1, hi2, hi3),
                    (ah[i, 2], bh[i, 2], ah[i, 3], bh[i, 3]),
                    (x2, x3),
                )
                lo0, lo1, lo2, lo3 = P.mma_bf16_m16n8k16(
                    (lo0, lo1, lo2, lo3),
                    (al[i, 0], bl[i, 0], al[i, 1], bl[i, 1]),
                    (x0, x1),
                )
                lo0, lo1, lo2, lo3 = P.mma_bf16_m16n8k16(
                    (lo0, lo1, lo2, lo3),
                    (al[i, 2], bl[i, 2], al[i, 3], bl[i, 3]),
                    (x2, x3),
                )
                if cutlass.const_expr(self.two_tiles):  # noqa: SIM102
                    if two:
                        y0 = Uint32(0)
                        y1 = Uint32(0)
                        y2 = Uint32(0)
                        y3 = Uint32(0)
                        if tok2:
                            y0, y1, y2, y3 = P.ld_shared_v4_u32(
                                xs_base + Uint32(2 * ((g + 8) * RXS_STRIDE + k0))
                            )
                        e0, e1, e2, e3 = P.mma_bf16_m16n8k16(
                            (e0, e1, e2, e3),
                            (ah[i, 0], bh[i, 0], ah[i, 1], bh[i, 1]),
                            (y0, y1),
                        )
                        e0, e1, e2, e3 = P.mma_bf16_m16n8k16(
                            (e0, e1, e2, e3),
                            (ah[i, 2], bh[i, 2], ah[i, 3], bh[i, 3]),
                            (y2, y3),
                        )
                        f0, f1, f2, f3 = P.mma_bf16_m16n8k16(
                            (f0, f1, f2, f3),
                            (al[i, 0], bl[i, 0], al[i, 1], bl[i, 1]),
                            (y0, y1),
                        )
                        f0, f1, f2, f3 = P.mma_bf16_m16n8k16(
                            (f0, f1, f2, f3),
                            (al[i, 2], bl[i, 2], al[i, 3], bl[i, 3]),
                            (y2, y3),
                        )
        red[warp, g, 2 * q] = C.add_rn(hi0, lo0)
        red[warp, g, 2 * q + 1] = C.add_rn(hi1, lo1)
        red[warp, g + 8, 2 * q] = C.add_rn(hi2, lo2)
        red[warp, g + 8, 2 * q + 1] = C.add_rn(hi3, lo3)
        if cutlass.const_expr(self.two_tiles):  # noqa: SIM102
            if two:
                red[warp, g, 8 + 2 * q] = C.add_rn(e0, f0)
                red[warp, g, 8 + 2 * q + 1] = C.add_rn(e1, f1)
                red[warp, g + 8, 8 + 2 * q] = C.add_rn(e2, f2)
                red[warp, g + 8, 8 + 2 * q + 1] = C.add_rn(e3, f3)
        cute.arch.sync_threads()
        if t < 16 * rows:
            ex = t // rows
            m = t % rows
            v = red[0, ex, m]
            for wi in cutlass.range_constexpr(1, 12):
                v = C.add_rn(v, red[wi, ex, m])
            if es.pz[0] != 0:
                v = P.bits_f32(Uint32(0x7FC00000))
            em = es.erow[m]
            word = Uint64(P.f32_bits(v)) | (Uint64(em) << 32)
            off = (
                (
                    (Int64(self.buffer_of(em)) * WORLD + Int64(mb.rank)) * self.max_rows
                    + Int64(m)
                )
                * EXPERTS
                + Int64(16 * h + ex)
            ) * 8
            for d in cutlass.range_constexpr(WORLD):
                sys_ops.st_relaxed_sys_u64(rp[d] + off, word)


class OProjArConsume(Consumer):
    """Consumer of an o_proj hop: all-reduce + residual + RMSNorm for up to ``max_rows``
    rows (``router``: + the router logits from the producer's terms and the per-row
    top-8 routing). Not cooperative: it spins only on earlier grids and on remote
    records; its shared-memory exchange is between the CTAs of one cluster."""

    wait_prod = True  # start edge: the last arriver waits for the producer grid

    def __init__(
        self,
        *,
        router: bool = False,
        residual: bool = True,
        norm: bool = True,
        max_rows: int = 8,
    ):
        if residual and not norm:
            raise ValueError("residual without RMSNorm is not provided")
        if router and not (residual and norm):
            raise ValueError("the router consumer is provided with residual + RMSNorm")
        if max_rows not in (8, 16) or (max_rows == 16 and not router):
            raise ValueError(
                "provided for up to 8 rows, or up to 16 rows with the router"
            )
        self.router, self.residual, self.norm, self.max_rows = (
            router,
            residual,
            norm,
            max_rows,
        )

    @cute.jit
    def __call__(
        self,
        residual: Int64,
        gamma: Int64,
        out_res: Int64,
        out_norm: Int64,
        eps: Float32,
        rows: Int32,
        rp_local: Int64,
        bias: Int64,
        rscale: Float32,
        ids: Int64,
        wts: Int64,
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
        """Router: ``rp_local`` this rank's router-partial mailbox, ``bias`` [256] fp32
        score bias, ``rscale`` the routed scaling factor, ``ids`` [rows, 8] int32 /
        ``wts`` [rows, 8] fp32 outputs (unused without the router)."""
        self.kernel(
            residual,
            gamma,
            out_res,
            out_norm,
            eps,
            rows,
            rp_local,
            bias,
            rscale,
            ids,
            wts,
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
            grid=[rows * 8 + 8, 1, 1],
            block=[CONSUMER_THREADS, 1, 1],
            max_number_threads=[CONSUMER_THREADS, 1, 1],
            cluster=[8, 1, 1],
            use_pdl=True,
            stream=stream,
        )

    @cute.jit
    def entry(self):
        """Start edge: no wait at entry (ordered by the producer's fenced trigger)."""
        cute.arch.griddepcontrol_launch_dependents()

    @cute.jit
    def exit_wait(self):
        cute.arch.griddepcontrol_wait()

    @cute.kernel
    def kernel(
        self,
        residual: Int64,
        gamma: Int64,
        out_res: Int64,
        out_norm: Int64,
        eps: Float32,
        rows: Int32,
        rp_local: Int64,
        bias: Int64,
        rscale: Float32,
        ids: Int64,
        wts: Int64,
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
        lg = smem.allocate_tensor(Float32, cute.make_layout(EXPERTS), 16)
        scr = smem.allocate_tensor(Float32, cute.make_layout(EXPERTS), 16)
        # one spare word (unused); this kernel's shared-memory size includes it
        smem.allocate_tensor(Float32, cute.make_layout(1), 4)
        nmain = rows * 8
        if cutlass.const_expr(self.norm):  # noqa: SIM102
            if bx < nmain:
                self.cons_prologue(cs)
        self.entry()
        # the residual may come from an async-proxy writer
        sys_ops.fence_proxy_async_global()
        if bx >= nmain:
            e = self.epoch_begin(mb, cs.es, False)
            if t == 0:
                if bx == nmain:
                    self.ack_cta(mb, e)
                self.epoch_arrive(mb, cs.es, rows, P.grid_size(), self.wait_prod)
                if cutlass.const_expr(self.wait_prod):
                    self.exit_wait()
            sys_ops.exit_thread()
        m = bx // 8
        c = bx % 8
        h0 = c * 768 + 6 * t
        bsv = cute.make_rmem_tensor((8,), Float32)
        bsv.fill(0.0)
        if cutlass.const_expr(self.router):  # noqa: SIM102
            # routing warp: the static score bias into registers now
            if c == 0 and t < 32:
                v = sys_ops.ld_nc_v4_f32(bias + Int64(4 * 8 * t))
                u = sys_ops.ld_nc_v4_f32(bias + Int64(4 * 8 * t + 16))
                for k in cutlass.range_constexpr(4):
                    bsv[k] = v[k]
                    bsv[4 + k] = u[k]
        ru = cute.make_rmem_tensor((3,), Uint32)
        gu = cute.make_rmem_tensor((3,), Uint32)
        ru.fill(0)
        gu.fill(0)
        if t < 128:
            if cutlass.const_expr(self.residual):
                off = Int64(2) * (Int64(m) * HIDDEN + Int64(h0))
                for k in cutlass.range_constexpr(3):
                    ru[k] = sys_ops.ld_cg_u32(residual + off + Int64(4 * k))
            if cutlass.const_expr(self.norm):
                for k in cutlass.range_constexpr(3):
                    gu[k] = sys_ops.ld_cg_u32(
                        gamma + Int64(2) * Int64(h0) + Int64(4 * k)
                    )
        self.epoch_begin(mb, cs.es, False)
        if t >= POLL_THREADS:  # the epoch warp arrives, then leaves the CTA
            if t == POLL_THREADS:
                self.epoch_arrive(mb, cs.es, rows, P.grid_size(), self.wait_prod)
            sys_ops.exit_thread()
        if cutlass.const_expr(self.router):
            self.consume_cta_router(
                mb,
                cs,
                lg,
                scr,
                m,
                c,
                cs.es.erow[m],
                cs.es.pz[0],
                ru,
                gu,
                bsv,
                out_res,
                out_norm,
                eps,
                rp_local,
                rscale,
                ids,
                wts,
            )
        else:
            self.consume_cta(
                mb,
                cs,
                m,
                c,
                cs.es.erow[m],
                cs.es.pz[0],
                ru,
                gu,
                out_res,
                out_norm,
                eps,
                self.residual,
                self.norm,
            )
        if cutlass.const_expr(self.wait_prod):  # noqa: SIM102
            if t == 0:
                self.exit_wait()

    # ------------------------------------------------------------ router consumer
    @cute.jit
    def consume_cta_router(
        self,
        mb: Mbox,
        cs: ConsSmem,
        lg,
        scr,
        m,
        c,
        e,
        poisoned,
        ru,
        gu,
        bsv,
        out_res: Int64,
        out_norm: Int64,
        eps: Float32,
        rp_local: Int64,
        rscale: Float32,
        ids: Int64,
        wts: Int64,
    ):
        """The consumer epilogue with the shared-memory sum of squares, plus (cluster
        rank 0) the router-term gather of threads 128..383 (expert ``t - 128``) and the
        routing warp."""
        t = P.thread_index()
        f = t % 128
        grp = t // 128
        buf = self.buffer_of(e)
        tag = Uint32(e) & Uint32(TAG_MASK)
        acc = cute.make_rmem_tensor((6,), Float32)
        x0 = cute.make_rmem_tensor((6,), Float32)
        x1 = cute.make_rmem_tensor((6,), Float32)
        acc.fill(0.0)
        self.poll_records(mb, buf, m, c * 256 + 2 * f, 2 * grp, tag, poisoned, x0, x1)
        if grp > 0:
            for k in cutlass.range_constexpr(6):
                cs.raw[grp - 1, f, k] = x0[k]
                cs.raw[grp - 1, f, 6 + k] = x1[k]
        else:
            for k in cutlass.range_constexpr(6):
                acc[k] = C.add_rn(C.add_rn(Float32(0.0), x0[k]), x1[k])
        # router terms of (m, expert t - 128): the eight source words are issued now
        rtr = (c == 0) and (t >= 128) and (t < 128 + EXPERTS)
        ex = t - 128
        pw = cute.make_rmem_tensor((WORLD,), Uint64)
        pw.fill(0)
        if rtr:
            for s in cutlass.range_constexpr(WORLD):
                pw[s] = sys_ops.ld_relaxed_sys_u64(
                    rp_local + self.rp_off(buf, s, m, ex)
                )
        P.bar_sync(1, 512)
        hsum = Float32(0.0)
        hto = Int32(0)
        if rtr:  # tag check + rank-order sum while group 0 finalizes the all-reduce
            hsum, hto = self.router_gather(mb, pw, rp_local, buf, m, ex, e)
        r = cute.make_rmem_tensor((6,), Float32)
        r.fill(0.0)
        h0 = c * 768 + 6 * f
        if grp == 0:
            for q in cutlass.range_constexpr(1, 4):
                for k in cutlass.range_constexpr(6):
                    acc[k] = C.add_rn(
                        C.add_rn(acc[k], cs.raw[q - 1, f, k]), cs.raw[q - 1, f, 6 + k]
                    )
            rb = cute.make_rmem_tensor((3,), Uint32)
            for k in cutlass.range_constexpr(3):
                ab = P.no_neg_zero(C.cvt_rn_bf16x2_f32(acc[2 * k], acc[2 * k + 1]))
                rb[k] = C.cvt_rn_bf16x2_f32(
                    C.add_rn(P.bf16_lo(ab), P.bf16_lo(ru[k])),
                    C.add_rn(P.bf16_hi(ab), P.bf16_hi(ru[k])),
                )
            row = out_res + Int64(2) * (Int64(m) * HIDDEN + Int64(h0))
            for k in cutlass.range_constexpr(3):
                sys_ops.st_global_u32(row + Int64(4 * k), rb[k])
            for k in cutlass.range_constexpr(3):
                r[2 * k] = P.bf16_lo(rb[k])
                r[2 * k + 1] = P.bf16_hi(rb[k])
            for k in cutlass.range_constexpr(6):
                cs.rs[6 * f + k] = r[k]
        lane = t % 32
        warp = t // 32
        P.bar_sync(1, 512)
        # 96 chunks of 8 elements: from 0 in element order, bf16-rounded squares
        if t < 96:
            ts = Float32(0.0)
            for i in cutlass.range_constexpr(4):
                a0 = cs.rs[8 * t + 2 * i]
                a1 = cs.rs[8 * t + 2 * i + 1]
                wq = C.cvt_rn_bf16x2_f32(C.mul_rn(a0, a0), C.mul_rn(a1, a1))
                ts = C.add_rn(C.add_rn(ts, P.bf16_lo(wq)), P.bf16_hi(wq))
            for i in cutlass.range_constexpr(5):
                o: cutlass.Constexpr = 16 >> i
                ts = C.add_rn(ts, P.shfl_xor(ts, o))
            if lane == 0:
                cs.wsum[warp] = ts
        P.bar_sync(1, 512)
        bs = Float32(0.0)
        if lane < 3:
            bs = cs.wsum[lane]
        for i in cutlass.range_constexpr(5):
            o2: cutlass.Constexpr = 16 >> i
            bs = C.add_rn(bs, P.shfl_xor(bs, o2))
        crank = Int32(cute.arch.block_idx_in_cluster())
        P.cluster_wait()
        if t == 0:
            la = Uint32((cs.csum.iterator + crank).toint())
            lm = cs.mbar_addr()
            for cc in cutlass.range_constexpr(8):
                P.st_async_b32(P.mapa_u32(la, cc), P.f32_bits(bs), P.mapa_u32(lm, cc))
        P.mbar_wait_cluster(cs.mbar_addr(), Uint32(0))
        P.cluster_arrive_relaxed()
        tot = Float32(0.0)
        for cc in cutlass.range_constexpr(8):
            tot = C.add_rn(tot, cs.csum[cc])
        inv = C.rsqrt_approx(C.fma_rn(tot, C.rcp_rn(Float32(6144.0)), eps))
        if rtr:
            if hto != 0:
                self.fail(mb, ERR_ROUTER_POLL, 1, 0, m, e)
            lg[ex] = C.mul_rn(hsum, inv)
        if grp == 0:
            row = out_norm + Int64(2) * (Int64(m) * HIDDEN + Int64(h0))
            for k in cutlass.range_constexpr(3):
                g0 = C.add_rn(Float32(0.0), P.bf16_lo(gu[k]))
                g1 = C.add_rn(Float32(0.0), P.bf16_hi(gu[k]))
                y = C.cvt_rn_bf16x2_f32(
                    C.mul_rn(C.mul_rn(r[2 * k], inv), g0),
                    C.mul_rn(C.mul_rn(r[2 * k + 1], inv), g1),
                )
                sys_ops.st_global_u32(row + Int64(4 * k), y)
            sys_ops.fence_proxy_async_global()
        if c == 0 and t < 128 + EXPERTS:
            P.bar_sync(4, 128 + EXPERTS)  # logits in shared memory, norm_out stored
            if warp == 0:
                self.route_row(lg, scr, bsv, lane, m, poisoned, rscale, ids, wts)
        elif grp == 0:
            P.bar_sync(2, 128)
        P.cluster_wait()  # exit barrier, wait

    def rp_off(self, buf, src, m, ex):
        return (
            ((Int64(buf) * WORLD + Int64(src)) * self.max_rows + Int64(m)) * EXPERTS
            + Int64(ex)
        ) * 8

    @cute.jit
    def router_gather(self, mb: Mbox, pw, rp_local: Int64, buf, m, ex, e):
        """Poll the eight words until each carries ``tag32 = e``; their fp32 halves
        summed in rank order. Returns (sum, timed out)."""
        t0 = Uint64(0)
        it = Uint32(0)
        hto = Int32(0)
        for s in cutlass.range_constexpr(WORLD):
            while (Uint32(pw[s] >> 32) != Uint32(e)) and (hto == 0):
                pw[s] = sys_ops.ld_relaxed_sys_u64(
                    rp_local + self.rp_off(buf, s, m, ex)
                )
                it = it + Uint32(1)
                if (it & Uint32(POLL_TIMER_STRIDE - 1)) == Uint32(0):
                    now = sys_ops.globaltimer()
                    if t0 == Uint64(0):
                        t0 = now
                    elif now - t0 > Uint64(self.timeout_ns):
                        hto = Int32(1)
        hsum = P.bits_f32(Uint32(pw[0] & Uint64(0xFFFFFFFF)))
        for s in cutlass.range_constexpr(1, WORLD):
            hsum = C.add_rn(hsum, P.bits_f32(Uint32(pw[s] & Uint64(0xFFFFFFFF))))
        return hsum, hto

    @cute.jit
    def route_row(
        self, lg, scr, bsv, lane, m, poisoned, rscale: Float32, ids: Int64, wts: Int64
    ):
        """One warp routes row ``m``: score = sigmoid(logit); key = score + bias (NaN ->
        -inf, -0 -> +0) mapped order-preserving to u32; top-8 by (key desc, expert id
        asc); weights ``bf16(rscale * s / (sum of the 8 s + 1e-20))``."""
        sc = cute.make_rmem_tensor((8,), Float32)
        key = cute.make_rmem_tensor((8,), Uint32)
        for jj in cutlass.range_constexpr(8):
            sc[jj] = P.routing_sigmoid(lg[8 * lane + jj])
            u = P.f32_bits(P.add_ftz(sc[jj], bsv[jj]))
            if (u & Uint32(0x7FFFFFFF)) > Uint32(0x7F800000):
                u = Uint32(0xFF800000)  # NaN -> -inf
            if (u & Uint32(0x7FFFFFFF)) == Uint32(0):
                u = Uint32(0)  # -0 -> +0
            if (u & Uint32(0x80000000)) != Uint32(0):
                key[jj] = ~u
            else:
                key[jj] = u | Uint32(0x80000000)
        myid, ok = self.top8_fast(key, lane)
        if ok == 0:
            myid = self.top8_merge(key, lane)
        for jj in cutlass.range_constexpr(8):
            scr[8 * lane + jj] = sc[jj]
        cute.arch.sync_warp()
        myw = Float32(0.0)
        if lane < TOPK:
            myw = scr[myid]
        wsum = myw
        for i in cutlass.range_constexpr(3):
            o: cutlass.Constexpr = 4 >> i
            wsum = P.add_ftz(wsum, P.shfl_xor(wsum, o))
        if lane < TOPK:
            wn = P.routing_weight(rscale, myw, wsum)
            if poisoned != 0:
                wn = P.bits_f32(Uint32(0x7FFFFFFF))
            sys_ops.st_global_u32(ids + Int64(4 * (m * TOPK + lane)), Uint32(myid))
            sys_ops.st_global_u32(wts + Int64(4 * (m * TOPK + lane)), P.f32_bits(wn))
        cute.arch.sync_warp()

    @cute.jit
    def top8_fast(self, key, lane):
        """Composite ``c = (key & ~0xff) | (255 - id)``: one unsigned max orders by (key
        >> 8 desc, id asc). Per-lane sort, then 9 rounds of one warp max over the lane
        heads. Exact iff the top-9 composites have pairwise distinct ``key >> 8``
        between neighbours (else the caller runs the merge). Returns (id of rank
        ``lane`` for lanes < 8, exact)."""
        s = cute.make_rmem_tensor((8,), Uint32)
        for j in cutlass.range_constexpr(8):
            s[j] = (key[j] & Uint32(0xFFFFFF00)) | Uint32(255 - 8 * lane - j)
        for a, b in (
            (0, 1),
            (2, 3),
            (4, 5),
            (6, 7),
            (0, 2),
            (1, 3),
            (4, 6),
            (5, 7),
            (1, 2),
            (5, 6),
            (0, 4),
            (1, 5),
            (2, 6),
            (3, 7),
            (2, 4),
            (3, 5),
            (1, 2),
            (3, 4),
            (5, 6),
        ):
            hi = P.umax(s[a], s[b])
            lo = P.umin(s[a], s[b])
            s[a] = hi
            s[b] = lo
        prev = Uint32(0)
        mine = Uint32(0)
        ok = Int32(1)
        for rr in cutlass.range_constexpr(TOPK + 1):
            mx = P.redux_max_u32(s[0])
            if cutlass.const_expr(rr > 0):  # noqa: SIM102
                if (prev >> 8) == (mx >> 8):
                    ok = Int32(0)
            prev = mx
            if cutlass.const_expr(rr < TOPK):
                if lane == rr:
                    mine = mx
                if s[0] == mx:
                    for i in cutlass.range_constexpr(7):
                        s[i] = s[i + 1]
                    s[7] = Uint32(0)
        return Int32(255) - Int32(mine & Uint32(0xFF)), ok

    @cute.jit
    def top8_merge(self, key, lane):
        """Exact fallback: each lane sorts its 8 (key, j) pairs (stable insertion
        network), then 8 rounds of a 32-way merge: max key, then min expert id among
        equal keys."""
        sk = cute.make_rmem_tensor((8,), Uint32)
        sj = cute.make_rmem_tensor((8,), Int32)
        for jj in cutlass.range_constexpr(8):
            sk[jj] = key[jj]
            sj[jj] = Int32(jj)
        for i1 in cutlass.range_constexpr(1, 8):
            for i2 in cutlass.range_constexpr(i1, 0, -1):
                ka = sk[i2 - 1]
                kb = sk[i2]
                ja = sj[i2 - 1]
                jb = sj[i2]
                if sk[i2] > sk[i2 - 1]:
                    ka = sk[i2]
                    kb = sk[i2 - 1]
                    ja = sj[i2]
                    jb = sj[i2 - 1]
                sk[i2 - 1] = ka
                sk[i2] = kb
                sj[i2 - 1] = ja
                sj[i2] = jb
        left = Int32(8)
        myid = Int32(0)
        for rr in cutlass.range_constexpr(TOPK):
            hk = Uint32(0)
            if left > 0:
                hk = sk[0]
            kmax = P.redux_max_u32(hk)
            cand = Uint32(0xFFFFFFFF)
            if left > 0 and hk == kmax:
                cand = Uint32(8 * lane) + Uint32(sj[0])
            win = Int32(P.redux_min_u32(cand))
            if lane == (win >> 3):
                for i in cutlass.range_constexpr(7):
                    sk[i] = sk[i + 1]
                    sj[i] = sj[i + 1]
                left = left - 1
            if lane == rr:
                myid = win
        return myid
