# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""The 9..16-row routing o_proj producer with its router partials stored row-contiguous.

``OProjArProduce`` (router) publishes each router CTA's terms (16 experts x ``rows``) as
``{fp32, tag32}`` words to every rank's router-partial mailbox ``[NBUF][8 src][rows][256]``
with thread ``t`` holding expert ``t // rows`` of row ``t % rows``: the 32 words of a warp lie
in 16 rows 2 KiB apart, 16 partial-sector writes per warp and rank over NVLink.
``OProjArProduceRc`` maps thread ``t`` to expert ``t % 16`` of row ``t // 16`` instead, so a
warp writes two whole 128-byte runs (16 experts of 2 rows) per rank. Each word is the same
sum over the same twelve warp partials in the same order, so the words, the routing and every
other output are bitwise the served unit's; only the store pattern changes. The GEMV CTAs, the
start edge and the protocol are ``OProjArProduce``'s.

Served for mailbox instance B (9..16 rows) with
``SGLANG_NVIDIA_MEGAKERNEL_GLM_HOP16_ROUTER_ROWS=1`` (``api.unit_specs``); instance A keeps
the served unit.
"""

import cutlass
import cutlass.cute as cute
from cutlass import Float32, Int32, Int64, Uint32, Uint64

from ..primitives import ptx_core as C
from ..primitives import ptx_sys as sys_ops
from . import _ptx_local as P
from .constants import HIDDEN, O_PROJ_IN, WORLD
from .o_proj import (
    EXPERTS,
    GEMV_THREADS,
    ROUTER_K,
    ROUTER_STEPS,
    RXS_STRIDE,
    OProjArProduce,
    _epoch_view,
    _view,
)
from .transport import Mbox


class OProjArProduceRc(OProjArProduce):
    """The 16-row routing o_proj producer with row-contiguous router-partial stores."""

    def __init__(self):
        super().__init__(router=True, max_rows=16)
        self.router_rows = True

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
        """``OProjArProduce.router_cta`` with the row-contiguous store mapping (thread
        ``t``: expert ``16 h + t % 16`` of row ``t // 16``); every value as there."""
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
        self.epoch_begin(mb, es, True, 0, self.epoch_overlap)
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
            # row-contiguous: a warp stores 2 rows x 16 experts (2 x 128 bytes) per rank
            ex = t % 16
            m = t // 16
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
