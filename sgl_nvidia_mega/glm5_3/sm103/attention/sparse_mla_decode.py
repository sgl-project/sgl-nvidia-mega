# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Sparse MLA decode / verify over the indexer's top-2048 tokens (FP8 MQA query and
cache).

For one tensor-parallel rank (8 heads), per query row ``r`` and head ``h``:

    s_t = bmm1_scale * q[r, h] . kv[t]           (576-wide FP8 dot products, FP32
    accumulate) out[r, h] = bmm2_scale * sum_t softmax(s)_t * kv[t][:512]       (BF16)

over the (up to) 2048 cache rows ``t`` selected for the row (ids < 0 are skipped). The
inputs are the stock decode call's: ``q`` [M, 8, 576] E4M3, the latent cache as flat
576-byte E4M3 rows, ``ids`` [M, 2048] int32 (physical cache rows, or logical token
positions mapped here through the request's block table).

Structure: one 16-CTA cluster per query row; CTA ``c`` takes tokens ``[128 c, 128 c +
128)`` of the row's list. Its first eight warps each gather 16 cache rows (one bulk copy
per row, completing on the warp's mbarrier; unselected rows are zero-filled), compute
``K Q^T`` with ``mma.sync`` m16n8k32 E4M3 (tokens on M, heads on N) and the CTA's
softmax statistics; every warp then computes ``V^T P^T`` for its slice of the 512 output
dims, with P held as two E4M3 terms (``hi`` and the residual ``lo``, scaled by 2^8). The
16 partial results of a row are merged in a fixed order: each CTA pushes its partial
outputs and (max, sum) pairs into the owning CTA's shared memory with ``st.async``,
completing on the owner's mbarrier, and the owner rescales and sums them. The result is
deterministic and does not depend on the other rows.

Ordering (PTX ISA memory consistency model):
  * The cluster co-start: every CTA initializes its mbarriers and arms its combine
  barrier,
    then arrives (relaxed) on the cluster barrier, and waits on it before its first
    ``st.async`` into a peer.
  * Everything that reads predecessor data (q, ids, block table, cache rows) follows
    ``griddepcontrol.wait`` and a ``fence.proxy.async.global``.
  * Gathers complete on per-warp mbarriers (``expect_tx`` with release, waits with
  acquire);
    the combine's ``st.async`` complete on the owner's mbarrier at cluster scope.
  * A CTA leaves only after the split exit cluster barrier, so no ``st.async`` is in
  flight
    toward an exited CTA; every thread runs ``fence.proxy.async.global`` after its
    stores.
  * With ``early_trigger`` the dependents are triggered right after the wait (for a
    successor that waits on this grid's completion before reading its output).
"""

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
from cutlass import Float32, Int32, Int64, Uint32

from ..primitives import ptx_core as ptx

HEADS = 8
QK_DIM = 576
V_DIM = 512
TOPK = 2048
TOKENS_PER_CTA = 128
CLUSTER = TOPK // TOKENS_PER_CTA  # 16 CTAs per query row
QK_WARPS = 8
TOKENS_PER_WARP = TOKENS_PER_CTA // QK_WARPS  # 16
ROW_BYTES = QK_DIM + 16  # shared-memory row pitch of a gathered cache row
QK_STEPS = QK_DIM // 32
DIMS_PER_CTA = V_DIM // CLUSTER  # 32 output dims merged by each CTA
P_PITCH = TOKENS_PER_CTA + 16  # bytes per head row of P (E4M3)
NEG_INF = float("-inf")

# Shared memory (bytes)
OFF_KV = 0
OFF_Q = OFF_KV + TOKENS_PER_CTA * ROW_BYTES
OFF_P = OFF_Q + HEADS * ROW_BYTES
P_BYTES = 2 * HEADS * P_PITCH
OFF_RED = OFF_P + P_BYTES  # float [2][8 warps][8 heads]: max, then sum
OFF_RECV = OFF_RED + 2 * QK_WARPS * HEADS * 4  # float [16 sources][32 dims][8 heads]
OFF_STATS = (
    OFF_RECV + CLUSTER * DIMS_PER_CTA * HEADS * 4
)  # float [16 sources][2][8 heads]
OFF_BAR = OFF_STATS + CLUSTER * 2 * HEADS * 4  # u64 [8 gathers, q, combine]
SMEM_BYTES = OFF_BAR + (QK_WARPS + 2) * 8
RECV_BYTES = CLUSTER * (DIMS_PER_CTA * HEADS + 2 * HEADS) * 4
BAR_Q = QK_WARPS
BAR_COMBINE = QK_WARPS + 1


class SparseMlaDecode:
    """Sparse MLA decode over ``rows`` query rows.

    ``warps``: 16 (one CTA per SM; for up to 4 rows) or 8 (two CTAs per SM; more rows
    per wave). ``logical_ids``: ``ids`` hold token positions, mapped through
    ``block_table[request_of_row[r]]`` (``-1`` when negative or past the table); the
    mapped ids are also written to ``phys_out`` when it is non-zero. ``early_trigger``:
    trigger the dependents right after the PDL wait.

    Call arguments (device addresses, element strides): ``q``, ``kv``, ``ids``, ``out``
    ([rows, 8, 512] BF16), ``kv_rows`` (number of addressable cache rows), ``rows``,
    ``scale_log2`` (bmm1_scale * log2(e)), ``bmm2_scale``, then for logical ids
    ``request_of_row``, ``block_table``, ``block_table_stride``, ``max_blocks``,
    ``block_shift`` (log2 of the block size), ``block_rows`` (cache rows per block),
    ``phys_out`` (0: none), then the stream.
    """

    def __init__(
        self, warps: int = 16, logical_ids: bool = False, early_trigger: bool = True
    ):
        if warps not in (8, 16):
            raise ValueError(f"warps must be 8 or 16, got {warps}")
        self.warps = warps
        self.min_blocks = 1 if warps == 16 else 2
        self.logical_ids = logical_ids
        self.early_trigger = early_trigger

    @cute.jit
    def __call__(
        self,
        q: Int64,
        kv: Int64,
        ids: Int64,
        out: Int64,
        kv_rows: Int64,
        rows: Int32,
        scale_log2: Float32,
        bmm2_scale: Float32,
        request_of_row: Int64,
        block_table: Int64,
        block_table_stride: Int64,
        max_blocks: Int32,
        block_shift: Int32,
        block_rows: Int64,
        phys_out: Int64,
        stream: cuda.CUstream,
    ):
        self.kernel(
            q,
            kv,
            ids,
            out,
            kv_rows,
            scale_log2,
            bmm2_scale,
            request_of_row,
            block_table,
            block_table_stride,
            max_blocks,
            block_shift,
            block_rows,
            phys_out,
        ).launch(
            grid=[CLUSTER, rows, 1],
            block=[self.warps * 32, 1, 1],
            cluster=[CLUSTER, 1, 1],
            stream=stream,
            use_pdl=True,
            max_number_threads=[self.warps * 32, 1, 1],
            min_blocks_per_mp=self.min_blocks,
        )

    @cute.kernel
    def kernel(
        self,
        q: Int64,
        kv: Int64,
        ids: Int64,
        out: Int64,
        kv_rows: Int64,
        scale_log2: Float32,
        bmm2_scale: Float32,
        request_of_row: Int64,
        block_table: Int64,
        block_table_stride: Int64,
        max_blocks: Int32,
        block_shift: Int32,
        block_rows: Int64,
        phys_out: Int64,
    ):
        warps = self.warps
        dims_per_warp = V_DIM // warps
        pairs = dims_per_warp // 32
        tid_, _, _ = cute.arch.thread_idx()
        crank_, r_, _ = cute.arch.block_idx()
        tid = Int32(tid_)
        crank = Int32(crank_)
        r = Int32(r_)
        warp = tid // 32
        lane = tid % 32
        g = lane // 4
        t4 = lane % 4

        smem = cutlass.utils.SmemAllocator()
        buf = smem.allocate_tensor(
            cutlass.Uint8, cute.make_layout(SMEM_BYTES), byte_alignment=128
        )
        base = ptx.shared_address(buf.iterator)
        s_kv = base + OFF_KV
        s_q = base + OFF_Q
        s_ph = base + OFF_P
        s_pl = base + OFF_P + HEADS * P_PITCH
        s_red = base + OFF_RED
        s_recv = base + OFF_RECV
        s_stats = base + OFF_STATS
        s_bar = base + OFF_BAR

        if tid == 0:
            for i in cutlass.range_constexpr(QK_WARPS + 2):
                ptx.mbarrier_init(s_bar + i * 8, 1)
            ptx.fence_mbarrier_init_release_cluster()
            # owner side of the combine: expect every peer's partial before the co-start
            ptx.mbarrier_arrive_expect_tx(
                s_bar + BAR_COMBINE * 8, RECV_BYTES, ".release.cluster"
            )
            ptx.fence_proxy_async_shared_cta()
        ptx.sync_threads()
        self.cluster_costart_arrive()

        self.wait_inputs()

        # q rows (8 x 576 B), warp 0
        if warp == 0:
            if lane == 0:
                ptx.mbarrier_arrive_expect_tx(
                    s_bar + BAR_Q * 8, HEADS * QK_DIM, ".release.cta"
                )
            cute.arch.sync_warp()
            if lane < HEADS:
                src = q + (Int64(r) * HEADS + Int64(lane)) * QK_DIM
                ptx.bulk_copy_g2s(
                    s_q + lane * ROW_BYTES, src, QK_DIM, s_bar + BAR_Q * 8
                )

        # gather: warp w < 8 takes list positions j = 128 c + 16 w + lane (lanes < 16)
        vmask = Uint32(0)
        s0 = Float32(0.0)
        s1 = Float32(0.0)
        s2 = Float32(0.0)
        s3 = Float32(0.0)
        if warp < QK_WARPS:
            j = crank * TOKENS_PER_CTA + warp * TOKENS_PER_WARP + lane
            tok_id = Int32(-1)
            if lane < TOKENS_PER_WARP:
                tok_id = ptx.ld_global_s32(ids + (Int64(r) * TOPK + Int64(j)) * 4)
            if cutlass.const_expr(self.logical_ids):
                tok_id = self.map_logical(
                    tok_id,
                    r,
                    j,
                    lane,
                    request_of_row,
                    block_table,
                    block_table_stride,
                    max_blocks,
                    block_shift,
                    block_rows,
                    phys_out,
                )
            ok = (lane < TOKENS_PER_WARP) & (tok_id >= 0) & (Int64(tok_id) < kv_rows)
            vmask = Uint32(cute.arch.vote_ballot_sync(ok))
            self.gather(s_kv, s_bar, kv, tok_id, ok, vmask, warp, lane)

            # S^T = K Q^T for this warp's 16 tokens (two accumulator chains: even / odd
            # k-steps)
            self.wait_q(s_bar + BAR_Q * 8)
            qa = s_q + (lane % 8) * ROW_BYTES + ((lane // 8) % 2) * 16
            qf = [ptx.ldmatrix_x2(qa + ks * 32) for ks in range(QK_STEPS)]
            self.wait_gather(s_bar + warp * 8)
            row = warp * TOKENS_PER_WARP + (lane % 8) + ((lane // 8) % 2) * 8
            ka = s_kv + row * ROW_BYTES + (lane // 16) * 16
            zero = Float32(0.0)
            even = (zero, zero, zero, zero)
            odd = (zero, zero, zero, zero)
            for ks in cutlass.range_constexpr(QK_STEPS):
                a = ptx.ldmatrix_x4(ka + ks * 32)
                if cutlass.const_expr(ks % 2 == 1):
                    odd = ptx.mma_m16n8k32_e4m3(a, qf[ks], odd)
                else:
                    even = ptx.mma_m16n8k32_e4m3(a, qf[ks], even)
            s0 = ptx.add_rn(odd[0], even[0])
            s1 = ptx.add_rn(odd[1], even[1])
            s2 = ptx.add_rn(odd[2], even[2])
            s3 = ptx.add_rn(odd[3], even[3])

            # softmax statistics (base 2). s0: (token g, head 2 t4), s1: (g, 2 t4 + 1),
            # s2: (g + 8, 2 t4), s3: (g + 8, 2 t4 + 1)
            v0 = (vmask >> Uint32(g)) & Uint32(1)
            v1 = (vmask >> Uint32(g + 8)) & Uint32(1)
            ninf = Float32(NEG_INF)
            t0 = ptx.mul_rn(scale_log2, s0)
            t1 = ptx.mul_rn(scale_log2, s1)
            t2 = ptx.mul_rn(scale_log2, s2)
            t3 = ptx.mul_rn(scale_log2, s3)
            s0 = ninf
            s1 = ninf
            s2 = ninf
            s3 = ninf
            if v0 != Uint32(0):
                s0 = t0
                s1 = t1
            if v1 != Uint32(0):
                s2 = t2
                s3 = t3
            mx0 = self.nan_max(ninf, self.nan_max(s0, s2))
            mx1 = self.nan_max(ninf, self.nan_max(s1, s3))
            for o in cutlass.range_constexpr(2, 5):
                mx0 = self.nan_max(mx0, cute.arch.shuffle_sync_bfly(mx0, 1 << o))
                mx1 = self.nan_max(mx1, cute.arch.shuffle_sync_bfly(mx1, 1 << o))
            if g == 0:
                ptx.st_shared_v2_f32(s_red + (warp * HEADS + 2 * t4) * 4, mx0, mx1)
        ptx.sync_threads()

        if warp < QK_WARPS:
            ninf = Float32(NEG_INF)
            m0 = ninf
            m1 = ninf
            for w in cutlass.range_constexpr(QK_WARPS):
                a0, a1 = ptx.ld_shared_v2_f32(s_red + (w * HEADS + 2 * t4) * 4)
                m0 = self.nan_max(m0, a0)
                m1 = self.nan_max(m1, a1)
            # m = -inf: every token of this CTA's slice is masked (p = 0)
            ms0 = m0
            ms1 = m1
            if m0 == ninf:
                ms0 = Float32(0.0)
            if m1 == ninf:
                ms1 = Float32(0.0)
            eight = Float32(8.0)
            p00 = ptx.ex2_approx(ptx.add_rn(ptx.sub_rn(s0, ms0), eight))
            p01 = ptx.ex2_approx(ptx.add_rn(ptx.sub_rn(s1, ms1), eight))
            p10 = ptx.ex2_approx(ptx.add_rn(ptx.sub_rn(s2, ms0), eight))
            p11 = ptx.ex2_approx(ptx.add_rn(ptx.sub_rn(s3, ms1), eight))
            # P scaled by 2^8, split into an E4M3 term and an E4M3 residual; l sums both
            # terms
            h0 = ptx.cvt_rn_satfinite_e4m3x2_f32(p00, p01)
            h1 = ptx.cvt_rn_satfinite_e4m3x2_f32(p10, p11)
            fh0x, fh0y = ptx.e4m3x2_to_f32x2(h0)
            fh1x, fh1y = ptx.e4m3x2_to_f32x2(h1)
            lo0 = ptx.cvt_rn_satfinite_e4m3x2_f32(
                ptx.sub_rn(p00, fh0x), ptx.sub_rn(p01, fh0y)
            )
            lo1 = ptx.cvt_rn_satfinite_e4m3x2_f32(
                ptx.sub_rn(p10, fh1x), ptx.sub_rn(p11, fh1y)
            )
            fl0x, fl0y = ptx.e4m3x2_to_f32x2(lo0)
            fl1x, fl1y = ptx.e4m3x2_to_f32x2(lo1)
            l0 = ptx.add_rn(
                ptx.add_rn(ptx.add_rn(fh0x, fl0x), ptx.add_rn(fh1x, fl1x)), Float32(0.0)
            )
            l1 = ptx.add_rn(
                ptx.add_rn(ptx.add_rn(fh0y, fl0y), ptx.add_rn(fh1y, fl1y)), Float32(0.0)
            )
            tok = warp * TOKENS_PER_WARP + g
            at = (2 * t4) * P_PITCH + tok
            ptx.st_shared_u8(s_ph + at, h0 & 0xFF)
            ptx.st_shared_u8(s_ph + at + P_PITCH, h0 >> 8)
            ptx.st_shared_u8(s_ph + at + 8, h1 & 0xFF)
            ptx.st_shared_u8(s_ph + at + P_PITCH + 8, h1 >> 8)
            ptx.st_shared_u8(s_pl + at, lo0 & 0xFF)
            ptx.st_shared_u8(s_pl + at + P_PITCH, lo0 >> 8)
            ptx.st_shared_u8(s_pl + at + 8, lo1 & 0xFF)
            ptx.st_shared_u8(s_pl + at + P_PITCH + 8, lo1 >> 8)
            for o in cutlass.range_constexpr(2, 5):
                l0 = ptx.add_rn(l0, cute.arch.shuffle_sync_bfly(l0, 1 << o))
                l1 = ptx.add_rn(l1, cute.arch.shuffle_sync_bfly(l1, 1 << o))
            if g == 0:
                ptx.st_shared_v2_f32(
                    s_red + (QK_WARPS * HEADS + warp * HEADS + 2 * t4) * 4, l0, l1
                )

        # every warp reads every gathered row below
        self.wait_all_gathers(s_bar)
        ptx.sync_threads()

        # O^T = V^T P^T: warp owns dims [dims_per_warp * warp, +dims_per_warp) as 32-dim
        # tile pairs
        pah = s_ph + (lane % 8) * P_PITCH + ((lane // 8) % 2) * 16
        pal = s_pl + (lane % 8) * P_PITCH + ((lane // 8) % 2) * 16
        i8 = lane % 8
        vset = (lane // 8) % 2
        doff = (lane // 16) * 16
        trow = (i8 // 2) * 4 + vset * 2 + (i8 % 2)
        zero = Float32(0.0)
        if cutlass.const_expr(self.warps == 16):
            # 16 warps (up to 4 rows): accumulators in registers with static indices and
            # the 32-token k-step loop rolled two at a time;
            # the same MMAs in the same order as the unrolled form, so the same result
            acc_t = cute.make_rmem_tensor((pairs, 2, 4), Float32)
            for pr in cutlass.range_constexpr(pairs):
                for eo in cutlass.range_constexpr(2):
                    for k in cutlass.range_constexpr(4):
                        acc_t[pr, eo, k] = zero
            for ks in cutlass.range(TOKENS_PER_CTA // 32, unroll=2):
                bh = ptx.ldmatrix_x2(pah + ks * 32)
                bl = ptx.ldmatrix_x2(pal + ks * 32)
                for pr in cutlass.range_constexpr(pairs):
                    d0 = warp * dims_per_warp + pr * 32
                    ra = ptx.ldmatrix_x4(
                        s_kv + (ks * 32 + trow) * ROW_BYTES + d0 + doff, trans=True
                    )
                    rb = ptx.ldmatrix_x4(
                        s_kv + (ks * 32 + 16 + trow) * ROW_BYTES + d0 + doff,
                        trans=True,
                    )
                    ev = (
                        ptx.prmt(ra[0], ra[1], 0x6420),
                        ptx.prmt(ra[2], ra[3], 0x6420),
                        ptx.prmt(rb[0], rb[1], 0x6420),
                        ptx.prmt(rb[2], rb[3], 0x6420),
                    )
                    od = (
                        ptx.prmt(ra[0], ra[1], 0x7531),
                        ptx.prmt(ra[2], ra[3], 0x7531),
                        ptx.prmt(rb[0], rb[1], 0x7531),
                        ptx.prmt(rb[2], rb[3], 0x7531),
                    )
                    c0 = tuple(acc_t[pr, 0, k] for k in range(4))
                    c1 = tuple(acc_t[pr, 1, k] for k in range(4))
                    c0 = ptx.mma_m16n8k32_e4m3(ev, bh, c0)
                    c1 = ptx.mma_m16n8k32_e4m3(od, bh, c1)
                    c0 = ptx.mma_m16n8k32_e4m3(ev, bl, c0)
                    c1 = ptx.mma_m16n8k32_e4m3(od, bl, c1)
                    for k in cutlass.range_constexpr(4):
                        acc_t[pr, 0, k] = c0[k]
                        acc_t[pr, 1, k] = c1[k]
            acc = [
                [tuple(acc_t[pr, eo, k] for k in range(4)) for eo in range(2)]
                for pr in range(pairs)
            ]
        else:
            acc = [
                [(zero, zero, zero, zero), (zero, zero, zero, zero)]
                for _ in range(pairs)
            ]
            for ks in cutlass.range_constexpr(TOKENS_PER_CTA // 32):
                bh = ptx.ldmatrix_x2(pah + ks * 32)
                bl = ptx.ldmatrix_x2(pal + ks * 32)
                for pr in cutlass.range_constexpr(pairs):
                    d0 = warp * dims_per_warp + pr * 32
                    ra = ptx.ldmatrix_x4(
                        s_kv + (ks * 32 + trow) * ROW_BYTES + d0 + doff, trans=True
                    )
                    rb = ptx.ldmatrix_x4(
                        s_kv + (ks * 32 + 16 + trow) * ROW_BYTES + d0 + doff,
                        trans=True,
                    )
                    ev = (
                        ptx.prmt(ra[0], ra[1], 0x6420),
                        ptx.prmt(ra[2], ra[3], 0x6420),
                        ptx.prmt(rb[0], rb[1], 0x6420),
                        ptx.prmt(rb[2], rb[3], 0x6420),
                    )
                    od = (
                        ptx.prmt(ra[0], ra[1], 0x7531),
                        ptx.prmt(ra[2], ra[3], 0x7531),
                        ptx.prmt(rb[0], rb[1], 0x7531),
                        ptx.prmt(rb[2], rb[3], 0x7531),
                    )
                    acc[pr][0] = ptx.mma_m16n8k32_e4m3(ev, bh, acc[pr][0])
                    acc[pr][1] = ptx.mma_m16n8k32_e4m3(od, bh, acc[pr][1])
                    acc[pr][0] = ptx.mma_m16n8k32_e4m3(ev, bl, acc[pr][0])
                    acc[pr][1] = ptx.mma_m16n8k32_e4m3(od, bl, acc[pr][1])

        # combine: push this CTA's partials into the owners' shared memory (st.async)
        self.cluster_costart_wait()
        for pr in cutlass.range_constexpr(pairs):
            dbase = warp * dims_per_warp + pr * 32
            dst = dbase // DIMS_PER_CTA
            rrecv = ptx.mapa_shared_cluster(
                s_recv + crank * (DIMS_PER_CTA * HEADS * 4), dst
            )
            rbar = ptx.mapa_shared_cluster(s_bar + BAR_COMBINE * 8, dst)
            for eo in cutlass.range_constexpr(2):
                ld = dbase % DIMS_PER_CTA + 2 * g + eo
                c = acc[pr][eo]
                ptx.st_async_v2_f32(rrecv + (ld * HEADS + 2 * t4) * 4, c[0], c[1], rbar)
                ptx.st_async_v2_f32(
                    rrecv + ((ld + 16) * HEADS + 2 * t4) * 4, c[2], c[3], rbar
                )
        if tid < CLUSTER * 2 * HEADS:
            dst = tid // (2 * HEADS)
            which = (tid // HEADS) % 2
            h = tid % HEADS
            v = Float32(NEG_INF)
            if which == 0:
                for w in cutlass.range_constexpr(QK_WARPS):
                    v = self.nan_max(v, ptx.ld_shared_f32(s_red + (w * HEADS + h) * 4))
            else:
                v = ptx.add_rn(
                    ptx.ld_shared_f32(s_red + (QK_WARPS * HEADS + h) * 4), Float32(0.0)
                )
                for w in cutlass.range_constexpr(1, QK_WARPS):
                    v = ptx.add_rn(
                        v,
                        ptx.ld_shared_f32(
                            s_red + (QK_WARPS * HEADS + w * HEADS + h) * 4
                        ),
                    )
            remote = ptx.mapa_shared_cluster(
                s_stats + ((crank * 2 + which) * HEADS + h) * 4, dst
            )
            ptx.st_async_f32(
                remote, v, ptx.mapa_shared_cluster(s_bar + BAR_COMBINE * 8, dst)
            )
        self.wait_combine(s_bar + BAR_COMBINE * 8)
        self.exit_arrive()

        # owner: rescale and sum the 16 partials of its 32 dims (one output element per
        # thread)
        if tid < HEADS * DIMS_PER_CTA:
            ld = tid // HEADS
            h = tid % HEADS
            ninf = Float32(NEG_INF)
            mi = [
                ptx.ld_shared_f32(s_stats + (i * 2 * HEADS + h) * 4)
                for i in range(CLUSTER)
            ]
            mx = ninf
            for i in cutlass.range_constexpr(CLUSTER):
                mx = self.nan_max(mx, mi[i])
            lsum = Float32(0.0)
            a = Float32(0.0)
            if mx != ninf:
                for i in cutlass.range_constexpr(CLUSTER):
                    w = ptx.ex2_approx(ptx.sub_rn(mi[i], mx))
                    lsum = ptx.fma_rn(
                        w,
                        ptx.ld_shared_f32(s_stats + ((i * 2 + 1) * HEADS + h) * 4),
                        lsum,
                    )
                    a = ptx.fma_rn(
                        w,
                        ptx.ld_shared_f32(
                            s_recv + ((i * DIMS_PER_CTA + ld) * HEADS + h) * 4
                        ),
                        a,
                    )
            res = Float32(0.0)
            if lsum != Float32(0.0):
                res = ptx.mul_rn(a, ptx.div_rn(bmm2_scale, lsum))
            dst_addr = (
                out
                + (
                    (Int64(r) * HEADS + Int64(h)) * V_DIM
                    + Int64(crank * DIMS_PER_CTA + ld)
                )
                * 2
            )
            ptx.st_global_b16(dst_addr, ptx.cvt_rn_bf16_f32(res))
        self.release_outputs()
        self.exit_wait()

    # ------------------------------------------------------------------ steps
    @cute.jit
    def map_logical(
        self,
        tok_id: Int32,
        r: Int32,
        j: Int32,
        lane: Int32,
        request_of_row: Int64,
        block_table: Int64,
        block_table_stride: Int64,
        max_blocks: Int32,
        block_shift: Int32,
        block_rows: Int64,
        phys_out: Int64,
    ):
        """Logical token position -> physical cache row (-1: none),
        recorded on request."""
        phys = Int32(-1)
        if tok_id >= 0:
            blk = Int32(Uint32(tok_id) >> Uint32(block_shift))
            if blk < max_blocks:
                req = ptx.ld_global_s32(request_of_row + Int64(r) * 4)
                at = (
                    block_table
                    + (Int64(block_table_stride) * Int64(req) + Int64(blk)) * 4
                )
                page = ptx.ld_global_s32(at)
                mask = ~(Int32(-1) << block_shift)
                phys = page * Int32(block_rows) + (tok_id & mask)
        if phys_out != Int64(0):  # noqa: SIM102
            if lane < TOKENS_PER_WARP:
                ptx.st_global_u32(phys_out + (Int64(r) * TOPK + Int64(j)) * 4, phys)
        return phys

    @cute.jit
    def gather(
        self,
        s_kv: Int32,
        s_bar: Int32,
        kv: Int64,
        tok_id: Int32,
        ok,
        vmask: Uint32,
        warp: Int32,
        lane: Int32,
    ):
        """This warp's 16 cache rows into shared memory (bulk copies on its mbarrier;
        zero rows when
        not selected)."""
        bar = s_bar + warp * 8
        if lane == 0:
            ptx.mbarrier_arrive_expect_tx(
                bar, Int32(cute.arch.popc(vmask)) * QK_DIM, ".release.cta"
            )
        cute.arch.sync_warp()
        dst = s_kv + (warp * TOKENS_PER_WARP + lane) * ROW_BYTES
        if ok:
            ptx.bulk_copy_g2s(dst, kv + Int64(tok_id) * QK_DIM, QK_DIM, bar)
        else:
            if lane < TOKENS_PER_WARP:
                zero = (Uint32(0), Uint32(0), Uint32(0), Uint32(0))
                for c in cutlass.range_constexpr(QK_DIM // 16):
                    ptx.st_shared_v4(dst + c * 16, zero)
        self.zero_fill_visible()

    @cute.jit
    def nan_max(self, a: Float32, b: Float32) -> Float32:
        """NaN-propagating max (max.NaN.f32): a NaN score keeps its softmax NaN, as an
        fp64 reference does."""
        return ptx.max_nan(a, b)

    @cute.jit
    def wait_inputs(self):
        ptx.pdl_wait()
        if cutlass.const_expr(self.early_trigger):
            ptx.pdl_launch_dependents()
        ptx.fence_proxy_async_global()

    @cute.jit
    def zero_fill_visible(self):
        """One lane's zero-fill stores -> every lane's ldmatrix of that row."""
        cute.arch.sync_warp()

    @cute.jit
    def wait_q(self, bar: Int32):
        ptx.mbarrier_wait_parity(bar, 0, ".acquire.cta")

    @cute.jit
    def wait_gather(self, bar: Int32):
        ptx.mbarrier_wait_parity(bar, 0, ".acquire.cta")

    @cute.jit
    def wait_all_gathers(self, s_bar: Int32):
        for w in cutlass.range_constexpr(QK_WARPS):
            ptx.mbarrier_wait_parity(s_bar + w * 8, 0, ".acquire.cta")

    @cute.jit
    def cluster_costart_arrive(self):
        ptx.cluster_arrive_relaxed()

    @cute.jit
    def cluster_costart_wait(self):
        ptx.cluster_wait_acquire()

    @cute.jit
    def wait_combine(self, bar: Int32):
        ptx.mbarrier_wait_parity(bar, 0, ".acquire.cluster")

    @cute.jit
    def exit_arrive(self):
        ptx.cluster_arrive_relaxed()

    @cute.jit
    def release_outputs(self):
        ptx.fence_proxy_async_global()

    @cute.jit
    def exit_wait(self):
        ptx.cluster_wait_acquire()
