# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""MLA attention prologue for decode / verify rows (rows <= 8): everything between the
fused q_a / kv_a projection and the sparse attention core, in one cluster launch, with
an engine's own rounding points.

For each row ``m < rows`` (8 local heads, q_lora 2048, kv_lora 512, nope 192, rope 64):

    q_a = bf16(RMSNorm(q_c[m]) * q_norm_w)          (optionally written to qc_out)
    q = bf16(q_a @ W_qb^T)                          [8, 256] = nope 192 | rope 64
    mqa_q[m, h, :512] = e4m3(bf16(q_nope[h] @ W_UK[h]^T) / q_scale)
    mqa_q[m, h, 512:] = e4m3(RoPE(q_rope[h]) / q_scale)
    MLA row of slot[m] = e4m3(RMSNorm(kv_c[m]) * kv_norm_w / k_scale)
                         | e4m3(RoPE(k_pe[m]) / k_scale)
    full (indexer) layers: index-K row = e4m3(RoPE(LayerNorm(idx_k[m])) / s) with
    s from max(amax, 1e-4) / 448 (stored with the row); top-k buffer rows set to -1.

Work split: 9 clusters of 8 CTAs (256 threads). Clusters 0..7 = heads: CTA ``r`` owns
q_b rows ``[32 r, 32 r + 32)`` of its head and absorbed columns ``[64 r, 64 r + 64)``;
its packed q_b (128 KiB) and W_UK^T (24 KiB) slices are bulk-copied before the PDL wait.
Ranks 0..5 publish their q_nope rows in shared memory; after a cluster barrier every
rank gathers the head's 192 nope values over distributed shared memory. Cluster 8: CTA 0
writes the MLA rows, CTA 1 the index-K rows, CTAs 2..7 the top-k fill.

Option fields (compile time): ``max_rows`` (8 or 16), ``numerics``
(``PrologueNumerics``: the sum orders, rounding points and scale rule, each named by
the arithmetic it selects), ``rope_cache_fp32`` and ``index_rope_cache_fp32`` (cos /
sin caches FP32, else BF16; layout [positions, 64] = cos | sin),
``index_scale_ue8m0x4`` (with the power-of-two index-K scale: store it as four equal
UE8M0 bytes instead of an FP32). The layer kind (indexer or not) and ``qc_out`` (null
= not written) are runtime.

Ordering: dependents are triggered at entry (``trigger_at_end``: after each CTA's last
output store and fence instead, so no prologue CTA is still resident when the dependents'
CTAs are placed; outputs are the same); only constant weights / norm parameters /
scales are read before ``griddepcontrol.wait``, and every CTA executes the wait on every
path; ``fence.proxy.async.global`` follows the wait and each thread's last output store.
The q_nope hand-off publishes with ``barrier.cluster.arrive`` (release) / ``wait``
(acquire); a second arrive after the remote reads and the final wait keep every CTA's
shared memory alive while its peers read it.
"""

from dataclasses import dataclass

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
from cutlass import Float32, Int32, Int64, Uint32

from ..primitives import ptx_core as ptx

HEADS = 8
Q_LORA = 2048
KV_LORA = 512
ROPE = 64
NOPE = 192
Q_HEAD = NOPE + ROPE  # 256
CLUSTER = 8
THREADS = 256
MQA_DIM = KV_LORA + ROPE  # 576 bytes per head per row
INDEX_DIM = 128
GRID = (HEADS + 1) * CLUSTER
NORM_ORDERS = ("strided_8x256", "lane_contiguous")
INDEX_MEAN_SUMS = ("sequential", "pairwise")
INDEX_ROPE_ODD = ("round_x1_cos", "round_x0_sin")
INDEX_SCALES = ("fp32_amax", "ue8m0")
Q_ROPES = ("one_product_rounded", "both_products_rounded")


@dataclass(frozen=True)
class PrologueNumerics:
    """The prologue's numeric options, each named by the arithmetic it selects.

    ``norm_order`` (q_a and kv_a RMSNorm sums of squares): ``"strided_8x256"``: one
    warp per row; lane l adds the separately rounded squares of elements 8 l + 256 b +
    v (q_a: b, v = 0..7 in (b, v) order; kv_a: b = 0, 1) to 0, then a butterfly xor 1,
    2, 4, 8, 16. ``"lane_contiguous"``: each thread accumulates the squares of its
    contiguous elements with FMA from 0 (kv_a: lane l owns 16 l .. 16 l + 15), then a
    butterfly xor 16, 8, 4, 2, 1; the 8 warp sums of a q_a row are added in warp order.
    ``norm_weight_plus_zero``: the norm weights enter as ``w + 0``.
    ``kv_norm_bf16``: the kv_a norm output is rounded to BF16 before the FP8
    quantization. ``skip_slot0``: cache slot 0 (a padding slot) is never written; else
    only negative slots are skipped.
    ``index_mean_sum`` (index-K LayerNorm, a lane's 4 elements): ``"sequential"``:
    (((0 + x0) + x1) + x2) + x3; ``"pairwise"``: (x0 + x1) + (x2 + x3).
    ``index_rope_odd`` (index-K RoPE, odd dims e0 sin + e1 cos): which product is
    rounded before the fused other one. ``index_k_bf16``: the RoPE output is rounded
    to BF16 before the scale. ``index_scale``: ``"fp32_amax"``: s = max(amax, 1e-4) /
    448, values x (1 / s) clipped to +-448; ``"ue8m0"``: s rounded up to a power of
    two, values x / s.
    ``q_rope`` (q_pe RoPE): ``"one_product_rounded"``: x0 cos - x1 sin with x1 sin
    rounded and x1 cos + x0 sin with x0 sin rounded, the other product fused;
    ``"both_products_rounded"``: both products rounded, then the difference / sum.
    """

    norm_order: str
    norm_weight_plus_zero: bool
    kv_norm_bf16: bool
    skip_slot0: bool
    index_mean_sum: str
    index_rope_odd: str
    index_k_bf16: bool
    index_scale: str
    q_rope: str

    def __post_init__(self):
        for value, allowed in (
            (self.norm_order, NORM_ORDERS),
            (self.index_mean_sum, INDEX_MEAN_SUMS),
            (self.index_rope_odd, INDEX_ROPE_ODD),
            (self.index_scale, INDEX_SCALES),
            (self.q_rope, Q_ROPES),
        ):
            if value not in allowed:
                raise ValueError(f"{value!r} is not one of {allowed}")


# shared memory of a head CTA (bytes)
WQB_BYTES = 2 * (Q_LORA // 32) * 1024  # 2 row tiles x 64 k-steps x 1 KiB
WUK_BYTES = 4 * (NOPE // 32) * 1024  # 4 row tiles x 6 k-steps x 1 KiB
XS_STRIDE = Q_LORA * 2 + 64
QN_STRIDE = NOPE * 2 + 64


def layout(max_rows):
    off = {"wqb": 0, "wuk": WQB_BYTES}
    off["xs"] = off["wuk"] + WUK_BYTES
    if max_rows == 8:
        off["pub"] = off["xs"] + max_rows * XS_STRIDE
        off["qn"] = off["pub"] + max_rows * 32 * 2
        off["red"] = off["qn"] + max_rows * QN_STRIDE
        off["ssq"] = off["red"] + 8 * 2 * 32 * 16
    else:
        # 16 rows: the partials and the gathered q_nope rows alias the token rows,
        # which are dead once the q_b products are done (a CTA barrier separates them)
        off["red"] = off["xs"]
        off["qn"] = off["red"] + 8 * 2 * 2 * 32 * 16
        off["pub"] = off["xs"] + max_rows * XS_STRIDE
        off["ssq"] = off["pub"] + max_rows * 32 * 2
    off["mbar"] = off["ssq"] + 8 * max_rows * 4
    off["total"] = off["mbar"] + 16
    return off


class MlaPrologue:
    """See the module docstring. Call arguments (device addresses; row strides in
    elements; cache strides in bytes): ``q_c``, ``kv_c``, ``k_pe``, ``idx_k`` with their
    row strides (``idx_k`` unused on shared layers), ``positions``, ``slot_mapping``,
    ``idx_slot_mapping`` (int64), ``rows``, ``has_indexer``, packed ``w_qb`` / ``w_uk``
    (``packing.pack_q_b`` / ``pack_w_uk_t``), ``q_norm_w`` / ``kv_norm_w`` (BF16),
    ``q_eps`` / ``kv_eps``, ``rope_cache`` and its row stride, ``q_scale`` / ``k_scale``
    (FP32 device scalars), ``ik_w`` / ``ik_b`` (FP32 [128]), ``ik_eps``,
    ``index_rope_cache`` and its row stride, ``mqa_q`` and its row stride, the MLA cache
    (base, block stride, entry stride, block size), the index-K cache (base, block
    stride, block size), ``topk`` (int32) with its row stride and
    width, ``qc_out`` (or 0) and its row stride, then the stream."""

    def __init__(
        self,
        max_rows: int,
        numerics: PrologueNumerics,
        rope_cache_fp32: bool = True,
        index_rope_cache_fp32: bool = True,
        index_scale_ue8m0x4: bool = False,
        trigger_at_end: bool = False,
    ):
        if max_rows not in (8, 16):
            raise ValueError(f"max_rows must be 8 or 16, got {max_rows}")
        self.max_rows = max_rows
        self.num = numerics
        self.rope_cache_fp32 = rope_cache_fp32
        self.index_rope_cache_fp32 = index_rope_cache_fp32
        if index_scale_ue8m0x4 and numerics.index_scale != "ue8m0":
            raise ValueError(
                "index_scale_ue8m0x4 applies to the power-of-two scale only"
            )
        self.index_scale_ue8m0x4 = index_scale_ue8m0x4
        self.trigger_at_end = trigger_at_end
        self.off = layout(max_rows)

    @cute.jit
    def __call__(
        self,
        q_c: Int64,
        q_c_stride: Int64,
        kv_c: Int64,
        kv_c_stride: Int64,
        k_pe: Int64,
        k_pe_stride: Int64,
        idx_k: Int64,
        idx_k_stride: Int64,
        positions: Int64,
        slot_mapping: Int64,
        idx_slot_mapping: Int64,
        rows: Int32,
        has_indexer: Int32,
        w_qb: Int64,
        w_uk: Int64,
        q_norm_w: Int64,
        kv_norm_w: Int64,
        q_eps: Float32,
        kv_eps: Float32,
        rope_cache: Int64,
        rope_stride: Int64,
        q_scale: Int64,
        k_scale: Int64,
        ik_w: Int64,
        ik_b: Int64,
        ik_eps: Float32,
        index_rope_cache: Int64,
        index_rope_stride: Int64,
        mqa_q: Int64,
        mqa_stride: Int64,
        mla_cache: Int64,
        mla_block_stride: Int64,
        mla_entry_stride: Int64,
        mla_block_size: Int64,
        idx_cache: Int64,
        idx_block_stride: Int64,
        idx_block_size: Int64,
        topk: Int64,
        topk_stride: Int64,
        topk_width: Int32,
        qc_out: Int64,
        qc_out_stride: Int64,
        stream: cuda.CUstream,
    ):
        self.kernel(
            q_c,
            q_c_stride,
            kv_c,
            kv_c_stride,
            k_pe,
            k_pe_stride,
            idx_k,
            idx_k_stride,
            positions,
            slot_mapping,
            idx_slot_mapping,
            rows,
            has_indexer,
            w_qb,
            w_uk,
            q_norm_w,
            kv_norm_w,
            q_eps,
            kv_eps,
            rope_cache,
            rope_stride,
            q_scale,
            k_scale,
            ik_w,
            ik_b,
            ik_eps,
            index_rope_cache,
            index_rope_stride,
            mqa_q,
            mqa_stride,
            mla_cache,
            mla_block_stride,
            mla_entry_stride,
            mla_block_size,
            idx_cache,
            idx_block_stride,
            idx_block_size,
            topk,
            topk_stride,
            topk_width,
            qc_out,
            qc_out_stride,
        ).launch(
            grid=[GRID, 1, 1],
            block=[THREADS, 1, 1],
            cluster=[CLUSTER, 1, 1],
            stream=stream,
            use_pdl=True,
            max_number_threads=[THREADS, 1, 1],
            min_blocks_per_mp=1,
        )

    @cute.kernel
    def kernel(
        self,
        q_c: Int64,
        q_c_stride: Int64,
        kv_c: Int64,
        kv_c_stride: Int64,
        k_pe: Int64,
        k_pe_stride: Int64,
        idx_k: Int64,
        idx_k_stride: Int64,
        positions: Int64,
        slot_mapping: Int64,
        idx_slot_mapping: Int64,
        rows: Int32,
        has_indexer: Int32,
        w_qb: Int64,
        w_uk: Int64,
        q_norm_w: Int64,
        kv_norm_w: Int64,
        q_eps: Float32,
        kv_eps: Float32,
        rope_cache: Int64,
        rope_stride: Int64,
        q_scale: Int64,
        k_scale: Int64,
        ik_w: Int64,
        ik_b: Int64,
        ik_eps: Float32,
        index_rope_cache: Int64,
        index_rope_stride: Int64,
        mqa_q: Int64,
        mqa_stride: Int64,
        mla_cache: Int64,
        mla_block_stride: Int64,
        mla_entry_stride: Int64,
        mla_block_size: Int64,
        idx_cache: Int64,
        idx_block_stride: Int64,
        idx_block_size: Int64,
        topk: Int64,
        topk_stride: Int64,
        topk_width: Int32,
        qc_out: Int64,
        qc_out_stride: Int64,
    ):
        tid_, _, _ = cute.arch.thread_idx()
        cta_, _, _ = cute.arch.block_idx()
        tid = Int32(tid_)
        cta = Int32(cta_)
        smem = cutlass.utils.SmemAllocator()
        buf = smem.allocate_tensor(
            cutlass.Uint8, cute.make_layout(self.off["total"]), byte_alignment=1024
        )
        base = ptx.shared_address(buf.iterator)

        if cutlass.const_expr(not self.trigger_at_end):
            ptx.pdl_launch_dependents()
        cluster_id = cta // CLUSTER
        r = cta % CLUSTER
        if cluster_id < HEADS:
            if cutlass.const_expr(self.max_rows == 16):
                self.head_cta16(
                    base,
                    cluster_id,
                    r,
                    tid,
                    q_c,
                    q_c_stride,
                    positions,
                    rows,
                    w_qb,
                    w_uk,
                    q_norm_w,
                    q_eps,
                    rope_cache,
                    rope_stride,
                    q_scale,
                    mqa_q,
                    mqa_stride,
                    qc_out,
                    qc_out_stride,
                )
            else:
                self.head_cta(
                    base,
                    cluster_id,
                    r,
                    tid,
                    q_c,
                    q_c_stride,
                    positions,
                    rows,
                    w_qb,
                    w_uk,
                    q_norm_w,
                    q_eps,
                    rope_cache,
                    rope_stride,
                    q_scale,
                    mqa_q,
                    mqa_stride,
                    qc_out,
                    qc_out_stride,
                )
        else:
            self.kv_cta(
                r,
                tid,
                kv_c,
                kv_c_stride,
                k_pe,
                k_pe_stride,
                idx_k,
                idx_k_stride,
                positions,
                slot_mapping,
                idx_slot_mapping,
                rows,
                has_indexer,
                kv_norm_w,
                kv_eps,
                rope_cache,
                rope_stride,
                k_scale,
                ik_w,
                ik_b,
                ik_eps,
                index_rope_cache,
                index_rope_stride,
                mla_cache,
                mla_block_stride,
                mla_entry_stride,
                mla_block_size,
                idx_cache,
                idx_block_stride,
                idx_block_size,
                topk,
                topk_stride,
                topk_width,
            )
            self.release_outputs()
        if cutlass.const_expr(self.trigger_at_end):
            ptx.pdl_launch_dependents()

    # ------------------------------------------------------------------ head clusters
    @cute.jit
    def head_cta(
        self,
        base: Int32,
        h: Int32,
        r: Int32,
        tid: Int32,
        q_c: Int64,
        q_c_stride: Int64,
        positions: Int64,
        rows: Int32,
        w_qb: Int64,
        w_uk: Int64,
        q_norm_w: Int64,
        q_eps: Float32,
        rope_cache: Int64,
        rope_stride: Int64,
        q_scale: Int64,
        mqa_q: Int64,
        mqa_stride: Int64,
        qc_out: Int64,
        qc_out_stride: Int64,
    ):
        off = self.off
        max_rows = self.max_rows
        lane = tid % 32
        warp = tid // 32
        bar = base + off["mbar"]
        s_xs = base + off["xs"]
        s_red = base + off["red"]
        s_ssq = base + off["ssq"]
        s_pub = base + off["pub"]
        s_qn = base + off["qn"]
        # constant weights: q_b slice (two 64 KiB halves) + W_UK^T slice (24 KiB),
        # before the PDL wait
        if tid == 0:
            ptx.mbarrier_init(bar, 1)
            ptx.fence_mbarrier_init_release_cluster()
            ptx.fence_proxy_async_shared_cta()
            ptx.mbarrier_arrive_expect_tx(bar, WQB_BYTES + WUK_BYTES)
            src_qb = w_qb + Int64(h * 16 + 2 * r) * ((Q_LORA // 32) * 1024)
            src_uk = w_uk + Int64(h * 32 + 4 * r) * ((NOPE // 32) * 1024)
            for part in cutlass.range_constexpr(2):
                ptx.bulk_copy_g2s(
                    base + off["wqb"] + part * (WQB_BYTES // 2),
                    src_qb + part * (WQB_BYTES // 2),
                    WQB_BYTES // 2,
                    bar,
                )
            ptx.bulk_copy_g2s(base + off["wuk"], src_uk, WUK_BYTES, bar)
        qw = ptx.ld_global_nc_v4(q_norm_w + Int64(tid) * 16)
        inv_qs = ptx.rcp_rn(ptx.ld_global_nc_f32(q_scale))
        ptx.sync_threads()

        self.wait_q_inputs()

        # every predecessor-data load first (their latencies overlap)
        qv = []
        for n in cutlass.range_constexpr(max_rows):
            qv.append(
                self.load16(
                    q_c + (Int64(n) * q_c_stride + Int64(tid * 8)) * 2, n < rows
                )
            )
        rope_c = Float32(0.0)
        rope_s = Float32(0.0)
        if (r >= NOPE // 32) & (tid < 32 * rows):
            pos = ptx.ld_global_s64(positions + Int64(tid // 32) * 8)
            pr = ((r - NOPE // 32) * 32 + lane) // 2
            rope_c = self.rope_value(
                rope_cache, rope_stride, pos, pr, self.rope_cache_fp32
            )
            rope_s = self.rope_value(
                rope_cache, rope_stride, pos, 32 + pr, self.rope_cache_fp32
            )
        if cutlass.const_expr(self.num.norm_order == "strided_8x256"):
            self.q_norm_sums_strided(s_xs, s_ssq, qv, warp, lane, tid, rows)
        else:
            self.q_norm_sums_contiguous(s_ssq, qv, warp, lane, rows)
        ptx.sync_threads()
        wv = []
        for e in cutlass.range_constexpr(4):
            lo, hi = ptx.bf16x2_to_f32x2(qw[e])
            if cutlass.const_expr(self.num.norm_weight_plus_zero):
                lo = ptx.add_rn(lo, Float32(0.0))
                hi = ptx.add_rn(hi, Float32(0.0))
            wv.append(lo)
            wv.append(hi)
        for n in cutlass.range_constexpr(max_rows):
            if n < rows:
                if cutlass.const_expr(self.num.norm_order == "strided_8x256"):
                    s = ptx.ld_shared_f32(s_ssq + n * 4)
                else:
                    s = Float32(0.0)
                    for w in cutlass.range_constexpr(8):
                        s = ptx.add_rn(
                            s, ptx.ld_shared_f32(s_ssq + (w * max_rows + n) * 4)
                        )
                rr = ptx.rsqrt_approx(ptx.fma_rn(s, Float32(1.0 / Q_LORA), q_eps))
                words = []
                for e in cutlass.range_constexpr(4):
                    x0, x1 = ptx.bf16x2_to_f32x2(qv[n][e])
                    words.append(
                        ptx.cvt_rn_bf16x2_f32(
                            ptx.mul_rn(ptx.mul_rn(x0, rr), wv[2 * e]),
                            ptx.mul_rn(ptx.mul_rn(x1, rr), wv[2 * e + 1]),
                        )
                    )
                packed = (words[0], words[1], words[2], words[3])
                ptx.st_shared_v4(s_xs + n * XS_STRIDE + tid * 16, packed)
                if (qc_out != Int64(0)) & (h == 0) & (r == 0):
                    ptx.st_global_v4(
                        qc_out + (Int64(n) * qc_out_stride + Int64(tid * 8)) * 2, packed
                    )
        ptx.sync_threads()

        self.wait_weights(bar)

        g = lane // 4
        t = lane % 4
        # q_b MMA (tokens on N = 8): warp w covers k-steps [8 w, 8 w + 8) for both
        # 16-row tiles
        zero = Float32(0.0)
        c = [(zero, zero, zero, zero), (zero, zero, zero, zero)]
        for i in cutlass.range_constexpr(8):
            s_ = warp * 8 + i
            bq = self.lds16(s_xs + g * XS_STRIDE + (s_ * 32 + 8 * t) * 2, g < rows)
            for j in cutlass.range_constexpr(2):
                blk = base + off["wqb"] + (j * (Q_LORA // 32) + s_) * 1024
                lo = ptx.ld_shared_v4(blk + lane * 16)
                hi = ptx.ld_shared_v4(blk + 512 + lane * 16)
                c1 = ptx.mma_m16n8k16_bf16(
                    (lo[0], hi[0], lo[1], hi[1]), (bq[0], bq[1]), c[j]
                )
                c[j] = ptx.mma_m16n8k16_bf16(
                    (lo[2], hi[2], lo[3], hi[3]), (bq[2], bq[3]), c1
                )
        for j in cutlass.range_constexpr(2):
            ptx.st_shared_v4(
                s_red + ((warp * 2 + j) * 32 + lane) * 16,
                (
                    ptx.f32_bits(c[j][0]),
                    ptx.f32_bits(c[j][1]),
                    ptx.f32_bits(c[j][2]),
                    ptx.f32_bits(c[j][3]),
                ),
            )
        self.sync_partials()
        if tid < 32 * rows:
            rho = tid % 32
            n = tid // 32
            jt = rho // 16
            rr_ = rho % 16
            ln = (rr_ % 8) * 4 + n // 2
            reg = (rr_ // 8) * 2 + n % 2
            v = Float32(0.0)
            for w in cutlass.range_constexpr(8):
                v = ptx.add_rn(
                    v,
                    ptx.ld_shared_f32(s_red + ((w * 2 + jt) * 32 + ln) * 16 + reg * 4),
                )
            qb = ptx.cvt_rn_bf16_f32(v)
            if r < NOPE // 32:
                ptx.st_shared_b16(s_pub + (n * 32 + rho) * 2, qb)
            else:
                d = (r - NOPE // 32) * 32 + rho
                own = ptx.bf16_to_f32(qb)
                other = cute.arch.shuffle_sync_bfly(own, 1)
                o = Float32(0.0)
                if d % 2 == 0:
                    o = self.q_rope_even(own, other, rope_c, rope_s)
                else:
                    o = self.q_rope_odd(other, own, rope_c, rope_s)
                o = ptx.mul_rn(o, inv_qs)
                dst = mqa_q + Int64(n) * mqa_stride + Int64(h * MQA_DIM + KV_LORA + d)
                ptx.st_global_u8(
                    dst,
                    cutlass.Uint16(
                        Uint32(ptx.cvt_rn_satfinite_e4m3x2_f32(o, Float32(0.0)))
                        & Uint32(0xFF)
                    ),
                )
        # q_nope hand-off: ranks 0..5 published 32 rows each
        self.handoff(base, tid, rows)
        self.done_reading()
        ptx.sync_threads()
        # W_UK absorb: rows l in [64 r, 64 r + 64) (4 tiles), K = 192 (6 k-steps); warp
        # -> (tile w % 4, half w // 4)
        tile = warp % 4
        half = warp // 4
        cw = (zero, zero, zero, zero)
        for i in cutlass.range_constexpr(3):
            s_ = half * 3 + i
            bq = self.lds16(s_qn + g * QN_STRIDE + (s_ * 32 + 8 * t) * 2, g < rows)
            blk = base + off["wuk"] + (tile * (NOPE // 32) + s_) * 1024
            lo = ptx.ld_shared_v4(blk + lane * 16)
            hi = ptx.ld_shared_v4(blk + 512 + lane * 16)
            c1 = ptx.mma_m16n8k16_bf16((lo[0], hi[0], lo[1], hi[1]), (bq[0], bq[1]), cw)
            cw = ptx.mma_m16n8k16_bf16((lo[2], hi[2], lo[3], hi[3]), (bq[2], bq[3]), c1)
        ptx.st_shared_v4(
            s_red + (warp * 32 + lane) * 16,
            (
                ptx.f32_bits(cw[0]),
                ptx.f32_bits(cw[1]),
                ptx.f32_bits(cw[2]),
                ptx.f32_bits(cw[3]),
            ),
        )
        self.sync_partials()
        for k in cutlass.range_constexpr((64 * max_rows + THREADS - 1) // THREADS):
            idx = tid + k * THREADS
            if idx < 64 * rows:
                rho = idx % 64
                n = idx // 64
                tl = rho // 16
                rr_ = rho % 16
                ln = (rr_ % 8) * 4 + n // 2
                reg = (rr_ // 8) * 2 + n % 2
                v = ptx.add_rn(
                    ptx.ld_shared_f32(s_red + (tl * 32 + ln) * 16 + reg * 4),
                    ptx.ld_shared_f32(s_red + ((tl + 4) * 32 + ln) * 16 + reg * 4),
                )
                qb = ptx.bf16_to_f32(ptx.cvt_rn_bf16_f32(v))
                dst = mqa_q + Int64(n) * mqa_stride + Int64(h * MQA_DIM + r * 64 + rho)
                ptx.st_global_u8(
                    dst,
                    cutlass.Uint16(
                        Uint32(
                            ptx.cvt_rn_satfinite_e4m3x2_f32(
                                ptx.mul_rn(qb, inv_qs), Float32(0.0)
                            )
                        )
                        & Uint32(0xFF)
                    ),
                )
        self.release_outputs()
        self.exit_wait()

    @cute.jit
    def head_cta16(
        self,
        base: Int32,
        h: Int32,
        r: Int32,
        tid: Int32,
        q_c: Int64,
        q_c_stride: Int64,
        positions: Int64,
        rows: Int32,
        w_qb: Int64,
        w_uk: Int64,
        q_norm_w: Int64,
        q_eps: Float32,
        rope_cache: Int64,
        rope_stride: Int64,
        q_scale: Int64,
        mqa_q: Int64,
        mqa_stride: Int64,
        qc_out: Int64,
        qc_out_stride: Int64,
    ):
        """The head CTA at up to 16 rows. Per row, every operation and its order are
        the 8-row form's (rows are independent columns of the tensor-core products),
        so each row equals that row of an 8-row call. Tokens use two 8-wide N tiles;
        the raw and normalized rows live in the XS slots (re-read, not registers); the
        partials and the gathered q_nope rows reuse the XS region after the q_b
        products (a CTA barrier separates them)."""
        off = self.off
        max_rows = self.max_rows
        lane = tid % 32
        warp = tid // 32
        bar = base + off["mbar"]
        s_xs = base + off["xs"]
        s_red = base + off["red"]
        s_ssq = base + off["ssq"]
        s_pub = base + off["pub"]
        s_qn = base + off["qn"]
        strided = self.num.norm_order == "strided_8x256"
        if tid == 0:
            ptx.mbarrier_init(bar, 1)
            ptx.fence_mbarrier_init_release_cluster()
            ptx.fence_proxy_async_shared_cta()
            ptx.mbarrier_arrive_expect_tx(bar, WQB_BYTES + WUK_BYTES)
            src_qb = w_qb + Int64(h * 16 + 2 * r) * ((Q_LORA // 32) * 1024)
            src_uk = w_uk + Int64(h * 32 + 4 * r) * ((NOPE // 32) * 1024)
            for part in cutlass.range_constexpr(2):
                ptx.bulk_copy_g2s(
                    base + off["wqb"] + part * (WQB_BYTES // 2),
                    src_qb + part * (WQB_BYTES // 2),
                    WQB_BYTES // 2,
                    bar,
                )
            ptx.bulk_copy_g2s(base + off["wuk"], src_uk, WUK_BYTES, bar)
        qw = ptx.ld_global_nc_v4(q_norm_w + Int64(tid) * 16)
        inv_qs = ptx.rcp_rn(ptx.ld_global_nc_f32(q_scale))
        ptx.sync_threads()

        self.wait_q_inputs()

        qv = []
        for n in cutlass.range_constexpr(max_rows):
            qv.append(
                self.load16(
                    q_c + (Int64(n) * q_c_stride + Int64(tid * 8)) * 2, n < rows
                )
            )
        # RoPE CTAs: two (token, pair) slots per thread (token tid / 32 and + 8)
        rope_c = [Float32(0.0), Float32(0.0)]
        rope_s = [Float32(0.0), Float32(0.0)]
        for it in cutlass.range_constexpr(2):
            rope_c[it], rope_s[it] = self.rope_pair(
                (r >= NOPE // 32) & (tid + THREADS * it < 32 * rows),
                positions,
                (tid + THREADS * it) // 32,
                ((r - NOPE // 32) * 32 + lane) // 2,
                rope_cache,
                rope_stride,
            )
        for n in cutlass.range_constexpr(max_rows):
            if n < rows:
                ptx.st_shared_v4(s_xs + n * XS_STRIDE + tid * 16, qv[n])
        if cutlass.const_expr(strided):
            ptx.sync_threads()
            for grp in cutlass.range_constexpr(2):
                n = warp + 8 * grp
                if n < rows:
                    s = Float32(0.0)
                    row = s_xs + n * XS_STRIDE
                    for b in cutlass.range_constexpr(8):
                        u = ptx.ld_shared_v4(row + (lane + 32 * b) * 16)
                        for e in cutlass.range_constexpr(4):
                            lo, hi = ptx.bf16x2_to_f32x2(u[e])
                            s = ptx.add_rn(s, ptx.mul_rn(lo, lo))
                            s = ptx.add_rn(s, ptx.mul_rn(hi, hi))
                    for k in cutlass.range_constexpr(5):
                        s = ptx.add_rn(s, cute.arch.shuffle_sync_bfly(s, 1 << k))
                    if lane == 0:
                        ptx.st_shared_f32(s_ssq + n * 4, s)
        else:
            self.q_norm_sums_contiguous(s_ssq, qv, warp, lane, rows)
        ptx.sync_threads()
        wv = []
        for e in cutlass.range_constexpr(4):
            lo, hi = ptx.bf16x2_to_f32x2(qw[e])
            if cutlass.const_expr(self.num.norm_weight_plus_zero):
                lo = ptx.add_rn(lo, Float32(0.0))
                hi = ptx.add_rn(hi, Float32(0.0))
            wv.append(lo)
            wv.append(hi)
        for n in cutlass.range_constexpr(max_rows):
            if n < rows:
                if cutlass.const_expr(strided):
                    s = ptx.ld_shared_f32(s_ssq + n * 4)
                else:
                    s = Float32(0.0)
                    for w in cutlass.range_constexpr(8):
                        s = ptx.add_rn(
                            s, ptx.ld_shared_f32(s_ssq + (w * max_rows + n) * 4)
                        )
                rr = ptx.rsqrt_approx(ptx.fma_rn(s, Float32(1.0 / Q_LORA), q_eps))
                raw = ptx.ld_shared_v4(s_xs + n * XS_STRIDE + tid * 16)
                words = []
                for e in cutlass.range_constexpr(4):
                    x0, x1 = ptx.bf16x2_to_f32x2(raw[e])
                    words.append(
                        ptx.cvt_rn_bf16x2_f32(
                            ptx.mul_rn(ptx.mul_rn(x0, rr), wv[2 * e]),
                            ptx.mul_rn(ptx.mul_rn(x1, rr), wv[2 * e + 1]),
                        )
                    )
                packed = (words[0], words[1], words[2], words[3])
                ptx.st_shared_v4(s_xs + n * XS_STRIDE + tid * 16, packed)
                if (qc_out != Int64(0)) & (h == 0) & (r == 0):
                    ptx.st_global_v4(
                        qc_out + (Int64(n) * qc_out_stride + Int64(tid * 8)) * 2, packed
                    )
        ptx.sync_threads()

        self.wait_weights(bar)

        g = lane // 4
        t = lane % 4
        zero = Float32(0.0)
        z4 = (zero, zero, zero, zero)
        c = [z4, z4, z4, z4]  # [j * 2 + token tile]
        for i in cutlass.range_constexpr(8):
            s_ = warp * 8 + i
            b0 = self.lds16(s_xs + g * XS_STRIDE + (s_ * 32 + 8 * t) * 2, g < rows)
            b1 = self.lds16(
                s_xs + (g + 8) * XS_STRIDE + (s_ * 32 + 8 * t) * 2, g + 8 < rows
            )
            for j in cutlass.range_constexpr(2):
                blk = base + off["wqb"] + (j * (Q_LORA // 32) + s_) * 1024
                lo = ptx.ld_shared_v4(blk + lane * 16)
                hi = ptx.ld_shared_v4(blk + 512 + lane * 16)
                for tt in cutlass.range_constexpr(2):
                    bq = b0 if tt == 0 else b1
                    c1 = ptx.mma_m16n8k16_bf16(
                        (lo[0], hi[0], lo[1], hi[1]), (bq[0], bq[1]), c[j * 2 + tt]
                    )
                    c[j * 2 + tt] = ptx.mma_m16n8k16_bf16(
                        (lo[2], hi[2], lo[3], hi[3]), (bq[2], bq[3]), c1
                    )
        # the partials reuse the token rows: every warp must be done reading them
        self.rows_dead()
        for j in cutlass.range_constexpr(2):
            for tt in cutlass.range_constexpr(2):
                cc = c[j * 2 + tt]
                ptx.st_shared_v4(
                    s_red + (((warp * 2 + j) * 2 + tt) * 32 + lane) * 16,
                    (
                        ptx.f32_bits(cc[0]),
                        ptx.f32_bits(cc[1]),
                        ptx.f32_bits(cc[2]),
                        ptx.f32_bits(cc[3]),
                    ),
                )
        self.sync_partials()
        for it in cutlass.range_constexpr(2):
            idx = tid + THREADS * it
            if idx < 32 * rows:
                rho = idx % 32
                n = idx // 32
                tt = n // 8
                nn = n % 8
                jt = rho // 16
                rr_ = rho % 16
                ln = (rr_ % 8) * 4 + nn // 2
                reg = (rr_ // 8) * 2 + nn % 2
                v = Float32(0.0)
                for w in cutlass.range_constexpr(8):
                    v = ptx.add_rn(
                        v,
                        ptx.ld_shared_f32(
                            s_red + (((w * 2 + jt) * 2 + tt) * 32 + ln) * 16 + reg * 4
                        ),
                    )
                qb = ptx.cvt_rn_bf16_f32(v)
                if r < NOPE // 32:
                    ptx.st_shared_b16(s_pub + (n * 32 + rho) * 2, qb)
                else:
                    d = (r - NOPE // 32) * 32 + rho
                    own = ptx.bf16_to_f32(qb)
                    other = cute.arch.shuffle_sync_bfly(own, 1)
                    o = Float32(0.0)
                    if d % 2 == 0:
                        o = self.q_rope_even(own, other, rope_c[it], rope_s[it])
                    else:
                        o = self.q_rope_odd(other, own, rope_c[it], rope_s[it])
                    o = ptx.mul_rn(o, inv_qs)
                    dst = (
                        mqa_q + Int64(n) * mqa_stride + Int64(h * MQA_DIM + KV_LORA + d)
                    )
                    ptx.st_global_u8(
                        dst,
                        cutlass.Uint16(
                            Uint32(ptx.cvt_rn_satfinite_e4m3x2_f32(o, Float32(0.0)))
                            & Uint32(0xFF)
                        ),
                    )
        self.publish()
        for it in cutlass.range_constexpr(2):
            self.gather(base, tid + THREADS * it, rows)
        self.done_reading()
        ptx.sync_threads()
        tile = warp % 4
        half = warp // 4
        cw = [z4, z4]
        for i in cutlass.range_constexpr(3):
            s_ = half * 3 + i
            b0 = self.lds16(s_qn + g * QN_STRIDE + (s_ * 32 + 8 * t) * 2, g < rows)
            b1 = self.lds16(
                s_qn + (g + 8) * QN_STRIDE + (s_ * 32 + 8 * t) * 2, g + 8 < rows
            )
            blk = base + off["wuk"] + (tile * (NOPE // 32) + s_) * 1024
            lo = ptx.ld_shared_v4(blk + lane * 16)
            hi = ptx.ld_shared_v4(blk + 512 + lane * 16)
            for tt in cutlass.range_constexpr(2):
                bq = b0 if tt == 0 else b1
                c1 = ptx.mma_m16n8k16_bf16(
                    (lo[0], hi[0], lo[1], hi[1]), (bq[0], bq[1]), cw[tt]
                )
                cw[tt] = ptx.mma_m16n8k16_bf16(
                    (lo[2], hi[2], lo[3], hi[3]), (bq[2], bq[3]), c1
                )
        for tt in cutlass.range_constexpr(2):
            ptx.st_shared_v4(
                s_red + ((warp * 2 + tt) * 32 + lane) * 16,
                (
                    ptx.f32_bits(cw[tt][0]),
                    ptx.f32_bits(cw[tt][1]),
                    ptx.f32_bits(cw[tt][2]),
                    ptx.f32_bits(cw[tt][3]),
                ),
            )
        self.sync_partials()
        for k in cutlass.range_constexpr((64 * max_rows + THREADS - 1) // THREADS):
            idx = tid + k * THREADS
            if idx < 64 * rows:
                rho = idx % 64
                n = idx // 64
                tt = n // 8
                nn = n % 8
                tl = rho // 16
                rr_ = rho % 16
                ln = (rr_ % 8) * 4 + nn // 2
                reg = (rr_ // 8) * 2 + nn % 2
                v = ptx.add_rn(
                    ptx.ld_shared_f32(s_red + ((tl * 2 + tt) * 32 + ln) * 16 + reg * 4),
                    ptx.ld_shared_f32(
                        s_red + (((tl + 4) * 2 + tt) * 32 + ln) * 16 + reg * 4
                    ),
                )
                qb = ptx.bf16_to_f32(ptx.cvt_rn_bf16_f32(v))
                dst = mqa_q + Int64(n) * mqa_stride + Int64(h * MQA_DIM + r * 64 + rho)
                ptx.st_global_u8(
                    dst,
                    cutlass.Uint16(
                        Uint32(
                            ptx.cvt_rn_satfinite_e4m3x2_f32(
                                ptx.mul_rn(qb, inv_qs), Float32(0.0)
                            )
                        )
                        & Uint32(0xFF)
                    ),
                )
        self.release_outputs()
        self.exit_wait()

    # ------------------------------------------------------------------ kv cluster
    @cute.jit
    def kv_cta(
        self,
        r: Int32,
        tid: Int32,
        kv_c: Int64,
        kv_c_stride: Int64,
        k_pe: Int64,
        k_pe_stride: Int64,
        idx_k: Int64,
        idx_k_stride: Int64,
        positions: Int64,
        slot_mapping: Int64,
        idx_slot_mapping: Int64,
        rows: Int32,
        has_indexer: Int32,
        kv_norm_w: Int64,
        kv_eps: Float32,
        rope_cache: Int64,
        rope_stride: Int64,
        k_scale: Int64,
        ik_w: Int64,
        ik_b: Int64,
        ik_eps: Float32,
        index_rope_cache: Int64,
        index_rope_stride: Int64,
        mla_cache: Int64,
        mla_block_stride: Int64,
        mla_entry_stride: Int64,
        mla_block_size: Int64,
        idx_cache: Int64,
        idx_block_stride: Int64,
        idx_block_size: Int64,
        topk: Int64,
        topk_stride: Int64,
        topk_width: Int32,
    ):
        lane = tid % 32
        warp = tid // 32
        if r == 0:
            self.mla_rows(
                lane,
                warp,
                kv_c,
                kv_c_stride,
                k_pe,
                k_pe_stride,
                positions,
                slot_mapping,
                rows,
                kv_norm_w,
                kv_eps,
                rope_cache,
                rope_stride,
                k_scale,
                mla_cache,
                mla_block_stride,
                mla_entry_stride,
                mla_block_size,
            )
        else:
            if has_indexer == 0:
                # shared layers: no work, but every path of the grid executes the PDL
                # wait
                ptx.pdl_wait()
            else:
                if r == 1:
                    self.index_k_rows(
                        lane,
                        warp,
                        idx_k,
                        idx_k_stride,
                        positions,
                        slot_mapping,
                        idx_slot_mapping,
                        rows,
                        ik_w,
                        ik_b,
                        ik_eps,
                        index_rope_cache,
                        index_rope_stride,
                        idx_cache,
                        idx_block_stride,
                        idx_block_size,
                    )
                else:
                    self.wait_inputs()
                    per_row = topk_width // 4
                    end = rows * per_row
                    neg = Uint32(0xFFFFFFFF)
                    for idx in cutlass.range(
                        (r - 2) * THREADS + tid, end, (CLUSTER - 2) * THREADS, unroll=1
                    ):
                        n = idx // per_row
                        cc = idx % per_row
                        ptx.st_global_v4(
                            topk + (Int64(n) * topk_stride + Int64(cc * 4)) * 4,
                            (neg, neg, neg, neg),
                        )

    @cute.jit
    def mla_rows(
        self,
        lane: Int32,
        warp: Int32,
        kv_c: Int64,
        kv_c_stride: Int64,
        k_pe: Int64,
        k_pe_stride: Int64,
        positions: Int64,
        slot_mapping: Int64,
        rows: Int32,
        kv_norm_w: Int64,
        kv_eps: Float32,
        rope_cache: Int64,
        rope_stride: Int64,
        k_scale: Int64,
        mla_cache: Int64,
        mla_block_stride: Int64,
        mla_entry_stride: Int64,
        mla_block_size: Int64,
    ):
        """The kv RMSNorm + BF16 + per-tensor FP8 of the latent and the k_pe RoPE: one
        MLA row per token (warp n = row n). Lane l owns elements 8 l + [0, 8) and
        256 + 8 l + [0, 8). Slot <= 0 is not written (slot 0 is the padding slot)."""
        strided = self.num.norm_order == "strided_8x256"
        # strided: lane l owns elements 8 l + [0, 8) and 256 + 8 l + [0, 8);
        # contiguous: 16 l + [0, 16)
        e0 = lane * 8 if strided else lane * 16
        e1 = KV_LORA // 2 + lane * 8 if strided else lane * 16 + 8
        w0 = ptx.ld_global_nc_v4(kv_norm_w + Int64(e0) * 2)
        w1 = ptx.ld_global_nc_v4(kv_norm_w + Int64(e1) * 2)
        ksc = ptx.ld_global_nc_f32(k_scale)
        self.wait_kv_inputs()
        for grp in cutlass.range_constexpr(self.max_rows // 8):
            self.mla_rows_row(
                warp + grp * 8,
                lane,
                kv_c,
                kv_c_stride,
                k_pe,
                k_pe_stride,
                positions,
                slot_mapping,
                rows,
                kv_eps,
                rope_cache,
                rope_stride,
                mla_cache,
                mla_block_stride,
                mla_entry_stride,
                mla_block_size,
                e0,
                e1,
                w0,
                w1,
                ksc,
            )

    @cute.jit
    def mla_rows_row(
        self,
        n: Int32,
        lane: Int32,
        kv_c: Int64,
        kv_c_stride: Int64,
        k_pe: Int64,
        k_pe_stride: Int64,
        positions: Int64,
        slot_mapping: Int64,
        rows: Int32,
        kv_eps: Float32,
        rope_cache: Int64,
        rope_stride: Int64,
        mla_cache: Int64,
        mla_block_stride: Int64,
        mla_entry_stride: Int64,
        mla_block_size: Int64,
        e0: Int32,
        e1: Int32,
        w0,
        w1,
        ksc: Float32,
    ):
        strided = self.num.norm_order == "strided_8x256"
        if n < rows:
            slot = ptx.ld_global_s64(slot_mapping + Int64(n) * 8)
            row = kv_c + Int64(n) * kv_c_stride * 2
            a = ptx.ld_global_v4(row + Int64(e0) * 2)
            b = ptx.ld_global_v4(row + Int64(e1) * 2)
            pos = ptx.ld_global_s64(positions + Int64(n) * 8)
            kp = ptx.ld_global_u32(
                k_pe + (Int64(n) * k_pe_stride + Int64(2 * lane)) * 2
            )
            c = self.rope_value(
                rope_cache, rope_stride, pos, lane, self.rope_cache_fp32
            )
            sn = self.rope_value(
                rope_cache, rope_stride, pos, 32 + lane, self.rope_cache_fp32
            )
            # negative slots are never written, slot 0 (padding) not under skip_slot0
            if slot > (Int64(0) if self.num.skip_slot0 else Int64(-1)):
                xs = (a[0], a[1], a[2], a[3], b[0], b[1], b[2], b[3])
                ws = (w0[0], w0[1], w0[2], w0[3], w1[0], w1[1], w1[2], w1[3])
                x = []
                for e in cutlass.range_constexpr(8):
                    lo, hi = ptx.bf16x2_to_f32x2(xs[e])
                    x.append(lo)
                    x.append(hi)
                s = Float32(0.0)
                for i in cutlass.range_constexpr(16):
                    if cutlass.const_expr(strided):
                        s = ptx.add_rn(s, ptx.mul_rn(x[i], x[i]))
                    else:
                        s = ptx.fma_rn(x[i], x[i], s)
                for k in cutlass.range_constexpr(5):
                    o = (1 << k) if strided else (16 >> k)
                    s = ptx.add_rn(s, cute.arch.shuffle_sync_bfly(s, o))
                rr = ptx.rsqrt_approx(ptx.fma_rn(s, Float32(1.0 / KV_LORA), kv_eps))
                entry = (
                    mla_cache
                    + (slot // mla_block_size) * mla_block_stride
                    + (slot % mla_block_size) * mla_entry_stride
                )
                q = []
                for e in cutlass.range_constexpr(4):
                    word = Uint32(0)
                    for k in cutlass.range_constexpr(4):
                        i = e * 4 + k
                        wlo, whi = ptx.bf16x2_to_f32x2(ws[i // 2])
                        wi = whi if i % 2 == 1 else wlo
                        xr = ptx.mul_rn(x[i], rr)
                        if cutlass.const_expr(self.num.norm_weight_plus_zero):
                            wi = ptx.add_rn(wi, Float32(0.0))
                        y = ptx.mul_rn(xr, wi)
                        if cutlass.const_expr(self.num.kv_norm_bf16):
                            # the norm output materialized in BF16
                            y = ptx.bf16_to_f32(ptx.cvt_rn_bf16_f32(y))
                        byte = Uint32(
                            ptx.cvt_rn_satfinite_e4m3x2_f32(
                                ptx.div_full(y, ksc), Float32(0.0)
                            )
                        ) & Uint32(0xFF)
                        word = word | (byte << Uint32(8 * k))
                    q.append(word)
                if cutlass.const_expr(strided):
                    ptx.st_global_v2(entry + Int64(e0), q[0], q[1])
                    ptx.st_global_v2(entry + Int64(e1), q[2], q[3])
                else:
                    ptx.st_global_v4(entry + Int64(e0), (q[0], q[1], q[2], q[3]))
                x1, x2 = ptx.bf16x2_to_f32x2(kp)
                r1 = ptx.fma_rn(x1, c, ptx.neg_f32(ptx.mul_rn(x2, sn)))
                r2 = ptx.fma_rn(x2, c, ptx.mul_rn(x1, sn))
                b1 = Uint32(
                    ptx.cvt_rn_satfinite_e4m3x2_f32(ptx.div_full(r1, ksc), Float32(0.0))
                ) & Uint32(0xFF)
                b2 = Uint32(
                    ptx.cvt_rn_satfinite_e4m3x2_f32(ptx.div_full(r2, ksc), Float32(0.0))
                ) & Uint32(0xFF)
                ptx.st_global_b16(
                    entry + Int64(KV_LORA + 2 * lane),
                    cutlass.Uint16(b1 | (b2 << Uint32(8))),
                )

    @cute.jit
    def index_k_rows(
        self,
        lane: Int32,
        warp: Int32,
        idx_k: Int64,
        idx_k_stride: Int64,
        positions: Int64,
        slot_mapping: Int64,
        idx_slot_mapping: Int64,
        rows: Int32,
        ik_w: Int64,
        ik_b: Int64,
        ik_eps: Float32,
        rope_cache: Int64,
        rope_stride: Int64,
        idx_cache: Int64,
        idx_block_stride: Int64,
        idx_block_size: Int64,
    ):
        """Index-K: LayerNorm(128) + interleaved RoPE (dims < 64) + BF16 + FP8 with an
        FP32 scale, one row per token
        (warp n = row n, lane l = dims 4 l .. 4 l + 3). Rows whose MLA or index slot is
        negative are not written."""
        lw = ptx.ld_global_nc_v4(ik_w + Int64(lane) * 16)
        lb = ptx.ld_global_nc_v4(ik_b + Int64(lane) * 16)
        self.wait_inputs()
        for grp in cutlass.range_constexpr(self.max_rows // 8):
            self.index_k_rows_row(
                warp + grp * 8,
                lane,
                idx_k,
                idx_k_stride,
                positions,
                slot_mapping,
                idx_slot_mapping,
                rows,
                ik_eps,
                rope_cache,
                rope_stride,
                idx_cache,
                idx_block_stride,
                idx_block_size,
                lw,
                lb,
            )

    @cute.jit
    def index_k_rows_row(
        self,
        n: Int32,
        lane: Int32,
        idx_k: Int64,
        idx_k_stride: Int64,
        positions: Int64,
        slot_mapping: Int64,
        idx_slot_mapping: Int64,
        rows: Int32,
        ik_eps: Float32,
        rope_cache: Int64,
        rope_stride: Int64,
        idx_cache: Int64,
        idx_block_stride: Int64,
        idx_block_size: Int64,
        lw,
        lb,
    ):
        if n < rows:
            mla_slot = ptx.ld_global_s64(slot_mapping + Int64(n) * 8)
            slot = ptx.ld_global_s64(idx_slot_mapping + Int64(n) * 8)
            ipos = ptx.ld_global_s64(positions + Int64(n) * 8)
            kv = ptx.ld_global_v2(
                idx_k + (Int64(n) * idx_k_stride + Int64(4 * lane)) * 2
            )
            x0, x1 = ptx.bf16x2_to_f32x2(kv[0])
            x2, x3 = ptx.bf16x2_to_f32x2(kv[1])
            x = (x0, x1, x2, x3)
            num = self.num
            if cutlass.const_expr(num.index_mean_sum == "sequential"):
                sm = ptx.add_rn(
                    ptx.add_rn(ptx.add_rn(ptx.add_rn(Float32(0.0), x0), x1), x2), x3
                )
            else:
                sm = ptx.add_rn(ptx.add_rn(x0, x1), ptx.add_rn(x2, x3))
            for k in cutlass.range_constexpr(5):
                sm = ptx.add_rn(sm, cute.arch.shuffle_sync_bfly(sm, 16 >> k))
            d = []
            vs = Float32(0.0)
            for e in cutlass.range_constexpr(4):
                # x - mean with the mean's multiply folded in (one rounding)
                d.append(ptx.fma_rn(sm, Float32(-1.0 / INDEX_DIM), x[e]))
                vs = ptx.fma_rn(d[e], d[e], vs)
            for k in cutlass.range_constexpr(5):
                vs = ptx.add_rn(vs, cute.arch.shuffle_sync_bfly(vs, 16 >> k))
            rstd = ptx.rsqrt_approx(ptx.fma_rn(vs, Float32(1.0 / INDEX_DIM), ik_eps))
            nm = []
            for e in cutlass.range_constexpr(4):
                nm.append(
                    ptx.fma_rn(
                        ptx.mul_rn(d[e], rstd), ptx.bits_f32(lw[e]), ptx.bits_f32(lb[e])
                    )
                )
            res = self.index_rope(
                lane < 16,
                lane,
                nm[0],
                nm[1],
                nm[2],
                nm[3],
                rope_cache,
                rope_stride,
                ipos,
            )
            rb = []
            for e in cutlass.range_constexpr(4):
                if cutlass.const_expr(num.index_k_bf16):
                    rb.append(ptx.bf16_to_f32(ptx.cvt_rn_bf16_f32(res[e])))
                else:
                    rb.append(res[e])
            am = ptx.max_f32(
                ptx.max_f32(ptx.abs_f32(rb[0]), ptx.abs_f32(rb[1])),
                ptx.max_f32(ptx.abs_f32(rb[2]), ptx.abs_f32(rb[3])),
            )
            for k in cutlass.range_constexpr(5):
                am = ptx.max_f32(am, cute.arch.shuffle_sync_bfly(am, 16 >> k))
            sc = ptx.div_rn(ptx.max_f32(Float32(1e-4), am), Float32(448.0))
            inv_sc = Float32(1.0)
            if cutlass.const_expr(num.index_scale == "fp32_amax"):
                inv_sc = ptx.div_rn(Float32(1.0), sc)
            else:
                # power-of-two scale 2^ceil(log2(s)), values divided by it
                sc = cute.math.exp2(cute.math.ceil(cute.math.log2(sc)))
            if (mla_slot >= Int64(0)) & (slot >= Int64(0)):
                blk = slot // idx_block_size
                ofs = slot % idx_block_size
                row_base = idx_cache + blk * idx_block_stride
                word = Uint32(0)
                for e in cutlass.range_constexpr(4):
                    if cutlass.const_expr(num.index_scale == "fp32_amax"):
                        # x (1 / s), clipped to +-448
                        v = ptx.max_f32(
                            ptx.min_f32(ptx.mul_rn(rb[e], inv_sc), Float32(448.0)),
                            Float32(-448.0),
                        )
                    else:
                        v = ptx.div_rn(rb[e], sc)
                    byte = Uint32(
                        ptx.cvt_rn_satfinite_e4m3x2_f32(v, Float32(0.0))
                    ) & Uint32(0xFF)
                    word = word | (byte << Uint32(8 * e))
                ptx.st_global_u32(row_base + ofs * INDEX_DIM + Int64(4 * lane), word)
                if lane == 0:
                    at = row_base + idx_block_size * INDEX_DIM + ofs * 4
                    if cutlass.const_expr(self.index_scale_ue8m0x4):
                        # the power-of-two scale as four equal UE8M0 bytes
                        e8 = (ptx.f32_bits(sc) >> Uint32(23)) & Uint32(0xFF)
                        ptx.st_global_u32(at, e8 * Uint32(0x01010101))
                    else:
                        ptx.st_global_f32(at, sc)

    # ------------------------------------------------------------------ steps
    @cute.jit
    def q_norm_sums_strided(
        self,
        s_xs: Int32,
        s_ssq: Int32,
        qv,
        warp: Int32,
        lane: Int32,
        tid: Int32,
        rows: Int32,
    ):
        """q_a sums of squares, strided order: the raw rows go to the XS slots; warp n
        sums row n, lane l adding the rounded squares of elements 8 l + 256 b + v
        (b, v = 0..7) to 0 in (b, v) order, then a butterfly xor 1, 2, 4, 8, 16."""
        for n in cutlass.range_constexpr(self.max_rows):
            if n < rows:
                ptx.st_shared_v4(s_xs + n * XS_STRIDE + tid * 16, qv[n])
        ptx.sync_threads()
        if warp < rows:
            s = Float32(0.0)
            row = s_xs + warp * XS_STRIDE
            for b in cutlass.range_constexpr(8):
                u = ptx.ld_shared_v4(row + (lane + 32 * b) * 16)
                for e in cutlass.range_constexpr(4):
                    lo, hi = ptx.bf16x2_to_f32x2(u[e])
                    s = ptx.add_rn(s, ptx.mul_rn(lo, lo))
                    s = ptx.add_rn(s, ptx.mul_rn(hi, hi))
            for k in cutlass.range_constexpr(5):
                s = ptx.add_rn(s, cute.arch.shuffle_sync_bfly(s, 1 << k))
            if lane == 0:
                ptx.st_shared_f32(s_ssq + warp * 4, s)

    @cute.jit
    def q_norm_sums_contiguous(
        self, s_ssq: Int32, qv, warp: Int32, lane: Int32, rows: Int32
    ):
        """q_a sums of squares, contiguous order: each thread fuses the squares of its 8
        elements into 0 in order, a butterfly xor 16 .. 1 sums the warp, and the 8 warp
        sums of row n are added in warp order when the row is normalized."""
        for n in cutlass.range_constexpr(self.max_rows):
            if n < rows:
                s = Float32(0.0)
                for e in cutlass.range_constexpr(4):
                    lo, hi = ptx.bf16x2_to_f32x2(qv[n][e])
                    s = ptx.fma_rn(lo, lo, s)
                    s = ptx.fma_rn(hi, hi, s)
                for k in cutlass.range_constexpr(5):
                    s = ptx.add_rn(s, cute.arch.shuffle_sync_bfly(s, 16 >> k))
                if lane == 0:
                    ptx.st_shared_f32(s_ssq + (warp * self.max_rows + n) * 4, s)

    @cute.jit
    def load16(self, addr: Int64, valid):
        w0 = Uint32(0)
        w1 = Uint32(0)
        w2 = Uint32(0)
        w3 = Uint32(0)
        if valid:
            w0, w1, w2, w3 = ptx.ld_global_v4(addr)
        return w0, w1, w2, w3

    @cute.jit
    def lds16(self, addr: Int32, valid):
        w0 = Uint32(0)
        w1 = Uint32(0)
        w2 = Uint32(0)
        w3 = Uint32(0)
        if valid:
            w0, w1, w2, w3 = ptx.ld_shared_v4(addr)
        return w0, w1, w2, w3

    @cute.jit
    def rope_value(
        self,
        cache: Int64,
        stride: Int64,
        pos: Int64,
        col: Int32,
        fp32: cutlass.Constexpr[bool],
    ):
        idx = pos * stride + Int64(col)
        if cutlass.const_expr(fp32):
            v = ptx.ld_global_f32(cache + idx * 4)
        else:
            v = ptx.bf16_to_f32(ptx.ld_global_u16(cache + idx * 2))
        return v

    @cute.jit
    def index_rope(
        self,
        valid,
        lane: Int32,
        n0: Float32,
        n1: Float32,
        n2: Float32,
        n3: Float32,
        cache: Int64,
        stride: Int64,
        pos: Int64,
    ):
        """Interleaved RoPE of this lane's dims 4 l .. 4 l + 3 when they are < 64 (pairs
        (4 l, 4 l + 1), (4 l + 2, 4 l + 3); cos / sin index = dim / 2). Even: e0 cos -
        e1 sin with e1 sin rounded. Odd: e0 sin + e1 cos with e1 cos rounded
        (``round_x1_cos``) or e1 cos + e0 sin with e0 sin rounded (``round_x0_sin``)."""
        o0 = n0
        o1 = n1
        o2 = n2
        o3 = n3
        if valid:
            c0 = self.rope_value(
                cache, stride, pos, 2 * lane, self.index_rope_cache_fp32
            )
            s0 = self.rope_value(
                cache, stride, pos, 32 + 2 * lane, self.index_rope_cache_fp32
            )
            c1 = self.rope_value(
                cache, stride, pos, 2 * lane + 1, self.index_rope_cache_fp32
            )
            s1 = self.rope_value(
                cache, stride, pos, 33 + 2 * lane, self.index_rope_cache_fp32
            )
            o0 = ptx.fma_rn(n0, c0, ptx.neg_f32(ptx.mul_rn(n1, s0)))
            o2 = ptx.fma_rn(n2, c1, ptx.neg_f32(ptx.mul_rn(n3, s1)))
            if cutlass.const_expr(self.num.index_rope_odd == "round_x1_cos"):
                o1 = ptx.fma_rn(n0, s0, ptx.mul_rn(n1, c0))
                o3 = ptx.fma_rn(n2, s1, ptx.mul_rn(n3, c1))
            else:
                o1 = ptx.fma_rn(n1, c0, ptx.mul_rn(n0, s0))
                o3 = ptx.fma_rn(n3, c1, ptx.mul_rn(n2, s1))
        return o0, o1, o2, o3

    @cute.jit
    def q_rope_even(self, x0: Float32, x1: Float32, c: Float32, s: Float32) -> Float32:
        """q_pe RoPE, even dim: x0 cos - x1 sin (one product rounded: x1 sin rounded,
        the rest fused; else both products rounded, then the difference)."""
        o = Float32(0.0)
        if cutlass.const_expr(self.num.q_rope == "one_product_rounded"):
            o = self.rope_even(x0, x1, c, s)
        else:
            o = ptx.sub_rn(ptx.mul_rn(x0, c), ptx.mul_rn(x1, s))
        return o

    @cute.jit
    def q_rope_odd(self, x0: Float32, x1: Float32, c: Float32, s: Float32) -> Float32:
        """q_pe RoPE, odd dim: x1 cos + x0 sin (one product rounded: x0 sin rounded, the
        rest fused; else both products rounded, then the sum)."""
        o = Float32(0.0)
        if cutlass.const_expr(self.num.q_rope == "one_product_rounded"):
            o = self.rope_odd(x0, x1, c, s)
        else:
            o = ptx.add_rn(ptx.mul_rn(x1, c), ptx.mul_rn(x0, s))
        return o

    @cute.jit
    def rope_even(self, x0: Float32, x1: Float32, c: Float32, s: Float32) -> Float32:
        """x0 cos - x1 sin, the product x1 sin rounded, the other fused."""
        return ptx.fma_rn(x0, c, ptx.neg_f32(ptx.mul_rn(x1, s)))

    @cute.jit
    def rope_odd(self, x0: Float32, x1: Float32, c: Float32, s: Float32) -> Float32:
        """x1 cos + x0 sin, the product x0 sin rounded, the other fused."""
        return ptx.fma_rn(x1, c, ptx.mul_rn(x0, s))

    @cute.jit
    def gather(self, base: Int32, tid: Int32, rows: Int32):
        """The head's 192 nope values per row from ranks 0..5 (distributed shared
        memory) into QN."""
        if tid < (NOPE // 32) * rows * 4:
            src = tid // (rows * 4)
            rem = tid % (rows * 4)
            n = rem // 4
            ch = rem % 4
            raddr = ptx.mapa_shared_cluster(
                base + self.off["pub"] + n * 64 + ch * 16, src
            )
            v = ptx.ld_shared_cluster_v4(raddr)
            ptx.st_shared_v4(
                base + self.off["qn"] + n * QN_STRIDE + src * 64 + ch * 16, v
            )

    @cute.jit
    def rope_pair(
        self,
        valid,
        positions: Int64,
        token: Int32,
        pair: Int32,
        cache: Int64,
        stride: Int64,
    ):
        """The cos / sin of one (token, pair) q_pe RoPE slot, zero when unused."""
        c = Float32(0.0)
        s = Float32(0.0)
        if valid:
            pos = ptx.ld_global_s64(positions + Int64(token) * 8)
            c = self.rope_value(cache, stride, pos, pair, self.rope_cache_fp32)
            s = self.rope_value(cache, stride, pos, 32 + pair, self.rope_cache_fp32)
        return c, s

    @cute.jit
    def rows_dead(self):
        """Every warp's reads of the token rows -> the partials that reuse them."""
        ptx.sync_threads()

    @cute.jit
    def handoff(self, base: Int32, tid: Int32, rows: Int32):
        self.publish()
        self.gather(base, tid, rows)

    @cute.jit
    def wait_inputs(self):
        ptx.pdl_wait()
        ptx.fence_proxy_async_global()

    @cute.jit
    def wait_q_inputs(self):
        ptx.pdl_wait()
        ptx.fence_proxy_async_global()

    @cute.jit
    def wait_kv_inputs(self):
        ptx.pdl_wait()
        ptx.fence_proxy_async_global()

    @cute.jit
    def wait_weights(self, bar: Int32):
        ptx.mbarrier_wait_parity(bar, 0)

    @cute.jit
    def sync_partials(self):
        ptx.sync_threads()

    @cute.jit
    def publish(self):
        ptx.cluster_arrive_release()
        ptx.cluster_wait_acquire()

    @cute.jit
    def done_reading(self):
        ptx.cluster_arrive_release()

    @cute.jit
    def release_outputs(self):
        ptx.fence_proxy_async_global()

    @cute.jit
    def exit_wait(self):
        ptx.cluster_wait_acquire()
