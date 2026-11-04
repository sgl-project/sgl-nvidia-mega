# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Indexer query projection with RoPE, power-of-two scaling and FP8 quantization (decode
/ verify rows).

For the 32 indexer heads (head dim 128) and each row ``m < rows``:

    q = bf16(q_c_normed[m] @ wq_b^T)                      [32, 128]   (FP32 accumulate,
    one BF16 rounding) q[:, :64] = interleaved RoPE of q[:, :64] at positions[m] amax_h
    = max(1e-4, max |q[h, :]|);   scale_h = 2^ceil(log2(amax_h / 448)) (exponent-bit
    rounding) q_fp8[m, h] = e4m3_satfinite(q[h] / scale_h);   w_out[m, h] =
    (index_weights[m, h] * scale_h) * w_scale

Work split: 32 clusters (one per head) of 4 CTAs (256 threads); CTA ``c`` of head ``h``
owns the 32 output dims ``[32 c, 32 c + 32)`` over the full K = 2048 (CTAs 0 and 1 hold
the RoPE dims). Each warp loads its 16 KiB of the CTA's packed weight slice with one
bulk copy before the PDL wait and sums its 256-wide K range with ``mma.sync`` m16n8k16
(rows on A); the 8 warp partials are added in warp order from 0.0. The per-row amax is
reduced within the CTA by shuffles and across the 4 CTAs through distributed shared
memory.

Option fields (compile time): ``max_rows`` (8 or 16), ``rope_form`` (the RoPE
arithmetic: ``"mul_sub"``: x0 c - x1 s and x1 c + x0 s with separately rounded products;
``"fma_x0_first"``: fma(x0, c, -(x1 s)) and fma(x1, c, x0 s); ``"fma_x1_first"``:
fma(-x1, s, x0 c) and fma(x0, s, x1 c)), ``rope_cache_fp32`` (the cos / sin cache is
FP32, else BF16; layout [positions, 64] = cos | sin).

Ordering: dependents are triggered at entry; only the constant weights are read before
``griddepcontrol.wait``; ``fence.proxy.async.global`` follows the wait and every
thread's last output store. The cluster amax exchange publishes with
``barrier.cluster.arrive`` (release) / ``wait`` (acquire); a second arrive after the
remote reads and the final wait keep every CTA's shared memory alive while its peers
read it.
"""

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
from cutlass import Float32, Int32, Int64, Uint32

from ..primitives import ptx_core as ptx

HEADS = 32
HEAD_DIM = 128
K = 2048
CLUSTER = 4
ROWS_PER_CTA = HEAD_DIM // CLUSTER  # 32 output dims per CTA
WARPS = 8
THREADS = WARPS * 32
STEPS_PER_WARP = K // 32 // WARPS  # 8 k-steps of 32
WARP_BYTES = STEPS_PER_WARP * ROWS_PER_CTA * 64  # 16 KiB
CTA_WEIGHT_BYTES = WARPS * WARP_BYTES  # 128 KiB
GRID = HEADS * CLUSTER
ROPE_FORMS = ("mul_sub", "fma_x0_first", "fma_x1_first")
AMAX_FLOOR = 1e-4
INV_448 = 1.0 / 448.0


class IndexerQProj:
    """See the module docstring. Call arguments (device addresses; strides in elements):
    ``weights`` (packed by ``packing.pack_indexer_q``), ``x`` ([rows, 2048] BF16) and
    its row stride, ``positions`` (int64 [rows]), ``rope_cache`` and its row stride,
    ``index_weights`` ([rows, 32] BF16) and its row stride, ``w_scale``, ``rows``,
    ``q_fp8`` ([rows, 32, 128] E4M3, head stride 128) and its row stride, ``w_out``
    ([rows, 32] FP32)
    and its row stride, then the stream."""

    def __init__(
        self, max_rows: int, rope_form: str = "mul_sub", rope_cache_fp32: bool = True
    ):
        if max_rows not in (8, 16):
            raise ValueError(f"max_rows must be 8 or 16, got {max_rows}")
        if rope_form not in ROPE_FORMS:
            raise ValueError(
                f"rope_form must be one of {ROPE_FORMS}, got {rope_form!r}"
            )
        self.max_rows = max_rows
        self.rope_form = rope_form
        self.rope_cache_fp32 = rope_cache_fp32

    @cute.jit
    def __call__(
        self,
        weights: Int64,
        x: Int64,
        x_stride: Int64,
        positions: Int64,
        rope_cache: Int64,
        rope_stride: Int64,
        index_weights: Int64,
        index_weights_stride: Int64,
        w_scale: Float32,
        rows: Int32,
        q_fp8: Int64,
        q_stride: Int64,
        w_out: Int64,
        w_out_stride: Int64,
        stream: cuda.CUstream,
    ):
        self.kernel(
            weights,
            x,
            x_stride,
            positions,
            rope_cache,
            rope_stride,
            index_weights,
            index_weights_stride,
            w_scale,
            rows,
            q_fp8,
            q_stride,
            w_out,
            w_out_stride,
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
        weights: Int64,
        x: Int64,
        x_stride: Int64,
        positions: Int64,
        rope_cache: Int64,
        rope_stride: Int64,
        index_weights: Int64,
        index_weights_stride: Int64,
        w_scale: Float32,
        rows: Int32,
        q_fp8: Int64,
        q_stride: Int64,
        w_out: Int64,
        w_out_stride: Int64,
    ):
        max_rows = self.max_rows
        tid_, _, _ = cute.arch.thread_idx()
        cta_, _, _ = cute.arch.block_idx()
        tid = Int32(tid_)
        cta = Int32(cta_)
        smem = cutlass.utils.SmemAllocator()
        wbuf = smem.allocate_tensor(
            cutlass.Uint8, cute.make_layout(CTA_WEIGHT_BYTES), byte_alignment=1024
        )
        red_buf = smem.allocate_tensor(
            Float32,
            cute.make_layout(WARPS * max_rows * ROWS_PER_CTA),
            byte_alignment=16,
        )
        amax_buf = smem.allocate_tensor(
            Float32, cute.make_layout(max_rows + 16), byte_alignment=16
        )
        bars = smem.allocate_array(Int64, WARPS)
        s_w = ptx.shared_address(wbuf.iterator)
        s_red = ptx.shared_address(red_buf.iterator)
        s_amax = ptx.shared_address(amax_buf.iterator)
        s_bar = ptx.shared_address(bars)

        ptx.pdl_launch_dependents()
        lane = tid % 32
        warp = tid // 32
        g = lane // 4
        t = lane % 4
        head = cta // CLUSTER
        part = cta % CLUSTER
        ring = s_w + warp * WARP_BYTES
        bar = s_bar + warp * 8
        # constant weights: this warp's 16 KiB, one bulk copy (evict-first L2 hint)
        if lane == 0:
            ptx.mbarrier_init(bar, 1)
            ptx.fence_mbarrier_init_release_cluster()
            ptx.mbarrier_arrive_expect_tx(bar, WARP_BYTES)
            src = weights + Int64(cta) * CTA_WEIGHT_BYTES + Int64(warp) * WARP_BYTES
            ptx.bulk_copy_g2s(ring, src, WARP_BYTES, bar, ptx.l2_policy("evict_first"))
        cute.arch.sync_warp()

        self.wait_inputs()

        # A fragments: rows g (and g + 8), latent [256 warp + 32 i + 8 t, +8)
        zero4 = (Uint32(0), Uint32(0), Uint32(0), Uint32(0))
        row_lo = (
            x + (Int64(g) * x_stride + Int64(warp * STEPS_PER_WARP * 32 + 8 * t)) * 2
        )
        row_hi = row_lo + Int64(8) * x_stride * 2
        lo = []
        hi = []
        for i in cutlass.range_constexpr(STEPS_PER_WARP):
            lo.append(self.load_row(row_lo + i * 64, g < rows))
            if cutlass.const_expr(max_rows == 16):
                hi.append(self.load_row(row_hi + i * 64, g + 8 < rows))
            else:
                hi.append(zero4)

        # epilogue mapping: (row em, dim pair ep)
        ep = lane % 16
        if cutlass.const_expr(max_rows == 16):
            em = 2 * warp + lane // 16
            active = em < rows
        else:
            em = warp
            active = (em < rows) & (lane < 16)
        rc = Float32(0.0)
        rs = Float32(0.0)
        wv = Float32(0.0)
        if active:
            pos = ptx.ld_global_s64(positions + Int64(em) * 8)
            if part < 2:
                rc = self.rope_value(rope_cache, rope_stride, pos, 16 * part + ep)
                rs = self.rope_value(rope_cache, rope_stride, pos, 32 + 16 * part + ep)
            if (part == 0) & (ep == 0):
                at = (
                    index_weights + (Int64(em) * index_weights_stride + Int64(head)) * 2
                )
                wv = ptx.bf16_to_f32(ptx.ld_global_u16(at))

        self.wait_weights(bar)

        zero = Float32(0.0)
        acc = [(zero, zero, zero, zero) for _ in range(4)]
        for i in cutlass.range_constexpr(STEPS_PER_WARP):
            a = lo[i]
            a8 = hi[i]
            blk = ring + i * (ROWS_PER_CTA * 64)
            for j in cutlass.range_constexpr(4):
                b = ptx.ld_shared_v4(blk + (8 * j + g) * 64 + t * 16)
                c = ptx.mma_m16n8k16_bf16(
                    (a[0], a8[0], a[1], a8[1]), (b[0], b[1]), acc[j]
                )
                acc[j] = ptx.mma_m16n8k16_bf16(
                    (a[2], a8[2], a[3], a8[3]), (b[2], b[3]), c
                )
        if g < rows:
            for j in cutlass.range_constexpr(4):
                at = s_red + ((warp * max_rows + g) * ROWS_PER_CTA + 8 * j + 2 * t) * 4
                ptx.st_shared_v2_f32(at, acc[j][0], acc[j][1])
        if cutlass.const_expr(max_rows == 16):  # noqa: SIM102
            if g + 8 < rows:
                for j in cutlass.range_constexpr(4):
                    at = (
                        s_red
                        + ((warp * max_rows + g + 8) * ROWS_PER_CTA + 8 * j + 2 * t) * 4
                    )
                    ptx.st_shared_v2_f32(at, acc[j][2], acc[j][3])
        self.sync_partials()

        # fixed-order sum, BF16 rounding, RoPE, local amax
        v0 = Float32(0.0)
        v1 = Float32(0.0)
        am = Float32(AMAX_FLOOR)
        if active:
            s0 = Float32(0.0)
            s1 = Float32(0.0)
            for w in cutlass.range_constexpr(WARPS):
                q0, q1 = ptx.ld_shared_v2_f32(
                    s_red + ((w * max_rows + em) * ROWS_PER_CTA + 2 * ep) * 4
                )
                s0 = ptx.add_rn(s0, q0)
                s1 = ptx.add_rn(s1, q1)
            v0 = ptx.bf16_to_f32(ptx.cvt_rn_bf16_f32(s0))
            v1 = ptx.bf16_to_f32(ptx.cvt_rn_bf16_f32(s1))
            if part < 2:
                o0, o1 = self.rope(v0, v1, rc, rs)
                v0 = o0
                v1 = o1
            am = ptx.max_f32(
                ptx.max_f32(ptx.abs_f32(v0), ptx.abs_f32(v1)), Float32(AMAX_FLOOR)
            )
        offsets = (16, 8, 4, 2, 1) if max_rows == 8 else (8, 4, 2, 1)
        for k in cutlass.range_constexpr(len(offsets)):
            am = ptx.max_f32(am, cute.arch.shuffle_sync_bfly(am, offsets[k]))
        if cutlass.const_expr(max_rows == 8):  # noqa: SIM108
            publisher = lane == 0
        else:
            publisher = lane % 16 == 0
        if publisher & (em < rows):
            ptx.st_shared_f32(s_amax + em * 4, am)

        self.publish_amax()
        amax_g = Float32(0.0)
        if active:
            for src in cutlass.range_constexpr(CLUSTER):
                v = ptx.ld_shared_cluster_f32(
                    ptx.mapa_shared_cluster(s_amax + em * 4, src)
                )
                if cutlass.const_expr(src == 0):
                    amax_g = ptx.max_f32(v, amax_g)
                else:
                    amax_g = ptx.max_f32(amax_g, v)
        self.done_reading()
        if active:
            raw = ptx.mul_rn(amax_g, Float32(INV_448))
            eb = (ptx.f32_bits(raw) + Uint32(0x007FFFFF)) & Uint32(0x7F800000)
            scale = ptx.bits_f32(eb)
            inv = ptx.bits_f32(Uint32(0x7F000000) - eb)
            b0 = Uint32(
                ptx.cvt_rn_satfinite_e4m3x2_f32(ptx.mul_rn(v0, inv), Float32(0.0))
            ) & Uint32(0xFF)
            b1 = Uint32(
                ptx.cvt_rn_satfinite_e4m3x2_f32(ptx.mul_rn(v1, inv), Float32(0.0))
            ) & Uint32(0xFF)
            dst = (
                q_fp8
                + Int64(em) * q_stride
                + Int64(head * HEAD_DIM + 32 * part + 2 * ep)
            )
            ptx.st_global_b16(dst, cutlass.Uint16(b0 | (b1 << Uint32(8))))
            if (part == 0) & (ep == 0):
                wo = ptx.mul_rn(ptx.mul_rn(wv, scale), w_scale)
                ptx.st_global_f32(
                    w_out + (Int64(em) * w_out_stride + Int64(head)) * 4, wo
                )
        self.release_outputs()
        self.exit_wait()

    # ------------------------------------------------------------------ steps
    @cute.jit
    def load_row(self, addr: Int64, valid):
        w0 = Uint32(0)
        w1 = Uint32(0)
        w2 = Uint32(0)
        w3 = Uint32(0)
        if valid:
            w0, w1, w2, w3 = ptx.ld_global_v4(addr)
        return w0, w1, w2, w3

    @cute.jit
    def rope_value(self, cache: Int64, stride: Int64, pos: Int64, col: Int32):
        idx = pos * stride + Int64(col)
        if cutlass.const_expr(self.rope_cache_fp32):
            v = ptx.ld_global_f32(cache + idx * 4)
        else:
            v = ptx.bf16_to_f32(ptx.ld_global_u16(cache + idx * 2))
        return v

    @cute.jit
    def rope(self, x0: Float32, x1: Float32, c: Float32, s: Float32):
        if cutlass.const_expr(self.rope_form == "mul_sub"):
            o0 = ptx.sub_rn(ptx.mul_rn(x0, c), ptx.mul_rn(x1, s))
            o1 = ptx.add_rn(ptx.mul_rn(x1, c), ptx.mul_rn(x0, s))
        elif cutlass.const_expr(self.rope_form == "fma_x0_first"):
            o0 = ptx.fma_rn(x0, c, ptx.neg_f32(ptx.mul_rn(x1, s)))
            o1 = ptx.fma_rn(x1, c, ptx.mul_rn(x0, s))
        else:
            o0 = ptx.fma_rn(ptx.neg_f32(x1), s, ptx.mul_rn(x0, c))
            o1 = ptx.fma_rn(x0, s, ptx.mul_rn(x1, c))
        return o0, o1

    @cute.jit
    def wait_inputs(self):
        ptx.pdl_wait()
        ptx.fence_proxy_async_global()

    @cute.jit
    def wait_weights(self, bar: Int32):
        ptx.mbarrier_wait_parity(bar, 0)

    @cute.jit
    def sync_partials(self):
        ptx.sync_threads()

    @cute.jit
    def publish_amax(self):
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
