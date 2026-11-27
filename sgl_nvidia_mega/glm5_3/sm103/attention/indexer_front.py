# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""The indexer's share of the MLA prologue (P_front), as its own launch: on a layer with
its own indexer, what the indexer query projection and the indexer logits read, so the
rest of the prologue (q_b, W_UK absorption, q / k RoPE, the MLA rows: ``MlaPrologue``
with ``has_indexer`` 0 and no ``qc_out``) can run beside the indexer.

For each row ``m < rows``:

    qc_out[m] = bf16(RMSNorm(q_c[m]) * q_norm_w)          (the q_a norm, 2048 values)
    index-K row of idx_slot_mapping[m] = e4m3(RoPE(LayerNorm(idx_k[m])) / s), s stored
    with the row (rows whose MLA or index slot is negative are not written)

with the arithmetic of ``MlaPrologue`` (``PrologueNumerics``), so the outputs equal the
prologue's bit for bit: the q_a sum of squares in the ``strided_8x256`` order (lane l adds
the rounded squares of elements 8 (l + 32 b) + v to 0 in (b, v) order, then a butterfly
xor 1 .. 16; every lane ends with the warp's sum, as lane 0 hands it on in the
prologue), ``rsqrt.approx`` of ``sum / 2048 + eps`` fused, then ``bf16((x rr) w)``; the
index-K row is ``MlaPrologue.index_k_rows_row`` itself (this class traces the
prologue's own method).

Work split: one CTA per row (``max_rows`` CTAs, rows >= ``rows`` exit after the PDL
wait), 64 threads: warp 0 the q_a row, warp 1 the index-K row.

Ordering: dependents are triggered at entry; only the norm weights / LayerNorm
parameters are read before ``griddepcontrol.wait``, which every CTA executes;
``fence.proxy.async.global`` follows the wait and each thread's last output store (as in
``MlaPrologue``).
"""

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
from cutlass import Float32, Int32, Int64

from ..primitives import ptx_core as ptx
from .mla_prologue import MlaPrologue, PrologueNumerics

Q_LORA = 2048
THREADS = 64
GROUPS = Q_LORA // 8 // 32  # 16-byte groups of a q_a row per lane: 8


class IndexerFront:
    """See the module docstring. Call arguments (device addresses; row strides in
    elements; cache strides in bytes): ``q_c`` (the fused q_a / kv_a output, BF16) and
    its row stride, ``idx_k`` (the indexer key projection, BF16) and its row stride,
    ``positions``, ``slot_mapping``, ``idx_slot_mapping`` (int64), ``rows``,
    ``q_norm_w`` (BF16 [2048]), ``q_eps``, ``ik_w`` / ``ik_b`` (FP32 [128]), ``ik_eps``,
    ``index_rope_cache`` and its row stride, the index-K cache (base, block stride, block
    size), ``qc_out`` and its row stride, then the stream."""

    def __init__(
        self,
        max_rows: int,
        numerics: PrologueNumerics,
        index_rope_cache_fp32: bool = True,
        index_scale_ue8m0x4: bool = False,
    ):
        if max_rows not in (8, 16):
            raise ValueError(f"max_rows must be 8 or 16, got {max_rows}")
        if numerics.norm_order != "strided_8x256":
            raise ValueError(
                f"IndexerFront implements the strided_8x256 q_a sum order only, got "
                f"{numerics.norm_order!r}"
            )
        self.max_rows = max_rows
        self.num = numerics
        # the prologue whose index-K arithmetic this launch traces (its RoPE cache dtype
        # flag for the q path is unused here)
        self.pro = MlaPrologue(
            max_rows, numerics, True, index_rope_cache_fp32, index_scale_ue8m0x4
        )

    @cute.jit
    def __call__(
        self,
        q_c: Int64,
        q_c_stride: Int64,
        idx_k: Int64,
        idx_k_stride: Int64,
        positions: Int64,
        slot_mapping: Int64,
        idx_slot_mapping: Int64,
        rows: Int32,
        q_norm_w: Int64,
        q_eps: Float32,
        ik_w: Int64,
        ik_b: Int64,
        ik_eps: Float32,
        index_rope_cache: Int64,
        index_rope_stride: Int64,
        idx_cache: Int64,
        idx_block_stride: Int64,
        idx_block_size: Int64,
        qc_out: Int64,
        qc_out_stride: Int64,
        stream: cuda.CUstream,
    ):
        self.kernel(
            q_c,
            q_c_stride,
            idx_k,
            idx_k_stride,
            positions,
            slot_mapping,
            idx_slot_mapping,
            rows,
            q_norm_w,
            q_eps,
            ik_w,
            ik_b,
            ik_eps,
            index_rope_cache,
            index_rope_stride,
            idx_cache,
            idx_block_stride,
            idx_block_size,
            qc_out,
            qc_out_stride,
        ).launch(
            grid=[self.max_rows, 1, 1],
            block=[THREADS, 1, 1],
            stream=stream,
            use_pdl=True,
            max_number_threads=[THREADS, 1, 1],
        )

    @cute.kernel
    def kernel(
        self,
        q_c: Int64,
        q_c_stride: Int64,
        idx_k: Int64,
        idx_k_stride: Int64,
        positions: Int64,
        slot_mapping: Int64,
        idx_slot_mapping: Int64,
        rows: Int32,
        q_norm_w: Int64,
        q_eps: Float32,
        ik_w: Int64,
        ik_b: Int64,
        ik_eps: Float32,
        index_rope_cache: Int64,
        index_rope_stride: Int64,
        idx_cache: Int64,
        idx_block_stride: Int64,
        idx_block_size: Int64,
        qc_out: Int64,
        qc_out_stride: Int64,
    ):
        tid_, _, _ = cute.arch.thread_idx()
        cta_, _, _ = cute.arch.block_idx()
        tid = Int32(tid_)
        n = Int32(cta_)
        ptx.pdl_launch_dependents()
        lane = tid % 32
        warp = tid // 32
        if warp == 0:
            self.q_row(n, lane, q_c, q_c_stride, rows, q_norm_w, q_eps, qc_out, qc_out_stride)
        else:
            # MlaPrologue.index_k_rows for one row (its LayerNorm parameters before the
            # wait, then the row)
            lw = ptx.ld_global_nc_v4(ik_w + Int64(lane) * 16)
            lb = ptx.ld_global_nc_v4(ik_b + Int64(lane) * 16)
            self.pro.wait_inputs()
            self.pro.index_k_rows_row(
                n,
                lane,
                idx_k,
                idx_k_stride,
                positions,
                slot_mapping,
                idx_slot_mapping,
                rows,
                ik_eps,
                index_rope_cache,
                index_rope_stride,
                idx_cache,
                idx_block_stride,
                idx_block_size,
                lw,
                lb,
            )
        self.pro.release_outputs()

    @cute.jit
    def q_row(
        self,
        n: Int32,
        lane: Int32,
        q_c: Int64,
        q_c_stride: Int64,
        rows: Int32,
        q_norm_w: Int64,
        q_eps: Float32,
        qc_out: Int64,
        qc_out_stride: Int64,
    ):
        """Row n's q_a norm (``MlaPrologue.head_cta``'s arithmetic): lane l owns the
        16-byte groups l + 32 b, b = 0..7."""
        # the norm weights of this lane's groups, before the wait
        wv = []
        for b in cutlass.range_constexpr(GROUPS):
            qw = ptx.ld_global_nc_v4(q_norm_w + Int64(lane + 32 * b) * 16)
            for e in cutlass.range_constexpr(4):
                lo, hi = ptx.bf16x2_to_f32x2(qw[e])
                if cutlass.const_expr(self.num.norm_weight_plus_zero):
                    lo = ptx.add_rn(lo, Float32(0.0))
                    hi = ptx.add_rn(hi, Float32(0.0))
                wv.append(lo)
                wv.append(hi)
        self.pro.wait_q_inputs()
        valid = n < rows
        u = []
        for b in cutlass.range_constexpr(GROUPS):
            u.append(
                self.pro.load16(
                    q_c + (Int64(n) * q_c_stride + Int64((lane + 32 * b) * 8)) * 2, valid
                )
            )
        if valid:
            s = Float32(0.0)
            for b in cutlass.range_constexpr(GROUPS):
                for e in cutlass.range_constexpr(4):
                    lo, hi = ptx.bf16x2_to_f32x2(u[b][e])
                    s = ptx.add_rn(s, ptx.mul_rn(lo, lo))
                    s = ptx.add_rn(s, ptx.mul_rn(hi, hi))
            for k in cutlass.range_constexpr(5):
                s = ptx.add_rn(s, cute.arch.shuffle_sync_bfly(s, 1 << k))
            rr = ptx.rsqrt_approx(ptx.fma_rn(s, Float32(1.0 / Q_LORA), q_eps))
            for b in cutlass.range_constexpr(GROUPS):
                words = []
                for e in cutlass.range_constexpr(4):
                    x0, x1 = ptx.bf16x2_to_f32x2(u[b][e])
                    words.append(
                        ptx.cvt_rn_bf16x2_f32(
                            ptx.mul_rn(ptx.mul_rn(x0, rr), wv[8 * b + 2 * e]),
                            ptx.mul_rn(ptx.mul_rn(x1, rr), wv[8 * b + 2 * e + 1]),
                        )
                    )
                ptx.st_global_v4(
                    qc_out
                    + (Int64(n) * qc_out_stride + Int64((lane + 32 * b) * 8)) * 2,
                    (words[0], words[1], words[2], words[3]),
                )
