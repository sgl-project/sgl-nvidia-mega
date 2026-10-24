# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Block output of the routed experts (+ the shared expert) from per-slot partials.

For row n and output column r, with p_q = partials[q][n][k][r] (q = 0..3) the four
quarter partials of the expert in top-k slot k, g_k its FC2 output scale and w_k its
BF16 routing weight (as FP32):

    routed = fma(w_k, bf16(g_k * (((p_0 + p_1) + p_2) + p_3)), routed)
             for k = 0..7 with ids[k] >= 0
    shared = (((s_0 + 0) + s_1) + s_2) + s_3,   s_j = shared_partials[j][n][r]
    out    = bf16(bf16(routed) + bf16(shared))   (shared on)
             bf16(routed)                        (shared off)

which is the stock routed finalize (every expert output rounded to BF16, combined in
slot order with BF16 weights) followed by the stock routed + shared addition. One
256-thread block covers 256 consecutive outputs of one row; all loads are issued
before the sums.

Scratch layout (FP32 unless noted), rows M <= 16:
    partials        [4][M][8][6144]
    shared_partials [4][M][6144]
    ids             [M][8] int32 (-1: no expert in this slot)
    weights, scales [M][8]
    out             [M][6144] bf16"""

from __future__ import annotations

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
from cutlass import Float32, Int32, Int64
from cutlass.memory import SmemAllocator

from ..primitives.ptx_core import cvt_rn_bf16_f32, fma_rn, st_global_b16
from ._core_ptx import add_f32, ld_cg_f32, ld_cg_s32, mul_f32, round_bf16
from .constants import HIDDEN, QUARTERS, TOP_K

THREADS = 256
BLOCKS_PER_ROW = HIDDEN // THREADS


class MoeFinalize:
    """``shared``: add the shared expert's partials (the served form); off is
    test-only."""

    def __init__(self, shared: bool = True):
        self.shared = shared

    @cute.jit
    def __call__(
        self,
        partials: Int64,
        shared_partials: Int64,
        ids: Int64,
        weights: Int64,
        scales: Int64,
        out: Int64,
        rows: Int32,
        stream: cuda.CUstream,
    ):
        self.kernel(partials, shared_partials, ids, weights, scales, out, rows).launch(
            grid=[rows * BLOCKS_PER_ROW, 1, 1],
            block=[THREADS, 1, 1],
            max_number_threads=[THREADS, 1, 1],
            stream=stream,
            use_pdl=True,
            hint_smem_base_uniform=False,
        )

    @cute.kernel
    def kernel(
        self,
        partials: Int64,
        shared_partials: Int64,
        ids: Int64,
        weights: Int64,
        scales: Int64,
        out: Int64,
        rows: Int32,
    ):
        tid, _, _ = cute.arch.thread_idx()
        block, _, _ = cute.arch.block_idx()
        smem = SmemAllocator()
        slot_ids = smem.allocate_tensor(Int32, cute.make_layout(TOP_K), 4)
        slot_w = smem.allocate_tensor(Float32, cute.make_layout(TOP_K), 4)
        slot_g = smem.allocate_tensor(Float32, cute.make_layout(TOP_K), 4)

        cute.arch.griddepcontrol_wait()
        idx = block * THREADS + tid
        n = (block * THREADS) // HIDDEN
        r = idx - n * HIDDEN
        live = idx < rows * HIDDEN

        part = cute.make_rmem_tensor(cute.make_layout(QUARTERS * TOP_K), Float32)
        part.fill(0.0)
        shared_sum = Float32(0.0)
        if live:
            column = partials + Int64(r) * 4
            for q in cutlass.range_constexpr(QUARTERS):
                for k in cutlass.range_constexpr(TOP_K):
                    row = (Int64(q) * Int64(rows) + Int64(n)) * TOP_K + k
                    part[q * TOP_K + k] = ld_cg_f32(column + row * (HIDDEN * 4))
            if cutlass.const_expr(self.shared):
                element = shared_partials + Int64(idx) * 4
                stride = Int64(rows) * (HIDDEN * 4)
                s0 = ld_cg_f32(element)
                s1 = ld_cg_f32(element + stride)
                s2 = ld_cg_f32(element + 2 * stride)
                s3 = ld_cg_f32(element + 3 * stride)
                shared_sum = add_f32(
                    add_f32(add_f32(add_f32(s0, Float32(0.0)), s1), s2), s3
                )
        if (tid < TOP_K) & (n < rows):
            offset = Int64(n * TOP_K + tid) * 4
            slot_ids[tid] = ld_cg_s32(ids + offset)
            slot_w[tid] = ld_cg_f32(weights + offset)
            slot_g[tid] = ld_cg_f32(scales + offset)
        cute.arch.sync_threads()
        if live:
            routed = Float32(0.0)
            for k in cutlass.range_constexpr(TOP_K):
                if slot_ids[k] >= 0:
                    acc = add_f32(
                        add_f32(add_f32(part[k], part[TOP_K + k]), part[2 * TOP_K + k]),
                        part[3 * TOP_K + k],
                    )
                    expert = round_bf16(mul_f32(acc, slot_g[k]))
                    routed = fma_rn(slot_w[k], expert, routed)
            if cutlass.const_expr(self.shared):
                routed = add_f32(round_bf16(routed), round_bf16(shared_sum))
            st_global_b16(out + Int64(idx) * 2, cvt_rn_bf16_f32(routed))
