# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Top-8 routing of GLM-5.3 (sigmoid scores + correction bias, one expert group).

Per token row: s_e = sigmoid(logit_e) (stock form), key_e = s_e + bias_e with NaN ranked
as -inf and -0 as +0, the 8 largest keys selected with ties going to the lower expert
id, and weights bf16(scale * s_e / (sum of the 8 selected s + 1e-20)). Every operation
and its order is the stock routing kernel's, so ids (in selection order) and weights are
bit-identical to it, except that a NaN score ranks last here (stock ranks it first).

One warp per row; lane l holds experts 8 l .. 8 l + 7."""

from __future__ import annotations

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
from cutlass import Float32, Int32, Int64, Uint32

from ..primitives.ptx_tc import (
    routing_add,
    routing_key,
    routing_sigmoid,
    routing_weight,
)
from ._core_ptx import ld_cg_v4_f32, ld_nc_v4_f32, st_global_b32, st_global_b32f
from .constants import MAX_ROWS, NUM_EXPERTS, TOP_K

PER_LANE = NUM_EXPERTS // 32


@cute.jit
def top8_rows(scores, keys, lane: Int32):
    """Select one row's top-8; the warp holds the row, ``scores`` / ``keys`` are this
    lane's 8 experts. Returns (the expert chosen in round ``lane``, its score, the sum
    of the 8 chosen scores); for lanes >= 8 the first two are (0, 0)."""
    taken = Uint32(0)
    chosen = Int32(0)
    for rnd in cutlass.range(TOP_K, unroll=1):
        best = Uint32(0)
        best_j = Int32(0)
        for j in cutlass.range_constexpr(PER_LANE - 1, -1, -1):
            if (((taken >> j) & Uint32(1)) == Uint32(0)) & (keys[j] >= best):
                best = keys[j]
                best_j = Int32(j)
        has = (taken & Uint32(0xFF)) != Uint32(0xFF)
        offer = Uint32(0)
        if has:
            offer = best
        top = cute.arch.warp_redux_sync(offer, "max")
        candidate = Uint32(0xFFFFFFFF)
        if has & (best == top):
            candidate = Uint32(lane * PER_LANE + best_j)
        winner = Int32(cute.arch.warp_redux_sync(candidate, "min"))
        if lane == (winner >> 3):
            taken = taken | (Uint32(1) << Uint32(winner & 7))
        if lane == rnd:
            chosen = winner
    source = (chosen >> 3) & 31
    mine = Float32(0.0)
    for j in cutlass.range_constexpr(PER_LANE):
        value = cute.arch.shuffle_sync(scores[j], source)
        if (chosen & 7) == j:
            mine = value
    if lane >= TOP_K:
        mine = Float32(0.0)
    total = mine
    for offset in (16, 8, 4, 2, 1):
        total = routing_add(total, cute.arch.shuffle_sync_bfly(total, offset))
    return chosen, mine, total


class MoeRouteTop8:
    """ids_out [M][8] int32 (selection order) and weights_out [M][8] fp32 (BF16 values)
    from fp32 logits [M][256] and an fp32 bias [256]; M <= 16. Launches its dependents
    at entry, then waits for its own inputs."""

    @cute.jit
    def __call__(
        self,
        logits: Int64,
        bias: Int64,
        scale: Float32,
        ids_out: Int64,
        weights_out: Int64,
        rows: Int32,
        stream: cuda.CUstream,
    ):
        self.kernel(logits, bias, scale, ids_out, weights_out, rows).launch(
            grid=[1, 1, 1],
            block=[32 * rows, 1, 1],
            max_number_threads=[32 * MAX_ROWS, 1, 1],
            stream=stream,
            use_pdl=True,
        )

    @cute.kernel
    def kernel(
        self,
        logits: Int64,
        bias: Int64,
        scale: Float32,
        ids_out: Int64,
        weights_out: Int64,
        rows: Int32,
    ):
        cute.arch.griddepcontrol_launch_dependents()
        cute.arch.griddepcontrol_wait()
        tid, _, _ = cute.arch.thread_idx()
        n = tid >> 5
        lane = tid & 31
        if n < rows:
            src = logits + Int64(n * NUM_EXPERTS + lane * PER_LANE) * 4
            l0 = ld_cg_v4_f32(src)
            l1 = ld_cg_v4_f32(src + 16)
            b0 = ld_nc_v4_f32(bias + Int64(lane * PER_LANE) * 4)
            b1 = ld_nc_v4_f32(bias + Int64(lane * PER_LANE) * 4 + 16)
            logit = (*l0, *l1)
            bias_v = (*b0, *b1)
            scores = [routing_sigmoid(logit[j]) for j in range(PER_LANE)]
            keys = [routing_key(scores[j], bias_v[j]) for j in range(PER_LANE)]
            chosen, mine, total = top8_rows(scores, keys, lane)
            if lane < TOP_K:
                slot = Int64(n * TOP_K + lane) * 4
                st_global_b32(ids_out + slot, Uint32(chosen))
                st_global_b32f(weights_out + slot, routing_weight(scale, mine, total))
