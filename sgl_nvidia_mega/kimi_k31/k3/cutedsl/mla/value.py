# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the SGLang project
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""MLA value projection (W_UV) and sigmoid gate of the BF16 attention latent."""

import cutlass
import cutlass.cute as cute
from cutlass import BFloat16, Float32, Int32, Uint32

from ..common.primitives import (
    alloc,
    asm,
    at,
    bf16x2,
    copy_g2s,
    div_rn,
    launch,
    load_words,
    mma_into,
    sigmoid,
    store_words,
    unpack_bf16,
    zeros,
)


class MlaValue:
    """W_UV and the sigmoid gate, [B, H, 512] BF16 latent -> [B, H*128]. One
    128-thread CTA per (32 outputs, head, 16 rows); warp w owns outputs
    8w..8w+7 over the whole latent (BF16 MMA, FP32 accumulation). W_UV must have a
    contiguous latent axis (the absorbed kv_b_proj view). Rows are tokens, ``q`` per
    request (``lengths`` per request).

    ``fence`` (default off): fence.acq_rel.gpu right before griddepcontrol.launch_dependents, so the trigger releases
    what this kernel's griddepcontrol.wait acquired to the dependents' pre-wait reads (KNOWN K-19 / K-19a: OprojFused
    prefetches the AttnRes prefix / blocks before its own wait; PTX 9.7.15.14). dsa/verify.py turns it on for the
    K31MK_FOLD_SPLIT_ROWS form under K31MK_MLAVALUE_FOLD_EARLY=f, where this kernel is the o-projection's PDL
    primary in place of the fenced MlaValueFold. Outputs are bitwise equal either way."""

    def __init__(self, heads: int, batch: int, q: int = 1, fence: bool = False):
        self.heads, self.batch, self.q = heads, batch, q
        self.fence = fence
        self.grid = (4, heads, (batch + 15) // 16)

    @cute.jit
    def __call__(self, *args):
        launch(self.kernel, args, self.grid, 128)

    @cute.kernel
    def kernel(self, args):
        latent, w_uv, gate, lengths, output = args
        rows = alloc(cutlass.utils.SmemAllocator(), BFloat16, (16, 544), 128)
        tid, _, _ = cute.arch.thread_idx()
        chunk, head, tile = cute.arch.block_idx()
        lane, warp = tid % 32, tid // 32
        group, quarter = lane // 4, lane % 4
        t0, v = tile * 16, chunk * 32 + warp * 8 + group
        # B fragments: 8 latent values per 32-wide chunk; A uses the same permutation.
        wb = [
            load_words(at(w_uv, head, 32 * c + 8 * quarter, v), count=4)
            for c in range(16)
        ]
        cute.arch.griddepcontrol_wait()
        for c in range(tid, 16 * 64, 128):
            r, j = c // 64, c % 64
            if t0 + r < self.batch:
                src = at(latent, t0 + r, head, j * 8)
                copy_g2s(rows.iterator + r * 544 + j * 8, src)
        cute.arch.cp_async_commit_group()
        col = head * 128 + chunk * 32 + warp * 8 + 2 * quarter
        gates = []
        for r in cutlass.range_constexpr(2):
            word = Uint32(0)
            if t0 + group + 8 * r < self.batch:
                word = load_words(
                    at(gate, t0 + group + 8 * r, col), count=1, mutable=True
                )[0]
            gates.append(unpack_bf16(word))
        if cutlass.const_expr(self.fence):
            cute.arch.fence_acq_rel_gpu()
        cute.arch.griddepcontrol_launch_dependents()
        cute.arch.cp_async_wait_group(0)
        cute.arch.sync_threads()
        parts = zeros((2, 4), Float32)  # two accumulation chains
        for c in cutlass.range_constexpr(16):
            x, y = (
                load_words(
                    rows.iterator + r * 544 + 32 * c + 8 * quarter, count=4, shared=True
                )
                for r in (group, group + 8)
            )
            for s in cutlass.range_constexpr(2):
                a = (x[2 * s], y[2 * s], x[2 * s + 1], y[2 * s + 1])
                b = (wb[c][2 * s], wb[c][2 * s + 1])
                mma_into(parts[s, None], a, b, bf16=True)
        acc = [parts[0, k] + parts[1, k] for k in range(4)]
        for r in cutlass.range_constexpr(2):
            t = t0 + group + 8 * r
            if t < self.batch:
                # Padded rows may hold a NaN gate; keep inactive rows zero.
                lo, hi = Float32(0), Float32(0)
                if lengths[t // self.q] > 0:
                    lo, hi = (
                        Float32(BFloat16(acc[2 * r + e]))
                        * Float32(BFloat16(sigmoid(gates[r][e])))
                        for e in range(2)
                    )
                store_words(at(output, t, col), (bf16x2(lo, hi),))


@asm(Float32, pure=True)
def _ex2(x):
    """exp2 of the split merge weights: the attention softmax's ex2.approx.ftz (the stock cores' class)."""
    return "ex2.approx.ftz.f32 $0,$1;", "=f,f", (x,)


class MlaValueFold(MlaValue):
    """MlaValue fed by the K3.1 TP16 sparse MLA6 core's split partials (kimi_k31/dsa/sparse_mla6.cuh): the split
    merge is folded into this kernel. One 128-thread CTA per (32 outputs, head, row); thread i first merges latent
    dims 4i..4i+3 of its (row, head) over the ``splits`` partials (FP32: M = max_c m_c, w_c = ex2(m_c - M),
    latent = bf16(sum_c w_c U_c / sum_c w_c l_c * out_scale)) into row 0 of the SMEM tile, then W_UV and the gate run
    exactly as MlaValue (rows 1-15 of the MMA tile are zero and never stored). Chunk-0 CTAs also write the merged
    latent to ``latent_out`` [B, H, 512] (the sparse core's latent view).

    ``early`` (K31MK_MLAVALUE_FOLD_EARLY, default off): release the PDL dependents (the o-projection) right after
    griddepcontrol.wait instead of after the merge, so their weight prologue overlaps the merge and W_UV. ``True``: the
    trigger alone (no release). ``"fence"``: wait -> fence.acq_rel.gpu -> trigger, so the trigger releases what the wait
    acquired to the dependents' pre-wait reads (OprojFused prefetches the AttnRes prefix / blocks before its own wait,
    KNOWN K-19; PTX 9.7.15.14: a release fence before launch_dependents synchronizes with the dependent's start).
    Dependents read this kernel's outputs only after their own griddepcontrol.wait; outputs are bitwise equal."""

    def __init__(self, heads: int, batch: int, q: int = 1, splits: int = 16, early: bool | str = False):
        super().__init__(heads, batch, q)
        self.splits = splits
        self.early = early
        # Trigger placement: "merge" (served), "wait" (early=True), "fence" (early="fence").
        self.trigger = {False: "merge", True: "wait"}.get(early, early)
        if self.trigger not in ("merge", "wait", "fence"):
            raise ValueError(f"MlaValueFold: unknown early mode {early!r}")
        self.grid = (4, heads, batch)

    @cute.jit
    def __call__(self, *args):
        launch(self.kernel, args, self.grid, 128)

    @cute.kernel
    def kernel(self, args):
        partial, stats, w_uv, gate, lengths, out_scale, output, latent_out = args
        rows = alloc(cutlass.utils.SmemAllocator(), BFloat16, (16, 544), 128)
        tid, _, _ = cute.arch.thread_idx()
        chunk, head, t = cute.arch.block_idx()
        lane, warp = tid % 32, tid // 32
        group, quarter = lane // 4, lane % 4
        v = chunk * 32 + warp * 8 + group
        wb = [
            load_words(at(w_uv, head, 32 * c + 8 * quarter, v), count=4)
            for c in range(16)
        ]
        for c in range(tid, 15 * 68, 128):  # rows 1-15: zero (no predecessor data)
            store_words(rows.iterator + 544 + c * 8, (Uint32(0),) * 4, shared=True)
        cute.arch.griddepcontrol_wait()
        if cutlass.const_expr(self.trigger == "fence"):
            cute.arch.fence_acq_rel_gpu()
        if cutlass.const_expr(self.trigger in ("wait", "fence")):
            cute.arch.griddepcontrol_launch_dependents()
        S = self.splits
        ms, ls, us = [], [], []
        for c in cutlass.range_constexpr(S):
            m, l = load_words(
                at(stats, t, c, head, 0), count=2, mutable=True, result_type=Float32, word_type="f32"
            )
            ms.append(m)
            ls.append(l)
        for c in cutlass.range_constexpr(S):
            us.append(load_words(
                at(partial, t, c, head, 4 * tid), count=4, mutable=True, result_type=Float32, word_type="f32"
            ))
        top = ms[0]
        for c in cutlass.range_constexpr(1, S):
            top = max(top, ms[c])
        if top == Float32(-float("inf")):
            top = Float32(0)  # every split empty: all weights 0
        acc = zeros(4, Float32)
        total = Float32(0)
        for c in cutlass.range_constexpr(S):
            w = _ex2(ms[c] - top)
            total = total + w * ls[c]
            for k in cutlass.range_constexpr(4):
                acc[k] = acc[k] + w * us[c][k]
        vals = zeros(4, Float32)
        if total > Float32(0):
            for k in cutlass.range_constexpr(4):
                vals[k] = div_rn(acc[k], total) * out_scale
        words = (bf16x2(vals[0], vals[1]), bf16x2(vals[2], vals[3]))
        store_words(rows.iterator + 4 * tid, words, shared=True)
        if chunk == 0:
            store_words(at(latent_out, t, head, 4 * tid), words)
        col = head * 128 + chunk * 32 + warp * 8 + 2 * quarter
        word = Uint32(0)
        if group == 0:
            word = load_words(at(gate, t, col), count=1, mutable=True)[0]
        g0 = unpack_bf16(word)
        if cutlass.const_expr(self.trigger == "merge"):
            cute.arch.griddepcontrol_launch_dependents()
        cute.arch.sync_threads()
        parts = zeros((2, 4), Float32)  # two accumulation chains
        for c in cutlass.range_constexpr(16):
            x, y = (
                load_words(
                    rows.iterator + r * 544 + 32 * c + 8 * quarter, count=4, shared=True
                )
                for r in (group, group + 8)
            )
            for s in cutlass.range_constexpr(2):
                a = (x[2 * s], y[2 * s], x[2 * s + 1], y[2 * s + 1])
                b = (wb[c][2 * s], wb[c][2 * s + 1])
                mma_into(parts[s, None], a, b, bf16=True)
        acc2 = [parts[0, k] + parts[1, k] for k in range(4)]
        if group == 0:
            lo, hi = Float32(0), Float32(0)
            n = load_words(at(lengths, t // self.q), count=1, mutable=True, result_type=Int32)[0]
            if n > 0:
                lo, hi = (
                    Float32(BFloat16(acc2[e])) * Float32(BFloat16(sigmoid(g0[e])))
                    for e in range(2)
                )
            store_words(at(output, t, col), (bf16x2(lo, hi),))


class MlaFoldMerge:
    """MlaValueFold's split merge alone (``K31MK_FOLD_SPLIT_ROWS``, dsa/verify.py): one 128-thread CTA per (head, row);
    thread i merges latent dims 4i..4i+3 of its (row, head) over the ``splits`` partials with MlaValueFold's arithmetic
    (same operations in the same order) and stores the BF16 latent to ``latent_out`` [B, H, 512] (the sparse core's
    latent view). The shipped MlaValue then applies W_UV and the gate over 16-row tiles. Per layer this reads each W_UV
    slice once per 16 rows (MlaValueFold: once per row) and each split partial once (MlaValueFold: once per 32-output
    chunk, 4 times), for one more kernel boundary. Reads nothing before griddepcontrol.wait."""

    def __init__(self, heads: int, batch: int, splits: int = 16):
        self.heads, self.batch, self.splits = heads, batch, splits
        self.grid = (heads, batch, 1)

    @cute.jit
    def __call__(self, *args):
        launch(self.kernel, args, self.grid, 128)

    @cute.kernel
    def kernel(self, args):
        partial, stats, out_scale, latent_out = args
        tid, _, _ = cute.arch.thread_idx()
        head, row, _ = cute.arch.block_idx()
        cute.arch.griddepcontrol_wait()
        n_split = self.splits
        maxes, sums, parts = [], [], []
        for s in cutlass.range_constexpr(n_split):
            m, l = load_words(
                at(stats, row, s, head, 0), count=2, mutable=True, result_type=Float32, word_type="f32"
            )
            maxes.append(m)
            sums.append(l)
        for s in cutlass.range_constexpr(n_split):
            parts.append(load_words(
                at(partial, row, s, head, 4 * tid), count=4, mutable=True, result_type=Float32, word_type="f32"
            ))
        peak = maxes[0]
        for s in cutlass.range_constexpr(1, n_split):
            peak = max(peak, maxes[s])
        if peak == Float32(-float("inf")):
            peak = Float32(0)  # every split empty: all weights 0
        num = zeros(4, Float32)
        den = Float32(0)
        for s in cutlass.range_constexpr(n_split):
            wt = _ex2(maxes[s] - peak)
            den = den + wt * sums[s]
            for k in cutlass.range_constexpr(4):
                num[k] = num[k] + wt * parts[s][k]
        vals = zeros(4, Float32)
        if den > Float32(0):
            for k in cutlass.range_constexpr(4):
                vals[k] = div_rn(num[k], den) * out_scale
        store_words(at(latent_out, row, head, 4 * tid), (bf16x2(vals[0], vals[1]), bf16x2(vals[2], vals[3])))
        cute.arch.griddepcontrol_launch_dependents()
