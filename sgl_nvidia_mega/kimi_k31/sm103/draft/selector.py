# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the SGLang project
"""DFlash2 draft head, stage 2b: ``SelectorWalk``, the selector's whole tail in
one kernel, after ``RankTopK`` (head.py). One 224-thread CTA per request, warp e =
predicted row e (7 per request); K = the selector's top_k, 16 or 32:

1. poll the TP rank top-K lists of the row from this GPU's head box, stage and re-arm;
2. merge them into the global top-K (value desc, smaller global id first, which equals
   torch.topk over the rank-major all-gather; shards are contiguous id ranges): a
   bitonic merge tree over the warp (two 16-lane halves at K 16, the whole warp at K 32);
   unary = the DFlash2 transform of the value;
3. score[p, c] = unary[c] + BF16(<BF16(A[pred_p] * h'), B[cand_c]>) with pred = the
   previous row's candidates (the anchor for row 0), as ``_score_edges``: the K x K x 256
   product on mma.sync; each row p keeps its argmax over c, exact ties to the smaller token
   id (torch.argmax takes the first position, which depends on the candidates' order);
4. warp 0 walks from row 0's p = 0, per request either greedy (the argmax maps, q one-hot)
   or sampled exactly as ``selector_walk_triton``: P = softmax(score / T) over the realized
   row, index = #{c: u_e >= cumsum(P)_c} (clamped), q = P, with Triton's lowered ops
   (div.full, ex2.approx of x log2 e, butterfly sum, Hillis-Steele scan; log2 K steps
   each, as Triton's one-warp reduce and scan over K lanes). Every warp then writes its
   row's token. The uniforms come from the caller's in-graph ``uniform_()``.
"""

from functools import cache

import cutlass
import cutlass.cute as cute
import numpy as np
import torch
from cutlass import BFloat16, Float32, Int32, Int64
from cutlass.cute.runtime import make_fake_stream

from ..k3.common.host import fake, stream, to_cute
from ..k3.common.mailbox import poll, rearm, rearm_fence
from ..k3.common.primitives import (
    alloc,
    asm,
    at,
    bf16x2,
    bits_float,
    launch,
    load_words,
    mma_into,
    store_words,
    unpack_bf16,
    zeros,
)
from .head import SELECTOR_RANK, TOPK, TOPKS, better, ordered

SLOTS = 7  # predicted rows per request (block 8 minus the verified anchor)
LOG2E = 1.4426950408889634


@asm(Float32, pure=True)
def div_full(a, b):
    """Triton's lowering of an FP32 ``/``."""
    return "div.full.f32 $0,$1,$2;", "=f,f,f", (Float32(a), Float32(b))


@asm(Float32, pure=True)
def ex2_approx(x):
    """Triton's lowering of ``tl.exp`` after the multiply by log2 e."""
    return "ex2.approx.f32 $0,$1;", "=f,f", (Float32(x),)


@asm(Float32, pure=True)
def mul_rn(a, b):
    return "mul.rn.f32 $0,$1,$2;", "=f,f,f", (Float32(a), Float32(b))


@cute.jit
def take(key, gid, lane_bit, distance: cutlass.Constexpr[int]):
    """One bitonic compare-exchange with lane ^ ``distance``: ``lane_bit`` clear keeps
    the better entry, set keeps the worse."""
    other_key = cute.arch.shuffle_sync_bfly(key, distance)
    other_gid = cute.arch.shuffle_sync_bfly(gid, distance)
    # Global ids are distinct, so exactly one of the two entries is better.
    lower_takes = better(other_key, other_gid, key, gid) & (lane_bit == 0)
    upper_takes = better(key, gid, other_key, other_gid) & (lane_bit != 0)
    if lower_takes | upper_takes:
        key, gid = other_key, other_gid
    return key, gid


@cute.jit
def sort_bitonic(key, gid, lane, width: cutlass.Constexpr[int]):
    """Sort each ``width``-lane group's bitonic sequence descending."""
    for d in cutlass.range_constexpr(width.bit_length() - 2, -1, -1):
        key, gid = take(key, gid, lane & (1 << d), 1 << d)
    return key, gid


@cute.jit
def half_clean(key, gid, other_key, other_gid):
    """The better of two entries (both lanes of a half-cleaner keep the better one)."""
    if better(other_key, other_gid, key, gid):
        key, gid = other_key, other_gid
    return key, gid


@cute.jit
def score_max(v1, id1, v2, id2):
    """NaN first (as torch.argmax), then the larger score, then the smaller token id: exact
    score ties do not depend on the candidates' order."""
    nan1, num1, num2 = v1 != v1, v1 == v1, v2 == v2
    greater = (v1 > v2) | ((v1 == v2) & (id1 < id2))
    return (nan1 & (num2 | (id1 < id2))) | (num1 & num2 & greater)


class SelectorWalk:
    """Arguments: box [TP, R, 2 K] int32 (this GPU's head box), head_out [R, num_org +
    256] BF16, anchor [B] int64, predecessor / successor codebooks [vocab, 256] BF16,
    uniforms [B, 7] FP32, temperatures [B] FP32, greedy [B] u8, tokens [R] int64,
    candidates [B, 7, K] int64, q [B, 7, K] FP32, scores [B, 7, K, K] FP32 (written
    only with ``export_scores``, a test hook); R = 7 B, K = ``topk``."""

    def __init__(
        self,
        tp,
        num_org,
        multiplier,
        softcap,
        deadline_ns=0,
        export_scores=False,
        topk=TOPK,
    ):
        if tp not in (4, 8, 16):
            raise ValueError(f"SelectorWalk merges 4, 8 or 16 rank lists, got tp={tp}")
        if topk not in TOPKS:
            raise ValueError(f"SelectorWalk serves top_k {TOPKS}, got {topk}")
        self.tp, self.num_org, self.deadline_ns, self.topk = (
            tp,
            num_org,
            deadline_ns,
            topk,
        )
        self.export_scores = export_scores
        self.multiplier = float(np.float32(multiplier))
        self.softcap = None if softcap is None else float(np.float32(softcap))
        # torch divides by a CPU scalar as a multiply by its FP32 reciprocal.
        self.inverse_softcap = (
            None if softcap is None else float(np.float32(1) / np.float32(softcap))
        )

    @cute.jit
    def __call__(self, *args):
        launch(self.kernel, args, (args[1].shape[0] // SLOTS, 1, 1), 32 * SLOTS)

    @cute.kernel
    def kernel(self, args):
        box, head_out, anchor, predecessor, successor = args[0:5]
        uniforms, temperatures, greedy, tokens, candidates, q, scores = args[5:12]
        request, _, _ = cute.arch.block_idx()
        tid, _, _ = cute.arch.thread_idx()
        lane, e = tid % 32, tid // 32
        row = request * SLOTS + e
        topk = self.topk
        smem = cutlass.utils.SmemAllocator()
        if cutlass.const_expr(topk == TOPK):
            lists = alloc(smem, Int32, (SLOTS, self.tp, 2 * TOPK))
        else:
            # K 32 at TP 16 would take 56 KB for lists + lattice: the lattice reuses the
            # lists' space, which every warp has finished reading at the sync before score.
            words = max(SLOTS * self.tp * 2 * topk, SLOTS * topk * topk)
            pool = alloc(smem, Int32, words)
            lists = cute.make_tensor(
                pool.iterator,
                cute.make_layout(
                    (SLOTS, self.tp, 2 * topk), stride=(self.tp * 2 * topk, 2 * topk, 1)
                ),
            )
            lattice = cute.make_tensor(
                cute.recast_ptr(pool.iterator, dtype=Float32),
                cute.make_layout((SLOTS, topk, topk), stride=(topk * topk, topk, 1)),
            )
        ids = alloc(smem, Int32, (SLOTS, topk))
        unary = alloc(smem, Float32, (SLOTS, topk))
        maps = alloc(smem, Int32, (SLOTS, topk))
        if cutlass.const_expr(topk == TOPK):
            lattice = alloc(smem, Float32, (SLOTS, TOPK, TOPK))
        path = alloc(smem, Int32, SLOTS)
        cute.arch.griddepcontrol_launch_dependents()
        self.gather(box, lists, row, e, lane)
        cute.arch.griddepcontrol_wait()
        quarter = lane % 4
        hidden = [
            load_words(
                at(head_out, row, self.num_org + 32 * k + 8 * quarter),
                count=4,
                mutable=True,
            )
            for k in range(SELECTOR_RANK // 32)
        ]
        key, gid = self.merge(lists, e, lane)
        if lane < topk:
            ids[e, lane] = gid
            unary[e, lane] = self.transform(bits_float(ordered(key)))
            candidates[request, e, lane] = Int64(gid)
        cute.arch.sync_threads()
        if cutlass.const_expr(topk == TOPK):
            self.score(
                predecessor,
                successor,
                anchor,
                hidden,
                ids,
                unary,
                maps,
                lattice,
                request,
                e,
                lane,
            )
        else:
            self.score_wide(
                predecessor,
                successor,
                anchor,
                hidden,
                ids,
                unary,
                maps,
                lattice,
                request,
                e,
                lane,
            )
        cute.arch.sync_threads()
        if cutlass.const_expr(self.export_scores):
            for i in range(tid, SLOTS * topk * topk, 32 * SLOTS):
                scores[request, i // (topk * topk), (i // topk) % topk, i % topk] = (
                    lattice[i // (topk * topk), (i // topk) % topk, i % topk]
                )
        if e == 0:
            self.walk(
                lattice, maps, path, q, uniforms, temperatures, greedy, request, lane
            )
        cute.arch.sync_threads()
        if lane == 0:
            tokens[row] = Int64(ids[e, path[e]])

    @cute.jit
    def gather(self, box, lists, row, e, lane):
        """Poll the row's TP lists (F = K / 2 16 B fragments per list; lane: rank lane // F
        + (32 / F) i, fragment lane % F), stage them in shared memory, re-arm."""
        fragments = self.topk // 2
        ranks = 32 // fragments  # per poll
        rank, fragment = lane // fragments, lane % fragments
        base = at(box, rank, row, 4 * fragment)
        stride = cute.crd2idx((ranks, 0, 0), box.layout)
        values = poll(base, stride, self.tp // ranks, self.deadline_ns)
        for i in cutlass.range_constexpr(self.tp // ranks):
            words = tuple(values[i, k] for k in range(4))
            store_words(
                at(lists, e, rank + ranks * i, 4 * fragment), words, shared=True
            )
            rearm(base + i * stride)
        rearm_fence()
        cute.arch.sync_warp()

    @cute.jit
    def entry(self, lists, e, rank, position):
        return ordered(lists[e, rank, 2 * position]), lists[e, rank, 2 * position + 1]

    @cute.jit
    def merge(self, lists, e, lane):
        """Global top-K of the row: each K-lane group (two 16-lane halves at K 16, the warp
        at K 32) merges its share of the TP lists pairwise (half-cleaner + bitonic sort);
        at K 16 the halves then merge. Lane l < K ends with the l-th best (key, global id)."""
        topk = self.topk
        groups = 32 // topk
        group, l = lane // topk, lane % topk
        width = self.tp // (2 * groups)
        keys = cute.make_rmem_tensor(width, Int32)
        gids = cute.make_rmem_tensor(width, Int32)
        for s in cutlass.range_constexpr(width):
            pair = group * width + s
            key, gid = self.entry(lists, e, 2 * pair, l)
            other_key, other_gid = self.entry(lists, e, 2 * pair + 1, topk - 1 - l)
            key, gid = half_clean(key, gid, other_key, other_gid)
            keys[s], gids[s] = sort_bitonic(key, gid, lane, topk)
        for level in cutlass.range_constexpr(width.bit_length() - 1):
            for s in cutlass.range_constexpr(width >> (level + 1)):
                # lane ^ (K - 1) holds the other list's entry K - 1 - l: its reversal.
                other_key = cute.arch.shuffle_sync_bfly(keys[2 * s + 1], topk - 1)
                other_gid = cute.arch.shuffle_sync_bfly(gids[2 * s + 1], topk - 1)
                key, gid = half_clean(keys[2 * s], gids[2 * s], other_key, other_gid)
                keys[s], gids[s] = sort_bitonic(key, gid, lane, topk)
        key, gid = keys[0], gids[0]
        if cutlass.const_expr(groups == 2):
            other_key = cute.arch.shuffle_sync_bfly(key, 31)
            other_gid = cute.arch.shuffle_sync_bfly(gid, 31)
            key, gid = half_clean(key, gid, other_key, other_gid)
            key, gid = sort_bitonic(key, gid, lane, 16)
        return key, gid

    @cute.jit
    def transform(self, value):
        """DFlash2DraftModel._transform_unary_logits on one FP32 value."""
        x = value
        if cutlass.const_expr(self.multiplier != 1.0):
            x = x * Float32(self.multiplier)
        if cutlass.const_expr(self.softcap is not None):
            x = cute.math.tanh(x * Float32(self.inverse_softcap), fastmath=False)
            x = x * Float32(self.softcap)
        return x

    @cute.jit
    def score(
        self,
        predecessor,
        successor,
        anchor,
        hidden,
        ids,
        unary,
        maps,
        lattice,
        request,
        e,
        lane,
    ):
        """Row e's 16 x 16 lattice on mma.sync m16n8k16 (A = 16 predecessor rows times
        h', B = 16 candidates in two n-tiles), then maps[e, p] = argmax over c (score_max)."""
        group, quarter = lane // 4, lane % 4
        pred = []
        for j in cutlass.range_constexpr(2):
            token = Int32(anchor[request])
            if e > 0:
                token = ids[e - 1, group + 8 * j]
            pred.append(token)
        # Every operand row loads 8 bf16 per 32-wide chunk at 8 * quarter; the fragments
        # use the same k permutation for A and B (as DraftMlaValue).
        rows_a = [
            [
                load_words(at(predecessor, pred[j], 32 * k + 8 * quarter), count=4)
                for k in range(SELECTOR_RANK // 32)
            ]
            for j in range(2)
        ]
        rows_b = [
            [
                load_words(
                    at(successor, ids[e, group + 8 * n], 32 * k + 8 * quarter), count=4
                )
                for k in range(SELECTOR_RANK // 32)
            ]
            for n in range(2)
        ]
        parts = zeros((2, 2, 4), Float32)  # (n-tile, accumulation chain, fragment)
        for k in cutlass.range_constexpr(SELECTOR_RANK // 32):
            x, y = (self.scaled(rows_a[j][k], hidden[k]) for j in range(2))
            for s in cutlass.range_constexpr(2):
                a = (x[2 * s], y[2 * s], x[2 * s + 1], y[2 * s + 1])
                for n in cutlass.range_constexpr(2):
                    b = (rows_b[n][k][2 * s], rows_b[n][k][2 * s + 1])
                    mma_into(parts[n, s, None], a, b, bf16=True)
        for h in cutlass.range_constexpr(2):  # rows p = group + 8 h
            scores = []  # this lane's 4 columns
            for n in cutlass.range_constexpr(2):
                for col in cutlass.range_constexpr(2):
                    c = 8 * n + 2 * quarter + col
                    product = parts[n, 0, 2 * h + col] + parts[n, 1, 2 * h + col]
                    v = unary[e, c] + Float32(BFloat16(product))
                    lattice[e, group + 8 * h, c] = v
                    scores.append((v, ids[e, c], Int32(c)))
            best_v, best_id, best_c = scores[0]
            for v, token, c in scores[1:]:
                if score_max(v, token, best_v, best_id):
                    best_v, best_id, best_c = v, token, c
            for d in cutlass.range_constexpr(2):
                other_v = cute.arch.shuffle_sync_bfly(best_v, 1 << d)
                other_id = cute.arch.shuffle_sync_bfly(best_id, 1 << d)
                other_c = cute.arch.shuffle_sync_bfly(best_c, 1 << d)
                if score_max(other_v, other_id, best_v, best_id):
                    best_v, best_id, best_c = other_v, other_id, other_c
            if quarter == 0:
                maps[e, group + 8 * h] = best_c

    @cute.jit
    def score_wide(
        self,
        predecessor,
        successor,
        anchor,
        hidden,
        ids,
        unary,
        maps,
        lattice,
        request,
        e,
        lane,
    ):
        """``score`` at K 32: the same per-entry math (k permutation, two accumulation
        chains, BF16 rounding, score_max) over 2 m-tiles (rows p = 16 m + group + 8 j) x 4
        n-tiles (columns 8 n + group); the operands load per 32-wide k chunk, as all 4 + 4
        rows at once (256 registers) would not fit next to the accumulators."""
        topk = self.topk
        mtiles, ntiles = topk // 16, topk // 8
        group, quarter = lane // 4, lane % 4
        pred = []
        for r in cutlass.range_constexpr(2 * mtiles):  # row group + 8 r
            token = Int32(anchor[request])
            if e > 0:
                token = ids[e - 1, group + 8 * r]
            pred.append(token)
        succ = [ids[e, group + 8 * n] for n in range(ntiles)]
        parts = zeros(
            (mtiles, ntiles, 2, 4), Float32
        )  # (m-tile, n-tile, chain, fragment)
        for k in cutlass.range_constexpr(SELECTOR_RANK // 32):
            rows_a = [
                load_words(at(predecessor, pred[r], 32 * k + 8 * quarter), count=4)
                for r in range(2 * mtiles)
            ]
            rows_b = [
                load_words(at(successor, succ[n], 32 * k + 8 * quarter), count=4)
                for n in range(ntiles)
            ]
            for m in cutlass.range_constexpr(mtiles):
                x, y = (self.scaled(rows_a[2 * m + j], hidden[k]) for j in range(2))
                for s in cutlass.range_constexpr(2):
                    a = (x[2 * s], y[2 * s], x[2 * s + 1], y[2 * s + 1])
                    for n in cutlass.range_constexpr(ntiles):
                        b = (rows_b[n][2 * s], rows_b[n][2 * s + 1])
                        mma_into(parts[m, n, s, None], a, b, bf16=True)
        for m in cutlass.range_constexpr(mtiles):
            for h in cutlass.range_constexpr(2):  # rows p = 16 m + group + 8 h
                p = 16 * m + group + 8 * h
                scores = []  # this lane's 2 x 4 columns
                for n in cutlass.range_constexpr(ntiles):
                    for col in cutlass.range_constexpr(2):
                        c = 8 * n + 2 * quarter + col
                        product = (
                            parts[m, n, 0, 2 * h + col] + parts[m, n, 1, 2 * h + col]
                        )
                        v = unary[e, c] + Float32(BFloat16(product))
                        lattice[e, p, c] = v
                        scores.append((v, ids[e, c], Int32(c)))
                best_v, best_id, best_c = scores[0]
                for v, token, c in scores[1:]:
                    if score_max(v, token, best_v, best_id):
                        best_v, best_id, best_c = v, token, c
                for d in cutlass.range_constexpr(2):
                    other_v = cute.arch.shuffle_sync_bfly(best_v, 1 << d)
                    other_id = cute.arch.shuffle_sync_bfly(best_id, 1 << d)
                    other_c = cute.arch.shuffle_sync_bfly(best_c, 1 << d)
                    if score_max(other_v, other_id, best_v, best_id):
                        best_v, best_id, best_c = other_v, other_id, other_c
                if quarter == 0:
                    maps[e, p] = best_c

    @cute.jit
    def walk(
        self, lattice, maps, path, q, uniforms, temperatures, greedy, request, lane
    ):
        """Warp 0, lane c < K = candidate c: 7 dependent steps from p = 0."""
        topk = self.topk
        steps = topk.bit_length() - 1  # Triton's one-warp reduce / scan over K lanes
        sampled = greedy[request] == 0
        temperature = temperatures[request]
        prev = Int32(0)
        for e in cutlass.range_constexpr(SLOTS):
            index = maps[e, prev]
            prob = Float32(0)
            if lane == index:
                prob = Float32(1)
            if sampled:
                s = Float32(float("-inf"))
                if lane < topk:
                    s = lattice[e, prev, lane]
                scaled = div_full(s, temperature)
                top = scaled
                for d in cutlass.range_constexpr(steps - 1, -1, -1):
                    top = cute.arch.fmax(top, cute.arch.shuffle_sync_bfly(top, 1 << d))
                ex = Float32(0)
                if lane < topk:
                    ex = ex2_approx(mul_rn(scaled - top, LOG2E))
                total = ex
                for d in cutlass.range_constexpr(steps - 1, -1, -1):
                    total = total + cute.arch.shuffle_sync_bfly(total, 1 << d)
                prob = div_full(ex, total)
                cum = prob
                for d in cutlass.range_constexpr(steps):
                    src = lane - (1 << d)
                    if src < 0:
                        src = Int32(0)
                    other = cute.arch.shuffle_sync(cum, src)
                    if lane >= (1 << d):
                        cum = other + cum
                hit = (uniforms[request, e] >= cum) & (lane < topk)
                index = Int32(cute.arch.popc(cute.arch.vote_ballot_sync(hit)))
                if index > topk - 1:
                    index = Int32(topk - 1)
            if lane < topk:
                q[request, e, lane] = prob
            if lane == 0:
                path[e] = index
            prev = index

    @cute.jit
    def scaled(self, a_words, h_words):
        """BF16(A * h') of 8 values (the elementwise product rounds to BF16 first)."""
        out = []
        for w in cutlass.range_constexpr(4):
            a_lo, a_hi = unpack_bf16(a_words[w])
            h_lo, h_hi = unpack_bf16(h_words[w])
            out.append(bf16x2(a_lo * h_lo, a_hi * h_hi))
        return tuple(out)


@cache
def compile_selector_walk(
    rows,
    width,
    tp,
    vocab,
    num_org,
    multiplier,
    softcap,
    deadline_ns,
    export_scores,
    topk=TOPK,
):
    requests = rows // SLOTS
    args = (
        fake((tp, rows, 2 * topk), (rows * 2 * topk, 2 * topk, 1), torch.int32),
        fake((rows, width), (width, 1)),
        fake((requests,), (1,), torch.int64, align=8),
        fake((vocab, SELECTOR_RANK), (SELECTOR_RANK, 1)),
        fake((vocab, SELECTOR_RANK), (SELECTOR_RANK, 1)),
        fake((requests, SLOTS), None, torch.float32),
        fake((requests,), (1,), torch.float32, align=4),
        fake((requests,), (1,), torch.uint8, align=1),
        fake((rows,), (1,), torch.int64),
        fake((requests, SLOTS, topk), None, torch.int64),
        fake((requests, SLOTS, topk), None, torch.float32),
        fake(
            (requests if export_scores else 1, SLOTS, topk, topk), None, torch.float32
        ),
    )
    walk = SelectorWalk(
        tp, num_org, multiplier, softcap, deadline_ns, export_scores, topk
    )
    return cute.compile(walk, *args, make_fake_stream())


class DraftSelector:
    """The selector tail over a DraftHead's ``box`` [TP, R, 2 K] and ``head_out`` [R,
    num_org + 256], K = ``topk`` (the selector's top_k, 16 or 32). One captured graph
    serves greedy and sampling batches, as ``_SelectorDraftSampler``: ``stage_sampling``
    refreshes the static ``temperatures`` [B] / ``greedy_mask`` [B] on the host before a
    replay; ``run(anchor)`` draws ``uniforms`` [B, 7] in-graph (the same philox consumption
    as today's sampler) and returns (tokens [R] int64, candidates [B, 7, K] int64, q [B, 7,
    K] FP32)."""

    def __init__(
        self,
        predecessor,
        successor,
        *,
        box,
        head_out,
        num_org,
        output_multiplier,
        final_logit_softcapping,
        deadline_ns=0,
        export_scores=False,
        topk=TOPK,
    ):
        for name, table in (("predecessor", predecessor), ("successor", successor)):
            if table.dtype != torch.bfloat16 or table.shape[1] != SELECTOR_RANK:
                raise ValueError(f"{name} codebook {table.dtype} {tuple(table.shape)}")
            if not table.is_contiguous():
                raise ValueError(f"{name} codebook is not contiguous")
        tp, rows = box.shape[0], box.shape[1]
        if (
            rows % SLOTS
            or box.shape[2] != 2 * topk
            or head_out.shape != (rows, num_org + SELECTOR_RANK)
        ):
            raise ValueError(
                f"box {tuple(box.shape)} / head_out {tuple(head_out.shape)} "
                f"at top_k {topk}"
            )
        self.device, self.requests, self.topk = box.device, rows // SLOTS, topk
        b, dev = self.requests, self.device
        self.tokens = torch.empty(rows, dtype=torch.int64, device=dev)
        self.candidates = torch.empty(b, SLOTS, topk, dtype=torch.int64, device=dev)
        self.q = torch.empty(b, SLOTS, topk, dtype=torch.float32, device=dev)
        self.uniforms = torch.empty(b, SLOTS, dtype=torch.float32, device=dev)
        self.temperatures = torch.ones(b, dtype=torch.float32, device=dev)
        self.greedy_mask = torch.ones(b, dtype=torch.bool, device=dev)
        # [B, 7, K, K] FP32 lattice for tests (``export_scores``); else a stub
        self.scores = torch.empty(
            b if export_scores else 1,
            SLOTS,
            topk,
            topk,
            dtype=torch.float32,
            device=dev,
        )
        self.code = compile_selector_walk(
            rows,
            head_out.shape[1],
            tp,
            predecessor.shape[0],
            num_org,
            output_multiplier,
            final_logit_softcapping,
            deadline_ns,
            export_scores,
            topk,
        )
        self.head_args = (to_cute(box), to_cute(head_out))
        self.out_args = (
            to_cute(predecessor),
            to_cute(successor),
            to_cute(self.uniforms),
            to_cute(self.temperatures, 4),
            to_cute(self.greedy_mask.view(torch.uint8), 1),
            to_cute(self.tokens),
            to_cute(self.candidates),
            to_cute(self.q),
            to_cute(self.scores),
        )

    def stage_sampling(self, temperatures=None, greedy_mask=None):
        """Host refresh before a replay, as ``_SelectorDraftSampler.stage_sampling_params``:
        temperatures clamped to >= 1e-5; None = greedy."""
        if temperatures is None:
            self.temperatures.fill_(1.0)
            self.greedy_mask.fill_(True)
            return
        torch.clamp(
            temperatures.view(-1)[: self.requests].to(torch.float32),
            min=1e-5,
            out=self.temperatures,
        )
        self.greedy_mask.copy_(greedy_mask.view(-1)[: self.requests])

    def run(self, anchor):
        if anchor.shape != (self.requests,) or anchor.dtype != torch.int64:
            raise ValueError(f"anchor {anchor.dtype} {tuple(anchor.shape)}")
        self.uniforms.uniform_()  # in-graph philox draw, one [B, 7] per replay
        self.code(
            *self.head_args,
            to_cute(anchor.contiguous(), 8),
            *self.out_args,
            stream(self.device),
        )
        return self.tokens, self.candidates, self.q
