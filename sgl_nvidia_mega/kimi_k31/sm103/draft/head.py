# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the SGLang project
"""DFlash2 draft head, stage 2a: the vocab-parallel candidate top-k's rank half.

``DraftHead.run(rows)``:
1. one split-K projection over cat(target lm_head rank shard [num_org, 7168], the
   selector's hidden_projection [256, 7168]) -> head_out [R, num_org + 256] BF16: this
   rank's logits (global id = vocab_start + column) and h' (BF16 like nn.Linear);
2. ``RankTopK`` (one 256-thread CTA per predicted row): the rank's top-K (K = the
   selector's top_k, 16 or 32) of the row's logits, ordered by (FP32 value desc, global id
   asc), published to every rank's head box [TP, R, 2 K] uint32 at [rank, row] as K (FP32
   bits of the value, global id) word pairs (K / 2 multicast 16 B vectors). -0.0 counts as
   +0.0 and is published as +0.0, so no word is the Lamport sentinel. ``SelectorWalk``
   polls the TP rows, merges them into the global top-K and re-arms.

The order (value desc, smaller global id first) equals torch.topk(sorted) over the
rank-major all-gather of per-rank top-k with ties to the smaller id (shards are
contiguous vocabulary ranges).

RankTopK's three merges (warp, CTA, cluster) are lane-distributed bitonic networks: each
thread's 8 values form 8 32-lane sequences, each sorted across the warp and merged pairwise
(half-cleaner + bitonic sort) into the warp's sorted top-32; the CTA and cluster lists merge
the same way. They give the exact (value desc, id asc) top-K of the K serial warp-merge
rounds they replaced, so the same published words (byte-equal at K16; one-GPU unit about
-5.5 us per round at K16, -25 us at K32)."""

from functools import cache

import cutlass
import cutlass.cute as cute
import torch
from cutlass import Int32, Int64, Uint32
from cutlass.cute.runtime import make_fake_stream

from ..k3.common.host import fake, stream, to_cute
from ..k3.common.primitives import (
    alloc,
    at,
    bits_float,
    launch,
    load_vector,
    multicast_words,
    push_float,
    store_words,
    wait_phase_zero_cluster,
)
from ..k3.gemm.projection import PreparedProjection
from .tactics import draft_tactic

TOPK = (
    16  # the default selector top_k; RankTopK / SelectorWalk take ``topk`` (16 or 32)
)
TOPKS = (16, 32)
SELECTOR_RANK = 256
PARTS = 8  # CTAs (one cluster) per row
EMPTY = -0x80000000  # signed ordered key below every value (incl. -inf)
NO_ID = 0x7FFFFFFF


@cute.jit
def ordered(bits):
    """Signed order-preserving key of FP32 ``bits`` (Int32; -0.0 already +0.0): signed
    comparison of keys is FP32 comparison of the values. Its own inverse."""
    key = Int32(bits)
    if key < Int32(0):
        key = key ^ Int32(0x7FFFFFFF)
    return key


@cute.jit
def better(v1, i1, v2, i2):
    return (v1 > v2) | ((v1 == v2) & (i1 < i2))


@cute.jit
def lane_take(key, gid, keep_worse, distance: cutlass.Constexpr[int]):
    """One compare-exchange with lane ^ ``distance``: ``keep_worse`` clear keeps the better
    entry, set keeps the worse (equal entries, the EMPTY pads, stay put)."""
    other_key = cute.arch.shuffle_sync_bfly(key, distance)
    other_gid = cute.arch.shuffle_sync_bfly(gid, distance)
    lower_takes = better(other_key, other_gid, key, gid) & (keep_worse == 0)
    upper_takes = better(key, gid, other_key, other_gid) & (keep_worse != 0)
    if lower_takes | upper_takes:
        key, gid = other_key, other_gid
    return key, gid


@cute.jit
def lane_sort(key, gid, lane):
    """Sort the warp's 32-lane sequence descending: the bitonic sort network (blocks of
    2..32, block ``lane >> s`` odd sorted the other way until the last)."""
    for s in cutlass.range_constexpr(1, 6):
        for d in cutlass.range_constexpr(s - 1, -1, -1):
            key, gid = lane_take(key, gid, ((lane >> d) ^ (lane >> s)) & 1, 1 << d)
    return key, gid


@cute.jit
def lane_merge(key, gid, other_key, other_gid, lane):
    """The sorted top-32 of two sorted 32-lane lists: lane ^ 31 holds the other list's
    entry 31 - lane (its reversal), the better of the two forms a bitonic sequence."""
    other_key = cute.arch.shuffle_sync_bfly(other_key, 31)
    other_gid = cute.arch.shuffle_sync_bfly(other_gid, 31)
    if better(other_key, other_gid, key, gid):
        key, gid = other_key, other_gid
    for d in cutlass.range_constexpr(4, -1, -1):
        key, gid = lane_take(key, gid, (lane >> d) & 1, 1 << d)
    return key, gid


@cute.jit
def lane_merge_all(
    keys, gids, lane, count: cutlass.Constexpr[int], real: cutlass.Constexpr[int]
):
    """Merge ``count`` (a power of two) sorted 32-lane lists, Python lists of registers,
    pairwise in place into one; lane l ends with the l-th best. Lists from ``real`` on are
    all EMPTY pads: a merge with one is the identity, so it is not built."""
    for level in cutlass.range_constexpr(count.bit_length() - 1):
        for i in cutlass.range_constexpr(0, count, 2 << level):
            if cutlass.const_expr(i + (1 << level) < real):
                keys[i], gids[i] = lane_merge(
                    keys[i],
                    gids[i],
                    keys[i + (1 << level)],
                    gids[i + (1 << level)],
                    lane,
                )
    return keys[0], gids[0]


@cute.jit
def lane_pair(key, gid, lane):
    """Lane v (< 16) gets entries 2 v and 2 v + 1 of a lane-distributed list: the
    (key, id, key, id) words of the list's v-th 16 B vector."""
    first = (2 * lane) & 31
    return (
        cute.arch.shuffle_sync(key, first),
        cute.arch.shuffle_sync(gid, first),
        cute.arch.shuffle_sync(key, first + 1),
        cute.arch.shuffle_sync(gid, first + 1),
    )


class RankTopK:
    """Arguments: head_out [R, >= num_org] BF16, box [TP, R, 2 topk] int32 (this GPU's view),
    mc (the box's multicast address; ``local``: unused, stores go to ``box``), rank,
    vocab_start, num_org.

    One 8-CTA cluster per row: CTA c takes columns [c * num_org / 8, (c + 1) * num_org / 8),
    one 16 B vector (8 logits) per thread; register top-``topk`` lists, warp merges, then a CTA
    top-``topk``. CTAs 1-7 push their lists into CTA 0's shared memory (DSMEM st.async on one
    mbarrier); CTA 0 merges the 8 lists and publishes the rank's top-``topk``."""

    def __init__(self, num_org: int, local: bool = False, topk: int = TOPK):
        if num_org % (PARTS * 8 * 32):
            raise ValueError(
                f"RankTopK wants num_org % {PARTS * 8 * 32} == 0, got {num_org}"
            )
        if topk not in TOPKS:
            raise ValueError(f"RankTopK serves top_k {TOPKS}, got {topk}")
        self.num_org, self.local, self.topk = num_org, local, topk
        self.threads = num_org // PARTS // 8  # 160 at num_org 10240
        self.warps = self.threads // 32
        if topk > TOPK and self.threads > 320:
            # K 32 keeps 4 x 32-entry register lists per thread: validated up to 320 threads
            # (TP 8); a wider shard would compile into an unverified register budget.
            raise ValueError(
                f"RankTopK top_k {topk} is untested at {self.threads} threads per CTA "
                f"(num_org {num_org})"
            )

    @cute.jit
    def __call__(self, *args):
        launch(self.kernel, args, (PARTS, args[0].shape[0], 1), self.threads, PARTS)

    @cute.kernel
    def kernel(self, args):
        head_out, box, mc, rank, vocab_start = args[0:5]
        topk = self.topk
        part, row, _ = cute.arch.block_idx()
        tid, _, _ = cute.arch.thread_idx()
        lane, warp = tid % 32, tid // 32
        smem = cutlass.utils.SmemAllocator()
        warp_keys = alloc(smem, Int32, (self.warps, topk))
        warp_ids = alloc(smem, Int32, (self.warps, topk))
        lists = alloc(
            smem, Int32, (PARTS, 2 * topk)
        )  # CTA 0: every CTA's (key, id) pairs
        bar = alloc(smem, cutlass.Uint64, 1)
        if (part == 0) & (tid == 0):
            cute.arch.mbarrier_init(bar.iterator, 1)
            cute.arch.mbarrier_init_fence()
            cute.arch.mbarrier_arrive_and_expect_tx(
                bar.iterator, (PARTS - 1) * 2 * topk * 4
            )
        cute.arch.cluster_arrive_relaxed()  # CTA 0's barrier before the peers' pushes
        common = (
            head_out,
            box,
            mc,
            rank,
            vocab_start,
            warp_keys,
            warp_ids,
            lists,
            bar,
            part,
            row,
            tid,
            lane,
            warp,
        )
        self.bitonic_merges(*common)

    @cute.jit
    def bitonic_merges(
        self,
        head_out,
        box,
        mc,
        rank,
        vocab_start,
        warp_keys,
        warp_ids,
        lists,
        bar,
        part,
        row,
        tid,
        lane,
        warp,
    ):
        """The kernel's three merges as lane-distributed bitonic networks (warp: 8 sorted
        32-lane sequences merged; CTA: the warps' lists; cluster: the 8 CTAs' lists)."""
        topk = self.topk
        cute.arch.griddepcontrol_wait()
        cute.arch.griddepcontrol_launch_dependents()
        col = (part * self.threads + tid) * 8
        words = load_vector(at(head_out, row, col))
        keys, gids = [], []
        for j in cutlass.range_constexpr(4):
            for half in cutlass.range_constexpr(2):
                bits = Int32(Uint32(words[j]) << 16)
                if cutlass.const_expr(half == 1):
                    bits = Int32(Uint32(words[j]) & Uint32(0xFFFF0000))
                if bits == Int32(EMPTY):  # -0.0 -> +0.0
                    bits = Int32(0)
                key, gid = lane_sort(ordered(bits), Int32(col + 2 * j + half), lane)
                keys.append(key)
                gids.append(gid)
        key, gid = lane_merge_all(
            keys, gids, lane, 8, 8
        )  # the warp's top-32, lane l = l-th best
        if lane < topk:
            warp_keys[warp, lane] = key
            warp_ids[warp, lane] = gid
        cute.arch.sync_threads()
        cute.arch.cluster_wait()
        if warp == 0:
            # Own names per stage: a list reassigned inside this dynamic branch must keep its type (length), and the
            # CTA stage holds a power of two >= the warp count (16 lists at TP 8), not the warp stage's 8.
            cta_keys, cta_gids = [], []
            for w in cutlass.range_constexpr(1 << (self.warps - 1).bit_length()):
                key, gid = Int32(EMPTY), Int32(NO_ID)
                if cutlass.const_expr(w < self.warps):
                    if lane < topk:
                        key, gid = warp_keys[w, lane], warp_ids[w, lane]
                cta_keys.append(key)
                cta_gids.append(gid)
            key, gid = lane_merge_all(
                cta_keys, cta_gids, lane, len(cta_keys), self.warps
            )  # this CTA's top
            pair = lane_pair(key, gid, lane)
            if lane < topk // 2:  # topk / 2 vectors of (key, id, key, id)
                if part == 0:
                    for k in cutlass.range_constexpr(4):
                        lists[0, 4 * lane + k] = pair[k]
                else:
                    floats = tuple(bits_float(Uint32(w)) for w in pair)
                    push_float(
                        lists.iterator + part * 2 * topk + 4 * lane,
                        floats,
                        bar.iterator,
                        0,
                    )
        if part == 0:
            wait_phase_zero_cluster(bar.iterator)
            cute.arch.sync_threads()
            if warp == 0:
                rank_keys, rank_gids = [], []
                for p in cutlass.range_constexpr(PARTS):
                    key, gid = Int32(EMPTY), Int32(NO_ID)
                    if lane < topk:
                        key, gid = lists[p, 2 * lane], lists[p, 2 * lane + 1]
                    rank_keys.append(key)
                    rank_gids.append(gid)
                key, gid = lane_merge_all(
                    rank_keys, rank_gids, lane, PARTS, PARTS
                )  # the rank's top
                pair = lane_pair(key, gid, lane)
                if lane < topk // 2:
                    out = (
                        Uint32(ordered(pair[0])),
                        Uint32(pair[1] + vocab_start),
                        Uint32(ordered(pair[2])),
                        Uint32(pair[3] + vocab_start),
                    )
                    if cutlass.const_expr(self.local):
                        store_words(at(box, rank, row, 4 * lane), out)
                    else:
                        offset = cute.crd2idx((rank, row, 4 * lane), box.layout)
                        # v4 multimem.st takes .f32 (stored as raw bits), not .b32.
                        multicast_words(mc + Int64(offset) * 4, out)


@cache
def compile_rank_topk(rows, width, tp, num_org, local, topk=TOPK):
    head_out = fake((rows, width), (width, 1))
    box = fake((tp, rows, 2 * topk), (rows * 2 * topk, 2 * topk, 1), torch.int32)
    args = (head_out, box, cutlass.Int64(0), cutlass.Int32(0), cutlass.Int32(0))
    return cute.compile(RankTopK(num_org, local, topk), *args, make_fake_stream())


class DraftHead:
    """The draft head's rank half for ``rows`` predicted rows (7 per request)."""

    def __init__(
        self,
        lm_head,
        hidden_projection,
        *,
        vocab_start,
        rows,
        tp,
        rank,
        box=None,
        mc=0,
        device=None,
        topk=TOPK,
    ):
        num_org = lm_head.shape[0]
        if hidden_projection.shape != (SELECTOR_RANK, lm_head.shape[1]):
            raise ValueError(f"hidden_projection {tuple(hidden_projection.shape)}")
        weight = torch.cat([lm_head, hidden_projection]).to(torch.bfloat16).contiguous()
        self.device = device or weight.device
        self.num_org, self.rows, self.tp, self.rank, self.topk = (
            num_org,
            rows,
            tp,
            rank,
            topk,
        )
        tactic = draft_tactic("head", self.device)  # tactics.py; K31DRAFT_HEAD_TACTIC
        self.projection = PreparedProjection(weight, {rows: tactic}).prepare()
        self.head_out = torch.empty(
            rows, weight.shape[0], dtype=torch.bfloat16, device=self.device
        )
        self.local = box is None
        if box is None:  # one-GPU checks: this rank's rows land in a plain tensor
            box = torch.full(
                (tp, rows, 2 * topk), -0x80000000, dtype=torch.int32, device=self.device
            )
        self.box, self.mc = box, mc
        if box.shape != (tp, rows, 2 * topk):
            raise ValueError(f"head box {tuple(box.shape)} for top_k {topk}")
        self.code = compile_rank_topk(
            rows, weight.shape[0], tp, num_org, self.local, topk
        )
        self.args = (to_cute(self.box), cutlass.Int64(mc), cutlass.Int32(rank))
        self.vocab_start = cutlass.Int32(vocab_start)

    def run(self, rows):
        self.projection.run(rows, self.head_out)
        self.code(
            to_cute(self.head_out), *self.args, self.vocab_start, stream(self.device)
        )
        return self.head_out
