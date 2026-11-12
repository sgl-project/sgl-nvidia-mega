# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the SGLang project
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# Options described as "ported" follow NVIDIA's Rubin (sm_107) specialized-decode kernels (the rubin package).
"""O-projection all-reduce consumer (``groups`` 128-thread groups poll disjoint
rank slots, group 0 sums in rank order into the AttnRes) and TP16 fused producer +
consumer. With ``twoshot`` the mailbox is the one-slot gather row of
``common.twoshot``: the threads owning this rank's columns first reduce their
receive slots into it (args 11-13: receive view, gather multicast address, rank)."""

import os

import cutlass
import cutlass.cute as cute
from cutlass import Float32, Int32, Int64, Uint32

from ..attn_res.kernel import arm_exchanges, rstd
from ..common.mailbox import REARM_FENCE, rearm, rearm_fence

# Stock's fused all-reduce + residual (all_reduce_push_res_kernel) adds the prefix
# to the FP32 rank sum and rounds once; K31MK_OPROJ_SINGLE_ROUND=1 does the same. Default off (base D rounds twice) until
# a compliance decision flips it. Read at import.
SINGLE_ROUND = os.environ.get("K31MK_OPROJ_SINGLE_ROUND", "0") == "1"
# EF1 (memory-model review: compliant, no extra fence): when xmoe is
# None (B8 verify) group 0 issues its re-arm stores + re-arm fence AFTER its output publish instead of before the cluster exchange,
# so the fence leaves group 0's critical path. Same instructions, scopes, proxies and slots; still before exit (the next grids'
# griddepcontrol.wait still order it before this rank's next publish). Default off; K31MK_OPROJ_EF1=1 enables. Read when a kernel
# is compiled (a test may set ``EF1`` before it compiles, like ``mailbox.REARM_FENCE``).
EF1 = os.environ.get("K31MK_OPROJ_EF1", "0") == "1"
# CDR (K31MK_OPROJ_DEFER_REARM, default unset; see prepared.oproj_defer_batches): the split consumer re-arms
# the previous layer's slots (other parity, ``defer`` elements away) + one re-arm fence before its own poll, and leaves its own
# slots for the next layer; ``keep_own`` (last o-proj layer) also re-arms its own.
# NMERGE (K31MK_OPROJ_NMERGE, default unset; see prepared.nmerge_batches; NUMERICS-CHANGING, not bitwise): the
# groups-1 split consumer (two-shot rows) drops its third cluster exchange (the output RMS norm). Its post exchange also carries
# M = sum mix^2 (``qpart``) and X = sum mix * prefix, and the norm is the quadratic identity alpha^2 M + 2 alpha ep X + ep^2 P
# (P = sum prefix^2, the post total it already has) instead of sum final^2 over the materialized FP32 vector (stock's graph).
# Prefix and the block / prefix logits are unchanged (same reductions, same order); only the output scale's FP32 rounding moves.
# Form 2 (NM4, table value 2): the same four post values travel in ONE reduce_scatter of 4 ([P, W, M, X]) instead of the served
# reduce_scatter of 2 plus a second one of 2 after the first pushes; the butterfly partners are the same (16, 8, 4, 2, 1) for
# every value, so P / W equal the served sums and M / X equal form 1's bit for bit (output == form 1's output, bitwise).
from ..common.primitives import (
    add_rn,
    alloc,
    asm,
    at,
    bits_float,
    copy_g2s,
    float_bits,
    fma,
    launch,
    load_vector,
    named_barrier,
    pack_pair,
    publish_hint,
    published,
    push_float,
    smem_words,
    store_relaxed,
    store_shared_words,
    store_words,
    unpack_bf16,
    view,
    wait_deadline_check,
    wait_deadline_start,
)
from ..common.primitives import wait_phase_zero_cluster as wait_phase_zero
from ..common.primitives import (
    warp_sum,
    zeros,
)
from ..common.twoshot import reduce
from .producer import OprojProducer, pack8, tree_reduce
from . import mfx_hhome as HH  # H-HOME forms (default off)


@asm((Float32,) * 2, pure=True)
def pair_op(a, b, c=(), *, op="add"):
    values = (*a, *b, *c)
    code = "{ .reg .b64 a,b,c,d; mov.b64 a,{$2,$3}; mov.b64 b,{$4,$5}; "
    if op == "fma":
        code += "mov.b64 c,{$6,$7}; fma.rn.f32x2 d,a,b,c; "
    else:
        code += f"{op}.rn.f32x2 d,a,b; "
    constraints = ",".join(["=f", "=f"] + ["f"] * len(values))
    return code + "mov.b64 {$0,$1},d; }", constraints, tuple(map(Float32, values))


@asm(Float32, pure=True)
def mul_rn(a, b):
    return "mul.rn.f32 $0,$1,$2;", "=f,f,f", (*(Float32(v) for v in (a, b)),)


def two(values, j):
    return values[2 * j], values[2 * j + 1]


@cute.jit
def dot8(a, b):
    s0 = pair_op(two(a, 0), two(b, 0), op="mul")
    s1 = pair_op(two(a, 1), two(b, 1), op="mul")
    s0 = pair_op(two(a, 2), two(b, 2), s0, op="fma")
    s1 = pair_op(two(a, 3), two(b, 3), s1, op="fma")
    s = pair_op(s0, s1)
    return add_rn(s[0], s[1])


@cute.jit
def reduce_scatter(values, size: cutlass.Constexpr[int], lane):
    """Warp-reduce ``size`` values; returns the index of the one this lane holds."""
    for shift in cutlass.range_constexpr(4, -1, -1):
        live: cutlass.Constexpr = max(1, size >> (4 - shift))
        half: cutlass.Constexpr = max(1, live // 2)
        if cutlass.const_expr(live == 1):
            values[0] = add_rn(
                values[0], cute.arch.shuffle_sync_bfly(values[0], 1 << shift)
            )
        else:
            upper = (lane & (1 << shift)) != 0
            for j in cutlass.range_constexpr(half):
                send, keep = values[j + half], values[j]
                if upper:
                    send, keep = values[j], values[j + half]
                values[j] = add_rn(keep, cute.arch.shuffle_sync_bfly(send, 1 << shift))
    idx = cutlass.Int32(0)
    for shift in cutlass.range_constexpr(4, -1, -1):
        width: cutlass.Constexpr = size >> (4 - shift)
        if cutlass.const_expr(width > 1):  # noqa: SIM102
            if (lane & (1 << shift)) != 0:
                idx += width // 2
    return idx


@cute.jit
def fetch(raw, rank, pointer):
    words = load_vector(pointer, cache="volatile")
    for j in cutlass.range_constexpr(4):
        raw[rank, j] = words[j]


class OprojConsumer:
    # HH-G (``K31MK_OPROJ_HHOME_G``, default off; set per instance by ``oproj.prepared.compile_oproj``): the
    # two-shot gather (args[0]) is re-laid out at entry by ``mfx_hhome.homed`` (rule bit k): same (token, column) words, homed
    # on the polling cluster's die; -1 = the served layout (no code).
    hhome = -1
    hhome_r = 0  # HH-R: 1 = this (consumer-reduced) two-shot consumer's owner reduce polls the homed receive layout (args[11])
    hhdyn = False  # HH-G dynamic (rule d<k>): clusters take die-matched tokens (mfx_hhome.cluster_token); args 14 / 15 = ticket, SM table
    # HH-M (K31MK_OPROJ_HHOME_M): this groups-4 one-shot consumer polls / re-arms the homed one-shot mailbox (args[0], re-laid out
    # by mfx_hhome.homed_mailbox); its cluster takes a token of its own die's segment by ticket ("d0": bit 0; args 11 / 12 = ticket,
    # SM table) or the static permutation of batch ``hhm_b`` ("s0": mfx_hhome.static_token, no extra args). "" = served code.
    hhm = ""
    hhm_b = 0
    # F-L (K31MK_MOE_FRONT_LAMX; set per instance by ``oproj.prepared.compile_oproj``; default False = served
    # code): ``hints`` (args[8], unused without x_moe) is this layer's [LAMX_ROWS, 7168] parity view of the MoE front's Lamport
    # x box; group 0 stores its x words there (relaxed gpu scope, packed with NaN halves canonical, so never the 0xFFFFFFFF
    # sentinel and bitwise x otherwise) before its output store. No fence and no re-arm here: the front re-arms the other parity.
    lamx = False
    # LEANP (``K31MK_OPROJ_LEANP``, default off; set per instance by ``oproj.prepared.compile_oproj``): a groups-1
    # (two-shot) consumer allocates no ``parts`` buffer (only groups > 1 use it: 4 KB per CTA), so at nb 8 eight CTAs fit an SM's
    # 228 KB instead of seven (1024-thread limit = 8 x 128). Same instructions on the same data: bitwise identical.
    leanp = False

    def __init__(
        self,
        nb,
        *,
        has_delta=True,
        groups=4,
        twoshot=False,
        deadline_ns=0,
        batched=False,
        reduced=False,
        vote=False,
        defer=0,
        keep_own=True,
        nmerge=False,
    ):
        if not 0 <= nb <= 8:
            raise ValueError("Output-projection AttnRes supports zero to eight blocks")
        self.nb, self.has_delta = nb, has_delta
        self.groups, self.twoshot = groups, twoshot
        self.deadline_ns = deadline_ns
        # Package (b) options (from the source's ``oproj/consumer.py``; knobs in
        # ``prepared.opr_options``; each bitwise identical, default off):
        # ``batched`` (two-shot, TWOSHOT_BATCHED): the owners poll their receive slots with
        # every load in flight (``common.twoshot.poll_slots``); same sums.
        # ``reduced`` (two-shot, PRODUCER_REDUCE): the producer's CTAs reduce the owned
        # columns and multicast the gather (``OprojUmma(reduce=True)``, with the prefix when
        # SINGLE_ROUND adds it); this kernel only polls the gather.
        # ``vote`` (OPROJ_VOTE): ``poll_sum`` reloads every rank slot each pass and leaves
        # the loop on a warp vote instead of per-lane loops over the pending slots.
        self.batched, self.reduced, self.vote = batched, reduced, vote
        # CDR (K31MK_OPROJ_DEFER_REARM): element offset from this layer's parity view to the previous layer's (0 = served code);
        # ``keep_own``: the last o-proj layer also re-arms its own slots (forward boundary: every slot armed).
        self.defer, self.keep_own = int(defer), bool(keep_own)
        if self.defer and twoshot and not reduced:
            raise ValueError("CDR: one-shot or producer-reduced two-shot consumers only")
        if reduced and not twoshot:
            raise ValueError("reduced needs the two-shot consumer")
        self.kpre = 1 << max(0, (2 * nb - 1).bit_length())
        self.npost = 2 if nb > 0 else 1
        # NMERGE: two more post values (M, X) and no norm exchange; nb 0 has no norm exchange (no-op there)
        self.nmerge = bool(nmerge) and nb > 0
        if self.nmerge and groups != 1 and int(nmerge) != 2:  # form 2 also runs in the groups-4 one-shot consumer (rows 9-16)
            raise ValueError("NMERGE form 1: groups-1 (two-shot) split consumer only")
        self.npx = 2 if self.nmerge else 0
        self.nm4 = self.nmerge and int(nmerge) == 2  # NMERGE form 2: [P, W, M, X] in one reduce_scatter of 4

    @cute.jit
    def wait_exchange(self, barrier):
        wait_phase_zero(barrier)  # peers complete it by st.async

    @cute.jit
    def allocate(self):
        smem = cutlass.utils.SmemAllocator()
        wpre = alloc(smem, Float32, (4, 16))
        pre = alloc(smem, Float32, (7, 16))
        blk = alloc(smem, Float32, 16)
        post = alloc(smem, Float32, (7, 4, 4))
        total = alloc(smem, Float32, 4)
        bars = alloc(smem, cutlass.Uint64, 3, 8)
        lean = self.groups == 1  # above B8: rows for this nb only (co-residency)
        if cutlass.const_expr(self.leanp and lean):  # LEANP: no parts buffer; pf follows bars
            pf = alloc(smem, Uint32, (4 + self.nb, 128, 4))
            parts = view(pf.iterator, 0, Float32, (1, 2, 128, 4))  # group > 0 code only: never runs at groups 1
        else:
            parts = alloc(smem, Float32, (max(1, self.groups - 1), 2, 128, 4))
            pf = alloc(smem, Uint32, (4 + self.nb if lean else 12, 128, 4))
        norm_post = alloc(smem, Float32, (7, 4))
        norm_total = alloc(smem, Float32, 1)
        if cutlass.const_expr(self.hhdyn or self.hhm == "d0"):  # HH-G dynamic / HH-M d0: the cluster's token (rank 0's copy, read by all)
            return wpre, pre, blk, post, total, bars, parts, pf, norm_post, norm_total, alloc(smem, Int32, 1)
        return wpre, pre, blk, post, total, bars, parts, pf, norm_post, norm_total

    @cute.jit
    def setup(self, bars):
        arm_exchanges(bars, (2 * self.nb, 4 * (self.npost + self.npx), 4 if self.nb > 0 and not self.nmerge else 0))
        cute.arch.cluster_arrive_relaxed()

    @cute.jit
    def prefetch(self, args, pf, rank, token):
        _, prefix, blocks, norm, query, out_norm = args[:6]
        tid, _, _ = cute.arch.thread_idx()
        h = (rank * 128 + tid) * 8
        if cutlass.const_expr(self.nb > 0):
            copy_g2s(pf.iterator + tid * 4, norm.iterator + h)
            copy_g2s(pf.iterator + 512 + tid * 4, query.iterator + h)
        copy_g2s(pf.iterator + 1024 + tid * 4, out_norm.iterator + h)
        if cutlass.const_expr(self.has_delta):
            copy_g2s(pf.iterator + 1536 + tid * 4, at(prefix, token, h))
        for block in cutlass.range_constexpr(self.nb):
            dst = pf.iterator + (4 + block) * 512 + tid * 4
            copy_g2s(dst, at(blocks, token, block, h))
        cute.arch.cp_async_commit_group()

    @cute.jit
    def fragment(self, pf, row, tid):
        words = smem_words(pf.iterator + (row * 128 + tid) * 4)
        values = cute.make_rmem_tensor(8, Float32)
        for j in cutlass.range_constexpr(4):
            values[2 * j], values[2 * j + 1] = unpack_bf16(words[j])
        return values

    @cute.jit
    def combined_weight(self, pf, tid):
        a, b = self.fragment(pf, 0, tid), self.fragment(pf, 1, tid)
        result = cute.make_rmem_tensor(8, Float32)
        for j in cutlass.range_constexpr(4):
            product = pair_op(two(a, j), two(b, j), op="mul")
            result[2 * j], result[2 * j + 1] = unpack_bf16(pack_pair(*product))
        return result

    @cute.jit
    def pre(self, state, rank, eps, mix, wait_groups: cutlass.Constexpr[int] = 0):
        wpre, pre, blk, _, _, bars, _, pf = state[:8]
        tid, _, _ = cute.arch.thread_idx()
        warp, lane = tid // 32, tid % 32
        maximum, exponent_sum, qpart = Float32(0), Float32(0), Float32(0)
        cute.arch.cp_async_wait_group(wait_groups)
        if cutlass.const_expr(self.nb > 0):
            weight = self.combined_weight(pf, tid)
            partial = zeros(self.kpre, Float32)
            for block in cutlass.range_constexpr(self.nb):
                value = self.fragment(pf, 4 + block, tid)
                partial[block] = dot8(value, value)
                partial[self.nb + block] = dot8(value, weight)
            idx = reduce_scatter(partial, self.kpre, lane)
            if ((lane & (32 // self.kpre - 1)) == 0) & (idx < 2 * self.nb):
                wpre[warp, idx] = partial[0]
            named_barrier()
            cute.arch.cluster_wait()
            if tid < 2 * self.nb:
                pre_sum = Float32(0)
                for w in cutlass.range_constexpr(4):
                    pre_sum = add_rn(pre_sum, wpre[w, tid])
                for peer in cutlass.range_constexpr(7):
                    push_float(at(pre, rank, tid), pre_sum, bars.iterator, peer)
            self.wait_exchange(bars.iterator)
            if tid < 2 * self.nb:
                sums = [Float32(0), Float32(0)]
                for peer in cutlass.range_constexpr(7):
                    sums[peer % 2] = add_rn(sums[peer % 2], pre[peer, tid])
                blk[tid] = add_rn(*sums)
            named_barrier()
            logits = [
                mul_rn(blk[self.nb + b], rstd(blk[b], eps)) for b in range(self.nb)
            ]
            maximum = tree_reduce(logits, max)
            exp = [cute.math.exp(x - maximum, fastmath=True) for x in logits]
            exponent_sum = tree_reduce(exp, add_rn)
            for block in cutlass.range_constexpr(self.nb):
                value = self.fragment(pf, 4 + block, tid)
                for j in cutlass.range_constexpr(4):
                    mix[2 * j], mix[2 * j + 1] = pair_op(
                        (exp[block], exp[block]), two(value, j), two(mix, j), op="fma"
                    )
            qpart = dot8(mix, mix)
        else:
            cute.arch.cluster_wait()
        return maximum, exponent_sum, qpart

    def slots(self, tp):
        """Rank slots each group polls (at least four)."""
        return max(4, (tp + self.groups - 1) // self.groups)

    @cute.jit
    def poll_sum(self, mailbox, token, h, first, count):
        n: cutlass.Constexpr = self.slots(mailbox.shape[0])
        raw = zeros((n, 4), Uint32)
        pending = Int32(0)
        for rank in cutlass.range_constexpr(n):
            if rank < count:
                fetch(raw, rank, at(mailbox, first + rank, token, h))
        if cutlass.const_expr(self.vote):
            ready = True
            for rank in cutlass.range_constexpr(n):
                if rank < count:
                    ready &= published(raw[rank, j] for j in range(4))
            started, polls = Int64(0), Int32(0)
            if not cute.arch.vote_all_sync(ready):
                started = wait_deadline_start(self.deadline_ns)
            while not cute.arch.vote_all_sync(ready):
                polls = wait_deadline_check(started, polls, self.deadline_ns)
                for rank in cutlass.range_constexpr(n):
                    if rank < count:
                        fetch(raw, rank, at(mailbox, first + rank, token, h))
                ready = True
                for rank in cutlass.range_constexpr(n):
                    if rank < count:
                        ready &= published(raw[rank, j] for j in range(4))
        else:
            for rank in cutlass.range_constexpr(n):
                if rank < count:  # noqa: SIM102
                    if not published(raw[rank, j] for j in range(4)):
                        pending |= 1 << rank
            started, polls = Int64(0), Int32(0)
            if pending != 0:
                started = wait_deadline_start(self.deadline_ns)
            while pending != 0:
                polls = wait_deadline_check(started, polls, self.deadline_ns)
                for rank in cutlass.range_constexpr(n):
                    if (pending & (1 << rank)) != 0:
                        fetch(raw, rank, at(mailbox, first + rank, token, h))
                for rank in cutlass.range_constexpr(n):
                    if (pending & (1 << rank)) != 0:  # noqa: SIM102
                        if published(raw[rank, j] for j in range(4)):
                            pending &= ~(1 << rank)
        values = cute.make_rmem_tensor(8, Float32)
        for j in cutlass.range_constexpr(4):
            pair = unpack_bf16(raw[0, j])
            for rank in cutlass.range_constexpr(1, n):
                if rank < count:
                    pair = pair_op(pair, unpack_bf16(raw[rank, j]))
            values[2 * j], values[2 * j + 1] = pair[0], pair[1]
        return values

    @cute.jit
    def publish(self, output, xmoe, hints, token, h, values):
        tid, _, _ = cute.arch.thread_idx()
        if cutlass.const_expr(self.lamx):  # F-L: the front polls these words (before the output store: earlier landing)
            store_words(at(hints, token, h), pack8(values, "stage"), scope="gpu")
        words = pack8(values)
        store_words(at(output, token, h), words)
        if cutlass.const_expr(xmoe is not None):
            words = pack8(values, "latent")
            for rep in cutlass.range_constexpr(xmoe.shape[0]):
                store_relaxed(at(xmoe, rep, token, h), words, scope="gpu")
            cute.arch.sync_warp()
            if tid % 32 == 0:
                publish_hint(hints.iterator)

    @cute.jit
    def post(self, args, state, block_mix, rank, token):
        mailbox, prefix, _, _, _, _, output, xmoe, hints, eps, out_eps = args[:11]
        _, _, _, post, total, bars, parts, pf = state[:8]
        norm_post, norm_total = state[8:10]
        mix, maximum, exponent_sum, qpart = block_mix
        thread, _, _ = cute.arch.thread_idx()
        group, tid = thread // 128, thread % 128
        warp, lane = tid // 32, tid % 32
        h = (rank * 128 + tid) * 8
        S: cutlass.Constexpr = (mailbox.shape[0] + self.groups - 1) // self.groups
        everyone: cutlass.Constexpr = 128 * self.groups
        if group > 0:
            first = group * S
            count = min(S, mailbox.shape[0] - first)
            if cutlass.const_expr(self.defer != 0):  # CDR: the previous layer's slots, under the hop wait
                for r in cutlass.range_constexpr(self.slots(mailbox.shape[0])):
                    if r < count:
                        rearm(at(mailbox, first + r, token, h) + self.defer)
                rearm_fence()
            values = self.poll_sum(mailbox, token, h, first, count)
            for half in cutlass.range_constexpr(2):
                words = tuple(float_bits(values[half * 4 + i]) for i in range(4))
                store_shared_words(at(parts, group - 1, half, tid, 0), words)
            named_barrier(2, everyone, arrive=True)
            if cutlass.const_expr(self.defer == 0 or self.keep_own):  # CDR: else the next layer re-arms these at its entry
                for r in cutlass.range_constexpr(self.slots(mailbox.shape[0])):
                    if r < count:
                        rearm(at(mailbox, first + r, token, h))
                rearm_fence()
        else:
            if cutlass.const_expr(self.has_delta and SINGLE_ROUND):
                old = self.fragment(pf, 3, tid)
            if cutlass.const_expr(self.twoshot and not self.reduced):
                recv, gather_mc, me = args[11:14]
                if h // recv.shape[2] == me:
                    # The owner adds the prefix (identical on every rank) to its FP32 sum
                    # before its one rounding: the gather row is then the new prefix.
                    if cutlass.const_expr(self.has_delta and SINGLE_ROUND):
                        reduce(
                            recv,
                            mailbox,
                            gather_mc,
                            me,
                            token,
                            h,
                            self.deadline_ns,
                            old,
                            self.batched,
                        )
                    else:
                        reduce(
                            recv,
                            mailbox,
                            gather_mc,
                            me,
                            token,
                            h,
                            self.deadline_ns,
                            None,
                            self.batched,
                        )
            if cutlass.const_expr(self.has_delta and not SINGLE_ROUND):
                old = self.fragment(pf, 3, tid)
            if cutlass.const_expr(self.defer != 0):  # CDR: the previous layer's slots, under the hop wait
                for r in cutlass.range_constexpr(min(S, mailbox.shape[0])):
                    rearm(at(mailbox, r, token, h) + self.defer)
                rearm_fence()
            attn = self.poll_sum(mailbox, token, h, 0, min(S, mailbox.shape[0]))
            if cutlass.const_expr(self.groups > 1):
                named_barrier(2, everyone)
            for other in cutlass.range_constexpr(self.groups - 1):
                for half in cutlass.range_constexpr(2):
                    words = smem_words(at(parts, other, half, tid, 0))
                    for j in cutlass.range_constexpr(2):
                        k: cutlass.Constexpr = half * 2 + j
                        part = tuple(map(bits_float, two(words, j)))
                        attn[2 * k], attn[2 * k + 1] = pair_op(two(attn, k), part)
            # Stock's fused all-reduce (all_reduce_push_res_kernel) adds the prefix to the
            # FP32 sum and rounds once; so does this (the two-shot gather row already
            # holds bf16(sum + prefix), see above).
            prefix_value = cute.make_rmem_tensor(8, Float32)
            for j in cutlass.range_constexpr(4):
                if cutlass.const_expr(SINGLE_ROUND):
                    acc = two(attn, j)
                    if cutlass.const_expr(self.has_delta and not self.twoshot):
                        acc = pair_op(two(old, j), acc)
                    rounded = unpack_bf16(pack_pair(acc[0], acc[1]))
                else:  # base D: bf16(old + bf16(sum)); the two-shot owner adds no prefix
                    rounded = unpack_bf16(pack_pair(*two(attn, j)))
                    if cutlass.const_expr(self.has_delta):
                        rounded = pair_op(two(old, j), rounded)
                        rounded = unpack_bf16(pack_pair(rounded[0], rounded[1]))
                prefix_value[2 * j], prefix_value[2 * j + 1] = rounded[0], rounded[1]
            nv: cutlass.Constexpr = 4 if self.nm4 else self.npost  # NMERGE form 2: post slots [P, W, M, X] from one reduce_scatter
            partial = cute.make_rmem_tensor(nv, Float32)
            if cutlass.const_expr(self.nb > 0):
                weight = self.combined_weight(pf, tid)
                partial[0] = dot8(prefix_value, prefix_value)
                partial[1] = dot8(prefix_value, weight)
                if cutlass.const_expr(self.nm4):
                    partial[2] = qpart
                    partial[3] = dot8(mix, prefix_value)
            else:
                partial[0] = dot8(prefix_value, prefix_value)
            idx = reduce_scatter(partial, nv, lane)
            if idx < nv:
                for peer in range(lane & (32 // nv - 1), 7, 32 // nv):
                    dst = at(post, rank, warp, idx)
                    push_float(dst, partial[0], bars.iterator + 1, peer)
            if cutlass.const_expr(self.nmerge and not self.nm4):  # NMERGE: M = sum mix^2, X = sum mix * prefix, post slots 2 / 3
                extra = cute.make_rmem_tensor(2, Float32)
                extra[0] = qpart
                extra[1] = dot8(mix, prefix_value)
                idx2 = reduce_scatter(extra, 2, lane)
                for peer in range(lane & 15, 7, 16):
                    push_float(at(post, rank, warp, 2 + idx2), extra[0], bars.iterator + 1, peer)
            words = pack8(prefix_value)
            store_words(at(prefix, token, h), words)
            if cutlass.const_expr(not (EF1 and xmoe is None) and (self.defer == 0 or self.keep_own)):  # CDR: deferred
                for r in cutlass.range_constexpr(min(S, mailbox.shape[0])):
                    rearm(at(mailbox, r, token, h))
                if cutlass.const_expr(xmoe is not None or bool(REARM_FENCE)):
                    rearm_fence(xmoe is not None)
            self.wait_exchange(bars.iterator + 1)
            if tid < self.npost + self.npx:
                sums = [Float32(0)] * 4
                for peer in cutlass.range_constexpr(7):
                    for w in cutlass.range_constexpr(4):
                        sums[w] = add_rn(sums[w], post[peer, w, tid])
                total[tid] = tree_reduce(sums, add_rn)
            named_barrier()
            norm = self.fragment(pf, 2, tid)
            final = cute.make_rmem_tensor(8, Float32)
            if cutlass.const_expr(self.nb > 0):
                lp = mul_rn(total[1], rstd(total[0], eps))
                mx = max(maximum, lp)
                alpha = cute.math.exp(maximum - mx, fastmath=True)
                ep = cute.math.exp(lp - mx, fastmath=True)
                den = fma(alpha, exponent_sum, ep)
                # Materialize the FP32 mixed vector before squaring it. A
                # quadratic identity is not the stock floating-point graph.
                for j in cutlass.range_constexpr(4):
                    p = pair_op((ep, ep), two(prefix_value, j), op="mul")
                    value = pair_op((alpha, alpha), two(mix, j), p, op="fma")
                    final[2 * j], final[2 * j + 1] = value[0], value[1]
                if cutlass.const_expr(self.nmerge):  # NMERGE: alpha^2 M + 2 alpha ep X + ep^2 P, no third exchange
                    ae = mul_rn(alpha, ep)
                    quad = fma(mul_rn(alpha, alpha), total[2], fma(add_rn(ae, ae), total[3], mul_rn(mul_rn(ep, ep), total[0])))
                    inverse = cute.arch.rcp_approx(den)
                    scale = mul_rn(inverse, rstd(mul_rn(mul_rn(quad, inverse), inverse), out_eps))
                else:
                    square = warp_sum(dot8(final, final))
                    if lane == 0:
                        for peer in cutlass.range_constexpr(7):
                            push_float(
                                at(norm_post, rank, warp), square, bars.iterator + 2, peer
                            )
                    self.wait_exchange(bars.iterator + 2)
                    if tid == 0:
                        sums = [Float32(0)] * 4
                        for peer in cutlass.range_constexpr(7):
                            for w in cutlass.range_constexpr(4):
                                sums[w] = add_rn(sums[w], norm_post[peer, w])
                        norm_total[0] = tree_reduce(sums, add_rn)
                    named_barrier()
                    inverse = cute.arch.rcp_approx(den)
                    scale = mul_rn(
                        inverse,
                        rstd(mul_rn(mul_rn(norm_total[0], inverse), inverse), out_eps),
                    )
            else:
                scale = rstd(total[0], out_eps)
                for i in cutlass.range_constexpr(8):
                    final[i] = prefix_value[i]
            for j in cutlass.range_constexpr(4):
                value = pair_op((scale, scale), two(final, j), op="mul")
                value = pair_op(value, two(norm, j), op="mul")
                final[2 * j], final[2 * j + 1] = value[0], value[1]
            self.publish(output, xmoe, hints, token, h, final)
            if cutlass.const_expr(EF1 and xmoe is None and (self.defer == 0 or self.keep_own)):  # EF1 (CDR: deferred)
                for r in cutlass.range_constexpr(min(S, mailbox.shape[0])):
                    rearm(at(mailbox, r, token, h))
                if cutlass.const_expr(bool(REARM_FENCE)):
                    rearm_fence()

    @cute.jit
    def __call__(self, *args):
        launch(self.kernel, args, (7, args[1].shape[0], 1), 128 * self.groups, 7)

    @cute.kernel
    def kernel(self, args):
        if cutlass.const_expr(self.hhome >= 0):  # HH-G
            args = (HH.homed(args[0], self.hhome), *args[1:])
        if cutlass.const_expr(self.hhome_r > 0):  # HH-R
            args = (*args[:11], HH.homed_recv(args[11]), *args[12:])
        if cutlass.const_expr(self.hhm):  # HH-M: the homed one-shot mailbox
            args = (HH.homed_mailbox(args[0]), *args[1:])
        rank, token, _ = cute.arch.block_idx()
        tid, _, _ = cute.arch.thread_idx()
        state = self.allocate()
        if cutlass.const_expr(self.hhdyn):  # HH-G dynamic: this cluster's token (one ticket per cluster, then a cluster barrier)
            token = HH.cluster_token(args[14], args[15], state[10], rank, self.hhome)
        if cutlass.const_expr(self.hhm == "d0"):  # HH-M d0: a token of this cluster's die's segment (bit 0), same ticket form as HH-G d0
            token = HH.cluster_token(args[11], args[12], state[10], rank, 0)
        if cutlass.const_expr(self.hhm == "s0"):  # HH-M s0: the static permutation of this batch (no ticket, no extra barrier)
            token = HH.static_token(token, self.hhm_b)
        self.setup(state[5])
        cute.arch.griddepcontrol_launch_dependents()
        mix = zeros(8, Float32)
        maximum, exponent_sum, qpart = Float32(0), Float32(0), Float32(0)
        if tid < 128:
            self.prefetch(args, state[7], rank, token)
            maximum, exponent_sum, qpart = self.pre(state, rank, args[9], mix)
        self.post(args, state, (mix, maximum, exponent_sum, qpart), rank, token)
        cute.arch.griddepcontrol_wait()


class OprojFused:
    """``x_shared`` and ``drain`` are the source's Rubin options of this kernel (
    ``oproj/consumer.py`` ``OprojFused``), ported here; both are bitwise identical.
    ``x_shared``: the producer copies x to shared memory once per CTA instead of every warp
    reading its K slice from L2. ``drain``: a GPU-scope fence after the multicast publish,
    off the critical path (a grid whose CTAs issued NVLink stores and never fenced them
    releases its PDL dependents later): the producer-only CTAs fence right after
    publishing; in the consumer CTAs the publishing threads arrive on a named barrier and
    the last warp syncs on it and fences once its poll group is done."""

    DRAIN_BARRIER = 5  # free in this kernel: the consumer uses 1 and 2

    def __init__(self, nb, *, has_delta=True, deadline_ns=0, x_shared=False, drain=False, vote=False):
        self.producer = OprojProducer(768, staged_blocks=0, x_shared=x_shared)
        # TP16 requires prefix/blocks to be ready at launch. The poll fails stop at the build deadline like the
        # split consumer (it was unbounded). ``vote``: OPROJ_VOTE of the consumer's poll (package (b)).
        self.consumer = OprojConsumer(nb, has_delta=has_delta, deadline_ns=deadline_ns, vote=vote)
        self.drain = drain

    @cute.jit
    def __call__(self, *args):
        if cutlass.const_expr(args[0].shape[0] > 8):
            raise ValueError("The selected fused schedule supports B1..8")
        if cutlass.const_expr(self.drain and args[0].shape[0] % 4 != 0):
            # The drain barrier counts 8 publishing threads per token plus one warp.
            raise ValueError("The fused drain supports B4 and B8")
        launch(self.kernel, args, (8, 14, 1), 512, 8, min_blocks_per_mp=1)

    @cute.kernel
    def kernel(self, args):
        x, weight, rows = args[0], args[1], (args[2], *args[5:])
        rank, cluster, _ = cute.arch.block_idx()
        tid, _, _ = cute.arch.thread_idx()
        cta = cluster * 8 + rank
        active = (cluster < x.shape[0]) & (rank < 7)
        state = self.consumer.allocate()
        smem = self.producer.allocate(x.shape[0])
        self.consumer.setup(state[5])
        cute.arch.griddepcontrol_launch_dependents()
        group0 = active & (tid < 128)
        if group0:
            self.consumer.prefetch(rows, state[7], rank, cluster)
        weights = self.producer.load_weight(weight, smem[2], cta)
        mix = zeros(8, Float32)
        maximum, exponent_sum, qpart = Float32(0), Float32(0), Float32(0)
        if group0:
            maximum, exponent_sum, qpart = self.consumer.pre(state, rank, rows[9], mix)
        self.producer.compute(args, smem, weights, cta, False)
        # The last publish's threads (``OprojProducer.publish``) and the drain warp.
        _, gt, ng = self.producer.schedule(x.shape[0])
        publishers: cutlass.Constexpr = 8 * min(x.shape[0] - (ng - 1) * gt * 8, gt * 8)
        drainers: cutlass.Constexpr = publishers + 32
        if cutlass.const_expr(self.drain):  # noqa: SIM102 (compile-time)
            if tid < publishers:
                if active:
                    named_barrier(self.DRAIN_BARRIER, drainers, arrive=True)
                else:
                    cute.arch.fence_acq_rel_gpu()
        if active:
            block_mix = (mix, maximum, exponent_sum, qpart)
            self.consumer.post(rows, state, block_mix, rank, cluster)
            if cutlass.const_expr(self.drain):  # noqa: SIM102 (compile-time)
                if tid >= 512 - 32:  # the last poll group's last warp
                    named_barrier(self.DRAIN_BARRIER, drainers)
                    cute.arch.fence_acq_rel_gpu()
