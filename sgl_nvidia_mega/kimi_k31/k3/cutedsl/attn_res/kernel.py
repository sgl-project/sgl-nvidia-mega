# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the SGLang project
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Seven-CTA cluster AttnRes: one cluster per token, 1024 hidden columns per CTA.

The delta is a tensor (read after the PDL wait) or ``slots`` Lamport mailbox slots
summed in slot order, rounded to BF16 and re-armed. Output RMS statistics use
the FP32 mixed vector, including at the final boundary. Prefix and saved blocks must be complete at launch.

``aux_fold`` (with ``aux_mixture``) computes the stock FP32-cw aux mixture inside this kernel
instead of launching ``AuxMixture``'s score and combine kernels after it. The products, the
summation trees and the special-function instructions are those passes' own: score chunk ``c``
(columns 1024c..1024c+1023) is cluster CTA ``c``; score thread ``2t+b`` is this CTA's thread ``t``,
columns 4b..4b+3; its 32-lane butterfly is replayed on 16 lanes x 2 values, its 8-warp butterfly
and its chunk-order sum run on the exchanged sums; the softmax and the row-order FMA mixture are
the combine pass's code. Saved-block scores are done before the delta arrives; only the prefix
row's score and the mixture follow it.
"""

import cutlass
import cutlass.cute as cute
from cutlass import Float32, Int32, Uint32, Uint64

from ..common.mailbox import K23_ARM, poll, rearm, rearm_fence
from ..common.primitives import (
    add_rn,
    alloc,
    at,
    bf16,
    bf16x2,
    dot,
    fma,
    group_barrier,
    launch,
    load_words,
    push_float,
    scatter_reduce,
    store_words,
    wait,
    warp_sum,
    zeros,
)
from .aux import AuxMixture, aux_div, aux_exp, aux_mul, aux_rrms


@cute.jit
def round_pair(t, row: cutlass.Constexpr, j: cutlass.Constexpr):
    word = bf16x2(t[row, 2 * j], t[row, 2 * j + 1])
    t[row, 2 * j], t[row, 2 * j + 1] = bf16(word, 0), bf16(word, 1)


@cute.jit
def unpack(words, out, row: cutlass.Constexpr):
    for i in cutlass.range_constexpr(8):
        out[row, i] = bf16(words[i // 2], i % 2)


def block_ptr(blocks, token, block, col):
    """``&blocks[token, block, col]`` from the tensor's strides."""
    return blocks.iterator + token * blocks.stride[0] + block * blocks.stride[1] + col


@cute.jit
def rstd(sum_sq, eps):
    """Reciprocal RMS of a 7168-wide row from its sum of squares."""
    return cute.math.rsqrt(fma(sum_sq, Float32(1 / 7168), eps), fastmath=True)


@cute.jit
def dot8(a, ar: cutlass.Constexpr, b, br: cutlass.Constexpr):
    return dot((a[ar, i] for i in range(8)), (b[br, i] for i in range(8)), Float32(0))


@cute.jit
def arm_exchanges(bars, counts):
    """Thread 0 arms barrier ``r`` for ``counts[r]`` FP32 pushes from each of 7 CTAs (cluster scope: K-23)."""
    if cute.arch.thread_idx()[0] == 0:
        for r, count in enumerate(counts):
            if cutlass.const_expr(count > 0):
                cute.arch.mbarrier_init(bars.iterator + r, 1)
                cute.arch.mbarrier_arrive_and_expect_tx(bars.iterator + r, 28 * count, **K23_ARM)
        cute.arch.mbarrier_init_fence()


@cute.jit
def aux_cw(norm, query, col):
    """The aux pass's FP32 combined weight of this thread's 8 columns (unrounded product)."""
    nw = load_words(norm.iterator + col, count=4)
    qw = load_words(query.iterator + col, count=4)
    return [aux_mul(bf16(nw[i // 2], i % 2), bf16(qw[i // 2], i % 2)) for i in range(8)]


@cute.jit
def aux_tree(red, kind, row):
    """The score pass's 8-warp butterfly (lanes xor 4, 2, 1) over one chunk's warp sums."""
    r = [red[kind, row, w] for w in range(8)]
    left = add_rn(add_rn(r[0], r[4]), add_rn(r[2], r[6]))
    return add_rn(left, add_rn(add_rn(r[1], r[5]), add_rn(r[3], r[7])))


@cute.jit
def aux_chunks(buf, index):
    """Chunk-order sum (CTA 0..6, from +0.0) of value ``index`` pushed to ``buf[peer * 16 + index]``."""
    total = Float32(0)
    for peer in cutlass.range_constexpr(7):
        total = add_rn(total, buf[peer * 16 + index])
    return total


class ClusterAttnRes:
    def __init__(
        self,
        nb,
        *,
        slots=1,
        has_delta=True,
        write=-1,
        final=False,
        deadline_ns=0,
        aux_mixture=False,
        aux_fold=False,
        aux_overlap=False,
        late_rearm=False,
        one_round=False,
        rearm_prev=False,
        rearm_own=True,
    ):
        self.nb, self.slots, self.has_delta = nb, slots, has_delta
        self.write, self.final = write, final
        self.deadline_ns = deadline_ns
        self.aux_mixture = aux_mixture
        # The fold mixes the saved blocks read before the delta; the unfused combine reads them
        # after this kernel, so a block write inside rows 0..nb-1 keeps the unfused passes.
        self.aux_fold = bool(aux_fold and aux_mixture and nb > 0 and not 0 <= write < nb)
        self.naux = 2 if self.aux_fold else 0
        # The folded mixture runs between round 2's pushes and its wait (it needs only round 1's totals).
        self.aux_overlap = self.aux_fold and aux_overlap
        # ``late_rearm`` (K31MK_ATTNRES_LATE_REARM, "EF4"; bitwise): the ring's re-arm
        # stores and rearm_fence run after the output stores, still before the trailing
        # griddepcontrol.wait and exit, instead of between the poll and round 1 (the H4 analogue
        # of the o-projection's EF1; same edge class).
        self.late_rearm = bool(late_rearm and slots > 0)
        # ``one_round`` (K31MK_ATTNRES_ONE_ROUND, "Q"; REORDER, not bitwise): the
        # output RMS comes from FP32 terms (sum mix^2 exchanged before the delta, sum mix*new in
        # round 1) instead of a second post-delta exchange of the mixed vector's squares.
        self.one_round = bool(one_round and nb > 0 and not self.aux_fold)
        self.npre = 2 * nb
        self.npost = (3 if self.one_round else 2) if nb else 1
        self.kpre = 1 << max(self.npre - 1, 0).bit_length()
        self.kpost = 1 << max(self.npost - 1, 0).bit_length()
        # ``rearm_prev`` / ``rearm_own`` (K31MK_ATTNRES_DEFER_REARM, DRA; bitwise: no value changes): the MoE
        # tail publishes ordinal k into mailbox parity k % 2, and this consumer re-arms the PREVIOUS ordinal's parity
        # rows (its own token and columns, one more tensor argument after ``out_eps``) before its poll, then
        # rearm_fence, instead of re-arming its own rows (and fencing) after the output. ``rearm_own`` False skips
        # this consumer's own re-arm and fence: the next ordinal's consumer re-arms them. The last MoE ordinal keeps
        # ``rearm_own`` so both parities are armed at every forward's end (any batch next).
        self.rearm_prev = bool(rearm_prev and slots > 0)
        self.rearm_own = bool(rearm_own or slots == 0)

    @cute.jit
    def __call__(self, *args):
        launch(self.kernel, args, (7, args[1].shape[0], 1), 128, 7)
        if cutlass.const_expr(self.aux_mixture and self.nb > 0 and not self.aux_fold):
            if cutlass.const_expr(self.rearm_prev):
                AuxMixture(self.nb)(*args[:9], *args[10:])
            else:
                AuxMixture(self.nb)(*args)

    @cute.jit
    def exchange(
        self,
        values,
        count: cutlass.Constexpr,
        kpad: cutlass.Constexpr,
        bufs,
        r,
        ids,
        extra: cutlass.Constexpr = 0,
        aux_args=None,
    ):
        """Cluster-sum ``count`` per-thread partials into round ``r``'s total row. With
        ``extra``, threads ``count..count+extra-1`` also push the aux prefix row's chunk sums
        (kind ``tid - count``) to slots ``count..``, which callers sum in chunk order."""
        wb, peer_buf, total, bars = bufs[:4]
        peer_buf, total, bar = peer_buf[r, None], total[r, None], bars.iterator + r
        rank, _, tid = ids
        lane, warp = tid % 32, tid // 32
        reduced, index = scatter_reduce(values, lane, kpad)
        if (lane & (32 // kpad - 1)) == 0 and index < count:
            wb[warp * 16 + index] = reduced
        group_barrier(0)
        if tid < count:
            value = (wb[tid] + wb[16 + tid]) + (wb[32 + tid] + wb[48 + tid])
            for peer in cutlass.range_constexpr(7):
                push_float(peer_buf.iterator + rank * 16 + tid, value, bar, Int32(peer))
        if cutlass.const_expr(extra > 0):
            if tid >= count and tid < count + extra:
                value = aux_tree(bufs[4][0], tid - count, self.nb)
                for peer in cutlass.range_constexpr(7):
                    push_float(
                        peer_buf.iterator + rank * 16 + tid, value, bar, Int32(peer)
                    )
        if cutlass.const_expr(aux_args is not None):
            self.aux_mix(*aux_args)
        wait(bar)
        if tid < count:
            sums = [Float32(0), Float32(0)]
            for peer in cutlass.range_constexpr(7):
                sums[peer % 2] += peer_buf[peer * 16 + tid]
            total[tid] = sums[0] + sums[1]
        group_barrier(0)

    @cute.kernel
    def kernel(self, args):
        delta, prefix, blocks, norm, query, out_norm, output, eps, out_eps = args[0:9]
        rank, token, _ = cute.arch.block_idx()
        tid, _, _ = cute.arch.thread_idx()
        bufs = self.setup(cutlass.utils.SmemAllocator())
        cute.arch.cluster_arrive_relaxed()
        cute.arch.griddepcontrol_launch_dependents()
        state = self.state()
        rows, ids = (prefix, blocks, norm, query, out_norm), (rank, token, tid)
        self.pre(state, bufs, rows, eps, ids)
        if cutlass.const_expr(self.rearm_prev):
            self.post(state, bufs, delta, rows, output, eps, out_eps, ids, args[10:], args[9])
        else:
            self.post(state, bufs, delta, rows, output, eps, out_eps, ids, args[9:])
        cute.arch.griddepcontrol_wait()

    def state(self):
        """Carried from pre() to post(); the rest reloads so a GEMM can run between.
        The aux fold also carries the saved blocks' words for its mixture."""
        state = cute.make_rmem_tensor((1, 8), Float32), cute.make_rmem_tensor(2, Float32)
        if self.aux_fold:
            state += (cute.make_rmem_tensor((self.nb, 4), Uint32),)
        return state

    @cute.jit
    def setup(self, smem):
        warp_buf = alloc(smem, Float32, 64)
        peer_buf = alloc(smem, Float32, (3, 112))
        total = alloc(smem, Float32, (3, 16))
        bars = alloc(smem, Uint64, 4 if self.aux_fold else 3, 8)
        arm_exchanges(
            bars,
            (self.npre, self.npost + self.naux, int(self.nb > 0), self.nb * self.naux),
        )
        aux = ()
        if cutlass.const_expr(self.aux_fold):
            # Score-warp sums [sq/dot, row, score warp], round-3 chunk sums, block scores.
            aux = (
                alloc(smem, Float32, (2, self.nb + 1, 8)),
                alloc(smem, Float32, 112),
                alloc(smem, Float32, 16),
            )
        return warp_buf, peer_buf, total, bars, aux

    @cute.jit
    def aux_partials(self, v, vrow: cutlass.Constexpr, row: cutlass.Constexpr, cw, red, tid):
        """Score-warp sums of ``v[vrow]`` (sum of squares; dot with ``cw``), formed as the
        score pass forms them, into ``red[:, row, :]``."""
        lane, warp = tid % 32, tid // 32
        sq, dt = [], []
        for b in cutlass.range_constexpr(2):
            x = [v[vrow, 4 * b + j] for j in range(4)]
            c = [cw[4 * b + j] for j in range(4)]
            sq.append(
                add_rn(
                    fma(x[0], x[0], aux_mul(x[2], x[2])),
                    fma(x[1], x[1], aux_mul(x[3], x[3])),
                )
            )
            dt.append(
                add_rn(
                    fma(x[0], c[0], aux_mul(x[2], c[2])),
                    fma(x[1], c[1], aux_mul(x[3], c[3])),
                )
            )
        # Score lanes 2u+b: lane xor 16, 8, 4, 2 is cluster-lane xor 8, 4, 2, 1; xor 1 is local.
        for step in cutlass.range_constexpr(4):
            for b in cutlass.range_constexpr(2):
                sq[b] = add_rn(sq[b], cute.arch.shuffle_sync_bfly(sq[b], 8 >> step))
                dt[b] = add_rn(dt[b], cute.arch.shuffle_sync_bfly(dt[b], 8 >> step))
        s, d = add_rn(sq[0], sq[1]), add_rn(dt[0], dt[1])
        if lane % 16 == 0:
            red[0, row, 2 * warp + lane // 16] = s
            red[1, row, 2 * warp + lane // 16] = d

    @cute.jit
    def pre(self, state, bufs, rows, eps: Float32, ids):
        """Inputs complete at launch, then the saved-block statistics exchange."""
        prefix, blocks, norm, query, out_norm = rows
        rank, token, tid = ids
        warp_buf, peer_buf, total, bars = bufs[:4]
        mixture, scal = state[0], state[1]
        # Rows: saved blocks, combined weight, gamma, old prefix, new prefix.
        v = cute.make_rmem_tensor((self.nb + 4, 8), Float32)
        col = (rank * 128 + tid) * 8
        W = self.nb
        for source in cutlass.range_constexpr(self.nb):
            words = load_words(
                block_ptr(blocks, token, source, col), count=4, mutable=True
            )
            unpack(words, v, source)
            if cutlass.const_expr(self.aux_fold):
                for k in cutlass.range_constexpr(4):
                    state[2][source, k] = words[k]
        self.weights(v, norm, query, col)
        cute.arch.cluster_wait()
        maximum, exp_sum = Float32(0), Float32(0)
        mixture.fill(0)
        if cutlass.const_expr(self.nb > 0):
            values = zeros(self.kpre, Float32)
            for source in cutlass.range_constexpr(self.nb):
                values[source] = dot8(v, source, v, source)
                values[self.nb + source] = dot8(v, source, v, W)
            self.exchange(values, self.npre, self.kpre, bufs, 0, ids)
            logits = cute.make_rmem_tensor(self.nb, Float32)
            for b in cutlass.range_constexpr(self.nb):
                logits[b] = total[0, self.nb + b] * rstd(total[0, b], eps)
            maximum = logits[0]
            for source in cutlass.range_constexpr(1, self.nb):
                maximum = cute.arch.fmax(maximum, logits[source])
            for source in cutlass.range_constexpr(self.nb):
                e = cute.math.exp(logits[source] - maximum, fastmath=True)
                exp_sum += e
                for i in cutlass.range_constexpr(8):
                    mixture[0, i] = fma(e, v[source, i], mixture[0, i])
        scal[0], scal[1] = maximum, exp_sum
        if cutlass.const_expr(self.one_round):
            # Round 2's slot carries sum(mix^2) here, before the delta (off the delta's path).
            mm = cute.make_rmem_tensor(1, Float32)
            mm[0] = dot8(mixture, 0, mixture, 0)
            self.exchange(mm, 1, 1, bufs, 2, ids)
        if cutlass.const_expr(self.aux_fold):
            self.aux_blocks(v, bufs, rows, eps, ids)

    @cute.jit
    def aux_blocks(self, v, bufs, rows, eps: Float32, ids):
        """Aux scores of the saved blocks (complete at launch): score-warp sums, chunk sums
        exchanged in round 3, chunk-order totals, score = dot * rrms (into smem)."""
        prefix, blocks, norm, query, out_norm = rows
        rank, token, tid = ids
        red, chunks, scores = bufs[4]
        bar = bufs[3].iterator + 3
        cw = aux_cw(norm, query, (rank * 128 + tid) * 8)
        for source in cutlass.range_constexpr(self.nb):
            self.aux_partials(v, source, source, cw, red, tid)
        group_barrier(0)
        if tid < 2 * self.nb:
            value = aux_tree(red, tid % 2, tid // 2)
            for peer in cutlass.range_constexpr(7):
                push_float(chunks.iterator + rank * 16 + tid, value, bar, Int32(peer))
        wait(bar)
        if tid < self.nb:
            sumsq, dotv = aux_chunks(chunks, 2 * tid), aux_chunks(chunks, 2 * tid + 1)
            scores[tid] = aux_mul(dotv, aux_rrms(sumsq, eps))

    @cute.jit
    def weights(self, v, norm, query, col):
        if cutlass.const_expr(self.nb > 0):
            nw = load_words(norm.iterator + col, count=4)
            qw = load_words(query.iterator + col, count=4)
            for i in cutlass.range_constexpr(8):
                v[self.nb, i] = bf16(nw[i // 2], i % 2) * bf16(qw[i // 2], i % 2)
            # The stock TMA main path consumes a BF16 combined weight. Aux
            # constructs its own FP32 product in AuxMixture.
            for j in cutlass.range_constexpr(4):
                round_pair(v, self.nb, j)

    @cute.jit
    def post(
        self, state, bufs, delta, rows, output, eps: Float32, out_eps: Float32, ids, aux, previous=None
    ):
        """Add the delta, update prefix (and its ``aux`` copy) and saved block, write
        the normalized output."""
        prefix, blocks, norm, query, out_norm = rows
        rank, token, tid = ids
        warp_buf, peer_buf, total, bars = bufs[:4]
        mixture, scal = state[0], state[1]
        maximum, exp_sum = scal[0], scal[1]
        v = cute.make_rmem_tensor((self.nb + 4, 8), Float32)
        col = (rank * 128 + tid) * 8
        W, G, OLD, NEW = self.nb, self.nb + 1, self.nb + 2, self.nb + 3
        self.weights(v, norm, query, col)
        unpack(load_words(out_norm.iterator + col, count=4), v, G)
        if cutlass.const_expr(self.has_delta):
            unpack(load_words(at(prefix, token, col), count=4, mutable=True), v, OLD)
        cur = cute.make_rmem_tensor((1, 8), Float32)
        if cutlass.const_expr(self.slots == 0):
            cute.arch.griddepcontrol_wait()
            unpack(load_words(at(delta, token, col), count=4, mutable=True), cur, 0)
        else:
            cur.fill(0)
            base = delta.iterator + token * 7168 + col
            if cutlass.const_expr(self.rearm_prev):
                # DRA: the previous ordinal's rows (read by its consumer, a completed earlier grid: the same
                # 'complete at launch' order as this kernel's prefix read), re-armed off the output path.
                prior = previous.iterator + token * 7168 + col
                for slot in cutlass.range_constexpr(self.slots):
                    rearm(prior + slot * previous.shape[1] * 7168)
                rearm_fence()
            got = poll(
                base,
                delta.shape[1] * 7168,
                self.slots,
                self.deadline_ns,
            )
            for slot in cutlass.range_constexpr(self.slots):
                if cutlass.const_expr(not self.late_rearm and self.rearm_own):
                    rearm(base + slot * delta.shape[1] * 7168)
                part = cute.make_rmem_tensor((1, 8), Float32)
                unpack(tuple(got[slot, k] for k in range(4)), part, 0)
                for i in cutlass.range_constexpr(8):
                    cur[0, i] = cur[0, i] + part[0, i]
            if cutlass.const_expr(not self.late_rearm and self.rearm_own):
                rearm_fence()
            if cutlass.const_expr(self.slots > 1):
                for j in cutlass.range_constexpr(4):  # all-reduce output boundary
                    round_pair(cur, 0, j)
        words = cute.make_rmem_tensor(4, Uint32)
        for j in cutlass.range_constexpr(4):
            lo, hi = cur[0, 2 * j], cur[0, 2 * j + 1]
            if cutlass.const_expr(self.has_delta):
                lo, hi = v[OLD, 2 * j] + lo, v[OLD, 2 * j + 1] + hi
            words[j] = bf16x2(lo, hi)
        unpack(words, v, NEW)
        values = zeros(self.kpost, Float32)
        values[0] = dot8(v, NEW, v, NEW)
        if cutlass.const_expr(self.nb > 0):
            values[1] = dot8(v, NEW, v, W)
        if cutlass.const_expr(self.one_round):
            values[2] = dot8(mixture, 0, v, NEW)
        if cutlass.const_expr(self.aux_fold):
            cw = aux_cw(norm, query, col)
            self.aux_partials(v, NEW, self.nb, cw, bufs[4][0], tid)
        self.exchange(values, self.npost, self.kpost, bufs, 1, ids, self.naux)
        row = (words[0], words[1], words[2], words[3])
        store_words(at(prefix, token, col), row)
        if cutlass.const_expr(not self.aux_mixture):
            for stream_out in aux:
                store_words(at(stream_out, token, col), row)
        if cutlass.const_expr(self.write >= 0):
            store_words(block_ptr(blocks, token, self.write, col), row)
        result = cute.make_rmem_tensor((1, 8), Float32)
        if cutlass.const_expr(self.nb == 0):
            if cutlass.const_expr(self.aux_mixture):
                for stream_out in aux:
                    store_words(at(stream_out, token, col), row)
            scale = rstd(total[1, 0], out_eps)
            for i in cutlass.range_constexpr(8):
                result[0, i] = v[NEW, i] * scale * v[G, i]
        else:
            ss, dot = total[1, 0], total[1, 1]
            logit = dot * rstd(ss, eps)
            top = cute.arch.fmax(maximum, logit)
            alpha = cute.math.exp(maximum - top, fastmath=True)
            ep = cute.math.exp(logit - top, fastmath=True)
            denominator = fma(alpha, exp_sum, ep)
            # Match stock's elementwise FP32 accumulator-square statistic;
            # the old quadratic identity changed the floating-point graph.
            for i in cutlass.range_constexpr(8):
                result[0, i] = fma(alpha, mixture[0, i], ep * v[NEW, i])
            fin = cute.make_rmem_tensor(1, Float32)
            if cutlass.const_expr(self.one_round):
                # |alpha m + ep p|^2 = alpha^2 sum(m^2) + 2 alpha ep sum(m p) + ep^2 sum(p^2), clamped at 0.
                quad = fma(
                    alpha * alpha,
                    total[2, 0],
                    fma(Float32(2) * alpha * ep, total[1, 2], ep * ep * ss),
                )
                fin[0] = cute.arch.fmax(quad, Float32(0))
            else:
                fin[0] = dot8(result, 0, result, 0)
                if cutlass.const_expr(self.aux_overlap):
                    prefix_now = tuple(words[j] for j in range(4))
                    self.exchange(
                        fin, 1, 1, bufs, 2, ids, aux_args=(state, bufs, prefix_now, eps, ids, aux)
                    )
                else:
                    self.exchange(fin, 1, 1, bufs, 2, ids)
            inverse = cute.arch.rcp_approx(denominator)
            if cutlass.const_expr(self.one_round):
                scale = inverse * rstd(fin[0] * inverse * inverse, out_eps)
            else:
                scale = inverse * rstd(total[2, 0] * inverse * inverse, out_eps)
            for i in cutlass.range_constexpr(8):
                result[0, i] = result[0, i] * scale * v[G, i]
        prefix_words = tuple(words[j] for j in range(4))
        for j in cutlass.range_constexpr(4):
            words[j] = bf16x2(result[0, 2 * j], result[0, 2 * j + 1])
        store_words(at(output, token, col), tuple(words[j] for j in range(4)))
        if cutlass.const_expr(self.aux_fold and not self.aux_overlap):
            self.aux_mix(state, bufs, prefix_words, eps, ids, aux)
        if cutlass.const_expr(self.late_rearm and self.rearm_own):
            # EF4: every thread re-arms exactly the fragments it polled, then fences; the
            # caller's trailing griddepcontrol.wait and the exit follow (the edge needs only
            # re-arm -> fence -> grid end; no reader of this grid's other stores reads the slots).
            late = delta.iterator + token * 7168 + col
            for slot in cutlass.range_constexpr(self.slots):
                rearm(late + slot * delta.shape[1] * 7168)
            rearm_fence()

    @cute.jit
    def aux_mix(self, state, bufs, prefix_words, eps: Float32, ids, aux):
        """The prefix row's aux score (round-1 extra slots), then the combine pass's softmax
        over rows 0..nb and row-order FMA mixture, stored as BF16 to every aux output."""
        rank, token, tid = ids
        lane = tid % 32
        received = bufs[1][1, None]
        scores, blockw = bufs[4][2], state[2]
        sumsq = aux_chunks(received, self.npost)
        dotv = aux_chunks(received, self.npost + 1)
        prefix_score = aux_mul(dotv, aux_rrms(sumsq, eps))
        raw = Float32(float("-inf"))
        if lane < self.nb:
            raw = scores[lane]
        if lane == self.nb:
            raw = prefix_score
        maximum = raw
        for mask in cutlass.range_constexpr(5):
            maximum = cute.arch.fmax(
                maximum, cute.arch.shuffle_sync_bfly(maximum, 16 >> mask)
            )
        e = Float32(0)
        if lane <= self.nb:
            e = aux_exp(raw - maximum)
        total = warp_sum(e)
        p = aux_div(e, total)
        weights = [cute.arch.shuffle_sync(p, j) for j in range(self.nb + 1)]
        col = (rank * 128 + tid) * 8
        acc = [Float32(0) for _ in range(8)]
        for row in cutlass.range_constexpr(self.nb + 1):
            for j in cutlass.range_constexpr(4):
                word = prefix_words[j]
                if cutlass.const_expr(row < self.nb):
                    word = blockw[row, j]
                acc[2 * j] = fma(weights[row], bf16(word), acc[2 * j])
                acc[2 * j + 1] = fma(weights[row], bf16(word, True), acc[2 * j + 1])
        mixed = tuple(bf16x2(acc[2 * j], acc[2 * j + 1]) for j in range(4))
        for stream_out in aux:
            store_words(at(stream_out, token, col), mixed)
