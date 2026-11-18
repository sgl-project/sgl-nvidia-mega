# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the SGLang project
"""Seven-CTA cluster AttnRes: one cluster per token, 1024 hidden columns per CTA.

The delta is a tensor (read after the PDL wait) or ``slots`` Lamport mailbox slots
summed in slot order, rounded to BF16 and re-armed. Output RMS statistics use
the FP32 mixed vector, including at the final boundary. Prefix and saved blocks must be complete at launch.
"""

import cutlass
import cutlass.cute as cute
from cutlass import Float32, Int32, Uint32, Uint64

from ..common.mailbox import (
    poll,
    rearm,
    rearm_fence_eager,
    rearm_fence_end,
    rearm_fence_late,
)
from ..common.primitives import (
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
    zeros,
)
from .aux import AuxMixture


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
    """Thread 0 arms barrier ``r`` for ``counts[r]`` FP32 pushes from each of 7 CTAs."""
    if cute.arch.thread_idx()[0] == 0:
        for r, count in enumerate(counts):
            if cutlass.const_expr(count > 0):
                cute.arch.mbarrier_init(bars.iterator + r, 1)
                cute.arch.mbarrier_arrive_and_expect_tx(bars.iterator + r, 28 * count)
        cute.arch.mbarrier_init_fence()


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
        aux_launch=True,
    ):
        self.nb, self.slots, self.has_delta = nb, slots, has_delta
        self.write, self.final = write, final
        self.deadline_ns = deadline_ns
        self.aux_mixture = aux_mixture
        self.aux_launch = (
            aux_launch  # False: the caller launches AuxMixture itself (side stream)
        )
        self.npre = 2 * nb
        self.npost = 2 if nb else 1
        self.kpre = 1 << max(self.npre - 1, 0).bit_length()
        self.kpost = 1 << max(self.npost - 1, 0).bit_length()

    @cute.jit
    def __call__(self, *args):
        launch(self.kernel, args, (7, args[1].shape[0], 1), 128, 7)
        if cutlass.const_expr(self.aux_mixture and self.nb > 0 and self.aux_launch):
            AuxMixture(self.nb)(*args)

    @cute.jit
    def exchange(
        self, values, count: cutlass.Constexpr, kpad: cutlass.Constexpr, bufs, r, ids
    ):
        """Cluster-sum ``count`` per-thread partials into round ``r``'s total row."""
        wb, peer_buf, total, bars = bufs
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
        self.post(state, bufs, delta, rows, output, eps, out_eps, ids, args[9:])
        rearm_fence_end()
        cute.arch.griddepcontrol_wait()

    def state(self):
        """Carried from pre() to post(); the rest reloads so a GEMM can run between."""
        return cute.make_rmem_tensor((1, 8), Float32), cute.make_rmem_tensor(2, Float32)

    @cute.jit
    def setup(self, smem):
        warp_buf = alloc(smem, Float32, 64)
        peer_buf = alloc(smem, Float32, (3, 112))
        total = alloc(smem, Float32, (3, 16))
        bars = alloc(smem, Uint64, 3, 8)
        arm_exchanges(bars, (self.npre, self.npost, int(self.nb > 0)))
        return warp_buf, peer_buf, total, bars

    @cute.jit
    def pre(self, state, bufs, rows, eps: Float32, ids):
        """Inputs complete at launch, then the saved-block statistics exchange."""
        prefix, blocks, norm, query, out_norm = rows
        rank, token, tid = ids
        warp_buf, peer_buf, total, bars = bufs
        mixture, scal = state
        # Rows: saved blocks, combined weight, gamma, old prefix, new prefix.
        v = cute.make_rmem_tensor((self.nb + 4, 8), Float32)
        col = (rank * 128 + tid) * 8
        W = self.nb
        for source in cutlass.range_constexpr(self.nb):
            words = load_words(
                block_ptr(blocks, token, source, col), count=4, mutable=True
            )
            unpack(words, v, source)
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
        self, state, bufs, delta, rows, output, eps: Float32, out_eps: Float32, ids, aux
    ):
        """Add the delta, update prefix (and its ``aux`` copy) and saved block, write
        the normalized output."""
        prefix, blocks, norm, query, out_norm = rows
        rank, token, tid = ids
        warp_buf, peer_buf, total, bars = bufs
        mixture, scal = state
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
            got = poll(
                base,
                delta.shape[1] * 7168,
                self.slots,
                self.deadline_ns,
            )
            for slot in cutlass.range_constexpr(self.slots):
                rearm(base + slot * delta.shape[1] * 7168)
                part = cute.make_rmem_tensor((1, 8), Float32)
                unpack(tuple(got[slot, k] for k in range(4)), part, 0)
                for i in cutlass.range_constexpr(8):
                    cur[0, i] = cur[0, i] + part[0, i]
            rearm_fence_eager()
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
        if cutlass.const_expr(self.slots > 0):
            rearm_fence_late()  # before the exchange's barrier, so before any thread's push
        self.exchange(values, self.npost, self.kpost, bufs, 1, ids)
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
            fin[0] = dot8(result, 0, result, 0)
            self.exchange(fin, 1, 1, bufs, 2, ids)
            inverse = cute.arch.rcp_approx(denominator)
            scale = inverse * rstd(total[2, 0] * inverse * inverse, out_eps)
            for i in cutlass.range_constexpr(8):
                result[0, i] = result[0, i] * scale * v[G, i]
        for j in cutlass.range_constexpr(4):
            words[j] = bf16x2(result[0, 2 * j], result[0, 2 * j + 1])
        store_words(at(output, token, col), tuple(words[j] for j in range(4)))
