# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the SGLang project
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""CONS1 (``K31MK_OPROJ_CONS1``, default off): the producer-reduced two-shot o-projection consumer
(``OprojConsumer(twoshot=True, reduced=True)``, one 128-thread group, i.e. the TP16 rows 17-128 under
``K31MK_OPROJ_PRODUCER_REDUCE``) as ONE 896-thread CTA per token instead of a cluster of 7 CTAs.

Part ``p = thread // 128`` does exactly what cluster rank ``p``'s CTA does (columns ``(p * 128 + tid) * 8``). The three
DSMEM exchanges of the cluster form (pre, post, norm: ``st.async`` of one FP32 value to each of the 7 peers + an mbarrier
wait) become one shared-memory store per value + a CTA barrier, and every part then sums the 7 parts' values in the
cluster form's peer order. Every FP32 operation, its operands and its order are the served ones, so prefix and output
are bitwise identical by construction (the values each peer received are the values each part stores once here).
As in the cluster form, thread ``i`` (of part 0) sums the post total ``i`` / the norm total from the shared values into
shared memory and a CTA barrier publishes them (v2, 11-07: v1 had all 896 threads repeat the 84 broadcast loads of these
sums, many shared-memory wavefronts on one SM; a test rig measured v1 SLOWER per call than the
served cluster at 24-128 rows). The gather poll, prefix store, gather re-arm (before the exchange, or after the publish under
``K31MK_OPROJ_EF1``, read from ``consumer.EF1`` when the kernel is compiled), publish and PDL edges (trigger at entry,
``griddepcontrol.wait`` at the end) are the served code.

Sync (memory-model audit): the only new edges are intra-CTA shared-memory hand-offs ordered by ``bar.sync 0``
(``cute.arch.sync_threads``; PTX ISA bar.sync: memory accesses requested by the threads before the barrier are performed
relative to all threads of the CTA participating), the standard __syncthreads exchange (proven in production: every
CUDA reduction kernel). The removed edges (cluster arrive / wait, remote ``st.async`` + ``mbarrier.try_wait.acquire
.cluster``) need no replacement. The Lamport gather poll + re-arm and the PDL edges are unchanged lines of the served
form.

CDR (K31MK_OPROJ_DEFER_REARM, consumer.py ``defer`` / ``keep_own``) is carried the same way: the previous layer's gather
cells (``defer`` elements away) re-armed + one rearm_fence() right before the poll, the own re-arm skipped unless ``keep_own``.

Grid (B, 1, 1) x 896 threads, no cluster. Shared memory ~ 7 x (4 + nb) x 2 KB (the per-part prefetch rows) + 3 KB."""

import cutlass
import cutlass.cute as cute
from cutlass import Float32, Uint32

from ..attn_res.kernel import rstd
from ..common import mailbox as MB
from ..common.mailbox import rearm, rearm_fence
from ..common.primitives import (
    add_rn,
    alloc,
    at,
    copy_g2s,
    fma,
    launch,
    pack_pair,
    smem_words,
    store_words,
    unpack_bf16,
    warp_sum,
    zeros,
)
from . import consumer as CM
from .consumer import OprojConsumer, dot8, mul_rn, pair_op, reduce_scatter, two
from .producer import pack8, tree_reduce

PARTS = 7  # 7 x 128 threads x 8 columns = 7168


class OprojConsumerCta(OprojConsumer):
    """``OprojConsumer(nb, twoshot=True, reduced=True)`` (groups 1) on one 896-thread CTA per token. ``ef1``: None = the
    module flag ``consumer.EF1`` at compile time (as the served consumer); True / False force it (tests)."""

    def __init__(self, nb, *, has_delta=True, deadline_ns=0, ef1=None, defer=0, keep_own=True):
        super().__init__(nb, has_delta=has_delta, groups=1, twoshot=True, deadline_ns=deadline_ns, reduced=True, defer=defer,
                         keep_own=keep_own)
        self.ef1 = ef1

    def _ef1(self):
        return CM.EF1 if self.ef1 is None else bool(self.ef1)

    @cute.jit
    def allocate_cta(self):
        smem = cutlass.utils.SmemAllocator()
        wpre = alloc(smem, Float32, (PARTS, 4, 16))
        pre = alloc(smem, Float32, (PARTS, 16))
        blk = alloc(smem, Float32, (PARTS, 16))
        post = alloc(smem, Float32, (PARTS, 4, 4))
        norm_post = alloc(smem, Float32, (PARTS, 4))
        pf = alloc(smem, Uint32, (PARTS, 4 + self.nb, 128, 4))
        tot = alloc(smem, Float32, (4,))  # v2: the post totals [0, npost) and the norm total [2], summed once per CTA
        return wpre, pre, blk, post, norm_post, pf, tot

    @cute.jit
    def row_ptr(self, pf, p, row: cutlass.Constexpr[int], tid):
        return pf.iterator + ((p * (4 + self.nb) + row) * 128 + tid) * 4

    @cute.jit
    def frag(self, pf, p, row: cutlass.Constexpr[int], tid):
        """``OprojConsumer.fragment`` of part ``p``."""
        words = smem_words(self.row_ptr(pf, p, row, tid))
        values = cute.make_rmem_tensor(8, Float32)
        for j in cutlass.range_constexpr(4):
            values[2 * j], values[2 * j + 1] = unpack_bf16(words[j])
        return values

    @cute.jit
    def cweight(self, pf, p, tid):
        """``OprojConsumer.combined_weight`` of part ``p``."""
        a, b = self.frag(pf, p, 0, tid), self.frag(pf, p, 1, tid)
        result = cute.make_rmem_tensor(8, Float32)
        for j in cutlass.range_constexpr(4):
            product = pair_op(two(a, j), two(b, j), op="mul")
            result[2 * j], result[2 * j + 1] = unpack_bf16(pack_pair(*product))
        return result

    @cute.jit
    def prefetch_cta(self, args, pf, p, tid, token):
        """``OprojConsumer.prefetch`` of part ``p`` (rank ``p``'s columns)."""
        _, prefix, blocks, norm, query, out_norm = args[:6]
        h = (p * 128 + tid) * 8
        if cutlass.const_expr(self.nb > 0):
            copy_g2s(self.row_ptr(pf, p, 0, tid), norm.iterator + h)
            copy_g2s(self.row_ptr(pf, p, 1, tid), query.iterator + h)
        copy_g2s(self.row_ptr(pf, p, 2, tid), out_norm.iterator + h)
        if cutlass.const_expr(self.has_delta):
            copy_g2s(self.row_ptr(pf, p, 3, tid), at(prefix, token, h))
        for block in cutlass.range_constexpr(self.nb):
            copy_g2s(self.row_ptr(pf, p, 4 + block, tid), at(blocks, token, block, h))
        cute.arch.cp_async_commit_group()

    @cute.jit
    def peer_total(self, post, i: cutlass.Constexpr[int]):
        """The cluster form's ``total[i]``: per warp slot the 7 parts in peer order, then the 4 slots' tree."""
        sums = [Float32(0)] * 4
        for peer in cutlass.range_constexpr(PARTS):
            for w in cutlass.range_constexpr(4):
                sums[w] = add_rn(sums[w], post[peer, w, i])
        return tree_reduce(sums, add_rn)

    @cute.jit
    def norm_peer_total(self, norm_post):
        """The cluster form's ``norm_total[0]``."""
        sums = [Float32(0)] * 4
        for peer in cutlass.range_constexpr(PARTS):
            for w in cutlass.range_constexpr(4):
                sums[w] = add_rn(sums[w], norm_post[peer, w])
        return tree_reduce(sums, add_rn)

    @cute.jit
    def pre_cta(self, state, p, tid, eps, mix):
        """``OprojConsumer.pre`` of part ``p``; the pre exchange through shared memory."""
        wpre, pre, blk, _, _, pf, _ = state
        warp, lane = tid // 32, tid % 32
        maximum, exponent_sum, qpart = Float32(0), Float32(0), Float32(0)
        cute.arch.cp_async_wait_group(0)
        if cutlass.const_expr(self.nb > 0):
            weight = self.cweight(pf, p, tid)
            partial = zeros(self.kpre, Float32)
            for block in cutlass.range_constexpr(self.nb):
                value = self.frag(pf, p, 4 + block, tid)
                partial[block] = dot8(value, value)
                partial[self.nb + block] = dot8(value, weight)
            idx = reduce_scatter(partial, self.kpre, lane)
            if ((lane & (32 // self.kpre - 1)) == 0) & (idx < 2 * self.nb):
                wpre[p, warp, idx] = partial[0]
            cute.arch.sync_threads()  # cluster form: named_barrier() + cluster_wait()
            if tid < 2 * self.nb:
                pre_sum = Float32(0)
                for w in cutlass.range_constexpr(4):
                    pre_sum = add_rn(pre_sum, wpre[p, w, tid])
                pre[p, tid] = pre_sum  # cluster form: push_float to the 7 peers' pre[rank, tid]
            cute.arch.sync_threads()  # cluster form: wait_exchange(bars)
            if tid < 2 * self.nb:
                sums = [Float32(0), Float32(0)]
                for peer in cutlass.range_constexpr(PARTS):
                    sums[peer % 2] = add_rn(sums[peer % 2], pre[peer, tid])
                blk[p, tid] = add_rn(*sums)
            cute.arch.sync_threads()  # cluster form: named_barrier()
            logits = [mul_rn(blk[p, self.nb + b], rstd(blk[p, b], eps)) for b in range(self.nb)]
            maximum = tree_reduce(logits, max)
            exp = [cute.math.exp(x - maximum, fastmath=True) for x in logits]
            exponent_sum = tree_reduce(exp, add_rn)
            for block in cutlass.range_constexpr(self.nb):
                value = self.frag(pf, p, 4 + block, tid)
                for j in cutlass.range_constexpr(4):
                    mix[2 * j], mix[2 * j + 1] = pair_op((exp[block], exp[block]), two(value, j), two(mix, j), op="fma")
            qpart = dot8(mix, mix)
        return maximum, exponent_sum, qpart

    @cute.jit
    def post_cta(self, args, state, block_mix, p, tid, token):
        """``OprojConsumer.post`` (groups 1, reduced) of part ``p``; post and norm exchanges through shared memory."""
        mailbox, prefix, _, _, _, _, output, xmoe, hints, eps, out_eps = args[:11]
        _, _, _, post, norm_post, pf, tot = state
        mix, maximum, exponent_sum, qpart = block_mix
        warp, lane = tid // 32, tid % 32
        h = (p * 128 + tid) * 8
        ef1: cutlass.Constexpr = self._ef1() and xmoe is None
        own: cutlass.Constexpr = self.defer == 0 or self.keep_own  # CDR: else the next layer re-arms these cells at its entry
        if cutlass.const_expr(self.has_delta):
            old = self.frag(pf, p, 3, tid)
        if cutlass.const_expr(self.defer != 0):  # CDR: the previous layer's cell, under the hop wait
            rearm(at(mailbox, 0, token, h) + self.defer)
            rearm_fence()
        attn = self.poll_sum(mailbox, token, h, 0, 1)
        prefix_value = cute.make_rmem_tensor(8, Float32)
        for j in cutlass.range_constexpr(4):
            if cutlass.const_expr(CM.SINGLE_ROUND):  # two-shot: the producer's owner already added the prefix
                acc = two(attn, j)
                rounded = unpack_bf16(pack_pair(acc[0], acc[1]))
            else:  # base D: bf16(old + bf16(sum))
                rounded = unpack_bf16(pack_pair(*two(attn, j)))
                if cutlass.const_expr(self.has_delta):
                    rounded = pair_op(two(old, j), rounded)
                    rounded = unpack_bf16(pack_pair(rounded[0], rounded[1]))
            prefix_value[2 * j], prefix_value[2 * j + 1] = rounded[0], rounded[1]
        partial = cute.make_rmem_tensor(self.npost, Float32)
        if cutlass.const_expr(self.nb > 0):
            weight = self.cweight(pf, p, tid)
            partial[0] = dot8(prefix_value, prefix_value)
            partial[1] = dot8(prefix_value, weight)
        else:
            partial[0] = dot8(prefix_value, prefix_value)
        idx = reduce_scatter(partial, self.npost, lane)
        if ((lane & (32 // self.npost - 1)) == 0) & (idx < self.npost):
            post[p, warp, idx] = partial[0]  # cluster form: the same value pushed to the 7 peers' post[rank, warp, idx]
        store_words(at(prefix, token, h), pack8(prefix_value))
        if cutlass.const_expr(not ef1 and own):  # served placement (before the post exchange)
            rearm(at(mailbox, 0, token, h))
            if cutlass.const_expr(xmoe is not None or bool(MB.REARM_FENCE)):
                rearm_fence(xmoe is not None)
        cute.arch.sync_threads()  # cluster form: wait_exchange(bars + 1)
        for i in cutlass.range_constexpr(self.npost):  # v2: thread i of part 0 sums total[i] once (cluster form: thread i)
            if (p == 0) & (tid == i):
                tot[i] = self.peer_total(post, i)
        cute.arch.sync_threads()  # cluster form: named_barrier()
        totals = cute.make_rmem_tensor(2, Float32)
        for i in cutlass.range_constexpr(self.npost):
            totals[i] = tot[i]
        norm = self.frag(pf, p, 2, tid)
        final = cute.make_rmem_tensor(8, Float32)
        if cutlass.const_expr(self.nb > 0):
            lp = mul_rn(totals[1], rstd(totals[0], eps))
            mx = max(maximum, lp)
            alpha = cute.math.exp(maximum - mx, fastmath=True)
            ep = cute.math.exp(lp - mx, fastmath=True)
            den = fma(alpha, exponent_sum, ep)
            for j in cutlass.range_constexpr(4):
                scaled = pair_op((ep, ep), two(prefix_value, j), op="mul")
                value = pair_op((alpha, alpha), two(mix, j), scaled, op="fma")
                final[2 * j], final[2 * j + 1] = value[0], value[1]
            square = warp_sum(dot8(final, final))
            if lane == 0:
                norm_post[p, warp] = square  # cluster form: pushed to the 7 peers' norm_post[rank, warp]
            cute.arch.sync_threads()  # cluster form: wait_exchange(bars + 2)
            if (p == 0) & (tid == 0):  # v2: summed once (cluster form: thread 0's norm_total[0])
                tot[2] = self.norm_peer_total(norm_post)
            cute.arch.sync_threads()  # cluster form: named_barrier()
            norm_total = tot[2]
            inverse = cute.arch.rcp_approx(den)
            scale = mul_rn(inverse, rstd(mul_rn(mul_rn(norm_total, inverse), inverse), out_eps))
        else:
            scale = rstd(totals[0], out_eps)
            for i in cutlass.range_constexpr(8):
                final[i] = prefix_value[i]
        for j in cutlass.range_constexpr(4):
            value = pair_op((scale, scale), two(final, j), op="mul")
            value = pair_op(value, two(norm, j), op="mul")
            final[2 * j], final[2 * j + 1] = value[0], value[1]
        self.publish(output, xmoe, hints, token, h, final)
        if cutlass.const_expr(ef1 and own):  # EF1: the same re-arm block after the output publish
            rearm(at(mailbox, 0, token, h))
            if cutlass.const_expr(bool(MB.REARM_FENCE)):
                rearm_fence()

    @cute.jit
    def __call__(self, *args):
        launch(self.kernel, args, args[1].shape[0], 128 * PARTS)

    @cute.kernel
    def kernel(self, args):
        token, _, _ = cute.arch.block_idx()
        thread, _, _ = cute.arch.thread_idx()
        p, tid = thread // 128, thread % 128
        state = self.allocate_cta()
        cute.arch.griddepcontrol_launch_dependents()
        mix = zeros(8, Float32)
        self.prefetch_cta(args, state[5], p, tid, token)
        maximum, exponent_sum, qpart = self.pre_cta(state, p, tid, args[9], mix)
        self.post_cta(args, state, (mix, maximum, exponent_sum, qpart), p, tid, token)
        cute.arch.griddepcontrol_wait()
