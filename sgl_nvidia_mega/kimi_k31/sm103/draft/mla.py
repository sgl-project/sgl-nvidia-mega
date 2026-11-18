# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the SGLang project
"""Draft MLA query / value: the K3 megakernel's ``MlaQuery`` (tiled, B >= TILED_FROM) and
``MlaValue`` with the two DFlash2 draft differences, as subclasses so the target kernels
stay untouched:

- ``DraftMlaQuery``: DeepSeek interleaved RoPE on q_pe (FP32 rotation of the BF16 Q_B row,
  then FP8, as RopeQuantize) and on k_pe (rotated in BF16, then FP8, as the stock
  rotary_emb + set_mla_kv_buffer), at ``positions`` from the ``cos_sin`` cache
  ([max_pos, 64] FP32, cos | sin halves). Extra arguments after MlaQuery's: positions
  [tokens] int64, cos_sin. A CTA owns R = 24 Q_B rows of a head; interleaved pairs never
  straddle CTAs (neox halves would), so the rotation stays CTA-local.
- ``DraftMlaValue``: W_UV without the sigmoid output gate (the draft has none); arguments
  latent, w_uv, lengths, output.

``insert`` / ``tiled`` / ``kernel`` are the target's MlaQuery / MlaValue with only
those edits."""

import cutlass
import cutlass.cute as cute
from cutlass import BFloat16, Float32, Int32

from ..k3.common.primitives import (
    alloc,
    at,
    bf16x2,
    copy_evict,
    copy_g2s,
    copy_s2cluster,
    fma,
    fma2,
    fp8_bits,
    ldmatrix,
    load_bf16x8,
    load_words,
    mma_into,
    mul2,
    prefetch_l2,
    query_norm_rsqrt,
    smem_words,
    store_fp8_pair,
    store_shared_words,
    store_words,
    unpack_bf16,
    view,
    wait,
    warp_sum,
    zeros,
)
from ..k3.mla.query import KP, TILED_FROM, MlaQuery
from ..k3.mla.value import MlaValue

ROPE = 64


class DraftMlaQuery(MlaQuery):
    def __init__(self, heads: int, batch: int, *, interleave: bool = True):
        super().__init__(heads, batch)
        if not self.wide:
            raise ValueError(
                f"DraftMlaQuery runs the tiled kernel: batch {batch} < {TILED_FROM[heads]}"
            )
        if not interleave:
            raise NotImplementedError("neox RoPE halves straddle the cluster's CTAs")

    @cute.jit
    def rotate(self, args, qpe, tok0, part, head, qsi, tid):
        """RoPE of this CTA's q_pe rows (Q_B rows R*part .. R*part + R that are >= 128),
        FP32, then FP8 into the query row."""
        output, positions, cos_sin = args[10], args[11], args[12]
        R, T = self.rows, self.tokens
        for e in range(tid, T * (R // 2), 256):
            t, k = e // (R // 2), e % (R // 2)
            p = R * part + 2 * k
            if (p >= 128) & self.live(tok0 + t):
                d = p - 128
                pos = Int32(positions[tok0 + t])
                cos, sin = cos_sin[pos, d // 2], cos_sin[pos, ROPE // 2 + d // 2]
                x1, x2 = qpe[t, d], qpe[t, d + 1]
                output[tok0 + t, head, 512 + d] = fp8_bits((x1 * cos - x2 * sin) * qsi)
                output[tok0 + t, head, 513 + d] = fp8_bits((x2 * cos + x1 * sin) * qsi)

    @cute.jit
    def insert(self, args, ss, tok0, j, valid, tid):
        """RMSNorm the current latent KV row of request tok0 + j and write it, with
        k_pe rotated at the row's position, to its FP8 cache slot."""
        qkv, w_kva, (cache, slots), (k_scale_inv, eps) = (
            args[0],
            args[2],
            args[5:7],
            args[8:10],
        )
        lane, warp = tid % 32, tid // 32
        slot = cutlass.Int64(-1)
        if valid:
            slot = slots[tok0 + j]
        if slot >= 0:
            ksi = k_scale_inv[0]
            xx = cute.make_rmem_tensor(8, Float32)
            total = Float32(0)
            if tid < 64:
                values = load_bf16x8(at(qkv, tok0 + j, 1536 + tid * 8), mutable=True)
                for i in cutlass.range_constexpr(8):
                    xx[i] = values[i]
                    total = fma(values[i], values[i], total)
            total = warp_sum(total)
            cute.arch.sync_threads()
            if (lane == 0) & (warp < 2):
                ss[warp, 0] = total
            cute.arch.sync_threads()
            page, offset = slot // cache.shape[1], slot % cache.shape[1]
            if tid < 64:
                mean = fma(ss[0, 0] + ss[1, 0], Float32(1 / 512), eps)
                rr = cute.math.rsqrt(mean, fastmath=True)
                weights = load_bf16x8(w_kva.iterator + tid * 8)
                for i in cutlass.range_constexpr(8):
                    normed = Float32(BFloat16(xx[i] * rr * weights[i]))
                    cache[page, offset, tid * 8 + i] = fp8_bits(normed * ksi)
            elif tid < 72:
                c = tid - 64
                values = load_bf16x8(at(qkv, tok0 + j, 2048 + c * 8), mutable=True)
                positions, cos_sin = args[11], args[12]
                pos = Int32(positions[tok0 + j])
                for m in cutlass.range_constexpr(4):
                    i = c * 4 + m  # interleaved pair (2i, 2i + 1)
                    cos, sin = cos_sin[pos, i], cos_sin[pos, ROPE // 2 + i]
                    x1, x2 = values[2 * m], values[2 * m + 1]
                    # The stock path rotates k_pe in BF16 (rotary_emb), then quantizes.
                    r1 = Float32(BFloat16(x1 * cos - x2 * sin))
                    r2 = Float32(BFloat16(x2 * cos + x1 * sin))
                    cache[page, offset, 512 + c * 8 + 2 * m] = fp8_bits(r1 * ksi)
                    cache[page, offset, 513 + c * 8 + 2 * m] = fp8_bits(r2 * ksi)

    @cute.kernel
    def tiled(self, args):
        """MlaQuery.tiled with q_pe rotated (``rotate``) before its FP8 store.

        B>8: T requests per CTA. Q_B fragments load to registers before the PDL
        wait; in-place RMSNorm of the staged rows, split-K Q_B MMAs over the warps,
        nope rows gathered from the six producing CTAs by bulk DSMEM copies (into the
        dead x rows), then W_UK MMAs (warp w owns latent columns 8w..8w+7)."""
        qkv, w_qa, w_qb, w_uk, q_scale_inv = (args[i] for i in (0, 1, 3, 4, 7))
        eps, output = args[9], args[10]
        tid, _, _ = cute.arch.thread_idx()
        part, head, tok0 = cute.arch.block_idx()
        tok0 *= self.tokens
        smem = cutlass.utils.SmemAllocator()
        T, R, L = self.tokens, self.rows, self.latent
        MT, senders = T // 16, (128 + R - 1) // R
        lane, warp = tid % 32, tid // 32
        group, quarter = lane // 4, lane % 4
        xq = alloc(smem, BFloat16, (T, KP), 128)
        uk = alloc(smem, BFloat16, (128, L + 8), 128)  # 144 B rows for ldmatrix
        qa = alloc(smem, BFloat16, 1536)
        ss = alloc(smem, Float32, (8, 2))
        qpe = alloc(smem, Float32, (T, ROPE))  # this CTA's q_pe rows, before RoPE
        barq = smem.allocate(cutlass.Uint64, 8)
        # After the Q_B MMAs the x rows hold the split-K partials, then our nope rows
        # and the six producers' (peers write them only after our cluster arrival).
        red = view(xq.iterator, 0, Float32, (8, T * R // 32, 32), (T * R, 32, 1))
        qblk = view(xq.iterator, 8 * T * R * 4, BFloat16, (T, R), (R, 1))
        qn = view(xq.iterator, 9 * T * R * 4, BFloat16, (senders, T, R), (T * R, R, 1))
        if tid == 0:
            cute.arch.mbarrier_init(barq, 1)
            cute.arch.mbarrier_init_fence()
            cute.arch.mbarrier_arrive_and_expect_tx(barq, senders * T * R * 2)
        # Split-K Q_B: warp w owns k = 192w..192w+191 (8 values per lane and 32-wide
        # chunk; the x fragments use the same permutation). L2 now, registers later.
        rows = head * 192 + part * R + group
        qb_at = [
            [
                at(w_qb, rows + 8 * n, (6 * warp + cc) * 32 + 8 * quarter)
                for n in range(3)
            ]
            for cc in range(6)
        ]
        for chunk in qb_at:
            for address in chunk:
                prefetch_l2(address)
        for c in range(tid, 128 * L // 8, 256):
            p, j = c // (L // 8), c % (L // 8)
            dst = uk.iterator + p * (L + 8) + j * 8
            copy_evict(dst, at(w_uk, head, p, part * L + j * 8))
        if tid < 192:
            copy_evict(qa.iterator + tid * 8, w_qa.iterator + tid * 8)
        cute.arch.cp_async_commit_group()
        cute.arch.griddepcontrol_wait()
        cute.arch.griddepcontrol_launch_dependents()
        for c in range(tid, T * 192, 256):
            t, j = c // 192, c % 192
            if self.live(tok0 + t):
                copy_g2s(xq.iterator + t * KP + j * 8, at(qkv, tok0 + t, j * 8))
        cute.arch.cp_async_commit_group()
        qsi = q_scale_inv[0]
        # The KV insert overlaps the staging of x (CTA j of the tile owns request j).
        j = head * self.cluster + part
        self.insert(args, ss, tok0, j, (j < T) & self.live(tok0 + j), tid)
        cute.arch.cp_async_wait_group(0)
        cute.arch.sync_threads()
        # Packed FP32 pairs: the same IEEE ops in the same order as the scalar form.
        gamma = []
        for i in cutlass.range_constexpr(6):
            gamma += list(load_bf16x8(qa.iterator + 256 * i + 8 * lane, shared=True))
        for pair in cutlass.range_constexpr(T // 16):  # requests warp + 8k, two at once
            xrows = [xq.iterator + (warp + 16 * pair + 8 * h) * KP for h in range(2)]
            values: list = []
            rr: list = []
            for h in cutlass.range_constexpr(2):
                values.append([])
                for i in cutlass.range_constexpr(6):
                    for word in smem_words(xrows[h] + 256 * i + 8 * lane):
                        values[h] += list(unpack_bf16(word))
            for h in cutlass.range_constexpr(2):
                zero = Float32(0), Float32(0)
                chains = [zero, zero]  # (0, 1), (2, 3): element e feeds chain e % 4
                for e in cutlass.range_constexpr(0, 48, 2):
                    x = values[h][e], values[h][e + 1]
                    chains[e % 4 // 2] = fma2(x, x, chains[e % 4 // 2])
                (c0, c1), (c2, c3) = chains
                rr.append(query_norm_rsqrt(warp_sum((c0 + c1) + (c2 + c3)), eps))
            for h in cutlass.range_constexpr(2):
                for i in cutlass.range_constexpr(6):
                    words = []
                    for e in cutlass.range_constexpr(8 * i, 8 * i + 8, 2):
                        x = values[h][e], values[h][e + 1]
                        y = mul2(mul2(x, (rr[h], rr[h])), (gamma[e], gamma[e + 1]))
                        words.append(bf16x2(*y))
                    store_shared_words(xrows[h] + 256 * i + 8 * lane, tuple(words))
        qbr = [[load_words(address, count=4) for address in chunk] for chunk in qb_at]
        cute.arch.sync_threads()
        acc = zeros((MT * 3, 4), Float32)
        for cc in cutlass.range_constexpr(6):
            k = (6 * warp + cc) * 32 + 8 * quarter
            z = qbr[cc]
            for m in cutlass.range_constexpr(MT):
                x, y = (
                    load_words(xq.iterator + r * KP + k, count=4, shared=True)
                    for r in (16 * m + group, 16 * m + group + 8)
                )
                for n in cutlass.range_constexpr(3):
                    for s in cutlass.range_constexpr(2):
                        a = (x[2 * s], y[2 * s], x[2 * s + 1], y[2 * s + 1])
                        b = (z[n][2 * s], z[n][2 * s + 1])
                        mma_into(acc[m * 3 + n, None], a, b, bf16=True)
        cute.arch.sync_threads()
        cute.arch.cluster_arrive_relaxed()  # x is consumed: peers may fill qn
        for i in cutlass.range_constexpr(MT * 12):
            red[warp, i, lane] = acc[i // 4, i % 4]
        cute.arch.sync_threads()
        for e in range(tid, T * R, 256):
            i, ln = e // 32, e % 32
            value = Float32(0)
            for w in cutlass.range_constexpr(8):
                value += red[w, i, ln]
            value = Float32(BFloat16(value))
            tile, c = i // 4, i % 4
            t = 16 * (tile // 3) + ln // 4 + 8 * (c // 2)
            r = 8 * (tile % 3) + 2 * (ln % 4) + c % 2
            p = R * part + r
            if p < 128:
                qblk[t, r] = BFloat16(value)
            elif self.live(tok0 + t):
                qpe[t, p - 128] = value
        cute.arch.fence_proxy("async.shared", space="cta")
        cute.arch.sync_threads()
        self.rotate(args, qpe, tok0, part, head, qsi, tid)
        cute.arch.cluster_wait()
        if (part < senders) & (tid < self.cluster):
            copy_s2cluster(
                qn.iterator + part * T * R, qblk.iterator, barq, tid, size=T * R * 2
            )
        wait(barq)
        cute.arch.cluster_arrive_relaxed()  # peers may exit once qblk is delivered
        b_frags = [
            ldmatrix(uk.iterator + (32 * kp + lane) * (L + 8) + 8 * warp, trans=True)
            for kp in range(4)
        ]
        for m in cutlass.range_constexpr(MT):
            out = zeros(4, Float32)
            t = 16 * m + lane % 8 + 8 * ((lane // 8) % 2)
            for ks in cutlass.range_constexpr(8):
                p = 16 * ks + 8 * (lane // 16)
                a = ldmatrix(qn.iterator + ((p // R) * T + t) * R + p % R)
                b = b_frags[ks // 2][2 * (ks % 2) : 2 * (ks % 2) + 2]
                mma_into(out, a, b, bf16=True)
            for h in cutlass.range_constexpr(2):  # rows group, group + 8: 2 bytes
                row = 16 * m + group + 8 * h
                if self.live(tok0 + row):
                    lo, hi = (Float32(BFloat16(out[2 * h + e])) * qsi for e in range(2))
                    col = L * part + 8 * warp + 2 * quarter
                    store_fp8_pair(at(output, tok0 + row, head, col), lo, hi)
        if cutlass.const_expr(self.dcp):
            self.publish(args, tok0, tid)
        cute.arch.cluster_wait()
        if cutlass.const_expr(self.dcp):
            self.gathered(args, tid)
        self.finish(args, tid, part, head, tok0)


class DraftMlaValue(MlaValue):
    @cute.kernel
    def kernel(self, args):
        latent, w_uv, lengths, output = args
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
        if cutlass.const_expr(self.early_trigger):
            cute.arch.griddepcontrol_launch_dependents()
        cute.arch.griddepcontrol_wait()
        for c in range(tid, 16 * 64, 128):
            r, j = c // 64, c % 64
            if t0 + r < self.batch:
                src = at(latent, t0 + r, head, j * 8)
                copy_g2s(rows.iterator + r * 544 + j * 8, src)
        cute.arch.cp_async_commit_group()
        col = head * 128 + chunk * 32 + warp * 8 + 2 * quarter
        if cutlass.const_expr(not self.early_trigger):
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
                lo, hi = Float32(0), Float32(0)
                if lengths[t // self.q] > 0:
                    lo, hi = (Float32(BFloat16(acc[2 * r + e])) for e in range(2))
                store_words(at(output, t, col), (bf16x2(lo, hi),))
