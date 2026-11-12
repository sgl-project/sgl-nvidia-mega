# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the SGLang project
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Stock FP32-cw auxiliary mixture, separate from the BF16-cw main read.

The score pass temporarily stores at most 16 FP32 scores in the destination
slice. Every combine CTA loads them before a cluster barrier, then overwrites
that slice with the BF16 mixture. No binding buffer or host allocation changes.
"""

import cutlass
import cutlass.cute as cute
from cutlass import Float32, Uint32

from ..common.primitives import (
    add_rn,
    alloc,
    asm,
    at,
    bf16,
    bf16x2,
    bits_float,
    float_bits,
    fma,
    launch,
    load_words,
    store_words,
    warp_sum,
)


@asm(Float32, pure=True)
def aux_mul(a: Float32, b: Float32):
    # Keep stock score's materialized FP32 products separate from its sum.
    return "mul.rn.f32 $0,$1,$2;", "=f,f,f", (a, b)


@asm(Float32, pure=True)
def aux_div(numerator: Float32, denominator: Float32):
    """Triton's div.full, including its non-RN approximate normal path."""
    return "div.full.f32 $0,$1,$2;", "=f,f,f", (numerator, denominator)


@asm(Float32, pure=True)
def aux_sqrt(value: Float32):
    return "sqrt.approx.ftz.f32 $0,$1;", "=f,f", (value,)


@asm(Float32, pure=True)
def aux_rrms(sumsq: Float32, eps: Float32):
    # Keep the same unqualified add as stock's PTX. ptxas may fuse the
    # constant divisor's multiply with epsilon; an explicit add.rn forbids it.
    code = "{ .reg .f32 mean,root; div.full.f32 mean,$1,0f45e00000; "
    code += "add.f32 mean,mean,$2; sqrt.approx.ftz.f32 root,mean; "
    return code + "div.full.f32 $0,0f3f800000,root; }", "=f,f,f", (sumsq, eps)


@asm(Float32, pure=True)
def aux_exp(value: Float32):
    code = "{ .reg .f32 x; mul.f32 x,$1,0f3fb8aa3b; ex2.approx.f32 $0,x; }"
    return code, "=f,f", (value,)


class AuxMixture:
    def __init__(self, nb):
        self.nb = nb

    @cute.jit
    def __call__(self, *args):
        launch(self.score, args, (args[1].shape[0], self.nb + 1, 1), 256)
        launch(self.combine, args, (7, args[1].shape[0], 1), 128, 7)

    @cute.kernel
    def score(self, args):
        _, prefix, blocks, norm, query, _, _, eps, _ = args[:9]
        token, source, _ = cute.arch.block_idx()
        tid, _, _ = cute.arch.thread_idx()
        lane, warp = tid % 32, tid // 32
        reductions = alloc(cutlass.utils.SmemAllocator(), Float32, (2, 8))
        cute.arch.griddepcontrol_wait()
        sumsq, dotv = Float32(0), Float32(0)
        for chunk in cutlass.range_constexpr(7):
            values, combined = [], []
            for j in cutlass.range_constexpr(4):
                col = chunk * 1024 + tid * 4 + j
                ptr = prefix.iterator + token * prefix.stride[0] + col
                if source < self.nb:
                    ptr = (
                        blocks.iterator
                        + token * blocks.stride[0]
                        + source * blocks.stride[1]
                        + col
                    )
                # Each BF16 load is kept independent, as in the stock score pass.
                x = Float32(cute.make_tensor(ptr, cute.make_layout(1))[0])
                nw = Float32(norm[col])
                qw = Float32(query[col])
                values.append(x)
                combined.append(aux_mul(nw, qw))
            # Runtime Triton specialization has four adjacent BF16s per lane,
            # with pairwise FP32 FMAs followed by the scalar add.
            local_sq = add_rn(
                fma(values[0], values[0], aux_mul(values[2], values[2])),
                fma(values[1], values[1], aux_mul(values[3], values[3])),
            )
            local_dot = add_rn(
                fma(values[0], combined[0], aux_mul(values[2], combined[2])),
                fma(values[1], combined[1], aux_mul(values[3], combined[3])),
            )
            sq, dv = warp_sum(local_sq), warp_sum(local_dot)
            if lane == 0:
                reductions[0, warp], reductions[1, warp] = sq, dv
            cute.arch.sync_threads()
            s, d = reductions[0, lane % 8], reductions[1, lane % 8]
            s, d = warp_sum(s, 8), warp_sum(d, 8)
            sumsq, dotv = add_rn(sumsq, s), add_rn(dotv, d)
            cute.arch.sync_threads()
        rrms = aux_rrms(sumsq, eps)
        if tid == 0:
            for output in args[9:]:
                store_words(
                    at(output, token, source * 2),
                    (Uint32(float_bits(aux_mul(dotv, rrms))),),
                )
        cute.arch.griddepcontrol_launch_dependents()

    @cute.kernel
    def combine(self, args):
        _, prefix, blocks, _, _, _, _, _, _ = args[:9]
        part, token, _ = cute.arch.block_idx()
        tid, _, _ = cute.arch.thread_idx()
        lane = tid % 32
        cute.arch.griddepcontrol_wait()
        scores = args[9]
        raw = Float32(float("-inf"))
        if lane <= self.nb:
            raw = bits_float(
                load_words(at(scores, token, 2 * lane), count=1, mutable=True)[0]
            )
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
        # Each CTA saves all row weights before the first score word is overwritten.
        weights = [cute.arch.shuffle_sync(p, j) for j in range(self.nb + 1)]
        cute.arch.cluster_arrive()  # release: the score reads above before CTA 0's stores to the same words
        cute.arch.cluster_wait()
        col = (part * 128 + tid) * 8
        acc = [Float32(0) for _ in range(8)]
        for row in cutlass.range_constexpr(self.nb + 1):
            for j in cutlass.range_constexpr(4):
                ptr = prefix.iterator + token * prefix.stride[0] + col + 2 * j
                if cutlass.const_expr(row < self.nb):
                    ptr = (
                        blocks.iterator
                        + token * blocks.stride[0]
                        + row * blocks.stride[1]
                        + col
                        + 2 * j
                    )
                word = load_words(ptr, count=1, mutable=True)[0]
                acc[2 * j] = fma(weights[row], bf16(word), acc[2 * j])
                acc[2 * j + 1] = fma(weights[row], bf16(word, True), acc[2 * j + 1])
        words = tuple(bf16x2(acc[2 * j], acc[2 * j + 1]) for j in range(4))
        for output in args[9:]:
            store_words(at(output, token, col), words)
        cute.arch.griddepcontrol_launch_dependents()
