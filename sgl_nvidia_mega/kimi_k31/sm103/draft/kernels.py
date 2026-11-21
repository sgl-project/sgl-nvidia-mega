# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the SGLang project
"""DFlash2 draft layer kernels on the K3 megakernel blocks (TP16, one 8-row block per
request, rows grouped by request).

The draft wraps attention and MLP in a grouped dynamic depthwise 2-tap convolution
(``models/dflash.py`` DFlashGroupedConv): a replicated projection of the sublayer input
gives per-row, per-16-channel-group coefficient deltas δ[row, side, tap, group]; with
c = base[side, tap, channel] + δ, ``prepare`` (side 0) mixes row t with row t - 1 of the
same block before the sublayer and ``finish`` (side 1) does the same to its output.

- ``ConvPrepare``: the side-0 conv of a BF16 row block (one CTA per 1024 channels x row).
- ``RowNorm``: the chain entry (layer 0 has no residual yet): residual = x, out = RMSNorm(x).
- ``DraftProducer``: ``OprojProducer`` (TP16 B<=8, register x) whose x is a BF16 row or,
  with ``silu``, the gate|up row (x = silu(gate) * up); its publish applies the side-1
  conv to this rank's partial rows before the multicast. The conv is linear in the summed
  rows and δ is replicated, so finishing each partial equals finishing the sum up to
  rounding.
- ``DraftFused``: that producer + ``OprojConsumer(0, has_delta=True)`` in one launch (the
  TP16 B<=8 ``OprojFused`` schedule): sum the TP partials in rank order, add the residual
  (BF16), RMSNorm with the next norm's weight."""

import cutlass
import cutlass.cute as cute
from cutlass import Float32, Int64, Uint32

from ..k3.attn_res.kernel import rstd
from ..k3.common.primitives import (
    add_rn,
    alloc,
    at,
    launch,
    load_vector,
    multicast_words,
    sigmoid,
    sleep_ns,
    store_words,
    unpack_bf16,
    warp_sum,
    zeros,
)
from ..k3.oproj.consumer import OprojConsumer
from ..k3.oproj.producer import OprojProducer, pack8

HIDDEN = 7168
GROUP = 16  # conv channels per coefficient group
GROUPS = HIDDEN // GROUP  # 448
TAPS = 2
BLOCK = 8  # DFlash block rows per request
DELTA = 2 * TAPS * GROUPS  # 1792: δ row [side, tap, group]


@cute.jit
def conv8(
    x,
    prev,
    base,
    delta,
    row,
    side: cutlass.Constexpr[int],
    group,
    live_prev,
    permuted: cutlass.Constexpr[bool] = False,
):
    """Side ``side`` conv of 8 channels of one group: (base[0] + δ0) x + (base[1] + δ1) prev,
    FP32; ``base`` is a (2, 8) register tensor, ``prev`` is ignored unless ``live_prev``.
    δ columns are [side, tap, group], or with ``permuted`` 4 group + 2 side + tap (the
    folded conv projection's order, convgemm.conv_row_order)."""
    if cutlass.const_expr(permuted):
        c0 = Float32(delta[row, 4 * group + 2 * side])
        c1 = Float32(delta[row, 4 * group + 2 * side + 1])
    else:
        c0 = Float32(delta[row, (side * TAPS) * GROUPS + group])
        c1 = Float32(delta[row, (side * TAPS + 1) * GROUPS + group])
    out = cute.make_rmem_tensor(8, Float32)
    for i in cutlass.range_constexpr(8):
        out[i] = (base[0, i] + c0) * x[i]
    if live_prev:
        for i in cutlass.range_constexpr(8):
            out[i] = out[i] + (base[1, i] + c1) * prev[i]
    return out


@cute.jit
def load8(pointer):
    """FP32 of 8 BF16 values at ``pointer`` (16 B)."""
    words = load_vector(pointer)
    values = cute.make_rmem_tensor(8, Float32)
    for j in cutlass.range_constexpr(4):
        values[2 * j], values[2 * j + 1] = unpack_bf16(words[j])
    return values


@cute.jit
def load_base(base, tap_row: cutlass.Constexpr[int], h):
    """(2, 8) FP32 registers of base[side rows tap_row .. tap_row + 1, h .. h + 8]."""
    values = cute.make_rmem_tensor((2, 8), Float32)
    for t in cutlass.range_constexpr(2):
        for i in cutlass.range_constexpr(8):
            values[t, i] = Float32(base[tap_row + t, h + i])
    return values


class ConvPrepare:
    """x' = conv_side0(x, δ) of [M, 7168] BF16 rows (M = 8 x requests). One 128-thread CTA
    per (1024 channels, row); thread owns 8 channels (one conv group). ``base`` is the FP32
    [2 * TAPS, 7168] table (side-major); weights load before the PDL wait."""

    @cute.jit
    def __call__(self, *args):
        launch(self.kernel, args, (HIDDEN // 1024, args[0].shape[0], 1), 128)

    @cute.kernel
    def kernel(self, args):
        x, delta, base, out = args[0:4]
        chunk, row, _ = cute.arch.block_idx()
        tid, _, _ = cute.arch.thread_idx()
        h = (chunk * 128 + tid) * 8
        coefficients = load_base(base, 0, h)
        cute.arch.griddepcontrol_wait()
        cute.arch.griddepcontrol_launch_dependents()
        live_prev = row % BLOCK != 0
        current = load8(at(x, row, h))
        prev = zeros(8, Float32)
        if live_prev:
            loaded = load8(at(x, row - 1, h))
            for i in cutlass.range_constexpr(8):
                prev[i] = loaded[i]
        mixed = conv8(current, prev, coefficients, delta, row, 0, h // GROUP, live_prev)
        store_words(at(out, row, h), pack8(mixed))


class RowNorm:
    """Chain entry: residual = x (copy), out = RMSNorm(x) * weight, [M, 7168] BF16; one
    896-thread CTA per row (8 channels per thread)."""

    @cute.jit
    def __call__(self, *args):
        launch(self.kernel, args, (args[0].shape[0], 1, 1), HIDDEN // 8)

    @cute.kernel
    def kernel(self, args):
        x, weight, residual, out, eps = args[0:5]
        row, _, _ = cute.arch.block_idx()
        tid, _, _ = cute.arch.thread_idx()
        lane, warp = tid % 32, tid // 32
        smem = cutlass.utils.SmemAllocator()
        partials = alloc(smem, Float32, 32)
        h = tid * 8
        w = load8(weight.iterator + h)
        cute.arch.griddepcontrol_wait()
        cute.arch.griddepcontrol_launch_dependents()
        words = load_vector(at(x, row, h))
        store_words(at(residual, row, h), words)
        values = cute.make_rmem_tensor(8, Float32)
        square = Float32(0)
        for j in cutlass.range_constexpr(4):
            values[2 * j], values[2 * j + 1] = unpack_bf16(words[j])
        for i in cutlass.range_constexpr(8):
            square = square + values[i] * values[i]
        square = warp_sum(square)
        if lane == 0:
            partials[warp] = square
        cute.arch.sync_threads()
        total = Float32(0)
        for w_ in cutlass.range_constexpr(HIDDEN // 8 // 32):
            total = add_rn(total, partials[w_])
        scale = rstd(total, eps)
        for i in cutlass.range_constexpr(8):
            values[i] = values[i] * scale * w[i]
        store_words(at(out, row, h), pack8(values))


class DraftProducer(OprojProducer):
    """TP16 B<=8 o-projection / down producer of a draft sublayer (112 CTAs x 64 output
    rows, weights in registers, x in registers for any ``k``). Arguments: x, weight, box,
    mc, slot, δ [M, 1792], base [2 * TAPS, 7168] FP32. ``silu``: x is [M, 2k] gate|up and
    the producer stages silu(gate) * up (FP32, rounded to BF16). ``conv``: the side-1 conv
    of the sublayer is applied to the rank partial in the publish."""

    def __init__(self, k, *, silu, conv=True, permuted=False):
        super().__init__(k, staged_blocks=0)
        self.register_x = True
        self.silu, self.conv, self.permuted = silu, conv, permuted

    @cute.jit
    def x_words(self, x, token, col):
        words = load_vector(at(x, token, col))
        if cutlass.const_expr(self.silu):
            up = load_vector(at(x, token, self.k + col))
            values = cute.make_rmem_tensor(8, Float32)
            for j in cutlass.range_constexpr(4):
                g = unpack_bf16(words[j])
                u = unpack_bf16(up[j])
                for s in cutlass.range_constexpr(2):
                    # Fast-math sigmoid (draft numerics, AL-gated).
                    values[2 * j + s] = g[s] * sigmoid(g[s], True) * u[s]
            words = pack8(values, None)
        return words

    @cute.jit
    def preload_base(self, args, cta):
        """The side-1 conv base of this thread's publish columns, static: loaded before
        the PDL wait."""
        tid, _, _ = cute.arch.thread_idx()
        out_half, row_tile = tid % 2, (tid // 2) % 4
        return load_base(args[6], TAPS, (cta * 4 + row_tile) * 16 + out_half * 8)

    @cute.jit
    def compute(self, args, smem, wr, cta, trigger_after_wait: cutlass.Constexpr[bool]):
        x = args[0]
        coefficients = self.preload_base(args, cta) if self.conv else None
        red = smem[0]
        M: cutlass.Constexpr = x.shape[0]
        NT, _, _ = self.schedule(M)
        tid, _, _ = cute.arch.thread_idx()
        lane, warp = tid % 32, tid // 32
        group, part = lane // 4, lane % 4
        kofs = warp % self.parts * self.kblocks * 32 + part * 8
        cute.arch.griddepcontrol_wait()
        if cutlass.const_expr(trigger_after_wait):
            cute.arch.griddepcontrol_launch_dependents()
        xr = cute.make_rmem_tensor((NT, self.kblocks, 4), Uint32)
        xr.fill(0)
        for nt in cutlass.range_constexpr(NT):
            token = nt * 8 + group
            if token < M:
                for j in cutlass.range_constexpr(self.kblocks):
                    words = self.x_words(x, token, kofs + j * 32)
                    for v in cutlass.range_constexpr(4):
                        xr[nt, j, v] = words[v]
        acc = self.multiply(smem, wr, xr, 0, 0, M)
        self.store_partials(red, acc, NT)
        cute.arch.sync_threads()
        self.publish(args, red, cta, 0, coefficients)

    @cute.jit
    def reduce_row(self, red, tile, row_tile, row, out_half):
        """The rank partial of 8 outputs of one token: K parts summed in order."""
        reduced = cute.make_rmem_tensor(8, Float32)
        for v in cutlass.range_constexpr(8):
            value = red[tile, row_tile, 0, row, out_half * 8 + v]
            for q in cutlass.range_constexpr(1, self.parts):
                value = add_rn(value, red[tile, row_tile, q, row, out_half * 8 + v])
            reduced[v] = value
        return reduced

    @cute.jit
    def publish(self, args, red, cta, first: cutlass.Constexpr[int], coefficients=None):
        """Rank partial rows (+ the side-1 conv), multicast into slot ``slot`` of the box."""
        x, _, mailbox, mc, slot, delta = args[:6]
        count: cutlass.Constexpr = x.shape[0] - first
        tid, _, _ = cute.arch.thread_idx()
        if tid < 8 * count:
            out_half, row_tile, token = tid % 2, (tid // 2) % 4, tid // 8
            tile, row = token // 8, token % 8
            out_col = (cta * 4 + row_tile) * 16 + out_half * 8
            reduced = self.reduce_row(red, tile, row_tile, row, out_half)
            if cutlass.const_expr(self.conv):
                live_prev = row != 0
                prev = zeros(8, Float32)
                if live_prev:
                    above = self.reduce_row(red, tile, row_tile, row - 1, out_half)
                    for i in cutlass.range_constexpr(8):
                        prev[i] = above[i]
                reduced = conv8(
                    reduced,
                    prev,
                    coefficients,
                    delta,
                    first + token,
                    1,
                    out_col // GROUP,
                    live_prev,
                    self.permuted,
                )
            words = pack8(reduced, "latent")
            offset = cute.crd2idx((slot, first + token, out_col), mailbox.layout)
            multicast_words(mc + Int64(offset) * 2, words)


class DraftFused:
    """``DraftProducer`` + ``OprojConsumer(0, has_delta=True)`` (residual add + RMSNorm) for
    one 8-row block (B1). Arguments: x, weight, box, mc, slot, δ, base, then the consumer
    rows (prefix = residual, blocks, norm, query, out_norm, output, xmoe, hints, eps,
    out_eps); norm / query / blocks are unused at nb = 0. ``has_delta=False`` (debug taps):
    prefix receives the BF16 all-reduce sum itself and no residual is added."""

    def __init__(
        self,
        k,
        *,
        silu,
        conv=True,
        deadline_ns=0,
        has_delta=True,
        permuted=False,
        delay_ns=0,
    ):
        self.delay_ns = delay_ns  # race tests: hold the residual write back
        self.producer = DraftProducer(k, silu=silu, conv=conv, permuted=permuted)
        self.consumer = OprojConsumer(0, has_delta=has_delta, deadline_ns=deadline_ns)

    @cute.jit
    def __call__(self, *args):
        if cutlass.const_expr(args[0].shape[0] > 8):
            raise ValueError("The fused draft schedule supports one 8-row block")
        launch(self.kernel, args, (8, 14, 1), 512, 8, min_blocks_per_mp=1)

    @cute.kernel
    def kernel(self, args):
        x, weight, rows = args[0], args[1], (args[2], *args[7:])
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
        if active:
            block_mix = (mix, maximum, exponent_sum, qpart)
            if cutlass.const_expr(self.delay_ns > 0):
                sleep_ns(self.delay_ns)
            self.consumer.post(rows, state, block_mix, rank, cluster)
