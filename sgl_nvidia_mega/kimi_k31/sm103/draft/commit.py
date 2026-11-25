# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the SGLang project
"""Stage 3b: the target's post-verify KDA commit as one megakernel kernel.

Today (``--enable-linear-replayssm-spec``, dflash_worker_v2 -> update_mamba_state_after_mtp_verify
-> ``commit_kda_replayssm_after_verify``): ``_prepare_dflash_mamba_commit`` (last step, track
step), ``kda_replayssm_exact_fold_kernel`` (Triton: replay the accepted prefix of the verify ring
into the fp32 checkpoint ``temporal[slot]`` in place, snapshot the track step into
``temporal[track_slot]``), then ``fused_conv_window_scatter_with_mask`` twice (conv rollback at
commit - 1, conv track snapshot). ``KdaCommit`` does all of it in one launch.

The fold is a bitwise clone of the Triton kernel as compiled (triton 3.8, identical PTX on sm_103
and sm_107). Lane l of a warp owns k = 4l..4l+3 of a column
(Triton's [4, 1] x [32, 1] layout); per step:
- norm: sq_j = k_j * k_j; (sq0 + sq2) + (sq1 + sq3); xor butterfly 16..1; + 1e-6f;
  ``sqrt.approx.ftz``; kn_j = ``div.full`` (k_j, root);
- decay: e_j = ``ex2.approx`` (gk_j * log2e f32); h'_j = h_j * e_j;
- dot: p_j = kn_j * h'_j; (p0 + p2) + (p1 + p3); xor butterfly 16..1;
- vv = (v - dot) * beta; h_j = fma(kn_j, vv, h'_j).
Every mul / add carries .rn so that ptxas cannot contract differently (Triton's SASS has no
contraction beyond the update FMA).

Grid (layers * HV, V / 64, B), 256 threads: warp w owns 8 columns of the CTA's 64. Before
``griddepcontrol.wait`` (the ring and h were written by kernels that completed before the
previous kernel's wait): warp t (< 8) normalizes step t's k and exponentiates its gate into
shared memory, and every warp loads its h columns. After the wait: commit_lens (the accept's
output) -> fold the first commit steps, snapshot, store; the CTA's share of the layer's conv
rollback / track windows (contiguous [K-1, dim] copies, KDA's dense window layout)."""

from functools import cache

import cutlass
import cutlass.cute as cute
import torch
from cutlass import Float32, Int64
from cutlass.cute.runtime import make_fake_stream

from ..k3.common.host import dynamic_view, fake, stream
from ..k3.common.primitives import (
    add_rn,
    alloc,
    asm,
    bf16,
    fma,
    launch,
    load_words,
    split,
    store_f32x4,
    store_words,
)

STEPS, LANE_K, COLUMNS, WARPS = 8, 4, 64, 8
WARP_COLUMNS = COLUMNS // WARPS
LOG2E = 1.4426950408889634  # f32 0x3FB8AA3B, Triton's exp lowering
EPS = 1e-6  # f32 0x358637BD


@asm(Float32, pure=True)
def mul_rn(a, b):
    return "mul.rn.f32 $0,$1,$2;", "=f,f,f", (Float32(a), Float32(b))


@asm(Float32, pure=True)
def sub_rn(a, b):
    return "sub.rn.f32 $0,$1,$2;", "=f,f,f", (Float32(a), Float32(b))


@asm(Float32, pure=True)
def sqrt_approx_ftz(a):
    return "sqrt.approx.ftz.f32 $0,$1;", "=f,f", (Float32(a),)


@asm(Float32, pure=True)
def div_full(a, b):
    return "div.full.f32 $0,$1,$2;", "=f,f,f", (Float32(a), Float32(b))


@asm(Float32, pure=True)
def ex2_approx(a):
    return "ex2.approx.f32 $0,$1;", "=f,f", (Float32(a),)


@cute.jit
def butterfly(value):
    """Triton's warp reduce: acc + shfl.bfly(acc, offset), offsets 16, 8, 4, 2, 1."""
    for stage in cutlass.range_constexpr(5):
        value = add_rn(value, cute.arch.shuffle_sync_bfly(value, 16 >> stage))
    return value


@cute.jit
def pair_sum(x0, x1, x2, x3):
    """Triton's in-thread tree over a lane's 4 elements: (x0 + x2) + (x1 + x3)."""
    return add_rn(add_rn(x0, x2), add_rn(x1, x3))


class KdaCommit:
    """Arguments (addresses in bytes, strides in elements): temporal, rawk, rawv, gk, beta
    addresses; their layer strides; slots [B] int32 (mamba slot, -1 = padded), commit_lens [B]
    int32, prefix_lens [B] int64, track_slots [B] int32 (-1 = no track); conv address, conv
    layer / slot strides, conv slot count; window address, window layer / request / step
    strides, window request and step counts."""

    def __init__(self, layers, heads, k_heads, dim, ring, conv_elems, track_interval):
        if dim != 128 or heads % k_heads or conv_elems % 8:
            raise ValueError("KdaCommit serves head dim 128 and 16-byte conv rows")
        self.layers, self.heads, self.k_heads, self.dim, self.ring = (
            layers,
            heads,
            k_heads,
            dim,
            ring,
        )
        self.conv_elems, self.track_interval = conv_elems, track_interval

    @cute.jit
    def __call__(self, *args):
        grid = (self.layers * self.heads, self.dim // COLUMNS, args[10].shape[0])
        launch(self.kernel, args, grid, WARPS * 32)

    @cute.kernel
    def kernel(self, args):
        temporal, rawk, rawv, gk, beta = args[0:5]
        temporal_layer, rawk_layer, rawv_layer, gk_layer, beta_layer = args[5:10]
        slots, commit_lens, prefix_lens, track_slots = args[10:14]
        conv, conv_layer, conv_slot, conv_slots = args[14:18]
        (
            window,
            window_layer,
            window_request,
            window_step,
            window_requests,
            window_steps,
        ) = args[18:24]
        head_layer, half, request = cute.arch.block_idx()
        tid, _, _ = cute.arch.thread_idx()
        warp, lane = tid // 32, tid % 32
        layer, head = head_layer // self.heads, head_layer % self.heads
        smem = cutlass.utils.SmemAllocator()
        kn = alloc(smem, Float32, (STEPS, self.dim))
        decay = alloc(smem, Float32, (STEPS, self.dim))
        values = alloc(smem, Float32, (STEPS, COLUMNS))
        betas = alloc(smem, Float32, STEPS)
        h = cute.make_rmem_tensor(WARP_COLUMNS * LANE_K, Float32)
        slot = Int64(slots[request])
        column0 = half * COLUMNS + warp * WARP_COLUMNS
        state = self.head_state(temporal, temporal_layer, layer, slot, head)
        if slot >= 0:
            self.prefetch(
                args,
                kn,
                decay,
                values,
                betas,
                h,
                layer,
                head,
                half,
                slot,
                state,
                column0,
                warp,
                lane,
            )
        cute.arch.sync_threads()
        cute.arch.griddepcontrol_wait()
        cute.arch.griddepcontrol_launch_dependents()
        commit = Int64(commit_lens[request])
        prefix = Int64(prefix_lens[request])
        track = Int64(track_slots[request])
        # _prepare_dflash_mamba_commit: the step whose post-state crosses a track interval
        interval = Int64(self.track_interval)
        post = prefix + commit
        track_step = Int64(-1)
        crossing = post // interval * interval - prefix - Int64(1)
        if crossing < Int64(0):
            crossing = Int64(0)
        if (prefix // interval != post // interval) & (crossing < commit):
            track_step = crossing
        if (slot >= 0) & (commit > 0):
            for step in cutlass.range_constexpr(STEPS):
                if Int64(step) < commit:
                    self.fold_step(h, kn, decay, values, betas, step, warp, lane)
                    if (Int64(step) == track_step) & (track >= 0):
                        self.store_columns(
                            self.head_state(
                                temporal, temporal_layer, layer, track, head
                            ),
                            h,
                            column0,
                            lane,
                        )
            self.store_columns(state, h, column0, lane)
        # conv: this CTA's share of the layer's [K-1, dim] window (rollback, then track)
        parts = self.heads * (self.dim // COLUMNS)
        first, count = split(
            self.conv_elems // 8, head * (self.dim // COLUMNS) + half, parts
        )
        last = commit - Int64(1)
        source = (
            window + (Int64(layer) * window_layer + Int64(request) * window_request) * 2
        )
        fits = Int64(request) < Int64(window_requests)
        if (
            (last >= 0)
            & (slot >= 0)
            & (slot < Int64(conv_slots))
            & fits
            & (last < Int64(window_steps))
        ):
            self.copy_window(
                conv + (Int64(layer) * conv_layer + slot * conv_slot) * 2,
                source + last * window_step * 2,
                first,
                count,
                tid,
            )
        if (
            (track_step >= 0)
            & (track >= 0)
            & (track < Int64(conv_slots))
            & fits
            & (track_step < Int64(window_steps))
        ):
            self.copy_window(
                conv + (Int64(layer) * conv_layer + track * conv_slot) * 2,
                source + track_step * window_step * 2,
                first,
                count,
                tid,
            )

    @cute.jit
    def head_state(self, temporal, temporal_layer, layer, slot, head):
        """Byte address of temporal[layer, slot, head]."""
        return (
            temporal
            + (
                Int64(layer) * temporal_layer
                + (slot * self.heads + head) * self.dim * self.dim
            )
            * 4
        )

    @cute.jit
    def prefetch(
        self,
        args,
        kn,
        decay,
        values,
        betas,
        h,
        layer,
        head,
        half,
        slot,
        state,
        column0,
        warp,
        lane,
    ):
        """Warp t: ring step t's normalized k, decay, v columns and beta into shared memory;
        every warp: its h columns into registers."""
        rawk, rawv, gk, beta = args[1:5]
        rawk_layer, rawv_layer, gk_layer, beta_layer = args[6:10]
        K, R = self.dim, self.ring
        k_head = head // (self.heads // self.k_heads)
        row = (slot * self.heads + head) * R + warp
        ring_k = (
            rawk
            + (
                Int64(layer) * rawk_layer
                + ((slot * self.k_heads + k_head) * R + warp) * K
            )
            * 2
        )
        words = load_words(ring_k + Int64(lane) * 8, count=2, mutable=True)
        k = (bf16(words[0]), bf16(words[0], True), bf16(words[1]), bf16(words[1], True))
        total = butterfly(
            pair_sum(
                mul_rn(k[0], k[0]),
                mul_rn(k[1], k[1]),
                mul_rn(k[2], k[2]),
                mul_rn(k[3], k[3]),
            )
        )
        root = sqrt_approx_ftz(add_rn(total, Float32(EPS)))
        ring_g = gk + (Int64(layer) * gk_layer + row * K) * 4
        g = load_words(
            ring_g + Int64(lane) * 16,
            count=4,
            result_type=Float32,
            word_type="f32",
            mutable=True,
        )
        for j in cutlass.range_constexpr(LANE_K):
            kn[warp, lane * LANE_K + j] = div_full(k[j], root)
            decay[warp, lane * LANE_K + j] = ex2_approx(mul_rn(g[j], Float32(LOG2E)))
        ring_v = (
            rawv + (Int64(layer) * rawv_layer + row * K + half * COLUMNS + 2 * lane) * 2
        )
        pair = load_words(ring_v, count=1, mutable=True)[0]
        values[warp, 2 * lane] = bf16(pair)
        values[warp, 2 * lane + 1] = bf16(pair, True)
        if lane == 0:
            ring_beta = beta + (Int64(layer) * beta_layer + row) * 4
            betas[warp] = load_words(
                ring_beta, count=1, result_type=Float32, word_type="f32", mutable=True
            )[0]
        for c in cutlass.range_constexpr(WARP_COLUMNS):
            got = load_words(
                state + (Int64(column0 + c) * K + lane * LANE_K) * 4,
                count=4,
                result_type=Float32,
                word_type="f32",
                mutable=True,
            )
            for j in cutlass.range_constexpr(LANE_K):
                h[c * LANE_K + j] = got[j]

    @cute.jit
    def fold_step(
        self, h, kn, decay, values, betas, step: cutlass.Constexpr[int], warp, lane
    ):
        """One recurrent step of kda_replayssm_exact_fold_kernel over the warp's columns."""
        n = tuple(kn[step, lane * LANE_K + j] for j in range(LANE_K))
        e = tuple(decay[step, lane * LANE_K + j] for j in range(LANE_K))
        b = betas[step]
        for c in cutlass.range_constexpr(WARP_COLUMNS):
            hp = tuple(mul_rn(h[c * LANE_K + j], e[j]) for j in range(LANE_K))
            dot = butterfly(pair_sum(*(mul_rn(n[j], hp[j]) for j in range(LANE_K))))
            vv = mul_rn(sub_rn(values[step, warp * WARP_COLUMNS + c], dot), b)
            for j in cutlass.range_constexpr(LANE_K):
                h[c * LANE_K + j] = fma(n[j], vv, hp[j])

    @cute.jit
    def store_columns(self, base, h, column0, lane):
        for c in cutlass.range_constexpr(WARP_COLUMNS):
            address = base + (Int64(column0 + c) * self.dim + lane * LANE_K) * 4
            store_f32x4(address, tuple(h[c * LANE_K + j] for j in range(LANE_K)))

    @cute.jit
    def copy_window(self, destination, source, first, count, tid):
        for i in cutlass.range(tid, count, WARPS * 32, unroll=1):
            offset = Int64(first + i) * 16
            store_words(
                destination + offset, load_words(source + offset, count=4, mutable=True)
            )


@cache
def compile_kda_commit(layers, heads, k_heads, dim, ring, conv_elems, track_interval):
    # The request count is symbolic: one compile serves every batch the host commits (an
    # unpadded bs keyed the cache and compiled at serving time).
    vec32 = fake((cute.sym_int(),), (1,), torch.int32, 4)
    vec64 = fake((cute.sym_int(),), (1,), torch.int64, 8)
    i64, i32 = cutlass.Int64(0), cutlass.Int32(0)
    args = (
        (i64,) * 10
        + (vec32, vec32, vec64, vec32)
        + (i64, i64, i64, i32)
        + (i64, i64, i64, i64, i32, i32)
    )
    kernel = KdaCommit(layers, heads, k_heads, dim, ring, conv_elems, track_interval)
    return cute.compile(kernel, *args, make_fake_stream())


def kda_commit(
    *,
    temporal,
    rawk,
    rawv,
    gk,
    beta,
    conv,
    window,
    slots,
    commit_lens,
    prefix_lens,
    track_slots,
    track_interval,
):
    """Launch KdaCommit on the current stream. Tensors as the MambaPool holds them: temporal
    [layers, slots, HV, V, K] fp32, rawk [layers, slots, H, L, K] / rawv [layers, slots, HV, L, V]
    bf16, gk [layers, slots, HV, L, K] / beta [layers, slots, HV, L] fp32, conv [layers, slots,
    K-1, dim] and its intermediate window [layers, requests, D, K-1, dim] (KDA's dense layout)."""
    layers, _, heads, dim, _ = temporal.shape
    k_heads, ring = rawk.shape[2], rawk.shape[3]
    conv_elems = conv.shape[2] * conv.shape[3]
    layouts = (
        all(t[0].is_contiguous() for t in (temporal, rawk, rawv, gk, beta, conv))
        and window[0, 0].is_contiguous()
        and temporal.shape[4] == dim
        and ring >= STEPS
        and window.shape[3:] == conv.shape[2:]
        and window.shape[2] <= STEPS
    )
    dtypes = (
        rawk.dtype == rawv.dtype == conv.dtype == window.dtype == torch.bfloat16
        and temporal.dtype == gk.dtype == beta.dtype == torch.float32
        and slots.dtype == commit_lens.dtype == track_slots.dtype == torch.int32
        and prefix_lens.dtype == torch.int64
    )
    if not (layouts and dtypes):
        raise ValueError(
            "KdaCommit needs the MambaPool's per-slot layouts and int32 / int64 indices"
        )
    code = compile_kda_commit(
        layers, heads, k_heads, dim, ring, conv_elems, track_interval
    )
    address = lambda t: cutlass.Int64(t.data_ptr())
    stride = lambda t, i=0: cutlass.Int64(t.stride(i))
    args = tuple(address(t) for t in (temporal, rawk, rawv, gk, beta))
    args += tuple(stride(t) for t in (temporal, rawk, rawv, gk, beta))
    args += (
        dynamic_view(slots, 4),
        dynamic_view(commit_lens, 4),
        dynamic_view(prefix_lens, 8),
        dynamic_view(track_slots, 4),
    )
    args += (address(conv), stride(conv), stride(conv, 1), cutlass.Int32(conv.shape[1]))
    args += (
        address(window),
        stride(window),
        stride(window, 1),
        stride(window, 2),
        cutlass.Int32(window.shape[1]),
        cutlass.Int32(window.shape[2]),
    )
    code(*args, stream(temporal.device))
