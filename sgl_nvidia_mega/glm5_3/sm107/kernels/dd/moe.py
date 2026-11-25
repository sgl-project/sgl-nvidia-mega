# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
import cutlass
import cutlass.cute as cute
from cutlass._mlir import ir
from cutlass._mlir.dialects import llvm, vector
from cutlass.cutlass_dsl import T, dsl_user_op

HIDDEN = 6144
INTER = 768
GU_TILES = INTER // 8
GU_UNITS = 2 * GU_TILES
GU_K = HIDDEN // 2
GU_VWARPS = 16
GU_STEPS = 6
GU_ROW_BYTES = GU_K * 2
GU_ROW_STRIDE = GU_ROW_BYTES + 64
GU_REGION_BYTES = 16 * GU_ROW_STRIDE

# Two 16-row tiles per down unit, so a gate/up unit and a down unit fit one CTA's region together.
DOWN_TILES = 2
DOWN_ROWS = 16 * DOWN_TILES
DOWN_WARPS = 8
DOWN_STEPS = 3
DOWN_ROW_BYTES = INTER * 2
DOWN_ROW_STRIDE = DOWN_ROW_BYTES + 64
DOWN_REGION_BYTES = DOWN_ROWS * DOWN_ROW_STRIDE

XK_WORDS = GU_TILES * 64 * 2
H_WORDS = 8 * (INTER // 2)
REGION_BYTES = GU_REGION_BYTES + DOWN_REGION_BYTES
RED_FLOATS = 4096


@dsl_user_op
def swiglu_ref(gate: cutlass.Float32, up: cutlass.Float32, *, loc=None, ip=None):
    def op(name, *values):
        return f32_intrinsic(name, *values, loc=loc, ip=ip)

    one = f32_const(1.0, loc, ip)
    # flushing subnormal inputs or results cannot change 1 / (1 + t), so the ftz form gives the same bits
    t = op(
        "llvm.nvvm.ex2.approx.ftz.f",
        op("llvm.nvvm.mul.rn.f", gate, f32_const(-1.4426950216293335, loc, ip)),
    )
    t = op("llvm.nvvm.div.approx.f", one, op("llvm.nvvm.add.rn.f", t, one))
    return cutlass.Float32(
        op("llvm.nvvm.mul.rn.f", up, op("llvm.nvvm.mul.rn.f", gate, t))
    )


@cute.jit
def bf16_round(value: cutlass.Float32):
    return bits_f32(bf16_pair(value, value) << 16)


@cute.jit
def issue_dense_gu(
    unit: cutlass.Int32, w_gu: cute.Tensor, region: cute.Tensor, bar: cute.Pointer
):
    """Weights never depend on activations, so callers issue this at kernel entry."""
    tid, _, _ = cute.arch.thread_idx()
    tile, khalf = unit // 2, unit % 2
    base = region.iterator.toint().to(cutlass.Uint32)
    if tid == 0:
        cute.arch.mbarrier_arrive_and_expect_tx(bar, 16 * GU_ROW_BYTES)
    cute.arch.sync_threads()
    if tid < 16:
        col = tile * 8 + (tid & 7)
        row = col
        if tid >= 8:
            row = INTER + col
        bulk_copy(
            base + cutlass.Uint32(tid * GU_ROW_STRIDE),
            w_gu.iterator + row * HIDDEN + khalf * GU_K,
            cutlass.Uint32(GU_ROW_BYTES),
            bar,
            evict_first(),
        )


@cute.jit
def issue_dense_down(
    unit: cutlass.Int32, w_dn: cute.Tensor, region: cute.Tensor, bar: cute.Pointer
):
    tid, _, _ = cute.arch.thread_idx()
    base = region.iterator.toint().to(cutlass.Uint32) + cutlass.Uint32(GU_REGION_BYTES)
    if tid == 0:
        cute.arch.mbarrier_arrive_and_expect_tx(bar, DOWN_ROWS * DOWN_ROW_BYTES)
    cute.arch.sync_threads()
    if tid < DOWN_ROWS:
        bulk_copy(
            base + cutlass.Uint32(tid * DOWN_ROW_STRIDE),
            w_dn.iterator + (unit * DOWN_ROWS + tid) * INTER,
            cutlass.Uint32(DOWN_ROW_BYTES),
            bar,
            evict_first(),
        )


@cute.jit
def poll_words(words: cute.Tensor, offset: cutlass.Int32):
    a, b, c, d = volatile_vector(words.iterator + offset)
    waiting = (
        (a == cutlass.Uint32(EMPTY))
        | (b == cutlass.Uint32(EMPTY))
        | (c == cutlass.Uint32(EMPTY))
        | (d == cutlass.Uint32(EMPTY))
    )
    while waiting:
        a, b, c, d = volatile_vector(words.iterator + offset)
        waiting = (
            (a == cutlass.Uint32(EMPTY))
            | (b == cutlass.Uint32(EMPTY))
            | (c == cutlass.Uint32(EMPTY))
            | (d == cutlass.Uint32(EMPTY))
        )
    return a, b, c, d


@cute.jit
def dense_gu(
    unit: cutlass.Int32,
    x: cute.Tensor,
    region: cute.Tensor,
    red: cute.Tensor,
    xk: cute.Tensor,
    h: cute.Tensor,
    rows: cutlass.Int32,
    bar: cute.Pointer,
    poll: cutlass.Constexpr,
):
    tid, _, _ = cute.arch.thread_idx()
    warp, lane = tid >> 5, tid & 31
    group, tig = lane >> 2, lane & 3
    tile, khalf = unit // 2, unit % 2
    base = region.iterator.toint().to(cutlass.Uint32)
    has_row = group < rows
    acc_a = cute.make_rmem_tensor((2, 4), cutlass.Float32)
    acc_b = cute.make_rmem_tensor((2, 4), cutlass.Float32)
    xv = cute.make_rmem_tensor((2, GU_STEPS, 4), cutlass.Uint32)
    for v in cutlass.range_constexpr(2):
        for e in cutlass.range_constexpr(4):
            acc_a[v, e] = cutlass.Float32(0)
            acc_b[v, e] = cutlass.Float32(0)
    for v in cutlass.range_constexpr(2):
        vw = warp + 8 * v
        for s in cutlass.range_constexpr(GU_STEPS):
            k = khalf * GU_K + vw * (GU_STEPS * 32) + s * 32 + tig * 8
            w0 = cutlass.Uint32(0)
            w1 = cutlass.Uint32(0)
            w2 = cutlass.Uint32(0)
            w3 = cutlass.Uint32(0)
            if has_row:
                if cutlass.const_expr(poll):
                    w0, w1, w2, w3 = poll_words(x, group * (HIDDEN // 2) + k // 2)
                else:
                    w0, w1, w2, w3 = volatile_vector(
                        x.iterator + group * (HIDDEN // 2) + k // 2
                    )
            xv[v, s, 0], xv[v, s, 1], xv[v, s, 2], xv[v, s, 3] = w0, w1, w2, w3
    cute.arch.mbarrier_wait(bar, 0)
    for s in cutlass.range_constexpr(GU_STEPS):
        for v in cutlass.range_constexpr(2):
            vw = warp + 8 * v
            seg = base + cutlass.Uint32(vw * (GU_STEPS * 32 * 2) + tig * 16 + s * 64)
            lo0, lo1, lo2, lo3 = lds128(seg + cutlass.Uint32(group * GU_ROW_STRIDE))
            hi0, hi1, hi2, hi3 = lds128(
                seg + cutlass.Uint32((group + 8) * GU_ROW_STRIDE)
            )
            c0, c1, c2, c3 = mma_bf16(
                acc_a[v, 0],
                acc_a[v, 1],
                acc_a[v, 2],
                acc_a[v, 3],
                lo0,
                hi0,
                lo1,
                hi1,
                xv[v, s, 0],
                xv[v, s, 1],
            )
            acc_a[v, 0], acc_a[v, 1], acc_a[v, 2], acc_a[v, 3] = c0, c1, c2, c3
            d0, d1, d2, d3 = mma_bf16(
                acc_b[v, 0],
                acc_b[v, 1],
                acc_b[v, 2],
                acc_b[v, 3],
                lo2,
                hi2,
                lo3,
                hi3,
                xv[v, s, 2],
                xv[v, s, 3],
            )
            acc_b[v, 0], acc_b[v, 1], acc_b[v, 2], acc_b[v, 3] = d0, d1, d2, d3
    for v in cutlass.range_constexpr(2):
        vw = warp + 8 * v
        for e in cutlass.range_constexpr(4):
            red[(vw * 4 + e) * 32 + lane] = acc_a[v, e] + acc_b[v, e]
    cute.arch.sync_threads()
    if tid < 64:
        e, l = tid // 32, tid % 32
        v0 = cutlass.Float32(0)
        v1 = cutlass.Float32(0)
        for w in cutlass.range_constexpr(GU_VWARPS):
            v0 += red[(w * 4 + e) * 32 + l]
            v1 += red[(w * 4 + e + 2) * 32 + l]
        slot = (tile * 64 + tid) * 2
        if khalf == 1:
            store_word(xk.iterator + slot, published_word(f32_bits(v0)))
            store_word(xk.iterator + slot + 1, published_word(f32_bits(v1)))
        else:
            r0, r1 = poll_pair(xk, slot)
            v0 += bits_f32(r0)
            v1 += bits_f32(r1)
            col, token = l // 4, 2 * (l % 4) + e
            red[2048 + token * 8 + col] = swiglu_ref(bf16_round(v0), bf16_round(v1))
    cute.arch.sync_threads()
    if khalf == 0:
        if tid < 32:
            token, pair = tid // 4, tid % 4
            word = bf16_pair(
                red[2048 + token * 8 + 2 * pair], red[2048 + token * 8 + 2 * pair + 1]
            )
            store_word(
                h.iterator + token * (INTER // 2) + tile * 4 + pair,
                published_word(word),
            )
    cute.arch.sync_threads()


@dsl_user_op
def volatile_pair_words(pointer: cute.Pointer, *, loc=None, ip=None):
    v = llvm.load(
        ir.VectorType.get([2], T.i32()),
        as_ptr(1, pointer, loc, ip),
        volatile_=True,
        alignment=8,
        loc=loc,
        ip=ip,
    )
    return tuple(
        cutlass.Uint32(vector.extract(v, [], [i], loc=loc, ip=ip)) for i in range(2)
    )


@cute.jit
def poll_pair(words: cute.Tensor, slot: cutlass.Int32):
    a, b = volatile_pair_words(words.iterator + slot)
    waiting = (a == cutlass.Uint32(EMPTY)) | (b == cutlass.Uint32(EMPTY))
    while waiting:
        a, b = volatile_pair_words(words.iterator + slot)
        waiting = (a == cutlass.Uint32(EMPTY)) | (b == cutlass.Uint32(EMPTY))
    return a, b


@cute.jit
def dense_down(
    unit: cutlass.Int32,
    h: cute.Tensor,
    region: cute.Tensor,
    red: cute.Tensor,
    out: cute.Tensor,
    rank: cutlass.Int32,
    rows: cutlass.Int32,
    bar: cute.Pointer,
):
    tid, _, _ = cute.arch.thread_idx()
    warp, lane = tid >> 5, tid & 31
    group, tig = lane >> 2, lane & 3
    base = region.iterator.toint().to(cutlass.Uint32) + cutlass.Uint32(GU_REGION_BYTES)
    has_row = group < rows
    xv = cute.make_rmem_tensor((DOWN_STEPS, 4), cutlass.Uint32)
    for s in cutlass.range_constexpr(DOWN_STEPS):
        k = warp * (DOWN_STEPS * 32) + s * 32 + tig * 8
        w0 = cutlass.Uint32(0)
        w1 = cutlass.Uint32(0)
        w2 = cutlass.Uint32(0)
        w3 = cutlass.Uint32(0)
        if has_row:
            w0, w1, w2, w3 = poll_words(h, group * (INTER // 2) + k // 2)
        xv[s, 0], xv[s, 1], xv[s, 2], xv[s, 3] = w0, w1, w2, w3
    acc_a = cute.make_rmem_tensor((DOWN_TILES, 4), cutlass.Float32)
    acc_b = cute.make_rmem_tensor((DOWN_TILES, 4), cutlass.Float32)
    for t in cutlass.range_constexpr(DOWN_TILES):
        for e in cutlass.range_constexpr(4):
            acc_a[t, e] = cutlass.Float32(0)
            acc_b[t, e] = cutlass.Float32(0)
    cute.arch.mbarrier_wait(bar, 0)
    seg = base + cutlass.Uint32(warp * (DOWN_STEPS * 32 * 2) + tig * 16)
    for s in cutlass.range_constexpr(DOWN_STEPS):
        for t in cutlass.range_constexpr(DOWN_TILES):
            row_lo = t * 16 + group
            lo0, lo1, lo2, lo3 = lds128(
                seg + cutlass.Uint32(row_lo * DOWN_ROW_STRIDE + s * 64)
            )
            hi0, hi1, hi2, hi3 = lds128(
                seg + cutlass.Uint32((row_lo + 8) * DOWN_ROW_STRIDE + s * 64)
            )
            c0, c1, c2, c3 = mma_bf16(
                acc_a[t, 0],
                acc_a[t, 1],
                acc_a[t, 2],
                acc_a[t, 3],
                lo0,
                hi0,
                lo1,
                hi1,
                xv[s, 0],
                xv[s, 1],
            )
            acc_a[t, 0], acc_a[t, 1], acc_a[t, 2], acc_a[t, 3] = c0, c1, c2, c3
            d0, d1, d2, d3 = mma_bf16(
                acc_b[t, 0],
                acc_b[t, 1],
                acc_b[t, 2],
                acc_b[t, 3],
                lo2,
                hi2,
                lo3,
                hi3,
                xv[s, 2],
                xv[s, 3],
            )
            acc_b[t, 0], acc_b[t, 1], acc_b[t, 2], acc_b[t, 3] = d0, d1, d2, d3
    for t in cutlass.range_constexpr(DOWN_TILES):
        for e in cutlass.range_constexpr(4):
            red[(warp * DOWN_TILES * 4 + t * 4 + e) * 32 + lane] = (
                acc_a[t, e] + acc_b[t, e]
            )
    cute.arch.sync_threads()
    for rep in cutlass.range_constexpr((DOWN_TILES * 128 + 255) // 256):
        j = tid + 256 * rep
        if j < DOWN_TILES * 128:
            q, l = j // 32, j % 32
            v0 = cutlass.Float32(0)
            for w in cutlass.range_constexpr(DOWN_WARPS):
                v0 += red[(w * DOWN_TILES * 4 + q) * 32 + l]
            col = (q // 4) * 16 + l // 4 + ((q % 4) // 2) * 8
            token = 2 * (l % 4) + q % 2
            red[DOWN_WARPS * DOWN_TILES * 128 + token * DOWN_ROWS + col] = v0
    cute.arch.sync_threads()
    vectors = DOWN_ROWS // 8
    if tid < 8 * vectors:
        token, vec = tid // vectors, tid % vectors
        stage = DOWN_WARPS * DOWN_TILES * 128 + token * DOWN_ROWS + 8 * vec
        words = cute.make_rmem_tensor((4,), cutlass.Uint32)
        for i in cutlass.range_constexpr(4):
            words[i] = published_word(
                bf16_pair(red[stage + 2 * i], red[stage + 2 * i + 1])
            )
        store_vector(
            out.iterator
            + (token * 16 + rank) * (HIDDEN // 2)
            + (unit * DOWN_ROWS + 8 * vec) // 2,
            words[0],
            words[1],
            words[2],
            words[3],
        )
    cute.arch.sync_threads()


@dsl_user_op
def volatile_word(pointer: cute.Pointer, *, loc=None, ip=None):
    return cutlass.Uint32(
        llvm.load(
            T.i32(),
            as_ptr(1, pointer, loc, ip),
            volatile_=True,
            alignment=4,
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def redux_max(value: cutlass.Uint32, *, loc=None, ip=None):
    return cutlass.Uint32(cute.arch.warp_redux_sync(value, "umax", loc=loc, ip=ip))


CTAS = 212
CUT_TOTAL = 1112
PIECE_BYTES = 55296
PIECE_WORDS = PIECE_BYTES // 4
FC2_WORKERS = 164


@cute.jit
def _fc1_piece_body(
    tmem: cutlass.Uint32,
    ring_base: cutlass.Uint32,
    hbq: cutlass.Uint32,
    sfa: cutlass.Uint32,
    sfb: cutlass.Uint32,
    accumulate_first: cutlass.Uint32,
    bar: cutlass.Uint32,
):
    tc_after()
    with cute.arch.elect_one():
        for step in cutlass.range_constexpr(12):
            copy_scales(
                sfa + cutlass.Uint32(4 * step),
                ring_base + cutlass.Uint32(49152 + 512 * step),
            )
        for step in cutlass.range_constexpr(12):
            accumulate = cutlass.Uint32(1)
            if (step == 0) & (accumulate_first == cutlass.Uint32(0)):
                accumulate = cutlass.Uint32(0)
            mma_nvfp4(
                tmem,
                ring_base + cutlass.Uint32((step >> 1) * 8192 + 32 * (step & 1)),
                hbq + cutlass.Uint32((step >> 1) * 512 + 32 * (step & 1)),
                sfa + cutlass.Uint32(4 * step),
                sfb + cutlass.Uint32(2 * step),
                accumulate,
            )
        commit_mma_addr(bar)
    cute.arch.sync_warp()


@cute.jit
def fc1_piece(
    tmem: cutlass.Uint32,
    ring_base: cutlass.Uint32,
    hbq: cutlass.Uint32,
    sfa: cutlass.Uint32,
    sfb: cutlass.Uint32,
    accumulate_first: cutlass.Boolean,
    bar: cute.Pointer,
):
    flag = cutlass.Uint32(0)
    if accumulate_first:
        flag = cutlass.Uint32(1)
    call_outlined(
        "glm52_fc1_piece",
        _fc1_piece_body,
        tmem,
        ring_base,
        hbq,
        sfa,
        sfb,
        flag,
        bar.toint().to(cutlass.Uint32),
    )


@cute.jit
def _fc2_piece_body(
    tmem: cutlass.Uint32,
    acc: cutlass.Uint32,
    ring_base: cutlass.Uint32,
    fc2b: cutlass.Uint32,
    fc2sfb: cutlass.Uint32,
    sfa: cutlass.Uint32,
    bar: cutlass.Uint32,
):
    tc_after()
    with cute.arch.elect_one():
        copy_scales(tmem + cutlass.Uint32(448), fc2sfb)
        for step in cutlass.range_constexpr(12):
            copy_scales(
                sfa + cutlass.Uint32(4 * step),
                ring_base + cutlass.Uint32(49152 + 512 * step),
            )
        for step in cutlass.range_constexpr(12):
            mma_nvfp4(
                acc + cutlass.Uint32(8 * (step >> 1)),
                ring_base + cutlass.Uint32((step >> 1) * 8192 + 32 * (step & 1)),
                fc2b + cutlass.Uint32(32 * (step & 1)),
                sfa + cutlass.Uint32(4 * step),
                tmem + cutlass.Uint32(448 + 2 * (step & 1)),
                cutlass.Uint32(step & 1),
            )
        commit_mma_addr(bar)
    cute.arch.sync_warp()


@cute.jit
def fc2_piece(
    tmem: cutlass.Uint32,
    ring_base: cutlass.Uint32,
    fc2b: cutlass.Uint32,
    fc2sfb: cutlass.Uint32,
    sfa: cutlass.Uint32,
    bar: cute.Pointer,
    acc=None,
):
    """One shared function for the real pieces and warm_moe, so the warm-up fetches the code the first real FC2
    piece runs."""
    if cutlass.const_expr(acc is None):
        acc = tmem
    call_outlined(
        "glm52_fc2_piece",
        _fc2_piece_body,
        tmem,
        cutlass.Uint32(acc),
        ring_base,
        fc2b,
        fc2sfb,
        sfa,
        bar.toint().to(cutlass.Uint32),
    )


@cute.jit
def warm_moe(region: cute.Tensor, tmem: cutlass.Uint32, bar: cute.Pointer):
    tid, _, _ = cute.arch.thread_idx()
    warp = tid // 32
    base = region.iterator.toint().to(cutlass.Uint32)
    proxy_fence()
    tc_before()
    cute.arch.sync_threads()
    tc_after()
    if warp == 1:
        fc1_piece(
            tmem,
            base,
            base + cutlass.Uint32(165888),
            tmem + cutlass.Uint32(64),
            tmem + cutlass.Uint32(256),
            cutlass.Boolean(False),
            bar,
        )
    cute.arch.mbarrier_wait(bar, 0)
    cute.arch.sync_threads()
    tc_after()
    if warp == 1:
        fc2_piece(
            tmem,
            base + cutlass.Uint32(55296),
            base + cutlass.Uint32(198656),
            base + cutlass.Uint32(199168),
            tmem + cutlass.Uint32(112),
            bar,
        )
    cute.arch.mbarrier_wait(bar, 1)
    tc_before()
    cute.arch.sync_threads()
    tc_after()


SHARED_BYTES = 223232
SHARED_WORDS = SHARED_BYTES // 4


@cute.jit
def shared_views(storage: cute.Tensor):
    region = cute.make_tensor(storage.iterator, cute.make_layout((50176,)))
    red = cute.make_tensor(
        cute.recast_ptr(storage.iterator + 50176, dtype=cutlass.Float32),
        cute.make_layout((4096,)),
    )
    residual = cute.make_tensor(storage.iterator + 54272, cute.make_layout((768,)))
    scalars = cute.make_tensor(
        cute.recast_ptr(storage.iterator + 55168, dtype=cutlass.Float32),
        cute.make_layout((128,)),
    )
    moe = cute.make_tensor(storage.iterator + 55296, cute.make_layout((240,)))
    bars = cute.make_tensor(
        cute.recast_ptr(storage.iterator + 55040, dtype=cutlass.Uint64),
        cute.make_layout((32,)),
    )
    tmem_slot = cute.make_tensor(storage.iterator + 55288, cute.make_layout((1,)))
    return region, red, residual, scalars, moe, bars, tmem_slot


from .attention import EMPTY, bf16_pair, mma_bf16, published_word, volatile_vector

# Imported last: the role modules import each other, and their names are used only when a kernel is traced.
from .common import (
    as_ptr,
    bits_f32,
    bulk_copy,
    commit_mma_addr,
    copy_scales,
    evict_first,
    f32_bits,
    f32_const,
    f32_intrinsic,
    mma_nvfp4,
    proxy_fence,
    tc_after,
    tc_before,
)
from .gpu_publication import lds128, store_vector, store_word
from .plain import call_outlined
