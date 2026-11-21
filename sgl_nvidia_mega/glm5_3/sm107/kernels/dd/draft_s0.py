# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
import cutlass
import cutlass.cute as cute
from cutlass._mlir import ir
from cutlass._mlir.dialects import nvvm, vector
from cutlass.cute.nvgpu import cpasync
from cutlass.cutlass_dsl import T, dsl_user_op

HIDDEN = 6144
RANKS = 16
VOCAB_SHARD = 9680
VOCAB = RANKS * VOCAB_SHARD
WINDOW = 2 * HIDDEN // RANKS
RMS_EPS = 1.0e-5

EH_TILES = HIDDEN // 64
EH_ROW_BYTES = WINDOW * 2
EH_ROW_STRIDE = EH_ROW_BYTES + 64
EH_X_BASE = 64 * EH_ROW_STRIDE
EH_STEPS = WINDOW // 2 // 32

NW_WORDS = 8 * WINDOW // 2
EH_WORDS = 8 * RANKS * HIDDEN // 2
EMB_WORDS = 8 * HIDDEN // 2

NORM_CTA = 200
OWNER_CTA = 204
AR_CTA = 180
FA_STOCK = True
EH_BARS = 8
EH_B_BASE = 6 * 16384


@dsl_user_op
def tmem_alloc32_keep(dst: cutlass.Uint32, *, loc=None, ip=None):
    nvvm.tcgen05_alloc(
        as_ptr(3, dst, loc, ip),
        cutlass.Int32(32).ir_value(loc=loc, ip=ip),
        group=nvvm.CTAGroupKind.CTA_1,
        loc=loc,
        ip=ip,
    )


@cute.jit
def llvm_alloc32(tmem_slot: cute.Tensor):
    tmem_alloc32_keep(tmem_slot.iterator.toint().to(cutlass.Uint32))


@dsl_user_op
def tmem_load16(address: cutlass.Uint32, *, loc=None, ip=None):
    v = nvvm.tcgen05_ld(
        ir.VectorType.get([16], T.i32()),
        nvvm.Tcgen05LdStShape.SHAPE_32X32B,
        as_ptr(6, address, loc, ip),
        loc=loc,
        ip=ip,
    )
    nvvm.tcgen05_wait(nvvm.Tcgen05WaitKind.LOAD, loc=loc, ip=ip)
    return tuple(
        cutlass.Uint32(vector.extract(v, [], [i], loc=loc, ip=ip)) for i in range(16)
    )


@dsl_user_op
def rsqrt_approx_ftz(x: cutlass.Float32, *, loc=None, ip=None):
    return cutlass.Float32(
        f32_intrinsic("llvm.nvvm.rsqrt.approx.ftz.f", x, loc=loc, ip=ip)
    )


@cute.jit
def xnorm_publish(
    unit: cutlass.Int32,
    xv: cute.Tensor,
    gamma_in: cute.Tensor,
    xn: cute.Tensor,
    ns: cute.Tensor,
    scalars: cute.Tensor,
    mnnvl: cutlass.Constexpr,
    rstd_out=None,
):
    tid, _, _ = cute.arch.thread_idx()
    warp, lane = tid >> 5, tid & 31
    token, quarter = unit // 4, unit % 4
    col = 1536 * quarter + 8 * tid
    acc = cutlass.Float32(0.0)
    if tid < 192:
        for j in cutlass.range_constexpr(8):
            if cutlass.const_expr(mnnvl):
                acc = add_rn(acc, round_bf16(xv[j] * xv[j]))
            else:
                acc = add_rn(acc, mul_rn(xv[j], xv[j]))
    for sh in cutlass.range_constexpr(5):
        acc = add_rn(acc, cute.arch.shuffle_sync_bfly(acc, 16 >> sh))
    if lane == 0:
        scalars[warp] = acc
    cute.arch.sync_threads()
    if tid < 2:
        b = 3 * tid
        cta_sum = cutlass.Float32(0.0)
        if cutlass.const_expr(mnnvl):
            cta_sum = add_rn(add_rn(scalars[b], scalars[b + 2]), scalars[b + 1])
        else:
            cta_sum = add_rn(add_rn(scalars[b], scalars[b + 1]), scalars[b + 2])
        store_word(
            ns.iterator + token * 8 + 2 * quarter + tid,
            published_word(f32_bits(cta_sum)),
        )
    if tid == 0:
        a0, a1, a2, a3 = poll4(ns.iterator + token * 8)
        c0, c1, c2, c3 = poll4(ns.iterator + token * 8 + 4)
        cw = (a0, a1, a2, a3, c0, c1, c2, c3)
        for c in cutlass.range_constexpr(8):
            scalars[8 + c] = bits_f32(cw[c])
    cute.arch.sync_threads()
    total = scalars[8]
    for c in cutlass.range_constexpr(1, 8):
        total = add_rn(total, scalars[8 + c])
    rstd = rsqrt_approx_ftz(
        add_rn(div_rn(total, cutlass.Float32(HIDDEN)), cutlass.Float32(RMS_EPS))
    )
    if cutlass.const_expr(rstd_out is not None):
        if (tid == 0) & (quarter == 0):
            rstd_out[token] = f32_bits(rstd)
    if tid < 192:
        g0, g1, g2, g3 = readonly_vector(gamma_in.iterator + col // 2)
        gw = (g0, g1, g2, g3)
        out = cute.make_rmem_tensor((4,), cutlass.Uint32)
        for q in cutlass.range_constexpr(4):
            glo = bf16_from_word(gw[q], False)
            ghi = bf16_from_word(gw[q], True)
            if cutlass.const_expr(not mnnvl):
                glo = add_rn(glo, cutlass.Float32(0.0))
                ghi = add_rn(ghi, cutlass.Float32(0.0))
            out[q] = published_word(
                bf16_pair(
                    mul_rn(mul_rn(xv[2 * q], rstd), glo),
                    mul_rn(mul_rn(xv[2 * q + 1], rstd), ghi),
                )
            )
        store_vector(
            xn.iterator + xfrag_word_offset(token, col), out[0], out[1], out[2], out[3]
        )
    cute.arch.sync_threads()


@cute.jit
def token_valid(token_id: cutlass.Int32):
    return (token_id >= 0) & (token_id < VOCAB)


@cute.jit
def embed_owner_unit(
    t: cutlass.Int32,
    rank: cutlass.Int32,
    ids: cute.Tensor,
    embed: cute.Tensor,
    emb_mc: cute.Tensor,
):
    tid, _, _ = cute.arch.thread_idx()
    token_id = ids[t]
    if token_valid(token_id) & (token_id // VOCAB_SHARD == rank):
        local = token_id - rank * VOCAB_SHARD
        for i in cutlass.range_constexpr(3):
            word = 12 * tid + 4 * i
            a, b, c, d = readonly_vector(embed.iterator + local * (HIDDEN // 2) + word)
            store_vector(
                emb_mc.iterator + t * (HIDDEN // 2) + word,
                published_word(a),
                published_word(b),
                published_word(c),
                published_word(d),
            )


@cute.jit
def poll4(pointer: cute.Pointer):
    a, b, c, d = volatile_vector(pointer)
    waiting = (
        (a == cutlass.Uint32(EMPTY))
        | (b == cutlass.Uint32(EMPTY))
        | (c == cutlass.Uint32(EMPTY))
        | (d == cutlass.Uint32(EMPTY))
    )
    while waiting:
        a, b, c, d = volatile_vector(pointer)
        waiting = (
            (a == cutlass.Uint32(EMPTY))
            | (b == cutlass.Uint32(EMPTY))
            | (c == cutlass.Uint32(EMPTY))
            | (d == cutlass.Uint32(EMPTY))
        )
    return a, b, c, d


@cute.jit
def norm_rows_unit(
    index: cutlass.Int32,
    rank: cutlass.Int32,
    ids: cute.Tensor,
    embed: cute.Tensor,
    sharded: cutlass.Boolean,
    emb_uc: cute.Tensor,
    hidden: cute.Tensor,
    gamma_e: cute.Tensor,
    gamma_h: cute.Tensor,
    nw: cute.Tensor,
    red: cute.Tensor,
):
    tid, _, _ = cute.arch.thread_idx()
    warp, lane = tid >> 5, tid & 31
    row = 4 * index + tid // 64
    t = tid % 64
    half = rank // 8
    lo = WINDOW * (rank % 8)
    token_id = ids[row]
    valid = token_valid(token_id)
    if sharded & valid & (half == 1):
        for b in cutlass.range_constexpr(12):
            pointer = emb_uc.iterator + row * (HIDDEN // 2) + (b * 512 + 8 * t) // 2
            poll4(pointer)
            store_empty(pointer, evict_first())
    xr = cute.make_rmem_tensor((96,), cutlass.Float32)
    for b in cutlass.range_constexpr(12):
        col = b * 512 + 8 * t
        w0, w1, w2, w3 = (
            cutlass.Uint32(0),
            cutlass.Uint32(0),
            cutlass.Uint32(0),
            cutlass.Uint32(0),
        )
        if half == 0:
            if valid:
                if sharded:
                    pointer = emb_uc.iterator + row * (HIDDEN // 2) + col // 2
                    w0, w1, w2, w3 = poll4(pointer)
                    store_empty(pointer, evict_first())
                else:
                    w0, w1, w2, w3 = readonly_vector(
                        embed.iterator + token_id * (HIDDEN // 2) + col // 2
                    )
        else:
            w0, w1, w2, w3 = global_vector(
                hidden.iterator + row * (HIDDEN // 2) + col // 2
            )
        ww = (w0, w1, w2, w3)
        for q in cutlass.range_constexpr(4):
            xr[8 * b + 2 * q] = bf16_from_word(ww[q], False)
            xr[8 * b + 2 * q + 1] = bf16_from_word(ww[q], True)
    xs = xr.load()
    local = (xs * xs).reduce(
        cute.ReductionOp.ADD, init_val=cutlass.Float32(0.0), reduction_profile=0
    )
    for i in cutlass.range_constexpr(5):
        local = local + cute.arch.shuffle_sync_bfly(local, offset=1 << i)
    if lane == 0:
        red[warp] = local
    cute.arch.sync_threads()
    total = cutlass.Float32(0.0)
    if lane < 2:
        total = red[2 * (warp // 2) + lane]
    for i in cutlass.range_constexpr(5):
        total = total + cute.arch.shuffle_sync_bfly(total, offset=1 << i)
    rstd = cute.math.rsqrt(
        total / cutlass.Float32(HIDDEN) + cutlass.Float32(RMS_EPS), fastmath=True
    )
    for b in cutlass.range_constexpr(12):
        col = b * 512 + 8 * t
        if (col >= lo) & (col < lo + WINDOW):
            gw = cute.make_rmem_tensor((4,), cutlass.Uint32)
            if half == 0:
                g0, g1, g2, g3 = readonly_vector(gamma_e.iterator + col // 2)
                gw[0], gw[1], gw[2], gw[3] = g0, g1, g2, g3
            else:
                g0, g1, g2, g3 = readonly_vector(gamma_h.iterator + col // 2)
                gw[0], gw[1], gw[2], gw[3] = g0, g1, g2, g3
            out = cute.make_rmem_tensor((4,), cutlass.Uint32)
            for q in cutlass.range_constexpr(4):
                ylo = (xr[8 * b + 2 * q] * rstd) * (
                    bf16_from_word(gw[q], False) + cutlass.Float32(0.0)
                )
                yhi = (xr[8 * b + 2 * q + 1] * rstd) * (
                    bf16_from_word(gw[q], True) + cutlass.Float32(0.0)
                )
                out[q] = published_word(bf16_pair(ylo, yhi))
            store_vector(
                nw.iterator + row * (WINDOW // 2) + (col - lo) // 2,
                out[0],
                out[1],
                out[2],
                out[3],
            )
    cute.arch.sync_threads()


@cute.jit
def ar_unit(
    unit: cutlass.Int32,
    eh_uc: cute.Tensor,
    gamma_in: cute.Tensor,
    residual0: cute.Tensor,
    x_frag: cute.Tensor,
    x_ssq: cute.Tensor,
    scalars: cute.Tensor,
    xn=None,
    ns=None,
):
    tid, _, _ = cute.arch.thread_idx()
    token, quarter = unit // 4, unit % 4
    col = 1536 * quarter + 8 * tid
    ssq = cutlass.Float32(0)
    xv = cute.make_rmem_tensor((8,), cutlass.Float32)
    for j in cutlass.range_constexpr(8):
        xv[j] = cutlass.Float32(0)
    if tid < 192:
        sums = reduce_rank16(eh_uc, token, col, False)
        policy = evict_first()
        for r in cutlass.range_constexpr(RANKS):
            store_empty(
                eh_uc.iterator + (token * RANKS + r) * (HIDDEN // 2) + col // 2, policy
            )
        for j in cutlass.range_constexpr(8):
            xv[j] = round_bf16(round_bf16(sums[j]) + cutlass.Float32(0.0))
        xw = cute.make_rmem_tensor((4,), cutlass.Uint32)
        for j in cutlass.range_constexpr(4):
            xw[j] = bf16_pair(xv[2 * j], xv[2 * j + 1])
        store_vector(
            residual0.iterator + token * (HIDDEN // 2) + col // 2,
            xw[0],
            xw[1],
            xw[2],
            xw[3],
        )
        if cutlass.const_expr(xn is None):
            g0, g1, g2, g3 = readonly_vector(gamma_in.iterator + col // 2)
            gw = (g0, g1, g2, g3)
            prime = cute.make_rmem_tensor((4,), cutlass.Uint32)
            for j in cutlass.range_constexpr(4):
                lo = xv[2 * j]
                hi = xv[2 * j + 1]
                ssq += round_bf16(lo * lo)
                ssq += round_bf16(hi * hi)
                prime[j] = published_word(
                    bf16_pair(
                        round_bf16(lo * bf16_from_word(gw[j], False)),
                        round_bf16(hi * bf16_from_word(gw[j], True)),
                    )
                )
            store_vector(
                x_frag.iterator + xfrag_word_offset(token, col),
                prime[0],
                prime[1],
                prime[2],
                prime[3],
            )
    if cutlass.const_expr(xn is not None):
        xnorm_publish(unit, xv, gamma_in, xn, ns, scalars, True)
    else:
        for sh in cutlass.range_constexpr(5):
            ssq += cute.arch.shuffle_sync_bfly(ssq, 16 >> sh)
        if (tid & 31) == 0:
            scalars[tid >> 5] = ssq
        cute.arch.sync_threads()
        if tid == 0:
            partial = cutlass.Float32(0)
            for w in cutlass.range_constexpr(8):
                partial += scalars[w]
            store_word(
                x_ssq.iterator + 4 * token + quarter, published_word(f32_bits(partial))
            )
        cute.arch.sync_threads()


@cute.jit
def make_eh_atom(eh_w: cute.Tensor):
    w_bytes = cute.make_tensor(
        cute.recast_ptr(eh_w.iterator, dtype=cutlass.Uint8),
        cute.make_layout((EH_ROW_BYTES, HIDDEN), stride=(1, EH_ROW_BYTES)),
    )
    layout = cute.make_composed_layout(
        cute.make_swizzle(3, 4, 3), 0, cute.make_layout((128, 64), stride=(1, 128))
    )
    atom, _ = cpasync.make_tiled_tma_atom(
        cpasync.CopyBulkTensorTileG2SOp(), w_bytes, layout, (128, 64)
    )
    return atom


@cute.jit
def issue_eh_weights_umma(
    tile: cutlass.Int32, eh_atom: cute.CopyAtom, region: cute.Tensor, bars: cute.Tensor
):
    tid, _, _ = cute.arch.thread_idx()
    if cute.arch.make_warp_uniform(tid >> 5) == 0:
        base = cute.arch.make_warp_uniform(
            region.iterator.toint().to(cutlass.Int32)
        ).to(cutlass.Uint32)
        bar0 = cute.arch.make_warp_uniform(bars.iterator.toint().to(cutlass.Int32)).to(
            cutlass.Uint32
        )
        utile = cute.arch.make_warp_uniform(tile)
        for kt in cutlass.range_constexpr(6):
            expect_tx_elect(bar0 + cutlass.Uint32(8 * kt), 16384)
            for atom in cutlass.range_constexpr(2):
                tma_2d_elect(
                    eh_atom,
                    base + cutlass.Uint32(kt * 16384 + atom * 8192),
                    cutlass.Int32(kt * 256 + atom * 128),
                    utile * 64,
                    bar0 + cutlass.Uint32(8 * kt),
                )


@cute.jit
def eh_unit_umma(
    tile: cutlass.Int32,
    rank: cutlass.Int32,
    nw: cute.Tensor,
    region: cute.Tensor,
    red: cute.Tensor,
    bars: cute.Tensor,
    tmem_slot: cute.Tensor,
    eh_mc: cute.Tensor,
):
    tid, _, _ = cute.arch.thread_idx()
    warp, lane = tid >> 5, tid & 31
    base = region.iterator.toint().to(cutlass.Uint32)
    bbase = base + cutlass.Uint32(EH_B_BASE)
    for j in cutlass.range_constexpr(3):
        vi = tid + 256 * j
        r, wq = vi // 96, vi % 96
        k = 8 * wq
        kt, atom, chunk = k // 128, (k % 128) // 64, (k % 64) // 8
        a, b, c, d = poll4(nw.iterator + r * (WINDOW // 2) + 4 * wq)
        shared_store4(
            bbase
            + (kt * 2048 + atom * 1024 + r * 128 + ((chunk ^ r) * 16)).to(
                cutlass.Uint32
            ),
            a,
            b,
            c,
            d,
        )
    proxy_fence()
    if warp == 1:
        llvm_alloc32(tmem_slot)
    tc_before()
    cute.arch.sync_threads()
    tc_after()
    tmem = tmem_slot[0]
    if cute.arch.make_warp_uniform(warp) == 1:
        ubase = cute.arch.make_warp_uniform(base.to(cutlass.Int32)).to(cutlass.Uint32)
        utmem = cute.arch.make_warp_uniform(tmem.to(cutlass.Int32)).to(cutlass.Uint32)
        for kt in cutlass.range_constexpr(6):
            cute.arch.mbarrier_wait(bars.iterator + kt, 0)
            stage = ubase + cutlass.Uint32(kt * 16384)
            bst = ubase + cutlass.Uint32(EH_B_BASE + kt * 2048)
            for kb in cutlass.range_constexpr(8):
                acc = cutlass.Uint32(0 if (kb == 0 and kt % 3 == 0) else 1)
                umma_bf16_elect(
                    utmem + cutlass.Uint32(8 * (kt // 3)),
                    stage + cutlass.Uint32((kb // 4) * 8192 + (kb % 4) * 32),
                    bst + cutlass.Uint32((kb // 4) * 1024 + (kb % 4) * 32),
                    acc,
                )
        commit_elect(
            cute.arch.make_warp_uniform(
                (bars.iterator + 6).toint().to(cutlass.Int32)
            ).to(cutlass.Uint32)
        )
    cute.arch.mbarrier_wait(bars.iterator + 6, 0)
    tc_after()
    if warp < 4:
        v = tmem_load16(tmem + ((32 * warp) << 16).to(cutlass.Uint32))
        if lane < 16:
            row = 16 * warp + lane
            for t in cutlass.range_constexpr(8):
                red[512 + t * 64 + row] = round_bf16(
                    bits_f32(v[t]) + bits_f32(v[8 + t])
                )
    tc_before()
    cute.arch.sync_threads()
    if warp == 1:
        tc_after()
        tmem_dealloc32(tmem)
    token, pair = tid // 32, tid % 32
    word = bf16_pair(
        red[512 + token * 64 + 2 * pair], red[512 + token * 64 + 2 * pair + 1]
    )
    store_word(
        eh_mc.iterator + (token * RANKS + rank) * (HIDDEN // 2) + tile * 32 + pair,
        published_word(word),
    )
    cute.arch.sync_threads()


@cute.jit
def s0_front(
    cta: cutlass.Int32,
    rank: cutlass.Int32,
    ids: cute.Tensor,
    embed: cute.Tensor,
    sharded: cutlass.Boolean,
    emb_uc: cute.Tensor,
    emb_mc: cute.Tensor,
    hidden: cute.Tensor,
    gamma_e: cute.Tensor,
    gamma_h: cute.Tensor,
    nw: cute.Tensor,
    eh_uc: cute.Tensor,
    eh_mc: cute.Tensor,
    gamma_in: cute.Tensor,
    residual0: cute.Tensor,
    x_frag: cute.Tensor,
    x_ssq: cute.Tensor,
    region: cute.Tensor,
    red: cute.Tensor,
    scalars: cute.Tensor,
    eh_bars: cute.Tensor,
    eh_atom=None,
    tmem_slot=None,
    xn=None,
    ns=None,
):
    if cta < EH_TILES:
        issue_eh_weights_umma(cta, eh_atom, region, eh_bars)
    cute.arch.griddepcontrol_wait()
    if (cta >= OWNER_CTA) & sharded:
        embed_owner_unit(cta - OWNER_CTA, rank, ids, embed, emb_mc)
    if (cta >= NORM_CTA) & (cta < NORM_CTA + 2):
        norm_rows_unit(
            cta - NORM_CTA,
            rank,
            ids,
            embed,
            sharded,
            emb_uc,
            hidden,
            gamma_e,
            gamma_h,
            nw,
            red,
        )
    if cta < EH_TILES:
        eh_unit_umma(cta, rank, nw, region, red, eh_bars, tmem_slot, eh_mc)
    if cta >= AR_CTA:
        ar_unit(
            cta - AR_CTA, eh_uc, gamma_in, residual0, x_frag, x_ssq, scalars, xn, ns
        )
    proxy_fence()
    cute.arch.sync_threads()


from .attention import (
    EMPTY,
    bf16_from_word,
    bf16_pair,
    published_word,
    shared_store4,
    volatile_vector,
)
from .bf16_moe import add_rn, div_rn

# Imported last: the role modules import each other, and their names are used only when a kernel is traced.
from .common import (
    as_ptr,
    bits_f32,
    evict_first,
    f32_bits,
    f32_intrinsic,
    mul_rn,
    proxy_fence,
    reduce_rank16,
    store_empty,
    tc_after,
    tc_before,
    xfrag_word_offset,
)
from .gpu_publication import global_vector, readonly_vector, store_vector, store_word
from .indexer import round_bf16
from .lm_umma import (
    commit_elect,
    expect_tx_elect,
    tma_2d_elect,
    tmem_dealloc32,
    umma_bf16_elect,
)
