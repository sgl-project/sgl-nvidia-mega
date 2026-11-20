# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
import cutlass
import cutlass.cute as cute
from cutlass._mlir import ir
from cutlass._mlir.dialects import arith, llvm, nvvm, vector
from cutlass.cutlass_dsl import T, dsl_user_op

HIDDEN = 6144
VOCAB_SHARD = 9680
TILES = (VOCAB_SHARD + 15) // 16
_CFG = [int(v) for v in "1536,3,1,0,3,0".split(",")]
TILES_PER_CTA = _CFG[4] if len(_CFG) > 4 else 3
LM_CTAS = (TILES + TILES_PER_CTA - 1) // TILES_PER_CTA
W_EVICT_FIRST = len(_CFG) < 6 or _CFG[5] != 0
CHUNK_K = _CFG[0]
CHUNKS = HIDDEN // CHUNK_K
STAGES = _CFG[1]
X_SMEM = _CFG[2] != 0
X_MODE = _CFG[6] if len(_CFG) > 6 else 0
X_CLUSTER = 4 if X_MODE == 2 else 1
X_STAGES = 2 if X_MODE == 0 else HIDDEN // _CFG[0]
ROW_BYTES = CHUNK_K * 2
ROW_STRIDE = ROW_BYTES + 64
STAGE_BYTES = 16 * ROW_STRIDE
X_STAGE_BYTES = 8 * ROW_STRIDE
X_BASE = STAGES * STAGE_BYTES
RED_FLOATS = TILES_PER_CTA * 1024 + 16
RED_KEYS = TILES_PER_CTA * 1024
REL_SYNC = len(_CFG) < 4 or _CFG[3] == 0
KEY_STRIDE = _CFG[10] if len(_CFG) > 10 else 1
RED_ALIAS = len(_CFG) > 7 and _CFG[7] != 0
FP8_KEEP = False
FP8_ELEMS = VOCAB_SHARD * HIDDEN + (VOCAB_SHARD * HIDDEN) // 2 + 2 * VOCAB_SHARD
FP4_ELEMS = (
    VOCAB_SHARD * HIDDEN
    + (VOCAB_SHARD * (HIDDEN // 2 + HIDDEN // 16) + 4 * VOCAB_SHARD) // 2
)
W8_STRIDE = CHUNK_K + 64
REGION_BYTES = X_BASE + (X_STAGES * X_STAGE_BYTES if X_SMEM else 0)
STAGES8 = STAGES
W8_STAGE_BYTES = 16 * (CHUNK_K + 64)
BARS = STAGES + X_STAGES + (STAGES8 - STAGES)
LOADS = TILES_PER_CTA * CHUNKS
STEPS = CHUNK_K // (8 * 32)
KEEP_CHUNKS = 0


@dsl_user_op
def lds64w(address: cutlass.Uint32, *, loc=None, ip=None):
    v = llvm.load(
        ir.VectorType.get([2], T.i32()),
        as_ptr(3, address, loc, ip),
        alignment=8,
        loc=loc,
        ip=ip,
    )
    return tuple(
        cutlass.Uint32(vector.extract(v, [], [i], loc=loc, ip=ip)) for i in range(2)
    )


@dsl_user_op
def e4m3x4_to_f16x2(word: cutlass.Uint32, *, loc=None, ip=None):
    halves = llvm.bitcast(
        ir.VectorType.get([2], T.i16()), word.ir_value(loc=loc, ip=ip), loc=loc, ip=ip
    )
    pairs = (
        e4m3x2_to_f16x2_ir(vector.extract(halves, [], [i], loc=loc, ip=ip), loc, ip)
        for i in range(2)
    )
    return tuple(
        cutlass.Uint32(llvm.bitcast(T.i32(), p, loc=loc, ip=ip)) for p in pairs
    )


@dsl_user_op
def bf16x2_to_f16x2(word: cutlass.Uint32, *, loc=None, ip=None):
    lo = llvm.bitcast(
        T.f32(), (word << cutlass.Uint32(16)).ir_value(loc=loc, ip=ip), loc=loc, ip=ip
    )
    hi = llvm.bitcast(
        T.f32(),
        (word & cutlass.Uint32(0xFFFF0000)).ir_value(loc=loc, ip=ip),
        loc=loc,
        ip=ip,
    )
    pair = nvvm.convert_f32x2_to_f16x2(
        ir.VectorType.get([2], ir.F16Type.get()),
        hi,
        lo,
        rnd=nvvm.FPRoundingMode.RN,
        loc=loc,
        ip=ip,
    )
    return cutlass.Uint32(llvm.bitcast(T.i32(), pair, loc=loc, ip=ip))


@dsl_user_op
def mma_e4m3(c0, c1, c2, c3, a0, a1, a2, a3, b0, b1, *, loc=None, ip=None):
    f32x4 = llvm.StructType.get_literal([T.f32()] * 4)
    a = [cutlass.Uint32(x).ir_value(loc=loc, ip=ip) for x in (a0, a1, a2, a3)]
    b = [cutlass.Uint32(x).ir_value(loc=loc, ip=ip) for x in (b0, b1)]
    c = [cutlass.Float32(x).ir_value(loc=loc, ip=ip) for x in (c0, c1, c2, c3)]
    attr = ir.Attribute.parse
    result = nvvm.mma_sync(
        f32x4,
        attr("#nvvm.shape<m = 16, n = 8, k = 32>"),
        attr("#nvvm.mma_layout<row>"),
        attr("#nvvm.mma_layout<col>"),
        a,
        b,
        c,
        multiplicand_a_ptx_type=attr("#nvvm.mma_type<e4m3>"),
        multiplicand_b_ptx_type=attr("#nvvm.mma_type<e4m3>"),
        loc=loc,
        ip=ip,
    )
    return tuple(
        cutlass.Float32(llvm.extractvalue(T.f32(), result, [j], loc=loc, ip=ip))
        for j in range(4)
    )


@dsl_user_op
def bf16x4_to_e4m3x4(
    w0: cutlass.Uint32, w1: cutlass.Uint32, inv: cutlass.Float32, *, loc=None, ip=None
):
    s = cutlass.Float32(inv).ir_value(loc=loc, ip=ip)
    high = cutlass.Uint32(0xFFFF0000)

    def scaled(bits):
        return mul_rn_ir(
            llvm.bitcast(T.f32(), bits.ir_value(loc=loc, ip=ip), loc=loc, ip=ip),
            s,
            loc,
            ip,
        )

    def e4m3x2(first, second):
        packed = llvm.call_intrinsic(
            T.i16(),
            "llvm.nvvm.ff.to.e4m3x2.rn",
            [first, second],
            [],
            [],
            loc=loc,
            ip=ip,
        )
        return cutlass.Uint32(arith.extui(T.i32(), packed, loc=loc, ip=ip))

    a, b = scaled(w0 << cutlass.Uint32(16)), scaled(w0 & high)
    c, d = scaled(w1 << cutlass.Uint32(16)), scaled(w1 & high)
    return e4m3x2(b, a) | (e4m3x2(d, c) << cutlass.Uint32(16))


@dsl_user_op
def evict_last(*, loc=None, ip=None):
    return cutlass.Uint64(0x14F0000000000000)


@dsl_user_op
def atom_max_u64(pointer: cute.Pointer, value: cutlass.Uint64, *, loc=None, ip=None):
    cute.arch.red(
        pointer,
        value,
        op="max",
        dtype="u64",
        sem="relaxed",
        scope="gpu",
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def shared_max_u64(
    address: cutlass.Uint32, value: cutlass.Uint64, *, loc=None, ip=None
):
    llvm.atomicrmw(
        llvm.AtomicBinOp.umax,
        as_ptr(3, address, loc, ip),
        value.ir_value(loc=loc, ip=ip),
        llvm.AtomicOrdering.monotonic,
        syncscope="block",
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def shared_add_u32(
    address: cutlass.Uint32, value: cutlass.Uint32, *, loc=None, ip=None
):
    return cutlass.Uint32(
        llvm.atomicrmw(
            llvm.AtomicBinOp.add,
            as_ptr(3, address, loc, ip),
            value.ir_value(loc=loc, ip=ip),
            llvm.AtomicOrdering.acq_rel,
            syncscope="block",
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def release_add_u32(pointer: cute.Pointer, value: cutlass.Uint32, *, loc=None, ip=None):
    cute.arch.red(
        pointer,
        value,
        op="add",
        dtype="u32",
        sem="release",
        scope="gpu",
        loc=loc,
        ip=ip,
    )


@cute.jit
def argmax_key(logit_bits: cutlass.Uint32, index: cutlass.Int32):
    exp_mant = logit_bits & cutlass.Uint32(0x7FFFFFFF)
    ordered = logit_bits ^ cutlass.Uint32(0x80000000)
    if (logit_bits & cutlass.Uint32(0x80000000)) != cutlass.Uint32(0):
        ordered = ~logit_bits
    if exp_mant > cutlass.Uint32(0x7F800000):
        ordered = cutlass.Uint32(0xFFFFFFFF)
    low = cutlass.Uint32(0xFFFFFFFF) - index.to(cutlass.Uint32)
    return (ordered.to(cutlass.Uint64) << cutlass.Uint64(32)) | low.to(cutlass.Uint64)


@cute.jit
def key_index(key: cutlass.Uint64):
    return (
        cutlass.Uint32(0xFFFFFFFF)
        - (key & cutlass.Uint64(0xFFFFFFFF)).to(cutlass.Uint32)
    ).to(cutlass.Int32)


@cute.jit
def issue_lm_chunk_lane(
    cta: cutlass.Int32,
    load: cutlass.Int32,
    weights: cute.Tensor,
    region: cute.Tensor,
    bars: cute.Tensor,
    tid,
    fp8: cutlass.Constexpr = False,
):
    tile = cta * TILES_PER_CTA + load % TILES_PER_CTA
    chunk = load // TILES_PER_CTA
    stage = load % STAGES
    rows = VOCAB_SHARD - tile * 16
    if rows > 16:
        rows = 16
    if tile >= TILES:
        rows = 0
    bar = bars.iterator + stage
    base = region.iterator.toint().to(cutlass.Uint32) + cutlass.Uint32(
        stage * STAGE_BYTES
    )
    keep = cutlass.Boolean(False)
    if cutlass.const_expr(KEEP_CHUNKS > 0):
        keep = (chunk >= 1) & (chunk <= KEEP_CHUNKS)
    if cutlass.const_expr(fp8):
        if cutlass.const_expr(STAGES8 != STAGES):
            stage = load % STAGES8
            bar = bars.iterator + stage
            if stage >= STAGES:
                bar = bars.iterator + (stage + X_STAGES)
            base = region.iterator.toint().to(cutlass.Uint32) + (
                stage * W8_STAGE_BYTES
            ).to(cutlass.Uint32)
        if tid == 0:
            cute.arch.mbarrier_arrive_and_expect_tx(bar, rows * CHUNK_K)
        if tid < rows:
            if cutlass.const_expr(FP8_KEEP):
                bulk_copy(
                    base + (tid * W8_STRIDE).to(cutlass.Uint32),
                    weights.iterator + (tile * 16 + tid) * HIDDEN + chunk * CHUNK_K,
                    cutlass.Uint32(CHUNK_K),
                    bar,
                    evict_last(),
                )
            else:
                bulk_copy(
                    base + (tid * W8_STRIDE).to(cutlass.Uint32),
                    weights.iterator + (tile * 16 + tid) * HIDDEN + chunk * CHUNK_K,
                    cutlass.Uint32(CHUNK_K),
                    bar,
                )
    else:
        if tid == 0:
            cute.arch.mbarrier_arrive_and_expect_tx(bar, rows * ROW_BYTES)
        if tid < rows:
            if cutlass.const_expr(KEEP_CHUNKS > 0):
                if keep:
                    bulk_copy(
                        base + (tid * ROW_STRIDE).to(cutlass.Uint32),
                        weights.iterator + (tile * 16 + tid) * HIDDEN + chunk * CHUNK_K,
                        cutlass.Uint32(ROW_BYTES),
                        bar,
                        evict_last(),
                    )
                else:
                    bulk_copy(
                        base + (tid * ROW_STRIDE).to(cutlass.Uint32),
                        weights.iterator + (tile * 16 + tid) * HIDDEN + chunk * CHUNK_K,
                        cutlass.Uint32(ROW_BYTES),
                        bar,
                        evict_first(),
                    )
            elif cutlass.const_expr(W_EVICT_FIRST):
                bulk_copy(
                    base + (tid * ROW_STRIDE).to(cutlass.Uint32),
                    weights.iterator + (tile * 16 + tid) * HIDDEN + chunk * CHUNK_K,
                    cutlass.Uint32(ROW_BYTES),
                    bar,
                    evict_first(),
                )
            else:
                bulk_copy(
                    base + (tid * ROW_STRIDE).to(cutlass.Uint32),
                    weights.iterator + (tile * 16 + tid) * HIDDEN + chunk * CHUNK_K,
                    cutlass.Uint32(ROW_BYTES),
                    bar,
                )


@cute.jit
def issue_lm_chunk(
    cta: cutlass.Int32,
    load: cutlass.Int32,
    weights: cute.Tensor,
    region: cute.Tensor,
    bars: cute.Tensor,
    fp8: cutlass.Constexpr = False,
):
    tid, _, _ = cute.arch.thread_idx()
    issue_lm_chunk_lane(cta, load, weights, region, bars, tid, fp8)


@cute.jit
def issue_x_chunk(
    x: cute.Tensor,
    c: cutlass.Constexpr,
    rows: cutlass.Int32,
    region: cute.Tensor,
    bars: cute.Tensor,
    tid,
):
    xs = c % X_STAGES
    bar = bars.iterator + (STAGES + xs)
    base = region.iterator.toint().to(cutlass.Uint32) + cutlass.Uint32(
        X_BASE + xs * X_STAGE_BYTES
    )
    if tid == 0:
        cute.arch.mbarrier_arrive_and_expect_tx(bar, rows * ROW_BYTES)
    if tid < rows:
        bulk_copy(
            base + (tid * ROW_STRIDE).to(cutlass.Uint32),
            x.iterator + tid * (HIDDEN // 2) + c * (CHUNK_K // 2),
            cutlass.Uint32(ROW_BYTES),
            bar,
        )


@cute.jit
def issue_lm_prologue(
    cta: cutlass.Int32,
    weights: cute.Tensor,
    region: cute.Tensor,
    bars: cute.Tensor,
    fp8: cutlass.Constexpr = False,
):
    tid, _, _ = cute.arch.thread_idx()
    if tid < 16:
        for load in cutlass.range_constexpr(STAGES8 if fp8 else STAGES):
            issue_lm_chunk(cta, cutlass.Int32(load), weights, region, bars, fp8)


@dsl_user_op
def bulk_copy_mc(
    dst: cutlass.Uint32,
    src: cute.Pointer,
    size: cutlass.Uint32,
    bar: cutlass.Uint32,
    mask: cutlass.Uint32,
    *,
    loc=None,
    ip=None,
):
    nvvm.cp_async_bulk_shared_cluster_global(
        as_ptr(7, dst, loc, ip),
        as_ptr(1, src, loc, ip),
        as_ptr(3, bar, loc, ip),
        size.ir_value(loc=loc, ip=ip),
        multicast_mask=cutlass.Uint16(mask).ir_value(loc=loc, ip=ip),
        loc=loc,
        ip=ip,
    )


@cute.jit
def issue_lm_x(
    x: cute.Tensor, rows: cutlass.Int32, region: cute.Tensor, bars: cute.Tensor
):
    tid, _, _ = cute.arch.thread_idx()
    cta, _, _ = cute.arch.block_idx()
    if cutlass.const_expr(X_MODE == 2):
        if tid == 0:
            for c in cutlass.range_constexpr(X_STAGES):
                cute.arch.mbarrier_arrive_and_expect_tx(
                    bars.iterator + (STAGES + c), rows * ROW_BYTES
                )
        row = cta % X_CLUSTER + X_CLUSTER * tid
        if (tid < 8 // X_CLUSTER) & (row < rows):
            for c in cutlass.range_constexpr(X_STAGES):
                bulk_copy_mc(
                    region.iterator.toint().to(cutlass.Uint32)
                    + cutlass.Uint32(X_BASE + c * X_STAGE_BYTES)
                    + (row * ROW_STRIDE).to(cutlass.Uint32),
                    x.iterator + row * (HIDDEN // 2) + c * (CHUNK_K // 2),
                    cutlass.Uint32(ROW_BYTES),
                    (bars.iterator + (STAGES + c)).toint().to(cutlass.Uint32),
                    cutlass.Uint32((1 << X_CLUSTER) - 1),
                )
    elif cutlass.const_expr(X_SMEM):
        if tid < 8:
            for c in cutlass.range_constexpr(X_STAGES):
                issue_x_chunk(x, c, rows, region, bars, tid)


@cute.jit
def load_x_chunk(
    x: cute.Tensor,
    xv: cute.Tensor,
    c: cutlass.Constexpr,
    has_row,
    group,
    warp,
    tig,
    poll: cutlass.Constexpr,
    region: cute.Tensor,
    bars: cute.Tensor,
    rows: cutlass.Int32,
):
    if cutlass.const_expr(X_SMEM):
        cute.arch.mbarrier_wait(
            bars.iterator + (STAGES + c % X_STAGES), (c // X_STAGES) % 2
        )
        xseg = (
            region.iterator.toint().to(cutlass.Uint32)
            + cutlass.Uint32(X_BASE + (c % X_STAGES) * X_STAGE_BYTES)
            + (group * ROW_STRIDE + warp * (STEPS * 64) + tig * 16).to(cutlass.Uint32)
        )
        for s in cutlass.range_constexpr(STEPS):
            w0, w1, w2, w3 = lds128(xseg + cutlass.Uint32(s * 64))
            if group >= rows:
                w0, w1, w2, w3 = (
                    cutlass.Uint32(0),
                    cutlass.Uint32(0),
                    cutlass.Uint32(0),
                    cutlass.Uint32(0),
                )
            xv[s, 0], xv[s, 1], xv[s, 2], xv[s, 3] = w0, w1, w2, w3
    else:
        for s in cutlass.range_constexpr(STEPS):
            k = c * CHUNK_K + warp * (STEPS * 32) + s * 32 + tig * 8
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
            xv[s, 0], xv[s, 1], xv[s, 2], xv[s, 3] = w0, w1, w2, w3


@cute.jit
def lm_head_stage(
    cta: cutlass.Int32,
    x: cute.Tensor,
    weights: cute.Tensor,
    region: cute.Tensor,
    red: cute.Tensor,
    bars: cute.Tensor,
    keys: cute.Tensor,
    arrivals: cute.Tensor,
    rows: cutlass.Int32,
    logits_out: cute.Tensor,
    poll: cutlass.Constexpr,
    write_logits: cutlass.Constexpr,
    table_prefetch=None,
    fp8: cutlass.Constexpr = False,
    scales=None,
    rescore=None,
    xscale=None,
):
    tid, _, _ = cute.arch.thread_idx()
    warp, lane = tid >> 5, tid & 31
    group, tig = lane >> 2, lane & 3
    has_row = group < rows
    acc_a = cute.make_rmem_tensor((TILES_PER_CTA, 4), cutlass.Float32)
    acc_b = cute.make_rmem_tensor((TILES_PER_CTA, 4), cutlass.Float32)
    for t in cutlass.range_constexpr(TILES_PER_CTA):
        for e in cutlass.range_constexpr(4):
            acc_a[t, e] = cutlass.Float32(0)
            acc_b[t, e] = cutlass.Float32(0)
    xbuf = [
        cute.make_rmem_tensor((STEPS, 4), cutlass.Uint32),
        cute.make_rmem_tensor((STEPS, 4), cutlass.Uint32),
    ]
    if cutlass.const_expr(not X_SMEM):
        load_x_chunk(x, xbuf[0], 0, has_row, group, warp, tig, poll, region, bars, rows)
    for c in cutlass.range_constexpr(CHUNKS):
        xv = xbuf[c % 2]
        if cutlass.const_expr(X_SMEM):
            load_x_chunk(x, xv, c, has_row, group, warp, tig, poll, region, bars, rows)
            if cutlass.const_expr(fp8 and xscale is not None):
                for s8 in cutlass.range_constexpr(STEPS):
                    xv[s8, 0] = bf16x4_to_e4m3x4(xv[s8, 0], xv[s8, 1], xscale[1])
                    xv[s8, 1] = bf16x4_to_e4m3x4(xv[s8, 2], xv[s8, 3], xscale[1])
            elif cutlass.const_expr(fp8):
                for s8 in cutlass.range_constexpr(STEPS):
                    for j8 in cutlass.range_constexpr(4):
                        xv[s8, j8] = bf16x2_to_f16x2(xv[s8, j8])
        elif cutlass.const_expr(c + 1 < CHUNKS):
            load_x_chunk(
                x,
                xbuf[(c + 1) % 2],
                c + 1,
                has_row,
                group,
                warp,
                tig,
                poll,
                region,
                bars,
                rows,
            )
        for t in cutlass.range_constexpr(TILES_PER_CTA):
            load = c * TILES_PER_CTA + t
            NS = STAGES8 if fp8 else STAGES
            stage = load % NS
            wbar = stage if stage < STAGES else stage + X_STAGES
            cute.arch.mbarrier_wait(bars.iterator + wbar, (load // NS) % 2)
            if cutlass.const_expr(fp8):
                sbytes = W8_STAGE_BYTES if STAGES8 != STAGES else STAGE_BYTES
                seg8 = region.iterator.toint().to(cutlass.Uint32) + cutlass.Uint32(
                    stage * sbytes + warp * (STEPS * 32) + tig * 8
                )
                for s in cutlass.range_constexpr(STEPS if xscale is not None else 0):
                    q0, q1 = lds64w(seg8 + cutlass.Uint32(group * W8_STRIDE + s * 32))
                    p0, p1 = lds64w(
                        seg8 + cutlass.Uint32((group + 8) * W8_STRIDE + s * 32)
                    )
                    c0, c1, c2, c3 = mma_e4m3(
                        acc_a[t, 0],
                        acc_a[t, 1],
                        acc_a[t, 2],
                        acc_a[t, 3],
                        q0,
                        p0,
                        q1,
                        p1,
                        xv[s, 0],
                        xv[s, 1],
                    )
                    acc_a[t, 0], acc_a[t, 1], acc_a[t, 2], acc_a[t, 3] = c0, c1, c2, c3
                for s in cutlass.range_constexpr(STEPS if xscale is None else 0):
                    q0, q1 = lds64w(seg8 + cutlass.Uint32(group * W8_STRIDE + s * 32))
                    p0, p1 = lds64w(
                        seg8 + cutlass.Uint32((group + 8) * W8_STRIDE + s * 32)
                    )
                    lo0, lo1 = e4m3x4_to_f16x2(q0)
                    lo2, lo3 = e4m3x4_to_f16x2(q1)
                    hi0, hi1 = e4m3x4_to_f16x2(p0)
                    hi2, hi3 = e4m3x4_to_f16x2(p1)
                    c0, c1, c2, c3 = mma_f16(
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
                    d0, d1, d2, d3 = mma_f16(
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
            seg = region.iterator.toint().to(cutlass.Uint32) + cutlass.Uint32(
                stage * STAGE_BYTES + warp * (STEPS * 32 * 2) + tig * 16
            )
            for s in cutlass.range_constexpr(0 if fp8 else STEPS):
                lo0, lo1, lo2, lo3 = lds128(
                    seg + cutlass.Uint32(group * ROW_STRIDE + s * 64)
                )
                hi0, hi1, hi2, hi3 = lds128(
                    seg + cutlass.Uint32((group + 8) * ROW_STRIDE + s * 64)
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
            if cutlass.const_expr(REL_SYNC):
                cute.arch.sync_threads()
                if cutlass.const_expr(load + NS < LOADS):
                    if tid < 16:
                        issue_lm_chunk(
                            cta, cutlass.Int32(load + NS), weights, region, bars, fp8
                        )
                if cutlass.const_expr(X_SMEM and t == 0 and c + X_STAGES < CHUNKS):
                    if (tid >= 32) & (tid < 40):
                        issue_x_chunk(x, c + X_STAGES, rows, region, bars, tid - 32)
            else:
                if cutlass.const_expr(
                    load + STAGES < LOADS
                    or (X_SMEM and t == 0 and c + X_STAGES < CHUNKS)
                ):
                    old = cutlass.Uint32(0)
                    if lane == 0:
                        old = shared_add_u32(
                            red.iterator.toint().to(cutlass.Uint32)
                            + cutlass.Uint32((RED_FLOATS + stage) * 4),
                            cutlass.Uint32(1),
                        )
                    old = cute.arch.shuffle_sync(old, 0)
                    if old == cutlass.Uint32(8 * (load // STAGES) + 7):
                        if cutlass.const_expr(load + STAGES < LOADS):
                            if lane < 16:
                                issue_lm_chunk_lane(
                                    cta,
                                    cutlass.Int32(load + STAGES),
                                    weights,
                                    region,
                                    bars,
                                    lane,
                                    fp8,
                                )
                        if cutlass.const_expr(
                            X_SMEM and t == 0 and c + X_STAGES < CHUNKS
                        ):
                            if (lane >= 16) & (lane < 24):
                                issue_x_chunk(
                                    x, c + X_STAGES, rows, region, bars, lane - 16
                                )
    if cutlass.const_expr(RED_ALIAS):
        cute.arch.sync_threads()
    for t in cutlass.range_constexpr(TILES_PER_CTA):
        for e in cutlass.range_constexpr(4):
            red[((t * 8 + warp) * 4 + e) * 32 + lane] = acc_a[t, e] + acc_b[t, e]
    if tid < 16:
        red[RED_KEYS + tid] = bits_f32(cutlass.Uint32(0))
    cute.arch.sync_threads()
    kaddr = red.iterator.toint().to(cutlass.Uint32) + cutlass.Uint32(RED_KEYS * 4)

    for j in cutlass.range_constexpr((TILES_PER_CTA * 128 + 255) // 256):
        vi = tid + 256 * j
        if vi < TILES_PER_CTA * 128:
            t = vi // 128
            e = (vi // 32) % 4
            l = vi % 32
            v = cutlass.Float32(0)
            for w in cutlass.range_constexpr(8):
                v += red[((t * 8 + w) * 4 + e) * 32 + l]
            row = l // 4 + (e // 2) * 8
            token = 2 * (l % 4) + e % 2
            index = (cta * TILES_PER_CTA + t) * 16 + row
            if cutlass.const_expr(fp8):
                if index < VOCAB_SHARD:
                    v = v * scales[index]
                    if cutlass.const_expr(xscale is not None):
                        v = v * xscale[0]
            bits = bf16_pair(v, v) << 16
            if (index < VOCAB_SHARD) & (token < rows):
                if cutlass.const_expr(write_logits):
                    logits_out[token * VOCAB_SHARD + index] = bits_f32(bits)
                shared_max_u64(
                    kaddr + (token * 8).to(cutlass.Uint32), argmax_key(bits, index)
                )
    cute.arch.sync_threads()
    if cutlass.const_expr(fp8 and rescore is not None):
        k0 = (
            f32_bits(red[RED_KEYS + 1]).to(cutlass.Uint64) << cutlass.Uint64(32)
        ) | f32_bits(red[RED_KEYS]).to(cutlass.Uint64)
        cand = key_index(k0)
        dot = cutlass.Float32(0)
        if (cand >= 0) & (cand < VOCAB_SHARD):
            for j in cutlass.range_constexpr(3):
                wofs = cand * (HIDDEN // 2) + (2048 * j + 8 * tid) // 2
                w0, w1, w2, w3 = readonly_vector(rescore.iterator + wofs)
                a0, a1, a2, a3 = readonly_vector(x.iterator + (2048 * j + 8 * tid) // 2)
                for wq, aq in ((w0, a0), (w1, a1), (w2, a2), (w3, a3)):
                    dot = dot + bits_f32(wq << 16) * bits_f32(aq << 16)
                    dot = dot + bits_f32(wq & cutlass.Uint32(0xFFFF0000)) * bits_f32(
                        aq & cutlass.Uint32(0xFFFF0000)
                    )
        for sh in cutlass.range_constexpr(5):
            dot = dot + cute.arch.shuffle_sync_bfly(dot, 16 >> sh)
        cute.arch.sync_threads()
        if lane == 0:
            red[warp] = dot
        cute.arch.sync_threads()
        if tid == 0:
            total = cutlass.Float32(0)
            for w8 in cutlass.range_constexpr(8):
                total += red[w8]
            if (cand >= 0) & (cand < VOCAB_SHARD):
                exact = argmax_key(bf16_pair(total, total) << 16, cand)
                red[RED_KEYS] = bits_f32(
                    (exact & cutlass.Uint64(0xFFFFFFFF)).to(cutlass.Uint32)
                )
                red[RED_KEYS + 1] = bits_f32(
                    (exact >> cutlass.Uint64(32)).to(cutlass.Uint32)
                )
        cute.arch.sync_threads()
    if tid == 0:
        for tk in cutlass.range_constexpr(8):
            if tk < rows:
                lo = f32_bits(red[RED_KEYS + 2 * tk]).to(cutlass.Uint64)
                hi = f32_bits(red[RED_KEYS + 2 * tk + 1]).to(cutlass.Uint64)
                atom_max_u64(
                    keys.iterator + tk * KEY_STRIDE, (hi << cutlass.Uint64(32)) | lo
                )
        release_add_u32(arrivals.iterator, cutlass.Uint32(1))
        if cutlass.const_expr(table_prefetch is not None):
            k0 = (
                f32_bits(red[RED_KEYS + 1]).to(cutlass.Uint64) << cutlass.Uint64(32)
            ) | f32_bits(red[RED_KEYS]).to(cutlass.Uint64)
            local = key_index(k0)
            if (local >= 0) & (local < VOCAB_SHARD):
                prefetch_bulk_l2(
                    table_prefetch.iterator + local * (HIDDEN // 2),
                    cutlass.Uint32(HIDDEN * 2),
                )


FP4_ROW_BYTES = HIDDEN // 2 + HIDDEN // 16
# 868 words = 4 mod 32, so the 8 row groups of a warp hit distinct shared-memory banks.
FP4_SMEM_STRIDE = FP4_ROW_BYTES + 16
FP4_X_OFFSET = TILES_PER_CTA * 16 * FP4_SMEM_STRIDE
FP4_DOT = False
FP4_INT_DECODE = 0
FP4_UNROLL = 2
FP4_RESCORE_TILES = False


@dsl_user_op
def e2m1x2_to_f16x2(byte: cutlass.Uint32, *, loc=None, ip=None):
    byte8 = arith.trunci(
        ir.IntegerType.get_signless(8), byte.ir_value(loc=loc, ip=ip), loc=loc, ip=ip
    )
    return cutlass.Uint32(
        llvm.bitcast(T.i32(), e2m1x2_to_f16x2_ir(byte8, loc, ip), loc=loc, ip=ip)
    )


@dsl_user_op
def e2m1x2_int_f16x2(word: cutlass.Uint32, j: cutlass.Constexpr, *, loc=None, ip=None):
    """Integer ops instead of cvt.rn.f16x2.e2m1x2: the same bits, on the ALU pipe instead of the conversion pipe."""
    t = word >> cutlass.Uint32(8 * j)
    selector = ((t & cutlass.Uint32(7)) << cutlass.Uint32(4)) | (
        (t & cutlass.Uint32(0x70)) << cutlass.Uint32(8)
    )
    magnitude = cutlass.Uint32(
        nvvm.prmt(
            cutlass.Uint32(1044133888).ir_value(loc=loc, ip=ip),
            selector.ir_value(loc=loc, ip=ip),
            nvvm.PermuteMode.DEFAULT,
            hi=cutlass.Uint32(1178878528).ir_value(loc=loc, ip=ip),
            loc=loc,
            ip=ip,
        )
    )
    return (
        magnitude
        | (t & cutlass.Uint32(8)) << cutlass.Uint32(12)
        | (t & cutlass.Uint32(128)) << cutlass.Uint32(24)
    )


@dsl_user_op
def e2m1x8_to_f16x2x4(word: cutlass.Uint32, *, loc=None, ip=None):
    lanes = llvm.bitcast(
        ir.VectorType.get([4], ir.IntegerType.get_signless(8)),
        word.ir_value(loc=loc, ip=ip),
        loc=loc,
        ip=ip,
    )
    pairs = (
        e2m1x2_to_f16x2_ir(vector.extract(lanes, [], [i], loc=loc, ip=ip), loc, ip)
        for i in range(4)
    )
    return tuple(
        cutlass.Uint32(llvm.bitcast(T.i32(), p, loc=loc, ip=ip)) for p in pairs
    )


@dsl_user_op
def hfma2(
    a: cutlass.Uint32, b: cutlass.Uint32, c: cutlass.Uint32, *, loc=None, ip=None
):
    r = fma_ir(
        *(f16x2_ir(v.ir_value(loc=loc, ip=ip), loc, ip) for v in (a, b, c)), loc, ip
    )
    return cutlass.Uint32(llvm.bitcast(T.i32(), r, loc=loc, ip=ip))


@dsl_user_op
def f16x2_sum_f32(a: cutlass.Uint32, *, loc=None, ip=None):
    pair = llvm.bitcast(
        ir.VectorType.get([2], ir.F16Type.get()),
        a.ir_value(loc=loc, ip=ip),
        loc=loc,
        ip=ip,
    )
    x, y = (
        arith.extf(
            T.f32(), vector.extract(pair, [], [i], loc=loc, ip=ip), loc=loc, ip=ip
        )
        for i in range(2)
    )
    return cutlass.Float32(arith.addf(x, y, loc=loc, ip=ip))


@dsl_user_op
def e4m3_to_f32(byte: cutlass.Uint32, *, loc=None, ip=None):
    half = vector.extract(
        e4m3x2_to_f16x2_ir(low_16(byte, loc, ip), loc, ip), [], [0], loc=loc, ip=ip
    )
    return cutlass.Float32(arith.extf(T.f32(), half, loc=loc, ip=ip))


@dsl_user_op
def hmul2(a: cutlass.Uint32, b: cutlass.Uint32, *, loc=None, ip=None):
    # fma(a, b, -0.0) is the correctly rounded product, signed zeros included
    neg0 = cutlass.Uint32(0x80008000)
    r = fma_ir(
        *(f16x2_ir(v.ir_value(loc=loc, ip=ip), loc, ip) for v in (a, b, neg0)), loc, ip
    )
    return cutlass.Uint32(llvm.bitcast(T.i32(), r, loc=loc, ip=ip))


@dsl_user_op
def sts32(address: cutlass.Uint32, value: cutlass.Uint32, *, loc=None, ip=None):
    llvm.store(
        value.ir_value(loc=loc, ip=ip),
        as_ptr(3, address, loc, ip),
        alignment=4,
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def lds32(address: cutlass.Uint32, *, loc=None, ip=None):
    return cutlass.Uint32(
        llvm.load(T.i32(), as_ptr(3, address, loc, ip), alignment=4, loc=loc, ip=ip)
    )


@cute.jit
def issue_lm_fp4(
    cta: cutlass.Int32, w4: cute.Tensor, region: cute.Tensor, bar: cute.Pointer
):
    tid, _, _ = cute.arch.thread_idx()
    rows = VOCAB_SHARD - cta * TILES_PER_CTA * 16
    if rows > TILES_PER_CTA * 16:
        rows = cutlass.Int32(TILES_PER_CTA * 16)
    if tid == 0:
        cute.arch.mbarrier_arrive_and_expect_tx(bar, rows * FP4_ROW_BYTES)
    if tid < rows:
        bulk_copy(
            region.iterator.toint().to(cutlass.Uint32)
            + (tid * FP4_SMEM_STRIDE).to(cutlass.Uint32),
            w4.iterator + (cta * TILES_PER_CTA * 16 + tid) * FP4_ROW_BYTES,
            cutlass.Uint32(FP4_ROW_BYTES),
            bar,
            evict_first(),
        )


@cute.jit
def rescore_row(
    cand: cutlass.Int32,
    x: cute.Tensor,
    wbf16: cute.Tensor,
    red: cute.Tensor,
    slot: cutlass.Constexpr,
):
    tid, _, _ = cute.arch.thread_idx()
    warp, lane = tid >> 5, tid & 31
    dot = cutlass.Float32(0)
    if (cand >= 0) & (cand < VOCAB_SHARD):
        for j in cutlass.range_constexpr(3):
            w0, w1, w2, w3 = readonly_vector(
                wbf16.iterator + cand * (HIDDEN // 2) + (2048 * j + 8 * tid) // 2
            )
            a0, a1, a2, a3 = readonly_vector(x.iterator + (2048 * j + 8 * tid) // 2)
            for wq, aq in ((w0, a0), (w1, a1), (w2, a2), (w3, a3)):
                dot = dot + bits_f32(wq << 16) * bits_f32(aq << 16)
                dot = dot + bits_f32(wq & cutlass.Uint32(0xFFFF0000)) * bits_f32(
                    aq & cutlass.Uint32(0xFFFF0000)
                )
    for sh in cutlass.range_constexpr(5):
        dot = dot + cute.arch.shuffle_sync_bfly(dot, 16 >> sh)
    if lane == 0:
        red[8 * slot + warp] = dot
    cute.arch.sync_threads()
    total = cutlass.Float32(0)
    for w8 in cutlass.range_constexpr(8):
        total += red[8 * slot + w8]
    return total


@cute.jit
def lm_head_fp4_stage(
    cta: cutlass.Int32,
    x: cute.Tensor,
    region: cute.Tensor,
    red: cute.Tensor,
    bar_w: cute.Pointer,
    bar_x: cute.Pointer,
    keys: cute.Tensor,
    arrivals: cute.Tensor,
    row_scales: cute.Tensor,
    wbf16: cute.Tensor,
):
    tid, _, _ = cute.arch.thread_idx()
    warp, lane = tid >> 5, tid & 31
    group, tig = lane >> 2, lane & 3
    base = region.iterator.toint().to(cutlass.Uint32)
    if tid == 0:
        cute.arch.mbarrier_arrive_and_expect_tx(bar_x, cutlass.Uint32(HIDDEN * 2))
        bulk_copy(
            base + cutlass.Uint32(FP4_X_OFFSET),
            x.iterator,
            cutlass.Uint32(HIDDEN * 2),
            bar_x,
        )
    acc_a = cute.make_rmem_tensor((TILES_PER_CTA, 4), cutlass.Float32)
    acc_b = cute.make_rmem_tensor((TILES_PER_CTA, 4), cutlass.Float32)
    for t in cutlass.range_constexpr(TILES_PER_CTA):
        for e in cutlass.range_constexpr(4):
            acc_a[t, e] = cutlass.Float32(0)
            acc_b[t, e] = cutlass.Float32(0)
    cute.arch.mbarrier_wait(bar_w, 0)
    cute.arch.mbarrier_wait(bar_x, 0)
    if cutlass.const_expr(FP4_DOT):
        for j in cutlass.range_constexpr(12):
            xa = (
                base
                + cutlass.Uint32(FP4_X_OFFSET)
                + ((j * 256 + tid) * 4).to(cutlass.Uint32)
            )
            sts32(xa, bf16x2_to_f16x2(lds32(xa)))
        cute.arch.sync_threads()
        ra0 = cute.make_rmem_tensor((TILES_PER_CTA,), cutlass.Float32)
        rb0 = cute.make_rmem_tensor((TILES_PER_CTA,), cutlass.Float32)
        for t in cutlass.range_constexpr(TILES_PER_CTA):
            ra0[t] = cutlass.Float32(0)
            rb0[t] = cutlass.Float32(0)
        for s in cutlass.range(HIDDEN // (8 * 32), unroll=FP4_UNROLL):
            k0 = warp * (HIDDEN // 8) + s * 32 + tig * 8
            x0, x1, x2, x3 = lds128(
                base + cutlass.Uint32(FP4_X_OFFSET) + (k0 * 2).to(cutlass.Uint32)
            )
            xs4 = (x0, x1, x2, x3)
            for t in cutlass.range_constexpr(TILES_PER_CTA):
                for half in cutlass.range_constexpr(2):
                    rr = base + ((t * 16 + group + 8 * half) * FP4_SMEM_STRIDE).to(
                        cutlass.Uint32
                    )
                    q = lds32(rr + (k0 // 2).to(cutlass.Uint32))
                    sw = lds32(
                        rr
                        + cutlass.Uint32(HIDDEN // 2)
                        + ((k0 // 16) & ~3).to(cutlass.Uint32)
                    )
                    sc = e4m3_to_f32(
                        (sw >> (((k0 // 16) & 3) * 8).to(cutlass.Uint32))
                        & cutlass.Uint32(0xFF)
                    )
                    d4 = e2m1x8_to_f16x2x4(q)
                    hacc = cutlass.Uint32(0)
                    for j in cutlass.range_constexpr(4):
                        hacc = hfma2(d4[j], xs4[j], hacc)
                    if cutlass.const_expr(half == 0):
                        ra0[t] = ra0[t] + f16x2_sum_f32(hacc) * sc
                    else:
                        rb0[t] = rb0[t] + f16x2_sum_f32(hacc) * sc
        for t in cutlass.range_constexpr(TILES_PER_CTA):
            for sh in cutlass.range_constexpr(2):
                ra0[t] = ra0[t] + cute.arch.shuffle_sync_bfly(ra0[t], 1 << sh)
                rb0[t] = rb0[t] + cute.arch.shuffle_sync_bfly(rb0[t], 1 << sh)
            if tig == 0:
                acc_a[t, 0] = ra0[t]
                acc_a[t, 2] = rb0[t]
    ksteps = HIDDEN // (8 * 32)
    for s in cutlass.range(0 if FP4_DOT else ksteps, unroll=FP4_UNROLL):
        k0 = warp * (ksteps * 32) + s * 32 + tig * 8
        xw = cute.make_rmem_tensor((4,), cutlass.Uint32)
        for j in cutlass.range_constexpr(4):
            xw[j] = cutlass.Uint32(0)
        if group == 0:
            x0, x1, x2, x3 = lds128(
                base + cutlass.Uint32(FP4_X_OFFSET) + (k0 * 2).to(cutlass.Uint32)
            )
            xw[0], xw[1], xw[2], xw[3] = (
                bf16x2_to_f16x2(x0),
                bf16x2_to_f16x2(x1),
                bf16x2_to_f16x2(x2),
                bf16x2_to_f16x2(x3),
            )
        for t in cutlass.range_constexpr(TILES_PER_CTA):
            ra = base + ((t * 16 + group) * FP4_SMEM_STRIDE).to(cutlass.Uint32)
            rb = ra + cutlass.Uint32(8 * FP4_SMEM_STRIDE)
            qa = lds32(ra + (k0 // 2).to(cutlass.Uint32))
            qb = lds32(rb + (k0 // 2).to(cutlass.Uint32))
            sa = lds32(
                ra + cutlass.Uint32(HIDDEN // 2) + ((k0 // 16) & ~3).to(cutlass.Uint32)
            )
            sb = lds32(
                rb + cutlass.Uint32(HIDDEN // 2) + ((k0 // 16) & ~3).to(cutlass.Uint32)
            )
            sh_a = ((k0 // 16) & 3) * 8
            sh_b = sh_a
            ba = (sa >> sh_a.to(cutlass.Uint32)) & cutlass.Uint32(0xFF)
            bb = (sb >> sh_b.to(cutlass.Uint32)) & cutlass.Uint32(0xFF)
            fa, _u = e4m3x4_to_f16x2(ba | (ba << 8))
            fb, _v = e4m3x4_to_f16x2(bb | (bb << 8))
            lo = cute.make_rmem_tensor((4,), cutlass.Uint32)
            hi = cute.make_rmem_tensor((4,), cutlass.Uint32)
            for j in cutlass.range_constexpr(4):
                if cutlass.const_expr(
                    (FP4_INT_DECODE == 4) or (FP4_INT_DECODE == 2 and j % 2 == 1)
                ):
                    lo[j] = hmul2(e2m1x2_int_f16x2(qa, j), fa)
                    hi[j] = hmul2(e2m1x2_int_f16x2(qb, j), fb)
                else:
                    lo[j] = hmul2(
                        e2m1x2_to_f16x2(
                            (qa >> cutlass.Uint32(8 * j)) & cutlass.Uint32(0xFF)
                        ),
                        fa,
                    )
                    hi[j] = hmul2(
                        e2m1x2_to_f16x2(
                            (qb >> cutlass.Uint32(8 * j)) & cutlass.Uint32(0xFF)
                        ),
                        fb,
                    )
            c0, c1, c2, c3 = mma_f16(
                acc_a[t, 0],
                acc_a[t, 1],
                acc_a[t, 2],
                acc_a[t, 3],
                lo[0],
                hi[0],
                lo[1],
                hi[1],
                xw[0],
                xw[1],
            )
            acc_a[t, 0], acc_a[t, 1], acc_a[t, 2], acc_a[t, 3] = c0, c1, c2, c3
            d0, d1, d2, d3 = mma_f16(
                acc_b[t, 0],
                acc_b[t, 1],
                acc_b[t, 2],
                acc_b[t, 3],
                lo[2],
                hi[2],
                lo[3],
                hi[3],
                xw[2],
                xw[3],
            )
            acc_b[t, 0], acc_b[t, 1], acc_b[t, 2], acc_b[t, 3] = d0, d1, d2, d3
    for t in cutlass.range_constexpr(TILES_PER_CTA):
        for e in cutlass.range_constexpr(4):
            red[((t * 8 + warp) * 4 + e) * 32 + lane] = acc_a[t, e] + acc_b[t, e]
    if tid < 16:
        red[RED_KEYS + tid] = bits_f32(cutlass.Uint32(0))
    cute.arch.sync_threads()
    kaddr = red.iterator.toint().to(cutlass.Uint32) + cutlass.Uint32(RED_KEYS * 4)
    for j in cutlass.range_constexpr((TILES_PER_CTA * 128 + 255) // 256):
        vi = tid + 256 * j
        if vi < TILES_PER_CTA * 128:
            t = vi // 128
            e = (vi // 32) % 4
            l = vi % 32
            row = l // 4 + (e // 2) * 8
            token = 2 * (l % 4) + e % 2
            index = (cta * TILES_PER_CTA + t) * 16 + row
            if (token == 0) & (index < VOCAB_SHARD):
                v = cutlass.Float32(0)
                for w in cutlass.range_constexpr(8):
                    v += red[((t * 8 + w) * 4 + e) * 32 + l]
                v = v * row_scales[index]
                bits = bf16_pair(v, v) << 16
                if cutlass.const_expr(FP4_RESCORE_TILES):
                    shared_max_u64(
                        kaddr + (8 * t).to(cutlass.Uint32), argmax_key(bits, index)
                    )
                else:
                    shared_max_u64(kaddr, argmax_key(bits, index))
    cute.arch.sync_threads()
    ncand = TILES_PER_CTA if FP4_RESCORE_TILES else 1
    best = cutlass.Uint64(0)
    for c in cutlass.range_constexpr(ncand):
        kc = (
            f32_bits(red[RED_KEYS + 2 * c + 1]).to(cutlass.Uint64) << cutlass.Uint64(32)
        ) | f32_bits(red[RED_KEYS + 2 * c]).to(cutlass.Uint64)
        cand = key_index(kc)
        total = rescore_row(cand, x, wbf16, red, 2 + c)
        if (tid == 0) & (cand >= 0) & (cand < VOCAB_SHARD):
            exact = argmax_key(bf16_pair(total, total) << 16, cand)
            if exact > best:
                best = exact
    if tid == 0:
        atom_max_u64(keys.iterator, best)
        release_add_u32(arrivals.iterator, cutlass.Uint32(1))


from .attention import bf16_pair, mma_bf16, mma_f16, volatile_vector
from .common import (
    as_ptr,
    bits_f32,
    bulk_copy,
    e2m1x2_to_f16x2_ir,
    e4m3x2_to_f16x2_ir,
    evict_first,
    f16x2_ir,
    f32_bits,
    fma_ir,
    low_16,
    mul_rn_ir,
    prefetch_bulk_l2,
)
from .gpu_publication import lds128, readonly_vector

# Imported last: the role modules import each other, and their names are used only when a kernel is traced.
from .moe import poll_words
