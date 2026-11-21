# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
import cutlass
import cutlass.cute as cute
from cutlass._mlir.dialects import arith, llvm
from cutlass.cutlass_dsl import T, dsl_user_op

HIDDEN = 6144
EXPERTS = 256
TOPK = 8
INTER = 128
ROW_BYTES = HIDDEN * 2
X_WORDS = HIDDEN // 2

ROUTER_UNITS = EXPERTS // 2

FC1_UNITS = TOPK * INTER
FC1_PAIRS = FC1_UNITS // 2
FC1_MAX_ROWS = 12
FC1_REGION = FC1_MAX_ROWS * ROW_BYTES

FC2_TILES = 2
FC2_ROWS = 16 * FC2_TILES
FC2_UNITS = HIDDEN // FC2_ROWS
FC2_SLICE = FC2_ROWS * INTER * 2
FC2_REGION = TOPK * FC2_SLICE

REGION_BYTES = FC1_REGION + FC2_REGION
ACT_WORDS = FC1_UNITS
LOGIT_WORDS = EXPERTS
SM_ROUTE, SM_SIG, SM_ROWSUM, SM_RSUM, SM_TERMS, SM_OUT, SM_ACT = (
    0,
    16,
    272,
    368,
    384,
    640,
    1024,
)
SM_WORDS = 1536
BARS = 1 + FC1_MAX_ROWS + 1


def routing_scale(routed_scaling_factor):
    import struct

    vf = struct.unpack(">f", struct.pack(">f", routed_scaling_factor))[0]
    bits = struct.unpack(">I", struct.pack(">f", routed_scaling_factor))[0]
    fast = (
        vf == routed_scaling_factor and (bits & 0xF) == 0 and 2.0**-20 <= vf <= 2.0**20
    )
    return float(routed_scaling_factor), vf, fast


@dsl_user_op
def sigmoid_accurate(x: cutlass.Float32, *, loc=None, ip=None):
    def k(v):
        return f32_const(v, loc, ip)

    def op(name, *values):
        return f32_intrinsic(name, *values, loc=loc, ip=ip)

    def fma(a, b, c):
        return op("llvm.nvvm.fma.rn.f", a, b, c)

    h = op("llvm.nvvm.mul.rn.f", x, k(0.5))
    a = op("llvm.nvvm.fabs.f", h)
    e = op(
        "llvm.nvvm.ex2.approx.ftz.f", op("llvm.nvvm.mul.rn.f", a, k(2.885390043258667))
    )
    e = op("llvm.nvvm.add.rn.f", e, k(1.0))
    f = fma(op("llvm.nvvm.rcp.approx.ftz.f", e), k(-2.0), k(1.0))
    f = arith.select(
        arith.cmpf(arith.CmpFPredicate.OGE, a, k(9.010913848876953), loc=loc, ip=ip),
        k(1.0),
        f,
        loc=loc,
        ip=ip,
    )
    s = llvm.call_intrinsic(
        T.f32(), "llvm.copysign.f32", [f, h], [], [], loc=loc, ip=ip
    )
    q = op("llvm.nvvm.mul.rn.f", h, h)
    p = fma(q, k(0.01573968306183815), k(-0.05230396240949631))
    p = fma(p, q, k(0.13315297663211823))
    p = fma(p, q, k(-0.33332768082618713))
    p = fma(p, q, k(0.0))
    p = fma(p, h, h)
    p = arith.select(
        arith.cmpf(arith.CmpFPredicate.OGE, a, k(0.6000000238418579), loc=loc, ip=ip),
        s,
        p,
        loc=loc,
        ip=ip,
    )
    return cutlass.Float32(fma(p, k(0.5), k(0.5)))


@dsl_user_op
def silu_mul(gate: cutlass.Float32, up: cutlass.Float32, *, loc=None, ip=None):
    def k(v):
        return f32_const(v, loc, ip)

    def op(name, *values):
        return f32_intrinsic(name, *values, loc=loc, ip=ip)

    def fma(a, b, c):
        return op("llvm.nvvm.fma.rn.f", a, b, c)

    x = cutlass.Float32(gate).ir_value(loc=loc, ip=ip)
    # clamp to [0, 1]; NaN goes to 0 because max returns the non-NaN operand
    a = op(
        "llvm.nvvm.fmin.f",
        op("llvm.nvvm.fmax.f", fma(x, k(-0.005724980030208826), k(0.5)), k(0.0)),
        k(1.0),
    )
    # a * 252 + C rounded toward -inf: C is an integer with unit ulp, so the result is C + floor(a * 252).
    # Round to nearest, then step down one when the exact residual shows it rounded up.
    r = fma(a, k(252.0), k(12582913.0))
    residual = fma(
        a,
        k(252.0),
        op("llvm.nvvm.add.rn.f", k(12582913.0), arith.negf(r, loc=loc, ip=ip)),
    )
    below = arith.cmpf(
        arith.CmpFPredicate.OLT,  # codespell:ignore
        residual,
        k(0.0),
        loc=loc,
        ip=ip,
    )
    b = arith.select(below, op("llvm.nvvm.add.rn.f", r, k(-1.0)), r, loc=loc, ip=ip)
    c = arith.negf(op("llvm.nvvm.add.rn.f", b, k(-12583039.0)), loc=loc, ip=ip)
    d = fma(x, k(-1.925963033500011e-08), fma(x, k(-1.4426950216293335), c))
    scale = arith.shli(
        llvm.bitcast(T.i32(), b, loc=loc, ip=ip),
        cutlass.Uint32(23).ir_value(loc=loc, ip=ip),
        loc=loc,
        ip=ip,
    )
    g = fma(
        op("llvm.nvvm.ex2.approx.ftz.f", d),
        llvm.bitcast(T.f32(), scale, loc=loc, ip=ip),
        k(1.0),
    )
    h = op("llvm.nvvm.rcp.rn.f", g)
    return cutlass.Float32(op("llvm.nvvm.mul.rn.f", up, op("llvm.nvvm.mul.rn.f", x, h)))


@dsl_user_op
def div_rn(a: cutlass.Float32, b: cutlass.Float32, *, loc=None, ip=None):
    return cutlass.Float32(f32_intrinsic("llvm.nvvm.div.rn.f", a, b, loc=loc, ip=ip))


@dsl_user_op
def add_rn(a: cutlass.Float32, b: cutlass.Float32, *, loc=None, ip=None):
    return cutlass.Float32(f32_intrinsic("llvm.nvvm.add.rn.f", a, b, loc=loc, ip=ip))


@dsl_user_op
def sub_rn(a: cutlass.Float32, b: cutlass.Float32, *, loc=None, ip=None):
    # a - b is a + (-b) exactly, signed zeros included
    minus_b = arith.negf(cutlass.Float32(b).ir_value(loc=loc, ip=ip), loc=loc, ip=ip)
    return cutlass.Float32(
        f32_intrinsic("llvm.nvvm.add.rn.f", a, minus_b, loc=loc, ip=ip)
    )


@dsl_user_op
def routing_weight_fp64(
    s: cutlass.Float32,
    total: cutlass.Float32,
    c: cutlass.Constexpr,
    *,
    loc=None,
    ip=None,
):
    def f64(name, x, y):
        return llvm.call_intrinsic(T.f64(), name, [x, y], [], [], loc=loc, ip=ip)

    a = arith.extf(T.f64(), cutlass.Float32(s).ir_value(loc=loc, ip=ip), loc=loc, ip=ip)
    b = arith.extf(
        T.f64(), cutlass.Float32(total).ir_value(loc=loc, ip=ip), loc=loc, ip=ip
    )
    n = f64("llvm.nvvm.mul.rn.d", a, arith.constant(T.f64(), float(c), loc=loc, ip=ip))
    d = f64("llvm.nvvm.add.rn.d", b, arith.constant(T.f64(), 1e-20, loc=loc, ip=ip))
    return cutlass.Float32(
        arith.truncf(T.f32(), f64("llvm.nvvm.div.rn.d", n, d), loc=loc, ip=ip)
    )


@cute.jit
def routing_weight(s: cutlass.Float32, total: cutlass.Float32, c: cutlass.Constexpr):
    cd, cf, fast = routing_scale(c)
    result = cutlass.Float32(0.0)
    done = cutlass.Boolean(False)
    if cutlass.const_expr(fast):
        if (total >= cutlass.Float32(2.0**-13)) & (s >= cutlass.Float32(2.0**-60)):
            p_hi = mul_rn(s, cutlass.Float32(cf))
            p_lo = fma_rn(s, cutlass.Float32(cf), -p_hi)
            q = div_rn(p_hi, total)
            r = fma_rn(-q, total, p_hi)
            e = div_rn(add_rn(r, p_lo), total)
            y = add_rn(q, e)
            bits = f32_bits(y)
            if (
                (y > cutlass.Float32(0.0))
                & ((bits & cutlass.Uint32(0x7FFFFF)) != cutlass.Uint32(0))
                & ((bits & cutlass.Uint32(0x7F800000)) != cutlass.Uint32(0x7F800000))
            ):
                ulp = mul_rn(
                    bits_f32(bits & cutlass.Uint32(0x7F800000)),
                    cutlass.Float32(2.0**-23),
                )
                g = sub_rn(e, sub_rn(y, q))
                bound = sub_rn(
                    mul_rn(cutlass.Float32(0.5), ulp),
                    mul_rn(cutlass.Float32(2.0**-20), ulp),
                )
                if cute.math.abs(g) < bound:
                    result = y
                    done = cutlass.Boolean(True)
    if not done:
        result = routing_weight_fp64(s, total, cd)
    return result


POLL_PARALLEL = True


@cute.jit
def poll_vectors(x: cute.Tensor, out: cute.Tensor, offsets):
    n = len(offsets)
    for v in cutlass.range_constexpr(n):
        a, b, c, d = volatile_vector(x.iterator + offsets[v])
        out[v, 0], out[v, 1], out[v, 2], out[v, 3] = a, b, c, d
    waiting = cutlass.Boolean(True)
    while waiting:
        full = cutlass.Boolean(True)
        for v in cutlass.range_constexpr(n):
            for j in cutlass.range_constexpr(4):
                full = full & (out[v, j] != cutlass.Uint32(EMPTY))
        if full:
            waiting = cutlass.Boolean(False)
        else:
            for v in cutlass.range_constexpr(n):
                ready = cutlass.Boolean(True)
                for j in cutlass.range_constexpr(4):
                    ready = ready & (out[v, j] != cutlass.Uint32(EMPTY))
                if not ready:
                    a, b, c, d = volatile_vector(x.iterator + offsets[v])
                    out[v, 0], out[v, 1], out[v, 2], out[v, 3] = a, b, c, d


@cute.jit
def round_bf16_value(value: cutlass.Float32):
    return bits_f32(bf16_pair(value, value) << 16)


@cute.jit
def bf16_word(value: cutlass.Float32):
    """The low bit is set so the word is never the Lamport empty value."""
    return (bf16_pair(value, value) & cutlass.Uint32(0xFFFF0000)) | cutlass.Uint32(1)


@cute.jit
def issue_router(
    unit: cutlass.Int32, w_router: cute.Tensor, base: cutlass.Uint32, bar: cute.Pointer
):
    tid, _, _ = cute.arch.thread_idx()
    if tid == 0:
        cute.arch.mbarrier_arrive_and_expect_tx(bar, 2 * ROW_BYTES)
        for h in cutlass.range_constexpr(2):
            bulk_copy(
                base + cutlass.Uint32(h * ROW_BYTES),
                w_router.iterator + (2 * unit + h) * HIDDEN,
                cutlass.Uint32(ROW_BYTES),
                bar,
                evict_first(),
            )


@cute.jit
def router_unit(
    unit: cutlass.Int32,
    x: cute.Tensor,
    x_off: cutlass.Int32,
    base: cutlass.Uint32,
    sm: cute.Tensor,
    logits: cute.Tensor,
    bar: cute.Pointer,
    poll: cutlass.Constexpr,
):
    tid, _, _ = cute.arch.thread_idx()
    warp, lane = tid >> 5, tid & 31
    half, t = tid >> 7, tid & 127
    xv = cute.make_rmem_tensor((6, 4), cutlass.Uint32)
    if cutlass.const_expr(poll and POLL_PARALLEL):
        poll_vectors(x, xv, [x_off + (ki * 1024 + t * 8) // 2 for ki in range(6)])
    else:
        for ki in cutlass.range_constexpr(6):
            k = ki * 1024 + t * 8
            if cutlass.const_expr(poll):
                a0, a1, a2, a3 = poll_words(x, x_off + k // 2)
            else:
                a0, a1, a2, a3 = volatile_vector(x.iterator + x_off + k // 2)
            xv[ki, 0], xv[ki, 1], xv[ki, 2], xv[ki, 3] = a0, a1, a2, a3
    cute.arch.mbarrier_wait(bar, 0)
    acc = cutlass.Float32(0.0)
    row = base + cutlass.Uint32(half * ROW_BYTES)
    for ki in cutlass.range_constexpr(6):
        b0, b1, b2, b3 = lds128(row + cutlass.Uint32((ki * 1024 + t * 8) * 2))
        bw = (b0, b1, b2, b3)
        for i in cutlass.range_constexpr(4):
            acc = fma_rn(bits_f32(xv[ki, i] << 16), bits_f32(bw[i] << 16), acc)
            acc = fma_rn(
                bits_f32(xv[ki, i] & cutlass.Uint32(0xFFFF0000)),
                bits_f32(bw[i] & cutlass.Uint32(0xFFFF0000)),
                acc,
            )
    for sh in cutlass.range_constexpr(5):
        acc = acc + cute.arch.shuffle_sync_bfly(acc, 16 >> sh)
    if lane == 0:
        sm[SM_RSUM + warp] = f32_bits(acc)
    cute.arch.sync_threads()
    if t == 0:
        total = cutlass.Float32(0.0)
        for w in cutlass.range_constexpr(4):
            total += bits_f32(sm[SM_RSUM + half * 4 + w])
        store_word(logits.iterator + 2 * unit + half, published_word(f32_bits(total)))
    cute.arch.sync_threads()


@cute.jit
def route_moe(
    logits: cute.Tensor, bias: cute.Tensor, sm: cute.Tensor, scale: cutlass.Constexpr
):
    tid, _, _ = cute.arch.thread_idx()
    lane = tid & 31
    if tid < 32:
        bv = cute.make_rmem_tensor((8,), cutlass.Float32)
        lw = cute.make_rmem_tensor((8,), cutlass.Uint32)
        for i in cutlass.range_constexpr(8):
            bv[i] = bias[i * 32 + lane]
            lw[i] = volatile_word(logits.iterator + i * 32 + lane)
        waiting = cutlass.Boolean(True)
        while waiting:
            full = cutlass.Boolean(True)
            for i in cutlass.range_constexpr(8):
                full = full & (lw[i] != cutlass.Uint32(EMPTY))
            if cute.arch.vote_all_sync(full):
                waiting = cutlass.Boolean(False)
            else:
                for i in cutlass.range_constexpr(8):
                    if lw[i] == cutlass.Uint32(EMPTY):
                        lw[i] = volatile_word(logits.iterator + i * 32 + lane)
        hi = cute.make_rmem_tensor((8,), cutlass.Uint32)
        for i in cutlass.range_constexpr(8):
            sig = sigmoid_accurate(bits_f32(lw[i]))
            sm[SM_SIG + i * 32 + lane] = f32_bits(sig)
            score = f32_bits(sig + bv[i])
            twiddled = score ^ cutlass.Uint32(0x80000000)
            if (score >> 31) != cutlass.Uint32(0):
                twiddled = ~score
            hi[i] = twiddled
        cute.arch.sync_warp()
        taken = cutlass.Uint32(0)
        mine = cutlass.Uint32(0)
        for k in cutlass.range_constexpr(TOPK):
            best_hi = cutlass.Uint32(0)
            best_i = cutlass.Int32(-1)
            for i in cutlass.range_constexpr(8):
                free = (taken & cutlass.Uint32(1 << i)) == cutlass.Uint32(0)
                if free & ((best_i < 0) | (hi[i] > best_hi)):
                    best_hi = hi[i]
                    best_i = cutlass.Int32(i)
            max_hi = redux_max(best_hi)
            cand = cutlass.Uint32(0)
            if (best_hi == max_hi) & (best_i >= 0):
                cand = cutlass.Uint32(65535) - (best_i * 32 + lane).to(cutlass.Uint32)
            max_lo = redux_max(cand)
            expert = cutlass.Uint32(65535) - max_lo
            if lane.to(cutlass.Uint32) == (expert & cutlass.Uint32(31)):
                taken |= cutlass.Uint32(1) << (expert >> cutlass.Uint32(5))
            if lane == k:
                mine = expert
        s = cutlass.Float32(0.0)
        if lane < TOPK:
            s = bits_f32(sm[SM_SIG + mine.to(cutlass.Int32)])
        total = s
        for sh in cutlass.range_constexpr(5):
            total = total + cute.arch.shuffle_sync_bfly(total, 16 >> sh)
        if lane < TOPK:
            w = routing_weight(s, total, scale)
            sm[SM_ROUTE + lane] = mine
            sm[SM_ROUTE + 8 + lane] = f32_bits(w)
    cute.arch.sync_threads()


ROUTE_PARALLEL = True


@cute.jit
def route_moe_parallel(
    logits: cute.Tensor,
    bias: cute.Tensor,
    sm: cute.Tensor,
    scale: cutlass.Constexpr,
    ssq=None,
    after_select=None,
):
    tid, _, _ = cute.arch.thread_idx()
    warp, lane = tid >> 5, tid & 31
    bv = bias[tid]
    lw = volatile_word(logits.iterator + tid)
    waiting = cutlass.Boolean(lw == cutlass.Uint32(EMPTY))
    while waiting:
        lw = volatile_word(logits.iterator + tid)
        waiting = cutlass.Boolean(lw == cutlass.Uint32(EMPTY))
    if cutlass.const_expr(ssq is not None):
        if tid < 4:
            sw = volatile_word(ssq.iterator + tid)
            while sw == cutlass.Uint32(EMPTY):
                sw = volatile_word(ssq.iterator + tid)
            sm[SM_RSUM + tid] = sw
        cute.arch.sync_threads()
        ssq_total = (
            (cutlass.Float32(0) + bits_f32(sm[SM_RSUM])) + bits_f32(sm[SM_RSUM + 1])
        ) + bits_f32(sm[SM_RSUM + 2])
        ssq_total += bits_f32(sm[SM_RSUM + 3])
        rstd = cute.math.rsqrt(ssq_total / 6144.0 + 1.0e-5, fastmath=True)
        lw = f32_bits((cutlass.Float32(0.0) + bits_f32(lw)) * rstd)
    sig = sigmoid_accurate(bits_f32(lw))
    sm[SM_SIG + tid] = f32_bits(sig)
    score = f32_bits(sig + bv)
    key = score ^ cutlass.Uint32(0x80000000)
    if (score >> 31) != cutlass.Uint32(0):
        key = ~score
    local = cutlass.Int32(0)
    for d in cutlass.range_constexpr(1, 32):
        src = (lane + d) & 31
        other = cute.arch.shuffle_sync(key, src)
        if (other > key) | ((other == key) & (src < lane)):
            local += 1
    if local < TOPK:
        sm[SM_TERMS + 2 * (warp * TOPK + local)] = key
        sm[SM_TERMS + 2 * (warp * TOPK + local) + 1] = tid.to(cutlass.Uint32)
    cute.arch.sync_threads()
    if local < TOPK:
        me = tid.to(cutlass.Uint32)
        rank = cutlass.Int32(0)
        for c in cutlass.range_constexpr(8 * TOPK):
            ck = sm[SM_TERMS + 2 * c]
            ci = sm[SM_TERMS + 2 * c + 1]
            if (ck > key) | ((ck == key) & (ci < me)):
                rank += 1
        if rank < TOPK:
            sm[SM_ROUTE + rank] = me
    cute.arch.sync_threads()
    if cutlass.const_expr(after_select is not None):
        after_select()
    if tid < 32:
        s = cutlass.Float32(0.0)
        if lane < TOPK:
            s = bits_f32(sm[SM_SIG + sm[SM_ROUTE + lane].to(cutlass.Int32)])
        total = s
        for sh in cutlass.range_constexpr(5):
            total = total + cute.arch.shuffle_sync_bfly(total, 16 >> sh)
        if lane < TOPK:
            sm[SM_ROUTE + 8 + lane] = f32_bits(routing_weight(s, total, scale))
    cute.arch.sync_threads()


@cute.jit
def fc1_range(
    fc1_cta: cutlass.Int32,
    fc1_ctas: cutlass.Constexpr,
    pairs: cutlass.Constexpr = FC1_PAIRS,
):
    pb = (fc1_cta * pairs) // fc1_ctas
    pe = ((fc1_cta + 1) * pairs) // fc1_ctas
    return 2 * pb, 2 * (pe - pb)


@cute.jit
def issue_fc1(
    fc1_cta: cutlass.Int32,
    fc1_ctas: cutlass.Constexpr,
    w3w1: cute.Tensor,
    sm: cute.Tensor,
    base: cutlass.Uint32,
    bars: cute.Pointer,
    w_shared=None,
):
    tid, _, _ = cute.arch.thread_idx()
    pairs = FC1_PAIRS
    if cutlass.const_expr(w_shared is not None):
        pairs = 9 * (INTER // 2)
    if tid == 0:
        u0, nu = fc1_range(fc1_cta, fc1_ctas, pairs)
        policy = evict_first()
        for r in cutlass.range(2 * nu, unroll=1):
            unit = u0 + r // 2
            slot, j = unit // INTER, unit % INTER
            row = j
            if r % 2 == 0:
                row = INTER + j
            cute.arch.mbarrier_arrive_and_expect_tx(bars + r, ROW_BYTES)
            if cutlass.const_expr(w_shared is not None):
                if slot == TOPK:
                    srow = j + INTER
                    if r % 2 == 0:
                        srow = j
                    bulk_copy(
                        base + (r * ROW_BYTES).to(cutlass.Uint32),
                        w_shared.iterator + srow * HIDDEN,
                        cutlass.Uint32(ROW_BYTES),
                        bars + r,
                        policy,
                    )
                else:
                    expert = sm[SM_ROUTE + slot].to(cutlass.Int32)
                    bulk_copy(
                        base + (r * ROW_BYTES).to(cutlass.Uint32),
                        w3w1.iterator + (expert * (2 * INTER) + row) * HIDDEN,
                        cutlass.Uint32(ROW_BYTES),
                        bars + r,
                        policy,
                    )
            else:
                expert = sm[SM_ROUTE + slot].to(cutlass.Int32)
                bulk_copy(
                    base + (r * ROW_BYTES).to(cutlass.Uint32),
                    w3w1.iterator + (expert * (2 * INTER) + row) * HIDDEN,
                    cutlass.Uint32(ROW_BYTES),
                    bars + r,
                    policy,
                )


@cute.jit
def fc1_unit(
    fc1_cta: cutlass.Int32,
    fc1_ctas: cutlass.Constexpr,
    x: cute.Tensor,
    x_off: cutlass.Int32,
    base: cutlass.Uint32,
    sm: cute.Tensor,
    act: cute.Tensor,
    bars: cute.Pointer,
    poll: cutlass.Constexpr,
    pairs: cutlass.Constexpr = FC1_PAIRS,
):
    tid, _, _ = cute.arch.thread_idx()
    warp, lane = tid >> 5, tid & 31
    u0, nu = fc1_range(fc1_cta, fc1_ctas, pairs)
    xf = cute.make_rmem_tensor((3, 8), cutlass.Float32)
    xw = cute.make_rmem_tensor((3, 4), cutlass.Uint32)
    if cutlass.const_expr(poll and POLL_PARALLEL):
        poll_vectors(x, xw, [x_off + 4 * (tid + 256 * i) for i in range(3)])
    for i in cutlass.range_constexpr(3):
        if cutlass.const_expr(poll and POLL_PARALLEL):
            a0, a1, a2, a3 = xw[i, 0], xw[i, 1], xw[i, 2], xw[i, 3]
        elif cutlass.const_expr(poll):
            a0, a1, a2, a3 = poll_words(x, x_off + 4 * (tid + 256 * i))
        else:
            a0, a1, a2, a3 = volatile_vector(x.iterator + x_off + 4 * (tid + 256 * i))
        aw = (a0, a1, a2, a3)
        for q in cutlass.range_constexpr(4):
            xf[i, 2 * q] = bits_f32(aw[q] << 16)
            xf[i, 2 * q + 1] = bits_f32(aw[q] & cutlass.Uint32(0xFFFF0000))
    for uu in cutlass.range(nu, unroll=1):
        dots = cute.make_rmem_tensor((2,), cutlass.Float32)
        for rr in cutlass.range_constexpr(2):
            r = 2 * uu + rr
            cute.arch.mbarrier_wait(bars + r, 0)
            row = base + (r * ROW_BYTES).to(cutlass.Uint32)
            dot = cutlass.Float32(0.0)
            for i in cutlass.range_constexpr(3):
                p0, p1, p2, p3 = lds128(row + ((tid + 256 * i) * 16).to(cutlass.Uint32))
                pw = (p0, p1, p2, p3)
                d = mul_rn(bits_f32(pw[0] << 16), xf[i, 0])
                d = fma_rn(bits_f32(pw[0] & cutlass.Uint32(0xFFFF0000)), xf[i, 1], d)
                for q in cutlass.range_constexpr(1, 4):
                    d = fma_rn(bits_f32(pw[q] << 16), xf[i, 2 * q], d)
                    d = fma_rn(
                        bits_f32(pw[q] & cutlass.Uint32(0xFFFF0000)),
                        xf[i, 2 * q + 1],
                        d,
                    )
                if cutlass.const_expr(i == 0):
                    dot = d
                else:
                    dot = add_rn(dot, d)
            dots[rr] = dot
        for rr in cutlass.range_constexpr(2):
            v = dots[rr]
            for sh in cutlass.range_constexpr(5):
                v = add_rn(v, cute.arch.shuffle_sync_bfly(v, 16 >> sh))
            dots[rr] = v
        if lane == 0:
            sm[SM_ROWSUM + (2 * uu) * 8 + warp] = f32_bits(dots[0])
            sm[SM_ROWSUM + (2 * uu + 1) * 8 + warp] = f32_bits(dots[1])
    cute.arch.sync_threads()
    if tid < nu:
        gate = cutlass.Float32(0.0)
        up = cutlass.Float32(0.0)
        for w in cutlass.range_constexpr(8):
            gate = add_rn(gate, bits_f32(sm[SM_ROWSUM + (2 * tid) * 8 + w]))
            up = add_rn(up, bits_f32(sm[SM_ROWSUM + (2 * tid + 1) * 8 + w]))
        gate = bits_f32(bf16_pair(gate, gate) << 16)
        up = bits_f32(bf16_pair(up, up) << 16)
        store_word(act.iterator + u0 + tid, bf16_word(silu_mul(gate, up)))
    cute.arch.sync_threads()


@cute.jit
def issue_fc2(
    unit: cutlass.Int32,
    w2: cute.Tensor,
    sm: cute.Tensor,
    base: cutlass.Uint32,
    bar: cute.Pointer,
):
    tid, _, _ = cute.arch.thread_idx()
    if tid == 0:
        policy = evict_first()
        cute.arch.mbarrier_arrive_and_expect_tx(bar, FC2_REGION)
        for k in cutlass.range_constexpr(TOPK):
            expert = sm[SM_ROUTE + k].to(cutlass.Int32)
            bulk_copy(
                base + cutlass.Uint32(k * FC2_SLICE),
                w2.iterator + (expert * HIDDEN + unit * FC2_ROWS) * INTER,
                cutlass.Uint32(FC2_SLICE),
                bar,
                policy,
            )


@cute.jit
def fc2_unit(
    unit: cutlass.Int32,
    act: cute.Tensor,
    base: cutlass.Uint32,
    sm: cute.Tensor,
    out: cute.Tensor,
    out_off: cutlass.Int32,
    bar: cute.Pointer,
    addend=None,
    addend_rows=None,
):
    tid, _, _ = cute.arch.thread_idx()
    warp, lane = tid >> 5, tid & 31
    g, t = lane >> 2, lane & 3
    swap = g & 1
    a0, a1, a2, a3 = poll_words(act, warp * INTER + 4 * lane)
    act_base = sm.iterator.toint().to(cutlass.Uint32) + cutlass.Uint32(SM_ACT * 4)
    sm[SM_ACT + warp * 64 + 2 * lane] = (a0 >> cutlass.Uint32(16)) | (
        a1 & cutlass.Uint32(0xFFFF0000)
    )
    sm[SM_ACT + warp * 64 + 2 * lane + 1] = (a2 >> cutlass.Uint32(16)) | (
        a3 & cutlass.Uint32(0xFFFF0000)
    )
    cute.arch.sync_warp()
    acc = cute.make_rmem_tensor((FC2_TILES, 2, 4), cutlass.Float32)
    for ti in cutlass.range_constexpr(FC2_TILES):
        for hf in cutlass.range_constexpr(2):
            for e in cutlass.range_constexpr(4):
                acc[ti, hf, e] = cutlass.Float32(0.0)
    cute.arch.mbarrier_wait(bar, 0)
    slice_base = base + (warp * FC2_SLICE).to(cutlass.Uint32)
    for j0 in cutlass.range_constexpr(0, INTER, 64):
        for hf in cutlass.range_constexpr(2):
            j = j0 + (hf ^ swap) * 32 + 8 * t
            v0, v1, v2, v3 = lds128(
                act_base + ((warp * 64 + j // 2) * 4).to(cutlass.Uint32)
            )
            for ti in cutlass.range_constexpr(FC2_TILES):
                lo0, lo1, lo2, lo3 = lds128(
                    slice_base + (((ti * 16 + g) * INTER + j) * 2).to(cutlass.Uint32)
                )
                hi0, hi1, hi2, hi3 = lds128(
                    slice_base
                    + (((ti * 16 + g + 8) * INTER + j) * 2).to(cutlass.Uint32)
                )
                c0, c1, c2, c3 = mma_bf16(
                    acc[ti, hf, 0],
                    acc[ti, hf, 1],
                    acc[ti, hf, 2],
                    acc[ti, hf, 3],
                    lo0,
                    hi0,
                    lo1,
                    hi1,
                    v0,
                    v1,
                )
                c0, c1, c2, c3 = mma_bf16(c0, c1, c2, c3, lo2, hi2, lo3, hi3, v2, v3)
                acc[ti, hf, 0], acc[ti, hf, 1], acc[ti, hf, 2], acc[ti, hf, 3] = (
                    c0,
                    c1,
                    c2,
                    c3,
                )
    weight = bits_f32(sm[SM_ROUTE + 8 + warp])
    if t == 0:
        for ti in cutlass.range_constexpr(FC2_TILES):
            ra0 = acc[ti, 0, 0]
            ra1 = acc[ti, 1, 0]
            rb0 = acc[ti, 0, 2]
            rb1 = acc[ti, 1, 2]
            if swap != 0:
                ra0 = acc[ti, 0, 1]
                ra1 = acc[ti, 1, 1]
                rb0 = acc[ti, 0, 3]
                rb1 = acc[ti, 1, 3]
            vlo = mul_rn(weight, add_rn(ra0, ra1))
            vhi = mul_rn(weight, add_rn(rb0, rb1))
            sm[SM_TERMS + warp * FC2_ROWS + ti * 16 + g] = bf16_pair(vlo, vlo) << 16
            sm[SM_TERMS + warp * FC2_ROWS + ti * 16 + g + 8] = bf16_pair(vhi, vhi) << 16
    cute.arch.sync_threads()
    if tid < FC2_ROWS:
        total = cutlass.Float32(0.0)
        for k in cutlass.range_constexpr(TOPK):
            total = add_rn(total, bits_f32(sm[SM_TERMS + k * FC2_ROWS + tid]))
        sm[SM_OUT + tid] = f32_bits(total)
    cute.arch.sync_threads()
    if tid < FC2_ROWS // 2:
        word = bf16_pair(
            bits_f32(sm[SM_OUT + 2 * tid]), bits_f32(sm[SM_OUT + 2 * tid + 1])
        )
        if cutlass.const_expr(addend_rows is not None):
            word = bf16_pair(
                add_rn(
                    round_bf16_value(addend_rows[2 * tid]), bf16_from_word(word, False)
                ),
                add_rn(
                    round_bf16_value(addend_rows[2 * tid + 1]),
                    bf16_from_word(word, True),
                ),
            )
        if cutlass.const_expr(addend is not None):
            sw = volatile_word(addend.iterator + unit * (FC2_ROWS // 2) + tid)
            while sw == cutlass.Uint32(EMPTY):
                sw = volatile_word(addend.iterator + unit * (FC2_ROWS // 2) + tid)
            word = bf16_pair(
                add_rn(bf16_from_word(sw, False), bf16_from_word(word, False)),
                add_rn(bf16_from_word(sw, True), bf16_from_word(word, True)),
            )
        store_word(
            out.iterator + out_off + unit * (FC2_ROWS // 2) + tid, published_word(word)
        )
    cute.arch.sync_threads()


from .attention import (
    EMPTY,
    bf16_from_word,
    bf16_pair,
    mma_bf16,
    published_word,
    volatile_vector,
)
from .common import (
    bits_f32,
    bulk_copy,
    evict_first,
    f32_bits,
    f32_const,
    f32_intrinsic,
    fma_rn,
    mul_rn,
)
from .gpu_publication import lds128, store_word

# Imported last: the role modules import each other, and their names are used only when a kernel is traced.
from .moe import poll_words, redux_max, volatile_word
