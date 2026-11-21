# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""nvvm / llvm dialect forms for the device operations cute.arch has no exact equivalent for."""

import cutlass
import cutlass.cute as cute
from cutlass._mlir import ir
from cutlass._mlir.dialects import arith, llvm, nvvm, vector
from cutlass.cutlass_dsl import T, dsl_user_op

ARENA_BYTES = 2469760
ARENA_WORDS = ARENA_BYTES // 4
COUNTER_WORDS = 800
COUNTED_CLEARS = 155


def as_ptr(space, value, loc, ip):
    """An LLVM pointer in `space` from a cute pointer or an integer address, so ops address memory exactly as given."""
    if hasattr(value, "toint"):
        value = value.toint(loc=loc, ip=ip)
    if hasattr(value, "ir_value"):
        value = value.ir_value(loc=loc, ip=ip)
    return llvm.inttoptr(ir.Type.parse(f"!llvm.ptr<{space}>"), value, loc=loc, ip=ip)


def word_vector(words, loc, ip):
    return vector.from_elements(
        ir.VectorType.get([len(words)], T.i32()),
        [cutlass.Uint32(w).ir_value(loc=loc, ip=ip) for w in words],
        loc=loc,
        ip=ip,
    )


def f32_intrinsic(name, *values, loc, ip):
    """One correctly specified PTX float op (rounding, ftz, approx) per intrinsic: no contraction, no range fix-ups."""
    operands = [
        v if isinstance(v, ir.Value) else cutlass.Float32(v).ir_value(loc=loc, ip=ip)
        for v in values
    ]
    return llvm.call_intrinsic(T.f32(), name, operands, [], [], loc=loc, ip=ip)


def f32_const(value, loc, ip):
    return arith.constant(T.f32(), value, loc=loc, ip=ip)


def fma_ir(a, b, c, loc, ip, rnd="rn"):
    """nvvm.fma takes its rounding mode as a parsed attribute in this SDK; the enum value is rejected."""
    return nvvm.fma(
        a, b, c, ir.Attribute.parse(f"#nvvm.fp_rnd_mode<{rnd}>"), loc=loc, ip=ip
    )


def mul_rn_ir(a, b, loc, ip):
    return f32_intrinsic("llvm.nvvm.mul.rn.f", a, b, loc=loc, ip=ip)


def f16x2_ir(word, loc, ip):
    return llvm.bitcast(ir.VectorType.get([2], ir.F16Type.get()), word, loc=loc, ip=ip)


def e4m3x2_to_f16x2_ir(bits16, loc, ip):
    """Two e4m3 bytes held in an i16 to an f16x2 vector."""
    pair = llvm.bitcast(
        ir.VectorType.get([2], ir.IntegerType.get_signless(8)), bits16, loc=loc, ip=ip
    )
    return nvvm.convert_f8x2_to_f16x2(
        ir.VectorType.get([2], ir.F16Type.get()),
        pair,
        ir.TypeAttr.get(ir.Type.parse("f8E4M3FN")),
        loc=loc,
        ip=ip,
    )


def e2m1x2_to_f16x2_ir(byte, loc, ip):
    """Two e2m1 nibbles held in an i8 to an f16x2 vector."""
    return nvvm.convert_f4x2_to_f16x2(
        ir.VectorType.get([2], ir.F16Type.get()),
        byte,
        ir.TypeAttr.get(ir.Type.parse("f4E2M1FN")),
        loc=loc,
        ip=ip,
    )


def low_16(word, loc, ip):
    return arith.trunci(
        T.i16(), cutlass.Uint32(word).ir_value(loc=loc, ip=ip), loc=loc, ip=ip
    )


@dsl_user_op
def evict_first(*, loc=None, ip=None):
    return cutlass.Uint64(0x12F0000000000000)


@dsl_user_op
def store_empty(pointer: cute.Pointer, policy: cutlass.Uint64, *, loc=None, ip=None):
    empty = cutlass.Uint32(0x80000000)
    nvvm.store_ext(
        word_vector((empty, empty, empty, empty), loc, ip),
        as_ptr(1, pointer, loc, ip),
        l2_cache_hint=policy.ir_value(loc=loc, ip=ip),
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def release_increment(pointer: cute.Pointer, *, loc=None, ip=None):
    cute.arch.red(
        pointer,
        cutlass.Uint32(1),
        op="add",
        dtype="u32",
        sem="release",
        scope="gpu",
        loc=loc,
        ip=ip,
    )


@cute.jit
def reset_next_counters(next_counters: cute.Tensor):
    tid, _, _ = cute.arch.thread_idx()
    cta, _, _ = cute.arch.block_idx()
    index = cta * 256 + tid
    if index < COUNTER_WORDS:
        next_counters[index] = cutlass.Uint32(0)


@cute.jit
def clear_assigned(
    next_arena: cute.Tensor,
    next_topk_exchange: cute.Tensor,
    current_clear_counter: cute.Tensor,
):
    tid, _, _ = cute.arch.thread_idx()
    cta, _, _ = cute.arch.block_idx()
    if cta < 151:
        policy = evict_first()
        for step in cutlass.range_constexpr(4):
            word = cta * 4096 + 4 * tid + step * 1024
            if word < ARENA_WORDS:
                store_empty(next_arena.iterator + word, policy)
        cute.arch.sync_threads()
        if tid == 0:
            release_increment(current_clear_counter.iterator)
    if (cta >= 192) & (cta < 196):
        policy = evict_first()
        for step in cutlass.range_constexpr(4):
            word = (cta - 192) * 4096 + 4 * tid + step * 1024
            store_empty(next_topk_exchange.iterator + word, policy)
        cute.arch.sync_threads()
        if tid == 0:
            release_increment(current_clear_counter.iterator)


def _emit_f32(mnemonic, values, loc, ip):
    names = {
        "fma.rn.f32": "llvm.nvvm.fma.rn.f",
        "mul.rn.f32": "llvm.nvvm.mul.rn.f",
        "div.rn.ftz.f32": "llvm.nvvm.div.rn.ftz.f",
        "max.f32": "llvm.nvvm.fmax.f",
    }
    return cutlass.Float32(f32_intrinsic(names[mnemonic], *values, loc=loc, ip=ip))


@dsl_user_op
def fma_rn(
    a: cutlass.Float32, b: cutlass.Float32, c: cutlass.Float32, *, loc=None, ip=None
):
    return _emit_f32("fma.rn.f32", (a, b, c), loc, ip)


@dsl_user_op
def mul_rn(a: cutlass.Float32, b: cutlass.Float32, *, loc=None, ip=None):
    """A separate rounding step: prevents FMA contraction with the consumer."""
    return _emit_f32("mul.rn.f32", (a, b), loc, ip)


@dsl_user_op
def div_rn_ftz(a: cutlass.Float32, b: cutlass.Float32, *, loc=None, ip=None):
    return _emit_f32("div.rn.ftz.f32", (a, b), loc, ip)


@dsl_user_op
def max_num(a: cutlass.Float32, b: cutlass.Float32, *, loc=None, ip=None):
    return _emit_f32("max.f32", (a, b), loc, ip)


@dsl_user_op
def bits_f32(bits: cutlass.Uint32, *, loc=None, ip=None):
    return cutlass.Float32(
        llvm.bitcast(T.f32(), bits.ir_value(loc=loc, ip=ip), loc=loc, ip=ip)
    )


@dsl_user_op
def f32_bits(value: cutlass.Float32, *, loc=None, ip=None):
    return cutlass.Uint32(
        llvm.bitcast(T.i32(), value.ir_value(loc=loc, ip=ip), loc=loc, ip=ip)
    )


@dsl_user_op
def prefetch_bulk_l2(src: cute.Pointer, size: cutlass.Uint32, *, loc=None, ip=None):
    """A cache hint, never a completion barrier."""
    nvvm.cp_async_bulk_prefetch(
        as_ptr(1, src, loc, ip), size.ir_value(loc=loc, ip=ip), loc=loc, ip=ip
    )


@dsl_user_op
def tc_before(*, loc=None, ip=None):
    nvvm.tcgen05_fence(nvvm.Tcgen05FenceKind.BEFORE_THREAD_SYNC, loc=loc, ip=ip)


@dsl_user_op
def tc_after(*, loc=None, ip=None):
    nvvm.tcgen05_fence(nvvm.Tcgen05FenceKind.AFTER_THREAD_SYNC, loc=loc, ip=ip)


@dsl_user_op
def proxy_fence(*, loc=None, ip=None):
    cute.arch.fence_proxy("async.shared", space="cta", loc=loc, ip=ip)


@dsl_user_op
def allocate_tmem(dst: cute.Pointer, *, loc=None, ip=None):
    cute.arch.alloc_tmem(512, dst, loc=loc, ip=ip)
    cute.arch.relinquish_tmem_alloc_permit(loc=loc, ip=ip)


@dsl_user_op
def deallocate_tmem(address: cutlass.Uint32, *, loc=None, ip=None):
    cute.arch.dealloc_tmem(
        cute.make_ptr(cutlass.Float32, address, cute.AddressSpace.tmem, loc=loc, ip=ip),
        512,
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def copy_scales(
    tmem: cutlass.Uint32, src: cutlass.Uint32, rows8: bool = False, *, loc=None, ip=None
):
    descriptor = cutlass.Uint64(1 << 46)
    if not rows8:
        descriptor |= cutlass.Uint64(8 << 32)
    descriptor |= ((src & cutlass.Uint32(0x3FFFF)) >> 4).to(cutlass.Uint64)
    nvvm.tcgen05_cp(
        nvvm.Tcgen05CpShape.SHAPE_32x128b,
        as_ptr(6, tmem, loc, ip),
        descriptor.ir_value(loc=loc, ip=ip),
        group=nvvm.CTAGroupKind.CTA_1,
        multicast=nvvm.Tcgen05CpMulticast.WARPX4,
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def mma_nvfp4(
    d: cutlass.Uint32,
    a: cutlass.Uint32,
    b: cutlass.Uint32,
    sfa: cutlass.Uint32,
    sfb: cutlass.Uint32,
    accumulate: cutlass.Uint32,
    *,
    loc=None,
    ip=None,
):
    common = cutlass.Uint64((1 << 16) | (32 << 32) | (1 << 46) | (4 << 61))
    da = common | ((a & cutlass.Uint32(0x3FFFF)) >> 4).to(cutlass.Uint64)
    db = common | ((b & cutlass.Uint32(0x3FFFF)) >> 4).to(cutlass.Uint64)
    descriptor = cutlass.Uint32((1 << 7) | (1 << 10) | (1 << 17) | (8 << 24))
    nvvm.tcgen05_mma_block_scale(
        nvvm.Tcgen05MMAKind.MXF4NVF4,
        nvvm.CTAGroupKind.CTA_1,
        as_ptr(6, d, loc, ip),
        da.ir_value(loc=loc, ip=ip),
        db.ir_value(loc=loc, ip=ip),
        descriptor.ir_value(loc=loc, ip=ip),
        (accumulate != cutlass.Uint32(0)).ir_value(loc=loc, ip=ip),
        as_ptr(6, sfa, loc, ip),
        as_ptr(6, sfb, loc, ip),
        block_scale=nvvm.Tcgen05MMABlockScale.BLOCK16,
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def commit_mma(bar: cute.Pointer, *, loc=None, ip=None):
    commit_mma_addr(cutlass.Uint32(bar.toint(loc=loc, ip=ip)), loc=loc, ip=ip)


@dsl_user_op
def commit_mma_addr(bar: cutlass.Uint32, *, loc=None, ip=None):
    nvvm.tcgen05_commit(
        as_ptr(3, bar, loc, ip), group=nvvm.CTAGroupKind.CTA_1, loc=loc, ip=ip
    )


@dsl_user_op
def fp4_pair(lo: cutlass.Float32, hi: cutlass.Float32, *, loc=None, ip=None):
    b = nvvm.convert_f32x2_to_f4x2(
        hi.ir_value(loc=loc, ip=ip),
        lo.ir_value(loc=loc, ip=ip),
        ir.TypeAttr.get(ir.Type.parse("f4E2M1FN")),
        rnd=nvvm.FPRoundingMode.RN,
        loc=loc,
        ip=ip,
    )
    # the packed byte alone, zero-extended: callers OR pairs into a word at byte offsets
    return cutlass.Uint32(arith.extui(T.i32(), b, loc=loc, ip=ip))


@dsl_user_op
def e4m3_float(bits: cutlass.Uint32, *, loc=None, ip=None):
    half = vector.extract(
        e4m3x2_to_f16x2_ir(low_16(bits, loc, ip), loc, ip), [], [0], loc=loc, ip=ip
    )
    return cutlass.Float32(arith.extf(T.f32(), half, loc=loc, ip=ip))


@dsl_user_op
def bulk_copy(
    dst: cutlass.Uint32,
    src: cute.Pointer,
    size: cutlass.Uint32,
    bar: cute.Pointer,
    policy=None,
    *,
    loc=None,
    ip=None,
):
    hint = None if policy is None else policy.ir_value(loc=loc, ip=ip)
    nvvm.cp_async_bulk_shared_cluster_global(
        as_ptr(7, dst, loc, ip),
        as_ptr(1, src, loc, ip),
        as_ptr(3, bar, loc, ip),
        size.ir_value(loc=loc, ip=ip),
        l2_cache_hint=hint,
        loc=loc,
        ip=ip,
    )


@cute.jit
def hop1_word_offset(token: cutlass.Int32, rank: cutlass.Int32, col: cutlass.Int32):
    return ((rank * 192 + col // 32) * 8 + token) * 16 + (col % 32) // 2


@cute.jit
def xfrag_word_offset(token: cutlass.Int32, col: cutlass.Int32):
    return (
        (col >> 6) * 256
        + ((col >> 3) & 1) * 128
        + (4 * token + ((col >> 4) & 3)) * 4
        + (col & 7) // 2
    )


@cute.jit
def reduce_rank16(
    buf: cute.Tensor,
    token: cutlass.Int32,
    col: cutlass.Int32,
    first: cutlass.Constexpr,
    force=None,
):
    payload = cute.make_rmem_tensor((16, 4), cutlass.Uint32)
    for rank in cutlass.range_constexpr(16):
        offset = (token * 16 + rank) * 3072 + col // 2
        if cutlass.const_expr(first):
            offset = hop1_word_offset(token, rank, col)
        a, b, c, d = volatile_vector(buf.iterator + offset)
        payload[rank, 0], payload[rank, 1], payload[rank, 2], payload[rank, 3] = (
            a,
            b,
            c,
            d,
        )
    waiting = cutlass.Boolean(True)
    while waiting:
        full = cutlass.Boolean(True)
        for rank in cutlass.range_constexpr(16):
            for j in cutlass.range_constexpr(4):
                full = full & (payload[rank, j] != cutlass.Uint32(0x80000000))
        if cutlass.const_expr(force is not None):
            full = full | force
        if cute.arch.vote_all_sync(full):
            waiting = cutlass.Boolean(False)
        else:
            for rank in cutlass.range_constexpr(16):
                ready = cutlass.Boolean(True)
                for j in cutlass.range_constexpr(4):
                    ready = ready & (payload[rank, j] != cutlass.Uint32(0x80000000))
                if not ready:
                    offset = (token * 16 + rank) * 3072 + col // 2
                    if cutlass.const_expr(first):
                        offset = hop1_word_offset(token, rank, col)
                    a, b, c, d = volatile_vector(buf.iterator + offset)
                    (
                        payload[rank, 0],
                        payload[rank, 1],
                        payload[rank, 2],
                        payload[rank, 3],
                    ) = a, b, c, d
    low = cute.make_rmem_tensor((8,), cutlass.Float32)
    high = cute.make_rmem_tensor((8,), cutlass.Float32)
    for j in cutlass.range_constexpr(8):
        low[j], high[j] = cutlass.Float32(0), cutlass.Float32(0)
    for rank in cutlass.range_constexpr(16):
        for j in cutlass.range_constexpr(4):
            if cutlass.const_expr(rank < 8):
                low[2 * j] += bf16_from_word(payload[rank, j], False)
                low[2 * j + 1] += bf16_from_word(payload[rank, j], True)
            else:
                high[2 * j] += bf16_from_word(payload[rank, j], False)
                high[2 * j + 1] += bf16_from_word(payload[rank, j], True)
    for j in cutlass.range_constexpr(8):
        low[j] = (cutlass.Float32(0) + low[j]) + high[j]
    return low


@cute.jit
def self_clear(
    hop1: cute.Tensor,
    previous_hop2: cute.Tensor,
    token: cutlass.Int32,
    col: cutlass.Int32,
):
    policy = evict_first()
    for rank in cutlass.range_constexpr(16):
        store_empty(hop1.iterator + hop1_word_offset(token, rank, col), policy)
        store_empty(
            previous_hop2.iterator + (token * 16 + rank) * 3072 + col // 2, policy
        )


@cute.jit
def redirect(t: cute.Tensor, scratch: cute.Tensor, force: cutlass.Boolean):
    delta = (
        scratch.iterator.toint().to(cutlass.Int64)
        - t.iterator.toint().to(cutlass.Int64)
    ) // 4
    offset = cutlass.Int64(0)
    if force:
        offset = delta
    return cute.make_tensor(t.iterator + offset, t.layout)


@cute.jit
def stage_hop(
    unit: cutlass.Int32,
    second: cutlass.Constexpr,
    x_only: cutlass.Boolean,
    hop1: cute.Tensor,
    hop2: cute.Tensor,
    previous_hop2: cute.Tensor,
    residual_input: cute.Tensor,
    gamma: cute.Tensor,
    in_scale: cutlass.Float32,
    shared_residual: cute.Tensor,
    scalars: cute.Tensor,
    ssq1: cute.Tensor,
    ssq2: cute.Tensor,
    h: cute.Tensor,
    hp: cute.Tensor,
    hfp4: cute.Tensor,
    next_x: cute.Tensor,
    next_ssq: cute.Tensor,
    normalized_output: cute.Tensor,
    residual_output: cute.Tensor,
    warm_region=None,
    warm_tmem=None,
    warm_bar=None,
    extra_token=None,
    lean: cutlass.Constexpr = False,
    warm=None,
    amax_out=None,
):
    if cutlass.const_expr(warm is None):
        hop_pass(
            unit,
            second,
            x_only,
            hop1,
            hop2,
            previous_hop2,
            residual_input,
            gamma,
            in_scale,
            shared_residual,
            scalars,
            ssq1,
            ssq2,
            h,
            hp,
            hfp4,
            next_x,
            next_ssq,
            normalized_output,
            residual_output,
            warm_region,
            warm_tmem,
            warm_bar,
            extra_token,
            lean,
            None,
            amax_out,
        )
    else:
        if cutlass.const_expr(
            not lean or warm_region is not None or extra_token is not None
        ):
            raise TypeError("stage_hop(warm) needs lean and no warm_moe / extra_token")
        passes, scratch, scratch_ssq = warm[0], warm[1], warm[2]
        for p in cutlass.range(passes, unroll=1):
            force = (p + 1) < passes
            if cutlass.const_expr(len(warm) > 3):
                force = force | warm[3]
            hop_pass(
                unit,
                second,
                x_only,
                hop1,
                hop2,
                previous_hop2,
                residual_input,
                gamma,
                in_scale,
                shared_residual,
                scalars,
                redirect(ssq1, scratch_ssq, force),
                redirect(ssq2, scratch_ssq, force),
                redirect(h, scratch, force),
                hp,
                hfp4,
                next_x,
                next_ssq,
                redirect(normalized_output, scratch, force),
                redirect(residual_output, scratch, force),
                None,
                None,
                None,
                None,
                lean,
                force,
                None if amax_out is None else redirect(amax_out, scratch_ssq, force),
            )


@cute.jit
def hop_pass(
    unit: cutlass.Int32,
    second: cutlass.Constexpr,
    x_only: cutlass.Boolean,
    hop1: cute.Tensor,
    hop2: cute.Tensor,
    previous_hop2: cute.Tensor,
    residual_input: cute.Tensor,
    gamma: cute.Tensor,
    in_scale: cutlass.Float32,
    shared_residual: cute.Tensor,
    scalars: cute.Tensor,
    ssq1: cute.Tensor,
    ssq2: cute.Tensor,
    h: cute.Tensor,
    hp: cute.Tensor,
    hfp4: cute.Tensor,
    next_x: cute.Tensor,
    next_ssq: cute.Tensor,
    normalized_output: cute.Tensor,
    residual_output: cute.Tensor,
    warm_region=None,
    warm_tmem=None,
    warm_bar=None,
    extra_token=None,
    lean: cutlass.Constexpr = False,
    force=None,
    amax_out=None,
):
    tid, _, _ = cute.arch.thread_idx()
    token, quarter = unit // 4, unit % 4
    col = 1536 * quarter + 8 * tid
    active = tid < 192
    res_prev = cute.make_rmem_tensor((4,), cutlass.Uint32)
    gam = cute.make_rmem_tensor((4,), cutlass.Uint32)
    residual = cute.make_rmem_tensor((8,), cutlass.Float32)
    ssq = cutlass.Float32(0)
    if active:
        if cutlass.const_expr(second):
            for j in cutlass.range_constexpr(4):
                res_prev[j] = shared_residual[4 * tid + j]
        else:
            r0, r1 = global_pair(residual_input.iterator + token * 3072 + col // 2)
            r2, r3 = global_pair(residual_input.iterator + token * 3072 + col // 2 + 2)
            res_prev[0], res_prev[1], res_prev[2], res_prev[3] = r0, r1, r2, r3
        g0, g1, g2, g3 = readonly_vector(gamma.iterator + col // 2)
        gam[0], gam[1], gam[2], gam[3] = g0, g1, g2, g3
    if cutlass.const_expr(not second and warm_region is not None):
        warm_moe(warm_region, warm_tmem, warm_bar)
    if active:
        if cutlass.const_expr(second):
            self_clear(hop1, previous_hop2, token, col)
            sums = reduce_rank16(hop2, token, col, False, force)
            if cutlass.const_expr(extra_token is not None):
                extra = reduce_rank16(hop2, cutlass.Int32(extra_token), col, False)
                for j in cutlass.range_constexpr(8):
                    sums[j] = sums[j] + extra[j]
        else:
            sums = reduce_rank16(hop1, token, col, True, force)
        for j in cutlass.range_constexpr(4):
            residual[2 * j] = round_bf16(
                round_bf16(sums[2 * j]) + bf16_from_word(res_prev[j], False)
            )
            residual[2 * j + 1] = round_bf16(
                round_bf16(sums[2 * j + 1]) + bf16_from_word(res_prev[j], True)
            )
        for j in cutlass.range_constexpr(8):
            ssq += round_bf16(residual[j] * residual[j])
        if cutlass.const_expr(not lean):
            prime = cute.make_rmem_tensor((4,), cutlass.Uint32)
            for j in cutlass.range_constexpr(4):
                prime[j] = published_word(
                    bf16_pair(
                        round_bf16(residual[2 * j] * bf16_from_word(gam[j], False)),
                        round_bf16(residual[2 * j + 1] * bf16_from_word(gam[j], True)),
                    )
                )
            if cutlass.const_expr(second):
                store_vector(
                    next_x.iterator + xfrag_word_offset(token, col),
                    prime[0],
                    prime[1],
                    prime[2],
                    prime[3],
                )
            else:
                store_vector(
                    hp.iterator + token * 3072 + col // 2,
                    prime[0],
                    prime[1],
                    prime[2],
                    prime[3],
                )
        packed_res = cute.make_rmem_tensor((4,), cutlass.Uint32)
        for j in cutlass.range_constexpr(4):
            packed_res[j] = bf16_pair(residual[2 * j], residual[2 * j + 1])
        if cutlass.const_expr(second):
            store_vector(
                residual_output.iterator + token * 3072 + col // 2,
                packed_res[0],
                packed_res[1],
                packed_res[2],
                packed_res[3],
            )
        else:
            for j in cutlass.range_constexpr(4):
                shared_residual[4 * tid + j] = packed_res[j]
    rcp_six = cutlass.Float32(0)
    in_rcp = cutlass.Float32(0)
    if cutlass.const_expr(not second and not lean):
        rcp_six = cute.arch.rcp_approx(cutlass.Float32(6))
        in_rcp = cute.arch.rcp_approx(in_scale)
    for step in cutlass.range_constexpr(5):
        ssq += cute.arch.shuffle_sync_bfly(ssq, 16 >> step)
    if cutlass.const_expr(second and amax_out is not None):
        xm = cutlass.Float32(0)
        if active:
            for j in cutlass.range_constexpr(4):
                xm = max_num(
                    xm, cute.math.abs(residual[2 * j] * bf16_from_word(gam[j], False))
                )
                xm = max_num(
                    xm,
                    cute.math.abs(residual[2 * j + 1] * bf16_from_word(gam[j], True)),
                )
        for step in cutlass.range_constexpr(5):
            xm = max_num(xm, cute.arch.shuffle_sync_bfly(xm, 16 >> step))
        if (tid & 31) == 0:
            scalars[12 + (tid >> 5)] = xm
    if (tid & 31) == 0:
        scalars[tid >> 5] = ssq
    cute.arch.sync_threads()
    if tid == 0:
        partial = cutlass.Float32(0)
        for warp in cutlass.range_constexpr(8):
            partial += scalars[warp]
        if cutlass.const_expr(second and amax_out is not None):
            m = cutlass.Float32(0)
            for warp in cutlass.range_constexpr(8):
                m = max_num(m, scalars[12 + warp])
            store_word(
                amax_out.iterator + 4 * token + quarter, published_word(f32_bits(m))
            )
        if cutlass.const_expr(second and not lean):
            store_word(
                next_ssq.iterator + 4 * token + quarter,
                published_word(f32_bits(partial)),
            )
        rendezvous = cutlass.Boolean(True)
        if cutlass.const_expr(second):
            rendezvous = not x_only
        if rendezvous:
            if cutlass.const_expr(second):
                store_word(
                    ssq2.iterator + 4 * token + quarter,
                    published_word(f32_bits(partial)),
                )
                src = ssq2.iterator + 4 * token
            else:
                store_word(
                    ssq1.iterator + 4 * token + quarter,
                    published_word(f32_bits(partial)),
                )
                src = ssq1.iterator + 4 * token
            waiting = cutlass.Boolean(True)
            while waiting:
                a, b, c, d = volatile_vector(src)
                arrived = (
                    (a != cutlass.Uint32(0x80000000))
                    & (b != cutlass.Uint32(0x80000000))
                    & (c != cutlass.Uint32(0x80000000))
                    & (d != cutlass.Uint32(0x80000000))
                )
                if cutlass.const_expr(force is not None):
                    arrived = arrived | force
                if arrived:
                    scalars[8], scalars[9], scalars[10], scalars[11] = (
                        bits_f32(a),
                        bits_f32(b),
                        bits_f32(c),
                        bits_f32(d),
                    )
                    waiting = cutlass.Boolean(False)
    cute.arch.sync_threads()
    materialize = cutlass.Boolean(True)
    if cutlass.const_expr(second):
        materialize = not x_only
    rstd = cutlass.Float32(0)
    if materialize:
        total = ((cutlass.Float32(0) + scalars[8]) + scalars[9]) + scalars[10]
        total += scalars[11]
        rstd = cute.math.rsqrt(total / 6144.0 + 1.0e-5, fastmath=True)
    if active:
        if materialize:
            hv = cute.make_rmem_tensor((8,), cutlass.Float32)
            packed_h = cute.make_rmem_tensor((4,), cutlass.Uint32)
            for j in cutlass.range_constexpr(4):
                hv[2 * j] = round_bf16(
                    (residual[2 * j] * rstd) * bf16_from_word(gam[j], False)
                )
                hv[2 * j + 1] = round_bf16(
                    (residual[2 * j + 1] * rstd) * bf16_from_word(gam[j], True)
                )
                packed_h[j] = bf16_pair(hv[2 * j], hv[2 * j + 1])
            if cutlass.const_expr(second):
                store_vector(
                    normalized_output.iterator + token * 3072 + col // 2,
                    packed_h[0],
                    packed_h[1],
                    packed_h[2],
                    packed_h[3],
                )
            else:
                store_vector(
                    h.iterator + token * 3072 + col // 2,
                    published_word(packed_h[0]),
                    published_word(packed_h[1]),
                    published_word(packed_h[2]),
                    published_word(packed_h[3]),
                )
            if cutlass.const_expr(not second and not lean):
                maximum = cutlass.Float32(0)
                for j in cutlass.range_constexpr(8):
                    maximum = max_num(maximum, cute.math.abs(hv[j]))
                maximum = max_num(maximum, cute.arch.shuffle_sync_bfly(maximum, 1))
                sf_value = in_scale * (maximum * rcp_six)
                sf = e4m3_pair(sf_value, cutlass.Float32(0)) & cutlass.Uint32(255)
                sf_float = e4m3_float(sf)
                output_scale = cutlass.Float32(0)
                if maximum != 0.0:
                    output_scale = cute.arch.rcp_approx(sf_float * in_rcp)
                packed = cutlass.Uint32(0)
                for j in cutlass.range_constexpr(4):
                    packed |= fp4_pair(
                        hv[2 * j] * output_scale, hv[2 * j + 1] * output_scale
                    ) << (8 * j)
                store_word(
                    hfp4.iterator + token * 768 + col // 8, published_word(packed)
                )
                s1 = cute.arch.shuffle_sync_down(sf, 2)
                s2 = cute.arch.shuffle_sync_down(sf, 4)
                s3 = cute.arch.shuffle_sync_down(sf, 6)
                if (tid & 7) == 0:
                    store_word(
                        hfp4.iterator + 6144 + token * 96 + col // 64,
                        published_word(sf | (s1 << 8) | (s2 << 16) | (s3 << 24)),
                    )


# Imported last: the role modules import each other, and their names are used only when a kernel is traced.
from .attention import (
    bf16_from_word,
    bf16_pair,
    e4m3_pair,
    published_word,
    volatile_vector,
)
from .gpu_publication import global_pair, readonly_vector, store_vector, store_word
from .indexer import round_bf16
from .moe import warm_moe
