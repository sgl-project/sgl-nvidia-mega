# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
import cutlass
from cutlass._mlir.dialects import arith, llvm, nvvm, vector
from cutlass._mlir import ir
from cutlass.cutlass_dsl import T, dsl_user_op
import cutlass.cute as cute


@cute.jit
def page_row(page: cutlass.Int32, position: cutlass.Int32, stride_rows: cutlass.Int64, layer_rows: cutlass.Int64):
    return page.to(cutlass.Int64) * stride_rows + layer_rows + (position & 63)


@dsl_user_op
def mma_bf16(c0, c1, c2, c3, a0, a1, a2, a3, b0, b1, *, loc=None, ip=None):
    f32x4 = llvm.StructType.get_literal([T.f32()] * 4)
    a = [cutlass.Uint32(x).ir_value(loc=loc, ip=ip) for x in (a0, a1, a2, a3)]
    b = [cutlass.Uint32(x).ir_value(loc=loc, ip=ip) for x in (b0, b1)]
    c = [cutlass.Float32(x).ir_value(loc=loc, ip=ip) for x in (c0, c1, c2, c3)]
    attr = ir.Attribute.parse
    result = nvvm.mma_sync(f32x4, attr("#nvvm.shape<m = 16, n = 8, k = 16>"), attr("#nvvm.mma_layout<row>"),
                           attr("#nvvm.mma_layout<col>"), a, b, c, multiplicand_a_ptx_type=attr("#nvvm.mma_type<bf16>"),
                           multiplicand_b_ptx_type=attr("#nvvm.mma_type<bf16>"), loc=loc, ip=ip)
    return tuple(cutlass.Float32(llvm.extractvalue(T.f32(), result, [j], loc=loc, ip=ip)) for j in range(4))


@dsl_user_op
def bf16_pair(lo, hi, *, loc=None, ip=None):
    pair = nvvm.convert_f32x2_to_bf16x2(ir.VectorType.get([2], ir.BF16Type.get()),
                                        cutlass.Float32(hi).ir_value(loc=loc, ip=ip),
                                        cutlass.Float32(lo).ir_value(loc=loc, ip=ip), rnd=nvvm.FPRoundingMode.RN,
                                        loc=loc, ip=ip)
    return cutlass.Uint32(llvm.bitcast(T.i32(), pair, loc=loc, ip=ip))


TOKENS = 8
LATENT = 512
WARP = 32
VALUES_PER_LANE = 16
EPSILON = 1.0e-5


EMPTY = 0x80000000


@dsl_user_op
def volatile_vector(pointer: cute.Pointer, *, loc=None, ip=None):
    v = llvm.load(ir.VectorType.get([4], T.i32()), as_ptr(1, pointer, loc, ip), volatile_=True, alignment=16, loc=loc,
                  ip=ip)
    return tuple(cutlass.Uint32(vector.extract(v, [], [i], loc=loc, ip=ip)) for i in range(4))


@dsl_user_op
def bf16_from_word(word: cutlass.Uint32, upper: bool, *, loc=None, ip=None):
    if upper:
        bits = word & cutlass.Uint32(0xffff0000)
    else:
        bits = word << 16
    return cutlass.Float32(llvm.bitcast(T.f32(), bits.ir_value(loc=loc, ip=ip), loc=loc, ip=ip))


@cute.jit
def vector_column(lane: cutlass.Int32, group: cutlass.Constexpr):
    col = 8 * lane
    if cutlass.const_expr(group == 1):
        col = 8 * (lane + 32)
    elif cutlass.const_expr(group == 2):
        col = 512 + 8 * (lane & 7)
    return col


@cute.jit
def read_payload(y: cute.Tensor, token: cutlass.Int32, lane: cutlass.Int32):
    words = cute.make_rmem_tensor((3, 4), cutlass.Uint32)
    for group in cutlass.range_constexpr(3):
        col = vector_column(lane, group)
        v0, v1, v2, v3 = volatile_vector(y.iterator + token * 2624 + col)
        words[group, 0] = v0
        words[group, 1] = v1
        words[group, 2] = v2
        words[group, 3] = v3
    waiting = cutlass.Boolean(True)
    while waiting:
        complete = cutlass.Boolean(True)
        for group in cutlass.range_constexpr(3):
            for j in cutlass.range_constexpr(4):
                complete = complete & (words[group, j] != cutlass.Uint32(EMPTY))
        if cute.arch.vote_all_sync(complete):
            waiting = cutlass.Boolean(False)
        else:
            for group in cutlass.range_constexpr(3):
                vector_complete = cutlass.Boolean(True)
                for j in cutlass.range_constexpr(4):
                    vector_complete = vector_complete & (words[group, j] != cutlass.Uint32(EMPTY))
                if not vector_complete:
                    col = vector_column(lane, group)
                    v0, v1, v2, v3 = volatile_vector(y.iterator + token * 2624 + col)
                    words[group, 0] = v0
                    words[group, 1] = v1
                    words[group, 2] = v2
                    words[group, 3] = v3
    return words


EMPTY_WORD = 0x80000000


@dsl_user_op
def e4m3_pair(lo, hi, *, loc=None, ip=None):
    p = nvvm.convert_f32x2_to_f8x2(T.i16(), cutlass.Float32(hi).ir_value(loc=loc, ip=ip),
                                   cutlass.Float32(lo).ir_value(loc=loc, ip=ip),
                                   ir.TypeAttr.get(ir.Type.parse('f8E4M3FN')), rnd=nvvm.FPRoundingMode.RN,
                                   sat=nvvm.SaturationMode.SATFINITE, loc=loc, ip=ip)
    return cutlass.Uint32(arith.extui(T.i32(), p, loc=loc, ip=ip))


@cute.jit
def published_word(word: cutlass.Uint32):
    result = word
    if word == cutlass.Uint32(EMPTY_WORD):
        result = cutlass.Uint32(0)
    return result


@cute.jit
def append_stage(
    y: cute.Tensor, gamma: cute.Tensor, rope: cute.Tensor,
    length: cute.Tensor, table: cute.Tensor, cache: cute.Tensor,
    arena: cute.Tensor, stride_rows: cutlass.Int64, layer_rows: cutlass.Int64,
    first_token=None, clamp_pos: cutlass.Constexpr = False,
):
    tid, _, _ = cute.arch.thread_idx()
    token = tid // WARP
    lane = tid % WARP
    pos = length[0] - TOKENS + token
    if cutlass.const_expr(clamp_pos):
        if pos < 0:
            pos = cutlass.Int32(0)
    page = table[pos >> 6]
    row = page_row(page, pos, stride_rows, layer_rows)
    if cutlass.const_expr(rope.stride[1] != 1 or cache.stride[1] != 1 or arena.stride[1] != 1):
        raise ValueError("RoPE, cache and arena rows must be unit-stride")
    gamma_words = cute.recast_ptr(gamma.iterator, dtype=cutlass.Uint32)
    gam = cute.make_rmem_tensor((16,), cutlass.Float32)
    for half in cutlass.range_constexpr(2):
        gw = readonly_vector(gamma_words + (4 * lane + 128 * half))
        for j in cutlass.range_constexpr(8):
            gam[8 * half + j] = bf16_from_word(gw[j // 2], j % 2 == 1)
    cs = cute.make_rmem_tensor((8,), cutlass.Float32)
    for j in cutlass.range_constexpr(8):
        cs[j] = cutlass.Float32(0)
    if lane < 8:
        rope_words = cute.recast_ptr(rope.iterator + pos.to(cutlass.Int64) * rope.stride[0],
                                     dtype=cutlass.Uint32)
        for q in cutlass.range_constexpr(2):
            rw = readonly_vector(rope_words + (8 * lane + 4 * q))
            for e in cutlass.range_constexpr(4):
                cs[4 * q + e] = bits_f32(rw[e])
    payload = read_payload(y, token, lane)
    values = cute.make_rmem_tensor((VALUES_PER_LANE,), cutlass.Float32)
    ssq = cutlass.Float32(0.0)
    for j in cutlass.range_constexpr(VALUES_PER_LANE):
        col = 8 * lane + j % 8 + (j // 8) * 256
        value = bf16_from_word(payload[j // 8, (j % 8) // 2], j % 2 == 1)
        values[j] = value
        ssq += value * value
    for step in cutlass.range_constexpr(5):
        ssq += cute.arch.shuffle_sync_bfly(ssq, 16 >> step)
    rstd = cute.math.rsqrt(ssq / LATENT + EPSILON, fastmath=True)
    dst = cache.iterator + row * cache.stride[0]
    dst_arena = arena.iterator + token * arena.stride[0]
    words = cute.make_rmem_tensor((4,), cutlass.Uint32)
    for q in cutlass.range_constexpr(4):
        j = 4 * q
        v0 = values[j] * (rstd * gam[j])
        v1 = values[j + 1] * (rstd * gam[j + 1])
        v2 = values[j + 2] * (rstd * gam[j + 2])
        v3 = values[j + 3] * (rstd * gam[j + 3])
        words[q] = published_word(e4m3_pair(v0, v1) | (e4m3_pair(v2, v3) << 16))
    writes = cutlass.Boolean(True)
    if cutlass.const_expr(first_token is not None):
        writes = token >= first_token
    if writes:
        for half in cutlass.range_constexpr(2):
            store_pair(dst + (2 * lane + 64 * half), words[2 * half], words[2 * half + 1])
            store_pair(dst_arena + (2 * lane + 64 * half), words[2 * half], words[2 * half + 1])
        if lane < 8:
            rope_words_out = cute.make_rmem_tensor((2,), cutlass.Uint32)
            for q in cutlass.range_constexpr(2):
                k0 = bf16_from_word(payload[2, 2 * q + 0], False)
                k1 = bf16_from_word(payload[2, 2 * q + 0], True)
                k2 = bf16_from_word(payload[2, 2 * q + 1], False)
                k3 = bf16_from_word(payload[2, 2 * q + 1], True)
                c0, s0, c1, s1 = cs[4 * q], cs[4 * q + 1], cs[4 * q + 2], cs[4 * q + 3]
                r0 = fma_rn(c0, k0, -mul_rn(s0, k1))
                r1 = fma_rn(s0, k0, mul_rn(c0, k1))
                r2 = fma_rn(c1, k2, -mul_rn(s1, k3))
                r3 = fma_rn(s1, k2, mul_rn(c1, k3))
                rope_words_out[q] = published_word(e4m3_pair(r0, r1) | (e4m3_pair(r2, r3) << 16))
            store_pair(dst + (128 + 2 * lane), rope_words_out[0], rope_words_out[1])
            store_pair(dst_arena + (128 + 2 * lane), rope_words_out[0], rope_words_out[1])


@dsl_user_op
def mma_f16(c0, c1, c2, c3, a0, a1, a2, a3, b0, b1, *, loc=None, ip=None):
    f32x4 = llvm.StructType.get_literal([T.f32()] * 4)
    a = [llvm.bitcast(ir.VectorType.get([2], ir.F16Type.get()), cutlass.Uint32(x).ir_value(loc=loc, ip=ip), loc=loc,
                      ip=ip) for x in (a0, a1, a2, a3)]
    b = [llvm.bitcast(ir.VectorType.get([2], ir.F16Type.get()), cutlass.Uint32(x).ir_value(loc=loc, ip=ip), loc=loc,
                      ip=ip) for x in (b0, b1)]
    c = [cutlass.Float32(x).ir_value(loc=loc, ip=ip) for x in (c0, c1, c2, c3)]
    attr = ir.Attribute.parse
    result = nvvm.mma_sync(f32x4, attr("#nvvm.shape<m = 16, n = 8, k = 16>"), attr("#nvvm.mma_layout<row>"),
                           attr("#nvvm.mma_layout<col>"), a, b, c, multiplicand_a_ptx_type=attr("#nvvm.mma_type<f16>"),
                           multiplicand_b_ptx_type=attr("#nvvm.mma_type<f16>"), loc=loc, ip=ip)
    return tuple(cutlass.Float32(llvm.extractvalue(T.f32(), result, [j], loc=loc, ip=ip)) for j in range(4))


@dsl_user_op
def ldmatrix4(address: cutlass.Uint32, trans: bool = False, *, loc=None, ip=None):
    shared = as_ptr(3, address, loc, ip)
    result = nvvm.ldmatrix(shared, 4, ir.Attribute.parse(f"#nvvm.mma_layout<{'col' if trans else 'row'}>"),
                           results=[llvm.StructType.get_literal([T.i32()] * 4)], loc=loc, ip=ip)
    return tuple(cutlass.Uint32(llvm.extractvalue(T.i32(), result, [i], loc=loc, ip=ip)) for i in range(4))


@dsl_user_op
def ldmatrix2(address: cutlass.Uint32, *, loc=None, ip=None):
    shared = as_ptr(3, address, loc, ip)
    result = nvvm.ldmatrix(shared, 2, ir.Attribute.parse("#nvvm.mma_layout<row>"),
                           results=[llvm.StructType.get_literal([T.i32()] * 2)], loc=loc, ip=ip)
    return tuple(cutlass.Uint32(llvm.extractvalue(T.i32(), result, [i], loc=loc, ip=ip)) for i in range(2))


@dsl_user_op
def fp8_half_pair(bits: cutlass.Uint32, *, loc=None, ip=None):
    return cutlass.Uint32(llvm.bitcast(T.i32(), e4m3x2_to_f16x2_ir(low_16(bits, loc, ip), loc, ip), loc=loc, ip=ip))


@dsl_user_op
def permute_bytes(a: cutlass.Uint32, b: cutlass.Uint32, selector: int, *, loc=None, ip=None):
    return cutlass.Uint32(cute.arch.prmt(a, b, cutlass.Int32(selector), loc=loc, ip=ip))


@dsl_user_op
def shared_store4(address: cutlass.Uint32, a: cutlass.Uint32, b: cutlass.Uint32,
                    c: cutlass.Uint32, d: cutlass.Uint32, *, loc=None, ip=None):
    llvm.store(word_vector((a, b, c, d,), loc, ip), as_ptr(3, address, loc, ip), alignment=16, loc=loc, ip=ip)


EMPTY = 0x80000000


@cute.jit
def x_word(token: cutlass.Int32, k: cutlass.Int32, fragment: cutlass.Constexpr):
    offset = token * 3072 + k // 2
    if cutlass.const_expr(fragment):
        offset = (k >> 6) * 256 + ((k >> 3) & 1) * 128 + (4 * token + ((k >> 4) & 3)) * 4 + (k & 7) // 2
    return offset


@cute.jit
def mma8_operands(b: cute.Tensor, source: cute.Tensor, fragment: cutlass.Constexpr, poll: cutlass.Constexpr):
    tid, _, _ = cute.arch.thread_idx()
    warp, lane = tid >> 5, tid & 31
    g, c = lane >> 2, lane & 3
    flat = cute.make_tensor(source.iterator, cute.make_layout((24576,)))
    for h in cutlass.range_constexpr(16):
        for e in cutlass.range_constexpr(2):
            offset = x_word(g, 256 * warp + 16 * h + 2 * c + 8 * e, fragment)
            if cutlass.const_expr(poll):
                b[2 * h + e] = volatile_word(source.iterator + offset)
            else:
                b[2 * h + e] = flat[offset]
    for j in cutlass.range_constexpr(2):
        for p in cutlass.range_constexpr(8):
            offset = x_word(g, 256 * (warp + 8 * (j + 1)) + 32 * p + 8 * c, fragment)
            if cutlass.const_expr(poll):
                v0, v1, v2, v3 = volatile_vector(source.iterator + offset)
            else:
                v0, v1, v2, v3 = cached_vector(source.iterator + offset)
            i = 32 + 32 * j + 4 * p
            b[i], b[i + 1], b[i + 2], b[i + 3] = v0, v1, v2, v3
    if cutlass.const_expr(poll):
        waiting = cutlass.Boolean(True)
        while waiting:
            full = cutlass.Boolean(True)
            for i in cutlass.range_constexpr(96):
                full = full & (b[i] != cutlass.Uint32(EMPTY))
            if cute.arch.vote_all_sync(full):
                waiting = cutlass.Boolean(False)
            else:
                for h in cutlass.range_constexpr(16):
                    for e in cutlass.range_constexpr(2):
                        if b[2 * h + e] == cutlass.Uint32(EMPTY):
                            b[2 * h + e] = volatile_word(source.iterator + x_word(g, 256 * warp + 16 * h + 2 * c + 8 * e,
                                                                                  fragment))
                for j in cutlass.range_constexpr(2):
                    for p in cutlass.range_constexpr(8):
                        i = 32 + 32 * j + 4 * p
                        ready = (b[i] != cutlass.Uint32(EMPTY)) & (b[i + 1] != cutlass.Uint32(EMPTY)) & \
                            (b[i + 2] != cutlass.Uint32(EMPTY)) & (b[i + 3] != cutlass.Uint32(EMPTY))
                        if not ready:
                            v0, v1, v2, v3 = volatile_vector(
                                source.iterator + x_word(g, 256 * (warp + 8 * (j + 1)) + 32 * p + 8 * c, fragment))
                            b[i], b[i + 1], b[i + 2], b[i + 3] = v0, v1, v2, v3


@cute.jit
def mma8_mainloop(region: cute.Tensor, b: cute.Tensor, acc: cute.Tensor):
    tid, _, _ = cute.arch.thread_idx()
    warp, lane = tid >> 5, tid & 31
    g, c = lane >> 2, lane & 3
    region_base = region.iterator.toint().to(cutlass.Uint32)
    for h in cutlass.range_constexpr(16):
        base = (4 * warp + h // 4) * 512 + 4 * (4 * g + h % 4) + c
        a0 = region[base]
        a2 = region[base + 128]
        a1 = region[base + 256]
        a3 = region[base + 384]
        e = h % 2
        c0, c1, c2, c3 = mma_bf16(acc[e, 0], acc[e, 1], acc[e, 2], acc[e, 3], a0, a1, a2, a3, b[2 * h], b[2 * h + 1])
        acc[e, 0], acc[e, 1], acc[e, 2], acc[e, 3] = c0, c1, c2, c3
    for j in cutlass.range_constexpr(2):
        stage = warp + 8 * (j + 1)
        for p in cutlass.range_constexpr(8):
            word = (4 * stage + p // 2) * 512 + (c & 1) * 128 + 4 * (4 * g + 2 * (p % 2) + c // 2)
            l0, l1, l2, l3 = lds128(region_base + (word * 4).to(cutlass.Uint32))
            h0, h1, h2, h3 = lds128(region_base + ((word + 256) * 4).to(cutlass.Uint32))
            i = 32 + 32 * j + 4 * p
            c0, c1, c2, c3 = mma_bf16(acc[0, 0], acc[0, 1], acc[0, 2], acc[0, 3], l0, h0, l1, h1, b[i], b[i + 1])
            acc[0, 0], acc[0, 1], acc[0, 2], acc[0, 3] = c0, c1, c2, c3
            c0, c1, c2, c3 = mma_bf16(acc[1, 0], acc[1, 1], acc[1, 2], acc[1, 3], l2, h2, l3, h3, b[i + 2], b[i + 3])
            acc[1, 0], acc[1, 1], acc[1, 2], acc[1, 3] = c0, c1, c2, c3


@cute.jit
def issue_o(unit: cutlass.Int32, wo: cute.Tensor, wlg: cute.Tensor,
            storage: cute.Tensor, bar: cute.Pointer, folded: cutlass.Constexpr):
    count = 131072
    if cutlass.const_expr(folded):
        count = 196608
    cute.arch.mbarrier_arrive_and_expect_tx(bar, count)
    base = storage.iterator.toint().to(cutlass.Uint32)
    policy = evict_first()
    bulk_copy(base, wo.iterator + unit * 32768, cutlass.Uint32(131072), bar, policy)
    if cutlass.const_expr(folded):
        bulk_copy(base + cutlass.Uint32(131072), wlg.iterator + (unit - 148) * 38 * 512,
                   cutlass.Uint32(65536), bar, policy)


@cute.jit
def chunk_mma(acc: cute.Tensor, tile: cutlass.Constexpr, a: cute.Tensor,
              b: cute.Tensor, b_chunk: cutlass.Constexpr):
    for step in cutlass.range_constexpr(4):
        index = (step // 2) * 4 + (step % 2) * 2
        v0, v1, v2, v3 = mma_bf16(acc[tile, 0], acc[tile, 1], acc[tile, 2], acc[tile, 3],
                                  a[index], a[index + 8], a[index + 1], a[index + 9],
                                  b[b_chunk, 2 * step], b[b_chunk, 2 * step + 1])
        acc[tile, 0], acc[tile, 1], acc[tile, 2], acc[tile, 3] = v0, v1, v2, v3


@cute.jit
def read_weight(a: cute.Tensor, region: cute.Tensor, chunk: cutlass.Int32):
    tid, _, _ = cute.arch.thread_idx()
    base = region.iterator.toint().to(cutlass.Uint32)
    for piece in cutlass.range_constexpr(4):
        w0, w1, w2, w3 = lds128(base + ((chunk * 512 + piece * 128 + 4 * (tid & 31)) * 4).to(cutlass.Uint32))
        a[4 * piece], a[4 * piece + 1], a[4 * piece + 2], a[4 * piece + 3] = w0, w1, w2, w3


@cute.jit
def reduce_tiles(acc: cute.Tensor, red: cute.Tensor, tiles: cutlass.Constexpr,
                    output_offset: cutlass.Constexpr = 2048):
    tid, _, _ = cute.arch.thread_idx()
    warp, lane = tid // 32, tid & 31
    for tile in cutlass.range_constexpr(tiles):
        for r in cutlass.range_constexpr(4):
            red[((warp * tiles + tile) * 32 + lane) * 4 + r] = acc[tile, r]
    cute.arch.sync_threads()
    for i in cutlass.range_constexpr((tiles * 128 + 255) // 256):
        index = tid + i * 256
        if index < tiles * 128:
            tile, owner, r = index // 128, (index // 4) % 32, index % 4
            value = cutlass.Float32(0)
            for w in cutlass.range_constexpr(8):
                value += red[((w * tiles + tile) * 32 + owner) * 4 + r]
            row = 16 * tile + (owner >> 2) + 8 * (r // 2)
            token = 2 * (owner & 3) + (r & 1)
            red[output_offset + row * 8 + token] = value
    cute.arch.sync_threads()


@cute.jit
def reduce_logits(unit: cutlass.Int32, lg_uc: cute.Tensor, logits: cute.Tensor):
    tid, _, _ = cute.arch.thread_idx()
    lane = tid & 31
    payload = cute.make_rmem_tensor((16, 4), cutlass.Uint32)
    for rank in cutlass.range_constexpr(16):
        a, b, c, d = volatile_vector(lg_uc.iterator + (rank * 16 + unit) * 128 + 4 * lane)
        payload[rank, 0], payload[rank, 1], payload[rank, 2], payload[rank, 3] = a, b, c, d
    waiting = cutlass.Boolean(True)
    while waiting:
        full = cutlass.Boolean(True)
        for rank in cutlass.range_constexpr(16):
            for j in cutlass.range_constexpr(4):
                full = full & (payload[rank, j] != cutlass.Uint32(0x80000000))
        if cute.arch.vote_all_sync(full):
            waiting = cutlass.Boolean(False)
        else:
            for rank in cutlass.range_constexpr(16):
                ready = cutlass.Boolean(True)
                for j in cutlass.range_constexpr(4):
                    ready = ready & (payload[rank, j] != cutlass.Uint32(0x80000000))
                if not ready:
                    a, b, c, d = volatile_vector(lg_uc.iterator + (rank * 16 + unit) * 128 + 4 * lane)
                    payload[rank, 0], payload[rank, 1], payload[rank, 2], payload[rank, 3] = a, b, c, d
    result = cute.make_rmem_tensor((4,), cutlass.Uint32)
    for j in cutlass.range_constexpr(4):
        low, high = cutlass.Float32(0), cutlass.Float32(0)
        for rank in cutlass.range_constexpr(8):
            low += bits_f32(payload[rank, j])
            high += bits_f32(payload[rank + 8, j])
        result[j] = published_word(f32_bits((cutlass.Float32(0) + low) + high))
    store_vector(logits.iterator + (lane >> 2) * 256 + 16 * unit + 4 * (lane & 3), result[0], result[1], result[2], result[3])
    policy = evict_first()
    for rank in cutlass.range_constexpr(16):
        store_empty(lg_uc.iterator + (rank * 16 + unit) * 128 + 4 * lane, policy)


@cute.jit
def o_stage(unit: cutlass.Int32, rank: cutlass.Int32, o: cute.Tensor,
            replicas: cute.Tensor, residual: cute.Tensor, wlg: cute.Tensor,
            region: cute.Tensor, red: cute.Tensor, bar: cute.Pointer,
            h1_mc: cute.Tensor, lg_uc: cute.Tensor, lg_mc: cute.Tensor,
            logits: cute.Tensor, tmem: cutlass.Uint32, warm_bar: cute.Pointer,
            folded: cutlass.Constexpr, only_token=None, warm: cutlass.Constexpr = True, passes=None):
    if cutlass.const_expr(passes is None):
        o_pass(unit, rank, o, replicas, residual, wlg, region, red, bar, h1_mc, lg_uc, lg_mc, logits, tmem, warm_bar,
               folded, only_token, warm)
    else:
        if cutlass.const_expr(folded):
            raise TypeError("o_stage(passes) is for the non-folded O units only")
        for p in cutlass.range(passes, unroll=1):
            o_pass(unit, rank, o, replicas, residual, wlg, region, red, bar, h1_mc, lg_uc, lg_mc, logits, tmem,
                   warm_bar, folded, only_token, warm, (p + 1) < passes)


@cute.jit
def o_pass(unit: cutlass.Int32, rank: cutlass.Int32, o: cute.Tensor,
           replicas: cute.Tensor, residual: cute.Tensor, wlg: cute.Tensor,
           region: cute.Tensor, red: cute.Tensor, bar: cute.Pointer,
           h1_mc: cute.Tensor, lg_uc: cute.Tensor, lg_mc: cute.Tensor,
           logits: cute.Tensor, tmem: cutlass.Uint32, warm_bar: cute.Pointer,
           folded: cutlass.Constexpr, only_token=None, warm: cutlass.Constexpr = True, force=None):
    tid, _, _ = cute.arch.thread_idx()
    warp, lane = tid // 32, tid & 31
    ar = cute.make_rmem_tensor((16,), cutlass.Uint32)
    br = cute.make_rmem_tensor((1, 8), cutlass.Uint32)
    if cutlass.const_expr(folded):
        for j in cutlass.range_constexpr(16):
            ar[j] = cutlass.Uint32(0)
        for j in cutlass.range_constexpr(8):
            br[0, j] = cutlass.Uint32(0)
        if warp < 6:
            operand_words(br, residual, 3072, 6 * rank + warp, 1, False, False)
            for piece in cutlass.range_constexpr(4):
                r0, r1, r2, r3 = cached_vector(wlg.iterator + ((unit - 148) * 38 + 32 + warp) * 512 + piece * 128 + lane * 4)
                ar[4 * piece], ar[4 * piece + 1], ar[4 * piece + 2], ar[4 * piece + 3] = r0, r1, r2, r3
    b = cute.make_rmem_tensor((4, 8), cutlass.Uint32)
    if (unit & 3) == 0:
        operand_words(b, o, 1024, 4 * warp, 4, False, True, force)
    else:
        source = cute.make_tensor(replicas.iterator + ((unit & 3) - 1) * 8192, cute.make_layout((8192,)))
        operand_words(b, source, 1024, 4 * warp, 4, False, True, force)
    cute.arch.mbarrier_wait(bar, 0)
    acc = cute.make_rmem_tensor((2, 4), cutlass.Float32)
    a = cute.make_rmem_tensor((16,), cutlass.Uint32)
    for tile in cutlass.range_constexpr(2):
        for r in cutlass.range_constexpr(4):
            acc[tile, r] = cutlass.Float32(0)
    for i in cutlass.range_constexpr(4):
        for tile in cutlass.range_constexpr(2):
            read_weight(a, region, tile * 32 + 4 * warp + i)
            chunk_mma(acc, tile, a, b, i)
    reduce_tiles(acc, red, 2)
    publish = tid < 32
    if cutlass.const_expr(only_token is not None):
        publish = publish & ((tid // 4) == only_token)
    if cutlass.const_expr(force is not None):
        if force:
            publish = cutlass.Boolean(False)
    if publish:
        token, vector = tid // 4, tid % 4
        words = cute.make_rmem_tensor((4,), cutlass.Uint32)
        for j in cutlass.range_constexpr(4):
            col = 8 * vector + 2 * j
            words[j] = published_word(bf16_pair(red[2048 + col * 8 + token], red[2048 + (col + 1) * 8 + token]))
        store_vector(h1_mc.iterator + hop1_word_offset(token, rank, 32 * unit + 8 * vector), words[0], words[1], words[2], words[3])
    if cutlass.const_expr(folded):
        acc1 = cute.make_rmem_tensor((1, 4), cutlass.Float32)
        for r in cutlass.range_constexpr(4):
            acc1[0, r] = cutlass.Float32(0)
        for i in cutlass.range_constexpr(4):
            read_weight(a, region, 64 + 4 * warp + i)
            chunk_mma(acc1, 0, a, b, i)
        if warp < 6:
            chunk_mma(acc1, 0, ar, br, 0)
        reduce_tiles(acc1, red, 1)
        if tid < 32:
            token, first = tid // 4, 4 * (tid & 3)
            words = cute.make_rmem_tensor((4,), cutlass.Uint32)
            for j in cutlass.range_constexpr(4):
                words[j] = published_word(f32_bits(red[2048 + (first + j) * 8 + token]))
            store_vector(lg_mc.iterator + (rank * 16 + unit - 148) * 128 + 4 * tid, words[0], words[1], words[2], words[3])
        if cutlass.const_expr(warm):
            warm_moe(region, tmem, warm_bar)
        if tid < 32:
            reduce_logits(unit - 148, lg_uc, logits)


@cute.jit
def permuted_slot(key: cutlass.Int32):
    u = key & 15
    slot = ((u >> 1) * 4 + (u & 1))
    if u >= 8:
        slot = (((u - 8) >> 1) * 4 + 2 + (u & 1))
    return (key & ~15) + slot


@cute.jit
def prefetch_plain_o(unit: cutlass.Int32, weights: cute.Tensor,
                       storage: cute.Tensor, bar: cute.Pointer):
    cute.arch.mbarrier_arrive_and_expect_tx(bar, 131072)
    policy = evict_first()
    bulk_copy(storage.iterator.toint().to(cutlass.Uint32), weights.iterator + unit * 32768,
               cutlass.Uint32(131072), bar, policy)


@cute.jit
def attention_stage(unit: cutlass.Int32, q8: cute.Tensor, fresh_kv: cute.Tensor,
                      length: cute.Tensor, table: cute.Tensor, topk: cute.Tensor,
                      cache: cute.Tensor, stride_rows: cutlass.Int64,
                      layer_rows: cutlass.Int64, max_blocks: cutlass.Int32,
                      storage: cute.Tensor, bars: cute.Tensor,
                      partials: cute.Tensor, stats: cute.Tensor,
                      o_weights=None, o_bar=None, leader_exchange=None,
                      leader_flag=None, fresh_window=None, topk_select=None):
    tid, _, _ = cute.arch.thread_idx()
    token, block = unit // 16, unit % 16
    warp, lane = tid >> 5, tid & 31
    smem_base = storage.iterator.toint().to(cutlass.Uint32)
    kv_base = smem_base + cutlass.Uint32(131072)
    q_base = kv_base + cutlass.Uint32(75776)
    p_base = q_base + cutlass.Uint32(4736)
    max_view = cute.make_tensor(cute.recast_ptr(storage.iterator + 53184, dtype=cutlass.Float32), cute.make_layout((64,)))
    sum_view = cute.make_tensor(cute.recast_ptr(storage.iterator + 53248, dtype=cutlass.Float32), cute.make_layout((64,)))
    fresh_view = cute.make_tensor(cute.recast_ptr(storage.iterator + 53312, dtype=cutlass.Int32), cute.make_layout((128,)))
    p_bytes = cute.make_tensor(cute.recast_ptr(storage.iterator, dtype=cutlass.Uint8), cute.make_layout((223232,)))
    p_offset = cutlass.Int32(211584)
    gather_bar = bars.iterator + 16 + warp
    L = length[0]
    num_valid = L - 7 + token
    if num_valid > 2048:
        num_valid = cutlass.Int32(2048)
    part_base = (token * 16 + block) * 4 * 512
    neg_inf = bits_f32(cutlass.Uint32(0xFF800000))
    if block * 128 >= num_valid:
        if cutlass.const_expr(topk_select is not None):
            leads = cutlass.Boolean(True)
            if cutlass.const_expr(leader_flag is not None):
                leads = leader_flag
            if leads & (token == topk_select) & (lane < 16):
                for r in cutlass.range_constexpr(8):
                    store_word(topk.iterator + r * 2048 + block * 128 + warp * 16 + lane, cutlass.Uint32(0xFFFFFFFF))
        if cutlass.const_expr(o_weights is not None):
            if tid == 0:
                prefetch_plain_o(unit, o_weights, storage, o_bar)
        for iteration in cutlass.range_constexpr(4):
            pair = tid + 256 * iteration
            store_pair(partials.iterator + part_base + 2 * pair, cutlass.Uint32(0), cutlass.Uint32(0))
        if tid < 4:
            store_pair(stats.iterator + ((token * 16 + block) * 4 + tid) * 2,
                        cutlass.Uint32(0xFF800000), cutlass.Uint32(0))
    else:
        slot = block * 128 + warp * 16 + lane
        position = cutlass.Int32(-1)
        if cutlass.const_expr(leader_exchange is None):
            if lane < 16:
                position = topk[token, slot]
        else:
            is_follower = cutlass.Boolean(False)
            if cutlass.const_expr(leader_flag is not None):
                is_follower = not leader_flag
            if is_follower:
                if lane < 16:
                    position = topk[token, slot]
            elif L <= 2048:
                if slot < num_valid:
                    position = slot
            else:
                packet = cutlass.Uint32(0)
                if lane < 16:
                    packet = volatile_word(leader_exchange.iterator + token * 2048 + slot)
                waiting_topk = cutlass.Boolean(True)
                while waiting_topk:
                    if cute.arch.vote_all_sync((lane >= 16) | (packet != cutlass.Uint32(0x80000000))):
                        waiting_topk = cutlass.Boolean(False)
                    elif (lane < 16) & (packet == cutlass.Uint32(0x80000000)):
                        packet = volatile_word(leader_exchange.iterator + token * 2048 + slot)
                position = packet.to(cutlass.Int32)
            if (lane < 16) & (not is_follower):
                if cutlass.const_expr(topk_select is None):
                    store_word(topk.iterator + token * 2048 + slot, position.to(cutlass.Uint32))
                elif token == topk_select:
                    for r in cutlass.range_constexpr(8):
                        store_word(topk.iterator + r * 2048 + slot, position.to(cutlass.Uint32))
        valid, fresh = cutlass.Boolean(False), cutlass.Boolean(False)
        fresh_index = cutlass.Int32(0)
        row = cutlass.Int64(0)
        if lane < 16:
            page = cutlass.Int32(-1)
            logical_block = position >> 6
            if (position >= 0) & (logical_block < max_blocks):
                page = table[logical_block]
            valid = (slot < num_valid) & (page >= 0)
            fresh_from = L - 8
            if cutlass.const_expr(fresh_window is not None):
                fresh_from = L - fresh_window
            fresh = valid & (position >= fresh_from)
            fresh_index = position - (L - 8)
            if valid:
                row = page_row(page, position, stride_rows, layer_rows)
        valid_mask = cute.arch.vote_ballot_sync(valid)
        fresh_mask = cute.arch.vote_ballot_sync(fresh)
        if lane == 0:
            cute.arch.mbarrier_arrive_and_expect_tx(gather_bar, cute.arch.popc(valid_mask & ~fresh_mask) * 576)
        cute.arch.sync_warp()
        if (lane < 16) & (not valid):
            zero = cutlass.Uint32(0)
            for batch in cutlass.range(9, unroll=1):
                for j in cutlass.range_constexpr(4):
                    vector = 4 * batch + j
                    shared_store4(kv_base + ((warp * 16 + lane) * 592 + 16 * vector).to(cutlass.Uint32), zero, zero, zero, zero)
        if valid & (not fresh):
            proxy_fence()
            policy = evict_first()
            bulk_copy(kv_base + ((warp * 16 + lane) * 592).to(cutlass.Uint32), cache.iterator + row * 144,
                       cutlass.Uint32(576), gather_bar, policy)
        if cutlass.const_expr(o_weights is not None):
            if tid == 0:
                prefetch_plain_o(unit, o_weights, storage, o_bar)
        zero_q_vector = 144 + tid
        if zero_q_vector < 288:
            zero = cutlass.Uint32(0)
            shared_store4(q_base + ((zero_q_vector // 36) * 592 + 16 * (zero_q_vector % 36)).to(cutlass.Uint32), zero, zero, zero, zero)
        if fresh:
            rank = cute.arch.popc(fresh_mask & ((cutlass.Uint32(1) << lane) - cutlass.Uint32(1)))
            fresh_view[warp * 16 + 2 * rank] = lane
            fresh_view[warp * 16 + 2 * rank + 1] = fresh_index
        cute.arch.sync_warp()
        n_vectors = cute.arch.popc(fresh_mask) * 36
        query = cute.make_rmem_tensor((4,), cutlass.Uint32)
        new_rows = cute.make_rmem_tensor((9, 4), cutlass.Uint32)
        for j in cutlass.range_constexpr(4):
            query[j] = cutlass.Uint32(0)
        if tid < 144:
            v0, v1, v2, v3 = volatile_vector(q8.iterator + token * 576 + 4 * tid)
            query[0], query[1], query[2], query[3] = v0, v1, v2, v3
        for i in cutlass.range_constexpr(9):
            vector = lane + 32 * i
            for j in cutlass.range_constexpr(4):
                new_rows[i, j] = cutlass.Uint32(0)
            if vector < n_vectors:
                source_row = fresh_view[warp * 16 + 2 * (vector // 36) + 1]
                v0, v1, v2, v3 = volatile_vector(fresh_kv.iterator + source_row * 144 + 4 * (vector % 36))
                new_rows[i, 0], new_rows[i, 1], new_rows[i, 2], new_rows[i, 3] = v0, v1, v2, v3
        waiting = cutlass.Boolean(True)
        while waiting:
            full = cutlass.Boolean(True)
            for j in cutlass.range_constexpr(4):
                full = full & (query[j] != cutlass.Uint32(0x80000000))
            for i in cutlass.range_constexpr(9):
                for j in cutlass.range_constexpr(4):
                    full = full & (new_rows[i, j] != cutlass.Uint32(0x80000000))
            if cute.arch.vote_all_sync(full):
                waiting = cutlass.Boolean(False)
            else:
                q_full = cutlass.Boolean(True)
                for j in cutlass.range_constexpr(4):
                    q_full = q_full & (query[j] != cutlass.Uint32(0x80000000))
                if not q_full:
                    v0, v1, v2, v3 = volatile_vector(q8.iterator + token * 576 + 4 * tid)
                    query[0], query[1], query[2], query[3] = v0, v1, v2, v3
                for i in cutlass.range_constexpr(9):
                    complete = cutlass.Boolean(True)
                    for j in cutlass.range_constexpr(4):
                        complete = complete & (new_rows[i, j] != cutlass.Uint32(0x80000000))
                    if not complete:
                        vector = lane + 32 * i
                        source_row = fresh_view[warp * 16 + 2 * (vector // 36) + 1]
                        v0, v1, v2, v3 = volatile_vector(fresh_kv.iterator + source_row * 144 + 4 * (vector % 36))
                        new_rows[i, 0], new_rows[i, 1], new_rows[i, 2], new_rows[i, 3] = v0, v1, v2, v3
        if tid < 144:
            shared_store4(q_base + ((tid // 36) * 592 + 16 * (tid % 36)).to(cutlass.Uint32), query[0], query[1], query[2], query[3])
        for i in cutlass.range_constexpr(9):
            vector = lane + 32 * i
            if vector < n_vectors:
                destination_row = fresh_view[warp * 16 + 2 * (vector // 36)]
                shared_store4(kv_base + ((warp * 16 + destination_row) * 592 + 16 * (vector % 36)).to(cutlass.Uint32),
                               new_rows[i, 0], new_rows[i, 1], new_rows[i, 2], new_rows[i, 3])
        cute.arch.sync_threads()
        g, c = lane // 4, lane % 4
        q_address = q_base + ((lane % 8) * 592 + 16 * ((lane // 8) % 2)).to(cutlass.Uint32)
        k_address = kv_base + ((warp * 16 + lane % 8 + 8 * ((lane // 8) % 2)) * 592 + 16 * (lane // 16)).to(cutlass.Uint32)
        cute.arch.mbarrier_wait(gather_bar, 0)
        acc = cute.make_rmem_tensor((4, 4), cutlass.Float32)
        for chain in cutlass.range_constexpr(4):
            for r in cutlass.range_constexpr(4):
                acc[chain, r] = cutlass.Float32(0)
        for batch in cutlass.range(9, unroll=1):
            for parity in cutlass.range_constexpr(2):
                kd = 2 * batch + parity
                b0, b1 = ldmatrix2(q_address + (32 * kd).to(cutlass.Uint32))
                a0, a1, a2, a3 = ldmatrix4(k_address + (32 * kd).to(cutlass.Uint32))
                r0, r1, r2, r3 = mma_f16(acc[parity, 0], acc[parity, 1], acc[parity, 2], acc[parity, 3],
                                        fp8_half_pair(a0), fp8_half_pair(a1), fp8_half_pair(a2), fp8_half_pair(a3), fp8_half_pair(b0), fp8_half_pair(b1))
                acc[parity, 0], acc[parity, 1], acc[parity, 2], acc[parity, 3] = r0, r1, r2, r3
                r0, r1, r2, r3 = mma_f16(acc[2 + parity, 0], acc[2 + parity, 1], acc[2 + parity, 2], acc[2 + parity, 3],
                                        fp8_half_pair(a0 >> 16), fp8_half_pair(a1 >> 16), fp8_half_pair(a2 >> 16), fp8_half_pair(a3 >> 16), fp8_half_pair(b0 >> 16), fp8_half_pair(b1 >> 16))
                acc[2 + parity, 0], acc[2 + parity, 1], acc[2 + parity, 2], acc[2 + parity, 3] = r0, r1, r2, r3
        scores = cute.make_rmem_tensor((4,), cutlass.Float32)
        for r in cutlass.range_constexpr(4):
            scores[r] = neg_inf
            key_row = g + 8 * (r // 2)
            if ((valid_mask >> key_row) & cutlass.Uint32(1)) != cutlass.Uint32(0):
                scores[r] = ((acc[0, r] + acc[1, r]) + (acc[2, r] + acc[3, r])) * 0.0901684
        max0, max1 = max_num(scores[0], scores[2]), max_num(scores[1], scores[3])
        for step in cutlass.range_constexpr(3):
            max0 = max_num(max0, cute.arch.shuffle_sync_bfly(max0, 4 << step))
            max1 = max_num(max1, cute.arch.shuffle_sync_bfly(max1, 4 << step))
        if g == 0:
            max_view[warp * 8 + 2 * c], max_view[warp * 8 + 2 * c + 1] = max0, max1
        cute.arch.sync_threads()
        cta_max0, cta_max1 = neg_inf, neg_inf
        for w in cutlass.range_constexpr(8):
            cta_max0 = max_num(cta_max0, max_view[w * 8 + 2 * c])
            cta_max1 = max_num(cta_max1, max_view[w * 8 + 2 * c + 1])
        offset0, offset1 = cta_max0, cta_max1
        if cta_max0 == neg_inf:
            offset0 = cutlass.Float32(0)
        if cta_max1 == neg_inf:
            offset1 = cutlass.Float32(0)
        p00, p01 = cute.math.exp2(scores[0] - offset0), cute.math.exp2(scores[1] - offset1)
        p10, p11 = cute.math.exp2(scores[2] - offset0), cute.math.exp2(scores[3] - offset1)
        slot0, slot1 = permuted_slot(warp * 16 + g), permuted_slot(warp * 16 + g + 8)
        p_bytes[p_offset + 2 * c * 144 + slot0] = (e4m3_pair(p00 * 448.0, cutlass.Float32(0)) & cutlass.Uint32(255)).to(cutlass.Uint8)
        p_bytes[p_offset + (2 * c + 1) * 144 + slot0] = (e4m3_pair(p01 * 448.0, cutlass.Float32(0)) & cutlass.Uint32(255)).to(cutlass.Uint8)
        p_bytes[p_offset + 2 * c * 144 + slot1] = (e4m3_pair(p10 * 448.0, cutlass.Float32(0)) & cutlass.Uint32(255)).to(cutlass.Uint8)
        p_bytes[p_offset + (2 * c + 1) * 144 + slot1] = (e4m3_pair(p11 * 448.0, cutlass.Float32(0)) & cutlass.Uint32(255)).to(cutlass.Uint8)
        sum0, sum1 = p00 + p10, p01 + p11
        for step in cutlass.range_constexpr(3):
            sum0 += cute.arch.shuffle_sync_bfly(sum0, 4 << step)
            sum1 += cute.arch.shuffle_sync_bfly(sum1, 4 << step)
        if g == 0:
            sum_view[warp * 8 + 2 * c], sum_view[warp * 8 + 2 * c + 1] = sum0, sum1
        cute.arch.sync_threads()
        p_address = p_base + ((lane % 8) * 144 + 16 * ((lane // 8) % 2)).to(cutlass.Uint32)
        v_address = kv_base + (lane * 592 + 16 * (4 * warp)).to(cutlass.Uint32)
        pb = cute.make_rmem_tensor((4, 4), cutlass.Uint32)
        for step in cutlass.range_constexpr(4):
            b0, b1 = ldmatrix2(p_address + cutlass.Uint32(32 * step))
            pb[step, 0], pb[step, 1] = fp8_half_pair(b0), fp8_half_pair(b1)
            pb[step, 2], pb[step, 3] = fp8_half_pair(b0 >> 16), fp8_half_pair(b1 >> 16)
        for pair_block in cutlass.range(2, unroll=1):
            output = cute.make_rmem_tensor((2, 4), cutlass.Float32)
            for u in cutlass.range_constexpr(2):
                for r in cutlass.range_constexpr(4):
                    output[u, r] = cutlass.Float32(0)
            for step in cutlass.range_constexpr(4):
                for u in cutlass.range_constexpr(2):
                    d0, d1, d2, d3 = ldmatrix4(v_address + (32 * step * 592 + 16 * (2 * pair_block + u)).to(cutlass.Uint32), True)
                    a0, a1 = permute_bytes(d0, d1, 0x6420), permute_bytes(d0, d1, 0x7531)
                    a2, a3 = permute_bytes(d2, d3, 0x6420), permute_bytes(d2, d3, 0x7531)
                    r0, r1, r2, r3 = mma_f16(output[u, 0], output[u, 1], output[u, 2], output[u, 3],
                                            fp8_half_pair(a0), fp8_half_pair(a1), fp8_half_pair(a2), fp8_half_pair(a3), pb[step, 0], pb[step, 1])
                    output[u, 0], output[u, 1], output[u, 2], output[u, 3] = r0, r1, r2, r3
                    r0, r1, r2, r3 = mma_f16(output[u, 0], output[u, 1], output[u, 2], output[u, 3],
                                            fp8_half_pair(a0 >> 16), fp8_half_pair(a1 >> 16), fp8_half_pair(a2 >> 16), fp8_half_pair(a3 >> 16), pb[step, 2], pb[step, 3])
                    output[u, 0], output[u, 1], output[u, 2], output[u, 3] = r0, r1, r2, r3
            for u in cutlass.range_constexpr(2):
                col = 16 * (4 * warp + 2 * pair_block + u) + 2 * g
                if c < 2:
                    store_pair(partials.iterator + part_base + 2 * c * 512 + col, published_word(f32_bits(output[u, 0])), published_word(f32_bits(output[u, 2])))
                    store_pair(partials.iterator + part_base + (2 * c + 1) * 512 + col, published_word(f32_bits(output[u, 1])), published_word(f32_bits(output[u, 3])))
        if tid < 4:
            maximum, total = neg_inf, cutlass.Float32(0)
            for w in cutlass.range_constexpr(8):
                maximum = max_num(maximum, max_view[w * 8 + tid])
                total += sum_view[w * 8 + tid]
            store_pair(stats.iterator + ((token * 16 + block) * 4 + tid) * 2, published_word(f32_bits(maximum)), published_word(f32_bits(total)))


@cute.jit
def merge_stage(unit: cutlass.Int32, partials: cute.Tensor, stats: cute.Tensor,
                  o: cute.Tensor, replicas: cute.Tensor, red: cute.Tensor,
                  clear_first: cutlass.Boolean, passes=None):
    if cutlass.const_expr(passes is None):
        merge_pass(unit, partials, stats, o, replicas, red, clear_first)
    else:
        for p in cutlass.range(passes, unroll=1):
            merge_pass(unit, partials, stats, o, replicas, red, clear_first, (p + 1) < passes)


@cute.jit
def merge_pass(unit: cutlass.Int32, partials: cute.Tensor, stats: cute.Tensor,
               o: cute.Tensor, replicas: cute.Tensor, red: cute.Tensor,
               clear_first: cutlass.Boolean, force=None):
    tid, _, _ = cute.arch.thread_idx()
    token, head = unit // 4, unit % 4
    words = cute.make_rmem_tensor((16, 2), cutlass.Uint32)
    has_stat = tid < 32
    stat_offset = ((token * 16 + tid // 2) * 4 + head) * 2 + (tid & 1)
    stat_word = cutlass.Uint32(0)
    part = partials.iterator + (token * 64 + head) * 512 + 2 * tid
    stat = stats.iterator + stat_offset
    if has_stat:
        stat_word = volatile_word(stat)
    for block in cutlass.range_constexpr(16):
        low, high = volatile_pair(part + block * 2048)
        words[block, 0], words[block, 1] = low, high
    waiting = cutlass.Boolean(True)
    while waiting:
        full = stat_word != cutlass.Uint32(0x80000000)
        for block in cutlass.range_constexpr(16):
            full = full & (words[block, 0] != cutlass.Uint32(0x80000000)) & (words[block, 1] != cutlass.Uint32(0x80000000))
        if cutlass.const_expr(force is not None):
            full = full | force
        if cute.arch.vote_all_sync(full):
            waiting = cutlass.Boolean(False)
        else:
            if stat_word == cutlass.Uint32(0x80000000):
                stat_word = volatile_word(stat)
            for block in cutlass.range_constexpr(16):
                if (words[block, 0] == cutlass.Uint32(0x80000000)) | (words[block, 1] == cutlass.Uint32(0x80000000)):
                    low, high = volatile_pair(part + block * 2048)
                    words[block, 0], words[block, 1] = low, high
    if has_stat:
        red[tid] = bits_f32(stat_word)
    empty = cutlass.Uint32(0x80000000)
    real = cutlass.Boolean(True)
    if cutlass.const_expr(force is not None):
        if force:
            real = cutlass.Boolean(False)
    if clear_first & real:
        if has_stat:
            store_word(stat, empty)
        for block in cutlass.range_constexpr(16):
            store_pair(part + block * 2048, empty, empty)
    cute.arch.sync_threads()
    if tid < 16:
        neg_inf = bits_f32(cutlass.Uint32(0xFF800000))
        merged = neg_inf
        for block in cutlass.range_constexpr(16):
            merged = max_num(merged, red[2 * block])
        mb = red[2 * tid]
        alpha = cutlass.Float32(0)
        if mb != neg_inf:
            alpha = cute.math.exp2(mb - merged)
        red[32 + tid] = alpha
        total = alpha * red[2 * tid + 1]
        for step in cutlass.range_constexpr(4):
            total += cute.arch.shuffle_sync_bfly(total, 8 >> step, mask=cutlass.Uint32(0xFFFF))
        if tid == 0:
            reciprocal = cutlass.Float32(0)
            if total > 0.0:
                reciprocal = cute.math.div(cutlass.Float32(1), total * 448.0,
                                           rounding=cute.math.RoundingMode.NEAREST_EVEN)
            red[48] = reciprocal
    cute.arch.sync_threads()
    acc0, acc1 = cutlass.Float32(0), cutlass.Float32(0)
    for block in cutlass.range_constexpr(16):
        alpha = red[32 + block]
        acc0 = fma_rn(alpha, bits_f32(words[block, 0]), acc0)
        acc1 = fma_rn(alpha, bits_f32(words[block, 1]), acc1)
    packet = published_word(bf16_pair(acc0 * red[48], acc1 * red[48]))
    offset = (token * 4 + head) * 256 + tid
    if real:
        store_word(o.iterator + offset, packet)
        replica_base = replicas.iterator + offset
        for replica in cutlass.range_constexpr(3):
            store_word(replica_base + replica * 8192, packet)
    if (not clear_first) & real:
        if has_stat:
            store_word(stat, empty)
        for block in cutlass.range_constexpr(16):
            store_pair(part + block * 2048, empty, empty)


# Imported last: the role modules import each other, and their names are used only when a kernel is traced.
from .gpu_publication import readonly_vector, cached_vector, store_vector, lds128, store_word, store_pair
from .common import (
    e4m3x2_to_f16x2_ir,
    low_16,
    as_ptr,
    word_vector,
    mul_rn,
    hop1_word_offset,
    store_empty,
    evict_first,
    bits_f32,
    f32_bits,
    max_num,
    fma_rn,
    bulk_copy,
    proxy_fence,
)
from .moe import volatile_word, warm_moe
from .plain import operand_words
from .indexer import volatile_pair
