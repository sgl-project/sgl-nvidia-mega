# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
import cutlass
from cutlass._mlir.dialects import llvm, nvvm, vector
from cutlass._mlir import ir
import cutlass.cute as cute
from cutlass.cutlass_dsl import T, dsl_user_op
from cutlass._mlir.dialects import cute as cute_ir
from cutlass._mlir.dialects import cute_nvgpu as nvgpu_ir




@dsl_user_op
def volatile_pair(pointer: cute.Pointer, *, loc=None, ip=None):
    v = llvm.load(ir.VectorType.get([2], T.i32()), as_ptr(1, pointer, loc, ip), volatile_=True, alignment=8, loc=loc,
                  ip=ip)
    return tuple(cutlass.Uint32(vector.extract(v, [], [i], loc=loc, ip=ip)) for i in range(2))


@cute.jit
def round_bf16(value: cutlass.Float32):
    return bf16_from_word(bf16_pair(value, value), False)


@cute.jit
def ue8m0_scale(amax: cutlass.Float32):
    minimum = bits_f32(cutlass.Uint32(0x2B8CBCCC))
    factor = bits_f32(cutlass.Uint32(0x3B124925))
    ratio = mul_rn(max_num(amax, minimum), factor)
    bits = f32_bits(ratio)
    exponent = bits & cutlass.Uint32(0x7F800000)
    if (bits & cutlass.Uint32(0x007FFFFF)) != cutlass.Uint32(0):
        exponent += cutlass.Uint32(0x00800000)
    return bits_f32(exponent)


@cute.jit
def indexer_key_stage(
    kw: cute.Tensor, weight: cute.Tensor, bias: cute.Tensor, rope: cute.Tensor,
    length: cute.Tensor, blocks: cute.Tensor, pool: cute.Tensor,
    arena_key: cute.Tensor, arena_scale: cute.Tensor,
):
    tid, _, _ = cute.arch.thread_idx()
    token = tid // 32
    lane = tid % 32
    pos = length[0] - 8 + token
    block = blocks[pos >> 6]
    w = cute.make_rmem_tensor((4,), cutlass.Float32)
    b = cute.make_rmem_tensor((4,), cutlass.Float32)
    cs = cute.make_rmem_tensor((2,), cutlass.Float32)
    sn = cute.make_rmem_tensor((2,), cutlass.Float32)
    for j in cutlass.range_constexpr(4):
        w[j] = weight[4 * lane + j]
        b[j] = bias[4 * lane + j]
    if lane < 16:
        for pair in cutlass.range_constexpr(2):
            cs[pair] = rope[pos, 2 * lane + pair]
            sn[pair] = rope[pos, 32 + 2 * lane + pair]
    words = cute.make_rmem_tensor((2,), cutlass.Uint32)
    v0, v1 = volatile_pair(kw.iterator + token * 160 + 4 * lane)
    words[0] = v0
    words[1] = v1
    waiting = cutlass.Boolean(True)
    while waiting:
        full = (words[0] != cutlass.Uint32(0x80000000)) & (words[1] != cutlass.Uint32(0x80000000))
        if cute.arch.vote_all_sync(full):
            waiting = cutlass.Boolean(False)
        elif not full:
            v0, v1 = volatile_pair(kw.iterator + token * 160 + 4 * lane)
            words[0] = v0
            words[1] = v1
    x = cute.make_rmem_tensor((4,), cutlass.Float32)
    kc = cute.make_rmem_tensor((4,), cutlass.Float32)
    k = cute.make_rmem_tensor((4,), cutlass.Float32)
    for j in cutlass.range_constexpr(4):
        x[j] = bf16_from_word(words[j // 2], j % 2 == 1)
    total = (x[0] + x[1]) + (x[2] + x[3])
    for step in cutlass.range_constexpr(5):
        total += cute.arch.shuffle_sync_bfly(total, 16 >> step)
    mean = total / 128.0
    variance = cutlass.Float32(0.0)
    for j in cutlass.range_constexpr(4):
        kc[j] = x[j] - mean
        variance = fma_rn(kc[j], kc[j], variance)
    for step in cutlass.range_constexpr(5):
        variance += cute.arch.shuffle_sync_bfly(variance, 16 >> step)
    rstd = cute.math.rsqrt(variance / 128.0 + 1.0e-6, fastmath=True)
    for j in cutlass.range_constexpr(4):
        k[j] = round_bf16(fma_rn(mul_rn(kc[j], rstd), w[j], b[j]))
    if lane < 16:
        for pair in cutlass.range_constexpr(2):
            left = k[2 * pair]
            right = k[2 * pair + 1]
            r0 = round_bf16(fma_rn(left, cs[pair], mul_rn(-right, sn[pair])))
            r1 = round_bf16(fma_rn(right, cs[pair], mul_rn(left, sn[pair])))
            k[2 * pair] = r0
            k[2 * pair + 1] = r1
    a0 = bits_f32(f32_bits(k[0]) & cutlass.Uint32(0x7FFFFFFF))
    a1 = bits_f32(f32_bits(k[1]) & cutlass.Uint32(0x7FFFFFFF))
    a2 = bits_f32(f32_bits(k[2]) & cutlass.Uint32(0x7FFFFFFF))
    a3 = bits_f32(f32_bits(k[3]) & cutlass.Uint32(0x7FFFFFFF))
    amax = max_num(max_num(a0, a1), max_num(a2, a3))
    for step in cutlass.range_constexpr(5):
        amax = max_num(amax, cute.arch.shuffle_sync_bfly(amax, 16 >> step))
    scale = ue8m0_scale(amax)
    inverse = cute.arch.rcp_approx(scale)
    q0 = mul_rn(k[0], inverse)
    q1 = mul_rn(k[1], inverse)
    q2 = mul_rn(k[2], inverse)
    q3 = mul_rn(k[3], inverse)
    word = e4m3_pair(q0, q1) | (e4m3_pair(q2, q3) << 16)
    pool[block, (pos & 63) * 32 + lane] = word
    arena_key[token, lane] = published_word(word)
    if lane == 0:
        scale_bits = f32_bits(scale)
        pool[block, 2048 + (pos & 63)] = scale_bits
        arena_scale[token] = published_word(scale_bits)


@cute.jit
def indexer_query_stage(unit: cutlass.Int32, y: cute.Tensor, kw: cute.Tensor,
                          region: cute.Tensor, red: cute.Tensor, scalars: cute.Tensor,
                          bar: cute.Pointer, rope: cute.Tensor, length: cute.Tensor,
                          query_bf16: cute.Tensor, head_weights: cute.Tensor, stock_idxq: cutlass.Constexpr):
    tid, _, _ = cute.arch.thread_idx()
    warp, lane = tid // 32, tid & 31
    if length[0] <= 2048:
        cute.arch.mbarrier_wait(bar, 0)
    else:
        token, pair = tid // 16, tid % 16
        head, quarter = unit // 4, unit % 4
        dim0 = 32 * quarter + 2 * pair
        cosv, sinv = cutlass.Float32(1), cutlass.Float32(0)
        if (tid < 128) & (dim0 < 64):
            position = length[0] - 8 + token
            cosv, sinv = rope[position, dim0 // 2], rope[position, 32 + dim0 // 2]
        b = cute.make_rmem_tensor((4, 8), cutlass.Uint32)
        operand_words(b, y, 1312, 4 * warp, 4, False, True)
        weight_offset = token * 80 + 64 + (head >> 1)
        raw_weight = cutlass.Uint32(0x80000000)
        if (tid < 128) & (quarter == 0) & (pair == 0):
            raw_weight = volatile_word(kw.iterator + weight_offset)
        ssq = cutlass.Float32(0)
        for i in cutlass.range_constexpr(4):
            for j in cutlass.range_constexpr(8):
                lo, hi = bf16_from_word(b[i, j], False), bf16_from_word(b[i, j], True)
                ssq = ssq + lo * lo + hi * hi
        ssq += cute.arch.shuffle_sync_bfly(ssq, 1)
        ssq += cute.arch.shuffle_sync_bfly(ssq, 2)
        if (lane & 3) == 0:
            scalars[warp * 8 + (lane >> 2)] = ssq
        tc_after()
        cute.arch.sync_threads()
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
        reduce_tiles(acc, red, 2, 3072)
        if tid < 8:
            total = cutlass.Float32(0)
            for w in cutlass.range_constexpr(8):
                total += scalars[w * 8 + tid]
            scalars[64 + tid] = cute.math.rsqrt(total / 2048.0 + 1.0e-5, fastmath=True)
        cute.arch.sync_threads()
        if tid < 128:
            rstd = scalars[64 + token]
            x0 = round_bf16(red[3072 + 2 * pair * 8 + token] * rstd)
            x1 = round_bf16(red[3072 + (2 * pair + 1) * 8 + token] * rstd)
            if (dim0 < 64) & (not cutlass.const_expr(stock_idxq)):
                y0 = round_bf16(fma_rn(x0, cosv, mul_rn(-x1, sinv)))
                y1 = round_bf16(fma_rn(x1, cosv, mul_rn(x0, sinv)))
                x0, x1 = y0, y1
            offset = ((token * 32 + head) * 128 + dim0) // 2
            store_word(query_bf16.iterator + offset, published_word(bf16_pair(x0, x1)))
            if (quarter == 0) & (pair == 0):
                while raw_weight == cutlass.Uint32(0x80000000):
                    raw_weight = volatile_word(kw.iterator + weight_offset)
                value = bf16_from_word(raw_weight, False)
                if (head & 1) != 0:
                    value = bf16_from_word(raw_weight, True)
                store_word(head_weights.iterator + token * 32 + head, published_word(f32_bits(value)))


@cute.jit
def selection_pages(cta: cutlass.Int32, length: cutlass.Int32):
    pages = (length + 63) // 64
    leading = pages
    if leading > 588:
        leading = cutlass.Int32(588)
    rounds = pages - 588
    extra = cutlass.Int32(0)
    if cta < rounds:
        extra = (rounds - 1 - cta) // 212 + 1
    count = extra
    if cta >= 128:
        free = cta - 128
        if free < leading:
            count += (leading - 1 - free) // 84 + 1
    return count


@cute.jit
def selection_query_stage(rank: cutlass.Int32, storage: cute.Tensor,
                             query_bf16: cute.Tensor, head_weights: cute.Tensor, stock_idxq: cutlass.Constexpr,
                             rope=None, length=None):
    tid, _, _ = cute.arch.thread_idx()
    row, head = rank // 2, tid >> 3
    chunk = tid & 7
    offset = (row * 32 + head) * 64 + chunk * 8
    rc = cute.make_rmem_tensor((8,), cutlass.Float32)
    rs = cute.make_rmem_tensor((8,), cutlass.Float32)
    if cutlass.const_expr(stock_idxq):
        # Loaded before the poll so the latency hides under it.
        position = length[0] - 8 + row
        for j in cutlass.range_constexpr(8):
            rc[j], rs[j] = cutlass.Float32(1), cutlass.Float32(0)
            if chunk < 4:
                rc[j] = rope[position, 8 * chunk + j]
                rs[j] = rope[position, 32 + 8 * chunk + j]
    first = cute.make_rmem_tensor((4,), cutlass.Uint32)
    second = cute.make_rmem_tensor((4,), cutlass.Uint32)
    v0, v1, v2, v3 = volatile_vector(query_bf16.iterator + offset)
    first[0], first[1], first[2], first[3] = v0, v1, v2, v3
    v0, v1, v2, v3 = volatile_vector(query_bf16.iterator + offset + 4)
    second[0], second[1], second[2], second[3] = v0, v1, v2, v3
    weight = cutlass.Uint32(0)
    if chunk == 0:
        weight = volatile_word(head_weights.iterator + row * 32 + head)
    waiting = cutlass.Boolean(True)
    while waiting:
        complete0, complete1 = cutlass.Boolean(True), cutlass.Boolean(True)
        for j in cutlass.range_constexpr(4):
            complete0 = complete0 & (first[j] != cutlass.Uint32(0x80000000))
            complete1 = complete1 & (second[j] != cutlass.Uint32(0x80000000))
        if cute.arch.vote_all_sync(complete0 & complete1 & (weight != cutlass.Uint32(0x80000000))):
            waiting = cutlass.Boolean(False)
        else:
            if not complete0:
                v0, v1, v2, v3 = volatile_vector(query_bf16.iterator + offset)
                first[0], first[1], first[2], first[3] = v0, v1, v2, v3
            if not complete1:
                v0, v1, v2, v3 = volatile_vector(query_bf16.iterator + offset + 4)
                second[0], second[1], second[2], second[3] = v0, v1, v2, v3
            if (chunk == 0) & (weight == cutlass.Uint32(0x80000000)):
                weight = volatile_word(head_weights.iterator + row * 32 + head)
    words = cute.make_rmem_tensor((8,), cutlass.Uint32)
    for j in cutlass.range_constexpr(4):
        words[j], words[j + 4] = first[j], second[j]
    vlo = cute.make_rmem_tensor((8,), cutlass.Float32)
    vhi = cute.make_rmem_tensor((8,), cutlass.Float32)
    for j in cutlass.range_constexpr(8):
        vlo[j] = bf16_from_word(words[j], False)
        vhi[j] = bf16_from_word(words[j], True)
        if cutlass.const_expr(stock_idxq):
            x0, x1 = vlo[j], vhi[j]
            vlo[j] = fma_rn(x0, rc[j], mul_rn(-x1, rs[j]))
            vhi[j] = fma_rn(x0, rs[j], mul_rn(x1, rc[j]))
    maximum = cutlass.Float32(0)
    for j in cutlass.range_constexpr(8):
        lo = bits_f32(f32_bits(vlo[j]) & cutlass.Uint32(0x7fffffff))
        hi = bits_f32(f32_bits(vhi[j]) & cutlass.Uint32(0x7fffffff))
        maximum = max_num(maximum, max_num(lo, hi))
    for step in cutlass.range_constexpr(3):
        maximum = max_num(maximum, cute.arch.shuffle_sync_bfly(maximum, 1 << step))
    scale = cutlass.Float32(0)
    inverse = cutlass.Float32(0)
    if cutlass.const_expr(stock_idxq):
        scale = div_rn_ftz(max_num(maximum, cutlass.Float32(1.0e-4)), cutlass.Float32(448.0))
        inverse = div_rn_ftz(cutlass.Float32(1.0), scale)
    else:
        scale = ue8m0_scale(maximum)
        inverse = cute.arch.rcp_approx(scale)
    packed = cute.make_rmem_tensor((4,), cutlass.Uint32)
    for j in cutlass.range_constexpr(4):
        low = e4m3_pair(mul_rn(vlo[2 * j], inverse), mul_rn(vhi[2 * j], inverse))
        high = e4m3_pair(mul_rn(vlo[2 * j + 1], inverse), mul_rn(vhi[2 * j + 1], inverse))
        packed[j] = low | (high << 16)
    base = storage.iterator.toint().to(cutlass.Uint32)
    shared_store4(base + (131072 + head * 128 + 16 * (chunk ^ (head & 7))).to(cutlass.Uint32),
                      packed[0], packed[1], packed[2], packed[3])
    if chunk == 0:
        weights = cute.make_tensor(cute.recast_ptr(storage.iterator + 167936 // 4, dtype=cutlass.Float32),
                                      cute.make_layout((32,)))
        if cutlass.const_expr(stock_idxq):
            weights[head] = mul_rn(mul_rn(bits_f32(weight), cutlass.Float32(0.015625)), scale)
        else:
            weights[head] = mul_rn(mul_rn(bits_f32(weight), scale), cutlass.Float32(0.015625))
    cute.arch.sync_threads()


@dsl_user_op
def selection_tma(atom: cute.CopyAtom, destination: cutlass.Uint32,
                  y: cutlass.Int32, bar: cute.Pointer, policy: cutlass.Uint64,
                  *, loc=None, ip=None):
    executable = nvgpu_ir.atom_make_exec_tma(atom._trait.value, loc=loc, ip=ip)
    ptr_type = cute_ir.PtrType.get(nvgpu_ir.TmaDescriptorTiledType.get(), cutlass.AddressSpace.generic, 64)
    pointer = nvgpu_ir.get_tma_desc_addr(ptr_type, executable, loc=loc, ip=ip)
    descriptor = as_ptr(0, pointer.toint().ir_value(loc=loc, ip=ip), loc, ip)
    nvvm.cp_async_bulk_tensor_shared_cluster_global(as_ptr(7, destination, loc, ip), descriptor,
                                                    [cutlass.Int32(0).ir_value(loc=loc, ip=ip),
                                                     y.ir_value(loc=loc, ip=ip)],
                                                    as_ptr(3, bar, loc, ip), [],
                                                    l2_cache_hint=policy.ir_value(loc=loc, ip=ip), loc=loc, ip=ip)


@dsl_user_op
def selection_scale_copy(destination: cutlass.Uint32, source: cute.Pointer,
                           bar: cute.Pointer, policy: cutlass.Uint64, *, loc=None, ip=None):
    nvvm.cp_async_bulk_shared_cluster_global(as_ptr(7, destination, loc, ip), as_ptr(1, source, loc, ip),
                                             as_ptr(3, bar, loc, ip), cutlass.Uint32(256).ir_value(loc=loc, ip=ip),
                                             l2_cache_hint=policy.ir_value(loc=loc, ip=ip), loc=loc, ip=ip)


@dsl_user_op
def count_bin(pointer: cute.Pointer, *, loc=None, ip=None):
    llvm.atomicrmw(llvm.AtomicBinOp.add, as_ptr(3, pointer, loc, ip), cutlass.Uint32(1).ir_value(loc=loc, ip=ip),
                   llvm.AtomicOrdering.monotonic, syncscope='block', loc=loc, ip=ip)

@dsl_user_op
def selection_mma(d: cutlass.Uint32, key: cutlass.Uint32, query: cutlass.Uint32,
                  accumulate: cutlass.Uint32, *, loc=None, ip=None):
    common = cutlass.Uint64((1 << 16) | (64 << 32) | (1 << 46) | (2 << 61))
    da = common | ((key & cutlass.Uint32(0x3ffff)) >> 4).to(cutlass.Uint64)
    db = common | ((query & cutlass.Uint32(0x3ffff)) >> 4).to(cutlass.Uint64)
    idesc = cutlass.Uint32((1 << 4) | (4 << 17) | (8 << 24))
    nvvm.tcgen05_mma(nvvm.Tcgen05MMAKind.F8F6F4, nvvm.CTAGroupKind.CTA_1, as_ptr(6, d, loc, ip),
                     da.ir_value(loc=loc, ip=ip), db.ir_value(loc=loc, ip=ip), idesc.ir_value(loc=loc, ip=ip),
                     (accumulate != cutlass.Uint32(0)).ir_value(loc=loc, ip=ip), loc=loc, ip=ip)

@dsl_user_op
def selection_load32(address: cutlass.Uint32, *, loc=None, ip=None):
    v = nvvm.tcgen05_ld(ir.VectorType.get([32], T.i32()), nvvm.Tcgen05LdStShape.SHAPE_32X32B,
                        as_ptr(6, address, loc, ip), loc=loc, ip=ip)
    nvvm.tcgen05_wait(nvvm.Tcgen05WaitKind.LOAD, loc=loc, ip=ip)
    return tuple(cutlass.Uint32(vector.extract(v, [], [i], loc=loc, ip=ip)) for i in range(32))

@cute.jit
def selection_logit(address: cutlass.Uint32, weights: cute.Tensor,
                    scale: cutlass.Float32, valid: cutlass.Boolean):
    words = selection_load32(address)
    accum = cute.make_rmem_tensor((4,), cutlass.Float32)
    for chain in cutlass.range_constexpr(4):
        accum[chain] = cutlass.Float32(0)
    for h4 in cutlass.range_constexpr(8):
        for chain in cutlass.range_constexpr(4):
            column = 4 * h4 + chain
            accum[chain] = fma_rn(weights[column], max_num(bits_f32(words[column]), cutlass.Float32(0)), accum[chain])
    logit = bits_f32(cutlass.Uint32(0xff800000))
    if valid:
        logit = ((accum[0] + accum[1]) + (accum[2] + accum[3])) * scale
    return logit


@cute.jit
def page_of(cta: cutlass.Int32, local: cutlass.Int32):
    page = 588 + cta + 212 * local
    if cta >= 128:
        if local < 7:
            page = cta - 128 + 84 * local
        else:
            page = 588 + cta + 212 * (local - 7)
    return page


@cute.jit
def page_slot(cta: cutlass.Int32, page: cutlass.Int32):
    local = cutlass.Int32(-1)
    if page < 588:
        if (cta >= 128) & ((page % 84) == cta - 128):
            local = page // 84
    elif ((page - 588) % 212) == cta:
        local = (page - 588) // 212
        if cta >= 128:
            local += 7
    return local


@cute.jit
def rs_bin(logit: cutlass.Float32):
    key = f32_bits(logit)
    if (key >> 31) != cutlass.Uint32(0):
        key = key ^ cutlass.Uint32(0xffffffff)
    else:
        key = key ^ cutlass.Uint32(0x80000000)
    positive = key >= cutlass.Uint32(0x80000000)
    origin, base = cutlass.Int32(0x3b80), cutlass.Int32(1)
    if positive:
        origin, base = cutlass.Int32(0xb900), cutlass.Int32(2949)
    delta = (key >> 16).to(cutlass.Int32) - origin
    if delta < -1:
        delta = cutlass.Int32(-1)
    if delta > 2944:
        delta = cutlass.Int32(2944)
    result = base + delta
    if key == cutlass.Uint32(0x80000000):
        result = cutlass.Int32(2947)
    if key == cutlass.Uint32(0x7fffffff):
        result = cutlass.Int32(2946)
    return result


@cute.jit
def issue_pages(cta: cutlass.Int32, first: cutlass.Int32, count: cutlass.Int32,
                    slot: cutlass.Int32, table: cute.Tensor, pool: cute.Tensor,
                    stride: cutlass.Int64, pages: cutlass.Int64,
                    atom: cute.CopyAtom, storage: cute.Tensor, bar: cute.Pointer,
                    from_shared: cutlass.Boolean):
    tid, _, _ = cute.arch.thread_idx()
    if (tid < 32) & (count > 0):
        base = storage.iterator.toint().to(cutlass.Uint32)
        indices = cute.make_tensor(cute.recast_ptr(storage.iterator + 135168 // 4, dtype=cutlass.Int32),
                                      cute.make_layout((124,)))
        block = cutlass.Int32(0)
        if tid < count:
            if from_shared:
                block = indices[first + tid]
            else:
                logical = page_of(cta, first + tid)
                block = table[logical]
        if tid == 0:
            proxy_fence()
            cute.arch.mbarrier_arrive_and_expect_tx(bar, count * 8448)
        cute.arch.sync_warp()
        if tid < count:
            y = block.to(cutlass.Int64) * (stride // 128)
            policy = evict_first()
            selection_tma(atom, base + ((slot + tid) * 8192).to(cutlass.Uint32),
                              y.to(cutlass.Int32), bar, policy)
            selection_scale_copy(base + (163840 + (slot + tid) * 256).to(cutlass.Uint32),
                          pool.iterator + block.to(cutlass.Int64) * stride + 8192,
                          bar, policy)


@cute.jit
def pages_in(cta: cutlass.Int32, first: cutlass.Int32, count: cutlass.Int32,
                 slot: cutlass.Int32, phase: cutlass.Int32, length: cutlass.Int32,
                 storage: cute.Tensor, bar: cute.Pointer,
                 fresh_keys: cute.Tensor, fresh_scales: cute.Tensor):
    tid, _, _ = cute.arch.thread_idx()
    base = storage.iterator.toint().to(cutlass.Uint32)
    scales = cute.make_tensor(cute.recast_ptr(storage.iterator + 163840 // 4, dtype=cutlass.Float32),
                                 cute.make_layout((1024,)))
    if count > 0:
        cute.arch.mbarrier_wait(bar, phase)
    if tid < 64:
        token, chunk = tid >> 3, tid & 7
        position = length - 8 + token
        local = page_slot(cta, position // 64) - first
        if (local >= 0) & (local < count):
            pointer = fresh_keys.iterator + token * 32 + 4 * chunk
            v0, v1, v2, v3 = volatile_vector(pointer)
            while (v0 == cutlass.Uint32(0x80000000)) | (v1 == cutlass.Uint32(0x80000000)) | (v2 == cutlass.Uint32(0x80000000)) | (v3 == cutlass.Uint32(0x80000000)):
                v0, v1, v2, v3 = volatile_vector(pointer)
            row = position & 63
            shared_store4(base + ((slot + local) * 8192 + row * 128 + 16 * (chunk ^ (row & 7))).to(cutlass.Uint32),
                              v0, v1, v2, v3)
            if chunk == 0:
                scale = volatile_word(fresh_scales.iterator + token)
                while scale == cutlass.Uint32(0x80000000):
                    scale = volatile_word(fresh_scales.iterator + token)
                scales[(slot + local) * 64 + row] = bits_f32(scale)
    proxy_fence()
    tc_before()
    cute.arch.sync_threads()
    tc_after()


@cute.jit
def batch_in(cta: cutlass.Int32, batch: cutlass.Int32, mine: cutlass.Int32,
                 length: cutlass.Int32, storage: cute.Tensor, bars: cute.Tensor,
                 tmem: cutlass.Uint32, fresh_keys: cute.Tensor, fresh_scales: cute.Tensor):
    tid, _, _ = cute.arch.thread_idx()
    count = mine - 8 * batch
    if count > 8:
        count = cutlass.Int32(8)
    side = batch & 1
    bar = bars.iterator + (28 + 3 * side)
    pages_in(cta, 8 * batch, count, 8 * side, (batch >> 1) & 1, length,
                 storage, bar, fresh_keys, fresh_scales)
    if tid // 32 == 1:
        tc_after()
        with cute.arch.elect_one():
            base = storage.iterator.toint().to(cutlass.Uint32)
            for tile in cutlass.range_constexpr(4):
                if tile < (count + 1) // 2:
                    for step in cutlass.range_constexpr(4):
                        selection_mma(tmem + cutlass.Uint32(32 * (4 * side + tile)),
                                          base + cutlass.Uint32(16384 * (4 * side + tile) + 32 * step),
                                          base + cutlass.Uint32(131072 + 32 * step), cutlass.Uint32(step > 0))
            if count > 0:
                commit_mma(bars.iterator + 29 + side)
        cute.arch.sync_warp()


@cute.jit
def selection_score_stage(cta: cutlass.Int32, rank: cutlass.Int32, length: cute.Tensor,
                              table: cute.Tensor, pool: cute.Tensor, stride: cutlass.Int64,
                              pages: cutlass.Int64, atom: cute.CopyAtom, storage: cute.Tensor,
                              bars: cute.Tensor, tmem: cutlass.Uint32, query: cute.Tensor,
                              weights: cute.Tensor, fresh_keys: cute.Tensor, fresh_scales: cute.Tensor,
                              stock_idxq: cutlass.Constexpr, rope=None):
    tid, _, _ = cute.arch.thread_idx()
    warp, lane = tid >> 5, tid & 31
    mine, L = selection_pages(cta, length[0]), length[0]
    histogram = cute.make_tensor(storage.iterator + 135680 // 4, cute.make_layout((6144,)))
    for i in cutlass.range_constexpr(24):
        histogram[tid + 256 * i] = cutlass.Uint32(0)
    count = mine
    if count > 8:
        count = cutlass.Int32(8)
    issue_pages(cta, 0, count, 0, table, pool, stride, pages, atom, storage, bars.iterator + 28, False)
    count2 = mine - 8
    if count2 < 0:
        count2 = cutlass.Int32(0)
    if count2 > 8:
        count2 = cutlass.Int32(8)
    issue_pages(cta, 8, count2, 8, table, pool, stride, pages, atom, storage, bars.iterator + 31, False)
    indices = cute.make_tensor(cute.recast_ptr(storage.iterator + 135168 // 4, dtype=cutlass.Int32),
                                  cute.make_layout((124,)))
    if 16 + tid < mine:
        logical = page_of(cta, 16 + tid)
        indices[16 + tid] = table[logical]
    selection_query_stage(rank, storage, query, weights, stock_idxq, rope, length)
    scales = cute.make_tensor(cute.recast_ptr(storage.iterator + 163840 // 4, dtype=cutlass.Float32),
                                 cute.make_layout((1024,)))
    head_weights = cute.make_tensor(cute.recast_ptr(storage.iterator + 167936 // 4, dtype=cutlass.Float32),
                                       cute.make_layout((32,)))
    logits = cute.make_tensor(cute.recast_ptr(storage.iterator + 168960 // 4, dtype=cutlass.Float32),
                                 cute.make_layout((7936,)))
    batches = (mine + 7) // 8
    if batches > 0:
        batch_in(cta, 0, mine, L, storage, bars, tmem, fresh_keys, fresh_scales)
    for batch in range(batches):
        first, side = 8 * batch, batch & 1
        count = mine - first
        if count > 8:
            count = cutlass.Int32(8)
        tiles = (count + 1) // 2
        key = 32 * (warp & 3) + lane
        held_scales = cute.make_rmem_tensor((2,), cutlass.Float32)
        for j in cutlass.range_constexpr(2):
            tile = (warp >> 2) + 2 * j
            local_page = first + 2 * tile + (key >> 6)
            held_scales[j] = cutlass.Float32(0)
            if (tile < tiles) & (local_page < mine):
                held_scales[j] = scales[(8 * side + local_page - first) * 64 + (key & 63)]
        cute.arch.mbarrier_wait(bars.iterator + 29 + side, (batch >> 1) & 1)
        tc_after()
        if batch + 2 < batches:
            cute.arch.sync_threads()
            future = first + 16
            future_count = mine - future
            if future_count > 8:
                future_count = cutlass.Int32(8)
            load_bar = bars.iterator + (28 + 3 * side)
            issue_pages(cta, future, future_count, 8 * side, table, pool, stride, pages,
                            atom, storage, load_bar, True)
        if batch + 1 < batches:
            batch_in(cta, batch + 1, mine, L, storage, bars, tmem, fresh_keys, fresh_scales)
        for j in cutlass.range_constexpr(2):
            tile = (warp >> 2) + 2 * j
            if tile < tiles:
                local_page = first + 2 * tile + (key >> 6)
                position = page_of(cta, local_page) * 64 + (key & 63)
                valid = (local_page < mine) & (position <= L - 8 + rank // 2)
                value = selection_logit(tmem + (32 * (4 * side + tile)).to(cutlass.Uint32)
                                           + ((32 * (warp & 3)).to(cutlass.Uint32) << 16),
                                           head_weights, held_scales[j], valid)
                logits[64 * first + 128 * tile + key] = value
                if value != bits_f32(cutlass.Uint32(0xff800000)):
                    count_bin(histogram.iterator + rs_bin(value))
    tc_before()
    cute.arch.sync_threads()


@dsl_user_op
def relaxed_add(pointer: cute.Pointer, value: cutlass.Uint32, *, loc=None, ip=None):
    cute.arch.red(pointer, value, op="add", dtype="u32", sem="relaxed", scope="gpu", loc=loc, ip=ip)


@dsl_user_op
def shared_add(pointer: cute.Pointer, value: cutlass.Uint32, *, loc=None, ip=None):
    return cutlass.Uint32(llvm.atomicrmw(llvm.AtomicBinOp.add, as_ptr(3, pointer, loc, ip),
                                         value.ir_value(loc=loc, ip=ip), llvm.AtomicOrdering.monotonic,
                                         syncscope='block', loc=loc, ip=ip))


@dsl_user_op
def redux_sum(value: cutlass.Uint32, *, loc=None, ip=None):
    return cutlass.Uint32(cute.arch.warp_redux_sync(value, "add", loc=loc, ip=ip))


@dsl_user_op
def global_add64(pointer: cute.Pointer, value: cutlass.Uint64, *, loc=None, ip=None):
    return cutlass.Uint64(llvm.atomicrmw(llvm.AtomicBinOp.add, as_ptr(1, pointer, loc, ip),
                                         value.ir_value(loc=loc, ip=ip), llvm.AtomicOrdering.monotonic,
                                         syncscope='device', loc=loc, ip=ip))


@dsl_user_op
def store64(pointer: cute.Pointer, value: cutlass.Uint64, *, loc=None, ip=None):
    cute.arch.store(pointer, value, sem="relaxed", scope="gpu", loc=loc, ip=ip)


@cute.jit
def order_key(value: cutlass.Float32):
    key = f32_bits(value)
    if key == cutlass.Uint32(0xff800000):
        key = cutlass.Uint32(0)
    elif (key >> 31) != cutlass.Uint32(0):
        key ^= cutlass.Uint32(0xffffffff)
    else:
        key ^= cutlass.Uint32(0x80000000)
    return key


@cute.jit
def bin_of_key(key: cutlass.Uint32):
    origin, base = cutlass.Int32(0x3b80), cutlass.Int32(1)
    if key >= cutlass.Uint32(0x80000000):
        origin, base = cutlass.Int32(0xb900), cutlass.Int32(2949)
    delta = (key >> 16).to(cutlass.Int32) - origin
    if delta < -1:
        delta = cutlass.Int32(-1)
    if delta > 2944:
        delta = cutlass.Int32(2944)
    result = base + delta
    if key == cutlass.Uint32(0x7fffffff):
        result = cutlass.Int32(2946)
    if key == cutlass.Uint32(0x80000000):
        result = cutlass.Int32(2947)
    return result


@cute.jit
def lowest_key(bin: cutlass.Uint32):
    lo = cutlass.Uint32(1)
    if bin > cutlass.Uint32(0):
        if bin <= cutlass.Uint32(2945):
            lo = (cutlass.Uint32(0x3b80) + bin - cutlass.Uint32(1)) << 16
        elif bin == cutlass.Uint32(2946):
            lo = cutlass.Uint32(0x7fffffff)
        elif bin == cutlass.Uint32(2947):
            lo = cutlass.Uint32(0x80000000)
        elif bin == cutlass.Uint32(2948):
            lo = cutlass.Uint32(0x80000001)
        else:
            lo = (cutlass.Uint32(0xb900) + bin - cutlass.Uint32(2949)) << 16
    return lo


@cute.jit
def rs_flush(cta: cutlass.Int32, storage: cute.Tensor, global_hist: cute.Tensor,
                 completion: cute.Tensor):
    tid, _, _ = cute.arch.thread_idx()
    warp, lane = tid >> 5, tid & 31
    hist = cute.make_tensor(storage.iterator + 135680 // 4, cute.make_layout((6144,)))
    scratch = cute.make_tensor(storage.iterator + 200704 // 4, cute.make_layout((16,)))
    total = cutlass.Uint32(0)
    for j in range(6):
        b = 24 * tid + 4 * j
        total += (hist[b].to(cutlass.Uint32) + hist[b + 1].to(cutlass.Uint32)) + (hist[b + 2].to(cutlass.Uint32) + hist[b + 3].to(cutlass.Uint32))
    inclusive = total
    for step in cutlass.range_constexpr(5):
        offset = 1 << step
        other = cute.arch.shuffle_sync_down(inclusive, offset)
        if lane + offset < 32:
            inclusive += other
    if lane == 0:
        scratch[warp] = inclusive
    cute.arch.sync_threads()
    above = inclusive - total
    for w in range(warp + 1, 8):
        above += scratch[w].to(cutlass.Uint32)
    acc, added, cut = above, cutlass.Uint32(0), cutlass.Uint32(0)
    for rev in range(6):
        j = 5 - rev
        for ii in cutlass.range_constexpr(4):
            i = 3 - ii
            count = hist[24 * tid + 4 * j + i].to(cutlass.Uint32)
            if (count != cutlass.Uint32(0)) & (acc < cutlass.Uint32(128)):
                relaxed_add(global_hist.iterator + 4 * (tid + 256 * j) + i, count)
                added += count
                if acc + count >= cutlass.Uint32(128):
                    cut = (24 * tid + 4 * j + i).to(cutlass.Uint32)
            acc += count
    added, cut = redux_sum(added), redux_max(cut)
    if lane == 0:
        store_word(completion.iterator + 8 * cta + warp,
                       (added + cutlass.Uint32(1)) | (cut << 16))
        scratch[8 + warp] = cut
    acc = above
    for rev in range(6):
        j = 5 - rev
        for ii in cutlass.range_constexpr(4):
            b = 24 * tid + 4 * j + 3 - ii
            count = hist[b].to(cutlass.Uint32)
            hist[b] = acc
            acc += count
    cute.arch.sync_threads()
    local_cut = cutlass.Uint32(0)
    for w in range(8):
        value = scratch[8 + w].to(cutlass.Uint32)
        if value > local_cut:
            local_cut = value
    return local_cut


@cute.jit
def rs_place(cta: cutlass.Int32, storage: cute.Tensor, k: cutlass.Int32,
                 lo: cutlass.Uint32, less: cutlass.Boolean, global_hist=None):
    logits = cute.make_tensor(cute.recast_ptr(storage.iterator + 168960 // 4, dtype=cutlass.Float32),
                                 cute.make_layout((7936,)))
    key = order_key(logits[k])
    selected = key >= lo
    if less:
        selected = (key != cutlass.Uint32(0)) & (key < lo)
    if selected:
        bin = bin_of_key(key)
        if cutlass.const_expr(global_hist is not None):
            word = 4 * (bin // 24 + 256 * ((bin % 24) >> 2)) + (bin & 3)
            relaxed_add(global_hist.iterator + word, cutlass.Uint32(1))
        index = shared_add(storage.iterator + 135680 // 4 + bin, cutlass.Uint32(1))
        storage[index] = key
        storage[7936 + index] = (64 * page_of(cta, k >> 6) + (k & 63)).to(cutlass.Uint32)


@cute.jit
def rs_sort(cta: cutlass.Int32, mine: cutlass.Int32, storage: cute.Tensor,
                local_cut: cutlass.Uint32, less: cutlass.Boolean, global_hist=None):
    tid, _, _ = cute.arch.thread_idx()
    lo = lowest_key(local_cut)
    logits = cute.make_tensor(cute.recast_ptr(storage.iterator + 168960 // 4, dtype=cutlass.Float32),
                                 cute.make_layout((7936,)))
    count, record = cutlass.Uint32(0), cutlass.Uint64(0)
    for i in range((64 * mine + 255 - tid) // 256):
        key = order_key(logits[tid + 256 * i])
        selected = key >= lo
        if less:
            selected = (key != cutlass.Uint32(0)) & (key < lo)
        if selected:
            count += cutlass.Uint32(1)
            record = (record << 8) | i.to(cutlass.Uint64)
    if count <= cutlass.Uint32(8):
        for j in range(count.to(cutlass.Int32)):
            i = ((record >> (8 * j).to(cutlass.Uint32)) & cutlass.Uint64(255)).to(cutlass.Int32)
            rs_place(cta, storage, tid + 256 * i, lo, less, global_hist)
    else:
        for k in range(tid, 64 * mine, 256):
            rs_place(cta, storage, k, lo, less, global_hist)
    cute.arch.sync_threads()


@cute.jit
def rs_row_cut(global_hist: cute.Tensor, completion, length: cutlass.Int32,
                   valid_keys: cutlass.Uint32, need: cutlass.Uint32, storage: cute.Tensor):
    tid, _, _ = cute.arch.thread_idx()
    warp, lane = tid >> 5, tid & 31
    scratch = cute.make_tensor(storage.iterator + (200704 + 64) // 4, cute.make_layout((68,)))
    pw = cute.make_rmem_tensor((8,), cutlass.Uint32)
    values = cute.make_rmem_tensor((24,), cutlass.Uint32)
    result = cute.make_rmem_tensor((4,), cutlass.Uint32)
    mine = cutlass.Boolean(False)
    if cutlass.const_expr(completion is not None):
        mine = (tid < 212) & (selection_pages(tid, length) > 0)
    for i in cutlass.range_constexpr(8):
        pw[i] = cutlass.Uint32(1)
    if cutlass.const_expr(completion is not None):
        if mine:
            a, b, c, d = volatile_vector(completion.iterator + 8 * tid)
            pw[0], pw[1], pw[2], pw[3] = a, b, c, d
            a, b, c, d = volatile_vector(completion.iterator + 8 * tid + 4)
            pw[4], pw[5], pw[6], pw[7] = a, b, c, d
    for j in cutlass.range_constexpr(6):
        a, b, c, d = volatile_vector(global_hist.iterator + 4 * (tid + 256 * j))
        values[4 * j], values[4 * j + 1], values[4 * j + 2], values[4 * j + 3] = a, b, c, d
    for i in cutlass.range_constexpr(4):
        result[i] = cutlass.Uint32(0)
    parity, waiting = cutlass.Int32(0), cutlass.Boolean(True)
    while waiting:
        count = cutlass.Uint32(0)
        for i in cutlass.range_constexpr(24):
            count += values[i].to(cutlass.Uint32)
        inclusive = count
        for step in cutlass.range_constexpr(5):
            offset = 1 << step
            other = cute.arch.shuffle_sync_down(inclusive, offset)
            if lane + offset < 32:
                inclusive += other
        have, expected, local_max = cutlass.Boolean(True), cutlass.Uint32(0), cutlass.Uint32(0)
        for i in cutlass.range_constexpr(8):
            have = have & (pw[i] != cutlass.Uint32(0))
            expected += (pw[i].to(cutlass.Uint32) & cutlass.Uint32(65535)) - cutlass.Uint32(1)
            b = pw[i].to(cutlass.Uint32) >> 16
            if b > local_max:
                local_max = b
        missing = cutlass.Uint32(1)
        if have:
            missing = cutlass.Uint32(0)
        known, maximum = cutlass.Uint32(0), cutlass.Uint32(0)
        if have:
            known, maximum = expected, local_max
        missing, known, maximum = redux_sum(missing), redux_sum(known), redux_max(maximum)
        if lane == 0:
            base = 32 * parity + 4 * warp
            scratch[base], scratch[base + 1] = inclusive, missing
            scratch[base + 2], scratch[base + 3] = known, maximum
        cute.arch.sync_threads()
        higher, total, missing, expected, largest = (cutlass.Uint32(0), cutlass.Uint32(0),
                                                    cutlass.Uint32(0), cutlass.Uint32(0), cutlass.Uint32(0))
        for w in cutlass.range_constexpr(8):
            base = 32 * parity + 4 * w
            s = scratch[base].to(cutlass.Uint32)
            if w > warp:
                higher += s
            total += s
            missing += scratch[base + 1].to(cutlass.Uint32)
            expected += scratch[base + 2].to(cutlass.Uint32)
            b = scratch[base + 3].to(cutlass.Uint32)
            if b > largest:
                largest = b
        if cutlass.const_expr(completion is None):
            expected = valid_keys
        acc = higher + inclusive - count
        cut, above, in_bin = cutlass.Uint32(0xffffffff), cutlass.Uint32(0), cutlass.Uint32(0)
        for rev in cutlass.range_constexpr(24):
            i = 23 - rev
            value = values[i].to(cutlass.Uint32)
            if (acc < need) & (acc + value >= need):
                cut = (24 * tid + i).to(cutlass.Uint32)
                above, in_bin = acc, value
            acc += value
        if (missing == cutlass.Uint32(0)) & (total == expected):
            if cut != cutlass.Uint32(0xffffffff):
                scratch[64], scratch[65], scratch[66] = cut, above, in_bin
            cute.arch.sync_threads()
            result[0], result[1], result[2] = (scratch[64].to(cutlass.Uint32),
                                                scratch[65].to(cutlass.Uint32), scratch[66].to(cutlass.Uint32))
            result[3] = largest
            waiting = cutlass.Boolean(False)
        else:
            if cutlass.const_expr(completion is not None):
                if not have:
                    a, b, c, d = volatile_vector(completion.iterator + 8 * tid)
                    pw[0], pw[1], pw[2], pw[3] = a, b, c, d
                    a, b, c, d = volatile_vector(completion.iterator + 8 * tid + 4)
                    pw[4], pw[5], pw[6], pw[7] = a, b, c, d
            for j in cutlass.range_constexpr(6):
                a, b, c, d = volatile_vector(global_hist.iterator + 4 * (tid + 256 * j))
                values[4 * j], values[4 * j + 1], values[4 * j + 2], values[4 * j + 3] = a, b, c, d
            parity ^= 1
    return (result[0].to(cutlass.Uint32), result[1].to(cutlass.Uint32),
                result[2].to(cutlass.Uint32), result[3].to(cutlass.Uint32))


@cute.jit
def rs_emit(storage: cute.Tensor, d0: cutlass.Uint32, d1: cutlass.Uint32,
                d2: cutlass.Uint32, d3: cutlass.Uint32, topk: cute.Tensor,
                candidates: cute.Tensor, slots: cute.Pointer):
    tid, _, _ = cute.arch.thread_idx()
    scratch = cute.make_tensor(storage.iterator + (200704 + 512) // 4, cute.make_layout((8,)))
    hist = cute.make_tensor(storage.iterator + 135680 // 4, cute.make_layout((6144,)))
    if tid == 0:
        base_above, base_bin, n_above, n_bin = (cutlass.Uint32(0), cutlass.Uint32(0),
                                              cutlass.Uint32(0), cutlass.Uint32(0))
        if (d3 >> 24) != cutlass.Uint32(4):
            n_above = hist[d0].to(cutlass.Uint32)
            n_bin = hist[d0 - cutlass.Uint32(1)].to(cutlass.Uint32) - n_above
            if n_above + n_bin != cutlass.Uint32(0):
                old = global_add64(slots, (n_bin.to(cutlass.Uint64) << 32) | n_above.to(cutlass.Uint64))
                base_above = (old & cutlass.Uint64(0xffffffff)).to(cutlass.Uint32)
                base_bin = (old >> 32).to(cutlass.Uint32)
        scratch[0], scratch[1], scratch[2], scratch[3] = base_above, base_bin, n_above, n_bin
        scratch[4], scratch[5], scratch[6], scratch[7] = d0, d1, d2, d3
    cute.arch.sync_threads()
    emitted = cutlass.Boolean(False)
    descriptor3 = scratch[7].to(cutlass.Uint32)
    if (descriptor3 >> 24) != cutlass.Uint32(4):
        emitted = cutlass.Boolean(True)
        descriptor2 = scratch[6].to(cutlass.Uint32)
        above = (descriptor2 & cutlass.Uint32(4095)) - cutlass.Uint32(1)
        want = (descriptor2 >> 12) - above
        direct = (descriptor3 >> 24) == cutlass.Uint32(1)
        base_above, base_bin = scratch[0].to(cutlass.Uint32), scratch[1].to(cutlass.Uint32)
        n_above, n_bin = scratch[2].to(cutlass.Uint32), scratch[3].to(cutlass.Uint32)
        for i in range(tid, n_bin.to(cutlass.Int32), 256):
            index = n_above + i.to(cutlass.Uint32)
            pos = storage[7936 + index].to(cutlass.Uint32)
            j = base_bin + i.to(cutlass.Uint32)
            if not direct:
                pair = (pos.to(cutlass.Uint64) << 32) | storage[index].to(cutlass.Uint32).to(cutlass.Uint64)
                store64(candidates.iterator + 2 * j, pair)
            elif j < want:
                store_word(topk.iterator + above + j, pos)
        for i in range(tid, n_above.to(cutlass.Int32), 256):
            store_word(topk.iterator + base_above + i,
                           storage[7936 + i].to(cutlass.Uint32))
    return emitted


@cute.jit
def rs_selection_emit(cta: cutlass.Int32, rank: cutlass.Int32, length: cutlass.Int32,
                          storage: cute.Tensor, global_hist: cute.Tensor,
                          completion: cute.Tensor, counters: cute.Tensor,
                          candidates: cute.Tensor, topk: cute.Tensor):
    tid, _, _ = cute.arch.thread_idx()
    mine = selection_pages(cta, length)
    local_cut = rs_flush(cta, storage, global_hist, completion)
    if cta != 204:
        rs_sort(cta, mine, storage, local_cut, False)
    valid_keys = (length - 8 + rank // 2 + 1).to(cutlass.Uint32)
    need = valid_keys
    if need > cutlass.Uint32(2048):
        need = cutlass.Uint32(2048)
    slots = counters.iterator + 40 + 2 * (rank // 2)
    phase, finished = cutlass.Int32(0), cutlass.Boolean(False)
    while (phase < 2) & (not finished):
        descriptor = counters.iterator + 644 + 4 * phase
        d0, d1, d2, d3 = cutlass.Uint32(0), cutlass.Uint32(0), cutlass.Uint32(0), cutlass.Uint32(0)
        if cta == 204:
            cut, above, in_bin, largest = cutlass.Uint32(0), cutlass.Uint32(0), cutlass.Uint32(0), cutlass.Uint32(0)
            if phase == 0:
                cut, above, in_bin, largest = rs_row_cut(global_hist, completion, length, valid_keys, need, storage)
            else:
                cut, above, in_bin, largest = rs_row_cut(global_hist, None, length, valid_keys, need, storage)
            if tid == 0:
                mode = cutlass.Uint32(3)
                if cut < largest:
                    mode = cutlass.Uint32(4)
                elif (need - above == in_bin) | (cut == cutlass.Uint32(2946)) | (cut == cutlass.Uint32(2947)):
                    mode = cutlass.Uint32(1)
                elif in_bin <= cutlass.Uint32(512):
                    mode = cutlass.Uint32(2)
                d0, d1 = cut + cutlass.Uint32(1), cutlass.Uint32(1)
                d2, d3 = (above + cutlass.Uint32(1)) | (need << 12), in_bin | (mode << 24)
                store_vector(descriptor, d0, d1, d2, d3)
            if phase == 0:
                rs_sort(cta, mine, storage, local_cut, False)
        elif tid < 32:
            d0, d1, d2, d3 = volatile_vector(descriptor)
            ready = cutlass.Boolean(False)
            while not ready:
                if cute.arch.vote_all_sync((d0 != cutlass.Uint32(0)) & (d1 != cutlass.Uint32(0))
                                              & (d2 != cutlass.Uint32(0)) & (d3 != cutlass.Uint32(0))):
                    ready = cutlass.Boolean(True)
                else:
                    d0, d1, d2, d3 = volatile_vector(descriptor)
        finished = rs_emit(storage, d0, d1, d2, d3, topk, candidates, slots)
        if not finished:
            rs_sort(cta, mine, storage, local_cut, True, global_hist)
        phase += 1
    if cta == 204:
        for i in range(need.to(cutlass.Int32) + tid, 2048, 256):
            store_word(topk.iterator + i, cutlass.Uint32(0xffffffff))


@dsl_user_op
def poll64(pointer: cute.Pointer, *, loc=None, ip=None):
    return cutlass.Uint64(llvm.load(T.i64(), as_ptr(1, pointer, loc, ip), volatile_=True, alignment=8, loc=loc, ip=ip))


@dsl_user_op
def cached64(pointer: cute.Pointer, *, loc=None, ip=None):
    return cutlass.Uint64(cute.arch.load(pointer, cutlass.Uint64, cop="cg", loc=loc, ip=ip))


@dsl_user_op
def shared_pair64(pointer: cute.Pointer, *, loc=None, ip=None):
    v = llvm.load(ir.VectorType.get([2], T.i64()), as_ptr(3, pointer, loc, ip), alignment=16, loc=loc, ip=ip)
    return tuple(cutlass.Uint64(vector.extract(v, [], [i], loc=loc, ip=ip)) for i in range(2))


@cute.jit
def bin_from_top(hist: cute.Tensor, bins: cutlass.Constexpr, need: cutlass.Uint32,
                     scalars: cute.Tensor):
    tid, _, _ = cute.arch.thread_idx()
    warp, lane = tid >> 5, tid & 31
    per = bins // 256
    high = bins - per * tid
    count = cutlass.Uint32(0)
    for j in cutlass.range_constexpr(per):
        count += hist[high - 1 - j].to(cutlass.Uint32)
    inclusive = count
    for step in cutlass.range_constexpr(5):
        offset = 1 << step
        other = cute.arch.shuffle_sync_up(inclusive, offset)
        if lane >= offset:
            inclusive += other
    if lane == 31:
        scalars[warp] = inclusive
    cute.arch.sync_threads()
    before = inclusive - count
    for w in range(warp):
        before += scalars[w].to(cutlass.Uint32)
    if (before < need) & (before + count >= need):
        found = cutlass.Boolean(False)
        for j in cutlass.range_constexpr(per):
            if not found:
                count_here = hist[high - 1 - j].to(cutlass.Uint32)
                if before + count_here >= need:
                    scalars[8], scalars[9] = (high - 1 - j).to(cutlass.Uint32), before
                    found = cutlass.Boolean(True)
                else:
                    before += count_here
    cute.arch.sync_threads()


@cute.jit
def select_global(candidates: cute.Tensor, count: cutlass.Int32, need: cutlass.Uint32,
                      topk: cute.Tensor, above: cutlass.Uint32, storage: cute.Tensor):
    tid, _, _ = cute.arch.thread_idx()
    hist = cute.make_tensor(storage.iterator, cute.make_layout((4096,)))
    scalars = cute.make_tensor(storage.iterator + 4096, cute.make_layout((12,)))
    for i in cutlass.range_constexpr(16):
        hist[tid + 256 * i] = cutlass.Uint32(0)
    if tid == 0:
        scalars[10], scalars[11] = cutlass.Uint32(0), cutlass.Uint32(0)
    cute.arch.sync_threads()
    all_keys = count.to(cutlass.Uint32) <= need
    for base in range(tid, count, 2048):
        held = cute.make_rmem_tensor((8,), cutlass.Uint64)
        for u in cutlass.range_constexpr(8):
            index = base + 256 * u
            held[u] = cutlass.Uint64(0)
            if index < count:
                held[u] = poll64(candidates.iterator + 2 * index)
        for u in cutlass.range_constexpr(8):
            index = base + 256 * u
            if index < count:
                word = held[u].to(cutlass.Uint64)
                while (word & cutlass.Uint64(0xffffffff)) == cutlass.Uint64(0):
                    word = poll64(candidates.iterator + 2 * index)
                if not all_keys:
                    shared_add(hist.iterator + ((word & cutlass.Uint64(0xffffffff)).to(cutlass.Uint32) >> 20),
                                   cutlass.Uint32(1))
    cute.arch.sync_threads()
    threshold, greater = cutlass.Uint32(0), cutlass.Uint32(0)
    if not all_keys:
        bin_from_top(hist, 4096, need, scalars)
        first, above1 = scalars[8].to(cutlass.Uint32), scalars[9].to(cutlass.Uint32)
        for j in cutlass.range_constexpr(16):
            hist[tid + 256 * j] = cutlass.Uint32(0)
        cute.arch.sync_threads()
        for i in range(tid, count, 256):
            key = (cached64(candidates.iterator + 2 * i) & cutlass.Uint64(0xffffffff)).to(cutlass.Uint32)
            if (key >> 20) == first:
                shared_add(hist.iterator + ((key >> 8) & cutlass.Uint32(4095)), cutlass.Uint32(1))
        cute.arch.sync_threads()
        bin_from_top(hist, 4096, need - above1, scalars)
        second, above2 = scalars[8].to(cutlass.Uint32), scalars[9].to(cutlass.Uint32)
        high24 = (first << 12) | second
        hist[tid] = cutlass.Uint32(0)
        cute.arch.sync_threads()
        for i in range(tid, count, 256):
            key = (cached64(candidates.iterator + 2 * i) & cutlass.Uint64(0xffffffff)).to(cutlass.Uint32)
            if (key >> 8) == high24:
                shared_add(hist.iterator + (key & cutlass.Uint32(255)), cutlass.Uint32(1))
        cute.arch.sync_threads()
        bin_from_top(hist, 256, need - above1 - above2, scalars)
        threshold = (high24 << 8) | scalars[8].to(cutlass.Uint32)
        greater = above1 + above2 + scalars[9].to(cutlass.Uint32)
    ties = cutlass.Uint32(0)
    if not all_keys:
        ties = need - greater
    for i in range(tid, count, 256):
        word = cached64(candidates.iterator + 2 * i)
        key = (word & cutlass.Uint64(0xffffffff)).to(cutlass.Uint32)
        slot = cutlass.Int32(-1)
        if all_keys | (key > threshold):
            slot = shared_add(scalars.iterator + 10, cutlass.Uint32(1)).to(cutlass.Int32)
        elif key == threshold:
            tie = shared_add(scalars.iterator + 11, cutlass.Uint32(1))
            if tie < ties:
                slot = (greater + tie).to(cutlass.Int32)
        if slot >= 0:
            store_word(topk.iterator + above + slot, (word >> 32).to(cutlass.Uint32))
        store64(candidates.iterator + 2 * i, cutlass.Uint64(0))
    filled = count
    if count.to(cutlass.Uint32) > need:
        filled = need.to(cutlass.Int32)
    for i in range(filled + tid, need.to(cutlass.Int32), 256):
        store_word(topk.iterator + above + i, cutlass.Uint32(0xffffffff))


@cute.jit
def rank_boundary(candidates: cute.Tensor, count: cutlass.Int32, want: cutlass.Uint32,
                      topk: cute.Tensor, above: cutlass.Uint32, storage: cute.Tensor):
    tid, _, _ = cute.arch.thread_idx()
    scratch = cute.make_tensor(cute.recast_ptr(storage.iterator + (200704 + 1024) // 4,
                                                  dtype=cutlass.Uint64), cute.make_layout((512,)))
    held = cute.make_rmem_tensor((2,), cutlass.Uint64)
    for u in cutlass.range_constexpr(2):
        held[u] = cutlass.Uint64(1)
        if tid + 256 * u < count:
            held[u] = poll64(candidates.iterator + 2 * (tid + 256 * u))
    waiting = cutlass.Boolean(True)
    while waiting:
        if cute.arch.vote_all_sync(((held[0].to(cutlass.Uint64) & cutlass.Uint64(0xffffffff)) != cutlass.Uint64(0))
                                      & ((held[1].to(cutlass.Uint64) & cutlass.Uint64(0xffffffff)) != cutlass.Uint64(0))):
            waiting = cutlass.Boolean(False)
        else:
            for u in cutlass.range_constexpr(2):
                if (held[u].to(cutlass.Uint64) & cutlass.Uint64(0xffffffff)) == cutlass.Uint64(0):
                    held[u] = poll64(candidates.iterator + 2 * (tid + 256 * u))
    for u in cutlass.range_constexpr(2):
        if tid + 256 * u < count:
            pair = held[u].to(cutlass.Uint64)
            key = (pair & cutlass.Uint64(0xffffffff)).to(cutlass.Uint32)
            position = (pair >> 32).to(cutlass.Uint32)
            scratch[tid + 256 * u] = (key.to(cutlass.Uint64) << 32) | (position ^ cutlass.Uint32(0xffffffff)).to(cutlass.Uint64)
    padded = (count + 7) & -8
    if tid < padded - count:
        scratch[count + tid] = cutlass.Uint64(0)
    cute.arch.sync_threads()
    for u in cutlass.range_constexpr(2):
        i = tid + 256 * u
        if (i & -32) < count:
            own = cutlass.Uint64(0)
            if i < count:
                own = scratch[i].to(cutlass.Uint64)
            ranks = cute.make_rmem_tensor((4,), cutlass.Uint32)
            for k in cutlass.range_constexpr(4):
                ranks[k] = cutlass.Uint32(0)
            for x in range(0, padded, 8):
                for k in cutlass.range_constexpr(4):
                    q0, q1 = shared_pair64(scratch.iterator + x + 2 * k)
                    add = cutlass.Uint32(0)
                    if q0 > own:
                        add += cutlass.Uint32(1)
                    if q1 > own:
                        add += cutlass.Uint32(1)
                    ranks[k] += add
            rank = (ranks[0].to(cutlass.Uint32) + ranks[1].to(cutlass.Uint32)) + (ranks[2].to(cutlass.Uint32) + ranks[3].to(cutlass.Uint32))
            if (i < count) & (rank < want):
                position = (own & cutlass.Uint64(0xffffffff)).to(cutlass.Uint32) ^ cutlass.Uint32(0xffffffff)
                store_word(topk.iterator + above + rank, position)
    for u in cutlass.range_constexpr(2):
        if tid + 256 * u < count:
            store64(candidates.iterator + 2 * (tid + 256 * u), cutlass.Uint64(0))


@cute.jit
def rs_boundary(storage: cute.Tensor, candidates: cute.Tensor, topk: cute.Tensor,
                    global_hist: cute.Tensor, completion: cute.Tensor):
    tid, _, _ = cute.arch.thread_idx()
    descriptor = cute.make_tensor(storage.iterator + (200704 + 512) // 4, cute.make_layout((8,)))
    d2, d3 = descriptor[6].to(cutlass.Uint32), descriptor[7].to(cutlass.Uint32)
    above = (d2 & cutlass.Uint32(4095)) - cutlass.Uint32(1)
    want = (d2 >> 12) - above
    count = (d3 & cutlass.Uint32(0xffffff)).to(cutlass.Int32)
    mode = d3 >> 24
    if mode == cutlass.Uint32(3):
        cute.arch.sync_threads()
        select_global(candidates, count, want, topk, above, storage)
    elif mode != cutlass.Uint32(1):
        rank_boundary(candidates, count, want, topk, above, storage)
    for j in cutlass.range_constexpr(6):
        store_vector(global_hist.iterator + 4 * (tid + 256 * j), cutlass.Uint32(0), cutlass.Uint32(0),
                         cutlass.Uint32(0), cutlass.Uint32(0))
    for i in range(tid, 424, 256):
        store_vector(completion.iterator + 4 * i, cutlass.Uint32(0), cutlass.Uint32(0),
                         cutlass.Uint32(0), cutlass.Uint32(0))


# Imported last: the role modules import each other, and their names are used only when a kernel is traced.
from .attention import (
    bf16_pair,
    published_word,
    read_weight,
    chunk_mma,
    reduce_tiles,
    e4m3_pair,
    bf16_from_word,
    shared_store4,
    volatile_vector,
)
from .plain import operand_words
from .moe import volatile_word, redux_max
from .common import (
    as_ptr,
    div_rn_ftz,
    mul_rn,
    fma_rn,
    max_num,
    evict_first,
    bits_f32,
    proxy_fence,
    tc_before,
    tc_after,
    commit_mma,
    f32_bits,
)
from .gpu_publication import store_word, store_vector
