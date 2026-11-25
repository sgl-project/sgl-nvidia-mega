# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
import cutlass
import cutlass.cute as cute
from cutlass._mlir import ir
from cutlass._mlir.dialects import func

MOE_E = 0
MOE_RCP = 8
MOE_SLOT_OF_TK = 32
MOE_SLOT_EXPERT = 96
MOE_WEIGHT = 160
MOE_SLOT_MASK = 224
ROUTE_E = 0
ROUTE_SLOT_EXPERT = 4
ROUTE_SLOT_OF_TK = 68
ROUTE_WEIGHT = 132


def _module_block():
    op = ir.InsertionPoint.current.block.owner
    while op is not None and op.name not in ("gpu.module", "builtin.module"):
        op = op.parent
    if op is None:
        raise RuntimeError("outlined call needs an enclosing gpu.module")
    return op, op.regions[0].blocks[0]


def call_outlined(name, body, *args):
    values = [cutlass.Uint32(a).ir_value() for a in args]
    module, block = _module_block()
    if name not in ir.SymbolTable(module.operation):
        with ir.InsertionPoint(block):
            callee = func.FuncOp(name, ([v.type for v in values], []))
            callee.sym_visibility = ir.StringAttr.get("private")
            callee.no_inline = True
            entry = callee.add_entry_block()
            with ir.InsertionPoint(entry):
                body(*[cutlass.Uint32(a) for a in entry.arguments])
                func.ReturnOp([])
    func.call([], name, values, no_inline=True)


@cute.jit
def word_view(
    raw: cute.Tensor,
    offset: cutlass.Int32,
    shape: cutlass.Constexpr,
    stride: cutlass.Constexpr,
):
    return cute.make_tensor(
        cute.recast_ptr(raw.iterator, dtype=cutlass.Uint32) + offset,
        cute.make_layout(shape, stride=stride),
    )


@cute.jit
def divide_hidden_rn(total: cutlass.Float32):
    bits = f32_bits(total)
    quotient = cutlass.Float32(0)
    if (bits >= cutlass.Uint32(0x07000000)) & (bits <= cutlass.Uint32(0x7F7FFFFF)):
        quotient = div_rn_ftz(total, cutlass.Float32(6144))
    else:
        quotient = total / cutlass.Float32(6144)
    return quotient


@cute.jit
def operand_words(
    destination: cute.Tensor,
    source: cute.Tensor,
    row_stride: cutlass.Int32,
    first_chunk: cutlass.Int32,
    chunks: cutlass.Constexpr,
    fragment: cutlass.Constexpr,
    poll: cutlass.Constexpr,
    force=None,
):
    tid, _, _ = cute.arch.thread_idx()
    lane = tid & 31
    for i in cutlass.range_constexpr(chunks):
        chunk = first_chunk + i
        for vector in cutlass.range_constexpr(2):
            offset = (lane >> 2) * row_stride + chunk * 32 + (lane & 3) * 8 + 4 * vector
            if cutlass.const_expr(fragment):
                offset = chunk * 256 + 4 * lane + 128 * vector
            if cutlass.const_expr(poll):
                v0, v1, v2, v3 = volatile_vector(source.iterator + offset)
            else:
                v0, v1, v2, v3 = cached_vector(source.iterator + offset)
            destination[i, 4 * vector], destination[i, 4 * vector + 1] = v0, v1
            destination[i, 4 * vector + 2], destination[i, 4 * vector + 3] = v2, v3
    if cutlass.const_expr(poll):
        waiting = cutlass.Boolean(True)
        while waiting:
            full = cutlass.Boolean(True)
            for i in cutlass.range_constexpr(chunks):
                for j in cutlass.range_constexpr(8):
                    full = full & (destination[i, j] != cutlass.Uint32(0x80000000))
            if cutlass.const_expr(force is not None):
                full = full | force
            if cute.arch.vote_all_sync(full):
                waiting = cutlass.Boolean(False)
            else:
                for i in cutlass.range_constexpr(chunks):
                    chunk = first_chunk + i
                    for vector in cutlass.range_constexpr(2):
                        ready = cutlass.Boolean(True)
                        for j in cutlass.range_constexpr(4):
                            ready = ready & (
                                destination[i, 4 * vector + j]
                                != cutlass.Uint32(0x80000000)
                            )
                        if not ready:
                            offset = (
                                (lane >> 2) * row_stride
                                + chunk * 32
                                + (lane & 3) * 8
                                + 4 * vector
                            )
                            if cutlass.const_expr(fragment):
                                offset = chunk * 256 + 4 * lane + 128 * vector
                            v0, v1, v2, v3 = volatile_vector(source.iterator + offset)
                            (
                                destination[i, 4 * vector],
                                destination[i, 4 * vector + 1],
                            ) = v0, v1
                            (
                                destination[i, 4 * vector + 2],
                                destination[i, 4 * vector + 3],
                            ) = v2, v3


@cute.jit
def fa_unit_present(unit: cutlass.Int32, y: cute.Tensor, red: cute.Tensor):
    """Every rank computes identical Fa bits from identical replicated x, so a unit an earlier rank published equals
    this rank's own result."""
    tid, _, _ = cute.arch.thread_idx()
    if tid < 32:
        full = cutlass.Boolean(True)
        if tid < 16:
            v0, v1, v2, v3 = volatile_vector(
                y.iterator + (tid >> 1) * 1312 + 8 * unit + 4 * (tid & 1)
            )
            full = (
                (v0 != cutlass.Uint32(0x80000000))
                & (v1 != cutlass.Uint32(0x80000000))
                & (v2 != cutlass.Uint32(0x80000000))
                & (v3 != cutlass.Uint32(0x80000000))
            )
        present = cute.arch.vote_all_sync(full)
        if tid == 0:
            red[2300] = cutlass.Float32(0)
            if present:
                red[2300] = cutlass.Float32(1)
    cute.arch.sync_threads()
    return red[2300] != cutlass.Float32(0)


@cute.jit
def first_weights(
    unit: cutlass.Int32, weights: cute.Tensor, region: cute.Tensor, bar: cute.Pointer
):
    cute.arch.mbarrier_arrive_and_expect_tx(bar, 196608)
    policy = evict_first()
    bulk_copy(
        region.iterator.toint().to(cutlass.Uint32),
        weights.iterator + unit * 49152,
        cutlass.Uint32(196608),
        bar,
        policy,
    )


@cute.jit
def fa_stage(
    unit: cutlass.Int32,
    x: cute.Tensor,
    x_frag: cute.Tensor,
    x_ssq: cute.Tensor,
    previous_clear: cute.Tensor,
    early_x: cutlass.Boolean,
    region: cute.Tensor,
    red: cute.Tensor,
    bar: cute.Pointer,
    y: cute.Tensor,
    idx_kw=None,
    query_weights=None,
    query_bar=None,
    system_fence: cutlass.Constexpr = True,
    scale_carry=None,
    stock_order: cutlass.Constexpr = False,
):
    tid, _, _ = cute.arch.thread_idx()
    warp, lane = tid >> 5, tid & 31
    region_base = region.iterator.toint().to(cutlass.Uint32)
    scaled = early_x
    if cutlass.const_expr(scale_carry is not None):
        scaled = scale_carry
    b = cute.make_rmem_tensor((12, 8), cutlass.Uint32)
    if cutlass.const_expr(stock_order):
        b = cute.make_rmem_tensor((96,), cutlass.Uint32)
    if early_x:
        if tid == 0:
            count = acquire_word(previous_clear.iterator)
            while count < cutlass.Uint32(COUNTED_CLEARS):
                count = acquire_word(previous_clear.iterator)
        cute.arch.sync_threads()
        if cutlass.const_expr(stock_order):
            mma8_operands(b, x_frag, True, True)
        else:
            operand_words(b, x_frag, 3072, 12 * warp, 12, True, True)
    else:
        if cutlass.const_expr(stock_order):
            mma8_operands(b, x, False, False)
        else:
            operand_words(b, x, 3072, 12 * warp, 12, False, False)
    xsq = cute.make_rmem_tensor((4,), cutlass.Uint32)
    if scaled & (warp == 0):
        v0, v1, v2, v3 = volatile_vector(x_ssq.iterator + 4 * (tid & 7))
        xsq[0], xsq[1], xsq[2], xsq[3] = v0, v1, v2, v3
    cute.arch.mbarrier_wait(bar, 0)
    a = cute.make_rmem_tensor((16,), cutlass.Uint32)
    partial = cute.make_rmem_tensor((4, 4), cutlass.Float32)
    for chain in cutlass.range_constexpr(4):
        for r in cutlass.range_constexpr(4):
            partial[chain, r] = cutlass.Float32(0)
    if cutlass.const_expr(stock_order):
        mma8_mainloop(region, b, partial)
    for i in cutlass.range_constexpr(0 if stock_order else 12):
        chunk = 12 * warp + i
        for piece in cutlass.range_constexpr(4):
            w0, w1, w2, w3 = lds128(
                region_base
                + ((chunk * 512 + piece * 128 + 4 * lane) * 4).to(cutlass.Uint32)
            )
            a[4 * piece], a[4 * piece + 1], a[4 * piece + 2], a[4 * piece + 3] = (
                w0,
                w1,
                w2,
                w3,
            )
        for step in cutlass.range_constexpr(4):
            index = (step // 2) * 4 + (step % 2) * 2
            c0, c1, c2, c3 = mma_bf16(
                partial[i & 3, 0],
                partial[i & 3, 1],
                partial[i & 3, 2],
                partial[i & 3, 3],
                a[index],
                a[index + 8],
                a[index + 1],
                a[index + 9],
                b[i, 2 * step],
                b[i, 2 * step + 1],
            )
            (
                partial[i & 3, 0],
                partial[i & 3, 1],
                partial[i & 3, 2],
                partial[i & 3, 3],
            ) = c0, c1, c2, c3
    if scaled & (warp == 0):
        waiting = cutlass.Boolean(True)
        while waiting:
            full = cutlass.Boolean(True)
            for j in cutlass.range_constexpr(4):
                full = full & (xsq[j] != cutlass.Uint32(0x80000000))
            if cute.arch.vote_all_sync(full):
                waiting = cutlass.Boolean(False)
            elif not full:
                v0, v1, v2, v3 = volatile_vector(x_ssq.iterator + 4 * (tid & 7))
                xsq[0], xsq[1], xsq[2], xsq[3] = v0, v1, v2, v3
        total = cutlass.Float32(0)
        for j in cutlass.range_constexpr(4):
            total += bits_f32(xsq[j])
        rstd = cute.math.rsqrt(divide_hidden_rn(total) + 1.0e-5, fastmath=True)
        if tid < 8:
            red[2176 + tid] = rstd
    for r in cutlass.range_constexpr(4):
        if cutlass.const_expr(stock_order):
            red[warp * 128 + lane * 4 + r] = partial[0, r] + partial[1, r]
        else:
            red[warp * 128 + lane * 4 + r] = (partial[0, r] + partial[1, r]) + (
                partial[2, r] + partial[3, r]
            )
    cute.arch.sync_threads()
    if tid < 128:
        owner, r = tid // 4, tid % 4
        value = cutlass.Float32(0)
        for w in cutlass.range_constexpr(8):
            value += red[w * 128 + owner * 4 + r]
        row = (owner >> 2) + (r // 2) * 8
        token = 2 * (owner & 3) + (r & 1)
        red[2048 + row * 8 + token] = value
    cute.arch.sync_threads()
    if cutlass.const_expr(query_weights is not None):
        if (unit < 128) & (tid == 0):
            cute.arch.mbarrier_arrive_and_expect_tx(query_bar, 131072)
            policy = evict_first()
            bulk_copy(
                region.iterator.toint().to(cutlass.Uint32),
                query_weights.iterator + unit * 32768,
                cutlass.Uint32(131072),
                query_bar,
                policy,
            )
    if tid < 64:
        token, pair = tid // 8, tid % 8
        scale = cutlass.Float32(1)
        if scaled:
            scale = red[2176 + token]
        lo, hi = (
            red[2048 + 2 * pair * 8 + token] * scale,
            red[2048 + (2 * pair + 1) * 8 + token] * scale,
        )
        packet = published_word(bf16_pair(lo, hi))
        if cutlass.const_expr(idx_kw is None):
            store_word(y.iterator + token * 1312 + 8 * unit + pair, packet)
            if cutlass.const_expr(system_fence):
                fence_sys()
        else:
            if unit < 164:
                store_word(y.iterator + token * 1312 + 8 * unit + pair, packet)
            else:
                store_word(
                    idx_kw.iterator + token * 80 + 8 * (unit - 164) + pair, packet
                )


@cute.jit
def qb_stage(
    unit: cutlass.Int32,
    y: cute.Tensor,
    region: cute.Tensor,
    red: cute.Tensor,
    stats: cute.Tensor,
    bar: cute.Pointer,
    rope: cute.Tensor,
    length: cute.Tensor,
    q8: cute.Tensor,
    clamp_pos: cutlass.Constexpr = False,
):
    tid, _, _ = cute.arch.thread_idx()
    warp, lane = tid >> 5, tid & 31
    region_base = region.iterator.toint().to(cutlass.Uint32)
    factors = cute.make_rmem_tensor((4,), cutlass.Float32)
    factors[0], factors[1], factors[2], factors[3] = (
        cutlass.Float32(1),
        cutlass.Float32(0),
        cutlass.Float32(1),
        cutlass.Float32(0),
    )
    ep_row = (48 * unit + 4 * (tid % 12)) % 576
    if (tid < 96) & (ep_row >= 512):
        pos = length[0] - 8 + tid // 12
        if cutlass.const_expr(clamp_pos):
            if pos < 0:
                pos = cutlass.Int32(0)
        for j in cutlass.range_constexpr(4):
            factors[j] = rope[pos, ep_row - 512 + j]
    b = cute.make_rmem_tensor((4, 8), cutlass.Uint32)
    operand_words(b, y, 1312, 4 * warp, 4, False, True)
    ssq = cutlass.Float32(0)
    for i in cutlass.range_constexpr(4):
        for j in cutlass.range_constexpr(8):
            lo, hi = bf16_from_word(b[i, j], False), bf16_from_word(b[i, j], True)
            ssq = ssq + lo * lo + hi * hi
    ssq += cute.arch.shuffle_sync_bfly(ssq, 1)
    ssq += cute.arch.shuffle_sync_bfly(ssq, 2)
    if (lane & 3) == 0:
        stats[warp * 8 + (lane >> 2)] = ssq
    cute.arch.mbarrier_wait(bar, 0)
    acc = cute.make_rmem_tensor((3, 4), cutlass.Float32)
    a = cute.make_rmem_tensor((16,), cutlass.Uint32)
    for tile in cutlass.range_constexpr(3):
        for r in cutlass.range_constexpr(4):
            acc[tile, r] = cutlass.Float32(0)
    for i in cutlass.range_constexpr(4):
        chunk = 4 * warp + i
        for tile in cutlass.range_constexpr(3):
            for piece in cutlass.range_constexpr(4):
                w0, w1, w2, w3 = lds128(
                    region_base
                    + (((tile * 32 + chunk) * 512 + piece * 128 + 4 * lane) * 4).to(
                        cutlass.Uint32
                    )
                )
                a[4 * piece], a[4 * piece + 1], a[4 * piece + 2], a[4 * piece + 3] = (
                    w0,
                    w1,
                    w2,
                    w3,
                )
            for step in cutlass.range_constexpr(4):
                index = (step // 2) * 4 + (step % 2) * 2
                c0, c1, c2, c3 = mma_bf16(
                    acc[tile, 0],
                    acc[tile, 1],
                    acc[tile, 2],
                    acc[tile, 3],
                    a[index],
                    a[index + 8],
                    a[index + 1],
                    a[index + 9],
                    b[i, 2 * step],
                    b[i, 2 * step + 1],
                )
                acc[tile, 0], acc[tile, 1], acc[tile, 2], acc[tile, 3] = c0, c1, c2, c3
    for tile in cutlass.range_constexpr(3):
        for r in cutlass.range_constexpr(4):
            red[((warp * 3 + tile) * 32 + lane) * 4 + r] = acc[tile, r]
    cute.arch.sync_threads()
    for step in cutlass.range_constexpr(2):
        index = tid + 256 * step
        if index < 384:
            tile, owner, r = index // 128, (index // 4) % 32, index % 4
            value = cutlass.Float32(0)
            for w in cutlass.range_constexpr(8):
                value += red[((w * 3 + tile) * 32 + owner) * 4 + r]
            row = 16 * tile + (owner >> 2) + (r // 2) * 8
            token = 2 * (owner & 3) + (r & 1)
            red[3072 + row * 8 + token] = value
    cute.arch.sync_threads()
    if tid < 8:
        total = cutlass.Float32(0)
        for w in cutlass.range_constexpr(8):
            total += stats[w * 8 + tid]
        stats[64 + tid] = cute.math.rsqrt(total / 2048.0 + 1.0e-5, fastmath=True)
    cute.arch.sync_threads()
    if tid < 96:
        token, q4 = tid // 12, tid % 12
        first_row = 48 * unit + 4 * q4
        scale = stats[64 + token]
        v0 = red[3072 + (4 * q4) * 8 + token] * scale
        v1 = red[3072 + (4 * q4 + 1) * 8 + token] * scale
        v2 = red[3072 + (4 * q4 + 2) * 8 + token] * scale
        v3 = red[3072 + (4 * q4 + 3) * 8 + token] * scale
        if first_row % 576 >= 512:
            n0 = fma_rn(factors[0], v0, -mul_rn(factors[1], v1))
            n1 = fma_rn(factors[1], v0, mul_rn(factors[0], v1))
            n2 = fma_rn(factors[2], v2, -mul_rn(factors[3], v3))
            n3 = fma_rn(factors[3], v2, mul_rn(factors[2], v3))
            v0, v1, v2, v3 = n0, n1, n2, n3
        store_word(
            q8.iterator + token * 576 + 12 * unit + q4,
            published_word(e4m3_pair(v0, v1) | (e4m3_pair(v2, v3) << 16)),
        )


@cute.jit
def leader_front_body(
    x_raw: cute.Tensor,
    wfa_raw: cute.Tensor,
    wqb_raw: cute.Tensor,
    widx_raw: cute.Tensor,
    gamma_kv: cute.Tensor,
    rope_raw: cute.Tensor,
    length: cute.Tensor,
    table_raw: cute.Tensor,
    cache_raw: cute.Tensor,
    arena_raw: cute.Tensor,
    counters_raw: cute.Tensor,
    copy: cutlass.Int32,
    rank: cutlass.Int32,
    early_x: cutlass.Boolean,
    stride_rows: cutlass.Int64,
    layer_rows: cutlass.Int64,
    idx_kn_w: cute.Tensor,
    idx_kn_b: cute.Tensor,
    idx_rope_raw: cute.Tensor,
    idx_pool_raw: cute.Tensor,
    idx_blocks_raw: cute.Tensor,
    idx_stride_bytes: cutlass.Int64,
    idx_pages: cutlass.Int64,
    idx_candidates_raw: cute.Tensor,
    mc_raw: cute.Tensor,
    uc_raw: cute.Tensor,
    topk_raw: cute.Tensor,
    partials_raw: cute.Tensor,
    stats_raw: cute.Tensor,
    max_blocks: cutlass.Int32,
    clear_first: cutlass.Boolean,
    key_atom: cute.CopyAtom,
    residual_raw: cute.Tensor,
    residual_out_raw: cute.Tensor,
    normalized_raw: cute.Tensor,
    wo_raw: cute.Tensor,
    wlg_raw: cute.Tensor,
    wgu_raw: cute.Tensor,
    wd_raw: cute.Tensor,
    w1_raw: cute.Tensor,
    w2_raw: cute.Tensor,
    gamma_post_raw: cute.Tensor,
    gamma_next_raw: cute.Tensor,
    bias_raw: cute.Tensor,
    alpha1: cute.Tensor,
    alpha2: cute.Tensor,
    fc2_raw: cute.Tensor,
    x_only: cutlass.Boolean,
    storage: cute.Tensor,
    x_carry=None,
    scale_carry=None,
    fa_stock: cutlass.Constexpr = False,
    stock_idxq: cutlass.Constexpr = False,
):
    tid, _, _ = cute.arch.thread_idx()
    cta, _, _ = cute.arch.block_idx()
    current = copy * ARENA_WORDS
    previous_copy = (copy + COPIES - 1) % COPIES
    x = word_view(x_raw, 0, (8, 3072), (3072, 1))
    x_frag = word_view(arena_raw, current + 2174720 // 4, (24576,), (1,))
    x_ssq = word_view(arena_raw, current + 2469632 // 4, (8, 4), (4, 1))
    if cutlass.const_expr(x_carry is not None):
        x_frag = x_carry
    previous_clear = word_view(counters_raw, previous_copy * COUNTER_WORDS, (1,), (1,))
    next_counters = word_view(
        counters_raw, ((copy + 1) % COPIES) * COUNTER_WORDS, (COUNTER_WORDS,), (1,)
    )
    wfa = word_view(wfa_raw, 0, (8552448,), (1,))
    wqb = word_view(wqb_raw, 0, (2359296,), (1,))
    widx = word_view(widx_raw, 0, (4194304,), (1,))
    y = word_view(arena_raw, current, (8, 1312), (1312, 1))
    q8 = word_view(arena_raw, current + 41984 // 4, (8, 576), (576, 1))
    idx_kw = word_view(arena_raw, current + 97792 // 4, (8, 80), (80, 1))
    kv = word_view(arena_raw, current + 60416 // 4, (8, 144), (144, 1))
    idx_key = word_view(arena_raw, current + 104448 // 4, (8, 32), (32, 1))
    idx_scale = word_view(arena_raw, current + 105472 // 4, (8,), (1,))
    idx_query = word_view(arena_raw, current + 2109184 // 4, (8, 32, 64), (2048, 64, 1))
    idx_weights = word_view(arena_raw, current + 2076416 // 4, (8, 32), (32, 1))
    topk = cute.make_tensor(
        topk_raw.iterator, cute.make_layout((8, 2048), stride=(2048, 1))
    )
    exchange = word_view(
        uc_raw, 2 * COPIES * HOP_WORDS + copy * TK_WORDS, (TK_WORDS,), (1,)
    )
    partials = word_view(partials_raw, 0, (262144,), (1,))
    stats = word_view(stats_raw, 0, (1024,), (1,))
    o = word_view(arena_raw, current + 65024 // 4, (8, 4, 256), (1024, 256, 1))
    replicas = word_view(
        arena_raw, current + 2273024 // 4, (3, 8, 4, 256), (8192, 1024, 256, 1)
    )
    rope = cute.make_tensor(
        rope_raw.iterator,
        cute.make_layout((cute.size(rope_raw) // 128, 64), stride=(128, 1)),
    )
    idx_rope = cute.make_tensor(
        idx_rope_raw.iterator,
        cute.make_layout((cute.size(idx_rope_raw) // 64, 64), stride=(64, 1)),
    )
    table = cute.make_tensor(
        table_raw.iterator, cute.make_layout((cute.size(table_raw),))
    )
    idx_blocks = cute.make_tensor(
        idx_blocks_raw.iterator, cute.make_layout((cute.size(idx_blocks_raw),))
    )
    cache = cute.make_tensor(
        cute.recast_ptr(cache_raw.iterator, dtype=cutlass.Uint32),
        cute.make_layout((cute.size(cache_raw) // 576, 144), stride=(144, 1)),
    )
    pool = cute.make_tensor(
        cute.recast_ptr(idx_pool_raw.iterator, dtype=cutlass.Uint32),
        cute.make_layout((idx_pages, 2112), stride=(idx_stride_bytes // 4, 1)),
    )
    region, red, residual, scalars, moe, bars, tmem_slot = shared_views(storage)
    if tid < 32:
        cute.arch.mbarrier_init(bars.iterator + tid, 1)
    storage[55296 + tid] = cutlass.Uint32(0)
    cute.arch.mbarrier_init_fence()
    if (cta < 174) & (tid == 0):
        first_weights(cta, wfa, region, bars.iterator)
    if tid // 32 == 1:
        allocate_tmem(tmem_slot.iterator)
    tc_before()
    cute.arch.sync_threads()
    tc_after()
    reset_next_counters(next_counters)
    early = early_x & (cta < 174)
    if not early:
        cute.arch.griddepcontrol_wait()
        cute.arch.griddepcontrol_launch_dependents()
    if cta < 174:
        fa_stage(
            cta,
            x,
            x_frag,
            x_ssq,
            previous_clear,
            early_x,
            region,
            red,
            bars.iterator,
            y,
            idx_kw,
            widx,
            bars.iterator + 5,
            scale_carry=scale_carry,
            stock_order=fa_stock,
        )
        proxy_fence()
        cute.arch.sync_threads()
    if early:
        cute.arch.griddepcontrol_wait()
        cute.arch.griddepcontrol_launch_dependents()
    if cta == 163:
        tail = cute.make_tensor(
            cute.recast_ptr(y.iterator + 1024, dtype=cutlass.BFloat16),
            cute.make_layout((8, 576), stride=(2624, 1)),
        )
        append_stage(
            tail, gamma_kv, rope, length, table, cache, kv, stride_rows, layer_rows
        )
        proxy_fence()
        cute.arch.sync_threads()
    if cta == 162:
        kw = cute.make_tensor(
            cute.recast_ptr(idx_kw.iterator, dtype=cutlass.BFloat16),
            cute.make_layout((8, 128), stride=(160, 1)),
        )
        indexer_key_stage(
            kw,
            idx_kn_w,
            idx_kn_b,
            idx_rope,
            length,
            idx_blocks,
            pool,
            idx_key,
            idx_scale,
        )
        proxy_fence()
        cute.arch.sync_threads()
    if cta < 128:
        indexer_query_stage(
            cta,
            y,
            idx_kw,
            region,
            red,
            scalars,
            bars.iterator + 5,
            idx_rope,
            length,
            idx_query,
            idx_weights,
            stock_idxq,
        )
        proxy_fence()
        cute.arch.sync_threads()
    if (length[0] > 2048) & ((rank & 1) == 0):
        if (selection_pages(cta, length[0]) > 0) | (cta == 204):
            selection_score_stage(
                cta,
                rank,
                length,
                idx_blocks,
                idx_pool_raw,
                idx_stride_bytes,
                idx_pages,
                key_atom,
                storage,
                bars,
                tmem_slot[0],
                idx_query,
                idx_weights,
                idx_key,
                idx_scale,
                stock_idxq,
                rope=idx_rope,
            )
            row_base = ((copy * 8 + rank // 2) * CAND_CAP * 2).to(cutlass.Int64)
            hist_pointer = cute.recast_ptr(
                idx_candidates_raw.iterator + row_base + 2 * CAND_LIST,
                dtype=cutlass.Uint32,
            )
            global_hist = cute.make_tensor(hist_pointer, cute.make_layout((6144,)))
            completion = cute.make_tensor(
                hist_pointer + 6144, cute.make_layout((1696,))
            )
            candidates = cute.make_tensor(
                cute.recast_ptr(
                    idx_candidates_raw.iterator + row_base, dtype=cutlass.Uint32
                ),
                cute.make_layout((2 * CAND_CAP,)),
            )
            counters = word_view(
                counters_raw, copy * COUNTER_WORDS, (COUNTER_WORDS,), (1,)
            )
            topk_mc = word_view(
                mc_raw,
                2 * COPIES * HOP_WORDS + copy * TK_WORDS + (rank // 2) * 2048,
                (2048,),
                (1,),
            )
            rs_selection_emit(
                cta,
                rank,
                length[0],
                storage,
                global_hist,
                completion,
                counters,
                candidates,
                topk_mc,
            )
            if cta == 204:
                rs_boundary(storage, candidates, topk_mc, global_hist, completion)
    if ((cta >= 164) & (cta < 204)) | ((cta >= 128) & (cta < 136)):
        proxy_fence()
        cute.arch.sync_threads()
        unit = cta - 164
        if cta < 136:
            unit = cta - 128 + 40
        if tid == 0:
            first_weights(unit, wqb, region, bars.iterator + 6)
        qb_stage(unit, y, region, red, scalars, bars.iterator + 6, rope, length, q8)
        proxy_fence()
        cute.arch.sync_threads()


@cute.jit
def issue_prefix_weights(
    wfa: cute.Tensor,
    wqb: cute.Tensor,
    region: cute.Tensor,
    bar: cute.Pointer,
    defer_fa=None,
):
    tid, _, _ = cute.arch.thread_idx()
    cta, _, _ = cute.arch.block_idx()
    issue_fa = cutlass.Boolean(True)
    if cutlass.const_expr(defer_fa is not None):
        issue_fa = not defer_fa
    if tid == 0:
        if cta < 164:
            if issue_fa:
                first_weights(cta, wfa, region, bar)
        else:
            first_weights(cta - 164, wqb, region, bar)


@cute.jit
def prefix_roles(
    x: cute.Tensor,
    x_frag: cute.Tensor,
    x_ssq: cute.Tensor,
    previous_clear: cute.Tensor,
    early_x: cutlass.Boolean,
    wfa: cute.Tensor,
    wqb: cute.Tensor,
    y: cute.Tensor,
    q8: cute.Tensor,
    gamma_kv: cute.Tensor,
    rope: cute.Tensor,
    length: cute.Tensor,
    table: cute.Tensor,
    cache: cute.Tensor,
    kv_arena: cute.Tensor,
    stride_rows: cutlass.Int64,
    layer_rows: cutlass.Int64,
    region: cute.Tensor,
    red: cute.Tensor,
    scalars: cute.Tensor,
    bar: cute.Pointer,
    weights_issued: cutlass.Constexpr = False,
    y_mc=None,
    append_from=None,
    fa_carry=None,
    fa_scale_carry=None,
    fa_stock: cutlass.Constexpr = False,
    short_ctx: cutlass.Constexpr = False,
):
    tid, _, _ = cute.arch.thread_idx()
    cta, _, _ = cute.arch.block_idx()
    if cutlass.const_expr(fa_carry is not None):
        x_frag = fa_carry
    if cutlass.const_expr(not weights_issued):
        issue_prefix_weights(wfa, wqb, region, bar)
    early = early_x & (cta < 164)
    if not early:
        cute.arch.griddepcontrol_wait()
        cute.arch.griddepcontrol_launch_dependents()
    if cta < 164:
        if cutlass.const_expr(y_mc is None):
            fa_stage(
                cta,
                x,
                x_frag,
                x_ssq,
                previous_clear,
                early_x,
                region,
                red,
                bar,
                y,
                system_fence=False,
                scale_carry=fa_scale_carry,
                stock_order=fa_stock,
            )
        else:
            compute = cutlass.Boolean(True)
            if not early:
                compute = not fa_unit_present(cta, y, red)
                if compute & (tid == 0):
                    first_weights(cta, wfa, region, bar)
            if compute:
                if early:
                    fa_stage(
                        cta,
                        x,
                        x_frag,
                        x_ssq,
                        previous_clear,
                        early_x,
                        region,
                        red,
                        bar,
                        y,
                        system_fence=False,
                        scale_carry=fa_scale_carry,
                        stock_order=fa_stock,
                    )
                else:
                    fa_stage(
                        cta,
                        x,
                        x_frag,
                        x_ssq,
                        previous_clear,
                        early_x,
                        region,
                        red,
                        bar,
                        y_mc,
                        scale_carry=fa_scale_carry,
                        stock_order=fa_stock,
                    )
        proxy_fence()
        cute.arch.sync_threads()
    if early:
        cute.arch.griddepcontrol_wait()
        cute.arch.griddepcontrol_launch_dependents()
    if cta >= 164:
        qb_stage(
            cta - 164,
            y,
            region,
            red,
            scalars,
            bar,
            rope,
            length,
            q8,
            clamp_pos=short_ctx,
        )
        proxy_fence()
        cute.arch.sync_threads()
    if cta == 163:
        tail = cute.make_tensor(
            cute.recast_ptr(y.iterator + 1024, dtype=cutlass.BFloat16),
            cute.make_layout((8, 576), stride=(2624, 1)),
        )
        append_stage(
            tail,
            gamma_kv,
            rope,
            length,
            table,
            cache,
            kv_arena,
            stride_rows,
            layer_rows,
            append_from,
            clamp_pos=short_ctx,
        )
        proxy_fence()
        cute.arch.sync_threads()


@cute.jit
def plain_front_body(
    x: cute.Tensor,
    x_frag: cute.Tensor,
    x_ssq: cute.Tensor,
    previous_clear: cute.Tensor,
    early_x: cutlass.Boolean,
    wfa: cute.Tensor,
    wqb: cute.Tensor,
    y: cute.Tensor,
    q8: cute.Tensor,
    gamma_kv: cute.Tensor,
    rope: cute.Tensor,
    length: cute.Tensor,
    table: cute.Tensor,
    cache: cute.Tensor,
    kv_arena: cute.Tensor,
    stride_rows: cutlass.Int64,
    layer_rows: cutlass.Int64,
    next_counters: cute.Tensor,
    y_mc: cute.Tensor,
    storage: cute.Tensor,
    append_from=None,
    fa_carry=None,
    fa_scale_carry=None,
    fa_stock: cutlass.Constexpr = False,
    short_ctx: cutlass.Constexpr = False,
):
    tid, _, _ = cute.arch.thread_idx()
    region, red, residual, scalars, moe, all_bars, tmem_slot = shared_views(storage)
    if tid < 32:
        cute.arch.mbarrier_init(all_bars.iterator + tid, 1)
    storage[55296 + tid] = cutlass.Uint32(0)
    cute.arch.mbarrier_init_fence()
    defer_fa = not early_x
    issue_prefix_weights(wfa, wqb, region, all_bars.iterator, defer_fa)
    if tid // 32 == 1:
        allocate_tmem(tmem_slot.iterator)
    tc_before()
    cute.arch.sync_threads()
    tc_after()
    reset_next_counters(next_counters)
    prefix_roles(
        x,
        x_frag,
        x_ssq,
        previous_clear,
        early_x,
        wfa,
        wqb,
        y,
        q8,
        gamma_kv,
        rope,
        length,
        table,
        cache,
        kv_arena,
        stride_rows,
        layer_rows,
        region,
        red,
        scalars,
        all_bars.iterator,
        weights_issued=True,
        y_mc=y_mc,
        append_from=append_from,
        fa_carry=fa_carry,
        fa_scale_carry=fa_scale_carry,
        fa_stock=fa_stock,
        short_ctx=short_ctx,
    )


# Imported last: the role modules import each other, and their names are used only when a kernel is traced.
from .attention import (
    append_stage,
    bf16_from_word,
    bf16_pair,
    e4m3_pair,
    mma8_mainloop,
    mma8_operands,
    mma_bf16,
    published_word,
    volatile_vector,
)
from .common import (
    COUNTED_CLEARS,
    allocate_tmem,
    bits_f32,
    bulk_copy,
    div_rn_ftz,
    evict_first,
    f32_bits,
    fma_rn,
    mul_rn,
    proxy_fence,
    reset_next_counters,
    tc_after,
    tc_before,
)
from .gpu_publication import acquire_word, cached_vector, fence_sys, lds128, store_word
from .indexer import (
    indexer_key_stage,
    indexer_query_stage,
    rs_boundary,
    rs_selection_emit,
    selection_pages,
    selection_score_stage,
)
from .moe import shared_views
from .preparation import (
    ARENA_WORDS,
    CAND_CAP,
    CAND_LIST,
    COPIES,
    COUNTER_WORDS,
    HOP_WORDS,
    TK_WORDS,
)
