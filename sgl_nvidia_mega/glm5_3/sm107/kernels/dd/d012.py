# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
import cutlass
import cutlass.cute as cute
from cutlass.cute.nvgpu import cpasync
from cutlass.memory import SmemAllocator

from .moe import XK_WORDS, dense_down, dense_gu, issue_dense_down, issue_dense_gu

P1_WORD = 306944 // 4
DENSE_H_WORD = P1_WORD + XK_WORDS


L0X_WORD = P1_WORD + 32768
EMBED_VOCAB = 154880
RMS_EPS = 1.0e-5


@cute.jit
def embed_norm_stage(
    unit: cutlass.Int32,
    ids: cute.Tensor,
    embed: cute.Tensor,
    gamma_in: cute.Tensor,
    residual_out0: cute.Tensor,
    x_l0: cute.Tensor,
    scalars: cute.Tensor,
):
    tid, _, _ = cute.arch.thread_idx()
    warp, lane = tid >> 5, tid & 31
    if unit < 8:
        token_id = ids[unit]
        valid = (token_id >= 0) & (token_id < EMBED_VOCAB)
        xr = cute.make_rmem_tensor((96,), cutlass.Float32)
        for b in cutlass.range_constexpr(12):
            col = b * 512 + 8 * (tid & 63)
            w0, w1, w2, w3 = (
                cutlass.Uint32(0),
                cutlass.Uint32(0),
                cutlass.Uint32(0),
                cutlass.Uint32(0),
            )
            if valid & (tid < 64):
                w0, w1, w2, w3 = readonly_vector(
                    embed.iterator + token_id * 3072 + col // 2
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
        if (lane == 0) & (warp < 2):
            scalars[warp] = local
        cute.arch.sync_threads()
        total = cutlass.Float32(0.0)
        if lane < 2:
            total = scalars[lane]
        for i in cutlass.range_constexpr(5):
            total = total + cute.arch.shuffle_sync_bfly(total, offset=1 << i)
        rstd = cute.math.rsqrt(
            total / cutlass.Float32(6144) + cutlass.Float32(RMS_EPS), fastmath=True
        )
        if tid < 64:
            for b in cutlass.range_constexpr(12):
                col = b * 512 + 8 * tid
                g0, g1, g2, g3 = readonly_vector(gamma_in.iterator + col // 2)
                gw = (g0, g1, g2, g3)
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
                    x_l0.iterator + xfrag_word_offset(unit, col),
                    out[0],
                    out[1],
                    out[2],
                    out[3],
                )
        cute.arch.sync_threads()
    token, quarter = unit // 4, unit % 4
    col = 1536 * quarter + 8 * tid
    if tid < 192:
        row = ids[token]
        e0, e1, e2, e3 = (
            cutlass.Uint32(0),
            cutlass.Uint32(0),
            cutlass.Uint32(0),
            cutlass.Uint32(0),
        )
        if (row >= 0) & (row < EMBED_VOCAB):
            e0, e1 = global_pair(embed.iterator + row * 3072 + col // 2)
            e2, e3 = global_pair(embed.iterator + row * 3072 + col // 2 + 2)
        store_vector(residual_out0.iterator + token * 3072 + col // 2, e0, e1, e2, e3)


@cute.jit
def wait_clears(clear_counter: cute.Tensor):
    tid, _, _ = cute.arch.thread_idx()
    if tid == 0:
        ready = acquire_word(clear_counter.iterator)
        while ready < cutlass.Uint32(COUNTED_CLEARS):
            ready = acquire_word(clear_counter.iterator)
    cute.arch.sync_threads()


@cute.jit
def dense_tail_roles(
    hop1: cute.Tensor,
    hop2: cute.Tensor,
    previous_hop2: cute.Tensor,
    h2_mc: cute.Tensor,
    residual_input: cute.Tensor,
    gamma_post: cute.Tensor,
    gamma_next: cute.Tensor,
    w_gu: cute.Tensor,
    w_dn: cute.Tensor,
    ssq1: cute.Tensor,
    ssq2: cute.Tensor,
    h: cute.Tensor,
    hp: cute.Tensor,
    hfp4: cute.Tensor,
    xk: cute.Tensor,
    hmid: cute.Tensor,
    next_x: cute.Tensor,
    next_ssq: cute.Tensor,
    xout: cute.Tensor,
    rout: cute.Tensor,
    clear_counter: cute.Tensor,
    next_arena: cute.Tensor,
    next_tk: cute.Tensor,
    rank: cutlass.Int32,
    x_only: cutlass.Boolean,
    region: cute.Tensor,
    red: cute.Tensor,
    residual: cute.Tensor,
    scalars: cute.Tensor,
    all_bars: cute.Tensor,
    tmem: cutlass.Uint32,
):
    tid, _, _ = cute.arch.thread_idx()
    cta, _, _ = cute.arch.block_idx()
    gu_bar = all_bars.iterator + 4
    dn_bar = all_bars.iterator + 24
    clear_assigned(next_arena, next_tk, clear_counter)
    if cta < 192:
        issue_dense_gu(cta, w_gu, region, gu_bar)
        issue_dense_down(cta, w_dn, region, dn_bar)
    if cta >= 180:
        stage_hop(
            cta - 180,
            False,
            x_only,
            hop1,
            hop2,
            previous_hop2,
            residual_input,
            gamma_post,
            cutlass.Float32(1),
            residual,
            scalars,
            ssq1,
            ssq2,
            h,
            hp,
            hfp4,
            next_x,
            next_ssq,
            xout,
            rout,
        )
    if cta < 192:
        dense_gu(cta, h, region, red, xk, hmid, cutlass.Int32(8), gu_bar, True)
        wait_clears(clear_counter)
        dense_down(cta, hmid, region, red, h2_mc, rank, cutlass.Int32(8), dn_bar)
    tc_before()
    cute.arch.sync_threads()
    if tid // 32 == 0:
        tc_after()
        deallocate_tmem(tmem)
    if cta >= 180:
        stage_hop(
            cta - 180,
            True,
            x_only,
            hop1,
            hop2,
            previous_hop2,
            residual_input,
            gamma_next,
            cutlass.Float32(1),
            residual,
            scalars,
            ssq1,
            ssq2,
            h,
            hp,
            hfp4,
            next_x,
            next_ssq,
            xout,
            rout,
        )


@cute.jit
def d012_suffix(
    residual_raw: cute.Tensor,
    residual_out_raw: cute.Tensor,
    normalized_raw: cute.Tensor,
    wo_raw: cute.Tensor,
    wgu_raw: cute.Tensor,
    wdn_raw: cute.Tensor,
    gamma_post_raw: cute.Tensor,
    gamma_next_raw: cute.Tensor,
    length: cute.Tensor,
    table_raw: cute.Tensor,
    cache_raw: cute.Tensor,
    topk_raw: cute.Tensor,
    arena_raw: cute.Tensor,
    partials_raw: cute.Tensor,
    stats_raw: cute.Tensor,
    counters_raw: cute.Tensor,
    uc_raw: cute.Tensor,
    mc_raw: cute.Tensor,
    copy: cutlass.Int32,
    rank: cutlass.Int32,
    x_only: cutlass.Boolean,
    stride_rows: cutlass.Int64,
    layer_rows: cutlass.Int64,
    max_blocks: cutlass.Int32,
    clear_first: cutlass.Boolean,
    storage: cute.Tensor,
):
    tid, _, _ = cute.arch.thread_idx()
    cta, _, _ = cute.arch.block_idx()
    region, red, residual, scalars, moe, all_bars, tmem_slot = shared_views(storage)
    current = copy * ARENA_WORDS
    next_copy, previous_copy = (copy + 1) % COPIES, (copy + COPIES - 1) % COPIES
    following = next_copy * ARENA_WORDS
    residual_input = word_view(residual_raw, 0, (8, 3072), (3072, 1))
    xout = word_view(normalized_raw, 0, (8, 3072), (3072, 1))
    rout = word_view(residual_out_raw, 0, (8, 3072), (3072, 1))
    wo = word_view(wo_raw, 0, (6291456,), (1,))
    w_gu = cute.make_tensor(
        cute.recast_ptr(wgu_raw.iterator, dtype=cutlass.BFloat16),
        cute.make_layout((1536 * 6144,)),
    )
    w_dn = cute.make_tensor(
        cute.recast_ptr(wdn_raw.iterator, dtype=cutlass.BFloat16),
        cute.make_layout((6144 * 768,)),
    )
    gamma_post = word_view(gamma_post_raw, 0, (3072,), (1,))
    gamma_next = word_view(gamma_next_raw, 0, (3072,), (1,))
    table = cute.make_tensor(
        table_raw.iterator, cute.make_layout((cute.size(table_raw),))
    )
    cache = cute.make_tensor(
        cute.recast_ptr(cache_raw.iterator, dtype=cutlass.Uint32),
        cute.make_layout((cute.size(cache_raw) // 576, 144), stride=(144, 1)),
    )
    topk = cute.make_tensor(
        topk_raw.iterator, cute.make_layout((8, 2048), stride=(2048, 1))
    )
    q8 = word_view(arena_raw, current + 41984 // 4, (8, 576), (576, 1))
    kv_arena = word_view(arena_raw, current + 60416 // 4, (8, 144), (144, 1))
    o = word_view(arena_raw, current + 65024 // 4, (8, 4, 256), (1024, 256, 1))
    replicas = word_view(
        arena_raw, current + 2273024 // 4, (3, 8, 4, 256), (8192, 1024, 256, 1)
    )
    logits = word_view(arena_raw, current + 105600 // 4, (8, 256), (256, 1))
    h = word_view(arena_raw, current + 114176 // 4, (24576,), (1,))
    hp = word_view(arena_raw, current + 2371328 // 4, (24576,), (1,))
    hfp4 = word_view(arena_raw, current + 212480 // 4, (6912,), (1,))
    ssq1 = word_view(arena_raw, current + 240128 // 4, (8, 4), (4, 1))
    ssq2 = word_view(arena_raw, current + 240256 // 4, (8, 4), (4, 1))
    xk = word_view(arena_raw, current + P1_WORD, (XK_WORDS,), (1,))
    hmid = word_view(arena_raw, current + DENSE_H_WORD, (3072,), (1,))
    next_x = word_view(arena_raw, following + 2174720 // 4, (24576,), (1,))
    next_ssq = word_view(arena_raw, following + 2469632 // 4, (8, 4), (4, 1))
    next_arena = word_view(arena_raw, following, (ARENA_WORDS,), (1,))
    partials = word_view(partials_raw, 0, (262144,), (1,))
    stats = word_view(stats_raw, 0, (1024,), (1,))
    clear_counter = word_view(counters_raw, copy * COUNTER_WORDS, (1,), (1,))
    hop1 = word_view(uc_raw, copy * HOP_WORDS, (HOP_WORDS,), (1,))
    h1_mc = word_view(mc_raw, copy * HOP_WORDS, (HOP_WORDS,), (1,))
    hop2 = word_view(
        uc_raw, (COPIES + copy) * HOP_WORDS, (8, 16, 3072), (49152, 3072, 1)
    )
    previous_hop2 = word_view(
        uc_raw, (COPIES + previous_copy) * HOP_WORDS, (8, 16, 3072), (49152, 3072, 1)
    )
    h2_mc = word_view(
        mc_raw, (COPIES + copy) * HOP_WORDS, (8, 16, 3072), (49152, 3072, 1)
    )
    next_tk = word_view(
        uc_raw, 2 * COPIES * HOP_WORDS + next_copy * TK_WORDS, (TK_WORDS,), (1,)
    )
    lg_offset = 2 * COPIES * HOP_WORDS + COPIES * TK_WORDS + copy * LG_WORDS
    lg_uc = word_view(uc_raw, lg_offset, (LG_WORDS,), (1,))
    lg_mc = word_view(mc_raw, lg_offset, (LG_WORDS,), (1,))
    exchange = word_view(
        uc_raw, 2 * COPIES * HOP_WORDS + copy * TK_WORDS, (TK_WORDS,), (1,)
    )
    if cta < 128:
        attention_stage(
            cta,
            q8,
            kv_arena,
            length,
            table,
            topk,
            cache,
            stride_rows,
            layer_rows,
            max_blocks,
            storage,
            all_bars,
            partials,
            stats,
            wo,
            all_bars.iterator + 1,
            exchange,
            True,
        )
        proxy_fence()
        cute.arch.sync_threads()
    elif cta < 192:
        if tid == 0:
            issue_o(cta, wo, wo, storage, all_bars.iterator + 1, False)
    if (cta >= 164) & (cta < 196):
        merge_stage(cta - 164, partials, stats, o, replicas, red, clear_first)
        proxy_fence()
        cute.arch.sync_threads()
    if cta < 192:
        o_stage(
            cta,
            rank,
            o,
            replicas,
            residual_input,
            wo,
            region,
            red,
            all_bars.iterator + 1,
            h1_mc,
            lg_uc,
            lg_mc,
            logits,
            tmem_slot[0],
            all_bars.iterator + 27,
            False,
        )
        proxy_fence()
        cute.arch.sync_threads()
    dense_tail_roles(
        hop1,
        hop2,
        previous_hop2,
        h2_mc,
        residual_input,
        gamma_post,
        gamma_next,
        w_gu,
        w_dn,
        ssq1,
        ssq2,
        h,
        hp,
        hfp4,
        xk,
        hmid,
        next_x,
        next_ssq,
        xout,
        rout,
        clear_counter,
        next_arena,
        next_tk,
        rank,
        x_only,
        region,
        red,
        residual,
        scalars,
        all_bars,
        tmem_slot[0],
    )


@cute.kernel
def d012_kernel(
    x_raw_ptr: cute.Pointer,
    residual_raw: cute.Tensor,
    residual_out_raw: cute.Tensor,
    normalized_raw_ptr: cute.Pointer,
    wfa_raw_ptr: cute.Pointer,
    wqb_raw: cute.Tensor,
    wo_raw: cute.Tensor,
    wgu_raw: cute.Tensor,
    wdn_raw: cute.Tensor,
    gamma_kv: cute.Tensor,
    gamma_post_raw: cute.Tensor,
    gamma_next_raw: cute.Tensor,
    rope_raw: cute.Tensor,
    length: cute.Tensor,
    table_raw: cute.Tensor,
    cache_raw: cute.Tensor,
    topk_raw: cute.Tensor,
    arena_raw: cute.Tensor,
    partials_raw: cute.Tensor,
    stats_raw: cute.Tensor,
    counters_raw: cute.Tensor,
    uc_raw: cute.Tensor,
    mc_raw: cute.Tensor,
    copy: cutlass.Int32,
    rank: cutlass.Int32,
    early_x: cutlass.Boolean,
    x_only: cutlass.Boolean,
    stride_rows: cutlass.Int64,
    layer_rows: cutlass.Int64,
    max_blocks: cutlass.Int32,
    clear_first: cutlass.Boolean,
    widx_raw_ptr: cute.Pointer,
    idx_kn_w_ptr: cute.Pointer,
    idx_kn_b_ptr: cute.Pointer,
    idx_rope_raw_ptr: cute.Pointer,
    idx_pool_raw_ptr: cute.Pointer,
    idx_blocks_raw_ptr: cute.Pointer,
    idx_stride_bytes: cutlass.Int64,
    idx_pages: cutlass.Int64,
    idx_candidates_raw: cute.Tensor,
    key_atom: cute.CopyAtom,
    layer0: cutlass.Boolean,
    ids: cute.Tensor,
    embed_raw: cute.Tensor,
    gamma_in_raw: cute.Tensor,
):
    tid, _, _ = cute.arch.thread_idx()
    cta, _, _ = cute.arch.block_idx()
    rope_rows = cute.size(rope_raw) // 128
    x_raw = cute.make_tensor(x_raw_ptr, cute.make_layout((24576,)))
    normalized_raw = cute.make_tensor(normalized_raw_ptr, cute.make_layout((24576,)))
    wfa_raw = cute.make_tensor(wfa_raw_ptr, cute.make_layout((174 * 196608,)))
    widx_raw = cute.make_tensor(widx_raw_ptr, cute.make_layout((256 * 65536,)))
    idx_kn_w = cute.make_tensor(idx_kn_w_ptr, cute.make_layout((128,)))
    idx_kn_b = cute.make_tensor(idx_kn_b_ptr, cute.make_layout((128,)))
    idx_rope_raw = cute.make_tensor(
        idx_rope_raw_ptr, cute.make_layout((rope_rows * 64,))
    )
    idx_pool_raw = cute.make_tensor(
        idx_pool_raw_ptr, cute.make_layout(((idx_pages - 1) * idx_stride_bytes + 8448,))
    )
    idx_blocks_raw = cute.make_tensor(
        idx_blocks_raw_ptr, cute.make_layout((rope_rows // 64,))
    )
    allocator = SmemAllocator()
    storage = allocator.allocate_tensor(
        cutlass.Uint32, cute.make_layout((SHARED_WORDS,)), byte_alignment=1024
    )
    if layer0 & (cta >= 180):
        cute.arch.griddepcontrol_wait()
        _, _, _, scalars, _, _, _ = shared_views(storage)
        residual_out0 = word_view(residual_raw, 0, (8, 3072), (3072, 1))
        embed = cute.make_tensor(
            cute.recast_ptr(embed_raw.iterator, dtype=cutlass.Uint32),
            cute.make_layout((cute.size(embed_raw) // 2,)),
        )
        gamma_in = word_view(gamma_in_raw, 0, (3072,), (1,))
        embed_norm_stage(
            cta - 180,
            ids,
            embed,
            gamma_in,
            residual_out0,
            word_view(arena_raw, copy * ARENA_WORDS + L0X_WORD, (24576,), (1,)),
            scalars,
        )
    carry_word = copy * ARENA_WORDS + 2174720 // 4
    scale_carry = early_x
    if layer0:
        carry_word = copy * ARENA_WORDS + L0X_WORD
        scale_carry = cutlass.Boolean(False)
    x_carry = word_view(arena_raw, carry_word, (24576,), (1,))
    leader_front_body(
        x_raw,
        wfa_raw,
        wqb_raw,
        widx_raw,
        gamma_kv,
        rope_raw,
        length,
        table_raw,
        cache_raw,
        arena_raw,
        counters_raw,
        copy,
        rank,
        early_x,
        stride_rows,
        layer_rows,
        idx_kn_w,
        idx_kn_b,
        idx_rope_raw,
        idx_pool_raw,
        idx_blocks_raw,
        idx_stride_bytes,
        idx_pages,
        idx_candidates_raw,
        mc_raw,
        uc_raw,
        topk_raw,
        partials_raw,
        stats_raw,
        max_blocks,
        clear_first,
        key_atom,
        residual_raw,
        residual_out_raw,
        normalized_raw,
        wo_raw,
        wo_raw,
        wgu_raw,
        wdn_raw,
        wgu_raw,
        wdn_raw,
        gamma_post_raw,
        gamma_next_raw,
        gamma_post_raw,
        partials_raw,
        partials_raw,
        partials_raw,
        x_only,
        storage,
        x_carry=x_carry,
        scale_carry=scale_carry,
        fa_stock=True,
        stock_idxq=True,
    )
    d012_suffix(
        residual_raw,
        residual_out_raw,
        normalized_raw,
        wo_raw,
        wgu_raw,
        wdn_raw,
        gamma_post_raw,
        gamma_next_raw,
        length,
        table_raw,
        cache_raw,
        topk_raw,
        arena_raw,
        partials_raw,
        stats_raw,
        counters_raw,
        uc_raw,
        mc_raw,
        copy,
        rank,
        x_only,
        stride_rows,
        layer_rows,
        max_blocks,
        clear_first,
        storage,
    )


@cute.jit
def d012_entry(
    x_raw: cute.Tensor,
    residual_raw: cute.Tensor,
    residual_out_raw: cute.Tensor,
    normalized_raw: cute.Tensor,
    wfa_raw: cute.Tensor,
    wqb_raw: cute.Tensor,
    wo_raw: cute.Tensor,
    wgu_raw: cute.Tensor,
    wdn_raw: cute.Tensor,
    gamma_kv: cute.Tensor,
    gamma_post_raw: cute.Tensor,
    gamma_next_raw: cute.Tensor,
    rope_raw: cute.Tensor,
    length: cute.Tensor,
    table_raw: cute.Tensor,
    cache_raw: cute.Tensor,
    topk_raw: cute.Tensor,
    arena_raw: cute.Tensor,
    partials_raw: cute.Tensor,
    stats_raw: cute.Tensor,
    counters_raw: cute.Tensor,
    uc_raw: cute.Tensor,
    mc_raw: cute.Tensor,
    copy: cutlass.Int32,
    rank: cutlass.Int32,
    early_x: cutlass.Boolean,
    x_only: cutlass.Boolean,
    stride_rows: cutlass.Int64,
    layer_rows: cutlass.Int64,
    max_blocks: cutlass.Int32,
    clear_first: cutlass.Boolean,
    widx_raw: cute.Tensor,
    idx_kn_w: cute.Tensor,
    idx_kn_b: cute.Tensor,
    idx_rope_raw: cute.Tensor,
    idx_pool_raw: cute.Tensor,
    idx_blocks_raw: cute.Tensor,
    idx_stride_bytes: cutlass.Int64,
    idx_pages: cutlass.Int64,
    idx_candidates_raw: cute.Tensor,
    layer0: cutlass.Boolean,
    ids: cute.Tensor,
    embed_raw: cute.Tensor,
    gamma_in_raw: cute.Tensor,
    stream,
    use_pdl: cutlass.Constexpr,
):
    pool_rows = cute.make_tensor(
        idx_pool_raw.iterator,
        cute.make_layout(
            (128, ((idx_pages - 1) * idx_stride_bytes + 8448) // 128), stride=(1, 128)
        ),
    )
    key_layout = cute.make_composed_layout(
        cute.make_swizzle(3, 4, 3), 0, cute.make_layout((128, 64), stride=(1, 128))
    )
    key_atom, _ = cpasync.make_tiled_tma_atom(
        cpasync.CopyBulkTensorTileG2SOp(), pool_rows, key_layout, (128, 64)
    )
    x_words = cute.make_ptr(
        cutlass.Uint32, x_raw.iterator.toint(), cute.AddressSpace.gmem, assumed_align=4
    )
    normalized_words = cute.make_ptr(
        cutlass.Uint32,
        normalized_raw.iterator.toint(),
        cute.AddressSpace.gmem,
        assumed_align=4,
    )
    d012_kernel(
        x_words,
        residual_raw,
        residual_out_raw,
        normalized_words,
        wfa_raw.iterator,
        wqb_raw,
        wo_raw,
        wgu_raw,
        wdn_raw,
        gamma_kv,
        gamma_post_raw,
        gamma_next_raw,
        rope_raw,
        length,
        table_raw,
        cache_raw,
        topk_raw,
        arena_raw,
        partials_raw,
        stats_raw,
        counters_raw,
        uc_raw,
        mc_raw,
        copy,
        rank,
        early_x,
        x_only,
        stride_rows,
        layer_rows,
        max_blocks,
        clear_first,
        widx_raw.iterator,
        idx_kn_w.iterator,
        idx_kn_b.iterator,
        idx_rope_raw.iterator,
        idx_pool_raw.iterator,
        idx_blocks_raw.iterator,
        idx_stride_bytes,
        idx_pages,
        idx_candidates_raw,
        key_atom,
        layer0,
        ids,
        embed_raw,
        gamma_in_raw,
    ).launch(
        grid=(212, 1, 1),
        block=(256, 1, 1),
        min_blocks_per_mp=1,
        stream=stream,
        use_pdl=use_pdl,
    )


from .attention import (
    attention_stage,
    bf16_from_word,
    bf16_pair,
    issue_o,
    merge_stage,
    o_stage,
    published_word,
)
from .common import (
    COUNTED_CLEARS,
    clear_assigned,
    deallocate_tmem,
    proxy_fence,
    stage_hop,
    tc_after,
    tc_before,
    xfrag_word_offset,
)
from .gpu_publication import acquire_word, global_pair, readonly_vector, store_vector
from .moe import SHARED_WORDS, shared_views

# Imported last: the role modules import each other, and their names are used only when a kernel is traced.
from .plain import leader_front_body, word_view
from .preparation import (
    ARENA_WORDS,
    COPIES,
    COUNTER_WORDS,
    HOP_WORDS,
    LG_WORDS,
    TK_WORDS,
)
