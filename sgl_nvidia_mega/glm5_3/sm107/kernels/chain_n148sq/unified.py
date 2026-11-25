# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
import cutlass
import cutlass.cute as cute
from cutlass.memory import SmemAllocator


@cute.kernel
def unified_kernel(
    x_raw_ptr: cute.Pointer,
    residual_raw: cute.Tensor,
    residual_out_raw: cute.Tensor,
    normalized_raw_ptr: cute.Pointer,
    wfa_raw_ptr: cute.Pointer,
    wqb_raw: cute.Tensor,
    wo_raw: cute.Tensor,
    wlg_raw: cute.Tensor,
    wgu_raw: cute.Tensor,
    wd_raw: cute.Tensor,
    w1_raw: cute.Tensor,
    w2_raw: cute.Tensor,
    gamma_kv: cute.Tensor,
    gamma_post_raw: cute.Tensor,
    gamma_next_raw: cute.Tensor,
    bias_raw: cute.Tensor,
    alpha1: cute.Tensor,
    alpha2: cute.Tensor,
    rope_raw: cute.Tensor,
    length: cute.Tensor,
    table_raw: cute.Tensor,
    cache_raw: cute.Tensor,
    topk_raw: cute.Tensor,
    arena_raw: cute.Tensor,
    partials_raw: cute.Tensor,
    stats_raw: cute.Tensor,
    fc2_raw: cute.Tensor,
    counters_raw: cute.Tensor,
    lists_raw: cute.Tensor,
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
):
    # x and normalized are read only through 32-bit word views, so the first layer (BF16) and the chained layers
    # (FP32 carry) share one CUBIN.
    is_leader = idx_pages > cutlass.Int64(0)
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
    if is_leader:
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
            wlg_raw,
            wgu_raw,
            wd_raw,
            w1_raw,
            w2_raw,
            gamma_post_raw,
            gamma_next_raw,
            bias_raw,
            alpha1,
            alpha2,
            fc2_raw,
            lists_raw,
            x_only,
            storage,
        )
    else:
        current = copy * ARENA_WORDS
        next_copy, previous_copy = (copy + 1) % COPIES, (copy + COPIES - 1) % COPIES
        x = word_view(x_raw, 0, (8, 3072), (3072, 1))
        wfa = word_view(wfa_raw, 0, (8060928,), (1,))
        wqb = word_view(wqb_raw, 0, (2359296,), (1,))
        rope = cute.make_tensor(
            rope_raw.iterator,
            cute.make_layout((cute.size(rope_raw) // 128, 64), stride=(128, 1)),
        )
        table = cute.make_tensor(
            table_raw.iterator, cute.make_layout((cute.size(table_raw),))
        )
        cache = cute.make_tensor(
            cute.recast_ptr(cache_raw.iterator, dtype=cutlass.Uint32),
            cute.make_layout((cute.size(cache_raw) // 576, 144), stride=(144, 1)),
        )
        x_frag = word_view(arena_raw, current + 2174720 // 4, (24576,), (1,))
        x_ssq = word_view(arena_raw, current + 2469632 // 4, (8, 4), (4, 1))
        previous_clear = word_view(
            counters_raw, previous_copy * COUNTER_WORDS, (1,), (1,)
        )
        y_offset = (
            2 * COPIES * HOP_WORDS
            + COPIES * TK_WORDS
            + COPIES * LG_WORDS
            + copy * Y_WORDS
        )
        y = word_view(uc_raw, y_offset, (8, 1312), (1312, 1))
        y_mc = word_view(mc_raw, y_offset, (8, 1312), (1312, 1))
        q8 = word_view(arena_raw, current + 41984 // 4, (8, 576), (576, 1))
        kv_arena = word_view(arena_raw, current + 60416 // 4, (8, 144), (144, 1))
        next_counters = word_view(
            counters_raw, next_copy * COUNTER_WORDS, (COUNTER_WORDS,), (1,)
        )
        plain_front_body(
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
            next_counters,
            y_mc,
            storage,
        )
    leader_suffix(
        x_raw,
        residual_raw,
        residual_out_raw,
        normalized_raw,
        wfa_raw,
        wqb_raw,
        wo_raw,
        wlg_raw,
        wgu_raw,
        wd_raw,
        w1_raw,
        w2_raw,
        gamma_kv,
        gamma_post_raw,
        gamma_next_raw,
        bias_raw,
        alpha1,
        alpha2,
        rope_raw,
        length,
        table_raw,
        cache_raw,
        topk_raw,
        arena_raw,
        partials_raw,
        stats_raw,
        fc2_raw,
        counters_raw,
        lists_raw,
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
        storage,
        is_leader,
    )


from .moe import SHARED_WORDS

# Imported last: the role modules import each other, and their names are used only when a kernel is traced.
from .plain import leader_front_body, leader_suffix, plain_front_body, word_view
from .preparation import (
    ARENA_WORDS,
    COPIES,
    COUNTER_WORDS,
    HOP_WORDS,
    LG_WORDS,
    TK_WORDS,
    Y_WORDS,
)
