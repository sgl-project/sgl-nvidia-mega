# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
import cutlass
import cutlass.cute as cute
from cutlass.cute.nvgpu import cpasync


@cute.jit
def decoder_entry(
    x_raw: cute.Tensor,
    residual_raw: cute.Tensor,
    residual_out_raw: cute.Tensor,
    normalized_raw: cute.Tensor,
    wfa_raw: cute.Tensor,
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
    widx_raw: cute.Tensor,
    idx_kn_w: cute.Tensor,
    idx_kn_b: cute.Tensor,
    idx_rope_raw: cute.Tensor,
    idx_pool_raw: cute.Tensor,
    idx_blocks_raw: cute.Tensor,
    idx_stride_bytes: cutlass.Int64,
    idx_pages: cutlass.Int64,
    idx_candidates_raw: cute.Tensor,
    leader: cutlass.Constexpr,
    stream,
    use_pdl: cutlass.Constexpr,
):
    # One TMA atom type for both layer kinds: followers pass a 198-row extent that is never read.
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
    unified_kernel(
        x_words,
        residual_raw,
        residual_out_raw,
        normalized_words,
        wfa_raw.iterator,
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
    ).launch(
        grid=(212, 1, 1),
        block=(256, 1, 1),
        min_blocks_per_mp=1,
        stream=stream,
        use_pdl=use_pdl,
    )


# Imported last: the role modules import each other, and their names are used only when a kernel is traced.
from .unified import unified_kernel
