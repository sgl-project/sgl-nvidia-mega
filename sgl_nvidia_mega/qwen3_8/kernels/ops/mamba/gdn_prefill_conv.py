# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Copyright (c) 2024, Tri Dao.
# Adapted from SGLang (Apache-2.0, SGLang Team):
# python/sglang/kernels/ops/mamba/causal_conv1d_triton.py, itself adapted from
# https://github.com/Dao-AILab/causal-conv1d/blob/main/causal_conv1d/causal_conv1d_interface.py
# and https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/layers/mamba/ops/causal_conv1d.py
"""The GDN prefill causal conv (forward, varlen) with the split q / k / v output.

Byte-identical to the source: ``_causal_conv1d_fwd_kernel`` (with its
contiguous q / k / v planes and the width-4 2D tile), ``causal_conv1d_fn``,
``causal_conv1d_fn_split_qkv`` and the token-strip glue (``_PrefillConvStrip``,
``_parse_prefill_conv_strip``, ``_prefill_conv_strip``). Changed: its environment
reads are arguments. SGLANG_CAUSAL_CONV1D_FWD_CONFIG is ``causal_conv1d_fn``'s
``fwd_launch`` (a ``_parse_fwd_launch_config`` dict; its once-log is gone);
SGLANG_OPT_GDN_PREFILL_CONV_STRIP with SGLANG_GDN_PREFILL_CONV_RETILE is
``causal_conv1d_fn_split_qkv``'s ``strip_cfg`` and ``_prefill_conv_strip``'s ``cfg``;
``warmup_prefill_conv_strip(cfg)`` raises where the source cleared its global. The
strip kernel is this package's ``gdn_prefill_conv_strip``. The update kernel (decode
and verify) is not copied: SGLang's serves those."""

import logging
from typing import List, NamedTuple, Optional, Union

import torch
import triton
import triton.language as tl

logger = logging.getLogger(__name__)

PAD_SLOT_ID = -1


# Tokens per program of _causal_conv1d_fwd_kernel; a block_table enumerates
# (sequence, chunk) pairs in these units.
CAUSAL_CONV1D_FWD_BLOCK_M = 8

_STOCK_FWD_LAUNCH = {
    "BLOCK_M": CAUSAL_CONV1D_FWD_BLOCK_M,
    "BLOCK_N": 256,
    "num_stages": 2,
}


def _split_fwd_launch(width: int, dim: int) -> dict:
    """The split q / k / v launch: the GDN front 2D tile (BLOCK_N 256, 4 warps,
    2 stages, the block-table BLOCK_M, 2D tile) at width 4; the stock tile otherwise."""
    if width != 4:
        return _STOCK_FWD_LAUNCH
    return {
        **_STOCK_FWD_LAUNCH,
        "num_warps": 4,
        "TILE_2D": True,
        "EVEN_FEATS": dim % _STOCK_FWD_LAUNCH["BLOCK_N"] == 0,
    }


def _parse_fwd_launch_config(value: str) -> Optional[dict]:
    """``"BLOCK_M,BLOCK_N,num_warps,num_stages"`` for the rectangular forward launch.

    Each output is computed by one thread from the same inputs in tap order with no
    cross-thread reduction, so the tile only decides which program computes it.
    BLOCK_M >= 8 keeps a chunk's prior-token reads inside its sequence and the grid
    no taller than the default's."""
    value = value.strip()
    if value in ("", "0"):
        return None
    try:
        block_m, block_n, num_warps, num_stages = (int(v) for v in value.split(","))
    except ValueError:
        raise ValueError(
            "SGLANG_CAUSAL_CONV1D_FWD_CONFIG must be "
            f"'BLOCK_M,BLOCK_N,num_warps,num_stages' (e.g. '32,1024,4,2'), got {value!r}"
        ) from None
    if not (
        8 <= block_m <= 1024
        and num_warps in (1, 2, 4, 8, 16)
        and block_n & (block_n - 1) == 0
        and 32 * num_warps <= block_n <= 4096
        and 1 <= num_stages <= 8
    ):
        raise ValueError(
            "SGLANG_CAUSAL_CONV1D_FWD_CONFIG out of range: BLOCK_M 8..1024, num_warps "
            "a power of 2 up to 16, BLOCK_N a power of 2 from 32 * num_warps to 4096, "
            f"num_stages 1..8; got {value!r}"
        )
    return {
        "BLOCK_M": block_m,
        "BLOCK_N": block_n,
        "num_warps": num_warps,
        "num_stages": num_stages,
    }


def _fwd_launch_config(
    block_table: Optional[torch.Tensor], width: int, fwd_launch: Optional[dict]
) -> dict:
    # The block-table launch keeps the default tile: its table counts chunks of
    # CAUSAL_CONV1D_FWD_BLOCK_M tokens. The kernel's tap chain covers widths 2-4.
    if fwd_launch is None or block_table is not None or not 2 <= width <= 4:
        return _STOCK_FWD_LAUNCH
    return fwd_launch


@triton.jit()
def _causal_conv1d_fwd_kernel(  # continuous batching
    # Pointers to matrices
    x_ptr,  # (dim, cu_seqlen) holding `batch` of actual sequences + padded sequences
    w_ptr,  # (dim, width)
    bias_ptr,
    initial_states_ptr,  # conv_states_ptr
    cache_indices_ptr,  # conv_state_indices_ptr
    has_initial_states_ptr,
    query_start_loc_ptr,
    o_ptr,  # (dim, seqlen) - actually pointing to x_ptr
    block_table_ptr,  # (num_blocks, 2) int32 (seq, chunk) rows; USE_BLOCK_TABLE only
    # Matrix dimensions
    dim: tl.constexpr,
    total_tokens: tl.int32,  # cu_seqlen
    num_cache_lines: tl.constexpr,  # added to support vLLM larger cache lines
    # Strides
    stride_x_seq: tl.constexpr,  # stride to get to next sequence,
    stride_x_dim: tl.constexpr,  # stride to get to next feature-value,
    stride_x_token: tl.constexpr,  # stride to get to next token (same feature-index, same sequence-index)
    stride_w_dim: tl.constexpr,  # stride to get to next dim-axis value
    stride_w_width: tl.constexpr,  # stride to get to next width-axis value
    stride_istate_seq: tl.constexpr,
    stride_istate_dim: tl.constexpr,
    stride_istate_token: tl.constexpr,
    stride_o_seq: tl.constexpr,
    stride_o_dim: tl.constexpr,
    stride_o_token: tl.constexpr,
    # others
    pad_slot_id: tl.constexpr,
    # Meta-parameters
    HAS_BIAS: tl.constexpr,
    KERNEL_WIDTH: tl.constexpr,
    SILU_ACTIVATION: tl.constexpr,
    HAS_INITIAL_STATES: tl.constexpr,
    HAS_CACHE: tl.constexpr,
    IS_CONTINUOUS_BATCHING: tl.constexpr,
    USE_PAD_SLOT: tl.constexpr,
    NP2_STATELEN: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    USE_BLOCK_TABLE: tl.constexpr = False,
    OUTPUT_SPLIT_SIZE: tl.constexpr = 0,
    # GDN split output: when SPLIT_D0 > 0, o_ptr is a flat buffer of three
    # token-major planes [T, SPLIT_D0] | [T, SPLIT_D1 - SPLIT_D0] | [T, dim - SPLIT_D1]
    # (q | k | v). BLOCK_N divides SPLIT_D0 and SPLIT_D1, so a program never straddles.
    SPLIT_D0: tl.constexpr = 0,
    SPLIT_D1: tl.constexpr = 0,
    # Split q / k / v launch only (KERNEL_WIDTH == 4): the whole [BLOCK_M, BLOCK_N]
    # chunk from KERNEL_WIDTH shifted 2D loads instead of the per-token loop, with the
    # same per-element taps in the same order (the GDN front 2D tile).
    TILE_2D: tl.constexpr = False,
    EVEN_FEATS: tl.constexpr = False,  # dim % BLOCK_N == 0 (unmasked interior chunks)
):
    conv_states_ptr = initial_states_ptr
    conv_state_indices_ptr = cache_indices_ptr
    stride_conv_state_seq = stride_istate_seq
    stride_conv_state_dim = stride_istate_dim
    stride_conv_state_tok = stride_istate_token
    state_len = (
        KERNEL_WIDTH - 1
    )  # can be passed via argument if it's not the same as this value

    # one program handles one chunk in a single sequence
    # rather than mixing sequences - to make updating initial_states across sequences efficiently

    if USE_BLOCK_TABLE:
        # CUDA-graph safe launch: the grid is a fixed upper bound on the
        # (sequence, BLOCK_M-chunk) pairs of a token bucket and each program
        # reads its pair from the table; padded rows carry pad_slot_id.
        idx_block = tl.program_id(0)
        idx_seq = tl.load(block_table_ptr + idx_block * 2)
        chunk_offset = tl.load(block_table_ptr + idx_block * 2 + 1)
    else:
        # single-sequence id
        idx_seq = tl.program_id(0)
        chunk_offset = tl.program_id(1)

    # BLOCK_N elements along the feature-dimension (channel)
    idx_feats = tl.program_id(2) * BLOCK_N + tl.arange(0, BLOCK_N)

    if idx_seq == pad_slot_id:
        return

    sequence_start_index = tl.load(query_start_loc_ptr + idx_seq)
    sequence_end_index = tl.load(query_start_loc_ptr + idx_seq + 1)
    # find the actual sequence length
    seqlen = sequence_end_index - sequence_start_index

    token_offset = BLOCK_M * chunk_offset
    segment_len = min(BLOCK_M, seqlen - token_offset)

    if segment_len <= 0:
        return

    # base of the sequence
    x_base = (
        x_ptr
        + sequence_start_index.to(tl.int64) * stride_x_token
        + idx_feats * stride_x_dim
    )  # [BLOCK_N,]

    if IS_CONTINUOUS_BATCHING:
        # cache_idx
        conv_state_batch_coord = tl.load(conv_state_indices_ptr + idx_seq).to(tl.int64)
    else:
        # cache_idx
        conv_state_batch_coord = idx_seq
    if USE_PAD_SLOT:  # noqa
        if conv_state_batch_coord == pad_slot_id:
            # not processing as this is not the actual sequence
            return
    conv_states_base = (
        conv_states_ptr
        + (conv_state_batch_coord * stride_conv_state_seq)
        + (idx_feats * stride_conv_state_dim)
    )  # [BLOCK_N,]

    w_base = w_ptr + (idx_feats * stride_w_dim)  # [BLOCK_N,]

    # Does 2 things:
    # 1. READ prior-block init-state data - [done by every Triton programs]
    # 2. update conv_state with new data [only by the Triton program handles chunk_offset=0]
    if chunk_offset == 0:
        # read from conv_states
        load_init_state = False
        if HAS_INITIAL_STATES:  # the new HAS_INITIAL_STATES
            load_init_state = tl.load(has_initial_states_ptr + idx_seq).to(tl.int1)
        if load_init_state:
            # load from conv_states. Cast to x's dtype so col* keep a single
            # dtype across the whole kernel: when x is fp16 but the conv-state
            # cache is bf16 (e.g. MiniCPM-V GDN prefill), the chunk_offset==0
            # branch would otherwise produce bf16 cols while the chunk_offset>0
            # else branch (and the sliding-window reassignment) produce fp16,
            # which trips Triton's if/else phi type check on col0.
            x_elem_ty = x_ptr.dtype.element_ty
            prior_tokens = conv_states_base + (state_len - 1) * stride_conv_state_tok
            mask_w = idx_feats < dim
            if KERNEL_WIDTH == 2:
                conv_states_ptrs = prior_tokens  # [BLOCK_N]
                col0 = tl.load(conv_states_ptrs, mask_w, 0.0).to(x_elem_ty)
            if KERNEL_WIDTH == 3:
                conv_states_ptrs = prior_tokens  # [BLOCK_N]
                col1 = tl.load(conv_states_ptrs, mask_w, 0.0).to(x_elem_ty)
                conv_states_ptrs = prior_tokens - 1 * stride_conv_state_tok  # [BLOCK_N]
                col0 = tl.load(conv_states_ptrs, mask_w, 0.0).to(x_elem_ty)
            if KERNEL_WIDTH == 4:
                conv_states_ptrs = prior_tokens  # [BLOCK_N]
                col2 = tl.load(conv_states_ptrs, mask_w, 0.0).to(x_elem_ty)
                conv_states_ptrs = prior_tokens - 1 * stride_conv_state_tok  # [BLOCK_N]
                col1 = tl.load(conv_states_ptrs, mask_w, 0.0).to(x_elem_ty)
                conv_states_ptrs = prior_tokens - 2 * stride_conv_state_tok  # [BLOCK_N]
                col0 = tl.load(conv_states_ptrs, mask_w, 0.0).to(x_elem_ty)
            if KERNEL_WIDTH == 5:
                conv_states_ptrs = prior_tokens  # [BLOCK_N]
                col3 = tl.load(conv_states_ptrs, mask_w, 0.0).to(x_elem_ty)
                conv_states_ptrs = prior_tokens - 1 * stride_conv_state_tok  # [BLOCK_N]
                col2 = tl.load(conv_states_ptrs, mask_w, 0.0).to(x_elem_ty)
                conv_states_ptrs = prior_tokens - 2 * stride_conv_state_tok  # [BLOCK_N]
                col1 = tl.load(conv_states_ptrs, mask_w, 0.0).to(x_elem_ty)
                conv_states_ptrs = prior_tokens - 3 * stride_conv_state_tok  # [BLOCK_N]
                col0 = tl.load(conv_states_ptrs, mask_w, 0.0).to(x_elem_ty)
        else:
            # prior-tokens are zeros (same x dtype as every other col* source)
            if KERNEL_WIDTH >= 2:  # STRATEGY1
                # first chunk and does not have prior-token, so just set to 0
                col0 = tl.zeros((BLOCK_N,), dtype=x_ptr.dtype.element_ty)
            if KERNEL_WIDTH >= 3:  # STRATEGY1
                col1 = tl.zeros((BLOCK_N,), dtype=x_ptr.dtype.element_ty)
            if KERNEL_WIDTH >= 4:  # STRATEGY1
                col2 = tl.zeros((BLOCK_N,), dtype=x_ptr.dtype.element_ty)
            if KERNEL_WIDTH >= 5:  # STRATEGY1
                col3 = tl.zeros((BLOCK_N,), dtype=x_ptr.dtype.element_ty)

        # STEP 2:
        # here prepare data for updating conv_state
        if (
            state_len <= seqlen
        ):  # SMALL_CACHE=True (only move part of 'x' into conv_state cache)
            # just read from 'x'
            # copy 'x' data to conv_state
            # load only 'x' data (and set 0 before 'x' if seqlen < state_len)
            idx_tokens_last = (seqlen - state_len) + tl.arange(
                0, NP2_STATELEN
            )  # [BLOCK_M]
            x_ptrs = (
                x_ptr
                + (
                    (sequence_start_index + idx_tokens_last).to(tl.int64)
                    * stride_x_token
                )[:, None]
                + (idx_feats * stride_x_dim)[None, :]
            )  # [BLOCK_M,BLOCK_N,]
            mask_x = (
                (idx_tokens_last >= 0)[:, None]
                & (idx_tokens_last < seqlen)[:, None]
                & (idx_feats < dim)[None, :]
            )  # token-index  # token-index  # feature-index
            loaded_x = tl.load(x_ptrs, mask_x, 0.0)
            new_conv_state = tl.load(x_ptrs, mask_x, 0.0)
            idx_tokens_conv = tl.arange(0, NP2_STATELEN)  # [BLOCK_M]
            conv_states_ptrs_target = (
                conv_states_base[None, :]
                + (idx_tokens_conv * stride_conv_state_tok)[:, None]
            )

            mask = (idx_tokens_conv < state_len)[:, None] & (idx_feats < dim)[None, :]
            tl.debug_barrier()  #  NOTE: use this due to bug in Triton compiler
            tl.store(conv_states_ptrs_target, new_conv_state, mask)

        else:
            if load_init_state:
                # update conv_state by shifting left, i.e. take last few cols from conv_state + cols from 'x'
                idx_tokens_conv = tl.arange(0, NP2_STATELEN)  # [BLOCK_M]

                conv_states_ptrs_source = (
                    conv_states_ptr
                    + (conv_state_batch_coord * stride_conv_state_seq)
                    + (idx_feats * stride_conv_state_dim)[None, :]
                    + ((idx_tokens_conv + seqlen) * stride_conv_state_tok)[:, None]
                )  # [BLOCK_M, BLOCK_N]
                mask = (
                    (conv_state_batch_coord < num_cache_lines)
                    & ((idx_tokens_conv + seqlen) < state_len)[:, None]
                    & (idx_feats < dim)[None, :]
                )
                conv_state = tl.load(conv_states_ptrs_source, mask, other=0.0)

                VAL = state_len - seqlen

                x_ptrs = (
                    x_base[None, :]
                    + ((idx_tokens_conv - VAL) * stride_x_token)[:, None]
                )  # [BLOCK_M, BLOCK_N]

                mask_x = (
                    (idx_tokens_conv - VAL >= 0)[:, None]
                    & (idx_tokens_conv - VAL < seqlen)[:, None]
                    & (idx_feats < dim)[None, :]
                )  # token-index  # token-index  # feature-index
                loaded_x = tl.load(x_ptrs, mask_x, 0.0)

                tl.debug_barrier()  # need this due to the bug in tl.where not enforcing this when data is the result of another tl.load
                new_conv_state = tl.where(
                    mask, conv_state, loaded_x
                )  # BUG in 'tl.where'  which requires a barrier before this
                conv_states_ptrs_target = (
                    conv_states_base
                    + (idx_tokens_conv * stride_conv_state_tok)[:, None]
                )  # [BLOCK_M, BLOCK_N]
                mask = (idx_tokens_conv < state_len)[:, None] & (idx_feats < dim)[
                    None, :
                ]
                tl.store(conv_states_ptrs_target, new_conv_state, mask)
            else:  # load_init_state == False
                # update conv_state by shifting left, BUT
                # set cols prior to 'x' as zeros + cols from 'x'
                idx_tokens_conv = tl.arange(0, NP2_STATELEN)  # [BLOCK_M]

                VAL = state_len - seqlen

                x_ptrs = (
                    x_base[None, :]
                    + ((idx_tokens_conv - VAL) * stride_x_token)[:, None]
                )  # [BLOCK_M, BLOCK_N]

                mask_x = (
                    (idx_tokens_conv - VAL >= 0)[:, None]
                    & (idx_tokens_conv - VAL < seqlen)[:, None]
                    & (idx_feats < dim)[None, :]
                )  # token-index  # token-index  # feature-index
                new_conv_state = tl.load(x_ptrs, mask_x, 0.0)

                conv_states_ptrs_target = (
                    conv_states_base
                    + (idx_tokens_conv * stride_conv_state_tok)[:, None]
                )  # [BLOCK_M, BLOCK_N]
                mask = (idx_tokens_conv < state_len)[:, None] & (idx_feats < dim)[
                    None, :
                ]
                tl.store(conv_states_ptrs_target, new_conv_state, mask)

    else:  # chunk_offset > 0
        # read prior-token data from `x`
        load_init_state = True
        prior_tokens = x_base + (token_offset - 1).to(tl.int64) * stride_x_token
        mask_w = idx_feats < dim
        if KERNEL_WIDTH == 2:
            conv_states_ptrs = prior_tokens  # [BLOCK_N]
            col0 = tl.load(conv_states_ptrs, mask_w, 0.0, cache_modifier=".ca")
        if KERNEL_WIDTH == 3:
            conv_states_ptrs = prior_tokens  # [BLOCK_N]
            col1 = tl.load(conv_states_ptrs, mask_w, 0.0, cache_modifier=".ca")
            conv_states_ptrs = prior_tokens - 1 * stride_x_token  # [BLOCK_N]
            col0 = tl.load(conv_states_ptrs, mask_w, 0.0, cache_modifier=".ca")
        if KERNEL_WIDTH == 4:
            conv_states_ptrs = prior_tokens  # [BLOCK_N]
            col2 = tl.load(conv_states_ptrs, mask_w, 0.0, cache_modifier=".ca")
            conv_states_ptrs = prior_tokens - 1 * stride_x_token  # [BLOCK_N]
            col1 = tl.load(conv_states_ptrs, mask_w, 0.0, cache_modifier=".ca")
            conv_states_ptrs = prior_tokens - 2 * stride_x_token  # [BLOCK_N]
            col0 = tl.load(conv_states_ptrs, mask_w, 0.0, cache_modifier=".ca")
        if KERNEL_WIDTH == 5:
            # ruff: noqa: F841
            conv_states_ptrs = prior_tokens  # [BLOCK_N]
            col3 = tl.load(conv_states_ptrs, mask_w, 0.0, cache_modifier=".ca")
            conv_states_ptrs = prior_tokens - 1 * stride_x_token  # [BLOCK_N]
            col2 = tl.load(conv_states_ptrs, mask_w, 0.0, cache_modifier=".ca")
            conv_states_ptrs = prior_tokens - 2 * stride_x_token  # [BLOCK_N]
            col1 = tl.load(conv_states_ptrs, mask_w, 0.0, cache_modifier=".ca")
            conv_states_ptrs = prior_tokens - 3 * stride_x_token  # [BLOCK_N]
            col0 = tl.load(conv_states_ptrs, mask_w, 0.0, cache_modifier=".ca")

    if HAS_BIAS:
        bias = bias_ptr + idx_feats
        mask_bias = idx_feats < dim
        acc_preload = tl.load(bias, mask=mask_bias, other=0.0).to(
            tl.float32
        )  # [BLOCK_N]
    else:
        acc_preload = tl.zeros((BLOCK_N,), dtype=tl.float32)

    x_base_1d = x_base + token_offset.to(tl.int64) * stride_x_token  # starting of chunk

    # PRE-LOAD WEIGHTS
    mask_w = idx_feats < dim
    if KERNEL_WIDTH >= 2:
        w_ptrs = w_base + (0 * stride_w_width)  # [BLOCK_N] tensor
        w_col0 = tl.load(w_ptrs, mask_w, other=0.0)
        w_ptrs = w_base + (1 * stride_w_width)  # [BLOCK_N] tensor
        w_col1 = tl.load(w_ptrs, mask_w, other=0.0)
    if KERNEL_WIDTH >= 3:
        w_ptrs = w_base + (2 * stride_w_width)  # [BLOCK_N] tensor
        w_col2 = tl.load(w_ptrs, mask_w, other=0.0)
    if KERNEL_WIDTH >= 4:
        w_ptrs = w_base + (3 * stride_w_width)  # [BLOCK_N] tensor
        w_col3 = tl.load(w_ptrs, mask_w, other=0.0)
    mask_x_1d = idx_feats < dim
    if TILE_2D:
        feat0 = tl.program_id(2) * BLOCK_N
        in_k = feat0 >= SPLIT_D0
        in_v = feat0 >= SPLIT_D1
        col_base = tl.where(in_v, SPLIT_D1, tl.where(in_k, SPLIT_D0, 0))
        o_width = tl.where(
            in_v, dim - SPLIT_D1, tl.where(in_k, SPLIT_D1 - SPLIT_D0, SPLIT_D0)
        ).to(tl.int64)
        t_loc = tl.arange(0, BLOCK_M)
        tok = token_offset + t_loc  # token index inside the sequence
        x_rows = (
            x_ptr
            + sequence_start_index.to(tl.int64) * stride_x_token
            + (idx_feats * stride_x_dim)[None, :]
        )
        o_ptrs2 = (
            o_ptr
            + col_base.to(tl.int64) * total_tokens
            + (idx_feats - col_base)[None, :]
            + ((sequence_start_index + tok).to(tl.int64) * o_width)[:, None]
        )
        acc2 = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32) + acc_preload[None, :]
        if (chunk_offset > 0) & (segment_len == BLOCK_M) & EVEN_FEATS:
            # Interior chunk: every tap row is a real token of this sequence.
            for j in tl.static_range(KERNEL_WIDTH):
                xv = tl.load(
                    x_rows
                    + ((tok - (KERNEL_WIDTH - 1 - j)).to(tl.int64) * stride_x_token)[
                        :, None
                    ]
                )
                if j == 0:
                    w2 = w_col0
                elif j == 1:
                    w2 = w_col1
                elif j == 2:
                    w2 = w_col2
                else:
                    w2 = w_col3
                acc2 += xv * w2[None, :]
            if SILU_ACTIVATION:
                acc2 = acc2 / (1 + tl.exp(-acc2))
            tl.store(o_ptrs2, acc2)
        else:
            row_ok = t_loc < segment_len
            for j in tl.static_range(KERNEL_WIDTH):
                src = tok - (KERNEL_WIDTH - 1 - j)
                m2 = (row_ok & (src >= 0))[:, None] & mask_x_1d[None, :]
                xv = tl.load(
                    x_rows + (src.to(tl.int64) * stride_x_token)[:, None],
                    mask=m2,
                    other=0.0,
                )
                # Taps before the sequence start take the prologue's col0..col2
                # (the conv state, or zeros without an initial state).
                if j <= 2:
                    xv = tl.where((src == -1)[:, None], col2[None, :], xv)
                if j <= 1:
                    xv = tl.where((src == -2)[:, None], col1[None, :], xv)
                if j == 0:
                    xv = tl.where((src == -3)[:, None], col0[None, :], xv)
                if j == 0:
                    w2 = w_col0
                elif j == 1:
                    w2 = w_col1
                elif j == 2:
                    w2 = w_col2
                else:
                    w2 = w_col3
                acc2 += xv * w2[None, :]
            if SILU_ACTIVATION:
                acc2 = acc2 / (1 + tl.exp(-acc2))
            tl.store(o_ptrs2, acc2, mask=row_ok[:, None] & mask_x_1d[None, :])
    else:
        for idx_token in range(segment_len):
            acc = acc_preload

            matrix_w = w_col0
            matrix_x = col0
            for j in tl.static_range(KERNEL_WIDTH):
                if KERNEL_WIDTH == 2:
                    if j == 1:  # KERNEL_WIDTH-1:
                        matrix_w = w_col1
                        x_ptrs_1d = x_base_1d + idx_token * stride_x_token  # [BLOCK_N]
                        matrix_x = tl.load(x_ptrs_1d, mask=mask_x_1d)
                elif KERNEL_WIDTH == 3:
                    if j == 1:
                        matrix_w = w_col1
                        matrix_x = col1
                    elif j == 2:
                        matrix_w = w_col2
                        x_ptrs_1d = x_base_1d + idx_token * stride_x_token  # [BLOCK_N]
                        matrix_x = tl.load(x_ptrs_1d, mask=mask_x_1d)
                elif KERNEL_WIDTH == 4:
                    if j == 1:
                        matrix_w = w_col1
                        matrix_x = col1
                    elif j == 2:
                        matrix_w = w_col2
                        matrix_x = col2
                    elif j == 3:
                        matrix_w = w_col3
                        x_ptrs_1d = x_base_1d + idx_token * stride_x_token  # [BLOCK_N]
                        matrix_x = tl.load(x_ptrs_1d, mask=mask_x_1d)

                acc += matrix_x * matrix_w  # [BLOCK_N]

            if KERNEL_WIDTH == 2:
                col0 = matrix_x
            elif KERNEL_WIDTH == 3:
                col0 = col1
                col1 = matrix_x
            elif KERNEL_WIDTH == 4:
                col0 = col1
                col1 = col2
                col2 = matrix_x

            if SILU_ACTIVATION:
                acc = acc / (1 + tl.exp(-acc))
            mask_1d = (idx_token < segment_len) & (
                idx_feats < dim
            )  # token-index  # feature-index
            output_token = (sequence_start_index + token_offset + idx_token).to(
                tl.int64
            )
            if OUTPUT_SPLIT_SIZE:
                # Store each channel group as a contiguous [tokens, channels] plane.
                # KDA can consume Q/K/V directly without three materialization copies.
                o_ptrs = (
                    o_ptr
                    + (idx_feats // OUTPUT_SPLIT_SIZE).to(tl.int64)
                    * total_tokens
                    * OUTPUT_SPLIT_SIZE
                    + output_token * OUTPUT_SPLIT_SIZE
                    + idx_feats % OUTPUT_SPLIT_SIZE
                )
            elif SPLIT_D0 > 0:
                feat0 = tl.program_id(2) * BLOCK_N
                in_k = feat0 >= SPLIT_D0
                in_v = feat0 >= SPLIT_D1
                col_base = tl.where(in_v, SPLIT_D1, tl.where(in_k, SPLIT_D0, 0))
                o_width = tl.where(
                    in_v, dim - SPLIT_D1, tl.where(in_k, SPLIT_D1 - SPLIT_D0, SPLIT_D0)
                ).to(tl.int64)
                o_ptrs = (
                    o_ptr
                    + col_base.to(tl.int64) * total_tokens
                    + output_token * o_width
                    + (idx_feats - col_base)
                )
            else:
                o_ptrs = (
                    o_ptr + output_token * stride_o_token + idx_feats * stride_o_dim
                )

            tl.store(o_ptrs, acc, mask=mask_1d)


def causal_conv1d_fn(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: Union[torch.Tensor, None],
    conv_states: torch.Tensor,
    query_start_loc: torch.Tensor,
    seq_lens_cpu: List[int],
    cache_indices: Optional[torch.Tensor] = None,
    has_initial_state: Optional[torch.Tensor] = None,
    activation: Optional[str] = "silu",
    pad_slot_id: int = PAD_SLOT_ID,
    validate_data=False,
    block_table: Optional[torch.Tensor] = None,
    output_split_size: int = 0,
    split_dims: Optional[tuple] = None,
    fwd_launch: Optional[dict] = None,
    **kwargs,
):
    """support varlen + continuous batching when x is 2D tensor

    x: (dim,cu_seq_len)
        cu_seq_len = total tokens of all seqs in that batch
        sequences are concatenated from left to right for varlen
    weight: (dim, width)
    conv_states: (...,dim,width - 1) itype
        updated inplace if provided
        [it use `cache_indices` to get the index to the cache of conv_state for that sequence

        conv_state[cache_indices[i]] for seq-i - to be used as initial_state when has_initial_state[i] = True
             and after that conv_state[cache_indices[i]] need to be shift-left and updated with values from 'x'
        ]
    query_start_loc: (batch + 1) int32
        The cumulative sequence lengths of the sequences in
        the batch, used to index into sequence. prepended by 0.
        if
        x = [5, 1, 1, 1] <- continuous batching (batch=4)
        then
        query_start_loc = [0, 5, 6, 7, 8] <- the starting index of the next sequence; while the last value is
           the ending index of the last sequence
        [length(query_start_loc)-1 == batch]
        for example: query_start_loc = torch.Tensor([0,10,16,17]),
        x.shape=(dim,17)
    seq_lens_cpu: (batch) int32
        The sequence lengths of the sequences in the batch
    cache_indices: (batch)  int32
        indicates the corresponding state index,
        like so: conv_state = conv_states[cache_indices[batch_id]]
    has_initial_state: (batch) bool
        indicates whether should the kernel take the current state as initial
        state for the calculations
        [single boolean for each sequence in the batch: True or False]
    bias: (dim,)
    activation: either None or "silu" or "swish" or True
    pad_slot_id: int
        if cache_indices is passed, lets the kernel identify padded
        entries that will not be processed,
        for example: cache_indices = [pad_slot_id, 1, 20, pad_slot_id]
        in this case, the kernel will not process entries at
        indices 0 and 3

    output_split_size: channel width of each independently contiguous output
        plane, or zero for the original layout. Must divide dim.
    out: same shape as `x`, or (dim // output_split_size, cu_seq_len,
        output_split_size) when output_split_size is nonzero.
    split_dims: (d0, d1) for a flat output of three token-major planes
        [cu_seq_len, d0] | [cu_seq_len, d1 - d0] | [cu_seq_len, dim - d1] (see
        causal_conv1d_fn_split_qkv); d0 and d1 must be multiples of 256.
    fwd_launch: the rectangular launch's tile (_parse_fwd_launch_config's dict),
        for widths 2-4 without a block table; None keeps the stock tile.
    """
    if isinstance(activation, bool) and activation:
        activation = "silu"

    is_channel_last = (x.stride(0) == 1) & (x.stride(1) > 1)
    dim, cu_seqlen = x.shape
    if split_dims is not None:
        # Flat [q | k | v] planes on the 256-wide split tile (_split_fwd_launch).
        if output_split_size or split_dims[0] % 256 or split_dims[1] % 256:
            raise ValueError(
                "split_dims must be multiples of 256, without a split size"
            )
        out = torch.empty(x.numel(), dtype=x.dtype, device=x.device)
    elif output_split_size:
        if output_split_size < 0 or dim % output_split_size:
            raise ValueError("output_split_size must be positive and divide dim")
        out = torch.empty(
            (dim // output_split_size, cu_seqlen, output_split_size),
            dtype=x.dtype,
            device=x.device,
        )
    else:
        out = torch.empty_like(x)
    _, width = weight.shape
    state_len = width - 1
    np2_statelen = triton.next_power_of_2(state_len)

    stride_x_seq = 0
    stride_x_dim = x.stride(0)
    stride_x_token = x.stride(1)
    stride_w_dim = weight.stride(0)
    stride_w_width = weight.stride(1)
    stride_istate_seq = 0
    stride_istate_dim = 0
    stride_istate_token = 0
    num_cache_lines = 0
    if conv_states is not None:
        # extensions to support vLLM:
        # 1. conv_states is used to replaced initial_states
        # 2. conv_states serve as a cache with num cache lines can be larger than batch size
        # 3. mapping from sequence x[idx] to a cache line at index as specified via cache_indices[idx]
        # 4. computation can be skipped if cache_indices[idx] == pad_slot_id
        num_cache_lines = conv_states.size(0)
        assert (
            num_cache_lines == conv_states.shape[0]
            and dim == conv_states.shape[1]
            and width - 1 <= conv_states.shape[2]
        )
        stride_istate_seq = conv_states.stride(0)
        stride_istate_dim = conv_states.stride(1)
        stride_istate_token = conv_states.stride(2)
        # assert stride_istate_dim == 1
    if split_dims is not None:
        # The kernel derives each plane's pitch from total_tokens.
        stride_o_seq = stride_o_dim = stride_o_token = 0
    elif output_split_size:
        # The plane pitch uses runtime total_tokens in the kernel; do not
        # specialize the launch on every ragged prefill's token count.
        stride_o_seq = 0
        stride_o_dim = 1
        stride_o_token = output_split_size
    elif out.dim() == 2:
        stride_o_seq = 0
        stride_o_dim = out.stride(0)
        stride_o_token = out.stride(1)
    else:
        stride_o_seq = out.stride(0)
        stride_o_dim = out.stride(1)
        stride_o_token = out.stride(2)

    if validate_data:
        assert x.dim() == 2
        assert query_start_loc is not None
        assert query_start_loc.dim() == 1
        assert x.stride(0) == 1 or x.stride(1) == 1
        padded_batch = query_start_loc.size(0) - 1
        if bias is not None:
            assert bias.dim() == 1
            assert dim == bias.size(0)
        if cache_indices is not None:
            assert cache_indices.dim() == 1
            assert padded_batch == cache_indices.size(0)
        if has_initial_state is not None:
            assert has_initial_state.size() == (padded_batch,)
            assert conv_states is not None, (
                "ERROR: `has_initial_state` is used, which needs also `conv_states`"
            )
        assert weight.stride(1) == 1
        assert (dim, width) == weight.shape
        assert is_channel_last, "Need to run in channel-last layout"

    def grid(META):
        if block_table is not None:
            # (num_blocks, 1, feature blocks): no host-side sequence lengths.
            return (block_table.shape[0], 1, triton.cdiv(dim, META["BLOCK_N"]))
        max_seq_len = max(seq_lens_cpu)
        return (
            len(seq_lens_cpu),  # batch_size
            (max_seq_len + META["BLOCK_M"] - 1) // META["BLOCK_M"],
            triton.cdiv(dim, META["BLOCK_N"]),
        )

    _causal_conv1d_fwd_kernel[grid](
        # Pointers to matrices
        x,
        weight,
        bias,
        conv_states,
        cache_indices,
        has_initial_state,
        query_start_loc,
        out,
        block_table,
        # Matrix dimensions
        dim,
        cu_seqlen,
        num_cache_lines,
        # stride
        stride_x_seq,
        stride_x_dim,
        stride_x_token,
        stride_w_dim,
        stride_w_width,
        stride_istate_seq,
        stride_istate_dim,
        stride_istate_token,
        stride_o_seq,
        stride_o_dim,
        stride_o_token,
        # others
        pad_slot_id,
        # META
        HAS_BIAS=bias is not None,
        KERNEL_WIDTH=width,
        SILU_ACTIVATION=activation in ["silu", "swish"],
        HAS_INITIAL_STATES=has_initial_state is not None,
        HAS_CACHE=conv_states is not None,
        IS_CONTINUOUS_BATCHING=cache_indices is not None,
        USE_PAD_SLOT=pad_slot_id is not None,
        NP2_STATELEN=np2_statelen,
        # launch_cooperative_grid=True
        USE_BLOCK_TABLE=block_table is not None,
        OUTPUT_SPLIT_SIZE=output_split_size,
        SPLIT_D0=split_dims[0] if split_dims is not None else 0,
        SPLIT_D1=split_dims[1] if split_dims is not None else 0,
        **(
            _split_fwd_launch(width, dim)
            if split_dims is not None
            else _fwd_launch_config(block_table, width, fwd_launch)
        ),
    )
    return out


class _PrefillConvStrip(NamedTuple):
    tokens: int
    threads: int
    unroll: int
    min_tokens: int


# Measured by the source on VR200 at 1K-16K prefill tokens; re-tune when the kernel changes.
_PREFILL_CONV_STRIP_DEFAULT = _PrefillConvStrip(
    tokens=32, threads=128, unroll=4, min_tokens=2048
)


def _parse_prefill_conv_strip(value: str) -> Optional[_PrefillConvStrip]:
    """SGLANG_GDN_PREFILL_CONV_RETILE (A's name): "" / "1" the default, "0" off, or
    "tokens,threads,unroll[,min_tokens]" for the token-strip kernel."""
    value = value.strip()
    if value == "0":
        return None
    if value in ("", "1"):
        return _PREFILL_CONV_STRIP_DEFAULT
    try:
        parts = [int(v) for v in value.split(",")]
    except ValueError:
        parts = []
    if len(parts) == 3:
        parts.append(_PREFILL_CONV_STRIP_DEFAULT.min_tokens)
    if not (
        len(parts) == 4
        and parts[0] in (8, 16, 32, 64, 128, 256)
        and parts[1] in (64, 128, 256)
        and parts[2] in (1, 2, 4, 8)
        and parts[0] % parts[2] == 0
        and parts[3] >= 1
    ):
        raise ValueError(
            "SGLANG_GDN_PREFILL_CONV_RETILE must be 0, 1 or "
            "'tokens,threads,unroll[,min_tokens]' (tokens 8-256 by powers of 2, "
            f"threads 64/128/256, unroll 1/2/4/8 dividing tokens), got {value!r}"
        )
    return _PrefillConvStrip(*parts)


_prefill_conv_strip_logged: set = set()


def _log_prefill_conv_strip_once(key: str, message: str, *args) -> None:
    if key not in _prefill_conv_strip_logged:
        _prefill_conv_strip_logged.add(key)
        logger.info(message, *args)


def warmup_prefill_conv_strip(cfg: _PrefillConvStrip) -> None:
    """Build the strip kernel before serving; raises if it does not build."""
    from sgl_nvidia_mega.qwen3_8.kernels.ops.mamba import (
        gdn_prefill_conv_strip as strip,
    )

    strip.warmup(cfg.threads, cfg.tokens, cfg.unroll)


def _prefill_conv_strip(
    *,
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: Optional[torch.Tensor],
    conv_states: Optional[torch.Tensor],
    query_start_loc: torch.Tensor,
    seq_lens_cpu: Optional[List[int]],
    q_dim: int,
    k_dim: int,
    kwargs: dict,
    cfg: _PrefillConvStrip,
) -> Optional[torch.Tensor]:
    """The strip kernel's flat [q | k | v] output, or None to keep the Triton launch."""
    from sgl_nvidia_mega.qwen3_8.kernels.ops.mamba import (
        gdn_prefill_conv_strip as strip,
    )

    total = x.shape[1]
    if total < cfg.min_tokens:
        return None
    if kwargs.get("block_table") is not None:
        _log_prefill_conv_strip_once(
            "table",
            "GDN prefill conv strip: Triton launch kept for a block-table launch "
            "(%d tokens)",
            total,
        )
        return None
    cache_indices = kwargs.get("cache_indices")
    has_initial_state = kwargs.get("has_initial_state")
    activation = kwargs.get("activation", "silu")
    if (
        not seq_lens_cpu
        or activation not in (None, "silu", "swish", True)
        or query_start_loc.dtype != torch.int32
        or (cache_indices is not None and cache_indices.dtype != torch.int32)
        or (has_initial_state is not None and has_initial_state.dtype != torch.bool)
        or (
            bias is not None
            and (bias.dtype != torch.bfloat16 or not bias.is_contiguous())
        )
        or not strip.supported(x, weight, conv_states, q_dim, k_dim)
    ):
        _log_prefill_conv_strip_once(
            "skip",
            "GDN prefill conv strip: Triton launch kept for an unsupported call "
            "(%d tokens)",
            total,
        )
        return None
    _log_prefill_conv_strip_once(
        "on",
        "GDN prefill conv strip engaged (SGLANG_OPT_GDN_PREFILL_CONV_STRIP): "
        "tokens=%d threads=%d unroll=%d min_tokens=%d, first tokens=%d seqs=%d",
        cfg.tokens,
        cfg.threads,
        cfg.unroll,
        cfg.min_tokens,
        total,
        len(seq_lens_cpu),
    )
    return strip.conv_split_qkv(
        x,
        weight,
        bias,
        conv_states,
        query_start_loc,
        max(seq_lens_cpu),
        q_dim,
        k_dim,
        cache_indices,
        has_initial_state,
        activation is not None,
        kwargs.get("pad_slot_id", PAD_SLOT_ID),
        cfg.threads,
        cfg.tokens,
        cfg.unroll,
    )


def causal_conv1d_fn_split_qkv(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: Union[torch.Tensor, None],
    conv_states: torch.Tensor,
    query_start_loc: torch.Tensor,
    seq_lens_cpu: Optional[List[int]],
    q_dim: int,
    k_dim: int,
    strip_cfg: Optional[_PrefillConvStrip] = None,
    **kwargs,
):
    """causal_conv1d_fn whose output is three contiguous token-major tensors
    q [T, q_dim], k [T, k_dim], v [T, dim - q_dim - k_dim]: bit-identical values to
    causal_conv1d_fn(...).transpose(0, 1) split along the channel axis. strip_cfg:
    the token-strip kernel's tile for calls it serves, None for the Triton launch."""
    dim, total = x.shape
    out = None
    if strip_cfg is not None:
        out = _prefill_conv_strip(
            x=x,
            weight=weight,
            bias=bias,
            conv_states=conv_states,
            query_start_loc=query_start_loc,
            seq_lens_cpu=seq_lens_cpu,
            q_dim=q_dim,
            k_dim=k_dim,
            kwargs=kwargs,
            cfg=strip_cfg,
        )
    if out is None:
        out = causal_conv1d_fn(
            x,
            weight,
            bias,
            conv_states=conv_states,
            query_start_loc=query_start_loc,
            seq_lens_cpu=seq_lens_cpu,
            split_dims=(q_dim, q_dim + k_dim),
            **kwargs,
        )
    qk_end = total * (q_dim + k_dim)
    q = out[: total * q_dim].view(total, q_dim)
    k = out[total * q_dim : qk_end].view(total, k_dim)
    v = out[qk_end:].view(total, dim - q_dim - k_dim)
    return q, k, v
