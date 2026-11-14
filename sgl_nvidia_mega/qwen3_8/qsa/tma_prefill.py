# SPDX-License-Identifier: Apache-2.0
"""TMA gather kernel: original online-softmax tile and operation order."""

import torch
import triton
import triton.language as tl

_HEAD_DIM = 256


@triton.jit(do_not_specialize=["oob_row"])
def _chunk_prefill_tma(
    q,
    k_desc,
    v_desc,
    out,
    indices,
    cu_q,
    cu_k,
    kv_lens,
    scale,
    topk,
    oob_row,
    sq_m: tl.constexpr,
    sq_h: tl.constexpr,
    so_m: tl.constexpr,
    so_h: tl.constexpr,
    si_m: tl.constexpr,
    NUM_KV_HEADS: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    STAGES: tl.constexpr,
):
    """``_sparse_gqa_chunk_prefill`` with TMA-gathered K/V.

    ``k_desc``/``v_desc`` view the packed K/V as [rows, kv_heads * head_dim] with
    block [1, head_dim]; ``indices`` hold positions relative to row ``cu_k[batch]``.
    Unselected entries read row ``oob_row``, one past the descriptor, which TMA
    zero-fills: the zeros the stock kernel's masked loads produce."""
    query_relative = tl.program_id(0).to(tl.int64)
    batch_group = tl.program_id(1)
    group = batch_group % NUM_KV_HEADS
    batch = batch_group // NUM_KV_HEADS
    q_start = tl.load(cu_q + batch)
    q_end = tl.load(cu_q + batch + 1)
    query = (q_start + query_relative).to(tl.int64)
    if query >= q_end:
        return
    # TMA gather row offsets are int32; the caller's cu_k is an int64 cumsum.
    k_start = tl.load(cu_k + batch).to(tl.int32)
    kv_len = tl.load(kv_lens + batch).to(tl.int64)
    visible = query_relative + kv_len - (q_end - q_start) + 1
    row_topk = tl.minimum(topk, visible)
    row_limit = tl.minimum(topk, ((row_topk + BLOCK_N - 1) // BLOCK_N) * BLOCK_N)
    offs_h = tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, HEAD_DIM)
    q_values = tl.load(
        q
        + query * sq_m
        + (group * GROUP_SIZE + offs_h[:, None]) * sq_h
        + offs_d[None, :],
        mask=(offs_h < GROUP_SIZE)[:, None],
        other=0.0,
    )
    q_values = (q_values * scale * 1.4426950408).to(q_values.dtype)
    idx_row = indices + query * si_m
    head_col = (group * HEAD_DIM).to(tl.int32)
    max_value = tl.full([BLOCK_M], -float("inf"), tl.float32)
    normalizer = tl.zeros([BLOCK_M], tl.float32)
    accumulator = tl.zeros([BLOCK_M, HEAD_DIM], tl.float32)
    offs_n = tl.arange(0, BLOCK_N)
    for start in tl.range(0, row_limit, BLOCK_N, num_stages=STAGES):
        current = start + offs_n
        token = tl.load(idx_row + current, mask=current < topk, other=-1)
        valid = token >= 0
        rows = tl.where(valid, k_start + token, oob_row)
        keys = tl.trans(k_desc.gather(rows, head_col).to(q_values.dtype))
        values = v_desc.gather(rows, head_col).to(q_values.dtype)
        scores = tl.where(valid[None, :], tl.dot(q_values, keys), -float("inf"))
        next_max = tl.maximum(max_value, tl.max(scores, 1))
        alpha = tl.math.exp2(max_value - next_max)
        probabilities = tl.math.exp2(scores - next_max[:, None])
        accumulator = tl.dot(
            probabilities.to(values.dtype), values, accumulator * alpha[:, None]
        )
        normalizer = normalizer * alpha + tl.sum(probabilities, 1)
        max_value = next_max
    output = accumulator / normalizer[:, None]
    tl.store(
        out
        + query * so_m
        + (group * GROUP_SIZE + offs_h[:, None]) * so_h
        + offs_d[None, :],
        output,
        mask=(offs_h < GROUP_SIZE)[:, None],
    )


def _descriptor(kv: torch.Tensor):
    from triton.tools.tensor_descriptor import TensorDescriptor

    return TensorDescriptor.from_tensor(kv.view(kv.shape[0], -1), [1, _HEAD_DIM])


def launch(io, *, max_q, stages):
    q, k, v, out = io.query, io.packed_k, io.packed_v, io.output
    n = io.cu_query.shape[0] - 1
    kv_heads = k.shape[1]
    _chunk_prefill_tma[(max_q, n * kv_heads)](
        q,
        _descriptor(k),
        _descriptor(v),
        out,
        io.selected_indices,
        io.cu_query,
        io.cu_key,
        io.sequence_lengths,
        io.scale,
        io.selected_indices.shape[-1],
        k.shape[0],
        q.stride(0),
        q.stride(1),
        out.stride(0),
        out.stride(1),
        io.selected_indices.stride(0),
        NUM_KV_HEADS=kv_heads,
        GROUP_SIZE=q.shape[1] // kv_heads,
        BLOCK_M=16,
        BLOCK_N=16,
        HEAD_DIM=256,
        STAGES=stages,
        num_warps=1,
    )
