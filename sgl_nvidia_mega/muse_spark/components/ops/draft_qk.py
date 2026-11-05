# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Draft QK RMSNorm + table RoPE that also writes the query heads contiguously.

SGLang's ``table_qk_norm_rope_`` (``sglang/srt/speculative/dflash_utils.py``) normalizes and rotates the query and key
heads of the fused QKV tensor in place; the attention backend then copies the (strided) query view into a contiguous
tensor. The kernel below performs the same per-(token, head) arithmetic, in the same order, and additionally stores the
rotated query heads into a contiguous ``[tokens, q_heads, head_dim]`` tensor, so the copy becomes a no-op.
"""

# The Triton kernel below is derived from SGLang (https://github.com/sgl-project/sglang), Apache License 2.0; see
# ../include/LICENSE.sglang. Source: SGLang python/sglang/srt/speculative/dflash_utils.py
# (_table_qk_norm_rope_kernel) at revision a55ab7d.
# Modified by NVIDIA: the rotated query heads are also stored into a contiguous output tensor; the arithmetic is
# unchanged.

from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _qk_norm_rope_qout_kernel(
    qkv_ptr,
    q_out_ptr,
    q_weight_ptr,
    k_weight_ptr,
    cos_sin_ptr,
    pos_ptr,
    row_stride,
    q_size,
    NHQ: tl.constexpr,
    D: tl.constexpr,
    EPS: tl.constexpr,
):
    t = tl.program_id(0).to(tl.int64)
    h = tl.program_id(1)
    pos = tl.load(pos_ptr + t).to(tl.int64)

    HALF: tl.constexpr = D // 2
    half_ar = tl.arange(0, HALF)
    d_ar = tl.arange(0, D)
    cos = tl.load(cos_sin_ptr + pos * D + half_ar).to(tl.float32)
    sin = tl.load(cos_sin_ptr + pos * D + HALF + half_ar).to(tl.float32)

    is_q = h < NHQ
    col0 = tl.where(is_q, h * D, q_size + (h - NHQ) * D).to(tl.int64)
    w_ptr = tl.where(is_q, q_weight_ptr.to(tl.int64), k_weight_ptr.to(tl.int64)).to(
        tl.pointer_type(tl.bfloat16)
    )

    row = qkv_ptr + t * row_stride + col0
    x = tl.load(row + d_ar).to(tl.float32)
    ms = tl.sum(x * x, 0) / D
    inv = 1.0 / tl.sqrt(ms + EPS)
    w1 = tl.load(w_ptr + half_ar).to(tl.float32)
    w2 = tl.load(w_ptr + HALF + half_ar).to(tl.float32)
    x1 = tl.load(row + half_ar).to(tl.float32) * inv * w1
    x2 = tl.load(row + HALF + half_ar).to(tl.float32) * inv * w2
    x1 = x1.to(tl.bfloat16).to(tl.float32)
    x2 = x2.to(tl.bfloat16).to(tl.float32)
    o1 = x1 * cos - x2 * sin
    o2 = x2 * cos + x1 * sin
    tl.store(row + half_ar, o1.to(tl.bfloat16))
    tl.store(row + HALF + half_ar, o2.to(tl.bfloat16))
    if is_q:
        qrow = q_out_ptr + (t * NHQ + h) * D
        tl.store(qrow + half_ar, o1.to(tl.bfloat16))
        tl.store(qrow + HALF + half_ar, o2.to(tl.bfloat16))


def qk_norm_rope_qout(
    qkv,
    positions,
    q_weight,
    k_weight,
    cos_sin_cache,
    num_q_heads,
    num_k_heads,
    head_dim,
    eps,
    q_out=None,
):
    """``table_qk_norm_rope_`` (in place on ``qkv``) plus the rotated query heads as a contiguous tensor."""
    t = qkv.shape[0]
    if q_out is None:
        q_out = torch.empty(
            (t, num_q_heads, head_dim), dtype=qkv.dtype, device=qkv.device
        )
    if t == 0:
        return q_out
    grid = (t, num_q_heads + num_k_heads)
    _qk_norm_rope_qout_kernel[grid](
        qkv,
        q_out,
        q_weight,
        k_weight,
        cos_sin_cache,
        positions,
        qkv.stride(0),
        num_q_heads * head_dim,
        NHQ=num_q_heads,
        D=head_dim,
        EPS=eps,
    )
    return q_out
