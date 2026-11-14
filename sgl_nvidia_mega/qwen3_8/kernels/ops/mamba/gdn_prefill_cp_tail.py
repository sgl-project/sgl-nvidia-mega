# SPDX-License-Identifier: Apache-2.0
# From SGLang python/sglang/kernels/ops/mamba/gdn_prefill_track.py
# (_gdn_cp_tail_kernel, gdn_cp_tail_neutralize); unchanged.
"""The rows a captured FlashInfer GDN prefill reads past a batch's last token.

In a breakable prefill CUDA graph the q / k / v / alpha / beta inputs have the token
bucket's rows. The rows after the batch's last token up to its 64-token block are
bucket padding (stale graph-pool memory, possibly not finite) where an exact-size
eager call would see FlashInfer's TMA fill past the tensor end.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _gdn_cp_tail_kernel(
    q,  # [tokens, H_QK, D]
    k,  # [tokens, H_QK, D]
    v,  # [tokens, H_V, D]
    alpha,  # [tokens, H_V] fp32
    beta,  # [tokens, H_V] fp32
    cu_seqlens,  # [2] int64, the one-sequence batch's [0, T]
    num_rows,
    stride_q,
    stride_k,
    stride_v,
    stride_alpha,
    stride_beta,
    QK: tl.constexpr,
    V: tl.constexpr,
    H: tl.constexpr,
    BLOCK_QK: tl.constexpr,
    BLOCK_V: tl.constexpr,
    BLOCK_H: tl.constexpr,
):
    end = tl.load(cu_seqlens + 1)
    row = end + tl.program_id(0)
    if row >= tl.minimum((end + 63) // 64 * 64, num_rows):
        return
    row = row.to(tl.int64)
    offs = tl.arange(0, BLOCK_QK)
    mask = offs < QK
    tl.store(
        q + row * stride_q + offs, tl.zeros([BLOCK_QK], q.dtype.element_ty), mask=mask
    )
    tl.store(
        k + row * stride_k + offs, tl.zeros([BLOCK_QK], k.dtype.element_ty), mask=mask
    )
    offs = tl.arange(0, BLOCK_V)
    tl.store(
        v + row * stride_v + offs,
        tl.zeros([BLOCK_V], v.dtype.element_ty),
        mask=offs < V,
    )
    offs = tl.arange(0, BLOCK_H)
    mask = offs < H
    tl.store(
        alpha + row * stride_alpha + offs,
        tl.full([BLOCK_H], 1.0, alpha.dtype.element_ty),
        mask=mask,
    )
    tl.store(
        beta + row * stride_beta + offs,
        tl.zeros([BLOCK_H], beta.dtype.element_ty),
        mask=mask,
    )


def gdn_cp_tail_neutralize(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    alpha: torch.Tensor,
    beta: torch.Tensor,
    cu_seqlens: torch.Tensor,
) -> None:
    """Rows [T, ceil64(T)) of a one-sequence batch as an exact-size eager call sees
    them: q / k / v zero (TMA fill past the tensor end), alpha 1, beta 0. FlashInfer's
    CP state kernel loads whole 64-token K / V blocks; in a graph those rows are
    bucket padding, i.e. stale graph-pool memory that may not be finite."""
    rows = q.shape[0]
    assert q[0].is_contiguous() and k[0].is_contiguous() and v[0].is_contiguous()
    assert alpha.stride(-1) == 1 and beta.stride(-1) == 1
    qk, vn, h = q[0].numel(), v[0].numel(), alpha.shape[-1]
    _gdn_cp_tail_kernel[(64,)](
        q,
        k,
        v,
        alpha,
        beta,
        cu_seqlens,
        rows,
        q.stride(0),
        k.stride(0),
        v.stride(0),
        alpha.stride(0),
        beta.stride(0),
        QK=qk,
        V=vn,
        H=h,
        BLOCK_QK=triton.next_power_of_2(qk),
        BLOCK_V=triton.next_power_of_2(vn),
        BLOCK_H=triton.next_power_of_2(h),
    )
