# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""K3.1 DSA index query, norm-folded wide form (K31MK_DSA_QFN; indexer.IndexProjection.project_query on 2+ rows).

The stock wide chain is rmsnorm(q_a) -> torch.mm(xn, wq_b^T) -> indexer_q_quant_head_gate. QFN drops the RMSNorm launch:
the GEMM reads the RAW q_a columns against wq_bg = bf16(wq_b * w_qa) (the fold QF-wide already builds, index_qfw.folded_query,
one copy per DSA layer), and this kernel applies the row's rstd before the stock quantization and head-gate arithmetic:

    acc   = bf16(x @ wq_bg^T)                        (torch.mm, BF16 out; x = projected[:, :1536] raw q_a)
    rstd  = rsqrt(sum(x^2) / 1536 + eps)             (FP32, this kernel; one program per row)
    q     = bf16(rstd * acc)
    scale, q8, w: indexer_q_quant_head_gate's arithmetic on q, unchanged (amax over 128, 1e-4 floor, /448, ue8m0 when
                  ROUND_SCALE; e4m3 store of the clamped quotient; float(bf16(gate * 32^-0.5)) * scale * 128^-0.5).

Numerics-changing vs stock (stock rounds xn = bf16(x rstd w_qa) before the product): gated by an FP64 reference
(test_k31_index_qfn) and an in-boot KL arm. No cross-CTA or cross-stream synchronization.

``K31MK_DSA_QFN_CUDA=1`` (default 0 = off; dsa/idx_qfn8.py): qfn_quant_head_gate's outputs from one CUDA kernel per row
(idx_qfn8.cuh, PDL), BITWISE equal to this Triton kernel (test_k31_idx_qfn8_cuda.py; the Triton kernel stays the reference and the
fallback). Engages for the served arguments only (softmax scale 128^-0.5, 32 heads)."""

import os

import torch
import triton
import triton.language as tl

QFN_CUDA = os.environ.get("K31MK_DSA_QFN_CUDA", "0") == "1"


@triton.jit
def _qfn_quant_head_gate_kernel(
    ACC,
    X,
    G,
    Q8,
    W,
    X_STRIDE,
    GATE_STRIDE,
    EPS,
    LORA: tl.constexpr,
    HEADS: tl.constexpr,
    GATE_SCALE: tl.constexpr,
    SOFTMAX_SCALE: tl.constexpr,
    ROUND_SCALE: tl.constexpr,
    BLOCK_X: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    xo = tl.arange(0, BLOCK_X)
    x = tl.load(X + row * X_STRIDE + xo, xo < LORA, other=0).to(tl.float32)
    rstd = tl.rsqrt(tl.sum(x * x, 0) / LORA + EPS)
    heads = tl.arange(0, HEADS)
    cols = tl.arange(0, 128)
    offs = row * (HEADS * 128) + heads[:, None] * 128 + cols[None, :]
    q = (tl.load(ACC + offs).to(tl.float32) * rstd).to(tl.bfloat16).to(tl.float32)
    # From here on: indexer_q_quant_head_gate's arithmetic (head_gate.py), one row of HEADS heads per program.
    amax = tl.maximum(tl.max(tl.abs(q), 1), 1e-4)
    scale = amax * (1.0 / 448.0)
    if ROUND_SCALE:
        scale = tl.exp2(tl.ceil(tl.log2(scale)))
    quant = tl.minimum(tl.maximum(q / scale[:, None], -448.0), 448.0)
    tl.store(Q8 + offs, quant)
    gate = tl.load(G + row * GATE_STRIDE + heads).to(tl.float32)
    gate = (gate * GATE_SCALE).to(tl.bfloat16).to(tl.float32)
    tl.store(W + row * HEADS + heads, gate * scale * SOFTMAX_SCALE)


def qfn_quant_head_gate(acc, projected, gate, eps, softmax_scale, scale_fmt, out=None):
    """(q8 [rows, 32, 128] FP8 e4m3, head weights [rows, 32, 1] FP32) from ``acc`` = bf16(x @ wq_bg^T) [rows, 32, 128]
    (contiguous), the raw q_a in ``projected[:, :1536]`` (row stride free) and the BF16 head gate [rows, 32]. ``out``: optional
    (q8, weights) contiguous tensors of those shapes to write instead of new ones (K31MK_DSA_REQ_SHARD_OWNQ writes the owner rows
    into row slices of fixed buffers); returned as given."""
    assert acc.is_cuda and acc.dtype == torch.bfloat16 and acc.is_contiguous() and acc.shape[-1] == 128
    rows, heads = acc.shape[0], acc.shape[1]
    assert projected.dtype == torch.bfloat16 and projected.shape[0] == rows and projected.shape[1] >= 1536
    assert projected.stride(1) == 1 and gate.shape == (rows, heads) and gate.stride(1) == 1 and gate.dtype == torch.bfloat16
    if out is None:
        q8 = torch.empty_like(acc, dtype=torch.float8_e4m3fn)
        weights = torch.empty((rows, heads, 1), device=acc.device, dtype=torch.float32)
    else:
        q8, weights = out
        assert q8.shape == acc.shape and q8.dtype == torch.float8_e4m3fn and q8.is_contiguous() and q8.device == acc.device
        assert weights.shape == (rows, heads, 1) and weights.dtype == torch.float32 and weights.is_contiguous()
    if rows and QFN_CUDA and heads == 32 and softmax_scale == 128**-0.5:
        from .idx_qfn8 import qfn8

        if qfn8(acc, projected, gate, q8, weights, float(eps), scale_fmt is not None):
            return q8, weights
    if rows:
        _qfn_quant_head_gate_kernel[(rows,)](
            acc,
            projected,
            gate,
            q8,
            weights,
            projected.stride(0),
            gate.stride(0),
            float(eps),
            1536,
            heads,
            heads**-0.5,
            softmax_scale,
            scale_fmt is not None,
            2048,
            num_warps=8,
            num_stages=0 if scale_fmt is not None else 2,  # as indexer_q_quant_head_gate launches
        )
    return q8, weights
