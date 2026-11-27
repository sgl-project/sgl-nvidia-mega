# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# Adapted from SGLang (Apache-2.0, SGLang Team):
# python/sglang/kernels/ops/elementwise/elementwise.py (_fused_sigmoid_mul_kernel,
# fused_sigmoid_mul) and
# python/sglang/kernels/ops/quantization/fp8_kernel.py (_static_quant_fp8).
"""The attention output gate and ``o_proj``'s static FP8 input quantization in one Triton
launch: the bf16 ``attn * sigmoid(gate)`` of SGLang's ``fused_sigmoid_mul``, then
``static_quant_fp8``'s ``clamp(y * (1 / scale))``, bit for bit the two launches.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

SGLANG_READS = {
    "kernels/jit/utils/__init__.py": ("is_arch_support_pdl",),
}

SGLANG_MIRRORS = {
    "kernels/ops/elementwise/elementwise.py:_fused_sigmoid_mul_kernel": "a00af816bd364384",
    "kernels/ops/elementwise/elementwise.py:fused_sigmoid_mul": "bf3c570a79e580a8",
}


@triton.jit
def _gate_quant_kernel(
    attn_ptr,
    gate_ptr,
    scale_ptr,
    out_ptr,
    hidden: tl.constexpr,
    BLOCK: tl.constexpr,
    STRIDED: tl.constexpr = False,
    GATE_ROW: tl.constexpr = 0,
    GATE_HEAD: tl.constexpr = 0,
    HEAD_DIM: tl.constexpr = 0,
    USE_PDL: tl.constexpr = True,
    # release the next launch before the store (False: SGLang's wait-only PDL)
    EARLY_TRIGGER: tl.constexpr = True,
):
    row = tl.program_id(0).to(tl.int64)
    cols = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    mask = cols < hidden
    if USE_PDL:
        tl.extra.cuda.gdc_wait()
    attn = tl.load(attn_ptr + row * hidden + cols, mask=mask, other=0.0).to(tl.float32)
    if STRIDED:
        head = cols // HEAD_DIM
        g_off = row * GATE_ROW + head * GATE_HEAD + (cols - head * HEAD_DIM)
        g = tl.load(gate_ptr + g_off, mask=mask, other=0.0).to(tl.float32)
    else:
        g = tl.load(gate_ptr + row * hidden + cols, mask=mask, other=0.0).to(tl.float32)
    y = (attn * tl.sigmoid(g)).to(tl.bfloat16).to(tl.float32)
    s_inv = 1.0 / tl.load(scale_ptr).to(tl.float32)
    if USE_PDL and EARLY_TRIGGER:
        tl.extra.cuda.gdc_launch_dependents()
    q = tl.clamp(y * s_inv, -448.0, 448.0).to(tl.float8e4nv)
    tl.store(out_ptr + row * hidden + cols, q, mask=mask)


def gate_quant(
    attn: torch.Tensor, gate: torch.Tensor, scale: torch.Tensor
) -> torch.Tensor:
    """FP8 [rows, hidden] of the gated attention rows ``attn`` [rows, hidden] (bf16, gate
    the same shape) at the static per-tensor ``scale``."""
    rows, hidden = attn.shape
    out = torch.empty_like(attn, dtype=torch.float8_e4m3fn)
    block = 1024
    _gate_quant_kernel[(rows, triton.cdiv(hidden, block))](
        attn, gate, scale, out, hidden, BLOCK=block, num_warps=4, launch_pdl=True
    )
    return out


def supported(attn_output: torch.Tensor, gate: torch.Tensor) -> bool:
    if (
        not attn_output.is_cuda
        or attn_output.dtype != torch.bfloat16
        or attn_output.dim() != 2
        or not attn_output.is_contiguous()
        or gate.dtype != torch.bfloat16
        or gate.stride(-1) != 1
    ):
        return False
    if gate.dim() == 3:
        return (
            gate.shape[0] == attn_output.shape[0]
            and gate.shape[1] * gate.shape[2] == attn_output.shape[1]
        )
    return gate.dim() == 2 and gate.shape == attn_output.shape


def fused_sigmoid_mul_fp8(
    attn_output: torch.Tensor, gate: torch.Tensor, scale: torch.Tensor
) -> torch.Tensor:
    from sglang.kernels.jit.utils import is_arch_support_pdl

    if not supported(attn_output, gate):
        raise ValueError("fused_sigmoid_mul_fp8: unsupported layout")
    if scale.numel() != 1 or scale.dtype != torch.float32:
        raise ValueError("fused_sigmoid_mul_fp8: a 1-element float32 static scale")
    if gate.dim() == 3:
        # the strided gate: [tokens, heads, head_dim]
        num_tokens, num_heads, head_dim = gate.shape
        hidden_dim = num_heads * head_dim
        gate_stride_row, gate_stride_head = gate.stride(0), gate.stride(1)
    else:
        # the flat gate: a [tokens, hidden] view, maybe of a wider row
        num_tokens, hidden_dim = gate.shape
        head_dim = hidden_dim
        gate_stride_row, gate_stride_head = gate.stride(0), hidden_dim
    q = torch.empty_like(attn_output, dtype=torch.float8_e4m3fn)
    # SGLang's launch: one program per row and hidden block
    block_h = 1024 if num_tokens < 1024 else 2048
    use_pdl = is_arch_support_pdl()
    pdl_kwargs = {"launch_pdl": True} if use_pdl else {}
    _gate_quant_kernel[(num_tokens, triton.cdiv(hidden_dim, block_h))](
        attn_output,
        gate,
        scale,
        q,
        hidden_dim,
        BLOCK=block_h,
        STRIDED=True,
        GATE_ROW=gate_stride_row,
        GATE_HEAD=gate_stride_head,
        HEAD_DIM=head_dim,
        USE_PDL=use_pdl,
        EARLY_TRIGGER=False,
        num_warps=4,
        **pdl_kwargs,
    )
    return q
