# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

from __future__ import annotations

import torch
import triton
import triton.language as tl

_BLOCK = 1024


@triton.jit
def _copy_rows_kernel(
    index_ptr,
    index_stride,
    src_ptr,
    out_ptr,
    hidden,
    src_stride,
    src_rows,
    BLOCK: tl.constexpr,
    PLUS_ZERO: tl.constexpr = False,
):
    row = tl.program_id(0).to(tl.int64)
    part = tl.program_id(1)
    src_row = tl.load(index_ptr + row * index_stride).to(tl.int64)
    # an index outside [0, src_rows) reads nothing and writes a zero row
    valid = (src_row >= 0) & (src_row < src_rows)
    src_row = tl.where(valid, src_row, 0)
    cols = part * BLOCK + tl.arange(0, BLOCK)
    mask = cols < hidden
    bits = tl.load(src_ptr + src_row * src_stride + cols, mask=mask & valid, other=0)
    if PLUS_ZERO:
        bits = tl.where(bits == -32768, 0, bits)
    tl.store(out_ptr + row * hidden + cols, bits, mask=mask)


def copy_rows(
    src: torch.Tensor, index: torch.Tensor, out: torch.Tensor
) -> torch.Tensor:
    if (
        src.dtype != torch.bfloat16
        or src.dim() != 2
        or src.stride(1) != 1
        or out.dtype != torch.bfloat16
        or not out.is_contiguous()
        or index.dim() != 1
        or tuple(out.shape) != (index.shape[0], src.shape[1])
    ):
        raise ValueError(
            f"copy_rows: src {src.dtype} {tuple(src.shape)} {tuple(src.stride())}, "
            f"index {tuple(index.shape)}, out {out.dtype} {tuple(out.shape)}"
        )
    m, hidden = int(index.shape[0]), int(src.shape[1])
    if m:
        _copy_rows_kernel[(m, triton.cdiv(hidden, _BLOCK))](
            index,
            index.stride(0),
            src.view(torch.int16),
            out.view(torch.int16),
            hidden,
            src.stride(0),
            int(src.shape[0]),
            BLOCK=_BLOCK,
            num_warps=4,
        )
    return out
