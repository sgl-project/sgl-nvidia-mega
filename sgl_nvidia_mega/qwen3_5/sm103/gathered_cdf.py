# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Target tokens of gathered verify logits by inverse CDF (slot ``gathered_cdf``)."""

from __future__ import annotations

import torch
import triton
import triton.language as tl

__all__ = ["RoundSeeds", "build", "cdf_tokens"]

_BLOCK = 2048
_ROWS_PER_ROUND = 1 << 16
SEED_BASE = 0x5EED_6A3B


@triton.jit
def _cdf_stats_kernel(
    logits_ptr,
    stride_row,
    temperature_ptr,
    m_ptr,
    z_ptr,
    vocab_size,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    part = tl.program_id(1)
    nparts = tl.num_programs(1)
    cols = part * BLOCK + tl.arange(0, BLOCK)
    mask = cols < vocab_size
    inv_t = 1.0 / tl.maximum(tl.load(temperature_ptr + row), 1e-20)
    x = tl.load(logits_ptr + row * stride_row + cols, mask=mask, other=float("-inf"))
    x = x.to(tl.float32) * inv_t
    x = tl.where(x == x, x, float("-inf"))
    m = tl.max(x, axis=0)
    below = tl.sum(tl.where(x < m, tl.exp(x - m), 0.0), axis=0)
    ties = tl.sum((x == m).to(tl.int32), axis=0)
    z = tl.where(m > float("-inf"), below + (ties - 1).to(tl.float32), 0.0)
    tl.store(m_ptr + row * nparts + part, m)
    tl.store(z_ptr + row * nparts + part, z)


@triton.jit
def _cdf_draw(
    logits_ptr,
    stride_row,
    temperature_ptr,
    m_ptr,
    z_ptr,
    row,
    vocab_size,
    nparts,
    u1,
    u2,
    u3,
    u4,
    BLOCK: tl.constexpr,
    PBLOCK: tl.constexpr,
):
    big = 1 << 30
    ps = tl.arange(0, PBLOCK)
    pmask = ps < nparts
    m = tl.load(m_ptr + row * nparts + ps, mask=pmask, other=float("-inf"))
    z = tl.load(z_ptr + row * nparts + ps, mask=pmask, other=0.0)
    M = tl.max(m, axis=0)
    top = tl.min(tl.where(pmask & (m == M), ps, big), axis=0)
    live = m > float("-inf")
    f = tl.where(live, tl.exp(m - M), 0.0)
    w = tl.where(live, tl.where(ps == top, z, f * (1.0 + z)), 0.0)
    cum = tl.cumsum(w, axis=0)
    rest = tl.max(cum, axis=0)
    # stage 1: the row's max column
    take_top = u1 >= rest / (1.0 + rest)
    t2 = u2 * rest
    s_hit = tl.min(tl.where((cum > t2) & (w > 0.0), ps, big), axis=0)
    s_last = tl.max(tl.where(w > 0.0, ps, -1), axis=0)
    s = tl.where(s_hit < big, s_hit, s_last)
    s = tl.where(take_top | (s < 0), top, s)
    m_s = tl.sum(tl.where(ps == s, m, 0.0), axis=0)
    z_s = tl.sum(tl.where(ps == s, z, 0.0), axis=0)
    # stage 3: another split's max column
    take_max = take_top | ((s != top) & (u3 >= z_s / (1.0 + z_s)))
    inv_t = 1.0 / tl.maximum(tl.load(temperature_ptr + row), 1e-20)
    idx = tl.arange(0, BLOCK)
    cols = s * BLOCK + idx
    x = tl.load(
        logits_ptr + row * stride_row + cols,
        mask=cols < vocab_size,
        other=float("-inf"),
    )
    x = x.to(tl.float32) * inv_t
    x = tl.where(x == x, x, float("-inf"))
    j_s = tl.min(tl.where(x == m_s, idx, BLOCK - 1), axis=0)
    e = tl.where((x > float("-inf")) & (idx != j_s), tl.exp(x - m_s), 0.0)
    ce = tl.cumsum(e, axis=0)
    t4 = u4 * tl.max(ce, axis=0)
    c_hit = tl.min(tl.where((ce > t4) & (e > 0.0), idx, big), axis=0)
    c_last = tl.max(tl.where(e > 0.0, idx, -1), axis=0)
    c = tl.where(c_hit < big, c_hit, c_last)
    c = tl.where(take_max | (c < 0), j_s, c)
    return s * BLOCK + c


@triton.jit
def _cdf_select_kernel(
    logits_ptr,
    stride_row,
    temperature_ptr,
    seed_ptr,
    m_ptr,
    z_ptr,
    out_ptr,
    vocab_size,
    nparts,
    BLOCK: tl.constexpr,
    PBLOCK: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    seed = tl.load(seed_ptr + row)
    r1, r2, r3, r4 = tl.rand4x(seed, tl.arange(0, 1))
    token = _cdf_draw(
        logits_ptr,
        stride_row,
        temperature_ptr,
        m_ptr,
        z_ptr,
        row,
        vocab_size,
        nparts,
        tl.sum(r1, axis=0),
        tl.sum(r2, axis=0),
        tl.sum(r3, axis=0),
        tl.sum(r4, axis=0),
        BLOCK,
        PBLOCK,
    )
    tl.store(out_ptr + row, token.to(tl.int64))


def _split_stats(
    logits: torch.Tensor, temperatures: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Pass 1: per (row, split) FP32 ``m`` and ``z``, [rows, splits]."""
    rows, vocab = logits.shape
    nparts = triton.cdiv(vocab, _BLOCK)
    m = torch.empty((rows, nparts), dtype=torch.float32, device=logits.device)
    z = torch.empty((rows, nparts), dtype=torch.float32, device=logits.device)
    _cdf_stats_kernel[(rows, nparts)](
        logits, logits.stride(0), temperatures, m, z, vocab, BLOCK=_BLOCK, num_warps=4
    )
    return m, z


def cdf_tokens(
    logits: torch.Tensor, temperatures: torch.Tensor, seeds: torch.Tensor
) -> torch.Tensor:
    """[rows] int64 token ids, one draw per row from ``softmax(logits / T)``."""
    if logits.dim() != 2 or logits.stride(1) != 1 or not logits.is_cuda:
        raise ValueError("logits must be a CUDA [rows, vocab] tensor")
    rows, vocab = logits.shape
    if temperatures.dtype != torch.float32 or tuple(temperatures.shape) != (rows,):
        raise ValueError("temperatures must be FP32 [rows]")
    if seeds.dtype != torch.int64 or tuple(seeds.shape) != (rows,):
        raise ValueError("seeds must be int64 [rows]")
    out = torch.empty(rows, dtype=torch.int64, device=logits.device)
    if rows == 0:
        return out
    t = temperatures.contiguous()
    m, z = _split_stats(logits, t)
    nparts = m.shape[1]
    _cdf_select_kernel[(rows,)](
        logits,
        logits.stride(0),
        t,
        seeds.contiguous(),
        m,
        z,
        out,
        vocab,
        nparts,
        BLOCK=_BLOCK,
        PBLOCK=triton.next_power_of_2(nparts),
        num_warps=8,
    )
    return out


class RoundSeeds:
    def __init__(
        self, device: torch.device, base: int = SEED_BASE, max_rows: int = 4096
    ):
        if max_rows > _ROWS_PER_ROUND:
            raise ValueError(f"max_rows {max_rows} > {_ROWS_PER_ROUND}")
        self.base = int(base) & ((1 << 40) - 1)
        self.round = torch.zeros(1, dtype=torch.int64, device=device)
        self.rows = torch.arange(max_rows, dtype=torch.int64, device=device)

    def next(self, rows: int) -> torch.Tensor:
        if rows > self.rows.numel():
            raise ValueError(f"{rows} rows > {self.rows.numel()} seeded rows")
        self.round.add_(1)
        return self.rows[:rows] + (self.round * _ROWS_PER_ROUND + self.base)


def build(device: torch.device, columns: int, max_rows: int = 4096) -> RoundSeeds:
    if torch.cuda.is_current_stream_capturing():
        raise RuntimeError("building gathered_cdf inside CUDA-graph capture")
    if columns <= 0:
        raise ValueError(f"vocabulary columns must be positive, got {columns}")
    seeds = RoundSeeds(device, max_rows=max_rows)
    temperatures = torch.ones(2, dtype=torch.float32, device=device)
    draws = torch.arange(2, dtype=torch.int64, device=device)
    for dtype in (torch.bfloat16, torch.float16, torch.float32):
        for pitch in sorted({columns, -(-columns // 16) * 16}):
            logits = torch.zeros((2, pitch), dtype=dtype, device=device)
            cdf_tokens(logits[:, :columns], temperatures, draws)
    torch.cuda.synchronize(device)
    return seeds
