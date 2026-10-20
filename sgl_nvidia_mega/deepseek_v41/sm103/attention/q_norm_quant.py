# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# Adapted from SGLang's python/sglang/kernels/ops/layernorm/rmsnorm_mxfp8.py and
# python/sglang/kernels/ops/quantization/mxfp8_epilogue.py (Apache-2.0).
"""The query-LoRA RMSNorm with its MXFP8 quantization as an epilogue.

At the speculative BS=1 shapes the standalone FlashInfer ``mxfp8_quantize`` launch
costs about as much as the norm it follows, even though it only re-reads what that
norm just wrote. This kernel does the norm and the quantization in one pass; the
quantized values come from the same BF16 rounding the standalone pair produces, so
both outputs are bitwise ``rmsnorm`` followed by
``mxfp8_quantize(..., is_sf_swizzled_layout=True)``.

The scale factors use FlashInfer's 128x4 swizzle:
``off = (g // 4) * 512 + ((r % 32) * 4 + ((r // 32) % 4)) * 4 + (g % 4)``
over a row count padded to a multiple of 128, with the UE8M0 conversion
(positive rounding, subnormals included) that ``mxfp8_quantize`` uses. With
``row_tiles`` each 8-row tile has a 128-row atom of its own, rows at the top:
``off = ((r // 8) * (K // 128) + g // 4) * 512 + (r % 8) * 16 + (g % 4)``, the
activation layout of the low-latency MXFP8 GEMM past 8 rows (``mxfp8_ll``).
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl


# FlashInfer's positive-rounding UE8M0 conversion of a per-group amax, subnormals
# included: the scale byte and the multiplier that maps the group into e4m3 range.
@triton.jit
def ue8m0_scale(amax):
    normalized = amax * (1.0 / 448.0)
    bits = normalized.to(tl.int32, bitcast=True)
    exponent = (bits >> 23) & 255
    mantissa = bits & 0x7FFFFF
    bump = (mantissa != 0) & ~((exponent == 0) & (mantissa <= 0x400000))
    sf = tl.where(normalized <= 0, 0, tl.minimum(exponent + bump.to(tl.int32), 254))
    inv = tl.where(sf == 0, 0, ((254 - sf) << 23)).to(tl.float32, bitcast=True)
    return sf, inv


@triton.jit
def mxfp8_epilogue(
    y,
    row,
    Q,
    S,
    K: tl.constexpr,
    BLOCK: tl.constexpr,
    GROUPS: tl.constexpr,
    ROW_TILES: tl.constexpr = False,
):
    # ROW_TILES: one 128-row scale atom per 8-row tile (module docstring).
    GP: tl.constexpr = BLOCK // 32
    g = tl.arange(0, GP)
    gmask = g < GROUPS
    e = tl.arange(0, 32)
    idx = g[:, None] * 32 + e[None, :]
    v = tl.reshape(y.to(tl.float32), (GP, 32))
    amax = tl.max(tl.abs(v), 1)
    sf, scale = ue8m0_scale(amax)
    q = tl.minimum(tl.maximum(v * scale[:, None], -448.0), 448.0).to(tl.float8e4nv)
    tl.store(Q + row * K + idx, q, gmask[:, None])
    if ROW_TILES:
        off = ((row // 8) * (K // 128) + g // 4) * 512 + (row % 8) * 16 + (g % 4)
    else:
        off = (g // 4) * 512 + ((row % 32) * 4 + ((row // 32) % 4)) * 4 + (g % 4)
    tl.store(S + off, sf.to(tl.uint8), gmask)


@triton.jit
def _rmsnorm_mxfp8_kernel(
    X,
    W,
    Y,
    Q,
    S,
    SX: tl.constexpr,
    M: tl.constexpr,
    K: tl.constexpr,
    EPS: tl.constexpr,
    BLOCK: tl.constexpr,
    GROUPS: tl.constexpr,
    USE_GDC: tl.constexpr = False,
    ROW_TILES: tl.constexpr = False,
):
    row = tl.program_id(0)
    if USE_GDC:
        # Let the next GEMM launch now and stream its weights while this grid waits.
        tl.extra.cuda.gdc_launch_dependents()
    if row < M:
        h = tl.arange(0, BLOCK)
        weight = tl.load(W + h, h < K, 0).to(tl.float32)
        if USE_GDC:
            tl.extra.cuda.gdc_wait()
        v = tl.load(X + row * SX + h, h < K, 0).to(tl.float32)
        inv = tl.rsqrt(tl.sum(v * v, 0) / K + EPS)
        y = (v * inv * weight).to(tl.bfloat16)
        tl.store(Y + row * K + h, y, h < K)
        mxfp8_epilogue(y, row, Q, S, K, BLOCK, GROUPS, ROW_TILES)
    elif ROW_TILES:
        if USE_GDC:
            tl.extra.cuda.gdc_wait()
        # Rows past M in the last 8-row tile: zero scales (one program per such row).
        GP: tl.constexpr = BLOCK // 32
        g = tl.arange(0, GP)
        off = ((row // 8) * (K // 128) + g // 4) * 512 + (row % 8) * 16 + (g % 4)
        tl.store(S + off, 0, g < GROUPS)
    else:
        if USE_GDC:
            # S may still be read by the previous grid's consumer.
            tl.extra.cuda.gdc_wait()
        # Padding scale entries are disjoint from the live rows above.
        off = (row - M) * 512 + tl.arange(0, 512)
        pad_row = ((off // 4) % 4) * 32 + ((off // 16) % 32)
        tl.store(S + off, 0, (off < GROUPS * 128) & (pad_row >= M))


def rmsnorm_mxfp8(
    x: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
    *,
    pdl: bool = False,
    row_tiles: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """RMSNorm returning ``(y_bf16, y_q, y_sf)`` with the same MXFP8 epilogue.
    ``pdl`` launches it as a programmatic dependent that triggers its own dependent at once.
    ``row_tiles`` writes the scales in the row-tile layout (same bytes, moved; rows past
    ``m`` in the last tile get zero scales).
    The BF16 row can differ from ``flashinfer.norm.rmsnorm`` by 1 ulp in rare elements (reduction order)."""
    m, k = x.shape
    # One 128-row scale atom: the plain swizzle FlashInfer's MXFP8 GEMMs read up to 128 rows.
    assert 0 < m <= 128 and k % 32 == 0
    assert not row_tiles or k % 128 == 0
    assert x.dtype == weight.dtype == torch.bfloat16 and x.stride(1) == 1
    y = torch.empty_like(x, memory_format=torch.contiguous_format)
    q = torch.empty((m, k), dtype=torch.float8_e4m3fn, device=x.device)
    if row_tiles:
        tiles = triton.cdiv(m, 8)
        s = torch.empty(tiles * (k // 128) * 512, dtype=torch.uint8, device=x.device)
        grid = tiles * 8
    else:
        s = torch.empty((k // 32) * 128, dtype=torch.uint8, device=x.device)
        grid = m + triton.cdiv(s.numel(), 512)
    _rmsnorm_mxfp8_kernel[(grid,)](
        x,
        weight,
        y,
        q,
        s,
        SX=x.stride(0),
        M=m,
        K=k,
        EPS=eps,
        BLOCK=triton.next_power_of_2(k),
        GROUPS=k // 32,
        num_warps=8,
        enable_fp_fusion=False,
        **({"USE_GDC": True, "launch_pdl": True} if pdl else {}),
        **({"ROW_TILES": True} if row_tiles else {}),
    )
    return y, q, s


@triton.jit
def _mxfp8_row_tiles_kernel(
    X,
    Q,
    S,
    SX: tl.constexpr,
    M: tl.constexpr,
    K: tl.constexpr,
    BLOCK: tl.constexpr,
    GROUPS: tl.constexpr,
):
    row = tl.program_id(0)
    # Let the GEMM that reads Q launch now and stream its weights.
    tl.extra.cuda.gdc_launch_dependents()
    tl.extra.cuda.gdc_wait()
    h = tl.arange(0, BLOCK)
    if row < M:
        v = tl.load(X + row * SX + h, h < K, 0)
        mxfp8_epilogue(v, row, Q, S, K, BLOCK, GROUPS, True)
    else:
        # Rows past M in the last 8-row tile: zero scales (one program per such row).
        GP: tl.constexpr = BLOCK // 32
        g = tl.arange(0, GP)
        off = ((row // 8) * (K // 128) + g // 4) * 512 + (row % 8) * 16 + (g % 4)
        tl.store(S + off, 0, g < GROUPS)


def mxfp8_row_tiles(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """BF16 rows as MXFP8 ``(q, sf)`` with the scales in the row-tile layout (the
    low-latency GEMM's activation layout; FlashInfer's 128x4 swizzle at 1-8 rows), the
    epilogue above without the norm: bitwise ``mxfp8_quantize(x, is_sf_swizzled_layout=True)``
    with its scales moved. Launched as a programmatic dependent that triggers its own
    dependent at once."""
    m, k = x.shape
    assert 0 < m <= 128 and k % 128 == 0
    assert x.dtype == torch.bfloat16 and x.stride(1) == 1
    q = torch.empty((m, k), dtype=torch.float8_e4m3fn, device=x.device)
    tiles = triton.cdiv(m, 8)
    s = torch.empty(tiles * (k // 128) * 512, dtype=torch.uint8, device=x.device)
    _mxfp8_row_tiles_kernel[(tiles * 8,)](
        x,
        q,
        s,
        SX=x.stride(0),
        M=m,
        K=k,
        BLOCK=triton.next_power_of_2(k),
        GROUPS=k // 32,
        num_warps=8,
        enable_fp_fusion=False,
        launch_pdl=True,
    )
    return q, s
