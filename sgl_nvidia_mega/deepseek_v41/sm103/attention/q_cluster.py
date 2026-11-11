# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Slot attn_q_cluster: q_lora RMSNorm + MXFP8 quant + wq_b in one cluster launch at 1-8
rows (src/q_cluster.cu), in place of the norm + quant and the low-latency wq_b GEMM.

The device code is derived from the query-projection cluster kernel of muLLM v0.4.1
(NVIDIA Corporation, Apache-2.0; src/q_cluster.cu lists what it carries over); this module is its host side: the one-time repack of
SGLang's loaded wq_b into the kernel's fragment-major order with 32x32 UE8M0 scales, and
a driver launch in clusters of 8 CTAs (two K splits of four 64-column tiles) with PDL.
At N = 4096 (attention TP8's 8 heads) it runs the fit entry: 15 clusters plus two extra
warps per CTA for the 16th cluster's tiles, so every cluster has one CTA per SM.

Numerics: not bitwise with the chain's pair (Triton norm + quant, FlashInfer's
low-latency GEMM). The RMS sum order differs (thread-strided then warp order), the
UE8M0 exponent floors amax at 1e-4, and the GEMM folds each 32-value k-block's FP8
mma.sync product into FP32 with its two scales, in a fixed split-K order. The weights,
their scales and the input quantization format are the same.
Variants compile at build (:func:`precompile`), never under CUDA-graph capture.
"""

from __future__ import annotations

import ctypes
from functools import cache
from pathlib import Path
from typing import NamedTuple

import torch

from sgl_nvidia_mega.common.jit import Kernel as _Kernel
from sgl_nvidia_mega.common.jit import compile_source as _compile_source
from sgl_nvidia_mega.common.jit import launch as _launch
from sgl_nvidia_mega.common.jit import load as _load
from sgl_nvidia_mega.common.jit import pointer as _pointer

SOURCE = Path(__file__).resolve().parent / "src" / "q_cluster.cu"
NVRTC_OPTIONS = ("-std=c++17", "-default-device")

K = 1280  # q_lora_rank
MAX_ROWS = 8
SPLITS = 2
CLUSTER = (SPLITS, 4, 1)  # two K splits of four 64-column tiles
TILE_N = 64
FIT_N = 4096  # the fit entry's width: 16 clusters of 8 run as 15
FIT_TILES = 60
THREADS, FIT_THREADS = 256, 320


class PackedWeight(NamedTuple):
    """wq_b in the kernel's order: e4m3 [N/16, K/64, 1024] fragment-major and UE8M0
    [N/32, K/32] block scales."""

    weight: torch.Tensor
    scales: torch.Tensor


def unswizzle_scales(swizzled: torch.Tensor, n: int, k: int) -> torch.Tensor:
    """FlashInfer's 128x4-swizzled UE8M0 scales of an [n, k] MXFP8 weight as [n, k/32]."""
    blocks = k // 32
    raw = swizzled.contiguous().view(torch.uint8).reshape(-1)
    rows_p, cols_p = -(-n // 128) * 128, -(-blocks // 4) * 4
    if raw.numel() < rows_p * cols_p:
        raise ValueError(f"{raw.numel()} scale bytes for a [{n}, {k}] weight")
    atoms = raw[: rows_p * cols_p].view(rows_p // 128, cols_p // 4, 32, 4, 4)
    # (row // 128, col // 4, row % 32, (row % 128) // 32, col % 4)
    return (
        atoms.permute(0, 3, 2, 1, 4).reshape(rows_p, cols_p)[:n, :blocks].contiguous()
    )


def swizzle_scales(scales: torch.Tensor) -> torch.Tensor:
    """Inverse of :func:`unswizzle_scales` (for tests): [n, k/32] -> flat swizzled bytes."""
    n, blocks = scales.shape
    rows_p, cols_p = -(-n // 128) * 128, -(-blocks // 4) * 4
    padded = scales.new_zeros(rows_p, cols_p, dtype=torch.uint8)
    padded[:n, :blocks] = scales.view(torch.uint8)
    return (
        padded.view(rows_p // 128, 4, 32, cols_p // 4, 4)
        .permute(0, 3, 2, 1, 4)
        .contiguous()
        .view(-1)
    )


def pack_fp8_weight(weight: torch.Tensor) -> torch.Tensor:
    """Row-major e4m3 [R, K] -> fragment-major [R/16, K/64, 1024] (muLLM's
    ``pack_fp8_weight``)."""
    raw = weight.contiguous().view(torch.uint8)
    rows, k = raw.shape
    if rows % 32 or k % 64:
        raise ValueError("FP8 weight must be [32n, 64m]")
    return (
        raw.view(rows // 16, 2, 8, k // 64, 2, 4, 2, 4)
        .permute(0, 3, 2, 5, 4, 1, 6, 7)
        .contiguous()
        .view(rows // 16, k // 64, 1024)
    )


def block_scales(row_scales: torch.Tensor) -> torch.Tensor | None:
    """[N, K/32] per-row UE8M0 -> [N/32, K/32] when every 32-row block shares its scales
    (a 32x32 or coarser block-scaled checkpoint), else None."""
    n, blocks = row_scales.shape
    if n % 32:
        return None
    grouped = row_scales.view(n // 32, 32, blocks)
    if not torch.equal(grouped, grouped[:, :1].expand_as(grouped)):
        return None
    return grouped[:, 0].contiguous()


def reason(weight: torch.Tensor, rows: int | None = None) -> str:
    """Why the kernel cannot take this wq_b (and row count), or ""."""
    if weight.dtype != torch.float8_e4m3fn or weight.dim() != 2:
        return f"wq_b {weight.dtype} {tuple(weight.shape)}"
    n, k = weight.shape
    if k != K:
        return f"q_lora_rank {k} (the kernel takes {K})"
    if n % (TILE_N * CLUSTER[1]):
        return f"wq_b width {n} (multiples of {TILE_N * CLUSTER[1]})"
    if rows is not None and not 1 <= rows <= MAX_ROWS:
        return f"{rows} rows (the kernel takes 1-{MAX_ROWS})"
    return ""


def pack(weight: torch.Tensor, swizzled_scales: torch.Tensor) -> PackedWeight:
    """SGLang's MXFP8 wq_b (``weight``, ``weight_scale_inv_swizzled``) in the kernel's
    order. Raises ValueError when the per-row scales are not uniform over 32-row blocks."""
    why = reason(weight)
    if why:
        raise ValueError(why)
    n, k = weight.shape
    scales = block_scales(unswizzle_scales(swizzled_scales, n, k))
    if scales is None:
        raise ValueError("wq_b's UE8M0 scales differ inside a 32-row block")
    return PackedWeight(pack_fp8_weight(weight), scales)


def fits(n: int) -> bool:
    return n == FIT_N


@cache
def load_kernel(device_index: int, fit: bool, early_trigger: bool) -> _Kernel:
    device = torch.device("cuda", device_index)
    with torch.cuda.device(device):
        torch.zeros(1, device=device)  # make the primary context current
        major, minor = torch.cuda.get_device_capability(device)
        cubin = _compile_source(
            SOURCE.read_text(),
            f"q_cluster_et{int(early_trigger)}",
            f"sm_{major}{minor}a",
            NVRTC_OPTIONS + (f"-DQ_EARLY_TRIGGER={int(early_trigger)}",),
        )
        symbol = "dsv41_q_cluster8_fit" if fit else "dsv41_q_cluster8"
        threads = FIT_THREADS if fit else THREADS
        return _load(cubin, symbol, (threads, 1, 1), 0)


def precompile(n: int, *, device_index: int | None = None, early_trigger: bool = True):
    if device_index is None:
        device_index = torch.cuda.current_device()
    load_kernel(device_index, fits(n), early_trigger)


def q_cluster(
    q_lora: torch.Tensor,
    q_norm: torch.Tensor,
    eps: float,
    packed: PackedWeight,
    *,
    early_trigger: bool = True,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    """q [rows, N] bf16 = MXFP8(RMSNorm(q_lora) * q_norm) wq_b^T for ``q_lora`` [rows,
    1280] bf16 (unit column stride, any row stride: a view of wqkv_a's output), rows 1-8."""
    rows = q_lora.shape[0]
    n = packed.weight.shape[0] * 16
    if not (
        q_lora.dtype == torch.bfloat16
        and q_lora.dim() == 2
        and q_lora.shape[1] == K
        and q_lora.stride(1) == 1
        and q_lora.stride(0) % 8 == 0
        and q_lora.data_ptr() % 16 == 0
        and 1 <= rows <= MAX_ROWS
    ):
        raise ValueError(
            f"q_lora must be bf16 [1-{MAX_ROWS}, {K}] with 16-byte aligned rows, got "
            f"{tuple(q_lora.shape)} {q_lora.dtype} strides {q_lora.stride()}"
        )
    if (
        q_norm.dtype != torch.bfloat16
        or q_norm.numel() != K
        or not q_norm.is_contiguous()
    ):
        raise ValueError(f"q_norm must be a contiguous bf16 [{K}]")
    if out is None:
        out = torch.empty(rows, n, dtype=torch.bfloat16, device=q_lora.device)
    fit = fits(n)
    kernel = load_kernel(q_lora.device.index, fit, early_trigger)
    grid = (SPLITS, FIT_TILES if fit else n // TILE_N, 1)
    _launch(
        kernel,
        grid,
        [
            _pointer(q_lora),
            _pointer(q_norm),
            _pointer(packed.weight),
            _pointer(packed.scales),
            _pointer(out),
            ctypes.c_int32(rows),
            ctypes.c_int32(n),
            ctypes.c_int32(q_lora.stride(0)),
            ctypes.c_float(eps),
        ],
        torch.cuda.current_stream(q_lora.device).cuda_stream,
        cluster=CLUSTER,
    )
    return out
