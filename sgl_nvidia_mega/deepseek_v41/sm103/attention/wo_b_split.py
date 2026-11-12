# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Slot wo_b_publish_split: wo_b at 1-8 rows as muLLM's cluster split-K FP8 projection
whose epilogue publishes into the hop's Lamport mailbox (src/wo_b_split.cu), in place of
FlashInfer's low-latency GEMM with the publishing epilogue (``wo_b_publish``) on hop A,
and of the low-latency GEMM + the hop's separate publish kernel on the draft's hop.

The device code is derived from muLLM v0.4.1's one-group publishing projection (NVIDIA
Corporation, Apache-2.0; src/wo_b_split.cu names its origin); this module is its host
side: the one-time repack of SGLang's MXFP8 wo_b into the kernel's fragment-major order
with 32x32 UE8M0 scales (``q_cluster.pack``), and a driver launch in clusters of S CTAs
(grid (S, N/64)) with PDL. S is muLLM's ``wo_b_splits``: 4 up to K = 1024 per rank.

Inputs: the fused wo_a's MXFP8 rows, e4m3 [rows, K] and FlashInfer's 128x4-swizzled
UE8M0 scales, as the low-latency GEMM takes them. Mailbox: ``AllReduceMHC.publish_target``
(generation ``stage_state[0]``, slot [generation][rank][row], BF16 with negative zero
cleared, then ``stage_state[1]``); the consumer is unchanged.

Numerics: not bitwise with FlashInfer's GEMM. Each 32-value k-block's FP8 mma.sync product
is folded into FP32 with its two scales, the warps' and the S CTAs' partials summed in a
fixed order; the weights, their scales and the input bytes are the same.
Variants compile at build (:func:`precompile`), never under CUDA-graph capture.
"""

from __future__ import annotations

import ctypes
from functools import cache
from pathlib import Path

import torch

from sgl_nvidia_mega.common.jit import Kernel as _Kernel
from sgl_nvidia_mega.common.jit import compile_source as _compile_source
from sgl_nvidia_mega.common.jit import launch as _launch
from sgl_nvidia_mega.common.jit import load as _load
from sgl_nvidia_mega.common.jit import pointer as _pointer
from sgl_nvidia_mega.deepseek_v41.sm103.attention import q_cluster

SOURCE = Path(__file__).resolve().parent / "src" / "wo_b_split.cu"
NVRTC_OPTIONS = ("-std=c++17", "-default-device")
SYMBOL = "dsv41_wo_b_split"

MAX_ROWS = 8  # one fragment group
THREADS = 256
TILE_N = 64
SPAN = 64
FX_WARP_SPANS = 4  # kFxWarpSpans: a warp holds at most four 64-value spans
# muLLM's WO_B_SPLITS and the native kernels' bound behind wo_b_splits (graph/attention.py,
# graph/cycle_batched.py): K / splits >= 128 * ceil(MAX_GROUPS / 4) with MAX_GROUPS = 6
WO_B_SPLITS = 4
_MIN_SPLIT_INPUTS = 128 * 2


def splits(k: int) -> int:
    """muLLM's ``wo_b_splits`` for K inputs per rank."""
    s = WO_B_SPLITS
    while s > 1 and k // s < _MIN_SPLIT_INPUTS:
        s //= 2
    return s


def reason(weight: torch.Tensor) -> str:
    """Why the kernel cannot take this wo_b, or ""."""
    if weight.dtype != torch.float8_e4m3fn or weight.dim() != 2:
        return f"wo_b {weight.dtype} {tuple(weight.shape)}"
    n, k = weight.shape
    s = splits(k)
    if n % TILE_N:
        return f"wo_b width {n} (multiples of {TILE_N})"
    if s < 2 or k % (SPAN * s * 2):
        return f"wo_b K {k} does not split over {s} CTAs of whole span pairs"
    if k // SPAN // s // 2 > FX_WARP_SPANS:
        return f"wo_b K {k} needs more than {FX_WARP_SPANS} spans a warp at {s} splits"
    return ""


def pack(weight: torch.Tensor, swizzled_scales: torch.Tensor) -> q_cluster.PackedWeight:
    """SGLang's MXFP8 wo_b (``weight``, ``weight_scale_inv_swizzled``) in the kernel's
    order. Raises ValueError when the kernel cannot take it or the per-row scales are not
    uniform over 32-row blocks."""
    why = reason(weight)
    if why:
        raise ValueError(why)
    n, k = weight.shape
    scales = q_cluster.block_scales(q_cluster.unswizzle_scales(swizzled_scales, n, k))
    if scales is None:
        raise ValueError("wo_b's UE8M0 scales differ inside a 32-row block")
    return q_cluster.PackedWeight(q_cluster.pack_fp8_weight(weight), scales)


@cache
def load_kernel(device_index: int) -> _Kernel:
    device = torch.device("cuda", device_index)
    with torch.cuda.device(device):
        torch.zeros(1, device=device)  # make the primary context current
        major, minor = torch.cuda.get_device_capability(device)
        cubin = _compile_source(
            SOURCE.read_text(), "wo_b_split", f"sm_{major}{minor}a", NVRTC_OPTIONS
        )
        return _load(cubin, SYMBOL, (THREADS, 1, 1), 0)


def precompile(*, device_index: int | None = None) -> None:
    if device_index is None:
        device_index = torch.cuda.current_device()
    load_kernel(device_index)


def wo_b_split(
    codes: torch.Tensor,
    scales: torch.Tensor,
    packed: q_cluster.PackedWeight,
    *,
    multicast: int,
    stage_state: int,
    rank: int,
    tp: int,
    capacity: int,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    """y [rows, N] bf16 = x wo_b^T for x the MXFP8 rows ``codes`` (e4m3 [rows, K],
    contiguous) with FlashInfer 128x4-swizzled ``scales``, rows 1-8; also published into
    the mailbox at ``multicast`` (generation from ``stage_state``) as this rank's slot."""
    rows, k = codes.shape
    n = packed.weight.shape[0] * 16
    if not (
        codes.dtype in (torch.float8_e4m3fn, torch.uint8)
        and codes.is_contiguous()
        and 1 <= rows <= MAX_ROWS
        and rows <= capacity
        and k == packed.weight.shape[1] * SPAN
        and scales.numel() >= -(-k // 32 // 4) * 512
    ):
        raise ValueError(
            f"wo_b_split takes e4m3 [1-{min(MAX_ROWS, capacity)}, {packed.weight.shape[1] * SPAN}]"
            f" contiguous rows with 128x4 scales, got {tuple(codes.shape)} {codes.dtype}"
        )
    if out is None:
        out = torch.empty(rows, n, dtype=torch.bfloat16, device=codes.device)
    s = splits(k)
    kernel = load_kernel(codes.device.index)
    _launch(
        kernel,
        (s, n // TILE_N, 1),
        [
            _pointer(codes),
            _pointer(scales),
            _pointer(packed.weight),
            _pointer(packed.scales),
            _pointer(out),
            ctypes.c_int32(rows),
            ctypes.c_int32(n),
            ctypes.c_int32(k),
            ctypes.c_uint64(multicast),
            ctypes.c_uint64(stage_state),
            ctypes.c_int32(rank),
            ctypes.c_int32(tp),
            ctypes.c_int32(capacity),
        ],
        torch.cuda.current_stream(codes.device).cuda_stream,
        cluster=(s, 1, 1),
    )
    return out
