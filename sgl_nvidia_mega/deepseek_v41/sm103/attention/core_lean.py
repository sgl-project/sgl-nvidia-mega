# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Slot attn_core_lean: the decode attention core in one launch at 1-8 rows on attention
TP8's 8 local heads (src/core_lean.cu), in place of swap-AB + combine, on SGLang's
``SGLANG_DSV4_KV_LAYOUT=v41_new`` pool (528-byte window rows, 288-byte FP4 compressed
rows).

The device code is derived from the sparse attention of muLLM v0.4.1 (NVIDIA
Corporation, Apache-2.0; src/core_lean.cu names its origin and lists the adaptations);
this module is its host side: the pool geometry and the launch, one cluster of 10 CTAs
per row (128 window + 512 compressed keys, 64 per CTA) with muLLM's one-CTA-per-SM
shared-memory pad, or 2 CTAs per row on window-only layers, with PDL. The kernel merges
the row's splits with the sink and applies the output's inverse RoPE itself, so its
consumer (the fused wo_a) runs with an identity rotation (:func:`identity_rope`).

Numerics: not bitwise with swap-AB + combine on the V4 layout. The KV rows are SGLang's
V4.1 encodings (E4M3 with one UE8M0 per 32 values, RoPE dims included; FP4 compressed
rows); P is BF16 (no compensation); scores, softmax statistics and the merge follow
muLLM's 64-key splits; the output is rounded to BF16 before its inverse RoPE.
Variants compile at build (:func:`precompile`), never under CUDA-graph capture.
"""

from __future__ import annotations

import ctypes
from functools import cache
from pathlib import Path
from typing import NamedTuple, Optional

import torch

from sgl_nvidia_mega.common.jit import Kernel as _Kernel
from sgl_nvidia_mega.common.jit import check as _check
from sgl_nvidia_mega.common.jit import compile_source as _compile_source
from sgl_nvidia_mega.common.jit import driver as _driver
from sgl_nvidia_mega.common.jit import launch as _launch
from sgl_nvidia_mega.common.jit import load as _load
from sgl_nvidia_mega.common.jit import pointer as _pointer

SOURCE = Path(__file__).resolve().parent / "src" / "core_lean.cu"
NVRTC_OPTIONS = ("-std=c++17", "-default-device")

HEADS = 8
HEAD_DIM = 512
MAX_ROWS = 8
WINDOW = 128
TOP_K = 512
KEYS_PER_SPLIT = 64
SPARSE_SPLITS = 10
WINDOW_SPLITS = 2
# the DSpark draft's window: up to 192 keys in 3 splits (dsv41_core_lean_window3)
DRAFT_WINDOW = 192
DRAFT_WINDOW_SPLITS = 3
THREADS = 512
WINDOW_ROW_BYTES = 528  # v41_new: 512 E4M3 + 16 UE8M0
EXTRA_ROW_BYTES = 288  # v41_fp4_new: 256 B E2M1 + 32 E4M3
# sizeof(SharedStorage) in the kernel, and the per-key valid flags after it
STORAGE_BYTES = 90_304
FLAG_BYTES = 64
# Shared memory every resident CTA holds back on these targets (muLLM's CTA_RESERVED_BYTES)
CTA_RESERVED_BYTES = 1024


class Caches(NamedTuple):
    """The step's views of one layer's caches, as ``chain.resolve_caches`` returns them."""

    swa: torch.Tensor  # [pages, page, 1, 528]
    swa_indices: torch.Tensor  # [rows, 1, NK] int32
    swa_lengths: torch.Tensor  # [rows] int32
    extra: Optional[torch.Tensor]  # [pages, page, 1, 288]
    extra_indices: Optional[torch.Tensor]  # [rows, 1, NE] int32
    extra_lengths: Optional[torch.Tensor]  # [rows] int32


def _page_shift(pool: torch.Tensor) -> int:
    page = pool.shape[1]
    return page.bit_length() - 1 if page > 0 and page & (page - 1) == 0 else -1


def reason(rows: int, heads: int, caches: Caches) -> str:
    """Why the kernel cannot take this call, or ""."""
    if not 0 < rows <= MAX_ROWS:
        return f"{rows} rows (the kernel takes 1-{MAX_ROWS})"
    if heads != HEADS:
        return f"{heads} local heads (the kernel takes {HEADS})"
    if caches.swa.shape[-1] != WINDOW_ROW_BYTES:
        return f"window rows of {caches.swa.shape[-1]} B (v41_new's {WINDOW_ROW_BYTES})"
    if _page_shift(caches.swa) < 0:
        return f"window page of {caches.swa.shape[1]} tokens (a power of two)"
    cap = WINDOW if caches.extra is not None else DRAFT_WINDOW
    if caches.swa_indices.shape[-1] > cap:
        return f"{caches.swa_indices.shape[-1]} window indices (at most {cap})"
    tensors = [caches.swa_indices, caches.swa_lengths]
    if caches.extra is not None:
        if caches.extra.shape[-1] != EXTRA_ROW_BYTES:
            return f"compressed rows of {caches.extra.shape[-1]} B (v41_fp4_new's {EXTRA_ROW_BYTES})"
        if _page_shift(caches.extra) < 0:
            return f"compressed page of {caches.extra.shape[1]} tokens (a power of two)"
        if caches.extra_indices.shape[-1] > TOP_K:
            return f"{caches.extra_indices.shape[-1]} sparse indices (at most {TOP_K})"
        tensors += [caches.extra_indices, caches.extra_lengths]
    for t in tensors:
        if t.dtype != torch.int32 or t.stride(-1) != 1:
            return f"index tensor {t.dtype} strides {t.stride()} (int32, unit stride)"
    if not caches.swa_lengths.is_contiguous() or (
        caches.extra_lengths is not None and not caches.extra_lengths.is_contiguous()
    ):
        return "lengths not contiguous"
    return ""


def _pad_bytes(device_index: int) -> int:
    """muLLM's sparse_pad_bytes: 16 bytes past half an SM's shared memory, so two of
    these CTAs never share an SM (115,728 bytes on GB300)."""
    props = torch.cuda.get_device_properties(device_index)
    limit = props.shared_memory_per_block_optin
    return (limit + CTA_RESERVED_BYTES) // 2 - CTA_RESERVED_BYTES + 16


def _window_splits(width: int) -> int:
    return WINDOW_SPLITS if width <= WINDOW else DRAFT_WINDOW_SPLITS


def _cubin(device_index: int, pos_int64: bool) -> bytes:
    major, minor = torch.cuda.get_device_capability(device_index)
    return _compile_source(
        SOURCE.read_text(),
        f"core_lean_p{int(pos_int64)}",
        f"sm_{major}{minor}a",
        NVRTC_OPTIONS + (f"-DCORE_POS_INT64={int(pos_int64)}",),
    )


@cache
def load_kernel(
    device_index: int,
    sparse: bool,
    early: bool,
    pos_int64: bool,
    splits: int = 0,
) -> _Kernel:
    device = torch.device("cuda", device_index)
    with torch.cuda.device(device):
        torch.zeros(1, device=device)  # make the primary context current
        cubin = _cubin(device_index, pos_int64)
        symbol = "dsv41_core_lean_" + (
            "sparse"
            if sparse
            else "window3"
            if splits == DRAFT_WINDOW_SPLITS
            else "window"
        )
        if early:
            symbol += "_early"
        shared = STORAGE_BYTES + FLAG_BYTES
        if sparse:
            shared = max(shared, _pad_bytes(device_index))
        kernel = _load(cubin, symbol, (THREADS, 1, 1), shared)
        if sparse:
            cu = _driver()
            _check(
                cu.cuFuncSetAttribute(
                    kernel.function,
                    cu.CUfunction_attribute.CU_FUNC_ATTRIBUTE_NON_PORTABLE_CLUSTER_SIZE_ALLOWED,
                    1,
                ),
                "cuFuncSetAttribute(non-portable cluster)",
            )
        return kernel


def precompile(*, device_index: int | None = None) -> None:
    if device_index is None:
        device_index = torch.cuda.current_device()
    for sparse in (False, True):
        for early in (False, True):
            for pos_int64 in (False, True):
                for splits in (0,) if sparse else (WINDOW_SPLITS, DRAFT_WINDOW_SPLITS):
                    load_kernel(device_index, sparse, early, pos_int64, splits)


@cache
def _identity(device_index: int) -> tuple[torch.Tensor, torch.Tensor]:
    device = torch.device("cuda", device_index)
    freqs = torch.zeros(1, 64, dtype=torch.float32, device=device)
    freqs[:, 0::2] = 1.0
    positions = torch.zeros(64, dtype=torch.int32, device=device)
    return freqs, positions


def identity_rope(rows: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    """(freqs [1, 64] = (cos 1, sin 0) pairs, positions [rows] zeros): the rotation the
    fused wo_a applies to this kernel's output, which is already inverse-roped."""
    index = device.index if device.index is not None else torch.cuda.current_device()
    freqs, positions = _identity(index)
    return freqs, positions[:rows]


def core_lean(
    q: torch.Tensor,
    freqs: torch.Tensor,
    positions: torch.Tensor,
    sink: torch.Tensor,
    caches: Caches,
    softmax_scale: float,
    *,
    early: bool = False,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """o [rows, 8, 512] bf16 (inverse-roped) for the unroped ``q`` [rows, 8, 512] bf16
    (unit stride inside a row), ``freqs`` fp32 [S, 64] (cos, sin interleaved),
    ``positions`` [rows] int32/int64, ``sink`` fp32 [>= 8]. ``early``: the window indices
    and lengths are read before the PDL wait (the caller's guarantee: they were written
    before the replay)."""
    rows = q.shape[0]
    q2 = q.reshape(rows, -1)
    why = reason(rows, q.shape[-2] if q.dim() == 3 else q2.shape[1] // HEAD_DIM, caches)
    if why:
        raise ValueError(f"core_lean: {why}")
    if not (
        q.dtype == torch.bfloat16
        and q2.shape[1] == HEADS * HEAD_DIM
        and q2.stride(1) == 1
        and q2.stride(0) % 8 == 0
        and q2.data_ptr() % 16 == 0
    ):
        raise ValueError(
            f"core_lean: q {q.dtype} {tuple(q.shape)} strides {q.stride()}"
        )
    if not (
        freqs.dtype == torch.float32
        and freqs.dim() == 2
        and freqs.shape[1] == 64
        and freqs.is_contiguous()
    ):
        raise ValueError("core_lean: freqs must be a contiguous fp32 [S, 64]")
    if not (
        positions.dtype in (torch.int32, torch.int64)
        and positions.shape == (rows,)
        and positions.is_contiguous()
    ):
        raise ValueError("core_lean: positions must be a contiguous int32/int64 [rows]")
    if sink.dtype != torch.float32 or sink.stride(0) != 1 or sink.numel() < HEADS:
        raise ValueError("core_lean: sink must be fp32 with >= 8 unit-stride values")
    sparse = caches.extra is not None
    splits = SPARSE_SPLITS if sparse else _window_splits(caches.swa_indices.shape[-1])
    if out is None:
        out = torch.empty(rows, HEADS, HEAD_DIM, dtype=torch.bfloat16, device=q.device)
    kernel = load_kernel(
        q.device.index,
        sparse,
        early,
        positions.dtype == torch.int64,
        0 if sparse else splits,
    )
    swa = caches.swa
    if sparse:
        extra, extra_idx, extra_len = (
            caches.extra,
            caches.extra_indices,
            caches.extra_lengths,
        )
        extra_args = [
            _pointer(extra),
            ctypes.c_int64(extra.stride(0) * extra.element_size()),
            ctypes.c_int32(_page_shift(extra)),
            _pointer(extra_idx),
            ctypes.c_int32(extra_idx.stride(0)),
            ctypes.c_int32(extra_idx.shape[-1]),
            _pointer(extra_len),
        ]
    else:
        extra_args = [
            ctypes.c_void_p(0),
            ctypes.c_int64(0),
            ctypes.c_int32(0),
            ctypes.c_void_p(0),
            ctypes.c_int32(0),
            ctypes.c_int32(0),
            ctypes.c_void_p(0),
        ]
    _launch(
        kernel,
        (splits, rows, 1),
        [
            _pointer(q2),
            ctypes.c_int32(q2.stride(0)),
            _pointer(freqs),
            _pointer(positions),
            _pointer(swa),
            ctypes.c_int64(swa.stride(0) * swa.element_size()),
            ctypes.c_int32(_page_shift(swa)),
            _pointer(caches.swa_indices),
            ctypes.c_int32(caches.swa_indices.stride(0)),
            _pointer(caches.swa_lengths),
            *extra_args,
            _pointer(sink),
            _pointer(out),
            ctypes.c_float(softmax_scale),
        ],
        torch.cuda.current_stream(q.device).cuda_stream,
        cluster=(splits, 1, 1),
    )
    return out
