# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""One-launch NVFP4 KV write (quantize K and V + scatter into the native NVFP4 pool layout), JIT CUDA.

See ``csrc/nvfp4_kv_store.cuh``: the bytes written equal the stock path's (FlashInfer ``nvfp4_kv_quantize`` for K and
V + SGLang ``store_nvfp4_kv_cache`` with native scales), which takes four launches and a copy per layer.
"""

from __future__ import annotations

import functools
from pathlib import Path

import torch

# The SGLang names this module reads (test/unit/test_contract.py pins them against the supported SGLang)
SGLANG_READS = {
    "kernels/jit/utils/__init__.py": ("load_jit", "make_cpp_args"),
}


_CSRC = Path(__file__).resolve().parent.parent / "csrc" / "nvfp4_kv_store.cuh"


@functools.lru_cache(maxsize=None)
def _module(loc64: bool, use_pdl: bool):
    from sglang.kernels.jit.utils import load_jit, make_cpp_args

    args = make_cpp_args(loc64, use_pdl)
    return load_jit(
        "muse_vr_nvfp4_kv_store",
        *args,
        cuda_files=[str(_CSRC)],
        cuda_wrappers=[("nvfp4_kv_store", f"muse_vr::Nvfp4KvStoreKernel<{args}>::run")],
        # the stock quantization kernel is compiled with -use_fast_math; same flags -> same arithmetic
        extra_cuda_cflags=["-use_fast_math"],
    )


def _rows_ok(t: torch.Tensor) -> bool:
    return (
        t.dim() == 3
        and t.dtype == torch.bfloat16
        and t.is_cuda
        and t.stride(2) == 1
        and t.stride(1) % 8 == 0
        and t.stride(0) % 8 == 0
        and t.data_ptr() % 16 == 0
        and t.shape[2] % 64 == 0
    )


def _scale_ok(s) -> bool:
    return (
        isinstance(s, torch.Tensor)
        and s.is_cuda
        and s.dtype == torch.float32
        and s.numel() >= 1
    )


def supported(
    cache_k,
    cache_v,
    loc,
    k_scale,
    v_scale,
    k_buffer,
    v_buffer,
    k_scale_buffer,
    v_scale_buffer,
    native_k_scale_buffer,
    native_v_scale_buffer,
    page_size: int,
) -> bool:
    """True when the one-launch write can replace the stock write for these arguments (native scales only)."""
    if k_scale_buffer is not None or v_scale_buffer is not None:
        return False
    if native_k_scale_buffer is None or native_v_scale_buffer is None:
        return False
    if not (isinstance(cache_k, torch.Tensor) and isinstance(cache_v, torch.Tensor)):
        return False
    if not (_rows_ok(cache_k) and _rows_ok(cache_v) and cache_k.shape == cache_v.shape):
        return False
    if not (
        isinstance(loc, torch.Tensor)
        and loc.dim() == 1
        and loc.dtype in (torch.int64, torch.int32)
        and loc.numel() == cache_k.shape[0]
        and loc.is_cuda
    ):
        return False
    if not (_scale_ok(k_scale) and _scale_ok(v_scale)):
        return False
    n_heads, hd = cache_k.shape[1], cache_k.shape[2]
    for b in (k_buffer, v_buffer):
        if not (
            b.dim() == 3
            and b.element_size() == 1
            and b.shape[1] == n_heads
            and b.shape[2] == hd // 2
            and b.is_contiguous()
        ):
            return False
    for s in (native_k_scale_buffer, native_v_scale_buffer):
        if not (
            s.dim() == 4
            and s.element_size() == 1
            and s.shape[1] == n_heads
            and s.shape[2] == page_size
            and s.shape[3] == hd // 16
            and s.is_contiguous()
            and s.shape[0] * page_size == k_buffer.shape[0]
        ):
            return False
    return page_size % 4 == 0


def nvfp4_kv_store(
    cache_k: torch.Tensor,
    cache_v: torch.Tensor,
    loc: torch.Tensor,
    k_scale: torch.Tensor,
    v_scale: torch.Tensor,
    k_buffer: torch.Tensor,
    v_buffer: torch.Tensor,
    native_k_scale_buffer: torch.Tensor,
    native_v_scale_buffer: torch.Tensor,
    page_size: int,
    use_pdl: bool = False,
) -> None:
    """Quantize ``cache_k`` / ``cache_v`` ([N, H, hd] bf16) and write them at ``loc`` (see ``supported``)."""
    _module(loc.dtype == torch.int64, bool(use_pdl)).nvfp4_kv_store(
        cache_k,
        cache_v,
        loc,
        k_scale.reshape(-1),
        v_scale.reshape(-1),
        k_buffer.view(torch.uint8),
        v_buffer.view(torch.uint8),
        native_k_scale_buffer.view(torch.uint8),
        native_v_scale_buffer.view(torch.uint8),
        int(page_size),
    )


def preload(device, use_pdl: bool = False) -> None:
    """Build and load both loc-dtype variants before any CUDA-graph capture."""
    for loc_dtype in (torch.int64, torch.int32):
        k = torch.zeros(1, 1, 64, dtype=torch.bfloat16, device=device)
        kb = torch.zeros(4, 1, 32, dtype=torch.uint8, device=device)
        sf = torch.zeros(1, 1, 4, 4, dtype=torch.uint8, device=device)
        gs = torch.ones(1, dtype=torch.float32, device=device)
        nvfp4_kv_store(
            k,
            k,
            torch.zeros(1, dtype=loc_dtype, device=device),
            gs,
            gs,
            kb,
            kb.clone(),
            sf,
            sf.clone(),
            4,
            use_pdl=use_pdl,
        )
    torch.cuda.current_stream(device).synchronize()
