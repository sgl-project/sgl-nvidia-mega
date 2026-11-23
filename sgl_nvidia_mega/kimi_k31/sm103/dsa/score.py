# SPDX-License-Identifier: Apache-2.0
# The host side follows DeepGEMM's fp8_fp4_paged_mqa_logits_legacy (csrc/apis/attention.hpp) and
# sm100_paged_mqa_logits (csrc/jit_kernels/impls/legacy/sm100_mqa_logits.hpp) (MIT License, Copyright (c)
# 2025 DeepSeek; the notice is in score.cuh).
"""DSA verify scoring with a request's draft rows in one Q block (score.cuh).

``deep_gemm.fp8_paged_mqa_logits`` sizes its Q block as 128 / heads rows, so a request's
draft rows past that run as two blocks that each stream the whole index-K range. This
kernel is DeepGEMM's own device core and scheduler (the installed headers, built with
NVRTC) with one block per request; the FP32 logits are DeepGEMM's bit for bit."""

from __future__ import annotations

import ctypes
from functools import cache
from pathlib import Path

import torch

from sgl_nvidia_mega.common.jit import Kernel, check, compile_source, driver, launch

SOURCE = Path(__file__).with_suffix(".cuh")
SYMBOL = "k31_score_one_block"
# The FP8 index-K page-64 layout the kernel's TMA maps read: per page, 64 tokens of 128 FP8
# codes, then their 64 FP32 scales; 32 index heads.
HEADS, HEAD_DIM, PAGE = 32, 128, 64
SPLIT_KV = 256
THREADS = 384
# NVRTC has no C++ standard library; DeepGEMM's device headers take these two from libcu++.
STD_HEADERS = {
    "cstdint": "#pragma once\n#include <cuda/std/cstdint>\n",
    "utility": "#pragma once\n#include <cuda/std/utility>\n"
    "namespace std { using cuda::std::integer_sequence; }\n",
}
# DeepGEMM's NVRTC flags; its shared-memory stores take 128-bit integers
OPTIONS = (
    "-std=c++20",
    "-default-device",
    "--device-int128",
    "--diag-suppress=39,161,174,177,186,940",
    "--ptxas-options=--register-usage-level=10",
)


def serves(device: torch.device, next_n: int) -> bool:
    """Whether a request's ``next_n`` rows take the one-block kernel: DeepGEMM already runs up to
    128 / HEADS rows as one block, and one block keeps its three TMEM accumulator stages of
    next_n * HEADS columns within the 512 columns (5 rows at 32 heads). tcgen05: SM100 family."""
    columns = -(-next_n * HEADS // 8) * 8
    return (
        torch.cuda.get_device_capability(device)[0] == 10
        and next_n > 128 // HEADS
        and 3 * columns <= 512
    )


@cache
def load_kernel(device_index: int, next_n: int) -> Kernel:
    """Compile (or read from the cache) and load the ``next_n``-row kernel; at build, never under capture."""
    import deep_gemm

    device = torch.device("cuda", device_index)
    if not serves(device, next_n):
        raise ValueError(f"the one-block score does not serve {next_n} rows here")
    cu = driver()
    with torch.cuda.device(device):
        torch.zeros(1, device=device)
        major, minor = torch.cuda.get_device_capability(device)
        cubin = compile_source(
            SOURCE.read_text(),
            SYMBOL,
            f"sm_{major}{minor}a",
            OPTIONS
            + (
                f"-I{Path(deep_gemm.__file__).resolve().parent / 'include'}",
                f"-DK31_NEXT_N={next_n}",
                f"-DK31_HEADS={HEADS}",
                f"-DK31_HEAD_DIM={HEAD_DIM}",
                f"-DK31_PAGE={PAGE}",
            ),
            STD_HEADERS,
        )
        (module,) = check(cu.cuModuleLoadData(cubin), "cuModuleLoadData")
        (function,) = check(cu.cuModuleGetFunction(module, SYMBOL.encode()), SYMBOL)
        address, size = check(
            cu.cuModuleGetGlobal(module, f"{SYMBOL}_smem".encode()), "smem size"
        )
        smem = ctypes.c_uint32()
        check(
            cu.cuMemcpyDtoH(ctypes.addressof(smem), address, int(size)), "cuMemcpyDtoH"
        )
        check(
            cu.cuFuncSetAttribute(
                function,
                cu.CUfunction_attribute.CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
                smem.value,
            ),
            "cuFuncSetAttribute",
        )
        return Kernel(function, (THREADS, 1, 1), smem.value, cubin)


def _map(dtype, base: int, dims, strides, box, swizzle):
    """DeepGEMM's make_tma_*d_desc: unit element strides, 256B L2 promotion, no OOB fill."""
    cu = driver()
    u64, u32 = cu.cuuint64_t, cu.cuuint32_t
    (tensor_map,) = check(
        cu.cuTensorMapEncodeTiled(
            dtype,
            len(dims),
            base,
            [u64(d) for d in dims],
            [u64(s) for s in strides],
            [u32(b) for b in box],
            [u32(1)] * len(dims),
            cu.CUtensorMapInterleave.CU_TENSOR_MAP_INTERLEAVE_NONE,
            swizzle,
            cu.CUtensorMapL2promotion.CU_TENSOR_MAP_L2_PROMOTION_L2_256B,
            cu.CUtensorMapFloatOOBfill.CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE,
        ),
        "cuTensorMapEncodeTiled(k31 score)",
    )
    return tensor_map


def fp8_paged_mqa_logits(
    q: torch.Tensor,
    kv_cache: torch.Tensor,
    weights: torch.Tensor,
    context_lens: torch.Tensor,
    block_table: torch.Tensor,
    schedule_meta: torch.Tensor,
    max_context_len: int,
    clean_logits: bool = False,
) -> torch.Tensor:
    """``deep_gemm.fp8_paged_mqa_logits``, with a request's ``n`` rows in one Q block where
    ``serves``; elsewhere DeepGEMM's own. Served: the page-64 FP8 index-K, FP8 ``q`` [B, n, 32,
    128], ``kv_cache`` [pages, 64, 1, 132] uint8, FP32 ``weights`` [B * n, 32], 2D
    ``context_lens`` [B, n] and DeepGEMM's own schedule metadata; the same FP32 logits [B * n,
    max_context_len] (garbage past each request's context, as DeepGEMM's)."""
    batch, next_n = q.shape[0], q.shape[1]
    if not serves(q.device, next_n):
        import deep_gemm

        return deep_gemm.fp8_paged_mqa_logits(
            q,
            kv_cache,
            weights,
            context_lens,
            block_table,
            schedule_meta,
            max_context_len,
            clean_logits=clean_logits,
        )
    rows = batch * next_n
    if not (
        not clean_logits  # DeepGEMM's paged logits never clean either
        and q.dtype == torch.float8_e4m3fn
        and q.dim() == 4
        and q.shape[2:] == (HEADS, HEAD_DIM)
        and q.is_contiguous()
        and kv_cache.dtype == torch.uint8
        and kv_cache.dim() == 4
        and kv_cache.shape[1:] == (PAGE, 1, HEAD_DIM + 4)
        and kv_cache.stride(1) == HEAD_DIM + 4
        and kv_cache.stride(3) == 1
        and weights.dtype == torch.float32
        and weights.shape == (rows, HEADS)
        and weights.stride(1) == 1
        and context_lens.dtype == torch.int32
        and context_lens.shape == (batch, next_n)
        and context_lens.is_contiguous()
        and block_table.dtype == torch.int32
        and block_table.shape[0] == batch
        and block_table.stride(1) == 1
        and schedule_meta.dtype == torch.int32
        and schedule_meta.dim() == 2
        and schedule_meta.shape[1] == 2
        and schedule_meta.is_contiguous()
    ):
        raise ValueError("k31 one-block score: inputs outside the page-64 FP8 layout")
    device = q.device
    kernel = load_kernel(device.index, next_n)
    # DeepGEMM's row stride: the context rounded up to the 256-key split (1 KiB of FP32)
    stride = -(-max_context_len // SPLIT_KV) * SPLIT_KV
    logits = torch.empty((rows, stride), dtype=torch.float32, device=device)
    cu = driver()
    t, swizzle = cu.CUtensorMapDataType, cu.CUtensorMapSwizzle
    pages, page_bytes = kv_cache.shape[0], kv_cache.stride(0)
    # q: a block of next_n * 32 head rows of 128 FP8 codes
    q_map = _map(
        t.CU_TENSOR_MAP_DATA_TYPE_UINT8,
        q.data_ptr(),
        (HEAD_DIM, rows * HEADS),
        (HEAD_DIM,),
        (HEAD_DIM, next_n * HEADS),
        swizzle.CU_TENSOR_MAP_SWIZZLE_128B,
    )
    # kv: a page of 64 tokens of 128 FP8 codes; their FP32 scales follow the page's codes
    kv_map = _map(
        t.CU_TENSOR_MAP_DATA_TYPE_UINT8,
        kv_cache.data_ptr(),
        (HEAD_DIM, PAGE, pages),
        (HEAD_DIM, page_bytes),
        (HEAD_DIM, PAGE, 1),
        swizzle.CU_TENSOR_MAP_SWIZZLE_128B,
    )
    sf_kv_map = _map(
        t.CU_TENSOR_MAP_DATA_TYPE_FLOAT32,
        kv_cache.data_ptr() + PAGE * HEAD_DIM,
        (PAGE, pages),
        (page_bytes,),
        (PAGE, 1),
        swizzle.CU_TENSOR_MAP_SWIZZLE_NONE,
    )
    # weights: next_n rows of 32 FP32 head weights
    w_map = _map(
        t.CU_TENSOR_MAP_DATA_TYPE_FLOAT32,
        weights.data_ptr(),
        (HEADS, rows),
        (weights.stride(0) * 4,),
        (HEADS, next_n),
        swizzle.CU_TENSOR_MAP_SWIZZLE_NONE,
    )
    launch(
        kernel,
        (schedule_meta.shape[0] - 1, 1, 1),
        [
            ctypes.c_uint32(rows),
            ctypes.c_uint32(stride),
            ctypes.c_uint32(block_table.stride(0)),
            ctypes.c_void_p(context_lens.data_ptr()),
            ctypes.c_void_p(logits.data_ptr()),
            ctypes.c_void_p(block_table.data_ptr()),
            ctypes.c_void_p(schedule_meta.data_ptr()),
            q_map,
            sf_kv_map,  # the unused sf_q slot: DeepGEMM passes the KV-scale map for FP8
            kv_map,
            sf_kv_map,
            w_map,
        ],
        torch.cuda.current_stream(device).cuda_stream,
    )
    return logits[:, :max_context_len]
