# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# The host side follows DeepGEMM's fp8_fp4_paged_mqa_logits (csrc/apis/attention.hpp) and
# sm100_paged_mqa_logits (csrc/jit_kernels/impls/sm100_mqa_logits.hpp) (MIT License,
# Copyright (c) 2025 DeepSeek; the full notice is in src/paged_logits_q6.cu).
"""DeepGEMM's paged MXFP4 index logits with all of a request's rows in one Q block
(slot attn_indexer_logits_q6).

DeepGEMM sizes the Q block as 128 / heads rows, 4 at 32 index heads, so the 6 verify rows
run as two Q blocks that each stream the request's whole index-K range: the source
layers' logits cost grows twice with the context. ``src/paged_logits_q6.cu`` is
DeepGEMM's kernel with one Q block of up to 7 rows (UMMA_N = rows * heads): the
same TMA maps, MMA columns and bf16 reduction, so the logits are bit for bit DeepGEMM's,
from DeepGEMM's own schedule metadata. Variants compile at build, never under capture.
"""

from __future__ import annotations

import ctypes
from pathlib import Path

import torch

from sgl_nvidia_mega.common.jit import Kernel, check, driver, launch
from sgl_nvidia_mega.deepseek_v41.sm103.hc_coeff import _deep_gemm_include, compile_nvcc

SOURCE = Path(__file__).resolve().parent / "src" / "paged_logits_q6.cu"
SYMBOL = "dsv41_paged_logits_q6"
HEADS, HEAD_DIM, SPLIT_KV = 32, 128, 256
THREADS = 384
KV_STAGES = 10
# the math warps release a two-stage variant's TMEM stage before they reduce it
EARLY_RELEASE = True
# 1-4 rows fit DeepGEMM's single Q block; 8 rows need 528 TMEM columns
ROWS = range(5, 8)
PAGES = (32, 64, 128)
# DeepGEMM's nvcc flags for its JIT kernels
NVCC_OPTIONS = (
    "-std=c++20",
    "-O3",
    "--expt-relaxed-constexpr",
    "--expt-extended-lambda",
    "--diag-suppress=39,161,174,177,186,940",
    "--ptxas-options=--register-usage-level=10",
)

_maps: dict[tuple, object] = {}
_kernels: dict[tuple, Kernel] = {}


def serves(rows: int, page: int) -> bool:
    return rows in ROWS and page in PAGES


def is_loaded(device_index: int, rows: int, page: int) -> bool:
    return (device_index, rows, page) in _kernels


def load_kernel(device_index: int, rows: int, page: int) -> Kernel:
    """Compile (or read from the cache) and load one variant on ``device_index``."""
    key = (device_index, rows, page)
    if key not in _kernels:
        _kernels[key] = _load_kernel(device_index, rows, page)
    return _kernels[key]


def _load_kernel(device_index: int, rows: int, page: int) -> Kernel:
    if not serves(rows, page):
        raise ValueError(f"paged_logits_q6 takes {list(ROWS)} rows and pages {PAGES}")
    device = torch.device("cuda", device_index)
    cu = driver()
    with torch.cuda.device(device):
        torch.zeros(1, device=device)
        major, minor = torch.cuda.get_device_capability(device)
        options = NVCC_OPTIONS + (
            f"-I{_deep_gemm_include()}",
            f"-DQ6_ROWS={rows}",
            f"-DQ6_PAGE_KV={page}",
            f"-DQ6_KV_STAGES={KV_STAGES}",
            f"-DQ6_EARLY_RELEASE={int(EARLY_RELEASE)}",
        )
        cubin = compile_nvcc(SOURCE, f"sm_{major}{minor}a", options)
        (module,) = check(cu.cuModuleLoadData(cubin), "cuModuleLoadData")
        (function,) = check(cu.cuModuleGetFunction(module, SYMBOL.encode()), SYMBOL)
        address, size = check(
            cu.cuModuleGetGlobal(module, b"dsv41_paged_logits_q6_smem"), "smem size"
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


def _encode(dtype, base: int, dims, strides, box, swizzle):
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
        "cuTensorMapEncodeTiled(paged_logits_q6)",
    )
    return tensor_map


def _cached(key, make):
    tensor_map = _maps.get(key)
    if tensor_map is None:
        tensor_map = _maps[key] = make()
    return tensor_map


def _kv_maps(k_cache: torch.Tensor):
    """DeepGEMM's views of the fused index-K pool [pages, page, 1, 64 + 4]: FP4 codes
    (3D, 64B swizzle) and the UE8M0 scales after each page's codes. A pool never moves."""
    cu = driver()
    t = cu.CUtensorMapDataType
    pages, page = k_cache.shape[0], k_cache.shape[1]
    stride = k_cache.stride(0)
    base = k_cache.data_ptr()
    kv = _cached(
        ("kv", base, pages, page, stride),
        lambda: _encode(
            t.CU_TENSOR_MAP_DATA_TYPE_16U4_ALIGN8B,
            base,
            (HEAD_DIM, page, pages),
            (HEAD_DIM // 2, stride),
            (HEAD_DIM, page, 1),
            cu.CUtensorMapSwizzle.CU_TENSOR_MAP_SWIZZLE_64B,
        ),
    )
    sf = _cached(
        ("sf", base, pages, page, stride),
        lambda: _encode(
            t.CU_TENSOR_MAP_DATA_TYPE_INT32,
            base + page * HEAD_DIM // 2,
            (page, pages),
            (stride,),
            (page, 1),
            cu.CUtensorMapSwizzle.CU_TENSOR_MAP_SWIZZLE_NONE,
        ),
    )
    return kv, sf


def _q_maps(q_fp4, q_sf, weights, rows: int):
    cu = driver()
    t, none = cu.CUtensorMapDataType, cu.CUtensorMapSwizzle.CU_TENSOR_MAP_SWIZZLE_NONE
    tokens = q_fp4.shape[0] * q_fp4.shape[1]
    q_key = (
        q_fp4.data_ptr(),
        q_sf.data_ptr(),
        weights.data_ptr(),
        weights.stride(0),
        tokens,
        rows,
    )
    return _cached(
        ("q",) + q_key,
        lambda: (
            _encode(
                t.CU_TENSOR_MAP_DATA_TYPE_16U4_ALIGN8B,
                q_fp4.data_ptr(),
                (HEAD_DIM, tokens * HEADS),
                (HEAD_DIM // 2,),
                (HEAD_DIM, rows * HEADS),
                cu.CUtensorMapSwizzle.CU_TENSOR_MAP_SWIZZLE_64B,
            ),
            _encode(
                t.CU_TENSOR_MAP_DATA_TYPE_INT32,
                q_sf.data_ptr(),
                (HEADS, tokens),
                (HEADS * 4,),
                (HEADS, rows),
                none,
            ),
            _encode(
                t.CU_TENSOR_MAP_DATA_TYPE_BFLOAT16,
                weights.data_ptr(),
                (HEADS, tokens),
                (weights.stride(0) * 2,),
                (HEADS, rows),
                none,
            ),
        ),
    )


def paged_logits(
    q: tuple[torch.Tensor, torch.Tensor],
    k_cache: torch.Tensor,
    weights: torch.Tensor,
    context_lens: torch.Tensor,
    block_table: torch.Tensor,
    schedule_meta: torch.Tensor,
    max_context_len: int,
) -> torch.Tensor:
    """``deep_gemm.fp8_fp4_paged_mqa_logits(q, k_cache, weights, context_lens, block_table,
    schedule_meta, max_context_len, False, logits_dtype=torch.bfloat16)`` for MXFP4 ``q``
    ([B, n, 32, 64] int8 codes, [B, n, 32] int32 scales), n in 5..7, bf16 ``weights``
    [B * n, 32] and 2D ``context_lens`` [B, n]: the same bf16 logits [B * n,
    max_context_len] (garbage past each row's length, as DeepGEMM's)."""
    q_fp4, q_sf = q
    batch, rows = q_fp4.shape[0], q_fp4.shape[1]
    page = k_cache.shape[1]
    if not (
        q_fp4.dim() == 4
        and q_fp4.shape[2:] == (HEADS, HEAD_DIM // 2)
        and q_fp4.is_contiguous()
        and q_sf.shape == (batch, rows, HEADS)
        and q_sf.dtype == torch.int32
        and q_sf.is_contiguous()
        and weights.dtype == torch.bfloat16
        and weights.shape == (batch * rows, HEADS)
        and weights.stride(1) == 1
        and context_lens.shape == (batch, rows)
        and context_lens.dtype == torch.int32
        and context_lens.is_contiguous()
        and block_table.dtype == torch.int32
        and block_table.stride(1) == 1
        and k_cache.dim() == 4
        and k_cache.shape[2:] == (1, HEAD_DIM // 2 + 4)
        and k_cache.stride(1) == HEAD_DIM // 2 + 4
        and k_cache.dtype == torch.uint8
        and serves(rows, page)
    ):
        raise ValueError("paged_logits_q6: inputs outside the build")
    device = q_fp4.device
    kernel = load_kernel(device.index, rows, page)
    aligned = -(-max_context_len // SPLIT_KV) * SPLIT_KV
    aligned = -(-aligned // 512) * 512
    logits = torch.empty((batch * rows, aligned), dtype=torch.bfloat16, device=device)
    kv_map, sf_kv_map = _kv_maps(k_cache)
    q_map, sf_q_map, w_map = _q_maps(q_fp4, q_sf, weights, rows)
    launch(
        kernel,
        (schedule_meta.shape[0] - 1, 1, 1),
        [
            ctypes.c_uint32(batch * rows),
            ctypes.c_uint32(aligned),
            ctypes.c_uint32(block_table.stride(0)),
            ctypes.c_void_p(context_lens.data_ptr()),
            ctypes.c_void_p(logits.data_ptr()),
            ctypes.c_void_p(block_table.data_ptr()),
            ctypes.c_void_p(schedule_meta.data_ptr()),
            q_map,
            sf_q_map,
            kv_map,
            sf_kv_map,
            w_map,
        ],
        torch.cuda.current_stream(device).cuda_stream,
    )
    return logits[:, :max_context_len]
