# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""MXFP4 verify index logits from the vendored DeepGEMM paged MQA logits header (dg_logits.cuh), launched with PDL
(``K31MK_DSA_IDX_FP4_VENDOR=1`` with ``K31MK_DSA_IDX_FP4=1``; default 0 = idx_fp4.paged_logits' stock DeepGEMM launch).

Drop-in for ``deep_gemm.fp8_fp4_paged_mqa_logits((q4, sf_q), fused_cache, weights, context_lens, block_table, schedule,
max_context_len, False, logits_dtype=float32)`` at the verify shapes (MXFP4 query [B, N, 32, 64] packed + int32 ue8m0 x4 scales
[B, N, 32], MXFP4 index-K pages [blocks, 64, 1, 68], FP32 head weights and logits, 2D context lengths). It instantiates the SAME
kernel template DeepGEMM's own JIT unit instantiates for that call (sgl-deep-gemm 0.2.3.post2, dumped by :
sm100_paged_mqa_logits<N, 32, 128, 64, MX true, 2D true, varlen false, 3, 10, 256, 16, 128, 256, float_e2m1_t, float, float>, shared
storage 202,240 B) from the vendored header, so the only code differences are the header's PDL additions (dg_logits.py
check_header: the grid-dependency wait, thread 0's fenced dependent trigger, the TMA warps' async-proxy fences), which serve the FP4
instantiation unchanged (the same kernel body; kIsFP4 changes tile sizes only). Tensor maps follow DeepGEMM's launcher
(csrc/jit_kernels/impls/sm100_mqa_logits.hpp sm100_paged_mqa_logits, runtime_utils.hpp make_tma_*_desc with fp4_unpacked_smem false:
4-bit packed type 16U4_ALIGN8B, 64-byte swizzle, box inner dim 2 x swizzle). A test checks the logits are bitwise equal to DeepGEMM's.
"""

from __future__ import annotations

import ctypes
from functools import cache

import torch

from sgl_nvidia_mega.common.jit import Kernel, check, driver, launch, load

from .dg_logits import HEADER, LOGITS_ALIGN, MATH_THREADS, SPECIALIZED_THREADS, SPLIT_KV, check_header, compile_unit

HEADS, HEAD_DIM, PAGE = 32, 128, 64
BLOCK_Q = 128 // HEADS
Q_STAGES, KV_STAGES = 3, 10  # DeepGEMM's FP4 paged launch: 10 index-K stages (FP8: 5)
SHARED_BYTES = 202240
VAL_B, ROW_B = 64, 68
SWIZZLE = HEAD_DIM // 2  # 64-byte swizzle for packed FP4 rows


def symbol(tokens_per_request: int) -> str:
    return (f"_ZN9deep_gemm22sm100_paged_mqa_logitsILj{tokens_per_request}ELj32ELj128ELj64ELb1ELb1ELb0ELj3ELj10ELj256ELj16"
            "ELj128ELj256EN7cutlass12float_e2m1_tEffLj2EEEvjjjPKjPT13_S4_S4_S4_14CUtensorMap_stS7_S7_S7_S7_")


def source(tokens_per_request: int, header: str = "dg_logits.cuh") -> str:
    """DeepGEMM's own instantiation unit for the FP4 paged call, including ``header`` instead of its sm100_mqa_logits.cuh. It follows the served
    FP8 unit's K31MK_DSA_LOGITS_EF (dg_logits.source: index-K loads, and at 2 the logits stores, L2 evict_first; cache hints only, the kernel's
    template arguments and symbol are unchanged; unset / 0: the unit is byte-identical to before)."""
    import os

    ef = os.environ.get("K31MK_DSA_LOGITS_EF", "0")
    if ef not in ("0", "1", "2"):
        raise ValueError(f"K31MK_DSA_LOGITS_EF={ef!r}: want 0, 1 or 2")
    return (f"#define K31_DG_KV_EF {ef}\n" if ef != "0" else "") + f"""
#include <{header}>

using namespace deep_gemm;

static_assert(sizeof(layout::MQALogitsSharedStorage<
    {HEADS}, {HEAD_DIM}, true, {BLOCK_Q}, {SPLIT_KV}, {Q_STAGES}, {KV_STAGES}, 3, cutlass::float_e2m1_t, float
>) == {SHARED_BYTES}, "Incorrect MQA logits shared-memory size");

static void __instantiate_kernel() {{
    auto ptr = reinterpret_cast<void*>(&sm100_paged_mqa_logits<
        {tokens_per_request}, {HEADS},
        {HEAD_DIM}, {PAGE},
        true, true, false,
        {Q_STAGES}, {KV_STAGES},
        {SPLIT_KV}, 16,
        {SPECIALIZED_THREADS}, {MATH_THREADS},
        cutlass::float_e2m1_t, float, float
    >);
}};
"""


@cache
def build(device_index: int, tokens_per_request: int) -> Kernel:
    """Check the header's synchronization, then compile (cached on disk) and load the FP4 kernel. Call outside graph capture."""
    check_header(HEADER.read_text())
    with torch.cuda.device(device_index):
        cubin = compile_unit(source(tokens_per_request))
        return load(cubin, symbol(tokens_per_request), (SPECIALIZED_THREADS + MATH_THREADS, 1, 1), SHARED_BYTES)


def _encode(dtype, base: int, dims, strides_bytes, box, swizzle):
    cu = driver()
    u64, u32 = cu.cuuint64_t, cu.cuuint32_t
    (tensor_map,) = check(
        cu.cuTensorMapEncodeTiled(
            dtype, len(dims), base, [u64(d) for d in dims], [u64(s) for s in strides_bytes], [u32(b) for b in box],
            [u32(1)] * len(dims), cu.CUtensorMapInterleave.CU_TENSOR_MAP_INTERLEAVE_NONE, swizzle,
            cu.CUtensorMapL2promotion.CU_TENSOR_MAP_L2_PROMOTION_L2_256B,
            cu.CUtensorMapFloatOOBfill.CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE,
        ),
        "cuTensorMapEncodeTiled(dg_logits_fp4)",
    )
    return tensor_map


_maps: dict = {}


def _tensor_maps(q4, sf_q, cache_bytes, weights):
    """(q, sf_q, kv, sf_kv, weights) tensor maps as DeepGEMM's FP4 paged launcher builds them; cached by address and shape."""
    rows = q4.shape[0] * q4.shape[1]
    blocks = cache_bytes.shape[0]
    page_bytes = cache_bytes.stride(0)
    key = (q4.data_ptr(), sf_q.data_ptr(), rows, cache_bytes.data_ptr(), blocks, page_bytes, weights.data_ptr(), weights.stride(0))
    maps = _maps.get(key)
    if maps is None:
        cu = driver()
        dt = cu.CUtensorMapDataType
        fp4, i32, f32 = dt.CU_TENSOR_MAP_DATA_TYPE_16U4_ALIGN8B, dt.CU_TENSOR_MAP_DATA_TYPE_INT32, dt.CU_TENSOR_MAP_DATA_TYPE_FLOAT32
        sw64, none = cu.CUtensorMapSwizzle.CU_TENSOR_MAP_SWIZZLE_64B, cu.CUtensorMapSwizzle.CU_TENSOR_MAP_SWIZZLE_NONE
        base = cache_bytes.data_ptr()
        # dims / boxes in elements (4-bit elements for the packed maps), strides in bytes
        tm_q = _encode(fp4, q4.data_ptr(), (HEAD_DIM, rows * HEADS), (VAL_B,), (2 * SWIZZLE, BLOCK_Q * HEADS), sw64)
        tm_sf_q = _encode(i32, sf_q.data_ptr(), (HEADS, rows), (HEADS * 4,), (HEADS, BLOCK_Q), none)
        tm_kv = _encode(fp4, base, (HEAD_DIM, PAGE, blocks), (VAL_B, page_bytes), (2 * SWIZZLE, PAGE, 1), sw64)
        tm_sf_kv = _encode(i32, base + PAGE * VAL_B, (PAGE, blocks), (page_bytes,), (PAGE, 1), none)
        tm_w = _encode(f32, weights.data_ptr(), (HEADS, rows), (weights.stride(0) * 4,), (HEADS, BLOCK_Q), none)
        maps = _maps[key] = (tm_q, tm_sf_q, tm_kv, tm_sf_kv, tm_w)
    return maps


def paged_mqa_logits(q, fused_cache, weights, context_lens, block_table, schedule, max_context_len):
    """``q`` = (q4 int8 [B, N, 32, 64] contiguous, sf_q int32 [B, N, 32] contiguous); fused_cache uint8 [blocks, 64, 1, 68] (page
    stride free); weights FP32 [B * N, 32]; context_lens [B, N] int32; block_table [B, W] int32; schedule [SMs + 1, 2] int32 from
    deep_gemm.get_paged_mqa_logits_metadata. Returns FP32 logits [B * N, max_context_len] (row stride padded as DG's)."""
    q4, sf_q = q
    batch, next_n, heads, half = q4.shape
    blocks, page, kv_heads, row = fused_cache.shape
    if (heads, half, page, kv_heads, row) != (HEADS, HEAD_DIM // 2, PAGE, 1, ROW_B) or not q4.is_contiguous():
        raise ValueError(f"dg_logits_fp4: unsupported shapes q {tuple(q4.shape)} cache {tuple(fused_cache.shape)}")
    if tuple(sf_q.shape) != (batch, next_n, heads) or sf_q.dtype != torch.int32 or not sf_q.is_contiguous():
        raise ValueError(f"dg_logits_fp4: sf_q int32 [B, N, 32] contiguous, got {sf_q.dtype} {tuple(sf_q.shape)}")
    if q4.dtype != torch.int8 or fused_cache.dtype != torch.uint8 or weights.dtype != torch.float32:
        raise ValueError("dg_logits_fp4: packed int8 query, uint8 cache, FP32 weights only")
    if fused_cache.stride(1) != ROW_B or fused_cache.stride(3) != 1 or weights.stride(1) != 1 or fused_cache.stride(0) % 16:
        raise ValueError("dg_logits_fp4: cache rows of 68 B (page stride a multiple of 16 B) and dense weight rows")
    if tuple(context_lens.shape) != (batch, next_n) or not context_lens.is_contiguous() or block_table.stride(1) != 1:
        raise ValueError("dg_logits_fp4: context_lens [B, N] contiguous, block_table unit column stride")
    rows = batch * next_n
    stride = -(-(-(-max_context_len // SPLIT_KV) * SPLIT_KV) // LOGITS_ALIGN) * LOGITS_ALIGN
    logits = torch.empty((rows, stride), device=q4.device, dtype=torch.float32)[:, :max_context_len]
    kernel = build(q4.device.index if q4.device.index is not None else torch.cuda.current_device(), next_n)
    pages = fused_cache.as_strided((blocks, page * row), (fused_cache.stride(0), 1))
    tm_q, tm_sf_q, tm_kv, tm_sf_kv, tm_w = _tensor_maps(q4, sf_q, pages, weights)
    args = [
        ctypes.c_uint32(rows), ctypes.c_uint32(stride), ctypes.c_uint32(block_table.stride(0)),
        ctypes.c_void_p(context_lens.data_ptr()), ctypes.c_void_p(logits.data_ptr()),
        ctypes.c_void_p(block_table.data_ptr()), ctypes.c_void_p(0), ctypes.c_void_p(schedule.data_ptr()),
        tm_q, tm_sf_q, tm_kv, tm_sf_kv, tm_w,
    ]
    launch(kernel, (schedule.shape[0] - 1, 1, 1), args, torch.cuda.current_stream().cuda_stream)
    return logits


_engaged = []


def verify_logits(query, gates, cache, context_lens, block_table, schedule, max_context_len, batch_requests, query_width):
    """VerifySelection's MXFP4 call form (idx_fp4.paged_logits' arguments: an idx_fp4.Fp4Query, gates [B * N, 32(, 1)], the
    [pages, 4352] MXFP4 pool) through the vendored PDL kernel."""
    import logging

    b, n = int(batch_requests), int(query_width)
    rows = b * n
    if query.q4.shape[0] != rows:
        raise ValueError(f"dg_logits_fp4: query rows {query.q4.shape[0]} != {b} x {n}")
    if not _engaged:
        _engaged.append(1)
        logging.getLogger(__name__).info("K31MK_DSA_IDX_FP4 ENGAGED site=logits_vendor (vendored PDL MXFP4 paged logits, Q%d)", n)
    return paged_mqa_logits(
        (query.q4.view(b, n, HEADS, HEAD_DIM // 2), query.sf.view(b, n, HEADS)), cache.view(-1, PAGE, 1, ROW_B),
        gates.reshape(rows, HEADS), context_lens, block_table, schedule, max_context_len,
    )
