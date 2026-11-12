# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""FP8 index query against MXFP4 index-K: the verify logits of the indexer-cache-FP4 form (``K31MK_DSA_IDX_FP4=1`` with
``K31MK_DSA_IDX_FP4_FP8Q=1``). The query and its FP32 head gates are the served FP8 ones (the q scale folded into
the gates, as deep_gemm.fp8_paged_mqa_logits reads them); only the index-K pages are MXFP4 (64 B e2m1 + 4 B ue8m0 per token, the
68 B rows of idx_fp4 / the host's K31HT_PREC_IDX_FP4 pool).

DeepGEMM has no FP8 x FP4 paged logits kernel, so this instantiates the vendored header (dg_logits.cuh) with qk_dtype_t =
cutlass::detail::float_e2m1_unpacksmem_t, the header's FP8-query x MXFP4-index-K form: the index-K TMA map is the padded 4-bit type
(16U4_ALIGN16B, .b4x16_p64, 128-byte swizzle; DeepGEMM's fp4_unpacked_smem form), smem tiles have the FP8 layout, and the block-scaled
MXF8F6F4 MMA reads A = e2m1 index-K with its ue8m0 scales and B = e4m3 query with unit scales (ue8m0 127 in every byte), so a logit
is sum_h gate[h] * relu(sum_d q[h, d] * k[d]) with k[d] = e2m1 value * 2^(ue8m0 - 127). Launch, PDL synchronization and scheduling
are the FP8 kernel's (dg_logits.py check_header; 3 query / 5 index-K stages as DeepGEMM's FP8 paged launch, since the smem tiles have
FP8 size). Tested bitwise against deep_gemm.fp8_fp4_paged_mqa_logits' FP8-MX form on the same values (e2m1 values are exact in e4m3)
and against FP64 (test_k31_dg_logits_fp4k.py).
"""

from __future__ import annotations

import ctypes
import logging
import os
import re
from functools import cache

import torch

from sgl_nvidia_mega.common.jit import Kernel, check, driver, launch, load

from .dg_logits import HEADER, LOGITS_ALIGN, MATH_THREADS, SPECIALIZED_THREADS, SPLIT_KV, SPLITS_PER_CHUNK, check_header, compile_unit

HEADS, HEAD_DIM, PAGE = 32, 128, 64
BLOCK_Q = 128 // HEADS
Q_STAGES, KV_STAGES = 3, 5
# MQALogitsSharedStorage<32, 128, MX, 4, 256, 3, 5, 3, e4m3, float>: q 3 x 16 KiB + index-K 5 x 32 KiB (1024-aligned), MX scales
# 3 x 512 B + 5 x 1 KiB, gates 3 x 512 B, 22 barriers + the TMEM pointer, rounded to the 1024-byte alignment (the source asserts it)
SHARED_BYTES = 222208
VAL_B, ROW_B = 64, 68
UNIT_SF = 0x7F7F7F7F  # four ue8m0 bytes of 127 = 2^0: the query's MX scales (its real scale is in the gates)
MAX_ROWS = 4096
QK_TYPE = "cutlass::detail::float_e2m1_unpacksmem_t"
_SYMBOL = re.compile(rb"_ZN9deep_gemm22sm100_paged_mqa_logits[A-Za-z0-9_]+")


def source(tokens_per_request: int, header: str = "dg_logits.cuh") -> str:
    """The instantiation unit: DeepGEMM's FP8 paged launch shape with MX scales and the FP8-query x MXFP4-index-K type. It follows
    the served FP8 unit's K31MK_DSA_LOGITS_EF (dg_logits.source: index-K loads, and at 2 the logits stores, L2 evict_first)."""
    ef = os.environ.get("K31MK_DSA_LOGITS_EF", "0")
    if ef not in ("0", "1", "2"):
        raise ValueError(f"K31MK_DSA_LOGITS_EF={ef!r}: want 0, 1 or 2")
    return (f"#define K31_DG_KV_EF {ef}\n" if ef != "0" else "") + f"""
#include <{header}>

using namespace deep_gemm;

static_assert(sizeof(layout::MQALogitsSharedStorage<
    {HEADS}, {HEAD_DIM}, true, {BLOCK_Q}, {SPLIT_KV}, {Q_STAGES}, {KV_STAGES}, 3, cutlass::float_e4m3_t, float
>) == {SHARED_BYTES}, "Incorrect MQA logits shared-memory size");

static void __instantiate_kernel() {{
    auto ptr = reinterpret_cast<void*>(&sm100_paged_mqa_logits<
        {tokens_per_request}, {HEADS},
        {HEAD_DIM}, {PAGE},
        true, true, false,
        {Q_STAGES}, {KV_STAGES},
        {SPLIT_KV}, {SPLITS_PER_CHUNK},
        {SPECIALIZED_THREADS}, {MATH_THREADS},
        {QK_TYPE}, float, float
    >);
}};
"""


def kernel_symbol(cubin: bytes, tokens_per_request: int) -> str:
    """The one sm100_paged_mqa_logits instantiation in ``cubin`` (read from its string tables, not re-mangled by hand): the shortest
    matching name, which every other match (its parameter / section names) must extend."""
    names = sorted({m.decode() for m in _SYMBOL.findall(cubin)}, key=len)
    names = [n for n in names if n.startswith(f"_ZN9deep_gemm22sm100_paged_mqa_logitsILj{tokens_per_request}E")]
    if not names or "float_e2m1_unpacksmem_t" not in names[0] or not all(n.startswith(names[0]) for n in names):
        raise RuntimeError(f"dg_logits_fp4k: expected one FP8q x FP4kv kernel symbol in the cubin, found {names}")
    return names[0]


_unit_sf: dict = {}


@cache
def build(device_index: int, tokens_per_request: int) -> Kernel:
    """Check the header's synchronization, compile (cached on disk) and load the kernel for ``tokens_per_request`` query rows per
    request, and allocate the device's unit query scales. Call outside CUDA-graph capture (prepare time)."""
    check_header(HEADER.read_text())
    with torch.cuda.device(device_index):
        cubin = compile_unit(source(tokens_per_request))
        if device_index not in _unit_sf:
            _unit_sf[device_index] = torch.full((MAX_ROWS, HEADS), UNIT_SF, dtype=torch.int32, device=f"cuda:{device_index}")
        return load(cubin, kernel_symbol(cubin, tokens_per_request), (SPECIALIZED_THREADS + MATH_THREADS, 1, 1), SHARED_BYTES)


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
        "cuTensorMapEncodeTiled(dg_logits_fp4k)",
    )
    return tensor_map


_maps: dict = {}


def _tensor_maps(q, sf_q, cache_bytes, weights):
    """(q, sf_q, kv, sf_kv, weights) tensor maps; cached by address and shape (graph-captured launches keep their buffers)."""
    rows = q.shape[0] * q.shape[1]
    blocks = cache_bytes.shape[0]
    page_bytes = cache_bytes.stride(0)
    key = (q.data_ptr(), sf_q.data_ptr(), rows, cache_bytes.data_ptr(), blocks, page_bytes, weights.data_ptr(), weights.stride(0))
    maps = _maps.get(key)
    if maps is None:
        cu = driver()
        dt = cu.CUtensorMapDataType
        u8, fp4, i32, f32 = (dt.CU_TENSOR_MAP_DATA_TYPE_UINT8, dt.CU_TENSOR_MAP_DATA_TYPE_16U4_ALIGN16B,
                             dt.CU_TENSOR_MAP_DATA_TYPE_INT32, dt.CU_TENSOR_MAP_DATA_TYPE_FLOAT32)
        sw128, none = cu.CUtensorMapSwizzle.CU_TENSOR_MAP_SWIZZLE_128B, cu.CUtensorMapSwizzle.CU_TENSOR_MAP_SWIZZLE_NONE
        base = cache_bytes.data_ptr()
        # dims / boxes in elements (4-bit elements for the index-K map: 128 per row = 64 B, .b4x16_p64's 64-byte box row), strides in bytes
        tm_q = _encode(u8, q.data_ptr(), (HEAD_DIM, rows * HEADS), (HEAD_DIM,), (HEAD_DIM, BLOCK_Q * HEADS), sw128)
        tm_sf_q = _encode(i32, sf_q.data_ptr(), (HEADS, rows), (HEADS * 4,), (HEADS, BLOCK_Q), none)
        tm_kv = _encode(fp4, base, (HEAD_DIM, PAGE, blocks), (VAL_B, page_bytes), (HEAD_DIM, PAGE, 1), sw128)
        tm_sf_kv = _encode(i32, base + PAGE * VAL_B, (PAGE, blocks), (page_bytes,), (PAGE, 1), none)
        tm_w = _encode(f32, weights.data_ptr(), (HEADS, rows), (weights.stride(0) * 4,), (HEADS, BLOCK_Q), none)
        maps = _maps[key] = (tm_q, tm_sf_q, tm_kv, tm_sf_kv, tm_w)
    return maps


def paged_mqa_logits(q, fused_cache, weights, context_lens, block_table, schedule, max_context_len):
    """``q`` FP8 [B, N, 32, 128] contiguous; fused_cache uint8 [blocks, 64, 1, 68] MXFP4 pages (page stride a multiple of 32 B);
    weights FP32 [B * N, 32] (the served gates); context_lens [B, N] int32; block_table [B, W] int32; schedule [SMs + 1, 2] int32
    from deep_gemm.get_paged_mqa_logits_metadata. Returns FP32 logits [B * N, max_context_len] (row stride padded as DG's)."""
    batch, next_n, heads, dim = q.shape
    blocks, page, kv_heads, row = fused_cache.shape
    if (heads, dim, page, kv_heads, row) != (HEADS, HEAD_DIM, PAGE, 1, ROW_B) or not q.is_contiguous():
        raise ValueError(f"dg_logits_fp4k: unsupported shapes q {tuple(q.shape)} cache {tuple(fused_cache.shape)}")
    if q.dtype != torch.float8_e4m3fn or fused_cache.dtype != torch.uint8 or weights.dtype != torch.float32:
        raise ValueError("dg_logits_fp4k: FP8 query, uint8 MXFP4 cache, FP32 gates only")
    # .b4x16_p64 TMA: global address and every stride 32-byte aligned
    if (fused_cache.stride(1) != ROW_B or fused_cache.stride(3) != 1 or weights.stride(1) != 1 or fused_cache.stride(0) % 32
            or fused_cache.data_ptr() % 32):
        raise ValueError("dg_logits_fp4k: cache rows of 68 B, 32-byte aligned base and page stride, dense gate rows")
    if tuple(context_lens.shape) != (batch, next_n) or not context_lens.is_contiguous() or block_table.stride(1) != 1:
        raise ValueError("dg_logits_fp4k: context_lens [B, N] contiguous, block_table unit column stride")
    rows = batch * next_n
    device = q.device.index if q.device.index is not None else torch.cuda.current_device()
    kernel = build(device, next_n)
    if rows > MAX_ROWS:
        raise ValueError(f"dg_logits_fp4k: {rows} query rows > {MAX_ROWS}")
    stride = -(-(-(-max_context_len // SPLIT_KV) * SPLIT_KV) // LOGITS_ALIGN) * LOGITS_ALIGN
    logits = torch.empty((rows, stride), device=q.device, dtype=torch.float32)[:, :max_context_len]
    pages = fused_cache.as_strided((blocks, page * row), (fused_cache.stride(0), 1))
    tm_q, tm_sf_q, tm_kv, tm_sf_kv, tm_w = _tensor_maps(q, _unit_sf[device], pages, weights)
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
    """VerifySelection's call form: the served FP8 query [B * N, 32, 128], gates [B * N, 32(, 1)] FP32, the [pages, 4352] MXFP4 pool."""
    b, n = int(batch_requests), int(query_width)
    rows = b * n
    if query.shape[0] != rows:
        raise ValueError(f"dg_logits_fp4k: query rows {query.shape[0]} != {b} x {n}")
    if not _engaged:
        _engaged.append(1)
        logging.getLogger(__name__).info("K31MK_DSA_IDX_FP4 ENGAGED site=logits_fp8q (FP8 query x MXFP4 index-K, vendored PDL, Q%d)", n)
    return paged_mqa_logits(
        query.view(b, n, HEADS, HEAD_DIM), cache.view(-1, PAGE, 1, ROW_B), gates.reshape(rows, HEADS), context_lens, block_table,
        schedule, max_context_len,
    )
