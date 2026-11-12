// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// MXFP4 index-K store (K31MK_DSA_IDX_FP4): the launch form of sglang's FP8 fused_store_index_k_cache (one warp per token
// row, 128 threads per block, PDL: griddepcontrol.wait before the first global read, launch_dependents after the stores) with the MXFP4 row
// math of idx_fp4.store_index_k / the host prefill writer (sglang fp4_indexer_quant.quantize_fp4_indexer_row, RNE off) as index_q.cuh's kKFP4
// store already reproduces it (the instructions Triton emits: div.full):
//   per 32-group g (lanes 8g .. 8g + 7 hold its 32 values): sf = max(amax_32 / 6, 1e-4); e = ceil_ue8m0(sf) (exponent + (mantissa != 0),
//   clamped to [1, 254]); code = e2m1 threshold code of k / 2^(e - 127)
//   page[slot / 64] (4352 B): codes at (slot % 64) * 64 (low nibble = even element), e at 4096 + (slot % 64) * 4 + g. A negative slot skips.
#include <sgl_kernel/tensor.h>
#include <sgl_kernel/utils.h>

#include <sgl_kernel/runtime.cuh>
#include <sgl_kernel/utils.cuh>

#include <dlpack/dlpack.h>
#include <tvm/ffi/container/tensor.h>

#include <cuda_bf16.h>
#include <cstdint>

namespace k31_idx_fp4_store {

__device__ __forceinline__ void pdl_wait() { asm volatile("griddepcontrol.wait;" ::: "memory"); }
__device__ __forceinline__ void pdl_trigger() { asm volatile("griddepcontrol.launch_dependents;" ::: "memory"); }
__device__ __forceinline__ float div_full(float a, float b) {
  float r;
  asm("div.full.f32 %0, %1, %2;\n" : "=f"(r) : "f"(a), "f"(b));
  return r;
}
// e2m1 threshold code of x (fp4_indexer_quant._fp4_e2m1_code: strict thresholds, sign only on a nonzero code).
__device__ __forceinline__ uint32_t e2m1_code(float x) {
  const float ax = fminf(fabsf(x), 6.0f);
  uint32_t idx = (ax > 0.25f) + (ax > 0.75f) + (ax > 1.25f) + (ax > 1.75f) + (ax > 2.5f) + (ax > 3.5f) + (ax > 5.0f);
  return idx | ((x < 0.f && idx != 0u) ? 8u : 0u);
}

struct Params {
  const __nv_bfloat16* key;  // [rows, 128] contiguous
  uint8_t* cache;            // [pages, 64 * 68]
  const int64_t* slots;      // [>= rows]
  uint32_t rows;
};

template <bool kPDL>
__global__ void __launch_bounds__(128) store_kernel(const __grid_constant__ Params p) {
  const uint32_t gtid = blockIdx.x * blockDim.x + threadIdx.x;
  const uint32_t row = gtid / 32, lane = threadIdx.x % 32;
  if (row >= p.rows) return;  // warp-uniform
  if constexpr (kPDL) pdl_wait();
  const int64_t slot = p.slots[row];
  const uint2 raw = reinterpret_cast<const uint2*>(p.key + static_cast<int64_t>(row) * 128)[lane];
  const __nv_bfloat16* e4 = reinterpret_cast<const __nv_bfloat16*>(&raw);
  float v[4], amax = 0.f;
#pragma unroll
  for (int j = 0; j < 4; ++j) {
    v[j] = __bfloat162float(e4[j]);
    amax = fmaxf(amax, fabsf(v[j]));
  }
#pragma unroll
  for (int d = 4; d; d >>= 1) amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, d));  // the lane's 32-group = lanes 8g .. 8g + 7
  if (slot >= 0) {
    const float sf = fmaxf(div_full(amax, 6.0f), 1.0e-4f);
    const uint32_t bits = __float_as_uint(sf);
    uint32_t ex = ((bits >> 23) & 0xffu) + ((bits & 0x7fffffu) != 0u);
    ex = ex < 1u ? 1u : (ex > 254u ? 254u : ex);
    const float scale = __uint_as_float(ex << 23);
    uint32_t packed = 0;
#pragma unroll
    for (int j = 0; j < 4; ++j) packed |= e2m1_code(div_full(v[j], scale)) << (4 * j);
    uint8_t* page = p.cache + (slot >> 6) * static_cast<int64_t>(68 * 64);
    const int off = static_cast<int>(slot & 63);
    reinterpret_cast<uint16_t*>(page + off * 64)[lane] = static_cast<uint16_t>(packed);
    if ((lane & 7u) == 0u) page[64 * 64 + off * 4 + (lane >> 3)] = static_cast<uint8_t>(ex);
  }
  if constexpr (kPDL) pdl_trigger();
}

struct Binding {
  static void run(const tvm::ffi::TensorView key, const tvm::ffi::TensorView cache, const tvm::ffi::TensorView slots, const bool pdl) {
    using namespace sglang::host;
    const int64_t rows = key.size(0);
    RuntimeCheck(key.ndim() == 2 && key.size(1) == 128 && key.stride(0) == 128 && key.stride(1) == 1 && key.dtype().bits == 16 &&
                     key.dtype().code == kDLBfloat,
                 "MXFP4 index-K store: key BF16 [rows, 128] contiguous");
    RuntimeCheck(cache.ndim() == 2 && cache.size(1) == 64 * 68 && cache.stride(0) == 64 * 68 && cache.stride(1) == 1 && cache.dtype().bits == 8,
                 "MXFP4 index-K store: pages uint8 [pages, 64 * 68]");
    RuntimeCheck(slots.ndim() == 1 && slots.size(0) >= rows && slots.stride(0) == 1 && slots.dtype().bits == 64, "int64 slots [>= rows]");
    if (rows == 0) return;
    Params p{static_cast<const __nv_bfloat16*>(key.data_ptr()), static_cast<uint8_t*>(cache.data_ptr()),
             static_cast<const int64_t*>(slots.data_ptr()), static_cast<uint32_t>(rows)};
    const uint32_t blocks = static_cast<uint32_t>((rows * 32 + 127) / 128);
    LaunchKernel(blocks, 128, key.device()).enable_pdl(pdl)(pdl ? store_kernel<true> : store_kernel<false>, p);
  }
};

}  // namespace k31_idx_fp4_store
