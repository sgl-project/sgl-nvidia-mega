// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// K3.1 DSA index-K LayerNorm + store as ONE kernel for the QFP profiles (K31MK_DSA_QFP_KST). Replaces QFP's two launches
// (FlashInfer LayerNorm of projected[:, k_col : k_col + 128] -> fused_store_index_k_cache / idx_fp4.store_index_k) with one warp per
// row, in the launch form of prec-l2mk's idx_fp4_store.cuh (128 threads per block, PDL: griddepcontrol.wait before the first global
// read, launch_dependents after the stores). The row math is index_q.cuh's query-front store (store_index_k, kQF), copied unchanged:
//   k     = bf16(LayerNorm128(projected[row, k_col : k_col + 128]) * gamma + beta)   (FP32 math, FP32 affine; warp-shuffle sums)
//   FP8   : scale = max(amax_128(k), 1e-4) / 448; page[slot / 64]: e4m3 codes at (slot % 64) * 128, scale at 8192 + (slot % 64) * 4
//   MXFP4 : per 32-group sf = max(amax_32 / 6, 1e-4), e = ceil_ue8m0(sf); e2m1 codes at (slot % 64) * 64, e at 4096 + (slot % 64) * 4 + g
// So the bytes equal the query front's index-K store for the same rows (bitwise), and differ from FlashInfer's LayerNorm path only by
// the FP32 summation order of the LayerNorm (KL class, like QF vs stock). A negative slot skips the row.
#include <sgl_kernel/tensor.h>
#include <sgl_kernel/utils.h>

#include <sgl_kernel/runtime.cuh>
#include <sgl_kernel/utils.cuh>

#include <dlpack/dlpack.h>
#include <tvm/ffi/container/tensor.h>

#include <cuda_bf16.h>
#include <cstdint>

namespace k31_index_kst {

__device__ __forceinline__ void pdl_wait() { asm volatile("griddepcontrol.wait;" ::: "memory"); }
__device__ __forceinline__ void pdl_trigger() { asm volatile("griddepcontrol.launch_dependents;" ::: "memory"); }
__device__ __forceinline__ float bf16_round(float x) { return __bfloat162float(__float2bfloat16_rn(x)); }
__device__ __forceinline__ float div_full(float a, float b) {
  float r;
  asm("div.full.f32 %0, %1, %2;\n" : "=f"(r) : "f"(a), "f"(b));
  return r;
}
__device__ __forceinline__ uint8_t to_e4m3(float x) {  // index_q.cuh's conversion
  uint16_t pair;
  asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;\n" : "=h"(pair) : "f"(0.f), "f"(x));
  return static_cast<uint8_t>(pair & 0xff);
}
// e2m1 threshold code of x (fp4_indexer_quant._fp4_e2m1_code: strict thresholds, sign only on a nonzero code).
__device__ __forceinline__ uint32_t e2m1_code(float x) {
  const float ax = fminf(fabsf(x), 6.0f);
  uint32_t idx = (ax > 0.25f) + (ax > 0.75f) + (ax > 1.25f) + (ax > 1.75f) + (ax > 2.5f) + (ax > 3.5f) + (ax > 5.0f);
  return idx | ((x < 0.f && idx != 0u) ? 8u : 0u);
}

struct Params {
  const __nv_bfloat16* projected;  // [rows, pitch]; index-K columns [k_col, k_col + 128)
  int64_t pitch;
  int k_col;
  uint32_t rows;
  const int64_t* slots;  // [>= rows]; a negative slot skips the row
  uint8_t* kcache;       // FP8 [pages, 64 * 132] or MXFP4 [pages, 64 * 68]
  const float* kgamma;   // [128] FP32
  const float* kbeta;    // [128] FP32
  float keps;
};

template <bool kKFP4, bool kPDL>
__global__ void __launch_bounds__(128) kst_kernel(const __grid_constant__ Params p) {
  const uint32_t gtid = blockIdx.x * blockDim.x + threadIdx.x;
  const uint32_t row = gtid / 32;
  const int lane = static_cast<int>(threadIdx.x % 32);
  if (row >= p.rows) return;  // warp-uniform
  if constexpr (kPDL) pdl_wait();
  const int64_t slot = p.slots[row];
  const uint2 raw = reinterpret_cast<const uint2*>(p.projected + static_cast<int64_t>(row) * p.pitch + p.k_col)[lane];
  // ---- index_q.cuh store_index_k (kQF) row math, unchanged ----
  const __nv_bfloat16* e = reinterpret_cast<const __nv_bfloat16*>(&raw);
  float v[4], sum = 0.f;
#pragma unroll
  for (int j = 0; j < 4; ++j) {
    v[j] = __bfloat162float(e[j]);
    sum += v[j];
  }
#pragma unroll
  for (int d = 16; d; d >>= 1) sum += __shfl_xor_sync(0xffffffffu, sum, d);
  const float mean = sum * (1.0f / 128.0f);
  float var = 0.f;
#pragma unroll
  for (int j = 0; j < 4; ++j) {
    v[j] -= mean;
    var = fmaf(v[j], v[j], var);
  }
#pragma unroll
  for (int d = 16; d; d >>= 1) var += __shfl_xor_sync(0xffffffffu, var, d);
  const float rstd = rsqrtf(var * (1.0f / 128.0f) + p.keps);
  float amax = 0.f;
#pragma unroll
  for (int j = 0; j < 4; ++j) {
    const int col = lane * 4 + j;
    v[j] = bf16_round(v[j] * rstd * p.kgamma[col] + p.kbeta[col]);
    amax = fmaxf(amax, fabsf(v[j]));
  }
  if constexpr (kKFP4) {
    // the lane's 32-group = lanes 8g .. 8g + 7
#pragma unroll
    for (int d = 4; d; d >>= 1) amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, d));
    if (slot >= 0) {
      const float sf = fmaxf(div_full(amax, 6.0f), 1.0e-4f);
      const uint32_t bits = __float_as_uint(sf);
      uint32_t ex = ((bits >> 23) & 0xffu) + ((bits & 0x7fffffu) != 0u);
      ex = ex < 1u ? 1u : (ex > 254u ? 254u : ex);
      const float scale = __uint_as_float(ex << 23);
      uint32_t packed = 0;
#pragma unroll
      for (int j = 0; j < 4; ++j) packed |= e2m1_code(div_full(v[j], scale)) << (4 * j);
      uint8_t* page = p.kcache + (slot >> 6) * static_cast<int64_t>(68 * 64);
      const int off = static_cast<int>(slot & 63);
      reinterpret_cast<uint16_t*>(page + off * 64)[lane] = static_cast<uint16_t>(packed);
      if ((lane & 7) == 0) page[64 * 64 + off * 4 + (lane >> 3)] = static_cast<uint8_t>(ex);
    }
  } else {
#pragma unroll
    for (int d = 16; d; d >>= 1) amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, d));
    if (slot >= 0) {
      const float scale = __fdiv_rn(fmaxf(1e-4f, amax), 448.0f);
      const float inv = __fdiv_rn(1.0f, scale);
      uint8_t* page = p.kcache + (slot >> 6) * static_cast<int64_t>(132 * 64);
      const int off = static_cast<int>(slot & 63);
      uint32_t packed = 0;
#pragma unroll
      for (int j = 0; j < 4; ++j)
        packed |= static_cast<uint32_t>(to_e4m3(fminf(fmaxf(v[j] * inv, -448.f), 448.f))) << (8 * j);
      reinterpret_cast<uint32_t*>(page + off * 128)[lane] = packed;
      if (lane == 0) *reinterpret_cast<float*>(page + 128 * 64 + off * 4) = scale;
    }
  }
  if constexpr (kPDL) pdl_trigger();
}

struct Binding {
  static void run(const tvm::ffi::TensorView projected, const int64_t k_col, const tvm::ffi::TensorView slots,
                  const tvm::ffi::TensorView kcache, const tvm::ffi::TensorView kgamma, const tvm::ffi::TensorView kbeta,
                  const double keps, const bool pdl) {
    using namespace sglang::host;
    const int64_t rows = projected.size(0);
    RuntimeCheck(projected.ndim() == 2 && projected.stride(1) == 1 && projected.stride(0) % 4 == 0 && projected.dtype().bits == 16 &&
                     projected.dtype().code == kDLBfloat,
                 "index-K store: projected BF16 [rows, pitch], 8-byte aligned rows");
    RuntimeCheck(k_col >= 0 && k_col % 4 == 0 && k_col + 128 <= projected.size(1), "index-K columns");
    RuntimeCheck(slots.ndim() == 1 && slots.size(0) >= rows && slots.stride(0) == 1 && slots.dtype().bits == 64, "int64 slots [>= rows]");
    const bool kfp4 = kcache.ndim() == 2 && kcache.size(1) == 64 * 68 && kcache.stride(0) == 64 * 68 && kcache.stride(1) == 1;
    RuntimeCheck(kfp4 || (kcache.ndim() == 2 && kcache.size(1) == 64 * 132 && kcache.stride(0) == 64 * 132 && kcache.stride(1) == 1),
                 "index-K pages [pages, 64 * 132] (or MXFP4 [pages, 64 * 68])");
    RuntimeCheck(kcache.dtype().bits == 8, "uint8 index-K pages");
    RuntimeCheck(kgamma.size(0) == 128 && kbeta.size(0) == 128 && kgamma.dtype().bits == 32 && kbeta.dtype().bits == 32 &&
                     kgamma.stride(0) == 1 && kbeta.stride(0) == 1,
                 "FP32 index-K LayerNorm");
    if (rows == 0) return;
    Params p{};
    p.projected = static_cast<const __nv_bfloat16*>(projected.data_ptr());
    p.pitch = projected.stride(0);
    p.k_col = static_cast<int>(k_col);
    p.rows = static_cast<uint32_t>(rows);
    p.slots = static_cast<const int64_t*>(slots.data_ptr());
    p.kcache = static_cast<uint8_t*>(kcache.data_ptr());
    p.kgamma = static_cast<const float*>(kgamma.data_ptr());
    p.kbeta = static_cast<const float*>(kbeta.data_ptr());
    p.keps = static_cast<float>(keps);
    const uint32_t blocks = static_cast<uint32_t>((rows * 32 + 127) / 128);
    auto kernel = kfp4 ? (pdl ? kst_kernel<true, true> : kst_kernel<true, false>) : (pdl ? kst_kernel<false, true> : kst_kernel<false, false>);
    LaunchKernel(blocks, 128, projected.device()).enable_pdl(pdl)(kernel, p);
  }
};

}  // namespace k31_index_kst
