// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// MXFP4 QFN index query as one CUDA kernel (K31MK_DSA_IDX_Q=fp4 on K31MK_DSA_QFN / QFP rows): the math of
// idx_fp4_q._qfn_q4_kernel (Triton), one CTA of 256 threads per row, PDL (griddepcontrol.wait before the first global read,
// launch_dependents after the stores), the launch form of idx_fp4_store.cuh:
//   rstd = rsqrt(sum(x^2) / 1536 + eps) over the raw q_a x = projected[row, :1536] (FP32 products, then a warp / CTA tree sum: another FP32
//          summation order than Triton's, so rstd may differ from the Triton kernel's in the last bit; the gate is test_k31_idx_fp4_qfn_cuda's)
//   q    = bf16(acc[row, h, d] * rstd) (acc = bf16(x @ wq_bg^T) [rows, 32, 128] contiguous)
//   per 32-group (head h, group g = d / 32; threads 2 (4h + g) and 2 (4h + g) + 1 hold 16 values each): sf = max(amax_32 / 6, 1e-4);
//          e = ceil_ue8m0(sf) (exponent + (mantissa != 0), clamped to [1, 254]); code = e2m1 threshold code of q / 2^(e - 127)
//          (fp4_indexer_quant's strict thresholds, sign only on a nonzero code; the instructions Triton emits: div.full) - the quantizer of
//          idx_fp4.quantize_query / index_q.cuh kQ4 / idx_fp4_store.cuh
//   q4[row, h, 64] packed e2m1 pairs (low nibble = even element), qsf[row, h] four ue8m0 bytes (byte g = group g),
//   w[row, h] = float(bf16(gate[row, h] * 32^-0.5)) * 128^-0.5 (no q scale: the MX block scales feed the logits UMMA).
// qdbg (tests only, may be null): the BF16 query this kernel quantized.
// kRstd = false (the stock chain's quantizer, idx_fp4.quantize_query's math; K31MK_DSA_IDX_FP4_QUANT=cuda): q = the BF16 query as given (acc is
//   the stock wq_b GEMM's output; no x, no rstd), the same quantizer and head weights - bit-identical to the Triton _query_kernel.
#include <sgl_kernel/tensor.h>
#include <sgl_kernel/utils.h>

#include <sgl_kernel/runtime.cuh>
#include <sgl_kernel/utils.cuh>

#include <dlpack/dlpack.h>
#include <tvm/ffi/container/tensor.h>

#include <cuda_bf16.h>
#include <cstdint>

namespace k31_idx_fp4_qfn {

constexpr int kThreads = 256, kLora = 1536, kHeads = 32, kDim = 128;

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
  const __nv_bfloat16* acc;   // [rows, 32, 128] contiguous
  const __nv_bfloat16* x;     // projected: [rows, >= 1536], unit column stride
  const __nv_bfloat16* gate;  // [rows, 32], unit column stride
  uint8_t* q4;                // [rows, 32, 64]
  uint8_t* qsf;               // [rows, 32, 4] (int32 [rows, 32])
  float* w;                   // [rows, 32]
  __nv_bfloat16* qdbg;        // tests: [rows, 32, 128] or null
  int64_t x_stride, g_stride;
  float eps;
};

template <bool kPDL, bool kRstd = true>
__global__ void __launch_bounds__(kThreads) qfn_q4_kernel(const __grid_constant__ Params p) {
  __shared__ float red[kThreads / 32];
  const int64_t row = blockIdx.x;
  const int t = threadIdx.x, lane = t & 31, warp = t >> 5;
  if constexpr (kPDL) pdl_wait();
  float rstd = 1.f;
  if constexpr (kRstd) {
    // rstd over the raw q_a row (FP32 products, no FMA contraction)
    const __nv_bfloat16* xr = p.x + row * p.x_stride;
    float ss = 0.f;
#pragma unroll
    for (int i = 0; i < kLora / kThreads; ++i) {
      const float v = __bfloat162float(xr[t + kThreads * i]);
      ss = __fadd_rn(ss, __fmul_rn(v, v));
    }
#pragma unroll
    for (int d = 16; d; d >>= 1) ss = __fadd_rn(ss, __shfl_xor_sync(0xffffffffu, ss, d));
    if (lane == 0) red[warp] = ss;
    __syncthreads();
    float tot = 0.f;
#pragma unroll
    for (int k = 0; k < kThreads / 32; ++k) tot = __fadd_rn(tot, red[k]);
    rstd = rsqrtf(__fadd_rn(div_full(tot, static_cast<float>(kLora)), p.eps));
  }
  // thread t: group g = t / 2 (head g / 4, group g % 4), half t % 2 -> query elements 16 t .. 16 t + 15 of the row
  const uint4* src = reinterpret_cast<const uint4*>(p.acc + row * (kHeads * kDim) + 16 * t);
  uint4 raw[2] = {src[0], src[1]};
  const __nv_bfloat16* a16 = reinterpret_cast<const __nv_bfloat16*>(raw);
  float q[16], amax = 0.f;
#pragma unroll
  for (int j = 0; j < 16; ++j) {
    const __nv_bfloat16 qb = kRstd ? __float2bfloat16_rn(__fmul_rn(__bfloat162float(a16[j]), rstd)) : a16[j];
    q[j] = __bfloat162float(qb);
    amax = fmaxf(amax, fabsf(q[j]));
    if (p.qdbg != nullptr) p.qdbg[row * (kHeads * kDim) + 16 * t + j] = qb;
  }
  amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, 1));
  const float sf = fmaxf(div_full(amax, 6.0f), 1.0e-4f);
  const uint32_t bits = __float_as_uint(sf);
  uint32_t ex = ((bits >> 23) & 0xffu) + ((bits & 0x7fffffu) != 0u);
  ex = ex < 1u ? 1u : (ex > 254u ? 254u : ex);
  const float scale = __uint_as_float(ex << 23);
  uint32_t lo = 0, hi = 0;
#pragma unroll
  for (int j = 0; j < 8; ++j) lo |= e2m1_code(div_full(q[j], scale)) << (4 * j);
#pragma unroll
  for (int j = 0; j < 8; ++j) hi |= e2m1_code(div_full(q[8 + j], scale)) << (4 * j);
  reinterpret_cast<uint2*>(p.q4 + row * (kHeads * kDim / 2))[t] = make_uint2(lo, hi);  // bytes 8 t .. 8 t + 7
  if ((t & 1) == 0) p.qsf[row * (kHeads * 4) + (t >> 1)] = static_cast<uint8_t>(ex);  // byte 4 h + g
  if (t < kHeads) {
    const float g = __bfloat162float(p.gate[row * p.g_stride + t]);
    const float g2 = __bfloat162float(__float2bfloat16_rn(__fmul_rn(g, 0.17677669529663687f)));  // bf16(gate * 32^-0.5)
    p.w[row * kHeads + t] = __fmul_rn(g2, 0.08838834764831845f);                              // * 128^-0.5
  }
  if constexpr (kPDL) pdl_trigger();
}

struct Binding {
  static void run(const tvm::ffi::TensorView acc, const tvm::ffi::TensorView x, const tvm::ffi::TensorView gate,
                  const tvm::ffi::TensorView q4, const tvm::ffi::TensorView qsf, const tvm::ffi::TensorView w,
                  const tvm::ffi::TensorView qdbg, const double eps, const bool pdl, const bool debug, const bool rstd) {
    using namespace sglang::host;
    const int64_t rows = acc.size(0);
    RuntimeCheck(acc.ndim() == 3 && acc.size(1) == kHeads && acc.size(2) == kDim && acc.stride(0) == kHeads * kDim && acc.stride(1) == kDim &&
                     acc.stride(2) == 1 && acc.dtype().bits == 16 && acc.dtype().code == kDLBfloat,
                 "MXFP4 QFN: acc BF16 [rows, 32, 128] contiguous");
    if (rstd)
      RuntimeCheck(x.ndim() == 2 && x.size(0) == rows && x.size(1) >= kLora && x.stride(1) == 1 && x.dtype().bits == 16 && x.dtype().code == kDLBfloat,
                   "MXFP4 QFN: projected BF16 [rows, >= 1536], unit column stride");
    RuntimeCheck(gate.ndim() == 2 && gate.size(0) == rows && gate.size(1) == kHeads && gate.stride(1) == 1 && gate.dtype().bits == 16 &&
                     gate.dtype().code == kDLBfloat,
                 "MXFP4 QFN: gate BF16 [rows, 32], unit column stride");
    RuntimeCheck(q4.ndim() == 3 && q4.size(0) >= rows && q4.size(1) == kHeads && q4.size(2) == kDim / 2 && q4.stride(0) == kHeads * kDim / 2 &&
                     q4.dtype().bits == 8,
                 "MXFP4 QFN: q4 int8 [>= rows, 32, 64] contiguous");
    RuntimeCheck(qsf.ndim() == 2 && qsf.size(0) >= rows && qsf.size(1) == kHeads && qsf.stride(0) == kHeads && qsf.dtype().bits == 32,
                 "MXFP4 QFN: sf int32 [>= rows, 32] contiguous");
    RuntimeCheck(w.size(0) >= rows && w.stride(0) == kHeads && w.dtype().bits == 32 && w.dtype().code == kDLFloat, "MXFP4 QFN: w FP32 [>= rows, 32(, 1)]");
    if (debug)
      RuntimeCheck(qdbg.ndim() == 3 && qdbg.size(0) >= rows && qdbg.stride(0) == kHeads * kDim && qdbg.dtype().bits == 16, "MXFP4 QFN: debug query");
    if (rows == 0) return;
    Params p{static_cast<const __nv_bfloat16*>(acc.data_ptr()), static_cast<const __nv_bfloat16*>(x.data_ptr()),
             static_cast<const __nv_bfloat16*>(gate.data_ptr()), static_cast<uint8_t*>(q4.data_ptr()), static_cast<uint8_t*>(qsf.data_ptr()),
             static_cast<float*>(w.data_ptr()), debug ? static_cast<__nv_bfloat16*>(qdbg.data_ptr()) : nullptr, x.stride(0), gate.stride(0),
             static_cast<float>(eps)};
    const auto kernel = rstd ? (pdl ? qfn_q4_kernel<true, true> : qfn_q4_kernel<false, true>)
                             : (pdl ? qfn_q4_kernel<true, false> : qfn_q4_kernel<false, false>);
    LaunchKernel(static_cast<uint32_t>(rows), kThreads, acc.device()).enable_pdl(pdl)(kernel, p);
  }
};

}  // namespace k31_idx_fp4_qfn
