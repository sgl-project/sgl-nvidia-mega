// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// FP8 QFN index query as one CUDA kernel (K31MK_DSA_QFN_CUDA=1, default off): the outputs of
// index_qfn._qfn_quant_head_gate_kernel (Triton, the served kernel and the reference), one CTA of 256 threads per row, PDL
// (griddepcontrol.wait before the first global read, launch_dependents after the last store; the launch form of idx_fp4_qfn.cuh):
//   rstd = rsqrt(sum(x^2) / 1536 + eps) over the raw q_a x = projected[row, :1536], computed with the instructions of the Triton kernel's PTX
//          (Triton 3.9, FP fusion on; dumped by test_k31_idx_qfn8_cuda.py): 8 elements e0..e7 per thread in its BLOCK_X 2048 /
//          8-warp layout (contiguous 8 t .. 8 t + 7 when the row is 16-byte aligned in Triton's specialization sense: pointer % 16 == 0 and
//          row stride % 16 == 0; else t + 256 j), in-thread f32x2 tree (fma(e0, e0, e2^2) + fma(e4, e4, e6^2)) + (fma(e1, e1, e3^2) +
//          fma(e5, e5, e7^2)), warp butterfly 16..1, butterfly over the 8 warp partials; then rsqrt.approx.ftz(fma(div.approx(1, 1536), sum, eps))
//   q    = bf16(acc[row, h, d] * rstd)  (acc = bf16(x @ wq_bg^T) [rows, 32, 128] contiguous)
//   per head h (threads 8h .. 8h + 7, 16 elements each): amax = max(max |q|, 1e-4); scale = amax * fp32(1 / 448);
//          ue8m0 (round_scale): scale = exp2f(ceilf(log2f(scale))) (tl.log2 lowers to libdevice's log2f, not lg2.approx);
//          q8 = e4m3 rn satfinite of min(max(div.full(q, scale), -448), 448)
//   w[row, h] = float(bf16(gate[row, h] * fp32(32^-0.5))) * scale * fp32(128^-0.5)
// Variants (K31_QFN8_VARIANTS builds only, tests): other rstd summation orders / tails and scale roundings, to show the bitwise gate sees them.
// qdbg / rdbg (tests only, may be null): the BF16 query and the rstd this kernel used.
#include <sgl_kernel/tensor.h>
#include <sgl_kernel/utils.h>

#include <sgl_kernel/runtime.cuh>
#include <sgl_kernel/utils.cuh>

#include <dlpack/dlpack.h>
#include <tvm/ffi/container/tensor.h>

#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cstdint>

#ifndef K31_QFN8_VARIANTS
#define K31_QFN8_VARIANTS 0
#endif

namespace k31_idx_qfn8 {

constexpr int kThreads = 256, kLora = 1536, kHeads = 32, kDim = 128;

__device__ __forceinline__ void pdl_wait() { asm volatile("griddepcontrol.wait;" ::: "memory"); }
__device__ __forceinline__ void pdl_trigger() { asm volatile("griddepcontrol.launch_dependents;" ::: "memory"); }
__device__ __forceinline__ float div_full(float a, float b) {
  float r;
  asm("div.full.f32 %0, %1, %2;\n" : "=f"(r) : "f"(a), "f"(b));
  return r;
}
__device__ __forceinline__ float ceil_pow2_approx(float s) {  // ex2.approx(cvt.rpi(lg2.approx(s)))
  float l, c, e;
  asm("lg2.approx.f32 %0, %1;\n" : "=f"(l) : "f"(s));
  asm("cvt.rpi.f32.f32 %0, %1;\n" : "=f"(c) : "f"(l));
  asm("ex2.approx.f32 %0, %1;\n" : "=f"(e) : "f"(c));
  return e;
}

struct Params {
  const __nv_bfloat16* acc;   // [rows, 32, 128] contiguous
  const __nv_bfloat16* x;     // projected: [rows, >= 1536], unit column stride
  const __nv_bfloat16* gate;  // [rows, 32], unit column stride
  uint8_t* q8;                // [rows, 32, 128] e4m3, contiguous
  float* w;                   // [rows, 32(, 1)] contiguous
  __nv_bfloat16* qdbg;        // tests: [rows, 32, 128] or null
  float* rdbg;                // tests: [rows] or null
  int64_t x_stride, g_stride;
  float eps;
  int vec8;         // the Triton kernel's row-load layout (see the header)
  int round_scale;  // ue8m0 scales (scale_fmt is not None)
  int inner, cross, tail, scl;  // variants (K31_QFN8_VARIANTS builds): 0, 0, 0, 0 = the served form
};

template <bool kPDL, bool kVariants>
__global__ void __launch_bounds__(kThreads) qfn8_kernel(const __grid_constant__ Params p) {
  __shared__ float red[kThreads / 32];
  const int64_t row = blockIdx.x;
  const int t = threadIdx.x, lane = t & 31, warp = t >> 5;
  const int inner = kVariants ? p.inner : 0, cross = kVariants ? p.cross : 0, tail = kVariants ? p.tail : 0, scl = kVariants ? p.scl : 0;
  if constexpr (kPDL) pdl_wait();
  // ---- rstd: the 8 values this thread holds in the Triton layout, masked to 0 past column 1535
  const __nv_bfloat16* xr = p.x + row * p.x_stride;
  float v[8];
#pragma unroll
  for (int i = 0; i < 8; ++i) {
    const int col = p.vec8 ? 8 * t + i : t + kThreads * i;
    v[i] = col < kLora ? __bfloat162float(xr[col]) : 0.f;
  }
  float s;
  if (inner == 0) {  // Triton's in-thread f32x2 tree (lanes (e0, e1) (e2, e3) (e4, e5) (e6, e7))
    const float a0 = __fmaf_rn(v[0], v[0], __fmul_rn(v[2], v[2])), a1 = __fmaf_rn(v[1], v[1], __fmul_rn(v[3], v[3]));
    const float b0 = __fmaf_rn(v[4], v[4], __fmul_rn(v[6], v[6])), b1 = __fmaf_rn(v[5], v[5], __fmul_rn(v[7], v[7]));
    s = __fadd_rn(__fadd_rn(a0, b0), __fadd_rn(a1, b1));
  } else if (inner == 1) {  // variant: a contracted chain fma(x0, x0, x1 * x1), then fma(xi, xi, acc)
    s = __fmaf_rn(v[0], v[0], __fmul_rn(v[1], v[1]));
#pragma unroll
    for (int i = 2; i < 8; ++i) s = __fmaf_rn(v[i], v[i], s);
  } else if (inner == 2) {  // variant: no contraction
    s = __fmul_rn(v[0], v[0]);
#pragma unroll
    for (int i = 1; i < 8; ++i) s = __fadd_rn(s, __fmul_rn(v[i], v[i]));
  } else {  // variant: pairwise tree, no contraction
    float a[8];
#pragma unroll
    for (int i = 0; i < 8; ++i) a[i] = __fmul_rn(v[i], v[i]);
    s = __fadd_rn(__fadd_rn(__fadd_rn(a[0], a[1]), __fadd_rn(a[2], a[3])), __fadd_rn(__fadd_rn(a[4], a[5]), __fadd_rn(a[6], a[7])));
  }
#pragma unroll
  for (int d = 16; d; d >>= 1) s = __fadd_rn(s, __shfl_xor_sync(0xffffffffu, s, d));
  if (lane == 0) red[warp] = s;
  __syncthreads();
  float tot;
  if (cross == 0) {  // butterfly over the 8 warp partials (xor 4, 2, 1)
    tot = __fadd_rn(__fadd_rn(__fadd_rn(red[0], red[4]), __fadd_rn(red[2], red[6])),
                    __fadd_rn(__fadd_rn(red[1], red[5]), __fadd_rn(red[3], red[7])));
  } else {  // variant: sequential
    tot = red[0];
#pragma unroll
    for (int k = 1; k < kThreads / 32; ++k) tot = __fadd_rn(tot, red[k]);
  }
  float rin;
  if (tail == 0) {  // Triton: div.approx(1, 1536) then one fma with eps
    float c;
    asm("div.approx.f32 %0, %1, %2;\n" : "=f"(c) : "f"(1.0f), "f"(static_cast<float>(kLora)));
    rin = __fmaf_rn(c, tot, p.eps);
  } else {  // variant: div.full + add
    rin = __fadd_rn(div_full(tot, static_cast<float>(kLora)), p.eps);
  }
  float rstd;
  asm("rsqrt.approx.ftz.f32 %0, %1;\n" : "=f"(rstd) : "f"(rin));
  if (p.rdbg != nullptr && t == 0) p.rdbg[row] = rstd;
  // ---- quantize: thread t holds elements 16 t .. 16 t + 15 of the row (head t / 8)
  const uint4* src = reinterpret_cast<const uint4*>(p.acc + row * (kHeads * kDim) + 16 * t);
  uint4 raw[2] = {src[0], src[1]};
  const __nv_bfloat16* a16 = reinterpret_cast<const __nv_bfloat16*>(raw);
  float q[16], amax = 0.f;
#pragma unroll
  for (int j = 0; j < 16; ++j) {
    const __nv_bfloat16 qb = __float2bfloat16_rn(__fmul_rn(__bfloat162float(a16[j]), rstd));
    q[j] = __bfloat162float(qb);
    amax = fmaxf(amax, fabsf(q[j]));
    if (p.qdbg != nullptr) p.qdbg[row * (kHeads * kDim) + 16 * t + j] = qb;
  }
  amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, 1));
  amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, 2));
  amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, 4));
  amax = fmaxf(amax, 1.0e-4f);
  float scale = __fmul_rn(amax, __uint_as_float(0x3B124925u));  // fp32(1 / 448)
  if (p.round_scale) {
    if (scl == 0) {  // Triton: libdevice log2f, cvt.rpi, ex2
      scale = exp2f(ceilf(log2f(scale)));
    } else if (scl == 1) {  // variant: exact ceil to a power of two (normal scales)
      const uint32_t b = __float_as_uint(scale);
      scale = __uint_as_float((((b >> 23) & 0xffu) + ((b & 0x7fffffu) != 0u)) << 23);
    } else {  // variant: lg2.approx
      scale = ceil_pow2_approx(scale);
    }
  }
  uint32_t packed[4];
#pragma unroll
  for (int j = 0; j < 8; ++j) {
    const float lo = fminf(fmaxf(div_full(q[2 * j], scale), -448.f), 448.f);
    const float hi = fminf(fmaxf(div_full(q[2 * j + 1], scale), -448.f), 448.f);
    const uint32_t pair = static_cast<uint32_t>(__nv_cvt_float2_to_fp8x2(make_float2(lo, hi), __NV_SATFINITE, __NV_E4M3));
    if (j & 1) packed[j >> 1] |= pair << 16; else packed[j >> 1] = pair;
  }
  reinterpret_cast<uint4*>(p.q8 + row * (kHeads * kDim))[t] = make_uint4(packed[0], packed[1], packed[2], packed[3]);
  if ((t & 7) == 0) {
    const int h = t >> 3;
    const float g = __bfloat162float(p.gate[row * p.g_stride + h]);
    const float g2 = __bfloat162float(__float2bfloat16_rn(__fmul_rn(g, 0.17677669529663687f)));  // bf16(gate * 32^-0.5)
    p.w[row * kHeads + h] = __fmul_rn(__fmul_rn(g2, scale), 0.08838834764831845f);              // * scale * 128^-0.5
  }
  if constexpr (kPDL) pdl_trigger();
}

struct Binding {
  static void run(const tvm::ffi::TensorView acc, const tvm::ffi::TensorView x, const tvm::ffi::TensorView gate,
                  const tvm::ffi::TensorView q8, const tvm::ffi::TensorView w, const tvm::ffi::TensorView qdbg,
                  const tvm::ffi::TensorView rdbg, const double eps, const bool pdl, const bool debug, const bool round_scale,
                  const int64_t inner, const int64_t cross, const int64_t tail, const int64_t scl) {
    using namespace sglang::host;
    const int64_t rows = acc.size(0);
    RuntimeCheck(acc.ndim() == 3 && acc.size(1) == kHeads && acc.size(2) == kDim && acc.stride(0) == kHeads * kDim && acc.stride(1) == kDim &&
                     acc.stride(2) == 1 && acc.dtype().bits == 16 && acc.dtype().code == kDLBfloat,
                 "FP8 QFN: acc BF16 [rows, 32, 128] contiguous");
    RuntimeCheck(reinterpret_cast<uintptr_t>(acc.data_ptr()) % 16 == 0, "FP8 QFN: acc 16-byte aligned");
    RuntimeCheck(x.ndim() == 2 && x.size(0) == rows && x.size(1) >= kLora && x.stride(1) == 1 && x.dtype().bits == 16 && x.dtype().code == kDLBfloat,
                 "FP8 QFN: projected BF16 [rows, >= 1536], unit column stride");
    RuntimeCheck(gate.ndim() == 2 && gate.size(0) == rows && gate.size(1) == kHeads && gate.stride(1) == 1 && gate.dtype().bits == 16 &&
                     gate.dtype().code == kDLBfloat,
                 "FP8 QFN: gate BF16 [rows, 32], unit column stride");
    RuntimeCheck(q8.ndim() == 3 && q8.size(0) == rows && q8.size(1) == kHeads && q8.size(2) == kDim && q8.stride(0) == kHeads * kDim &&
                     q8.stride(1) == kDim && q8.stride(2) == 1 && q8.dtype().bits == 8,
                 "FP8 QFN: q8 e4m3 [rows, 32, 128] contiguous");
    RuntimeCheck(reinterpret_cast<uintptr_t>(q8.data_ptr()) % 16 == 0, "FP8 QFN: q8 16-byte aligned");
    RuntimeCheck(w.size(0) == rows && w.size(1) == kHeads && w.stride(0) == kHeads && w.dtype().bits == 32 && w.dtype().code == kDLFloat,
                 "FP8 QFN: w FP32 [rows, 32(, 1)] contiguous");
    if (debug)
      RuntimeCheck(qdbg.ndim() == 3 && qdbg.size(0) >= rows && qdbg.stride(0) == kHeads * kDim && qdbg.dtype().bits == 16 && rdbg.size(0) >= rows &&
                       rdbg.dtype().bits == 32,
                   "FP8 QFN: debug query / rstd");
    RuntimeCheck(K31_QFN8_VARIANTS || (inner == 0 && cross == 0 && tail == 0 && scl == 0), "FP8 QFN: variants need a K31_QFN8_VARIANTS build");
    if (rows == 0) return;
    // Triton's specialization of the row load: 16-byte aligned rows only when the pointer and the row stride (elements) divide by 16
    const int vec8 = (reinterpret_cast<uintptr_t>(x.data_ptr()) % 16 == 0) && (x.stride(0) % 16 == 0);
    Params p{static_cast<const __nv_bfloat16*>(acc.data_ptr()), static_cast<const __nv_bfloat16*>(x.data_ptr()),
             static_cast<const __nv_bfloat16*>(gate.data_ptr()), static_cast<uint8_t*>(q8.data_ptr()), static_cast<float*>(w.data_ptr()),
             debug ? static_cast<__nv_bfloat16*>(qdbg.data_ptr()) : nullptr, debug ? static_cast<float*>(rdbg.data_ptr()) : nullptr,
             x.stride(0), gate.stride(0), static_cast<float>(eps), vec8, round_scale ? 1 : 0, static_cast<int>(inner),
             static_cast<int>(cross), static_cast<int>(tail), static_cast<int>(scl)};
#if K31_QFN8_VARIANTS
    const auto kernel = pdl ? qfn8_kernel<true, true> : qfn8_kernel<false, true>;
#else
    const auto kernel = pdl ? qfn8_kernel<true, false> : qfn8_kernel<false, false>;
#endif
    LaunchKernel(static_cast<uint32_t>(rows), kThreads, acc.device()).enable_pdl(pdl)(kernel, p);
  }
};

}  // namespace k31_idx_qfn8
