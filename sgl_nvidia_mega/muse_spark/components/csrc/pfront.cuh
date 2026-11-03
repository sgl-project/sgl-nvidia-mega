// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
/**
 * \brief Sequence-parallel MoE front in one pass: the pre-expert RMSNorm of a [T, 4096] bf16 row (with bf16 weight) and
 * the per-token NVFP4 quantization of its bf16 result, with the arithmetic of the two stock kernels it replaces (the
 * torch RMSNorm kernel at N = 4096 and FlashInfer's per-token NVFP4 quantizer, fast-math scale path, linear scale
 * layout), bitwise to that sequence. The normalized bf16 row is never written to memory.
 *
 * Layout: 64 threads per row, 2 rows per 128-thread CTA; row thread c holds columns c*8 + 512*k + v (k < 8, v < 8).
 * Norm (stock RMSNorm order): per thread s = sum over (k, v) in that order of x*x (sequential; x*x is exact for bf16
 * x, so fma(x, x, s) rounds as s + x*x), xor-butterfly 16, 8, 4, 2, 1 per warp, lane 0 of each of the row's 2 warps ->
 * smem, total = w0 + w1, rstd = rsqrt.approx.ftz(fma(total, 2^-12, eps)), y = bf16_rn((x * rstd) * w).
 * Quant (stock per-token order-free maxima + fixed scale arithmetic): row amax = max |y|; per-token scale = amax * gsi
 * (gsi = 1 / (e4m3 max * 6)); encode = rcp.approx.ftz(per-token scale) (0 amax: scale 0, encode FLT_MAX); per 16-element
 * block (this thread's 8 values and its xor-1 partner's): block amax, sf = e4m3_rn_satfinite(encode * (amax *
 * rcp.approx.ftz(6))), out_scale = rcp.approx.ftz(f32(sf) * rcp.approx.ftz(encode)) (0 when sf is 0), codes =
 * e2m1_rn_satfinite(y * out_scale) packed two per byte (odd element in the high nibble).
 * No barriers beyond the CTA's two __syncthreads; no waits.
 */

#include <sgl_kernel/tensor.h>
#include <sgl_kernel/utils.h>

#include <sgl_kernel/runtime.cuh>
#include <sgl_kernel/utils.cuh>

#include <cuda_bf16.h>
#include <dlpack/dlpack.h>
#include <tvm/ffi/container/tensor.h>

#include <cstdint>

namespace muse_vr {
namespace pfront {

constexpr int kHidden = 4096;
constexpr int kRowThreads = 64;
constexpr int kRowsPerCta = 2;
constexpr int kVec = 8;
constexpr int kIters = kHidden / (kRowThreads * kVec);  // 8

struct Params {
  const __nv_bfloat16* x;      // [T, 4096]
  const __nv_bfloat16* w;      // [4096]
  uint8_t* codes;              // [T, 2048]
  uint8_t* sf;                 // [T, 256] (linear)
  float* pts;                  // [T]
  int32_t rows;
  float eps;
  float gsi;                   // global scale inverse (1 / (e4m3 max * 6))
};

__device__ __forceinline__ float rsqrt_approx_ftz(float x) {
  float y;
  asm("rsqrt.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x));
  return y;
}

__device__ __forceinline__ float rcp_approx_ftz(float x) {
  float y;
  asm("rcp.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x));
  return y;
}

__device__ __forceinline__ float fmax_ptx(float a, float b) {
  float y;
  asm("max.f32 %0, %1, %2;" : "=f"(y) : "f"(a), "f"(b));
  return y;
}

__device__ __forceinline__ float mul_ptx(float a, float b) {
  float y;
  asm("mul.f32 %0, %1, %2;" : "=f"(y) : "f"(a), "f"(b));
  return y;
}

__device__ __forceinline__ uint32_t f32_to_e4m3(float a) {
  uint32_t r;
  asm("{\n\t.reg .b16 p;\n\t.reg .f32 z;\n\tmov.f32 z, 0f00000000;\n\t"
      "cvt.rn.satfinite.e4m3x2.f32 p, z, %1;\n\tcvt.u32.u16 %0, p;\n\t}"
      : "=r"(r)
      : "f"(a));
  return r;
}

__device__ __forceinline__ float out_scale_from_e4m3(uint32_t fp8, float encode) {
  float r;
  asm("{\n\t.reg .pred pz;\n\t.reg .b16 p;\n\t.reg .b32 h2;\n\t.reg .b16 lo, hi;\n\t"
      ".reg .f32 s, rg, pr, res;\n\t"
      "cvt.u16.u32 p, %1;\n\tcvt.rn.f16x2.e4m3x2 h2, p;\n\tmov.b32 {lo, hi}, h2;\n\tcvt.f32.f16 s, lo;\n\t"
      "rcp.approx.ftz.f32 rg, %2;\n\tmul.f32 pr, s, rg;\n\trcp.approx.ftz.f32 res, pr;\n\t"
      "setp.eq.f32 pz, s, 0f00000000;\n\tselp.f32 %0, 0f00000000, res, pz;\n\t}"
      : "=f"(r)
      : "r"(fp8), "f"(encode));
  return r;
}

__device__ __forceinline__ uint32_t e2m1x8(const float (&s)[kVec]) {
  uint32_t r;
  asm("{\n\t.reg .b8 b0, b1, b2, b3;\n\t"
      "cvt.rn.satfinite.e2m1x2.f32 b0, %2, %1;\n\t"
      "cvt.rn.satfinite.e2m1x2.f32 b1, %4, %3;\n\t"
      "cvt.rn.satfinite.e2m1x2.f32 b2, %6, %5;\n\t"
      "cvt.rn.satfinite.e2m1x2.f32 b3, %8, %7;\n\t"
      "mov.b32 %0, {b0, b1, b2, b3};\n\t}"
      : "=r"(r)
      : "f"(s[0]), "f"(s[1]), "f"(s[2]), "f"(s[3]), "f"(s[4]), "f"(s[5]), "f"(s[6]), "f"(s[7]));
  return r;
}

__device__ __forceinline__ float bf_lo(uint32_t w) { return __uint_as_float(w << 16); }
__device__ __forceinline__ float bf_hi(uint32_t w) { return __uint_as_float(w & 0xffff0000u); }

__device__ __forceinline__ void mul2_rn(float& d0, float& d1, float a0, float a1, float b0, float b1) {
  asm("{\n\t.reg .b64 ra, rb, rd;\n\tmov.b64 ra, {%2, %3};\n\tmov.b64 rb, {%4, %5};\n\t"
      "mul.rn.f32x2 rd, ra, rb;\n\tmov.b64 {%0, %1}, rd;\n\t}"
      : "=f"(d0), "=f"(d1)
      : "f"(a0), "f"(a1), "f"(b0), "f"(b1));
}

__device__ __forceinline__ uint32_t bf16x2_rn(float lo, float hi) {
  uint32_t r;
  asm("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(r) : "f"(hi), "f"(lo));
  return r;
}

__device__ __forceinline__ uint32_t bf16x2_max(uint32_t a, uint32_t b) {
  uint32_t r;
  asm("max.bf16x2 %0, %1, %2;" : "=r"(r) : "r"(a), "r"(b));
  return r;
}

// max of the two bf16 halves as fp32 (as the stock quantizer reduces a bf16x2 block maximum)
__device__ __forceinline__ float bf16x2_hmax_f32(uint32_t x) { return fmax_ptx(bf_lo(x), bf_hi(x)); }

// The row fragment is kept as packed bf16 words (x, then y in place). Element-wise products run as packed fp32 pairs
// (the same IEEE round-to-nearest operation per element); maxima use the stock quantizer's bf16x2 forms.
__global__ void __launch_bounds__(kRowThreads * kRowsPerCta) pfront_kernel(const __grid_constant__ Params p) {
  __shared__ float red[kRowsPerCta][2];
  __shared__ float amx[kRowsPerCta][2];
  __shared__ __align__(16) uint4 wsm[kHidden / kVec];  // the norm weight, staged once per CTA
  const int tid = threadIdx.x;
  const int r_in = tid / kRowThreads;
  const int c = tid % kRowThreads;
  const int lane = tid & 31;
  const int wr = c >> 5;  // warp within the row
  const int row = static_cast<int>(blockIdx.x) * kRowsPerCta + r_in;
  const bool live = row < p.rows;
  const int64_t rbase = static_cast<int64_t>(live ? row : 0) * kHidden;

  uint32_t xw[kIters][4];  // packed bf16 pairs: x, later y
#pragma unroll
  for (int k = 0; k < kIters; ++k) {
    const uint4 q = *reinterpret_cast<const uint4*>(p.x + rbase + c * kVec + k * kRowThreads * kVec);
    xw[k][0] = q.x;
    xw[k][1] = q.y;
    xw[k][2] = q.z;
    xw[k][3] = q.w;
  }
  // stage the weight while the row loads are in flight (ordered for the reads below by the first __syncthreads)
#pragma unroll
  for (int i = tid; i < kHidden / kVec; i += kRowThreads * kRowsPerCta) wsm[i] = reinterpret_cast<const uint4*>(p.w)[i];
  // sum of squares in the stock order; x*x is exact for bf16 x, so fma(x, x, s) rounds exactly as s + x*x
  float s = 0.f;
#pragma unroll
  for (int k = 0; k < kIters; ++k)
#pragma unroll
    for (int h = 0; h < 4; ++h) {
      const float a = bf_lo(xw[k][h]);
      const float b = bf_hi(xw[k][h]);
      s = __fmaf_rn(a, a, s);
      s = __fmaf_rn(b, b, s);
    }
#pragma unroll
  for (int off = 16; off >= 1; off >>= 1) s = __fadd_rn(s, __shfl_xor_sync(0xffffffffu, s, off));
  if (lane == 0) red[r_in][wr] = s;
  __syncthreads();
  const float total = __fadd_rn(red[r_in][0], red[r_in][1]);
  const float rstd = rsqrt_approx_ftz(__fmaf_rn(total, 1.0f / 4096.0f, p.eps));

  // y = bf16((x * rstd) * w) in place; |y| maxima per 8-vector (bf16x2 tree, unseeded) and for the thread (seeded 0)
  float vmax[kIters];
  float tmax = 0.f;
#pragma unroll
  for (int k = 0; k < kIters; ++k) {
    const uint4 q = wsm[c + k * kRowThreads];
    const uint32_t w4[4] = {q.x, q.y, q.z, q.w};
    uint32_t am[4];
#pragma unroll
    for (int h = 0; h < 4; ++h) {
      float ta, tb, ya, yb;
      mul2_rn(ta, tb, bf_lo(xw[k][h]), bf_hi(xw[k][h]), rstd, rstd);
      mul2_rn(ya, yb, ta, tb, bf_lo(w4[h]), bf_hi(w4[h]));
      xw[k][h] = bf16x2_rn(ya, yb);
      am[h] = xw[k][h] & 0x7fff7fffu;
    }
    const uint32_t m2 = bf16x2_max(bf16x2_max(am[0], am[1]), bf16x2_max(am[2], am[3]));
    vmax[k] = bf16x2_hmax_f32(m2);
    tmax = fmax_ptx(tmax, vmax[k]);
  }
#pragma unroll
  for (int off = 16; off >= 1; off >>= 1) tmax = fmax_ptx(tmax, __shfl_xor_sync(0xffffffffu, tmax, off));
  if (lane == 0) amx[r_in][wr] = tmax;
  __syncthreads();
  const float row_amax = fmax_ptx(amx[r_in][0], amx[r_in][1]);
  float pts;
  float encode;
  if (row_amax == 0.f) {
    pts = 0.f;
    encode = 3.402823466e38f;
  } else {
    pts = mul_ptx(row_amax, p.gsi);
    encode = rcp_approx_ftz(pts);
  }
  if (live && c == 0) p.pts[row] = pts;
  const float rcp6 = rcp_approx_ftz(6.0f);
#pragma unroll
  for (int k = 0; k < kIters; ++k) {
    const float bmax = fmax_ptx(vmax[k], __shfl_xor_sync(0xffffffffu, vmax[k], 1));
    const float scale = mul_ptx(encode, mul_ptx(bmax, rcp6));
    const uint32_t fp8 = f32_to_e4m3(scale);
    const float osc = out_scale_from_e4m3(fp8, encode);
    float sv[kVec];
#pragma unroll
    for (int h = 0; h < 4; ++h) mul2_rn(sv[2 * h], sv[2 * h + 1], bf_lo(xw[k][h]), bf_hi(xw[k][h]), osc, osc);
    const uint32_t packed = e2m1x8(sv);
    const int col = c * kVec + k * kRowThreads * kVec;  // first column of this thread's 8
    if (live) {
      *reinterpret_cast<uint32_t*>(p.codes + static_cast<int64_t>(row) * (kHidden / 2) + col / 2) = packed;
      if ((c & 1) == 0) p.sf[static_cast<int64_t>(row) * (kHidden / 16) + col / 16] = static_cast<uint8_t>(fp8 & 0xffu);
    }
  }
}

}  // namespace pfront

struct PFront {
  static void run(const tvm::ffi::TensorView x, const tvm::ffi::TensorView w, const tvm::ffi::TensorView codes,
                  const tvm::ffi::TensorView sf, const tvm::ffi::TensorView pts, double eps, double gsi) {
    using namespace sglang::host;
    pfront::Params prm{};
    RuntimeCheck(x.dim() == 2 && x.size(1) == pfront::kHidden && x.stride(1) == 1 && x.stride(0) == pfront::kHidden,
                 "x layout");
    RuntimeCheck(w.numel() == pfront::kHidden, "w size");
    RuntimeCheck(codes.numel() == x.size(0) * (pfront::kHidden / 2) && sf.numel() == x.size(0) * (pfront::kHidden / 16) &&
                     pts.numel() == x.size(0),
                 "output sizes");
    RuntimeCheck((reinterpret_cast<uintptr_t>(x.data_ptr()) & 15) == 0 &&
                     (reinterpret_cast<uintptr_t>(w.data_ptr()) & 15) == 0 &&
                     (reinterpret_cast<uintptr_t>(codes.data_ptr()) & 3) == 0,
                 "alignment");
    prm.x = static_cast<const __nv_bfloat16*>(x.data_ptr());
    prm.w = static_cast<const __nv_bfloat16*>(w.data_ptr());
    prm.codes = static_cast<uint8_t*>(codes.data_ptr());
    prm.sf = static_cast<uint8_t*>(sf.data_ptr());
    prm.pts = static_cast<float*>(pts.data_ptr());
    prm.rows = static_cast<int32_t>(x.size(0));
    prm.eps = static_cast<float>(eps);
    prm.gsi = static_cast<float>(gsi);
    if (prm.rows == 0) return;
    const unsigned grid = static_cast<unsigned>((prm.rows + pfront::kRowsPerCta - 1) / pfront::kRowsPerCta);
    LaunchKernel(dim3(grid), dim3(pfront::kRowThreads * pfront::kRowsPerCta), x.device(), 0)(pfront::pfront_kernel, prm);
  }
};

}  // namespace muse_vr
