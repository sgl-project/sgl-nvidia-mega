// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
/**
 * \brief Prefill QK path in one pass: per-head RMS norm of the q and k heads of a QKV projection output, RoPE, q -> e4m3,
 * and the e4m3 K / V cache store, with the arithmetic of SGLang's fused QK-norm Triton kernel followed by its fused FP8
 * q / KV-cache store (bitwise to that two-kernel sequence: 8 columns per thread, the Triton kernel's sum-of-squares order).
 *
 * Layout: one warp per token; each pass covers 4 head rows of the token (8 lanes per 64-column row, 8 columns per lane);
 * the RoPE cosines / sines of the token are loaded once per lane (its 4 frequencies) and reused for every q / k row.
 * Rows: q heads, then k heads, then v heads (v rows are stored unnormalized). Every token's K / V rows are stored at its
 * cache location, including location 0 (as the stock FP8 store does).
 * No barriers; no waits.
 */

#include <sgl_kernel/tensor.h>
#include <sgl_kernel/utils.h>

#include <sgl_kernel/runtime.cuh>
#include <sgl_kernel/utils.cuh>

#include <cuda_bf16.h>
#include <dlpack/dlpack.h>
#include <tvm/ffi/container/tensor.h>

#include <cstdint>

// Arithmetic form: 0 = SGLang's fused QK-norm Triton kernel as compiled with FP fusion (fma contractions, the mean as
// div.full), 1 = as compiled without it (every product and sum rounded separately, the mean as a product with
// div.approx(1, 64)). Set by the host build of the module (ops/prefill_qkn.py; ops/swa_verify_attn.py qk_arith).
#ifndef MUSE_VR_QK_UNFUSED
#define MUSE_VR_QK_UNFUSED 0
#endif

namespace muse_vr {
#if MUSE_VR_QK_UNFUSED
// The unfused build gets its own symbols, so both builds can be loaded in one process.
inline namespace qk_unfused {
#endif
namespace pqk {

constexpr int kHeadDim = 64;
constexpr int kMaxPasses = 16;  // up to 64 head rows per token

struct Params {
  const __nv_bfloat16* __restrict__ qkv;  // [T, >= (Hq + 2 Hkv) * 64]
  int64_t qkv_token_stride;               // elements
  const int64_t* __restrict__ positions;  // [T]
  const float* __restrict__ cos_sin;      // [max position, 64]: cos in columns 0..31, sin in 32..63
  const int64_t* __restrict__ cache_loc;  // [T] pool row of each token
  uint8_t* __restrict__ k_pool;           // [slots, Hkv * 64] e4m3
  uint8_t* __restrict__ v_pool;
  int64_t pool_row_stride;  // bytes
  uint8_t* __restrict__ q_out;  // [T, Hq * 64] e4m3 (store mode)
  __nv_bfloat16* __restrict__ q_bf;  // norm-only mode: [T, Hq * 64] bf16 normed (+ RoPE) q
  __nv_bfloat16* __restrict__ k_bf;  // norm-only mode: [T, Hkv * 64] bf16 normed (+ RoPE) k
  int32_t tokens;
  int32_t num_q_heads;
  int32_t num_kv_heads;
  float eps;
};

// two values rounded to bf16 (nearest even) with one packed conversion, returned as fp32
__device__ __forceinline__ void bf16_round2(float& a, float& b) {
  uint32_t d;
  asm("cvt.rn.bf16x2.f32 %0, %1, %2;\n" : "=r"(d) : "f"(b), "f"(a));
  a = __uint_as_float(d << 16);
  b = __uint_as_float(d & 0xffff0000u);
}

// sum of squares of 8 consecutive columns in the Triton kernel's order (even / odd columns, pairwise)
__device__ __forceinline__ float row_sumsq8(const float (&x)[8]) {
  const float even = __fadd_rn(__fadd_rn(__fmul_rn(x[0], x[0]), __fmul_rn(x[2], x[2])),
                               __fadd_rn(__fmul_rn(x[4], x[4]), __fmul_rn(x[6], x[6])));
  const float odd = __fadd_rn(__fadd_rn(__fmul_rn(x[1], x[1]), __fmul_rn(x[3], x[3])),
                              __fadd_rn(__fmul_rn(x[5], x[5]), __fmul_rn(x[7], x[7])));
  return __fadd_rn(even, odd);
}

__device__ __forceinline__ float qk_rstd(float ss, float eps) {
  float var, r;
  if constexpr (MUSE_VR_QK_UNFUSED != 0) {
    float inv;
    asm("div.approx.f32 %0, %1, %2;\n" : "=f"(inv) : "f"(1.0f), "f"(64.0f));
    var = __fmul_rn(inv, ss);
  } else {
    asm("div.full.f32 %0, %1, %2;\n" : "=f"(var) : "f"(ss), "f"(64.0f));
  }
  const float t = __fadd_rn(eps, var);
  asm("rsqrt.approx.ftz.f32 %0, %1;\n" : "=f"(r) : "f"(t));
  return r;
}

// e4m3(x0), e4m3(x1) (nearest even, satfinite) as two bytes {x0 low, x1 high}
__device__ __forceinline__ uint32_t e4m3x2(float x0, float x1) {
  uint16_t e;
  asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;\n" : "=h"(e) : "f"(x1), "f"(x0));
  return e;
}

// kStore: e4m3 q + e4m3 K / V store (the QK norm and the backend's FP8 store fused); else norm-only: bf16 q and k exactly as
// the Triton kernel's no-store launch writes them (the backend's FP8 store then runs as in stock).
template <bool kRope, int kPasses, bool kStore>
__global__ void __launch_bounds__(128) prefill_qkn_kernel(const Params p) {
  const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
  const int tok = blockIdx.x * 4 + warp;
  if (tok >= p.tokens) return;  // whole warp
  const int li = lane & 7, rsub = lane >> 3;
  const int nq = p.num_q_heads, nkv = p.num_kv_heads, heads = kStore ? nq + 2 * nkv : nq + nkv;
  const __nv_bfloat16* const src = p.qkv + static_cast<int64_t>(tok) * p.qkv_token_stride + 8 * li;

  float cs[8];  // cos of frequencies 4 li .. 4 li + 3, then their sin
  if constexpr (kRope) {
    const int64_t pos = p.positions[tok];
    const float4 c = *reinterpret_cast<const float4*>(p.cos_sin + pos * kHeadDim + 4 * li);
    const float4 s = *reinterpret_cast<const float4*>(p.cos_sin + pos * kHeadDim + kHeadDim / 2 + 4 * li);
    cs[0] = c.x; cs[1] = c.y; cs[2] = c.z; cs[3] = c.w;
    cs[4] = s.x; cs[5] = s.y; cs[6] = s.z; cs[7] = s.w;
  }
  const int64_t loc = kStore ? p.cache_loc[tok] : 0;

  uint4 w[kPasses];
#pragma unroll
  for (int ps = 0; ps < kPasses; ++ps) {
    const int h = 4 * ps + rsub;
    w[ps] = h < heads ? *reinterpret_cast<const uint4*>(src + h * kHeadDim) : make_uint4(0, 0, 0, 0);
  }
#pragma unroll
  for (int ps = 0; ps < kPasses; ++ps) {
    const int h = 4 * ps + rsub;
    const bool is_q = h < nq, is_k = h >= nq && h < nq + nkv, is_v = h >= nq + nkv && h < heads;
    float x[8];
    {
      const uint32_t u[4] = {w[ps].x, w[ps].y, w[ps].z, w[ps].w};
#pragma unroll
      for (int i = 0; i < 4; ++i) {
        const float2 f = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&u[i]));
        x[2 * i] = f.x;
        x[2 * i + 1] = f.y;
      }
    }
    float ss = row_sumsq8(x);
#pragma unroll
    for (int m = 4; m >= 1; m >>= 1) ss = __fadd_rn(ss, __shfl_xor_sync(0xffffffffu, ss, m));
    const float r = qk_rstd(ss, p.eps);
    float y[8];
#pragma unroll
    for (int e = 0; e < 8; e += 2) {
      y[e] = __fmul_rn(r, x[e]);
      y[e + 1] = __fmul_rn(r, x[e + 1]);
      bf16_round2(y[e], y[e + 1]);
    }
    if constexpr (kRope) {
      if (!is_v) {
#pragma unroll
        for (int m = 0; m < 4; ++m) {
          const float c = cs[m], s = cs[4 + m];
          const float a0 = y[2 * m], a1 = y[2 * m + 1];
          if constexpr (MUSE_VR_QK_UNFUSED != 0) {
            y[2 * m] = __fsub_rn(__fmul_rn(a0, c), __fmul_rn(a1, s));
            y[2 * m + 1] = __fadd_rn(__fmul_rn(a0, s), __fmul_rn(a1, c));
          } else {
            y[2 * m] = __fmaf_rn(a0, c, -__fmul_rn(a1, s));
            y[2 * m + 1] = __fmaf_rn(a1, c, __fmul_rn(a0, s));
          }
          bf16_round2(y[2 * m], y[2 * m + 1]);
        }
      }
    }
    if constexpr (!kStore) {  // the rounded values are bf16: their upper halves are the bf16 bits
      const uint4 o = make_uint4((__float_as_uint(y[0]) >> 16) | (__float_as_uint(y[1]) & 0xffff0000u),
                                 (__float_as_uint(y[2]) >> 16) | (__float_as_uint(y[3]) & 0xffff0000u),
                                 (__float_as_uint(y[4]) >> 16) | (__float_as_uint(y[5]) & 0xffff0000u),
                                 (__float_as_uint(y[6]) >> 16) | (__float_as_uint(y[7]) & 0xffff0000u));
      if (is_q) {
        *reinterpret_cast<uint4*>(p.q_bf + (static_cast<int64_t>(tok) * nq + h) * kHeadDim + 8 * li) = o;
      } else if (is_k) {
        *reinterpret_cast<uint4*>(p.k_bf + (static_cast<int64_t>(tok) * nkv + h - nq) * kHeadDim + 8 * li) = o;
      }
      continue;
    }
    if (is_v) {
#pragma unroll
      for (int e = 0; e < 8; ++e) y[e] = x[e];
    }
    uint32_t b[4];
#pragma unroll
    for (int m = 0; m < 4; ++m)
      b[m] = e4m3x2(y[2 * m], y[2 * m + 1]);  // satfinite: |x| > 448 -> +-448, NaN stays NaN (as the stock FP8 cast)
    const uint2 out = make_uint2(b[0] | (b[1] << 16), b[2] | (b[3] << 16));
    if (is_q) {
      *reinterpret_cast<uint2*>(p.q_out + (static_cast<int64_t>(tok) * nq + h) * kHeadDim + 8 * li) = out;
    } else if (is_k || is_v) {
      const int kvh = is_k ? h - nq : h - nq - nkv;
      *reinterpret_cast<uint2*>((is_k ? p.k_pool : p.v_pool) + loc * p.pool_row_stride + kvh * kHeadDim + 8 * li) = out;
    }
  }
}

}  // namespace pqk

template <bool kRope, int kPasses, bool kStore>
static void launch_qkn(const pqk::Params& p, DLDevice dev) {
  using namespace sglang::host;
  LaunchKernel(dim3(static_cast<unsigned>((p.tokens + 3) / 4)), 128, dev, 0)(
      pqk::prefill_qkn_kernel<kRope, kPasses, kStore>, p);
}

template <bool kRope, bool kStore>
static void launch_qkn_passes(const pqk::Params& p, int heads, DLDevice dev) {
  const int passes = (heads + 3) / 4;
  if (passes <= 3) {
    launch_qkn<kRope, 3, kStore>(p, dev);
  } else if (passes <= 5) {
    launch_qkn<kRope, 5, kStore>(p, dev);
  } else if (passes <= 9) {
    launch_qkn<kRope, 9, kStore>(p, dev);
  } else if (passes <= 10) {
    launch_qkn<kRope, 10, kStore>(p, dev);
  } else {
    launch_qkn<kRope, pqk::kMaxPasses, kStore>(p, dev);
  }
}

template <bool kRope>
struct PrefillQkn {
  static void run(const tvm::ffi::TensorView qkv, const tvm::ffi::TensorView positions, const tvm::ffi::TensorView cos_sin,
                  const tvm::ffi::TensorView k_pool, const tvm::ffi::TensorView v_pool, const tvm::ffi::TensorView cache_loc,
                  const tvm::ffi::TensorView q_out, int64_t num_q_heads, int64_t num_kv_heads, double eps) {
    using namespace sglang::host;
    pqk::Params p{};
    p.qkv = static_cast<const __nv_bfloat16*>(qkv.data_ptr());
    p.qkv_token_stride = qkv.stride(0);
    p.positions = static_cast<const int64_t*>(positions.data_ptr());
    p.cos_sin = static_cast<const float*>(cos_sin.data_ptr());
    p.cache_loc = static_cast<const int64_t*>(cache_loc.data_ptr());
    p.k_pool = static_cast<uint8_t*>(k_pool.data_ptr());
    p.v_pool = static_cast<uint8_t*>(v_pool.data_ptr());
    p.pool_row_stride = k_pool.stride(0);
    p.q_out = static_cast<uint8_t*>(q_out.data_ptr());
    p.tokens = static_cast<int32_t>(qkv.size(0));
    p.num_q_heads = static_cast<int32_t>(num_q_heads);
    p.num_kv_heads = static_cast<int32_t>(num_kv_heads);
    p.eps = static_cast<float>(eps);
    RuntimeCheck(num_q_heads + 2 * num_kv_heads <= 4 * pqk::kMaxPasses, "too many heads");
    if (p.tokens == 0) return;
    launch_qkn_passes<kRope, true>(p, static_cast<int>(num_q_heads + 2 * num_kv_heads), qkv.device());
  }
};

// Norm-only: bf16 q / k as the Triton kernel's no-store launch writes them.
template <bool kRope>
struct PrefillQkNorm {
  static void run(const tvm::ffi::TensorView qkv, const tvm::ffi::TensorView positions, const tvm::ffi::TensorView cos_sin,
                  const tvm::ffi::TensorView q_out, const tvm::ffi::TensorView k_out, int64_t num_q_heads,
                  int64_t num_kv_heads, double eps) {
    using namespace sglang::host;
    pqk::Params p{};
    p.qkv = static_cast<const __nv_bfloat16*>(qkv.data_ptr());
    p.qkv_token_stride = qkv.stride(0);
    p.positions = static_cast<const int64_t*>(positions.data_ptr());
    p.cos_sin = static_cast<const float*>(cos_sin.data_ptr());
    p.q_bf = static_cast<__nv_bfloat16*>(q_out.data_ptr());
    p.k_bf = static_cast<__nv_bfloat16*>(k_out.data_ptr());
    p.tokens = static_cast<int32_t>(qkv.size(0));
    p.num_q_heads = static_cast<int32_t>(num_q_heads);
    p.num_kv_heads = static_cast<int32_t>(num_kv_heads);
    p.eps = static_cast<float>(eps);
    RuntimeCheck(num_q_heads + num_kv_heads <= 4 * pqk::kMaxPasses, "too many heads");
    if (p.tokens == 0) return;
    launch_qkn_passes<kRope, false>(p, static_cast<int>(num_q_heads + num_kv_heads), qkv.device());
  }
};

#if MUSE_VR_QK_UNFUSED
}  // inline namespace qk_unfused
#endif
}  // namespace muse_vr
