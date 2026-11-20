// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
/**
 * \brief Prefill QK norm with an e4m3 q: the norm-only form of prefill_qkn.cuh (per-head RMS norm of the q and k heads
 * of a QKV projection output, RoPE when given; the same explicitly rounded operations in the same order) with q written
 * as e4m3 bytes instead of bf16. k is written as bf16, exactly as the norm-only form writes it.
 *
 * The q bytes equal PyTorch's ``bf16_q.to(torch.float8_e4m3fn)`` of the bf16 q the norm-only form writes: each value is
 * rounded to bf16 first (as there), then converted with PyTorch's float -> float8_e4m3fn semantics (``e4m3fn_of``):
 * round to nearest even; NaN, +-Inf and magnitudes above 464 (the values that round past the largest finite e4m3 value,
 * 448) give 0x7F (NaN) with the input's sign bit; no saturation. This differs from the satfinite conversion of the store
 * form (``e4m3x2`` in prefill_qkn.cuh, which clamps to +-448) only for those inputs.
 *
 * Layout as in prefill_qkn.cuh: one warp per token; each pass covers 4 head rows of the token (8 lanes per 64-column
 * row, 8 columns per lane). No barriers; no waits. A separate module from the existing ones (their sources are
 * unchanged).
 */

#include "prefill_qkn.cuh"

namespace muse_vr {
#if MUSE_VR_QK_UNFUSED
inline namespace qk_unfused {
#endif
namespace pqk {

// e4m3 (fn) byte of an fp32 value, PyTorch's float -> float8_e4m3fn semantics (see the file comment)
__device__ __forceinline__ uint32_t e4m3fn_of(float x) {
  const uint32_t u = __float_as_uint(x);
  const uint32_t sign = (u >> 24) & 0x80u;
  const uint32_t a = u & 0x7fffffffu;
  if (a > 0x43e80000u) return sign | 0x7fu;  // |x| > 464, +-Inf, NaN
  if (a < 0x3c800000u) {                     // |x| < 2^-6: multiples of 2^-9 (subnormal steps), nearest even, 0 .. 8
    return sign | __float2uint_rn(__fmul_rn(__uint_as_float(a), 512.0f));
  }
  // 2^-6 <= |x| <= 464: exponent rebiased (127 -> 7), 3 mantissa bits kept, the 20 dropped bits rounded to nearest even
  const uint32_t r = a - (120u << 23);
  const uint32_t keep = r >> 20, rest = r & 0xfffffu;
  const uint32_t up = (rest > 0x80000u || (rest == 0x80000u && (keep & 1u))) ? 1u : 0u;
  return sign | (keep + up);
}

template <bool kRope, int kPasses>
__global__ void __launch_bounds__(128) prefill_qk_norm_q8_kernel(const Params p) {
  const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
  const int tok = blockIdx.x * 4 + warp;
  if (tok >= p.tokens) return;  // whole warp
  const int li = lane & 7, rsub = lane >> 3;
  const int nq = p.num_q_heads, nkv = p.num_kv_heads, heads = nq + nkv;
  const __nv_bfloat16* const src = p.qkv + static_cast<int64_t>(tok) * p.qkv_token_stride + 8 * li;

  float cs[8];  // cos of frequencies 4 li .. 4 li + 3, then their sin
  if constexpr (kRope) {
    const int64_t pos = p.positions[tok];
    const float4 c = *reinterpret_cast<const float4*>(p.cos_sin + pos * kHeadDim + 4 * li);
    const float4 s = *reinterpret_cast<const float4*>(p.cos_sin + pos * kHeadDim + kHeadDim / 2 + 4 * li);
    cs[0] = c.x; cs[1] = c.y; cs[2] = c.z; cs[3] = c.w;
    cs[4] = s.x; cs[5] = s.y; cs[6] = s.z; cs[7] = s.w;
  }

  uint4 w[kPasses];
#pragma unroll
  for (int ps = 0; ps < kPasses; ++ps) {
    const int h = 4 * ps + rsub;
    w[ps] = h < heads ? *reinterpret_cast<const uint4*>(src + h * kHeadDim) : make_uint4(0, 0, 0, 0);
  }
#pragma unroll
  for (int ps = 0; ps < kPasses; ++ps) {
    const int h = 4 * ps + rsub;
    const bool is_q = h < nq, is_k = h >= nq && h < heads;
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
    // the rounded values are bf16: their upper halves are the bf16 bits
    if (is_q) {
      uint32_t b[4];
#pragma unroll
      for (int m = 0; m < 4; ++m) b[m] = e4m3fn_of(y[2 * m]) | (e4m3fn_of(y[2 * m + 1]) << 8);
      *reinterpret_cast<uint2*>(p.q_out + (static_cast<int64_t>(tok) * nq + h) * kHeadDim + 8 * li) =
          make_uint2(b[0] | (b[1] << 16), b[2] | (b[3] << 16));
    } else if (is_k) {
      const uint4 o = make_uint4((__float_as_uint(y[0]) >> 16) | (__float_as_uint(y[1]) & 0xffff0000u),
                                 (__float_as_uint(y[2]) >> 16) | (__float_as_uint(y[3]) & 0xffff0000u),
                                 (__float_as_uint(y[4]) >> 16) | (__float_as_uint(y[5]) & 0xffff0000u),
                                 (__float_as_uint(y[6]) >> 16) | (__float_as_uint(y[7]) & 0xffff0000u));
      *reinterpret_cast<uint4*>(p.k_bf + (static_cast<int64_t>(tok) * nkv + h - nq) * kHeadDim + 8 * li) = o;
    }
  }
}

// The conversion alone, element-wise over a bf16 vector (tests: every bf16 bit pattern against PyTorch's conversion).
__global__ void __launch_bounds__(256) e4m3fn_of_bf16_kernel(const uint16_t* __restrict__ x, uint8_t* __restrict__ out,
                                                             int64_t n) {
  const int64_t i = static_cast<int64_t>(blockIdx.x) * 256 + threadIdx.x;
  if (i < n) out[i] = static_cast<uint8_t>(e4m3fn_of(__uint_as_float(static_cast<uint32_t>(x[i]) << 16)));
}

}  // namespace pqk

template <bool kRope, int kPasses>
static void launch_qk_norm_q8(const pqk::Params& p, DLDevice dev) {
  using namespace sglang::host;
  LaunchKernel(dim3(static_cast<unsigned>((p.tokens + 3) / 4)), 128, dev, 0)(
      pqk::prefill_qk_norm_q8_kernel<kRope, kPasses>, p);
}

template <bool kRope>
static void launch_qk_norm_q8_passes(const pqk::Params& p, int heads, DLDevice dev) {
  const int passes = (heads + 3) / 4;
  if (passes <= 3) {
    launch_qk_norm_q8<kRope, 3>(p, dev);
  } else if (passes <= 5) {
    launch_qk_norm_q8<kRope, 5>(p, dev);
  } else if (passes <= 9) {
    launch_qk_norm_q8<kRope, 9>(p, dev);
  } else if (passes <= 10) {
    launch_qk_norm_q8<kRope, 10>(p, dev);
  } else {
    launch_qk_norm_q8<kRope, pqk::kMaxPasses>(p, dev);
  }
}

// Norm-only with an e4m3 q: q_out [T, Hq * 64] e4m3, k_out [T, Hkv * 64] bf16 (both contiguous).
template <bool kRope>
struct PrefillQkNormQ8 {
  static void run(const tvm::ffi::TensorView qkv, const tvm::ffi::TensorView positions, const tvm::ffi::TensorView cos_sin,
                  const tvm::ffi::TensorView q_out, const tvm::ffi::TensorView k_out, int64_t num_q_heads,
                  int64_t num_kv_heads, double eps) {
    using namespace sglang::host;
    RuntimeCheck(num_q_heads + num_kv_heads <= 4 * pqk::kMaxPasses, "too many heads");
    RuntimeCheck(q_out.dim() == 2 && q_out.size(0) == qkv.size(0) && q_out.size(1) == num_q_heads * pqk::kHeadDim &&
                     q_out.dtype().bits == 8,
                 "q_out must be [T, Hq * 64] e4m3");
    RuntimeCheck(k_out.dim() == 2 && k_out.size(0) == qkv.size(0) && k_out.size(1) == num_kv_heads * pqk::kHeadDim &&
                     k_out.dtype().code == DLDataTypeCode::kDLBfloat && k_out.dtype().bits == 16,
                 "k_out must be [T, Hkv * 64] bf16");
    pqk::Params p{};
    p.qkv = static_cast<const __nv_bfloat16*>(qkv.data_ptr());
    p.qkv_token_stride = qkv.stride(0);
    p.positions = static_cast<const int64_t*>(positions.data_ptr());
    p.cos_sin = static_cast<const float*>(cos_sin.data_ptr());
    p.q_out = static_cast<uint8_t*>(q_out.data_ptr());
    p.k_bf = static_cast<__nv_bfloat16*>(k_out.data_ptr());
    p.tokens = static_cast<int32_t>(qkv.size(0));
    p.num_q_heads = static_cast<int32_t>(num_q_heads);
    p.num_kv_heads = static_cast<int32_t>(num_kv_heads);
    p.eps = static_cast<float>(eps);
    if (p.tokens == 0) return;
    launch_qk_norm_q8_passes<kRope>(p, static_cast<int>(num_q_heads + num_kv_heads), qkv.device());
  }
};

// out[i] = e4m3fn_of(x[i]) for a 1-D bf16 x and a 1-D uint8 out of the same length.
struct E4m3OfBf16 {
  static void run(const tvm::ffi::TensorView x, const tvm::ffi::TensorView out) {
    using namespace sglang::host;
    RuntimeCheck(x.dim() == 1 && out.dim() == 1 && out.size(0) == x.size(0) && x.dtype().bits == 16 &&
                     out.dtype().bits == 8 && x.stride(0) == 1 && out.stride(0) == 1,
                 "x: 1-D contiguous bf16, out: 1-D contiguous uint8 of the same length");
    const int64_t n = x.size(0);
    if (n == 0) return;
    LaunchKernel(dim3(static_cast<unsigned>((n + 255) / 256)), 256, x.device(), 0)(
        pqk::e4m3fn_of_bf16_kernel, static_cast<const uint16_t*>(x.data_ptr()), static_cast<uint8_t*>(out.data_ptr()), n);
  }
};

#if MUSE_VR_QK_UNFUSED
}  // inline namespace qk_unfused
#endif
}  // namespace muse_vr
