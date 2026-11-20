// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
//
// The Muse Spark output boundary in one kernel: the last decoder layer's post-feedforward norm (weight applied to the
// input, then an RMSNorm without weight), its norm-preserving gated residual into the fp32 stream, the cast of the
// stream to bf16 and the model's final RMSNorm. SGLang runs these as separate PyTorch operations; this kernel performs
// the same operations in the same order with the same roundings:
//
//   a    = bf16(w_post * ffn)                                    (bf16 x bf16 product in fp32, rounded)
//   b    = bf16(rsqrt(sum(a * a) / H + eps_post) * a)            (row RMSNorm, no weight)
//   s    = alpha * stream + beta * float(b)                      (fp32, each product and the sum rounded separately)
//   n    = bf16(s)
//   out  = bf16(w_final * (rsqrt(sum(n * n) / H + eps_final) * n))
//
// Each row is handled by one block of 128 threads (32 x 4); thread t accumulates the squares of the kV-element vectors
// t, t + 128, ... in order (fused multiply-add), then the per-thread sums are added with a shuffle-down tree inside
// each warp and the four warp sums are added pairwise (warp 0 + warp 2, warp 1 + warp 3, then the two results) — the
// summation order of PyTorch's row-wise RMSNorm kernel for these sizes. The vector width kV is that of the running
// PyTorch build (final_norm.py, TORCH_VEC: 4 or 8; tests/test_final_norm.py checks the outputs bit for bit against the
// PyTorch operations).

#pragma once

#include <sgl_kernel/tensor.h>
#include <sgl_kernel/utils.h>

#include <sgl_kernel/utils.cuh>

#include <tvm/ffi/container/tensor.h>

#include <cuda_bf16.h>
#include <cstdint>

namespace muse_vr::final_norm {

using namespace sglang;

constexpr int kThreads = 128;

struct Params {
  const uint16_t* ffn;     // [M, H] bf16
  const float* stream;     // [M, H] fp32
  const uint16_t* w_post;  // [H] bf16 (effective weight of the post-feedforward norm)
  const float* alpha;      // [H] fp32
  const float* beta;       // [H] fp32
  const uint16_t* w_final; // [H] bf16 (effective weight of the final norm)
  uint16_t* out;           // [M, H] bf16
  float eps_post;
  float eps_final;
  int hidden;
};

SGL_DEVICE float bf(uint32_t b) {
  return __uint_as_float(b << 16);
}

SGL_DEVICE uint32_t rb(float v) {
  return static_cast<uint32_t>(__bfloat16_as_ushort(__float2bfloat16_rn(v)));
}

// kV consecutive bf16 values (kV / 2 words), kV consecutive fp32 values, and a kV-wide bf16 store.
template <int kV>
SGL_DEVICE void ld_bf16(const uint16_t* ptr, uint32_t (&w)[kV / 2]) {
  if constexpr (kV == 4) {
    const uint2 u = *reinterpret_cast<const uint2*>(ptr);
    w[0] = u.x;
    w[1] = u.y;
  } else {
    static_assert(kV == 8);
    const uint4 u = *reinterpret_cast<const uint4*>(ptr);
    w[0] = u.x;
    w[1] = u.y;
    w[2] = u.z;
    w[3] = u.w;
  }
}

template <int kV>
SGL_DEVICE void ld_f32(const float* ptr, float (&f)[kV]) {
#pragma unroll
  for (int i = 0; i < kV / 4; ++i) {
    const float4 u = *reinterpret_cast<const float4*>(ptr + 4 * i);
    f[4 * i] = u.x;
    f[4 * i + 1] = u.y;
    f[4 * i + 2] = u.z;
    f[4 * i + 3] = u.w;
  }
}

template <int kV>
SGL_DEVICE void st_bf16(uint16_t* ptr, const uint32_t (&o)[kV]) {
  if constexpr (kV == 4) {
    *reinterpret_cast<uint2*>(ptr) = make_uint2(o[0] | (o[1] << 16), o[2] | (o[3] << 16));
  } else {
    *reinterpret_cast<uint4*>(ptr) =
        make_uint4(o[0] | (o[1] << 16), o[2] | (o[3] << 16), o[4] | (o[5] << 16), o[6] | (o[7] << 16));
  }
}

SGL_DEVICE uint32_t half_of(const uint32_t* w, int e) {
  return (w[e / 2] >> ((e % 2) * 16)) & 0xffffu;
}

// Row sum of `sq` (per-thread partial sums) in the order described above; every thread gets the result.
SGL_DEVICE float row_sum(float sq, float* buf) {
  const int x = threadIdx.x % 32, y = threadIdx.x / 32;
#pragma unroll
  for (int off = 16; off > 0; off >>= 1) sq = __fadd_rn(sq, __shfl_down_sync(0xffffffffu, sq, off));
  if (x == 0 && y >= 2) buf[y - 2] = sq;
  __syncthreads();
  if (x == 0 && y < 2) sq = __fadd_rn(sq, buf[y]);
  __syncthreads();
  if (x == 0 && y == 1) buf[0] = sq;
  __syncthreads();
  if (x == 0 && y == 0) {
    sq = __fadd_rn(sq, buf[0]);
    buf[2] = sq;
  }
  __syncthreads();
  const float total = buf[2];
  __syncthreads();
  return total;
}

template <int kPerThread, bool kUsePDL, int kV>
__global__ __launch_bounds__(kThreads) void final_norm_kernel(const __grid_constant__ Params p) {
  __shared__ float buf[4];
  const int t = threadIdx.x;
  const int64_t row = blockIdx.x;
  const int H = p.hidden;
  device::PDLWaitPrimary<kUsePDL>();
  uint32_t a[kPerThread][kV];
  float sq = 0.f;
#pragma unroll
  for (int k = 0; k < kPerThread; ++k) {
    const int v = t + k * kThreads;
    uint32_t f2[kV / 2], w2[kV / 2];
    ld_bf16<kV>(p.ffn + row * H + v * kV, f2);
    ld_bf16<kV>(p.w_post + v * kV, w2);
#pragma unroll
    for (int e = 0; e < kV; ++e) {
      const uint32_t fw = half_of(f2, e);
      const uint32_t ww = half_of(w2, e);
      a[k][e] = rb(__fmul_rn(bf(ww), bf(fw)));
      const float av = bf(a[k][e]);
      sq = __fmaf_rn(av, av, sq);
    }
  }
  const float r1 = rsqrtf(__fadd_rn(__fdiv_rn(row_sum(sq, buf), static_cast<float>(H)), p.eps_post));
  sq = 0.f;
  uint32_t n[kPerThread][kV];
#pragma unroll
  for (int k = 0; k < kPerThread; ++k) {
    const int v = t + k * kThreads;
    float sv[kV], av[kV], bv[kV];
    ld_f32<kV>(p.stream + row * H + v * kV, sv);
    ld_f32<kV>(p.alpha + v * kV, av);
    ld_f32<kV>(p.beta + v * kV, bv);
#pragma unroll
    for (int e = 0; e < kV; ++e) {
      const float b = bf(rb(__fmul_rn(r1, bf(a[k][e]))));
      const float s = __fadd_rn(__fmul_rn(av[e], sv[e]), __fmul_rn(bv[e], b));
      n[k][e] = rb(s);
      const float nv = bf(n[k][e]);
      sq = __fmaf_rn(nv, nv, sq);
    }
  }
  const float r2 = rsqrtf(__fadd_rn(__fdiv_rn(row_sum(sq, buf), static_cast<float>(H)), p.eps_final));
#pragma unroll
  for (int k = 0; k < kPerThread; ++k) {
    const int v = t + k * kThreads;
    uint32_t g2[kV / 2];
    ld_bf16<kV>(p.w_final + v * kV, g2);
    uint32_t o[kV];
#pragma unroll
    for (int e = 0; e < kV; ++e) {
      const uint32_t gw = half_of(g2, e);
      o[e] = rb(__fmul_rn(bf(gw), __fmul_rn(r2, bf(n[k][e]))));
    }
    st_bf16<kV>(p.out + row * H + v * kV, o);
  }
  device::PDLTriggerSecondary<kUsePDL>();
}

template <int kHidden, bool kUsePDL, int kV = 4>
struct FinalNorm {
  static void run(const tvm::ffi::TensorView ffn, const tvm::ffi::TensorView stream, const tvm::ffi::TensorView w_post,
                  const tvm::ffi::TensorView alpha, const tvm::ffi::TensorView beta,
                  const tvm::ffi::TensorView w_final, const tvm::ffi::TensorView out, double eps_post,
                  double eps_final) {
    using namespace host;
    static_assert(kHidden % (kThreads * kV) == 0);
    auto M = SymbolicSize{"m"};
    auto device_ = SymbolicDevice{};
    device_.set_options<kDLCUDA>();
    TensorMatcher({M, kHidden}).with_dtype<bf16_t>().with_device(device_).verify(ffn);
    TensorMatcher({M, kHidden}).with_dtype<fp32_t>().with_device(device_).verify(stream);
    TensorMatcher({M, kHidden}).with_dtype<bf16_t>().with_device(device_).verify(out);
    TensorMatcher({kHidden}).with_dtype<bf16_t>().with_device(device_).verify(w_post);
    TensorMatcher({kHidden}).with_dtype<bf16_t>().with_device(device_).verify(w_final);
    TensorMatcher({kHidden}).with_dtype<fp32_t>().with_device(device_).verify(alpha);
    TensorMatcher({kHidden}).with_dtype<fp32_t>().with_device(device_).verify(beta);
    const int64_t m = M.unwrap();
    if (m == 0) return;
    Params p{};
    p.ffn = static_cast<const uint16_t*>(ffn.data_ptr());
    p.stream = static_cast<const float*>(stream.data_ptr());
    p.w_post = static_cast<const uint16_t*>(w_post.data_ptr());
    p.alpha = static_cast<const float*>(alpha.data_ptr());
    p.beta = static_cast<const float*>(beta.data_ptr());
    p.w_final = static_cast<const uint16_t*>(w_final.data_ptr());
    p.out = static_cast<uint16_t*>(out.data_ptr());
    p.eps_post = static_cast<float>(eps_post);
    p.eps_final = static_cast<float>(eps_final);
    p.hidden = kHidden;
    LaunchKernel(static_cast<unsigned>(m), kThreads, device_.unwrap())
        .enable_pdl(kUsePDL)(final_norm_kernel<kHidden / (kThreads * kV), kUsePDL, kV>, p);
  }
};

}  // namespace muse_vr::final_norm
