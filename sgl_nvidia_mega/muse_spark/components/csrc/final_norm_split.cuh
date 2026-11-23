// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
//
// The Muse Spark output boundary (see final_norm.cuh for the operations and roundings) with each row split over a
// thread-block cluster of kC CTAs. The arithmetic and the summation order are those of final_norm.cuh, so the outputs
// are bit-identical to it and to the PyTorch operations (tests/test_final_norm.py):
//
//   final_norm.cuh runs one 128-thread block per row; thread t accumulates the squares of the kV-element vectors
//   t, t + 128, ..., in order with fused multiply-adds (kV = 4 or 8, the running PyTorch build's width), and the 128 per-thread sums are then added with a fixed
//   warp/block tree. Here the 128 per-thread sums ("virtual threads") are split over the cluster: CTA c owns virtual
//   threads [c * 128 / kC, (c + 1) * 128 / kC). All 128 threads of a CTA load that CTA's vectors (the weights before
//   waiting for the preceding kernel), compute the element values, and park them in shared memory; one thread per
//   virtual thread then runs that virtual thread's fused multiply-add chain over the parked values in the original
//   order. The chain results are written to every CTA of the cluster through distributed shared memory, and after a
//   cluster barrier every CTA adds the 128 values with the same tree as final_norm.cuh (same instructions, so the same
//   result in every CTA). The two sums of squares of a row take one exchange each.

#pragma once

#include <sgl_kernel/tensor.h>
#include <sgl_kernel/utils.h>

#include <sgl_kernel/utils.cuh>

#include <tvm/ffi/container/tensor.h>

#include <cuda_bf16.h>
#include <cstdint>

namespace muse_vr::final_norm_split {

using namespace sglang;

constexpr int kThreads = 128;  // threads per CTA, and virtual threads per row

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

// kV consecutive bf16 values (kV / 2 words) and fp32 values, and a kV-wide bf16 store.
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

SGL_DEVICE uint32_t cluster_ctarank() {
  uint32_t r;
  asm volatile("mov.u32 %0, %%cluster_ctarank;" : "=r"(r));
  return r;
}

SGL_DEVICE void cluster_arrive_relaxed() {
  asm volatile("barrier.cluster.arrive.relaxed.aligned;" ::: "memory");
}

SGL_DEVICE void cluster_arrive_release() {
  asm volatile("barrier.cluster.arrive.release.aligned;" ::: "memory");
}

SGL_DEVICE void cluster_wait_acquire() {
  asm volatile("barrier.cluster.wait.acquire.aligned;" ::: "memory");
}

SGL_DEVICE void st_cluster_f32(const float* local_addr, uint32_t cta, float v) {
  const uint32_t a = static_cast<uint32_t>(__cvta_generic_to_shared(local_addr));
  uint32_t remote;
  asm volatile("mapa.shared::cluster.u32 %0, %1, %2;" : "=r"(remote) : "r"(a), "r"(cta));
  asm volatile("st.shared::cluster.f32 [%0], %1;" ::"r"(remote), "f"(v) : "memory");
}

// The block tree of final_norm.cuh's row_sum (identical instructions); every thread gets the result.
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

// kPerRow: vectors per virtual thread (H / (128 * kV)); kC: CTAs per row (one cluster); kV: elements per vector.
template <int kC, int kPerRow, bool kUsePDL, int kV>
__global__ __launch_bounds__(kThreads) void final_norm_split_kernel(const __grid_constant__ Params p) {
  static_assert(kThreads % kC == 0 && kPerRow % kC == 0, "split must divide the virtual threads and their vectors");
  constexpr int kVT = kThreads / kC;  // virtual threads owned by this CTA
  constexpr int kPer = kPerRow / kC;  // vectors per real thread
  constexpr int kQ = kV / 4;          // float4 slots per vector
  // [k][virtual thread][q]: element values of phase 1, then of phase 2
  __shared__ float4 parked[kPerRow * kVT * kQ];
  __shared__ float part1[kThreads];
  __shared__ float part2[kThreads];
  __shared__ float buf[4];

  const int j = threadIdx.x;
  const uint32_t c = cluster_ctarank();
  const int64_t row = blockIdx.x / kC;
  const int H = p.hidden;
  const int tl = j % kVT, kg = j / kVT;
  const int t = static_cast<int>(c) * kVT + tl;  // this thread's virtual thread; its vectors are t + 128 k

  cluster_arrive_relaxed();  // matched below: every CTA of the cluster has started before any remote store

  uint32_t wp[kPer][kV / 2], wf[kPer][kV / 2];
  float al[kPer][kV], be[kPer][kV];
#pragma unroll
  for (int i = 0; i < kPer; ++i) {
    const int v = t + (kg + kC * i) * kThreads;
    ld_bf16<kV>(p.w_post + v * kV, wp[i]);
    ld_bf16<kV>(p.w_final + v * kV, wf[i]);
    ld_f32<kV>(p.alpha + v * kV, al[i]);
    ld_f32<kV>(p.beta + v * kV, be[i]);
  }
  device::PDLWaitPrimary<kUsePDL>();
  uint32_t fv[kPer][kV / 2];
  float sv[kPer][kV];
#pragma unroll
  for (int i = 0; i < kPer; ++i) {
    const int v = t + (kg + kC * i) * kThreads;
    ld_bf16<kV>(p.ffn + row * H + v * kV, fv[i]);
    ld_f32<kV>(p.stream + row * H + v * kV, sv[i]);
  }

  // phase 1: a = bf16(w_post * ffn)
  uint32_t a[kPer][kV];
#pragma unroll
  for (int i = 0; i < kPer; ++i) {
#pragma unroll
    for (int e = 0; e < kV; ++e) a[i][e] = rb(__fmul_rn(bf(half_of(wp[i], e)), bf(half_of(fv[i], e))));
#pragma unroll
    for (int q = 0; q < kQ; ++q) {
      parked[((kg + kC * i) * kVT + tl) * kQ + q] =
          make_float4(bf(a[i][4 * q]), bf(a[i][4 * q + 1]), bf(a[i][4 * q + 2]), bf(a[i][4 * q + 3]));
    }
  }
  __syncthreads();
  cluster_wait_acquire();
  if (j < kVT) {
    float sq = 0.f;
#pragma unroll
    for (int k = 0; k < kPerRow; ++k) {
#pragma unroll
      for (int q = 0; q < kQ; ++q) {
        const float4 x = parked[(k * kVT + j) * kQ + q];
        sq = __fmaf_rn(x.x, x.x, sq);
        sq = __fmaf_rn(x.y, x.y, sq);
        sq = __fmaf_rn(x.z, x.z, sq);
        sq = __fmaf_rn(x.w, x.w, sq);
      }
    }
#pragma unroll
    for (int r = 0; r < kC; ++r) st_cluster_f32(&part1[static_cast<int>(c) * kVT + j], static_cast<uint32_t>(r), sq);
  }
  cluster_arrive_release();
  cluster_wait_acquire();
  const float r1 = rsqrtf(__fadd_rn(__fdiv_rn(row_sum(part1[j], buf), static_cast<float>(H)), p.eps_post));

  // phase 2: b = bf16(r1 * a); s = alpha * stream + beta * b; n = bf16(s)
  uint32_t n[kPer][kV];
#pragma unroll
  for (int i = 0; i < kPer; ++i) {
#pragma unroll
    for (int e = 0; e < kV; ++e) {
      const float b = bf(rb(__fmul_rn(r1, bf(a[i][e]))));
      const float s = __fadd_rn(__fmul_rn(al[i][e], sv[i][e]), __fmul_rn(be[i][e], b));
      n[i][e] = rb(s);
    }
#pragma unroll
    for (int q = 0; q < kQ; ++q) {
      parked[((kg + kC * i) * kVT + tl) * kQ + q] =
          make_float4(bf(n[i][4 * q]), bf(n[i][4 * q + 1]), bf(n[i][4 * q + 2]), bf(n[i][4 * q + 3]));
    }
  }
  __syncthreads();
  if (j < kVT) {
    float sq = 0.f;
#pragma unroll
    for (int k = 0; k < kPerRow; ++k) {
#pragma unroll
      for (int q = 0; q < kQ; ++q) {
        const float4 x = parked[(k * kVT + j) * kQ + q];
        sq = __fmaf_rn(x.x, x.x, sq);
        sq = __fmaf_rn(x.y, x.y, sq);
        sq = __fmaf_rn(x.z, x.z, sq);
        sq = __fmaf_rn(x.w, x.w, sq);
      }
    }
#pragma unroll
    for (int r = 0; r < kC; ++r) st_cluster_f32(&part2[static_cast<int>(c) * kVT + j], static_cast<uint32_t>(r), sq);
  }
  cluster_arrive_release();
  cluster_wait_acquire();
  const float r2 = rsqrtf(__fadd_rn(__fdiv_rn(row_sum(part2[j], buf), static_cast<float>(H)), p.eps_final));

  // phase 3: out = bf16(w_final * (r2 * n))
#pragma unroll
  for (int i = 0; i < kPer; ++i) {
    const int v = t + (kg + kC * i) * kThreads;
    uint32_t o[kV];
#pragma unroll
    for (int e = 0; e < kV; ++e) o[e] = rb(__fmul_rn(bf(half_of(wf[i], e)), __fmul_rn(r2, bf(n[i][e]))));
    st_bf16<kV>(p.out + row * H + v * kV, o);
  }
  device::PDLTriggerSecondary<kUsePDL>();
}

template <int kHidden, int kC, bool kUsePDL, int kV = 4>
struct FinalNormSplit {
  static void run(const tvm::ffi::TensorView ffn, const tvm::ffi::TensorView stream, const tvm::ffi::TensorView w_post,
                  const tvm::ffi::TensorView alpha, const tvm::ffi::TensorView beta,
                  const tvm::ffi::TensorView w_final, const tvm::ffi::TensorView out, double eps_post,
                  double eps_final) {
    using namespace host;
    static_assert(kHidden % (kThreads * kV) == 0);
    constexpr int kPerRow = kHidden / (kThreads * kV);
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
    constexpr auto kernel = final_norm_split_kernel<kC, kPerRow, kUsePDL, kV>;
    LaunchKernel(static_cast<unsigned>(m * kC), kThreads, device_.unwrap())
        .config({.use_pdl = kUsePDL, .cluster_dim = dim3(kC, 1, 1)})(kernel, p);
  }
};

}  // namespace muse_vr::final_norm_split
