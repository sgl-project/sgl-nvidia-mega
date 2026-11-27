// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-FileCopyrightText: Copyright (c) 2026 by FlashInfer team.
// This file has been modified from its original version (a CUDA C++ port of the RMSNorm stage with a
// static FP8 output).
// Adapted from FlashInfer (Apache-2.0, Copyright (c) 2026 by FlashInfer team), flashinfer-python 0.7.4:
// flashinfer/comm/mnnvl_cutedsl/kernel_bt/device_kernels.py (_MaterializeRMSNormDeviceKernel, _block_sum).

#include "q35_hop.cuh"

namespace q35bt {

using namespace q35hop;

constexpr int kThreads = kFrags;
constexpr int kWarps = kThreads / 32;
constexpr uint32_t kActiveStage = 1;  // stage_state[ACTIVE_STAGE], written by the publish
static_assert(kWarps <= 32, "the warp sums fold in one warp");

struct Args {
  const uint16_t* prenorm;       // this rank's prenorm mailbox [3][m_cap][kHidden] bf16
  const uint32_t* stage_state;
  const uint16_t* gamma;         // [kHidden] bf16, the applied multiplier
  const float* fp8_scale;        // [1], or null
  uint16_t* residual_out;        // [M, kHidden] bf16
  uint16_t* norm_out;            // [M, kHidden] bf16
  uint8_t* fp8_out;              // [M, kHidden] e4m3, or null
  int m_cap;
  float eps;
};

__device__ __forceinline__ uint4 ld_v4(const void* p) { return *static_cast<const uint4*>(p); }

__device__ __forceinline__ uint4 poll(const uint16_t* p, uint4 v) {
  while (frag_has_negzero(v)) v = ld_volatile_v4(p);
  return v;
}

__device__ __forceinline__ float block_sum(float v, float* warp_sums) {
  const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
#pragma unroll
  for (int off = 16; off >= 1; off >>= 1) v = __fadd_rn(v, __shfl_xor_sync(0xffffffffu, v, off));
  if (lane == 0) warp_sums[warp] = v;
  __syncthreads();
  if (warp == 0) {
    float t = lane < kWarps ? warp_sums[lane] : 0.f;
#pragma unroll
    for (int off = 16; off >= 1; off >>= 1) t = __fadd_rn(t, __shfl_xor_sync(0xffffffffu, t, off));
    if (lane == 0) warp_sums[0] = t;
  }
  __syncthreads();
  return warp_sums[0];
}

__device__ __forceinline__ float inv_rms_of(float total, float eps) {
  const float x = __fadd_rn(__fmul_rn(total, 1.f / static_cast<float>(kHidden)), eps);
  float r;
  asm("rsqrt.approx.ftz.f32 %0, %1;" : "=f"(r) : "f"(x));
  return r;
}

__device__ __forceinline__ float inv_scale_here(float scale) {
  float r;
  asm volatile("div.full.f32 %0, %1, %2;" : "=f"(r) : "f"(1.f), "f"(scale));
  return r;
}

template <int TRIGGER>
__device__ __forceinline__ void tail_kernel(const Args& a) {
  __shared__ float warp_sums[kWarps];
  const int token = blockIdx.x;
  const int frag = threadIdx.x;
  const size_t elem = static_cast<size_t>(token) * kHidden + static_cast<size_t>(frag) * kVec;
  const size_t stride = static_cast<size_t>(a.m_cap) * kHidden;
  uint16_t* const box = const_cast<uint16_t*>(a.prenorm);
  const uint4 gamma = ld_v4(a.gamma + static_cast<size_t>(frag) * kVec);
  const float scale = a.fp8_out != nullptr ? *a.fp8_scale : 1.f;
  pdl_wait();
  if (TRIGGER == 1) pdl_trigger();
  const uint32_t stage = ld_volatile_u32(a.stage_state + kActiveStage);
  uint16_t* const slot = box + stage * stride + elem;
  const uint4 p = poll(slot, ld_volatile_v4(slot));
  *reinterpret_cast<uint4*>(a.residual_out + elem) = p;
  float pre[8];
  unpack8(p, pre);
  const float total = block_sum(sumsq8(pre), warp_sums);
  float gam[8];
  unpack8(gamma, gam);
  const uint4 norm = norm_apply(pre, inv_rms_of(total, a.eps), gam);
  *reinterpret_cast<uint4*>(a.norm_out + elem) = norm;
  if (a.fp8_out != nullptr)
    *reinterpret_cast<uint2*>(a.fp8_out + elem) = quant_fp8(norm, inv_scale_here(scale));
  if (TRIGGER == 0) {
    __syncthreads();
    pdl_trigger();
  }
  st_v4(slot, make_uint4(kNegZeroPair, kNegZeroPair, kNegZeroPair, kNegZeroPair));
}

}

using q35bt::Args;

extern "C" __global__ void __launch_bounds__(q35bt::kThreads) q35bt_tail(Args a) { q35bt::tail_kernel<0>(a); }

extern "C" __global__ void __launch_bounds__(q35bt::kThreads) q35bt_tail_early(Args a) {
  q35bt::tail_kernel<1>(a);
}
