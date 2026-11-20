// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
//
// The row-split FC's one-launch cross-rank sum: every rank's fp32 partial [m, 4096] is pushed through the
// multicast mapping and summed on every rank in rank order from rank 0, rounded once to bf16, in
// one launch: ordered_partial_sum's arithmetic on the bytes fc_gather moves, without the
// [m, world * 4096] intermediate or a second launch. Lamport transport: fp32 words, all ones (a
// NaN no partial carries) as "not arrived", three stages rotated by a control block, every
// consumer re-arming the words it read. Thread t of CTA c owns columns [4j, 4j + 4) of row
// c / 8, j = (c % 8) * 128 + t.
//
// Built with NVRTC for MKD_WORLD (the TP size).

#include "mkd_common.cuh"

#ifndef MKD_WORLD
#define MKD_WORLD 8
#endif

namespace sglang {

namespace fc_allsum {

inline constexpr int kStages = 3;
inline constexpr int kMaxTokens = 16;  // the split fc's widest batch, as fc_gather
inline constexpr int kThreads = 128;
inline constexpr int kCols = 4;  // fp32 columns a thread owns: one 16-byte word group
inline constexpr uint32_t kNotArrived = 0xFFFFFFFFu;
inline constexpr uint32_t kSpinLimit = 1u << 27;

// Per-process control block in device memory, zero at build.
struct Ctrl {
  uint32_t stage;   // stage of the next call
  uint32_t arrive;  // CTA tickets of the running call
};

SGL_DEVICE uint32_t ld_volatile_u32(const uint32_t* p) {
  uint32_t v;
  asm volatile("ld.volatile.global.u32 %0, [%1];" : "=r"(v) : "l"(p) : "memory");
  return v;
}

SGL_DEVICE uint4 ld_volatile_v4(const void* p) {
  uint4 v;
  asm volatile("ld.volatile.global.v4.u32 {%0,%1,%2,%3}, [%4];"
               : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w)
               : "l"(p)
               : "memory");
  return v;
}

SGL_DEVICE void st_v4(void* p, uint4 v) {
  asm volatile("st.global.v4.u32 [%0], {%1,%2,%3,%4};" ::"l"(p), "r"(v.x), "r"(v.y), "r"(v.z), "r"(v.w) : "memory");
}

SGL_DEVICE void mc_st_v4(void* mc, uint4 v) {
  asm volatile("multimem.st.relaxed.sys.global.v4.f32 [%0], {%1,%2,%3,%4};" ::"l"(mc), "f"(__uint_as_float(v.x)),
               "f"(__uint_as_float(v.y)), "f"(__uint_as_float(v.z)), "f"(__uint_as_float(v.w))
               : "memory");
}

// A partial word equal to the "not arrived" pattern (a NaN) is sent as another NaN.
SGL_DEVICE uint32_t sendable(uint32_t w) {
  return w == kNotArrived ? 0x7FFFFFFFu : w;
}

SGL_DEVICE uint32_t pending_mask(uint4 v) {
  return (v.x == kNotArrived) | (v.y == kNotArrived) | (v.z == kNotArrived) | (v.w == kNotArrived);
}

}  // namespace fc_allsum

struct FcAllsumParams {
  const float* partial;    // [m, kH] this rank's fp32 partial, written by an earlier grid
  bf16_t* out;             // [m, kH] the partials summed in rank order, rounded once
  float* mc;               // multicast base of the mailbox [kStages, world, kMaxTokens, kH] fp32
  float* uc;               // this rank's copy of it
  fc_allsum::Ctrl* ctrl;   // this rank's control block
  int32_t rank;
  int32_t m;
};

/**
 * \brief One call of the split fc's cross-rank sum: push this rank's partial, wait for every
 * rank's, sum them as ordered_partial_sum does (acc = rank 0, then + rank 1, ... in fp32 round to
 * nearest) and round once to bf16. \tparam kH hidden size, \tparam kWorld TP size. The grid is
 * m * (kH / 512) CTAs of 128 threads, all resident (no CTA waits on another CTA of its rank).
 */
template <int kH, int kWorld>
SGL_DEVICE void fc_allsum_kernel(const FcAllsumParams& p) {
  using namespace fc_allsum;
  constexpr int kCtasPerRow = kH / (kCols * kThreads);
  static_assert(kCtasPerRow * kCols * kThreads == kH, "a row is one word group per thread of its CTAs");
  __shared__ uint32_t s_stage;

  device::PDLWaitPrimary<true>();
  uint32_t ticket = 0;
  if (threadIdx.x == 0) {
    s_stage = ld_volatile_u32(&p.ctrl->stage);
    // taken after the stage is read: the last ticket's advance cannot reach this call
    ticket = atomicAdd(&p.ctrl->arrive, 1u);
  }
  __syncthreads();
  const int64_t stage = s_stage;
  const int row = static_cast<int>(blockIdx.x) / kCtasPerRow;
  const int col = ((static_cast<int>(blockIdx.x) % kCtasPerRow) * kThreads + static_cast<int>(threadIdx.x)) * kCols;
  const auto word = [&](int src) { return ((stage * kWorld + src) * kMaxTokens + row) * kH + col; };

  uint4 own = *reinterpret_cast<const uint4*>(p.partial + static_cast<int64_t>(row) * kH + col);
  own = make_uint4(sendable(own.x), sendable(own.y), sendable(own.z), sendable(own.w));
  mc_st_v4(p.mc + word(p.rank), own);

  uint4 v[kWorld];
#pragma unroll
  for (int r = 0; r < kWorld; ++r) v[r] = ld_volatile_v4(p.uc + word(r));
  uint32_t pending = 0;
#pragma unroll
  for (int r = 0; r < kWorld; ++r) pending |= pending_mask(v[r]) << r;
  uint32_t spins = 0;
  while (pending) {
    if (++spins > kSpinLimit) __trap();
    uint32_t still = 0;
#pragma unroll
    for (int r = 0; r < kWorld; ++r) {
      if (pending & (1u << r)) {
        v[r] = ld_volatile_v4(p.uc + word(r));
        still |= pending_mask(v[r]) << r;
      }
    }
    pending = still;
  }

  float acc[kCols] = {__uint_as_float(v[0].x), __uint_as_float(v[0].y), __uint_as_float(v[0].z),
                      __uint_as_float(v[0].w)};
#pragma unroll
  for (int r = 1; r < kWorld; ++r) {
    acc[0] = __fadd_rn(acc[0], __uint_as_float(v[r].x));
    acc[1] = __fadd_rn(acc[1], __uint_as_float(v[r].y));
    acc[2] = __fadd_rn(acc[2], __uint_as_float(v[r].z));
    acc[3] = __fadd_rn(acc[3], __uint_as_float(v[r].w));
  }
  __nv_bfloat162 lo = __floats2bfloat162_rn(acc[0], acc[1]);
  __nv_bfloat162 hi = __floats2bfloat162_rn(acc[2], acc[3]);
  uint2 packed = make_uint2(*reinterpret_cast<uint32_t*>(&lo), *reinterpret_cast<uint32_t*>(&hi));
  *reinterpret_cast<uint2*>(p.out + static_cast<int64_t>(row) * kH + col) = packed;

  const uint4 rearm = make_uint4(kNotArrived, kNotArrived, kNotArrived, kNotArrived);
#pragma unroll
  for (int r = 0; r < kWorld; ++r) st_v4(p.uc + word(r), rearm);

  if (threadIdx.x == 0 && ticket == gridDim.x - 1) {
    p.ctrl->arrive = 0u;
    p.ctrl->stage = static_cast<uint32_t>((stage + 1) % kStages);
  }
}

}  // namespace sglang

using sglang::FcAllsumParams;

extern "C" __global__ void __launch_bounds__(128) fc_allsum(const FcAllsumParams __grid_constant__ p) {
  sglang::fc_allsum_kernel<4096, MKD_WORLD>(p);
}
