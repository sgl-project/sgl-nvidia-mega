// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

#include "mkd_common.cuh"

namespace sglang {

namespace head_xchg {

inline constexpr int kSlots = 3;
inline constexpr int kMaxWorld = 32;
inline constexpr int kMaxRows = 512;
inline constexpr int kRowsPerCta = 8;
inline constexpr int kMaxCtas = kMaxRows / kRowsPerCta;
inline constexpr int kEpochWords = 2 + kMaxCtas;
inline constexpr int kThreads = kRowsPerCta * 32;
inline constexpr long long kFloor = (-9223372036854775807ll - 1ll);
inline constexpr uint32_t kSpinLimit = 1u << 27;

SGL_DEVICE void st_record(uint64_t* dst, uint64_t lo, uint64_t hi) {
  asm volatile("st.relaxed.sys.global.v2.u64 [%0], {%1, %2};" ::"l"(dst), "l"(lo), "l"(hi) : "memory");
}

SGL_DEVICE void ld_record(const uint64_t* src, uint64_t& lo, uint64_t& hi) {
  asm volatile("ld.relaxed.sys.global.v2.u64 {%0, %1}, [%2];" : "=l"(lo), "=l"(hi) : "l"(src) : "memory");
}

SGL_DEVICE uint32_t ld_acquire_u32(const uint32_t* p) {
  uint32_t v;
  asm volatile("ld.acquire.gpu.global.u32 %0, [%1];" : "=r"(v) : "l"(p) : "memory");
  return v;
}

}  // namespace head_xchg

struct HeadXchgParams {
  unsigned long long peers[head_xchg::kMaxWorld];  // receive buffer of every rank (this process's VA)
  long long* keys;
  long long* win_keys;      // [instances, kMaxRows] winning keys, or null
  uint32_t* done;           // [instances] head programs that released their fold, or null
  uint32_t* epoch;          // [instances, kEpochWords] tags of the next call (from 1), arrivals
  int64_t* out;             // out[inst * out_inst_stride + row * out_stride + column] = global id
  float* topk_p;            // [m] 1.0, or null (draft steps: slot draft_multi's contract)
  int64_t* positions;       // [m] += 1, or null
  int64_t* mrope;           // [3, mrope_stride] += 1 in columns [0, m), or null
  int64_t mrope_stride;
  int64_t keys_stride;
  int64_t key_row_stride;
  int64_t out_inst_stride;
  int64_t out_stride;
  int32_t m;
  int32_t world;
  int32_t rank;             // rank of instance 0
  int32_t column;
  uint32_t head_parts;      // tickets `done` reaches when the head has finished
};

__device__ __forceinline__ void head_xchg_kernel(const HeadXchgParams& p) {
  using namespace head_xchg;
  const int inst = static_cast<int>(blockIdx.y);
  const int rank = p.rank + inst;
  const int warp = static_cast<int>(threadIdx.x) >> 5;
  const int lane = static_cast<int>(threadIdx.x) & 31;
  const int row = static_cast<int>(blockIdx.x) * kRowsPerCta + warp;
  long long* keys = p.keys + inst * p.keys_stride;
  uint32_t* epoch = p.epoch + kEpochWords * inst;
  // one mode per exchange object (the two tag sequences must not share a buffer)
  uint32_t* tag_word = p.done != nullptr ? epoch : epoch + 2 + blockIdx.x;

  if (p.done != nullptr) {
    if (threadIdx.x == 0) {
      uint32_t spins = 0;
      while (ld_acquire_u32(p.done + inst) < p.head_parts) {
        if (++spins > kSpinLimit) __trap();
      }
    }
    __syncthreads();
  } else {
    device::PDLWaitPrimary<true>();
  }
  const uint32_t tag = *reinterpret_cast<volatile uint32_t*>(tag_word);
  const int64_t slot = static_cast<int64_t>(tag % kSlots);
  const unsigned long long own = p.peers[rank];

  long long best = kFloor;
  if (row < p.m) {
    long long k = 0;
    if (lane == 0) {
      k = __ldcg(keys + row * p.key_row_stride);
      __stcg(keys + row * p.key_row_stride, kFloor);
    }
    k = __shfl_sync(0xffffffffu, k, 0);
    const uint64_t t = static_cast<uint64_t>(tag) << 32;
    const uint64_t lo = t | (static_cast<uint64_t>(k) & 0xffffffffull);
    const uint64_t hi = t | (static_cast<uint64_t>(k) >> 32);
    if (lane < p.world) {
      uint64_t* dst = reinterpret_cast<uint64_t*>(p.peers[lane]) + ((slot * p.world + rank) * kMaxRows + row) * 2;
      st_record(dst, lo, hi);
      const uint64_t* src = reinterpret_cast<const uint64_t*>(own) + ((slot * p.world + lane) * kMaxRows + row) * 2;
      uint64_t a, b;
      ld_record(src, a, b);
      uint32_t spins = 0;
      while (static_cast<uint32_t>(a >> 32) != tag || static_cast<uint32_t>(b >> 32) != tag) {
        if (++spins > kSpinLimit) __trap();
        ld_record(src, a, b);
      }
      best = static_cast<long long>((b << 32) | (a & 0xffffffffull));
    }
#pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
      const long long o = __shfl_xor_sync(0xffffffffu, best, off);
      best = o > best ? o : best;
    }
    if (lane == 0) {
      const int64_t id = static_cast<int64_t>(0xffffffffu - static_cast<uint32_t>(best));
      p.out[inst * p.out_inst_stride + row * p.out_stride + p.column] = id;
      if (p.win_keys != nullptr) p.win_keys[inst * kMaxRows + row] = best;
      if (p.topk_p != nullptr) p.topk_p[row] = 1.0f;
      if (p.positions != nullptr) p.positions[row] += 1;
      if (p.mrope != nullptr) {
#pragma unroll
        for (int a = 0; a < 3; ++a) p.mrope[a * p.mrope_stride + row] += 1;
      }
    }
  }
  if (p.done != nullptr) device::PDLWaitPrimary<true>();
  device::PDLTriggerSecondary<true>();
  __syncthreads();
  if (threadIdx.x == 0) {
    const uint32_t next = tag + 1u == 0u ? 1u : tag + 1u;  // 0 is the zeroed buffer's tag
    if (p.done == nullptr) {
      *tag_word = next;
    } else {
      __threadfence();
      if (atomicAdd(epoch + 1, 1u) == gridDim.x - 1) {
        epoch[1] = 0u;
        epoch[0] = next;
        p.done[inst] = 0u;
      }
    }
  }
}

}  // namespace sglang

using sglang::HeadXchgParams;

extern "C" __global__ void __launch_bounds__(sglang::head_xchg::kThreads)
    head_xchg(const HeadXchgParams __grid_constant__ p) {
  sglang::head_xchg_kernel(p);
}
