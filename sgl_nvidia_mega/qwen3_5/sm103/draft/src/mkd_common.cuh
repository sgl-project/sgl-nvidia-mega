// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// Adapted from SGLang (Apache-2.0): python/sglang/kernels/jit/include/sgl_kernel/{utils,type,vec,
// warp,mbarrier}.cuh (branch nvidia/qwen35-agentx-gb300), the subset the MKD draft-boundary
// kernels use.
//
// NVRTC prelude of the MTP draft step-boundary kernels (mkd_xchg.cu, mkd_fc.cu, mkd_head.cu,
// draft_fc.cu): the same definitions as SGLang's JIT headers for sm_103 (PDL on, 32-lane warps),
// so the kernel bodies compile unchanged.

#pragma once

#include <cuda_bf16.h>

// NVRTC has no <cstdint>
typedef unsigned char uint8_t;
typedef unsigned short uint16_t;
typedef int int32_t;
typedef unsigned int uint32_t;
typedef long long int64_t;
typedef unsigned long long uint64_t;

#define SGL_DEVICE __forceinline__ __device__

namespace sglang {

using fp32_t = float;
using bf16_t = __nv_bfloat16;

namespace device {

inline constexpr uint32_t kWarpThreads = 32u;

SGL_DEVICE uint32_t get_lane_id() {
  uint32_t lane_id;
  asm volatile("mov.u32 %0, %%laneid;" : "=r"(lane_id));
  return lane_id;
}

// griddepcontrol.wait / launch_dependents (sm_90+), as SGLang's PDLWaitPrimary / PDLTriggerSecondary
template <bool kUsePDL>
SGL_DEVICE void PDLWaitPrimary() {
  if constexpr (kUsePDL) {
    asm volatile("griddepcontrol.wait;" ::: "memory");
  }
}

template <bool kUsePDL>
SGL_DEVICE void PDLTriggerSecondary() {
  if constexpr (kUsePDL) {
    asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
  }
}

namespace details {

template <typename T>
struct sized_int {};

template <>
struct sized_int<bf16_t> {
  using type = uint16_t;
};

}  // namespace details

template <typename T, int N>
struct alignas(sizeof(T) * N) AlignedStorage {
  T data[N];
};

// SGLang's AlignedVector: N elements loaded / stored as one aligned vector
template <typename T, int N>
struct AlignedVector {
 private:
  using element_t = typename details::sized_int<T>::type;
  using storage_t = AlignedStorage<element_t, N>;

 public:
  SGL_DEVICE void load(const void* ptr, int64_t offset = 0) {
    m_storage = reinterpret_cast<const storage_t*>(ptr)[offset];
  }
  SGL_DEVICE void store(void* ptr, int64_t offset = 0) const {
    reinterpret_cast<storage_t*>(ptr)[offset] = m_storage;
  }
  SGL_DEVICE T& operator[](int idx) {
    return reinterpret_cast<T*>(&m_storage)[idx];
  }
  SGL_DEVICE T operator[](int idx) const {
    return reinterpret_cast<const T*>(&m_storage)[idx];
  }

 private:
  storage_t m_storage;
};

// SGLang's DTypeTrait conversions: bf16 -> fp32 exact, fp32 -> bf16 round to nearest even
template <typename To, typename From>
SGL_DEVICE To cast(const From& value);

template <>
SGL_DEVICE fp32_t cast<fp32_t, bf16_t>(const bf16_t& value) {
  return __bfloat162float(value);
}

template <>
SGL_DEVICE bf16_t cast<bf16_t, fp32_t>(const fp32_t& value) {
  return __float2bfloat16_rn(value);
}

namespace warp {

// SGLang's warp::reduce_sum<32, 1> for fp32: xor butterfly 16, 8, 4, 2, 1, value + other
template <typename T>
SGL_DEVICE T reduce_sum(T value) {
#pragma unroll
  for (uint32_t mask = kWarpThreads / 2; mask >= 1; mask >>= 1) {
    value = static_cast<T>(value + __shfl_xor_sync(0xffffffffu, value, mask, kWarpThreads));
  }
  return value;
}

}  // namespace warp

namespace ptx {

template <typename T>
SGL_DEVICE uint32_t to_shared(T* ptr) {
  return static_cast<uint32_t>(__cvta_generic_to_shared(ptr));
}

SGL_DEVICE void mbar_init(uint64_t* bar, uint32_t count) {
  asm volatile("mbarrier.init.shared.b64 [%0], %1;" ::"r"(to_shared(bar)), "r"(count));
}

SGL_DEVICE void mbar_arrive_expect_tx(uint64_t* bar, uint32_t bytes) {
  asm volatile("mbarrier.arrive.expect_tx.shared.b64 _, [%0], %1;" ::"r"(to_shared(bar)), "r"(bytes));
}

SGL_DEVICE void mbar_wait_parity(uint64_t* bar, uint32_t parity) {
  asm volatile(
      "{\n\t.reg .pred p;\n\t"
      "WAIT_%=: mbarrier.try_wait.parity.shared.b64 p, [%0], %1;\n\t"
      "@!p bra WAIT_%=;\n\t}\n" ::"r"(to_shared(bar)),
      "r"(parity));
}

}  // namespace ptx

}  // namespace device

}  // namespace sglang
