/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
// Producer/consumer page ring with kRingSlots slots (generalizes the 4-slot msa_v2
// m3_ring.cuh). Item i uses slot i % N with phase (i / N) & 1. One producer thread fills
// each item (full[s] count 1 plus transaction bytes); every consumer warp releases it once
// (empty[s] count kConsumerWarps). Items that are never released ("terminal") must form a
// suffix of the per-CTA item sequence, at most kRingSlots long.
#pragma once

#include "ring_sync.cuh"

namespace attn_ring {

struct Ring {
  uint32_t base;  // shared address of slot 0 (1024-byte aligned)
  uint32_t full;  // full[0]; full[s] = full + 8 s
  uint32_t empty; // empty[0]
  __device__ __forceinline__ uint32_t slot(uint32_t item) const {
    return base + (item % kRingSlots) * kSlotBytes;
  }
  __device__ __forceinline__ uint32_t full_bar(uint32_t item) const {
    return full + (item % kRingSlots) * 8u;
  }
  __device__ __forceinline__ uint32_t empty_bar(uint32_t item) const {
    return empty + (item % kRingSlots) * 8u;
  }
  __device__ __forceinline__ static uint32_t phase(uint32_t item) {
    return (item / kRingSlots) & 1u;
  }
};

// bars: uint64_t[2 * kRingSlots] in static shared memory; initialised by one leader thread.
__device__ __forceinline__ void ring_init(uint64_t *bars, bool leader) {
  if (leader) {
    for (int s = 0; s < kRingSlots; ++s) {
      mbar_init(smem_u32(bars + s), 1);
      mbar_init(smem_u32(bars + kRingSlots + s), kConsumerWarps);
    }
    mbar_fence_init();
  }
}
__device__ __forceinline__ Ring make_ring(uint8_t *ring_mem, uint64_t *bars) {
  return Ring{smem_u32(ring_mem), smem_u32(bars), smem_u32(bars + kRingSlots)};
}

// ---- consumer side (all 256 consumer threads call with the same item) -------------
template <bool kDeadline>
__device__ __forceinline__ uint32_t ring_acquire(const Params &p, const Deadline<kDeadline> &d,
                                                 const Ring &ring, uint32_t item) {
  mbar_wait<kDeadline>(p, d, 0x100u | (item & 0xffu), ring.full_bar(item), Ring::phase(item));
  return ring.slot(item);
}
// Every lane of the warp has retired its last read of the slot before lane 0 arrives.
__device__ __forceinline__ void ring_release(const Ring &ring, uint32_t item, int lane) {
  __syncwarp();
  if (lane == 0)
    mbar_arrive(ring.empty_bar(item));
}

// ---- producer side (one thread) ---------------------------------------------------
template <bool kDeadline>
__device__ __forceinline__ uint32_t ring_reserve(const Params &p, const Deadline<kDeadline> &d,
                                                 const Ring &ring, uint32_t item) {
  mbar_wait<kDeadline>(p, d, 0x200u | (item & 0xffu), ring.empty_bar(item), Ring::phase(item) ^ 1u);
  return ring.slot(item);
}
__device__ __forceinline__ void ring_expect(const Ring &ring, uint32_t item, uint32_t bytes) {
  mbar_arrive_expect_tx(ring.full_bar(item), bytes);
}
// An item with no payload ("hole"): complete the phase without transactions.
__device__ __forceinline__ void ring_fill_empty(const Ring &ring, uint32_t item) {
  mbar_arrive(ring.full_bar(item));
}

__device__ __forceinline__ void setmaxnreg_consumer() {
  asm volatile("setmaxnreg.inc.sync.aligned.u32 %0;" ::"n"(kConsumerRegs));
}
__device__ __forceinline__ void setmaxnreg_producer() {
  asm volatile("setmaxnreg.dec.sync.aligned.u32 %0;" ::"n"(kProducerRegs));
}

} // namespace attn_ring
