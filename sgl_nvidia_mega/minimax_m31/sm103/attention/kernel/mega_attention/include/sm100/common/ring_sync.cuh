/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
// Synchronization primitives: monotonic u64 counters (target = count * epoch), epoch-tagged
// records, mbarriers, bulk and tensor copies, proxy fences and the optional wait deadline.
#pragma once

#include <cstdio>

#include "ring_config.cuh"

namespace attn_ring {

__device__ __forceinline__ uint32_t smem_u32(const void *p) {
  return static_cast<uint32_t>(__cvta_generic_to_shared(p));
}

// ---- wait deadline (the kDeadline instantiation, AttentionConfig.wait_timeout_s > 0) ----
__device__ __forceinline__ unsigned long long globaltimer() {
  unsigned long long t;
  asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
  return t;
}

template <bool kDeadline> struct Deadline {
  unsigned long long end;
  __device__ __forceinline__ void start(const Params &p) {
    if constexpr (kDeadline) {
      end = p.wait_timeout_ns ? globaltimer() + p.wait_timeout_ns : 0ull;
    }
  }
  // Report (cta, thread, site, observed, target), then trap; the context is lost after the
  // trap, so the report goes to the device printf buffer.
  __device__ __forceinline__ void check(const Params &p, uint32_t site, uint64_t observed,
                                        uint64_t target) const {
    if constexpr (kDeadline) {
      if (end != 0ull && globaltimer() > end) {
        printf("ring attention wait timeout: rank %d cta %d thread %d site 0x%x observed %llu "
               "target %llu\n",
               p.tp_rank, int(blockIdx.x), int(threadIdx.x), site, (unsigned long long)observed,
               (unsigned long long)target);
        asm volatile("trap;");
      }
    }
  }
};

// ---- remote wait bound (every instantiation) -------------------------------------
// A launch built with MEGA_ATTENTION_WAIT_TIMEOUT_NS > 0 traps when a poll on another
// rank's mailbox is still pending that long after its first timed poll: a peer that never
// runs this call fails the launch instead of hanging the GPU. Polls satisfied within 1024
// tries never read the timer.
#ifndef MEGA_ATTENTION_WAIT_TIMEOUT_NS
#define MEGA_ATTENTION_WAIT_TIMEOUT_NS 0
#endif
struct RemoteWait {
  uint32_t polls = 0;
  unsigned long long started = 0;
  __device__ __forceinline__ void poll() {
    if constexpr (MEGA_ATTENTION_WAIT_TIMEOUT_NS > 0) {
      if ((++polls & 1023u) != 0)
        return;
      const unsigned long long now = globaltimer();
      if (started == 0)
        started = now;
      else if (now - started > (unsigned long long)(MEGA_ATTENTION_WAIT_TIMEOUT_NS))
        __trap();
    }
  }
};

// ---- scoped global accesses ---------------------------------------------------
__device__ __forceinline__ uint64_t ld_relaxed_u64(const uint64_t *a) {
  uint64_t v;
  asm volatile("ld.relaxed.gpu.global.u64 %0, [%1];" : "=l"(v) : "l"(a) : "memory");
  return v;
}
__device__ __forceinline__ void red_relaxed_add(uint64_t *a, uint64_t v) {
  asm volatile("red.relaxed.gpu.global.add.u64 [%0], %1;" ::"l"(a), "l"(v) : "memory");
}
__device__ __forceinline__ uint64_t atom_acq_rel_add(uint64_t *a, uint64_t v) {
  uint64_t old;
  asm volatile("atom.acq_rel.gpu.global.add.u64 %0, [%1], %2;"
               : "=l"(old)
               : "l"(a), "l"(v)
               : "memory");
  return old;
}
__device__ __forceinline__ void fence_acquire_gpu() {
  asm volatile("fence.acquire.gpu;" ::: "memory");
}
__device__ __forceinline__ void fence_acq_rel_gpu() {
  asm volatile("fence.acq_rel.gpu;" ::: "memory");
}
// Release of several counters with one fence: "fence.acq_rel; red.relaxed [M]" is a release pattern
// on every M (PTX memory model, release patterns, form 3). red.release lowers to MEMBAR.GPU + RED,
// so k back-to-back red.release cost k serialized round trips; this costs one.
__device__ __forceinline__ void release_counters(uint64_t *const *words, int count) {
  fence_acq_rel_gpu();
  for (int i = 0; i < count; ++i)
    red_relaxed_add(words[i], 1ull);
}
// Order this thread's generic global writes before later async-proxy (TMA / bulk) reads.
__device__ __forceinline__ void fence_proxy_async_global() {
  asm volatile("fence.proxy.async.global;" ::: "memory");
}
// Read-only loads the compiler may not sink below a later poll loop (asm volatile keeps the issue
// point): prefetches of inputs whose latency should overlap a cross-CTA wait.
__device__ __forceinline__ uint32_t ldg_nc_u32(const void *a) {
  uint32_t v;
  asm volatile("ld.global.nc.u32 %0, [%1];" : "=r"(v) : "l"(a));
  return v;
}
__device__ __forceinline__ uint4 ldg_nc_128(const void *a) {
  uint4 v;
  asm volatile("ld.global.nc.v4.u32 {%0,%1,%2,%3}, [%4];"
               : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w)
               : "l"(a));
  return v;
}
// 16-byte records: a scalar .b128 access is single-copy atomic, so the payload and its tag are
// observed together (a .v4.u32 access is four accesses and may tear).
__device__ __forceinline__ void st_rec(void *a, uint32_t w0, uint32_t w1, uint32_t w2,
                                       uint32_t tag) {
  const uint64_t lo = uint64_t(w0) | (uint64_t(w1) << 32),
                 hi = uint64_t(w2) | (uint64_t(tag) << 32);
  asm volatile(
      "{\n\t.reg .b128 r;\n\tmov.b128 r, {%1, %2};\n\tst.relaxed.gpu.global.b128 [%0], r;\n\t}" ::
          "l"(a),
      "l"(lo), "l"(hi)
      : "memory");
}
// 8-byte records {32-bit payload, 32-bit tag}: one single-copy-atomic .b64 store.
__device__ __forceinline__ void st_rec64(void *a, uint32_t payload, uint32_t tag) {
  asm volatile("st.relaxed.gpu.global.b64 [%0], %1;" ::"l"(a),
               "l"(uint64_t(payload) | (uint64_t(tag) << 32))
               : "memory");
}
__device__ __forceinline__ uint2 ld_rec64(const void *a) {
  uint64_t v;
  asm volatile("ld.relaxed.gpu.global.b64 %0, [%1];" : "=l"(v) : "l"(a) : "memory");
  return make_uint2(uint32_t(v), uint32_t(v >> 32));
}
__device__ __forceinline__ uint4 ld_rec(const void *a) {
  uint64_t lo, hi;
  asm volatile(
      "{\n\t.reg .b128 r;\n\tld.relaxed.gpu.global.b128 r, [%2];\n\tmov.b128 {%0, %1}, r;\n\t}"
      : "=l"(lo), "=l"(hi)
      : "l"(a)
      : "memory");
  return make_uint4(uint32_t(lo), uint32_t(lo >> 32), uint32_t(hi), uint32_t(hi >> 32));
}

// ---- asynchronous global -> shared copies (no registers held while in flight) ----
__device__ __forceinline__ void cp_async_4(uint32_t dst, const void *src) {
  asm volatile("cp.async.ca.shared.global [%0], [%1], 4;" ::"r"(dst), "l"(src) : "memory");
}
__device__ __forceinline__ void cp_async_commit() {
  asm volatile("cp.async.commit_group;" ::: "memory");
}
__device__ __forceinline__ void cp_async_wait_all() {
  asm volatile("cp.async.wait_all;" ::: "memory");
}

// ---- named barriers (0: whole CTA; 1: consumer warps; 2: runtime TopK; 6: COMBINE merge) ----
__device__ __forceinline__ void consumer_sync() {
  asm volatile("bar.sync 1, %0;" ::"n"(kConsumerThreads) : "memory");
}

// ---- monotonic counters ---------------------------------------------------------
struct Counter {
  uint64_t *word;
  __device__ __forceinline__ static Counter at(uint8_t *arena_base, int id) {
    return Counter{
        reinterpret_cast<uint64_t *>(arena_base + arena::kCounters + size_t(id) * arena::kLine)};
  }
  template <bool kDeadline>
  __device__ __forceinline__ void spin(const Params &p, const Deadline<kDeadline> &d, uint32_t site,
                                       uint64_t target) const {
    uint64_t v = ld_relaxed_u64(word);
    while (v < target) {
      d.check(p, site, v, target);
      v = ld_relaxed_u64(word); // no back-off: every hop is on the critical path
    }
    fence_acquire_gpu();
  }
};

// ---- mbarrier / bulk copy ---------------------------------------------------------
__device__ __forceinline__ void mbar_init(uint32_t bar, uint32_t count) {
  asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" ::"r"(bar), "r"(count) : "memory");
}
__device__ __forceinline__ void mbar_fence_init() {
  asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
  asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
}
__device__ __forceinline__ bool mbar_try(uint32_t bar, uint32_t parity) {
  uint32_t ok;
  asm volatile("{\n\t.reg .pred P1;\n\tmbarrier.try_wait.parity.shared::cta.b64 P1, [%1], %2;\n\t"
               "selp.u32 %0, 1, 0, P1;\n\t}"
               : "=r"(ok)
               : "r"(bar), "r"(parity)
               : "memory");
  return ok != 0;
}
template <bool kDeadline>
__device__ __forceinline__ void mbar_wait(const Params &p, const Deadline<kDeadline> &d,
                                          uint32_t site, uint32_t bar, uint32_t parity) {
  while (!mbar_try(bar, parity)) {
    d.check(p, site, parity, bar);
  }
}
__device__ __forceinline__ void mbar_arrive(uint32_t bar) {
  asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];" ::"r"(bar) : "memory");
}
__device__ __forceinline__ void mbar_arrive_expect_tx(uint32_t bar, uint32_t bytes) {
  asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;" ::"r"(bar), "r"(bytes)
               : "memory");
}
__device__ __forceinline__ uint64_t l2_evict_first() {
  uint64_t policy;
  asm volatile("createpolicy.fractional.L2::evict_first.b64 %0, 1.0;" : "=l"(policy));
  return policy;
}
__device__ __forceinline__ void bulk_g2s(uint32_t dst, const void *src, uint32_t bytes,
                                         uint32_t bar, uint64_t policy) {
  asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes.L2::cache_hint "
               "[%0], [%1], %2, [%3], %4;" ::"r"(dst),
               "l"(src), "r"(bytes), "r"(bar), "l"(policy)
               : "memory");
}
__device__ __forceinline__ void tma_load_3d(uint32_t dst, const CUtensorMap *map, int c0, int c1,
                                            int c2, uint32_t bar, uint64_t policy) {
  asm volatile("cp.async.bulk.tensor.3d.shared::cluster.global.mbarrier::complete_tx::bytes"
               ".L2::cache_hint [%0], [%1, {%2, %3, %4}], [%5], %6;" ::"r"(dst),
               "l"(reinterpret_cast<uint64_t>(map)), "r"(c0), "r"(c1), "r"(c2), "r"(bar),
               "l"(policy)
               : "memory");
}
__device__ __forceinline__ void prefetch_tensormap(const CUtensorMap *map) {
  asm volatile("prefetch.tensormap [%0];" ::"l"(reinterpret_cast<uint64_t>(map)) : "memory");
}
__device__ __forceinline__ void prefetch_l2(const void *src, uint32_t bytes) {
  asm volatile("cp.async.bulk.prefetch.L2.global [%0], %1;" ::"l"(src), "r"(bytes) : "memory");
}

// ---- shared-memory access by address ----------------------------------------------
__device__ __forceinline__ uint2 lds_64(uint32_t a) {
  uint2 v;
  asm volatile("ld.shared.v2.u32 {%0,%1}, [%2];" : "=r"(v.x), "=r"(v.y) : "r"(a));
  return v;
}
__device__ __forceinline__ uint4 lds_128(uint32_t a) {
  uint4 v;
  asm volatile("ld.shared.v4.u32 {%0,%1,%2,%3}, [%4];"
               : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w)
               : "r"(a));
  return v;
}
__device__ __forceinline__ void sts_128(uint32_t a, uint4 v) {
  asm volatile("st.shared.v4.u32 [%0], {%1,%2,%3,%4};" ::"r"(a), "r"(v.x), "r"(v.y), "r"(v.z),
               "r"(v.w)
               : "memory");
}
__device__ __forceinline__ void sts_u8(uint32_t a, uint32_t v) {
  asm volatile("st.shared.u8 [%0], %1;" ::"r"(a), "h"(uint16_t(v)) : "memory");
}

__device__ __forceinline__ float warp_sum(float v) {
#pragma unroll
  for (int o = 16; o; o >>= 1)
    v += __shfl_xor_sync(0xffffffffu, v, o);
  return v;
}

} // namespace attn_ring
