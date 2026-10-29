#pragma once

// Waits with the deadline trap, shared-memory copies, the mbarrier ring and TMA, the
// cross-CTA / cross-rank hand-over words and the per-call context passed to every OP.

#include "include/sm100/common/numeric.cuh"

namespace m31_draft {
namespace {

__device__ __forceinline__ unsigned long long globaltimer() {
  unsigned long long t;
  asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
  return t;
}
// A wait longer than kWaitTimeoutNs (e.g. a rank that never issues the call) traps, so
// the launch fails with an error instead of hanging the device.
__device__ __forceinline__ void check_deadline(unsigned long long start) {
  if (globaltimer() - start > kWaitTimeoutNs)
    __trap();
}

__device__ __forceinline__ uint32_t smem_u32(void const *p) {
  return static_cast<uint32_t>(__cvta_generic_to_shared(p));
}
// 16-byte global -> shared copies that bypass L1 (data written by other CTAs), all in flight
// together.
__device__ __forceinline__ void cp_async_16(uint32_t dst, void const *src) {
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" ::"r"(dst), "l"(src) : "memory");
}
// 8-byte global -> shared copy through L1 (read-only or this CTA's own earlier writes).
__device__ __forceinline__ void cp_async_8(uint32_t dst, void const *src) {
  asm volatile("cp.async.ca.shared.global [%0], [%1], 8;" ::"r"(dst), "l"(src) : "memory");
}
__device__ __forceinline__ void cp_async_wait_all() {
  asm volatile("cp.async.wait_all;" ::: "memory");
}
__device__ __forceinline__ void lds_u64(uint32_t addr, uint32_t (&v)[2]) {
  asm volatile("ld.shared.v2.u32 {%0, %1}, [%2];" : "=r"(v[0]), "=r"(v[1]) : "r"(addr) : "memory");
}
// the low 48 bits, 2-byte aligned
__device__ __forceinline__ void sts_u16x3(uint32_t addr, uint64_t v) {
  asm volatile(
      "st.shared.u16 [%0], %1;\n\tst.shared.u16 [%0+2], %2;\n\tst.shared.u16 [%0+4], %3;" ::"r"(
          addr),
      "h"(uint16_t(v)), "h"(uint16_t(v >> 16)), "h"(uint16_t(v >> 32))
      : "memory");
}
__device__ __forceinline__ uint32_t lds_u32(uint32_t addr) {
  uint32_t v;
  asm volatile("ld.shared.u32 %0, [%1];" : "=r"(v) : "r"(addr) : "memory");
  return v;
}
__device__ __forceinline__ void ldsm_x4(uint32_t (&r)[4], uint32_t addr) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];"
               : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3])
               : "r"(addr)
               : "memory");
}
__device__ __forceinline__ void ldsm_x4_t(uint32_t (&r)[4], uint32_t addr) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];"
               : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3])
               : "r"(addr)
               : "memory");
}

// mbarrier ring and TMA.
__device__ __forceinline__ void mbar_init(uint32_t bar, uint32_t count) {
  asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" ::"r"(bar), "r"(count) : "memory");
}
__device__ __forceinline__ void mbar_expect_tx(uint32_t bar, uint32_t bytes) {
  asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;" ::"r"(bar), "r"(bytes)
               : "memory");
}
__device__ __forceinline__ void mbar_arrive(uint32_t bar) {
  asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];" ::"r"(bar) : "memory");
}
__device__ __forceinline__ bool mbar_try_wait(uint32_t bar, uint32_t parity) {
  uint32_t ok;
  asm volatile("{\n .reg .pred p;\n mbarrier.try_wait.parity.shared::cta.b64 p, [%1], %2;\n "
               "selp.u32 %0, 1, 0, p;\n}\n"
               : "=r"(ok)
               : "r"(bar), "r"(parity)
               : "memory");
  return ok != 0;
}
__device__ __forceinline__ void mbar_wait(uint32_t bar, uint32_t parity) {
  if (mbar_try_wait(bar, parity))
    return;
  unsigned long long const start = globaltimer();
  while (!mbar_try_wait(bar, parity))
    check_deadline(start);
}
// Weights are streamed once per call: they go first when L2 needs room, so the
// latency-critical inputs prefetched at entry and the activations stay resident.
__device__ __forceinline__ uint64_t evict_first_policy() {
  uint64_t policy;
  asm volatile("createpolicy.fractional.L2::evict_first.b64 %0, 1.0;" : "=l"(policy));
  return policy;
}
__device__ __forceinline__ void bulk_load(uint32_t dst, void const *src, uint32_t bytes,
                                          uint32_t bar, uint64_t policy) {
  asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes.L2::cache_hint "
               "[%0], [%1], %2, [%3], %4;" ::"r"(dst),
               "l"(src), "r"(bytes), "r"(bar), "l"(policy)
               : "memory");
}
__device__ __forceinline__ void tma_load_3d(uint32_t dst, CUtensorMap const *map, int c0, int c1,
                                            int c2, uint32_t bar, uint64_t policy) {
  asm volatile("cp.async.bulk.tensor.3d.shared::cluster.global.tile.mbarrier::complete_tx::bytes."
               "L2::cache_hint "
               "[%0], [%1, {%2, %3, %4}], [%5], %6;" ::"r"(dst),
               "l"(reinterpret_cast<uint64_t>(map)), "r"(c0), "r"(c1), "r"(c2), "r"(bar),
               "l"(policy)
               : "memory");
}
__device__ __forceinline__ void bulk_prefetch_l2(void const *src, uint32_t bytes) {
  asm volatile("cp.async.bulk.prefetch.L2.global [%0], %1;" ::"l"(src), "r"(bytes) : "memory");
}

// Cross-CTA hand-over (GPU scope) and the cross-rank TP4 words (system scope).
__device__ __forceinline__ uint64_t ld_relaxed_gpu_u64(uint64_t const *p) {
  uint64_t v;
  asm volatile("ld.relaxed.gpu.global.u64 %0, [%1];" : "=l"(v) : "l"(p) : "memory");
  return v;
}
__device__ __forceinline__ void st_relaxed_gpu_u64(uint64_t *p, uint64_t v) {
  asm volatile("st.relaxed.gpu.global.u64 [%0], %1;" ::"l"(p), "l"(v) : "memory");
}
__device__ __forceinline__ uint32_t ld_relaxed_gpu(uint32_t const *p) {
  uint32_t v;
  asm volatile("ld.relaxed.gpu.global.u32 %0, [%1];" : "=r"(v) : "l"(p) : "memory");
  return v;
}
__device__ __forceinline__ void fence_acq_rel_gpu() {
  asm volatile("fence.acq_rel.gpu;" ::: "memory");
}
__device__ __forceinline__ uint32_t ld_acquire_gpu(uint32_t const *p) {
  uint32_t v;
  asm volatile("ld.acquire.gpu.global.u32 %0, [%1];" : "=r"(v) : "l"(p) : "memory");
  return v;
}
__device__ __forceinline__ void red_add_release_gpu(uint32_t *p, uint32_t v) {
  asm volatile("red.release.gpu.global.add.u32 [%0], %1;" ::"l"(p), "r"(v) : "memory");
}
__device__ __forceinline__ void st_ll2(uint64_t *p, uint64_t a, uint64_t b) {
  asm volatile("st.relaxed.sys.global.v2.u64 [%0], {%1, %2};" ::"l"(p), "l"(a), "l"(b) : "memory");
}
__device__ __forceinline__ void st_ll(uint64_t *p, uint64_t a) {
  asm volatile("st.relaxed.sys.global.u64 [%0], %1;" ::"l"(p), "l"(a) : "memory");
}
__device__ __forceinline__ uint64_t ld_relaxed_sys_u64(uint64_t const *p) {
  uint64_t v;
  asm volatile("ld.relaxed.sys.global.u64 %0, [%1];" : "=l"(v) : "l"(p) : "memory");
  return v;
}
__device__ __forceinline__ void ld_ll2(uint64_t const *p, uint64_t &a, uint64_t &b) {
  asm volatile("ld.relaxed.sys.global.v2.u64 {%0, %1}, [%2];"
               : "=l"(a), "=l"(b)
               : "l"(p)
               : "memory");
}
__device__ __forceinline__ void ld_relaxed_gpu_v2(uint64_t const *p, uint64_t &a, uint64_t &b) {
  asm volatile("ld.relaxed.gpu.global.v2.u64 {%0, %1}, [%2];"
               : "=l"(a), "=l"(b)
               : "l"(p)
               : "memory");
}
__device__ __forceinline__ void consumer_sync() {
  asm volatile("bar.sync 1, %0;" ::"n"(kConsumerThreads) : "memory");
}

// LL6 words: 48 payload bits and a 16-bit tag of the call generation (never 0, so the
// zeroed arena never validates; consecutive calls always differ).
__device__ __forceinline__ uint32_t ll_tag(uint32_t epoch) { return epoch % 65535u + 1u; }
__device__ __forceinline__ bool ll_ok(uint64_t word, uint32_t tag) {
  return uint32_t(word >> 48) == tag;
}
__device__ __forceinline__ uint64_t ll_word(uint64_t payload, uint32_t tag) {
  return payload | (uint64_t(tag) << 48);
}

// One thread waits until a monotonic counter reaches this call's target.
__device__ void wait_count(uint32_t const *counter, uint32_t target) {
  if (int32_t(ld_acquire_gpu(counter) - target) >= 0)
    return;
  unsigned long long const start = globaltimer();
  while (int32_t(ld_acquire_gpu(counter) - target) < 0)
    check_deadline(start);
}

// The CTA's weight ring: kRing slots of kSlotBytes with a full barrier (TMA transaction
// bytes) and an empty barrier (one arrival per consumer warp) per slot. Both sides count
// items over the whole call; the slot and the barrier parity derive from the item index.
struct Ring {
  uint32_t payload; // shared address of slot 0
  SmemMisc *misc;
  __device__ uint32_t slot(uint32_t item) const { return payload + (item % kRing) * kSlotBytes; }
  __device__ uint32_t full(uint32_t item) const { return smem_u32(&misc->full[item % kRing]); }
  __device__ uint32_t empty(uint32_t item) const { return smem_u32(&misc->empty[item % kRing]); }
};
__device__ __forceinline__ void init_ring(SmemMisc *misc) {
  for (int i = 0; i < kRing; ++i) {
    mbar_init(smem_u32(&misc->full[i]), 1);
    mbar_init(smem_u32(&misc->empty[i]), kConsumerWarps);
  }
}
// The weight-stream warp's side: wait until the item's slot is free, then fill it.
struct Producer {
  Ring ring;
  uint32_t item = 0;
  __device__ uint32_t acquire() const {
    if (item >= kRing)
      mbar_wait(ring.empty(item), ((item / kRing) & 1u) ^ 1u);
    return ring.slot(item);
  }
  __device__ uint32_t full() const { return ring.full(item); }
  __device__ void advance() { ++item; }
};
// A consumer warp's side: wait until the item's stage landed, read it, then release it.
struct Consumer {
  Ring ring;
  uint32_t item = 0;
  __device__ uint32_t acquire() const {
    mbar_wait(ring.full(item), (item / kRing) & 1u);
    return ring.slot(item);
  }
  __device__ void retire(int lane) {
    __syncwarp();
    if (lane == 0)
      mbar_arrive(ring.empty(item));
    ++item;
  }
};

struct Ctx;
template <class T> __device__ __forceinline__ T *at(uint8_t *base, size_t offset) {
  return reinterpret_cast<T *>(base + offset);
}
__device__ __forceinline__ uint32_t *signal(uint8_t *arena, int kind, int index, int id) {
  return at<uint32_t>(
      arena,
      kSignals + ((size_t(kind) * kSignalSlots + index) * kSignalSubs + id % kSignalSubs) * 128);
}
// TP4 words of one token row from one sender, in a receiver's arena.
__device__ __forceinline__ uint64_t *ll_row(uint8_t *arena, int buf, int sender, int t) {
  return at<uint64_t>(arena, kLL) + (size_t(buf * kTp + sender) * kRows + t) * kLLWords;
}
__device__ __forceinline__ uint64_t *act_ll(uint8_t *arena, int stage) {
  return at<uint64_t>(arena, kActLL) + size_t(stage) * kActLLWords;
}
__device__ __forceinline__ uint64_t *qkv_ll(uint8_t *arena, int layer) {
  return at<uint64_t>(arena, kQkvLL) + size_t(layer) * kQkvLLWords;
}
__device__ __forceinline__ uint64_t *swi_ll(uint8_t *arena, int layer) {
  return at<uint64_t>(arena, kSwiLL) + size_t(layer) * kSwiLLWords;
}

struct Ctx {
  Args const &a;
  uint8_t *arena; // this rank's
  uint8_t *smem;
  SmemMisc *misc;
  uint32_t epoch; // this call's generation, 1-based
  int cta, grid, tid, warp, lane;
  Consumer weights; // this warp's side of the weight ring
};

// Waits until all `arrivals` (ids 0 .. arrivals-1, one red per id and call) of this call
// reached a split counter: thread s polls sub-counter s, then acquires.
__device__ void wait_signal(Ctx const &c, int kind, int index, uint32_t arrivals) {
  if (c.tid < kSignalSubs) {
    uint32_t const per = (arrivals + kSignalSubs - 1 - uint32_t(c.tid)) / kSignalSubs;
    if (per != 0u) {
      uint32_t const *counter = signal(c.arena, kind, index, c.tid);
      uint32_t const target = c.epoch * per;
      // Acquire polls: the load that sees the last arrival orders this CTA's later reads (through
      // the barrier below) after the producers' releases, with no separate fence.
      unsigned long long start = 0;
      for (uint32_t spin = 0; int32_t(ld_acquire_gpu(counter) - target) < 0; ++spin) {
        if ((spin & 255u) == 0u) {
          if (start == 0)
            start = globaltimer();
          else
            check_deadline(start);
        }
      }
    }
  }
  consumer_sync();
}

__device__ __forceinline__ uint32_t sum_generation(Ctx const &c, int stage) {
  return (c.epoch - 1u) * uint32_t(kSums) + uint32_t(stage);
}

// Polling gates. A hand-off's consumers first spin (one thread per gate word) on a word of the
// GEMM's last gemm unit (last_gemm_unit, which ends the phase), and only then every consumer thread
// polls its share of the hand-off words. The full poll re-reads every missing word each round;
// without the gate a GEMM's ~148 consumer CTAs would keep re-reading for the whole preceding wait
// (TP4 sum and norm). The gate words are written every call like every other hand-off word, and the
// full poll still validates every word: the gate only shapes timing.
__device__ int gemm_gate_word(int g, int grid);
__device__ __forceinline__ void gate_ll6(Ctx const &c, uint64_t const *word) {
  if (c.tid == 0) {
    uint32_t const tag = ll_tag(c.epoch);
    unsigned long long start = 0;
    for (uint32_t spin = 0; !ll_ok(ld_relaxed_gpu_u64(word), tag); ++spin) {
      if ((spin & 63u) == 0u) {
        if (start == 0)
          start = globaltimer();
        else
          check_deadline(start);
      }
    }
  }
  consumer_sync();
}
// The TP4 sum of norm stage `stage` is (nearly) complete: every sender's gate word of the GEMM
// that produced it; the embedding (stage 0) gates on word 0, which CTA 0 writes.
__device__ __forceinline__ void gate_tp4(Ctx const &c, int stage) {
  uint32_t const gen = sum_generation(c, stage);
  if (c.tid < kTp) {
    // stage 2l + 1 follows o_proj, stage 2l + 2 follows down
    int const word_index = stage == 0 ? 0 : gemm_gate_word(stage % 2 ? kGemmO : kGemmDown, c.grid);
    uint64_t const *word = ll_row(c.arena, int(gen & 1u), c.tid, 0) + word_index;
    unsigned long long start = 0;
    for (uint32_t spin = 0; uint32_t(ld_relaxed_sys_u64(word) >> 32) != gen + 1u; ++spin) {
      if ((spin & 63u) == 0u) {
        if (start == 0)
          start = globaltimer();
        else
          check_deadline(start);
      }
    }
  }
  consumer_sync();
}

} // namespace
} // namespace m31_draft
