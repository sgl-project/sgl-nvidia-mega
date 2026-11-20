// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// Exact work-slot claim for persistent grids that map their work to slots, one slot per CTA, with the SM's locality
// domain as a placement hint.
//
// Why: the PTX ISA (Special Registers: %smid) says %smid returns the SM a thread runs on at the moment it is read, that
// the value may change during execution (for example after preemption), and that it is meant for profiling and
// diagnostic code. A slot computed only from %smid can therefore be taken twice in one launch while another slot is
// never taken. Here %smid only picks the preferred slot; which slots are covered never depends on it.
//
// Slots: slot = domain << 7 | rank (rank < n_dom[domain], n_dom0 + n_dom1 == G == gridDim.x <= 256); bit index
// i = rank for domain 0 and n_dom0 + rank for domain 1, in a bitmap of kBitmapWords 32-bit words.
// Protocol, per call site (one claim area per control block):
//   claim:  one thread per CTA sets the bit of its hinted slot with one atomic OR (atom.relaxed.gpu.or). If the bit was
//           already set, it sets the first clear bit of its own domain's range, then of the other domain's, the same
//           way. Atomic operations on one word are totally ordered, so each bit is won by exactly one CTA, and G CTAs
//           that each win one bit cover all G slots.
//   finish: once per CTA, ordered after its claim (program order, or a block barrier in between: PTX ISA, Memory
//           Consistency Model, Memory synchronization), one atom.acq_rel.gpu.inc on the claim counter with bound G - 1,
//           so the counter is 0 again after each launch. The CTA that reads G - 1 is the last one; through the chain of
//           atomic increments it observes every other CTA's claim (PTX ISA, Memory Consistency Model, Observation
//           order). It checks that the bitmap holds exactly the G low bits, clears it for the next launch of the site
//           and counts one verified launch. Launches of one call site are ordered by completion, as every other word of
//           the control block already requires (stream order: completion of a task synchronizes with the start of the
//           next one), so the next launch starts from a clear bitmap and a zero counter.
//   failure: a CTA that finds no clear bit, or a last CTA that finds the bitmap not exactly full, sets kErrClaim in the
//           sticky error word and traps, so a launch never completes with a slot silently uncovered.
// The previous form (slot = table[%smid], no claim) compiles only with -DMUSE_VR_SLOT_UNSAFE=1 (not guaranteed by the
// PTX ISA; kept for A/B captures). -DMUSE_VR_SLOT_TRACE=1 (debug builds) records, per CTA and launch, %smid at the
// claim and at exit plus the hinted and the claimed slot, so tests can check coverage launch by launch.

#pragma once

#include <cstdint>

#ifndef MUSE_VR_SLOT_UNSAFE
#define MUSE_VR_SLOT_UNSAFE 0
#endif
// Launch-counter form (MoE-LD-W, -DMUSE_VR_SLOT_EPOCH=1): see the section near the end of this file.
#ifndef MUSE_VR_SLOT_EPOCH
#define MUSE_VR_SLOT_EPOCH 0
#endif
#ifndef MUSE_VR_SLOT_TRACE
#define MUSE_VR_SLOT_TRACE 0
#endif
// Tests only: 1 = odd CTAs prefer rank 0 of their domain (collisions: exercises the search for a free slot);
// 2 = CTA 0 skips its claim (negative control: the last CTA must find the bitmap incomplete and trap);
// 3 = the kernels delay one FC1 epilogue warp per tile (warp skew; see their FC1 publish);
// 4 = the last CTA of each launch sleeps ~200 us (200 x __nanosleep(1000)) before it checks and clears the bitmap (a late finish of
//     the previous launch: tests/test_moe_ldw_order.py, the start-order test of MoE-LD-W's claim); a claim that finds no free
//     slot then scans again (bounded) instead of failing at once, so the test's pre-wait negative control runs clean.
#ifndef MUSE_VR_SLOT_TEST
#define MUSE_VR_SLOT_TEST 0
#endif

namespace muse_vr_slot {

constexpr int kBitmapWords = 8;                    // 256 slots
constexpr int kCounterWord = kBitmapWords;         // claims of the current launch (0 between launches)
constexpr int kMissWord = kBitmapWords + 1;        // CTAs whose hinted slot was already taken (all launches)
constexpr int kVerifiedWord = kBitmapWords + 2;    // launches whose last CTA found the bitmap exactly full
constexpr int kWords = kBitmapWords + 3;           // size of the claim area
constexpr int kErrClaim = 256;                     // sticky error bit: claim failed or coverage incomplete
constexpr unsigned kInvalid = 0xFFFFFFFFu;

__device__ __forceinline__ unsigned atom_or_relaxed_gpu(unsigned* p, unsigned v) {
  unsigned old;
  asm volatile("atom.relaxed.gpu.global.or.b32 %0, [%1], %2;" : "=r"(old) : "l"(p), "r"(v) : "memory");
  return old;
}
__device__ __forceinline__ unsigned atom_inc_acq_rel_gpu(unsigned* p, unsigned bound) {
  unsigned old;
  asm volatile("atom.acq_rel.gpu.global.inc.u32 %0, [%1], %2;" : "=r"(old) : "l"(p), "r"(bound) : "memory");
  return old;
}
__device__ __forceinline__ unsigned ld_relaxed_gpu(const unsigned* p) {
  unsigned v;
  asm volatile("ld.relaxed.gpu.global.u32 %0, [%1];" : "=r"(v) : "l"(p) : "memory");
  return v;
}
__device__ __forceinline__ void st_relaxed_gpu(unsigned* p, unsigned v) {
  asm volatile("st.relaxed.gpu.global.u32 [%0], %1;" ::"l"(p), "r"(v) : "memory");
}
__device__ __forceinline__ void red_add_relaxed_gpu(unsigned* p, unsigned v) {
  asm volatile("red.relaxed.gpu.global.add.u32 [%0], %1;" ::"l"(p), "r"(v) : "memory");
}
__device__ __forceinline__ unsigned read_smid() {
  unsigned smid;
  asm volatile("mov.u32 %0, %%smid;" : "=r"(smid));
  return smid;
}
__device__ __noinline__ void claim_fail(int32_t* err) {
  atomicOr(err, kErrClaim);
  __threadfence();
  __trap();
}

// bit index of a slot (kInvalid when the slot is outside the domains' ranges)
__device__ __forceinline__ unsigned index_of(unsigned slot, int n_dom0, int n_dom1) {
  const unsigned dom = (slot >> 7) & 1u, r = slot & 127u;
  if (r >= unsigned(dom ? n_dom1 : n_dom0)) return kInvalid;
  return dom ? unsigned(n_dom0) + r : r;
}
__device__ __forceinline__ unsigned slot_of(unsigned i, int n_dom0) {
  return i < unsigned(n_dom0) ? i : (1u << 7) | (i - unsigned(n_dom0));
}

// Step 1 (one thread): try the hinted slot. Returns the bitmap word as it was before this CTA's bit was set, or
// kInvalid for a hint outside the table; its value is first needed by settle(), so the caller can overlap other loads.
__device__ __forceinline__ unsigned test_hint(unsigned hint) {
#if MUSE_VR_SLOT_TEST == 1
  if (blockIdx.x & 1u) hint &= ~127u;
#endif
  return hint;
}
__device__ __forceinline__ unsigned try_hint(unsigned* area, unsigned hint, int n_dom0, int n_dom1) {
#if MUSE_VR_SLOT_TEST == 2
  if (blockIdx.x == 0) return 0u;
#endif
  hint = test_hint(hint);
  const unsigned i = index_of(hint, n_dom0, n_dom1);
  if (i == kInvalid) return kInvalid;
  return atom_or_relaxed_gpu(area + (i >> 5), 1u << (i & 31u));
}

// Step 2 (same thread): the claimed slot. Takes the hinted slot when try_hint() won its bit; otherwise the first clear
// bit of the hint's domain, then of the other domain (bounded: at most one atomic per slot); traps when none is left.
__device__ __noinline__ unsigned settle_slow(unsigned* area, unsigned hint, int n_dom0, int n_dom1, int32_t* err) {
  red_add_relaxed_gpu(area + kMissWord, 1u);
  const unsigned g = unsigned(n_dom0 + n_dom1);
  const unsigned first_dom = (hint >> 7) & 1u;
#if MUSE_VR_SLOT_TEST == 4
  // tests only: a claim that ran into the previous launch's still-full bitmap (the order test's negative control) scans
  // again until that launch's last CTA has cleared it (bounded, ~2 ms), so the control runs clean and shows as hint misses
#pragma unroll 1
  for (int retry = 0; retry < 2000; ++retry, __nanosleep(1000)) {
#endif
#pragma unroll 1
  for (unsigned pass = 0; pass < 2; ++pass) {
    const unsigned d = first_dom ^ pass;
    const unsigned lo = d ? unsigned(n_dom0) : 0u, hi = d ? g : unsigned(n_dom0);
#pragma unroll 1
    for (unsigned i = lo; i < hi; ++i) {
      const unsigned bit = 1u << (i & 31u);
      if (ld_relaxed_gpu(area + (i >> 5)) & bit) continue;
      if ((atom_or_relaxed_gpu(area + (i >> 5), bit) & bit) == 0u) return slot_of(i, n_dom0);
    }
  }
#if MUSE_VR_SLOT_TEST == 4
  }
#endif
  claim_fail(err);
  return 0;
}
__device__ __forceinline__ unsigned settle(unsigned* area, unsigned hint, unsigned old, int n_dom0, int n_dom1,
                                          int32_t* err) {
  hint = test_hint(hint);
  if (old != kInvalid) {
    const unsigned i = index_of(hint, n_dom0, n_dom1);
    if ((old & (1u << (i & 31u))) == 0u) return hint;
  }
  return settle_slow(area, hint, n_dom0, n_dom1, err);
}

// Once per CTA, after its claim (see the header): the last CTA of the launch checks and clears the bitmap.
__device__ __noinline__ void finish_last(unsigned* area, unsigned g, int32_t* err) {
#if MUSE_VR_SLOT_TEST == 4
#pragma unroll 1
  for (int i = 0; i < 200; ++i) __nanosleep(1000);
#endif
  bool exact = true;
#pragma unroll 1
  for (int k = 0; k < kBitmapWords; ++k) {
    const unsigned lo = unsigned(k) * 32u;
    const unsigned n = g > lo ? (g - lo >= 32u ? 32u : g - lo) : 0u;
    const unsigned full = n == 32u ? 0xFFFFFFFFu : ((1u << n) - 1u);
    if (ld_relaxed_gpu(area + k) != full) exact = false;
  }
  if (!exact) claim_fail(err);
#pragma unroll 1
  for (int k = 0; k < kBitmapWords; ++k) st_relaxed_gpu(area + k, 0u);
  red_add_relaxed_gpu(area + kVerifiedWord, 1u);
}
__device__ __forceinline__ void finish(unsigned* area, int32_t* err) {
  const unsigned g = gridDim.x;
  if (atom_inc_acq_rel_gpu(area + kCounterWord, g - 1u) == g - 1u) finish_last(area, g, err);
}

// ---- Launch-counter form (-DMUSE_VR_SLOT_EPOCH=1). One counter per SM of the slot table (SM id -> slot, a bijection from
// the device's SMs onto the G slots, built on the host), kEpochStride words apart. Before launch k of the kernel every
// counter holds k mod cycle (cycle = the start ticket's cycle in launches); a CTA learns k from its own start ticket.
// Claim: one relaxed atom.inc (wrapping at cycle) on the counter of the SM it runs on; the CTA whose increment read k
// owns that SM's slot (atomic operations on one location are totally ordered, so exactly one does). A CTA that read
// another value undoes its increment (atom.dec, same wrap) and takes the first counter (its own domain's SMs first)
// that still reads k, by atom.cas k -> k + 1. Every launch moves every counter by exactly one (G CTAs, G counters, one
// win each), so the next launch (ordered after this one: stream order) starts from k + 1 everywhere: no bitmap, no
// clear and no last-CTA check. The counters publish no data (the slot is used only by its owner), so no release or
// acquire is needed. A value no launch could leave (old - k outside [0, G]) or no counter left to take: sticky error
// and trap. The SM id only picks the counter tried first; ownership is the RMW result.
#if MUSE_VR_SLOT_EPOCH
constexpr int kEpochStride = 8;  // one counter per 32-byte sector
constexpr int kEpochWords = 256 * kEpochStride;
__device__ __forceinline__ unsigned atom_inc_relaxed_gpu(unsigned* p, unsigned bound) {
  unsigned old;
  asm volatile("atom.relaxed.gpu.global.inc.u32 %0, [%1], %2;" : "=r"(old) : "l"(p), "r"(bound) : "memory");
  return old;
}
__device__ __forceinline__ unsigned atom_dec_relaxed_gpu(unsigned* p, unsigned bound) {
  unsigned old;
  asm volatile("atom.relaxed.gpu.global.dec.u32 %0, [%1], %2;" : "=r"(old) : "l"(p), "r"(bound) : "memory");
  return old;
}
__device__ __forceinline__ unsigned atom_cas_relaxed_gpu(unsigned* p, unsigned cmp, unsigned v) {
  unsigned old;
  asm volatile("atom.relaxed.gpu.global.cas.b32 %0, [%1], %2, %3;" : "=r"(old) : "l"(p), "r"(cmp), "r"(v) : "memory");
  return old;
}
// tests (MUSE_VR_SLOT_TEST == 1): odd CTAs try the counter of the SM whose slot is rank 0 of their domain (collisions)
__device__ __forceinline__ unsigned epoch_test_smid(const uint8_t* table, unsigned smid) {
#if MUSE_VR_SLOT_TEST == 1
  if ((blockIdx.x & 1u) && smid < gridDim.x) {
    const unsigned want = unsigned(table[smid]) & ~127u;
#pragma unroll 1
    for (unsigned j = 0; j < gridDim.x; ++j)
      if (unsigned(table[j]) == want) return j;
  }
#endif
  return smid;
}
// Step 1 (one thread): the RMW on the counter of SM smid (the table has one entry per SM: grid = SM count).
__device__ __forceinline__ unsigned epoch_try_sm(unsigned* counters, unsigned smid, unsigned cycle) {
#if MUSE_VR_SLOT_TEST == 2
  if (blockIdx.x == 0) return kInvalid - 1u;  // negative control: CTA 0 takes its hint without claiming it
#endif
  if (smid >= gridDim.x) return kInvalid;
  return atom_inc_relaxed_gpu(counters + smid * kEpochStride, cycle - 1u);
}
__device__ __noinline__ unsigned epoch_settle_sm_slow(unsigned* counters, unsigned* miss, const uint8_t* table,
                                                      unsigned smid, unsigned old, unsigned k, unsigned cycle, int n_dom0,
                                                      int n_dom1, int32_t* err) {
  const unsigned g = unsigned(n_dom0 + n_dom1);
  if (old != kInvalid) {
    const unsigned d = (old + cycle - k) % cycle;
    if (d == 0u || d > g) claim_fail(err);
    atom_dec_relaxed_gpu(counters + smid * kEpochStride, cycle - 1u);
  }
  red_add_relaxed_gpu(miss, 1u);
  const unsigned first_dom = smid < gridDim.x ? (unsigned(table[smid]) >> 7) & 1u : 0u;
#pragma unroll 1
  for (unsigned pass = 0; pass < 2; ++pass) {
#pragma unroll 1
    for (unsigned j = 0; j < gridDim.x; ++j) {
      const unsigned e = table[j];
      if (index_of(e, n_dom0, n_dom1) == kInvalid || ((e >> 7) & 1u) != (first_dom ^ pass)) continue;
      unsigned* const c = counters + j * kEpochStride;
      if (ld_relaxed_gpu(c) != k) continue;
      if (atom_cas_relaxed_gpu(c, k, (k + 1u) % cycle) == k) return e;
    }
  }
  claim_fail(err);
  return 0;
}
// Step 2 (same thread): the owned slot. k = this launch's index (start ticket / G).
__device__ __forceinline__ unsigned epoch_settle_sm(unsigned* counters, unsigned* miss, const uint8_t* table,
                                                   unsigned smid, unsigned hint, unsigned old, unsigned k, unsigned cycle,
                                                   int n_dom0, int n_dom1, int32_t* err) {
#if MUSE_VR_SLOT_TEST == 2
  if (old == kInvalid - 1u) return hint;
#endif
  if (old == k && index_of(hint, n_dom0, n_dom1) != kInvalid) return hint;
  return epoch_settle_sm_slow(counters, miss, table, smid, old, k, cycle, n_dom0, n_dom1, err);
}
#endif  // MUSE_VR_SLOT_EPOCH

#if MUSE_VR_SLOT_TRACE
// Debug builds: rec[0] = %smid at the claim, rec[1] = hinted slot << 16 | claimed slot, rec[2] = %smid at exit.
__device__ __forceinline__ void trace_claim(unsigned long long* rec, unsigned smid, unsigned hint, unsigned slot) {
  if (rec == nullptr) return;
  rec[0] = smid;
  rec[1] = (static_cast<unsigned long long>(hint) << 16) | slot;
}
__device__ __forceinline__ void trace_exit(unsigned long long* rec) {
  if (rec == nullptr) return;
  rec[2] = read_smid();
}
#endif

}  // namespace muse_vr_slot
