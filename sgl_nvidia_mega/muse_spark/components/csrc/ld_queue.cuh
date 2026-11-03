// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// Per-locality-domain work queues for persistent kernels that stream weights placed by domain.
//
// A launch owns two queues of work units, one per locality domain: queue d lists the units whose weights are backed
// by memory on domain d (for example the row halves placed by the ld_experts component). Each CTA looks up the domain
// of the SM it runs on in a table built on the host (ld_layout.sm_domain_table), claims runs of units from its own
// domain's queue and, once that queue is empty, from the other queue. A claim is one atomic add on the queue's
// counter, so every unit is handed to exactly one CTA whatever the grid size, the order CTAs start in or the SMs they
// land on; nothing assumes the whole grid is resident (no CTA ever waits for another one here).
//
// Which CTA processes which unit depends on timing. A kernel whose result must not depend on it has to fix every
// reduction order independently of the claim pattern (fixed unit boundaries, fixed order of combining parts).
//
// Counter reuse across launches. One control block (Ctl) per call site (e.g. per layer and stream), zero-initialised
// once. Claim and completion counters exist in two halves; a launch uses half (launch & 1):
//   - every CTA reads `launch` in begin(), before its first claim;
//   - every CTA adds 1 to done[half] in end(), after its last claim (atom.acq_rel.gpu);
//   - the CTA that brings done[half] to gridDim.x has, through that read-modify-write chain, observed the end() of
//     every other CTA of the launch, so all of them have read `launch` and made their last claim. It clears the
//     other half (last used by the previous launch, next used by the following one) and then advances `launch`.
// Launches that share a control block must be ordered by completion (stream order or a graph edge without a
// programmatic launch between them); the next launch then observes the cleared half and the new `launch` value
// (CUDA Programming Guide: work in a stream is ordered after the completion of preceding work). Under programmatic
// dependent launch, give consecutive launches different control blocks or claim only after griddepcontrol.wait.
//
// Memory-model notes (PTX ISA, memory consistency model): claims are atomic operations on the same location, so the
// returned values are distinct prefixes of a single modification order; begin() reads `launch` with ld.relaxed.gpu;
// end() uses atom.add.acq_rel.gpu so the last CTA's clears and its write of `launch` are ordered after every other
// CTA's claims.

#pragma once

#include <cstdint>

namespace muse_vr_ldq {

struct alignas(128) Ctl {
  unsigned claim[2][2];  // [half][queue]
  unsigned done[2];      // [half] CTAs that called end()
  unsigned launch;       // completed launches
  unsigned reserved[25];
};
static_assert(sizeof(Ctl) == 128, "Ctl must stay one 128-byte block");

__device__ __forceinline__ unsigned ld_relaxed_gpu(const unsigned* p) {
  unsigned v;
  asm volatile("ld.relaxed.gpu.global.u32 %0, [%1];" : "=r"(v) : "l"(p) : "memory");
  return v;
}
__device__ __forceinline__ void st_relaxed_gpu(unsigned* p, unsigned v) {
  asm volatile("st.relaxed.gpu.global.u32 [%0], %1;" ::"l"(p), "r"(v) : "memory");
}
__device__ __forceinline__ unsigned atom_add_relaxed_gpu(unsigned* p, unsigned v) {
  unsigned old;
  asm volatile("atom.relaxed.gpu.global.add.u32 %0, [%1], %2;" : "=r"(old) : "l"(p), "r"(v) : "memory");
  return old;
}
__device__ __forceinline__ unsigned atom_add_acq_rel_gpu(unsigned* p, unsigned v) {
  unsigned old;
  asm volatile("atom.acq_rel.gpu.global.add.u32 %0, [%1], %2;" : "=r"(old) : "l"(p), "r"(v) : "memory");
  return old;
}

// Domain (0 or 1) of the SM the calling thread runs on.
__device__ __forceinline__ int sm_domain(const int8_t* table) {
  unsigned smid;
  asm volatile("mov.u32 %0, %%smid;" : "=r"(smid));
  return table[smid] & 1;
}

struct Queues {
  Ctl* ctl;
  unsigned half;
  unsigned n[2];  // units in queue 0 / 1 for this launch
  int home;       // the calling CTA's domain
  int cur;        // queue claims currently go to
  bool switched;  // already moved to the other queue
};

// One thread per CTA, before its first claim.
__device__ __forceinline__ Queues begin(Ctl* ctl, unsigned n0, unsigned n1, int home) {
  Queues q;
  q.ctl = ctl;
  q.half = ld_relaxed_gpu(&ctl->launch) & 1u;
  q.n[0] = n0;
  q.n[1] = n1;
  q.home = home & 1;
  q.cur = q.home;
  q.switched = false;
  return q;
}

struct Claim {
  unsigned begin, end;  // unit range [begin, end) of queue `queue`
  int queue;
};

// Claims up to `want` (>= 1) units: from the home queue while it has units, then from the other queue.
// Returns false when both queues are exhausted.
__device__ __forceinline__ bool claim(Queues& q, unsigned want, Claim& c) {
  while (true) {
    if (q.n[q.cur]) {
      const unsigned b = atom_add_relaxed_gpu(&q.ctl->claim[q.half][q.cur], want);
      if (b < q.n[q.cur]) {
        c.begin = b;
        c.end = min(q.n[q.cur], b + want);
        c.queue = q.cur;
        return true;
      }
    }
    if (q.switched) return false;
    q.switched = true;
    q.cur ^= 1;
  }
}

// Per-domain static schedules. Instead of claiming runs of units, a CTA can take one schedule slot: a rank inside one
// domain's group of n_d slots (n0 + n1 == number of CTAs). It asks its home domain first and takes a slot of the other
// domain only when all of its home domain's slots are taken, so with n0 + n1 CTAs every slot is taken exactly once
// whatever the landing pattern. A kernel then deals the units of domain `domain` over (rank, count) with any fixed
// rule; the result does not depend on which CTA got which slot. Uses the claim counters of the launch's half (begin()
// with n0, n1 = the slot counts) and the same end().
struct Slot {
  int domain;      // domain whose units this CTA works on
  unsigned rank;   // rank among that domain's slots
  unsigned count;  // slots of that domain
  unsigned first;  // global index of the domain's first slot (n0 for domain 1)
  bool valid;      // false only when more CTAs than n0 + n1 slots were launched
};
__device__ __forceinline__ Slot take_slot(Queues& q) {
  Slot s;
  s.domain = q.home;
  s.rank = atom_add_relaxed_gpu(&q.ctl->claim[q.half][s.domain], 1u);
  if (s.rank >= q.n[s.domain]) {
    s.domain ^= 1;
    s.rank = atom_add_relaxed_gpu(&q.ctl->claim[q.half][s.domain], 1u);
  }
  s.count = q.n[s.domain];
  s.first = s.domain ? q.n[0] : 0u;
  s.valid = s.rank < s.count;
  return s;
}

// One thread per CTA, after its last claim (every CTA of the grid must call it exactly once).
__device__ __forceinline__ void end(const Queues& q) {
  Ctl* ctl = q.ctl;
  const unsigned prev = atom_add_acq_rel_gpu(&ctl->done[q.half], 1u);
  if (prev + 1 == gridDim.x * gridDim.y * gridDim.z) {
    const unsigned other = q.half ^ 1u;
    st_relaxed_gpu(&ctl->claim[other][0], 0u);
    st_relaxed_gpu(&ctl->claim[other][1], 0u);
    st_relaxed_gpu(&ctl->done[other], 0u);
    const unsigned l = ld_relaxed_gpu(&ctl->launch);
    st_relaxed_gpu(&ctl->launch, l + 1u);
  }
}

}  // namespace muse_vr_ldq
