// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// Cross-rank hop device API, version 2 (header only). Arena and host helpers: ../ops/lb_hop.py; test kernels:
// lb_hop_ops.cuh.
//
// A hop moves one block of rows from every rank of a tensor-parallel group to every rank (itself included). Two uses:
// an all-reduce (every consumer sums every source's block in source-rank order) and an all-gather (every consumer reads
// every source's block as it is). The world size W is a template parameter (1, 2, 4, 8 or 16).
//
// Words (format kTag64). Three 16-bit elements travel in one aligned 64-bit word whose top 16 bits hold the transfer
// tag (bits 63..48; elements in bits 15..0, 31..16, 47..32). Producers store pairs of words with st.relaxed.sys.v2.b64,
// consumers read them with ld.relaxed.sys.v2.b64 and accept a word once its tag equals the tag of the transfer being
// read. Every 64-bit element carries its own tag, so a vector access is two independent single-copy-atomic 64-bit
// accesses (PTX ISA, memory consistency model: vector accesses are modelled as separate scalar accesses; aligned
// accesses up to 64 bits are single-copy atomic). A tagged word is never read with TMA, .nc loads or another width,
// and it has exactly one producer thread. Tag 0 never names a transfer: it marks an empty word.
// Format kS32: one 16-bit element per aligned 32-bit word, tag in bits 31..16 (two-shot reduce-scatter / gather).
//
// Sites and slots. A site is one hop call (layer, hop, call index). Policy kUnique gives the site its own receive region
// (one copy, or two copies selected by the forward parity when parity is set): no slot is reused within a forward, so
// no credit protocol runs per transfer. Policy kRing shares a three-slot ring per hop between call sites and orders
// every slot reuse with credits (see "Credits").
//
// Sequence and tags. seq64 is a 64-bit per-forward sequence word in device memory, raised once per model forward by a
// normally launched operation before the forward's first kernel (never a kernel argument: graph replays repeat
// arguments). kUnique: tag = seq64 mod 65535 + 1. kRing: t = ring_base + index (ring_base: a gap-free per-hop transfer
// counter in device memory, raised after every step by the number of transfers the step made; index: the call site's
// position among them), slot = t mod 3, tag = t mod 65535 + 1.
//
// Stale words. kUnique: a site's words are rewritten by every forward that uses the site, for the rows that forward
// reads; rows above a smaller row count keep older tags. Tags repeat only after 65535 forwards, and the host zeroes the
// whole arena (followed by a cross-rank barrier) before that can happen ("scrub on wrap", ../ops/lb_hop.py). kRing: as
// in the first version, the consumer rewrites rows [rows, high-water) of the slot it read with tag 0 after its
// dependency wait, so a slot never holds a word of an older transfer with a later transfer's tag.
//
// Missed advance. If seq64 were not raised before a replay, producers would write the same tags as the forward before
// and a consumer could not tell new words from old ones. Every consumer unit (a CTA index, or a role slot in a merged
// kernel) keeps last_seq, the seq64 value of its last consume; at entry, seq64 == last_seq means the advance was missed:
// the unit consumes nothing, poisons its outputs and sets the sticky error word (guard_enter / guard_exit).
//
// Credits (kRing). Before a producer overwrites slot s for transfer t it acquires (ld.acquire.sys) credit >= t - 2 from
// every consumer; a consumer releases credit t + 1 from a LATER kernel after that kernel's dependency wait (one
// fence.acq_rel.sys, then relaxed stores: a release pattern), i.e. after the consuming kernel completed.
//
// Cross-forward order (kUnique). A consumer's reads of forward f and a producer's writes of forward f + 1 into the same
// site are ordered by one step token per forward and rank pair: step_release (fence.acq_rel.sys, then a relaxed store
// of seq64 into every rank's token word) runs on every rank in the forward's first hop kernel, before any hop word of
// the forward is read on that rank and after the previous forward completed (full stream dependency); step_acquire
// (poll every rank's token for this forward, then fence.acq_rel.sys, then a CTA barrier) runs in every CTA that stores
// hop words after that point of the forward. Sites whose words are stored before the acquire point of a forward use two
// copies selected by the forward parity, so that their previous writer is ordered by the previous forward's tokens.
// Preconditions: (1) the forward head (the seq64 raise) is separated from the forward's first kernel by a full, non-
// programmatic stream dependency (CUDA Programming Guide, stream ordering and programmatic dependent launch), so every
// read of an earlier forward precedes step_release; (2) every kernel between the acquire host and a later kernel that
// stores hop words executes griddepcontrol.wait before it completes (or is launched without programmatic
// serialization), so the later stores are ordered after the acquire through kernel completion.
//
// Waits. Every wait counts polling ROUNDS (all loads of a round are issued before any of them is checked); after
// Ctl::spin_rounds rounds it backs off with nanosleep and checks the sticky error word every 16 rounds; after
// Ctl::max_rounds rounds it sets the sticky error word (0x80000000 | code << 16 | rank << 8 | site). An expired wait
// never consumes: the caller poisons its outputs. Kernels trap only when Ctl::flags has kTrap (tests).
// Collect loops are warp-uniform: every lane of the warp calls them and every lane leaves at the same round.
// Loading. A kernel that spins on words of other ranks must be loaded on every rank before its first spinning launch
// (lazy module loading can otherwise block that launch behind a peer's spinning kernel): the warm-up / capture order
// runs every such kernel once before steady state, and harnesses that launch them directly preload them first.
//
// Writers. Rows are padded to a pitch that is a whole number of 128-byte lines, and every 128-byte line of a receive
// region is written by one producer CTA per transfer (push_row_* write whole rows; push_tile_* require tile columns
// on line boundaries and report a violation through the error word).
#pragma once

#include <cstdint>

#define LB_HOP_VERSION 2

namespace muse_vr {
namespace lb_hop {

inline constexpr int kMaxWorld = 16;
inline constexpr uint32_t kRingSlots = 3;
inline constexpr uint32_t kLineWords = 16;    // 64-bit words per 128-byte line
inline constexpr uint32_t kLineElems64 = 48;  // kTag64 elements per line
inline constexpr uint32_t kLineElems32 = 32;  // kS32 elements per line

// Ctl::flags
inline constexpr uint32_t kTrap = 1u;               // trap on an expired wait (tests only)
inline constexpr uint32_t kSkipCreditUnsafe = 2u;   // kRing: producers skip the credit acquire; slot reuse is then
                                                    // ordered only by stream order and data dependence, which the PTX
                                                    // memory model does not guarantee
inline constexpr uint32_t kAsyncCredit = 4u;        // kRing: every consumer CTA releases its credit from the consuming
                                                    // kernel itself (red.async.release.sys add after its last read) into
                                                    // ITS OWN credit word; producers wait until every consumer CTA's
                                                    // word shows the earlier transfers (credit_release_async)
// site policies and word formats
inline constexpr uint32_t kUnique = 0u;
inline constexpr uint32_t kRing = 1u;
inline constexpr uint32_t kTag64 = 0u;
inline constexpr uint32_t kS32 = 1u;
// policy word bit (host side, development twin): every push of the site stores this rank's words into its OWN copy at
// every source position instead of into every rank's copy, so consumers complete without any cross-GPU traffic (same
// kernels, same word counts); measures a hop's exposed time against a no-flight twin. Never used for results.
inline constexpr uint32_t kLoopbackDev = 0x100u;

// error codes (bits 30..16 of the error word)
inline constexpr uint32_t kErrPayload = 1u;        // a payload wait ran out of rounds
inline constexpr uint32_t kErrCredit = 2u;         // a credit wait ran out of rounds
inline constexpr uint32_t kErrArgs = 3u;           // bad arguments (e.g. a tile not on line boundaries)
inline constexpr uint32_t kErrMissedAdvance = 4u;  // seq64 not raised since this unit's last consume
inline constexpr uint32_t kErrStepToken = 5u;      // a step-token wait ran out of rounds
inline constexpr uint32_t kErrOverrun = 6u;        // a polled word carries a NEWER tag (1..3 transfers ahead): the slot
                                                   // was overwritten before this read (write-after-read violation)

inline constexpr uint16_t kPoisonBf16 = 0x7fc0u;  // quiet NaN

struct Ctl {
  uint32_t* err_dev;   // sticky error word (device memory)
  uint32_t* err_host;  // host-visible mirror (mapped pinned memory), may be null
  uint32_t max_rounds;
  uint32_t spin_rounds;
  uint32_t flags;
  uint32_t backoff_ns;
};

template <int W>
struct Site {
  static_assert(W == 1 || W == 2 || W == 4 || W == 8 || W == 16, "world size 1, 2, 4, 8 or 16");
  unsigned long long* recv[W];      // copy 0 of this site's receive region on every rank (mapped in this process)
  uint32_t* credit[W];              // kRing: credit words of every rank: sync form entry c = consumer rank c; async
                                    // form entry c * units + k = consumer CTA k of rank c
  const unsigned long long* seq64;  // forward sequence word (this rank)
  unsigned long long* ring_base;    // kRing: transfer counter of the hop (this rank)
  uint32_t* hwm;                    // kRing: row high-water mark per slot, two alternating copies (this rank) [2 x kRingSlots]
  unsigned long long* last_seq;     // missed-advance guard: one word per consumer unit (this rank) [units]
  unsigned long long copy_words;    // 64-bit words per copy (= W * rows_max * pitch)
  uint32_t rank;
  uint32_t rows_max;
  uint32_t width;   // elements per row (kTag64: multiple of 6; kS32: multiple of 2)
  uint32_t pitch;   // 64-bit words per row, multiple of kLineWords
  uint32_t policy;  // kUnique or kRing
  uint32_t format;  // kTag64 or kS32
  uint32_t parity;  // kUnique: 1 = two copies selected by seq64 parity
  uint32_t index;   // kRing: call-site position among the step's transfers of this hop
  uint32_t site;    // site number reported in the error word (low 8 bits)
  uint32_t units;   // number of last_seq words (consumer units)
  uint32_t loop;    // 1: development loopback twin (kLoopbackDev)
  uint32_t* audit;  // diagnostic builds (LB_HOP_AUDIT) only, else null: writer CTA + 1 per (row, line) of this rank's
                    // block [rows_max][pitch / kLineWords], then one counter of multi-writer events
  unsigned long long* mc;  // optional multicast address of copy 0 of this site's receive region (same layout as
                           // recv[r]); non-null: kTag64 pushes store each word once to every rank (multimem), else
                           // one store per rank
};

// One transfer as every thread of a kernel sees it (computed from device words, never from per-forward arguments).
struct Xfer {
  unsigned long long seq;  // seq64 of this forward
  unsigned long long t;    // kUnique: seq; kRing: ring transfer number
  uint32_t tag;            // 1..65535
  uint32_t copy;           // kUnique: copy index (0 / 1); kRing: slot
};

// ------------------------------------------------------------------------------------------------------- PTX helpers
__device__ __forceinline__ void st_relaxed_sys_v2_b64(unsigned long long* p, unsigned long long a, unsigned long long b) {
  asm volatile("st.relaxed.sys.global.v2.b64 [%0], {%1, %2};" ::"l"(p), "l"(a), "l"(b) : "memory");
}
// One relaxed system-scope 64-bit store to every rank's copy behind a multicast address (PTX ISA multimem.st): each
// rank's location receives the whole 8-byte word.
__device__ __forceinline__ void st_mc_relaxed_sys_b64(unsigned long long* p, unsigned long long a) {
  asm volatile("multimem.st.relaxed.sys.global.b64 [%0], %1;" ::"l"(p), "l"(a) : "memory");
}
__device__ __forceinline__ ulonglong2 ld_relaxed_sys_v2_b64(const unsigned long long* p) {
  ulonglong2 v;
  asm volatile("ld.relaxed.sys.global.v2.b64 {%0, %1}, [%2];" : "=l"(v.x), "=l"(v.y) : "l"(p) : "memory");
  return v;
}
__device__ __forceinline__ void st_relaxed_sys_u64(unsigned long long* p, unsigned long long v) {
  asm volatile("st.relaxed.sys.global.u64 [%0], %1;" ::"l"(p), "l"(v) : "memory");
}
__device__ __forceinline__ unsigned long long ld_relaxed_sys_u64(const unsigned long long* p) {
  unsigned long long v;
  asm volatile("ld.relaxed.sys.global.u64 %0, [%1];" : "=l"(v) : "l"(p) : "memory");
  return v;
}
__device__ __forceinline__ unsigned long long ld_relaxed_gpu_u64(const unsigned long long* p) {
  unsigned long long v;
  asm volatile("ld.relaxed.gpu.global.u64 %0, [%1];" : "=l"(v) : "l"(p) : "memory");
  return v;
}
__device__ __forceinline__ void st_relaxed_sys_v4_b32(uint32_t* p, uint4 v) {
  asm volatile("st.relaxed.sys.global.v4.b32 [%0], {%1, %2, %3, %4};" ::"l"(p), "r"(v.x), "r"(v.y), "r"(v.z), "r"(v.w)
               : "memory");
}
__device__ __forceinline__ uint4 ld_relaxed_sys_v4_b32(const uint32_t* p) {
  uint4 v;
  asm volatile("ld.relaxed.sys.global.v4.b32 {%0, %1, %2, %3}, [%4];"
               : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w)
               : "l"(p)
               : "memory");
  return v;
}
__device__ __forceinline__ uint32_t ld_acquire_sys_u32(const uint32_t* p) {
  uint32_t v;
  asm volatile("ld.acquire.sys.global.u32 %0, [%1];" : "=r"(v) : "l"(p) : "memory");
  return v;
}
__device__ __forceinline__ void st_relaxed_sys_u32(uint32_t* p, uint32_t v) {
  asm volatile("st.relaxed.sys.global.u32 [%0], %1;" ::"l"(p), "r"(v) : "memory");
}
__device__ __forceinline__ void fence_acq_rel_sys() { asm volatile("fence.acq_rel.sys;" ::: "memory"); }
__device__ __forceinline__ unsigned long long ld_acquire_sys_u64(const unsigned long long* p) {
  unsigned long long v;
  asm volatile("ld.acquire.sys.global.u64 %0, [%1];" : "=l"(v) : "l"(p) : "memory");
  return v;
}

__device__ __forceinline__ float bf16_bits_to_float(uint32_t b) { return __uint_as_float((b & 0xffffu) << 16); }

// ------------------------------------------------------------------------------------------------------- errors
__device__ __forceinline__ bool failed(const Ctl& c) {
  return *reinterpret_cast<const volatile uint32_t*>(c.err_dev) != 0u;
}

__device__ __noinline__ void fail(const Ctl& c, uint32_t code, uint32_t rank, uint32_t site) {
  const uint32_t v = 0x80000000u | ((code & 0x7fffu) << 16) | ((rank & 0xffu) << 8) | (site & 0xffu);
  atomicOr(c.err_dev, v);
  if (c.err_host) *reinterpret_cast<volatile uint32_t*>(c.err_host) = v;
  __threadfence_system();
  if (c.flags & kTrap) __trap();
}

// ------------------------------------------------------------------------------------------------------- transfer
__device__ __forceinline__ uint32_t tag_of(unsigned long long t) { return uint32_t(t % 65535ull) + 1u; }

template <int W>
__device__ __forceinline__ Xfer xfer(const Site<W>& s) {
  Xfer x;
  x.seq = ld_relaxed_gpu_u64(s.seq64);
  if (s.policy == kRing) {
    x.t = ld_relaxed_gpu_u64(s.ring_base) + s.index;
    x.copy = uint32_t(x.t % kRingSlots);
  } else {
    x.t = x.seq;
    x.copy = s.parity ? uint32_t(x.seq & 1ull) : 0u;
  }
  x.tag = tag_of(x.t);
  return x;
}

// 64-bit word w of row `row` of source `src`'s block, in rank `dst`'s copy `copy` of the site.
template <int W>
__device__ __forceinline__ unsigned long long* word_ptr(const Site<W>& s, uint32_t dst, uint32_t copy, uint32_t src,
                                                        uint32_t row, uint32_t w) {
  return s.recv[dst] + copy * s.copy_words + (uint64_t(src) * s.rows_max + row) * s.pitch + w;
}

// ------------------------------------------------------------------------------------------------------- guard
// One thread per consumer unit, at the unit's entry (it may run before griddepcontrol.wait): true when the unit may
// consume this forward's words. On false the error word is set; the unit consumes nothing and poisons its outputs.
template <int W>
__device__ __forceinline__ bool guard_enter(const Site<W>& s, const Ctl& c, uint32_t unit, unsigned long long seq) {
  if (unit >= s.units) {
    fail(c, kErrArgs, s.rank, s.site);
    return false;
  }
  const unsigned long long last = *reinterpret_cast<const volatile unsigned long long*>(s.last_seq + unit);
  if (last == seq) {
    fail(c, kErrMissedAdvance, s.rank, s.site);
    return false;
  }
  return true;
}
// The same thread, after the CTA barrier that ends the unit's consume phase.
template <int W>
__device__ __forceinline__ void guard_exit(const Site<W>& s, uint32_t unit, unsigned long long seq) {
  if (unit < s.units) *reinterpret_cast<volatile unsigned long long*>(s.last_seq + unit) = seq;
}

// ------------------------------------------------------------------------------------------------------- producer
#ifdef LB_HOP_AUDIT
// Writer audit (diagnostic builds): records the pushing CTA of the 128-byte line holding (row, col); a second CTA on the
// same line counts a multi-writer event (the rule: one writer CTA per line).
template <int W>
__device__ __forceinline__ void audit_push(const Site<W>& s, uint32_t row, uint32_t col) {
  if (s.audit == nullptr) return;
  const uint32_t lines = s.pitch / kLineWords, line = (col / 3u) / kLineWords;
  const uint32_t me = blockIdx.x + 1u;
  const uint32_t old = atomicCAS(s.audit + uint64_t(row) * lines + line, 0u, me);
  if (old != 0u && old != me) atomicAdd(s.audit + uint64_t(s.rows_max) * lines, 1u);
}
#endif

// kTag64: one thread stores elements col..col+5 (col % 6 == 0; 16-bit values in e[0..5]) of row `row` of this rank's
// block into every rank's copy, as one 2-word vector per rank.
template <int W>
__device__ __forceinline__ void push6(const Site<W>& s, const Xfer& x, uint32_t row, uint32_t col, const uint32_t (&e)[6]) {
  const unsigned long long tag = static_cast<unsigned long long>(x.tag) << 48;
  const unsigned long long w0 = tag | (static_cast<unsigned long long>(e[2] & 0xffffu) << 32) |
                                (static_cast<unsigned long long>(e[1] & 0xffffu) << 16) | (e[0] & 0xffffu);
  const unsigned long long w1 = tag | (static_cast<unsigned long long>(e[5] & 0xffffu) << 32) |
                                (static_cast<unsigned long long>(e[4] & 0xffffu) << 16) | (e[3] & 0xffffu);
  if (s.mc != nullptr && !s.loop) {
    unsigned long long* p = s.mc + x.copy * s.copy_words + (uint64_t(s.rank) * s.rows_max + row) * s.pitch + col / 3u;
    st_mc_relaxed_sys_b64(p, w0);
    st_mc_relaxed_sys_b64(p + 1, w1);
  } else {
#pragma unroll
    for (int r = 0; r < W; ++r) {
      const uint32_t dst = s.loop ? s.rank : uint32_t(r), src = s.loop ? uint32_t(r) : s.rank;
      st_relaxed_sys_v2_b64(word_ptr(s, dst, x.copy, src, row, col / 3u), w0, w1);
    }
  }
#ifdef LB_HOP_AUDIT
  audit_push(s, row, col);
#endif
}

// kTag64, one destination per call: as push6, but only into the copy of rank `r` (0 <= r < W), so a CTA can spread the
// items x destinations of a line over many threads and warps. With multicast pushes on, the call for r == 0 stores to
// every copy and the other calls do nothing. Writer audit: recorded by the r == 0 call.
template <int W>
__device__ __forceinline__ void push6_to(const Site<W>& s, const Xfer& x, uint32_t r, uint32_t row, uint32_t col,
                                         const uint32_t (&e)[6]) {
  const unsigned long long tag = static_cast<unsigned long long>(x.tag) << 48;
  const unsigned long long w0 = tag | (static_cast<unsigned long long>(e[2] & 0xffffu) << 32) |
                                (static_cast<unsigned long long>(e[1] & 0xffffu) << 16) | (e[0] & 0xffffu);
  const unsigned long long w1 = tag | (static_cast<unsigned long long>(e[5] & 0xffffu) << 32) |
                                (static_cast<unsigned long long>(e[4] & 0xffffu) << 16) | (e[3] & 0xffffu);
  if (s.mc != nullptr && !s.loop) {
    if (r == 0u) {
      unsigned long long* p = s.mc + x.copy * s.copy_words + (uint64_t(s.rank) * s.rows_max + row) * s.pitch + col / 3u;
      st_mc_relaxed_sys_b64(p, w0);
      st_mc_relaxed_sys_b64(p + 1, w1);
    }
  } else {
    const uint32_t dst = s.loop ? s.rank : r, src = s.loop ? r : s.rank;
    st_relaxed_sys_v2_b64(word_ptr(s, dst, x.copy, src, row, col / 3u), w0, w1);
  }
#ifdef LB_HOP_AUDIT
  if (r == 0u) audit_push(s, row, col);
#endif
}

// kTag64 building blocks for producers that stage their destinations once (e.g. a table in shared memory) and then
// address each word themselves:
//   push_dests(s)    number of destination bases: W, or 1 with multicast pushes on (one base reaches every copy);
//   push_base(s,x,d) base d = row 0, word 0 of this rank's block in copy x.copy of destination d (multicast: d ignored);
//   push6_at / push6_mc_at(p, x, e)  the item's two tagged words at word pointer p = base + row * pitch + col / 3
//   (col % 6 == 0); the second form for a multicast base. Writer audit: the caller records each item once.
template <int W>
__device__ __forceinline__ bool push_mc(const Site<W>& s) { return s.mc != nullptr && !s.loop; }
template <int W>
__device__ __forceinline__ uint32_t push_dests(const Site<W>& s) { return push_mc(s) ? 1u : uint32_t(W); }
template <int W>
__device__ __forceinline__ unsigned long long* push_base(const Site<W>& s, const Xfer& x, uint32_t d) {
  if (push_mc(s)) return s.mc + x.copy * s.copy_words + uint64_t(s.rank) * s.rows_max * s.pitch;
  const uint32_t dst = s.loop ? s.rank : d, src = s.loop ? d : s.rank;
  return word_ptr(s, dst, x.copy, src, 0u, 0u);
}
__device__ __forceinline__ void words6(const Xfer& x, const uint32_t (&e)[6], unsigned long long& w0,
                                       unsigned long long& w1) {
  const unsigned long long tag = static_cast<unsigned long long>(x.tag) << 48;
  w0 = tag | (static_cast<unsigned long long>(e[2] & 0xffffu) << 32) | (static_cast<unsigned long long>(e[1] & 0xffffu) << 16) |
       (e[0] & 0xffffu);
  w1 = tag | (static_cast<unsigned long long>(e[5] & 0xffffu) << 32) | (static_cast<unsigned long long>(e[4] & 0xffffu) << 16) |
       (e[3] & 0xffffu);
}
__device__ __forceinline__ void push6_at(unsigned long long* p, const Xfer& x, const uint32_t (&e)[6]) {
  unsigned long long w0, w1;
  words6(x, e, w0, w1);
  st_relaxed_sys_v2_b64(p, w0, w1);
}
__device__ __forceinline__ void push6_mc_at(unsigned long long* p, const Xfer& x, const uint32_t (&e)[6]) {
  unsigned long long w0, w1;
  words6(x, e, w0, w1);
  st_mc_relaxed_sys_b64(p, w0);
  st_mc_relaxed_sys_b64(p + 1, w1);
}

// kTag64: the threads (tid, nthreads) of one CTA push whole rows [r0, r0 + nr) of this rank's block from a 16-bit row
// buffer (row stride ld elements, columns 0..width-1; columns past the end of the buffer row are not read: elements at
// or above `valid` are sent as 0). One CTA writing whole rows keeps one writer per 128-byte line.
template <int W>
__device__ __forceinline__ void push_rows_u16(const Site<W>& s, const Xfer& x, uint32_t r0, uint32_t nr,
                                              const uint16_t* buf, uint32_t ld, uint32_t valid, uint32_t tid,
                                              uint32_t nthreads) {
  const uint32_t groups = s.width / 6u;
  const uint32_t n = nr * groups;
  for (uint32_t i = tid; i < n; i += nthreads) {
    const uint32_t rr = i / groups, col = (i % groups) * 6u;
    const uint16_t* src = buf + uint64_t(rr) * ld;
    uint32_t e[6];
#pragma unroll
    for (int k = 0; k < 6; ++k) e[k] = (col + k < valid) ? uint32_t(src[col + k]) : 0u;
    push6(s, x, r0 + rr, col, e);
  }
}

// kTag64: tile [r0, r0 + nr) x [c0, c0 + nc) from a 16-bit tile buffer (row stride ld). Tile columns must start on a
// line boundary and end on one or at the row end, so that every line has one writer CTA; otherwise the error word is
// set (kErrArgs) and nothing is pushed. Returns false on a violation.
template <int W>
__device__ __forceinline__ bool push_tile_u16(const Site<W>& s, const Ctl& c, const Xfer& x, uint32_t r0, uint32_t nr,
                                              uint32_t c0, uint32_t nc, const uint16_t* tile, uint32_t ld, uint32_t tid,
                                              uint32_t nthreads) {
  const bool ok = (c0 % kLineElems64) == 0u && (nc % 6u) == 0u &&
                  (((c0 + nc) % kLineElems64) == 0u || c0 + nc == s.width) && c0 + nc <= s.width && r0 + nr <= s.rows_max;
  if (!ok) {
    if (tid == 0) fail(c, kErrArgs, s.rank, s.site);
    return false;
  }
  const uint32_t groups = nc / 6u;
  for (uint32_t i = tid; i < nr * groups; i += nthreads) {
    const uint32_t rr = i / groups, cc = (i % groups) * 6u;
    const uint16_t* src = tile + uint64_t(rr) * ld + cc;
    const uint32_t e[6] = {src[0], src[1], src[2], src[3], src[4], src[5]};
    push6(s, x, r0 + rr, c0 + cc, e);
  }
  return true;
}

// ------------------------------------------------------------------------------------------------------- consumer
// True when tag t (1..65535; 0 = empty) is 1..3 transfers AHEAD of the awaited tag w on the 65535-tag cycle: the next
// forward of a unique site or the next use of a ring slot, i.e. the slot was overwritten before this consumer read it.
__device__ __forceinline__ bool tag_ahead(uint32_t t, uint32_t w) {
  if (t == 0u) return false;
  const uint32_t a = t - 1u, b = w - 1u;
  const uint32_t d = a >= b ? a - b : a + 65535u - b;
  return d >= 1u && d <= kRingSlots;
}

// Polls the 2-word vectors a[i][r] (item i, source r; items i >= n and sources r >= W are not polled) until both words
// of every vector carry `tag`. Warp-uniform: every lane of the warp calls it with its own items; all lanes leave at
// the same round and return the same value (false: the bound ran out here or the error word was set elsewhere).
template <int W, int N>
__device__ __forceinline__ bool poll_v2(const Ctl& c, const unsigned long long* const (&a)[N][W], ulonglong2 (&v)[N][W],
                                        int n, uint32_t tag, uint32_t rank, uint32_t site, uint32_t code = kErrPayload) {
  static_assert(N * W <= 32, "at most 32 vectors per lane and round");
  const unsigned long long want = tag;
  uint32_t pend = 0;
#pragma unroll
  for (int i = 0; i < N; ++i)
    if (i < n) pend |= ((W == 32 ? ~0u : ((1u << W) - 1u)) << (i * W));
  uint32_t round = 0;
  while (__any_sync(0xffffffffu, pend != 0u)) {
#pragma unroll
    for (int i = 0; i < N; ++i)
#pragma unroll
      for (int r = 0; r < W; ++r)
        if (pend >> (i * W + r) & 1u) v[i][r] = ld_relaxed_sys_v2_b64(a[i][r]);
    bool over = false;
#pragma unroll
    for (int i = 0; i < N; ++i)
#pragma unroll
      for (int r = 0; r < W; ++r)
        if (pend >> (i * W + r) & 1u) {
          const uint32_t t0 = uint32_t(v[i][r].x >> 48), t1 = uint32_t(v[i][r].y >> 48);
          if (t0 == uint32_t(want) && t1 == uint32_t(want)) pend &= ~(1u << (i * W + r));
          else over = over || tag_ahead(t0, uint32_t(want)) || tag_ahead(t1, uint32_t(want));
        }
    if (__any_sync(0xffffffffu, over)) {  // fast fail: waiting longer cannot help
      if (over) fail(c, kErrOverrun, rank, site);
      __syncwarp();
      return false;
    }
    ++round;
    if (round > c.spin_rounds) {
      if ((round & 15u) == 0u && __any_sync(0xffffffffu, failed(c))) return false;
      if (round > c.max_rounds) {
        if (pend) fail(c, code, rank, site);
        __syncwarp();
        return false;
      }
      __nanosleep(c.backoff_ns);
    }
  }
  return true;
}

// Element e (0..5) of a 2-word vector.
__device__ __forceinline__ uint32_t elem6(const ulonglong2& w, int e) {
  const unsigned long long q = e < 3 ? w.x : w.y;
  return uint32_t(q >> (16 * (e % 3))) & 0xffffu;
}

// kTag64 gather / reduce of n <= N items per lane: item i = elements col[i]..col[i]+5 of row row[i] (col % 6 == 0)
// from every source rank's block. raw[i][r] receives source r's two words; sum[i][e] (when sum != nullptr) the fp32 sum
// of the six elements over sources 0..W-1 in rank order (first source's value, then one rounded fp32 add per source).
template <int W, int N>
__device__ __forceinline__ bool collect6(const Site<W>& s, const Ctl& c, const Xfer& x, const uint32_t (&row)[N],
                                         const uint32_t (&col)[N], int n, ulonglong2 (&raw)[N][W], float (*sum)[6]) {
  const unsigned long long* a[N][W];
#pragma unroll
  for (int i = 0; i < N; ++i)
#pragma unroll
    for (int r = 0; r < W; ++r)
      a[i][r] = word_ptr(s, s.rank, x.copy, uint32_t(r), i < n ? row[i] : row[0], (i < n ? col[i] : col[0]) / 3u);
  const bool ok = poll_v2<W, N>(c, a, raw, n, x.tag, s.rank, s.site);
  if (sum != nullptr) {
#pragma unroll
    for (int i = 0; i < N; ++i) {
#pragma unroll
      for (int e = 0; e < 6; ++e) {
        float acc = bf16_bits_to_float(elem6(raw[i][0], e));
#pragma unroll
        for (int r = 1; r < W; ++r) acc = __fadd_rn(acc, bf16_bits_to_float(elem6(raw[i][r], e)));
        sum[i][e] = acc;
      }
    }
  }
  return ok;
}

// kTag64 carrying fp32: an item = 3 fp32 values as 6 halves {lo, hi} x 3 (value j = element 2 j | element 2 j + 1 << 16),
// i.e. 3 values per 2 tagged words (elements col..col+5 with col = 6 * item). push3f stores item `item` of row `row` into
// every rank's copy (as push6, multicast-capable); collect3f is collect6 for such items: n <= N items per lane from every
// source, sum[i][j] = fp32 sum of value j over sources 0..W-1 in rank order (first source's value, then one rounded add
// per source).
template <int W>
__device__ __forceinline__ void push3f(const Site<W>& s, const Xfer& x, uint32_t row, uint32_t item, const float (&v)[3]) {
  uint32_t e[6];
#pragma unroll
  for (int j = 0; j < 3; ++j) {
    const uint32_t b = __float_as_uint(v[j]);
    e[2 * j] = b & 0xffffu;
    e[2 * j + 1] = b >> 16;
  }
  push6(s, x, row, 6u * item, e);
}
__device__ __forceinline__ float elem3f(const ulonglong2& w, int j) {
  return __uint_as_float(elem6(w, 2 * j) | (elem6(w, 2 * j + 1) << 16));
}
template <int W, int N>
__device__ __forceinline__ bool collect3f(const Site<W>& s, const Ctl& c, const Xfer& x, const uint32_t (&row)[N],
                                          const uint32_t (&item)[N], int n, float (&sum)[N][3]) {
  uint32_t col[N];
#pragma unroll
  for (int i = 0; i < N; ++i) col[i] = 6u * item[i];
  ulonglong2 raw[N][W];
  const bool ok = collect6<W, N>(s, c, x, row, col, n, raw, nullptr);
#pragma unroll
  for (int i = 0; i < N; ++i)
#pragma unroll
    for (int j = 0; j < 3; ++j) {
      float acc = elem3f(raw[i][0], j);
#pragma unroll
      for (int r = 1; r < W; ++r) acc = __fadd_rn(acc, elem3f(raw[i][r], j));
      sum[i][j] = acc;
    }
  return ok;
}

// kRing high-water marks: transfer t writes entry (t mod 3) of copy (t / 3) mod 2 and reads what t - 3 wrote (the other
// copy), so a consumer kernel can record its mark while its own CTAs still read the previous one.
template <int W>
__device__ __forceinline__ uint32_t hwm_read(const Site<W>& s, const Xfer& x) {
  if (x.t < kRingSlots) return 0u;
  const uint64_t prev = x.t - kRingSlots;
  return *reinterpret_cast<const volatile uint32_t*>(s.hwm + (prev % kRingSlots) + kRingSlots * ((prev / kRingSlots) & 1u));
}
template <int W>
__device__ __forceinline__ void hwm_write(const Site<W>& s, const Xfer& x, uint32_t rows) {
  s.hwm[(x.t % kRingSlots) + kRingSlots * ((x.t / kRingSlots) & 1u)] = rows;
}

// kRing: rows [rows, hwm) of this rank's copy of the slot read for x back to tag 0, every source block. Call after the
// consuming kernel's griddepcontrol.wait, with hwm = hwm_read(s, x) read after that wait; split over (tid, nthreads).
template <int W>
__device__ __forceinline__ void ring_scrub(const Site<W>& s, const Xfer& x, uint32_t rows, uint32_t hwm, uint32_t tid,
                                           uint32_t nthreads) {
  if (hwm <= rows) return;
  const uint32_t pairs = s.pitch / 2u;
  const uint64_t per_src = uint64_t(hwm - rows) * pairs;
  for (uint64_t i = tid; i < uint64_t(W) * per_src; i += nthreads) {
    const uint32_t src = uint32_t(i / per_src);
    const uint64_t rem = i % per_src;
    st_relaxed_sys_v2_b64(word_ptr(s, s.rank, x.copy, src, rows + uint32_t(rem / pairs), uint32_t(rem % pairs) * 2u), 0ull,
                          0ull);
  }
}

// ------------------------------------------------------------------------------------------------------- credits (kRing)
// Lanes 0..W-1 of one warp: wait until every consumer released the transfer that used this slot before (t - 3).
// Warp-uniform (all 32 lanes call it). The caller follows it with a CTA barrier before any thread stores into the slot.
template <int W>
__device__ __forceinline__ bool credit_acquire(const Site<W>& s, const Ctl& c, const Xfer& x) {
  if ((c.flags & kSkipCreditUnsafe) || x.t < kRingSlots) return true;
  const uint32_t lane = threadIdx.x & 31u;
  // sync form: one word per consumer rank holding the released transfer index (need t - 2); async form: one word per
  // (consumer rank, consumer CTA), +1 per consumed transfer (need t - 2 in EVERY word: no sums, so a CTA with nothing to
  // read or running ahead never stands in for another). Words compared as signed differences (wrap mod 2^32).
  const uint32_t need = uint32_t(x.t - kRingSlots + 1);
  const uint32_t* cr = s.credit[s.rank];
  const bool per_cta = (c.flags & kAsyncCredit) != 0u;
  const uint32_t nwords = per_cta ? uint32_t(W) * s.units : uint32_t(W);
  bool mine = lane >= nwords;
  uint32_t round = 0;
  while (!__all_sync(0xffffffffu, mine)) {
    if (!mine) {
      mine = true;
      for (uint32_t i = lane; i < nwords; i += 32u) mine = mine && int32_t(ld_acquire_sys_u32(cr + i) - need) >= 0;
    }
    ++round;
    if (round > c.spin_rounds) {
      if ((round & 15u) == 0u && __any_sync(0xffffffffu, failed(c))) return false;
      if (round > c.max_rounds) {
        if (!mine) fail(c, kErrCredit, s.rank, s.site);
        __syncwarp();
        return false;
      }
      __nanosleep(c.backoff_ns);
    }
  }
  return true;
}

// One thread of a kernel that runs after the consuming kernel, after its griddepcontrol.wait: records the slot's row
// high-water mark and releases credit t + 1 to every producer (one system-scope release fence followed by relaxed
// stores: a release pattern per store).
template <int W>
__device__ __forceinline__ void credit_release(const Site<W>& s, const Xfer& x, uint32_t rows) {
  hwm_write(s, x, rows);
  fence_acq_rel_sys();
#pragma unroll
  for (int r = 0; r < W; ++r) st_relaxed_sys_u32(s.credit[r] + s.rank, uint32_t(x.t + 1));
}

// Async-credit form (Ctl kAsyncCredit): thread 0 of EVERY consumer CTA k of the transfer, after the
// CTA barrier that ends the CTA's reads of the slot (and its scrub stores): CTA 0 records the high-water mark, then one
// red.async.release.sys add of 1 to word (this rank, k) of the credit area on every producer rank (a strong release at
// system scope, generic proxy; single writer per word; the producer's ld.acquire synchronizes with the add). The
// consumer grid must have exactly s.units CTAs on every transfer of the site.
template <int W>
__device__ __forceinline__ void credit_release_async(const Site<W>& s, const Xfer& x, uint32_t rows) {
  if (blockIdx.x == 0) hwm_write(s, x, rows);
  const uint32_t word = s.rank * s.units + blockIdx.x;
#pragma unroll
  for (int r = 0; r < W; ++r)
    asm volatile("red.async.release.sys.global.add.u32 [%0], %1;" ::"l"(s.credit[r] + word), "r"(1u) : "memory");
}

// ------------------------------------------------------------------------------------------------------- step tokens
// Token words: on every rank one 64-bit word per source rank, each on its own 128-byte line (tok[r] + src * kLineWords).
template <int W>
struct StepTok {
  unsigned long long* tok[W];
  const unsigned long long* seq64;
  uint32_t rank;
  uint32_t site;
};

// One thread, in the forward's first hop kernel on this rank: every read of earlier forwards' hop words on this rank
// precedes it (the previous forward completed before this forward's first kernel started).
template <int W>
__device__ __forceinline__ void step_release(const StepTok<W>& t, unsigned long long seq) {
  fence_acq_rel_sys();
#pragma unroll
  for (int r = 0; r < W; ++r) st_relaxed_sys_u64(t.tok[r] + uint64_t(t.rank) * kLineWords, seq);
}

// Lanes 0..W-1 of one warp in every CTA that stores hop words after this point: lane r waits until rank r's token holds
// this forward's seq (or a later one), polling with system-scope ACQUIRE loads, so the load that observes it
// synchronizes with rank r's release pattern (step_release: fence.acq_rel.sys, then the relaxed token store); no
// separate fence (the poll-then-fence form measured slower). Warp-uniform; the
// caller follows it with a CTA barrier before any thread stores hop words.
template <int W>
__device__ __forceinline__ bool step_acquire(const StepTok<W>& t, const Ctl& c, unsigned long long seq) {
  const uint32_t lane = threadIdx.x & 31u;
  bool mine = lane >= uint32_t(W);
  uint32_t round = 0;
  while (!__all_sync(0xffffffffu, mine)) {
    if (!mine) mine = ld_acquire_sys_u64(t.tok[t.rank] + uint64_t(lane) * kLineWords) >= seq;
    ++round;
    if (round > c.spin_rounds) {
      if ((round & 15u) == 0u && __any_sync(0xffffffffu, failed(c))) return false;
      if (round > c.max_rounds) {
        if (!mine) fail(c, kErrStepToken, t.rank, t.site);
        __syncwarp();
        return false;
      }
      __nanosleep(c.backoff_ns);
    }
  }
  return true;
}

}  // namespace lb_hop
}  // namespace muse_vr
