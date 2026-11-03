// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// Launch wrappers for the route all-gather form of the expert-output hop (c3_hb0.cuh):
//   push    stand-in producer after the expert kernel: every local slot row to every other rank (one warp per row);
//   consume one 512-thread CTA per token: the engine's fused finalize + one-shot push all-reduce result, computed on every
//           rank from all 8 slot rows: per slot an RMS norm with the engine's reduction tree for more than 64 tokens (512
//           threads, 8 columns per thread, fma chain, warp butterfly, then the 16 warp sums), then per owning rank in rank
//           order the slot-order fma accumulation, one bf16 rounding (that rank's partial), the push all-reduce's zero-word
//           rewrite, and the fp32 fold over ranks starting from rank 0's value with one bf16 rounding.
// Default off: used only through ops/c3_hb0.py, which callers select with a default-off knob.
#pragma once

#include "lb_hop_ops.cuh"  // make_site / make_ctl

#include <cuda_bf16.h>

#include "c3_hb0.cuh"

namespace sglang {
namespace muse_vr_hb0_ops {

namespace lh = muse_vr::lb_hop;
namespace hb = muse_vr::hb0;

inline constexpr int kThreads = 512;
inline constexpr int kWarps = kThreads / 32;
inline constexpr int kSmemRows = hb::kTopK * hb::kH * 2;  // the 8 slot rows in the tree's thread order, bf16
inline constexpr int kChunk = 4;                          // slots per staging round (all their loads in flight together)

// consume flags
inline constexpr uint32_t kEwF32 = 1u;        // route weights are fp32 (else bf16)
inline constexpr uint32_t kNoStockZero = 2u;  // skip the push all-reduce's zero-word rewrite
inline constexpr uint32_t kEarlyTrigger = 4u;  // trigger dependents at entry (else after the last store)
inline constexpr uint32_t kDeferRelease = 16u;  // sync-credit arena: release the PREVIOUS transfer from one extra CTA (see below)
inline constexpr uint32_t kPreWaitTable = 32u;  // far only: read the expert kernel's published slot table (expanded-index map +
                                                // route weights) BEFORE griddepcontrol.wait; requires the immediate predecessor
                                                // to publish it before its trigger (moe_ldw MUSE_VR_LDW_EARLY_SLOTTAB)
inline constexpr uint32_t kRowsReady = 64u;  // far only (with kPreWaitTable): no dependency wait before the rows; each token's
                                             // local rows are read once the expert kernel's flags of their passes carry its
                                             // launch epoch (moe_ldw MUSE_VR_LDW_ROWS_READY); the wait comes before exit
inline constexpr uint32_t kRrPerPass = 128u;  // flags per pass: FC2 tiles (4096 / 128) x epilogue warps (4)

struct ConsumeArgs {
  const int32_t* packed;  // [M, 8] expert << 16 | bf16 weight (expert from the high half)
  const int32_t* e2p;     // [M * 8] permuted row of a local slot, -1 otherwise
  const void* ew;         // [M, 8] route weights (bf16 or fp32), the values the engine's finalize multiplies by
  const uint16_t* g2;     // expert kernel output rows [*, 4096] bf16
  const uint16_t* norm_w; // [4096] bf16 post-expert norm weight
  uint16_t* out;          // [M, 4096] bf16
  float eps;
  int M;
  int n_local;  // experts per rank (owner = expert / n_local)
  uint32_t flags;
  unsigned long long* done;    // [3] per ring slot: count form = CTAs of the current call done reading; deferred form = the
                               // completion marker (transfer + 1) of the site's last call that used the slot
  unsigned long long* stamps;  // timing builds only (MUSE_VR_HB0_STAMPS): [CTA 0..7][8] global timer per phase, else null
  const unsigned long long* rr_pub;    // kRowsReady: the expert kernel's published launch epoch [1]
  const unsigned long long* rr_flags;  // kRowsReady: flags [pass][tile][warp] (epoch of the launch that stored the item)
  const int16_t* e2pass;               // kRowsReady: [M * 8] pass of each routed slot's row, -1 otherwise
  uint32_t rr_nflags;                  // kRowsReady: flags in rr_flags
  unsigned long long* rr_last;         // kRowsReady: per CTA, the epoch it last consumed (this kernel's words): a published
                                       // epoch not above it means the expert kernel did not publish for this launch (loud)
};

__device__ __forceinline__ unsigned long long ld_acquire_gpu_u64(const unsigned long long* p) {
  unsigned long long v;
  asm volatile("ld.acquire.gpu.global.u64 %0, [%1];" : "=l"(v) : "l"(p) : "memory");
  return v;
}

#ifdef MUSE_VR_HB0_STAMPS  // timing builds only: the stamp header is supplied by the test harness, not part of the plugin
#include "hb0_stamp_dev.cuh"
#else
#define HB0_STAMP(a, i) \
  do {                  \
  } while (0)
#endif

__device__ __forceinline__ float bf(uint32_t b) { return __uint_as_float((b & 0xffffu) << 16); }
__device__ __forceinline__ uint32_t bf16_rn(float f) { return uint32_t(__bfloat16_as_ushort(__float2bfloat16_rn(f))); }

__device__ __forceinline__ unsigned long long atom_add_acq_rel_gpu_u64(unsigned long long* p, unsigned long long v) {
  unsigned long long old;
  asm volatile("atom.acq_rel.gpu.global.add.u64 %0, [%1], %2;" : "=l"(old) : "l"(p), "l"(v) : "memory");
  return old;
}
__device__ __forceinline__ void st_relaxed_gpu_u64(unsigned long long* p, unsigned long long v) {
  asm volatile("st.relaxed.gpu.global.u64 [%0], %1;" ::"l"(p), "l"(v) : "memory");
}

// Thread tid of the tree owns columns 8 tid .. 8 tid + 7: tile tid / 16, tile columns c0 .. c0 + 7 with c0 = 8 (tid % 16).
// In the tile-aligned layout these are the last 6 - o values of group c0 / 6 and the first o + 2 of the next group of the
// same tile (o = c0 % 6, even; never a third group). take8: the 12 values of the two groups' word pairs as six bf16 pairs,
// this thread's 8 values = pairs o / 2 .. o / 2 + 3, in column order (the layout of a 16-byte load of the row).
__device__ __forceinline__ uint4 take8(const ulonglong2& p, const ulonglong2& q, uint32_t o2) {
  const uint32_t a0 = uint32_t(p.x), a1 = (uint32_t(p.x >> 32) & 0xffffu) | (uint32_t(p.y) << 16), a2 = uint32_t(p.y >> 16);
  const uint32_t b0 = uint32_t(q.x), b1 = (uint32_t(q.x >> 32) & 0xffffu) | (uint32_t(q.y) << 16), b2 = uint32_t(q.y >> 16);
  uint4 u;
  u.x = o2 == 0u ? a0 : (o2 == 1u ? a1 : a2);
  u.y = o2 == 0u ? a1 : (o2 == 1u ? a2 : b0);
  u.z = o2 == 0u ? a2 : (o2 == 1u ? b0 : b1);
  u.w = o2 == 0u ? b0 : (o2 == 1u ? b1 : b2);
  return u;
}

// Two bf16 of one 32-bit word as fp32 (column 2 p, 2 p + 1).
__device__ __forceinline__ float2 bf2(uint32_t w) { return make_float2(__uint_as_float(w << 16), __uint_as_float(w & 0xffff0000u)); }

// The tree's per-thread squared norm of one slot: per column y = value * weight (rounded; paired multiplies round per
// component like the scalar ones), then fma(y, y, acc) over the 8 columns in order.
__device__ __forceinline__ float sumsq8(const uint4& u, const float2 (&nw2)[4]) {
  const uint32_t q[4] = {u.x, u.y, u.z, u.w};
  float acc = 0.0f;
#pragma unroll
  for (int p = 0; p < 4; ++p) {
    const float2 y = __fmul2_rn(bf2(q[p]), nw2[p]);
    acc = __fmaf_rn(y.x, y.x, acc);
    acc = __fmaf_rn(y.y, y.y, acc);
  }
  return acc;
}


// Thread 0 of every consumer CTA, after the CTA barrier that ends the CTA's reads of the slot (and its scrub stores).
// Async-credit arena: the CTA's own release (lb_hop credit_release_async; the grid is then the site's unit count).
// Otherwise the count form: one gpu-scope acq_rel add per CTA on this rank's count word of the ring slot; the CTA whose add
// completes the grid's count (it reads grid - 1: every other CTA's add precedes its own in the word's order and is acquired
// by it) resets the word and runs the sync-form release once for the whole grid (lb_hop credit_release: high-water mark,
// fence.acq_rel.sys, relaxed stores to every producer rank). The next consume of the same ring slot is three transfers later
// and its CTAs count after their dependency wait (lb_hop precondition 2), so the reset precedes its adds.
// Deferred form (flag kDeferRelease, sync-credit arena): no count. The grid has one extra CTA (the last) that, after the
// dependency wait, releases the PREVIOUS transfer of the site (t - 1: consecutive consumes of a site are consecutive transfers)
// with the sync-form pattern (fence.acq_rel.sys, then relaxed stores of t into every producer rank's credit word): that consume
// completed before this kernel's wait returned (stream order through kernels that all wait, lb_hop precondition 2), so all its
// reads precede the release; the release overlaps this consume's token work instead of ending it. CTA 0 records the high-water
// mark of transfer t (read by the consume of t + 3, a later kernel on this rank). Producers of transfer T need T - 3 released,
// i.e. the consume of T - 2 started: no cycle with ring depth 3. The release happens only if the call of t - 1 actually ran: CTA
// 0 of every call writes the completion marker done[t mod 3] = t + 1 after its final barrier; the release CTA (after the
// dependency wait, so after that call completed) requires done[(t - 1) mod 3] == t, else it releases nothing and sets the sticky
// error word (a skipped call may have left received words unread / uncleared; the producers' credit waits then expire loudly).
template <int W>
__device__ __forceinline__ void deferred_release(const lh::Site<W>& s, const lh::Ctl& c, const ConsumeArgs& a, const lh::Xfer& x) {
  if (x.t < 1ull) return;
  if (*reinterpret_cast<const volatile unsigned long long*>(a.done + (x.t - 1ull) % lh::kRingSlots) != x.t) {
    if (!lh::failed(c)) lh::fail(c, lh::kErrCredit, s.rank, s.site);
    return;
  }
#if defined(MUSE_VR_HB0_PLANT_REL) && MUSE_VR_HB0_PLANT_REL == 3  // tests only: a planted EARLY deferred release (of t, not t - 1)
  const uint32_t released = uint32_t(x.t + 1ull);
#else
  const uint32_t released = uint32_t(x.t);  // transfers <= t - 1 released (word = last released transfer + 1)
#endif
  lh::fence_acq_rel_sys();
#pragma unroll
  for (int r = 0; r < W; ++r) lh::st_relaxed_sys_u32(s.credit[r] + s.rank, released);
}

template <int W>
__device__ __forceinline__ void consume_release(const lh::Site<W>& s, const lh::Ctl& c, const ConsumeArgs& a, const lh::Xfer& x,
                                                uint32_t rows_lin) {
  if (s.policy != lh::kRing) return;
  if (c.flags & lh::kAsyncCredit) {
    lh::credit_release_async(s, x, rows_lin);
    return;
  }
  unsigned long long* const d = a.done + x.copy;
  const unsigned long long old = atom_add_acq_rel_gpu_u64(d, 1ull);
#if defined(MUSE_VR_HB0_PLANT_REL) && MUSE_VR_HB0_PLANT_REL == 2  // tests only: a planted EARLY release (by the first CTA to count)
  if (old == 0ull) lh::credit_release(s, x, rows_lin);
  if (old + 1ull == uint64_t(gridDim.x)) st_relaxed_gpu_u64(d, 0ull);
#else
  if (old + 1ull == uint64_t(gridDim.x)) {
    st_relaxed_gpu_u64(d, 0ull);
    lh::credit_release(s, x, rows_lin);
  }
#endif
}

// One CTA per token (grid-stride over tokens). Per token, every thread works on its own 8 columns only, so the staging needs
// no CTA barrier: lanes 0..7 of each warp read the slot table (ballots and shuffles spread it), every thread loads its 16 bytes
// of each local slot row, polls the two word pairs of each received slot that hold its columns (tag-checked; all loads of a
// chunk of slots in flight together), keeps the 8 values per slot in its own shared-memory slot (written and read back only
// by this thread) and feeds the slot's squared norm through the warp butterfly right away. One CTA barrier per token (the warp
// sums, double-buffered by token parity); lane k of every warp then runs slot k's 16-value second level (the butterfly's add
// tree as scalar adds) and shuffles the inverse norm to the warp.
// The missed-advance guard is tested per thread (no barrier); a failed guard, slot map or poll poisons the CTA's rows.
template <int W, bool kUsePDL>
__global__ void __launch_bounds__(kThreads, 2)
    hb0_consume_kernel(const __grid_constant__ lh::Site<W> s, const __grid_constant__ lh::Ctl c,
                       const __grid_constant__ ConsumeArgs a) {
  extern __shared__ __align__(16) uint8_t dsm[];
  uint4* const rows = reinterpret_cast<uint4*>(dsm);  // [8][512]: thread tid's 8 columns of slot k at rows[k * 512 + tid]
  __shared__ float wsum[2][hb::kTopK][kWarps];
  __shared__ int s_fail;  // sticky for the CTA: a failure poisons this and every later token of the CTA
  const uint32_t tid = threadIdx.x, warp = tid >> 5, lane = tid & 31u;
  HB0_STAMP(a, 0);
  if (a.flags & kEarlyTrigger) device::PDLTriggerSecondary<kUsePDL>();
  float2 nw2[4];  // static weights before the dependency wait: this thread's 8 norm weights as fp32 pairs
  {
    const uint4 v = reinterpret_cast<const uint4*>(a.norm_w)[tid];
    nw2[0] = bf2(v.x);
    nw2[1] = bf2(v.y);
    nw2[2] = bf2(v.z);
    nw2[3] = bf2(v.w);
  }
  if (tid == 0) s_fail = 0;
  __syncthreads();  // s_fail is reset before any thread can set it (before the dependency wait: off the critical path)
  device::PDLWaitPrimary<kUsePDL>();  // expert output, slot map, weights; peers' words are tag-checked
  HB0_STAMP(a, 1);
  const lh::Xfer x = lh::xfer(s);
  const uint32_t rows_lin = uint32_t(a.M) * uint32_t(hb::kTopK);
  const bool defer = (a.flags & kDeferRelease) != 0u;
  const uint32_t ntok = defer ? gridDim.x - 1u : gridDim.x;  // token CTAs (deferred form: the last CTA only releases)
  if (defer && blockIdx.x == ntok) {
    if (tid == 0) deferred_release(s, c, a, x);
    if (!(a.flags & kEarlyTrigger)) device::PDLTriggerSecondary<kUsePDL>();
    return;
  }
  // the unit's missed-advance guard (lb_hop guard_enter's test, in every thread; thread 0 reports a failure)
  const bool in_units = blockIdx.x < s.units;
  const bool ok = in_units && *reinterpret_cast<const volatile unsigned long long*>(s.last_seq + blockIdx.x) != x.seq;
  if (!ok && tid == 0) lh::fail(c, in_units ? lh::kErrMissedAdvance : lh::kErrArgs, s.rank, s.site);
  const uint32_t hwm = lh::hwm_read(s, x);  // rows [rows_lin, hwm) are scrubbed after the tokens (not this transfer's rows)
  const uint32_t c0 = (tid & 15u) * 8u;
  const uint32_t off0 = (tid >> 4) * hb::kTileWords + 2u * (c0 / 6u), o2 = (c0 % 6u) >> 1;

  uint32_t it = 0;
  for (int t = blockIdx.x; t < a.M; t += int(ntok), ++it) {
    const uint32_t buf = it & 1u;
    // slot table: lane k (< 8) of every warp reads slot k (no barrier); owners / masks / positions / weights reach the other
    // lanes by ballot and shuffles
    int my_owner = -1, my_pos = -1;
    float my_w = 0.0f;
    bool my_bad = false;
    if (lane < uint32_t(hb::kTopK)) {
      const int32_t pk = a.packed[t * hb::kTopK + int(lane)];
      my_pos = a.e2p[t * hb::kTopK + int(lane)];
      my_w = (a.flags & kEwF32) ? static_cast<const float*>(a.ew)[t * hb::kTopK + int(lane)]
                                : bf(static_cast<const uint16_t*>(a.ew)[t * hb::kTopK + int(lane)]);
      const int e = pk >> 16;
      my_owner = (pk >= 0 && e < W * a.n_local) ? e / a.n_local : -1;
      if ((my_owner == int(s.rank)) != (my_pos >= 0)) {  // slot map and routing disagree
        my_bad = true;
        my_owner = -1;
      }
    }
    const uint32_t loc = __ballot_sync(0xffffffffu, my_owner == int(s.rank)) & 0xffu;
    const uint32_t rem = __ballot_sync(0xffffffffu, my_owner >= 0 && my_owner != int(s.rank)) & 0xffu;
    if (__ballot_sync(0xffffffffu, my_bad) != 0u && tid == 0) {
      lh::fail(c, lh::kErrArgs, s.rank, s.site);
      atomicOr(&s_fail, 1);
    }
    if (t == int(blockIdx.x)) HB0_STAMP(a, 2);
#ifdef MUSE_VR_HB0_PLANT_REL  // tests only: CTA 0 reads its first token's slots late (a busy wait of ~1e6 clock cycles, well inside
                              // the producers' bounded credit wait; modes 1-3; 2 / 3 also plant an early release)
    if (blockIdx.x == 0 && t == 0) {
      const long long c0 = clock64();
      while (clock64() - c0 < 1000000ll) {
      }
    }
#endif
    // staging in chunks of kChunk slots (all loads of a chunk in flight together); each slot's 8 values go to this thread's own
    // shared slot, its squared norm through the warp butterfly straight into the warp sums
    bool live = ok;
#pragma unroll
    for (int k0 = 0; k0 < hb::kTopK; k0 += kChunk) {
      ulonglong2 v[kChunk][2];  // per chunk slot: the local row's 16 bytes in v[i][0], or the two received word pairs
      int pos[kChunk];          // permuted rows of the chunk's local slots (shuffled by every lane: loc is warp-uniform)
#pragma unroll
      for (int i = 0; i < kChunk; ++i) pos[i] = (loc >> (k0 + i) & 1u) ? __shfl_sync(0xffffffffu, my_pos, k0 + i) : 0;
      uint32_t pend = 0;        // bit 2 i + h: word pair h of chunk slot i not accepted yet
      if (live) {
#pragma unroll
        for (int i = 0; i < kChunk; ++i) {
          const int k = k0 + i;
          if (loc >> k & 1u) {
            v[i][0] = *reinterpret_cast<const ulonglong2*>(a.g2 + size_t(pos[i]) * hb::kH + 8u * tid);
          } else if (rem >> k & 1u) {
            const unsigned long long* const q = hb::lin_word(s, s.rank, x.copy, uint32_t(t) * hb::kTopK + uint32_t(k), off0);
            v[i][0] = lh::ld_relaxed_sys_v2_b64(q);
            v[i][1] = lh::ld_relaxed_sys_v2_b64(q + 2);
            pend |= 3u << (2 * i);
          }
        }
        // tag checks; the misses are re-polled (every load of a round is issued before any is checked); bounded like
        // lb_hop's polls; an expired or overrun poll never consumes
        uint32_t round = 0;
        int st = 0;
        for (;;) {
#pragma unroll
          for (int b = 0; b < 2 * kChunk; ++b) {
            if (!(pend >> b & 1u)) continue;
            const uint32_t t0 = uint32_t(v[b >> 1][b & 1].x >> 48), t1 = uint32_t(v[b >> 1][b & 1].y >> 48);
            if (t0 == x.tag && t1 == x.tag)
              pend &= ~(1u << b);
            else if (lh::tag_ahead(t0, x.tag) || lh::tag_ahead(t1, x.tag))
              st = 2;
          }
          if (st || !pend) break;
          ++round;
          if (round > c.spin_rounds) {
            if ((round & 15u) == 0u && lh::failed(c)) {  // failed elsewhere: poison, keep its code
              st = 3;
              break;
            }
            if (round > c.max_rounds) {
              st = 1;
              break;
            }
            __nanosleep(c.backoff_ns);
          }
#pragma unroll
          for (int b = 0; b < 2 * kChunk; ++b)
            if (pend >> b & 1u)
              v[b >> 1][b & 1] = lh::ld_relaxed_sys_v2_b64(hb::lin_word(s, s.rank, x.copy, uint32_t(t) * hb::kTopK + uint32_t(k0 + (b >> 1)),
                                                                        off0 + 2u * uint32_t(b & 1)));
        }
        if (st) {
          if (st != 3) lh::fail(c, st == 2 ? lh::kErrOverrun : lh::kErrPayload, s.rank, s.site);
          atomicOr(&s_fail, 1);
          live = false;
        }
      }
      // the tree's first level for this chunk's slots (every lane takes part; a slot without an owner, or a failed thread,
      // contributes 0): per-thread fma chain over its 8 columns, warp butterfly, lane 0 -> the warp sum
#pragma unroll
      for (int i = 0; i < kChunk; ++i) {
        const int k = k0 + i;
        float z = 0.0f;
        if (live && ((loc | rem) >> k & 1u)) {
          const uint4 u = (loc >> k & 1u) ? make_uint4(uint32_t(v[i][0].x), uint32_t(v[i][0].x >> 32), uint32_t(v[i][0].y),
                                                       uint32_t(v[i][0].y >> 32))
                                          : take8(v[i][0], v[i][1], o2);
          rows[k * kThreads + tid] = u;
          z = sumsq8(u, nw2);
        }
#pragma unroll
        for (int o = 16; o > 0; o >>= 1) z = __fadd_rn(z, __shfl_xor_sync(0xffffffffu, z, o));
        if (lane == 0) wsum[buf][k][warp] = z;
      }
    }
    if (t == int(blockIdx.x)) HB0_STAMP(a, 3);
    if (t == int(blockIdx.x)) HB0_STAMP(a, 4);
    __syncwarp();
#ifndef MUSE_VR_HB0_PLANT_NOBAR  // tests only: a planted race (the warp-sum barrier dropped) that racecheck must flag
    __syncthreads();  // every warp's sums and every thread's failure flag of this token
#endif
    if (t == int(blockIdx.x)) HB0_STAMP(a, 5);
    uint4* const dst = reinterpret_cast<uint4*>(a.out + size_t(t) * hb::kH) + tid;
    // volatile (strong) read: with several tokens per CTA a fast thread's atomicOr for the next token may race it (a 1 only adds
    // NaN); made warp-uniform, since the pass below shuffles with a full mask
    if (__any_sync(0xffffffffu, !ok || *reinterpret_cast<volatile int*>(&s_fail) != 0)) {
      const uint32_t nan2 = uint32_t(lh::kPoisonBf16) | (uint32_t(lh::kPoisonBf16) << 16);
      *dst = make_uint4(nan2, nan2, nan2, nan2);
      continue;
    }
    // second level: lane k (< 8) of every warp sums slot k's 16 warp sums in the butterfly's exact add tree (xor 16 adds +0 to
    // non-negative sums: exact; then pairs 8, 4, 2, 1), then the inverse RMS (sum / 4096 is exact as sum * 2^-12); shuffled to
    // the other lanes per slot below
    float my_inv = 0.0f;
    if (lane < uint32_t(hb::kTopK)) {
      const float* z = wsum[buf][lane];
      const float a0 = __fadd_rn(z[0], z[8]), a1 = __fadd_rn(z[1], z[9]), a2 = __fadd_rn(z[2], z[10]), a3 = __fadd_rn(z[3], z[11]);
      const float a4 = __fadd_rn(z[4], z[12]), a5 = __fadd_rn(z[5], z[13]), a6 = __fadd_rn(z[6], z[14]), a7 = __fadd_rn(z[7], z[15]);
      const float b0 = __fadd_rn(a0, a4), b1 = __fadd_rn(a1, a5), b2 = __fadd_rn(a2, a6), b3 = __fadd_rn(a3, a7);
      const float sum = __fadd_rn(__fadd_rn(b0, b2), __fadd_rn(b1, b3));
      my_inv = rsqrtf(__fadd_rn(__fmul_rn(sum, 1.0f / float(hb::kH)), a.eps));
    }
    // per owning rank in rank order: slot-order fma accumulation, bf16 partial, zero-word rewrite, fp32 fold
    // (paired fp32 operations round per component exactly like the scalar ones: same values, half the instructions)
    float fold[8];
#pragma unroll
    for (int r = 0; r < W; ++r) {
      float2 acc[4] = {make_float2(0.f, 0.f), make_float2(0.f, 0.f), make_float2(0.f, 0.f), make_float2(0.f, 0.f)};
#pragma unroll
      for (int k = 0; k < hb::kTopK; ++k) {
        if (__shfl_sync(0xffffffffu, my_owner, k) != r) continue;
        const float wk = __shfl_sync(0xffffffffu, my_w, k), inv = __shfl_sync(0xffffffffu, my_inv, k);
        const float2 wk2 = make_float2(wk, wk), inv2 = make_float2(inv, inv);
        const uint4 u = rows[k * kThreads + tid];
        const uint32_t q[4] = {u.x, u.y, u.z, u.w};
#pragma unroll
        for (int p = 0; p < 4; ++p) acc[p] = __ffma2_rn(wk2, __fmul2_rn(__fmul2_rn(bf2(q[p]), nw2[p]), inv2), acc[p]);
      }
      uint32_t pb[8];
#pragma unroll
      for (int p = 0; p < 4; ++p) {
        pb[2 * p] = bf16_rn(acc[p].x);
        pb[2 * p + 1] = bf16_rn(acc[p].y);
      }
      if (!(a.flags & kNoStockZero)) {
#pragma unroll
        for (int j = 0; j < 4; ++j)
          if ((pb[2 * j] | pb[2 * j + 1]) == 0u) pb[2 * j] = 0x8000u;
      }
#pragma unroll
      for (int j = 0; j < 8; ++j) fold[j] = (r == 0) ? bf(pb[j]) : __fadd_rn(fold[j], bf(pb[j]));
    }
    uint32_t o[4];
#pragma unroll
    for (int j = 0; j < 4; ++j) o[j] = bf16_rn(fold[2 * j]) | (bf16_rn(fold[2 * j + 1]) << 16);
    *dst = make_uint4(o[0], o[1], o[2], o[3]);
    if (t == int(blockIdx.x)) HB0_STAMP(a, 6);
  }
  // kRing: linear rows [rows_lin, high-water of the slot's previous use) of this rank's copy back to tag 0
  if (hwm > rows_lin) {
    const uint32_t pairs = s.pitch / 2u;
    const uint64_t total = uint64_t(hwm - rows_lin) * pairs;
    for (uint64_t i = uint64_t(blockIdx.x) * kThreads + tid; i < total; i += uint64_t(ntok) * kThreads)
      lh::st_relaxed_sys_v2_b64(hb::lin_word(s, s.rank, x.copy, rows_lin + uint32_t(i / pairs), uint32_t(i % pairs) * 2u),
                                0ull, 0ull);
  }
  __syncwarp();
  __syncthreads();  // every thread's reads of the slot and scrub stores precede the release
  if (tid == 0) {
    if (ok) lh::guard_exit(s, blockIdx.x, x.seq);
    if (!defer) {
      consume_release(s, c, a, x, rows_lin);
    } else if (blockIdx.x == 0) {
      lh::hwm_write(s, x, rows_lin);
      *reinterpret_cast<volatile unsigned long long*>(a.done + x.copy) = x.t + 1ull;  // completion marker (deferred release)
    }
  }
  HB0_STAMP(a, 7);
  if (!(a.flags & kEarlyTrigger)) device::PDLTriggerSecondary<kUsePDL>();
}

// FAR form (ops/c3_hb0.py far): the engine's fused finalize + one-shot push all-reduce as ONE kernel after the expert kernel,
// with no producer in the expert kernel. One 512-thread CTA per token; thread tid = the tree's columns 8 tid .. 8 tid + 7.
// Per token: lanes 0..7 of each warp read the slot table (issued first after the dependency wait); every thread loads its 16
// bytes of each LOCAL slot row (the local slots compacted, up to 4 rows in flight per round), computes the slot's squared norm
// (fma chain, warp butterfly -> warp sums), one CTA barrier, lane k runs local slot k's second level; then this rank's partial
// over its local slots in slot order (one bf16 rounding, the push all-reduce's zero-word rewrite) is stored as-is, 16 bytes per
// thread, into every other rank's copy, and the other ranks' partials are polled the same way. Ready test (the push all-reduce's
// own protocol): a received 32-bit word is ready when it is NOT zero - the zero-word rewrite guarantees a pushed word is never
// all-zero bits, and every slot word is zero before its producer writes (zeroed at allocation, cleared by its consumer right
// after the read and before the slot's release, and by the ring scrub). Bounded polls; an expired poll never consumes; a CTA
// that finds the sticky error word already set consumes nothing. Then the fp32 fold in rank order from rank 0's value, one bf16
// rounding. Ring site (far_spec): row = token, one block per source rank, 1024 words per row; slot reuse by sync credits (warp 0
// acquires after the first token's warp-sum barrier, overlapping the second level; a CTA barrier before the first push orders it
// before every store); the far call of t releases (count
// form or, with kDeferRelease, t - 1 from one extra CTA). Needs the zero-word rewrite (refuses kNoStockZero).
inline constexpr uint32_t kFarRowWords = 2u * kThreads;   // 1024 words per (source, token) row: 16 bytes per thread
inline constexpr uint32_t kFarWidth = 3u * kFarRowWords;  // site width (value slots) = 3072
inline constexpr int kFarPeers = 4;                       // peer ranks polled together

// the transfer from already-loaded sequence / ring words (the same arithmetic as lh::xfer, without its loads)
template <int W>
__device__ __forceinline__ lh::Xfer xfer_of(const lh::Site<W>& s, unsigned long long seq, unsigned long long ring_base) {
  lh::Xfer x;
  x.seq = seq;
  if (s.policy == lh::kRing) {
    x.t = ring_base + s.index;
    x.copy = uint32_t(x.t % lh::kRingSlots);
  } else {
    x.t = seq;
    x.copy = s.parity ? uint32_t(seq & 1ull) : 0u;
  }
  x.tag = lh::tag_of(x.t);
  return x;
}

template <int W, bool kUsePDL>
__global__ void __launch_bounds__(kThreads, 2)
    hb0_far_kernel(const __grid_constant__ lh::Site<W> s, const __grid_constant__ lh::Ctl c,
                   const __grid_constant__ ConsumeArgs a) {
  extern __shared__ __align__(16) uint8_t dsm[];
  uint4* const rows = reinterpret_cast<uint4*>(dsm);  // [8][512]: thread tid's 8 columns of local slot k at rows[k * 512 + tid]
  __shared__ float wsum[2][hb::kTopK][kWarps];
  __shared__ int s_fail;  // sticky for the CTA
  __shared__ int s_cred;  // warp 0's credit acquire result (read after the first token barrier)
  const uint32_t tid = threadIdx.x, warp = tid >> 5, lane = tid & 31u;
  HB0_STAMP(a, 0);
  if (a.flags & kEarlyTrigger) device::PDLTriggerSecondary<kUsePDL>();
  float2 nw2[4];
  {
    const uint4 v = reinterpret_cast<const uint4*>(a.norm_w)[tid];
    nw2[0] = bf2(v.x);
    nw2[1] = bf2(v.y);
    nw2[2] = bf2(v.z);
    nw2[3] = bf2(v.w);
  }
  if (tid == 0) {
    s_fail = 0;
    s_cred = 1;
  }
  const bool defer = (a.flags & kDeferRelease) != 0u;
  const bool rr = (a.flags & kRowsReady) != 0u;
  const uint32_t ntok = defer ? gridDim.x - 1u : gridDim.x;
  const bool tok_cta = !(defer && blockIdx.x == ntok);
  // kPreWaitTable: the first token's expanded-index map and route weights BEFORE the dependency wait (the expert kernel, our
  // immediate predecessor, published them in every CTA before its fence + trigger); re-checked after the wait below
  const bool prew = (a.flags & kPreWaitTable) != 0u && tok_cta;
  int32_t pk0 = -1, pos0 = -1, posc = -1;
  float w0 = 0.0f;
  if (prew && lane < uint32_t(hb::kTopK)) {
    const int t0 = int(blockIdx.x);
    pos0 = a.e2p[t0 * hb::kTopK + int(lane)];
    w0 = (a.flags & kEwF32) ? static_cast<const float*>(a.ew)[t0 * hb::kTopK + int(lane)]
                            : bf(static_cast<const uint16_t*>(a.ew)[t0 * hb::kTopK + int(lane)]);
  }
  __syncthreads();
  if (!rr) device::PDLWaitPrimary<kUsePDL>();  // kRowsReady: the wait comes before exit (the rows by their flags)
  HB0_STAMP(a, 1);
  // HEAD, one round trip: every word needed before the first staging load is loaded here, branch-free, before any of them is used
  // (no load waits on another load's value). Words: the first token's slot table (lane k < 8 reads slot k; other lanes, and CTAs
  // without a first token, read an in-range duplicate and ignore it), the sequence and ring words, this CTA's last sequence and
  // the sticky error word (read after griddepcontrol.wait: an earlier far kernel may be the writer, and the wait makes its write
  // visible). With kPreWaitTable the map read here is the post-wait re-read, compared before any push.
  const int th = (tok_cta && int(blockIdx.x) < a.M) ? int(blockIdx.x) : 0;
  static_assert((hb::kTopK & (hb::kTopK - 1)) == 0, "lane duplicate index needs a power-of-two top-k");
  const int hi = th * hb::kTopK + int(lane & uint32_t(hb::kTopK - 1));
  const int32_t pk_h = a.packed[hi];
  const int32_t pos_h = a.e2p[hi];
  const float w_h = (a.flags & kEwF32) ? static_cast<const float*>(a.ew)[hi] : bf(static_cast<const uint16_t*>(a.ew)[hi]);
  const unsigned long long seq_h = lh::ld_relaxed_gpu_u64(s.seq64);
  const unsigned long long rb_h = s.policy == lh::kRing ? lh::ld_relaxed_gpu_u64(s.ring_base) : 0ull;
  const bool in_units = blockIdx.x < s.units;
  const unsigned long long last_seq =
      *reinterpret_cast<const volatile unsigned long long*>(s.last_seq + (in_units ? blockIdx.x : 0u));
  const bool err0 = lh::failed(c);
  const unsigned long long rr_ep = rr ? *reinterpret_cast<const volatile unsigned long long*>(a.rr_pub) : 0ull;
  const unsigned long long rr_prev =
      (rr && tok_cta) ? *reinterpret_cast<const volatile unsigned long long*>(a.rr_last + blockIdx.x) : 0ull;
  if (tok_cta && lane < uint32_t(hb::kTopK)) {
    pk0 = pk_h;
    posc = pos_h;
    if (!prew) {
      pos0 = pos_h;
      w0 = w_h;
    }
  }
  const lh::Xfer x = xfer_of(s, seq_h, rb_h);
  const uint32_t rows_m = uint32_t(a.M);
  if (defer && blockIdx.x == ntok) {
    if (tid == 0) deferred_release(s, c, a, x);
    if (!(a.flags & kEarlyTrigger)) device::PDLTriggerSecondary<kUsePDL>();
    // every exit is behind a dependency wait (with kRowsReady this is the release CTA's only one; otherwise it returns at once)
    device::PDLWaitPrimary<kUsePDL>();
    return;
  }
  // a set sticky error word means an earlier failure may have left slot words non-zero: consume nothing until a reset
  const bool ok = in_units & (last_seq != x.seq) & !err0;
  if (!ok && tid == 0 && !lh::failed(c)) lh::fail(c, in_units ? lh::kErrMissedAdvance : lh::kErrArgs, s.rank, s.site);
  const uint32_t hwm = lh::hwm_read(s, x);  // issued now, used by the scrub at the end (off the tail's critical path)

  uint32_t it = 0;
  for (int t = blockIdx.x; t < a.M; t += int(ntok), ++it) {
    const uint32_t buf = it & 1u;
    int32_t pk = pk0, my_pos = pos0;
    float my_w = w0;
    if (it > 0u && lane < uint32_t(hb::kTopK)) {
      pk = a.packed[t * hb::kTopK + int(lane)];
      my_pos = a.e2p[t * hb::kTopK + int(lane)];
      my_w = (a.flags & kEwF32) ? static_cast<const float*>(a.ew)[t * hb::kTopK + int(lane)]
                                : bf(static_cast<const uint16_t*>(a.ew)[t * hb::kTopK + int(lane)]);
    }
    const bool pw = prew && it == 0u;  // this token's map came from the pre-wait read
    int my_owner = -1;
    bool my_bad = false;
    if (!pw && lane < uint32_t(hb::kTopK)) {
      const int e = pk >> 16;
      my_owner = (pk >= 0 && e < W * a.n_local) ? e / a.n_local : -1;
      if ((my_owner == int(s.rank)) != (my_pos >= 0)) {
        my_bad = true;
        my_owner = -1;
      }
    }
    // local slots: from the routing word, or (pre-wait map) from the map itself; the pre-wait map is checked against the
    // routing word and the post-wait re-read after the staging loads are issued (its round trip overlaps theirs)
    const uint32_t loc = __ballot_sync(0xffffffffu, pw ? (lane < uint32_t(hb::kTopK) && my_pos >= 0) : (my_owner == int(s.rank))) & 0xffu;
    if (!pw && __ballot_sync(0xffffffffu, my_bad) != 0u && tid == 0) {
      lh::fail(c, lh::kErrArgs, s.rank, s.site);
      atomicOr(&s_fail, 1);
    }
    if (t == int(blockIdx.x)) HB0_STAMP(a, 2);
#if defined(MUSE_VR_HB0_PLANT_REL) && MUSE_VR_HB0_PLANT_REL == 4
    constexpr bool rr_poll = false;  // tests only: rows read without their flags (the non-vacuity twin of the flag edge)
#else
    const bool rr_poll = rr;
#endif
    if (rr_poll) {  // this token's local rows: every flag of their passes must carry the epoch (2 flags per thread, slot = tid / 64)
      const uint32_t k = tid >> 6;
      int st = 0;  // 0 ready, 1 expired, 2 newer epoch or bad pass
      if ((loc >> k) & 1u) {
        const int pass = int(a.e2pass[t * hb::kTopK + int(k)]);
        const uint32_t f0 = uint32_t(pass) * kRrPerPass + 2u * (tid & 63u);
        if (pass < 0 || rr_ep <= rr_prev || f0 + 1u >= a.rr_nflags) {
          st = 2;
        } else {
          unsigned long long v0 = ld_acquire_gpu_u64(a.rr_flags + f0), v1 = ld_acquire_gpu_u64(a.rr_flags + f0 + 1u);
          uint32_t round = 0;
          while (!(v0 == rr_ep && v1 == rr_ep)) {
            if (v0 > rr_ep || v1 > rr_ep) {
              st = 2;
              break;
            }
            ++round;
            if (round > c.spin_rounds) {
              if (round > c.max_rounds || ((round & 15u) == 0u && lh::failed(c))) {
                st = 1;
                break;
              }
              __nanosleep(c.backoff_ns);
            }
            v0 = ld_acquire_gpu_u64(a.rr_flags + f0);
            v1 = ld_acquire_gpu_u64(a.rr_flags + f0 + 1u);
          }
        }
      }
      // every thread's acquires before any thread's row reads (CTA barrier); a failed poll: nothing pushed, NaN out
      if (__syncthreads_or(st != 0)) {
        if (st != 0 && !lh::failed(c)) lh::fail(c, lh::kErrPayload, s.rank, s.site);
        if (tid == 0) atomicOr(&s_fail, 1);
      }
    }
    // local slots, compacted: up to 4 rows in flight per round (16 bytes per thread each), own shared slot, squared norm ->
    // warp butterfly -> warp sum (the loop bounds and slot indices are warp-uniform)
    uint32_t lrem = loc;
    while (lrem) {
      int kk[4];
      uint4 v[4];
#pragma unroll
      for (int i = 0; i < 4; ++i) {
        kk[i] = lrem ? __ffs(lrem) - 1 : -1;
        if (lrem) lrem &= lrem - 1u;
        const int pos = kk[i] >= 0 ? __shfl_sync(0xffffffffu, my_pos, kk[i]) : 0;
        if (kk[i] >= 0) v[i] = reinterpret_cast<const uint4*>(a.g2 + size_t(pos) * hb::kH)[tid];
      }
#pragma unroll
      for (int i = 0; i < 4; ++i) {
        if (kk[i] < 0) continue;
        rows[kk[i] * kThreads + tid] = v[i];
        float z = sumsq8(v[i], nw2);
#pragma unroll
        for (int o = 16; o > 0; o >>= 1) z = __fadd_rn(z, __shfl_xor_sync(0xffffffffu, z, o));
        if (lane == 0) wsum[buf][kk[i]][warp] = z;
      }
    }
    if (pw) {  // the pre-wait map must equal the post-wait re-read and agree with the routing word; else consume nothing
      bool bad = false;
      if (lane < uint32_t(hb::kTopK)) {
        const int e = pk >> 16;
        const int owner = (pk >= 0 && e < W * a.n_local) ? e / a.n_local : -1;
        bad = posc != my_pos || (owner == int(s.rank)) != (my_pos >= 0);
      }
      if (__ballot_sync(0xffffffffu, bad) != 0u && lane == 0) {
        if (warp == 0) lh::fail(c, lh::kErrArgs, s.rank, s.site);
        atomicOr(&s_fail, 1);
      }
    }
    if (t == int(blockIdx.x)) HB0_STAMP(a, 3);
    __syncwarp();
#ifndef MUSE_VR_HB0_PLANT_NOBAR  // tests only: a planted race (the warp-sum barrier dropped) that racecheck must flag
    __syncthreads();  // the warp sums; warp 0's credit acquire before any store below
#endif
    if (t == int(blockIdx.x)) HB0_STAMP(a, 4);
    // producer side, first token only: warp 0 acquires the slot's credits (sync form) here, so the round trip overlaps the
    // other warps' second level / partial; the CTA barrier before the first push orders it before every store
    if (W > 1 && it == 0u && warp == 0) {
      const bool okc = lh::credit_acquire(s, c, x);
      if (lane == 0 && !okc) s_cred = 0;
    }
    float my_inv = 0.0f;
    if (lane < uint32_t(hb::kTopK) && (loc >> lane & 1u)) {
      const float* z = wsum[buf][lane];
      const float a0 = __fadd_rn(z[0], z[8]), a1 = __fadd_rn(z[1], z[9]), a2 = __fadd_rn(z[2], z[10]), a3 = __fadd_rn(z[3], z[11]);
      const float a4 = __fadd_rn(z[4], z[12]), a5 = __fadd_rn(z[5], z[13]), a6 = __fadd_rn(z[6], z[14]), a7 = __fadd_rn(z[7], z[15]);
      const float b0 = __fadd_rn(a0, a4), b1 = __fadd_rn(a1, a5), b2 = __fadd_rn(a2, a6), b3 = __fadd_rn(a3, a7);
      const float sum = __fadd_rn(__fadd_rn(b0, b2), __fadd_rn(b1, b3));
      my_inv = rsqrtf(__fadd_rn(__fmul_rn(sum, 1.0f / float(hb::kH)), a.eps));
    }
    float2 acc[4] = {make_float2(0.f, 0.f), make_float2(0.f, 0.f), make_float2(0.f, 0.f), make_float2(0.f, 0.f)};
#pragma unroll
    for (int k = 0; k < hb::kTopK; ++k) {
      if (!(loc >> k & 1u)) continue;
      const float wk = __shfl_sync(0xffffffffu, my_w, k), inv = __shfl_sync(0xffffffffu, my_inv, k);
      const float2 wk2 = make_float2(wk, wk), inv2 = make_float2(inv, inv);
      const uint4 u = rows[k * kThreads + tid];
      const uint32_t q[4] = {u.x, u.y, u.z, u.w};
#pragma unroll
      for (int p = 0; p < 4; ++p) acc[p] = __ffma2_rn(wk2, __fmul2_rn(__fmul2_rn(bf2(q[p]), nw2[p]), inv2), acc[p]);
    }
    uint32_t mine[4];  // this rank's partial: bf16 pairs, never an all-zero word (the zero-word rewrite)
#pragma unroll
    for (int p = 0; p < 4; ++p) {
      uint32_t lo = bf16_rn(acc[p].x), hi = bf16_rn(acc[p].y);
      if ((lo | hi) == 0u) lo = 0x8000u;
      mine[p] = lo | (hi << 16);
    }
    if (W > 1 && it == 0u) __syncthreads();  // warp 0's credit acquire (and s_cred) before any thread's first push
    if (t == int(blockIdx.x)) HB0_STAMP(a, 5);
    const bool fail_now = !ok || *reinterpret_cast<volatile int*>(&s_cred) == 0 || *reinterpret_cast<volatile int*>(&s_fail) != 0;
    if (!fail_now) {  // push: this thread's 16 bytes into row t of block `rank` of every other rank's copy
      const uint4 m4 = make_uint4(mine[0], mine[1], mine[2], mine[3]);
#pragma unroll
      for (int r = 0; r < W; ++r) {
        if (uint32_t(r) == s.rank) continue;
        lh::st_relaxed_sys_v4_b32(reinterpret_cast<uint32_t*>(lh::word_ptr(s, uint32_t(r), x.copy, s.rank, uint32_t(t), 2u * tid)), m4);
      }
    }
    // the other ranks' partials: up to kFarPeers ranks' 16 bytes in flight together; a word is ready when non-zero; bounded
    float fold[8];
    bool good = !fail_now;
#pragma unroll
    for (int r0 = 0; r0 < W; r0 += kFarPeers) {
      uint4 wv[kFarPeers];
      uint32_t pend = 0;  // bit j: rank r0 + j not complete yet
      if (good) {
#pragma unroll
        for (int j = 0; j < kFarPeers; ++j) {
          const int r = r0 + j;
          if (r >= W || uint32_t(r) == s.rank) continue;
          wv[j] = lh::ld_relaxed_sys_v4_b32(reinterpret_cast<const uint32_t*>(lh::word_ptr(s, s.rank, x.copy, uint32_t(r), uint32_t(t), 2u * tid)));
          pend |= 1u << j;
        }
        uint32_t round = 0;
        int st = 0;
        for (;;) {
#pragma unroll
          for (int j = 0; j < kFarPeers; ++j)
            if ((pend >> j & 1u) && wv[j].x != 0u && wv[j].y != 0u && wv[j].z != 0u && wv[j].w != 0u) pend &= ~(1u << j);
          if (!pend) break;
          ++round;
          if (round > c.spin_rounds) {
            if ((round & 15u) == 0u && lh::failed(c)) {
              st = 3;
              break;
            }
            if (round > c.max_rounds) {
              st = 1;
              break;
            }
            __nanosleep(c.backoff_ns);
          }
#pragma unroll
          for (int j = 0; j < kFarPeers; ++j)
            if (pend >> j & 1u)
              wv[j] = lh::ld_relaxed_sys_v4_b32(
                  reinterpret_cast<const uint32_t*>(lh::word_ptr(s, s.rank, x.copy, uint32_t(r0 + j), uint32_t(t), 2u * tid)));
        }
        if (st) {
          if (st != 3) lh::fail(c, lh::kErrPayload, s.rank, s.site);
          atomicOr(&s_fail, 1);
          good = false;
        } else {  // consumed: clear the words for the slot's next use (before the release below)
#pragma unroll
          for (int j = 0; j < kFarPeers; ++j) {
            const int r = r0 + j;
            if (r >= W || uint32_t(r) == s.rank) continue;
            lh::st_relaxed_sys_v4_b32(reinterpret_cast<uint32_t*>(lh::word_ptr(s, s.rank, x.copy, uint32_t(r), uint32_t(t), 2u * tid)),
                                      make_uint4(0u, 0u, 0u, 0u));
          }
        }
      }
#pragma unroll
      for (int j = 0; j < kFarPeers; ++j) {
        const int r = r0 + j;
        if (r >= W) continue;
        const uint4 pr = (uint32_t(r) == s.rank) ? make_uint4(mine[0], mine[1], mine[2], mine[3]) : wv[j];
        const uint32_t q[4] = {pr.x, pr.y, pr.z, pr.w};
#pragma unroll
        for (int p = 0; p < 4; ++p) {
          const float2 f = bf2(q[p]);
          fold[2 * p] = (r == 0) ? f.x : __fadd_rn(fold[2 * p], f.x);
          fold[2 * p + 1] = (r == 0) ? f.y : __fadd_rn(fold[2 * p + 1], f.y);
        }
      }
    }
    if (t == int(blockIdx.x)) HB0_STAMP(a, 6);
    uint4* const dst = reinterpret_cast<uint4*>(a.out + size_t(t) * hb::kH) + tid;
    if (!good) {
      const uint32_t nan2 = uint32_t(lh::kPoisonBf16) | (uint32_t(lh::kPoisonBf16) << 16);
      *dst = make_uint4(nan2, nan2, nan2, nan2);
    } else {
      uint32_t o[4];
#pragma unroll
      for (int j = 0; j < 4; ++j) o[j] = bf16_rn(fold[2 * j]) | (bf16_rn(fold[2 * j + 1]) << 16);
      *dst = make_uint4(o[0], o[1], o[2], o[3]);
    }
  }
  // kRowsReady: the expert kernel has completed before this grid completes (the stream's completion order for later kernels)
  // every exit is behind a dependency wait: with kRowsReady this is the expert kernel's completion (the grid completes after it);
  // otherwise the wait above already returned and this one returns at once
  device::PDLWaitPrimary<kUsePDL>();
  if (rr && tid == 0) a.rr_last[blockIdx.x] = rr_ep;  // read by this CTA index of the next call on this expert kernel's buffer
  // rows [M, hwm) of every source block of this rank's copy back to zero (the slot's previous use was larger)
  lh::ring_scrub(s, x, rows_m, hwm, blockIdx.x * kThreads + tid, ntok * kThreads);
  __syncwarp();
  __syncthreads();  // every thread's reads and clears of the slot (and scrub stores) precede the release
  if (tid == 0) {
    if (ok) lh::guard_exit(s, blockIdx.x, x.seq);
    if (!defer) {
      consume_release(s, c, a, x, rows_m);
    } else if (blockIdx.x == 0) {
      lh::hwm_write(s, x, rows_m);
      *reinterpret_cast<volatile unsigned long long*>(a.done + x.copy) = x.t + 1ull;  // completion marker (deferred release)
    }
  }
  HB0_STAMP(a, 7);
  if (!(a.flags & kEarlyTrigger)) device::PDLTriggerSecondary<kUsePDL>();
}

// Stand-in producer (after the expert kernel, in stream order): one warp per local slot row.
template <int W, bool kUsePDL>
__global__ void __launch_bounds__(256) hb0_push_kernel(const __grid_constant__ lh::Site<W> s, const __grid_constant__ lh::Ctl c,
                                                       const int32_t* e2p, const uint16_t* g2, int M, uint32_t gi_limit) {
  __shared__ int s_ok;
  device::PDLWaitPrimary<kUsePDL>();
  const lh::Xfer x = lh::xfer(s);
  if (threadIdx.x < 32) {
    const bool okc = lh::credit_acquire(s, c, x);
    if (threadIdx.x == 0) s_ok = okc ? 1 : 0;
  }
  __syncthreads();
  if (s_ok) {
    const uint32_t lane = threadIdx.x & 31u, nwarps = gridDim.x * (blockDim.x >> 5);
    for (uint32_t lin = blockIdx.x * (blockDim.x >> 5) + (threadIdx.x >> 5); lin < uint32_t(M) * hb::kTopK; lin += nwarps) {
      const int pos = e2p[lin];
      if (pos < 0) continue;
      hb::push_route_row<W>(s, x, lin, g2 + size_t(pos) * hb::kH, lane, gi_limit);
    }
  }
  device::PDLTriggerSecondary<kUsePDL>();
}

}  // namespace muse_vr_hb0_ops

template <int W, bool kUsePDL>
struct MuseVrHb0 {
  static void check_site(int64_t rows_max, int64_t width, int64_t pitch, int64_t policy, int64_t M) {
    using namespace host;
    RuntimeCheck(width == int64_t(muse_vr::hb0::kWidth) && pitch >= int64_t(muse_vr::hb0::kRowWords),
                 "hb0: site width 4608 (tile-aligned rows, 48 words per tile), pitch >= 1536 words");
    RuntimeCheck((policy & 0xff) == muse_vr::lb_hop::kRing && !(policy & muse_vr::lb_hop::kLoopbackDev),
                 "hb0: ring site, no loopback twin");
    RuntimeCheck(M >= 1 && int64_t(W) * rows_max >= M * muse_vr::hb0::kTopK, "hb0: M * 8 linear rows must fit (W * rows_max)");
  }

  static void consume(const tvm::ffi::TensorView table, const tvm::ffi::TensorView ctl, int64_t rank, int64_t rows_max,
                      int64_t width, int64_t pitch, int64_t policy, int64_t parity, int64_t index, int64_t site, int64_t units,
                      int64_t max_rounds, int64_t spin_rounds, int64_t flags, int64_t backoff_ns, int64_t step_flags,
                      const tvm::ffi::TensorView packed, const tvm::ffi::TensorView e2p, const tvm::ffi::TensorView ew,
                      const tvm::ffi::TensorView g2, const tvm::ffi::TensorView norm_w, const tvm::ffi::TensorView out,
                      double eps, int64_t M, int64_t n_local, const tvm::ffi::TensorView done,
                      const tvm::ffi::TensorView stamps, int64_t use_stamps) {
    using namespace host;
    namespace o = muse_vr_hb0_ops;
    check_site(rows_max, width, pitch, policy, M);
    RuntimeCheck(out.size(0) >= M && out.size(1) == muse_vr::hb0::kH && g2.size(1) == muse_vr::hb0::kH &&
                     norm_w.numel() == muse_vr::hb0::kH && packed.size(0) >= M && packed.size(1) == muse_vr::hb0::kTopK &&
                     e2p.numel() >= M * muse_vr::hb0::kTopK && ew.size(0) >= M && ew.size(1) == muse_vr::hb0::kTopK,
                 "hb0 consume: out [M, 4096], g2 [*, 4096], norm_w [4096], packed / ew [M, 8], e2p [M * 8]");
    RuntimeCheck(n_local >= 1 && units >= 1, "hb0 consume: n_local, units");
    RuntimeCheck(!(uint32_t(step_flags) & 8u), "hb0 consume: the two-CTA cluster form was removed (single-CTA form only)");
    const bool acred = (uint32_t(flags) & muse_vr::lb_hop::kAsyncCredit) != 0u;
    const bool defer = (uint32_t(step_flags) & o::kDeferRelease) != 0u;
    RuntimeCheck(!(defer && acred), "hb0 consume: the deferred release needs a sync-credit arena");
    RuntimeCheck(acred || done.numel() >= int64_t(muse_vr::lb_hop::kRingSlots),
                 "hb0 consume: the count form needs one count word per ring slot");
    const auto s = muse_vr_lb_hop_ops::make_site<W>(table, rank, rows_max, width, pitch, policy, parity, index, site, units);
    const auto c = muse_vr_lb_hop_ops::make_ctl(ctl, max_rounds, spin_rounds, flags, backoff_ns);
    o::ConsumeArgs a{};
    a.packed = static_cast<const int32_t*>(packed.data_ptr());
    a.e2p = static_cast<const int32_t*>(e2p.data_ptr());
    a.ew = ew.data_ptr();
    a.g2 = static_cast<const uint16_t*>(g2.data_ptr());
    a.norm_w = static_cast<const uint16_t*>(norm_w.data_ptr());
    a.out = static_cast<uint16_t*>(out.data_ptr());
    a.eps = float(eps);
    a.M = int(M);
    a.n_local = int(n_local);
    a.flags = uint32_t(step_flags);
    a.done = acred ? nullptr : static_cast<unsigned long long*>(done.data_ptr());
    a.stamps = use_stamps ? static_cast<unsigned long long*>(stamps.data_ptr()) : nullptr;
    const auto stream = LaunchKernel::resolve_device(out.device());
    // one missed-advance word per CTA (grid <= units); async credits: one credit word per CTA, the grid is the unit count;
    // count form: one CTA per token up to the unit count
    const int64_t grid = (acred ? units : (M < units ? M : units)) + (defer ? 1 : 0);
    const auto kern = o::hb0_consume_kernel<W, kUsePDL>;
    static bool attr = false;
    if (!attr) {
      RuntimeDeviceCheck(cudaFuncSetAttribute(kern, cudaFuncAttributeMaxDynamicSharedMemorySize, o::kSmemRows));
      attr = true;
    }
    LaunchKernel(dim3(unsigned(grid)), o::kThreads, stream, o::kSmemRows).enable_pdl(kUsePDL)(kern, s, c, a);
  }

  static void far(const tvm::ffi::TensorView table, const tvm::ffi::TensorView ctl, int64_t rank, int64_t rows_max,
                  int64_t width, int64_t pitch, int64_t policy, int64_t parity, int64_t index, int64_t site, int64_t units,
                  int64_t max_rounds, int64_t spin_rounds, int64_t flags, int64_t backoff_ns, int64_t step_flags,
                  const tvm::ffi::TensorView packed, const tvm::ffi::TensorView e2p, const tvm::ffi::TensorView ew,
                  const tvm::ffi::TensorView g2, const tvm::ffi::TensorView norm_w, const tvm::ffi::TensorView out, double eps,
                  int64_t M, int64_t n_local, const tvm::ffi::TensorView done, const tvm::ffi::TensorView stamps,
                  int64_t use_stamps, const tvm::ffi::TensorView rr_pub, const tvm::ffi::TensorView rr_last,
                  const tvm::ffi::TensorView rr_flags, const tvm::ffi::TensorView e2pass) {
    using namespace host;
    namespace o = muse_vr_hb0_ops;
    RuntimeCheck(width == int64_t(o::kFarWidth) && pitch >= int64_t(o::kFarRowWords),
                 "hb0 far: site width 3072 (1024 words per token row), pitch >= 1024 words");
    RuntimeCheck((policy & 0xff) == muse_vr::lb_hop::kRing && !(policy & muse_vr::lb_hop::kLoopbackDev),
                 "hb0 far: ring site, no loopback twin");
    RuntimeCheck(M >= 1 && rows_max >= M, "hb0 far: M rows per source block must fit");
    RuntimeCheck(out.size(0) >= M && out.size(1) == muse_vr::hb0::kH && g2.size(1) == muse_vr::hb0::kH &&
                     norm_w.numel() == muse_vr::hb0::kH && packed.size(0) >= M && packed.size(1) == muse_vr::hb0::kTopK &&
                     e2p.numel() >= M * muse_vr::hb0::kTopK && ew.size(0) >= M && ew.size(1) == muse_vr::hb0::kTopK,
                 "hb0 far: out [M, 4096], g2 [*, 4096], norm_w [4096], packed / ew [M, 8], e2p [M * 8]");
    RuntimeCheck(n_local >= 1 && units >= 1, "hb0 far: n_local, units");
    const bool acred = (uint32_t(flags) & muse_vr::lb_hop::kAsyncCredit) != 0u;
    RuntimeCheck(!acred, "hb0 far: sync-credit arena only");
    RuntimeCheck(!(uint32_t(step_flags) & o::kNoStockZero), "hb0 far: needs the zero-word rewrite (its ready test is a non-zero word)");
    RuntimeCheck(done.numel() >= int64_t(muse_vr::lb_hop::kRingSlots), "hb0 far: one count word per ring slot");
    const bool defer = (uint32_t(step_flags) & o::kDeferRelease) != 0u;
    const bool rr = (uint32_t(step_flags) & o::kRowsReady) != 0u;
    RuntimeCheck(!rr || (uint32_t(step_flags) & o::kPreWaitTable) != 0u, "hb0 far: ROWS_READY needs PRE_WAIT_TABLE");
    const int64_t grid_rr = (M < units ? M : units) + (defer ? 1 : 0);
    RuntimeCheck(!rr || (rr_pub.numel() >= 1 && rr_last.numel() >= grid_rr && rr_flags.numel() >= int64_t(o::kRrPerPass) &&
                         e2pass.numel() >= M * muse_vr::hb0::kTopK),
                 "hb0 far: ROWS_READY needs the expert kernel's epoch word, one last-epoch word per CTA, flags, e2pass [M * 8]");
    const auto s = muse_vr_lb_hop_ops::make_site<W>(table, rank, rows_max, width, pitch, policy, parity, index, site, units);
    const auto c = muse_vr_lb_hop_ops::make_ctl(ctl, max_rounds, spin_rounds, flags, backoff_ns);
    o::ConsumeArgs a{};
    a.packed = static_cast<const int32_t*>(packed.data_ptr());
    a.e2p = static_cast<const int32_t*>(e2p.data_ptr());
    a.ew = ew.data_ptr();
    a.g2 = static_cast<const uint16_t*>(g2.data_ptr());
    a.norm_w = static_cast<const uint16_t*>(norm_w.data_ptr());
    a.out = static_cast<uint16_t*>(out.data_ptr());
    a.eps = float(eps);
    a.M = int(M);
    a.n_local = int(n_local);
    a.flags = uint32_t(step_flags);
    a.done = static_cast<unsigned long long*>(done.data_ptr());
    a.stamps = use_stamps ? static_cast<unsigned long long*>(stamps.data_ptr()) : nullptr;
    a.rr_pub = rr ? static_cast<const unsigned long long*>(rr_pub.data_ptr()) : nullptr;
    a.rr_flags = rr ? static_cast<const unsigned long long*>(rr_flags.data_ptr()) : nullptr;
    a.e2pass = rr ? static_cast<const int16_t*>(e2pass.data_ptr()) : nullptr;
    a.rr_nflags = rr ? uint32_t(rr_flags.numel()) : 0u;
    a.rr_last = rr ? static_cast<unsigned long long*>(rr_last.data_ptr()) : nullptr;
    const auto stream = LaunchKernel::resolve_device(out.device());
    const int64_t grid = (M < units ? M : units) + (defer ? 1 : 0);
    const auto kern = o::hb0_far_kernel<W, kUsePDL>;
    static bool attr = false;
    if (!attr) {
      RuntimeDeviceCheck(cudaFuncSetAttribute(kern, cudaFuncAttributeMaxDynamicSharedMemorySize, o::kSmemRows));
      attr = true;
    }
    LaunchKernel(dim3(unsigned(grid)), o::kThreads, stream, o::kSmemRows).enable_pdl(kUsePDL)(kern, s, c, a);
  }

  static void push(const tvm::ffi::TensorView table, const tvm::ffi::TensorView ctl, int64_t rank, int64_t rows_max,
                   int64_t width, int64_t pitch, int64_t policy, int64_t parity, int64_t index, int64_t site, int64_t units,
                   int64_t max_rounds, int64_t spin_rounds, int64_t flags, int64_t backoff_ns,
                   const tvm::ffi::TensorView e2p, const tvm::ffi::TensorView g2, int64_t M, int64_t grid, int64_t gi_limit) {
    using namespace host;
    check_site(rows_max, width, pitch, policy, M);
    RuntimeCheck(g2.size(1) == muse_vr::hb0::kH && e2p.numel() >= M * muse_vr::hb0::kTopK && grid >= 1,
                 "hb0 push: g2 [*, 4096], e2p [M * 8], grid >= 1");
    const auto s = muse_vr_lb_hop_ops::make_site<W>(table, rank, rows_max, width, pitch, policy, parity, index, site, units);
    const auto c = muse_vr_lb_hop_ops::make_ctl(ctl, max_rounds, spin_rounds, flags, backoff_ns);
    const auto stream = LaunchKernel::resolve_device(g2.device());
    LaunchKernel(dim3(unsigned(grid)), 256, stream)
        .enable_pdl(kUsePDL)(muse_vr_hb0_ops::hb0_push_kernel<W, kUsePDL>, s, c, static_cast<const int32_t*>(e2p.data_ptr()),
                             static_cast<const uint16_t*>(g2.data_ptr()), int(M),
                             uint32_t(gi_limit > 0 && gi_limit < int64_t(muse_vr::hb0::kGroups) ? gi_limit : muse_vr::hb0::kGroups));
  }
};

}  // namespace sglang
