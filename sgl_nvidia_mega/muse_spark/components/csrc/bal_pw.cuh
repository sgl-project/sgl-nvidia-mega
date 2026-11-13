// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
//
// Per-step expert-parallel assignment over a fixed replica pool, pairwise form (component ``bal``; development form, not wired
// into the server). Integer arithmetic only.
//
// pw_role(): one CTA (kPwThreads or more threads, a multiple of 32) computes the assignment of one step from the step's routed
// ids (all rows, every rank's rows), off the expert stage's own threads:
//   1. counts the routed rows per expert while it polls the ids words (each accepted on its own matching tag);
//   2. gives every routed expert the cost c(e) of bal_assign.cuh (Cost<world>, passes of pass_rows rows) and starts it on its
//      home rank;
//   3. groups the routed replicated experts by their holder pair {a < b} (static: the placement's id arithmetic), each pair in
//      (cost descending, id ascending) order;
//   4. runs `sweeps` sweeps over a fixed perfect-matching schedule of the rank pairs (world - 1 rounds per sweep): the pairs of
//      one round share no rank, so one lane each runs them at once; a pair takes its members off both holders and re-adds them
//      in order, each to the holder with the lower load (ties: a);
//   5. publishes the result as 4-bit holder codes (kNoHolder = not routed), 8 experts per tagged 64-bit word (32 words), plus
//      a digest word: word = tag << 32 | payload, so a consumer accepts each word on its own matching load.
// The result depends only on the ids, the static pair table and pass_rows / sweeps, so every rank and every CTA that runs it on
// the same ids publishes the same words, with no communication.
//
// Static pair table (built on the host from the placement, no traffic data; pw_table() in bal.py), uint32 words:
//   [0, 256)                       placement words (bal_assign.cuh packing)
//   [256, 512)                     per expert: pair id (bits 0-7, 0xff = not replicated) | index in the pair's list (8-15)
//   [512, 512 + kPwMaxPairs)       per pair: a | b << 4 | members << 8
//   [kOffMem, + kPwMaxPairs * 32)  per pair: member expert ids in id order (32 slots)
//   [kOffSch, + 7 * 4)             per round (world - 1 rounds) and slot (world / 2): pair id (0xff = none)
//
// Memory model: the ids words and the published words carry their tag in the same aligned 64-bit word as their payload
// (single-copy atomic per the PTX ISA memory consistency model) and each has one producer thread; consumers accept a word on
// its own matching load. Shared-memory phases are ordered by bar.sync (CTA) and bar.warp.sync (the one-warp rounds). Every
// wait is bounded and sets a sticky error bit on expiry; the role then publishes nothing.

#pragma once

#include "bal_assign.cuh"

#include <cstdint>

namespace muse_vr::bal {

inline constexpr int kPwMaxPairs = 28;    // world 8: 8 * 7 / 2
inline constexpr int kPwMaxMembers = 32;  // members of one holder pair (decision bits per pair fit one word)
inline constexpr int kPwMaxRounds = 7;
inline constexpr int kPwMaxSlots = 4;
inline constexpr int kPwOffPair = 256, kPwOffAB = 512, kPwOffMem = kPwOffAB + kPwMaxPairs,
                     kPwOffSch = kPwOffMem + kPwMaxPairs * kPwMaxMembers,
                     kPwTableWords = kPwOffSch + kPwMaxRounds * kPwMaxSlots;
inline constexpr int kPwThreads = 256;  // minimum CTA size (one thread per expert in the per-expert phases)
inline constexpr int kPwWords = 33;     // 32 assignment words + 1 digest word
inline constexpr int kPwMaxRows = 1024;
inline constexpr uint32_t kNoHolder = 0xfu;
inline constexpr uint32_t kNoPair = 0xffu;
inline constexpr uint32_t kErrWait = 4u;

struct PwSmem {
  uint32_t tab[kNumExperts];
  uint32_t pinfo[kNumExperts];
  uint32_t pab[kPwMaxPairs];
  uint8_t mem[kPwMaxPairs][kPwMaxMembers];
  uint8_t sch[kPwMaxRounds][kPwMaxSlots];
  uint8_t rnd[kPwMaxPairs];                  // the round of each pair in a sweep (kNoPair: none)
  uint8_t src[kPwMaxRounds][kMaxWorld];      // the pair holding rank q in round r
  uint32_t cnt[kNumExperts];
  uint32_t cost[kNumExperts];
  uint32_t scost[kPwMaxPairs][kPwMaxMembers + 1];  // the pair's routed members' costs in assignment order (padded row)
  uint32_t pkey[kPwMaxPairs][kPwMaxMembers];       // per static member: cost << 8 | (255 - index), 0 if not routed
  uint8_t pos[kNumExperts];                    // a routed replicated expert's position in its pair's order
  uint32_t nact[kPwMaxPairs];
  uint32_t dec[kPwMaxPairs];                   // bit j: member j of the order sits on b
  int32_t load[kMaxWorld];
  uint8_t asg[kNumExperts];
  uint32_t err;
};

// Static part (may run before the ids are available): the pair table into shared memory, the counters zeroed. All threads.
__device__ __forceinline__ void pw_stage(const uint32_t* __restrict__ tbl, PwSmem& s, int tid, int nt) {
  for (int i = tid; i < kNumExperts; i += nt) {
    s.tab[i] = tbl[i];
    s.pinfo[i] = tbl[kPwOffPair + i];
    s.cnt[i] = 0u;
  }
  for (int i = tid; i < kPwMaxPairs; i += nt) {
    s.pab[i] = tbl[kPwOffAB + i];
    s.nact[i] = 0u;
    s.dec[i] = 0u;
  }
  for (int i = tid; i < kPwMaxPairs * kPwMaxMembers; i += nt) {
    (&s.mem[0][0])[i] = static_cast<uint8_t>(tbl[kPwOffMem + i]);
  }
  for (int i = tid; i < kPwMaxRounds * kPwMaxSlots; i += nt) (&s.sch[0][0])[i] = static_cast<uint8_t>(tbl[kPwOffSch + i]);
  for (int i = tid; i < kPwMaxPairs * kPwMaxMembers; i += nt) (&s.pkey[0][0])[i] = 0u;  // padding entries stay 0
  for (int i = tid; i < kMaxWorld; i += nt) s.load[i] = 0;
  if (tid == 0) s.err = 0u;
}

// Count the step's routed rows per expert from M rows x 8 ids words. load(row, k) makes ONE 64-bit load of that word (no
// ordering constraint: the payload rides in the word) and a word is accepted when its high half equals tag32. Each thread owns
// words i = tid + j * nt (j < 32; M * 8 <= 32 * nt); per round it issues the loads of all its pending words, 8 at a time, before
// it checks any of them, so one round costs about one load round trip. spin_rounds rounds without backoff, then backoff_ns
// sleeps, at most max_rounds rounds. Returns false (and sets kErrWait) on expiry.
template <class Load>
__device__ __forceinline__ bool pw_count(const Load& load, uint32_t tag32, int M, PwSmem& s, int tid, int nt,
                                         uint32_t spin_rounds, uint32_t max_rounds, uint32_t backoff_ns) {
  const int n = M * kTopK;
  uint32_t pending = 0u;
  {
    int k = 0;
    for (int i = tid; i < n; i += nt, ++k) pending |= 1u << k;
  }
  uint32_t err = 0u;
  for (uint32_t round = 0; pending != 0u; ++round) {
    for (int base = 0; base < 32 && (pending >> base) != 0u; base += 8) {
      unsigned long long v[8];
#pragma unroll
      for (int u = 0; u < 8; ++u) {
        const int jj = base + u;
        const int i = tid + jj * nt;
        v[u] = ((pending >> jj) & 1u) ? load(i >> 3, i & 7) : 0ull;
      }
#pragma unroll
      for (int u = 0; u < 8; ++u) {
        const int jj = base + u;
        if (((pending >> jj) & 1u) != 0u && static_cast<uint32_t>(v[u] >> 32) == tag32) {
          pending &= ~(1u << jj);
          const uint32_t e = (static_cast<uint32_t>(v[u]) >> 16) & 0xffffu;
          if (e < uint32_t(kNumExperts)) {
            atomicAdd(&s.cnt[e], 1u);
          } else if (e != kPadId) {
            err |= kErrBadId;
          }
        }
      }
    }
    if (pending == 0u) break;
    if (round >= max_rounds) {
      err |= kErrWait;
      break;
    }
    if (round >= spin_rounds) __nanosleep(backoff_ns);
  }
  if (err != 0u) atomicOr(&s.err, err);
  return (err & kErrWait) == 0u;
}

// The assignment from the counts (after a CTA barrier covering pw_count). All threads of the CTA call it (nt >= 256); it
// ends with the 4-bit holder codes in s.asg and returns after a CTA barrier.
template <int kWorld>
__device__ __forceinline__ void pw_assign(uint32_t pass_rows, uint32_t pass_inv, int sweeps, PwSmem& s, int tid,
                                          int variant = 1) {
  constexpr uint32_t kPer = kNumExperts / kWorld;
  static_assert(kPer >= 32, "one warp's 32 consecutive experts share their home rank");
  const int warp = tid >> 5, lane = tid & 31;
  // costs; every routed expert starts on its home rank
  if (tid < kNumExperts) {
    const uint32_t e = static_cast<uint32_t>(tid);
    const uint32_t rows = s.cnt[e];
    const uint32_t p = ((rows + pass_rows - 1u) * pass_inv) >> 16;
    const uint32_t c = rows != 0u ? Cost<kWorld>::a + ((Cost<kWorld>::b * rows) >> 8) + Cost<kWorld>::p * (p - 1u) : 0u;
    s.cost[e] = c;
    s.asg[e] = static_cast<uint8_t>(rows != 0u ? (s.tab[e] & 0xfu) : kNoHolder);
    if (tid < kPwMaxRounds * kPwMaxSlots) {  // static schedule tables for the register rounds (from sch, staged before)
      const int r = tid / kPwMaxSlots;
      const uint32_t pi = s.sch[r][tid % kPwMaxSlots];
      if (pi != kNoPair) {
        const uint32_t ab = s.pab[pi];
        s.rnd[pi] = static_cast<uint8_t>(r);
        s.src[r][ab & 0xfu] = static_cast<uint8_t>(pi);
        s.src[r][(ab >> 4) & 0xfu] = static_cast<uint8_t>(pi);
      }
    }
    uint32_t sum = c;
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) sum += __shfl_xor_sync(kFull, sum, o);
    if (lane == 0 && sum != 0u) atomicAdd(&s.load[e / kPer], static_cast<int32_t>(sum));
  }
  __syncthreads();
  // each routed replicated expert finds its position in its pair's (cost descending, id ascending) order
  if (tid < kNumExperts) {
    const uint32_t e = static_cast<uint32_t>(tid);
    const uint32_t pi = s.pinfo[e] & 0xffu;
    const uint32_t c = s.cost[e];
    if (pi != kNoPair && c != 0u) {
      const uint32_t ab = s.pab[pi];
      const uint32_t nm = (ab >> 8) & 0xffu;
      uint32_t pos = 0u, act = 0u;
      for (uint32_t j = 0; j < nm; ++j) {
        const uint32_t f = s.mem[pi][j];
        const uint32_t cf = s.cost[f];
        act += cf != 0u ? 1u : 0u;
        pos += (cf > c || (cf == c && f < e)) ? 1u : 0u;
      }
      s.pos[e] = static_cast<uint8_t>(pos);
      s.scost[pi][pos] = c;
      if ((s.tab[e] & 0xfu) == ((ab >> 4) & 0xfu)) atomicOr(&s.dec[pi], 1u << pos);  // starts on its home = b
      if (pos == 0u) s.nact[pi] = act;
    }
  }
#ifndef MUSE_VR_BAL_PW_TEST_PLANT_RACE
  __syncthreads();
#endif  // test-only build: the barrier dropped, a planted shared-memory race (sanitizer non-vacuity control)
  // rounds, register form (variant 1): one warp; lane p owns pair p for every round; every lane keeps all rank loads in
  // registers; in round r the lanes of the round's pairs re-add their members, then each rank's new load is broadcast from
  // the lane of the pair that holds it in round r (a perfect matching: every rank is in exactly one pair of the round).
  if (variant == 1 && warp == 0) {
    constexpr int kPairs = kWorld * (kWorld - 1) / 2;
    int32_t L[kWorld];
#pragma unroll
    for (int q = 0; q < kWorld; ++q) L[q] = s.load[q];
    const bool mine = lane < kPairs;
    const uint32_t ab = mine ? s.pab[lane] : 0u;
    const uint32_t a = ab & 0xfu, b = (ab >> 4) & 0xfu;
    const uint32_t n = mine ? s.nact[lane] : 0u;
    const uint32_t myr = mine ? s.rnd[lane] : kNoPair;
    uint32_t d = mine ? s.dec[lane] : 0u;
    int32_t sa = 0, sb = 0;
    for (uint32_t j = 0; j < n; ++j) {
      const int32_t c = static_cast<int32_t>(s.scost[lane][j]);
      if ((d >> j) & 1u) {
        sb += c;
      } else {
        sa += c;
      }
    }
    for (int sw = 0; sw < sweeps; ++sw) {
      for (int r = 0; r < kWorld - 1; ++r) {
        int32_t na = 0, nb = 0;
        if (myr == static_cast<uint32_t>(r)) {
#pragma unroll
          for (int q = 0; q < kWorld; ++q) {
            na = (uint32_t(q) == a) ? L[q] : na;
            nb = (uint32_t(q) == b) ? L[q] : nb;
          }
          if (n != 0u) {
            na -= sa;
            nb -= sb;
            uint32_t nd = 0u;
            int32_t xa = 0, xb = 0;
            for (uint32_t j = 0; j < n; ++j) {
              const int32_t c = static_cast<int32_t>(s.scost[lane][j]);
              if (na <= nb) {
                na += c;
                xa += c;
              } else {
                nb += c;
                xb += c;
                nd |= 1u << j;
              }
            }
            d = nd;
            sa = xa;
            sb = xb;
          }
        }
#pragma unroll
        for (int q = 0; q < kWorld; ++q) {
          const uint32_t sl = s.src[r][q];
          const int32_t v = (uint32_t(q) == a) ? na : nb;
          L[q] = __shfl_sync(kFull, v, static_cast<int>(sl & 31u));
        }
      }
    }
    if (mine) s.dec[lane] = d;
  }
  // rounds, shared-memory form (variant 0): in round r, lane i runs pair sch[r][i] (the pairs of a round share no rank)
  if (variant == 0 && warp == 0) {
    for (int sw = 0; sw < sweeps; ++sw) {
      for (int r = 0; r < kWorld - 1; ++r) {
        const uint32_t pi = lane < kWorld / 2 ? s.sch[r][lane] : kNoPair;
        if (pi != kNoPair) {
          const uint32_t n = s.nact[pi];
          if (n != 0u) {
            const uint32_t ab = s.pab[pi];
            const uint32_t a = ab & 0xfu, b = (ab >> 4) & 0xfu;
            int32_t la = s.load[a], lb = s.load[b];
            const uint32_t d = s.dec[pi];
            for (uint32_t j = 0; j < n; ++j) {
              const int32_t c = static_cast<int32_t>(s.scost[pi][j]);
              if ((d >> j) & 1u) {
                lb -= c;
              } else {
                la -= c;
              }
            }
            uint32_t nd = 0u;
            for (uint32_t j = 0; j < n; ++j) {
              const int32_t c = static_cast<int32_t>(s.scost[pi][j]);
              if (la <= lb) {
                la += c;
              } else {
                lb += c;
                nd |= 1u << j;
              }
            }
            s.load[a] = la;
            s.load[b] = lb;
            s.dec[pi] = nd;
          }
        }
        __syncwarp();
      }
    }
  }
  __syncthreads();
  if (tid < kNumExperts) {
    const uint32_t e = static_cast<uint32_t>(tid);
    const uint32_t pi = s.pinfo[e] & 0xffu;
    if (pi != kNoPair && s.cost[e] != 0u) {
      const uint32_t ab = s.pab[pi];
      s.asg[e] = static_cast<uint8_t>(((s.dec[pi] >> s.pos[e]) & 1u) ? ((ab >> 4) & 0xfu) : (ab & 0xfu));
    }
  }
  __syncthreads();
}

// Publish (one full warp, after pw_assign): lane l stores word l = tag << 32 | the codes of experts 8l .. 8l + 7 (4 bits each,
// expert 8l in bits 3..0); lane 0 then stores word 32 = tag << 32 | digest. write(i, v) stores one 64-bit word. Returns the
// digest (same value in every lane).
template <class Write>
__device__ __forceinline__ uint32_t pw_publish(const Write& write, uint32_t tag32, const PwSmem& s, uint32_t lane) {
  uint32_t payload = 0u, h = 0u;
#pragma unroll
  for (int j = 0; j < 8; ++j) {
    const uint32_t e = lane * 8u + uint32_t(j);
    const uint32_t code = s.asg[e] & 0xfu;
    payload |= code << (4 * j);
    const uint32_t y = ((e << 8) | code) * 0x9e3779b1u;
    h += y ^ (y >> 16);
  }
#pragma unroll
  for (int o = 16; o > 0; o >>= 1) h += __shfl_xor_sync(kFull, h, o);
  h ^= h >> 16;
  h *= 0x85ebca6bu;
  h ^= h >> 13;
  write(static_cast<int>(lane), (static_cast<unsigned long long>(tag32) << 32) | payload);
  if (lane == 0u) write(32, (static_cast<unsigned long long>(tag32) << 32) | h);
  return h;
}

// Variant 2 (the read form): the same function as pw_assign + pw_publish, with the dependent chains shortened:
//   - pair order from packed keys (one vector shared load per 4 members, no member -> cost indirection);
//   - warp 0 keeps each pair's sorted costs (kMaxM >= the pair's static members), its decision word, all rank loads and the
//     round schedule in registers; a round is a predicated unrolled greedy + kWorld shuffles, no shared memory inside;
//   - warp 0 then forms the 32 code words itself (each expert's pair decision by a shuffle from the pair's lane) and publishes,
//     with no further CTA barrier.
// All threads call it (nt >= 256, after the barrier covering pw_count); warps other than 0 return after the order phase.
template <int kWorld, int kMaxM, class Write>
__device__ __forceinline__ void pw_assign_publish(uint32_t pass_rows, uint32_t pass_inv, int sweeps, PwSmem& s, int tid,
                                                  const Write& write, uint32_t tag32) {
  constexpr uint32_t kPer = kNumExperts / kWorld;
  constexpr int kPairs = kWorld * (kWorld - 1) / 2;
  static_assert(kPer >= 32 && kPairs <= 32, "warp layout");
  const int warp = tid >> 5, lane = tid & 31;
  if (tid < kNumExperts) {
    const uint32_t e = static_cast<uint32_t>(tid);
    const uint32_t rows = s.cnt[e];
    const uint32_t p = ((rows + pass_rows - 1u) * pass_inv) >> 16;
    const uint32_t c = rows != 0u ? Cost<kWorld>::a + ((Cost<kWorld>::b * rows) >> 8) + Cost<kWorld>::p * (p - 1u) : 0u;
    s.cost[e] = c;
    uint32_t sum = c;
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) sum += __shfl_xor_sync(kFull, sum, o);
    if (lane == 0 && sum != 0u) atomicAdd(&s.load[e / kPer], static_cast<int32_t>(sum));
    const uint32_t pinf = s.pinfo[e];
    const uint32_t pi = pinf & 0xffu, j = (pinf >> 8) & 0xffu;
    if (pi != kNoPair) s.pkey[pi][j] = c != 0u ? ((c << 8) | (255u - j)) : 0u;
    if (tid < kPwMaxRounds * kPwMaxSlots) {
      const int r = tid / kPwMaxSlots;
      const uint32_t q = s.sch[r][tid % kPwMaxSlots];
      if (q != kNoPair) {
        const uint32_t ab = s.pab[q];
        s.rnd[q] = static_cast<uint8_t>(r);
        s.src[r][ab & 0xfu] = static_cast<uint8_t>(q);
        s.src[r][(ab >> 4) & 0xfu] = static_cast<uint8_t>(q);
      }
    }
  }
  __syncthreads();
  if (tid < kNumExperts) {
    const uint32_t e = static_cast<uint32_t>(tid);
    const uint32_t pinf = s.pinfo[e];
    const uint32_t pi = pinf & 0xffu, j = (pinf >> 8) & 0xffu;
    const uint32_t c = s.cost[e];
    if (pi != kNoPair && c != 0u) {
      const uint32_t ab = s.pab[pi];
      const uint32_t nm = (ab >> 8) & 0xffu;
      const uint32_t key = (c << 8) | (255u - j);
      uint32_t pos = 0u, act = 0u;
      const uint4* row = reinterpret_cast<const uint4*>(&s.pkey[pi][0]);
      for (uint32_t q = 0; q < (nm + 3u) / 4u; ++q) {
        const uint4 k4 = row[q];
        act += (k4.x != 0u) + (k4.y != 0u) + (k4.z != 0u) + (k4.w != 0u);
        pos += (k4.x > key) + (k4.y > key) + (k4.z > key) + (k4.w > key);
      }
      s.pos[e] = static_cast<uint8_t>(pos);
      s.scost[pi][pos] = c;
      if ((s.tab[e] & 0xfu) == ((ab >> 4) & 0xfu)) atomicOr(&s.dec[pi], 1u << pos);
      if (pos == 0u) s.nact[pi] = act;
    }
  }
#ifndef MUSE_VR_BAL_PW_TEST_PLANT_RACE
  __syncthreads();
#endif
  if (warp != 0) return;
  int32_t L[kWorld];
#pragma unroll
  for (int q = 0; q < kWorld; ++q) L[q] = s.load[q];
  const bool mine = lane < kPairs;
  const uint32_t ab = mine ? s.pab[lane] : 0u;
  const uint32_t a = ab & 0xfu, b = (ab >> 4) & 0xfu;
  const uint32_t n = mine ? s.nact[lane] : 0u;
  const uint32_t myr = mine ? s.rnd[lane] : kNoPair;
  uint32_t d = mine ? s.dec[lane] : 0u;
  int32_t cr[kMaxM];
  int32_t sa = 0, sb = 0;
#pragma unroll
  for (int j = 0; j < kMaxM; ++j) {
    cr[j] = (uint32_t(j) < n) ? static_cast<int32_t>(s.scost[lane][j]) : 0;
    if ((d >> j) & 1u) {
      sb += cr[j];
    } else {
      sa += cr[j];
    }
  }
  uint32_t srow[kWorld - 1];
#pragma unroll
  for (int r = 0; r < kWorld - 1; ++r) {
    uint32_t w = 0u;
#pragma unroll
    for (int q = 0; q < kWorld; ++q) w |= (static_cast<uint32_t>(s.src[r][q]) & 15u) << (4 * q);
    srow[r] = w;  // low 4 bits of each rank's pair id; bit 4 (pair ids up to 27) in srow_hi below
  }
  uint32_t srow_hi[kWorld - 1];
#pragma unroll
  for (int r = 0; r < kWorld - 1; ++r) {
    uint32_t w = 0u;
#pragma unroll
    for (int q = 0; q < kWorld; ++q) w |= ((static_cast<uint32_t>(s.src[r][q]) >> 4) & 1u) << q;
    srow_hi[r] = w;
  }
  for (int sw = 0; sw < sweeps; ++sw) {
#pragma unroll
    for (int r = 0; r < kWorld - 1; ++r) {
      int32_t na = 0, nb = 0;
      if (myr == static_cast<uint32_t>(r)) {
#pragma unroll
        for (int q = 0; q < kWorld; ++q) {
          na = (uint32_t(q) == a) ? L[q] : na;
          nb = (uint32_t(q) == b) ? L[q] : nb;
        }
        if (n != 0u) {
          na -= sa;
          nb -= sb;
          uint32_t nd = 0u;
          int32_t xa = 0, xb = 0;
#pragma unroll
          for (int j = 0; j < kMaxM; ++j) {
            if (uint32_t(j) < n) {
              const int32_t c = cr[j];
              if (na <= nb) {
                na += c;
                xa += c;
              } else {
                nb += c;
                xb += c;
                nd |= 1u << j;
              }
            }
          }
          d = nd;
          sa = xa;
          sb = xb;
        }
      }
#pragma unroll
      for (int q = 0; q < kWorld; ++q) {
        const int sl = static_cast<int>(((srow[r] >> (4 * q)) & 15u) | (((srow_hi[r] >> q) & 1u) << 4));
        const int32_t v = (uint32_t(q) == a) ? na : nb;
        L[q] = __shfl_sync(kFull, v, sl);
      }
    }
  }
  uint32_t payload = 0u, h = 0u;
#pragma unroll
  for (int j = 0; j < 8; ++j) {
    const uint32_t e = static_cast<uint32_t>(lane) * 8u + uint32_t(j);
    const uint32_t rows = s.cnt[e];
    const uint32_t pinf = s.pinfo[e];
    const uint32_t pi = pinf & 0xffu;
    const uint32_t dpi = __shfl_sync(kFull, d, static_cast<int>(pi & 31u));
    uint32_t code = rows != 0u ? (s.tab[e] & 0xfu) : kNoHolder;
    if (rows != 0u && pi != kNoPair) {
      const uint32_t pab = s.pab[pi];
      code = ((dpi >> s.pos[e]) & 1u) ? ((pab >> 4) & 0xfu) : (pab & 0xfu);
    }
    payload |= code << (4 * j);
    const uint32_t y = ((e << 8) | code) * 0x9e3779b1u;
    h += y ^ (y >> 16);
  }
#pragma unroll
  for (int o = 16; o > 0; o >>= 1) h += __shfl_xor_sync(kFull, h, o);
  h ^= h >> 16;
  h *= 0x85ebca6bu;
  h ^= h >> 13;
  write(lane, (static_cast<unsigned long long>(tag32) << 32) | payload);
  if (lane == 0) write(32, (static_cast<unsigned long long>(tag32) << 32) | h);
}

// Consumer side (one full warp): poll the 32 assignment words of this step's tag (bounded like pw_count) and unpack the codes
// into out[256] (shared memory). Returns false on expiry (nothing written). load(i) makes ONE 64-bit load of word i.
template <class Load>
__device__ __forceinline__ bool pw_read(const Load& load, uint32_t tag32, uint8_t* out, uint32_t lane, uint32_t spin_rounds,
                                        uint32_t max_rounds, uint32_t backoff_ns) {
  unsigned long long v = 0ull;
  for (uint32_t round = 0;; ++round) {
    v = load(static_cast<int>(lane));
    const bool ok = static_cast<uint32_t>(v >> 32) == tag32;
    if (__all_sync(kFull, ok)) break;
    if (round >= max_rounds) return false;
    if (round >= spin_rounds) __nanosleep(backoff_ns);
  }
  const uint32_t payload = static_cast<uint32_t>(v);
#pragma unroll
  for (int j = 0; j < 8; ++j) out[lane * 8u + uint32_t(j)] = static_cast<uint8_t>((payload >> (4 * j)) & 0xfu);
  __syncwarp();
  return true;
}


// ------------------------------------------------------------------------------------------------ development kernels (tests)
// Tagged 64-bit word helpers (gpu scope: producer and consumer on one GPU).
__device__ __forceinline__ unsigned long long pw_ld(const unsigned long long* p) {
  unsigned long long v;
  asm volatile("ld.relaxed.gpu.global.b64 %0, [%1];" : "=l"(v) : "l"(p) : "memory");
  return v;
}
// The same load without a compiler memory barrier, so the poll loop keeps several in flight (tagged words carry their payload;
// nothing else is ordered by these loads).
__device__ __forceinline__ unsigned long long pw_ldx(const unsigned long long* p) {
  unsigned long long v;
  asm volatile("ld.relaxed.gpu.global.b64 %0, [%1];" : "=l"(v) : "l"(p));
  return v;
}
__device__ __forceinline__ void pw_st(unsigned long long* p, unsigned long long v) {
  asm volatile("st.relaxed.gpu.global.b64 [%0], %1;" ::"l"(p), "l"(v) : "memory");
}
#ifdef MUSE_VR_BAL_PW_STAMPS
#include "bal_pw_stamps.cuh"  // timing builds only (not part of the package)
#else
#define PW_STAMP(lv) ((void)0)
#endif
// The 32-bit tag of a step from its sequence word (written before the launch): seq mod (2^32 - 1) + 1, never 0.
__device__ __forceinline__ uint32_t pw_tag32(const unsigned long long* seq_word) {
  return static_cast<uint32_t>(*seq_word % 0xFFFFFFFFull) + 1u;
}

struct PwParams {
  const unsigned long long* ids;  // [M][8] tagged ids words (tag32 << 32 | packed)
  const unsigned long long* seq;  // [1] step sequence word
  const uint32_t* tbl;            // [kPwTableWords] static pair table
  unsigned long long* out;        // [grid][kPwWords] every CTA publishes into its own row
  uint32_t* err;                  // [1] sticky
  unsigned long long* stamps;     // timing builds: [grid][4] counting starts, counted, assigned, published
  int32_t M, sweeps, variant, maxm;
  uint32_t pass_rows, pass_inv, spin_rounds, max_rounds, backoff_ns;
};

// The assignment + publish of one CTA for the requested variant (0: shared-memory rounds, 1: register rounds, 2: the read
// form, pw_assign_publish; maxm = the largest pair's static member count, <= 16 for variant 2). All threads call it.
template <int kWorld, class Write>
__device__ __forceinline__ void pw_assign_any(int variant, int maxm, uint32_t pass_rows, uint32_t pass_inv, int sweeps,
                                              PwSmem& s, int tid, const Write& write, uint32_t tag32) {
  if (variant == 2) {
    if (maxm <= 8) {
      pw_assign_publish<kWorld, 8>(pass_rows, pass_inv, sweeps, s, tid, write, tag32);
    } else if (maxm <= 12) {
      pw_assign_publish<kWorld, 12>(pass_rows, pass_inv, sweeps, s, tid, write, tag32);
    } else {
      pw_assign_publish<kWorld, 16>(pass_rows, pass_inv, sweeps, s, tid, write, tag32);
    }
  } else {
    pw_assign<kWorld>(pass_rows, pass_inv, sweeps, s, tid, variant);
    if (tid < 32) pw_publish(write, tag32, s, static_cast<uint32_t>(tid));
  }
}

// Every CTA runs the role on the same words (determinism across CTAs is part of the tests).
template <int kWorld>
__global__ void __launch_bounds__(512, 1) bal_pw_kernel(const __grid_constant__ PwParams p) {
  __shared__ __align__(16) PwSmem s;
  const int tid = static_cast<int>(threadIdx.x), nt = static_cast<int>(blockDim.x);
  const uint32_t tag = pw_tag32(p.seq);
  unsigned long long* st = (p.stamps != nullptr && tid == 0) ? p.stamps + size_t(blockIdx.x) * 4 : nullptr;
  pw_stage(p.tbl, s, tid, nt);
  __syncthreads();
  if (st != nullptr) PW_STAMP(st[0]);
  const unsigned long long* ids = p.ids;
  auto ld = [ids](int row, int k) { return pw_ldx(ids + size_t(row) * kTopK + k); };
  pw_count(ld, tag, p.M, s, tid, nt, p.spin_rounds, p.max_rounds, p.backoff_ns);
  __syncthreads();
  if (st != nullptr) PW_STAMP(st[1]);
  const bool ok = (s.err & kErrWait) == 0u;
  if (ok) {
    unsigned long long* o = p.out + size_t(blockIdx.x) * kPwWords;
    auto wr = [o](int i, unsigned long long v) { pw_st(o + i, v); };
    pw_assign_any<kWorld>(p.variant, p.maxm, p.pass_rows, p.pass_inv, p.sweeps, s, tid, wr, tag);
  }
  if (tid < 32) {
    __syncwarp();
    if (st != nullptr) {
      PW_STAMP(st[2]);
      PW_STAMP(st[3]);
    }
  }
  if (tid == 0 && s.err != 0u) atomicOr(p.err, s.err);
}

// Consumer stand-in for the expert stage's prologue (tests and the development bench): every CTA polls all M x 8 ids words,
// counts the routed rows per local pool slot, and builds the exclusive bases of the local slots. mode 0: no assignment
// (home experts only); 1: reads the assignment words first and counts only experts assigned here; 2: counts every pool
// slot first, then reads the assignment and keeps the slots whose expert is assigned here. CTA 0 writes bases [pool + 1].
// Optional token edge (tok != nullptr): then the CTA waits for the first and the last word of each of tok_rows token rows
// (16-bit tag in bits 63..48), as the expert stage cannot start its first pass before its rows land.
struct PwConsumerParams {
  const unsigned long long* ids;  // [M][8]
  const unsigned long long* seq;  // [1]
  const uint32_t* tab;            // [256] placement words
  const unsigned long long* asg;  // [kPwWords] or nullptr (mode 0)
  int32_t* bases;                 // [pool + 1]
  uint32_t* err;                  // [1] sticky
  const unsigned long long* tok;  // [tok_rows][tok_pitch] or nullptr
  unsigned long long* stamps;     // timing builds: [grid][kPwConsumerStamps] start, ids counted, assignment read, bases, tokens
  int32_t M, rank, pool, mode, tok_rows, tok_pitch, tok_words;
  uint32_t spin_rounds, max_rounds, backoff_ns;
};
inline constexpr int kPwConsumerStamps = 5;

inline constexpr int kPwMaxPool = 128;

__global__ void __launch_bounds__(kPwThreads, 1) bal_pw_consumer_kernel(const __grid_constant__ PwConsumerParams p) {
  __shared__ uint32_t tab[kNumExperts];
  __shared__ uint32_t cnt[kPwMaxPool];
  __shared__ int32_t se[kPwMaxPool];
  __shared__ uint8_t asg[kNumExperts];
  __shared__ uint32_t wsum[kPwThreads / 32];
  __shared__ uint32_t serr;
  const int tid = static_cast<int>(threadIdx.x), nt = static_cast<int>(blockDim.x);
  const uint32_t lane = static_cast<uint32_t>(tid & 31), warp = static_cast<uint32_t>(tid >> 5);
  const uint32_t tag = pw_tag32(p.seq);
  const uint32_t rank = static_cast<uint32_t>(p.rank);
  unsigned long long* st = (p.stamps != nullptr && tid == 0) ? p.stamps + size_t(blockIdx.x) * kPwConsumerStamps : nullptr;
  for (int i = tid; i < kNumExperts; i += nt) tab[i] = p.tab[i];
  for (int i = tid; i < kPwMaxPool; i += nt) {
    cnt[i] = 0u;
    se[i] = -1;
  }
  if (tid == 0) serr = 0u;
  __syncthreads();
  for (int e = tid; e < kNumExperts; e += nt) {
    const uint32_t t = tab[e];
    if ((t & 0xfu) == rank) se[(t >> 8) & 0xffu] = e;
    if (((t >> 4) & 0xfu) == rank) se[(t >> 16) & 0xffu] = e;
  }
  if (st != nullptr) PW_STAMP(st[0]);
  const unsigned long long* asg_words = p.asg;
  const auto ld = [asg_words](int i) { return pw_ld(asg_words + i); };
  if (p.mode == 1) {
    if (warp == 0 && !pw_read(ld, tag, asg, lane, p.spin_rounds, p.max_rounds, p.backoff_ns) && lane == 0)
      atomicOr(&serr, kErrWait);
    __syncthreads();
    if (st != nullptr) PW_STAMP(st[2]);
  }
  const bool early_fail = (serr & kErrWait) != 0u;  // read after the barrier above (or the initial one); no later writer
  __syncthreads();
  // poll and count (up to 32 pending words per thread)
  {
    const int n = p.M * kTopK;
    uint32_t pending = 0u;
    int k = 0;
    for (int i = tid; i < n; i += nt, ++k) pending |= 1u << k;
    uint32_t err = 0u;
    for (uint32_t round = 0; pending != 0u && !early_fail; ++round) {
      int j = 0;
      for (int i = tid; i < n; i += nt, ++j) {
        if (((pending >> j) & 1u) == 0u) continue;
        const unsigned long long v = pw_ld(p.ids + i);
        if (static_cast<uint32_t>(v >> 32) != tag) continue;
        pending &= ~(1u << j);
        const uint32_t e = (static_cast<uint32_t>(v) >> 16) & 0xffffu;
        if (e >= uint32_t(kNumExperts)) continue;
        const uint32_t t = tab[e];
        const uint32_t h = t & 0xfu, sc = (t >> 4) & 0xfu;
        uint32_t slot = kNoPair;
        if (p.mode == 0) {
          if (h == rank) slot = (t >> 8) & 0xffu;
        } else if (p.mode == 1) {
          if (asg[e] == rank) slot = h == rank ? (t >> 8) & 0xffu : (t >> 16) & 0xffu;
        } else {
          if (h == rank) slot = (t >> 8) & 0xffu;
          else if (sc == rank) slot = (t >> 16) & 0xffu;
        }
        if (slot != kNoPair) atomicAdd(&cnt[slot], 1u);
      }
      if (pending == 0u) break;
      if (round >= p.max_rounds) {
        err |= kErrWait;
        break;
      }
      if (round >= p.spin_rounds) __nanosleep(p.backoff_ns);
    }
    if (err != 0u) atomicOr(&serr, err);
  }
  __syncthreads();
  if (st != nullptr) PW_STAMP(st[1]);
  if (p.mode == 2) {
    if (warp == 0 && !pw_read(ld, tag, asg, lane, p.spin_rounds, p.max_rounds, p.backoff_ns) && lane == 0)
      atomicOr(&serr, kErrWait);
    __syncthreads();
    if (st != nullptr) PW_STAMP(st[2]);
  }
  // exclusive bases over the local slots (pool <= 128: warps 0-3, one slot per thread)
  uint32_t v = 0u;
  if (tid < p.pool && se[tid] >= 0) {
    const uint32_t c = cnt[tid];
    const bool local = p.mode == 2 ? asg[se[tid]] == rank : true;
    v = local ? c : 0u;
  }
  uint32_t inc = v;
#pragma unroll
  for (int o = 1; o < 32; o <<= 1) {
    const uint32_t y = __shfl_up_sync(kFull, inc, o);
    if (lane >= uint32_t(o)) inc += y;
  }
  if (lane == 31u) wsum[warp] = inc;
  __syncthreads();
  uint32_t off = 0u;
  for (uint32_t w = 0; w < warp; ++w) off += wsum[w];
  if (blockIdx.x == 0 && tid < p.pool) p.bases[tid] = static_cast<int32_t>(off + inc - v);
  if (blockIdx.x == 0 && tid == p.pool - 1) p.bases[p.pool] = static_cast<int32_t>(off + inc);
  __syncthreads();
  if (st != nullptr) PW_STAMP(st[3]);
  if (p.tok != nullptr) {  // token edge: thread t < 2 * tok_rows polls one word (row t / 2, first or last)
    const uint32_t tag16 = static_cast<uint32_t>(*p.seq % 65535ull) + 1u;
    uint32_t err = 0u;
    for (int t = tid; t < 2 * p.tok_rows; t += nt) {
      const unsigned long long* w = p.tok + size_t(t >> 1) * p.tok_pitch + ((t & 1) ? p.tok_words - 1 : 0);
      for (uint32_t round = 0;; ++round) {
        if (static_cast<uint32_t>(pw_ld(w) >> 48) == tag16) break;
        if (round >= p.max_rounds) {
          err |= kErrWait;
          break;
        }
        if (round >= p.spin_rounds) __nanosleep(p.backoff_ns);
      }
    }
    if (err != 0u) atomicOr(&serr, err);
    __syncthreads();
    if (st != nullptr) PW_STAMP(st[4]);
  }
  __syncthreads();
  if (tid == 0 && serr != 0u) atomicOr(p.err, serr);
}

// Host entries (development). ids [M, 8] int64 tagged words; seq [1] int64; tbl [kPwTableWords] int32 (pw_table());
// out [grid, 33] int64; err [1] int32; stamps optional [grid * 4] int64.
struct BalPw {
  static void run(const tvm::ffi::TensorView ids, const tvm::ffi::TensorView seq, const tvm::ffi::TensorView tbl,
                  const tvm::ffi::TensorView out, const tvm::ffi::TensorView err,
                  tvm::ffi::Optional<tvm::ffi::TensorView> stamps, int64_t world, int64_t M, int64_t pass_rows,
                  int64_t sweeps, int64_t grid, int64_t threads, int64_t spin_rounds, int64_t max_rounds,
                  int64_t backoff_ns, int64_t variant, int64_t maxm) {
    using namespace sglang;
    using namespace sglang::host;
    RuntimeCheck(world == 4 || world == 8, "bal_pw: world must be 4 or 8");
    RuntimeCheck(M >= 1 && M <= kPwMaxRows && ids.numel() >= M * kTopK, "bal_pw: rows");
    RuntimeCheck(threads >= kPwThreads && threads % 32 == 0 && threads <= 512 && M * kTopK <= 32 * threads,
                 "bal_pw: threads");
    RuntimeCheck(tbl.numel() >= kPwTableWords, "bal_pw: pair table size");
    RuntimeCheck(grid >= 1 && grid <= 1024 && out.numel() >= grid * kPwWords, "bal_pw: out");
    RuntimeCheck(pass_rows >= 1 && pass_rows <= 128 && sweeps >= 0 && sweeps <= 8, "bal_pw: pass rows / sweeps");
    PwParams p{};
    p.ids = static_cast<const unsigned long long*>(ids.data_ptr());
    p.seq = static_cast<const unsigned long long*>(seq.data_ptr());
    p.tbl = static_cast<const uint32_t*>(tbl.data_ptr());
    p.out = static_cast<unsigned long long*>(out.data_ptr());
    p.err = static_cast<uint32_t*>(err.data_ptr());
    p.stamps = nullptr;
    if (stamps.has_value()) {
      RuntimeCheck(stamps.value().numel() >= grid * 4, "bal_pw: stamps [grid * 4]");
      p.stamps = static_cast<unsigned long long*>(stamps.value().data_ptr());
    }
    p.M = static_cast<int32_t>(M);
    p.sweeps = static_cast<int32_t>(sweeps);
    RuntimeCheck(variant >= 0 && variant <= 2 && maxm >= 1 && maxm <= kPwMaxMembers && (variant != 2 || maxm <= 16),
                 "bal_pw: variant / maxm");
    p.variant = static_cast<int32_t>(variant);
    p.maxm = static_cast<int32_t>(maxm);
    p.pass_rows = static_cast<uint32_t>(pass_rows);
    p.pass_inv = static_cast<uint32_t>((65536 + pass_rows - 1) / pass_rows);
    for (int64_t x = 0; x <= kPwMaxRows + pass_rows; ++x) {
      RuntimeCheck(((uint64_t(x) * p.pass_inv) >> 16) == uint64_t(x / pass_rows), "bal_pw: pass_rows reciprocal not exact");
    }
    p.spin_rounds = static_cast<uint32_t>(spin_rounds);
    p.max_rounds = static_cast<uint32_t>(max_rounds);
    p.backoff_ns = static_cast<uint32_t>(backoff_ns);
    const cudaStream_t stream = LaunchKernel::resolve_device(ids.device());
    if (world == 8)
      bal_pw_kernel<8><<<static_cast<unsigned>(grid), static_cast<unsigned>(threads), 0, stream>>>(p);
    else
      bal_pw_kernel<4><<<static_cast<unsigned>(grid), static_cast<unsigned>(threads), 0, stream>>>(p);
  }
};

struct BalPwConsumer {
  static void run(const tvm::ffi::TensorView ids, const tvm::ffi::TensorView seq, const tvm::ffi::TensorView tab,
                  const tvm::ffi::TensorView asg, const tvm::ffi::TensorView bases, const tvm::ffi::TensorView err,
                  tvm::ffi::Optional<tvm::ffi::TensorView> stamps, tvm::ffi::Optional<tvm::ffi::TensorView> tok,
                  int64_t tok_rows, int64_t tok_words, int64_t M, int64_t rank, int64_t pool, int64_t mode, int64_t grid,
                  int64_t spin_rounds, int64_t max_rounds, int64_t backoff_ns) {
    using namespace sglang;
    using namespace sglang::host;
    RuntimeCheck(M >= 1 && M <= kPwMaxRows && ids.numel() >= M * kTopK && M * kTopK <= 32 * kPwThreads, "consumer: rows");
    RuntimeCheck(pool >= 1 && pool <= kPwMaxPool && bases.numel() >= pool + 1, "consumer: pool");
    RuntimeCheck(mode >= 0 && mode <= 2 && (mode == 0 || asg.numel() >= kPwWords), "consumer: mode");
    RuntimeCheck(tab.numel() >= kNumExperts && grid >= 1 && grid <= 1024, "consumer: table / grid");
    PwConsumerParams p{};
    p.ids = static_cast<const unsigned long long*>(ids.data_ptr());
    p.seq = static_cast<const unsigned long long*>(seq.data_ptr());
    p.tab = static_cast<const uint32_t*>(tab.data_ptr());
    p.asg = static_cast<const unsigned long long*>(asg.data_ptr());
    p.bases = static_cast<int32_t*>(bases.data_ptr());
    p.err = static_cast<uint32_t*>(err.data_ptr());
    p.stamps = nullptr;
    if (stamps.has_value()) {
      RuntimeCheck(stamps.value().numel() >= grid * kPwConsumerStamps, "consumer: stamps [grid * 5]");
      p.stamps = static_cast<unsigned long long*>(stamps.value().data_ptr());
    }
    p.tok = nullptr;
    if (tok.has_value()) {
      RuntimeCheck(tok.value().dim() == 2 && tok.value().size(0) >= tok_rows && tok.value().size(1) >= tok_words &&
                       tok_rows >= 1 && tok_words >= 1,
                   "consumer: token board");
      p.tok = static_cast<const unsigned long long*>(tok.value().data_ptr());
      p.tok_rows = static_cast<int32_t>(tok_rows);
      p.tok_pitch = static_cast<int32_t>(tok.value().size(1));
      p.tok_words = static_cast<int32_t>(tok_words);
    }
    p.M = static_cast<int32_t>(M);
    p.rank = static_cast<int32_t>(rank);
    p.pool = static_cast<int32_t>(pool);
    p.mode = static_cast<int32_t>(mode);
    p.spin_rounds = static_cast<uint32_t>(spin_rounds);
    p.max_rounds = static_cast<uint32_t>(max_rounds);
    p.backoff_ns = static_cast<uint32_t>(backoff_ns);
    const cudaStream_t stream = LaunchKernel::resolve_device(ids.device());
    bal_pw_consumer_kernel<<<static_cast<unsigned>(grid), kPwThreads, 0, stream>>>(p);
  }
};

}  // namespace muse_vr::bal
