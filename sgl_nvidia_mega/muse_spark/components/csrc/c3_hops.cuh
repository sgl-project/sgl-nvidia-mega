// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// Hop roles for the fused MoE-half launch and its split form (header only, device API). Built on the hop device API v2
// (lb_hop.cuh). Default off: only kernels that include this header use it, and they are selected by a default-off knob.
//
// Rows. A pass has M rows and a tensor-parallel group of W ranks. Rank r owns the contiguous rows [own0(r), own0(r + 1)),
// own0(r) = floor(M r / W).
//
// RS_A (reduce-scatter of a per-rank partial to the row owner). Each rank stores its partial of row `row` only into the owner's
// receive region of a unicast kTag64 site (push6_to). The owner reads every source and folds them in source-rank order 0..W-1
// in fp32 (first source's value, then one rounded add per source); the caller rounds the fold once to bf16.
//
// RS_B (per-token partial of the expert stage to the row owner). As RS_A, except that a rank stores a partial only for tokens
// with at least one local route; the owner polls only the expected sources (mask from the routing ids) and adds +0.0f at the
// rank position of every absent source, so the fold equals the fold over all W partials with +0.0 for the absent ones.
// Optional zero-sign parity with the engine's push all-reduce: see stock_zero_pairs below.
//
// AG_tok (all-gather of the owner's quantized token rows). Byte payloads in format kB48: each aligned 64-bit word carries 48
// payload bits (6 bytes, little-endian) and the transfer tag in bits 63..48. Producers store each word once through the site's
// multicast address (or once per rank); consumers accept each word on its own matching relaxed load and keep that payload
// (the poll_v2 rule of lb_hop.cuh); a row is complete when every word has matched.
//
// AG_x (all-gather of the owner's final rows into a plain bf16 buffer read by a library kernel). Plain 16-byte multicast stores
// (relaxed, system scope) into copy (layer & 1) of the landing buffer, then, after a CTA barrier, one thread per producer CTA
// raises its flag word (src, cta) to the forward-and-layer stamp with a releasing multicast max. The consumer tail acquires every
// (src, cta) flag before its grid completes; the library kernel reads after its dependency wait. Copy reuse two layers later
// is ordered by the same flags: a rank's reads of copy p (layer L) precede its own release of layer L + 1's flags, which every
// writer of layer L + 2 acquired in its layer L + 1 tail.
//
// Memory model. Tagged words: payload and tag in one aligned 64-bit word, single-copy atomic (PTX ISA, memory consistency
// model); a tagged word is never read with TMA, .nc loads or another width and has exactly one producer thread. Every wait is
// bounded (lb_hop Ctl rounds) and sets the sticky error word on expiry; an expired wait never consumes.
#pragma once

#include <cstdint>

#include "lb_hop.cuh"

namespace muse_vr {
namespace c3 {

namespace lh = muse_vr::lb_hop;

inline constexpr uint32_t kSiteBase = 0x30u;  // error-word site codes 0x30..0x3f
inline constexpr uint32_t kB48 = 2u;          // byte format: 48 payload bits + 16-bit tag per 64-bit word

// ------------------------------------------------------------------------------------------------------------ rows
__device__ __forceinline__ uint32_t own0(uint32_t M, uint32_t W, uint32_t r) {
  return uint32_t((uint64_t(M) * r) / W);
}
__device__ __forceinline__ uint32_t owner_of(uint32_t M, uint32_t W, uint32_t row) {
  uint32_t r = uint32_t((uint64_t(row) * W) / (M ? M : 1u));
  while (r + 1u < W && own0(M, W, r + 1u) <= row) ++r;
  while (r > 0u && own0(M, W, r) > row) --r;
  return r;
}

// ------------------------------------------------------------------------------------------------------------ RS_A / RS_B
// Zero-sign parity with the engine's push all-reduce (optional, per launch): that all-reduce reserves an all-zero 32-bit word
// as its empty-slot value, so it sends a bf16 pair (+0.0, +0.0) as (-0.0, +0.0). Applying the same rewrite to each aligned pair
// before the push (and using (-0.0, +0.0) for an absent source in the masked fold) keeps the owner fold byte-equal to that
// all-reduce when a column sums to zero. Pairs are (even column, odd column); col % 6 == 0 keeps them aligned.
__device__ __forceinline__ void stock_zero_pairs(uint32_t (&e)[6]) {
#pragma unroll
  for (int k = 0; k < 3; ++k)
    if ((e[2 * k] | e[2 * k + 1]) == 0u) e[2 * k] = 0x8000u;
}

// Producer: elements col..col+5 (col % 6 == 0, bf16 bits in the low halves of e) of row `row` to the row's owner.
template <int W>
__device__ __forceinline__ void rs_push6(const lh::Site<W>& s, const lh::Xfer& x, uint32_t M, uint32_t row, uint32_t col,
                                         const uint32_t (&e)[6]) {
  lh::push6_to<W>(s, x, owner_of(M, uint32_t(W), row), row, col, e);
}

// Owner (RS_A): fp32 rank-order folds of n <= N items; false = bounded wait expired (outputs must be poisoned).
template <int W, int N>
__device__ __forceinline__ bool rsA_collect6(const lh::Site<W>& s, const lh::Ctl& c, const lh::Xfer& x,
                                             const uint32_t (&row)[N], const uint32_t (&col)[N], int n, float (&sum)[N][6]) {
  ulonglong2 raw[N][W];
  return lh::collect6<W, N>(s, c, x, row, col, n, raw, sum);
}

// Owner (RS_B): polls only the sources in src_mask[i] (bit r = rank r stored a partial for item i's token); the fold runs over
// all ranks 0..W-1 in order with +0.0f at absent positions (with stock_zero: -0.0f in even columns, +0.0f in odd ones, the
// rewritten (+0.0, +0.0) pair of stock_zero_pairs). Warp-uniform like lb_hop's collects.
template <int W, int N>
__device__ __forceinline__ bool rsB_collect6(const lh::Site<W>& s, const lh::Ctl& c, const lh::Xfer& x,
                                             const uint32_t (&row)[N], const uint32_t (&col)[N], const uint32_t (&src_mask)[N],
                                             int n, float (&sum)[N][6], bool stock_zero = false) {
  static_assert(N * W <= 32, "at most 32 vectors per lane and round");
  const unsigned long long* a[N][W];
  ulonglong2 v[N][W];
  uint32_t pend = 0;
#pragma unroll
  for (int i = 0; i < N; ++i)
#pragma unroll
    for (int r = 0; r < W; ++r) {
      a[i][r] = lh::word_ptr<W>(s, s.rank, x.copy, uint32_t(r), i < n ? row[i] : row[0], (i < n ? col[i] : col[0]) / 3u);
      v[i][r] = make_ulonglong2(0ull, 0ull);
      if (i < n && (src_mask[i] >> r & 1u)) pend |= 1u << (i * W + r);
    }
  uint32_t round = 0;
  bool ok = true;
  while (__any_sync(0xffffffffu, pend != 0u)) {
#pragma unroll
    for (int k = 0; k < N * W; ++k)
      if (pend >> k & 1u) v[k / W][k % W] = lh::ld_relaxed_sys_v2_b64(a[k / W][k % W]);
#pragma unroll
    for (int k = 0; k < N * W; ++k)
      if ((pend >> k & 1u) && uint32_t(v[k / W][k % W].x >> 48) == x.tag && uint32_t(v[k / W][k % W].y >> 48) == x.tag)
        pend &= ~(1u << k);
    ++round;
    if (round > c.spin_rounds) {
      if ((round & 15u) == 0u && __any_sync(0xffffffffu, lh::failed(c))) { ok = false; break; }
      if (round > c.max_rounds) {
        if (pend) lh::fail(c, lh::kErrPayload, s.rank, s.site);
        ok = false;
        break;
      }
      __nanosleep(c.backoff_ns);
    }
  }
#pragma unroll
  for (int i = 0; i < N; ++i)
#pragma unroll
    for (int e = 0; e < 6; ++e) {
      float acc = 0.0f;
      const float absent = (stock_zero && (e & 1) == 0) ? -0.0f : 0.0f;  // col[i] is even, so e's parity is the column's
#pragma unroll
      for (int r = 0; r < W; ++r) {
        const float t = (i < n && (src_mask[i] >> r & 1u)) ? lh::bf16_bits_to_float(lh::elem6(v[i][r], e)) : absent;
        acc = (r == 0) ? t : __fadd_rn(acc, t);
      }
      sum[i][e] = acc;
    }
  return ok;
}

// ------------------------------------------------------------------------------------------------------------ AG_tok (kB48)
__host__ __device__ __forceinline__ unsigned long long b48_word(uint32_t tag, unsigned long long payload48) {
  return (static_cast<unsigned long long>(tag) << 48) | (payload48 & 0xFFFFFFFFFFFFull);
}
// Byte packing of a row of nbytes into kB48 words: word w carries bytes [6 w, 6 w + 6) little-endian, the tail zero-padded.
__host__ __device__ __forceinline__ unsigned long long b48_pack(const uint8_t* row_bytes, uint32_t nbytes, uint32_t w) {
  unsigned long long v = 0ull;
#pragma unroll
  for (int b = 0; b < 6; ++b) {
    const uint32_t i = 6u * w + uint32_t(b);
    if (i < nbytes) v |= static_cast<unsigned long long>(row_bytes[i]) << (8 * b);
  }
  return v;
}
__host__ __device__ __forceinline__ void b48_unpack(unsigned long long payload48, uint8_t* row_bytes, uint32_t nbytes, uint32_t w) {
#pragma unroll
  for (int b = 0; b < 6; ++b) {
    const uint32_t i = 6u * w + uint32_t(b);
    if (i < nbytes) row_bytes[i] = uint8_t(payload48 >> (8 * b));
  }
}
__host__ __device__ __forceinline__ uint32_t b48_words(uint32_t nbytes) { return (nbytes + 5u) / 6u; }
// Producer: word w of row `row` of this rank's block, to every rank (one multicast store when the site has one).
template <int W>
__device__ __forceinline__ void agTok_push_word(const lh::Site<W>& s, const lh::Xfer& x, uint32_t row, uint32_t w,
                                                unsigned long long payload48) {
  const unsigned long long v = b48_word(x.tag, payload48);
  if (s.mc != nullptr && !s.loop) {
    lh::st_mc_relaxed_sys_b64(s.mc + x.copy * s.copy_words + (uint64_t(s.rank) * s.rows_max + row) * s.pitch + w, v);
  } else {
#pragma unroll 1
    for (uint32_t r = 0; r < uint32_t(W); ++r) lh::st_relaxed_sys_u64(lh::word_ptr<W>(s, r, x.copy, s.rank, row, w), v);
  }
}
// Consumer (one warp): row `row` of source `src`, nw words (at most 32 * kMaxWordsPerLane). Each word is accepted on its OWN
// matching load (lb_hop poll_v2 rule): matched words are kept from that load and never re-read; unmatched words are re-read, all
// loads of a round issued before any check. out[k] = the 48-bit payloads; true when every word matched (warp-uniform, bounded).
template <int W, int kMaxWordsPerLane>
__device__ __forceinline__ bool agTok_read_row(const lh::Site<W>& s, const lh::Ctl& c, const lh::Xfer& x, uint32_t src,
                                               uint32_t row, uint32_t nw, unsigned long long (&out)[kMaxWordsPerLane]) {
  static_assert(kMaxWordsPerLane <= 32, "pend mask is 32 bits");
  const uint32_t lane = threadIdx.x & 31u;
  const unsigned long long* base = lh::word_ptr<W>(s, s.rank, x.copy, src, row, 0u);
  uint32_t pend = 0;
#pragma unroll
  for (int k = 0; k < kMaxWordsPerLane; ++k)
    if (lane + 32u * uint32_t(k) < nw) pend |= 1u << k;
  for (uint32_t round = 0; __any_sync(0xffffffffu, pend != 0u); ++round) {
    unsigned long long v[kMaxWordsPerLane];
#pragma unroll
    for (int k = 0; k < kMaxWordsPerLane; ++k)
      if (pend >> k & 1u) v[k] = lh::ld_relaxed_sys_u64(base + lane + 32u * uint32_t(k));
    bool over = false;
#pragma unroll
    for (int k = 0; k < kMaxWordsPerLane; ++k)
      if (pend >> k & 1u) {
        if (uint32_t(v[k] >> 48) == x.tag) {
          out[k] = v[k] & 0xFFFFFFFFFFFFull;
          pend &= ~(1u << k);
        } else {
          over = over || lh::tag_ahead(uint32_t(v[k] >> 48), x.tag);
        }
      }
    if (__any_sync(0xffffffffu, over)) {  // a newer forward's word: the slot was overwritten, waiting cannot help
      if (over) lh::fail(c, lh::kErrOverrun, s.rank, s.site);
      __syncwarp();
      return false;
    }
    if (round > c.spin_rounds) {
      if ((round & 15u) == 0u && __any_sync(0xffffffffu, lh::failed(c))) return false;
      if (round > c.max_rounds) {
        if (pend) lh::fail(c, lh::kErrPayload, s.rank, s.site);
        __syncwarp();
        return false;
      }
      __nanosleep(c.backoff_ns);
    }
  }
  return true;
}

// ------------------------------------------------------------------------------------------------------------ ids (kT32)
// The routing ids of an owned row travel ids-first on their own site: 8 aligned 64-bit words per row, a nonzero 32-bit tag in
// bits 63..32 and the producer's 32-bit packed value in bits 31..0 (the consumer's ids64 format, unchanged). Site layout as
// lb_hop ([copy][src][row][pitch], pitch >= 8 words); the 32-bit tag is derived on the device from the forward sequence word
// (never a kernel argument: graph replays repeat arguments), nonzero, unique per forward within 2^32 - 1 forwards.
__device__ __forceinline__ uint32_t ids_tag(const lh::Xfer& x) { return uint32_t(x.t % 0xFFFFFFFFull) + 1u; }
// True when the 32-bit ids tag t (0 = empty) is 1..3 forwards AHEAD of the awaited tag w on its (2^32 - 1)-tag cycle.
__device__ __forceinline__ bool ids_tag_ahead(uint32_t t, uint32_t w) {
  if (t == 0u) return false;
  const unsigned long long a = t - 1u, b = w - 1u, n = 0xFFFFFFFFull;
  const unsigned long long d = a >= b ? a - b : a + n - b;
  return d >= 1ull && d <= 3ull;
}
template <int W>
__device__ __forceinline__ void ids_push(const lh::Site<W>& s, const lh::Xfer& x, uint32_t row, uint32_t k, uint32_t packed) {
  const unsigned long long v = (static_cast<unsigned long long>(ids_tag(x)) << 32) | packed;
  if (s.mc != nullptr && !s.loop) {
    lh::st_mc_relaxed_sys_b64(s.mc + x.copy * s.copy_words + (uint64_t(s.rank) * s.rows_max + row) * s.pitch + k, v);
  } else {
#pragma unroll 1
    for (uint32_t r = 0; r < uint32_t(W); ++r) lh::st_relaxed_sys_u64(lh::word_ptr<W>(s, r, x.copy, s.rank, row, k), v);
  }
}
// Consumer (lanes 0..7 of a warp read the 8 words of one row; other lanes idle): true when all 8 carry tag32 in one pass;
// packed[k] = the low halves of exactly those loads (lane k returns its own word in *mine). Warp-uniform, bounded.
template <int W>
__device__ __forceinline__ bool ids_read_row(const lh::Site<W>& s, const lh::Ctl& c, const lh::Xfer& x, uint32_t src,
                                             uint32_t row, uint32_t* mine) {
  const uint32_t tag32 = ids_tag(x);
  const uint32_t lane = threadIdx.x & 31u;
  const unsigned long long* p = lh::word_ptr<W>(s, s.rank, x.copy, src, row, lane < 8u ? lane : 0u);
  for (uint32_t round = 0;; ++round) {
    bool ok = true, over = false;
    if (lane < 8u) {
      const unsigned long long v = lh::ld_relaxed_sys_u64(p);
      ok = uint32_t(v >> 32) == tag32;
      over = !ok && ids_tag_ahead(uint32_t(v >> 32), tag32);
      *mine = uint32_t(v);
    }
    if (__all_sync(0xffffffffu, ok)) return true;
    if (__any_sync(0xffffffffu, over)) {  // a newer forward's word: the slot was overwritten, waiting cannot help
      if (over) lh::fail(c, lh::kErrOverrun, s.rank, kSiteBase + 4u);
      __syncwarp();
      return false;
    }
    if (round > c.spin_rounds) {
      if ((round & 15u) == 0u && __any_sync(0xffffffffu, lh::failed(c))) return false;
      if (round > c.max_rounds) {
        if (lane == 0) lh::fail(c, lh::kErrPayload, s.rank, kSiteBase + 4u);
        __syncwarp();
        return false;
      }
      __nanosleep(c.backoff_ns);
    }
  }
}

// ------------------------------------------------------------------------------------------------------------ AG_x
struct AgX {
  unsigned char* mc;            // multicast base of the landing buffer: 2 copies x [M_max][row_bytes]
  unsigned char* uc;            // this rank's unicast view of the same buffer (what the library kernel reads)
  unsigned long long* flag_mc;  // multicast base of the flag words: [W][ctas_max] (s64 max, monotonic stamps)
  const unsigned long long* flag_uc;  // this rank's unicast view of the flags
  unsigned long long* last;          // this rank: the last stamp each CTA published [ctas_max] (missed-advance guard)
  uint32_t row_bytes, rows_max, ctas_max, world, rank;
};
__device__ __forceinline__ void agx_st16(const AgX& g, uint32_t copy, uint32_t row, uint32_t byte, uint4 v) {
  unsigned char* p = g.mc + (uint64_t(copy) * g.rows_max + row) * g.row_bytes + byte;
  asm volatile("multimem.st.relaxed.sys.global.v4.f32 [%0], {%1, %2, %3, %4};" ::"l"(p), "r"(v.x), "r"(v.y), "r"(v.z),
               "r"(v.w)
               : "memory");
}
// After the CTA's last agx_st16 and a CTA barrier, one thread: releasing multicast max of this (src, cta) flag to `stamp`.
__device__ __forceinline__ void agx_publish(const AgX& g, uint32_t cta, unsigned long long stamp) {
  unsigned long long* p = g.flag_mc + uint64_t(g.rank) * g.ctas_max + cta;
  asm volatile("multimem.red.release.sys.global.max.s64 [%0], %1;" ::"l"(p), "l"(stamp) : "memory");
}
// Tail (one warp): acquires every (src, cta < nctas) flag >= stamp. Warp-uniform; false = bounded wait expired.
__device__ __forceinline__ bool agx_wait_all(const AgX& g, const lh::Ctl& c, uint32_t nctas, unsigned long long stamp) {
  const uint32_t lane = threadIdx.x & 31u, total = g.world * nctas;
  for (uint32_t i0 = 0; i0 < total; i0 += 32u) {
    const uint32_t i = i0 + lane;
    for (uint32_t round = 0;; ++round) {
      bool ok = true;
      if (i < total) {
        unsigned long long v;
        asm volatile("ld.acquire.sys.global.u64 %0, [%1];"
                     : "=l"(v)
                     : "l"(g.flag_uc + uint64_t(i / nctas) * g.ctas_max + (i % nctas))
                     : "memory");
        ok = static_cast<long long>(v) >= static_cast<long long>(stamp);
      }
      if (__all_sync(0xffffffffu, ok)) break;
      if (round > c.spin_rounds) {
        if ((round & 15u) == 0u && __any_sync(0xffffffffu, lh::failed(c))) return false;
        if (round > c.max_rounds) {
          if (lane == 0) lh::fail(c, lh::kErrPayload, g.rank, kSiteBase + 3u);
          __syncwarp();
          return false;
        }
        __nanosleep(c.backoff_ns);
      }
    }
  }
  // The landing data was written through the multicast alias and is read through the unicast alias (PTX ISA: virtually
  // aliased addresses; fence.proxy.alias orders accesses made through aliases and composes with the acquire above).
  asm volatile("fence.proxy.alias;" ::: "memory");
  return true;
}

}  // namespace c3
}  // namespace muse_vr
