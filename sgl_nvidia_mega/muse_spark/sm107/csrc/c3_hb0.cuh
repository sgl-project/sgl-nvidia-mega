// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// Expert-output hop, route all-gather form (header only, device API). Default off: only kernels that include this header
// use it, and they are selected by a default-off knob.
//
// Rows. A pass has M tokens with 8 routed slots each. Slot k of token t is computed on exactly one rank of the
// tensor-parallel group (the rank holding the slot's expert). That rank stores the slot's output row (4096 bf16, as the
// expert kernel wrote it) into every OTHER rank's receive region at linear row t * 8 + k, so after the transfer every rank
// holds every slot of every token (its own slots in the expert kernel's output, the others in its receive region).
//
// Words. lb_hop format kTag64: each aligned 64-bit word carries three bf16 values and the transfer tag (bits 63..48);
// producers store word pairs with st.relaxed.sys.v2.b64, consumers accept a word once its own tag matches. Tile-aligned
// rows: each 128-column tile of a row owns three whole 128-byte lines (48 words): 22 six-value groups (21 full groups, then
// columns 126..127 and four zero pads) in words 0..43, and four pad words that are never stored or read. A producer that
// owns one 128-column tile of a row therefore writes whole lines of its own (one writer per line); a row is 32 tiles = 1536
// words (site width 4608 value slots). Column c: tile c / 128, group (c % 128) / 6, word 48 * tile + 2 * group +
// (c % 128 % 6) / 3, element (c % 128 % 6) % 3. The site uses policy kRing: the region is addressed by linear row
// (rows_max of the site spec = rows / W), slot reuse is ordered by credits.
//
// Memory model. Tagged words: payload and tag in one aligned 64-bit word, single-copy atomic for morally strong accesses
// of that width (PTX ISA, memory consistency model: vector accesses are modelled as separate scalar accesses). A tagged
// word is never read with TMA, .nc loads or another width and has exactly one producer thread. Every wait is bounded and
// sets the sticky error word on expiry; an expired wait never consumes.
#pragma once

#include <cstdint>

#include "lb_hop.cuh"

namespace muse_vr {
namespace hb0 {

namespace lh = muse_vr::lb_hop;

inline constexpr int kH = 4096;
inline constexpr int kTopK = 8;
inline constexpr uint32_t kTiles = 32;                     // 128-column tiles per row
inline constexpr uint32_t kTileGroups = 22;                // six-value groups per tile (the last one: 2 values + 4 pads)
inline constexpr uint32_t kTileWords = 48;                 // words per tile: 44 tagged + 4 never-used pads (3 lines)
inline constexpr uint32_t kGroups = kTiles * kTileGroups;  // 704 groups per row
inline constexpr uint32_t kRowWords = kTiles * kTileWords;  // 1536 words per row
inline constexpr uint32_t kWidth = 3u * kRowWords;          // site width 4608 (value slots, multiple of 6)

// 64-bit word w of linear row `lin` in rank `dst`'s copy `copy` of the site.
template <int W>
__device__ __forceinline__ unsigned long long* lin_word(const lh::Site<W>& s, uint32_t dst, uint32_t copy, uint32_t lin,
                                                        uint32_t w) {
  return s.recv[dst] + copy * s.copy_words + uint64_t(lin) * s.pitch + w;
}

// Group gi (0..703) of the tile-aligned layout: tile gi / 22, group gi % 22 within the tile.
__device__ __forceinline__ uint32_t group_col(uint32_t gi) { return (gi / kTileGroups) * 128u + (gi % kTileGroups) * 6u; }
__device__ __forceinline__ uint32_t group_word(uint32_t gi) { return (gi / kTileGroups) * kTileWords + 2u * (gi % kTileGroups); }
__device__ __forceinline__ uint32_t group_values(uint32_t gi) { return (gi % kTileGroups) + 1u < kTileGroups ? 6u : 2u; }

// The values of group gi of a row (the last group of a tile: two values and four zeros). The row is 4-byte aligned.
__device__ __forceinline__ void load6(const uint16_t* row, uint32_t gi, uint32_t (&e)[6]) {
  const uint32_t* p = reinterpret_cast<const uint32_t*>(row + group_col(gi));
  const bool full = group_values(gi) == 6u;
  const uint32_t a = p[0], b = full ? p[1] : 0u, c = full ? p[2] : 0u;
  e[0] = a & 0xffffu;
  e[1] = a >> 16;
  e[2] = b & 0xffffu;
  e[3] = b >> 16;
  e[4] = c & 0xffffu;
  e[5] = c >> 16;
}

// Producer, one warp: the 4096 bf16 of `row` (this rank's output of slot `lin` = t * 8 + k) into every other rank's copy.
// The caller has acquired the slot's credits (lh::credit_acquire, then a barrier) and ordered its reads of `row` after
// the row's writes (stream order or an acquire of the writer's release).
template <int W>
__device__ __forceinline__ void push_route_row(const lh::Site<W>& s, const lh::Xfer& x, uint32_t lin, const uint16_t* row,
                                               uint32_t lane, uint32_t gi_limit = kGroups) {
  for (uint32_t gi = lane; gi < gi_limit; gi += 32u) {  // gi_limit < kGroups: tests only (the row's later tiles never stored)
    uint32_t e[6];
    load6(row, gi, e);
    unsigned long long w0, w1;
    lh::words6(x, e, w0, w1);
#pragma unroll
    for (int r = 0; r < W; ++r)
      if (uint32_t(r) != s.rank) lh::st_relaxed_sys_v2_b64(lin_word(s, uint32_t(r), x.copy, lin, group_word(gi)), w0, w1);
  }
}

// Bounded poll of one word pair (tag check on both words). Returns 0 = accepted, 1 = rounds ran out or the error word was
// set elsewhere, 2 = a newer tag (the slot was overwritten before this read). Per thread (not warp-uniform).
__device__ __forceinline__ int poll_pair(const lh::Ctl& c, const unsigned long long* p, uint32_t tag, ulonglong2& v) {
  uint32_t round = 0;
  for (;;) {
    v = lh::ld_relaxed_sys_v2_b64(p);
    const uint32_t t0 = uint32_t(v.x >> 48), t1 = uint32_t(v.y >> 48);
    if (t0 == tag && t1 == tag) return 0;
    if (lh::tag_ahead(t0, tag) || lh::tag_ahead(t1, tag)) return 2;
    ++round;
    if (round > c.spin_rounds) {
      if ((round & 15u) == 0u && lh::failed(c)) return 1;
      if (round > c.max_rounds) return 1;
      __nanosleep(c.backoff_ns);
    }
  }
}

}  // namespace hb0
}  // namespace muse_vr
