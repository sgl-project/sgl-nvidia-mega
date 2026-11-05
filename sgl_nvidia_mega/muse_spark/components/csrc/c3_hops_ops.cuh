// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// Launch wrappers for the hop roles of c3_hops.cuh: the owner-row reduce-scatter as one persistent launch (every CTA first
// pushes its share of this rank's rows to their owners, then folds the owned rows it serves), plain or source-masked.
// Default off: used only through ops/c3_hops.py, which callers select with a default-off knob.
#pragma once

#include "lb_hop_ops.cuh"  // make_site / make_ctl / make_tok

#include <cuda_bf16.h>

#include <type_traits>

#include "c3_hops.cuh"

// Read-build hook: a bench may define C3_STAMP(j) before including this header to record per-CTA phase times; in every other
// build it expands to nothing.
#ifndef C3_STAMP
#define C3_STAMP(j)
#endif


namespace sglang {
namespace muse_vr_c3_ops {

namespace lh = muse_vr::lb_hop;
namespace c3 = muse_vr::c3;

inline constexpr uint32_t kStepRelease = 1u;  // this launch is the forward's first hop kernel: release the step token
inline constexpr uint32_t kStepAcquire = 2u;  // acquire the forward's step tokens before storing hop words
inline constexpr uint32_t kCollectOnly = 4u;  // skip the push phase (the partials were pushed by another kernel)
inline constexpr uint32_t kStockZeroSign = 8u;  // zero-sign parity with the engine's push all-reduce (c3::stock_zero_pairs)
inline constexpr uint32_t kFlatGrid = 16u;      // host-side launch choice: c3_rs_rows_flat_kernel (never read on the device)
// Early dependent launch: trigger the PDL dependents at entry instead of after the last store (their griddepcontrol.wait still
// waits for this grid's completion and memory flush), and prefer the max shared-memory carveout so the dependents' CTAs can be
// co-resident (a carveout mismatch keeps them off this grid's SMs). The trigger fires only once every CTA of this grid executed
// it, so all hop CTAs are resident before any dependent CTA launches.
inline constexpr uint32_t kEarlyTrigger = 32u;
inline constexpr uint32_t kSplitRoles = 64u;        // flat rs_rows / tagged AG: half the warps push, half fold / gather from the start
inline constexpr uint32_t kPushBeforeGuard = 128u;  // flat rs_rows / tagged AG without STEP_ACQUIRE: push before the guard barrier
inline constexpr uint32_t kRowFold = 256u;          // flat rs_rows: fold each owned row inside one CTA (row epilogue capable)

// Grid: at most one resident wave (the caller sizes it), so every CTA pushes before any CTA waits on another rank.
// part: [M, H] bf16 (this rank's partial); out: [own_n, H] bf16 (owned rows, bf16 of the fp32 rank-order fold);
// skip: [M] u8 or null (1 = no partial for this row on this rank); mask: [M] i32 or null (expected source ranks per row).
template <int W, bool kUsePDL>
__global__ void __launch_bounds__(256) c3_rs_rows_kernel(const __grid_constant__ lh::Site<W> s, const __grid_constant__ lh::Ctl c,
                                                         const __grid_constant__ lh::StepTok<W> tok, uint32_t step_flags,
                                                         const __nv_bfloat16* __restrict__ part, __nv_bfloat16* __restrict__ out,
                                                         int M, int H, const uint8_t* __restrict__ skip,
                                                         const int32_t* __restrict__ mask) {
  const int Hp = int(s.width);
  device::PDLWaitPrimary<kUsePDL>();
  if (step_flags & kEarlyTrigger) device::PDLTriggerSecondary<kUsePDL>();
  const lh::Xfer x = lh::xfer<W>(s);
  __shared__ int go_smem;
  if (threadIdx.x < 32) {  // warp 0: the guard (one thread), then the warp-collective step-token acquire
    int go = 0;
    if (threadIdx.x == 0) {
      go = lh::guard_enter<W>(s, c, blockIdx.x, x.seq) ? 1 : 0;
      if (go && (step_flags & kStepRelease) && blockIdx.x == 0) lh::step_release<W>(tok, x.seq);
    }
    go = __shfl_sync(0xffffffffu, go, 0);
    if (go && (step_flags & kStepAcquire)) go = lh::step_acquire<W>(tok, c, x.seq) ? 1 : 0;
    if (threadIdx.x == 0) go_smem = go;
  }
  __syncthreads();
  const bool go = go_smem != 0;
  // phase 1: push this rank's rows, one row per CTA iteration, 6-column groups per thread
  if (go && !(step_flags & kCollectOnly)) {
    for (int row = blockIdx.x; row < M; row += gridDim.x) {
      if (skip != nullptr && skip[row]) continue;
      const uint16_t* p = reinterpret_cast<const uint16_t*>(part) + size_t(row) * H;
      const uint32_t* p32 = reinterpret_cast<const uint32_t*>(p);  // H is even: 6-element groups start 4-byte aligned
      for (int g = threadIdx.x; g * 6 < Hp; g += blockDim.x) {
        uint32_t e[6];
        if (g * 6 + 6 <= H) {
          const uint32_t a = p32[g * 3], b = p32[g * 3 + 1], d = p32[g * 3 + 2];
          e[0] = a & 0xffffu; e[1] = a >> 16; e[2] = b & 0xffffu; e[3] = b >> 16; e[4] = d & 0xffffu; e[5] = d >> 16;
        } else {
#pragma unroll
          for (int j = 0; j < 6; ++j) e[j] = (g * 6 + j < H) ? uint32_t(p[g * 6 + j]) : 0u;
        }
        if (step_flags & kStockZeroSign) c3::stock_zero_pairs(e);
        c3::rs_push6<W>(s, x, uint32_t(M), uint32_t(row), uint32_t(g * 6), e);
      }
    }
  }
  // phase 2: fold the owned rows, one warp per row; each lane polls N 6-column groups per round (N x W vectors in flight)
  constexpr int N = (W >= 16) ? 1 : 16 / W;
  const uint32_t own0 = c3::own0(uint32_t(M), uint32_t(W), s.rank), own1 = c3::own0(uint32_t(M), uint32_t(W), s.rank + 1u);
  const uint32_t lane = threadIdx.x & 31u, wpc = blockDim.x >> 5, ngroups = uint32_t(Hp / 6);
  for (uint32_t row = own0 + blockIdx.x * wpc + (threadIdx.x >> 5); row < own1; row += gridDim.x * wpc) {
    uint16_t* o = reinterpret_cast<uint16_t*>(out) + size_t(row - own0) * H;
    for (uint32_t g0 = 0; g0 < ngroups; g0 += 32u * N) {
      uint32_t rows[N], cols[N], m[N];
      int n = 0;
#pragma unroll
      for (int i = 0; i < N; ++i) {
        const uint32_t g = g0 + lane + 32u * uint32_t(i);
        rows[i] = row;
        cols[i] = (g < ngroups ? g : 0u) * 6u;
        m[i] = mask != nullptr ? uint32_t(mask[row]) : 0u;
        if (g < ngroups) n = i + 1;
      }
      float sum[N][6];
      bool ok = go;
      if (go) {
        ok = (mask == nullptr) ? c3::rsA_collect6<W, N>(s, c, x, rows, cols, n, sum)
                               : c3::rsB_collect6<W, N>(s, c, x, rows, cols, m, n, sum, (step_flags & kStockZeroSign) != 0u);
      }
#pragma unroll
      for (int i = 0; i < N; ++i) {
        if (i >= n) break;
        const int gcol = int(cols[i]);
#pragma unroll
        for (int j = 0; j < 6; ++j) {
          const int col = gcol + j;
          if (col < H) o[col] = ok ? __bfloat16_as_ushort(__float2bfloat16_rn(sum[i][j])) : lh::kPoisonBf16;
        }
      }
    }
  }
  __syncthreads();  // ends this CTA's reads of the site
  if (threadIdx.x == 0 && go) lh::guard_exit<W>(s, blockIdx.x, x.seq);
  if (!(step_flags & kEarlyTrigger)) device::PDLTriggerSecondary<kUsePDL>();
}

// Row epilogue of the row-owned fold (kRowFold): a type with `static constexpr bool kActive` and
// `__device__ void row(uint32_t owned_row, const __nv_bfloat16* row_ptr, int H) const`, called by every thread of the CTA after a
// CTA barrier that follows the CTA's stores of that owned row into `out` (and followed by another CTA barrier). A composing
// build instantiates c3_rs_rows_flat_kernel with its own type (e.g. an owner-row norm role); the default does nothing.
// Optional `static constexpr int kMinCtasPerSm` (1 or 2, default 2) sets the kernel's __launch_bounds__ minimum CTAs per SM:
// 2 caps registers at 128 per thread, 1 lifts the cap for a heavier epilogue (launch at most one CTA per SM then).
struct NoRowEpilogue {
  static constexpr bool kActive = false;
  __device__ __forceinline__ void row(uint32_t, const __nv_bfloat16*, int) const {}
};

template <class E, class = void>
struct EpiMinCtas {
  static constexpr int value = 2;
};
template <class E>
struct EpiMinCtas<E, std::void_t<decltype(E::kMinCtasPerSm)>> {
  static constexpr int value = E::kMinCtasPerSm;
};

// Row-fold consumer (optional `static constexpr bool kFoldsRow`, default false; unmasked, own-local RS_A only): the epilogue
// folds each owned row itself, in the element layout it consumes, through RowFold::fold8, and writes its own outputs (and,
// if it wants, the fold's bf16 row into `out`); the kernel then skips its own fold and calls
// `epi.fold_row(owned_row, row, rf, out_row)` once per owned row with every thread of the CTA, followed by a CTA barrier.
template <class E, class = void>
struct EpiFoldsRow {
  static constexpr bool value = false;
};
template <class E>
struct EpiFoldsRow<E, std::void_t<decltype(E::kFoldsRow)>> {
  static constexpr bool value = E::kFoldsRow;
};

inline constexpr int kPushBatch = 4;  // push items per thread whose loads are in flight together

// Elements g*6 .. g*6+5 of row `row` of a [*, H] bf16 matrix as 16-bit values in e (zero past H; nothing loaded when !valid).
__device__ __forceinline__ void load_group6(const __nv_bfloat16* __restrict__ m, int H, uint32_t row, uint32_t g, bool valid,
                                            uint32_t (&e)[6]) {
  const uint16_t* p = reinterpret_cast<const uint16_t*>(m) + size_t(row) * H;
  if (valid && g * 6u + 6u <= uint32_t(H)) {
    const uint32_t* p32 = reinterpret_cast<const uint32_t*>(p);  // H is even: groups start 4-byte aligned
    const uint32_t a = p32[g * 3u], b = p32[g * 3u + 1u], d = p32[g * 3u + 2u];
    e[0] = a & 0xffffu; e[1] = a >> 16; e[2] = b & 0xffffu; e[3] = b >> 16; e[4] = d & 0xffffu; e[5] = d >> 16;
  } else {
#pragma unroll
    for (int j = 0; j < 6; ++j) e[j] = (valid && g * 6u + uint32_t(j) < uint32_t(H)) ? uint32_t(p[g * 6u + uint32_t(j)]) : 0u;
  }
}

// Writes 6 bf16 values (16-bit in e, or the poison value when !ok) at row-major [*, H] position (row, col .. col + 5) as
// 32-bit pairs (col even, H even).
__device__ __forceinline__ void store_group6(__nv_bfloat16* __restrict__ m, int H, uint32_t row, uint32_t col,
                                             const uint32_t (&e)[6], bool ok) {
  uint32_t* o32 = reinterpret_cast<uint32_t*>(m + size_t(row) * H);
#pragma unroll
  for (int j = 0; j < 6; j += 2)
    if (col + uint32_t(j) < uint32_t(H)) {
      const uint32_t lo = ok ? e[j] : lh::kPoisonBf16, hi = ok ? e[j + 1] : lh::kPoisonBf16;
      o32[(col + uint32_t(j)) >> 1] = (lo & 0xffffu) | (hi << 16);
    }
}

// kTag64 push of elements col..col+5 of row `row` of this rank's block to the OTHER ranks only (the caller keeps its own copy):
// at W > 2, one multicast store pair when the site table has a multicast address (it also lands in this rank's copy, unused);
// otherwise one store per peer (at W = 2 the single unicast store measured faster than the multicast pair at large M).
template <int W>
__device__ __forceinline__ void push6_peers(const lh::Site<W>& s, const lh::Xfer& x, uint32_t row, uint32_t col,
                                            const uint32_t (&e)[6]) {
  if (W > 2 && s.mc != nullptr && !s.loop) {
    lh::push6<W>(s, x, row, col, e);
    return;
  }
  const unsigned long long tag = static_cast<unsigned long long>(x.tag) << 48;
  const unsigned long long w0 = tag | (static_cast<unsigned long long>(e[2] & 0xffffu) << 32) |
                                (static_cast<unsigned long long>(e[1] & 0xffffu) << 16) | (e[0] & 0xffffu);
  const unsigned long long w1 = tag | (static_cast<unsigned long long>(e[5] & 0xffffu) << 32) |
                                (static_cast<unsigned long long>(e[4] & 0xffffu) << 16) | (e[3] & 0xffffu);
#pragma unroll
  for (int r = 0; r < W; ++r)
    if (uint32_t(r) != s.rank) lh::st_relaxed_sys_v2_b64(lh::word_ptr<W>(s, uint32_t(r), x.copy, s.rank, row, col / 3u), w0, w1);
}

// Owner fold with the owner's OWN partial read locally (never pushed to itself): items i < n of rows[i] / cols[i]; the other
// ranks' words are polled from this rank's receive region (all of them, or those in mask m[i] when masked), the own 6 values
// come from `part` (written by the predecessor before this grid's dependency wait). Rank-order fp32 fold, absent sources as in
// rsB_collect6 (+0.0, or the stock zero-sign pair when sz). Warp-uniform; false = bounded wait expired.
template <int W, int N>
__device__ __forceinline__ bool fold_own_local(const lh::Site<W>& s, const lh::Ctl& c, const lh::Xfer& x, const uint32_t (&row)[N],
                                               const uint32_t (&col)[N], const uint32_t (&m)[N], int n, bool masked,
                                               const __nv_bfloat16* __restrict__ part, int H, bool sz, float (&sum)[N][6]) {
  constexpr int R = (W > 1) ? W - 1 : 1;  // remote sources per item
  static_assert(N * R <= 32, "at most 32 vectors per lane and round");
  ulonglong2 v[N][R];
  uint32_t own[N][6];
  uint32_t pend = 0;
#pragma unroll
  for (int i = 0; i < N; ++i) {
    const bool self_in = i < n && (!masked || (m[i] >> s.rank & 1u));
    load_group6(part, H, row[i], col[i] / 6u, self_in, own[i]);
    if (self_in && sz) c3::stock_zero_pairs(own[i]);
#pragma unroll
    for (int j = 0; j < R; ++j) {
      const uint32_t r = uint32_t(j) < s.rank ? uint32_t(j) : uint32_t(j) + 1u;
      v[i][j] = make_ulonglong2(0ull, 0ull);
      if (W > 1 && i < n && (!masked || (m[i] >> r & 1u))) pend |= 1u << (i * R + j);
    }
  }
  uint32_t round = 0;
  bool ok = true;
  while (__any_sync(0xffffffffu, pend != 0u)) {
#pragma unroll
    for (int k = 0; k < N * R; ++k)
      if (pend >> k & 1u) {
        const int i = k / R, j = k % R;
        const uint32_t r = uint32_t(j) < s.rank ? uint32_t(j) : uint32_t(j) + 1u;
        v[i][j] = lh::ld_relaxed_sys_v2_b64(lh::word_ptr<W>(s, s.rank, x.copy, r, row[i], col[i] / 3u));
      }
#pragma unroll
    for (int k = 0; k < N * R; ++k)
      if ((pend >> k & 1u) && uint32_t(v[k / R][k % R].x >> 48) == x.tag && uint32_t(v[k / R][k % R].y >> 48) == x.tag)
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
      const float absent = (sz && (e & 1) == 0) ? -0.0f : 0.0f;
      float acc = 0.0f;
#pragma unroll
      for (int r = 0; r < W; ++r) {
        const bool in = i < n && (!masked || (m[i] >> r & 1u));
        const int j = r < int(s.rank) ? r : r - 1;
        const uint32_t bits = uint32_t(r) == s.rank ? own[i][e] : lh::elem6(v[i][j < 0 ? 0 : (j >= R ? R - 1 : j)], e);
        const float t = in ? lh::bf16_bits_to_float(bits) : absent;
        acc = (r == 0) ? t : __fadd_rn(acc, t);
      }
      sum[i][e] = acc;
    }
  return ok;
}

// The owner fold of RS_A in 8-column packs, for a row-fold consumer (EpiFoldsRow): fold8<P> returns the fp32 rank-order sums
// of columns c0[p] .. c0[p] + 7 (c0 % 8 == 0, c0 + 8 <= H) of row `row` - this rank's own partial read locally (stock zero
// pairs when sz), every other rank's two 6-column groups that cover the 8 columns polled from this rank's receive region,
// each word accepted on a matching tag - with the arithmetic of fold_own_local (first source's value, then one rounded fp32
// add per source in rank order), so the bf16 rounding of a sum equals the kernel's own fold output for that column.
// false = this launch's guard failed or the bounded wait expired (the consumer poisons its outputs). Warp-uniform polls.
template <int W>
struct RowFold {
  const lh::Site<W>& s;  // the kernel's __grid_constant__ parameters (a by-value copy would put the per-rank tables in
  const lh::Ctl& c;      // local memory, since fold8 indexes them by a runtime rank)
  lh::Xfer x;
  const __nv_bfloat16* part;
  int H;
  bool sz, go;

  template <int P>
  __device__ __forceinline__ bool fold8(uint32_t row, const uint32_t (&c0)[P], float (&sum)[P][8]) const {
    constexpr int R = (W > 1) ? W - 1 : 1;  // remote sources
    static_assert(P * R * 2 <= 32, "at most 32 vectors per lane and round");
    uint32_t own[P][4];  // packed bf16 pairs (even column low): 4 words per pack
    ulonglong2 v[P][R][2];
    uint32_t pend = 0;
#pragma unroll
    for (int p = 0; p < P; ++p) {
      const uint4 q = *reinterpret_cast<const uint4*>(part + size_t(row) * size_t(H) + c0[p]);
      own[p][0] = q.x;
      own[p][1] = q.y;
      own[p][2] = q.z;
      own[p][3] = q.w;
#pragma unroll
      for (int k = 0; k < 4; ++k)
        if (sz && own[p][k] == 0u) own[p][k] = 0x8000u;  // stock_zero_pairs: (+0, +0) -> (-0, +0), same aligned pairs
#pragma unroll
      for (int j = 0; j < R; ++j)
#pragma unroll
        for (int h = 0; h < 2; ++h) {
          v[p][j][h] = make_ulonglong2(0ull, 0ull);
          if (W > 1) pend |= 1u << ((p * R + j) * 2 + h);
        }
    }
    if (!go) return false;
    uint32_t round = 0;
    bool ok = true;
    while (__any_sync(0xffffffffu, pend != 0u)) {
#pragma unroll
      for (int k = 0; k < P * R * 2; ++k)
        if (pend >> k & 1u) {
          const int p = k / (R * 2), j = (k / 2) % R, h = k % 2;
          const uint32_t r = uint32_t(j) < s.rank ? uint32_t(j) : uint32_t(j) + 1u;
          v[p][j][h] = lh::ld_relaxed_sys_v2_b64(lh::word_ptr<W>(s, s.rank, x.copy, r, row, (c0[p] / 6u + uint32_t(h)) * 2u));
        }
#pragma unroll
      for (int k = 0; k < P * R * 2; ++k) {
        const ulonglong2& w = v[k / (R * 2)][(k / 2) % R][k % 2];
        if ((pend >> k & 1u) && uint32_t(w.x >> 48) == x.tag && uint32_t(w.y >> 48) == x.tag) pend &= ~(1u << k);
      }
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
    for (int p = 0; p < P; ++p) {
      const uint32_t g0 = c0[p] / 6u;
#pragma unroll
      for (int e = 0; e < 8; ++e) {
        // column c0 + e sits in group g0 (h = 0) or g0 + 1 (h = 1); c0 % 6 is a runtime value, so the group is picked with a
        // value select (no runtime index into the register arrays)
        const uint32_t col = c0[p] + uint32_t(e);
        const bool h = (col / 6u) != g0;
        const int k = int(col - (g0 + (h ? 1u : 0u)) * 6u);
        float acc = 0.0f;
#pragma unroll
        for (int r = 0; r < W; ++r) {
          uint32_t bits = (own[p][e >> 1] >> (16 * (e & 1))) & 0xffffu;
          if (uint32_t(r) != s.rank) {
            const int j = r < int(s.rank) ? r : r - 1;  // runtime (rank): picked by an unrolled select, not an index
            ulonglong2 w = h ? v[p][0][1] : v[p][0][0];
#pragma unroll
            for (int q = 1; q < R; ++q)
              if (q == j) w = h ? v[p][q][1] : v[p][q][0];
            bits = lh::elem6(w, k);
          }
          const float t = lh::bf16_bits_to_float(bits);
          acc = (r == 0) ? t : __fadd_rn(acc, t);
        }
        sum[p][e] = acc;
      }
    }
    return ok;
  }
};

// Flat-grid form of c3_rs_rows_kernel (same protocol, same arithmetic, same arguments): the push spreads the (row, 6-column
// group) items of the partial over every thread of the grid, and the fold spreads the owned (row, group) items over every
// warp of the grid in chunks of 32 x N consecutive items, so all CTAs take part in both phases at any M.
template <int W, bool kUsePDL, class Epi = NoRowEpilogue>
__global__ void __launch_bounds__(256, EpiMinCtas<Epi>::value) c3_rs_rows_flat_kernel(const __grid_constant__ lh::Site<W> s,
                                                              const __grid_constant__ lh::Ctl c,
                                                              const __grid_constant__ lh::StepTok<W> tok, uint32_t step_flags,
                                                              const __nv_bfloat16* __restrict__ part,
                                                              __nv_bfloat16* __restrict__ out, int M, int H,
                                                              const uint8_t* __restrict__ skip,
                                                              const int32_t* __restrict__ mask,
                                                              const __grid_constant__ Epi epi) {
  const int Hp = int(s.width);
  device::PDLWaitPrimary<kUsePDL>();
  C3_STAMP(0);
  if (step_flags & kEarlyTrigger) device::PDLTriggerSecondary<kUsePDL>();
  const uint32_t ngroups = uint32_t(Hp / 6);
  const uint32_t own0 = c3::own0(uint32_t(M), uint32_t(W), s.rank), own1 = c3::own0(uint32_t(M), uint32_t(W), s.rank + 1u);
  const uint32_t own_n = own1 - own0;
  // roles: with kSplitRoles the first half of each CTA's warps push and the second half fold from the start (push and fold
  // overlap); otherwise every warp pushes, then folds
  const uint32_t wpc = blockDim.x >> 5, warp = threadIdx.x >> 5;
  const bool split = (step_flags & kSplitRoles) != 0u;
  const uint32_t pw = split ? wpc / 2u : wpc, fw = split ? wpc - pw : wpc;
  const bool pusher = warp < pw, folder = !split || !pusher;
  const bool own_local = !(step_flags & kCollectOnly);
  const bool do_push = own_local && pusher;
  // push items: the (row, group) items of the OTHER ranks' rows only, kPushBatch per thread per round with every item's loads
  // in flight before the round's first push; the first round's loads are issued before the guard / step-token barrier
  const uint32_t ptotal = (uint32_t(M) - own_n) * ngroups, pstride = gridDim.x * pw * 32u;
  uint32_t it0 = blockIdx.x * pw * 32u + threadIdx.x;
  uint32_t e[kPushBatch][6], rowu[kPushBatch], colu[kPushBatch];
  bool vu[kPushBatch];
  auto load_batch = [&](uint32_t b) {
#pragma unroll
    for (int u = 0; u < kPushBatch; ++u) {
      const uint32_t it = b + uint32_t(u) * pstride;
      vu[u] = it < ptotal;
      const uint32_t rl = vu[u] ? it / ngroups : 0u, g = vu[u] ? it - rl * ngroups : 0u;
      rowu[u] = rl < own0 ? rl : rl + own_n;
      colu[u] = g * 6u;
      load_group6(part, H, rowu[u], g, vu[u], e[u]);
    }
  };
  if (do_push) load_batch(it0);
  const lh::Xfer x = lh::xfer<W>(s);
  // without a step-token acquire the pushes need only the tag (x): they start before thread 0's guard, which gates the
  // consume phase (a failed guard consumes nothing and poisons; the early words carry a tag no advanced consumer accepts)
  // the early push also requires this CTA's own missed-advance word to differ from the tag's forward (each thread reads it, no
  // barrier): a launch that missed the forward advance pushes nothing, so its stale-tag words can never reach a peer that is
  // still reading the previous forward
  const bool early_push = do_push && !(step_flags & kStepAcquire) && (step_flags & kPushBeforeGuard) &&
                          blockIdx.x < s.units &&
                          *reinterpret_cast<const volatile unsigned long long*>(s.last_seq + blockIdx.x) != x.seq;
  auto push_all = [&]() {
    for (bool first = true; it0 < ptotal; it0 += pstride * uint32_t(kPushBatch), first = false) {
      if (!first) load_batch(it0);
#pragma unroll
      for (int u = 0; u < kPushBatch; ++u) {
        if (!vu[u] || (skip != nullptr && skip[rowu[u]])) continue;
        if (step_flags & kStockZeroSign) c3::stock_zero_pairs(e[u]);
        c3::rs_push6<W>(s, x, uint32_t(M), rowu[u], colu[u], e[u]);
      }
    }
  };
  if (early_push) push_all();
  __shared__ int go_smem;
  if (threadIdx.x < 32) {
    int go = 0;
    if (threadIdx.x == 0) {
      go = lh::guard_enter<W>(s, c, blockIdx.x, x.seq) ? 1 : 0;
      if (go && (step_flags & kStepRelease) && blockIdx.x == 0) lh::step_release<W>(tok, x.seq);
    }
    go = __shfl_sync(0xffffffffu, go, 0);
    if (go && (step_flags & kStepAcquire)) go = lh::step_acquire<W>(tok, c, x.seq) ? 1 : 0;
    if (threadIdx.x == 0) go_smem = go;
  }
  __syncthreads();
  const bool go = go_smem != 0;
  C3_STAMP(1);
  if (go && do_push && !early_push) push_all();
  C3_STAMP(2);
  // items per lane, sized so the fold stays inside the 128-register budget of __launch_bounds__(256, 2) without spills
  // (measured with cuobjdump -res-usage: 4 items spill at W 4 / 8)
  constexpr int N = (W <= 2) ? 4 : ((W <= 4) ? 2 : 1);
  static_assert(EpiMinCtas<Epi>::value == 1 || EpiMinCtas<Epi>::value == 2, "row epilogue: kMinCtasPerSm 1 or 2");
  const uint32_t lane = threadIdx.x & 31u;
  if constexpr (EpiFoldsRow<Epi>::value) {
    // row-fold consumer: the epilogue folds each owned row in its own layout (RowFold::fold8); no fold of the kernel's own
    if (step_flags & kRowFold) {
      const RowFold<W> rf{s, c, x, part, H, (step_flags & kStockZeroSign) != 0u, go};
      for (uint32_t rr = blockIdx.x; rr < own_n; rr += gridDim.x) {
        epi.fold_row(rr, own0 + rr, rf, out + size_t(rr) * H);
        __syncthreads();
      }
    }
  } else if (step_flags & kRowFold) {
    // row-owned fold: each owned row is folded inside ONE CTA (all its threads, N groups per lane per round), so a per-row
    // epilogue (e.g. an owner norm) can follow a CTA barrier with the whole row in this CTA
    for (uint32_t rr = blockIdx.x; rr < own_n; rr += gridDim.x) {
      const uint32_t row = own0 + rr;
      const uint32_t mrow = mask != nullptr ? uint32_t(mask[row]) : 0u;
      for (uint32_t g0 = 0; g0 < ngroups; g0 += blockDim.x * uint32_t(N)) {
        uint32_t rows[N], cols[N], m[N];
        int n = 0;
#pragma unroll
        for (int i = 0; i < N; ++i) {
          const uint32_t g = g0 + threadIdx.x + blockDim.x * uint32_t(i);
          rows[i] = row;
          cols[i] = g < ngroups ? g * 6u : 0u;
          m[i] = mrow;
          if (g < ngroups) n = i + 1;
        }
        float sum[N][6];
        bool ok = go;
        if (go) {
          const bool sz = (step_flags & kStockZeroSign) != 0u;
          if (own_local)
            ok = fold_own_local<W, N>(s, c, x, rows, cols, m, n, mask != nullptr, part, H, sz, sum);
          else
            ok = (mask == nullptr) ? c3::rsA_collect6<W, N>(s, c, x, rows, cols, n, sum)
                                   : c3::rsB_collect6<W, N>(s, c, x, rows, cols, m, n, sum, sz);
        }
#pragma unroll
        for (int i = 0; i < N; ++i) {
          if (i >= n) break;
          uint32_t* o32 = reinterpret_cast<uint32_t*>(out + size_t(rr) * H);
#pragma unroll
          for (int j = 0; j < 6; j += 2) {
            const uint32_t col = cols[i] + uint32_t(j);
            if (col < uint32_t(H)) {
              const uint32_t lo = ok ? __bfloat16_as_ushort(__float2bfloat16_rn(sum[i][j])) : lh::kPoisonBf16;
              const uint32_t hi = ok ? __bfloat16_as_ushort(__float2bfloat16_rn(sum[i][j + 1])) : lh::kPoisonBf16;
              o32[col >> 1] = lo | (hi << 16);
            }
          }
        }
      }
      if constexpr (Epi::kActive) {
        __syncthreads();  // this CTA's stores of owned row rr precede the epilogue's reads of it
        epi.row(rr, out + size_t(rr) * H, H);
        __syncthreads();
      }
    }
  }
  const uint32_t items = (step_flags & kRowFold) ? 0u : own_n * ngroups;
  const uint32_t gw = blockIdx.x * fw + (warp - (split ? pw : 0u)), nw = gridDim.x * fw, chunk = 32u * uint32_t(N);
  for (uint32_t base = folder ? gw * chunk : items; base < items; base += nw * chunk) {
    uint32_t rows[N], cols[N], m[N];
    int n = 0;
#pragma unroll
    for (int i = 0; i < N; ++i) {
      const uint32_t idx = base + lane + 32u * uint32_t(i);
      const bool valid = idx < items;
      const uint32_t r = valid ? idx / ngroups : 0u;
      rows[i] = own0 + r;
      cols[i] = valid ? (idx - r * ngroups) * 6u : 0u;
      m[i] = mask != nullptr ? uint32_t(mask[rows[i]]) : 0u;
      if (valid) n = i + 1;
    }
    float sum[N][6];
    bool ok = go;
    if (go) {
      const bool sz = (step_flags & kStockZeroSign) != 0u;
      if (own_local)
        ok = fold_own_local<W, N>(s, c, x, rows, cols, m, n, mask != nullptr, part, H, sz, sum);
      else
        ok = (mask == nullptr) ? c3::rsA_collect6<W, N>(s, c, x, rows, cols, n, sum)
                               : c3::rsB_collect6<W, N>(s, c, x, rows, cols, m, n, sum, sz);
    }
#pragma unroll
    for (int i = 0; i < N; ++i) {
      if (i >= n) break;
      uint32_t* o32 = reinterpret_cast<uint32_t*>(out + size_t(rows[i] - own0) * H);
#pragma unroll
      for (int j = 0; j < 6; j += 2) {
        const uint32_t col = cols[i] + uint32_t(j);
        if (col < uint32_t(H)) {  // H even: col + 1 < H too
          const uint32_t lo = ok ? __bfloat16_as_ushort(__float2bfloat16_rn(sum[i][j])) : lh::kPoisonBf16;
          const uint32_t hi = ok ? __bfloat16_as_ushort(__float2bfloat16_rn(sum[i][j + 1])) : lh::kPoisonBf16;
          o32[col >> 1] = lo | (hi << 16);
        }
      }
    }
  }
  __syncthreads();
  C3_STAMP(3);
  if (threadIdx.x == 0 && go) lh::guard_exit<W>(s, blockIdx.x, x.seq);
  if (!(step_flags & kEarlyTrigger)) device::PDLTriggerSecondary<kUsePDL>();
}


// AG_x, tagged form (no flags, no release): every rank pushes its owned rows of x_owned [own_n, H] as kTag64 words into its
// block of the unique site `s` (one multicast store per word when the site table carries a multicast address, else one
// store per rank), and copies them into out [M, H]; then every warp of the grid polls the other ranks' rows of the pass,
// each word accepted on its own matching load (the AG_tok rule of lb_hop's poll_v2), and writes them as plain bf16 into
// out, the buffer the library kernel reads after this grid completes. Grid: one resident wave; one consumer unit per CTA.
template <int W, bool kUsePDL>
__global__ void __launch_bounds__(256, 2) c3_ag_tagged_kernel(const __grid_constant__ lh::Site<W> s, const __grid_constant__ lh::Ctl c,
                                                           const __grid_constant__ lh::StepTok<W> tok, uint32_t step_flags,
                                                           const __nv_bfloat16* __restrict__ x_owned,
                                                           __nv_bfloat16* __restrict__ out, int M, int H) {
  const int Hp = int(s.width);
  device::PDLWaitPrimary<kUsePDL>();
  C3_STAMP(0);
  if (step_flags & kEarlyTrigger) device::PDLTriggerSecondary<kUsePDL>();
  const uint32_t ngroups = uint32_t(Hp / 6);
  const uint32_t own0 = c3::own0(uint32_t(M), uint32_t(W), s.rank), own1 = c3::own0(uint32_t(M), uint32_t(W), s.rank + 1u);
  const uint32_t own_n = own1 - own0;
  const uint32_t wpc = blockDim.x >> 5, warp = threadIdx.x >> 5;
  const bool split = (step_flags & kSplitRoles) != 0u;
  const uint32_t pw = split ? wpc / 2u : wpc, fw = split ? wpc - pw : wpc;
  const bool pusher = warp < pw, folder = !split || !pusher;
  const uint32_t ptotal = own_n * ngroups, pstride = gridDim.x * pw * 32u;
  uint32_t it0 = blockIdx.x * pw * 32u + threadIdx.x;
  uint32_t e[kPushBatch][6], rowu[kPushBatch], gu[kPushBatch];
  bool vu[kPushBatch];
  auto load_batch = [&](uint32_t b) {
#pragma unroll
    for (int u = 0; u < kPushBatch; ++u) {
      const uint32_t it = b + uint32_t(u) * pstride;
      vu[u] = it < ptotal;
      rowu[u] = vu[u] ? it / ngroups : 0u;
      gu[u] = vu[u] ? it - rowu[u] * ngroups : 0u;
      load_group6(x_owned, H, rowu[u], gu[u], vu[u], e[u]);
    }
  };
  if (pusher) load_batch(it0);  // in flight across the guard / step-token barrier
  const lh::Xfer x = lh::xfer<W>(s);
  const bool early_push = pusher && !(step_flags & kStepAcquire) && (step_flags & kPushBeforeGuard) &&
                          blockIdx.x < s.units &&
                          *reinterpret_cast<const volatile unsigned long long*>(s.last_seq + blockIdx.x) != x.seq;
  auto push_all = [&]() {
    for (bool first = true; it0 < ptotal; it0 += pstride * uint32_t(kPushBatch), first = false) {
      if (!first) load_batch(it0);
#pragma unroll
      for (int u = 0; u < kPushBatch; ++u) {
        if (!vu[u]) continue;
        push6_peers<W>(s, x, own0 + rowu[u], gu[u] * 6u, e[u]);
        store_group6(out, H, own0 + rowu[u], gu[u] * 6u, e[u], true);
      }
    }
  };
  if (early_push) push_all();
  __shared__ int go_smem;
  if (threadIdx.x < 32) {
    int go = 0;
    if (threadIdx.x == 0) {
      go = lh::guard_enter<W>(s, c, blockIdx.x, x.seq) ? 1 : 0;
      if (go && (step_flags & kStepRelease) && blockIdx.x == 0) lh::step_release<W>(tok, x.seq);
    }
    go = __shfl_sync(0xffffffffu, go, 0);
    if (go && (step_flags & kStepAcquire)) go = lh::step_acquire<W>(tok, c, x.seq) ? 1 : 0;
    if (threadIdx.x == 0) go_smem = go;
  }
  __syncthreads();
  const bool go = go_smem != 0;
  C3_STAMP(1);
  if (go && pusher && !early_push) push_all();
  C3_STAMP(2);
  // gather: the other ranks' rows, chunks of 32 x N consecutive (row, group) items per gathering warp
  constexpr int N = 8;
  const uint32_t items = (uint32_t(M) - own_n) * ngroups;
  const uint32_t lane = threadIdx.x & 31u;
  const uint32_t gw = blockIdx.x * fw + (warp - (split ? pw : 0u)), nw = gridDim.x * fw, chunk = 32u * uint32_t(N);
  for (uint32_t base = folder ? gw * chunk : items; base < items; base += nw * chunk) {
    const unsigned long long* a[N][1];
    ulonglong2 v[N][1];
    uint32_t rows[N], cols[N];
    int n = 0;
#pragma unroll
    for (int i = 0; i < N; ++i) {
      const uint32_t idx = base + lane + 32u * uint32_t(i);
      const bool valid = idx < items;
      const uint32_t rl = valid ? idx / ngroups : 0u;
      const uint32_t row = rl < own0 ? rl : rl + own_n;
      rows[i] = valid ? row : (own0 == 0u ? own1 : 0u);
      cols[i] = valid ? (idx - rl * ngroups) * 6u : 0u;
      a[i][0] = lh::word_ptr<W>(s, s.rank, x.copy, c3::owner_of(uint32_t(M), uint32_t(W), rows[i]), rows[i], cols[i] / 3u);
      if (valid) n = i + 1;
    }
    bool ok = go;
    if (go) ok = lh::poll_v2<1, N>(c, a, v, n, x.tag, s.rank, s.site);
#pragma unroll
    for (int i = 0; i < N; ++i) {
      if (i >= n) break;
      uint32_t e[6];
#pragma unroll
      for (int j = 0; j < 6; ++j) e[j] = lh::elem6(v[i][0], j);
      store_group6(out, H, rows[i], cols[i], e, ok);
    }
  }
  __syncthreads();  // ends this CTA's reads of the site
  C3_STAMP(3);
  if (threadIdx.x == 0 && go) lh::guard_exit<W>(s, blockIdx.x, x.seq);
  if (!(step_flags & kEarlyTrigger)) device::PDLTriggerSecondary<kUsePDL>();
}

// AG_x: every rank's owned rows (bf16 [own_n, H]) into copy `copy` of every rank's landing [copies][rows_max][H] through the
// multicast address; per-(src, CTA) flags raised by a releasing multicast max after the CTA's stores; CTA 0 acquires every
// flag (bounded) and orders the unicast reads that follow the grid with fence.proxy.alias. Grid: one resident wave.
template <int W, bool kUsePDL>
__global__ void __launch_bounds__(256) c3_ag_rows_kernel(const c3::AgX g, const __grid_constant__ lh::Ctl c,
                                                         const __nv_bfloat16* __restrict__ x_owned, int M, int H, uint32_t copy,
                                                         const unsigned long long* __restrict__ seq64, uint32_t layer,
                                                         uint32_t opts) {
  device::PDLWaitPrimary<kUsePDL>();
  if (opts & kEarlyTrigger) device::PDLTriggerSecondary<kUsePDL>();
  // the stamp comes from the forward sequence word on the device (graph replays repeat kernel arguments): strictly increasing
  // across forwards and layers (layer < 64)
  const unsigned long long stamp = lh::ld_relaxed_gpu_u64(seq64) * 64ull + layer;
  // missed advance (the forward word was not raised since this CTA's last publish): reject loudly, store and publish nothing
  __shared__ int stale_smem;
  if (threadIdx.x == 0) {
    stale_smem = stamp <= g.last[blockIdx.x] ? 1 : 0;
    if (stale_smem) lh::fail(c, lh::kErrMissedAdvance, g.rank, c3::kSiteBase + 3u);
  }
  __syncthreads();
  if (stale_smem) return;
  const uint32_t own0 = c3::own0(uint32_t(M), uint32_t(W), g.rank), own1 = c3::own0(uint32_t(M), uint32_t(W), g.rank + 1u);
  const uint32_t row_bytes = uint32_t(H) * 2u;
  for (uint32_t row = own0 + blockIdx.x; row < own1; row += gridDim.x) {
    const uint4* src = reinterpret_cast<const uint4*>(x_owned + size_t(row - own0) * H);
    for (uint32_t v = threadIdx.x; v * 16u < row_bytes; v += blockDim.x) c3::agx_st16(g, copy, row, v * 16u, src[v]);
  }
  __syncthreads();
  if (threadIdx.x == 0) {
    c3::agx_publish(g, blockIdx.x, stamp);
    g.last[blockIdx.x] = stamp;
  }
  if (blockIdx.x == 0 && threadIdx.x < 32) c3::agx_wait_all(g, c, gridDim.x, stamp);
  __syncthreads();
  if (!(opts & kEarlyTrigger)) device::PDLTriggerSecondary<kUsePDL>();
}

// Front-role landing unpack (default off): one warp per step row (any rank's row; source = the row's owner). The row's token
// words (the front role's kB48 token row: 2320 bytes = 387 words) and its 8 ids words are read from this rank's receive regions
// of the two unique sites with the AG_tok / ids consumer rules (each word accepted on its own matching load, tags derived from
// the forward sequence word, bounded waits that set the sticky error word), and written as plain bytes into the tensors the
// stock expert stage reads: hs_fp4 [M, 2048] u8 = token-row bytes [0, 2048), hs_sf [M, 256] u8 = [2048, 2304), pts [M] fp32 =
// [2304, 2308), packed [M, 8] i32 = the ids words' low halves. An expired wait poisons that row (0xFF bytes, NaN, -1).
inline constexpr uint32_t kFrRowBytes = 2320u, kFrRowWords = 387u, kFrFp4Bytes = 2048u, kFrSfBytes = 256u, kFrPtsOff = 2304u;
inline constexpr int kFrWarps = 4;

template <int W, bool kUsePDL>
__global__ void __launch_bounds__(32 * kFrWarps) c3_fr_unpack_kernel(const __grid_constant__ lh::Site<W> ids,
                                                                    const __grid_constant__ lh::Site<W> tok,
                                                                    const __grid_constant__ lh::Ctl c, int M,
                                                                    uint8_t* __restrict__ hs_fp4, uint8_t* __restrict__ hs_sf,
                                                                    float* __restrict__ pts, int32_t* __restrict__ packed) {
  constexpr int kLaneWords = int((kFrRowWords + 31u) / 32u);  // 13
  __shared__ __align__(16) uint8_t stage[kFrWarps][kFrRowBytes + 16u];
  device::PDLWaitPrimary<kUsePDL>();
  const uint32_t warp = threadIdx.x >> 5, lane = threadIdx.x & 31u;
  const lh::Xfer xi = lh::xfer<W>(ids), xt = lh::xfer<W>(tok);
  uint8_t* st = stage[warp];
  for (uint32_t row = blockIdx.x * kFrWarps + warp; row < uint32_t(M); row += gridDim.x * kFrWarps) {
    const uint32_t src = c3::owner_of(uint32_t(M), uint32_t(W), row);
    unsigned long long w[kLaneWords];
    const bool ok = c3::agTok_read_row<W, kLaneWords>(tok, c, xt, src, row, kFrRowWords, w);
#pragma unroll
    for (int k = 0; k < kLaneWords; ++k) {
      const uint32_t wi = lane + 32u * uint32_t(k);
      if (wi < kFrRowWords) {
#pragma unroll
        for (int b = 0; b < 6; ++b) {
          const uint32_t i = 6u * wi + uint32_t(b);
          if (i < kFrRowBytes) st[i] = ok ? uint8_t(w[k] >> (8 * b)) : uint8_t(0xFF);
        }
      }
    }
    __syncwarp();
    const uint4* s4 = reinterpret_cast<const uint4*>(st);
    uint4* d4 = reinterpret_cast<uint4*>(hs_fp4 + size_t(row) * kFrFp4Bytes);
#pragma unroll
    for (int j = 0; j < int(kFrFp4Bytes / 16u / 32u); ++j) d4[lane + 32u * uint32_t(j)] = s4[lane + 32u * uint32_t(j)];
    if (lane < kFrSfBytes / 16u)
      reinterpret_cast<uint4*>(hs_sf + size_t(row) * kFrSfBytes)[lane] = s4[kFrFp4Bytes / 16u + lane];
    if (lane == 0) pts[row] = ok ? *reinterpret_cast<const float*>(st + kFrPtsOff) : __int_as_float(0x7fc00000);
    uint32_t mine = 0u;
    const bool oki = c3::ids_read_row<W>(ids, c, xi, src, row, &mine);
    if (lane < 8u) packed[size_t(row) * 8u + lane] = oki ? int32_t(mine) : -1;
    __syncwarp();  // the stage is reused by this warp's next row
  }
  device::PDLTriggerSecondary<kUsePDL>();
}

}  // namespace muse_vr_c3_ops

template <int W, bool kUsePDL>
struct MuseVrC3Rs {
  // front-role landing unpack: the ids / token sites' CONSUMER tables (this rank's receive regions), FR's site geometry
  static void fr_unpack(const tvm::ffi::TensorView ids_table, const tvm::ffi::TensorView tok_table, const tvm::ffi::TensorView ctl,
                        int64_t rank, int64_t ids_rows_max, int64_t ids_width, int64_t ids_pitch, int64_t ids_policy,
                        int64_t ids_parity, int64_t ids_site, int64_t tok_rows_max, int64_t tok_width, int64_t tok_pitch,
                        int64_t tok_policy, int64_t tok_parity, int64_t tok_site, int64_t max_rounds, int64_t spin_rounds,
                        int64_t flags, int64_t backoff_ns, int64_t M, const tvm::ffi::TensorView hs_fp4,
                        const tvm::ffi::TensorView hs_sf, const tvm::ffi::TensorView pts, const tvm::ffi::TensorView packed) {
    using namespace host;
    namespace o = muse_vr_c3_ops;
    RuntimeCheck(M >= 1 && M <= ids_rows_max && M <= tok_rows_max, "c3 fr_unpack: M <= the sites' rows_max (rows are global)");
    RuntimeCheck(ids_pitch >= 8 && tok_pitch >= int64_t(o::kFrRowWords), "c3 fr_unpack: site pitch (8 ids words, 387 token words)");
    RuntimeCheck(hs_fp4.size(0) >= M && hs_fp4.size(1) == int64_t(o::kFrFp4Bytes) && hs_sf.size(0) >= M &&
                     hs_sf.size(1) == int64_t(o::kFrSfBytes) && pts.size(0) >= M && packed.size(0) >= M && packed.size(1) == 8,
                 "c3 fr_unpack: hs_fp4 [M, 2048] u8, hs_sf [M, 256] u8, pts [M] f32, packed [M, 8] i32");
    const auto si = muse_vr_lb_hop_ops::make_site<W>(ids_table, rank, ids_rows_max, ids_width, ids_pitch, ids_policy, ids_parity,
                                                     0, ids_site, 1);
    const auto sk = muse_vr_lb_hop_ops::make_site<W>(tok_table, rank, tok_rows_max, tok_width, tok_pitch, tok_policy, tok_parity,
                                                     0, tok_site, 1);
    const auto c = muse_vr_lb_hop_ops::make_ctl(ctl, max_rounds, spin_rounds, flags, backoff_ns);
    const auto stream = LaunchKernel::resolve_device(hs_fp4.device());
    const unsigned grid = unsigned((M + o::kFrWarps - 1) / o::kFrWarps);
    LaunchKernel(dim3(grid), 32 * o::kFrWarps, stream)
        .enable_pdl(kUsePDL)(o::c3_fr_unpack_kernel<W, kUsePDL>, si, sk, c, int(M), static_cast<uint8_t*>(hs_fp4.data_ptr()),
                             static_cast<uint8_t*>(hs_sf.data_ptr()), static_cast<float*>(pts.data_ptr()),
                             static_cast<int32_t*>(packed.data_ptr()));
  }

  // AG_x tagged form: table = the site's PRODUCER table (with the multicast address when the arena has one).
  static void agt(const tvm::ffi::TensorView table, const tvm::ffi::TensorView ctl, const tvm::ffi::TensorView tok_table,
                  int64_t rank, int64_t rows_max, int64_t width, int64_t pitch, int64_t policy, int64_t parity, int64_t index,
                  int64_t site, int64_t units, int64_t max_rounds, int64_t spin_rounds, int64_t flags, int64_t backoff_ns,
                  int64_t step_flags, const tvm::ffi::TensorView x_owned, const tvm::ffi::TensorView out, int64_t M) {
    using namespace host;
    const int64_t H = out.size(1);
    RuntimeCheck(width % 6 == 0 && width >= H && width < H + 6 && H % 2 == 0, "c3 agt: even H; site width = H rounded up to a multiple of 6");
    RuntimeCheck(M >= 1 && M <= rows_max && out.size(0) >= M && x_owned.size(1) == H, "c3 agt: shapes");
    const auto s = muse_vr_lb_hop_ops::make_site<W>(table, rank, rows_max, width, pitch, policy, parity, index, site, units);
    const auto c = muse_vr_lb_hop_ops::make_ctl(ctl, max_rounds, spin_rounds, flags, backoff_ns);
    const auto t = muse_vr_lb_hop_ops::make_tok<W>(tok_table, rank, site);
    const auto stream = LaunchKernel::resolve_device(out.device());
    const auto kern = muse_vr_c3_ops::c3_ag_tagged_kernel<W, kUsePDL>;
    if (uint32_t(step_flags) & muse_vr_c3_ops::kEarlyTrigger) {
      static bool carve = false;
      if (!carve) {
        RuntimeDeviceCheck(cudaFuncSetAttribute(kern, cudaFuncAttributePreferredSharedMemoryCarveout,
                                                cudaSharedmemCarveoutMaxShared));
        carve = true;
      }
    }
    LaunchKernel(dim3(unsigned(units)), 256, stream)
        .enable_pdl(kUsePDL)(kern, s, c, t, uint32_t(step_flags), static_cast<const __nv_bfloat16*>(x_owned.data_ptr()),
                             static_cast<__nv_bfloat16*>(out.data_ptr()), int(M), int(H));
  }
  static void rows(const tvm::ffi::TensorView table, const tvm::ffi::TensorView ctl, const tvm::ffi::TensorView tok_table,
                   int64_t rank, int64_t rows_max, int64_t width, int64_t pitch, int64_t policy, int64_t parity, int64_t index,
                   int64_t site, int64_t units, int64_t max_rounds, int64_t spin_rounds, int64_t flags, int64_t backoff_ns,
                   int64_t step_flags, const tvm::ffi::TensorView part, const tvm::ffi::TensorView out, int64_t M,
                   const tvm::ffi::TensorView skip, int64_t use_skip, const tvm::ffi::TensorView mask, int64_t use_mask) {
    using namespace host;
    const int64_t H = part.size(1);
    RuntimeCheck(width % 6 == 0 && width >= H && width < H + 6 && H % 2 == 0, "c3 rs: even H; site width = H rounded up to a multiple of 6");
    RuntimeCheck(M >= 1 && M <= rows_max, "c3 rs: 1 <= M <= rows_max");
    const auto s = muse_vr_lb_hop_ops::make_site<W>(table, rank, rows_max, width, pitch, policy, parity, index, site, units);
    const auto c = muse_vr_lb_hop_ops::make_ctl(ctl, max_rounds, spin_rounds, flags, backoff_ns);
    const auto t = muse_vr_lb_hop_ops::make_tok<W>(tok_table, rank, site);
    const auto stream = LaunchKernel::resolve_device(part.device());
    // one consumer unit (missed-advance word) per CTA: the grid is the site's unit count
    const bool flat = (uint32_t(step_flags) & muse_vr_c3_ops::kFlatGrid) != 0u;
    const uint32_t fl = uint32_t(step_flags) & ~muse_vr_c3_ops::kFlatGrid;
    const auto* pp = static_cast<const __nv_bfloat16*>(part.data_ptr());
    auto* op = static_cast<__nv_bfloat16*>(out.data_ptr());
    const auto* sk = use_skip ? static_cast<const uint8_t*>(skip.data_ptr()) : nullptr;
    const auto* mk = use_mask ? static_cast<const int32_t*>(mask.data_ptr()) : nullptr;
    if (flat) {
      const auto kern = muse_vr_c3_ops::c3_rs_rows_flat_kernel<W, kUsePDL, muse_vr_c3_ops::NoRowEpilogue>;
      if (fl & muse_vr_c3_ops::kEarlyTrigger) {
        static bool carve = false;
        if (!carve) {
          RuntimeDeviceCheck(cudaFuncSetAttribute(kern, cudaFuncAttributePreferredSharedMemoryCarveout,
                                                  cudaSharedmemCarveoutMaxShared));
          carve = true;
        }
      }
      LaunchKernel(dim3(unsigned(units)), 256, stream)
          .enable_pdl(kUsePDL)(kern, s, c, t, fl, pp, op, int(M), int(H), sk, mk, muse_vr_c3_ops::NoRowEpilogue{});
    } else {
      const auto kern = muse_vr_c3_ops::c3_rs_rows_kernel<W, kUsePDL>;
      if (fl & muse_vr_c3_ops::kEarlyTrigger) {
        static bool carve = false;
        if (!carve) {
          RuntimeDeviceCheck(cudaFuncSetAttribute(kern, cudaFuncAttributePreferredSharedMemoryCarveout,
                                                  cudaSharedmemCarveoutMaxShared));
          carve = true;
        }
      }
      LaunchKernel(dim3(unsigned(units)), 256, stream).enable_pdl(kUsePDL)(kern, s, c, t, fl, pp, op, int(M), int(H), sk, mk);
    }
  }
};


template <int W, bool kUsePDL>
struct MuseVrC3Ag {
  // landing: int64 host table [mc_data, uc_data, mc_flags, uc_flags, last]; rows_max / ctas_max: the landing's sizes.
  static void rows(const tvm::ffi::TensorView landing, const tvm::ffi::TensorView ctl, int64_t rank, int64_t rows_max,
                   int64_t ctas_max, int64_t max_rounds, int64_t spin_rounds, int64_t flags, int64_t backoff_ns,
                   const tvm::ffi::TensorView x_owned, int64_t M, int64_t copy, const tvm::ffi::TensorView seq64, int64_t layer,
                   int64_t ctas, int64_t opts) {
    using namespace host;
    const int64_t H = x_owned.size(1);
    RuntimeCheck((H * 2) % 16 == 0, "c3 ag: row bytes a multiple of 16");
    RuntimeCheck(M >= 1 && M <= rows_max && ctas >= 1 && ctas <= ctas_max && copy >= 0, "c3 ag: arguments");
    RuntimeCheck(landing.numel() == 5, "c3 ag: landing table size");
    RuntimeCheck(layer >= 0 && layer < 64, "c3 ag: layer < 64");
    const auto* v = static_cast<const int64_t*>(landing.data_ptr());
    muse_vr::c3::AgX g{};
    g.mc = reinterpret_cast<unsigned char*>(v[0]);
    g.uc = reinterpret_cast<unsigned char*>(v[1]);
    g.flag_mc = reinterpret_cast<unsigned long long*>(v[2]);
    g.flag_uc = reinterpret_cast<const unsigned long long*>(v[3]);
    g.last = reinterpret_cast<unsigned long long*>(v[4]);
    g.row_bytes = uint32_t(H * 2);
    g.rows_max = uint32_t(rows_max);
    g.ctas_max = uint32_t(ctas_max);
    g.world = uint32_t(W);
    g.rank = uint32_t(rank);
    const auto c = muse_vr_lb_hop_ops::make_ctl(ctl, max_rounds, spin_rounds, flags, backoff_ns);
    const auto stream = LaunchKernel::resolve_device(x_owned.device());
    const auto kern = muse_vr_c3_ops::c3_ag_rows_kernel<W, kUsePDL>;
    if (uint32_t(opts) & muse_vr_c3_ops::kEarlyTrigger) {
      static bool carve = false;
      if (!carve) {
        RuntimeDeviceCheck(cudaFuncSetAttribute(kern, cudaFuncAttributePreferredSharedMemoryCarveout,
                                                cudaSharedmemCarveoutMaxShared));
        carve = true;
      }
    }
    LaunchKernel(dim3(unsigned(ctas)), 256, stream)
        .enable_pdl(kUsePDL)(kern, g, c, static_cast<const __nv_bfloat16*>(x_owned.data_ptr()), int(M), int(H),
                             uint32_t(copy), static_cast<const unsigned long long*>(seq64.data_ptr()), uint32_t(layer),
                             uint32_t(opts));
  }
};

}  // namespace sglang
