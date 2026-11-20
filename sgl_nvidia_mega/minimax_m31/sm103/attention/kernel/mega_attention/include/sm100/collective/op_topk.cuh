/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
// Exact TopK of one row: the 15 largest history-page scores (lower page wins ties) plus the
// local page, ascending. The order is the one of runtime_topk (ordered FP32 score, then lower
// page); only the search differs: the ordered score is bucketed in integer space, relative to
// the row minimum, in two 16-bit levels. Level 1 ranks (o - lo) >> shift; when the 15th and
// 16th candidates share a level-1 bucket, level 2 ranks the exact low bits (o - lo) & mask of
// that bucket's pages. Both levels together resolve the full 32-bit key, so the selection is
// exact for every score distribution with at most two passes over registers.
#pragma once

#include <climits>

#include "../common/ring_sync.cuh"
#include "attn/kernel/include/sm100_m3_decode_topk_runtime.hpp" // ordered_score

namespace attn_ring::topk {

namespace rt = cutlass::m3_decode::collective::runtime_topk;

constexpr int kMaxCandidates = 512; // threshold pass; more candidates take the bucket path

struct alignas(16) Scratch {
  uint64_t exact[kMaxCandidates];    // threshold pass: exact keys (ordered score << 32 | ~page)
  uint32_t cand[kConsumerWarps][16]; // per-warp top-16 keys, descending
  uint32_t slots[16];                // block top-16 keys, descending
  uint32_t lo[kConsumerWarps];
  uint32_t hi[kConsumerWarps];
  uint32_t count[kConsumerWarps];
  int selected[16];
};

// Ordered scores of pages tid + i * 256 (i < W; pages >= n are not loaded) from the tagged 8-byte
// score records; with poll, every load is re-issued until its record carries `tag` (all W loads
// stay in flight at once).
template <int W>
__device__ __forceinline__ void load_scores(const uint8_t *recs, uint32_t tag, bool poll, int n,
                                            int tid, uint32_t (&o)[W]) {
  // Pages >= n are not ranked (callers ignore their value): no load, so no thread polls a clamped
  // copy.
  uint2 r[W];
#pragma unroll
  for (int i = 0; i < W; ++i) {
    const int page = tid + i * kConsumerThreads;
    r[i] = page < n ? ld_rec64(recs + size_t(page) * 8) : make_uint2(0u, tag);
  }
  uint32_t pending = 0u;
#pragma unroll
  for (int i = 0; i < W; ++i)
    if (poll && r[i].y != tag)
      pending |= 1u << i;
  while (pending) {
#pragma unroll
    for (int i = 0; i < W; ++i)
      if (pending & (1u << i))
        r[i] = ld_rec64(recs + size_t(tid + i * kConsumerThreads) * 8);
#pragma unroll
    for (int i = 0; i < W; ++i)
      if ((pending & (1u << i)) && r[i].y == tag)
        pending &= ~(1u << i);
  }
#pragma unroll
  for (int i = 0; i < W; ++i)
    o[i] = r[i].x;
}

// Descending bitonic sort of W (power of two) keys; every index is a compile-time constant, so
// the keys stay in registers.
template <int W> __device__ __forceinline__ void sort_desc(uint32_t (&a)[W]) {
#pragma unroll
  for (int k = 2; k <= W; k <<= 1) {
#pragma unroll
    for (int j = k >> 1; j > 0; j >>= 1) {
#pragma unroll
      for (int i = 0; i < W; ++i) {
        const int l = i ^ j;
        if (l > i) {
          const uint32_t x = a[i], y = a[l];
          const uint32_t big = max(x, y), small = min(x, y);
          const bool desc = (i & k) == 0;
          a[i] = desc ? big : small;
          a[l] = desc ? small : big;
        }
      }
    }
  }
}

// Block top-16 of 32-bit keys (0 = no candidate; valid keys are unique). All 256 consumer
// threads call; s.slots holds the result when this returns.
template <int W>
__device__ __forceinline__ void block_top16(uint32_t (&keys)[W], Scratch &s, int tid) {
  const int lane = tid & 31, warp = tid >> 5;
  sort_desc(keys);
  constexpr int kLive = W < 16 ? W : 16;
#pragma unroll
  for (int round = 0; round < 16; ++round) {
    const uint32_t winner = __reduce_max_sync(0xffffffffu, keys[0]);
    if (lane == 0)
      s.cand[warp][round] = winner;
    const bool won = keys[0] == winner;
#pragma unroll
    for (int i = 0; i < kLive - 1; ++i)
      keys[i] = won ? keys[i + 1] : keys[i];
    keys[kLive - 1] = won ? 0u : keys[kLive - 1];
  }
  consumer_sync();
  if (warp == 0) {
    const uint4 w = reinterpret_cast<const uint4 *>(s.cand)[lane];
    uint32_t run[4] = {w.x, w.y, w.z, w.w};
#pragma unroll
    for (int round = 0; round < 16; ++round) {
      const uint32_t winner = __reduce_max_sync(0xffffffffu, run[0]);
      if (lane == 0)
        s.slots[round] = winner;
      const bool won = run[0] == winner;
#pragma unroll
      for (int i = 0; i < 3; ++i)
        run[i] = won ? run[i + 1] : run[i];
      run[3] = won ? 0u : run[3];
    }
  }
  consumer_sync();
}

__device__ __forceinline__ int key_page(uint32_t key) { return int(0xffffu - (key & 0xffffu)); }

// n = ranked history pages (16 <= n <= W * 256). Writes s.selected[0..14].
template <int W>
__device__ __noinline__ void select_history(const uint8_t *recs, uint32_t tag, int n, Scratch &s) {
  const int tid = int(threadIdx.x), lane = tid & 31, warp = tid >> 5;
  uint32_t o[W];
  uint32_t lo = 0xffffffffu, hi = 0u;
  load_scores<W>(recs, tag, false, n, tid, o); // complete: select_threshold saw every tag
#pragma unroll
  for (int i = 0; i < W; ++i) {
    const int page = tid + i * kConsumerThreads;
    o[i] = page < n ? rt::ordered_score(__uint_as_float(o[i])) : 0u;
    if (page < n) {
      lo = min(lo, o[i]);
      hi = max(hi, o[i]);
    }
  }
  lo = __reduce_min_sync(0xffffffffu, lo);
  hi = __reduce_max_sync(0xffffffffu, hi);
  if (lane == 0) {
    s.lo[warp] = lo;
    s.hi[warp] = hi;
  }
  consumer_sync();
#pragma unroll
  for (int w = 0; w < kConsumerWarps; ++w) {
    lo = min(lo, s.lo[w]);
    hi = max(hi, s.hi[w]);
  }
  const uint32_t range = hi - lo;
  const int shift = max(0, 16 - __clz(range)); // bit length of the range minus 16
  uint32_t keys[W];
#pragma unroll
  for (int i = 0; i < W; ++i) {
    const int page = tid + i * kConsumerThreads;
    keys[i] = page < n ? (((o[i] - lo) >> shift) << 16) | (~uint32_t(page) & 0xffffu) : 0u;
  }
  block_top16(keys, s, tid);
  const uint32_t boundary = s.slots[14] >> 16;
  if (shift == 0 || (s.slots[15] >> 16) != boundary) {
    // Level-1 buckets are exact, or the boundary bucket lies entirely inside the top 15.
    if (tid < 15)
      s.selected[tid] = key_page(s.slots[tid]);
    consumer_sync();
    return;
  }
  // The boundary bucket is shared by the 15th and the 16th candidate: rank its pages by the
  // exact low bits. Every page of a higher bucket is selected.
  int strict = 0;
  while (strict < 15 && (s.slots[strict] >> 16) > boundary)
    ++strict;
  if (tid < strict)
    s.selected[tid] = key_page(s.slots[tid]);
  const uint32_t mask = (1u << shift) - 1u; // shift in [1, 16]
#pragma unroll
  for (int i = 0; i < W; ++i) {
    const int page = tid + i * kConsumerThreads;
    const bool in = page < n && ((o[i] - lo) >> shift) == boundary;
    keys[i] = in ? (((o[i] - lo) & mask) << 16) | (~uint32_t(page) & 0xffffu) : 0u;
  }
  block_top16(keys, s, tid); // its first barrier orders every read of the level-1 slots
  if (tid >= strict && tid < 15)
    s.selected[tid] = key_page(s.slots[tid - strict]);
  consumer_sync();
}

// Threshold pass. T = the smallest over the eight warps of the warp's second-largest lane maximum:
// every warp holds two values >= T, so at least 16 pages score >= T and the top 15 all do. The
// pages >= T are compacted (exact 64-bit keys) and each one is ranked by counting the keys above
// it. Returns false (nothing selected) when more than kMaxCandidates pages reach T.
template <int W>
__device__ __noinline__ bool select_threshold(const uint8_t *recs, uint32_t tag, int n,
                                              Scratch &s) {
  const int tid = int(threadIdx.x), lane = tid & 31, warp = tid >> 5;
  uint32_t o[W];
  uint32_t m = 0u;
  bool has = false;
  // Every record polled at once, from clamped (always valid) addresses.
  load_scores<W>(recs, tag, true, n, tid, o);
#pragma unroll
  for (int i = 0; i < W; ++i) {
    const int page = tid + i * kConsumerThreads;
    o[i] = page < n ? rt::ordered_score(__uint_as_float(o[i])) : 0u;
    if (page < n) {
      m = has ? max(m, o[i]) : o[i];
      has = true;
    }
  }
  // Second-largest lane maximum of this warp (lanes without pages do not count).
  const uint32_t holders = __ballot_sync(0xffffffffu, has);
  const uint32_t first = __reduce_max_sync(0xffffffffu, has ? m : 0u);
  const uint32_t tied = __ballot_sync(0xffffffffu, has && m == first);
  const uint32_t second =
      __popc(tied) >= 2 ? first : __reduce_max_sync(0xffffffffu, (has && m != first) ? m : 0u);
  if (lane == 0)
    s.lo[warp] = __popc(holders) >= 2 ? second : 0u;
  if (lane == 0)
    s.hi[warp] = __popc(holders) >= 2 ? 1u : 0u;
  consumer_sync();
  uint32_t threshold = 0xffffffffu;
  bool all = true;
#pragma unroll
  for (int w = 0; w < kConsumerWarps; ++w) {
    threshold = min(threshold, s.lo[w]);
    all = all && s.hi[w] != 0u;
  }
  if (!all)
    threshold = 0u; // fewer than 16 guaranteed values: every page is a candidate (n < 256)
  int c = 0;
#pragma unroll
  for (int i = 0; i < W; ++i)
    c += (tid + i * kConsumerThreads < n && o[i] >= threshold) ? 1 : 0;
  int incl = c;
#pragma unroll
  for (int d = 1; d < 32; d <<= 1) {
    const int v = __shfl_up_sync(0xffffffffu, incl, d);
    if (lane >= d)
      incl += v;
  }
  if (lane == 31)
    s.count[warp] = uint32_t(incl);
  consumer_sync();
  int offset = 0, total = 0;
#pragma unroll
  for (int w = 0; w < kConsumerWarps; ++w) {
    const int cw = int(s.count[w]);
    offset += w < warp ? cw : 0;
    total += cw;
  }
  if (total > kMaxCandidates)
    return false; // uniform: every thread read the same counts
  int pos = offset + incl - c;
#pragma unroll
  for (int i = 0; i < W; ++i) {
    const int page = tid + i * kConsumerThreads;
    if (page < n && o[i] >= threshold) {
      s.exact[pos] = (uint64_t(o[i]) << 32) | uint64_t(~uint32_t(page));
      ++pos;
    }
  }
  consumer_sync();
  if (total <= kConsumerThreads / 4) {
    // Four consecutive lanes rank one candidate, each over a quarter of the keys.
    const int t = tid >> 2, q = tid & 3;
    const uint64_t key = t < total ? s.exact[t] : 0ull;
    int rank = 0;
#pragma unroll 4
    for (int j = q; j < total; j += 4)
      rank += s.exact[j] > key ? 1 : 0;
    rank += __shfl_xor_sync(0xffffffffu, rank, 1);
    rank += __shfl_xor_sync(0xffffffffu, rank, 2);
    if (t < total && q == 0 && rank < 15)
      s.selected[rank] = int(~uint32_t(key));
  } else {
    for (int t = tid; t < total; t += kConsumerThreads) {
      const uint64_t key = s.exact[t];
      int rank = 0;
#pragma unroll 4
      for (int j = 0; j < total; ++j)
        rank += s.exact[j] > key ? 1 : 0;
      if (rank < 15)
        s.selected[rank] = int(~uint32_t(key));
    }
  }
  consumer_sync();
  return true;
}

// 16 <= n <= 64 ranked pages: every page is a candidate. Thread p < n polls page p's record and
// stores its exact key; four lanes rank each candidate over all n keys. The same keys, tie-break
// and selection as the threshold pass, without its threshold / count / compaction steps. Writes
// s.selected[0..14].
__device__ __noinline__ void select_direct64(const uint8_t *recs, uint32_t tag, int n, Scratch &s) {
  const int tid = int(threadIdx.x);
  uint32_t o[1];
  load_scores<1>(recs, tag, true, n, tid, o);
  if (tid < n)
    s.exact[tid] =
        (uint64_t(rt::ordered_score(__uint_as_float(o[0]))) << 32) | uint64_t(~uint32_t(tid));
  consumer_sync();
  const int t = tid >> 2, q = tid & 3;
  const uint64_t key = t < n ? s.exact[t] : 0ull;
  int rank = 0;
#pragma unroll 4
  for (int j = q; j < n; j += 4)
    rank += s.exact[j] > key ? 1 : 0;
  rank += __shfl_xor_sync(0xffffffffu, rank, 1);
  rank += __shfl_xor_sync(0xffffffffu, rank, 2);
  if (t < n && q == 0 && rank < 15)
    s.selected[rank] = t;
  consumer_sync();
}

// All 256 consumer threads call. output[0..15]: selected logical pages ascending (-1 padded).
__device__ __forceinline__ void select(const uint8_t *recs, uint32_t tag, int visible_pages,
                                       Scratch &s, int *output) {
  const int tid = int(threadIdx.x);
  if (visible_pages <= 16) {
    if (tid < 16)
      output[tid] = tid < visible_pages ? tid : -1;
    consumer_sync();
    return;
  }
  const int n = visible_pages - 1;
  // Every row polls block-wide (one record per thread for n <= 256): a single polling warp behind
  // waiting warps was measured to stall for many microseconds.
  if (n <= 64) {
    select_direct64(recs, tag, n, s);
  } else if (n <= 256) {
    if (!select_threshold<1>(recs, tag, n, s))
      select_history<1>(recs, tag, n, s);
  } else if (n <= 512) {
    if (!select_threshold<2>(recs, tag, n, s))
      select_history<2>(recs, tag, n, s);
  } else if (n <= 1024) {
    if (!select_threshold<4>(recs, tag, n, s))
      select_history<4>(recs, tag, n, s);
  } else if (n <= 2048) {
    if (!select_threshold<8>(recs, tag, n, s))
      select_history<8>(recs, tag, n, s);
  } else if (n <= 4096) {
    if (!select_threshold<16>(recs, tag, n, s))
      select_history<16>(recs, tag, n, s);
  } else {
    if (!select_threshold<32>(recs, tag, n, s))
      select_history<32>(recs, tag, n, s);
  }
  if (tid < 32) {
    int page = tid < 15 ? s.selected[tid] : (tid == 15 ? n : INT_MAX);
#pragma unroll
    for (int span = 2; span <= 16; span *= 2) {
#pragma unroll
      for (int stride = span / 2; stride > 0; stride /= 2) {
        const int peer = __shfl_xor_sync(0xffffffffu, page, stride);
        page = ((tid & span) == 0) == ((tid & stride) == 0) ? min(page, peer) : max(page, peer);
      }
    }
    if (tid < 16)
      output[tid] = page;
  }
  consumer_sync();
}

} // namespace attn_ring::topk
