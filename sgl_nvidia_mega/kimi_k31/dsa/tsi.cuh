// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// K3.1 MK token-split indexer (TSI) for the sparse DSA verify at TP16/DCP1.
// Exact top-k merge adapted from SGLang's DcpTopKMerge (Apache-2.0; commit f3532057e15f)
// (kernels/jit/csrc/dsa/dcp_topk_merge.cuh: make_topk_key, select_kth, certificate, fallback); exchange protocol
// from kimi_k31/dcp/comm.cuh (uc_epoch records).
//
// Page striping: rank r scores the global pages p with p % W == r (W = world). Local token l <-> global position
// pos = (W * (l / 64) + r) * 64 + l % 64 (monotonic in l). local_len(n, r) tokens of a global length n are local.
//
// Kernels (all PDL: griddepcontrol.wait before any predecessor output is read):
//   prep_kernel   once per forward: epoch += 1; local page table, local contexts [b, q], local causal lengths [rows]
//   merge_kernel  per DSA layer, one CTA of 32 * W threads per row (warp q owns source q):
//                 own local top-2048 (raw local indices from topk_v2 + this rank's local logits) -> 64-bit keys
//                 (score desc, then position asc; a strict total order); the full list -> this rank's own FULL
//                 region (peers read it only on fallback); own top-C -> every peer's CAND region; poll the W x C
//                 candidates; t* = the min(n_cand, 2048)-th key; stock certificate + exact fallback (failing
//                 sources' FULL lists read through peer pointers); output every key >= t* as KV slots
//                 table[b, pos / 64] * 64 + pos % 64 sorted by position (block radix sort; K31_TSI_EMIT 3: source-major,
//                 see below), -1 after; fallback flag.
//                 launch_dependents after the last remote store is issued.
//   fill_kernel   tests/benchmarks only: tagged records of virtual sources.
//
// Synchronization (memory-model audit): every cross-GPU datum is
// a 64-bit record {payload32 low, tag32 high} written with st.relaxed.sys.global.v2.u64 through the destination's
// unicast VA and polled with ld.relaxed.sys.global.v2.u64 until both elements carry the expected tag. No sentinel, no
// re-arm, no fence. The tag is unique per (forward epoch, DSA ordinal, kind); regions alternate parity by ordinal.
// A rank never pushes to itself and never polls itself (own candidates come from shared memory, own fallback list
// from registers), so no CTA waits on a CTA of its own grid. Every poll traps at ``deadline_ns``: no output, no
// fallback flag, no epoch change on that path (outputs are written only after the last poll).
#pragma once

#ifndef K31_TSI_PREFIX_SKIP
#define K31_TSI_PREFIX_SKIP 0
#endif
#ifndef K31_TSI_EMIT
#define K31_TSI_EMIT 0
#endif
#define K31_TSI_EMIT_BITMAP (K31_TSI_EMIT == 1)
#include <algorithm>
#include <cstdint>
#if !K31_TSI_EMIT_BITMAP
#include <cub/block/block_radix_sort.cuh>
#endif

// K31_TSI_T1L 1 ("T1-lite", tsi.py knob K31MK_TSI_T1L=1; needs K31_TSI_EMIT 3): the scan's local top-k is the local
// top-C (topk_v2 at k = C, raw [rows, C]), and that list IS the sender's candidate set (no select over 2048 keys). The
// FULL list (local top-2048) is no longer written before the push: after the poll and the certificate, a rank whose own
// certificate failed on a row computes it from its local logits and publishes it (publish_full); peers poll it by tag
// as before. Every rank derives the same fail mask from the same W x C candidates. Phases: 1 push, 2 poll + certificate
// + publish, 4 fallback + output (serving: 7 in one launch; the 1-GPU emulation launches 1, 2, 4 for every rank in turn,
// so that every FULL list a fallback reads is published before any rank reads it).
#ifndef K31_TSI_T1L
#define K31_TSI_T1L 0
#endif
#if K31_TSI_T1L && K31_TSI_EMIT != 3
#error "K31_TSI_T1L needs K31_TSI_EMIT 3 (s1r)"
#endif
// Merge variants (benchmarks build the others as separate modules; the shipped default is the tested one):
//   K31_TSI_PREFIX_SKIP 1: select_kth starts at the highest byte where the keys differ; 0: at the top byte (stock)
//   K31_TSI_EMIT 0: compaction + block radix sort (cub); 1: SMEM bitmap; 2: bucket counting sort by position (2048
//                   buckets, rank inside a bucket by counting), the radix sort only when a bucket holds > 32 keys
//   (bench_tsi.py variants: prefix skip, the bitmap and the buckets were all slower than the stock
//   select + radix sort, which is the default)
//   K31_TSI_EMIT 3 ("s1r", tsi.py knob K31MK_TSI_S1R=1; measured faster per merge at
//                   131K-786K tokens, rows 8): the SENDER ranks its own top-C by position (rank counting, two threads per
//                   candidate; positions are unique) before the push, so every source's candidate array is
//                   position-ascending. Non-fallback rows then emit without a block sort: warp q ballot-compacts its kept
//                   keys after the kept keys of sources < q -> SOURCE-MAJOR order (source q = (pos / 64) % W ascending,
//                   position ascending inside a source), deterministic (a function of the selected set) but not the
//                   position order of 0: the sparse core's summation order changes (REORDER class vs 0). Fallback rows
//                   keep the radix sort (position order). Needs 32 * W == 2 * C (W 16, C 256).

namespace k31_tsi {

constexpr int kPage = 64;
constexpr int kTopK = 2048;
constexpr int kMaxWorld = 16;
constexpr int kKinds = 2;
constexpr int kKindCand = 0, kKindFull = 1;
constexpr int kPrepThreads = 256;
constexpr uint32_t kEmptyPos = 0xFFFFFFFFu;

// ------------------------------------------------------------------------------------------------ primitives
__device__ __forceinline__ void pdl_wait() { asm volatile("griddepcontrol.wait;" ::: "memory"); }
__device__ __forceinline__ void pdl_trigger() { asm volatile("griddepcontrol.launch_dependents;" ::: "memory"); }
__device__ __forceinline__ uint64_t gtime() {
  uint64_t t;
  asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
  return t;
}
__device__ __forceinline__ uint64_t rec(uint32_t payload, uint32_t tag) {
  return (static_cast<uint64_t>(tag) << 32) | payload;
}
__device__ __forceinline__ uint32_t lo32(uint64_t r) { return static_cast<uint32_t>(r); }
__device__ __forceinline__ uint32_t hi32(uint64_t r) { return static_cast<uint32_t>(r >> 32); }
// Two records (16 B aligned) to a peer's (or this rank's) unicast address; system scope (the reader is another GPU).
__device__ __forceinline__ void st_rec2(uint64_t addr, uint64_t a, uint64_t b) {
  asm volatile("st.relaxed.sys.global.v2.u64 [%0], {%1, %2};" ::"l"(addr), "l"(a), "l"(b) : "memory");
}
__device__ __forceinline__ void ld_rec2(const uint64_t* p, uint64_t& a, uint64_t& b) {
  asm volatile("ld.relaxed.sys.global.v2.u64 {%0, %1}, [%2];" : "=l"(a), "=l"(b) : "l"(p) : "memory");
}
// A device-memory value a previous grid of this stream wrote (read after griddepcontrol.wait; never ld.global.nc).
__device__ __forceinline__ uint32_t ld_gpu_u32(const uint32_t* p) {
  uint32_t v;
  asm volatile("ld.relaxed.gpu.global.u32 %0, [%1];" : "=r"(v) : "l"(p) : "memory");
  return v;
}
__device__ __forceinline__ bool tag_ok(uint32_t seen, uint32_t want) {
  return seen == want;
}
// Poll kN record pairs (``stride`` records apart from ``p``) until every pair carries ``want`` in both records; all
// loads of a round are in flight together. Traps ``deadline_ns`` after ``start`` (fail closed: nothing after it runs).
template <int kN>
__device__ __forceinline__ void poll_pairs(const uint64_t* p, int64_t stride, uint32_t want, uint64_t start,
                                           uint64_t deadline_ns, uint64_t (&a)[kN], uint64_t (&b)[kN]) {
  static_assert(kN <= 32, "pending mask is 32 bits");
  uint32_t pending = kN == 32 ? ~0u : ((1u << kN) - 1u);
  uint32_t spins = 0;
  while (true) {
#pragma unroll
    for (int j = 0; j < kN; ++j)
      if (pending >> j & 1u) ld_rec2(p + j * stride, a[j], b[j]);
#pragma unroll
    for (int j = 0; j < kN; ++j)
      if ((pending >> j & 1u) && tag_ok(hi32(a[j]), want) && tag_ok(hi32(b[j]), want)) pending &= ~(1u << j);
    if (pending == 0u) break;
    if ((++spins & 63u) == 0 && gtime() - start > deadline_ns) __trap();
  }
}

__host__ __device__ __forceinline__ uint32_t make_tag(uint32_t epoch, int ordinal, int kind, int layers) {
  // Must equal tsi.py tag(): ((epoch * layers + ordinal) * 2 + kind + 1) mod 2^32, never 0.
  const uint32_t v = (epoch * static_cast<uint32_t>(layers) + static_cast<uint32_t>(ordinal)) * kKinds +
                     static_cast<uint32_t>(kind) + 1u;
  return v ? v : 1u;
}

// Tokens of a global length n on rank r (page p on rank p % world): the full local pages, plus the partial last page
// when this rank owns it.
__host__ __device__ __forceinline__ int local_len(int n, int world, int r) {
  if (n <= 0) return 0;
  const int pages = n / kPage, rem = n % kPage;
  const int full = (pages + world - 1 - r) / world;
  return full * kPage + (pages % world == r ? rem : 0);
}

__device__ __forceinline__ int global_pos(int l, int world, int r) {
  return (world * (l / kPage) + r) * kPage + l % kPage;
}

// ------------------------------------------------------------------------------------------------ keys (stock)
__device__ __forceinline__ uint32_t ordered_bits(uint32_t bits) {
  return (bits & 0x80000000u) ? ~bits : (bits | 0x80000000u);
}
// (score bits, global position) -> 64-bit key, larger is better; 0 for an empty slot (position < 0)
__device__ __forceinline__ uint64_t make_key(uint32_t score_bits, int32_t pos) {
  if (pos < 0) return 0;
  return (static_cast<uint64_t>(ordered_bits(score_bits)) << 32) | static_cast<uint32_t>(~static_cast<uint32_t>(pos));
}
__device__ __forceinline__ uint32_t key_pos(uint64_t k) { return ~static_cast<uint32_t>(k); }

// ------------------------------------------------------------------------------------------------ region layout
// Records (int64) of one rank's region:
//   CAND [2 parity][W src][cap rows][C cand][2]      (score bits, position) pushed by source src
//   FULL [2 parity][cap rows][2048][2]                this rank's own full local top-2048 (peers read on fallback)
template <int kWorld, int kCand>
__host__ __device__ __forceinline__ int64_t cand_index(int parity, int src, int row, int e, int cap) {
  return ((((static_cast<int64_t>(parity) * kWorld + src) * cap + row) * kCand) + e) * 2;
}
template <int kWorld, int kCand>
__host__ __device__ __forceinline__ int64_t full_index(int parity, int row, int i, int cap) {
  return 2LL * kWorld * cap * kCand * 2 + ((static_cast<int64_t>(parity) * cap + row) * kTopK + i) * 2;
}
template <int kWorld, int kCand>
__host__ __device__ __forceinline__ int64_t region_records(int cap) {
  return 2LL * kWorld * cap * kCand * 2 + 2LL * cap * kTopK * 2;
}

// ------------------------------------------------------------------------------------------------ block helpers
template <int kThreads, int kCand>
struct Smem {
  static constexpr int kWarps = kThreads / 32;
  int hist[2][256];
  int warp_sum[kWarps];
  int digit, above, cnt;
  unsigned long long min_key;
  uint64_t wmax[kWarps], wmin[kWarps];
  int sent[kWarps], kept[kWarps];
  uint2 cpay[kCand];      // own candidates (score bits, position), compacted, (0, -1) padded
  uint64_t ckey[kCand];   // their keys (0 padded)
#if K31_TSI_EMIT == 3
  uint16_t order[kCand];  // own candidate indices in position order (the push and the own source read through it)
  alignas(16) uint32_t cposv[kCand];  // own candidate positions (padding 0x80000000 | e: unique, above all)
#endif
#if !K31_TSI_EMIT_BITMAP
  using Sort = cub::BlockRadixSort<uint32_t, kThreads, kTopK / kThreads>;
  static constexpr int kBuckets = 2048;
  union {
    uint32_t pos[kTopK];  // selected positions, compacted, kEmptyPos padded
    typename Sort::TempStorage sort;
#if K31_TSI_EMIT == 2
    struct {
      int count[kBuckets];   // keys per position bucket
      int end[kBuckets];     // exclusive scan, then (after the scatter) bucket end
      uint32_t arr[kTopK];   // positions scattered by bucket
    } bk;
#endif
  };
#if K31_TSI_EMIT == 2
  int max_count;
#endif
#endif
};

__device__ __forceinline__ int warp_inclusive_sum(int v, int lane) {
#pragma unroll
  for (int o = 1; o < 32; o <<= 1) {
    const int x = __shfl_up_sync(0xffffffffu, v, o);
    if (lane >= o) v += x;
  }
  return v;
}
__device__ __forceinline__ uint64_t warp_reduce_max(uint64_t v) {
#pragma unroll
  for (int o = 16; o > 0; o >>= 1) {
    const uint64_t x = __shfl_xor_sync(0xffffffffu, v, o);
    v = x > v ? x : v;
  }
  return v;
}
__device__ __forceinline__ uint64_t warp_reduce_min(uint64_t v) {
#pragma unroll
  for (int o = 16; o > 0; o >>= 1) {
    const uint64_t x = __shfl_xor_sync(0xffffffffu, v, o);
    v = x < v ? x : v;
  }
  return v;
}

template <int kN>
__device__ __forceinline__ auto visit_keys(const uint64_t (&keys)[kN]) {
  return [&keys](auto f) {
#pragma unroll
    for (int i = 0; i < kN; ++i) f(keys[i]);
  };
}

// Exclusive prefix sum over the CTA (one value per thread); ``total`` is the CTA sum.
template <int kThreads, class S>
__device__ __forceinline__ int block_exclusive_sum(int v, int& total, S& s) {
  constexpr int kWarps = kThreads / 32;
  const int lane = threadIdx.x % 32, warp = threadIdx.x / 32;
  const int incl = warp_inclusive_sum(v, lane);
  if (lane == 31) s.warp_sum[warp] = incl;
  __syncthreads();
  int before = 0;
  total = 0;
#pragma unroll
  for (int w = 0; w < kWarps; ++w) {
    before += w < warp ? s.warp_sum[w] : 0;
    total += s.warp_sum[w];
  }
  __syncthreads();
  return before + incl - v;
}

// Writes the CTA's selected entries in thread order to slots [0, n) through emit(slot, i), and pad(slot) to
// [n, kWidth). Slots >= kWidth are never emitted (n <= kWidth by construction; the guard keeps SMEM safe).
template <int kThreads, int kWidth, int kN, class Emit, class Pad, class S>
__device__ __forceinline__ int compact(const bool (&sel)[kN], Emit emit, Pad pad, S& s) {
  int mine = 0;
#pragma unroll
  for (int i = 0; i < kN; ++i) mine += sel[i];
  int n;
  int at = block_exclusive_sum<kThreads>(mine, n, s);
#pragma unroll
  for (int i = 0; i < kN; ++i)
    if (sel[i]) {
      if (at < kWidth) emit(at, i);
      ++at;
    }
  for (int e = n + static_cast<int>(threadIdx.x); e < kWidth; e += kThreads) pad(e);
  return n;
}

#if K31_TSI_EMIT == 3
// sender: s.order by rank counting. Thread 2e + h counts the positions below candidate e's in half h of
// s.cposv (keys unique, padding included), the pair adds through one shuffle, thread 2e writes s.order[rank] = e.
// Every thread calls it; caller: __syncthreads before (cposv complete) and after (order complete).
template <int kThreads, int kCand, class S>
__device__ __forceinline__ void rank_cand_order(S& s) {
  static_assert(kThreads == 2 * kCand && kCand % 8 == 0, "two threads per candidate");
  constexpr int kHalf = kCand / 2;
  const int tid = threadIdx.x, e = tid >> 1, h = tid & 1;
  const uint32_t pe = s.cposv[e];
  const uint4* v = reinterpret_cast<const uint4*>(s.cposv + h * kHalf);
  int r0 = 0, r1 = 0, r2 = 0, r3 = 0;
#pragma unroll 8
  for (int j = 0; j < kHalf / 4; ++j) {
    const uint4 x = v[j];
    r0 += x.x < pe;
    r1 += x.y < pe;
    r2 += x.z < pe;
    r3 += x.w < pe;
  }
  int r = (r0 + r1) + (r2 + r3);
  r += __shfl_xor_sync(0xffffffffu, r, 1);
  if (h == 0) s.order[r] = static_cast<uint16_t>(e);
}
#endif

// The rem-th largest key (rem >= 1) among the keys visited by ``visit(f)`` (stock select_kth: 8-bit radix rounds
// from the top byte, double-buffered histograms, one-warp scan; returns the bucket's smallest key once the bucket is
// exactly the remaining quota). 0 when rem < 1. Every thread of the CTA calls it.
template <int kThreads, class Visit, class S>
__device__ __forceinline__ uint64_t select_kth(Visit visit, int rem, S& s) {
  if (rem < 1) return 0;
  constexpr int kWarps = kThreads / 32;
  const int tid = threadIdx.x, warp = tid / 32, lane = tid % 32;
#if K31_TSI_PREFIX_SKIP
  // The bytes every nonzero key shares are skipped: the radix starts at the byte of the highest bit where the largest
  // and smallest key differ (the top bytes of ordered FP32 scores of one row are nearly constant).
  uint64_t mx = 0, mn = ~uint64_t{0};
  visit([&](uint64_t k) {
    if (k) {
      mx = k > mx ? k : mx;
      mn = k < mn ? k : mn;
    }
  });
  mx = warp_reduce_max(mx);
  mn = warp_reduce_min(mn);
  if (lane == 0) {
    s.wmax[warp] = mx;
    s.wmin[warp] = mn;
  }
  if (tid < 256) {
    s.hist[0][tid] = 0;
    s.hist[1][tid] = 0;
  }
  __syncthreads();
  mx = 0;
  mn = ~uint64_t{0};
#pragma unroll
  for (int w = 0; w < kWarps; ++w) {
    mx = s.wmax[w] > mx ? s.wmax[w] : mx;
    mn = s.wmin[w] < mn ? s.wmin[w] : mn;
  }
  if (mx == mn) return mx;  // one distinct key (keys are unique, so rem == 1)
  const int top_byte = (63 - __clzll(static_cast<long long>(mx ^ mn))) / 8;
  uint64_t prefix = top_byte == 7 ? 0 : mx >> (8 * top_byte + 8);
#else
  if (tid < 256) {
    s.hist[0][tid] = 0;
    s.hist[1][tid] = 0;
  }
  __syncthreads();
  constexpr int top_byte = 7;
  uint64_t prefix = 0;
#endif
  int b = 0;
  for (int shift = 8 * top_byte; shift >= 0; shift -= 8, b ^= 1) {
    int* h = s.hist[b];
    visit([&](uint64_t k) {
      if (k && (shift == 56 || (k >> (shift + 8)) == prefix)) atomicAdd(&h[(k >> shift) & 255], 1);
    });
    if (tid < 256) s.hist[b ^ 1][tid] = 0;  // its last reader finished before the previous round's last barrier
    __syncthreads();
    if (warp == 0) {
      int c[8], tot = 0;
#pragma unroll
      for (int i = 0; i < 8; ++i) {
        c[i] = h[255 - (lane * 8 + i)];
        tot += c[i];
      }
      int run = warp_inclusive_sum(tot, lane) - tot;
#pragma unroll
      for (int i = 0; i < 8; ++i) {
        const int excl = run;
        run += c[i];
        if (run >= rem && excl < rem) {
          s.digit = 255 - (lane * 8 + i);
          s.above = excl;
          s.cnt = c[i];
          s.min_key = ~0ull;
        }
      }
    }
    __syncthreads();
    rem -= s.above;
    prefix = (prefix << 8) | static_cast<uint64_t>(s.digit);
    if (s.cnt == rem) {
      const uint64_t pre = prefix;
      uint64_t mk = ~uint64_t{0};
      visit([&](uint64_t k) {
        if (k && (k >> shift) == pre) mk = k < mk ? k : mk;
      });
      mk = warp_reduce_min(mk);
      if (lane == 0) atomicMin(&s.min_key, static_cast<unsigned long long>(mk));
      __syncthreads();
      return s.min_key;
    }
  }
  return prefix;
}

// ------------------------------------------------------------------------------------------------ prep
struct PrepParams {
  uint32_t* epoch;
  const int32_t* table;  // [b, pages] global page table
  int64_t table_stride;
  int pages;
  const int32_t* contexts;  // [b, q] request contexts
  const int32_t* causal;    // [rows] global causal lengths
  int32_t* ltab;            // [b, width] contiguous local page table
  int32_t* lctx;            // [b, q]
  int32_t* lcausal;         // [rows]
  int b, q, rows, width, world, rank;
};

__global__ void __launch_bounds__(kPrepThreads) prep_kernel(const __grid_constant__ PrepParams p) {
  pdl_wait();
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  const int n = gridDim.x * blockDim.x;
  if (i == 0) *p.epoch = *p.epoch + 1u;  // read by successors only after their griddepcontrol.wait
  for (int k = i; k < p.b * p.width; k += n) {
    const int bb = k / p.width, w = k % p.width;
    const int g = p.world * w + p.rank;
    p.ltab[k] = g < p.pages ? p.table[bb * p.table_stride + g] : 0;
  }
  for (int k = i; k < p.b * p.q; k += n) p.lctx[k] = local_len(p.contexts[k], p.world, p.rank);
  for (int k = i; k < p.rows; k += n) p.lcausal[k] = local_len(p.causal[k], p.world, p.rank);
  pdl_trigger();
}

// ------------------------------------------------------------------------------------------------ merge
struct MergeParams {
  const int32_t* raw;  // [rows, raw_stride] local indices of the local top-2048 (-1 padded), topk_v2 raw output
  int64_t raw_stride;
  const float* scores;  // [rows, score_stride] this rank's local logits (DeepGEMM)
  int64_t score_stride;
  const int32_t* causal;  // [rows] global causal lengths
  const int32_t* table;   // [b, pages] global page table
  int64_t table_stride;
  int32_t* out;           // [rows, 2048] KV slots sorted by position, -1 after
  int32_t* fallback;      // [rows] 1 where the certificate failed
  uint64_t* scratch;      // [cap, W * 2048] fallback union
  const uint32_t* epoch;  // device epoch counter (advanced by prep_kernel)
  uint64_t peers[kMaxWorld];  // every rank's region base (records), own included
  uint64_t deadline_ns;
  int rank, ordinal, layers, cap, phases, rows_per_request, end_bit;
  int bm_chunk;     // pages per thread of the output bitmap (ceil(table width / threads)); dynamic SMEM sized by host
};

// Fallback of a row whose certificate failed for the sources in ``fail``: the k-th key of the union in ``scratch``,
// where segment q holds source q's full local top-2048 if it failed (read from its FULL region; this rank's own from
// registers, or under T1L already in its segment from publish_full), else its candidates (zero padded). Stock fallback_kth.
template <int kWorld, int kCand, class S>
__device__ uint64_t fallback_kth(const MergeParams& p, int row, int parity, unsigned fail, uint32_t ftag, uint64_t start,
                                 const uint64_t (&own)[kTopK / (32 * kWorld)],
                                 const uint64_t (&cand)[kCand / 32], S& s) {
  constexpr int kThreads = 32 * kWorld;
  constexpr int kPer = kTopK / kThreads;
  constexpr int kPerLane = kCand / 32;
  const int tid = threadIdx.x, warp = tid / 32, lane = tid % 32;
  uint64_t* const keys = p.scratch + static_cast<int64_t>(row) * kWorld * kTopK;
#pragma unroll 1
  for (int q = 0; q < kWorld; ++q) {
    if (fail >> q & 1u) {
      if (q == p.rank) {
#if !K31_TSI_T1L
#pragma unroll
        for (int i = 0; i < kPer; ++i) keys[q * kTopK + tid + i * kThreads] = own[i];
#endif
      } else {
        const uint64_t* src =
            reinterpret_cast<const uint64_t*>(p.peers[q]) + full_index<kWorld, kCand>(parity, row, tid, p.cap);
        uint64_t a[kPer], b[kPer];
        poll_pairs<kPer>(src, 2LL * kThreads, ftag, start, p.deadline_ns, a, b);
#pragma unroll
        for (int i = 0; i < kPer; ++i)
          keys[q * kTopK + tid + i * kThreads] = make_key(lo32(a[i]), static_cast<int32_t>(lo32(b[i])));
      }
    } else if (warp == q) {
#pragma unroll
      for (int i = 0; i < kPerLane; ++i) keys[q * kTopK + lane + 32 * i] = cand[i];
      for (int e = kCand + lane; e < kTopK; e += 32) keys[q * kTopK + e] = 0;
    }
  }
  __syncthreads();
  auto visit = [&](auto f) {
#pragma unroll 4
    for (int e = tid; e < kWorld * kTopK; e += kThreads) f(keys[e]);
  };
  int mine = 0, n;
  visit([&](uint64_t k) { mine += k != 0; });
  block_exclusive_sum<kThreads>(mine, n, s);
  return select_kth<kThreads>(visit, min(n, kTopK), s);
}

#if K31_TSI_T1L
// T1L, a row whose certificate failed for this rank: its full local top-2048 by key, computed from its local logits (the
// keys >= the min(n_local, 2048)-th largest; keys are unique, so exactly that many), written to this rank's scratch
// segment of the row (fallback_kth's own segment) and, tagged ftag, to its FULL records (peers poll them). Slots past
// the list are (0, -1) as before. Every thread of the CTA calls it.
template <int kWorld, int kCand, class S>
__device__ void publish_full(const MergeParams& p, int row, int n, int parity, uint32_t ftag, S& s) {
  constexpr int kThreads = 32 * kWorld;
  const int tid = threadIdx.x;
  const int nl = local_len(n, kWorld, p.rank);
  const float* const sc = p.scores + static_cast<int64_t>(row) * p.score_stride;
  auto visit = [&](auto f) {
#pragma unroll 4
    for (int l = tid; l < nl; l += kThreads) f(make_key(__float_as_uint(sc[l]), global_pos(l, kWorld, p.rank)));
  };
  const int want = min(nl, kTopK);
  const uint64_t tk = select_kth<kThreads>(visit, want, s);
  int mine = 0, total;
  visit([&](uint64_t k) { mine += tk != 0 && k >= tk; });
  int at = block_exclusive_sum<kThreads>(mine, total, s);
  uint64_t* const keys = p.scratch + static_cast<int64_t>(row) * kWorld * kTopK + static_cast<int64_t>(p.rank) * kTopK;
  uint64_t* const own_region = reinterpret_cast<uint64_t*>(p.peers[p.rank]);
  for (int l = tid; l < nl; l += kThreads) {
    const uint32_t sb = __float_as_uint(sc[l]);
    const int32_t ps = global_pos(l, kWorld, p.rank);
    const uint64_t k = make_key(sb, ps);
    if (tk != 0 && k >= tk) {
      if (at < kTopK) {
        keys[at] = k;
        st_rec2(reinterpret_cast<uint64_t>(own_region + full_index<kWorld, kCand>(parity, row, at, p.cap)), rec(sb, ftag),
                rec(static_cast<uint32_t>(ps), ftag));
      }
      ++at;
    }
  }
  for (int e = min(total, kTopK) + tid; e < kTopK; e += kThreads) {
    keys[e] = 0;
    st_rec2(reinterpret_cast<uint64_t>(own_region + full_index<kWorld, kCand>(parity, row, e, p.cap)), rec(0u, ftag),
            rec(kEmptyPos, ftag));
  }
}
#endif

template <int kWorld, int kCand>
__global__ void __launch_bounds__(32 * kWorld) merge_kernel(const __grid_constant__ MergeParams p) {
  constexpr int kThreads = 32 * kWorld;
  constexpr int kPer = kTopK / kThreads;  // local top-2048 entries per thread
  constexpr int kPerLane = kCand / 32;    // candidates per lane
  static_assert(kWorld >= 2 && kWorld <= kMaxWorld && kTopK % kThreads == 0, "world");
  static_assert(kCand % 32 == 0 && kCand <= kTopK && kPerLane <= 32, "candidates");
  using S = Smem<kThreads, kCand>;
  __shared__ S s;

  const int row = blockIdx.x, tid = threadIdx.x, warp = tid / 32, lane = tid % 32;
  pdl_wait();
  const uint32_t epoch = ld_gpu_u32(p.epoch);
  const uint32_t ctag = make_tag(epoch, p.ordinal, kKindCand, p.layers);
  const uint32_t ftag = make_tag(epoch, p.ordinal, kKindFull, p.layers);
  const int parity = p.ordinal & 1;
  uint64_t* const own_region = reinterpret_cast<uint64_t*>(p.peers[p.rank]);

#if K31_TSI_T1L
  // ---- T1L: this rank's local top-C (raw, topk_v2 at k = C) = its candidates -> keys, compacted into SMEM ----
  uint64_t own[kPer] = {};  // unused under T1L (fallback_kth reads this rank's segment that publish_full wrote)
  {
    uint32_t sb0 = 0u;
    int32_t ps0 = -1;
    if (tid < kCand) {
      int l = p.raw[static_cast<int64_t>(row) * p.raw_stride + tid];
      if (l >= 0) {
        sb0 = __float_as_uint(p.scores[static_cast<int64_t>(row) * p.score_stride + l]);
        ps0 = global_pos(l, kWorld, p.rank);
      }
    }
    const uint64_t k0 = make_key(sb0, ps0);
    const bool sel[1] = {k0 != 0};
    compact<kThreads, kCand>(
        sel,
        [&](int e, int) {
          s.cpay[e] = make_uint2(sb0, static_cast<uint32_t>(ps0));
          s.ckey[e] = k0;
          s.cposv[e] = static_cast<uint32_t>(ps0);
        },
        [&](int e) {
          s.cpay[e] = make_uint2(0u, kEmptyPos);
          s.ckey[e] = 0;
          s.cposv[e] = 0x80000000u | static_cast<uint32_t>(e);
        },
        s);
  }
#else
  // ---- this rank's local top-2048 -> (score bits, global position) -> keys ----
  uint32_t sb[kPer];
  int32_t ps[kPer];
  uint64_t own[kPer];
  int n_valid = 0;
#pragma unroll
  for (int i = 0; i < kPer; ++i) {
    const int l = p.raw[static_cast<int64_t>(row) * p.raw_stride + tid + i * kThreads];
    if (l >= 0) {
      sb[i] = __float_as_uint(p.scores[static_cast<int64_t>(row) * p.score_stride + l]);
      ps[i] = global_pos(l, kWorld, p.rank);
    } else {
      sb[i] = 0u;
      ps[i] = -1;
    }
    own[i] = make_key(sb[i], ps[i]);
    n_valid += own[i] != 0;
  }
  int n_own;
  block_exclusive_sum<kThreads>(n_valid, n_own, s);
  if (p.phases & 1) {
#pragma unroll
    for (int i = 0; i < kPer; ++i)
      st_rec2(reinterpret_cast<uint64_t>(own_region + full_index<kWorld, kCand>(parity, row, tid + i * kThreads, p.cap)),
              rec(sb[i], ftag), rec(static_cast<uint32_t>(ps[i]), ftag));
  }

  // ---- own top-C, compacted into SMEM, pushed to every peer ----
  {
    bool sel[kPer];
    const uint64_t tc = select_kth<kThreads>(visit_keys(own), min(n_own, kCand), s);
#pragma unroll
    for (int i = 0; i < kPer; ++i) sel[i] = tc != 0 && own[i] >= tc;
    compact<kThreads, kCand>(
        sel,
        [&](int e, int i) {
          s.cpay[e] = make_uint2(sb[i], static_cast<uint32_t>(ps[i]));
          s.ckey[e] = own[i];
#if K31_TSI_EMIT == 3
          s.cposv[e] = static_cast<uint32_t>(ps[i]);
#endif
        },
        [&](int e) {
          s.cpay[e] = make_uint2(0u, kEmptyPos);
          s.ckey[e] = 0;
#if K31_TSI_EMIT == 3
          s.cposv[e] = 0x80000000u | static_cast<uint32_t>(e);
#endif
        },
        s);
  }
#endif  // K31_TSI_T1L
  __syncthreads();
#if K31_TSI_EMIT == 3
  rank_cand_order<kThreads, kCand>(s);
  __syncthreads();  // s.order complete before the push and the own-source read
#define TSI_ORD(e) static_cast<int>(s.order[(e)])
#else
#define TSI_ORD(e) (e)
#endif
  if (p.phases & 1) {
    for (int i = tid; i < (kWorld - 1) * kCand; i += kThreads) {
      const int d = (p.rank + 1 + i / kCand) % kWorld, e = i % kCand;
      const uint2 v = s.cpay[TSI_ORD(e)];
      st_rec2(p.peers[d] + 8 * cand_index<kWorld, kCand>(parity, p.rank, row, e, p.cap), rec(v.x, ctag),
              rec(v.y, ctag));
    }
  }
  pdl_trigger();  // after the last remote store is issued (dependents still wait for this grid's completion)
#if K31_TSI_T1L
  if (!(p.phases & 6)) return;
#else
  if (!(p.phases & 2)) return;
#endif

  // ---- poll the W x C candidates (warp q: source q; own source from SMEM) ----
  const uint64_t start = gtime();
  uint64_t cand[kPerLane];
  if (warp == p.rank) {
#pragma unroll
    for (int i = 0; i < kPerLane; ++i) cand[i] = s.ckey[TSI_ORD(lane + 32 * i)];
  } else {
    uint64_t a[kPerLane], b[kPerLane];
    poll_pairs<kPerLane>(own_region + cand_index<kWorld, kCand>(parity, warp, row, lane, p.cap), 64, ctag, start,
                         p.deadline_ns, a, b);
#pragma unroll
    for (int i = 0; i < kPerLane; ++i) cand[i] = make_key(lo32(a[i]), static_cast<int32_t>(lo32(b[i])));
  }
  int sent = 0;
#pragma unroll
  for (int i = 0; i < kPerLane; ++i) sent += __popc(__ballot_sync(0xffffffffu, cand[i] != 0));
  if (lane == 0) s.sent[warp] = sent;
  __syncthreads();
  int n_cand = 0;
#pragma unroll
  for (int q = 0; q < kWorld; ++q) n_cand += s.sent[q];
  const int want = min(n_cand, kTopK);
  uint64_t t = select_kth<kThreads>(visit_keys(cand), want, s);
  int kept = 0;
#pragma unroll
  for (int i = 0; i < kPerLane; ++i) kept += __popc(__ballot_sync(0xffffffffu, cand[i] != 0 && cand[i] >= t));
  if (lane == 0) s.kept[warp] = kept;
  __syncthreads();

  // ---- certificate (stock): source q may hold more of the global top-k only if it sent everything >= t*, sent
  // fewer than its local length, i.e. it sent C ----
  const int n = p.causal[row];
  unsigned fail = 0;
#pragma unroll
  for (int q = 0; q < kWorld; ++q)
    if (want > 0 && s.kept[q] == s.sent[q] && local_len(n, kWorld, q) > s.sent[q]) fail |= 1u << q;
#if K31_TSI_T1L
  if ((p.phases & 2) && (fail >> p.rank & 1u)) publish_full<kWorld, kCand>(p, row, n, parity, ftag, s);
  if (!(p.phases & 4)) return;
#endif
  if (fail) t = fallback_kth<kWorld, kCand>(p, row, parity, fail, ftag, start, own, cand, s);

  const int32_t* table = p.table + static_cast<int64_t>(row / p.rows_per_request) * p.table_stride;
  int32_t* out = p.out + static_cast<int64_t>(row) * kTopK;
#if !K31_TSI_EMIT_BITMAP
  bool done = false;
#if K31_TSI_EMIT == 2
  // ---- bucket counting sort by position: bucket = pos >> shift (<= 2048 buckets over the row's prefix); a key's slot
  // is its bucket's start plus the number of keys in its bucket with a smaller position (positions are unique), so the
  // order is the sorted one and independent of atomic order. Rows with a bucket of > 32 keys take the radix sort. ----
  {
    const int shift = max(0, 32 - __clz(max(n - 1, 1)) - 11);
    const uint64_t* keys = p.scratch + static_cast<int64_t>(row) * kWorld * kTopK;
    auto each = [&](auto f) {  // every selected position held by this thread
      if (t == 0) return;
      if (!fail) {
#pragma unroll
        for (int i = 0; i < kPerLane; ++i)
          if (cand[i] >= t) f(key_pos(cand[i]));
      } else {
        for (int e = tid; e < kWorld * kTopK; e += kThreads)
          if (keys[e] >= t) f(key_pos(keys[e]));
      }
    };
    for (int b = tid; b < S::kBuckets; b += kThreads) s.bk.count[b] = 0;
    if (tid == 0) s.max_count = 0;
    __syncthreads();
    each([&](uint32_t pos) { atomicAdd(&s.bk.count[min(static_cast<int>(pos >> shift), S::kBuckets - 1)], 1); });
    __syncthreads();
    constexpr int kPerT = S::kBuckets / kThreads;
    int c[kPerT], sum = 0, mx = 0;
#pragma unroll
    for (int j = 0; j < kPerT; ++j) {
      c[j] = s.bk.count[tid * kPerT + j];
      sum += c[j];
      mx = max(mx, c[j]);
    }
    int total;
    int run = block_exclusive_sum<kThreads>(sum, total, s);
    if (mx > 0) atomicMax(&s.max_count, mx);
#pragma unroll
    for (int j = 0; j < kPerT; ++j) {
      s.bk.end[tid * kPerT + j] = run;
      run += c[j];
    }
    __syncthreads();
    if (s.max_count <= 32) {
      each([&](uint32_t pos) {
        const int b = min(static_cast<int>(pos >> shift), S::kBuckets - 1);
        s.bk.arr[atomicAdd(&s.bk.end[b], 1)] = pos;
      });
      __syncthreads();
      each([&](uint32_t pos) {
        const int b = min(static_cast<int>(pos >> shift), S::kBuckets - 1);
        const int cnt = s.bk.count[b], start = s.bk.end[b] - cnt;
        int r = 0;
        for (int k = 0; k < cnt; ++k) r += s.bk.arr[start + k] < pos;
        if (start + r < kTopK) out[start + r] = table[pos / kPage] * kPage + static_cast<int32_t>(pos % kPage);
      });
      for (int e = min(total, kTopK) + tid; e < kTopK; e += kThreads) out[e] = -1;
      done = true;
    }
    __syncthreads();  // the union is reused by the radix-sort path
  }
#endif
#if K31_TSI_EMIT == 3
  // ---- s1r, non-fallback rows: every source's candidates arrive position-ascending (sender ranking); warp q compacts
  // its kept keys in element order after the kept keys of sources < q (s.kept, complete since the barrier above) ----
  if (!fail) {
    int bnd[kWorld + 1];
    bnd[0] = 0;
#pragma unroll
    for (int q = 0; q < kWorld; ++q) bnd[q + 1] = bnd[q] + s.kept[q];
    int base = 0;
#pragma unroll
    for (int q = 0; q < kWorld; ++q) base = q == warp ? bnd[q] : base;
    const unsigned lt = (1u << lane) - 1u;
#pragma unroll
    for (int i = 0; i < kPerLane; ++i) {
      const bool keep = t != 0 && cand[i] >= t;
      const unsigned m = __ballot_sync(0xffffffffu, keep);
      const int slot = base + __popc(m & lt);
      if (keep && slot < kTopK) {
        const uint32_t pos = key_pos(cand[i]);
        out[slot] = table[pos / kPage] * kPage + static_cast<int32_t>(pos % kPage);
      }
      base += __popc(m);
    }
    for (int e = min(bnd[kWorld], kTopK) + tid; e < kTopK; e += kThreads) out[e] = -1;
    done = true;
  }
#endif
  if (!done) {
  // ---- every key >= t*, positions compacted into SMEM (kEmptyPos padded), block radix sort, KV slots ----
  if (!fail) {
    bool sel[kPerLane];
#pragma unroll
    for (int i = 0; i < kPerLane; ++i) sel[i] = t != 0 && cand[i] >= t;
    compact<kThreads, kTopK>(
        sel, [&](int e, int i) { s.pos[e] = key_pos(cand[i]); }, [&](int e) { s.pos[e] = kEmptyPos; }, s);
  } else {
    const uint64_t* keys = p.scratch + static_cast<int64_t>(row) * kWorld * kTopK;
    int mine = 0, total;
    for (int e = tid; e < kWorld * kTopK; e += kThreads) mine += t != 0 && keys[e] >= t;
    int at = block_exclusive_sum<kThreads>(mine, total, s);
    for (int e = tid; e < kWorld * kTopK; e += kThreads) {
      const uint64_t k = keys[e];
      if (t != 0 && k >= t) {
        if (at < kTopK) s.pos[at] = key_pos(k);
        ++at;
      }
    }
    for (int e = total + tid; e < kTopK; e += kThreads) s.pos[e] = kEmptyPos;
  }
  __syncthreads();
  uint32_t items[kPer];
#pragma unroll
  for (int j = 0; j < kPer; ++j) items[j] = s.pos[tid * kPer + j];
  __syncthreads();  // the SMEM union is reused as the sort's temporary storage
  typename S::Sort sorter(s.sort);
  sorter.SortBlockedToStriped(items, 0, p.end_bit);
#pragma unroll
  for (int j = 0; j < kPer; ++j) {
    const uint32_t pos = items[j];
    out[j * kThreads + tid] = pos == kEmptyPos ? -1 : table[pos / kPage] * kPage + static_cast<int32_t>(pos % kPage);
  }
  }
#else
  // ---- every key >= t* -> one bit per position in a SMEM bitmap (one u64 word per page); thread q owns the
  // contiguous pages [q * chunk, (q + 1) * chunk) at SMEM words q * stride + j (stride odd: conflict-free walks);
  // a block scan of the per-thread popcounts gives each thread its output offset, and every thread emits its pages'
  // slots in position order: the sorted, deterministic order of the sparse core's sum with no comparison sort ----
  extern __shared__ __align__(16) unsigned long long bm[];
  const int words = (n + kPage - 1) / kPage;  // pages of this row's causal prefix (every selected position is < n)
  const int chunk = p.bm_chunk, stride = chunk | 1;
  for (int j = 0; j < chunk; ++j)
    if (tid * chunk + j < words) bm[tid * stride + j] = 0ull;
  __syncthreads();
  auto mark = [&](uint32_t pos) {
    const int w = static_cast<int>(pos / kPage);
    if (w < words) {
      const int q = w / chunk;
      atomicOr(&bm[q * stride + (w - q * chunk)], 1ull << (pos % kPage));
    }
  };
  if (t != 0) {
    if (!fail) {
#pragma unroll
      for (int i = 0; i < kPerLane; ++i)
        if (cand[i] >= t) mark(key_pos(cand[i]));
    } else {
      const uint64_t* keys = p.scratch + static_cast<int64_t>(row) * kWorld * kTopK;
      for (int e = tid; e < kWorld * kTopK; e += kThreads)
        if (keys[e] >= t) mark(key_pos(keys[e]));
    }
  }
  __syncthreads();
  int mine = 0, total;
  for (int j = 0; j < chunk; ++j)
    if (tid * chunk + j < words) mine += __popcll(bm[tid * stride + j]);
  int at = block_exclusive_sum<kThreads>(mine, total, s);
  for (int j = 0; j < chunk; ++j) {
    const int w = tid * chunk + j;
    if (w >= words) break;
    unsigned long long m = bm[tid * stride + j];
    if (m == 0ull) continue;
    const int base = table[w] * kPage;
    do {
      const int b = __ffsll(static_cast<long long>(m)) - 1;
      m &= m - 1;
      if (at < kTopK) out[at] = base + b;
      ++at;
    } while (m);
  }
  for (int e = min(total, kTopK) + tid; e < kTopK; e += kThreads) out[e] = -1;
#endif
  if (tid == 0) p.fallback[row] = fail != 0;
}

// ------------------------------------------------------------------------------------------------ fill (tests)
// Tagged records of the current (epoch, ordinal) into ``region``: kind CAND -> [parity][src_lo + s][row][e] for
// payload [nsrc, rows, n, 2]; kind FULL -> [parity][row][e] for payload [1, rows, n, 2]. Benchmarks use it for
// virtual ranks; serving never launches it.
struct FillParams {
  uint64_t* region;
  const uint32_t* epoch;
  const int32_t* payload;
  int nsrc, src_lo, rows, n, kind, ordinal, layers, cap;
};

template <int kWorld, int kCand>
__global__ void __launch_bounds__(256) fill_kernel(const __grid_constant__ FillParams p) {
  pdl_wait();
  const uint32_t tag = make_tag(ld_gpu_u32(p.epoch), p.ordinal, p.kind, p.layers);
  const int parity = p.ordinal & 1;
  const int64_t total = static_cast<int64_t>(p.nsrc) * p.rows * p.n;
  for (int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x; i < total;
       i += static_cast<int64_t>(gridDim.x) * blockDim.x) {
    const int e = static_cast<int>(i % p.n), row = static_cast<int>((i / p.n) % p.rows);
    const int src = static_cast<int>(i / (static_cast<int64_t>(p.n) * p.rows));
    const int2 v = reinterpret_cast<const int2*>(p.payload)[i];
    const int64_t at = p.kind == kKindCand ? cand_index<kWorld, kCand>(parity, p.src_lo + src, row, e, p.cap)
                                           : full_index<kWorld, kCand>(parity, row, e, p.cap);
    st_rec2(reinterpret_cast<uint64_t>(p.region + at), rec(static_cast<uint32_t>(v.x), tag),
            rec(static_cast<uint32_t>(v.y), tag));
  }
  pdl_trigger();
}

}  // namespace k31_tsi

#ifdef K31_TSI_PTX_INSTANTIATE
// Device-only build for PTX inspection (nvcc -ptx -DK31_TSI_PTX_INSTANTIATE -DK31_TSI_DEVICE_ONLY): the shipped
// instantiations.
template __global__ void k31_tsi::merge_kernel<16, 256>(const __grid_constant__ k31_tsi::MergeParams);
template __global__ void k31_tsi::fill_kernel<16, 256>(const __grid_constant__ k31_tsi::FillParams);
#endif

#ifndef K31_TSI_DEVICE_ONLY
#include <sgl_kernel/tensor.h>
#include <sgl_kernel/utils.h>

#include <sgl_kernel/runtime.cuh>
#include <sgl_kernel/utils.cuh>

#include <dlpack/dlpack.h>
#include <tvm/ffi/container/tensor.h>

namespace k31_tsi {

// Host binding. ``meta`` is a CPU int64 tensor: [world, rank, ordinal, layers, cap, deadline_ns, phases, 0,
// peer_0 .. peer_{world-1}] (region base addresses, own included). The epoch is read from device memory.
template <int kWorld, int kCand>
struct Binding {
  static constexpr int kMeta = 8;

  static const int64_t* meta_ptr(const tvm::ffi::TensorView meta) {
    using namespace sglang::host;
    const int64_t* m = static_cast<const int64_t*>(meta.data_ptr());
    RuntimeCheck(meta.device().device_type == kDLCPU && meta.size(0) == kMeta + kWorld, "TSI meta: CPU int64 [8 + W]");
    RuntimeCheck(m[0] == kWorld && m[1] >= 0 && m[1] < kWorld, "TSI meta: world/rank");
    return m;
  }

  static void check_i32(const tvm::ffi::TensorView t, const char* what) {
    using namespace sglang::host;
    RuntimeCheck(t.dtype().code == kDLInt && t.dtype().bits == 32 && t.device().device_type == kDLCUDA, what,
                 ": CUDA int32");
  }

  static void prep(const tvm::ffi::TensorView epoch, const tvm::ffi::TensorView table,
                   const tvm::ffi::TensorView contexts, const tvm::ffi::TensorView causal,
                   const tvm::ffi::TensorView ltab, const tvm::ffi::TensorView lctx,
                   const tvm::ffi::TensorView lcausal, int64_t rank) {
    using namespace sglang::host;
    check_i32(table, "table");
    check_i32(contexts, "contexts");
    check_i32(causal, "causal");
    check_i32(ltab, "ltab");
    check_i32(lctx, "lctx");
    check_i32(lcausal, "lcausal");
    RuntimeCheck(table.ndim() == 2 && table.stride(1) == 1, "table [b, pages] (unit column stride)");
    RuntimeCheck(contexts.ndim() == 2 && contexts.size(0) == table.size(0) && contexts.stride(1) == 1 &&
                     contexts.stride(0) == contexts.size(1),
                 "contexts [b, q] contiguous");
    const int b = static_cast<int>(table.size(0)), pages = static_cast<int>(table.size(1));
    const int width = (pages + kWorld - 1) / kWorld;
    RuntimeCheck(ltab.ndim() == 2 && ltab.size(0) == b && ltab.size(1) == width && ltab.stride(0) == width &&
                     ltab.stride(1) == 1,
                 "ltab [b, ceil(pages / W)] contiguous");
    RuntimeCheck(lctx.ndim() == 2 && lctx.size(0) == b && lctx.size(1) == contexts.size(1) &&
                     lctx.stride(0) == lctx.size(1) && lctx.stride(1) == 1,
                 "lctx [b, q] contiguous");
    RuntimeCheck(causal.ndim() == 1 && lcausal.ndim() == 1 && lcausal.size(0) == causal.size(0), "causal [rows]");
    RuntimeCheck(rank >= 0 && rank < kWorld, "rank");
    PrepParams p{};
    p.epoch = static_cast<uint32_t*>(epoch.data_ptr());
    p.table = static_cast<const int32_t*>(table.data_ptr());
    p.table_stride = table.stride(0);
    p.pages = pages;
    p.contexts = static_cast<const int32_t*>(contexts.data_ptr());
    p.causal = static_cast<const int32_t*>(causal.data_ptr());
    p.ltab = static_cast<int32_t*>(ltab.data_ptr());
    p.lctx = static_cast<int32_t*>(lctx.data_ptr());
    p.lcausal = static_cast<int32_t*>(lcausal.data_ptr());
    p.b = b;
    p.q = static_cast<int>(contexts.size(1));
    p.rows = static_cast<int>(causal.size(0));
    p.width = width;
    p.world = kWorld;
    p.rank = static_cast<int>(rank);
    const int work = std::max(b * width, std::max(b * p.q, p.rows));
    const int blocks = std::max(1, std::min((work + kPrepThreads - 1) / kPrepThreads, 64));
    LaunchKernel(blocks, kPrepThreads, epoch.device()).enable_pdl(true)(prep_kernel, p);
  }

  static void merge(const tvm::ffi::TensorView meta, const tvm::ffi::TensorView epoch,
                    const tvm::ffi::TensorView raw, const tvm::ffi::TensorView scores,
                    const tvm::ffi::TensorView causal, const tvm::ffi::TensorView table,
                    const tvm::ffi::TensorView out, const tvm::ffi::TensorView fallback,
                    const tvm::ffi::TensorView scratch, int64_t rows_per_request,
                    int64_t end_bit) {
    using namespace sglang::host;
    const int64_t* m = meta_ptr(meta);
    check_i32(raw, "raw");
    check_i32(causal, "causal");
    check_i32(table, "table");
    check_i32(out, "out");
    check_i32(fallback, "fallback");
    RuntimeCheck(scores.dtype().code == kDLFloat && scores.dtype().bits == 32, "scores: float32");
    RuntimeCheck(scratch.dtype().bits == 64, "scratch: int64");
    const int rows = static_cast<int>(raw.size(0));
    const int cap = static_cast<int>(m[4]);
    RuntimeCheck(rows <= cap, "TSI: rows ", rows, " exceed the region capacity ", cap);
#if K31_TSI_T1L
    RuntimeCheck(raw.ndim() == 2 && raw.size(1) == kCand && raw.stride(1) == 1, "raw [rows, C] (T1L: the local top-C)");
#else
    RuntimeCheck(raw.ndim() == 2 && raw.size(1) == kTopK && raw.stride(1) == 1, "raw [rows, 2048]");
#endif
    RuntimeCheck(scores.ndim() == 2 && scores.size(0) == rows && scores.stride(1) == 1, "scores [rows, width]");
    RuntimeCheck(causal.ndim() == 1 && causal.size(0) == rows, "causal [rows]");
    RuntimeCheck(table.ndim() == 2 && table.stride(1) == 1 && table.size(0) * rows_per_request == rows,
                 "table [rows / rows_per_request, pages]");
    RuntimeCheck(out.ndim() == 2 && out.size(0) == rows && out.size(1) == kTopK && out.stride(0) == kTopK &&
                     out.stride(1) == 1,
                 "out [rows, 2048] contiguous");
    RuntimeCheck(fallback.ndim() == 1 && fallback.size(0) >= rows, "fallback [rows]");
    RuntimeCheck(scratch.ndim() == 2 && scratch.size(0) >= rows && scratch.size(1) == kWorld * kTopK &&
                     scratch.stride(0) == kWorld * kTopK,
                 "scratch [cap, W * 2048] contiguous");
    RuntimeCheck(end_bit >= 1 && end_bit <= 31 && (table.size(1) * kPage) < (int64_t{1} << end_bit),
                 "end_bit must cover every position");
    if (rows == 0) return;
    MergeParams p{};
    p.raw = static_cast<const int32_t*>(raw.data_ptr());
    p.raw_stride = raw.stride(0);
    p.scores = static_cast<const float*>(scores.data_ptr());
    p.score_stride = scores.stride(0);
    p.causal = static_cast<const int32_t*>(causal.data_ptr());
    p.table = static_cast<const int32_t*>(table.data_ptr());
    p.table_stride = table.stride(0);
    p.out = static_cast<int32_t*>(out.data_ptr());
    p.fallback = static_cast<int32_t*>(fallback.data_ptr());
    p.scratch = static_cast<uint64_t*>(scratch.data_ptr());
    p.epoch = static_cast<const uint32_t*>(epoch.data_ptr());
    for (int i = 0; i < kWorld; ++i) p.peers[i] = static_cast<uint64_t>(m[kMeta + i]);
    p.rank = static_cast<int>(m[1]);
    p.ordinal = static_cast<int>(m[2]);
    p.layers = static_cast<int>(m[3]);
    p.cap = cap;
    p.deadline_ns = static_cast<uint64_t>(m[5]);
    p.phases = static_cast<int>(m[6]);
    p.rows_per_request = static_cast<int>(rows_per_request);
    p.end_bit = static_cast<int>(end_bit);
    // Output bitmap: one u64 word per page of the table, ceil(pages / threads) contiguous pages per thread, each
    // thread's words at an odd stride.
    constexpr int kThreads = 32 * kWorld;
    const int64_t pages = table.size(1);
    p.bm_chunk = static_cast<int>(std::max<int64_t>(1, (pages + kThreads - 1) / kThreads));
    const size_t smem = K31_TSI_EMIT_BITMAP ? static_cast<size_t>(kThreads) * (p.bm_chunk | 1) * sizeof(unsigned long long) : 0;
    if (smem > 0) {
      // Checked on the kernel itself every launch (a cached static would be shared by every JIT module that
      // instantiates this binding: GNU-unique template statics).
      cudaFuncAttributes attr{};
      RuntimeDeviceCheck(cudaFuncGetAttributes(&attr, merge_kernel<kWorld, kCand>));
      if (static_cast<size_t>(attr.maxDynamicSharedSizeBytes) < smem) {
        int dev = 0, optin = 0;
        RuntimeDeviceCheck(cudaGetDevice(&dev));
        RuntimeDeviceCheck(cudaDeviceGetAttribute(&optin, cudaDevAttrMaxSharedMemoryPerBlockOptin, dev));
        RuntimeCheck(smem + attr.sharedSizeBytes <= static_cast<size_t>(optin), "TSI merge: a table of ", pages,
                     " pages needs ", smem, " B of dynamic SMEM (+", attr.sharedSizeBytes, " static) > ", optin);
        RuntimeDeviceCheck(cudaFuncSetAttribute(merge_kernel<kWorld, kCand>,
                                                cudaFuncAttributeMaxDynamicSharedMemorySize, static_cast<int>(smem)));
      }
    }
    LaunchKernel(rows, kThreads, raw.device(), smem).enable_pdl(true)(merge_kernel<kWorld, kCand>, p);
  }

  static void fill(const tvm::ffi::TensorView meta, const tvm::ffi::TensorView epoch,
                   const tvm::ffi::TensorView payload, int64_t region, int64_t kind, int64_t src_lo) {
    using namespace sglang::host;
    const int64_t* m = meta_ptr(meta);
    check_i32(payload, "payload");
    RuntimeCheck(payload.ndim() == 4 && payload.size(3) == 2 && payload.stride(3) == 1 &&
                     payload.stride(2) == 2 && payload.stride(1) == 2 * payload.size(2) &&
                     payload.stride(0) == payload.size(1) * payload.stride(1),
                 "payload [nsrc, rows, n, 2] contiguous");
    FillParams p{};
    p.region = reinterpret_cast<uint64_t*>(region);
    p.epoch = static_cast<const uint32_t*>(epoch.data_ptr());
    p.payload = static_cast<const int32_t*>(payload.data_ptr());
    p.nsrc = static_cast<int>(payload.size(0));
    p.rows = static_cast<int>(payload.size(1));
    p.n = static_cast<int>(payload.size(2));
    p.src_lo = static_cast<int>(src_lo);
    p.kind = static_cast<int>(kind);
    p.ordinal = static_cast<int>(m[2]);
    p.layers = static_cast<int>(m[3]);
    p.cap = static_cast<int>(m[4]);
    RuntimeCheck(p.rows <= p.cap && (kind == kKindCand ? (p.n <= kCand && p.src_lo + p.nsrc <= kWorld)
                                                       : (p.n <= kTopK && p.nsrc == 1)),
                 "fill: payload shape");
    const int64_t total = static_cast<int64_t>(p.nsrc) * p.rows * p.n;
    const int blocks = static_cast<int>(std::max<int64_t>(1, std::min<int64_t>((total + 255) / 256, 256)));
    LaunchKernel(blocks, 256, payload.device()).enable_pdl(true)(fill_kernel<kWorld, kCand>, p);
  }
};

}  // namespace k31_tsi
#endif
