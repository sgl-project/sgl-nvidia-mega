// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// Portions Copyright 2023-2024 SGLang Team (Apache-2.0)
// Derived from SGLang (https://github.com/sgl-project/sglang), Apache License 2.0; see ../include/LICENSE.sglang.
// Source: SGLang python/sglang/kernels/jit/csrc/muse_spark_v12/routing.cuh (routing_kernel) at revision a55ab7d; the helpers it
// uses are included from that header unchanged.
// Modified by NVIDIA: the routing arithmetic (normalization, bias, stable per-lane sort, iterative top-k with the lowest-id
// tie-break, renormalization, packing) runs as a role inside a one-launch router kernel that computes the logits with this
// package's fixed-split TF32 router contract; an N-sharded variant selects the top-k from per-shard candidate lists with the
// same order and the same weight arithmetic.
//
// Spread router role (probe form; dev, not wired into the server).
//   logits[M, 256] = x[M, 8192] (fp32) @ w[256, 8192]^T (fp32), router contract: the K range is cut into 16 chunks of 512
//   columns; each chunk partial is one mma.sync.m16n8k8 TF32 accumulator chain (operands rounded with cvt.rna.tf32.f32) over the
//   chunk's 64 k8 steps in ascending order from +0; the 16 partials are summed in chunk order in fp32 (P0, + P1, + P2, ...).
//   Results for a row depend only on that row (batch invariant). PTX ISA: mma.sync, cvt.rna.tf32.f32, ldmatrix.
// One launch, one CTA per SM slot, warp roles per CTA:
//   - chain warps: the CTA owns one K chunk and 8 expert tiles (8 experts each); their router weights are staged in shared
//     memory (already TF32-rounded, in MMA fragment order) BEFORE the go word, so after it only x is read. x arrives as a
//     chunk-major padded image (written by the producer) through a ring of bulk async copies; every chain publishes its
//     partials as tagged 64-bit words (tag = launch epoch in the high half, fp32 bits in the low half; one store width).
//   - route warps: per row, poll the 16 chunk partials of every expert (tag-in-word, no flags), sum them in chunk order,
//     then run the routing arithmetic and release the row's ready word.
//   - an optional stand-in that streams a byte share of another weight matrix from memory into L2 (probe co-load).
// Every wait is bounded; a timeout sets the error word and poisons the row (ids -1).
// Co-residency: the probe uses a cooperative launch (all CTAs resident) for the go-word wait.

#pragma once

#include "muse_spark_v12/routing.cuh"

#include <cuda.h>
#include <cuda_runtime.h>

#include <cstdint>

namespace sglang {
namespace muse_vr_k1s {

using muse_spark::div_full;
using muse_spark::kNegInf;
using muse_spark::pack_id_weight;
using muse_spark::triton_sigmoid;

inline constexpr int kK = 8192;
inline constexpr int kKC = 512;
inline constexpr int kChunks = 16;
inline constexpr int kSub = 64;                // columns per ring stage
inline constexpr int kSubs = kKC / kSub;       // 8 stages per chunk
inline constexpr int kPad = kSub + 4;          // padded row (floats) of the x image: conflict-free ldmatrix
inline constexpr int kTopK = 8;
inline constexpr int kTiles = 8;               // expert tiles (8 experts each) per CTA
inline constexpr int kChainWarps = 8;
inline constexpr int kRouteWarps = 4;
inline constexpr int kWarps = 16;              // 8 chain + 4 route + 1 ring producer, 1 stream issuer, 2 pipe stand-in
inline constexpr int kThreads = kWarps * 32;
inline constexpr int kRMax = 3;                // row tiles per chain warp
inline constexpr int kRowsMax = 96;            // x rows per CTA (6 row tiles)
inline constexpr int kNSMax = 8;               // ring stages (runtime depth from the x budget)
inline constexpr int kXBudget = 96256;          // shared-memory bytes for the x ring
inline constexpr int kWBytes = kTiles * (kKC / 8) * 32 * 8;
inline constexpr int kPoolPiece = 16384;
inline constexpr int kLogitFloats = 2 * 256;   // up to 2 route jobs per CTA
inline constexpr int kSmemBytes = kWBytes + kXBudget + kLogitFloats * 4 + (2 * kNSMax + 4) * 8 + 1024;  // + alignment slack

struct Params {
  CUtensorMap xmap;               // production form: router input [M, 8192] fp32, boxes of 16 rows x 32 columns, 128-B swizzle
  const long long* seq_word;      // production form: forward sequence word (tag = seq * 64 + layer)
  int layer;
  const float* ximg;              // probe form: [16][8][Mpad][kPad] fp32
  const float* w;                 // [256][8192] fp32
  const float* bias;              // [256] fp32
  unsigned long long* board;      // [16][Mpad][ne] tagged partial words
  float* topk_w;                  // [M][8]
  int32_t* topk_ids;              // [M][8]
  int32_t* packed;                // [M][8]
  unsigned long long* ready;      // [M][8] tagged packed ids (tag << 32 | packed)
  unsigned long long* ctl;        // [0] go, [1] staged count, [2] epoch, [3] error bits
  unsigned long long* stamps;     // [0] t0, [8 + 2 cta] chain start / end, [8 + 2 G + m] row ready
  const char* pool_src;           // stand-in source: 3 copies of pool_copy_bytes
  long long pool_copy_bytes;
  unsigned long long* pool_cnt;   // [G] bytes streamed this launch
  float* dbg;                     // [M][ne] logits (nullable)
  const unsigned long long* cand; // shard variant: [M][4][8] (key bits << 32 | id) of the other shards
  const float* cand_score;        // shard variant: [M][4][8] scores
  unsigned long long* cand_out;   // shard variant: [M][8] this shard's local list (key bits << 32 | id)
  float* cand_out_score;          // shard variant: [M][8]
  float* cand_dst;                // shard variant, candidates-only mode (nullable): [8 keys | 8 ids as float | 8 scores] of
  int cand_stride;                //   row m at cand_dst + m * cand_stride (cand_per == 0), or packed cand_per rows per line:
  int cand_per;                   //   cand_dst + (m / cand_per) * cand_line + (m % cand_per) * cand_stride (extra rows of a
  long long cand_line;            //   gather input); no merge in the kernel
  int M, Mpad, G, tg, rslab, jobs, q0, pmma_iters, pool_pace_ns, tmo_scale;
  float mult, route_scale, eps;
};

// Timing builds include a private stamp header (not part of this package); production builds compile the stamps out.
#ifdef MUSE_VR_WFC_K1S_STAMPS
#include "wfc_k1s_stamps.cuh"
#else
#define K1S_STAMP(lv) ((void)0)
#define K1S_CLOCK(lv) ((void)0)
#define K1S_NOW(v) ((v) = 0ull)
#define K1S_PACE_WAIT(t_s, i, pace) ((void)0)
#endif
inline constexpr unsigned kPollIters = 1u << 22;   // tagged-word / counter polls (each a memory round trip at least)
inline constexpr unsigned kSpinIters = 1u << 24;   // shared-memory barrier test loops
__device__ __forceinline__ uint32_t to_tf32(uint32_t bits) {
  uint32_t r;
  asm("cvt.rna.tf32.f32 %0, %1;" : "=r"(r) : "f"(__uint_as_float(bits)));
  return r;
}
__device__ __forceinline__ void mma1688(float (&d)[4], const uint32_t (&a)[4], uint32_t b0, uint32_t b1) {
  asm volatile(
      "mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
      : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}
__device__ __forceinline__ void ldsm_x4(uint32_t (&a)[4], uint32_t addr) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];"
               : "=r"(a[0]), "=r"(a[1]), "=r"(a[2]), "=r"(a[3])
               : "r"(addr));
}
__device__ __forceinline__ uint32_t smem_u32(const void* p) {
  return static_cast<uint32_t>(__cvta_generic_to_shared(p));
}
__device__ __forceinline__ void mbar_init(uint32_t bar, unsigned count) {
  asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" ::"r"(bar), "r"(count));
}
__device__ __forceinline__ void mbar_arrive(uint32_t bar) {
  asm volatile("mbarrier.arrive.release.cta.shared::cta.b64 _, [%0];" ::"r"(bar) : "memory");
}
__device__ __forceinline__ void mbar_expect_tx(uint32_t bar, unsigned bytes) {
  asm volatile("mbarrier.arrive.expect_tx.release.cta.shared::cta.b64 _, [%0], %1;" ::"r"(bar), "r"(bytes) : "memory");
}
// Long waits: try_wait with a suspend-time hint, so the waiting warp sleeps until the phase completes (or the hint
// expires) instead of spinning; the bound is an iteration count (each iteration sleeps at most the hint).
__device__ __forceinline__ bool mbar_try_sleep(uint32_t bar, unsigned parity) {
  uint32_t ok;
  asm volatile(
      "{ .reg .pred p; mbarrier.try_wait.parity.acquire.cta.shared::cta.b64 p, [%1], %2, %3; selp.u32 %0, 1, 0, p; }"
      : "=r"(ok)
      : "r"(bar), "r"(parity), "r"(1000000u)
      : "memory");
  return ok != 0;
}
inline constexpr int kSleepIters = 4000;
__device__ __forceinline__ bool mbar_try(uint32_t bar, unsigned parity) {
  uint32_t ok;
  asm volatile(
      "{ .reg .pred p; mbarrier.try_wait.parity.acquire.cta.shared::cta.b64 p, [%1], %2; selp.u32 %0, 1, 0, p; }"
      : "=r"(ok)
      : "r"(bar), "r"(parity)
      : "memory");
  return ok != 0;
}
__device__ __forceinline__ void bulk_g2s(uint32_t dst, const void* src, unsigned bytes, uint32_t bar) {
  asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];" ::"r"(dst), "l"(src),
               "r"(bytes), "r"(bar)
               : "memory");
}
__device__ __forceinline__ unsigned long long ld_acq(const unsigned long long* p) {
  unsigned long long v;
  asm volatile("ld.acquire.gpu.global.b64 %0, [%1];" : "=l"(v) : "l"(p) : "memory");
  return v;
}
__device__ __forceinline__ unsigned long long ld_rlx(const unsigned long long* p) {
  unsigned long long v;
  asm volatile("ld.relaxed.gpu.global.b64 %0, [%1];" : "=l"(v) : "l"(p) : "memory");
  return v;
}
__device__ __forceinline__ void st_rel(unsigned long long* p, unsigned long long v) {
  asm volatile("st.release.gpu.global.b64 [%0], %1;" ::"l"(p), "l"(v) : "memory");
}
__device__ __forceinline__ void st_rlx2(unsigned long long* p, unsigned long long a, unsigned long long b) {
  asm volatile("st.relaxed.gpu.global.v2.b64 [%0], {%1, %2};" ::"l"(p), "l"(a), "l"(b) : "memory");
}
__device__ __forceinline__ unsigned long long atom_add_rel(unsigned long long* p, unsigned long long v) {
  unsigned long long o;
  asm volatile("atom.add.release.gpu.global.u64 %0, [%1], %2;" : "=l"(o) : "l"(p), "l"(v) : "memory");
  return o;
}
__device__ __forceinline__ unsigned long long tagged(uint32_t tag, float v) {
  return (static_cast<unsigned long long>(tag) << 32) | __float_as_uint(v);
}
__device__ __forceinline__ void bar_arrive_named(int id, int threads) {
  asm volatile("bar.arrive %0, %1;" ::"r"(id), "r"(threads) : "memory");
}
__device__ __forceinline__ void bar_named(int id, int threads) {
  asm volatile("bar.sync %0, %1;" ::"r"(id), "r"(threads) : "memory");
}
__device__ __forceinline__ void fail(const Params& p, unsigned long long bit) {
  atomicOr(p.ctl + 3, bit);
}

// SGLang routing arithmetic for one row held in shared memory (256 logits), executed by one warp; outputs + ready word.
// The body after the logit load is SGLang's routing_kernel (topk 8, 8 experts per lane, sigmoid mode) unchanged.
__device__ __forceinline__ void route_row_256(const Params& p, const float* lg, int m, uint32_t tag, bool poisoned,
                                              const float (&biased)[8]) {
  using namespace device;
  constexpr int kEPL = 8;
  constexpr int kVecs = 2;
  constexpr int kGroupExperts = 4 * kWarpThreads;
  const int lane = threadIdx.x % kWarpThreads;
  float score[kEPL];
  int expert[kEPL];
#pragma unroll
  for (int v = 0; v < kVecs; ++v) {
    const int base = v * kGroupExperts + lane * 4;
    const float4 g = *reinterpret_cast<const float4*>(lg + base);
    score[v * 4 + 0] = g.x;
    score[v * 4 + 1] = g.y;
    score[v * 4 + 2] = g.z;
    score[v * 4 + 3] = g.w;
#pragma unroll
    for (int j = 0; j < 4; ++j) expert[v * 4 + j] = base + j;
  }
#pragma unroll
  for (int s = 0; s < kEPL; ++s) score[s] *= p.mult;
#pragma unroll
  for (int s = 0; s < kEPL; ++s) score[s] = triton_sigmoid(score[s]);
  float key[kEPL];
#pragma unroll
  for (int s = 0; s < kEPL; ++s) key[s] = biased[s] + score[s];
#pragma unroll
  for (int r = 0; r < kEPL; ++r) {
#pragma unroll
    for (int i = r % 2; i + 1 < kEPL; i += 2) {
      const bool swap = key[i + 1] > key[i];
      const float k0 = swap ? key[i + 1] : key[i];
      const float k1 = swap ? key[i] : key[i + 1];
      const int e0 = swap ? expert[i + 1] : expert[i];
      const int e1 = swap ? expert[i] : expert[i + 1];
      const float s0 = swap ? score[i + 1] : score[i];
      const float s1 = swap ? score[i] : score[i + 1];
      key[i] = k0;
      key[i + 1] = k1;
      expert[i] = e0;
      expert[i + 1] = e1;
      score[i] = s0;
      score[i + 1] = s1;
    }
  }
  float score_sum = 0.0f;
  int out_id = 0;
  float out_score = 0.0f;
#pragma unroll
  for (int k = 0; k < kTopK; ++k) {
    const float round_max = warp::reduce_max(key[0]);
    const bool is_head = key[0] == round_max;
    unsigned owners = 0;
#pragma unroll
    for (int g = kVecs - 1; g >= 0; --g) {
      const unsigned msk = __ballot_sync(warp::kFullMask, is_head && expert[0] / kGroupExperts == g);
      owners = (msk != 0u) ? msk : owners;
    }
    const int owner = __ffs(owners) - 1;
    const int winner = __shfl_sync(warp::kFullMask, expert[0], owner);
    const float selected = __shfl_sync(warp::kFullMask, score[0], owner);
    const bool pop = expert[0] == winner;
#pragma unroll
    for (int s = 0; s + 1 < kEPL; ++s)
      if (s + k <= kTopK - 2) {
        key[s] = pop ? key[s + 1] : key[s];
        expert[s] = pop ? expert[s + 1] : expert[s];
        score[s] = pop ? score[s + 1] : score[s];
      }
    if (kEPL - 1 + k <= kTopK - 2) key[kEPL - 1] = pop ? kNegInf : key[kEPL - 1];
    score_sum += selected;
    if (lane == k) {
      out_id = winner;
      out_score = selected;
    }
  }
  score_sum += p.eps;
  const float inv_sum_scaled = div_full(p.route_scale, score_sum);
  if (lane < kTopK) {
    const int32_t id = poisoned ? -1 : static_cast<int32_t>(out_id);
    const float weight = out_score * inv_sum_scaled;
    p.topk_ids[m * kTopK + lane] = id;
    p.topk_w[m * kTopK + lane] = weight;
    p.packed[m * kTopK + lane] = pack_id_weight(id, weight);
  }
  if (lane < kTopK) {
    // the consumer's word: launch tag in the high half, packed id / weight in the low half (tag-in-word, no flag)
    const unsigned long long wd = (static_cast<unsigned long long>(tag) << 32) |
                                  static_cast<uint32_t>(pack_id_weight(poisoned ? -1 : static_cast<int32_t>(out_id),
                                                                       out_score * inv_sum_scaled));
    asm volatile("st.relaxed.gpu.global.b64 [%0], %1;" ::"l"(p.ready + m * kTopK + lane), "l"(wd) : "memory");
  }
  __syncwarp();
  if (lane == 0) K1S_STAMP(p.stamps[8 + 2 * p.G + m]);
}

// Shard variant: this rank's 64 experts (lg[0..63] = experts 64 q0 ..), local top-8 by (key desc, id asc), then the merge with
// the other 3 shards' candidate lists in the same order, then SGLang's weight arithmetic in selection order.
__device__ __forceinline__ void route_row_shard(const Params& p, const float* lg, int m, uint32_t tag, bool poisoned,
                                                const float (&biased)[8]) {
  using namespace device;
  const int lane = threadIdx.x % kWarpThreads;
  const int e0 = p.q0 * 64 + lane * 2;
  float sc[2], ky[2];
#pragma unroll
  for (int j = 0; j < 2; ++j) {
    sc[j] = triton_sigmoid(lg[lane * 2 + j] * p.mult);
    ky[j] = biased[j] + sc[j];
  }
  // local top-8 (lane l < 8 keeps candidate l)
  float my_key = kNegInf, my_score = 0.f;
  int my_id = 0x7fffffff;
  bool used0 = false, used1 = false;
#pragma unroll 1
  for (int k = 0; k < kTopK; ++k) {
    // lane best: lower id first on ties (e0 < e0 + 1)
    float bk = kNegInf;
    int bid = 0x7fffffff, bj = -1;
    if (!used0) { bk = ky[0]; bid = e0; bj = 0; }
    if (!used1 && (bj < 0 || ky[1] > bk)) { bk = ky[1]; bid = e0 + 1; bj = 1; }
    const float mx = warp::reduce_max(bk);
    const int cand_id = (bj >= 0 && bk == mx) ? bid : 0x7fffffff;
    const int win = __reduce_min_sync(warp::kFullMask, cand_id);
    const bool mine = (bj >= 0 && bid == win);
    const float wsc = __shfl_sync(warp::kFullMask, bj == 0 ? sc[0] : sc[1], __ffs(__ballot_sync(warp::kFullMask, mine)) - 1);
    if (mine) { if (bj == 0) used0 = true; else used1 = true; }
    if (lane == k) { my_key = mx; my_score = wsc; my_id = win; }
  }
  if (lane < kTopK) {
    p.cand_out[m * kTopK + lane] = (static_cast<unsigned long long>(__float_as_uint(my_key)) << 32) |
                                   static_cast<uint32_t>(my_id);
    p.cand_out_score[m * kTopK + lane] = my_score;
  }
  if (p.cand_dst != nullptr) {  // candidates-only: the merge runs after the gather (k1s_merge_kernel)
    if (lane < kTopK) {
      const long long ro = p.cand_per > 0 ? static_cast<long long>(m / p.cand_per) * p.cand_line +
                                                static_cast<long long>(m % p.cand_per) * p.cand_stride
                                          : static_cast<long long>(m) * p.cand_stride;
      float* row = p.cand_dst + ro;
      row[lane] = my_key;
      row[kTopK + lane] = static_cast<float>(my_id);
      row[2 * kTopK + lane] = my_score;
    }
    __syncwarp();
    if (lane == 0) K1S_STAMP(p.stamps[8 + 2 * p.G + m]);
    return;
  }
  // merge: lane l holds candidate (shard l / 8, rank l % 8); own shard from the local list
  const int sh = lane / 8, rk = lane % 8;
  float ck, cs;
  int cid;
  const float lk = __shfl_sync(warp::kFullMask, my_key, rk);
  const float ls = __shfl_sync(warp::kFullMask, my_score, rk);
  const int li = __shfl_sync(warp::kFullMask, my_id, rk);
  if (sh == p.q0) {
    ck = lk; cs = ls; cid = li;
  } else {
    const unsigned long long c = p.cand[(static_cast<long long>(m) * 4 + sh) * 8 + rk];
    ck = __uint_as_float(static_cast<uint32_t>(c >> 32));
    cid = static_cast<int>(static_cast<uint32_t>(c));
    cs = p.cand_score[(static_cast<long long>(m) * 4 + sh) * 8 + rk];
  }
  bool taken = false;
  float score_sum = 0.0f;
  int out_id = 0;
  float out_score = 0.0f;
#pragma unroll 1
  for (int k = 0; k < kTopK; ++k) {
    const float kk = taken ? kNegInf : ck;
    const float mx = warp::reduce_max(kk);
    const int cand_id = (!taken && ck == mx) ? cid : 0x7fffffff;
    const int win = __reduce_min_sync(warp::kFullMask, cand_id);
    const bool mine = (!taken && cid == win);
    const float sel = __shfl_sync(warp::kFullMask, cs, __ffs(__ballot_sync(warp::kFullMask, mine)) - 1);
    if (mine) taken = true;
    score_sum += sel;
    if (lane == k) { out_id = win; out_score = sel; }
  }
  score_sum += p.eps;
  const float inv_sum_scaled = div_full(p.route_scale, score_sum);
  if (lane < kTopK) {
    const int32_t id = poisoned ? -1 : static_cast<int32_t>(out_id);
    const float weight = out_score * inv_sum_scaled;
    p.topk_ids[m * kTopK + lane] = id;
    p.topk_w[m * kTopK + lane] = weight;
    p.packed[m * kTopK + lane] = pack_id_weight(id, weight);
  }
  if (lane < kTopK) {
    // the consumer's word: launch tag in the high half, packed id / weight in the low half (tag-in-word, no flag)
    const unsigned long long wd = (static_cast<unsigned long long>(tag) << 32) |
                                  static_cast<uint32_t>(pack_id_weight(poisoned ? -1 : static_cast<int32_t>(out_id),
                                                                       out_score * inv_sum_scaled));
    asm volatile("st.relaxed.gpu.global.b64 [%0], %1;" ::"l"(p.ready + m * kTopK + lane), "l"(wd) : "memory");
  }
  __syncwarp();
  if (lane == 0) K1S_STAMP(p.stamps[8 + 2 * p.G + m]);
}

__device__ __forceinline__ uint2 lds64(uint32_t addr) {
  uint2 v;
  asm volatile("ld.shared.v2.u32 {%0, %1}, [%2];" : "=r"(v.x), "=r"(v.y) : "r"(addr));
  return v;
}

// One chain warp: kR row tiles x 2 expert tiles of this CTA's chunk; publishes the tagged partials. Fragments are loaded
// two k8 steps ahead (4 register buffers, index = step mod 4) across stage boundaries, so shared-memory latency overlaps
// the MMAs of the two previous steps; every address is precomputed (no division in the loop). x is already TF32-rounded by
// its producer (cvt.rna is idempotent); W was rounded when staged.
template <int kR, int kShard, int kSwz>
__device__ __forceinline__ void chain_run(const Params& p, uint32_t w_addr, uint32_t x_addr, unsigned long long* bars,
                                          int NS, int stage_bytes, int r0, int pair, int t_lo, int c, int q, int b,
                                          int warp, int lane, uint32_t tag, unsigned long long& wait_ns, int n_active) {
  constexpr int ne = kShard ? 64 : 256;
  float acc[kR][2][4];
#pragma unroll
  for (int r = 0; r < kR; ++r)
#pragma unroll
    for (int e = 0; e < 2; ++e) acc[r][e][0] = acc[r][e][1] = acc[r][e][2] = acc[r][e][3] = 0.f;
  const int lrow = ((lane >> 3) & 1) * 8 + (lane & 7);
  const int lcol = (lane >> 4) * 4;
  // probe form: padded rows (kPad floats); production form: per (row tile, 32-column half) a 16 x 128-B box, 128-B swizzled
  const uint32_t a_lane =
      kSwz ? x_addr + static_cast<uint32_t>(r0 * 2 * 2048 + lrow * 128)
           : x_addr + static_cast<uint32_t>(((r0 * 16 + lrow) * kPad + lcol) * 4);
  const int sw = lrow & 7, hi = lane >> 4;
  const uint32_t b_lane = w_addr + static_cast<uint32_t>(((pair * 2) * 64 * 32 + lane) * 8);
  uint32_t fa[4][kR][4];
  uint2 fw0[4], fw1[4];
  auto wait_slot = [&](int slot, unsigned parity, int s2) {
    const uint32_t fb = smem_u32(bars + slot);
    if (!mbar_try(fb, parity)) {
      unsigned long long t_w, t_e;
      K1S_NOW(t_w);
      unsigned it = 0;
      while (!mbar_try(fb, parity)) {
        if (++it > kSpinIters * static_cast<unsigned>(p.tmo_scale)) { fail(p, 4ull); break; }
      }
      K1S_NOW(t_e);
      if (warp == 0 && lane == 0) wait_ns += t_e - t_w;
    }
    if (p.dbg != nullptr && warp == 0 && lane == 0 && b == 0) K1S_STAMP(p.stamps[8 + 3 * p.G + 5 * p.M + 2 * s2]);
  };
  auto load_frag = [&](int buf, uint32_t a_stage, int ks, int kk) {
#pragma unroll
    for (int r = 0; r < kR; ++r) {
      if (kSwz) {
        const int j = ((kk & 3) << 1) + hi;  // 16-byte chunk within the 128-byte box row
        ldsm_x4(fa[buf][r], a_stage + static_cast<uint32_t>((r * 2 + (kk >> 2)) * 2048 + ((j ^ sw) << 4)));
#pragma unroll
        for (int t = 0; t < 4; ++t) fa[buf][r][t] = to_tf32(fa[buf][r][t]);  // raw fp32 input: the contract's operand rounding
      } else {
        ldsm_x4(fa[buf][r], a_stage + static_cast<uint32_t>(r * 16 * kPad * 4 + kk * 32));
      }
    }
    fw0[buf] = lds64(b_lane + static_cast<uint32_t>(ks * 256));
    fw1[buf] = lds64(b_lane + static_cast<uint32_t>(64 * 32 * 8 + ks * 256));
  };
  int slot = 0;
  unsigned parity = 0;
  wait_slot(0, 0, 0);
  uint32_t a_cur = a_lane;
  load_frag(0, a_cur, 0, 0);
  load_frag(1, a_cur, 1, 1);
#pragma unroll 1
  for (int s = 0; s < kSubs; ++s) {
    int nslot = slot + 1;
    unsigned nparity = parity;
    if (nslot == NS) { nslot = 0; nparity ^= 1u; }
    const uint32_t a_next = a_lane + static_cast<uint32_t>(nslot * stage_bytes);
#pragma unroll
    for (int kk = 0; kk < 8; ++kk) {
      const int nk = kk + 2;
      if (nk < 8) {
        load_frag(nk & 3, a_cur, s * 8 + nk, nk);
      } else if (s + 1 < kSubs) {
        if (nk == 8) wait_slot(nslot, nparity, s + 1);
        load_frag(nk & 3, a_next, (s + 1) * 8 + nk - 8, nk - 8);
      }
      const int cur = kk & 3;
#pragma unroll
      for (int r = 0; r < kR; ++r) {
        mma1688(acc[r][0], fa[cur][r], fw0[cur].x, fw0[cur].y);
        mma1688(acc[r][1], fa[cur][r], fw1[cur].x, fw1[cur].y);
      }
    }
    // every shared-memory read of stage s has completed (its MMAs consumed the registers): release the slot to the
    // producer through the slot's named barrier (8 + slot), only if the slot is refilled later
    __syncwarp();
    if (s + NS < kSubs) bar_arrive_named(8 + slot, (n_active + 1) * 32);
    if (p.dbg != nullptr && warp == 0 && lane == 0 && b == 0) K1S_STAMP(p.stamps[8 + 3 * p.G + 5 * p.M + 2 * s + 1]);
    slot = nslot;
    parity = nparity;
    a_cur = a_next;
  }
  // publish partials: board[c][row][expert], tagged words
  const int gq = lane >> 2, t4 = lane & 3;
#pragma unroll
  for (int r = 0; r < kR; ++r) {
#pragma unroll
    for (int e = 0; e < 2; ++e) {
      const int row0 = (t_lo + r0 + r) * 16 + gq, row1 = row0 + 8;
      const int col = (kShard ? 0 : q * 64) + (pair * 2 + e) * 8 + t4 * 2;
      if (row0 < p.M)
        st_rlx2(p.board + (static_cast<long long>(c) * p.Mpad + row0) * ne + col, tagged(tag, acc[r][e][0]),
                tagged(tag, acc[r][e][1]));
      if (row1 < p.M)
        st_rlx2(p.board + (static_cast<long long>(c) * p.Mpad + row1) * ne + col, tagged(tag, acc[r][e][2]),
                tagged(tag, acc[r][e][3]));
    }
  }
}

// kShard 0: every CTA = (chunk, row group of 3, expert quarter); 1: this rank's quarter only, 12 row groups.
// kPool 0: no stand-in; 1: byte stream; 2: byte stream + MMA-pipe stand-in.
template <int kShard, int kPool, int kProd>
__global__ void __launch_bounds__(kThreads, 1) k1s_kernel(const __grid_constant__ Params p) {
  extern __shared__ __align__(128) unsigned char smem_raw[];
  // the x ring must start on a 1024-byte boundary (128-byte swizzle pattern base): align the whole layout
  unsigned char* smem = smem_raw + ((1024u - (smem_u32(smem_raw) & 1023u)) & 1023u);
  uint2* wsm = reinterpret_cast<uint2*>(smem);
  float* xring = reinterpret_cast<float*>(smem + kWBytes);
  float* logit_s = reinterpret_cast<float*>(smem + kWBytes + kXBudget);
  unsigned long long* bars = reinterpret_cast<unsigned long long*>(logit_s + kLogitFloats);  // full[], empty[], chains_done
  __shared__ unsigned long long s_epoch;

  constexpr int ne = kShard ? 64 : 256;
  const int tid = static_cast<int>(threadIdx.x), warp = tid >> 5, lane = tid & 31;
  const int b = static_cast<int>(blockIdx.x);
  const int c = b % kChunks, grp = b / kChunks;
  const int g = kShard ? grp : grp % 3;
  const int q = kShard ? p.q0 : grp / 3;
  const int tiles_all = (p.M + 15) / 16;
  const int t_lo = g * p.tg;
  const int t_cnt = max(0, min(p.tg, tiles_all - t_lo));
  const int rows = t_cnt * 16;
  const int nslab = (t_cnt + p.rslab - 1) / max(p.rslab, 1);
  const int stage_bytes = kProd ? max(rows, 16) * 256 : max(rows, 16) * kPad * 4;
  const int NS = max(1, min(kNSMax, kXBudget / stage_bytes));
  int chain_active = 0;
  for (int w = 0; w < kChainWarps; ++w)
    if ((w / 4) < nslab) ++chain_active;

  unsigned long long go_start = 0;
  if (tid == 0) {
    go_start = ld_rlx(p.ctl + 0);
    for (int i = 0; i < NS; ++i) {
      mbar_init(smem_u32(bars + i), 1);
    }
    mbar_init(smem_u32(bars + 2 * kNSMax), max(chain_active, 1));  // this CTA's chains published
    mbar_init(smem_u32(bars + 2 * kNSMax + 1), ne / 32);  // route job 0: logit groups summed
    mbar_init(smem_u32(bars + 2 * kNSMax + 2), ne / 32);  // route job 1
    asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
  }
  // pre-go: this CTA's router weight slice, TF32-rounded, MMA fragment order: wsm[(tile * 64 + ks) * 32 + lane] = (b0, b1)
  for (int i = tid; i < kTiles * 64 * 32; i += kThreads) {
    const int tile = i >> 11, rem = i & 2047, ks = rem >> 5, ln = rem & 31;
    const int e = q * 64 + tile * 8 + (ln >> 2);
    const int k = c * kKC + ks * 8 + (ln & 3);
    const float* wr = p.w + static_cast<long long>(e) * kK + k;
    wsm[i] = make_uint2(to_tf32(__float_as_uint(wr[0])), to_tf32(__float_as_uint(wr[4])));
  }
  __syncthreads();
  if (kProd) {
    // production form: the router input is the predecessor grid's output (PDL); the tag comes from the forward word
    asm volatile("griddepcontrol.wait;" ::: "memory");
    if (tid == 0) {
      s_epoch = static_cast<unsigned long long>(*p.seq_word) * 64ull + static_cast<unsigned long long>(p.layer);
      if (b == 0) K1S_STAMP(p.stamps[0]);
    }
  } else if (tid == 0) {
    atom_add_rel(p.ctl + 1, 1ull);
    if (b == 0) {
      const unsigned long long e = p.ctl[2] + 1;  // previous launch's epoch, ordered by the kernel boundary
      unsigned it = 0;
      while (ld_acq(p.ctl + 1) < static_cast<unsigned long long>(p.G) * e) {
        if (++it > kPollIters * static_cast<unsigned>(p.tmo_scale)) { fail(p, 1ull); break; }
      }
      K1S_STAMP(p.stamps[0]);
      st_rel(p.ctl + 0, e);
      p.ctl[2] = e;
      s_epoch = e;
    } else {
      unsigned it = 0;
      unsigned long long v;
      while ((v = ld_acq(p.ctl + 0)) == go_start) {
        if (++it > kPollIters * static_cast<unsigned>(p.tmo_scale)) { fail(p, 2ull); break; }
      }
      s_epoch = v;
    }
  }
  __syncthreads();
  const unsigned long long epoch = s_epoch;
  const uint32_t tag = static_cast<uint32_t>(epoch);

  if (warp < kChainWarps) {
    // ---------------- chain warps ----------------
    const int slab = warp / 4, pair = warp % 4;
    const int r0 = slab * p.rslab;
    const int R = max(0, min(p.rslab, t_cnt - r0));
    if (R > 0) {
      unsigned long long wait_ns = 0;
      if (lane == 0 && warp == 0) {
        K1S_STAMP(p.stamps[8 + 2 * b]);
        if (b == 0) K1S_CLOCK(p.stamps[4]);
      }
      const uint32_t w_addr = smem_u32(wsm), x_addr = smem_u32(xring);
      if (R == 3)
        chain_run<3, kShard, kProd>(p, w_addr, x_addr, bars, NS, stage_bytes, r0, pair, t_lo, c, q, b, warp, lane, tag, wait_ns,
                             chain_active);
      else if (R == 2)
        chain_run<2, kShard, kProd>(p, w_addr, x_addr, bars, NS, stage_bytes, r0, pair, t_lo, c, q, b, warp, lane, tag, wait_ns,
                             chain_active);
      else
        chain_run<1, kShard, kProd>(p, w_addr, x_addr, bars, NS, stage_bytes, r0, pair, t_lo, c, q, b, warp, lane, tag, wait_ns,
                             chain_active);
      __syncwarp();
      if (lane == 0) mbar_arrive(smem_u32(bars + 2 * kNSMax));
      if (lane == 0 && warp == 0) {
        K1S_STAMP(p.stamps[8 + 2 * b + 1]);
        if (b == 0) K1S_CLOCK(p.stamps[5]);
        p.stamps[8 + 2 * p.G + 5 * p.M + b] = wait_ns;
      }
    }
  }
  if (warp < kChainWarps + kRouteWarps) {
    // ---------------- route slots (route warps first, then chain warps once their chains are published) ----------------
    // slot = (job, group of 32 experts); a warp loads the 16 chunk partials of its experts (tag-in-word, retried until
    // every word carries this launch's tag), sums them in chunk order into shared memory, and arrives on the job's
    // barrier; afterwards the warp holding a job's group 0 runs the routing arithmetic for that row.
    const int ngrp = ne / 32;
    const int nslots = p.jobs * ngrp;
    const int wr = warp >= kChainWarps ? warp - kChainWarps : warp + kRouteWarps;
    constexpr int kSlotWarps = kChainWarps + kRouteWarps;
    bool poisoned = false;
    if (warp >= kChainWarps && chain_active > 0 && wr < nslots) {
      // no polling while this CTA's own chains run: every CTA's chains run concurrently, so the remote partials land
      // about when the local ones do; the wait sleeps on a CTA-local barrier (no memory traffic)
      int it = 0;
      while (!mbar_try_sleep(smem_u32(bars + 2 * kNSMax), 0)) {
        if (++it > kSleepIters * p.tmo_scale) { fail(p, 64ull); poisoned = true; break; }
      }
    }
    for (int sl = wr; sl < nslots; sl += kSlotWarps) {
      const int jb = sl / ngrp, gi = sl % ngrp;
      const int m = b + jb * p.G;
      if (m >= p.M) continue;
      const int e = gi * 32 + lane;
      if (lane == 0 && gi == 0) K1S_STAMP(p.stamps[8 + 2 * p.G + p.M + 4 * m + 0]);
      unsigned long long v[kChunks];
      unsigned pend = 0xffffu;
      unsigned it = 0;
      while (pend && !poisoned) {
        // issue every pending load first (pend is not modified in this loop), then check the tags
#pragma unroll
        for (int cc = 0; cc < kChunks; ++cc)
          if (pend & (1u << cc)) v[cc] = ld_rlx(p.board + (static_cast<long long>(cc) * p.Mpad + m) * ne + e);
        unsigned left = 0;
#pragma unroll
        for (int cc = 0; cc < kChunks; ++cc)
          if ((pend & (1u << cc)) && static_cast<uint32_t>(v[cc] >> 32) != tag) left |= 1u << cc;
        pend = left;
        if (pend) {
          if (++it > kPollIters * static_cast<unsigned>(p.tmo_scale)) { fail(p, 8ull); poisoned = true; break; }
          __nanosleep(32);
        }
      }
      float a = __uint_as_float(static_cast<uint32_t>(v[0]));
#pragma unroll
      for (int cc = 1; cc < kChunks; ++cc) a = __fadd_rn(a, __uint_as_float(static_cast<uint32_t>(v[cc])));
      logit_s[jb * ne + e] = a;
      if (p.dbg != nullptr) p.dbg[static_cast<long long>(m) * ne + e] = a;
      // producer side of the job's named barrier (1 + job): every slot warp except the job leader arrives here
      __syncwarp();
      if (gi != 0) bar_arrive_named(1 + jb, ngrp * 32);
    }
    for (int sl = wr; sl < nslots; sl += kSlotWarps) {
      const int jb = sl / ngrp, gi = sl % ngrp;
      const int m = b + jb * p.G;
      if (m >= p.M || gi != 0) continue;
      if (lane == 0) K1S_STAMP(p.stamps[8 + 2 * p.G + p.M + 4 * m + 1]);
      // the bias is static: load it before waiting on the job's logits, off the critical path
      float biased[8];
      if (kShard) {
        biased[0] = p.bias[p.q0 * 64 + lane * 2];
        biased[1] = p.bias[p.q0 * 64 + lane * 2 + 1];
      } else {
#pragma unroll
        for (int v = 0; v < 2; ++v) {
          const float4 bb = *reinterpret_cast<const float4*>(p.bias + v * 128 + lane * 4);
          biased[v * 4 + 0] = bb.x;
          biased[v * 4 + 1] = bb.y;
          biased[v * 4 + 2] = bb.z;
          biased[v * 4 + 3] = bb.w;
        }
      }
      bar_named(1 + jb, ngrp * 32);  // the job leader waits for every slot warp of its row
      if (lane == 0) K1S_STAMP(p.stamps[8 + 2 * p.G + p.M + 4 * m + 2]);
      const bool any_poison = __any_sync(0xffffffffu, poisoned) || (ld_rlx(p.ctl + 3) != 0ull);
      if (kShard) route_row_shard(p, logit_s + jb * ne, m, tag, any_poison, biased);
      else route_row_256(p, logit_s + jb * ne, m, tag, any_poison, biased);
    }
  } else if (warp == kChainWarps + kRouteWarps) {
    // ---------------- x ring producer ----------------
    if (rows > 0) {
      const unsigned bytes = static_cast<unsigned>(rows * kPad * 4);
      const int first = min(NS, kSubs);
      for (int s = 0; s < kSubs; ++s) {
        const int st = s % NS;
        // slot reuse: wait until every chain warp released the slot (named barrier 8 + slot, whole warp)
        if (s >= NS) bar_named(8 + st, (chain_active + 1) * 32);
        if (lane == 0) {
          // the slot's previous generic-proxy reads (ordered by the barrier) before this async-proxy write
          if (s >= NS) asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
          if (kProd) {
            const uint32_t dst0 = smem_u32(xring + st * (stage_bytes / 4));
            mbar_expect_tx(smem_u32(bars + st), static_cast<unsigned>(rows * 256));
            for (int t = 0; t < t_cnt; ++t)
#pragma unroll
              for (int h = 0; h < 2; ++h)
                asm volatile(
                    "cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1, {%2, %3}], [%4];"
                    ::"r"(dst0 + static_cast<uint32_t>((t * 2 + h) * 2048)), "l"(reinterpret_cast<uint64_t>(&p.xmap)),
                    "r"(c * kKC + s * kSub + h * 32), "r"((t_lo + t) * 16), "r"(smem_u32(bars + st))
                    : "memory");
          } else {
            const float* src = p.ximg + ((static_cast<long long>(c) * kSubs + s) * p.Mpad + t_lo * 16) * kPad;
            mbar_expect_tx(smem_u32(bars + st), bytes);
            bulk_g2s(smem_u32(xring + st * (stage_bytes / 4)), src, bytes, smem_u32(bars + st));
          }
        }
        // role order: the stand-in stream starts only after the router input's first copies are issued (barrier 7)
        if (kPool > 0 && s + 1 == first) {
          __syncwarp();
          bar_arrive_named(7, 64);
        }
      }
    }
  } else if (warp == kChainWarps + kRouteWarps + 1) {
    // ---------------- stand-in byte stream ----------------
    // Bulk L2 prefetches of this CTA's byte share of the other matrix (3 rotating copies): memory traffic only.
    if (kPool > 0 && rows > 0) bar_named(7, 64);  // after the router input's first copies are issued (role order)
    if (kPool > 0 && lane == 0) {
      const long long share = (p.pool_copy_bytes / p.G) & ~static_cast<long long>(kPoolPiece - 1);
      const char* src = p.pool_src + static_cast<long long>(epoch % 3) * p.pool_copy_bytes + share * b;
      unsigned long long done = 0;
      unsigned long long t_s;
      K1S_NOW(t_s);
      long long i = 0;
      for (long long off = 0; off < share; off += kPoolPiece, ++i) {
        // optional pacing (timing builds): piece i is issued no earlier than t_s + i * pace (a bounded-depth stream)
        if (p.pool_pace_ns > 0) K1S_PACE_WAIT(t_s, i, p.pool_pace_ns);
        asm volatile("cp.async.bulk.prefetch.L2.global [%0], %1;" ::"l"(src + off), "r"(static_cast<unsigned>(kPoolPiece))
                     : "memory");
        done += kPoolPiece;
      }
      p.pool_cnt[b] = done;
    }
  } else {
    // ---------------- MMA-pipe stand-in ----------------
    if (kPool == 2) {
      float d[4][4] = {};
      uint32_t a = static_cast<uint32_t>(tid) * 7u + 1u;
#pragma unroll 1
      for (int i = 0; i < p.pmma_iters; ++i) {
#pragma unroll
        for (int j = 0; j < 4; ++j)
          asm volatile(
              "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, "
              "{%0,%1,%2,%3};"
              : "+f"(d[j][0]), "+f"(d[j][1]), "+f"(d[j][2]), "+f"(d[j][3])
              : "r"(a), "r"(a + 1u), "r"(a + 2u), "r"(a + 3u), "r"(a + 5u), "r"(a + 9u));
      }
      if (d[0][0] + d[1][1] + d[2][2] + d[3][3] == 1234.5f) p.stamps[1] = 0;  // keeps the stand-in live
    }
  }
}

// x[M, 8192] fp32 -> chunk-major padded image [16][8][Mpad][kPad] (rows >= M zero), TF32-rounded (cvt.rna, the contract's
// operand rounding, applied once by the producer), the producer's layout.
__global__ void k1s_ximg_kernel(const float* __restrict__ x, int M, int Mpad, float* __restrict__ img) {
  const long long n = static_cast<long long>(kChunks) * kSubs * Mpad * kPad;
  for (long long i = blockIdx.x * static_cast<long long>(blockDim.x) + threadIdx.x; i < n;
       i += static_cast<long long>(gridDim.x) * blockDim.x) {
    const int col = static_cast<int>(i % kPad);
    long long r = i / kPad;
    const int m = static_cast<int>(r % Mpad);
    r /= Mpad;
    const int s = static_cast<int>(r % kSubs);
    const int c = static_cast<int>(r / kSubs);
    const float v = (col < kSub && m < M) ? x[static_cast<long long>(m) * kK + c * kKC + s * kSub + col] : 0.f;
    img[i] = __uint_as_float(to_tf32(__float_as_uint(v)));
  }
}

// Test-only reference: the contract's two-kernel form (partials with fragments read straight from global, then the
// chunk-order sum), as the numerics tooling's tf32_router_partials + chunk_reduce.
__global__ void k1s_ref_partials(const float* __restrict__ x, int M, const float* __restrict__ w, float* __restrict__ part) {
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
  const int g = lane >> 2, t4 = lane & 3;
  const int n0 = blockIdx.x * 32 + warp * 8;
  const int chunk = blockIdx.y;
  const int m0 = blockIdx.z * 16;
  const int r0 = m0 + g, r1 = m0 + g + 8;
  const int n = n0 + g;
  float acc[4] = {0.f, 0.f, 0.f, 0.f};
  for (int ks = 0; ks < kKC; ks += 8) {
    const int k = chunk * kKC + ks + t4;
    uint32_t a[4];
    a[0] = to_tf32(__float_as_uint(r0 < M ? x[static_cast<long long>(r0) * kK + k] : 0.f));
    a[1] = to_tf32(__float_as_uint(r1 < M ? x[static_cast<long long>(r1) * kK + k] : 0.f));
    a[2] = to_tf32(__float_as_uint(r0 < M ? x[static_cast<long long>(r0) * kK + k + 4] : 0.f));
    a[3] = to_tf32(__float_as_uint(r1 < M ? x[static_cast<long long>(r1) * kK + k + 4] : 0.f));
    const uint32_t b0 = to_tf32(__float_as_uint(w[static_cast<long long>(n) * kK + k]));
    const uint32_t b1 = to_tf32(__float_as_uint(w[static_cast<long long>(n) * kK + k + 4]));
    mma1688(acc, a, b0, b1);
  }
  const int col = n0 + t4 * 2;
  float* pp = part + static_cast<long long>(chunk) * M * 256;
  if (r0 < M) {
    pp[static_cast<long long>(r0) * 256 + col] = acc[0];
    pp[static_cast<long long>(r0) * 256 + col + 1] = acc[1];
  }
  if (r1 < M) {
    pp[static_cast<long long>(r1) * 256 + col] = acc[2];
    pp[static_cast<long long>(r1) * 256 + col + 1] = acc[3];
  }
}

__global__ void k1s_ref_reduce(const float* __restrict__ part, long long MN, float* __restrict__ out) {
  const long long i = static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= MN) return;
  float a = part[i];
  for (int c = 1; c < kChunks; ++c) a = __fadd_rn(a, part[static_cast<long long>(c) * MN + i]);
  out[i] = a;
}

// Merge of per-shard candidate lists after their gather. List s, row m (f32: 8 keys, 8 ids as float, 8 scores, sorted by
// (key desc, id asc)) at cand + offs.o[s] + m * stride (per == 0) or + (m / per) * line + (m % per) * stride (lists packed as
// extra rows of a column-block gather) -> top-8 by (key desc, id asc) over the W x 8 candidates -> SGLang's weight arithmetic
// in selection order (sum of the selected scores, + eps, route_scale / sum, weight = score * that). One warp per row. Bitwise
// to SGLang's routing on the full logits when every list holds its shard's exact top-8. Error relay: when any of the n_err
// producer control blocks (err_ctl, 8 words each, word 3 = the producer's sticky error word) is nonzero, every row is poisoned
// (ids -1) and err_code is OR-ed into err_dst (the consumer's sticky error word) once.
struct MergeOffs {
  long long o[4];
};

__global__ void k1s_merge_kernel(const float* __restrict__ cand, int W, int M, int stride, int per, long long line,
                                 const MergeOffs offs, float route_scale, float eps, float* __restrict__ topk_w,
                                 int32_t* __restrict__ topk_ids, int32_t* __restrict__ packed,
                                 const unsigned long long* __restrict__ err_ctl, int n_err, int* __restrict__ err_dst,
                                 int err_code) {
  using namespace device;
  const int lane = threadIdx.x % 32;
  const int m = blockIdx.x * (blockDim.x / 32) + threadIdx.x / 32;
  if (m >= M) return;
  bool bad = false;
  for (int j = 0; j < n_err; ++j) bad |= err_ctl[j * 8 + 3] != 0ull;
  if (bad) {
    if (m == 0 && lane == 0 && err_dst != nullptr) atomicOr(err_dst, err_code);
    if (lane < kTopK) {
      topk_ids[m * kTopK + lane] = -1;
      topk_w[m * kTopK + lane] = 0.0f;
      packed[m * kTopK + lane] = pack_id_weight(-1, 0.0f);
    }
    return;
  }
  const int sh = lane / kTopK, rk = lane % kTopK;
  float ck = kNegInf, cs = 0.f;
  int cid = 0x7fffffff;
  if (sh < W) {
    const long long ro = per > 0 ? static_cast<long long>(m / per) * line + static_cast<long long>(m % per) * stride
                                 : static_cast<long long>(m) * stride;
    const float* row = cand + offs.o[sh] + ro;
    ck = row[rk];
    cid = static_cast<int>(row[kTopK + rk]);
    cs = row[2 * kTopK + rk];
  }
  bool taken = sh >= W;
  float score_sum = 0.0f;
  int out_id = 0;
  float out_score = 0.0f;
#pragma unroll 1
  for (int k = 0; k < kTopK; ++k) {
    const float kk = taken ? kNegInf : ck;
    const float mx = warp::reduce_max(kk);
    const int cand_id = (!taken && ck == mx) ? cid : 0x7fffffff;
    const int win = __reduce_min_sync(warp::kFullMask, cand_id);
    const bool mine = (!taken && cid == win);
    const float sel = __shfl_sync(warp::kFullMask, cs, __ffs(__ballot_sync(warp::kFullMask, mine)) - 1);
    if (mine) taken = true;
    score_sum += sel;
    if (lane == k) { out_id = win; out_score = sel; }
  }
  score_sum += eps;
  const float inv_sum_scaled = div_full(route_scale, score_sum);
  if (lane < kTopK) {
    const float weight = out_score * inv_sum_scaled;
    topk_ids[m * kTopK + lane] = out_id;
    topk_w[m * kTopK + lane] = weight;
    packed[m * kTopK + lane] = pack_id_weight(out_id, weight);
  }
}

}  // namespace muse_vr_k1s

template <int kShard, int kPool>
struct WfcK1S {
  static void run(tvm::ffi::TensorView ximg, tvm::ffi::TensorView w, tvm::ffi::TensorView bias, tvm::ffi::TensorView board,
                  tvm::ffi::TensorView topk_w, tvm::ffi::TensorView topk_ids, tvm::ffi::TensorView packed,
                  tvm::ffi::TensorView ready, tvm::ffi::TensorView ctl, tvm::ffi::TensorView stamps,
                  tvm::ffi::TensorView pool_src, tvm::ffi::TensorView pool_cnt, tvm::ffi::TensorView dbg,
                  tvm::ffi::TensorView cand, tvm::ffi::TensorView cand_score, tvm::ffi::TensorView cand_out,
                  tvm::ffi::TensorView cand_out_score, int64_t M, int64_t Mpad, int64_t G, int64_t tg,
                  int64_t rslab, int64_t jobs, int64_t q0, int64_t pmma_iters, int64_t use_dbg, double mult, double route_scale,
                  double eps, int64_t pool_pace_ns, int64_t tmo_scale) {
    using namespace host;
    namespace k = muse_vr_k1s;
    k::Params p{};
    p.ximg = static_cast<const float*>(ximg.data_ptr());
    p.w = static_cast<const float*>(w.data_ptr());
    p.bias = static_cast<const float*>(bias.data_ptr());
    p.board = static_cast<unsigned long long*>(board.data_ptr());
    p.topk_w = static_cast<float*>(topk_w.data_ptr());
    p.topk_ids = static_cast<int32_t*>(topk_ids.data_ptr());
    p.packed = static_cast<int32_t*>(packed.data_ptr());
    p.ready = static_cast<unsigned long long*>(ready.data_ptr());
    p.ctl = static_cast<unsigned long long*>(ctl.data_ptr());
    p.stamps = static_cast<unsigned long long*>(stamps.data_ptr());
    p.pool_src = static_cast<const char*>(pool_src.data_ptr());
    p.pool_copy_bytes = pool_src.numel() / 3;
    p.pool_cnt = static_cast<unsigned long long*>(pool_cnt.data_ptr());
    p.dbg = use_dbg ? static_cast<float*>(dbg.data_ptr()) : nullptr;
    p.cand = static_cast<const unsigned long long*>(cand.data_ptr());
    p.cand_score = static_cast<const float*>(cand_score.data_ptr());
    p.cand_out = static_cast<unsigned long long*>(cand_out.data_ptr());
    p.cand_out_score = static_cast<float*>(cand_out_score.data_ptr());
    p.M = static_cast<int>(M);
    p.Mpad = static_cast<int>(Mpad);
    p.G = static_cast<int>(G);
    p.tg = static_cast<int>(tg);
    p.rslab = static_cast<int>(rslab);
    p.jobs = static_cast<int>(jobs);
    p.q0 = static_cast<int>(q0);
    p.pmma_iters = static_cast<int>(pmma_iters);
    p.pool_pace_ns = static_cast<int>(pool_pace_ns);
    p.tmo_scale = static_cast<int>(tmo_scale < 1 ? 1 : tmo_scale);
    p.mult = static_cast<float>(mult);
    p.route_scale = static_cast<float>(route_scale);
    p.eps = static_cast<float>(eps);
    RuntimeCheck(p.M >= 1 && p.M <= 256, "wfc_k1s: 1..256 rows");
    RuntimeCheck(p.G == 192, "wfc_k1s: probe grid is 192 CTAs");
    RuntimeCheck(p.tg * 16 <= k::kRowsMax && p.rslab >= 1 && p.rslab <= k::kRMax && (p.tg + p.rslab - 1) / p.rslab <= 2,
                 "wfc_k1s: row tiling");
    RuntimeCheck(p.jobs >= 1 && p.jobs <= 2 && p.jobs * p.G >= p.M, "wfc_k1s: route jobs");
    auto kernel = k::k1s_kernel<kShard, kPool, 0>;
    static bool configured = false;
    if (!configured) {
      RuntimeDeviceCheck(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, k::kSmemBytes));
      configured = true;
    }
    const cudaStream_t stream = LaunchKernel::resolve_device(ximg.device());
    cudaLaunchConfig_t cfg{};
    cfg.gridDim = dim3(static_cast<unsigned>(p.G));
    cfg.blockDim = dim3(k::kThreads);
    cfg.dynamicSmemBytes = k::kSmemBytes;
    cfg.stream = stream;
    cudaLaunchAttribute attr[1];
    attr[0].id = cudaLaunchAttributeCooperative;
    attr[0].val.cooperative = 1;
    cfg.attrs = attr;
    cfg.numAttrs = 1;
    RuntimeDeviceCheck(cudaLaunchKernelEx(&cfg, kernel, p));
  }

  // Production form: router input x [M, 8192] fp32 row-major (the boundary's router output), read through a 2-D tensor map
  // after griddepcontrol.wait; launched with programmatic stream serialization (PDL) so the weight staging overlaps the
  // predecessor; tag = seq_word[0] * 64 + layer. CTAs of one launch wait on each other's partials: the grid is sized to one CTA
  // per SM slot and every wait is bounded (sticky error word, poisoned rows) - co-residency is not guaranteed by the launch.
  static void run_prod(tvm::ffi::TensorView x, tvm::ffi::TensorView w, tvm::ffi::TensorView bias, tvm::ffi::TensorView board,
                       tvm::ffi::TensorView topk_w, tvm::ffi::TensorView topk_ids, tvm::ffi::TensorView packed,
                       tvm::ffi::TensorView ready, tvm::ffi::TensorView ctl, tvm::ffi::TensorView stamps,
                       tvm::ffi::TensorView dbg, tvm::ffi::TensorView seq_word, int64_t layer, int64_t M, int64_t Mpad,
                       int64_t G, int64_t tg, int64_t rslab, int64_t jobs, int64_t q0, int64_t use_dbg, double mult,
                       double route_scale, double eps, int64_t tmo_scale, tvm::ffi::TensorView cand_out,
                       tvm::ffi::TensorView cand_out_score, tvm::ffi::TensorView cand_dst, int64_t cand_stride,
                       int64_t cand_per, int64_t cand_line) {
    using namespace host;
    namespace k = muse_vr_k1s;
    static_assert(kPool == 0, "the production form has no stand-in");
    k::Params p{};
    using EncodeFn = CUresult (*)(CUtensorMap*, CUtensorMapDataType, cuuint32_t, void*, const cuuint64_t*, const cuuint64_t*,
                                  const cuuint32_t*, const cuuint32_t*, CUtensorMapInterleave, CUtensorMapSwizzle,
                                  CUtensorMapL2promotion, CUtensorMapFloatOOBfill);
    static EncodeFn encode = nullptr;
    if (encode == nullptr) {
      cudaDriverEntryPointQueryResult q{};
      void* fn = nullptr;
      RuntimeDeviceCheck(cudaGetDriverEntryPoint("cuTensorMapEncodeTiled", &fn, cudaEnableDefault, &q));
      RuntimeCheck(fn != nullptr && q == cudaDriverEntryPointSuccess, "wfc_k1s: tensor-map encoder not available");
      encode = reinterpret_cast<EncodeFn>(fn);
    }
    const cuuint64_t dims[2] = {static_cast<cuuint64_t>(k::kK), static_cast<cuuint64_t>(M)};
    const cuuint64_t strides[1] = {static_cast<cuuint64_t>(k::kK) * 4};
    const cuuint32_t box[2] = {32, 16};
    const cuuint32_t estr[2] = {1, 1};
    RuntimeCheck(encode(&p.xmap, CU_TENSOR_MAP_DATA_TYPE_FLOAT32, 2, x.data_ptr(), dims, strides, box, estr,
                        CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_L2_128B,
                        CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE) == CUDA_SUCCESS,
                 "wfc_k1s: tensor map encode failed");
    p.seq_word = static_cast<const long long*>(seq_word.data_ptr());
    p.layer = static_cast<int>(layer);
    p.w = static_cast<const float*>(w.data_ptr());
    p.bias = static_cast<const float*>(bias.data_ptr());
    p.board = static_cast<unsigned long long*>(board.data_ptr());
    p.topk_w = static_cast<float*>(topk_w.data_ptr());
    p.topk_ids = static_cast<int32_t*>(topk_ids.data_ptr());
    p.packed = static_cast<int32_t*>(packed.data_ptr());
    p.ready = static_cast<unsigned long long*>(ready.data_ptr());
    p.ctl = static_cast<unsigned long long*>(ctl.data_ptr());
    p.stamps = static_cast<unsigned long long*>(stamps.data_ptr());
    p.dbg = use_dbg ? static_cast<float*>(dbg.data_ptr()) : nullptr;
    p.M = static_cast<int>(M);
    p.Mpad = static_cast<int>(Mpad);
    p.G = static_cast<int>(G);
    p.tg = static_cast<int>(tg);
    p.rslab = static_cast<int>(rslab);
    p.jobs = static_cast<int>(jobs);
    p.q0 = static_cast<int>(q0);
    p.mult = static_cast<float>(mult);
    p.route_scale = static_cast<float>(route_scale);
    p.eps = static_cast<float>(eps);
    p.tmo_scale = static_cast<int>(tmo_scale < 1 ? 1 : tmo_scale);
    p.cand_out = static_cast<unsigned long long*>(cand_out.data_ptr());
    p.cand_out_score = static_cast<float*>(cand_out_score.data_ptr());
    p.cand_dst = (kShard && cand_stride > 0) ? static_cast<float*>(cand_dst.data_ptr()) : nullptr;
    p.cand_stride = static_cast<int>(cand_stride);
    p.cand_per = static_cast<int>(cand_per);
    p.cand_line = cand_line;
    RuntimeCheck(p.cand_dst == nullptr || (p.cand_stride >= 24 && p.cand_per >= 0 &&
                                           (p.cand_per == 0 || p.cand_line >= static_cast<long long>(p.cand_per) * 24)),
                 "wfc_k1s: candidate layout");
    RuntimeCheck(p.M >= 1 && p.M <= 256, "wfc_k1s: 1..256 rows");
    // replicated: 192 CTAs = 16 chunks x 3 row groups x 4 expert quarters; shard: 16 chunks x G / 16 row groups (a reserved
    // slice of the SMs when G < 192, so a library GEMM can run beside it)
    RuntimeCheck(kShard ? (p.G % 16 == 0 && p.G >= 16 && p.G <= 192) : p.G == 192, "wfc_k1s: grid");
    RuntimeCheck(p.tg * 16 <= k::kRowsMax && p.rslab >= 1 && p.rslab <= k::kRMax && (p.tg + p.rslab - 1) / p.rslab <= 2,
                 "wfc_k1s: row tiling");
    RuntimeCheck(p.jobs >= 1 && p.jobs <= (kShard ? 6 : 2) && p.jobs * p.G >= p.M, "wfc_k1s: route jobs");
    auto kernel = k::k1s_kernel<kShard, 0, 1>;
    static bool configured = false;
    if (!configured) {
      RuntimeDeviceCheck(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, k::kSmemBytes));
      configured = true;
    }
    const cudaStream_t stream = LaunchKernel::resolve_device(x.device());
    cudaLaunchConfig_t cfg{};
    cfg.gridDim = dim3(static_cast<unsigned>(p.G));
    cfg.blockDim = dim3(k::kThreads);
    cfg.dynamicSmemBytes = k::kSmemBytes;
    cfg.stream = stream;
    cudaLaunchAttribute attr[1];
    attr[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
    attr[0].val.programmaticStreamSerializationAllowed = 1;
    cfg.attrs = attr;
    cfg.numAttrs = 1;
    RuntimeDeviceCheck(cudaLaunchKernelEx(&cfg, kernel, p));
  }

  // cand: f32 view of the gathered buffer; list s at element offset off_s (see k1s_merge_kernel for the row addressing).
  static void merge(tvm::ffi::TensorView cand, int64_t W, int64_t M, int64_t stride, int64_t per, int64_t line, int64_t off0,
                    int64_t off1, int64_t off2, int64_t off3, double route_scale, double eps, tvm::ffi::TensorView topk_w,
                    tvm::ffi::TensorView topk_ids, tvm::ffi::TensorView packed, tvm::ffi::TensorView err_ctl,
                    int64_t n_err, tvm::ffi::TensorView err_dst, int64_t err_code) {
    using namespace host;
    RuntimeCheck(W >= 1 && W <= 4 && M >= 1 && stride >= 24 && per >= 0 && (per == 0 || line >= per * 24),
                 "wfc_k1s merge: 1..4 lists, layout");
    RuntimeCheck(n_err >= 0 && n_err <= 4, "wfc_k1s merge: error blocks");
    muse_vr_k1s::MergeOffs offs{{off0, off1, off2, off3}};
    const cudaStream_t stream = LaunchKernel::resolve_device(cand.device());
    const unsigned blocks = static_cast<unsigned>((M + 3) / 4);
    muse_vr_k1s::k1s_merge_kernel<<<blocks, 128, 0, stream>>>(
        static_cast<const float*>(cand.data_ptr()), static_cast<int>(W), static_cast<int>(M), static_cast<int>(stride),
        static_cast<int>(per), static_cast<long long>(line), offs, static_cast<float>(route_scale), static_cast<float>(eps),
        static_cast<float*>(topk_w.data_ptr()), static_cast<int32_t*>(topk_ids.data_ptr()),
        static_cast<int32_t*>(packed.data_ptr()), static_cast<const unsigned long long*>(err_ctl.data_ptr()),
        static_cast<int>(n_err), err_code != 0 ? static_cast<int*>(err_dst.data_ptr()) : nullptr,
        static_cast<int>(err_code));
  }

  static void ximg(tvm::ffi::TensorView x, int64_t M, int64_t Mpad, tvm::ffi::TensorView img) {
    using namespace host;
    const cudaStream_t stream = LaunchKernel::resolve_device(x.device());
    muse_vr_k1s::k1s_ximg_kernel<<<1024, 256, 0, stream>>>(static_cast<const float*>(x.data_ptr()), static_cast<int>(M),
                                                            static_cast<int>(Mpad), static_cast<float*>(img.data_ptr()));
  }

  static void ref(tvm::ffi::TensorView x, tvm::ffi::TensorView w, int64_t M, tvm::ffi::TensorView part,
                  tvm::ffi::TensorView out) {
    using namespace host;
    const cudaStream_t stream = LaunchKernel::resolve_device(x.device());
    muse_vr_k1s::k1s_ref_partials<<<dim3(256 / 32, muse_vr_k1s::kChunks, static_cast<unsigned>((M + 15) / 16)), 128, 0,
                                    stream>>>(static_cast<const float*>(x.data_ptr()), static_cast<int>(M),
                                              static_cast<const float*>(w.data_ptr()), static_cast<float*>(part.data_ptr()));
    const long long mn = M * 256;
    muse_vr_k1s::k1s_ref_reduce<<<static_cast<unsigned>((mn + 255) / 256), 256, 0, stream>>>(
        static_cast<const float*>(part.data_ptr()), mn, static_cast<float*>(out.data_ptr()));
  }
};

}  // namespace sglang
