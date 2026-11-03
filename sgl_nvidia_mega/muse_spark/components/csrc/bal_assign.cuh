// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
//
// Per-step expert-parallel assignment over a fixed replica pool (component ``bal``). One warp, integer arithmetic only.
//
// assign_warp(): from the routed top-k ids of one step (every rank holds the same ids), count the routed rows per
// expert, give each routed expert a cost c(e) = a + (b * rows) / 256 + p * (passes - 1) (Cost<world>; passes = ceil(rows /
// pass_rows)),
// and assign every routed expert to one of its holders:
//   1. single-holder experts go to their home rank;
//   2. replicated experts, in (row passes descending, key ascending) order with key(e) = (e % per) * world + e / per (home
//      ranks interleaved, per = 256 / world), go to the holder with the lower load (ties: home), where the loads are
//      read once per group of kGroup consecutive entries (all decisions of a group see the loads at the group's start);
//   3. at most kRepairMax moves: on the max-load rank (lowest rank among ties), the replicated expert assigned there
//      whose move to its other holder gives the smallest max(load[max] - c, load[other] + c) (ties: lowest id) moves
//      iff that value is below load[max].
// The result depends only on the ids, the placement table and pass_rows, so every CTA and every rank that runs it on
// the same ids computes the same assignment, with no communication.
//
// Placement table (same packing as ep_balance): per expert, bits 0-3 home rank, 4-7 second holder (0xF = none),
// 8-15 slot on the home rank, 16-23 slot on the second holder.
//
// Building blocks for a caller kernel: stage_table (static, before the ids wait), count_ids (any thread count, into a
// zeroed shared histogram), then assign_warp (one warp) after a barrier; digest_warp gives a 32-bit assignment digest.
//
// bal_ids_kernel (development form, used by the tests and benches): every CTA runs assign_warp; CTA 0 writes the packed
// ids with virtual ids (rank * pool + slot; or this rank's local view), which the routed MoE of a rank reads with
// local offset rank * pool; any CTA may write its assignment (determinism tests) and CTA 0 lane 0 may write per-phase
// cycle stamps. Every loop has a compile-time or warp-uniform bound; there are no waits on other CTAs.
#pragma once

#include <sgl_kernel/tensor.h>
#include <sgl_kernel/utils.h>

#include <sgl_kernel/utils.cuh>

#include <tvm/ffi/container/tensor.h>
#include <tvm/ffi/optional.h>

#include <cstdint>

namespace muse_vr::bal {

inline constexpr int kNumExperts = 256;
inline constexpr int kTopK = 8;
inline constexpr int kMaxRows = 512;
inline constexpr int kMaxWorld = 8;
inline constexpr int kRepairMax = 2;
inline constexpr uint32_t kGroup = 4;  // greedy decisions per load snapshot (one vector load of entries)
inline constexpr int kExpPerLane = kNumExperts / 32;  // experts lane l owns: l * 8 .. l * 8 + 7
// Per-expert cost of one rank's routed MoE, in 1/100 us-like units: a per expert, b / 256 per routed row, p per further
// row pass. Tunable; fitted to measured per-rank times of the routed MoE (bench/bench_bal.py) per world.
template <int kWorld>
struct Cost;
template <>
struct Cost<4> {
  static constexpr uint32_t a = 184, b = 197, p = 21;
};
template <>
struct Cost<8> {
  static constexpr uint32_t a = 147, b = 440, p = 0;
};
inline constexpr uint32_t kUnrouted = 0xffu;
inline constexpr uint32_t kNone = 0xfu;
inline constexpr uint32_t kPadId = 0xffffu;          // expert field of a padded row: passed through, not routed
inline constexpr uint32_t kErrBadId = 1u;
inline constexpr uint32_t kErrBadParams = 2u;
inline constexpr uint32_t kFull = 0xffffffffu;
inline constexpr int kNumStamps = 6;

struct Smem {
  uint32_t tab[kNumExperts];   // placement table (static; staged before the ids are read)
  uint32_t cnt[kNumExperts];   // routed rows per expert
  uint32_t ent[kNumExperts + 4];  // routed replicated experts in assignment order: e << 24 | cost << 8 | second << 4 | home
  uint32_t attr[kNumExperts];  // cost << 8 | second holder << 4 | home (cost 0: not routed)
  uint8_t asg[kNumExperts];    // assigned rank (kUnrouted: not routed)
};

// SM cycle counter (diagnostic phase stamps only).
__device__ __forceinline__ unsigned long long stamp_now() { return static_cast<unsigned long long>(clock64()); }

__device__ __forceinline__ void pdl_wait() { asm volatile("griddepcontrol.wait;" ::: "memory"); }

__device__ __forceinline__ void pdl_launch_dependents() { asm volatile("griddepcontrol.launch_dependents;" ::: "memory"); }

// Static part (may run before the ids are available): the placement table into shared memory. One full warp.
__device__ __forceinline__ void stage_table(const uint32_t* __restrict__ table, Smem& sm, uint32_t lane) {
  const uint4* src = reinterpret_cast<const uint4*>(table) + lane * 2;
  uint4* dst = reinterpret_cast<uint4*>(sm.tab) + lane * 2;
  dst[0] = src[0];
  dst[1] = src[1];
}

// Count the routed rows per expert from n packed ids (any number of threads: tid / nthreads), into sm.cnt (zeroed by
// the caller before). Returns this thread's error bits.
__device__ __forceinline__ uint32_t count_ids(const int32_t* __restrict__ ids, int n, Smem& sm, int tid, int nthreads) {
  uint32_t err = 0;
  for (int i = tid; i < n; i += nthreads) {
    const uint32_t e = (uint32_t(ids[i]) >> 16) & 0xffffu;
    if (e < uint32_t(kNumExperts)) {
      atomicAdd(&sm.cnt[e], 1u);
    } else if (e != kPadId) {
      err |= kErrBadId;
    }
  }
  return err;
}

// Rank loads, replicated in every lane: four 16-bit loads per 64-bit word (ranks 0-3 in w0, 4-7 in w1); a rank load
// stays below 2^16. Reads and adds are shifts by the rank, so no indexed register array is formed.
struct Loads {
  unsigned long long w0, w1;
};

template <int kWorld>
__device__ __forceinline__ uint32_t load_of(const Loads& L, uint32_t r) {
  const unsigned long long w = (kWorld == 8 && (r & 4u)) ? L.w1 : L.w0;
  return uint32_t(w >> ((r & 3u) << 4)) & 0xffffu;
}

template <int kWorld>
__device__ __forceinline__ void add_to(Loads& L, uint32_t r, uint32_t inc) {
  const unsigned long long d = static_cast<unsigned long long>(inc) << ((r & 3u) << 4);
  if constexpr (kWorld == 8) {
    L.w0 += (r & 4u) ? 0ull : d;
    L.w1 += (r & 4u) ? d : 0ull;
  } else {
    L.w0 += d;
  }
}

template <int kWorld>
__device__ __forceinline__ void sub_from(Loads& L, uint32_t r, uint32_t dec) {
  const unsigned long long d = static_cast<unsigned long long>(dec) << ((r & 3u) << 4);
  if constexpr (kWorld == 8) {
    L.w0 -= (r & 4u) ? 0ull : d;
    L.w1 -= (r & 4u) ? d : 0ull;
  } else {
    L.w0 -= d;
  }
}

// One full warp (all 32 lanes, converged), after stage_table and count_ids (and a barrier covering both). Writes sm.asg.
// pass_inv = ceil(65536 / pass_rows) (exact division for the row counts in range; checked on the host). stamps:
// non-null on at most one lane (writes stamps[2..5]). Returns this lane's error bits.
//
// Lane l owns the experts with order keys l * 8 .. l * 8 + 7, key(e) = (e % per) * kWorld + e / per (home ranks
// interleaved), so the per-class compaction below yields (cost descending, key ascending) order. Every lane runs the
// greedy on the same data with the rank loads in registers; the decision for entry i is kept by lane i % 32.
template <int kWorld>
__device__ __forceinline__ uint32_t assign_warp(uint32_t pass_rows, uint32_t pass_inv, Smem& sm, uint32_t lane,
                                                unsigned long long* stamps) {
  constexpr uint32_t kPer = kNumExperts / kWorld;
  constexpr int kPairs = kWorld / 2;
  uint32_t err = 0;
  uint32_t ej[kExpPerLane], hj[kExpPerLane], sj[kExpPerLane], cj[kExpPerLane], pj[kExpPerLane];
  uint32_t movable = 0u, max_pass = 0u;
  uint32_t acc[kPairs];  // fixed loads, two 16-bit ranks per word (a rank load stays below 2^16)
#pragma unroll
  for (int k = 0; k < kPairs; ++k) acc[k] = 0u;
#pragma unroll
  for (int j = 0; j < kExpPerLane; ++j) {
    const uint32_t key = lane * kExpPerLane + j;
    const uint32_t e = (key % kWorld) * kPer + key / kWorld;
    const uint32_t t = sm.tab[e];
    const uint32_t h = t & 0xfu, s = (t >> 4) & 0xfu;
    const uint32_t rows = sm.cnt[e];
    const uint32_t p = ((rows + pass_rows - 1u) * pass_inv) >> 16;
    const uint32_t c = rows != 0u ? Cost<kWorld>::a + ((Cost<kWorld>::b * rows) >> 8) + Cost<kWorld>::p * (p - 1u) : 0u;
    if (h >= uint32_t(kWorld) || (s != kNone && (s >= uint32_t(kWorld) || s == h))) err |= kErrBadParams;
    ej[j] = e;
    hj[j] = h;
    sj[j] = s;
    cj[j] = c;
    pj[j] = p;
    sm.attr[e] = (c << 8) | ((s & 0xfu) << 4) | (h & 0xfu);
    sm.asg[e] = uint8_t(rows != 0u ? h : kUnrouted);
    const uint32_t cf = (rows != 0u && s == kNone) ? (c << ((h & 1u) * 16u)) : 0u;
#pragma unroll
    for (int k = 0; k < kPairs; ++k) acc[k] += (h >> 1) == uint32_t(k) ? cf : 0u;
    if (rows != 0u && s != kNone) {
      movable |= 1u << j;
      max_pass = max(max_pass, p);
    }
  }
#pragma unroll
  for (int o = 16; o > 0; o >>= 1) {
#pragma unroll
    for (int k = 0; k < kPairs; ++k) acc[k] += __shfl_xor_sync(kFull, acc[k], o);
    max_pass = max(max_pass, __shfl_xor_sync(kFull, max_pass, o));
  }
  Loads L{};
  L.w0 = static_cast<unsigned long long>(acc[0]) | (static_cast<unsigned long long>(acc[1]) << 32);
  if constexpr (kWorld == 8) {
    L.w1 = static_cast<unsigned long long>(acc[2]) | (static_cast<unsigned long long>(acc[3]) << 32);
  }
  if (stamps != nullptr) stamps[2] = stamp_now();

  // ---- order: one class per row-pass count (highest first), key order inside a class
  uint32_t nmov = 0u;
  for (uint32_t p = max_pass; p >= 1u; --p) {  // warp-uniform bound
    uint32_t mask = 0u;
#pragma unroll
    for (int j = 0; j < kExpPerLane; ++j) mask |= (((movable >> j) & 1u) != 0u && pj[j] == p) ? (1u << j) : 0u;
    const uint32_t k = __popc(mask);
    uint32_t incl = k;
#pragma unroll
    for (int o = 1; o < 32; o <<= 1) {
      const uint32_t y = __shfl_up_sync(kFull, incl, o);
      if (lane >= uint32_t(o)) incl += y;
    }
    uint32_t pos = nmov + incl - k;
#pragma unroll
    for (int j = 0; j < kExpPerLane; ++j) {
      if ((mask >> j) & 1u) sm.ent[pos++] = (ej[j] << 24) | (cj[j] << 8) | (sj[j] << 4) | hj[j];
    }
    nmov += __shfl_sync(kFull, incl, 31);
  }
  if (lane < uint32_t(kGroup) && nmov + lane < ((nmov + kGroup - 1u) / kGroup) * kGroup) sm.ent[nmov + lane] = 0u;
  __syncwarp();
  if (stamps != nullptr) stamps[3] = stamp_now();

  // ---- greedy in groups of kGroup consecutive entries: every decision of a group compares the two holders' loads as
  //      they were at the group's start; the group's costs are then added. Every lane runs it on the same data (loads
  //      in registers, no shuffles); entries come from shared memory, kGroup per vector load, padded with no-op entries.
  uint32_t dec = 0u;
  const uint4* ent4 = reinterpret_cast<const uint4*>(sm.ent);
  for (uint32_t g = 0; g * kGroup < nmov; ++g) {  // warp-uniform bound
    const uint4 q4 = ent4[g];
    const uint32_t en[kGroup] = {q4.x, q4.y, q4.z, q4.w};
    Loads D{};
#pragma unroll
    for (int k = 0; k < kGroup; ++k) {
      const uint32_t h = en[k] & 0xfu, s = (en[k] >> 4) & 0xfu, c = (en[k] >> 8) & 0xffffu;
      const bool to_s = load_of<kWorld>(L, s) < load_of<kWorld>(L, h);
      add_to<kWorld>(D, to_s ? s : h, c);
      const uint32_t i = g * kGroup + uint32_t(k);
      dec |= (lane == (i & 31u) && to_s) ? (1u << (i >> 5)) : 0u;
    }
    L.w0 += D.w0;
    L.w1 += D.w1;
  }
#pragma unroll
  for (int q = 0; q < kExpPerLane; ++q) {
    const uint32_t i = uint32_t(q) * 32u + lane;
    if (i < nmov) {
      const uint32_t en = sm.ent[i];
      sm.asg[en >> 24] = uint8_t(((dec >> q) & 1u) ? ((en >> 4) & 0xfu) : (en & 0xfu));
    }
  }
  __syncwarp();
  if (stamps != nullptr) stamps[4] = stamp_now();

  // ---- bounded repair: only while some move off the max rank lowers its pair maximum (needs max - load[other] > cost)
  for (int it = 0; it < kRepairMax; ++it) {
    uint32_t mx = load_of<kWorld>(L, 0u), rm = 0u, mn = mx;
#pragma unroll
    for (int r = 1; r < kWorld; ++r) {
      const uint32_t v = load_of<kWorld>(L, uint32_t(r));
      if (v > mx) {
        mx = v;
        rm = uint32_t(r);
      }
      mn = min(mn, v);
    }
    if (mx - mn <= Cost<kWorld>::a) break;  // warp-uniform: no single move can qualify (every cost exceeds a)
    uint32_t best = kFull;  // (new pair max << 9) | expert
#pragma unroll
    for (int j = 0; j < kExpPerLane; ++j) {
      if (((movable >> j) & 1u) != 0u && uint32_t(sm.asg[ej[j]]) == rm) {
        const uint32_t ot = rm == hj[j] ? sj[j] : hj[j];
        const uint32_t nm = max(mx - cj[j], load_of<kWorld>(L, ot) + cj[j]);
        if (nm < mx) best = min(best, (nm << 9) | ej[j]);
      }
    }
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) best = min(best, __shfl_xor_sync(kFull, best, o));
    if (best == kFull) break;  // warp-uniform
    const uint32_t e = best & 0x1ffu;
    const uint32_t a = sm.attr[e];
    const uint32_t h = a & 0xfu, s = (a >> 4) & 0xfu, c = a >> 8;
    const uint32_t ot = rm == h ? s : h;
    add_to<kWorld>(L, ot, c);
    sub_from<kWorld>(L, rm, c);
    __syncwarp();
    if (lane == 0u) sm.asg[e] = uint8_t(ot);
    __syncwarp();
  }
  __syncwarp();
  if (stamps != nullptr) stamps[5] = stamp_now();
  return err;
}

// 32-bit digest of an assignment (for cross-rank agreement checks), same value in every lane.
__device__ __forceinline__ uint32_t digest_warp(const Smem& sm, uint32_t lane) {
  uint32_t h = 0u;
#pragma unroll
  for (int j = 0; j < kExpPerLane; ++j) {
    const uint32_t e = lane * kExpPerLane + j;
    const uint32_t y = ((e << 8) | sm.asg[e]) * 0x9e3779b1u;
    h += y ^ (y >> 16);
  }
#pragma unroll
  for (int o = 16; o > 0; o >>= 1) h += __shfl_xor_sync(kFull, h, o);
  h ^= h >> 16;
  h *= 0x85ebca6bu;
  h ^= h >> 13;
  return h;
}

struct Params {
  const int32_t* ids_in;    // [m * 8] packed
  const uint32_t* table;    // [256]
  int32_t* ids_out;         // [m * 8] packed, virtual ids (CTA 0 writes)
  uint8_t* assign_out;      // optional [grid][256]: every CTA's assignment
  uint32_t* err;            // [1] sticky error bits
  unsigned long long* stamps;  // optional [kNumStamps] (CTA 0, lane 0)
  uint32_t* digest_out;        // optional [grid]: every CTA's assignment digest
  int32_t m;
  int32_t pool;
  int32_t rank;
  int32_t local_view;  // 1: this rank's view (own slots 0..pool-1, entries of other ranks kPadId)
  uint32_t pass_rows;
  uint32_t pass_inv;  // ceil(65536 / pass_rows)
};

template <int kWorld>
__global__ void __launch_bounds__(32, 1) bal_ids_kernel(const __grid_constant__ Params p) {
  __shared__ __align__(16) Smem sm;
  const uint32_t lane = threadIdx.x;
  stage_table(p.table, sm, lane);  // static: before the wait
#pragma unroll
  for (int j = 0; j < kExpPerLane; ++j) sm.cnt[lane * kExpPerLane + j] = 0u;
  __syncwarp();
  unsigned long long* st = (p.stamps != nullptr && blockIdx.x == 0 && lane == 0u) ? p.stamps : nullptr;
  pdl_wait();
  if (st != nullptr) st[0] = stamp_now();
  uint32_t err = count_ids(p.ids_in, p.m * kTopK, sm, int(lane), 32);
  __syncwarp();
  if (st != nullptr) st[1] = stamp_now();
  err |= assign_warp<kWorld>(p.pass_rows, p.pass_inv, sm, lane, st);
  const uint32_t dg = digest_warp(sm, lane);
  if (p.assign_out != nullptr) {
#pragma unroll
    for (int j = 0; j < kExpPerLane; ++j) {
      p.assign_out[size_t(blockIdx.x) * kNumExperts + lane * kExpPerLane + j] = sm.asg[lane * kExpPerLane + j];
    }
  }
  if (p.digest_out != nullptr && lane == 0u) p.digest_out[blockIdx.x] = dg;
  if (blockIdx.x == 0) {
    const int n = p.m * kTopK;
    const uint32_t pool = uint32_t(p.pool);
    for (int i = int(lane); i < n; i += 32) {
      const uint32_t w = uint32_t(p.ids_in[i]);
      const uint32_t e = (w >> 16) & 0xffffu;
      uint32_t out = w;
      if (e < uint32_t(kNumExperts)) {
        const uint32_t r = sm.asg[e];
        const uint32_t t = sm.tab[e];
        const uint32_t slot = r == (t & 0xfu) ? ((t >> 8) & 0xffu) : ((t >> 16) & 0xffu);
        uint32_t v = r * pool + slot;
        if (p.local_view) v = r == uint32_t(p.rank) ? slot : kPadId;
        out = (v << 16) | (w & 0xffffu);
      }
      p.ids_out[i] = int32_t(out);
    }
  }
  __syncwarp();
  pdl_launch_dependents();
#pragma unroll
  for (int o = 16; o > 0; o >>= 1) err |= __shfl_xor_sync(kFull, err, o);
  if (lane == 0u && err != 0u) atomicOr(p.err, err);
}

// Host entry. ids_in / ids_out [M, 8] int32 (1 <= M <= kMaxRows), table [256] int32 (16-byte aligned), err [1] int32
// (sticky), assign_out optional [grid, 256] uint8, stamps optional [kNumStamps] int64, digest_out optional [grid] int32.
template <bool kUsePDL>
struct BalIds {
  static void run(const tvm::ffi::TensorView ids_in, const tvm::ffi::TensorView table, const tvm::ffi::TensorView ids_out,
                  const tvm::ffi::TensorView err, tvm::ffi::Optional<tvm::ffi::TensorView> assign_out,
                  tvm::ffi::Optional<tvm::ffi::TensorView> stamps, tvm::ffi::Optional<tvm::ffi::TensorView> digest_out,
                  int64_t world, int64_t pool, int64_t pass_rows,
                  int64_t rank, int64_t local_view, int64_t grid) {
    using namespace sglang;
    using namespace sglang::host;
    const int64_t m = ids_in.size(0);
    RuntimeCheck(ids_in.dim() == 2 && ids_in.size(1) == kTopK, "bal: packed ids must be [M, 8]");
    RuntimeCheck(m >= 1 && m <= kMaxRows, "bal: rows out of range");
    RuntimeCheck(ids_out.size(0) == m && ids_out.size(1) == kTopK, "bal: output shape");
    RuntimeCheck(table.numel() == kNumExperts && reinterpret_cast<uintptr_t>(table.data_ptr()) % 16 == 0,
                 "bal: table must hold 256 aligned words");
    RuntimeCheck(world == 4 || world == 8, "bal: world must be 4 or 8");
    RuntimeCheck(pass_rows >= 1 && pass_rows <= 128, "bal: pass rows out of range");
    RuntimeCheck(pool >= kNumExperts / world && pool <= 128 && rank >= 0 && rank < world, "bal: pool / rank");
    RuntimeCheck(grid >= 1 && grid <= 1024, "bal: grid out of range");
    Params p{};
    p.ids_in = static_cast<const int32_t*>(ids_in.data_ptr());
    p.table = static_cast<const uint32_t*>(table.data_ptr());
    p.ids_out = static_cast<int32_t*>(ids_out.data_ptr());
    p.err = static_cast<uint32_t*>(err.data_ptr());
    p.assign_out = nullptr;
    if (assign_out.has_value()) {
      RuntimeCheck(assign_out.value().numel() >= grid * kNumExperts, "bal: assign_out needs [grid, 256] bytes");
      p.assign_out = static_cast<uint8_t*>(assign_out.value().data_ptr());
    }
    p.stamps = nullptr;
    if (stamps.has_value()) {
      RuntimeCheck(stamps.value().numel() >= kNumStamps, "bal: stamps needs 6 int64");
      p.stamps = static_cast<unsigned long long*>(stamps.value().data_ptr());
    }
    p.digest_out = nullptr;
    if (digest_out.has_value()) {
      RuntimeCheck(digest_out.value().numel() >= grid, "bal: digest_out needs [grid] words");
      p.digest_out = static_cast<uint32_t*>(digest_out.value().data_ptr());
    }
    p.m = static_cast<int32_t>(m);
    p.pool = static_cast<int32_t>(pool);
    p.rank = static_cast<int32_t>(rank);
    p.local_view = local_view != 0 ? 1 : 0;
    p.pass_rows = static_cast<uint32_t>(pass_rows);
    p.pass_inv = static_cast<uint32_t>((65536 + pass_rows - 1) / pass_rows);
    for (int64_t x = 0; x <= kMaxRows + pass_rows; ++x) {  // the multiply-shift equals the division for every count
      RuntimeCheck(((uint64_t(x) * p.pass_inv) >> 16) == uint64_t(x / pass_rows), "bal: pass_rows reciprocal not exact");
    }
    const DLDevice dev = ids_in.device();
    if (world == 8) {
      LaunchKernel(static_cast<unsigned>(grid), 32u, dev).enable_pdl(kUsePDL)(bal_ids_kernel<8>, p);
    } else {
      LaunchKernel(static_cast<unsigned>(grid), 32u, dev).enable_pdl(kUsePDL)(bal_ids_kernel<4>, p);
    }
  }
};

}  // namespace muse_vr::bal
