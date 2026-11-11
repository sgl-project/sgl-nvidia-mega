// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//
// DeepSeek-V4.1 query projection in one cluster launch at 1-8 rows (slot attn_q_cluster):
// q [rows][N] = MXFP8(q_norm(q_lora)) wq_b^T, the q_lora RMSNorm and its MXFP8 quant done
// once per cluster and shared through distributed shared memory, then warp-level FP8
// mma.sync on wq_b in fragment-major order with 32x32 UE8M0 scales.
//
// Derived from muLLM v0.4.1 (NVIDIA Corporation, Apache-2.0): its DeepSeek-V4.1
// query-projection cluster kernel and the push reductions it calls (its push, early
// trigger and cluster-fit builds at world 8), its warp weight loader (WarpWeights,
// LoadWarpWeights), its shared projection layout (ProjShared), and its MX FP8, window-KV
// format and evict-first device primitives (below). This file has been modified from
// those: one self-contained NVRTC source; the chain-stamp macros, the multi-group,
// staged, wo_a and Lamport code and the pull reduction are dropped; WarpMmaB keeps its
// one-group branch only; std::conditional_t is a local template. The arithmetic, the
// protocol and the order of every sum are the original's. Host side: ../q_cluster.py.
//
// Build switches: Q_EARLY_TRIGGER (0/1, the original's early-trigger build at world 8).
// Entries: dsv41_q_cluster8 (grid (2, N/64), cluster (2, 4, 1), 256 threads) and
// dsv41_q_cluster8_fit (N = 4096 only: grid (2, 60), cluster (2, 4, 1), 320 threads).

#if !defined(Q_EARLY_TRIGGER)
#error "q_cluster.cu: the launcher passes Q_EARLY_TRIGGER"
#endif

typedef unsigned char uint8_t;
typedef unsigned short uint16_t;
typedef unsigned int uint32_t;
typedef signed int int32_t;
#if defined(__CUDACC_RTC__)
typedef unsigned long long uint64_t;
typedef signed long long int64_t;
#else
typedef unsigned long uint64_t;
typedef signed long int64_t;
#endif

#include <cuda_bf16.h>
#include <cuda_fp8.h>

// ---------------------------------------------------------------- MX FP8 and evict-first primitives

namespace d41 {

constexpr int kFragmentRows = 8;
constexpr int kSpanValues = 64;
constexpr int kBlockValues = 32;
constexpr int kTileRows = 16;
constexpr float kFp8Max = 448.0F;
// The reference computes `1 / 448` in double precision and narrows it to FP32.
constexpr float kFp8MaxInverse = static_cast<float>(1.0 / 448.0);

__device__ __forceinline__ void GridDependencyWait() {
  asm volatile("griddepcontrol.wait;" ::: "memory");
}

__device__ __forceinline__ void LaunchDependents() {
  asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
}

// The 64-bit policy of `createpolicy.fractional.L2::evict_first` over the whole access
// (fraction 1.0). The operands are constants, so ptxas folds the policy into the loads.
__device__ __forceinline__ uint64_t EvictFirstPolicy() {
  uint64_t policy;
  asm("createpolicy.fractional.L2::evict_first.b64 %0, 1.0;" : "=l"(policy));
  return policy;
}

// Streaming 16-byte weight load: bypass L1 allocation, ask L2 to fetch the
// surrounding 256 bytes, which the next unrolled span of the same warp reads, and
// mark the lines first to be evicted.
__device__ __forceinline__ uint4 LoadWeight16(const uint8_t* address) {
  uint4 value;
  asm volatile(
      "ld.global.nc.L1::no_allocate.L2::cache_hint.L2::256B.v4.u32 {%0, %1, %2, %3}, "
      "[%4], %5;"
      : "=r"(value.x), "=r"(value.y), "=r"(value.z), "=r"(value.w)
      : "l"(address), "l"(EvictFirstPolicy()));
  return value;
}

// E8M0 byte `index` of a packed word, decoded to its float value. Zero
// exponents (2^-127) are flushed to zero; no checkpoint block uses them.
template <int kIndex>
__device__ __forceinline__ float PackedE8m0(uint32_t word) {
  if constexpr (kIndex == 0) return __uint_as_float((word << 23) & 0x7F800000U);
  if constexpr (kIndex == 1) return __uint_as_float((word << 15) & 0x7F800000U);
  if constexpr (kIndex == 2) return __uint_as_float((word << 7) & 0x7F800000U);
  return __uint_as_float((word >> 1) & 0x7F800000U);
}

__device__ __forceinline__ void MmaE4m3(float (&d)[4], uint32_t a0, uint32_t a1,
                                        uint32_t a2, uint32_t a3, uint32_t b0,
                                        uint32_t b1) {
  asm volatile(
      "mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 "
      "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %10, %10, %10};"
      : "=f"(d[0]), "=f"(d[1]), "=f"(d[2]), "=f"(d[3])
      : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1), "f"(0.0F));
}

// Per-thread accumulator of one 16-row weight tile against eight query rows:
// [0] = (row g, query 2t), [1] = (g, 2t+1), [2] = (g+8, 2t), [3] = (g+8, 2t+1).
struct TileAccumulator {
  float value[4] = {0.0F, 0.0F, 0.0F, 0.0F};

  __device__ __forceinline__ void Fold(const float (&d)[4], float row_scale,
                                       float row8_scale, float act0,
                                       float act1) {
    value[0] = fmaf(d[0], row_scale * act0, value[0]);
    value[1] = fmaf(d[1], row_scale * act1, value[1]);
    value[2] = fmaf(d[2], row8_scale * act0, value[2]);
    value[3] = fmaf(d[3], row8_scale * act1, value[3]);
  }
};

// One 64-value span of an FP8 tile whose 16 rows share one 32x32 scale block:
// `low`/`high` are the k-block 0/1 fragment halves, `block_scales` two E8M0
// bytes (k-block 0 in the low byte).
__device__ __forceinline__ void SpanFp8(TileAccumulator& acc, uint4 low,
                                        uint4 high, uint32_t block_scales,
                                        uint4 act, float4 act_scale) {
  float d[4];
  MmaE4m3(d, low.x, low.z, low.y, low.w, act.x, act.y);
  const float scale0 = PackedE8m0<0>(block_scales);
  acc.Fold(d, scale0, scale0, act_scale.x, act_scale.y);
  MmaE4m3(d, high.x, high.z, high.y, high.w, act.z, act.w);
  const float scale1 = PackedE8m0<1>(block_scales);
  acc.Fold(d, scale1, scale1, act_scale.z, act_scale.w);
}

// The reference `act_quant(..., scale_fmt="ue8m0")` scale exponent for one
// 32-value block maximum: ceil(log2(max(amax, 1e-4) / 448)).
__device__ __forceinline__ int Ue8m0Exponent(float amax) {
  const float ratio = fmaxf(amax, 1.0e-4F) * kFp8MaxInverse;
  const uint32_t bits = __float_as_uint(ratio);
  const int exponent = static_cast<int>((bits >> 23) & 0xFFU) - 127;
  return exponent + ((bits & 0x7FFFFFU) != 0U ? 1 : 0);
}

__device__ __forceinline__ uint8_t QuantizeE4m3(float value, int exponent) {
  // x / 2^e is exact for a power-of-two scale; clamp then round to nearest.
  const float scaled = fminf(
      fmaxf(value * __uint_as_float(static_cast<uint32_t>(127 - exponent) << 23),
            -kFp8Max),
      kFp8Max);
  return static_cast<uint8_t>(
      __nv_cvt_float_to_fp8(scaled, __NV_SATFINITE, __NV_E4M3));
}

__device__ __forceinline__ void StoreFragmentScale(float* scales, int row,
                                                   int block32, int exponent) {
  const int span = block32 / 2;
  const int block = block32 % 2;
  scales[(span * 4 + row / 2) * 4 + 2 * block + (row & 1)] =
      __uint_as_float(static_cast<uint32_t>(exponent + 127) << 23);
}

}  // namespace d41

// ---------------------------------------------------------------- window-KV formats

namespace d41attn {

__device__ __forceinline__ float RoundBf16(float value) {
  return __bfloat162float(__float2bfloat16_rn(value));
}

__device__ __forceinline__ float Bf16ToFloat(__nv_bfloat16 value) {
  return __bfloat162float(value);
}

__device__ __forceinline__ float WarpSum(float value) {
#pragma unroll
  for (int offset = 16; offset > 0; offset >>= 1) {
    value += __shfl_xor_sync(0xffffffffU, value, offset);
  }
  return value;
}

// RMSNorm scale `rsqrt(mean(x^2) + eps)` from a sum of squares over `count` values.
// PyTorch's mean multiplies the sum by the FP32 reciprocal of the count.
__device__ __forceinline__ float RmsScale(float sum_squares, int count, float eps) {
  const float mean = sum_squares * (1.0F / static_cast<float>(count));
  return rsqrtf(mean + eps);
}

// `weight * (x * scale)` rounded to BF16 (two roundings, as in the reference).
__device__ __forceinline__ float RmsApply(float x, float scale, __nv_bfloat16 weight) {
  return RoundBf16(__fmul_rn(Bf16ToFloat(weight), __fmul_rn(x, scale)));
}

}  // namespace d41attn

// ---------------------------------------------------------------- shared projection layout

namespace qproj {

constexpr int kWarps = 8;
constexpr int kCtaRows = 64;
constexpr int kGroupRows = 8;
constexpr int kMaxCtaSpans = 10;
constexpr int kQLora = 1280;
// A warp's 32 x 4 partial values (lane l, value i at l * 5 + i) and one float of
// padding: the stores of a warp and the transpose reads of the tile sums (rows of
// two tiles, lanes four apart) then fall in distinct shared-memory banks.
constexpr int kPartialWarpStride = 32 * 5 + 1;

// Offset of (warp, lane, value) in one group's warp_partial.
__device__ constexpr int PartialIndex(int warp, int lane, int index) {
  return warp * kPartialWarpStride + lane * 5 + index;
}

// kGroups eight-row fragment groups; kFragments: the CTA stages activation fragments of
// its spans.
template <int kGroups, bool kFragments>
struct alignas(16) ProjShared {
  // The staged activation fragments are dead once every warp of the CTA has run its
  // MMAs, so the warps' partial tiles reuse their bytes.
  union {
    struct {
      // [group][span * 32 + lane] 16-byte fragment words, as uint32 quadruples.
      uint32_t fragments[kFragments ? kGroups : 1][kFragments ? kMaxCtaSpans * 32 * 4 : 4];
      // [group][span * 4 + t] float4 fragment scales.
      float scales[kFragments ? kGroups : 1][kFragments ? kMaxCtaSpans * 4 * 4 : 4];
    };
    float warp_partial[kGroups][kWarps * kPartialWarpStride];  // PartialIndex
  };
  float cta_partial[kGroups * kGroupRows][kCtaRows];  // [query][row]
};

}  // namespace qproj

namespace {

// ---------------------------------------------------------------- warp weights

constexpr int kWarps = 8;
constexpr int kThreads = kWarps * 32;
constexpr int kTiles = 4;
constexpr int kCtaRows = kTiles * d41::kTileRows;  // 64
constexpr int kMaxWarpSpans = 5;
constexpr int kQLora = 1280;
constexpr int kMaxClusterRanks = 8;

template <bool kCondition, typename T, typename F>
struct Conditional {
  using type = T;
};
template <typename T, typename F>
struct Conditional<false, T, F> {
  using type = F;
};

// Weight registers of one warp: its spans of tile (warp % 4), loaded before the wait.
// kSpans bounds the warp's span count at compile time and so its register footprint.
template <int kSpans = kMaxWarpSpans>
struct WarpWeights {
  uint4 low[kSpans];
  uint4 high[kSpans];
  uint32_t block_scales[kSpans];
};

// Issue the loads of one warp's `warp_spans` (<= kSpans) weight spans and their
// block scales into `w`.
template <int kSpans>
__device__ __forceinline__ void LoadWarpWeights(WarpWeights<kSpans>& w,
                                                const uint8_t* __restrict__ weight,
                                                const uint8_t* __restrict__ weight_scales,
                                                int k, int group, int first_span,
                                                int warp_spans) {
  const int warp = static_cast<int>(threadIdx.x) >> 5;
  const int lane = static_cast<int>(threadIdx.x) & 31;
  const int spans = k / d41::kSpanValues;
  const int tile = group * kTiles + (warp % kTiles);
  const uint8_t* tile_weight =
      weight + (static_cast<int64_t>(tile) * spans + first_span) * 1024 + lane * 32;
  const uint8_t* tile_scales =
      weight_scales + static_cast<int64_t>(tile / 2) * (k / d41::kBlockValues) + 2 * first_span;
#pragma unroll
  for (int i = 0; i < kSpans; ++i) {
    if (i < warp_spans) {
      w.low[i] = d41::LoadWeight16(tile_weight + static_cast<int64_t>(i) * 1024);
      w.high[i] = d41::LoadWeight16(tile_weight + static_cast<int64_t>(i) * 1024 + 16);
      w.block_scales[i] = *reinterpret_cast<const uint16_t*>(tile_scales + 2 * i);
    }
  }
}

// ---------------------------------------------------------------- the projection cluster kernel

using qproj::kGroupRows;

template <int kGroups, bool kFragments>
using BShared = qproj::ProjShared<kGroups, kFragments>;

template <typename Shared>
__device__ __forceinline__ uint4* GroupFragments(Shared& smem, int group) {
  return reinterpret_cast<uint4*>(smem.fragments[group]);
}

template <typename Shared>
__device__ __forceinline__ float4* GroupScales(Shared& smem, int group) {
  return reinterpret_cast<float4*>(smem.scales[group]);
}

// The rows of fragment group `group` that hold queries.
__device__ __forceinline__ int GroupRows(int rows, int group) {
  return min(kGroupRows, rows - group * kGroupRows);
}

// The inbox wait's stall bound (ns).
constexpr uint64_t kInboxStallNs = 300ULL * 1000 * 1000 * 1000;

// This CTA's shared `pointer` in the shared memory of cluster CTA `rank`.
__device__ __forceinline__ uint32_t ClusterSharedAddress(const void* pointer, int rank) {
  uint32_t address;
  asm volatile("mapa.shared::cluster.u32 %0, %1, %2;"
               : "=r"(address)
               : "r"(static_cast<uint32_t>(__cvta_generic_to_shared(pointer))), "r"(rank));
  return address;
}

// Zero the fragment rows of the last group that hold no query, with unit scales.
template <int kGroups, typename Shared>
__device__ __forceinline__ void ClearInactiveRowsB(Shared& smem, int spans, int rows) {
  const int group = kGroups - 1;
  const int active = GroupRows(rows, group);
  const int lanes = (kGroupRows - active) * 4;
  uint4* fragments = GroupFragments(smem, group);
  for (int i = static_cast<int>(threadIdx.x); i < lanes * spans; i += kThreads) {
    fragments[(i / lanes) * 32 + 4 * active + i % lanes] = make_uint4(0U, 0U, 0U, 0U);
  }
  const int inactive = kGroupRows - active;
  float* scales = smem.scales[group];
  for (int i = static_cast<int>(threadIdx.x); i < inactive * 2 * spans; i += kThreads) {
    d41::StoreFragmentScale(scales, active + i % inactive, i / inactive, 0);
  }
}

// The warp's partial tile over its spans, fragments staged in smem (the original's
// one-group branch).
template <int kGroups, typename Shared>
__device__ __forceinline__ void WarpMmaB(Shared& smem, const WarpWeights<>& w, int local_first,
                                         int warp_spans, float (&out)[kGroups][4]) {
  static_assert(kGroups == 1, "one fragment group");
  const int lane = static_cast<int>(threadIdx.x) & 31;
#pragma unroll
  for (int group = 0; group < kGroups; ++group) {
    const uint4* fragments = GroupFragments(smem, group);
    const float4* scales = GroupScales(smem, group);
    d41::TileAccumulator acc;
#pragma unroll
    for (int i = 0; i < kMaxWarpSpans; ++i) {
      if (i < warp_spans) {
        const int span = local_first + i;
        d41::SpanFp8(acc, w.low[i], w.high[i], w.block_scales[i],
                     fragments[span * 32 + lane], scales[span * 4 + (lane & 3)]);
      }
    }
#pragma unroll
    for (int index = 0; index < 4; ++index) out[group][index] = acc.value[index];
  }
}

// The push reductions. `sources` CTAs (cluster ranks base .. base + sources - 1) each
// hold a partial of the 8F x 64 outputs of one column tile (query-major: output = query *
// 64 + row). Owner o, the CTA of rank base + o, stores outputs o * share .. (o + 1) *
// share - 1, with share = 8F x 64 / sources. Sender s stores its partial of an output
// straight into slot s * share + output % share of the owner's cta_partial, read as one
// flat array of sources x share = 8F x 64 floats, and the store completes on the owner's
// inbox mbarrier. The owner adds its slots in sender order from 0.0. Protocol:
//
//   1. Before the grid wait, thread 0 initializes the inbox (one arrival), fences the
//      initialization and, for an owner of at least one output whose query is below `rows`,
//      arrives on it expecting sources x 4 bytes per such output (InitPushInbox). Every
//      thread then arrives on the cluster barrier that precedes the grid wait.
//   2. After its MMAs and the cluster wait, every CTA stores its partial of each output
//      whose query is below `rows` into the owner's slot (PushPartialsB).
//   3. An owner of such an output waits on its inbox (trapping after kInboxStallNs), adds
//      and stores y (PushSumStoreB). A CTA that owns none exits after its stores, which
//      read only registers.
//
// Invariants: (I1) no store completes on an uninitialized mbarrier: every store follows a
// cluster wait that needs every rank's arrival, made after its initialization and fence;
// (I2) exact byte counts: an owner's senders store exactly the bytes it expects, into
// distinct slots, each once; (I3) no store into an exited CTA: every store into a CTA
// completes on its inbox, which it waits on before exiting, and a CTA that owns no such
// output receives none (the fragment stores complete on the receive barrier, which every
// CTA waits on before its MMAs); (I4) progress: after the grid wait a CTA blocks only on
// its own cluster. cta_partial lies outside the union that the staging and the MMAs use.

// The bytes owner `owner` of `sources` senders expects: sources x 4 for each of its `share`
// outputs whose query is below `rows`.
__device__ __forceinline__ uint32_t PushInboxBytes(int rows, int owner, int sources,
                                                   int share) {
  const int valid = min(max(rows * kCtaRows - owner * share, 0), share);
  return static_cast<uint32_t>(valid * sources * 4);
}

// Step 1 for thread 0: the inbox barrier (one arrival), its fence and the expect_tx of owner
// `owner` of `sources` senders with `share` outputs.
__device__ __forceinline__ void InitPushInbox(uint64_t* inbox, int rows, int owner, int sources,
                                              int share) {
  if (threadIdx.x == 0) {
    const uint32_t bytes = PushInboxBytes(rows, owner, sources, share);
    const uint32_t address = static_cast<uint32_t>(__cvta_generic_to_shared(inbox));
    asm volatile("mbarrier.init.shared::cta.b64 [%0], 1;" ::"r"(address) : "memory");
    asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
    if (bytes > 0) {
      asm volatile("mbarrier.arrive.expect_tx.release.cta.shared::cta.b64 _, [%0], %1;" ::"r"(
                       address),
                   "r"(bytes)
                   : "memory");
    }
  }
}

// Step 2's first half, for each of the CTA's eight warps: its partial tiles into
// warp_partial.
template <int kGroups, typename Shared>
__device__ __forceinline__ void StorePartialsB(Shared& smem, const float (&acc)[kGroups][4]) {
  const int warp = static_cast<int>(threadIdx.x) >> 5;
  const int lane = static_cast<int>(threadIdx.x) & 31;
#pragma unroll
  for (int group = 0; group < kGroups; ++group) {
#pragma unroll
    for (int index = 0; index < 4; ++index) {
      smem.warp_partial[group][qproj::PartialIndex(warp, lane, index)] = acc[group][index];
    }
  }
}

// Step 2's second half, once warp_partial is complete, for the CTA's 256 threads: the
// two-warp tile sums of every output whose query is below `rows`, stored into the inboxes
// of the `sources` owners as sender `sender` (share = 8F x 64 / sources).
template <int kGroups, typename Shared>
__device__ __forceinline__ void PushOutputsB(Shared& smem, uint64_t* inbox, int base, int sender,
                                             int sources, int share, int rows) {
  float* const slots = &smem.cta_partial[0][0];
#pragma unroll
  for (int pass = 0; pass < 2 * kGroups; ++pass) {
    const int output = static_cast<int>(threadIdx.x) + pass * kThreads;  // group*512 + q*64 + r
    if (output < rows * kCtaRows) {
      const int group = output / (kGroupRows * kCtaRows);
      const int query = (output / kCtaRows) % kGroupRows;
      const int row = output % kCtaRows;
      const int tile = row / 16;
      const int within = row % 16;
      const int source_lane = (within % 8) * 4 + query / 2;
      const int source_index = (within / 8) * 2 + (query & 1);
      const float value =
          smem.warp_partial[group][qproj::PartialIndex(tile, source_lane, source_index)] +
          smem.warp_partial[group][qproj::PartialIndex(tile + kTiles, source_lane,
                                                           source_index)];
      // output / share, as a division by a constant: owner * share <= output < (owner + 1)
      // * share, and share * sources = 8F x 64.
      const int owner = output * sources / (kGroups * kGroupRows * kCtaRows);
      const int rank = base + owner;
      asm volatile(
          "st.async.shared::cluster.mbarrier::complete_tx::bytes.f32 [%0], %1, [%2];" ::"r"(
              ClusterSharedAddress(slots + sender * share + (output - owner * share), rank)),
          "f"(value), "r"(ClusterSharedAddress(inbox, rank))
          : "memory");
    }
  }
}

// Step 2: StorePartialsB and PushOutputsB between the CTA barriers that order them.
template <int kGroups, typename Shared>
__device__ __forceinline__ void PushPartialsB(Shared& smem, uint64_t* inbox,
                                              const float (&acc)[kGroups][4], int base,
                                              int sender, int sources, int share, int rows) {
  // warp_partial shares its bytes with the staged fragments.
  __syncthreads();
  StorePartialsB<kGroups>(smem, acc);
  __syncthreads();
  PushOutputsB<kGroups>(smem, inbox, base, sender, sources, share, rows);
}

// Step 3's wait: phase 0 of `inbox`, completed by the senders' st.async. A thread that has
// waited kInboxStallNs sees a broken exchange and traps. The clock is read only after a
// failed try_wait.
__device__ __forceinline__ void WaitInboxB(uint64_t* inbox) {
  const uint32_t barrier = static_cast<uint32_t>(__cvta_generic_to_shared(inbox));
  asm volatile(
      "{\n\t.reg .pred done;\n\t.reg .u64 t0, t1;\n\t"
      "mov.u64 t0, 0;\n\t"
      "WAIT_%=:\n\t"
      "mbarrier.try_wait.parity.acquire.cluster.shared::cta.b64 done, [%0], 0;\n\t"
      "@done bra DONE_%=;\n\t"
      "mov.u64 t1, %%globaltimer;\n\t"
      "setp.eq.u64 done, t0, 0;\n\t"
      "@done mov.u64 t0, t1;\n\t"
      "sub.u64 t1, t1, t0;\n\t"
      "setp.gt.u64 done, t1, %1;\n\t"
      "@done trap;\n\t"
      "bra WAIT_%=;\n\t"
      "DONE_%=:\n\t}" ::"r"(barrier),
      "l"(kInboxStallNs)
      : "memory");
}

// Step 3: owner `owner` waits for its inbox, adds its `sources` slots of each output in
// sender order and stores y (query-major outputs owner * share ..) as BF16.
template <typename Shared>
__device__ __forceinline__ void PushSumStoreB(Shared& smem, uint64_t* inbox,
                                              __nv_bfloat16* __restrict__ y, int y_stride,
                                              int row_base, int rows, int owner, int sources,
                                              int share) {
  if (static_cast<int>(threadIdx.x) >= share ||
      PushInboxBytes(rows, owner, sources, share) == 0) {
    return;
  }
  WaitInboxB(inbox);
  const float* const slots = &smem.cta_partial[0][0];
  for (int local = static_cast<int>(threadIdx.x); local < share; local += kThreads) {
    const int output = owner * share + local;
    const int query = output / kCtaRows;
    const int row = output % kCtaRows;
    if (query < rows) {
      float sum = 0.0F;
#pragma unroll
      for (int source = 0; source < kMaxClusterRanks; ++source) {
        if (source < sources) sum += slots[source * share + local];
      }
      y[static_cast<int64_t>(query) * y_stride + row_base + row] = __float2bfloat16_rn(sum);
    }
  }
}

// q_norm + MXFP8 quant + wq_b for one fragment group (at most eight rows), in clusters that
// share the quantized rows.
//
// Geometry. Grid (2, N/64): split x = blockIdx.x owns spans 10x .. 10x + 9 of K = 1280,
// column tile blockIdx.y owns 64 outputs. The clusters are (2, T, 1), the two splits of T
// consecutive column tiles (T = 4 here: 8 CTAs); cluster CTA (x, t) has rank x + 2t. A
// cluster runs inside one GPC; at one CTA per SM a GB300 fits only 15 clusters of 8, so the
// fit build runs N = 4096 in 15 clusters (below).
//
// Fragments. Every split-x CTA needs spans 10x .. 10x + 9 of all eight fragment rows. Each
// fragment row r < rows is produced once per cluster: at T = 4 the CTA of rank r produces
// both halves. The producer
//   - forms row r's q_norm RMS: thread j adds the squares of columns j, j + 256, ... in
//     column order, the warps reduce with xor shuffles, and the eight warp sums are added in
//     warp order;
//   - quantizes each half with 40 lanes: lane u owns span u / 4 of the half and the 16
//     values whose fragment bytes form one 16-byte word of that span (columns 8 (u % 4) ..
//     + 7 of each of its two 32-value blocks). It applies q_norm and quantizes: each block's
//     UE8M0 exponent from the amax of its 32 values (the span's four lanes exchange it),
//     then every E4M3 byte on its own;
//   - stores the word, and for u % 4 < 2 the scale of block u % 4, into the staged
//     fragments of the cluster's T CTAs of that half's split (itself included, when it is
//     one) with st.async, which completes the bytes on each receiver's mbarrier.
// A receiver expects rows x 720 bytes. It zeroes the rows >= `rows` itself before the wait.
//
// Projection. Once its mbarrier completes, a CTA runs the MMAs on the staged fragments and
// reduces with the other split of its column tile (ranks 2t and 2t + 1) by push.
//
// Synchronization. Thread 0 initializes the mbarrier (one arrival: its own expect_tx) and
// fences the init for the cluster. Every thread arrives (relaxed) at one barrier.cluster
// phase and waits for it, both before the grid wait, so every peer has started and
// initialized its mbarrier before any st.async. The receive wait acquires at cluster scope.
// After its grid wait the kernel blocks only on its own cluster, never on a dependent. The
// receive wait traps after 300 s.
//
// The early-trigger build (Q_EARLY_TRIGGER) triggers the dependents after the receive wait,
// before the MMAs, in place of after them. It still follows the grid wait, so after it the
// kernel blocks only on its own cluster. Between the two points the MMAs only read shared
// memory and the read-only weights and scales, and this kernel's one global write (q)
// follows both, so dependents that start earlier see no new write before their own
// griddepcontrol.wait, which waits for this whole grid.
//
// The fit build (entry dsv41_q_cluster8_fit) runs N = 4096, 16 clusters of 8, in the 15
// clusters that fit one CTA per SM. Grid (2, 60), 320 threads. Warps 0-7 run the ordinary
// kernel for column tiles 0 .. 59. The 256 outputs of the dropped cluster are 16 tiles of 16
// rows; pair `group` computes leftover tile j = (group % 4) * 15 + group / 4 when j < 16, rows
// 3840 + 16 j .. + 15, on two extra warps: warp 8 over spans 0-4 of the split's half and
// warp 9 over spans 5-9, on the same staged fragments. They add no RMS, quantization or
// hand-off and join every CTA barrier. The leftover tile reduces by the pair push into a
// second inbox (QClusterFitShared): split x owns the tile's queries 4x .. 4x + 3 and adds
// the two splits' partials from 0.0 in split order, the ordinary pair reduction's order.
constexpr int kQSplits = 2;
constexpr int kQCtaSpans = kQLora / d41::kSpanValues / kQSplits;  // 10
constexpr int kQHalfUnits = 4 * kQCtaSpans;                       // producer lanes per half
// Bytes one active fragment row delivers to each CTA of a split: 10 spans x 64 fragment
// bytes and 20 four-byte block scales.
constexpr uint32_t kQRowBytes = kQCtaSpans * 64 + 2 * kQCtaSpans * 4;
static_assert(kGroupRows * kCtaRows / kQSplits == kThreads, "one reduced output per thread");

// kRowSlots: the fragment rows each CTA produces (2 in clusters of 4 CTAs, else 1).
template <int kRowSlots>
struct QClusterShared {
  BShared<1, true> proj;
  float warp_sums[kRowSlots][kWarps];
  uint64_t received;  // completes when the cluster's st.async of this CTA's fragments land
  uint64_t inbox;     // the pair reduction's (PushPartialsB; completes when its partials land)
};

// The fit build's extra warps, for one 16-row tile.
constexpr int kFitWarps = 2;
constexpr int kFitThreads = kFitWarps * 32;
// Outputs of one leftover tile (eight queries x 16 rows, query-major) and each split's share.
constexpr int kFitOutputs = kGroupRows * d41::kTileRows;
constexpr int kFitShare = kFitOutputs / kQSplits;
static_assert(kFitShare == kFitThreads, "one owned leftover output per extra thread");

// QClusterShared with the fit build's leftover tile: the extra warps' partials, its inbox
// slots ([sender split][owned output]) and their mbarrier. None shares bytes with the staged
// fragments.
template <int kRowSlots>
struct QClusterFitShared : QClusterShared<kRowSlots> {
  float fit_partial[kFitWarps][qproj::kPartialWarpStride];  // PartialIndex(0, ...)
  float fit_slots[kQSplits * kFitShare];
  uint64_t fit_inbox;  // completes when the pair's leftover partials land
};

__device__ __forceinline__ uint32_t SharedAddress(const void* pointer) {
  return static_cast<uint32_t>(__cvta_generic_to_shared(pointer));
}

// shared::cluster address of this CTA's shared `address` in cluster CTA `rank`.
__device__ __forceinline__ uint32_t MapShared(uint32_t address, uint32_t rank) {
  uint32_t out;
  asm volatile("mapa.shared::cluster.u32 %0, %1, %2;" : "=r"(out) : "r"(address), "r"(rank));
  return out;
}

// 16 bytes into a cluster CTA's shared memory, completing on that CTA's mbarrier (both
// shared::cluster addresses from MapShared).
__device__ __forceinline__ void StoreAsync16(uint32_t remote, uint4 value, uint32_t barrier) {
  asm volatile(
      "st.async.shared::cluster.mbarrier::complete_tx::bytes.v4.b32 [%0], {%1, %2, %3, %4}, "
      "[%5];" ::"r"(remote),
      "r"(value.x), "r"(value.y), "r"(value.z), "r"(value.w), "r"(barrier)
      : "memory");
}

// StoreAsync16 for 4 bytes.
__device__ __forceinline__ void StoreAsync4(uint32_t remote, uint32_t value, uint32_t barrier) {
  asm volatile(
      "st.async.shared::cluster.mbarrier::complete_tx::bytes.b32 [%0], %1, [%2];" ::"r"(remote),
      "r"(value), "r"(barrier)
      : "memory");
}

// Stall hook of WaitReceived: a CTA whose fragments have not all landed after kQStallNs
// traps, so a lost or miscounted st.async fails the step with a CUDA error from this kernel
// instead of hanging the instance. The clock is read once every kQStallRounds failed polls.
constexpr uint64_t kQStallNs = 300ULL * 1000 * 1000 * 1000;
constexpr unsigned kQStallRounds = 64;

// Wait for phase 0 of an mbarrier that cluster peers complete with st.async.
__device__ __forceinline__ void WaitReceived(uint32_t barrier) {
  uint32_t done = 0;
  unsigned rounds = 0;
  uint64_t since = 0;
  while (true) {
    asm volatile(
        "{ .reg .pred p; mbarrier.try_wait.parity.acquire.cluster.shared::cta.b64 p, [%1], 0; "
        "selp.u32 %0, 1, 0, p; }"
        : "=r"(done)
        : "r"(barrier)
        : "memory");
    if (done) return;
    if (++rounds % kQStallRounds != 0) continue;
    uint64_t now;
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(now));
    if (since == 0) {
      since = now;
    } else if (now - since > kQStallNs) {
      __trap();
    }
  }
}

// LoadWarpWeights of an explicit 16-row weight tile `tile`: the fit build's warps 8 and 9
// load spans of a leftover tile, not of their CTA's column tile.
template <int kSpans>
__device__ __forceinline__ void LoadTileWeights(WarpWeights<kSpans>& w,
                                                const uint8_t* __restrict__ weight,
                                                const uint8_t* __restrict__ weight_scales,
                                                int k, int tile, int first_span,
                                                int warp_spans) {
  const int lane = static_cast<int>(threadIdx.x) & 31;
  const int spans = k / d41::kSpanValues;
  const uint8_t* tile_weight =
      weight + (static_cast<int64_t>(tile) * spans + first_span) * 1024 + lane * 32;
  const uint8_t* tile_scales =
      weight_scales + static_cast<int64_t>(tile / 2) * (k / d41::kBlockValues) + 2 * first_span;
#pragma unroll
  for (int i = 0; i < kSpans; ++i) {
    if (i < warp_spans) {
      w.low[i] = d41::LoadWeight16(tile_weight + static_cast<int64_t>(i) * 1024);
      w.high[i] = d41::LoadWeight16(tile_weight + static_cast<int64_t>(i) * 1024 + 16);
      w.block_scales[i] = *reinterpret_cast<const uint16_t*>(tile_scales + 2 * i);
    }
  }
}

// The bytes owner `owner` (a split) of a pair that carries a leftover tile expects in its
// fit_inbox: 2 x 4 for each of its outputs (queries 4 owner ..) whose query is below `rows`.
__device__ __forceinline__ uint32_t FitInboxBytes(int rows, int owner) {
  const int valid = min(max(rows * d41::kTileRows - owner * kFitShare, 0), kFitShare);
  return static_cast<uint32_t>(valid * kQSplits * 4);
}

// InitPushInbox for fit_inbox (thread 0, before the cluster barrier): its expect_tx only
// when the pair carries a leftover tile and `owner` owns one of its active outputs.
__device__ __forceinline__ void InitFitInbox(uint64_t* inbox, int rows, int owner,
                                             bool carries) {
  if (threadIdx.x == 0) {
    const uint32_t bytes = carries ? FitInboxBytes(rows, owner) : 0U;
    const uint32_t address = static_cast<uint32_t>(__cvta_generic_to_shared(inbox));
    asm volatile("mbarrier.init.shared::cta.b64 [%0], 1;" ::"r"(address) : "memory");
    asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
    if (bytes > 0) {
      asm volatile("mbarrier.arrive.expect_tx.release.cta.shared::cta.b64 _, [%0], %1;" ::"r"(
                       address),
                   "r"(bytes)
                   : "memory");
    }
  }
}

// Step 2 of both pushes in the fit build. Warps 0-7 run PushPartialsB<1> (StorePartialsB,
// PushOutputsB), and warps 8 and 9 join its two CTA barriers. Between them, the extra warps
// of a pair that carries a leftover tile store their partials into fit_partial; after the
// second, their 64 threads add the two warps' partials of each active leftover output
// (query * 16 + row) and store the sum into slot sender x 64 + output % 64 of its owner
// (the pair's split output / 64, rank base + owner), completing on that CTA's fit_inbox.
template <typename Shared>
__device__ __forceinline__ void PushPartialsFitB(Shared& shared, const float (&acc)[1][4],
                                                 int base, int sender, int rows, bool carries) {
  const int warp = static_cast<int>(threadIdx.x) >> 5;
  const int lane = static_cast<int>(threadIdx.x) & 31;
  // warp_partial shares its bytes with the staged fragments.
  __syncthreads();
  if (warp < kWarps) {
    StorePartialsB<1>(shared.proj, acc);
  } else if (carries) {
#pragma unroll
    for (int index = 0; index < 4; ++index) {
      shared.fit_partial[warp - kWarps][qproj::PartialIndex(0, lane, index)] = acc[0][index];
    }
  }
  __syncthreads();
  if (warp < kWarps) {
    PushOutputsB<1>(shared.proj, &shared.inbox, base, sender, kQSplits, kThreads, rows);
  } else if (carries) {
    const int local = static_cast<int>(threadIdx.x) - kThreads;
#pragma unroll
    for (int pass = 0; pass < kFitOutputs / kFitThreads; ++pass) {
      const int output = local + pass * kFitThreads;
      if (output < rows * d41::kTileRows) {
        const int query = output / d41::kTileRows;
        const int row = output % d41::kTileRows;
        const int source_lane = (row % 8) * 4 + query / 2;
        const int source_index = (row / 8) * 2 + (query & 1);
        const float value =
            shared.fit_partial[0][qproj::PartialIndex(0, source_lane, source_index)] +
            shared.fit_partial[1][qproj::PartialIndex(0, source_lane, source_index)];
        const int owner = output / kFitShare;
        const int rank = base + owner;
        asm volatile(
            "st.async.shared::cluster.mbarrier::complete_tx::bytes.f32 [%0], %1, [%2];" ::"r"(
                ClusterSharedAddress(
                    shared.fit_slots + sender * kFitShare + (output - owner * kFitShare), rank)),
            "f"(value), "r"(ClusterSharedAddress(&shared.fit_inbox, rank))
            : "memory");
      }
    }
  }
}

// Step 3 for warps 8 and 9 of a pair that carries a leftover tile: owner `owner` waits for
// fit_inbox, adds the two splits' slots of each owned output in split order from 0.0 (as
// PushSumStoreB) and stores q rows `row_base` .. + 15 as BF16.
template <typename Shared>
__device__ __forceinline__ void FitSumStoreB(Shared& shared, __nv_bfloat16* __restrict__ q,
                                             int n, int row_base, int rows, int owner) {
  if (FitInboxBytes(rows, owner) == 0) return;
  WaitInboxB(&shared.fit_inbox);
  const int local = static_cast<int>(threadIdx.x) - kThreads;
  const int output = owner * kFitShare + local;
  const int query = output / d41::kTileRows;
  const int row = output % d41::kTileRows;
  if (query < rows) {
    float sum = 0.0F;
#pragma unroll
    for (int source = 0; source < kQSplits; ++source) {
      sum += shared.fit_slots[source * kFitShare + local];
    }
    q[static_cast<int64_t>(query) * n + row_base + row] = __float2bfloat16_rn(sum);
  }
}

// The kernel body for clusters of the two splits of kClusterTiles column tiles; kFit: the
// fit build's entry. The output q has row stride n.
template <int kClusterTiles, bool kFit = false>
__device__ __forceinline__ void QClusterProjection(const __nv_bfloat16* qkv,
                                             const __nv_bfloat16* __restrict__ q_norm,
                                             const uint8_t* __restrict__ weight,
                                             const uint8_t* __restrict__ weight_scales,
                                             __nv_bfloat16* __restrict__ q, int rows, int n,
                                             int qkv_stride, float eps) {
  static_assert(kClusterTiles == 8 || kClusterTiles == 4 || kClusterTiles == 2,
                "clusters of 16, 8 or 4 CTAs");
  static_assert(!kFit || kClusterTiles == 4, "the fit build: clusters of 8");
  // Halves of a fragment row each CTA produces: 1 at 16 CTAs, 2 at 8 and 4.
  constexpr int kHalves = kQSplits * kClusterTiles == 2 * kGroupRows ? 1 : 2;
  constexpr int kUnits = kHalves * kQHalfUnits;
  // Fragment rows each CTA produces: 1 at 16 and 8 CTAs; 2 at 4, rows `row` and `row` + 4.
  constexpr int kRowSlots = kHalves == 1 ? 1 : kGroupRows / (kQSplits * kClusterTiles);
  constexpr int kRowStride = kQSplits * kClusterTiles;
  __shared__ typename Conditional<kFit, QClusterFitShared<kRowSlots>,
                                  QClusterShared<kRowSlots>>::type shared;
  auto& smem = shared.proj;
  const int warp = static_cast<int>(threadIdx.x) >> 5;
  const int lane = static_cast<int>(threadIdx.x) & 31;
  const int split = static_cast<int>(blockIdx.x);
  const int group = static_cast<int>(blockIdx.y);
  const int tile = group % kClusterTiles;  // cluster CTA (split, tile)
  const int row = kHalves == 1 ? tile : split + kQSplits * tile;  // the row it produces
  // The fit build: warps 8 and 9, and the leftover tile this pair computes when `fit_tile`
  // is below the leftover count.
  const bool fit_warp = kFit && warp >= kWarps;
  const int fit_tile = (group % kClusterTiles) * (static_cast<int>(gridDim.y) / kClusterTiles) +
                       group / kClusterTiles;
  const int fit_base = static_cast<int>(gridDim.y) * kTiles;  // the first leftover tile
  const bool carries = kFit && fit_tile < n / d41::kTileRows - fit_base;
  constexpr int kWarpSpans = kQCtaSpans / 2;
  const int local_first = (fit_warp ? warp - kWarps : warp / kTiles) * kWarpSpans;
  WarpWeights<> w;
  if (fit_warp) {
    if (carries) {
      LoadTileWeights(w, weight, weight_scales, kQLora, fit_base + fit_tile,
                      split * kQCtaSpans + local_first, kWarpSpans);
    }
  } else {
    LoadWarpWeights(w, weight, weight_scales, kQLora, group, split * kQCtaSpans + local_first,
                    kWarpSpans);
  }
  // Producer lane `unit` < kUnits of row slot `slot`: half `half` of the slot's row, local
  // span `span` of that half, columns 8 `sub` .. + 7 of both of its blocks; k is block 0's
  // first column (block 1's is k + 32). Its words go to the cluster's CTAs of split `half`.
  const int slot = kRowSlots == 1 ? 0 : static_cast<int>(threadIdx.x) / kUnits;
  const int unit = static_cast<int>(threadIdx.x) - slot * kUnits;
  const bool quantizes = static_cast<int>(threadIdx.x) < kRowSlots * kUnits;
  const int half = kHalves == 1 ? split : unit / kQHalfUnits;
  const int span = (unit % kQHalfUnits) / 4;
  const int sub = unit % 4;
  const int k = (half * kQCtaSpans + span) * d41::kSpanValues + sub * 8;
  // The lane's q_norm weights depend on nothing the predecessor writes: loaded before the
  // wait.
  uint4 norm_words[2] = {};
  if (quantizes) {
#pragma unroll
    for (int block = 0; block < 2; ++block) {
      norm_words[block] =
          d41::LoadWeight16(reinterpret_cast<const uint8_t*>(q_norm + k + 32 * block));
    }
  }
  const uint32_t received = SharedAddress(&shared.received);
  // Step 1 of the pair push (PushPartialsB): CTA `split` of the pair owns outputs
  // split * 256 .. of the tile. The cluster barrier below covers it.
  InitPushInbox(&shared.inbox, rows, split, kQSplits, kThreads);
  if constexpr (kFit) InitFitInbox(&shared.fit_inbox, rows, split, carries);
  if (threadIdx.x == 0) {
    asm volatile("mbarrier.init.shared::cta.b64 [%0], 1;" ::"r"(received) : "memory");
    asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
    asm volatile("mbarrier.arrive.expect_tx.release.cta.shared::cta.b64 _, [%0], %1;" ::"r"(
                     received),
                 "r"(static_cast<uint32_t>(rows) * kQRowBytes)
                 : "memory");
  }
  if (!fit_warp) ClearInactiveRowsB<1>(smem, kQCtaSpans, rows);
  __syncthreads();  // the cleared rows and the initialized mbarrier, for every warp
  asm volatile("barrier.cluster.arrive.relaxed.aligned;" ::: "memory");
  asm volatile("barrier.cluster.wait.aligned;" ::: "memory");  // every peer has started
  d41::GridDependencyWait();
  if (row < rows) {  // uniform over the CTA
    const __nv_bfloat16* source = qkv + static_cast<int64_t>(row) * qkv_stride;
    // The lane's row (its slot's) and whether it is active; one row per CTA keeps `quantizes`.
    const int unit_row = row + kRowStride * slot;
    const bool active = kRowSlots == 1 ? quantizes : quantizes && unit_row < rows;
    const __nv_bfloat16* unit_source =
        kRowSlots == 1 ? source : qkv + static_cast<int64_t>(unit_row) * qkv_stride;
    uint4 raw[2] = {};
    if (active) {
#pragma unroll
      for (int block = 0; block < 2; ++block) {
        raw[block] = *reinterpret_cast<const uint4*>(unit_source + k + 32 * block);
      }
    }
    // Every thread adds the squares of each produced row, one sum per row (the second only
    // in clusters of 4, when row + 4 is active). The fit build's warps 8 and 9 add none, so
    // threads 0-255 keep that order.
    float squares[kRowSlots] = {};
    if (!fit_warp) {
#pragma unroll
      for (int index = threadIdx.x; index < kQLora; index += kThreads) {
        const float value = d41attn::Bf16ToFloat(source[index]);
        squares[0] = __fadd_rn(squares[0], __fmul_rn(value, value));
        if constexpr (kRowSlots == 2) {
          if (row + kRowStride < rows) {  // uniform over the CTA
            const float other = d41attn::Bf16ToFloat(
                source[static_cast<int64_t>(kRowStride) * qkv_stride + index]);
            squares[1] = __fadd_rn(squares[1], __fmul_rn(other, other));
          }
        }
      }
    }
#pragma unroll
    for (int s = 0; s < kRowSlots; ++s) {
      const float sum = d41attn::WarpSum(squares[s]);
      if (lane == 0 && !fit_warp) shared.warp_sums[s][warp] = sum;
    }
    __syncthreads();
    if (warp < (kRowSlots * kUnits + 31) / 32) {  // whole warps (shuffles)
      float total = 0.0F;
#pragma unroll
      for (int index = 0; index < kWarps; ++index) total += shared.warp_sums[slot][index];
      const float scale = d41attn::RmsScale(total, kQLora, eps);
      float values[2][8];
      float amax[2];
#pragma unroll
      for (int block = 0; block < 2; ++block) {
        const uint32_t words[4] = {raw[block].x, raw[block].y, raw[block].z, raw[block].w};
        const uint32_t norms[4] = {norm_words[block].x, norm_words[block].y,
                                   norm_words[block].z, norm_words[block].w};
        amax[block] = 0.0F;
#pragma unroll
        for (int index = 0; index < 8; ++index) {
          const uint32_t pair = words[index / 2];
          const uint32_t norm = norms[index / 2];
          const __nv_bfloat16 value = __ushort_as_bfloat16(
              static_cast<unsigned short>(index % 2 ? pair >> 16 : pair));
          const __nv_bfloat16 weight_value = __ushort_as_bfloat16(
              static_cast<unsigned short>(index % 2 ? norm >> 16 : norm));
          values[block][index] =
              active ? d41attn::RmsApply(d41attn::Bf16ToFloat(value), scale, weight_value)
                     : 0.0F;
          amax[block] = fmaxf(amax[block], fabsf(values[block][index]));
        }
#pragma unroll
        for (int offset = 2; offset > 0; offset >>= 1) {
          amax[block] = fmaxf(amax[block], __shfl_xor_sync(0xffffffffU, amax[block], offset));
        }
      }
      if (active) {
        int exponents[2];
        uint32_t packed[4];
#pragma unroll
        for (int block = 0; block < 2; ++block) {
          exponents[block] = d41::Ue8m0Exponent(amax[block]);
#pragma unroll
          for (int part = 0; part < 2; ++part) {
            uint32_t word = 0U;
#pragma unroll
            for (int index = 0; index < 4; ++index) {
              word |= static_cast<uint32_t>(
                          d41::QuantizeE4m3(values[block][4 * part + index], exponents[block]))
                      << (8 * index);
            }
            packed[2 * block + part] = word;
          }
        }
        const uint4 pack = make_uint4(packed[0], packed[1], packed[2], packed[3]);
        // The word's place in the staged fragments and, for sub < 2, block sub's scale.
        const uint32_t fragment =
            SharedAddress(GroupFragments(smem, 0) + span * 32 + unit_row * 4 + sub);
        const uint32_t scale_slot = SharedAddress(
            &smem.scales[0][(span * 4 + unit_row / 2) * 4 + 2 * (sub & 1) + (unit_row & 1)]);
        const int exponent = (sub & 1) != 0 ? exponents[1] : exponents[0];
        const uint32_t scale_bits = static_cast<uint32_t>(exponent + 127) << 23;
#pragma unroll
        for (int target = 0; target < kClusterTiles; ++target) {
          const uint32_t rank = static_cast<uint32_t>(half + kQSplits * target);
          const uint32_t barrier = MapShared(received, rank);
          StoreAsync16(MapShared(fragment, rank), pack, barrier);
          if (sub < 2) StoreAsync4(MapShared(scale_slot, rank), scale_bits, barrier);
        }
      }
    }
  }
  WaitReceived(received);
#if Q_EARLY_TRIGGER
  // The early-trigger build: before the MMAs.
  d41::LaunchDependents();
#endif
  float acc[1][4];
  if (!fit_warp || carries) WarpMmaB<1>(smem, w, local_first, kWarpSpans, acc);
#if !Q_EARLY_TRIGGER
  // After the MMAs.
  d41::LaunchDependents();
#endif
  // The pair reduction by push (PushPartialsB): the pair is ranks 2 tile and 2 tile + 1, and
  // its owners keep the pull's outputs and order. The cluster wait is the one above.
  if constexpr (kFit) {
    PushPartialsFitB(shared, acc, kQSplits * tile, split, rows, carries);
  } else {
    PushPartialsB<1>(smem, &shared.inbox, acc, kQSplits * tile, split, kQSplits, kThreads, rows);
  }
  // Threads 256-319 of the fit build return at once (the share is 256 outputs).
  PushSumStoreB(smem, &shared.inbox, q, n, group * kCtaRows, rows, split, kQSplits, kThreads);
  if constexpr (kFit) {
    if (fit_warp && carries) {
      FitSumStoreB(shared, q, n, (fit_base + fit_tile) * d41::kTileRows, rows, split);
    }
  }
}

}  // namespace

// Grid (2, N/64), cluster (2, 4, 1), 256 threads, static shared memory:
// q [rows][N] = MXFP8(q_norm(qkv[:, :1280])) W^T for rows <= 8.
extern "C" __global__ __launch_bounds__(kThreads) void dsv41_q_cluster8(
    const __nv_bfloat16* qkv, const __nv_bfloat16* __restrict__ q_norm,
    const uint8_t* __restrict__ weight, const uint8_t* __restrict__ weight_scales,
    __nv_bfloat16* __restrict__ q, int rows, int n, int qkv_stride, float eps) {
  QClusterProjection<4>(qkv, q_norm, weight, weight_scales, q, rows, n, qkv_stride, eps);
}

// Grid (2, 60), cluster (2, 4, 1), 320 threads, static shared memory: dsv41_q_cluster8's
// q [rows][4096] in 15 clusters, warps 8 and 9 computing the dropped cluster's 16-row tiles.
extern "C" __global__ __launch_bounds__(kThreads + kFitThreads) void dsv41_q_cluster8_fit(
    const __nv_bfloat16* qkv, const __nv_bfloat16* __restrict__ q_norm,
    const uint8_t* __restrict__ weight, const uint8_t* __restrict__ weight_scales,
    __nv_bfloat16* __restrict__ q, int rows, int n, int qkv_stride, float eps) {
  QClusterProjection<4, true>(qkv, q_norm, weight, weight_scales, q, rows, n, qkv_stride, eps);
}
