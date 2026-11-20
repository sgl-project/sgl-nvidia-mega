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
// DeepSeek-V4.1 attention output projection wo_b at 1-8 rows that publishes its rows into
// hop A's Lamport mailbox (slot wo_b_publish_split): cluster split-K over S CTAs, every
// warp's FP8 weight spans in registers before griddepcontrol.wait, warp-level FP8
// mma.sync with each 32-value k-block folded into FP32 with its two UE8M0 scales, the S
// partial tiles summed in cluster rank order, and each CTA's share of the output packed
// into 16-byte BF16 packs that go to y and to the mailbox.
//
// Derived from muLLM v0.4.1 (NVIDIA Corporation, Apache-2.0): its batched DeepSeek-V4.1
// attention-projection kernel with the publishing cluster reduction
// (BatchedFxProjPublish, ClusterReducePublishB, PublishPartialsB, GatherPartialB, its
// one-group publishing entry and its launch bounds), its warp weight loader and warp MMA
// (WarpWeights, LoadWarpWeights, LoadFxWarpWeights, WarpMmaGlobal), its shared projection
// layout (ProjShared, PartialIndex), and its MX FP8, window-KV format and evict-first
// device primitives (below). This file has been modified from
// those: one self-contained NVRTC source for one fragment group (at most eight rows); the
// chain-stamp macros, the staged multi-group, payload-hop and pre-wait paths are dropped;
// static instead of dynamic shared memory; the cluster's rank and peer reads by PTX
// (mapa, ld.shared::cluster) instead of cooperative_groups. Interfaces adapted to the
// plugin: (1) the activation operands are read from the fused wo_a's row-major E4M3
// [rows][K] and FlashInfer 128x4-swizzled UE8M0 scale bytes instead of muLLM's activation
// fragments (WarpMmaGlobal's loads; the operand registers are the same); (2) the publish
// writes the plugin's mailbox (all_reduce_mhc.AllReduceMHC.publish_target): generation =
// stage_state[0] read after the grid wait, one multicast store of the BF16 pack with
// negative zero cleared into slot [generation][rank][row][column], and stage_state[1] =
// generation once per grid, in place of muLLM's per-block epochs, its sentinel and its
// per-peer stores. The weight is SGLang's MXFP8 wo_b in muLLM's fragment-major order
// (../wo_b_split.py). The arithmetic and the order of every sum are the original's.
//
// Entry: dsv41_wo_b_split (grid (S, N/64), cluster (S, 1, 1), 256 threads).

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

}  // namespace d41

// ---------------------------------------------------------------- window-KV formats

namespace d41attn {

// ClusterSyncShared publishes this CTA's prior shared-memory writes to its peers with
// fences restricted to shared memory. ClusterSyncRelaxed only keeps peers' shared memory
// alive until every CTA has finished reading it (no memory ordering).
__device__ __forceinline__ void ClusterSyncShared() {
  asm volatile(
      "fence.release.sync_restrict::shared::cta.cluster;\n\t"
      "barrier.cluster.arrive.relaxed.aligned;\n\t"
      "barrier.cluster.wait.aligned;\n\t"
      "fence.acquire.sync_restrict::shared::cluster.cluster;" ::: "memory");
}

__device__ __forceinline__ void ClusterSyncRelaxed() {
  asm volatile("barrier.cluster.arrive.relaxed.aligned;\n\tbarrier.cluster.wait.aligned;"
               ::: "memory");
}

}  // namespace d41attn

// ---------------------------------------------------------------- shared projection layout

namespace wo_b_proj {

constexpr int kWarps = 8;
constexpr int kCtaRows = 64;
constexpr int kGroupRows = 8;
// A warp's 32 x 4 partial values (lane l, value i at l * 5 + i) and one float of
// padding: the stores of a warp and the transpose reads of PublishPartialsB (rows of
// two tiles, lanes four apart) then fall in distinct shared-memory banks.
constexpr int kPartialWarpStride = 32 * 5 + 1;

// Offset of (warp, lane, value) in one group's warp_partial.
__device__ constexpr int PartialIndex(int warp, int lane, int index) {
  return warp * kPartialWarpStride + lane * 5 + index;
}

// ProjShared<1, false, false> (one group, no staged fragments): the warps' partial tiles
// and the CTA's partial tile the cluster reads.
struct alignas(16) ProjShared {
  float warp_partial[1][kWarps * kPartialWarpStride];  // PartialIndex
  float cta_partial[kGroupRows][kCtaRows];             // [query][row]
};

}  // namespace wo_b_proj

// ---------------------------------------------------------------- attn_proj/kernel.cu

namespace {

using wo_b_proj::kGroupRows;

constexpr int kWarps = 8;
constexpr int kThreads = kWarps * 32;
constexpr int kTiles = 4;
constexpr int kCtaRows = kTiles * d41::kTileRows;  // 64
constexpr int kMaxWarpSpans = 5;
// The fragment-input projections hold at most four spans per warp and are compiled for
// three resident CTAs per SM (80 registers), so the (S, N/64) = (4, 80) wo_b grid is one
// wave on 148 SMs (444 slots) even while other kernels hold SMs.
constexpr int kFxWarpSpans = 4;
constexpr int kFxMinBlocksPerSm = 3;
constexpr int kMaxClusterRanks = 8;

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

// LoadWarpWeights for the fragment-input projections. Inactive spans get defined zero
// scales: left undefined, ptxas merges the predicated scale loads it spills with stale
// stack contents through local-memory loads issued before the grid wait.
__device__ __forceinline__ void LoadFxWarpWeights(WarpWeights<kFxWarpSpans>& w,
                                                  const uint8_t* __restrict__ weight,
                                                  const uint8_t* __restrict__ weight_scales,
                                                  int k, int group, int first_span,
                                                  int warp_spans) {
#pragma unroll
  for (int i = 0; i < kFxWarpSpans; ++i) w.block_scales[i] = 0U;
  LoadWarpWeights(w, weight, weight_scales, k, group, first_span, warp_spans);
}

// Adapted: the activation operand of WarpMmaGlobal for lane (g = lane / 4, t = lane % 4)
// and span s, read from the fused wo_a's row-major E4M3 rows. muLLM's fragment word
// 2c + q holds values 64s + 32c + 8t + 4q .. +3 of row g, which are the 8 bytes at
// 64s + 8t (c = 0) and at 64s + 32 + 8t (c = 1) of that row. Rows >= `rows` are zero
// (the fragments' contract; the wo_a never writes them). Plain coherent loads: the PDL
// predecessor wrote them.
__device__ __forceinline__ uint4 ActivationWords(const uint8_t* codes, int k, int rows,
                                                 int span, int g, int t) {
  if (g >= rows) return make_uint4(0U, 0U, 0U, 0U);
  const uint8_t* row = codes + static_cast<int64_t>(g) * k + span * d41::kSpanValues + 8 * t;
  const uint2 low = *reinterpret_cast<const uint2*>(row);
  const uint2 high = *reinterpret_cast<const uint2*>(row + d41::kBlockValues);
  return make_uint4(low.x, low.y, high.x, high.y);
}

// Adapted: muLLM's activation scale float4 [s][t] (entry 2c + r: row 2t + r, k-block
// 2s + c) from FlashInfer's 128x4-swizzled UE8M0 bytes, whose offset for block b and row
// r < 8 is (b / 4) * 512 + r * 16 + b % 4: blocks 2s and 2s + 1 of a row are adjacent.
// A byte e decodes to 2^(e - 127), muLLM's (exponent + 127) << 23. Rows >= `rows` are 0.
__device__ __forceinline__ float4 ActivationScales(const uint8_t* scale_bytes, int rows,
                                                   int span, int t) {
  const int block = 2 * span;
  const int offset = (block >> 2) * 512 + (block & 3);
  const int r0 = 2 * t;
  const uint32_t pair0 =
      r0 < rows ? *reinterpret_cast<const uint16_t*>(scale_bytes + offset + r0 * 16) : 0U;
  const uint32_t pair1 =
      r0 + 1 < rows ? *reinterpret_cast<const uint16_t*>(scale_bytes + offset + (r0 + 1) * 16)
                    : 0U;
  return make_float4(__uint_as_float((pair0 & 0xFFU) << 23), __uint_as_float((pair1 & 0xFFU) << 23),
                     __uint_as_float((pair0 >> 8) << 23), __uint_as_float((pair1 >> 8) << 23));
}

// The warp's partial tile over its spans with the activation operands read from global
// memory (all loads issued before the first MMA).
template <int kSpans>
__device__ __forceinline__ void WarpMmaGlobal(const uint8_t* codes, const uint8_t* scale_bytes,
                                              int k, int rows, const WarpWeights<kSpans>& w,
                                              int first_span, int warp_spans, float (&out)[4]) {
  const int lane = static_cast<int>(threadIdx.x) & 31;
  uint4 act[kSpans];
  float4 act_scale[kSpans];
#pragma unroll
  for (int i = 0; i < kSpans; ++i) {
    if (i < warp_spans) {
      act[i] = ActivationWords(codes, k, rows, first_span + i, lane >> 2, lane & 3);
      act_scale[i] = ActivationScales(scale_bytes, rows, first_span + i, lane & 3);
    }
  }
  d41::TileAccumulator acc;
#pragma unroll
  for (int i = 0; i < kSpans; ++i) {
    if (i < warp_spans) {
      d41::SpanFp8(acc, w.low[i], w.high[i], w.block_scales[i], act[i], act_scale[i]);
    }
  }
#pragma unroll
  for (int index = 0; index < 4; ++index) out[index] = acc.value[index];
}

// ---------------------------------------------------------------- attn_proj/batched.cu

__device__ __forceinline__ int ClusterRank() {
  uint32_t rank;
  asm volatile("mov.u32 %0, %%cluster_ctarank;" : "=r"(rank));
  return static_cast<int>(rank);
}

__device__ __forceinline__ int ClusterRanks() {
  uint32_t ranks;
  asm volatile("mov.u32 %0, %%cluster_nctarank;" : "=r"(ranks));
  return static_cast<int>(ranks);
}

// cta_partial[query][row] of cluster CTA `source` (cg's map_shared_rank read).
__device__ __forceinline__ float PeerPartial(const wo_b_proj::ProjShared& smem, int source,
                                             int query, int row) {
  const uint32_t local =
      static_cast<uint32_t>(__cvta_generic_to_shared(&smem.cta_partial[query][row]));
  uint32_t remote;
  asm volatile("mapa.shared::cluster.u32 %0, %1, %2;" : "=r"(remote) : "r"(local), "r"(source));
  float value;
  asm volatile("ld.shared::cluster.f32 %0, [%1];" : "=f"(value) : "r"(remote) : "memory");
  return value;
}

// PublishPartials for one group: the two warps of each tile are added in order and the
// CTA's partial tile [8 queries][64 rows] is published to the cluster.
__device__ __forceinline__ void PublishPartialsB(wo_b_proj::ProjShared& smem,
                                                 const float (&acc)[1][4]) {
  const int warp = static_cast<int>(threadIdx.x) >> 5;
  const int lane = static_cast<int>(threadIdx.x) & 31;
  __syncthreads();
#pragma unroll
  for (int index = 0; index < 4; ++index) {
    smem.warp_partial[0][wo_b_proj::PartialIndex(warp, lane, index)] = acc[0][index];
  }
  __syncthreads();
#pragma unroll
  for (int pass = 0; pass < 2; ++pass) {
    const int output = static_cast<int>(threadIdx.x) + pass * kThreads;  // q*64 + r
    const int query = (output / kCtaRows) % kGroupRows;
    const int row = output % kCtaRows;
    const int tile = row / 16;
    const int within = row % 16;
    const int source_lane = (within % 8) * 4 + query / 2;
    const int source_index = (within / 8) * 2 + (query & 1);
    smem.cta_partial[query][row] =
        smem.warp_partial[0][wo_b_proj::PartialIndex(tile, source_lane, source_index)] +
        smem.warp_partial[0][wo_b_proj::PartialIndex(tile + kTiles, source_lane, source_index)];
  }
  d41attn::ClusterSyncShared();
}

// GatherPartial over the partial tiles (rank order, all peer reads first).
__device__ __forceinline__ float GatherPartialB(const wo_b_proj::ProjShared& smem, int query,
                                                int row) {
  const int ranks = ClusterRanks();
  float parts[kMaxClusterRanks];
#pragma unroll
  for (int source = 0; source < kMaxClusterRanks; ++source) {
    if (source < ranks) parts[source] = PeerPartial(smem, source, query, row);
  }
  float sum = 0.0F;
#pragma unroll
  for (int source = 0; source < kMaxClusterRanks; ++source) {
    if (source < ranks) sum += parts[source];
  }
  return sum;
}

// Adapted: the plugin's mailbox (all_reduce_mhc.AllReduceMHC.publish_target) in place of
// muLLM's MailboxArgs: the multicast base of the [generations][tp][capacity][hidden] BF16
// mailbox, this rank, and the generation stage_state[0] read after the grid wait.
struct MailboxArgs {
  uint64_t multicast;
  int rank;
  int tp;
  int capacity;
  int hidden;
  uint32_t stage;
};

// Negative zero is the mailbox's empty value: a published BF16 -0 becomes +0
// (common/sync.py sanitize_negative_zero_u32, the plugin's publishers).
__device__ __forceinline__ uint32_t SanitizeNegativeZero(uint32_t word) {
  if ((word & 0xFFFFU) == 0x8000U) word &= 0xFFFF0000U;
  if ((word >> 16) == 0x8000U) word &= 0x0000FFFFU;
  return word;
}

__device__ __forceinline__ void StoreMulticast(uint64_t address, uint4 value) {
  asm volatile("multimem.st.relaxed.sys.global.v4.bf16x2 [%0], {%1, %2, %3, %4};" ::"l"(address),
               "r"(value.x), "r"(value.y), "r"(value.z), "r"(value.w)
               : "memory");
}

// ClusterReducePublish for 8 queries: CTA rank r reduces outputs r * (512 / S) ..
// (query-major); each group of eight lanes packs its eight consecutive BF16 outputs into
// one 16-byte pack, stored to y and published to the mailbox.
__device__ __forceinline__ void ClusterReducePublishB(wo_b_proj::ProjShared& smem,
                                                      const float (&acc)[1][4],
                                                      __nv_bfloat16* __restrict__ y,
                                                      int y_stride, int row_base, int rows,
                                                      const MailboxArgs& mailbox) {
  PublishPartialsB(smem, acc);
  const int ranks = ClusterRanks();
  const int rank = ClusterRank();
  const int share = kGroupRows * kCtaRows / ranks;
  const int lane = static_cast<int>(threadIdx.x) & 31;
  const int local = static_cast<int>(threadIdx.x);
  if (local < share) {  // whole warps (share % 32 == 0)
    const int output = rank * share + local;
    const int query = output / kCtaRows;
    const int row = output % kCtaRows;
    const float value = query < rows ? GatherPartialB(smem, query, row) : 0.0F;
    const uint32_t bits = __bfloat16_as_ushort(__float2bfloat16_rn(value));
    const uint32_t word = bits | (__shfl_down_sync(0xffffffffU, bits, 1) << 16);
    uint4 pack;
    pack.x = word;
    pack.y = __shfl_down_sync(0xffffffffU, word, 2);
    pack.z = __shfl_down_sync(0xffffffffU, word, 4);
    pack.w = __shfl_down_sync(0xffffffffU, word, 6);
    if ((lane & 7) == 0 && query < rows) {
      const int column = row_base + row;
      *reinterpret_cast<uint4*>(y + static_cast<int64_t>(query) * y_stride + column) = pack;
      const uint4 clean =
          make_uint4(SanitizeNegativeZero(pack.x), SanitizeNegativeZero(pack.y),
                     SanitizeNegativeZero(pack.z), SanitizeNegativeZero(pack.w));
      const uint64_t element =
          ((static_cast<uint64_t>(mailbox.stage) * mailbox.tp + mailbox.rank) * mailbox.capacity +
           query) * mailbox.hidden + column;
      StoreMulticast(mailbox.multicast + element * 2, clean);
    }
  }
  d41attn::ClusterSyncRelaxed();
}

__device__ __forceinline__ uint32_t LoadVolatile(const uint32_t* address) {
  uint32_t value;
  asm volatile("ld.volatile.global.u32 %0, [%1];" : "=r"(value) : "l"(address) : "memory");
  return value;
}

// muLLM's FP8 projection publishing y into the mailbox (one group, S >= 2). The generation is a
// plain (volatile) load after the grid wait, as muLLM's epochs: the previous consumer on
// this rank, which advanced it, has completed by then.
__device__ __forceinline__ void BatchedFxProjPublish(
    const uint8_t* codes, const uint8_t* scale_bytes, const uint8_t* __restrict__ weight,
    const uint8_t* __restrict__ weight_scales, __nv_bfloat16* __restrict__ y, int rows, int n,
    int k, uint64_t multicast, uint32_t* stage_state, int rank, int tp, int capacity) {
  __shared__ wo_b_proj::ProjShared smem;
  const int warp = static_cast<int>(threadIdx.x) >> 5;
  const int split = static_cast<int>(blockIdx.x);
  const int group = static_cast<int>(blockIdx.y);
  const int cta_spans = k / d41::kSpanValues / static_cast<int>(gridDim.x);
  const int warp_spans = cta_spans / 2;
  const int first_span = split * cta_spans + (warp / kTiles) * warp_spans;
  WarpWeights<kFxWarpSpans> w;
  LoadFxWarpWeights(w, weight, weight_scales, k, group, first_span, warp_spans);
  d41::LaunchDependents();
  d41::GridDependencyWait();
  const uint32_t stage = LoadVolatile(stage_state);  // NEXT_STAGE
  float acc[1][4];
  WarpMmaGlobal(codes, scale_bytes, k, rows, w, first_span, warp_spans, acc[0]);
  const MailboxArgs mailbox = {multicast, rank, tp, capacity, n, stage};
  ClusterReducePublishB(smem, acc, y, n, group * kCtaRows, rows, mailbox);
  // Hand the generation to the consumer, once per grid (ACTIVE_STAGE).
  if (blockIdx.x == 0 && blockIdx.y == 0 && threadIdx.x == 0) {
    asm volatile("st.global.u32 [%0], %1;" ::"l"(stage_state + 1), "r"(stage) : "memory");
  }
}

}  // namespace

// Activation and control pointers are not __restrict__ (muLLM: a const __restrict__
// pointer allows the non-coherent load path, which under PDL returned lines cached by an
// earlier replay). Only the weights keep it.
extern "C" __global__ __launch_bounds__(kThreads, kFxMinBlocksPerSm) void dsv41_wo_b_split(
    const uint8_t* codes, const uint8_t* scale_bytes, const uint8_t* __restrict__ weight,
    const uint8_t* __restrict__ weight_scales, __nv_bfloat16* __restrict__ y, int rows, int n,
    int k, uint64_t multicast, uint32_t* stage_state, int rank, int tp, int capacity) {
  BatchedFxProjPublish(codes, scale_bytes, weight, weight_scales, y, rows, n, k, multicast,
                       stage_state, rank, tp, capacity);
}
