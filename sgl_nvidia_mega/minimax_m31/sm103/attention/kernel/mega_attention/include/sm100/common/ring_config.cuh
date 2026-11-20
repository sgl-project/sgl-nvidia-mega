/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
// Geometry, arena layout, kernel parameters and the static work schedule of the single-request
// MiniMax-M3.1 TP4 decode Attention ring kernel.
//
// The kernel is a persistent 148-CTA grid. Every CTA runs a fixed chain of OPs (NORM -> PROJ ->
// QKFIN -> SCORE -> TOPK -> ATTN -> COMBINE -> OPROJ -> FINAL, each gated by its role). One
// producer thread streams weight tiles and cache pages into a shared-memory page ring while eight
// consumer warps compute. Both sides enumerate the same item sequence through the pure functions
// below.
#pragma once

#include <cuda.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cstdint>

namespace attn_ring {

// ---- launch geometry --------------------------------------------------------
constexpr int kGrid = 148;
constexpr int kConsumerWarps = 8;
constexpr int kProducerWarps = 4;
constexpr int kConsumerThreads = kConsumerWarps * 32;
constexpr int kProducerThreads = kProducerWarps * 32;
constexpr int kBlock = kConsumerThreads + kProducerThreads;
constexpr int kConsumerRegs = 216;
constexpr int kProducerRegs = 72;

// ---- model geometry (per TP rank) -------------------------------------------
constexpr int kWorld = 4;
constexpr int kTokens = 8;
constexpr int kHidden = 6144;
constexpr int kHeads = 16;
constexpr int kHeadDim = 128;
constexpr int kHeadSlots = 20; // 16 Q, K, V, QI, KI
constexpr int kHeadK = 16;
constexpr int kHeadV = 17;
constexpr int kHeadQi = 18;
constexpr int kHeadKi = 19;
constexpr int kProjRows = kHeadSlots * kHeadDim; // 2560
constexpr int kAttnK = kHeads * kHeadDim;        // 2048, OProj K
constexpr int kRotaryDim = 64;
constexpr int kPage = 128;
constexpr int kTopK = 16;
constexpr int kMaxPages = 8192;
constexpr int kPageDataBytes = kPage * kHeadDim / 2;   // 8192 packed E2M1
constexpr int kPageScaleBytes = kPage * kHeadDim / 16; // 1024 E4M3

// ---- tiles ------------------------------------------------------------------
constexpr int kTileRows = 16;
constexpr int kTileK = 2048;
constexpr int kProjTiles = kProjRows / kTileRows;    // 160
constexpr int kKSegs = kHidden / kTileK;             // 3
constexpr int kTilesPerHead = kHeadDim / kTileRows;  // 8
constexpr int kWoTiles = kHidden / kTileRows;        // 384
constexpr int kAttnCtas = kTokens * kTopK;           // 128
constexpr int kWoTilesPerCta = kWoTiles / kAttnCtas; // 3
constexpr int kFinalCtaBase = 128;                   // 8 FINAL CTAs, one per token
constexpr int kTopkCtaBase = 140;                    // 8 TOPK CTAs, one per row
constexpr int kFinishCtaBase = 128;                  // 12 CTAs finish the heads (no SCORE pages)
constexpr int kFinishCtas = 12;
constexpr int kNormCtaBase = 84; // 64 NORM CTAs: token = c % 8, slice = c / 8
constexpr int kNormSlices = 8;
constexpr int kNormCtas = kTokens * kNormSlices;
constexpr int kNormSliceK = kHidden / kNormSlices;           // 768
constexpr int kGroupsPerWarp = kTileK / 32 / kConsumerWarps; // 8 K32 groups per warp
static_assert(kWoTiles % kAttnCtas == 0, "Wo tiles divide over the attention CTAs");
static_assert(kTopkCtaBase + kTokens <= kGrid && kFinalCtaBase + kTokens <= kTopkCtaBase,
              "role CTAs are disjoint");
static_assert(kNormCtaBase + kNormCtas == kGrid, "NORM CTAs are the last 64");

// ---- shared-memory page ring ---------------------------------------------------
constexpr int kRingSlots = 6;
constexpr uint32_t kSlotBytes = 32768;
constexpr uint32_t kRingBytes = kRingSlots * kSlotBytes;
constexpr int kKiPagesPerItem = 3;
constexpr uint32_t kKiPageStride = kPageDataBytes + kPageScaleBytes; // 9216, 1024 aligned
constexpr uint32_t kKvKData = 0;
constexpr uint32_t kKvKScale = kPageDataBytes;
constexpr uint32_t kKvVData = kPageDataBytes + kPageScaleBytes;
constexpr uint32_t kKvVScale = 2 * kPageDataBytes + kPageScaleBytes;
constexpr uint32_t kKvBytes = 2 * (kPageDataBytes + kPageScaleBytes);
constexpr uint32_t kScratchBytes = 33792;
constexpr uint32_t kDynamicSmem = kRingBytes + kScratchBytes;
static_assert(kKiPagesPerItem * kKiPageStride <= kSlotBytes, "KI item fits one slot");
static_assert(kKvBytes <= kSlotBytes, "K/V page pair fits one slot");
static_assert(kTileRows * kTileK == kSlotBytes, "one weight tile fills one slot");
constexpr uint32_t kDynamicSmemAlloc = kDynamicSmem + 1024; // slack to 1024-align the ring
static_assert(kDynamicSmemAlloc + 512 <= 232448, "dynamic plus static shared memory fits");

// ---- arena (rank-private, zero-initialized once) ------------------------------
// Cross-CTA hand-offs are monotonic counters (target = arrivals * epoch) or records that carry the
// call's epoch as a tag next to their payload (single-copy-atomic .b64 / .b128 accesses), so no
// reset between calls is needed.
namespace arena {
constexpr size_t kLine = 128;
constexpr size_t kCounters = 0;  // u64 counters, one per 128-byte line
constexpr int kCntKiReady = 0;   // KI cache rows of this call written (KI finisher)
constexpr int kCntKvReady = 1;   // K and V cache rows written (K, V finishers)
constexpr int kCntOprojDone = 2; // attention CTAs that published their Wo tiles
constexpr int kCounterLines = 8;
constexpr size_t kEpochs = kCounters + kCounterLines * kLine; // u64 call count per CTA
// TopK records: 16 B {logical page, physical page, valid tokens, epoch tag} per (row, slot).
constexpr size_t kTopkRec = (kEpochs + kGrid * 8 + 127) / 128 * 128;
constexpr size_t kTopkRecBytes = size_t(kTokens) * kTopK * 16;
// QI records: 16 B {dims 8m .. 8m + 7 of one token (E4M3), 0, epoch tag} per (token, m), written
// by the QI finisher and polled by SCORE.
constexpr size_t kQiRec = kTopkRec + kTopkRecBytes;
constexpr size_t kQiRecBytes = size_t(kTokens) * (kHeadDim / 8) * 16;
// Q records: the same 16 B format per (token, head, m), written by the Q finishers, polled by ATTN.
constexpr size_t kQRec = kQiRec + kQiRecBytes;
constexpr size_t kQRecBytes = size_t(kTokens) * kHeads * (kHeadDim / 8) * 16;
// ATTN split partials, polled by the COMBINE CTA of each head: O as 8 B {bf16x2 of dims 2j, 2j + 1,
// epoch} per (row, split, head, j); statistics as 16 B {m, l, 0, epoch} per (row, split, head).
constexpr size_t kSplitORec = kQRec + kQRecBytes;
constexpr size_t kSplitORecBytes = size_t(kTokens) * kTopK * kHeads * (kHeadDim / 2) * 8;
constexpr size_t kSplitStatRec = kSplitORec + kSplitORecBytes;
constexpr size_t kSplitStatRecBytes = size_t(kTokens) * kTopK * kHeads * 16;
// O image (fragment-order MXFP8 payload + scales of the OProj activation): per (row, head) block of
// 128 payload + 4 scale bytes = 33 words, 11 records of {3 words, epoch}, polled by OPROJ.
constexpr int kORecsPerBlock = 11;
constexpr size_t kORec = kSplitStatRec + kSplitStatRecBytes;
constexpr size_t kORecBytes = size_t(kTokens) * kHeads * kORecsPerBlock * 16;
// MXFP8 activation: per (token, K128 block) 128 payload (fragment order) + 4 scale bytes = 33 words
// = 11 records {3 words, epoch}, written by the NORM CTAs, polled by PROJ.
constexpr int kXqBlocks = kHidden / 128; // 48 blocks per token
constexpr size_t kXqRec = kORec + kORecBytes;
constexpr size_t kXqRecBytes = size_t(kTokens) * kXqBlocks * kORecsPerBlock * 16;
// Page scores: 8 B {fp32, epoch} per (row, history page), written by SCORE, polled by TOPK.
constexpr size_t kScoreRec = kXqRec + kXqRecBytes;
constexpr size_t kScoreRecBytes = size_t(kTokens) * kMaxPages * 8;
// PROJ partials: 8 B {fp32, epoch} per (tile, K segment, token, row), polled by the head finishers.
constexpr size_t kPartRec = kScoreRec + kScoreRecBytes;
constexpr size_t kPartRecBytes = size_t(kProjTiles) * kKSegs * kTokens * kTileRows * 8;
constexpr size_t kBytes = (kPartRec + kPartRecBytes + 4095) / 4096 * 4096;
static_assert(kQiRec % 128 == 0 && kXqRec % 128 == 0 && kScoreRec % 128 == 0 && kPartRec % 128 == 0,
              "record regions stay 128-byte aligned");
} // namespace arena

// ---- TP4 mailbox (shared with the mega backend; see tp4_mailbox_layout.hpp) ----
namespace mailbox {
constexpr int kGenerations = 3;
constexpr unsigned short kEmpty = 0x8000;
constexpr int kOutputTileChannels = 128;
constexpr int kOutputTiles = kHidden / kOutputTileChannels; // 48 publish words
constexpr int kOwnersPerToken = 8;                          // consume words per token
} // namespace mailbox

// ---- kernel parameters ---------------------------------------------------------
struct alignas(64) Params {
  CUtensorMap tm_proj; // projection weight [2560][6144] B, box 128 B x 16 rows, 128B swizzle
  CUtensorMap tm_wo;   // OProj weight [6144][2048] B, same box
  CUtensorMap tm_ki;   // KI cache data [page][64 rows][128 B], 128B swizzle
  CUtensorMap tm_k;    // K cache data, same
  // inputs
  const __nv_bfloat16 *layer_input;  // [8][6144]
  const __nv_bfloat16 *input_norm;   // [6144]
  const __nv_bfloat16 *head_norm[4]; // q, k, qi, ki gammas [128]
  const __nv_bfloat16 *post_norm;    // [6144]
  const __nv_bfloat16 *cos_sin;      // [rows][64]
  const int64_t *positions;          // [8]
  const int64_t *slot_mapping;       // [8]
  const int *kv_len;                 // device int32 [1]
  const uint8_t *proj_sf;            // UE8M0 128x4 layout
  const uint8_t *wo_sf;
  // paged caches (byte pointers and page strides)
  uint8_t *k_data;
  uint8_t *k_scale;
  uint8_t *v_data;
  uint8_t *v_scale;
  uint8_t *ki_data;
  uint8_t *ki_scale;
  int64_t k_data_stride, k_scale_stride, v_data_stride, v_scale_stride, ki_data_stride,
      ki_scale_stride;
  const int *kv_indices;    // logical -> physical page, attention
  const int *indexer_table; // logical -> physical page, KI
  // outputs
  uint8_t *q_out;                     // e4m3 [8][16][128]
  uint8_t *qi_out;                    // e4m3 [8][128]
  int *kv_block_indexes;              // int32 [8][1][16]
  uint8_t *o_out;                     // e4m3 [8][16][128]
  uint8_t *o_sf_out;                  // u8, o_sf[head * 512 + token * 16 + group]
  __nv_bfloat16 *attention_residual;  // [8][6144]
  __nv_bfloat16 *post_attention_norm; // [8][6144]
  // workspaces
  uint8_t *arena;
  uint8_t *tp_unicast;
  uint8_t *tp_multicast;
  uint64_t tp_payload_offset; // mailbox payload base inside the TP workspace
  uint64_t tp_publish_offset; // u32[48] publish generations
  uint64_t tp_consume_offset; // u32[64] consume generations
  int tp_rank;
  // scalars
  float input_eps, head_eps, post_eps;
  float k_global, v_global, ki_global; // NVFP4 dequant global scales
  float softmax_log2_scale;            // scale_q * scale_k * sm_scale * log2(e)
  float v_out_scale;                   // scale_v applied to partial O
  uint64_t wait_timeout_ns; // deadline of every cross-CTA wait (kDeadline instantiation only)
};

// ---- static schedule -------------------------------------------------------------
// PROJ: 480 items = 160 row tiles x 3 K segments over the 146 CTAs that are not indexer-head
// finishers. Projection CTA ordinal c' owns segment c' % 3 and walks a priority-ordered tile
// list (QI, KI, K, V, then Q) with stride = CTAs per segment, so the QI/KI items are the
// first item of 48 CTAs.
constexpr int kQiFinisher = 136; // finishes the QI head (no PROJ items, no SCORE pages)
constexpr int kKiFinisher = 137; // finishes the KI head
constexpr int kKFinisher = 138;  // finishes the K head
constexpr int kVFinisher = 139;  // finishes the V head
constexpr int kProjCtas = kGrid - 2;
__host__ __device__ constexpr int proj_ordinal(int cta) {
  return cta < kQiFinisher ? cta : (cta > kKiFinisher ? cta - 2 : -1);
}
__host__ __device__ constexpr int proj_seg(int cta) { return proj_ordinal(cta) % kKSegs; }
__host__ __device__ constexpr int proj_seg_ctas(int seg) {
  return (kProjCtas - seg + kKSegs - 1) / kKSegs;
}
__host__ __device__ constexpr int proj_priority_tile(int order) {
  // [144..159] QI then KI, [128..143] K then V, then Q tiles 0..127
  return order < 16 ? 144 + order : (order < 32 ? 128 + (order - 16) : order - 32);
}
__host__ __device__ constexpr int proj_items(int cta) {
  const int c = proj_ordinal(cta);
  if (c < 0)
    return 0;
  const int j = c / kKSegs, n = proj_seg_ctas(c % kKSegs);
  return j < kProjTiles ? (kProjTiles - 1 - j) / n + 1 : 0;
}
__host__ __device__ constexpr int proj_tile(int cta, int i) {
  const int c = proj_ordinal(cta);
  return proj_priority_tile(c / kKSegs + i * proj_seg_ctas(c % kKSegs));
}

// SCORE: pages 0..P-2 are the only pages any row ranks. Every CTA except the 12 head
// finishers owns a contiguous range.
__host__ __device__ inline int score_pages(int kv) {
  const int pages = (kv + kPage - 1) / kPage;
  return pages > kTopK ? pages - 1 : 0;
}
// Largest SCORE range of one CTA (kMaxPages - 1 history pages over kScoreCtas CTAs), rounded up.
constexpr int kMaxScorePagesPerCta = 64;
__host__ __device__ constexpr bool is_finish_cta(int cta) {
  return cta >= kFinishCtaBase && cta < kFinishCtaBase + kFinishCtas;
}
// SCORE CTAs: 0..127 and 140..147 (the 12 Q/K/V finishers include the QI/KI finishers).
constexpr int kScoreCtas = kGrid - kFinishCtas;
static_assert((kMaxPages - 1 + kScoreCtas - 1) / kScoreCtas <= kMaxScorePagesPerCta,
              "the SCORE page table of one CTA fits its shared-memory cache");
__host__ __device__ inline void score_range(int kv, int cta, int &begin, int &end) {
  const int s = score_pages(kv);
  const int ordinal = cta < kFinishCtaBase ? cta : cta - kFinishCtas;
  begin = is_finish_cta(cta) ? 0 : int((int64_t(s) * ordinal) / kScoreCtas);
  end = is_finish_cta(cta) ? 0 : int((int64_t(s) * (ordinal + 1)) / kScoreCtas);
}
// Q heads h and h + 8 are finished together by CTA 128 + h (CTAs 128..135); QI, KI, K and V by
// CTAs 136..139. QI / KI finishers own no projection tiles and start polling at once.
constexpr int kQkvFinishers = 8;
__host__ __device__ inline int score_items(int kv, int cta) {
  int b, e;
  score_range(kv, cta, b, e);
  return (e - b + kKiPagesPerItem - 1) / kKiPagesPerItem;
}

// Wo / KV: attention CTAs 0..127 (row = cta / 16, slot = cta % 16) own three
// contiguous 16-channel Wo tiles and one K/V page item, all terminal ring items.
__host__ __device__ constexpr bool is_attn_cta(int cta) { return cta < kAttnCtas; }
__host__ __device__ constexpr int wo_items(int cta) {
  return is_attn_cta(cta) ? kWoTilesPerCta : 0;
}

// First page that may hold rows written by this call.
__host__ __device__ inline int new_row_page(int kv) { return (kv - kTokens) / kPage; }
// Pages visible to physical row r (positions kv-8 .. kv-1).
__host__ __device__ inline int visible_pages(int kv, int row) {
  return (kv - kTokens + row) / kPage + 1;
}

struct ItemCounts {
  int proj, score, wo, kv;
};
__host__ __device__ inline ItemCounts item_counts(int kv, int cta) {
  return ItemCounts{proj_items(cta), score_items(kv, cta), wo_items(cta), is_attn_cta(cta) ? 1 : 0};
}

} // namespace attn_ring
