/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
// SCORE + TOPK. A page's score for row r is the maximum over its causally visible tokens of
// qi[r] . QMUL4(ki): the MegaAttention indexer formula (sm100_m3_decode_indexer_score_pipeline.hpp)
// with FP32 accumulation of the 128-dim dot product in two chains. TOPK selects the exact FP32
// top-15 historical pages plus the local page (sm100_m3_decode_topk_runtime.hpp key order).
#pragma once

#include "../common/ring_math.cuh"
#include "../common/ring_pipe.cuh"
#include "op_topk.cuh"

namespace attn_ring::score {

constexpr uint32_t kPartial = 0;     // f32 [local page][8 rows][8 warps] (SCORE), then TopK scratch
constexpr uint32_t kTopkScratch = 0; // topk::Scratch (after the SCORE fold)
constexpr uint32_t kIds = kTopkScratch + sizeof(topk::Scratch); // int [16]

// Byte offset of (token, 16-byte quarter) inside a 128B-swizzled 64-byte-row page.
__device__ __forceinline__ int swizzled_page_offset(int token, int quarter) {
  return (token >> 1) * 128 + ((((token & 1) * 4 + quarter) ^ ((token >> 1) & 7)) * 16);
}

// S^T tile of 16 page tokens (rows) x 8 queries (cols): A = QMUL4-dequantized K rows,
// B = E4M3 queries [8][128] given as per-lane uint2 fragments q[kc].
struct PageFrag {
  uint4 a, b;
  uint2 sa, sb;
};
__device__ __forceinline__ void load_page_frag(uint32_t page, int token_begin, int lane,
                                               PageFrag &f) {
  f.a = ldsm_x4(page + swizzled_page_offset(token_begin + (lane & 7), lane >> 3));
  f.b = ldsm_x4(page + swizzled_page_offset(token_begin + 8 + (lane & 7), lane >> 3));
  const uint32_t scales = page + kPageDataBytes + (token_begin + (lane >> 2)) * 8;
  f.sa = lds_64(scales);
  f.sb = lds_64(scales + 64);
}
// Dequantized A fragments of one 16-token page tile (QMUL4), shared by several query tiles.
__device__ __forceinline__ void page_a_frags(const PageFrag &f, int lane, uint4 (&a)[4]) {
  const uint32_t ra[4] = {f.a.x, f.a.y, f.a.z, f.a.w};
  const uint32_t rb[4] = {f.b.x, f.b.y, f.b.z, f.b.w};
  const int tg = lane & 3;
#pragma unroll
  for (int kc = 0; kc < 4; ++kc) {
    const uint32_t scale_a = broadcast_scale_byte(f.sa, 2 * kc + (tg >> 1));
    const uint32_t scale_b = broadcast_scale_byte(f.sb, 2 * kc + (tg >> 1));
    a[kc].x = qmul4(uint16_t(ra[kc]), scale_a);
    a[kc].z = qmul4(uint16_t(ra[kc] >> 16), scale_a);
    a[kc].y = qmul4(uint16_t(rb[kc]), scale_b);
    a[kc].w = qmul4(uint16_t(rb[kc] >> 16), scale_b);
  }
}
// One 16-token page tile x 8 queries: QMUL4 dequant, then the e4m3 dot product as explicit f16
// HMMA pairs chained through the FP32 accumulator (qh = the query B fragments unpacked once).
// The FP32 dot product is accumulated in two independent HMMA chains (K dims 0..63 and 64..127)
// that are added at the end: half the dependent-HMMA latency per page (FP32 reassociation inside
// one dot product).
__device__ __forceinline__ void page_mma(const PageFrag &f, const HalfB (&qh)[4], int lane,
                                         float (&acc)[4]) {
  float part[2][4] = {{0.f, 0.f, 0.f, 0.f}, {0.f, 0.f, 0.f, 0.f}};
  const uint32_t ra[4] = {f.a.x, f.a.y, f.a.z, f.a.w};
  const uint32_t rb[4] = {f.b.x, f.b.y, f.b.z, f.b.w};
  const int tg = lane & 3;
#pragma unroll
  for (int kc = 0; kc < 4; ++kc) {
    const uint32_t scale_a = broadcast_scale_byte(f.sa, 2 * kc + (tg >> 1));
    const uint32_t scale_b = broadcast_scale_byte(f.sb, 2 * kc + (tg >> 1));
    uint4 a;
    a.x = qmul4(uint16_t(ra[kc]), scale_a);
    a.z = qmul4(uint16_t(ra[kc] >> 16), scale_a);
    a.y = qmul4(uint16_t(rb[kc]), scale_b);
    a.w = qmul4(uint16_t(rb[kc] >> 16), scale_b);
    qmma_h(part[kc >> 1], a, qh[kc]);
  }
#pragma unroll
  for (int e = 0; e < 4; ++e)
    acc[e] = __fadd_rn(part[0][e], part[1][e]);
}

// The QMUL4-dequantized A fragments of one 16-token page tile, unpacked to the f16 halves of both
// HMMAs of every K32 chunk (exactly the conversions page_mma performs). Built before the queries
// arrive; page_mma_pre then issues the same HMMA pairs in the same order.
struct PageA {
  uint32_t lo[4][4], hi[4][4];
};
__device__ __forceinline__ void page_a_half(const PageFrag &f, int lane, PageA &h) {
  uint4 a[4];
  page_a_frags(f, lane, a);
#pragma unroll
  for (int kc = 0; kc < 4; ++kc) {
    h.lo[kc][0] = f16_lo(a[kc].x);
    h.lo[kc][1] = f16_lo(a[kc].y);
    h.lo[kc][2] = f16_lo(a[kc].z);
    h.lo[kc][3] = f16_lo(a[kc].w);
    h.hi[kc][0] = f16_hi(a[kc].x);
    h.hi[kc][1] = f16_hi(a[kc].y);
    h.hi[kc][2] = f16_hi(a[kc].z);
    h.hi[kc][3] = f16_hi(a[kc].w);
  }
}
__device__ __forceinline__ void page_mma_pre(const PageA &h, const HalfB (&qh)[4],
                                             float (&acc)[4]) {
  float part[2][4] = {{0.f, 0.f, 0.f, 0.f}, {0.f, 0.f, 0.f, 0.f}};
#pragma unroll
  for (int kc = 0; kc < 4; ++kc) {
    hmma16816(part[kc >> 1], h.lo[kc][0], h.lo[kc][1], h.lo[kc][2], h.lo[kc][3], qh[kc].lo[0],
              qh[kc].lo[1]);
    hmma16816(part[kc >> 1], h.hi[kc][0], h.hi[kc][1], h.hi[kc][2], h.hi[kc][3], qh[kc].hi[0],
              qh[kc].hi[1]);
  }
#pragma unroll
  for (int e = 0; e < 4; ++e)
    acc[e] = __fadd_rn(part[0][e], part[1][e]);
}

// QI B fragments from the finisher's records: lane (row t = lane / 4, s = lane % 4) needs record
// (t, 4 kc + s) for kc = 0..3. Polls until every record carries this call's tag.
constexpr uint32_t kQiStage = 16384; // uint2 [32 lanes][4] (after the SCORE partials)
static_assert(kQiStage >= kMaxScorePagesPerCta * kTokens * kConsumerWarps * 4 &&
                  kQiStage + 1024 <= kScratchBytes,
              "QI staging fits the scratch after the SCORE partials");
// recs: records of 8 consecutive query rows (16 per row), e.g. QI or the Q heads 0..7 / 8..15 of
// one token.
template <bool kDeadline>
__device__ __forceinline__ void poll_query_frag(const Params &p, const Deadline<kDeadline> &dl,
                                                const uint8_t *recs, uint32_t tag, int lane,
                                                uint2 (&f)[4]) {
  uint32_t pending = 0xfu;
  while (pending) {
    uint4 r[4];
#pragma unroll
    for (int kc = 0; kc < 4; ++kc)
      if (pending & (1u << kc))
        r[kc] = ld_rec(recs + (size_t(lane >> 2) * (kHeadDim / 8) + 4 * kc + (lane & 3)) * 16);
#pragma unroll
    for (int kc = 0; kc < 4; ++kc)
      if ((pending & (1u << kc)) && r[kc].w == tag) {
        f[kc] = make_uint2(r[kc].x, r[kc].y);
        pending &= ~(1u << kc);
      }
    if (pending) {
      dl.check(p, 0x10u, pending, tag);
    }
  }
}

// Score the <= 3 pages of one ring item; warp w covers page tokens 16w .. 16w+15 and keeps
// its per-(page, row) maximum in shared memory: partial[local page][row][warp]. No barrier per
// item; score_fold reduces the whole range once after the last item.
constexpr uint32_t kPartialPage = kTokens * kConsumerWarps; // floats per local page
static_assert(kMaxScorePagesPerCta * kPartialPage * 4 <= kScratchBytes,
              "SCORE partials fit the scratch");
__device__ __forceinline__ void score_pages(uint8_t *scratch, uint32_t slot, int first_page,
                                            int first_local, int pages, int kv, const HalfB (&q)[4],
                                            int warp, int lane) {
  float *partial = reinterpret_cast<float *>(scratch + kPartial);
  const int mask_page = new_row_page(kv);
  const int token_begin = warp * 16;
  if (pages < kKiPagesPerItem) {
    // Partial item (the last of a range): one page at a time.
#pragma unroll 1
    for (int j = 0; j < pages; ++j) {
      PageFrag f;
      load_page_frag(slot + j * kKiPageStride, token_begin, lane, f);
      float acc[4];
      page_mma(f, q, lane, acc);
      const int lp = first_page + j;
      if (lp >= mask_page) {
        const int token = lp * kPage + token_begin + (lane >> 2);
        const int query = kv - kTokens + 2 * (lane & 3);
        if (token > query)
          acc[0] = -1.0e30f;
        if (token > query + 1)
          acc[1] = -1.0e30f;
        if (token + 8 > query)
          acc[2] = -1.0e30f;
        if (token + 8 > query + 1)
          acc[3] = -1.0e30f;
      }
      float m0 = fmaxf(acc[0], acc[2]), m1 = fmaxf(acc[1], acc[3]);
#pragma unroll
      for (int o = 4; o <= 16; o <<= 1) {
        m0 = fmaxf(m0, __shfl_xor_sync(0xffffffffu, m0, o));
        m1 = fmaxf(m1, __shfl_xor_sync(0xffffffffu, m1, o));
      }
      if (lane < 4) {
        float *dst = partial + (first_local + j) * kPartialPage + warp;
        dst[(2 * lane) * kConsumerWarps] = m0;
        dst[(2 * lane + 1) * kConsumerWarps] = m1;
      }
    }
    return;
  }
  // A full item: the three pages are independent chains (ldmatrix -> QMUL4 -> MMA -> shuffles)
  // and are interleaved.
  PageFrag f[kKiPagesPerItem];
  float acc[kKiPagesPerItem][4];
#pragma unroll
  for (int j = 0; j < kKiPagesPerItem; ++j)
    load_page_frag(slot + j * kKiPageStride, token_begin, lane, f[j]);
#pragma unroll
  for (int j = 0; j < kKiPagesPerItem; ++j)
    page_mma(f[j], q, lane, acc[j]);
  float m0[kKiPagesPerItem], m1[kKiPagesPerItem];
#pragma unroll
  for (int j = 0; j < kKiPagesPerItem; ++j) {
    const int lp = first_page + j;
    if (lp >= mask_page) {
      const int token = lp * kPage + token_begin + (lane >> 2);
      const int query = kv - kTokens + 2 * (lane & 3);
      if (token > query)
        acc[j][0] = -1.0e30f;
      if (token > query + 1)
        acc[j][1] = -1.0e30f;
      if (token + 8 > query)
        acc[j][2] = -1.0e30f;
      if (token + 8 > query + 1)
        acc[j][3] = -1.0e30f;
    }
    m0[j] = fmaxf(acc[j][0], acc[j][2]);
    m1[j] = fmaxf(acc[j][1], acc[j][3]);
  }
#pragma unroll
  for (int o = 4; o <= 16; o <<= 1) {
#pragma unroll
    for (int j = 0; j < kKiPagesPerItem; ++j) {
      m0[j] = fmaxf(m0[j], __shfl_xor_sync(0xffffffffu, m0[j], o));
      m1[j] = fmaxf(m1[j], __shfl_xor_sync(0xffffffffu, m1[j], o));
    }
  }
  if (lane < 4) {
#pragma unroll
    for (int j = 0; j < kKiPagesPerItem; ++j) {
      float *dst = partial + (first_local + j) * kPartialPage + warp;
      dst[(2 * lane) * kConsumerWarps] = m0[j];
      dst[(2 * lane + 1) * kConsumerWarps] = m1[j];
    }
  }
}
// The first item of a SCORE range with its A fragments pre-converted (pa[j] for pages j < pages):
// the same per-page MMA, causal mask, maximum and partial store as score_pages.
__device__ __forceinline__ void score_item_pre(uint8_t *scratch, const PageA (&pa)[kKiPagesPerItem],
                                               int first_page, int first_local, int pages, int kv,
                                               const HalfB (&q)[4], int warp, int lane) {
  float *partial = reinterpret_cast<float *>(scratch + kPartial);
  const int mask_page = new_row_page(kv);
  const int token_begin = warp * 16;
  float acc[kKiPagesPerItem][4];
#pragma unroll
  for (int j = 0; j < kKiPagesPerItem; ++j)
    if (j < pages)
      page_mma_pre(pa[j], q, acc[j]);
  float m0[kKiPagesPerItem], m1[kKiPagesPerItem];
#pragma unroll
  for (int j = 0; j < kKiPagesPerItem; ++j) {
    if (j >= pages) {
      m0[j] = m1[j] = 0.f;
      continue;
    }
    const int lp = first_page + j;
    if (lp >= mask_page) {
      const int token = lp * kPage + token_begin + (lane >> 2);
      const int query = kv - kTokens + 2 * (lane & 3);
      if (token > query)
        acc[j][0] = -1.0e30f;
      if (token > query + 1)
        acc[j][1] = -1.0e30f;
      if (token + 8 > query)
        acc[j][2] = -1.0e30f;
      if (token + 8 > query + 1)
        acc[j][3] = -1.0e30f;
    }
    m0[j] = fmaxf(acc[j][0], acc[j][2]);
    m1[j] = fmaxf(acc[j][1], acc[j][3]);
  }
#pragma unroll
  for (int o = 4; o <= 16; o <<= 1) {
#pragma unroll
    for (int j = 0; j < kKiPagesPerItem; ++j) {
      m0[j] = fmaxf(m0[j], __shfl_xor_sync(0xffffffffu, m0[j], o));
      m1[j] = fmaxf(m1[j], __shfl_xor_sync(0xffffffffu, m1[j], o));
    }
  }
  if (lane < 4) {
#pragma unroll
    for (int j = 0; j < kKiPagesPerItem; ++j) {
      if (j >= pages)
        continue;
      float *dst = partial + (first_local + j) * kPartialPage + warp;
      dst[(2 * lane) * kConsumerWarps] = m0[j];
      dst[(2 * lane + 1) * kConsumerWarps] = m1[j];
    }
  }
}

// Page score = max over the eight warps' token slices, for every page of [begin, begin + count).
__device__ __forceinline__ uint8_t *score_rec(const Params &p, int row, int page) {
  return p.arena + arena::kScoreRec + (size_t(row) * kMaxPages + page) * 8;
}
__device__ __forceinline__ void score_fold(const Params &p, uint8_t *scratch, int begin, int count,
                                           int kv, uint32_t tag, int tid) {
  const float *partial = reinterpret_cast<const float *>(scratch + kPartial);
  consumer_sync();
  for (int idx = tid; idx < count * kTokens; idx += kConsumerThreads) {
    const int j = idx / kTokens, row = idx % kTokens;
    const float4 lo =
        *reinterpret_cast<const float4 *>(partial + j * kPartialPage + row * kConsumerWarps);
    const float4 hi =
        *reinterpret_cast<const float4 *>(partial + j * kPartialPage + row * kConsumerWarps + 4);
    const float s = fmaxf(fmaxf(fmaxf(lo.x, lo.y), fmaxf(lo.z, lo.w)),
                          fmaxf(fmaxf(hi.x, hi.y), fmaxf(hi.z, hi.w)));
    // A tagged record (value and epoch in one single-copy-atomic store), polled directly by TOPK.
    st_rec64(score_rec(p, row, begin + j), __float_as_uint(s), tag);
  }
}

// ---- TOPK ---------------------------------------------------------------------------
// Record per (row, slot): {logical page, physical page, valid tokens, epoch tag}.
__device__ __forceinline__ uint8_t *topk_record(const Params &p, int row, int slot) {
  return p.arena + arena::kTopkRec + (size_t(row) * kTopK + slot) * 16;
}
__device__ __forceinline__ int valid_tokens(int kv, int row, int logical) {
  const int last = kv - kTokens + row; // position of this row
  const int local = last / kPage;
  return logical == local ? last + 1 - local * kPage : kPage;
}

// All 256 consumer threads call (the selection polls the row's score records until every one
// carries this call's tag); runtime_topk uses named barrier 2 for them.
__device__ __forceinline__ void topk_row(const Params &p, uint8_t *scratch, const int *kv_table,
                                         int row, int kv, uint32_t tag, int tid) {
  int *ids = reinterpret_cast<int *>(scratch + kIds);
  auto &rt = *reinterpret_cast<topk::Scratch *>(scratch + kTopkScratch);
  static_assert(kIds + 64 <= kScratchBytes, "TopK scratch fits");
  const int visible = visible_pages(kv, row);
  topk::select(score_rec(p, row, 0), tag, visible, rt, ids);
  // The page table (cp.async copies issued before the selection) is complete and CTA-visible.
  cp_async_wait_all();
  consumer_sync();
  if (tid < kTopK) {
    const int logical = ids[tid];
    p.kv_block_indexes[row * kTopK + tid] = logical;
    const int physical = logical >= 0 ? kv_table[logical] : -1;
    st_rec(topk_record(p, row, tid), uint32_t(logical), uint32_t(physical),
           uint32_t(logical >= 0 ? valid_tokens(kv, row, logical) : 0), tag);
  }
}

} // namespace attn_ring::score
