/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
// The persistent ring kernel. Producer thread 256 fills the per-CTA item sequence
// [PROJ tiles][KI page groups][Wo tiles (terminal)][K/V page (terminal)]; the consumer warps run
// the OP chain of their CTA role. The kDeadline instantiation bounds every cross-CTA wait by
// Params::wait_timeout_ns (AttentionConfig.wait_timeout_s) and traps with a report.
#pragma once

#include "../collective/op_attn.cuh"
#include "../collective/op_oproj.cuh"

namespace attn_ring {

struct SharedControl {
  uint64_t bars[2 * kRingSlots];
  uint64_t epoch;
  int kv;
  int stage[2];                // mailbox generations: publish (first/last Wo tile group) or consume
  volatile uint32_t proj_done; // consumer -> producer: PROJ tiles consumed, Wo may stream
  attn::KvInfo kv_info;
  uint32_t norm_tag;                  // NORM: this call's epoch tag for the activation records
  int ki_pages[kMaxScorePagesPerCta]; // physical KI pages of this CTA's SCORE range
};

// Producer prologue: reserve every PROJ slot and issue item 0 (the QI/KI tiles of the indexer
// critical path come first in the priority order) before anything else touches memory.
template <bool kDeadline>
__device__ __forceinline__ void produce_first(const Params &p, const Deadline<kDeadline> &dl,
                                              const Ring &ring, int n_proj, int cta) {
  const uint64_t policy = l2_evict_first();
  for (int i = 0; i < n_proj; ++i) {
    ring_reserve<kDeadline>(p, dl, ring, uint32_t(i));
    ring_expect(ring, uint32_t(i), kSlotBytes);
  }
  if (n_proj)
    tma_load_3d(ring.slot(0), &p.tm_proj, 0, proj_tile(cta, 0) * kTileRows,
                proj_seg(cta) * (kTileK / 128), ring.full_bar(0), policy);
}

template <bool kDeadline>
__device__ __forceinline__ void produce(const Params &p, const Deadline<kDeadline> &dl,
                                        const Ring &ring, SharedControl &ctl, const ItemCounts &n,
                                        int kv, uint64_t epoch, int cta) {
  const uint64_t policy = l2_evict_first();
  const int seg = proj_seg(cta);
  // K/V page of an attention CTA: a static page (no TopK) resolves its physical page now.
  int kv_logical = -1, kv_physical = -1;
  bool kv_static = false;
  if (n.kv) {
    kv_static = attn::static_page(kv, cta / kTopK, cta % kTopK, kv_logical);
    if (kv_static && kv_logical >= 0)
      kv_physical = __ldg(p.kv_indices + kv_logical);
  }
  uint32_t item = uint32_t(n.proj);
  int begin = 0, end = 0;
  score_range(kv, cta, begin, end);
  const int gate = new_row_page(kv);
  bool ki_ready = false;
  auto issue_ki = [&](int it) {
    const int first = begin + it * kKiPagesPerItem;
    const int pages = min(kKiPagesPerItem, end - first);
    const uint32_t slot = ring_reserve<kDeadline>(p, dl, ring, item);
    // Rows written by this call: wait for the KI head (all earlier items are issued).
    if (!ki_ready && first + pages - 1 >= gate) {
      Counter::at(p.arena, arena::kCntKiReady).spin<kDeadline>(p, dl, 0x300u, epoch);
      fence_proxy_async_global();
      ki_ready = true;
    }
    ring_expect(ring, item, pages * kKiPageStride);
    for (int j = 0; j < pages; ++j) {
      const int physical = ctl.ki_pages[first - begin + j];
      const uint32_t dst = slot + j * kKiPageStride;
      tma_load_3d(dst, &p.tm_ki, 0, 0, physical, ring.full_bar(item), policy);
      bulk_g2s(dst + kPageDataBytes, p.ki_scale + int64_t(physical) * p.ki_scale_stride,
               kPageScaleBytes, ring.full_bar(item), policy);
    }
    ++item;
  };
  // The memory system serves outstanding copies fairly, so a lower issue order alone does not
  // prioritize: with indexer work, keep only item 0 in flight until it lands (all CTAs together:
  // 4.7 MB) so the QI / KI tiles of the critical path come first. Without it, everything at once.
  if (n.proj && score_pages(kv) > 0)
    mbar_wait<kDeadline>(p, dl, 0x700u, ring.full_bar(0), 0u);
  // The remaining projection tiles (SCORE cannot start before this CTA's PROJ ends), then the KI
  // pages that need no wait into the free slots.
  for (int i = 1; i < n.proj; ++i)
    tma_load_3d(ring.slot(uint32_t(i)), &p.tm_proj, 0, proj_tile(cta, i) * kTileRows,
                seg * (kTileK / 128), ring.full_bar(uint32_t(i)), policy);
  int ki_issued = 0;
  while (ki_issued < n.score && int(item) < kRingSlots) {
    const int first = begin + ki_issued * kKiPagesPerItem;
    if (min(first + kKiPagesPerItem, end) - 1 >= gate)
      break; // gated: after the PROJ tiles
    issue_ki(ki_issued++);
  }
  while (ki_issued < n.score)
    issue_ki(ki_issued++);
  if (n.wo) {
    // Wo is needed last. Without indexer work (rows of <= 16 pages: no SCORE / TOPK hand-offs to
    // disturb) it streams as soon as this CTA's projection tiles have landed (HBM idles during the
    // PROJ GEMVs; no KI item exists, so the last PROJ full barrier is still in phase 0). Otherwise
    // it waits until the projection tiles are consumed, keeping 12.6 MB off HBM while the
    // latency-bound QI / SCORE hand-offs start.
    if (n.proj && score_pages(kv) == 0)
      mbar_wait<kDeadline>(p, dl, 0x701u, ring.full_bar(uint32_t(n.proj - 1)), 0u);
    else
      while (ctl.proj_done == 0u)
        __nanosleep(64);
    // The UE8M0 scales of this CTA's Wo rows into L2 now; OPROJ loads them before its O-record
    // poll.
    const int block_bytes = (kAttnK / 128) * 512;
    const int b0 = (cta * kWoTilesPerCta * kTileRows) >> 7,
              b1 = ((cta + 1) * kWoTilesPerCta * kTileRows - 1) >> 7;
    prefetch_l2(p.wo_sf + size_t(b0) * block_bytes, block_bytes);
    if (b1 != b0)
      prefetch_l2(p.wo_sf + size_t(b1) * block_bytes, block_bytes);
  }
  for (int t = 0; t < n.wo; ++t, ++item) {
    const int tile = cta * kWoTilesPerCta + t;
    const uint32_t slot = ring_reserve<kDeadline>(p, dl, ring, item);
    ring_expect(ring, item, kSlotBytes);
    tma_load_3d(slot, &p.tm_wo, 0, tile * kTileRows, 0, ring.full_bar(item), policy);
  }
  if (n.kv) {
    const int row = cta / kTopK, split = cta % kTopK;
    int logical = kv_logical, physical = kv_physical;
    if (!kv_static) {
      // The TopK record carries the physical page: no dependent page-table load here.
      const uint8_t *rec = score::topk_record(p, row, split);
      uint4 r = ld_rec(rec);
      while (r.w != uint32_t(epoch)) {
        dl.check(p, 0x400u + uint32_t(cta), r.w, epoch);
        r = ld_rec(rec);
      }
      logical = int(r.x);
      physical = int(r.y);
    }
    const uint32_t slot = ring_reserve<kDeadline>(p, dl, ring, item);
    ctl.kv_info.logical = logical;
    ctl.kv_info.valid = logical >= 0 ? score::valid_tokens(kv, row, logical) : 0;
    if (logical < 0) {
      ring_fill_empty(ring, item);
    } else {
      if (logical >= new_row_page(kv)) {
        Counter::at(p.arena, arena::kCntKvReady).spin<kDeadline>(p, dl, 0x500u, 2ull * epoch);
        fence_proxy_async_global();
      }
      ring_expect(ring, item, kKvBytes);
      tma_load_3d(slot + kKvKData, &p.tm_k, 0, 0, physical, ring.full_bar(item), policy);
      bulk_g2s(slot + kKvKScale, p.k_scale + int64_t(physical) * p.k_scale_stride, kPageScaleBytes,
               ring.full_bar(item), policy);
      bulk_g2s(slot + kKvVData, p.v_data + int64_t(physical) * p.v_data_stride, kPageDataBytes,
               ring.full_bar(item), policy);
      bulk_g2s(slot + kKvVScale, p.v_scale + int64_t(physical) * p.v_scale_stride, kPageScaleBytes,
               ring.full_bar(item), policy);
    }
  }
}

template <bool kDeadline>
__global__ __launch_bounds__(kBlock, 1) void attn_ring_kernel(const __grid_constant__ Params p) {
  extern __shared__ uint8_t dyn_smem[];
  __shared__ __align__(16) SharedControl ctl;
  // 1024-aligned ring base derived by pointer arithmetic on the shared array, so the compiler keeps
  // the shared state space for every scratch access (LDS/STS rather than generic LD/ST).
  uint8_t *ring_mem = dyn_smem + ((1024u - (smem_u32(dyn_smem) & 1023u)) & 1023u);
  uint8_t *scratch = ring_mem + kRingBytes;
  const int cta = blockIdx.x, tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;

  // The producer thread owns the ring barriers and issues PROJ item 0 right away: neither depends
  // on the epoch or on kv, so the first weight tile is in flight before the CTA-wide barrier.
  if (tid == kConsumerThreads) {
    ring_init(ctl.bars, true);
    prefetch_tensormap(&p.tm_proj);
    prefetch_tensormap(&p.tm_wo);
    prefetch_tensormap(&p.tm_ki);
    prefetch_tensormap(&p.tm_k);
    Deadline<kDeadline> d0;
    d0.start(p);
    produce_first<kDeadline>(p, d0, make_ring(ring_mem, ctl.bars), proj_items(cta), cta);
  }
  uint64_t *epoch_word = nullptr;
  uint64_t epoch_next = 0ull;
  int kv_in = 0;
  if (tid == 0) {
    epoch_word = reinterpret_cast<uint64_t *>(p.arena + arena::kEpochs) + cta;
    epoch_next = *epoch_word + 1ull;
    kv_in = __ldg(p.kv_len);
  }
  // ---- NORM (64 CTAs; independent of the epoch and of kv, overlaps their loads) ----
  if (cta >= kNormCtaBase && tid < kConsumerThreads) {
    const int c = cta - kNormCtaBase;
    proj::norm_slice(p, scratch, c % kTokens, c / kTokens, tid, warp, lane, &ctl.norm_tag,
                     uint32_t(epoch_next));
  }
  if (tid == 0) {
    *epoch_word = epoch_next;
    ctl.epoch = epoch_next;
    ctl.kv = kv_in;
    ctl.proj_done = 0u;
  }
  __syncthreads();
  const uint64_t epoch = ctl.epoch;
  const int kv = ctl.kv;
  Deadline<kDeadline> dl;
  dl.start(p);
  const Ring ring = make_ring(ring_mem, ctl.bars);
  const ItemCounts n = item_counts(kv, cta);

  const uint32_t i_score = n.proj, i_wo = n.proj + n.score, i_kv = i_wo + n.wo;

  // The producer warpgroup (setmaxnreg works per warpgroup): warp 8 streams the ring, warp 11
  // prefetches FINAL / TOPK inputs into L2, warps 9 and 10 have no work.
  if (warp >= kConsumerWarps) {
    setmaxnreg_producer();
    if (warp == kConsumerWarps) {
      // The physical KI pages of this CTA's SCORE range, loaded at once by the whole warp so
      // the issuing thread never waits on a page-table load between two page copies.
      if (n.score) {
        int begin, end;
        score_range(kv, cta, begin, end);
        for (int j = lane; j < end - begin; j += 32)
          ctl.ki_pages[j] = __ldg(p.indexer_table + begin + j);
      }
      __syncwarp();
      if (lane == 0)
        produce<kDeadline>(p, dl, ring, ctl, n, kv, epoch, cta);
    } else if (warp == kConsumerWarps + 3) {
      // FINAL CTAs: the post-attention norm gamma into L2 early (read after the mailbox poll).
      if (cta >= kFinalCtaBase && cta < kFinalCtaBase + kTokens && lane == 0)
        prefetch_l2(p.post_norm, kHidden * 2);
      // TOPK CTAs: the row's page table into L2 (staged after this CTA's SCORE, before the
      // selection).
      if (cta >= kTopkCtaBase && cta < kTopkCtaBase + kTokens && lane == 0) {
        const uint32_t bytes = uint32_t(visible_pages(kv, cta - kTopkCtaBase)) * 4u & ~15u;
        if (bytes)
          prefetch_l2(p.kv_indices, bytes);
      }
    }
    return;
  }
  setmaxnreg_consumer();
  // Mailbox generations advance only between launches. Thread 0 issues the loads now and
  // publishes the values to the CTA right before OPROJ / FINAL, hiding their latency.
  int stage_a = 0, stage_b = 0;
  if (tid == 0) {
    if (is_attn_cta(cta)) {
      const int first = cta * kWoTilesPerCta * kTileRows;
      const int last = first + kWoTilesPerCta * kTileRows - 1;
      stage_a = int(oproj::publish_words(p)[first / mailbox::kOutputTileChannels]);
      stage_b = int(oproj::publish_words(p)[last / mailbox::kOutputTileChannels]);
    } else if (cta >= kFinalCtaBase && cta < kFinalCtaBase + kTokens) {
      stage_a = int(oproj::consume_words(p)[(cta - kFinalCtaBase) * mailbox::kOwnersPerToken]);
    }
  }

  // ---- PROJ + QKFIN ----
  if (n.proj) {
    const int seg = proj_seg(cta);
    uint32_t sfw[4][2][2];
#pragma unroll
    for (int i = 0; i < 4; ++i)
      if (i < n.proj)
        proj::load_weight_scales(p.proj_sf, kHidden, proj_tile(cta, i),
                                 seg * (kTileK / 128) + 2 * warp, lane, sfw[i]);
    // This CTA's K segment of the activation image (8 tokens x 2 KB + scales), after the fold
    // buffers.
    constexpr uint32_t kXqStage = 16384, kSegGroups = kTileK / 32; // 64 K32 groups per segment
    static_assert(kXqStage >= proj::kFold + 2 * kConsumerWarps * kTileRows * kTokens * 4 &&
                      kXqStage + kTokens * (kSegGroups * 32 + kSegGroups) <= kScratchBytes,
                  "activation staging fits the scratch after the fold buffers");
    // The activation segment straight from the NORM CTAs' tagged records: each warp polls only the
    // 2 blocks (8 K32 groups) x 8 tokens x 11 records of its own K slice into its own part of the
    // staging and starts its GEMV at once (the first cross-warp dependency is the split-K fold).
    {
      constexpr int kRecs = kTokens * 2 * arena::kORecsPerBlock; // 176 per warp
      constexpr int kPer = (kRecs + 31) / 32;                    // 6
      uint32_t pending = 0u;
#pragma unroll
      for (int k = 0; k < kPer; ++k)
        if (lane + k * 32 < kRecs)
          pending |= 1u << k;
      uint4 r[kPer];
      while (pending) {
#pragma unroll
        for (int k = 0; k < kPer; ++k)
          if (pending & (1u << k)) {
            const int i = lane + k * 32;
            const int blk = i / arena::kORecsPerBlock, j = i % arena::kORecsPerBlock;
            const int t = blk >> 1, sb = 2 * warp + (blk & 1);
            r[k] = ld_rec(
                p.arena + arena::kXqRec +
                ((size_t(t) * arena::kXqBlocks + seg * 16 + sb) * arena::kORecsPerBlock + j) * 16);
          }
#pragma unroll
        for (int k = 0; k < kPer; ++k) {
          if (!(pending & (1u << k)) || r[k].w != uint32_t(epoch))
            continue;
          pending &= ~(1u << k);
          const int i = lane + k * 32;
          const int blk = i / arena::kORecsPerBlock, j = i % arena::kORecsPerBlock;
          const int t = blk >> 1, sb = 2 * warp + (blk & 1);
          const uint32_t words[3] = {r[k].x, r[k].y, r[k].z};
#pragma unroll
          for (int q = 0; q < 3; ++q) {
            const int wd = 3 * j + q;
            uint32_t *dst =
                wd < 32
                    ? reinterpret_cast<uint32_t *>(scratch + kXqStage + t * (kSegGroups * 32) +
                                                   sb * 128 + 4 * wd)
                    : reinterpret_cast<uint32_t *>(scratch + kXqStage + kTokens * kSegGroups * 32 +
                                                   t * kSegGroups + 4 * sb);
            *dst = words[q];
          }
        }
        if (pending)
          dl.check(p, 0x08u, pending, epoch);
      }
      __syncwarp();
    }
    proj::BFragH frag;
    {
      proj::BFrag raw;
      proj::smem_bfrag(scratch + kXqStage, scratch + kXqStage + kTokens * kSegGroups * 32,
                       kSegGroups, warp * kGroupsPerWarp, lane, raw);
      proj::unpack_bfrag(raw, frag);
    }
    float *fold = reinterpret_cast<float *>(scratch + proj::kFold);
    // Item i's fold and its partials as tagged records (polled by the head finishers).
    auto finish_item = [&](int i, const float (&run)[4]) {
      const int tile = proj_tile(cta, i);
      // Two fold buffers: item i+1 may start writing before every reader of item i is done.
      const float v = proj::fold_tile(fold + (i & 1) * (kConsumerWarps * kTileRows * kTokens), run,
                                      tid, warp, lane);
      if (tid < kTileRows * kTokens) {
        const int idx = (tid % kTokens) * kTileRows + tid / kTokens;
        st_rec64(proj::part_rec(p, tile, seg) + size_t(idx) * 8, __float_as_uint(v),
                 uint32_t(epoch));
      }
    };
    auto scales_of = [&](int i, uint32_t (&w)[2][2]) {
#pragma unroll
      for (int k = 0; k < 4; ++k)
        if (k == i) {
          w[0][0] = sfw[k][0][0];
          w[0][1] = sfw[k][0][1];
          w[1][0] = sfw[k][1][0];
          w[1][1] = sfw[k][1][1];
        }
    };
    // Item 0 (the indexer tiles go first) alone, then the others in pairs: two interleaved HMMA
    // chains per warp hide their latency.
#pragma unroll 1
    for (int i = 0; i < n.proj;) {
      if (i > 0 && i + 1 < n.proj) {
        uint32_t slots[2];
        slots[0] = ring_acquire<kDeadline>(p, dl, ring, uint32_t(i));
        slots[1] = ring_acquire<kDeadline>(p, dl, ring, uint32_t(i + 1));
        uint32_t w[2][2][2];
        scales_of(i, w[0]);
        scales_of(i + 1, w[1]);
        float run[2][4];
        proj::tiles_gemv_h<2>(slots, w, frag, warp, lane, run);
        ring_release(ring, uint32_t(i), lane);
        ring_release(ring, uint32_t(i + 1), lane);
        finish_item(i, run[0]);
        finish_item(i + 1, run[1]);
        i += 2;
      } else {
        const uint32_t slot = ring_acquire<kDeadline>(p, dl, ring, uint32_t(i));
        uint32_t w[2][2];
        scales_of(i, w);
        float run[4];
        proj::tile_gemv_h(slot, w, frag, warp, lane, run);
        ring_release(ring, uint32_t(i), lane);
        finish_item(i, run);
        ++i;
      }
    }
  }
  if (tid == 0)
    ctl.proj_done = 1u;

  // ---- SCORE ----
  if (n.score) {
    int begin, end;
    score_range(kv, cta, begin, end);
    // The first item's KI pages are dequantized (QMUL4) and unpacked to f16 while QI is still in
    // flight; its slot is released at once, so the producer refills it early.
    const int pages0 = min(kKiPagesPerItem, end - begin);
    score::PageA pa[kKiPagesPerItem];
    {
      const uint32_t slot0 = ring_acquire<kDeadline>(p, dl, ring, i_score);
#pragma unroll
      for (int j = 0; j < kKiPagesPerItem; ++j)
        if (j < pages0) {
          score::PageFrag f;
          score::load_page_frag(slot0 + j * kKiPageStride, warp * 16, lane, f);
          score::page_a_half(f, lane, pa[j]);
        }
      ring_release(ring, i_score, lane);
    }
    // QI: warp 0 polls the finisher's tagged records (data and readiness in one round trip)
    // and hands the B fragments to the other warps through shared memory.
    uint2 q[4];
    {
      uint2 *qs = reinterpret_cast<uint2 *>(scratch + score::kQiStage);
      if (warp == 0) {
        score::poll_query_frag<kDeadline>(p, dl, p.arena + arena::kQiRec, uint32_t(epoch), lane, q);
#pragma unroll
        for (int kc = 0; kc < 4; ++kc)
          qs[lane * 4 + kc] = q[kc];
      }
      consumer_sync();
      if (warp != 0) {
#pragma unroll
        for (int kc = 0; kc < 4; ++kc)
          q[kc] = qs[lane * 4 + kc];
      }
    }
    HalfB qh[4];
#pragma unroll
    for (int kc = 0; kc < 4; ++kc)
      qh[kc] = half_b(q[kc].x, q[kc].y);
    score::score_item_pre(scratch, pa, begin, 0, pages0, kv, qh, warp, lane);
    for (int it = 1; it < n.score; ++it) {
      const int first = begin + it * kKiPagesPerItem, pages = min(kKiPagesPerItem, end - first);
      const uint32_t slot = ring_acquire<kDeadline>(p, dl, ring, i_score + it);
      score::score_pages(scratch, slot, first, it * kKiPagesPerItem, pages, kv, qh, warp, lane);
      ring_release(ring, i_score + it, lane);
    }
    score::score_fold(p, scratch, begin, end - begin, kv, uint32_t(epoch), tid);
    consumer_sync();
  }
  // ---- head finishers: Q heads (c, c + 8) on CTA 128 + c, QI / KI / K / V on CTAs 136..139 ----
  if (is_finish_cta(cta)) {
    int heads[2] = {0, 0}, nh = 1;
    if (cta < kFinishCtaBase + kQkvFinishers) {
      heads[0] = cta - kFinishCtaBase;
      heads[1] = heads[0] + kQkvFinishers;
      nh = 2;
    } else {
      heads[0] = cta == kQiFinisher   ? kHeadQi
                 : cta == kKiFinisher ? kHeadKi
                 : cta == kKFinisher  ? kHeadK
                                      : kHeadV;
    }
    proj::FinInputs fin;
    proj::fin_prefetch(p, heads, nh, tid, fin);
    // Every head straight from its tagged partial records.
    proj::qkfin_records<kDeadline>(p, dl, heads, nh, fin, uint32_t(epoch), tid);
    consumer_sync();
    if (tid == 0) {
      if (cta == kQiFinisher) {
        // The fence after the CTA barrier pushes the QI records out at once (SCORE polls them).
        fence_acq_rel_gpu();
      } else if (cta >= kKiFinisher) {
        // KI / K / V cache rows: later page copies of other CTAs wait for these counters.
        uint64_t *w[1] = {Counter::at(p.arena, proj::ready_counter(heads[0])).word};
        release_counters(w, 1);
      }
      // Q heads need no signal: ATTN polls their tagged records.
    }
  }

  // ---- TOPK ----
  if (cta >= kTopkCtaBase && cta < kTopkCtaBase + kTokens) {
    // The row's page table, staged in ring slot 0 while SCORE finishes elsewhere: every ring
    // item of a TOPK CTA (PROJ tiles, KI pages) has been consumed by now.
    int *kv_table = reinterpret_cast<int *>(ring_mem);
    static_assert(kMaxPages * 4 <= kSlotBytes, "one row's page table fits a ring slot");
    {
      // Asynchronous copies (cp.async, no registers held): the table lands while the selection
      // polls the score records; topk_row waits for it only before the physical-page lookups.
      const int visible = visible_pages(kv, cta - kTopkCtaBase);
      for (int j = tid; j < visible; j += kConsumerThreads)
        cp_async_4(smem_u32(kv_table + j), p.kv_indices + j);
      cp_async_commit();
    }
    // The selection polls the row's tagged score records directly; the barrier orders every earlier
    // read of the scratch the selection reuses.
    consumer_sync();
    score::topk_row(p, scratch, kv_table, cta - kTopkCtaBase, kv, uint32_t(epoch), tid);
  }

  // ---- ATTN + COMBINE ----
  if (n.kv) {
    const int row = cta / kTopK, split = cta % kTopK;
    // Wo tile 0 of this warp unpacked to f16 now, while the CTA waits for its Q and K/V page,
    // instead of on the O-image hand-off after COMBINE. Its image goes to the two ring slots after
    // the terminal K/V item (items i_wo - 2 and i_wo - 1: consumed by now and never refilled).
    if (n.wo)
      proj::stage_tile_a_f16(
          ring_acquire<kDeadline>(p, dl, ring, i_wo),
          ring.slot(i_kv + 1u + uint32_t(warp >> 2)) + uint32_t(warp & 3) * 8192u, warp, lane);
    // Q of this row: warp 0 polls the finishers' tagged records (heads 0..7 and 8..15) and hands
    // the B fragments to the other warps through shared memory.
    uint2 q0[4], q1[4];
    {
      uint2 *qs = reinterpret_cast<uint2 *>(scratch + attn::kQStage);
      if (warp == 0) {
        const uint8_t *recs = p.arena + arena::kQRec + size_t(row) * kHeads * (kHeadDim / 8) * 16;
        score::poll_query_frag<kDeadline>(p, dl, recs, uint32_t(epoch), lane, q0);
        score::poll_query_frag<kDeadline>(p, dl, recs + 8 * (kHeadDim / 8) * 16, uint32_t(epoch),
                                          lane, q1);
#pragma unroll
        for (int kc = 0; kc < 4; ++kc) {
          qs[lane * 8 + kc] = q0[kc];
          qs[lane * 8 + 4 + kc] = q1[kc];
        }
      }
      consumer_sync();
      if (warp != 0) {
#pragma unroll
        for (int kc = 0; kc < 4; ++kc) {
          q0[kc] = qs[lane * 8 + kc];
          q1[kc] = qs[lane * 8 + 4 + kc];
        }
      }
    }
    const uint32_t slot = ring_acquire<kDeadline>(p, dl, ring, i_kv);
    const attn::KvInfo info = ctl.kv_info;
    if (info.logical >= 0)
      attn::split_compute(p, scratch, slot, info, row, split, q0, q1, tid, warp, lane,
                          uint32_t(epoch));
    else
      attn::split_empty(p, row, split, tid, uint32_t(epoch));
    // COMBINE: split CTA s merges head s of the row, polling the 16 splits' tagged records.
    attn::combine_head<kDeadline>(p, dl, scratch, row, split, uint32_t(epoch), tid);
  }

  // ---- OPROJ ----
  if (n.wo) {
    uint32_t sfw[kWoTilesPerCta][2][2];
#pragma unroll
    for (int t = 0; t < kWoTilesPerCta; ++t)
      proj::load_weight_scales(p.wo_sf, kAttnK, cta * kWoTilesPerCta + t, 2 * warp, lane, sfw[t]);
    if (tid == 0) {
      ctl.stage[0] = stage_a;
      ctl.stage[1] = stage_b;
    }
    consumer_sync(); // the COMBINE scratch is free again
    // This warp's f16 image of Wo tile 0 (staged before ATTN in the two free ring slots).
    const uint32_t image0 = ring.slot(i_kv + 1u + uint32_t(warp >> 2)) + uint32_t(warp & 3) * 8192u;
    // The O image straight from the COMBINE CTAs' tagged records: each warp polls only the 8 rows x
    // 2 heads x 11 records of its own K slice (heads 2w, 2w + 1) into shared memory and starts its
    // GEMV at once; no CTA barrier before the split-K fold.
    {
      constexpr int kRecs = kTokens * 2 * arena::kORecsPerBlock; // 176 per warp
      constexpr int kPer = (kRecs + 31) / 32;                    // 6
      uint32_t pending = 0u;
#pragma unroll
      for (int k = 0; k < kPer; ++k)
        if (lane + k * 32 < kRecs)
          pending |= 1u << k;
      uint4 r[kPer];
      while (pending) {
#pragma unroll
        for (int k = 0; k < kPer; ++k)
          if (pending & (1u << k)) {
            const int i = lane + k * 32;
            const int blk = i / arena::kORecsPerBlock, j = i % arena::kORecsPerBlock;
            const int t = blk >> 1, h = 2 * warp + (blk & 1);
            r[k] = ld_rec(p.arena + arena::kORec +
                          ((size_t(t) * kHeads + h) * arena::kORecsPerBlock + j) * 16);
          }
#pragma unroll
        for (int k = 0; k < kPer; ++k) {
          if (!(pending & (1u << k)) || r[k].w != uint32_t(epoch))
            continue;
          pending &= ~(1u << k);
          const int i = lane + k * 32;
          const int blk = i / arena::kORecsPerBlock, j = i % arena::kORecsPerBlock;
          const int t = blk >> 1, h = 2 * warp + (blk & 1);
          const uint32_t words[3] = {r[k].x, r[k].y, r[k].z};
#pragma unroll
          for (int q = 0; q < 3; ++q) {
            const int wd = 3 * j + q;
            uint32_t *dst = wd < 32 ? reinterpret_cast<uint32_t *>(scratch + oproj::kPayload +
                                                                   t * kAttnK + h * 128 + 4 * wd)
                                    : reinterpret_cast<uint32_t *>(scratch + oproj::kScale +
                                                                   t * (kAttnK / 32) + 4 * h);
            *dst = words[q];
          }
        }
        if (pending)
          dl.check(p, 0x40u, pending, epoch);
      }
      __syncwarp();
    }
    proj::BFragH f;
    {
      proj::BFrag raw;
      proj::smem_bfrag(scratch + oproj::kPayload, scratch + oproj::kScale, kAttnK / 32,
                       warp * kGroupsPerWarp, lane, raw);
      proj::unpack_bfrag(raw, f);
    }
    // The three tiles (all landed long before) in one interleaved pass: three independent HMMA
    // chains per warp instead of one, which hides their latency.
    {
      uint32_t slots[kWoTilesPerCta];
#pragma unroll
      for (int t = 0; t < kWoTilesPerCta; ++t)
        slots[t] = ring_acquire<kDeadline>(p, dl, ring, i_wo + t);
      float run[kWoTilesPerCta][4];
      proj::tiles_gemv_h_pre0<kWoTilesPerCta>(slots, image0, sfw, f, warp, lane, run);
#pragma unroll
      for (int t = 0; t < kWoTilesPerCta; ++t)
        oproj::stage_tile(scratch, t, run[t], warp, lane);
    }
    oproj::publish_tiles(p, scratch, cta, ctl.stage, tid);
    consumer_sync();
    if (tid == 0) {
      const uint64_t old = atom_acq_rel_add(Counter::at(p.arena, arena::kCntOprojDone).word, 1ull);
      if (old + 1ull == uint64_t(kAttnCtas) * epoch) {
        uint32_t *words = oproj::publish_words(p);
        for (int t = 0; t < mailbox::kOutputTiles; ++t)
          words[t] = (words[t] + 1u) % uint32_t(mailbox::kGenerations);
      }
    }
  }

  // ---- FINAL ----
  if (cta >= kFinalCtaBase && cta < kFinalCtaBase + kTokens) {
    if (tid == 0)
      ctl.stage[0] = stage_a;
    consumer_sync();
    oproj::final_token<kDeadline>(p, dl, scratch, cta - kFinalCtaBase, ctl.stage[0], tid, warp,
                                  lane);
  }
}

} // namespace attn_ring
