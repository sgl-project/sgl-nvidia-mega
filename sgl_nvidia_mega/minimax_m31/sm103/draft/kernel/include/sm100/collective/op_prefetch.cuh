#pragma once

// Entry prefetch of the latency-critical inputs into L2.

#include "include/sm100/common/sync.cuh"

namespace m31_draft {
namespace {

// Latency-critical inputs go to L2 before the weight stream starts, so they never
// queue behind it: every layer's norm weights, q/k norm weights, the block's
// cos/sin rows and every layer's K/V rows of this CTA's attention chunk (the
// window and its page-table entries are the same for all layers). Issued by the
// weight-stream warp before it waits for the consumers' go, so the consumer warps
// start the embedding at once (no page-table load ahead of their embedding chunks).
__device__ void entry_prefetch(Args const &a, SmemMisc const *misc, int cta, int lane) {
  int const kv_len = misc->kv_len;
  int const keys = kv_len < kWindow ? kv_len : kWindow;
  int const start = kv_len - keys;
  int const i = lane; // 32 small tensors
  if (i < 2 * kLayers) {
    LayerArgs const &layer = a.layer[i >> 1];
    bulk_prefetch_l2((i & 1) ? layer.post_norm : layer.input_norm, kHidden * 2);
  } else if (i == 2 * kLayers) {
    bulk_prefetch_l2(a.final_norm, kHidden * 2);
  } else if (i < 3 * kLayers + 1) {
    LayerArgs const &layer = a.layer[i - (2 * kLayers + 1)];
    bulk_prefetch_l2(layer.q_norm, kHeadDim * 2);
    bulk_prefetch_l2(layer.k_norm, kHeadDim * 2);
  } else if (i < 3 * kLayers + 1 + kRows) {
    bulk_prefetch_l2(a.cos_sin + size_t(misc->positions[i - (3 * kLayers + 1)]) * kRotaryDim,
                     kRotaryDim * 4);
  }
  if (lane < 2 * kLayers && item_half(first_attention_item(cta)) == 0) {
    // K/V rows of this CTA's chunk (once per chunk: the CTA of its first head half): at most two
    // contiguous runs (pages) of slots.
    int const first = first_attention_chunk(cta) * kChunk;
    if (first < keys) {
      int const last = (first + kChunk < keys ? first + kChunk : keys) - 1;
      LayerArgs const &layer = a.layer[lane >> 1];
      uint16_t const *cache = (lane & 1) ? layer.v_cache : layer.k_cache;
      int pos = start + first;
      int const end = start + last;
      while (pos <= end) {
        int const page_end = (pos / kPage) * kPage + kPage - 1;
        int const run_end = page_end < end ? page_end : end;
        size_t const slot = size_t(a.page_table[pos / kPage]) * kPage + pos % kPage;
        bulk_prefetch_l2(cache + slot * kHeadDim, uint32_t(run_end - pos + 1) * kHeadDim * 2);
        pos = run_end + 1;
      }
    }
  }
}

} // namespace
} // namespace m31_draft
