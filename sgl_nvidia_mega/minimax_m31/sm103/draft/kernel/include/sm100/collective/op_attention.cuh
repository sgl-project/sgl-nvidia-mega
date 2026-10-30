#pragma once

// OP attention: q/k norm + RoPE, the block's cache write and split-KV attention chunks.

#include "include/sm100/common/sync.cuh"

namespace m31_draft {
namespace {

// ---------------------------------------------------------------------------
__device__ __forceinline__ bool later_duplicate(SmemMisc const *misc, int t) {
  bool dup = false;
  for (int u = t + 1; u < kRows; ++u)
    dup |= misc->slot_mapping[u] == misc->slot_mapping[t];
  return dup;
}

// ---------------------------------------------------------------------------
// Attention: every block row attends the same window ending at kv_len. One work
// item = one 64-key chunk for the 8 heads of one half (56 query rows, local row
// token * 8 + head % 8); warp w < 4 owns local rows 16w..16w+15 (tokens 2w, 2w + 1; the
// last tile pads token 7). Unnormalized output and (max, sum) per row go to the arena at
// the query row token * 16 + head.
__device__ void attention_chunk(Ctx &c, int chunk, int half, int keys, int first) {
  int const lane = c.lane, g = lane >> 2, tig = lane & 3;
  int const m0 = c.warp * 16;
  uint32_t const sq = smem_u32(c.smem + kSmemQ), sk = smem_u32(c.smem + kSmemK),
                 sv = smem_u32(c.smem + kSmemV);
  float s[kChunk / 8][4];
#pragma unroll
  for (int nt = 0; nt < kChunk / 8; ++nt)
    s[nt][0] = s[nt][1] = s[nt][2] = s[nt][3] = 0.0f;
#pragma unroll
  for (int ks = 0; ks < kHeadDim / 16; ++ks) {
    uint32_t qa[4];
    ldsm_x4(qa, sq + ((m0 + (lane & 15)) * kARow + ks * 16 + (lane >> 4) * 8) * 2);
#pragma unroll
    for (int np = 0; np < kChunk / 16; ++np) {
      uint32_t kb[4];
      ldsm_x4(kb, sk + ((np * 16 + (lane & 7) + (lane >> 4) * 8) * kARow + ks * 16 +
                        ((lane >> 3) & 1) * 8) *
                           2);
      mma_bf16(s[2 * np], qa, kb[0], kb[1]);
      mma_bf16(s[2 * np + 1], qa, kb[2], kb[3]);
    }
  }
  float row_max[2] = {kNegInf, kNegInf};
#pragma unroll
  for (int nt = 0; nt < kChunk / 8; ++nt) {
#pragma unroll
    for (int j = 0; j < 4; ++j) {
      int const key = first + nt * 8 + tig * 2 + (j & 1);
      s[nt][j] = key < keys ? s[nt][j] * kSoftmaxScaleLog2 : kNegInf;
      row_max[j >> 1] = fmaxf(row_max[j >> 1], s[nt][j]);
    }
  }
  float row_sum[2] = {0.0f, 0.0f};
#pragma unroll
  for (int h = 0; h < 2; ++h) {
    row_max[h] = fmaxf(row_max[h], __shfl_xor_sync(0xffffffffu, row_max[h], 1));
    row_max[h] = fmaxf(row_max[h], __shfl_xor_sync(0xffffffffu, row_max[h], 2));
  }
#pragma unroll
  for (int nt = 0; nt < kChunk / 8; ++nt) {
#pragma unroll
    for (int j = 0; j < 4; ++j) {
      s[nt][j] = exp2f(s[nt][j] - row_max[j >> 1]);
      row_sum[j >> 1] += s[nt][j];
    }
  }
#pragma unroll
  for (int h = 0; h < 2; ++h) {
    row_sum[h] += __shfl_xor_sync(0xffffffffu, row_sum[h], 1);
    row_sum[h] += __shfl_xor_sync(0xffffffffu, row_sum[h], 2);
  }
  float o[kHeadDim / 8][4];
#pragma unroll
  for (int dt = 0; dt < kHeadDim / 8; ++dt)
    o[dt][0] = o[dt][1] = o[dt][2] = o[dt][3] = 0.0f;
#pragma unroll
  for (int ks = 0; ks < kChunk / 16; ++ks) {
    uint32_t const a[4] = {pack_bf16(s[2 * ks][0], s[2 * ks][1]),
                           pack_bf16(s[2 * ks][2], s[2 * ks][3]),
                           pack_bf16(s[2 * ks + 1][0], s[2 * ks + 1][1]),
                           pack_bf16(s[2 * ks + 1][2], s[2 * ks + 1][3])};
#pragma unroll
    for (int dp = 0; dp < kHeadDim / 16; ++dp) {
      uint32_t vb[4];
      ldsm_x4_t(vb, sv + ((ks * 16 + (lane & 7) + ((lane >> 3) & 1) * 8) * kARow + dp * 16 +
                          (lane >> 4) * 8) *
                             2);
      mma_bf16(o[2 * dp], a, vb[0], vb[1]);
      mma_bf16(o[2 * dp + 1], a, vb[2], vb[3]);
    }
  }
  float *partial = at<float>(c.arena, kAttnPartial) + size_t(chunk) * kAttnRows * kHeadDim;
  // local rows m0 + g (token 2 warp) and m0 + g + 8 (token 2 warp + 1; token 7 is padding)
  int const r0 = (m0 >> 3) * kQHeads + half * kHalfHeads + g, r1 = r0 + kQHeads;
  bool const has1 = (m0 >> 3) + 1 < kRows;
#pragma unroll
  for (int dt = 0; dt < kHeadDim / 8; ++dt) {
    int const dim = dt * 8 + tig * 2;
    *reinterpret_cast<float2 *>(partial + size_t(r0) * kHeadDim + dim) =
        make_float2(o[dt][0], o[dt][1]);
    if (has1)
      *reinterpret_cast<float2 *>(partial + size_t(r1) * kHeadDim + dim) =
          make_float2(o[dt][2], o[dt][3]);
  }
  if (tig == 0) {
    // [row][chunk] (max, sum): the combine CTA of a row reads its chunks' stats contiguously
    float *stats = at<float>(c.arena, kAttnStats) + size_t(chunk) * 2;
    *reinterpret_cast<float2 *>(stats + size_t(r0) * kMaxChunks * 2) =
        make_float2(row_max[0], row_sum[0]);
    if (has1)
      *reinterpret_cast<float2 *>(stats + size_t(r1) * kMaxChunks * 2) =
          make_float2(row_max[1], row_sum[1]);
  }
}

__device__ __forceinline__ int64_t window_slot(Args const &a, int pos) {
  return int64_t(a.page_table[pos / kPage]) * kPage + pos % kPage;
}

// The attention inputs that do not depend on the layer's activations are staged into shared
// memory once per call, so no attention phase waits on a global load before it polls the qkv
// words: the window (and its page-table entries) is the same for every layer, and so are the
// block's slots and positions. Issued at entry (loads into registers, consumed after the
// embedding push, off the critical path); attention_phase reads them after several barriers.
struct AttnStaging {
  int4 qk;      // threads < 160: 8 q/k norm weights (layer tid / 32, q | k, part tid % 16)
  float4 cs;    // threads < 112: 4 cos | sin values of block row tid / 16
  int32_t page; // threads < kChunk: page of window row tid of the first chunk
};
__device__ AttnStaging stage_attention_issue(Ctx const &c) {
  AttnStaging s{};
  int const tid = c.tid;
  if (tid < kLayers * 2 * (kHeadDim / 8)) {
    LayerArgs const &layer = c.a.layer[tid / (2 * (kHeadDim / 8))];
    int const which = (tid / (kHeadDim / 8)) % 2, part = tid % (kHeadDim / 8);
    s.qk = __ldg(reinterpret_cast<int4 const *>(which ? layer.k_norm : layer.q_norm) + part);
  }
  if (tid < kRows * (kRotaryDim / 4)) {
    int const t = tid / (kRotaryDim / 4), j = tid % (kRotaryDim / 4);
    s.cs = __ldg(
        reinterpret_cast<float4 const *>(c.a.cos_sin + size_t(c.misc->positions[t]) * kRotaryDim) +
        j);
  }
  int const kv_len = c.misc->kv_len;
  int const keys = kv_len < kWindow ? kv_len : kWindow;
  int const first = first_attention_chunk(c.cta) * kChunk;
  if (tid < kChunk && first + tid < keys)
    s.page = c.a.page_table[(kv_len - keys + first + tid) / kPage];
  return s;
}
__device__ void stage_attention_store(Ctx &c, AttnStaging const &s) {
  int const tid = c.tid;
  if (tid < kLayers * 2 * (kHeadDim / 8)) {
    int const l = tid / (2 * (kHeadDim / 8)), which = (tid / (kHeadDim / 8)) % 2,
              part = tid % (kHeadDim / 8);
    float *dst = &c.misc->qkw[l][which][part * 8];
    uint32_t const w[4] = {uint32_t(s.qk.x), uint32_t(s.qk.y), uint32_t(s.qk.z), uint32_t(s.qk.w)};
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      float2 const v = bf2x2(w[i]);
      dst[2 * i] = 1.0f + v.x;
      dst[2 * i + 1] = 1.0f + v.y;
    }
  }
  if (tid < kRows * (kRotaryDim / 4)) {
    int const t = tid / (kRotaryDim / 4), j = tid % (kRotaryDim / 4);
    *reinterpret_cast<float4 *>(&c.misc->cs[t][4 * j]) = s.cs;
  }
  if (tid < kChunk) {
    int const kv_len = c.misc->kv_len;
    int const keys = kv_len < kWindow ? kv_len : kWindow;
    int const first = first_attention_chunk(c.cta) * kChunk;
    int64_t slot = -1;
    int fix = -1;
    if (first + tid < keys) {
      int const pos = kv_len - keys + first + tid;
      slot = int64_t(s.page) * kPage + pos % kPage;
      for (int t = 0; t < kRows; ++t)
        fix = c.misc->slot_mapping[t] == slot ? t : fix;
    }
    c.misc->win_slot[tid] = slot;
    c.misc->win_fix[tid] = fix;
    // whether any row of the staged chunk takes a block row's fresh K/V (read after barriers)
    bool const any = __any_sync(0xffffffffu, fix >= 0);
    if ((tid & 31) == 0)
      c.misc->win_any[tid >> 5] = any ? 1 : 0;
  }
}

// The layer's attention operands of one head half in shared memory: q of the half's 56 block rows
// after the Gemma q norm + NeoX RoPE on the first 64 dims (SGLang's fused kernel order), and the
// block's fresh K (k norm + RoPE) and V rows, all BF16. The raw rows are polled directly from the
// qkv LL words: warp w owns rows w, w + 8, ... (q rows 0-55: token * 8 + head % 8, k rows 56-62,
// v rows 63-69); lane l holds dims 2l, 2l + 1 (word l) and 64 + 2l, 65 + 2l (word 32 + l), so the
// RoPE partner of its first two dims sits in lane l ^ 16. The 1 + q/k norm weights and the block
// rows' cos/sin were staged once per call (stage_attention_store).
__device__ void build_qkv(Ctx &c, int layer, int half) {
  constexpr int kAllRows = kItemRows + 2 * kRows;
  constexpr int kPer = (kAllRows + kConsumerWarps - 1) / kConsumerWarps;
  uint16_t *sq = reinterpret_cast<uint16_t *>(c.smem + kSmemQ);
  uint16_t *fresh = reinterpret_cast<uint16_t *>(c.smem + kSmemFresh);
  float const *s_qkw = &c.misc->qkw[layer][0][0];
  float const *s_cs = &c.misc->cs[0][0];
  uint64_t const *src = qkv_ll(c.arena, layer);
  uint32_t const tag = ll_tag(c.epoch);
  int const lane = c.lane;
  // No gate: the attention CTAs arrive here as the qkv units finish (most of them computed one),
  // so the full poll below picks every word up one round trip after it lands.
  auto row_words = [&](int row) -> uint64_t const * {
    int const t = row < kItemRows ? row / kHalfHeads : (row - kItemRows) % kRows;
    int const col = row < kItemRows ? (half * kHalfHeads + row % kHalfHeads) * kHeadDim
                                    : kQSize + (row - kItemRows) / kRows * kHeadDim;
    return src + t * kQkvLLRowWords + col / 2;
  };
  uint64_t w[kPer][2];
#pragma unroll
  for (int b = 0; b < kPer; ++b) {
    int const row = c.warp + kConsumerWarps * b;
    if (row < kAllRows) {
      uint64_t const *p = row_words(row);
      w[b][0] = ld_relaxed_gpu_u64(p + lane);
      w[b][1] = ld_relaxed_gpu_u64(p + 32 + lane);
    } else {
      w[b][0] = w[b][1] = ll_word(0, tag);
    }
  }
  unsigned long long start = 0;
  for (;;) {
    bool ready = true;
#pragma unroll
    for (int b = 0; b < kPer; ++b)
      ready &= ll_ok(w[b][0], tag) && ll_ok(w[b][1], tag);
    if (ready)
      break;
    if (start == 0)
      start = globaltimer();
    else
      check_deadline(start);
#pragma unroll
    for (int b = 0; b < kPer; ++b) {
      int const row = c.warp + kConsumerWarps * b;
#pragma unroll
      for (int h = 0; h < 2; ++h)
        if (!ll_ok(w[b][h], tag))
          w[b][h] = ld_relaxed_gpu_u64(row_words(row) + 32 * h + lane);
    }
  }
  // the last tile's padding rows (token 7) read zeros
  for (int i = c.tid; i < (kItemTiles * 16 - kItemRows) * (kHeadDim / 8); i += kConsumerThreads)
    *reinterpret_cast<int4 *>(sq + (kItemRows + i / (kHeadDim / 8)) * kARow +
                              (i % (kHeadDim / 8)) * 8) = make_int4(0, 0, 0, 0);
  // cos/sin index of the lane's first two dims (mod the rotary half)
  int const ci = (2 * lane) & 31;
#pragma unroll
  for (int b = 0; b < kPer; ++b) {
    int const row = c.warp + kConsumerWarps * b;
    if (row >= kAllRows)
      continue;
    if (row >= kItemRows + kRows) { // v: stored as is
      uint32_t *dst = reinterpret_cast<uint32_t *>(fresh + (row - kItemRows) * kHeadDim);
      dst[lane] = uint32_t(w[b][0]);
      dst[32 + lane] = uint32_t(w[b][1]);
      continue;
    }
    bool const q = row < kItemRows;
    int const t = q ? row / kHalfHeads : row - kItemRows;
    float const x0 = bf2f(uint16_t(w[b][0])), x1 = bf2f(uint16_t(w[b][0] >> 16));
    float const x2 = bf2f(uint16_t(w[b][1])), x3 = bf2f(uint16_t(w[b][1] >> 16));
    float ss = x0 * x0 + x1 * x1 + x2 * x2 + x3 * x3;
#pragma unroll
    for (int mask = 16; mask > 0; mask >>= 1)
      ss += __shfl_xor_sync(0xffffffffu, ss, mask);
    float const inv_rms = rsqrtf(ss / float(kHeadDim) + c.a.eps);
    float const *wt = s_qkw + (q ? 0 : kHeadDim);
    float const n0 = x0 * inv_rms * wt[2 * lane];
    float const n1 = x1 * inv_rms * wt[2 * lane + 1];
    float const n2 = x2 * inv_rms * wt[kHeadDim / 2 + 2 * lane];
    float const n3 = x3 * inv_rms * wt[kHeadDim / 2 + 2 * lane + 1];
    float const p0 = __shfl_xor_sync(0xffffffffu, n0, 16),
                p1 = __shfl_xor_sync(0xffffffffu, n1, 16);
    float const *cs = s_cs + t * kRotaryDim;
    float const cos0 = cs[ci], cos1 = cs[ci + 1], sin0 = cs[kRotaryDim / 2 + ci],
                sin1 = cs[kRotaryDim / 2 + ci + 1];
    // dims d < 32: n[d] cos - n[d + 32] sin; dims d + 32: n[d + 32] cos + n[d] sin
    float const o0 = lane < 16 ? n0 * cos0 - p0 * sin0 : n0 * cos0 + p0 * sin0;
    float const o1 = lane < 16 ? n1 * cos1 - p1 * sin1 : n1 * cos1 + p1 * sin1;
    uint16_t *dst = q ? sq + row * kARow : fresh + t * kHeadDim;
    reinterpret_cast<uint32_t *>(dst)[lane] = pack_bf16(o0, o1);
    reinterpret_cast<uint32_t *>(dst + kHeadDim / 2)[lane] = pack_bf16(n2, n3);
  }
  consumer_sync();
}

// Attention: every block row attends the same window ending at kv_len. One work item = one
// 64-key chunk for the 56 query rows of one head half. Window rows are read from the cache as it
// was before this call; rows that are one of the block's slots take the block's fresh K/V (the
// last block row of a slot wins, as in the cache write), so the result equals writing the block
// first and reading afterwards. CTA 0 writes the block's K/V to the cache.
__device__ void attention_phase(Ctx &c, int layer) {
  LayerArgs const &layer_args = c.a.layer[layer];
  int const kv_len = c.misc->kv_len;
  int const keys = kv_len < kWindow ? kv_len : kWindow;
  int const start = kv_len - keys;
  int const chunks = (keys + kChunk - 1) / kChunk;
  int const items = 2 * chunks;
  uint16_t *sk = reinterpret_cast<uint16_t *>(c.smem + kSmemK);
  uint16_t *sv = reinterpret_cast<uint16_t *>(c.smem + kSmemV);
  uint16_t const *fresh = reinterpret_cast<uint16_t const *>(c.smem + kSmemFresh);
  bool built = false;
  int built_half = -1;
  for (int item = first_attention_item(c.cta); item < items; item += c.grid) {
    int const chunk = item_chunk(item), half = item_half(item);
    int const first = chunk * kChunk;
    // The first item's chunk slots and replacement rows were staged at entry.
    bool const staged = item == first_attention_item(c.cta);
    int const *fix_rows = staged ? c.misc->win_fix : c.misc->fix;
    for (int i = c.tid; i < kChunk * (kHeadDim / 8); i += kConsumerThreads) {
      int const row = i / (kHeadDim / 8), part = i % (kHeadDim / 8);
      if (first + row < keys) {
        size_t const slot =
            size_t(staged ? c.misc->win_slot[row] : window_slot(c.a, start + first + row));
        cp_async_16(smem_u32(sk + row * kARow + part * 8),
                    layer_args.k_cache + slot * kHeadDim + part * 8);
        cp_async_16(smem_u32(sv + row * kARow + part * 8),
                    layer_args.v_cache + slot * kHeadDim + part * 8);
      } else {
        *reinterpret_cast<int4 *>(sk + row * kARow + part * 8) = make_int4(0, 0, 0, 0);
        *reinterpret_cast<int4 *>(sv + row * kARow + part * 8) = make_int4(0, 0, 0, 0);
      }
    }
    if (!staged && c.tid < kChunk) {
      int fix = -1;
      if (first + c.tid < keys) {
        int64_t const slot = window_slot(c.a, start + first + c.tid);
        for (int t = 0; t < kRows; ++t)
          fix = c.misc->slot_mapping[t] == slot ? t : fix;
      }
      c.misc->fix[c.tid] = fix;
    }
    if (built_half != half) {
      // polls the q/k/v words while this chunk's K/V copies are in flight
      build_qkv(c, layer, half);
      if (writes_cache(c.cta) && !built) {
        for (int i = c.tid; i < 2 * kRows * (kHeadDim / 8); i += kConsumerThreads) {
          int const k = i / (kHeadDim / 8), part = i % (kHeadDim / 8);
          int const which = k / kRows, t = k % kRows;
          int64_t const slot = c.misc->slot_mapping[t];
          if (slot >= 0 && !later_duplicate(c.misc, t))
            *(reinterpret_cast<int4 *>((which ? layer_args.v_cache : layer_args.k_cache) +
                                       size_t(slot) * kHeadDim) +
              part) = *reinterpret_cast<int4 const *>(fresh + k * kHeadDim + part * 8);
        }
      }
      built = true;
      built_half = half;
    }
    cp_async_wait_all();
    consumer_sync(); // the chunk's K/V rows are visible before the block rows replace some of them
    // The staged chunk records whether any of its rows is a block slot; other chunks always check.
    if (!staged || c.misc->win_any[0] || c.misc->win_any[1]) {
      for (int i = c.tid; i < kChunk * (kHeadDim / 8); i += kConsumerThreads) {
        int const row = i / (kHeadDim / 8), part = i % (kHeadDim / 8);
        int const t = fix_rows[row];
        if (t >= 0) {
          *reinterpret_cast<int4 *>(sk + row * kARow + part * 8) =
              *reinterpret_cast<int4 const *>(fresh + t * kHeadDim + part * 8);
          *reinterpret_cast<int4 *>(sv + row * kARow + part * 8) =
              *reinterpret_cast<int4 const *>(fresh + (kRows + t) * kHeadDim + part * 8);
        }
      }
      consumer_sync();
    }
    if (c.warp < kItemTiles)
      attention_chunk(c, chunk, half, keys, first);
    consumer_sync();
  }
  if (c.tid == 0)
    red_add_release_gpu(signal(c.arena, kSigAttn, layer, c.cta), 1u);
}

} // namespace
} // namespace m31_draft
