// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// The trtllm-gen routing of one token's 512 bf16 router logits (top-10 expert ids and bf16 weights, the bits of
// FlashInfer's single-block routing, RenormalizeNaive) and the routing tables the trtllm-gen MoE body reads (what its
// routing kernels write, for the rank's window of local experts; a slot outside it gets permuted index -1). Shared by
// route_topk.cu and route_big.cu; the includer provides uint16_t, uint32_t, int32_t, cuda_bf16.h and
// cooperative_groups (with its reduce header).

#pragma once

#ifndef INFINITY
#define INFINITY __int_as_float(0x7f800000)
#endif

namespace q35_route_tables {

constexpr uint32_t kExperts = 512, kTopK = 10, kThreads = 256;

// ---- the routing (mkm_moe_m16.cu): key = bf16 order above 65535 - expert (higher score first, smaller expert on ties)
__device__ __forceinline__ uint32_t topk_key(uint32_t bits, uint32_t expert) {
  const uint32_t twiddled = bits ^ ((bits & 0x8000u) ? 0xFFFFu : 0x8000u);
  return (twiddled << 16) | (65535u - expert);
}
__device__ __forceinline__ void topk_unkey(uint32_t key, uint32_t& bits, uint32_t& expert) {
  expert = 65535u - (key & 0xFFFFu);
  const uint32_t twiddled = key >> 16;
  bits = twiddled ^ ((twiddled & 0x8000u) ? 0x8000u : 0xFFFFu);
}
__device__ __forceinline__ uint32_t max16(const uint32_t (&k)[16]) {
  const uint32_t a = max(max(max(k[0], k[1]), max(k[2], k[3])), max(max(k[4], k[5]), max(k[6], k[7])));
  const uint32_t b = max(max(max(k[8], k[9]), max(k[10], k[11])), max(max(k[12], k[13]), max(k[14], k[15])));
  return max(a, b);
}
__device__ __forceinline__ uint32_t warp_max_u32(uint32_t v) {
  uint32_t r;
  asm volatile("redux.sync.max.u32 %0, %1, 0xffffffff;" : "=r"(r) : "r"(v));
  return r;
}
__device__ __forceinline__ uint32_t warp_sum_u32(uint32_t v) {
  uint32_t r;
  asm volatile("redux.sync.add.u32 %0, %1, 0xffffffff;" : "=r"(r) : "r"(v));
  return r;
}
__device__ __forceinline__ uint32_t warp_topk_of_lanes(uint32_t v, uint32_t lane) {
  uint32_t mine = 0;
#pragma unroll
  for (uint32_t k = 0; k < kTopK; ++k) {
    const uint32_t top = warp_max_u32(v);
    v = v == top ? 0u : v;
    mine = lane == k ? top : mine;
  }
  return mine;
}
// Top-10 keys of 512 (16 per lane), lane k holding the k-th largest (keys unique and nonzero).
__device__ __forceinline__ uint32_t warp_top10_redux(uint32_t (&keys)[16], uint32_t lane, uint32_t* s_cand) {
  uint32_t v = max16(keys);
  uint32_t bound = 0;
#pragma unroll
  for (uint32_t k = 0; k < kTopK; ++k) {
    bound = warp_max_u32(v);
    v = v == bound ? 0u : v;
  }
  uint32_t count = 0;
#pragma unroll
  for (uint32_t i = 0; i < 16; ++i) count += keys[i] >= bound;
  const uint32_t total = warp_sum_u32(count);
  if (total <= 32) {
    uint32_t offset = count;
#pragma unroll
    for (uint32_t d = 1; d < 32; d <<= 1) {
      const uint32_t up = __shfl_up_sync(0xffffffffu, offset, d);
      offset += lane >= d ? up : 0u;
    }
    offset -= count;
#pragma unroll
    for (uint32_t i = 0; i < 16; ++i) {
      if (keys[i] >= bound) s_cand[offset++] = keys[i];
    }
    __syncwarp();
    const uint32_t cand = lane < total ? s_cand[lane] : 0u;
    __syncwarp();
    return warp_topk_of_lanes(cand, lane);
  }
  uint32_t mine = 0;
#pragma unroll 1
  for (uint32_t k = 0; k < kTopK; ++k) {
    const uint32_t top = warp_max_u32(max16(keys));
#pragma unroll
    for (uint32_t i = 0; i < 16; ++i) keys[i] = keys[i] == top ? 0u : keys[i];
    mine = lane == k ? top : mine;
  }
  return mine;
}
// FlashInfer calcSoftmax (bf16 scores) as built with -use_fast_math.
__device__ __forceinline__ float fast_expf(float x) {
  float y;
  asm("{\n .reg .f32 t;\n mul.ftz.f32 t, %1, 0f3FB8AA3B;\n ex2.approx.ftz.f32 %0, t;\n}" : "=f"(y) : "f"(x));
  return y;
}
__device__ __forceinline__ float fast_div(float a, float b) {
  float y;
  asm("div.full.ftz.f32 %0, %1, %2;" : "=f"(y) : "f"(a), "f"(b));
  return y;
}
__device__ __forceinline__ uint16_t softmax_topk(uint16_t score_bits, uint32_t lane) {
  namespace cg = cooperative_groups;
  const auto warp = cg::tiled_partition<32>(cg::this_thread_block());
  const float si = __uint_as_float((uint32_t)score_bits << 16);
  float max_score = -INFINITY;
  if (lane < kTopK) max_score = si >= max_score ? si : max_score;
  max_score = cg::reduce(warp, max_score, cg::greater<float>());
  float sum_score = 0.f, new_score = 0.f;
  if (lane < kTopK) {
    new_score = fast_expf(__fsub_rn(si, max_score));
    sum_score = __fadd_rn(sum_score, new_score);
  }
  sum_score = cg::reduce(warp, sum_score, cg::plus<float>());
  uint16_t out = score_bits;
  if (lane < kTopK) {
    const __nv_bfloat16 b = __float2bfloat16_rn(fast_div(new_score, sum_score));
    out = *reinterpret_cast<const uint16_t*>(&b);
  }
  return out;
}


// One token's routing by one warp from its logits (lane l holds experts [16 l, 16 l + 16), loaded by the caller): lane
// k < 10 gets the k-th expert and its bf16 weight bits.
__device__ __forceinline__ void route_row(
    const uint4& v0, const uint4& v1, uint32_t lane, uint32_t* s_cand, uint32_t& expert, uint16_t& wbits) {
  uint32_t keys[16];
#pragma unroll
  for (int i = 0; i < 8; ++i) {
    const uint32_t w = i < 4 ? (&v0.x)[i] : (&v1.x)[i - 4];
    keys[2 * i] = topk_key(w & 0xFFFFu, lane * 16 + 2 * i);
    keys[2 * i + 1] = topk_key(w >> 16, lane * 16 + 2 * i + 1);
  }
  const uint32_t key = warp_top10_redux(keys, lane, s_cand);
  uint32_t bits;
  topk_unkey(key, bits, expert);
  wbits = softmax_topk(static_cast<uint16_t>(bits), lane);
}

// Two tokens' routing by one warp, their steps interleaved (each token's operations as route_row's); with more than
// a warp of candidates at or past either token's 10th key both take route_row. s_cand: [2][32].
__device__ __forceinline__ void route_row2(
    const uint4 (&v)[2][2], uint32_t lane, uint32_t* s_cand, uint32_t (&expert)[2], uint16_t (&wbits)[2]) {
  uint32_t keys[2][16], mx[2], bound[2] = {0u, 0u}, count[2] = {0u, 0u}, total[2];
#pragma unroll
  for (int t = 0; t < 2; ++t) {
#pragma unroll
    for (int i = 0; i < 8; ++i) {
      const uint32_t w = i < 4 ? (&v[t][0].x)[i] : (&v[t][1].x)[i - 4];
      keys[t][2 * i] = topk_key(w & 0xFFFFu, lane * 16 + 2 * i);
      keys[t][2 * i + 1] = topk_key(w >> 16, lane * 16 + 2 * i + 1);
    }
    mx[t] = max16(keys[t]);
  }
#pragma unroll
  for (uint32_t k = 0; k < kTopK; ++k) {
#pragma unroll
    for (int t = 0; t < 2; ++t) {
      bound[t] = warp_max_u32(mx[t]);
      mx[t] = mx[t] == bound[t] ? 0u : mx[t];
    }
  }
#pragma unroll
  for (int t = 0; t < 2; ++t) {
#pragma unroll
    for (uint32_t i = 0; i < 16; ++i) count[t] += keys[t][i] >= bound[t];
    total[t] = warp_sum_u32(count[t]);
  }
  if (total[0] > 32 || total[1] > 32) {
#pragma unroll
    for (int t = 0; t < 2; ++t) route_row(v[t][0], v[t][1], lane, s_cand, expert[t], wbits[t]);
    return;
  }
  uint32_t offset[2] = {count[0], count[1]};
#pragma unroll
  for (uint32_t d = 1; d < 32; d <<= 1) {
#pragma unroll
    for (int t = 0; t < 2; ++t) {
      const uint32_t up = __shfl_up_sync(0xffffffffu, offset[t], d);
      offset[t] += lane >= d ? up : 0u;
    }
  }
#pragma unroll
  for (int t = 0; t < 2; ++t) {
    offset[t] -= count[t];
#pragma unroll
    for (uint32_t i = 0; i < 16; ++i) {
      if (keys[t][i] >= bound[t]) s_cand[t * 32 + offset[t]++] = keys[t][i];
    }
  }
  __syncwarp();
  uint32_t cand[2], mine[2] = {0u, 0u};
#pragma unroll
  for (int t = 0; t < 2; ++t) cand[t] = lane < total[t] ? s_cand[t * 32 + lane] : 0u;
  __syncwarp();
#pragma unroll
  for (uint32_t k = 0; k < kTopK; ++k) {
#pragma unroll
    for (int t = 0; t < 2; ++t) {
      const uint32_t top = warp_max_u32(cand[t]);
      cand[t] = cand[t] == top ? 0u : cand[t];
      mine[t] = lane == k ? top : mine[t];
    }
  }
#pragma unroll
  for (int t = 0; t < 2; ++t) {
    uint32_t bits;
    topk_unkey(mine[t], bits, expert[t]);
    wbits[t] = softmax_topk(static_cast<uint16_t>(bits), lane);
  }
}

// ---- TRT-LLM r3's candidate top-10 (q35_moe_front_big._route_warp_fast): each lane keeps the 4 largest of its 16
// keys (a selection network), the 10 rounds pop from those; when a lane runs dry by the ninth round (it holds 5 or more
// of the top 10) the warp takes the full path. The same 10 keys in the same order as warp_top10_redux.
__device__ __forceinline__ void cas_u32(uint32_t& hi, uint32_t& lo) {
  const uint32_t a = max(hi, lo), b = min(hi, lo);
  hi = a;
  lo = b;
}
// the 4 largest of two descending 4s, descending
__device__ __forceinline__ void top4of8(const uint32_t (&a)[4], const uint32_t (&b)[4], uint32_t (&o)[4]) {
  uint32_t t0 = max(a[0], b[3]), t1 = max(a[1], b[2]), t2 = max(a[2], b[1]), t3 = max(a[3], b[0]);
  cas_u32(t0, t2);
  cas_u32(t1, t3);
  cas_u32(t0, t1);
  cas_u32(t2, t3);
  o[0] = t0;
  o[1] = t1;
  o[2] = t2;
  o[3] = t3;
}
// this lane's 4 largest of 16 keys, descending
__device__ __forceinline__ void top4of16(const uint32_t (&k)[16], uint32_t (&c)[4]) {
  uint32_t q[4][4];
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    uint32_t a0 = k[4 * i], a1 = k[4 * i + 1], b0 = k[4 * i + 2], b1 = k[4 * i + 3];
    cas_u32(a0, a1);
    cas_u32(b0, b1);
    // merge two descending pairs into the descending 4
    uint32_t c0 = a0, m1 = b0, m2 = a1, c3 = b1;
    cas_u32(c0, m1);
    cas_u32(m2, c3);
    cas_u32(m1, m2);
    q[i][0] = c0;
    q[i][1] = m1;
    q[i][2] = m2;
    q[i][3] = c3;
  }
  uint32_t h0[4], h1[4];
  top4of8(q[0], q[1], h0);
  top4of8(q[2], q[3], h1);
  top4of8(h0, h1, c);
}
// warp_top10_redux by candidates: false when a lane ran dry (the caller takes the full path)
__device__ __forceinline__ bool warp_top10_fast(const uint32_t (&keys)[16], uint32_t lane, uint32_t& mine) {
  uint32_t c[4];
  top4of16(keys, c);
  mine = 0;
  bool dry = false;
#pragma unroll
  for (uint32_t k = 0; k < kTopK; ++k) {
    const uint32_t top = warp_max_u32(c[0]);
    mine = lane == k ? top : mine;
    if (k + 1 < kTopK) {
      const bool own = c[0] == top;
      c[0] = own ? c[1] : c[0];
      c[1] = own ? c[2] : c[1];
      c[2] = own ? c[3] : c[2];
      c[3] = own ? 0u : c[3];
    }
    if (k == kTopK - 2) dry = __any_sync(0xffffffffu, c[0] == 0u);
  }
  return !dry;
}
__device__ __forceinline__ void row_keys(const uint4& v0, const uint4& v1, uint32_t lane, uint32_t (&keys)[16]) {
#pragma unroll
  for (int i = 0; i < 8; ++i) {
    const uint32_t w = i < 4 ? (&v0.x)[i] : (&v1.x)[i - 4];
    keys[2 * i] = topk_key(w & 0xFFFFu, lane * 16 + 2 * i);
    keys[2 * i + 1] = topk_key(w >> 16, lane * 16 + 2 * i + 1);
  }
}
// route_row with the candidate top-10 (bit for bit route_row)
__device__ __forceinline__ void route_row_fast(
    const uint4& v0, const uint4& v1, uint32_t lane, uint32_t* s_cand, uint32_t& expert, uint16_t& wbits) {
  uint32_t keys[16];
  row_keys(v0, v1, lane, keys);
  uint32_t key;
  if (!warp_top10_fast(keys, lane, key)) key = warp_top10_redux(keys, lane, s_cand);
  uint32_t bits;
  topk_unkey(key, bits, expert);
  wbits = softmax_topk(static_cast<uint16_t>(bits), lane);
}

// ---- the routing tables
struct Tables {
  int32_t* counter;               // [1] the launch's CTA arrivals
  int32_t* total_padded;          // [1] live CTAs * tile
  int32_t* expanded_to_permuted;  // [M * 10], -1 for a slot outside the window
  int32_t* permuted_to_token;     // [max CTAs * tile + 1], -1 in a tile's padding
  int32_t* tokens_per_expert;     // [local_count]
  int32_t* cta_expert;            // [max CTAs] the CTA's local expert
  int32_t* cta_row_limit;         // [max CTAs]
  int32_t* live_ctas;             // [1]
  int32_t tile;                   // tile_n
  int32_t local_offset;           // the rank's first expert
  int32_t local_count;            // and its number of experts
};

constexpr uint16_t kRemote = 0xFFFFu;

// The slot's local expert, or kRemote outside the window.
__device__ __forceinline__ uint16_t local_slot(int32_t expert, const Tables& t) {
  const int32_t l = expert - t.local_offset;
  return (l >= 0 && l < t.local_count) ? static_cast<uint16_t>(l) : kRemote;
}

// The tables of n expanded slots (token i / 10) whose local experts are in s_local (visible to the CTA), by the whole
// CTA of kThreads: per-expert counts, the tile-padded offsets (local experts in order), the CTA -> local expert and
// CTA -> row-limit maps, the live CTA count and padded size, and the expanded <-> permuted maps. kTokenOrder (at most
// 32 tokens): an expert's rows in token order, as the single-block routing writes them; otherwise in the order of
// shared-memory atomics, as the cluster routing does. Scratch: s_count / s_next [512], s_warp [8].
template <bool kTokenOrder>
__device__ __forceinline__ void write_tables(
    const Tables& t, uint32_t n, const uint16_t* s_local, int32_t* s_count, int32_t* s_next, int32_t* s_warp) {
  constexpr uint32_t kWarps = kThreads / 32;
  static_assert(2 * kThreads == kExperts, "the scan takes two experts a thread");
  const uint32_t tid = threadIdx.x;
  const int32_t tile = t.tile;
  for (uint32_t e = tid; e < kExperts; e += kThreads) s_count[e] = 0;
  __syncthreads();
  for (uint32_t i = tid; i < n; i += kThreads) {
    const uint32_t l = s_local[i];
    if (l == kRemote) continue;
    if constexpr (kTokenOrder) {
      atomicOr(reinterpret_cast<uint32_t*>(s_count) + l, 1u << (i / kTopK));
    } else {
      atomicAdd(s_count + l, 1);
    }
  }
  __syncthreads();
  // exclusive scan of the experts' CTA counts, two experts a thread
  const uint32_t e0 = 2 * tid, lane = tid % 32, warp = tid / 32;
  const int32_t c0 = kTokenOrder ? __popc(s_count[e0]) : s_count[e0];
  const int32_t c1 = kTokenOrder ? __popc(s_count[e0 + 1]) : s_count[e0 + 1];
  const int32_t n0 = (c0 + tile - 1) / tile, n1 = (c1 + tile - 1) / tile;
  int32_t incl = n0 + n1;
#pragma unroll
  for (uint32_t d = 1; d < 32; d <<= 1) {
    const int32_t up = __shfl_up_sync(0xffffffffu, incl, d);
    incl += lane >= d ? up : 0;
  }
  if (lane == 31) s_warp[warp] = incl;
  __syncthreads();
  int32_t base = 0;
#pragma unroll
  for (uint32_t w = 0; w < kWarps; ++w) base += w < warp ? s_warp[w] : 0;
  const int32_t off0 = base + incl - n0 - n1, off1 = off0 + n0;
  const int32_t offs[2] = {off0, off1}, counts[2] = {c0, c1}, ctas[2] = {n0, n1};
#pragma unroll
  for (int j = 0; j < 2; ++j) {
    const int32_t e = static_cast<int32_t>(e0) + j, off = offs[j], count = counts[j];
    if (e < t.local_count) t.tokens_per_expert[e] = count;
    s_next[e] = off * tile;
    for (int32_t c = 0; c < ctas[j]; ++c) {
      t.cta_expert[off + c] = e;
      const int32_t limit = min((off + c + 1) * tile, off * tile + count);
      t.cta_row_limit[off + c] = limit;
      for (int32_t r = limit; r < (off + c + 1) * tile; ++r) t.permuted_to_token[r] = -1;
    }
  }
  if (tid == kThreads - 1) {
    t.live_ctas[0] = off1 + n1;
    t.total_padded[0] = (off1 + n1) * tile;
  }
  __syncthreads();
  for (uint32_t i = tid; i < n; i += kThreads) {
    const uint32_t l = s_local[i];
    if (l == kRemote) {
      t.expanded_to_permuted[i] = -1;
      continue;
    }
    const uint32_t token = i / kTopK;
    int32_t pos;
    if constexpr (kTokenOrder) {
      pos = s_next[l] + __popc(static_cast<uint32_t>(s_count[l]) & ((1u << token) - 1u));
    } else {
      pos = atomicAdd(s_next + l, 1);
    }
    t.expanded_to_permuted[i] = pos;
    t.permuted_to_token[pos] = static_cast<int32_t>(token);
  }
}

}  // namespace q35_route_tables
