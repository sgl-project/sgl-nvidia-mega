/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
// OPROJ + FINAL. Attention CTAs multiply three 16-channel Wo tiles by the MXFP8 attention
// output and publish BF16 rows into the TP4 multicast mailbox shared with MegaAttention
// (common/kernel/include/tp4_mailbox_layout.hpp: 0x8000 empty sentinel, three generations,
// rank-local publish / consume generation words). One FINAL CTA per token sums the four ranks in
// rank order, adds the residual and applies the post-attention RMSNorm.
#pragma once

#include "op_proj.cuh"

namespace attn_ring::oproj {

constexpr uint32_t kOut = 0;         // bf16 [3 tiles][8 tokens][16 channels]
constexpr uint32_t kWarpSq = 1024;   // f32 [8 warps]
constexpr uint32_t kFold = 2048;     // f32 [3 tiles][8 warps][16 rows][8 tokens]
constexpr uint32_t kPayload = 16384; // e4m3 [8][2048] fragment order, staged from the O records
constexpr uint32_t kScale = kPayload + kTokens * kAttnK; // u8 [8][64]
static_assert(kFold + kWoTilesPerCta * kConsumerWarps * kTileRows * kTokens * 4 <= kPayload &&
                  kScale + kTokens * (kAttnK / 32) <= kScratchBytes,
              "OPROJ scratch regions are disjoint and fit");
constexpr uint32_t kEmptyPair = (uint32_t(mailbox::kEmpty) << 16) | mailbox::kEmpty;

__device__ __forceinline__ uint32_t *publish_words(const Params &p) {
  return reinterpret_cast<uint32_t *>(p.tp_unicast + p.tp_publish_offset);
}
__device__ __forceinline__ uint32_t *consume_words(const Params &p) {
  return reinterpret_cast<uint32_t *>(p.tp_unicast + p.tp_consume_offset);
}
__device__ __forceinline__ uint64_t mailbox_offset(const Params &p, int stage, int rank, int token,
                                                   int channel) {
  return p.tp_payload_offset +
         uint64_t((stage * kWorld + rank) * (kTokens * kHidden) + token * kHidden + channel) * 2u;
}
__device__ __forceinline__ uint32_t sanitize(uint32_t w) {
  if ((w & 0xffffu) == mailbox::kEmpty)
    w &= 0xffff0000u;
  if ((w >> 16) == mailbox::kEmpty)
    w &= 0x0000ffffu;
  return w;
}
__device__ __forceinline__ bool incomplete(uint32_t w) {
  return (w & 0xffffu) == mailbox::kEmpty || (w >> 16) == mailbox::kEmpty;
}
__device__ __forceinline__ void multicast_store(uint64_t address, uint4 v) {
  asm volatile("multimem.st.relaxed.sys.global.v4.bf16x2 [%0], {%1,%2,%3,%4};" ::"l"(address),
               "r"(sanitize(v.x)), "r"(sanitize(v.y)), "r"(sanitize(v.z)), "r"(sanitize(v.w))
               : "memory");
}
__device__ __forceinline__ uint4 ld_volatile_128(uint64_t address) {
  uint4 v;
  asm volatile("ld.volatile.global.v4.b32 {%0,%1,%2,%3}, [%4];"
               : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w)
               : "l"(address)
               : "memory");
  return v;
}

// Per-warp partials of tile t: fold[t][warp][row][token].
__device__ __forceinline__ void stage_tile(uint8_t *scratch, int t, const float (&run)[4], int warp,
                                           int lane) {
  constexpr int kTileValues = kTileRows * kTokens;
  float *w = reinterpret_cast<float *>(scratch + kFold) + (t * kConsumerWarps + warp) * kTileValues;
  const int g = lane >> 2, t2 = (lane & 3) * 2;
  w[g * kTokens + t2] = run[0];
  w[g * kTokens + t2 + 1] = run[1];
  w[(g + 8) * kTokens + t2] = run[2];
  w[(g + 8) * kTokens + t2 + 1] = run[3];
}

// Fold the three tiles' per-warp partials in warp order, round to BF16 and publish the
// 48 output vectors (3 tiles x 8 tokens x 2 halves of 8 channels) with one barrier pair.
// stage[g]: publish generation of 128-channel group g of this CTA's channels (first, last).
__device__ __forceinline__ void publish_tiles(const Params &p, uint8_t *scratch, int cta,
                                              const int (&stage)[2], int tid) {
  float *fold = reinterpret_cast<float *>(scratch + kFold);
  __nv_bfloat16 *out = reinterpret_cast<__nv_bfloat16 *>(scratch + kOut);
  constexpr int kTileValues = kTileRows * kTokens;
  consumer_sync();
  for (int v = tid; v < kWoTilesPerCta * kTileValues; v += kConsumerThreads) {
    const int t = v / kTileValues, idx = v % kTileValues; // idx = row * 8 + token
    float sum = 0.f;
#pragma unroll
    for (int k = 0; k < kConsumerWarps; ++k)
      sum = __fadd_rn(sum, fold[(t * kConsumerWarps + k) * kTileValues + idx]);
    out[(t * kTokens + idx % kTokens) * kTileRows + idx / kTokens] = __float2bfloat16_rn(sum);
  }
  consumer_sync();
  if (tid < kWoTilesPerCta * kTokens * 2) {
    const int t = tid / (kTokens * 2), token = (tid / 2) % kTokens, half = tid & 1;
    const int tile = cta * kWoTilesPerCta + t;
    const int channel = tile * kTileRows + half * 8;
    const int first_group = (cta * kWoTilesPerCta * kTileRows) / mailbox::kOutputTileChannels;
    const int st = stage[channel / mailbox::kOutputTileChannels == first_group ? 0 : 1];
    const uint4 value =
        *reinterpret_cast<const uint4 *>(out + (t * kTokens + token) * kTileRows + half * 8);
    multicast_store(reinterpret_cast<uint64_t>(p.tp_multicast) +
                        mailbox_offset(p, st, p.tp_rank, token, channel),
                    value);
  }
}

// ---- FINAL ---------------------------------------------------------------------------
template <bool kDeadline>
__device__ __forceinline__ void final_token(const Params &p, const Deadline<kDeadline> &dl,
                                            uint8_t *scratch, int token, int stage, int tid,
                                            int warp, int lane) {
  float *warp_sq = reinterpret_cast<float *>(scratch + kWarpSq);
  constexpr int kVec = kHidden / 8;             // 768 vectors per token
  constexpr int kPer = kVec / kConsumerThreads; // 3
  const uint64_t unicast = reinterpret_cast<uint64_t>(p.tp_unicast);
  // Inputs first (they do not depend on any rank), then poll all 3 x 4 mailbox vectors at
  // once so a slow peer costs one wait, not one per vector.
  uint4 res[kPer], gam[kPer], peer[kPer][kWorld];
#pragma unroll
  for (int i = 0; i < kPer; ++i) {
    const int channel = (tid + i * kConsumerThreads) * 8;
    res[i] = ldg_nc_128(p.layer_input + size_t(token) * kHidden + channel);
    gam[i] = ldg_nc_128(p.post_norm + channel);
  }
  uint32_t pending = (1u << (kPer * kWorld)) - 1u;
  RemoteWait wait;
  while (pending) {
#pragma unroll
    for (int i = 0; i < kPer; ++i)
#pragma unroll
      for (int r = 0; r < kWorld; ++r)
        if (pending & (1u << (i * kWorld + r)))
          peer[i][r] = ld_volatile_128(
              unicast + mailbox_offset(p, stage, r, token, (tid + i * kConsumerThreads) * 8));
#pragma unroll
    for (int i = 0; i < kPer; ++i)
#pragma unroll
      for (int r = 0; r < kWorld; ++r) {
        const uint4 v = peer[i][r];
        if (!(incomplete(v.x) || incomplete(v.y) || incomplete(v.z) || incomplete(v.w)))
          pending &= ~(1u << (i * kWorld + r));
      }
    if (pending) {
      dl.check(p, 0x600u + token, uint64_t(pending), uint64_t(stage));
      wait.poll();
    }
  }
  // Clear only the local copies (the data is in registers): issued first so these stores drain
  // under the residual / norm arithmetic; three generations separate reuse from a peer's publish,
  // and the consume word advances only after the final barrier.
#pragma unroll
  for (int i = 0; i < kPer; ++i) {
    const int channel = (tid + i * kConsumerThreads) * 8;
#pragma unroll
    for (int r = 0; r < kWorld; ++r)
      asm volatile("st.volatile.global.v4.u32 [%0], {%1,%1,%1,%1};" ::"l"(
                       unicast + mailbox_offset(p, stage, r, token, channel)),
                   "r"(kEmptyPair)
                   : "memory");
  }
  float x[kPer][8];
  float sq = 0.f;
#pragma unroll
  for (int i = 0; i < kPer; ++i) {
    const int channel = (tid + i * kConsumerThreads) * 8;
    const uint32_t rw[4] = {res[i].x, res[i].y, res[i].z, res[i].w};
    uint32_t outw[4];
#pragma unroll
    for (int j = 0; j < 4; ++j) {
      float lo = 0.f, hi = 0.f;
#pragma unroll
      for (int r = 0; r < kWorld; ++r) {
        const uint32_t w = j == 0   ? peer[i][r].x
                           : j == 1 ? peer[i][r].y
                           : j == 2 ? peer[i][r].z
                                    : peer[i][r].w;
        lo = __fadd_rn(lo, bf16_lo(w));
        hi = __fadd_rn(hi, bf16_hi(w));
      }
      const __nv_bfloat162 pre =
          __floats2bfloat162_rn(__fadd_rn(lo, bf16_lo(rw[j])), __fadd_rn(hi, bf16_hi(rw[j])));
      outw[j] = *reinterpret_cast<const uint32_t *>(&pre);
      x[i][2 * j] = bf16_lo(outw[j]);
      x[i][2 * j + 1] = bf16_hi(outw[j]);
      sq = __fmaf_rn(x[i][2 * j], x[i][2 * j], sq);
      sq = __fmaf_rn(x[i][2 * j + 1], x[i][2 * j + 1], sq);
    }
    *reinterpret_cast<uint4 *>(p.attention_residual + size_t(token) * kHidden + channel) =
        make_uint4(outw[0], outw[1], outw[2], outw[3]);
  }
  sq = warp_sum(sq);
  if (lane == 0)
    warp_sq[warp] = sq;
  consumer_sync();
  float total = 0.f;
#pragma unroll
  for (int w = 0; w < kConsumerWarps; ++w)
    total = __fadd_rn(total, warp_sq[w]);
  const float inv = rsqrtf(__fadd_rn(__fdiv_rn(total, float(kHidden)), p.post_eps));
#pragma unroll
  for (int i = 0; i < kPer; ++i) {
    const int channel = (tid + i * kConsumerThreads) * 8;
    const uint32_t gw[4] = {gam[i].x, gam[i].y, gam[i].z, gam[i].w};
    uint32_t outw[4];
#pragma unroll
    for (int j = 0; j < 4; ++j) {
      const __nv_bfloat162 n = __floats2bfloat162_rn(
          __fmul_rn(__fmul_rn(x[i][2 * j], inv), __fadd_rn(1.f, bf16_lo(gw[j]))),
          __fmul_rn(__fmul_rn(x[i][2 * j + 1], inv), __fadd_rn(1.f, bf16_hi(gw[j]))));
      outw[j] = *reinterpret_cast<const uint32_t *>(&n);
    }
    *reinterpret_cast<uint4 *>(p.post_attention_norm + size_t(token) * kHidden + channel) =
        make_uint4(outw[0], outw[1], outw[2], outw[3]);
  }
  consumer_sync();
  if (tid < mailbox::kOwnersPerToken)
    consume_words(p)[token * mailbox::kOwnersPerToken + tid] =
        uint32_t((stage + 1) % mailbox::kGenerations);
}

} // namespace attn_ring::oproj
