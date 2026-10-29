#pragma once

// OP combine: merge the attention chunks, round to BF16 and quantize the o_proj input.

#include "include/sm100/common/sync.cuh"

namespace m31_draft {
namespace {

// Merge the chunks of one query row, round to BF16 (the attention output), and
// quantize it for o_proj. Thread (group cg, quad q) adds chunks cg, cg + 8, ...
// for dims 4q..4q+3; the eight group sums are then added in group order.
__device__ void combine_phase(Ctx &c, int layer) {
  int const kv_len = c.misc->kv_len;
  int const keys = kv_len < kWindow ? kv_len : kWindow;
  int const chunks = (keys + kChunk - 1) / kChunk;
  float *acc_s = reinterpret_cast<float *>(c.smem + kSmemCombAcc);
  bool waited = false;
  for (int row = first_combine_row(c.cta); row < kAttnRows; row += c.grid) {
    if (!waited)
      wait_signal(c, kSigAttn, layer, uint32_t(c.grid));
    else
      consumer_sync();
    waited = true;
    int const quad = c.tid & 31, cg = c.tid >> 5;
    float const *partial = at<float>(c.arena, kAttnPartial) + size_t(row) * kHeadDim + quad * 4;
    float4 pv[kMaxChunks / 8];
#pragma unroll
    for (int k = 0; k < kMaxChunks / 8; ++k) {
      int const ch = cg + 8 * k;
      pv[k] = ch < chunks ? __ldcg(reinterpret_cast<float4 const *>(
                                partial + size_t(ch) * kAttnRows * kHeadDim))
                          : make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    }
    float const *stats = at<float>(c.arena, kAttnStats);
    if (c.tid < chunks) {
      float2 const st =
          __ldcg(reinterpret_cast<float2 const *>(stats + (size_t(row) * kMaxChunks + c.tid) * 2));
      c.misc->comb_stat[c.tid][0] = st.x;
      c.misc->comb_stat[c.tid][1] = st.y;
    }
    consumer_sync();
    if (c.warp == 0) {
      float m = kNegInf;
      for (int ch = c.lane; ch < chunks; ch += 32)
        m = fmaxf(m, c.misc->comb_stat[ch][0]);
      for (int mask = 16; mask > 0; mask >>= 1)
        m = fmaxf(m, __shfl_xor_sync(0xffffffffu, m, mask));
      // The weighted sum of the chunk sums: lane-strided FP32 partial sums, then a fixed
      // butterfly (the same order every call), instead of one lane's serial chain.
      float l = 0.0f;
      for (int ch = c.lane; ch < chunks; ch += 32) {
        float const w = exp2f(c.misc->comb_stat[ch][0] - m);
        c.misc->comb_w[ch] = w;
        l = fmaf(c.misc->comb_stat[ch][1], w, l);
      }
      for (int mask = 16; mask > 0; mask >>= 1)
        l += __shfl_xor_sync(0xffffffffu, l, mask);
      if (c.lane == 0)
        c.misc->comb_l = l;
    }
    consumer_sync();
    float a4[4] = {0.0f, 0.0f, 0.0f, 0.0f};
#pragma unroll
    for (int k = 0; k < kMaxChunks / 8; ++k) {
      int const ch = cg + 8 * k;
      if (ch < chunks) {
        float const w = c.misc->comb_w[ch];
        a4[0] = fmaf(pv[k].x, w, a4[0]);
        a4[1] = fmaf(pv[k].y, w, a4[1]);
        a4[2] = fmaf(pv[k].z, w, a4[2]);
        a4[3] = fmaf(pv[k].w, w, a4[3]);
      }
    }
    *reinterpret_cast<float4 *>(acc_s + cg * kHeadDim + quad * 4) =
        make_float4(a4[0], a4[1], a4[2], a4[3]);
    consumer_sync();
    if (c.warp == 0) {
      int const q = c.lane; // dims 4q..4q+3; a 32-dim scale group is eight lanes
      float value[4], amax = 0.0f;
#pragma unroll
      for (int i = 0; i < 4; ++i) {
        float total = acc_s[4 * q + i];
#pragma unroll
        for (int k = 1; k < 8; ++k)
          total += acc_s[k * kHeadDim + 4 * q + i];
        value[i] = bf16r(total / c.misc->comb_l);
        amax = fmaxf(amax, fabsf(value[i]));
      }
      for (int mask = 1; mask < 8; mask <<= 1)
        amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, mask));
      float inv;
      int const scale = e8m0_from_amax(amax, &inv);
      int const t = row / kQHeads, head = row % kQHeads;
      uint64_t *ll = at<uint64_t>(c.arena, kAttnLL) + size_t(layer) * kAttnLLWords;
      uint64_t const tag = uint64_t(c.epoch) << 32;
      uint32_t const bytes = to_e4m3x2(value[0] * inv, value[1] * inv) |
                             (to_e4m3x2(value[2] * inv, value[3] * inv) << 16);
      st_relaxed_gpu_u64(ll + t * (kQSize / 4) + head * (kHeadDim / 4) + q, tag | bytes);
      if ((q & 7) == 0)
        st_relaxed_gpu_u64(ll + kAttnActWords + t * (kQSize / kScaleBlock) +
                               head * (kHeadDim / kScaleBlock) + q / 8,
                           tag | uint32_t(scale));
    }
    consumer_sync();
  }
}

} // namespace
} // namespace m31_draft
