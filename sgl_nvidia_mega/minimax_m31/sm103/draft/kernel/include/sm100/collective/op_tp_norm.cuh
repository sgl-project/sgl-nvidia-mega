#pragma once

// OP tp_norm: TP4 rank-ordered sum + residual + Gemma RMSNorm + MXFP8 publish.

#include "include/sm100/common/sync.cuh"

namespace m31_draft {
namespace {

// ---------------------------------------------------------------------------
// Row phase: TP4 sum + residual + Gemma RMSNorm, then MXFP8 quantization of
// the normed row (or the final output). The norm reads the FP32 sum; the
// residual stream keeps its BF16 rounding. CTA t < kRows owns token row t;
// thread i owns the 4-column pieces i + 256 m (m < 6): each piece is one 16-byte TP4 word pair per
// sender, so every poll instruction of a warp reads 512 consecutive bytes; a 32-column scale
// group is the pieces of eight consecutive lanes.
__device__ void norm_phase(Ctx &c, int stage, uint16_t const *weight, bool output) {
  if (!owns_row(c.cta))
    return;
  constexpr int kPieces = kHidden / 4 / kConsumerThreads; // 6
  int const t = c.cta;
  uint32_t const gen = sum_generation(c, stage);
  uint32_t const flag = gen + 1u;
  int const buf = int(gen & 1u);
  uint16_t *residual = at<uint16_t>(c.arena, kResidual) + size_t(t) * kHidden;
  // The norm weight and the residual row go to shared memory with cp.async while the TP4 words are
  // polled (no registers held across the poll); each thread reads back only its own pieces.
  uint2 *s_w = reinterpret_cast<uint2 *>(c.smem + kSmemNormW);
  uint2 *s_r = reinterpret_cast<uint2 *>(c.smem + kSmemNormRes);
#pragma unroll
  for (int m = 0; m < kPieces; ++m) {
    int const pc = c.tid + kConsumerThreads * m;
    cp_async_8(smem_u32(s_w + pc), reinterpret_cast<uint2 const *>(weight) + pc);
    if (stage > 0)
      cp_async_8(smem_u32(s_r + pc), reinterpret_cast<uint2 const *>(residual) + pc);
  }
  uint64_t raw[kPieces][kTp][2];
#pragma unroll
  for (int m = 0; m < kPieces; ++m) {
#pragma unroll
    for (int sender = 0; sender < kTp; ++sender) {
      uint64_t const *src = ll_row(c.arena, buf, sender, t) + (c.tid + kConsumerThreads * m) * 2;
      ld_ll2(src, raw[m][sender][0], raw[m][sender][1]);
    }
  }
  unsigned long long start = 0;
  for (;;) {
    bool ready = true;
#pragma unroll
    for (int m = 0; m < kPieces; ++m) {
#pragma unroll
      for (int sender = 0; sender < kTp; ++sender) {
        if (uint32_t(raw[m][sender][0] >> 32) != flag ||
            uint32_t(raw[m][sender][1] >> 32) != flag) {
          ready = false;
          uint64_t const *src =
              ll_row(c.arena, buf, sender, t) + (c.tid + kConsumerThreads * m) * 2;
          ld_ll2(src, raw[m][sender][0], raw[m][sender][1]);
        }
      }
    }
    if (ready)
      break;
    if (start == 0)
      start = globaltimer();
    else
      check_deadline(start);
  }
  cp_async_wait_all(); // this thread's own pieces (read back by the same thread)
  uint2 wv[kPieces];
  float x[kPieces][4];
  float ss = 0.0f;
#pragma unroll
  for (int m = 0; m < kPieces; ++m) {
    int const pc = c.tid + kConsumerThreads * m;
    wv[m] = s_w[pc];
    uint2 const rv = stage > 0 ? s_r[pc] : make_uint2(0u, 0u);
    uint32_t const res[2] = {rv.x, rv.y};
    uint32_t keep[2];
    // Columns 2w, 2w + 1 of the piece as FP32 pairs (per element as the scalar ops).
#pragma unroll
    for (int w = 0; w < 2; ++w) {
      float2 sum = bf2x2(uint32_t(raw[m][0][w]));
#pragma unroll
      for (int sender = 1; sender < kTp; ++sender)
        sum = __fadd2_rn(sum, bf2x2(uint32_t(raw[m][sender][w])));
      // The embedding (stage 0) starts the residual stream; every later stage adds to it.
      float2 value = bf2x2(f2bf2(sum.x, sum.y));
      if (stage != 0)
        value = __fadd2_rn(value, bf2x2(res[w]));
      x[m][2 * w] = value.x;
      x[m][2 * w + 1] = value.y;
      ss = fmaf(value.x, value.x, ss);
      ss = fmaf(value.y, value.y, ss);
      keep[w] = f2bf2(value.x, value.y);
    }
    if (!output)
      *reinterpret_cast<uint2 *>(residual + (c.tid + kConsumerThreads * m) * 4) =
          make_uint2(keep[0], keep[1]);
  }
  for (int mask = 16; mask > 0; mask >>= 1)
    ss += __shfl_xor_sync(0xffffffffu, ss, mask);
  if (c.lane == 0)
    c.misc->red[c.warp] = ss;
  consumer_sync();
  float total = 0.0f;
#pragma unroll
  for (int w = 0; w < kConsumerWarps; ++w)
    total += c.misc->red[w];
  float const factor = rsqrtf(total / float(kHidden) + c.a.eps);
  if (output) {
#pragma unroll
    for (int m = 0; m < kPieces; ++m) {
      uint32_t const wb[2] = {wv[m].x, wv[m].y};
      uint32_t o[2];
#pragma unroll
      for (int w = 0; w < 2; ++w) {
        float2 const p = __fmul2_rn(
            __fmul2_rn(make_float2(x[m][2 * w], x[m][2 * w + 1]), make_float2(factor, factor)),
            __fadd2_rn(make_float2(1.0f, 1.0f), bf2x2(wb[w])));
        o[w] = f2bf2(p.x, p.y);
      }
      *reinterpret_cast<uint2 *>(c.a.out + size_t(t) * kHidden +
                                 (c.tid + kConsumerThreads * m) * 4) = make_uint2(o[0], o[1]);
    }
    return;
  }
  // The quantized row goes to shared memory first, then out as LL6 words (6 bytes each).
  uint8_t *act = c.smem + kSmemNormRow;
  uint8_t *act_sf = act + kHidden;
#pragma unroll
  for (int m = 0; m < kPieces; ++m) {
    int const pc = c.tid + kConsumerThreads * m;
    uint32_t const wb[2] = {wv[m].x, wv[m].y};
    float y[4], amax = 0.0f;
    // (x * factor) * (1 + w), rounded to bf16, per element as the scalar ops.
#pragma unroll
    for (int w = 0; w < 2; ++w) {
      float2 const p = __fmul2_rn(
          __fmul2_rn(make_float2(x[m][2 * w], x[m][2 * w + 1]), make_float2(factor, factor)),
          __fadd2_rn(make_float2(1.0f, 1.0f), bf2x2(wb[w])));
      float2 const q = bf2x2(f2bf2(p.x, p.y));
      y[2 * w] = q.x;
      y[2 * w + 1] = q.y;
      amax = fmaxf(amax, fmaxf(fabsf(q.x), fabsf(q.y)));
    }
    // A 32-value scale group is eight consecutive pieces: eight consecutive lanes.
    amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, 1));
    amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, 2));
    amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, 4));
    float inv;
    int const scale = e8m0_from_amax(amax, &inv);
    *reinterpret_cast<uint32_t *>(act + pc * 4) =
        to_e4m3x2(y[0] * inv, y[1] * inv) | (to_e4m3x2(y[2] * inv, y[3] * inv) << 16);
    if ((c.tid & 7) == 0)
      act_sf[pc / 8] = uint8_t(scale);
  }
  consumer_sync();
  // Publish: the GEMM CTAs poll these words directly (no release, no counter).
  uint64_t *dst = act_ll(c.arena, stage) + size_t(t) * kActLLRowWords;
  uint32_t const tag = ll_tag(c.epoch);
  for (int i = c.tid; i < kActLLRowWords; i += kConsumerThreads) {
    // scales follow the bytes
    uint16_t const *src = reinterpret_cast<uint16_t const *>(act + 6 * i);
    st_relaxed_gpu_u64(
        dst + i,
        ll_word(uint64_t(src[0]) | (uint64_t(src[1]) << 16) | (uint64_t(src[2]) << 32), tag));
  }
  consumer_sync(); // the row buffer is reused by the next phase
}

} // namespace
} // namespace m31_draft
