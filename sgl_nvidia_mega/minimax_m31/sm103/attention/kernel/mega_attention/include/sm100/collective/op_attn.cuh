/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
// ATTN + COMBINE. CTA (row, slot) computes one selected page for the 16 local heads:
// S^T = QMUL4(K) . Q^T, P = exp2(s c - m c + log2 448) -> E4M3, l = sum of FP32 P,
// O = P . QMUL4(V), and publishes the unnormalized BF16(O * scale_v) with (m, l) as tagged
// records. COMBINE CTA (row, s) then merges head s of the row over the 16 splits in slot order
// and writes the MXFP8 output (MegaAttention numerics: sm100_m3_decode_attention_softmax.hpp,
// sm100_m3_decode_attention_output_epilogue.hpp).
#pragma once

#include "op_score.cuh"

namespace attn_ring::attn {

constexpr uint32_t kWarpMax = 0;    // f32 [8 warps][16 heads]
constexpr uint32_t kWarpSum = 512;  // f32 [8 warps][16 heads]
constexpr uint32_t kP = 2048;       // e4m3 [16 heads][128 keys], 16-byte chunks XOR head & 7
constexpr uint32_t kV = 4096;       // e4m3 [128 keys][128 dims], 16-byte chunks XOR key & 7
constexpr uint32_t kQStage = 24576; // uint2 [32 lanes][8]: Q fragments handed over by warp 0
static_assert(kQStage >= kV + 16384 && kQStage + 2048 <= kScratchBytes,
              "Q staging fits the scratch");
constexpr uint32_t kCombineW = 0; // f32 [2][16 slots]: (m, l) of the merged head
constexpr uint32_t kCombineO =
    20480; // uint4 [16 splits][16 dim groups]: polled split partials of one head
constexpr uint32_t kOBlock = 1024; // u32 [33]: this head's O image block (fragment order + scales)
static_assert(kCombineO >= kV + 16384 && kCombineO + 4096 <= kQStage,
              "combine staging is free during COMBINE");

struct KvInfo {
  int logical;
  int valid;
};

__device__ __forceinline__ uint8_t *rec_o(const Params &p, int row, int split, int head, int pair) {
  return p.arena + arena::kSplitORec +
         (((size_t(row) * kTopK + split) * kHeads + head) * (kHeadDim / 2) + pair) * 8;
}
__device__ __forceinline__ uint8_t *rec_stat(const Params &p, int row, int split, int head) {
  return p.arena + arena::kSplitStatRec + ((size_t(row) * kTopK + split) * kHeads + head) * 16;
}

// Logical page of (row, slot) when it is known without TopK: every slot of a row that sees
// at most 16 pages, and slot 15 (the local page) otherwise. Returns false when TopK decides.
__device__ __forceinline__ bool static_page(int kv, int row, int slot, int &logical) {
  const int visible = visible_pages(kv, row);
  if (visible <= kTopK) {
    logical = slot < visible ? slot : -1;
    return true;
  }
  if (slot == kTopK - 1) {
    logical = visible - 1;
    return true;
  }
  return false;
}

// V page (128 keys x 64 packed bytes + 8 scale bytes per key) -> E4M3 [key][128] in smem.
__device__ __forceinline__ void dequant_v(uint32_t slot, uint32_t v_smem, int tid) {
  const int key = tid >> 1, half = tid & 1;
  const uint32_t src = slot + kKvVData + key * 64 + half * 32;
  const uint4 lo = lds_128(src), hi = lds_128(src + 16);
  const uint2 sc = lds_64(slot + kKvVScale + key * 8);
  const uint32_t packed[8] = {lo.x, lo.y, lo.z, lo.w, hi.x, hi.y, hi.z, hi.w};
#pragma unroll
  for (int j = 0; j < 4; ++j) {
    const uint32_t s = broadcast_scale_byte(sc, 4 * half + j);
    uint4 out;
    out.x = qmul4(uint16_t(packed[2 * j]), s);
    out.y = qmul4(uint16_t(packed[2 * j] >> 16), s);
    out.z = qmul4(uint16_t(packed[2 * j + 1]), s);
    out.w = qmul4(uint16_t(packed[2 * j + 1] >> 16), s);
    const int chunk = 4 * half + j;
    sts_128(v_smem + key * 128 + (uint32_t(chunk ^ (key & 7)) << 4), out);
  }
}

// One split. q: Q[row] B fragments for heads 0..7 (q0) and 8..15 (q1).
__device__ __forceinline__ void split_compute(const Params &p, uint8_t *scratch, uint32_t slot,
                                              const KvInfo &info, int row, int split,
                                              const uint2 (&q0)[4], const uint2 (&q1)[4], int tid,
                                              int warp, int lane, uint32_t tag) {
  float *warp_max_s = reinterpret_cast<float *>(scratch + kWarpMax);
  float *warp_sum_s = reinterpret_cast<float *>(scratch + kWarpSum);
  const uint32_t p_smem = smem_u32(scratch + kP), v_smem = smem_u32(scratch + kV);
  const float c = p.softmax_log2_scale;
  dequant_v(slot, v_smem, tid);

  // S^T: warp w owns keys 16w .. 16w+15; acc[nt] covers heads 8nt .. 8nt+7.
  score::PageFrag f;
  score::load_page_frag(slot + kKvKData, warp * 16, lane, f);
  float acc[2][4] = {{0.f, 0.f, 0.f, 0.f}, {0.f, 0.f, 0.f, 0.f}};
  uint4 kfrag[4];
  score::page_a_frags(f, lane, kfrag);
  {
    // Explicit f16 HMMA pairs (the lowering of the e4m3 MMA) chained through the FP32 accumulator;
    // the K fragments are unpacked once for both query tiles.
    HalfB qh0[4], qh1[4];
#pragma unroll
    for (int kc = 0; kc < 4; ++kc) {
      qh0[kc] = half_b(q0[kc].x, q0[kc].y);
      qh1[kc] = half_b(q1[kc].x, q1[kc].y);
    }
#pragma unroll
    for (int kc = 0; kc < 4; ++kc) {
      qmma_h(acc[0], kfrag[kc], qh0[kc]);
      qmma_h(acc[1], kfrag[kc], qh1[kc]);
    }
  }
  const int key0 = warp * 16 + (lane >> 2), key1 = key0 + 8;
#pragma unroll
  for (int nt = 0; nt < 2; ++nt) {
    if (key0 >= info.valid)
      acc[nt][0] = acc[nt][1] = -3.402823466e+38f;
    if (key1 >= info.valid)
      acc[nt][2] = acc[nt][3] = -3.402823466e+38f;
  }
  // Per-head maximum: lanes with equal lane & 3 share heads.
  float mx[2][2];
#pragma unroll
  for (int nt = 0; nt < 2; ++nt) {
#pragma unroll
    for (int e = 0; e < 2; ++e) {
      float m = fmaxf(acc[nt][e], acc[nt][e + 2]);
#pragma unroll
      for (int o = 4; o <= 16; o <<= 1)
        m = fmaxf(m, __shfl_xor_sync(0xffffffffu, m, o));
      mx[nt][e] = m;
    }
  }
  if (lane < 4) {
#pragma unroll
    for (int nt = 0; nt < 2; ++nt) {
      warp_max_s[warp * kHeads + 8 * nt + 2 * lane] = mx[nt][0];
      warp_max_s[warp * kHeads + 8 * nt + 2 * lane + 1] = mx[nt][1];
    }
  }
  consumer_sync();
  float head_max[2][2];
#pragma unroll
  for (int nt = 0; nt < 2; ++nt) {
#pragma unroll
    for (int e = 0; e < 2; ++e) {
      const int h = 8 * nt + 2 * (lane & 3) + e;
      float m = warp_max_s[h];
#pragma unroll
      for (int w = 1; w < kConsumerWarps; ++w)
        m = fmaxf(m, warp_max_s[w * kHeads + h]);
      head_max[nt][e] = m;
    }
  }
  // P (FP32 for the sum, E4M3 for the PV MMA).
  float sum[2][2];
#pragma unroll
  for (int nt = 0; nt < 2; ++nt) {
#pragma unroll
    for (int e = 0; e < 2; ++e) {
      const float bias = __fmaf_rn(head_max[nt][e], -c, 8.807354922057604f);
      const float p0 = ex2_approx(__fmaf_rn(acc[nt][e], c, bias));
      const float p1 = ex2_approx(__fmaf_rn(acc[nt][e + 2], c, bias));
      const int h = 8 * nt + 2 * (lane & 3) + e;
      const uint16_t pk = f32x2_to_e4m3x2(p0, p1);
      const uint32_t rowbase = p_smem + h * 128;
      sts_u8(rowbase + (uint32_t(((key0 >> 4) ^ (h & 7)) << 4) | (key0 & 15)), pk & 0xffu);
      sts_u8(rowbase + (uint32_t(((key1 >> 4) ^ (h & 7)) << 4) | (key1 & 15)), pk >> 8);
      float s = __fadd_rn(p0, p1);
#pragma unroll
      for (int o = 4; o <= 16; o <<= 1)
        s = __fadd_rn(s, __shfl_xor_sync(0xffffffffu, s, o));
      sum[nt][e] = s;
    }
  }
  if (lane < 4) {
#pragma unroll
    for (int nt = 0; nt < 2; ++nt) {
      warp_sum_s[warp * kHeads + 8 * nt + 2 * lane] = sum[nt][0];
      warp_sum_s[warp * kHeads + 8 * nt + 2 * lane + 1] = sum[nt][1];
    }
  }
  consumer_sync(); // P image, V image and the per-warp sums are complete

  // O = P . V: warp w owns dims 16w .. 16w+15 (two n8 tiles), K = 128 keys.
  float o[2][4] = {{0.f, 0.f, 0.f, 0.f}, {0.f, 0.f, 0.f, 0.f}};
#pragma unroll
  for (int kc = 0; kc < 4; ++kc) {
    const int prow = (lane & 7) + 8 * ((lane >> 3) & 1);
    const int pchunk = 2 * kc + (lane >> 4);
    const uint4 a = ldsm_x4(p_smem + prow * 128 + (uint32_t(pchunk ^ (prow & 7)) << 4));
    const int key = 32 * kc + lane;
    const uint4 b = ldsm_32x16_trans_b8(v_smem + key * 128 + (uint32_t(warp ^ (key & 7)) << 4));
    qmma_h(o[0], a, half_b(b.x, b.z));
    qmma_h(o[1], a, half_b(b.y, b.w));
  }
  // Unnormalized BF16 partials and (m, l) as tagged records (value and tag in one atomic store).
  const int g = lane >> 2, t2 = (lane & 3) * 2;
#pragma unroll
  for (int nt = 0; nt < 2; ++nt) {
    const int d = warp * 16 + nt * 8 + t2;
    const __nv_bfloat162 lo = __floats2bfloat162_rn(__fmul_rn(o[nt][0], p.v_out_scale),
                                                    __fmul_rn(o[nt][1], p.v_out_scale));
    const __nv_bfloat162 hi = __floats2bfloat162_rn(__fmul_rn(o[nt][2], p.v_out_scale),
                                                    __fmul_rn(o[nt][3], p.v_out_scale));
    st_rec64(rec_o(p, row, split, g, d / 2), *reinterpret_cast<const uint32_t *>(&lo), tag);
    st_rec64(rec_o(p, row, split, g + 8, d / 2), *reinterpret_cast<const uint32_t *>(&hi), tag);
  }
  if (tid < kHeads) {
    float l = 0.f;
#pragma unroll
    for (int w = 0; w < kConsumerWarps; ++w)
      l = __fadd_rn(l, warp_sum_s[w * kHeads + tid]);
    float m = warp_max_s[tid];
#pragma unroll
    for (int w = 1; w < kConsumerWarps; ++w)
      m = fmaxf(m, warp_max_s[w * kHeads + tid]);
    st_rec(rec_stat(p, row, split, tid), __float_as_uint(m), __float_as_uint(l), 0u, tag);
  }
}

// An empty split publishes m = -FLT_MAX, l = 0 and zero partials (the COMBINE polls every record).
__device__ __forceinline__ void split_empty(const Params &p, int row, int split, int tid,
                                            uint32_t tag) {
  for (int i = tid; i < kHeads * (kHeadDim / 2); i += kConsumerThreads)
    st_rec64(rec_o(p, row, split, i / (kHeadDim / 2), i % (kHeadDim / 2)), 0u, tag);
  if (tid < kHeads)
    st_rec(rec_stat(p, row, split, tid), __float_as_uint(-3.402823466e+38f), 0u, 0u, tag);
}

// Head `head` of `row`: merge the 16 splits in slot order (FP32), then the MXFP8 output of the
// head. All 256 consumer threads call: thread (split s = tid / 16, dims 8q .. 8q + 7, q = tid % 16)
// polls its two 16-byte record pairs (and threads 0..15 the (m, l) record of split tid) until every
// one carries this call's tag, then stages them in shared memory for the merge.
template <bool kDeadline>
__device__ __forceinline__ void combine_head(const Params &p, const Deadline<kDeadline> &dl,
                                             uint8_t *scratch, int row, int head, uint32_t tag,
                                             int tid) {
  const float c = p.softmax_log2_scale;
  float *stats = reinterpret_cast<float *>(scratch + kCombineW); // m [16], l [16]
  uint4 *staged = reinterpret_cast<uint4 *>(scratch + kCombineO);
  {
    const int s = tid >> 4, q = tid & 15;
    const uint8_t *src = rec_o(p, row, s, head, 4 * q);
    const uint8_t *st = rec_stat(p, row, tid & 15, head);
    uint4 a = make_uint4(0u, 0u, 0u, 0u), b = a, m = a;
    uint32_t pending = tid < kTopK ? 7u : 3u;
    while (pending) {
      if (pending & 1u)
        a = ld_rec(src);
      if (pending & 2u)
        b = ld_rec(src + 16);
      if (pending & 4u)
        m = ld_rec(st);
      if ((pending & 1u) && a.y == tag && a.w == tag)
        pending &= ~1u;
      if ((pending & 2u) && b.y == tag && b.w == tag)
        pending &= ~2u;
      if ((pending & 4u) && m.w == tag)
        pending &= ~4u;
      if (pending)
        dl.check(p, 0x48u + uint32_t(row), pending, tag);
    }
    staged[s * 16 + q] = make_uint4(a.x, a.z, b.x, b.z);
    if (tid < kTopK) {
      stats[tid] = __uint_as_float(m.x);
      stats[kTopK + tid] = __uint_as_float(m.y);
    }
  }
  consumer_sync();
  // The merge: thread d < 128 owns head dim d (one MXFP8 group of 32 dims per warp); per dim the
  // arithmetic and the slot order are those of the 8-dims-per-thread merge.
  if (tid >= kHeadDim)
    return;
  const int d = tid, lane = tid & 31;
  const float *m = stats, *l = stats + kTopK;
  // M over the live splits (max is exact in any order); lane s < 16 evaluates split s's weight once
  // and the others receive it by shuffle; L accumulates in split order as before.
  float M = -3.402823466e+38f;
#pragma unroll
  for (int s = 0; s < kTopK; ++s)
    if (l[s] > 0.f)
      M = fmaxf(M, m[s]);
  const int sl = lane & (kTopK - 1);
  const float w_mine = l[sl] > 0.f ? ex2_approx(__fmul_rn(c, __fsub_rn(m[sl], M))) : 0.f;
  float L = 0.f, w[kTopK];
#pragma unroll
  for (int s = 0; s < kTopK; ++s) {
    w[s] = __shfl_sync(0xffffffffu, w_mine, s);
    L = l[s] > 0.f ? __fmaf_rn(l[s], w[s], L) : L;
  }
  const float inv_l = L > 0.f ? __frcp_rn(L) : 0.f;
  const uint32_t *words = reinterpret_cast<const uint32_t *>(staged); // [split][16 groups][4 words]
  const int widx = (d >> 3) * 4 + ((d & 7) >> 1);
  float acc = 0.f;
#pragma unroll
  for (int s = 0; s < kTopK; ++s) {
    const uint32_t wd = words[s * 64 + widx];
    const float o = (d & 1) ? bf16_hi(wd) : bf16_lo(wd);
    acc = l[s] > 0.f ? __fmaf_rn(o, w[s], acc) : acc;
  }
  acc = __fmul_rn(acc, inv_l);
  float amax;
  asm("redux.sync.max.abs.f32 %0, %1, 0xffffffff;" : "=f"(amax) : "f"(acc));
  uint32_t code;
  float inv;
  attn_out_scale(amax, code, inv);
  // Four consecutive dims per E4M3 word (lanes 4k .. 4k + 3 -> lane 4k).
  const uint32_t q = uint32_t(f32x2_to_e4m3x2(__fmul_rn(acc, inv), 0.f) & 0xffu);
  const uint32_t q1 = __shfl_down_sync(0xffffffffu, q, 1), q2 = __shfl_down_sync(0xffffffffu, q, 2),
                 q3 = __shfl_down_sync(0xffffffffu, q, 3);
  const uint32_t word = q | (q1 << 8) | (q2 << 16) | (q3 << 24);
  uint32_t *block = reinterpret_cast<uint32_t *>(scratch + kOBlock);
  if ((d & 3) == 0) {
    *reinterpret_cast<uint32_t *>(p.o_out + (size_t(row) * kHeads + head) * kHeadDim + d) = word;
    // Fragment order inside the head's block (groups 4 head .. 4 head + 3): K 32 g + 16 half + 4
    // sub
    // .. + 3 is word g * 8 + 2 sub + half.
    const int g = d >> 5, k = d & 31;
    block[g * 8 + 2 * ((k & 15) >> 2) + (k >> 4)] = word;
  }
  if (lane == 0) {
    p.o_sf_out[head * 512 + row * 16 + (d >> 5)] = uint8_t(code);
    reinterpret_cast<uint8_t *>(block + 32)[d >> 5] = uint8_t(code);
  }
  asm volatile("bar.sync 6, %0;" ::"n"(kHeadDim) : "memory"); // the four merge warps
  if (d < arena::kORecsPerBlock)
    st_rec(p.arena + arena::kORec +
               ((size_t(row) * kHeads + head) * arena::kORecsPerBlock + d) * 16,
           block[3 * d], block[3 * d + 1], block[3 * d + 2], tag);
}

} // namespace attn_ring::attn
