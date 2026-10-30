/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
// NORM + PROJ + QKFIN: input RMSNorm and MXFP8 quantization of (token, K slice) published as
// tagged records, the QKV / QI / KI MXFP8 GEMV over 16-row x 2048-K tiles with split-K partial
// records, and the per-head finish (head RMSNorm, NeoX RoPE, E4M3 Q / QI, NVFP4 K / V / KI rows).
#pragma once

#include "../common/ring_math.cuh"
#include "../common/ring_pipe.cuh"

namespace attn_ring::proj {

// Consumer scratch layout (bytes).
constexpr uint32_t kSumsq = 0;   // f32 [8 warps]
constexpr uint32_t kFold = 1024; // f32 [8 warps][16 rows][8 tokens]

// ---- NORM ----------------------------------------------------------------------------
// CTA (token t, slice s): full-row RMS statistics of token t, then input RMSNorm (gamma form
// x * (1 + g), BF16 rounding) and dynamic MXFP8 of K slice s. The E4M3 bytes are stored in
// m16n8k32 B-fragment order: ((t * 192 + G) * 4 + sub) * 8 = {K 32G+4sub .. +3, 32G+16+4sub .. +3}.
// The slice's 6 blocks of 4 K32 groups are staged in shared memory (block-local word (G % 4) * 8 +
// 2 (part % 4) + part / 4, scales in word 32) and published as 11 tagged records per block.
constexpr uint32_t kNormStage = 1024; // u32 [6 blocks][33]
__device__ __forceinline__ void norm_slice(const Params &p, uint8_t *scratch, int token, int slice,
                                           int tid, int warp, int lane, uint32_t *tag_slot,
                                           uint32_t tag_in) {
  float *sumsq = reinterpret_cast<float *>(scratch + kSumsq);
  const uint4 *row = reinterpret_cast<const uint4 *>(p.layer_input + size_t(token) * kHidden);
  constexpr int kPer = kHidden / 8 / kConsumerThreads; // 3 vectors per thread
  uint4 v[kPer];
#pragma unroll
  for (int i = 0; i < kPer; ++i)
    v[i] = __ldg(row + tid + i * kConsumerThreads);
  // This thread's quantization group: 24 groups of 32 per slice, 8 threads x 4 values each.
  const int group_local = tid >> 3, part = tid & 7; // groups 0..31; only < 24 are real
  const bool quantizer = group_local < kNormSliceK / 32;
  const int k = slice * kNormSliceK + group_local * 32 + part * 4;
  uint2 xv = make_uint2(0u, 0u), gv = make_uint2(0u, 0u);
  if (quantizer) {
    xv = __ldg(reinterpret_cast<const uint2 *>(p.layer_input + size_t(token) * kHidden + k));
    gv = __ldg(reinterpret_cast<const uint2 *>(p.input_norm + k));
  }
  float acc = 0.f;
#pragma unroll
  for (int i = 0; i < kPer; ++i) {
    const uint32_t w[4] = {v[i].x, v[i].y, v[i].z, v[i].w};
#pragma unroll
    for (int j = 0; j < 4; ++j) {
      const float lo = bf16_lo(w[j]), hi = bf16_hi(w[j]);
      acc = __fmaf_rn(lo, lo, acc);
      acc = __fmaf_rn(hi, hi, acc);
    }
  }
  acc = warp_sum(acc);
  if (lane == 0)
    sumsq[warp] = acc;
  if (tid == 0)
    *tag_slot = tag_in; // the epoch, loaded by thread 0 at entry
  consumer_sync();
  const uint32_t tag = *tag_slot;
  float total = 0.f;
#pragma unroll
  for (int w = 0; w < kConsumerWarps; ++w)
    total = __fadd_rn(total, sumsq[w]);
  const float r = __frcp_rn(__fsqrt_rn(__fadd_rn(__fdiv_rn(total, float(kHidden)), p.input_eps)));
  if (quantizer) {
    const uint32_t xw[2] = {xv.x, xv.y}, gw[2] = {gv.x, gv.y};
    float y[4];
#pragma unroll
    for (int j = 0; j < 2; ++j) {
      y[2 * j] =
          bf16_round(__fmul_rn(__fmul_rn(bf16_lo(xw[j]), r), __fadd_rn(1.f, bf16_lo(gw[j]))));
      y[2 * j + 1] =
          bf16_round(__fmul_rn(__fmul_rn(bf16_hi(xw[j]), r), __fadd_rn(1.f, bf16_hi(gw[j]))));
    }
    float amax = fmaxf(fmaxf(fabsf(y[0]), fabsf(y[1])), fmaxf(fabsf(y[2]), fabsf(y[3])));
#pragma unroll
    for (int o = 1; o < 8; o <<= 1)
      amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, o));
    const uint32_t code = ue8m0_round_up(__fdiv_rn(amax, 448.f));
    const float inv = ue8m0_inverse(code);
    const uint32_t q = f32x4_to_e4m3x4(__fmul_rn(y[0], inv), __fmul_rn(y[1], inv),
                                       __fmul_rn(y[2], inv), __fmul_rn(y[3], inv));
    // part p holds K bytes 4p .. 4p+3 of the group: sub = p % 4, half = p / 4.
    uint32_t *stage = reinterpret_cast<uint32_t *>(scratch + kNormStage) + (group_local / 4) * 33;
    stage[(group_local % 4) * 8 + (part & 3) * 2 + (part >> 2)] = q;
    if (part == 0)
      reinterpret_cast<uint8_t *>(stage + 32)[group_local % 4] = uint8_t(code);
  }
  consumer_sync();
  constexpr int kBlocks = kNormSliceK / 128; // 6
  if (tid < kBlocks * arena::kORecsPerBlock) {
    const int b = tid / arena::kORecsPerBlock, j = tid % arena::kORecsPerBlock;
    const uint32_t *w = reinterpret_cast<const uint32_t *>(scratch + kNormStage) + b * 33 + 3 * j;
    st_rec(
        p.arena + arena::kXqRec +
            ((size_t(token) * arena::kXqBlocks + slice * kBlocks + b) * arena::kORecsPerBlock + j) *
                16,
        w[0], w[1], w[2], tag);
  }
}

// MXFP8 B fragments of this warp's 256-wide K slice of segment `seg`, plus the UE8M0 scales
// of output tokens 2s and 2s+1 (s = lane % 4), from a fragment-ordered activation image.
struct BFrag {
  uint2 b[kGroupsPerWarp];
  uint2 scale_lo, scale_hi;
};

// Load them from a shared-memory copy of the payload / scales (fragment order).
__device__ __forceinline__ void smem_bfrag(const uint8_t *payload, const uint8_t *scales,
                                           int groups_per_row, int first_group, int lane,
                                           BFrag &f) {
  const int t = lane >> 2, s = lane & 3;
#pragma unroll
  for (int g = 0; g < kGroupsPerWarp; ++g)
    f.b[g] = *reinterpret_cast<const uint2 *>(
        payload + (size_t(t * groups_per_row + first_group + g) * 4 + s) * 8);
  f.scale_lo = *reinterpret_cast<const uint2 *>(scales + (2 * s) * groups_per_row + first_group);
  f.scale_hi =
      *reinterpret_cast<const uint2 *>(scales + (2 * s + 1) * groups_per_row + first_group);
}

// UE8M0 weight scales of rows m0 = 16 * tile + lane / 4 and m0 + 8 for this warp's two
// 128-K chunks, read from the 128x4-swizzled scale layout (one u32 = four K32 groups).
__device__ __forceinline__ void load_weight_scales(const uint8_t *sf, int k_row_bytes, int tile,
                                                   int k_first_chunk, int lane,
                                                   uint32_t (&w)[2][2]) {
  const int block_bytes = (k_row_bytes / 128) * 512; // one 128-row block
#pragma unroll
  for (int h = 0; h < 2; ++h) {
    const int m = tile * kTileRows + (lane >> 2) + 8 * h;
#pragma unroll
    for (int c = 0; c < 2; ++c) {
      const size_t off = size_t(m >> 7) * block_bytes + size_t(k_first_chunk + c) * 512 +
                         (m & 31) * 16 + ((m & 127) >> 5) * 4;
      w[h][c] = ldg_nc_u32(sf + off); // issued here even when the first use follows a poll loop
    }
  }
}

// The B fragments of a CTA's activation unpacked to f16 once (they are the same for every tile
// of the CTA), with the activation UE8M0 scales of tokens 2s / 2s+1 as floats.
struct BFragH {
  HalfB b[kGroupsPerWarp];
  float sb0[kGroupsPerWarp], sb1[kGroupsPerWarp];
};
__device__ __forceinline__ void unpack_bfrag(const BFrag &f, BFragH &h) {
#pragma unroll
  for (int g = 0; g < kGroupsPerWarp; ++g)
    h.b[g] = half_b(f.b[g].x, f.b[g].y);
#pragma unroll
  for (int g = 0; g < 4; ++g) {
    const uint32_t lo[2] = {f.scale_lo.x, f.scale_lo.y}, hi[2] = {f.scale_hi.x, f.scale_hi.y};
#pragma unroll
    for (int c = 0; c < 2; ++c) {
      const int gg = 4 * c + g;
      h.sb0[gg] = g == 0   ? ue8m0_byte_to_f32<0>(lo[c])
                  : g == 1 ? ue8m0_byte_to_f32<1>(lo[c])
                  : g == 2 ? ue8m0_byte_to_f32<2>(lo[c])
                           : ue8m0_byte_to_f32<3>(lo[c]);
      h.sb1[gg] = g == 0   ? ue8m0_byte_to_f32<0>(hi[c])
                  : g == 1 ? ue8m0_byte_to_f32<1>(hi[c])
                  : g == 2 ? ue8m0_byte_to_f32<2>(hi[c])
                           : ue8m0_byte_to_f32<3>(hi[c]);
    }
  }
}
template <int k> __device__ __forceinline__ float sa_of(uint32_t w) {
  return ue8m0_byte_to_f32<k>(w);
}

// One 16-row x 2048-K tile (this warp's eight K32 groups) with the unpacked activation: per K32
// group the two HMMAs of the ptxas lowering of the e4m3 MMA (minus its zero-add), then RN(sa * sb)
// and one FMA into the FP32 run. run[0] = (row g, token 2s), run[1] = (g, 2s + 1), run[2] =
// (g + 8, 2s), run[3] = (g + 8, 2s + 1) with g = lane / 4, s = lane % 4.
__device__ __forceinline__ void tile_gemv_h(uint32_t tile_smem, const uint32_t (&sfw)[2][2],
                                            const BFragH &f, int warp, int lane, float (&run)[4]) {
  run[0] = run[1] = run[2] = run[3] = 0.f;
#pragma unroll
  for (int g = 0; g < kGroupsPerWarp; ++g) {
    const uint4 a = weight_a_frag(tile_smem, warp * 2 + g / 4, g % 4, lane);
    float d[4] = {0.f, 0.f, 0.f, 0.f};
    qmma_h(d, a, f.b[g]);
    const uint32_t w0 = sfw[0][g / 4], w1 = sfw[1][g / 4];
    const float sa0 = (g % 4) == 0   ? sa_of<0>(w0)
                      : (g % 4) == 1 ? sa_of<1>(w0)
                      : (g % 4) == 2 ? sa_of<2>(w0)
                                     : sa_of<3>(w0);
    const float sa1 = (g % 4) == 0   ? sa_of<0>(w1)
                      : (g % 4) == 1 ? sa_of<1>(w1)
                      : (g % 4) == 2 ? sa_of<2>(w1)
                                     : sa_of<3>(w1);
    run[0] = __fmaf_rn(d[0], __fmul_rn(sa0, f.sb0[g]), run[0]);
    run[2] = __fmaf_rn(d[2], __fmul_rn(sa1, f.sb0[g]), run[2]);
    run[1] = __fmaf_rn(d[1], __fmul_rn(sa0, f.sb1[g]), run[1]);
    run[3] = __fmaf_rn(d[3], __fmul_rn(sa1, f.sb1[g]), run[3]);
  }
}

// T tiles in one interleaved pass (independent HMMA / scale chains per tile for latency hiding);
// per tile the arithmetic and its order are exactly those of tile_gemv_h.
template <int T>
__device__ __forceinline__ void tiles_gemv_h(const uint32_t (&slot)[T],
                                             const uint32_t (&sfw)[T][2][2], const BFragH &f,
                                             int warp, int lane, float (&run)[T][4]) {
#pragma unroll
  for (int t = 0; t < T; ++t)
    run[t][0] = run[t][1] = run[t][2] = run[t][3] = 0.f;
#pragma unroll
  for (int g = 0; g < kGroupsPerWarp; ++g) {
    uint4 a[T];
#pragma unroll
    for (int t = 0; t < T; ++t)
      a[t] = weight_a_frag(slot[t], warp * 2 + g / 4, g % 4, lane);
#pragma unroll
    for (int t = 0; t < T; ++t) {
      float d[4] = {0.f, 0.f, 0.f, 0.f};
      qmma_h(d, a[t], f.b[g]);
      const uint32_t w0 = sfw[t][0][g / 4], w1 = sfw[t][1][g / 4];
      const float sa0 = (g % 4) == 0   ? sa_of<0>(w0)
                        : (g % 4) == 1 ? sa_of<1>(w0)
                        : (g % 4) == 2 ? sa_of<2>(w0)
                                       : sa_of<3>(w0);
      const float sa1 = (g % 4) == 0   ? sa_of<0>(w1)
                        : (g % 4) == 1 ? sa_of<1>(w1)
                        : (g % 4) == 2 ? sa_of<2>(w1)
                                       : sa_of<3>(w1);
      run[t][0] = __fmaf_rn(d[0], __fmul_rn(sa0, f.sb0[g]), run[t][0]);
      run[t][2] = __fmaf_rn(d[2], __fmul_rn(sa1, f.sb0[g]), run[t][2]);
      run[t][1] = __fmaf_rn(d[1], __fmul_rn(sa0, f.sb1[g]), run[t][1]);
      run[t][3] = __fmaf_rn(d[3], __fmul_rn(sa1, f.sb1[g]), run[t][3]);
    }
  }
}

// This warp's A fragments of a tile unpacked to f16 into a private shared-memory image (8 KB per
// warp: [group][lo / hi half][lane] uint4, conflict-free), built ahead of the GEMV in an idle
// window.
__device__ __forceinline__ void stage_tile_a_f16(uint32_t tile_smem, uint32_t image, int warp,
                                                 int lane) {
#pragma unroll
  for (int g = 0; g < kGroupsPerWarp; ++g) {
    const uint4 a = weight_a_frag(tile_smem, warp * 2 + g / 4, g % 4, lane);
    sts_128(image + uint32_t(g * 2) * 512u + uint32_t(lane) * 16u,
            make_uint4(f16_lo(a.x), f16_lo(a.y), f16_lo(a.z), f16_lo(a.w)));
    sts_128(image + uint32_t(g * 2 + 1) * 512u + uint32_t(lane) * 16u,
            make_uint4(f16_hi(a.x), f16_hi(a.y), f16_hi(a.z), f16_hi(a.w)));
  }
}
// tiles_gemv_h with tile 0's A operand read from its f16 image (stage_tile_a_f16): per tile the
// same HMMA pairs, scales and order.
template <int T>
__device__ __forceinline__ void tiles_gemv_h_pre0(const uint32_t (&slot)[T], uint32_t image0,
                                                  const uint32_t (&sfw)[T][2][2], const BFragH &f,
                                                  int warp, int lane, float (&run)[T][4]) {
#pragma unroll
  for (int t = 0; t < T; ++t)
    run[t][0] = run[t][1] = run[t][2] = run[t][3] = 0.f;
#pragma unroll
  for (int g = 0; g < kGroupsPerWarp; ++g) {
    uint4 a[T];
#pragma unroll
    for (int t = 1; t < T; ++t)
      a[t] = weight_a_frag(slot[t], warp * 2 + g / 4, g % 4, lane);
    const uint4 lo = lds_128(image0 + uint32_t(g * 2) * 512u + uint32_t(lane) * 16u);
    const uint4 hi = lds_128(image0 + uint32_t(g * 2 + 1) * 512u + uint32_t(lane) * 16u);
#pragma unroll
    for (int t = 0; t < T; ++t) {
      float d[4] = {0.f, 0.f, 0.f, 0.f};
      if (t == 0) {
        hmma16816(d, lo.x, lo.y, lo.z, lo.w, f.b[g].lo[0], f.b[g].lo[1]);
        hmma16816(d, hi.x, hi.y, hi.z, hi.w, f.b[g].hi[0], f.b[g].hi[1]);
      } else {
        qmma_h(d, a[t], f.b[g]);
      }
      const uint32_t w0 = sfw[t][0][g / 4], w1 = sfw[t][1][g / 4];
      const float sa0 = (g % 4) == 0   ? sa_of<0>(w0)
                        : (g % 4) == 1 ? sa_of<1>(w0)
                        : (g % 4) == 2 ? sa_of<2>(w0)
                                       : sa_of<3>(w0);
      const float sa1 = (g % 4) == 0   ? sa_of<0>(w1)
                        : (g % 4) == 1 ? sa_of<1>(w1)
                        : (g % 4) == 2 ? sa_of<2>(w1)
                                       : sa_of<3>(w1);
      run[t][0] = __fmaf_rn(d[0], __fmul_rn(sa0, f.sb0[g]), run[t][0]);
      run[t][2] = __fmaf_rn(d[2], __fmul_rn(sa1, f.sb0[g]), run[t][2]);
      run[t][1] = __fmaf_rn(d[1], __fmul_rn(sa0, f.sb1[g]), run[t][1]);
      run[t][3] = __fmaf_rn(d[3], __fmul_rn(sa1, f.sb1[g]), run[t][3]);
    }
  }
}

// Ordered fold 0 + w0 + ... + w7 for 128 (token, row) pairs; threads 0..127 return it.
__device__ __forceinline__ float fold_tile(float *fold, const float (&run)[4], int tid, int warp,
                                           int lane) {
  const int g = lane >> 2, t2 = (lane & 3) * 2;
  float *w = fold + warp * (kTileRows * kTokens);
  w[g * kTokens + t2] = run[0];
  w[g * kTokens + t2 + 1] = run[1];
  w[(g + 8) * kTokens + t2] = run[2];
  w[(g + 8) * kTokens + t2 + 1] = run[3];
  consumer_sync();
  float sum = 0.f;
  if (tid < kTileRows * kTokens) {
#pragma unroll
    for (int k = 0; k < kConsumerWarps; ++k)
      sum = __fadd_rn(sum, fold[k * (kTileRows * kTokens) + tid]);
  }
  return sum; // (row = tid / 8, token = tid % 8)
}

// ---- QKFIN --------------------------------------------------------------------------
__device__ __forceinline__ void store_nvfp4_row(uint8_t *data, uint8_t *scale, int64_t data_stride,
                                                int64_t scale_stride, int64_t slot, int lane,
                                                const float (&v)[4], float global_scale) {
  const float inv_gs = __frcp_rn(global_scale);
  float amax = fmaxf(fmaxf(fabsf(v[0]), fabsf(v[1])), fmaxf(fabsf(v[2]), fabsf(v[3])));
  amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, 1));
  amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, 2));
  const float sf = __fdiv_rn(__fmul_rn(amax, inv_gs), 6.f);
  const uint32_t byte = f32x2_to_e4m3x2(sf, sf) & 0xffu;
  float q[4] = {0.f, 0.f, 0.f, 0.f};
  if (byte != 0u) {
    const float inv_rs = __frcp_rn(e4m3_to_f32(byte));
#pragma unroll
    for (int i = 0; i < 4; ++i)
      q[i] = __fmul_rn(__fmul_rn(v[i], inv_gs), inv_rs);
  }
  const uint16_t codes = f32x4_to_e2m1x4(q[0], q[1], q[2], q[3]);
  const int64_t page = slot / kPage, row = slot % kPage;
  *reinterpret_cast<uint16_t *>(data + page * data_stride + row * (kHeadDim / 2) + 2 * lane) =
      codes;
  if ((lane & 3) == 0)
    scale[page * scale_stride + row * (kHeadDim / 16) + lane / 4] = uint8_t(byte);
}

// Per-token inputs of the head finish that do not depend on the projection; loaded before the
// finisher polls its heads' partials. Warp `tok` finishes token tok; lane owns dims 4 lane .. 4
// lane + 3.
struct FinInputs {
  float gamma[2][4];
  float cosv[4], sinv[4]; // NeoX RoPE of dims 0..63 (lanes < 16)
  int64_t slot;
};
__device__ __forceinline__ int norm_index(int head) {
  return head < kHeads ? 0 : (head == kHeadK ? 1 : (head == kHeadQi ? 2 : 3));
}
__device__ __forceinline__ void fin_prefetch(const Params &p, const int (&heads)[2], int nh,
                                             int tid, FinInputs &in) {
  const int tok = tid >> 5, lane = tid & 31, d0 = lane * 4;
#pragma unroll
  for (int h = 0; h < 2; ++h) {
    if (h < nh && heads[h] != kHeadV) {
      const uint2 g = *reinterpret_cast<const uint2 *>(p.head_norm[norm_index(heads[h])] + d0);
      in.gamma[h][0] = bf16_lo(g.x);
      in.gamma[h][1] = bf16_hi(g.x);
      in.gamma[h][2] = bf16_lo(g.y);
      in.gamma[h][3] = bf16_hi(g.y);
    }
  }
  in.slot = p.slot_mapping[tok];
  if (lane < 16) {
    const int64_t pos = p.positions[tok];
    const __nv_bfloat16 *cs = p.cos_sin + pos * kRotaryDim;
    const int base = (lane < 8 ? d0 : d0 - 32);
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      in.cosv[i] = __bfloat162float(cs[base + i]);
      in.sinv[i] = __bfloat162float(cs[32 + base + i]);
    }
  }
}

// Head RMSNorm (not V), NeoX RoPE (not V), then E4M3 Q / QI or an NVFP4 cache row.
__device__ __forceinline__ void finish_head(const Params &p, int head, const float (&g)[4],
                                            const FinInputs &in, int tok, int lane, float (&v)[4],
                                            uint32_t tag) {
  const int d0 = lane * 4;
  if (head != kHeadV) {
    float ss = 0.f;
#pragma unroll
    for (int i = 0; i < 4; ++i)
      ss = __fmaf_rn(v[i], v[i], ss);
    ss = warp_sum(ss);
    const float inv = __frcp_rn(__fsqrt_rn(__fadd_rn(__fdiv_rn(ss, float(kHeadDim)), p.head_eps)));
#pragma unroll
    for (int i = 0; i < 4; ++i)
      v[i] = __fmul_rn(__fmul_rn(v[i], inv), __fadd_rn(1.f, g[i]));
    // NeoX RoPE on dims 0..63: lane l < 8 pairs with lane l + 8.
    float partner[4];
#pragma unroll
    for (int i = 0; i < 4; ++i)
      partner[i] = __shfl_xor_sync(0xffffffffu, v[i], 8);
    if (lane < 16) {
#pragma unroll
      for (int i = 0; i < 4; ++i) {
        const float c = in.cosv[i], s = in.sinv[i];
        v[i] = lane < 8 ? __fsub_rn(__fmul_rn(v[i], c), __fmul_rn(partner[i], s))
                        : __fadd_rn(__fmul_rn(v[i], c), __fmul_rn(partner[i], s));
      }
    }
  }
  if (head < kHeads) {
    const uint32_t q = f32x4_to_e4m3x4(v[0], v[1], v[2], v[3]);
    const size_t off = (size_t(tok) * kHeads + head) * kHeadDim + d0;
    *reinterpret_cast<uint32_t *>(p.q_out + off) = q;
    // Tagged records for ATTN: lane 2m writes dims 8m .. 8m + 7 of (token, head).
    const uint32_t partner = __shfl_xor_sync(0xffffffffu, q, 1);
    if ((lane & 1) == 0)
      st_rec(p.arena + arena::kQRec +
                 ((size_t(tok) * kHeads + head) * (kHeadDim / 8) + lane / 2) * 16,
             q, partner, 0u, tag);
  } else if (head == kHeadQi) {
    const uint32_t q = f32x4_to_e4m3x4(v[0], v[1], v[2], v[3]);
    const size_t off = size_t(tok) * kHeadDim + d0;
    *reinterpret_cast<uint32_t *>(p.qi_out + off) = q;
    // The same bytes as tagged records for SCORE: lane 2m writes dims 8m .. 8m + 7.
    const uint32_t partner = __shfl_xor_sync(0xffffffffu, q, 1);
    if ((lane & 1) == 0)
      st_rec(p.arena + arena::kQiRec + (size_t(tok) * (kHeadDim / 8) + lane / 2) * 16, q, partner,
             0u, tag);
  } else {
    const int64_t slot = in.slot;
    if (slot >= 0) {
      if (head == kHeadK)
        store_nvfp4_row(p.k_data, p.k_scale, p.k_data_stride, p.k_scale_stride, slot, lane, v,
                        p.k_global);
      else if (head == kHeadV)
        store_nvfp4_row(p.v_data, p.v_scale, p.v_data_stride, p.v_scale_stride, slot, lane, v,
                        p.v_global);
      else
        store_nvfp4_row(p.ki_data, p.ki_scale, p.ki_data_stride, p.ki_scale_stride, slot, lane, v,
                        p.ki_global);
      // Later TMA / bulk copies in other CTAs read these rows through the async proxy.
      fence_proxy_async_global();
    }
  }
}

// Partial records of (projection tile, seg): [tok][row] {fp32 bits, epoch tag}.
__device__ __forceinline__ uint8_t *part_rec(const Params &p, int tile, int seg) {
  return p.arena + arena::kPartRec + (size_t(tile) * kKSegs + seg) * (kTokens * kTileRows) * 8;
}
// Finish nh (1 or 2) heads from their tagged partial records (rows row0 .. row0 + 3 of token tok:
// two 16-byte loads per K segment and head), polled until every record carries this call's tag;
// the segments are summed in order: (s0 + s1) + s2.
template <bool kDeadline>
__device__ __forceinline__ void qkfin_records(const Params &p, const Deadline<kDeadline> &dl,
                                              const int (&heads)[2], int nh, const FinInputs &in,
                                              uint32_t tag, int tid) {
  const int tok = tid >> 5, lane = tid & 31;
  const int d0 = lane * 4, row0 = d0 % kTileRows, t = d0 / kTileRows;
  uint4 r[2][kKSegs][2];
  uint32_t pending = nh > 1 ? 0xfffu : 0x3fu;
  while (pending) {
#pragma unroll
    for (int h = 0; h < 2; ++h)
#pragma unroll
      for (int sg = 0; sg < kKSegs; ++sg)
#pragma unroll
        for (int q = 0; q < 2; ++q)
          if (pending & (1u << (h * 6 + sg * 2 + q)))
            r[h][sg][q] = ld_rec(part_rec(p, heads[h] * kTilesPerHead + t, sg) +
                                 (size_t(tok) * kTileRows + row0 + 2 * q) * 8);
#pragma unroll
    for (int h = 0; h < 2; ++h)
#pragma unroll
      for (int sg = 0; sg < kKSegs; ++sg)
#pragma unroll
        for (int q = 0; q < 2; ++q) {
          const uint32_t bit = 1u << (h * 6 + sg * 2 + q);
          if ((pending & bit) && r[h][sg][q].y == tag && r[h][sg][q].w == tag)
            pending &= ~bit;
        }
    if (pending)
      dl.check(p, 0x50u + uint32_t(heads[0]), pending, tag);
  }
#pragma unroll
  for (int h = 0; h < 2; ++h) {
    if (h < nh) {
      float v[4];
      v[0] = __fadd_rn(__fadd_rn(__uint_as_float(r[h][0][0].x), __uint_as_float(r[h][1][0].x)),
                       __uint_as_float(r[h][2][0].x));
      v[1] = __fadd_rn(__fadd_rn(__uint_as_float(r[h][0][0].z), __uint_as_float(r[h][1][0].z)),
                       __uint_as_float(r[h][2][0].z));
      v[2] = __fadd_rn(__fadd_rn(__uint_as_float(r[h][0][1].x), __uint_as_float(r[h][1][1].x)),
                       __uint_as_float(r[h][2][1].x));
      v[3] = __fadd_rn(__fadd_rn(__uint_as_float(r[h][0][1].z), __uint_as_float(r[h][1][1].z)),
                       __uint_as_float(r[h][2][1].z));
      finish_head(p, heads[h], in.gamma[h], in, tok, lane, v, tag);
    }
  }
}

// Readiness counter of the cache rows a K, V or KI finisher writes (the producers of later KI / K /
// V page copies wait for it); Q and QI are polled as tagged records instead.
__device__ __forceinline__ int ready_counter(int head) {
  return head == kHeadKi ? arena::kCntKiReady : arena::kCntKvReady;
}

} // namespace attn_ring::proj
