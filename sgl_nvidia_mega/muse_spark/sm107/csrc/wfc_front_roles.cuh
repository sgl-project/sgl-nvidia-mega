// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// Portions Copyright 2023-2024 SGLang Team (Apache-2.0)
// Derived from SGLang (https://github.com/sgl-project/sglang), Apache License 2.0; see ../include/LICENSE.sglang.
// Sources: SGLang python/sglang/kernels/jit/csrc/muse_spark/routing.cuh (routing_kernel) and
// python/sglang/kernels/jit/csrc/muse_spark/norm_fp4_quant.cuh (muse_spark_norm_fp4_quant_kernel); the helpers they use are
// included from those headers unchanged.
// Modified by NVIDIA: the per-row bodies of the two kernels run as device functions (roles) that a larger launch calls for
// the rows it owns: routing = one warp per row; RMS norm + per-token NVFP4 quant = a group of kNumThreads threads per row that
// synchronizes through a named barrier instead of __syncthreads. After its outputs, each row releases tagged 64-bit ready
// words (tag in the high half), so a consumer in another CTA of the same grid can wait for that row alone. Whole-warp f32 max
// reductions are issued as volatile statements (as current SGLang does), so the compiler cannot move them into a
// lane-divergent branch.
//
// Front roles for a fused MoE front launch (dev; not wired into the server).
//   route_row: logits [M, E] fp32 (the router output) -> topk ids / weights / packed ids, as SGLang's routing op computes
//     them (sigmoid scores x output multiplier, + bias for selection, iterative top-k with the lowest-id tie-break, weights in
//     selection order renormalized by the sum of the selected scores + eps, times route_scale). Each lane k < kTopK then
//     releases word[k] = tag << 32 | packed bits (st.release.gpu after the lane's own output stores).
//   quant_row: pooled hidden [M, H] bf16 -> RMS norm (weighted) -> bf16 rounding -> per-token NVFP4 (fp4 pairs, linear e4m3
//     block scales, per-token scale), as SGLang's fused norm + quant op computes them; the group's reductions keep that
//     kernel's order (warp shuffle tree, then one warp over the per-warp partials in warp order). After the group's stores
//     and a named-barrier round, thread 0 of the group releases word = tag << 32 | row (st.release.gpu).
//   Ordering argument (PTX ISA, memory consistency model): the group's stores precede bar.sync in program order; bar.sync
//   orders them before thread 0's release (CTA-scope synchronization), and the release / acquire pair at gpu scope extends
//   causality order to the consumer that observes the word with ld.acquire.gpu. Every word is written once per tag (no
//   re-arm); tags come from a device-memory epoch (never frozen in a captured kernel argument).
//   wait_word: bounded consumer poll (ld.acquire.gpu at the publish width); on timeout it sets a sticky error bit and
//   returns false so the caller can poison its outputs.

#pragma once

#include "muse_spark_v12/routing.cuh"
#include "muse_spark_v12/norm_fp4_quant.cuh"

#include <cuda_runtime.h>

#include <cstdint>

namespace sglang {
namespace muse_vr_front {

using muse_spark::div_full;
using muse_spark::kNegInf;
using muse_spark::pack_id_weight;
using muse_spark::triton_sigmoid;

__device__ __forceinline__ void st_release_gpu(unsigned long long* p, unsigned long long v) {
  asm volatile("st.release.gpu.global.b64 [%0], %1;" ::"l"(p), "l"(v) : "memory");
}
__device__ __forceinline__ unsigned long long ld_acquire_gpu(const unsigned long long* p) {
  unsigned long long v;
  asm volatile("ld.acquire.gpu.global.b64 %0, [%1];" : "=l"(v) : "l"(p) : "memory");
  return v;
}
// Whole-warp f32 max as the served headers compute it (redux.sync.max.f32 on the targets where SGLang uses it, else its
// shuffle tree), always as a volatile statement: a non-volatile warp collective may be moved into a lane-divergent branch.
__device__ __forceinline__ float warp_max_f32(float v) {
#if defined(SGL_CUDA_ARCH) && SGL_CUDA_ARCH >= 1000 && SGL_CUDA_ARCH < 1100
  float r;
  asm volatile("redux.sync.max.f32 %0, %1, %2;" : "=f"(r) : "f"(v), "r"(0xffffffffu));
  return r;
#else
  return device::warp::reduce_max(v);
#endif
}
__device__ __forceinline__ void group_bar(int id, int threads) {
  asm volatile("bar.sync %0, %1;" ::"r"(id), "r"(threads) : "memory");
}

// ---------------------------------------------------------------------------------------------------------------------
// Routing role.
struct RouteArgs {
  const float* logits;               // [M, E] fp32, row stride stride_gm (written by an earlier grid)
  const float* bias;                 // [E] fp32 (static)
  float* topk_w;                     // [M, kTopK] or nullptr
  int32_t* topk_ids;                 // [M, kTopK] or nullptr
  int32_t* packed;                   // [M, kTopK] or nullptr
  const int32_t* local_expert_mapping;  // [E] or nullptr (identity)
  const int32_t* num_valid_tokens;   // [1] or nullptr (every row real)
  unsigned long long* words;         // [M, kTopK] tagged ready words or nullptr
  int num_tokens;
  int num_experts;
  int64_t stride_gm;
  float router_output_multiplier;
  float route_scale;
  float division_eps;
};

// One warp, one row. Same arithmetic and selection order as SGLang's routing_kernel (sigmoid mode).
template <int kTopK, int kEPL>
__device__ __forceinline__ void route_row(const RouteArgs& a, int token, int lane, uint32_t tag) {
  using namespace device;
  constexpr int kVecsPerLane = kEPL / 4;
  constexpr int kGroupExperts = 4 * kWarpThreads;
  static_assert(kEPL % 4 == 0 && kTopK >= 1 && kTopK <= 32);
  if (a.num_valid_tokens != nullptr && token >= *a.num_valid_tokens) {
    if (lane < kTopK) {
      if (a.topk_ids != nullptr) a.topk_ids[token * kTopK + lane] = -1;
      if (a.packed != nullptr) a.packed[token * kTopK + lane] = -1 << 16;
      if (a.words != nullptr)
        st_release_gpu(a.words + token * kTopK + lane,
                       (static_cast<unsigned long long>(tag) << 32) | static_cast<uint32_t>(-1 << 16));
    }
    return;
  }
  const float* row = a.logits + static_cast<int64_t>(token) * a.stride_gm;
  float score[kEPL];
  int expert[kEPL];
#pragma unroll
  for (int v = 0; v < kVecsPerLane; ++v) {
    const int base = v * kGroupExperts + lane * 4;
    float4 g = make_float4(kNegInf, kNegInf, kNegInf, kNegInf);
    if (base < a.num_experts) g = *reinterpret_cast<const float4*>(row + base);
    score[v * 4 + 0] = g.x;
    score[v * 4 + 1] = g.y;
    score[v * 4 + 2] = g.z;
    score[v * 4 + 3] = g.w;
#pragma unroll
    for (int j = 0; j < 4; ++j) expert[v * 4 + j] = base + j;
  }
#pragma unroll
  for (int s = 0; s < kEPL; ++s) score[s] *= a.router_output_multiplier;
#pragma unroll
  for (int s = 0; s < kEPL; ++s) score[s] = triton_sigmoid(score[s]);
  float biased[kEPL];
#pragma unroll
  for (int v = 0; v < kVecsPerLane; ++v) {
    const int base = v * kGroupExperts + lane * 4;
    float4 b = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    if (base < a.num_experts) b = *reinterpret_cast<const float4*>(a.bias + base);
    biased[v * 4 + 0] = b.x;
    biased[v * 4 + 1] = b.y;
    biased[v * 4 + 2] = b.z;
    biased[v * 4 + 3] = b.w;
  }
  float key[kEPL];
#pragma unroll
  for (int s = 0; s < kEPL; ++s) {
    if (expert[s] < a.num_experts) {
      key[s] = biased[s] + score[s];
    } else {
      score[s] = 0.0f;
      key[s] = kNegInf;
    }
  }
  // Stable odd-even sort per lane (equal keys keep slot order; slots are monotone in expert id).
#pragma unroll
  for (int r = 0; r < kEPL; ++r) {
#pragma unroll
    for (int i = r % 2; i + 1 < kEPL; i += 2) {
      const bool swap = key[i + 1] > key[i];
      const float k0 = swap ? key[i + 1] : key[i];
      const float k1 = swap ? key[i] : key[i + 1];
      const int e0 = swap ? expert[i + 1] : expert[i];
      const int e1 = swap ? expert[i] : expert[i + 1];
      const float s0 = swap ? score[i + 1] : score[i];
      const float s1 = swap ? score[i] : score[i + 1];
      key[i] = k0;
      key[i + 1] = k1;
      expert[i] = e0;
      expert[i + 1] = e1;
      score[i] = s0;
      score[i + 1] = s1;
    }
  }
  float score_sum = 0.0f;
  int out_id = 0;
  float out_score = 0.0f;
#pragma unroll
  for (int k = 0; k < kTopK; ++k) {
    const float round_max = warp_max_f32(key[0]);
    const bool is_head = key[0] == round_max;
    unsigned owners = 0;
#pragma unroll
    for (int g = kVecsPerLane - 1; g >= 0; --g) {
      const unsigned m = __ballot_sync(warp::kFullMask, is_head && expert[0] / kGroupExperts == g);
      owners = (m != 0u) ? m : owners;
    }
    const int owner = __ffs(owners) - 1;
    const int winner = __shfl_sync(warp::kFullMask, expert[0], owner);
    const float selected = __shfl_sync(warp::kFullMask, score[0], owner);
    const bool pop = expert[0] == winner;
#pragma unroll
    for (int s = 0; s + 1 < kEPL; ++s)
      if (s + k <= kTopK - 2) {
        key[s] = pop ? key[s + 1] : key[s];
        expert[s] = pop ? expert[s + 1] : expert[s];
        score[s] = pop ? score[s + 1] : score[s];
      }
    if (kEPL - 1 + k <= kTopK - 2) key[kEPL - 1] = pop ? kNegInf : key[kEPL - 1];
    score_sum += selected;
    if (lane == k) {
      out_id = winner;
      out_score = selected;
    }
  }
  score_sum += a.division_eps;
  const float inv_sum_scaled = div_full(a.route_scale, score_sum);
  if (lane < kTopK) {
    const int32_t id =
        (a.local_expert_mapping != nullptr) ? a.local_expert_mapping[out_id] : static_cast<int32_t>(out_id);
    const float weight = out_score * inv_sum_scaled;
    const int32_t pk = pack_id_weight(id, weight);
    if (a.topk_ids != nullptr) a.topk_ids[token * kTopK + lane] = id;
    if (a.topk_w != nullptr) a.topk_w[token * kTopK + lane] = weight;
    if (a.packed != nullptr) a.packed[token * kTopK + lane] = pk;
    if (a.words != nullptr)
      st_release_gpu(a.words + token * kTopK + lane,
                     (static_cast<unsigned long long>(tag) << 32) | static_cast<uint32_t>(pk));
  }
}

// ---------------------------------------------------------------------------------------------------------------------
// Norm + quant role.
inline constexpr int64_t kPooledHidden = 4096;  // pooled (latent) width the test / K_FM launches instantiate

struct QuantArgs {
  const void* x;                 // [M, H] bf16 (written by an earlier grid)
  const void* weight;            // [H] bf16 effective norm weight, or nullptr (weightless)
  void* fp4_out;                 // [M, H/2] u8
  void* sf_out;                  // [M, H/16] u8, linear layout
  float* per_token_scale;        // [M]
  unsigned long long* words;     // [M] tagged ready words, or nullptr
  float eps;
  float global_scale_inv;
  const float* partials;         // kS > 0: [kS slices][M][H] fp32 K-slice partials of the pooled row (slice stride pstride)
  int64_t pstride;               //   elements between slices
  void* pooled_out;              //   [M, H] bf16 pooled row (the rounded slice sum), or nullptr
  uint8_t* hs_perm;              // kOut = 1 (permuted image): [rows, H/2] fp4 rows in the expert-major order
  uint8_t* img1;                 //   scale image: scale word g (4 bytes) of permuted row r at img1 + row_img[r] + g * 512
  const int32_t* row_img;        //   [rows] byte offset of each permuted row's word 0 in img1
  float* pts_perm;               //   [rows] per-token scales in the permuted order
};

// A group of kNumThreads threads (whole warps, gtid = thread index within the group), one row. smem: 2 * kNumWarps floats
// private to the group. bar_id: a named barrier id reserved for this group (1..15).
// kS = 0: the input row is the pooled bf16 row. kS > 0: the input is kS fp32 K-slice partials of the pooled row; they are summed
// in slice order (fp32, round to nearest), rounded once to bf16 (that is the pooled row), and the rest runs unchanged.
// kPlantNoBar (tests only): drops the first group barrier, a planted shared-memory race for sanitizer non-vacuity checks.
// kOut = 0: outputs at row `row` of fp4_out / sf_out / per_token_scale (linear layout). kOut = 1: outputs at permuted row
// `out_row` of hs_perm / img1 (+ row_img) / pts_perm (the expert-major image a MoE kernel reads); same bytes, other addresses.
// kPreW: the norm weight's loads are issued with the row's loads at the start (same values; they leave the chain after the
// first reduction).
template <int64_t kHidden, int kNumThreads, bool kHasWeight, bool kPlantNoBar = false, int kS = 0, int kOut = 0,
          bool kPreW = false>
__device__ __forceinline__ void quant_row(const QuantArgs& a, int64_t row, int gtid, int bar_id, float* smem, uint32_t tag,
                                          int64_t out_row = -1) {
  using namespace device;
  using namespace device::norm_fp4_quant;
  constexpr int kPacks = kHidden / (kNumThreads * kBlock);
  constexpr int kNumWarps = kNumThreads / kWarpThreads;
  static_assert(kHidden % (kNumThreads * kBlock) == 0);
  static_assert(kNumThreads % kWarpThreads == 0 && kNumWarps > 1);
  using BF16Vec = AlignedVector<bf16x2_t, 4>;
  const auto x = static_cast<const bf16_t*>(a.x) + row * kHidden;
  const auto weight = static_cast<const bf16_t*>(a.weight);
  const int lane = gtid % kWarpThreads;
  const int gwarp = gtid / kWarpThreads;
  fp32x2_t vals[kPacks][kBlock / 2];
  float sum_sq = 0.0f;
  BF16Vec wpre_lo[kPreW && kHasWeight ? kPacks : 1], wpre_hi[kPreW && kHasWeight ? kPacks : 1];
  if constexpr (kPreW && kHasWeight) {
#pragma unroll
    for (int p = 0; p < kPacks; ++p) {
      const int blk = gtid + p * kNumThreads;
      wpre_lo[p].load(weight, blk * 2);
      wpre_hi[p].load(weight, blk * 2 + 1);
    }
  }
#pragma unroll
  for (int p = 0; p < kPacks; ++p) {
    const int blk = gtid + p * kNumThreads;
    if constexpr (kS == 0) {
      BF16Vec lo, hi;
      lo.load(x, blk * 2);
      hi.load(x, blk * 2 + 1);
#pragma unroll
      for (int q = 0; q < 4; ++q) {
        vals[p][q] = cast<fp32x2_t>(lo[q]);
        vals[p][q + 4] = cast<fp32x2_t>(hi[q]);
      }
    } else {
      // element blk * 16 + 2j (+1) -> vals[p][j]: the same element order as the bf16 path above
      const float* p0 = a.partials + row * kHidden + static_cast<int64_t>(blk) * kBlock;
      float acc[kBlock];
#pragma unroll
      for (int v = 0; v < kBlock / 4; ++v) {
        const float4 t = *reinterpret_cast<const float4*>(p0 + v * 4);
        acc[v * 4 + 0] = t.x;
        acc[v * 4 + 1] = t.y;
        acc[v * 4 + 2] = t.z;
        acc[v * 4 + 3] = t.w;
      }
#pragma unroll
      for (int sl = 1; sl < kS; ++sl) {
#pragma unroll
        for (int v = 0; v < kBlock / 4; ++v) {
          const float4 t = *reinterpret_cast<const float4*>(p0 + sl * a.pstride + v * 4);
          acc[v * 4 + 0] = __fadd_rn(acc[v * 4 + 0], t.x);
          acc[v * 4 + 1] = __fadd_rn(acc[v * 4 + 1], t.y);
          acc[v * 4 + 2] = __fadd_rn(acc[v * 4 + 2], t.z);
          acc[v * 4 + 3] = __fadd_rn(acc[v * 4 + 3], t.w);
        }
      }
      BF16Vec lo, hi;
#pragma unroll
      for (int q = 0; q < 4; ++q) {
        fp32x2_t l, h;
        l.x = acc[2 * q];
        l.y = acc[2 * q + 1];
        h.x = acc[8 + 2 * q];
        h.y = acc[8 + 2 * q + 1];
        lo[q] = cast<bf16x2_t>(l);
        hi[q] = cast<bf16x2_t>(h);
        vals[p][q] = cast<fp32x2_t>(lo[q]);
        vals[p][q + 4] = cast<fp32x2_t>(hi[q]);
      }
      if (a.pooled_out != nullptr) {
        const auto po = static_cast<bf16_t*>(a.pooled_out) + row * kHidden;
        lo.store(po, blk * 2);
        hi.store(po, blk * 2 + 1);
      }
    }
#pragma unroll
    for (int q = 0; q < kBlock / 2; ++q) {
      sum_sq = fmaf(vals[p][q].x, vals[p][q].x, sum_sq);
      sum_sq = fmaf(vals[p][q].y, vals[p][q].y, sum_sq);
    }
  }
  // cta::reduce_sum<kNumThreads> with the group's named barrier in place of __syncthreads (same operations, same order).
  sum_sq = warp::reduce_sum(sum_sq);
  if (lane == 0) smem[gwarp] = sum_sq;
  if constexpr (!kPlantNoBar) group_bar(bar_id, kNumThreads);
  sum_sq = warp::reduce_sum(lane < kNumWarps ? smem[lane] : 0.0f);
  const float rstd = math::rsqrt(sum_sq / kHidden + a.eps);
  bf16x2_t normed[kPacks][kBlock / 2];
  float local_amax = 0.0f;
#pragma unroll
  for (int p = 0; p < kPacks; ++p) {
    const int blk = gtid + p * kNumThreads;
    BF16Vec w_lo, w_hi;
    if constexpr (kHasWeight && kPreW) {
      w_lo = wpre_lo[p];
      w_hi = wpre_hi[p];
    } else if constexpr (kHasWeight) {
      w_lo.load(weight, blk * 2);
      w_hi.load(weight, blk * 2 + 1);
    }
#pragma unroll
    for (int q = 0; q < kBlock / 2; ++q) {
      fp32x2_t v = vals[p][q];
      v.x *= rstd;
      v.y *= rstd;
      if constexpr (kHasWeight) {
        const auto wf = cast<fp32x2_t>(q < 4 ? w_lo[q] : w_hi[q - 4]);
        v.x *= wf.x;
        v.y *= wf.y;
      }
      normed[p][q] = cast<bf16x2_t>(v);
      const auto r = cast<fp32x2_t>(normed[p][q]);
      local_amax = fmaxf(local_amax, fmaxf(fabsf(r.x), fabsf(r.y)));
    }
  }
  float* max_smem = smem + kNumWarps;
  float row_amax = warp_max_f32(local_amax);
  if (lane == 0) max_smem[gwarp] = row_amax;
  group_bar(bar_id, kNumThreads);
  row_amax = warp_max_f32(lane < kNumWarps ? max_smem[lane] : 0.0f);
  float per_token_scale = 0.0f;
  float encode_scale = kFloat32Max;
  if (row_amax != 0.0f) {
    per_token_scale = mul_f32(row_amax, a.global_scale_inv);
    encode_scale = rcp_approx_ftz(per_token_scale);
  }
  if (gtid == 0) {
    if constexpr (kOut == 0) a.per_token_scale[row] = per_token_scale;
    else a.pts_perm[out_row] = per_token_scale;
  }
  const float fp4_max_rcp = rcp_approx_ftz(6.0f);
  const auto fp4 = kOut == 0 ? static_cast<uint64_t*>(a.fp4_out) + row * (kHidden / kBlock)
                             : reinterpret_cast<uint64_t*>(a.hs_perm + out_row * (kHidden / 2));
  const auto sf = kOut == 0 ? static_cast<uint8_t*>(a.sf_out) + row * (kHidden / kBlock) : nullptr;
#pragma unroll
  for (int p = 0; p < kPacks; ++p) {
    const int blk = gtid + p * kNumThreads;
    float f[kBlock];
    float block_max = 0.0f;
#pragma unroll
    for (int q = 0; q < kBlock / 2; ++q) {
      const auto r = cast<fp32x2_t>(normed[p][q]);
      f[2 * q] = r.x;
      f[2 * q + 1] = r.y;
      block_max = fmaxf(block_max, fmaxf(fabsf(r.x), fabsf(r.y)));
    }
    const float scale_f = mul_f32(encode_scale, mul_f32(block_max, fp4_max_rcp));
    const uint32_t fp8 = cvt_f32_to_e4m3(scale_f) & 0xFFu;
    const float out_scale = output_scale(fp8, encode_scale);
#pragma unroll
    for (int i = 0; i < kBlock; ++i) f[i] = mul_f32(f[i], out_scale);
    const uint64_t pk = (static_cast<uint64_t>(cvt_e2m1x8(f + 8)) << 32) | static_cast<uint64_t>(cvt_e2m1x8(f));
    fp4[blk] = pk;
    if constexpr (kOut == 0) {
      sf[blk] = static_cast<uint8_t>(fp8);
    } else {  // byte (blk % 4) of scale word blk / 4
      a.img1[static_cast<int64_t>(a.row_img[out_row]) + (blk >> 2) * 512 + (blk & 3)] = static_cast<uint8_t>(fp8);
    }
  }
  if (a.words != nullptr) {
    group_bar(bar_id, kNumThreads);  // every thread's stores of this row precede the release below
    if (gtid == 0) st_release_gpu(a.words + row, (static_cast<unsigned long long>(tag) << 32) | static_cast<uint32_t>(row));
  }
}

// Bounded consumer wait on one tagged word (one lane). Returns the word's low half; on timeout sets err bit 1 and returns
// 0xffffffff with *ok = false (the caller poisons what it would have produced from this row).
__device__ __forceinline__ uint32_t wait_word(const unsigned long long* w, uint32_t tag, long long max_polls,
                                              unsigned long long* err, bool* ok) {
  for (long long i = 0; i < max_polls; ++i) {
    const unsigned long long v = ld_acquire_gpu(w);
    if (static_cast<uint32_t>(v >> 32) == tag) {
      *ok = true;
      return static_cast<uint32_t>(v);
    }
    if ((i & 1023) == 1023) __nanosleep(64);
  }
  atomicOr(err, 1ull);
  *ok = false;
  return 0xffffffffu;
}

// ---------------------------------------------------------------------------------------------------------------------
// Test launch: kGroups quant groups (kQThreads threads each, named barriers 1..kGroups) + kRouteWarps routing warps per CTA.
// Quant group q of CTA b owns rows b * kGroups + q (+ grid * kGroups); routing warp w owns rows b * kRouteWarps + w (+ ...).
// The tag comes from a device word (epoch) so graph replays can advance it.
template <int kQThreads, int kGroups, int kRouteWarps, bool kHasWeight, int kS = 0>
__global__ void __launch_bounds__(kQThreads * kGroups + 32 * kRouteWarps)
    front_roles_test_kernel(const RouteArgs ra, const QuantArgs qa, int M, const unsigned long long* epoch, int layer) {
  __shared__ float smem[kGroups][2 * (kQThreads / 32)];
  const uint32_t tag = static_cast<uint32_t>(*epoch * 64 + layer);
  const int t = threadIdx.x;
  if (t < kQThreads * kGroups) {
    const int q = t / kQThreads;
    for (int row = blockIdx.x * kGroups + q; row < M; row += gridDim.x * kGroups)
      quant_row<4096, kQThreads, kHasWeight, false, kS>(qa, row, t % kQThreads, 1 + q, smem[q], tag);
  } else {
    const int w = (t - kQThreads * kGroups) / 32;
    const int lane = t % 32;
    for (int row = blockIdx.x * kRouteWarps + w; row < M; row += gridDim.x * kRouteWarps)
      route_row<8, 8>(ra, row, lane, tag);
  }
}

// Test consumer: one warp per row waits on the row's route words (lanes 0..7) and quant word (lane 8); copies the low
// halves out (got[row * 9 + j]); poisoned on a timeout.
__global__ void front_roles_wait_kernel(const unsigned long long* rwords, const unsigned long long* qwords, int M,
                                        const unsigned long long* epoch, int layer, long long max_polls,
                                        unsigned long long* err, int32_t* got) {
  const uint32_t tag = static_cast<uint32_t>(*epoch * 64 + layer);
  const int row = blockIdx.x * (blockDim.x / 32) + threadIdx.x / 32;
  const int lane = threadIdx.x % 32;
  if (row >= M || lane > 8) return;
  bool ok = false;
  const uint32_t v = lane < 8 ? wait_word(rwords + row * 8 + lane, tag, max_polls, err, &ok)
                              : wait_word(qwords + row, tag, max_polls, err, &ok);
  got[row * 9 + lane] = ok ? static_cast<int32_t>(v) : -1;
}

__global__ void front_roles_bump_kernel(unsigned long long* epoch) { *epoch += 1; }

// ---------------------------------------------------------------------------------------------------------------------
// Entries for a fused front + MoE launch (K_FM). Phase 1a: the routing role for one row, the row's 8 words (tag << 32 |
// packed) in the MoE kernel's routed-ids board layout; the topk arrays are optional. Phase 1b: the norm + quant role for one
// token row written straight into permuted row `pos` of the MoE kernel's expert-major token image; no release inside (the
// MoE kernel releases its rows-ready count after its own barrier over the group).
template <int kTopK, int kEPL>
__device__ __forceinline__ void kfm_phase1a(const RouteArgs& a, int token, int lane, uint32_t tag) {
  route_row<kTopK, kEPL>(a, token, lane, tag);
}
template <int64_t kHidden, bool kHasWeight, int kS>
__device__ __forceinline__ void kfm_phase1b(const QuantArgs& a, int token, int pos, int gtid, int bar_id, float* smem) {
  quant_row<kHidden, 256, kHasWeight, false, kS, 1>(a, token, gtid, bar_id, smem, 0u, pos);
}

// Test launches for the entries: 1a over every row (one warp per row), 1b over every permuted row r (token row_tok[r]).
__global__ void __launch_bounds__(256) kfm_1a_test_kernel(const RouteArgs ra, int M, const unsigned long long* epoch,
                                                          int layer) {
  const uint32_t tag = static_cast<uint32_t>(*epoch * 64 + layer);
  const int w = threadIdx.x / 32, lane = threadIdx.x % 32;
  for (int row = blockIdx.x * 8 + w; row < M; row += gridDim.x * 8) kfm_phase1a<8, 8>(ra, row, lane, tag);
}
template <bool kHasWeight, int kS>
__global__ void __launch_bounds__(256) kfm_1b_test_kernel(const QuantArgs qa, const int32_t* row_tok, int nrows) {
  __shared__ float smem[16];
  for (int r = blockIdx.x; r < nrows; r += gridDim.x)
    kfm_phase1b<4096, kHasWeight, kS>(qa, row_tok[r], r, threadIdx.x, 12, smem);
}

// Sanitizer non-vacuity control (tests only): one quant group per CTA with the planted race.
__global__ void __launch_bounds__(256) front_roles_plant_kernel(const QuantArgs qa, int M) {
  __shared__ float smem[16];
  for (int row = blockIdx.x; row < M; row += gridDim.x)
    quant_row<4096, 256, false, true>(qa, row, threadIdx.x, 1, smem, 1u);
}

}  // namespace muse_vr_front

template <int kQThreads, int kGroups, int kRouteWarps>
struct WfcFrontRoles {
  static void run(tvm::ffi::TensorView logits, tvm::ffi::TensorView bias, tvm::ffi::TensorView topk_w,
                  tvm::ffi::TensorView topk_ids, tvm::ffi::TensorView packed, tvm::ffi::TensorView lem, int64_t use_lem,
                  tvm::ffi::TensorView nvalid, int64_t use_nvalid, tvm::ffi::TensorView rwords, tvm::ffi::TensorView x,
                  tvm::ffi::TensorView weight, int64_t has_weight, tvm::ffi::TensorView fp4, tvm::ffi::TensorView sf,
                  tvm::ffi::TensorView pts, tvm::ffi::TensorView qwords, tvm::ffi::TensorView epoch, int64_t layer,
                  int64_t M, int64_t grid, double mult, double route_scale, double div_eps, double qeps, double gsi) {
    using namespace host;
    RuntimeCheck(M >= 1 && grid >= 1 && logits.size(1) == 256 && x.size(1) == 4096, "wfc_front_roles: shapes");
    muse_vr_front::RouteArgs ra{
        .logits = static_cast<const float*>(logits.data_ptr()),
        .bias = static_cast<const float*>(bias.data_ptr()),
        .topk_w = static_cast<float*>(topk_w.data_ptr()),
        .topk_ids = static_cast<int32_t*>(topk_ids.data_ptr()),
        .packed = static_cast<int32_t*>(packed.data_ptr()),
        .local_expert_mapping = use_lem ? static_cast<const int32_t*>(lem.data_ptr()) : nullptr,
        .num_valid_tokens = use_nvalid ? static_cast<const int32_t*>(nvalid.data_ptr()) : nullptr,
        .words = static_cast<unsigned long long*>(rwords.data_ptr()),
        .num_tokens = static_cast<int>(M),
        .num_experts = 256,
        .stride_gm = logits.stride(0),
        .router_output_multiplier = static_cast<float>(mult),
        .route_scale = static_cast<float>(route_scale),
        .division_eps = static_cast<float>(div_eps),
    };
    muse_vr_front::QuantArgs qa{
        .x = x.data_ptr(),
        .weight = has_weight ? weight.data_ptr() : nullptr,
        .fp4_out = fp4.data_ptr(),
        .sf_out = sf.data_ptr(),
        .per_token_scale = static_cast<float*>(pts.data_ptr()),
        .words = static_cast<unsigned long long*>(qwords.data_ptr()),
        .eps = static_cast<float>(qeps),
        .global_scale_inv = static_cast<float>(gsi),
    };
    const cudaStream_t stream = LaunchKernel::resolve_device(x.device());
    const dim3 blocks(static_cast<unsigned>(grid));
    const dim3 threads(kQThreads * kGroups + 32 * kRouteWarps);
    const auto* ep = static_cast<const unsigned long long*>(epoch.data_ptr());
    if (has_weight)
      muse_vr_front::front_roles_test_kernel<kQThreads, kGroups, kRouteWarps, true>
          <<<blocks, threads, 0, stream>>>(ra, qa, static_cast<int>(M), ep, static_cast<int>(layer));
    else
      muse_vr_front::front_roles_test_kernel<kQThreads, kGroups, kRouteWarps, false>
          <<<blocks, threads, 0, stream>>>(ra, qa, static_cast<int>(M), ep, static_cast<int>(layer));
  }

  // Partials form: the norm + quant role reads S fp32 K-slice partials of the pooled rows (partials [S, M, H], contiguous,
  // H = the pooled width the roles are built for) and optionally writes the pooled bf16 rows; the routing role as in run
  // (no mapping / valid count).
  static void run_p(tvm::ffi::TensorView logits, tvm::ffi::TensorView bias, tvm::ffi::TensorView topk_w,
                    tvm::ffi::TensorView topk_ids, tvm::ffi::TensorView packed, tvm::ffi::TensorView rwords,
                    tvm::ffi::TensorView partials, int64_t S, tvm::ffi::TensorView pooled, int64_t want_pooled,
                    tvm::ffi::TensorView weight, int64_t has_weight, tvm::ffi::TensorView fp4, tvm::ffi::TensorView sf,
                    tvm::ffi::TensorView pts, tvm::ffi::TensorView qwords, tvm::ffi::TensorView epoch, int64_t layer, int64_t M,
                    int64_t grid, double mult, double route_scale, double div_eps, double qeps, double gsi) {
    using namespace host;
    RuntimeCheck(M >= 1 && grid >= 1 && S >= 1 && S <= 4 && logits.size(1) == 256, "wfc_front_roles run_p: shapes");
    const int64_t H = partials.size(2);
    RuntimeCheck(H == muse_vr_front::kPooledHidden, "wfc_front_roles run_p: pooled width");
    RuntimeCheck(partials.stride(0) == M * H, "wfc_front_roles run_p: partials layout");
    muse_vr_front::RouteArgs ra{
        .logits = static_cast<const float*>(logits.data_ptr()),
        .bias = static_cast<const float*>(bias.data_ptr()),
        .topk_w = static_cast<float*>(topk_w.data_ptr()),
        .topk_ids = static_cast<int32_t*>(topk_ids.data_ptr()),
        .packed = static_cast<int32_t*>(packed.data_ptr()),
        .local_expert_mapping = nullptr,
        .num_valid_tokens = nullptr,
        .words = static_cast<unsigned long long*>(rwords.data_ptr()),
        .num_tokens = static_cast<int>(M),
        .num_experts = 256,
        .stride_gm = logits.stride(0),
        .router_output_multiplier = static_cast<float>(mult),
        .route_scale = static_cast<float>(route_scale),
        .division_eps = static_cast<float>(div_eps),
    };
    muse_vr_front::QuantArgs qa{
        .x = nullptr,
        .weight = has_weight ? weight.data_ptr() : nullptr,
        .fp4_out = fp4.data_ptr(),
        .sf_out = sf.data_ptr(),
        .per_token_scale = static_cast<float*>(pts.data_ptr()),
        .words = static_cast<unsigned long long*>(qwords.data_ptr()),
        .eps = static_cast<float>(qeps),
        .global_scale_inv = static_cast<float>(gsi),
        .partials = static_cast<const float*>(partials.data_ptr()),
        .pstride = M * H,
        .pooled_out = want_pooled ? pooled.data_ptr() : nullptr,
    };
    const cudaStream_t stream = LaunchKernel::resolve_device(partials.device());
    const dim3 blocks(static_cast<unsigned>(grid));
    const dim3 threads(kQThreads * kGroups + 32 * kRouteWarps);
    const auto* ep = static_cast<const unsigned long long*>(epoch.data_ptr());
    const int m = static_cast<int>(M), ly = static_cast<int>(layer);
#define WFC_FR_LAUNCH(W, SS) \
  muse_vr_front::front_roles_test_kernel<kQThreads, kGroups, kRouteWarps, W, SS><<<blocks, threads, 0, stream>>>(ra, qa, m, ep, ly)
    if (has_weight) {
      if (S == 1) WFC_FR_LAUNCH(true, 1); else if (S == 2) WFC_FR_LAUNCH(true, 2);
      else if (S == 3) WFC_FR_LAUNCH(true, 3); else WFC_FR_LAUNCH(true, 4);
    } else {
      if (S == 1) WFC_FR_LAUNCH(false, 1); else if (S == 2) WFC_FR_LAUNCH(false, 2);
      else if (S == 3) WFC_FR_LAUNCH(false, 3); else WFC_FR_LAUNCH(false, 4);
    }
#undef WFC_FR_LAUNCH
  }

  static void kfm_1a(tvm::ffi::TensorView logits, tvm::ffi::TensorView bias, tvm::ffi::TensorView ids64,
                     tvm::ffi::TensorView epoch, int64_t layer, int64_t M, int64_t grid, double mult, double route_scale,
                     double div_eps) {
    using namespace host;
    RuntimeCheck(M >= 1 && grid >= 1 && logits.size(1) == 256, "wfc_front_roles kfm_1a: shapes");
    muse_vr_front::RouteArgs ra{
        .logits = static_cast<const float*>(logits.data_ptr()),
        .bias = static_cast<const float*>(bias.data_ptr()),
        .topk_w = nullptr,
        .topk_ids = nullptr,
        .packed = nullptr,
        .local_expert_mapping = nullptr,
        .num_valid_tokens = nullptr,
        .words = static_cast<unsigned long long*>(ids64.data_ptr()),
        .num_tokens = static_cast<int>(M),
        .num_experts = 256,
        .stride_gm = logits.stride(0),
        .router_output_multiplier = static_cast<float>(mult),
        .route_scale = static_cast<float>(route_scale),
        .division_eps = static_cast<float>(div_eps),
    };
    const cudaStream_t stream = LaunchKernel::resolve_device(logits.device());
    muse_vr_front::kfm_1a_test_kernel<<<static_cast<unsigned>(grid), 256, 0, stream>>>(
        ra, static_cast<int>(M), static_cast<const unsigned long long*>(epoch.data_ptr()), static_cast<int>(layer));
  }

  // src: bf16 pooled rows [M, H] (S = 0) or fp32 partials [S, M, H] (S = 1..2)
  static void kfm_1b(tvm::ffi::TensorView src, int64_t S, int64_t M, tvm::ffi::TensorView weight, int64_t has_weight,
                     tvm::ffi::TensorView hs_perm, tvm::ffi::TensorView img1, tvm::ffi::TensorView row_img,
                     tvm::ffi::TensorView pts_perm, tvm::ffi::TensorView row_tok, int64_t nrows, int64_t grid, double qeps,
                     double gsi) {
    using namespace host;
    RuntimeCheck(nrows >= 1 && grid >= 1 && S >= 0 && S <= 2, "wfc_front_roles kfm_1b: shapes");
    RuntimeCheck((S == 0 ? src.size(1) : src.size(2)) == muse_vr_front::kPooledHidden, "wfc_front_roles kfm_1b: pooled width");
    muse_vr_front::QuantArgs qa{};
    qa.x = S == 0 ? src.data_ptr() : nullptr;
    qa.weight = has_weight ? weight.data_ptr() : nullptr;
    qa.eps = static_cast<float>(qeps);
    qa.global_scale_inv = static_cast<float>(gsi);
    qa.partials = S > 0 ? static_cast<const float*>(src.data_ptr()) : nullptr;
    qa.pstride = S > 0 ? M * src.size(2) : 0;
    qa.hs_perm = static_cast<uint8_t*>(hs_perm.data_ptr());
    qa.img1 = static_cast<uint8_t*>(img1.data_ptr());
    qa.row_img = static_cast<const int32_t*>(row_img.data_ptr());
    qa.pts_perm = static_cast<float*>(pts_perm.data_ptr());
    const cudaStream_t stream = LaunchKernel::resolve_device(hs_perm.device());
    const auto* rt = static_cast<const int32_t*>(row_tok.data_ptr());
    const unsigned g = static_cast<unsigned>(grid);
    const int nr = static_cast<int>(nrows);
#define WFC_KFM_1B(W, SS) muse_vr_front::kfm_1b_test_kernel<W, SS><<<g, 256, 0, stream>>>(qa, rt, nr)
    if (has_weight) {
      if (S == 0) WFC_KFM_1B(true, 0); else if (S == 1) WFC_KFM_1B(true, 1); else WFC_KFM_1B(true, 2);
    } else {
      if (S == 0) WFC_KFM_1B(false, 0); else if (S == 1) WFC_KFM_1B(false, 1); else WFC_KFM_1B(false, 2);
    }
#undef WFC_KFM_1B
  }

  static void wait(tvm::ffi::TensorView rwords, tvm::ffi::TensorView qwords, int64_t M, tvm::ffi::TensorView epoch,
                   int64_t layer, int64_t max_polls, tvm::ffi::TensorView err, tvm::ffi::TensorView got) {
    using namespace host;
    const cudaStream_t stream = LaunchKernel::resolve_device(rwords.device());
    muse_vr_front::front_roles_wait_kernel<<<static_cast<unsigned>((M + 3) / 4), 128, 0, stream>>>(
        static_cast<const unsigned long long*>(rwords.data_ptr()), static_cast<const unsigned long long*>(qwords.data_ptr()),
        static_cast<int>(M), static_cast<const unsigned long long*>(epoch.data_ptr()), static_cast<int>(layer), max_polls,
        static_cast<unsigned long long*>(err.data_ptr()), static_cast<int32_t*>(got.data_ptr()));
  }

  static void plant(tvm::ffi::TensorView x, tvm::ffi::TensorView fp4, tvm::ffi::TensorView sf, tvm::ffi::TensorView pts,
                    int64_t M) {
    using namespace host;
    muse_vr_front::QuantArgs qa{.x = x.data_ptr(), .weight = nullptr, .fp4_out = fp4.data_ptr(), .sf_out = sf.data_ptr(),
                                .per_token_scale = static_cast<float*>(pts.data_ptr()), .words = nullptr, .eps = 1e-6f,
                                .global_scale_inv = 1.0f};
    const cudaStream_t stream = LaunchKernel::resolve_device(x.device());
    muse_vr_front::front_roles_plant_kernel<<<static_cast<unsigned>(M), 256, 0, stream>>>(qa, static_cast<int>(M));
  }

  static void bump(tvm::ffi::TensorView epoch) {
    using namespace host;
    const cudaStream_t stream = LaunchKernel::resolve_device(epoch.device());
    muse_vr_front::front_roles_bump_kernel<<<1, 1, 0, stream>>>(static_cast<unsigned long long*>(epoch.data_ptr()));
  }
};

}  // namespace sglang
