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
// Modified by NVIDIA: route_row_sel, a form of the routing role (wfc_front_roles.cuh route_row) whose selection rounds do
// not wait on the winner's broadcast: the pops of round k depend only on round k's maximum, and the winner's id / score
// broadcast runs beside the next rounds. Rows on which that shortcut is not exact (heads tied at a round maximum, or a NaN
// key) are recomputed by route_row, so the outputs equal route_row's for every input.
//
// Selection order of route_row (and of SGLang's routing op): descending key (biased score), ties to the lowest expert id.
// Each lane holds kEPL experts in a stable per-lane order (descending key, then ascending slot = ascending expert id), so a
// lane's head is its best remaining expert. route_row's round k: m = warp max of the heads; among the heads equal to m, the
// lowest expert id wins (lowest group, then lowest lane); only the winner's lane pops; the winner's score joins the sum.
// route_row_sel's round k: m = the same warp max (same instruction); every lane whose head equals m pops. When exactly one
// head equals m (ids are distinct across lanes), that head IS route_row's winner and the pops are route_row's pops. A ballot
// of the heads equal to m (beside the rounds) gives the winner lane for the id / score broadcast and flags any round with
// more than one such head; the flag (warp-uniform) and a warp vote on NaN keys send the row to route_row.

#pragma once

#include "wfc_front_roles.cuh"

namespace sglang {
namespace muse_vr_front {

// The recompute path, out of line: it runs only on rows the selection rounds cannot settle, so its code stays out of the
// common path's instruction stream.
template <int kTopK, int kEPL>
__device__ __noinline__ void route_row_cold(const RouteArgs& a, int token, int lane, uint32_t tag) {
  route_row<kTopK, kEPL>(a, token, lane, tag);
}

// kForm 1: the winner's id / score broadcast inside each round (beside the next round's maximum).
// kForm 2: each lane keeps the head it popped per round; the broadcasts run after the last round (no shuffle in the rounds).
template <int kTopK, int kEPL, int kForm = 2>
__device__ __forceinline__ void route_row_sel(const RouteArgs& a, int token, int lane, uint32_t tag) {
  using namespace device;
  constexpr int kVecsPerLane = kEPL / 4;
  constexpr int kGroupExperts = 4 * kWarpThreads;
  static_assert(kEPL % 4 == 0 && kTopK >= 1 && kTopK <= 32);
  if (a.num_valid_tokens != nullptr && token >= *a.num_valid_tokens) {
    route_row_cold<kTopK, kEPL>(a, token, lane, tag);  // the invalid-row path (no selection)
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
  bool has_nan = false;
#pragma unroll
  for (int s = 0; s < kEPL; ++s) {
    if (expert[s] < a.num_experts) {
      key[s] = biased[s] + score[s];
    } else {
      score[s] = 0.0f;
      key[s] = kNegInf;
    }
    has_nan |= key[s] != key[s];
  }
  // Stable odd-even sort per lane (equal keys keep slot order; slots are monotone in expert id), as route_row.
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
  unsigned multi = 0u;  // nonzero iff some round had more than one head equal to its maximum (warp-uniform)
  unsigned heads_k[kTopK];
  int won_e[kTopK];
  float won_s[kTopK];
#pragma unroll
  for (int k = 0; k < kTopK; ++k) {
    const float round_max = warp_max_f32(key[0]);
    const bool pop = key[0] == round_max;
    // beside the rounds: the winner lane (unique when multi stays 0), its id and score
    const unsigned heads = __ballot_sync(warp::kFullMask, pop);
    multi |= heads & (heads - 1u);
    if constexpr (kForm == 1) {
      const int owner = (__ffs(heads) - 1) & (kWarpThreads - 1);
      const int winner = __shfl_sync(warp::kFullMask, expert[0], owner);
      const float selected = __shfl_sync(warp::kFullMask, score[0], owner);
      score_sum += selected;
      if (lane == k) {
        out_id = winner;
        out_score = selected;
      }
    } else {
      heads_k[k] = heads;
      won_e[k] = expert[0];
      won_s[k] = score[0];
    }
#pragma unroll
    for (int s = 0; s + 1 < kEPL; ++s)
      if (s + k <= kTopK - 2) {
        key[s] = pop ? key[s + 1] : key[s];
        expert[s] = pop ? expert[s + 1] : expert[s];
        score[s] = pop ? score[s + 1] : score[s];
      }
    if (kEPL - 1 + k <= kTopK - 2) key[kEPL - 1] = pop ? kNegInf : key[kEPL - 1];
  }
  if constexpr (kForm != 1) {
    // round k's winner = the head lane k's ballot names; its expert / score as that lane held them in round k
#pragma unroll
    for (int k = 0; k < kTopK; ++k) {
      const int owner = (__ffs(heads_k[k]) - 1) & (kWarpThreads - 1);
      const int winner = __shfl_sync(warp::kFullMask, won_e[k], owner);
      const float selected = __shfl_sync(warp::kFullMask, won_s[k], owner);
      score_sum += selected;
      if (lane == k) {
        out_id = winner;
        out_score = selected;
      }
    }
  }
#ifndef MUSE_VR_C4_ROUTE_TEST_NO_FALLBACK
  if (__any_sync(warp::kFullMask, has_nan) || multi != 0u) {  // warp-uniform: recompute the row as route_row does
    route_row_cold<kTopK, kEPL>(a, token, lane, tag);
    return;
  }
#else  // test-only build: the recompute dropped (mutation control: the units must fail on tied / NaN rows)
  (void)has_nan;
  (void)multi;
#endif
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

// route_row_sel with the scores in a shared-memory table: the per-lane sort and the rounds carry (key, expert) only; the
// selected scores are read back by expert id after the rounds. sm_score: kEPL * 32 floats private to this warp (indexed
// by expert id), written by each lane for its own experts before a __syncwarp and read after it (PTX bar.warp.sync orders
// the warp's shared-memory accesses). Same selection rounds, same recompute rule, same outputs as route_row.
template <int kTopK, int kEPL>
__device__ __forceinline__ void route_row_sel_sm(const RouteArgs& a, int token, int lane, uint32_t tag, float* sm_score) {
  using namespace device;
  constexpr int kVecsPerLane = kEPL / 4;
  constexpr int kGroupExperts = 4 * kWarpThreads;
  static_assert(kEPL % 4 == 0 && kTopK >= 1 && kTopK <= 32);
  if (a.num_valid_tokens != nullptr && token >= *a.num_valid_tokens) {
    route_row_cold<kTopK, kEPL>(a, token, lane, tag);  // the invalid-row path (no selection)
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
  bool has_nan = false;
#pragma unroll
  for (int s = 0; s < kEPL; ++s) {
    if (expert[s] < a.num_experts) {
      key[s] = biased[s] + score[s];
    } else {
      score[s] = 0.0f;
      key[s] = kNegInf;
    }
    has_nan |= key[s] != key[s];
  }
#pragma unroll
  for (int v = 0; v < kVecsPerLane; ++v)  // scores by expert id (slot order = expert order)
    *reinterpret_cast<float4*>(sm_score + v * kGroupExperts + lane * 4) =
        make_float4(score[v * 4 + 0], score[v * 4 + 1], score[v * 4 + 2], score[v * 4 + 3]);
  // Stable odd-even sort per lane on (key, expert), as route_row orders (key, expert, score).
#pragma unroll
  for (int r = 0; r < kEPL; ++r) {
#pragma unroll
    for (int i = r % 2; i + 1 < kEPL; i += 2) {
      const bool swap = key[i + 1] > key[i];
      const float k0 = swap ? key[i + 1] : key[i];
      const float k1 = swap ? key[i] : key[i + 1];
      const int e0 = swap ? expert[i + 1] : expert[i];
      const int e1 = swap ? expert[i] : expert[i + 1];
      key[i] = k0;
      key[i + 1] = k1;
      expert[i] = e0;
      expert[i + 1] = e1;
    }
  }
  unsigned multi = 0u;
  unsigned heads_k[kTopK];
  int won_e[kTopK];
#pragma unroll
  for (int k = 0; k < kTopK; ++k) {
    const float round_max = warp_max_f32(key[0]);
    const bool pop = key[0] == round_max;
    const unsigned heads = __ballot_sync(warp::kFullMask, pop);
    multi |= heads & (heads - 1u);
    heads_k[k] = heads;
    won_e[k] = expert[0];
#pragma unroll
    for (int s = 0; s + 1 < kEPL; ++s)
      if (s + k <= kTopK - 2) {
        key[s] = pop ? key[s + 1] : key[s];
        expert[s] = pop ? expert[s + 1] : expert[s];
      }
    if (kEPL - 1 + k <= kTopK - 2) key[kEPL - 1] = pop ? kNegInf : key[kEPL - 1];
  }
#ifndef MUSE_VR_C4_ROUTE_TEST_NO_FALLBACK
  if (__any_sync(warp::kFullMask, has_nan) || multi != 0u) {  // warp-uniform: recompute the row as route_row does
    route_row_cold<kTopK, kEPL>(a, token, lane, tag);
    return;
  }
#else  // test-only build: the recompute dropped (mutation control: the units must fail on tied / NaN rows)
  (void)has_nan;
  (void)multi;
#endif
  int winner[kTopK];
#pragma unroll
  for (int k = 0; k < kTopK; ++k)
    winner[k] = __shfl_sync(warp::kFullMask, won_e[k], (__ffs(heads_k[k]) - 1) & (kWarpThreads - 1));
  __syncwarp();  // every lane's score-table stores precede the reads below
  float score_sum = 0.0f;
  int out_id = 0;
  float out_score = 0.0f;
#pragma unroll
  for (int k = 0; k < kTopK; ++k) {
    const float selected = sm_score[winner[k]];
    score_sum += selected;
    if (lane == k) {
      out_id = winner[k];
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

}  // namespace muse_vr_front
}  // namespace sglang
