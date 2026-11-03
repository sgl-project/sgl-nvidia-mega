// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// Portions Copyright 2023-2024 SGLang Team (Apache-2.0)
// Derived from SGLang (https://github.com/sgl-project/sglang), Apache License 2.0; see ../include/LICENSE.sglang.
// Sources: SGLang python/sglang/kernels/jit/csrc/muse_spark_v12/norm_residual.cuh and distributed/custom_all_reduce.cuh
// (2shot_push) at revision a55ab7d, through this package's csrc/ar_norm_residual.cuh.
// Modified by NVIDIA: one kernel performs a two-shot push all-reduce of the bf16 sublayer partials and the post-norm +
// gated residual + pre-norm boundary on every row. Rows are split into one contiguous chunk per rank; each rank reduces
// the rows it owns (lamport push, fp32 accumulation in rank order, bf16 rounding, the same empty-marker remap of the
// reduced vector as 2shot_push) and pushes the sums to every peer; every rank then runs the boundary arithmetic on every
// row with the stock boundary kernel's per-row thread map. Slot polls are bounded (trap after kMaxPolls polls).
//
// partial:    [M, H] bf16, this rank's partial sum of the sublayer output (row-parallel GEMM output); M % world == 0.
// stream:     [M, H] fp32, the residual stream (identical on every rank).
// output:     [M, H] bf16, pre-norm of the new stream.
// new_stream: [M, H] fp32.
// router_out: [M, H] fp32 copy of output, or null.
//
// Exchange: slot (dst, src) of the current phase holds two halves of chunk_vecs 16-byte vectors: [0, chunk) takes src's
// partial of dst's rows, [chunk, 2 * chunk) takes src's reduced rows. The block (or cluster) that handles a row on one
// rank handles the same row on every rank, so a block only waits on peers' blocks, never on a block of its own grid.

#include <sgl_kernel/distributed/communicator.cuh>

#include <tvm/ffi/container/tensor.h>
#include <tvm/ffi/optional.h>

#include "muse_spark_v12/norm_residual.cuh"

#include <algorithm>
#include <cstdint>

namespace sglang {

namespace muse_vr_ar2_norm {

template <uint32_t kWorld>
struct Params {
  device::distributed::PushWorkSpace<kWorld> ws;
  uint32_t num_counters;   // push plane counter array length
  uint32_t rank;
  uint32_t num_rows;
  uint32_t rows_per_rank;  // num_rows / kWorld
  uint32_t chunk_vecs;     // rows_per_rank * vectors per row: start of the gathered half of a slot
  const bf16_t* partial;   // [M, H]
  NormResidualParams norm;
};

using BF16Vec = device::AlignedVector<bf16x2_t, device::norm_residual::kPairs>;
using Lamport = device::distributed::LamportTrait<bf16_t, 8, /*kAtom=*/4>;

// Sanitizer negative controls (test builds only; MUSE_VR_AR2_NEGCTL is never defined in a served build): each makes one
// sanitizer tool report an error so a clean run proves the tool instrumented this kernel. 1: invalid global write (memcheck),
// 2: shared-memory write-write race (racecheck), 3: barriers reached from divergent paths of a warp (synccheck), 4: read of uninitialized global
// memory (initcheck; the router output buffer, read before this kernel writes it).
#ifndef MUSE_VR_AR2_NEGCTL
#define MUSE_VR_AR2_NEGCTL 0
#endif

SGL_DEVICE void negative_control(const void* uninit) {
  if constexpr (MUSE_VR_AR2_NEGCTL == 1) {
    if (blockIdx.x == 0 && threadIdx.x == 0) *reinterpret_cast<volatile int*>(static_cast<uintptr_t>(64)) = 1;
  } else if constexpr (MUSE_VR_AR2_NEGCTL == 2) {
    __shared__ volatile int race_word;  // volatile: the racing stores must survive dead-store elimination
    if (threadIdx.x < 64) race_word = static_cast<int>(threadIdx.x);
    __syncthreads();
  } else if constexpr (MUSE_VR_AR2_NEGCTL == 3) {
    if ((threadIdx.x & 31) < 16) {  // every thread reaches a barrier, from two divergent paths of one warp
      __syncthreads();
    } else {
      __syncthreads();
    }
  } else if constexpr (MUSE_VR_AR2_NEGCTL == 4) {
    if (blockIdx.x == 0 && threadIdx.x == 0 && uninit != nullptr) {
      __shared__ volatile float sink;
      sink = *static_cast<const volatile float*>(uninit);
    }
  }
}

// The phase of this round; block 0 also advances the counters of blocks this grid does not launch, so every counter
// holds the same phase for the next push kernel on this plane.
template <uint32_t kWorld>
SGL_DEVICE uint32_t begin_phase(const Params<kWorld>& params) {
  negative_control(params.norm.router_out);  // no-op unless a test build defines MUSE_VR_AR2_NEGCTL
  const uint32_t phase = params.ws.counter[blockIdx.x].get() & 1;
  if (blockIdx.x == 0) {
    for (uint32_t i = gridDim.x + threadIdx.x; i < params.num_counters; i += blockDim.x) {
      params.ws.counter[i].set(phase ^ 1);
    }
  }
  return phase;
}

template <uint32_t kWorld>
SGL_DEVICE uint8_t* slot_ptr(const Params<kWorld>& params, uint32_t phase, uint32_t dst, uint32_t src) {
  return params.ws.workspaces[dst] + (src + phase * kWorld) * params.ws.slot_bytes;
}

template <uint32_t kWorld>
SGL_DEVICE void end_phase(const Params<kWorld>& params, uint32_t phase) {
  __syncthreads();
  if (threadIdx.x == 0) params.ws.counter[blockIdx.x].set(phase ^ 1);
}

// Upper bound on polls of one vector's slots. A peer that never delivers (a crashed or diverged rank) then ends the
// kernel with a trap, which surfaces as a CUDA error, instead of leaving the process spinning.
inline constexpr uint32_t kMaxPolls = 1u << 24;

__device__ __noinline__ void poll_bound_exceeded() {
  __trap();
}

// Non-owner: this rank's partial of global vector `vid` to the owner's slot, partial half, local vector `lvid`.
template <uint32_t kWorld>
SGL_DEVICE void push_partial(const Params<kWorld>& params, uint32_t phase, uint32_t owner, uint32_t vid, uint32_t lvid) {
  BF16Vec vec;
  vec.load(params.partial, vid);
  Lamport::clear_pos_zero(vec.data());
  device::ptx::st_relaxed_16B(vec, slot_ptr(params, phase, owner, params.rank), lvid);
}

// Owner: waits for every peer's partial of `vid`, reduces over ranks 0..kWorld-1 in fp32 (bf16 rounding), remaps the
// empty marker in the sum (as 2shot_push does before its local store, so every rank holds identical bits), pushes the
// sum into every peer's gathered half and restores the empty marker in its own partial slots.
template <uint32_t kWorld>
SGL_DEVICE BF16Vec reduce_and_share(const Params<kWorld>& params, uint32_t phase, uint32_t vid, uint32_t lvid) {
  BF16Vec vec[kWorld];
#pragma unroll
  for (uint32_t i = 0; i < kWorld; ++i) {
    if (i == params.rank) {
      vec[i].load(params.partial, vid);
      Lamport::clear_pos_zero(vec[i].data());
    }
  }
  uint32_t polls = 0;
  do {
    bool pending = false;
#pragma unroll
    for (uint32_t i = 0; i < kWorld; ++i) {
      if (i != params.rank) device::ptx::ld_relaxed_16B(vec[i], slot_ptr(params, phase, params.rank, i), lvid);
    }
#pragma unroll
    for (uint32_t i = 0; i < kWorld; ++i) {
      if (i != params.rank) pending |= Lamport::has_pos_zero(vec[i].data());
    }
    if (!pending) break;
    if (++polls == kMaxPolls) [[unlikely]] poll_bound_exceeded();
  } while (true);
  BF16Vec out = device::reduce_vec(vec);
  Lamport::clear_pos_zero(out.data());
#pragma unroll
  for (uint32_t i = 0; i < kWorld; ++i) {
    if (i != params.rank) device::ptx::st_relaxed_16B(out, slot_ptr(params, phase, i, params.rank), params.chunk_vecs + lvid);
  }
  BF16Vec empty;
  Lamport::fill_pos_zero(empty.data());
#pragma unroll
  for (uint32_t i = 0; i < kWorld; ++i) {
    if (i != params.rank) device::ptx::st_global_16B(empty, slot_ptr(params, phase, params.rank, i), lvid);
  }
  return out;
}

// Non-owner: waits for the owner's sum of local vector `lvid` and restores the empty marker in that gathered slot.
template <uint32_t kWorld>
SGL_DEVICE BF16Vec receive_sum(const Params<kWorld>& params, uint32_t phase, uint32_t owner, uint32_t lvid) {
  BF16Vec vec;
  uint8_t* const slot = slot_ptr(params, phase, params.rank, owner);
  uint32_t polls = 0;
  do {
    device::ptx::ld_relaxed_16B(vec, slot, params.chunk_vecs + lvid);
    if (!Lamport::has_pos_zero(vec.data())) break;
    if (++polls == kMaxPolls) [[unlikely]] poll_bound_exceeded();
  } while (true);
  BF16Vec empty;
  Lamport::fill_pos_zero(empty.data());
  device::ptx::st_global_16B(empty, slot, params.chunk_vecs + lvid);
  return vec;
}

template <uint32_t kWorld>
SGL_DEVICE BF16Vec row_sum_vector(
    const Params<kWorld>& params, uint32_t phase, uint32_t owner, uint32_t vid, uint32_t lvid) {
  return owner == params.rank ? reduce_and_share(params, phase, vid, lvid) : receive_sum(params, phase, owner, lvid);
}

}  // namespace muse_vr_ar2_norm

// C > 1: one cluster per row, same arithmetic as muse_spark_norm_residual_cluster.
template <
    int64_t kHidden,
    uint32_t kWorld,
    int kNumThreads,
    int kClusterSize,
    bool kUsePDL,
    bool kPrefetchWeights,
    bool kPostNormIsInputScale,
    bool kHasRouter,
    bool kPostWeight,
    bool kGate>
__global__ void __launch_bounds__(kNumThreads)
    muse_vr_ar2_norm_residual_cluster(const __grid_constant__ muse_vr_ar2_norm::Params<kWorld> params) {
  using namespace device;
  using namespace device::norm_residual;
  using namespace device::norm_residual_cluster;

  constexpr int kTotalThreads = kNumThreads * kClusterSize;
  constexpr int kPacks = kHidden / (kTotalThreads * kVec);
  constexpr int kNumWarps = kNumThreads / kWarpThreads;
  constexpr int kPreSlots = kPrefetchWeights ? kPacks : 1;
  constexpr float kInvHidden = 1.0f / static_cast<float>(kHidden);
  constexpr uint32_t kRowVecs = static_cast<uint32_t>(kHidden / kVec);
  static_assert(kClusterSize == 2 || kClusterSize == 4 || kClusterSize == 8 || kClusterSize == 16);
  static_assert(kHidden % (kTotalThreads * kVec) == 0);

  using BF16Vec = AlignedVector<bf16x2_t, kPairs>;
  using F32Vec = AlignedVector<fp32x2_t, kFPairs>;

  const auto& norm = params.norm;
  const int cluster_rank = cooperative_groups::this_cluster().block_rank();
  const int row = static_cast<int>(blockIdx.x) / kClusterSize;
  const uint32_t owner = static_cast<uint32_t>(row) / params.rows_per_rank;
  const uint32_t lrow = static_cast<uint32_t>(row) - owner * params.rows_per_rank;
  const int64_t base = static_cast<int64_t>(row) * kHidden;
  const auto stream = static_cast<const fp32_t*>(norm.stream) + base;
  const auto out = static_cast<bf16_t*>(norm.output) + base;
  const auto new_stream = static_cast<fp32_t*>(norm.new_stream) + base;
  const auto router = static_cast<fp32_t*>(norm.router_out) + (kHasRouter ? base : 0);
  const auto post_weight = static_cast<const bf16_t*>(norm.post_weight);
  const auto pre_weight = static_cast<const bf16_t*>(norm.pre_weight);
  const auto alpha_ptr = static_cast<const fp32_t*>(norm.alpha);
  const auto beta_ptr = static_cast<const fp32_t*>(norm.beta);
  const bool channel_wise = kGate && norm.gate_channel_wise;
  const bool has_pre_weight = norm.has_pre_weight;

  __shared__ ReductionStorage4<kClusterSize, kNumWarps> reduction;
  if (threadIdx.x == 0) {
    ptx::mbar_init(&reduction.barrier, 1);
    ptx::mbar_arrive_expect_tx(&reduction.barrier, (kClusterSize - 1) * sizeof(float4));
    ptx::publish_cluster_barriers();
  }
  ptx::cluster_arrive_relaxed();

  fp32x2_t values[kPacks][kPairs];
  fp32x2_t post_w[kPostWeight ? kPacks : 1][kPairs];
  fp32x2_t residual_terms[kPacks][kPairs];
  fp32x2_t beta_values[kPacks][kPairs];
  BF16Vec pre_w_pf[kPreSlots];

  PDLWaitPrimary<kUsePDL>();
  const uint32_t phase = muse_vr_ar2_norm::begin_phase(params);
  if (owner != params.rank) {
#pragma unroll
    for (int pack = 0; pack < kPacks; ++pack) {
      const uint32_t index = static_cast<uint32_t>(cluster_rank * kNumThreads + threadIdx.x + pack * kTotalThreads);
      muse_vr_ar2_norm::push_partial(params, phase, owner, static_cast<uint32_t>(row) * kRowVecs + index,
                                     lrow * kRowVecs + index);
    }
  }

  float branch_sum = 0.0f;
  float residual_sum = 0.0f;
  float cross_sum = 0.0f;
  float scaled_branch_sum = 0.0f;
  fp32x2_t gate_a = {1.0f, 1.0f};
  fp32x2_t gate_b = {1.0f, 1.0f};
  if constexpr (kGate) {
    if (!channel_wise) {
      gate_a.x = gate_a.y = *alpha_ptr;
      gate_b.x = gate_b.y = *beta_ptr;
    }
  }
#pragma unroll
  for (int pack = 0; pack < kPacks; ++pack) {
    const int index = cluster_rank * kNumThreads + threadIdx.x + pack * kTotalThreads;
    BF16Vec post_weights;
    F32Vec stream_values[kFParts];
    F32Vec alpha_values[kFParts];
    F32Vec beta_vector[kFParts];
    if constexpr (kPostWeight) post_weights.load(post_weight, index);
    if constexpr (kPrefetchWeights) {
      if (has_pre_weight) pre_w_pf[pack].load(pre_weight, index);
    }
#pragma unroll
    for (int part = 0; part < kFParts; ++part) {
      const int float_index = index * kFParts + part;
      stream_values[part].load(stream, float_index);
      if constexpr (kGate) {
        if (channel_wise) {
          alpha_values[part].load(alpha_ptr, float_index);
          beta_vector[part].load(beta_ptr, float_index);
        }
      }
    }
    const BF16Vec branch_values = muse_vr_ar2_norm::row_sum_vector(
        params, phase, owner, static_cast<uint32_t>(row) * kRowVecs + index, lrow * kRowVecs + index);

#pragma unroll
    for (int pair = 0; pair < kPairs; ++pair) {
      const int part = pair / kFPairs;
      const int lane = pair % kFPairs;
      auto value = cast<fp32x2_t>(branch_values[pair]);
      if constexpr (kPostWeight) {
        const auto weight = cast<fp32x2_t>(post_weights[pair]);
        post_w[pack][pair] = weight;
        if constexpr (kPostNormIsInputScale) {
          value.x *= weight.x;
          value.y *= weight.y;
          value = round_bf16(value);
        }
      }
      auto moment_value = value;
      if constexpr (!kPostNormIsInputScale && kPostWeight) {
        const auto weight = post_w[pack][pair];
        moment_value.x *= weight.x;
        moment_value.y *= weight.y;
      }
      const auto stream_value = stream_values[part][lane];
      auto alpha_value = gate_a;
      auto beta_value = gate_b;
      if constexpr (kGate) {
        if (channel_wise) {
          alpha_value = alpha_values[part][lane];
          beta_value = beta_vector[part][lane];
        }
      }
      const fp32x2_t residual = {
          alpha_value.x * stream_value.x,
          alpha_value.y * stream_value.y,
      };
      const fp32x2_t scaled_branch = {
          beta_value.x * moment_value.x,
          beta_value.y * moment_value.y,
      };
      values[pack][pair] = value;
      residual_terms[pack][pair] = residual;
      beta_values[pack][pair] = beta_value;
      branch_sum = fmaf(value.x, value.x, branch_sum);
      branch_sum = fmaf(value.y, value.y, branch_sum);
      residual_sum = fmaf(residual.x, residual.x, residual_sum);
      residual_sum = fmaf(residual.y, residual.y, residual_sum);
      cross_sum = fmaf(residual.x, scaled_branch.x, cross_sum);
      cross_sum = fmaf(residual.y, scaled_branch.y, cross_sum);
      scaled_branch_sum = fmaf(scaled_branch.x, scaled_branch.x, scaled_branch_sum);
      scaled_branch_sum = fmaf(scaled_branch.y, scaled_branch.y, scaled_branch_sum);
    }
  }
  PDLTriggerSecondary<kUsePDL>();

  local_reduce(branch_sum, residual_sum, cross_sum, scaled_branch_sum, reduction);
  ptx::cluster_wait();
  dsmem_reduce(branch_sum, residual_sum, cross_sum, scaled_branch_sum, reduction, cluster_rank);
  const float post_rstd = math::rsqrt(branch_sum * kInvHidden + norm.post_eps);
  const float pre_sum = fmaf(post_rstd, fmaf(post_rstd, scaled_branch_sum, 2.0f * cross_sum), residual_sum);
  const float pre_rstd = math::rsqrt(fmaxf(pre_sum, 0.0f) * kInvHidden + norm.pre_eps);

#pragma unroll
  for (int pack = 0; pack < kPacks; ++pack) {
    const int index = cluster_rank * kNumThreads + threadIdx.x + pack * kTotalThreads;
    BF16Vec output_values;
    BF16Vec weight_v;
    F32Vec new_stream_values[kFParts];
    F32Vec router_values[kFParts];
    if (has_pre_weight) {
      if constexpr (kPrefetchWeights)
        weight_v = pre_w_pf[pack];
      else
        weight_v.load(pre_weight, index);
    }
#pragma unroll
    for (int pair = 0; pair < kPairs; ++pair) {
      const int part = pair / kFPairs;
      const int lane = pair % kFPairs;
      auto normed = values[pack][pair];
      normed.x *= post_rstd;
      normed.y *= post_rstd;
      if constexpr (!kPostNormIsInputScale && kPostWeight) {
        const auto weight = post_w[pack][pair];
        normed.x *= weight.x;
        normed.y *= weight.y;
      }
      normed = round_bf16(normed);
      const auto beta_value = beta_values[pack][pair];
      const auto residual = residual_terms[pack][pair];
      const fp32x2_t updated = {
          fmaf(beta_value.x, normed.x, residual.x),
          fmaf(beta_value.y, normed.y, residual.y),
      };
      new_stream_values[part][lane] = updated;
      fp32x2_t normalized = {updated.x * pre_rstd, updated.y * pre_rstd};
      if (has_pre_weight) {
        const auto weight = cast<fp32x2_t>(weight_v[pair]);
        normalized.x *= weight.x;
        normalized.y *= weight.y;
      }
      output_values[pair] = cast<bf16x2_t>(normalized);
      if constexpr (kHasRouter) router_values[part][lane] = normalized;
    }
    output_values.store(out, index);
#pragma unroll
    for (int part = 0; part < kFParts; ++part) {
      const int float_index = index * kFParts + part;
      new_stream_values[part].store(new_stream, float_index);
      if constexpr (kHasRouter) router_values[part].store(router, float_index);
    }
  }

  muse_vr_ar2_norm::end_phase(params, phase);
}

// C == 1: grid-stride over rows (the grid is capped at the push plane's counter count); per-row arithmetic is that
// of muse_spark_norm_residual_kernel.
template <
    int64_t kHidden,
    uint32_t kWorld,
    int kNumThreads,
    bool kUsePDL,
    bool kPrefetchWeights,
    bool kPostNormIsInputScale,
    bool kHasRouter,
    bool kPostWeight,
    bool kGate>
__global__ void __launch_bounds__(kNumThreads)
    muse_vr_ar2_norm_residual_row(const __grid_constant__ muse_vr_ar2_norm::Params<kWorld> params) {
  using namespace device;
  using namespace device::norm_residual;

  constexpr int kPacks = kHidden / (kNumThreads * kVec);
  constexpr int kNumWarps = kNumThreads / kWarpThreads;
  constexpr int kPreSlots = kPrefetchWeights ? kPacks : 1;
  constexpr uint32_t kRowVecs = static_cast<uint32_t>(kHidden / kVec);
  static_assert(kHidden % (kNumThreads * kVec) == 0);

  using BF16Vec = AlignedVector<bf16x2_t, kPairs>;
  using F32Vec = AlignedVector<fp32x2_t, kFPairs>;

  const auto& norm = params.norm;
  const auto post_weight = static_cast<const bf16_t*>(norm.post_weight);
  const auto pre_weight = static_cast<const bf16_t*>(norm.pre_weight);
  const auto alpha_ptr = static_cast<const fp32_t*>(norm.alpha);
  const auto beta_ptr = static_cast<const fp32_t*>(norm.beta);

  __shared__ float smem[kNumWarps == 1 ? 1 : 2 * kNumWarps];

  const bool channel_wise = kGate && norm.gate_channel_wise;
  const bool has_pre_weight = norm.has_pre_weight;
  const bool pre_bf16 = norm.pre_norm_bf16;
  const uint32_t rows_per_rank = params.rows_per_rank;

  PDLWaitPrimary<kUsePDL>();
  const uint32_t phase = muse_vr_ar2_norm::begin_phase(params);
  // Reduce-scatter: push this rank's partial of every row this block handles and does not own, before any wait.
  for (uint32_t row = blockIdx.x; row < params.num_rows; row += gridDim.x) {
    const uint32_t owner = row / rows_per_rank;
    if (owner == params.rank) continue;
    const uint32_t lrow = row - owner * rows_per_rank;
#pragma unroll
    for (int p = 0; p < kPacks; ++p) {
      const uint32_t idx = threadIdx.x + p * kNumThreads;
      muse_vr_ar2_norm::push_partial(params, phase, owner, row * kRowVecs + idx, lrow * kRowVecs + idx);
    }
  }

  for (uint32_t row = blockIdx.x; row < params.num_rows; row += gridDim.x) {
    const uint32_t owner = row / rows_per_rank;
    const uint32_t lrow = row - owner * rows_per_rank;
    const int64_t base = static_cast<int64_t>(row) * kHidden;
    const auto stream = static_cast<const fp32_t*>(norm.stream) + base;
    const auto out = static_cast<bf16_t*>(norm.output) + base;
    const auto new_stream = static_cast<fp32_t*>(norm.new_stream) + base;
    const auto router = static_cast<fp32_t*>(norm.router_out) + (kHasRouter ? base : 0);

    fp32x2_t vals[kPacks][kPairs];
    fp32x2_t post_w[kPostWeight ? kPacks : 1][kPairs];
    F32Vec stream_v[kPacks][kFParts];
    F32Vec alpha_pf[kPreSlots][kFParts], beta_pf[kPreSlots][kFParts];
    BF16Vec pre_w_pf[kPreSlots];

    float sum_sq = 0.0f;
#pragma unroll
    for (int p = 0; p < kPacks; ++p) {
      const int idx = threadIdx.x + p * kNumThreads;
#pragma unroll
      for (int j = 0; j < kFParts; ++j)
        stream_v[p][j].load(stream, idx * kFParts + j);
      if constexpr (kPrefetchWeights) {
        if (has_pre_weight) pre_w_pf[p].load(pre_weight, idx);
        if constexpr (kGate) {
          if (channel_wise) {
#pragma unroll
            for (int j = 0; j < kFParts; ++j) {
              alpha_pf[p][j].load(alpha_ptr, idx * kFParts + j);
              beta_pf[p][j].load(beta_ptr, idx * kFParts + j);
            }
          }
        }
      }
      if constexpr (kPostWeight) {
        BF16Vec weight_v;
        weight_v.load(post_weight, idx);
#pragma unroll
        for (int q = 0; q < kPairs; ++q)
          post_w[p][q] = cast<fp32x2_t>(weight_v[q]);
      }
      const BF16Vec branch_v =
          muse_vr_ar2_norm::row_sum_vector(params, phase, owner, row * kRowVecs + idx, lrow * kRowVecs + idx);
#pragma unroll
      for (int q = 0; q < kPairs; ++q) {
        auto v = cast<fp32x2_t>(branch_v[q]);
        if constexpr (kPostNormIsInputScale && kPostWeight) {
          v.x *= post_w[p][q].x;
          v.y *= post_w[p][q].y;
          v = round_bf16(v);
        }
        vals[p][q] = v;
        sum_sq = fmaf(v.x, v.x, sum_sq);
        sum_sq = fmaf(v.y, v.y, sum_sq);
      }
    }

    sum_sq = cta::reduce_sum<kNumThreads>(sum_sq, smem);
    const float post_rstd = math::rsqrt(sum_sq / kHidden + norm.post_eps);

    fp32x2_t gate_a = {1.0f, 1.0f};
    fp32x2_t gate_b = {1.0f, 1.0f};
    if constexpr (kGate) {
      if (!channel_wise) {
        gate_a.x = gate_a.y = *alpha_ptr;
        gate_b.x = gate_b.y = *beta_ptr;
      }
    }

    float pre_sum_sq = 0.0f;
#pragma unroll
    for (int p = 0; p < kPacks; ++p) {
      const int idx = threadIdx.x + p * kNumThreads;
      F32Vec alpha_v[kFParts], beta_v[kFParts];
      if constexpr (kGate) {
        if (channel_wise) {
#pragma unroll
          for (int j = 0; j < kFParts; ++j) {
            if constexpr (kPrefetchWeights) {
              alpha_v[j] = alpha_pf[p][j];
              beta_v[j] = beta_pf[p][j];
            } else {
              alpha_v[j].load(alpha_ptr, idx * kFParts + j);
              beta_v[j].load(beta_ptr, idx * kFParts + j);
            }
          }
        }
      }
      F32Vec new_stream_v[kFParts];
#pragma unroll
      for (int q = 0; q < kPairs; ++q) {
        const int h = q / kFPairs;
        const int l = q % kFPairs;
        fp32x2_t normed_b = vals[p][q];
        normed_b.x *= post_rstd;
        normed_b.y *= post_rstd;
        if constexpr (!kPostNormIsInputScale && kPostWeight) {
          normed_b.x *= post_w[p][q].x;
          normed_b.y *= post_w[p][q].y;
        }
        normed_b = round_bf16(normed_b);
        const fp32x2_t s = stream_v[p][h][l];
        fp32x2_t new_s;
        if constexpr (kGate) {
          const fp32x2_t ga = channel_wise ? alpha_v[h][l] : gate_a;
          const fp32x2_t gb = channel_wise ? beta_v[h][l] : gate_b;
          new_s.x = ga.x * s.x + gb.x * normed_b.x;
          new_s.y = ga.y * s.y + gb.y * normed_b.y;
        } else {
          new_s.x = s.x + normed_b.x;
          new_s.y = s.y + normed_b.y;
        }
        new_stream_v[h][l] = new_s;
        const fp32x2_t pre_v = pre_bf16 ? round_bf16(new_s) : new_s;
        vals[p][q] = pre_v;
        pre_sum_sq = fmaf(pre_v.x, pre_v.x, pre_sum_sq);
        pre_sum_sq = fmaf(pre_v.y, pre_v.y, pre_sum_sq);
      }
#pragma unroll
      for (int j = 0; j < kFParts; ++j)
        new_stream_v[j].store(new_stream, idx * kFParts + j);
    }

    pre_sum_sq = cta::reduce_sum<kNumThreads>(pre_sum_sq, kNumWarps == 1 ? smem : smem + kNumWarps);
    const float pre_rstd = math::rsqrt(pre_sum_sq / kHidden + norm.pre_eps);

#pragma unroll
    for (int p = 0; p < kPacks; ++p) {
      const int idx = threadIdx.x + p * kNumThreads;
      BF16Vec weight_v;
      if (has_pre_weight) {
        if constexpr (kPrefetchWeights) {
          weight_v = pre_w_pf[p];
        } else {
          weight_v.load(pre_weight, idx);
        }
      }
      BF16Vec out_v;
      F32Vec router_v[kFParts];
#pragma unroll
      for (int q = 0; q < kPairs; ++q) {
        fp32x2_t v = vals[p][q];
        v.x *= pre_rstd;
        v.y *= pre_rstd;
        if (has_pre_weight) {
          const auto wf = cast<fp32x2_t>(weight_v[q]);
          v.x *= wf.x;
          v.y *= wf.y;
        }
        if (pre_bf16) v = round_bf16(v);
        out_v[q] = cast<bf16x2_t>(v);
        if constexpr (kHasRouter) {
          router_v[q / kFPairs][q % kFPairs] = v;
        }
      }
      out_v.store(out, idx);
      if constexpr (kHasRouter) {
#pragma unroll
        for (int j = 0; j < kFParts; ++j)
          router_v[j].store(router, idx * kFParts + j);
      }
    }
  }

  PDLTriggerSecondary<kUsePDL>();
  muse_vr_ar2_norm::end_phase(params, phase);
}

using host::distributed::CommunicatorRef;

template <
    int64_t kHidden,
    uint32_t kWorld,
    int kNumThreads,
    bool kPrefetchWeights,
    int kClusterSize,
    bool kUsePDL,
    bool kPostNormIsInputScale,
    bool kHasRouter>
struct MuseVrAr2NormResidual {
  static_assert(kClusterSize == 1 || kClusterSize == 2 || kClusterSize == 4 || kClusterSize == 8 || kClusterSize == 16);

  static void
  run(CommunicatorRef ref,
      const tvm::ffi::TensorView partial,
      const tvm::ffi::TensorView stream,
      const tvm::ffi::TensorView output,
      const tvm::ffi::TensorView new_stream,
      const tvm::ffi::Optional<tvm::ffi::TensorView> router_out,
      const tvm::ffi::Optional<tvm::ffi::TensorView> post_weight,
      const tvm::ffi::Optional<tvm::ffi::TensorView> pre_weight,
      const tvm::ffi::Optional<tvm::ffi::TensorView> alpha,
      const tvm::ffi::Optional<tvm::ffi::TensorView> beta,
      double post_eps,
      double pre_eps,
      bool pre_norm_bf16) {
    using namespace host;
    const auto& push = ref.get()->get_push_obj();

    auto M = SymbolicSize{"num_tokens"};
    auto H = SymbolicSize{"hidden_size"};
    auto G = SymbolicSize{"gate_dim"};
    auto device_ = SymbolicDevice{};
    H.set_value(kHidden);
    device_.set_options<kDLCUDA>();

    TensorMatcher({M, H}).with_dtype<bf16_t>().with_device(device_).verify(partial);
    TensorMatcher({M, H}).with_dtype<fp32_t>().with_device(device_).verify(stream);
    TensorMatcher({M, H}).with_dtype<bf16_t>().with_device(device_).verify(output);
    TensorMatcher({M, H}).with_dtype<fp32_t>().with_device(device_).verify(new_stream);
    RuntimeCheck(router_out.has_value() == kHasRouter, "router_out presence must match the kHasRouter instantiation");
    if constexpr (kHasRouter) {
      TensorMatcher({M, H}).with_dtype<fp32_t>().with_device(device_).verify(router_out.value());
    }
    if (post_weight.has_value()) {
      TensorMatcher({H}).with_dtype<bf16_t>().with_device(device_).verify(post_weight.value());
    }
    if (pre_weight.has_value()) {
      TensorMatcher({H}).with_dtype<bf16_t>().with_device(device_).verify(pre_weight.value());
    }
    const bool gate = alpha.has_value();
    RuntimeCheck(gate == beta.has_value(), "alpha and beta must both be present or both absent");
    bool gate_channel_wise = false;
    if (gate) {
      TensorMatcher({G}).with_dtype<fp32_t>().with_device(device_).verify(alpha.value());
      TensorMatcher({G}).with_dtype<fp32_t>().with_device(device_).verify(beta.value());
      const auto n = G.unwrap();
      RuntimeCheck(n == 1 || n == kHidden, "alpha/beta must have 1 or hidden_size elements");
      gate_channel_wise = (n == kHidden);
    }
    if constexpr (kClusterSize > 1) RuntimeCheck(!pre_norm_bf16, "Cluster path requires FP32 pre-norm input");

    const auto num_rows = static_cast<uint32_t>(M.unwrap());
    if (num_rows == 0) return;
    RuntimeCheck(push.world_size == kWorld, "the kernel is compiled for TP", kWorld);
    RuntimeCheck(num_rows % kWorld == 0, "the row count must be a multiple of the world size");
    const uint32_t rows_per_rank = num_rows / kWorld;
    constexpr uint32_t kRowVecs = static_cast<uint32_t>(kHidden * 2 / 16);
    const uint32_t chunk_vecs = rows_per_rank * kRowVecs;
    const int64_t slot_need = int64_t(2) * chunk_vecs * 16;
    RuntimeCheck(slot_need <= int64_t(push.slot_bytes), "rows exceed the push slot size");
    uint32_t grid = num_rows * kClusterSize;
    if constexpr (kClusterSize == 1) {
      grid = std::min<uint32_t>(num_rows, push.num_blocks);
    } else {
      RuntimeCheck(grid <= push.num_blocks, "M * cluster_size exceeds the push plane's block counters");
    }

    const auto params = muse_vr_ar2_norm::Params<kWorld>{
        .ws = push.get_workspace<kWorld>(slot_need),
        .num_counters = push.num_blocks,
        .rank = push.rank,
        .num_rows = num_rows,
        .rows_per_rank = rows_per_rank,
        .chunk_vecs = chunk_vecs,
        .partial = static_cast<const bf16_t*>(partial.data_ptr()),
        .norm =
            NormResidualParams{
                .branch = nullptr,
                .stream = stream.data_ptr(),
                .output = output.data_ptr(),
                .new_stream = new_stream.data_ptr(),
                .router_out = kHasRouter ? router_out.value().data_ptr() : nullptr,
                .post_weight = post_weight.has_value() ? post_weight.value().data_ptr() : nullptr,
                .pre_weight = pre_weight.has_value() ? pre_weight.value().data_ptr() : nullptr,
                .alpha = gate ? alpha.value().data_ptr() : nullptr,
                .beta = gate ? beta.value().data_ptr() : nullptr,
                .post_eps = static_cast<float>(post_eps),
                .pre_eps = static_cast<float>(pre_eps),
                .num_rows = static_cast<int>(num_rows),
                .gate_channel_wise = gate_channel_wise,
                .has_pre_weight = pre_weight.has_value(),
                .pre_norm_bf16 = pre_norm_bf16,
            },
    };
    const auto device = device_.unwrap();

    const auto launch = [&]<bool kHasPostWeight, bool kHasGate>() {
      if constexpr (kClusterSize == 1) {
        LaunchKernel(grid, kNumThreads, device)
            .config({.use_pdl = kUsePDL})(
                muse_vr_ar2_norm_residual_row<
                    kHidden,
                    kWorld,
                    kNumThreads,
                    kUsePDL,
                    kPrefetchWeights,
                    kPostNormIsInputScale,
                    kHasRouter,
                    kHasPostWeight,
                    kHasGate>,
                params);
      } else {
        constexpr auto kernel = muse_vr_ar2_norm_residual_cluster<
            kHidden,
            kWorld,
            kNumThreads,
            kClusterSize,
            kUsePDL,
            kPrefetchWeights,
            kPostNormIsInputScale,
            kHasRouter,
            kHasPostWeight,
            kHasGate>;
        [[maybe_unused]] static const bool cluster_configured = [kernel] {
          if constexpr (kClusterSize > 8) {
            RuntimeDeviceCheck(cudaFuncSetAttribute(kernel, cudaFuncAttributeNonPortableClusterSizeAllowed, 1));
          }
          cudaLaunchConfig_t occupancy_config{};
          occupancy_config.gridDim = dim3(kClusterSize);
          occupancy_config.blockDim = dim3(kNumThreads);
          int max_cluster_size = 0;
          RuntimeDeviceCheck(cudaOccupancyMaxPotentialClusterSize(&max_cluster_size, kernel, &occupancy_config));
          RuntimeCheck(
              max_cluster_size >= kClusterSize,
              "Requested cluster size ",
              kClusterSize,
              " exceeds this kernel/device limit ",
              max_cluster_size);
          return true;
        }();
        LaunchKernel(grid, kNumThreads, device)
            .config({
                .use_pdl = kUsePDL,
                .cluster_dim = dim3(kClusterSize, 1, 1),
            })(kernel, params);
      }
    };

    if (post_weight.has_value()) {
      if (gate)
        launch.template operator()<true, true>();
      else
        launch.template operator()<true, false>();
    } else {
      if (gate)
        launch.template operator()<false, true>();
      else
        launch.template operator()<false, false>();
    }
  }
};

}  // namespace sglang
