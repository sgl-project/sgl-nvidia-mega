// SPDX-License-Identifier: Apache-2.0
// K3.1 ranking-only histogram normalization. Select and tie helpers are reused
// from the pinned SGLang TopKRegister; the input FP32 scores are never modified.
#include "deepseek_v4/topk_v2.cuh"
namespace k31_rank_topk {
using namespace sglang;
using namespace sglang::device;
using namespace sglang::device::topk;
SGL_DEVICE float binary_mul(float x, float y) {
  float result;
  asm("mul.rn.f32 %0, %1, %2;" : "=f"(result) : "f"(x), "f"(y));
  return result;
}
template <uint32_t kLocalVecs_>
struct RankTopKRegister : TopKRadixBase<12> {
  static constexpr uint32_t kLocalVecs = kLocalVecs_;
  static constexpr uint32_t kMaxSeqLen = kBlockSize * kVecSize * kLocalVecs;
  using Smem = typename TopKRadixBase<12>::Smem;

  template <bool kUsePDL>
  SGL_DEVICE static void forward(const TopKProblem& problem, void* _smem) {
    const auto tx = threadIdx.x;
    const auto smem = static_cast<Smem*>(_smem);

    init_histogram(smem->histogram, tx);
    if (tx == 0) {
      smem->count_eq = 0;
      smem->count_gt = 0;
    }

    __syncthreads();
    PDLWaitPrimary<kUsePDL>();
    const uint32_t num_full = div_ceil(problem.seq_len, kVecSize);

    // Phase 1: load full vectors + build histogram
    vec_t local_vecs[kLocalVecs];
#pragma unroll
    for (uint32_t i = 0; i < kLocalVecs; ++i) {
      const auto vi = tx + kBlockSize * i;
      if (vi < num_full) local_vecs[i].load(problem.in, vi);
    }

    const auto tail_start = (problem.seq_len - 1) % kVecSize + 1;
    // Read the original FP32 values once. Normalize only histogram keys;
    // the comparisons, tie refinement, and emitted indices use original scores.
    uint32_t maximum_bits = 0;
#pragma unroll
    for (uint32_t i = 0; i < kLocalVecs; ++i) {
      const auto vi = tx + kBlockSize * i;
      if (vi >= num_full) break;
#pragma unroll
      for (uint32_t j = 0; j < kVecSize; ++j) {
        if (vi * kVecSize + j < problem.seq_len)
          maximum_bits = max(maximum_bits, __float_as_uint(local_vecs[i][j]) & 0x7FFFFFFFu);
      }
    }
#pragma unroll
    for (uint32_t d = 16; d; d >>= 1)
      maximum_bits = max(maximum_bits, __shfl_xor_sync(0xFFFFFFFFu, maximum_bits, d));
    if ((tx & 31) == 0) smem->warp_sum[tx / 32] = maximum_bits;
    __syncthreads();
    maximum_bits = smem->warp_sum[tx & 31];
#pragma unroll
    for (uint32_t d = 16; d; d >>= 1)
      maximum_bits = max(maximum_bits, __shfl_xor_sync(0xFFFFFFFFu, maximum_bits, d));
    const int shift = min(max(125 - int((maximum_bits >> 23) & 255), 0), 64);
    // Preserve the stock path for all-zero or already large scores. It also
    // retains stock's stable_ties=false contract in heavily tied normal inputs.
    if (maximum_bits == 0 || shift == 0)
      return Register4::forward<kUsePDL>(problem, _smem);
    const float factor = __int_as_float((127 + shift) << 23);
    const float inverse = __int_as_float((127 - shift) << 23);
#pragma unroll
    for (uint32_t i = 0; i < kLocalVecs; ++i) {
      const auto vi = tx + kBlockSize * i;
      if (vi >= num_full) break;
      if (vi == num_full - 1) {
#pragma unroll
        for (uint32_t j = 0; j < kVecSize; ++j) {
          if (j >= tail_start) local_vecs[i][j] = padding_value();
        }
      }
#pragma unroll
      for (uint32_t j = 0; j < kVecSize; ++j) {
        atomicAdd(&smem->histogram[extract_coarse_bin<kHistBits>(binary_mul(local_vecs[i][j], factor))], 1);
      }
    }
    const auto num_padding = kVecSize - tail_start + problem.input_start;
    if (tx == 0 && num_padding > 0) {
      // Ask the histogram's own binning where the platform's NaN landed.
      atomicSub(&smem->histogram[extract_coarse_bin<kHistBits>(padding_value())], num_padding);
      atomicAdd(&smem->histogram[0], num_padding);
    }
    __syncthreads();

    // Phase 2: Find the threshold bin
    find_threshold(problem.topk, num_full * kVecSize, smem, [&](uint32_t threshold_bin) {
      const auto [v_lo, v_hi] = coarse_bin_bounds<kHistBits>(threshold_bin);
      smem->v_hi = binary_mul(v_hi, inverse);
      smem->v_lo = binary_mul(v_lo, inverse);
    });

    // Phase 3: collect by two fp32 boundaries
    const auto topk = problem.topk;
    const auto v_hi = smem->v_hi;
    const auto v_lo = smem->v_lo;
    const bool zero_bin = is_zero_bin(v_lo);
    if (zero_bin) {
      // find_threshold has finished using the overlaid warp sums.
      if (tx == 0) smem->tie_nonzero = 0;
      __syncthreads();
    }

#pragma unroll
    for (uint32_t i = 0; i < kLocalVecs; ++i) {
      const auto vi = tx + kBlockSize * i;
      const auto base = vi * kVecSize;
      if (vi >= num_full) break;
#pragma unroll
      for (uint32_t j = 0; j < kVecSize; ++j) {
        const auto idx = base + j;
        const auto val = local_vecs[i][j];
        if (val >= v_hi) {
          const auto pos = atomicAdd(&smem->count_gt, 1);
          if (pos < topk) [[likely]] {
            problem.emit(pos, idx);
          }
        } else if (val >= v_lo) {
          const auto pos = atomicAdd(&smem->count_eq, 1);
          if (zero_bin && __float_as_uint(val) != 0u) atomicOr(&smem->tie_nonzero, 1u);
          if (pos < kMaxNumTie) [[likely]] {
            smem->tie_values[pos] = {val, idx};
          }
        }
      }
    }

    // Phase 4: Handle ties.
    __syncthreads();
    if (smem->count_eq > kMaxNumTie && (!zero_bin || smem->tie_nonzero != 0)) [[unlikely]] {
      // Buffer holds an arrival-order subset; re-stage it on the exact key.
      refine_ties(problem, smem, v_lo, v_hi, smem->count_eq);
    }
    const auto count_gt = smem->count_gt;
    const auto count_eq = smem->count_eq;
    const auto remain_topk = count_gt < topk ? topk - count_gt : 0;
    const auto tie_count = min(count_eq, kMaxNumTie);
    handle_tie(smem->tie_values, problem, count_gt, tie_count, remain_topk, &smem->tie_handle);
  }
};
TOPK_KERNEL void rank_kernel(const __grid_constant__ TopKPagedParams params, const uint32_t queries_per_request) {
  device::enable_smem_spilling();
  __shared__ impl::MaxSmem<Register4::Smem, Streaming::Smem> smem;
  __shared__ int32_t staging[kMaxTopK];
  const auto row = blockIdx.x;
  auto problem = params.problem(row);
  if (problem.seq_len <= problem.topk)
    return trivial_transform<true, TopKMode::PAGE_TABLE>(problem, params.get_transform<TopKMode::PAGE_TABLE>(row / queries_per_request));
  problem.out = staging;
  if (problem.seq_len <= kReg2MaxSeqLen)
    RankTopKRegister<2>::forward<true>(problem, &smem);
  else if (problem.seq_len <= kReg4MaxSeqLen)
    RankTopKRegister<4>::forward<true>(problem, &smem);
  else
    Streaming::forward<true>(problem, &smem);
  device::PDLTriggerSecondary<true>();
  __syncthreads();
  paged_transform<TopKMode::PAGE_TABLE>(problem, params.get_output_ptr(row), params.get_transform<TopKMode::PAGE_TABLE>(row / queries_per_request));
}
struct Binding {
 static void run(const tvm::ffi::TensorView scores, const tvm::ffi::TensorView lens,
                 const tvm::ffi::TensorView table, const tvm::ffi::TensorView out,
                 const tvm::ffi::TensorView plan, const int64_t queries_per_request) {
  using namespace sglang::host;
  auto B=SymbolicSize{"batch"}, L=SymbolicSize{"length"}, S=SymbolicSize{"stride"};
  auto dev=SymbolicDevice{};dev.set_options<kDLGPU>();
  TensorMatcher({B,L}).with_strides({S,1}).with_dtype<float>().with_device(dev).verify(scores);
  TensorMatcher({B}).with_dtype<int32_t>().with_device(dev).verify(lens);
  auto requests=SymbolicSize{"requests"};
  TensorMatcher({requests,-1}).with_strides({-1,1}).with_dtype<int32_t>().with_device(dev).verify(table);
  RuntimeCheck(queries_per_request > 0 && B.unwrap() == requests.unwrap() * queries_per_request,
               "top-k score rows must equal requests times query width");
  TensorMatcher({B,2048}).with_dtype<int32_t>().with_device(dev).verify(out);
  const auto params=TopKPagedParams{
    .scores=static_cast<const float*>(scores.data_ptr()),
    .seq_lens=static_cast<const int32_t*>(lens.data_ptr()),
    .page_table=static_cast<const int32_t*>(table.data_ptr()),
    .page_indices=static_cast<int32_t*>(out.data_ptr()), .raw_indices=nullptr,
    .metadata=static_cast<const PlanItem*>(plan.data_ptr()),
    .score_stride=S.unwrap(), .page_table_stride=table.stride(0),
    .topk=2048, .page_bits=6, .static_cluster_floor=0,
    .batch_size=static_cast<uint32_t>(B.unwrap()), .out_stride=2048,
    .token_seq_lens=nullptr, .out_offsets=nullptr, .pool_bits=0, .out_cols=2048,
  };
  LaunchKernel(B.unwrap(),kBlockSize,dev.unwrap()).config({.use_pdl=true}).launch(rank_kernel,params,static_cast<uint32_t>(queries_per_request));
 }
};
} // namespace k31_rank_topk
