// SPDX-License-Identifier: Apache-2.0
// The entry follows DeepGEMM's sm100_paged_mqa_logits (deep_gemm/legacy/impls/sm100_mqa_logits.cuh,
// https://github.com/deepseek-ai/DeepGEMM, MIT License, Copyright (c) 2025 DeepSeek) and runs its device
// core and scheduler from the installed sgl-deep-gemm headers; only the Q block differs.
//
// DeepGEMM's FP8 paged index logits (FP32 weights and logits, 2D context lengths, no clean) with a
// request's K31_NEXT_N draft rows in one Q block. DeepGEMM sizes the Q block as 128 / heads rows (4 at 32
// heads), so 5 rows run as two blocks (3 + 2) that each stream the request's whole index-K range. Here one
// block holds the rows (UMMA_N = K31_NEXT_N * K31_HEADS) and the index-K streams once. Every logit is the
// same UMMA column reduced by the same thread with the same FP32 FMAs and KV scale: the bits do not change.
#ifdef __CUDACC_RTC__
// NVRTC does not declare the device runtime's PDL wait the core calls; this is its CUDA definition
// (cuda_device_runtime_api.h), as the package's other NVRTC kernels wait.
static __device__ __forceinline__ void cudaGridDependencySynchronize() {
    asm volatile("griddepcontrol.wait;" ::: "memory");
}
#endif
#include <deep_gemm/legacy/impls/sm100_mqa_logits.cuh>

#if !defined(K31_NEXT_N) || !defined(K31_HEADS) || !defined(K31_HEAD_DIM) || !defined(K31_PAGE)
#error "define K31_NEXT_N, K31_HEADS, K31_HEAD_DIM and K31_PAGE"
#endif

namespace k31_score {

using namespace deep_gemm::legacy;

// DeepGEMM's paged FP8 launch: 256-key splits, 16 splits per chunk, 3 Q stages, 128 + 256 threads
constexpr uint32_t kSplitKV = 256, kSplitsPerChunk = 16, kQStages = 3, kTmemStages = 3;
constexpr uint32_t kSpecializedThreads = 128, kMathThreads = 256;
constexpr uint32_t kSmemCapacity = 232448;  // SM100 dynamic shared memory per block

template <uint32_t kKVStages>
using Storage = layout::MQALogitsSharedStorage<K31_HEADS, K31_HEAD_DIM, false, K31_NEXT_N, kSplitKV, kQStages,
                                               kKVStages, kTmemStages, cutlass::float_e4m3_t, float>;
// DeepGEMM's FP8 depth is 5 KV stages; a block of 5 x 32 rows leaves room for 4 (199680 bytes)
constexpr uint32_t kKVStages = sizeof(Storage<5>) <= kSmemCapacity ? 5 : 4;
DG_STATIC_ASSERT(sizeof(Storage<kKVStages>) <= kSmemCapacity, "the Q block does not fit shared memory");

// The paged scheduler takes the head count only to size its Q block as 128 / heads rows: 128 / K31_NEXT_N
// "heads" give a block of at least K31_NEXT_N rows, so a request's rows are one block.
using Scheduler = sched::SM100PagedMQALogitsScheduler<K31_NEXT_N, true, false, 128 / K31_NEXT_N, kSplitKV,
                                                      K31_PAGE, kSplitsPerChunk>;
DG_STATIC_ASSERT(Scheduler::BLOCK_Q >= K31_NEXT_N, "a request's rows must be one Q block");

}  // namespace k31_score

extern "C" __device__ const uint32_t k31_score_one_block_smem =
    sizeof(k31_score::Storage<k31_score::kKVStages>);

extern "C" __global__ __launch_bounds__(k31_score::kSpecializedThreads + k31_score::kMathThreads, 1)
void k31_score_one_block(const uint32_t num_q_tokens_total, const uint32_t logits_stride,
                         const uint32_t block_table_stride, const uint32_t* context_lens, float* logits,
                         const uint32_t* block_table, const uint32_t* schedule_meta,
                         const __grid_constant__ cute::TmaDescriptor tensor_map_q,
                         const __grid_constant__ cute::TmaDescriptor tensor_map_sf_q,
                         const __grid_constant__ cute::TmaDescriptor tensor_map_kv,
                         const __grid_constant__ cute::TmaDescriptor tensor_map_sf_kv,
                         const __grid_constant__ cute::TmaDescriptor tensor_map_weights) {
    using namespace k31_score;
    const auto make_scheduler = [&](const uint32_t& sm_idx) {
        return Scheduler(sm_idx, context_lens, schedule_meta, nullptr, block_table, block_table_stride,
                         num_q_tokens_total);
    };
    // Paged uses `kNumSMs = 0`; the schedule metadata drives the grid stride
    sm100_mqa_logits_core_impl<K31_HEADS, K31_HEAD_DIM, false, false, false, K31_NEXT_N, kSplitKV, kQStages,
                               kKVStages, 0, kSpecializedThreads, kMathThreads, cutlass::float_e4m3_t, float,
                               float, decltype(make_scheduler)>(
        logits_stride, logits, tensor_map_q, tensor_map_sf_q, tensor_map_kv, tensor_map_sf_kv, tensor_map_weights,
        make_scheduler);
}
