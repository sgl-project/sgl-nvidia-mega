#pragma once

#include <common/wait.cuh>

#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cutlass/arch/barrier.h>

#include <cfloat>
#include <cmath>
#include <cstddef>
#include <cstdint>

#include <common/hmma_b16.cuh>
#include <mla/args.h>

#ifndef MEGAMLA_USE_NATIVE_BF16X2_E4M3X2_CVT
#if __CUDACC_VER_MAJOR__ > 13 || \
    (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2)
#define MEGAMLA_USE_NATIVE_BF16X2_E4M3X2_CVT 1
#else
#define MEGAMLA_USE_NATIVE_BF16X2_E4M3X2_CVT 0
#endif
#endif

namespace glm5::mla::detail {

inline constexpr unsigned kFullWarpMask = 0xffffffffU;
inline constexpr float kRmsEpsilon = 1.0e-5F;
inline constexpr float kAttentionLog2Scale = 0.09016843885183334F;  // log2(e) / sqrt(256)
inline constexpr int kPipelineStages = 4;
inline constexpr std::uint32_t kPhaseInitializedTag = 0x4d4c4138U;  // "MLA8"

// Separate ready and empty barriers for each pipeline stage.
struct alignas(16) PipelineControl {
  alignas(8) std::uint64_t ready[kPipelineStages];
  std::uint8_t ready_to_empty_padding[0x60];
  alignas(8) std::uint64_t empty[kPipelineStages];
};

static_assert(offsetof(PipelineControl, empty) == 0x80);
static_assert(sizeof(PipelineControl) == 0xa0);

__device__ __forceinline__ void consumer_group_sync() {
  cutlass::arch::NamedBarrier::sync(kConsumerThreads,
                                    cutlass::arch::ReservedNamedBarriers::StreamkBarrier0);
}

__device__ __forceinline__ void combine_group_sync() {
  cutlass::arch::NamedBarrier::sync(kConsumerThreads,
                                    cutlass::arch::ReservedNamedBarriers::StreamkBarrier1);
}

__device__ __forceinline__ void full_cta_group_sync() {
  cutlass::arch::NamedBarrier::sync(kThreads,
                                    cutlass::arch::ReservedNamedBarriers::StreamkBarrier0);
}

__device__ __forceinline__ void producer_group_sync() {
  cutlass::arch::NamedBarrier::sync(kThreads - kConsumerThreads,
                                    cutlass::arch::ReservedNamedBarriers::TmemAllocBarrier);
}

__device__ __forceinline__ std::uint8_t* pipeline_stage(std::uint8_t* dynamic_storage, int task,
                                                        int stage_bytes) {
  return dynamic_storage + static_cast<std::size_t>(task & (kPipelineStages - 1)) * stage_bytes;
}

__device__ __forceinline__ std::uint8_t const* pipeline_stage(std::uint8_t const* dynamic_storage,
                                                              int task, int stage_bytes) {
  return dynamic_storage + static_cast<std::size_t>(task & (kPipelineStages - 1)) * stage_bytes;
}

__device__ __forceinline__ void initialize_pipeline(PipelineControl* control) {
  int const tid = static_cast<int>(threadIdx.x);
  if (tid < kPipelineStages) {
    cutlass::arch::ClusterTransactionBarrier::init(control->ready + tid,
                                                   (kThreads - kConsumerThreads) / 32);
    cutlass::arch::ClusterBarrier::init(control->empty + tid, kConsumerThreads);
    cutlass::arch::fence_barrier_init();
  }
}

__device__ __forceinline__ void wait_barrier_parity(std::uint64_t const* barrier,
                                                    std::uint32_t phase) {
  glm5::wait_barrier(barrier, phase);
}

__device__ __forceinline__ void producer_wait_stage(PipelineControl* control, int task) {
  int const stage = task & (kPipelineStages - 1);
  unsigned int const cycle = static_cast<unsigned int>(task / kPipelineStages);
  if (cycle != 0) {
    unsigned int const empty_phase = (cycle - 1U) & 1U;
    wait_barrier_parity(control->empty + stage, empty_phase);
  }
}

__device__ __forceinline__ void producer_expect_transaction(std::uint64_t* barrier,
                                                            std::uint32_t transaction_bytes) {
  std::uint32_t const shared_address =
      static_cast<std::uint32_t>(__cvta_generic_to_shared(barrier));
  asm volatile(
      "mbarrier.expect_tx.relaxed.cta.shared::cta.b64 "
      "[%0], %1;"
      :
      : "r"(shared_address), "r"(transaction_bytes)
      : "memory");
}

__device__ __forceinline__ void producer_arrive_stage(PipelineControl* control, int task) {
  if ((static_cast<int>(threadIdx.x) & 31) == 0) {
    int const stage = task & (kPipelineStages - 1);
    cutlass::arch::ClusterBarrier::arrive(control->ready + stage);
  }
}

__device__ __forceinline__ void consumer_wait_stage(PipelineControl* control, int task) {
  int const stage = task & (kPipelineStages - 1);
  unsigned int const phase = static_cast<unsigned int>(task / kPipelineStages) & 1U;
  wait_barrier_parity(control->ready + stage, phase);
}

__device__ __forceinline__ void consumer_release_stage(PipelineControl* control, int task) {
  int const stage = task & (kPipelineStages - 1);
  cutlass::arch::ClusterBarrier::arrive(control->empty + stage);
}

template <class T>
__device__ __forceinline__ T* workspace_pointer(Glm5MegaMlaArgs const& args, std::size_t offset) {
  return reinterpret_cast<T*>(static_cast<std::uint8_t*>(args.workspace) + offset);
}

template <class T>
__device__ __forceinline__ T const* workspace_pointer_const(Glm5MegaMlaArgs const& args,
                                                            std::size_t offset) {
  return reinterpret_cast<T const*>(static_cast<std::uint8_t const*>(args.workspace) + offset);
}

__device__ __forceinline__ float bf16_to_float(__nv_bfloat16 value) {
  return __bfloat162float(value);
}

__device__ __forceinline__ __nv_bfloat16 float_to_bf16(float value) {
  return __float2bfloat16_rn(value);
}

// Drain all explicitly committed async-copy groups. wait_group 0 avoids
// adding an empty group to the sequence.
__device__ __forceinline__ void wait_cp_async_group_0() {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
  asm volatile("cp.async.wait_group 0;" : : : "memory");
#endif
}

__device__ __forceinline__ void wait_cp_async_group_1() {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
  asm volatile("cp.async.wait_group 1;" : : : "memory");
#endif
}

// TMA reads global memory through the async proxy.  A normal global store is
// not ordered with that read by __threadfence() alone, so every thread that
// publishes KV cache data executes this fence before the CTA releases the KV
// phase word to the attention CTAs.
__device__ __forceinline__ void publish_global_writes_to_async_proxy() {
  asm volatile("fence.proxy.async.global;" : : : "memory");
}

// Generic shared stores must cross into the async proxy before a
// shared-to-global bulk operation can consume them.  Attention commits its
// two payload copies together with the following async reduction arrival.
__device__ __forceinline__ void publish_shared_writes_to_async_proxy() {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
  asm volatile("fence.proxy.async.shared::cta;" : : : "memory");
#endif
}

__device__ __forceinline__ void copy_bulk_shared_to_global(void* global_destination,
                                                           void const* shared_source,
                                                           std::uint32_t bytes) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
  std::uint64_t const global_address = reinterpret_cast<std::uint64_t>(global_destination);
  std::uint32_t const shared_address =
      static_cast<std::uint32_t>(__cvta_generic_to_shared(shared_source));
  asm volatile(
      "cp.async.bulk.global.shared::cta.bulk_group "
      "[%0], [%1], %2;"
      :
      : "l"(global_address), "r"(shared_address), "r"(bytes)
      : "memory");
#else
  (void)global_destination;
  (void)shared_source;
  (void)bytes;
#endif
}

__device__ __forceinline__ void reduce_bulk_shared_to_global_add_u32(
    void* global_destination, void const* shared_source, std::uint32_t bytes) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
  std::uint64_t const global_address = reinterpret_cast<std::uint64_t>(global_destination);
  std::uint32_t const shared_address =
      static_cast<std::uint32_t>(__cvta_generic_to_shared(shared_source));
  asm volatile(
      "cp.reduce.async.bulk.global.shared::cta.bulk_group.add.u32 "
      "[%0], [%1], %2;"
      :
      : "l"(global_address), "r"(shared_address), "r"(bytes)
      : "memory");
#else
  (void)global_destination;
  (void)shared_source;
  (void)bytes;
#endif
}

__device__ __forceinline__ void commit_bulk_copy_group() {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
  asm volatile("cp.async.bulk.commit_group;" : : : "memory");
#endif
}

__device__ __forceinline__ void wait_bulk_copy_group() {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
  asm volatile("cp.async.bulk.wait_group 0;" : : : "memory");
#endif
}

__device__ __forceinline__ std::uint32_t pack_bf16x2(float low, float high) {
  std::uint32_t packed;
  asm volatile("cvt.rn.bf16x2.f32 %0, %2, %1;" : "=r"(packed) : "f"(low), "f"(high));
  return packed;
}

__device__ __forceinline__ std::uint32_t add_bf16x2(std::uint32_t lhs, std::uint32_t rhs) {
  std::uint32_t result;
  asm volatile("add.rn.bf16x2 %0, %1, %2;" : "=r"(result) : "r"(lhs), "r"(rhs));
  return result;
}

__device__ __forceinline__ std::uint32_t e4m3x2_to_mma_x2(
    std::uint16_t packed_fp8) {
  std::uint32_t packed_mma;
#if MEGAMLA_USE_NATIVE_BF16X2_E4M3X2_CVT
  asm volatile("cvt.rn.bf16x2.e4m3x2 %0, %1;"
               : "=r"(packed_mma)
               : "h"(packed_fp8));
#else
  asm volatile("cvt.rn.f16x2.e4m3x2 %0, %1;"
               : "=r"(packed_mma)
               : "h"(packed_fp8));
#endif
  return packed_mma;
}

__device__ __forceinline__ std::uint16_t mma_x2_to_e4m3x2(
    std::uint32_t packed_mma) {
  std::uint16_t packed_fp8;
#if MEGAMLA_USE_NATIVE_BF16X2_E4M3X2_CVT
  asm volatile("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;"
               : "=h"(packed_fp8)
               : "r"(packed_mma));
#else
  asm volatile("cvt.rn.satfinite.e4m3x2.f16x2 %0, %1;"
               : "=h"(packed_fp8)
               : "r"(packed_mma));
#endif
  return packed_fp8;
}

__device__ __forceinline__ float warp_sum(float value) {
#pragma unroll
  for (int offset = 16; offset != 0; offset >>= 1) {
    value += __shfl_down_sync(kFullWarpMask, value, offset);
  }
  return value;
}

__device__ __forceinline__ float warp_max(float value) {
#pragma unroll
  for (int offset = 16; offset != 0; offset >>= 1) {
    value = fmaxf(value, __shfl_down_sync(kFullWarpMask, value, offset));
  }
  return value;
}

__device__ __forceinline__ std::uint64_t load_relaxed_gpu_u64(std::uint64_t const* address) {
  std::uint64_t value;
  asm volatile("ld.global.relaxed.gpu.u64 %0, [%1];" : "=l"(value) : "l"(address) : "memory");
  return value;
}

__device__ __forceinline__ std::uint32_t load_relaxed_gpu_u32(std::uint32_t const* address) {
  std::uint32_t value;
  asm volatile("ld.global.relaxed.gpu.u32 %0, [%1];" : "=r"(value) : "l"(address) : "memory");
  return value;
}

__device__ __forceinline__ void store_relaxed_gpu_u32(std::uint32_t* address,
                                                       std::uint32_t value) {
  asm volatile("st.global.relaxed.gpu.u32 [%0], %1;" : : "l"(address), "r"(value) : "memory");
}

__device__ __forceinline__ void acquire_fence_gpu() {
  asm volatile("fence.acquire.gpu;" : : : "memory");
}

// CUDA 13.3 otherwise keeps the first-generation CAS loop in every caller,
// increasing steady-state register pressure. Older compilers keep their layout.
#if __CUDACC_VER_MAJOR__ > 13 || \
    (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 3)
__device__ __noinline__
#else
__device__ __forceinline__
#endif
void initialize_phase_once(std::uint64_t* address) {
  // The phase workspace and per-CTA epoch counters form one persistent state.
  // Initialize dirty caller storage once, then keep the low word cumulative.
  std::uint64_t const desired = static_cast<std::uint64_t>(kPhaseInitializedTag) << 32;
  for (;;) {
    std::uint64_t const observed = load_relaxed_gpu_u64(address);
    if (static_cast<std::uint32_t>(observed >> 32) == kPhaseInitializedTag) {
      return;
    }
    auto const prior = atomicCAS(reinterpret_cast<unsigned long long*>(address), observed, desired);
    if (prior == observed) {
      return;
    }
  }
}

__device__ __forceinline__ void prepare_phase(std::uint64_t* address,
                                              std::uint32_t generation) {
  if (generation == 1U) {
    initialize_phase_once(address);
  }
}

__device__ __forceinline__ void phase_release_add(std::uint64_t* address,
                                                  std::uint32_t generation,
                                                  std::uint32_t contribution) {
  prepare_phase(address, generation);
  __threadfence();
  atomicAdd(reinterpret_cast<unsigned int*>(address), contribution);
}

__device__ __forceinline__ void wait_phase_count(std::uint64_t* address, std::uint32_t generation,
                                                 std::uint32_t expected) {
  auto const* count = reinterpret_cast<std::uint32_t const*>(address);
  bool const expected_bit = (generation & 1U) != 0U;
  for (;;) {
    std::uint32_t const observed = load_relaxed_gpu_u32(count);
    if (((observed & expected) != 0U) == expected_bit) {
      acquire_fence_gpu();
      return;
    }
    asm volatile("" : : : "memory");
  }
}

__device__ __forceinline__ void wait_phase_signed(std::uint64_t* address,
                                                  std::uint32_t generation) {
  auto const* count = reinterpret_cast<std::uint32_t const*>(address);
  bool const expected_sign = (generation & 1U) != 0U;
  for (;;) {
    std::uint32_t const observed = load_relaxed_gpu_u32(count);
    if ((static_cast<std::int32_t>(observed) < 0) == expected_sign) {
      acquire_fence_gpu();
      return;
    }
    asm volatile("" : : : "memory");
  }
}

// Main-CTA producer and consumer warps execute independent persistent paths.
// Their global phase operations therefore use group-local named barriers;
// auxiliary CTAs use their matching consumer/full-CTA barriers inline.
__device__ __forceinline__ void consumer_phase_arrive_and_wait(std::uint64_t* address,
                                                               std::uint32_t generation,
                                                               std::uint32_t contribution,
                                                               std::uint32_t expected) {
  consumer_group_sync();
  if (threadIdx.x == 0) {
    if (contribution != 0U) {
      phase_release_add(address, generation, contribution);
    } else {
      prepare_phase(address, generation);
    }
    wait_phase_count(address, generation, expected);
  }
  consumer_group_sync();
}

// Publish the Q-absorb contribution before waiting for RoPE, KV, and the
// aggregate Q-absorb count.
__device__ __forceinline__ void consumer_publish_q_absorb_and_wait_preattention(
    std::uint64_t* q_absorb, std::uint64_t* rope, std::uint64_t* kv, std::uint32_t generation,
    std::uint32_t q_absorb_contribution) {
  consumer_group_sync();
  if (threadIdx.x == 0) {
    if (q_absorb_contribution != 0U) {
      phase_release_add(q_absorb, generation, q_absorb_contribution);
    } else {
      prepare_phase(q_absorb, generation);
    }
    prepare_phase(rope, generation);
    wait_phase_count(rope, generation, 256);
  }
  consumer_group_sync();
  if (threadIdx.x == 0) {
    prepare_phase(kv, generation);
    wait_phase_count(kv, generation, 256);
  }
  consumer_group_sync();
  if (threadIdx.x == 0) {
    wait_phase_count(q_absorb, generation, 256);
  }
  consumer_group_sync();
}

__device__ __forceinline__ void producer_phase_wait(std::uint64_t* address,
                                                    std::uint32_t generation,
                                                    std::uint32_t expected) {
  if (threadIdx.x == kConsumerThreads) {
    prepare_phase(address, generation);
    wait_phase_count(address, generation, expected);
  }
  producer_group_sync();
}

// Initialize the barrier before producers register and issue their transfers.
__device__ __forceinline__ void initialize_transaction_barrier(std::uint64_t* barrier,
                                                               std::uint32_t arrive_count) {
  if (threadIdx.x == 0) {
    cutlass::arch::ClusterTransactionBarrier::init(barrier, arrive_count);
    cutlass::arch::fence_barrier_init();
  }
  full_cta_group_sync();
}

__device__ __forceinline__ void producer_arrive_and_expect_tx_count(
    std::uint64_t* barrier, std::uint32_t arrive_count, std::uint32_t transaction_bytes) {
  std::uint32_t const shared_address =
      static_cast<std::uint32_t>(__cvta_generic_to_shared(barrier));
  asm volatile(
      "mbarrier.expect_tx.relaxed.cta.shared::cta.b64 "
      "[%0], %1;\n\t"
      "mbarrier.arrive.release.cta.shared::cta.b64 "
      "_, [%0], %2;"
      :
      : "r"(shared_address), "r"(transaction_bytes), "r"(arrive_count)
      : "memory");
}

__device__ __forceinline__ void wait_transaction_barrier(std::uint64_t const* barrier,
                                                         std::uint32_t phase) {
  wait_barrier_parity(barrier, phase);
}

__device__ __forceinline__ void invalidate_transaction_barrier(std::uint64_t* barrier) {
  full_cta_group_sync();
  if (threadIdx.x == 0) {
    cutlass::arch::ClusterBarrier::invalidate(barrier);
  }
  full_cta_group_sync();
}

// Signal partial completion through an async reduction and wait for participating CTAs.
// Complete pending bulk copies before reusing their shared staging.
template <int kParticipants = kMainCtas>
__device__ __forceinline__ void consumer_partial_async_arrive_and_wait(
    std::uint64_t* address, std::uint32_t generation, int cta,
    std::uint32_t* shared_contribution) {
  // Join the consumers before publishing their completion contribution.
  consumer_group_sync();
  if (threadIdx.x == 0) {
    prepare_phase(address, generation);
    static_assert(kParticipants > 0 && kParticipants <= kGridCtas);
    std::uint32_t const contribution =
        cta == 0 ? 0x80000000U - static_cast<std::uint32_t>(kParticipants - 1)
                 : 1U;
    uint4 const contribution_vector{contribution, 0U, 0U, 0U};
    *reinterpret_cast<uint4*>(shared_contribution) = contribution_vector;
    __threadfence_block();
    publish_shared_writes_to_async_proxy();
    reduce_bulk_shared_to_global_add_u32(address, shared_contribution, sizeof(uint4));
  }
  if (threadIdx.x < 32) {
    wait_phase_signed(address, generation);
  }
  if constexpr (kTokens >= 6) {
    // Finish attention payload copies before reusing their staging.
    if (threadIdx.x == 0) {
      commit_bulk_copy_group();
      wait_bulk_copy_group();
    }
  }
  consumer_group_sync();
}

template <int kElements>
__device__ __forceinline__ float compute_one_token_rms_inverse(__nv_bfloat16 const* input,
                                                               float* scratch) {
  int const tid = static_cast<int>(threadIdx.x);
  int const warp = tid >> 5;
  int const lane = tid & 31;
  if (tid < kConsumerThreads) {
    float sum = 0.0F;
    for (int index = tid; index < kElements; index += kConsumerThreads) {
      float const value = bf16_to_float(input[index]);
      sum = fmaf(value, value, sum);
    }
    sum = warp_sum(sum);
    if (lane == 0) {
      scratch[warp] = sum;
    }
  }
  full_cta_group_sync();
  if (tid == 0) {
    float sum = 0.0F;
#pragma unroll
    for (int source_warp = 0; source_warp < 8; ++source_warp) {
      sum += scratch[source_warp];
    }
    scratch[8] = rsqrtf(sum / static_cast<float>(kElements) + kRmsEpsilon);
  }
  full_cta_group_sync();
  return scratch[8];
}

template <class Element>
__device__ __forceinline__ void unpack_e4m3_fragment(
    std::uint8_t const* source, Element (&destination)[8]) {
  std::uint64_t const packed_bytes = *reinterpret_cast<std::uint64_t const*>(source);
  auto* destination_words = reinterpret_cast<std::uint32_t*>(destination);
  std::uint32_t converted0;
  std::uint32_t converted1;
  std::uint32_t converted2;
  std::uint32_t converted3;
#if MEGAMLA_USE_NATIVE_BF16X2_E4M3X2_CVT
  asm volatile(
      "{\n\t"
      ".reg .b32 fp8_low, fp8_high;\n\t"
      ".reg .b16 fp8_pair0, fp8_pair1, fp8_pair2, fp8_pair3;\n\t"
      "mov.b64 {fp8_low, fp8_high}, %4;\n\t"
      "mov.b32 {fp8_pair0, fp8_pair1}, fp8_low;\n\t"
      "mov.b32 {fp8_pair2, fp8_pair3}, fp8_high;\n\t"
      "cvt.rn.bf16x2.e4m3x2 %0, fp8_pair0;\n\t"
      "cvt.rn.bf16x2.e4m3x2 %1, fp8_pair1;\n\t"
      "cvt.rn.bf16x2.e4m3x2 %2, fp8_pair2;\n\t"
      "cvt.rn.bf16x2.e4m3x2 %3, fp8_pair3;\n\t"
      "}\n"
      : "=r"(converted0), "=r"(converted1), "=r"(converted2), "=r"(converted3)
      : "l"(packed_bytes));
#else
  asm volatile(
      "{\n\t"
      ".reg .b32 fp8_low, fp8_high;\n\t"
      ".reg .b16 fp8_pair0, fp8_pair1, fp8_pair2, fp8_pair3;\n\t"
      "mov.b64 {fp8_low, fp8_high}, %4;\n\t"
      "mov.b32 {fp8_pair0, fp8_pair1}, fp8_low;\n\t"
      "mov.b32 {fp8_pair2, fp8_pair3}, fp8_high;\n\t"
      "cvt.rn.f16x2.e4m3x2 %0, fp8_pair0;\n\t"
      "cvt.rn.f16x2.e4m3x2 %1, fp8_pair1;\n\t"
      "cvt.rn.f16x2.e4m3x2 %2, fp8_pair2;\n\t"
      "cvt.rn.f16x2.e4m3x2 %3, fp8_pair3;\n\t"
      "}\n"
      : "=r"(converted0), "=r"(converted1), "=r"(converted2), "=r"(converted3)
      : "l"(packed_bytes));
#endif
  destination_words[0] = converted0;
  destination_words[1] = converted1;
  destination_words[2] = converted2;
  destination_words[3] = converted3;
}

__device__ __forceinline__ float exp2_underflow_safe(float exponent) {
  bool const normal = exponent >= -126.0F;
  float const adjusted = normal ? exponent : exponent * 0.5F;
  float value;
  asm volatile("ex2.approx.ftz.f32 %0, %1;" : "=f"(value) : "f"(adjusted));
  return normal ? value : value * value;
}

__device__ __forceinline__ float reciprocal_approx_ftz(float value) {
  float result;
  asm volatile("rcp.approx.ftz.f32 %0, %1;" : "=f"(result) : "f"(value));
  return result;
}

__device__ __forceinline__ float log2_approx(float value) {
  float result;
  asm volatile("lg2.approx.f32 %0, %1;" : "=f"(result) : "f"(value));
  return result;
}

// Global-to-shared copies are shared by projection and Attention stages.
__device__ __forceinline__ void copy_16b_g2s(void const* global_source, void* shared_destination) {
  std::uint32_t const shared_address =
      static_cast<std::uint32_t>(__cvta_generic_to_shared(shared_destination));
  std::uint64_t const global_address = reinterpret_cast<std::uint64_t>(global_source);
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16;"
               :
               : "r"(shared_address), "l"(global_address)
               : "memory");
}


__device__ __forceinline__ void copy_16b_g2s_zfill(void const* global_source,
                                                   void* shared_destination, bool valid) {
  std::uint32_t const shared_address =
      static_cast<std::uint32_t>(__cvta_generic_to_shared(shared_destination));
  std::uint64_t const global_address = reinterpret_cast<std::uint64_t>(global_source);
  int const source_bytes = valid ? 16 : 0;
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;"
               :
               : "r"(shared_address), "l"(global_address), "r"(source_bytes)
               : "memory");
}

}  // namespace glm5::mla::detail
