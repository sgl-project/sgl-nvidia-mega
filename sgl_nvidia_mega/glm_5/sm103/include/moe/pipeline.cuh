#pragma once

#include <common/wait.cuh>

#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cutlass/arch/barrier.h>
#include <cutlass/arch/memory_sm80.h>
#include <cute/arch/copy_sm100.hpp>
#include <cute/arch/copy_sm90_tma.hpp>

#include <cstddef>
#include <cstdint>
#include <type_traits>

#include <moe/args.h>

#ifndef MEGAMOE_USE_NATIVE_E4M3X2_TO_BF16X2
#if __CUDACC_VER_MAJOR__ > 13 || \
    (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2)
#define MEGAMOE_USE_NATIVE_E4M3X2_TO_BF16X2 1
#else
#define MEGAMOE_USE_NATIVE_E4M3X2_TO_BF16X2 0
#endif
#endif

namespace glm5::moe::detail {

// Device-side leaf operations shared by every GLM5 MegaMoE shape.
__device__ __forceinline__ __nv_bfloat16 float_to_bf16(float value) {
  return __float2bfloat16_rn(value);
}

__device__ __forceinline__ std::uint64_t float2_to_bits(float2 value) {
  return static_cast<std::uint64_t>(__float_as_uint(value.x)) |
         (static_cast<std::uint64_t>(__float_as_uint(value.y)) << 32);
}

__device__ __forceinline__ float2 bits_to_float2(std::uint64_t value) {
  return make_float2(__uint_as_float(static_cast<std::uint32_t>(value)),
                     __uint_as_float(static_cast<std::uint32_t>(value >> 32)));
}

__device__ __forceinline__ float2 fma_f32x2(float2 lhs, float2 rhs, float2 accumulator) {
  std::uint64_t result;
  asm volatile("fma.rn.f32x2 %0, %1, %2, %3;"
               : "=l"(result)
               : "l"(float2_to_bits(lhs)), "l"(float2_to_bits(rhs)),
                 "l"(float2_to_bits(accumulator)));
  return bits_to_float2(result);
}

__device__ __forceinline__ std::uint32_t bf16x2_to_mma_x2(
    std::uint32_t packed_bf16) {
#if MEGAMOE_USE_NATIVE_E4M3X2_TO_BF16X2
  return packed_bf16;
#else
  std::uint32_t packed_fp16;
  asm volatile(
      "{\n\t"
      ".reg .b16 bf16_low, bf16_high, fp16_low, fp16_high;\n\t"
      "mov.b32 {bf16_low, bf16_high}, %1;\n\t"
      "cvt.rn.f16.bf16 fp16_low, bf16_low;\n\t"
      "cvt.rn.f16.bf16 fp16_high, bf16_high;\n\t"
      "mov.b32 %0, {fp16_low, fp16_high};\n\t"
      "}\n"
      : "=r"(packed_fp16)
      : "r"(packed_bf16));
  return packed_fp16;
#endif
}

__device__ __forceinline__ std::uint32_t load_shared_u32(void const* address) {
  std::uint32_t value;
  std::uint32_t const shared_address =
      static_cast<std::uint32_t>(__cvta_generic_to_shared(address));
  asm volatile("ld.shared.u32 %0, [%1];"
               : "=r"(value)
               : "r"(shared_address)
               : "memory");
  return value;
}

__device__ __forceinline__ uint2 load_k_permuted_native_a_rows(
    std::uint8_t const* row_address) {
  std::uint32_t row_low;
  std::uint32_t row_high;
  std::uint32_t const shared_address =
      static_cast<std::uint32_t>(__cvta_generic_to_shared(row_address));
  asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0, %1}, [%2];"
               : "=r"(row_low), "=r"(row_high)
               : "r"(shared_address)
               : "memory");
  return make_uint2(row_low, row_high);
}

template <class Fragment>
__device__ __forceinline__ void unpack_k_permuted_native_a_rows(uint2 rows,
                                                                Fragment& fragment) {
  auto* fragment_words = reinterpret_cast<std::uint32_t*>(fragment.element);
  std::uint32_t converted0;
  std::uint32_t converted1;
  std::uint32_t converted2;
  std::uint32_t converted3;
#if MEGAMOE_USE_NATIVE_E4M3X2_TO_BF16X2
  asm volatile(
      "{\n\t"
      ".reg .b16 fp8_pair0, fp8_pair1, fp8_pair2, fp8_pair3;\n\t"
      "mov.b32 {fp8_pair0, fp8_pair2}, %4;\n\t"
      "mov.b32 {fp8_pair1, fp8_pair3}, %5;\n\t"
      "cvt.rn.bf16x2.e4m3x2 %0, fp8_pair0;\n\t"
      "cvt.rn.bf16x2.e4m3x2 %1, fp8_pair1;\n\t"
      "cvt.rn.bf16x2.e4m3x2 %2, fp8_pair2;\n\t"
      "cvt.rn.bf16x2.e4m3x2 %3, fp8_pair3;\n\t"
      "}\n"
      : "=r"(converted0), "=r"(converted1), "=r"(converted2), "=r"(converted3)
      : "r"(rows.x), "r"(rows.y));
#else
  // CUDA 13.0/13.1: keep the exact FP8 values in FP16 and use FP16 MMA.
  asm volatile(
      "{\n\t"
      ".reg .b16 fp8_pair0, fp8_pair1, fp8_pair2, fp8_pair3;\n\t"
      "mov.b32 {fp8_pair0, fp8_pair2}, %4;\n\t"
      "mov.b32 {fp8_pair1, fp8_pair3}, %5;\n\t"
      "cvt.rn.f16x2.e4m3x2 %0, fp8_pair0;\n\t"
      "cvt.rn.f16x2.e4m3x2 %1, fp8_pair1;\n\t"
      "cvt.rn.f16x2.e4m3x2 %2, fp8_pair2;\n\t"
      "cvt.rn.f16x2.e4m3x2 %3, fp8_pair3;\n\t"
      "}\n"
      : "=r"(converted0), "=r"(converted1), "=r"(converted2), "=r"(converted3)
      : "r"(rows.x), "r"(rows.y));
#endif
  fragment_words[0] = converted0;
  fragment_words[1] = converted1;
  fragment_words[2] = converted2;
  fragment_words[3] = converted3;
}

template <int kValidColumns, class Element, class Fragment>
__device__ __forceinline__ void load_k_permuted_activation_fragment(
    Element const* columns, int column_stride, int k_base, Fragment& fragment) {
  static_assert(kValidColumns >= 1 && kValidColumns <= 8);
  static_assert(std::is_same_v<Element, __nv_bfloat16> ||
                std::is_same_v<Element, __half>);
  int const lane = static_cast<int>(threadIdx.x) & 31;
  int const column = lane >> 2;
  int const k_quad = (lane & 3) * 4;
  auto* fragment_words = reinterpret_cast<std::uint32_t*>(fragment.element);
  if (column < kValidColumns) {
    Element const* source =
        columns + static_cast<std::size_t>(column) * column_stride + k_base + k_quad;
    std::uint64_t const values = *reinterpret_cast<std::uint64_t const*>(source);
    if constexpr (std::is_same_v<Element, __nv_bfloat16>) {
      fragment_words[0] = bf16x2_to_mma_x2(static_cast<std::uint32_t>(values));
      fragment_words[1] = bf16x2_to_mma_x2(static_cast<std::uint32_t>(values >> 32));
    } else {
      fragment_words[0] = static_cast<std::uint32_t>(values);
      fragment_words[1] = static_cast<std::uint32_t>(values >> 32);
    }
  } else {
    fragment_words[0] = 0;
    fragment_words[1] = 0;
  }
}

__device__ __forceinline__ float fused_moe_tanh(float value) {
  float result;
  asm volatile("tanh.approx.f32 %0, %1;" : "=f"(result) : "f"(value));
  return result;
}

__device__ __forceinline__ bool fp32_is_nan(float value) {
  std::uint32_t const absolute_bits = __float_as_uint(value) & 0x7fffffffU;
  return absolute_bits >= 0x7f800001U;
}

__device__ __forceinline__ float router_sigmoid(float value) {
  float raw = fmaf(0.5F, fused_moe_tanh(0.5F * value), 0.5F);
  if (fp32_is_nan(raw)) {
    raw = 0.0F;
  }
  return raw;
}

__device__ __forceinline__ float gate_silu_multiply(float gate, float up) {
  float const half_gate = 0.5F * gate;
  float const silu = fmaf(half_gate, fused_moe_tanh(half_gate), half_gate);
  return silu * up;
}

__device__ __forceinline__ void copy_16b_g2s(void const* global_source,
                                              void* shared_destination) {
  cutlass::arch::cp_async<16, cutlass::arch::CacheOperation::Global>(shared_destination,
                                                                     global_source);
}

// Four-stage TMA pipeline. All shape-specific task assignment remains in the
// corresponding MTP kernel; this type only owns the reusable barrier protocol.
template <int kStages, std::size_t kBytesPerStage, int kCtaThreads,
          int kDefaultConsumerThreads, int kTmaProducerWarps>
struct alignas(16) PipelineState {
  static constexpr int kStageCount = kStages;
  static constexpr std::size_t kStageBytes = kBytesPerStage;
  static constexpr int kThreads = kCtaThreads;
  static constexpr int kConsumerThreads = kDefaultConsumerThreads;
  static constexpr int kProducerWarps = kTmaProducerWarps;

  alignas(8) std::uint64_t ready[kStages];
  alignas(8) std::uint64_t empty[kStages];
};

__device__ __forceinline__ int pipeline_task_index(int local_task, int persistent_base) {
  return persistent_base + local_task;
}

template <class Control>
__device__ __forceinline__ std::uint8_t* pipeline_stage(std::uint8_t* dynamic_storage,
                                                         int task) {
  return dynamic_storage +
         static_cast<std::size_t>(task & (Control::kStageCount - 1)) * Control::kStageBytes;
}

template <class Control>
__device__ __forceinline__ void initialize_pipeline(
    Control* control, int consumer_arrivals = Control::kConsumerThreads) {
  if (threadIdx.x == 0) {
#pragma unroll
    for (int stage = 0; stage < Control::kStageCount; ++stage) {
      cutlass::arch::ClusterTransactionBarrier::init(control->ready + stage,
                                                     Control::kProducerWarps);
      cutlass::arch::ClusterBarrier::init(control->empty + stage, consumer_arrivals);
    }
    cutlass::arch::fence_barrier_init();
  }
  cutlass::arch::NamedBarrier::sync(
      Control::kThreads, cutlass::arch::ReservedNamedBarriers::StreamkBarrier1);
}

template <class Control>
__device__ __forceinline__ void producer_acquire_stage(Control* control, int task,
                                                       int transaction_bytes) {
  int const stage = task & (Control::kStageCount - 1);
  unsigned int const cycle = static_cast<unsigned int>(task / Control::kStageCount);
  int const lane = static_cast<int>(threadIdx.x) & 31;
  if (lane == 0) {
    if (cycle != 0) {
      unsigned int const empty_phase = (cycle - 1U) & 1U;
      glm5::wait_barrier(control->empty + stage, empty_phase);
    }
    if (transaction_bytes != 0) {
      cutlass::arch::ClusterTransactionBarrier::arrive_and_expect_tx(control->ready + stage,
                                                                     transaction_bytes);
    } else {
      cutlass::arch::ClusterBarrier::arrive(control->ready + stage);
    }
  }
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

template <class Control>
__device__ __forceinline__ int single_leader_acquire_stage(Control* control, int task,
                                                           int transaction_bytes) {
  int const stage = task & (Control::kStageCount - 1);
  unsigned int const cycle = static_cast<unsigned int>(task / Control::kStageCount);
  if (cycle != 0) {
    unsigned int const empty_phase = (cycle - 1U) & 1U;
    glm5::wait_barrier(control->empty + stage, empty_phase);
  }
  producer_arrive_and_expect_tx_count(control->ready + stage, Control::kProducerWarps,
                                      transaction_bytes);
  return stage;
}

template <class Control>
__device__ __forceinline__ void consumer_wait_stage(Control* control, int task) {
  int const stage = task & (Control::kStageCount - 1);
  unsigned int const phase = static_cast<unsigned int>(task / Control::kStageCount) & 1U;
  glm5::wait_barrier(control->ready + stage, phase);
}

template <class Control>
__device__ __forceinline__ void consumer_release_stage(Control* control, int task) {
  int const stage = task & (Control::kStageCount - 1);
  cutlass::arch::ClusterBarrier::arrive(control->empty + stage);
}

__device__ __forceinline__ void bulk_copy_g2s(void const* global_source,
                                              std::uint64_t* ready,
                                              void* shared_destination, int bytes) {
  cute::SM90_BULK_COPY_G2S::copy(global_source, ready, shared_destination, bytes);
}

template <cute::TMA::CacheHintSm100 kCachePolicy>
__device__ __forceinline__ void bulk_copy_g2s_cache_hint(void const* global_source,
                                                         std::uint64_t* ready,
                                                         void* shared_destination, int bytes) {
  std::uint32_t const shared_ready = cute::cast_smem_ptr_to_uint(ready);
  std::uint32_t const shared_destination_address =
      cute::cast_smem_ptr_to_uint(shared_destination);
  constexpr std::uint64_t cache_policy = static_cast<std::uint64_t>(kCachePolicy);
  asm volatile(
      "cp.async.bulk.shared::cluster.global."
      "mbarrier::complete_tx::bytes.L2::cache_hint "
      "[%0], [%1], %2, [%3], %4;"
      :
      : "r"(shared_destination_address), "l"(global_source), "r"(bytes), "r"(shared_ready),
        "l"(cache_policy)
      : "memory");
}

__device__ __forceinline__ void tensor_copy_2d_g2s(
    CUtensorMap const* tensor_map, int coord0, int coord1, std::uint64_t* ready,
    void* shared_destination, std::uint64_t cache_policy) {
  std::uint32_t const shared_ready = cute::cast_smem_ptr_to_uint(ready);
  std::uint32_t const shared_destination_address =
      cute::cast_smem_ptr_to_uint(shared_destination);
  asm volatile(
      "cp.async.bulk.tensor.2d.shared::cluster.global.tile."
      "mbarrier::complete_tx::bytes.L2::cache_hint "
      "[%0], [%1, {%2, %3}], [%4], %5;"
      :
      : "r"(shared_destination_address), "l"(tensor_map), "r"(coord0), "r"(coord1),
        "r"(shared_ready), "l"(cache_policy)
      : "memory");
}

__device__ __forceinline__ void tensor_copy_3d_g2s(
    CUtensorMap const* tensor_map, int coord0, int coord1, int coord2,
    std::uint64_t* ready, void* shared_destination, std::uint64_t cache_policy) {
  cute::SM90_TMA_LOAD_3D::copy(tensor_map, ready, cache_policy, shared_destination,
                               coord0, coord1, coord2);
}


// Common memory operations and configurable consumer/staging helpers.
__device__ __forceinline__ void store_release_gpu_u32(std::uint32_t* address, std::uint32_t value) {
  asm volatile("st.global.release.gpu.u32 [%0], %1;" : : "l"(address), "r"(value) : "memory");
}

__device__ __forceinline__ void store_relaxed_gpu_u32(std::uint32_t* address, std::uint32_t value) {
  asm volatile("st.global.relaxed.gpu.u32 [%0], %1;" : : "l"(address), "r"(value) : "memory");
}

__device__ __forceinline__ std::uint64_t load_relaxed_gpu_u64(std::uint64_t const* address) {
  std::uint64_t value;
  asm volatile("ld.global.relaxed.gpu.u64 %0, [%1];" : "=l"(value) : "l"(address) : "memory");
  return value;
}

__device__ __forceinline__ bool fp32_bits_are_nan(std::uint32_t bits) {
  return (bits & 0x7fffffffU) >= 0x7f800001U;
}

__device__ __forceinline__ bool router_logit_pair_has_nan(std::uint64_t pair) {
  std::uint32_t const low = static_cast<std::uint32_t>(pair);
  std::uint32_t const high = static_cast<std::uint32_t>(pair >> 32);
  return fp32_bits_are_nan(low) || fp32_bits_are_nan(high);
}

__device__ __forceinline__ void cluster_barrier_wait(std::uint64_t const* barrier,
                                                     std::uint32_t phase) {
  glm5::wait_barrier(barrier, phase);
}

template <int kConsumerThreads>
__device__ __forceinline__ void consumer_group_sync() {
  cutlass::arch::NamedBarrier::sync(kConsumerThreads,
                                    cutlass::arch::ReservedNamedBarriers::TransposeBarrier);
}

template <int kThreads>
__device__ __forceinline__ void gate_up_phase_sync() {
  cutlass::arch::NamedBarrier::sync(kThreads,
                                    cutlass::arch::ReservedNamedBarriers::TransformBarrier);
}

template <int kTmaAtomK>
__device__ __forceinline__ std::size_t tma_swizzle_128b_offset(int row, int column,
                                                               int atoms_per_row) {
  int const atom = column / kTmaAtomK;
  int const atom_column = column % kTmaAtomK;
  int const swizzle_row = row * atoms_per_row + atom;
  std::size_t const linear = static_cast<std::size_t>(swizzle_row) * kTmaAtomK + atom_column;
  return linear ^ static_cast<std::size_t>((swizzle_row & 7) << 4);
}

template <int kConsumerThreads, int kRmsInputCopiesPerConsumer,
          int kRmsInputCopyBytes>
__device__ __forceinline__ void stage_rms_input(__nv_bfloat16 const* input,
                                                __nv_bfloat16* shared_input) {
  int const tid = static_cast<int>(threadIdx.x);
  auto const* input_bytes = reinterpret_cast<std::uint8_t const*>(input);
  auto* shared_bytes = reinterpret_cast<std::uint8_t*>(shared_input);

#pragma unroll
  for (int copy = 0; copy < kRmsInputCopiesPerConsumer; ++copy) {
    int const vector = tid + copy * kConsumerThreads;
    int const byte_offset = vector * kRmsInputCopyBytes;
    copy_16b_g2s(input_bytes + byte_offset, shared_bytes + byte_offset);
  }
  cutlass::arch::cp_async_fence();
  cutlass::arch::cp_async_wait<0>();
  consumer_group_sync<kConsumerThreads>();
}

}  // namespace glm5::moe::detail
