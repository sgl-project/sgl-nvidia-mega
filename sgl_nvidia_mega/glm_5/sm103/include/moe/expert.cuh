#pragma once

// Native FP8 GateUp, Shared, Down and the validated finalizers.
#include <moe/routing.cuh>

namespace glm5::moe::detail {
template <class Config>
  requires(Config::kTokens == 4)
__device__ __forceinline__ constexpr int physical_shared_expert(Layout) {
  return kRoutedExperts;
}

template <class Config>
  requires(Config::kTokens == 4)
__device__ __forceinline__ constexpr int physical_routed_expert(Layout, int routed_expert) {
  return routed_expert;
}

template <class Config>
__device__ __forceinline__ void store_release_gpu_u16(std::uint16_t* address, std::uint16_t value) {
  asm volatile("st.global.release.gpu.u16 [%0], %1;" : : "l"(address), "h"(value) : "memory");
}

template <class Config>
__device__ __forceinline__ std::size_t gate_up_stage_weight_offset(int matrix, int row,
                                                                   int k_in_page) {
  int const chunk = k_in_page / kGateUpTmaK;
  int const atom = (k_in_page % kGateUpTmaK) / kTmaAtomK;
  int const atom_column = k_in_page % kTmaAtomK;
  return static_cast<std::size_t>(chunk) * kGateUpTmaChunkBytes +
         static_cast<std::size_t>(atom) * kGateUpTmaAtomPairBytes +
         static_cast<std::size_t>(matrix) * kGateUpTmaAtomBytes +
         tma_swizzle_128b_offset<kTmaAtomK>(row, atom_column, 1);
}

template <class Config>
  requires(Config::kTokens == 4)
__device__ __forceinline__ constexpr std::size_t gate_up_scale_matrix_stride(Layout) {
  return kOriGateUpScaleMatrixStride;
}

template <class Config, int kValidColumns>
  requires(Config::kTokens == 4)
__device__ __forceinline__ void load_layout_activation_fragment(Layout,
                                                                StagedActivation const* activation,
                                                                int activation_stride, int k_base,
                                                                Hmma::BFragment& fragment) {
  load_k_permuted_activation_fragment<kValidColumns>(activation, activation_stride, k_base,
                                                     fragment);
}

template <class Config>
  requires((Config::kTokens == 4 && Config::kTpSize == 8))
__device__ __forceinline__ void produce_gate_up_page(Layout, Glm5MegaMoeArgs const& args,
                                                     PipelineControl* control,
                                                     std::uint8_t* dynamic_storage, int expert,
                                                     int row_group, int page, int task,
                                                     int selected_slot, int producer_warp) {
  constexpr int kWeightCopiesPerProducer = kGateUpTmaWeightCopies / kProducerWarps;
  constexpr int kWeightBytesPerProducer = kWeightCopiesPerProducer * kGateUpTmaAtomPairBytes;
  std::uint64_t const cache_policy = static_cast<std::uint64_t>(
      selected_slot < kGateUpL2EvictLastSlots    ? cute::TMA::CacheHintSm100::EVICT_LAST
      : selected_slot >= kGateUpL2EvictFirstSlot ? cute::TMA::CacheHintSm100::EVICT_FIRST
                                                 : cute::TMA::CacheHintSm100::EVICT_NORMAL);
  int const transaction_bytes =
      kWeightBytesPerProducer + (producer_warp == 0 ? kGateUpScaleTmaBytes : 0);
  producer_acquire_stage(control, task, transaction_bytes);

  std::uint8_t* stage = pipeline_stage<PipelineControl>(dynamic_storage, task);
  int const stage_index = task & (kPipelineStages - 1);
  std::uint64_t* ready = control->ready + stage_index;
#pragma unroll
  for (int copy = producer_warp; copy < kGateUpTmaWeightCopies; copy += kProducerWarps) {
    int const atom = copy;
    int const chunk = atom / kGateUpTmaKAtomsPerChunk;
    int const atom_in_chunk = atom % kGateUpTmaKAtomsPerChunk;
    std::size_t const destination_offset =
        static_cast<std::size_t>(chunk) * kGateUpTmaChunkBytes +
        static_cast<std::size_t>(atom_in_chunk) * kGateUpTmaAtomPairBytes;
    tensor_copy_3d_g2s(&args.up_gate_weight_tma, page * 1024 + atom * kTmaAtomK, row_group * 16,
                       expert * 2, ready, stage + destination_offset, cache_policy);
  }
  if (producer_warp == 0) {
    int const scale_row = expert * (2 * kGateUpScaleBlocksPerMatrix) + (row_group >> 3);
    tensor_copy_2d_g2s(&args.up_gate_scale_tma, page * 8, scale_row, ready,
                       stage + kGateUpScaleOffset, cache_policy);
  }
}

template <class Config>
  requires((Config::kTokens == 4 && Config::kTpSize == 4))
__device__ __forceinline__ void produce_gate_up_page(
    Layout, Glm5MegaMoeArgs const& args, PipelineControl* control, std::uint8_t* dynamic_storage,
    int expert, int row_group, int page, int task, std::uint64_t cache_policy, int producer_warp) {
  constexpr int kWeightCopiesPerProducer = kGateUpTmaWeightCopies / kProducerWarps;
  constexpr int kWeightBytesPerProducer = kWeightCopiesPerProducer * kGateUpTmaAtomPairBytes;
  int const transaction_bytes =
      kWeightBytesPerProducer + (producer_warp == 0 ? kGateUpScaleTmaBytes : 0);
  producer_acquire_stage(control, task, transaction_bytes);

  std::uint8_t* stage = pipeline_stage<PipelineControl>(dynamic_storage, task);
  int const stage_index = task & (kPipelineStages - 1);
  std::uint64_t* ready = control->ready + stage_index;
#pragma unroll
  for (int copy = producer_warp; copy < kGateUpTmaWeightCopies; copy += kProducerWarps) {
    int const atom = copy;
    int const chunk = atom / kGateUpTmaKAtomsPerChunk;
    int const atom_in_chunk = atom % kGateUpTmaKAtomsPerChunk;
    std::size_t const destination_offset =
        static_cast<std::size_t>(chunk) * kGateUpTmaChunkBytes +
        static_cast<std::size_t>(atom_in_chunk) * kGateUpTmaAtomPairBytes;
    tensor_copy_3d_g2s(&args.up_gate_weight_tma, page * 1024 + atom * kTmaAtomK, row_group * 16,
                       expert * 2, ready, stage + destination_offset, cache_policy);
  }
  if (producer_warp == 0) {
    int const scale_row = expert * (2 * kGateUpScaleBlocksPerMatrix) + (row_group >> 3);
    tensor_copy_2d_g2s(&args.up_gate_scale_tma, page * 8, scale_row, ready,
                       stage + kGateUpScaleOffset, cache_policy);
  }
}

template <class Config>
  requires((Config::kTokens == 8 && Config::kTpSize == 8))
__device__ __forceinline__ void produce_gate_up_page(Glm5MegaMoeArgs const& args,
                                                     PipelineControl* control,
                                                     std::uint8_t* dynamic_storage, int expert,
                                                     int row_group, int page, int task,
                                                     int selected_slot, int producer_warp) {
  constexpr int kWeightCopiesPerProducer = kGateUpTmaWeightCopies / kProducerWarps;
  constexpr int kWeightBytesPerProducer = kWeightCopiesPerProducer * kGateUpTmaAtomPairBytes;
  std::uint64_t const cache_policy = static_cast<std::uint64_t>(
      selected_slot < kGateUpL2EvictLastSlots    ? cute::TMA::CacheHintSm100::EVICT_LAST
      : selected_slot >= kGateUpL2EvictFirstSlot ? cute::TMA::CacheHintSm100::EVICT_FIRST
                                                 : cute::TMA::CacheHintSm100::EVICT_NORMAL);
  int const transaction_bytes =
      kWeightBytesPerProducer + (producer_warp == 0 ? kGateUpScaleTmaBytes : 0);
  producer_acquire_stage(control, task, transaction_bytes);

  std::uint8_t* stage = pipeline_stage<PipelineControl>(dynamic_storage, task);
  int const stage_index = task & (kPipelineStages - 1);
  std::uint64_t* ready = control->ready + stage_index;
#pragma unroll
  for (int copy = producer_warp; copy < kGateUpTmaWeightCopies; copy += kProducerWarps) {
    int const chunk = copy / kGateUpTmaKAtomsPerChunk;
    int const atom_in_chunk = copy % kGateUpTmaKAtomsPerChunk;
    std::size_t const destination_offset =
        static_cast<std::size_t>(chunk) * kGateUpTmaChunkBytes +
        static_cast<std::size_t>(atom_in_chunk) * kGateUpTmaAtomPairBytes;
    tensor_copy_3d_g2s(&args.up_gate_weight_tma, page * 1024 + copy * kTmaAtomK, row_group * 16,
                       expert * 2, ready, stage + destination_offset, cache_policy);
  }
  if (producer_warp == 0) {
    int const scale_row = expert * 4 + (row_group >> 3);
    tensor_copy_2d_g2s(&args.up_gate_scale_tma, page * 8, scale_row, ready,
                       stage + kGateUpScaleOffset, cache_policy);
  }
}

template <class Config>
  requires((Config::kTokens == 8 && Config::kTpSize == 4))
__device__ __forceinline__ void produce_gate_up_page(
    Glm5MegaMoeArgs const& args, PipelineControl* control, std::uint8_t* dynamic_storage,
    int expert, int row_group, int page, int task, std::uint64_t cache_policy, int producer_warp) {
  constexpr int kWeightCopiesPerProducer = kGateUpTmaWeightCopies / kProducerWarps;
  constexpr int kWeightBytesPerProducer = kWeightCopiesPerProducer * kGateUpTmaAtomPairBytes;
  int const transaction_bytes =
      kWeightBytesPerProducer + (producer_warp == 0 ? kGateUpScaleTmaBytes : 0);
  producer_acquire_stage(control, task, transaction_bytes);

  std::uint8_t* stage = pipeline_stage<PipelineControl>(dynamic_storage, task);
  int const stage_index = task & (kPipelineStages - 1);
  std::uint64_t* ready = control->ready + stage_index;
#pragma unroll
  for (int copy = producer_warp; copy < kGateUpTmaWeightCopies; copy += kProducerWarps) {
    int const chunk = copy / kGateUpTmaKAtomsPerChunk;
    int const atom_in_chunk = copy % kGateUpTmaKAtomsPerChunk;
    std::size_t const destination_offset =
        static_cast<std::size_t>(chunk) * kGateUpTmaChunkBytes +
        static_cast<std::size_t>(atom_in_chunk) * kGateUpTmaAtomPairBytes;
    tensor_copy_3d_g2s(&args.up_gate_weight_tma, page * 1024 + copy * kTmaAtomK, row_group * 16,
                       expert * 2, ready, stage + destination_offset, cache_policy);
  }
  if (producer_warp == 0) {
    int const scale_row = expert * (2 * kGateUpScaleBlocksPerMatrix) + (row_group >> 3);
    tensor_copy_2d_g2s(&args.up_gate_scale_tma, page * 8, scale_row, ready,
                       stage + kGateUpScaleOffset, cache_policy);
  }
}

template <class Config>
  requires(Config::kTokens == 4)
__device__ __forceinline__ bool is_gate_up_producer(Layout, int tid) {
  return tid >= kConsumerThreads && (tid & (kWarpSize - 1)) == 0;
}

template <class Config>
  requires(Config::kTokens == 4)
__device__ __forceinline__ int gate_up_producer_warp(Layout, int tid) {
  return (tid - kConsumerThreads) / kWarpSize;
}

template <class Config>
  requires(Config::kTokens == 4)
__device__ __forceinline__ std::size_t down_k32_partition_scale_offset(Layout, int expert_in_pair,
                                                                       int m16, int k32_partition,
                                                                       int virtual_tile) {
  int const first_scale_block = (virtual_tile * 48) >> 7;
  int const stored_scale_block = first_scale_block & ~1;
  int const scale_block = (virtual_tile * 48 + m16 * 16) >> 7;
  int const scale_index =
      (scale_block - stored_scale_block) * kDownScaleBlocks + (k32_partition >> 2);
  return static_cast<std::size_t>(expert_in_pair) * kDownStageExpertStride +
         kDownWeightTransactionBytes + static_cast<std::size_t>(scale_index) * sizeof(float);
}

template <class Config>
  requires((Config::kTokens == 8 && Config::kTpSize == 4))
__device__ __forceinline__ std::size_t down_k32_partition_scale_offset(int expert_in_pair, int m16,
                                                                       int global_k32_partition,
                                                                       int global_m16) {
  int const virtual_tile = global_m16 / kDownM16PerVirtualTile;
  int const first_scale_block = (virtual_tile * 48) >> 7;
  int const stored_scale_block = first_scale_block & ~1;
  int const scale_block = (global_m16 * 16 + m16 * 16) >> 7;
  int const scale_index =
      (scale_block - stored_scale_block) * kDownScaleBlocks + (global_k32_partition >> 2);
  return static_cast<std::size_t>(expert_in_pair) * kDownExpertStageStride + kDownStageScaleOffset +
         static_cast<std::size_t>(scale_index) * sizeof(float);
}

template <class Config>
  requires(Config::kTokens == 4)
__device__ __forceinline__ void unpack_down_weight_fragment(Layout, uint2 words,
                                                            Hmma::AFragment& fragment) {
  unpack_k_permuted_native_a_rows(words, fragment);
}

template <class Config>
  requires(Config::kTokens == 4)
__device__ __forceinline__ std::uint8_t* ar_tile_base(void* peer_workspace, int token,
                                                      int source_slot, int phase, int tile) {
  return static_cast<std::uint8_t*>(peer_workspace) +
         static_cast<std::size_t>(token) * kArTokenStride +
         static_cast<std::size_t>(source_slot) * kArSourceStride +
         static_cast<std::size_t>(phase) * kArPhaseStride +
         static_cast<std::size_t>(tile) * kArTileStride;
}

template <class Config>
  requires((Config::kTokens == 4 && Config::kTpSize == 4))
__device__ __noinline__ void prepare_routed_gate_up_cache(
    int token, int tile, std::int32_t const* local_selected_indices,
    std::uint32_t const* hot_expert_mask, std::uint32_t const* frequency_ready,
    std::uint32_t const* ar_epoch_counters, std::uint64_t* gate_up_cache_policy,
    std::uint64_t* topk_handoff) {
  int const counter = token * kArTilesPerToken + tile;
  std::uint32_t const generation =
      next_generation<Config>(load_relaxed_gpu_u32<Config>(ar_epoch_counters + counter));
  std::uint32_t observed = load_relaxed_gpu_u32<Config>(frequency_ready);
  while (observed != generation) {
    __nanosleep(32);
    observed = load_relaxed_gpu_u32<Config>(frequency_ready);
  }
  acquire_fence_gpu<Config>();

  int const expert = local_selected_indices[tile >> 2];
  std::uint32_t const retained =
      load_relaxed_gpu_u32<Config>(hot_expert_mask + (expert >> 5)) & (1U << (expert & 31));
  *gate_up_cache_policy =
      static_cast<std::uint64_t>(retained != 0U ? cute::TMA::CacheHintSm100::EVICT_LAST
                                                : cute::TMA::CacheHintSm100::EVICT_FIRST);
  cutlass::arch::ClusterBarrier::arrive(topk_handoff);
}

template <class Config, int kElements>
  requires(Config::kTokens == 8)
__device__ __forceinline__ void convert_staged_bf16_for_mma(__nv_bfloat16 const* source,
                                                            void* destination) {
#if MEGAMOE_USE_NATIVE_E4M3X2_TO_BF16X2
  (void)source;
  (void)destination;
#else
  static_assert(kElements % 8 == 0);
  constexpr int kVectors = kElements / 8;
  int const tid = static_cast<int>(threadIdx.x);
  auto const* source_vectors = reinterpret_cast<uint4 const*>(source);
  auto* destination_vectors = reinterpret_cast<uint4*>(destination);
#pragma unroll 1
  for (int vector = tid; vector < kVectors; vector += kConsumerThreads) {
    uint4 values = comm::load_shared_u32x4(source_vectors + vector);
    values.x = Hmma::from_bf16x2(values.x);
    values.y = Hmma::from_bf16x2(values.y);
    values.z = Hmma::from_bf16x2(values.z);
    values.w = Hmma::from_bf16x2(values.w);
    comm::store_shared_u32x4(destination_vectors + vector, values);
  }
#endif
}

template <class Config>
  requires(Config::kTokens == 8)
__device__ __forceinline__ void store_up_gate_value(__nv_bfloat16* destination,
                                                    __nv_bfloat16 value) {
  *destination = value;
}

template <class Config>
  requires(Config::kTokens == 8)
__device__ __forceinline__ void issue_shared_activation_page(__nv_bfloat16 const* global_normalized,
                                                             int page,
                                                             __nv_bfloat16* activation_page) {
  int const tid = static_cast<int>(threadIdx.x);
  auto const* global_bytes = reinterpret_cast<std::uint8_t const*>(global_normalized);
  auto* page_bytes = reinterpret_cast<std::uint8_t*>(activation_page);
  constexpr int kVectorsPerTokenPage = 1024 * sizeof(__nv_bfloat16) / kRmsInputCopyBytes;

#pragma unroll 1
  for (int vector = tid; vector < kSharedActivationPageCopyVectors; vector += kConsumerThreads) {
    int const token = vector / kVectorsPerTokenPage;
    int const vector_in_token = vector % kVectorsPerTokenPage;
    int const page_byte_offset = vector_in_token * kRmsInputCopyBytes;
    copy_16b_g2s(global_bytes + static_cast<std::size_t>(token) * kHidden * sizeof(__nv_bfloat16) +
                     page * 1024 * sizeof(__nv_bfloat16) + page_byte_offset,
                 page_bytes + vector * kRmsInputCopyBytes);
  }
  cutlass::arch::cp_async_fence();
}

template <class Config, int kTmaProducer, int kExpertsInPair>
  requires(Config::kTokens == 8)
__device__ __forceinline__ constexpr int down_producer_transaction_bytes() {
  static_assert(kTmaProducer >= 0 && kTmaProducer < kDownTmaIssueWarps);
  static_assert(kExpertsInPair == 1 || kExpertsInPair == 2);
  if constexpr (kTmaProducer == 0) {
    return kDownWeightTransactionBytes;
  } else if constexpr (kTmaProducer == 1) {
    return kExpertsInPair == 2 ? kDownWeightTransactionBytes : 0;
  } else {
    return kExpertsInPair * kDownNativeScaleTransactionBytes;
  }
}

template <class Config, int kTmaProducer, int kExpertInPair>
  requires((Config::kTokens == 8 && Config::kTpSize == 8))
__device__ __forceinline__ void issue_down_expert_tma(Glm5MegaMoeArgs const& args, int expert,
                                                      int virtual_tile, int stored_scale_block,
                                                      std::uint64_t* ready,
                                                      std::uint8_t* expert_stage) {
  static_assert(kTmaProducer >= 0 && kTmaProducer < kDownTmaIssueWarps);
  static_assert(kExpertInPair == 0 || kExpertInPair == 1);

  if constexpr (kTmaProducer == kExpertInPair) {
    int const weight_atom = (expert * kHidden + virtual_tile * 48) * kDownTmaKAtoms;
    tensor_copy_2d_g2s(&args.down_weight_tma, 0, weight_atom, ready, expert_stage,
                       static_cast<std::uint64_t>(cute::TMA::CacheHintSm100::EVICT_FIRST));
  } else if constexpr (kTmaProducer == 2) {
    bulk_copy_g2s(
        args.down_scale + (static_cast<std::size_t>(expert) * 48 + stored_scale_block) * 2, ready,
        expert_stage + kDownStageScaleOffset, 0x20);
  }
}

template <class Config>
  requires((Config::kTokens == 8 && Config::kTpSize == 4))
__device__ __forceinline__ std::size_t down_native_stage_weight_offset(int row, int k) {
  return tma_swizzle_128b_offset<kTmaAtomK>(row, k, kDownStagedTmaKAtoms);
}

template <class Config, int kSlot, class Layout>
  requires(Config::kTokens == 4)
__device__ __forceinline__ int physical_expert_for_down_slot(
    Layout layout, std::int32_t const* local_selected_indices) {
  static_assert(kSlot >= 0 && kSlot < kDownActivationSlots);
  if constexpr (kSlot == 0) {
    return physical_shared_expert<Config>(layout);
  } else {
    return physical_routed_expert<Config>(layout, local_selected_indices[kSlot - 1]);
  }
}

template <class Config, int kSlot>
  requires(Config::kTokens == 8)
__device__ __forceinline__ int physical_expert_for_down_slot(
    std::int32_t const* local_selected_indices) {
  static_assert(kSlot >= 0 && kSlot < kDownActivationSlots);
  if constexpr (kSlot == 0) {
    return kRoutedExperts;
  } else {
    return local_selected_indices[kSlot - 1];
  }
}

template <class Config>
  requires(Config::kTokens == 4)
__device__ __forceinline__ std::uint32_t finalize_bf16x2(std::uint32_t reduced,
                                                         std::uint32_t residual) {
  float const reduced0 =
      __bfloat162float(__ushort_as_bfloat16(static_cast<std::uint16_t>(reduced)));
  float const reduced1 =
      __bfloat162float(__ushort_as_bfloat16(static_cast<std::uint16_t>(reduced >> 16)));
  float const residual0 =
      __bfloat162float(__ushort_as_bfloat16(static_cast<std::uint16_t>(residual)));
  float const residual1 =
      __bfloat162float(__ushort_as_bfloat16(static_cast<std::uint16_t>(residual >> 16)));
  return comm::pack_bf16x2(reduced0 + residual0, reduced1 + residual1);
}

template <class Config>
  requires(Config::kTokens == 4)
__device__ __forceinline__ uint2 pack_bf16x4(float const (&values)[kArReductionValuesPerThread]) {
  return make_uint2(comm::pack_bf16x2(values[0], values[1]),
                    comm::pack_bf16x2(values[2], values[3]));
}

template <class Config, class Layout>
  requires(Config::kTokens == 4)
__device__ __forceinline__ std::size_t gate_up_stage_scale_offset(Layout layout, int matrix,
                                                                  int k_in_page) {
  int const k_block_in_page = k_in_page >> 7;
  return kGateUpScaleOffset +
         static_cast<std::size_t>(matrix) * gate_up_scale_matrix_stride<Config>(layout) +
         static_cast<std::size_t>(k_block_in_page) * sizeof(float);
}

template <class Config>
  requires(Config::kTokens == 8)
__device__ __forceinline__ std::size_t gate_up_stage_scale_offset(int matrix, int k_in_page) {
  int const k_block_in_page = k_in_page >> 7;
  return kGateUpScaleOffset + static_cast<std::size_t>(matrix) * kGateUpScaleMatrixStride +
         static_cast<std::size_t>(k_block_in_page) * sizeof(float);
}

template <class Config>
  requires(Config::kTokens == 4)
__device__ __forceinline__ void load_gate_up_stage_weight_fragment(Layout layout,
                                                                   std::uint8_t const* stage,
                                                                   int matrix, int k_in_page,
                                                                   Hmma::AFragment& fragment) {
  int const lane = static_cast<int>(threadIdx.x) & 31;
  int const row = lane & 15;
  std::uint8_t const* row_address =
      stage + gate_up_stage_weight_offset<Config>(matrix, row, k_in_page);
  unpack_k_permuted_native_a_rows(load_k_permuted_native_a_rows(row_address), fragment);
}

template <class Config>
  requires(Config::kTokens == 8)
__device__ __forceinline__ void load_gate_up_stage_weight_fragment(std::uint8_t const* stage,
                                                                   int matrix, int k_in_page,
                                                                   Hmma::AFragment& fragment) {
  int const lane = static_cast<int>(threadIdx.x) & 31;
  int const row = lane & 15;
  std::uint8_t const* row_address =
      stage + gate_up_stage_weight_offset<Config>(matrix, row, k_in_page);
  unpack_k_permuted_native_a_rows(load_k_permuted_native_a_rows(row_address), fragment);
}

template <class Config, int kElements>
  requires(Config::kTokens == 4)
__device__ __forceinline__ StagedActivation const* stage_bf16_for_mma(__nv_bfloat16 const* source,
                                                                      void* destination) {
#if MEGAMOE_USE_NATIVE_E4M3X2_TO_BF16X2
  (void)destination;
  return source;
#else
  static_assert(kElements % 8 == 0);
  constexpr int kVectors = kElements / 8;
  int const tid = static_cast<int>(threadIdx.x);
  auto const* source_vectors = reinterpret_cast<uint4 const*>(source);
  auto* destination_vectors = reinterpret_cast<uint4*>(destination);
#pragma unroll 1
  for (int vector = tid; vector < kVectors; vector += kConsumerThreads) {
    uint4 values = comm::load_shared_u32x4(source_vectors + vector);
    values.x = Hmma::from_bf16x2(values.x);
    values.y = Hmma::from_bf16x2(values.y);
    values.z = Hmma::from_bf16x2(values.z);
    values.w = Hmma::from_bf16x2(values.w);
    comm::store_shared_u32x4(destination_vectors + vector, values);
  }
  consumer_group_sync<kConsumerThreads>();
  return reinterpret_cast<StagedActivation const*>(destination);
#endif
}

template <class Config, int kElements>
  requires(Config::kTokens == 8)
__device__ __forceinline__ StagedActivation const* stage_bf16_for_mma(__nv_bfloat16 const* source,
                                                                      void* destination) {
#if MEGAMOE_USE_NATIVE_E4M3X2_TO_BF16X2
  (void)destination;
  return source;
#else
  convert_staged_bf16_for_mma<Config, kElements>(source, destination);
  consumer_group_sync<kConsumerThreads>();
  return reinterpret_cast<StagedActivation const*>(destination);
#endif
}

template <class Config>
__device__ __forceinline__ void publish_up_gate_sentinel(__nv_bfloat16* sentinel,
                                                         std::uint16_t value) {
  auto* address = reinterpret_cast<std::uint16_t*>(sentinel);
  store_release_gpu_u16<Config>(address, value);
}

template <class Config>
  requires(Config::kTokens == 4)
__device__ __forceinline__ void wait_up_gate_sentinels(__nv_bfloat16 const* up_gate, int token) {
  int const tid = static_cast<int>(threadIdx.x);
  if (tid < kUpGateSentinelsPerToken) {
    int slot;
    int channel;
    if (tid < kSharedGateUpSentinelsPerToken) {
      slot = 0;
      channel = tid * 16 + 15;
    } else {
      int const routed_chunk = tid - kSharedGateUpSentinelsPerToken;
      slot = routed_chunk / 4 + 1;
      channel = (routed_chunk & 3) * kRoutedGateUpRowsPerCta + kRoutedGateUpRowsPerCta - 1;
    }
    std::size_t const sentinel_index =
        (static_cast<std::size_t>(token) * 9 + slot) * kLocalInter + channel;
    auto const* address = reinterpret_cast<std::uint16_t const*>(up_gate + sentinel_index);
    std::uint16_t value = comm::load_relaxed_gpu_u16(address);
    while (bf16_bits_are_nan<Config>(value)) {
      __nanosleep(32);
      value = comm::load_relaxed_gpu_u16(address);
    }
    acquire_fence_gpu<Config>();
  }
  consumer_group_sync<kConsumerThreads>();
}

template <class Config>
  requires(Config::kTokens == 8)
__device__ __forceinline__ void wait_up_gate_sentinels(__nv_bfloat16 const* up_gate, int token) {
  int const tid = static_cast<int>(threadIdx.x);
  if (tid < kSharedRowGroups + kRoutedCtasPerToken) {
    int slot;
    int channel;
    if (tid < kSharedRowGroups) {
      slot = 0;
      channel = tid * 16 + 15;
    } else {
      int const routed_local = tid - kSharedRowGroups;
      RoutedCtaDescriptor const descriptor = routed_cta_descriptor<Config>(routed_local);
      int const last_flat = descriptor.gate_flat_base + descriptor.gate_group_count - 1;
      slot = Config::kTpSize == 8 ? (last_flat >> 4) + 1
                                  : last_flat / kGateUpRowGroupsPerExpert + 1;
      channel = Config::kTpSize == 8 ? (last_flat & 15) * 16 + 15
                                     : (last_flat % kGateUpRowGroupsPerExpert) * 16 + 15;
    }
    std::size_t const sentinel_index =
        (static_cast<std::size_t>(token) * 9 + slot) * kLocalInter + channel;
    auto const* address = reinterpret_cast<std::uint16_t const*>(up_gate + sentinel_index);
    std::uint16_t value = comm::load_relaxed_gpu_u16(address);
    while (bf16_bits_are_nan<Config>(value)) {
      __nanosleep(32);
      value = comm::load_relaxed_gpu_u16(address);
    }
    comm::acquire_fence_gpu();
  }
  consumer_group_sync<kConsumerThreads>();
}

template <class Config, class Args>
  requires(Config::kTokens == 4)
__device__ __forceinline__ void reset_inputs_from_producer_warp(Args const& args,
                                                                __nv_bfloat16* up_gate, int token,
                                                                std::uint8_t* dynamic_storage) {
  // Tile 0's Down producer warp is idle after issuing the persistent TMA
  // pipeline.  Use that otherwise hidden tail to wait for all 32 staging
  // epochs and restore the next launch's sentinels off the consumer path.
  int const lane = static_cast<int>(threadIdx.x) - kConsumerThreads;
  auto const* generation_scratch =
      reinterpret_cast<std::uint32_t const*>(dynamic_storage + kArRemoteOffset);
  std::uint32_t const expected = generation_scratch[0];
  std::uint32_t observed =
      load_relaxed_gpu_u32<Config>(args.ar_epoch_counters + token * kArTilesPerToken + lane);
  while (observed != expected) {
    asm volatile("" : : : "memory");
    observed =
        load_relaxed_gpu_u32<Config>(args.ar_epoch_counters + token * kArTilesPerToken + lane);
  }
  acquire_fence_gpu<Config>();
  __syncwarp(kFullWarpMask);

  auto* workspace = static_cast<std::uint8_t*>(args.workspace);
  auto* router_vectors =
      reinterpret_cast<uint4*>(workspace + kRouterLogitsOffset +
                               static_cast<std::size_t>(token) * kRoutedExperts * sizeof(float));
  uint4 const not_ready =
      make_uint4(kGlobalNotReady, kGlobalNotReady, kGlobalNotReady, kGlobalNotReady);
  comm::store_relaxed_gpu_u32x4(router_vectors + lane, not_ready);
  comm::store_relaxed_gpu_u32x4(router_vectors + lane + kArTilesPerToken, not_ready);

  {
    int const slot = (lane >> 2) + 1;
    int const channel = (lane & 3) * kRoutedGateUpRowsPerCta + kRoutedGateUpRowsPerCta - 1;
    std::size_t const sentinel_index =
        (static_cast<std::size_t>(token) * kDownActivationSlots + slot) * kLocalInter + channel;
    comm::store_relaxed_gpu_u16(reinterpret_cast<std::uint16_t*>(up_gate + sentinel_index),
                                  kBf16NotReady);
  }
  if (lane < kSharedGateUpSentinelsPerToken) {
    std::size_t const sentinel_index =
        static_cast<std::size_t>(token) * kDownActivationSlots * kLocalInter + lane * 16 + 15;
    comm::store_relaxed_gpu_u16(reinterpret_cast<std::uint16_t*>(up_gate + sentinel_index),
                                  kBf16NotReady);
  }
  if (lane == 0) {
    auto* normalized_ready = reinterpret_cast<std::uint32_t*>(workspace + kNormalizedReadyOffset);
    store_relaxed_gpu_u32(normalized_ready + token, kGlobalNotReady);
  }
}

template <class Config>
  requires(Config::kTokens == 8)
__device__ __forceinline__ void reset_inputs_from_producer_warp(Glm5MegaMoeArgs const& args,
                                                                __nv_bfloat16* up_gate, int token,
                                                                std::uint8_t* dynamic_storage) {
  int const lane =
      static_cast<int>(threadIdx.x) - kConsumerThreads - kDownResetProducerWarp * kWarpSize;
  auto const* generation_scratch =
      reinterpret_cast<std::uint32_t const*>(dynamic_storage + kArRemoteOffset);
  std::uint32_t const expected = generation_scratch[0];

  if (lane < kRoutedCtasPerToken) {
    std::uint32_t observed =
        comm::load_relaxed_gpu_u32(args.ar_epoch_counters + token * kRoutedCtasPerToken + lane);
    while (observed != expected) {
      asm volatile("" : : : "memory");
      observed =
          comm::load_relaxed_gpu_u32(args.ar_epoch_counters + token * kRoutedCtasPerToken + lane);
    }
  }
  comm::acquire_fence_gpu();
  __syncwarp(kFullWarpMask);

  auto* workspace = static_cast<std::uint8_t*>(args.workspace);
  auto* router_vectors =
      reinterpret_cast<uint4*>(workspace + kRouterLogitsOffset +
                               static_cast<std::size_t>(token) * kRoutedExperts * sizeof(float));
  uint4 const not_ready =
      make_uint4(kGlobalNotReady, kGlobalNotReady, kGlobalNotReady, kGlobalNotReady);
  comm::store_relaxed_gpu_u32x4(router_vectors + lane, not_ready);
  comm::store_relaxed_gpu_u32x4(router_vectors + lane + kWarpSize, not_ready);

  if (lane < kRoutedCtasPerToken) {
    RoutedCtaDescriptor const descriptor = routed_cta_descriptor<Config>(lane);
    int const last_flat = descriptor.gate_flat_base + descriptor.gate_group_count - 1;
    int const slot = Config::kTpSize == 8 ? (last_flat >> 4) + 1
                                         : last_flat / kGateUpRowGroupsPerExpert + 1;
    int const channel = Config::kTpSize == 8 ? (last_flat & 15) * 16 + 15
                                            : (last_flat % kGateUpRowGroupsPerExpert) * 16 + 15;
    std::size_t const sentinel_index =
        (static_cast<std::size_t>(token) * kDownActivationSlots + slot) * kLocalInter + channel;
    comm::store_relaxed_gpu_u16(reinterpret_cast<std::uint16_t*>(up_gate + sentinel_index),
                               kBf16NotReady);
  }
  if (lane < kSharedRowGroups) {
    std::size_t const sentinel_index =
        static_cast<std::size_t>(token) * kDownActivationSlots * kLocalInter + lane * 16 + 15;
    comm::store_relaxed_gpu_u16(reinterpret_cast<std::uint16_t*>(up_gate + sentinel_index),
                               kBf16NotReady);
  }
  if (lane == 0) {
    auto* normalized_ready = reinterpret_cast<std::uint32_t*>(workspace + kNormalizedReadyOffset);
    store_relaxed_gpu_u32(normalized_ready + token, kGlobalNotReady);
  }
}

template <class Config, class Layout>
  requires(Config::kTokens == 4)
__device__ __forceinline__ float load_down_k32_partition_scale(Layout layout,
                                                               std::uint8_t const* stage,
                                                               int expert_in_pair, int m16,
                                                               int k32_partition,
                                                               int virtual_tile) {
  int const lane = static_cast<int>(threadIdx.x) & 31;
  float scale = 0.0F;
  if (lane == 0) {
    std::size_t const scale_offset = down_k32_partition_scale_offset<Config>(
        layout, expert_in_pair, m16, k32_partition, virtual_tile);
    scale = *reinterpret_cast<float const*>(stage + scale_offset);
  }
  return __shfl_sync(kFullWarpMask, scale, 0);
}

template <class Config>
  requires((Config::kTokens == 8 && Config::kTpSize == 8))
__device__ __forceinline__ float load_down_k32_partition_scale(std::uint8_t const* stage,
                                                               int expert_in_pair, int m16,
                                                               int warp, int global_m16) {
  int const lane = static_cast<int>(threadIdx.x) & 31;
  float scale = 0.0F;
  if (lane == 0) {
    int const virtual_tile = global_m16 / kDownM16PerVirtualTile;
    int const first_scale_block = (virtual_tile * 48) >> 7;
    int const stored_scale_block = first_scale_block & ~1;
    int const scale_block = (global_m16 * 16 + m16 * 16) >> 7;
    int const scale_index = (scale_block - stored_scale_block) * 2 + (warp >> 2);
    std::size_t const scale_offset =
        static_cast<std::size_t>(expert_in_pair) * kDownExpertStageStride + kDownStageScaleOffset +
        static_cast<std::size_t>(scale_index) * sizeof(float);
    scale = *reinterpret_cast<float const*>(stage + scale_offset);
  }
  return __shfl_sync(kFullWarpMask, scale, 0);
}

template <class Config>
  requires((Config::kTokens == 8 && Config::kTpSize == 4))
__device__ __forceinline__ float load_down_k32_partition_scale(std::uint8_t const* stage,
                                                               int expert_in_pair, int m16,
                                                               int global_k32_partition,
                                                               int global_m16) {
  int const lane = static_cast<int>(threadIdx.x) & 31;
  float scale = 0.0F;
  if (lane == 0) {
    std::size_t const scale_offset = down_k32_partition_scale_offset<Config>(
        expert_in_pair, m16, global_k32_partition, global_m16);
    scale = *reinterpret_cast<float const*>(stage + scale_offset);
  }
  return __shfl_sync(kFullWarpMask, scale, 0);
}

template <class Config>
  requires(Config::kTokens == 4)
__device__ __forceinline__ uint4 load_down_k32_partition_weight(Layout, std::uint8_t const* stage,
                                                                int expert_in_pair, int m16,
                                                                int staged_k32_partition) {
  int const lane = static_cast<int>(threadIdx.x) & 31;
  int const row = m16 * 16 + (lane >> 2);
  int const k_quad = (lane & 3) * 4;
  std::uint8_t const* expert_stage =
      stage + static_cast<std::size_t>(expert_in_pair) * kDownStageExpertStride;
  auto load_fragment = [&](int k) {
    return make_uint2(load_shared_u32(expert_stage + tma_swizzle_128b_offset<kTmaAtomK>(
                                                         row, k, kDownStagedTmaKAtoms)),
                      load_shared_u32(expert_stage + tma_swizzle_128b_offset<kTmaAtomK>(
                                                         row + 8, k, kDownStagedTmaKAtoms)));
  };
  uint2 const low = load_fragment(staged_k32_partition * 32 + k_quad);
  uint2 const high = load_fragment(staged_k32_partition * 32 + 16 + k_quad);
  return make_uint4(low.x, low.y, high.x, high.y);
}

template <class Config>
  requires((Config::kTokens == 8 && Config::kTpSize == 8))
__device__ __forceinline__ uint4 load_down_k32_partition_weight(std::uint8_t const* stage,
                                                                int expert_in_pair, int m16,
                                                                int warp) {
  int const lane = static_cast<int>(threadIdx.x) & 31;
  int const row = m16 * 16 + (lane >> 2);
  int const k_quad = (lane & 3) * 4;
  std::uint8_t const* expert_stage =
      stage + static_cast<std::size_t>(expert_in_pair) * kDownExpertStageStride;
  auto load_fragment = [&](int k) {
    return make_uint2(
        load_shared_u32(expert_stage + tma_swizzle_128b_offset<kTmaAtomK>(row, k, kDownTmaKAtoms)),
        load_shared_u32(expert_stage +
                        tma_swizzle_128b_offset<kTmaAtomK>(row + 8, k, kDownTmaKAtoms)));
  };
  uint2 const low = load_fragment(warp * 32 + k_quad);
  uint2 const high = load_fragment(warp * 32 + 16 + k_quad);
  return make_uint4(low.x, low.y, high.x, high.y);
}

template <class Config>
  requires((Config::kTokens == 8 && Config::kTpSize == 4))
__device__ __forceinline__ uint4 load_down_k32_partition_weight(std::uint8_t const* stage,
                                                                int expert_in_pair, int m16,
                                                                int staged_k32_partition) {
  int const lane = static_cast<int>(threadIdx.x) & 31;
  int const row = m16 * 16 + (lane >> 2);
  int const k_quad = (lane & 3) * 4;
  std::uint8_t const* expert_stage =
      stage + static_cast<std::size_t>(expert_in_pair) * kDownExpertStageStride;
  auto load_fragment = [&](int k) {
    return make_uint2(
        load_shared_u32(expert_stage + down_native_stage_weight_offset<Config>(row, k)),
        load_shared_u32(expert_stage + down_native_stage_weight_offset<Config>(row + 8, k)));
  };
  uint2 const low = load_fragment(staged_k32_partition * 32 + k_quad);
  uint2 const high = load_fragment(staged_k32_partition * 32 + 16 + k_quad);
  return make_uint4(low.x, low.y, high.x, high.y);
}

template <class Config, class Args>
  requires(Config::kTokens == 4)
__device__ __forceinline__ void store_deferred_local_output(Args const& args, int token, int tile,
                                                            uint4 const* local_vectors) {
  int const warp = static_cast<int>(threadIdx.x) >> 5;
  int const lane = static_cast<int>(threadIdx.x) & (kWarpSize - 1);
  if (args.local_output == nullptr || warp >= kArProducerWarps ||
      lane >= kArPayloadVectorsPerProducer) {
    return;
  }

  int const vector = warp * kArPayloadVectorsPerProducer + lane;
  auto* output = reinterpret_cast<uint4*>(static_cast<__nv_bfloat16*>(args.local_output) +
                                          static_cast<std::size_t>(token) * kHidden +
                                          static_cast<std::size_t>(tile) * kArTileElements);
  comm::store_relaxed_gpu_u32x4(output + vector,
                                  comm::load_shared_u32x4(local_vectors + vector));
}

template <class Config>
  requires(Config::kTokens == 4)
__device__ __forceinline__ void wait_ar_marker(uint4 const* address, std::uint32_t expected) {
  uint4 observed;
  do {
    observed = comm::load_relaxed_system_u32x4(address);
    comm::poll_backoff();
  } while (observed.z != expected || observed.w != expected);
}

template <class Config>
  requires(Config::kTokens == 8)
__device__ __forceinline__ void wait_shared_activation_page(__nv_bfloat16* activation_page) {
  cutlass::arch::cp_async_wait<0>();
  convert_staged_bf16_for_mma<Config, kSharedActivationPageElements>(activation_page,
                                                                     activation_page);
  consumer_group_sync<kConsumerThreads>();
}

template <class Config, class Layout>
  requires(Config::kTokens == 4)
__device__ __forceinline__ float load_gate_up_stage_scale(Layout layout, std::uint8_t const* stage,
                                                          int matrix, int k_in_page) {
  int const lane = static_cast<int>(threadIdx.x) & 31;
  float scale = 0.0F;
  if (lane == 0) {
    std::size_t const scale_offset = gate_up_stage_scale_offset<Config>(layout, matrix, k_in_page);
    scale = *reinterpret_cast<float const*>(stage + scale_offset);
  }
  return __shfl_sync(kFullWarpMask, scale, 0);
}

template <class Config>
  requires(Config::kTokens == 8)
__device__ __forceinline__ float load_gate_up_stage_scale(std::uint8_t const* stage, int matrix,
                                                          int k_in_page) {
  int const lane = static_cast<int>(threadIdx.x) & 31;
  float scale = 0.0F;
  if (lane == 0) {
    std::size_t const scale_offset = gate_up_stage_scale_offset<Config>(matrix, k_in_page);
    scale = *reinterpret_cast<float const*>(stage + scale_offset);
  }
  return __shfl_sync(kFullWarpMask, scale, 0);
}

template <class Config, int kDownProducer, int kExpertPair>
  requires((Config::kTokens == 4 && Config::kTpSize == 8))
__device__ __forceinline__ void produce_down_expert_pair(Layout layout, Glm5MegaMoeArgs const& args,
                                                         PipelineControl* control,
                                                         std::uint8_t* dynamic_storage,
                                                         std::int32_t const* local_selected_indices,
                                                         int tile, int virtual_tile_in_cta,
                                                         int down_k_task, int task) {
  static_assert(kDownProducer >= 0 && kDownProducer < kDownTmaIssueWarps);
  static_assert(kExpertPair >= 0);
  static_assert(kExpertPair < kDownExpertPairs);
  constexpr int kExpertsInPair = kExpertPair + 1 == kDownExpertPairs ? 1 : 2;
  constexpr int kFirstSlot = kExpertPair * 2;
  constexpr bool kSecondExpert = kDownProducer == 1;
  constexpr bool kActiveProducer = !kSecondExpert || kExpertsInPair == 2;
  constexpr int kTransactionBytes =
      kActiveProducer ? kDownWeightTransactionBytes + kDownNativeScaleTransactionBytes : 0;

  int const virtual_tile = tile * kDownVirtualTilesPerCta + virtual_tile_in_cta;
  if constexpr (kDownProducer == 0) {
    int const stage_index = task & (kPipelineStages - 1);
    unsigned int const cycle = static_cast<unsigned int>(task / kPipelineStages);
    if (cycle != 0) {
      unsigned int const empty_phase = (cycle - 1U) & 1U;
      cluster_barrier_wait(control->empty + stage_index, empty_phase);
    }
    constexpr int kArriveCount = kProducerWarps - kDownTmaIssueWarps + 1;
    producer_arrive_and_expect_tx_count(control->ready + stage_index, kArriveCount,
                                        kTransactionBytes);
  } else {
    producer_acquire_stage(control, task, kTransactionBytes);
  }

  std::uint8_t* stage = pipeline_stage<PipelineControl>(dynamic_storage, task);
  std::uint64_t* ready = control->ready + (task & (kPipelineStages - 1));
  if constexpr (kActiveProducer) {
    constexpr int kSlot = kFirstSlot + (kSecondExpert ? 1 : 0);
    int const expert = physical_expert_for_down_slot<Config, kSlot>(layout, local_selected_indices);
    std::uint8_t* expert_stage = stage + (kSecondExpert ? kDownStageExpertStride : 0);
    constexpr std::uint64_t kCachePolicy =
        static_cast<std::uint64_t>(cute::TMA::CacheHintSm100::EVICT_FIRST);
    int const weight_atom = (expert * kHidden + virtual_tile * 48) * kDownPhysicalTmaKAtoms;
    tensor_copy_2d_g2s(&args.down_weight_tma, 0, weight_atom, ready, expert_stage, kCachePolicy);
    int const first_scale_block = (virtual_tile * 48) >> 7;
    int const stored_scale_block = first_scale_block & ~1;
    bulk_copy_g2s(args.down_scale + (static_cast<std::size_t>(expert) * 48 + stored_scale_block) *
                                        kDownScaleBlocks,
                  ready, expert_stage + kDownWeightTransactionBytes,
                  kDownNativeScaleTransactionBytes);
  }
}

template <class Config, int kDownProducer, int kExpertPair>
  requires((Config::kTokens == 4 && Config::kTpSize == 4))
__device__ __forceinline__ void produce_down_expert_pair(Layout layout, Glm5MegaMoeArgs const& args,
                                                         PipelineControl* control,
                                                         std::uint8_t* dynamic_storage,
                                                         std::int32_t const* local_selected_indices,
                                                         int tile, int virtual_tile_in_cta,
                                                         int down_k_task, int task) {
  static_assert(kDownProducer >= 0 && kDownProducer < kDownTmaIssueWarps);
  static_assert(kExpertPair >= 0);
  static_assert(kExpertPair < kDownExpertPairs);
  constexpr int kExpertsInPair = kExpertPair + 1 == kDownExpertPairs ? 1 : 2;
  constexpr int kFirstSlot = kExpertPair * 2;
  constexpr bool kSecondExpert = kDownProducer == 1;
  constexpr bool kActiveProducer = !kSecondExpert || kExpertsInPair == 2;
  constexpr int kTransactionBytes =
      kActiveProducer ? kDownWeightTransactionBytes + kDownNativeScaleTransactionBytes : 0;

  int const virtual_tile = tile * kDownVirtualTilesPerCta + virtual_tile_in_cta;
  if constexpr (kDownProducer == 0) {
    int const stage_index = task & (kPipelineStages - 1);
    unsigned int const cycle = static_cast<unsigned int>(task / kPipelineStages);
    if (cycle != 0) {
      unsigned int const empty_phase = (cycle - 1U) & 1U;
      cluster_barrier_wait(control->empty + stage_index, empty_phase);
    }
    constexpr int kArriveCount = kProducerWarps - kDownTmaIssueWarps + 1;
    producer_arrive_and_expect_tx_count(control->ready + stage_index, kArriveCount,
                                        kTransactionBytes);
  } else {
    producer_acquire_stage(control, task, kTransactionBytes);
  }

  std::uint8_t* stage = pipeline_stage<PipelineControl>(dynamic_storage, task);
  std::uint64_t* ready = control->ready + (task & (kPipelineStages - 1));
  if constexpr (kActiveProducer) {
    constexpr int kSlot = kFirstSlot + (kSecondExpert ? 1 : 0);
    int const expert = physical_expert_for_down_slot<Config, kSlot>(layout, local_selected_indices);
    std::uint8_t* expert_stage = stage + (kSecondExpert ? kDownStageExpertStride : 0);
    constexpr std::uint64_t kCachePolicy =
        static_cast<std::uint64_t>(cute::TMA::CacheHintSm100::EVICT_FIRST);
    tensor_copy_3d_g2s(&args.down_weight_tma, 0, down_k_task * kDownStagedTmaKAtoms,
                       expert * kHidden + virtual_tile * 48, ready, expert_stage, kCachePolicy);
    int const first_scale_block = (virtual_tile * 48) >> 7;
    int const stored_scale_block = first_scale_block & ~1;
    bulk_copy_g2s(args.down_scale + (static_cast<std::size_t>(expert) * 48 + stored_scale_block) *
                                        kDownScaleBlocks,
                  ready, expert_stage + kDownWeightTransactionBytes,
                  kDownNativeScaleTransactionBytes);
  }
}

template <class Config>
__device__ __forceinline__ StagedActivation const* stage_up_gate_activations(
    __nv_bfloat16 const* up_gate, int token, std::uint8_t* dynamic_storage) {
  int const tid = static_cast<int>(threadIdx.x);
  auto const* source = reinterpret_cast<std::uint8_t const*>(
      up_gate + static_cast<std::size_t>(token) * kDownActivationSlots * kLocalInter);
  std::uint8_t* destination = dynamic_storage + kAuxOffset;

#pragma unroll 1
  for (int vector = tid; vector < kDownActivationCopyVectors; vector += kConsumerThreads) {
    int const byte_offset = vector * kRmsInputCopyBytes;
    copy_16b_g2s(source + byte_offset, destination + byte_offset);
  }
  cutlass::arch::cp_async_fence();
  cutlass::arch::cp_async_wait<0>();
  consumer_group_sync<kConsumerThreads>();
  return stage_bf16_for_mma<Config, kDownActivationSlots * kLocalInter>(
      reinterpret_cast<__nv_bfloat16 const*>(destination), destination);
}

template <class Config, class Layout>
  requires(Config::kTokens == 4)
__device__ __forceinline__ void accumulate_down_k32_partition(
    Layout layout, std::uint8_t const* stage, StagedActivation const* activation,
    int expert_in_pair, int staged_k32_partition, int global_k32_partition, int virtual_tile,
    float probability, float (&partial_low)[3], float (&partial_high)[3]) {
  int const thread_in_group = static_cast<int>(threadIdx.x) & 3;
  int const k32_base = global_k32_partition * 32;

  Hmma::BFragment activation_fragment_low;
  Hmma::BFragment activation_fragment_high;
  load_layout_activation_fragment<Config, 1>(layout, activation, kLocalInter, k32_base,
                                             activation_fragment_low);
  load_layout_activation_fragment<Config, 1>(layout, activation, kLocalInter, k32_base + 16,
                                             activation_fragment_high);

#pragma unroll
  for (int m16 = 0; m16 < 3; ++m16) {
    float const scale = load_down_k32_partition_scale<Config>(layout, stage, expert_in_pair, m16,
                                                              global_k32_partition, virtual_tile);
    uint4 const packed_words = load_down_k32_partition_weight<Config>(layout, stage, expert_in_pair,
                                                                      m16, staged_k32_partition);
    Hmma::Accumulator accumulator{{0.0F, 0.0F, 0.0F, 0.0F}};
    Hmma::AFragment weight_fragment;

    unpack_down_weight_fragment<Config>(layout, make_uint2(packed_words.x, packed_words.y),
                                        weight_fragment);
    Hmma::fma(accumulator, weight_fragment, activation_fragment_low);

    unpack_down_weight_fragment<Config>(layout, make_uint2(packed_words.z, packed_words.w),
                                        weight_fragment);
    Hmma::fma(accumulator, weight_fragment, activation_fragment_high);

    if (thread_in_group == 0) {
      float const weighted_low = accumulator.element[0] * probability;
      float const weighted_high = accumulator.element[2] * probability;
      partial_low[m16] = fmaf(scale, weighted_low, partial_low[m16]);
      partial_high[m16] = fmaf(scale, weighted_high, partial_high[m16]);
    }
  }
}

template <class Config>
  requires((Config::kTokens == 8 && Config::kTpSize == 8))
__device__ __forceinline__ void accumulate_down_k32_partition(
    std::uint8_t const* stage, StagedActivation const* activation, int expert_in_pair, int warp,
    int global_m16, float probability, float (&partial_low)[kDownM16PerVirtualTile],
    float (&partial_high)[kDownM16PerVirtualTile]) {
  int const thread_in_group = static_cast<int>(threadIdx.x) & 3;
  int const k32_base = warp * 32;

  Hmma::BFragment activation_fragment_low;
  Hmma::BFragment activation_fragment_high;
  load_k_permuted_activation_fragment<1>(activation, kLocalInter, k32_base,
                                         activation_fragment_low);
  load_k_permuted_activation_fragment<1>(activation, kLocalInter, k32_base + 16,
                                         activation_fragment_high);

#pragma unroll
  for (int m16 = 0; m16 < kDownM16PerVirtualTile; ++m16) {
    float const scale =
        load_down_k32_partition_scale<Config>(stage, expert_in_pair, m16, warp, global_m16);
    uint4 const packed_words =
        load_down_k32_partition_weight<Config>(stage, expert_in_pair, m16, warp);
    Hmma::Accumulator accumulator{{0.0F, 0.0F, 0.0F, 0.0F}};
    Hmma::AFragment weight_fragment;

    unpack_k_permuted_native_a_rows(make_uint2(packed_words.x, packed_words.y), weight_fragment);
    Hmma::fma(accumulator, weight_fragment, activation_fragment_low);

    unpack_k_permuted_native_a_rows(make_uint2(packed_words.z, packed_words.w), weight_fragment);
    Hmma::fma(accumulator, weight_fragment, activation_fragment_high);

    if (thread_in_group == 0) {
      float const weighted_low = accumulator.element[0] * probability;
      float const weighted_high = accumulator.element[2] * probability;
      partial_low[m16] = fmaf(scale, weighted_low, partial_low[m16]);
      partial_high[m16] = fmaf(scale, weighted_high, partial_high[m16]);
    }
  }
}

template <class Config>
  requires((Config::kTokens == 8 && Config::kTpSize == 4))
__device__ __forceinline__ void accumulate_down_k32_partition(
    std::uint8_t const* stage, StagedActivation const* activation, int expert_in_pair,
    int staged_k32_partition, int global_k32_partition, int global_m16, float probability,
    float (&partial_low)[kDownM16PerVirtualTile], float (&partial_high)[kDownM16PerVirtualTile]) {
  int const thread_in_group = static_cast<int>(threadIdx.x) & 3;
  int const k32_base = global_k32_partition * 32;

  Hmma::BFragment activation_fragment_low;
  Hmma::BFragment activation_fragment_high;
  load_k_permuted_activation_fragment<1>(activation, kLocalInter, k32_base,
                                         activation_fragment_low);
  load_k_permuted_activation_fragment<1>(activation, kLocalInter, k32_base + 16,
                                         activation_fragment_high);

#pragma unroll
  for (int m16 = 0; m16 < kDownM16PerVirtualTile; ++m16) {
    float const scale = load_down_k32_partition_scale<Config>(stage, expert_in_pair, m16,
                                                              global_k32_partition, global_m16);
    uint4 const packed_words =
        load_down_k32_partition_weight<Config>(stage, expert_in_pair, m16, staged_k32_partition);
    Hmma::Accumulator accumulator{{0.0F, 0.0F, 0.0F, 0.0F}};
    Hmma::AFragment weight_fragment;

    unpack_k_permuted_native_a_rows(make_uint2(packed_words.x, packed_words.y), weight_fragment);
    Hmma::fma(accumulator, weight_fragment, activation_fragment_low);

    unpack_k_permuted_native_a_rows(make_uint2(packed_words.z, packed_words.w), weight_fragment);
    Hmma::fma(accumulator, weight_fragment, activation_fragment_high);

    if (thread_in_group == 0) {
      float const weighted_low = accumulator.element[0] * probability;
      float const weighted_high = accumulator.element[2] * probability;
      partial_low[m16] = fmaf(scale, weighted_low, partial_low[m16]);
      partial_high[m16] = fmaf(scale, weighted_high, partial_high[m16]);
    }
  }
}

template <class Config, class Args>
  requires(Config::kTokens == 4)
__device__ void tp_allreduce_and_finalize(Args const& args, void* const* peer_workspaces, int token,
                                          int tile, std::uint8_t* dynamic_storage) {
  int const tid = static_cast<int>(threadIdx.x);
  if (tid >= kConsumerThreads) {
    return;
  }

  int const world_size = args.world_size;
  int const rank = args.rank;
  auto* local_vectors = reinterpret_cast<uint4*>(dynamic_storage + kArLocalOffset);
  auto* remote_vectors = reinterpret_cast<uint4*>(dynamic_storage + kArRemoteOffset);
  int phase;
  std::uint32_t marker;
  if (args.ar_epoch_counters != nullptr) {
    marker = reinterpret_cast<std::uint32_t const*>(remote_vectors)[0];
    phase = static_cast<int>(marker & 1U);
  } else {
    phase = static_cast<int>(args.epoch & 1U);
    marker = static_cast<std::uint32_t>(args.epoch) + 1U;
    if (marker == 0U) {
      marker = 1U;
    }
  }

  if (world_size == kFusedMoeWorldSize) {
    int const warp = tid >> 5;
    int const lane = tid & (kWarpSize - 1);
    // Transpose peer writes to keep each warp's payload stores contiguous.
    int const peer_slot = ((warp & 1) << 2) | (lane & 3);
    int const column = warp >> 1;
    int const row = lane >> 2;
    int const vector = column * 8 + row;
    int const payload_vector = column * 6 + row;
    bool const payload_lane = lane < 24;
    bool const marker_lane = lane >= 28;
    bool const selected_peer = peer_slot < kArRemoteSources;

    if (selected_peer) {
      int const destination = comm::expand_peer_slot(peer_slot, rank);
      int const source_slot = comm::compact_source_slot(rank, destination);
      auto* destination_vectors = reinterpret_cast<uint4*>(
          ar_tile_base<Config>(comm::load_shared_pointer(peer_workspaces + destination), token,
                               source_slot, phase, tile));
      uint4 const values = payload_lane ? comm::load_shared_u32x4(local_vectors + payload_vector)
                                        : make_uint4(marker, marker, marker, marker);
      comm::store_relaxed_system_u32x4(destination_vectors + vector, values);
    }
    consumer_group_sync<kConsumerThreads>();

    store_deferred_local_output<Config>(args, token, tile, local_vectors);

    int const source_slot = peer_slot;
    bool const valid_source = selected_peer;
    uint4* source_vectors = valid_source
                                ? reinterpret_cast<uint4*>(ar_tile_base<Config>(
                                      args.peer_workspaces[rank], token, source_slot, phase, tile))
                                : nullptr;
    if (valid_source && marker_lane) {
      wait_ar_marker<Config>(source_vectors + vector, marker);
    }
    __syncwarp();

    if (valid_source && payload_lane) {
      uint4 const values = comm::load_relaxed_system_u32x4(source_vectors + vector);
      comm::store_shared_u32x4(remote_vectors + source_slot * kArPayloadVectors + payload_vector,
                                 values);
    }
  }

  // One rendezvous makes both the staged peer payloads and the much earlier
  // residual cp.async copies visible to the distributed finalizer below.
  cutlass::arch::cp_async_wait<0>();
  consumer_group_sync<kConsumerThreads>();

  int const reduction_warp = tid >> 5;
  int const reduction_lane = tid & (kWarpSize - 1);
  bool const reduction_thread =
      reduction_warp < kArReductionWarps && reduction_lane < kArReductionLanesPerWarp;
  if (reduction_thread) {
    int const vector = reduction_warp * kArReductionLanesPerWarp + reduction_lane;
    std::uint32_t const local_address =
        static_cast<std::uint32_t>(__cvta_generic_to_shared(local_vectors)) +
        vector * sizeof(uint2);
    uint2 rounded;
    if (world_size == kFusedMoeWorldSize) {
      std::uint32_t const remote_address =
          static_cast<std::uint32_t>(__cvta_generic_to_shared(remote_vectors));
      float sums[kArReductionValuesPerThread] = {0.0F, 0.0F, 0.0F, 0.0F};
#pragma unroll
      for (int source = 0; source < kFusedMoeWorldSize; ++source) {
        int const source_slot = source == 0 ? 0 : source - static_cast<int>(source >= rank);
        std::uint32_t const source_address =
            source == rank ? local_address
                           : remote_address + (source_slot * kArTileElements +
                                               vector * kArReductionValuesPerThread) *
                                                  sizeof(__nv_bfloat16);
        uint2 const values = comm::load_shared_u32x2(source_address);
#pragma unroll
        for (int element = 0; element < kArReductionValuesPerThread; ++element) {
          sums[element] +=
              __bfloat162float(__ushort_as_bfloat16(comm::bf16_bits_from_u32x2(values, element)));
        }
      }
      // Preserve the second BF16 rounding boundary before residual addition.
      rounded = pack_bf16x4<Config>(sums);
    } else {
      rounded = comm::load_shared_u32x2(local_address);
    }

    std::uint32_t const residual_address =
        static_cast<std::uint32_t>(__cvta_generic_to_shared(dynamic_storage + kArResidualOffset)) +
        vector * sizeof(uint2);
    uint2 const residual_values = comm::load_shared_u32x2(residual_address);
    uint2 const finalized = make_uint2(finalize_bf16x2<Config>(rounded.x, residual_values.x),
                                       finalize_bf16x2<Config>(rounded.y, residual_values.y));
    auto* final_output_vectors = reinterpret_cast<uint2*>(
        static_cast<__nv_bfloat16*>(args.final_output) + static_cast<std::size_t>(token) * kHidden +
        static_cast<std::size_t>(tile) * kArTileElements);
    final_output_vectors[vector] = finalized;
  }
}

template <class Config, int kTmaProducer, int kExpertPair>
  requires((Config::kTokens == 8 && Config::kTpSize == 8))
__device__ __forceinline__ void produce_down_expert_pair_virtual_tile(
    Glm5MegaMoeArgs const& args, PipelineControl* control, std::uint8_t* dynamic_storage,
    std::int32_t const* local_selected_indices, int global_m16, int task) {
  static_assert(kTmaProducer >= 0 && kTmaProducer < kDownTmaIssueWarps);
  static_assert(kExpertPair >= 0);
  static_assert(kExpertPair < kDownExpertPairs);
  constexpr int kExpertsInPair = kExpertPair + 1 == kDownExpertPairs ? 1 : 2;
  constexpr int kFirstSlot = kExpertPair * 2;
  constexpr int kTransactionBytes =
      down_producer_transaction_bytes<Config, kTmaProducer, kExpertsInPair>();
  constexpr int kArriveCount = kTmaProducer == 0 ? kProducerWarps - kDownTmaIssueWarps + 1 : 1;

  int const virtual_tile = global_m16 / kDownM16PerVirtualTile;
  int const first_scale_block = (virtual_tile * 48) >> 7;
  int const stored_scale_block = first_scale_block & ~1;
  int const stage_index = task & (kPipelineStages - 1);
  unsigned int const cycle = static_cast<unsigned int>(task / kPipelineStages);
  if (cycle != 0) {
    unsigned int const empty_phase = (cycle - 1U) & 1U;
    cluster_barrier_wait(control->empty + stage_index, empty_phase);
  }
  if constexpr (kTransactionBytes != 0) {
    producer_arrive_and_expect_tx_count(control->ready + stage_index, kArriveCount,
                                        kTransactionBytes);
  } else {
    static_assert(kArriveCount == 1);
    cutlass::arch::ClusterBarrier::arrive(control->ready + stage_index);
  }

  std::uint8_t* stage = pipeline_stage<PipelineControl>(dynamic_storage, task);
  std::uint64_t* ready = control->ready + stage_index;

  int const first_expert =
      physical_expert_for_down_slot<Config, kFirstSlot>(local_selected_indices);
  issue_down_expert_tma<Config, kTmaProducer, 0>(args, first_expert, virtual_tile,
                                                 stored_scale_block, ready, stage);

  if constexpr (kExpertsInPair == 2) {
    constexpr int kSecondSlot = kFirstSlot + 1;
    int const second_expert =
        physical_expert_for_down_slot<Config, kSecondSlot>(local_selected_indices);
    std::uint8_t* second_stage = stage + kDownExpertStageStride;
    issue_down_expert_tma<Config, kTmaProducer, 1>(args, second_expert, virtual_tile,
                                                   stored_scale_block, ready, second_stage);
  }
}

template <class Config, int kTmaProducer, int kExpertPair>
  requires((Config::kTokens == 8 && Config::kTpSize == 4))
__device__ __forceinline__ void produce_down_expert_pair_virtual_tile(
    Glm5MegaMoeArgs const& args, PipelineControl* control, std::uint8_t* dynamic_storage,
    std::int32_t const* local_selected_indices, int global_m16, int down_k_task, int task) {
  static_assert(kTmaProducer >= 0 && kTmaProducer < kDownTmaIssueWarps);
  static_assert(kExpertPair >= 0);
  static_assert(kExpertPair < kDownExpertPairs);
  constexpr int kExpertsInPair = kExpertPair + 1 == kDownExpertPairs ? 1 : 2;
  constexpr int kFirstSlot = kExpertPair * 2;
  constexpr int kTransactionBytes =
      down_producer_transaction_bytes<Config, kTmaProducer, kExpertsInPair>();
  constexpr int kArriveCount = kTmaProducer == 0 ? kProducerWarps - kDownTmaIssueWarps + 1 : 1;

  int const virtual_tile = global_m16 / kDownM16PerVirtualTile;
  int const first_expert =
      physical_expert_for_down_slot<Config, kFirstSlot>(local_selected_indices);
  int const first_weight_row = first_expert * kHidden + global_m16 * kDownM16Rows;
  int const first_scale_block = (virtual_tile * 48) >> 7;
  int const stored_scale_block = first_scale_block & ~1;
  int const stage_index = task & (kPipelineStages - 1);
  unsigned int const cycle = static_cast<unsigned int>(task / kPipelineStages);
  if (cycle != 0) {
    unsigned int const empty_phase = (cycle - 1U) & 1U;
    cluster_barrier_wait(control->empty + stage_index, empty_phase);
  }
  if constexpr (kTransactionBytes != 0) {
    producer_arrive_and_expect_tx_count(control->ready + stage_index, kArriveCount,
                                        kTransactionBytes);
  } else {
    static_assert(kArriveCount == 1);
    cutlass::arch::ClusterBarrier::arrive(control->ready + stage_index);
  }

  std::uint8_t* stage = pipeline_stage<PipelineControl>(dynamic_storage, task);
  std::uint64_t* ready = control->ready + stage_index;
  constexpr std::uint64_t kCachePolicy =
      static_cast<std::uint64_t>(cute::TMA::CacheHintSm100::EVICT_FIRST);
  if constexpr (kTmaProducer == 0) {
    tensor_copy_3d_g2s(&args.down_weight_tma, 0, down_k_task * kDownStagedTmaKAtoms,
                       first_weight_row, ready, stage, kCachePolicy);
  } else if constexpr (kTmaProducer == 1 && kExpertsInPair == 2) {
    constexpr int kSecondSlot = kFirstSlot + 1;
    int const second_expert =
        physical_expert_for_down_slot<Config, kSecondSlot>(local_selected_indices);
    int const second_weight_row = second_expert * kHidden + global_m16 * kDownM16Rows;
    tensor_copy_3d_g2s(&args.down_weight_tma, 0, down_k_task * kDownStagedTmaKAtoms,
                       second_weight_row, ready, stage + kDownExpertStageStride, kCachePolicy);
  } else if constexpr (kTmaProducer == 2) {
    bulk_copy_g2s(
        args.down_scale +
            (static_cast<std::size_t>(first_expert) * 48 + stored_scale_block) * kDownScaleBlocks,
        ready, stage + kDownStageScaleOffset, kDownNativeScaleTransactionBytes);
    if constexpr (kExpertsInPair == 2) {
      constexpr int kSecondSlot = kFirstSlot + 1;
      int const second_expert =
          physical_expert_for_down_slot<Config, kSecondSlot>(local_selected_indices);
      bulk_copy_g2s(
          args.down_scale + (static_cast<std::size_t>(second_expert) * 48 + stored_scale_block) *
                                kDownScaleBlocks,
          ready, stage + kDownExpertStageStride + kDownStageScaleOffset,
          kDownNativeScaleTransactionBytes);
    }
  }
}

template <class Config, class Layout, int kValidColumns>
  requires(Config::kTokens == 4)
__device__ __forceinline__ void accumulate_gate_up_page(Layout layout, std::uint8_t const* stage,
                                                        StagedActivation const* activation,
                                                        int activation_stride, int matrix,
                                                        int k_partition, int page,
                                                        Hmma::Accumulator& accumulator) {
  int const partition_base = k_partition * 256;
#pragma unroll
  for (int block_delta = 0; block_delta < 256; block_delta += 128) {
    int const block_base = partition_base + block_delta;
    float const scale = load_gate_up_stage_scale<Config>(layout, stage, matrix, block_base);
    Hmma::Accumulator block_accumulator{{0.0F, 0.0F, 0.0F, 0.0F}};

#pragma unroll
    for (int k_delta = 0; k_delta < 128; k_delta += 16) {
      int const k_in_page = block_base + k_delta;
      Hmma::AFragment weight_fragment;
      Hmma::BFragment activation_fragment;
      load_gate_up_stage_weight_fragment<Config>(layout, stage, matrix, k_in_page, weight_fragment);
      load_layout_activation_fragment<Config, kValidColumns>(
          layout, activation, activation_stride, page * 1024 + k_in_page, activation_fragment);
      Hmma::fma(block_accumulator, weight_fragment, activation_fragment);
    }

#pragma unroll
    for (int element = 0; element < 4; ++element) {
      accumulator.element[element] =
          fmaf(scale, block_accumulator.element[element], accumulator.element[element]);
    }
  }
}

template <class Config, class Layout, int kValidColumns>
  requires(Config::kTokens == 4)
__device__ __forceinline__ void accumulate_gate_up_page_dual(
    Layout layout, std::uint8_t const* stage, StagedActivation const* activation,
    int activation_stride, int k_partition, int page, Hmma::Accumulator& gate_accumulator,
    Hmma::Accumulator& up_accumulator) {
  int const block_base = k_partition * 128;
  float const gate_scale = load_gate_up_stage_scale<Config>(layout, stage, 0, block_base);
  float const up_scale = load_gate_up_stage_scale<Config>(layout, stage, 1, block_base);
  Hmma::Accumulator gate_block_accumulator{{0.0F, 0.0F, 0.0F, 0.0F}};
  Hmma::Accumulator up_block_accumulator{{0.0F, 0.0F, 0.0F, 0.0F}};

#pragma unroll
  for (int k_delta = 0; k_delta < 128; k_delta += 16) {
    int const k_in_page = block_base + k_delta;
    Hmma::BFragment activation_fragment;
    load_layout_activation_fragment<Config, kValidColumns>(
        layout, activation, activation_stride, page * 1024 + k_in_page, activation_fragment);
    Hmma::AFragment weight_fragment;
    load_gate_up_stage_weight_fragment<Config>(layout, stage, 0, k_in_page, weight_fragment);
    Hmma::fma(gate_block_accumulator, weight_fragment, activation_fragment);
    load_gate_up_stage_weight_fragment<Config>(layout, stage, 1, k_in_page, weight_fragment);
    Hmma::fma(up_block_accumulator, weight_fragment, activation_fragment);
  }

#pragma unroll
  for (int element = 0; element < 4; ++element) {
    gate_accumulator.element[element] = fmaf(gate_scale, gate_block_accumulator.element[element],
                                             gate_accumulator.element[element]);
    up_accumulator.element[element] =
        fmaf(up_scale, up_block_accumulator.element[element], up_accumulator.element[element]);
  }
}

template <class Config, int kValidColumns>
  requires(Config::kTokens == 8)
__device__ __forceinline__ void accumulate_gate_up_page_dual(std::uint8_t const* stage,
                                                             StagedActivation const* activation,
                                                             int activation_stride, int k_partition,
                                                             int page,
                                                             Hmma::Accumulator& gate_accumulator,
                                                             Hmma::Accumulator& up_accumulator) {
  int const block_base = k_partition * 128;
  float const gate_scale = load_gate_up_stage_scale<Config>(stage, 0, block_base);
  float const up_scale = load_gate_up_stage_scale<Config>(stage, 1, block_base);
  Hmma::Accumulator gate_block_accumulator{{0.0F, 0.0F, 0.0F, 0.0F}};
  Hmma::Accumulator up_block_accumulator{{0.0F, 0.0F, 0.0F, 0.0F}};

#pragma unroll
  for (int k_delta = 0; k_delta < 128; k_delta += 16) {
    int const k_in_page = block_base + k_delta;
    Hmma::AFragment weight_fragment;
    Hmma::BFragment activation_fragment;
    load_k_permuted_activation_fragment<kValidColumns>(
        activation, activation_stride, page * 1024 + k_in_page, activation_fragment);
    load_gate_up_stage_weight_fragment<Config>(stage, 0, k_in_page, weight_fragment);
    Hmma::fma(gate_block_accumulator, weight_fragment, activation_fragment);
    load_gate_up_stage_weight_fragment<Config>(stage, 1, k_in_page, weight_fragment);
    Hmma::fma(up_block_accumulator, weight_fragment, activation_fragment);
  }

#pragma unroll
  for (int element = 0; element < 4; ++element) {
    gate_accumulator.element[element] = fmaf(gate_scale, gate_block_accumulator.element[element],
                                             gate_accumulator.element[element]);
    up_accumulator.element[element] =
        fmaf(up_scale, up_block_accumulator.element[element], up_accumulator.element[element]);
  }
}

template <class Config, int kDownProducer, int kExpertPair, class Layout>
  requires(Config::kTokens == 4)
__device__ __forceinline__ void produce_down_expert_pair_k_tasks(
    Layout layout, typename Layout::Args const& args, PipelineControl* control,
    std::uint8_t* dynamic_storage, std::int32_t const* local_selected_indices, int tile,
    int virtual_tile) {
#pragma unroll
  for (int down_k_task = 0; down_k_task < kDownKTasks; ++down_k_task) {
    int const local_task =
        (virtual_tile * kDownExpertPairs + kExpertPair) * kDownKTasks + down_k_task;
    produce_down_expert_pair<Config, kDownProducer, kExpertPair>(
        layout, args, control, dynamic_storage, local_selected_indices, tile, virtual_tile,
        down_k_task, pipeline_task_index(local_task, kRoutedDownPipelineTaskBase));
  }
}

template <class Config, int kTmaProducer, int kExpertPair = 0>
  requires((Config::kTokens == 8 && Config::kTpSize == 8))
__device__ __forceinline__ void produce_down_expert_pairs(
    Glm5MegaMoeArgs const& args, PipelineControl* control, std::uint8_t* dynamic_storage,
    std::int32_t const* local_selected_indices, int global_m16, int task_base) {
  produce_down_expert_pair_virtual_tile<Config, kTmaProducer, kExpertPair>(
      args, control, dynamic_storage, local_selected_indices, global_m16, task_base + kExpertPair);
  if constexpr (kExpertPair + 1 < kDownExpertPairs) {
    produce_down_expert_pairs<Config, kTmaProducer, kExpertPair + 1>(
        args, control, dynamic_storage, local_selected_indices, global_m16, task_base);
  }
}

template <class Config, int kTmaProducer, int kExpertPair = 0>
  requires((Config::kTokens == 8 && Config::kTpSize == 4))
__device__ __forceinline__ void produce_down_expert_pairs(
    Glm5MegaMoeArgs const& args, PipelineControl* control, std::uint8_t* dynamic_storage,
    std::int32_t const* local_selected_indices, int global_m16, int task_base) {
#pragma unroll
  for (int down_k_task = 0; down_k_task < kDownKTasks; ++down_k_task) {
    int const task = task_base + kExpertPair * kDownKTasks + down_k_task;
    produce_down_expert_pair_virtual_tile<Config, kTmaProducer, kExpertPair>(
        args, control, dynamic_storage, local_selected_indices, global_m16, down_k_task, task);
  }
  if constexpr (kExpertPair + 1 < kDownExpertPairs) {
    produce_down_expert_pairs<Config, kTmaProducer, kExpertPair + 1>(
        args, control, dynamic_storage, local_selected_indices, global_m16, task_base);
  }
}

template <class Config, class Layout>
  requires(Config::kTokens == 4)
__device__ void routed_gate_up(Layout layout, typename Layout::Args const& args,
                               __nv_bfloat16 const* normalized, int token, int tile,
                               std::int32_t const* local_selected_indices,
                               std::uint64_t const* gate_up_cache_policy,
                               std::uint64_t const* topk_handoff, __nv_bfloat16* up_gate,
                               float* gate_up_scratch, PipelineControl* control,
                               std::uint8_t* dynamic_storage) {
  int const tid = static_cast<int>(threadIdx.x);
  int const selected_slot = tile >> 2;
  int const quarter = tile & 3;

  // Snapshot the next device epoch before the existing whole-CTA GateUp
  // rendezvous.  Routed Down consumers publish this epoch only after their
  // complete per-token activation slice has reached shared memory.
  if (tid == 0 && args.ar_epoch_counters != nullptr) {
    int const counter = token * kArTilesPerToken + tile;
    auto* generation_scratch = reinterpret_cast<std::uint32_t*>(dynamic_storage + kArRemoteOffset);
    generation_scratch[0] =
        next_generation<Config>(load_relaxed_gpu_u32<Config>(args.ar_epoch_counters + counter));
  }

  if (is_gate_up_producer<Config>(layout, tid)) {
    int const producer_warp = gate_up_producer_warp<Config>(layout, tid);
    cluster_barrier_wait(topk_handoff, 0);
    int const expert =
        physical_routed_expert<Config>(layout, local_selected_indices[selected_slot]);
    std::uint64_t cache_policy = 0;
    if constexpr (Config::kTpSize == 4) {
      cache_policy = *gate_up_cache_policy;
    }
#pragma unroll 1
    for (int local_task = 0; local_task < kRoutedGateUpPipelineTasks; ++local_task) {
      int const local_row_group = local_task / 6;
      int const page = local_task % 6;
      int const row_group = quarter * kRoutedGateUpRowGroupsPerCta + local_row_group;
      int const task = pipeline_task_index(local_task, kRoutedGateUpPipelineTaskBase);
      if constexpr (Config::kTpSize == 4) {
        produce_gate_up_page<Config>(layout, args, control, dynamic_storage, expert, row_group, page,
                                     task, cache_policy, producer_warp);
      } else {
        produce_gate_up_page<Config>(layout, args, control, dynamic_storage, expert, row_group, page,
                                     task, selected_slot, producer_warp);
      }
    }
  } else if (tid < kConsumerThreads) {
    StagedActivation const* mma_normalized =
        stage_bf16_for_mma<Config, kHidden>(normalized, dynamic_storage + kSharedActivationOffset);
    int const warp = tid >> 5;
    int const lane = tid & 31;
    int const k_partition = warp;
    auto* partial = reinterpret_cast<float*>(dynamic_storage + kAuxOffset);

#pragma unroll 1
    for (int local_row_group = 0; local_row_group < kRoutedGateUpRowGroupsPerCta;
         ++local_row_group) {
      Hmma::Accumulator gate_accumulator{{0.0F, 0.0F, 0.0F, 0.0F}};
      Hmma::Accumulator up_accumulator{{0.0F, 0.0F, 0.0F, 0.0F}};
#pragma unroll 1
      for (int page = 0; page < 6; ++page) {
        int const local_task = local_row_group * 6 + page;
        int const task = pipeline_task_index(local_task, kRoutedGateUpPipelineTaskBase);
        consumer_wait_stage(control, task);
        accumulate_gate_up_page_dual<Config, Layout, 1>(
            layout, pipeline_stage<PipelineControl>(dynamic_storage, task), mma_normalized, kHidden,
            k_partition, page, gate_accumulator, up_accumulator);
        consumer_release_stage(control, task);
      }

      if ((lane & 3) == 0) {
        int const row = lane >> 2;
        int const partial_base = k_partition * 16;
        partial[partial_base + row] = gate_accumulator.element[0];
        partial[partial_base + row + 8] = gate_accumulator.element[2];
        partial[128 + partial_base + row] = up_accumulator.element[0];
        partial[128 + partial_base + row + 8] = up_accumulator.element[2];
      }
      consumer_group_sync<kConsumerThreads>();

      if (tid < 32) {
        int const output_matrix = tid >> 4;
        int const row = tid & 15;
        float value = 0.0F;
#pragma unroll
        for (int partition = 0; partition < 8; ++partition) {
          value += partial[output_matrix * 128 + partition * 16 + row];
        }
        gate_up_scratch[output_matrix * 64 + row] = value;
      }
      consumer_group_sync<kConsumerThreads>();

      if (tid < 16) {
        int const output_row = quarter * kRoutedGateUpRowsPerCta + local_row_group * 16 + tid;
        float const gate = gate_up_scratch[tid];
        float const up = gate_up_scratch[64 + tid];
        float const activated = gate_silu_multiply(gate, up);
        __nv_bfloat16 const activated_bf16 = float_to_bf16(activated);
        std::size_t const output_index =
            (static_cast<std::size_t>(token) * 9 + selected_slot + 1) * kLocalInter + output_row;
        up_gate[output_index] = activated_bf16;
        if (local_row_group == kRoutedGateUpRowGroupsPerCta - 1 && tid == 15) {
          auto* sentinel_scratch = reinterpret_cast<std::uint16_t*>(gate_up_scratch);
          sentinel_scratch[0] = __bfloat16_as_ushort(activated_bf16);
        }
      }
      consumer_group_sync<kConsumerThreads>();
    }
    if (tid == 0) {
      std::size_t const sentinel_index =
          (static_cast<std::size_t>(token) * 9 + selected_slot + 1) * kLocalInter +
          quarter * kRoutedGateUpRowsPerCta + kRoutedGateUpRowsPerCta - 1;
      publish_up_gate_sentinel<Config>(up_gate + sentinel_index,
                                       reinterpret_cast<std::uint16_t*>(gate_up_scratch)[0]);
    }
  }
  gate_up_phase_sync<kThreads>();
}

template <class Config>
  requires((Config::kTokens == 8 && Config::kTpSize == 8))
__device__ void routed_gate_up(Glm5MegaMoeArgs const& args, __nv_bfloat16 const* normalized,
                               int token, int flat_base, int group_count, int gate_task_base,
                               std::int32_t const* local_selected_indices,
                               std::uint64_t const* topk_handoff, __nv_bfloat16* up_gate,
                               float* gate_up_scratch, PipelineControl* control,
                               std::uint8_t* dynamic_storage) {
  int const tid = static_cast<int>(threadIdx.x);

  if (tid >= kConsumerThreads && (tid & (kWarpSize - 1)) == 0) {
    int const producer_warp = (tid - kConsumerThreads) / kWarpSize;
    int expert = 0;
    int current_slot = -1;
    std::uint32_t handoff_phase = 0;
#pragma unroll 1
    for (int local_group = 0; local_group < group_count; ++local_group) {
      int const flat = flat_base + local_group;
      int const selected_slot = flat >> 4;
      int const row_group = flat & 15;
      if (selected_slot != current_slot) {
        cluster_barrier_wait(topk_handoff, handoff_phase);
        expert = local_selected_indices[selected_slot];
        current_slot = selected_slot;
        handoff_phase ^= 1U;
      }
#pragma unroll
      for (int page = 0; page < kGateUpPagesPerGroup; ++page) {
        int const local_task = local_group * kGateUpPagesPerGroup + page;
        int const task = pipeline_task_index(local_task, gate_task_base);
        produce_gate_up_page<Config>(args, control, dynamic_storage, expert, row_group, page, task,
                                     selected_slot, producer_warp);
      }
    }
  } else if (tid < kConsumerThreads) {
    StagedActivation const* mma_normalized =
        stage_bf16_for_mma<Config, kHidden>(normalized, dynamic_storage + kRoutedActivationOffset);
    int const warp = tid >> 5;
    int const lane = tid & 31;
    int const k_partition = warp;
    auto* partial = reinterpret_cast<float*>(dynamic_storage + kAuxOffset);

#pragma unroll 1
    for (int local_group = 0; local_group < group_count; ++local_group) {
      int const flat = flat_base + local_group;
      int const selected_slot = flat >> 4;
      int const row_group = flat & 15;
      Hmma::Accumulator gate_accumulator{{0.0F, 0.0F, 0.0F, 0.0F}};
      Hmma::Accumulator up_accumulator{{0.0F, 0.0F, 0.0F, 0.0F}};
#pragma unroll 1
      for (int page = 0; page < kGateUpPagesPerGroup; ++page) {
        int const local_task = local_group * kGateUpPagesPerGroup + page;
        int const task = pipeline_task_index(local_task, gate_task_base);
        consumer_wait_stage(control, task);
        accumulate_gate_up_page_dual<Config, 1>(
            pipeline_stage<PipelineControl>(dynamic_storage, task), mma_normalized, kHidden,
            k_partition, page, gate_accumulator, up_accumulator);
        consumer_release_stage(control, task);
      }

      if ((lane & 3) == 0) {
        int const row = lane >> 2;
        int const partial_base = k_partition * 16;
        partial[partial_base + row] = gate_accumulator.element[0];
        partial[partial_base + row + 8] = gate_accumulator.element[2];
        partial[128 + partial_base + row] = up_accumulator.element[0];
        partial[128 + partial_base + row + 8] = up_accumulator.element[2];
      }
      consumer_group_sync<kConsumerThreads>();

      if (tid < 32) {
        int const output_matrix = tid >> 4;
        int const row = tid & 15;
        float value = 0.0F;
#pragma unroll
        for (int partition = 0; partition < 8; ++partition) {
          value += partial[output_matrix * 128 + partition * 16 + row];
        }
        gate_up_scratch[output_matrix * 16 + row] = value;
      }
      consumer_group_sync<kConsumerThreads>();

      if (tid < 16) {
        int const output_row = row_group * 16 + tid;
        float const gate = gate_up_scratch[tid];
        float const up = gate_up_scratch[16 + tid];
        float const activated = gate_silu_multiply(gate, up);
        __nv_bfloat16 const activated_bf16 = float_to_bf16(activated);
        std::size_t const output_index =
            (static_cast<std::size_t>(token) * 9 + selected_slot + 1) * kLocalInter + output_row;
        bool const is_sentinel = local_group + 1 == group_count && tid == 15;
        if (is_sentinel) {
          auto* sentinel_scratch = reinterpret_cast<std::uint16_t*>(gate_up_scratch);
          sentinel_scratch[0] = __bfloat16_as_ushort(activated_bf16);
        } else {
          store_up_gate_value<Config>(up_gate + output_index, activated_bf16);
        }
      }
      consumer_group_sync<kConsumerThreads>();
    }
    if (tid == 0) {
      int const last_flat = flat_base + group_count - 1;
      int const selected_slot = last_flat >> 4;
      int const row_group = last_flat & 15;
      std::size_t const sentinel_index =
          (static_cast<std::size_t>(token) * 9 + selected_slot + 1) * kLocalInter + row_group * 16 +
          15;
      publish_up_gate_sentinel<Config>(up_gate + sentinel_index,
                                       reinterpret_cast<std::uint16_t*>(gate_up_scratch)[0]);
    }
  }
  gate_up_phase_sync<kThreads>();
}

template <class Config>
  requires((Config::kTokens == 8 && Config::kTpSize == 4))
__device__ void routed_gate_up(Glm5MegaMoeArgs const& args, __nv_bfloat16 const* normalized,
                               int token, int flat_base, int group_count, int gate_task_base,
                               std::int32_t const* local_selected_indices,
                               std::uint64_t const* topk_handoff, __nv_bfloat16* up_gate,
                               float* gate_up_scratch, PipelineControl* control,
                               std::uint8_t* dynamic_storage) {
  int const tid = static_cast<int>(threadIdx.x);

  if (tid >= kConsumerThreads && (tid & (kWarpSize - 1)) == 0) {
    int const producer_warp = (tid - kConsumerThreads) / kWarpSize;
    int expert = 0;
    int current_slot = -1;
    std::uint64_t cache_policy = 0;
    std::uint32_t handoff_phase = 0;
#pragma unroll 1
    for (int local_pair = 0; local_pair < group_count / 2; ++local_pair) {
      int const first_local_group = local_pair * 2;
      int const flat = flat_base + first_local_group;
      int const selected_slot = flat / kGateUpRowGroupsPerExpert;
      int const first_row_group = flat % kGateUpRowGroupsPerExpert;
      if (selected_slot != current_slot) {
        cluster_barrier_wait(topk_handoff, handoff_phase);
        expert = local_selected_indices[selected_slot];
        cache_policy = static_cast<std::uint64_t>((expert & kGateUpL2RetainedExpertMask) == 0
                                                      ? cute::TMA::CacheHintSm100::EVICT_LAST
                                                      : cute::TMA::CacheHintSm100::EVICT_FIRST);
        current_slot = selected_slot;
        handoff_phase ^= 1U;
      }
#pragma unroll
      for (int page = 0; page < kGateUpPagesPerGroup; ++page) {
#pragma unroll
        for (int group_in_pair = 0; group_in_pair < 2; ++group_in_pair) {
          int const local_task = (local_pair * kGateUpPagesPerGroup + page) * 2 + group_in_pair;
          int const task = pipeline_task_index(local_task, gate_task_base);
          produce_gate_up_page<Config>(args, control, dynamic_storage, expert,
                                       first_row_group + group_in_pair, page, task, cache_policy,
                                       producer_warp);
        }
      }
    }
  } else if (tid < kConsumerThreads) {
    StagedActivation const* mma_normalized =
        stage_bf16_for_mma<Config, kHidden>(normalized, dynamic_storage + kRoutedActivationOffset);
    int const warp = tid >> 5;
    int const lane = tid & 31;
    int const k_partition = warp;
    auto* partial = reinterpret_cast<float*>(dynamic_storage + kAuxOffset);

#pragma unroll 1
    for (int local_pair = 0; local_pair < group_count / 2; ++local_pair) {
      int const first_local_group = local_pair * 2;
      int const flat = flat_base + first_local_group;
      int const selected_slot = flat / kGateUpRowGroupsPerExpert;
      int const first_row_group = flat % kGateUpRowGroupsPerExpert;
      Hmma::Accumulator gate_accumulator0{{0.0F, 0.0F, 0.0F, 0.0F}};
      Hmma::Accumulator up_accumulator0{{0.0F, 0.0F, 0.0F, 0.0F}};
      Hmma::Accumulator gate_accumulator1{{0.0F, 0.0F, 0.0F, 0.0F}};
      Hmma::Accumulator up_accumulator1{{0.0F, 0.0F, 0.0F, 0.0F}};
#pragma unroll 1
      for (int page = 0; page < kGateUpPagesPerGroup; ++page) {
#pragma unroll
        for (int group_in_pair = 0; group_in_pair < 2; ++group_in_pair) {
          int const local_task = (local_pair * kGateUpPagesPerGroup + page) * 2 + group_in_pair;
          int const task = pipeline_task_index(local_task, gate_task_base);
          consumer_wait_stage(control, task);
          if (group_in_pair == 0) {
            accumulate_gate_up_page_dual<Config, 1>(
                pipeline_stage<PipelineControl>(dynamic_storage, task), mma_normalized, kHidden,
                k_partition, page, gate_accumulator0, up_accumulator0);
          } else {
            accumulate_gate_up_page_dual<Config, 1>(
                pipeline_stage<PipelineControl>(dynamic_storage, task), mma_normalized, kHidden,
                k_partition, page, gate_accumulator1, up_accumulator1);
          }
          consumer_release_stage(control, task);
        }
      }

      if ((lane & 3) == 0) {
        int const row = lane >> 2;
        int const partial_base = k_partition * 16;
        partial[partial_base + row] = gate_accumulator0.element[0];
        partial[partial_base + row + 8] = gate_accumulator0.element[2];
        partial[128 + partial_base + row] = up_accumulator0.element[0];
        partial[128 + partial_base + row + 8] = up_accumulator0.element[2];
        partial[256 + partial_base + row] = gate_accumulator1.element[0];
        partial[256 + partial_base + row + 8] = gate_accumulator1.element[2];
        partial[384 + partial_base + row] = up_accumulator1.element[0];
        partial[384 + partial_base + row + 8] = up_accumulator1.element[2];
      }
      consumer_group_sync<kConsumerThreads>();

      if (tid < 64) {
        int const group_in_pair = tid >> 5;
        int const item = tid & 31;
        int const output_matrix = item >> 4;
        int const row = item & 15;
        float value = 0.0F;
#pragma unroll
        for (int partition = 0; partition < 8; ++partition) {
          value += partial[group_in_pair * 256 + output_matrix * 128 + partition * 16 + row];
        }
        gate_up_scratch[group_in_pair * 32 + output_matrix * 16 + row] = value;
      }
      consumer_group_sync<kConsumerThreads>();

      if (tid < 32) {
        int const group_in_pair = tid >> 4;
        int const row = tid & 15;
        int const output_row = (first_row_group + group_in_pair) * 16 + row;
        float const gate = gate_up_scratch[group_in_pair * 32 + row];
        float const up = gate_up_scratch[group_in_pair * 32 + 16 + row];
        float const activated = gate_silu_multiply(gate, up);
        __nv_bfloat16 const activated_bf16 = float_to_bf16(activated);
        std::size_t const output_index =
            (static_cast<std::size_t>(token) * 9 + selected_slot + 1) * kLocalInter + output_row;
        bool const is_sentinel = first_local_group + group_in_pair + 1 == group_count && row == 15;
        if (is_sentinel) {
          auto* sentinel_scratch = reinterpret_cast<std::uint16_t*>(gate_up_scratch);
          sentinel_scratch[0] = __bfloat16_as_ushort(activated_bf16);
        } else {
          store_up_gate_value<Config>(up_gate + output_index, activated_bf16);
        }
      }
      consumer_group_sync<kConsumerThreads>();
    }
    if (tid == 0) {
      int const last_flat = flat_base + group_count - 1;
      int const selected_slot = last_flat / kGateUpRowGroupsPerExpert;
      int const row_group = last_flat % kGateUpRowGroupsPerExpert;
      std::size_t const sentinel_index =
          (static_cast<std::size_t>(token) * 9 + selected_slot + 1) * kLocalInter + row_group * 16 +
          15;
      publish_up_gate_sentinel<Config>(up_gate + sentinel_index,
                                       reinterpret_cast<std::uint16_t*>(gate_up_scratch)[0]);
    }
  }
  gate_up_phase_sync<kThreads>();
}

template <class Config, class Layout>
  requires(Config::kTokens == 4)
__device__ void shared_gate_up(Layout layout, typename Layout::Args const& args,
                               __nv_bfloat16 const* shared_normalized,
                               std::uint32_t const* normalized_ready, int row_group,
                               __nv_bfloat16* up_gate, float* gate_up_scratch,
                               PipelineControl* control, std::uint8_t* dynamic_storage) {
  int const tid = static_cast<int>(threadIdx.x);
  int const pipeline_task_base = row_group < kTokens ? kSharedOwnerGateUpPipelineTaskBase
                                                     : kSharedFollowerGateUpPipelineTaskBase;

  if (is_gate_up_producer<Config>(layout, tid)) {
    int const producer_warp = gate_up_producer_warp<Config>(layout, tid);
#pragma unroll 1
    for (int local_task = 0; local_task < kSharedGateUpPipelineTasks; ++local_task) {
      int const task = pipeline_task_index(local_task, pipeline_task_base);
      if constexpr (Config::kTpSize == 4) {
        produce_gate_up_page<Config>(
            layout, args, control, dynamic_storage, physical_shared_expert<Config>(layout), row_group,
            local_task, task, static_cast<std::uint64_t>(cute::TMA::CacheHintSm100::EVICT_FIRST),
            producer_warp);
      } else {
        produce_gate_up_page<Config>(layout, args, control, dynamic_storage,
                                     physical_shared_expert<Config>(layout), row_group, local_task,
                                     task, -1, producer_warp);
      }
    }
  } else if (tid < kConsumerThreads) {
    wait_normalized_tokens<Config>(normalized_ready);
    auto const* normalized_source = reinterpret_cast<std::uint8_t const*>(shared_normalized);
    auto* normalized_stage = dynamic_storage + kSharedActivationOffset;
#pragma unroll 1
    for (int vector = tid; vector < kSharedActivationCopyVectors; vector += kConsumerThreads) {
      int const byte_offset = vector * kRmsInputCopyBytes;
      copy_16b_g2s(normalized_source + byte_offset, normalized_stage + byte_offset);
    }
    cutlass::arch::cp_async_fence();
    cutlass::arch::cp_async_wait<0>();
    consumer_group_sync<kConsumerThreads>();
    StagedActivation const* mma_normalized = stage_bf16_for_mma<Config, kTokens * kHidden>(
        reinterpret_cast<__nv_bfloat16 const*>(normalized_stage), normalized_stage);
    int const warp = tid >> 5;
    int const lane = tid & 31;
    int const matrix = warp >> 2;
    int const k_partition = warp & 3;
    auto* partial = reinterpret_cast<float*>(dynamic_storage + kAuxOffset);
    Hmma::Accumulator accumulator{{0.0F, 0.0F, 0.0F, 0.0F}};

#pragma unroll 1
    for (int page = 0; page < 6; ++page) {
      int const task = pipeline_task_index(page, pipeline_task_base);
      consumer_wait_stage(control, task);
      accumulate_gate_up_page<Config, Layout, 4>(
          layout, pipeline_stage<PipelineControl>(dynamic_storage, task), mma_normalized, kHidden,
          matrix, k_partition, page, accumulator);
      consumer_release_stage(control, task);
    }

    int const thread_in_group = lane & 3;
    if (thread_in_group < 2) {
      int const row = lane >> 2;
      int const token0 = thread_in_group * 2;
      int const token1 = token0 + 1;
      int const partial_base = matrix * 256 + k_partition * 64;
      partial[partial_base + token0 * 16 + row] = accumulator.element[0];
      partial[partial_base + token1 * 16 + row] = accumulator.element[1];
      partial[partial_base + token0 * 16 + row + 8] = accumulator.element[2];
      partial[partial_base + token1 * 16 + row + 8] = accumulator.element[3];
    }
    consumer_group_sync<kConsumerThreads>();

    if (tid < 128) {
      int const output_matrix = tid >> 6;
      int const item = tid & 63;
      float value = 0.0F;
#pragma unroll
      for (int partition = 0; partition < 4; ++partition) {
        value += partial[output_matrix * 256 + partition * 64 + item];
      }
      gate_up_scratch[output_matrix * 64 + item] = value;
    }
    consumer_group_sync<kConsumerThreads>();

    if (tid < 64) {
      int const token = tid >> 4;
      int const row = tid & 15;
      int const output_row = row_group * 16 + row;
      float const gate = gate_up_scratch[tid];
      float const up = gate_up_scratch[64 + tid];
      float const activated = gate_silu_multiply(gate, up);
      __nv_bfloat16 const activated_bf16 = float_to_bf16(activated);
      std::size_t const output_index =
          (static_cast<std::size_t>(token) * 9) * kLocalInter + output_row;
      up_gate[output_index] = activated_bf16;
      if (row == 15) {
        auto* sentinel_scratch = reinterpret_cast<std::uint16_t*>(gate_up_scratch);
        sentinel_scratch[token] = __bfloat16_as_ushort(activated_bf16);
      }
    }
    consumer_group_sync<kConsumerThreads>();
    if (tid < kTokens) {
      std::size_t const sentinel_index =
          (static_cast<std::size_t>(tid) * 9) * kLocalInter + row_group * 16 + 15;
      publish_up_gate_sentinel<Config>(up_gate + sentinel_index,
                                       reinterpret_cast<std::uint16_t*>(gate_up_scratch)[tid]);
    }
  }
  gate_up_phase_sync<kThreads>();
}

template <class Config>
  requires((Config::kTokens == 8 && Config::kTpSize == 8))
__device__ void shared_gate_up(Glm5MegaMoeArgs const& args, __nv_bfloat16 const* global_normalized,
                               std::uint32_t const* normalized_ready, int row_group,
                               int pipeline_task_base, __nv_bfloat16* up_gate,
                               PipelineControl* control, std::uint8_t* dynamic_storage) {
  int const tid = static_cast<int>(threadIdx.x);

  if (tid >= kConsumerThreads && (tid & (kWarpSize - 1)) == 0) {
    int const producer_warp = (tid - kConsumerThreads) / kWarpSize;
#pragma unroll 1
    for (int page = 0; page < kGateUpPagesPerGroup; ++page) {
#pragma unroll
      for (int local_group = 0; local_group < kSharedRowGroupsPerCta; ++local_group) {
        int const local_task = page * kSharedRowGroupsPerCta + local_group;
        int const task = pipeline_task_index(local_task, pipeline_task_base);
        produce_gate_up_page<Config>(args, control, dynamic_storage, kRoutedExperts,
                                     row_group + local_group, page, task, 0, producer_warp);
      }
    }
  } else if (tid < kConsumerThreads) {
    wait_normalized_tokens<Config>(normalized_ready);
    auto* activation_pages =
        reinterpret_cast<__nv_bfloat16*>(dynamic_storage + kSharedActivationOffset);
    int const warp = tid >> 5;
    int const lane = tid & 31;
    int const local_group = warp / kSharedConsumerWarpsPerGroup;
    int const k_partition = warp % kSharedConsumerWarpsPerGroup;
    auto* partial = reinterpret_cast<float*>(dynamic_storage + kSharedPartialOffset);
    auto* results = reinterpret_cast<float*>(dynamic_storage + kSharedResultOffset);
    auto* sentinels = reinterpret_cast<std::uint16_t*>(dynamic_storage + kSharedSentinelOffset);
    Hmma::Accumulator gate_accumulator{{0.0F, 0.0F, 0.0F, 0.0F}};
    Hmma::Accumulator up_accumulator{{0.0F, 0.0F, 0.0F, 0.0F}};

    issue_shared_activation_page<Config>(global_normalized, 0, activation_pages);
    wait_shared_activation_page<Config>(activation_pages);
#pragma unroll 1
    for (int page = 0; page < kGateUpPagesPerGroup; ++page) {
      int const page_buffer = page & 1;
      if (page + 1 < kGateUpPagesPerGroup) {
        issue_shared_activation_page<Config>(
            global_normalized, page + 1,
            activation_pages + ((page_buffer ^ 1) * kSharedActivationPageElements));
      }
      int const local_task = page * kSharedRowGroupsPerCta + local_group;
      int const task = pipeline_task_index(local_task, pipeline_task_base);
      consumer_wait_stage(control, task);
#pragma unroll 1
      for (int sub_partition = 0; sub_partition < 4; ++sub_partition) {
        accumulate_gate_up_page_dual<Config, kTokens>(
            pipeline_stage<PipelineControl>(dynamic_storage, task),
            reinterpret_cast<StagedActivation const*>(activation_pages +
                                                      page_buffer * kSharedActivationPageElements),
            1024, k_partition * 4 + sub_partition, 0, gate_accumulator, up_accumulator);
      }
      consumer_release_stage(control, task);
      if (page + 1 < kGateUpPagesPerGroup) {
        wait_shared_activation_page<Config>(activation_pages +
                                            ((page_buffer ^ 1) * kSharedActivationPageElements));
      }
    }
    consumer_group_sync<kConsumerThreads>();

    int const thread_in_group = lane & 3;
    int const row = lane >> 2;
    int const token0 = thread_in_group * 2;
    int const token1 = token0 + 1;
    int const gate_partial_base = ((local_group * 2) * 2 + k_partition) * (kTokens * 16);
    int const up_partial_base = ((local_group * 2 + 1) * 2 + k_partition) * (kTokens * 16);
    partial[gate_partial_base + token0 * 16 + row] = gate_accumulator.element[0];
    partial[gate_partial_base + token1 * 16 + row] = gate_accumulator.element[1];
    partial[gate_partial_base + token0 * 16 + row + 8] = gate_accumulator.element[2];
    partial[gate_partial_base + token1 * 16 + row + 8] = gate_accumulator.element[3];
    partial[up_partial_base + token0 * 16 + row] = up_accumulator.element[0];
    partial[up_partial_base + token1 * 16 + row] = up_accumulator.element[1];
    partial[up_partial_base + token0 * 16 + row + 8] = up_accumulator.element[2];
    partial[up_partial_base + token1 * 16 + row + 8] = up_accumulator.element[3];
    consumer_group_sync<kConsumerThreads>();

#pragma unroll 1
    for (int item = tid; item < kSharedGateUpResultElements; item += kConsumerThreads) {
      int const local_group = item / (2 * kTokens * 16);
      int const matrix_item = item % (2 * kTokens * 16);
      int const output_matrix = matrix_item / (kTokens * 16);
      int const output_item = matrix_item % (kTokens * 16);
      float value = 0.0F;
#pragma unroll
      for (int partition = 0; partition < kSharedConsumerWarpsPerGroup; ++partition) {
        value +=
            partial[((local_group * 2 + output_matrix) * kSharedConsumerWarpsPerGroup + partition) *
                        (kTokens * 16) +
                    output_item];
      }
      results[item] = value;
    }
    consumer_group_sync<kConsumerThreads>();

    constexpr int kSharedOutputElements = kSharedRowGroupsPerCta * kTokens * 16;
#pragma unroll 1
    for (int item = tid; item < kSharedOutputElements; item += kConsumerThreads) {
      int const local_group = item / (kTokens * 16);
      int const token_row = item % (kTokens * 16);
      int const token = token_row >> 4;
      int const output_row = (row_group + local_group) * 16 + (token_row & 15);
      int const result_base = local_group * 2 * kTokens * 16;
      float const gate = results[result_base + token_row];
      float const up = results[result_base + kTokens * 16 + token_row];
      float const activated = gate_silu_multiply(gate, up);
      __nv_bfloat16 const activated_bf16 = float_to_bf16(activated);
      std::size_t const output_index =
          (static_cast<std::size_t>(token) * 9) * kLocalInter + output_row;
      if ((token_row & 15) == 15) {
        sentinels[local_group * kTokens + token] = __bfloat16_as_ushort(activated_bf16);
      } else {
        store_up_gate_value<Config>(up_gate + output_index, activated_bf16);
      }
    }
    consumer_group_sync<kConsumerThreads>();
    if (tid < kSharedRowGroupsPerCta * kTokens) {
      int const local_group = tid / kTokens;
      int const token = tid % kTokens;
      std::size_t const sentinel_index =
          (static_cast<std::size_t>(token) * 9) * kLocalInter + (row_group + local_group) * 16 + 15;
      publish_up_gate_sentinel<Config>(up_gate + sentinel_index,
                                       sentinels[local_group * kTokens + token]);
    }
  }
  gate_up_phase_sync<kThreads>();
}

template <class Config>
  requires((Config::kTokens == 8 && Config::kTpSize == 4))
__device__ void shared_gate_up(Glm5MegaMoeArgs const& args, __nv_bfloat16 const* global_normalized,
                               std::uint32_t const* normalized_ready, int row_group,
                               int pipeline_task_base, __nv_bfloat16* up_gate,
                               PipelineControl* control, std::uint8_t* dynamic_storage) {
  int const tid = static_cast<int>(threadIdx.x);

  if (tid >= kConsumerThreads && (tid & (kWarpSize - 1)) == 0) {
    int const producer_warp = (tid - kConsumerThreads) / kWarpSize;
#pragma unroll 1
    for (int page = 0; page < kGateUpPagesPerGroup; ++page) {
#pragma unroll
      for (int local_group = 0; local_group < kSharedRowGroupsPerCta; ++local_group) {
        int const local_task = page * kSharedRowGroupsPerCta + local_group;
        int const task = pipeline_task_index(local_task, pipeline_task_base);
        produce_gate_up_page<Config>(
            args, control, dynamic_storage, kRoutedExperts, row_group + local_group, page, task,
            static_cast<std::uint64_t>(cute::TMA::CacheHintSm100::EVICT_FIRST), producer_warp);
      }
    }
  } else if (tid < kConsumerThreads) {
    wait_normalized_tokens<Config>(normalized_ready);
    auto* activation_pages =
        reinterpret_cast<__nv_bfloat16*>(dynamic_storage + kSharedActivationOffset);
    int const warp = tid >> 5;
    int const lane = tid & 31;
    int const local_group = warp;
    auto* sentinels = reinterpret_cast<std::uint16_t*>(dynamic_storage + kSharedSentinelOffset);
    Hmma::Accumulator gate_accumulator{{0.0F, 0.0F, 0.0F, 0.0F}};
    Hmma::Accumulator up_accumulator{{0.0F, 0.0F, 0.0F, 0.0F}};

    issue_shared_activation_page<Config>(global_normalized, 0, activation_pages);
    wait_shared_activation_page<Config>(activation_pages);
#pragma unroll 1
    for (int page = 0; page < kGateUpPagesPerGroup; ++page) {
      int const page_buffer = page & 1;
      if (page + 1 < kGateUpPagesPerGroup) {
        issue_shared_activation_page<Config>(
            global_normalized, page + 1,
            activation_pages + ((page_buffer ^ 1) * kSharedActivationPageElements));
      }
      // Four pipeline stages can hold four row groups at a time. Consume the
      // eight groups in two ordered batches so no warp can observe the stale
      // phase of a stage that is still owned by the preceding group.
#pragma unroll 1
      for (int group_batch = 0; group_batch < 2; ++group_batch) {
        int const first_group = group_batch * kPipelineStages;
        if (local_group >= first_group && local_group < first_group + kPipelineStages) {
          int const local_task = page * kSharedRowGroupsPerCta + local_group;
          int const task = pipeline_task_index(local_task, pipeline_task_base);
          consumer_wait_stage(control, task);
#pragma unroll 1
          for (int sub_partition = 0; sub_partition < kConsumerWarps; ++sub_partition) {
            accumulate_gate_up_page_dual<Config, kTokens>(
                pipeline_stage<PipelineControl>(dynamic_storage, task),
                reinterpret_cast<StagedActivation const*>(
                    activation_pages + page_buffer * kSharedActivationPageElements),
                1024, sub_partition, 0, gate_accumulator, up_accumulator);
          }
          consumer_release_stage(control, task);
        }
        if (group_batch == 0) {
          consumer_group_sync<kConsumerThreads>();
        }
      }
      if (page + 1 < kGateUpPagesPerGroup) {
        wait_shared_activation_page<Config>(activation_pages +
                                            ((page_buffer ^ 1) * kSharedActivationPageElements));
      }
    }
    int const thread_in_group = lane & 3;
    int const row = lane >> 2;
    int const token0 = thread_in_group * 2;
    int const token1 = token0 + 1;
#pragma unroll
    for (int element = 0; element < 4; ++element) {
      int const token = (element & 1) != 0 ? token1 : token0;
      int const output_row = (row_group + local_group) * 16 + row + (element >= 2 ? 8 : 0);
      float const activated =
          gate_silu_multiply(gate_accumulator.element[element], up_accumulator.element[element]);
      __nv_bfloat16 const activated_bf16 = float_to_bf16(activated);
      std::size_t const output_index =
          (static_cast<std::size_t>(token) * 9) * kLocalInter + output_row;
      if ((output_row & 15) == 15) {
        sentinels[local_group * kTokens + token] = __bfloat16_as_ushort(activated_bf16);
      } else {
        store_up_gate_value<Config>(up_gate + output_index, activated_bf16);
      }
    }
    consumer_group_sync<kConsumerThreads>();
    if (tid < kSharedRowGroupsPerCta * kTokens) {
      int const local_group = tid / kTokens;
      int const token = tid % kTokens;
      std::size_t const sentinel_index =
          (static_cast<std::size_t>(token) * 9) * kLocalInter + (row_group + local_group) * 16 + 15;
      publish_up_gate_sentinel<Config>(up_gate + sentinel_index,
                                       sentinels[local_group * kTokens + token]);
    }
  }
  gate_up_phase_sync<kThreads>();
}

template <class Config, int kDownProducer, class Layout>
  requires(Config::kTokens == 4)
__device__ __forceinline__ void produce_down_virtual_tiles(
    Layout layout, typename Layout::Args const& args, PipelineControl* control,
    std::uint8_t* dynamic_storage, std::int32_t const* local_selected_indices, int tile) {
#pragma unroll 1
  // 48 output dim for each virtual tile
  for (int virtual_tile = 0; virtual_tile < kDownVirtualTilesPerCta; ++virtual_tile) {
    produce_down_expert_pair_k_tasks<Config, kDownProducer, 0>(
        layout, args, control, dynamic_storage, local_selected_indices, tile, virtual_tile);
    produce_down_expert_pair_k_tasks<Config, kDownProducer, 1>(
        layout, args, control, dynamic_storage, local_selected_indices, tile, virtual_tile);
    produce_down_expert_pair_k_tasks<Config, kDownProducer, 2>(
        layout, args, control, dynamic_storage, local_selected_indices, tile, virtual_tile);
    produce_down_expert_pair_k_tasks<Config, kDownProducer, 3>(
        layout, args, control, dynamic_storage, local_selected_indices, tile, virtual_tile);
    produce_down_expert_pair_k_tasks<Config, kDownProducer, 4>(
        layout, args, control, dynamic_storage, local_selected_indices, tile, virtual_tile);
  }
}

template <class Config, int kTmaProducer>
  requires(Config::kTokens == 8)
__device__ __forceinline__ void produce_down_virtual_tiles(
    Glm5MegaMoeArgs const& args, PipelineControl* control, std::uint8_t* dynamic_storage,
    std::int32_t const* local_selected_indices, int down_m16_base, int virtual_tile_count,
    int down_task_base) {
#pragma unroll 1
  for (int local_tile = 0; local_tile < virtual_tile_count; ++local_tile) {
    int const global_m16 = down_m16_base + local_tile * kDownM16PerVirtualTile;
    int const task_base = down_task_base + local_tile * kDownExpertPairs * kDownKTasks;
    produce_down_expert_pairs<Config, kTmaProducer>(args, control, dynamic_storage,
                                                    local_selected_indices, global_m16, task_base);
  }
}

template <class Config>
  requires(Config::kTokens == 4)
__device__ __forceinline__ void produce_down_from_thread(Layout layout, Glm5MegaMoeArgs const& args,
                                                         PipelineControl* control,
                                                         std::uint8_t* dynamic_storage,
                                                         std::int32_t const* local_selected_indices,
                                                         int tile, int tid) {
  int const producer_warp = (tid - kConsumerThreads) / kWarpSize;
  if ((tid & (kWarpSize - 1)) != 0) {
    return;
  }
  if (producer_warp == 0) {
    produce_down_virtual_tiles<Config, 0>(layout, args, control, dynamic_storage,
                                          local_selected_indices, tile);
  } else if (producer_warp == 1) {
    produce_down_virtual_tiles<Config, 1>(layout, args, control, dynamic_storage,
                                          local_selected_indices, tile);
  }
}

template <class Config>
  requires((Config::kTokens == 8 && Config::kTpSize == 8))
__device__ void down_and_finalize_k32_partitioned(
    Glm5MegaMoeArgs const& args, __nv_bfloat16* up_gate, std::int32_t const* local_selected_indices,
    float const* local_selected_probs, int token, int routed_local, int down_m16_base,
    int down_m16_count, int down_task_base, __nv_bfloat16* local_output,
    __nv_bfloat16* ar_local_partial, PipelineControl* control, std::uint8_t* dynamic_storage) {
  int const tid = static_cast<int>(threadIdx.x);

  int const virtual_tile_count = down_virtual_tile_count<Config>(down_m16_count);
  if (tid >= kConsumerThreads) {
    int const producer_warp = (tid - kConsumerThreads) / kWarpSize;
    int const lane = tid & (kWarpSize - 1);
    if (lane == 0) {
      if (producer_warp == 0) {
        produce_down_virtual_tiles<Config, 0>(args, control, dynamic_storage,
                                              local_selected_indices, down_m16_base,
                                              virtual_tile_count, down_task_base);
      } else if (producer_warp == 2) {
        produce_down_virtual_tiles<Config, 1>(args, control, dynamic_storage,
                                              local_selected_indices, down_m16_base,
                                              virtual_tile_count, down_task_base);
      } else if (producer_warp == 3) {
        produce_down_virtual_tiles<Config, 2>(args, control, dynamic_storage,
                                              local_selected_indices, down_m16_base,
                                              virtual_tile_count, down_task_base);
      }
    }
    if (args.ar_epoch_counters != nullptr && routed_local == 0 &&
        producer_warp == kDownResetProducerWarp) {
      reset_inputs_from_producer_warp<Config>(args, up_gate, token, dynamic_storage);
    }
    return;
  }

  int const world_size = args.world_size;
  int const warp = tid >> 5;
  int const lane = tid & 31;
  int const group = lane >> 2;
  int const thread_in_group = lane & 3;

  wait_up_gate_sentinels<Config>(up_gate, token);
  StagedActivation const* down_activation =
      stage_up_gate_activations<Config>(up_gate, token, dynamic_storage);
  if (args.ar_epoch_counters != nullptr && tid == 0) {
    std::uint32_t const generation =
        *reinterpret_cast<std::uint32_t const*>(dynamic_storage + kArRemoteOffset);
    store_release_gpu_u32(args.ar_epoch_counters + token * kRoutedCtasPerToken + routed_local,
                          generation);
  }
  int const result_elements = down_m16_count * kDownM16Rows;
  int const payload_vectors = result_elements / kArValuesPerVector;
  if (world_size != 0 && tid < payload_vectors) {
    auto const* residual_vectors = reinterpret_cast<uint4 const*>(
        static_cast<__nv_bfloat16 const*>(args.residual) +
        static_cast<std::size_t>(token) * kHidden + down_m16_base * kDownM16Rows);
    auto* staged_residual = reinterpret_cast<uint4*>(dynamic_storage + kArResidualOffset);
    copy_16b_g2s(residual_vectors + tid, staged_residual + tid);
  }
  if (world_size != 0) {
    cutlass::arch::cp_async_fence();
  }
  auto* reduction = reinterpret_cast<float*>(dynamic_storage + kDownReductionOffset);

#pragma unroll 1
  for (int local_tile = 0; local_tile < virtual_tile_count; ++local_tile) {
    int const global_m16 = down_m16_base + local_tile * kDownM16PerVirtualTile;
    float partial_low[kDownM16PerVirtualTile] = {};
    float partial_high[kDownM16PerVirtualTile] = {};

#pragma unroll
    for (int expert_pair = 0; expert_pair < kDownExpertPairs; ++expert_pair) {
      int const pipeline_task = down_task_base + local_tile * kDownExpertPairs + expert_pair;
      consumer_wait_stage(control, pipeline_task);
      std::uint8_t const* stage = pipeline_stage<PipelineControl>(dynamic_storage, pipeline_task);
      int const experts_in_pair = expert_pair + 1 == kDownExpertPairs ? 1 : 2;

#pragma unroll
      for (int expert_in_pair = 0; expert_in_pair < experts_in_pair; ++expert_in_pair) {
        int const slot = expert_pair * 2 + expert_in_pair;
        float const probability = slot == 0 ? 1.0F : local_selected_probs[slot - 1];
        StagedActivation const* activation =
            down_activation + static_cast<std::size_t>(slot) * kLocalInter;
        accumulate_down_k32_partition<Config>(stage, activation, expert_in_pair, warp, global_m16,
                                              probability, partial_low, partial_high);
      }
      consumer_release_stage(control, pipeline_task);
    }

    if (thread_in_group == 0) {
#pragma unroll
      for (int m16 = 0; m16 < kDownM16PerVirtualTile; ++m16) {
        int const local_m16 = local_tile * kDownM16PerVirtualTile + m16;
        std::size_t const partial_base =
            (static_cast<std::size_t>(warp) * kDownMaxM16PerCta + local_m16) * kDownM16Rows;
        reduction[partial_base + group] = partial_low[m16];
        reduction[partial_base + group + 8] = partial_high[m16];
      }
    }
  }

  consumer_group_sync<kConsumerThreads>();
#pragma unroll 1
  for (int item = tid; item < result_elements; item += kConsumerThreads) {
    int const local_m16 = item / kDownM16Rows;
    int const row = item % kDownM16Rows;
    float result = 0.0F;
#pragma unroll
    for (int source_warp = 0; source_warp < kConsumerWarps; ++source_warp) {
      std::size_t const partial_index =
          (static_cast<std::size_t>(source_warp) * kDownMaxM16PerCta + local_m16) * kDownM16Rows +
          row;
      result += reduction[partial_index];
    }
    int const output_row = (down_m16_base + local_m16) * kDownM16Rows + row;
    __nv_bfloat16 const rounded = float_to_bf16(result);
    if (world_size != kFusedMoeTp8WorldSize && local_output != nullptr) {
      local_output[token * kHidden + output_row] = rounded;
    }
    if (ar_local_partial != nullptr) {
      ar_local_partial[item] = rounded;
    }
  }
  consumer_group_sync<kConsumerThreads>();
}

template <class Config>
  requires((Config::kTokens == 8 && Config::kTpSize == 4))
__device__ void down_and_finalize_k32_partitioned(
    Glm5MegaMoeArgs const& args, __nv_bfloat16* up_gate, std::int32_t const* local_selected_indices,
    float const* local_selected_probs, int token, int routed_local, int down_m16_base,
    int down_m16_count, int down_task_base, __nv_bfloat16* local_output,
    __nv_bfloat16* ar_local_partial, PipelineControl* control, std::uint8_t* dynamic_storage) {
  int const tid = static_cast<int>(threadIdx.x);

  int const virtual_tile_count = down_virtual_tile_count<Config>(down_m16_count);
  if (tid >= kConsumerThreads) {
    int const producer_warp = (tid - kConsumerThreads) / kWarpSize;
    int const lane = tid & (kWarpSize - 1);
    if (lane == 0) {
      if (producer_warp == 0) {
        produce_down_virtual_tiles<Config, 0>(args, control, dynamic_storage,
                                              local_selected_indices, down_m16_base,
                                              virtual_tile_count, down_task_base);
      } else if (producer_warp == 2) {
        produce_down_virtual_tiles<Config, 1>(args, control, dynamic_storage,
                                              local_selected_indices, down_m16_base,
                                              virtual_tile_count, down_task_base);
      } else if (producer_warp == 3) {
        produce_down_virtual_tiles<Config, 2>(args, control, dynamic_storage,
                                              local_selected_indices, down_m16_base,
                                              virtual_tile_count, down_task_base);
      }
    }
    if (args.ar_epoch_counters != nullptr && routed_local == 0 &&
        producer_warp == kDownResetProducerWarp) {
      reset_inputs_from_producer_warp<Config>(args, up_gate, token, dynamic_storage);
    }
    return;
  }

  int const world_size = args.world_size;
  int const warp = tid >> 5;
  int const lane = tid & 31;
  int const group = lane >> 2;
  int const thread_in_group = lane & 3;

  wait_up_gate_sentinels<Config>(up_gate, token);
  StagedActivation const* down_activation =
      stage_up_gate_activations<Config>(up_gate, token, dynamic_storage);
  if (args.ar_epoch_counters != nullptr && tid == 0) {
    std::uint32_t const generation =
        *reinterpret_cast<std::uint32_t const*>(dynamic_storage + kArRemoteOffset);
    store_release_gpu_u32(args.ar_epoch_counters + token * kRoutedCtasPerToken + routed_local,
                          generation);
  }
  int const result_elements = down_m16_count * kDownM16Rows;
  int const payload_vectors = result_elements / kArValuesPerVector;
  if (world_size != 0 && tid < payload_vectors) {
    auto const* residual_vectors = reinterpret_cast<uint4 const*>(
        static_cast<__nv_bfloat16 const*>(args.residual) +
        static_cast<std::size_t>(token) * kHidden + down_m16_base * kDownM16Rows);
    auto* staged_residual = reinterpret_cast<uint4*>(dynamic_storage + kArResidualOffset);
    copy_16b_g2s(residual_vectors + tid, staged_residual + tid);
  }
  if (world_size != 0) {
    cutlass::arch::cp_async_fence();
  }
  auto* reduction = reinterpret_cast<float*>(dynamic_storage + kDownReductionOffset);

#pragma unroll 1
  for (int local_tile = 0; local_tile < virtual_tile_count; ++local_tile) {
    float partial_low[kDownM16PerVirtualTile] = {};
    float partial_high[kDownM16PerVirtualTile] = {};

#pragma unroll
    for (int expert_pair = 0; expert_pair < kDownExpertPairs; ++expert_pair) {
      int const experts_in_pair = expert_pair + 1 == kDownExpertPairs ? 1 : 2;

#pragma unroll 1
      for (int down_k_task = 0; down_k_task < kDownKTasks; ++down_k_task) {
        int const pipeline_task = down_task_base +
                                  (local_tile * kDownExpertPairs + expert_pair) * kDownKTasks +
                                  down_k_task;
        consumer_wait_stage(control, pipeline_task);
        std::uint8_t const* stage = pipeline_stage<PipelineControl>(dynamic_storage, pipeline_task);
        int const global_k32_partition = down_k_task * kDownK32PartitionsPerTask + warp;

#pragma unroll
        for (int expert_in_pair = 0; expert_in_pair < experts_in_pair; ++expert_in_pair) {
          int const slot = expert_pair * 2 + expert_in_pair;
          float const probability = slot == 0 ? 1.0F : local_selected_probs[slot - 1];
          StagedActivation const* activation =
              down_activation + static_cast<std::size_t>(slot) * kLocalInter;
          int const global_m16 = down_m16_base + local_tile * kDownM16PerVirtualTile;
          accumulate_down_k32_partition<Config>(stage, activation, expert_in_pair, warp,
                                                global_k32_partition, global_m16, probability,
                                                partial_low, partial_high);
        }
        consumer_release_stage(control, pipeline_task);
      }
    }

    if (thread_in_group == 0) {
#pragma unroll
      for (int m16 = 0; m16 < kDownM16PerVirtualTile; ++m16) {
        int const local_m16 = local_tile * kDownM16PerVirtualTile + m16;
        std::size_t const partial_base =
            (static_cast<std::size_t>(warp) * kDownMaxM16PerCta + local_m16) * kDownM16Rows;
        reduction[partial_base + group] = partial_low[m16];
        reduction[partial_base + group + 8] = partial_high[m16];
      }
    }
  }
  consumer_group_sync<kConsumerThreads>();
  int const result_pairs = result_elements / 2;
  if (tid < result_pairs) {
    float2 result = make_float2(0.0F, 0.0F);
#pragma unroll
    for (int source_warp = 0; source_warp < kConsumerWarps; ++source_warp) {
      std::size_t const partial_index =
          static_cast<std::size_t>(source_warp) * kDownMaxM16PerCta * kDownM16Rows + tid * 2;
      uint2 const values =
          comm::load_shared_u32x2(cute::cast_smem_ptr_to_uint(reduction + partial_index));
      result =
          comm::add_f32x2(result, make_float2(__uint_as_float(values.x), __uint_as_float(values.y)));
    }
    std::uint32_t const rounded = comm::pack_bf16x2(result.x, result.y);
    if (world_size != kFusedMoeWorldSize && local_output != nullptr) {
      auto* output_pairs = reinterpret_cast<std::uint32_t*>(
          local_output + static_cast<std::size_t>(token) * kHidden + down_m16_base * kDownM16Rows);
      output_pairs[tid] = rounded;
    }
    if (ar_local_partial != nullptr) {
      reinterpret_cast<std::uint32_t*>(ar_local_partial)[tid] = rounded;
    }
  }
  consumer_group_sync<kConsumerThreads>();
}

template <class Config, class Layout>
  requires(Config::kTokens == 4)
__device__ void routed_down(Layout layout, typename Layout::Args const& args,
                            __nv_bfloat16* up_gate, __nv_bfloat16 const* residual,
                            std::int32_t const* local_selected_indices,
                            float const* local_selected_probs, int token, int tile,
                            __nv_bfloat16* local_output, __nv_bfloat16* ar_local_partial,
                            PipelineControl* control, std::uint8_t* dynamic_storage) {
  int const tid = static_cast<int>(threadIdx.x);

  if (tid >= kConsumerThreads) {
    int const producer_warp = (tid - kConsumerThreads) / kWarpSize;
    produce_down_from_thread<Config>(layout, args, control, dynamic_storage, local_selected_indices,
                                     tile, tid);
    if (args.ar_epoch_counters != nullptr && tile == 0 && producer_warp == 0) {
      reset_inputs_from_producer_warp<Config>(args, up_gate, token, dynamic_storage);
    }
    return;
  }

  int const world_size = args.world_size;
  int const warp = tid >> 5;
  int const lane = tid & 31;
  int const group = lane >> 2;
  int const thread_in_group = lane & 3;

  wait_up_gate_sentinels<Config>(up_gate, token);
  StagedActivation const* down_activation =
      stage_up_gate_activations<Config>(up_gate, token, dynamic_storage);
  if (args.ar_epoch_counters != nullptr && tid == 0) {
    int const counter = token * kArTilesPerToken + tile;
    std::uint32_t const generation =
        *reinterpret_cast<std::uint32_t const*>(dynamic_storage + kArRemoteOffset);
    store_release_gpu_u32(args.ar_epoch_counters + counter, generation);
  }
  if (residual != nullptr) {
    if (tid < kArPayloadVectors) {
      auto const* residual_vectors =
          reinterpret_cast<uint4 const*>(residual + static_cast<std::size_t>(token) * kHidden +
                                         static_cast<std::size_t>(tile) * kArTileElements);
      auto* staged_residual = reinterpret_cast<uint4*>(dynamic_storage + kArResidualOffset);
      copy_16b_g2s(residual_vectors + tid, staged_residual + tid);
    }
    cutlass::arch::cp_async_fence();
  }
  auto* reduction = reinterpret_cast<float*>(dynamic_storage + kDownReductionOffset);

#pragma unroll 1
  for (int virtual_tile = 0; virtual_tile < kDownVirtualTilesPerCta; ++virtual_tile) {
    float partial_low[3] = {0.0F, 0.0F, 0.0F};
    float partial_high[3] = {0.0F, 0.0F, 0.0F};

#pragma unroll
    for (int expert_pair = 0; expert_pair < kDownExpertPairs; ++expert_pair) {
      int const experts_in_pair = expert_pair + 1 == kDownExpertPairs ? 1 : 2;

#pragma unroll
      for (int down_k_task = 0; down_k_task < kDownKTasks; ++down_k_task) {
        int const task =
            (virtual_tile * kDownExpertPairs + expert_pair) * kDownKTasks + down_k_task;
        int const pipeline_task = pipeline_task_index(task, kRoutedDownPipelineTaskBase);
        consumer_wait_stage(control, pipeline_task);
        std::uint8_t const* stage = pipeline_stage<PipelineControl>(dynamic_storage, pipeline_task);
        int const global_k32_partition = down_k_task * kDownK32PartitionsPerTask + warp;

#pragma unroll
        for (int expert_in_pair = 0; expert_in_pair < experts_in_pair; ++expert_in_pair) {
          int const slot = expert_pair * 2 + expert_in_pair;
          float const probability = slot == 0 ? 1.0F : local_selected_probs[slot - 1];
          StagedActivation const* activation =
              down_activation + static_cast<std::size_t>(slot) * kLocalInter;
          int const global_virtual_tile = tile * kDownVirtualTilesPerCta + virtual_tile;
          accumulate_down_k32_partition<Config>(layout, stage, activation, expert_in_pair, warp,
                                                global_k32_partition, global_virtual_tile,
                                                probability, partial_low, partial_high);
        }
        consumer_release_stage(control, pipeline_task);
      }
    }

    if (thread_in_group == 0) {
      std::size_t const partial_base =
          (static_cast<std::size_t>(warp) * kDownVirtualTilesPerCta + virtual_tile) * 48;
#pragma unroll
      for (int m16 = 0; m16 < 3; ++m16) {
        int const row = m16 * 16 + group;
        reduction[partial_base + row] = partial_low[m16];
        reduction[partial_base + row + 8] = partial_high[m16];
      }
    }
  }

  consumer_group_sync<kConsumerThreads>();
  if (tid < kArTileElements) {
    float result = 0.0F;
#pragma unroll
    for (int source_warp = 0; source_warp < kConsumerWarps; ++source_warp) {
      std::size_t const partial_index =
          static_cast<std::size_t>(source_warp) * kArTileElements + tid;
      result += reduction[partial_index];
    }
    int const output_row = tile * kArTileElements + tid;
    __nv_bfloat16 const rounded = float_to_bf16(result);
    if (world_size != kFusedMoeWorldSize && local_output != nullptr) {
      local_output[token * kHidden + output_row] = rounded;
    }
    if (ar_local_partial != nullptr) {
      ar_local_partial[tid] = rounded;
    }
  }
  consumer_group_sync<kConsumerThreads>();
}

}  // namespace glm5::moe::detail
