#pragma once

// Shared Attention addressing, FP8 decoding, softmax and Combine fragments. Inline attributes, reduction order and MMA shapes are preserved.
#include <mla/projection.cuh>
#include <cutlass/arch/reg_reconfig.h>

namespace glm5::mla::detail {

__device__ __forceinline__ std::size_t swizzle_128b_byte_offset(std::size_t logical_byte_offset) {
  return logical_byte_offset ^ ((logical_byte_offset & 0x380U) >> 3U);
}

template <class Byte>
__device__ __forceinline__ Byte* attention_swizzled_matrix_address(Byte* chunk_base, int row,
                                                                   int column) {
  std::size_t const logical_byte_offset = static_cast<std::size_t>(row) * kGatherRowBytes +
                                          static_cast<std::size_t>(column) *
                                              sizeof(InternalActivation);
  return chunk_base + swizzle_128b_byte_offset(logical_byte_offset);
}

template <class Byte>
__device__ __forceinline__ Byte* attention_swizzled_fp8_address(Byte* chunk_base, int row,
                                                                int column, int row_bytes) {
  std::size_t const logical_byte_offset =
      static_cast<std::size_t>(row) * row_bytes + static_cast<std::size_t>(column);
  return chunk_base + swizzle_128b_byte_offset(logical_byte_offset);
}



template <class Byte>
__device__ __forceinline__ Byte* query_fragment_row_address(Byte* dynamic_storage,
                                                            int scratch_offset, int column_base,
                                                            int lane) {
  int const matrix = lane >> 3;
  int const row = (lane & 7) + ((matrix & 1) << 3);
  int const column = column_base + (matrix >> 1) * 8;
  int const chunk = column >> 6;
  return attention_swizzled_matrix_address(dynamic_storage + scratch_offset + chunk * 0x800, row,
                                           column & 63);
}

__device__ __forceinline__ std::uint8_t const* key_pe_fp8_address(
    std::uint8_t const* dynamic_storage, int local_key, int column) {
  return attention_swizzled_fp8_address(
      dynamic_storage + kAttentionKeyPeOffset + (local_key >> 4) * 0x400, local_key & 15, column,
      kRopeDim);
}

__device__ __forceinline__ void load_fp8_col_major_b_fragment(std::uint8_t const* pair0_address,
                                                              std::uint8_t const* pair1_address,
                                                              Hmma::BFragment& fragment) {
  auto* words = reinterpret_cast<std::uint32_t*>(fragment.element);
  words[0] = e4m3x2_to_mma_x2(*reinterpret_cast<std::uint16_t const*>(pair0_address));
  words[1] = e4m3x2_to_mma_x2(*reinterpret_cast<std::uint16_t const*>(pair1_address));
}

__device__ __forceinline__ void decode_fp8_m16n16_b_fragments(std::uint8_t const* address,
                                                              Hmma::BFragment& low_fragment,
                                                              Hmma::BFragment& high_fragment) {
  std::uint32_t packed_low_high0;
  std::uint32_t packed_low_high1;
  cute::SM100_U8x8_LDSM_T::copy(*reinterpret_cast<cute::uint128_t const*>(address),
                                packed_low_high0, packed_low_high1);
  auto* low_words = reinterpret_cast<std::uint32_t*>(low_fragment.element);
  auto* high_words = reinterpret_cast<std::uint32_t*>(high_fragment.element);
  low_words[0] = e4m3x2_to_mma_x2(static_cast<std::uint16_t>(packed_low_high0));
  low_words[1] = e4m3x2_to_mma_x2(static_cast<std::uint16_t>(packed_low_high1));
  high_words[0] = e4m3x2_to_mma_x2(static_cast<std::uint16_t>(packed_low_high0 >> 16));
  high_words[1] = e4m3x2_to_mma_x2(static_cast<std::uint16_t>(packed_low_high1 >> 16));
}

__device__ __forceinline__ void load_query_attention_fragment(std::uint8_t const* dynamic_storage,
                                                              int scratch_offset, int column_base,
                                                              int lane, Hmma::AFragment& fragment) {
  Hmma::load_a_shared(
      fragment, query_fragment_row_address(dynamic_storage, scratch_offset, column_base, lane));
}

__device__ __forceinline__ float run_warp_distributed_attention_softmax(
    Hmma::Accumulator const& score_low, Hmma::Accumulator const& score_high, int warp, int lane,
    int valid_count, std::uint8_t* dynamic_storage, float* lse_staging) {
  constexpr int kHeadGroupWarps = kTpWorldSize == 4 ? 4 : kAttentionQkWarps;
  int const head_group = kTpWorldSize == 4 ? warp >> 2 : 0;
  int const group_warp = kTpWorldSize == 4 ? warp & 3 : warp;
  int const head = lane >> 2;
  int const lane_in_head = lane & 3;
  int const key0 = group_warp * 16 + 2 * lane_in_head;
  int const key1 = key0 + 1;
  int const key2 = key0 + 8;
  int const key3 = key2 + 1;

  float scores[4];
  scores[0] = key0 < valid_count ? score_low.element[0] : -FLT_MAX;
  scores[1] = key1 < valid_count ? score_low.element[1] : -FLT_MAX;
  scores[2] = key2 < valid_count ? score_high.element[0] : -FLT_MAX;
  scores[3] = key3 < valid_count ? score_high.element[1] : -FLT_MAX;

  float local_maximum = fmaxf(fmaxf(scores[0], scores[1]), fmaxf(scores[2], scores[3]));
  local_maximum = fmaxf(local_maximum, __shfl_down_sync(kFullWarpMask, local_maximum, 2));
  local_maximum = fmaxf(local_maximum, __shfl_down_sync(kFullWarpMask, local_maximum, 1));

  auto* softmax_scratch =
      reinterpret_cast<volatile float*>(dynamic_storage + kAttentionSoftmaxScratchOffset);
  int const scratch_index = warp * kAttentionSoftmaxWarpStride + head * kAttentionSoftmaxHeadStride;
  if (lane_in_head == 0) {
    softmax_scratch[scratch_index] = local_maximum;
  }
  // Synchronize QK warps before reducing their per-warp maxima.
  cutlass::arch::NamedBarrier::sync(kAttentionQkThreads,
                                    cutlass::arch::ReservedNamedBarriers::StreamkBarrier1);

  float maximum = -FLT_MAX;
  if (lane_in_head == 0) {
#pragma unroll
    for (int source_warp = 0; source_warp < kHeadGroupWarps; ++source_warp) {
      maximum = fmaxf(maximum, softmax_scratch[(head_group * kHeadGroupWarps + source_warp) * kAttentionSoftmaxWarpStride +
                                               head * kAttentionSoftmaxHeadStride]);
    }
  }
  maximum = __shfl_sync(kFullWarpMask, maximum, head * 4);

  float probabilities[4];
  probabilities[0] =
      key0 < valid_count ? exp2_underflow_safe((scores[0] - maximum) * kAttentionLog2Scale) : 0.0F;
  probabilities[1] =
      key1 < valid_count ? exp2_underflow_safe((scores[1] - maximum) * kAttentionLog2Scale) : 0.0F;
  probabilities[2] =
      key2 < valid_count ? exp2_underflow_safe((scores[2] - maximum) * kAttentionLog2Scale) : 0.0F;
  probabilities[3] =
      key3 < valid_count ? exp2_underflow_safe((scores[3] - maximum) * kAttentionLog2Scale) : 0.0F;

  float local_denominator =
      probabilities[0] + probabilities[1] + probabilities[2] + probabilities[3];
  local_denominator += __shfl_down_sync(kFullWarpMask, local_denominator, 2);
  local_denominator += __shfl_down_sync(kFullWarpMask, local_denominator, 1);
  if (lane_in_head == 0) {
    softmax_scratch[scratch_index] = local_denominator;
  }
  // Make all per-warp denominators visible before the final reduction.
  cutlass::arch::NamedBarrier::sync(kAttentionQkThreads,
                                    cutlass::arch::ReservedNamedBarriers::StreamkBarrier1);

  float denominator = 0.0F;
  if (lane_in_head == 0) {
#pragma unroll
    for (int source_warp = 0; source_warp < kHeadGroupWarps; ++source_warp) {
      denominator += softmax_scratch[(head_group * kHeadGroupWarps + source_warp) * kAttentionSoftmaxWarpStride +
                                     head * kAttentionSoftmaxHeadStride];
    }
  }
  denominator = __shfl_sync(kFullWarpMask, denominator, head * 4);
  // Clamp empty-split denominators to avoid division by zero.
  // Their probabilities remain zero and LSE remains -inf.
  float const inverse = reciprocal_approx_ftz(fmaxf(denominator, 1.0F));

  // Accumulate PV in FP32 using unnormalized b16 probabilities, then normalize.
  Hmma::AFragment probability_fragment{};
  probability_fragment.element[0] = float_to_internal(probabilities[0]);
  probability_fragment.element[1] = float_to_internal(probabilities[1]);
  probability_fragment.element[4] = float_to_internal(probabilities[2]);
  probability_fragment.element[5] = float_to_internal(probabilities[3]);
  if constexpr (kTpWorldSize == 4) {
    // Write only this group's rows to avoid overwriting the other group's data.
    auto const* words = reinterpret_cast<std::uint32_t const*>(probability_fragment.element);
    auto* destination = attention_swizzled_matrix_address(
        dynamic_storage + kAttentionProbabilityOffset,
        head_group * 8 + (lane & 7),
        group_warp * 16 + ((lane >> 3) & 1) * 8);
    std::uint32_t const address =
        static_cast<std::uint32_t>(__cvta_generic_to_shared(destination));
    asm volatile("stmatrix.sync.aligned.m8n8.x2.shared.b16 [%0], {%1, %2};"
                 :
                 : "r"(address), "r"(words[0]), "r"(words[2])
                 : "memory");
  } else {
    Hmma::store_a_shared(
        probability_fragment,
        query_fragment_row_address(dynamic_storage, kAttentionProbabilityOffset, warp * 16, lane));
  }

  if (group_warp == 0 && lane_in_head == 0) {
    lse_staging[head_group * 8 + head] = denominator == 0.0F
                            ? -CUDART_INF_F
                            : maximum * kAttentionLog2Scale + log2_approx(denominator);
  }
  // Finish staging probability fragments before the PV matrix loads.
  cutlass::arch::NamedBarrier::sync(kAttentionQkThreads,
                                    cutlass::arch::ReservedNamedBarriers::StreamkBarrier1);
  return inverse;
}

__device__ __forceinline__ InternalActivation* combine_partial_address(
    std::uint8_t* dynamic_storage, int attention_task, int split, int element) {
  // Reuse attention stages for partial outputs.
  // Keep the attention swizzle to avoid row aliasing.
  int const task = attention_task + 1 + split / kCombineSplitsPerStage;
  auto* stage = pipeline_stage(dynamic_storage, task, kAttentionStageBytes);
  int const chunk = element >> 6;
  int const column = element & 63;
  return reinterpret_cast<InternalActivation*>(attention_swizzled_matrix_address(
      stage + static_cast<std::size_t>(chunk) * 0x800,
      split % kCombineSplitsPerStage, column));
}

__device__ __forceinline__ void load_combine_weight_fragment(
    float const* normalized_weights, int split_base, int lane, Hmma::AFragment& fragment) {
  // Replicate the normalized split weights across MMA rows.
  int const pair = 2 * (lane & 3);
  std::uint32_t const low = pack_internal_x2(
      normalized_weights[split_base + pair], normalized_weights[split_base + pair + 1]);
  std::uint32_t const high = pack_internal_x2(
      normalized_weights[split_base + 8 + pair],
      normalized_weights[split_base + 8 + pair + 1]);
  auto* words = reinterpret_cast<std::uint32_t*>(fragment.element);
  words[0] = low;
  words[1] = low;
  words[2] = high;
  words[3] = high;
}

__device__ __forceinline__ void load_combine_partial_fragment(
    std::uint8_t* dynamic_storage, int attention_task, int split_base, int value_column, int lane,
    Hmma::BFragment& fragment) {
  int const address_lane = lane & 15;
  int const matrix = address_lane >> 3;
  int const split = split_base + matrix * 8 + (address_lane & 7);
  Hmma::template load_b_shared_transposed<false>(
      fragment,
      combine_partial_address(dynamic_storage, attention_task, split, value_column));
}

// Fill M16 with keys/value columns and N8 with eight real head columns.
__device__ __forceinline__ std::uint8_t const* tiled_gathered_kv_fp8_address(
    std::uint8_t const* dynamic_storage, int first_kv_task, int local_key, int latent_column) {
  int const chunk = latent_column >> 7;
  int const column_in_chunk = latent_column & 127;
  int const half = chunk / kGatherChunksPerKvStage;
  int const local_chunk = chunk % kGatherChunksPerKvStage;
  auto const* stage = pipeline_stage(
      dynamic_storage, first_kv_task + half, kAttentionStageBytes);
  return attention_swizzled_fp8_address(
      stage + static_cast<std::size_t>(local_chunk) * kGatherChunkBytes,
      local_key, column_in_chunk, kGatherRowBytes);
}

__device__ __forceinline__ void load_tiled_attention_head_b_fragment(
    std::uint8_t const* dynamic_storage, int scratch_offset, int column_base,
    int lane, Hmma::BFragment& fragment, int head_base = 0) {
  int const head = head_base + (lane & 7);
  int const column = column_base + ((lane >> 3) & 1) * 8;
  auto const* address = attention_swizzled_matrix_address(
      dynamic_storage + scratch_offset + (column >> 6) * 0x800, head,
      column & 63);
  Hmma::template load_b_shared<false>(fragment, address);
}

__device__ __forceinline__ void load_tiled_key_pe_row_a_fragment(
    std::uint8_t const* dynamic_storage, int key_base, int column_base,
    int lane, Hmma::AFragment& fragment) {
  int const key = key_base + (lane >> 2);
  int const column = column_base + 2 * (lane & 3);
  auto* words = reinterpret_cast<std::uint32_t*>(fragment.element);
  words[0] = e4m3x2_to_mma_x2(*reinterpret_cast<std::uint16_t const*>(
      key_pe_fp8_address(dynamic_storage, key, column)));
  words[1] = e4m3x2_to_mma_x2(*reinterpret_cast<std::uint16_t const*>(
      key_pe_fp8_address(dynamic_storage, key + 8, column)));
  words[2] = e4m3x2_to_mma_x2(*reinterpret_cast<std::uint16_t const*>(
      key_pe_fp8_address(dynamic_storage, key, column + 8)));
  words[3] = e4m3x2_to_mma_x2(*reinterpret_cast<std::uint16_t const*>(
      key_pe_fp8_address(dynamic_storage, key + 8, column + 8)));
}

__device__ __forceinline__ void load_tiled_key_latent_row_a_fragment(
    std::uint8_t const* dynamic_storage, int key_base, int column_base,
    int first_kv_task, int lane, Hmma::AFragment& fragment) {
  int const key = key_base + (lane >> 2);
  int const column = column_base + 2 * (lane & 3);
  auto* words = reinterpret_cast<std::uint32_t*>(fragment.element);
  words[0] = e4m3x2_to_mma_x2(*reinterpret_cast<std::uint16_t const*>(
      tiled_gathered_kv_fp8_address(dynamic_storage, first_kv_task, key, column)));
  words[1] = e4m3x2_to_mma_x2(*reinterpret_cast<std::uint16_t const*>(
      tiled_gathered_kv_fp8_address(dynamic_storage, first_kv_task, key + 8, column)));
  words[2] = e4m3x2_to_mma_x2(*reinterpret_cast<std::uint16_t const*>(
      tiled_gathered_kv_fp8_address(dynamic_storage, first_kv_task, key, column + 8)));
  words[3] = e4m3x2_to_mma_x2(*reinterpret_cast<std::uint16_t const*>(
      tiled_gathered_kv_fp8_address(dynamic_storage, first_kv_task, key + 8, column + 8)));
}

__device__ __forceinline__ void restore_tiled_attention_head_scores(
    Hmma::Accumulator const& transposed, int lane,
    Hmma::Accumulator& low, Hmma::Accumulator& high) {
  int const head = lane >> 2;
  int const source0 = 8 * (lane & 3) + (head >> 1);
  int const source1 = source0 + 4;
  bool const odd_head = (head & 1) != 0;
  float const l00 = __shfl_sync(kFullWarpMask, transposed.element[0], source0);
  float const l01 = __shfl_sync(kFullWarpMask, transposed.element[1], source0);
  float const l10 = __shfl_sync(kFullWarpMask, transposed.element[0], source1);
  float const l11 = __shfl_sync(kFullWarpMask, transposed.element[1], source1);
  float const h00 = __shfl_sync(kFullWarpMask, transposed.element[2], source0);
  float const h01 = __shfl_sync(kFullWarpMask, transposed.element[3], source0);
  float const h10 = __shfl_sync(kFullWarpMask, transposed.element[2], source1);
  float const h11 = __shfl_sync(kFullWarpMask, transposed.element[3], source1);
  low.element[0] = odd_head ? l01 : l00;
  low.element[1] = odd_head ? l11 : l10;
  high.element[0] = odd_head ? h01 : h00;
  high.element[1] = odd_head ? h11 : h10;
}


#if GLM5_MEGAMLA_TP_SIZE == 4 && GLM5_MEGAMLA_TOKENS == 8
// M16 holds all sixteen heads; four warps each own thirty-two keys.
__device__ __forceinline__ void run_full_head_attention_softmax_128(
    Hmma::Accumulator const (&scores)[4], int warp, int lane, int valid_count,
    std::uint8_t* dynamic_storage, float* lse_staging, float (&inverse)[2]) {
  constexpr int kComputeThreads = 128;
  constexpr int kScratchWarpStride = 32;
  int const head = lane >> 2;
  int const lane_in_head = lane & 3;
  auto* scratch = reinterpret_cast<volatile float*>(
      dynamic_storage + kAttentionSoftmaxScratchOffset);
  float score[2][8];
  float maximum[2];
#pragma unroll
  for (int half = 0; half < 2; ++half) {
    float local_maximum = -FLT_MAX;
#pragma unroll
    for (int key = 0; key < 8; ++key) {
      int const position = warp * 32 + (key >> 1) * 8 +
                           lane_in_head * 2 + (key & 1);
      score[half][key] = position < valid_count
          ? scores[key >> 1].element[half * 2 + (key & 1)] : -FLT_MAX;
      local_maximum = fmaxf(local_maximum, score[half][key]);
    }
    local_maximum = fmaxf(local_maximum,
        __shfl_down_sync(kFullWarpMask, local_maximum, 2));
    local_maximum = fmaxf(local_maximum,
        __shfl_down_sync(kFullWarpMask, local_maximum, 1));
    if (lane_in_head == 0) {
      scratch[warp * kScratchWarpStride + (head + half * 8) * 2] =
          local_maximum;
    }
  }
  cutlass::arch::NamedBarrier::sync(
      kComputeThreads, cutlass::arch::ReservedNamedBarriers::StreamkBarrier1);
#pragma unroll
  for (int half = 0; half < 2; ++half) {
    float result = -FLT_MAX;
    if (lane_in_head == 0) {
#pragma unroll
      for (int source_warp = 0; source_warp < 4; ++source_warp) {
        result = fmaxf(result, scratch[source_warp * kScratchWarpStride +
                                      (head + half * 8) * 2]);
      }
    }
    maximum[half] = __shfl_sync(kFullWarpMask, result, head * 4);
  }
  // Finish maximum reads before scratch changes from maxima to denominators.
  cutlass::arch::NamedBarrier::sync(
      kComputeThreads, cutlass::arch::ReservedNamedBarriers::StreamkBarrier1);
  float probability[2][8];
#pragma unroll
  for (int half = 0; half < 2; ++half) {
    float denominator = 0.0F;
#pragma unroll
    for (int key = 0; key < 8; ++key) {
      int const position = warp * 32 + (key >> 1) * 8 +
                           lane_in_head * 2 + (key & 1);
      float const value = position < valid_count
          ? exp2_underflow_safe((score[half][key] - maximum[half]) *
                                kAttentionLog2Scale) : 0.0F;
      probability[half][key] = value;
      denominator += value;
    }
    denominator += __shfl_down_sync(kFullWarpMask, denominator, 2);
    denominator += __shfl_down_sync(kFullWarpMask, denominator, 1);
    if (lane_in_head == 0) {
      scratch[warp * kScratchWarpStride + (head + half * 8) * 2] = denominator;
    }
  }
  cutlass::arch::NamedBarrier::sync(
      kComputeThreads, cutlass::arch::ReservedNamedBarriers::StreamkBarrier1);
#pragma unroll
  for (int half = 0; half < 2; ++half) {
    float denominator = 0.0F;
    if (lane_in_head == 0) {
#pragma unroll
      for (int source_warp = 0; source_warp < 4; ++source_warp) {
        denominator += scratch[source_warp * kScratchWarpStride +
                               (head + half * 8) * 2];
      }
    }
    denominator = __shfl_sync(kFullWarpMask, denominator, head * 4);
    inverse[half] = reciprocal_approx_ftz(fmaxf(denominator, 1.0F));
    if (warp == 0 && lane_in_head == 0) {
      lse_staging[head + half * 8] = denominator == 0.0F
          ? -CUDART_INF_F
          : maximum[half] * kAttentionLog2Scale + log2_approx(denominator);
    }
  }
#pragma unroll
  for (int tile = 0; tile < 2; ++tile) {
    Hmma::AFragment fragment;
#pragma unroll
    for (int half = 0; half < 2; ++half) {
      fragment.element[half * 2] =
          float_to_internal(probability[half][tile * 4]);
      fragment.element[half * 2 + 1] =
          float_to_internal(probability[half][tile * 4 + 1]);
      fragment.element[half * 2 + 4] =
          float_to_internal(probability[half][tile * 4 + 2]);
      fragment.element[half * 2 + 5] =
          float_to_internal(probability[half][tile * 4 + 3]);
    }
    Hmma::store_a_shared(
        fragment, query_fragment_row_address(dynamic_storage,
            kAttentionProbabilityOffset, warp * 32 + tile * 16, lane));
  }
  cutlass::arch::NamedBarrier::sync(
      kComputeThreads, cutlass::arch::ReservedNamedBarriers::StreamkBarrier1);
}
#endif


// MTP314 uses transposed QK/PV; TP4 groups the same eight-head tile twice.
template <class Config>
__device__ __forceinline__ void stage_attention_query_and_pe(Glm5MegaMlaArgs const& args, int token,
                                                             int valid_count,
                                                             std::int32_t const* selected,
                                                             InternalActivation const* q_absorb,
                                                             InternalActivation const* proj_q,
                                                             std::uint8_t* dynamic_storage) {
  if constexpr (Config::kKvPerPass == 128) {
    int const tid = static_cast<int>(threadIdx.x);
    if (tid < kQueryHeads * (kRopeDim / 8)) {
      int const head = tid >> 3;
      int const column = (tid & 7) * 8;
      copy_16b_g2s(
          proj_q + token * kProjQRows + kQueryHeads * kQkNopeDim + head * kRopeDim + column,
          attention_swizzled_matrix_address(dynamic_storage + kAttentionQueryPeOffset, head,
                                            column));
    }
#pragma unroll
    for (int half = 0; half < 2; ++half) {
      int const key = half * 64 + (tid >> 2);
      int const column = (tid & 3) * 16;
      auto const* source = static_cast<std::uint8_t const*>(args.pe_cache) +
                           static_cast<std::size_t>(selected[key]) * args.pe_cache_row_stride +
                           column;
      copy_16b_g2s_zfill(source,
                         attention_swizzled_fp8_address(
                             dynamic_storage + kAttentionKeyPeOffset + (key >> 4) * 0x400, key & 15,
                             column, kRopeDim),
                         key < valid_count);
    }
    cutlass::arch::cp_async_fence();
#pragma unroll
    for (int block = 0; block < 4; ++block) {
      int const head = tid >> 4;
      int const column = block * 128 + (tid & 15) * 8;
      auto const* source =
          q_absorb + (static_cast<std::size_t>(token) * kQueryHeads + head) * kKvLatentDim + column;
      copy_16b_g2s(source,
                   attention_swizzled_matrix_address(
                       dynamic_storage + kAttentionQueryLatentOffset + (column >> 6) * 0x800, head,
                       column & 63));
    }
    cutlass::arch::cp_async_fence();
  } else {
    int const tid = static_cast<int>(threadIdx.x);
    if (tid >= kAttentionQkThreads) {
      return;
    }

    if (tid < kQueryHeads * 8) {
      int const head = tid >> 3;
      int const column = (tid & 7) * 8;
      auto const* source =
          proj_q + token * kProjQRows + kQueryHeads * kQkNopeDim + head * kRopeDim + column;
      auto* destination = attention_swizzled_matrix_address(
          dynamic_storage + kAttentionQueryPeOffset, head, column);
      copy_16b_g2s(source, destination);
    }

    if constexpr (kFp8KvCache) {
      if (tid < 128) {
#pragma unroll
        for (int row_half = 0; row_half < 2; ++row_half) {
          int const local_key = row_half * 32 + (tid >> 2);
          int const column = (tid & 3) * 16;
          bool const valid = local_key < valid_count;
          auto const* pe_cache = static_cast<std::uint8_t const*>(args.pe_cache);
          auto const* source =
              pe_cache + static_cast<std::size_t>(selected[local_key]) * args.pe_cache_row_stride +
              column;
          auto* destination = attention_swizzled_fp8_address(
              dynamic_storage + kAttentionKeyPeOffset + (local_key >> 4) * 0x400, local_key & 15,
              column, kRopeDim);
          copy_16b_g2s_zfill(source, destination, valid);
        }
      }
    } else {
#pragma unroll
      for (int row_block = 0; row_block < 4; ++row_block) {
        int const local_key = row_block * 16 + (tid >> 3);
        int const column = (tid & 7) * 8;
        bool const valid = local_key < valid_count;
        auto const* pe_cache = static_cast<__nv_bfloat16 const*>(args.pe_cache);
        auto const* source =
            pe_cache + static_cast<std::size_t>(selected[local_key]) * args.pe_cache_row_stride +
            column;
        auto* destination = attention_swizzled_matrix_address(
            dynamic_storage + kAttentionKeyPeOffset + row_block * 0x800, local_key & 15, column);
        copy_16b_g2s_zfill(source, destination, valid);
      }
    }
    // Commit q_pe/k_pe first so PE HMMA can overlap the following q_abs copies.
    cutlass::arch::cp_async_fence();

#pragma unroll
    for (int column_block = 0; column_block < 4; ++column_block) {
      int const head = tid >> 4;
      int const column = column_block * 128 + (tid & 15) * 8;
      int const chunk = column >> 6;
      int const column_in_chunk = column & 63;
      auto const* source =
          q_absorb + (static_cast<std::size_t>(token) * kQueryHeads + head) * kKvLatentDim + column;
      auto* destination = attention_swizzled_matrix_address(
          dynamic_storage + kAttentionQueryLatentOffset + chunk * 0x800, head, column_in_chunk);
      copy_16b_g2s(source, destination);
    }
    cutlass::arch::cp_async_fence();
  }
}

template <class Config>
__device__ __forceinline__ std::uint8_t const* key_pe_fragment_row_address(
    std::uint8_t const* dynamic_storage, int key_base, int column_base, int lane) {
  int const address_lane = lane & 15;
  int const matrix = address_lane >> 3;
  int const local_key = key_base + (address_lane & 7);
  int const column = column_base + matrix * 8;
  return attention_swizzled_matrix_address(
      dynamic_storage + kAttentionKeyPeOffset + (local_key >> 4) * 0x800, local_key & 15, column);
}

template <class Config>
__device__ __forceinline__ std::uint8_t const* gathered_kv_fragment_row_address(
    std::uint8_t const* dynamic_storage, int first_kv_task, int local_key, int latent_column) {
  int const half = latent_column >> 8;
  int const column_in_half = latent_column & 255;
  int const chunk = column_in_half >> 6;
  int const column_in_chunk = column_in_half & 63;
  auto const* stage = pipeline_stage(dynamic_storage, first_kv_task + half, kAttentionStageBytes);
  return attention_swizzled_matrix_address(
      stage + static_cast<std::size_t>(chunk) * kGatherChunkBytes, local_key, column_in_chunk);
}

template <class Config>
__device__ __forceinline__ std::uint8_t const* gathered_kv_fp8_address(
    std::uint8_t const* dynamic_storage, int first_kv_task, int local_key, int latent_column) {
  if constexpr (Config::kKvPerPass == 128) {
    int const chunk = latent_column >> 7;
    auto const* payload = pipeline_stage(
        dynamic_storage, first_kv_task + chunk / kGatherChunksPerKvStage, kAttentionStageBytes);
    return attention_swizzled_fp8_address(
        payload + (chunk % kGatherChunksPerKvStage) * kGatherChunkBytes, local_key,
        latent_column & 127, kGatherRowBytes);
  } else {
    int const chunk = latent_column >> 7;
    int const column_in_chunk = latent_column & 127;
    auto const* stage = pipeline_stage(dynamic_storage, first_kv_task, kAttentionStageBytes);
    return attention_swizzled_fp8_address(
        stage + static_cast<std::size_t>(chunk) * kGatherChunkBytes, local_key, column_in_chunk,
        kGatherRowBytes);
  }
}

template <class Config>
__device__ __forceinline__ void load_key_pe_attention_fragment(std::uint8_t const* dynamic_storage,
                                                               int key_base, int column_base,
                                                               int lane,
                                                               Hmma::BFragment& fragment) {
  if constexpr (kFp8KvCache) {
    int const key = key_base + (lane >> 2);
    int const column = column_base + 2 * (lane & 3);
    load_fp8_col_major_b_fragment(key_pe_fp8_address(dynamic_storage, key, column),
                                  key_pe_fp8_address(dynamic_storage, key, column + 8), fragment);
  } else {
    Hmma::load_b_shared(fragment, key_pe_fragment_row_address<Config>(dynamic_storage, key_base,
                                                                      column_base, lane));
  }
}

template <class Config>
__device__ __forceinline__ void load_key_latent_attention_fragment(
    std::uint8_t const* dynamic_storage, int key_base, int column_base, int first_kv_task, int lane,
    Hmma::BFragment& fragment) {
  if constexpr (kFp8KvCache) {
    int const key = key_base + (lane >> 2);
    int const column = column_base + 2 * (lane & 3);
    load_fp8_col_major_b_fragment(
        gathered_kv_fp8_address<Config>(dynamic_storage, first_kv_task, key, column),
        gathered_kv_fp8_address<Config>(dynamic_storage, first_kv_task, key, column + 8), fragment);
  } else {
    int const address_lane = lane & 15;
    int const matrix = address_lane >> 3;
    int const local_key = key_base + (address_lane & 7);
    int const latent_column = column_base + matrix * 8;
    Hmma::load_b_shared(fragment, gathered_kv_fragment_row_address<Config>(
                                      dynamic_storage, first_kv_task, local_key, latent_column));
  }
}

template <class Config>
__device__ __forceinline__ void load_value_attention_fragment(std::uint8_t const* dynamic_storage,
                                                              int key_base, int value_column,
                                                              int first_kv_task, int lane,
                                                              Hmma::BFragment& fragment) {
  if constexpr (kFp8KvCache) {
    int const value = value_column + (lane >> 2);
    int const key0 = key_base + 2 * (lane & 3);
    int const key8 = key0 + 8;
    std::uint16_t const pair0 = static_cast<std::uint16_t>(
        *gathered_kv_fp8_address<Config>(dynamic_storage, first_kv_task, key0, value) |
        (*gathered_kv_fp8_address<Config>(dynamic_storage, first_kv_task, key0 + 1, value) << 8));
    std::uint16_t const pair1 = static_cast<std::uint16_t>(
        *gathered_kv_fp8_address<Config>(dynamic_storage, first_kv_task, key8, value) |
        (*gathered_kv_fp8_address<Config>(dynamic_storage, first_kv_task, key8 + 1, value) << 8));
    auto* words = reinterpret_cast<std::uint32_t*>(fragment.element);
    words[0] = e4m3x2_to_mma_x2(pair0);
    words[1] = e4m3x2_to_mma_x2(pair1);
  } else {
    int const address_lane = lane & 15;
    int const matrix = address_lane >> 3;
    int const local_key = key_base + matrix * 8 + (address_lane & 7);
    Hmma::load_b_shared_transposed(
        fragment, gathered_kv_fragment_row_address<Config>(dynamic_storage, first_kv_task,
                                                           local_key, value_column));
  }
}

template <class Config>
__device__ __forceinline__ void load_value_attention_fragment_pair_fp8(
    std::uint8_t const* dynamic_storage, int key_base, int value_column, int first_kv_task,
    int lane, Hmma::BFragment& low_fragment, Hmma::BFragment& high_fragment) {
  int const address_row = (lane & 1) + ((lane >> 1) & 1) * 8 + ((lane >> 2) & 3) * 2;
  auto const* address = gathered_kv_fp8_address<Config>(dynamic_storage, first_kv_task,
                                                        key_base + address_row, value_column);
  decode_fp8_m16n16_b_fragments(address, low_fragment, high_fragment);
}

template <class Config>
__device__ __forceinline__ void issue_selected_kv_gather4(Glm5MegaMlaArgs const& args,
                                                          std::int32_t const* selected,
                                                          std::uint8_t* dynamic_storage,
                                                          int first_kv_task, std::uint64_t* ready) {
  int const tid = static_cast<int>(threadIdx.x);
  int const warp = tid >> 5;
  int const lane = tid & 31;

  if (lane == 0) {
    int const producer_warp = warp - kAttentionProducerWarpBegin;
#pragma unroll
    for (int row_group = 0; row_group < kGatherRowGroupsPerProducer; ++row_group) {
      int const first_row =
          row_group * (kProducerWarps * kGatherRows) + producer_warp * kGatherRows;
      int4 const rows = *reinterpret_cast<int4 const*>(selected + first_row);
#pragma unroll
      for (int chunk = 0; chunk < kGatherChunks; ++chunk) {
        int const half = chunk / kGatherChunksPerKvStage;
        int const chunk_in_stage = chunk % kGatherChunksPerKvStage;
        auto* stage = pipeline_stage(dynamic_storage, first_kv_task + half, kAttentionStageBytes);
        auto* destination = stage + row_group * kGatherRowGroupBytes +
                            producer_warp * kGatherInstructionBytes +
                            chunk_in_stage * kGatherChunkBytes;
        cute::SM100_TMA_LOAD_2D_GATHER4::copy(
            &args.kv_cache_tma, ready,
            static_cast<std::uint64_t>(cute::TMA::CacheHintSm100::EVICT_NORMAL), destination,
            chunk * kGatherColumns, rows.x, rows.y, rows.z, rows.w);
      }
    }
  }
}

template <class Config>
  requires(Config::kTokens == 4)
__device__ __forceinline__ void run_attention_producer(Glm5MegaMlaArgs const& args, int cta,
                                                       PipelineControl* pipeline,
                                                       std::uint8_t* dynamic_storage,
                                                       int task_base) {
  static_assert(Config::kAttentionPasses == 1 && Config::kAttentionCtasPerToken == 32 &&
                Config::kKvPerPass == 64);
  int const tid = static_cast<int>(threadIdx.x);
  int const producer_thread = tid - kConsumerThreads;
  int const token = cta >> 5;
  int const split = cta & 31;
  int const selected_base = split * 64;
  int const selected_count = clamp_int(args.selected_counts[token], 0, kSelectedCapacity);
  int const valid_count = clamp_int(selected_count - selected_base, 0, 64);
  auto* selected = reinterpret_cast<std::int32_t*>(dynamic_storage + kAttentionSelectedOffset);

  // task+0: producer stages the 64 selected ids, then four producer-warp
  // leaders publish ready.
  producer_wait_stage(pipeline, task_base);
  if (producer_thread < 64) {
    selected[producer_thread] =
        producer_thread < valid_count
            ? args.selected_indices[token * kSelectedCapacity + selected_base + producer_thread]
            : 0;
  }
  producer_group_sync();
  producer_arrive_stage(pipeline, task_base);

  // task+1 owns the KV payload: two BF16 stages or one compact E4M3 stage.  It
  // has no transaction of its own; task+2 carries the combined completion.
  int const first_kv_task = task_base + 1;
  producer_wait_stage(pipeline, first_kv_task);
  producer_arrive_stage(pipeline, first_kv_task);

  int const transaction_task = task_base + 2;
  int const transaction_stage = transaction_task & (kPipelineStages - 1);
  producer_wait_stage(pipeline, transaction_task);
  if (tid == kConsumerThreads) {
    producer_expect_transaction(pipeline->ready + transaction_stage,
                                kAttentionGatherTransactionBytes);
  }
  producer_arrive_stage(pipeline, transaction_task);
  producer_group_sync();
  issue_selected_kv_gather4<Config>(args, selected, dynamic_storage, first_kv_task,
                                    pipeline->ready + transaction_stage);
}

template <class Config>
  requires(Config::kTokens == 8)
__device__ __forceinline__ void run_attention_producer(Glm5MegaMlaArgs const& args, int cta,
                                                       PipelineControl* pipeline,
                                                       std::uint8_t* dynamic_storage,
                                                       int task_base) {
  int const tid = static_cast<int>(threadIdx.x);
  int const producer_thread = tid - kConsumerThreads;
  int const token = cta / Config::kAttentionCtasPerToken;
  int const selected_count = clamp_int(args.selected_counts[token], 0, kSelectedCapacity);
  auto* selected = reinterpret_cast<std::int32_t*>(dynamic_storage + kAttentionSelectedOffset);
#pragma unroll 1
  for (int pass = 0; pass < Config::kAttentionPasses; ++pass) {
    int const pass_task = task_base + 3 * pass;
    int const split = (cta % Config::kAttentionCtasPerToken) * Config::kAttentionPasses + pass;
    int const selected_base = split * Config::kKvPerPass;
    int const valid_count = clamp_int(selected_count - selected_base, 0, Config::kKvPerPass);
    if (pass != 0) {
      // The selected-id scratch is reused after the prior query/PE stage.
      producer_wait_stage(pipeline, pass_task + 1);
    }
    // task+0 stages 64 selected ids before producer leaders publish ready.
    producer_wait_stage(pipeline, pass_task);
    if (producer_thread < Config::kKvPerPass) {
      selected[producer_thread] =
          producer_thread < valid_count
              ? args.selected_indices[token * kSelectedCapacity + selected_base + producer_thread]
              : 0;
    }
    producer_group_sync();
    producer_arrive_stage(pipeline, pass_task);

    // task+1 owns one compact E4M3 KV stage; task+2 carries its transaction
    // completion. Both task slots remain part of the four-stage ring.
    int const first_kv_task = pass_task + 1;
    producer_wait_stage(pipeline, first_kv_task);
    producer_arrive_stage(pipeline, first_kv_task);

    int const transaction_task = pass_task + 2;
    int const transaction_stage = transaction_task & (kPipelineStages - 1);
    producer_wait_stage(pipeline, transaction_task);
    if (tid == kConsumerThreads) {
      producer_expect_transaction(pipeline->ready + transaction_stage,
                                  kAttentionGatherTransactionBytes);
    }
    producer_arrive_stage(pipeline, transaction_task);
    producer_group_sync();
    issue_selected_kv_gather4<Config>(args, selected, dynamic_storage, first_kv_task,
                                      pipeline->ready + transaction_stage);
  }
}

template <class Config>
__device__ __forceinline__ void load_attention_head_b_fragment(std::uint8_t const* dynamic_storage,
                                                               int scratch_offset, int column_base,
                                                               int lane, Hmma::BFragment& fragment,
                                                               int head_base = 0) {
  int const head = head_base + (lane & 7);
  int const column = column_base + ((lane >> 3) & 1) * 8;
  auto const* address = attention_swizzled_matrix_address(
      dynamic_storage + scratch_offset + (column >> 6) * 0x800, head, column & 63);
  Hmma::template load_b_shared<false>(fragment, address);
}

template <class Config>
__device__ __forceinline__ void load_key_pe_row_a_fragment(std::uint8_t const* dynamic_storage,
                                                           int key_base, int column_base, int lane,
                                                           Hmma::AFragment& fragment) {
  int const key = key_base + (lane >> 2);
  int const column = column_base + 2 * (lane & 3);
  auto* words = reinterpret_cast<std::uint32_t*>(fragment.element);
  words[0] = e4m3x2_to_mma_x2(
      *reinterpret_cast<std::uint16_t const*>(key_pe_fp8_address(dynamic_storage, key, column)));
  words[1] = e4m3x2_to_mma_x2(*reinterpret_cast<std::uint16_t const*>(
      key_pe_fp8_address(dynamic_storage, key + 8, column)));
  words[2] = e4m3x2_to_mma_x2(*reinterpret_cast<std::uint16_t const*>(
      key_pe_fp8_address(dynamic_storage, key, column + 8)));
  words[3] = e4m3x2_to_mma_x2(*reinterpret_cast<std::uint16_t const*>(
      key_pe_fp8_address(dynamic_storage, key + 8, column + 8)));
}

template <class Config>
__device__ __forceinline__ void load_key_latent_row_a_fragment(std::uint8_t const* dynamic_storage,
                                                               int key_base, int column_base,
                                                               int first_kv_task, int lane,
                                                               Hmma::AFragment& fragment) {
  int const key = key_base + (lane >> 2);
  int const column = column_base + 2 * (lane & 3);
  auto* words = reinterpret_cast<std::uint32_t*>(fragment.element);
  words[0] = e4m3x2_to_mma_x2(*reinterpret_cast<std::uint16_t const*>(
      gathered_kv_fp8_address<Config>(dynamic_storage, first_kv_task, key, column)));
  words[1] = e4m3x2_to_mma_x2(*reinterpret_cast<std::uint16_t const*>(
      gathered_kv_fp8_address<Config>(dynamic_storage, first_kv_task, key + 8, column)));
  words[2] = e4m3x2_to_mma_x2(*reinterpret_cast<std::uint16_t const*>(
      gathered_kv_fp8_address<Config>(dynamic_storage, first_kv_task, key, column + 8)));
  words[3] = e4m3x2_to_mma_x2(*reinterpret_cast<std::uint16_t const*>(
      gathered_kv_fp8_address<Config>(dynamic_storage, first_kv_task, key + 8, column + 8)));
}

template <class Config>
__device__ __forceinline__ void restore_attention_head_scores(Hmma::Accumulator const& transposed,
                                                              int lane, Hmma::Accumulator& low,
                                                              Hmma::Accumulator& high) {
  int const head = lane >> 2;
  int const source0 = 8 * (lane & 3) + (head >> 1);
  int const source1 = source0 + 4;
  bool const odd_head = (head & 1) != 0;
  float const l00 = __shfl_sync(kFullWarpMask, transposed.element[0], source0);
  float const l01 = __shfl_sync(kFullWarpMask, transposed.element[1], source0);
  float const l10 = __shfl_sync(kFullWarpMask, transposed.element[0], source1);
  float const l11 = __shfl_sync(kFullWarpMask, transposed.element[1], source1);
  float const h00 = __shfl_sync(kFullWarpMask, transposed.element[2], source0);
  float const h01 = __shfl_sync(kFullWarpMask, transposed.element[3], source0);
  float const h10 = __shfl_sync(kFullWarpMask, transposed.element[2], source1);
  float const h11 = __shfl_sync(kFullWarpMask, transposed.element[3], source1);
  low.element[0] = odd_head ? l01 : l00;
  low.element[1] = odd_head ? l11 : l10;
  high.element[0] = odd_head ? h01 : h00;
  high.element[1] = odd_head ? h11 : h10;
}


__device__ __forceinline__ void run_full_head_attention_softmax_128(
    Hmma::Accumulator const (&scores)[4], int warp, int lane, int valid_count,
    std::uint8_t* dynamic_storage, float* lse_staging, float (&inverse)[2]);

template <class Config>
__device__ __forceinline__ void run_transposed_attention_consumer(Glm5MegaMlaArgs const& args,
                                                                  int cta,
                                                                  PipelineControl* pipeline,
                                                                  std::uint8_t* dynamic_storage,
                                                                  int task_base) {
  int const tid = static_cast<int>(threadIdx.x);
  int const warp = tid >> 5;
  int const head_group = Config::kTpSize == 8 ? 0 : warp >> 2;
  int const group_warp = Config::kTpSize == 8 ? warp : warp & 3;
  int const lane = tid & 31;
  int const token = cta >> 5;
  int const split = cta & 31;
  int const selected_base = split * 64;
  int const selected_count = clamp_int(args.selected_counts[token], 0, kSelectedCapacity);
  int const valid_count = clamp_int(selected_count - selected_base, 0, 64);

  auto const* q_absorb =
      workspace_pointer_const<InternalActivation>(args, WorkspaceLayout::kQAbsorb);
  auto const* proj_q = workspace_pointer_const<InternalActivation>(args, WorkspaceLayout::kProjQ);
  auto* partial_o = workspace_pointer<InternalActivation>(args, WorkspaceLayout::kPartialO) +
                    static_cast<std::size_t>(cta) * kQueryHeads * kValueDim;
  auto* partial_lse =
      workspace_pointer<float>(args, WorkspaceLayout::kPartialLse) + cta * kQueryHeads;
  auto* partial_o_staging =
      reinterpret_cast<InternalActivation*>(dynamic_storage + kAttentionPartialOStagingOffset);
  auto* partial_lse_staging =
      reinterpret_cast<float*>(dynamic_storage + kAttentionLseStagingOffset);

  auto* selected = reinterpret_cast<std::int32_t*>(dynamic_storage + kAttentionSelectedOffset);
  consumer_wait_stage(pipeline, task_base);
  stage_attention_query_and_pe<Config>(args, token, valid_count, selected, q_absorb, proj_q,
                                       dynamic_storage);

  // Synchronize consumers before waiting on the PE copy group. Keep the
  // absorbed-query copies in flight while computing the PE term.
  consumer_group_sync();
  consumer_release_stage(pipeline, task_base);
  if (tid < kAttentionQkThreads) {
    wait_cp_async_group_1();
    cutlass::arch::NamedBarrier::sync(kAttentionQkThreads,
                                      cutlass::arch::ReservedNamedBarriers::StreamkBarrier1);
  }

  if constexpr (kFp8KvCache) {
    // Compute the PE term while latent KV is being gathered.
    // Wait for the query and KV inputs before computing the latent term.
    float attention_denominator_inverse = 0.0F;
    if (warp < kAttentionQkWarps) {
      Hmma::Accumulator score_transposed;
      Hmma::clear(score_transposed);
#pragma unroll
      for (int k_tile = 0; k_tile < kRopeDim / 16; ++k_tile) {
        Hmma::AFragment k_fragment;
        Hmma::BFragment q_fragment;
        load_key_pe_row_a_fragment<Config>(dynamic_storage, group_warp * 16, k_tile * 16, lane,
                                           k_fragment);
        load_attention_head_b_fragment<Config>(dynamic_storage, kAttentionQueryPeOffset,
                                               k_tile * 16, lane, q_fragment, head_group * 8);
        Hmma::fma(score_transposed, k_fragment, q_fragment);
      }

      // Wait for query copies and gathered KV before the latent-attention stage.
      wait_cp_async_group_0();
      consumer_wait_stage(pipeline, task_base + 1);
      consumer_wait_stage(pipeline, task_base + 2);
      cutlass::arch::NamedBarrier::sync(kAttentionQkThreads,
                                        cutlass::arch::ReservedNamedBarriers::StreamkBarrier1);

#pragma unroll 8
      for (int k_tile = 0; k_tile < kKvLatentDim / 16; ++k_tile) {
        Hmma::AFragment k_fragment;
        Hmma::BFragment q_fragment;
        load_key_latent_row_a_fragment<Config>(dynamic_storage, group_warp * 16, k_tile * 16,
                                               task_base + 1, lane, k_fragment);
        load_attention_head_b_fragment<Config>(dynamic_storage, kAttentionQueryLatentOffset,
                                               k_tile * 16, lane, q_fragment, head_group * 8);
        Hmma::fma(score_transposed, k_fragment, q_fragment);
      }

      Hmma::Accumulator qk[2];
      restore_attention_head_scores<Config>(score_transposed, lane, qk[0], qk[1]);
      attention_denominator_inverse = run_warp_distributed_attention_softmax(
          qk[0], qk[1], warp, lane, valid_count, dynamic_storage, partial_lse_staging);
    }

    // V^T[16,16] * P^T[16,8] fills all M16N8 outputs: eight heads.
    if (warp < kAttentionQkWarps) {
      Hmma::Accumulator pv_transposed[8];
#pragma unroll
      for (int pair_tile = 0; pair_tile < 8; ++pair_tile) {
        Hmma::clear(pv_transposed[pair_tile]);
      }
#pragma unroll 1
      for (int k_tile = 0; k_tile < 4; ++k_tile) {
        Hmma::BFragment probability_fragment;
        load_attention_head_b_fragment<Config>(dynamic_storage, kAttentionProbabilityOffset,
                                               k_tile * 16, lane, probability_fragment,
                                               head_group * 8);
#pragma unroll
        for (int pair_tile = 0; pair_tile < 8; ++pair_tile) {
          Hmma::BFragment value_low;
          Hmma::BFragment value_high;
          int const value_column = pair_tile * 64 + group_warp * 16;
          load_value_attention_fragment_pair_fp8<Config>(dynamic_storage, k_tile * 16, value_column,
                                                         task_base + 1, lane, value_low,
                                                         value_high);
          auto const* low = reinterpret_cast<std::uint32_t const*>(value_low.element);
          auto const* high = reinterpret_cast<std::uint32_t const*>(value_high.element);
          Hmma::AFragment value_fragment;
          auto* words = reinterpret_cast<std::uint32_t*>(value_fragment.element);
          words[0] = low[0];
          words[1] = high[0];
          words[2] = low[1];
          words[3] = high[1];
          Hmma::fma(pv_transposed[pair_tile], value_fragment, probability_fragment);
        }
      }
      float const inverse_even =
          __shfl_sync(kFullWarpMask, attention_denominator_inverse, 8 * (lane & 3));
      float const inverse_odd =
          __shfl_sync(kFullWarpMask, attention_denominator_inverse, 8 * (lane & 3) + 4);
#pragma unroll
      for (int pair_tile = 0; pair_tile < 8; ++pair_tile) {
        Hmma::Accumulator const& value = pv_transposed[pair_tile];
        std::uint32_t const first =
            pack_internal_x2(value.element[0] * inverse_even, value.element[1] * inverse_odd);
        std::uint32_t const second =
            pack_internal_x2(value.element[2] * inverse_even, value.element[3] * inverse_odd);
        int const column = pair_tile * 64 + group_warp * 16 + ((lane >> 3) & 1) * 8;
        auto* destination = partial_o_staging + (head_group * 8 + (lane & 7)) * kValueDim + column;
        std::uint32_t const shared_address =
            static_cast<std::uint32_t>(__cvta_generic_to_shared(destination));
        asm volatile("stmatrix.sync.aligned.m8n8.x2.trans.shared.b16 [%0], {%1, %2};"
                     :
                     : "r"(shared_address), "r"(first), "r"(second)
                     : "memory");
      }

      // Publish internal O and FP32 LSE to the async proxy. The caller
      // appends the signed partial arrival before committing the bulk group.
      cutlass::arch::NamedBarrier::sync(kAttentionQkThreads,
                                        cutlass::arch::ReservedNamedBarriers::StreamkBarrier1);
      __threadfence_block();
      publish_shared_writes_to_async_proxy();
      if (tid == 0) {
        copy_bulk_shared_to_global(partial_o, partial_o_staging, kAttentionPartialOBytes);
        copy_bulk_shared_to_global(partial_lse, partial_lse_staging, kAttentionPartialLseBytes);
      }
    }
  } else {
    // Compute the PE term while latent KV is being gathered.
    // Wait for the query and KV inputs before computing the latent term.
    float attention_denominator_inverse = 0.0F;
    if (warp < 4) {
      Hmma::Accumulator qk[2];
      Hmma::clear(qk[0]);
      Hmma::clear(qk[1]);

#pragma unroll
      for (int k_tile = 0; k_tile < kRopeDim / 16; ++k_tile) {
        Hmma::AFragment q_fragment;
        load_query_attention_fragment(dynamic_storage, kAttentionQueryPeOffset, k_tile * 16, lane,
                                      q_fragment);

#pragma unroll
        for (int group = 0; group < 2; ++group) {
          Hmma::BFragment k_fragment;
          load_key_pe_attention_fragment<Config>(dynamic_storage, warp * 16 + group * 8,
                                                 k_tile * 16, lane, k_fragment);
          Hmma::fma(qk[group], q_fragment, k_fragment);
        }
      }

      // q_abs copies overlap the PE HMMA prefix and must complete before
      // the latent-query matrix loads.
      wait_cp_async_group_0();
      // Wait for the complete KV gather before consuming the tile.
      consumer_wait_stage(pipeline, task_base + 1);
      consumer_wait_stage(pipeline, task_base + 2);
      cutlass::arch::NamedBarrier::sync(kAttentionQkThreads,
                                        cutlass::arch::ReservedNamedBarriers::StreamkBarrier1);

#pragma unroll
      for (int k_tile = 0; k_tile < kKvLatentDim / 16; ++k_tile) {
        Hmma::AFragment q_fragment;
        load_query_attention_fragment(dynamic_storage, kAttentionQueryLatentOffset, k_tile * 16,
                                      lane, q_fragment);

#pragma unroll
        for (int group = 0; group < 2; ++group) {
          Hmma::BFragment k_fragment;
          load_key_latent_attention_fragment<Config>(dynamic_storage, warp * 16 + group * 8,
                                                     k_tile * 16, task_base + 1, lane, k_fragment);
          Hmma::fma(qk[group], q_fragment, k_fragment);
        }
      }

      attention_denominator_inverse = run_warp_distributed_attention_softmax(
          qk[0], qk[1], warp, lane, valid_count, dynamic_storage, partial_lse_staging);
    }

    // Load value fragments according to the cache format.
    if (warp < 4) {
      Hmma::Accumulator pv[16];
#pragma unroll
      for (int local_n_tile = 0; local_n_tile < 16; ++local_n_tile) {
        Hmma::clear(pv[local_n_tile]);
      }

#pragma unroll
      for (int k_tile = 0; k_tile < 4; ++k_tile) {
        Hmma::AFragment probability_fragment;
        load_query_attention_fragment(dynamic_storage, kAttentionProbabilityOffset, k_tile * 16,
                                      lane, probability_fragment);
        if constexpr (kFp8KvCache) {
#pragma unroll
          for (int pair_tile = 0; pair_tile < 8; ++pair_tile) {
            Hmma::BFragment value_low;
            Hmma::BFragment value_high;
            int const value_column = pair_tile * 64 + warp * 16;
            load_value_attention_fragment_pair_fp8<Config>(dynamic_storage, k_tile * 16,
                                                           value_column, task_base + 1, lane,
                                                           value_low, value_high);
            Hmma::fma(pv[pair_tile * 2], probability_fragment, value_low);
            Hmma::fma(pv[pair_tile * 2 + 1], probability_fragment, value_high);
          }
        } else {
#pragma unroll
          for (int local_n_tile = 0; local_n_tile < 16; ++local_n_tile) {
            Hmma::BFragment value_fragment;
            int const value_column = local_n_tile * 32 + warp * 8;
            load_value_attention_fragment<Config>(dynamic_storage, k_tile * 16, value_column,
                                                  task_base + 1, lane, value_fragment);
            Hmma::fma(pv[local_n_tile], probability_fragment, value_fragment);
          }
        }
      }

      // Normalize all accumulators before any lane stages output in shared memory.
#pragma unroll
      for (int local_n_tile = 0; local_n_tile < 16; ++local_n_tile) {
        for (int i = 0; i < 2; ++i) {
          pv[local_n_tile].element[i] *= attention_denominator_inverse;
        }
      }
      cutlass::arch::NamedBarrier::sync(kAttentionQkThreads,
                                        cutlass::arch::ReservedNamedBarriers::StreamkBarrier1);

#pragma unroll
      for (int local_n_tile = 0; local_n_tile < 16; ++local_n_tile) {
        int const head = lane >> 2;
        int const value_column =
            (kFp8KvCache ? (local_n_tile >> 1) * 64 + warp * 16 + (local_n_tile & 1) * 8
                         : local_n_tile * 32 + warp * 8) +
            2 * (lane & 3);
        *reinterpret_cast<std::uint32_t*>(partial_o_staging + head * kValueDim + value_column) =
            pack_internal_x2(pv[local_n_tile].element[0], pv[local_n_tile].element[1]);
      }

      // Publish output and LSE tiles to the async proxy.
      // The caller adds the completion signal before committing the copies.
      cutlass::arch::NamedBarrier::sync(kAttentionQkThreads,
                                        cutlass::arch::ReservedNamedBarriers::StreamkBarrier1);
      __threadfence_block();
      publish_shared_writes_to_async_proxy();
      if (tid == 0) {
        copy_bulk_shared_to_global(partial_o, partial_o_staging, kAttentionPartialOBytes);
        copy_bulk_shared_to_global(partial_lse, partial_lse_staging, kAttentionPartialLseBytes);
      }
    }
  }
}

template <class Config>
__device__ __forceinline__ void run_head_major_attention_128(Glm5MegaMlaArgs const& args, int cta,
                                                             PipelineControl* pipeline,
                                                             std::uint8_t* dynamic_storage,
                                                             int task_base) {
  int const tid = static_cast<int>(threadIdx.x);
  int const warp = tid >> 5;
  int const lane = tid & 31;
  int const token = cta / kAttentionCtasPerToken;
  int const selected_base = (cta % kAttentionCtasPerToken) * 128;
  int const count = clamp_int(args.selected_counts[token], 0, kSelectedCapacity);
  int const valid_count = clamp_int(count - selected_base, 0, 128);
  auto const* q_absorb =
      workspace_pointer_const<InternalActivation>(args, WorkspaceLayout::kQAbsorb);
  auto const* proj_q = workspace_pointer_const<InternalActivation>(args, WorkspaceLayout::kProjQ);
  auto const* selected =
      reinterpret_cast<std::int32_t const*>(dynamic_storage + kAttentionSelectedOffset);
  auto* partial =
      reinterpret_cast<InternalActivation*>(dynamic_storage + kAttentionPartialOStagingOffset);
  auto* lse = reinterpret_cast<float*>(dynamic_storage + kAttentionLseStagingOffset);
  consumer_wait_stage(pipeline, task_base);
  stage_attention_query_and_pe<Config>(args, token, valid_count, selected, q_absorb, proj_q,
                                       dynamic_storage);
  consumer_group_sync();
  consumer_release_stage(pipeline, task_base);
  wait_cp_async_group_1();
  consumer_group_sync();

  Hmma::Accumulator qk[4];
  if (warp < 4) {
#pragma unroll
    for (int group = 0; group < 4; ++group) {
      Hmma::clear(qk[group]);
    }
#pragma unroll
    for (int tile = 0; tile < kRopeDim / 16; ++tile) {
      Hmma::AFragment query;
      load_query_attention_fragment(dynamic_storage, kAttentionQueryPeOffset, tile * 16, lane,
                                    query);
#pragma unroll
      for (int group = 0; group < 4; ++group) {
        int const key = warp * 32 + group * 8 + (lane >> 2);
        int const column = tile * 16 + 2 * (lane & 3);
        Hmma::BFragment cache;
        load_fp8_col_major_b_fragment(key_pe_fp8_address(dynamic_storage, key, column),
                                      key_pe_fp8_address(dynamic_storage, key, column + 8), cache);
        Hmma::fma(qk[group], query, cache);
      }
    }
  }
  // All eight staging warps finish q_abs before four compute warps read it.
  wait_cp_async_group_0();
  consumer_wait_stage(pipeline, task_base + 1);
  consumer_wait_stage(pipeline, task_base + 2);
  consumer_group_sync();
  if (warp < 4) {
#pragma unroll 4
    for (int tile = 0; tile < kKvLatentDim / 16; ++tile) {
      Hmma::AFragment query;
      load_query_attention_fragment(dynamic_storage, kAttentionQueryLatentOffset, tile * 16, lane,
                                    query);
#pragma unroll
      for (int group = 0; group < 4; ++group) {
        int const key = warp * 32 + group * 8 + (lane >> 2);
        int const column = tile * 16 + 2 * (lane & 3);
        Hmma::BFragment cache;
        load_fp8_col_major_b_fragment(
            gathered_kv_fp8_address<Config>(dynamic_storage, task_base + 1, key, column),
            gathered_kv_fp8_address<Config>(dynamic_storage, task_base + 1, key, column + 8),
            cache);
        Hmma::fma(qk[group], query, cache);
      }
    }
    float inverse[2];
    run_full_head_attention_softmax_128(qk, warp, lane, valid_count, dynamic_storage, lse, inverse);
    Hmma::Accumulator pv[16];
#pragma unroll
    for (int tile = 0; tile < 16; ++tile) {
      Hmma::clear(pv[tile]);
    }
#pragma unroll 2
    for (int key_tile = 0; key_tile < 8; ++key_tile) {
      Hmma::AFragment probability;
      load_query_attention_fragment(dynamic_storage, kAttentionProbabilityOffset, key_tile * 16,
                                    lane, probability);
#pragma unroll
      for (int pair = 0; pair < 8; ++pair) {
        Hmma::BFragment low, high;
        load_value_attention_fragment_pair_fp8<Config>(
            dynamic_storage, key_tile * 16, pair * 64 + warp * 16, task_base + 1, lane, low, high);
        Hmma::fma(pv[pair * 2], probability, low);
        Hmma::fma(pv[pair * 2 + 1], probability, high);
      }
    }
#pragma unroll
    for (int tile = 0; tile < 16; ++tile) {
      int const head = lane >> 2;
      int const column = (tile >> 1) * 64 + warp * 16 + (tile & 1) * 8 + 2 * (lane & 3);
      *reinterpret_cast<std::uint32_t*>(partial + head * kValueDim + column) =
          pack_internal_x2(pv[tile].element[0] * inverse[0], pv[tile].element[1] * inverse[0]);
      *reinterpret_cast<std::uint32_t*>(partial + (head + 8) * kValueDim + column) =
          pack_internal_x2(pv[tile].element[2] * inverse[1], pv[tile].element[3] * inverse[1]);
    }
    cutlass::arch::NamedBarrier::sync(128, cutlass::arch::ReservedNamedBarriers::StreamkBarrier1);
    publish_shared_writes_to_async_proxy();
    if (tid == 0) {
      copy_bulk_shared_to_global(
          workspace_pointer<InternalActivation>(args, WorkspaceLayout::kPartialO) +
              static_cast<std::size_t>(cta) * kQueryHeads * kValueDim,
          partial, kAttentionPartialOBytes);
      copy_bulk_shared_to_global(
          workspace_pointer<float>(args, WorkspaceLayout::kPartialLse) + cta * kQueryHeads, lse,
          kAttentionPartialLseBytes);
    }
  }
  consumer_group_sync();
  if (tid == 0) {
    commit_bulk_copy_group();
    wait_bulk_copy_group();
  }
  consumer_group_sync();
}

template <class Config>
__device__ __forceinline__ void run_head_major_attention_64(Glm5MegaMlaArgs const& args, int cta,
                                                            PipelineControl* pipeline,
                                                            std::uint8_t* dynamic_storage,
                                                            int task_base) {
  int const tid = static_cast<int>(threadIdx.x);
  int const warp = tid >> 5;
  int const lane = tid & 31;
  int const token = cta / kAttentionCtasPerToken;
  int const selected_count = clamp_int(args.selected_counts[token], 0, kSelectedCapacity);
  auto const* q_absorb =
      workspace_pointer_const<InternalActivation>(args, WorkspaceLayout::kQAbsorb);
  auto const* proj_q = workspace_pointer_const<InternalActivation>(args, WorkspaceLayout::kProjQ);
  auto* partial_o_staging =
      reinterpret_cast<InternalActivation*>(dynamic_storage + kAttentionPartialOStagingOffset);
  auto* partial_lse_staging =
      reinterpret_cast<float*>(dynamic_storage + kAttentionLseStagingOffset);
#pragma unroll 1
  for (int pass = 0; pass < kAttentionPasses; ++pass) {
    int const pass_task = task_base + 3 * pass;
    int const split = (cta % kAttentionCtasPerToken) * kAttentionPasses + pass;
    int const selected_base = split * 64;
    int const valid_count = clamp_int(selected_count - selected_base, 0, 64);
    int const partial_index = token * 32 + split;
    auto* partial_o = workspace_pointer<InternalActivation>(args, WorkspaceLayout::kPartialO) +
                      static_cast<std::size_t>(partial_index) * kQueryHeads * kValueDim;
    auto* partial_lse =
        workspace_pointer<float>(args, WorkspaceLayout::kPartialLse) + partial_index * kQueryHeads;
    auto* selected = reinterpret_cast<std::int32_t*>(dynamic_storage + kAttentionSelectedOffset);
    consumer_wait_stage(pipeline, pass_task);
    stage_attention_query_and_pe<Config>(args, token, valid_count, selected, q_absorb, proj_q,
                                         dynamic_storage);

    // Group 0 owns q_pe/k_pe; group 1 owns q_abs, which remains in flight
    // during the PE HMMA prefix.
    consumer_group_sync();
    consumer_release_stage(pipeline, pass_task);
    if (tid < kAttentionQkThreads) {
      wait_cp_async_group_1();
      cutlass::arch::NamedBarrier::sync(kAttentionQkThreads,
                                        cutlass::arch::ReservedNamedBarriers::StreamkBarrier1);
    }

    // Compute the PE term while latent KV is being gathered.
    // Wait for the query and KV inputs before computing the latent term.
    float attention_denominator_inverse = 0.0F;
    if (warp < 4) {
      Hmma::Accumulator qk[2];
      Hmma::clear(qk[0]);
      Hmma::clear(qk[1]);

#pragma unroll
      for (int k_tile = 0; k_tile < kRopeDim / 16; ++k_tile) {
        Hmma::AFragment q_fragment;
        load_query_attention_fragment(dynamic_storage, kAttentionQueryPeOffset, k_tile * 16, lane,
                                      q_fragment);

#pragma unroll
        for (int group = 0; group < 2; ++group) {
          Hmma::BFragment k_fragment;
          load_key_pe_attention_fragment<Config>(dynamic_storage, warp * 16 + group * 8,
                                                 k_tile * 16, lane, k_fragment);
          Hmma::fma(qk[group], q_fragment, k_fragment);
        }
      }

      // q_abs copies overlap the PE HMMA prefix and must complete before
      // the latent-query matrix loads.
      wait_cp_async_group_0();
      // task+2 gates the complete 32 KiB E4M3 GATHER4 transaction.
      consumer_wait_stage(pipeline, pass_task + 1);
      consumer_wait_stage(pipeline, pass_task + 2);
      cutlass::arch::NamedBarrier::sync(kAttentionQkThreads,
                                        cutlass::arch::ReservedNamedBarriers::StreamkBarrier1);

#pragma unroll
      for (int k_tile = 0; k_tile < kKvLatentDim / 16; ++k_tile) {
        Hmma::AFragment q_fragment;
        load_query_attention_fragment(dynamic_storage, kAttentionQueryLatentOffset, k_tile * 16,
                                      lane, q_fragment);

#pragma unroll
        for (int group = 0; group < 2; ++group) {
          Hmma::BFragment k_fragment;
          load_key_latent_attention_fragment<Config>(dynamic_storage, warp * 16 + group * 8,
                                                     k_tile * 16, pass_task + 1, lane, k_fragment);
          Hmma::fma(qk[group], q_fragment, k_fragment);
        }
      }

      attention_denominator_inverse = run_warp_distributed_attention_softmax(
          qk[0], qk[1], warp, lane, valid_count, dynamic_storage, partial_lse_staging);
    }

    // Pair adjacent value tiles to reuse each matrix load.
    if (warp < 4) {
      Hmma::Accumulator pv[16];
#pragma unroll
      for (int local_n_tile = 0; local_n_tile < 16; ++local_n_tile) {
        Hmma::clear(pv[local_n_tile]);
      }

#pragma unroll
      for (int k_tile = 0; k_tile < 4; ++k_tile) {
        Hmma::AFragment probability_fragment;
        load_query_attention_fragment(dynamic_storage, kAttentionProbabilityOffset, k_tile * 16,
                                      lane, probability_fragment);
#pragma unroll
        for (int pair_tile = 0; pair_tile < 8; ++pair_tile) {
          Hmma::BFragment value_low;
          Hmma::BFragment value_high;
          int const value_column = pair_tile * 64 + warp * 16;
          load_value_attention_fragment_pair_fp8<Config>(dynamic_storage, k_tile * 16, value_column,
                                                         pass_task + 1, lane, value_low,
                                                         value_high);
          Hmma::fma(pv[pair_tile * 2], probability_fragment, value_low);
          Hmma::fma(pv[pair_tile * 2 + 1], probability_fragment, value_high);
        }
      }

      // Normalize all 16 accumulators before staging partial O in shared memory.
#pragma unroll
      for (int local_n_tile = 0; local_n_tile < 16; ++local_n_tile) {
        for (int i = 0; i < 2; ++i) {
          pv[local_n_tile].element[i] *= attention_denominator_inverse;
        }
      }
      cutlass::arch::NamedBarrier::sync(kAttentionQkThreads,
                                        cutlass::arch::ReservedNamedBarriers::StreamkBarrier1);

#pragma unroll
      for (int local_n_tile = 0; local_n_tile < 16; ++local_n_tile) {
        int const head = lane >> 2;
        int const value_column =
            ((local_n_tile >> 1) * 64 + warp * 16 + (local_n_tile & 1) * 8) + 2 * (lane & 3);
        *reinterpret_cast<std::uint32_t*>(partial_o_staging + head * kValueDim + value_column) =
            pack_internal_x2(pv[local_n_tile].element[0], pv[local_n_tile].element[1]);
      }

      // Publish output and LSE tiles to the async proxy.
      // Finish the copies before signaling phase completion.
      cutlass::arch::NamedBarrier::sync(kAttentionQkThreads,
                                        cutlass::arch::ReservedNamedBarriers::StreamkBarrier1);
      __threadfence_block();
      publish_shared_writes_to_async_proxy();
      if (tid == 0) {
        copy_bulk_shared_to_global(partial_o, partial_o_staging, kAttentionPartialOBytes);
        copy_bulk_shared_to_global(partial_lse, partial_lse_staging, kAttentionPartialLseBytes);
      }
    }
    if constexpr (kAttentionPasses == 2) {
      // Finish each pass before reusing staging or publishing its partials.
      consumer_group_sync();
      if (tid == 0) {
        commit_bulk_copy_group();
        wait_bulk_copy_group();
      }
      consumer_group_sync();
      if (pass == 0) {
        consumer_release_stage(pipeline, pass_task + 1);
        consumer_release_stage(pipeline, pass_task + 2);
      }
    }
  }
}

template <class Config>
__device__ __forceinline__ void run_attention_consumer(Glm5MegaMlaArgs const& args, int cta,
                                                       PipelineControl* pipeline,
                                                       std::uint8_t* dynamic_storage,
                                                       int task_base) {
  if constexpr (Config::kTokens == 4) {
    run_transposed_attention_consumer<Config>(args, cta, pipeline, dynamic_storage, task_base);
  } else if constexpr (Config::kTpSize == 4) {
    run_head_major_attention_128<Config>(args, cta, pipeline, dynamic_storage, task_base);
  } else {
    run_head_major_attention_64<Config>(args, cta, pipeline, dynamic_storage, task_base);
  }
}

template <class Config>
__device__ __forceinline__ void run_combine(Glm5MegaMlaArgs const& args, int cta,
                                            std::uint8_t* dynamic_storage,
                                            PipelineControl* pipeline, int attention_task) {
  int const tid = static_cast<int>(threadIdx.x);
  int const token = Config::kTokens == 4 ? (cta >> 5) : cta / Config::kAttentionCtasPerToken;
  int const head = cta & (kQueryHeads - 1);
  auto const* partial_o =
      workspace_pointer_const<InternalActivation>(args, WorkspaceLayout::kPartialO);
  auto const* partial_lse = workspace_pointer_const<float>(args, WorkspaceLayout::kPartialLse);
  auto* combined = workspace_pointer<InternalActivation>(args, WorkspaceLayout::kCombined);
  auto* normalized_weights = reinterpret_cast<float*>(dynamic_storage + kCombineWeightsOffset);
  int const selected_count = clamp_int(args.selected_counts[token], 0, 2048);
  int const valid_splits = (selected_count + Config::kKvPerPass - 1) >> (Config::kKvPerPass == 128 ? 7 : 6);

  // Zero-fill invalid splits because the reduction still reads their entries.
  constexpr int kSplitUnroll = Config::kCombineUnroll;
#pragma unroll kSplitUnroll
  for (int split = 0; split < Config::kPartialSplitsPerToken; ++split) {
    if (tid < 64) {
      int const source_cta = token * Config::kPartialSplitsPerToken + split;
      auto const* source = partial_o + (source_cta * kQueryHeads + head) * kValueDim + tid * 8;
      auto* destination = combine_partial_address(dynamic_storage, attention_task, split, tid * 8);
      copy_16b_g2s_zfill(source, destination, split < valid_splits);
    }
  }
  cutlass::arch::cp_async_fence();
  cutlass::arch::cp_async_wait<0>();

  int const warp = tid >> 5;
  int const lane = tid & 31;
  if (warp == 0) {
    bool const valid = lane < valid_splits;
    int const source_cta = token * Config::kPartialSplitsPerToken + lane;
    float const local_lse = valid ? partial_lse[source_cta * kQueryHeads + head] : -FLT_MAX;
    float maximum = local_lse;
#pragma unroll
    for (int mask = 16; mask != 0; mask >>= 1) {
      maximum = fmaxf(maximum, __shfl_xor_sync(kFullWarpMask, maximum, mask));
    }
    if (valid_splits == 0) {
      maximum = 0.0F;
    }

    float probability = valid ? exp2_underflow_safe(local_lse - maximum) : 0.0F;
    float denominator = probability;
#pragma unroll
    for (int mask = 16; mask != 0; mask >>= 1) {
      denominator += __shfl_xor_sync(kFullWarpMask, denominator, mask);
    }
    // Reuse the split probabilities and keep empty-split weights at zero.
    float const inverse_denominator = reciprocal_approx_ftz(fmaxf(denominator, 1.0F));
    float const weight = probability * inverse_denominator;
    normalized_weights[lane] = weight;
  }
  combine_group_sync();

  Hmma::Accumulator accumulated[8];
#pragma unroll
  for (int local_n_tile = 0; local_n_tile < 8; ++local_n_tile) {
    Hmma::clear(accumulated[local_n_tile]);
  }

  Hmma::AFragment weight_fragment;
  load_combine_weight_fragment(normalized_weights, 0, lane, weight_fragment);
#pragma unroll
  for (int local_n_tile = 0; local_n_tile < 8; ++local_n_tile) {
    Hmma::BFragment partial_fragment;
    int const value_column = (warp * 8 + local_n_tile) * 8;
    load_combine_partial_fragment(dynamic_storage, attention_task, 0, value_column, lane,
                                  partial_fragment);
    Hmma::fma(accumulated[local_n_tile], weight_fragment, partial_fragment);
  }
  consumer_release_stage(pipeline, attention_task + 1);

  if constexpr (Config::kPartialSplitsPerToken > 16) {
    load_combine_weight_fragment(normalized_weights, 16, lane, weight_fragment);
#pragma unroll
    for (int local_n_tile = 0; local_n_tile < 8; ++local_n_tile) {
      Hmma::BFragment partial_fragment;
      int const value_column = (warp * 8 + local_n_tile) * 8;
      load_combine_partial_fragment(dynamic_storage, attention_task, 16, value_column, lane,
                                    partial_fragment);
      Hmma::fma(accumulated[local_n_tile], weight_fragment, partial_fragment);
    }
  }

  // All MMA rows contain the same result; store only the first row.
  if (lane < 4) {
#pragma unroll
    for (int local_n_tile = 0; local_n_tile < 8; ++local_n_tile) {
      int const value_column = (warp * 8 + local_n_tile) * 8 + lane * 2;
      *reinterpret_cast<std::uint32_t*>(combined + (token * kQueryHeads + head) * kValueDim +
                                        value_column) =
          pack_internal_x2(accumulated[local_n_tile].element[0],
                           accumulated[local_n_tile].element[1]);
    }
  }
}
}  // namespace glm5::mla::detail
