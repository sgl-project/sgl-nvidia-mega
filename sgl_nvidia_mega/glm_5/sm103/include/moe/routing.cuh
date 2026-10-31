#pragma once

// Shared RMS, router and TopK stages; optimized ownership stays in Config.
#include <moe/config.cuh>

#include <common/norm.cuh>

namespace glm5::moe::detail {
template <class Config>
  requires(Config::kTokens == 4)
__device__ __forceinline__ std::uint32_t load_relaxed_gpu_u32(std::uint32_t const* address) {
  std::uint32_t value;
  asm volatile("ld.global.relaxed.gpu.u32 %0, [%1];" : "=r"(value) : "l"(address) : "memory");
  return value;
}

template <class Config>
  requires(Config::kTokens == 4)
__device__ __forceinline__ std::uint32_t next_generation(std::uint32_t generation) {
  generation += 1U;
  return generation == 0U ? 1U : generation;
}

template <class Config>
  requires(Config::kTokens == 4)
__device__ __forceinline__ void acquire_fence_gpu() {
  asm volatile("fence.acquire.gpu;" : : : "memory");
}

template <class Config>
__device__ __forceinline__ bool bf16_bits_are_nan(std::uint16_t bits) {
  return (bits & 0x7fffU) > 0x7f80U;
}

template <class Config>
  requires(Config::kTokens == 4)
__device__ __forceinline__ std::uint32_t router_rank_key(float rank) {
  if (fp32_is_nan(rank)) {
    rank = kNegativeFloatMax;
  }
  if (rank >= 0.0F) {
    return __float_as_uint(-fabsf(rank));
  }
  return ~__float_as_uint(rank);
}

template <class Config>
  requires(Config::kTokens == 8)
__device__ __forceinline__ std::uint32_t router_rank_key(float rank) {
  if (rank >= 0.0F) {
    return __float_as_uint(-fabsf(rank));
  }
  return ~__float_as_uint(rank);
}

template <class Config>
  requires(Config::kTokens == 4)
__device__ __forceinline__ void produce_rms_gamma(PipelineControl* control,
                                                  std::uint8_t* dynamic_storage,
                                                  NormGain const* gamma) {
  constexpr int kTask = 0;

  int const producer_warp = (static_cast<int>(threadIdx.x) - kConsumerThreads) / kWarpSize;
  int const lane = static_cast<int>(threadIdx.x) & (kWarpSize - 1);
  producer_acquire_stage(control, kTask, kRmsGammaBytesPerProducer);
  if (lane == 0) {
    int const byte_offset = producer_warp * kRmsGammaBytesPerProducer;
    auto const* gamma_bytes = reinterpret_cast<std::uint8_t const*>(gamma);
    bulk_copy_g2s(gamma_bytes + byte_offset, control->ready,
                  pipeline_stage<PipelineControl>(dynamic_storage, kTask) + byte_offset,
                  kRmsGammaBytesPerProducer);
  }
}

template <class Config>
  requires(Config::kTokens == 8)
__device__ __forceinline__ void produce_rms_gamma(PipelineControl* control,
                                                  std::uint8_t* dynamic_storage, NormGain const* gamma,
                                                  int task) {
  int const producer_warp = (static_cast<int>(threadIdx.x) - kConsumerThreads) / kWarpSize;
  int const lane = static_cast<int>(threadIdx.x) & (kWarpSize - 1);
  producer_acquire_stage(control, task, kRmsGammaBytesPerProducer);
  if (lane == 0) {
    int const byte_offset = producer_warp * kRmsGammaBytesPerProducer;
    auto const* gamma_bytes = reinterpret_cast<std::uint8_t const*>(gamma);
    bulk_copy_g2s(gamma_bytes + byte_offset, control->ready + (task & (kPipelineStages - 1)),
                  pipeline_stage<PipelineControl>(dynamic_storage, task) + byte_offset,
                  kRmsGammaBytesPerProducer);
  }
}

template <class Config, int kPair>
  requires(Config::kTokens == 4)
__device__ __forceinline__ void produce_router_pair(PipelineControl* control,
                                                    std::uint8_t* dynamic_storage,
                                                    __nv_bfloat16 const* router_weight, int tile) {
  static_assert(kPair >= 0 && kPair < kRouterPairsPerCta);
  constexpr int kTask = kRouterPipelineTaskBase + kPair;
  constexpr int kStage = kTask & (kPipelineStages - 1);
  constexpr unsigned int kCycle = kTask / kPipelineStages;

  if (threadIdx.x == kConsumerThreads) {
    if constexpr (kCycle != 0) {
      constexpr unsigned int kEmptyPhase = (kCycle - 1U) & 1U;
      cluster_barrier_wait(control->empty + kStage, kEmptyPhase);
    }
    producer_arrive_and_expect_tx_count(control->ready + kStage, kProducerWarps,
                                        kRouterPairTransactionBytes);

    int const first_expert = tile * kRouterExpertsPerCta + 2 * kPair;
    __nv_bfloat16 const* source = router_weight + static_cast<std::size_t>(first_expert) * kHidden;
    // Keep near-token reuse, then let dead Router rows yield L2 to GateUp.
    bulk_copy_g2s_cache_hint<cute::TMA::CacheHintSm100::EVICT_FIRST>(
        source, control->ready + kStage, pipeline_stage<PipelineControl>(dynamic_storage, kTask),
        kRouterPairTransactionBytes);
  }
}

template <class Config, bool kPublishGlobal>
  requires(Config::kTokens == 4)
__device__ void rms_norm_consumer(__nv_bfloat16 const* input, __nv_bfloat16* shared_input,
                                  __nv_bfloat16* global_output, float* scratch,
                                  PipelineControl* control, std::uint8_t* dynamic_storage) {
  int const tid = static_cast<int>(threadIdx.x);
  int const lane = tid & (kWarpSize - 1);
  int const warp = tid / kWarpSize;
  constexpr int kPairs = kHidden / 2;

  stage_rms_input<kConsumerThreads, kRmsInputCopiesPerConsumer, kRmsInputCopyBytes>(input,
                                                                                    shared_input);

  auto const* input_pairs = reinterpret_cast<__nv_bfloat162 const*>(shared_input);
  float sum = 0.0F;
#pragma unroll 1
  for (int pair = tid; pair < kPairs; pair += kConsumerThreads) {
    float2 const values = __bfloat1622float2(input_pairs[pair]);
    sum = fmaf(values.x, values.x, sum);
    sum = fmaf(values.y, values.y, sum);
  }

#pragma unroll
  for (int offset = kWarpSize / 2; offset > 0; offset >>= 1) {
    sum += __shfl_down_sync(kFullWarpMask, sum, offset);
  }
  if (lane == 0) {
    scratch[warp] = sum;
  }
  consumer_group_sync<kConsumerThreads>();

  if (warp == 0) {
    float warp_sum = lane < kConsumerWarps ? scratch[lane] : 0.0F;
#pragma unroll
    for (int offset = kConsumerWarps / 2; offset > 0; offset >>= 1) {
      warp_sum += __shfl_down_sync(kFullWarpMask, warp_sum, offset, kConsumerWarps);
    }
    if (lane == 0) {
      scratch[0] = warp_sum;
    }
  }
  consumer_group_sync<kConsumerThreads>();

  consumer_wait_stage(control, 0);
  float const inv_rms = rsqrtf(scratch[0] / static_cast<float>(kHidden) + kEps);
  auto const* gamma_pairs =
      reinterpret_cast<NormGain const*>(pipeline_stage<PipelineControl>(dynamic_storage, 0));
  auto* shared_output_pairs = reinterpret_cast<__nv_bfloat162*>(shared_input);
  auto* global_output_pairs = reinterpret_cast<__nv_bfloat162*>(global_output);

#pragma unroll 1
  for (int pair = tid; pair < kPairs; pair += kConsumerThreads) {
    float2 const values = __bfloat1622float2(input_pairs[pair]);
    float2 const gamma = load_norm2(gamma_pairs + pair * 2);
    __nv_bfloat162 const normalized =
        __floats2bfloat162_rn(values.x * inv_rms * gamma.x, values.y * inv_rms * gamma.y);
    if constexpr (kPublishGlobal) {
      global_output_pairs[pair] = normalized;
    } else {
      shared_output_pairs[pair] = normalized;
    }
  }
  consumer_release_stage(control, 0);
}

template <class Config, bool kPublishGlobal, int kReleaseThreads>
  requires(Config::kTokens == 8)
__device__ void rms_norm_consumer(__nv_bfloat16 const* input, __nv_bfloat16* shared_input,
                                  __nv_bfloat16* global_output, float* scratch,
                                  PipelineControl* control, std::uint8_t* dynamic_storage,
                                  int gamma_task) {
  int const tid = static_cast<int>(threadIdx.x);
  int const lane = tid & (kWarpSize - 1);
  int const warp = tid / kWarpSize;
  constexpr int kPairs = kHidden / 2;

  stage_rms_input<kConsumerThreads, kRmsInputCopiesPerConsumer, kRmsInputCopyBytes>(input,
                                                                                    shared_input);

  auto const* input_pairs = reinterpret_cast<__nv_bfloat162 const*>(shared_input);
  float sum = 0.0F;
#pragma unroll 1
  for (int pair = tid; pair < kPairs; pair += kConsumerThreads) {
    float2 const values = __bfloat1622float2(input_pairs[pair]);
    sum = fmaf(values.x, values.x, sum);
    sum = fmaf(values.y, values.y, sum);
  }

#pragma unroll
  for (int offset = kWarpSize / 2; offset > 0; offset >>= 1) {
    sum += __shfl_down_sync(kFullWarpMask, sum, offset);
  }
  if (lane == 0) {
    scratch[warp] = sum;
  }
  consumer_group_sync<kConsumerThreads>();

  if (warp == 0) {
    float warp_sum = lane < kConsumerWarps ? scratch[lane] : 0.0F;
#pragma unroll
    for (int offset = kConsumerWarps / 2; offset > 0; offset >>= 1) {
      warp_sum += __shfl_down_sync(kFullWarpMask, warp_sum, offset, kConsumerWarps);
    }
    if (lane == 0) {
      scratch[0] = warp_sum;
    }
  }
  consumer_group_sync<kConsumerThreads>();

  consumer_wait_stage(control, gamma_task);
  float const inv_rms = rsqrtf(scratch[0] / static_cast<float>(kHidden) + kEps);
  auto const* gamma_pairs =
      reinterpret_cast<NormGain const*>(pipeline_stage<PipelineControl>(dynamic_storage, gamma_task));
  auto* shared_output_pairs = reinterpret_cast<__nv_bfloat162*>(shared_input);
  auto* global_output_pairs = reinterpret_cast<__nv_bfloat162*>(global_output);

#pragma unroll 1
  for (int pair = tid; pair < kPairs; pair += kConsumerThreads) {
    float2 const values = __bfloat1622float2(input_pairs[pair]);
    float2 const gamma = load_norm2(gamma_pairs + pair * 2);
    __nv_bfloat162 const normalized =
        __floats2bfloat162_rn(values.x * inv_rms * gamma.x, values.y * inv_rms * gamma.y);
    if constexpr (kPublishGlobal) {
      shared_output_pairs[pair] = normalized;
      global_output_pairs[pair] = normalized;
    } else {
      shared_output_pairs[pair] = normalized;
    }
  }
  if constexpr (kReleaseThreads < kConsumerThreads) {
    consumer_group_sync<kConsumerThreads>();
    if (tid < kReleaseThreads) {
      consumer_release_stage(control, gamma_task);
    }
  } else {
    consumer_release_stage(control, gamma_task);
  }
}

template <class Config>
  requires(Config::kTokens == 4)
__device__ void router_projection_pair_consumer(__nv_bfloat16 const* normalized, int token,
                                                int tile, float* router_logits, float* scratch,
                                                PipelineControl* control,
                                                std::uint8_t* dynamic_storage) {
  int const tid = static_cast<int>(threadIdx.x);
  int const warp = tid / kWarpSize;
  int const lane = tid & (kWarpSize - 1);
  auto const* activation_pairs = reinterpret_cast<__nv_bfloat162 const*>(normalized);

#pragma unroll 1
  for (int expert_pair = 0; expert_pair < kRouterPairsPerCta; ++expert_pair) {
    int const task = kRouterPipelineTaskBase + expert_pair;
    consumer_wait_stage(control, task);

    auto const* weight_a_pairs = reinterpret_cast<__nv_bfloat162 const*>(
        pipeline_stage<PipelineControl>(dynamic_storage, task));
    auto const* weight_b_pairs = reinterpret_cast<__nv_bfloat162 const*>(
        pipeline_stage<PipelineControl>(dynamic_storage, task) + kRouterPairExpertBytes);

    float2 partial_a = make_float2(0.0F, 0.0F);
    float2 partial_b = make_float2(0.0F, 0.0F);
#pragma unroll
    for (int chunk = 0; chunk < kRouterPairChunksPerThread; ++chunk) {
      int const packed_index = tid + chunk * kConsumerThreads;
      float2 const activation = __bfloat1622float2(activation_pairs[packed_index]);
      float2 const weight_a = __bfloat1622float2(weight_a_pairs[packed_index]);
      float2 const weight_b = __bfloat1622float2(weight_b_pairs[packed_index]);
      partial_a = fma_f32x2(activation, weight_a, partial_a);
      partial_b = fma_f32x2(activation, weight_b, partial_b);
    }

    float sum_a = partial_a.x + partial_a.y;
#pragma unroll
    for (int offset = kWarpSize / 2; offset > 0; offset >>= 1) {
      sum_a += __shfl_down_sync(kFullWarpMask, sum_a, offset);
    }
    if (lane == 0) {
      scratch[(2 * expert_pair) * kConsumerWarps + warp] = sum_a;
    }

    float sum_b = partial_b.x + partial_b.y;
#pragma unroll
    for (int offset = kWarpSize / 2; offset > 0; offset >>= 1) {
      sum_b += __shfl_down_sync(kFullWarpMask, sum_b, offset);
    }
    if (lane == 0) {
      scratch[(2 * expert_pair + 1) * kConsumerWarps + warp] = sum_b;
    }
    consumer_release_stage(control, task);
  }

  consumer_group_sync<kConsumerThreads>();

  int const subgroup = lane / kConsumerWarps;
  int const source_warp = lane % kConsumerWarps;
  int const expert_in_cta = warp * kRouterReductionExpertsPerWarp + subgroup;
  if (warp < kRouterReductionWarps) {
    float total = scratch[expert_in_cta * kConsumerWarps + source_warp];
#pragma unroll
    for (int offset = kConsumerWarps / 2; offset > 0; offset >>= 1) {
      total += __shfl_down_sync(kFullWarpMask, total, offset, kConsumerWarps);
    }
    if (source_warp == 0) {
      auto* destination = reinterpret_cast<std::uint32_t*>(
          router_logits + token * kRoutedExperts + tile * kRouterExpertsPerCta + expert_in_cta);
      store_relaxed_gpu_u32(destination, __float_as_uint(total));
    }
  }
  consumer_group_sync<kConsumerThreads>();
}

template <class Config>
  requires(Config::kTokens == 8)
__device__ void router_projection_pair_consumer(__nv_bfloat16 const* normalized, int token,
                                                int expert_base, int expert_count,
                                                float* router_logits, float* scratch,
                                                PipelineControl* control,
                                                std::uint8_t* dynamic_storage) {
  int const tid = static_cast<int>(threadIdx.x);
  int const warp = tid / kWarpSize;
  int const lane = tid & (kWarpSize - 1);
  auto const* activation_pairs = reinterpret_cast<__nv_bfloat162 const*>(normalized);

  int const pair_count = (expert_count + 1) / 2;
#pragma unroll 1
  for (int expert_pair = 0; expert_pair < pair_count; ++expert_pair) {
    int const task = kRouterPipelineTaskBase + expert_pair;
    bool const has_second = 2 * expert_pair + 1 < expert_count;
    consumer_wait_stage(control, task);

    auto const* weight_a_pairs = reinterpret_cast<__nv_bfloat162 const*>(
        pipeline_stage<PipelineControl>(dynamic_storage, task));
    auto const* weight_b_pairs = reinterpret_cast<__nv_bfloat162 const*>(
        pipeline_stage<PipelineControl>(dynamic_storage, task) + kRouterPairExpertBytes);

    float2 partial_a = make_float2(0.0F, 0.0F);
    float2 partial_b = make_float2(0.0F, 0.0F);
#pragma unroll
    for (int chunk = 0; chunk < kRouterPairChunksPerThread; ++chunk) {
      int const packed_index = tid + chunk * kConsumerThreads;
      float2 const activation = __bfloat1622float2(activation_pairs[packed_index]);
      float2 const weight_a = __bfloat1622float2(weight_a_pairs[packed_index]);
      partial_a = fma_f32x2(activation, weight_a, partial_a);
      if (has_second) {
        float2 const weight_b = __bfloat1622float2(weight_b_pairs[packed_index]);
        partial_b = fma_f32x2(activation, weight_b, partial_b);
      }
    }

    float sum_a = partial_a.x + partial_a.y;
#pragma unroll
    for (int offset = kWarpSize / 2; offset > 0; offset >>= 1) {
      sum_a += __shfl_down_sync(kFullWarpMask, sum_a, offset);
    }
    if (lane == 0) {
      int const expert_a = 2 * expert_pair;
      scratch[expert_a * kConsumerWarps + warp] = sum_a;
    }

    if (has_second) {
      float sum_b = partial_b.x + partial_b.y;
#pragma unroll
      for (int offset = kWarpSize / 2; offset > 0; offset >>= 1) {
        sum_b += __shfl_down_sync(kFullWarpMask, sum_b, offset);
      }
      if (lane == 0) {
        int const expert_b = 2 * expert_pair + 1;
        scratch[expert_b * kConsumerWarps + warp] = sum_b;
      }
    }
    consumer_release_stage(control, task);
  }

  consumer_group_sync<kConsumerThreads>();

  int const subgroup = lane / kConsumerWarps;
  int const source_warp = lane % kConsumerWarps;
  int const expert_in_cta = warp * kRouterReductionExpertsPerWarp + subgroup;
  if (warp < kRouterReductionWarps) {
    float total =
        expert_in_cta < expert_count ? scratch[expert_in_cta * kConsumerWarps + source_warp] : 0.0F;
#pragma unroll
    for (int offset = kConsumerWarps / 2; offset > 0; offset >>= 1) {
      total += __shfl_down_sync(kFullWarpMask, total, offset, kConsumerWarps);
    }
    if (source_warp == 0 && expert_in_cta < expert_count) {
      auto* destination = reinterpret_cast<std::uint32_t*>(router_logits + token * kRoutedExperts +
                                                           expert_base + expert_in_cta);
      store_relaxed_gpu_u32(destination, __float_as_uint(total));
    }
  }
  consumer_group_sync<kConsumerThreads>();
}

template <class Config>
  requires((Config::kTokens == 8 && Config::kTpSize == 8))
__host__ __device__ constexpr RoutedCtaDescriptor routed_cta_descriptor(int local) {
  int const virtual_tile_base = local * 7 + (local > 2 ? 1 : 0) + (local > 3 ? 1 : 0);
  int const virtual_tile_count = local == 2 || local == 3 ? 8 : 7;
  return RoutedCtaDescriptor{local < 14 ? local * 14 : 196 + (local - 14) * 15,
                             local < 14 ? 14 : 15,
                             local < 2 ? local * 8 : 16 + (local - 2) * 7,
                             local < 2 ? 8 : 7,
                             virtual_tile_base * kDownM16PerVirtualTile,
                             virtual_tile_count * kDownM16PerVirtualTile};
}

template <class Config>
  requires((Config::kTokens == 8 && Config::kTpSize == 4))
__host__ __device__ constexpr RoutedCtaDescriptor routed_cta_descriptor(int local) {
  int const virtual_tile_base = local * 7 + (local > 2 ? 1 : 0) + (local > 3 ? 1 : 0);
  int const virtual_tile_count = local == 2 || local == 3 ? 8 : 7;
  return RoutedCtaDescriptor{local < 14 ? local * 14 : 196 + (local - 14) * 15,
                             local < 14 ? 14 : 15,
                             (local < 2 ? local * 8 : 16 + (local - 2) * 7) * kGateUpLayoutFactor,
                             (local < 2 ? 8 : 7) * kGateUpLayoutFactor,
                             virtual_tile_base * kDownM16PerVirtualTile,
                             virtual_tile_count * kDownM16PerVirtualTile};
}

template <class Config>
  requires(Config::kTokens == 8)
constexpr int down_virtual_tile_count(int m16_count) {
  return m16_count / kDownM16PerVirtualTile;
}

template <class Config>
  requires(Config::kTokens == 8)
__device__ __forceinline__ std::uint64_t load_router_logit_pair(std::uint64_t const* address) {
  return load_relaxed_gpu_u64(address);
}

template <class Config>
  requires(Config::kTokens == 8)
__device__ __forceinline__ void router_poll_backoff() {
  __nanosleep(32);
}

template <class Config>
  requires(Config::kTokens == 8)
__device__ __forceinline__ void produce_router_pairs(PipelineControl* control,
                                                     std::uint8_t* dynamic_storage,
                                                     __nv_bfloat16 const* router_weight,
                                                     int expert_base, int expert_count) {
  if (threadIdx.x == kConsumerThreads) {
    int const pair_count = (expert_count + 1) / 2;
#pragma unroll 1
    for (int pair = 0; pair < pair_count; ++pair) {
      int const first_in_cta = pair * 2;
      int const experts_in_pair = first_in_cta + 1 < expert_count ? 2 : 1;
      int const transaction_bytes = experts_in_pair * kRouterPairExpertBytes;
      int const task = kRouterPipelineTaskBase + pair;
      int const stage = single_leader_acquire_stage(control, task, transaction_bytes);
      __nv_bfloat16 const* source =
          router_weight + static_cast<std::size_t>(expert_base + first_in_cta) * kHidden;
      bulk_copy_g2s_cache_hint<cute::TMA::CacheHintSm100::EVICT_FIRST>(
          source, control->ready + stage, pipeline_stage<PipelineControl>(dynamic_storage, task),
          transaction_bytes);
    }
  }
}

template <class Config>
  requires(Config::kTokens == 4)
__device__ __forceinline__ std::uint64_t wait_router_logit_pair(std::uint64_t const* address) {
  std::uint64_t pair = load_relaxed_gpu_u64(address);
  while (router_logit_pair_has_nan(pair)) {
    __nanosleep(32);
    pair = load_relaxed_gpu_u64(address);
  }
  return pair;
}

template <class Config>
  requires(Config::kTokens == 8)
__device__ __forceinline__ std::uint64_t wait_router_logit_pair(std::uint64_t const* address) {
  std::uint64_t pair = load_router_logit_pair<Config>(address);
  while (router_logit_pair_has_nan(pair)) {
    router_poll_backoff<Config>();
    pair = load_router_logit_pair<Config>(address);
  }
  return pair;
}

template <class Config>
  requires(Config::kTokens == 4)
__device__ __forceinline__ void wait_normalized_tokens(std::uint32_t const* normalized_ready) {
  int const tid = static_cast<int>(threadIdx.x);
  if (tid < kTokens) {
    std::uint32_t ready = load_relaxed_gpu_u32<Config>(normalized_ready + tid);
    while (ready != kGlobalReady) {
      __nanosleep(32);
      ready = load_relaxed_gpu_u32<Config>(normalized_ready + tid);
    }
    acquire_fence_gpu<Config>();
  }
  consumer_group_sync<kConsumerThreads>();
}

template <class Config>
  requires(Config::kTokens == 8)
__device__ __forceinline__ void wait_normalized_tokens(std::uint32_t const* normalized_ready) {
  int const tid = static_cast<int>(threadIdx.x);
  if (tid < kTokens) {
    std::uint32_t ready = comm::load_relaxed_gpu_u32(normalized_ready + tid);
    while (ready != kGlobalReady) {
      __nanosleep(32);
      ready = comm::load_relaxed_gpu_u32(normalized_ready + tid);
    }
    comm::acquire_fence_gpu();
  }
  consumer_group_sync<kConsumerThreads>();
}

template <class Config>
  requires(Config::kTokens == 4)
__device__ void routed_rms_norm_and_router_projection(
    __nv_bfloat16 const* input, NormGain const* gamma, __nv_bfloat16 const* router_weight, int token,
    int tile, __nv_bfloat16* normalized, float* router_logits, float* scratch,
    PipelineControl* control, std::uint8_t* dynamic_storage) {
  int const tid = static_cast<int>(threadIdx.x);
  if (tid >= kConsumerThreads) {
    produce_rms_gamma<Config>(control, dynamic_storage, gamma);
    produce_router_pair<Config, 0>(control, dynamic_storage, router_weight, tile);
    produce_router_pair<Config, 1>(control, dynamic_storage, router_weight, tile);
    produce_router_pair<Config, 2>(control, dynamic_storage, router_weight, tile);
    produce_router_pair<Config, 3>(control, dynamic_storage, router_weight, tile);
  } else {
    rms_norm_consumer<Config, false>(input, normalized, nullptr, scratch, control, dynamic_storage);
    router_projection_pair_consumer<Config>(normalized, token, tile, router_logits, scratch,
                                            control, dynamic_storage);
  }
}

template <class Config>
  requires(Config::kTokens == 8)
__device__ void routed_rms_norm_and_router_projection(
    __nv_bfloat16 const* input, NormGain const* gamma, __nv_bfloat16 const* router_weight, int token,
    int expert_base, int expert_count, __nv_bfloat16* normalized, float* router_logits,
    float* scratch, PipelineControl* control, std::uint8_t* dynamic_storage) {
  int const tid = static_cast<int>(threadIdx.x);
  if (tid >= kConsumerThreads) {
    produce_rms_gamma<Config>(control, dynamic_storage, gamma, 0);
    produce_router_pairs<Config>(control, dynamic_storage, router_weight, expert_base,
                                 expert_count);
  } else {
    rms_norm_consumer<Config, false, kConsumerThreads>(input, normalized, nullptr, scratch, control,
                                                       dynamic_storage, 0);
    router_projection_pair_consumer<Config>(normalized, token, expert_base, expert_count,
                                            router_logits, scratch, control, dynamic_storage);
  }
}

template <class Config>
  requires(Config::kTokens == 4)
__device__ void shared_rms_norm_and_publish(__nv_bfloat16 const* input, NormGain const* gamma,
                                            __nv_bfloat16* shared_input,
                                            __nv_bfloat16* global_output,
                                            std::uint32_t* normalized_ready, int token,
                                            float* scratch, PipelineControl* control,
                                            std::uint8_t* dynamic_storage) {
  if (threadIdx.x >= kConsumerThreads) {
    produce_rms_gamma<Config>(control, dynamic_storage, gamma);
  } else {
    rms_norm_consumer<Config, true>(input, shared_input, global_output, scratch, control,
                                    dynamic_storage);
    // Baseline: every consumer fences the slice it published. The local
    // consumer barrier turns the 256 writer fences into one token release.
    __threadfence();
    consumer_group_sync<kConsumerThreads>();
    if (threadIdx.x == 0) {
      store_release_gpu_u32(normalized_ready + token, kGlobalReady);
    }
  }
}

template <class Config>
  requires((Config::kTokens == 4 && Config::kTpSize == 4))
__device__ __noinline__ void collect_topk_frequency(std::int32_t const* published_indices,
                                                    std::uint32_t const* topk_ready,
                                                    std::uint32_t* hot_expert_mask,
                                                    std::uint32_t* frequency_ready,
                                                    std::uint32_t const* ar_epoch_counters,
                                                    std::uint8_t* dynamic_storage) {
  int const lane = static_cast<int>(threadIdx.x);
  std::uint32_t const generation =
      next_generation<Config>(load_relaxed_gpu_u32<Config>(ar_epoch_counters));
  int const token = lane / kTopK;
  std::uint32_t observed = load_relaxed_gpu_u32<Config>(topk_ready + token);
  while (observed != generation) {
    __nanosleep(32);
    observed = load_relaxed_gpu_u32<Config>(topk_ready + token);
  }
  acquire_fence_gpu<Config>();
  __syncwarp(kFullWarpMask);

  int const expert = published_indices[lane];
  std::uint32_t const matches = __match_any_sync(kFullWarpMask, expert);
  int const frequency = __popc(matches);
  bool const leader = lane == __ffs(matches) - 1;
  std::uint32_t const lanes_before = (1U << lane) - 1U;
  int retained_before = 0;
  bool retain = false;
#pragma unroll
  for (int target_frequency = kTokens; target_frequency >= 2; --target_frequency) {
    std::uint32_t const bucket =
        __ballot_sync(kFullWarpMask, leader && frequency == target_frequency);
    if (leader && frequency == target_frequency) {
      int const rank = retained_before + __popc(bucket & lanes_before);
      retain = rank < kHotExpertBudget;
    }
    retained_before += __popc(bucket);
  }

  // Build the 256-bit result in otherwise-idle CTA-local scratch, then let
  // lane 0 publish all words before the generation-tagged release.
  auto* local_mask = reinterpret_cast<std::uint32_t*>(dynamic_storage + kAuxOffset);
  if (lane < kHotExpertMaskWords) {
    local_mask[lane] = 0U;
  }
  __syncwarp(kFullWarpMask);
  if (retain) {
    atomicOr(local_mask + (expert >> 5), 1U << (expert & 31));
  }
  __syncwarp(kFullWarpMask);
  if (lane == 0) {
#pragma unroll
    for (int word = 0; word < kHotExpertMaskWords; ++word) {
      store_relaxed_gpu_u32(hot_expert_mask + word, local_mask[word]);
    }
    store_release_gpu_u32(frequency_ready, generation);
  }
}

template <class Config>
  requires((Config::kTokens == 8 && Config::kTpSize == 8))
constexpr bool validate_routed_descriptors() {
  bool router_seen[kRoutedExperts] = {};
  bool gate_seen[kTopK * 16] = {};
  bool down_seen[kDownM16Tiles] = {};
  int total_virtual_tiles = 0;

  for (int local = 0; local < kRoutedCtasPerToken; ++local) {
    RoutedCtaDescriptor const descriptor = routed_cta_descriptor<Config>(local);
    if (descriptor.router_expert_count < 1 ||
        descriptor.router_expert_count > kRouterMaxExpertsPerCta ||
        descriptor.gate_group_count < 1 || descriptor.gate_group_count > kRoutedGateUpMaxGroups ||
        descriptor.down_m16_count < 1 || descriptor.down_m16_count > kDownMaxM16PerCta ||
        descriptor.down_m16_base % kDownM16PerVirtualTile != 0 ||
        descriptor.down_m16_count % kDownM16PerVirtualTile != 0) {
      return false;
    }
    int const first_slot = descriptor.gate_flat_base / 16;
    int const last_slot = (descriptor.gate_flat_base + descriptor.gate_group_count - 1) / 16;
    if (last_slot - first_slot > 1) {
      return false;
    }
    for (int i = 0; i < descriptor.router_expert_count; ++i) {
      int const expert = descriptor.router_expert_base + i;
      if (expert < 0 || expert >= kRoutedExperts || router_seen[expert]) {
        return false;
      }
      router_seen[expert] = true;
    }
    for (int i = 0; i < descriptor.gate_group_count; ++i) {
      int const flat = descriptor.gate_flat_base + i;
      if (flat < 0 || flat >= kTopK * 16 || gate_seen[flat]) {
        return false;
      }
      gate_seen[flat] = true;
    }
    for (int i = 0; i < descriptor.down_m16_count; ++i) {
      int const m16 = descriptor.down_m16_base + i;
      if (m16 < 0 || m16 >= kDownM16Tiles || down_seen[m16]) {
        return false;
      }
      down_seen[m16] = true;
    }
    total_virtual_tiles += down_virtual_tile_count<Config>(descriptor.down_m16_count);
  }

  for (int expert = 0; expert < kRoutedExperts; ++expert) {
    if (!router_seen[expert]) {
      return false;
    }
  }
  for (int flat = 0; flat < kTopK * 16; ++flat) {
    if (!gate_seen[flat]) {
      return false;
    }
  }
  for (int m16 = 0; m16 < kDownM16Tiles; ++m16) {
    if (!down_seen[m16]) {
      return false;
    }
  }
  return total_virtual_tiles == 128;
}

template <class Config>
  requires((Config::kTokens == 8 && Config::kTpSize == 4))
constexpr bool validate_routed_descriptors() {
  bool router_seen[kRoutedExperts] = {};
  bool gate_seen[kTopK * kGateUpRowGroupsPerExpert] = {};
  bool down_seen[kDownM16Tiles] = {};
  int total_virtual_tiles = 0;

  for (int local = 0; local < kRoutedCtasPerToken; ++local) {
    RoutedCtaDescriptor const descriptor = routed_cta_descriptor<Config>(local);
    if (descriptor.router_expert_count < 1 ||
        descriptor.router_expert_count > kRouterMaxExpertsPerCta ||
        descriptor.gate_group_count < 1 || descriptor.gate_group_count > kRoutedGateUpMaxGroups ||
        (descriptor.gate_flat_base & 1) != 0 || (descriptor.gate_group_count & 1) != 0 ||
        descriptor.down_m16_count < 1 || descriptor.down_m16_count > kDownMaxM16PerCta ||
        descriptor.down_m16_base % kDownM16PerVirtualTile != 0 ||
        descriptor.down_m16_count % kDownM16PerVirtualTile != 0) {
      return false;
    }
    int const first_slot = descriptor.gate_flat_base / kGateUpRowGroupsPerExpert;
    int const last_slot =
        (descriptor.gate_flat_base + descriptor.gate_group_count - 1) / kGateUpRowGroupsPerExpert;
    if (last_slot - first_slot > 1) {
      return false;
    }
    for (int i = 0; i < descriptor.gate_group_count; i += 2) {
      int const first_flat = descriptor.gate_flat_base + i;
      if (first_flat / kGateUpRowGroupsPerExpert != (first_flat + 1) / kGateUpRowGroupsPerExpert) {
        return false;
      }
    }
    for (int i = 0; i < descriptor.router_expert_count; ++i) {
      int const expert = descriptor.router_expert_base + i;
      if (expert < 0 || expert >= kRoutedExperts || router_seen[expert]) {
        return false;
      }
      router_seen[expert] = true;
    }
    for (int i = 0; i < descriptor.gate_group_count; ++i) {
      int const flat = descriptor.gate_flat_base + i;
      if (flat < 0 || flat >= kTopK * kGateUpRowGroupsPerExpert || gate_seen[flat]) {
        return false;
      }
      gate_seen[flat] = true;
    }
    for (int i = 0; i < descriptor.down_m16_count; ++i) {
      int const m16 = descriptor.down_m16_base + i;
      if (m16 < 0 || m16 >= kDownM16Tiles || down_seen[m16]) {
        return false;
      }
      down_seen[m16] = true;
    }
    total_virtual_tiles += down_virtual_tile_count<Config>(descriptor.down_m16_count);
  }

  for (int expert = 0; expert < kRoutedExperts; ++expert) {
    if (!router_seen[expert]) {
      return false;
    }
  }
  for (int flat = 0; flat < kTopK * kGateUpRowGroupsPerExpert; ++flat) {
    if (!gate_seen[flat]) {
      return false;
    }
  }
  for (int m16 = 0; m16 < kDownM16Tiles; ++m16) {
    if (!down_seen[m16]) {
      return false;
    }
  }
  return total_virtual_tiles == 128;
}

template <class Config>
  requires(Config::kTokens == 8)
__device__ void shared_rms_norm_and_publish_pair(__nv_bfloat16 const* input, NormGain const* gamma,
                                                 __nv_bfloat16* shared_input,
                                                 __nv_bfloat16* global_output,
                                                 std::uint32_t* normalized_ready, int first_token,
                                                 float* scratch, PipelineControl* control,
                                                 std::uint8_t* dynamic_storage) {
  if (threadIdx.x >= kConsumerThreads) {
#pragma unroll
    for (int local_token = 0; local_token < kSharedTokensPerCta; ++local_token) {
      produce_rms_gamma<Config>(control, dynamic_storage, gamma, local_token);
    }
  } else {
#pragma unroll
    for (int local_token = 0; local_token < kSharedTokensPerCta; ++local_token) {
      int const token = first_token + local_token;
      rms_norm_consumer<Config, true, kSharedConsumerThreadsPerGroup>(
          input + static_cast<std::size_t>(local_token) * kHidden, shared_input,
          global_output + static_cast<std::size_t>(local_token) * kHidden, scratch, control,
          dynamic_storage, local_token);
      __threadfence();
      consumer_group_sync<kConsumerThreads>();
      if (threadIdx.x == 0) {
        store_release_gpu_u32(normalized_ready + token, kGlobalReady);
      }
    }
  }
}

template <class Config>
  requires((Config::kTokens == 4 && Config::kTpSize == 8))
__device__ void select_topk(float const* router_logits, float const* router_bias, int token,
                            int selected_slot, std::int32_t* local_selected_indices,
                            float* local_selected_probs, std::int32_t* published_indices,
                            float* published_probs, bool publish_global,
                            std::uint64_t* topk_handoff, std::uint8_t* dynamic_storage) {
  int const tid = static_cast<int>(threadIdx.x);
  if (tid >= kConsumerThreads) {
    return;
  }

  auto* raw_scores = reinterpret_cast<float*>(dynamic_storage + kTopkRawOffset);
  auto* staged_bias = reinterpret_cast<float*>(dynamic_storage + kTopkBiasOffset);

  if (tid < kRoutedExperts / 2) {
    auto const* global_pairs = reinterpret_cast<std::uint64_t const*>(
        router_logits + static_cast<std::size_t>(token) * kRoutedExperts);
    auto* shared_pairs = reinterpret_cast<std::uint64_t*>(raw_scores);
    shared_pairs[tid] = wait_router_logit_pair<Config>(global_pairs + tid);
  }
  if (tid < kRoutedExperts / 4) {
    int const expert_base = tid * 4;
    copy_16b_g2s(router_bias + expert_base, staged_bias + expert_base);
  }
  cutlass::arch::cp_async_fence();
  cutlass::arch::cp_async_wait<0>();
  consumer_group_sync<kConsumerThreads>();

  raw_scores[tid] = router_sigmoid(raw_scores[tid]);
  consumer_group_sync<kConsumerThreads>();

  if (tid < kWarpSize) {
    int const lane = tid & (kWarpSize - 1);
    std::uint32_t candidate_keys[kTopK];
#pragma unroll
    for (int slot = 0; slot < kTopK; ++slot) {
      int const expert = lane + slot * kWarpSize;
      float const rank = raw_scores[expert] + staged_bias[expert];
      candidate_keys[slot] = router_rank_key<Config>(rank);
    }

    std::uint32_t selected_id = 0xffffffffU;
    float selected_raw = 0.0F;
#pragma unroll
    for (int pick = 0; pick < kTopK; ++pick) {
      std::uint32_t local_key = 0;
      std::uint32_t local_id = 0xffffffffU;
#pragma unroll
      for (int slot = 0; slot < kTopK; ++slot) {
        std::uint32_t const key = candidate_keys[slot];
        std::uint32_t const expert = static_cast<std::uint32_t>(lane + slot * kWarpSize);
        if (key > local_key || (key == local_key && expert < local_id)) {
          local_key = key;
          local_id = expert;
        }
      }

      std::uint32_t const winner_key = __reduce_max_sync(kFullWarpMask, local_key);
      std::uint32_t const eligible_id = local_key == winner_key ? local_id : 0xffffffffU;
      std::uint32_t const winner_id = __reduce_min_sync(kFullWarpMask, eligible_id);

      if (lane == pick) {
        selected_id = winner_id;
        selected_raw = raw_scores[winner_id];
        local_selected_indices[pick] = static_cast<std::int32_t>(winner_id);
        if (pick == selected_slot) {
          cutlass::arch::ClusterBarrier::arrive(topk_handoff);
        }
      }
      if (lane == (winner_id & (kWarpSize - 1))) {
        candidate_keys[winner_id / kWarpSize] = 0;
      }
    }

    float score_sum = lane < kTopK ? selected_raw : 0.0F;
#pragma unroll
    for (int offset = kTopK / 2; offset > 0; offset >>= 1) {
      score_sum += __shfl_down_sync(kFullWarpMask, score_sum, offset, kTopK);
    }

    float normalizer = 0.0F;
    if (lane == 0 && score_sum > 0.0F) {
      normalizer = kRouteScale / score_sum;
    }
    normalizer = __shfl_sync(kFullWarpMask, normalizer, 0);

    if (lane < kTopK) {
      local_selected_probs[lane] = selected_raw * normalizer;
      if (publish_global) {
        published_indices[token * kTopK + lane] = static_cast<std::int32_t>(selected_id);
        published_probs[token * kTopK + lane] = selected_raw * normalizer;
      }
    }
  }
  consumer_group_sync<kConsumerThreads>();
}

template <class Config>
  requires((Config::kTokens == 4 && Config::kTpSize == 4))
__device__ void select_topk(float const* router_logits, float const* router_bias, int token,
                            int counter, std::int32_t* local_selected_indices,
                            float* local_selected_probs, std::int32_t* published_indices,
                            float* published_probs, std::uint32_t* topk_claim,
                            std::uint32_t* topk_ready, std::uint32_t const* ar_epoch_counters,
                            std::uint8_t* dynamic_storage) {
  int const tid = static_cast<int>(threadIdx.x);
  if (tid >= kConsumerThreads) {
    return;
  }

  auto* raw_scores = reinterpret_cast<float*>(dynamic_storage + kTopkRawOffset);
  auto* staged_bias = reinterpret_cast<float*>(dynamic_storage + kTopkBiasOffset);

  if (tid < kRoutedExperts / 2) {
    auto const* global_pairs = reinterpret_cast<std::uint64_t const*>(
        router_logits + static_cast<std::size_t>(token) * kRoutedExperts);
    auto* shared_pairs = reinterpret_cast<std::uint64_t*>(raw_scores);
    shared_pairs[tid] = wait_router_logit_pair<Config>(global_pairs + tid);
  }
  if (tid < kRoutedExperts / 4) {
    int const expert_base = tid * 4;
    copy_16b_g2s(router_bias + expert_base, staged_bias + expert_base);
  }
  cutlass::arch::cp_async_fence();
  cutlass::arch::cp_async_wait<0>();
  consumer_group_sync<kConsumerThreads>();

  raw_scores[tid] = router_sigmoid(raw_scores[tid]);
  consumer_group_sync<kConsumerThreads>();

  if (tid < kWarpSize) {
    int const lane = tid & (kWarpSize - 1);
    std::uint32_t candidate_keys[kTopK];
#pragma unroll
    for (int slot = 0; slot < kTopK; ++slot) {
      int const expert = lane + slot * kWarpSize;
      float const rank = raw_scores[expert] + staged_bias[expert];
      candidate_keys[slot] = router_rank_key<Config>(rank);
    }

    float selected_raw = 0.0F;
#pragma unroll
    for (int pick = 0; pick < kTopK; ++pick) {
      std::uint32_t local_key = 0;
      std::uint32_t local_id = 0xffffffffU;
#pragma unroll
      for (int slot = 0; slot < kTopK; ++slot) {
        std::uint32_t const key = candidate_keys[slot];
        std::uint32_t const expert = static_cast<std::uint32_t>(lane + slot * kWarpSize);
        if (key > local_key || (key == local_key && expert < local_id)) {
          local_key = key;
          local_id = expert;
        }
      }

      std::uint32_t const winner_key = __reduce_max_sync(kFullWarpMask, local_key);
      std::uint32_t const eligible_id = local_key == winner_key ? local_id : 0xffffffffU;
      std::uint32_t const winner_id = __reduce_min_sync(kFullWarpMask, eligible_id);

      if (lane == pick) {
        selected_raw = raw_scores[winner_id];
        local_selected_indices[pick] = static_cast<std::int32_t>(winner_id);
      }
      if (lane == (winner_id & (kWarpSize - 1))) {
        candidate_keys[winner_id / kWarpSize] = 0;
      }
    }

    float score_sum = lane < kTopK ? selected_raw : 0.0F;
#pragma unroll
    for (int offset = kTopK / 2; offset > 0; offset >>= 1) {
      score_sum += __shfl_down_sync(kFullWarpMask, score_sum, offset, kTopK);
    }

    float normalizer = 0.0F;
    if (lane == 0 && score_sum > 0.0F) {
      normalizer = kRouteScale / score_sum;
    }
    normalizer = __shfl_sync(kFullWarpMask, normalizer, 0);

    float probability;
    if (lane < kTopK) {
      probability = selected_raw * normalizer;
      local_selected_probs[lane] = probability;
    }

    // The first CTA that completes this token's TopK claims the current
    // generation.  The persistent generation tag makes the election reusable
    // without clearing workspace between launches.  Usually late CTAs only
    // execute the relaxed load; CAS is limited to the first completion wave.
    std::uint32_t generation = 0U;
    int publish_global = 0;
    if (lane == 0) {
      generation =
          next_generation<Config>(load_relaxed_gpu_u32<Config>(ar_epoch_counters + counter));
      std::uint32_t const observed = load_relaxed_gpu_u32<Config>(topk_claim + token);
      if (observed != generation) {
        publish_global = atomicCAS(reinterpret_cast<unsigned int*>(topk_claim + token), observed,
                                   generation) == observed;
      }
    }
    publish_global = __shfl_sync(kFullWarpMask, publish_global, 0);

    if (lane < kTopK && publish_global) {
      published_indices[token * kTopK + lane] = local_selected_indices[lane];
      published_probs[token * kTopK + lane] = probability;
    }
    // Order the two coalesced TopK stores before lane 0 publishes readiness.
    __syncwarp(kFullWarpMask);
    if (publish_global && lane == 0) {
      store_release_gpu_u32(topk_ready + token, generation);
    }
  }
  consumer_group_sync<kConsumerThreads>();
}

template <class Config>
  requires(Config::kTokens == 8)
__device__ void select_topk(float const* router_logits, float const* router_bias, int token,
                            int first_selected_slot, int last_selected_slot,
                            std::int32_t* local_selected_indices, float* local_selected_probs,
                            std::int32_t* published_indices, float* published_probs,
                            bool publish_global, std::uint64_t* topk_handoff,
                            std::uint8_t* dynamic_storage) {
  int const tid = static_cast<int>(threadIdx.x);
  if (tid >= kConsumerThreads) {
    return;
  }

  auto* raw_scores = reinterpret_cast<float*>(dynamic_storage + kTopkRawOffset);
  auto* staged_bias = reinterpret_cast<float*>(dynamic_storage + kTopkBiasOffset);

  if (tid < kRoutedExperts / 2) {
    auto const* global_pairs = reinterpret_cast<std::uint64_t const*>(
        router_logits + static_cast<std::size_t>(token) * kRoutedExperts);
    auto* shared_pairs = reinterpret_cast<std::uint64_t*>(raw_scores);
    shared_pairs[tid] = wait_router_logit_pair<Config>(global_pairs + tid);
  }
  if (tid < kRoutedExperts / 4) {
    int const expert_base = tid * 4;
    copy_16b_g2s(router_bias + expert_base, staged_bias + expert_base);
  }
  cutlass::arch::cp_async_fence();
  cutlass::arch::cp_async_wait<0>();
  consumer_group_sync<kConsumerThreads>();

  raw_scores[tid] = router_sigmoid(raw_scores[tid]);
  consumer_group_sync<kConsumerThreads>();

  if (tid < kWarpSize) {
    int const lane = tid & (kWarpSize - 1);
    std::uint32_t candidate_keys[kTopK];
#pragma unroll
    for (int slot = 0; slot < kTopK; ++slot) {
      int const expert = lane + slot * kWarpSize;
      float const rank = raw_scores[expert] + staged_bias[expert];
      candidate_keys[slot] = router_rank_key<Config>(rank);
    }

    std::uint32_t selected_id = 0xffffffffU;
    float selected_raw = 0.0F;
#pragma unroll
    for (int pick = 0; pick < kTopK; ++pick) {
      std::uint32_t local_key = 0;
      std::uint32_t local_id = 0xffffffffU;
#pragma unroll
      for (int slot = 0; slot < kTopK; ++slot) {
        std::uint32_t const key = candidate_keys[slot];
        std::uint32_t const expert = static_cast<std::uint32_t>(lane + slot * kWarpSize);
        if (key > local_key || (key == local_key && expert < local_id)) {
          local_key = key;
          local_id = expert;
        }
      }

      std::uint32_t const winner_key = __reduce_max_sync(kFullWarpMask, local_key);
      std::uint32_t const eligible_id = local_key == winner_key ? local_id : 0xffffffffU;
      std::uint32_t const winner_id = __reduce_min_sync(kFullWarpMask, eligible_id);

      if (lane == pick) {
        selected_id = winner_id;
        selected_raw = raw_scores[winner_id];
        local_selected_indices[pick] = static_cast<std::int32_t>(winner_id);
        if (pick == first_selected_slot ||
            (last_selected_slot != first_selected_slot && pick == last_selected_slot)) {
          cutlass::arch::ClusterBarrier::arrive(topk_handoff);
        }
      }
      if (lane == (winner_id & (kWarpSize - 1))) {
        candidate_keys[winner_id / kWarpSize] = 0;
      }
    }

    float score_sum = lane < kTopK ? selected_raw : 0.0F;
#pragma unroll
    for (int offset = kTopK / 2; offset > 0; offset >>= 1) {
      score_sum += __shfl_down_sync(kFullWarpMask, score_sum, offset, kTopK);
    }

    float normalizer = 0.0F;
    if (lane == 0 && score_sum > 0.0F) {
      normalizer = kRouteScale / score_sum;
    }
    normalizer = __shfl_sync(kFullWarpMask, normalizer, 0);

    if (lane < kTopK) {
      local_selected_probs[lane] = selected_raw * normalizer;
      if (publish_global) {
        published_indices[token * kTopK + lane] = static_cast<std::int32_t>(selected_id);
        published_probs[token * kTopK + lane] = selected_raw * normalizer;
      }
    }
  }
  consumer_group_sync<kConsumerThreads>();
}

}  // namespace glm5::moe::detail
