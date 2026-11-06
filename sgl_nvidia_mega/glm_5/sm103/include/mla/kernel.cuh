#pragma once

// Shared phase orchestration; each JIT module binds one compile-time Config.
#include <comm/oneshot_allreduce.cuh>
#include <mla/projection.cuh>
#include <mla/attention.cuh>

namespace glm5::mla::detail {

template <class Config>
struct MlaAllReduceConfig {
  static constexpr int kWorldSize = Config::kTpSize;
  static constexpr int kTokens = Config::kTokens;
  static constexpr int kConsumerThreads =
      ::glm5::mla::kConsumerThreads;
  static constexpr int kPayloadVectors = 6;
  static constexpr int kHidden = ::glm5::mla::kHidden;
  static constexpr std::size_t kCtaStride =
      ::glm5::mla::kArCtaStride;
  static constexpr std::size_t kPhaseStride =
      ::glm5::mla::kArPhaseStride;
  static constexpr std::size_t kSourceStride =
      ::glm5::mla::kArSourceStride;
  static constexpr std::size_t kTokenStride =
      ::glm5::mla::kArTokenStride;
  static constexpr auto kNamedBarrier = cutlass::arch::ReservedNamedBarriers::StreamkBarrier0;
};

template <class Config>
__device__ __forceinline__ void run_main_consumer(Glm5MegaMlaArgs const& args, int cta,
                                                  std::uint32_t generation,
                                                  std::uint8_t* static_storage,
                                                  std::uint8_t* dynamic_storage,
                                                  PipelineControl* pipeline) {
  int task = 0;

  if (cta < kProjXActiveCtas) {
    if constexpr (Config::kSharedProjX) {
      run_projx_normalized_consumer(args, cta, static_storage, dynamic_storage, pipeline, task);
    } else {
      run_proj_x_consumer<Config>(args, cta, static_storage, dynamic_storage, pipeline, task);
    }
    task += 6;
  }
  consumer_phase_arrive_and_wait(phase_state(args, Phase::kProjX), generation,
                                 contiguous_phase_contribution<kProjXActiveCtas>(cta), 256);

  if (cta < kProjQActiveCtas) {
    run_proj_q_consumer(args, cta, static_storage, dynamic_storage, pipeline, task);
    task += 2;
  }
  consumer_phase_arrive_and_wait(phase_state(args, Phase::kProjQ), generation,
                                 contiguous_phase_contribution<kProjQActiveCtas>(cta), 256);

  if (cta < kQAbsorbActiveCtas) {
    run_q_absorb_consumer<Config>(args, cta, dynamic_storage, pipeline, task);
    task += 1;
  }
  consumer_publish_q_absorb_and_wait_preattention(
      phase_state(args, Phase::kQAbsorb), phase_state(args, Phase::kRope),
      phase_state(args, Phase::kKv), generation,
      contiguous_phase_contribution<kQAbsorbActiveCtas>(cta));

  int const attention_task = task;
  run_attention_consumer<Config>(args, cta, pipeline, dynamic_storage, attention_task);
  task += 3 * Config::kAttentionPasses;

  auto* partial_async_state =
      workspace_pointer<std::uint64_t>(args, WorkspaceLayout::kAttentionPartialAsyncState);
  auto* partial_contribution =
      reinterpret_cast<std::uint32_t*>(static_storage + kAttentionPartialContributionOffset);
  consumer_partial_async_arrive_and_wait(partial_async_state, generation, cta,
                                         partial_contribution);
  int const combine_task = attention_task + 3 * (Config::kAttentionPasses - 1);
  bool const combine_cta = (cta % Config::kAttentionCtasPerToken) < kQueryHeads;
  if (combine_cta) {
    run_combine<Config>(args, cta, dynamic_storage, pipeline, combine_task);
  }

  // Combine CTAs release stage1 after split 15; other CTAs release it here.
  // Stage2 remains live until every active Combine consumer has finished.
  consumer_group_sync();
  if (!combine_cta) {
    consumer_release_stage(pipeline, combine_task + 1);
  }
  consumer_release_stage(pipeline, combine_task + 2);
  consumer_phase_arrive_and_wait(phase_state(args, Phase::kCombine), generation,
                                 combine_cta ? 256U / (kTokens * kQueryHeads) : 0U, 256);

  if (cta < kOAbsorbActiveCtas) {
    run_o_absorb_consumer<Config>(args, cta, dynamic_storage, pipeline, task);
    task += 1;
  }
  consumer_phase_arrive_and_wait(phase_state(args, Phase::kOAbsorb), generation,
                                 contiguous_phase_contribution<kOAbsorbActiveCtas>(cta), 256);

  run_o_proj_consumer<Config>(args, cta, static_storage, dynamic_storage, pipeline, task);
  auto* shared_local = reinterpret_cast<__nv_bfloat16*>(
      (Config::kStaticOProjTail ? static_storage : dynamic_storage) +
      (Config::kStaticOProjTail ? Config::kOProjLocalStaticOffset : kOProjReducedOffset));
  auto const* shared_residual = reinterpret_cast<__nv_bfloat16 const*>(
      dynamic_storage +
      (Config::kStaticOProjTail ? Config::kOProjResidualDynamicOffset : kOProjResidualOffset));
  if (args.world_size == 1) {
    run_single_rank_epilogue<Config>(args, cta, shared_local, shared_residual);
  } else {
    ::glm5::comm::allreduce_48_rows_and_add_residual<MlaAllReduceConfig<Config>>(
        args, generation, shared_local, shared_residual);
  }
}

template <class Config>
__device__ __forceinline__ void prefetch_selected_kv_l2(Glm5MegaMlaArgs const& args, int cta) {
  int const tid = static_cast<int>(threadIdx.x);
  if ((tid & 31) != 0) {
    return;
  }
  int const producer_warp = (tid >> 5) - kAttentionProducerWarpBegin;
#pragma unroll 1
  for (int pass = 0; pass < Config::kAttentionPasses; ++pass) {
    int const token = cta / Config::kAttentionCtasPerToken;
    int const split = (cta % Config::kAttentionCtasPerToken) * Config::kAttentionPasses + pass;
    int const selected_base = split * Config::kKvPerPass;
    int const selected_count = clamp_int(args.selected_counts[token], 0, kSelectedCapacity);
    int const valid_count = clamp_int(selected_count - selected_base, 0, Config::kKvPerPass);
    auto const* indices = args.selected_indices + token * kSelectedCapacity + selected_base;
#pragma unroll 1
    for (int row_group = 0; row_group < kGatherRowGroupsPerProducer; ++row_group) {
      int const first_row =
          row_group * (kProducerWarps * kGatherRows) + producer_warp * kGatherRows;
      if (first_row >= valid_count) {
        continue;
      }
      // selected_indices only promises int32 alignment.
      int4 rows{indices[first_row], indices[first_row + 1], indices[first_row + 2],
                indices[first_row + 3]};
      if (first_row + 1 >= valid_count) rows.y = 0;
      if (first_row + 2 >= valid_count) rows.z = 0;
      if (first_row + 3 >= valid_count) rows.w = 0;
#pragma unroll
      for (int chunk = 0; chunk < kGatherChunks; ++chunk) {
        cute::SM100_TMA_LOAD_2D_GATHER4::PREFETCH::copy(&args.kv_cache_tma, chunk * kGatherColumns,
                                                        rows.x, rows.y, rows.z, rows.w);
      }
    }
  }
}

template <class Config>
__device__ __forceinline__ void run_main_producer(Glm5MegaMlaArgs const& args, int cta,
                                                  std::uint32_t generation,
                                                  std::uint8_t* dynamic_storage,
                                                  PipelineControl* pipeline) {
  int task = 0;

  if (cta < kProjXActiveCtas) {
    ProjXWeightAccessor const weight{args, cta};
    produce_projection_pages<6>(weight, pipeline, dynamic_storage, task);
    task += 6;
  }

  if (cta < kProjQActiveCtas) {
    ProjQWeightAccessor const weight{args, cta};
    produce_projection_pages<2>(weight, pipeline, dynamic_storage, task);
    task += 2;

    QAbsorbWeightAccessor const qabs_weight{args, cta};
    produce_projection_pages<1>(qabs_weight, pipeline, dynamic_storage, task);
    task += 1;
  }

  // GATHER4 only consumes selected indices and kv_cache.  Start it as soon as
  // KV is visible; the consumer independently retains the full RoPE/KV/QAbs
  // pre-attention join before it reads q_pe/k_pe and consumes the KV stages.
  prefetch_selected_kv_l2<Config>(args, cta);
  producer_phase_wait(phase_state(args, Phase::kKv), generation, 256);
  run_attention_producer<Config>(args, cta, pipeline, dynamic_storage, task);
  task += 3 * Config::kAttentionPasses;

  if (cta < kOAbsorbActiveCtas) {
    OAbsorbWeightAccessor const weight{args, cta};
    produce_projection_pages<1>(weight, pipeline, dynamic_storage, task);
    task += 1;
  }

  OProjWeightAccessor const oproj_weight{args, cta};
  produce_projection_pages<kOProjPages>(oproj_weight, pipeline, dynamic_storage, task);
}

template <class Config>
__device__ __forceinline__ void run_megamla(Glm5MegaMlaArgs const& args,
                                            std::uint8_t* static_storage,
                                            std::uint8_t* dynamic_storage) {
  static_assert(Config::kTokens == kTokens && Config::kTpSize == kTpWorldSize);
  int const cta = static_cast<int>(blockIdx.x);
  auto* epoch_counter = args.phase_epoch_counters + cta;
  auto* generation_storage =
      reinterpret_cast<std::uint32_t*>(static_storage + kGenerationScratchOffset);
  if (threadIdx.x == 0) {
    *generation_storage = load_relaxed_gpu_u32(epoch_counter) + 1U;
  }

  auto* pipeline = reinterpret_cast<PipelineControl*>(static_storage + kAttentionBarrierOffset);
  initialize_pipeline(pipeline);
  // One CTA barrier publishes the generation and all four pipeline barriers
  // before the main and auxiliary paths diverge.
  __syncthreads();
  std::uint32_t const generation = *generation_storage;

  if (cta >= kMainCtas) {
    if (cta >= kKvAuxCtaBegin && cta < kKvAuxCtaEnd) {
      if (threadIdx.x < kConsumerThreads) {
        if constexpr (Config::kSharedProjX) {
          precompute_projx_normalized_input(args, cta - kKvAuxCtaBegin, generation, static_storage,
                                            dynamic_storage);
        }
        if (threadIdx.x == 0) {
          auto* proj_x = phase_state(args, Phase::kProjX);
          prepare_phase(proj_x, generation);
          wait_phase_count(proj_x, generation, 256);
        }
        consumer_group_sync();
        run_kv_norm<Config>(args, cta - kKvAuxCtaBegin, dynamic_storage);
        if (threadIdx.x == 0) {
          phase_release_add(phase_state(args, Phase::kKv), generation, 256 / Config::kTokens);
        }
      }
    } else if (cta >= kRopeAuxCtaBegin && cta < kRopeAuxCtaEnd) {
      if (threadIdx.x < kConsumerThreads) {
        if (threadIdx.x == 0) {
          auto* proj_q = phase_state(args, Phase::kProjQ);
          prepare_phase(proj_q, generation);
          wait_phase_count(proj_q, generation, 256);
        }
        consumer_group_sync();
        run_rope<Config>(args, cta - kRopeAuxCtaBegin);
        consumer_group_sync();
        if (threadIdx.x == 0) {
          phase_release_add(phase_state(args, Phase::kRope), generation, 256 / Config::kTokens);
        }
      }
    }
    if (threadIdx.x == 0) {
      // Every CTA owns one epoch word.  Publish it at the tail so the 8 words
      // sharing each 32-byte L2 sector do not contend on the startup path.
      store_relaxed_gpu_u32(epoch_counter, generation);
    }
    return;
  }

  configure_warp_specialized_registers();
  if (threadIdx.x < kConsumerThreads) {
    run_main_consumer<Config>(args, cta, generation, static_storage, dynamic_storage, pipeline);
  } else {
    run_main_producer<Config>(args, cta, generation, dynamic_storage, pipeline);
  }
  if (threadIdx.x == 0) {
    store_relaxed_gpu_u32(epoch_counter, generation);
  }
}
}  // namespace glm5::mla::detail
