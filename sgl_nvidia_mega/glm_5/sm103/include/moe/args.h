#pragma once

#include <cuda.h>
#include <cuda_runtime_api.h>

#include <cstddef>
#include <cstdint>
#include <common/norm.h>

namespace glm5::moe {

#ifndef GLM5_MEGAMOE_SEQ_LEN
#define GLM5_MEGAMOE_SEQ_LEN 4
#endif

#ifndef GLM5_MEGAMOE_TP_SIZE
#define GLM5_MEGAMOE_TP_SIZE 8
#endif

inline constexpr int kFusedMoeSeqLen = GLM5_MEGAMOE_SEQ_LEN;
inline constexpr int kFusedMoeTpSize = GLM5_MEGAMOE_TP_SIZE;
inline constexpr int kFusedMoeWorldSize = kFusedMoeTpSize;
inline constexpr int kFusedMoeMaxWorldSize = 8;
inline constexpr int kFusedMoeGlobalInter = 2048;
inline constexpr int kFusedMoeLocalInter = kFusedMoeGlobalInter / kFusedMoeTpSize;
inline constexpr int kFusedMoeRemoteSources = kFusedMoeWorldSize - 1;
inline constexpr int kFusedMoeTp8WorldSize = 8;
static_assert(kFusedMoeSeqLen == 4 || kFusedMoeSeqLen == 8);
static_assert(kFusedMoeTpSize == 4 || kFusedMoeTpSize == 8);
static_assert(kFusedMoeGlobalInter % kFusedMoeTpSize == 0);

// Native GLM5 FP8 layout: routed experts are [0, 255], shared expert is 256.
struct alignas(alignof(CUtensorMap)) Glm5MegaMoeArgs {
  void const* x;
  NormGain const* rms_gamma;
  void const* router_weight;
  float const* router_bias;
  void const* up_gate_weight;
  float const* up_gate_scale;
  void const* down_weight;
  float const* down_scale;
  void* workspace;
  std::size_t workspace_bytes;
  // Optional rank-local BF16 partial. Set to nullptr on the production path
  // to skip the diagnostic global-memory writeback.
  void* local_output;
  std::int32_t* selected_indices;
  float* selected_probs;
  void const* residual;
  void* final_output;
  void* peer_workspaces[kFusedMoeMaxWorldSize];
  std::uint32_t* ar_epoch_counters;
  std::uint64_t epoch;
  std::int32_t rank;
  std::int32_t world_size;
  CUtensorMap up_gate_weight_tma;
  CUtensorMap up_gate_scale_tma;
  CUtensorMap down_weight_tma;
};

static_assert(sizeof(Glm5MegaMoeArgs) <= 1024,
              "Glm5MegaMoeArgs must fit the CUDA kernel parameter budget");

}  // namespace glm5::moe
