#pragma once

#include <cuda.h>
#include <cuda_runtime_api.h>

#include <cstddef>
#include <cstdint>
#include <common/norm.h>

#ifndef GLM5_MEGAMLA_TOKENS
#define GLM5_MEGAMLA_TOKENS 4
#endif
#ifndef GLM5_MEGAMLA_TP_SIZE
#define GLM5_MEGAMLA_TP_SIZE 8
#endif

namespace glm5::mla {

#ifndef GLM5_MEGAMLA_FP8_KV_CACHE
#define GLM5_MEGAMLA_FP8_KV_CACHE 0
#endif

inline constexpr bool kFp8KvCache = GLM5_MEGAMLA_FP8_KV_CACHE != 0;
inline constexpr int kKvCacheElementBytes = kFp8KvCache ? 1 : 2;

inline constexpr int kTokens = GLM5_MEGAMLA_TOKENS;
static_assert(kTokens == 4 || kTokens == 8);
inline constexpr int kHidden = 6144;
inline constexpr int kTpWorldSize = GLM5_MEGAMLA_TP_SIZE;
static_assert(kTpWorldSize == 4 || kTpWorldSize == 8);
inline constexpr int kQueryHeads = 64 / kTpWorldSize;
inline constexpr int kQkNopeDim = 192;
inline constexpr int kRopeDim = 64;
inline constexpr int kKvLatentDim = 512;
inline constexpr int kSglangKvCacheDim = kKvLatentDim + kRopeDim;
inline constexpr int kQkDim = kKvLatentDim + kRopeDim;
inline constexpr int kValueDim = 512;
inline constexpr int kHeadDim = 256;
inline constexpr int kSelectedCapacity = 2048;

// A descriptor and scale view of an original FP8 matrix. No device weight
// page is allocated: every projection stages only its live shared tile.
struct alignas(alignof(CUtensorMap)) NativeWeightSource {
  CUtensorMap tensor_map;
  float const* scales;
  std::int32_t scale_row_stride;
  std::int32_t packed_scale_col_stride;
};

struct alignas(alignof(CUtensorMap)) Glm5MegaMlaArgs {
  void const* x;
  NormGain const* x_gamma;
  NormGain const* q_gamma;
  NormGain const* kv_gamma;
  NativeWeightSource q_a_weight;
  NativeWeightSource kv_a_weight;
  NativeWeightSource q_b_weight;
  NativeWeightSource k_absorb_weight;
  NativeWeightSource v_absorb_weight;
  NativeWeightSource o_proj_weight;

  std::int32_t const* selected_indices;
  std::int32_t const* selected_counts;
  void* kv_cache;
  void* pe_cache;
  void const* cache_positions;
  float const* rope_freqs;
  void const* rope_positions;

  void* workspace;
  std::size_t workspace_bytes;
  void const* residual;
  void* local_output;
  void* final_output;
  void* peer_workspaces[kTpWorldSize];
  std::uint32_t* phase_epoch_counters;

  std::int32_t rank;
  std::int32_t world_size;
  std::int32_t max_seq_len;
  std::int32_t kv_cache_row_stride;
  std::int32_t pe_cache_row_stride;
  std::int32_t cache_position_bytes;
  std::int32_t rope_position_bytes;
  std::int32_t rope_layout;
  std::int32_t reserved;

  // SM100 GATHER4 descriptor for the replicated latent KV cache.  Its element
  // type is selected at compile time together with the attention MMA operand.
  // descriptor is passed by value so the persistent kernel can issue TMA from
  // its four producer warps without a second global descriptor allocation.
  CUtensorMap kv_cache_tma;
};

static_assert(sizeof(Glm5MegaMlaArgs) <= 4096,
              "Glm5MegaMlaArgs must fit the portable 4-KiB CUDA parameter budget");

}  // namespace glm5::mla
