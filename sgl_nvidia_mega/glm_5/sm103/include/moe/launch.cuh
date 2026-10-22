// SPDX-License-Identifier: Apache-2.0
#pragma once

// Host-only descriptors. Launch geometry and resource sizes come from cubin metadata.
#include <moe/args.h>
#include <common/host_helpers.cuh>

namespace glm5::moe::host {
inline void encode_expert_tma_descriptors(Glm5MegaMoeArgs* args) {
  using namespace ::glm5::host_detail;
  constexpr int h = 6144, inter = kFusedMoeLocalInter;
  encode_tma_3d(&args->up_gate_weight_tma, CU_TENSOR_MAP_DATA_TYPE_UINT8,
      args->up_gate_weight, h, inter, 257 * 2, h, inter * h,
      128, 16, 2, CU_TENSOR_MAP_SWIZZLE_128B);
  encode_tma_2d(&args->up_gate_scale_tma, CU_TENSOR_MAP_DATA_TYPE_FLOAT32,
      args->up_gate_scale, 48, 257 * 2 * inter / 128, 48 * sizeof(float),
      8, inter / 128 + 1, CU_TENSOR_MAP_SWIZZLE_NONE);
#if GLM5_MEGAMOE_TP_SIZE == 4
  encode_tma_3d(&args->down_weight_tma, CU_TENSOR_MAP_DATA_TYPE_UINT8,
      args->down_weight, 128, 4, 257 * h, 128, inter, 128, 2, 48,
      CU_TENSOR_MAP_SWIZZLE_128B);
#else
  encode_tma_2d(&args->down_weight_tma, CU_TENSOR_MAP_DATA_TYPE_UINT8,
      args->down_weight, 128, 257 * h * (inter / 128), 128,
      128, 48 * (inter / 128), CU_TENSOR_MAP_SWIZZLE_128B);
#endif
 }
}  // namespace glm5::moe::host
