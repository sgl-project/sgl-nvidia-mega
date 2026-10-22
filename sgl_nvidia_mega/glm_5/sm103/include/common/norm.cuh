// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <common/norm.h>
#include <cuda_runtime.h>

namespace glm5 {
__device__ __forceinline__ float2 load_norm2(NormGain const* pointer) {
#if GLM5_NORM_BF16
  return __bfloat1622float2(*reinterpret_cast<__nv_bfloat162 const*>(pointer));
#else
  return *reinterpret_cast<float2 const*>(pointer);
#endif
}

__device__ __forceinline__ float4 load_norm4(NormGain const* pointer) {
#if GLM5_NORM_BF16
  float2 const low = load_norm2(pointer);
  float2 const high = load_norm2(pointer + 2);
  return make_float4(low.x, low.y, high.x, high.y);
#else
  return *reinterpret_cast<float4 const*>(pointer);
#endif
}
}  // namespace glm5
