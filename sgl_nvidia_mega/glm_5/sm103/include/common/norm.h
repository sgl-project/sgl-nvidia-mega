// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <cuda_bf16.h>

#ifndef GLM5_NORM_BF16
#define GLM5_NORM_BF16 0
#endif

namespace glm5 {
#if GLM5_NORM_BF16
using NormGain = __nv_bfloat16;
inline constexpr char kNormDtype[] = "bfloat16";
#else
using NormGain = float;
inline constexpr char kNormDtype[] = "float32";
#endif
}  // namespace glm5
