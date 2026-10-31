// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <cutlass/arch/barrier.h>
#include <cstdint>

namespace glm5 {
// Keep the native CUTLASS wait and its warp convergence unchanged.
__device__ __forceinline__ void wait_barrier(std::uint64_t const* barrier,
                                           std::uint32_t phase) {
  cutlass::arch::ClusterBarrier::wait(barrier, phase);
}
}  // namespace glm5
