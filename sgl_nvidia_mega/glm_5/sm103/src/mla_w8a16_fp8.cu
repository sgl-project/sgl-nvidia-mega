#include <cuda.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cutlass/arch/barrier.h>
#include <cutlass/arch/memory_sm80.h>
#include <cutlass/arch/reg_reconfig.h>
#include <math_constants.h>

#include <algorithm>
#include <cfloat>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cute/arch/copy_sm100.hpp>
#include <cute/arch/copy_sm100_tma.hpp>
#include <cute/arch/copy_sm90_tma.hpp>

#include <mla/kernel.cuh>


namespace glm5::mla {
extern "C" __global__ __launch_bounds__(kThreads, 1)
void GLM5_KERNEL_NAME(__grid_constant__ Glm5MegaMlaArgs const args) {
  __shared__ __align__(1024) std::uint8_t static_storage[kStaticSmemBytes];
  extern __shared__ __align__(1024) std::uint8_t dynamic_storage[];
  detail::run_megamla<detail::MlaConfig<kTokens, kTpWorldSize>>(
      args, static_storage, dynamic_storage);
}
}  // namespace glm5::mla

#include <common/metadata.cuh>

namespace glm5::mla::detail {
using Config = MlaConfig<kTokens, kTpWorldSize>;
extern "C" __device__ __constant__ ::glm5::KernelMetadata glm5_metadata = {
    1, Config::kTokens, Config::kTpSize, Config::kWorkspaceRequiredBytes,
    Config::kEpochCount, Config::kPeerBytes, Config::kWorkspaceFill,
    Config::kRequiredSms, MEGAMLA_USE_NATIVE_BF16X2_E4M3X2_CVT,
    kGridCtas, kThreads, kDynamicSmemBytes,
    sizeof(Glm5MegaMlaArgs), alignof(Glm5MegaMlaArgs)};
}  // namespace glm5::mla::detail
