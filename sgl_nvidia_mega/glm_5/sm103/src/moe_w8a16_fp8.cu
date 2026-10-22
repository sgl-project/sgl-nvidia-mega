#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cutlass/arch/barrier.h>
#include <cutlass/arch/memory_sm80.h>
#include <cutlass/float8.h>

#include <cstddef>
#include <cstdint>
#include <cute/arch/copy_sm100.hpp>
#include <cute/arch/copy_sm90_tma.hpp>

#include <common/hmma_b16.cuh>
#include <moe/pipeline.cuh>
#include <comm/oneshot_allreduce.cuh>


#define GLM5_MEGAMOE_KERNEL_NAME GLM5_KERNEL_NAME
#include <moe/kernel.cuh>
#undef GLM5_MEGAMOE_KERNEL_NAME

#if GLM5_MEGAMOE_SEQ_LEN == 4
namespace glm5::moe {
using Glm5Config = detail::ActiveMoeConfig;
extern "C" __global__
#if GLM5_MEGAMOE_TP_SIZE == 4
__launch_bounds__(Glm5Config::kThreads)
#else
__launch_bounds__(Glm5Config::kThreads, 1)
#endif
void GLM5_KERNEL_NAME(__grid_constant__ Glm5MegaMoeArgs const args) {
  __shared__ __align__(16) std::uint8_t static_storage[Glm5Config::kStaticSmemBytes];
  extern __shared__ __align__(16) std::uint8_t dynamic_storage[];
  detail::run_megamoe<Glm5Config>(
      args, reinterpret_cast<detail::SharedStorage*>(static_storage), dynamic_storage);
}
}  // namespace glm5::moe
#endif

#include <common/metadata.cuh>

namespace glm5::moe::detail {
using Config = ActiveMoeConfig;
extern "C" __device__ __constant__ ::glm5::KernelMetadata glm5_metadata = {
    1, Config::kTokens, Config::kTpSize, Config::kWorkspaceRequiredBytes,
    Config::kEpochCount, Config::kPeerBytes, Config::kWorkspaceFill,
    Config::kRequiredSms, MEGAMOE_USE_NATIVE_E4M3X2_TO_BF16X2,
    Config::kGridCtas, Config::kThreads, Config::kDynamicSmemBytes,
    sizeof(Glm5MegaMoeArgs), alignof(Glm5MegaMoeArgs)};
}  // namespace glm5::moe::detail
