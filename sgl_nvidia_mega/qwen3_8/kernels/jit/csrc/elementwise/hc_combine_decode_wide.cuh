#pragma once
#include <cooperative_groups.h>

#include "elementwise/hc_combine.cuh"  // SGLang's

namespace sglang {
// Expose the existing two stock stages separately; all arithmetic is unchanged.
struct HcCombineDecode {
  static void gate(tvm::ffi::TensorView normed, tvm::ffi::TensorView weight, tvm::ffi::TensorView partials) {
    using namespace host;
    SymbolicSize m{"rows"};
    SymbolicDevice dev;
    dev.set_options<kDLGPU>();
    TensorMatcher({m, 10240}).with_dtype<bf16_t>().with_device(dev).verify(normed);
    TensorMatcher({4, 10240}).with_dtype<bf16_t>().with_device(dev).verify(weight);
    TensorMatcher({m, 8, 4}).with_dtype<fp32_t>().with_device(dev).verify(partials);
    const HcCombineSplitParams params{
        .block_output = nullptr,
        .residual = nullptr,
        .normed_residual = normed.data_ptr(),
        .inject_weight = weight.data_ptr(),
        .output = nullptr,
        .partials = static_cast<float*>(partials.data_ptr())};
    LaunchKernel(dim3(m.unwrap(), 32, 1), 32, dev.unwrap())
        .enable_pdl(true)(hc_combine_gate_kernel<4, 2560, true, bf16_t>, params);
  }
  static void apply(
      tvm::ffi::TensorView block,
      tvm::ffi::TensorView residual,
      tvm::ffi::TensorView partials,
      tvm::ffi::TensorView out) {
    using namespace host;
    SymbolicSize m{"rows"};
    SymbolicDevice dev;
    dev.set_options<kDLGPU>();
    TensorMatcher({m, 2560}).with_dtype<bf16_t>().with_device(dev).verify(block);
    TensorMatcher({m, 10240}).with_dtype<bf16_t>().with_device(dev).verify(residual).verify(out);
    TensorMatcher({m, 8, 4}).with_dtype<fp32_t>().with_device(dev).verify(partials);
    const HcCombineSplitParams params{
        .block_output = block.data_ptr(),
        .residual = residual.data_ptr(),
        .normed_residual = nullptr,
        .inject_weight = nullptr,
        .output = out.data_ptr(),
        .partials = static_cast<float*>(partials.data_ptr())};
    LaunchKernel(dim3(m.unwrap(), 8, 1), 160, dev.unwrap())
        .enable_pdl(true)(hc_combine_apply_kernel<4, 2560, true, bf16_t>, params);
  }
};
struct HcCombineNormDecodeParams {
  const bf16_t* block;
  const bf16_t* residual;
  const float* partials;
  const bf16_t* weight;
  bf16_t* combined;
  bf16_t* normalized;
  float eps;
};

template <int kParts>
__global__
__launch_bounds__(160) void hc_combine_norm_decode_kernel(const HcCombineNormDecodeParams __grid_constant__ params) {
  using namespace device;
  using BFloat16x2 = packed_t<bf16_t>;
  using Storage = AlignedVector<BFloat16x2, 8>;
  constexpr uint32_t kThreads = 160;
  constexpr uint32_t kWarps = 5;
  static_assert(kThreads % kParts == 0);
  const uint32_t row = blockIdx.x / (4 * kParts);
  const uint32_t branch = (blockIdx.x / kParts) % 4;
  const uint32_t part = blockIdx.x % kParts;
  const uint32_t tid = threadIdx.x;
  const auto gmem = tile::Memory<Storage>::cta(kThreads);
  const auto residual = params.residual + (row * 4 + branch) * 2560;
  const auto block_output = params.block + row * 2560;
  const auto norm_weight = params.weight + branch * 2560;
  __shared__ float warp_sums[kWarpThreads];
  PDLWaitPrimary<true>();
  float total = 0.0f;
#pragma unroll
  for (uint32_t split = 0; split < 8; ++split)
    total += params.partials[(row * 8 + split) * 4 + branch];
  const float gate = 2.0f / (1.0f + math::exp(-total / 4));
  const auto residual_values = gmem.load(residual, 0);
  const auto block_values = gmem.load(block_output, 0);
  const auto weight_values = gmem.load(norm_weight, 0);
  Storage combined;
#pragma unroll
  for (uint32_t i = 0; i < 8; ++i) {
    const auto [rx, ry] = cast<fp32x2_t>(residual_values[i]);
    const auto [yx, yy] = cast<fp32x2_t>(block_values[i]);
    // Match the stock apply's FMA and its BF16 store before normalization.
    combined[i] = cast<BFloat16x2>(fp32x2_t{rx + gate * yx, ry + gate * yy});
  }
  float sum = 0.0f;
#pragma unroll
  for (uint32_t i = 0; i < 8; ++i) {
    const auto [x, y] = cast<fp32x2_t>(combined[i]);
    sum += x * x + y * y;
  }
  sum = warp::reduce_sum(sum);
  const auto warp_id = tid / kWarpThreads;
  warp_sums[warp_id] = sum;
  __syncthreads();
  if (warp_id == 0) {
    const auto local_sum = tid < kWarps ? warp_sums[tid] : 0.0f;
    sum = warp::reduce_sum(local_sum);
    warp_sums[tid] = math::rsqrt(sum / 2560 + params.eps);
  }
  __syncthreads();
  const float inv_rms = warp_sums[warp_id];
  if (tid >= part * (kThreads / kParts) && tid < (part + 1) * (kThreads / kParts)) {
    Storage normalized;
#pragma unroll
    for (uint32_t i = 0; i < 8; ++i) {
      const auto [x, y] = cast<fp32x2_t>(combined[i]);
      const auto [wx, wy] = cast<fp32x2_t>(weight_values[i]);
      normalized[i] = cast<BFloat16x2>(fp32x2_t{x * inv_rms * (1.0f + wx), y * inv_rms * (1.0f + wy)});
    }
    gmem.store(params.combined + (row * 4 + branch) * 2560, combined, 0);
    gmem.store(params.normalized + (row * 4 + branch) * 2560, normalized, 0);
  }
  PDLTriggerSecondary<true>();
}

template <int kParts>
struct HcCombineNormDecode {
  static void
  run(tvm::ffi::TensorView block,
      tvm::ffi::TensorView residual,
      tvm::ffi::TensorView partials,
      tvm::ffi::TensorView weight,
      tvm::ffi::TensorView combined,
      tvm::ffi::TensorView normalized,
      float eps) {
    using namespace host;
    SymbolicSize m{"rows"};
    SymbolicDevice dev;
    dev.set_options<kDLGPU>();
    TensorMatcher({m, 2560}).with_dtype<bf16_t>().with_device(dev).verify(block);
    TensorMatcher({m, 10240})
        .with_dtype<bf16_t>()
        .with_device(dev)
        .verify(residual)
        .verify(combined)
        .verify(normalized);
    TensorMatcher({10240}).with_dtype<bf16_t>().with_device(dev).verify(weight);
    TensorMatcher({m, 8, 4}).with_dtype<fp32_t>().with_device(dev).verify(partials);
    CHECK_HOST(m.unwrap() > 0 && m.unwrap() <= 4);
    const HcCombineNormDecodeParams p{
        static_cast<const bf16_t*>(block.data_ptr()),
        static_cast<const bf16_t*>(residual.data_ptr()),
        static_cast<const float*>(partials.data_ptr()),
        static_cast<const bf16_t*>(weight.data_ptr()),
        static_cast<bf16_t*>(combined.data_ptr()),
        static_cast<bf16_t*>(normalized.data_ptr()),
        eps};
    LaunchKernel(m.unwrap() * 4 * kParts, 160, dev.unwrap()).enable_pdl(true)(hc_combine_norm_decode_kernel<kParts>, p);
  }
};

// One row is genuinely partitioned over two CTAs, preserving the five stock
// warp input slices and both stock BF16 materialization boundaries.
__global__ __launch_bounds__(96)
void hc_combine_norm_decode_kernel_cluster2(const HcCombineNormDecodeParams __grid_constant__ params) {
  using namespace device;
  constexpr uint32_t kClusterSize = 2;
  constexpr uint32_t kLocalWarps = 3;
  constexpr uint32_t kOriginalWarps = 5;
  using BFloat16x2 = packed_t<bf16_t>;
  using Storage = AlignedVector<BFloat16x2, 8>;
  const auto cluster = cooperative_groups::this_cluster();
  const uint32_t rank = cluster.block_rank();
  const uint32_t lane = threadIdx.x % kWarpThreads;
  const uint32_t original_warp = rank * kLocalWarps + threadIdx.x / kWarpThreads;
  const uint32_t original_tid = original_warp * kWarpThreads + lane;
  const bool active = original_warp < kOriginalWarps;
  const uint32_t row_branch = blockIdx.x / kClusterSize;
  const uint32_t row = row_branch / 4;
  const uint32_t branch = row_branch % 4;
  const tile::Memory<Storage> gmem{original_tid, 160};
  __shared__ float rms_sums[kOriginalWarps];

  // Keep the existing early dependent trigger. Establish all DSMEM lifetimes
  // before any remote store, while the original producer may still be running.
  PDLTriggerSecondary<true>();
  cluster.sync();
  PDLWaitPrimary<true>();
  float total = 0.0f;
#pragma unroll
  for (uint32_t split = 0; split < 8; ++split)
    total += params.partials[(row * 8 + split) * 4 + branch];
  const float gate = 2.0f / (1.0f + math::exp(-total / 4));
  Storage residual_values, block_values, weight_values, gated_block;
  if (active) {
    residual_values = gmem.load(params.residual + (row * 4 + branch) * 2560);
    block_values = gmem.load(params.block + row * 2560);
    weight_values = gmem.load(params.weight + branch * 2560);
    gated_block = block_values;
  }
  Storage combined;
  float sum = 0.0f;
  if (active) {
#pragma unroll
    for (uint32_t i = 0; i < 8; ++i) {
      const auto [rx, ry] = cast<fp32x2_t>(residual_values[i]);
      const auto [yx, yy] = cast<fp32x2_t>(gated_block[i]);
      combined[i] = cast<BFloat16x2>(fp32x2_t{rx + gate * yx, ry + gate * yy});
    }
#pragma unroll
    for (uint32_t i = 0; i < 8; ++i) {
      const auto [x, y] = cast<fp32x2_t>(combined[i]);
      sum += x * x + y * y;
    }
  }
  sum = warp::reduce_sum(sum);
  if (active && lane < kClusterSize) {
    float* dst = cluster.map_shared_rank(&rms_sums[original_warp], lane);
    *dst = sum;
  }
  cluster.sync();
  // Identical five ordered warp partials + 27 zeros and stock XOR stages.
  const float local_sum = lane < kOriginalWarps ? rms_sums[lane] : 0.0f;
  sum = warp::reduce_sum(local_sum);
  const float lane_inv_rms = math::rsqrt(sum / 2560 + params.eps);
  const float inv_rms = __shfl_sync(0xffffffffu, lane_inv_rms, original_warp);
  if (active) {
    Storage normalized;
#pragma unroll
    for (uint32_t i = 0; i < 8; ++i) {
      const auto [x, y] = cast<fp32x2_t>(combined[i]);
      const auto [wx, wy] = cast<fp32x2_t>(weight_values[i]);
      normalized[i] = cast<BFloat16x2>(fp32x2_t{x * inv_rms * (1.0f + wx), y * inv_rms * (1.0f + wy)});
    }
    gmem.store(params.combined + (row * 4 + branch) * 2560, combined);
    gmem.store(params.normalized + (row * 4 + branch) * 2560, normalized);
  }
}


// Combine + next norm at 33..128 rows (SGLANG_HC_FUSED_COMBINE_MAX_ROWS): one row-branch over a
// two-CTA cluster, the same arithmetic as the kernel above.
struct HcCombineNormDecodeWide {
  static void
  run(tvm::ffi::TensorView block,
      tvm::ffi::TensorView residual,
      tvm::ffi::TensorView partials,
      tvm::ffi::TensorView weight,
      tvm::ffi::TensorView combined,
      tvm::ffi::TensorView normalized,
      float eps) {
    using namespace host;
    SymbolicSize m{"rows"};
    SymbolicDevice dev;
    dev.set_options<kDLGPU>();
    TensorMatcher({m, 2560}).with_dtype<bf16_t>().with_device(dev).verify(block);
    TensorMatcher({m, 10240})
        .with_dtype<bf16_t>()
        .with_device(dev)
        .verify(residual)
        .verify(combined)
        .verify(normalized);
    TensorMatcher({10240}).with_dtype<bf16_t>().with_device(dev).verify(weight);
    TensorMatcher({m, 8, 4}).with_dtype<fp32_t>().with_device(dev).verify(partials);
    CHECK_HOST(m.unwrap() > 0 && m.unwrap() <= 128);
    const HcCombineNormDecodeParams p{
        static_cast<const bf16_t*>(block.data_ptr()),
        static_cast<const bf16_t*>(residual.data_ptr()),
        static_cast<const float*>(partials.data_ptr()),
        static_cast<const bf16_t*>(weight.data_ptr()),
        static_cast<bf16_t*>(combined.data_ptr()),
        static_cast<bf16_t*>(normalized.data_ptr()),
        eps};
    LaunchKernel(m.unwrap() * 4 * 2, 96, dev.unwrap())
        .config({.use_pdl = true, .cluster_dim = dim3{2, 1, 1}})(hc_combine_norm_decode_kernel_cluster2, p);
  }
};
}  // namespace sglang
