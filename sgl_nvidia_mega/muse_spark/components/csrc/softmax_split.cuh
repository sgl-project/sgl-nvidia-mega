// SPDX-License-Identifier: Apache-2.0
/*
 * Copyright (c) 2024 by FlashInfer team.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
// Derived from FlashInfer (https://github.com/flashinfer-ai/flashinfer, Apache License 2.0; see
// ../include/LICENSE.flashinfer and ../include/NOTICE.flashinfer): include/flashinfer/sampling.cuh (OnlineSoftmaxReduceKernel and the
// vocabulary-splitting branch of OnlineSoftmax), FlashInfer 0.7.1.
// Modified by NVIDIA: the normalization pass is split across blocks along the vocabulary; every block repeats the
// row's merge of the slice partials (same code, same block size) and writes only its own column range. The slice
// pass is FlashInfer's own OnlineSoftmaxMapKernel, launched as FlashInfer launches it.

#pragma once

#include <sgl_kernel/tensor.h>
#include <sgl_kernel/utils.cuh>

#include <flashinfer/sampling.cuh>

#include <tvm/ffi/optional.h>

#include <cstdint>
#include <numeric>

namespace sglang {
namespace muse_vr {

using flashinfer::ceil_div;
using flashinfer::vec_t;
using flashinfer::sampling::Float2SoftmaxReduceOp;
using flashinfer::sampling::OnlineSoftmaxMapKernel;
using flashinfer::sampling::OnlineSoftmaxTempStorage;
using flashinfer::sampling::PartialSoftmaxResult;

template <uint32_t BLOCK_THREADS, uint32_t VEC_SIZE, typename DType>
__global__ FLASHINFER_SAMPLING_LAUNCH_BOUNDS(BLOCK_THREADS) void SplitOnlineSoftmaxReduceKernel(
    DType* logits, DType* output, PartialSoftmaxResult* partial_results, DType* temperature_arr,
    float temperature_val, uint32_t d, uint32_t num_slices, uint32_t cols_per_block) {
  const uint32_t bx = blockIdx.x;
  const uint32_t tx = threadIdx.x;
  float temperature = temperature_arr == nullptr ? temperature_val : temperature_arr[bx];
  const float inv_temp = (temperature == 0.f) ? 0.f : 1.f / temperature;

  using TempStorage = OnlineSoftmaxTempStorage<BLOCK_THREADS>;
  extern __shared__ __align__(alignof(TempStorage)) uint8_t smem[];
  auto& temp_storage = reinterpret_cast<TempStorage&>(smem);

  const Float2SoftmaxReduceOp reduce_op;

  float2 thread_aggregate = make_float2(-cuda::std::numeric_limits<float>::infinity(), 0.0f);

#if (__CUDACC_VER_MAJOR__ >= 12 && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
  asm volatile("griddepcontrol.wait;");
#endif

  for (uint32_t i = tx; i < num_slices; i += BLOCK_THREADS) {
    PartialSoftmaxResult partial = partial_results[bx * num_slices + i];
    float2 partial_pair = make_float2(partial.max_val, partial.denominator);
    thread_aggregate = reduce_op(thread_aggregate, partial_pair);
  }

  float2 block_result = cub::BlockReduce<float2, BLOCK_THREADS>(temp_storage.block_prim.reduce_pair)
                            .Reduce(thread_aggregate, reduce_op);

  if (tx == 0) {
    temp_storage.shared_state.max_val = block_result.x;
    temp_storage.shared_state.denominator = block_result.y;
  }
  __syncthreads();

  const float final_max = temp_storage.shared_state.max_val;
  const float inv_denominator = 1.0f / temp_storage.shared_state.denominator;

  // This block's column range of the row.
  const uint32_t col_begin = blockIdx.y * cols_per_block;
  const uint32_t col_end = min(d, col_begin + cols_per_block);
  const uint32_t cols = col_end > col_begin ? col_end - col_begin : 0;

  vec_t<DType, VEC_SIZE> logits_vec;
  vec_t<DType, VEC_SIZE> prob_vec;

  for (uint32_t i = 0; i < ceil_div(cols, BLOCK_THREADS * VEC_SIZE); ++i) {
    const uint32_t col = col_begin + (i * BLOCK_THREADS + tx) * VEC_SIZE;
    logits_vec.fill(-cuda::std::numeric_limits<DType>::infinity());

    if (col < col_end) {
      logits_vec.cast_load(logits + bx * d + col);
    }

#pragma unroll
    for (uint32_t j = 0; j < VEC_SIZE; ++j) {
      logits_vec[j] = __fmul_rn(static_cast<float>(logits_vec[j]), inv_temp);
      float p = __expf(static_cast<float>(logits_vec[j]) - final_max) * inv_denominator;
      prob_vec[j] = static_cast<DType>(p);
    }

    if (col < col_end) {
      prob_vec.cast_store(output + bx * d + col);
    }
  }
#if (__CUDACC_VER_MAJOR__ >= 12 && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
  asm volatile("griddepcontrol.launch_dependents;");
#endif
}

// probs = softmax(logits / temperature) for [batch, d] fp32 rows, with FlashInfer's slice pass and the split
// normalization pass. Only FlashInfer's vocabulary-splitting regime is handled (the caller checks it).
// workspace: at least batch * ceil(d / 8192) * sizeof(PartialSoftmaxResult) bytes.
struct SplitSoftmax {
  static constexpr uint32_t kSliceSize = 8192;  // FlashInfer OnlineSoftmax DEFAULT_SLICE_SIZE

  static void run(
      const tvm::ffi::TensorView workspace,
      const tvm::ffi::TensorView logits,
      const tvm::ffi::TensorView output,
      const tvm::ffi::Optional<tvm::ffi::TensorView> temperature_arr,
      double temperature_val,
      int64_t cols_per_block,
      bool enable_pdl) {
    using namespace host;
    SymbolicSize batch{"batch"}, vocab{"vocab"};
    SymbolicDevice device;
    TensorMatcher({batch, vocab}).with_dtype<float>().with_device<kDLCUDA>(device).verify(logits).verify(output);
    const uint32_t b = static_cast<uint32_t>(batch.unwrap());
    const uint32_t d = static_cast<uint32_t>(vocab.unwrap());
    float* temp_ptr = nullptr;
    if (temperature_arr.has_value()) {
      TensorMatcher({batch}).with_dtype<float>().with_device(device).verify(temperature_arr.value());
      temp_ptr = static_cast<float*>(temperature_arr.value().data_ptr());
    }
    if (b == 0) return;
    const uint32_t num_slices = ceil_div(d, kSliceSize);
    CHECK_HOST(static_cast<int64_t>(workspace.numel() * workspace.dtype().bits / 8) >=
               static_cast<int64_t>(b) * num_slices * static_cast<int64_t>(sizeof(PartialSoftmaxResult)));
    CHECK_HOST(cols_per_block > 0 && cols_per_block % 16 == 0);
    auto* partials = static_cast<PartialSoftmaxResult*>(workspace.data_ptr());
    float* in = static_cast<float*>(logits.data_ptr());
    float* out = static_cast<float*>(output.data_ptr());
    const float tval = static_cast<float>(temperature_val);
    const uint32_t cpb = static_cast<uint32_t>(cols_per_block);
    const uint32_t col_blocks = ceil_div(d, cpb);

    const uint32_t vec_size = std::gcd(16 / sizeof(float), d);
    auto compute_capacity = flashinfer::GetCudaComputeCapability();
    DISPATCH_COMPUTE_CAP_NUM_THREADS(compute_capacity, BLOCK_THREADS, {
      DISPATCH_ALIGNED_VEC_SIZE(vec_size, VEC_SIZE, {
        const size_t smem_size = sizeof(OnlineSoftmaxTempStorage<BLOCK_THREADS>);
        auto map_kernel = OnlineSoftmaxMapKernel<BLOCK_THREADS, VEC_SIZE, float>;
        auto reduce_kernel = SplitOnlineSoftmaxReduceKernel<BLOCK_THREADS, VEC_SIZE, float>;
        CHECK_CUDA(cudaFuncSetAttribute(map_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size));
        CHECK_CUDA(
            cudaFuncSetAttribute(reduce_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size));
        LaunchKernel(dim3(b, num_slices), dim3(BLOCK_THREADS), device.unwrap(), smem_size)
            .enable_pdl(enable_pdl)(map_kernel, in, partials, temp_ptr, tval, d, num_slices);
        LaunchKernel(dim3(b, col_blocks), dim3(BLOCK_THREADS), device.unwrap(), smem_size)
            .enable_pdl(enable_pdl)(reduce_kernel, in, out, partials, temp_ptr, tval, d, num_slices, cpb);
      });
    });
  }
};

}  // namespace muse_vr
}  // namespace sglang
