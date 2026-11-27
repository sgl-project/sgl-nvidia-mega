// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// This file has been modified from its original version (a static FP8 store).
// Adapted from SGLang (Apache-2.0, SGLang Team):
// python/sglang/kernels/jit/csrc/elementwise/activation.cuh (act_and_mul_kernel,
// ActivationKernel::launch).

#include "elementwise/activation.cuh"

namespace sglang {

namespace prefill_act_fp8 {

struct Params {
  const void* __restrict__ input;   // [num_tokens, 2 * hidden_dim]: gate, then up
  void* __restrict__ out_q;         // E4M3 [num_tokens, hidden_dim]
  const float* __restrict__ scale;  // the consumer linear's per-tensor static input scale
  uint32_t hidden_dim;
  uint32_t num_tokens;
};

SGL_DEVICE float div_full_rcp_f32(float a) {
  float r;
  asm("div.full.f32 %0, 0f3F800000, %1;" : "=f"(r) : "f"(a));
  return r;
}

SGL_DEVICE uint16_t f32x2_to_e4m3x2_satfinite(float lo, float hi) {
  uint16_t v;
  asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(v) : "f"(hi), "f"(lo));
  return v;
}

template <typename T, bool kUsePDL, bool kFastMath>
__global__ void silu_and_mul_static_fp8_kernel(const __grid_constant__ Params params) {
  using namespace device;
  constexpr auto kVecSize = kMaxVecBytes / sizeof(T);
  static_assert(kVecSize % 2 == 0, "pairs of values are converted together");
  using vec_t = AlignedVector<T, kMaxVecBytes / sizeof(T)>;
  using qvec_t = AlignedVector<uint16_t, kVecSize / 2>;
  const auto num_vecs = params.hidden_dim / kVecSize;  // per token
  const auto tid = blockIdx.x * blockDim.x + threadIdx.x;
  const auto token_id = tid / num_vecs;

  if (token_id >= params.num_tokens) return;
  const auto offset = tid % num_vecs;
  const auto input_offset = token_id * (num_vecs * 2) + offset;
  PDLWaitPrimary<kUsePDL>();
  const auto input = assume_global(const_cast<void*>(params.input));
  vec_t gate, up;
  gate.load(input, input_offset);
  up.load(input, input_offset + num_vecs);
  const float inv_scale = div_full_rcp_f32(*params.scale);
  vec_t out;
#pragma unroll
  for (int i = 0; i < kVecSize; ++i) {
    float gate_f32 = device::cast<fp32_t>(gate[i]);
    float up_f32 = device::cast<fp32_t>(up[i]);
    out[i] = device::cast<T>(apply_activation_f32<ActivationKind::kSiLU, kFastMath>(gate_f32) * up_f32);
  }
  qvec_t q;
#pragma unroll
  for (int i = 0; i < kVecSize / 2; ++i) {
    q[i] = f32x2_to_e4m3x2_satfinite(
        device::cast<fp32_t>(out[2 * i]) * inv_scale, device::cast<fp32_t>(out[2 * i + 1]) * inv_scale);
  }
  q.store(params.out_q, tid);
  PDLTriggerSecondary<kUsePDL>();
}

}

template <typename T, bool kUsePDL, bool kFastMath>
struct PrefillActFp8Kernel {
  static constexpr auto kVecSize = device::kMaxVecBytes / sizeof(T);
  static constexpr auto kBlockSize = 256u;

  static void run(const tvm::ffi::TensorView input, const tvm::ffi::TensorView out_q, const tvm::ffi::TensorView scale) {
    using namespace host;

    auto N = SymbolicSize{"num_tokens"};
    auto D_in = SymbolicSize{"input_width"};
    auto D_out = SymbolicSize{"output_width"};
    auto device_ = SymbolicDevice{};
    device_.set_options<kDLGPU>();

    TensorMatcher({N, D_in}).with_strides({D_in, 1}).with_dtype<T>().with_device(device_).verify(input);
    TensorMatcher({N, D_out}).with_strides({D_out, 1}).with_device(device_).verify(out_q);
    RuntimeCheck(out_q.dtype().bits == 8, "out_q must be an 8-bit (E4M3) tensor");
    RuntimeCheck(is_type<float>(scale.dtype()) && scale.numel() == 1, "scale must be one float32");

    const auto hidden_size = static_cast<uint32_t>(D_out.unwrap());
    const auto num_tokens = static_cast<uint32_t>(N.unwrap());
    const auto device = device_.unwrap();
    if (num_tokens == 0) return;
    RuntimeCheck(hidden_size * 2 == D_in.unwrap(), "invalid activation dimension");
    RuntimeCheck(hidden_size % kVecSize == 0, "hidden size must be divisible by vector size");
    const auto num_total_items = static_cast<uint64_t>(num_tokens) * (hidden_size / kVecSize);
    RuntimeCheck(num_total_items <= std::numeric_limits<uint32_t>::max(), "too many items for 32-bit indexing");
    const auto num_blocks = div_ceil(static_cast<uint32_t>(num_total_items), kBlockSize);
    const auto params = prefill_act_fp8::Params{
        .input = input.data_ptr(),
        .out_q = out_q.data_ptr(),
        .scale = static_cast<const float*>(scale.data_ptr()),
        .hidden_dim = hidden_size,
        .num_tokens = num_tokens,
    };
    const auto kernel = prefill_act_fp8::silu_and_mul_static_fp8_kernel<T, kUsePDL, kFastMath>;
    LaunchKernel(num_blocks, kBlockSize, device).enable_pdl(kUsePDL)(kernel, params);
  }
};

}  // namespace sglang
