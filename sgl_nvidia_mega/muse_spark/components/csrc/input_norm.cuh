// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
//
// The Muse Spark input boundary around one PyTorch reduction. SGLang runs, after the embedding all-reduce, the
// embedding RMSNorm (no weight; PyTorch's row-wise RMSNorm kernel on bf16), the cast of its output to the fp32
// stream, and the first layer's input RMSNorm on the fp32 stream with a bf16 weight (PyTorch's composite form:
// x * x, row mean, + eps, rsqrt, x * r, * weight, cast to bf16) as ten kernels. Here:
//
//   embed_stream:  y = bf16(rsqrt(sum(x * x) / H + eps_embed) * x)   (PyTorch's row RMSNorm summation order)
//                  stream = float(y);  sq = stream * stream           (one kernel)
//   (the row mean of sq stays PyTorch's reduction kernel)
//   input_finish:  out = bf16((stream * rsqrt(mean + eps_in)) * float(weight))
//   stream_sq:     stream = float(x);  sq = stream * stream           (when the embeddings arrive already normalized)
//
// with the roundings of the PyTorch operations they replace (tests/test_input_norm.py: bit-identical outputs).
// embed_stream uses one block of 128 threads (32 x 4) per row with the summation order of PyTorch's row-wise RMSNorm
// (per-thread fused multiply-add over the 4-element vectors t, t + 128, ..., shuffle-down tree per warp, warp sums
// added pairwise).

#pragma once

#include <sgl_kernel/tensor.h>
#include <sgl_kernel/utils.h>

#include <sgl_kernel/utils.cuh>

#include <tvm/ffi/container/tensor.h>

#include <cuda_bf16.h>
#include <algorithm>
#include <cstdint>

namespace muse_vr::input_norm {

using namespace sglang;

constexpr int kThreads = 128;
constexpr int kVec = 4;

SGL_DEVICE float bf(uint32_t b) {
  return __uint_as_float(b << 16);
}

SGL_DEVICE uint32_t rb(float v) {
  return static_cast<uint32_t>(__bfloat16_as_ushort(__float2bfloat16_rn(v)));
}

SGL_DEVICE float row_sum(float sq, float* buf) {
  const int x = threadIdx.x % 32, y = threadIdx.x / 32;
#pragma unroll
  for (int off = 16; off > 0; off >>= 1) sq = __fadd_rn(sq, __shfl_down_sync(0xffffffffu, sq, off));
  if (x == 0 && y >= 2) buf[y - 2] = sq;
  __syncthreads();
  if (x == 0 && y < 2) sq = __fadd_rn(sq, buf[y]);
  __syncthreads();
  if (x == 0 && y == 1) buf[0] = sq;
  __syncthreads();
  if (x == 0 && y == 0) buf[2] = __fadd_rn(sq, buf[0]);
  __syncthreads();
  return buf[2];
}

struct EmbedParams {
  const uint16_t* x;  // [M, H] bf16
  float* stream;      // [M, H] fp32
  float* sq;          // [M, H] fp32
  float eps;
  int hidden;
};

template <int kPerThread, bool kUsePDL>
__global__ __launch_bounds__(kThreads) void embed_stream_kernel(const __grid_constant__ EmbedParams p) {
  __shared__ float buf[4];
  const int t = threadIdx.x;
  const int64_t base = static_cast<int64_t>(blockIdx.x) * p.hidden;
  device::PDLWaitPrimary<kUsePDL>();
  uint2 xv[kPerThread];
  float s = 0.f;
#pragma unroll
  for (int k = 0; k < kPerThread; ++k) {
    xv[k] = *reinterpret_cast<const uint2*>(p.x + base + (t + k * kThreads) * kVec);
#pragma unroll
    for (int e = 0; e < kVec; ++e) {
      const float v = bf(((e < 2 ? xv[k].x : xv[k].y) >> ((e % 2) * 16)) & 0xffffu);
      s = __fmaf_rn(v, v, s);
    }
  }
  const float r = rsqrtf(__fadd_rn(__fdiv_rn(row_sum(s, buf), static_cast<float>(p.hidden)), p.eps));
#pragma unroll
  for (int k = 0; k < kPerThread; ++k) {
    float y[kVec], q[kVec];
#pragma unroll
    for (int e = 0; e < kVec; ++e) {
      const float v = bf(((e < 2 ? xv[k].x : xv[k].y) >> ((e % 2) * 16)) & 0xffffu);
      y[e] = bf(rb(__fmul_rn(r, v)));
      q[e] = __fmul_rn(y[e], y[e]);
    }
    const int64_t o = base + (t + k * kThreads) * kVec;
    *reinterpret_cast<float4*>(p.stream + o) = make_float4(y[0], y[1], y[2], y[3]);
    *reinterpret_cast<float4*>(p.sq + o) = make_float4(q[0], q[1], q[2], q[3]);
  }
  device::PDLTriggerSecondary<kUsePDL>();
}

struct StreamParams {
  const uint16_t* x;  // [M, H] bf16
  float* stream;      // [M, H] fp32
  float* sq;          // [M, H] fp32
  int64_t n;          // M * H
};

template <bool kUsePDL>
__global__ __launch_bounds__(256) void stream_sq_kernel(const __grid_constant__ StreamParams p) {
  device::PDLWaitPrimary<kUsePDL>();
  for (int64_t v = blockIdx.x * int64_t(blockDim.x) + threadIdx.x; v < p.n / kVec; v += int64_t(gridDim.x) * blockDim.x) {
    const uint2 w = *reinterpret_cast<const uint2*>(p.x + v * kVec);
    float y[kVec], q[kVec];
#pragma unroll
    for (int e = 0; e < kVec; ++e) {
      y[e] = bf(((e < 2 ? w.x : w.y) >> ((e % 2) * 16)) & 0xffffu);
      q[e] = __fmul_rn(y[e], y[e]);
    }
    *reinterpret_cast<float4*>(p.stream + v * kVec) = make_float4(y[0], y[1], y[2], y[3]);
    *reinterpret_cast<float4*>(p.sq + v * kVec) = make_float4(q[0], q[1], q[2], q[3]);
  }
  device::PDLTriggerSecondary<kUsePDL>();
}

struct FinishParams {
  const float* stream;     // [M, H]
  const float* mean;       // [M]
  const uint16_t* weight;  // [H] bf16
  uint16_t* out;           // [M, H] bf16
  float eps;
  int hidden;
};

template <bool kUsePDL>
__global__ __launch_bounds__(256) void input_finish_kernel(const __grid_constant__ FinishParams p) {
  device::PDLWaitPrimary<kUsePDL>();
  const int64_t row = blockIdx.y;
  const float r = rsqrtf(__fadd_rn(p.mean[row], p.eps));
  for (int v = blockIdx.x * blockDim.x + threadIdx.x; v < p.hidden / kVec; v += gridDim.x * blockDim.x) {
    const float4 s = *reinterpret_cast<const float4*>(p.stream + row * p.hidden + v * kVec);
    const uint2 w = *reinterpret_cast<const uint2*>(p.weight + v * kVec);
    const float sv[4] = {s.x, s.y, s.z, s.w};
    uint32_t o[4];
#pragma unroll
    for (int e = 0; e < kVec; ++e) {
      const float wv = bf(((e < 2 ? w.x : w.y) >> ((e % 2) * 16)) & 0xffffu);
      o[e] = rb(__fmul_rn(__fmul_rn(sv[e], r), wv));
    }
    *reinterpret_cast<uint2*>(p.out + row * p.hidden + v * kVec) = make_uint2(o[0] | (o[1] << 16), o[2] | (o[3] << 16));
  }
  device::PDLTriggerSecondary<kUsePDL>();
}

template <int kHidden, bool kUsePDL>
struct InputNorm {
  static_assert(kHidden % (kThreads * kVec) == 0);

  static void embed_stream(const tvm::ffi::TensorView x, const tvm::ffi::TensorView stream,
                           const tvm::ffi::TensorView sq, double eps) {
    using namespace host;
    auto M = SymbolicSize{"m"};
    auto device_ = SymbolicDevice{};
    device_.set_options<kDLCUDA>();
    TensorMatcher({M, kHidden}).with_dtype<bf16_t>().with_device(device_).verify(x);
    TensorMatcher({M, kHidden}).with_dtype<fp32_t>().with_device(device_).verify(stream);
    TensorMatcher({M, kHidden}).with_dtype<fp32_t>().with_device(device_).verify(sq);
    if (M.unwrap() == 0) return;
    EmbedParams p{static_cast<const uint16_t*>(x.data_ptr()), static_cast<float*>(stream.data_ptr()),
                  static_cast<float*>(sq.data_ptr()), static_cast<float>(eps), kHidden};
    LaunchKernel(static_cast<unsigned>(M.unwrap()), kThreads, device_.unwrap())
        .enable_pdl(kUsePDL)(embed_stream_kernel<kHidden / (kThreads * kVec), kUsePDL>, p);
  }

  static void stream_sq(const tvm::ffi::TensorView x, const tvm::ffi::TensorView stream,
                        const tvm::ffi::TensorView sq) {
    using namespace host;
    auto M = SymbolicSize{"m"};
    auto device_ = SymbolicDevice{};
    device_.set_options<kDLCUDA>();
    TensorMatcher({M, kHidden}).with_dtype<bf16_t>().with_device(device_).verify(x);
    TensorMatcher({M, kHidden}).with_dtype<fp32_t>().with_device(device_).verify(stream);
    TensorMatcher({M, kHidden}).with_dtype<fp32_t>().with_device(device_).verify(sq);
    if (M.unwrap() == 0) return;
    StreamParams p{static_cast<const uint16_t*>(x.data_ptr()), static_cast<float*>(stream.data_ptr()),
                   static_cast<float*>(sq.data_ptr()), M.unwrap() * kHidden};
    const int64_t vecs = p.n / kVec;
    const unsigned grid = static_cast<unsigned>(std::min<int64_t>((vecs + 255) / 256, 1024));
    LaunchKernel(grid, 256, device_.unwrap()).enable_pdl(kUsePDL)(stream_sq_kernel<kUsePDL>, p);
  }

  static void finish(const tvm::ffi::TensorView stream, const tvm::ffi::TensorView mean,
                     const tvm::ffi::TensorView weight, const tvm::ffi::TensorView out, double eps) {
    using namespace host;
    auto M = SymbolicSize{"m"};
    auto device_ = SymbolicDevice{};
    device_.set_options<kDLCUDA>();
    TensorMatcher({M, kHidden}).with_dtype<fp32_t>().with_device(device_).verify(stream);
    TensorMatcher({M, 1}).with_dtype<fp32_t>().with_device(device_).verify(mean);
    TensorMatcher({kHidden}).with_dtype<bf16_t>().with_device(device_).verify(weight);
    TensorMatcher({M, kHidden}).with_dtype<bf16_t>().with_device(device_).verify(out);
    if (M.unwrap() == 0) return;
    FinishParams p{static_cast<const float*>(stream.data_ptr()), static_cast<const float*>(mean.data_ptr()),
                   static_cast<const uint16_t*>(weight.data_ptr()), static_cast<uint16_t*>(out.data_ptr()),
                   static_cast<float>(eps), kHidden};
    LaunchKernel(dim3(kHidden / kVec / 256, static_cast<unsigned>(M.unwrap())), 256, device_.unwrap())
        .enable_pdl(kUsePDL)(input_finish_kernel<kUsePDL>, p);
  }
};

}  // namespace muse_vr::input_norm
