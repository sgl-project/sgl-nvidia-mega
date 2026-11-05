// SPDX-License-Identifier: Apache-2.0
// Portions Copyright 2023-2024 SGLang Team (Apache-2.0)
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// Derived from SGLang (https://github.com/sgl-project/sglang), Apache License 2.0; see ../include/LICENSE.sglang.
// Source: SGLang python/sglang/kernels/jit/csrc/muse_spark/norm_residual.cuh (muse_spark_norm_residual_kernel, the
// one-CTA-per-row path; included below from the installed SGLang through the plugin's forwarding header).
// Modified by NVIDIA: the per-row arithmetic is unchanged (the same element ownership per thread, accumulation order,
// reductions, roundings and SGLang helpers), and
//   - the kernel also writes the pre-norm output as MXFP8 (E4M3 values, one E8M0 scale per 32 elements along the row,
//     scales in the 128 x 4 swizzled layout), computed from the BF16 output bytes with the arithmetic of mxfp8_fq.cuh
//     (so the bytes equal what the MXFP8 quantizer gives for the BF16 output);
//   - the grid also covers the scale layout's padding rows (up to the next multiple of 128): with kPadQuant they are
//     written as quantized zero rows (zero values and the scale byte of a zero block), otherwise they get scale bytes 0
//     and no values.
//
// branch: [M, H] bf16; stream: [M, H] fp32; output: [M, H] bf16; new_stream: [M, H] fp32; router_out: [M, H] fp32 or null;
// post_weight / pre_weight: [H] bf16 or null; alpha / beta: [H] or [1] fp32 or null;
// q_out: [R, H] uint8 (E4M3 bytes), R = M, or R >= rows_pad with kPadQuant; sf_out: >= rows_pad * H / 32 uint8.

#pragma once

#include <sgl_kernel/tensor.h>
#include <sgl_kernel/utils.h>

#include <sgl_kernel/cta.cuh>
#include <sgl_kernel/math.cuh>
#include <sgl_kernel/runtime.cuh>
#include <sgl_kernel/type.cuh>
#include <sgl_kernel/utils.cuh>
#include <sgl_kernel/vec.cuh>

#include <tvm/ffi/container/tensor.h>
#include <tvm/ffi/optional.h>

#include "muse_spark_v12/norm_residual.cuh"
#include "mxfp8_fq.cuh"

#include <cuda_bf16.h>
#include <cuda_fp8.h>

#include <cstdint>

namespace sglang {
namespace muse_vr_b3 {

struct B3Params {
  NormResidualParams nr;  // the parameters of SGLang's kernel, unchanged
  uint8_t* __restrict__ q_out;
  uint8_t* __restrict__ sf_out;
  int rows_pad;
};

// Byte offset of scale row m's first scale in the 128 x 4 swizzled layout (as mxfp8_fq.cuh's sf_offset(m, 0, H / 32)).
template <int64_t kHidden>
SGL_DEVICE int64_t sf_row_offset(int m) {
  constexpr int64_t kKTiles = kHidden / 128;
  return static_cast<int64_t>(m >> 7) * kKTiles * 512 + (m & 31) * 16 + ((m & 127) >> 5) * 4;
}

// MXFP8 of the 8 BF16 values a thread owns in one pack. The scale block (32 elements) spans 4 consecutive lanes, so the
// block maximum combines the 4 lanes (xor 1, xor 2); every lane of the warp must call it. The rest is mxfp8_fq.cuh's
// quant_item in its lean form: s = amax * rcp(448), E8M0 rounded toward +inf (saturating), r = rcp(scale) or 0, E4M3
// satfinite of x * r with NaN codes mapped to +448.
template <bool kFtz>
SGL_DEVICE uint2 quant8(const bf16x2_t (&v)[4], uint8_t& scale_byte) {
  using namespace ::muse_vr::fq;
  __nv_bfloat162 m = __habs2(v[0]);
#pragma unroll
  for (int i = 1; i < 4; ++i) m = __hmax2(m, __habs2(v[i]));
  uint32_t mu = *reinterpret_cast<uint32_t*>(&m);
  uint32_t ou = __shfl_xor_sync(0xffffffffu, mu, 1);
  m = __hmax2(*reinterpret_cast<__nv_bfloat162*>(&ou), m);
  mu = *reinterpret_cast<uint32_t*>(&m);
  ou = __shfl_xor_sync(0xffffffffu, mu, 2);
  m = __hmax2(*reinterpret_cast<__nv_bfloat162*>(&ou), m);
  const float amax = __bfloat162float(__hmax(m.x, m.y));
  const float s = fmul<kFtz>(amax, rcp_approx_ftz(448.0f));
  __nv_fp8_e8m0 e;
  e.__x = __nv_cvt_float_to_e8m0(s, __NV_SATFINITE, cudaRoundPosInf);
  const float sv = static_cast<float>(e);
  const float r = nonzero<kFtz>(sv) ? rcp_approx_ftz(sv) : 0.0f;
  scale_byte = e.__x;
  union {
    uint2 u;
    uint32_t w[2];
    __nv_fp8x2_e4m3 q[4];
  } o;
  const float2 rr = make_float2(r, r);
#pragma unroll
  for (int i = 0; i < 4; ++i) o.q[i] = __nv_fp8x2_e4m3(fmul2<kFtz>(__bfloat1622float2(v[i]), rr));
#pragma unroll
  for (int j = 0; j < 2; ++j) o.w[j] = nan_to_448(o.w[j]);
  return o.u;
}

// SGLang's muse_spark_norm_residual_kernel with the MXFP8 epilogue (marked "B3"); everything else is its code.
template <
    int64_t kHidden,
    int kNumThreads,
    bool kUsePDL,
    bool kPrefetchWeights,
    bool kPostNormIsInputScale,
    bool kHasRouter,
    bool kPostWeight,
    bool kGate,
    bool kPadQuant,
    bool kFtz>
__global__ void __launch_bounds__(kNumThreads) b3_norm_mx_kernel(const __grid_constant__ B3Params bp) {
  using namespace device;
  using namespace device::norm_residual;
  const NormResidualParams& params = bp.nr;

  constexpr int kPacks = kHidden / (kNumThreads * kVec);
  constexpr int kNumWarps = kNumThreads / kWarpThreads;
  constexpr int kPreSlots = kPrefetchWeights ? kPacks : 1;
  static_assert(kHidden % (kNumThreads * kVec) == 0);
  static_assert(kHidden % 128 == 0, "whole scale tiles");
  static_assert(kVec == 8 && kPairs == 4, "B3: 8 elements per pack (a scale block is 4 lanes)");

  using BF16Vec = AlignedVector<bf16x2_t, kPairs>;
  using F32Vec = AlignedVector<fp32x2_t, kFPairs>;

  // B3: this CTA's rows of the MXFP8 outputs
  const int row = static_cast<int>(blockIdx.x);
  uint8_t* const q_row = bp.q_out + static_cast<int64_t>(row) * kHidden;
  uint8_t* const sf_row = bp.sf_out + sf_row_offset<kHidden>(row);

  if (row >= params.num_rows) {  // B3: a padding row of the scale layout (block-uniform)
    PDLWaitPrimary<kUsePDL>();
#pragma unroll
    for (int p = 0; p < kPacks; ++p) {
      const int idx = threadIdx.x + p * kNumThreads;
      const int k = idx >> 2;
      if constexpr (kPadQuant) {
        bf16x2_t z[4];
#pragma unroll
        for (int q = 0; q < 4; ++q) z[q] = __float2bfloat162_rn(0.0f);
        uint8_t sb;
        const uint2 qv = quant8<kFtz>(z, sb);
        *reinterpret_cast<uint2*>(q_row + static_cast<int64_t>(idx) * kVec) = qv;
        if ((threadIdx.x & 3) == 0) sf_row[(k >> 2) * 512 + (k & 3)] = sb;
      } else {
        if ((threadIdx.x & 3) == 0) sf_row[(k >> 2) * 512 + (k & 3)] = 0;
      }
    }
    PDLTriggerSecondary<kUsePDL>();
    return;
  }

  const int64_t base = static_cast<int64_t>(blockIdx.x) * kHidden;
  const auto branch = static_cast<const bf16_t*>(params.branch) + base;
  const auto stream = static_cast<const fp32_t*>(params.stream) + base;
  const auto out = static_cast<bf16_t*>(params.output) + base;
  const auto new_stream = static_cast<fp32_t*>(params.new_stream) + base;
  const auto router = static_cast<fp32_t*>(params.router_out) + (kHasRouter ? base : 0);
  const auto post_weight = static_cast<const bf16_t*>(params.post_weight);
  const auto pre_weight = static_cast<const bf16_t*>(params.pre_weight);
  const auto alpha_ptr = static_cast<const fp32_t*>(params.alpha);
  const auto beta_ptr = static_cast<const fp32_t*>(params.beta);

  __shared__ float smem[kNumWarps == 1 ? 1 : 2 * kNumWarps];

  const bool channel_wise = kGate && params.gate_channel_wise;
  const bool has_pre_weight = params.has_pre_weight;
  const bool pre_bf16 = params.pre_norm_bf16;

  fp32x2_t vals[kPacks][kPairs];
  fp32x2_t post_w[kPostWeight ? kPacks : 1][kPairs];
  F32Vec stream_v[kPacks][kFParts];
  F32Vec alpha_pf[kPreSlots][kFParts], beta_pf[kPreSlots][kFParts];
  BF16Vec pre_w_pf[kPreSlots];

  PDLWaitPrimary<kUsePDL>();

  float sum_sq = 0.0f;
#pragma unroll
  for (int p = 0; p < kPacks; ++p) {
    const int idx = threadIdx.x + p * kNumThreads;
    BF16Vec branch_v;
    branch_v.load(branch, idx);
#pragma unroll
    for (int j = 0; j < kFParts; ++j)
      stream_v[p][j].load(stream, idx * kFParts + j);
    if constexpr (kPrefetchWeights) {
      if (has_pre_weight) pre_w_pf[p].load(pre_weight, idx);
      if constexpr (kGate) {
        if (channel_wise) {
#pragma unroll
          for (int j = 0; j < kFParts; ++j) {
            alpha_pf[p][j].load(alpha_ptr, idx * kFParts + j);
            beta_pf[p][j].load(beta_ptr, idx * kFParts + j);
          }
        }
      }
    }
    if constexpr (kPostWeight) {
      BF16Vec weight_v;
      weight_v.load(post_weight, idx);
#pragma unroll
      for (int q = 0; q < kPairs; ++q)
        post_w[p][q] = cast<fp32x2_t>(weight_v[q]);
    }
#pragma unroll
    for (int q = 0; q < kPairs; ++q) {
      auto v = cast<fp32x2_t>(branch_v[q]);
      if constexpr (kPostNormIsInputScale && kPostWeight) {
        v.x *= post_w[p][q].x;
        v.y *= post_w[p][q].y;
        v = round_bf16(v);
      }
      vals[p][q] = v;
      sum_sq = fmaf(v.x, v.x, sum_sq);
      sum_sq = fmaf(v.y, v.y, sum_sq);
    }
  }

  sum_sq = cta::reduce_sum<kNumThreads>(sum_sq, smem);
  const float post_rstd = math::rsqrt(sum_sq / kHidden + params.post_eps);

  fp32x2_t gate_a = {1.0f, 1.0f};
  fp32x2_t gate_b = {1.0f, 1.0f};
  if constexpr (kGate) {
    if (!channel_wise) {
      gate_a.x = gate_a.y = *alpha_ptr;
      gate_b.x = gate_b.y = *beta_ptr;
    }
  }

  float pre_sum_sq = 0.0f;
#pragma unroll
  for (int p = 0; p < kPacks; ++p) {
    const int idx = threadIdx.x + p * kNumThreads;
    F32Vec alpha_v[kFParts], beta_v[kFParts];
    if constexpr (kGate) {
      if (channel_wise) {
#pragma unroll
        for (int j = 0; j < kFParts; ++j) {
          if constexpr (kPrefetchWeights) {
            alpha_v[j] = alpha_pf[p][j];
            beta_v[j] = beta_pf[p][j];
          } else {
            alpha_v[j].load(alpha_ptr, idx * kFParts + j);
            beta_v[j].load(beta_ptr, idx * kFParts + j);
          }
        }
      }
    }
    F32Vec new_stream_v[kFParts];
#pragma unroll
    for (int q = 0; q < kPairs; ++q) {
      const int h = q / kFPairs;
      const int l = q % kFPairs;
      fp32x2_t normed_b = vals[p][q];
      normed_b.x *= post_rstd;
      normed_b.y *= post_rstd;
      if constexpr (!kPostNormIsInputScale && kPostWeight) {
        normed_b.x *= post_w[p][q].x;
        normed_b.y *= post_w[p][q].y;
      }
      normed_b = round_bf16(normed_b);
      const fp32x2_t s = stream_v[p][h][l];
      fp32x2_t new_s;
      if constexpr (kGate) {
        const fp32x2_t ga = channel_wise ? alpha_v[h][l] : gate_a;
        const fp32x2_t gb = channel_wise ? beta_v[h][l] : gate_b;
        new_s.x = ga.x * s.x + gb.x * normed_b.x;
        new_s.y = ga.y * s.y + gb.y * normed_b.y;
      } else {
        new_s.x = s.x + normed_b.x;
        new_s.y = s.y + normed_b.y;
      }
      new_stream_v[h][l] = new_s;
      const fp32x2_t pre_v = pre_bf16 ? round_bf16(new_s) : new_s;
      vals[p][q] = pre_v;
      pre_sum_sq = fmaf(pre_v.x, pre_v.x, pre_sum_sq);
      pre_sum_sq = fmaf(pre_v.y, pre_v.y, pre_sum_sq);
    }
#pragma unroll
    for (int j = 0; j < kFParts; ++j)
      new_stream_v[j].store(new_stream, idx * kFParts + j);
  }

  pre_sum_sq = cta::reduce_sum<kNumThreads>(pre_sum_sq, kNumWarps == 1 ? smem : smem + kNumWarps);
  const float pre_rstd = math::rsqrt(pre_sum_sq / kHidden + params.pre_eps);

#pragma unroll
  for (int p = 0; p < kPacks; ++p) {
    const int idx = threadIdx.x + p * kNumThreads;
    BF16Vec weight_v;
    if (has_pre_weight) {
      if constexpr (kPrefetchWeights) {
        weight_v = pre_w_pf[p];
      } else {
        weight_v.load(pre_weight, idx);
      }
    }
    BF16Vec out_v;
    F32Vec router_v[kFParts];
#pragma unroll
    for (int q = 0; q < kPairs; ++q) {
      fp32x2_t v = vals[p][q];
      v.x *= pre_rstd;
      v.y *= pre_rstd;
      if (has_pre_weight) {
        const auto wf = cast<fp32x2_t>(weight_v[q]);
        v.x *= wf.x;
        v.y *= wf.y;
      }
      if (pre_bf16) v = round_bf16(v);
      out_v[q] = cast<bf16x2_t>(v);
      if constexpr (kHasRouter) {
        router_v[q / kFPairs][q % kFPairs] = v;
      }
    }
    out_v.store(out, idx);
    if constexpr (kHasRouter) {
#pragma unroll
      for (int j = 0; j < kFParts; ++j)
        router_v[j].store(router, idx * kFParts + j);
    }
    // B3: MXFP8 of the BF16 output just stored (the 4 lanes of a scale block hold one 32-element run)
    {
      bf16x2_t ov[4];
#pragma unroll
      for (int q = 0; q < 4; ++q) ov[q] = out_v[q];
      uint8_t sb;
      const uint2 qv = quant8<kFtz>(ov, sb);
      *reinterpret_cast<uint2*>(q_row + static_cast<int64_t>(idx) * kVec) = qv;
      if ((threadIdx.x & 3) == 0) {
        const int k = idx >> 2;
        sf_row[(k >> 2) * 512 + (k & 3)] = sb;
      }
    }
  }

  PDLTriggerSecondary<kUsePDL>();
}

// Host entry. kPadQuant: see the file header. The kernel's launch shape is SGLang's for the row count (one CTA of
// kNumThreads per row; the caller passes SGLang's thread count and weight-prefetch choice), plus the padding rows.
template <int64_t kHidden, int kNumThreads, bool kUsePDL, bool kPrefetchWeights, bool kPostNormIsInputScale, bool kHasRouter,
          bool kPadQuant>
struct B3NormMx {
  static void
  run(const tvm::ffi::TensorView branch,
      const tvm::ffi::TensorView stream,
      const tvm::ffi::TensorView output,
      const tvm::ffi::TensorView new_stream,
      const tvm::ffi::Optional<tvm::ffi::TensorView> router_out,
      const tvm::ffi::Optional<tvm::ffi::TensorView> post_weight,
      const tvm::ffi::Optional<tvm::ffi::TensorView> pre_weight,
      const tvm::ffi::Optional<tvm::ffi::TensorView> alpha,
      const tvm::ffi::Optional<tvm::ffi::TensorView> beta,
      double post_eps,
      double pre_eps,
      bool pre_norm_bf16,
      const tvm::ffi::TensorView q_out,
      const tvm::ffi::TensorView sf_out) {
    using namespace host;

    auto M = SymbolicSize{"num_tokens"};
    auto H = SymbolicSize{"hidden_size"};
    auto G = SymbolicSize{"gate_dim"};
    auto device_ = SymbolicDevice{};
    H.set_value(kHidden);
    device_.set_options<kDLCUDA>();

    TensorMatcher({M, H}).with_dtype<bf16_t>().with_device(device_).verify(branch);
    TensorMatcher({M, H}).with_dtype<fp32_t>().with_device(device_).verify(stream);
    TensorMatcher({M, H}).with_dtype<bf16_t>().with_device(device_).verify(output);
    TensorMatcher({M, H}).with_dtype<fp32_t>().with_device(device_).verify(new_stream);
    RuntimeCheck(router_out.has_value() == kHasRouter, "router_out presence must match the kHasRouter instantiation");
    if constexpr (kHasRouter) {
      TensorMatcher({M, H}).with_dtype<fp32_t>().with_device(device_).verify(router_out.value());
    }
    if (post_weight.has_value()) {
      TensorMatcher({H}).with_dtype<bf16_t>().with_device(device_).verify(post_weight.value());
    }
    if (pre_weight.has_value()) {
      TensorMatcher({H}).with_dtype<bf16_t>().with_device(device_).verify(pre_weight.value());
    }
    const bool gate = alpha.has_value();
    RuntimeCheck(gate == beta.has_value(), "alpha and beta must both be present or both absent");
    bool gate_channel_wise = false;
    if (gate) {
      TensorMatcher({G}).with_dtype<fp32_t>().with_device(device_).verify(alpha.value());
      TensorMatcher({G}).with_dtype<fp32_t>().with_device(device_).verify(beta.value());
      const auto n = G.unwrap();
      RuntimeCheck(n == 1 || n == kHidden, "alpha/beta must have 1 or hidden_size elements");
      gate_channel_wise = (n == kHidden);
    }

    const auto num_rows = static_cast<int>(M.unwrap());
    if (num_rows == 0) return;
    const int rows_pad = (num_rows + 127) / 128 * 128;
    RuntimeCheck(q_out.dim() == 2 && q_out.dtype().bits == 8 && q_out.size(1) == kHidden && q_out.stride(1) == 1 &&
                     q_out.stride(0) == kHidden,
                 "q_out: contiguous [rows, H] 8-bit");
    const auto q_rows = static_cast<int>(q_out.size(0));
    RuntimeCheck(kPadQuant ? q_rows >= rows_pad : q_rows == num_rows, "q_out rows: num_rows, or >= rows_pad with kPadQuant");
    RuntimeCheck(sf_out.dim() == 1 && sf_out.dtype().bits == 8 && sf_out.numel() >= static_cast<int64_t>(rows_pad) * (kHidden / 32),
                 "sf_out: 1-D 8-bit, at least rows_pad * H / 32 bytes");
    RuntimeCheck(reinterpret_cast<uintptr_t>(q_out.data_ptr()) % 16 == 0, "q_out 16-byte aligned");

    const auto params = B3Params{
        .nr =
            NormResidualParams{
                .branch = branch.data_ptr(),
                .stream = stream.data_ptr(),
                .output = output.data_ptr(),
                .new_stream = new_stream.data_ptr(),
                .router_out = kHasRouter ? router_out.value().data_ptr() : nullptr,
                .post_weight = post_weight.has_value() ? post_weight.value().data_ptr() : nullptr,
                .pre_weight = pre_weight.has_value() ? pre_weight.value().data_ptr() : nullptr,
                .alpha = gate ? alpha.value().data_ptr() : nullptr,
                .beta = gate ? beta.value().data_ptr() : nullptr,
                .post_eps = static_cast<float>(post_eps),
                .pre_eps = static_cast<float>(pre_eps),
                .num_rows = num_rows,
                .gate_channel_wise = gate_channel_wise,
                .has_pre_weight = pre_weight.has_value(),
                .pre_norm_bf16 = pre_norm_bf16,
            },
        .q_out = static_cast<uint8_t*>(q_out.data_ptr()),
        .sf_out = static_cast<uint8_t*>(sf_out.data_ptr()),
        .rows_pad = rows_pad,
    };
    const auto device = device_.unwrap();

    const auto launch = [&]<bool kHasPostWeight, bool kHasGate>() {
      LaunchKernel(rows_pad, kNumThreads, device)
          .config({.use_pdl = kUsePDL})(
              b3_norm_mx_kernel<
                  kHidden,
                  kNumThreads,
                  kUsePDL,
                  kPrefetchWeights,
                  kPostNormIsInputScale,
                  kHasRouter,
                  kHasPostWeight,
                  kHasGate,
                  kPadQuant,
                  true>,
              params);
    };
    if (post_weight.has_value()) {
      if (gate)
        launch.template operator()<true, true>();
      else
        launch.template operator()<true, false>();
    } else {
      if (gate)
        launch.template operator()<false, true>();
      else
        launch.template operator()<false, false>();
    }
  }
};

}  // namespace muse_vr_b3
}  // namespace sglang
