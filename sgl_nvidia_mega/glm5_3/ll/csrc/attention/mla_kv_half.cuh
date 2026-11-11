// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
//
// The MLA KV half of a GLM-5 DSA verify / draft-extend step after the absorb BMM, in one
// PDL launch: (kFuseNorm) the kv_a RMSNorm of the raw 512-wide latent, GPT-J RoPE of
// k_pe and q_pe, E4M3 quantization of q = [q_nope | q_pe] and k = [k_nope | k_pe], and
// the 576-byte append of k at kv_buffer[loc] (rows equal to reserved_skip_index skipped).
//
// Bytes match the three kernels it replaces on SGLang's DSA TRT-LLM FP8 path:
// FlashInfer's CuTe-DSL RMSNormKernel (H = 512), RopeQuantizeKernel (interleaved)
// and SGLang's set_mla_kv_buffer kernel. The nope halves are quantized from the
// bf16-rounded values, the rotated rope halves straight from FP32, as there. The
// folded norm's bits are a cross-library contract: ll/norm_backend.py checks them
// against the stock norm at build before the norm is folded (kWriteNormOut is the
// probe's instance).

#pragma once

#include <sgl_kernel/tensor.h>  // TensorMatcher, SymbolicSize, SymbolicDType, SymbolicDevice
#include <sgl_kernel/utils.h>   // RuntimeCheck, div_ceil

#include <sgl_kernel/utils.cuh>  // LaunchKernel, SGL_DEVICE, PDL helpers, bf16_t

#include <dlpack/dlpack.h>
#include <tvm/ffi/container/tensor.h>

#include <cstdint>
#include <cuda_bf16.h>
#include <cuda_fp8.h>

// Inside sglang:: for the sgl_kernel helpers (host::, device::, bf16_t), as spec_step's sources.
namespace sglang::glm_ll::mla_kv_half {

inline constexpr int kNopeDim = 512;
inline constexpr int kRopeDim = 64;
inline constexpr int kRowBytes = kNopeDim + kRopeDim;  // one FP8 row of q or of the KV cache
inline constexpr int kWarpsPerBlock = 4;
inline constexpr int kThreadsPerBlock = kWarpsPerBlock * 32;

struct Params {
  const bf16_t* __restrict__ q_nope;        // [T, H, 512], absorbed q_nope
  const bf16_t* __restrict__ q_pe;          // [T, H, 64]
  const bf16_t* __restrict__ k_nope;        // [T, 512]: raw latent (kFuseNorm) or normalized
  const bf16_t* __restrict__ k_pe;          // [T, 64]
  const bf16_t* __restrict__ norm_weight;   // [512] (kFuseNorm only)
  const float* __restrict__ cos_sin_cache;  // [P, 64]: cos[0, 32) | sin[32, 64)
  const void* __restrict__ positions;       // [T], int32 or int64
  const void* __restrict__ loc;             // [T], int32 or int64
  uint8_t* __restrict__ q_out;              // [T, H, 576] FP8
  uint8_t* __restrict__ kv_buffer;          // [S, >= 576] FP8 rows
  bf16_t* __restrict__ k_norm_out;          // [T, 512] normalized latent (kWriteNormOut, harness probe)
  int64_t q_nope_stride_t;                  // elements
  int64_t q_nope_stride_h;
  int64_t q_pe_stride_t;
  int64_t q_pe_stride_h;
  int64_t k_nope_stride_t;
  int64_t k_pe_stride_t;
  int64_t q_out_stride_t;  // bytes (FP8)
  int64_t q_out_stride_h;
  int64_t kv_stride;  // bytes
  int64_t reserved_skip_index;
  float norm_eps;
  float quant_scale_q;
  float quant_scale_kv;
  uint32_t num_tokens;
  uint32_t num_heads;
};

SGL_DEVICE float2 unpack_bf16x2(uint32_t word) {
  bf16x2_t v;
  *reinterpret_cast<uint32_t*>(&v) = word;
  return __bfloat1622float2(v);  // exact
}

// RopeQuantizeKernel is built with -use_fast_math: its FP32 ops are FTZ. Pin the
// same instructions here (this file is built without fast math).
SGL_DEVICE float mul_ftz(float a, float b) {
  float d;
  asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(d) : "f"(a), "f"(b));
  return d;
}

SGL_DEVICE float fma_ftz(float a, float b, float c) {
  float d;
  asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(d) : "f"(a), "f"(b), "f"(c));
  return d;
}

// Two FP32 values to E4M3 as FlashInfer's vec_cast<__nv_fp8_e4m3, float> does
// (cvt.rn.satfinite.e4m3x2.f32); the first value lands in the low byte.
SGL_DEVICE uint16_t to_fp8x2(float a, float b) {
  return static_cast<uint16_t>(__nv_cvt_float2_to_fp8x2(make_float2(a, b), __NV_SATFINITE, __NV_E4M3));
}

// Eight bf16 (one 16-byte word) -> eight E4M3 bytes, each value times `scale`.
SGL_DEVICE uint2 bf16x8_to_fp8x8(const uint4 v, float scale) {
  const uint32_t words[4] = {v.x, v.y, v.z, v.w};
  uint16_t pairs[4];
#pragma unroll
  for (int j = 0; j < 4; ++j) {
    const float2 f = unpack_bf16x2(words[j]);
    pairs[j] = to_fp8x2(mul_ftz(f.x, scale), mul_ftz(f.y, scale));
  }
  return make_uint2(
      static_cast<uint32_t>(pairs[0]) | (static_cast<uint32_t>(pairs[1]) << 16),
      static_cast<uint32_t>(pairs[2]) | (static_cast<uint32_t>(pairs[3]) << 16));
}

// GPT-J RoPE of one pair, then the quantization scale, with the FMA shapes of
// the FlashInfer 0.7.1 sm_103a RopeQuantizeKernel SASS (jit-cache rope.so):
//   even = fma(x0, c, -(x1 * s)), odd = fma(x1, c, x0 * s).
SGL_DEVICE uint16_t rope_pair_to_fp8x2(uint32_t word, float cos_v, float sin_v, float scale) {
  const float2 x = unpack_bf16x2(word);
  const float r0 = fma_ftz(x.x, cos_v, -mul_ftz(x.y, sin_v));
  const float r1 = fma_ftz(x.y, cos_v, mul_ftz(x.x, sin_v));
  return to_fp8x2(mul_ftz(r0, scale), mul_ftz(r1, scale));
}

SGL_DEVICE float rsqrt_approx_ftz(float x) {
  float y;
  asm("rsqrt.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x));
  return y;
}

// rstd of one 512-wide row, as FlashInfer 0.7.1's CuTe-DSL RMSNormKernel computes it
// (PTX of the sm_103a build under nvidia-cutlass-dsl 4.8.0): lane l folds its 16
// columns {8l + v} then {256 + 8l + v}, v ascending, into one fma.rn chain from 0,
// then a butterfly add over xor 1, 2, 4, 8, 16, then rsqrt.approx.ftz(sum * 2^-9 + eps).
SGL_DEVICE float row_rstd(const float (&x)[16], float eps) {
  float sum = 0.f;
#pragma unroll
  for (int i = 0; i < 16; ++i)
    sum = __fmaf_rn(x[i], x[i], sum);
#pragma unroll
  for (int offset = 1; offset < 32; offset <<= 1)
    sum = __fadd_rn(__shfl_xor_sync(0xffffffffu, sum, offset), sum);
  return rsqrt_approx_ftz(__fmaf_rn(sum, 1.0f / static_cast<float>(kNopeDim), eps));
}

// kTriggerAtEntry: the PDL dependents may launch at entry instead of after this kernel's
// own wait and loads; they read its outputs only after their own wait. For a sparse MLA
// flash decode that gathers the KV rows of earlier steps before its wait.
template <bool kFuseNorm, bool kWriteNormOut, bool kUsePDL, bool kTriggerAtEntry, typename TPos, typename TLoc>
__global__ void __launch_bounds__(kThreadsPerBlock) mla_kv_half_kernel(const __grid_constant__ Params p) {
  using namespace device;
  if constexpr (kTriggerAtEntry) {
    PDLTriggerSecondary<kUsePDL>();
  }
  const uint32_t lane = threadIdx.x % kWarpThreads;
  const uint32_t flat_warp = blockIdx.x * kWarpsPerBlock + threadIdx.x / kWarpThreads;
  const uint32_t num_q_rows = p.num_tokens * p.num_heads;

  if (flat_warp < p.num_tokens) {
    // ---- KV role: one warp per token --------------------------------------
    const uint32_t t = flat_warp;
    uint4 w_vec[2];
    if constexpr (kFuseNorm) {
      // The norm weight is static: read it before the wait.
      w_vec[0] = reinterpret_cast<const uint4*>(p.norm_weight)[lane];
      w_vec[1] = reinterpret_cast<const uint4*>(p.norm_weight + kNopeDim / 2)[lane];
    }

    PDLWaitPrimary<kUsePDL>();

    const int64_t pos = static_cast<int64_t>(static_cast<const TPos*>(p.positions)[t]);
    const int64_t loc = static_cast<int64_t>(static_cast<const TLoc*>(p.loc)[t]);
    const bf16_t* nope_row = p.k_nope + static_cast<int64_t>(t) * p.k_nope_stride_t;
    const uint4 x_vec0 = reinterpret_cast<const uint4*>(nope_row)[lane];
    const uint4 x_vec1 = reinterpret_cast<const uint4*>(nope_row + kNopeDim / 2)[lane];
    const uint32_t pe_word =
        reinterpret_cast<const uint32_t*>(p.k_pe + static_cast<int64_t>(t) * p.k_pe_stride_t)[lane];
    const float cos_v = p.cos_sin_cache[pos * kRopeDim + lane];
    const float sin_v = p.cos_sin_cache[pos * kRopeDim + kRopeDim / 2 + lane];

    PDLTriggerSecondary<kUsePDL>();

    uint2 nope_fp8[2];
    if constexpr (kFuseNorm) {
      float x[16];
      float w[16];
      {
        const uint32_t xw[8] = {x_vec0.x, x_vec0.y, x_vec0.z, x_vec0.w, x_vec1.x, x_vec1.y, x_vec1.z, x_vec1.w};
        const uint32_t ww[8] = {
            w_vec[0].x, w_vec[0].y, w_vec[0].z, w_vec[0].w, w_vec[1].x, w_vec[1].y, w_vec[1].z, w_vec[1].w};
#pragma unroll
        for (int j = 0; j < 8; ++j) {
          const float2 xf = unpack_bf16x2(xw[j]);
          const float2 wf = unpack_bf16x2(ww[j]);
          x[2 * j] = xf.x;
          x[2 * j + 1] = xf.y;
          w[2 * j] = wf.x;
          w[2 * j + 1] = wf.y;
        }
      }
      const float rstd = row_rstd(x, p.norm_eps);
      uint4 y_vec[2];
#pragma unroll
      for (int b = 0; b < 2; ++b) {
        uint32_t y_words[4];
#pragma unroll
        for (int j = 0; j < 4; ++j) {
          const int i0 = b * 8 + 2 * j;
          // (x * rstd) * (w + 0), rounded to bf16 as the norm kernel stores it.
          const bf16_t y0 = __float2bfloat16_rn(__fmul_rn(__fmul_rn(x[i0], rstd), __fadd_rn(w[i0], 0.0f)));
          const bf16_t y1 = __float2bfloat16_rn(__fmul_rn(__fmul_rn(x[i0 + 1], rstd), __fadd_rn(w[i0 + 1], 0.0f)));
          y_words[j] =
              static_cast<uint32_t>(__bfloat16_as_ushort(y0)) | (static_cast<uint32_t>(__bfloat16_as_ushort(y1)) << 16);
        }
        y_vec[b] = make_uint4(y_words[0], y_words[1], y_words[2], y_words[3]);
      }
      if constexpr (kWriteNormOut) {
        bf16_t* norm_row = p.k_norm_out + static_cast<int64_t>(t) * kNopeDim;
        reinterpret_cast<uint4*>(norm_row)[lane] = y_vec[0];
        reinterpret_cast<uint4*>(norm_row + kNopeDim / 2)[lane] = y_vec[1];
      }
      // The default path re-reads the stored bf16 latent for the FP8 cast.
      nope_fp8[0] = bf16x8_to_fp8x8(y_vec[0], p.quant_scale_kv);
      nope_fp8[1] = bf16x8_to_fp8x8(y_vec[1], p.quant_scale_kv);
    } else {
      nope_fp8[0] = bf16x8_to_fp8x8(x_vec0, p.quant_scale_kv);
      nope_fp8[1] = bf16x8_to_fp8x8(x_vec1, p.quant_scale_kv);
    }
    const uint16_t pe_fp8 = rope_pair_to_fp8x2(pe_word, cos_v, sin_v, p.quant_scale_kv);

    if (loc != p.reserved_skip_index) {
      uint8_t* row = p.kv_buffer + loc * p.kv_stride;
      reinterpret_cast<uint2*>(row)[lane] = nope_fp8[0];
      reinterpret_cast<uint2*>(row + kNopeDim / 2)[lane] = nope_fp8[1];
      reinterpret_cast<uint16_t*>(row + kNopeDim)[lane] = pe_fp8;
    }
  } else if (flat_warp - p.num_tokens < num_q_rows) {
    // ---- Q role: one warp per (token, head) -------------------------------
    const uint32_t q_row = flat_warp - p.num_tokens;
    const uint32_t t = q_row / p.num_heads;
    const uint32_t h = q_row % p.num_heads;

    PDLWaitPrimary<kUsePDL>();

    const int64_t pos = static_cast<int64_t>(static_cast<const TPos*>(p.positions)[t]);
    const bf16_t* nope_row = p.q_nope + static_cast<int64_t>(t) * p.q_nope_stride_t + h * p.q_nope_stride_h;
    const bf16_t* pe_row = p.q_pe + static_cast<int64_t>(t) * p.q_pe_stride_t + h * p.q_pe_stride_h;
    const uint4 a0 = reinterpret_cast<const uint4*>(nope_row)[2 * lane];
    const uint4 a1 = reinterpret_cast<const uint4*>(nope_row)[2 * lane + 1];
    const uint32_t pe_word = reinterpret_cast<const uint32_t*>(pe_row)[lane];
    const float cos_v = p.cos_sin_cache[pos * kRopeDim + lane];
    const float sin_v = p.cos_sin_cache[pos * kRopeDim + kRopeDim / 2 + lane];

    PDLTriggerSecondary<kUsePDL>();

    const uint2 lo = bf16x8_to_fp8x8(a0, p.quant_scale_q);
    const uint2 hi = bf16x8_to_fp8x8(a1, p.quant_scale_q);
    uint8_t* out = p.q_out + static_cast<int64_t>(t) * p.q_out_stride_t + h * p.q_out_stride_h;
    reinterpret_cast<uint4*>(out)[lane] = make_uint4(lo.x, lo.y, hi.x, hi.y);
    reinterpret_cast<uint16_t*>(out + kNopeDim)[lane] = rope_pair_to_fp8x2(pe_word, cos_v, sin_v, p.quant_scale_q);
  } else {
    PDLWaitPrimary<kUsePDL>();
    PDLTriggerSecondary<kUsePDL>();
  }
}

template <bool kFuseNorm, bool kWriteNormOut, bool kUsePDL, bool kTriggerAtEntry = false>
struct MlaKvHalfKernel {
  static_assert(kFuseNorm || !kWriteNormOut, "the normalized latent exists only with the fused norm");
  template <typename TPos, typename TLoc>
  static constexpr auto kernel = mla_kv_half_kernel<kFuseNorm, kWriteNormOut, kUsePDL, kTriggerAtEntry, TPos, TLoc>;

  static void
  run(tvm::ffi::TensorView q_nope,
      tvm::ffi::TensorView q_pe,
      tvm::ffi::TensorView k_nope,
      tvm::ffi::TensorView k_pe,
      tvm::ffi::TensorView norm_weight,
      tvm::ffi::TensorView cos_sin_cache,
      tvm::ffi::TensorView positions,
      tvm::ffi::TensorView loc,
      tvm::ffi::TensorView q_out,
      tvm::ffi::TensorView kv_buffer,
      tvm::ffi::TensorView k_norm_out,
      float norm_eps,
      float quant_scale_q,
      float quant_scale_kv,
      int64_t reserved_skip_index) {
    using namespace host;

    auto T = SymbolicSize{"num_tokens"};
    auto H = SymbolicSize{"num_heads"};
    auto D_nope = SymbolicSize{"nope_dim"};
    auto D_rope = SymbolicSize{"rope_dim"};
    auto D_row = SymbolicSize{"row_dim"};
    auto S_qn_t = SymbolicSize{"q_nope_stride_t"};
    auto S_qn_h = SymbolicSize{"q_nope_stride_h"};
    auto S_qp_t = SymbolicSize{"q_pe_stride_t"};
    auto S_qp_h = SymbolicSize{"q_pe_stride_h"};
    auto S_kn_t = SymbolicSize{"k_nope_stride_t"};
    auto S_kp_t = SymbolicSize{"k_pe_stride_t"};
    auto S_qo_t = SymbolicSize{"q_out_stride_t"};
    auto S_qo_h = SymbolicSize{"q_out_stride_h"};
    auto S_kv = SymbolicSize{"kv_stride"};
    auto pos_dtype = SymbolicDType{};
    auto loc_dtype = SymbolicDType{};
    auto device = SymbolicDevice{};
    device.set_options<kDLCUDA>();
    D_nope.set_value(kNopeDim);
    D_rope.set_value(kRopeDim);
    D_row.set_value(kRowBytes);

    TensorMatcher({T, H, D_nope})
        .with_strides({S_qn_t, S_qn_h, 1})
        .with_dtype<bf16_t>()
        .with_device(device)
        .verify(q_nope);
    TensorMatcher({T, H, D_rope})
        .with_strides({S_qp_t, S_qp_h, 1})
        .with_dtype<bf16_t>()
        .with_device(device)
        .verify(q_pe);
    TensorMatcher({T, D_nope}).with_strides({S_kn_t, 1}).with_dtype<bf16_t>().with_device(device).verify(k_nope);
    TensorMatcher({T, D_rope}).with_strides({S_kp_t, 1}).with_dtype<bf16_t>().with_device(device).verify(k_pe);
    if constexpr (kFuseNorm) {
      TensorMatcher({D_nope}).with_strides({1}).with_dtype<bf16_t>().with_device(device).verify(norm_weight);
    }
    if constexpr (kWriteNormOut) {
      TensorMatcher({T, D_nope}).with_strides({D_nope, 1}).with_dtype<bf16_t>().with_device(device).verify(k_norm_out);
    }
    TensorMatcher({-1, D_rope}).with_strides({D_rope, 1}).with_dtype<float>().with_device(device).verify(cos_sin_cache);
    TensorMatcher({T}).with_strides({1}).with_dtype<int32_t, int64_t>(pos_dtype).with_device(device).verify(positions);
    TensorMatcher({T}).with_strides({1}).with_dtype<int32_t, int64_t>(loc_dtype).with_device(device).verify(loc);
    TensorMatcher({T, H, D_row})
        .with_strides({S_qo_t, S_qo_h, 1})
        .with_dtype<fp8_e4m3_t, uint8_t>()
        .with_device(device)
        .verify(q_out);
    TensorMatcher({-1, -1}).with_strides({S_kv, 1}).with_dtype<fp8_e4m3_t, uint8_t>().with_device(device).verify(
        kv_buffer);
    RuntimeCheck(kv_buffer.size(1) >= kRowBytes, "kv_buffer rows are narrower than ", kRowBytes, " bytes");

    // The device code uses 16-byte accesses on the nope rows, q_out rows and the
    // norm weight, 8-byte stores into the KV row and 4-byte accesses on the rope
    // rows; the Python gate mirrors these so other layouts take the fallback.
    const auto aligned = [](const void* ptr, int64_t align) {
      return reinterpret_cast<uintptr_t>(ptr) % static_cast<uintptr_t>(align) == 0;
    };
    RuntimeCheck(
        aligned(q_nope.data_ptr(), 16) && (S_qn_t.unwrap() * 2) % 16 == 0 && (S_qn_h.unwrap() * 2) % 16 == 0,
        "q_nope base/strides must be 16-byte aligned");
    RuntimeCheck(
        aligned(q_pe.data_ptr(), 4) && (S_qp_t.unwrap() * 2) % 4 == 0 && (S_qp_h.unwrap() * 2) % 4 == 0,
        "q_pe base/strides must be 4-byte aligned");
    RuntimeCheck(
        aligned(k_nope.data_ptr(), 16) && (S_kn_t.unwrap() * 2) % 16 == 0,
        "k_nope base/stride must be 16-byte aligned");
    RuntimeCheck(
        aligned(k_pe.data_ptr(), 4) && (S_kp_t.unwrap() * 2) % 4 == 0, "k_pe base/stride must be 4-byte aligned");
    RuntimeCheck(!kFuseNorm || aligned(norm_weight.data_ptr(), 16), "norm_weight must be 16-byte aligned");
    RuntimeCheck(!kWriteNormOut || aligned(k_norm_out.data_ptr(), 16), "k_norm_out must be 16-byte aligned");
    RuntimeCheck(
        aligned(q_out.data_ptr(), 16) && S_qo_t.unwrap() % 16 == 0 && S_qo_h.unwrap() % 16 == 0,
        "q_out base/strides must be 16-byte aligned");
    RuntimeCheck(
        aligned(kv_buffer.data_ptr(), 16) && S_kv.unwrap() % 16 == 0, "kv_buffer base/stride must be 16-byte aligned");

    const auto num_tokens = static_cast<uint32_t>(T.unwrap());
    const auto num_heads = static_cast<uint32_t>(H.unwrap());
    if (num_tokens == 0) return;

    const auto params = Params{
        .q_nope = static_cast<const bf16_t*>(q_nope.data_ptr()),
        .q_pe = static_cast<const bf16_t*>(q_pe.data_ptr()),
        .k_nope = static_cast<const bf16_t*>(k_nope.data_ptr()),
        .k_pe = static_cast<const bf16_t*>(k_pe.data_ptr()),
        .norm_weight = kFuseNorm ? static_cast<const bf16_t*>(norm_weight.data_ptr()) : nullptr,
        .cos_sin_cache = static_cast<const float*>(cos_sin_cache.data_ptr()),
        .positions = positions.data_ptr(),
        .loc = loc.data_ptr(),
        .q_out = static_cast<uint8_t*>(q_out.data_ptr()),
        .kv_buffer = static_cast<uint8_t*>(kv_buffer.data_ptr()),
        .k_norm_out = kWriteNormOut ? static_cast<bf16_t*>(k_norm_out.data_ptr()) : nullptr,
        .q_nope_stride_t = S_qn_t.unwrap(),
        .q_nope_stride_h = S_qn_h.unwrap(),
        .q_pe_stride_t = S_qp_t.unwrap(),
        .q_pe_stride_h = S_qp_h.unwrap(),
        .k_nope_stride_t = S_kn_t.unwrap(),
        .k_pe_stride_t = S_kp_t.unwrap(),
        .q_out_stride_t = S_qo_t.unwrap(),
        .q_out_stride_h = S_qo_h.unwrap(),
        .kv_stride = S_kv.unwrap(),
        .reserved_skip_index = reserved_skip_index,
        .norm_eps = norm_eps,
        .quant_scale_q = quant_scale_q,
        .quant_scale_kv = quant_scale_kv,
        .num_tokens = num_tokens,
        .num_heads = num_heads,
    };

    const bool pos32 = pos_dtype.is_type<int32_t>();
    const bool loc32 = loc_dtype.is_type<int32_t>();
    const auto kernel_ptr = pos32 ? (loc32 ? kernel<int32_t, int32_t> : kernel<int32_t, int64_t>)
                                  : (loc32 ? kernel<int64_t, int32_t> : kernel<int64_t, int64_t>);
    const uint32_t total_warps = num_tokens + num_tokens * num_heads;
    const uint32_t num_blocks = div_ceil(total_warps, static_cast<uint32_t>(kWarpsPerBlock));
    LaunchKernel(num_blocks, kThreadsPerBlock, device.unwrap()).enable_pdl(kUsePDL)(kernel_ptr, params);
  }
};

}  // namespace sglang::glm_ll::mla_kv_half
