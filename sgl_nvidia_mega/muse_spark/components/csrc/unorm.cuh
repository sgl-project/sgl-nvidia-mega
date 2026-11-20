// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// Portions Copyright 2023-2024 SGLang Team (Apache-2.0)
// Derived from SGLang (https://github.com/sgl-project/sglang), Apache License 2.0; see ../include/LICENSE.sglang.
// Source: SGLang python/sglang/kernels/jit/csrc/muse_spark_v12/norm_residual.cuh (muse_spark_norm_residual_kernel, the
// one-CTA-per-row path; the same file is kernels/jit/csrc/muse_spark/norm_residual.cuh on SGLang main) at revision a55ab7d.
// Modified by NVIDIA: the per-row arithmetic is unchanged (same element ownership per SGLang thread, same accumulation
// order, same two-level reductions and helpers, included from the installed SGLang), but
//   - one CTA owns kRows rows (a group of kG threads per row, each group thread playing kVT / kG of SGLang's kVT threads),
//     so the launch occupies fewer SMs and leaves room for the dependent grid;
//   - the dependent grid is released at kernel entry (griddepcontrol.launch_dependents carries no memory claim: the
//     dependent reads this kernel's outputs only after its own griddepcontrol.wait);
//   - the static operands (gate coefficients, norm weights; never written while a decode graph runs) are copied to
//     shared memory with cp.async before griddepcontrol.wait, and read from there (waited for right before first use).
//
// branch: [M, H] bf16; stream: [M, H] fp32; output: [M, H] bf16; new_stream: [M, H] fp32; router_out: [M, H] fp32 or null;
// post_weight / pre_weight: [H] bf16; alpha / beta: [H] fp32 (channel-wise residual gate; the served Muse Spark form).

#pragma once

#include <sgl_kernel/tensor.h>
#include <sgl_kernel/utils.h>

#include <sgl_kernel/cta.cuh>
#include <sgl_kernel/math.cuh>
#include <sgl_kernel/runtime.cuh>
#include <sgl_kernel/type.cuh>
#include <sgl_kernel/utils.cuh>
#include <sgl_kernel/vec.cuh>
#include <sgl_kernel/warp.cuh>

#include <tvm/ffi/container/tensor.h>
#include <tvm/ffi/optional.h>

#include "muse_spark_v12/norm_residual.cuh"

#include <cstdint>

namespace sglang {
namespace muse_vr_unorm {

// Sanitizer negative controls (test builds only; never defined in a served build). 1: invalid global write (memcheck);
// 2: shared-memory write-write race (racecheck); 3: barrier reached from divergent paths (synccheck); 4: read of
// uninitialized global memory (initcheck: the new_stream buffer before this kernel writes it).
#ifndef MUSE_VR_UNORM_NEGCTL
#define MUSE_VR_UNORM_NEGCTL 0
#endif
// Tests only (never defined in a served build): issue the second static-copy group late, so the first group's data is
// read while the second is still in flight (checks the commit-group accounting).
#ifndef MUSE_VR_UNORM_TEST_DELAY_G2
#define MUSE_VR_UNORM_TEST_DELAY_G2 0
#endif

// Activation and weight pointers carry no __restrict__: the activations are written by the predecessor grid while this
// one runs (before its griddepcontrol.wait), so every load of them must be a plain coherent load (no .nc path).
struct Params {
  const bf16_t* branch;
  const fp32_t* stream;
  bf16_t* __restrict__ output;
  fp32_t* __restrict__ new_stream;
  fp32_t* __restrict__ router_out;
  const bf16_t* post_weight;
  const bf16_t* pre_weight;
  const fp32_t* alpha;
  const fp32_t* beta;
  float post_eps;
  float pre_eps;
  int num_rows;
  bool pre_norm_bf16;
  bool static_pre_wait;  // false: the static copies are issued after griddepcontrol.wait (first call after a weight write)
  bool prefetch_stream;  // L2 prefetch hint for this CTA's residual-stream bytes before griddepcontrol.wait (no data read)
};

// L2 prefetch hint (PTX cp.async.bulk.prefetch.L2.global): returns no data to the program; the stream is read only by the
// plain loads after griddepcontrol.wait.
SGL_DEVICE void prefetch_l2(const void* src, uint32_t bytes) {
  asm volatile("cp.async.bulk.prefetch.L2.global [%0], %1;" ::"l"(src), "r"(bytes) : "memory");
}

SGL_DEVICE void cp_async16(void* smem_dst, const void* gmem_src) {
  const uint32_t d = static_cast<uint32_t>(__cvta_generic_to_shared(smem_dst));
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" ::"r"(d), "l"(gmem_src) : "memory");
}
SGL_DEVICE void cp_async_wait_all() { asm volatile("cp.async.wait_all;" ::: "memory"); }
SGL_DEVICE void cp_async_commit() { asm volatile("cp.async.commit_group;" ::: "memory"); }
template <int kN>
SGL_DEVICE void cp_async_wait_group() { asm volatile("cp.async.wait_group %0;" ::"n"(kN) : "memory"); }
SGL_DEVICE void launch_dependents() { asm volatile("griddepcontrol.launch_dependents;" ::: "memory"); }
SGL_DEVICE void grid_wait() { asm volatile("griddepcontrol.wait;" ::: "memory"); }

// Shared memory: [alpha fp32 H][beta fp32 H][pre_w bf16 H][post_w bf16 H, std form only][2 * kRows * kVT / 32 floats].
template <int64_t kHidden, int kVT, int kRows, bool kPostNormIsInputScale>
struct Smem {
  static constexpr int kVW = kVT / 32;
  static constexpr int64_t kAlpha = 0;
  static constexpr int64_t kBeta = kAlpha + kHidden * 4;
  static constexpr int64_t kPreW = kBeta + kHidden * 4;
  static constexpr int64_t kPostW = kPreW + kHidden * 2;
  static constexpr int64_t kRed = kPostW + (kPostNormIsInputScale ? 0 : kHidden * 2);
  static constexpr int64_t kBytes = kRed + 2 * kRows * kVW * 4;
};

// Two-level sum of SGLang's cta::reduce_sum<kVT> for one row played by a group of kG threads: per virtual warp a warp
// shuffle tree (warp::reduce_sum), its lane 0 stores the partial; a CTA barrier; then every lane reduces the kVW partials
// (lanes >= kVW contribute 0). Virtual thread v of group thread g is SGLang thread g + v * kG, so virtual warp
// (g / 32) + v * (kG / 32). All groups of the CTA call this together (the barrier is CTA-wide).
template <int kG, int kVT>
SGL_DEVICE float row_reduce(const float (&val)[kVT / kG], float* red, int g) {
  constexpr int kV = kVT / kG;
  constexpr int kVW = kVT / 32;
  const int lane = g % 32, gw = g / 32;
#pragma unroll
  for (int v = 0; v < kV; ++v) {
    const float s = device::warp::reduce_sum(val[v]);
    if (lane == 0) red[gw + v * (kG / 32)] = s;
  }
  __syncthreads();
  return device::warp::reduce_sum(lane < kVW ? red[lane] : 0.0f);
}

template <
    int64_t kHidden,
    int kVT,     // SGLang's thread count for this row count (256 below 192 rows, 512 from 192)
    int kRows,   // rows per CTA
    int kG,      // threads per row group
    bool kUsePDL,
    int kTrig,   // 0: launch_dependents at entry; 1: right after griddepcontrol.wait; 2: at the end (SGLang's place)
    bool kPostNormIsInputScale,
    bool kHasRouter>
__global__ void __launch_bounds__(kRows * kG, 1) unorm_kernel(const __grid_constant__ Params p) {
  using namespace device;
  using namespace device::norm_residual;
  constexpr int kThreads = kRows * kG;
  constexpr int kV = kVT / kG;
  constexpr int kPacks = kHidden / (kVT * kVec);
  constexpr int kVW = kVT / 32;
  using L = Smem<kHidden, kVT, kRows, kPostNormIsInputScale>;
  static_assert(kHidden % (kVT * kVec) == 0 && kVT % kG == 0 && kG % 32 == 0);
  static_assert(kFParts == 1, "fp32 rows are covered by one 256-bit vector per 8 elements on this target");

  using BF16Vec = AlignedVector<bf16x2_t, kPairs>;
  using F32Vec = AlignedVector<fp32x2_t, kFPairs>;

  extern __shared__ __align__(128) unsigned char smem_raw[];
  fp32_t* s_alpha = reinterpret_cast<fp32_t*>(smem_raw + L::kAlpha);
  fp32_t* s_beta = reinterpret_cast<fp32_t*>(smem_raw + L::kBeta);
  bf16_t* s_pre_w = reinterpret_cast<bf16_t*>(smem_raw + L::kPreW);
  bf16_t* s_post_w = reinterpret_cast<bf16_t*>(smem_raw + L::kPostW);
  float* s_red = reinterpret_cast<float*>(smem_raw + L::kRed);

  if constexpr (kUsePDL && kTrig == 0) launch_dependents();

  const int tid = threadIdx.x;
  const int grp = tid / kG;  // row slot in this CTA
  const int g = tid % kG;
  const int64_t row = static_cast<int64_t>(blockIdx.x) * kRows + grp;
  const bool live = row < p.num_rows;
  float* red1 = s_red + grp * kVW;
  float* red2 = s_red + kRows * kVW + grp * kVW;

  // Static operands -> shared memory (16-byte cp.async), in two commit groups: group 1 = gate coefficients (+ the
  // post-norm weight of the standard form), needed after the first reduction; group 2 = the pre-norm weight, needed after
  // the second. One row per CTA: each thread copies exactly the packs it consumes (thread-private: visible to it after its
  // own cp.async.wait_group, no barrier involved), and waits for each group right before its first use, so the copies
  // overlap the activation loads and the reductions. Several rows per CTA: the copies are CTA-strided and consumed by
  // every row group, so group 1 completes before the first reduction's CTA barrier and group 2 before the second's.
  const auto copy_static = [&]() {
    if constexpr (kRows == 1) {
#pragma unroll
      for (int v = 0; v < kV; ++v)
#pragma unroll
        for (int pk = 0; pk < kPacks; ++pk) {
          const int idx = g + v * kG + pk * kVT;
          cp_async16(reinterpret_cast<unsigned char*>(s_alpha) + idx * 32, reinterpret_cast<const unsigned char*>(p.alpha) + idx * 32);
          cp_async16(reinterpret_cast<unsigned char*>(s_alpha) + idx * 32 + 16, reinterpret_cast<const unsigned char*>(p.alpha) + idx * 32 + 16);
          cp_async16(reinterpret_cast<unsigned char*>(s_beta) + idx * 32, reinterpret_cast<const unsigned char*>(p.beta) + idx * 32);
          cp_async16(reinterpret_cast<unsigned char*>(s_beta) + idx * 32 + 16, reinterpret_cast<const unsigned char*>(p.beta) + idx * 32 + 16);
          if constexpr (!kPostNormIsInputScale)
            cp_async16(reinterpret_cast<unsigned char*>(s_post_w) + idx * 16, reinterpret_cast<const unsigned char*>(p.post_weight) + idx * 16);
        }
      cp_async_commit();  // group 1 (gate coefficients [+ post weight]); no other group precedes it
      if constexpr (MUSE_VR_UNORM_TEST_DELAY_G2) __nanosleep(20000);  // tests only: group 2 issued late
#pragma unroll
      for (int v = 0; v < kV; ++v)
#pragma unroll
        for (int pk = 0; pk < kPacks; ++pk) {
          const int idx = g + v * kG + pk * kVT;
          cp_async16(reinterpret_cast<unsigned char*>(s_pre_w) + idx * 16, reinterpret_cast<const unsigned char*>(p.pre_weight) + idx * 16);
        }
      cp_async_commit();
    } else {
      constexpr int kA = static_cast<int>(kHidden * 4 / 16);  // 16-byte chunks per fp32 vector
      constexpr int kW = static_cast<int>(kHidden * 2 / 16);  // per bf16 vector
      for (int i = tid; i < kA; i += kThreads) {
        cp_async16(reinterpret_cast<unsigned char*>(s_alpha) + i * 16, reinterpret_cast<const unsigned char*>(p.alpha) + i * 16);
        cp_async16(reinterpret_cast<unsigned char*>(s_beta) + i * 16, reinterpret_cast<const unsigned char*>(p.beta) + i * 16);
      }
      if constexpr (!kPostNormIsInputScale) {
        for (int i = tid; i < kW; i += kThreads)
          cp_async16(reinterpret_cast<unsigned char*>(s_post_w) + i * 16, reinterpret_cast<const unsigned char*>(p.post_weight) + i * 16);
      }
      cp_async_commit();  // group 1 (gate coefficients [+ post weight]); no other group precedes it
      if constexpr (MUSE_VR_UNORM_TEST_DELAY_G2) __nanosleep(20000);  // tests only: group 2 issued late
      for (int i = tid; i < kW; i += kThreads)
        cp_async16(reinterpret_cast<unsigned char*>(s_pre_w) + i * 16, reinterpret_cast<const unsigned char*>(p.pre_weight) + i * 16);
      cp_async_commit();
    }
  };

  // Scale form: the post-norm weight multiplies the branch before the first reduction; each thread loads its own packs
  // into registers (static, so before the wait when allowed), as SGLang does after its wait.
  BF16Vec post_wv[kPostNormIsInputScale ? kV : 1][kPostNormIsInputScale ? kPacks : 1];
  const auto load_post_w = [&]() {
    if constexpr (kPostNormIsInputScale) {
#pragma unroll
      for (int v = 0; v < kV; ++v)
#pragma unroll
        for (int pk = 0; pk < kPacks; ++pk) post_wv[v][pk].load(p.post_weight, g + v * kG + pk * kVT);
    }
  };

  if (p.static_pre_wait) {
    copy_static();
    load_post_w();
  }
  if (p.prefetch_stream && g == 0 && row < p.num_rows)  // one hint per owned row: its whole fp32 stream row
    prefetch_l2(p.stream + row * kHidden, static_cast<uint32_t>(kHidden * 4));
  if constexpr (kUsePDL) grid_wait();
  if constexpr (kUsePDL && kTrig == 1) launch_dependents();
  if (!p.static_pre_wait) {
    copy_static();
    load_post_w();
  }

  if constexpr (MUSE_VR_UNORM_NEGCTL == 1) {
    if (blockIdx.x == 0 && tid == 0) *reinterpret_cast<volatile int*>(static_cast<uintptr_t>(64)) = 1;
  }
  if constexpr (MUSE_VR_UNORM_NEGCTL == 4) {
    if (blockIdx.x == 0 && tid == 0) s_red[0] = *reinterpret_cast<volatile const float*>(p.new_stream);
  }

  const bf16_t* branch = p.branch + (live ? row : 0) * kHidden;
  const fp32_t* stream = p.stream + (live ? row : 0) * kHidden;
  bf16_t* out = p.output + row * kHidden;
  fp32_t* new_stream = p.new_stream + row * kHidden;
  fp32_t* router = kHasRouter ? p.router_out + row * kHidden : nullptr;
  const bool pre_bf16 = p.pre_norm_bf16;

  fp32x2_t vals[kV][kPacks][kPairs];
  F32Vec stream_v[kV][kPacks];
  float sum_sq[kV];
#pragma unroll
  for (int v = 0; v < kV; ++v) {
    sum_sq[v] = 0.0f;
#pragma unroll
    for (int pk = 0; pk < kPacks; ++pk) {
      const int idx = g + v * kG + pk * kVT;
      BF16Vec branch_v;
      if (live) {
        branch_v.load(branch, idx);
        stream_v[v][pk].load(stream, idx);
      } else {
#pragma unroll
        for (int q = 0; q < kPairs; ++q) branch_v[q] = cast<bf16x2_t>(fp32x2_t{0.0f, 0.0f});
#pragma unroll
        for (int q = 0; q < kFPairs; ++q) stream_v[v][pk][q] = fp32x2_t{0.0f, 0.0f};
      }
#pragma unroll
      for (int q = 0; q < kPairs; ++q) {
        auto x = cast<fp32x2_t>(branch_v[q]);
        if constexpr (kPostNormIsInputScale) {
          const auto w = cast<fp32x2_t>(post_wv[v][pk][q]);
          x.x *= w.x;
          x.y *= w.y;
          x = round_bf16(x);
        }
        vals[v][pk][q] = x;
        sum_sq[v] = fmaf(x.x, x.x, sum_sq[v]);
        sum_sq[v] = fmaf(x.y, x.y, sum_sq[v]);
      }
    }
  }

  if constexpr (kRows > 1) cp_async_wait_group<1>();  // group 1 before the CTA barrier inside the first reduction
  if constexpr (MUSE_VR_UNORM_NEGCTL == 2) {
    if (tid == 32) s_red[2 * kRows * kVW - 1] = 1.0f;
    if (tid == 64) s_red[2 * kRows * kVW - 1] = 2.0f;
  }
  if constexpr (MUSE_VR_UNORM_NEGCTL == 3) {
    if (tid % 32 < 16) __syncthreads();
  }
  const float post_tot = row_reduce<kG, kVT>(sum_sq, red1, g);
  const float post_rstd = math::rsqrt(post_tot / kHidden + p.post_eps);
  if constexpr (kRows == 1) cp_async_wait_group<1>();  // own group-1 copies, right before their first use

  float pre_sum_sq[kV];
#pragma unroll
  for (int v = 0; v < kV; ++v) {
    pre_sum_sq[v] = 0.0f;
#pragma unroll
    for (int pk = 0; pk < kPacks; ++pk) {
      const int idx = g + v * kG + pk * kVT;
      F32Vec alpha_v, beta_v;
      alpha_v.load(s_alpha, idx);
      beta_v.load(s_beta, idx);
      BF16Vec post_w_v;
      if constexpr (!kPostNormIsInputScale) post_w_v.load(s_post_w, idx);
      F32Vec new_stream_v;
#pragma unroll
      for (int q = 0; q < kPairs; ++q) {
        fp32x2_t normed_b = vals[v][pk][q];
        normed_b.x *= post_rstd;
        normed_b.y *= post_rstd;
        if constexpr (!kPostNormIsInputScale) {
          const auto w = cast<fp32x2_t>(post_w_v[q]);
          normed_b.x *= w.x;
          normed_b.y *= w.y;
        }
        normed_b = round_bf16(normed_b);
        const fp32x2_t s = stream_v[v][pk][q];
        const fp32x2_t ga = alpha_v[q];
        const fp32x2_t gb = beta_v[q];
        fp32x2_t new_s;
        new_s.x = ga.x * s.x + gb.x * normed_b.x;
        new_s.y = ga.y * s.y + gb.y * normed_b.y;
        new_stream_v[q] = new_s;
        const fp32x2_t pre_v = pre_bf16 ? round_bf16(new_s) : new_s;
        vals[v][pk][q] = pre_v;
        pre_sum_sq[v] = fmaf(pre_v.x, pre_v.x, pre_sum_sq[v]);
        pre_sum_sq[v] = fmaf(pre_v.y, pre_v.y, pre_sum_sq[v]);
      }
      if (live) new_stream_v.store(new_stream, idx);
    }
  }

  if constexpr (kRows > 1) cp_async_wait_all();  // group 2 before the CTA barrier inside the second reduction
  const float pre_tot = row_reduce<kG, kVT>(pre_sum_sq, red2, g);
  const float pre_rstd = math::rsqrt(pre_tot / kHidden + p.pre_eps);
  if constexpr (kRows == 1) cp_async_wait_all();  // own group-2 copies

#pragma unroll
  for (int v = 0; v < kV; ++v) {
#pragma unroll
    for (int pk = 0; pk < kPacks; ++pk) {
      const int idx = g + v * kG + pk * kVT;
      BF16Vec weight_v;
      weight_v.load(s_pre_w, idx);
      BF16Vec out_v;
      F32Vec router_v;
#pragma unroll
      for (int q = 0; q < kPairs; ++q) {
        fp32x2_t x = vals[v][pk][q];
        x.x *= pre_rstd;
        x.y *= pre_rstd;
        const auto wf = cast<fp32x2_t>(weight_v[q]);
        x.x *= wf.x;
        x.y *= wf.y;
        if (pre_bf16) x = round_bf16(x);
        out_v[q] = cast<bf16x2_t>(x);
        if constexpr (kHasRouter) router_v[q] = x;
      }
      if (live) {
        out_v.store(out, idx);
        if constexpr (kHasRouter) router_v.store(router, idx);
      }
    }
  }

  if constexpr (kUsePDL && kTrig == 2) launch_dependents();
}

inline constexpr int kGroupThreads = 256;

template <int64_t kHidden, int kVT, int kRows, bool kUsePDL, int kTrig, bool kPostNormIsInputScale, bool kHasRouter>
struct UnormNormResidual {
  static constexpr int kG = kGroupThreads;
  using L = Smem<kHidden, kVT, kRows, kPostNormIsInputScale>;

  static void
  run(const tvm::ffi::TensorView branch,
      const tvm::ffi::TensorView stream,
      const tvm::ffi::TensorView output,
      const tvm::ffi::TensorView new_stream,
      const tvm::ffi::Optional<tvm::ffi::TensorView> router_out,
      const tvm::ffi::TensorView post_weight,
      const tvm::ffi::TensorView pre_weight,
      const tvm::ffi::TensorView alpha,
      const tvm::ffi::TensorView beta,
      double post_eps,
      double pre_eps,
      bool pre_norm_bf16,
      bool static_pre_wait,
      bool prefetch_stream) {
    using namespace host;
    auto M = SymbolicSize{"num_tokens"};
    auto device_ = SymbolicDevice{};
    device_.set_options<kDLCUDA>();
    TensorMatcher({M, kHidden}).with_dtype<bf16_t>().with_device(device_).verify(branch);
    TensorMatcher({M, kHidden}).with_dtype<fp32_t>().with_device(device_).verify(stream);
    TensorMatcher({M, kHidden}).with_dtype<bf16_t>().with_device(device_).verify(output);
    TensorMatcher({M, kHidden}).with_dtype<fp32_t>().with_device(device_).verify(new_stream);
    RuntimeCheck(router_out.has_value() == kHasRouter, "router_out presence must match the kHasRouter instantiation");
    if constexpr (kHasRouter) TensorMatcher({M, kHidden}).with_dtype<fp32_t>().with_device(device_).verify(router_out.value());
    TensorMatcher({kHidden}).with_dtype<bf16_t>().with_device(device_).verify(post_weight);
    TensorMatcher({kHidden}).with_dtype<bf16_t>().with_device(device_).verify(pre_weight);
    TensorMatcher({kHidden}).with_dtype<fp32_t>().with_device(device_).verify(alpha);
    TensorMatcher({kHidden}).with_dtype<fp32_t>().with_device(device_).verify(beta);
    const int num_rows = static_cast<int>(M.unwrap());
    if (num_rows == 0) return;
    const Params params{
        .branch = static_cast<const bf16_t*>(branch.data_ptr()),
        .stream = static_cast<const fp32_t*>(stream.data_ptr()),
        .output = static_cast<bf16_t*>(output.data_ptr()),
        .new_stream = static_cast<fp32_t*>(new_stream.data_ptr()),
        .router_out = kHasRouter ? static_cast<fp32_t*>(router_out.value().data_ptr()) : nullptr,
        .post_weight = static_cast<const bf16_t*>(post_weight.data_ptr()),
        .pre_weight = static_cast<const bf16_t*>(pre_weight.data_ptr()),
        .alpha = static_cast<const fp32_t*>(alpha.data_ptr()),
        .beta = static_cast<const fp32_t*>(beta.data_ptr()),
        .post_eps = static_cast<float>(post_eps),
        .pre_eps = static_cast<float>(pre_eps),
        .num_rows = num_rows,
        .pre_norm_bf16 = pre_norm_bf16,
        .static_pre_wait = static_pre_wait,
        .prefetch_stream = prefetch_stream,
    };
    constexpr auto kernel = unorm_kernel<kHidden, kVT, kRows, kG, kUsePDL, kTrig, kPostNormIsInputScale, kHasRouter>;
    [[maybe_unused]] static const bool configured = [] {
      RuntimeDeviceCheck(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, static_cast<int>(L::kBytes)));
      return true;
    }();
    const unsigned grid = static_cast<unsigned>((num_rows + kRows - 1) / kRows);
    LaunchKernel(grid, kRows * kG, device_.unwrap(), static_cast<std::size_t>(L::kBytes))
        .config({.use_pdl = kUsePDL})(kernel, params);
  }
};

}  // namespace muse_vr_unorm
}  // namespace sglang
