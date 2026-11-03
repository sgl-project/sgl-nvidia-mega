// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// Portions Copyright 2023-2024 SGLang Team (Apache-2.0)
// Derived from SGLang (https://github.com/sgl-project/sglang), Apache License 2.0; see ../include/LICENSE.sglang.
// Source: SGLang python/sglang/kernels/jit/csrc/muse_spark_v12/norm_residual.cuh (muse_spark_norm_residual_kernel, the
// one-CTA-per-row path, and the cluster partial exchange of muse_spark_norm_residual_cluster; the same file is
// kernels/jit/csrc/muse_spark/norm_residual.cuh on SGLang main) at revision a55ab7d, through this package's csrc/unorm.cuh.
// Modified by NVIDIA: column-split form of csrc/unorm.cuh. A cluster of kK CTAs owns kRows rows; CTA k plays SGLang's
// threads [k * kVT / kK, (k + 1) * kVT / kK) of every owned row (same element ownership, same per-thread accumulation
// order), so it stages only its slice of the static operands. Each reduction keeps SGLang's cta::reduce_sum order: every
// CTA computes the partials of its virtual warps, sends them to the other CTAs of the cluster (st.async with an mbarrier
// transaction count, as SGLang's cluster path does), and every CTA then reduces the full set of virtual-warp partials in
// SGLang's slot order. Dependent release at entry and static copies before griddepcontrol.wait as in csrc/unorm.cuh.

#pragma once

#include <sgl_kernel/mbarrier.cuh>

#include <cooperative_groups.h>

#include "unorm.cuh"

namespace sglang {
namespace muse_vr_unorm {

// Tests only (never defined in a served build): MUSE_VR_UNORM_CL_TEST_BOUNDED makes the partial-exchange wait bounded
// (trap on expiry instead of waiting forever); MUSE_VR_UNORM_NEGCTL 5 = one sender skips one partial (must end in that
// trap), 6 = a plain remote store races the owner's local store to the same slot (racecheck must report it).
#ifndef MUSE_VR_UNORM_CL_TEST_BOUNDED
#define MUSE_VR_UNORM_CL_TEST_BOUNDED 0
#endif

inline constexpr uint32_t kPartialBytes = 4;  // one fp32 partial per st.async (.b32)

SGL_DEVICE void mbar_wait_exchange(uint64_t* bar) {
  if constexpr (MUSE_VR_UNORM_CL_TEST_BOUNDED) {
    const uint32_t a = device::ptx::to_shared(bar);
    for (uint32_t i = 0;; ++i) {
      uint32_t ready = 0;
      asm volatile(
          "{\n\t.reg .pred p;\n\tmbarrier.try_wait.parity.acquire.cluster.shared::cta.b64 p, [%1], 0;\n\t"
          "selp.b32 %0, 1, 0, p;\n\t}"
          : "=r"(ready)
          : "r"(a)
          : "memory");
      if (ready) return;
      if (i > (1u << 22)) asm volatile("trap;");
    }
  } else {
    device::ptx::mbar_wait_parity_acquire_cluster(bar);  // SGLang's own wait (unbounded), as its cluster path
  }
}

SGL_DEVICE void st_async_f32(float* local_dst, float value, uint64_t* local_bar, int peer) {
  static_assert(sizeof(float) == kPartialBytes);
  uint32_t rd, rb;
  const uint32_t d = device::ptx::to_shared(local_dst);
  const uint32_t b = device::ptx::to_shared(local_bar);
  asm volatile("mapa.shared::cluster.u32 %0, %1, %2;" : "=r"(rd) : "r"(d), "r"(peer));
  asm volatile("mapa.shared::cluster.u32 %0, %1, %2;" : "=r"(rb) : "r"(b), "r"(peer));
  asm volatile("st.async.shared::cluster.mbarrier::complete_tx::bytes.b32 [%0], %1, [%2];"
               :
               : "r"(rd), "r"(__float_as_uint(value)), "r"(rb)
               : "memory");
}

// Shared memory of the column-split form: static slices [alpha fp32][beta fp32][pre_w bf16][post_w bf16, std form]
// (kHidden / kK elements each), reduction slots 2 x kRows x kVW floats, two mbarriers.
template <int64_t kHidden, int kVT, int kRows, int kK, bool kPostNormIsInputScale>
struct SmemCl {
  static constexpr int kVW = kVT / 32;
  static constexpr int64_t kE = kHidden / kK;
  static constexpr int64_t kAlpha = 0;
  static constexpr int64_t kBeta = kAlpha + kE * 4;
  static constexpr int64_t kPreW = kBeta + kE * 4;
  static constexpr int64_t kPostW = kPreW + kE * 2;
  static constexpr int64_t kRed = kPostW + (kPostNormIsInputScale ? 0 : kE * 2);
  static constexpr int64_t kBar = kRed + ((2 * kRows * kVW * 4 + 7) / 8) * 8;
  static constexpr int64_t kBytes = kBar + 16;
};

template <
    int64_t kHidden,
    int kVT,     // SGLang's thread count for this row count
    int kRows,   // rows per cluster
    int kK,      // CTAs per cluster (column split)
    bool kUsePDL,
    int kTrig,
    bool kPostNormIsInputScale,
    bool kHasRouter>
__global__ void __launch_bounds__(kRows * (kVT / kK), 1) unorm_cl_kernel(const __grid_constant__ Params p) {
  using namespace device;
  using namespace device::norm_residual;
  constexpr int kG = kVT / kK;  // threads per row in this CTA (one SGLang thread each)
  constexpr int kThreads = kRows * kG;
  constexpr int kPacks = kHidden / (kVT * kVec);
  constexpr int kVW = kVT / 32;
  constexpr int kVWc = kG / 32;  // virtual warps per row in this CTA
  constexpr int kWarpsPerCta = kThreads / 32;    // every warp sends one partial to each peer per reduction
  static_assert(kWarpsPerCta == kRows * kVWc);
  constexpr uint32_t kTxBytes = (kK - 1) * kWarpsPerCta * kPartialBytes;  // = what the peers' send loops deliver
  using L = SmemCl<kHidden, kVT, kRows, kK, kPostNormIsInputScale>;
  static_assert(kHidden % (kVT * kVec) == 0 && kVT % kK == 0 && kG % 32 == 0 && kK >= 2 && kK <= 8);
  static_assert(kFParts == 1, "fp32 rows are covered by one 256-bit vector per 8 elements on this target");

  using BF16Vec = AlignedVector<bf16x2_t, kPairs>;
  using F32Vec = AlignedVector<fp32x2_t, kFPairs>;

  extern __shared__ __align__(128) unsigned char smem_raw[];
  fp32_t* s_alpha = reinterpret_cast<fp32_t*>(smem_raw + L::kAlpha);
  fp32_t* s_beta = reinterpret_cast<fp32_t*>(smem_raw + L::kBeta);
  bf16_t* s_pre_w = reinterpret_cast<bf16_t*>(smem_raw + L::kPreW);
  bf16_t* s_post_w = reinterpret_cast<bf16_t*>(smem_raw + L::kPostW);
  float* s_red = reinterpret_cast<float*>(smem_raw + L::kRed);
  uint64_t* s_bar = reinterpret_cast<uint64_t*>(smem_raw + L::kBar);

  if constexpr (kUsePDL && kTrig == 0) launch_dependents();

  const int tid = threadIdx.x;
  const int k = static_cast<int>(cooperative_groups::this_cluster().block_rank());
  const int grp = tid / kG;
  const int g = tid % kG;
  const int vt0 = k * kG;  // first SGLang thread of this CTA in every row
  const int64_t row = static_cast<int64_t>(blockIdx.x / kK) * kRows + grp;
  const bool live = row < p.num_rows;
  const int lane = g % 32;
  const int vw = (vt0 + g) / 32;  // this warp's virtual warp in the row

  if (tid == 0) {
    ptx::mbar_init(&s_bar[0], 1);
    ptx::mbar_init(&s_bar[1], 1);
    ptx::mbar_arrive_expect_tx(&s_bar[0], kTxBytes);
    ptx::mbar_arrive_expect_tx(&s_bar[1], kTxBytes);
    ptx::publish_cluster_barriers();
  }
  ptx::cluster_arrive_relaxed();

  // Static slices: pack j of this CTA's packs in phase pk is SGLang pack vt0 + j + pk * kVT; local index pk * kG + j.
  // Two commit groups (gate coefficients [+ standard-form post weight]; pre weight). One row per cluster: each thread
  // copies the packs it consumes (thread-private) and waits right before first use; several rows: CTA-strided copies,
  // group 1 complete before the first reduction's barriers, group 2 before the second's (as csrc/unorm.cuh).
  const auto copy_static = [&]() {
    if constexpr (kRows == 1) {
#pragma unroll
      for (int pk = 0; pk < kPacks; ++pk) {
        const int li = pk * kG + g;
        const int64_t gi = static_cast<int64_t>(vt0) + g + pk * kVT;
        cp_async16(reinterpret_cast<unsigned char*>(s_alpha) + li * 32, reinterpret_cast<const unsigned char*>(p.alpha) + gi * 32);
        cp_async16(reinterpret_cast<unsigned char*>(s_alpha) + li * 32 + 16, reinterpret_cast<const unsigned char*>(p.alpha) + gi * 32 + 16);
        cp_async16(reinterpret_cast<unsigned char*>(s_beta) + li * 32, reinterpret_cast<const unsigned char*>(p.beta) + gi * 32);
        cp_async16(reinterpret_cast<unsigned char*>(s_beta) + li * 32 + 16, reinterpret_cast<const unsigned char*>(p.beta) + gi * 32 + 16);
        if constexpr (!kPostNormIsInputScale)
          cp_async16(reinterpret_cast<unsigned char*>(s_post_w) + li * 16, reinterpret_cast<const unsigned char*>(p.post_weight) + gi * 16);
      }
      cp_async_commit();  // group 1 (gate coefficients [+ post weight]); no other group precedes it
      if constexpr (MUSE_VR_UNORM_TEST_DELAY_G2) __nanosleep(20000);  // tests only: group 2 issued late
#pragma unroll
      for (int pk = 0; pk < kPacks; ++pk) {
        const int li = pk * kG + g;
        const int64_t gi = static_cast<int64_t>(vt0) + g + pk * kVT;
        cp_async16(reinterpret_cast<unsigned char*>(s_pre_w) + li * 16, reinterpret_cast<const unsigned char*>(p.pre_weight) + gi * 16);
      }
      cp_async_commit();
    } else {
      constexpr int kA = kPacks * kG * 2;  // 16-byte chunks per fp32 slice (32 bytes per pack)
      constexpr int kW = kPacks * kG;      // per bf16 slice (16 bytes per pack)
      for (int c = tid; c < kA; c += kThreads) {
        const int pk = c / (kG * 2), r = c % (kG * 2);
        const int64_t goff = (static_cast<int64_t>(vt0) + pk * kVT) * 32 + r * 16;
        cp_async16(reinterpret_cast<unsigned char*>(s_alpha) + c * 16, reinterpret_cast<const unsigned char*>(p.alpha) + goff);
        cp_async16(reinterpret_cast<unsigned char*>(s_beta) + c * 16, reinterpret_cast<const unsigned char*>(p.beta) + goff);
      }
      if constexpr (!kPostNormIsInputScale) {
        for (int c = tid; c < kW; c += kThreads) {
          const int pk = c / kG, j = c % kG;
          const int64_t goff = (static_cast<int64_t>(vt0) + j + pk * kVT) * 16;
          cp_async16(reinterpret_cast<unsigned char*>(s_post_w) + c * 16, reinterpret_cast<const unsigned char*>(p.post_weight) + goff);
        }
      }
      cp_async_commit();  // group 1 (gate coefficients [+ post weight]); no other group precedes it
      if constexpr (MUSE_VR_UNORM_TEST_DELAY_G2) __nanosleep(20000);  // tests only: group 2 issued late
      for (int c = tid; c < kW; c += kThreads) {
        const int pk = c / kG, j = c % kG;
        const int64_t goff = (static_cast<int64_t>(vt0) + j + pk * kVT) * 16;
        cp_async16(reinterpret_cast<unsigned char*>(s_pre_w) + c * 16, reinterpret_cast<const unsigned char*>(p.pre_weight) + goff);
      }
      cp_async_commit();
    }
  };
  BF16Vec post_wv[kPostNormIsInputScale ? kPacks : 1];
  const auto load_post_w = [&]() {
    if constexpr (kPostNormIsInputScale) {
#pragma unroll
      for (int pk = 0; pk < kPacks; ++pk) post_wv[pk].load(p.post_weight, vt0 + g + pk * kVT);
    }
  };

  if (p.static_pre_wait) {
    copy_static();
    load_post_w();
  }
  if (p.prefetch_stream && g < kPacks && row < p.num_rows)  // L2 hints for this CTA's stream slice of its row (one per pack)
    prefetch_l2(p.stream + row * kHidden + (static_cast<int64_t>(vt0) + g * kVT) * 8, static_cast<uint32_t>(kG * 8 * 4));
  if constexpr (kUsePDL) grid_wait();
  if constexpr (kUsePDL && kTrig == 1) launch_dependents();
  if (!p.static_pre_wait) {
    copy_static();
    load_post_w();
  }

  const bf16_t* branch = p.branch + (live ? row : 0) * kHidden;
  const fp32_t* stream = p.stream + (live ? row : 0) * kHidden;
  bf16_t* out = p.output + row * kHidden;
  fp32_t* new_stream = p.new_stream + row * kHidden;
  fp32_t* router = kHasRouter ? p.router_out + row * kHidden : nullptr;
  const bool pre_bf16 = p.pre_norm_bf16;

  // Cross-CTA sum in SGLang's order: own virtual-warp partials go to every CTA's slot vw (locally by store, to the
  // peers by st.async); a CTA barrier + this CTA's mbarrier phase complete the slot set; every lane reduces it.
  const auto row_reduce_cl = [&](float val, float* red, uint64_t* bar) -> float {
    const float s = warp::reduce_sum(val);
    if (lane == 0) {
      red[vw] = s;
#pragma unroll
      for (int q = 1; q < kK; ++q) {
        if constexpr (MUSE_VR_UNORM_NEGCTL == 5) {
          if (k == 0 && tid == 0 && q == 1) continue;  // one partial never sent
        }
        st_async_f32(&red[vw], s, bar, (k + q) % kK);  // one kPartialBytes send per (warp, peer)
      }
    }
    if constexpr (MUSE_VR_UNORM_NEGCTL == 6) {
      if (k == 1 && tid == 32) {  // plain remote store into CTA 0's slot 0 while CTA 0's lane 0 writes it locally
        uint32_t rd;
        const uint32_t d = ptx::to_shared(&red[0]);
        asm volatile("mapa.shared::cluster.u32 %0, %1, %2;" : "=r"(rd) : "r"(d), "r"(0));
        asm volatile("st.shared::cluster.b32 [%0], %1;" ::"r"(rd), "r"(0u) : "memory");
      }
    }
    __syncthreads();
    mbar_wait_exchange(bar);
    return warp::reduce_sum(lane < kVW ? red[lane] : 0.0f);
  };

  fp32x2_t vals[kPacks][kPairs];
  F32Vec stream_v[kPacks];
  float sum_sq = 0.0f;
#pragma unroll
  for (int pk = 0; pk < kPacks; ++pk) {
    const int idx = vt0 + g + pk * kVT;
    BF16Vec branch_v;
    if (live) {
      branch_v.load(branch, idx);
      stream_v[pk].load(stream, idx);
    } else {
#pragma unroll
      for (int q = 0; q < kPairs; ++q) branch_v[q] = cast<bf16x2_t>(fp32x2_t{0.0f, 0.0f});
#pragma unroll
      for (int q = 0; q < kFPairs; ++q) stream_v[pk][q] = fp32x2_t{0.0f, 0.0f};
    }
#pragma unroll
    for (int q = 0; q < kPairs; ++q) {
      auto x = cast<fp32x2_t>(branch_v[q]);
      if constexpr (kPostNormIsInputScale) {
        const auto w = cast<fp32x2_t>(post_wv[pk][q]);
        x.x *= w.x;
        x.y *= w.y;
        x = round_bf16(x);
      }
      vals[pk][q] = x;
      sum_sq = fmaf(x.x, x.x, sum_sq);
      sum_sq = fmaf(x.y, x.y, sum_sq);
    }
  }

  if constexpr (kRows > 1) cp_async_wait_group<1>();  // group 1 before the first reduction's barriers
  ptx::cluster_wait();  // every CTA of the cluster has initialised its mbarriers before any st.async
  const float post_tot = row_reduce_cl(sum_sq, s_red + grp * kVW, &s_bar[0]);
  const float post_rstd = math::rsqrt(post_tot / kHidden + p.post_eps);
  if constexpr (kRows == 1) cp_async_wait_group<1>();  // own group-1 copies, right before their first use

  float pre_sum_sq = 0.0f;
#pragma unroll
  for (int pk = 0; pk < kPacks; ++pk) {
    const int idx = vt0 + g + pk * kVT;
    const int li = pk * kG + g;  // this pack's slot in the static slices
    F32Vec alpha_v, beta_v;
    alpha_v.load(s_alpha, li);
    beta_v.load(s_beta, li);
    BF16Vec post_w_v;
    if constexpr (!kPostNormIsInputScale) post_w_v.load(s_post_w, li);
    F32Vec new_stream_v;
#pragma unroll
    for (int q = 0; q < kPairs; ++q) {
      fp32x2_t normed_b = vals[pk][q];
      normed_b.x *= post_rstd;
      normed_b.y *= post_rstd;
      if constexpr (!kPostNormIsInputScale) {
        const auto w = cast<fp32x2_t>(post_w_v[q]);
        normed_b.x *= w.x;
        normed_b.y *= w.y;
      }
      normed_b = round_bf16(normed_b);
      const fp32x2_t s = stream_v[pk][q];
      const fp32x2_t ga = alpha_v[q];
      const fp32x2_t gb = beta_v[q];
      fp32x2_t new_s;
      new_s.x = ga.x * s.x + gb.x * normed_b.x;
      new_s.y = ga.y * s.y + gb.y * normed_b.y;
      new_stream_v[q] = new_s;
      const fp32x2_t pre_v = pre_bf16 ? round_bf16(new_s) : new_s;
      vals[pk][q] = pre_v;
      pre_sum_sq = fmaf(pre_v.x, pre_v.x, pre_sum_sq);
      pre_sum_sq = fmaf(pre_v.y, pre_v.y, pre_sum_sq);
    }
    if (live) new_stream_v.store(new_stream, idx);
  }

  if constexpr (kRows > 1) cp_async_wait_all();  // group 2 before the second reduction's barriers
  const float pre_tot = row_reduce_cl(pre_sum_sq, s_red + kRows * kVW + grp * kVW, &s_bar[1]);
  const float pre_rstd = math::rsqrt(pre_tot / kHidden + p.pre_eps);
  if constexpr (kRows == 1) cp_async_wait_all();  // own group-2 copies

#pragma unroll
  for (int pk = 0; pk < kPacks; ++pk) {
    const int idx = vt0 + g + pk * kVT;
    BF16Vec weight_v;
    weight_v.load(s_pre_w, pk * kG + g);
    BF16Vec out_v;
    F32Vec router_v;
#pragma unroll
    for (int q = 0; q < kPairs; ++q) {
      fp32x2_t x = vals[pk][q];
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

  if constexpr (kUsePDL && kTrig == 2) launch_dependents();
}

template <int64_t kHidden, int kVT, int kRows, int kK, bool kUsePDL, int kTrig, bool kPostNormIsInputScale, bool kHasRouter>
struct UnormClNormResidual {
  using L = SmemCl<kHidden, kVT, kRows, kK, kPostNormIsInputScale>;

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
    constexpr auto kernel = unorm_cl_kernel<kHidden, kVT, kRows, kK, kUsePDL, kTrig, kPostNormIsInputScale, kHasRouter>;
    [[maybe_unused]] static const bool configured = [] {
      RuntimeDeviceCheck(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, static_cast<int>(L::kBytes)));
      return true;
    }();
    const unsigned groups = static_cast<unsigned>((num_rows + kRows - 1) / kRows);
    LaunchKernel(groups * kK, kRows * (kVT / kK), device_.unwrap(), static_cast<std::size_t>(L::kBytes))
        .config({.use_pdl = kUsePDL, .cluster_dim = dim3(kK, 1, 1)})(kernel, params);
  }
};

}  // namespace muse_vr_unorm
}  // namespace sglang
