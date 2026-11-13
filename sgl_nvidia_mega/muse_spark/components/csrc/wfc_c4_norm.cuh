// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// Portions Copyright 2023-2024 SGLang Team (Apache-2.0)
// Derived from SGLang (https://github.com/sgl-project/sglang), Apache License 2.0; see ../include/LICENSE.sglang.
// Source: SGLang python/sglang/kernels/jit/csrc/muse_spark/norm_residual.cuh (muse_spark_norm_residual_kernel, the
// one-CTA-per-row path) at revision 306ccf6; the helpers it uses come from the sgl_kernel headers unchanged.
// Modified by NVIDIA: the per-row body runs as a device function (role) for a group of threads that a larger launch
// assigns to one owned row. The group may be smaller than the thread count of the launch policy SGLang picks for the step;
// each group thread then plays several of those threads (virtual threads), with the same element ownership, the same
// per-thread accumulation order and the same two-level reductions (warp shuffle tree, then one warp over the per-warp
// partials in warp order), so every output bit equals SGLang's kernel at that thread count. The group synchronizes through
// a named barrier. Outputs can also be written in the layouts of the front roles (wfc_c4f.cuh): the pooler-input chunk
// image and the router-input A-fragment image.
// Modified by NVIDIA: owner_norm_row_cl is derived the same way from the cluster path of the same file
// (muse_spark_norm_residual_cluster, its local and cross-CTA moment reductions) for the steps where SGLang launches one row
// as a cluster of CTAs.
//
// Owner-row boundary role (dev; not wired into the server):
//   post-norm(branch) -> gated residual (new_stream = alpha * stream + beta * bf16(normed branch)) -> pre-norm(new_stream)
//   -> output (bf16) and router input (the fp32 value before the bf16 output cast), per SGLang's kernel.

#pragma once

#include <sgl_kernel/math.cuh>
#include <sgl_kernel/type.cuh>
#include <sgl_kernel/utils.cuh>
#include <sgl_kernel/warp.cuh>

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cstdint>

namespace sglang {
namespace muse_vr_c4n {

inline constexpr int kVec = 8;  // elements per owned block (SGLang's kVec)

struct NormArgs {
  const uint16_t* branch;     // [M, H] bf16 row-major (row stride H)
  const float* stream;        // [M, H] fp32
  float* new_stream;          // [M, H] fp32
  uint16_t* out;              // [M, H] bf16 or nullptr
  float* router;              // [M, H] fp32 or nullptr
  uint16_t* xp_img;           // pooler-input chunk image [16][Rpad][520] bf16 or nullptr (row = the row's position)
  float* xr_img;              // router-input A-fragment image [16][mt][64][32][4] fp32 or nullptr
  int Rpad, mt;               // image geometry
  const uint16_t* post_weight;  // [H] bf16 or nullptr
  const uint16_t* pre_weight;   // [H] bf16 or nullptr
  const float* alpha;         // [1] or [H] fp32 (kGate)
  const float* beta;
  float post_eps, pre_eps;
  bool channel_wise, pre_norm_bf16;
};

__device__ __forceinline__ void group_bar(int id, int threads) {
  asm volatile("bar.sync %0, %1;" ::"r"(id), "r"(threads) : "memory");
}

__device__ __forceinline__ float bf2f(uint16_t b) { return __uint_as_float(static_cast<uint32_t>(b) << 16); }
// SGLang's round_bf16: fp32 -> bf16 (round to nearest even) -> fp32
__device__ __forceinline__ float rbf(float v) { return __bfloat162float(__float2bfloat16_rn(v)); }
__device__ __forceinline__ uint16_t f2bf(float v) { return __bfloat16_as_ushort(__float2bfloat16_rn(v)); }

// Two-level sum over the kVT virtual threads (SGLang cta::reduce_sum order): per virtual warp a warp shuffle tree, then
// one warp over the per-warp partials (lanes >= kVW contribute 0). Virtual thread v of group thread g = g + v * kG.
template <int kG, int kVT>
__device__ __forceinline__ float vreduce(const float (&val)[kVT / kG], float* smem, int gtid, int bar_id) {
  using namespace device;
  constexpr int kV = kVT / kG;
  constexpr int kVW = kVT / 32;
  const int lane = gtid % 32, gw = gtid / 32;
#pragma unroll
  for (int v = 0; v < kV; ++v) {
    const float s = warp::reduce_sum(val[v]);
    if (lane == 0) smem[gw + v * (kG / 32)] = s;
  }
  group_bar(bar_id, kG);
  return warp::reduce_sum(lane < kVW ? smem[lane] : 0.0f);
}

// One owned row by a group of kG threads (gtid = 0..kG-1) playing SGLang's kVT threads. smem: 2 * (kVT / 32) floats private
// to the group (two reduction areas, as SGLang uses, so the second reduction needs no WAR barrier). img_row: the row's
// position in the image layouts (owned-row index).
// Output emit hook: owner_norm_row calls emit.pack(row, h0, ov) once per owned 8-column pack, on the thread that owns it,
// after that pack's NormArgs stores (out / images / router), in the thread's pack order; h0 = the pack's first column
// (a multiple of 8), ov = its 8 bf16 outputs (the bytes `out` receives). The default does nothing. A functor performs
// stores only: no barrier, no warp collective, no wait. A composing kernel's functor may, e.g., also store ov into a
// shared-memory staging row of the group; ordering of those stores against other threads' reads is the caller's (a group
// barrier after owner_norm_row returns; when the same group's threads read the staging row with ordinary loads before
// their next call, that call's first reduction barrier orders those reads before its emits).
struct EmitNone {
  __device__ __forceinline__ void pack(int64_t, int, const uint4&) const {}
};

template <int64_t kHidden, int kG, int kVT, bool kPostNormIsInputScale, bool kHasRouter, bool kPostWeight, bool kGate,
          class Emit = EmitNone>
__device__ __forceinline__ void owner_norm_row(const NormArgs& a, int64_t row, int img_row, int gtid, int bar_id,
                                               float* smem, const Emit& emit = Emit{}) {
  using namespace device;
  constexpr int kV = kVT / kG;
  constexpr int kPacks = kHidden / (kVT * kVec);
  constexpr int kVW = kVT / 32;
  static_assert(kVT % kG == 0 && kG % 32 == 0 && kHidden % (kVT * kVec) == 0);
  const uint16_t* branch = a.branch + row * kHidden;
  const float* stream = a.stream + row * kHidden;
  float* new_stream = a.new_stream + row * kHidden;
  const bool channel_wise = kGate && a.channel_wise;
  const bool has_pre_weight = a.pre_weight != nullptr;
  const bool pre_bf16 = a.pre_norm_bf16;

  float vals[kV][kPacks][kVec];
  float svals[kV][kPacks][kVec];
  float sum_sq[kV];
#pragma unroll
  for (int v = 0; v < kV; ++v) {
    sum_sq[v] = 0.0f;
#pragma unroll
    for (int p = 0; p < kPacks; ++p) {
      const int idx = gtid + v * kG + p * kVT;  // SGLang thread (gtid + v * kG)'s block p
      const uint4 bv = *reinterpret_cast<const uint4*>(branch + idx * kVec);
      const uint16_t bh[8] = {static_cast<uint16_t>(bv.x), static_cast<uint16_t>(bv.x >> 16),
                              static_cast<uint16_t>(bv.y), static_cast<uint16_t>(bv.y >> 16),
                              static_cast<uint16_t>(bv.z), static_cast<uint16_t>(bv.z >> 16),
                              static_cast<uint16_t>(bv.w), static_cast<uint16_t>(bv.w >> 16)};
      const float4 s0 = *reinterpret_cast<const float4*>(stream + idx * kVec);
      const float4 s1 = *reinterpret_cast<const float4*>(stream + idx * kVec + 4);
      svals[v][p][0] = s0.x; svals[v][p][1] = s0.y; svals[v][p][2] = s0.z; svals[v][p][3] = s0.w;
      svals[v][p][4] = s1.x; svals[v][p][5] = s1.y; svals[v][p][6] = s1.z; svals[v][p][7] = s1.w;
#pragma unroll
      for (int e = 0; e < kVec; ++e) {
        float x = bf2f(bh[e]);
        if constexpr (kPostNormIsInputScale && kPostWeight) {
          x *= bf2f(a.post_weight[idx * kVec + e]);
          x = rbf(x);
        }
        vals[v][p][e] = x;
        sum_sq[v] = fmaf(x, x, sum_sq[v]);
      }
    }
  }
  float tot = vreduce<kG, kVT>(sum_sq, smem, gtid, bar_id);
  const float post_rstd = math::rsqrt(tot / kHidden + a.post_eps);

  float ga = 1.0f, gb = 1.0f;
  if constexpr (kGate) {
    if (!channel_wise) {
      ga = *a.alpha;
      gb = *a.beta;
    }
  }
  float pre_sum_sq[kV];
#pragma unroll
  for (int v = 0; v < kV; ++v) {
    pre_sum_sq[v] = 0.0f;
#pragma unroll
    for (int p = 0; p < kPacks; ++p) {
      const int idx = gtid + v * kG + p * kVT;
      float ns[kVec];
#pragma unroll
      for (int e = 0; e < kVec; ++e) {
        float nb = vals[v][p][e] * post_rstd;
        if constexpr (!kPostNormIsInputScale && kPostWeight) nb *= bf2f(a.post_weight[idx * kVec + e]);
        nb = rbf(nb);
        const float s = svals[v][p][e];
        float n;
        if constexpr (kGate) {
          const float aa = channel_wise ? a.alpha[idx * kVec + e] : ga;
          const float bb = channel_wise ? a.beta[idx * kVec + e] : gb;
          n = aa * s + bb * nb;
        } else {
          n = s + nb;
        }
        ns[e] = n;
        const float pv = pre_bf16 ? rbf(n) : n;
        vals[v][p][e] = pv;
        pre_sum_sq[v] = fmaf(pv, pv, pre_sum_sq[v]);
      }
      *reinterpret_cast<float4*>(new_stream + idx * kVec) = make_float4(ns[0], ns[1], ns[2], ns[3]);
      *reinterpret_cast<float4*>(new_stream + idx * kVec + 4) = make_float4(ns[4], ns[5], ns[6], ns[7]);
    }
  }
  tot = vreduce<kG, kVT>(pre_sum_sq, smem + kVW, gtid, bar_id);
  const float pre_rstd = math::rsqrt(tot / kHidden + a.pre_eps);

#pragma unroll
  for (int v = 0; v < kV; ++v) {
#pragma unroll
    for (int p = 0; p < kPacks; ++p) {
      const int idx = gtid + v * kG + p * kVT;
      float o[kVec];
      uint16_t ob[kVec];
#pragma unroll
      for (int e = 0; e < kVec; ++e) {
        float x = vals[v][p][e] * pre_rstd;
        if (has_pre_weight) x *= bf2f(a.pre_weight[idx * kVec + e]);
        if (pre_bf16) x = rbf(x);
        o[e] = x;
        ob[e] = f2bf(x);
      }
      const int h0 = idx * kVec;
      const uint4 ov = make_uint4(static_cast<uint32_t>(ob[0]) | (static_cast<uint32_t>(ob[1]) << 16),
                                  static_cast<uint32_t>(ob[2]) | (static_cast<uint32_t>(ob[3]) << 16),
                                  static_cast<uint32_t>(ob[4]) | (static_cast<uint32_t>(ob[5]) << 16),
                                  static_cast<uint32_t>(ob[6]) | (static_cast<uint32_t>(ob[7]) << 16));
      if (a.out != nullptr) *reinterpret_cast<uint4*>(a.out + row * kHidden + h0) = ov;
      emit.pack(row, h0, ov);
      if (a.xp_img != nullptr) {
        const int c = h0 / 512, col = h0 % 512;
        *reinterpret_cast<uint4*>(a.xp_img + (static_cast<int64_t>(c) * a.Rpad + img_row) * 520 + col) = ov;
      }
      if constexpr (kHasRouter) {
        if (a.router != nullptr) {
          *reinterpret_cast<float4*>(a.router + row * kHidden + h0) = make_float4(o[0], o[1], o[2], o[3]);
          *reinterpret_cast<float4*>(a.router + row * kHidden + h0 + 4) = make_float4(o[4], o[5], o[6], o[7]);
        }
        if (a.xr_img != nullptr) {
          // A-fragment order of the TF32 router contract (m16n8k8): element (row r, column k) of k8 step ks of chunk c
          // sits in lane 4 * (r % 8) + (k % 4), register (k % 8 >= 4 ? 2 : 0) + (r % 16 >= 8 ? 1 : 0)
          const int c = h0 / 512, ks = (h0 % 512) / 8, mi = img_row / 16, rr = img_row % 16;
          float* base = a.xr_img + ((static_cast<int64_t>(c) * a.mt + mi) * 64 + ks) * 128;
#pragma unroll
          for (int e = 0; e < kVec; ++e) {
            const int lane = 4 * (rr % 8) + (e % 4);
            const int reg = (e >= 4 ? 2 : 0) + (rr >= 8 ? 1 : 0);
            base[lane * 4 + reg] = o[e];
          }
        }
      }
    }
  }
}

// Cluster-order form: SGLang's small-M path runs one row as a cluster of kCS CTAs of kNT threads; each CTA accumulates
// four moments of the row (branch, gated stream, cross, gated branch) over its packs, reduces them (warp shuffle tree, then
// one warp over the CTA's per-warp partials), and adds the other CTAs' partials (a kCS-lane shuffle tree in slot order,
// its own slot zero); every CTA then derives both RMS scales from its own four totals. The group of kG threads plays the
// kCS x kNT threads with the same element ownership, the same per-thread accumulation order and the same reductions, so
// every output bit equals SGLang's cluster kernel at (kCS, kNT). smem: 4 * (kCS * kNT / 32) + 8 * kCS floats private to
// the group. No input images; the pre-norm input is fp32 (the cluster path's only form). Same emit hook as owner_norm_row.
template <int64_t kHidden, int kG, int kCS, int kNT, bool kPostNormIsInputScale, bool kHasRouter, bool kPostWeight,
          bool kGate, class Emit = EmitNone>
__device__ __forceinline__ void owner_norm_row_cl(const NormArgs& a, int64_t row, int gtid, int bar_id, float* smem,
                                                  const Emit& emit = Emit{}) {
  using namespace device;
  constexpr int kTot = kCS * kNT;
  constexpr int kV = kTot / kG;
  constexpr int kPacks = kHidden / (kTot * kVec);
  constexpr int kNW = kNT / 32;   // warps per cluster CTA
  constexpr int kTW = kTot / 32;  // warps of the whole cluster
  constexpr int kGW = kG / 32;    // warps of the group
  constexpr float kInvHidden = 1.0f / static_cast<float>(kHidden);
  static_assert(kTot % kG == 0 && kG % 32 == 0 && kNT % 32 == 0 && kHidden % (kTot * kVec) == 0);
  static_assert(kCS >= 2 && kCS <= 16 && (kCS & (kCS - 1)) == 0 && kNW <= 32);
  float* wsum = smem;              // [4][kTW]   per-warp partials of every cluster warp
  float* part = smem + 4 * kTW;    // [kCS][4]   each cluster CTA's own total
  float* fin = part + 4 * kCS;     // [kCS][4]   each cluster CTA's total after the cross-CTA sum
  const uint16_t* branch = a.branch + row * kHidden;
  const float* stream = a.stream + row * kHidden;
  float* new_stream = a.new_stream + row * kHidden;
  const bool channel_wise = kGate && a.channel_wise;
  const bool has_pre_weight = a.pre_weight != nullptr;
  const int lane = gtid & 31, gw = gtid >> 5;
  float ga = 1.0f, gb = 1.0f;
  if constexpr (kGate) {
    if (!channel_wise) {
      ga = *a.alpha;
      gb = *a.beta;
    }
  }
  float vals[kV][kPacks][kVec];
  float res[kV][kPacks][kVec];
  float mom[kV][4];
#pragma unroll
  for (int v = 0; v < kV; ++v) {
    mom[v][0] = mom[v][1] = mom[v][2] = mom[v][3] = 0.0f;
#pragma unroll
    for (int p = 0; p < kPacks; ++p) {
      const int idx = gtid + v * kG + p * kTot;  // cluster thread (gtid + v * kG)'s pack p
      const uint4 bv = *reinterpret_cast<const uint4*>(branch + idx * kVec);
      const uint16_t bh[8] = {static_cast<uint16_t>(bv.x), static_cast<uint16_t>(bv.x >> 16),
                              static_cast<uint16_t>(bv.y), static_cast<uint16_t>(bv.y >> 16),
                              static_cast<uint16_t>(bv.z), static_cast<uint16_t>(bv.z >> 16),
                              static_cast<uint16_t>(bv.w), static_cast<uint16_t>(bv.w >> 16)};
      const float4 s0 = *reinterpret_cast<const float4*>(stream + idx * kVec);
      const float4 s1 = *reinterpret_cast<const float4*>(stream + idx * kVec + 4);
      const float sv[8] = {s0.x, s0.y, s0.z, s0.w, s1.x, s1.y, s1.z, s1.w};
#pragma unroll
      for (int e = 0; e < kVec; ++e) {
        float value = bf2f(bh[e]);
        float moment_value = value;
        if constexpr (kPostWeight) {
          const float w = bf2f(a.post_weight[idx * kVec + e]);
          if constexpr (kPostNormIsInputScale) {
            value = rbf(value * w);
            moment_value = value;
          } else {
            moment_value = value * w;
          }
        }
        const float aa = channel_wise ? a.alpha[idx * kVec + e] : ga;
        const float bb = channel_wise ? a.beta[idx * kVec + e] : gb;
        const float residual = aa * sv[e];
        const float scaled_branch = bb * moment_value;
        vals[v][p][e] = value;
        res[v][p][e] = residual;
        mom[v][0] = fmaf(value, value, mom[v][0]);
        mom[v][1] = fmaf(residual, residual, mom[v][1]);
        mom[v][2] = fmaf(residual, scaled_branch, mom[v][2]);
        mom[v][3] = fmaf(scaled_branch, scaled_branch, mom[v][3]);
      }
    }
  }
  // per cluster warp: the warp shuffle tree (real warp gw plays cluster warps gw + v * kGW; lanes coincide)
#pragma unroll
  for (int v = 0; v < kV; ++v) {
#pragma unroll
    for (int m = 0; m < 4; ++m) {
      const float r = warp::reduce_sum(mom[v][m]);
      if (lane == 0) wsum[m * kTW + gw + v * kGW] = r;
    }
  }
  group_bar(bar_id, kG);
  // per cluster CTA c: one warp over its kNW per-warp partials (lanes >= kNW contribute 0)
  for (int c = gw; c < kCS; c += kGW) {
#pragma unroll
    for (int m = 0; m < 4; ++m) {
      const float r = warp::reduce_sum(lane < kNW ? wsum[m * kTW + c * kNW + lane] : 0.0f);
      if (lane == 0) part[c * 4 + m] = r;
    }
  }
#ifndef MUSE_VR_C4N_TEST_PLANT_CL  // test-only build drops this barrier: a planted race (sanitizer non-vacuity control)
  group_bar(bar_id, kG);
#endif
  // per cluster CTA r: the other CTAs' totals by slot (own slot 0), a kCS-lane shuffle tree, then own + others
  for (int r = gw; r < kCS; r += kGW) {
    const int slot = lane & (kCS - 1);
#pragma unroll
    for (int m = 0; m < 4; ++m) {
      float t = slot != r ? part[slot * 4 + m] : 0.0f;
      t = warp::reduce_sum<kCS>(t);
      if (lane == 0) fin[r * 4 + m] = part[r * 4 + m] + t;
    }
  }
  group_bar(bar_id, kG);
#pragma unroll
  for (int v = 0; v < kV; ++v) {
    const int c = (gtid + v * kG) / kNT;  // this cluster thread's CTA
    const float branch_sum = fin[c * 4 + 0], residual_sum = fin[c * 4 + 1], cross_sum = fin[c * 4 + 2],
                scaled_branch_sum = fin[c * 4 + 3];
    const float post_rstd = math::rsqrt(branch_sum * kInvHidden + a.post_eps);
    const float pre_sum = fmaf(post_rstd, fmaf(post_rstd, scaled_branch_sum, 2.0f * cross_sum), residual_sum);
    const float pre_rstd = math::rsqrt(fmaxf(pre_sum, 0.0f) * kInvHidden + a.pre_eps);
#pragma unroll
    for (int p = 0; p < kPacks; ++p) {
      const int idx = gtid + v * kG + p * kTot;
      const int h0 = idx * kVec;
      float ns[kVec], o[kVec];
      uint16_t ob[kVec];
#pragma unroll
      for (int e = 0; e < kVec; ++e) {
        float normed = vals[v][p][e] * post_rstd;
        if constexpr (!kPostNormIsInputScale && kPostWeight) normed *= bf2f(a.post_weight[h0 + e]);
        normed = rbf(normed);
        const float bb = channel_wise ? a.beta[h0 + e] : gb;
        const float updated = fmaf(bb, normed, res[v][p][e]);
        ns[e] = updated;
        float x = updated * pre_rstd;
        if (has_pre_weight) x *= bf2f(a.pre_weight[h0 + e]);
        o[e] = x;
        ob[e] = f2bf(x);
      }
      *reinterpret_cast<float4*>(new_stream + h0) = make_float4(ns[0], ns[1], ns[2], ns[3]);
      *reinterpret_cast<float4*>(new_stream + h0 + 4) = make_float4(ns[4], ns[5], ns[6], ns[7]);
      const uint4 ov = make_uint4(static_cast<uint32_t>(ob[0]) | (static_cast<uint32_t>(ob[1]) << 16),
                                  static_cast<uint32_t>(ob[2]) | (static_cast<uint32_t>(ob[3]) << 16),
                                  static_cast<uint32_t>(ob[4]) | (static_cast<uint32_t>(ob[5]) << 16),
                                  static_cast<uint32_t>(ob[6]) | (static_cast<uint32_t>(ob[7]) << 16));
      if (a.out != nullptr) *reinterpret_cast<uint4*>(a.out + row * kHidden + h0) = ov;
      emit.pack(row, h0, ov);
      if constexpr (kHasRouter) {
        if (a.router != nullptr) {
          *reinterpret_cast<float4*>(a.router + row * kHidden + h0) = make_float4(o[0], o[1], o[2], o[3]);
          *reinterpret_cast<float4*>(a.router + row * kHidden + h0 + 4) = make_float4(o[4], o[5], o[6], o[7]);
        }
      }
    }
  }
}

#ifndef MUSE_VR_C4N_DEVICE_ONLY  // composing kernels may define it to include only the device role (no test launches)
// Test launch: one CTA of kG threads per row (rows [0, M)); image row = row - img_row0 for rows in [img_row0, img_row0 + R).
template <int64_t kHidden, int kG, int kVT, bool kIS, bool kRouter, bool kPW, bool kGate>
__global__ void __launch_bounds__(kG) c4n_test_kernel(const NormArgs a, int img_row0, int R) {
  __shared__ float smem[2 * (kVT / 32)];
  const int row = static_cast<int>(blockIdx.x);
  NormArgs b = a;
  const bool in_img = row >= img_row0 && row < img_row0 + R;
  if (!in_img) {
    b.xp_img = nullptr;
    b.xr_img = nullptr;
  }
  owner_norm_row<kHidden, kG, kVT, kIS, kRouter, kPW, kGate>(b, row, in_img ? row - img_row0 : 0,
                                                             static_cast<int>(threadIdx.x), 1, smem);
}

// Emit-hook test launch: the model's two boundary forms (attention: no post weight, router; FFN: input-scale post weight, no
// router), channel-wise or scalar gate, one CTA of 256 threads per row. The functor stores each pack into a 16 KB
// shared-memory staging row; after a group barrier the CTA copies the staging row to staged[row] (it must equal out[row]).
struct EmitStageTest {
  uint16_t* stage;  // [kHidden] bf16, shared memory
  __device__ __forceinline__ void pack(int64_t, int h0, const uint4& ov) const {
    *reinterpret_cast<uint4*>(stage + h0) = ov;
  }
};

// kCS = 1: the one-CTA form at SGLang's thread count kVT; kCS > 1: the cluster-order form (kCS CTAs of kVT threads).
template <int kCS, int kVT, bool kIS, bool kRouter, bool kPW, bool kGate>
__global__ void __launch_bounds__(256) c4n_emit_test_kernel(const NormArgs a, uint16_t* staged) {
  __shared__ float smem[kCS == 1 ? 2 * (kVT / 32) : 4 * (kCS * kVT / 32) + 8 * kCS];
  __shared__ __align__(16) uint16_t stage[8192];
  const int row = static_cast<int>(blockIdx.x);
  if constexpr (kCS == 1) {
    owner_norm_row<8192, 256, kVT, kIS, kRouter, kPW, kGate>(a, row, 0, static_cast<int>(threadIdx.x), 1, smem,
                                                              EmitStageTest{stage});
  } else {
    owner_norm_row_cl<8192, 256, kCS, kVT, kIS, kRouter, kPW, kGate>(a, row, static_cast<int>(threadIdx.x), 1, smem,
                                                                      EmitStageTest{stage});
  }
  group_bar(2, 256);  // every thread's staging stores precede the reads below
  for (int i = static_cast<int>(threadIdx.x); i < 8192 / 8; i += 256)
    *reinterpret_cast<uint4*>(staged + static_cast<int64_t>(row) * 8192 + i * 8) =
        *reinterpret_cast<const uint4*>(stage + i * 8);
}
#endif  // MUSE_VR_C4N_DEVICE_ONLY

}  // namespace muse_vr_c4n

#ifndef MUSE_VR_C4N_DEVICE_ONLY
struct WfcC4Norm {
  // branch / stream / new_stream / out / router [M, 8192]; post / pre weights [8192] bf16 (has_post / has_pre); alpha / beta
  // [1] or [8192] fp32 (gate); vt = SGLang's thread count for this step size; xp_img / xr_img written for rows
  // [img_row0, img_row0 + R).
  static void run(tvm::ffi::TensorView branch, tvm::ffi::TensorView stream, tvm::ffi::TensorView new_stream,
                  tvm::ffi::TensorView out, tvm::ffi::TensorView router, tvm::ffi::TensorView post_w, int64_t has_post,
                  tvm::ffi::TensorView pre_w, int64_t has_pre, tvm::ffi::TensorView alpha, tvm::ffi::TensorView beta,
                  int64_t gate, double post_eps, double pre_eps, int64_t pre_bf16, int64_t input_scale, int64_t vt,
                  tvm::ffi::TensorView xp_img, tvm::ffi::TensorView xr_img, int64_t img_row0, int64_t R, int64_t Rpad,
                  int64_t mt) {
    using namespace host;
    namespace n = muse_vr_c4n;
    const int64_t M = branch.size(0);
    RuntimeCheck(branch.size(1) == 8192 && (vt == 256 || vt == 512), "wfc_c4_norm: shape / thread count");
    n::NormArgs a{};
    a.branch = static_cast<const uint16_t*>(branch.data_ptr());
    a.stream = static_cast<const float*>(stream.data_ptr());
    a.new_stream = static_cast<float*>(new_stream.data_ptr());
    a.out = static_cast<uint16_t*>(out.data_ptr());
    a.router = static_cast<float*>(router.data_ptr());
    a.xp_img = R > 0 ? static_cast<uint16_t*>(xp_img.data_ptr()) : nullptr;
    a.xr_img = R > 0 ? static_cast<float*>(xr_img.data_ptr()) : nullptr;
    a.Rpad = static_cast<int>(Rpad);
    a.mt = static_cast<int>(mt);
    a.post_weight = has_post ? static_cast<const uint16_t*>(post_w.data_ptr()) : nullptr;
    a.pre_weight = has_pre ? static_cast<const uint16_t*>(pre_w.data_ptr()) : nullptr;
    a.alpha = gate ? static_cast<const float*>(alpha.data_ptr()) : nullptr;
    a.beta = gate ? static_cast<const float*>(beta.data_ptr()) : nullptr;
    a.post_eps = static_cast<float>(post_eps);
    a.pre_eps = static_cast<float>(pre_eps);
    a.channel_wise = gate && alpha.numel() == 8192;
    a.pre_norm_bf16 = pre_bf16 != 0;
    const cudaStream_t stream_ = LaunchKernel::resolve_device(branch.device());
    const int r0 = static_cast<int>(img_row0), rr = static_cast<int>(R);
#define C4N_L(VT, IS, PW, GT) \
  n::c4n_test_kernel<8192, 256, VT, IS, true, PW, GT><<<static_cast<unsigned>(M), 256, 0, stream_>>>(a, r0, rr)
#define C4N_G(VT, IS, PW) \
  if (gate) C4N_L(VT, IS, PW, true); else C4N_L(VT, IS, PW, false)
#define C4N_P(VT, IS) \
  if (has_post) { C4N_G(VT, IS, true); } else { C4N_G(VT, IS, false); }
    if (vt == 256) {
      if (input_scale) { C4N_P(256, true) } else { C4N_P(256, false) }
    } else {
      if (input_scale) { C4N_P(512, true) } else { C4N_P(512, false) }
    }
#undef C4N_P
#undef C4N_G
#undef C4N_L
  }

  // Emit-hook test: boundary = 0 (attention form: no post weight, router outputs) or 1 (FFN form: input-scale post weight,
  // no router); gate on (alpha / beta [1] or [8192]); staged [M, 8192] bf16 receives each row's staging copy.
  static void run_emit(tvm::ffi::TensorView branch, tvm::ffi::TensorView stream, tvm::ffi::TensorView new_stream,
                       tvm::ffi::TensorView out, tvm::ffi::TensorView router, tvm::ffi::TensorView post_w,
                       tvm::ffi::TensorView pre_w, int64_t has_pre, tvm::ffi::TensorView alpha, tvm::ffi::TensorView beta,
                       double post_eps, double pre_eps, int64_t pre_bf16, int64_t vt, int64_t boundary,
                       tvm::ffi::TensorView staged, int64_t cs) {
    using namespace host;
    namespace n = muse_vr_c4n;
    const int64_t M = branch.size(0);
    RuntimeCheck(branch.size(1) == 8192 && (boundary == 0 || boundary == 1) && staged.size(0) >= M &&
                     staged.size(1) == 8192,
                 "wfc_c4_norm emit: shapes / boundary");
    // the launch forms SGLang uses at hidden 8192: one CTA (256 / 512 threads) or a cluster of cs CTAs of vt threads
    RuntimeCheck((cs == 1 && (vt == 256 || vt == 512)) || (cs == 2 && vt == 256) || (cs == 4 && vt == 256) ||
                     (cs == 8 && vt == 128) || (cs == 16 && vt == 64),
                 "wfc_c4_norm emit: launch form");
    RuntimeCheck(cs == 1 || pre_bf16 == 0, "wfc_c4_norm emit: the cluster form takes an fp32 pre-norm input");
    n::NormArgs a{};
    a.branch = static_cast<const uint16_t*>(branch.data_ptr());
    a.stream = static_cast<const float*>(stream.data_ptr());
    a.new_stream = static_cast<float*>(new_stream.data_ptr());
    a.out = static_cast<uint16_t*>(out.data_ptr());
    a.router = boundary == 0 ? static_cast<float*>(router.data_ptr()) : nullptr;
    a.post_weight = boundary == 1 ? static_cast<const uint16_t*>(post_w.data_ptr()) : nullptr;
    a.pre_weight = has_pre ? static_cast<const uint16_t*>(pre_w.data_ptr()) : nullptr;
    a.alpha = static_cast<const float*>(alpha.data_ptr());
    a.beta = static_cast<const float*>(beta.data_ptr());
    a.post_eps = static_cast<float>(post_eps);
    a.pre_eps = static_cast<float>(pre_eps);
    a.channel_wise = alpha.numel() == 8192;
    a.pre_norm_bf16 = pre_bf16 != 0;
    const cudaStream_t stream_ = LaunchKernel::resolve_device(branch.device());
    auto* st = static_cast<uint16_t*>(staged.data_ptr());
    const unsigned g = static_cast<unsigned>(M);
#define C4N_E(CS, VT)                                                                          \
  if (cs == CS && vt == VT) {                                                                  \
    if (boundary == 0)                                                                         \
      n::c4n_emit_test_kernel<CS, VT, false, true, false, true><<<g, 256, 0, stream_>>>(a, st); \
    else                                                                                       \
      n::c4n_emit_test_kernel<CS, VT, true, false, true, true><<<g, 256, 0, stream_>>>(a, st);  \
  }
    C4N_E(1, 256) C4N_E(1, 512) C4N_E(2, 256) C4N_E(4, 256) C4N_E(8, 128) C4N_E(16, 64)
#undef C4N_E
  }
};

#endif  // MUSE_VR_C4N_DEVICE_ONLY

}  // namespace sglang
