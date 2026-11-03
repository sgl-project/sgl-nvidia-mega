// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// Portions Copyright 2023-2024 SGLang Team (Apache-2.0)
// Derived from SGLang (https://github.com/sgl-project/sglang), Apache License 2.0; see ../include/LICENSE.sglang.
// Source: SGLang python/sglang/kernels/jit/csrc/muse_spark/norm_residual.cuh (the one-CTA-per-row path) at revision 306ccf6,
// through the owner-row norm role owner_norm_row (wfc_c4_norm.cuh).
// Modified by NVIDIA: the norm body runs on a row whose branch input is the reduce-scatter's owner fold computed in the norm's
// load phase (no branch store / re-read), in a register-lean form for a composing persistent kernel; default off.
//
// RS_A fold + owner-row norm role (device API, header only). Needs c3_hops_ops.cuh (RowFold, fold_own_local) and the owner-row
// norm helpers of wfc_c4_norm.cuh (NormArgs, vreduce, bf16 conversions) on the include path.
//
// Call contract for a persistent kernel of 512 threads per CTA: two groups of kG = 256 threads (group = threadIdx.x / 256),
// each on its own owned row (rows rr = 2 * blockIdx.x + group, stride 2 * gridDim.x), each with its own named barrier id
// (count 256) used by no other phase with another count while the group is inside, barrier 0 untouched; per group 2 * (kVT /
// 32) floats of reduction smem and (kSmemFold) one branch row of 8192 bf16, the two groups' rows disjoint. RowFold<W> holds
// references to the kernel's grid-constant site / control parameters (see c3_hops_ops.cuh).

#pragma once

#include "c3_hops_ops.cuh"
#include "wfc_c4_norm.cuh"

#include <cuda_bf16.h>
#include <cstdint>

#ifndef C3_STAMP
#define C3_STAMP(j)
#endif

namespace sglang {
namespace muse_vr_c3_fn {

namespace c4n = muse_vr_c4n;
namespace lh = muse_vr::lb_hop;

// bf16 element e (0..7) of a packed 8-element vector (element 0 in the low half of .x)
__device__ __forceinline__ uint16_t bf16_of(const uint4& q, int e) {
  const uint32_t w = e < 2 ? q.x : (e < 4 ? q.y : (e < 6 ? q.z : q.w));
  return uint16_t((w >> (16 * (e & 1))) & 0xffffu);
}

// AND of a predicate over the kG threads of a group, as a barrier on the group's named barrier (bar.red.and, id bar_id, count
// kG): it also orders the group's memory accesses like bar.sync.
template <int kG>
__device__ __forceinline__ bool group_and(bool v, int bar_id) {
  uint32_t r;
  asm volatile(
      "{\n .reg .pred p, q;\n setp.ne.u32 p, %1, 0;\n bar.red.and.pred q, %2, %3, p;\n selp.u32 %0, 1, 0, q;\n}"
      : "=r"(r)
      : "r"(v ? 1u : 0u), "r"(bar_id), "r"(kG)
      : "memory");
  return r != 0u;
}

// One owned row (global index `row` of the [M, 8192] norm arrays) by a group of kG threads (gtid 0..kG-1) playing the norm's
// kVT threads (the element ownership and reduction order of owner_norm_row): the RS_A owner fold of the row, its one bf16
// rounding (the branch), then post-norm -> gated residual -> pre-norm -> out (bf16) and router input (fp32), the new stream
// (fp32). Post weight on, router on, input-scale off, scalar gate when kGate. The gated residual is computed as
// fma(alpha, stream, beta * normed_branch): the rounding of the reference kernel as compiled (checked byte-equal by unit).
//   kSmemFold = true (register-lean form, <= 128 registers / thread): the fold runs in 6-column groups spread over the
//     group's threads (fold_own_local: the kernel's own fold arithmetic and polls) into the group-private shared-memory
//     branch row `sbr` (8192 bf16, 16-byte aligned), then each thread reads its packs from it; weights load after the fold.
//   kSmemFold = false: RowFold::fold8 on the thread's own 8-column packs in registers (about twice the registers).
// Barriers: the group's named barrier (bar_id, kG) only: one and-barrier after the fold (and, with kSmemFold, the edge
// between the fold's shared-memory stores and the reads of other threads) + one per reduction. smem: 2 * (kVT / 32) floats
// private to the group. Outputs are plain stores; a consumer outside the group needs its own ordering (kernel boundary or a
// release / acquire ready word placed by the caller after the group's last barrier).
// A failed guard or an expired wait (sticky error word already set by the polls) makes the row's outputs NaN.
template <int W, int kG, int kVT, bool kGate, bool kSmemFold = false>
__device__ __forceinline__ void fold_norm_row(const c4n::NormArgs& a, const muse_vr_c3_ops::RowFold<W>& rf, int64_t row,
                                              __nv_bfloat16* branch_out, float* smem, int gtid, int bar_id,
                                              uint16_t* sbr = nullptr) {
  using namespace device;
  constexpr int64_t kHidden = 8192;
  constexpr int kVec = c4n::kVec;
  constexpr int kV = kVT / kG;
  constexpr int kPacks = int(kHidden) / (kVT * kVec);
  constexpr int kVW = kVT / 32;
  constexpr int P = kV * kPacks;
  static_assert(kVT % kG == 0 && kHidden % (kVT * kVec) == 0, "c4 mapping");
  const float* stream = a.stream + row * kHidden;
  float* new_stream = a.new_stream + row * kHidden;
  const bool has_pre_weight = a.pre_weight != nullptr;
  const bool pre_bf16 = a.pre_norm_bf16;

  // 1) the row's stream values (independent of the hop): issued before the fold's polls
  float svals[kV][kPacks][kVec];
  uint32_t c0[P];
#pragma unroll
  for (int v = 0; v < kV; ++v)
#pragma unroll
    for (int p = 0; p < kPacks; ++p) {
      const int idx = gtid + v * kG + p * kVT;
      c0[v * kPacks + p] = uint32_t(idx * kVec);
      const float4 s0 = *reinterpret_cast<const float4*>(stream + idx * kVec);
      const float4 s1 = *reinterpret_cast<const float4*>(stream + idx * kVec + 4);
      svals[v][p][0] = s0.x; svals[v][p][1] = s0.y; svals[v][p][2] = s0.z; svals[v][p][3] = s0.w;
      svals[v][p][4] = s1.x; svals[v][p][5] = s1.y; svals[v][p][6] = s1.z; svals[v][p][7] = s1.w;
    }
  // the post / pre weights of those packs (constants): packed loads issued here, so their latency hides under the fold's
  // polls (one 16-byte load per pack and weight instead of a scalar load per element after each reduction)
  uint4 pw4[kV][kPacks], qw4[kV][kPacks];
  float ga = 1.0f, gb = 1.0f;  // gate scalars: loaded here too (after the stores below they could not be hoisted)
  if constexpr (kGate) {
    ga = *a.alpha;
    gb = *a.beta;
  }
  auto load_weights = [&]() {
#pragma unroll
    for (int v = 0; v < kV; ++v)
#pragma unroll
      for (int p = 0; p < kPacks; ++p) {
        const int idx = gtid + v * kG + p * kVT;
        pw4[v][p] = *reinterpret_cast<const uint4*>(a.post_weight + idx * kVec);
        qw4[v][p] = has_pre_weight ? *reinterpret_cast<const uint4*>(a.pre_weight + idx * kVec) : make_uint4(0u, 0u, 0u, 0u);
      }
  };
  if constexpr (!kSmemFold) load_weights();
  C3_STAMP(4);
  // 2) the fold of exactly the packs this thread's norm threads read
  float sum[P][kVec];
  bool ok_t;
  if constexpr (kSmemFold) {
    // 6-column groups g = gtid + i * kG over the row, NB per round (fold_own_local's vector budget, register-lean)
    constexpr int kNG = (int(kHidden) + 5) / 6;
    constexpr int NI = (kNG + kG - 1) / kG;
    constexpr int NB = (W <= 2) ? 3 : ((W <= 4) ? 2 : 1);
    ok_t = rf.go;
    if (rf.go) {
#pragma unroll 1
      for (int i0 = 0; i0 < NI; i0 += NB) {
        uint32_t rows[NB], cols[NB], m[NB];
        int n = 0;
#pragma unroll
        for (int i = 0; i < NB; ++i) {
          const int g = gtid + (i0 + i) * kG;
          const bool valid = (i0 + i) < NI && g < kNG;
          rows[i] = uint32_t(row);
          cols[i] = valid ? uint32_t(g) * 6u : 0u;
          m[i] = 0u;
          if (valid) n = i + 1;
        }
        float s6[NB][6];
        ok_t = muse_vr_c3_ops::fold_own_local<W, NB>(rf.s, rf.c, rf.x, rows, cols, m, n, false, rf.part, rf.H, rf.sz, s6) &&
               ok_t;
#pragma unroll
        for (int i = 0; i < NB; ++i) {
          if (i >= n) break;
          uint32_t* o32 = reinterpret_cast<uint32_t*>(sbr);
#pragma unroll
          for (int e = 0; e < 6; e += 2) {
            const uint32_t col = cols[i] + uint32_t(e);
            if (col < uint32_t(kHidden))
              o32[col >> 1] = uint32_t(c4n::f2bf(s6[i][e])) | (uint32_t(c4n::f2bf(s6[i][e + 1])) << 16);
          }
        }
      }
    }
    load_weights();  // after the fold: issued now, needed after the first reduction
  } else if constexpr (W <= 2) {
    ok_t = rf.template fold8<P>(uint32_t(row), c0, sum);
  } else {
    // W > 2: two half rounds of polls (12 instead of 24 polled vectors per thread at W 4: inside the register budget)
    constexpr int PH = P / 2;
    uint32_t ca[PH], cb[PH];
    float sa[PH][kVec], sb[PH][kVec];
#pragma unroll
    for (int i = 0; i < PH; ++i) {
      ca[i] = c0[i];
      cb[i] = c0[PH + i];
    }
    ok_t = rf.template fold8<PH>(uint32_t(row), ca, sa);
    ok_t = rf.template fold8<PH>(uint32_t(row), cb, sb) && ok_t;
#pragma unroll
    for (int i = 0; i < PH; ++i)
#pragma unroll
      for (int e = 0; e < kVec; ++e) {
        sum[i][e] = sa[i][e];
        sum[PH + i][e] = sb[i][e];
      }
  }
  C3_STAMP(5);
#ifdef C3_FOLD_NORM_TEST_DROP_GROUP_BARRIER
  const bool ok = ok_t;  // test builds only (a racecheck control): no group barrier between the fold's stores and the reads
#else
  // smem edge (kSmemFold): thread t stores branch columns [6 g, 6 g + 6) for its groups g = t + i * kG; norm thread t then
  // reads the 8-column packs idx * 8 (idx = t + v * kG + p * kVT), written by OTHER threads -> this and-barrier (bar.red on
  // bar_id x kG) orders every fold store before any read. The next row's fold stores come after the first reduction's
  // bar.sync (same id / count), which every read of this row precedes (WAR ordered).
  const bool ok = group_and<kG>(ok_t, bar_id);
#endif
  C3_STAMP(6);
  // 3) branch = the fold's one bf16 rounding (what the kernel's own fold would store), then C4's owner_norm_row body
  float vals[kV][kPacks][kVec];
  float sum_sq[kV];
#pragma unroll
  for (int v = 0; v < kV; ++v) {
    sum_sq[v] = 0.0f;
#pragma unroll
    for (int p = 0; p < kPacks; ++p) {
      uint16_t bh[kVec];
#pragma unroll
      if constexpr (kSmemFold) {
        const int idx = gtid + v * kG + p * kVT;
        const uint4 q = *reinterpret_cast<const uint4*>(sbr + idx * kVec);  // after the group's and-barrier
#pragma unroll
        for (int e = 0; e < kVec; ++e) bh[e] = ok ? bf16_of(q, e) : lh::kPoisonBf16;
      } else {
#pragma unroll
        for (int e = 0; e < kVec; ++e) bh[e] = ok ? c4n::f2bf(sum[v * kPacks + p][e]) : lh::kPoisonBf16;
      }
      if (branch_out != nullptr) {
        const int idx = gtid + v * kG + p * kVT;
        *reinterpret_cast<uint4*>(branch_out + idx * kVec) =
            make_uint4(uint32_t(bh[0]) | (uint32_t(bh[1]) << 16), uint32_t(bh[2]) | (uint32_t(bh[3]) << 16),
                       uint32_t(bh[4]) | (uint32_t(bh[5]) << 16), uint32_t(bh[6]) | (uint32_t(bh[7]) << 16));
      }
#pragma unroll
      for (int e = 0; e < kVec; ++e) {
        const float x = ok ? c4n::bf2f(bh[e]) : __int_as_float(0x7fc00000);
        vals[v][p][e] = x;
        sum_sq[v] = fmaf(x, x, sum_sq[v]);
      }
    }
  }
  float tot = c4n::vreduce<kG, kVT>(sum_sq, smem, gtid, bar_id);
  const float post_rstd = math::rsqrt(tot / kHidden + a.post_eps);
  C3_STAMP(7);
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
        nb *= c4n::bf2f(bf16_of(pw4[v][p], e));
        nb = c4n::rbf(nb);
        const float s = svals[v][p][e];
        float n;
        if constexpr (kGate) {
          n = __fmaf_rn(ga, s, __fmul_rn(gb, nb));
        } else {
          n = s + nb;
        }
        ns[e] = n;
        const float pv = pre_bf16 ? c4n::rbf(n) : n;
        vals[v][p][e] = pv;
        pre_sum_sq[v] = fmaf(pv, pv, pre_sum_sq[v]);
      }
      *reinterpret_cast<float4*>(new_stream + idx * kVec) = make_float4(ns[0], ns[1], ns[2], ns[3]);
      *reinterpret_cast<float4*>(new_stream + idx * kVec + 4) = make_float4(ns[4], ns[5], ns[6], ns[7]);
    }
  }
  tot = c4n::vreduce<kG, kVT>(pre_sum_sq, smem + kVW, gtid, bar_id);
  const float pre_rstd = math::rsqrt(tot / kHidden + a.pre_eps);
  C3_STAMP(8);
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
        if (has_pre_weight) x *= c4n::bf2f(bf16_of(qw4[v][p], e));
        if (pre_bf16) x = c4n::rbf(x);
        o[e] = x;
        ob[e] = c4n::f2bf(x);
      }
      const int64_t h0 = int64_t(idx) * kVec;
      if (a.out != nullptr)
        *reinterpret_cast<uint4*>(a.out + row * kHidden + h0) =
            make_uint4(uint32_t(ob[0]) | (uint32_t(ob[1]) << 16), uint32_t(ob[2]) | (uint32_t(ob[3]) << 16),
                       uint32_t(ob[4]) | (uint32_t(ob[5]) << 16), uint32_t(ob[6]) | (uint32_t(ob[7]) << 16));
      if (a.router != nullptr) {
        *reinterpret_cast<float4*>(a.router + row * kHidden + h0) = make_float4(o[0], o[1], o[2], o[3]);
        *reinterpret_cast<float4*>(a.router + row * kHidden + h0 + 4) = make_float4(o[4], o[5], o[6], o[7]);
      }
    }
  }
  C3_STAMP(9);
}

}  // namespace muse_vr_c3_fn
}  // namespace sglang
