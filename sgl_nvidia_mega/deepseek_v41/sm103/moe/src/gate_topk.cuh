// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
//
// Router top-k of the DeepSeek-V4.1 MoE for the low-batch routed-expert GEMMs
// (routed_expert_gemm.py, gate_topk_and_gemms): SGLang moe_fused_gate's routing (sqrtsoftplus,
// fp32 bias, top-k, renormalize, routed scale) with one CTA per row and one expert per thread,
// so a row's scoring and selection take a few warp reductions. GateTopK<kExperts, kTopK> is
// instantiated by gate_topk.cu (target, 384 experts, top-6) and gate_topk_draft.cu (DSpark
// draft, 128 experts, top-3), each its own NVRTC program and extern "C" symbol.
//
// Numerics follow the kernel that serves moe_fused_gate in the installed SGLang bit for bit;
// DSV41_GATE_REFERENCE (routed_expert_gemm.gate_topk_nvrtc_options, from gate_reference)
// names it:
//   0  the Triton _router_triton_kernel (sqrtsoftplus_log1p): libdevice exp / log1p / sqrt
//      under Triton's reflection (flush to zero, approximate sqrt; GATE_TOPK_NVRTC_OPTIONS),
//      NaNs ranked above every finite score, the routed sum in the Triton reduction's order;
//   1  the one-warp-per-token CUDA router route_warp.cuh (sgl-project/sglang#47641):
//      fast-math sqrt_softplus (a degree-7 2^f polynomial and an atanh series, MUFU rcp and
//      sqrt), NaN scores ranked last (as -1e30), the routed sum as its warp reduce_sum<8, 1>;
//   2  route_warp.cuh after sgl-project/sglang#49017: the same with reduce_sum<1, 8>.
// All three: the lowest expert id wins ties, weights are the unbiased scores divided by the
// guarded sum (or sum + eps) with div.full.f32, then the scale. For 1 and 2 every float
// operation is written as the PTX instruction SGLang's nvcc build emits (-O3, no fast math:
// no flush to zero, and only the multiply-adds it contracts), so NVRTC options cannot change
// the bits.
//
// PDL: dependents (the route kernel) launch at once and read the ids after their own wait;
// the static bias is read before this kernel's grid-dependency wait, the logits after it.

#pragma once

// Timeline hook of benchmark builds (benchmark/bench_deepseek_v41_moe_chain.py); empty here.
#ifndef DSV41_STAMP
#define DSV41_STAMP(i)
#endif

// No default: a program built without routed_expert_gemm.gate_topk_nvrtc_options would
// reproduce some router, not necessarily the installed one.
#ifndef DSV41_GATE_REFERENCE
#error "DSV41_GATE_REFERENCE: build with routed_expert_gemm.gate_topk_nvrtc_options"
#endif
static_assert(DSV41_GATE_REFERENCE >= 0 && DSV41_GATE_REFERENCE <= 2, "unknown gate reference");

namespace {
// The Triton kernel's fp32 division and the warp router's ptx::div_full: div.full.f32, not
// the IEEE div.rn.
__device__ __forceinline__ float DivFull(float a, float b) {
  float q;
  asm("div.full.f32 %0, %1, %2;" : "=f"(q) : "f"(a), "f"(b));
  return q;
}

#if DSV41_GATE_REFERENCE != 0
// The warp router's float operations, one PTX instruction each (no ftz, no contraction).
__device__ __forceinline__ float FmaRn(float a, float b, float c) {
  float d;
  asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(d) : "f"(a), "f"(b), "f"(c));
  return d;
}
__device__ __forceinline__ float MulRn(float a, float b) {
  float d;
  asm("mul.rn.f32 %0, %1, %2;" : "=f"(d) : "f"(a), "f"(b));
  return d;
}
__device__ __forceinline__ float AddRn(float a, float b) {
  float d;
  asm("add.rn.f32 %0, %1, %2;" : "=f"(d) : "f"(a), "f"(b));
  return d;
}
__device__ __forceinline__ float MaxF(float a, float b) {
  float d;
  asm("max.f32 %0, %1, %2;" : "=f"(d) : "f"(a), "f"(b));
  return d;
}
__device__ __forceinline__ float NegAbs(float a) {
  float d;
  asm("{\n\t.reg .f32 t;\n\tabs.f32 t, %1;\n\tneg.f32 %0, t;\n\t}" : "=f"(d) : "f"(a));
  return d;
}
__device__ __forceinline__ float RcpApproxFtz(float a) {
  float d;
  asm("rcp.approx.ftz.f32 %0, %1;" : "=f"(d) : "f"(a));
  return d;
}
__device__ __forceinline__ float SqrtApproxFtz(float a) {
  float d;
  asm("sqrt.approx.ftz.f32 %0, %1;" : "=f"(d) : "f"(a));
  return d;
}
__device__ __forceinline__ bool GreaterUnordered(float a, float b) {  // !(a <= b)
  unsigned int p;
  asm("{\n\t.reg .pred q;\n\tsetp.gtu.f32 q, %1, %2;\n\tselp.u32 %0, 1, 0, q;\n\t}"
      : "=r"(p)
      : "f"(a), "f"(b));
  return p != 0U;
}
#endif

// sqrt(softplus(s)): the router's unbiased score, which is also the expert's weight.
__device__ __forceinline__ float GateScore(float s) {
#if DSV41_GATE_REFERENCE == 0
  const float sp = s > 20.0f ? s : log1pf(expf(s));
  return sqrtf(sp);
#else
  // SGLang activation::sqrt_softplus<true> (route_warp.cuh's sqrtsoftplus before #48942), in
  // the order its SASS evaluates it: n_shift is contracted into one fma; z = w * w and
  // d = 2 + z, s = z * inv, the final score * select and the bias add stay separate.
  const float kHalfL2e = 0.72134751081466674805f;     // 0x3f38aa3b: 0.5f * log2e
  const float kHalfL2eLo = 9.6298151675000553951e-09f;  // 0x32257060
  const float kLn2 = 0.69314718246459960938f;          // 0x3f317218
  const float nx = MaxF(NegAbs(s), -500.0f);
  const float n_shift = FmaRn(nx, kHalfL2e, 12582912.0f);
  const float n = AddRn(n_shift, -12582912.0f);
  const float f = FmaRn(nx, kHalfL2eLo, FmaRn(nx, kHalfL2e, -n));
  const float fl = MulRn(f, kLn2);
  float e = FmaRn(fl, 1.0f / 5040.0f, 1.0f / 720.0f);
  e = FmaRn(e, fl, 1.0f / 120.0f);
  e = FmaRn(e, fl, 1.0f / 24.0f);
  e = FmaRn(e, fl, 1.0f / 6.0f);
  e = FmaRn(e, fl, 0.5f);
  e = FmaRn(e, fl, 1.0f);
  e = FmaRn(e, fl, 1.0f);
  const float w = MaxF(__int_as_float(__float_as_int(e) + (__float_as_int(n_shift) << 23)), 0.0f);
  const float z = MulRn(w, w);
  const float d = AddRn(z, 2.0f);
  const float inv0 = RcpApproxFtz(d);
  const float inv = FmaRn(inv0, FmaRn(-d, inv0, 1.0f), inv0);
  const float t = MulRn(z, inv);
  const float t2 = MulRn(t, t);
  float poly = FmaRn(t2, 1.0f / 13.0f, 1.0f / 11.0f);
  poly = FmaRn(poly, t2, 1.0f / 9.0f);
  poly = FmaRn(poly, t2, 1.0f / 7.0f);
  poly = FmaRn(poly, t2, 1.0f / 5.0f);
  poly = FmaRn(poly, t2, 1.0f / 3.0f);
  const float two_inv = AddRn(inv, inv);
  const float r = FmaRn(MulRn(two_inv, t2), poly, two_inv);
  const bool pos = GreaterUnordered(s, 0.0f);  // NaN takes this side and stays NaN
  return MulRn(SqrtApproxFtz(pos ? FmaRn(z, r, s) : r), pos ? 1.0f : w);
#endif
}

// Order-preserving u32 key of a score + bias (-0 ranks as +0); 0 marks a taken expert.
__device__ __forceinline__ unsigned int GateKey(float score, float b) {
#if DSV41_GATE_REFERENCE == 0
  const float biased = score + b;
  const float v = (biased == biased ? biased : __int_as_float(0x7f800000)) + 0.0f;  // NaN first
#else
  const float biased = AddRn(score, b);
  const float v = biased == biased ? AddRn(biased, 0.0f) : -1e30f;  // NaN last
#endif
  const unsigned int bits = __float_as_uint(v);
  return (bits & 0x80000000U) ? ~bits : (bits | 0x80000000U);
}

// The routed sum (non-negative weights; padding slots add an exact +0.0).
template <int kTopK>
__device__ __forceinline__ float RoutedSum(const float (&w)[kTopK]) {
  static_assert(kTopK == 6 || kTopK == 3, "no reduction order recorded for this top-k");
#if DSV41_GATE_REFERENCE == 0
  // The order moe_fused_gate's tl.sum adds its BLOCK_K slots. Top-6 (BLOCK_K 8): two
  // threads hold 4 of the 8 slots each. Top-3 (BLOCK_K 4): one thread, (0 + 2) + (1 + 3).
  if constexpr (kTopK == 6) {
    return ((w[0] + w[2]) + (w[1] + w[3])) + (w[4] + w[5]);
  } else {
    return (w[0] + w[2]) + w[1];
  }
#elif DSV41_GATE_REFERENCE == 1
  // Lane k holds slot k (lanes up to bit_ceil(K) hold 0); xor-butterfly 4, 2, 1 (or 2, 1).
  if constexpr (kTopK == 6) {
    return AddRn(AddRn(AddRn(w[0], w[4]), w[2]), AddRn(AddRn(w[1], w[5]), w[3]));
  } else {
    return AddRn(AddRn(w[0], w[2]), w[1]);
  }
#else
  // xor-butterfly 1, 2, 4 (or 1, 2).
  if constexpr (kTopK == 6) {
    return AddRn(AddRn(AddRn(w[0], w[1]), AddRn(w[2], w[3])), AddRn(w[4], w[5]));
  } else {
    return AddRn(AddRn(w[0], w[1]), w[2]);
  }
#endif
}

// A selected weight, normalized by the routed sum (eps > 0: sum + eps, as
// renormalize_epsilon; else the guarded sum), then scaled.
__device__ __forceinline__ float GateWeight(float w, float sum, float eps, float scale) {
#if DSV41_GATE_REFERENCE == 0
  const float norm = eps > 0.0f ? sum + eps : (sum > 0.0f ? sum : 1.0f);
  return DivFull(w, norm) * scale;
#else
  const float norm = eps > 0.0f ? AddRn(sum, eps) : (sum > 0.0f ? sum : 1.0f);
  return MulRn(DivFull(w, norm), scale);
#endif
}

// One row's routing, block = kExperts threads (one expert each): the row's score s and its
// bias b (thread e's), the row's validity. Every thread of the block calls it (it
// synchronizes the block once); afterwards warp 0's lanes < kTopK hold slot `lane`'s weight
// and id (id -1, weight 0 for an invalid row) and every lane of warp 0 holds all kTopK ids
// in row_ids; the other warps' results are unspecified. eps > 0 normalizes by sum + eps,
// as renormalize_epsilon; else by the guarded sum.
template <int kExperts, int kTopK>
struct GateScratch {
  unsigned int cand_key[(kExperts / 32) * kTopK];
  int cand_id[(kExperts / 32) * kTopK];
  float cand_act[(kExperts / 32) * kTopK];
};

template <int kExperts, int kTopK>
__device__ __forceinline__ void SelectRow(
    GateScratch<kExperts, kTopK>& scratch, float s, float b, bool valid_row, float scale, float eps,
    float& out_w, int& out_id, int (&row_ids)[kTopK]) {
  constexpr int kThreads = kExperts;
  constexpr int kWarps = kThreads / 32;
  constexpr int kCandidates = kWarps * kTopK;
  constexpr int kSlots = (kCandidates + 31) / 32;  // candidates per merging lane
  static_assert(kExperts % 32 == 0 && kTopK <= 32, "one expert per thread, whole warps");
  const int e = static_cast<int>(threadIdx.x);
  const int lane = e & 31;
  const int warp = e >> 5;
  const float act = GateScore(s);
  unsigned int key = GateKey(act, b);

  // This warp's top-k (its ids are contiguous, so the lowest lane wins a tie).
#pragma unroll
  for (int k = 0; k < kTopK; ++k) {
    const unsigned int top = __reduce_max_sync(0xffffffffU, key);
    const unsigned int win = __reduce_min_sync(0xffffffffU, key == top ? static_cast<unsigned int>(e) : 0xffffffffU);
    if (static_cast<unsigned int>(e) == win) {
      scratch.cand_key[warp * kTopK + k] = key;
      scratch.cand_id[warp * kTopK + k] = e;
      scratch.cand_act[warp * kTopK + k] = act;
      key = 0U;
    }
  }
  __syncthreads();
  if (warp != 0) return;

  // Merge the kWarps * kTopK candidates: every global top-k expert is in its warp's top-k.
  unsigned int ck[kSlots];
  int ci[kSlots];
#pragma unroll
  for (int j = 0; j < kSlots; ++j) {
    const int c = lane + 32 * j;
    ck[j] = c < kCandidates ? scratch.cand_key[c] : 0U;
    ci[j] = c < kCandidates ? scratch.cand_id[c] : kExperts;
  }
  float sel_w[kTopK];
  int sel_i[kTopK];
#pragma unroll
  for (int k = 0; k < kTopK; ++k) {
    unsigned int best = ck[0];
#pragma unroll
    for (int j = 1; j < kSlots; ++j) best = max(best, ck[j]);
    const unsigned int top = __reduce_max_sync(0xffffffffU, best);
    unsigned int mine = 0xffffffffU;
#pragma unroll
    for (int j = 0; j < kSlots; ++j) {
      if (ck[j] == top) mine = min(mine, static_cast<unsigned int>(ci[j]));
    }
    const int id = static_cast<int>(__reduce_min_sync(0xffffffffU, mine));
    int slot = -1;
#pragma unroll
    for (int j = 0; j < kSlots; ++j) {
      if (ck[j] == top && ci[j] == id) { slot = lane + 32 * j; ck[j] = 0U; }
    }
    const unsigned int owner = __ballot_sync(0xffffffffU, slot >= 0);
    const int src = __shfl_sync(0xffffffffU, slot, __ffs(owner) - 1);
    sel_w[k] = scratch.cand_act[src];
    sel_i[k] = id;
  }
  const float sum = RoutedSum<kTopK>(sel_w);
#pragma unroll
  for (int k = 0; k < kTopK; ++k) row_ids[k] = valid_row ? sel_i[k] : -1;
  float w = 0.0f;
  int id = 0;
  if (lane < kTopK) {
#pragma unroll
    for (int k = 0; k < kTopK; ++k) {
      if (lane == k) { w = sel_w[k]; id = sel_i[k]; }
    }
    w = GateWeight(w, sum, eps, scale);
    if (!valid_row) { w = 0.0f; id = -1; }
  }
  out_w = w;
  out_id = id;
}

// grid = rows, block = kExperts. num_valid: rows >= *num_valid get weight 0 and id -1 (padded
// graph rows), as moe_fused_gate's num_token_non_padded; null means every row is valid. With
// input_ids (int32, or int64 when ids64), a row whose token is alt_token_id ranks with
// bias_alt instead of bias (DSV4.1 VL routing). eps > 0 normalizes by sum + eps, as
// renormalize_epsilon; else by the guarded sum.
template <int kExperts, int kTopK>
__device__ __forceinline__ void GateTopK(
    const float* logits, const float* bias, const float* bias_alt, const void* input_ids, int ids64,
    long long alt_token_id, const int* num_valid, float* __restrict__ weights, int* __restrict__ ids,
    float scale, float eps) {
  asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
  DSV41_STAMP(0);
  __shared__ GateScratch<kExperts, kTopK> scratch;
  const int e = static_cast<int>(threadIdx.x);
  const int lane = e & 31;
  const int warp = e >> 5;
  const int row = static_cast<int>(blockIdx.x);
  const float b_main = bias[e];
  const float b_alt = bias_alt != nullptr ? bias_alt[e] : b_main;
  asm volatile("griddepcontrol.wait;" ::: "memory");
  DSV41_STAMP(1);

  const int valid = num_valid != nullptr ? *num_valid : static_cast<int>(gridDim.x);
  bool alt = false;
  if (input_ids != nullptr && row < valid) {
    const long long token = ids64 ? static_cast<const long long*>(input_ids)[row]
                                  : static_cast<long long>(static_cast<const int*>(input_ids)[row]);
    alt = token == alt_token_id;
  }
  const float b = alt ? b_alt : b_main;
  const float s = logits[row * kExperts + e];
  float w;
  int id;
  int row_ids[kTopK];
  SelectRow<kExperts, kTopK>(scratch, s, b, row < valid, scale, eps, w, id, row_ids);
  DSV41_STAMP(2);
  if (warp == 0 && lane < kTopK) {
    weights[row * kTopK + lane] = w;
    ids[row * kTopK + lane] = id;
  }
  DSV41_STAMP(3);
}
}  // namespace
