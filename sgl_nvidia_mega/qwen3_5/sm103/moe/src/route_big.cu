// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// Adapted from TensorRT-LLM (Apache-2.0, NVIDIA): tensorrt_llm/_torch/cute_dsl_kernels/blackwell/q35_moe_front_big.py
// (Q35MoeRouteBigKernel, the route-only big front of TRTLLM_Q35_MOE_FRONT_BIG: launch shape, token split, per-expert
// token masks, the cluster's counts through distributed shared memory, the table writes).
//
// TEP8's front at wide verify steps: behind SGLang's router GEMM, the top-10
// of every token's 512 bf16 logits, the trtllm-gen MoE body's routing tables for the rank's window of local experts
// and the NVFP4 quantization of the MoE input, in one launch of thread-block clusters of CL CTAs:
//
// * cluster 0 routes: CTA c takes tokens [c TPC, (c + 1) TPC), one warp a token (two, interleaved, past 32), with
//   route_tables.cuh's routing (FlashInfer's trtllm-gen bits, as slot route_topk); a local pick sets the token's bit
//   in the CTA's per-expert token mask (shared memory, MW words an expert). After a cluster barrier every routing CTA
//   reads the cluster's per-expert counts through distributed shared memory, scans the CTAs per expert (tile) and
//   writes the expanded -> permuted / permuted -> token entries of its own picks (an expert's rows in token order),
//   the CTA tables and padding rows (-1) of every CL-th expert; CTA 0 also the totals and the tokens per expert.
// * the other clusters quantize x (route_topk.cu's quantizer, bit for bit).
//
// The tables are route_tables.cuh's write_tables' (the same counts, CTA tables and totals; an expert's rows in token
// order where write_tables<false> takes the order of shared-memory atomics, which the body's output does not depend
// on). No CTA waits for another cluster. PDL: every read after griddepcontrol.wait (the global scale, static, before);
// dependents released after a CTA barrier that follows the CTA's last global store.
//
// Defines: CL (CTAs a cluster: 8, or 16, non-portable), MW (token-mask words an expert: a routing CTA takes up to
// 32 MW <= 64 tokens, Params.tpc), NL (local experts, a multiple of 32, <= 1024). Built with NVRTC.

#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
#include <cuda_bf16.h>

// NVRTC has no <cstdint>
typedef unsigned char uint8_t;
typedef unsigned short uint16_t;
typedef int int32_t;
typedef unsigned int uint32_t;

#include "route_tables.cuh"

namespace q35_route_big {

using namespace q35_route_tables;

constexpr uint32_t kThreadsBig = 1024, kWarpsBig = kThreadsBig / 32;
constexpr uint32_t kHidden = 4096, kSfVec = 16, kBlocksPerRow = kHidden / kSfVec;  // 256
constexpr uint32_t kMaxTpc = 32 * MW;  // tokens a routing CTA takes at most
static_assert(MW >= 1 && kMaxTpc <= 2 * kWarpsBig, "at most two tokens a warp");
static_assert(NL % 32 == 0 && NL <= kThreadsBig, "one thread an expert");
static_assert(NL * MW <= kThreadsBig, "one thread a token-mask word");

struct Params {
  const __nv_bfloat16* logits;  // [M, 512]
  const __nv_bfloat16* x;       // [M, 4096]
  const float* global_scale;    // [1]
  int32_t* topk_ids;            // [M, 10]
  uint16_t* topk_weights;       // [M, 10] bf16
  uint8_t* x_fp4;               // [M, 2048]
  uint8_t* x_sf;                // [M, 256]
  Tables tables;                // the counter is unused
  int32_t num_tokens;           // M
  int32_t tpc;                  // tokens a routing CTA: ceil(M / CL) <= 32 MW
};

// ---- the NVFP4 quantizer (route_topk.cu, router_front.cu)
__device__ __forceinline__ float bf16_lo(uint32_t v) { return __uint_as_float(v << 16); }
__device__ __forceinline__ float bf16_hi(uint32_t v) { return __uint_as_float(v & 0xFFFF0000u); }
__device__ __forceinline__ float rcp_approx_ftz(float a) {
  float y;
  asm("rcp.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(a));
  return y;
}
__device__ __forceinline__ uint32_t cvt_e4m3(float a) {
  uint32_t out;
  asm("{\n.reg .b16 pair;\n.reg .f32 zero;\nmov.f32 zero, 0f00000000;\n"
      "cvt.rn.satfinite.e4m3x2.f32 pair, zero, %1;\ncvt.u32.u16 %0, pair;\n}"
      : "=r"(out)
      : "f"(a));
  return out & 0xFFu;
}
__device__ __forceinline__ float e4m3_to_f32(uint32_t v) {
  float out;
  asm("{\n.reg .b16 pair;\n.reg .b32 h2;\n.reg .b16 lo, hi;\ncvt.u16.u32 pair, %1;\n"
      "cvt.rn.f16x2.e4m3x2 h2, pair;\nmov.b32 {lo, hi}, h2;\ncvt.f32.f16 %0, lo;\n}"
      : "=f"(out)
      : "r"(v));
  return out;
}
__device__ __forceinline__ uint32_t cvt_e2m1x8(const float (&v)[8]) {
  uint32_t out;
  asm("{\n.reg .b8 b0, b1, b2, b3;\n"
      "cvt.rn.satfinite.e2m1x2.f32 b0, %2, %1;\ncvt.rn.satfinite.e2m1x2.f32 b1, %4, %3;\n"
      "cvt.rn.satfinite.e2m1x2.f32 b2, %6, %5;\ncvt.rn.satfinite.e2m1x2.f32 b3, %8, %7;\n"
      "mov.b32 %0, {b0, b1, b2, b3};\n}"
      : "=r"(out)
      : "f"(v[0]), "f"(v[1]), "f"(v[2]), "f"(v[3]), "f"(v[4]), "f"(v[5]), "f"(v[6]), "f"(v[7]));
  return out;
}
__device__ __forceinline__ uint32_t bf16x2_abs_max(uint32_t a, uint32_t b) {
  uint32_t out;
  asm("{\n.reg .b32 ta, tb;\nand.b32 ta, %1, 0x7FFF7FFF;\nand.b32 tb, %2, 0x7FFF7FFF;\nmax.bf16x2 %0, ta, tb;\n}"
      : "=r"(out)
      : "r"(a), "r"(b));
  return out;
}
__device__ __forceinline__ uint32_t bf16x2_max(uint32_t a, uint32_t b) {
  uint32_t out;
  asm("max.bf16x2 %0, %1, %2;" : "=r"(out) : "r"(a), "r"(b));
  return out;
}
__device__ __forceinline__ void quantize_block(const uint4& a, const uint4& b, float gs, uint32_t& sf, uint2& packed) {
  const uint32_t w[8] = {a.x, a.y, a.z, a.w, b.x, b.y, b.z, b.w};
  const uint32_t m = bf16x2_max(bf16x2_max(bf16x2_abs_max(w[0], w[1]), bf16x2_abs_max(w[2], w[3])),
                                bf16x2_max(bf16x2_abs_max(w[4], w[5]), bf16x2_abs_max(w[6], w[7])));
  const float amax = fmaxf(bf16_lo(m), bf16_hi(m));
  const float scale = __fmul_rn(gs, __fmul_rn(amax, rcp_approx_ftz(6.0f)));
  sf = cvt_e4m3(scale);
  const float sf_f = e4m3_to_f32(sf);
  const float out_scale = (sf_f == 0.0f) ? 0.0f : rcp_approx_ftz(__fmul_rn(sf_f, rcp_approx_ftz(gs)));
  float lo[8], hi[8];
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    lo[2 * i] = __fmul_rn(bf16_lo(w[i]), out_scale);
    lo[2 * i + 1] = __fmul_rn(bf16_hi(w[i]), out_scale);
    hi[2 * i] = __fmul_rn(bf16_lo(w[4 + i]), out_scale);
    hi[2 * i + 1] = __fmul_rn(bf16_hi(w[4 + i]), out_scale);
  }
  packed.x = cvt_e2m1x8(lo);
  packed.y = cvt_e2m1x8(hi);
}

// ---- cluster primitives
__device__ __forceinline__ void cluster_arrive_release() {
  asm volatile("barrier.cluster.arrive.release.aligned;" ::: "memory");
}
__device__ __forceinline__ void cluster_arrive_relaxed() {
  asm volatile("barrier.cluster.arrive.relaxed.aligned;" ::: "memory");
}
__device__ __forceinline__ void cluster_wait_acquire() {
  asm volatile("barrier.cluster.wait.acquire.aligned;" ::: "memory");
}
// the word at this CTA's shared address `p` in cluster CTA `rank`
__device__ __forceinline__ int32_t ld_dsmem(const int32_t* p, uint32_t rank) {
  const uint32_t a = static_cast<uint32_t>(__cvta_generic_to_shared(p));
  int32_t v;
  // no memory clobber: the cluster barrier before (which has one) orders these reads; without it the peers'
  // loads issue back to back instead of one round trip each
  asm volatile("{\n.reg .b32 ra;\nmapa.shared::cluster.u32 ra, %1, %2;\nld.shared::cluster.b32 %0, [ra];\n}"
               : "=r"(v)
               : "r"(a), "r"(rank));
  return v;
}

// one token's logits (lane l: experts 16 l .. 16 l + 15)
__device__ __forceinline__ void load_logits(const Params& p, int32_t token, uint32_t lane, uint4& v0, uint4& v1) {
  const uint4* src = reinterpret_cast<const uint4*>(p.logits + static_cast<size_t>(token) * kExperts + lane * 16);
  v0 = __ldcg(src);
  v1 = __ldcg(src + 1);
}

__device__ __forceinline__ void route_cta(const Params& p, uint32_t c) {
  __shared__ uint32_t s_mask[NL * MW];
  __shared__ int32_t s_cnt[NL], s_pre[NL], s_offs[NL], s_tot[NL], s_wsum[kWarpsBig];
  __shared__ uint16_t s_pick[kMaxTpc * kTopK];
  __shared__ __align__(16) uint32_t s_cand[kWarpsBig][32];
  const uint32_t tid = threadIdx.x, warp = tid / 32, lane = tid % 32;
  const Tables& t = p.tables;
  const int32_t tile = t.tile;
  if (tid < NL * MW) s_mask[tid] = 0u;
  const int32_t t_first = static_cast<int32_t>(c) * p.tpc;
  const int32_t n_tok = min(p.tpc, p.num_tokens - t_first);
  asm volatile("griddepcontrol.wait;" ::: "memory");
  __syncthreads();
  // ---- the routing: warp w takes tokens w and w + 32 of this CTA
  const bool two = static_cast<int32_t>(warp + kWarpsBig) < n_tok;
  if (static_cast<int32_t>(warp) < n_tok) {
    uint32_t expert[2] = {0u, 0u};
    uint16_t wbits[2] = {0, 0};
    // TRT-LLM's candidate top-10 (route_row_fast: the same bits as route_row)
    uint4 v[2][2];
    load_logits(p, t_first + static_cast<int32_t>(warp), lane, v[0][0], v[0][1]);
    if (two) load_logits(p, t_first + static_cast<int32_t>(warp + kWarpsBig), lane, v[1][0], v[1][1]);
    route_row_fast(v[0][0], v[0][1], lane, s_cand[warp], expert[0], wbits[0]);
    if (two) route_row_fast(v[1][0], v[1][1], lane, s_cand[warp], expert[1], wbits[1]);
#pragma unroll
    for (int j = 0; j < 2; ++j) {
      if (j == 1 && !two) break;
      const uint32_t tl = warp + j * kWarpsBig;  // the token within the CTA
      if (lane < kTopK) {
        const size_t o = static_cast<size_t>(t_first + static_cast<int32_t>(tl)) * kTopK + lane;
        p.topk_ids[o] = static_cast<int32_t>(expert[j]);
        p.topk_weights[o] = wbits[j];
        const uint16_t l = local_slot(static_cast<int32_t>(expert[j]), t);
        s_pick[tl * kTopK + lane] = l;
        if (l != kRemote) atomicOr(&s_mask[l * MW + tl / 32], 1u << (tl % 32));
      }
    }
  }
  __syncthreads();
  if (tid < NL) {
    int32_t cnt = 0;
#pragma unroll
    for (uint32_t w = 0; w < MW; ++w) cnt += __popc(s_mask[tid * MW + w]);
    s_cnt[tid] = cnt;
  }
  // ---- the cluster's per-expert counts through distributed shared memory
  cluster_arrive_release();
  cluster_wait_acquire();
  int32_t tot = 0, nct = 0;
  if (tid < NL) {
    int32_t v[CL];
#pragma unroll
    for (uint32_t k = 0; k < CL; ++k) v[k] = ld_dsmem(&s_cnt[tid], k);  // every peer's count in flight at once
    int32_t pre = 0;
#pragma unroll
    for (uint32_t k = 0; k < CL; ++k) {
      tot += v[k];
      pre += k < c ? v[k] : 0;
    }
    s_pre[tid] = pre;
    nct = (tot + tile - 1) / tile;
  }
  cluster_arrive_relaxed();  // this CTA has read its peers' counts (the matching wait is at the end)
  // exclusive scan of the CTAs per expert, one expert a thread
  int32_t incl = nct;
#pragma unroll
  for (uint32_t d = 1; d < 32; d <<= 1) {
    const int32_t up = __shfl_up_sync(0xffffffffu, incl, d);
    incl += lane >= d ? up : 0;
  }
  if (lane == 31) s_wsum[warp] = incl;
  __syncthreads();
  int32_t wpre = 0, total = 0;
#pragma unroll
  for (uint32_t w = 0; w < NL / 32; ++w) {
    const int32_t ws = s_wsum[w];
    wpre += w < warp ? ws : 0;
    total += ws;
  }
  const int32_t offs = incl - nct + wpre;
  if (tid < NL) {
    s_offs[tid] = offs;
    s_tot[tid] = tot;
    if (c == 0) t.tokens_per_expert[tid] = tot;
    if (tid % CL == c) {  // expert tid's CTA tables from routing CTA tid % CL
      for (int32_t cc = 0; cc < nct; ++cc) {
        const int32_t cta = offs + cc;
        t.cta_expert[cta] = static_cast<int32_t>(tid);
        t.cta_row_limit[cta] = min((cta + 1) * tile, offs * tile + tot);
      }
    }
  }
  if (c == 0 && tid == 0) {
    t.live_ctas[0] = total;
    t.total_padded[0] = total * tile;
  }
  __syncthreads();
  // the padding rows (-1) past an expert's last row: a warp an expert, spread over the routing CTAs
  for (uint32_t e = c + CL * warp; e < NL; e += CL * kWarpsBig) {
    const int32_t n = s_tot[e], first = s_offs[e] * tile;
    for (int32_t r = first + n + static_cast<int32_t>(lane); r < first + (n + tile - 1) / tile * tile; r += 32) {
      t.permuted_to_token[r] = -1;
    }
  }
  // ---- this CTA's picks: expanded -> permuted, permuted -> token (an expert's rows in token order)
  for (int32_t i = static_cast<int32_t>(tid); i < n_tok * static_cast<int32_t>(kTopK); i += kThreadsBig) {
    const uint32_t tl = static_cast<uint32_t>(i) / kTopK;
    const int32_t tok = t_first + static_cast<int32_t>(tl);
    const uint16_t l = s_pick[i];
    int32_t perm = -1;
    if (l != kRemote) {
      int32_t rank = s_pre[l];
#pragma unroll
      for (uint32_t w = 0; w < MW; ++w) {
        const uint32_t word = s_mask[l * MW + w];
        const uint32_t lo = 32 * w;
        rank += tl >= lo + 32 ? __popc(word) : (tl > lo ? __popc(word & ((1u << (tl - lo)) - 1u)) : 0);
      }
      perm = s_offs[l] * tile + rank;
      t.permuted_to_token[perm] = tok;
    }
    t.expanded_to_permuted[static_cast<size_t>(tok) * kTopK + (static_cast<uint32_t>(i) - tl * kTopK)] = perm;
  }
  cluster_wait_acquire();  // every peer has read this CTA's counts
  __syncthreads();
  asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
}

__device__ __forceinline__ void quant_cta(const Params& p, uint32_t qc, uint32_t q_ctas) {
  const float gs = p.global_scale[0];  // static (the experts' input scale)
  asm volatile("griddepcontrol.wait;" ::: "memory");
  const uint32_t blocks = static_cast<uint32_t>(p.num_tokens) * kBlocksPerRow;
  for (uint32_t blk = qc * kThreadsBig + threadIdx.x; blk < blocks; blk += q_ctas * kThreadsBig) {
    const uint32_t row = blk / kBlocksPerRow, cb = blk % kBlocksPerRow;
    const uint4* src = reinterpret_cast<const uint4*>(p.x + static_cast<size_t>(row) * kHidden + cb * kSfVec);
    uint32_t sf;
    uint2 packed;
    quantize_block(__ldcg(src), __ldcg(src + 1), gs, sf, packed);
    *reinterpret_cast<uint2*>(p.x_fp4 + static_cast<size_t>(row) * (kHidden / 2) + cb * 8) = packed;
    p.x_sf[static_cast<size_t>(row) * kBlocksPerRow + cb] = static_cast<uint8_t>(sf);
  }
  __syncthreads();
  asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
}

}  // namespace q35_route_big

extern "C" __global__ __launch_bounds__(q35_route_big::kThreadsBig, 1) void q35_route_big_quant(
    const __grid_constant__ q35_route_big::Params p) {
  if (blockIdx.x < CL) {
    q35_route_big::route_cta(p, blockIdx.x);
  } else {
    q35_route_big::quant_cta(p, blockIdx.x - CL, gridDim.x - CL);
  }
}
