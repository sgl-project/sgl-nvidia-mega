// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// The MoE front behind SGLang's router GEMM for verify steps wider than slot route serves (rows 17+): per token, the
// top-10 of its 512 bf16 router logits with the softmax renormalized over them (expert ids and bf16 weights, in the
// order and with the bits of FlashInfer's trtllm-gen routing, RenormalizeNaive), and the NVFP4 quantization of its
// bf16 row (16-element blocks, e4m3 scales, linear layout, the trtllm runner's quantizer), in one launch. The routed
// trtllm-gen MoE then takes the materialized routing and skips its own top-k kernel.
//
// One CTA per token: warp 0 ranks the logits (mkm_moe_m16.cu's routing, bit-exact with the single-block routing), the
// other 7 warps quantize the row (router_front.cu's quantize_block). Launched as a programmatic dependent; it waits
// before reading anything and releases its dependents at once (they read its outputs after their own wait).
//
// With routing tables the last CTA to finish its token also writes the trtllm-gen MoE body's routing
// tables for tile_n and the rank's window of local experts (what its routingIndicesCluster kernel writes, which the body
// then skips; route_tables.cuh), rows of one expert in the order of shared-memory atomics, as there.
//
// Built with NVRTC.

#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
#include <cuda_bf16.h>

// NVRTC has no <cstdint>
typedef unsigned char uint8_t;
typedef unsigned short uint16_t;
typedef int int32_t;
typedef unsigned int uint32_t;

#include "route_tables.cuh"

namespace q35_route_topk {

using namespace q35_route_tables;

constexpr uint32_t kHidden = 4096, kSfVec = 16;
constexpr uint32_t kBlocksPerRow = kHidden / kSfVec;  // 256

struct Params {
  const __nv_bfloat16* logits;  // [M, 512]
  const __nv_bfloat16* x;       // [M, 4096]
  const float* global_scale;    // [1]
  int32_t* topk_ids;            // [M, 10]
  uint16_t* topk_weights;       // [M, 10] bf16
  uint8_t* x_fp4;               // [M, 2048]
  uint8_t* x_sf;                // [M, 256]
  // routing tables (null counter: none); the counter is 0 between launches, the last CTA resets it
  Tables tables;
  int32_t num_tokens;  // M
};

// The routing tables from every token's top-10 (in global memory, visible to this CTA), by the whole CTA.
constexpr uint32_t kMaxRows = 896;  // router.py TOPK_MAX_ROWS
__device__ __forceinline__ void write_tables(const Params& p) {
  __shared__ uint16_t s_local[kMaxRows * kTopK];
  __shared__ int32_t s_count[kExperts], s_next[kExperts], s_warp[kThreads / 32];
  const uint32_t n = static_cast<uint32_t>(p.num_tokens) * kTopK;
  for (uint32_t i = threadIdx.x; i < n; i += kThreads) s_local[i] = local_slot(__ldcg(p.topk_ids + i), p.tables);
  q35_route_tables::write_tables<false>(p.tables, n, s_local, s_count, s_next, s_warp);
}

// ---- the NVFP4 quantizer (router_front.cu)
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

__device__ __forceinline__ void route_topk_quant(const Params& p) {
  __shared__ __align__(16) uint32_t s_cand[32];
  __shared__ int32_t s_last;
  const uint32_t t = blockIdx.x, warp = threadIdx.x / 32, lane = threadIdx.x % 32;
  asm volatile("griddepcontrol.wait;" ::: "memory");
  asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
  if (warp == 0) {
    const uint4* src = reinterpret_cast<const uint4*>(p.logits + static_cast<size_t>(t) * kExperts + lane * 16);
    uint32_t expert;
    uint16_t wbits;
    route_row(src[0], src[1], lane, s_cand, expert, wbits);
    if (lane < kTopK) {
      p.topk_ids[t * kTopK + lane] = static_cast<int32_t>(expert);
      p.topk_weights[t * kTopK + lane] = wbits;
    }
    if (p.tables.counter != nullptr) __threadfence();
  } else if (p.x_fp4 != nullptr) {  // null: the router front already quantized the rows
    const float gs = p.global_scale[0];
    const __nv_bfloat16* row = p.x + static_cast<size_t>(t) * kHidden;
    for (uint32_t blk = threadIdx.x - 32; blk < kBlocksPerRow; blk += kThreads - 32) {
      const uint4* src = reinterpret_cast<const uint4*>(row + blk * kSfVec);
      uint32_t sf;
      uint2 packed;
      quantize_block(src[0], src[1], gs, sf, packed);
      *reinterpret_cast<uint2*>(p.x_fp4 + static_cast<size_t>(t) * (kHidden / 2) + blk * 8) = packed;
      p.x_sf[static_cast<size_t>(t) * kBlocksPerRow + blk] = static_cast<uint8_t>(sf);
    }
  }
  if (p.tables.counter == nullptr) return;
  __syncthreads();
  if (threadIdx.x == 0) s_last = atomicAdd(p.tables.counter, 1) == p.num_tokens - 1;
  __syncthreads();
  if (!s_last) return;
  __threadfence();
  write_tables(p);
  if (threadIdx.x == 0) *p.tables.counter = 0;
}

}  // namespace q35_route_topk

extern "C" __global__ __launch_bounds__(q35_route_topk::kThreads, 1) void q35_route_topk_quant(
    const __grid_constant__ q35_route_topk::Params p) {
  q35_route_topk::route_topk_quant(p);
}
