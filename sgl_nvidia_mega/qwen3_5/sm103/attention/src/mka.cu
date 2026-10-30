// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// Copied from SGLang branch nvidia/qwen35-agentx-gb300:
// python/sglang/kernels/ops/attention/mka_fused.cu, device code unchanged. Built with NVRTC: the
// kernels are device bodies under C-linkage entries (below), and the push reads the plugin's hop.
//
// Fused full-attention prologue / epilogue for Qwen3.5 bs1 verify (M <= 8 tokens), SM100+.
//
// prologue: normed hidden (bf16) -> static FP8 quant -> qkv(+gate) GEMM -> q/k Gemma RMSNorm -> partial
//           NeoX M-RoPE -> FP8 Q, FP8 K/V into the paged cache (one KV head per rank), bf16 gate.
// epilogue: attention output (bf16) * sigmoid(gate) -> bf16 -> static FP8 -> o_proj GEMM -> bf16.
//
// Both kernels give one 8-CTA cluster 256 output rows and split K over the cluster ranks. Weights are
// pre-packed as mma.m16n8k32 e4m3 A fragments and loaded into registers before griddepcontrol.wait, so
// the weight stream overlaps the predecessor. Rank r owns token r: every rank pushes its partial rows of
// token r into rank r's inbox (st.async + mbarrier), which keeps a whole head local for the q/k norm.
// Rounding steps copy the stock Triton kernels (static quant, ATTN_FRONT, ATTN_GATE_QUANT) instruction
// for instruction; only the FP32 summation order of the two GEMMs and of the norm differs.
#include <cuda_bf16.h>

// the hop's device protocol (sm103/hop.headers()): q35hop::Site, push_frag, stage_peek; it also
// declares the fixed-width integers NVRTC lacks, but for int64_t
#include "q35_hop.cuh"

typedef long long int64_t;

namespace q35mka {

constexpr int kThreads = 256;
constexpr int kCluster = 8;
constexpr int kHeadDim = 256;
constexpr int kHalfRot = 32;  // rotary_dim 64

__device__ __forceinline__ void pdl_wait() { asm volatile("griddepcontrol.wait;" ::: "memory"); }
__device__ __forceinline__ void pdl_trigger() { asm volatile("griddepcontrol.launch_dependents;" ::: "memory"); }
__device__ __forceinline__ void cluster_arrive_relaxed() {
  asm volatile("barrier.cluster.arrive.relaxed.aligned;" ::: "memory");
}
__device__ __forceinline__ void cluster_wait() { asm volatile("barrier.cluster.wait.aligned;" ::: "memory"); }
__device__ __forceinline__ uint32_t cluster_ctarank() {
  uint32_t r;
  asm("mov.u32 %0, %%cluster_ctarank;" : "=r"(r));
  return r;
}
__device__ __forceinline__ uint32_t smem_u32(const void* ptr) {
  return static_cast<uint32_t>(__cvta_generic_to_shared(ptr));
}
__device__ __forceinline__ uint32_t mapa(uint32_t addr, uint32_t rank) {
  uint32_t out;
  asm volatile("mapa.shared::cluster.u32 %0, %1, %2;" : "=r"(out) : "r"(addr), "r"(rank));
  return out;
}
__device__ __forceinline__ void mbar_init(uint64_t* bar, uint32_t count) {
  asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" ::"r"(smem_u32(bar)), "r"(count) : "memory");
}
// Publishes the mbarrier init to the cluster before peers' st.async can credit it.
__device__ __forceinline__ void mbar_init_fence() {
  asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
}
__device__ __forceinline__ void mbar_expect_tx(uint64_t* bar, uint32_t bytes) {
  asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;" ::"r"(smem_u32(bar)), "r"(bytes)
               : "memory");
}
__device__ __forceinline__ void mbar_wait(uint64_t* bar, uint32_t parity) {
  asm volatile(
      "{\n .reg .pred p;\n WAIT_%=:\n mbarrier.try_wait.parity.shared::cta.b64 p, [%0], %1;\n @!p bra WAIT_%=;\n}" ::"r"(
          smem_u32(bar)),
      "r"(parity)
      : "memory");
}
__device__ __forceinline__ void st_async_v4(uint32_t dst, float a, float b, float c, float d, uint32_t bar) {
  asm volatile("st.async.shared::cluster.mbarrier::complete_tx::bytes.v4.f32 [%0], {%1, %2, %3, %4}, [%5];" ::"r"(dst),
               "f"(a), "f"(b), "f"(c), "f"(d), "r"(bar)
               : "memory");
}
__device__ __forceinline__ uint4 ldg_nc_v4(const uint4* p) {
  uint4 r;
  asm volatile("ld.global.nc.L1::no_allocate.v4.u32 {%0,%1,%2,%3}, [%4];"
               : "=r"(r.x), "=r"(r.y), "=r"(r.z), "=r"(r.w)
               : "l"(p));
  return r;
}
__device__ __forceinline__ float div_full(float a, float b) {
  float r;
  asm("div.full.f32 %0, %1, %2;" : "=f"(r) : "f"(a), "f"(b));
  return r;
}
__device__ __forceinline__ float rsqrt_approx_ftz(float a) {
  float r;
  asm("rsqrt.approx.ftz.f32 %0, %1;" : "=f"(r) : "f"(a));
  return r;
}
__device__ __forceinline__ float ex2_approx(float a) {
  float r;
  asm("ex2.approx.f32 %0, %1;" : "=f"(r) : "f"(a));
  return r;
}
__device__ __forceinline__ float bf16_round(float a) { return __bfloat162float(__float2bfloat16_rn(a)); }
// low byte = e4m3(lo), high byte = e4m3(hi)
__device__ __forceinline__ uint32_t e4m3x2(float lo, float hi) {
  uint16_t r;
  asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(r) : "f"(hi), "f"(lo));
  return r;
}
__device__ __forceinline__ uint8_t e4m3(float v) { return (uint8_t)(e4m3x2(v, 0.f) & 0xff); }

__device__ __forceinline__ void mma_e4m3(float (&d)[4], const uint4& a, uint32_t b0, uint32_t b1) {
  asm volatile(
      "mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, "
      "{%0,%1,%2,%3};"
      : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
      : "r"(a.x), "r"(a.y), "r"(a.z), "r"(a.w), "r"(b0), "r"(b1));
}

// _static_quant_fp8: y * div.full(1, s), clamp +-448, cvt.rn.satfinite.e4m3
__device__ __forceinline__ uint32_t quant4_static(const __nv_bfloat16* p, float inv) {
  const uint2 raw = *reinterpret_cast<const uint2*>(p);
  const __nv_bfloat162 v01 = *reinterpret_cast<const __nv_bfloat162*>(&raw.x);
  const __nv_bfloat162 v23 = *reinterpret_cast<const __nv_bfloat162*>(&raw.y);
  float f[4] = {__low2float(v01), __high2float(v01), __low2float(v23), __high2float(v23)};
#pragma unroll
  for (int i = 0; i < 4; ++i) f[i] = fminf(fmaxf(__fmul_rn(f[i], inv), -448.f), 448.f);
  return e4m3x2(f[0], f[1]) | (e4m3x2(f[2], f[3]) << 16);
}

// ATTN_GATE_QUANT: sig = div.full(1, 1 + ex2.approx(g * -log2e)); bf16(sig * o) * div.full(1, s); clamp
__device__ __forceinline__ float gate_quant1(float o, float g, float inv) {
  const float e = ex2_approx(__fmul_rn(g, __int_as_float(0xBFB8AA3B)));
  const float sig = div_full(1.0f, __fadd_rn(e, 1.0f));
  const float y = __fmul_rn(inv, bf16_round(__fmul_rn(sig, o)));
  return copysignf(fminf(fabsf(y), 448.f), y);
}
__device__ __forceinline__ uint32_t gate_quant4(const __nv_bfloat16* o, const __nv_bfloat16* g, float inv) {
  const uint2 ro = *reinterpret_cast<const uint2*>(o);
  const uint2 rg = *reinterpret_cast<const uint2*>(g);
  const __nv_bfloat162 o01 = *reinterpret_cast<const __nv_bfloat162*>(&ro.x);
  const __nv_bfloat162 o23 = *reinterpret_cast<const __nv_bfloat162*>(&ro.y);
  const __nv_bfloat162 g01 = *reinterpret_cast<const __nv_bfloat162*>(&rg.x);
  const __nv_bfloat162 g23 = *reinterpret_cast<const __nv_bfloat162*>(&rg.y);
  return e4m3x2(gate_quant1(__low2float(o01), __low2float(g01), inv),
                gate_quant1(__high2float(o01), __high2float(g01), inv)) |
         (e4m3x2(gate_quant1(__low2float(o23), __low2float(g23), inv),
                 gate_quant1(__high2float(o23), __high2float(g23), inv))
          << 16);
}

// The same roundings on operands already in registers (the wide kernels batch their loads).
__device__ __forceinline__ uint32_t quant4_static_r(const uint2 raw, float inv) {
  const __nv_bfloat162 v01 = *reinterpret_cast<const __nv_bfloat162*>(&raw.x);
  const __nv_bfloat162 v23 = *reinterpret_cast<const __nv_bfloat162*>(&raw.y);
  float f[4] = {__low2float(v01), __high2float(v01), __low2float(v23), __high2float(v23)};
#pragma unroll
  for (int i = 0; i < 4; ++i) f[i] = fminf(fmaxf(__fmul_rn(f[i], inv), -448.f), 448.f);
  return e4m3x2(f[0], f[1]) | (e4m3x2(f[2], f[3]) << 16);
}
__device__ __forceinline__ uint32_t gate_quant4_r(const uint2 ro, const uint2 rg, float inv) {
  const __nv_bfloat162 o01 = *reinterpret_cast<const __nv_bfloat162*>(&ro.x);
  const __nv_bfloat162 o23 = *reinterpret_cast<const __nv_bfloat162*>(&ro.y);
  const __nv_bfloat162 g01 = *reinterpret_cast<const __nv_bfloat162*>(&rg.x);
  const __nv_bfloat162 g23 = *reinterpret_cast<const __nv_bfloat162*>(&rg.y);
  return e4m3x2(gate_quant1(__low2float(o01), __low2float(g01), inv),
                gate_quant1(__high2float(o01), __high2float(g01), inv)) |
         (e4m3x2(gate_quant1(__low2float(o23), __low2float(g23), inv),
                 gate_quant1(__high2float(o23), __high2float(g23), inv))
          << 16);
}

// Inbox slot s of a 256-row tile -> row: warp w = s/32 holds rows 32w + {g, g+8, g+16, g+24}.
__device__ __forceinline__ int slot_row(int s) { return 32 * (s >> 5) + ((s >> 2) & 7) + 8 * (s & 3); }

struct ProParams {
  const __nv_bfloat16* x;  // [M, 4096]
  const uint4* w;          // packed qkv(+gate) weights
  const float* sx;
  const float* sw;
  const __nv_bfloat16* qnw;
  const __nv_bfloat16* knw;
  const float* cos_sin;    // [max_pos, 64] fp32
  const int64_t* pos;      // [3, M]
  const int64_t* axis;     // [32]
  const void* loc;         // [M] cache slots, int64 or int32
  int loc_i32;
  uint8_t* k_cache;        // [slots, 256] e4m3
  uint8_t* v_cache;
  const float* k_scale;
  const float* v_scale;
  uint8_t* q_out;          // [M, nq*256] e4m3
  __nv_bfloat16* gate_out; // [M, nq*256]
  int m;
  int nq;
  int pos_stride;
  int cs_stride;
  float eps;
  int split;               // 1: this launch skips the gate units, which mka_gate_kernel computes
  const uint4* wg;         // split: gate rows packed 32 rows x full K per CTA
};

// grid = (2*nq + 2) units x 8 ranks; unit u = qkv rows [256u, 256u + 256) (q_h, gate_h per head, then k,
// v); rank r = K chunk [512r, 512r + 512) for the GEMM and token r for the epilogue.
template <bool kEarlyTrigger>
__device__ __forceinline__ void mka_prologue_kernel(const ProParams p) {
  __shared__ __align__(16) float inbox[kCluster][256];
  __shared__ __align__(16) uint2 bfr[16][32];
  __shared__ float red8[8];
  __shared__ float nrm[256];
  __shared__ __align__(8) uint64_t ibar;
  const int rank = (int)cluster_ctarank();
  const int nq = p.nq;
  // split launches cover q_0..q_{nq-1}, k, v; natural units interleave q_h and gate_h, then k, v
  const int u = blockIdx.x / kCluster;
  const int unit = !p.split ? u : (u < nq ? 2 * u : 2 * nq + (u - nq));
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
  const int g = lane >> 2, t = lane & 3;
  if (tid == 0) {
    mbar_init(&ibar, 1);
    mbar_init_fence();
  }
  __syncthreads();
  cluster_arrive_relaxed();

  uint4 a[16][2];
  {
    const uint4* wb = p.w + (size_t)(unit * kCluster + rank) * (8 * 16 * 2 * 32) + warp * 1024 + lane;
#pragma unroll
    for (int s = 0; s < 16; ++s)
#pragma unroll
      for (int b = 0; b < 2; ++b) a[s][b] = ldg_nc_v4(wb + (s * 2 + b) * 32);
  }
  // Static operands only before the wait, and nothing consumes them there.
  const bool is_q = unit < 2 * nq && (unit & 1) == 0;
  const bool is_gate = unit < 2 * nq && (unit & 1) == 1;
  const bool is_k = unit == 2 * nq;
  const bool needs_norm = is_q || is_k;
  const int f = slot_row(tid);
  const bool rot = needs_norm && f < 2 * kHalfRot;
  const int rl = f & (kHalfRot - 1);
  const float sx = __ldg(p.sx), sw = __ldg(p.sw);
  __nv_bfloat16 nwr = __float2bfloat16_rn(0.f);
  if (needs_norm) nwr = (is_q ? p.qnw : p.knw)[f];
  int64_t ax = 0;
  if (rot) ax = __ldg(p.axis + rl);
  float kvs = 1.f;
  if (is_k) kvs = __ldg(p.k_scale);
  if (!needs_norm && !is_gate) kvs = __ldg(p.v_scale);

  if (!kEarlyTrigger) {
    uint32_t weight_dep = __float_as_uint(sx) ^ __float_as_uint(sw);
#pragma unroll
    for (int s = 0; s < 16; ++s) weight_dep ^= a[s][0].x ^ a[s][1].x;
    q35hop::pin_before_wait(weight_dep, reinterpret_cast<uint64_t>(p.w));
  }
  pdl_wait();
  // Keep this release after the wait: q35mka_gate reads x (the hop's output) before its own wait.
  if (kEarlyTrigger) pdl_trigger();
  const float inv_sx = div_full(1.0f, sx);
  const float alpha = __fmul_rn(sx, sw);
  const int m = p.m;
  const bool owner = rank < m;
  int64_t slot = 0;
  float cs_c = 0.f, cs_s = 0.f;
  if (owner) {
    slot = p.loc_i32 ? (int64_t)__ldg(static_cast<const int32_t*>(p.loc) + rank)
                     : __ldg(static_cast<const int64_t*>(p.loc) + rank);
    if (rot) {
      const int64_t ps = __ldg(p.pos + ax * p.pos_stride + rank);
      cs_c = __ldg(p.cos_sin + ps * p.cs_stride + rl);
      cs_s = __ldg(p.cos_sin + ps * p.cs_stride + kHalfRot + rl);
    }
  }
  {
    const bool tok_ok = g < m;
    const __nv_bfloat16* xr = p.x + (size_t)g * 4096 + rank * 512 + 4 * t;
#pragma unroll
    for (int i = 0; i < 2; ++i) {
      const int s = 2 * warp + i;
      bfr[s][lane] = tok_ok ? make_uint2(quant4_static(xr + s * 32, inv_sx), quant4_static(xr + s * 32 + 16, inv_sx))
                            : make_uint2(0u, 0u);
    }
  }
  __syncthreads();
  float ch[2][4][4];
#pragma unroll
  for (int b = 0; b < 2; ++b)
#pragma unroll
    for (int c = 0; c < 4; ++c)
#pragma unroll
      for (int i = 0; i < 4; ++i) ch[b][c][i] = 0.f;
#pragma unroll
  for (int s = 0; s < 16; ++s) {
    const uint2 bb = bfr[s][lane];
    mma_e4m3(ch[0][s & 3], a[s][0], bb.x, bb.y);
    mma_e4m3(ch[1][s & 3], a[s][1], bb.x, bb.y);
  }
  float acc[2][4];
#pragma unroll
  for (int b = 0; b < 2; ++b)
#pragma unroll
    for (int i = 0; i < 4; ++i)
      acc[b][i] = __fadd_rn(__fadd_rn(ch[b][0][i], ch[b][1][i]), __fadd_rn(ch[b][2][i], ch[b][3][i]));

  if (tid == 0 && owner) mbar_expect_tx(&ibar, kCluster * 256 * 4);
  cluster_wait();
  {
    const uint32_t src = smem_u32(&inbox[rank][warp * 32 + g * 4]);
    const uint32_t bar = smem_u32(&ibar);
#pragma unroll
    for (int j = 0; j < 2; ++j) {
      const int tok = 2 * t + j;
      if (tok < m) st_async_v4(mapa(src, tok), acc[0][j], acc[0][2 + j], acc[1][j], acc[1][2 + j], mapa(bar, tok));
    }
  }
  if (!owner) {
    if (!kEarlyTrigger) pdl_trigger();
    return;
  }
  mbar_wait(&ibar, 0);  // also keeps this CTA alive until every peer's push has landed

  float v = 0.f;
#pragma unroll
  for (int r = 0; r < kCluster; ++r) v = __fadd_rn(v, inbox[r][tid]);
  v = bf16_round(__fmul_rn(v, alpha));
  const int tok = rank;
  if (is_gate) {
    p.gate_out[(size_t)tok * nq * kHeadDim + (unit >> 1) * kHeadDim + f] = __float2bfloat16_rn(v);
  } else if (!needs_norm) {
    p.v_cache[(size_t)slot * kHeadDim + f] = e4m3(__fmul_rn(__fdiv_rn(1.0f, kvs), v));
  } else {
    float ss = __fmul_rn(v, v);
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) ss = __fadd_rn(ss, __shfl_xor_sync(0xffffffffu, ss, o));
    if (lane == 0) red8[warp] = ss;
    __syncthreads();
    float tot = 0.f;
#pragma unroll
    for (int i = 0; i < 8; ++i) tot = __fadd_rn(tot, red8[i]);
    const float inv = rsqrt_approx_ftz(__fadd_rn(div_full(tot, 256.0f), p.eps));
    float n = bf16_round(__fmul_rn(__fadd_rn(__bfloat162float(nwr), 1.0f), __fmul_rn(inv, v)));
    nrm[f] = n;
    __syncthreads();
    if (rot) {
      // first half: fma(cos, x1, -(sin * x2)); second half: fma(x1, sin, cos * x2)
      const float x1 = f < kHalfRot ? n : nrm[f - kHalfRot];
      const float x2 = f < kHalfRot ? nrm[f + kHalfRot] : n;
      n = f < kHalfRot ? bf16_round(__fmaf_rn(cs_c, x1, -__fmul_rn(cs_s, x2)))
                       : bf16_round(__fmaf_rn(x1, cs_s, __fmul_rn(cs_c, x2)));
    }
    if (is_k)
      p.k_cache[(size_t)slot * kHeadDim + f] = e4m3(__fmul_rn(__fdiv_rn(1.0f, kvs), n));
    else
      p.q_out[(size_t)tok * nq * kHeadDim + (unit >> 1) * kHeadDim + f] = e4m3(n);
  }
  if (!kEarlyTrigger) pdl_trigger();
}

// Gate rows for split launches: no cluster, CTA c owns gate rows [32c, 32c + 32) (head c/8) over the
// full K = 4096; warp w takes K [512w, 512w + 512) and the 8 warp partials are summed in warp order.
// It runs beside the prologue: launched when the prologue triggers (after the hop's output is visible),
// it reads x through L2 without its own griddepcontrol.wait and waits for the prologue only at the end,
// so a consumer of this launch also sees the prologue's Q / KV.
__device__ __forceinline__ void mka_gate_kernel(const ProParams p) {
  __shared__ __align__(16) float red[8][32][8];
  const int cta = blockIdx.x;
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
  const int g = lane >> 2, t = lane & 3;
  pdl_trigger();
  uint4 a[16][2];
  {
    const uint4* wb = p.wg + ((size_t)cta * 8 + warp) * (16 * 2 * 32) + lane;
#pragma unroll
    for (int s = 0; s < 16; ++s)
#pragma unroll
      for (int b = 0; b < 2; ++b) a[s][b] = ldg_nc_v4(wb + (s * 2 + b) * 32);
  }
  const float sx = __ldg(p.sx), sw = __ldg(p.sw);
  const float inv_sx = div_full(1.0f, sx);
  const float alpha = __fmul_rn(sx, sw);
  const int m = p.m;
  const bool tok_ok = g < m;
  const __nv_bfloat16* xr = p.x + (size_t)g * 4096 + warp * 512 + 4 * t;
  uint2 raw[16][2];
#pragma unroll
  for (int s = 0; s < 16; ++s)
#pragma unroll
    for (int h = 0; h < 2; ++h)
      raw[s][h] = tok_ok ? __ldcg(reinterpret_cast<const uint2*>(xr + s * 32 + h * 16)) : make_uint2(0u, 0u);
  float ch[2][4][4];
#pragma unroll
  for (int b = 0; b < 2; ++b)
#pragma unroll
    for (int c = 0; c < 4; ++c)
#pragma unroll
      for (int i = 0; i < 4; ++i) ch[b][c][i] = 0.f;
#pragma unroll
  for (int s = 0; s < 16; ++s) {
    const uint32_t b0 = quant4_static(reinterpret_cast<const __nv_bfloat16*>(&raw[s][0]), inv_sx);
    const uint32_t b1 = quant4_static(reinterpret_cast<const __nv_bfloat16*>(&raw[s][1]), inv_sx);
    mma_e4m3(ch[0][s & 3], a[s][0], b0, b1);
    mma_e4m3(ch[1][s & 3], a[s][1], b0, b1);
  }
  {
    float4* dst = reinterpret_cast<float4*>(&red[warp][lane][0]);
#pragma unroll
    for (int b = 0; b < 2; ++b)
      dst[b] = make_float4(__fadd_rn(__fadd_rn(ch[b][0][0], ch[b][1][0]), __fadd_rn(ch[b][2][0], ch[b][3][0])),
                           __fadd_rn(__fadd_rn(ch[b][0][1], ch[b][1][1]), __fadd_rn(ch[b][2][1], ch[b][3][1])),
                           __fadd_rn(__fadd_rn(ch[b][0][2], ch[b][1][2]), __fadd_rn(ch[b][2][2], ch[b][3][2])),
                           __fadd_rn(__fadd_rn(ch[b][0][3], ch[b][1][3]), __fadd_rn(ch[b][2][3], ch[b][3][3])));
  }
  __syncthreads();
  {
    // thread -> fragment lane l = tid/8, element e = tid%8: block e/4, row 16*(e/4) + g' + 8*((e%4)/2),
    // token 2t' + (e%2)
    const int l = tid >> 3, e = tid & 7;
    float v = red[0][l][e];
#pragma unroll
    for (int w = 1; w < 8; ++w) v = __fadd_rn(v, red[w][l][e]);
    const int row = 16 * (e >> 2) + (l >> 2) + 8 * ((e & 3) >> 1);
    const int tok = 2 * (l & 3) + (e & 1);
    const int h = cta >> 3, r = (cta & 7) * 32 + row;
    if (tok < m)
      p.gate_out[(size_t)tok * p.nq * kHeadDim + h * kHeadDim + r] = __float2bfloat16_rn(__fmul_rn(v, alpha));
  }
  pdl_wait();  // completes only after the prologue grid has completed
}

struct EpiParams {
  const __nv_bfloat16* attn;  // [M, nq*256]
  const __nv_bfloat16* gate;  // [M, nq*256]
  const uint4* w;             // packed o_proj weights
  const float* sx;
  const float* sw;
  __nv_bfloat16* out;         // [M, 4096]
  int m;
  int push;                   // 1: also push the bf16 partial into the q35 hop mailbox (MKH interface v1)
  q35hop::Site site;
};

// grid = 16 units x 8 ranks; NQ = q heads per rank (K = NQ*256, K chunk NQ*32 per rank).
template <int NQ, bool kEarlyTrigger>
__device__ __forceinline__ void mka_epilogue_kernel(const EpiParams p) {
  constexpr int kSteps = NQ;
  constexpr int kK = NQ * kHeadDim;
  constexpr int kWarpU4 = kSteps * 2 * 32;
  __shared__ __align__(16) float inbox[kCluster][256];
  __shared__ __align__(16) uint2 bfr[kSteps][32];
  __shared__ __align__(16) __nv_bfloat16 ob[256];
  __shared__ __align__(8) uint64_t ibar;
  const int rank = (int)cluster_ctarank();
  const int unit = blockIdx.x / kCluster;
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
  const int g = lane >> 2, t = lane & 3;
  if (tid == 0) {
    mbar_init(&ibar, 1);
    mbar_init_fence();
  }
  __syncthreads();
  cluster_arrive_relaxed();

  uint4 a[kSteps][2];
  {
    const uint4* wb = p.w + (size_t)(unit * kCluster + rank) * (8 * kWarpU4) + warp * kWarpU4 + lane;
#pragma unroll
    for (int s = 0; s < kSteps; ++s)
#pragma unroll
      for (int b = 0; b < 2; ++b) a[s][b] = ldg_nc_v4(wb + (s * 2 + b) * 32);
  }
  const float sx = __ldg(p.sx), sw = __ldg(p.sw);
  uint32_t weight_dep = __float_as_uint(sx) ^ __float_as_uint(sw);
#pragma unroll
  for (int s = 0; s < kSteps; ++s) weight_dep ^= a[s][0].x ^ a[s][1].x;
  q35hop::pin_before_wait(weight_dep, reinterpret_cast<uint64_t>(p.w));

  pdl_wait();
  if (kEarlyTrigger) pdl_trigger();
  const float inv_sx = div_full(1.0f, sx);
  const float alpha = __fmul_rn(sx, sw);
  const int m = p.m;
  {
    const bool tok_ok = g < m;
    const size_t off = (size_t)g * kK + rank * (kSteps * 32) + 4 * t;
    uint32_t* bw = reinterpret_cast<uint32_t*>(&bfr[0][0]);
#pragma unroll
    for (int i = warp; i < 2 * kSteps; i += 8) {
      const int s = i >> 1, h = i & 1;
      bw[(s * 32 + lane) * 2 + h] =
          tok_ok ? gate_quant4(p.attn + off + s * 32 + h * 16, p.gate + off + s * 32 + h * 16, inv_sx) : 0u;
    }
  }
  __syncthreads();
  constexpr int kCh = kSteps < 4 ? kSteps : 4;
  float ch[2][kCh][4];
#pragma unroll
  for (int b = 0; b < 2; ++b)
#pragma unroll
    for (int c = 0; c < kCh; ++c)
#pragma unroll
      for (int i = 0; i < 4; ++i) ch[b][c][i] = 0.f;
#pragma unroll
  for (int s = 0; s < kSteps; ++s) {
    const uint2 bb = bfr[s][lane];
    mma_e4m3(ch[0][s % kCh], a[s][0], bb.x, bb.y);
    mma_e4m3(ch[1][s % kCh], a[s][1], bb.x, bb.y);
  }
  float acc[2][4];
#pragma unroll
  for (int b = 0; b < 2; ++b)
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      float x = ch[b][0][i];
#pragma unroll
      for (int c = 1; c < kCh; ++c) x = __fadd_rn(x, ch[b][c][i]);
      acc[b][i] = x;
    }
  const bool owner = rank < m;
  if (tid == 0 && owner) mbar_expect_tx(&ibar, kCluster * 256 * 4);
  cluster_wait();
  {
    const uint32_t src = smem_u32(&inbox[rank][warp * 32 + g * 4]);
    const uint32_t bar = smem_u32(&ibar);
#pragma unroll
    for (int j = 0; j < 2; ++j) {
      const int tok = 2 * t + j;
      if (tok < m) st_async_v4(mapa(src, tok), acc[0][j], acc[0][2 + j], acc[1][j], acc[1][2 + j], mapa(bar, tok));
    }
  }
  if (!owner) {
    if (!kEarlyTrigger) pdl_trigger();
    return;
  }
  mbar_wait(&ibar, 0);
  float v = 0.f;
#pragma unroll
  for (int r = 0; r < kCluster; ++r) v = __fadd_rn(v, inbox[r][tid]);
  const __nv_bfloat16 o = __float2bfloat16_rn(__fmul_rn(v, alpha));
  if (p.push) {
    // fragments are 8 consecutive columns, so regroup the row through smem; push before the store
    ob[slot_row(tid)] = o;
    __syncthreads();
    if (tid < 32)
      q35hop::push_frag(p.site, q35hop::stage_peek(p.site), rank, unit * 32 + tid,
                        *reinterpret_cast<const uint4*>(&ob[8 * tid]));
  }
  p.out[(size_t)rank * 4096 + unit * kHeadDim + slot_row(tid)] = o;
  if (!kEarlyTrigger) pdl_trigger();
}


// Wide steps (9-112 rows, several requests' verify rows; TP8, no split gate launch): the same per-CTA math as
// the kernels above on ceil(M/8) token tiles in one epoch. Weights load once; every tile's MMA reuses them, and
// each tile's partial rows go straight to their owner (token t to rank t mod 8, inbox slot t / 8) with no
// barrier between tiles. Tile t's arithmetic equals the narrow kernel's on rows [8t, 8t + 8), bit for bit.
constexpr int kWideTiles = 14;  // 112 rows
constexpr int kInboxBytes = kCluster * 256 * 4;  // one owned token's inbox

template <bool kEarlyTrigger>
__device__ __forceinline__ void mka_prologue_wide_kernel(const ProParams p) {
  extern __shared__ __align__(16) unsigned char wsm[];
  float* inbox = reinterpret_cast<float*>(wsm);                                       // [tile][rank][256]
  uint2* bfr = reinterpret_cast<uint2*>(wsm + kWideTiles * kInboxBytes);              // [tile][16][32]
  __shared__ float red8[8];
  __shared__ float nrm[256];
  __shared__ float tcs[kWideTiles][2 * kHalfRot];
  __shared__ int64_t tslot[kWideTiles];
  __shared__ __align__(8) uint64_t ibar;
  const int rank = (int)cluster_ctarank();
  const int nq = p.nq;
  const int unit = blockIdx.x / kCluster;
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
  const int g = lane >> 2, t4 = lane & 3;
  if (tid == 0) {
    mbar_init(&ibar, 1);
    mbar_init_fence();
  }
  __syncthreads();
  cluster_arrive_relaxed();

  uint4 a[16][2];
  {
    const uint4* wb = p.w + (size_t)(unit * kCluster + rank) * (8 * 16 * 2 * 32) + warp * 1024 + lane;
#pragma unroll
    for (int s = 0; s < 16; ++s)
#pragma unroll
      for (int b = 0; b < 2; ++b) a[s][b] = ldg_nc_v4(wb + (s * 2 + b) * 32);
  }
  const bool is_q = unit < 2 * nq && (unit & 1) == 0;
  const bool is_gate = unit < 2 * nq && (unit & 1) == 1;
  const bool is_k = unit == 2 * nq;
  const bool needs_norm = is_q || is_k;
  const int f = slot_row(tid);
  const bool rot = needs_norm && f < 2 * kHalfRot;
  const int rl = f & (kHalfRot - 1);
  const float sx = __ldg(p.sx), sw = __ldg(p.sw);
  __nv_bfloat16 nwr = __float2bfloat16_rn(0.f);
  if (needs_norm) nwr = (is_q ? p.qnw : p.knw)[f];
  float kvs = 1.f;
  if (is_k) kvs = __ldg(p.k_scale);
  if (!needs_norm && !is_gate) kvs = __ldg(p.v_scale);

  pdl_wait();
  if (kEarlyTrigger) pdl_trigger();
  const float inv_sx = div_full(1.0f, sx);
  const float alpha = __fmul_rn(sx, sw);
  const int m = p.m;
  const int tiles = (m + 7) >> 3;
  const int own = rank < m ? (m - rank + 7) >> 3 : 0;  // tokens rank, rank + 8, ...
  // the owned tokens' cache slots and cos / sin, all in flight at once
  if (tid < own) {
    const int tok = 8 * tid + rank;
    tslot[tid] = p.loc_i32 ? (int64_t)__ldg(static_cast<const int32_t*>(p.loc) + tok)
                           : __ldg(static_cast<const int64_t*>(p.loc) + tok);
  }
  if (needs_norm) {
    for (int i = tid; i < own * kHalfRot; i += kThreads) {
      const int o = i / kHalfRot, l = i % kHalfRot;
      const int64_t ps = __ldg(p.pos + __ldg(p.axis + l) * p.pos_stride + 8 * o + rank);
      tcs[o][l] = __ldg(p.cos_sin + ps * p.cs_stride + l);
      tcs[o][kHalfRot + l] = __ldg(p.cos_sin + ps * p.cs_stride + kHalfRot + l);
    }
  }
  // x's tiles in batches of 4, every load of a batch in flight before its first use
  for (int t0 = 0; t0 < tiles; t0 += 4) {
    uint2 raw[4][2][2];
#pragma unroll
    for (int u = 0; u < 4; ++u) {
      const bool tok_ok = 8 * (t0 + u) + g < m;
      const __nv_bfloat16* xr = p.x + (size_t)(8 * (t0 + u) + g) * 4096 + rank * 512 + 4 * t4;
#pragma unroll
      for (int i = 0; i < 2; ++i)
#pragma unroll
        for (int hh = 0; hh < 2; ++hh)
          raw[u][i][hh] = tok_ok ? *reinterpret_cast<const uint2*>(xr + (2 * warp + i) * 32 + hh * 16)
                                 : make_uint2(0u, 0u);
    }
#pragma unroll
    for (int u = 0; u < 4; ++u) {
      if (t0 + u >= tiles) break;
      const bool tok_ok = 8 * (t0 + u) + g < m;
#pragma unroll
      for (int i = 0; i < 2; ++i)
        bfr[((t0 + u) * 16 + 2 * warp + i) * 32 + lane] =
            tok_ok ? make_uint2(quant4_static_r(raw[u][i][0], inv_sx), quant4_static_r(raw[u][i][1], inv_sx))
                   : make_uint2(0u, 0u);
    }
  }
  __syncthreads();
  if (tid == 0 && own) mbar_expect_tx(&ibar, own * kInboxBytes);
  cluster_wait();
  const uint32_t bar = smem_u32(&ibar);
  for (int t = 0; t < tiles; ++t) {
    float ch[2][4][4];
#pragma unroll
    for (int b = 0; b < 2; ++b)
#pragma unroll
      for (int c = 0; c < 4; ++c)
#pragma unroll
        for (int i = 0; i < 4; ++i) ch[b][c][i] = 0.f;
#pragma unroll
    for (int s = 0; s < 16; ++s) {
      const uint2 bb = bfr[(t * 16 + s) * 32 + lane];
      mma_e4m3(ch[0][s & 3], a[s][0], bb.x, bb.y);
      mma_e4m3(ch[1][s & 3], a[s][1], bb.x, bb.y);
    }
    float acc[2][4];
#pragma unroll
    for (int b = 0; b < 2; ++b)
#pragma unroll
      for (int i = 0; i < 4; ++i)
        acc[b][i] = __fadd_rn(__fadd_rn(ch[b][0][i], ch[b][1][i]), __fadd_rn(ch[b][2][i], ch[b][3][i]));
    const uint32_t src = smem_u32(&inbox[(t * kCluster + rank) * 256 + warp * 32 + g * 4]);
#pragma unroll
    for (int j = 0; j < 2; ++j) {
      const int tok = 2 * t4 + j;
      if (8 * t + tok < m)
        st_async_v4(mapa(src, tok), acc[0][j], acc[0][2 + j], acc[1][j], acc[1][2 + j], mapa(bar, tok));
    }
  }
  if (!own) {
    if (!kEarlyTrigger) pdl_trigger();
    return;
  }
  mbar_wait(&ibar, 0);  // also keeps this CTA alive until every peer's push has landed

  for (int o = 0; o < own; ++o) {
    const int tok = 8 * o + rank;
    float v = 0.f;
#pragma unroll
    for (int r = 0; r < kCluster; ++r) v = __fadd_rn(v, inbox[(o * kCluster + r) * 256 + tid]);
    v = bf16_round(__fmul_rn(v, alpha));
    const int64_t slot = tslot[o];
    if (is_gate) {
      p.gate_out[(size_t)tok * nq * kHeadDim + (unit >> 1) * kHeadDim + f] = __float2bfloat16_rn(v);
    } else if (!needs_norm) {
      p.v_cache[(size_t)slot * kHeadDim + f] = e4m3(__fmul_rn(__fdiv_rn(1.0f, kvs), v));
    } else {
      float ss = __fmul_rn(v, v);
#pragma unroll
      for (int sh = 16; sh > 0; sh >>= 1) ss = __fadd_rn(ss, __shfl_xor_sync(0xffffffffu, ss, sh));
      if (lane == 0) red8[warp] = ss;
      __syncthreads();
      float tot = 0.f;
#pragma unroll
      for (int i = 0; i < 8; ++i) tot = __fadd_rn(tot, red8[i]);
      const float inv = rsqrt_approx_ftz(__fadd_rn(div_full(tot, 256.0f), p.eps));
      float n = bf16_round(__fmul_rn(__fadd_rn(__bfloat162float(nwr), 1.0f), __fmul_rn(inv, v)));
      nrm[f] = n;
      __syncthreads();
      if (rot) {
        const float cs_c = tcs[o][rl], cs_s = tcs[o][kHalfRot + rl];
        const float x1 = f < kHalfRot ? n : nrm[f - kHalfRot];
        const float x2 = f < kHalfRot ? nrm[f + kHalfRot] : n;
        n = f < kHalfRot ? bf16_round(__fmaf_rn(cs_c, x1, -__fmul_rn(cs_s, x2)))
                         : bf16_round(__fmaf_rn(x1, cs_s, __fmul_rn(cs_c, x2)));
      }
      if (is_k)
        p.k_cache[(size_t)slot * kHeadDim + f] = e4m3(__fmul_rn(__fdiv_rn(1.0f, kvs), n));
      else
        p.q_out[(size_t)tok * nq * kHeadDim + (unit >> 1) * kHeadDim + f] = e4m3(n);
      __syncthreads();  // red8 / nrm are reused by the next token
    }
  }
  if (!kEarlyTrigger) pdl_trigger();
}

template <int NQ, bool kEarlyTrigger>
__device__ __forceinline__ void mka_epilogue_wide_kernel(const EpiParams p) {
  constexpr int kSteps = NQ;
  constexpr int kK = NQ * kHeadDim;
  constexpr int kWarpU4 = kSteps * 2 * 32;
  extern __shared__ __align__(16) unsigned char wsm[];
  float* inbox = reinterpret_cast<float*>(wsm);                           // [tile][rank][256]
  uint2* bfr = reinterpret_cast<uint2*>(wsm + kWideTiles * kInboxBytes);  // [tile][kSteps][32]
  __shared__ __align__(16) __nv_bfloat16 ob[256];
  __shared__ __align__(8) uint64_t ibar;
  const int rank = (int)cluster_ctarank();
  const int unit = blockIdx.x / kCluster;
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
  const int g = lane >> 2, t4 = lane & 3;
  if (tid == 0) {
    mbar_init(&ibar, 1);
    mbar_init_fence();
  }
  __syncthreads();
  cluster_arrive_relaxed();

  uint4 a[kSteps][2];
  {
    const uint4* wb = p.w + (size_t)(unit * kCluster + rank) * (8 * kWarpU4) + warp * kWarpU4 + lane;
#pragma unroll
    for (int s = 0; s < kSteps; ++s)
#pragma unroll
      for (int b = 0; b < 2; ++b) a[s][b] = ldg_nc_v4(wb + (s * 2 + b) * 32);
  }
  const float sx = __ldg(p.sx), sw = __ldg(p.sw);
  uint32_t weight_dep = __float_as_uint(sx) ^ __float_as_uint(sw);
#pragma unroll
  for (int s = 0; s < kSteps; ++s) weight_dep ^= a[s][0].x ^ a[s][1].x;
  q35hop::pin_before_wait(weight_dep, reinterpret_cast<uint64_t>(p.w));

  pdl_wait();
  if (kEarlyTrigger) pdl_trigger();
  const float inv_sx = div_full(1.0f, sx);
  const float alpha = __fmul_rn(sx, sw);
  const int m = p.m;
  const int tiles = (m + 7) >> 3;
  const int own = rank < m ? (m - rank + 7) >> 3 : 0;
  static_assert(2 * kSteps == 8, "one (step, half) per warp");
  {
    // every tile's operands in flight before the first use
    const int s = warp >> 1, h = warp & 1;
    uint2 ro[kWideTiles], rg[kWideTiles];
#pragma unroll
    for (int t = 0; t < kWideTiles; ++t) {
      const bool ok = 8 * t + g < m;
      const size_t off = (size_t)(8 * t + g) * kK + rank * (kSteps * 32) + 4 * t4 + s * 32 + h * 16;
      ro[t] = ok ? *reinterpret_cast<const uint2*>(p.attn + off) : make_uint2(0u, 0u);
      rg[t] = ok ? *reinterpret_cast<const uint2*>(p.gate + off) : make_uint2(0u, 0u);
    }
#pragma unroll
    for (int t = 0; t < kWideTiles; ++t) {
      if (t >= tiles) break;
      uint32_t* bw = reinterpret_cast<uint32_t*>(bfr + t * kSteps * 32);
      bw[(s * 32 + lane) * 2 + h] = 8 * t + g < m ? gate_quant4_r(ro[t], rg[t], inv_sx) : 0u;
    }
  }
  __syncthreads();
  if (tid == 0 && own) mbar_expect_tx(&ibar, own * kInboxBytes);
  cluster_wait();
  const uint32_t bar = smem_u32(&ibar);
  constexpr int kCh = kSteps < 4 ? kSteps : 4;
  for (int t = 0; t < tiles; ++t) {
    float ch[2][kCh][4];
#pragma unroll
    for (int b = 0; b < 2; ++b)
#pragma unroll
      for (int c = 0; c < kCh; ++c)
#pragma unroll
        for (int i = 0; i < 4; ++i) ch[b][c][i] = 0.f;
#pragma unroll
    for (int s = 0; s < kSteps; ++s) {
      const uint2 bb = bfr[(t * kSteps + s) * 32 + lane];
      mma_e4m3(ch[0][s % kCh], a[s][0], bb.x, bb.y);
      mma_e4m3(ch[1][s % kCh], a[s][1], bb.x, bb.y);
    }
    float acc[2][4];
#pragma unroll
    for (int b = 0; b < 2; ++b)
#pragma unroll
      for (int i = 0; i < 4; ++i) {
        float x = ch[b][0][i];
#pragma unroll
        for (int c = 1; c < kCh; ++c) x = __fadd_rn(x, ch[b][c][i]);
        acc[b][i] = x;
      }
    const uint32_t src = smem_u32(&inbox[(t * kCluster + rank) * 256 + warp * 32 + g * 4]);
#pragma unroll
    for (int j = 0; j < 2; ++j) {
      const int tok = 2 * t4 + j;
      if (8 * t + tok < m)
        st_async_v4(mapa(src, tok), acc[0][j], acc[0][2 + j], acc[1][j], acc[1][2 + j], mapa(bar, tok));
    }
  }
  if (!own) {
    if (!kEarlyTrigger) pdl_trigger();
    return;
  }
  mbar_wait(&ibar, 0);
  const uint32_t stage = p.push ? q35hop::stage_peek(p.site) : 0u;
  for (int o = 0; o < own; ++o) {
    const int tok = 8 * o + rank;
    float v = 0.f;
#pragma unroll
    for (int r = 0; r < kCluster; ++r) v = __fadd_rn(v, inbox[(o * kCluster + r) * 256 + tid]);
    const __nv_bfloat16 out = __float2bfloat16_rn(__fmul_rn(v, alpha));
    if (p.push) {
      ob[slot_row(tid)] = out;
      __syncthreads();
      if (tid < 32)
        q35hop::push_frag(p.site, stage, tok, unit * 32 + tid, *reinterpret_cast<const uint4*>(&ob[8 * tid]));
      __syncthreads();  // ob is reused by the next token
    }
    p.out[(size_t)tok * 4096 + unit * kHeadDim + slot_row(tid)] = out;
  }
  if (!kEarlyTrigger) pdl_trigger();
}

}  // namespace q35mka

// C-linkage entries, one per instantiation the launcher uses (mka.py)
extern "C" __global__ void __launch_bounds__(q35mka::kThreads, 1) q35mka_prologue(const q35mka::ProParams p) {
  q35mka::mka_prologue_kernel<true>(p);
}
namespace q35mka {
// ---------------------------------------------------------------- slot route_lin: router before the all-reduce
// The attention layers' form of sm103/hop/src/rlin.cu (numeric): after the o_proj epilogue of a TP8 B1
// step, 16 CTAs compute this rank's router partial
// (A W_o,r * sx * sw on the gated FP8 o_proj input, plus A[:, K_r] on the layer's residual input) and push it, fp32,
// into the hop's router mailbox; the MoE input's consume_rl sums the partials into the logits. The epilogue CTAs are
// unchanged.
struct RlArgs {  // = q35rl::RlArgs (sm103/hop/src/rlin.cu)
  void* rmc;
  void* ruc;
  const uint4* rw;
  const uint16_t* h0;
  int k0;
  uint16_t* logits;
  uint8_t* x_fp4;
  uint8_t* x_sf;
  const float* gs;
  unsigned long long* trace;
};
constexpr int kRlExperts = 512, kRlCoreK = 1024, kRlSliceK = 512, kRlSteps = 12, kRlCtas = 16;
constexpr uint32_t kRlSent = 0x80000000u;
struct RlSmem {
  uint32_t xb[8][(kRlCoreK + kRlSliceK) / 2 + 4];
};
__device__ __forceinline__ void rl_mma_bf16(float (&d)[4], const uint4& a, uint32_t b0, uint32_t b1) {
  asm volatile(
      "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, "
      "{%0,%1,%2,%3};"
      : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
      : "r"(a.x), "r"(a.y), "r"(a.z), "r"(a.w), "r"(b0), "r"(b1));
}
__device__ __forceinline__ uint32_t rl_e4m3x2_bf16x2(uint16_t v) {
  float lo, hi;
  asm("{\n\t.reg .b32 h2;\n\t.reg .b16 l, h;\n\t"
      "cvt.rn.f16x2.e4m3x2 h2, %2;\n\t"
      "mov.b32 {l, h}, h2;\n\t"
      "cvt.f32.f16 %0, l;\n\t"
      "cvt.f32.f16 %1, h;\n\t}"
      : "=f"(lo), "=f"(hi)
      : "h"(v));
  __nv_bfloat162 b = __floats2bfloat162_rn(lo, hi);
  return *reinterpret_cast<uint32_t*>(&b);
}
__device__ __forceinline__ void rl_mc_st_v4f(float* mc, float a, float b, float c, float d) {
  asm volatile("multimem.st.relaxed.sys.global.v4.f32 [%0], {%1,%2,%3,%4};" ::"l"(mc), "f"(a), "f"(b), "f"(c), "f"(d)
               : "memory");
}

// Router partial rows [32c, 32c + 32); K is the gated FP8 o_proj input (1024),
// then h0[K_r] (512).
__device__ __forceinline__ uint2 rl_ld_cg2(const void* p) {
  uint2 v;
  asm volatile("ld.global.cg.v2.u32 {%0,%1}, [%2];" : "=r"(v.x), "=r"(v.y) : "l"(p) : "memory");
  return v;
}
__device__ __forceinline__ uint4 rl_ld_cg4(const void* p) {
  uint4 v;
  asm volatile("ld.global.cg.v4.u32 {%0,%1,%2,%3}, [%4];" : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w) : "l"(p) : "memory");
  return v;
}
__device__ __forceinline__ void mka_epilogue_rl_router(const EpiParams& p, const RlArgs& r, float (&red)[8][32][8],
                                                       RlSmem& sm) {
  const int c = blockIdx.x;
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
  const int g = lane >> 2, t = lane & 3;
  uint4 a[kRlSteps][2];
  {
    const uint4* wb = r.rw + ((size_t)c * 8 + warp) * (kRlSteps * 2 * 32) + lane;
#pragma unroll
    for (int st = 0; st < kRlSteps; ++st)
#pragma unroll
      for (int mt = 0; mt < 2; ++mt) a[st][mt] = ldg_nc_v4(wb + (st * 2 + mt) * 32);
  }
  const float sx = __ldg(p.sx);
  pdl_trigger();
  const float inv_sx = div_full(1.0f, sx);
  const int m = p.m;
  // stage: 8 rows x 256 quads of the gated input (attn * sigmoid(gate) -> static e4m3, the epilogue's rounding), then
  // 8 rows x 64 chunks of 8 bf16 of h0's slice; every load goes out before any is used
  uint2 ao[8], ag[8];
  uint4 hv[2];
#pragma unroll
  for (int u = 0; u < 8; ++u) {
    const int q = tid + u * 256, row = q >> 8, col = (q & 255) * 4;
    ao[u] = make_uint2(0u, 0u);
    ag[u] = make_uint2(0u, 0u);
    if (row < m) {
      ao[u] = rl_ld_cg2(p.attn + (size_t)row * kRlCoreK + col);
      ag[u] = rl_ld_cg2(p.gate + (size_t)row * kRlCoreK + col);
    }
  }
#pragma unroll
  for (int u = 0; u < 2; ++u) {
    const int q = tid + u * 256, row = q >> 6, chk = q & 63;
    hv[u] = row < m ? rl_ld_cg4(r.h0 + (size_t)row * 4096 + r.k0 + chk * 8) : make_uint4(0u, 0u, 0u, 0u);
  }
#pragma unroll
  for (int u = 0; u < 8; ++u) {
    const int q = tid + u * 256, row = q >> 8, col = (q & 255) * 4;
    const uint32_t f8 = row < m ? gate_quant4_r(ao[u], ag[u], inv_sx) : 0u;
    sm.xb[row][col / 2] = rl_e4m3x2_bf16x2((uint16_t)(f8 & 0xFFFFu));
    sm.xb[row][col / 2 + 1] = rl_e4m3x2_bf16x2((uint16_t)(f8 >> 16));
  }
#pragma unroll
  for (int u = 0; u < 2; ++u) {
    const int q = tid + u * 256, row = q >> 6, chk = q & 63;
    sm.xb[row][kRlCoreK / 2 + chk * 4 + 0] = hv[u].x;
    sm.xb[row][kRlCoreK / 2 + chk * 4 + 1] = hv[u].y;
    sm.xb[row][kRlCoreK / 2 + chk * 4 + 2] = hv[u].z;
    sm.xb[row][kRlCoreK / 2 + chk * 4 + 3] = hv[u].w;
  }
  __syncthreads();
  float ch[2][4][4];
#pragma unroll
  for (int mt = 0; mt < 2; ++mt)
#pragma unroll
    for (int k = 0; k < 4; ++k)
#pragma unroll
      for (int e = 0; e < 4; ++e) ch[mt][k][e] = 0.f;
#pragma unroll
  for (int st = 0; st < kRlSteps; ++st) {
    const int kk = (warp * kRlSteps + st) * 16;
    const uint32_t b0 = sm.xb[g][kk / 2 + t];
    const uint32_t b1 = sm.xb[g][kk / 2 + 4 + t];
    rl_mma_bf16(ch[0][st & 3], a[st][0], b0, b1);
    rl_mma_bf16(ch[1][st & 3], a[st][1], b0, b1);
  }
  {
    float4* dst = reinterpret_cast<float4*>(&red[warp][lane][0]);
#pragma unroll
    for (int mt = 0; mt < 2; ++mt) {
      float s4[4];
#pragma unroll
      for (int e = 0; e < 4; ++e)
        s4[e] = __fadd_rn(__fadd_rn(ch[mt][0][e], ch[mt][1][e]), __fadd_rn(ch[mt][2][e], ch[mt][3][e]));
      dst[mt] = make_float4(s4[0], s4[1], s4[2], s4[3]);
    }
  }
  __syncthreads();
  const int tok = tid >> 2, j = tid & 3;
  if (tok < m) {
    float v[8];
#pragma unroll
    for (int q = 0; q < 8; ++q) {
      const int rr = 8 * j + q;
      const int l = (rr % 8) * 4 + tok / 2;
      const int e = (rr / 16) * 4 + ((rr % 16) / 8) * 2 + (tok % 2);
      float x = red[0][l][e];
#pragma unroll
      for (int w = 1; w < 8; ++w) x = __fadd_rn(x, red[w][l][e]);
      v[q] = __float_as_uint(x) == kRlSent ? 0.f : x;
    }
    const q35hop::Site& s = p.site;
    const uint32_t stage = q35hop::stage_peek(s);
    float* dst = static_cast<float*>(r.rmc) +
                 (((size_t)stage * s.world + s.rank) * s.m_cap + tok) * kRlExperts + c * 32 + 8 * j;
    rl_mc_st_v4f(dst, v[0], v[1], v[2], v[3]);
    rl_mc_st_v4f(dst + 4, v[4], v[5], v[6], v[7]);
  }
}
}  // namespace q35mka

extern "C" __global__ void __launch_bounds__(q35mka::kThreads, 1) q35mka_prologue_late(const q35mka::ProParams p) {
  q35mka::mka_prologue_kernel<false>(p);
}
extern "C" __global__ void __launch_bounds__(q35mka::kThreads, 1) q35mka_gate(const q35mka::ProParams p) {
  q35mka::mka_gate_kernel(p);
}
extern "C" __global__ void __launch_bounds__(q35mka::kThreads, 1) q35mka_epilogue_q4(const q35mka::EpiParams p) {
  q35mka::mka_epilogue_kernel<4, true>(p);
}
// The router partial runs immediately after q35mka_epilogue_q4 (16 CTAs, no cluster).
extern "C" __global__ void __launch_bounds__(q35mka::kThreads, 1) q35mka_rl_router(const q35mka::EpiParams p,
                                                                               const q35mka::RlArgs r) {
  __shared__ __align__(16) float red[8][32][8];
  __shared__ __align__(16) q35mka::RlSmem sm;
  q35mka::mka_epilogue_rl_router(p, r, red, sm);
}
extern "C" __global__ void __launch_bounds__(q35mka::kThreads, 1) q35mka_epilogue_q4_late(const q35mka::EpiParams p) {
  q35mka::mka_epilogue_kernel<4, false>(p);
}
extern "C" __global__ void __launch_bounds__(q35mka::kThreads, 1) q35mka_epilogue_q8(const q35mka::EpiParams p) {
  q35mka::mka_epilogue_kernel<8, true>(p);
}
extern "C" __global__ void __launch_bounds__(q35mka::kThreads, 1) q35mka_epilogue_q8_late(const q35mka::EpiParams p) {
  q35mka::mka_epilogue_kernel<8, false>(p);
}
extern "C" __global__ void __launch_bounds__(q35mka::kThreads, 1) q35mka_prologue_wide(const q35mka::ProParams p) {
  q35mka::mka_prologue_wide_kernel<true>(p);
}
extern "C" __global__ void __launch_bounds__(q35mka::kThreads, 1) q35mka_prologue_wide_late(const q35mka::ProParams p) {
  q35mka::mka_prologue_wide_kernel<false>(p);
}
extern "C" __global__ void __launch_bounds__(q35mka::kThreads, 1) q35mka_epilogue_wide_q4(const q35mka::EpiParams p) {
  q35mka::mka_epilogue_wide_kernel<4, true>(p);
}
extern "C" __global__ void __launch_bounds__(q35mka::kThreads, 1) q35mka_epilogue_wide_q4_late(const q35mka::EpiParams p) {
  q35mka::mka_epilogue_wide_kernel<4, false>(p);
}
