// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// Slot route_lin (numeric): the MoE router's logits of a TP8 B1 verify step come
// out of the mixer's all-reduce instead of a router launch after it. RMSNorm scales its input by one number per row,
// so the router GEMV of the post-attention norm is
//   logits = W_router . (h1 * inv_rms(h1) * gamma) = inv_rms(h1) * A . (h0 + sum_r attn_r),   A = W_router diag(gamma).
// Each rank's out_proj launch adds 16 CTAs that compute its router partial (A W_out,r) . core_r + A[:, K_r] . h0[K_r]
// (the out_proj's static FP8 scales folded into the packed weights, fp32 accumulation) and push it, fp32, into a second
// mailbox with the hidden partial's stage.
//
//   q35hop_consume_rl_w8     grid (M, 4), cluster (1, 4, 1): consume_w8 + the logits + the NVFP4 input
//
// Built with NVRTC on the hop's header (q35_hop.cuh).

#include "q35_hop.cuh"

namespace q35rl {

using namespace q35hop;

// = hop.cu's Args / OprojArgs (the ctypes structures of sm103/hop/__init__.py)
struct Args {
  Site site;
  const uint16_t* partial;
  const uint16_t* routed;
  const uint16_t* expert_w;
  const int32_t* permuted;
  const uint16_t* shared;
  const uint16_t* residual_in;
  const uint16_t* gamma;
  const float* fp8_scale;
  uint16_t* residual_out;
  uint16_t* norm_out;
  uint8_t* fp8_out;
  int m, top_k;
  float eps;
};

struct OprojArgs {
  Site site;
  const uint4* w;
  const uint8_t* x;
  const float* sx;
  const float* sw;
  uint16_t* out;
  int m;
};

struct RlArgs {
  void* rmc;
  void* ruc;            // this rank's view of it
  const uint4* rw;
  const uint16_t* h0;   // [M, 4096] bf16: the layer's residual input
  int k0;               // this rank's h0 slice [k0, k0 + 512)
  uint16_t* logits;     // [M, 512] bf16 (consume)
  uint8_t* x_fp4;       // [M, 2048] (consume)
  uint8_t* x_sf;        // [M, 256] (consume)
  const float* gs;      // [1] the routed experts' NVFP4 global scale (consume)
};

constexpr int kExperts = 512;
constexpr int kHidCtas = kHidden / 32;    // 128
constexpr int kRlCtas = kExperts / 32;    // 16
static_assert(kRlCtas == 16, "the launcher's grid is 128 + 16 CTAs");
constexpr int kCoreK = 1024;              // core_r columns at TP8
constexpr int kSliceK = kHidden / 8;      // h0 columns a rank: 512
constexpr int kRlCoreSteps = kCoreK / 16 / 8;
constexpr int kRlSliceSteps = kSliceK / 16 / 8;
constexpr int kRlSteps = kRlCoreSteps + kRlSliceSteps;
constexpr int kRlFrags = kRlCtas * 8 * kRlSteps * 2 * 32;
__device__ __forceinline__ bool is_router_cta() { return blockIdx.x >= kHidCtas; }
__device__ __forceinline__ int hid_cta() { return static_cast<int>(blockIdx.x); }
__device__ __forceinline__ int rl_cta() { return static_cast<int>(blockIdx.x) - kHidCtas; }
constexpr uint32_t kSentF32 = 0x80000000u;  // f32 -0.0: an empty router slot
constexpr int kCells = kHidden / 32;
static_assert(kCells == kHidCtas && kCells * 4 == kFrags, "a cell group is one hidden CTA's rows and 4 consume frags");

__device__ __forceinline__ size_t rslot(const Site& s, uint32_t stage, int src, int token, int e) {
  return ((static_cast<size_t>(stage) * s.world + src) * s.m_cap + token) * kExperts + e;
}
__device__ __forceinline__ size_t cslot(const Site& s, uint32_t stage, int src, int token, int g) {
  return static_cast<size_t>(kStages) * s.world * s.m_cap * kExperts +
         ((static_cast<size_t>(stage) * s.world + src) * kCells + g) * static_cast<size_t>((s.m_cap + 3) & ~3) + token;
}
__device__ __forceinline__ uint32_t cell_key(float e, float h0, float v, int idx) {
  const float w = __fmul_rn(fabsf(e), __fadd_rn(fabsf(h0), fabsf(v)));
  return (__float_as_uint(w) & ~31u) | static_cast<uint32_t>(idx);
}
__device__ __forceinline__ uint32_t cell_entry(float e, int idx) {
  const uint32_t b = __float_as_uint(e) & ~31u;
  return (b == kSentF32 ? 0u : b) | static_cast<uint32_t>(idx);
}

__device__ __forceinline__ size_t row_elem(int token, int frag) {
  return static_cast<size_t>(token) * kHidden + static_cast<size_t>(frag) * kVec;
}
__device__ __forceinline__ uint4 ld_v4(const void* p) { return *static_cast<const uint4*>(p); }
__device__ __forceinline__ uint4 ldg_nc_v4(const uint4* q) {
  uint4 r;
  asm volatile("ld.global.nc.L1::no_allocate.v4.u32 {%0,%1,%2,%3}, [%4];"
               : "=r"(r.x), "=r"(r.y), "=r"(r.z), "=r"(r.w)
               : "l"(q));
  return r;
}

__device__ __forceinline__ void mma_e4m3(float (&d)[4], const uint4& a, uint32_t b0, uint32_t b1) {
  asm volatile(
      "mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, "
      "{%0,%1,%2,%3};"
      : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
      : "r"(a.x), "r"(a.y), "r"(a.z), "r"(a.w), "r"(b0), "r"(b1));
}
__device__ __forceinline__ void mma_bf16(float (&d)[4], const uint4& a, uint32_t b0, uint32_t b1) {
  asm volatile(
      "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, "
      "{%0,%1,%2,%3};"
      : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
      : "r"(a.x), "r"(a.y), "r"(a.z), "r"(a.w), "r"(b0), "r"(b1));
}
__device__ __forceinline__ void mma_f16(float (&d)[4], const uint4& a, uint32_t b0, uint32_t b1) {
  asm volatile(
      "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, "
      "{%0,%1,%2,%3};"
      : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
      : "r"(a.x), "r"(a.y), "r"(a.z), "r"(a.w), "r"(b0), "r"(b1));
}
__device__ __forceinline__ uint32_t e4m3x2_f16x2(uint16_t v) {
  uint32_t h2;
  asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(h2) : "h"(v));
  return h2;
}
__device__ __forceinline__ void mul_split_bf16x2(uint32_t h, uint32_t g, uint32_t& hi, uint32_t& lo) {
  const float p0 = __fmul_rn(bf_lo(h), bf_lo(g)), p1 = __fmul_rn(bf_hi(h), bf_hi(g));
  hi = pack_bf16x2(p0, p1);
  lo = pack_bf16x2(__fsub_rn(p0, bf_lo(hi)), __fsub_rn(p1, bf_hi(hi)));
}
__device__ __forceinline__ void mc_st_v4f(float* mc, float a, float b, float c, float d) {
  asm volatile("multimem.st.relaxed.sys.global.v4.f32 [%0], {%1,%2,%3,%4};" ::"l"(mc), "f"(a), "f"(b), "f"(c), "f"(d)
               : "memory");
}
__device__ __forceinline__ uint32_t umax32(uint32_t a, uint32_t b) { return a > b ? a : b; }
__device__ __forceinline__ float sanitize_f32(float x) { return __float_as_uint(x) == kSentF32 ? 0.f : x; }

// ---------------------------------------------------------------- out_proj + router partials

__device__ __forceinline__ void oproj_rows(const OprojArgs& o, const RlArgs& r, float (&red)[8][32][8]) {
  constexpr int S = 4;
  constexpr int kK = 8 * S * 32;
  const int cta = hid_cta();
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
  const int g = lane >> 2, t = lane & 3;
  uint4 a[S][2];
  {
    const uint4* wb = o.w + (static_cast<size_t>(cta) * 8 + warp) * (S * 2 * 32) + lane;
#pragma unroll
    for (int st = 0; st < S; ++st)
#pragma unroll
      for (int b = 0; b < 2; ++b) a[st][b] = ldg_nc_v4(wb + (st * 2 + b) * 32);
  }
  const float alpha = __fmul_rn(*o.sx, *o.sw);
  uint32_t weight_dep = __float_as_uint(alpha);
#pragma unroll
  for (int st = 0; st < S; ++st) weight_dep ^= a[st][0].x ^ a[st][1].x;
  pin_before_wait(weight_dep, reinterpret_cast<uint64_t>(o.w));
  pdl_wait();
  const int m = o.m;
  const bool tok_ok = g < m;
  const bool cell_tok = warp == 0 && (tid >> 2) < m;
  const uint4 h0v = cell_tok ? ld_v4(r.h0 + row_elem(tid >> 2, cta * 4 + (tid & 3))) : make_uint4(0u, 0u, 0u, 0u);
  const uint32_t cstage = warp == 0 ? stage_peek(o.site) : 0u;
  const uint8_t* xr = o.x + static_cast<size_t>(g) * kK + warp * S * 32 + 4 * t;
  float ch[2][2][4] = {};
#pragma unroll
  for (int st = 0; st < S; ++st) {
    const uint32_t b0 = tok_ok ? *reinterpret_cast<const uint32_t*>(xr + st * 32) : 0u;
    const uint32_t b1 = tok_ok ? *reinterpret_cast<const uint32_t*>(xr + st * 32 + 16) : 0u;
    mma_e4m3(ch[0][st & 1], a[st][0], b0, b1);
    mma_e4m3(ch[1][st & 1], a[st][1], b0, b1);
  }
  {
    float4* dst = reinterpret_cast<float4*>(&red[warp][lane][0]);
    dst[0] = make_float4(__fadd_rn(ch[0][0][0], ch[0][1][0]), __fadd_rn(ch[0][0][1], ch[0][1][1]),
                         __fadd_rn(ch[0][0][2], ch[0][1][2]), __fadd_rn(ch[0][0][3], ch[0][1][3]));
    dst[1] = make_float4(__fadd_rn(ch[1][0][0], ch[1][1][0]), __fadd_rn(ch[1][0][1], ch[1][1][1]),
                         __fadd_rn(ch[1][0][2], ch[1][1][2]), __fadd_rn(ch[1][0][3], ch[1][1][3]));
  }
  __syncthreads();
  const int tok = tid >> 2, j = tid & 3;
  uint32_t best = 0u;
  float be = 0.f;
  if (tok < m) {
    float v[8];
#pragma unroll
    for (int q = 0; q < 8; ++q) {
      const int r = 8 * j + q;
      const int l = (r % 8) * 4 + tok / 2;
      const int e = (r / 16) * 4 + ((r % 16) / 8) * 2 + (tok % 2);
      float x = red[0][l][e];
#pragma unroll
      for (int w = 1; w < 8; ++w) x = __fadd_rn(x, red[w][l][e]);
      v[q] = __fmul_rn(x, alpha);
    }
    const uint4 frag = pack8(v);
    const int fr = cta * 4 + j;
    push_frag(o.site, stage_peek(o.site), tok, fr, frag);
    *reinterpret_cast<uint4*>(o.out + row_elem(tok, fr)) = frag;
    float f[8], hz[8];
    unpack8(frag, f);
    unpack8(h0v, hz);
#pragma unroll
    for (int q = 0; q < 8; ++q) {
      const float e = __fsub_rn(v[q], f[q]);  // exact: f is v rounded to bf16
      const uint32_t k = cell_key(e, hz[q], v[q], 8 * j + q);
      if (q == 0 || k > best) {
        best = k;
        be = e;
      }
    }
  }
  if (warp == 0) {
    uint32_t gk = umax32(best, __shfl_xor_sync(0xffffffffu, best, 1));
    gk = umax32(gk, __shfl_xor_sync(0xffffffffu, gk, 2));
    uint32_t ent = best == gk ? cell_entry(be, best & 31u) : 0u;
    ent |= __shfl_xor_sync(0xffffffffu, ent, 1);
    ent |= __shfl_xor_sync(0xffffffffu, ent, 2);
    if (!cell_tok) ent = kSentF32;  // a row >= m keeps its slot empty
    float c4[4];
#pragma unroll
    for (int k = 0; k < 4; ++k) c4[k] = __uint_as_float(__shfl_sync(0xffffffffu, ent, (lane & 16) + 4 * k));
    if ((lane & 15) == 0 && (lane == 0 || m > 4))
      mc_st_v4f(static_cast<float*>(r.rmc) + cslot(o.site, cstage, o.site.rank, lane >> 2, cta), c4[0], c4[1], c4[2], c4[3]);
  }
  pdl_trigger();
}

struct RlSmem {
  uint32_t xb[8][kCoreK / 2 + kSliceK + 4];
};
__device__ __forceinline__ void router_rows(const OprojArgs& o, const RlArgs& r, float (&red)[8][32][8], RlSmem& sm) {
  const int c = rl_cta();
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
  const int g = lane >> 2, t = lane & 3;
  uint4 a[kRlSteps][2];
  {
    const uint4* wb = r.rw + (static_cast<size_t>(c) * 8 + warp) * (kRlSteps * 2 * 32) + lane;
#pragma unroll
    for (int st = 0; st < kRlSteps; ++st)
#pragma unroll
      for (int mt = 0; mt < 2; ++mt) a[st][mt] = ldg_nc_v4(wb + (st * 2 + mt) * 32);
  }
  const uint4 g2s = ldg_nc_v4(r.rw + kRlFrags + (tid & 63));
  const float unscale = __uint_as_float(ldg_nc_v4(r.rw + kRlFrags + kSliceK / 8).x);
  uint32_t weight_dep = g2s.x ^ __float_as_uint(unscale);
#pragma unroll
  for (int st = 0; st < kRlSteps; ++st)
    weight_dep ^= a[st][0].x ^ a[st][1].x;
  pin_before_wait(weight_dep, reinterpret_cast<uint64_t>(r.rw));
  pdl_wait();
  const int m = o.m;
  uint4 ld[4];
#pragma unroll
  for (int u = 0; u < 4; ++u) {
    const int i = tid + (u & 1) * 256, row = i >> 6, chk = i & 63;
    ld[u] = make_uint4(0u, 0u, 0u, 0u);
    if (row < m) {
      ld[u] = u < 2 ? *reinterpret_cast<const uint4*>(o.x + static_cast<size_t>(row) * kCoreK + chk * 16)
                    : *reinterpret_cast<const uint4*>(r.h0 + static_cast<size_t>(row) * kHidden + r.k0 + chk * 8);
    }
  }
#pragma unroll
  for (int u = 0; u < 4; ++u) {
    const int i = tid + (u & 1) * 256, row = i >> 6, chk = i & 63;
    const uint32_t vw[4] = {ld[u].x, ld[u].y, ld[u].z, ld[u].w};
    if (u < 2) {
#pragma unroll
      for (int q = 0; q < 4; ++q) {
        sm.xb[row][chk * 8 + 2 * q] = e4m3x2_f16x2(static_cast<uint16_t>(vw[q] & 0xFFFFu));
        sm.xb[row][chk * 8 + 2 * q + 1] = e4m3x2_f16x2(static_cast<uint16_t>(vw[q] >> 16));
      }
    } else {
      const uint32_t gw[4] = {g2s.x, g2s.y, g2s.z, g2s.w};
#pragma unroll
      for (int q = 0; q < 4; ++q)
        mul_split_bf16x2(vw[q], gw[q], sm.xb[row][kCoreK / 2 + chk * 4 + q],
                         sm.xb[row][kCoreK / 2 + kSliceK / 2 + chk * 4 + q]);
    }
  }
  __syncthreads();
  float ch[2][4][4] = {};
#pragma unroll
  for (int st = 0; st < kRlCoreSteps; ++st) {
    const int kw = (warp * kRlCoreSteps + st) * 8;
    const uint32_t b0 = sm.xb[g][kw + t];
    const uint32_t b1 = sm.xb[g][kw + 4 + t];
    mma_f16(ch[0][st & 3], a[st][0], b0, b1);
    mma_f16(ch[1][st & 3], a[st][1], b0, b1);
  }
#pragma unroll
  for (int st = 0; st < kRlSliceSteps; ++st) {
    const int kw = kCoreK / 2 + (warp * kRlSliceSteps + st) * 8;
    const uint32_t hi0 = sm.xb[g][kw + t], hi1 = sm.xb[g][kw + 4 + t];
    const uint32_t lo0 = sm.xb[g][kw + kSliceK / 2 + t], lo1 = sm.xb[g][kw + kSliceK / 2 + 4 + t];
    mma_bf16(ch[0][st & 3], a[kRlCoreSteps + st][0], hi0, hi1);
    mma_bf16(ch[1][st & 3], a[kRlCoreSteps + st][1], hi0, hi1);
    mma_bf16(ch[0][(st + 2) & 3], a[kRlCoreSteps + st][0], lo0, lo1);
    mma_bf16(ch[1][(st + 2) & 3], a[kRlCoreSteps + st][1], lo0, lo1);
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
      v[q] = sanitize_f32(__fmul_rn(x, unscale));
    }
    float* dst = static_cast<float*>(r.rmc) + rslot(o.site, stage_peek(o.site), o.site.rank, tok, c * 32 + 8 * j);
    mc_st_v4f(dst, v[0], v[1], v[2], v[3]);
    mc_st_v4f(dst + 4, v[4], v[5], v[6], v[7]);
  }
  pdl_trigger();
}

// ---------------------------------------------------------------- consume + logits + NVFP4 input

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

struct RowSmem2 {
  float2 xs[kRowCtas][kRowWarps];
  uint64_t mbar;
};
__device__ __forceinline__ void row_init2(RowSmem2& rs) {
  if (threadIdx.x == 0) {
    const uint32_t a = static_cast<uint32_t>(__cvta_generic_to_shared(&rs.mbar));
    asm volatile("mbarrier.init.shared::cta.b64 [%0], 1;" ::"r"(a) : "memory");
    asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;" ::"r"(a),
                 "r"(static_cast<uint32_t>(sizeof(rs.xs)))
                 : "memory");
    asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
  }
  asm volatile("barrier.cluster.arrive.relaxed.aligned;\n\tbarrier.cluster.wait.acquire.aligned;" ::: "memory");
}
__device__ __forceinline__ float2 row_sumsq_fold2(float2 v, RowSmem2& rs) {
  const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
#pragma unroll
  for (int off = 1; off < 32; off <<= 1) {
    v.x = __fadd_rn(v.x, __shfl_xor_sync(0xffffffffu, v.x, off));
    v.y = __fadd_rn(v.y, __shfl_xor_sync(0xffffffffu, v.y, off));
  }
  if (lane < kRowCtas) {
    const uint32_t me = cluster_ctarank();
    const uint32_t slot_l = static_cast<uint32_t>(__cvta_generic_to_shared(&rs.xs[me][warp]));
    const uint32_t mbar_l = static_cast<uint32_t>(__cvta_generic_to_shared(&rs.mbar));
    uint32_t slot_r, mbar_r;
    asm volatile("mapa.shared::cluster.u32 %0, %1, %2;" : "=r"(slot_r) : "r"(slot_l), "r"(lane));
    asm volatile("mapa.shared::cluster.u32 %0, %1, %2;" : "=r"(mbar_r) : "r"(mbar_l), "r"(lane));
    asm volatile("st.async.shared::cluster.mbarrier::complete_tx::bytes.v2.f32 [%0], {%1, %2}, [%3];" ::"r"(slot_r),
                 "f"(v.x), "f"(v.y), "r"(mbar_r)
                 : "memory");
  }
  {
    const uint32_t a = static_cast<uint32_t>(__cvta_generic_to_shared(&rs.mbar));
    asm volatile(
        "{\n\t.reg .pred P;\n\tWAIT_ROW2:\n\t"
        "mbarrier.try_wait.parity.acquire.cluster.shared::cta.b64 P, [%0], 0;\n\t"
        "@!P bra WAIT_ROW2;\n\t}" ::"r"(a)
        : "memory");
  }
  float2 total = make_float2(0.f, 0.f);
#pragma unroll
  for (int c = 0; c < kRowCtas; ++c) {
    const float2 w0 = rs.xs[c][0], w1 = rs.xs[c][1], w2 = rs.xs[c][2], w3 = rs.xs[c][3];
    total.x = __fadd_rn(total.x, __fadd_rn(__fadd_rn(w0.x, w2.x), __fadd_rn(w1.x, w3.x)));
    total.y = __fadd_rn(total.y, __fadd_rn(__fadd_rn(w0.y, w2.y), __fadd_rn(w1.y, w3.y)));
  }
  return total;
}

template <int W>
__device__ __forceinline__ void pull_both(const RlArgs& r, const Site& s, uint32_t stage, int token, int frag,
                                          float acc[8], float& lsum, uint32_t (&cell)[2]) {
  static_assert(W == 8, "a lane takes the cells of ranks j and j + 4");
  const uint16_t* uc = static_cast<const uint16_t*>(s.uc);
  const uint32_t* ru = static_cast<const uint32_t*>(r.ruc);
  const int cg = frag >> 2, cj = frag & 3;
  uint4 v[W];
  uint32_t u[W];
#pragma unroll
  for (int k = 0; k < W; ++k) v[k] = ld_volatile_v4(uc + slot(s, stage, k, token, frag));
#pragma unroll
  for (int k = 0; k < W; ++k) u[k] = ld_volatile_u32(ru + rslot(s, stage, k, token, frag));
#pragma unroll
  for (int i = 0; i < 2; ++i) cell[i] = ld_volatile_u32(ru + cslot(s, stage, cj + 4 * i, token, cg));
  uint32_t pending = 0, rpend = 0, cpend = 0;
#pragma unroll
  for (int k = 0; k < W; ++k) pending |= frag_has_negzero(v[k]) ? (1u << k) : 0u;
#pragma unroll
  for (int k = 0; k < W; ++k) rpend |= u[k] == kSentF32 ? (1u << k) : 0u;
#pragma unroll
  for (int i = 0; i < 2; ++i) cpend |= cell[i] == kSentF32 ? (1u << i) : 0u;
  unsigned long long t0 = 0;
  uint32_t spins = 0;
  while (pending | rpend | cpend) {
#pragma unroll
    for (int k = 0; k < W; ++k)
      if (pending & (1u << k)) v[k] = ld_volatile_v4(uc + slot(s, stage, k, token, frag));
#pragma unroll
    for (int k = 0; k < W; ++k)
      if (rpend & (1u << k)) u[k] = ld_volatile_u32(ru + rslot(s, stage, k, token, frag));
#pragma unroll
    for (int i = 0; i < 2; ++i)
      if (cpend & (1u << i)) cell[i] = ld_volatile_u32(ru + cslot(s, stage, cj + 4 * i, token, cg));
    uint32_t still = 0, rstill = 0, cstill = 0;
#pragma unroll
    for (int k = 0; k < W; ++k)
      if ((pending & (1u << k)) && frag_has_negzero(v[k])) still |= 1u << k;
#pragma unroll
    for (int k = 0; k < W; ++k)
      if ((rpend & (1u << k)) && u[k] == kSentF32) rstill |= 1u << k;
#pragma unroll
    for (int i = 0; i < 2; ++i)
      if ((cpend & (1u << i)) && cell[i] == kSentF32) cstill |= 1u << i;
    pending = still;
    rpend = rstill;
    cpend = cstill;
    if (Q35_HOP_DEADLINE_NS != 0 && (pending | rpend | cpend) && ((++spins & 255u) == 0)) {
      const unsigned long long now = gtimer();
      if (t0 == 0) {
        t0 = now;
      } else if (now - t0 > Q35_HOP_DEADLINE_NS) {
        fault_at_null();
      }
    }
  }
#pragma unroll
  for (int e = 0; e < 8; ++e) acc[e] = 0.f;
#pragma unroll
  for (int k = 0; k < W; ++k) {
    float f[8];
    unpack8(v[k], f);
#pragma unroll
    for (int e = 0; e < 8; ++e) acc[e] = __fadd_rn(acc[e], f[e]);
  }
  lsum = 0.f;
#pragma unroll
  for (int k = 0; k < W; ++k) lsum = __fadd_rn(lsum, __uint_as_float(u[k]));
}

template <int W>
__device__ __forceinline__ void consume_rl_kernel(const Args& a, const RlArgs& r) {
  __shared__ RowSmem2 rs;
  __shared__ __align__(16) float pu_s[kRowThreads * 8];
  const int token = blockIdx.x;
  const int frag = blockIdx.y * kRowThreads + threadIdx.x;
  pdl_trigger();
  float gam[8];
  unpack8(ld_v4(a.gamma + static_cast<size_t>(frag) * kVec), gam);
  const float inv_s = a.fp8_out != nullptr ? inv_scale_of(*a.fp8_scale) : 1.f;
  const float gs = *r.gs;
  row_init2(rs);
  pdl_wait();
  const Hop h = hop_enter(a.site);
  const uint4 res = ld_v4(a.residual_in + row_elem(token, frag));
  float acc[8], lsum = 0.f;
  uint32_t cell[2];
  pull_both<W>(r, a.site, h.stage, token, frag, acc, lsum, cell);
  float pre_u[8];
  {
    float rf[8];
    unpack8(res, rf);
#pragma unroll
    for (int e = 0; e < 8; ++e) pre_u[e] = __fadd_rn(acc[e], rf[e]);  // = residual_add's sums, before the bf16 rounding
  }
  {
    float4* pw = reinterpret_cast<float4*>(&pu_s[threadIdx.x * 8]);
    pw[0] = make_float4(pre_u[0], pre_u[1], pre_u[2], pre_u[3]);
    pw[1] = make_float4(pre_u[4], pre_u[5], pre_u[6], pre_u[7]);
  }
  __syncwarp();
  float corr = 0.f;  // (h + e)^2 - h^2 of the two cells, rank frag & 3 first
#pragma unroll
  for (int i = 0; i < 2; ++i) {
    const float e = __uint_as_float(cell[i] & ~31u);
    const float hc = pu_s[(threadIdx.x & ~3) * 8 + (cell[i] & 31u)];
    corr = __fadd_rn(corr, __fmul_rn(e, __fadd_rn(__fadd_rn(hc, hc), e)));
  }
  float pre[8];
  const uint4 res_out = residual_add(acc, res, pre);
  const float2 total = row_sumsq_fold2(make_float2(sumsq8(pre), __fadd_rn(sumsq8(pre_u), corr)), rs);
  const float inv = inv_rms(total.x, a.eps);
  const float inv_u = inv_rms(total.y, a.eps);
  const uint4 norm = norm_apply(pre, inv, gam);
  *reinterpret_cast<uint4*>(a.residual_out + row_elem(token, frag)) = res_out;
  if (a.norm_out != nullptr) *reinterpret_cast<uint4*>(a.norm_out + row_elem(token, frag)) = norm;
  if (a.fp8_out != nullptr) *reinterpret_cast<uint2*>(a.fp8_out + row_elem(token, frag)) = quant_fp8(norm, inv_s);
  r.logits[static_cast<size_t>(token) * kExperts + frag] = static_cast<uint16_t>(pack_bf16x2(__fmul_rn(lsum, inv_u), 0.f));
  {
    const uint32_t w4[4] = {norm.x, norm.y, norm.z, norm.w};
    const uint32_t mine = bf16x2_max(bf16x2_abs_max(w4[0], w4[1]), bf16x2_abs_max(w4[2], w4[3]));
    const uint32_t peer = __shfl_xor_sync(0xffffffffu, mine, 1);
    const uint32_t m = (frag & 1) ? bf16x2_max(peer, mine) : bf16x2_max(mine, peer);
    const float amax = fmaxf(bf_lo(m), bf_hi(m));
    const float scale = __fmul_rn(gs, __fmul_rn(amax, rcp_approx_ftz(6.0f)));
    const uint32_t sf = cvt_e4m3(scale);
    const float sf_f = e4m3_to_f32(sf);
    const float out_scale = (sf_f == 0.0f) ? 0.0f : rcp_approx_ftz(__fmul_rn(sf_f, rcp_approx_ftz(gs)));
    float q[8];
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      q[2 * i] = __fmul_rn(bf_lo(w4[i]), out_scale);
      q[2 * i + 1] = __fmul_rn(bf_hi(w4[i]), out_scale);
    }
    const int blk = frag >> 1;
    *reinterpret_cast<uint32_t*>(r.x_fp4 + static_cast<size_t>(token) * (kHidden / 2) + blk * 8 + (frag & 1) * 4) =
        cvt_e2m1x8(q);
    if (!(frag & 1)) r.x_sf[static_cast<size_t>(token) * (kHidden / 16) + blk] = static_cast<uint8_t>(sf);
  }
  rearm_frag<W>(a.site, h.stage, token, frag);
  uint32_t* uc = static_cast<uint32_t*>(r.ruc);
#pragma unroll
  for (int k = 0; k < W; ++k) uc[rslot(a.site, h.stage, k, token, frag)] = kSentF32;
#pragma unroll
  for (int i = 0; i < 2; ++i) uc[cslot(a.site, h.stage, (frag & 3) + 4 * i, token, frag >> 2)] = kSentF32;
  hop_leave(a.site, h, gridDim.x * gridDim.y);
}

}  // namespace q35rl

extern "C" __global__ void __launch_bounds__(256, 1) q35hop_oproj_push_rl_s4(q35rl::OprojArgs o, q35rl::RlArgs r) {
  __shared__ __align__(16) float red[8][32][8];
  __shared__ __align__(16) q35rl::RlSmem sm;
  if (!q35rl::is_router_cta()) {
    q35rl::oproj_rows(o, r, red);
  } else {
    q35rl::router_rows(o, r, red, sm);
  }
}

extern "C" __global__ void __launch_bounds__(q35hop::kRowThreads) q35hop_consume_rl_w8(q35rl::Args a, q35rl::RlArgs r) {
  q35rl::consume_rl_kernel<8>(a, r);
}
