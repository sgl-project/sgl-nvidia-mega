// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// Low-M projections of a Qwen3.5 verify step on warp MMA with register-prefetched weights: C = A @ W^T for
// M_ROWS (<= 64) rows. The work layout follows TensorRT-LLM's lowm_mma_gemm (Qwen3.5 decode, Apache-2.0):
// a CTA owns NT groups of 8 weight rows and splits K over its KW warps; each lane holds 16 contiguous K
// columns of a 64-wide chunk for its weight row and its activation rows, which are the MMA fragments once
// the reduction order inside the chunk is permuted the same way for both operands; the warps reduce through
// shared memory in warp order.
//
// Two weight segments, each a whole number of CTAs: blocks [0, CTAS8) take the FP8 segment (E4M3 weights,
// per-tensor weight scale), blocks [CTAS8, CTAS8 + CTAS16) the BF16 segment (BF16 activations).
// ACT 0: the FP8 segment reads E4M3 activations quantized by the producer; ACT 1: BF16 activations quantized
// in registers with the static input scale, with the arithmetic of SGLang's static quantize kernel
// (1 / s by div.full, multiply, clamp to +-448, cvt.rn.satfinite: the same E4M3 values).
// EPI8 0: BF16 out = acc * (s_in * s_w); 1: SwiGLU on (8 gate rows, 8 up rows) tile pairs of BF16-rounded
// outputs, written as E4M3 with the static scale s_out; 2: BF16 out times sigmoid(gate[row]) (the shared
// expert's gate, FP32 logits).
// PDL: weights and scales load before griddepcontrol.wait, activations after it; TRIGGER 1 releases the
// dependent launch once the weight loads are issued (a dependent waits for this grid before reading).

#include <cuda_bf16.h>
#ifndef PUSH
#define PUSH 0
#endif
#if PUSH
#include "q35_hop.cuh"
#endif

typedef unsigned char uint8_t;
typedef unsigned short uint16_t;
typedef unsigned int uint32_t;
typedef long long int64_t;

#ifndef M_ROWS
#error "M_ROWS"
#endif
#ifndef KSPLIT
#define KSPLIT 1
#endif
#define MT ((M_ROWS + 15) / 16)
#define CHUNKS (K_DIM / 64)
#define CPW (CHUNKS / (KW * KSPLIT))
#define NTMAX (NT8 > NT16 ? NT8 : NT16)
#define NC (NTMAX * 8)

struct Params {
  const uint8_t* x8;          // [M, K] E4M3 (ACT 0)
  int64_t ld8;
  const __nv_bfloat16* x16;   // [M, K] BF16 (ACT 1, BF16 segment)
  int64_t ld16;
  const uint8_t* w8;          // [N8, K] E4M3, row-major
  const float* s_w;
  const float* s_in;
  const float* s_out;         // EPI8 1
  const __nv_bfloat16* w16;   // [N16, K] BF16, row-major
  void* out;                  // FP8 segment: BF16 [M, N8] (EPI8 0, 2) or E4M3 [M, N8 / 2] (EPI8 1)
  int64_t ldo;
  void* out2;                 // BF16 segment: BF16 or FP32 [M, N16]
  int64_t ldo2;
  const float* gate;          // EPI8 2: FP32 logits, row stride ldg
  int64_t ldg;
  // PUSH: the decode hop's mailbox site (q35hop::Site); the BF16 output rows are also pushed into it
  void* hop_mc;
  void* hop_uc;
  void* hop_ctrl;
  int hop_rank, hop_world, hop_m_cap;
};

__device__ __forceinline__ void pdl_wait() { asm volatile("griddepcontrol.wait;" ::: "memory"); }
__device__ __forceinline__ void pdl_launch() { asm volatile("griddepcontrol.launch_dependents;" ::: "memory"); }

#ifndef WLD
#define WLD 2
#endif
#ifndef ALD
#define ALD 1
#endif
#ifndef QCLAMP
#define QCLAMP 0
#endif
__device__ __forceinline__ uint4 ld_weight(const void* p) {
  uint4 v;
#if WLD == 0
  asm volatile("ld.global.nc.L1::no_allocate.v4.u32 {%0,%1,%2,%3}, [%4];"
               : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w) : "l"(p));
#elif WLD == 1
  asm volatile("ld.global.cs.v4.u32 {%0,%1,%2,%3}, [%4];" : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w) : "l"(p));
#else
  asm volatile("ld.global.L1::no_allocate.v4.u32 {%0,%1,%2,%3}, [%4];"
               : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w) : "l"(p));
#endif
  return v;
}
// activations: not volatile, so the compiler may schedule them among the MMAs; their pointers come from
// launder() after the dependency wait, which keeps every such load behind it
__device__ __forceinline__ uint4 ld_act(const void* p) {
  uint4 v;
#if ALD == 0
  asm("ld.global.cg.v4.u32 {%0,%1,%2,%3}, [%4];" : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w) : "l"(p));
#else
  asm("ld.global.ca.v4.u32 {%0,%1,%2,%3}, [%4];" : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w) : "l"(p));
#endif
  return v;
}
template <class T>
__device__ __forceinline__ T* launder(T* p) {
  asm volatile("" : "+l"(p));
  return p;
}

__device__ __forceinline__ void mma_e4m3(float (&c)[4], uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3,
                                         uint32_t b0, uint32_t b1) {
  asm(
      "mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, "
      "{%0,%1,%2,%3};"
      : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
      : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}
__device__ __forceinline__ void mma_bf16(float (&c)[4], uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3,
                                         uint32_t b0, uint32_t b1) {
  asm(
      "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, "
      "{%0,%1,%2,%3};"
      : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
      : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}

// SGLang's static quantize (Triton): y * (1.0 / s) (div.full.f32), clamp to [-448, 448], E4M3 rn satfinite
__device__ __forceinline__ float inverse(float s) {
  float r;
  asm("div.full.f32 %0, %1, %2;" : "=f"(r) : "f"(1.0f), "f"(s));
  return r;
}
__device__ __forceinline__ uint16_t e4m3x2(float lo, float hi) {
  uint16_t r;
  asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(r) : "f"(hi), "f"(lo));
  return r;
}
// the clamp of SGLang's kernel; satfinite alone gives the same E4M3 for every non-NaN input
#if QCLAMP
__device__ __forceinline__ float qclamp(float x) { return fminf(fmaxf(x, -448.f), 448.f); }
#else
__device__ __forceinline__ float qclamp(float x) { return x; }
#endif
// two bf16x2 words (element 0 in the low half) -> four E4M3, element 0 in the low byte
__device__ __forceinline__ uint32_t quant4(uint32_t w01, uint32_t w23, float inv) {
  const float f0 = qclamp(__uint_as_float(w01 << 16) * inv);
  const float f1 = qclamp(__uint_as_float(w01 & 0xffff0000u) * inv);
  const float f2 = qclamp(__uint_as_float(w23 << 16) * inv);
  const float f3 = qclamp(__uint_as_float(w23 & 0xffff0000u) * inv);
  return uint32_t(e4m3x2(f0, f1)) | (uint32_t(e4m3x2(f2, f3)) << 16);
}

// 16 activation columns of a row as E4M3 (zero beyond M_ROWS)
// rows past M_ROWS read the last row: no branch around the loads; their sums are never stored
__device__ __forceinline__ uint4 act8(const uint8_t* x8, int64_t ld8, const __nv_bfloat16* x16, int64_t ld16,
                                      int row, int col, float inv) {
  row = row < M_ROWS ? row : M_ROWS - 1;
#if ACT == 0
  return ld_act(x8 + row * ld8 + col);
#else
  const __nv_bfloat16* src = x16 + row * ld16 + col;
  const uint4 lo = ld_act(src), hi = ld_act(src + 8);
  return make_uint4(quant4(lo.x, lo.y, inv), quant4(lo.z, lo.w, inv), quant4(hi.x, hi.y, inv),
                    quant4(hi.z, hi.w, inv));
#endif
}

__device__ __forceinline__ float bf16r(float x) { return __bfloat162float(__float2bfloat16_rn(x)); }

__device__ __forceinline__ float SUM(const float* red, int row, int col) {
  float s = 0.f;
#pragma unroll
  for (int w = 0; w < KW; ++w) s += red[(w * MT * 16 + row) * NC + col];
  return s;
}

#if KSPLIT > 1
__device__ __forceinline__ void cluster_sync() {
  asm volatile("barrier.cluster.arrive.release.aligned;\n\tbarrier.cluster.wait.acquire.aligned;" ::: "memory");
}
__device__ __forceinline__ unsigned cluster_rank() {
  unsigned r;
  asm volatile("mov.u32 %0, %%cluster_ctarank;" : "=r"(r));
  return r;
}
// the cluster's sum of an output: the KSPLIT CTAs' partial sums in rank order (distributed shared memory)
__device__ __forceinline__ float TOTAL(const float* part, int row, int col) {
  const unsigned a = static_cast<unsigned>(__cvta_generic_to_shared(part + row * NC + col));
  float s = 0.f;
#pragma unroll
  for (int r = 0; r < KSPLIT; ++r) {
    unsigned ra;
    float v;
    asm volatile("mapa.shared::cluster.u32 %0, %1, %2;" : "=r"(ra) : "r"(a), "r"(r));
    asm volatile("ld.shared::cluster.f32 %0, [%1];" : "=f"(v) : "r"(ra) : "memory");
    s += v;
  }
  return s;
}
#define T_SUM(row, col) TOTAL(part, row, col)
#else
#define T_SUM(row, col) SUM(red, row, col)
#endif

extern "C" __global__ void __launch_bounds__(KW * 32, MIN_BLOCKS) q35_lowm(const Params p) {
  extern __shared__ float red[];  // [KW][MT * 16][NC], then (KSPLIT > 1) the CTA's sums [M_ROWS][NC]
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31, g = lane >> 2, q = lane & 3;
#if KSPLIT > 1
  const int rank = cluster_rank();
  float* part = red + KW * MT * 16 * NC;
#else
  const int rank = 0;
#endif
  const int group = blockIdx.x / KSPLIT;  // the CTA's weight-row group; rank: its K slice
  const int k0 = (rank * KW + warp) * CPW * 64 + 16 * q;
  float acc[MT][NTMAX][4];
#pragma unroll
  for (int t = 0; t < MT; ++t)
#pragma unroll
    for (int n = 0; n < NTMAX; ++n)
#pragma unroll
      for (int i = 0; i < 4; ++i) acc[t][n][i] = 0.f;
  const bool fp8 = blockIdx.x < CTAS8;
  int ntiles;
  if (fp8) {
#if CTAS8 > 0
    ntiles = NT8;
    const int n0 = group * NT8 * 8;
    uint4 w[CPW][NT8];
#pragma unroll
    for (int c = 0; c < CPW; ++c)
#pragma unroll
      for (int n = 0; n < NT8; ++n) w[c][n] = ld_weight(p.w8 + (int64_t)(n0 + 8 * n + g) * K_DIM + k0 + 64 * c);
#if ACT == 1
    const float s_in = *p.s_in;
#endif
#if TRIGGER
    pdl_launch();
#endif
    pdl_wait();
    const uint8_t* x8 = launder(p.x8);
    const __nv_bfloat16* x16 = launder(p.x16);
#if ACT == 1
    const float inv = inverse(s_in);
#else
    const float inv = 0.f;
#endif
#pragma unroll
    for (int c = 0; c < CPW; ++c) {
      uint4 a[MT][2];
#pragma unroll
      for (int t = 0; t < MT; ++t) {
        a[t][0] = act8(x8, p.ld8, x16, p.ld16, 16 * t + g, k0 + 64 * c, inv);
        a[t][1] = (16 * t + 8 < M_ROWS) ? act8(x8, p.ld8, x16, p.ld16, 16 * t + g + 8, k0 + 64 * c, inv)
                                        : make_uint4(0, 0, 0, 0);
      }
#pragma unroll
      for (int t = 0; t < MT; ++t)
#pragma unroll
        for (int n = 0; n < NT8; ++n) {
          mma_e4m3(acc[t][n], a[t][0].x, a[t][1].x, a[t][0].y, a[t][1].y, w[c][n].x, w[c][n].y);
          mma_e4m3(acc[t][n], a[t][0].z, a[t][1].z, a[t][0].w, a[t][1].w, w[c][n].z, w[c][n].w);
        }
    }
#endif
  } else {
#if CTAS16 > 0
    ntiles = NT16;
    const int n0 = (group - CTAS8 / KSPLIT) * NT16 * 8;
    uint4 w[CPW][NT16][2];
#pragma unroll
    for (int c = 0; c < CPW; ++c)
#pragma unroll
      for (int n = 0; n < NT16; ++n) {
        const __nv_bfloat16* src = p.w16 + (int64_t)(n0 + 8 * n + g) * K_DIM + k0 + 64 * c;
        w[c][n][0] = ld_weight(src);
        w[c][n][1] = ld_weight(src + 8);
      }
#if TRIGGER
    pdl_launch();
#endif
    pdl_wait();
    const __nv_bfloat16* x16 = launder(p.x16);
#pragma unroll
    for (int c = 0; c < CPW; ++c) {
      uint4 a[MT][2][2];
#pragma unroll
      for (int t = 0; t < MT; ++t)
#pragma unroll
        for (int h = 0; h < 2; ++h) {
          if (16 * t + 8 * h < M_ROWS) {
            int row = 16 * t + g + 8 * h;
            row = row < M_ROWS ? row : M_ROWS - 1;
            const __nv_bfloat16* src = x16 + row * p.ld16 + k0 + 64 * c;
            a[t][h][0] = ld_act(src);
            a[t][h][1] = ld_act(src + 8);
          } else {
            a[t][h][0] = a[t][h][1] = make_uint4(0, 0, 0, 0);
          }
        }
#pragma unroll
      for (int t = 0; t < MT; ++t)
#pragma unroll
        for (int n = 0; n < NT16; ++n) {
          // step j takes words 2j / 2j + 1 of the lane's 16 BF16 columns (logical k 2q+i / 2q+8+i)
          mma_bf16(acc[t][n], a[t][0][0].x, a[t][1][0].x, a[t][0][0].y, a[t][1][0].y, w[c][n][0].x, w[c][n][0].y);
          mma_bf16(acc[t][n], a[t][0][0].z, a[t][1][0].z, a[t][0][0].w, a[t][1][0].w, w[c][n][0].z, w[c][n][0].w);
          mma_bf16(acc[t][n], a[t][0][1].x, a[t][1][1].x, a[t][0][1].y, a[t][1][1].y, w[c][n][1].x, w[c][n][1].y);
          mma_bf16(acc[t][n], a[t][0][1].z, a[t][1][1].z, a[t][0][1].w, a[t][1][1].w, w[c][n][1].z, w[c][n][1].w);
        }
    }
#endif
  }
  // reduce the KW warps' partial sums in warp order
#pragma unroll
  for (int t = 0; t < MT; ++t)
#pragma unroll
    for (int n = 0; n < NTMAX; ++n) {
      if (n < ntiles) {
        float* dst = red + ((warp * MT * 16) + 16 * t + g) * NC + 8 * n + 2 * q;
        dst[0] = acc[t][n][0];
        dst[1] = acc[t][n][1];
        dst[8 * NC] = acc[t][n][2];
        dst[8 * NC + 1] = acc[t][n][3];
      }
    }
  __syncthreads();
#if KSPLIT > 1
  for (int i = threadIdx.x; i < M_ROWS * ntiles * 8; i += KW * 32) {
    const int row = i / (ntiles * 8), col = i % (ntiles * 8);
    part[row * NC + col] = SUM(red, row, col);
  }
  cluster_sync();
#define OUT_START (rank * KW * 32 + threadIdx.x)
#define OUT_STEP (KW * 32 * KSPLIT)
#else
#define OUT_START (threadIdx.x)
#define OUT_STEP (KW * 32)
#endif
  if (fp8) {
#if CTAS8 > 0
    const float alpha = *p.s_in * *p.s_w;
#if EPI8 == 1
    // SwiGLU: tile 2j = gate rows, tile 2j + 1 = the matching up rows; NT8 * 4 outputs a CTA
    const float inv_out = inverse(*p.s_out);
    uint8_t* out = reinterpret_cast<uint8_t*>(p.out);
    for (int i = OUT_START; i < M_ROWS * NT8 * 4; i += OUT_STEP) {
      const int row = i / (NT8 * 4), col = i % (NT8 * 4), pair = col >> 3, c8 = col & 7;
      const float gt = bf16r(T_SUM(row, 16 * pair + c8) * alpha);
      const float up = bf16r(T_SUM(row, 16 * pair + 8 + c8) * alpha);
      const float h = bf16r(gt / (1.f + __expf(-gt)) * up);
      const uint16_t v = e4m3x2(qclamp(h * inv_out), 0.f);
      out[row * p.ldo + group * NT8 * 4 + col] = uint8_t(v & 0xff);
    }
#elif PUSH
    // the decode hop's push (its all-reduce then only consumes): one 8-column fragment a thread, pushed
    // before it is stored
    const q35hop::Site site{p.hop_mc, p.hop_uc, static_cast<q35hop::Ctrl*>(p.hop_ctrl), p.hop_rank, p.hop_world,
                            p.hop_m_cap};
    const uint32_t stage = q35hop::stage_peek(site);
    for (int i = OUT_START; i < M_ROWS * NT8; i += OUT_STEP) {
      const int row = i / NT8, f = i % NT8;
      float v[8];
#pragma unroll
      for (int c = 0; c < 8; ++c) v[c] = T_SUM(row, 8 * f + c) * alpha;
      const uint4 frag = q35hop::pack8(v);
      q35hop::push_frag(site, stage, row, group * NT8 + f, frag);
      *reinterpret_cast<uint4*>(reinterpret_cast<__nv_bfloat16*>(p.out) + row * p.ldo + group * NT8 * 8 + 8 * f) = frag;
    }
#else
    __nv_bfloat16* out = reinterpret_cast<__nv_bfloat16*>(p.out);
    for (int i = OUT_START; i < M_ROWS * NT8 * 8; i += OUT_STEP) {
      const int row = i / (NT8 * 8), col = i % (NT8 * 8);
      float v = T_SUM(row, col) * alpha;
#if EPI8 == 2
      const float gl = p.gate[row * p.ldg];
      v = bf16r(v) * (1.f / (1.f + __expf(-gl)));
#endif
      out[row * p.ldo + group * NT8 * 8 + col] = __float2bfloat16_rn(v);
    }
#endif
#endif
  } else {
#if CTAS16 > 0
    for (int i = OUT_START; i < M_ROWS * NT16 * 8; i += OUT_STEP) {
      const int row = i / (NT16 * 8), col = i % (NT16 * 8);
      const float v = T_SUM(row, col);
      const int col0 = (group - CTAS8 / KSPLIT) * NT16 * 8;
#if OUT2_F32
      reinterpret_cast<float*>(p.out2)[row * p.ldo2 + col0 + col] = v;
#else
      reinterpret_cast<__nv_bfloat16*>(p.out2)[row * p.ldo2 + col0 + col] = __float2bfloat16_rn(v);
#endif
    }
#endif
  }
#if KSPLIT > 1
  cluster_sync();  // the peers' partial sums stay until every rank has read them
#endif
}
