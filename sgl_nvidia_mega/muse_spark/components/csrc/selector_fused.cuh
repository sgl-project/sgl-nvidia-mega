// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
//
// DFlash2 candidate selector: edge scores and the sampling walk in one launch (MUSE_VR_SELECTOR_FUSED, default off).
//
// The stock selector builds the whole lattice score[b, e, p, c] = unary[b, e, c] + <A[pred(b, e, p)] * h[b, e], B[cand(b, e, c)]>
// (A / B: the predecessor / successor codebooks, h: the projected draft hidden state, pred(b, 0, p) = the anchor token,
// pred(b, e, p) = cand(b, e - 1, p) for e > 0) with several small kernels and a batched GEMM, then walks it slot by slot
// with one program per request. The walk reads one row per slot: the row of the candidate it picked at the slot before.
// This kernel does both in one block per request, in shared memory: it gathers every slot's K successor rows and every
// possible predecessor row (the anchor, and the K candidates of each slot but the last), computes the lattice, then
// every row's pick (the uniform draw depends only on the slot, so every row can be decided before the walk), and walks
// the picks: a chain of S shared-memory reads instead of S dependent passes.
//
// Arithmetic per score, following the stock dtypes: prod[r] = A[pred, r] * h[r] (rounded to the product dtype, as the
// stock elementwise product), dot = sum_r prod[r] * B[c, r] accumulated in fp32 (bf16 codebooks: tensor-core MMA steps
// of 16 ranks, like the stock batched GEMM; fp32 codebooks: r in order), rounded to the GEMM output dtype, then
// score = unary + dot in fp32. The picks: greedy rows take the first NaN, else the first maximum (one-hot q row);
// sampling rows: scaled = score / temperature, p = exp(scaled - max) / sum (sums in candidate order), index = count of
// the inclusive cumulative sums at or below the uniform draw (clamped to K - 1), q row = p. The sum orders (and the exp)
// are not the stock ones, so this is draft-numerics: proposals and q rows can differ in the last places from the stock
// path.

#pragma once

#include <sgl_kernel/tensor.h>
#include <sgl_kernel/utils.h>

#include <sgl_kernel/utils.cuh>

#include <tvm/ffi/container/tensor.h>

#include <cuda_bf16.h>
#include <cstdint>
#include <type_traits>

namespace muse_vr::selector_fused {

using namespace sglang;

constexpr int kThreads = 256;
constexpr int kWarps = kThreads / 32;
constexpr int kMaxK = 32;
constexpr int kMaxSlots = 16;
constexpr int kChunk = 32;  // successor-row elements held in registers per pass of the lattice phase

template <typename T>
SGL_DEVICE float to_f(T v) {
  if constexpr (std::is_same_v<T, __nv_bfloat16>) return __bfloat162float(v);
  else return static_cast<float>(v);
}

// the stock tensors' rounding of an fp32 value to dtype T
template <typename T>
SGL_DEVICE float round_to(float v) {
  if constexpr (std::is_same_v<T, __nv_bfloat16>) return __bfloat162float(__float2bfloat16(v));
  else return v;
}

struct Params {
  const int64_t* cand;      // [bs, slots, K]
  const float* unary;       // [bs, slots, K]
  const void* hproj;        // [bs, slots, R]   (TH)
  const int64_t* anchor;    // [bs]
  const void* pred_book;    // [V, R]           (TC)
  const void* succ_book;    // [V, R]           (TC)
  const float* uniforms;    // [bs, slots]
  const float* temperatures;  // [bs]
  const uint8_t* greedy;    // [bs] (the bool mask viewed as bytes)
  int64_t* tokens;          // [bs, slots]
  float* q;                 // [bs, slots, K]
  int slots, k, r;
};

// TC: codebook dtype, TH: projected-hidden dtype, TP: product dtype (promote(TC, TH)), TD: GEMM output dtype (promote(TP, TC))
template <typename TC>
SGL_DEVICE void unpack(const uint4& w, float (&out)[16 / sizeof(TC)]) {
  if constexpr (std::is_same_v<TC, __nv_bfloat16>) {
    const uint32_t ws[4] = {w.x, w.y, w.z, w.w};
#pragma unroll
    for (int i = 0; i < 4; ++i) {  // a bf16 value is the high half of its fp32 value (exact)
      out[2 * i] = __uint_as_float(ws[i] << 16);
      out[2 * i + 1] = __uint_as_float(ws[i] & 0xffff0000u);
    }
  } else {
    out[0] = __uint_as_float(w.x);
    out[1] = __uint_as_float(w.y);
    out[2] = __uint_as_float(w.z);
    out[3] = __uint_as_float(w.w);
  }
}

// D += A (16 x 16, row-major) * B (16 x 8, column-major), bf16 inputs, fp32 accumulation (PTX ISA mma.sync m16n8k16)
SGL_DEVICE void mma_16816(float (&d)[4], const uint32_t (&a)[4], const uint32_t (&b)[2]) {
  asm volatile(
      "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, "
      "{%0, %1, %2, %3};\n"
      : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
}

// reductions over the 4 lanes of a quad (lanes 4q .. 4q + 3); every lane gets the same value
SGL_DEVICE float quad_max(float v) {
  v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, 1));
  return fmaxf(v, __shfl_xor_sync(0xffffffffu, v, 2));
}
SGL_DEVICE float quad_sum(float v) {  // (v0 + v1) + (v2 + v3) on every lane (additions commute exactly)
  v += __shfl_xor_sync(0xffffffffu, v, 1);
  return v + __shfl_xor_sync(0xffffffffu, v, 2);
}
SGL_DEVICE int quad_min(int v) {
  v = min(v, __shfl_xor_sync(0xffffffffu, v, 1));
  return min(v, __shfl_xor_sync(0xffffffffu, v, 2));
}
SGL_DEVICE int quad_sum(int v) {
  v += __shfl_xor_sync(0xffffffffu, v, 1);
  return v + __shfl_xor_sync(0xffffffffu, v, 2);
}

// one row's pick, K known at compile time (KT > 0) or not (KT == 0); the row is overwritten with its q values
template <int KT>
SGL_DEVICE int pick_row(float* row, int K, bool greedy, float temperature, float u) {
  constexpr int N = KT > 0 ? KT : 1;
  if constexpr (KT > 0) {
    float x[N];
#pragma unroll
    for (int c = 0; c < N; ++c) x[c] = row[c];
    int index;
    if (greedy) {
      int nan_at = N, hit = N;
      float best = -INFINITY;
#pragma unroll
      for (int c = 0; c < N; ++c) {
        if (x[c] != x[c] && nan_at == N) nan_at = c;
        best = fmaxf(best, x[c]);  // fmaxf skips NaN
      }
#pragma unroll
      for (int c = 0; c < N; ++c)
        if (hit == N && x[c] == best) hit = c;
      index = nan_at < N ? nan_at : (hit < N ? hit : 0);
#pragma unroll
      for (int c = 0; c < N; ++c) row[c] = c == index ? 1.0f : 0.0f;
    } else {
      float mx = -INFINITY;
#pragma unroll
      for (int c = 0; c < N; ++c) {
        x[c] = x[c] / temperature;
        mx = fmaxf(mx, x[c]);
      }
      float sum = 0.0f;
#pragma unroll
      for (int c = 0; c < N; ++c) {
        x[c] = __expf(x[c] - mx);
        sum += x[c];
      }
      float cum = 0.0f;
      int cnt = 0;
#pragma unroll
      for (int c = 0; c < N; ++c) {
        x[c] = x[c] / sum;
        cum += x[c];
        cnt += u >= cum ? 1 : 0;
        row[c] = x[c];
      }
      index = cnt < N - 1 ? cnt : N - 1;
    }
    return index;
  } else {
    int index;
    if (greedy) {
      int nan_at = -1, hit = -1;
      float best = -INFINITY;
      for (int c = 0; c < K; ++c) {
        const float s = row[c];
        if (s != s && nan_at < 0) nan_at = c;
        best = fmaxf(best, s);
      }
      for (int c = 0; c < K; ++c)
        if (hit < 0 && row[c] == best) hit = c;
      index = nan_at >= 0 ? nan_at : (hit >= 0 ? hit : 0);
      for (int c = 0; c < K; ++c) row[c] = c == index ? 1.0f : 0.0f;
    } else {
      float mx = -INFINITY;
      for (int c = 0; c < K; ++c) {
        row[c] = row[c] / temperature;
        mx = fmaxf(mx, row[c]);
      }
      float sum = 0.0f;
      for (int c = 0; c < K; ++c) {
        row[c] = __expf(row[c] - mx);
        sum += row[c];
      }
      float cum = 0.0f;
      int cnt = 0;
      for (int c = 0; c < K; ++c) {
        row[c] = row[c] / sum;
        cum += row[c];
        cnt += u >= cum ? 1 : 0;
      }
      index = cnt < K - 1 ? cnt : K - 1;
    }
    return index;
  }
}

// One lattice row held by a quad (lanes 4q .. 4q + 3): lane t4 holds the scores x[nt][i] of candidates nt * 8 + 2 t4 + i
// (ok: a real candidate). Returns the row's pick (the same on the 4 lanes) and leaves the row's q values in x: greedy =
// the first NaN, else the first maximum, q one-hot; sampling: scaled = score / temperature, p = exp(scaled - max) / sum,
// index = the count of inclusive cumulative sums (candidate order: tiles in order, quad lanes in order within a tile)
// <= the draw, clamped to K - 1. Called with all 32 lanes (full-warp shuffles).
template <int NTM>
SGL_DEVICE int quad_pick(float (&x)[NTM][2], const bool (&ok)[NTM][2], int t4, int lane, int K, bool greedy,
                         float temperature, float u_e) {
  constexpr int kNone = 1 << 30;
  int index;
  if (greedy) {
    int nan_c = kNone, hit = kNone;
    float best = -INFINITY;
#pragma unroll
    for (int nt = 0; nt < NTM; ++nt)
#pragma unroll
      for (int i = 0; i < 2; ++i)
        if (ok[nt][i]) {
          if (x[nt][i] != x[nt][i]) nan_c = min(nan_c, nt * 8 + 2 * t4 + i);
          best = fmaxf(best, x[nt][i]);  // fmaxf skips NaN
        }
    nan_c = quad_min(nan_c);
    best = quad_max(best);
#pragma unroll
    for (int nt = 0; nt < NTM; ++nt)
#pragma unroll
      for (int i = 0; i < 2; ++i)
        if (ok[nt][i] && x[nt][i] == best) hit = min(hit, nt * 8 + 2 * t4 + i);
    hit = quad_min(hit);
    index = nan_c < kNone ? nan_c : (hit < kNone ? hit : 0);
#pragma unroll
    for (int nt = 0; nt < NTM; ++nt)
#pragma unroll
      for (int i = 0; i < 2; ++i) x[nt][i] = nt * 8 + 2 * t4 + i == index ? 1.0f : 0.0f;
  } else {
    float mx = -INFINITY;
#pragma unroll
    for (int nt = 0; nt < NTM; ++nt)
#pragma unroll
      for (int i = 0; i < 2; ++i) {
        x[nt][i] = x[nt][i] / temperature;
        if (ok[nt][i]) mx = fmaxf(mx, x[nt][i]);
      }
    mx = quad_max(mx);
    float sum = 0.0f;
#pragma unroll
    for (int nt = 0; nt < NTM; ++nt)
#pragma unroll
      for (int i = 0; i < 2; ++i) {
        x[nt][i] = ok[nt][i] ? __expf(x[nt][i] - mx) : 0.0f;
        sum += x[nt][i];
      }
    sum = quad_sum(sum);  // the same value on the 4 lanes
    // inclusive cumulative sums in candidate order: tiles in order, quad lanes in order within a tile
    float base = 0.0f;
    int cnt = 0;
#pragma unroll
    for (int nt = 0; nt < NTM; ++nt) {
      x[nt][0] = x[nt][0] / sum;
      x[nt][1] = x[nt][1] / sum;
      const float pair = x[nt][0] + x[nt][1];
      float incl = pair;
      float o = __shfl_up_sync(0xffffffffu, incl, 1, 4);
      if (t4 >= 1) incl += o;
      o = __shfl_up_sync(0xffffffffu, incl, 2, 4);
      if (t4 >= 2) incl += o;
      const float up = __shfl_up_sync(0xffffffffu, incl, 1, 4);
      const float excl = t4 >= 1 ? up : 0.0f;
      const float tile = __shfl_sync(0xffffffffu, incl, (lane & ~3) | 3);
      const float c0 = (base + excl) + x[nt][0];
      const float c1 = (base + excl) + pair;
      cnt += (ok[nt][0] && u_e >= c0 ? 1 : 0) + (ok[nt][1] && u_e >= c1 ? 1 : 0);
      base += tile;
    }
    cnt = quad_sum(cnt);
    index = cnt < K - 1 ? cnt : K - 1;
  }
  return index;
}

template <typename TC, typename TH, typename TP, typename TD, int KT>
__global__ __launch_bounds__(kThreads) void selector_fused_kernel(const __grid_constant__ Params p) {
  static_assert(sizeof(TP) == sizeof(TC), "the product rows reuse the codebook vector width");
  constexpr int V = 16 / static_cast<int>(sizeof(TC));  // elements per 16-byte vector
  constexpr bool kMma = std::is_same_v<TC, __nv_bfloat16> && std::is_same_v<TP, __nv_bfloat16>;
  extern __shared__ __align__(16) unsigned char smem_raw[];
  const int b = blockIdx.x, tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
  const int S = p.slots, K = KT > 0 ? KT : p.k, R = p.r;
  // predecessor rows of the lattice: row 0 = the anchor (slot 0), row 1 + (e - 1) K + p = cand(e - 1, p) (slot e > 0)
  const int n_rows = 1 + (S - 1) * K, n_succ = S * K;
  const int RS = R + V, LS = K + 1;  // padded row strides: product / successor rows (elements), lattice rows (floats)
  float* h = reinterpret_cast<float*>(smem_raw);                                                // [S][R]
  TP* prod = reinterpret_cast<TP*>(h + ((S * R + 3) / 4) * 4);                                   // [n_rows][RS]
  float* lat = reinterpret_cast<float*>(prod + static_cast<int64_t>(n_rows) * RS);               // [n_rows][LS]
  TC* succ = reinterpret_cast<TC*>(lat + ((static_cast<int64_t>(n_rows) * LS + 3) / 4) * 4);    // [S K][RS]
  __shared__ int64_t cids[kMaxSlots * kMaxK + 1];
  __shared__ float un[kMaxSlots * kMaxK];
  __shared__ float uni[kMaxSlots];
  __shared__ int idx[1 + (kMaxSlots - 1) * kMaxK];
  __shared__ int path[kMaxSlots];

  // (1) the request's candidates, anchor, unary scores, draws and projected hidden state
  for (int i = tid; i < n_succ; i += kThreads) {
    cids[i] = p.cand[static_cast<int64_t>(b) * n_succ + i];
    un[i] = p.unary[static_cast<int64_t>(b) * n_succ + i];
  }
  if (tid == 0) cids[n_succ] = p.anchor[b];
  if (tid < S) uni[tid] = p.uniforms[static_cast<int64_t>(b) * S + tid];
  const TH* H = static_cast<const TH*>(p.hproj) + static_cast<int64_t>(b) * S * R;
  for (int i = tid; i < S * R; i += kThreads) h[i] = to_f(H[i]);
  __syncthreads();

  // (2) gather every successor row and every predecessor row in 16-byte vectors (each thread issues kBatch loads
  // before using any); predecessor rows are stored as their products with h of their slot, rounded to TP
  const int vpr = R / V;
  const int total = (n_succ + n_rows) * vpr;
  constexpr int kBatch = 8;
  for (int base = 0; base < total; base += kThreads * kBatch) {
    uint4 tmp[kBatch];
#pragma unroll
    for (int u = 0; u < kBatch; ++u) {
      const int v = base + u * kThreads + tid;
      if (v < total) {
        const int row = v / vpr, col = v - row * vpr;
        const int64_t tok = row < n_succ ? cids[row] : (row == n_succ ? cids[n_succ] : cids[row - n_succ - 1]);
        const TC* book = static_cast<const TC*>(row < n_succ ? p.succ_book : p.pred_book);
        tmp[u] = __ldg(reinterpret_cast<const uint4*>(book + tok * R) + col);
      }
    }
#pragma unroll
    for (int u = 0; u < kBatch; ++u) {
      const int v = base + u * kThreads + tid;
      if (v < total) {
        const int row = v / vpr, col = v - row * vpr;
        if (row < n_succ) {
          *reinterpret_cast<uint4*>(succ + static_cast<int64_t>(row) * RS + col * V) = tmp[u];
        } else {
          const int j = row - n_succ;
          const int e = j == 0 ? 0 : 1 + (j - 1) / K;
          float x[V];
          unpack<TC>(tmp[u], x);
          const float* hh = h + e * R + col * V;
          TP* d = prod + static_cast<int64_t>(j) * RS + col * V;
          if constexpr (std::is_same_v<TP, __nv_bfloat16>) {
            uint32_t w[4];
#pragma unroll
            for (int i = 0; i < 4; ++i) {
              const __nv_bfloat162 pr = __floats2bfloat162_rn(x[2 * i] * hh[2 * i], x[2 * i + 1] * hh[2 * i + 1]);
              w[i] = *reinterpret_cast<const uint32_t*>(&pr);
            }
            *reinterpret_cast<uint4*>(d) = make_uint4(w[0], w[1], w[2], w[3]);
          } else {
            *reinterpret_cast<float4*>(d) = make_float4(x[0] * hh[0], x[1] * hh[1], x[2] * hh[2], x[3] * hh[3]);
          }
        }
      }
    }
  }
  __syncthreads();

  // (3) + (4) the lattice lat[j][c] = unary[e][c] + round_to<TD>(sum_r prod[j][r] * B[cand(e, c)][r]) for every row j of
  // slot e, and every row's pick, in parallel (the draw depends only on the slot): greedy = the first NaN, else the first
  // maximum, q one-hot; sampling: scaled = score / temperature, p = exp(scaled - max) / sum, index = the count of
  // inclusive cumulative sums <= the draw, clamped to K - 1. The lattice rows are left holding their q values.
  const float temperature = p.temperatures[b];
  const bool greedy = p.greedy[b] != 0;
  if constexpr (kMma) {
    // one warp per tile of 16 rows x all K candidates (8-wide MMA tiles, R / 16 tensor-core steps); rows / candidates
    // past the slot's are clamped to valid ones and dropped. Lane (g8, t4) holds rows g8 and g8 + 8 of the tile at
    // candidates nt * 8 + 2 t4 + {0, 1}: each row's pick is reduced over the 4 lanes of its quad.
    constexpr int NTM = KT > 0 ? (KT + 7) / 8 : kMaxK / 8;
    const int g8 = lane >> 2, t4 = lane & 3;
    const int nt_n = (K + 7) / 8, mt_n = (K + 15) / 16;
    const int units = 1 + (S - 1) * mt_n;
    for (int u = warp; u < units; u += kWarps) {
      const int e = u == 0 ? 0 : 1 + (u - 1) / mt_n;
      const int m0 = u == 0 ? 0 : ((u - 1) - (e - 1) * mt_n) * 16;
      const int j0 = e == 0 ? 0 : 1 + (e - 1) * K;
      const int nr = e == 0 ? 1 : K;
      const uint32_t* pa = reinterpret_cast<const uint32_t*>(prod + static_cast<int64_t>(min(j0 + m0 + g8, n_rows - 1)) * RS);
      const uint32_t* pb = reinterpret_cast<const uint32_t*>(prod + static_cast<int64_t>(min(j0 + m0 + g8 + 8, n_rows - 1)) * RS);
      float d[NTM][4];
#pragma unroll
      for (int nt = 0; nt < NTM; ++nt) d[nt][0] = d[nt][1] = d[nt][2] = d[nt][3] = 0.0f;
      for (int k0 = 0; k0 < R; k0 += 16) {
        const int w0 = (k0 >> 1) + t4;
        const uint32_t a[4] = {pa[w0], pb[w0], pa[w0 + 4], pb[w0 + 4]};
#pragma unroll
        for (int nt = 0; nt < NTM; ++nt) {
          if (nt < nt_n) {
            const uint32_t* pc =
                reinterpret_cast<const uint32_t*>(succ + static_cast<int64_t>(e * K + min(nt * 8 + g8, K - 1)) * RS);
            const uint32_t bb[2] = {pc[w0], pc[w0 + 4]};
            mma_16816(d[nt], a, bb);
          }
        }
      }
      const float u_e = uni[e];
#pragma unroll
      for (int hr = 0; hr < 2; ++hr) {
        const int m = m0 + g8 + 8 * hr;
        float x[NTM][2];
        bool ok[NTM][2];
#pragma unroll
        for (int nt = 0; nt < NTM; ++nt)
#pragma unroll
          for (int i = 0; i < 2; ++i) {
            const int c = nt * 8 + 2 * t4 + i;
            ok[nt][i] = nt < nt_n && c < K;
            x[nt][i] = ok[nt][i] ? un[e * K + c] + round_to<TD>(d[nt][2 * hr + i]) : 0.0f;
          }
        const int index = quad_pick<NTM>(x, ok, t4, lane, K, greedy, temperature, u_e);
        if (m < nr) {
#pragma unroll
          for (int nt = 0; nt < NTM; ++nt)
#pragma unroll
            for (int i = 0; i < 2; ++i)
              if (ok[nt][i]) lat[(j0 + m) * LS + nt * 8 + 2 * t4 + i] = x[nt][i];
          if (t4 == 0) idx[j0 + m] = index;
        }
      }
    }
  } else {
    for (int w = tid; w < n_rows * K; w += kThreads) {
      const int j = w / K, c = w - (w / K) * K;
      const int e = j == 0 ? 0 : 1 + (j - 1) / K;
      const TP* pr = prod + static_cast<int64_t>(j) * RS;
      const TC* br = succ + static_cast<int64_t>(e * K + c) * RS;
      float acc = 0.0f;
      for (int r = 0; r < R; ++r) acc = fmaf(to_f(pr[r]), to_f(br[r]), acc);
      lat[j * LS + c] = un[e * K + c] + round_to<TD>(acc);
    }
    __syncthreads();
    for (int j = tid; j < n_rows; j += kThreads) {
      const int e = j == 0 ? 0 : 1 + (j - 1) / K;
      idx[j] = pick_row<KT>(lat + j * LS, K, greedy, temperature, uni[e]);
    }
  }
  __syncthreads();

  // (5) the walk (a chain of shared-memory reads), then the path's tokens and q rows, by warp 0
  if (warp == 0) {
    if (lane == 0) {
      int prev = idx[0];
      path[0] = 0;
      for (int e = 1; e < S; ++e) {
        const int j = 1 + (e - 1) * K + prev;
        path[e] = j;
        prev = idx[j];
      }
    }
    __syncwarp();
    for (int i = lane; i < S * K; i += 32) {
      const int e = i / K, c = i - e * K;
      p.q[(static_cast<int64_t>(b) * S + e) * K + c] = lat[path[e] * LS + c];
    }
    if (lane < S) p.tokens[static_cast<int64_t>(b) * S + lane] = cids[lane * K + idx[path[lane]]];
  }
}

// ---- one cluster per request, one CTA per slot (bf16 codebooks and hidden state, slots <= 8, K 8 or 16) ----
// CTA e gathers its slot's K successor rows and its predecessor rows (the anchor for slot 0, else the K candidates of slot
// e - 1), computes its lattice rows on the tensor cores (warps split the candidates' 8-wide tiles and the rank), picks every
// row (quad epilogue) and stores the q rows, picks and candidate ids into CTA 0's shared memory (distributed shared
// memory). After the cluster barrier (release / acquire) CTA 0 walks the picks and writes the outputs.
constexpr int kMaxClusterSlots = 8;

SGL_DEVICE uint32_t map_to_rank(const void* local, uint32_t rank) {
  uint32_t d;
  asm volatile("mapa.shared::cluster.u32 %0, %1, %2;"
               : "=r"(d)
               : "r"(static_cast<uint32_t>(__cvta_generic_to_shared(local))), "r"(rank));
  return d;
}
SGL_DEVICE void st_cluster(uint32_t addr, float v) {
  asm volatile("st.shared::cluster.f32 [%0], %1;" ::"r"(addr), "f"(v) : "memory");
}
SGL_DEVICE void st_cluster(uint32_t addr, int v) {
  asm volatile("st.shared::cluster.s32 [%0], %1;" ::"r"(addr), "r"(v) : "memory");
}
SGL_DEVICE void st_cluster(uint32_t addr, int64_t v) {
  asm volatile("st.shared::cluster.s64 [%0], %1;" ::"r"(addr), "l"(v) : "memory");
}

template <int KT>
__global__ __launch_bounds__(kThreads) void selector_cluster_kernel(const __grid_constant__ Params p) {
  using BF = __nv_bfloat16;
  static_assert(KT == 8 || KT == 16, "8 or 16 candidates per slot");
  constexpr int V = 8, NT = KT / 8, KP = kWarps / NT;  // warp w: candidate tile w % NT, rank part w / NT
  constexpr int kBatch = 8;
  extern __shared__ __align__(16) unsigned char smem_raw[];
  const int e = blockIdx.x, b = blockIdx.y, tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
  const int S = p.slots, R = p.r;
  const int nr = e == 0 ? 1 : KT;  // predecessor rows of this slot
  const int RS = R + V;            // padded row stride (elements)
  float* h = reinterpret_cast<float*>(smem_raw);            // [R]
  BF* prod = reinterpret_cast<BF*>(h + ((R + 3) / 4) * 4);  // [16][RS]: predecessor rows * h, rounded to bf16
  BF* succ = prod + 16 * RS;                                // [KT][RS]
  __shared__ float red[KP][16][KT];
  __shared__ float un[KT];
  __shared__ int64_t ids[2 * KT];  // [0, KT): this slot's candidates, [KT, KT + nr): its predecessors
  // written by every CTA of the cluster into CTA 0's copy (unused in the other CTAs)
  __shared__ float all_q[kMaxClusterSlots][16][KT];
  __shared__ int all_idx[kMaxClusterSlots][16];
  __shared__ int64_t all_c[kMaxClusterSlots][KT];
  __shared__ int path[kMaxClusterSlots];

  // first cluster barrier phase: every CTA has started before any remote store (the wait is right before the stores)
  asm volatile("barrier.cluster.arrive.relaxed.aligned;" ::: "memory");

  // (1) ids, unary scores and h of this slot
  const int64_t* cand = p.cand + static_cast<int64_t>(b) * S * KT;
  if (tid < KT) {
    ids[tid] = cand[e * KT + tid];
    un[tid] = p.unary[(static_cast<int64_t>(b) * S + e) * KT + tid];
  } else if (tid < KT + nr) {
    ids[tid] = e == 0 ? p.anchor[b] : cand[(e - 1) * KT + (tid - KT)];
  }
  const BF* H = static_cast<const BF*>(p.hproj) + (static_cast<int64_t>(b) * S + e) * R;
  for (int i = tid; i < R; i += kThreads) h[i] = __bfloat162float(H[i]);
  __syncthreads();

  // (2) gather (16-byte vectors, kBatch loads in flight per thread); predecessor rows become their products with h
  const int vpr = R / V, total = (KT + nr) * vpr;
  for (int base = 0; base < total; base += kThreads * kBatch) {
    uint4 tmp[kBatch];
#pragma unroll
    for (int u = 0; u < kBatch; ++u) {
      const int v = base + u * kThreads + tid;
      if (v < total) {
        const int row = v / vpr, col = v - (v / vpr) * vpr;
        const BF* book = static_cast<const BF*>(row < KT ? p.succ_book : p.pred_book);
        tmp[u] = __ldg(reinterpret_cast<const uint4*>(book + ids[row] * R) + col);
      }
    }
#pragma unroll
    for (int u = 0; u < kBatch; ++u) {
      const int v = base + u * kThreads + tid;
      if (v < total) {
        const int row = v / vpr, col = v - (v / vpr) * vpr;
        if (row < KT) {
          *reinterpret_cast<uint4*>(succ + row * RS + col * V) = tmp[u];
        } else {
          float x[V];
          unpack<BF>(tmp[u], x);
          const float* hh = h + col * V;
          uint32_t w[4];
#pragma unroll
          for (int i = 0; i < 4; ++i) {
            const __nv_bfloat162 pr = __floats2bfloat162_rn(x[2 * i] * hh[2 * i], x[2 * i + 1] * hh[2 * i + 1]);
            w[i] = *reinterpret_cast<const uint32_t*>(&pr);
          }
          *reinterpret_cast<uint4*>(prod + (row - KT) * RS + col * V) = make_uint4(w[0], w[1], w[2], w[3]);
        }
      }
    }
  }
  __syncthreads();

  // (3) partial lattice per (candidate tile, rank part); predecessor rows past nr are clamped (dropped below)
  const int g8 = lane >> 2, t4 = lane & 3;
  {
    const int nt = warp % NT, kp = warp / NT;
    const uint32_t* pa = reinterpret_cast<const uint32_t*>(prod + min(g8, nr - 1) * RS);
    const uint32_t* pb = reinterpret_cast<const uint32_t*>(prod + min(g8 + 8, nr - 1) * RS);
    const uint32_t* pc = reinterpret_cast<const uint32_t*>(succ + (nt * 8 + g8) * RS);
    float d[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    for (int st = kp; st < R / 16; st += KP) {
      const int w0 = st * 8 + t4;
      const uint32_t a[4] = {pa[w0], pb[w0], pa[w0 + 4], pb[w0 + 4]};
      const uint32_t bb[2] = {pc[w0], pc[w0 + 4]};
      mma_16816(d, a, bb);
    }
    red[kp][g8][nt * 8 + 2 * t4] = d[0];
    red[kp][g8][nt * 8 + 2 * t4 + 1] = d[1];
    red[kp][g8 + 8][nt * 8 + 2 * t4] = d[2];
    red[kp][g8 + 8][nt * 8 + 2 * t4 + 1] = d[3];
  }
  __syncthreads();
  asm volatile("barrier.cluster.wait.aligned;" ::: "memory");

  // (4) warps 0 / 1: rows g8 / g8 + 8, rank parts summed in order, then the quad pick; results into CTA 0
  if (warp < 2) {
    const float temperature = p.temperatures[b];
    const bool greedy = p.greedy[b] != 0;
    const float u_e = p.uniforms[static_cast<int64_t>(b) * S + e];
    const uint32_t q0 = map_to_rank(&all_q[e][0][0], 0), i0 = map_to_rank(&all_idx[e][0], 0);
    {
      const int m = g8 + 8 * warp;
      float x[NT][2];
      bool ok[NT][2];
#pragma unroll
      for (int nt = 0; nt < NT; ++nt)
#pragma unroll
        for (int i = 0; i < 2; ++i) {
          const int c = nt * 8 + 2 * t4 + i;
          float acc = red[0][m][c];
#pragma unroll
          for (int kp = 1; kp < KP; ++kp) acc += red[kp][m][c];
          ok[nt][i] = true;
          x[nt][i] = un[c] + round_to<BF>(acc);
        }
      const int index = quad_pick<NT>(x, ok, t4, lane, KT, greedy, temperature, u_e);
      if (m < nr) {
#pragma unroll
        for (int nt = 0; nt < NT; ++nt)
#pragma unroll
          for (int i = 0; i < 2; ++i) st_cluster(q0 + 4u * static_cast<uint32_t>(m * KT + nt * 8 + 2 * t4 + i), x[nt][i]);
        if (t4 == 0) st_cluster(i0 + 4u * static_cast<uint32_t>(m), index);
      }
    }
    if (warp == 0 && lane < KT) st_cluster(map_to_rank(&all_c[e][lane], 0), ids[lane]);
  }
  // second phase: the remote stores are visible to CTA 0 (release / acquire, cluster scope); after it no CTA accesses
  // another CTA's shared memory, so the others may exit
  asm volatile("barrier.cluster.arrive.release.aligned;" ::: "memory");
  asm volatile("barrier.cluster.wait.acquire.aligned;" ::: "memory");
  if (e != 0) return;

  // (5) CTA 0, warp 0: pick[0] = the anchor row's pick, pick[s] = slot s's pick in row pick[s - 1]; q row of slot s =
  // row (s == 0 ? 0 : pick[s - 1]) of slot s
  if (warp == 0) {
    if (lane == 0) {
      int prev = all_idx[0][0];
      path[0] = prev;
      for (int s = 1; s < S; ++s) {
        prev = all_idx[s][prev];
        path[s] = prev;
      }
    }
    __syncwarp();
    for (int i = lane; i < S * KT; i += 32) {
      const int s = i / KT, c = i - (i / KT) * KT;
      p.q[(static_cast<int64_t>(b) * S + s) * KT + c] = all_q[s][s == 0 ? 0 : path[s - 1]][c];
    }
    if (lane < S) p.tokens[static_cast<int64_t>(b) * S + lane] = all_c[lane][path[lane]];
  }
}

inline std::size_t cluster_smem_bytes(int64_t k, int64_t r) {
  return sizeof(float) * static_cast<std::size_t>(((r + 3) / 4) * 4) +
         sizeof(__nv_bfloat16) * static_cast<std::size_t>((16 + k) * (r + 8));
}

template <int KT>
inline void launch_cluster(const Params& p, int bs, int s, DLDevice dev) {
  auto kernel = selector_cluster_kernel<KT>;
  const std::size_t smem = cluster_smem_bytes(KT, p.r);
  host::RuntimeDeviceCheck(::cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
                                                  static_cast<int>(smem)));
  host::LaunchKernel(dim3(static_cast<unsigned>(s), static_cast<unsigned>(bs)), dim3(kThreads), dev, smem)
      .enable_cluster(dim3(static_cast<unsigned>(s), 1, 1))(kernel, p);
}

template <typename TC>
inline std::size_t smem_bytes(int64_t s, int64_t k, int64_t r) {
  const int64_t n = 1 + (s - 1) * k, v = 16 / static_cast<int64_t>(sizeof(TC));
  return sizeof(float) * static_cast<std::size_t>(((s * r + 3) / 4) * 4 + ((n * (k + 1) + 3) / 4) * 4) +
         sizeof(TC) * static_cast<std::size_t>((n + s * k) * (r + v));
}

template <typename TC, typename TH, typename TP, typename TD, int KT>
inline void launch(const Params& p, int bs, std::size_t smem, DLDevice dev) {
  auto kernel = selector_fused_kernel<TC, TH, TP, TD, KT>;
  host::RuntimeDeviceCheck(::cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
                                                  static_cast<int>(smem)));
  host::LaunchKernel(static_cast<unsigned>(bs), kThreads, dev, smem)(kernel, p);
}

struct SelectorFused {
  // codebook dtype code: 0 bf16, 1 fp32; hidden dtype code: 0 bf16, 1 fp32
  static void run(const tvm::ffi::TensorView cand, const tvm::ffi::TensorView unary, const tvm::ffi::TensorView hproj,
                  const tvm::ffi::TensorView anchor, const tvm::ffi::TensorView pred_book,
                  const tvm::ffi::TensorView succ_book, const tvm::ffi::TensorView uniforms,
                  const tvm::ffi::TensorView temperatures, const tvm::ffi::TensorView greedy,
                  const tvm::ffi::TensorView tokens, const tvm::ffi::TensorView q, int64_t book_code, int64_t hid_code,
                  int64_t single_block) {
    using namespace host;
    auto BS = SymbolicSize{"bs"};
    auto S = SymbolicSize{"slots"};
    auto K = SymbolicSize{"k"};
    auto R = SymbolicSize{"rank"};
    auto V = SymbolicSize{"vocab"};
    auto device_ = SymbolicDevice{};
    device_.set_options<kDLCUDA>();
    TensorMatcher({BS, S, K}).with_dtype<int64_t>().with_device(device_).verify(cand);
    TensorMatcher({BS, S, K}).with_dtype<float>().with_device(device_).verify(unary);
    TensorMatcher({BS, S, R}).with_device(device_).verify(hproj);
    TensorMatcher({BS}).with_dtype<int64_t>().with_device(device_).verify(anchor);
    TensorMatcher({V, R}).with_device(device_).verify(pred_book);
    TensorMatcher({V, R}).with_device(device_).verify(succ_book);
    TensorMatcher({BS, S}).with_dtype<float>().with_device(device_).verify(uniforms);
    TensorMatcher({BS}).with_dtype<float>().with_device(device_).verify(temperatures);
    TensorMatcher({BS}).with_dtype<uint8_t>().with_device(device_).verify(greedy);
    TensorMatcher({BS, S}).with_dtype<int64_t>().with_device(device_).verify(tokens);
    TensorMatcher({BS, S, K}).with_dtype<float>().with_device(device_).verify(q);
    const int64_t bs = BS.unwrap(), s = S.unwrap(), k = K.unwrap(), r = R.unwrap();
    if (bs == 0) return;
    RuntimeCheck(k >= 1 && k <= kMaxK && s >= 1 && s <= kMaxSlots && r >= 1 && r <= 1024, "selector_fused: shape");
    RuntimeCheck((r * (book_code == 0 ? 2 : 4)) % 16 == 0 && reinterpret_cast<uintptr_t>(pred_book.data_ptr()) % 16 == 0 &&
                     reinterpret_cast<uintptr_t>(succ_book.data_ptr()) % 16 == 0,
                 "selector_fused: codebook rows must be whole 16-byte vectors");
    const bool cluster = !single_block && book_code == 0 && hid_code == 0 && s <= kMaxClusterSlots && (k == 16 || k == 8) &&
                         r % 16 == 0;
    const std::size_t smem = book_code == 0 ? smem_bytes<__nv_bfloat16>(s, k, r) : smem_bytes<float>(s, k, r);
    RuntimeCheck(cluster || smem <= 200 * 1024, "selector_fused: staged rows do not fit shared memory");
    Params p{};
    p.cand = static_cast<const int64_t*>(cand.data_ptr());
    p.unary = static_cast<const float*>(unary.data_ptr());
    p.hproj = hproj.data_ptr();
    p.anchor = static_cast<const int64_t*>(anchor.data_ptr());
    p.pred_book = pred_book.data_ptr();
    p.succ_book = succ_book.data_ptr();
    p.uniforms = static_cast<const float*>(uniforms.data_ptr());
    p.temperatures = static_cast<const float*>(temperatures.data_ptr());
    p.greedy = static_cast<const uint8_t*>(greedy.data_ptr());
    p.tokens = static_cast<int64_t*>(tokens.data_ptr());
    p.q = static_cast<float*>(q.data_ptr());
    p.slots = static_cast<int>(s);
    p.k = static_cast<int>(k);
    p.r = static_cast<int>(r);
    const DLDevice dev = device_.unwrap();
    if (cluster) {
      if (k == 16) {
        launch_cluster<16>(p, static_cast<int>(bs), static_cast<int>(s), dev);
      } else {
        launch_cluster<8>(p, static_cast<int>(bs), static_cast<int>(s), dev);
      }
    } else if (book_code == 0 && hid_code == 0) {
      using BF = __nv_bfloat16;
      RuntimeCheck(r % 16 == 0, "selector_fused: bf16 rank must be a multiple of 16");
      if (k == 16) {
        launch<BF, BF, BF, BF, 16>(p, static_cast<int>(bs), smem, dev);
      } else if (k == 8) {
        launch<BF, BF, BF, BF, 8>(p, static_cast<int>(bs), smem, dev);
      } else {
        launch<BF, BF, BF, BF, 0>(p, static_cast<int>(bs), smem, dev);
      }
    } else if (book_code == 1 && hid_code == 0) {
      launch<float, __nv_bfloat16, float, float, 0>(p, static_cast<int>(bs), smem, dev);
    } else if (book_code == 1 && hid_code == 1) {
      launch<float, float, float, float, 0>(p, static_cast<int>(bs), smem, dev);
    } else {
      RuntimeCheck(false, "selector_fused: dtype combination");
    }
  }
};

}  // namespace muse_vr::selector_fused
