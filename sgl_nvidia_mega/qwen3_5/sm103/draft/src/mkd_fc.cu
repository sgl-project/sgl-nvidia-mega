// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// Adapted from SGLang (Apache-2.0): python/sglang/kernels/jit/csrc/speculative/mkd_draft_fc.cuh
// (branch nvidia/qwen35-agentx-gb300), device code unchanged.
//
// MTP draft fc for one token from an exchanged input, bitwise equal to the row-split fc
// (draft_fc_partial on each rank's input-column slice, fp32 partials summed in rank order,
// one bf16 rounding).
//
// sum:   every rank computes the partials of the first kEmbed slices (the embedding half)
//        itself from the exchanged embedding row, with the weight rows bulk-copied into
//        shared memory before the PDL wait, then adds the received partials of the hidden
//        slices in rank order. With a memo row (T = the embedding-half partials already
//        summed in rank order) the embedding-half dots are skipped and the sum starts at T.
// memo:  builds those T rows for a list of embedding rows with the same arithmetic.
//
// Per partial: Gemma RMSNorm of the half (256-thread statistics, bf16 output), then one warp
// per weight row over the slice in 16-byte lane-strided vectors, warp::reduce_sum; this is
// the arithmetic of draft_fc_partial_kernel.
//
// Built with NVRTC (MKD_WORLD = the TP size, 4 or 8): the kernels are device functions, and the
// extern "C" entries at the end instantiate them for hidden 4096 (MKD_FC_MAX_ROWS = 27 fc rows a
// CTA, the 2.2 MAX_ROWS; memo blocks of 8 rows), PDL on where 2.2 launches with PDL.

#include "mkd_common.cuh"

#ifndef MKD_WORLD
#define MKD_WORLD 8
#endif
#ifndef MKD_FC_MAX_ROWS
#define MKD_FC_MAX_ROWS 27
#endif

namespace sglang {

namespace mkd_fc {

SGL_DEVICE void bulk_g2s(void* smem_dst, const void* gmem_src, uint32_t bytes, uint64_t* bar) {
  asm volatile(
      "cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];" ::"r"(
          device::ptx::to_shared(smem_dst)),
      "l"(gmem_src),
      "r"(bytes),
      "r"(device::ptx::to_shared(bar))
      : "memory");
}

/// \brief Gemma RMSNorm of one row x[kH] into s_x (bf16), with draft_fc_partial's
/// statistics order: 256 threads, thread t owns 16-byte vectors t, t + 256, ...
template <int kH>
SGL_DEVICE void gemma_norm_row(
    const bf16_t* x, const bf16_t* norm_w, float eps, bf16_t* s_x, float* s_red) {
  using namespace device;
  constexpr int kThreads = 256;
  constexpr int kWarps = kThreads / 32;
  using vec_t = AlignedVector<bf16_t, 8>;
  constexpr int kNormVecs = kH / (8 * kThreads);
  const int warp = static_cast<int>(threadIdx.x / kWarpThreads);
  const int lane = static_cast<int>(get_lane_id());
  float ss = 0.0f;
#pragma unroll
  for (int k = 0; k < kNormVecs; ++k) {
    vec_t v;
    v.load(x, k * kThreads + static_cast<int>(threadIdx.x));
#pragma unroll
    for (int j = 0; j < 8; ++j) {
      const float f = cast<fp32_t>(v[j]);
      ss = fmaf(f, f, ss);
    }
  }
  ss = warp::reduce_sum(ss);
  if (lane == 0) s_red[warp] = ss;
  __syncthreads();
  float tot = 0.0f;
#pragma unroll
  for (int w = 0; w < kWarps; ++w) tot += s_red[w];
  const float inv = rsqrtf(tot / static_cast<float>(kH) + eps);
  for (int k = static_cast<int>(threadIdx.x); k < kH / 8; k += kThreads) {
    vec_t v, g, o;
    v.load(x, k);
    g.load(norm_w, k);
#pragma unroll
    for (int j = 0; j < 8; ++j) {
      o[j] = cast<bf16_t>(cast<fp32_t>(v[j]) * inv * (1.0f + cast<fp32_t>(g[j])));
    }
    o.store(s_x, k);
  }
  __syncthreads();
}

/// \brief One slice partial: warp dot of a weight row segment [kW] with s_x[kW].
template <int kW>
SGL_DEVICE float slice_dot(const bf16_t* w_row, const bf16_t* s_x) {
  using namespace device;
  using vec_t = AlignedVector<bf16_t, 8>;
  constexpr int kDotVecs = kW / (8 * kWarpThreads);
  const int lane = static_cast<int>(get_lane_id());
  vec_t wv[kDotVecs];
#pragma unroll
  for (int k = 0; k < kDotVecs; ++k) wv[k].load(w_row, k * kWarpThreads + lane);
  float acc = 0.0f;
#pragma unroll
  for (int k = 0; k < kDotVecs; ++k) {
    vec_t xv;
    xv.load(s_x, k * kWarpThreads + lane);
#pragma unroll
    for (int j = 0; j < 8; ++j) acc = fmaf(cast<fp32_t>(wv[k][j]), cast<fp32_t>(xv[j]), acc);
  }
  return warp::reduce_sum(acc);
}

}  // namespace mkd_fc

struct MkdFcSumParams {
  const bf16_t* emb;                  // [kH] exchanged embedding row (null with a memo row)
  const float* memo;                  // [kH] memo row T (null without)
  const float* hidden_partials;       // [kWorld - kEmbed, kH] fp32 partials of the hidden slices
  const bf16_t* __restrict__ fc_w;    // [kH, 2 * kH] fc weight
  const bf16_t* __restrict__ norm_w;  // [kH] pre-fc embedding norm weight
  bf16_t* out;                        // [kH]
  float eps;
};

/**
 * \brief fc output of one token. Grid = one CTA per SM, 256 threads; CTA c owns a
 * contiguous block of output rows. \tparam kMemo selects the memo-row path.
 */
template <int kH, int kWorld, int kMaxRows, bool kMemo, bool kUsePDL>
SGL_DEVICE void mkd_fc_sum_kernel(const MkdFcSumParams& p) {
  using namespace device;
  constexpr int kThreads = 256;
  constexpr int kWarps = kThreads / 32;
  constexpr int kW = 2 * kH / kWorld;       // slice width
  constexpr int kEmbed = kH / kW;           // embedding-half slices
  constexpr int kRowBytes = kH * static_cast<int>(sizeof(bf16_t));  // embedding half of one fc row
  static_assert(kW % (8 * 32) == 0 && kH % kW == 0);

  extern __shared__ __align__(128) uint8_t smem[];  // [kMaxRows, kH] embedding-half weight rows
  __shared__ alignas(16) bf16_t s_x[kMemo ? 8 : kH];
  __shared__ float s_red[kWarps];
  __shared__ uint64_t bar;

  const int warp = static_cast<int>(threadIdx.x / kWarpThreads);
  const int lane = static_cast<int>(get_lane_id());
  const int G = static_cast<int>(gridDim.x);
  const int c = static_cast<int>(blockIdx.x);
  const int base = kH / G, rem = kH % G;
  const int r0 = c * base + min(c, rem);
  const int nrows = base + (c < rem ? 1 : 0);

  if constexpr (!kMemo) {
    if (threadIdx.x == 0) {
      ptx::mbar_init(&bar, 1);
      asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
      if (nrows > 0) {
        ptx::mbar_arrive_expect_tx(&bar, static_cast<uint32_t>(nrows) * kRowBytes);
        // fc_w rows are [embedding half | hidden half]; copy the embedding half of each row.
        for (int r = 0; r < nrows; ++r) {
          mkd_fc::bulk_g2s(
              smem + static_cast<int64_t>(r) * kRowBytes,
              p.fc_w + static_cast<int64_t>(r0 + r) * 2 * kH,
              kRowBytes,
              &bar);
        }
      }
    }
    __syncthreads();
  }
  PDLWaitPrimary<kUsePDL>();
  PDLTriggerSecondary<kUsePDL>();
  if constexpr (!kMemo) {
    mkd_fc::gemma_norm_row<kH>(p.emb, p.norm_w, p.eps, s_x, s_red);
    if (nrows > 0) ptx::mbar_wait_parity(&bar, 0);
  }
  for (int r = warp; r < nrows; r += kWarps) {
    const int j = r0 + r;
    float acc;
    if constexpr (kMemo) {
      acc = p.memo[j];
    } else {
      const auto* row = reinterpret_cast<const bf16_t*>(smem + static_cast<int64_t>(r) * kRowBytes);
      acc = mkd_fc::slice_dot<kW>(row, s_x);
#pragma unroll
      for (int s = 1; s < kEmbed; ++s) acc += mkd_fc::slice_dot<kW>(row + s * kW, s_x + s * kW);
    }
    if (lane == 0) {
#pragma unroll
      for (int s = 0; s < kWorld - kEmbed; ++s) acc += p.hidden_partials[static_cast<int64_t>(s) * kH + j];
      p.out[j] = cast<bf16_t>(acc);
    }
  }
}

struct MkdFcMemoParams {
  const bf16_t* __restrict__ emb;     // [N, kH] embedding rows
  const bf16_t* __restrict__ fc_w;    // [kH, 2 * kH]
  const bf16_t* __restrict__ norm_w;  // [kH]
  float* memo;                        // [N, kH]
  float eps;
  int32_t n;
};

/**
 * \brief Memo rows T[i] = embedding-half partials of emb[i] summed in rank order.
 * A CTA normalizes kB rows into shared memory, then each warp loads one weight-row slice
 * into registers and dots it with all kB rows (the per-row arithmetic is unchanged).
 */
template <int kH, int kWorld, int kB>
SGL_DEVICE void mkd_fc_memo_kernel(const MkdFcMemoParams& p) {
  using namespace device;
  using vec_t = AlignedVector<bf16_t, 8>;
  constexpr int kWarps = 8;
  constexpr int kW = 2 * kH / kWorld;
  constexpr int kEmbed = kH / kW;
  constexpr int kDotVecs = kW / (8 * kWarpThreads);
  extern __shared__ __align__(128) uint8_t smem[];  // [kB, kH] normalized rows
  __shared__ float s_red[kWarps];
  auto* s_x = reinterpret_cast<bf16_t*>(smem);
  const int warp = static_cast<int>(threadIdx.x / kWarpThreads);
  const int lane = static_cast<int>(get_lane_id());
  for (int i0 = static_cast<int>(blockIdx.x) * kB; i0 < p.n; i0 += static_cast<int>(gridDim.x) * kB) {
    const int nb = min(kB, p.n - i0);
    for (int b = 0; b < nb; ++b) {
      mkd_fc::gemma_norm_row<kH>(p.emb + static_cast<int64_t>(i0 + b) * kH, p.norm_w, p.eps, s_x + b * kH, s_red);
    }
    for (int j = warp; j < kH; j += kWarps) {
      const bf16_t* row = p.fc_w + static_cast<int64_t>(j) * 2 * kH;
      float tot[kB];
#pragma unroll
      for (int s = 0; s < kEmbed; ++s) {
        vec_t wv[kDotVecs];
#pragma unroll
        for (int k = 0; k < kDotVecs; ++k) wv[k].load(row + s * kW, k * kWarpThreads + lane);
#pragma unroll
        for (int b = 0; b < kB; ++b) {
          float acc = 0.0f;
          if (b < nb) {
#pragma unroll
            for (int k = 0; k < kDotVecs; ++k) {
              vec_t xv;
              xv.load(s_x + b * kH + s * kW, k * kWarpThreads + lane);
#pragma unroll
              for (int e = 0; e < 8; ++e) acc = fmaf(cast<fp32_t>(wv[k][e]), cast<fp32_t>(xv[e]), acc);
            }
          }
          acc = warp::reduce_sum(acc);
          tot[b] = s == 0 ? acc : tot[b] + acc;
        }
      }
      if (lane == 0) {
        for (int b = 0; b < nb; ++b) p.memo[static_cast<int64_t>(i0 + b) * kH + j] = tot[b];
      }
    }
    __syncthreads();
  }
}

struct MkdHiddenPartialParams {
  const bf16_t* __restrict__ h;       // [kH] hidden state row
  const bf16_t* __restrict__ w;       // [kH, kW] this rank's fc input-column slice (contiguous)
  const bf16_t* __restrict__ norm_w;  // [kH] pre_fc_norm_hidden
  float* out;                         // [kH] fp32 partial
  const uint32_t* epoch;              // tag of the exchange that sends out
  uint32_t* done;                     // [1] CTA arrival counter, zero between calls
  uint32_t* flag;                     // [1] set to the tag once out is complete
  float eps;
  int32_t col;                        // slice start inside the hidden half
};

/**
 * \brief The hidden-slice fc partial of one row (draft_fc_partial arithmetic), then a flag:
 * the step-boundary exchange waits on the flag instead of a stream join, so it keeps its
 * programmatic early launch behind the lm_head.
 */
template <int kH, int kWorld>
SGL_DEVICE void mkd_hidden_partial_kernel(const MkdHiddenPartialParams& p) {
  using namespace device;
  constexpr int kW = 2 * kH / kWorld;
  constexpr int kWarps = 8;
  __shared__ alignas(16) bf16_t s_x[kH];
  __shared__ float s_red[kWarps];
  const int warp = static_cast<int>(threadIdx.x / kWarpThreads);
  const int lane = static_cast<int>(get_lane_id());
  mkd_fc::gemma_norm_row<kH>(p.h, p.norm_w, p.eps, s_x, s_red);
  for (int j = static_cast<int>(blockIdx.x) * kWarps + warp; j < kH; j += static_cast<int>(gridDim.x) * kWarps) {
    const float acc = mkd_fc::slice_dot<kW>(p.w + static_cast<int64_t>(j) * kW, s_x + p.col);
    if (lane == 0) p.out[j] = acc;
  }
  __syncthreads();
  if (threadIdx.x == 0) {
    __threadfence();
    if (atomicAdd(p.done, 1u) == gridDim.x - 1) {
      *p.done = 0u;
      uint32_t e;
      asm volatile("ld.relaxed.gpu.global.u32 %0, [%1];" : "=r"(e) : "l"(p.epoch) : "memory");
      asm volatile("st.release.gpu.global.u32 [%0], %1;" ::"l"(p.flag), "r"(e) : "memory");
    }
  }
}

}  // namespace sglang

using sglang::MkdFcMemoParams;
using sglang::MkdFcSumParams;
using sglang::MkdHiddenPartialParams;

extern "C" __global__ void __launch_bounds__(256, 1) mkd_fc_sum_memo(const MkdFcSumParams __grid_constant__ p) {
  sglang::mkd_fc_sum_kernel<4096, MKD_WORLD, MKD_FC_MAX_ROWS, true, true>(p);
}

extern "C" __global__ void __launch_bounds__(256, 1) mkd_fc_sum_emb(const MkdFcSumParams __grid_constant__ p) {
  sglang::mkd_fc_sum_kernel<4096, MKD_WORLD, MKD_FC_MAX_ROWS, false, true>(p);
}

extern "C" __global__ void __launch_bounds__(256) mkd_fc_memo(const MkdFcMemoParams __grid_constant__ p) {
  sglang::mkd_fc_memo_kernel<4096, MKD_WORLD, 8>(p);
}

// One CTA per SM (SM-count CTAs, or the SMs the draft head leaves it). With 2.2's __launch_bounds__(256)
// ptxas holds NVRTC's build to 32 registers and splits each row's four weight loads into two batches (+24%).
extern "C" __global__ void __launch_bounds__(256, 1) mkd_hidden_partial(const MkdHiddenPartialParams __grid_constant__ p) {
  sglang::mkd_hidden_partial_kernel<4096, MKD_WORLD>(p);
}
