// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// Adapted from SGLang (Apache-2.0): python/sglang/kernels/jit/csrc/speculative/draft_fc_partial.cuh
// (branch nvidia/qwen35-agentx-gb300), device code unchanged; fc_gather is new (see below).
//
// Row-parallel MTP fc for draft-width batches: per token, the Gemma RMSNorm of
// one input half (fp32 statistics, bf16 output as the stock norm stores it),
// then this rank's slice of the fc GEMV with an fp32 result per output row.
// One CTA per SM owns a contiguous block of weight rows; the static rows are
// copied into shared memory before the PDL wait, so they stream in while the
// previous grid finishes.
//
// fc_gather moves the fp32 partials of every rank into the layout ordered_partial_sum reads, in
// place of SGLang's multimem all-gather that 2.2 used (bytes only, no arithmetic). It uses the
// mkd_xchg transport: epoch-tagged 64-bit words through the multicast mapping, three slots
// rotating by epoch, arrival read from the data, bounded waits that trap.
//
// Built with NVRTC (MKD_WORLD = the TP size, 4 or 8; DRAFT_FC_MAX_ROWS = weight rows a CTA,
// draft_fc_partial_rows of 2.2): the kernels are device functions, and the extern "C" entries
// at the end instantiate them for hidden 4096, 4096 fc rows, 1 or 2 tokens, PDL on.

#include "mkd_common.cuh"

#ifndef MKD_WORLD
#define MKD_WORLD 8
#endif
#ifndef DRAFT_FC_MAX_ROWS
#define DRAFT_FC_MAX_ROWS 26
#endif

namespace sglang {

namespace draft_fc {

// global -> shared bulk copy completed on an smem mbarrier (PTX ISA cp.async.bulk);
// size and both addresses must be 16-byte aligned.
SGL_DEVICE void bulk_g2s(void* smem_dst, const void* gmem_src, uint32_t bytes, uint64_t* bar) {
  asm volatile(
      "cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];" ::"r"(
          device::ptx::to_shared(smem_dst)),
      "l"(gmem_src),
      "r"(bytes),
      "r"(device::ptx::to_shared(bar))
      : "memory");
}

}  // namespace draft_fc

struct DraftFcPartialParams {
  const bf16_t* x;                    // [M, H] input half, written by an earlier grid
  const bf16_t* __restrict__ norm_w;  // [H] Gemma norm weight (scale = 1 + w)
  const bf16_t* __restrict__ w;       // [N, W] this rank's fc input-column slice
  float* out;                         // [M, N] fp32 partial
  float eps;
  int32_t col_start;  // first column of the slice inside the half
  int32_t rows;       // M
};

/**
 * \brief Fused RMSNorm + slice GEMV. \tparam kH half width (norm length),
 * \tparam kW slice width, \tparam kN output rows, \tparam kMaxRows weight rows per CTA,
 * \tparam kTokens rows of x (exact). The whole CTA computes the norm statistics of
 * every token in one pass; each warp then holds one weight row in registers and
 * dots it with every normalized token.
 */
template <int kH, int kW, int kN, int kMaxRows, int kTokens, bool kUsePDL>
SGL_DEVICE void draft_fc_partial_kernel(const DraftFcPartialParams& p) {
  using namespace device;
  constexpr int kThreads = 256;
  constexpr int kWarps = kThreads / 32;
  using vec_t = AlignedVector<bf16_t, 8>;
  static_assert(kH % (8 * kThreads) == 0, "norm length must tile the CTA in 16-byte vectors");
  static_assert(kW % (8 * kWarpThreads) == 0, "slice width must tile a warp in 16-byte vectors");
  constexpr int kNormVecs = kH / (8 * kThreads);
  constexpr int kDotVecs = kW / (8 * kWarpThreads);
  constexpr int kRowBytes = kW * static_cast<int>(sizeof(bf16_t));

  extern __shared__ __align__(128) uint8_t smem[];  // [kMaxRows, kW] weight rows
  __shared__ alignas(16) bf16_t s_x[kTokens][kW];
  __shared__ float s_red[kTokens][kWarps];
  __shared__ uint64_t bar;

  const int warp = static_cast<int>(threadIdx.x / kWarpThreads);
  const int lane = static_cast<int>(get_lane_id());
  const int G = static_cast<int>(gridDim.x);
  const int c = static_cast<int>(blockIdx.x);
  const int base = kN / G, rem = kN % G;
  const int r0 = c * base + min(c, rem);
  const int nrows = base + (c < rem ? 1 : 0);

  if (threadIdx.x == 0) {
    ptx::mbar_init(&bar, 1);
    asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
    if (nrows > 0) {
      ptx::mbar_arrive_expect_tx(&bar, static_cast<uint32_t>(nrows) * kRowBytes);
      draft_fc::bulk_g2s(smem, p.w + static_cast<int64_t>(r0) * kW, static_cast<uint32_t>(nrows) * kRowBytes, &bar);
    }
  }
  __syncthreads();
  PDLWaitPrimary<kUsePDL>();

  // Norm statistics of every token over the whole half.
#pragma unroll
  for (int m = 0; m < kTokens; ++m) {
    const bf16_t* xm = p.x + static_cast<int64_t>(m) * kH;
    float ss = 0.0f;
#pragma unroll
    for (int k = 0; k < kNormVecs; ++k) {
      vec_t v;
      v.load(xm, k * kThreads + static_cast<int>(threadIdx.x));
#pragma unroll
      for (int j = 0; j < 8; ++j) {
        const float f = cast<fp32_t>(v[j]);
        ss = fmaf(f, f, ss);
      }
    }
    ss = warp::reduce_sum(ss);
    if (lane == 0) s_red[m][warp] = ss;
  }
  __syncthreads();
  // Normalized slices, rounded to bf16 as the norm output is stored.
#pragma unroll
  for (int m = 0; m < kTokens; ++m) {
    float tot = 0.0f;
#pragma unroll
    for (int w = 0; w < kWarps; ++w) tot += s_red[m][w];
    const float inv = rsqrtf(tot / static_cast<float>(kH) + p.eps);
    const bf16_t* xm = p.x + static_cast<int64_t>(m) * kH + p.col_start;
    for (int k = static_cast<int>(threadIdx.x); k < kW / 8; k += kThreads) {
      vec_t v, g, o;
      v.load(xm, k);
      g.load(p.norm_w + p.col_start, k);
#pragma unroll
      for (int j = 0; j < 8; ++j) {
        o[j] = cast<bf16_t>(cast<fp32_t>(v[j]) * inv * (1.0f + cast<fp32_t>(g[j])));
      }
      o.store(s_x[m], k);
    }
  }
  __syncthreads();
  if (nrows > 0) ptx::mbar_wait_parity(&bar, 0);
  for (int r = warp; r < nrows; r += kWarps) {
    const auto* row = reinterpret_cast<const bf16_t*>(smem + static_cast<int64_t>(r) * kRowBytes);
    vec_t wv[kDotVecs];
#pragma unroll
    for (int k = 0; k < kDotVecs; ++k) wv[k].load(row, k * kWarpThreads + lane);
#pragma unroll
    for (int m = 0; m < kTokens; ++m) {
      float acc = 0.0f;
#pragma unroll
      for (int k = 0; k < kDotVecs; ++k) {
        vec_t xv;
        xv.load(s_x[m], k * kWarpThreads + lane);
#pragma unroll
        for (int j = 0; j < 8; ++j) acc = fmaf(cast<fp32_t>(wv[k][j]), cast<fp32_t>(xv[j]), acc);
      }
      acc = warp::reduce_sum(acc);
      if (lane == 0) p.out[static_cast<int64_t>(m) * kN + r0 + r] = acc;
    }
  }
  PDLTriggerSecondary<kUsePDL>();
}

namespace fc_gather {

inline constexpr int kSlots = 3;
inline constexpr uint32_t kSpinLimit = 1u << 27;
inline constexpr int kMaxTokens = 16;  // the fc split's widest batch (2.2 _FC_SPLIT_MAX_TOKENS)

SGL_DEVICE void mc_store(uint64_t* addr, uint32_t payload, uint32_t tag) {
  const uint64_t w = static_cast<uint64_t>(payload) | (static_cast<uint64_t>(tag) << 32);
  asm volatile("multimem.st.relaxed.sys.global.b64 [%0], %1;" ::"l"(addr), "l"(w) : "memory");
}

}  // namespace fc_gather

struct FcGatherParams {
  const float* partial;  // [m, kH] this rank's fp32 partial, written by an earlier grid
  float* out;            // [m, world * kH]: rank r's row i at columns [r * kH, (r + 1) * kH)
  uint64_t* mc;          // multicast base of the gather buffer [kSlots, world, kMaxTokens, kH]
  const uint64_t* uc;    // this rank's own copy of it
  uint32_t* epoch;       // [1] call counter shared by all ranks, starts at 1
  uint32_t* done;        // [1] CTA arrival counter, zero between calls
  int32_t rank;
  int32_t m;
};

// Every CTA pushes its share of this rank's packet, then polls its share of all packets; no CTA
// waits on another CTA of its rank, so the launch needs no grid barrier (keep the grid resident).
template <int kH, int kWorld>
SGL_DEVICE void fc_gather_kernel(const FcGatherParams& p) {
  constexpr int64_t kPacket = static_cast<int64_t>(fc_gather::kMaxTokens) * kH;
  __shared__ uint32_t s_tag;
  if (threadIdx.x == 0) {
    uint32_t e;
    asm volatile("ld.relaxed.gpu.global.u32 %0, [%1];" : "=r"(e) : "l"(p.epoch) : "memory");
    s_tag = e;
  }
  __syncthreads();
  const uint32_t tag = s_tag;
  const int64_t slot = static_cast<int64_t>(tag % fc_gather::kSlots);
  const int64_t n = static_cast<int64_t>(p.m) * kH;
  const int64_t stride = static_cast<int64_t>(gridDim.x) * blockDim.x;
  const int64_t first = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  uint64_t* dst = p.mc + (slot * kWorld + p.rank) * kPacket;
  for (int64_t i = first; i < n; i += stride) {
    fc_gather::mc_store(dst + i, __float_as_uint(p.partial[i]), tag);
  }
  for (int64_t i = first; i < kWorld * n; i += stride) {
    const int64_t r = i / n, j = i % n;
    const uint64_t* src = p.uc + (slot * kWorld + r) * kPacket + j;
    uint64_t w;
    uint32_t spins = 0;
    do {
      asm volatile("ld.relaxed.sys.global.u64 %0, [%1];" : "=l"(w) : "l"(src) : "memory");
      if (++spins > fc_gather::kSpinLimit) __trap();
    } while (static_cast<uint32_t>(w >> 32) != tag);
    p.out[(j / kH) * kWorld * kH + r * kH + j % kH] = __uint_as_float(static_cast<uint32_t>(w));
  }
  __syncthreads();
  if (threadIdx.x == 0) {
    __threadfence();
    if (atomicAdd(p.done, 1u) == gridDim.x - 1) {
      *p.done = 0u;
      *p.epoch = tag + 1;
    }
  }
}


}  // namespace sglang

using sglang::DraftFcPartialParams;
using sglang::FcGatherParams;

extern "C" __global__ void __launch_bounds__(256, 1) draft_fc_partial_m1(const DraftFcPartialParams __grid_constant__ p) {
  sglang::draft_fc_partial_kernel<4096, 2 * 4096 / MKD_WORLD, 4096, DRAFT_FC_MAX_ROWS, 1, true>(p);
}

extern "C" __global__ void __launch_bounds__(256, 1) draft_fc_partial_m2(const DraftFcPartialParams __grid_constant__ p) {
  sglang::draft_fc_partial_kernel<4096, 2 * 4096 / MKD_WORLD, 4096, DRAFT_FC_MAX_ROWS, 2, true>(p);
}

extern "C" __global__ void __launch_bounds__(512) fc_gather(const FcGatherParams __grid_constant__ p) {
  sglang::fc_gather_kernel<4096, MKD_WORLD>(p);
}
