// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
//
// Weight-streaming BF16 GEMV for the DFlash draft layers:  out[M, N] = x[M, K] @ w[N, K]^T  with M <= kTokTile <= 64,
// with an optional fused epilogue.
//
// Work split: a tile is kRows weight rows taken as kBoxes blocks of kRows / kBoxes rows; block b of tile t is weight
// rows [block_off[b] + t * kRows / kBoxes, ... + kRows / kBoxes). One thread-block cluster of kCluster CTAs handles a
// tile; cluster rank r streams the r-th of kCluster contiguous, near-equal parts of the K range (steps of kStepK).
//
// Data movement: one producer thread copies each step's weight blocks and activation tile into a ring of kStages
// shared-memory stages with tensor copies (cp.async.bulk.tensor.3d, completion counted in bytes on an mbarrier;
// activation rows past M are zero-filled). Tiles are stored as 64-column slabs of 128-byte lines with the 128-byte
// swizzle (each weight block slab-major inside its own 1024-byte aligned region), so ldmatrix reads are bank-conflict
// free. Each of kWarps consumer warps owns kRows / kWarps rows of the tile and runs mma.sync.m16n8k16 (bf16 inputs,
// fp32 accumulators); it releases a stage once its fragments are in registers (PTX ISA: cp.async.bulk.tensor,
// mbarrier, ldmatrix, mma).
//
// Rounding points: bf16 operands, fp32 accumulation, fp32 partial sums, one bf16 rounding per GEMV output element
// (the same rounding points as the stock dense path; the fp32 summation order differs).
//
// Reduction across the cluster (kCluster > 1): every thread arrives on the cluster barrier at kernel start and waits on
// that phase after its main loop, so all CTAs of the cluster run before any shared::cluster access. Ranks r > 0 push
// their fp32 partial accumulators into rank 0's shared memory (st.shared::cluster) and arrive with release; rank 0
// waits with acquire and adds the partials in rank (K) order (PTX ISA: barrier.cluster, memory consistency model).
// The summation order depends only on the shapes and the tactic, so results are bitwise reproducible.
//
// Epilogues (rank 0, on the bf16-rounded tile staged in shared memory):
//   kEpiPlain:    store the tile.
//   kEpiConvIn:   the tile holds, for kRows / 4 groups of 16 channels, the four coefficient sets of a two-tap block
//                 convolution (blocks: input side tap 0 / tap 1, output side tap 0 / tap 1). Writes the input-side
//                 convolution of x, conv_out[m, c] = bf16(fma(base0[0, c] + k00, x[m, c], (base0[1, c] + k01) *
//                 x[m - 1, c])) with x[m - 1] = 0 for the first token of each 8-token block, and the output-side
//                 coefficients into coef_out at their positions in the [M, 4 * groups] coefficient layout.
//   kEpiSwiGLU:   the tile holds kRows / 2 gate rows and the matching up rows; writes
//                 act_out[m, i] = bf16(silu(gate) * up) with silu(g) = g / (1 + exp(-g)) in fp32.
//
// Stage reuse: the producer refills a stage only after every consumer warp released it (mbarrier) and after a
// generic-to-async proxy fence, since the consumers read stages with ldmatrix and the refill is a tensor copy (PTX ISA:
// fence.proxy.async). Without the fence some tile shapes produced run-to-run different outputs.
//
// PDL: weight copies for the first stages are issued before griddepcontrol.wait (no kernel writes the weights);
// activation copies (after a generic-to-async proxy fence), activation reads and every global store come after it.
//
// Waits are bounded: an mbarrier wait that exceeds kSpinLimit polls traps instead of hanging.

#pragma once

#include <sgl_kernel/tensor.h>
#include <sgl_kernel/utils.h>

#include <sgl_kernel/runtime.cuh>
#include <sgl_kernel/utils.cuh>

#include <tvm/ffi/container/tensor.h>

#include <cuda.h>
#include <cuda_bf16.h>
#include <algorithm>
#include <cstdint>

namespace muse_vr::draft_gemv {

using namespace sglang;

inline constexpr int kEpiPlain = 0;
inline constexpr int kEpiConvIn = 1;
inline constexpr int kEpiSwiGLU = 2;

struct alignas(64) Params {
  CUtensorMap w_map;         // w [N, K] viewed as [K/64][N][64]
  CUtensorMap x_map;         // x [m, K] viewed as [K/64][m][64]
  bf16_t* __restrict__ out;  // plain: [m, N], row stride ldo; swiglu: [m, N / 2], row stride ldo
  int64_t ldo;
  int m;
  int n;
  int steps_per_tile;  // K / kStepK
  int block_off[4];    // weight row of each block of tile 0
  // kEpiConvIn
  const bf16_t* __restrict__ xin;  // [m, hidden] activation (the GEMV input), row stride ldx
  int64_t ldx;
  const bf16_t* __restrict__ base0;  // [2, hidden] input-side base kernel
  bf16_t* __restrict__ conv_out;     // [m, hidden]
  bf16_t* __restrict__ coef_out;     // [m, 4 * groups]
  int groups;                        // channel groups (hidden / 16)
};

inline constexpr uint32_t kSpinLimit = 1u << 30;


template <int kTokTile, int kRows, int kStepK, int kStages, int kWarps, int kCluster, int kBoxes = 1>
struct Cfg {
  static constexpr int kBoxRows = kRows / kBoxes;
  static constexpr int kSlabs = kStepK / 64;  // 64-column (128-byte) slabs per stage
  static constexpr int kWStageBytes = kSlabs * kRows * 128;
  static constexpr int kXStageBytes = kSlabs * kTokTile * 128;
  static constexpr int kStageBytes = kWStageBytes + kXStageBytes;
  static constexpr int kTileElems = kTokTile * kRows;
  static constexpr int kStageOffset = 1024;  // barriers live below
  static constexpr int kPartOffset = kStageOffset + kStages * kStageBytes;
  static constexpr int kPartBytes = (kCluster - 1) * kTileElems * 4;  // peers' partial tiles, fragment order
  static constexpr int kSmemBytes = kPartOffset + kPartBytes + 1024;  // + realignment slack
  static constexpr int kThreads = (kWarps + 1) * 32;
  static constexpr int kConsumerThreads = kWarps * 32;
  static constexpr int kMTiles = kRows / 16;
  static constexpr int kMTilesPerWarp = kMTiles / kWarps;
  static constexpr int kNTiles = kTokTile / 8;
  static constexpr int kK16 = kStepK / 16;
  // Fragment registers per k16 sub-step: 4 per m16 tile (A) and 2 per n8 tile (B).
  static constexpr int kFragRegs = 4 * kMTilesPerWarp + 2 * kNTiles;
  static constexpr int kChunk = kFragRegs * kK16 <= 96 ? kK16 : (kFragRegs * 4 <= 96 ? 4 : 2);
  static constexpr int kAcc = kMTilesPerWarp * kNTiles <= 2 ? 2 : 1;
  static_assert(kTokTile == 8 || kTokTile == 16 || kTokTile == 32 || kTokTile == 64, "token tile");
  static_assert(kRows % 16 == 0 && kRows <= 256, "rows per tile");
  static_assert(kRows % kBoxes == 0 && kBoxRows % 8 == 0, "weight blocks must be whole 1024-byte line groups");
  static_assert(kMTiles % kWarps == 0, "consumer warps must split the tile rows evenly");
  static_assert(kStepK % 64 == 0 && kSlabs <= 256, "step");
  static_assert(kCluster >= 1 && kCluster <= 8, "cluster size");
  static_assert(2 * kStages * 8 <= kStageOffset, "barrier area");
  static_assert(kStageBytes % 1024 == 0, "stages must stay 1024-byte aligned for the 128-byte swizzle");
};

SGL_DEVICE uint32_t smem_u32(const void* p) {
  return static_cast<uint32_t>(__cvta_generic_to_shared(p));
}

SGL_DEVICE void mbar_init(uint32_t bar, uint32_t count) {
  asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" ::"r"(bar), "r"(count) : "memory");
}

SGL_DEVICE void mbar_arrive_expect_tx(uint32_t bar, uint32_t bytes) {
  asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;" ::"r"(bar), "r"(bytes) : "memory");
}

SGL_DEVICE void mbar_arrive(uint32_t bar) {
  asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];" ::"r"(bar) : "memory");
}

SGL_DEVICE bool mbar_try_wait(uint32_t bar, uint32_t parity) {
  uint32_t ok;
  asm volatile(
      "{\n\t.reg .pred p;\n\t"
      "mbarrier.try_wait.parity.shared::cta.b64 p, [%1], %2;\n\t"
      "selp.u32 %0, 1, 0, p;\n\t}"
      : "=r"(ok)
      : "r"(bar), "r"(parity)
      : "memory");
  return ok != 0;
}

SGL_DEVICE void mbar_wait(uint32_t bar, uint32_t parity) {
  uint32_t spins = 0;
  while (!mbar_try_wait(bar, parity)) {
    if (++spins == kSpinLimit) __trap();
  }
}

SGL_DEVICE void tma_load_3d(uint32_t dst, const CUtensorMap* map, int c0, int c1, int c2, uint32_t bar) {
  asm volatile(
      "cp.async.bulk.tensor.3d.shared::cluster.global.tile.mbarrier::complete_tx::bytes"
      " [%0], [%1, {%2, %3, %4}], [%5];" ::"r"(dst),
      "l"(reinterpret_cast<uint64_t>(map)),
      "r"(c0),
      "r"(c1),
      "r"(c2),
      "r"(bar)
      : "memory");
}

SGL_DEVICE void tma_load_3d_hint(
    uint32_t dst, const CUtensorMap* map, int c0, int c1, int c2, uint32_t bar, uint64_t policy) {
  asm volatile(
      "cp.async.bulk.tensor.3d.shared::cluster.global.tile.mbarrier::complete_tx::bytes.L2::cache_hint"
      " [%0], [%1, {%2, %3, %4}], [%5], %6;" ::"r"(dst),
      "l"(reinterpret_cast<uint64_t>(map)),
      "r"(c0),
      "r"(c1),
      "r"(c2),
      "r"(bar),
      "l"(policy)
      : "memory");
}

// Byte offset of 16-byte chunk `chunk` (0..7) of slab `slab`, row `row` in a stage tile of kTileRows rows stored
// slab-major as 128-byte-swizzled lines (line = slab * kTileRows + row).
template <int kTileRows>
SGL_DEVICE uint32_t tile_off(int row, int slab, int chunk) {
  const int line = slab * kTileRows + row;
  return static_cast<uint32_t>(line * 128 + ((chunk ^ (line & 7)) << 4));
}

// Byte offset of 16-byte chunk `chunk` of slab `slab`, tile row `row` in a weight stage made of blocks of kBoxRows
// rows, each block stored slab-major (line = block * kSlabs * kBoxRows + slab * kBoxRows + row in block).
template <int kBoxRows, int kSlabs>
SGL_DEVICE uint32_t w_off(int row, int slab, int chunk) {
  const int line = (row / kBoxRows) * (kSlabs * kBoxRows) + slab * kBoxRows + (row % kBoxRows);
  return static_cast<uint32_t>(line * 128 + ((chunk ^ (line & 7)) << 4));
}

SGL_DEVICE void ldsm_x4(uint32_t addr, uint32_t& r0, uint32_t& r1, uint32_t& r2, uint32_t& r3) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];"
               : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3)
               : "r"(addr));
}

SGL_DEVICE void ldsm_x2(uint32_t addr, uint32_t& r0, uint32_t& r1) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0, %1}, [%2];" : "=r"(r0), "=r"(r1) : "r"(addr));
}

SGL_DEVICE void mma_bf16_16816(float* d, const uint32_t* a, const uint32_t* b) {
  asm volatile(
      "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, "
      "{%0, %1, %2, %3};"
      : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
}

SGL_DEVICE uint32_t cluster_rank() {
  uint32_t r;
  asm volatile("mov.u32 %0, %%cluster_ctarank;" : "=r"(r));
  return r;
}

SGL_DEVICE void cluster_arrive_relaxed() {
  asm volatile("barrier.cluster.arrive.relaxed.aligned;" ::: "memory");
}

SGL_DEVICE void cluster_wait() {
  asm volatile("barrier.cluster.wait.aligned;" ::: "memory");
}

SGL_DEVICE void cluster_arrive_release() {
  asm volatile("barrier.cluster.arrive.release.aligned;" ::: "memory");
}

SGL_DEVICE void cluster_wait_acquire() {
  asm volatile("barrier.cluster.wait.acquire.aligned;" ::: "memory");
}

// Store four floats to the shared memory of cluster CTA `rank` at the address `local_addr` has in this CTA.
SGL_DEVICE void st_shared_cluster_v4(uint32_t local_addr, uint32_t rank, const float* v) {
  uint32_t remote;
  asm volatile("mapa.shared::cluster.u32 %0, %1, %2;" : "=r"(remote) : "r"(local_addr), "r"(rank));
  asm volatile("st.shared::cluster.v4.f32 [%0], {%1, %2, %3, %4};" ::"r"(remote), "f"(v[0]), "f"(v[1]), "f"(v[2]),
               "f"(v[3])
               : "memory");
}

template <bool kUsePDL>
SGL_DEVICE void pdl_wait() {
  if constexpr (kUsePDL) asm volatile("griddepcontrol.wait;" ::: "memory");
}

template <bool kUsePDL>
SGL_DEVICE void pdl_trigger() {
  if constexpr (kUsePDL) asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
}

// Weight row of tile row `r` (block r / kBoxRows).
template <typename C>
SGL_DEVICE int weight_row(const Params& p, int tile, int r) {
  return p.block_off[r / C::kBoxRows] + tile * C::kBoxRows + (r % C::kBoxRows);
}

SGL_DEVICE float bf16_bits(uint32_t v16) {
  return __uint_as_float(v16 << 16);
}

SGL_DEVICE float silu_f32(float g) {
  return g / (1.0f + expf(-g));
}

// Consumer warps only (named barrier 1, kWarps warps). `acc` is the warp's final fp32 tile fragment; `scratch` is free
// shared memory (the stage ring, no longer read once the main loop has ended).
template <typename C, int kEpi>
SGL_DEVICE void epilogue(const Params& p, float (&acc)[C::kMTilesPerWarp][C::kNTiles][4], int tile, int row_base,
                         int lane, uint8_t* scratch) {
  constexpr int kTokTile = C::kNTiles * 8;
  if constexpr (kEpi == kEpiPlain) {
#pragma unroll
    for (int mt = 0; mt < C::kMTilesPerWarp; ++mt)
#pragma unroll
      for (int nt = 0; nt < C::kNTiles; ++nt)
#pragma unroll
        for (int q = 0; q < 4; ++q) {
          const int row = row_base + mt * 16 + (lane >> 2) + 8 * (q >> 1);
          const int tok = nt * 8 + 2 * (lane & 3) + (q & 1);
          const int n = weight_row<C>(p, tile, row);
          if (tok < p.m && n < p.n) p.out[tok * p.ldo + n] = __float2bfloat16_rn(acc[mt][nt][q]);
        }
    return;
  } else {
    // stage the bf16-rounded tile as [row][token] (bf16 bits) in the ring's first bytes: every consumer warp must be
    // done reading its last stage before any warp overwrites it
    asm volatile("bar.sync 1, %0;" ::"r"(C::kConsumerThreads) : "memory");
    uint16_t* t16 = reinterpret_cast<uint16_t*>(scratch);
#pragma unroll
    for (int mt = 0; mt < C::kMTilesPerWarp; ++mt)
#pragma unroll
      for (int nt = 0; nt < C::kNTiles; ++nt)
#pragma unroll
        for (int q = 0; q < 4; ++q) {
          const int row = row_base + mt * 16 + (lane >> 2) + 8 * (q >> 1);
          const int tok = nt * 8 + 2 * (lane & 3) + (q & 1);
          t16[row * kTokTile + tok] = __bfloat16_as_ushort(__float2bfloat16_rn(acc[mt][nt][q]));
        }
    asm volatile("bar.sync 1, %0;" ::"r"(C::kConsumerThreads) : "memory");
    const int tid = static_cast<int>(threadIdx.x);
    if constexpr (kEpi == kEpiSwiGLU) {
      // rows [0, kBoxRows) gate, [kBoxRows, 2 kBoxRows) up, intermediate index tile * kBoxRows + i
      for (int e = tid; e < p.m * C::kBoxRows; e += C::kConsumerThreads) {
        const int tok = e / C::kBoxRows, i = e % C::kBoxRows;
        if (tile * C::kBoxRows + i >= p.n / 2) continue;  // partial last tile
        const float g = bf16_bits(t16[i * kTokTile + tok]);
        const float u = bf16_bits(t16[(C::kBoxRows + i) * kTokTile + tok]);
        p.out[tok * p.ldo + tile * C::kBoxRows + i] = __float2bfloat16_rn(silu_f32(g) * u);
      }
    } else {
      // rows: block 0 input tap 0, block 1 input tap 1, block 2 output tap 0, block 3 output tap 1;
      // group = tile * kBoxRows + i, channels [16 group, 16 group + 16)
      constexpr int kVecs = C::kBoxRows * 2;  // 8-channel vectors per token in this tile
      for (int e = tid; e < p.m * kVecs; e += C::kConsumerThreads) {
        const int tok = e / kVecs, v = e % kVecs;
        const int i = v / 2;
        const int ch = (tile * C::kBoxRows + i) * 16 + (v % 2) * 8;
        const float k00 = bf16_bits(t16[i * kTokTile + tok]);
        const float k01 = bf16_bits(t16[(C::kBoxRows + i) * kTokTile + tok]);
        const uint4 x4 = *reinterpret_cast<const uint4*>(p.xin + tok * p.ldx + ch);
        uint4 xp4 = make_uint4(0u, 0u, 0u, 0u);
        const bool has_prev = (tok & 7) >= 1;
        if (has_prev) xp4 = *reinterpret_cast<const uint4*>(p.xin + (tok - 1) * p.ldx + ch);
        const uint4 b04 = *reinterpret_cast<const uint4*>(p.base0 + ch);
        const uint4 b14 = *reinterpret_cast<const uint4*>(p.base0 + p.groups * 16 + ch);
        uint32_t o[4];
#pragma unroll
        for (int k = 0; k < 8; ++k) {
          const int w = k / 2, sh = (k % 2) * 16;
          const float c0 = __fadd_rn(bf16_bits(((&b04.x)[w] >> sh) & 0xffffu), k00);
          const float c1 = __fadd_rn(bf16_bits(((&b14.x)[w] >> sh) & 0xffffu), k01);
          const float xv = bf16_bits(((&x4.x)[w] >> sh) & 0xffffu);
          const float pv = has_prev ? bf16_bits(((&xp4.x)[w] >> sh) & 0xffffu) : 0.0f;
          const uint32_t y = __bfloat16_as_ushort(__float2bfloat16_rn(__fmaf_rn(c0, xv, __fmul_rn(c1, pv))));
          if (k % 2 == 0)
            o[w] = y;
          else
            o[w] |= y << 16;
        }
        *reinterpret_cast<uint4*>(p.conv_out + tok * p.ldx + ch) = make_uint4(o[0], o[1], o[2], o[3]);
      }
      for (int e = tid; e < p.m * 2 * C::kBoxRows; e += C::kConsumerThreads) {
        const int tok = e / (2 * C::kBoxRows), r = e % (2 * C::kBoxRows);
        const int tap = r / C::kBoxRows, i = r % C::kBoxRows;
        p.coef_out[tok * (4 * p.groups) + 2 * p.groups + tap * p.groups + tile * C::kBoxRows + i] =
            __ushort_as_bfloat16(t16[(2 * C::kBoxRows + r) * kTokTile + tok]);
      }
    }
  }
}

template <int kTokTile, int kRows, int kStepK, int kStages, int kWarps, int kCluster, int kBoxes, int kEpi,
          bool kUsePDL>
__global__ void __launch_bounds__(Cfg<kTokTile, kRows, kStepK, kStages, kWarps, kCluster, kBoxes>::kThreads, 1)
    draft_gemv_kernel(const __grid_constant__ Params p) {
  using C = Cfg<kTokTile, kRows, kStepK, kStages, kWarps, kCluster, kBoxes>;
  constexpr bool kEvictFirst = false;
  extern __shared__ __align__(1024) uint8_t smem_raw[];
  // Swizzled tensor copies need 1024-byte aligned stages: realign the dynamic window explicitly. The offset is
  // the same in every CTA of the kernel, so cluster peers find each other's partial tile at the same address.
  uint8_t* smem = smem_raw + ((1024 - (smem_u32(smem_raw) & 1023)) & 1023);
  uint64_t* full = reinterpret_cast<uint64_t*>(smem);
  uint64_t* empty = full + kStages;
  uint8_t* stages = smem + C::kStageOffset;
  float* part = reinterpret_cast<float*>(smem + C::kPartOffset);  // [token][row], kCluster > 1 only

  const int warp = threadIdx.x / 32;
  const int lane = threadIdx.x % 32;
  const int rank = kCluster > 1 ? static_cast<int>(cluster_rank()) : 0;
  const int tile = static_cast<int>(blockIdx.x) / kCluster;
  const int ks = p.steps_per_tile;
  const int s_begin = rank * ks / kCluster;
  const int nsteps = (rank + 1) * ks / kCluster - s_begin;

  if (warp == kWarps && lane == 0) {
    // Fetch both tensor-map descriptors early (PTX ISA: prefetch.tensormap).
    asm volatile("prefetch.tensormap [%0];" ::"l"(reinterpret_cast<uint64_t>(&p.w_map)) : "memory");
    asm volatile("prefetch.tensormap [%0];" ::"l"(reinterpret_cast<uint64_t>(&p.x_map)) : "memory");
  }
  if (threadIdx.x == 0) {
#pragma unroll
    for (int s = 0; s < kStages; ++s) {
      mbar_init(smem_u32(&full[s]), 1);
      mbar_init(smem_u32(&empty[s]), kWarps);
    }
    asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
  }
  __syncthreads();
  // First cluster barrier phase: arrive now, wait before any shared::cluster access, so every CTA of the cluster
  // is known to be running before a peer writes into rank 0's shared memory.
  if constexpr (kCluster > 1) cluster_arrive_relaxed();

  if (warp == kWarps) {
    // ---------------- producer warp: one thread issues every copy ----------------
    if (lane == 0) {
      uint64_t policy = 0;
      if constexpr (kEvictFirst) {
        asm volatile("createpolicy.fractional.L2::evict_first.b64 %0, 1.0;" : "=l"(policy));
      }
      auto issue_w = [&](int i) {
        const int s = i % kStages;
        const uint32_t bar = smem_u32(&full[s]);
        mbar_arrive_expect_tx(bar, C::kStageBytes);
        const uint32_t dst = smem_u32(stages + s * C::kStageBytes);
        const int slab0 = (s_begin + i) * C::kSlabs;
#pragma unroll
        for (int b = 0; b < kBoxes; ++b) {
          const uint32_t d = dst + b * (C::kSlabs * C::kBoxRows * 128);
          const int r0 = p.block_off[b] + tile * C::kBoxRows;
          if constexpr (kEvictFirst) {
            tma_load_3d_hint(d, &p.w_map, 0, r0, slab0, bar, policy);
          } else {
            tma_load_3d(d, &p.w_map, 0, r0, slab0, bar);
          }
        }
      };
      auto issue_x = [&](int i) {
        const int s = i % kStages;
        tma_load_3d(smem_u32(stages + s * C::kStageBytes + C::kWStageBytes), &p.x_map, 0, 0,
                    (s_begin + i) * C::kSlabs, smem_u32(&full[s]));
      };
      const int pre = nsteps < kStages ? nsteps : kStages;
      for (int i = 0; i < pre; ++i) issue_w(i);
      pdl_wait<kUsePDL>();
      // the activation was written by a preceding kernel through the generic proxy and is read here by tensor
      // copies (async proxy): order the two proxies (PTX ISA: fence.proxy.async)
      asm volatile("fence.proxy.async.global;" ::: "memory");
      for (int i = 0; i < pre; ++i) issue_x(i);
      for (int i = pre; i < nsteps; ++i) {
        const int s = i % kStages;
        mbar_wait(smem_u32(&empty[s]), ((i / kStages) & 1) ^ 1);
        // the consumers read this stage through the generic proxy (ldmatrix); the refill writes it through the async proxy:
        // order the two proxies before reusing the stage (PTX ISA: fence.proxy.async)
        asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
        issue_w(i);
        issue_x(i);
      }
      pdl_trigger<kUsePDL>();
    }
    __syncwarp();
    if constexpr (kCluster > 1) {
      cluster_wait();
      cluster_arrive_release();
    }
    return;
  }

  // ---------------- consumer warps ----------------
  pdl_wait<kUsePDL>();
  // kAcc independent accumulator sets (k16 sub-steps alternate between them) shorten the dependent mma chain
  // when a warp has few output fragments; they are added in set order at the end.
  float acc[C::kAcc][C::kMTilesPerWarp][C::kNTiles][4];
#pragma unroll
  for (int u = 0; u < C::kAcc; ++u)
#pragma unroll
    for (int mt = 0; mt < C::kMTilesPerWarp; ++mt)
#pragma unroll
      for (int nt = 0; nt < C::kNTiles; ++nt)
#pragma unroll
        for (int q = 0; q < 4; ++q) acc[u][mt][nt][q] = 0.f;

  const int row_base = warp * C::kMTilesPerWarp * 16;
  for (int i = 0; i < nsteps; ++i) {
    const int s = i % kStages;
    mbar_wait(smem_u32(&full[s]), (i / kStages) & 1);
    const uint32_t wbase = smem_u32(stages + s * C::kStageBytes);
    const uint32_t xbase = wbase + C::kWStageBytes;
    // The stage is read in chunks of kChunk k16 sub-steps: fragments of a chunk are loaded into registers first,
    // and the stage is released as soon as the last chunk's fragments are loaded, before its mma instructions.
#pragma unroll
    for (int j0 = 0; j0 < C::kK16; j0 += C::kChunk) {
      uint32_t a[C::kChunk][C::kMTilesPerWarp][4];
      uint32_t b[C::kChunk][C::kNTiles][2];
#pragma unroll
      for (int jj = 0; jj < C::kChunk; ++jj) {
        const int j = j0 + jj;
        const int slab = j >> 2;
        const int chunk = (j & 3) * 2;  // first 16-byte chunk of this k16 in the slab row
        if constexpr (C::kNTiles == 1) {
          const int l = lane & 15;
          ldsm_x2(xbase + tile_off<kTokTile>(l & 7, slab, chunk + (l >> 3)), b[jj][0][0], b[jj][0][1]);
        } else {
#pragma unroll
          for (int nt = 0; nt < C::kNTiles; nt += 2) {
            const int tok = nt * 8 + (lane & 7) + ((lane >> 4) << 3);
            ldsm_x4(xbase + tile_off<kTokTile>(tok, slab, chunk + ((lane >> 3) & 1)), b[jj][nt][0], b[jj][nt][1],
                    b[jj][nt + 1][0], b[jj][nt + 1][1]);
          }
        }
#pragma unroll
        for (int mt = 0; mt < C::kMTilesPerWarp; ++mt) {
          const int row = row_base + mt * 16 + (lane & 7) + (((lane >> 3) & 1) << 3);
          ldsm_x4(wbase + w_off<C::kBoxRows, C::kSlabs>(row, slab, chunk + (lane >> 4)), a[jj][mt][0], a[jj][mt][1],
                  a[jj][mt][2], a[jj][mt][3]);
        }
      }
      if (j0 + C::kChunk == C::kK16) {
        __syncwarp();
        if (lane == 0) mbar_arrive(smem_u32(&empty[s]));
      }
#pragma unroll
      for (int jj = 0; jj < C::kChunk; ++jj) {
        const int u = (j0 + jj) % C::kAcc;
#pragma unroll
        for (int mt = 0; mt < C::kMTilesPerWarp; ++mt)
#pragma unroll
          for (int nt = 0; nt < C::kNTiles; ++nt) mma_bf16_16816(acc[u][mt][nt], a[jj][mt], b[jj][nt]);
      }
    }
  }
#pragma unroll
  for (int u = 1; u < C::kAcc; ++u)
#pragma unroll
    for (int mt = 0; mt < C::kMTilesPerWarp; ++mt)
#pragma unroll
      for (int nt = 0; nt < C::kNTiles; ++nt)
#pragma unroll
        for (int q = 0; q < 4; ++q) acc[0][mt][nt][q] += acc[u][mt][nt][q];

  // Fragment layout of m16n8 accumulators: acc[q] holds row (lane / 4) + 8 * (q / 2), token 2 * (lane % 4) + q % 2.
  if constexpr (kCluster == 1) {
    epilogue<C, kEpi>(p, acc[0], tile, row_base, lane, stages);
  } else {
    // Partial tiles of ranks 1.. are pushed into rank 0's shared memory in fragment order (one float4 per thread
    // and fragment); rank 0 adds them to its own accumulators in rank order and stores the tile.
    const int frag_base = (warp * 32 + lane) * C::kMTilesPerWarp * C::kNTiles;
    cluster_wait();  // first phase: every CTA of the cluster has started
    if (rank != 0) {
#pragma unroll
      for (int mt = 0; mt < C::kMTilesPerWarp; ++mt)
#pragma unroll
        for (int nt = 0; nt < C::kNTiles; ++nt) {
          float* slot = part + ((rank - 1) * C::kTileElems / 4 + frag_base + mt * C::kNTiles + nt) * 4;
          st_shared_cluster_v4(smem_u32(slot), 0, acc[0][mt][nt]);
        }
      cluster_arrive_release();  // release: the pushed partials are visible to rank 0's acquire
    } else {
      cluster_arrive_release();
      cluster_wait_acquire();  // every peer has pushed its partial tile
#pragma unroll
      for (int r = 1; r < kCluster; ++r)
#pragma unroll
        for (int mt = 0; mt < C::kMTilesPerWarp; ++mt)
#pragma unroll
          for (int nt = 0; nt < C::kNTiles; ++nt) {
            const float4 v = *reinterpret_cast<const float4*>(
                part + ((r - 1) * C::kTileElems / 4 + frag_base + mt * C::kNTiles + nt) * 4);
            acc[0][mt][nt][0] += v.x;
            acc[0][mt][nt][1] += v.y;
            acc[0][mt][nt][2] += v.z;
            acc[0][mt][nt][3] += v.w;
          }
      epilogue<C, kEpi>(p, acc[0], tile, row_base, lane, stages);
    }
  }
}

template <int kTokTile, int kRows, int kStepK, int kStages, int kWarps, int kCluster, int kBoxes, int kEpi, bool kUsePDL>
struct DraftGemv {
  using C = Cfg<kTokTile, kRows, kStepK, kStages, kWarps, kCluster, kBoxes>;

  using EncodeTiled = CUresult (*)(CUtensorMap*, CUtensorMapDataType, cuuint32_t, void*, const cuuint64_t*,
                                   const cuuint64_t*, const cuuint32_t*, const cuuint32_t*, CUtensorMapInterleave,
                                   CUtensorMapSwizzle, CUtensorMapL2promotion, CUtensorMapFloatOOBfill);

  static EncodeTiled encoder() {
    static EncodeTiled fn = [] {
      void* ptr = nullptr;
      cudaDriverEntryPointQueryResult q{};
      host::RuntimeDeviceCheck(
          cudaGetDriverEntryPointByVersion("cuTensorMapEncodeTiled", &ptr, 12000, cudaEnableDefault, &q));
      host::RuntimeCheck(ptr != nullptr && q == cudaDriverEntryPointSuccess, "draft_gemv: no cuTensorMapEncodeTiled");
      return reinterpret_cast<EncodeTiled>(ptr);
    }();
    return fn;
  }

  // A [rows, k] bf16 matrix with row pitch `pitch_bytes`, viewed as [k/64][rows][64] so that one box of
  // {64, box_rows, kSlabs} is kSlabs swizzled slabs. Rows past `rows` are zero-filled.
  static void encode_map(CUtensorMap* map, const void* base, uint64_t rows, uint64_t k, int64_t pitch_bytes,
                         int box_rows) {
    const cuuint64_t dims[3] = {64, rows, k / 64};
    const cuuint64_t strides[2] = {static_cast<cuuint64_t>(pitch_bytes), 128};
    const cuuint32_t box[3] = {64, static_cast<cuuint32_t>(box_rows), static_cast<cuuint32_t>(C::kSlabs)};
    const cuuint32_t elem_strides[3] = {1, 1, 1};
    const CUresult res =
        encoder()(map, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 3, const_cast<void*>(base), dims, strides, box, elem_strides,
                  CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_L2_256B,
                  CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
    host::RuntimeCheck(res == CUDA_SUCCESS, "draft_gemv: cuTensorMapEncodeTiled failed (", static_cast<int>(res), ")");
  }

  // Shared setup; `rows_per_block` is the weight rows of one block (N for kBoxes == 1).
  static void launch(Params& params, const tvm::ffi::TensorView x, const tvm::ffi::TensorView w,
                     host::SymbolicDevice& device_, int64_t& m_out) {
    using namespace host;
    auto M = SymbolicSize{"m"};
    auto N = SymbolicSize{"n"};
    auto K = SymbolicSize{"k"};
    auto LDX = SymbolicSize{"ldx"};
    device_.set_options<kDLCUDA>();
    TensorMatcher({M, K}).with_strides({LDX, 1}).with_dtype<bf16_t>().with_device(device_).verify(x);
    TensorMatcher({N, K}).with_strides({K, 1}).with_dtype<bf16_t>().with_device(device_).verify(w);
    const int64_t m = M.unwrap(), n = N.unwrap(), k = K.unwrap();
    m_out = m;
    RuntimeCheck(m >= 1 && m <= kTokTile, "draft_gemv: m ", m, " outside 1..", kTokTile);
    RuntimeCheck(k % kStepK == 0, "draft_gemv: k ", k, " not a multiple of ", kStepK);
    RuntimeCheck(k / kStepK >= kCluster, "draft_gemv: fewer K steps than cluster ranks");
    RuntimeCheck(n % kBoxes == 0, "draft_gemv: n must split into kBoxes equal blocks");
    RuntimeCheck(kEpi != kEpiConvIn || (n / kBoxes) % C::kBoxRows == 0, "draft_gemv: groups must fill whole tiles");
    RuntimeCheck(LDX.unwrap() % 8 == 0, "draft_gemv: x row stride must be a multiple of 8 elements");
    RuntimeCheck(reinterpret_cast<uintptr_t>(x.data_ptr()) % 16 == 0, "draft_gemv: x not 16-byte aligned");
    RuntimeCheck(reinterpret_cast<uintptr_t>(w.data_ptr()) % 16 == 0, "draft_gemv: w not 16-byte aligned");
    const int64_t rows_per_block = n / kBoxes;
    for (int b = 0; b < 4; ++b) params.block_off[b] = b < kBoxes ? static_cast<int>(b * rows_per_block) : 0;

    const auto dev = device_.unwrap();
    auto kernel = draft_gemv_kernel<kTokTile, kRows, kStepK, kStages, kWarps, kCluster, kBoxes, kEpi, kUsePDL>;
    static bool configured = false;
    if (!configured) {
      RuntimeDeviceCheck(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, C::kSmemBytes));
      if (kCluster > 1) {
        RuntimeDeviceCheck(cudaFuncSetAttribute(kernel, cudaFuncAttributeNonPortableClusterSizeAllowed, 1));
      }
      configured = true;
    }
    encode_map(&params.w_map, w.data_ptr(), static_cast<uint64_t>(n), static_cast<uint64_t>(k), k * 2, C::kBoxRows);
    encode_map(&params.x_map, x.data_ptr(), static_cast<uint64_t>(m), static_cast<uint64_t>(k), LDX.unwrap() * 2,
               kTokTile);
    params.m = static_cast<int>(m);
    params.n = static_cast<int>(n);
    params.steps_per_tile = static_cast<int>(k / kStepK);
    const unsigned grid = static_cast<unsigned>((rows_per_block + C::kBoxRows - 1) / C::kBoxRows * kCluster);
    auto lk = LaunchKernel(grid, C::kThreads, dev, C::kSmemBytes);
    lk.enable_pdl(kUsePDL);
    if (kCluster > 1) lk.enable_cluster(dim3(kCluster, 1, 1));
    lk(kernel, params);
  }

  // kEpiPlain: out[M, N]
  static void run(const tvm::ffi::TensorView x, const tvm::ffi::TensorView w, const tvm::ffi::TensorView out) {
    using namespace host;
    static_assert(kEpi == kEpiPlain);
    Params params{};
    auto device_ = SymbolicDevice{};
    auto M = SymbolicSize{"m"};
    auto N = SymbolicSize{"n"};
    auto LDO = SymbolicSize{"ldo"};
    device_.set_options<kDLCUDA>();
    TensorMatcher({M, N}).with_strides({LDO, 1}).with_dtype<bf16_t>().with_device(device_).verify(out);
    params.out = static_cast<bf16_t*>(out.data_ptr());
    params.ldo = LDO.unwrap();
    int64_t m = 0;
    launch(params, x, w, device_, m);
  }

  // kEpiSwiGLU: w = [gate; up] rows (2 I, K); act_out[M, I]
  static void run_swiglu(const tvm::ffi::TensorView x, const tvm::ffi::TensorView w, const tvm::ffi::TensorView act) {
    using namespace host;
    static_assert(kEpi == kEpiSwiGLU && kBoxes == 2);
    Params params{};
    auto device_ = SymbolicDevice{};
    auto M = SymbolicSize{"m"};
    auto I = SymbolicSize{"i"};
    auto LDO = SymbolicSize{"ldo"};
    device_.set_options<kDLCUDA>();
    TensorMatcher({M, I}).with_strides({LDO, 1}).with_dtype<bf16_t>().with_device(device_).verify(act);
    RuntimeCheck(I.unwrap() * 2 == w.size(0), "draft_gemv: act width must be half the weight rows");
    params.out = static_cast<bf16_t*>(act.data_ptr());
    params.ldo = LDO.unwrap();
    int64_t m = 0;
    launch(params, x, w, device_, m);
  }

  // kEpiConvIn: w = coefficient projection [4 G, H]; base0 [2, H]; conv_out [M, H]; coef_out [M, 4 G]
  static void run_conv(const tvm::ffi::TensorView x, const tvm::ffi::TensorView w, const tvm::ffi::TensorView base0,
                       const tvm::ffi::TensorView conv_out, const tvm::ffi::TensorView coef_out) {
    using namespace host;
    static_assert(kEpi == kEpiConvIn && kBoxes == 4);
    Params params{};
    auto device_ = SymbolicDevice{};
    auto M = SymbolicSize{"m"};
    auto H = SymbolicSize{"h"};
    auto G4 = SymbolicSize{"g4"};
    device_.set_options<kDLCUDA>();
    TensorMatcher({M, H}).with_strides({H, 1}).with_dtype<bf16_t>().with_device(device_).verify(conv_out);
    TensorMatcher({M, G4}).with_strides({G4, 1}).with_dtype<bf16_t>().with_device(device_).verify(coef_out);
    TensorMatcher({2, H}).with_strides({H, 1}).with_dtype<bf16_t>().with_device(device_).verify(base0);
    RuntimeCheck(G4.unwrap() * 4 == H.unwrap(), "draft_gemv: 16-channel groups expected");
    RuntimeCheck(w.size(0) == G4.unwrap() && w.size(1) == H.unwrap(), "draft_gemv: projection shape");
    RuntimeCheck(x.stride(0) == H.unwrap(), "draft_gemv: x must be contiguous rows");
    RuntimeCheck(M.unwrap() % 8 == 0, "draft_gemv: whole 8-token blocks expected");
    params.xin = static_cast<const bf16_t*>(x.data_ptr());
    params.ldx = H.unwrap();
    params.base0 = static_cast<const bf16_t*>(base0.data_ptr());
    params.conv_out = static_cast<bf16_t*>(conv_out.data_ptr());
    params.coef_out = static_cast<bf16_t*>(coef_out.data_ptr());
    params.groups = static_cast<int>(G4.unwrap() / 4);
    int64_t m = 0;
    launch(params, x, w, device_, m);
  }
};

}  // namespace muse_vr::draft_gemv
