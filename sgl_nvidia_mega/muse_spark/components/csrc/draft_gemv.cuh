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
//   kEpiPush:     plain store, and every 16-byte output vector is also pushed into the peers' one-shot push slots: one
//                 store per peer, or (push_mc set) one store through the plane's multicast address, which writes the
//                 slot (dst = every rank, src = this rank) on every rank, this rank's own included, as the layer
//                 boundary's kMcPush does (draft_ar_norm.cuh).
//   kEpiQK:       the attention QKV projection, one 128-row head per tile (query heads, key heads, value heads in that order).
//                 Query and key heads: the per-head RMSNorm and table RoPE of the draft's QK norm + RoPE kernel
//                 (ops/draft_qk.py), the same arithmetic in the same order (the sum of squares in that kernel's reduction
//                 tree); query heads go to q_out[m, head, :], key heads to the KV cache row of each token's write slot.
//                 Value heads are copied to their KV cache rows. The QKV tensor itself is not stored.
//
// Stage reuse: the producer refills a stage only after every consumer warp released it (mbarrier) and after a
// generic-to-async proxy fence, since the consumers read stages with ldmatrix and the refill is a tensor copy (PTX ISA:
// fence.proxy.async). Without the fence some tile shapes produced run-to-run different outputs.
//
// Stage reuse: the producer refills a stage only after every consumer warp released it (mbarrier) and after a
// generic-to-async proxy fence, since the consumers read stages with ldmatrix and the refill is a tensor copy (PTX ISA:
// fence.proxy.async). Without the fence some tile shapes produced run-to-run different outputs.
//
// PDL: weight copies for the first stages are issued before griddepcontrol.wait (no kernel writes the weights);
// activation copies (after a generic-to-async proxy fence), activation reads and every global store come after it.
// The dependent launch is signalled once the producer has issued every copy (default) or, with kEarlyTrigger, right after
// the first stages' weight copies, so the next programmatic-dependent kernel can start its own independent prologue earlier;
// its griddepcontrol.wait still waits until this grid has completed and its writes are visible (PTX ISA: griddepcontrol).
//
// L2 lookahead (kPfSteps > 0): after the first stages' weight copies and before griddepcontrol.wait, the producer also
// requests the weight boxes of the next kPfSteps steps into L2 (cp.async.bulk.prefetch.tensor, a non-blocking hint; PTX ISA).
// The weights are static, so these reads may start while the predecessor grid is still running; the main loop then copies
// the same boxes as before. Outputs are unchanged bit for bit; only the timing of the weight reads moves. Tunable
// (tests/test_draft_gemv.py --bench, tests/dist_draft_boundary.py --bench).
//
// Epilogue read-ahead (kEpiPf, conv and QK epilogues): the global inputs an epilogue reads besides the tile (the conv
// epilogue's activation vectors; the QK epilogue's RoPE table entries and key-cache rows) are loaded right after the grid
// dependency wait, before the MMA loop, instead of after it. Same loads of the same values: outputs bit for bit the same.
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
#include <type_traits>

// Arithmetic form of the QK epilogue: 0 = the draft QK-norm Triton kernel as compiled with FP fusion (fma contractions,
// the mean as div.full), 1 = as compiled without it (each product and sum rounded separately, the mean as a product
// with div.approx(1, 128)). Set by the host build of the QK module (ops/draft_gemv.py; ops/swa_verify_attn.qk_arith).
#ifndef MUSE_VR_QK_UNFUSED
#define MUSE_VR_QK_UNFUSED 0
#endif

namespace muse_vr::draft_gemv {
#if MUSE_VR_QK_UNFUSED
// The unfused build gets its own symbols, so both builds can be loaded in one process.
inline namespace qk_unfused {
#endif

using namespace sglang;

inline constexpr int kEpiPlain = 0;
inline constexpr int kEpiConvIn = 1;
inline constexpr int kEpiSwiGLU = 2;
inline constexpr int kEpiPush = 3;  // plain store + one-shot push of the tile into the peers' all-reduce push slots
inline constexpr int kEpiQK = 4;    // QKV projection + QK norm + RoPE + KV cache write (one head per 128-row tile)

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
  // kEpiPush: the consumer is the draft layer boundary (draft_ar_norm, one-shot push protocol): its block b = request block x
  // push_cluster + column slice reads phase counter[b] & 1 and polls slot (src = rank + phase x world) of its own workspace
  uint8_t* push_ws[8];               // every rank's push workspace base
  const uint32_t* push_counter;      // the plane's block counters (this rank)
  uint32_t push_slot_bytes;
  int push_world;
  int push_rank;
  int push_cluster;                  // consumer CTAs per request block (columns split this many ways)
  uint8_t* push_mc;                  // the push workspace's multicast address (nullptr: one store per peer); the consumer
                                     // then runs with kMcPush and polls its own slot too
  // kEpiQK
  bf16_t* __restrict__ q_out;          // [m, nhq, 128]
  bf16_t* __restrict__ k_cache;        // [slots, nhk, 128], row stride cache_ld
  bf16_t* __restrict__ v_cache;        // [slots, nhk, 128], row stride cache_ld
  int64_t cache_ld;
  const int64_t* __restrict__ loc;     // [m] write slots
  const int64_t* __restrict__ pos;     // [m] positions
  const void* __restrict__ cos_sin;    // [positions, 128]: cos (64) then sin (64), fp32 or bf16
  int cos_sin_bf16;
  const bf16_t* __restrict__ q_w;      // [128] query norm weight
  const bf16_t* __restrict__ k_w;      // [128] key norm weight
  float qk_eps;
  int nhq;
  int nhk;
  int qk_half;  // which half-warp's sum of squares the reference reduction keeps (tests/test_draft_gemv.py --qk)
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
// One 16-byte store through a multicast address, relaxed at system scope like the per-peer stores it replaces (the same
// instruction as draft_ar_norm.cuh's kMcPush store).
SGL_DEVICE void st_multimem_relaxed_16B(const uint4& v, uint8_t* addr) {
  asm volatile("multimem.st.relaxed.sys.global.v4.f32 [%4], {%0, %1, %2, %3};"
               :
               : "f"(__uint_as_float(v.x)), "f"(__uint_as_float(v.y)), "f"(__uint_as_float(v.z)), "f"(__uint_as_float(v.w)),
                 "l"(addr)
               : "memory");
}

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

// The QK norm + RoPE kernel's float operations, instruction for instruction (PTX ISA: div.full, sqrt.approx.ftz).
SGL_DEVICE float div_full(float a, float b) {
  float r;
  asm("div.full.f32 %0, %1, %2;" : "=f"(r) : "f"(a), "f"(b));
  return r;
}

SGL_DEVICE float sqrt_approx_ftz(float a) {
  float r;
  asm("sqrt.approx.ftz.f32 %0, %1;" : "=f"(r) : "f"(a));
  return r;
}

// Mean of the sum of squares over the 128 columns: div.full(ss, 128) (fused form), or ss times div.approx(1, 128).
SGL_DEVICE float qk_mean128(float ss) {
  if constexpr (MUSE_VR_QK_UNFUSED != 0) {
    float inv;
    asm("div.approx.f32 %0, %1, %2;" : "=f"(inv) : "f"(1.0f), "f"(128.0f));
    return __fmul_rn(inv, ss);
  } else {
    return div_full(ss, 128.0f);
  }
}

// RoPE of the pair (x1, x2): fused form fma(c, x1, -(s * x2)), fma(s, x1, c * x2); unfused: c*x1 - s*x2, c*x2 + s*x1.
SGL_DEVICE void qk_rope(float c, float sn, float x1, float x2, float& o1, float& o2) {
  if constexpr (MUSE_VR_QK_UNFUSED != 0) {
    o1 = __fsub_rn(__fmul_rn(c, x1), __fmul_rn(sn, x2));
    o2 = __fadd_rn(__fmul_rn(c, x2), __fmul_rn(sn, x1));
  } else {
    o1 = __fmaf_rn(c, x1, -__fmul_rn(sn, x2));
    o2 = __fmaf_rn(sn, x1, __fmul_rn(c, x2));
  }
}

SGL_DEVICE float bf16_round(float v) {
  return __bfloat162float(__float2bfloat16_rn(v));
}

// Sum of squares of one 128-element head row (x[e] held as bf16 bits in t16[e * stride]) in the reduction tree of the QK
// norm + RoPE kernel: four warps of 32 lanes with element e at lane e % 32 of warp e / 32; in each warp, lane l first forms
// fma(x[l], x[l], rn(x[l ^ 16]^2)), then xor-butterflies over offsets 8, 4, 2, 1 inside each half-warp; the warp's sum is
// the value of half `half` (lanes 0-15: 0, lanes 16-31: 1; the two halves can differ in the last place, and that kernel
// keeps one of them per warp); the four warp sums are added as (w0 + w2) + (w1 + w3). Every lane returns the same value.
SGL_DEVICE float qk_sum_squares(const uint16_t* t16, int stride, int lane, int half) {
  float ws[4];
#pragma unroll
  for (int w = 0; w < 4; ++w) {
    const float x = bf16_bits(t16[(32 * w + lane) * stride]);
    const float xp = bf16_bits(t16[(32 * w + (lane ^ 16)) * stride]);
    float v = MUSE_VR_QK_UNFUSED != 0 ? __fadd_rn(__fmul_rn(x, x), __fmul_rn(xp, xp)) : __fmaf_rn(x, x, __fmul_rn(xp, xp));
#pragma unroll
    for (int o = 8; o >= 1; o >>= 1) v = __fadd_rn(v, __shfl_xor_sync(0xffffffffu, v, o));
    ws[w] = __shfl_sync(0xffffffffu, v, half * 16);
  }
  return __fadd_rn(__fadd_rn(ws[0], ws[2]), __fadd_rn(ws[1], ws[3]));
}

// Consumer warps only (named barrier 1, kWarps warps). `acc` is the warp's final fp32 tile fragment; `scratch` is free
// shared memory (the stage ring, no longer read once the main loop has ended).
// Conv epilogue inputs read ahead (kEpiPf): each consumer thread of the epilogue CTA loads the activation vectors its
// epilogue iterations will convolve (the token's 8-channel vector and its predecessor's).
template <typename C, bool kOn>
struct ConvPrefetch {
  static constexpr int kVecs = C::kBoxRows * 2;  // 8-channel vectors per token in a tile
  static constexpr int kIters = kOn ? (C::kNTiles * 8 * kVecs + C::kConsumerThreads - 1) / C::kConsumerThreads : 1;
  uint4 x[kIters], xp[kIters];

  SGL_DEVICE void load(const Params& p, int tile, int tid) {
#pragma unroll
    for (int it = 0; it < kIters; ++it) {
      const int e = tid + it * C::kConsumerThreads;
      if (e >= p.m * kVecs) break;
      const int tok = e / kVecs, v = e % kVecs;
      const int ch = (tile * C::kBoxRows + v / 2) * 16 + (v % 2) * 8;
      x[it] = *reinterpret_cast<const uint4*>(p.xin + tok * p.ldx + ch);
      if ((tok & 7) >= 1) xp[it] = *reinterpret_cast<const uint4*>(p.xin + (tok - 1) * p.ldx + ch);
    }
  }
};

// QK epilogue inputs read ahead (kEpiPf): each consumer warp loads, for the tokens its epilogue handles on a query / key
// head tile, the RoPE table entries of the token's position (and the cache row for key heads).
template <typename C, bool kOn>
struct QkPrefetch {
  static constexpr int kWarpsC = C::kConsumerThreads / 32;
  static constexpr int kIters = kOn ? (C::kNTiles * 8 + kWarpsC - 1) / kWarpsC : 1;
  int64_t loc[kIters];
  float c[kIters][2], sn[kIters][2];

  SGL_DEVICE void load(const Params& p, int tile, int tid) {
    if (tile >= p.nhq + p.nhk) return;  // value heads read only the cache rows
    const int warp = tid / 32, lane = tid % 32;
    const bool is_q = tile < p.nhq;
#pragma unroll
    for (int it = 0; it < kIters; ++it) {
      const int tok = warp + it * kWarpsC;
      if (tok >= p.m) break;
      const int64_t pos = p.pos[tok];
      if (!is_q) loc[it] = p.loc[tok];
#pragma unroll
      for (int half = 0; half < 2; ++half) {
        const int i = lane + 32 * half;
        if (p.cos_sin_bf16) {
          const uint16_t* cs = static_cast<const uint16_t*>(p.cos_sin) + pos * 128;
          c[it][half] = bf16_bits(cs[i]);
          sn[it][half] = bf16_bits(cs[64 + i]);
        } else {
          const float* cs = static_cast<const float*>(p.cos_sin) + pos * 128;
          c[it][half] = cs[i];
          sn[it][half] = cs[64 + i];
        }
      }
    }
  }
};

template <typename C, int kEpi, bool kOn>
using EpiPrefetch = std::conditional_t<kEpi == kEpiQK, QkPrefetch<C, kOn>, ConvPrefetch<C, kOn>>;

template <typename C, int kEpi, bool kUsePf = false>
SGL_DEVICE void epilogue(const Params& p, float (&acc)[C::kMTilesPerWarp][C::kNTiles][4], int tile, int row_base,
                         int lane, uint8_t* scratch, const EpiPrefetch<C, kEpi, kUsePf>* pf = nullptr) {
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
    if constexpr (kEpi == kEpiPush) {
      // 8-column vectors of each token row: stored to out and pushed to every peer (a 32-bit word of two +0.0 is sent as
      // (+0.0, -0.0), as the push protocol requires; the consumer applies the same rewrite to its own vector)
      constexpr int kVecs = C::kBoxRows / 8;
      const int row_vecs = p.n / 8;
      const int slice = p.n / p.push_cluster;
      for (int e = tid; e < p.m * kVecs; e += C::kConsumerThreads) {
        const int tok = e / kVecs, v = e % kVecs;
        const int n0 = tile * C::kBoxRows + v * 8;
        if (n0 >= p.n) continue;
        uint32_t w[4];
#pragma unroll
        for (int k = 0; k < 4; ++k)
          w[k] = static_cast<uint32_t>(t16[(v * 8 + 2 * k) * kTokTile + tok]) |
                 (static_cast<uint32_t>(t16[(v * 8 + 2 * k + 1) * kTokTile + tok]) << 16);
        uint4 val = make_uint4(w[0], w[1], w[2], w[3]);
        *reinterpret_cast<uint4*>(p.out + static_cast<int64_t>(tok) * p.ldo + n0) = val;
        if (val.x == 0u) val.x = 0x8000u;
        if (val.y == 0u) val.y = 0x8000u;
        if (val.z == 0u) val.z = 0x8000u;
        if (val.w == 0u) val.w = 0x8000u;
        const int b = (tok / 8) * p.push_cluster + n0 / slice;
        const uint32_t phase = p.push_counter[b] & 1u;
        const int64_t vid = static_cast<int64_t>(tok) * row_vecs + n0 / 8;
        const uint32_t src = static_cast<uint32_t>(p.push_rank) + phase * p.push_world;
        if (p.push_mc != nullptr) {
          st_multimem_relaxed_16B(val, p.push_mc + src * p.push_slot_bytes + vid * 16);
        } else {
          for (int i = 0; i < p.push_world; ++i) {
            if (i == p.push_rank) continue;
            uint8_t* slot = p.push_ws[i] + src * p.push_slot_bytes;
            asm volatile("st.relaxed.sys.global.v4.b32 [%4], {%0, %1, %2, %3};" ::"r"(val.x), "r"(val.y), "r"(val.z),
                         "r"(val.w), "l"(slot + vid * 16)
                         : "memory");
          }
        }
      }
    } else if constexpr (kEpi == kEpiQK) {
      static_assert(C::kBoxRows == 128, "kEpiQK: one 128-row head per tile");
      const int warp = tid / 32;
      if (tile >= p.nhq + p.nhk) {
        // value head: copy to the cache rows
        const int hv = tile - p.nhq - p.nhk;
        for (int e = tid; e < p.m * 16; e += C::kConsumerThreads) {
          const int tok = e / 16, v = e % 16;
          uint32_t w[4];
#pragma unroll
          for (int k = 0; k < 4; ++k)
            w[k] = static_cast<uint32_t>(t16[(v * 8 + 2 * k) * kTokTile + tok]) |
                   (static_cast<uint32_t>(t16[(v * 8 + 2 * k + 1) * kTokTile + tok]) << 16);
          *reinterpret_cast<uint4*>(p.v_cache + p.loc[tok] * p.cache_ld + hv * 128 + v * 8) = make_uint4(w[0], w[1], w[2], w[3]);
        }
        return;
      }
      const bool is_q = tile < p.nhq;
      const bf16_t* nw = is_q ? p.q_w : p.k_w;
      if constexpr (kUsePf) {
        // the read-ahead variant: token warp + it * (warps) of iteration it, its RoPE entries / cache row prefetched
        constexpr int kWarpsC = C::kConsumerThreads / 32;
#pragma unroll
        for (int it = 0; it < QkPrefetch<C, kUsePf>::kIters; ++it) {
          const int tok = warp + it * kWarpsC;
          if (tok >= p.m) break;
          const float ss = qk_sum_squares(t16 + tok, kTokTile, lane, p.qk_half);
          const float inv = div_full(1.0f, sqrt_approx_ftz(__fadd_rn(qk_mean128(ss), p.qk_eps)));
          bf16_t* dst = is_q ? p.q_out + (static_cast<int64_t>(tok) * p.nhq + tile) * 128
                             : p.k_cache + pf->loc[it] * p.cache_ld + (tile - p.nhq) * 128;
#pragma unroll
          for (int half = 0; half < 2; ++half) {
            const int i = lane + 32 * half;  // rotary pair (i, i + 64)
            const float c = pf->c[it][half], sn = pf->sn[it][half];
            const float x1 = bf16_round(__fmul_rn(__fmul_rn(inv, bf16_bits(t16[i * kTokTile + tok])),
                                                  __bfloat162float(nw[i])));
            const float x2 = bf16_round(__fmul_rn(__fmul_rn(inv, bf16_bits(t16[(64 + i) * kTokTile + tok])),
                                                  __bfloat162float(nw[64 + i])));
            float o1, o2;
            qk_rope(c, sn, x1, x2, o1, o2);
            dst[i] = __float2bfloat16_rn(o1);
            dst[64 + i] = __float2bfloat16_rn(o2);
          }
        }
      } else {
      for (int tok = warp; tok < p.m; tok += C::kConsumerThreads / 32) {
        const float ss = qk_sum_squares(t16 + tok, kTokTile, lane, p.qk_half);
        const float inv = div_full(1.0f, sqrt_approx_ftz(__fadd_rn(qk_mean128(ss), p.qk_eps)));
        const int64_t pos = p.pos[tok];
        bf16_t* dst = is_q ? p.q_out + (static_cast<int64_t>(tok) * p.nhq + tile) * 128
                           : p.k_cache + p.loc[tok] * p.cache_ld + (tile - p.nhq) * 128;
#pragma unroll
        for (int half = 0; half < 2; ++half) {
          const int i = lane + 32 * half;  // rotary pair (i, i + 64)
          float c, sn;
          if (p.cos_sin_bf16) {
            const uint16_t* cs = static_cast<const uint16_t*>(p.cos_sin) + pos * 128;
            c = bf16_bits(cs[i]);
            sn = bf16_bits(cs[64 + i]);
          } else {
            const float* cs = static_cast<const float*>(p.cos_sin) + pos * 128;
            c = cs[i];
            sn = cs[64 + i];
          }
          const float x1 = bf16_round(__fmul_rn(__fmul_rn(inv, bf16_bits(t16[i * kTokTile + tok])),
                                                __bfloat162float(nw[i])));
          const float x2 = bf16_round(__fmul_rn(__fmul_rn(inv, bf16_bits(t16[(64 + i) * kTokTile + tok])),
                                                __bfloat162float(nw[64 + i])));
          float o1, o2;
          qk_rope(c, sn, x1, x2, o1, o2);
          dst[i] = __float2bfloat16_rn(o1);
          dst[64 + i] = __float2bfloat16_rn(o2);
        }
      }
      }
    } else if constexpr (kEpi == kEpiSwiGLU) {
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
      constexpr int kIters = kUsePf ? ConvPrefetch<C, kUsePf>::kIters : 1;
#pragma unroll
      for (int it = 0; it < kIters; ++it)
      for (int e = tid + it * C::kConsumerThreads; e < p.m * kVecs; e += C::kConsumerThreads) {
        if constexpr (kUsePf) {
          if (e != tid + it * C::kConsumerThreads) break;  // one vector per iteration: the prefetched one
        }
        const int tok = e / kVecs, v = e % kVecs;
        const int i = v / 2;
        const int ch = (tile * C::kBoxRows + i) * 16 + (v % 2) * 8;
        const float k00 = bf16_bits(t16[i * kTokTile + tok]);
        const float k01 = bf16_bits(t16[(C::kBoxRows + i) * kTokTile + tok]);
        uint4 x4, xp4 = make_uint4(0u, 0u, 0u, 0u);
        const bool has_prev = (tok & 7) >= 1;
        if constexpr (kUsePf) {
          x4 = pf->x[it];
          if (has_prev) xp4 = pf->xp[it];
        } else {
          x4 = *reinterpret_cast<const uint4*>(p.xin + tok * p.ldx + ch);
          if (has_prev) xp4 = *reinterpret_cast<const uint4*>(p.xin + (tok - 1) * p.ldx + ch);
        }
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
          bool kUsePDL, bool kEarlyTrigger = false, int kPfSteps = 0, bool kEpiPf = false>
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
      auto prefetch_w = [&](int i) {
        const int slab0 = (s_begin + i) * C::kSlabs;
#pragma unroll
        for (int b = 0; b < kBoxes; ++b) {
          const int r0 = p.block_off[b] + tile * C::kBoxRows;
          asm volatile("cp.async.bulk.prefetch.tensor.3d.L2.global.tile [%0, {%1, %2, %3}];" ::"l"(
                           reinterpret_cast<uint64_t>(&p.w_map)),
                       "r"(0), "r"(r0), "r"(slab0)
                       : "memory");
        }
      };
      auto issue_x = [&](int i) {
        const int s = i % kStages;
        tma_load_3d(smem_u32(stages + s * C::kStageBytes + C::kWStageBytes), &p.x_map, 0, 0,
                    (s_begin + i) * C::kSlabs, smem_u32(&full[s]));
      };
      const int pre = nsteps < kStages ? nsteps : kStages;
      for (int i = 0; i < pre; ++i) issue_w(i);
      if constexpr (kPfSteps > 0) {
        const int pf_end = nsteps < pre + kPfSteps ? nsteps : pre + kPfSteps;
        for (int i = pre; i < pf_end; ++i) prefetch_w(i);
      }
      if constexpr (kEarlyTrigger) pdl_trigger<kUsePDL>();
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
  // kEpiPf: the epilogue CTA's other global inputs are read ahead here, before the MMA loop
  constexpr bool kUsePf = kEpiPf && (kEpi == kEpiConvIn || kEpi == kEpiQK);
  EpiPrefetch<C, kEpi, kUsePf> pf;
  if constexpr (kUsePf) {
    if (rank == 0) pf.load(p, tile, static_cast<int>(threadIdx.x));
  }
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
    epilogue<C, kEpi, kUsePf>(p, acc[0], tile, row_base, lane, stages, &pf);
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
      epilogue<C, kEpi, kUsePf>(p, acc[0], tile, row_base, lane, stages, &pf);
    }
  }
}

template <int kTokTile, int kRows, int kStepK, int kStages, int kWarps, int kCluster, int kBoxes, int kEpi, bool kUsePDL,
          bool kEarlyTrigger = false, int kPfSteps = 0, bool kEpiPf = false>
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
    auto kernel = draft_gemv_kernel<kTokTile, kRows, kStepK, kStages, kWarps, kCluster, kBoxes, kEpi, kUsePDL, kEarlyTrigger,
                                    kPfSteps, kEpiPf>;
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

  // kEpiQK: w = QKV projection [(nhq + 2 nhk) * 128, K] (query heads, key heads, value heads); q_out [M, nhq, 128]
  // contiguous; k_cache / v_cache [slots, nhk, 128] with contiguous heads and one row stride; loc / pos [M] int64;
  // cos_sin [positions, 128] fp32 or bf16 contiguous; q_w / k_w [128] bf16.
  static void run_qk(const tvm::ffi::TensorView x, const tvm::ffi::TensorView w, const tvm::ffi::TensorView q_out,
                     const tvm::ffi::TensorView k_cache, const tvm::ffi::TensorView v_cache,
                     const tvm::ffi::TensorView loc, const tvm::ffi::TensorView pos, const tvm::ffi::TensorView cos_sin,
                     const tvm::ffi::TensorView q_w, const tvm::ffi::TensorView k_w, double eps, int64_t qk_half) {
    using namespace host;
    static_assert(kEpi == kEpiQK && kBoxes == 1 && kRows == 128);
    Params params{};
    auto device_ = SymbolicDevice{};
    auto M = SymbolicSize{"m"};
    auto NHQ = SymbolicSize{"nhq"};
    auto NHK = SymbolicSize{"nhk"};
    auto S = SymbolicSize{"slots"};
    auto LDC = SymbolicSize{"cache_ld"};
    auto LDQ = SymbolicSize{"q_ld"};
    device_.set_options<kDLCUDA>();
    TensorMatcher({M, NHQ, 128}).with_strides({LDQ, 128, 1}).with_dtype<bf16_t>().with_device(device_).verify(q_out);
    TensorMatcher({S, NHK, 128}).with_strides({LDC, 128, 1}).with_dtype<bf16_t>().with_device(device_).verify(k_cache);
    TensorMatcher({S, NHK, 128}).with_strides({LDC, 128, 1}).with_dtype<bf16_t>().with_device(device_).verify(v_cache);
    TensorMatcher({M}).with_strides({1}).with_dtype<int64_t>().with_device(device_).verify(loc);
    TensorMatcher({M}).with_strides({1}).with_dtype<int64_t>().with_device(device_).verify(pos);
    TensorMatcher({128}).with_strides({1}).with_dtype<bf16_t>().with_device(device_).verify(q_w);
    TensorMatcher({128}).with_strides({1}).with_dtype<bf16_t>().with_device(device_).verify(k_w);
    RuntimeCheck(cos_sin.dim() == 2 && cos_sin.size(1) == 128 && cos_sin.stride(0) == 128 && cos_sin.stride(1) == 1,
                 "draft_gemv qk: cos_sin must be a contiguous [positions, 128] table");
    const bool cs_bf16 = cos_sin.dtype().code == kDLBfloat && cos_sin.dtype().bits == 16;
    const bool cs_f32 = cos_sin.dtype().code == kDLFloat && cos_sin.dtype().bits == 32;
    RuntimeCheck(cs_bf16 || cs_f32, "draft_gemv qk: cos_sin must be fp32 or bf16");
    RuntimeCheck(LDQ.unwrap() == NHQ.unwrap() * 128, "draft_gemv qk: q_out must be contiguous");
    RuntimeCheck(w.size(0) == (NHQ.unwrap() + 2 * NHK.unwrap()) * 128, "draft_gemv qk: weight rows must be (nhq + 2 nhk) heads");
    RuntimeCheck(qk_half == 0 || qk_half == 1, "draft_gemv qk: qk_half is 0 or 1");
    (void)S;
    params.q_out = static_cast<bf16_t*>(q_out.data_ptr());
    params.k_cache = static_cast<bf16_t*>(k_cache.data_ptr());
    params.v_cache = static_cast<bf16_t*>(v_cache.data_ptr());
    params.cache_ld = LDC.unwrap();
    params.loc = static_cast<const int64_t*>(loc.data_ptr());
    params.pos = static_cast<const int64_t*>(pos.data_ptr());
    params.cos_sin = cos_sin.data_ptr();
    params.cos_sin_bf16 = cs_bf16 ? 1 : 0;
    params.q_w = static_cast<const bf16_t*>(q_w.data_ptr());
    params.k_w = static_cast<const bf16_t*>(k_w.data_ptr());
    params.qk_eps = static_cast<float>(eps);
    params.nhq = static_cast<int>(NHQ.unwrap());
    params.nhk = static_cast<int>(NHK.unwrap());
    params.qk_half = static_cast<int>(qk_half);
    int64_t m = 0;
    launch(params, x, w, device_, m);
    RuntimeCheck(m == M.unwrap(), "draft_gemv qk: x rows must equal q_out rows");
  }
};

#if MUSE_VR_QK_UNFUSED
}  // inline namespace qk_unfused
#endif
}  // namespace muse_vr::draft_gemv
