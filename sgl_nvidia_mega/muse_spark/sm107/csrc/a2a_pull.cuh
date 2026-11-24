// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
/**
 * \brief All-to-all pull of MXFP8 row blocks for the row-shard output projection (``o_rowshard``'s pull form).
 *
 * Every rank has staged, in a symmetric buffer, one block per destination rank: E4M3 values [rows, W] (row-major), then
 * their scales in the 128 x 4 swizzled layout (tiles of 128 rows x 4 scale columns, 512 bytes each, row tiles outer).
 * One launch copies, from every rank r (this rank included), the block r staged for this rank into this rank's
 * full-width operand:
 *   values of block r      -> columns [r * W, (r + 1) * W) of a row-major [rows, world * W] buffer;
 *   scale tiles of row tile t of block r -> tile columns [r * T, (r + 1) * T) of row tile t of the full operand's
 *   scales (T = W / 128 tiles per block row tile), which is the swizzled layout of the [rows, world * W] operand.
 * Bytes are copied unchanged (16-byte loads and stores). Peer blocks are read through their mapped addresses.
 *
 * Ordering is the caller's: every rank's staging must be complete and visible before the launch (a symmetric-memory
 * barrier) and must not be rewritten until every rank's launch has completed (the next barrier).
 *
 * The barrier (``A2aBarrierKernel``): one CTA, one thread per peer. Each rank owns a signal pad of ``world`` 32-bit
 * slots in symmetric memory, zero at start; slot s of rank r's pad carries rank s's signal to rank r. Thread t (t !=
 * rank) first puts: a compare-and-swap 0 -> 1 on slot ``rank`` of rank t's pad with release semantics at system scope
 * (PTX ``atom.release.sys``), retried while the slot is still 1 (rank t has not yet consumed this rank's previous
 * signal). It then waits: a compare-and-swap 1 -> 0 on slot t of this rank's own pad with acquire semantics at system
 * scope (``atom.acquire.sys``), retried until it succeeds (rank t's signal is consumed). A ``fence.proxy.alias`` precedes
 * the put and follows the wait (the buffers are reached through different virtual addresses on different ranks). Only
 * 32-bit atomic accesses touch the pads. Each retry loop is bounded by ``max_polls`` unsuccessful attempts; past that the thread prints one
 * line and traps, which the host sees as a CUDA error, instead of waiting forever.
 *
 * Work split: per source rank, the block's values then its scales form one list of 16-byte vectors. Chunks of
 * kUnroll x 32 consecutive vectors are dealt round-robin over source ranks (chunk c reads rank c % world), and warps take
 * chunks block-fastest, so concurrently running warps read from different ranks. Each thread issues its kUnroll loads
 * before its stores; consecutive lanes touch consecutive vectors.
 */

#include <sgl_kernel/tensor.h>
#include <sgl_kernel/utils.h>

#include <sgl_kernel/runtime.cuh>
#include <sgl_kernel/utils.cuh>

#include <cuda_runtime.h>

#include <cstdint>
#include <cstdio>

namespace muse_vr {
namespace a2a {

constexpr int kThreads = 512;
constexpr int kMaxRanks = 8;
constexpr int kTileRows = 128;
constexpr int kTileBytes = 512;
constexpr int kTileCols = 128;  // value columns covered by one scale tile (4 scale columns x 32 elements)

struct Params {
  const uint8_t* src[kMaxRanks];  // per source rank: start of the block it staged for this rank
  uint8_t* dst_q;                 // [rows, world * W] values
  uint8_t* dst_sf;                // swizzled scales of the [rows, world * W] operand
  uint32_t world;
  uint32_t vpr;         // 16-byte vectors per value row of one block (W / 16)
  uint32_t svpt;        // 16-byte vectors of scales per row tile of one block (T * 512 / 16)
  uint32_t nv;          // value vectors of one block (rows * vpr)
  uint32_t ntot;        // nv + scale vectors of one block
  uint32_t nchunks;     // world * chunks per block
  uint64_t q_bytes;     // value bytes of one block (rows * W); its scales start there
  uint64_t row_pitch;   // bytes per value row of the full operand (world * W)
  uint64_t tile_pitch;  // scale bytes per row tile of the full operand (world * T * 512)
};

template <int kUnroll>
__global__ void __launch_bounds__(kThreads) a2a_pull_kernel(const __grid_constant__ Params p) {
  constexpr uint32_t kChunk = kUnroll * 32;
  const uint32_t lane = threadIdx.x & 31u;
  const uint32_t warp = blockIdx.x + gridDim.x * (threadIdx.x >> 5);
  const uint32_t nwarps = gridDim.x * (kThreads / 32);
  for (uint32_t c = warp; c < p.nchunks; c += nwarps) {
    const uint32_t r = c % p.world;
    const uint32_t v0 = (c / p.world) * kChunk;
    const uint8_t* src = p.src[r];
    uint4 val[kUnroll];
    uint4* dst[kUnroll];
#pragma unroll
    for (int i = 0; i < kUnroll; ++i) {
      const uint32_t e = v0 + static_cast<uint32_t>(i) * 32u + lane;
      dst[i] = nullptr;
      if (e < p.nv) {
        const uint32_t row = e / p.vpr;
        const uint32_t col = e - row * p.vpr;
        dst[i] = reinterpret_cast<uint4*>(p.dst_q + row * p.row_pitch + (uint64_t(r) * p.vpr + col) * 16u);
        val[i] = __ldcg(reinterpret_cast<const uint4*>(src) + e);
      } else if (e < p.ntot) {
        const uint32_t u = e - p.nv;
        const uint32_t t = u / p.svpt;
        const uint32_t cc = u - t * p.svpt;
        dst[i] = reinterpret_cast<uint4*>(p.dst_sf + t * p.tile_pitch + (uint64_t(r) * p.svpt + cc) * 16u);
        val[i] = __ldcg(reinterpret_cast<const uint4*>(src + p.q_bytes) + u);
      }
    }
#pragma unroll
    for (int i = 0; i < kUnroll; ++i) {
      if (dst[i] != nullptr) *dst[i] = val[i];
    }
  }
}

template <int kUnroll>
struct A2aPullKernel {
  // dst_q: uint8 [>= rows, world * width]; dst_sf: uint8 1-D, at least (rows / 128) * world * (width / 128) * 512 bytes.
  // s0..s7: per source rank, the address of the block it staged for this rank (ranks >= world ignored).
  static void run(const tvm::ffi::TensorView dst_q, const tvm::ffi::TensorView dst_sf, int64_t s0, int64_t s1, int64_t s2,
                  int64_t s3, int64_t s4, int64_t s5, int64_t s6, int64_t s7, int64_t world, int64_t width, int64_t rows,
                  int64_t grid) {
    using namespace sglang::host;
    RuntimeCheck(world >= 1 && world <= kMaxRanks, "1 to 8 ranks");
    RuntimeCheck(width > 0 && width % kTileCols == 0, "block width must be a whole number of scale tiles");
    RuntimeCheck(rows > 0 && rows % kTileRows == 0, "block rows must be a whole number of scale tiles");
    RuntimeCheck(dst_q.dim() == 2 && dst_q.dtype().bits == 8 && dst_q.size(1) == world * width && dst_q.stride(1) == 1 &&
                     dst_q.stride(0) == world * width && dst_q.size(0) >= rows,
                 "dst_q must be a contiguous 8-bit [>= rows, world * width] buffer");
    const int64_t tiles = width / kTileCols;
    RuntimeCheck(dst_sf.dim() == 1 && dst_sf.dtype().bits == 8 && dst_sf.stride(0) == 1 &&
                     dst_sf.size(0) >= (rows / kTileRows) * world * tiles * kTileBytes,
                 "dst_sf must be a contiguous 8-bit buffer of the full operand's scale tiles");
    RuntimeCheck(reinterpret_cast<uintptr_t>(dst_q.data_ptr()) % 16 == 0 &&
                     reinterpret_cast<uintptr_t>(dst_sf.data_ptr()) % 16 == 0,
                 "16-byte aligned destinations");
    const int64_t srcs[kMaxRanks] = {s0, s1, s2, s3, s4, s5, s6, s7};
    Params p{};
    for (int i = 0; i < world; ++i) {
      RuntimeCheck(srcs[i] != 0 && srcs[i] % 16 == 0, "16-byte aligned source blocks");
      p.src[i] = reinterpret_cast<const uint8_t*>(srcs[i]);
    }
    const int64_t vpr = width / 16;
    const int64_t svpt = tiles * kTileBytes / 16;
    const int64_t nv = rows * vpr;
    const int64_t ntot = nv + (rows / kTileRows) * svpt;
    constexpr int64_t kChunk = kUnroll * 32;
    const int64_t nchunks = world * ((ntot + kChunk - 1) / kChunk);
    RuntimeCheck(nchunks < (int64_t(1) << 31), "block too large");
    p.dst_q = static_cast<uint8_t*>(dst_q.data_ptr());
    p.dst_sf = static_cast<uint8_t*>(dst_sf.data_ptr());
    p.world = static_cast<uint32_t>(world);
    p.vpr = static_cast<uint32_t>(vpr);
    p.svpt = static_cast<uint32_t>(svpt);
    p.nv = static_cast<uint32_t>(nv);
    p.ntot = static_cast<uint32_t>(ntot);
    p.nchunks = static_cast<uint32_t>(nchunks);
    p.q_bytes = static_cast<uint64_t>(rows * width);
    p.row_pitch = static_cast<uint64_t>(world * width);
    p.tile_pitch = static_cast<uint64_t>(world * tiles * kTileBytes);
    RuntimeCheck(grid >= 1, "grid >= 1");
    LaunchKernel(dim3(static_cast<unsigned>(grid)), kThreads, dst_q.device(), 0)(a2a_pull_kernel<kUnroll>, p);
  }
};

__device__ __forceinline__ uint32_t cas_release_sys(uint32_t* addr, uint32_t expect, uint32_t value) {
  uint32_t old;
  asm volatile("atom.release.sys.global.cas.b32 %0, [%1], %2, %3;"
               : "=r"(old)
               : "l"(addr), "r"(expect), "r"(value)
               : "memory");
  return old;
}

__device__ __forceinline__ uint32_t cas_acquire_sys(uint32_t* addr, uint32_t expect, uint32_t value) {
  uint32_t old;
  asm volatile("atom.acquire.sys.global.cas.b32 %0, [%1], %2, %3;"
               : "=r"(old)
               : "l"(addr), "r"(expect), "r"(value)
               : "memory");
  return old;
}

struct BarrierParams {
  uint32_t* pad[kMaxRanks];  // per rank: the address of its signal pad (world slots)
  uint32_t world;
  uint32_t rank;
  uint32_t max_polls;
};

__device__ __noinline__ void barrier_exceeded(uint32_t rank, uint32_t peer, uint32_t phase, uint32_t polls) {
  printf("muse_vr_plugin a2a barrier: rank %u %s rank %u not done after %u attempts; trap\n", rank,
         phase == 0 ? "signal to" : "wait for", peer, polls);
  __trap();
}

__global__ void a2a_barrier_kernel(const __grid_constant__ BarrierParams p) {
  const uint32_t t = threadIdx.x;
  if (t >= p.world || t == p.rank) return;
  // The staged data and the pads are reached through different virtual addresses on the writing and the reading rank
  // (the owner's own mapping and the peers' mappings of it): virtually aliased addresses in PTX terms. fence.proxy.alias
  // orders accesses made through aliases and composes with the release below and the acquire further down.
  asm volatile("fence.proxy.alias;" ::: "memory");
  uint32_t polls = 0;
  while (cas_release_sys(p.pad[t] + p.rank, 0u, 1u) != 0u) {
    if (++polls == p.max_polls) [[unlikely]]
      barrier_exceeded(p.rank, t, 0, polls);
  }
  polls = 0;
  while (cas_acquire_sys(p.pad[p.rank] + t, 1u, 0u) != 1u) {
    if (++polls == p.max_polls) [[unlikely]]
      barrier_exceeded(p.rank, t, 1, polls);
  }
  asm volatile("fence.proxy.alias;" ::: "memory");
}

struct A2aBarrierKernel {
  // p0..p7: per rank, the address of its signal pad (``world`` zero-initialized 32-bit slots; ranks >= world ignored).
  // like: any CUDA tensor on this rank's device (selects the device / stream).
  static void run(const tvm::ffi::TensorView like, int64_t p0, int64_t p1, int64_t p2, int64_t p3, int64_t p4, int64_t p5,
                  int64_t p6, int64_t p7, int64_t world, int64_t rank, int64_t max_polls) {
    using namespace sglang::host;
    RuntimeCheck(world >= 1 && world <= kMaxRanks && rank >= 0 && rank < world, "1 to 8 ranks, rank in range");
    RuntimeCheck(max_polls >= 1 && max_polls <= int64_t(0xffffffff), "max_polls in [1, 2^32 - 1]");
    const int64_t pads[kMaxRanks] = {p0, p1, p2, p3, p4, p5, p6, p7};
    BarrierParams p{};
    for (int i = 0; i < world; ++i) {
      RuntimeCheck(pads[i] != 0 && pads[i] % 4 == 0, "4-byte aligned signal pads");
      p.pad[i] = reinterpret_cast<uint32_t*>(pads[i]);
    }
    p.world = static_cast<uint32_t>(world);
    p.rank = static_cast<uint32_t>(rank);
    p.max_polls = static_cast<uint32_t>(max_polls);
    LaunchKernel(dim3(1), dim3(32), like.device(), 0)(a2a_barrier_kernel, p);
  }
};

}  // namespace a2a
}  // namespace muse_vr
