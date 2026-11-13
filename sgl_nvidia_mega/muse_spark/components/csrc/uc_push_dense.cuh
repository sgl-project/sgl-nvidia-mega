// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
/**
 * \brief Dense unicast push of one rank's MXFP8 shard into every rank's gathered operand (the unicast form of
 * mc_push_dense.cuh).
 *
 * Same destination layout and write sets as mc_push_dense.cuh: rank p's rows land at global rows [p x rows, (p + 1) x rows) of
 * the DENSE [world x rows, K] operand, and each of its scale words (4 scale columns of one row) is stored at the global word
 *   W(m, t) = (m / 128) x KT x 512 + t x 512 + (m % 32) x 16 + ((m % 128) / 32) x 4,   KT = K / 128.
 * Instead of one store through a multicast address, each value vector (16 bytes) and each scale word (4 bytes) is stored once
 * per rank, through that rank's mapping of its gathered slot (the peers' imported mappings and this rank's own address).
 *
 * Each thread issues kUnroll loads before its stores. With kFence = 1 every storing thread ends with a fence.proxy.alias after
 * its last store (the form used by mc_push_dense.cuh); with kFence = 0 it does not, and the alias fence of the barrier kernel
 * that follows is the only one. Ordering is the caller's, as for mc_push_dense.cuh: the shard is complete before the launch
 * (stream order); the next launch on the stream is ops.a2a_pull's bounded barrier (fence.proxy.alias, release at system scope,
 * acquire on the own pad); a consumer reads its gathered operand only after that barrier; a slot is rewritten only after every
 * rank has finished reading it (two slots by layer parity).
 */

#include <sgl_kernel/tensor.h>
#include <sgl_kernel/utils.h>

#include <sgl_kernel/runtime.cuh>
#include <sgl_kernel/utils.cuh>

#include <cuda_runtime.h>

#include <cstdint>

namespace muse_vr {
namespace ucpd {

constexpr int kThreads = 512;
constexpr int kMaxRanks = 8;
constexpr uint32_t kTileRows = 128;
constexpr uint32_t kTileBytes = 512;

// In-tile word index of row m (0 .. 127) and its inverse (the same map as mc_push_dense.cuh).
__host__ __device__ constexpr uint32_t in_tile_word(uint32_t m) { return (m & 31u) * 4u + ((m & 127u) >> 5); }
__host__ __device__ constexpr uint32_t in_tile_row(uint32_t wi) { return (wi & 3u) * 32u + (wi >> 2); }

constexpr bool in_tile_map_is_bijection() {
  bool seen[kTileRows] = {};
  for (uint32_t m = 0; m < kTileRows; ++m) {
    const uint32_t w = in_tile_word(m);
    if (w >= kTileRows || seen[w] || in_tile_row(w) != m) return false;
    seen[w] = true;
  }
  return true;
}
static_assert(in_tile_map_is_bijection(), "the 128-row in-tile scale word map must be a bijection");

struct Params {
  const uint4* src_q;          // local values, nq 16-byte vectors ([rows, K] row-major)
  const uint32_t* src_sf;      // local swizzled scales of the [rows padded to 128, K] shard, as 4-byte words
  uint64_t dst_q[kMaxRanks];   // per rank: its mapping of global row rank_self x rows in its gathered values
  uint64_t dst_sf[kMaxRanks];  // per rank: its mapping of its gathered scales' byte 0
  uint32_t world;
  uint32_t nq;                 // rows x K / 16
  uint32_t nw;                 // local scale words incl. padding rows: rows rounded up to 128, times KT
  uint32_t rows;               // this rank's rows
  uint32_t row0;               // rank x rows
  uint32_t kt;                 // column tiles (K / 128)
};

__device__ __forceinline__ uint64_t sf_word_byte(uint32_t m, uint32_t t, uint32_t kt) {
  return (static_cast<uint64_t>(m >> 7) * kt + t) * kTileBytes + in_tile_word(m) * 4u;
}

__device__ __forceinline__ void st16(uint64_t addr, const uint4& v) {
  asm volatile("st.global.v4.u32 [%0], {%1, %2, %3, %4};" ::"l"(addr), "r"(v.x), "r"(v.y), "r"(v.z), "r"(v.w) : "memory");
}

__device__ __forceinline__ void st4(uint64_t addr, uint32_t v) {
  asm volatile("st.global.b32 [%0], %1;" ::"l"(addr), "r"(v) : "memory");
}

template <int kUnroll, int kFence>
__global__ void __launch_bounds__(kThreads) uc_push_dense_kernel(const __grid_constant__ Params p) {
  const uint32_t n = p.nq + p.nw;
  const uint32_t stride = gridDim.x * kThreads * kUnroll;
  for (uint32_t e0 = blockIdx.x * kThreads * kUnroll + threadIdx.x; e0 < n; e0 += stride) {
    uint4 val[kUnroll];
    uint32_t word[kUnroll];
    uint64_t off[kUnroll];
    bool val_part[kUnroll];
    bool live[kUnroll];
#pragma unroll
    for (int i = 0; i < kUnroll; ++i) {
      const uint32_t e = e0 + static_cast<uint32_t>(i) * kThreads;
      off[i] = 0;
      word[i] = 0;
      live[i] = false;
      val_part[i] = e < p.nq;
      if (e < p.nq) {
        val[i] = __ldcg(p.src_q + e);
        off[i] = static_cast<uint64_t>(e) * 16u;
        live[i] = true;
      } else if (e < n) {
        const uint32_t w = e - p.nq;     // local word, in the local layout's memory order
        const uint32_t lt = w >> 7;      // local tile: row tile lt / KT, column tile lt % KT
        const uint32_t rt = lt / p.kt;
        const uint32_t t = lt - rt * p.kt;
        const uint32_t r = rt * kTileRows + in_tile_row(w & 127u);
        if (r < p.rows) {
          word[i] = __ldcg(p.src_sf + w);
          off[i] = sf_word_byte(p.row0 + r, t, p.kt);
          live[i] = true;
        }
      }
    }
#pragma unroll
    for (int i = 0; i < kUnroll; ++i) {
      if (!live[i]) continue;
      for (uint32_t r = 0; r < p.world; ++r) {
        if (val_part[i]) {
          st16(p.dst_q[r] + off[i], val[i]);
        } else {
          st4(p.dst_sf[r] + off[i], word[i]);
        }
      }
    }
  }
  if constexpr (kFence) {
    // The stores above reached the peers' copies through imported mappings (aliases of the owners' addresses); every rank
    // reads its copy through its own address after the following barrier. fence.proxy.alias orders accesses made through
    // aliased addresses (PTX ISA, memory consistency model).
    asm volatile("fence.proxy.alias;" ::: "memory");
  }
}

template <int kUnroll, int kFence>
struct UcPushDenseKernel {
  // src_q: uint8 values of this rank's shard (>= rows x K bytes); src_sf: uint8 swizzled scales of the shard (>= the scale bytes
  // of rows rounded up to 128); q0..q7 / s0..s7: per rank (ranks >= world ignored), the address through which this rank reaches
  // that rank's gathered slot value byte 0 / scale byte 0 (its own address for itself); rows, rank, world, k: this rank's rows
  // (the same on every rank), its rank, the group size, the row length (a multiple of 128); q_cap / sf_cap: bytes of the slot's
  // value / scale regions.
  static void run(const tvm::ffi::TensorView src_q, const tvm::ffi::TensorView src_sf, int64_t q0, int64_t q1, int64_t q2,
                  int64_t q3, int64_t q4, int64_t q5, int64_t q6, int64_t q7, int64_t s0, int64_t s1, int64_t s2, int64_t s3,
                  int64_t s4, int64_t s5, int64_t s6, int64_t s7, int64_t rows, int64_t rank, int64_t world, int64_t k,
                  int64_t q_cap, int64_t sf_cap, int64_t grid) {
    using namespace sglang::host;
    RuntimeCheck(src_q.dim() == 1 && src_q.dtype().bits == 8 && src_q.stride(0) == 1 && src_sf.dim() == 1 &&
                     src_sf.dtype().bits == 8 && src_sf.stride(0) == 1,
                 "contiguous 1-D 8-bit sources");
    RuntimeCheck(rows > 0 && world > 0 && world <= kMaxRanks && rank >= 0 && rank < world && k > 0 && k % 128 == 0,
                 "rows > 0, 1 <= world <= 8, 0 <= rank < world, K a multiple of 128");
    const int64_t kt = k / 128;
    const int64_t rows_pad = (rows + 127) / 128 * 128;
    const int64_t m_pad = (world * rows + 127) / 128 * 128;
    RuntimeCheck(src_q.size(0) >= rows * k && src_sf.size(0) >= rows_pad * kt * 4, "sources hold the bytes pushed");
    RuntimeCheck(world * rows * k <= q_cap && m_pad * kt * 4 <= sf_cap, "the gathered slot holds world x rows rows");
    RuntimeCheck(reinterpret_cast<uintptr_t>(src_q.data_ptr()) % 16 == 0 && reinterpret_cast<uintptr_t>(src_sf.data_ptr()) % 4 == 0,
                 "16-byte aligned values, 4-byte aligned scales");
    RuntimeCheck((rows * k) / 16 + rows_pad * kt < (int64_t(1) << 31), "shard too large");
    RuntimeCheck(grid >= 1, "grid >= 1");
    const int64_t qs[kMaxRanks] = {q0, q1, q2, q3, q4, q5, q6, q7};
    const int64_t ss[kMaxRanks] = {s0, s1, s2, s3, s4, s5, s6, s7};
    Params p{};
    for (int64_t r = 0; r < world; ++r) {
      RuntimeCheck(qs[r] != 0 && qs[r] % 16 == 0 && ss[r] != 0 && ss[r] % 4 == 0, "aligned destinations for every rank");
      p.dst_q[r] = static_cast<uint64_t>(qs[r]) + static_cast<uint64_t>(rank * rows * k);
      p.dst_sf[r] = static_cast<uint64_t>(ss[r]);
    }
    p.src_q = static_cast<const uint4*>(src_q.data_ptr());
    p.src_sf = static_cast<const uint32_t*>(src_sf.data_ptr());
    p.world = static_cast<uint32_t>(world);
    p.nq = static_cast<uint32_t>(rows * k / 16);
    p.nw = static_cast<uint32_t>(rows_pad * kt);
    p.rows = static_cast<uint32_t>(rows);
    p.row0 = static_cast<uint32_t>(rank * rows);
    p.kt = static_cast<uint32_t>(kt);
    LaunchKernel(dim3(static_cast<unsigned>(grid)), kThreads, src_q.device(), 0)(uc_push_dense_kernel<kUnroll, kFence>, p);
  }
};

}  // namespace ucpd
}  // namespace muse_vr
