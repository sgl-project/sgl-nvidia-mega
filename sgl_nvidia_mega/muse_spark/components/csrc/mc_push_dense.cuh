// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
/**
 * \brief Dense multicast push of one rank's MXFP8 shard into every rank's gathered operand (a variant of mc_push.cuh).
 *
 * mc_push.cuh places rank p's block at p x rp rows (rp = rows rounded up to whole 128-row scale tiles), so the gathered operand
 * holds padding rows between the rank blocks and its GEMM output must be compacted. This kernel places rank p's rows at
 * p x rows instead: the gathered operand is the DENSE [world x rows, K] operand, and its scales are written straight into that
 * operand's 128 x 4 swizzled layout, so one GEMM over world x rows rows gives the final output.
 *
 * Scale layout (the 128 x 4 swizzled layout the GEMM reads; same formula as b3_norm_mx.cuh's sf_row_offset plus the column
 * tile): the 4 scales of row m and column tile t (scale columns 4t .. 4t + 3) are one 4-byte word at byte
 *   W(m, t) = (m / 128) x KT x 512 + t x 512 + (m % 32) x 16 + ((m % 128) / 32) x 4,   KT = K / 128.
 *
 * Values: this rank's [rows, K] E4M3 rows (row-major, contiguous) are stored byte for byte with 16-byte multicast stores at
 * global row rank x rows of the gathered values.
 * Scales: the local scales (the same layout over this rank's rows, padded to 128) are read word by word in their memory order
 * (consecutive lanes read consecutive words); local word (r, t) with r < rows is stored with a 4-byte multicast bit store at
 * the global word W(rank x rows + r, t). Padding rows of the local layout are not stored.
 *
 * Write sets: the kernel derives its first global row as rank x rows, so rank p stores exactly the value bytes of rows
 * [p x rows, (p + 1) x rows) and the scale words of those rows. W is injective on {(m, t) : t < KT} (the in-tile map is a
 * bijection, checked below at compile time, and distinct (m / 128, t) are distinct 512-byte tiles), so with the same rows on
 * every rank no two ranks store the same byte. The global layout's padding rows (world x rows up to the next 128) are never
 * stored; the caller zeroes the buffer once at allocation.
 *
 * One launch stores both parts through the multicast address (PTX ISA, multimem.st: one store reaches every rank's copy);
 * each thread issues kUnroll loads before its stores and ends with a fence.proxy.alias after its last store of either kind.
 * Ordering is the caller's, as for mc_push.cuh: the shard is complete before the launch (stream order); the next launch on the
 * stream is ops.a2a_pull's bounded barrier (fence.proxy.alias, release at system scope); a consumer reads its gathered operand
 * only after that barrier's acquire; a slot is rewritten only after every rank has finished reading it (two slots by layer
 * parity).
 */

#include <sgl_kernel/tensor.h>
#include <sgl_kernel/utils.h>

#include <sgl_kernel/runtime.cuh>
#include <sgl_kernel/utils.cuh>

#include <cuda_runtime.h>

#include <cstdint>

namespace muse_vr {
namespace mcpd {

constexpr int kThreads = 512;
constexpr uint32_t kTileRows = 128;
constexpr uint32_t kTileBytes = 512;

// In-tile word index of row m (0 .. 127): byte ((m % 32) x 16 + ((m % 128) / 32) x 4) / 4.
__host__ __device__ constexpr uint32_t in_tile_word(uint32_t m) { return (m & 31u) * 4u + ((m & 127u) >> 5); }

// Its inverse: the row (within the tile) whose word is wi.
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
  const uint4* src_q;         // local values, nq 16-byte vectors ([rows, K] row-major)
  const uint32_t* src_sf;     // local swizzled scales of the [rows padded to 128, K] shard, as 4-byte words
  uint64_t dst_q;             // multicast address of global row rank x rows in the gathered values
  uint64_t dst_sf;            // multicast address of the gathered scales' byte 0
  uint32_t nq;                // rows x K / 16
  uint32_t nw;                // local scale words incl. padding rows: rows rounded up to 128, times KT
  uint32_t rows;              // this rank's rows
  uint32_t row0;              // rank x rows
  uint32_t kt;                // column tiles (K / 128)
};

__device__ __forceinline__ uint64_t sf_word_byte(uint32_t m, uint32_t t, uint32_t kt) {
  return (static_cast<uint64_t>(m >> 7) * kt + t) * kTileBytes + in_tile_word(m) * 4u;
}

__device__ __forceinline__ void mc_store16(uint64_t addr, const uint4& v) {
  asm volatile("multimem.st.relaxed.sys.global.v4.f32 [%0], {%1, %2, %3, %4};" ::"l"(addr), "r"(v.x), "r"(v.y), "r"(v.z),
               "r"(v.w)
               : "memory");
}

__device__ __forceinline__ void mc_store4(uint64_t addr, uint32_t v) {
  asm volatile("multimem.st.relaxed.sys.global.b32 [%0], %1;" ::"l"(addr), "r"(v) : "memory");
}

template <int kUnroll>
__global__ void __launch_bounds__(kThreads) mc_push_dense_kernel(const __grid_constant__ Params p) {
  const uint32_t n = p.nq + p.nw;
  const uint32_t stride = gridDim.x * kThreads * kUnroll;
  for (uint32_t e0 = blockIdx.x * kThreads * kUnroll + threadIdx.x; e0 < n; e0 += stride) {
    uint4 val[kUnroll];
    uint32_t word[kUnroll];
    uint64_t dst[kUnroll];
#pragma unroll
    for (int i = 0; i < kUnroll; ++i) {
      const uint32_t e = e0 + static_cast<uint32_t>(i) * kThreads;
      dst[i] = 0;
      word[i] = 0;
      if (e < p.nq) {
        val[i] = __ldcg(p.src_q + e);
        dst[i] = p.dst_q + static_cast<uint64_t>(e) * 16u;
      } else if (e < n) {
        const uint32_t w = e - p.nq;     // local word, in the local layout's memory order
        const uint32_t lt = w >> 7;      // local tile: row tile lt / KT, column tile lt % KT
        const uint32_t rt = lt / p.kt;
        const uint32_t t = lt - rt * p.kt;
        const uint32_t r = rt * kTileRows + in_tile_row(w & 127u);
        if (r < p.rows) {
          word[i] = __ldcg(p.src_sf + w);
          dst[i] = p.dst_sf + sf_word_byte(p.row0 + r, t, p.kt);
        }
      }
    }
#pragma unroll
    for (int i = 0; i < kUnroll; ++i) {
      const uint32_t e = e0 + static_cast<uint32_t>(i) * kThreads;
      if (dst[i] == 0) continue;
      if (e < p.nq) {
        mc_store16(dst[i], val[i]);
      } else {
        mc_store4(dst[i], word[i]);
      }
    }
  }
  // The stores above (values and scales) went through the multicast address; every rank (this one included) reads its copy
  // through its own unicast mapping after the following barrier. fence.proxy.alias orders accesses made through aliased
  // addresses (PTX ISA, memory consistency model); it sits here, in the storing thread after its last store of either kind,
  // as well as in the barrier kernel.
  asm volatile("fence.proxy.alias;" ::: "memory");
}

template <int kUnroll>
struct McPushDenseKernel {
  // src_q: uint8 values of this rank's shard (>= rows x K bytes); src_sf: uint8 swizzled scales of the shard (>= the scale bytes
  // of rows rounded up to 128); dst_q / dst_sf: multicast addresses of the gathered slot's value byte 0 / scale byte 0;
  // rows, rank, world, k: this rank's rows (the same on every rank), its rank, the group size, the row length (a multiple of
  // 128); q_cap / sf_cap: bytes of the slot's value / scale regions.
  static void run(const tvm::ffi::TensorView src_q, const tvm::ffi::TensorView src_sf, int64_t dst_q, int64_t dst_sf, int64_t rows,
                  int64_t rank, int64_t world, int64_t k, int64_t q_cap, int64_t sf_cap, int64_t grid) {
    using namespace sglang::host;
    RuntimeCheck(src_q.dim() == 1 && src_q.dtype().bits == 8 && src_q.stride(0) == 1 && src_sf.dim() == 1 &&
                     src_sf.dtype().bits == 8 && src_sf.stride(0) == 1,
                 "contiguous 1-D 8-bit sources");
    RuntimeCheck(rows > 0 && world > 0 && rank >= 0 && rank < world && k > 0 && k % 128 == 0,
                 "rows > 0, 0 <= rank < world, K a multiple of 128");
    const int64_t kt = k / 128;
    const int64_t rows_pad = (rows + 127) / 128 * 128;
    const int64_t m_pad = (world * rows + 127) / 128 * 128;
    RuntimeCheck(src_q.size(0) >= rows * k && src_sf.size(0) >= rows_pad * kt * 4, "sources hold the bytes pushed");
    RuntimeCheck(world * rows * k <= q_cap && m_pad * kt * 4 <= sf_cap, "the gathered slot holds world x rows rows");
    RuntimeCheck(reinterpret_cast<uintptr_t>(src_q.data_ptr()) % 16 == 0 && reinterpret_cast<uintptr_t>(src_sf.data_ptr()) % 4 == 0,
                 "16-byte aligned values, 4-byte aligned scales");
    RuntimeCheck(dst_q != 0 && dst_q % 16 == 0 && dst_sf != 0 && dst_sf % 4 == 0, "aligned multicast destinations");
    RuntimeCheck((rows * k) / 16 + rows_pad * kt < (int64_t(1) << 31), "shard too large");
    RuntimeCheck(grid >= 1, "grid >= 1");
    Params p{};
    p.src_q = static_cast<const uint4*>(src_q.data_ptr());
    p.src_sf = static_cast<const uint32_t*>(src_sf.data_ptr());
    p.dst_q = static_cast<uint64_t>(dst_q) + static_cast<uint64_t>(rank * rows * k);
    p.dst_sf = static_cast<uint64_t>(dst_sf);
    p.nq = static_cast<uint32_t>(rows * k / 16);
    p.nw = static_cast<uint32_t>(rows_pad * kt);
    p.rows = static_cast<uint32_t>(rows);
    p.row0 = static_cast<uint32_t>(rank * rows);
    p.kt = static_cast<uint32_t>(kt);
    LaunchKernel(dim3(static_cast<unsigned>(grid)), kThreads, src_q.device(), 0)(mc_push_dense_kernel<kUnroll>, p);
  }
};

}  // namespace mcpd
}  // namespace muse_vr
