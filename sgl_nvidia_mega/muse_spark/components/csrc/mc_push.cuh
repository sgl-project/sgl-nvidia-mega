// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
/**
 * \brief Multicast push of one rank's MXFP8 shard into every rank's gathered operand (``fp8_prefill``'s AGF form).
 *
 * Source: this rank's local MXFP8 shard, E4M3 values [rows, K] (row-major, rows a whole number of 128-row scale tiles)
 * and their scales in the 128 x 4 swizzled layout. Destination: a symmetric buffer with a multicast mapping that every
 * rank of the group allocated with the same layout (the gathered operand of the merged GEMM): the values of the
 * [world * rows, K] operand, then the scales of that operand. Because rows is a whole number of 128-row tiles, the
 * swizzled scale layout of the [world * rows, K] operand is the per-rank layouts one after another, so this rank's
 * values go to row offset rank * rows and its scales to byte offset rank * sf_bytes(rows) of the scale region.
 *
 * One launch stores both parts through the multicast address (PTX ISA, multimem.st: one store reaches every rank's
 * copy): 16-byte loads of the local shard, 16-byte multicast stores in the .v4.f32 form that c3_hops.cuh uses (the
 * pushed bytes are meant to be unchanged; the units compare every gathered byte with the source shard). Consecutive
 * lanes touch consecutive 16-byte vectors; each thread issues kUnroll loads before its stores.
 *
 * Ordering is the caller's: the shard must be complete before the launch (stream order); each thread ends with a
 * fence.proxy.alias after its last multicast store; the next launch on the stream is the bounded barrier of
 * ops/a2a_pull.py (fence.proxy.alias, then a release at system scope); a consumer reads
 * its gathered operand only after that barrier's acquire. The destination slot must not be rewritten until every rank
 * has finished reading it (the caller alternates two slots by layer parity, each reuse separated by a barrier).
 */

#include <sgl_kernel/tensor.h>
#include <sgl_kernel/utils.h>

#include <sgl_kernel/runtime.cuh>
#include <sgl_kernel/utils.cuh>

#include <cuda_runtime.h>

#include <cstdint>

namespace muse_vr {
namespace mcp {

constexpr int kThreads = 512;

struct Params {
  const uint4* src_q;   // local values, nq 16-byte vectors
  const uint4* src_sf;  // local scales, nsf 16-byte vectors
  uint64_t dst_q;       // multicast address of this rank's value rows in the gathered operand
  uint64_t dst_sf;      // multicast address of this rank's scales in the gathered operand
  uint32_t nq;
  uint32_t nsf;
};

__device__ __forceinline__ void mc_store16(uint64_t addr, const uint4& v) {
  asm volatile("multimem.st.relaxed.sys.global.v4.f32 [%0], {%1, %2, %3, %4};" ::"l"(addr), "r"(v.x), "r"(v.y), "r"(v.z),
               "r"(v.w)
               : "memory");
}

template <int kUnroll>
__global__ void __launch_bounds__(kThreads) mc_push_kernel(const __grid_constant__ Params p) {
  const uint32_t n = p.nq + p.nsf;
  const uint32_t stride = gridDim.x * kThreads * kUnroll;
  for (uint32_t e0 = blockIdx.x * kThreads * kUnroll + threadIdx.x; e0 < n; e0 += stride) {
    uint4 val[kUnroll];
    uint64_t dst[kUnroll];
#pragma unroll
    for (int i = 0; i < kUnroll; ++i) {
      const uint32_t e = e0 + static_cast<uint32_t>(i) * kThreads;
      dst[i] = 0;
      if (e < p.nq) {
        val[i] = __ldcg(p.src_q + e);
        dst[i] = p.dst_q + static_cast<uint64_t>(e) * 16u;
      } else if (e < n) {
        const uint32_t u = e - p.nq;
        val[i] = __ldcg(p.src_sf + u);
        dst[i] = p.dst_sf + static_cast<uint64_t>(u) * 16u;
      }
    }
#pragma unroll
    for (int i = 0; i < kUnroll; ++i) {
      if (dst[i] != 0) mc_store16(dst[i], val[i]);
    }
  }
  // The stores above went through the multicast address; every rank (this one included) reads its copy through its own
  // unicast mapping after the following barrier. fence.proxy.alias orders accesses made through aliased addresses (PTX ISA,
  // memory consistency model); it sits here, in the storing thread, as well as in the barrier kernel.
  asm volatile("fence.proxy.alias;" ::: "memory");
}

template <int kUnroll>
struct McPushKernel {
  // src_q: uint8 values of this rank's shard (>= q_bytes); src_sf: uint8 scales (>= sf_bytes); dst_q / dst_sf: multicast
  // addresses of this rank's values / scales in the gathered operand; q_bytes / sf_bytes: bytes to push (multiples of 16).
  static void run(const tvm::ffi::TensorView src_q, const tvm::ffi::TensorView src_sf, int64_t dst_q, int64_t dst_sf,
                  int64_t q_bytes, int64_t sf_bytes, int64_t grid) {
    using namespace sglang::host;
    RuntimeCheck(src_q.dim() == 1 && src_q.dtype().bits == 8 && src_q.stride(0) == 1 && src_sf.dim() == 1 &&
                     src_sf.dtype().bits == 8 && src_sf.stride(0) == 1,
                 "contiguous 1-D 8-bit sources");
    RuntimeCheck(q_bytes > 0 && q_bytes % 16 == 0 && sf_bytes > 0 && sf_bytes % 16 == 0, "whole 16-byte vectors");
    RuntimeCheck(src_q.size(0) >= q_bytes && src_sf.size(0) >= sf_bytes, "sources hold the bytes pushed");
    RuntimeCheck(reinterpret_cast<uintptr_t>(src_q.data_ptr()) % 16 == 0 &&
                     reinterpret_cast<uintptr_t>(src_sf.data_ptr()) % 16 == 0,
                 "16-byte aligned sources");
    RuntimeCheck(dst_q != 0 && dst_q % 16 == 0 && dst_sf != 0 && dst_sf % 16 == 0, "16-byte aligned multicast destinations");
    RuntimeCheck((q_bytes + sf_bytes) / 16 < (int64_t(1) << 31), "shard too large");
    RuntimeCheck(grid >= 1, "grid >= 1");
    Params p{};
    p.src_q = static_cast<const uint4*>(src_q.data_ptr());
    p.src_sf = static_cast<const uint4*>(src_sf.data_ptr());
    p.dst_q = static_cast<uint64_t>(dst_q);
    p.dst_sf = static_cast<uint64_t>(dst_sf);
    p.nq = static_cast<uint32_t>(q_bytes / 16);
    p.nsf = static_cast<uint32_t>(sf_bytes / 16);
    LaunchKernel(dim3(static_cast<unsigned>(grid)), kThreads, src_q.device(), 0)(mc_push_kernel<kUnroll>, p);
  }
};

}  // namespace mcp
}  // namespace muse_vr
