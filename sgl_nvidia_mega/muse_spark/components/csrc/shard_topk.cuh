// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
//
// Row-wise top-16 of a bf16 matrix (the draft head's vocabulary-shard logits), sorted by value in descending order,
// equal values by ascending column index. Two launches:
//
//   chunk_topk: one warp per chunk of kChunk columns of one row; every lane holds kChunk / 32 entries as 64-bit keys
//               key = ord(value) << 48 | (0xFFFFFFFF - column) << 16 | value bits
//               where ord maps bf16 bit patterns to unsigned integers with the order of the values (+0 and -0 equal,
//               NaN above +inf, -NaN below -inf). Keys are unique per row and their numeric order is the output order.
//               The warp extracts its 16 largest keys (16 rounds of a warp-wide maximum; the owning lane drops the
//               winner) and stores them in order.
//   merge:      one block per row merges the chunks' keys the same way (per warp, then across warps) and writes the
//               values (the original bits) and the columns (int64).
//
// The selected set and order are a function of the values and columns only (no dependence on scheduling).

#pragma once

#include <sgl_kernel/tensor.h>
#include <sgl_kernel/utils.h>

#include <sgl_kernel/utils.cuh>

#include <tvm/ffi/container/tensor.h>

#include <cstdint>

namespace muse_vr::shard_topk {

using namespace sglang;

constexpr int kK = 16;
constexpr int kChunk = 512;  // columns per warp in chunk_topk
constexpr int kPerLane = kChunk / 32;
constexpr int kWarpsA = 4;
constexpr int kThreadsB = 256;

SGL_DEVICE uint32_t ord16(uint32_t b) {
  if ((b & 0x7FFFu) == 0u) return 0x8000u;  // +0 and -0
  return (b & 0x8000u) ? (~b & 0xFFFFu) : (b | 0x8000u);
}

SGL_DEVICE uint64_t make_key(uint32_t bits, uint32_t col) {
  return (static_cast<uint64_t>(ord16(bits)) << 48) | (static_cast<uint64_t>(0xFFFFFFFFu - col) << 16) |
         static_cast<uint64_t>(bits);
}

SGL_DEVICE uint64_t warp_max(uint64_t v) {
#pragma unroll
  for (int off = 16; off > 0; off >>= 1) {
    const uint64_t o = __shfl_xor_sync(0xffffffffu, v, off);
    v = o > v ? o : v;
  }
  return v;
}

// The warp's 16 largest keys among `keys` (N per lane; 0 = empty), in descending order; lane r < 16 receives key r.
template <int N>
SGL_DEVICE uint64_t warp_top16(uint64_t (&keys)[N]) {
  const int lane = threadIdx.x % 32;
  uint64_t best = 0;
#pragma unroll
  for (int i = 0; i < N; ++i) best = keys[i] > best ? keys[i] : best;
  uint64_t mine = 0;
#pragma unroll 1
  for (int r = 0; r < kK; ++r) {
    const uint64_t m = warp_max(best);
    if (lane == r) mine = m;
    if (best == m && m != 0) {
      uint64_t nb = 0;
#pragma unroll
      for (int i = 0; i < N; ++i) {
        if (keys[i] == m) keys[i] = 0;
        nb = keys[i] > nb ? keys[i] : nb;
      }
      best = nb;
    }
  }
  return mine;
}

struct Params {
  const uint16_t* x;  // [rows, ld] bf16 bits
  int64_t ld;
  int cols;
  int chunks;         // ceil(cols / kChunk)
  uint64_t* scratch;  // [rows, chunks * kK]
  uint16_t* vals;     // [rows, kK]
  int64_t* ids;       // [rows, kK]
};

template <bool kUsePDL>
__global__ __launch_bounds__(kWarpsA * 32) void chunk_topk_kernel(const __grid_constant__ Params p) {
  const int row = blockIdx.y;
  const int warp = threadIdx.x / 32, lane = threadIdx.x % 32;
  const int chunk = blockIdx.x * kWarpsA + warp;
  device::PDLWaitPrimary<kUsePDL>();
  if (chunk < p.chunks) {
    const uint16_t* xr = p.x + static_cast<int64_t>(row) * p.ld;
    uint64_t keys[kPerLane];
#pragma unroll
    for (int v = 0; v < kPerLane / 8; ++v) {
      const int col0 = chunk * kChunk + (v * 32 + lane) * 8;
      if (col0 + 8 <= p.cols) {
        const uint4 w = *reinterpret_cast<const uint4*>(xr + col0);
        const uint32_t words[4] = {w.x, w.y, w.z, w.w};
#pragma unroll
        for (int e = 0; e < 8; ++e) {
          const uint32_t b = (words[e / 2] >> ((e % 2) * 16)) & 0xFFFFu;
          keys[v * 8 + e] = make_key(b, static_cast<uint32_t>(col0 + e));
        }
      } else {
#pragma unroll
        for (int e = 0; e < 8; ++e) {
          keys[v * 8 + e] = col0 + e < p.cols ? make_key(xr[col0 + e], static_cast<uint32_t>(col0 + e)) : 0;
        }
      }
    }
    const uint64_t top = warp_top16(keys);
    if (lane < kK) p.scratch[(static_cast<int64_t>(row) * p.chunks + chunk) * kK + lane] = top;
  }
  device::PDLTriggerSecondary<kUsePDL>();
}

template <bool kUsePDL>
__global__ __launch_bounds__(kThreadsB) void merge_kernel(const __grid_constant__ Params p) {
  constexpr int kWarps = kThreadsB / 32;
  constexpr int kMaxPerLane = 8;  // up to kThreadsB * kMaxPerLane / kK chunks per row
  __shared__ uint64_t stage[kWarps * kK];
  const int row = blockIdx.x;
  const int warp = threadIdx.x / 32, lane = threadIdx.x % 32;
  device::PDLWaitPrimary<kUsePDL>();
  const int n = p.chunks * kK;
  const uint64_t* sr = p.scratch + static_cast<int64_t>(row) * n;
  uint64_t keys[kMaxPerLane];
#pragma unroll
  for (int i = 0; i < kMaxPerLane; ++i) {
    const int j = (i * kWarps + warp) * 32 + lane;
    keys[i] = j < n ? sr[j] : 0;
  }
  const uint64_t top = warp_top16(keys);
  if (lane < kK) stage[warp * kK + lane] = top;
  __syncthreads();
  if (warp == 0) {
    uint64_t k2[kWarps * kK / 32];
#pragma unroll
    for (int i = 0; i < kWarps * kK / 32; ++i) k2[i] = stage[i * 32 + lane];
    const uint64_t t = warp_top16(k2);
    if (lane < kK) {
      p.vals[row * kK + lane] = static_cast<uint16_t>(t & 0xFFFFu);
      p.ids[row * kK + lane] = static_cast<int64_t>(0xFFFFFFFFu - static_cast<uint32_t>((t >> 16) & 0xFFFFFFFFu));
    }
  }
  device::PDLTriggerSecondary<kUsePDL>();
}

template <bool kUsePDL>
struct ShardTopK {
  static constexpr int kMaxChunks = kThreadsB * 8 / kK;

  static void run(const tvm::ffi::TensorView x, const tvm::ffi::TensorView scratch, const tvm::ffi::TensorView vals,
                  const tvm::ffi::TensorView ids) {
    using namespace host;
    auto R = SymbolicSize{"rows"};
    auto C = SymbolicSize{"cols"};
    auto LD = SymbolicSize{"ld"};
    auto S = SymbolicSize{"scratch"};
    auto device_ = SymbolicDevice{};
    device_.set_options<kDLCUDA>();
    TensorMatcher({R, C}).with_strides({LD, 1}).with_dtype<bf16_t>().with_device(device_).verify(x);
    TensorMatcher({R, S}).with_dtype<int64_t>().with_device(device_).verify(scratch);
    TensorMatcher({R, kK}).with_dtype<bf16_t>().with_device(device_).verify(vals);
    TensorMatcher({R, kK}).with_dtype<int64_t>().with_device(device_).verify(ids);
    const int64_t rows = R.unwrap(), cols = C.unwrap();
    if (rows == 0) return;
    RuntimeCheck(cols >= kK && cols < (int64_t(1) << 31), "shard_topk: column count out of range");
    RuntimeCheck(LD.unwrap() % 8 == 0 && reinterpret_cast<uintptr_t>(x.data_ptr()) % 16 == 0,
                 "shard_topk: rows must be 16-byte aligned");
    const int chunks = static_cast<int>((cols + kChunk - 1) / kChunk);
    RuntimeCheck(chunks <= kMaxChunks, "shard_topk: more than ", kMaxChunks, " chunks");
    RuntimeCheck(S.unwrap() >= int64_t(chunks) * kK, "shard_topk: scratch too small");
    Params p{};
    p.x = static_cast<const uint16_t*>(x.data_ptr());
    p.ld = LD.unwrap();
    p.cols = static_cast<int>(cols);
    p.chunks = chunks;
    p.scratch = static_cast<uint64_t*>(scratch.data_ptr());
    p.vals = static_cast<uint16_t*>(vals.data_ptr());
    p.ids = static_cast<int64_t*>(ids.data_ptr());
    const auto dev = device_.unwrap();
    LaunchKernel(dim3((chunks + kWarpsA - 1) / kWarpsA, static_cast<unsigned>(rows)), kWarpsA * 32, dev)
        .enable_pdl(kUsePDL)(chunk_topk_kernel<kUsePDL>, p);
    LaunchKernel(static_cast<unsigned>(rows), kThreadsB, dev).enable_pdl(kUsePDL)(merge_kernel<kUsePDL>, p);
  }
};

}  // namespace muse_vr::shard_topk
