// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
//
// Row-wise exact top-16 of a [rows, cols] bf16 tensor (cols <= 65528, cols % 8 == 0, 16-byte aligned rows), sorted,
// in one launch, reading each row from memory once.
//
// Order: the same composite key as csrc/draft_topk.cuh's 16-bit form: (order_key16(value) << 16) | (0xFFFF - column),
// descending, where order_key16 maps the bf16 bits to an unsigned 16-bit integer with the same total order (NaN above
// everything, -0 below +0). The key is unique per row, so the selection and the order are functions of the values and
// columns only: equal values come in ascending column order, and among equal values at the 16th place the lower
// columns are selected. Values are written back from the input (bit for bit), columns as int64.
//
// Work split: one CTA (kThreads threads) per row.
//   1. The row is streamed in with 16-byte loads, converted to order keys and stored in dynamic shared memory; a
//      256-bin histogram of the keys' high byte is built (warp-aggregated shared-memory atomics).
//   2. One warp finds the high-byte bin B holding the 16th largest key and the count of keys in higher bins.
//   3. A second 256-bin histogram of the low byte over the keys whose high byte is B; one warp finds the threshold
//      key T (the 16th largest key's order value) and A = the count of keys strictly above T (A <= 15).
//   4. Collection in column order: thread t scans the contiguous column range [t * P, (t + 1) * P) of the staged keys;
//      keys above T are placed by an atomic slot counter (their order is fixed in step 5), keys equal to T are ranked by
//      column through a block-wide exclusive scan of the per-thread tie counts, and the first 16 - A of them are kept.
//   5. One warp sorts the 16 selected composites (rank = number of larger composites; composites are unique) and
//      writes values and columns.
// No state outlives a launch (no counters or scratch), so launches and CUDA-graph replays are independent.

#pragma once

#include <sgl_kernel/tensor.h>
#include <sgl_kernel/utils.h>

#include <sgl_kernel/utils.cuh>

#include <tvm/ffi/container/tensor.h>

#include <cuda_bf16.h>
#include <cstdint>

namespace muse_vr::draft_topk_bw {

using namespace sglang;

constexpr int kThreads = 512;
constexpr int kWarps = kThreads / 32;
constexpr int kK = 16;
constexpr int kMaxCols = 65528;  // 16-bit column field, a multiple of 8

struct Params {
  const uint16_t* x;  // [rows, ld] bf16 bits
  int64_t ld;         // row stride in elements (a multiple of 8)
  int rows;
  int cols;           // a multiple of 8
  uint16_t* vals;     // [rows, 16] bf16 bits
  int64_t* ids;       // [rows, 16]
};

SGL_DEVICE uint32_t key16(uint32_t b) {
  if ((b & 0x7FFFu) > 0x7F80u) return 0xFFFFu;  // NaN
  return (b & 0x8000u) ? (~b & 0xFFFFu) : (b | 0x8000u);
}

SGL_DEVICE void hist_add(uint32_t* hist, uint32_t bin, bool active) {
  // warp-aggregated increment of hist[bin] for the lanes with active set
  const unsigned m = __ballot_sync(0xFFFFFFFFu, active);
  if (!active) return;
  const unsigned peers = __match_any_sync(m, bin);
  if ((threadIdx.x & 31) == __ffs(peers) - 1) atomicAdd(&hist[bin], static_cast<uint32_t>(__popc(peers)));
}

// warp 0: given hist[256] and the number already above (prior), find the bin b holding the need-th largest
// (need = kK - prior) and the count of entries in bins above b; results to *bin_out / *above_out (prior included)
SGL_DEVICE void find_bin(const uint32_t* hist, uint32_t prior, int* bin_out, uint32_t* above_out) {
  const int lane = threadIdx.x & 31;
  // lane l owns bins 255 - 8l .. 248 - 8l (descending)
  uint32_t c[8];
  uint32_t s = 0;
#pragma unroll
  for (int i = 0; i < 8; ++i) {
    c[i] = hist[255 - 8 * lane - i];
    s += c[i];
  }
  uint32_t incl = s;
#pragma unroll
  for (int o = 1; o < 32; o <<= 1) {
    const uint32_t v = __shfl_up_sync(0xFFFFFFFFu, incl, o);
    if (lane >= o) incl += v;
  }
  const uint32_t excl = incl - s;
  const uint32_t need = static_cast<uint32_t>(kK) - prior;
  // the lane whose range crosses need
  const bool mine = excl < need && incl >= need;
  if (mine) {
    uint32_t run = excl;
#pragma unroll
    for (int i = 0; i < 8; ++i) {
      if (run + c[i] >= need) {
        *bin_out = 255 - 8 * lane - i;
        *above_out = prior + run;
        break;
      }
      run += c[i];
    }
  }
}

__global__ __launch_bounds__(kThreads) void topk_bw_kernel(const __grid_constant__ Params p) {
  extern __shared__ __align__(16) uint16_t keys[];
  __shared__ uint32_t hist[256];
  __shared__ uint32_t wsum[kWarps];
  __shared__ uint32_t sel[kK];
  __shared__ int s_bhi, s_blo;
  __shared__ uint32_t s_ahi, s_above, s_slot;
  const int row = blockIdx.x, t = threadIdx.x, lane = t & 31, warp = t >> 5;
  const uint16_t* xr = p.x + static_cast<int64_t>(row) * p.ld;
  for (int i = t; i < 256; i += kThreads) hist[i] = 0u;
  if (t == 0) s_slot = 0u;
  __syncthreads();

  // 1. stream the row in, keys to shared memory, high-byte histogram
  const int nvec = p.cols >> 3;
  const uint4* xv = reinterpret_cast<const uint4*>(xr);
  uint4* kv = reinterpret_cast<uint4*>(keys);
  const int iters = (nvec + kThreads - 1) / kThreads;
  for (int it = 0; it < iters; ++it) {
    const int v = it * kThreads + t;
    const bool ok = v < nvec;
    uint4 w = make_uint4(0u, 0u, 0u, 0u);
    if (ok) w = __ldcs(xv + v);
    uint32_t in[4] = {w.x, w.y, w.z, w.w};
    uint32_t outw[4];
#pragma unroll
    for (int j = 0; j < 4; ++j) {
      const uint32_t lo = key16(in[j] & 0xFFFFu), hi = key16(in[j] >> 16);
      outw[j] = lo | (hi << 16);
      hist_add(hist, lo >> 8, ok);
      hist_add(hist, hi >> 8, ok);
    }
    if (ok) kv[v] = make_uint4(outw[0], outw[1], outw[2], outw[3]);
  }
  __syncthreads();

  // 2. high byte bin
  if (warp == 0) find_bin(hist, 0u, &s_bhi, &s_ahi);
  __syncthreads();
  const uint32_t bhi = static_cast<uint32_t>(s_bhi);
  for (int i = t; i < 256; i += kThreads) hist[i] = 0u;
  __syncthreads();

  // 3. low byte histogram inside the high bin
  for (int it = 0; it < iters; ++it) {
    const int v = it * kThreads + t;
    const bool ok = v < nvec;
    uint4 w = make_uint4(0u, 0u, 0u, 0u);
    if (ok) w = kv[v];
    const uint32_t in[4] = {w.x, w.y, w.z, w.w};
#pragma unroll
    for (int j = 0; j < 4; ++j) {
      const uint32_t lo = in[j] & 0xFFFFu, hi = in[j] >> 16;
      hist_add(hist, lo & 0xFFu, ok && (lo >> 8) == bhi);
      hist_add(hist, hi & 0xFFu, ok && (hi >> 8) == bhi);
    }
  }
  __syncthreads();
  if (warp == 0) find_bin(hist, s_ahi, &s_blo, &s_above);
  __syncthreads();
  const uint32_t T = (bhi << 8) | static_cast<uint32_t>(s_blo);
  const uint32_t above = s_above;  // keys strictly greater than T
  const uint32_t need = static_cast<uint32_t>(kK) - above;  // ties at T to keep, lowest columns first

  // 4. collection in column order over contiguous per-thread ranges
  const int per = (p.cols + kThreads - 1) / kThreads;
  const int c0 = t * per, c1 = min(p.cols, c0 + per);
  uint32_t ties = 0u;
  for (int c = c0; c < c1; ++c) {
    const uint32_t k = keys[c];
    if (k > T) {
      const uint32_t s = atomicAdd(&s_slot, 1u);
      sel[s] = (k << 16) | (0xFFFFu - static_cast<uint32_t>(c));
    } else if (k == T) {
      ++ties;
    }
  }
  // block exclusive scan of ties
  uint32_t incl = ties;
#pragma unroll
  for (int o = 1; o < 32; o <<= 1) {
    const uint32_t v = __shfl_up_sync(0xFFFFFFFFu, incl, o);
    if (lane >= o) incl += v;
  }
  if (lane == 31) wsum[warp] = incl;
  __syncthreads();
  if (warp == 0) {
    uint32_t ws = lane < kWarps ? wsum[lane] : 0u;
    uint32_t wi = ws;
#pragma unroll
    for (int o = 1; o < 32; o <<= 1) {
      const uint32_t v = __shfl_up_sync(0xFFFFFFFFu, wi, o);
      if (lane >= o) wi += v;
    }
    if (lane < kWarps) wsum[lane] = wi - ws;  // exclusive warp offsets
  }
  __syncthreads();
  uint32_t rank = wsum[warp] + incl - ties;
  if (rank < need && ties > 0u) {
    for (int c = c0; c < c1 && rank < need; ++c) {
      if (keys[c] == T) {
        sel[above + rank] = (T << 16) | (0xFFFFu - static_cast<uint32_t>(c));
        ++rank;
      }
    }
  }
  __syncthreads();

  // 5. sort the 16 composites (descending) and write
  if (warp == 0) {
    const uint32_t v = lane < kK ? sel[lane] : 0u;
    uint32_t r = 0u;
#pragma unroll
    for (int j = 0; j < kK; ++j) {
      const uint32_t o = __shfl_sync(0xFFFFFFFFu, v, j);
      r += (o > v) ? 1u : 0u;
    }
    if (lane < kK) {
      const uint32_t col = 0xFFFFu - (v & 0xFFFFu);
      p.ids[static_cast<int64_t>(row) * kK + r] = static_cast<int64_t>(col);
      p.vals[static_cast<int64_t>(row) * kK + r] = xr[col];
    }
  }
}

struct DraftTopKBw {
  static void run(const tvm::ffi::TensorView x, const tvm::ffi::TensorView vals, const tvm::ffi::TensorView ids) {
    using namespace host;
    auto R = SymbolicSize{"rows"};
    auto C = SymbolicSize{"cols"};
    auto K = SymbolicSize{"k"};
    auto device_ = SymbolicDevice{};
    device_.set_options<kDLCUDA>();
    TensorMatcher({R, C}).with_dtype<bf16_t>().with_device(device_).verify(x);
    TensorMatcher({R, K}).with_dtype<int64_t>().with_device(device_).verify(ids);
    TensorMatcher({R, K}).with_dtype<bf16_t>().with_device(device_).verify(vals);
    const int64_t rows = R.unwrap(), cols = C.unwrap();
    RuntimeCheck(K.unwrap() == kK, "k must be 16");
    RuntimeCheck(cols >= kK && cols <= kMaxCols && cols % 8 == 0, "cols out of range");
    RuntimeCheck(reinterpret_cast<uintptr_t>(x.data_ptr()) % 16 == 0, "x must be 16-byte aligned");
    if (rows == 0) return;
    Params p{};
    p.x = static_cast<const uint16_t*>(x.data_ptr());
    p.ld = cols;  // contiguous (checked above)
    p.rows = static_cast<int>(rows);
    p.cols = static_cast<int>(cols);
    p.vals = static_cast<uint16_t*>(vals.data_ptr());
    p.ids = static_cast<int64_t*>(ids.data_ptr());
    const int smem = static_cast<int>(cols) * 2;
    static int attr = 0;
    if (smem > attr) {
      RuntimeDeviceCheck(cudaFuncSetAttribute(topk_bw_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem));
      attr = smem;
    }
    const auto stream = LaunchKernel::resolve_device(x.device());
    LaunchKernel(dim3(static_cast<unsigned>(rows)), kThreads, stream, smem)(topk_bw_kernel, p);
  }
};

}  // namespace muse_vr::draft_topk_bw
