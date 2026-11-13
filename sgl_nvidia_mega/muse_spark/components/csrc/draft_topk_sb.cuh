// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
//
// Row-wise exact top-16 of a [rows, cols] bf16 tensor (16 <= cols <= 65528, cols % 8 == 0, 16-byte aligned rows), sorted, in one
// launch, one CTA per row, streaming the row once without staging it.
//
// Order: the composite key of csrc/draft_topk_bw.cuh (and csrc/draft_topk.cuh's 16-bit form):
// (order_key16(value) << 16) | (0xFFFF - column), descending, where order_key16 maps the bf16 bits to an unsigned 16-bit integer
// with the same total order (NaN above everything, -0 below +0). The composite is unique per row, so the selected set and its
// order are functions of the values and columns only: equal values come in ascending column order, and among equal values at the
// 16th place the lower columns are selected. Values are written back from the input (bit for bit), columns as int64.
//
// Work split: one CTA (kThreads threads) per row.
//   1. Bound. Thread t reads one 16-column chunk at an evenly spaced position of the row (absent chunks count as key 0, below every
//      real key) and keeps the largest order key of its chunk. A two-pass radix select over the kThreads chunk maxima gives B, the
//      16th largest of them. The maxima belong to distinct elements of the row, and the 16th largest of any subset of the row is at
//      most the row's 16th largest, so every element of the row's top 16 has key >= B.
//   2. Filter. The row is streamed once with 16-byte loads. A packed bf16 comparison (x >= b, unordered so NaN passes, b = the value
//      whose key is B) screens two values per instruction. It never rejects a value whose key is >= B: the key order refines the
//      float order (only -0 < +0 is added, and those compare equal as floats), and NaN passes. Each screened value then gets its
//      order key, and those with key >= B (ties at B included) are appended to a shared candidate buffer as composites. The append
//      order is never used: steps 3 and 4 depend only on the set of composites.
//   3. Select. A four-pass radix select (8 bits per pass, most significant first) over the candidates gives C, the 16th largest
//      composite; the candidates >= C are exactly the row's top 16.
//   4. One warp rank-sorts the 16 composites and writes values and columns.
//   Overflow: if more than kCap values pass the filter (tie-heavy or constant rows), the CTA instead runs the full two-byte radix
//   select of csrc/draft_topk_bw.cuh over the row in global memory, with an in-order tie collection; the output is the same. When
//   the caller passes a flag buffer, flags[row] is the number of values that passed the filter (above kCap: the overflow path
//   ran; the count stops early once past kCap).
// No state outlives a launch (no counters or scratch), so launches and CUDA-graph replays are independent.

#pragma once

#include <sgl_kernel/tensor.h>
#include <sgl_kernel/utils.h>

#include <sgl_kernel/utils.cuh>

#include <tvm/ffi/container/tensor.h>

#include <cuda_bf16.h>
#include <cstdint>
#include <cstring>

namespace muse_vr::draft_topk_sb {

using namespace sglang;

constexpr int kThreads = 256;
constexpr int kWarps = kThreads / 32;
constexpr int kK = 16;
constexpr int kCap = 2048;   // candidate buffer entries
constexpr int kUnroll = 8;   // 16-byte loads in flight per thread in the filter pass
constexpr int kMaxCols = 65528;  // 16-bit column field, a multiple of 8

struct Params {
  const uint16_t* x;  // [rows, ld] bf16 bits
  int64_t ld;         // row stride in elements (a multiple of 8)
  int rows;
  int cols;           // a multiple of 8
  uint16_t* vals;     // [rows, 16] bf16 bits
  int64_t* ids;       // [rows, 16]
  int32_t* flags;     // [rows] or nullptr: the filter's pass count (> kCap: the overflow path ran)
};

SGL_DEVICE uint32_t key16(uint32_t b) {
  if ((b & 0x7FFFu) > 0x7F80u) return 0xFFFFu;  // NaN
  return (b & 0x8000u) ? (~b & 0xFFFFu) : (b | 0x8000u);
}

// bf16 bits of a value whose order key is k (for k == 0xFFFF a NaN, which every value passes in the unordered comparison)
SGL_DEVICE uint32_t unkey16(uint32_t k) { return (k & 0x8000u) ? (k & 0x7FFFu) : (~k & 0xFFFFu); }

// 0xFFFF in each 16-bit half where x >= b (unordered: true when either is NaN)
SGL_DEVICE uint32_t geu_mask(uint32_t x, uint32_t b) {
  __nv_bfloat162 hx, hb;
  memcpy(&hx, &x, 4);
  memcpy(&hb, &b, 4);
  return __hgeu2_mask(hx, hb);
}

SGL_DEVICE void hist_add(uint32_t* hist, uint32_t bin, bool active) {
  // warp-aggregated increment of hist[bin] for the lanes with active set (all 32 lanes must call)
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

__global__ __launch_bounds__(kThreads, 4) void topk_sb_kernel(const __grid_constant__ Params p) {
  __shared__ uint32_t cand[kCap];
  __shared__ uint32_t hist[256];
  __shared__ uint32_t wsum[kWarps];
  __shared__ uint32_t sel[kK];
  __shared__ int s_bin;
  __shared__ uint32_t s_above, s_cnt, s_slot;
  const int row = blockIdx.x, t = threadIdx.x, lane = t & 31, warp = t >> 5;
  const uint16_t* xr = p.x + static_cast<int64_t>(row) * p.ld;
  const uint4* xv = reinterpret_cast<const uint4*>(xr);
  const int nvec = p.cols >> 3;
  hist[t] = 0u;  // kThreads == 256
  if (t == 0) {
    s_cnt = 0u;
    s_slot = 0u;
  }

  // 1. bound: the 16th largest of the chunk maxima
  uint32_t mk = 0u;
  {
    const int nch = p.cols >> 4;
    int j = -1;
    if (nch >= kThreads) {
      j = static_cast<int>((static_cast<int64_t>(t) * nch) / kThreads);
    } else if (t < nch) {
      j = t;
    }
    if (j >= 0) {
      const uint4 a = xv[2 * j], b = xv[2 * j + 1];
      const uint32_t w[8] = {a.x, a.y, a.z, a.w, b.x, b.y, b.z, b.w};
#pragma unroll
      for (int i = 0; i < 8; ++i) mk = max(mk, max(key16(w[i] & 0xFFFFu), key16(w[i] >> 16)));
    }
  }
  __syncthreads();
  hist_add(hist, mk >> 8, true);
  __syncthreads();
  if (warp == 0) find_bin(hist, 0u, &s_bin, &s_above);
  __syncthreads();
  const uint32_t bhi = static_cast<uint32_t>(s_bin);
  const uint32_t ahi = s_above;
  hist[t] = 0u;
  __syncthreads();
  hist_add(hist, mk & 0xFFu, (mk >> 8) == bhi);
  __syncthreads();
  if (warp == 0) find_bin(hist, ahi, &s_bin, &s_above);
  __syncthreads();
  const uint32_t B = (bhi << 8) | static_cast<uint32_t>(s_bin);
  const uint32_t ub = unkey16(B);
  const uint32_t bb = ub | (ub << 16);

  // 2. filter: one streaming pass; screen two values per compare, exact key test on the screened ones
  for (int v0 = t; v0 < nvec; v0 += kThreads * kUnroll) {
    if (*reinterpret_cast<volatile uint32_t*>(&s_cnt) > static_cast<uint32_t>(kCap)) break;  // overflow: stop early
    uint4 w[kUnroll];
#pragma unroll
    for (int u = 0; u < kUnroll; ++u) {
      const int v = v0 + u * kThreads;
      w[u] = v < nvec ? __ldcs(xv + v) : make_uint4(0u, 0u, 0u, 0u);
    }
#pragma unroll
    for (int u = 0; u < kUnroll; ++u) {
      const int v = v0 + u * kThreads;
      const uint32_t in[4] = {w[u].x, w[u].y, w[u].z, w[u].w};
      const uint32_t m = geu_mask(in[0], bb) | geu_mask(in[1], bb) | geu_mask(in[2], bb) | geu_mask(in[3], bb);
      if (v < nvec && m != 0u) {
#pragma unroll
        for (int h = 0; h < 8; ++h) {
          const uint32_t k = key16((in[h >> 1] >> (16 * (h & 1))) & 0xFFFFu);
          if (k >= B) {
            const uint32_t s = atomicAdd(&s_cnt, 1u);
            if (s < static_cast<uint32_t>(kCap)) cand[s] = (k << 16) | (0xFFFFu - static_cast<uint32_t>(v * 8 + h));
          }
        }
      }
    }
  }
  __syncthreads();
  const uint32_t n = s_cnt;
  const bool overflow = n > static_cast<uint32_t>(kCap);

  if (!overflow) {
    // 3. exact select of the 16th largest composite among the n candidates (n >= 16: the bound's own 16 elements pass)
    uint32_t prefix = 0u, pmask = 0u, prior = 0u;
#pragma unroll 1
    for (int pass = 0; pass < 4; ++pass) {
      const int sh = 24 - 8 * pass;
      hist[t] = 0u;
      __syncthreads();
      for (uint32_t i0 = 0; i0 < n; i0 += kThreads) {
        const uint32_t i = i0 + static_cast<uint32_t>(t);
        const uint32_t c = i < n ? cand[i] : 0u;
        hist_add(hist, (c >> sh) & 0xFFu, i < n && (c & pmask) == prefix);
      }
      __syncthreads();
      if (warp == 0) find_bin(hist, prior, &s_bin, &s_above);
      __syncthreads();
      prefix |= static_cast<uint32_t>(s_bin) << sh;
      pmask |= 0xFFu << sh;
      prior = s_above;
      __syncthreads();
    }
    for (uint32_t i = t; i < n; i += kThreads) {
      const uint32_t c = cand[i];
      if (c >= prefix) sel[atomicAdd(&s_slot, 1u)] = c;
    }
  } else {
    // overflow path: two-byte radix over the row in global memory, then an in-order tie collection
    hist[t] = 0u;
    __syncthreads();
    for (int v0 = 0; v0 < nvec; v0 += kThreads) {
      const int v = v0 + t;
      const bool ok = v < nvec;
      const uint4 w = ok ? xv[v] : make_uint4(0u, 0u, 0u, 0u);
      const uint32_t in[4] = {w.x, w.y, w.z, w.w};
#pragma unroll
      for (int j = 0; j < 4; ++j) {
        hist_add(hist, key16(in[j] & 0xFFFFu) >> 8, ok);
        hist_add(hist, key16(in[j] >> 16) >> 8, ok);
      }
    }
    __syncthreads();
    if (warp == 0) find_bin(hist, 0u, &s_bin, &s_above);
    __syncthreads();
    const uint32_t fhi = static_cast<uint32_t>(s_bin);
    const uint32_t fahi = s_above;
    hist[t] = 0u;
    __syncthreads();
    for (int v0 = 0; v0 < nvec; v0 += kThreads) {
      const int v = v0 + t;
      const bool ok = v < nvec;
      const uint4 w = ok ? xv[v] : make_uint4(0u, 0u, 0u, 0u);
      const uint32_t in[4] = {w.x, w.y, w.z, w.w};
#pragma unroll
      for (int j = 0; j < 4; ++j) {
        const uint32_t lo = key16(in[j] & 0xFFFFu), hi = key16(in[j] >> 16);
        hist_add(hist, lo & 0xFFu, ok && (lo >> 8) == fhi);
        hist_add(hist, hi & 0xFFu, ok && (hi >> 8) == fhi);
      }
    }
    __syncthreads();
    if (warp == 0) find_bin(hist, fahi, &s_bin, &s_above);
    __syncthreads();
    const uint32_t T = (fhi << 8) | static_cast<uint32_t>(s_bin);
    const uint32_t above = s_above;  // keys strictly greater than T (<= 15)
    const uint32_t need = static_cast<uint32_t>(kK) - above;  // ties at T to keep, lowest columns first
    const int per = (p.cols + kThreads - 1) / kThreads;
    const int c0 = t * per, c1 = min(p.cols, c0 + per);
    uint32_t ties = 0u;
    for (int c = c0; c < c1; ++c) {
      const uint32_t k = key16(xr[c]);
      if (k > T) {
        sel[atomicAdd(&s_slot, 1u)] = (k << 16) | (0xFFFFu - static_cast<uint32_t>(c));
      } else if (k == T) {
        ++ties;
      }
    }
    uint32_t incl = ties;
#pragma unroll
    for (int o = 1; o < 32; o <<= 1) {
      const uint32_t v = __shfl_up_sync(0xFFFFFFFFu, incl, o);
      if (lane >= o) incl += v;
    }
    if (lane == 31) wsum[warp] = incl;
    __syncthreads();
    if (warp == 0) {
      const uint32_t ws = lane < kWarps ? wsum[lane] : 0u;
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
        if (key16(xr[c]) == T) {
          sel[above + rank] = (T << 16) | (0xFFFFu - static_cast<uint32_t>(c));
          ++rank;
        }
      }
    }
  }
  __syncthreads();

  // 4. sort the 16 composites (descending; they are unique) and write
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
    if (lane == 0 && p.flags != nullptr) p.flags[row] = static_cast<int32_t>(n);
  }
}

struct DraftTopKSb {
  static void launch(const tvm::ffi::TensorView x, const tvm::ffi::TensorView vals, const tvm::ffi::TensorView ids,
                     int32_t* flags) {
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
    p.ld = cols;  // contiguous (checked by the caller)
    p.rows = static_cast<int>(rows);
    p.cols = static_cast<int>(cols);
    p.vals = static_cast<uint16_t*>(vals.data_ptr());
    p.ids = static_cast<int64_t*>(ids.data_ptr());
    p.flags = flags;
    const auto stream = LaunchKernel::resolve_device(x.device());
    LaunchKernel(dim3(static_cast<unsigned>(rows)), kThreads, stream)(topk_sb_kernel, p);
  }
  static void run(const tvm::ffi::TensorView x, const tvm::ffi::TensorView vals, const tvm::ffi::TensorView ids) {
    launch(x, vals, ids, nullptr);
  }
  static void run_flags(const tvm::ffi::TensorView x, const tvm::ffi::TensorView vals, const tvm::ffi::TensorView ids,
                        const tvm::ffi::TensorView flags) {
    using namespace host;
    auto R = SymbolicSize{"rows"};
    auto device_ = SymbolicDevice{};
    device_.set_options<kDLCUDA>();
    TensorMatcher({R}).with_dtype<int32_t>().with_device(device_).verify(flags);
    RuntimeCheck(R.unwrap() == x.size(0), "flags must have one entry per row");
    launch(x, vals, ids, static_cast<int32_t*>(flags.data_ptr()));
  }
};

}  // namespace muse_vr::draft_topk_sb
