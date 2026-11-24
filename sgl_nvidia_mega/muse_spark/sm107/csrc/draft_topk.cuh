// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
//
// Row-wise exact top-k (k <= kK <= 32) of a [rows, cols] bf16 / fp16 / fp32 tensor, sorted, in one launch.
//
// Order: the composite key (key(v) << 32 | (0xFFFFFFFF - column)), descending, where key() maps the value (converted to
// fp32 exactly) to an unsigned integer with the same total order (NaN above everything, -0 below +0). It is unique per
// row, so the selection and the order are functions of the values and columns only: equal values come in ascending
// column order, and among equal values at the k-th place the lower columns are selected. For 16-bit inputs the same
// order is carried by a 32-bit key (16-bit order key of the value << 16 | 0xFFFF - column; columns < 65536), and a
// warp's maximum is one warp-wide reduce instruction.
//
// Work split: grid (rows, S). CTA (r, s) handles columns [s * L, min(cols, (s + 1) * L)), L = ceil(cols / S), with 256
// threads; thread t holds the columns t, t + 256, ... of the slice (at most kE of them), sorted in registers. Each warp
// extracts its top-k (k rounds of a warp-wide maximum; the lane holding it drops it), the CTA merges its warps' lists
// the same way, and writes its k keys to the scratch. The CTA that increments the row's counter last merges the S lists
// of the row and writes the values (read back from the input, bit for bit) and the int64 columns, then resets the
// counter, so consecutive launches (and CUDA graph replays) reuse the counters. Launches that use the same counters
// must not overlap in time (stream order is enough).

#pragma once

#include <sgl_kernel/tensor.h>
#include <sgl_kernel/utils.h>

#include <sgl_kernel/utils.cuh>

#include <tvm/ffi/container/tensor.h>

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cstdint>

namespace muse_vr::draft_topk {

using namespace sglang;

constexpr int kThreads = 256;
constexpr int kWarps = kThreads / 32;
constexpr int kMaxSplits = 32;
constexpr int kMaxSlice = 2048;  // columns per CTA
constexpr int kE = kMaxSlice / kThreads;
constexpr int kMaxCols = kMaxSplits * kMaxSlice;

SGL_DEVICE uint32_t order_key(uint32_t bits) {
  const float f = __uint_as_float(bits);
  if (f != f) return 0xFFFFFFFFu;
  return (bits & 0x80000000u) ? ~bits : (bits | 0x80000000u);
}

SGL_DEVICE uint64_t composite(uint32_t bits, uint32_t col) {
  return (static_cast<uint64_t>(order_key(bits)) << 32) | static_cast<uint64_t>(0xFFFFFFFFu - col);
}

// 16-bit order key: NaN above everything, negative values bit-inverted, positive values with the sign bit set
SGL_DEVICE uint32_t order_key16(uint32_t bits16, uint32_t exp_mask) {
  if ((bits16 & 0x7FFFu) > exp_mask) return 0xFFFFu;
  return (bits16 & 0x8000u) ? (~bits16 & 0xFFFFu) : (bits16 | 0x8000u);
}

template <typename T>
struct Key;  // key type, construction and column of a composite key per input dtype

template <>
struct Key<uint16_t> {  // bf16
  using type = uint32_t;
  SGL_DEVICE static type make(const uint16_t* p, int c) {
    return (order_key16(p[c], 0x7F80u) << 16) | (0xFFFFu - static_cast<uint32_t>(c));
  }
  SGL_DEVICE static int col(type k) { return static_cast<int>(0xFFFFu - (k & 0xFFFFu)); }
};

template <>
struct Key<__half> {
  using type = uint32_t;
  SGL_DEVICE static type make(const __half* p, int c) {
    return (order_key16(__half_as_ushort(p[c]), 0x7C00u) << 16) | (0xFFFFu - static_cast<uint32_t>(c));
  }
  SGL_DEVICE static int col(type k) { return static_cast<int>(0xFFFFu - (k & 0xFFFFu)); }
};

template <>
struct Key<float> {
  using type = uint64_t;
  SGL_DEVICE static type make(const float* p, int c) { return composite(__float_as_uint(p[c]), static_cast<uint32_t>(c)); }
  SGL_DEVICE static int col(type k) { return static_cast<int>(0xFFFFFFFFu - static_cast<uint32_t>(k & 0xFFFFFFFFu)); }
};

SGL_DEVICE uint32_t warp_max(uint32_t v) {
  return __reduce_max_sync(0xffffffffu, v);
}

SGL_DEVICE uint64_t warp_max(uint64_t v) {
#pragma unroll
  for (int off = 16; off > 0; off >>= 1) {
    const uint64_t o = __shfl_xor_sync(0xffffffffu, v, off);
    v = o > v ? o : v;
  }
  return v;
}

// k rounds of extraction over per-lane descending lists a[0..kN) (0 = empty): out[r] (lane 0) = r-th largest key.
template <int kN, typename KT>
SGL_DEVICE void extract(KT (&a)[kN], int k, KT* out) {
  const int lane = threadIdx.x % 32;
  for (int r = 0; r < k; ++r) {
    const KT m = warp_max(a[0]);
    if (m != 0 && a[0] == m) {
#pragma unroll
      for (int i = 0; i + 1 < kN; ++i) a[i] = a[i + 1];
      a[kN - 1] = 0;
    }
    if (lane == 0) out[r] = m;
  }
}

template <int kN, typename KT>
SGL_DEVICE void sort_desc(KT (&a)[kN]) {
#pragma unroll
  for (int i = 1; i < kN; ++i) {
#pragma unroll
    for (int j = i; j > 0; --j) {
      if (a[j] > a[j - 1]) {
        const KT t = a[j];
        a[j] = a[j - 1];
        a[j - 1] = t;
      }
    }
  }
}

struct Params {
  const void* x;
  int64_t ld;  // row stride of x (elements; x is contiguous)
  int rows;
  int cols;
  int k;
  int splits;
  int slice;
  void* scratch;      // [rows, splits, kK] keys (32- or 64-bit)
  uint32_t* counters; // [rows], zero between launches
  void* vals;         // [rows, k] dtype of x
  int64_t* ids;       // [rows, k]
};

template <typename T, int kK>
__global__ __launch_bounds__(kThreads) void topk_kernel(const __grid_constant__ Params p) {
  using KT = typename Key<T>::type;
  __shared__ KT wl[kWarps][kK];
  __shared__ KT cl[kK];
  __shared__ bool last;
  const int row = blockIdx.x, s = blockIdx.y, t = threadIdx.x, lane = t % 32, warp = t / 32;
  const T* xr = static_cast<const T*>(p.x) + static_cast<int64_t>(row) * p.ld;
  const int c0 = s * p.slice;
  const int c1 = min(p.cols, c0 + p.slice);

  KT a[kE];
#pragma unroll
  for (int i = 0; i < kE; ++i) {
    const int c = c0 + t + i * kThreads;
    a[i] = c < c1 ? Key<T>::make(xr, c) : KT(0);
  }
  sort_desc(a);
  extract(a, p.k, wl[warp]);
  __syncthreads();
  if (warp == 0) {
    KT b[kK];
#pragma unroll
    for (int i = 0; i < kK; ++i) b[i] = (lane < kWarps && i < p.k) ? wl[lane][i] : KT(0);
    extract(b, p.k, cl);
  }
  __syncthreads();
  KT* sc = static_cast<KT*>(p.scratch) + (static_cast<int64_t>(row) * p.splits + s) * kK;
  if (t < p.k) {
    sc[t] = cl[t];
    __threadfence();
  }
  __syncthreads();
  if (t == 0) last = atomicAdd(&p.counters[row], 1u) == static_cast<uint32_t>(p.splits - 1);
  __syncthreads();
  if (!last) return;
  __threadfence();
  if (warp == 0) {
    KT b[kK];
    const KT* rr = static_cast<const KT*>(p.scratch) + static_cast<int64_t>(row) * p.splits * kK;
#pragma unroll
    for (int i = 0; i < kK; ++i) b[i] = (lane < p.splits && i < p.k) ? __ldcg(rr + lane * kK + i) : KT(0);
    extract(b, p.k, cl);
    __syncwarp();
    for (int r = lane; r < p.k; r += 32) {
      const int c = Key<T>::col(cl[r]);
      p.ids[static_cast<int64_t>(row) * p.k + r] = c;
      static_cast<T*>(p.vals)[static_cast<int64_t>(row) * p.k + r] = xr[c];
    }
    if (lane == 0) p.counters[row] = 0u;
  }
}

template <int kK>
struct DraftTopK {
  static void run(const tvm::ffi::TensorView x, const tvm::ffi::TensorView vals, const tvm::ffi::TensorView ids,
                  const tvm::ffi::TensorView scratch, const tvm::ffi::TensorView counters, int64_t k, int64_t splits) {
    using namespace host;
    auto R = SymbolicSize{"rows"};
    auto C = SymbolicSize{"cols"};
    auto K = SymbolicSize{"k"};
    auto device_ = SymbolicDevice{};
    device_.set_options<kDLCUDA>();
    TensorMatcher({R, C}).with_dtype<bf16_t, fp16_t, fp32_t>().with_device(device_).verify(x);
    TensorMatcher({R, K}).with_dtype<int64_t>().with_device(device_).verify(ids);
    TensorMatcher({R, K}).with_dtype<bf16_t, fp16_t, fp32_t>().with_device(device_).verify(vals);
    const int64_t rows = R.unwrap(), cols = C.unwrap();
    RuntimeCheck(K.unwrap() == k && k >= 1 && k <= kK, "k out of range");
    RuntimeCheck(cols >= k && cols <= kMaxCols, "cols out of range");
    RuntimeCheck(splits >= 1 && splits <= kMaxSplits, "splits out of range");
    const int64_t slice = (cols + splits - 1) / splits;
    RuntimeCheck(slice <= kMaxSlice, "slice too large for the split count");
    RuntimeCheck(scratch.numel() >= rows * splits * kK, "scratch too small");
    RuntimeCheck(counters.numel() >= rows, "counters too small");
    RuntimeCheck(x.dtype().code == vals.dtype().code && x.dtype().bits == vals.dtype().bits, "vals must have the dtype of x");
    if (rows == 0) return;
    Params p{};
    p.x = x.data_ptr();
    p.ld = cols;  // contiguous (checked above)
    p.rows = static_cast<int>(rows);
    p.cols = static_cast<int>(cols);
    p.k = static_cast<int>(k);
    p.splits = static_cast<int>(splits);
    p.slice = static_cast<int>(slice);
    p.scratch = scratch.data_ptr();
    p.counters = static_cast<uint32_t*>(counters.data_ptr());
    p.vals = vals.data_ptr();
    p.ids = static_cast<int64_t*>(ids.data_ptr());
    const dim3 grid(static_cast<unsigned>(rows), static_cast<unsigned>(splits));
    const auto code = x.dtype().code;
    const auto bits = x.dtype().bits;
    if (code == kDLBfloat && bits == 16) {
      LaunchKernel(grid, kThreads, device_.unwrap())(topk_kernel<uint16_t, kK>, p);
    } else if (code == kDLFloat && bits == 16) {
      LaunchKernel(grid, kThreads, device_.unwrap())(topk_kernel<__half, kK>, p);
    } else if (code == kDLFloat && bits == 32) {
      LaunchKernel(grid, kThreads, device_.unwrap())(topk_kernel<float, kK>, p);
    } else {
      RuntimeCheck(false, "unsupported dtype");
    }
  }
};

}  // namespace muse_vr::draft_topk
