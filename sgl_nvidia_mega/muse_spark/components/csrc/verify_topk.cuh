// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
//
// Exact distributed top-k for target-verify sampling over a vocabulary-parallel head.
//
// Order of the top-k: the composite key (key(s) << 32 | ~column), descending, where key() is the radix order torch.topk
// uses for fp32 (NaN above everything, -0 below +0). It is unique per row; equal values come in ascending column
// order. Order of the argmax: (key2(u) << 32 | ~column), descending, where key2() is the order torch.argmax uses
// (NaN above everything, -0 equal to +0): the first maximum, or the first NaN.
//
// The prep pass (Triton, in verify_topk_sample.py) writes, per row of one rank's shard: the scaled values s, the
// largest key(s) of every bucket of kBucket consecutive columns, and (optionally) the argmax composite of every block.
//
// select (one block per row):
//   1. bucket order: B_j = key_j << 32 | ~(first column of bucket j). For buckets with different maximum keys it is
//      the order of their maximum elements' composites; for equal keys the lower bucket holds the lower column, so it
//      is that order too. L = the k-th largest B_j (exact: a coarse histogram of the top bits finds the highest bin
//      whose count from the top reaches k, then the B_j at or above that bin's lower edge are ranked among
//      themselves). The k buckets with B_j >= L have k distinct maximum elements whose composites are >= those of
//      every other bucket's elements, so every top-k element lies in them and has key >= key(L);
//   2. the elements of those k buckets with key >= key(L) are collected and ranked by composite; ranks < k are
//      written in order;
//   3. the argmax (when requested) is the largest block composite.
//   Output per row (int32 words): k value bits of s, k global ids, argmax key, argmax global id.
//
// merge (one block per row): the world * k records of a row (all ranks, all-gathered) are ranked the same way with
//   global ids, giving the row's global top-k (value descending, equal values by ascending id) and the global argmax.
//   Shards are contiguous id ranges, so each rank's local top-k contains every global winner of its shard.
//
// Selection and order are functions of the values and ids only (no dependence on scheduling or atomics' order).

#pragma once

#include <sgl_kernel/tensor.h>
#include <sgl_kernel/utils.h>

#include <sgl_kernel/utils.cuh>

#include <tvm/ffi/container/tensor.h>

#include <cstdint>

namespace muse_vr::verify_topk {

using namespace sglang;

constexpr int kMaxK = 64;
constexpr int kBucket = 64;
constexpr int kMaxCols = 65536;
constexpr int kMaxBuckets = kMaxCols / kBucket;
constexpr int kThreads = 256;
constexpr int kBinBits = 12;
constexpr int kBins = 1 << kBinBits;
constexpr int kMaxCand = kMaxK * kBucket;
constexpr int kMergeThreads = 512;
constexpr int kMaxWorld = 8;
constexpr int kMaxBlocks = 64;

SGL_DEVICE uint32_t order_key(uint32_t bits) {
  const float f = __uint_as_float(bits);
  if (f != f) return 0xFFFFFFFFu;
  return (bits & 0x80000000u) ? ~bits : (bits | 0x80000000u);
}

SGL_DEVICE uint64_t composite(uint32_t key, uint32_t id) {
  return (static_cast<uint64_t>(key) << 32) | static_cast<uint64_t>(0xFFFFFFFFu - id);
}

SGL_DEVICE uint32_t comp_id(uint64_t c) { return 0xFFFFFFFFu - static_cast<uint32_t>(c & 0xFFFFFFFFu); }

// shared memory layout of select_kernel (dynamic)
struct SelectSmem {
  int hist[kBins];
  uint64_t list[kMaxBuckets];
  uint64_t cand[kMaxCand];
  uint32_t cbits[kMaxCand];
  int blist[kMaxK];
  int wsum[kThreads / 32];
  int nlist, ncand, nb_sel, bin_lo;
  uint64_t thresh;
};

struct SelectParams {
  const float* s;         // [rows, ld] scaled values of the shard
  int64_t ld;
  int cols;
  int col_offset;         // global id of column 0
  const uint32_t* bkey;   // [rows, nb] bucket maximum keys (int32 tensor holding the unsigned keys)
  int nb;
  const uint64_t* amax;   // [rows, amax_ld]
  int64_t amax_ld;
  int nblk;
  int has_amax;
  int k;
  int32_t* out;           // [rows, out_ld]
  int64_t out_ld;
};

__global__ __launch_bounds__(kThreads) void select_kernel(const __grid_constant__ SelectParams p) {
  extern __shared__ __align__(16) unsigned char smem_raw[];
  SelectSmem& sm = *reinterpret_cast<SelectSmem*>(smem_raw);
  const int row = blockIdx.x, tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
  const uint32_t* bk = p.bkey + static_cast<int64_t>(row) * p.nb;

  for (int i = tid; i < kBins; i += kThreads) sm.hist[i] = 0;
  if (tid == 0) { sm.nlist = 0; sm.ncand = 0; sm.nb_sel = 0; }
  constexpr int kPerT = kMaxBuckets / kThreads;
  uint64_t mine[kPerT];
#pragma unroll
  for (int i = 0; i < kPerT; ++i) {
    const int b = tid + i * kThreads;
    mine[i] = b < p.nb ? ((static_cast<uint64_t>(bk[b]) << 32) | static_cast<uint64_t>(0xFFFFFFFFu - b * kBucket)) : 0;
  }
  __syncthreads();

  // 1. k-th largest bucket maximum
#pragma unroll
  for (int i = 0; i < kPerT; ++i)
    if (tid + i * kThreads < p.nb) atomicAdd(&sm.hist[static_cast<int>(mine[i] >> (64 - kBinBits))], 1);
  __syncthreads();
  {
    constexpr int kPer = kBins / kThreads;
    const int base = (kThreads - 1 - tid) * kPer;  // thread 0 owns the highest bins
    int loc[kPer];
    int tot = 0;
#pragma unroll
    for (int i = kPer - 1; i >= 0; --i) { loc[i] = sm.hist[base + i]; tot += loc[i]; }
    int incl = tot;
#pragma unroll
    for (int off = 1; off < 32; off <<= 1) {
      const int o = __shfl_up_sync(0xffffffffu, incl, off);
      if (lane >= off) incl += o;
    }
    if (lane == 31) sm.wsum[warp] = incl;
    __syncthreads();
    int before = 0;
    for (int w = 0; w < warp; ++w) before += sm.wsum[w];
    int above = before + incl - tot;
    if (above < p.k && above + tot >= p.k) {
#pragma unroll
      for (int i = kPer - 1; i >= 0; --i) {
        above += loc[i];
        if (above >= p.k) { sm.bin_lo = base + i; break; }
      }
    }
  }
  __syncthreads();
  const uint64_t edge = static_cast<uint64_t>(sm.bin_lo) << (64 - kBinBits);
#pragma unroll
  for (int i = 0; i < kPerT; ++i)
    if (tid + i * kThreads < p.nb && mine[i] >= edge) sm.list[atomicAdd(&sm.nlist, 1)] = mine[i];
  __syncthreads();
  {
    const int m = sm.nlist;
    for (int i = tid; i < m; i += kThreads) {
      const uint64_t me = sm.list[i];
      int r = 0;
      for (int j = 0; j < m; ++j) r += sm.list[j] > me ? 1 : 0;
      if (r == p.k - 1) sm.thresh = me;
    }
  }
  __syncthreads();
  const uint64_t lo_b = sm.thresh;
#pragma unroll
  for (int i = 0; i < kPerT; ++i) {
    const int b = tid + i * kThreads;
    if (b < p.nb && mine[i] >= lo_b) sm.blist[atomicAdd(&sm.nb_sel, 1)] = b;
  }
  __syncthreads();
  const uint64_t lo = (lo_b >> 32) << 32;  // key(L), any column

  // 2. elements with key >= key(L) of the selected buckets
  const float* sr = p.s + static_cast<int64_t>(row) * p.ld;
  const int nsel = sm.nb_sel;  // == k
  for (int i = tid; i < nsel * kBucket; i += kThreads) {
    const int c = sm.blist[i / kBucket] * kBucket + (i % kBucket);
    if (c < p.cols) {
      const uint32_t sb = __float_as_uint(sr[c]);
      const uint64_t ck = composite(order_key(sb), static_cast<uint32_t>(c));
      if (ck >= lo) {
        const int pos = atomicAdd(&sm.ncand, 1);
        sm.cand[pos] = ck;
        sm.cbits[pos] = sb;
      }
    }
  }
  __syncthreads();
  const int n = sm.ncand;
  int32_t* o = p.out + static_cast<int64_t>(row) * p.out_ld;
  for (int i = tid; i < n; i += kThreads) {
    const uint64_t me = sm.cand[i];
    int r = 0;
    for (int j = 0; j < n; ++j) r += sm.cand[j] > me ? 1 : 0;
    if (r < p.k) {
      o[r] = static_cast<int32_t>(sm.cbits[i]);
      o[p.k + r] = static_cast<int32_t>(comp_id(me)) + p.col_offset;
    }
  }
  // 3. argmax
  if (warp == 0 && !p.has_amax) {
    if (lane == 0) { o[2 * p.k] = 0; o[2 * p.k + 1] = 0; }
  } else if (warp == 0) {
    uint64_t best = 0;
    const uint64_t* ar = p.amax + static_cast<int64_t>(row) * p.amax_ld;
    for (int b = lane; b < p.nblk; b += 32) best = ar[b] > best ? ar[b] : best;
#pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
      const uint64_t ob = __shfl_xor_sync(0xffffffffu, best, off);
      best = ob > best ? ob : best;
    }
    if (lane == 0) {
      o[2 * p.k] = static_cast<int32_t>(static_cast<uint32_t>(best >> 32));
      o[2 * p.k + 1] = static_cast<int32_t>(comp_id(best)) + p.col_offset;
    }
  }
}

struct MergeParams {
  const int32_t* in;   // [world, rows, in_ld] records of select_kernel
  int64_t in_ld;
  int rows;
  int world;
  int k;
  float* vals;         // [rows, k]  s values, descending (equal values by ascending id)
  int64_t* ids;        // [rows, k]
  int64_t* amax;       // [rows]
};

__global__ __launch_bounds__(kMergeThreads) void merge_kernel(const __grid_constant__ MergeParams p) {
  __shared__ uint64_t cand[kMaxWorld * kMaxK];
  __shared__ uint32_t cbits[kMaxWorld * kMaxK];
  const int row = blockIdx.x, tid = threadIdx.x;
  const int n = p.world * p.k;
  for (int i = tid; i < n; i += kMergeThreads) {
    const int w = i / p.k, j = i % p.k;
    const int32_t* r = p.in + (static_cast<int64_t>(w) * p.rows + row) * p.in_ld;
    const uint32_t b = static_cast<uint32_t>(r[j]);
    cbits[i] = b;
    cand[i] = composite(order_key(b), static_cast<uint32_t>(r[p.k + j]));
  }
  __syncthreads();
  for (int i = tid; i < n; i += kMergeThreads) {
    const uint64_t me = cand[i];
    int rk = 0;
    for (int j = 0; j < n; ++j) rk += cand[j] > me ? 1 : 0;
    if (rk < p.k) {
      p.vals[static_cast<int64_t>(row) * p.k + rk] = __uint_as_float(cbits[i]);
      p.ids[static_cast<int64_t>(row) * p.k + rk] = static_cast<int64_t>(comp_id(me));
    }
  }
  if (tid == 0) {
    uint64_t best = 0;
    for (int w = 0; w < p.world; ++w) {
      const int32_t* r = p.in + (static_cast<int64_t>(w) * p.rows + row) * p.in_ld;
      const uint64_t c = composite(static_cast<uint32_t>(r[2 * p.k]), static_cast<uint32_t>(r[2 * p.k + 1]));
      best = c > best ? c : best;
    }
    p.amax[row] = static_cast<int64_t>(comp_id(best));
  }
}

struct VerifyTopK {
  static void select(const tvm::ffi::TensorView s, const tvm::ffi::TensorView bkey, const tvm::ffi::TensorView amax,
                     const tvm::ffi::TensorView out, int64_t col_offset, int64_t k, int64_t has_amax) {
    using namespace host;
    auto R = SymbolicSize{"rows"};
    auto C = SymbolicSize{"cols"};
    auto LD = SymbolicSize{"ld"};
    auto NB = SymbolicSize{"buckets"};
    auto NK = SymbolicSize{"blocks"};
    auto LA = SymbolicSize{"amax_ld"};
    auto O = SymbolicSize{"out_ld"};
    auto device_ = SymbolicDevice{};
    device_.set_options<kDLCUDA>();
    TensorMatcher({R, C}).with_strides({LD, 1}).with_dtype<float>().with_device(device_).verify(s);
    TensorMatcher({R, NB}).with_strides({NB, 1}).with_dtype<int32_t>().with_device(device_).verify(bkey);
    TensorMatcher({R, NK}).with_strides({LA, 1}).with_dtype<int64_t>().with_device(device_).verify(amax);
    TensorMatcher({R, O}).with_strides({O, 1}).with_dtype<int32_t>().with_device(device_).verify(out);
    const int64_t rows = R.unwrap(), cols = C.unwrap(), nb = NB.unwrap();
    if (rows == 0) return;
    RuntimeCheck(k >= 1 && k <= kMaxK, "verify_topk: k out of range");
    RuntimeCheck(cols >= kMaxK * kBucket && cols <= kMaxCols, "verify_topk: shard width out of range");
    RuntimeCheck(nb == (cols + kBucket - 1) / kBucket, "verify_topk: bucket count");
    RuntimeCheck(NK.unwrap() >= 1 && NK.unwrap() <= kMaxBlocks, "verify_topk: block count");
    RuntimeCheck(O.unwrap() >= 2 * k + 2, "verify_topk: output row too short");
    SelectParams p{};
    p.s = static_cast<const float*>(s.data_ptr());
    p.ld = LD.unwrap();
    p.cols = static_cast<int>(cols);
    p.col_offset = static_cast<int>(col_offset);
    p.bkey = static_cast<const uint32_t*>(bkey.data_ptr());
    p.nb = static_cast<int>(nb);
    p.amax = static_cast<const uint64_t*>(amax.data_ptr());
    p.nblk = static_cast<int>(NK.unwrap());
    p.amax_ld = LA.unwrap();
    p.has_amax = has_amax ? 1 : 0;
    p.k = static_cast<int>(k);
    p.out = static_cast<int32_t*>(out.data_ptr());
    p.out_ld = O.unwrap();
    constexpr std::size_t smem = sizeof(SelectSmem);
    static bool attr_set = false;
    if (!attr_set) {
      RuntimeDeviceCheck(
          ::cudaFuncSetAttribute(select_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, static_cast<int>(smem)));
      attr_set = true;
    }
    LaunchKernel(static_cast<unsigned>(rows), kThreads, device_.unwrap(), smem)(select_kernel, p);
  }

  static void merge(const tvm::ffi::TensorView in, const tvm::ffi::TensorView vals, const tvm::ffi::TensorView ids,
                    const tvm::ffi::TensorView amax, int64_t k) {
    using namespace host;
    auto W = SymbolicSize{"world"};
    auto R = SymbolicSize{"rows"};
    auto L = SymbolicSize{"in_ld"};
    auto K = SymbolicSize{"k"};
    auto device_ = SymbolicDevice{};
    device_.set_options<kDLCUDA>();
    TensorMatcher({W, R, L}).with_dtype<int32_t>().with_device(device_).verify(in);
    TensorMatcher({R, K}).with_strides({K, 1}).with_dtype<float>().with_device(device_).verify(vals);
    TensorMatcher({R, K}).with_strides({K, 1}).with_dtype<int64_t>().with_device(device_).verify(ids);
    TensorMatcher({R}).with_dtype<int64_t>().with_device(device_).verify(amax);
    const int64_t rows = R.unwrap();
    if (rows == 0) return;
    RuntimeCheck(K.unwrap() == k && k >= 1 && k <= kMaxK, "verify_topk: k out of range");
    RuntimeCheck(W.unwrap() >= 1 && W.unwrap() <= kMaxWorld, "verify_topk: world out of range");
    RuntimeCheck(L.unwrap() >= 2 * k + 2, "verify_topk: record too short");
    MergeParams p{};
    p.in = static_cast<const int32_t*>(in.data_ptr());
    p.in_ld = L.unwrap();
    p.rows = static_cast<int>(rows);
    p.world = static_cast<int>(W.unwrap());
    p.k = static_cast<int>(k);
    p.vals = static_cast<float*>(vals.data_ptr());
    p.ids = static_cast<int64_t*>(ids.data_ptr());
    p.amax = static_cast<int64_t*>(amax.data_ptr());
    LaunchKernel(static_cast<unsigned>(rows), kMergeThreads, device_.unwrap())(merge_kernel, p);
  }
};

}  // namespace muse_vr::verify_topk
