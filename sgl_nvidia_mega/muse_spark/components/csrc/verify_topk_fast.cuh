// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
//
// Wider variants of csrc/verify_topk.cuh's select and merge (MUSE_VR_VERIFY_TOPK_FAST, default off). Same selection,
// same order, same output words: only the work split changes.
//
// select_wide (one block per row, kWideThreads threads): verify_topk.cuh's select with four times the threads and the
//   hoisted candidate loads. Every phase is the same function of the row's values: bucket keys, the coarse histogram
//   and its scan, the k-th largest bucket key, the candidate set (appended in any order) and the rank of each candidate
//   by comparison against all others (ranks are unique because the composites are).
//
// merge_sorted (one block per row): every rank's record holds its k entries in composite order (select writes rank r
//   at position r). A candidate's rank in the union is therefore its position in its own record plus, for every other
//   record, the number of entries with a larger composite, found by binary search in that record (its entries are
//   distinct and sorted). The union's composites are distinct (global ids differ), so the ranks are the same as the
//   all-pairs count of verify_topk.cuh's merge, and so are the written values, ids and argmax.
//
// merge_sorted with probabilities (kSoftmax): the same merge, then one warp of the block applies PyTorch's
//   softmax_warp_forward to the row's k values in place of the separate torch.softmax launch: lanes = min(next power of
//   two of k, 32), elements lane + i * lanes, the maximum by its ternary, exp(x - max) summed per lane in element order,
//   both reductions by the xor butterfly from lanes / 2 down to 1, and x / sum (expf and IEEE division: this file is
//   built without fast-math flags, as PyTorch's kernels are).
//
// Record exchange by tagged push (kPush / kPoll; MUSE_VR_VERIFY_RECORDS_PUSH, default off). In place of select ->
//   all-gather (entry barrier, stores, exit barrier) -> merge: select_wide stages its record in shared memory and stores
//   every word, as one 64-bit word (word | epoch << 32), through the multicast address of a symmetric buffer into slot
//   [epoch parity][this rank][row] of every rank (multimem.st.relaxed.sys.b64: each copy is a single-copy-atomic 64-bit
//   store). merge_sorted reads every rank's slot of its row from this rank's copy with ld.relaxed.sys.b64 until each
//   word carries the epoch (data as flag; a bounded wait that traps after max_polls rounds), then merges as above.
//   Epochs: ctr[row] (this rank, device memory) is read by select and merge (epoch = ctr[row] + 1) and advanced by the
//   merge block after its reads; every rank processes the same rows each step, so all ranks use the same epoch.
//   Slot reuse (write-after-read), for any two ranks A and B and the parity-p slot of a row:
//     - B's merge(n) reads A's parity-p slot, and that merge kernel completes in B's stream;
//     - B's select(n + 1) then performs a release (fence.acq_rel.sys) and the tagged store of epoch n + 1;
//     - A's merge(n + 1) polls EVERY rank and acquires B's epoch n + 1 record (fence.acq_rel.sys after the polls); only
//       after that, A's select(n + 2) overwrites its parity-p slot at B.
//   The kernel-completion step is CUDA stream ordering (documented API semantics, as the stock pull's relaxed entry
//   barrier uses); the cross-GPU steps are PTX release / acquire at system scope. Conditions this relies on:
//     (1) the multicast stores are strong at system scope: multimem.st.relaxed.sys after the release fence, never weak;
//     (2) the polls are strong loads (ld.relaxed.sys, then fence.acq_rel.sys), bounded (trap after max_polls rounds);
//     (3) every tagged word is a naturally aligned 64-bit element (word | epoch << 32), so each word validates itself;
//         no payload word travels without its epoch tag.
//   PushParams::light (MUSE_VR_VERIFY_RECORDS_PUSH_LIGHT=1) takes the release by multimem.st.release.sys per word and
//   the acquire by an ld.acquire.sys re-read of each matched word, instead of the two fences; same chain and conditions.

#pragma once

#include "verify_topk.cuh"

namespace muse_vr::verify_topk_fast {

using namespace muse_vr::verify_topk;

constexpr int kWideThreads = 1024;
constexpr int kRecWords = 2 * kMaxK + 2;

struct PushParams {
  uint64_t* mc;            // multicast address of the tagged buffer [2][world][rows_max][width] (kPush)
  const uint64_t* local;   // this rank's copy of it (kPoll)
  uint32_t* ctr;           // [rows_max] per-row epoch counters of this rank
  int rank, world, rows_max, width;
  uint32_t max_polls;
  int light;  // 1: release stores / acquire re-reads instead of the two fences (same conditions (1) - (3))
};

SGL_DEVICE void st_mc_relaxed_b64(uint64_t* addr, uint64_t v) {
  asm volatile("multimem.st.relaxed.sys.global.b64 [%0], %1;" : : "l"(addr), "l"(v) : "memory");
}

SGL_DEVICE uint64_t ld_relaxed_sys_b64(const uint64_t* addr) {
  uint64_t v;
  asm volatile("ld.relaxed.sys.global.b64 %0, [%1];" : "=l"(v) : "l"(addr) : "memory");
  return v;
}

SGL_DEVICE void fence_acq_rel_sys() { asm volatile("fence.acq_rel.sys;" : : : "memory"); }

SGL_DEVICE void st_mc_release_b64(uint64_t* addr, uint64_t v) {
  asm volatile("multimem.st.release.sys.global.b64 [%0], %1;" : : "l"(addr), "l"(v) : "memory");
}

SGL_DEVICE uint64_t ld_acquire_sys_b64(const uint64_t* addr) {
  uint64_t v;
  asm volatile("ld.acquire.sys.global.b64 %0, [%1];" : "=l"(v) : "l"(addr) : "memory");
  return v;
}

SGL_DEVICE int64_t slot_index(const PushParams& q, uint32_t epoch, int src, int row) {
  return ((static_cast<int64_t>(epoch & 1u) * q.world + src) * q.rows_max + row) * q.width;
}

__device__ __noinline__ void records_wait_expired(int row, int src, uint32_t epoch) {
  printf("verify_records_push: no epoch %u record from rank %d for row %d after the poll bound\n", epoch, src, row);
  __trap();
}

struct WideSmem {
  int hist[kBins];
  uint64_t list[kMaxBuckets];
  uint64_t cand[kMaxCand];
  uint32_t cbits[kMaxCand];
  int blist[kMaxK];
  int wsum[kWideThreads / 32];
  int nlist, ncand, nb_sel, bin_lo;
  uint64_t thresh;
};

template <bool kPush>
__global__ __launch_bounds__(kWideThreads) void select_wide_kernel(const __grid_constant__ SelectParams p,
                                                                   const __grid_constant__ PushParams q) {
  extern __shared__ __align__(16) unsigned char smem_raw[];
  WideSmem& sm = *reinterpret_cast<WideSmem*>(smem_raw);
  __shared__ int32_t stage[kRecWords];  // kPush: this row's record before the tagged stores
  const int row = blockIdx.x, tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
  const uint32_t* bk = p.bkey + static_cast<int64_t>(row) * p.nb;

  for (int i = tid; i < kBins; i += kWideThreads) sm.hist[i] = 0;
  if (tid == 0) { sm.nlist = 0; sm.ncand = 0; sm.nb_sel = 0; }
  constexpr int kPerT = kMaxBuckets / kWideThreads;
  uint64_t mine[kPerT];
#pragma unroll
  for (int i = 0; i < kPerT; ++i) {
    const int b = tid + i * kWideThreads;
    mine[i] = b < p.nb ? ((static_cast<uint64_t>(bk[b]) << 32) | static_cast<uint64_t>(0xFFFFFFFFu - b * kBucket)) : 0;
  }
  __syncthreads();

  // 1. k-th largest bucket maximum (coarse histogram of the top bits, then exact ranking at and above the bin)
#pragma unroll
  for (int i = 0; i < kPerT; ++i)
    if (tid + i * kWideThreads < p.nb) atomicAdd(&sm.hist[static_cast<int>(mine[i] >> (64 - kBinBits))], 1);
  __syncthreads();
  {
    constexpr int kPer = kBins / kWideThreads;
    const int base = (kWideThreads - 1 - tid) * kPer;  // thread 0 owns the highest bins
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
    if (tid + i * kWideThreads < p.nb && mine[i] >= edge) sm.list[atomicAdd(&sm.nlist, 1)] = mine[i];
  __syncthreads();
  {
    const int m = sm.nlist;
    for (int i = tid; i < m; i += kWideThreads) {
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
    const int b = tid + i * kWideThreads;
    if (b < p.nb && mine[i] >= lo_b) sm.blist[atomicAdd(&sm.nb_sel, 1)] = b;
  }
  __syncthreads();
  const uint64_t lo = (lo_b >> 32) << 32;  // key(L), any column

  // 2. elements with key >= key(L) of the selected buckets (all loads of a thread issued before any comparison)
  const float* sr = p.s + static_cast<int64_t>(row) * p.ld;
  const int nsel = sm.nb_sel;  // == k
  {
    constexpr int kPer = kMaxCand / kWideThreads;
    int cc[kPer];
    uint32_t sb[kPer];
#pragma unroll
    for (int j = 0; j < kPer; ++j) {
      const int i = tid + j * kWideThreads;
      const int c = i < nsel * kBucket ? sm.blist[i / kBucket] * kBucket + (i % kBucket) : p.cols;
      cc[j] = c < p.cols ? c : -1;
    }
#pragma unroll
    for (int j = 0; j < kPer; ++j) sb[j] = cc[j] >= 0 ? __float_as_uint(__ldg(sr + cc[j])) : 0u;
#pragma unroll
    for (int j = 0; j < kPer; ++j) {
      if (cc[j] < 0) continue;
      const uint64_t ck = composite(order_key(sb[j]), static_cast<uint32_t>(cc[j]));
      if (ck >= lo) {
        const int pos = atomicAdd(&sm.ncand, 1);
        sm.cand[pos] = ck;
        sm.cbits[pos] = sb[j];
      }
    }
  }
  __syncthreads();
  const int n = sm.ncand;
  int32_t* o = kPush ? stage : p.out + static_cast<int64_t>(row) * p.out_ld;
  for (int i = tid; i < n; i += kWideThreads) {
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
  if constexpr (kPush) {
    __syncthreads();
    const int width = 2 * p.k + 2;
    if (tid < width) {
      const uint32_t epoch = q.ctr[row] + 1u;
      const uint64_t word = (static_cast<uint64_t>(epoch) << 32) | static_cast<uint32_t>(stage[tid]);
      if (q.light) {
        st_mc_release_b64(q.mc + slot_index(q, epoch, q.rank, row) + tid, word);  // release store
      } else {
        fence_acq_rel_sys();  // release: this rank's earlier reads of the slot (stream order) before the new record
        st_mc_relaxed_b64(q.mc + slot_index(q, epoch, q.rank, row) + tid, word);
      }
    }
  }
}

struct SoftmaxParams {
  float* probs;  // [rows, k]
};

template <bool kSoftmax, bool kPoll>
__global__ __launch_bounds__(kMergeThreads) void merge_sorted_kernel(const __grid_constant__ MergeParams p,
                                                                     const __grid_constant__ SoftmaxParams sp,
                                                                     const __grid_constant__ PushParams q) {
  __shared__ uint64_t cand[kMaxWorld * kMaxK];
  __shared__ uint32_t cbits[kMaxWorld * kMaxK];
  __shared__ float topv[kMaxK];
  __shared__ int32_t recs[kPoll ? kMaxWorld : 1][kRecWords];  // kPoll: every rank's record of this row
  const int row = blockIdx.x, tid = threadIdx.x;
  const int n = p.world * p.k;
  uint32_t epoch = 0;
  if constexpr (kPoll) {
    epoch = q.ctr[row] + 1u;
    const int width = 2 * p.k + 2;
    for (int i = tid; i < p.world * width; i += kMergeThreads) {
      const int w = i / width, j = i % width;
      const uint64_t* a = q.local + slot_index(q, epoch, w, row) + j;
      uint64_t v = ld_relaxed_sys_b64(a);
      uint32_t polls = 0;
      while (static_cast<uint32_t>(v >> 32) != epoch) {
        if (++polls >= q.max_polls) [[unlikely]] records_wait_expired(row, w, epoch);
        v = ld_relaxed_sys_b64(a);
      }
      if (q.light) v = ld_acquire_sys_b64(a);  // acquire re-read of the matched word (same store: same value)
      recs[w][j] = static_cast<int32_t>(static_cast<uint32_t>(v));
    }
    if (!q.light) fence_acq_rel_sys();  // acquire: every rank's epoch record before this rank's later writes of the slot
    __syncthreads();
  }
  auto rec_of = [&](int w) -> const int32_t* {
    if constexpr (kPoll) return recs[w];
    else return p.in + (static_cast<int64_t>(w) * p.rows + row) * p.in_ld;
  };
  for (int i = tid; i < n; i += kMergeThreads) {
    const int w = i / p.k, j = i % p.k;
    const int32_t* r = rec_of(w);
    const uint32_t b = static_cast<uint32_t>(r[j]);
    cbits[i] = b;
    cand[i] = composite(order_key(b), static_cast<uint32_t>(r[p.k + j]));
  }
  __syncthreads();
  for (int i = tid; i < n; i += kMergeThreads) {
    const uint64_t me = cand[i];
    const int w0 = i / p.k;
    int rk = i % p.k;  // entries of its own record above it (the record is in descending composite order)
    for (int w = 0; w < p.world; ++w) {
      if (w == w0) continue;
      const uint64_t* l = cand + w * p.k;
      int lo = 0, hi = p.k;  // first position whose composite is below me = the count of larger ones
      while (lo < hi) {
        const int mid = (lo + hi) >> 1;
        if (l[mid] > me) lo = mid + 1; else hi = mid;
      }
      rk += lo;
    }
    if (rk < p.k) {
      p.vals[static_cast<int64_t>(row) * p.k + rk] = __uint_as_float(cbits[i]);
      p.ids[static_cast<int64_t>(row) * p.k + rk] = static_cast<int64_t>(comp_id(me));
      if constexpr (kSoftmax) topv[rk] = __uint_as_float(cbits[i]);
    }
  }
  if constexpr (kSoftmax) {
    __syncthreads();
    // PyTorch softmax_warp_forward on one row of k elements (log2_elements = ceil(log2(k)), no mask)
    int next_pow2 = 1;
    while (next_pow2 < p.k) next_pow2 <<= 1;
    const int lanes = next_pow2 < 32 ? next_pow2 : 32;
    const int iters = next_pow2 / lanes;  // 1 or 2 (k <= 64)
    const int lane = tid & 31;
    if (tid < 32) {
      float e[2];
#pragma unroll
      for (int it = 0; it < 2; ++it) {
        const int idx = lane + it * lanes;
        e[it] = (it < iters && lane < lanes && idx < p.k) ? topv[idx] : -INFINITY;
      }
      float mx = e[0];
      if (iters > 1) mx = (mx > e[1]) ? mx : e[1];
      for (int off = lanes / 2; off > 0; off /= 2) {
        const float b = __shfl_xor_sync(0xffffffffu, mx, off, lanes);
        mx = mx < b ? b : mx;
      }
      float sum = 0.0f;
#pragma unroll
      for (int it = 0; it < 2; ++it) {
        if (it < iters) {
          e[it] = expf(e[it] - mx);
          sum += e[it];
        }
      }
      for (int off = lanes / 2; off > 0; off /= 2) sum = sum + __shfl_xor_sync(0xffffffffu, sum, off, lanes);
#pragma unroll
      for (int it = 0; it < 2; ++it) {
        const int idx = lane + it * lanes;
        if (it < iters && lane < lanes && idx < p.k) sp.probs[static_cast<int64_t>(row) * p.k + idx] = e[it] / sum;
      }
    }
  }
  if (tid == 0) {
    uint64_t best = 0;
    for (int w = 0; w < p.world; ++w) {
      const int32_t* r = rec_of(w);
      const uint64_t c = composite(static_cast<uint32_t>(r[2 * p.k]), static_cast<uint32_t>(r[2 * p.k + 1]));
      best = c > best ? c : best;
    }
    p.amax[row] = static_cast<int64_t>(comp_id(best));
  }
  if constexpr (kPoll) {
    __syncthreads();  // every read of this epoch's slots is done
    if (tid == 0) q.ctr[row] = epoch;
  }
}

struct VerifyTopKFast {
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
    RuntimeCheck(k >= 1 && k <= kMaxK, "verify_topk_fast: k out of range");
    RuntimeCheck(cols >= kMaxK * kBucket && cols <= kMaxCols, "verify_topk_fast: shard width out of range");
    RuntimeCheck(nb == (cols + kBucket - 1) / kBucket, "verify_topk_fast: bucket count");
    RuntimeCheck(NK.unwrap() >= 1 && NK.unwrap() <= kMaxBlocks, "verify_topk_fast: block count");
    RuntimeCheck(O.unwrap() >= 2 * k + 2, "verify_topk_fast: output row too short");
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
    p.hoist = 1;
    constexpr std::size_t smem = sizeof(WideSmem);
    static bool attr_set = false;
    if (!attr_set) {
      RuntimeDeviceCheck(::cudaFuncSetAttribute(select_wide_kernel<false>,
                                                cudaFuncAttributeMaxDynamicSharedMemorySize, static_cast<int>(smem)));
      attr_set = true;
    }
    LaunchKernel(static_cast<unsigned>(rows), kWideThreads, device_.unwrap(), smem)(select_wide_kernel<false>, p,
                                                                                    PushParams{});
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
    RuntimeCheck(K.unwrap() == k && k >= 1 && k <= kMaxK, "verify_topk_fast: k out of range");
    RuntimeCheck(W.unwrap() >= 1 && W.unwrap() <= kMaxWorld, "verify_topk_fast: world out of range");
    RuntimeCheck(L.unwrap() >= 2 * k + 2, "verify_topk_fast: record too short");
    MergeParams p{};
    p.in = static_cast<const int32_t*>(in.data_ptr());
    p.in_ld = L.unwrap();
    p.rows = static_cast<int>(rows);
    p.world = static_cast<int>(W.unwrap());
    p.k = static_cast<int>(k);
    p.vals = static_cast<float*>(vals.data_ptr());
    p.ids = static_cast<int64_t*>(ids.data_ptr());
    p.amax = static_cast<int64_t*>(amax.data_ptr());
    LaunchKernel(static_cast<unsigned>(rows), kMergeThreads, device_.unwrap())(merge_sorted_kernel<false, false>, p,
                                                                            SoftmaxParams{nullptr}, PushParams{});
  }

  // merge + the row softmax of the merged values (PyTorch's softmax_warp_forward arithmetic) into probs [rows, k]
  static void merge_softmax(const tvm::ffi::TensorView in, const tvm::ffi::TensorView vals,
                            const tvm::ffi::TensorView ids, const tvm::ffi::TensorView amax,
                            const tvm::ffi::TensorView probs, int64_t k) {
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
    TensorMatcher({R, K}).with_strides({K, 1}).with_dtype<float>().with_device(device_).verify(probs);
    const int64_t rows = R.unwrap();
    if (rows == 0) return;
    RuntimeCheck(K.unwrap() == k && k >= 1 && k <= kMaxK, "verify_topk_fast: k out of range");
    RuntimeCheck(W.unwrap() >= 1 && W.unwrap() <= kMaxWorld, "verify_topk_fast: world out of range");
    RuntimeCheck(L.unwrap() >= 2 * k + 2, "verify_topk_fast: record too short");
    MergeParams p{};
    p.in = static_cast<const int32_t*>(in.data_ptr());
    p.in_ld = L.unwrap();
    p.rows = static_cast<int>(rows);
    p.world = static_cast<int>(W.unwrap());
    p.k = static_cast<int>(k);
    p.vals = static_cast<float*>(vals.data_ptr());
    p.ids = static_cast<int64_t*>(ids.data_ptr());
    p.amax = static_cast<int64_t*>(amax.data_ptr());
    LaunchKernel(static_cast<unsigned>(rows), kMergeThreads, device_.unwrap())(
        merge_sorted_kernel<true, false>, p, SoftmaxParams{static_cast<float*>(probs.data_ptr())}, PushParams{});
  }

  // select with its record pushed (tagged, multicast) into every rank's slot of the exchange buffer
  static void select_push(const tvm::ffi::TensorView s, const tvm::ffi::TensorView bkey, const tvm::ffi::TensorView amax,
                          int64_t mc_ptr, const tvm::ffi::TensorView ctr, int64_t rank, int64_t world, int64_t rows_max,
                          int64_t col_offset, int64_t k, int64_t has_amax, int64_t light) {
    using namespace host;
    auto R = SymbolicSize{"rows"};
    auto C = SymbolicSize{"cols"};
    auto LD = SymbolicSize{"ld"};
    auto NB = SymbolicSize{"buckets"};
    auto NK = SymbolicSize{"blocks"};
    auto LA = SymbolicSize{"amax_ld"};
    auto RM = SymbolicSize{"rows_max"};
    auto device_ = SymbolicDevice{};
    device_.set_options<kDLCUDA>();
    TensorMatcher({R, C}).with_strides({LD, 1}).with_dtype<float>().with_device(device_).verify(s);
    TensorMatcher({R, NB}).with_strides({NB, 1}).with_dtype<int32_t>().with_device(device_).verify(bkey);
    TensorMatcher({R, NK}).with_strides({LA, 1}).with_dtype<int64_t>().with_device(device_).verify(amax);
    TensorMatcher({RM}).with_dtype<int32_t>().with_device(device_).verify(ctr);
    const int64_t rows = R.unwrap(), cols = C.unwrap(), nb = NB.unwrap();
    if (rows == 0) return;
    RuntimeCheck(k >= 1 && k <= kMaxK, "verify_records_push: k out of range");
    RuntimeCheck(cols >= kMaxK * kBucket && cols <= kMaxCols, "verify_records_push: shard width out of range");
    RuntimeCheck(nb == (cols + kBucket - 1) / kBucket, "verify_records_push: bucket count");
    RuntimeCheck(NK.unwrap() >= 1 && NK.unwrap() <= kMaxBlocks, "verify_records_push: block count");
    RuntimeCheck(rows <= rows_max && RM.unwrap() == rows_max, "verify_records_push: rows above the buffer");
    RuntimeCheck(world >= 1 && world <= kMaxWorld && rank >= 0 && rank < world && mc_ptr != 0,
                 "verify_records_push: plane");
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
    p.out = nullptr;
    p.out_ld = 0;
    p.hoist = 1;
    PushParams q{};
    q.mc = reinterpret_cast<uint64_t*>(static_cast<uintptr_t>(mc_ptr));
    q.ctr = static_cast<uint32_t*>(ctr.data_ptr());
    q.rank = static_cast<int>(rank);
    q.world = static_cast<int>(world);
    q.rows_max = static_cast<int>(rows_max);
    q.width = static_cast<int>(2 * k + 2);
    q.light = light ? 1 : 0;
    constexpr std::size_t smem = sizeof(WideSmem);
    static bool attr_set = false;
    if (!attr_set) {
      RuntimeDeviceCheck(::cudaFuncSetAttribute(select_wide_kernel<true>, cudaFuncAttributeMaxDynamicSharedMemorySize,
                                                static_cast<int>(smem)));
      attr_set = true;
    }
    LaunchKernel(static_cast<unsigned>(rows), kWideThreads, device_.unwrap(), smem)(select_wide_kernel<true>, p, q);
  }

  // merge (+ row softmax) of every rank's pushed record of each row, polled from this rank's copy of the buffer
  static void merge_poll(const tvm::ffi::TensorView local, const tvm::ffi::TensorView ctr, const tvm::ffi::TensorView vals,
                         const tvm::ffi::TensorView ids, const tvm::ffi::TensorView amax,
                         const tvm::ffi::TensorView probs, int64_t world, int64_t rows_max, int64_t k,
                         int64_t max_polls, int64_t light) {
    using namespace host;
    auto R = SymbolicSize{"rows"};
    auto K = SymbolicSize{"k"};
    auto RM = SymbolicSize{"rows_max"};
    auto device_ = SymbolicDevice{};
    device_.set_options<kDLCUDA>();
    TensorMatcher({R, K}).with_strides({K, 1}).with_dtype<float>().with_device(device_).verify(vals);
    TensorMatcher({R, K}).with_strides({K, 1}).with_dtype<int64_t>().with_device(device_).verify(ids);
    TensorMatcher({R}).with_dtype<int64_t>().with_device(device_).verify(amax);
    TensorMatcher({R, K}).with_strides({K, 1}).with_dtype<float>().with_device(device_).verify(probs);
    TensorMatcher({RM}).with_dtype<int32_t>().with_device(device_).verify(ctr);
    const int64_t rows = R.unwrap();
    if (rows == 0) return;
    RuntimeCheck(K.unwrap() == k && k >= 1 && k <= kMaxK, "verify_records_push: k out of range");
    RuntimeCheck(world >= 1 && world <= kMaxWorld, "verify_records_push: world out of range");
    RuntimeCheck(rows <= rows_max && RM.unwrap() == rows_max, "verify_records_push: rows above the buffer");
    auto LN = SymbolicSize{"slots"};
    auto LW = SymbolicSize{"width"};
    TensorMatcher({LN, LW}).with_strides({LW, 1}).with_dtype<int64_t>().with_device(device_).verify(local);
    RuntimeCheck(LN.unwrap() == 2 * world * rows_max && LW.unwrap() == 2 * k + 2, "verify_records_push: buffer");
    MergeParams p{};
    p.in = nullptr;
    p.in_ld = 0;
    p.rows = static_cast<int>(rows);
    p.world = static_cast<int>(world);
    p.k = static_cast<int>(k);
    p.vals = static_cast<float*>(vals.data_ptr());
    p.ids = static_cast<int64_t*>(ids.data_ptr());
    p.amax = static_cast<int64_t*>(amax.data_ptr());
    PushParams q{};
    q.local = static_cast<const uint64_t*>(local.data_ptr());
    q.ctr = static_cast<uint32_t*>(ctr.data_ptr());
    q.world = static_cast<int>(world);
    q.rows_max = static_cast<int>(rows_max);
    q.width = static_cast<int>(2 * k + 2);
    q.max_polls = static_cast<uint32_t>(max_polls);
    q.light = light ? 1 : 0;
    LaunchKernel(static_cast<unsigned>(rows), kMergeThreads, device_.unwrap())(
        merge_sorted_kernel<true, true>, p, SoftmaxParams{static_cast<float*>(probs.data_ptr())}, q);
  }
};

}  // namespace muse_vr::verify_topk_fast
