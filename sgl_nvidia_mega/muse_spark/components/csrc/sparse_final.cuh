// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// Portions Copyright 2023-2024 SGLang Team (Apache-2.0)
// Derived from SGLang (https://github.com/sgl-project/sglang), Apache License 2.0; see ../include/LICENSE.sglang.
// Source: SGLang python/sglang/kernels/ops/speculative/reject_sampling.py (speculative_sampling_classic_kernel,
// non-block-verification branch) at revision a55ab7d, through this package's Triton kernels in verify_topk_sample.py.
// Modified by NVIDIA: the walk, the residual and the final-token draw of one request in one CUDA kernel; the residual's
// tile sums are evaluated on its nonzero entries with the stock reduction tree and only candidate tiles are scanned.
//
// Fused final-token pass of sparse chain verification (verify_topk_sample.sparse_chain_sample). One CTA per request does
// what the Triton kernels _walk_kernel, _tile_sum_kernel and _tile_cdf_kernel do together, with the same writes to
// predicts / accept_index / accept_token_num / bonus and bit-identical values:
//
//   1. Walk: the accepted prefix of the draft chain from the rows' top-k (probabilities, ids) and the draft rows'
//      (candidate ids, q). Every masked sum of the stock walk has at most one nonzero term (ids are unique within a
//      row), so each one is that term exactly; the comparisons are the stock ones (coin * q < p in fp32).
//   2. Residual of the final row at its top-k ids (p, or max(p - q, 0) when a draft was rejected); zero elsewhere.
//   3. Tile sums without materializing the residual row: the stock pass reduces 4096-element tiles (4 warps, 4-element
//      vectors per thread) with a fixed tree, and adds +0 for every element outside the top-k ids (x + 0 = x for x >= +0),
//      so a tile sum equals that tree evaluated on its nonzero leaves only. Leaf key bits, from the first addition to
//      the last: (i >> 1) | k << 1 | (i & 1) << 4 | lane bits 4..0 (butterfly order) << 5 | warp bits 1..0 << 10, for
//      element 512 k + 4 (32 w + lane) + i of the tile. The sorted nonzero leaves are merged level by level.
//   4. Normalizer and tile prefix sums in tile order from 0.0 (fp32), target = coin * normalizer.
//   5. The first element whose running total (prefix in front of its tile + prefix inside the tile) exceeds the target:
//      tiles in order; an all-zero tile is crossed at its first element iff the sum in front of it exceeds the target;
//      a tile with nonzeros is scanned exactly when the sum in front of it plus a bound on any prefix inside it can
//      exceed the target: a shared-memory copy of the tile and the stock scan's association (sequential within a
//      vector; an inclusive lane scan of the vector totals by doubling offsets; per (vector index, warp) chunk totals
//      carried left to right; the last element of a vector = carry + lane scan, the others = (carry + previous lane's
//      scan) + own vector prefix). Positions past the vocabulary in the last tile take part as zeros, as in the stock
//      pass. No crossing gives vocab - 1.
//   6. predicts[last accepted index] = token, bonus = token.
// The stock pass's dense residual row, tile sums, arrival counter and result scratch are not used.
//
// Bounds: K, KQ <= 64 (top-k widths), num_slots <= 16, tiles <= 64 per row.

#pragma once

#include <sgl_kernel/tensor.h>
#include <sgl_kernel/utils.h>

#include <sgl_kernel/utils.cuh>

#include <tvm/ffi/container/tensor.h>

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cstdint>

namespace muse_vr::sparse_final {

using namespace sglang;

constexpr int kThreads = 128;
constexpr int kBV = 4096;
constexpr int kMaxK = 64;
constexpr int kMaxSlots = 16;
constexpr int kMaxTiles = 64;

struct Params {
  int32_t* predicts;
  int32_t* accept_index;
  int32_t* accept_num;
  const int64_t* candidates;
  int64_t s_cand_b, s_cand_s;
  const int64_t* retrive_index;
  int64_t s_idx_b, s_idx_s;
  const float* uniform;
  int64_t s_uni_b, s_uni_s;
  const float* top_probs;  // [bs * slots, K]
  const int64_t* top_ids;  // [bs * slots, K]
  const int64_t* cand_ids;
  int64_t s_ci_b, s_ci_s;
  const void* q_rows;
  int64_t s_q_b, s_q_s;
  const float* uniform_final;
  int64_t* bonus;
  int K, KQ, slots, vocab, tiles;
};

template <typename QT>
SGL_DEVICE float q_load(const void* base, int64_t off) {
  if constexpr (std::is_same_v<QT, float>) {
    return static_cast<const float*>(base)[off];
  } else if constexpr (std::is_same_v<QT, __nv_bfloat16>) {
    return __bfloat162float(static_cast<const __nv_bfloat16*>(base)[off]);
  } else {
    return __half2float(static_cast<const __half*>(base)[off]);
  }
}

// leaf key of element e (0..4095) of a tile in the stock reduction tree (see the header)
SGL_DEVICE uint32_t leaf_key(uint32_t e) {
  const uint32_t i = e & 3u, t = (e >> 2) & 127u, k = e >> 9;
  const uint32_t lane = t & 31u, w = t >> 5;
  const uint32_t lane_rev = __brev(lane) >> 27;  // lane bit 4 first
  const uint32_t w_rev = ((w & 1u) << 1) | (w >> 1);
  return (i >> 1) | (k << 1) | ((i & 1u) << 4) | (lane_rev << 5) | (w_rev << 10);
}

struct Smem {
  int64_t cand[kMaxSlots];
  int64_t retr[kMaxSlots];
  float uni[kMaxSlots];
  int64_t tid[kMaxSlots][kMaxK];
  float tpr[kMaxSlots][kMaxK];
  int64_t cid[kMaxSlots][kMaxK];
  float qv[kMaxSlots][kMaxK];
  // residual
  int nz;               // nonzero residual entries
  int r_id[kMaxK];      // vocabulary id
  float r_val[kMaxK];
  // sorted leaves -> tile roots
  uint32_t tk[kMaxK];  // leaf keys by top-k position (0xffffffff = zero residual)
  uint32_t key[kMaxK];
  uint32_t rkey[kMaxK];
  float rval[kMaxK];
  int wcount[kThreads / 32];
  int wcount2[kThreads / 32];
  float val[kMaxK];
  int nodes;
  // walk results
  int cur, num_accept;
  int64_t last;
  // scan
  alignas(16) float tile[kBV];
  float tot[8][4];
  float carry[8][4];
  int found;
  // crossing
  float cum[kMaxK];
  float target;
  int act, act_at, act_tok;
};

template <typename QT>
__global__ void __launch_bounds__(kThreads, 1) sparse_final_kernel(const __grid_constant__ Params p) {
  extern __shared__ __align__(16) uint8_t smem_raw[];
  Smem& s = *reinterpret_cast<Smem*>(smem_raw);
  const int pid = blockIdx.x, th = threadIdx.x, lane = th & 31, warp = th >> 5;
  const int K = p.K, KQ = p.KQ, NS = p.slots;

  // ---- stage the request's inputs (one round trip) ----
  for (int j = th; j < NS; j += kThreads) {
    s.cand[j] = p.candidates[pid * p.s_cand_b + j * p.s_cand_s];
    s.retr[j] = p.retrive_index[pid * p.s_idx_b + j * p.s_idx_s];
    if (j + 1 < NS) s.uni[j] = p.uniform[pid * p.s_uni_b + j * p.s_uni_s];
  }
  for (int j = th; j < NS * K; j += kThreads) {
    const int r = j / K, c = j % K;
    const int64_t row = static_cast<int64_t>(pid) * NS + r;
    s.tid[r][c] = p.top_ids[row * K + c];
    s.tpr[r][c] = p.top_probs[row * K + c];
  }
  for (int j = th; j < (NS - 1) * KQ; j += kThreads) {
    const int r = j / KQ, c = j % KQ;
    s.cid[r][c] = p.cand_ids[pid * p.s_ci_b + r * p.s_ci_s + c];
    s.qv[r][c] = q_load<QT>(p.q_rows, pid * p.s_q_b + r * p.s_q_s + c);
  }
  __syncthreads();

  // ---- walk (warp 0) ----
  if (warp == 0) {
    int64_t last = s.retr[0];
    if (lane == 0) p.accept_index[pid * p.s_idx_b] = static_cast<int32_t>(last);
    int num_accept = 0, cur = 0, step = 1;
    while (step < NS) {
      const int64_t tok = s.cand[step];
      float pv = 0.0f, qv = 0.0f;
      for (int c = lane; c < K; c += 32)
        if (s.tid[cur][c] == tok) pv = s.tpr[cur][c];
      for (int c = lane; c < KQ; c += 32)
        if (s.cid[cur][c] == tok) qv = s.qv[cur][c];
      // at most one lane holds a nonzero term: the warp sum is that term (+0 + +0 = +0 otherwise)
#pragma unroll
      for (int off = 16; off > 0; off >>= 1) {
        pv += __shfl_xor_sync(0xffffffffu, pv, off);
        qv += __shfl_xor_sync(0xffffffffu, qv, off);
      }
      const float coin = s.uni[step - 1];
      const bool q_is_prob = (qv > 0.0f) && (qv <= 1.0f);
      if (q_is_prob && __fmul_rn(coin, qv) < pv) {
        ++num_accept;
        cur = step;
        if (lane == 0) p.predicts[last] = static_cast<int32_t>(tok);
        last = s.retr[step];
        if (lane == 0) p.accept_index[pid * p.s_idx_b + num_accept * p.s_idx_s] = static_cast<int32_t>(last);
        ++step;
      } else {
        break;
      }
    }
    if (lane == 0) {
      p.accept_num[pid] = num_accept;
      s.cur = cur;
      s.num_accept = num_accept;
      s.last = last;
    }
  }
  __syncthreads();

  // ---- residual of the final row at its top-k ids (one thread per top-k entry; slot = top-k position) ----
  {
    const int cur = s.cur;
    const bool all_acc = s.num_accept == NS - 1;
    if (th < K) {
      const int64_t id = s.tid[cur][th];
      const float pv = s.tpr[cur][th];
      float v;
      if (all_acc) {
        v = pv;
      } else {
        float qv = 0.0f;  // candidate ids are unique: the masked sum of the stock walk is the one matching q (or +0)
        for (int j = 0; j < KQ; ++j)
          if (s.cid[cur][j] == id) qv = s.qv[cur][j];
        qv = (qv >= 0.0f && qv <= 1.0f) ? qv : 0.0f;
        const float diff = __fsub_rn(__fmul_rn(1.0f, pv), qv);
        v = diff > 0.0f ? diff : 0.0f;
      }
      s.r_id[th] = static_cast<int>(id);
      s.r_val[th] = v;  // zero entries are skipped below (x + 0 = x)
      const uint32_t uid = static_cast<uint32_t>(id);
      s.tk[th] = v > 0.0f ? (((uid / kBV) << 12) | leaf_key(uid % kBV)) : 0xffffffffu;
    }
  }
  __syncthreads();
  const int nz = K;

  // ---- tile sums: rank the nonzero leaves by (tile, leaf key); one thread per tile merges its leaves in the stock tree order ----
  // (repeatedly the adjacent pair with the longest common key prefix: those two are siblings of the reduction tree)
  if (th < nz) {
    const uint32_t kj = s.tk[th];
    if (kj != 0xffffffffu) {
      int rank = 0;
      for (int o = 0; o < nz; ++o) rank += s.tk[o] < kj;
      s.key[rank] = kj;
      s.val[rank] = s.r_val[th];
    }
  }
  {
    const uint32_t m = __ballot_sync(0xffffffffu, th < nz && s.tk[th] != 0xffffffffu);
    if (lane == 0) s.wcount[warp] = __popc(m);
  }
  __syncthreads();
  int n = 0;
#pragma unroll
  for (int w = 0; w < kThreads / 32; ++w) n += s.wcount[w];
  {
    const int j = th;
    bool start = false;
    float root = 0.0f;
    uint32_t tkey = 0;
    if (j < n) {
      const uint32_t tj = s.key[j] >> 12;
      start = (j == 0) || ((s.key[j - 1] >> 12) != tj);
      if (start) {
        int e = j + 1;
        while (e < n && (s.key[e] >> 12) == tj) ++e;
        int cnt = e - j;
        uint32_t kk[kMaxK];
        float vv[kMaxK];
        for (int i = 0; i < cnt; ++i) { kk[i] = s.key[j + i]; vv[i] = s.val[j + i]; }
        while (cnt > 1) {
          int best = 0, bl = -1;
          for (int i = 0; i + 1 < cnt; ++i) {
            const int l = __clz(kk[i] ^ kk[i + 1]);
            if (l > bl) { bl = l; best = i; }
          }
          vv[best] = __fadd_rn(vv[best], vv[best + 1]);
          for (int i = best + 1; i + 1 < cnt; ++i) { kk[i] = kk[i + 1]; vv[i] = vv[i + 1]; }
          --cnt;
        }
        root = vv[0];
        tkey = tj << 12;
      }
    }
    // compact the roots in tile order: slot = tile starts before j
    const uint32_t sm = __ballot_sync(0xffffffffu, start);
    if (lane == 0) s.wcount2[warp] = __popc(sm);
    __syncthreads();
    int base = 0;
    for (int w = 0; w < warp; ++w) base += s.wcount2[w];
    if (start) {
      const int slot = base + __popc(sm & ((1u << lane) - 1u));
      s.rkey[slot] = tkey;
      s.rval[slot] = root;
    }
    if (th == 0) {
      int tot = 0;
#pragma unroll
      for (int w = 0; w < kThreads / 32; ++w) tot += s.wcount2[w];
      s.nodes = tot;
    }
  }
  __syncthreads();
  if (th < s.nodes) {
    s.key[th] = s.rkey[th];
    s.val[th] = s.rval[th];
  }
  __syncthreads();

  // ---- normalizer, tile prefix sums (fp32, tile order, from 0.0), target ----
  const int nodes = s.nodes;
  if (th == 0) {
    float norm = 0.0f;
#pragma unroll 4
    for (int j = 0; j < nodes; ++j) {
      s.cum[j] = norm;
      norm = __fadd_rn(norm, s.val[j]);
    }
    s.target = __fmul_rn(p.uniform_final[pid], norm);
  }
  __syncthreads();
  const float target = s.target;
  // ---- crossing: the first event in (tile, rule) order, every node evaluated by its own thread ----
  //   rule 0: the tile's first element crosses (cum > target; its total is cum + a prefix >= +0)
  //   rule 1: the tile may cross inside (cum + 1.001 st > 0.999 target: a strict over-approximation) -> exact scan
  //   rule 2: the zero tiles after the tile cross at their first element (cum + st > target)
  int token = p.vocab - 1;
  int after = -1;  // events at or below this code were handled
  for (;;) {
    int code = 0x7fffffff;
    if (th < nodes) {
      const int j = th;
      const int t = static_cast<int>(s.key[j] >> 12);
      const float cum = s.cum[j];
      const float st = s.val[j];
      const int next_t = (j + 1 < nodes) ? static_cast<int>(s.key[j + 1] >> 12) : p.tiles;
      const int c0 = 3 * j, c1 = 3 * j + 1, c2 = 3 * j + 2;
      if (c0 > after && cum > target) code = c0;
      else if (c1 > after && __fadd_rn(cum, __fmul_rn(st, 1.001f)) > __fmul_rn(target, 0.999f)) code = c1;
      else if (c2 > after && t + 1 < next_t && __fadd_rn(cum, st) > target) code = c2;
    }
    if (th == 0) s.act = 0x7fffffff;
    __syncthreads();
    if (code != 0x7fffffff) atomicMin(&s.act, code);
    __syncthreads();
    const int ev = s.act;
    __syncthreads();
    if (ev == 0x7fffffff) break;
    const int j = ev / 3, rule = ev % 3;
    const int t = static_cast<int>(s.key[j] >> 12);
    if (rule == 0) { token = t * kBV; break; }
    if (rule == 2) { token = (t + 1) * kBV; break; }
    const float cum = s.cum[j];

    // exact scan of the tile of node `at`
    {
      float4* t4 = reinterpret_cast<float4*>(s.tile);
      for (int e = th; e < kBV / 4; e += kThreads) t4[e] = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
      if (th == 0) s.found = kBV;
    }
    __syncthreads();
    for (int j2 = th; j2 < nz; j2 += kThreads)
      if (s.r_id[j2] / kBV == t) s.tile[s.r_id[j2] % kBV] = s.r_val[j2];
    __syncthreads();
    float pre[8][4];
    float S[8];
#pragma unroll
    for (int k = 0; k < 8; ++k) {
      const float4 x = *reinterpret_cast<const float4*>(&s.tile[512 * k + 4 * th]);
      pre[k][0] = x.x;
      pre[k][1] = __fadd_rn(pre[k][0], x.y);
      pre[k][2] = __fadd_rn(pre[k][1], x.z);
      pre[k][3] = __fadd_rn(pre[k][2], x.w);
      float v = pre[k][3];
#pragma unroll
      for (int off = 1; off < 32; off <<= 1) {
        const float y = __shfl_up_sync(0xffffffffu, v, off);
        if (lane >= off) v = __fadd_rn(y, v);
      }
      S[k] = v;
      if (lane == 31) s.tot[k][warp] = v;
    }
    __syncthreads();
    if (th == 0) {
      float tt[32];
#pragma unroll
      for (int q = 0; q < 32; ++q) tt[q] = s.tot[q >> 2][q & 3];
      float acc = tt[0];
      s.carry[0][0] = 0.0f;  // unused (no carry for chunk 0, warp 0)
#pragma unroll
      for (int q = 1; q < 32; ++q) {
        s.carry[q >> 2][q & 3] = acc;
        acc = __fadd_rn(acc, tt[q]);
      }
    }
    __syncthreads();
    int first = kBV;
#pragma unroll
    for (int k = 0; k < 8; ++k) {
      const bool none = (k == 0 && warp == 0);
      const float c = s.carry[k][warp];
      const float temp = none ? S[k] : __fadd_rn(c, S[k]);
      const float prev = __shfl_up_sync(0xffffffffu, temp, 1);
      const int e0 = 512 * k + 4 * th;
#pragma unroll
      for (int i = 0; i < 4; ++i) {
        float pr;
        if (i == 3) {
          pr = temp;
        } else if (lane > 0) {
          pr = __fadd_rn(prev, pre[k][i]);
        } else {
          pr = none ? pre[k][i] : __fadd_rn(c, pre[k][i]);
        }
        if (__fadd_rn(cum, pr) > target && e0 + i < first) first = e0 + i;
      }
    }
    atomicMin(&s.found, first);
    __syncthreads();
    const int f = s.found;
    __syncthreads();
    if (f < kBV) { token = t * kBV + f; break; }
    after = ev;  // continue after this tile's scan: its zero-tile rule, then the following tiles
  }

  if (th == 0) {
    p.predicts[s.last] = token;
    p.bonus[pid] = static_cast<int64_t>(token);
  }
}

struct SparseFinal {
  static void run(const tvm::ffi::TensorView predicts, const tvm::ffi::TensorView accept_index,
                  const tvm::ffi::TensorView accept_num, const tvm::ffi::TensorView candidates,
                  const tvm::ffi::TensorView retrive_index, const tvm::ffi::TensorView uniform,
                  const tvm::ffi::TensorView top_probs, const tvm::ffi::TensorView top_ids,
                  const tvm::ffi::TensorView cand_ids, const tvm::ffi::TensorView q_rows,
                  const tvm::ffi::TensorView uniform_final, const tvm::ffi::TensorView bonus, int64_t vocab) {
    using namespace host;
    auto device_ = SymbolicDevice{};
    device_.set_options<kDLCUDA>();
    auto BS = SymbolicSize{"bs"};
    auto NSz = SymbolicSize{"slots"};
    auto KS = SymbolicSize{"k"};
    auto KQS = SymbolicSize{"kq"};
    auto R = SymbolicSize{"rows"};
    auto SCB = SymbolicSize{"s_cand_b"}, SCS = SymbolicSize{"s_cand_s"};
    auto SIB = SymbolicSize{"s_idx_b"}, SIS = SymbolicSize{"s_idx_s"};
    auto SUB = SymbolicSize{"s_uni_b"}, SUS = SymbolicSize{"s_uni_s"};
    auto SCIB = SymbolicSize{"s_ci_b"}, SCIS = SymbolicSize{"s_ci_s"};
    auto SQB = SymbolicSize{"s_q_b"}, SQS = SymbolicSize{"s_q_s"};
    auto NS1 = SymbolicSize{"slots_m1"};
    TensorMatcher({BS, NSz}).with_strides({SCB, SCS}).with_dtype<int64_t>().with_device(device_).verify(candidates);
    TensorMatcher({BS, NSz}).with_strides({SIB, SIS}).with_dtype<int64_t>().with_device(device_).verify(retrive_index);
    TensorMatcher({BS, NSz}).with_strides({SIB, SIS}).with_dtype<int32_t>().with_device(device_).verify(accept_index);
    TensorMatcher({BS}).with_dtype<int32_t>().with_device(device_).verify(accept_num);
    TensorMatcher({BS, NS1}).with_strides({SUB, SUS}).with_dtype<float>().with_device(device_).verify(uniform);
    TensorMatcher({R, KS}).with_dtype<float>().with_device(device_).verify(top_probs);
    TensorMatcher({R, KS}).with_dtype<int64_t>().with_device(device_).verify(top_ids);
    TensorMatcher({BS, NS1, KQS}).with_strides({SCIB, SCIS, 1}).with_dtype<int64_t>().with_device(device_).verify(
        cand_ids);
    TensorMatcher({BS, NS1, KQS}).with_strides({SQB, SQS, 1}).with_dtype<float, bf16_t, fp16_t>().with_device(device_)
        .verify(q_rows);
    TensorMatcher({BS}).with_dtype<float>().with_device(device_).verify(uniform_final);
    TensorMatcher({BS}).with_dtype<int64_t>().with_device(device_).verify(bonus);
    const int64_t bs = BS.unwrap(), ns = NSz.unwrap(), k = KS.unwrap(), kq = KQS.unwrap();
    RuntimeCheck(NS1.unwrap() == ns - 1, "uniform / draft rows must have slots - 1 rows");
    RuntimeCheck(R.unwrap() == bs * ns, "top-k rows must be bs * slots");
    RuntimeCheck(ns >= 2 && ns <= kMaxSlots && k >= 1 && k <= kMaxK && kq >= 1 && kq <= kMaxK, "sizes out of range");
    const int64_t tiles = (vocab + kBV - 1) / kBV;
    RuntimeCheck(vocab >= 1 && tiles <= kMaxTiles && vocab <= INT32_MAX, "vocabulary out of range");
    RuntimeCheck(predicts.dtype().code == kDLInt && predicts.dtype().bits == 32, "predicts must be int32");
    if (bs == 0) return;
    Params p{};
    p.predicts = static_cast<int32_t*>(predicts.data_ptr());
    p.accept_index = static_cast<int32_t*>(accept_index.data_ptr());
    p.accept_num = static_cast<int32_t*>(accept_num.data_ptr());
    p.candidates = static_cast<const int64_t*>(candidates.data_ptr());
    p.s_cand_b = SCB.unwrap();
    p.s_cand_s = SCS.unwrap();
    p.retrive_index = static_cast<const int64_t*>(retrive_index.data_ptr());
    p.s_idx_b = SIB.unwrap();
    p.s_idx_s = SIS.unwrap();
    p.uniform = static_cast<const float*>(uniform.data_ptr());
    p.s_uni_b = SUB.unwrap();
    p.s_uni_s = SUS.unwrap();
    p.top_probs = static_cast<const float*>(top_probs.data_ptr());
    p.top_ids = static_cast<const int64_t*>(top_ids.data_ptr());
    p.cand_ids = static_cast<const int64_t*>(cand_ids.data_ptr());
    p.s_ci_b = SCIB.unwrap();
    p.s_ci_s = SCIS.unwrap();
    p.q_rows = q_rows.data_ptr();
    p.s_q_b = SQB.unwrap();
    p.s_q_s = SQS.unwrap();
    p.uniform_final = static_cast<const float*>(uniform_final.data_ptr());
    p.bonus = static_cast<int64_t*>(bonus.data_ptr());
    p.K = static_cast<int>(k);
    p.KQ = static_cast<int>(kq);
    p.slots = static_cast<int>(ns);
    p.vocab = static_cast<int>(vocab);
    p.tiles = static_cast<int>(tiles);
    const auto dev = device_.unwrap();
    const auto code = q_rows.dtype().code;
    const auto bits = q_rows.dtype().bits;
    const size_t smem = sizeof(Smem);
    auto launch = [&](auto kernel) {
      static bool configured = false;
      if (!configured) {
        RuntimeDeviceCheck(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem));
        configured = true;
      }
      LaunchKernel(static_cast<unsigned>(bs), kThreads, dev, smem)(kernel, p);
    };
    if (code == kDLFloat && bits == 32) {
      launch(sparse_final_kernel<float>);
    } else if (code == kDLBfloat && bits == 16) {
      launch(sparse_final_kernel<__nv_bfloat16>);
    } else {
      launch(sparse_final_kernel<__half>);
    }
  }
};

}  // namespace muse_vr::sparse_final
