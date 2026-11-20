// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
//
// The sharded top-1 of a multi-row MTP draft step (draft_top1_xchg): every rank holds its
// vocabulary shard's bf16 logits [m, V_local]; the global argmax of each row is chosen without
// gathering the logits. Every CTA scans one chunk of one row into an order key
// (value key << 32 | ~global id: the larger value, then the lower id, wins), folds it into the
// row's key with atomicMax, and the last CTA to arrive exchanges the m row keys with the other
// ranks through the multicast mapping (mkd_xchg's transport: epoch-tagged 64-bit words, three
// slots rotating by epoch, bounded waits that trap), takes each row's largest key and writes the
// token into its column of the draft tokens, 1.0 into the top-k probability, and advances the
// positions (and the three M-RoPE rows) by one.
//
// The choice equals the argmax of the all-gathered logits: the value order is the float order
// with -0.0 as +0.0; ties go to the lowest vocabulary id, which is rank-major; a NaN maps to
// nan_key (0xFFFFFFFF: the first NaN wins, as torch.argmax; the key of -1e30: as the split
// argmax of SGLang's draft_topk1_postprocess).
//
// Built with NVRTC (MKD_WORLD = the TP size, 4 or 8).

#include "mkd_common.cuh"

#ifndef MKD_WORLD
#define MKD_WORLD 8
#endif

namespace sglang {

namespace draft_top1 {

inline constexpr int kSlots = 3;
inline constexpr int kMaxRows = 128;  // the draft graphs' widest batch
inline constexpr int kThreads = 256;
inline constexpr int kVec = 8;  // bf16 per 16-byte load
inline constexpr uint32_t kSpinLimit = 1u << 27;

SGL_DEVICE uint32_t value_key(float v, uint32_t nan_key) {
  if (v != v) return nan_key;
  if (v == 0.0f) v = 0.0f;  // -0.0 orders as +0.0
  const uint32_t bits = __float_as_uint(v);
  return (bits & 0x80000000u) ? ~bits : (bits | 0x80000000u);
}

SGL_DEVICE uint64_t pack(uint32_t vkey, uint32_t id) {
  return (static_cast<uint64_t>(vkey) << 32) | static_cast<uint64_t>(0xFFFFFFFFu - id);
}

SGL_DEVICE uint64_t max64(uint64_t a, uint64_t b) {
  return a > b ? a : b;
}

SGL_DEVICE void mc_store(uint64_t* addr, uint32_t payload, uint32_t tag) {
  const uint64_t w = static_cast<uint64_t>(payload) | (static_cast<uint64_t>(tag) << 32);
  asm volatile("multimem.st.relaxed.sys.global.b64 [%0], %1;" ::"l"(addr), "l"(w) : "memory");
}

SGL_DEVICE uint32_t poll(const uint64_t* src, uint32_t tag) {
  uint64_t w;
  uint32_t spins = 0;
  do {
    asm volatile("ld.relaxed.sys.global.u64 %0, [%1];" : "=l"(w) : "l"(src) : "memory");
    if (++spins > kSpinLimit) __trap();
  } while (static_cast<uint32_t>(w >> 32) != tag);
  return static_cast<uint32_t>(w);
}

}  // namespace draft_top1

struct DraftTop1Params {
  const bf16_t* logits;        // [m, row_stride] this rank's shard logits, written by an earlier grid
  unsigned long long* keys;    // [kMaxRows] per-row fold, zero between calls
  uint32_t* epoch;             // [2]: call counter shared by all ranks (from 1), CTA arrivals
  uint64_t* mc;                // multicast base of [kSlots, world, kMaxRows, 2] words
  const uint64_t* uc;          // this rank's own copy of it
  int64_t* tokens;             // tokens[row * tokens_stride + column] = the global argmax
  float* topk_p;               // [m] 1.0, or null
  int64_t* positions;          // [m] += 1, or null
  int64_t* mrope;              // [3, mrope_stride] += 1 in columns [0, m), or null
  int64_t tokens_stride;
  int64_t mrope_stride;
  int64_t row_stride;          // elements between logits rows
  int32_t m;
  int32_t vocab_local;         // V_local: columns of a row this rank scans
  int32_t vocab_start;         // global id of this rank's column 0
  int32_t vocab_size;          // ids >= vocab_size are padding and never chosen
  int32_t column;
  int32_t rank;
  uint32_t nan_key;
};

// grid (chunks, m): chunk c of row r scans columns [c * kThreads * kVec, ...) of the row.
template <int kWorld>
SGL_DEVICE void draft_top1_xchg_kernel(const DraftTop1Params& p) {
  using namespace device;
  using namespace draft_top1;
  using vec_t = AlignedVector<bf16_t, kVec>;
  constexpr int kWarps = kThreads / kWarpThreads;
  __shared__ uint64_t s_red[kWarps];
  __shared__ bool s_last;

  const int row = static_cast<int>(blockIdx.y);
  const int chunk = static_cast<int>(blockIdx.x);
  const int span = kThreads * kVec;
  const int col0 = chunk * span + static_cast<int>(threadIdx.x) * kVec;
  const int ids_end = min(p.vocab_local, p.vocab_size - p.vocab_start);
  const bf16_t* src = p.logits + static_cast<int64_t>(row) * p.row_stride;

  PDLWaitPrimary<true>();
  uint64_t best = 0;
  if (col0 + kVec <= ids_end && (p.row_stride % kVec) == 0) {
    vec_t v;
    v.load(src + col0);
#pragma unroll
    for (int j = 0; j < kVec; ++j) {
      best = max64(best, pack(value_key(cast<fp32_t>(v[j]), p.nan_key),
                              static_cast<uint32_t>(p.vocab_start + col0 + j)));
    }
  } else {
    for (int j = 0; j < kVec && col0 + j < ids_end; ++j) {
      best = max64(best, pack(value_key(cast<fp32_t>(src[col0 + j]), p.nan_key),
                              static_cast<uint32_t>(p.vocab_start + col0 + j)));
    }
  }
#pragma unroll
  for (uint32_t mask = kWarpThreads / 2; mask >= 1; mask >>= 1) {
    best = max64(best, __shfl_xor_sync(0xffffffffu, best, mask, kWarpThreads));
  }
  const int warp = static_cast<int>(threadIdx.x / kWarpThreads);
  if (get_lane_id() == 0) s_red[warp] = best;
  __syncthreads();
  if (threadIdx.x == 0) {
    uint64_t b = s_red[0];
#pragma unroll
    for (int w = 1; w < kWarps; ++w) b = max64(b, s_red[w]);
    atomicMax(p.keys + row, static_cast<unsigned long long>(b));
    __threadfence();
    const uint32_t total = gridDim.x * gridDim.y;
    s_last = atomicAdd(p.epoch + 1, 1u) == total - 1;
  }
  __syncthreads();
  if (!s_last) return;

  // the last CTA: this rank's row keys -> every rank, then the largest key of each row
  __shared__ uint32_t s_tag;
  if (threadIdx.x == 0) {
    __threadfence();
    s_tag = *reinterpret_cast<volatile uint32_t*>(p.epoch);
  }
  __syncthreads();
  const uint32_t tag = s_tag;
  const int64_t slot = static_cast<int64_t>(tag % kSlots);
  const int t = static_cast<int>(threadIdx.x);
  if (t < p.m) {
    const uint64_t k = atomicExch(p.keys + t, 0ull);  // read the fold and leave it zero
    uint64_t* dst = p.mc + ((slot * kWorld + p.rank) * kMaxRows + t) * 2;
    mc_store(dst, static_cast<uint32_t>(k), tag);
    mc_store(dst + 1, static_cast<uint32_t>(k >> 32), tag);
  }
  if (t < p.m) {
    // every rank's two words of this row in flight together, then each re-polled only while its tag
    // is stale: the keys that already landed cost one round trip, not one each
    uint64_t v[2 * kWorld];
#pragma unroll
    for (int k = 0; k < 2 * kWorld; ++k) {
      const uint64_t* w = p.uc + ((slot * kWorld + k / 2) * kMaxRows + t) * 2 + k % 2;
      asm volatile("ld.relaxed.sys.global.u64 %0, [%1];" : "=l"(v[k]) : "l"(w) : "memory");
    }
    uint64_t win = 0;
#pragma unroll
    for (int r = 0; r < kWorld; ++r) {
      const uint64_t* w = p.uc + ((slot * kWorld + r) * kMaxRows + t) * 2;
      const uint32_t lo = static_cast<uint32_t>(v[2 * r] >> 32) == tag ? static_cast<uint32_t>(v[2 * r]) : poll(w, tag);
      const uint32_t hi =
          static_cast<uint32_t>(v[2 * r + 1] >> 32) == tag ? static_cast<uint32_t>(v[2 * r + 1]) : poll(w + 1, tag);
      win = max64(win, (static_cast<uint64_t>(hi) << 32) | lo);
    }
    const int64_t id = static_cast<int64_t>(0xFFFFFFFFu - static_cast<uint32_t>(win));
    p.tokens[t * p.tokens_stride + p.column] = id;
    if (p.topk_p != nullptr) p.topk_p[t] = 1.0f;
    if (p.positions != nullptr) p.positions[t] += 1;
    if (p.mrope != nullptr) {
#pragma unroll
      for (int a = 0; a < 3; ++a) p.mrope[a * p.mrope_stride + t] += 1;
    }
  }
  __syncthreads();
  if (threadIdx.x == 0) {
    p.epoch[1] = 0u;
    p.epoch[0] = tag + 1;
  }
}

}  // namespace sglang

using sglang::DraftTop1Params;

extern "C" __global__ void __launch_bounds__(256) draft_top1_xchg(const DraftTop1Params __grid_constant__ p) {
  sglang::draft_top1_xchg_kernel<MKD_WORLD>(p);
}
