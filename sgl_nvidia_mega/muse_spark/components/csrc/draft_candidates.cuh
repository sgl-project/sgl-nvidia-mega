// Derived from SGLang (https://github.com/sgl-project/sglang), Apache License 2.0; see ../include/LICENSE.sglang.
// SPDX-License-Identifier: Apache-2.0
// Portions Copyright 2023-2024 SGLang Team (Apache-2.0)
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// Source: SGLang python/sglang/kernels/jit/csrc/muse_spark_v12/all_gather.cuh (push-plane gather protocol) at
// revision a55ab7d.
// Modified by NVIDIA: gathers the per-rank top-k candidate records of the draft head instead of a bf16 column slice,
// and merges them on every rank into the global top-k (same selection and order as SGLang's candidate_topk), then
// applies the draft's unary logit transform.
//
// Inputs per rank (vocabulary shard of the head):
//   vals: [N, kTopK] bf16, the shard's top-k logits (sorted, as the shard top-k returns them)
//   ids:  [N, kTopK] int64, their positions inside the shard
// Outputs (identical on every rank):
//   out_ids:  [N, kTopK] int64, global token ids
//   out_vals: [N, kTopK] fp32, transformed logits
//
// Exchange: every rank stores its records (two 32-bit words per candidate: the bf16 logit in the high half with a
// non-zero low half, and the global id plus one) into slot (dst = peer, src = rank) of every peer's push workspace for
// the current phase. No stored word is zero, so a zero word means "not arrived yet" (the plane's empty marker); the
// consumer restores zeros after reading. Row r is pushed and consumed by the same warp index on every rank.
//
// Merge (per row, one warp; E = kWorld * kTopK gathered entries, entry index = src * kTopK + j):
//   1. key(v) maps a float to an unsigned integer with the same total order (+0 above -0, NaN above everything).
//      rank(e) = #{f : key(f) > key(e)} + #{f < e : key(f) == key(e)}; the entries with rank < kTopK are selected.
//   2. Selected entries are laid out as a gather that writes all entries strictly above the k-th key in index order,
//      followed by the entries equal to the k-th key in index order.
//   3. That layout is ordered by a 32-slot bitonic sorting network run by 16 lanes (compare-exchange with
//      "greater, NaN first" order, invalid slots last), which fixes the order among equal logits; with
//      `stable_ties` set, the layout is ordered stably instead (key descending, entry ascending).
//   The host picks the mode that reproduces the output of torch.topk(gathered, k) on the running PyTorch build, tie
//   order included (draft_candidates.py, TIE_RULES; tests/test_draft_candidates.py checks both modes on tie-heavy
//   inputs, including the row counts where the build changes its tie order).
#include <sgl_kernel/tensor.h>
#include <sgl_kernel/utils.h>

#include <sgl_kernel/type.cuh>
#include <sgl_kernel/utils.cuh>

#include <sgl_kernel/distributed/communicator.cuh>

#include <tvm/ffi/container/tensor.h>

#include <algorithm>
#include <cstdint>

namespace sglang {

namespace muse_vr_draft_cand {

constexpr uint32_t kTopK = 16;
constexpr uint32_t kSortSlots = 32;  // power of two >= kTopK, slots kTopK.. hold invalid entries
constexpr uint32_t kWarpsPerBlock = 4;
constexpr uint32_t kThreads = 32 * kWarpsPerBlock;
constexpr uint32_t kRecordBytes = 8;
constexpr uint32_t kRowBytes = kTopK * kRecordBytes;  // 128 B = 8 x 16 B
constexpr uint32_t kVecsPerRow = kRowBytes / 16;

template <uint32_t kWorld>
struct Params {
  device::distributed::PushWorkSpace<kWorld> ws;
  uint32_t num_counters;
  uint32_t rank;
  uint32_t num_rows;
  int64_t vocab_start;  // this rank's first global id
  const uint16_t* vals;  // [N, kTopK] bf16 bits
  const int64_t* ids;    // [N, kTopK]
  int64_t* out_ids;      // [N, kTopK]
  float* out_vals;       // [N, kTopK]
  // unary transform: x = x * mult (if has_mult); x = tanh(x * inv_cap) * cap (if has_cap)
  float mult;
  float inv_cap;
  float cap;
  bool has_mult;
  bool has_cap;
  // order of the selected entries: stable (key descending, entry ascending) instead of the sorting network's
  bool stable_ties;
  // test path: read the gathered records from `gathered` ([N, kWorld * kTopK] x 2 words) instead of the plane
  const uint32_t* gathered;
};

SGL_DEVICE uint32_t order_key(float v) {
  const uint32_t x = __float_as_uint(v);
  const uint32_t mask = (x & 0x80000000u) ? 0xffffffffu : 0x80000000u;
  return (v == v) ? (x ^ mask) : 0xffffffffu;
}

SGL_DEVICE bool greater_nan_first(float a, float b) {
  return (isnan(a) && !isnan(b)) || (a > b);
}

SGL_DEVICE float bf16_bits_to_float(uint32_t bits16) {
  return __uint_as_float(bits16 << 16);
}

SGL_DEVICE float unary(float x, const float mult, const float inv_cap, const float cap, bool has_mult, bool has_cap) {
  if (has_mult) x = __fmul_rn(x, mult);
  if (has_cap) {
    x = __fmul_rn(x, inv_cap);
    x = tanhf(x);
    x = __fmul_rn(x, cap);
  }
  return x;
}

struct RowSmem {
  float key[kSortSlots];
  int32_t idx[kSortSlots];
  bool valid[kSortSlots];
};

// Stable order (key descending, entry ascending) of the kTopK selected slots. The slots hold the entries above the
// k-th key and then the entries equal to it, each group in entry order, so the stable rank of a slot is the number of
// slots with a larger key plus the number of earlier slots with the same key.
SGL_DEVICE void stable_slots(RowSmem& sm, uint32_t lane) {
  __syncwarp();
  float kv = 0.0f;
  int32_t iv = 0;
  uint32_t r = 0;
  if (lane < kTopK) {
    kv = sm.key[lane];
    iv = sm.idx[lane];
    const uint32_t mk = order_key(kv);
#pragma unroll
    for (uint32_t j = 0; j < kTopK; ++j) {
      const uint32_t kj = order_key(sm.key[j]);
      r += (kj > mk || (kj == mk && j < lane)) ? 1u : 0u;
    }
  }
  __syncwarp();
  if (lane < kTopK) {
    sm.key[r] = kv;
    sm.idx[r] = iv;
    sm.valid[r] = true;
  }
}

// Merges one row. `w0[q], w1[q]` (q < kItems) are this lane's gathered words for entries e = 2 * (lane + 32 * (q / 2))
// + (q % 2), i.e. the two candidates of vector v = lane + 32 * (q / 2). Lane 0..15 write the kTopK outputs.
template <uint32_t kWorld>
SGL_DEVICE void merge_row(
    const Params<kWorld>& params, uint32_t row, const uint32_t (&w0)[2 * ((kWorld * kVecsPerRow + 31) / 32)],
    const uint32_t (&w1)[2 * ((kWorld * kVecsPerRow + 31) / 32)], RowSmem& sm) {
  constexpr uint32_t kVecs = kWorld * kVecsPerRow;  // 16 B vectors per gathered row
  constexpr uint32_t kRounds = (kVecs + 31) / 32;
  constexpr uint32_t kItems = 2 * kRounds;
  const uint32_t lane = threadIdx.x % 32;

  uint32_t key[kItems];
  bool valid[kItems];
  uint32_t entry[kItems];
#pragma unroll
  for (uint32_t q = 0; q < kItems; ++q) {
    const uint32_t v = lane + 32 * (q / 2);
    valid[q] = v < kVecs;
    entry[q] = 2 * v + (q % 2);
    key[q] = valid[q] ? order_key(bf16_bits_to_float(w0[q] >> 16)) : 0u;
  }

  // rank of every entry in (key descending, entry ascending) order
  uint32_t rank[kItems];
#pragma unroll
  for (uint32_t q = 0; q < kItems; ++q) rank[q] = 0;
#pragma unroll
  for (uint32_t qs = 0; qs < kItems; ++qs) {
    for (uint32_t s = 0; s < 32; ++s) {
      const uint32_t ks = __shfl_sync(0xffffffffu, key[qs], s);
      const bool vs = (s + 32 * (qs / 2)) < kVecs;
      const uint32_t es = 2 * (s + 32 * (qs / 2)) + (qs % 2);
#pragma unroll
      for (uint32_t q = 0; q < kItems; ++q) {
        rank[q] += (vs && (ks > key[q] || (ks == key[q] && es < entry[q]))) ? 1u : 0u;
      }
    }
  }

  // key of the entry ranked kTopK - 1 (exactly one entry has that rank)
  uint32_t kth_local = 0;
  bool has_kth = false;
#pragma unroll
  for (uint32_t q = 0; q < kItems; ++q) {
    if (valid[q] && rank[q] == kTopK - 1) {
      kth_local = key[q];
      has_kth = true;
    }
  }
  const uint32_t owner = __ffs(__ballot_sync(0xffffffffu, has_kth)) - 1;
  const uint32_t kth = __shfl_sync(0xffffffffu, kth_local, owner);

  // output position of each selected entry
  uint32_t pos[kItems];
#pragma unroll
  for (uint32_t q = 0; q < kItems; ++q) pos[q] = 0;
#pragma unroll
  for (uint32_t qs = 0; qs < kItems; ++qs) {
    for (uint32_t s = 0; s < 32; ++s) {
      const uint32_t ks = __shfl_sync(0xffffffffu, key[qs], s);
      const bool vs = (s + 32 * (qs / 2)) < kVecs;
      const uint32_t es = 2 * (s + 32 * (qs / 2)) + (qs % 2);
#pragma unroll
      for (uint32_t q = 0; q < kItems; ++q) {
        pos[q] += (vs && ks > kth && es < entry[q]) ? 1u : 0u;
      }
    }
  }

  // slots: invalid placeholders, then the selected entries
  if (lane < kSortSlots) {
    sm.key[lane] = 0.0f;
    sm.idx[lane] = 0;
    sm.valid[lane] = false;
  }
  __syncwarp();
#pragma unroll
  for (uint32_t q = 0; q < kItems; ++q) {
    if (valid[q] && rank[q] < kTopK) {
      const uint32_t p = key[q] > kth ? pos[q] : rank[q];
      sm.key[p] = bf16_bits_to_float(w0[q] >> 16);
      sm.idx[p] = static_cast<int32_t>(entry[q]);
      sm.valid[p] = true;
    }
  }

  if (params.stable_ties) {
    stable_slots(sm, lane);
  } else {
    // bitonic sorting network over kSortSlots slots, one compare-exchange per lane and step
#pragma unroll
    for (uint32_t size = 2; size <= kSortSlots; size *= 2) {
#pragma unroll
      for (uint32_t stride = size / 2; stride > 0; stride /= 2) {
        __syncwarp();
        if (lane < kSortSlots / 2) {
          const uint32_t lo = 2 * lane - (lane & (stride - 1));
          const uint32_t hi = lo + stride;
          const bool dir = size == kSortSlots ? false : ((lane & (size / 2)) != 0);
          const float ka = sm.key[lo], kb = sm.key[hi];
          const bool va = sm.valid[lo], vb = sm.valid[hi];
          const bool a_first = (greater_nan_first(ka, kb) && va) || !vb;
          if (a_first == dir) {
            const int32_t ia = sm.idx[lo], ib = sm.idx[hi];
            sm.key[lo] = kb;
            sm.key[hi] = ka;
            sm.idx[lo] = ib;
            sm.idx[hi] = ia;
            sm.valid[lo] = vb;
            sm.valid[hi] = va;
          }
        }
      }
    }
  }
  __syncwarp();

  // global ids of the sorted entries: gathered word 1 of entry e, held by lane (e / 2) % 32, item 2 * (e / 64) + e % 2
  uint32_t gid_word = 0;
  int32_t my_entry = lane < kTopK ? sm.idx[lane] : 0;
#pragma unroll
  for (uint32_t q = 0; q < kItems; ++q) {
    const uint32_t src_lane = (static_cast<uint32_t>(my_entry) / 2) % 32;
    const uint32_t w = __shfl_sync(0xffffffffu, w1[q], src_lane);
    if (2 * (static_cast<uint32_t>(my_entry) / 64) + (static_cast<uint32_t>(my_entry) % 2) == q) gid_word = w;
  }
  if (lane < kTopK) {
    const size_t o = static_cast<size_t>(row) * kTopK + lane;
    params.out_ids[o] = static_cast<int64_t>(gid_word) - 1;
    params.out_vals[o] =
        unary(sm.key[lane], params.mult, params.inv_cap, params.cap, params.has_mult, params.has_cap);
  }
  __syncwarp();
}

template <uint32_t kWorld>
SGL_DEVICE uint4 own_vector(const Params<kWorld>& params, uint32_t row, uint32_t v) {
  // candidates 2v and 2v + 1 of this rank's row
  const size_t base = static_cast<size_t>(row) * kTopK + 2 * v;
  uint4 out;
  out.x = (static_cast<uint32_t>(params.vals[base]) << 16) | 1u;
  out.y = static_cast<uint32_t>(params.ids[base] + params.vocab_start + 1);
  out.z = (static_cast<uint32_t>(params.vals[base + 1]) << 16) | 1u;
  out.w = static_cast<uint32_t>(params.ids[base + 1] + params.vocab_start + 1);
  return out;
}

inline constexpr uint32_t kMaxPolls = 1u << 28;

__device__ __noinline__ void poll_bound_exceeded() {
  __trap();
}

template <uint32_t kWorld, bool kUsePDL, bool kFromGathered>
__global__ __launch_bounds__(kThreads) void draft_candidates_kernel(const __grid_constant__ Params<kWorld> params) {
  constexpr uint32_t kVecs = kWorld * kVecsPerRow;
  constexpr uint32_t kRounds = (kVecs + 31) / 32;
  constexpr uint32_t kItems = 2 * kRounds;
  __shared__ RowSmem smem[kWarpsPerBlock];

  const uint32_t lane = threadIdx.x % 32;
  const uint32_t warp = threadIdx.x / 32;
  const uint32_t bx = blockIdx.x;

  device::PDLWaitPrimary<kUsePDL>();

  if constexpr (kFromGathered) {
    for (uint32_t row = bx * kWarpsPerBlock + warp; row < params.num_rows; row += gridDim.x * kWarpsPerBlock) {
      uint32_t w0[kItems], w1[kItems];
#pragma unroll
      for (uint32_t q = 0; q < kItems; ++q) {
        const uint32_t v = lane + 32 * (q / 2);
        const uint32_t e = 2 * v + (q % 2);
        const bool ok = v < kVecs;
        const size_t o = (static_cast<size_t>(row) * kWorld * kTopK + e) * 2;
        w0[q] = ok ? params.gathered[o] : 0u;
        w1[q] = ok ? params.gathered[o + 1] : 0u;
      }
      merge_row<kWorld>(params, row, w0, w1, smem[warp]);
    }
    return;
  } else {
    // the last block keeps the phase counters of the blocks this grid does not use in step
    const uint32_t phase = params.ws.counter[bx].get() & 1;
    if (bx == gridDim.x - 1) {
      device::PDLTriggerSecondary<kUsePDL>();
      for (uint32_t i = bx + threadIdx.x; i < params.num_counters; i += kThreads) {
        params.ws.counter[i].set(phase ^ 1);
      }
      return;
    }
    const uint32_t work_blocks = gridDim.x - 1;
    const uint32_t half = phase * kWorld;

    // push this rank's rows of this block to every peer
    for (uint32_t row = bx * kWarpsPerBlock + warp; row < params.num_rows; row += work_blocks * kWarpsPerBlock) {
      if (lane < kVecsPerRow) {
        const uint4 vec = own_vector(params, row, lane);
#pragma unroll
        for (uint32_t i = 0; i < kWorld; ++i) {
          if (i == params.rank) continue;
          uint8_t* slot = params.ws.workspaces[i] + (params.rank + half) * params.ws.slot_bytes;
          device::ptx::st_relaxed_16B(vec, slot, static_cast<int64_t>(row) * kVecsPerRow + lane);
        }
      }
    }

    // consume every peer's rows (same rows as pushed above)
    for (uint32_t row = bx * kWarpsPerBlock + warp; row < params.num_rows; row += work_blocks * kWarpsPerBlock) {
      uint32_t w0[kItems], w1[kItems];
#pragma unroll
      for (uint32_t r = 0; r < kRounds; ++r) {
        const uint32_t v = lane + 32 * r;  // gathered vector: src rank v / kVecsPerRow, vector v % kVecsPerRow
        uint4 vec = make_uint4(0u, 0u, 0u, 0u);
        if (v < kVecs) {
          const uint32_t src = v / kVecsPerRow;
          const uint32_t vi = v % kVecsPerRow;
          if (src == params.rank) {
            vec = own_vector(params, row, vi);
          } else {
            uint8_t* slot = params.ws.workspaces[params.rank] + (src + half) * params.ws.slot_bytes;
            const int64_t off = static_cast<int64_t>(row) * kVecsPerRow + vi;
            uint32_t polls = 0;
            do {
              device::ptx::ld_relaxed_16B(vec, slot, off);
              if (vec.x != 0u && vec.y != 0u && vec.z != 0u && vec.w != 0u) break;
              if (++polls == kMaxPolls) [[unlikely]] poll_bound_exceeded();
            } while (true);
            const uint4 zero = make_uint4(0u, 0u, 0u, 0u);
            device::ptx::st_global_16B(zero, slot, off);
          }
        }
        w0[2 * r] = vec.x;
        w1[2 * r] = vec.y;
        w0[2 * r + 1] = vec.z;
        w1[2 * r + 1] = vec.w;
      }
      merge_row<kWorld>(params, row, w0, w1, smem[warp]);
    }
    device::PDLTriggerSecondary<kUsePDL>();
    __syncthreads();
    if (threadIdx.x == 0) params.ws.counter[bx].set(phase ^ 1);
  }
}

}  // namespace muse_vr_draft_cand

using host::distributed::CommunicatorRef;

template <uint32_t kWorld, bool kUsePDL>
struct MuseVrDraftCandidates {
  using TensorView = tvm::ffi::TensorView;
  static constexpr uint32_t kTopK = muse_vr_draft_cand::kTopK;

  static void fill(
      muse_vr_draft_cand::Params<kWorld>& p,
      TensorView vals,
      TensorView ids,
      TensorView out_ids,
      TensorView out_vals,
      double mult,
      double cap,
      bool has_mult,
      bool has_cap,
      host::SymbolicDevice& device) {
    using namespace host;
    auto N = SymbolicSize{"num_rows"};
    device.set_options<kDLCUDA>();
    TensorMatcher({N, kTopK}).with_dtype<bf16_t>().with_device(device).verify(vals);
    TensorMatcher({N, kTopK}).with_dtype<int64_t>().with_device(device).verify(ids);
    TensorMatcher({N, kTopK}).with_dtype<int64_t>().with_device(device).verify(out_ids);
    TensorMatcher({N, kTopK}).with_dtype<fp32_t>().with_device(device).verify(out_vals);
    p.num_rows = static_cast<uint32_t>(N.unwrap());
    p.vals = static_cast<const uint16_t*>(vals.data_ptr());
    p.ids = static_cast<const int64_t*>(ids.data_ptr());
    p.out_ids = static_cast<int64_t*>(out_ids.data_ptr());
    p.out_vals = static_cast<float*>(out_vals.data_ptr());
    // scalars as torch applies them to an fp32 tensor: mul_(m) multiplies by float(m); div_(c) multiplies by the
    // float reciprocal 1.0f / float(c); mul_(c) multiplies by float(c)
    p.mult = static_cast<float>(mult);
    p.cap = static_cast<float>(cap);
    p.inv_cap = 1.0f / static_cast<float>(cap);
    p.has_mult = has_mult;
    p.has_cap = has_cap;
  }

  static void
  run(CommunicatorRef ref,
      TensorView vals,
      TensorView ids,
      TensorView out_ids,
      TensorView out_vals,
      int64_t vocab_start,
      double mult,
      double cap,
      bool has_mult,
      bool has_cap,
      bool stable_ties) {
    using namespace host;
    const auto& push = ref.get()->get_push_obj();
    auto device = SymbolicDevice{};
    muse_vr_draft_cand::Params<kWorld> p{};
    fill(p, vals, ids, out_ids, out_vals, mult, cap, has_mult, has_cap, device);
    p.stable_ties = stable_ties;
    if (p.num_rows == 0) return;
    RuntimeCheck(push.world_size == kWorld, "the kernel is compiled for TP", kWorld);
    const int64_t nbytes = int64_t(p.num_rows) * muse_vr_draft_cand::kRowBytes;
    RuntimeCheck(nbytes <= int64_t(push.slot_bytes), "rows exceed the push slot size");
    RuntimeCheck(push.num_blocks >= 2, "needs a work block and a counter block");
    p.ws = push.get_workspace<kWorld>(nbytes);
    p.num_counters = push.num_blocks;
    p.rank = push.rank;
    p.vocab_start = vocab_start;
    p.gathered = nullptr;
    const uint32_t want = div_ceil(p.num_rows, muse_vr_draft_cand::kWarpsPerBlock);
    const uint32_t work = std::min<uint32_t>(want, push.num_blocks - 1);
    LaunchKernel(work + 1, muse_vr_draft_cand::kThreads, device.unwrap())
        .enable_pdl(kUsePDL)(muse_vr_draft_cand::draft_candidates_kernel<kWorld, kUsePDL, false>, p);
  }

  // Test entry: merge already-gathered records ([N, kWorld * kTopK, 2] uint32 words as pushed).
  static void merge_only(
      TensorView gathered,
      TensorView out_ids,
      TensorView out_vals,
      TensorView vals_like,
      TensorView ids_like,
      double mult,
      double cap,
      bool has_mult,
      bool has_cap,
      bool stable_ties) {
    using namespace host;
    auto device = SymbolicDevice{};
    muse_vr_draft_cand::Params<kWorld> p{};
    fill(p, vals_like, ids_like, out_ids, out_vals, mult, cap, has_mult, has_cap, device);
    p.stable_ties = stable_ties;
    if (p.num_rows == 0) return;
    auto N = SymbolicSize{"num_rows"};
    N.set_value(p.num_rows);
    TensorMatcher({N, int64_t(kWorld * kTopK), 2}).with_dtype<int32_t>().with_device(device).verify(gathered);
    p.gathered = static_cast<const uint32_t*>(gathered.data_ptr());
    const uint32_t grid = div_ceil(p.num_rows, muse_vr_draft_cand::kWarpsPerBlock);
    LaunchKernel(grid, muse_vr_draft_cand::kThreads, device.unwrap())
        .enable_pdl(false)(muse_vr_draft_cand::draft_candidates_kernel<kWorld, false, true>, p);
  }
};

}  // namespace sglang
