// Derived in part from SGLang (https://github.com/sgl-project/sglang), Apache License 2.0; see ../include/LICENSE.sglang.
// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// Source: SGLang python/sglang/kernels/jit/csrc/muse_spark_v12/all_gather_norm_residual.cuh at revision a55ab7d
// (the cluster boundary kernel: thread map, moment accumulation and output arithmetic).
// Modified by NVIDIA: the gathered rows that feed the boundary are produced by a separate projection kernel that stores
// its output columns straight into every rank's receive buffer; the boundary kernel waits for them with the hand-off
// described below instead of SGLang's lamport-marker gather. The projection kernel is new.
//
// Two kernels, launched back to back on one stream:
//
// 1. moe_tail_push_kernel: z[:, cols of this rank] = y @ w_local^T for y [M, K] bf16 (identical on every rank) and
//    this rank's contiguous row block w_local [N_local, K] of the projection weight. Each CTA owns kRows weight rows,
//    copies them into shared memory before the programmatic-dependency wait (weights do not depend on the previous
//    kernel), then reads y, accumulates in fp32 with bf16 tensor-core MMAs (each warp a K range, warps summed in index
//    order), rounds to bf16 and stores the [M, kRows] tile into the receive buffer of every rank (itself included).
//
// 2. moe_tail_norm_kernel: one cluster per row, the post-norm + gated residual + pre-norm arithmetic of the SGLang
//    cluster boundary kernel, applied to the gathered row once every rank's columns have arrived.
//
// Hand-off (kTagged = false): after a CTA barrier, one thread of each projection CTA adds 1 to a local counter with an
// acq_rel gpu-scope atomic; the CTA that completes the call's count issues fence.acq_rel.sys and adds 1 to arrive[rank]
// on every rank; threads 0..world-1 of each boundary CTA poll arrive[src] with ld.acquire.sys until it reaches the
// call count (release / acquire chains, PTX ISA memory consistency model).
// Hand-off (kTagged = true): every value travels as one 32-bit word (tag << 16 | bf16 bits) written with relaxed
// system-scope accesses; the boundary CTA polls its words until all carry this call's tag. No fences: a word is
// accepted on its own value (single-copy atomicity of aligned 32-bit accesses), but reuse of a receive half two calls
// later is ordered only by stream order and the data dependence between calls, not by a release/acquire pair, so this
// mode relies on ordering the PTX memory model does not guarantee.
//
// Buffer reuse: the receive region has two halves selected by call parity. A rank writes half p for call n+2 only
// after it observed every rank's data for call n+1, which each rank produces only after its call n finished reading
// half p (stream order). Counters are monotonic modulo 2^32 and compared with a signed difference; tags cycle through
// 1..65535 and are never 0 (the initial buffer contents). seq[] holds this rank's call count per projection CTA (only
// that CTA writes its entry; the boundary kernel reads entry 0 after the dependency wait).
// Every wait is bounded (kMaxPolls) and traps instead of hanging.

#include <sgl_kernel/distributed/ptx.cuh>
#include <sgl_kernel/mbarrier.cuh>

#include <tvm/ffi/container/tensor.h>
#include <tvm/ffi/optional.h>

#include "muse_spark_v12/norm_residual.cuh"

#include <cstdint>

namespace sglang {

namespace muse_vr_moe_tail {

inline constexpr int kRows = 16;        // weight rows (output columns) per push CTA: one MMA M tile
inline constexpr int kWarps = 8;        // push CTA warps, each owns K / kWarps
inline constexpr int kThreads = kWarps * 32;
inline constexpr int kRowPad = 32;      // bf16 elements of padding per shared-memory weight row
inline constexpr uint32_t kMaxPolls = 1u << 26;  // bounded waits: trap instead of hanging forever
inline constexpr int kMaxWorld = 8;

struct Pointers {
  uint8_t* recv[kMaxWorld];     // receive region base of every rank (mapped in this process)
  uint32_t* arrive[kMaxWorld];  // arrival counter array of every rank, indexed by source rank
};

struct PushParams {
  const bf16_t* y;  // [M, K]
  const bf16_t* w;  // [N_local, K]
  Pointers ptrs;
  uint32_t* seq;    // [gridDim.x]
  uint32_t* done;   // [1] CTAs of this rank that finished, all calls (fenced hand-off)
  uint32_t rank;
  uint32_t num_rows;
  uint32_t parity_bytes;
  uint32_t col_base;  // rank * N_local
};

struct WaitParams {
  const uint8_t* recv;     // this rank's receive region
  const uint32_t* arrive;  // this rank's arrival counters
  const uint32_t* seq;     // the push kernel's call counters (entry 0 is read)
  uint32_t push_grid;
  uint32_t parity_bytes;
  uint32_t num_rows;
  NormResidualParams norm;
};

// Test-only (race fuzz): sleeps a pseudo-random 0..max_us microseconds chosen by hashing the key.
SGL_DEVICE void fuzz_delay(uint32_t key, uint32_t max_us) {
  uint32_t h = key * 2654435761u;
  h ^= h >> 15;
  h *= 2246822519u;
  h ^= h >> 13;
  for (uint32_t i = h % max_us; i > 0; --i) __nanosleep(1000);
}

SGL_DEVICE void bulk_copy_g2s(void* smem_dst, const void* gmem_src, uint32_t bytes, uint64_t* bar) {
  asm volatile(
      "cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];" ::"r"(
          device::ptx::to_shared(smem_dst)),
      "l"(gmem_src),
      "r"(bytes),
      "r"(device::ptx::to_shared(bar))
      : "memory");
}

SGL_DEVICE void wait_bar_bounded(uint64_t* bar, uint32_t parity) {
  const uint32_t addr = device::ptx::to_shared(bar);
  uint32_t ready = 0;
  for (uint32_t polls = 0;; ++polls) {
    asm volatile(
        "{\n\t.reg .pred p;\n\t"
        "mbarrier.try_wait.parity.shared::cta.b64 p, [%1], %2;\n\t"
        "selp.b32 %0, 1, 0, p;\n\t}"
        : "=r"(ready)
        : "r"(addr), "r"(parity)
        : "memory");
    if (ready) return;
    if (polls > kMaxPolls) __trap();
  }
}

SGL_DEVICE uint4 lds128(const void* p) {
  uint4 v;
  asm volatile("ld.shared.v4.b32 {%0, %1, %2, %3}, [%4];"
               : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w)
               : "r"(device::ptx::to_shared(p)));
  return v;
}

SGL_DEVICE uint4 ldg128(const void* p) {
  uint4 v;
  asm volatile("ld.global.v4.b32 {%0, %1, %2, %3}, [%4];" : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w) : "l"(p));
  return v;
}

SGL_DEVICE void stg128(void* p, uint4 v) {
  asm volatile("st.global.v4.b32 [%0], {%1, %2, %3, %4};" ::"l"(p), "r"(v.x), "r"(v.y), "r"(v.z), "r"(v.w)
               : "memory");
}

// D += A * B, m16n8k16, bf16 inputs, fp32 accumulate (PTX ISA, warp-level matrix multiply-accumulate).
SGL_DEVICE void mma_bf16_16816(float (&d)[4], uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3, uint32_t b0, uint32_t b1) {
  asm volatile(
      "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, "
      "{%0, %1, %2, %3};"
      : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
      : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}

SGL_DEVICE uint32_t pack_bf16x2(float lo, float hi) {
  const __nv_bfloat162 v = __floats2bfloat162_rn(lo, hi);
  return *reinterpret_cast<const uint32_t*>(&v);
}

// The MMA k slots of one 32-wide K chunk are assigned so that lane (g, t) reads 8 contiguous elements, k = 8t..8t+7,
// of its weight rows (g, g+8) and of its token g: step 0 takes 8t..8t+3, step 1 takes 8t+4..8t+7, identically for
// both operands, so every product y[n,k] * w[r,k] is formed exactly once.
template <int64_t kHidden, int kK, int kNT, bool kUsePDL, bool kTagged, bool kFuzz>
__global__ void __launch_bounds__(kThreads, 1) moe_tail_push_kernel(const __grid_constant__ PushParams p) {
  using namespace device;
  constexpr int kRowStride = kK + kRowPad;
  constexpr int kKPerWarp = kK / kWarps;
  constexpr int kChunks = kKPerWarp / 32;
  constexpr int kCPG = kChunks / kNT >= 1 ? kChunks / kNT : 1;
  constexpr int kGroups = kChunks / kCPG;
  constexpr int kN = kNT * 8;
  static_assert(kK % (kWarps * 32) == 0 && kChunks % kCPG == 0);
  static_assert((kRowStride * 2) % 16 == 0);

  extern __shared__ __align__(128) uint8_t smem_raw[];
  bf16_t* w_s = reinterpret_cast<bf16_t*>(smem_raw);
  float* red_s = reinterpret_cast<float*>(smem_raw + kRows * kRowStride * 2);  // [kWarps][kRows][kN]
  __shared__ alignas(8) uint64_t bar;
  __shared__ uint32_t seq_s;

  const int tid = threadIdx.x;
  const int warp = tid / 32;
  const int lane = tid % 32;
  const int g = lane / 4;
  const int t = lane % 4;
  const bf16_t* w_g = p.w + static_cast<int64_t>(blockIdx.x) * kRows * kK;

  // Weight rows are static: stage them before waiting on the previous kernel.
  if (tid == 0) {
    ptx::mbar_init(&bar, 1);
    asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
    ptx::mbar_arrive_expect_tx(&bar, kRows * kK * 2);
#pragma unroll 1
    for (int r = 0; r < kRows; ++r) {
      bulk_copy_g2s(w_s + r * kRowStride, w_g + static_cast<int64_t>(r) * kK, kK * 2, &bar);
    }
  }
  __syncthreads();
  PDLTriggerSecondary<kUsePDL>();
  PDLWaitPrimary<kUsePDL>();
  if (tid == 0) seq_s = p.seq[blockIdx.x];

  const uint32_t m = p.num_rows;
  const bf16_t* y_base = p.y + warp * kKPerWarp + t * 8;
  float acc[kNT][4];
#pragma unroll
  for (int nt = 0; nt < kNT; ++nt) acc[nt][0] = acc[nt][1] = acc[nt][2] = acc[nt][3] = 0.0f;

  uint4 yb[2][kNT][kCPG];
  auto load_group = [&](int grp, uint4 (&dst)[kNT][kCPG]) {
#pragma unroll
    for (int nt = 0; nt < kNT; ++nt) {
      const uint32_t token = nt * 8 + g;
#pragma unroll
      for (int c = 0; c < kCPG; ++c) {
        if (token < m) {
          dst[nt][c] = ldg128(y_base + static_cast<int64_t>(token) * kK + (grp * kCPG + c) * 32);
        } else {
          dst[nt][c] = make_uint4(0, 0, 0, 0);
        }
      }
    }
  };

  if constexpr (kFuzz) {
    __syncthreads();
    if (tid == 0) fuzz_delay(seq_s * 7919u + blockIdx.x * 131u + p.rank * 31u, 30u);
    __syncthreads();
  }
  load_group(0, yb[0]);
  wait_bar_bounded(&bar, 0);
#pragma unroll
  for (int grp = 0; grp < kGroups; ++grp) {
    if (grp + 1 < kGroups) load_group(grp + 1, yb[(grp + 1) & 1]);
#pragma unroll
    for (int c = 0; c < kCPG; ++c) {
      const int kk = warp * kKPerWarp + (grp * kCPG + c) * 32 + t * 8;
      const uint4 va = lds128(w_s + g * kRowStride + kk);
      const uint4 vb = lds128(w_s + (g + 8) * kRowStride + kk);
#pragma unroll
      for (int nt = 0; nt < kNT; ++nt) {
        const uint4 yv = yb[grp & 1][nt][c];
        mma_bf16_16816(acc[nt], va.x, vb.x, va.y, vb.y, yv.x, yv.y);
        mma_bf16_16816(acc[nt], va.z, vb.z, va.w, vb.w, yv.z, yv.w);
      }
    }
  }

  // acc[nt]: {(row g, n 2t), (row g, n 2t+1), (row g+8, n 2t), (row g+8, n 2t+1)} with n relative to nt * 8
#pragma unroll
  for (int nt = 0; nt < kNT; ++nt) {
    const int n = nt * 8 + 2 * t;
    red_s[(warp * kRows + g) * kN + n] = acc[nt][0];
    red_s[(warp * kRows + g) * kN + n + 1] = acc[nt][1];
    red_s[(warp * kRows + g + 8) * kN + n] = acc[nt][2];
    red_s[(warp * kRows + g + 8) * kN + n + 1] = acc[nt][3];
  }
  __syncthreads();

  const uint32_t parity = seq_s & 1u;
  const int64_t half_off = static_cast<int64_t>(parity) * p.parity_bytes;
  for (uint32_t v = tid; v < m * 2; v += kThreads) {
    const uint32_t n = v / 2;
    const int h = v % 2;
    float s[8];
#pragma unroll
    for (int r = 0; r < 8; ++r) {
      const int row = h * 8 + r;
      float sum = red_s[row * kN + n];
#pragma unroll
      for (int w = 1; w < kWarps; ++w) sum += red_s[(w * kRows + row) * kN + n];
      s[r] = sum;
    }
    const uint4 packed = make_uint4(
        pack_bf16x2(s[0], s[1]), pack_bf16x2(s[2], s[3]), pack_bf16x2(s[4], s[5]), pack_bf16x2(s[6], s[7]));
    const int64_t elem = static_cast<int64_t>(n) * kHidden + p.col_base + blockIdx.x * kRows + h * 8;
    if constexpr (kTagged) {
      // one 32-bit word per value: (tag << 16) | bf16 bits, stored with relaxed system-scope 32-bit element accesses
      const uint32_t tag = (seq_s % 65535u + 1u) << 16;
      const uint4 lo = make_uint4(tag | (packed.x & 0xffffu), tag | (packed.x >> 16), tag | (packed.y & 0xffffu), tag | (packed.y >> 16));
      const uint4 hi = make_uint4(tag | (packed.z & 0xffffu), tag | (packed.z >> 16), tag | (packed.w & 0xffffu), tag | (packed.w >> 16));
#pragma unroll
      for (int d = 0; d < kMaxWorld; ++d) {
        if (p.ptrs.recv[d] == nullptr) continue;
        uint8_t* dst = p.ptrs.recv[d] + half_off + elem * 4;
        ptx::st_relaxed_16B(lo, dst, 0);
        ptx::st_relaxed_16B(hi, dst, 1);
      }
    } else {
#pragma unroll
      for (int d = 0; d < kMaxWorld; ++d) {
        if (p.ptrs.recv[d] != nullptr) stg128(p.ptrs.recv[d] + half_off + elem * 2, packed);
      }
    }
  }
  if constexpr (kTagged) {
    if (tid == 0) p.seq[blockIdx.x] = seq_s + 1u;
    return;
  }
  __syncthreads();
  if (tid == 0) {
    // Every store of this CTA is ordered before this thread by the barrier. The acq_rel RMW on the local counter
    // releases them to the CTA that finishes last (gpu scope); that CTA alone issues the system-scope fence and
    // publishes one arrival per call on every rank, which consumers observe with acquire loads.
    const uint32_t prev = ptx::atomic_add_acq_rel_gpu(p.done, 1u);
    if (prev == (seq_s + 1u) * gridDim.x - 1u) {
      asm volatile("fence.acq_rel.sys;" ::: "memory");
#pragma unroll
      for (int d = 0; d < kMaxWorld; ++d) {
        if (p.ptrs.arrive[d] != nullptr) ptx::red_add_relaxed_sys(p.ptrs.arrive[d] + p.rank, 1u);
      }
    }
    p.seq[blockIdx.x] = seq_s + 1u;
  }
}

SGL_DEVICE fp32x2_t mul_pair(fp32x2_t a, fp32x2_t b) {
  return {a.x * b.x, a.y * b.y};
}

SGL_DEVICE fp32x2_t fma_pair(fp32x2_t a, fp32x2_t b, fp32x2_t c) {
  return {fmaf(a.x, b.x, c.x), fmaf(a.y, b.y, c.y)};
}

}  // namespace muse_vr_moe_tail

template <
    int64_t kHidden,
    uint32_t kWorld,
    int kNumThreads,
    int kClusterSize,
    bool kUsePDL,
    bool kPostNormIsInputScale,
    bool kPostWeight,
    bool kGate,
    bool kTagged,
    bool kFuzz>
__global__ void __launch_bounds__(kNumThreads)
    moe_tail_norm_kernel(const __grid_constant__ muse_vr_moe_tail::WaitParams params) {
  using namespace device;
  using namespace device::norm_residual;
  using namespace device::norm_residual_cluster;
  namespace mt = muse_vr_moe_tail;

  constexpr int kVecE = 8;
  constexpr int kTotalThreads = kNumThreads * kClusterSize;
  constexpr int kPacks = kHidden / (kTotalThreads * kVecE);
  constexpr int kNumWarps = kNumThreads / kWarpThreads;
  constexpr float kInvHidden = 1.0f / static_cast<float>(kHidden);
  static_assert(kClusterSize == 2 || kClusterSize == 4 || kClusterSize == 8 || kClusterSize == 16);
  static_assert(kHidden % (kTotalThreads * kVecE) == 0);

  using BF16Vec = AlignedVector<bf16x2_t, kPairs>;
  using F32Vec = AlignedVector<fp32x2_t, kFPairs>;

  const auto& norm = params.norm;
  const int cluster_rank = cooperative_groups::this_cluster().block_rank();
  const int row = static_cast<int>(blockIdx.x) / kClusterSize;
  const int64_t base = static_cast<int64_t>(row) * kHidden;
  const auto stream = static_cast<const fp32_t*>(norm.stream) + base;
  const auto out = static_cast<bf16_t*>(norm.output) + base;
  const auto new_stream = static_cast<fp32_t*>(norm.new_stream) + base;
  const auto post_weight = static_cast<const bf16_t*>(norm.post_weight);
  const auto pre_weight = static_cast<const bf16_t*>(norm.pre_weight);
  const auto alpha_ptr = static_cast<const fp32_t*>(norm.alpha);
  const auto beta_ptr = static_cast<const fp32_t*>(norm.beta);
  const bool channel_wise = kGate && norm.gate_channel_wise;
  const bool has_pre_weight = norm.has_pre_weight;

  __shared__ ReductionStorage4<kClusterSize, kNumWarps> reduction;
  __shared__ uint32_t parity_s;
  if (threadIdx.x == 0) {
    ptx::mbar_init(&reduction.barrier, 1);
    ptx::mbar_arrive_expect_tx(&reduction.barrier, (kClusterSize - 1) * sizeof(float4));
    ptx::publish_cluster_barriers();
  }
  ptx::cluster_arrive_relaxed();

  PDLWaitPrimary<kUsePDL>();

  // Local operands first so their latency overlaps the wait for peer data.
  BF16Vec post_weights[kPacks];
  F32Vec stream_values[kPacks][kFParts];
  F32Vec alpha_values[kPacks][kFParts];
  F32Vec beta_vector[kPacks][kFParts];
  BF16Vec pre_w_pf[kPacks];
#pragma unroll
  for (int pack = 0; pack < kPacks; ++pack) {
    const int index = cluster_rank * kNumThreads + threadIdx.x + pack * kTotalThreads;
    if constexpr (kPostWeight) post_weights[pack].load(post_weight, index);
    if (has_pre_weight) pre_w_pf[pack].load(pre_weight, index);
#pragma unroll
    for (int part = 0; part < kFParts; ++part) {
      const int float_index = index * kFParts + part;
      stream_values[pack][part].load(stream, float_index);
      if constexpr (kGate) {
        if (channel_wise) {
          alpha_values[pack][part].load(alpha_ptr, float_index);
          beta_vector[pack][part].load(beta_ptr, float_index);
        }
      }
    }
  }

  // Local operands are loaded; let the next kernel start its own prologue while this one waits for peer data.
  PDLTriggerSecondary<kUsePDL>();
  if constexpr (kFuzz) {
    if (threadIdx.x == 0) mt::fuzz_delay(params.seq[0] * 104729u + blockIdx.x * 17u, 30u);
    __syncthreads();
  }
  if constexpr (kTagged) {
    if (threadIdx.x == 0) parity_s = params.seq[0];
  } else if (threadIdx.x < kWorld) {
    const uint32_t calls = params.seq[0];
    const uint32_t expected = calls * params.push_grid;
    for (uint32_t polls = 0; static_cast<int32_t>(ptx::load_acquire_sys(params.arrive + threadIdx.x) - expected) < 0;
         ++polls) {
      if (polls > mt::kMaxPolls) __trap();
    }
    if (threadIdx.x == 0) parity_s = (calls - 1u) & 1u;
  }
  __syncthreads();
  uint32_t parity = parity_s;
  uint32_t want_tag = 0;
  if constexpr (kTagged) {
    const uint32_t calls = parity_s;  // the push kernel's call count, read after the dependency wait
    parity = (calls - 1u) & 1u;
    want_tag = (calls - 1u) % 65535u + 1u;
  }
  const auto gathered =
      reinterpret_cast<const bf16_t*>(params.recv + static_cast<int64_t>(parity) * params.parity_bytes) + base;
  const auto gathered_words =
      reinterpret_cast<const uint32_t*>(params.recv + static_cast<int64_t>(parity) * params.parity_bytes) + base;

  fp32x2_t values[kPacks][kPairs];
  fp32x2_t post_w[kPostWeight ? kPacks : 1][kPairs];
  fp32x2_t residual_terms[kPacks][kPairs];
  fp32x2_t beta_values[kPacks][kPairs];

  float branch_sum = 0.0f;
  float residual_sum = 0.0f;
  float cross_sum = 0.0f;
  float scaled_branch_sum = 0.0f;
  fp32x2_t gate_a = {1.0f, 1.0f};
  fp32x2_t gate_b = {1.0f, 1.0f};
  if constexpr (kGate) {
    if (!channel_wise) {
      gate_a.x = gate_a.y = *alpha_ptr;
      gate_b.x = gate_b.y = *beta_ptr;
    }
  }
#pragma unroll
  for (int pack = 0; pack < kPacks; ++pack) {
    const int index = cluster_rank * kNumThreads + threadIdx.x + pack * kTotalThreads;
    uint4 raw;
    if constexpr (kTagged) {
      uint4 w0, w1;
      const uint32_t* src = gathered_words + static_cast<int64_t>(index) * kVecE;
      for (uint32_t polls = 0;; ++polls) {
        ptx::ld_relaxed_16B(w0, src, 0);
        ptx::ld_relaxed_16B(w1, src, 1);
        const bool ready = (w0.x >> 16) == want_tag && (w0.y >> 16) == want_tag && (w0.z >> 16) == want_tag &&
                           (w0.w >> 16) == want_tag && (w1.x >> 16) == want_tag && (w1.y >> 16) == want_tag &&
                           (w1.z >> 16) == want_tag && (w1.w >> 16) == want_tag;
        if (ready) break;
        if (polls > mt::kMaxPolls) __trap();
      }
      raw = make_uint4(
          (w0.x & 0xffffu) | (w0.y << 16), (w0.z & 0xffffu) | (w0.w << 16), (w1.x & 0xffffu) | (w1.y << 16),
          (w1.z & 0xffffu) | (w1.w << 16));
    } else {
      ptx::ld_relaxed_16B(raw, gathered + static_cast<int64_t>(index) * kVecE, 0);
    }
    const auto& branch_values = *reinterpret_cast<const BF16Vec*>(&raw);

#pragma unroll
    for (int pair = 0; pair < kPairs; ++pair) {
      const int part = pair / kFPairs;
      const int lane = pair % kFPairs;
      auto value = cast<fp32x2_t>(branch_values[pair]);
      if constexpr (kPostWeight) {
        const auto weight = cast<fp32x2_t>(post_weights[pack][pair]);
        post_w[pack][pair] = weight;
        if constexpr (kPostNormIsInputScale) {
          value = mt::mul_pair(value, weight);
          value = round_bf16(value);
        }
      }
      auto moment_value = value;
      if constexpr (!kPostNormIsInputScale && kPostWeight) {
        const auto weight = post_w[pack][pair];
        moment_value = mt::mul_pair(moment_value, weight);
      }
      const auto stream_value = stream_values[pack][part][lane];
      auto alpha_value = gate_a;
      auto beta_value = gate_b;
      if constexpr (kGate) {
        if (channel_wise) {
          alpha_value = alpha_values[pack][part][lane];
          beta_value = beta_vector[pack][part][lane];
        }
      }
      const fp32x2_t residual = mt::mul_pair(alpha_value, stream_value);
      const fp32x2_t scaled_branch = mt::mul_pair(beta_value, moment_value);
      values[pack][pair] = value;
      residual_terms[pack][pair] = residual;
      beta_values[pack][pair] = beta_value;
      branch_sum = fmaf(value.x, value.x, branch_sum);
      branch_sum = fmaf(value.y, value.y, branch_sum);
      residual_sum = fmaf(residual.x, residual.x, residual_sum);
      residual_sum = fmaf(residual.y, residual.y, residual_sum);
      cross_sum = fmaf(residual.x, scaled_branch.x, cross_sum);
      cross_sum = fmaf(residual.y, scaled_branch.y, cross_sum);
      scaled_branch_sum = fmaf(scaled_branch.x, scaled_branch.x, scaled_branch_sum);
      scaled_branch_sum = fmaf(scaled_branch.y, scaled_branch.y, scaled_branch_sum);
    }
  }

  local_reduce(branch_sum, residual_sum, cross_sum, scaled_branch_sum, reduction);
  ptx::cluster_wait();
  dsmem_reduce(branch_sum, residual_sum, cross_sum, scaled_branch_sum, reduction, cluster_rank);
  const float post_rstd = math::rsqrt(branch_sum * kInvHidden + norm.post_eps);
  const float pre_sum = fmaf(post_rstd, fmaf(post_rstd, scaled_branch_sum, 2.0f * cross_sum), residual_sum);
  const float pre_rstd = math::rsqrt(fmaxf(pre_sum, 0.0f) * kInvHidden + norm.pre_eps);

#pragma unroll
  for (int pack = 0; pack < kPacks; ++pack) {
    const int index = cluster_rank * kNumThreads + threadIdx.x + pack * kTotalThreads;
    BF16Vec output_values;
    F32Vec new_stream_values[kFParts];
#pragma unroll
    for (int pair = 0; pair < kPairs; ++pair) {
      const int part = pair / kFPairs;
      const int lane = pair % kFPairs;
      auto normed = values[pack][pair];
      normed = mt::mul_pair(normed, {post_rstd, post_rstd});
      if constexpr (!kPostNormIsInputScale && kPostWeight) {
        const auto weight = post_w[pack][pair];
        normed = mt::mul_pair(normed, weight);
      }
      normed = round_bf16(normed);
      const auto beta_value = beta_values[pack][pair];
      const auto residual = residual_terms[pack][pair];
      const fp32x2_t updated = mt::fma_pair(beta_value, normed, residual);
      new_stream_values[part][lane] = updated;
      fp32x2_t normalized = mt::mul_pair(updated, {pre_rstd, pre_rstd});
      if (has_pre_weight) {
        const auto weight = cast<fp32x2_t>(pre_w_pf[pack][pair]);
        normalized = mt::mul_pair(normalized, weight);
      }
      output_values[pair] = cast<bf16x2_t>(normalized);
    }
    output_values.store(out, index);
#pragma unroll
    for (int part = 0; part < kFParts; ++part) {
      new_stream_values[part].store(new_stream, index * kFParts + part);
    }
  }
}

namespace muse_vr_moe_tail {

inline Pointers read_pointers(const tvm::ffi::TensorView ptrs, uint32_t world) {
  using namespace host;
  RuntimeCheck(ptrs.dtype().code == kDLInt && ptrs.dtype().bits == 64, "pointer table must be int64");
  RuntimeCheck(ptrs.device().device_type == kDLCPU, "pointer table must live on the host");
  RuntimeCheck(ptrs.numel() == 2 * world, "pointer table must hold 2 * world entries");
  const auto* v = static_cast<const int64_t*>(ptrs.data_ptr());
  Pointers p{};
  for (uint32_t i = 0; i < world; ++i) {
    p.recv[i] = reinterpret_cast<uint8_t*>(v[i]);
    p.arrive[i] = reinterpret_cast<uint32_t*>(v[world + i]);
  }
  return p;
}

}  // namespace muse_vr_moe_tail

template <int64_t kHidden, int64_t kK, uint32_t kWorld, int kNT, bool kUsePDL, bool kTagged, bool kFuzz>
struct MuseVrMoeTailPush {
  static constexpr int64_t kNLocal = kHidden / kWorld;
  static_assert(kNLocal % muse_vr_moe_tail::kRows == 0);
  static_assert(kWorld >= 2 && kWorld <= muse_vr_moe_tail::kMaxWorld);

  static void
  run(const tvm::ffi::TensorView y,
      const tvm::ffi::TensorView w_local,
      const tvm::ffi::TensorView ptrs,
      const tvm::ffi::TensorView seq,
      const tvm::ffi::TensorView done,
      int64_t rank,
      int64_t parity_bytes) {
    using namespace host;
    namespace mt = muse_vr_moe_tail;
    auto M = SymbolicSize{"num_tokens"};
    auto device_ = SymbolicDevice{};
    device_.set_options<kDLCUDA>();
    TensorMatcher({M, kK}).with_dtype<bf16_t>().with_device(device_).verify(y);
    TensorMatcher({kNLocal, kK}).with_dtype<bf16_t>().with_device(device_).verify(w_local);
    constexpr int64_t kGrid = kNLocal / mt::kRows;
    TensorMatcher({kGrid}).with_dtype<int32_t>().with_device(device_).verify(seq);
    TensorMatcher({1}).with_dtype<int32_t>().with_device(device_).verify(done);
    const auto num_rows = static_cast<uint32_t>(M.unwrap());
    RuntimeCheck(num_rows >= 1 && num_rows <= uint32_t(kNT) * 8, "row count exceeds the instantiated tile");
    RuntimeCheck(rank >= 0 && rank < int64_t(kWorld), "bad rank");
    RuntimeCheck(int64_t(num_rows) * kHidden * 2 <= parity_bytes, "rows exceed the receive region");
    const mt::PushParams params{
        .y = static_cast<const bf16_t*>(y.data_ptr()),
        .w = static_cast<const bf16_t*>(w_local.data_ptr()),
        .ptrs = mt::read_pointers(ptrs, kWorld),
        .seq = static_cast<uint32_t*>(seq.data_ptr()),  // int32 storage, used as uint32
        .done = static_cast<uint32_t*>(done.data_ptr()),
        .rank = static_cast<uint32_t>(rank),
        .num_rows = num_rows,
        .parity_bytes = static_cast<uint32_t>(parity_bytes),
        .col_base = static_cast<uint32_t>(rank * kNLocal),
    };
    constexpr auto kernel = muse_vr_moe_tail::moe_tail_push_kernel<kHidden, kK, kNT, kUsePDL, kTagged, kFuzz>;
    constexpr std::size_t kSmem =
        std::size_t(mt::kRows) * (kK + mt::kRowPad) * 2 + std::size_t(mt::kWarps) * mt::kRows * kNT * 8 * 4;
    [[maybe_unused]] static const bool configured = [] {
      RuntimeDeviceCheck(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, int(kSmem)));
      return true;
    }();
    LaunchKernel(dim3(kGrid), dim3(mt::kThreads), device_.unwrap(), kSmem).config({.use_pdl = kUsePDL})(kernel, params);
  }
};

template <
    int64_t kHidden,
    uint32_t kWorld,
    int kNumThreads,
    int kClusterSize,
    bool kUsePDL,
    bool kPostNormIsInputScale,
    bool kHasPostWeight,
    bool kHasGate,
    bool kTagged,
    bool kFuzz>
struct MuseVrMoeTailNorm {
  static void
  run(const tvm::ffi::TensorView ptrs,
      const tvm::ffi::TensorView seq,
      int64_t rank,
      int64_t parity_bytes,
      const tvm::ffi::TensorView stream,
      const tvm::ffi::TensorView output,
      const tvm::ffi::TensorView new_stream,
      const tvm::ffi::Optional<tvm::ffi::TensorView> post_weight,
      const tvm::ffi::Optional<tvm::ffi::TensorView> pre_weight,
      const tvm::ffi::Optional<tvm::ffi::TensorView> alpha,
      const tvm::ffi::Optional<tvm::ffi::TensorView> beta,
      double post_eps,
      double pre_eps) {
    using namespace host;
    namespace mt = muse_vr_moe_tail;
    auto M = SymbolicSize{"num_tokens"};
    auto H = SymbolicSize{"hidden_size"};
    auto G = SymbolicSize{"gate_dim"};
    auto P = SymbolicSize{"push_grid"};
    auto device_ = SymbolicDevice{};
    H.set_value(kHidden);
    device_.set_options<kDLCUDA>();
    TensorMatcher({M, H}).with_dtype<fp32_t>().with_device(device_).verify(stream);
    TensorMatcher({M, H}).with_dtype<bf16_t>().with_device(device_).verify(output);
    TensorMatcher({M, H}).with_dtype<fp32_t>().with_device(device_).verify(new_stream);
    TensorMatcher({P}).with_dtype<int32_t>().with_device(device_).verify(seq);
    if (post_weight.has_value()) {
      TensorMatcher({H}).with_dtype<bf16_t>().with_device(device_).verify(post_weight.value());
    }
    if (pre_weight.has_value()) {
      TensorMatcher({H}).with_dtype<bf16_t>().with_device(device_).verify(pre_weight.value());
    }
    const bool gate = alpha.has_value();
    RuntimeCheck(gate == beta.has_value(), "alpha and beta must both be present or both absent");
    RuntimeCheck(post_weight.has_value() == kHasPostWeight, "post_weight presence must match kHasPostWeight");
    RuntimeCheck(gate == kHasGate, "alpha/beta presence must match kHasGate");
    bool gate_channel_wise = false;
    if (gate) {
      TensorMatcher({G}).with_dtype<fp32_t>().with_device(device_).verify(alpha.value());
      TensorMatcher({G}).with_dtype<fp32_t>().with_device(device_).verify(beta.value());
      const auto n = G.unwrap();
      RuntimeCheck(n == 1 || n == kHidden, "alpha/beta must have 1 or hidden_size elements");
      gate_channel_wise = (n == kHidden);
    }
    const auto num_rows = static_cast<uint32_t>(M.unwrap());
    RuntimeCheck(num_rows >= 1, "requires M >= 1");
    RuntimeCheck(int64_t(num_rows) * kHidden * 2 <= parity_bytes, "rows exceed the receive region");
    RuntimeCheck(rank >= 0 && rank < int64_t(kWorld), "bad rank");
    const auto table = mt::read_pointers(ptrs, kWorld);
    const mt::WaitParams params{
        .recv = table.recv[rank],
        .arrive = table.arrive[rank],
        .seq = static_cast<const uint32_t*>(seq.data_ptr()),
        .push_grid = 1u,  // one arrival per call and source rank
        .parity_bytes = static_cast<uint32_t>(parity_bytes),
        .num_rows = num_rows,
        .norm =
            NormResidualParams{
                .branch = nullptr,
                .stream = stream.data_ptr(),
                .output = output.data_ptr(),
                .new_stream = new_stream.data_ptr(),
                .router_out = nullptr,
                .post_weight = post_weight.has_value() ? post_weight.value().data_ptr() : nullptr,
                .pre_weight = pre_weight.has_value() ? pre_weight.value().data_ptr() : nullptr,
                .alpha = gate ? alpha.value().data_ptr() : nullptr,
                .beta = gate ? beta.value().data_ptr() : nullptr,
                .post_eps = static_cast<float>(post_eps),
                .pre_eps = static_cast<float>(pre_eps),
                .num_rows = static_cast<int>(num_rows),
                .gate_channel_wise = gate_channel_wise,
                .has_pre_weight = pre_weight.has_value(),
                .pre_norm_bf16 = false,
            },
    };
    constexpr auto kernel = moe_tail_norm_kernel<
        kHidden,
        kWorld,
        kNumThreads,
        kClusterSize,
        kUsePDL,
        kPostNormIsInputScale,
        kHasPostWeight,
        kHasGate,
        kTagged,
        kFuzz>;
    [[maybe_unused]] static const bool cluster_configured = [] {
      if constexpr (kClusterSize > 8) {
        RuntimeDeviceCheck(cudaFuncSetAttribute(kernel, cudaFuncAttributeNonPortableClusterSizeAllowed, 1));
      }
      return true;
    }();
    LaunchKernel(dim3(num_rows * kClusterSize), dim3(kNumThreads), device_.unwrap())
        .config({.use_pdl = kUsePDL, .cluster_dim = dim3(kClusterSize, 1, 1)})(kernel, params);
  }
};

}  // namespace sglang
