// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
/**
 * \brief Sequence-parallel MoE finalize (unpermute + post-expert RMSNorm + router-weighted top-k sum) with the per-element
 * arithmetic of SGLang's single-CTA CuTe-DSL finalize kernel (``MOEFinalizeRMSNormKernel`` at cluster size 1, the launch
 * the stock op makes for more than 128 tokens at hidden 4096), bitwise to it, on a different schedule.
 *
 * Stock: one 128-thread CTA per token; for each local expert of the token, in ascending top-k slot order, the CTA loads
 * the expert's GEMM2 row, reduces its sum of squares and accumulates, so every expert row is a dependent load ->
 * reduce chain.
 *
 * Here: persistent CTAs of one producer warp + kGroups consumer groups of 4 warps, each CTA on a contiguous token range
 * whose slot indices and router weights are staged in shared memory first. The producer walks the range and issues one
 * 8 KB bulk copy (global -> shared::cta, mbarrier transaction count) per local expert row into a ring of row slots
 * shared by the groups, running ahead of the consumers as far as the ring allows; group g consumes every kGroups-th
 * token of the range. Each group
 * keeps the stock thread map exactly (group thread t holds columns t*8 + p*1024 + v, p < 4, v < 8) and the stock
 * expressions:
 *   value = float(x) * float(w_norm)                      (weight-first form; standard form: value = float(x))
 *   sq    = fma(value, value, sq)       over p = 0..3, v = 0..7, from sq = 0
 *   s     = butterfly sum over the warp (xor 16, 8, 4, 2, 1); lane 0's value per warp -> smem
 *   s     = butterfly sum over lanes of {warp sums of warps 0..3, zeros}   (every lane the same value)
 *   inv   = rsqrt.approx.ftz(fma(s, 2^-12, eps))
 *   out   = value * inv (standard form: then * float(w_norm))
 *   acc   = fma(w_e, out, acc)          experts in ascending slot order, acc from 0
 *   one round-to-nearest bf16 conversion of acc per element; 16-byte stores.
 * The element-wise products and the accumulation run as packed fp32 pairs (.rn.f32x2), the same IEEE operation per
 * element; the sum-of-squares chain stays scalar in the stock order.
 * Only the schedule changes: both reductions of every expert of a token run before any accumulation (each expert's
 * own reduction tree is unchanged, and the accumulation order over experts is unchanged); ``value`` is recomputed from
 * the same bf16 input and weight in the second pass (the same single product).
 *
 * Hand-offs (all inside one CTA): full[stage] (producer arrive + expect-tx, bulk-copy complete-tx; consumers wait),
 * tok_empty[stage] and slot_empty[slot] (thread 0 of the consuming group arrives after the group's 128-thread named
 * barrier 1 + g; the producer waits before reusing the metadata stage / row slot, then issues
 * fence.proxy.async.shared::cta before the next bulk copy into the slot). Every wait is bounded (``kMaxPolls``) and traps on overflow. No cross-CTA or cross-GPU communication.
 */

#include <sgl_kernel/tensor.h>
#include <sgl_kernel/utils.h>

#include <sgl_kernel/runtime.cuh>
#include <sgl_kernel/utils.cuh>

#include <cuda_bf16.h>
#include <dlpack/dlpack.h>
#include <tvm/ffi/container/tensor.h>

#include <cstdint>

// Test-only builds (tests/test_pfin.py), serving builds leave both 0:
//   MUSE_VR_PFIN_FUZZ = 1: randomized delays in the producer and consumers (hand-off robustness);
//   MUSE_VR_PFIN_SLOW_CONSUMER = cycles: consumer thread 0 spins that many clock64 cycles before releasing each token's
//   row slots (a slow consumer against the smallest ring).
#ifndef MUSE_VR_PFIN_FUZZ
#define MUSE_VR_PFIN_FUZZ 0
#endif
#ifndef MUSE_VR_PFIN_SLOW_CONSUMER
#define MUSE_VR_PFIN_SLOW_CONSUMER 0
#endif

namespace muse_vr {
#if MUSE_VR_PFIN_FUZZ || MUSE_VR_PFIN_SLOW_CONSUMER
inline namespace pfin_test_build {
#endif
namespace pfin {

constexpr int kTopK = 8;
constexpr int kHidden = 4096;
constexpr int kRowBytes = kHidden * 2;              // one bf16 GEMM2 row
constexpr int kConsumers = 128;                     // one consumer group = the stock thread map
constexpr int kVec = 8;                             // bf16 per 16-byte access
constexpr int kIters = kHidden / (kConsumers * kVec);  // 4
constexpr int kTokStages = 32;                      // token metadata ring
constexpr int kMaxChunk = 256;                      // tokens per CTA (contiguous range; slot indices / weights staged)
constexpr uint32_t kMaxPolls = 1u << 26;

struct Params {
  const __nv_bfloat16* gemm2;  // [rows, row_stride]
  const void* expert_weights;  // [T * 8] fp32 or bf16
  const int32_t* idx;          // [T * 8], -1 = not local
  const __nv_bfloat16* rms_w;  // [4096]
  __nv_bfloat16* out;          // [T, 4096], contiguous
  int64_t row_stride;          // GEMM2 row stride in elements
  int32_t tokens;
  float eps;
};

struct TokMeta {
  int32_t count;
  int32_t token;
  int32_t slot[kTopK];
  float weight[kTopK];
};

template <int kGroups>
constexpr int threads_for() {
  return kConsumers * kGroups + 32;  // consumer groups + one producer warp
}

template <int kSlots, int kGroups>
struct Smem {
  // No circular wait: a token is consumed once all of its rows (<= top-k) are resident, so the row ring must hold a
  // whole token; the metadata ring must hold one token per consumer group plus at least one the producer publishes
  // ahead. Groups consume published tokens in order, so every slot / stage the producer waits for is released.
  static_assert(kSlots >= kTopK, "row ring must hold every expert row of one token");
  static_assert(kGroups >= 1 && kTokStages >= kGroups + 1, "metadata ring must hold every group's token and the next");
  alignas(128) uint8_t rows[kSlots][kRowBytes];
  int32_t chunk_idx[kMaxChunk * kTopK];  // the CTA's tokens' slot -> GEMM2 row indices
  float chunk_w[kMaxChunk * kTopK];      // and router weights (as float)
  TokMeta meta[kTokStages];
  float warp_sums[kGroups][kTopK][4];
  alignas(8) uint64_t full[kTokStages];
  alignas(8) uint64_t tok_empty[kTokStages];
  alignas(8) uint64_t slot_empty[kSlots];
};

__device__ __forceinline__ uint32_t smem_u32(const void* p) {
  return static_cast<uint32_t>(__cvta_generic_to_shared(p));
}

__device__ __forceinline__ void bar_init(uint64_t* bar, uint32_t count) {
  asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" ::"r"(smem_u32(bar)), "r"(count) : "memory");
}

__device__ __forceinline__ void bar_arrive(uint64_t* bar) {
  asm volatile(
      "{\n\t.reg .b64 st;\n\t"
      "mbarrier.arrive.release.cta.shared::cta.b64 st, [%0];\n\t}" ::"r"(smem_u32(bar))
      : "memory");
}

__device__ __forceinline__ void bar_arrive_expect_tx(uint64_t* bar, uint32_t bytes) {
  asm volatile(
      "{\n\t.reg .b64 st;\n\t"
      "mbarrier.arrive.expect_tx.release.cta.shared::cta.b64 st, [%0], %1;\n\t}" ::"r"(smem_u32(bar)),
      "r"(bytes)
      : "memory");
}

// Wait until the phase with parity ``parity`` has completed (acquire, CTA scope); bounded, traps on overflow.
__device__ __forceinline__ void bar_wait(uint64_t* bar, uint32_t parity) {
  const uint32_t addr = smem_u32(bar);
  for (uint32_t polls = 0;; ++polls) {
    uint32_t ready;
    asm volatile(
        "{\n\t.reg .pred p;\n\t"
        "mbarrier.try_wait.parity.acquire.cta.shared::cta.b64 p, [%1], %2;\n\t"
        "selp.b32 %0, 1, 0, p;\n\t}"
        : "=r"(ready)
        : "r"(addr), "r"(parity)
        : "memory");
    if (ready) return;
    if (polls > kMaxPolls) __trap();
  }
}

__device__ __forceinline__ void bulk_g2s(void* dst, const void* src, uint32_t bytes, uint64_t* bar) {
  asm volatile(
      "cp.async.bulk.shared::cta.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];" ::"r"(smem_u32(dst)),
      "l"(src), "r"(bytes), "r"(smem_u32(bar))
      : "memory");
}

__device__ __forceinline__ void fence_proxy_async_smem() {
  asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
}

// Barrier 1 + g: exactly the 128 threads of consumer group g (the producer warp never uses ids >= 1).
__device__ __forceinline__ void named_sync_group(int g) {
  asm volatile("bar.sync %0, %1;" ::"r"(1 + g), "n"(kConsumers) : "memory");
}

__device__ __forceinline__ float rsqrt_approx_ftz(float x) {
  float y;
  asm("rsqrt.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x));
  return y;
}

__device__ __forceinline__ float warp_sum(float x) {
#pragma unroll
  for (int off = 16; off >= 1; off >>= 1) x = __fadd_rn(x, __shfl_xor_sync(0xffffffffu, x, off));
  return x;
}

__device__ __forceinline__ void fuzz_delay(uint32_t key, uint32_t max_ns) {
  uint32_t h = key * 2654435761u;
  h ^= h >> 15;
  h *= 2246822519u;
  h ^= h >> 13;
  for (uint32_t i = (h % max_ns) / 64; i > 0; --i) __nanosleep(64);
}

__device__ __forceinline__ uint4 lds128(const void* p) {
  uint4 v;
  asm volatile("ld.shared.v4.b32 {%0, %1, %2, %3}, [%4];"
               : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w)
               : "r"(smem_u32(p)));
  return v;
}

__device__ __forceinline__ float bf_lo(uint32_t w) { return __uint_as_float(w << 16); }
__device__ __forceinline__ float bf_hi(uint32_t w) { return __uint_as_float(w & 0xffff0000u); }

// Packed fp32 pairs (PTX ISA mul / fma .rn.f32x2): each lane of the pair is the same IEEE round-to-nearest operation
// as the scalar .rn instruction (subnormals kept), so the per-element results are unchanged; only issue slots halve.
__device__ __forceinline__ void mul2_rn(float& d0, float& d1, float a0, float a1, float b0, float b1) {
  asm("{\n\t.reg .b64 ra, rb, rd;\n\t"
      "mov.b64 ra, {%2, %3};\n\tmov.b64 rb, {%4, %5};\n\t"
      "mul.rn.f32x2 rd, ra, rb;\n\t"
      "mov.b64 {%0, %1}, rd;\n\t}"
      : "=f"(d0), "=f"(d1)
      : "f"(a0), "f"(a1), "f"(b0), "f"(b1));
}

__device__ __forceinline__ void fma2_rn(float& d0, float& d1, float a0, float a1, float b0, float b1, float c0,
                                        float c1) {
  asm("{\n\t.reg .b64 ra, rb, rc, rd;\n\t"
      "mov.b64 ra, {%2, %3};\n\tmov.b64 rb, {%4, %5};\n\tmov.b64 rc, {%6, %7};\n\t"
      "fma.rn.f32x2 rd, ra, rb, rc;\n\t"
      "mov.b64 {%0, %1}, rd;\n\t}"
      : "=f"(d0), "=f"(d1)
      : "f"(a0), "f"(a1), "f"(b0), "f"(b1), "f"(c0), "f"(c1));
}

template <bool kWeightFirst>
__device__ __forceinline__ void load_values(const uint8_t* row, int t, const float (&wn)[kIters][kVec],
                                            float (&val)[kIters][kVec]) {
#pragma unroll
  for (int p = 0; p < kIters; ++p) {
    const uint4 q = lds128(row + (t * kVec + p * kConsumers * kVec) * 2);
    const uint32_t w4[4] = {q.x, q.y, q.z, q.w};
#pragma unroll
    for (int h = 0; h < 4; ++h) {
      const float a = bf_lo(w4[h]);
      const float b = bf_hi(w4[h]);
      if constexpr (kWeightFirst) {
        mul2_rn(val[p][2 * h], val[p][2 * h + 1], a, b, wn[p][2 * h], wn[p][2 * h + 1]);
      } else {
        val[p][2 * h] = a;
        val[p][2 * h + 1] = b;
      }
    }
  }
}

template <int kSlots, int kGroups, bool kWeightFirst, bool kBf16Weights>
__global__ void __launch_bounds__(threads_for<kGroups>()) pfin_kernel(const __grid_constant__ Params p) {
  extern __shared__ __align__(128) uint8_t smem_raw[];
  auto& s = *reinterpret_cast<Smem<kSlots, kGroups>*>(smem_raw);
  const int tid = threadIdx.x;
  const int warp = tid >> 5;
  const int lane = tid & 31;

  if (tid == 0) {
    for (int i = 0; i < kTokStages; ++i) {
      bar_init(&s.full[i], 1);
      bar_init(&s.tok_empty[i], 1);
    }
    for (int i = 0; i < kSlots; ++i) bar_init(&s.slot_empty[i], 1);
    asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
  }
  // This CTA's contiguous token range; its slot indices and router weights are staged once (coalesced loads).
  const int per_cta = (p.tokens + static_cast<int>(gridDim.x) - 1) / static_cast<int>(gridDim.x);
  const int tok_begin = static_cast<int>(blockIdx.x) * per_cta;
  const int n_tok = max(0, min(p.tokens - tok_begin, per_cta));
  for (int e = tid; e < n_tok * kTopK; e += static_cast<int>(blockDim.x)) {
    const int64_t ge = static_cast<int64_t>(tok_begin) * kTopK + e;
    s.chunk_idx[e] = p.idx[ge];
    if constexpr (kBf16Weights) {
      s.chunk_w[e] = __bfloat162float(static_cast<const __nv_bfloat16*>(p.expert_weights)[ge]);
    } else {
      s.chunk_w[e] = static_cast<const float*>(p.expert_weights)[ge];
    }
  }
  __syncthreads();

  if (warp == kGroups * (kConsumers / 32)) {
    // ---------------- producer warp ----------------
    // kBatch tokens per iteration, one 8-lane group per token (lane group k = lanes 8k..8k+7 hold the token's 8 slots).
    // A batch needs at most kBatch * 8 <= kSlots row slots, so its rows land in distinct slots whose previous uses
    // belong to earlier batches: the per-lane waits below never depend on another lane of the same batch.
    constexpr int kBatch = kSlots / kTopK;
    static_assert(kBatch >= 1 && kBatch * kTopK <= 32, "batch must fit one warp");
    const int k = lane / kTopK;      // token of the batch this lane serves
    const int e = lane % kTopK;      // its top-k slot
    const bool leader = (e == 0) && (k < kBatch);
    uint32_t row_ctr = 0;
    for (int i0 = 0; i0 < n_tok; i0 += kBatch) {
      const int i = i0 + k;
      const bool active = (k < kBatch) && (i < n_tok);
      int32_t row = -1;
      float w = 0.f;
      if (active) {
        row = s.chunk_idx[i * kTopK + e];
        w = s.chunk_w[i * kTopK + e];
      }
      const uint32_t ball = __ballot_sync(0xffffffffu, active && row >= 0);
      const uint32_t mask = (ball >> (k * kTopK)) & 0xffu;  // this lane's token, slot order
      const int n = __popc(mask);
      // rows before this token in the batch (ascending k), for the FIFO slot counter
      const uint32_t below = k == 0 ? 0u : (ball & ((1u << (k * kTopK)) - 1u));
      const uint32_t base = row_ctr + static_cast<uint32_t>(__popc(below));
      // the token's local experts in ascending slot order (the stock kernel's ballot order), gathered to its leader
      int32_t rj[kTopK];
      float wj[kTopK];
      uint32_t m = mask;
#pragma unroll
      for (int j = 0; j < kTopK; ++j) {
        const int src = k * kTopK + (m ? __ffs(static_cast<int>(m)) - 1 : 0);
        rj[j] = __shfl_sync(0xffffffffu, row, src & 31);
        wj[j] = __shfl_sync(0xffffffffu, w, src & 31);
        if (m) m &= m - 1u;
      }
      // Each leader does every wait, metadata write, arrive and copy issue of its token (program order).
      if (leader && active) {
        const int tok = tok_begin + i;
        const int stage = i % kTokStages;
        const uint32_t round = static_cast<uint32_t>(i / kTokStages);
        // metadata stage free (consumers finished token i - kTokStages)
        bar_wait(&s.tok_empty[stage], (round & 1u) ^ 1u);
        int slot_of[kTopK];
#pragma unroll
        for (int j = 0; j < kTopK; ++j) {
          slot_of[j] = 0;
          if (j < n) {
            const uint32_t use = base + static_cast<uint32_t>(j);
            slot_of[j] = static_cast<int>(use % kSlots);
            // row slot free (consumers finished its previous use)
            bar_wait(&s.slot_empty[slot_of[j]], ((use / kSlots) & 1u) ^ 1u);
            s.meta[stage].slot[j] = slot_of[j];
            s.meta[stage].weight[j] = wj[j];
          }
        }
        s.meta[stage].count = n;
        s.meta[stage].token = tok;
        if constexpr (MUSE_VR_PFIN_FUZZ) fuzz_delay(static_cast<uint32_t>(tok) * 7u + 1u, 3000u);
        bar_arrive_expect_tx(&s.full[stage], static_cast<uint32_t>(n) * kRowBytes);
        // order the consumers' earlier generic-proxy reads of the reused slots (acquired above) before the
        // async-proxy writes of the copies below
        fence_proxy_async_smem();
#pragma unroll
        for (int j = 0; j < kTopK; ++j) {
          if (j < n) {
            bulk_g2s(s.rows[slot_of[j]], p.gemm2 + static_cast<int64_t>(rj[j]) * p.row_stride, kRowBytes,
                     &s.full[stage]);
          }
        }
      }
      __syncwarp();
      row_ctr += static_cast<uint32_t>(__popc(ball));
    }
    return;
  }

  // ---------------- consumer groups (each the stock thread map); group g takes every kGroups-th token ----------------
  const int g = tid / kConsumers;
  const int t = tid % kConsumers;
  const int gwarp = t >> 5;
  float wn[kIters][kVec];
#pragma unroll
  for (int pp = 0; pp < kIters; ++pp) {
    const uint4 q = *reinterpret_cast<const uint4*>(p.rms_w + t * kVec + pp * kConsumers * kVec);
    const uint32_t w4[4] = {q.x, q.y, q.z, q.w};
#pragma unroll
    for (int h = 0; h < 4; ++h) {
      wn[pp][2 * h] = bf_lo(w4[h]);
      wn[pp][2 * h + 1] = bf_hi(w4[h]);
    }
  }
  const float scale = 1.0f / 4096.0f;  // exact 2^-12: the stock kernel's sum / hidden_dim + eps is fma(sum, 2^-12, eps)

  for (int i = g; i < n_tok; i += kGroups) {
    const int tok = tok_begin + i;
    const int stage = i % kTokStages;
    const uint32_t round = static_cast<uint32_t>(i / kTokStages);
    if constexpr (MUSE_VR_PFIN_FUZZ) fuzz_delay(static_cast<uint32_t>(tok) * 13u + static_cast<uint32_t>(t), 2000u);
    bar_wait(&s.full[stage], round & 1u);
    const int n = s.meta[stage].count;

    // pass 1: per expert, the per-thread sum of squares and its warp butterfly; lane 0 publishes the warp sum
    for (int j = 0; j < n; ++j) {
      float val[kIters][kVec];
      load_values<kWeightFirst>(s.rows[s.meta[stage].slot[j]], t, wn, val);
      float sq = 0.f;
#pragma unroll
      for (int pp = 0; pp < kIters; ++pp)
#pragma unroll
        for (int v = 0; v < kVec; ++v) sq = __fmaf_rn(val[pp][v], val[pp][v], sq);
      sq = warp_sum(sq);
      if (lane == 0) s.warp_sums[g][j][gwarp] = sq;
    }
    named_sync_group(g);

    float acc[kIters][kVec];
#pragma unroll
    for (int pp = 0; pp < kIters; ++pp)
#pragma unroll
      for (int v = 0; v < kVec; ++v) acc[pp][v] = 0.f;

    // pass 2: per expert in slot order, the cross-warp sum, rsqrt and the router-weighted accumulation
    for (int j = 0; j < n; ++j) {
      float sw = lane < 4 ? s.warp_sums[g][j][lane] : 0.f;
      sw = warp_sum(sw);
      const float inv = rsqrt_approx_ftz(__fmaf_rn(sw, scale, p.eps));
      const float we = s.meta[stage].weight[j];
      float val[kIters][kVec];
      load_values<kWeightFirst>(s.rows[s.meta[stage].slot[j]], t, wn, val);
#pragma unroll
      for (int pp = 0; pp < kIters; ++pp)
#pragma unroll
        for (int v = 0; v < kVec; v += 2) {
          float o0, o1;
          mul2_rn(o0, o1, val[pp][v], val[pp][v + 1], inv, inv);
          if constexpr (!kWeightFirst) mul2_rn(o0, o1, o0, o1, wn[pp][v], wn[pp][v + 1]);
          fma2_rn(acc[pp][v], acc[pp][v + 1], we, we, o0, o1, acc[pp][v], acc[pp][v + 1]);
        }
    }
    // every thread of the group is done with this token's rows, metadata and warp sums
    named_sync_group(g);
    if (t == 0) {
      if constexpr (MUSE_VR_PFIN_FUZZ) fuzz_delay(static_cast<uint32_t>(tok) * 31u + 5u, 3000u);
      if constexpr (MUSE_VR_PFIN_SLOW_CONSUMER > 0) {
        const long long t_end = clock64() + static_cast<long long>(MUSE_VR_PFIN_SLOW_CONSUMER);
        while (clock64() < t_end) {
        }
      }
      for (int j = 0; j < n; ++j) bar_arrive(&s.slot_empty[s.meta[stage].slot[j]]);
      bar_arrive(&s.tok_empty[stage]);
    }

    __nv_bfloat16* orow = p.out + static_cast<int64_t>(tok) * kHidden;
#pragma unroll
    for (int pp = 0; pp < kIters; ++pp) {
      uint4 q;
      uint32_t* qw = reinterpret_cast<uint32_t*>(&q);
#pragma unroll
      for (int h = 0; h < 4; ++h) {
        const __nv_bfloat162 b2 = __floats2bfloat162_rn(acc[pp][2 * h], acc[pp][2 * h + 1]);
        qw[h] = *reinterpret_cast<const uint32_t*>(&b2);
      }
      *reinterpret_cast<uint4*>(orow + t * kVec + pp * kConsumers * kVec) = q;
    }
  }
}

template <int kSlots, int kGroups, bool kWeightFirst, bool kBf16Weights>
static void launch(const Params& prm, int grid, DLDevice dev) {
  using namespace sglang::host;
  constexpr auto kernel = pfin_kernel<kSlots, kGroups, kWeightFirst, kBf16Weights>;
  constexpr std::size_t kSmem = sizeof(Smem<kSlots, kGroups>) + 128;
  [[maybe_unused]] static const bool configured = [] {
    RuntimeDeviceCheck(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, int(kSmem)));
    return true;
  }();
  LaunchKernel(dim3(static_cast<unsigned>(grid)), dim3(threads_for<kGroups>()), dev, kSmem)(kernel, prm);
}

}  // namespace pfin

template <int kSlots, int kGroups, bool kWeightFirst, bool kBf16Weights>
struct PFin {
  static void run(const tvm::ffi::TensorView gemm2, const tvm::ffi::TensorView expert_weights,
                  const tvm::ffi::TensorView idx, const tvm::ffi::TensorView rms_w, const tvm::ffi::TensorView out,
                  double eps, int64_t grid) {
    using namespace sglang::host;
    pfin::Params prm{};
    RuntimeCheck(gemm2.dim() == 2 && gemm2.size(1) >= pfin::kHidden && gemm2.stride(1) == 1, "gemm2 layout");
    RuntimeCheck(out.dim() == 2 && out.size(1) == pfin::kHidden && out.stride(0) == pfin::kHidden && out.stride(1) == 1,
                 "out layout");
    RuntimeCheck(idx.numel() == out.size(0) * pfin::kTopK, "idx size");
    RuntimeCheck(expert_weights.numel() == out.size(0) * pfin::kTopK, "expert_weights size");
    RuntimeCheck((reinterpret_cast<uintptr_t>(gemm2.data_ptr()) & 15) == 0 && (gemm2.stride(0) * 2) % 16 == 0 &&
                     (reinterpret_cast<uintptr_t>(out.data_ptr()) & 15) == 0 &&
                     (reinterpret_cast<uintptr_t>(rms_w.data_ptr()) & 15) == 0,
                 "16-byte alignment");
    prm.gemm2 = static_cast<const __nv_bfloat16*>(gemm2.data_ptr());
    prm.expert_weights = expert_weights.data_ptr();
    prm.idx = static_cast<const int32_t*>(idx.data_ptr());
    prm.rms_w = static_cast<const __nv_bfloat16*>(rms_w.data_ptr());
    prm.out = static_cast<__nv_bfloat16*>(out.data_ptr());
    prm.row_stride = gemm2.stride(0);
    prm.tokens = static_cast<int32_t>(out.size(0));
    prm.eps = static_cast<float>(eps);
    if (prm.tokens == 0) return;
    int g = static_cast<int>(grid < prm.tokens ? grid : prm.tokens);
    const int min_grid = (prm.tokens + pfin::kMaxChunk - 1) / pfin::kMaxChunk;  // every CTA's range fits the stage
    if (g < min_grid) g = min_grid;
    pfin::launch<kSlots, kGroups, kWeightFirst, kBf16Weights>(prm, g, gemm2.device());
  }
};

#if MUSE_VR_PFIN_FUZZ || MUSE_VR_PFIN_SLOW_CONSUMER
}  // inline namespace pfin_test_build
#endif
}  // namespace muse_vr
