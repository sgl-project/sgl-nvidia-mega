// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// Portions Copyright 2023-2024 SGLang Team (Apache-2.0)
// Adapted from SGLang (python/sglang/kernels/jit/csrc/distributed/glm_ll_mnnvl_oneshot.cuh of the
// GLM-5.2 low-latency stack, with the nvcc 13.4 template disambiguator), which adapts FlashInfer v0.7.1
// include/flashinfer/comm/trtllm_mnnvl_allreduce.cuh (Apache-2.0). Modified for the plugin: every
// cross-GPU wait is bounded (deadline, host-mapped error record, trap), and the Lamport buffer and its
// flags are the plugin's own (glm5_3/ll/transport.py), not FlashInfer's workspace.
//
// One-shot MNNVL all-reduce + residual add + RMSNorm (bf16) on a Lamport
// buffer in FlashInfer's MNNVL layout, for the GLM-5.2 low-latency decode hop.
//
// FlashInfer v0.7.1 trtllm_mnnvl_allreduce.cuh::oneshotAllreduceFusionKernel
// (kARResidualRMSNorm) with the TensorRT-LLM GLM decode deltas:
//   * residual/gamma copied to smem with cp.async before the Lamport exchange,
//     warp sums of squares exchanged with st.async + a transaction mbarrier
//     (TRT-LLM 77fee291, a1a2ed03);
//   * per-CTA arrival without a cluster barrier, flags rotated right after the
//     poll (b6dd1dfe); max smem carveout, set per launch (edd689ce);
//   * optional PDL trigger right after the grid dependency wait (ecb66c84).
//
// Bit-identical to FlashInfer's one-shot kernel for the same launch geometry
// (the host passes FlashInfer's adjustGridConfig geometry where FlashInfer runs
// one-shot) and the same flags (--use_fast_math). Above FlashInfer's one-shot
// threshold the host may pick another cluster size (the sum-of-squares order
// follows it) and a unicast publish (kUnicast: one store per peer instead of
// the multicast store, so the switch does not return this rank's own copy;
// same values, same bits as the multicast publish). Buffers, flag layout,
// rotation values and clears are FlashInfer's:
//   flags = [cur idx, dirty idx, bytes per buffer, dirty num stages,
//            bytes to clear x4, arrival counter]
// Every rank of the buffer must run this kernel for the same call.
//
// Fail-stop: every poll of a peer's data reads %globaltimer once per
// kPollTimerStride polls; past Params::deadline_ns it writes the host-mapped
// error record (magic, kind, site, rank | peer << 8 | row << 16, buffer index)
// and executes trap, so a rank that never joins a hop stops its peers within
// the deadline instead of hanging them. The values and the bits are unchanged.
//
// Two optional ends turn it into the other GLM decode hops (same protocol, plan
// and bits for the shared part):
//   * finalize prologue (kTopK > 0): this rank's partial is the MoE finalize of
//     its FC2 rows (trtllm-gen do_finalize=False operands, or W1a's token-major
//     rows with kIdentityRows) plus the shared-expert row, computed as
//     moe_finalize_fuse_shared does (fp32 FMA in k order, one RN rounding), so
//     the hop equals that kernel followed by the plain hop bit for bit;
//   * NVFP4 epilogue (kFp4Out): the norm output is also quantized to NVFP4 with
//     FlashInfer's MNNVL quant epilogue (cvt_warp_fp16_to_fp4, linear SF layout),
//     the routed-MoE input of the next layer's FC1.
#pragma once

#include <sgl_kernel/tensor.h>
#include <sgl_kernel/utils.h>

#include <sgl_kernel/utils.cuh>

#include <tvm/ffi/extra/stl.h>

#include <cooperative_groups.h>
#include <cstdint>
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cuda_pipeline.h>
#include <optional>
#include <type_traits>
#include <utility>

namespace sglang {
namespace glm_ll_mnnvl {

namespace cg = cooperative_groups;

inline constexpr int kElts = 8;  // bf16 per 16-byte vector, one vector per thread
inline constexpr int kMaxClusterSize = 8;
inline constexpr int kWarpSize = 32;
inline constexpr uint32_t kNegZeroF32 = 0x80000000u;
inline constexpr uint16_t kNegZeroBF16 = 0x8000u;
// Fail-stop: %globaltimer is read once every kPollTimerStride polls; the error
// record's magic word and wait kinds (glm5_3/ll/transport.py decodes them).
inline constexpr uint32_t kPollTimerStride = 16;
inline constexpr uint32_t kErrMagic = 0x474C4C48u;  // "GLLH"
inline constexpr uint32_t kErrOneshotPoll = 1;
inline constexpr uint32_t kErrScatterPoll = 2;
inline constexpr uint32_t kErrBroadcastPoll = 3;
inline constexpr uint32_t kErrArrivalWait = 4;

union Pack {
  uint4 u4;
  uint32_t words[4];
  __nv_bfloat16 e[kElts];
};

struct Params {
  __nv_bfloat16* norm_out;            // [M, H]
  __nv_bfloat16* residual_out;        // [M, H]
  const __nv_bfloat16* input;         // [M, H], this rank's partial sum
  const __nv_bfloat16* residual_in;   // [M, H]
  const __nv_bfloat16* gamma;         // [H]
  __nv_bfloat16* const* buffer_ptrs;  // [W] unicast Lamport buffer bases (device array)
  __nv_bfloat16* mcast_ptr;           // multicast Lamport buffer base
  uint32_t* flags;                    // FlashInfer buffer_flags (9 x u32)
  int rank;
  int num_tokens;
  int token_dim;
  float eps;
  float weight_bias;  // FlashInfer's `weight_bias`, 0 for the standard RMSNorm
  // Fail-stop: the host-mapped error record (8 x u32), the bound of every wait
  // on a peer and the caller's site id (copied into the record).
  uint32_t* err;
  unsigned long long deadline_ns;
  uint32_t site;
};

// The variants' extra kernel argument. The plain hop takes NoExt, so its
// code stays the plain one-shot kernel's.
struct NoExt {};

// Finalize prologue (kTopK > 0); Params::input is unused then.
struct FinalizeExt {
  const __nv_bfloat16* gemm2;  // [P, gemm2_stride] FC2 rows
  int64_t gemm2_stride;
  const int32_t* permuted_idx;  // [M, kTopK], -1 = empty slot (not read with kIdentityRows)
  const void* expert_weights;   // [M, kTopK] WeightT, routed scaling folded in
  const __nv_bfloat16* shared;  // [M, H] shared-expert output, or null
};

// NVFP4 epilogue (kFp4Out).
struct Fp4Ext {
  uint32_t* out;       // [M, H / 8] u32, eight e2m1 codes per word
  uint8_t* sf_out;     // [M, H / 16] e4m3 block scales, linear layout
  const float* scale;  // [1] global scale
};

template <int kTopK, bool kFp4Out>
using HopExt = std::conditional_t<(kTopK > 0), FinalizeExt, std::conditional_t<kFp4Out, Fp4Ext, NoExt>>;

#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900)
#define SGL_GLM_LL_SM90 1
#else
#define SGL_GLM_LL_SM90 0
#endif

__device__ __forceinline__ uint4 load_volatile_v4(const void* ptr) {
  uint4 v;
  asm volatile("ld.volatile.global.v4.u32 {%0, %1, %2, %3}, [%4];\n"
               : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w)
               : "l"(ptr)
               : "memory");
  return v;
}

// The dirty sentinel is the raw fp32 -0.0 word; compare bits, every 32-bit word
// (the memory model only guarantees the atomicity of each word).
__device__ __forceinline__ bool is_dirty(const uint4& v) {
  return v.x == kNegZeroF32 || v.y == kNegZeroF32 || v.z == kNegZeroF32 || v.w == kNegZeroF32;
}

// bf16 -0.0 would make a payload word look dirty; send +0.0 instead (as FlashInfer).
__device__ __forceinline__ void sanitize(Pack& v) {
#pragma unroll
  for (int i = 0; i < kElts; ++i) {
    if (__bfloat16_as_ushort(v.e[i]) == kNegZeroBF16) {
      v.e[i] = __float2bfloat16(0.f);
    }
  }
}

__device__ __forceinline__ uint32_t load_relaxed_u32(const uint32_t* ptr) {
  uint32_t v;
  asm volatile("ld.relaxed.gpu.global.u32 %0, [%1];" : "=r"(v) : "l"(ptr) : "memory");
  return v;
}

__device__ __forceinline__ uint32_t load_acquire_u32(const uint32_t* ptr) {
  uint32_t v;
  asm volatile("ld.acquire.gpu.global.u32 %0, [%1];" : "=r"(v) : "l"(ptr) : "memory");
  return v;
}

__device__ __forceinline__ unsigned long long global_timer_ns() {
  unsigned long long t;
  asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t)::"memory");
  return t;
}

// The bound of one wait: the clock starts at the first timer read (the
// kPollTimerStride-th poll), so a wait that succeeds early never reads it.
struct Deadline {
  unsigned long long start = 0;
  uint32_t polls = 0;
  __device__ __forceinline__ bool expired(unsigned long long limit_ns) {
    if ((++polls & (kPollTimerStride - 1)) != 0) return false;
    const unsigned long long now = global_timer_ns();
    if (start == 0) {
      start = now;
      return false;
    }
    return now - start > limit_ns;
  }
};

// A wait on a peer timed out: the host-mapped error record, then trap. The
// record survives the lost CUDA context; whichever timed-out thread writes last
// wins (every one names a real late peer).
static __device__ __noinline__ void fail_stop(const uint32_t* err, uint32_t kind, uint32_t site, int rank, int peer,
                                       int row, uint32_t buffer_idx) {
  const uint32_t meta = static_cast<uint32_t>(rank & 0xFF) | (static_cast<uint32_t>(peer & 0xFF) << 8) |
                        (static_cast<uint32_t>(row & 0xFFFF) << 16);
  asm volatile("st.relaxed.sys.global.v4.u32 [%0], {%1, %2, %3, %4};" ::"l"(err), "r"(kErrMagic), "r"(kind),
               "r"(site), "r"(meta)
               : "memory");
  asm volatile("st.relaxed.sys.global.u32 [%0], %1;" ::"l"(err + 4), "r"(buffer_idx) : "memory");
  asm volatile("fence.sc.sys;" ::: "memory");
  __trap();
}

// Poll the remote slots of this thread's vector until every word has landed,
// then reduce in rank order (fp32, starting from 0) and round to bf16. Same
// loads, same validity test and same arithmetic as FlashInfer's
// waitOneshotRemoteRanks + reduceOneshotDeterministic. A thread with a non-null
// `counter` also reads the arrival counter (relaxed) on every poll iteration.
template <int kWorldSize, int kLocalRank>
__device__ __forceinline__ Pack poll_and_reduce(
    const Pack& local,
    const __nv_bfloat16* stage_local,
    int token,
    int token_dim,
    int packed_idx,
    const uint32_t* counter,
    uint32_t* arrivals,
    const Params& p,
    uint32_t buffer_idx) {
  Pack remote[kWorldSize];
  const __nv_bfloat16* base = stage_local + static_cast<int64_t>(token) * token_dim * kWorldSize + packed_idx * kElts;
  Deadline deadline;
  while (true) {
    if (counter != nullptr) {
      *arrivals = load_relaxed_u32(counter);
    }
    bool valid = true;
    int late = -1;
#pragma unroll
    for (int r = 0; r < kWorldSize; ++r) {
      if (r != kLocalRank) {
        remote[r].u4 = load_volatile_v4(base + r * token_dim);
        const bool dirty = is_dirty(remote[r].u4);
        late = (dirty && late < 0) ? r : late;
        valid &= !dirty;
      }
    }
    if (valid) break;
    if (deadline.expired(p.deadline_ns)) {
      fail_stop(p.err, kErrOneshotPoll, p.site, p.rank, late, token, buffer_idx);
    }
  }
  float acc[kElts];
#pragma unroll
  for (int i = 0; i < kElts; ++i)
    acc[i] = 0.f;
#pragma unroll
  for (int r = 0; r < kWorldSize; ++r) {
    const Pack& v = (r == kLocalRank) ? local : remote[r];
#pragma unroll
    for (int i = 0; i < kElts; ++i)
      acc[i] += __bfloat162float(v.e[i]);
  }
  Pack out;
#pragma unroll
  for (int i = 0; i < kElts; ++i)
    out.e[i] = __float2bfloat16(acc[i]);
  return out;
}

// FlashInfer's blockReduceSumFull adds the warp sums with a second 32-lane xor
// butterfly over {w0, ..., w(n-1), 0, ...}; every lane ends with lane 0's value.
// For n <= kMaxWarps the levels with mask >= kMaxWarps only add +0 to every
// lane, which changes no bit here (the sums are FADD results under FTZ: never
// -0, never denormal, NaN already canonical), so lane 0's value is this tree
// over kMaxWarps leaves.
template <int kMaxWarps>
__device__ __forceinline__ float block_sum(const float* warp_sums, int num_warps) {
  float t[kMaxWarps];
#pragma unroll
  for (int w = 0; w < kMaxWarps; ++w)
    t[w] = w < num_warps ? warp_sums[w] : 0.f;
#pragma unroll
  for (int m = kMaxWarps / 2; m >= 1; m >>= 1) {
#pragma unroll
    for (int l = 0; l < m; ++l)
      t[l] = t[l] + t[l + m];
  }
  return t[0];
}

// Explicit IEEE (non-ftz) fp32 ops for the finalize, whatever the module's
// fast-math flags: moe_finalize_fuse_shared is built without them.
__device__ __forceinline__ float fma_rn_ieee(float a, float b, float c) {
  float d;
  asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(d) : "f"(a), "f"(b), "f"(c));
  return d;
}

__device__ __forceinline__ float add_rn_ieee(float a, float b) {
  float d;
  asm("add.rn.f32 %0, %1, %2;" : "=f"(d) : "f"(a), "f"(b));
  return d;
}

__device__ __forceinline__ float load_weight(const float* ptr) {
  return *ptr;
}

__device__ __forceinline__ float load_weight(const __nv_bfloat16* ptr) {
  return __bfloat162float(*ptr);
}

// This rank's finalized 8 elements of `token`: sum over the routed slots k (in
// order, -1 skipped) of fma(w_k, row_k, acc) from 0, plus the shared-expert row,
// one RN rounding to bf16 (moe_finalize_fuse_shared). Every load is issued
// before the first FMA; with kIdentityRows slot k of token t is FC2 row t * kTopK + k.
template <int kTopK, typename WeightT, bool kIdentityRows>
__device__ __forceinline__ Pack finalize_rows(const FinalizeExt& p, int token, int packed_idx, int64_t thread_offset) {
  const int64_t slot0 = static_cast<int64_t>(token) * kTopK;
  int32_t idx[kTopK];
#pragma unroll
  for (int k = 0; k < kTopK; ++k) {
    idx[k] = kIdentityRows ? static_cast<int32_t>(slot0 + k) : p.permuted_idx[slot0 + k];
  }
  const WeightT* weights = static_cast<const WeightT*>(p.expert_weights) + slot0;
  float w[kTopK];
#pragma unroll
  for (int k = 0; k < kTopK; ++k) {
    w[k] = load_weight(weights + k);
  }
  const bool has_shared = p.shared != nullptr;
  Pack shared_row;
  shared_row.u4 = has_shared ? *reinterpret_cast<const uint4*>(p.shared + thread_offset) : make_uint4(0u, 0u, 0u, 0u);
  Pack rows[kTopK];
#pragma unroll
  for (int k = 0; k < kTopK; ++k) {
    rows[k].u4 = idx[k] == -1 ? make_uint4(0u, 0u, 0u, 0u)
                              : *reinterpret_cast<const uint4*>(
                                    p.gemm2 + static_cast<int64_t>(idx[k]) * p.gemm2_stride + packed_idx * kElts);
  }
  float acc[kElts];
#pragma unroll
  for (int i = 0; i < kElts; ++i)
    acc[i] = 0.f;
#pragma unroll
  for (int k = 0; k < kTopK; ++k) {
    if (idx[k] == -1) continue;
#pragma unroll
    for (int i = 0; i < kElts; ++i)
      acc[i] = fma_rn_ieee(w[k], __bfloat162float(rows[k].e[i]), acc[i]);
  }
  Pack out;
#pragma unroll
  for (int i = 0; i < kElts; ++i) {
    float a = acc[i];
    if (has_shared) a = add_rn_ieee(a, __bfloat162float(shared_row.e[i]));
    out.e[i] = __float2bfloat16(a);
  }
  return out;
}

// FlashInfer v0.7.1 trtllm_allreduce_fusion.cuh fp32_pair_to_e2m1.
__device__ __forceinline__ uint8_t fp32_pair_to_e2m1(float2 pair) {
  uint32_t val32;
  asm volatile(
      "{\n"
      ".reg .b8 byte0;\n"
      "cvt.rn.satfinite.e2m1x2.f32   byte0, %2, %1;\n"
      "mov.b32 %0, {byte0, 0, 0, 0};\n"
      "}"
      : "=r"(val32)
      : "f"(pair.x), "f"(pair.y));
  return static_cast<uint8_t>(val32 & 0xFF);
}

// FlashInfer v0.7.1 MNNVL quant_nvfp4 (cvt_warp_fp16_to_fp4 for bf16, E4M3 block
// scales, linear SF layout): lanes 2j and 2j+1 hold one 16-element block, the
// even one stores its scale. Same source and flags, same codes and scales.
__device__ __forceinline__ void
store_nvfp4(const Pack& v, float sf_scale, const Fp4Ext& e, int token, int token_dim, int packed_idx) {
  constexpr float kRcp6 = 1.0f / 6.0f;
  const __nv_bfloat162* v2 = reinterpret_cast<const __nv_bfloat162*>(v.e);
  __nv_bfloat162 local_max = __habs2(v2[0]);
#pragma unroll
  for (int i = 1; i < kElts / 2; ++i)
    local_max = __hmax2(local_max, __habs2(v2[i]));
  local_max = __hmax2(__shfl_xor_sync(0xffffffffu, local_max, 1), local_max);
  const __nv_bfloat16 block_max = (local_max.x > local_max.y) ? local_max.x : local_max.y;
  const float vec_max = float(block_max);
  const __nv_fp8_e4m3 sf = __nv_fp8_e4m3(sf_scale * (vec_max * kRcp6));
  const float quantized_sf = static_cast<float>(sf);
  const float out_scale = quantized_sf != 0 ? sf_scale / quantized_sf : 0.0f;
  if (threadIdx.x % 2 == 0) {
    e.sf_out[static_cast<int64_t>(token) * (token_dim / 16) + packed_idx / 2] = sf.__x;
  }
  uint32_t codes = 0;
#pragma unroll
  for (int i = 0; i < kElts / 2; ++i) {
    float2 f = __bfloat1622float2(v2[i]);
    f.x *= out_scale;
    f.y *= out_scale;
    codes |= static_cast<uint32_t>(fp32_pair_to_e2m1(f)) << (i * 8);
  }
  e.out[static_cast<int64_t>(token) * (token_dim / kElts) + packed_idx] = codes;
}

// One CTA per (token, cluster rank); the cluster of `num_blocks` CTAs covers
// the row exactly, one 16-byte vector per thread, full warps only. kTopK > 0:
// finalize prologue (see the file comment); kFp4Out: NVFP4 epilogue. The plain
// hop keeps the 1024-thread bound it was measured with; the others bound the
// block by kMaxWarps so the FC2 rows stay in registers.
template <
    int kWorldSize,
    bool kTriggerEarly,
    int kMaxWarps,
    bool kUnicast,
    int kTopK = 0,
    typename WeightT = __nv_bfloat16,
    bool kIdentityRows = false,
    bool kFp4Out = false>
__global__ void __launch_bounds__((kTopK == 0 && !kFp4Out) ? 1024 : kMaxWarps * kWarpSize)
    oneshot_ar_rmsnorm_kernel(const Params p, const HopExt<kTopK, kFp4Out> e) {
#if SGL_GLM_LL_SM90
  static_assert(kWorldSize >= 2 && kWorldSize <= 8);
  static_assert(kMaxWarps >= 2 && kMaxWarps <= 32 && (kMaxWarps & (kMaxWarps - 1)) == 0);
  static_assert(kTopK >= 0 && kTopK <= 16);
  static_assert(!(kTopK > 0 && kFp4Out), "the finalize hop feeds attention, not the MoE");
  extern __shared__ __align__(16) uint4 s_rows[];  // [2 * blockDim.x]: residual, gamma
  __shared__ __align__(16) float s_warp_sums[kMaxClusterSize * kMaxWarps];
  __shared__ __align__(8) uint64_t s_bar;

  cg::cluster_group cluster = cg::this_cluster();
  const int num_blocks = static_cast<int>(cluster.num_blocks());
  const int block_rank = static_cast<int>(cluster.block_rank());
  const int num_warps = static_cast<int>(blockDim.x) / kWarpSize;
  const int warp_id = static_cast<int>(threadIdx.x) / kWarpSize;
  const int lane_id = static_cast<int>(threadIdx.x) % kWarpSize;
  const int packed_idx = static_cast<int>(cluster.thread_rank());
  const int token = blockIdx.x;
  const int token_dim = p.token_dim;
  const int64_t thread_offset = static_cast<int64_t>(token) * token_dim + packed_idx * kElts;
  uint4* s_residual = s_rows;
  uint4* s_gamma = s_rows + blockDim.x;

  // ---- Before the grid dependency wait: only this CTA's shared memory, the
  // static peer table and the norm weight, none of which the previous grid
  // writes. The transaction barrier expects one arrival (this one) and the warp
  // sums of every CTA of the cluster; the relaxed cluster arrive publishes it.
  const uint32_t bar_addr = static_cast<uint32_t>(__cvta_generic_to_shared(&s_bar));
  if (threadIdx.x == 0) {
    const uint32_t tx_bytes = static_cast<uint32_t>(num_blocks * num_warps * sizeof(float));
    asm volatile("mbarrier.init.shared::cta.b64 [%0], 1;" ::"r"(bar_addr) : "memory");
    asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;" ::"r"(bar_addr), "r"(tx_bytes) : "memory");
    asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
  }
  asm volatile("barrier.cluster.arrive.relaxed;" ::: "memory");
  __pipeline_memcpy_async(&s_gamma[threadIdx.x], p.gamma + packed_idx * kElts, sizeof(uint4));
  __pipeline_commit();
  // The NVFP4 global scale is a weight, like gamma.
  [[maybe_unused]] float fp4_scale = 0.f;
  if constexpr (kFp4Out) {
    fp4_scale = *e.scale;
  }
  __nv_bfloat16* const local_base = p.buffer_ptrs[p.rank];
  // Unicast publish: the peers' buffer bases (static table), next rank first.
  [[maybe_unused]] __nv_bfloat16* peer_base[kUnicast ? kWorldSize - 1 : 1];
  if constexpr (kUnicast) {
#pragma unroll
    for (int i = 0; i < kWorldSize - 1; ++i) {
      int r = p.rank + 1 + i;
      r = r >= kWorldSize ? r - kWorldSize : r;
      peer_base[i] = p.buffer_ptrs[r];
    }
  }

  asm volatile("griddepcontrol.wait;" ::: "memory");
  if constexpr (kTriggerEarly) {
    asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
  }

  // ---- Lamport flags (written by the previous hop on this workspace).
  const uint4 flag0 = reinterpret_cast<const uint4*>(p.flags)[0];
  const uint4 flag1 = reinterpret_cast<const uint4*>(p.flags)[1];
  const uint32_t cur_idx = flag0.x;
  const uint32_t dirty_idx = flag0.y;
  const uint32_t bytes_per_buffer = flag0.z;
  const uint32_t dirty_stages = flag0.w;
  uint32_t* const counter = p.flags + 8;
  const size_t cur_offset = static_cast<size_t>(cur_idx) * bytes_per_buffer;  // one stage
  __nv_bfloat16* const stage_mcast =
      reinterpret_cast<__nv_bfloat16*>(reinterpret_cast<char*>(p.mcast_ptr) + cur_offset);
  const __nv_bfloat16* const stage_local =
      reinterpret_cast<const __nv_bfloat16*>(reinterpret_cast<const char*>(local_base) + cur_offset);

  // ---- Publish this rank's vector to every rank: one multicast store, or one
  // unicast store per peer (this rank reduces its own vector from registers).
  Pack val;
  if constexpr (kTopK == 0) {
    val.u4 = *reinterpret_cast<const uint4*>(p.input + thread_offset);
    __pipeline_memcpy_async(&s_residual[threadIdx.x], p.residual_in + thread_offset, sizeof(uint4));
    __pipeline_commit();
  } else {
    __pipeline_memcpy_async(&s_residual[threadIdx.x], p.residual_in + thread_offset, sizeof(uint4));
    __pipeline_commit();
    val = finalize_rows<kTopK, WeightT, kIdentityRows>(e, token, packed_idx, thread_offset);
  }
  sanitize(val);
  const int64_t slot =
      static_cast<int64_t>(token) * token_dim * kWorldSize + p.rank * token_dim + packed_idx * kElts;
  if constexpr (kUnicast) {
#pragma unroll
    for (int i = 0; i < kWorldSize - 1; ++i) {
      __nv_bfloat16* const stage_peer =
          reinterpret_cast<__nv_bfloat16*>(reinterpret_cast<char*>(peer_base[i]) + cur_offset);
      *reinterpret_cast<uint4*>(stage_peer + slot) = val.u4;
    }
  } else {
    *reinterpret_cast<uint4*>(stage_mcast + slot) = val.u4;
  }

  // ---- Arrival of this CTA (fast protocol): the flag reads of every thread
  // precede it (CTA barrier, then a release reduction by thread 0).
  {
    const uint32_t barrier_threads = static_cast<uint32_t>(blockDim.x);  // full warps
    if (threadIdx.x < kWarpSize) {
      asm volatile("barrier.cta.sync 1, %0;" ::"r"(barrier_threads) : "memory");
      if (threadIdx.x == 0) {
        asm volatile("red.release.gpu.global.add.u32 [%0], %1;" ::"l"(counter), "r"(1u) : "memory");
      }
    } else {
      asm volatile("barrier.cta.arrive 1, %0;" ::"r"(barrier_threads) : "memory");
    }
  }

  // ---- Clear this rank's copy of the buffer the previous hop used.
  {
    const uint32_t global_cta = blockIdx.x * gridDim.y + blockIdx.y;
    const uint32_t global_tid = global_cta * blockDim.x + threadIdx.x;
    const uint32_t num_threads = gridDim.x * gridDim.y * blockDim.x;
    const uint4 neg_zero = make_uint4(kNegZeroF32, kNegZeroF32, kNegZeroF32, kNegZeroF32);
    const uint32_t bytes_to_clear[4] = {flag1.x, flag1.y, flag1.z, flag1.w};
    for (uint32_t s = 0; s < dirty_stages && s < 4; ++s) {
      const size_t stage_bytes = bytes_per_buffer / dirty_stages;
      uint4* dirty = reinterpret_cast<uint4*>(
          reinterpret_cast<char*>(local_base) + static_cast<size_t>(dirty_idx * dirty_stages + s) * stage_bytes);
      const uint32_t clear_vecs = (bytes_to_clear[s] + sizeof(uint4) - 1) / sizeof(uint4);
      for (uint32_t i = global_tid; i < clear_vecs; i += num_threads) {
        dirty[i] = neg_zero;
      }
    }
  }

  // ---- Poll + rank-order reduction. The thread FlashInfer's waitAndUpdate
  // elects (block (0,0,0), thread 0) tracks the arrival counter.
  const uint32_t num_ctas = gridDim.x * gridDim.y * gridDim.z;
  bool rotation_pending = threadIdx.x == 0 && blockIdx.x == 0 && blockIdx.y == 0 && blockIdx.z == 0;
  uint32_t arrivals = 0;
  const uint32_t* track = rotation_pending ? counter : nullptr;
  Pack acc;
  switch (p.rank) {
#define SGL_GLM_LL_RANK_CASE(R)                                                                           \
  case R:                                                                                                 \
    if constexpr (R < kWorldSize) {                                                                       \
      acc = poll_and_reduce<kWorldSize, R>(                                                               \
          val, stage_local, token, token_dim, packed_idx, track, &arrivals, p, cur_idx);                  \
    }                                                                                                     \
    break;
    SGL_GLM_LL_RANK_CASE(0)
    SGL_GLM_LL_RANK_CASE(1)
    SGL_GLM_LL_RANK_CASE(2)
    SGL_GLM_LL_RANK_CASE(3)
    SGL_GLM_LL_RANK_CASE(4)
    SGL_GLM_LL_RANK_CASE(5)
    SGL_GLM_LL_RANK_CASE(6)
    SGL_GLM_LL_RANK_CASE(7)
#undef SGL_GLM_LL_RANK_CASE
    default:
      __trap();
  }
  if constexpr (!kTriggerEarly) {
    // FlashInfer's trigger point: after the reduction, before the epilogue.
    asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
  }

  // ---- Rotate the flags right after the poll when every CTA has arrived: the
  // fence turns the last relaxed read into an acquire of every CTA's release
  // arrival, so every flag read of this grid precedes the rotation. The next
  // kernel on the workspace reads the flags only after its grid dependency wait.
  const uint4 next_flag0 = make_uint4((cur_idx + 1) % 3, cur_idx, bytes_per_buffer, 1u);
  const uint4 next_flag1 = make_uint4(
      static_cast<uint32_t>(p.num_tokens) * static_cast<uint32_t>(token_dim) * kWorldSize *
          static_cast<uint32_t>(sizeof(__nv_bfloat16)),
      0u,
      0u,
      0u);
  if (rotation_pending) {
    if (arrivals >= num_ctas) {
      asm volatile("fence.acq_rel.gpu;" ::: "memory");
      reinterpret_cast<uint4*>(p.flags)[0] = next_flag0;
      reinterpret_cast<uint4*>(p.flags)[1] = next_flag1;
      *counter = 0;
      rotation_pending = false;
    }
  }

  // ---- Residual add (bf16), as FlashInfer's `packedAccum += residualIn`.
  __pipeline_wait_prior(0);  // this thread's own residual and gamma copies
  Pack res;
  res.u4 = s_residual[threadIdx.x];
#pragma unroll
  for (int i = 0; i < kElts; ++i)
    acc.e[i] = __hadd(acc.e[i], res.e[i]);
  Pack gam;
  gam.u4 = s_gamma[threadIdx.x];

  // ---- Sum of squares: bf16 squares summed in fp32 in element order, then the
  // warp butterfly (FlashInfer's warpReduceSumFull).
  float thread_sum = 0.f;
#pragma unroll
  for (int i = 0; i < kElts; ++i)
    thread_sum += __bfloat162float(__hmul(acc.e[i], acc.e[i]));
  float warp_sum = thread_sum;
#pragma unroll
  for (int mask = 16; mask > 0; mask >>= 1)
    warp_sum += __shfl_xor_sync(0xffffffffu, warp_sum, mask, kWarpSize);

  // Every CTA's transaction barrier is initialized once the cluster barrier
  // phase completes; then lane l sends this warp's sum to CTA l.
  asm volatile("barrier.cluster.wait;" ::: "memory");
  if (lane_id < num_blocks) {
    const uint32_t slot =
        static_cast<uint32_t>(__cvta_generic_to_shared(&s_warp_sums[block_rank * kMaxWarps + warp_id]));
    uint32_t peer_slot;
    uint32_t peer_bar;
    asm volatile("mapa.shared::cluster.u32 %0, %1, %2;" : "=r"(peer_slot) : "r"(slot), "r"(lane_id));
    asm volatile("mapa.shared::cluster.u32 %0, %1, %2;" : "=r"(peer_bar) : "r"(bar_addr), "r"(lane_id));
    asm volatile("st.async.shared::cluster.mbarrier::complete_tx::bytes.b32 [%0], %1, [%2];" ::"r"(peer_slot),
                 "r"(__float_as_uint(warp_sum)),
                 "r"(peer_bar)
                 : "memory");
  }
  uint32_t done = 0;
  while (!done) {
    asm volatile(
        "{\n"
        ".reg .pred p;\n"
        "mbarrier.try_wait.parity.acquire.cluster.shared::cta.b64 p, [%1], 0;\n"
        "selp.u32 %0, 1, 0, p;\n"
        "}\n"
        : "=r"(done)
        : "r"(bar_addr)
        : "memory");
  }
  float full_sum = 0.f;
#pragma unroll
  for (int b = 0; b < kMaxClusterSize; ++b) {
    if (b < num_blocks) full_sum += block_sum<kMaxWarps>(&s_warp_sums[b * kMaxWarps], num_warps);
  }

  // Residual output after the exchange (a store just before it would only
  // delay this thread's sends).
  *reinterpret_cast<uint4*>(p.residual_out + thread_offset) = acc.u4;

  const float rcp_rms = rsqrtf(full_sum / token_dim + p.eps);
#pragma unroll
  for (int i = 0; i < kElts; ++i) {
    acc.e[i] = __float2bfloat16(__bfloat162float(acc.e[i]) * rcp_rms * (p.weight_bias + __bfloat162float(gam.e[i])));
  }
  *reinterpret_cast<uint4*>(p.norm_out + thread_offset) = acc.u4;
  if constexpr (kFp4Out) {
    store_nvfp4(acc, fp4_scale, e, token, token_dim, packed_idx);
  }

  // ---- Late rotation: the count was short after the poll; wait for it as
  // FlashInfer's waitAndUpdate does.
  if (rotation_pending) {
    Deadline deadline;
    while (load_acquire_u32(counter) < num_ctas) {
      if (deadline.expired(p.deadline_ns)) {
        fail_stop(p.err, kErrArrivalWait, p.site, p.rank, p.rank, 0, cur_idx);
      }
    }
    reinterpret_cast<uint4*>(p.flags)[0] = next_flag0;
    reinterpret_cast<uint4*>(p.flags)[1] = next_flag1;
    *counter = 0;
  }
#endif  // SGL_GLM_LL_SM90
}

// Two-shot all-reduce + residual + RMSNorm in one kernel: FlashInfer v0.7.1's
// twoshotAllreduceKernel (WaitForResults = false) followed by rmsNormLamport,
// same buffers, stages and results, with this file's fast protocol and epilogue.
// Grid and block are rmsNormLamport's (FlashInfer's adjustGridConfig with a
// cluster, one 16-byte vector per thread); CTA (t, c) runs both phases for row t:
//   1. scatter: row t goes to rank t % W's SCATTER stage, slot [t / W][rank];
//   2. the owner rank reduces its rows over the W slots (rank order, fp32 from
//      0, bf16, -0 sanitized) and multicasts them to every rank's BROADCAST stage;
//   3. every rank polls its BROADCAST stage for row t, adds the residual in bf16,
//      sums the fp32 squares of the bf16 sum (rmsNormLamport's arithmetic) and
//      writes (w_bias + gamma) * x * rcp_rms.
// CTA (t, c) only waits on CTAs (t, c) of the other ranks, which scatter before
// they wait on anything. The flags rotate as FlashInfer's two-shot leaves them
// (two dirty stages), so any FlashInfer or SGL hop can run next on the workspace.
template <
    int kWorldSize,
    bool kTriggerEarly,
    int kMaxWarps,
    int kTopK = 0,
    typename WeightT = __nv_bfloat16,
    bool kIdentityRows = false,
    bool kFp4Out = false>
__global__ void __launch_bounds__(kMaxWarps * kWarpSize)
    twoshot_ar_rmsnorm_kernel(const Params p, const HopExt<kTopK, kFp4Out> e) {
#if SGL_GLM_LL_SM90
  static_assert(kWorldSize >= 2 && kWorldSize <= 8);
  static_assert(kMaxWarps >= 2 && kMaxWarps <= 32 && (kMaxWarps & (kMaxWarps - 1)) == 0);
  static_assert(!(kTopK > 0 && kFp4Out), "the finalize hop feeds attention, not the MoE");
  constexpr uint32_t kScatter = 0;
  constexpr uint32_t kBroadcast = 1;
  constexpr uint32_t kStages = 2;
  extern __shared__ __align__(16) uint4 s_rows[];  // [2 * blockDim.x]: residual, gamma
  __shared__ __align__(16) float s_warp_sums[kMaxClusterSize * kMaxWarps];
  __shared__ __align__(8) uint64_t s_bar;

  cg::cluster_group cluster = cg::this_cluster();
  const int num_blocks = static_cast<int>(cluster.num_blocks());
  const int block_rank = static_cast<int>(cluster.block_rank());
  const int num_warps = static_cast<int>(blockDim.x) / kWarpSize;
  const int warp_id = static_cast<int>(threadIdx.x) / kWarpSize;
  const int lane_id = static_cast<int>(threadIdx.x) % kWarpSize;
  const int packed_idx = static_cast<int>(cluster.thread_rank());
  const int token = blockIdx.x;
  const int token_dim = p.token_dim;
  const int64_t thread_offset = static_cast<int64_t>(token) * token_dim + packed_idx * kElts;
  const int owner = token % kWorldSize;
  uint4* s_residual = s_rows;
  uint4* s_gamma = s_rows + blockDim.x;

  // ---- Before the grid dependency wait (as the one-shot kernel).
  const uint32_t bar_addr = static_cast<uint32_t>(__cvta_generic_to_shared(&s_bar));
  if (threadIdx.x == 0) {
    const uint32_t tx_bytes = static_cast<uint32_t>(num_blocks * num_warps * sizeof(float));
    asm volatile("mbarrier.init.shared::cta.b64 [%0], 1;" ::"r"(bar_addr) : "memory");
    asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;" ::"r"(bar_addr), "r"(tx_bytes) : "memory");
    asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
  }
  asm volatile("barrier.cluster.arrive.relaxed;" ::: "memory");
  __pipeline_memcpy_async(&s_gamma[threadIdx.x], p.gamma + packed_idx * kElts, sizeof(uint4));
  __pipeline_commit();
  [[maybe_unused]] float fp4_scale = 0.f;
  if constexpr (kFp4Out) {
    fp4_scale = *e.scale;
  }
  __nv_bfloat16* const local_base = p.buffer_ptrs[p.rank];
  __nv_bfloat16* const owner_base = p.buffer_ptrs[owner];

  asm volatile("griddepcontrol.wait;" ::: "memory");
  if constexpr (kTriggerEarly) {
    asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
  }

  const uint4 flag0 = reinterpret_cast<const uint4*>(p.flags)[0];
  const uint4 flag1 = reinterpret_cast<const uint4*>(p.flags)[1];
  const uint32_t cur_idx = flag0.x;
  const uint32_t dirty_idx = flag0.y;
  const uint32_t bytes_per_buffer = flag0.z;
  const uint32_t dirty_stages = flag0.w;
  uint32_t* const counter = p.flags + 8;
  const size_t stage_bytes = bytes_per_buffer / kStages;
  const size_t scatter_offset = static_cast<size_t>(cur_idx * kStages + kScatter) * stage_bytes;
  const size_t broadcast_offset = static_cast<size_t>(cur_idx * kStages + kBroadcast) * stage_bytes;

  // ---- This rank's row t, into the owner's SCATTER stage.
  Pack val;
  if constexpr (kTopK == 0) {
    val.u4 = *reinterpret_cast<const uint4*>(p.input + thread_offset);
    __pipeline_memcpy_async(&s_residual[threadIdx.x], p.residual_in + thread_offset, sizeof(uint4));
    __pipeline_commit();
  } else {
    __pipeline_memcpy_async(&s_residual[threadIdx.x], p.residual_in + thread_offset, sizeof(uint4));
    __pipeline_commit();
    val = finalize_rows<kTopK, WeightT, kIdentityRows>(e, token, packed_idx, thread_offset);
  }
  sanitize(val);
  const int64_t scatter_slot = static_cast<int64_t>(token / kWorldSize) * token_dim * kWorldSize +
                               static_cast<int64_t>(p.rank) * token_dim + packed_idx * kElts;
  *reinterpret_cast<uint4*>(
      reinterpret_cast<__nv_bfloat16*>(reinterpret_cast<char*>(owner_base) + scatter_offset) + scatter_slot) = val.u4;

  // ---- Arrival of this CTA: every flag read precedes it.
  {
    const uint32_t barrier_threads = static_cast<uint32_t>(blockDim.x);
    if (threadIdx.x < kWarpSize) {
      asm volatile("barrier.cta.sync 1, %0;" ::"r"(barrier_threads) : "memory");
      if (threadIdx.x == 0) {
        asm volatile("red.release.gpu.global.add.u32 [%0], %1;" ::"l"(counter), "r"(1u) : "memory");
      }
    } else {
      asm volatile("barrier.cta.arrive 1, %0;" ::"r"(barrier_threads) : "memory");
    }
  }

  // ---- Clear this rank's copy of the buffer the previous hop used.
  {
    const uint32_t global_cta = blockIdx.x * gridDim.y + blockIdx.y;
    const uint32_t global_tid = global_cta * blockDim.x + threadIdx.x;
    const uint32_t num_threads = gridDim.x * gridDim.y * blockDim.x;
    const uint4 neg_zero = make_uint4(kNegZeroF32, kNegZeroF32, kNegZeroF32, kNegZeroF32);
    const uint32_t bytes_to_clear[4] = {flag1.x, flag1.y, flag1.z, flag1.w};
    for (uint32_t s = 0; s < dirty_stages && s < 4; ++s) {
      const size_t dirty_stage_bytes = bytes_per_buffer / dirty_stages;
      uint4* dirty = reinterpret_cast<uint4*>(
          reinterpret_cast<char*>(local_base) + static_cast<size_t>(dirty_idx * dirty_stages + s) * dirty_stage_bytes);
      const uint32_t clear_vecs = (bytes_to_clear[s] + sizeof(uint4) - 1) / sizeof(uint4);
      for (uint32_t i = global_tid; i < clear_vecs; i += num_threads) {
        dirty[i] = neg_zero;
      }
    }
  }

  // ---- The owner reduces row t over the W slots and multicasts it.
  const uint32_t num_ctas = gridDim.x * gridDim.y * gridDim.z;
  bool rotation_pending = threadIdx.x == 0 && blockIdx.x == 0 && blockIdx.y == 0 && blockIdx.z == 0;
  uint32_t arrivals = 0;
  if (owner == p.rank) {
    const __nv_bfloat16* const slots = reinterpret_cast<const __nv_bfloat16*>(
        reinterpret_cast<const char*>(local_base) + scatter_offset) +
        static_cast<int64_t>(token / kWorldSize) * token_dim * kWorldSize + packed_idx * kElts;
    Pack parts[kWorldSize];
    Deadline deadline;
    while (true) {
      bool valid = true;
      int late = -1;
#pragma unroll
      for (int r = 0; r < kWorldSize; ++r) {
        parts[r].u4 = load_volatile_v4(slots + r * token_dim);
        const bool dirty = is_dirty(parts[r].u4);
        late = (dirty && late < 0) ? r : late;
        valid &= !dirty;
      }
      if (valid) break;
      if (deadline.expired(p.deadline_ns)) {
        fail_stop(p.err, kErrScatterPoll, p.site, p.rank, late, token, cur_idx);
      }
    }
    float acc[kElts];
#pragma unroll
    for (int i = 0; i < kElts; ++i)
      acc[i] = 0.f;
#pragma unroll
    for (int r = 0; r < kWorldSize; ++r) {
#pragma unroll
      for (int i = 0; i < kElts; ++i)
        acc[i] += __bfloat162float(parts[r].e[i]);
    }
    Pack reduced;
#pragma unroll
    for (int i = 0; i < kElts; ++i)
      reduced.e[i] = __float2bfloat16(acc[i]);
    sanitize(reduced);
    __nv_bfloat16* const broadcast_mcast =
        reinterpret_cast<__nv_bfloat16*>(reinterpret_cast<char*>(p.mcast_ptr) + broadcast_offset);
    *reinterpret_cast<uint4*>(broadcast_mcast + thread_offset) = reduced.u4;
  }

  // ---- Every rank: row t from the BROADCAST stage. The elected thread tracks
  // the arrival counter meanwhile.
  Pack acc;
  {
    const __nv_bfloat16* const broadcast_local =
        reinterpret_cast<const __nv_bfloat16*>(reinterpret_cast<const char*>(local_base) + broadcast_offset) +
        thread_offset;
    Deadline deadline;
    while (true) {
      if (rotation_pending) {
        arrivals = load_relaxed_u32(counter);
      }
      acc.u4 = load_volatile_v4(broadcast_local);
      if (!is_dirty(acc.u4)) break;
      if (deadline.expired(p.deadline_ns)) {
        fail_stop(p.err, kErrBroadcastPoll, p.site, p.rank, owner, token, cur_idx);
      }
    }
  }
  if constexpr (!kTriggerEarly) {
    asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
  }

  const uint4 next_flag0 = make_uint4((cur_idx + 1) % 3, cur_idx, bytes_per_buffer, kStages);
  const uint32_t row_bytes = static_cast<uint32_t>(token_dim) * static_cast<uint32_t>(sizeof(__nv_bfloat16));
  const uint32_t padded_tokens = (static_cast<uint32_t>(p.num_tokens) + kWorldSize - 1) / kWorldSize * kWorldSize;
  const uint4 next_flag1 = make_uint4(padded_tokens * row_bytes, static_cast<uint32_t>(p.num_tokens) * row_bytes, 0u, 0u);
  if (rotation_pending) {
    if (arrivals >= num_ctas) {
      asm volatile("fence.acq_rel.gpu;" ::: "memory");
      reinterpret_cast<uint4*>(p.flags)[0] = next_flag0;
      reinterpret_cast<uint4*>(p.flags)[1] = next_flag1;
      *counter = 0;
      rotation_pending = false;
    }
  }

  // ---- rmsNormLamport: bf16 residual add, fp32 squares of the bf16 sum.
  __pipeline_wait_prior(0);
  Pack res;
  res.u4 = s_residual[threadIdx.x];
#pragma unroll
  for (int i = 0; i < kElts; ++i)
    acc.e[i] = acc.e[i] + res.e[i];
  Pack gam;
  gam.u4 = s_gamma[threadIdx.x];
  float values[kElts];
  float thread_sum = 0.f;
#pragma unroll
  for (int i = 0; i < kElts; ++i) {
    const float value = __bfloat162float(acc.e[i]);
    values[i] = value;
    thread_sum += value * value;
  }
  float warp_sum = thread_sum;
#pragma unroll
  for (int mask = 16; mask > 0; mask >>= 1)
    warp_sum += __shfl_xor_sync(0xffffffffu, warp_sum, mask, kWarpSize);

  asm volatile("barrier.cluster.wait;" ::: "memory");
  if (lane_id < num_blocks) {
    const uint32_t slot =
        static_cast<uint32_t>(__cvta_generic_to_shared(&s_warp_sums[block_rank * kMaxWarps + warp_id]));
    uint32_t peer_slot;
    uint32_t peer_bar;
    asm volatile("mapa.shared::cluster.u32 %0, %1, %2;" : "=r"(peer_slot) : "r"(slot), "r"(lane_id));
    asm volatile("mapa.shared::cluster.u32 %0, %1, %2;" : "=r"(peer_bar) : "r"(bar_addr), "r"(lane_id));
    asm volatile("st.async.shared::cluster.mbarrier::complete_tx::bytes.b32 [%0], %1, [%2];" ::"r"(peer_slot),
                 "r"(__float_as_uint(warp_sum)),
                 "r"(peer_bar)
                 : "memory");
  }
  uint32_t done = 0;
  while (!done) {
    asm volatile(
        "{\n"
        ".reg .pred p;\n"
        "mbarrier.try_wait.parity.acquire.cluster.shared::cta.b64 p, [%1], 0;\n"
        "selp.u32 %0, 1, 0, p;\n"
        "}\n"
        : "=r"(done)
        : "r"(bar_addr)
        : "memory");
  }
  float full_sum = 0.f;
#pragma unroll
  for (int b = 0; b < kMaxClusterSize; ++b) {
    if (b < num_blocks) full_sum += block_sum<kMaxWarps>(&s_warp_sums[b * kMaxWarps], num_warps);
  }

  *reinterpret_cast<uint4*>(p.residual_out + thread_offset) = acc.u4;

  const float rcp_rms = rsqrtf(full_sum / token_dim + p.eps);
#pragma unroll
  for (int i = 0; i < kElts; ++i) {
    acc.e[i] = __float2bfloat16((p.weight_bias + __bfloat162float(gam.e[i])) * values[i] * rcp_rms);
  }
  *reinterpret_cast<uint4*>(p.norm_out + thread_offset) = acc.u4;
  if constexpr (kFp4Out) {
    store_nvfp4(acc, fp4_scale, e, token, token_dim, packed_idx);
  }

  if (rotation_pending) {
    Deadline deadline;
    while (load_acquire_u32(counter) < num_ctas) {
      if (deadline.expired(p.deadline_ns)) {
        fail_stop(p.err, kErrArrivalWait, p.site, p.rank, p.rank, 0, cur_idx);
      }
    }
    reinterpret_cast<uint4*>(p.flags)[0] = next_flag0;
    reinterpret_cast<uint4*>(p.flags)[1] = next_flag1;
    *counter = 0;
  }
#endif  // SGL_GLM_LL_SM90
}

template <typename Ext>
using HopKernel = void (*)(Params, Ext);

template <typename Ext>
inline void load_kernel(HopKernel<Ext> kernel) {
  using namespace host;
  cudaFuncAttributes attr;
  RuntimeDeviceCheck(::cudaFuncGetAttributes(&attr, kernel));
}

// The kernel instantiation for a block of `num_warps` warps.
template <
    int kWorldSize,
    bool kTriggerEarly,
    bool kUnicast,
    int kTopK = 0,
    typename WeightT = __nv_bfloat16,
    bool kIdentityRows = false,
    bool kFp4Out = false>
inline HopKernel<HopExt<kTopK, kFp4Out>> select_kernel(int num_warps) {
  return num_warps <= 4
             ? &oneshot_ar_rmsnorm_kernel<kWorldSize, kTriggerEarly, 4, kUnicast, kTopK, WeightT, kIdentityRows, kFp4Out>
         : num_warps <= 8
             ? &oneshot_ar_rmsnorm_kernel<kWorldSize, kTriggerEarly, 8, kUnicast, kTopK, WeightT, kIdentityRows, kFp4Out>
         : num_warps <= 16
             ? &oneshot_ar_rmsnorm_kernel<kWorldSize, kTriggerEarly, 16, kUnicast, kTopK, WeightT, kIdentityRows, kFp4Out>
             : &oneshot_ar_rmsnorm_kernel<kWorldSize, kTriggerEarly, 32, kUnicast, kTopK, WeightT, kIdentityRows, kFp4Out>;
}

template <
    int kWorldSize,
    bool kTriggerEarly,
    int kTopK = 0,
    typename WeightT = __nv_bfloat16,
    bool kIdentityRows = false,
    bool kFp4Out = false>
inline HopKernel<HopExt<kTopK, kFp4Out>> select_twoshot(int num_warps) {
  return num_warps <= 4    ? &twoshot_ar_rmsnorm_kernel<kWorldSize, kTriggerEarly, 4, kTopK, WeightT, kIdentityRows, kFp4Out>
         : num_warps <= 8  ? &twoshot_ar_rmsnorm_kernel<kWorldSize, kTriggerEarly, 8, kTopK, WeightT, kIdentityRows, kFp4Out>
         : num_warps <= 16 ? &twoshot_ar_rmsnorm_kernel<kWorldSize, kTriggerEarly, 16, kTopK, WeightT, kIdentityRows, kFp4Out>
                           : &twoshot_ar_rmsnorm_kernel<kWorldSize, kTriggerEarly, 32, kTopK, WeightT, kIdentityRows, kFp4Out>;
}

// One-shot (unicast or multicast publish) or two-shot kernel of a variant.
template <int kWorldSize, bool kTriggerEarly, int kTopK, typename WeightT, bool kIdentityRows, bool kFp4Out>
inline HopKernel<HopExt<kTopK, kFp4Out>> select_hop(int num_warps, int64_t unicast, int64_t two_shot) {
  if (two_shot != 0) {
    return select_twoshot<kWorldSize, kTriggerEarly, kTopK, WeightT, kIdentityRows, kFp4Out>(num_warps);
  }
  return unicast != 0
             ? select_kernel<kWorldSize, kTriggerEarly, true, kTopK, WeightT, kIdentityRows, kFp4Out>(num_warps)
             : select_kernel<kWorldSize, kTriggerEarly, false, kTopK, WeightT, kIdentityRows, kFp4Out>(num_warps);
}

// Loads every instantiation of one variant on the current device (call before
// graph capture), so a load failure shows up in the host-side vote, not in a hop.
template <int kWorldSize, bool kTriggerEarly, int kTopK, typename WeightT, bool kIdentityRows, bool kFp4Out>
inline void load_variant() {
  for (int num_warps : {4, 8, 16, 32}) {
    load_kernel(select_kernel<kWorldSize, kTriggerEarly, false, kTopK, WeightT, kIdentityRows, kFp4Out>(num_warps));
    load_kernel(select_kernel<kWorldSize, kTriggerEarly, true, kTopK, WeightT, kIdentityRows, kFp4Out>(num_warps));
    load_kernel(select_twoshot<kWorldSize, kTriggerEarly, kTopK, WeightT, kIdentityRows, kFp4Out>(num_warps));
  }
}

// The checks every variant shares: the norm and residual tensors, the flags and
// a geometry that covers the row exactly. Returns (num_tokens, hidden).
template <int kWorldSize>
inline std::pair<int64_t, int64_t> verify_hop(
    tvm::ffi::TensorView residual_in,
    tvm::ffi::TensorView gamma,
    tvm::ffi::TensorView norm_out,
    tvm::ffi::TensorView residual_out,
    tvm::ffi::TensorView buffer_flags,
    int64_t mcast_ptr,
    int64_t buffer_ptrs_dev,
    int64_t rank,
    int64_t block_size,
    int64_t cluster_size,
    int64_t two_shot,
    int64_t err_ptr,
    int64_t deadline_ns) {
  using namespace host;
  auto M = SymbolicSize{"num_tokens"};
  auto H = SymbolicSize{"hidden_dim"};
  auto device = SymbolicDevice{};
  device.set_options<kDLCUDA>();
  TensorMatcher({M, H}).with_strides({-1, 1}).with_dtype<bf16_t>().with_device(device).verify(residual_in);
  TensorMatcher({M, H}).with_strides({-1, 1}).with_dtype<bf16_t>().with_device(device).verify(norm_out);
  TensorMatcher({M, H}).with_strides({-1, 1}).with_dtype<bf16_t>().with_device(device).verify(residual_out);
  TensorMatcher({H}).with_strides({1}).with_dtype<bf16_t>().with_device(device).verify(gamma);
  auto F = SymbolicSize{"num_flags"};
  TensorMatcher({F}).with_dtype<uint32_t>().with_device(device).verify(buffer_flags);

  const int64_t num_tokens = M.unwrap();
  const int64_t hidden = H.unwrap();
  CHECK_HOST(F.unwrap() >= 9) << "buffer_flags must hold 9 words";
  CHECK_HOST(num_tokens > 0) << "num_tokens must be positive";
  CHECK_HOST(rank >= 0 && rank < kWorldSize) << "rank " << rank << " out of range";
  CHECK_HOST(block_size >= kWarpSize && block_size <= 1024 && block_size % kWarpSize == 0)
      << "block_size must be whole warps, got " << block_size;
  // The one-shot kernel exchanges flags per cluster; the two-shot kernel also
  // runs rmsNormLamport's single-CTA rows.
  CHECK_HOST(cluster_size >= (two_shot != 0 ? 1 : 2) && cluster_size <= kMaxClusterSize)
      << "cluster_size must be in [" << (two_shot != 0 ? 1 : 2) << ", 8], got " << cluster_size;
  CHECK_HOST(block_size * cluster_size * kElts == hidden)
      << "the cluster must cover the row exactly: " << block_size << " x " << cluster_size << " x 8 != " << hidden;
  CHECK_HOST(mcast_ptr != 0 && buffer_ptrs_dev != 0) << "null Lamport buffer pointers";
  CHECK_HOST(err_ptr != 0 && err_ptr % 16 == 0) << "the error record must be a 16-byte aligned mapped address";
  CHECK_HOST(deadline_ns > 0) << "every cross-GPU wait needs a positive deadline";
  for (const void* ptr : {residual_in.data_ptr(), norm_out.data_ptr(), residual_out.data_ptr(), gamma.data_ptr()}) {
    CHECK_HOST(reinterpret_cast<uintptr_t>(ptr) % 16 == 0) << "tensors must be 16-byte aligned";
  }
  return {num_tokens, hidden};
}

template <typename Ext>
inline void launch_hop(
    HopKernel<Ext> kernel,
    const Params& params,
    const Ext& ext,
    int64_t num_tokens,
    int64_t block_size,
    int64_t cluster_size,
    int64_t max_carveout,
    DLDevice device) {
  using namespace host;
  const cudaStream_t stream = LaunchKernel::resolve_device(device);
  cudaLaunchAttribute attrs[3];
  attrs[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
  attrs[0].val.programmaticStreamSerializationAllowed = 1;
  attrs[1].id = cudaLaunchAttributeClusterDimension;
  attrs[1].val.clusterDim.x = 1;
  attrs[1].val.clusterDim.y = static_cast<unsigned>(cluster_size);
  attrs[1].val.clusterDim.z = 1;
  // Max shared-memory carveout (TRT-LLM edd689ce): the hop's CTAs take ~1-5 KB,
  // and a PDL dependent launched early (fused-A: ~200 KB/CTA) can only start on
  // an SM whose carveout already leaves it room. Per launch, so the host can
  // keep the default where it does not pay (it also shrinks L1 for co-resident
  // dependents).
  attrs[2].id = cudaLaunchAttributePreferredSharedMemoryCarveout;
  attrs[2].val.sharedMemCarveout = static_cast<unsigned>(cudaSharedmemCarveoutMaxShared);
  cudaLaunchConfig_t config{};
  config.gridDim = dim3(static_cast<unsigned>(num_tokens), static_cast<unsigned>(cluster_size), 1);
  config.blockDim = dim3(static_cast<unsigned>(block_size), 1, 1);
  config.dynamicSmemBytes = static_cast<size_t>(2 * block_size * sizeof(uint4));
  config.stream = stream;
  config.attrs = attrs;
  config.numAttrs = max_carveout != 0 ? 3 : 2;
  RuntimeDeviceCheck(::cudaLaunchKernelEx(&config, kernel, params, ext));
}

inline Params hop_params(
    tvm::ffi::TensorView residual_in,
    tvm::ffi::TensorView gamma,
    tvm::ffi::TensorView norm_out,
    tvm::ffi::TensorView residual_out,
    tvm::ffi::TensorView buffer_flags,
    int64_t mcast_ptr,
    int64_t buffer_ptrs_dev,
    int64_t rank,
    int64_t num_tokens,
    int64_t hidden,
    double eps,
    double weight_bias,
    int64_t err_ptr,
    int64_t deadline_ns,
    int64_t site) {
  return Params{
      .norm_out = static_cast<__nv_bfloat16*>(norm_out.data_ptr()),
      .residual_out = static_cast<__nv_bfloat16*>(residual_out.data_ptr()),
      .input = nullptr,
      .residual_in = static_cast<const __nv_bfloat16*>(residual_in.data_ptr()),
      .gamma = static_cast<const __nv_bfloat16*>(gamma.data_ptr()),
      .buffer_ptrs = reinterpret_cast<__nv_bfloat16* const*>(buffer_ptrs_dev),
      .mcast_ptr = reinterpret_cast<__nv_bfloat16*>(mcast_ptr),
      .flags = static_cast<uint32_t*>(buffer_flags.data_ptr()),
      .rank = static_cast<int>(rank),
      .num_tokens = static_cast<int>(num_tokens),
      .token_dim = static_cast<int>(hidden),
      .eps = static_cast<float>(eps),
      .weight_bias = static_cast<float>(weight_bias),
      .err = reinterpret_cast<uint32_t*>(err_ptr),
      .deadline_ns = static_cast<unsigned long long>(deadline_ns),
      .site = static_cast<uint32_t>(site),
  };
}

/// \brief Plain hop: norm_out, residual_out = RMSNorm and value of
///        residual_in + sum over ranks of input. `two_shot` selects the two-shot
///        kernel (FlashInfer's two-shot + rmsNormLamport bits and geometry).
template <int kWorldSize, bool kTriggerEarly>
struct GlmLLMnnvlOneshot {
  static void configure() {
    load_variant<kWorldSize, kTriggerEarly, 0, __nv_bfloat16, false, false>();
  }

  static void
  run(tvm::ffi::TensorView input,
      tvm::ffi::TensorView residual_in,
      tvm::ffi::TensorView gamma,
      tvm::ffi::TensorView norm_out,
      tvm::ffi::TensorView residual_out,
      tvm::ffi::TensorView buffer_flags,
      int64_t mcast_ptr,
      int64_t buffer_ptrs_dev,
      int64_t rank,
      double eps,
      double weight_bias,
      int64_t block_size,
      int64_t cluster_size,
      int64_t unicast,
      int64_t max_carveout,
      int64_t two_shot,
      int64_t err_ptr,
      int64_t deadline_ns,
      int64_t site) {
    using namespace host;
    const auto [num_tokens, hidden] = verify_hop<kWorldSize>(
        residual_in, gamma, norm_out, residual_out, buffer_flags, mcast_ptr, buffer_ptrs_dev, rank, block_size,
        cluster_size, two_shot, err_ptr, deadline_ns);
    auto device = SymbolicDevice{};
    device.set_options<kDLCUDA>();
    TensorMatcher({num_tokens, hidden}).with_strides({-1, 1}).with_dtype<bf16_t>().with_device(device).verify(input);
    CHECK_HOST(device.unwrap().device_id == norm_out.device().device_id) << "input on another device";
    CHECK_HOST(reinterpret_cast<uintptr_t>(input.data_ptr()) % 16 == 0) << "tensors must be 16-byte aligned";
    Params params = hop_params(
        residual_in, gamma, norm_out, residual_out, buffer_flags, mcast_ptr, buffer_ptrs_dev, rank, num_tokens, hidden,
        eps, weight_bias, err_ptr, deadline_ns, site);
    params.input = static_cast<const __nv_bfloat16*>(input.data_ptr());
    const int num_warps = static_cast<int>(block_size / kWarpSize);
    launch_hop(
        select_hop<kWorldSize, kTriggerEarly, 0, __nv_bfloat16, false, false>(num_warps, unicast, two_shot),
        params, NoExt{}, num_tokens, block_size, cluster_size, max_carveout, norm_out.device());
  }
};

/// \brief Finalize hop: as the plain hop, with this rank's input the MoE finalize
///        of its FC2 rows plus the shared-expert output.
/// \tparam kTopK Routed slots per token.
/// \tparam WeightT Routing weight dtype (bf16_t or fp32_t).
/// \tparam kIdentityRows Slot k of token t is FC2 row t * kTopK + k (W1a's layout);
///         the permutation is not read.
template <int kWorldSize, bool kTriggerEarly, int kTopK, typename WeightT, bool kIdentityRows>
struct GlmLLMnnvlFinalizeOneshot {
  static_assert(std::is_same_v<WeightT, bf16_t> || std::is_same_v<WeightT, fp32_t>);

  static void configure() {
    load_variant<kWorldSize, kTriggerEarly, kTopK, WeightT, kIdentityRows, false>();
  }

  /// \param gemm2_out [P, >= hidden] bf16 FC2 rows, unit inner stride.
  /// \param permuted_idx [M, kTopK] int32 (-1 = empty slot); ignored with kIdentityRows.
  /// \param expert_weights [M, kTopK] WeightT, routed scaling folded in.
  /// \param shared_output optional [M, hidden] bf16 added before the reduction.
  static void
  run(tvm::ffi::TensorView gemm2_out,
      std::optional<tvm::ffi::TensorView> permuted_idx,
      tvm::ffi::TensorView expert_weights,
      std::optional<tvm::ffi::TensorView> shared_output,
      tvm::ffi::TensorView residual_in,
      tvm::ffi::TensorView gamma,
      tvm::ffi::TensorView norm_out,
      tvm::ffi::TensorView residual_out,
      tvm::ffi::TensorView buffer_flags,
      int64_t mcast_ptr,
      int64_t buffer_ptrs_dev,
      int64_t rank,
      double eps,
      double weight_bias,
      int64_t block_size,
      int64_t cluster_size,
      int64_t unicast,
      int64_t max_carveout,
      int64_t two_shot,
      int64_t err_ptr,
      int64_t deadline_ns,
      int64_t site) {
    using namespace host;
    const auto [num_tokens, hidden] = verify_hop<kWorldSize>(
        residual_in, gamma, norm_out, residual_out, buffer_flags, mcast_ptr, buffer_ptrs_dev, rank, block_size,
        cluster_size, two_shot, err_ptr, deadline_ns);
    auto P = SymbolicSize{"gemm2_rows"};
    auto S = SymbolicSize{"gemm2_stride"};
    auto device = SymbolicDevice{};
    device.set_options<kDLCUDA>();
    TensorMatcher({P, -1}).with_strides({S, 1}).with_dtype<bf16_t>().with_device(device).verify(gemm2_out);
    TensorMatcher({num_tokens, kTopK})
        .with_strides({kTopK, 1})
        .template with_dtype<WeightT>()
        .template with_device<kDLCUDA>(device)
        .verify(expert_weights);
    CHECK_HOST(gemm2_out.size(1) >= hidden) << "gemm2_out rows are narrower than hidden=" << hidden;
    CHECK_HOST(S.unwrap() % kElts == 0 && reinterpret_cast<uintptr_t>(gemm2_out.data_ptr()) % 16 == 0)
        << "FC2 rows must be 16-byte aligned";
    const int32_t* idx_ptr = nullptr;
    if constexpr (kIdentityRows) {
      CHECK_HOST(P.unwrap() >= num_tokens * kTopK) << "identity rows need " << kTopK << " FC2 rows per token";
    } else {
      CHECK_HOST(permuted_idx.has_value()) << "the permutation is required without identity rows";
      TensorMatcher({num_tokens, kTopK})
          .with_strides({kTopK, 1})
          .with_dtype<int32_t>()
          .with_device(device)
          .verify(permuted_idx.value());
      idx_ptr = static_cast<const int32_t*>(permuted_idx.value().data_ptr());
    }
    const __nv_bfloat16* shared_ptr = nullptr;
    if (shared_output.has_value()) {
      TensorMatcher({num_tokens, hidden})
          .with_strides({-1, 1})
          .with_dtype<bf16_t>()
          .with_device(device)
          .verify(shared_output.value());
      shared_ptr = static_cast<const __nv_bfloat16*>(shared_output.value().data_ptr());
      CHECK_HOST(reinterpret_cast<uintptr_t>(shared_ptr) % 16 == 0) << "tensors must be 16-byte aligned";
    }
    CHECK_HOST(device.unwrap().device_id == norm_out.device().device_id) << "MoE outputs on another device";
    Params params = hop_params(
        residual_in, gamma, norm_out, residual_out, buffer_flags, mcast_ptr, buffer_ptrs_dev, rank, num_tokens, hidden,
        eps, weight_bias, err_ptr, deadline_ns, site);
    const FinalizeExt ext{
        .gemm2 = static_cast<const __nv_bfloat16*>(gemm2_out.data_ptr()),
        .gemm2_stride = S.unwrap(),
        .permuted_idx = idx_ptr,
        .expert_weights = expert_weights.data_ptr(),
        .shared = shared_ptr,
    };
    const int num_warps = static_cast<int>(block_size / kWarpSize);
    launch_hop(
        select_hop<kWorldSize, kTriggerEarly, kTopK, WeightT, kIdentityRows, false>(num_warps, unicast, two_shot),
        params, ext, num_tokens, block_size, cluster_size, max_carveout, norm_out.device());
  }
};

/// \brief NVFP4 hop: the plain hop that also writes the NVFP4 quant of norm_out,
///        as fp4_quantize(norm_out, global_scale, is_sf_swizzled_layout=False)
///        lays it out: fp4_out [M, hidden / 2] uint8, sf_out [M, hidden / 16] e4m3.
template <int kWorldSize, bool kTriggerEarly>
struct GlmLLMnnvlOneshotNvfp4 {
  static void configure() {
    load_variant<kWorldSize, kTriggerEarly, 0, __nv_bfloat16, false, true>();
  }

  static void
  run(tvm::ffi::TensorView input,
      tvm::ffi::TensorView residual_in,
      tvm::ffi::TensorView gamma,
      tvm::ffi::TensorView norm_out,
      tvm::ffi::TensorView residual_out,
      tvm::ffi::TensorView fp4_out,
      tvm::ffi::TensorView sf_out,
      tvm::ffi::TensorView global_scale,
      tvm::ffi::TensorView buffer_flags,
      int64_t mcast_ptr,
      int64_t buffer_ptrs_dev,
      int64_t rank,
      double eps,
      double weight_bias,
      int64_t block_size,
      int64_t cluster_size,
      int64_t unicast,
      int64_t max_carveout,
      int64_t two_shot,
      int64_t err_ptr,
      int64_t deadline_ns,
      int64_t site) {
    using namespace host;
    const auto [num_tokens, hidden] = verify_hop<kWorldSize>(
        residual_in, gamma, norm_out, residual_out, buffer_flags, mcast_ptr, buffer_ptrs_dev, rank, block_size,
        cluster_size, two_shot, err_ptr, deadline_ns);
    auto device = SymbolicDevice{};
    device.set_options<kDLCUDA>();
    TensorMatcher({num_tokens, hidden}).with_strides({-1, 1}).with_dtype<bf16_t>().with_device(device).verify(input);
    CHECK_HOST(hidden % 16 == 0) << "NVFP4 blocks of 16 need hidden % 16 == 0";
    // hidden is type-dependent here (a structured binding of verify_hop<kWorldSize>), so the
    // chain is dependent from with_strides on: nvcc 13.4 requires the template disambiguator.
    TensorMatcher({num_tokens, hidden / 2})
        .with_strides({hidden / 2, 1})
        .template with_dtype<uint8_t>()
        .with_device(device)
        .verify(fp4_out);
    TensorMatcher({num_tokens, hidden / 16})
        .with_strides({hidden / 16, 1})
        .template with_dtype<fp8_e4m3_t>()
        .with_device(device)
        .verify(sf_out);
    TensorMatcher({1}).with_dtype<fp32_t>().with_device(device).verify(global_scale);
    CHECK_HOST(device.unwrap().device_id == norm_out.device().device_id) << "input on another device";
    CHECK_HOST(
        reinterpret_cast<uintptr_t>(input.data_ptr()) % 16 == 0 && reinterpret_cast<uintptr_t>(fp4_out.data_ptr()) % 4 == 0)
        << "input must be 16-byte and fp4_out 4-byte aligned";
    Params params = hop_params(
        residual_in, gamma, norm_out, residual_out, buffer_flags, mcast_ptr, buffer_ptrs_dev, rank, num_tokens, hidden,
        eps, weight_bias, err_ptr, deadline_ns, site);
    params.input = static_cast<const __nv_bfloat16*>(input.data_ptr());
    const Fp4Ext ext{
        .out = static_cast<uint32_t*>(fp4_out.data_ptr()),
        .sf_out = static_cast<uint8_t*>(sf_out.data_ptr()),
        .scale = static_cast<const float*>(global_scale.data_ptr()),
    };
    const int num_warps = static_cast<int>(block_size / kWarpSize);
    launch_hop(
        select_hop<kWorldSize, kTriggerEarly, 0, __nv_bfloat16, false, true>(num_warps, unicast, two_shot),
        params, ext, num_tokens, block_size, cluster_size, max_carveout, norm_out.device());
  }
};

}  // namespace glm_ll_mnnvl
}  // namespace sglang
