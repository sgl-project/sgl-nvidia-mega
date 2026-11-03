// Derived from SGLang (https://github.com/sgl-project/sglang), Apache License 2.0; see ../include/LICENSE.sglang.
// SPDX-License-Identifier: Apache-2.0
// Portions Copyright 2023-2024 SGLang Team (Apache-2.0)
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// Source: SGLang python/sglang/kernels/jit/csrc/distributed/custom_all_reduce.cuh (one-shot push all-reduce and its
// phase protocol) at revision a55ab7d.
// Modified by NVIDIA: one kernel performs, for a DFlash draft layer boundary, the one-shot push all-reduce of the
// row-parallel bf16 partials, the output side of the grouped two-tap block convolution, the residual add and the
// RMSNorm of the next sublayer.
//
// Layout: M = R * kBlock rows (R requests, kBlock tokens per request block), H = kHidden columns.
//   partial:  [M, H] bf16, this rank's row-parallel partial
//   coef:     the convolution coefficients of the output side: element (row, tap, group) at
//             coef[row * coef_stride + tap * kGroups + group] (bf16)
//   base:     [2, H] bf16, the output side's base kernel (tap 0, tap 1)
//   residual: [M, H] bf16, updated in place with the new residual
//   weight:   [H] bf16 norm weight; out: [M, H] bf16 normalized rows
//
// Arithmetic, per element (as the stock kernels it replaces compute it):
//   x    = bf16(sum of the ranks' partials in rank order, fp32 accumulation; a 32-bit word of two +0.0 is sent as
//          (+0.0, -0.0) as the push protocol requires)
//   y    = bf16(fma(base0 + coef0, x[t], (base1 + coef1) * x[t - 1]))   (x[t - 1] = 0 for the first token of a block)
//   h    = float(y) + float(residual);  residual = bf16(h)
//   out  = bf16(h * rsqrt(sum(h * h) / H + eps) * weight)
// The sum of squares is accumulated per thread, then across a warp (butterfly) and across the cluster in CTA order:
// a different summation order than the stock norm kernel (reduction order only).
//
// Work split: one thread-block cluster of kCluster CTAs per request block (a cluster loops over several request blocks
// when the grid is capped by the push plane's block counters); CTA c handles columns
// [c * H / kCluster, (c + 1) * H / kCluster) of the block's kBlock rows, so a token's predecessor row is in the same
// CTA. The CTA that pushes a vector is the CTA that consumes it on every rank. Row sums of squares are exchanged
// through distributed shared memory after a cluster barrier (arrive.release / wait.acquire).
#include <sgl_kernel/tensor.h>
#include <sgl_kernel/utils.h>

#include <sgl_kernel/type.cuh>
#include <sgl_kernel/utils.cuh>

#include <sgl_kernel/distributed/communicator.cuh>

#include <tvm/ffi/container/tensor.h>

#include <algorithm>
#include <cstdint>

namespace sglang {

namespace muse_vr_draft_arn {

constexpr uint32_t kBlock = 8;     // tokens per request block (one warp per token row)
constexpr uint32_t kThreads = 256;  // kBlock warps
constexpr uint32_t kMaxPolls = 1u << 28;

__device__ __noinline__ void poll_bound_exceeded() {
  __trap();
}

template <uint32_t kWorld>
struct Params {
  device::distributed::PushWorkSpace<kWorld> ws;
  uint32_t num_counters;
  uint32_t rank;
  uint32_t num_requests;
  const uint4* partial;      // [M, H] bf16 as 16-byte vectors
  const uint4* partial_all;  // loopback test: [kWorld, M, H] bf16
  const uint16_t* coef;
  int64_t coef_stride;
  const uint16_t* base;
  uint16_t* residual;
  const uint16_t* weight;
  uint16_t* out;
  float eps;
};

SGL_DEVICE float bf(uint32_t bits16) {
  return __uint_as_float(bits16 << 16);
}

SGL_DEVICE uint32_t to_bf16_bits(float v) {
  return static_cast<uint32_t>(__bfloat16_as_ushort(__float2bfloat16_rn(v)));
}

SGL_DEVICE void clear_pos_zero(uint4& v) {
  if (v.x == 0u) v.x = 0x8000u;
  if (v.y == 0u) v.y = 0x8000u;
  if (v.z == 0u) v.z = 0x8000u;
  if (v.w == 0u) v.w = 0x8000u;
}

SGL_DEVICE bool has_zero(const uint4& v) {
  return v.x == 0u || v.y == 0u || v.z == 0u || v.w == 0u;
}

SGL_DEVICE float rsqrt_approx(float x) {
  float y;
  asm("rsqrt.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x));
  return y;
}

SGL_DEVICE uint32_t cluster_ctarank() {
  uint32_t r;
  asm volatile("mov.u32 %0, %%cluster_ctarank;" : "=r"(r));
  return r;
}

SGL_DEVICE void cluster_arrive_relaxed() {
  asm volatile("barrier.cluster.arrive.relaxed.aligned;" ::: "memory");
}
SGL_DEVICE void cluster_arrive_release() {
  asm volatile("barrier.cluster.arrive.release.aligned;" ::: "memory");
}
SGL_DEVICE void cluster_wait_acquire() {
  asm volatile("barrier.cluster.wait.acquire.aligned;" ::: "memory");
}

SGL_DEVICE void st_cluster_f32(const float* local_addr, uint32_t cta, float v) {
  const uint32_t a = static_cast<uint32_t>(__cvta_generic_to_shared(local_addr));
  uint32_t remote;
  asm volatile("mapa.shared::cluster.u32 %0, %1, %2;" : "=r"(remote) : "r"(a), "r"(cta));
  asm volatile("st.shared::cluster.f32 [%0], %1;" ::"r"(remote), "f"(v) : "memory");
}

template <uint32_t kHidden, uint32_t kWorld, uint32_t kCluster, bool kUsePDL, bool kLoopback>
__global__ __launch_bounds__(kThreads) void draft_ar_norm_kernel(const __grid_constant__ Params<kWorld> params) {
  constexpr uint32_t kSlice = kHidden / kCluster;  // columns per CTA
  constexpr uint32_t kVecsPerThread = kSlice / (32 * 8);
  constexpr uint32_t kGroups = kHidden / 16;
  static_assert(kSlice % 256 == 0 && kVecsPerThread >= 1, "slice must be whole warp-wide vector rows");
  constexpr uint32_t kRowVecs = kHidden / 8;

  __shared__ uint16_t xs[kBlock][kSlice];
  __shared__ float part[kCluster][kBlock];

  const uint32_t c = cluster_ctarank();
  const uint32_t t = threadIdx.x / 32;
  const uint32_t lane = threadIdx.x % 32;
  const uint32_t col0 = c * kSlice + lane * 8;
  const size_t rows_total = static_cast<size_t>(params.num_requests) * kBlock;

  cluster_arrive_relaxed();
  // static parameters (never written by any kernel) before the grid dependency wait
  uint4 w4[kVecsPerThread], b0[kVecsPerThread], b1[kVecsPerThread];
#pragma unroll
  for (uint32_t v = 0; v < kVecsPerThread; ++v) {
    const uint32_t col = col0 + v * 256;
    w4[v] = *reinterpret_cast<const uint4*>(params.weight + col);
    b0[v] = *reinterpret_cast<const uint4*>(params.base + col);
    b1[v] = *reinterpret_cast<const uint4*>(params.base + kHidden + col);
  }
  device::PDLWaitPrimary<kUsePDL>();

  uint32_t phase = 0;
  if constexpr (!kLoopback) {
    phase = params.ws.counter[blockIdx.x].get() & 1;
    if (blockIdx.x == 0) {
      for (uint32_t i = gridDim.x + threadIdx.x; i < params.num_counters; i += kThreads) {
        params.ws.counter[i].set(phase ^ 1);
      }
    }
  }

  // request blocks of this cluster (several when the grid is capped by the plane's block counters)
  for (uint32_t req = blockIdx.x / kCluster; req < params.num_requests; req += gridDim.x / kCluster) {
  const uint32_t row = req * kBlock + t;
  // later request blocks: this CTA's reads of the previous `part` are ordered before the peers' next stores into it
  if (req != blockIdx.x / kCluster) cluster_arrive_release();
  // this rank's partial (and, in the test path, every rank's), residual and coefficients, issued together
  uint4 vec[kVecsPerThread][kWorld];  // peers' vectors (every rank's in the test path)
  uint4 own[kVecsPerThread];
  uint4 res4[kVecsPerThread];
  float d0[kVecsPerThread], d1[kVecsPerThread];
  const uint16_t* coef_row = params.coef + static_cast<int64_t>(row) * params.coef_stride;
#pragma unroll
  for (uint32_t v = 0; v < kVecsPerThread; ++v) {
    const uint32_t col = col0 + v * 256;
    const uint32_t vid = row * kRowVecs + col / 8;
    if constexpr (kLoopback) {
#pragma unroll
      for (uint32_t i = 0; i < kWorld; ++i) vec[v][i] = params.partial_all[i * rows_total * kRowVecs + vid];
    } else {
      own[v] = params.partial[vid];
    }
    res4[v] = *reinterpret_cast<const uint4*>(params.residual + static_cast<size_t>(row) * kHidden + col);
    d0[v] = bf(coef_row[col / 16]);
    d1[v] = bf(coef_row[kGroups + col / 16]);
  }
  if constexpr (kLoopback) {
#pragma unroll
    for (uint32_t v = 0; v < kVecsPerThread; ++v)
#pragma unroll
      for (uint32_t i = 0; i < kWorld; ++i) clear_pos_zero(vec[v][i]);
  } else {
#pragma unroll
    for (uint32_t v = 0; v < kVecsPerThread; ++v) {
      clear_pos_zero(own[v]);
      const uint32_t vid = row * kRowVecs + (col0 + v * 256) / 8;
#pragma unroll
      for (uint32_t i = 0; i < kWorld; ++i) {
        if (i == params.rank) continue;
        uint8_t* slot = params.ws.workspaces[i] + (params.rank + phase * kWorld) * params.ws.slot_bytes;
        device::ptx::st_relaxed_16B(own[v], slot, vid);
      }
    }
#pragma unroll
    for (uint32_t v = 0; v < kVecsPerThread; ++v) {
      const uint32_t vid = row * kRowVecs + (col0 + v * 256) / 8;
#pragma unroll
      for (uint32_t i = 0; i < kWorld; ++i) {
        if (i == params.rank) continue;
        uint8_t* slot = params.ws.workspaces[params.rank] + (i + phase * kWorld) * params.ws.slot_bytes;
        uint32_t polls = 0;
        do {
          device::ptx::ld_relaxed_16B(vec[v][i], slot, vid);
          if (!has_zero(vec[v][i])) break;
          if (++polls == kMaxPolls) [[unlikely]] poll_bound_exceeded();
        } while (true);
        const uint4 zero = make_uint4(0u, 0u, 0u, 0u);
        device::ptx::st_global_16B(zero, slot, vid);
      }
    }
  }

  // fp32 sum in rank order, one bf16 rounding
  float x[kVecsPerThread][8];
#pragma unroll
  for (uint32_t v = 0; v < kVecsPerThread; ++v) {
#pragma unroll
    for (uint32_t w = 0; w < 4; ++w) {
      float lo = 0.0f, hi = 0.0f;
#pragma unroll
      for (uint32_t i = 0; i < kWorld; ++i) {
        const uint32_t word = (!kLoopback && i == params.rank) ? (&own[v].x)[w] : (&vec[v][i].x)[w];
        const float a = bf(word & 0xffffu), b = bf(word >> 16);
        lo = i == 0 ? a : __fadd_rn(lo, a);
        hi = i == 0 ? b : __fadd_rn(hi, b);
      }
      const uint32_t lo16 = to_bf16_bits(lo), hi16 = to_bf16_bits(hi);
      x[v][2 * w] = bf(lo16);
      x[v][2 * w + 1] = bf(hi16);
      *reinterpret_cast<uint32_t*>(&xs[t][lane * 8 + v * 256 + 2 * w]) = lo16 | (hi16 << 16);
    }
  }
  __syncthreads();

  // output-side convolution, residual add, sum of squares
  float h[kVecsPerThread][8];
  float ss = 0.0f;
#pragma unroll
  for (uint32_t v = 0; v < kVecsPerThread; ++v) {
    const uint32_t col = col0 + v * 256;
    uint4 prev4 = make_uint4(0u, 0u, 0u, 0u);
    if (t >= 1) prev4 = *reinterpret_cast<const uint4*>(&xs[t - 1][lane * 8 + v * 256]);
    uint32_t res_out[4];
#pragma unroll
    for (uint32_t e = 0; e < 8; ++e) {
      const uint32_t w = e / 2;
      const uint32_t sh = (e % 2) * 16;
      const float c0 = __fadd_rn(bf(((&b0[v].x)[w] >> sh) & 0xffffu), d0[v]);
      const float c1 = __fadd_rn(bf(((&b1[v].x)[w] >> sh) & 0xffffu), d1[v]);
      const float prev = t >= 1 ? bf(((&prev4.x)[w] >> sh) & 0xffffu) : 0.0f;
      const float yv = bf(to_bf16_bits(__fmaf_rn(c0, x[v][e], __fmul_rn(c1, prev))));
      const float hv = __fadd_rn(yv, bf(((&res4[v].x)[w] >> sh) & 0xffffu));
      h[v][e] = hv;
      ss = __fmaf_rn(hv, hv, ss);
      const uint32_t r16 = to_bf16_bits(hv);
      if (e % 2 == 0)
        res_out[w] = r16;
      else
        res_out[w] |= r16 << 16;
    }
    *reinterpret_cast<uint4*>(params.residual + static_cast<size_t>(row) * kHidden + col) =
        make_uint4(res_out[0], res_out[1], res_out[2], res_out[3]);
  }
#pragma unroll
  for (uint32_t m = 16; m >= 1; m /= 2) ss = __fadd_rn(ss, __shfl_xor_sync(0xffffffffu, ss, m));

  cluster_wait_acquire();  // every CTA of the cluster is running (pairs with the arrive at entry)
  if (lane < kCluster) st_cluster_f32(&part[c][t], lane, ss);
  cluster_arrive_release();
  cluster_wait_acquire();
  float total = part[0][t];
#pragma unroll
  for (uint32_t src = 1; src < kCluster; ++src) total = __fadd_rn(total, part[src][t]);
  const float rstd = rsqrt_approx(__fadd_rn(__fdiv_rn(total, static_cast<float>(kHidden)), params.eps));

#pragma unroll
  for (uint32_t v = 0; v < kVecsPerThread; ++v) {
    const uint32_t col = col0 + v * 256;
    uint32_t o[4];
#pragma unroll
    for (uint32_t e = 0; e < 8; ++e) {
      const uint32_t w = e / 2;
      const uint32_t sh = (e % 2) * 16;
      const float wv = __fadd_rn(bf(((&w4[v].x)[w] >> sh) & 0xffffu), 0.0f);
      const uint32_t o16 = to_bf16_bits(__fmul_rn(__fmul_rn(h[v][e], rstd), wv));
      if (e % 2 == 0)
        o[w] = o16;
      else
        o[w] |= o16 << 16;
    }
    *reinterpret_cast<uint4*>(params.out + static_cast<size_t>(row) * kHidden + col) = make_uint4(o[0], o[1], o[2], o[3]);
  }
  __syncthreads();  // `xs` is rewritten by the next request block
  }
  device::PDLTriggerSecondary<kUsePDL>();
  if constexpr (!kLoopback) {
    __syncthreads();
    if (threadIdx.x == 0) params.ws.counter[blockIdx.x].set(phase ^ 1);
  }
}

}  // namespace muse_vr_draft_arn

using host::distributed::CommunicatorRef;

template <uint32_t kHidden, uint32_t kWorld, uint32_t kCluster, bool kUsePDL>
struct MuseVrDraftArNorm {
  using TensorView = tvm::ffi::TensorView;
  using P = muse_vr_draft_arn::Params<kWorld>;
  static constexpr uint32_t kBlock = muse_vr_draft_arn::kBlock;

  template <typename K>
  static void allow_cluster(K kernel) {
    if constexpr (kCluster > 8) {
      static const bool done = [kernel] {
        host::RuntimeDeviceCheck(cudaFuncSetAttribute(kernel, cudaFuncAttributeNonPortableClusterSizeAllowed, 1));
        return true;
      }();
      (void)done;
    }
  }

  static void fill(P& p, TensorView coef, int64_t coef_offset, int64_t coef_stride, TensorView base, TensorView residual,
                   TensorView weight, TensorView out, double eps, host::SymbolicDevice& device, uint32_t& rows) {
    using namespace host;
    auto M = SymbolicSize{"num_rows"};
    device.set_options<kDLCUDA>();
    TensorMatcher({M, int64_t(kHidden)}).with_dtype<bf16_t>().with_device(device).verify(residual);
    TensorMatcher({M, int64_t(kHidden)}).with_dtype<bf16_t>().with_device(device).verify(out);
    TensorMatcher({2, int64_t(kHidden)}).with_dtype<bf16_t>().with_device(device).verify(base);
    TensorMatcher({int64_t(kHidden)}).with_dtype<bf16_t>().with_device(device).verify(weight);
    rows = static_cast<uint32_t>(M.unwrap());
    RuntimeCheck(rows % kBlock == 0, "rows must be whole request blocks");
    RuntimeCheck(coef.dtype().code == kDLBfloat && coef.dtype().bits == 16, "coefficients must be bf16");
    p.num_requests = rows / kBlock;
    p.coef = static_cast<const uint16_t*>(coef.data_ptr()) + coef_offset;
    p.coef_stride = coef_stride;
    p.base = static_cast<const uint16_t*>(base.data_ptr());
    p.residual = static_cast<uint16_t*>(residual.data_ptr());
    p.weight = static_cast<const uint16_t*>(weight.data_ptr());
    p.out = static_cast<uint16_t*>(out.data_ptr());
    p.eps = static_cast<float>(eps);
  }

  static void
  run(CommunicatorRef ref, TensorView partial, TensorView coef, int64_t coef_offset, int64_t coef_stride, TensorView base,
      TensorView residual, TensorView weight, TensorView out, double eps) {
    using namespace host;
    const auto& push = ref.get()->get_push_obj();
    auto device = SymbolicDevice{};
    P p{};
    uint32_t rows = 0;
    fill(p, coef, coef_offset, coef_stride, base, residual, weight, out, eps, device, rows);
    auto M = SymbolicSize{"num_rows"};
    M.set_value(rows);
    TensorMatcher({M, int64_t(kHidden)}).with_dtype<bf16_t>().with_device(device).verify(partial);
    if (rows == 0) return;
    RuntimeCheck(push.world_size == kWorld, "the kernel is compiled for TP", kWorld);
    const int64_t nbytes = int64_t(rows) * kHidden * 2;
    RuntimeCheck(nbytes <= int64_t(push.slot_bytes), "rows exceed the push slot size");
    const uint32_t clusters = std::min<uint32_t>(p.num_requests, push.num_blocks / kCluster);
    RuntimeCheck(clusters >= 1, "the push plane has fewer block counters than one cluster");
    const uint32_t grid = clusters * kCluster;
    p.ws = push.get_workspace<kWorld>(nbytes);
    p.num_counters = push.num_blocks;
    p.rank = push.rank;
    p.partial = static_cast<const uint4*>(partial.data_ptr());
    p.partial_all = nullptr;
    constexpr auto kernel = muse_vr_draft_arn::draft_ar_norm_kernel<kHidden, kWorld, kCluster, kUsePDL, false>;
    allow_cluster(kernel);
    LaunchKernel(grid, muse_vr_draft_arn::kThreads, device.unwrap())
        .config({.use_pdl = kUsePDL, .cluster_dim = dim3(kCluster, 1, 1)})(kernel, p);
  }

  // Test entry (one GPU): every rank's partial given as [kWorld, M, H] instead of the push plane; the call plays rank
  // `rank` (same cluster split and arithmetic as `run`).
  static void loopback(
      TensorView partial_all, int64_t rank, TensorView coef, int64_t coef_offset, int64_t coef_stride, TensorView base,
      TensorView residual, TensorView weight, TensorView out, double eps, int64_t max_clusters) {
    using namespace host;
    auto device = SymbolicDevice{};
    P p{};
    uint32_t rows = 0;
    fill(p, coef, coef_offset, coef_stride, base, residual, weight, out, eps, device, rows);
    auto M = SymbolicSize{"num_rows"};
    M.set_value(rows);
    TensorMatcher({int64_t(kWorld), M, int64_t(kHidden)}).with_dtype<bf16_t>().with_device(device).verify(partial_all);
    if (rows == 0) return;
    p.rank = static_cast<uint32_t>(rank);
    p.partial_all = static_cast<const uint4*>(partial_all.data_ptr());
    p.partial = nullptr;
    constexpr auto kernel = muse_vr_draft_arn::draft_ar_norm_kernel<kHidden, kWorld, kCluster, false, true>;
    allow_cluster(kernel);
    const uint32_t clusters =
        max_clusters > 0 ? std::min<uint32_t>(p.num_requests, static_cast<uint32_t>(max_clusters)) : p.num_requests;
    LaunchKernel(clusters * kCluster, muse_vr_draft_arn::kThreads, device.unwrap())
        .config({.use_pdl = false, .cluster_dim = dim3(kCluster, 1, 1)})(kernel, p);
  }
};

}  // namespace sglang
