// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// K3.1 DSA index query, query-front WIDE form (K31MK_DSA_QFW, with K31MK_DSA_QF): sparse verify steps of 9-32 rows
// (Breq 2-4 x Q8) as one PDL kernel behind the q in-proj. The 1-8-row path (index_q.cuh) is unchanged.
//
// Method (NOT the stock rounding points; numerics-changing, gated by an FP64 reference and an in-boot KL arm):
//   wq_bg = bf16(wq_b * w_qa)                                  (load time, FusedIndexQueryWide; gamma folded into wq_b)
//   acc   = x @ wq_bg^T                                        (x = projected[:, :1536] raw BF16 q_a, FP32 accumulation)
//   rstd  = rsqrt(sum(x^2) / 1536 + eps)                       (FP32; see the sum order below)
//   q     = bf16(rstd * acc)
//   scale = exp2(ceil(log2(max(bf16(rstd * amax_128(acc)), 1e-4) * (1/448))))   (ue8m0, per row and head; = amax of q:
//           RN rounding is monotone and odd, so max|bf16(rstd * acc)| == bf16(rstd * max|acc|))
//   q8    = e4m3_rn(clamp(q / scale, +-448))
//   w     = float(bf16(gate * 32^-0.5)) * scale * 128^-0.5      (gate = projected[:, g_col + head], BF16)
// Stock instead rounds xn = bf16(x * rstd * w_qa) BEFORE the product and q = bf16(xn @ wq_b^T). Measured (1-GPU
// chain rig, 16 / 24 / 32 rows): FP8 codes vs an FP64 reference differ in 7.32-7.39 % of cells, stock's own 7.35-7.40 %;
// vs stock 7.9 % of codes differ.
//
// Grid: 32 index heads x 8 CTAs (one cluster per head). CTA rank c owns wq_bg rows [128 head + 16 c, + 16) as mma A fragments
// in registers (staged before griddepcontrol.wait: weights only) and the RMSNorm partial of q_a columns [192 c, 192 c + 192).
// Warp w accumulates K columns [192 w, 192 w + 192) of the 16 x kR product (kR / 8 n8 tiles per k-step).
// The per-rank sum-of-squares partial rides the head's amax DSMEM exchange (one more float per row on the same edge: same
// st.relaxed.cluster stores, same barrier.cluster arrive.release / wait.acquire pair as index_q.cuh); after the wait every CTA
// sums the 8 partials in rank order, so all 8 CTAs of a head hold the same rstd.
// Sum order of a row's sum of squares: rank c's partial = lanes qq 0..3 each over k = 192 c + 16 ks + {2 qq, 2 qq + 1, 8 + 2 qq,
// 8 + 2 qq + 1}, ks = 0..11 (sequential fma), then xor-1 and xor-2 shuffles; total = sum over c in order 0..7.
//
// Index-K store (the QF form of index_q.cuh, unchanged method and code): row b's index-K
// (projected[b, k_col : k_col + 128]) is stored by warp 7 of the CTA (head b, rank 7) between the exchange's arrive and wait;
// rows <= 32 = heads, so every row has an owner. Its readers (the index logits) run after this grid completes.
//
// Instances: kR 16 (9-16 rows), 24 (17-24), 32 (25-32); two CTAs per SM (Smem<32> 103 KB). Each row's arithmetic does not
// depend on kR or on other rows, so a row's outputs are bitwise the same in every instance (test_k31_index_qfw).
#include <sgl_kernel/tensor.h>
#include <sgl_kernel/utils.h>

#include <sgl_kernel/runtime.cuh>
#include <sgl_kernel/utils.cuh>

#include <dlpack/dlpack.h>
#include <tvm/ffi/container/tensor.h>

#include <cuda_bf16.h>
#include <cstdint>

namespace k31_index_qfw {

constexpr int kHeads = 32, kDim = 128, kLora = 1536, kSplit = 8, kOwn = kDim / kSplit, kThreads = 256;
constexpr int kStride = kLora * 2 + 16;  // padded SMEM row (bytes): conflict-free fragment loads
constexpr int kKSteps = kLora / 8 / 16;  // k-steps per warp (192 columns)
constexpr int kSlice = kLora / kSplit;   // 192 columns per cluster rank


__device__ __forceinline__ uint32_t smem_addr(const void* p) {
  return static_cast<uint32_t>(__cvta_generic_to_shared(p));
}
__device__ __forceinline__ void cp_async16(void* dst, const void* src) {
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" ::"r"(smem_addr(dst)), "l"(src) : "memory");
}
__device__ __forceinline__ void cp_async_commit() { asm volatile("cp.async.commit_group;\n" ::: "memory"); }
__device__ __forceinline__ void cp_async_wait_all() { asm volatile("cp.async.wait_group 0;\n" ::: "memory"); }
__device__ __forceinline__ void pdl_wait() { asm volatile("griddepcontrol.wait;" ::: "memory"); }
__device__ __forceinline__ void pdl_trigger() { asm volatile("griddepcontrol.launch_dependents;" ::: "memory"); }
__device__ __forceinline__ uint32_t cluster_rank() {
  uint32_t r;
  asm volatile("mov.u32 %0, %%cluster_ctarank;\n" : "=r"(r));
  return r;
}
__device__ __forceinline__ void cluster_arrive_release() {
  asm volatile("barrier.cluster.arrive.release.aligned;\n" ::: "memory");
}
__device__ __forceinline__ void cluster_wait_acquire() {
  asm volatile("barrier.cluster.wait.acquire.aligned;\n" ::: "memory");
}
__device__ __forceinline__ void cluster_arrive_relaxed() {
  asm volatile("barrier.cluster.arrive.relaxed.aligned;\n" ::: "memory");
}
__device__ __forceinline__ void st_peer_f32(const void* p, uint32_t rank, float v) {
  uint32_t a;
  asm volatile("mapa.shared::cluster.u32 %0, %1, %2;\n" : "=r"(a) : "r"(smem_addr(p)), "r"(rank));
  asm volatile("st.relaxed.cluster.shared::cluster.f32 [%0], %1;\n" ::"r"(a), "f"(v) : "memory");
}
__device__ __forceinline__ void mma_bf16(float (&d)[4], uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3,
                                         uint32_t b0, uint32_t b1) {
  asm volatile(
      "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, "
      "{%0, %1, %2, %3};\n"
      : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
      : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}
__device__ __forceinline__ uint8_t to_e4m3(float x) {
  uint16_t pair;
  asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;\n" : "=h"(pair) : "f"(0.f), "f"(x));
  return static_cast<uint8_t>(pair & 0xff);
}
__device__ __forceinline__ float bf16_round(float x) { return __bfloat162float(__float2bfloat16_rn(x)); }
__device__ __forceinline__ float div_full(float a, float b) {
  float r;
  asm("div.full.f32 %0, %1, %2;\n" : "=f"(r) : "f"(a), "f"(b));
  return r;
}
// act_quant_triton.py round_scale: exp2(ceil(log2(max(amax, 1e-4) * (1 / 448)))) (index_q.cuh ue8m0_scale).
__device__ __forceinline__ float ue8m0_scale(float amax) {
  const float x = fmaxf(amax, 1e-4f) * (1.0f / 448.0f);
  const float e = ceilf(log2f(x));
  float scale;
  asm("ex2.approx.ftz.f32 %0, %1;\n" : "=f"(scale) : "f"(e));
  return scale;
}
__device__ __forceinline__ float sq2(uint32_t v, float acc) {
  const __nv_bfloat162 h = *reinterpret_cast<const __nv_bfloat162*>(&v);
  const float2 f = __bfloat1622float2(h);
  return fmaf(f.y, f.y, fmaf(f.x, f.x, acc));
}

struct Params {
  const __nv_bfloat16* projected;  // [rows, pitch]; q_a = columns [0, 1536)
  const __nv_bfloat16* wq_bg;      // [4096, 1536] = bf16(wq_b * w_qa)
  uint8_t* q8;                     // [rows, 32, 128] FP8 E4M3
  float* w;                        // [rows, 32]
  int64_t pitch;
  int rows;
  float eps;
  int k_col, g_col;      // index-K / head-gate columns of `projected`
  const int64_t* slots;  // [rows]; a negative slot skips the store
  uint8_t* kcache;       // index-K pages [pages, 64 * 132]
  const float* kgamma;   // [128] FP32
  const float* kbeta;    // [128] FP32
  float keps;
};

// One index-K row (warp-wide; lane owns columns 4 lane .. 4 lane + 3): index_q.cuh store_index_k, same code.
__device__ __forceinline__ void store_index_k(const Params& p, uint2 raw, int64_t slot, int lane) {
  const __nv_bfloat16* e = reinterpret_cast<const __nv_bfloat16*>(&raw);
  float v[4], sum = 0.f;
#pragma unroll
  for (int j = 0; j < 4; ++j) {
    v[j] = __bfloat162float(e[j]);
    sum += v[j];
  }
#pragma unroll
  for (int d = 16; d; d >>= 1) sum += __shfl_xor_sync(0xffffffffu, sum, d);
  const float mean = sum * (1.0f / 128.0f);
  float var = 0.f;
#pragma unroll
  for (int j = 0; j < 4; ++j) {
    v[j] -= mean;
    var = fmaf(v[j], v[j], var);
  }
#pragma unroll
  for (int d = 16; d; d >>= 1) var += __shfl_xor_sync(0xffffffffu, var, d);
  const float rstd = rsqrtf(var * (1.0f / 128.0f) + p.keps);
  float amax = 0.f;
#pragma unroll
  for (int j = 0; j < 4; ++j) {
    const int col = lane * 4 + j;
    v[j] = bf16_round(v[j] * rstd * p.kgamma[col] + p.kbeta[col]);
    amax = fmaxf(amax, fabsf(v[j]));
  }
#pragma unroll
  for (int d = 16; d; d >>= 1) amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, d));
  if (slot < 0) return;
  const float scale = __fdiv_rn(fmaxf(1e-4f, amax), 448.0f);
  const float inv = __fdiv_rn(1.0f, scale);
  uint8_t* page = p.kcache + (slot >> 6) * static_cast<int64_t>(132 * 64);
  const int off = static_cast<int>(slot & 63);
  uint32_t packed = 0;
#pragma unroll
  for (int j = 0; j < 4; ++j)
    packed |= static_cast<uint32_t>(to_e4m3(fminf(fmaxf(v[j] * inv, -448.f), 448.f))) << (8 * j);
  reinterpret_cast<uint32_t*>(page + off * 128)[lane] = packed;
  if (lane == 0) *reinterpret_cast<float*>(page + 128 * 64 + off * 4) = scale;
}

template <int kR>
struct Smem {
  alignas(16) uint8_t x[kR][kStride];  // wq_bg staging, then q_a, then the per-warp partials
  float val[kOwn][kR];                 // FP32 acc (unrounded) until the exchange
  float ex_amax[kSplit][kR];
  float ex_ss[kSplit][kR];             // per-rank sum-of-squares partials (ride the amax exchange)
  float ssl[kR];                       // this CTA's partial
  float rr[kR];
  float scale[kR];
};

template <int kR>
__global__ void __launch_bounds__(kThreads, 2) index_qfw_kernel(const __grid_constant__ Params p) {
  using S = Smem<kR>;
  constexpr int kTiles = kR / 8;
  static_assert(kR % 8 == 0 && kR >= kOwn && kR <= kHeads, "n8 tiles; staging needs 16 rows; one index-K owner per row");
  static_assert(sizeof(S::x) >= 8 * kOwn * kR * sizeof(float), "partials fit in the activation buffer");
  static_assert(4 * kR <= kThreads && (4 * kR) % 32 == 0, "slice pass: 4 lanes per row, whole warps");
  extern __shared__ __align__(16) uint8_t smem_raw[];
  S& s = *reinterpret_cast<S*>(smem_raw);
  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5, g = lane >> 2, qd = lane & 3;
  const int head = blockIdx.x / kSplit;
  const uint32_t rank = cluster_rank();
  const int row0 = head * kDim + static_cast<int>(rank) * kOwn;  // first wq_bg row of this CTA
  const bool k_owner = rank == kSplit - 1 && warp == 7 && head < p.rows;

  // Every peer has started (its shared memory exists) before anyone stores into it: arrive now, wait before the first
  // DSMEM store (CUDA PG, distributed shared memory).
  cluster_arrive_relaxed();
  // Weights only before the wait: this CTA's 16 wq_bg rows -> SMEM -> A fragments (registers).
  for (int i = tid; i < kOwn * (kLora / 8); i += kThreads) {
    const int r = i / (kLora / 8), c = i % (kLora / 8);
    cp_async16(&s.x[r][c * 16], p.wq_bg + static_cast<int64_t>(row0 + r) * kLora + c * 8);
  }
  cp_async_commit();
  cp_async_wait_all();
  __syncthreads();
  uint32_t af[kKSteps][4];
#pragma unroll
  for (int ks = 0; ks < kKSteps; ++ks) {
    const int k0 = warp * (kLora / 8) + ks * 16;
    auto wq = [&](int r, int k) { return *reinterpret_cast<const uint32_t*>(&s.x[r][k * 2]); };
    af[ks][0] = wq(g, k0 + 2 * qd);
    af[ks][1] = wq(g + 8, k0 + 2 * qd);
    af[ks][2] = wq(g, k0 + 8 + 2 * qd);
    af[ks][3] = wq(g + 8, k0 + 8 + 2 * qd);
  }
  __syncthreads();  // the staging rows are overwritten by q_a below
  pdl_wait();
  pdl_trigger();

  for (int i = tid; i < kR * (kLora / 8); i += kThreads) {
    const int r = i / (kLora / 8), c = i % (kLora / 8);
    if (r < p.rows)
      cp_async16(&s.x[r][c * 16], p.projected + static_cast<int64_t>(r) * p.pitch + c * 8);
    else
      *reinterpret_cast<uint4*>(&s.x[r][c * 16]) = make_uint4(0, 0, 0, 0);
  }
  cp_async_commit();
  float gq = 0.f;          // rank 0, thread b < rows: row b's head-gate logit (BF16 value)
  uint2 kraw = {0u, 0u};   // index-K owner: lane's 4 index-K values of row `head`
  int64_t kslot = -1;
  const int gcol = p.g_col + head;
  if (rank == 0 && tid < p.rows) gq = __bfloat162float(p.projected[static_cast<int64_t>(tid) * p.pitch + gcol]);
  if (k_owner) {
    kraw = reinterpret_cast<const uint2*>(p.projected + static_cast<int64_t>(head) * p.pitch + p.k_col)[lane];
    kslot = p.slots[head];
  }
  cp_async_wait_all();
  __syncthreads();
  // Product only: d[j] = 16 wq_bg rows x rows 8 j .. 8 j + 7, this warp's 192 K columns.
  float d[kTiles][4];
#pragma unroll
  for (int j = 0; j < kTiles; ++j) d[j][0] = d[j][1] = d[j][2] = d[j][3] = 0.f;
#pragma unroll
  for (int ks = 0; ks < kKSteps; ++ks) {
    const int k0 = warp * (kLora / 8) + ks * 16;
#pragma unroll
    for (int j = 0; j < kTiles; ++j) {
      const int xr = 8 * j + g;
      const uint32_t b0 = *reinterpret_cast<const uint32_t*>(&s.x[xr][(k0 + 2 * qd) * 2]);
      const uint32_t b1 = *reinterpret_cast<const uint32_t*>(&s.x[xr][(k0 + 8 + 2 * qd) * 2]);
      mma_bf16(d[j], af[ks][0], af[ks][1], af[ks][2], af[ks][3], b0, b1);
    }
  }
  // This rank's sum-of-squares partial: thread t < 4 kR owns row t / 4, lane quad qq = t % 4 (order in the header).
  if (tid < 4 * kR) {
    const int sr = tid >> 2, qq = tid & 3;
    float v = 0.f;
#pragma unroll
    for (int ks = 0; ks < kKSteps; ++ks) {
      const int k0 = static_cast<int>(rank) * kSlice + ks * 16;
      const uint32_t b0 = *reinterpret_cast<const uint32_t*>(&s.x[sr][(k0 + 2 * qq) * 2]);
      const uint32_t b1 = *reinterpret_cast<const uint32_t*>(&s.x[sr][(k0 + 8 + 2 * qq) * 2]);
      v = sq2(b1, sq2(b0, v));
    }
    v += __shfl_xor_sync(0xffffffffu, v, 1);
    v += __shfl_xor_sync(0xffffffffu, v, 2);
    if (qq == 0) s.ssl[sr] = v;
  }
  __syncthreads();  // every warp has read q_a: the partials reuse the buffer
  float* part = reinterpret_cast<float*>(&s.x[0][0]);
#pragma unroll
  for (int j = 0; j < kTiles; ++j) {
    const int b = 8 * j + 2 * qd;
    part[(warp * kOwn + g) * kR + b] = d[j][0];
    part[(warp * kOwn + g) * kR + b + 1] = d[j][1];
    part[(warp * kOwn + g + 8) * kR + b] = d[j][2];
    part[(warp * kOwn + g + 8) * kR + b + 1] = d[j][3];
  }
  __syncthreads();
  for (int i = tid; i < kOwn * kR; i += kThreads) {  // fixed-order sum over the 8 K slices (no rounding yet)
    const int r = i / kR, b = i % kR;
    float acc = 0.f;
#pragma unroll
    for (int w8 = 0; w8 < 8; ++w8) acc += part[(w8 * kOwn + r) * kR + b];
    s.val[r][b] = acc;
  }
  __syncthreads();
  cluster_wait_acquire();  // pairs with the arrive at kernel start: every peer is running
  if (tid < kR) {  // this CTA's amax over its 16 rows and its ss partial: to every CTA of the head's cluster
    const int b = tid;
    float amax = 0.f;
#pragma unroll
    for (int r = 0; r < kOwn; ++r) amax = fmaxf(amax, fabsf(s.val[r][b]));
    const float ssl = s.ssl[b];
#pragma unroll
    for (int c = 0; c < kSplit; ++c) {
      st_peer_f32(&s.ex_amax[rank][b], c, amax);
      st_peer_f32(&s.ex_ss[rank][b], c, ssl);
    }
  }
  cluster_arrive_release();
  // The index-K row of this CTA's head (rows < p.rows), off the exchange: between arrive and wait.
  if (k_owner) store_index_k(p, kraw, kslot, lane);
  cluster_wait_acquire();
  if (tid < kR) {
    const int b = tid;
    float amax = 0.f, ss = 0.f;
#pragma unroll
    for (int c = 0; c < kSplit; ++c) amax = fmaxf(amax, s.ex_amax[c][b]);
#pragma unroll
    for (int c = 0; c < kSplit; ++c) {
      ss += s.ex_ss[c][b];
    }
    const float rr = rsqrtf(ss / static_cast<float>(kLora) + p.eps);
    s.rr[b] = rr;
    s.scale[b] = ue8m0_scale(bf16_round(amax * rr));
    if (rank == 0 && b < p.rows) {
      const float g2 = bf16_round(bf16_round(gq) * 0.17677669529663687f);  // 32^-0.5
      p.w[b * kHeads + head] = g2 * s.scale[b] * 0.08838834764831845f;     // 128^-0.5
    }
  }
  __syncthreads();
  for (int i = tid; i < kOwn * kR; i += kThreads) {
    const int r = i % kOwn, b = i / kOwn;
    if (b < p.rows) {
      const float v = bf16_round(s.val[r][b] * s.rr[b]);
      const float y = fminf(fmaxf(div_full(v, s.scale[b]), -448.f), 448.f);
      p.q8[(static_cast<int64_t>(b) * kHeads + head) * kDim + rank * kOwn + r] = to_e4m3(y);
    }
  }
}

struct Binding {
  // `projected` [rows, pitch] carries q_a at [0, 1536), the index-K rows at [k_col, k_col + 128) and the 32 head-gate logits
  // at [g_col, g_col + 32); the new tokens' index-K rows go to `kcache` at `slots`. `kr` 0 = the smallest instance that holds
  // the rows (16 / 24 / 32); 16 / 24 / 32 force an instance (tests).
  static void run_qfw(const tvm::ffi::TensorView projected, const tvm::ffi::TensorView wq_bg, const tvm::ffi::TensorView q8,
                      const tvm::ffi::TensorView w, const double eps, const int64_t k_col, const int64_t g_col,
                      const tvm::ffi::TensorView slots, const tvm::ffi::TensorView kcache, const tvm::ffi::TensorView kgamma,
                      const tvm::ffi::TensorView kbeta, const double keps, const int64_t kr) {
    using namespace sglang::host;
    const int64_t rows = projected.size(0);
    RuntimeCheck(rows >= 9 && rows <= 32, "QF-wide index query serves 9-32 rows");
    RuntimeCheck(projected.stride(1) == 1 && projected.stride(0) % 8 == 0, "16-byte aligned projected rows");
    RuntimeCheck(k_col >= kLora && k_col % 4 == 0 && k_col + kDim <= projected.size(1), "index-K columns");
    RuntimeCheck(g_col >= kLora && g_col + kHeads <= projected.size(1), "head-gate columns");
    RuntimeCheck(wq_bg.size(0) == kHeads * kDim && wq_bg.size(1) == kLora && wq_bg.stride(0) == kLora, "wq_bg");
    RuntimeCheck(q8.size(0) == rows && w.size(0) == rows, "outputs");
    RuntimeCheck(slots.size(0) >= rows && slots.stride(0) == 1 && slots.dtype().bits == 64, "int64 write slots");
    RuntimeCheck(kcache.ndim() == 2 && kcache.size(1) == 64 * 132 && kcache.stride(0) == 64 * 132 && kcache.stride(1) == 1,
                 "index-K pages [pages, 64 * 132]");
    RuntimeCheck(kgamma.size(0) == kDim && kbeta.size(0) == kDim && kgamma.dtype().bits == 32 && kbeta.dtype().bits == 32,
                 "FP32 index-K LayerNorm");
    RuntimeCheck(kr == 0 || ((kr == 16 || kr == 24 || kr == 32) && kr >= rows), "instance 16 / 24 / 32 >= rows");
    Params p{};
    p.projected = static_cast<const __nv_bfloat16*>(projected.data_ptr());
    p.wq_bg = static_cast<const __nv_bfloat16*>(wq_bg.data_ptr());
    p.q8 = static_cast<uint8_t*>(q8.data_ptr());
    p.w = static_cast<float*>(w.data_ptr());
    p.pitch = projected.stride(0);
    p.rows = static_cast<int>(rows);
    p.eps = static_cast<float>(eps);
    p.k_col = static_cast<int>(k_col);
    p.g_col = static_cast<int>(g_col);
    p.slots = static_cast<const int64_t*>(slots.data_ptr());
    p.kcache = static_cast<uint8_t*>(kcache.data_ptr());
    p.kgamma = static_cast<const float*>(kgamma.data_ptr());
    p.kbeta = static_cast<const float*>(kbeta.data_ptr());
    p.keps = static_cast<float>(keps);
    const int64_t inst = kr != 0 ? kr : (rows <= 16 ? 16 : rows <= 24 ? 24 : 32);
    auto go = [&](auto kernel, size_t bytes) {
      // Two CTAs per SM need the full shared-memory carveout.
      CHECK_CUDA(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, static_cast<int>(bytes)));
      CHECK_CUDA(cudaFuncSetAttribute(kernel, cudaFuncAttributePreferredSharedMemoryCarveout, 100));
      LaunchKernel(kHeads * kSplit, kThreads, projected.device(), bytes)
          .config({.use_pdl = true, .cluster_dim = dim3(kSplit, 1, 1)})(kernel, p);
    };
    if (inst == 16)
      go(index_qfw_kernel<16>, sizeof(Smem<16>));
    else if (inst == 24)
      go(index_qfw_kernel<24>, sizeof(Smem<24>));
    else
      go(index_qfw_kernel<32>, sizeof(Smem<32>));
  }
};

}  // namespace k31_index_qfw
