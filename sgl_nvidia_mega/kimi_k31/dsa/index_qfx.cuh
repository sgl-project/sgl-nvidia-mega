// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// K3.1 DSA index query, query-front CROSS form (K31MK_DSA_QFX, with K31MK_DSA_QF): sparse verify steps of 33-128 rows
// (Breq 5-16 x Q8) as one PDL kernel behind the q in-proj. The 1-32-row forms (index_q.cuh, index_qfw.cuh) are unchanged.
// K31MK_DSA_QFX_LOW: the same kernel also serves the QF-wide profiles of 9-32 rows (kT 2-4) in place of
// index_qfw.cuh; rows 9-32 are BITWISE index_qfw's by the argument below (test_k31_index_qfx: rows 9-32 vs QF-wide).
//
// Same method and rounding points as index_qfw.cuh (QF-wide): wq_bg = bf16(wq_b * w_qa) folded at load, acc = x @ wq_bg^T in FP32,
// rstd from the row's sum of squares, q = bf16(rstd * acc), ue8m0 scale of bf16(rstd * amax), e4m3 codes, w from the BF16 gate.
// Each row's outputs are BITWISE those of index_qfw.cuh on the same row (whose rows are independent of each other and of kR):
//   - acc[d][b] (head dim d, row b) is the fixed-order sum, from 0.f, over K slices c = 0..7 (192 columns each) of the slice
//     partial; each partial is an mma.sync m16n8k16 chain over the slice's 12 k-steps from 0 with the same fragments
//     (index_qfw: warp w computes slice w of 16 dims; here: cluster rank c computes slice c of all 128 dims);
//   - the row's sum of squares: slice c's partial in index_qfw's lane order, summed from 0.f in slice order;
//   - amax, rstd, scale, q8 and w: index_qfw's expressions.
//
// Why a different split: index_qfw gives each of a head's 8 CTAs 16 dims x the whole K, so every CTA holds whole rows of q_a in
// SMEM (rows x 3 KB: 32 rows max at 2 CTAs / SM). Here CTA rank c holds only its K slice of every row (rows x 384 B: 48 KB at
// 128 rows) and the 128 dims x slice c of wq_bg as mma A fragments in registers (warp w: dims [16 w, 16 w + 16)).
//
// Grid: 32 index heads x 8 CTAs (one cluster per head), 256 threads, two CTAs per SM. Row tile t (rows 8 t .. 8 t + 7) is OWNED
// by cluster rank t % 8. The product runs in passes of <= 8 tiles (pass p = tiles 8 p .. 8 p + 7, so tile 8 p + j goes to rank
// j); after each pass every warp stores its 16 dims x 8 rows partial of tile t into the owner's SMEM slot (sender rank) over
// DSMEM (the owner keeps its own partial in registers), and after the passes each rank's sum-of-squares partials of every row go
// to the row's owner likewise. ONE exchange: entry barrier.cluster.arrive.relaxed + wait.acquire before the first remote store
// (every peer is running), weak st.shared::cluster, barrier.cluster.arrive.release / wait.acquire, then the owner reads its
// slots (index_qfw's exchange pattern with more data). The owner then holds all 128 dims of its rows: amax, rstd, scale, w and
// q8 need no second exchange. q8 rows are staged in SMEM and stored as 128-byte rows.
//
// Index-K store (index_qfw's code): row b by warp 7 of the CTA (head b % 32, rank 7 - b / 32), between the exchange's arrive and
// wait; rows <= 128 = 4 x 32 heads, so every row has an owner. Its readers (the index logits) run after this grid completes.
// Instances: kT = 2..16 n8 row tiles (9-128 rows; 2-4 only under K31MK_DSA_QFX_LOW); rows above p.rows are zero-filled and never stored.
// Dependent launch (p.trig): griddepcontrol.launch_dependents right after griddepcontrol.wait (0, index_qfw's placement), after the exchange's
// wait (1) or at the kernel end (2). No memory effect (dependents still wait for this grid's completion): outputs are identical in every mode.
// Why: a chain rig (128 rows) showed the early-launched logits kernel's CTAs holding SMs while this grid ran.
#include <sgl_kernel/tensor.h>
#include <sgl_kernel/utils.h>

#include <sgl_kernel/runtime.cuh>
#include <sgl_kernel/utils.cuh>

#include <dlpack/dlpack.h>
#include <tvm/ffi/container/tensor.h>

#include <cuda_bf16.h>
#include <cstdint>

namespace k31_index_qfx {

constexpr int kHeads = 32, kDim = 128, kLora = 1536, kSplit = 8, kThreads = 256;
constexpr int kSlice = kLora / kSplit;   // 192 K columns per cluster rank
constexpr int kKSteps = kSlice / 16;     // 12 k-steps per slice
constexpr int kStride = kSlice * 2 + 16; // padded SMEM row (bytes): conflict-free fragment loads (index_qfw's padding rule)
constexpr int kMinTiles = 2, kMaxTiles = 16;


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
// Weak DSMEM stores; the exchange's barrier.cluster.arrive.release / wait.acquire orders them before the owner's reads.
__device__ __forceinline__ void st_peer_f32(const void* p, uint32_t rank, float v) {
  uint32_t a;
  asm volatile("mapa.shared::cluster.u32 %0, %1, %2;\n" : "=r"(a) : "r"(smem_addr(p)), "r"(rank));
  asm volatile("st.shared::cluster.f32 [%0], %1;\n" ::"r"(a), "f"(v) : "memory");
}
__device__ __forceinline__ void st_peer_v2(const void* p, uint32_t rank, float x, float y) {
  uint32_t a;
  asm volatile("mapa.shared::cluster.u32 %0, %1, %2;\n" : "=r"(a) : "r"(smem_addr(p)), "r"(rank));
  asm volatile("st.shared::cluster.v2.f32 [%0], {%1, %2};\n" ::"r"(a), "f"(x), "f"(y) : "memory");
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
// act_quant_triton.py round_scale: exp2(ceil(log2(max(amax, 1e-4) * (1 / 448)))) (index_qfw.cuh ue8m0_scale).
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
__device__ __forceinline__ float qcode_val(float acc, float rr, float scale) {  // index_qfw's q8 expression, before to_e4m3
  const float v = bf16_round(acc * rr);
  return fminf(fmaxf(div_full(v, scale), -448.f), 448.f);
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
  int trig;  // dependent-launch placement: 0 after the PDL wait, 1 after the exchange, 2 at the end
};

// One index-K row (warp-wide; lane owns columns 4 lane .. 4 lane + 3): index_qfw.cuh / index_q.cuh store_index_k, same code.
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

template <int kO>  // owned row tiles per CTA (1: <= 8 tiles, 2: 9-16 tiles)
struct Smem {
  alignas(16) uint8_t buf[kDim][kStride];  // wq_bg staging (128 dims x slice), then x (rows x slice), then the q8 rows
  alignas(16) float slot[kSplit - 1][kO][kDim][8];  // peers' partials of the owned tiles (slot = sender rank, own rank skipped)
  float ssx[kSplit][kO * 8];  // every rank's sum-of-squares partial of the owned rows (index = sender rank)
  float amx[kThreads / 32][kO * 8];  // per-warp amax of the owned rows (16 dims each)
  float rr[kO * 8];
  float scale[kO * 8];
};

template <int kT>
__global__ void __launch_bounds__(kThreads, 2) index_qfx_kernel(const __grid_constant__ Params p) {
  constexpr int kO = (kT + 7) / 8;
  static_assert(kT >= kMinTiles && kT <= kMaxTiles, "9-128 rows");
  static_assert(kT * 8 <= kDim, "x rows fit the staging buffer");
  using S = Smem<kO>;
  extern __shared__ __align__(16) uint8_t smem_raw[];
  S& s = *reinterpret_cast<S*>(smem_raw);
  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5, g = lane >> 2, qd = lane & 3;
  const int head = blockIdx.x / kSplit;
  const uint32_t rank = cluster_rank();
  const int col0 = static_cast<int>(rank) * kSlice;                           // this CTA's K slice
  const int krow = head + kHeads * (kSplit - 1 - static_cast<int>(rank));    // the index-K row this CTA stores
  const bool k_owner = warp == 7 && krow < p.rows;

  // Every peer has started (its shared memory exists) before anyone stores into it: arrive now, wait before the first
  // DSMEM store (CUDA PG, distributed shared memory).
  cluster_arrive_relaxed();
  // Weights only before the wait: wq_bg rows [128 head, + 128) x columns [col0, + 192) -> SMEM -> A fragments (registers).
  for (int i = tid; i < kDim * (kSlice / 8); i += kThreads) {
    const int r = i / (kSlice / 8), c = i % (kSlice / 8);
    cp_async16(&s.buf[r][c * 16], p.wq_bg + static_cast<int64_t>(head * kDim + r) * kLora + col0 + c * 8);
  }
  cp_async_commit();
  cp_async_wait_all();
  __syncthreads();
  uint32_t af[kKSteps][4];
#pragma unroll
  for (int ks = 0; ks < kKSteps; ++ks) {
    const int k0 = ks * 16;
    auto wq = [&](int r, int k) { return *reinterpret_cast<const uint32_t*>(&s.buf[r][k * 2]); };
    af[ks][0] = wq(16 * warp + g, k0 + 2 * qd);
    af[ks][1] = wq(16 * warp + g + 8, k0 + 2 * qd);
    af[ks][2] = wq(16 * warp + g, k0 + 8 + 2 * qd);
    af[ks][3] = wq(16 * warp + g + 8, k0 + 8 + 2 * qd);
  }
  __syncthreads();  // the staging rows are overwritten by x below
  pdl_wait();
  if (p.trig == 0) pdl_trigger();

  for (int i = tid; i < kT * 8 * (kSlice / 8); i += kThreads) {
    const int r = i / (kSlice / 8), c = i % (kSlice / 8);
    if (r < p.rows)
      cp_async16(&s.buf[r][c * 16], p.projected + static_cast<int64_t>(r) * p.pitch + col0 + c * 8);
    else
      *reinterpret_cast<uint4*>(&s.buf[r][c * 16]) = make_uint4(0, 0, 0, 0);
  }
  cp_async_commit();
  // Owned row index i = 8 m + r8 (m = 0 .. kO - 1): tile rank + 8 m, row 8 (rank + 8 m) + r8. Thread i < 8 kO reads its gate.
  float gq = 0.f;
  const int gcol = p.g_col + head;
  if (tid < kO * 8) {
    const int t = static_cast<int>(rank) + 8 * (tid >> 3), b = 8 * t + (tid & 7);
    if (t < kT && b < p.rows) gq = __bfloat162float(p.projected[static_cast<int64_t>(b) * p.pitch + gcol]);
  }
  uint2 kraw = {0u, 0u};
  int64_t kslot = -1;
  if (k_owner) {
    kraw = reinterpret_cast<const uint2*>(p.projected + static_cast<int64_t>(krow) * p.pitch + p.k_col)[lane];
    kslot = p.slots[krow];
  }
  cp_async_wait_all();
  __syncthreads();

  // Product in passes of <= 8 tiles; tile 8 pp + j is owned by rank j. own[pp]: this CTA's partial of its tile rank + 8 pp.
  float own[kO][4];
#pragma unroll
  for (int pp = 0; pp < kO; ++pp) {
    const int nt = kT - 8 * pp < 8 ? kT - 8 * pp : 8;
    float d[8][4];
#pragma unroll
    for (int j = 0; j < 8; ++j) d[j][0] = d[j][1] = d[j][2] = d[j][3] = 0.f;
#pragma unroll
    for (int ks = 0; ks < kKSteps; ++ks) {
      const int k0 = ks * 16;
#pragma unroll
      for (int j = 0; j < 8; ++j) {
        if (j < nt) {
          const int xr = 8 * (8 * pp + j) + g;
          const uint32_t b0 = *reinterpret_cast<const uint32_t*>(&s.buf[xr][(k0 + 2 * qd) * 2]);
          const uint32_t b1 = *reinterpret_cast<const uint32_t*>(&s.buf[xr][(k0 + 8 + 2 * qd) * 2]);
          mma_bf16(d[j], af[ks][0], af[ks][1], af[ks][2], af[ks][3], b0, b1);
        }
      }
    }
    if (pp == 0) cluster_wait_acquire();  // pairs with the arrive at kernel start: every peer is running
    own[pp][0] = own[pp][1] = own[pp][2] = own[pp][3] = 0.f;
#pragma unroll
    for (int j = 0; j < 8; ++j) {
      if (j < nt) {
        if (j == static_cast<int>(rank)) {
          own[pp][0] = d[j][0];
          own[pp][1] = d[j][1];
          own[pp][2] = d[j][2];
          own[pp][3] = d[j][3];
        } else {
          const uint32_t dst = static_cast<uint32_t>(j);
          const int sl = static_cast<int>(rank) < j ? static_cast<int>(rank) : static_cast<int>(rank) - 1;
          st_peer_v2(&s.slot[sl][pp][16 * warp + g][2 * qd], dst, d[j][0], d[j][1]);
          st_peer_v2(&s.slot[sl][pp][16 * warp + g + 8][2 * qd], dst, d[j][2], d[j][3]);
        }
      }
    }
  }
  // This rank's sum-of-squares partial of every row (index_qfw order: 4 lanes per row, slice-local k), to the row's owner.
#pragma unroll
  for (int sw = 0; sw < (kT * 8 + 63) / 64; ++sw) {
    if (sw * 64 + 8 * warp < kT * 8) {  // warp-uniform: warp covers rows sw * 64 + 8 warp .. + 8
      const int sr = sw * 64 + (tid >> 2), qq = tid & 3;
      float v = 0.f;
#pragma unroll
      for (int ks = 0; ks < kKSteps; ++ks) {
        const int k0 = ks * 16;
        const uint32_t b0 = *reinterpret_cast<const uint32_t*>(&s.buf[sr][(k0 + 2 * qq) * 2]);
        const uint32_t b1 = *reinterpret_cast<const uint32_t*>(&s.buf[sr][(k0 + 8 + 2 * qq) * 2]);
        v = sq2(b1, sq2(b0, v));
      }
      v += __shfl_xor_sync(0xffffffffu, v, 1);
      v += __shfl_xor_sync(0xffffffffu, v, 2);
      if (qq == 0) {
        const int t = sr >> 3;
        st_peer_f32(&s.ssx[rank][8 * (t >> 3) + (sr & 7)], static_cast<uint32_t>(t & 7), v);
      }
    }
  }

  // Owner side: the 8 slice partials of each owned tile, summed from 0.f in slice order (index_qfw's order).
  float acc[kO][4];
  auto reduce = [&]() {
#pragma unroll
    for (int m = 0; m < kO; ++m) {
      acc[m][0] = acc[m][1] = acc[m][2] = acc[m][3] = 0.f;
      if (static_cast<int>(rank) + 8 * m < kT) {
#pragma unroll
        for (int c = 0; c < kSplit; ++c) {
          float v0, v1, v2, v3;
          if (c == static_cast<int>(rank)) {
            v0 = own[m][0];
            v1 = own[m][1];
            v2 = own[m][2];
            v3 = own[m][3];
          } else {
            const int sl = c < static_cast<int>(rank) ? c : c - 1;
            const float2 lo = *reinterpret_cast<const float2*>(&s.slot[sl][m][16 * warp + g][2 * qd]);
            const float2 hi = *reinterpret_cast<const float2*>(&s.slot[sl][m][16 * warp + g + 8][2 * qd]);
            v0 = lo.x;
            v1 = lo.y;
            v2 = hi.x;
            v3 = hi.y;
          }
          acc[m][0] += v0;
          acc[m][1] += v1;
          acc[m][2] += v2;
          acc[m][3] += v3;
        }
      }
    }
  };
  cluster_arrive_release();
  // The index-K row of this CTA (rows < p.rows), off the exchange: between arrive and wait.
  if (k_owner) store_index_k(p, kraw, kslot, lane);
  cluster_wait_acquire();
  if (p.trig == 1) pdl_trigger();
  reduce();
  // Per-warp amax of the owned rows over this warp's 16 dims (lanes 4 g + qd: reduce over g), then over the 8 warps.
#pragma unroll
  for (int m = 0; m < kO; ++m) {
    if (static_cast<int>(rank) + 8 * m < kT) {  // CTA-uniform
      float a0 = fmaxf(fabsf(acc[m][0]), fabsf(acc[m][2]));  // row 2 qd
      float a1 = fmaxf(fabsf(acc[m][1]), fabsf(acc[m][3]));  // row 2 qd + 1
#pragma unroll
      for (int dd = 4; dd < 32; dd <<= 1) {
        a0 = fmaxf(a0, __shfl_xor_sync(0xffffffffu, a0, dd));
        a1 = fmaxf(a1, __shfl_xor_sync(0xffffffffu, a1, dd));
      }
      if (g == 0) {
        s.amx[warp][8 * m + 2 * qd] = a0;
        s.amx[warp][8 * m + 2 * qd + 1] = a1;
      }
    }
  }
  __syncthreads();
  if (tid < kO * 8) {
    const int t = static_cast<int>(rank) + 8 * (tid >> 3), b = 8 * t + (tid & 7);
    if (t < kT) {
      float amax = 0.f, ss = 0.f;
#pragma unroll
      for (int w8 = 0; w8 < kThreads / 32; ++w8) amax = fmaxf(amax, s.amx[w8][tid]);
#pragma unroll
      for (int c = 0; c < kSplit; ++c) {
        ss += s.ssx[c][tid];
      }
      const float rr = rsqrtf(ss / static_cast<float>(kLora) + p.eps);
      s.rr[tid] = rr;
      s.scale[tid] = ue8m0_scale(bf16_round(amax * rr));
      if (b < p.rows) {
        const float g2 = bf16_round(bf16_round(gq) * 0.17677669529663687f);  // 32^-0.5
        p.w[b * kHeads + head] = g2 * s.scale[tid] * 0.08838834764831845f;   // 128^-0.5
      }
    }
  }
  __syncthreads();
  // q8 of the owned rows -> SMEM rows of 128 codes (x is dead: every thread passed the exchange after its last x read).
  uint8_t* qb = &s.buf[0][0];
#pragma unroll
  for (int m = 0; m < kO; ++m) {
    if (static_cast<int>(rank) + 8 * m < kT) {
      const int i0 = 8 * m + 2 * qd, i1 = i0 + 1, dlo = 16 * warp + g, dhi = dlo + 8;
      qb[i0 * kDim + dlo] = to_e4m3(qcode_val(acc[m][0], s.rr[i0], s.scale[i0]));
      qb[i1 * kDim + dlo] = to_e4m3(qcode_val(acc[m][1], s.rr[i1], s.scale[i1]));
      qb[i0 * kDim + dhi] = to_e4m3(qcode_val(acc[m][2], s.rr[i0], s.scale[i0]));
      qb[i1 * kDim + dhi] = to_e4m3(qcode_val(acc[m][3], s.rr[i1], s.scale[i1]));
    }
  }
  __syncthreads();
  if (tid < kO * 8 * 8) {  // 16-byte chunks: owned row i = tid / 8, chunk tid % 8
    const int i = tid >> 3, t = static_cast<int>(rank) + 8 * (i >> 3), b = 8 * t + (i & 7);
    if (t < kT && b < p.rows)
      *reinterpret_cast<uint4*>(p.q8 + (static_cast<int64_t>(b) * kHeads + head) * kDim + (tid & 7) * 16) =
          *reinterpret_cast<const uint4*>(&qb[i * kDim + (tid & 7) * 16]);
  }
  if (p.trig == 2) pdl_trigger();
}

struct Binding {
  // `projected` [rows, pitch] carries q_a at [0, 1536), the index-K rows at [k_col, k_col + 128) and the 32 head-gate logits
  // at [g_col, g_col + 32); the new tokens' index-K rows go to `kcache` at `slots`. `kt` 0 = ceil(rows / 8) n8 tiles; 5-16
  // forces a larger instance (tests).
  static void run_qfx(const tvm::ffi::TensorView projected, const tvm::ffi::TensorView wq_bg, const tvm::ffi::TensorView q8,
                      const tvm::ffi::TensorView w, const double eps, const int64_t k_col, const int64_t g_col,
                      const tvm::ffi::TensorView slots, const tvm::ffi::TensorView kcache, const tvm::ffi::TensorView kgamma,
                      const tvm::ffi::TensorView kbeta, const double keps, const int64_t kt, const int64_t trig) {
    using namespace sglang::host;
    const int64_t rows = projected.size(0);
    RuntimeCheck(rows >= 9 && rows <= 128, "QF-cross index query serves 9-128 rows");
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
    const int64_t need = (rows + 7) / 8;
    RuntimeCheck(kt == 0 || (kt >= need && kt <= kMaxTiles), "instance kt >= ceil(rows / 8), <= 16");
    RuntimeCheck(trig >= 0 && trig <= 2, "trig 0 / 1 / 2");
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
    p.trig = static_cast<int>(trig);
    const int64_t inst = kt != 0 ? kt : need;
    auto go = [&](auto kernel, size_t bytes) {
      // Two CTAs per SM need the full shared-memory carveout.
      CHECK_CUDA(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, static_cast<int>(bytes)));
      CHECK_CUDA(cudaFuncSetAttribute(kernel, cudaFuncAttributePreferredSharedMemoryCarveout, 100));
      LaunchKernel(kHeads * kSplit, kThreads, projected.device(), bytes)
          .config({.use_pdl = true, .cluster_dim = dim3(kSplit, 1, 1)})(kernel, p);
    };
    switch (inst) {
      case 2: go(index_qfx_kernel<2>, sizeof(Smem<1>)); break;
      case 3: go(index_qfx_kernel<3>, sizeof(Smem<1>)); break;
      case 4: go(index_qfx_kernel<4>, sizeof(Smem<1>)); break;
      case 5: go(index_qfx_kernel<5>, sizeof(Smem<1>)); break;
      case 6: go(index_qfx_kernel<6>, sizeof(Smem<1>)); break;
      case 7: go(index_qfx_kernel<7>, sizeof(Smem<1>)); break;
      case 8: go(index_qfx_kernel<8>, sizeof(Smem<1>)); break;
      case 9: go(index_qfx_kernel<9>, sizeof(Smem<2>)); break;
      case 10: go(index_qfx_kernel<10>, sizeof(Smem<2>)); break;
      case 11: go(index_qfx_kernel<11>, sizeof(Smem<2>)); break;
      case 12: go(index_qfx_kernel<12>, sizeof(Smem<2>)); break;
      case 13: go(index_qfx_kernel<13>, sizeof(Smem<2>)); break;
      case 14: go(index_qfx_kernel<14>, sizeof(Smem<2>)); break;
      case 15: go(index_qfx_kernel<15>, sizeof(Smem<2>)); break;
      default: go(index_qfx_kernel<16>, sizeof(Smem<2>)); break;
    }
  }
};

}  // namespace k31_index_qfx
