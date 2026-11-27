// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// Adapted from SGLang (Apache-2.0, SGLang Team):
// python/sglang/kernels/ops/attention/linear/gdn_replayssm_spec_decode.py
// (gdn_replayssm_compact_commit_kernel and _finish_gdn_replayssm_circular_fold_kernel).

#include <cuda_bf16.h>

#ifndef MAX_L
#define MAX_L 16
#endif
#ifndef BV
#define BV 64
#endif
#ifndef STAGES
#define STAGES 2
#endif

typedef int i32;
typedef long long i64;
typedef unsigned int u32;
typedef signed char i8;

namespace q35_gdn_commit_detail {

constexpr int kK = 128;                // key head dim (checkpoint minor dim)
constexpr int kV = 128;                // value head dim
constexpr int kVec = 8;
constexpr int kColGroups = kK / kVec;
constexpr int kThreads = 256;

struct Params {
  __nv_bfloat16* h0;  // [layers, slots, HV, V, K]
  const __nv_bfloat16* d;
  const __nv_bfloat16* dr;  // [layers, ring rows, HV, L, V]
  const __nv_bfloat16* k;
  const __nv_bfloat16* kr;  // [layers, ring rows, H, L, K]
  const float* g;           // [layers, ring rows, HV, L]
  const void* state_idx;
  const void* replay_idx;
  const void* accept_lens;
  const void* track_idx;
  const void* track_steps;
  i32* write_pos;
  i32* cache_base;
  i8* is_flush;
  i64 s_state_layer, s_state_slot;
  i64 s_d_layer, s_d_slot, s_dr_layer, s_dr_slot;
  i64 s_k_layer, s_k_slot, s_kr_layer, s_kr_slot;
  i64 s_g_layer, s_g_slot;
  i32 num_layers, layers_per_cta, hv, hk, L;
  i32 null_block;
  i32 idx64;
  i32 has_track;
  i32 chunks;  // layer chunks per (row, value head, V tile)
  i32 items;   // rows x value heads x V tiles x chunks
};

__device__ __forceinline__ int imin(int a, int b) {
  return a < b ? a : b;
}

__device__ __forceinline__ int imax(int a, int b) {
  return a > b ? a : b;
}

__device__ __forceinline__ i64 ld_idx(const void* p, i64 i, bool is64) {
  return is64 ? static_cast<const i64*>(p)[i] : static_cast<i64>(static_cast<const i32*>(p)[i]);
}

__device__ __forceinline__ unsigned smem_u32(const void* p) {
  return static_cast<unsigned>(__cvta_generic_to_shared(p));
}

__device__ __forceinline__ void cp_async16(void* smem, const void* gmem) {
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" ::"r"(smem_u32(smem)), "l"(gmem) : "memory");
}

__device__ __forceinline__ void cp_async4(void* smem, const void* gmem) {
  asm volatile("cp.async.ca.shared.global [%0], [%1], 4;\n" ::"r"(smem_u32(smem)), "l"(gmem) : "memory");
}

__device__ __forceinline__ void cp_commit() {
  asm volatile("cp.async.commit_group;\n" ::: "memory");
}

template <int N>
__device__ __forceinline__ void cp_wait() {
  asm volatile("cp.async.wait_group %0;\n" ::"n"(N) : "memory");
}

__device__ __forceinline__ void ldsm_x4(unsigned addr, u32 (&r)[4]) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
               : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3])
               : "r"(addr));
}

__device__ __forceinline__ void ldsm_x4_t(unsigned addr, u32 (&r)[4]) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];\n"
               : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3])
               : "r"(addr));
}

__device__ __forceinline__ void stsm_x4(unsigned addr, const u32 (&r)[4]) {
  asm volatile("stmatrix.sync.aligned.m8n8.x4.shared.b16 [%0], {%1,%2,%3,%4};\n" ::"r"(addr), "r"(r[0]), "r"(r[1]),
               "r"(r[2]), "r"(r[3])
               : "memory");
}

__device__ __forceinline__ void mma_bf16(float (&d)[4], const u32 (&a)[4], u32 b0, u32 b1) {
  asm volatile(
      "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, "
      "{%0,%1,%2,%3};\n"
      : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}

__device__ __forceinline__ int swz(int r, int c) {
  return c ^ (r & 7);
}

template <int kMaxL, int kBV>
struct Stage {
  __nv_bfloat16 state[kBV * kK];    // swizzled rows
  __nv_bfloat16 k[2][kMaxL * kK];   // hi, lo; swizzled rows; rows >= n stay zero
  __nv_bfloat16 d[2][kMaxL * kBV];  // hi, lo
  float g[kMaxL];
};

__device__ __forceinline__ u32 pack_bf16(__nv_bfloat16 lo, __nv_bfloat16 hi) {
  return static_cast<u32>(__bfloat16_as_ushort(lo)) | (static_cast<u32>(__bfloat16_as_ushort(hi)) << 16);
}

__device__ __forceinline__ float2 unpack_bf16x2(u32 u) {
  return __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&u));
}

template <int kMaxL, int kBV>
__device__ __forceinline__ float warp_decay(const Stage<kMaxL, kBV>& st, int n, int lane, float* dec) {
  float x = lane < n ? st.g[lane] : 0.f;
#pragma unroll
  for (int off = 1; off < 32; off *= 2) {
    const float y = __shfl_up_sync(0xffffffffu, x, off);
    if (lane >= off) x += y;
  }
  const float total = __shfl_sync(0xffffffffu, x, 31);
  *dec = lane < n ? expf(total - x) : 0.f;
  return expf(total);
}

template <int kMaxL, int kBV, bool kInPlace>
__device__ __forceinline__ void mma_store(
    Stage<kMaxL, kBV>& st, int n, int ksteps, int lane, int warp, __nv_bfloat16* dst) {
  static_assert(kBV == 64, "warp tiling assumes 64 rows x 128 columns over 8 warps");
  const int mt = warp % 4;
  const int nb = (warp / 4) * 64;
  const int c4 = lane % 4;
  const int q = lane / 8, r = lane % 8;
  float dec;
  const float et = warp_decay(st, n, lane, &dec);
  float acc[8][4];
#pragma unroll
  for (int j = 0; j < 8; ++j)
#pragma unroll
    for (int e = 0; e < 4; ++e) acc[j][e] = 0.f;
  for (int ks = 0; ks < ksteps; ++ks) {
    u32 u_hi[4], u_lo[4];
    {
      const int t = ks * 16 + (q >> 1) * 8 + r;
      const int chunk = mt * 2 + (q & 1);
      const int off = t * kBV + swz(t, chunk) * kVec;
      ldsm_x4_t(smem_u32(&st.d[0][off]), u_hi);
      ldsm_x4_t(smem_u32(&st.d[1][off]), u_lo);
    }
    u32 a_hi[4], a_lo[4];
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      const int t = ks * 16 + 2 * c4 + 8 * (i >> 1);
      const float d0 = __shfl_sync(0xffffffffu, dec, t);
      const float d1 = __shfl_sync(0xffffffffu, dec, t + 1);
      const float2 h = unpack_bf16x2(u_hi[i]);
      const float2 l = unpack_bf16x2(u_lo[i]);
      const float x0 = t < n ? (h.x + l.x) * d0 : 0.f;
      const float x1 = t + 1 < n ? (h.y + l.y) * d1 : 0.f;
      const __nv_bfloat16 h0 = __float2bfloat16_rn(x0), h1 = __float2bfloat16_rn(x1);
      a_hi[i] = pack_bf16(h0, h1);
      a_lo[i] = pack_bf16(
          __float2bfloat16_rn(x0 - __bfloat162float(h0)), __float2bfloat16_rn(x1 - __bfloat162float(h1)));
    }
#pragma unroll
    for (int pr = 0; pr < 4; ++pr) {  // pairs of n-tiles
      const int t = ks * 16 + (q & 1) * 8 + r;
      const int chunk = nb / kVec + 2 * pr + (q >> 1);
      const int off = t * kK + swz(t, chunk) * kVec;
      u32 k_hi[4], k_lo[4];
      ldsm_x4_t(smem_u32(&st.k[0][off]), k_hi);
      ldsm_x4_t(smem_u32(&st.k[1][off]), k_lo);
      // SGLang's product order: K_hi A_hi, K_hi A_lo, K_lo A_hi, K_lo A_lo
      mma_bf16(acc[2 * pr], a_hi, k_hi[0], k_hi[1]);
      mma_bf16(acc[2 * pr], a_lo, k_hi[0], k_hi[1]);
      mma_bf16(acc[2 * pr], a_hi, k_lo[0], k_lo[1]);
      mma_bf16(acc[2 * pr], a_lo, k_lo[0], k_lo[1]);
      mma_bf16(acc[2 * pr + 1], a_hi, k_hi[2], k_hi[3]);
      mma_bf16(acc[2 * pr + 1], a_lo, k_hi[2], k_hi[3]);
      mma_bf16(acc[2 * pr + 1], a_hi, k_lo[2], k_lo[3]);
      mma_bf16(acc[2 * pr + 1], a_lo, k_lo[2], k_lo[3]);
    }
  }
#pragma unroll
  for (int pr = 0; pr < 4; ++pr) {
    const int v = mt * 16 + (q & 1) * 8 + r;
    const int chunk = nb / kVec + 2 * pr + (q >> 1);
    const unsigned addr = smem_u32(&st.state[v * kK + swz(v, chunk) * kVec]);
    u32 old[4];
    ldsm_x4(addr, old);
    u32 out[4];
#pragma unroll
    for (int m = 0; m < 4; ++m) {
      const float* a = &acc[2 * pr + (m >> 1)][2 * (m & 1)];
      const float2 o = unpack_bf16x2(old[m]);
      const __nv_bfloat162 y = __floats2bfloat162_rn(fmaf(o.x, et, a[0]), fmaf(o.y, et, a[1]));
      out[m] = *reinterpret_cast<const u32*>(&y);
    }
    if constexpr (kInPlace) {
      stsm_x4(addr, out);
    } else {
      const int g8 = lane / 4;
#pragma unroll
      for (int m = 0; m < 4; ++m) {
        const int vv = mt * 16 + g8 + 8 * (m & 1);
        const int kk = nb + 16 * pr + 8 * (m >> 1) + 2 * c4;
        *reinterpret_cast<u32*>(dst + static_cast<i64>(vv) * kK + kk) = out[m];
      }
    }
  }
}

template <int kMaxL, int kBV, int kStages>
__device__ __forceinline__ void run_item(const Params& p, Stage<kMaxL, kBV>* stages, int item) {
  using StageT = Stage<kMaxL, kBV>;
  constexpr int kNV = kV / kBV;
  constexpr int kStateVecs = kBV * kColGroups;

  const int tid = threadIdx.x;
  const int lane = tid % 32, warp = tid / 32;
  const int per_row = p.hv * kNV * p.chunks;
  const i64 i_n = item / per_row;
  const int rem = item % per_row;
  const int chunk = rem / (p.hv * kNV);
  const int i_hvt = rem % (p.hv * kNV);
  const int i_vt = i_hvt % kNV;
  const int i_hv = i_hvt / kNV;
  const int i_h = i_hv / (p.hv / p.hk);
  const int l_begin = chunk * p.layers_per_cta;
  const int l_end = imin(l_begin + p.layers_per_cta, p.num_layers);

  const i64 sidx = ld_idx(p.state_idx, i_n, p.idx64 & 1);
  if (sidx <= p.null_block) return;
  const i64 ridx = ld_idx(p.replay_idx, i_n, p.idx64 & 2);
  const i32 n_hist = p.write_pos[ridx];
  const bool fold = p.is_flush[ridx] != 0;
  bool has_track = false;
  i64 tidx = 0;
  i32 track_len = 0;
  if (p.has_track) {
    tidx = ld_idx(p.track_idx, i_n, p.idx64 & 8);
    const i64 tstep = ld_idx(p.track_steps, i_n, p.idx64 & 16);
    has_track = (tidx > p.null_block) && (tstep >= 0);
    const i64 nacc = ld_idx(p.accept_lens, i_n, p.idx64 & 4);
    track_len = static_cast<i32>(n_hist - nacc + tstep + 1);
  }
  const i32 base = p.cache_base[ridx];
  if (!fold && !has_track) return;
  const int Lmask = p.L - 1;
  const int n_fold = fold ? imax(0, imin(n_hist, p.L)) : 0;
  const int n_track = has_track ? imax(0, imin(track_len, p.L)) : 0;
  const int n_stage = imax(n_fold, n_track);
  const int ksteps = imax(1, (n_stage + 15) / 16);

  const int v0 = i_vt * kBV;
  const i64 tile_off = static_cast<i64>(i_hv) * kV * kK + static_cast<i64>(v0) * kK;
  const __nv_bfloat16* st_src = p.h0 + sidx * p.s_state_slot + tile_off;
  const __nv_bfloat16* k_src = p.k + ridx * p.s_k_slot + static_cast<i64>(i_h) * p.L * kK;
  const __nv_bfloat16* kr_src = p.kr + ridx * p.s_kr_slot + static_cast<i64>(i_h) * p.L * kK;
  const __nv_bfloat16* d_src = p.d + ridx * p.s_d_slot + static_cast<i64>(i_hv) * p.L * kV + v0;
  const __nv_bfloat16* dr_src = p.dr + ridx * p.s_dr_slot + static_cast<i64>(i_hv) * p.L * kV + v0;
  const float* g_src = p.g + ridx * p.s_g_slot + static_cast<i64>(i_hv) * p.L;

  {
    const uint4 z = make_uint4(0, 0, 0, 0);
    const int from = n_stage * kK / kVec, to = ksteps * 16 * kK / kVec;
    for (int s = 0; s < kStages; ++s)
      for (int i = from + tid; i < to; i += kThreads) {
        reinterpret_cast<uint4*>(stages[s].k[0])[i] = z;
        reinterpret_cast<uint4*>(stages[s].k[1])[i] = z;
      }
  }

  auto issue = [&](int l, StageT& s) {
#pragma unroll
    for (int i = tid; i < kStateVecs; i += kThreads) {
      const int v = i / kColGroups, c = i % kColGroups;
      cp_async16(&s.state[v * kK + swz(v, c) * kVec], st_src + l * p.s_state_layer + static_cast<i64>(i) * kVec);
    }
    for (int i = tid; i < n_stage * kColGroups; i += kThreads) {
      const int t = i / kColGroups, c = i % kColGroups;
      const i64 off = static_cast<i64>((base + t) & Lmask) * kK + c * kVec;
      cp_async16(&s.k[0][t * kK + swz(t, c) * kVec], k_src + l * p.s_k_layer + off);
      cp_async16(&s.k[1][t * kK + swz(t, c) * kVec], kr_src + l * p.s_kr_layer + off);
    }
    constexpr int kUVecs = kBV / kVec;
    for (int i = tid; i < n_stage * kUVecs; i += kThreads) {
      const int t = i / kUVecs, c = i % kUVecs;
      const i64 off = static_cast<i64>((base + t) & Lmask) * kV + c * kVec;
      cp_async16(&s.d[0][t * kBV + swz(t, c) * kVec], d_src + l * p.s_d_layer + off);
      cp_async16(&s.d[1][t * kBV + swz(t, c) * kVec], dr_src + l * p.s_dr_layer + off);
    }
    if (tid < n_stage) cp_async4(&s.g[tid], g_src + l * p.s_g_layer + ((base + tid) & Lmask));
  };
  __nv_bfloat16* st_dst = p.h0 + sidx * p.s_state_slot + tile_off;
  __nv_bfloat16* tr_dst = p.h0 + tidx * p.s_state_slot + tile_off;
  auto copy_out = [&](int l, const StageT& s) {
    __nv_bfloat16* dst = st_dst + l * p.s_state_layer;
#pragma unroll
    for (int i = tid; i < kStateVecs; i += kThreads) {
      const int v = i / kColGroups, c = i % kColGroups;
      *reinterpret_cast<uint4*>(dst + static_cast<i64>(i) * kVec) =
          *reinterpret_cast<const uint4*>(&s.state[v * kK + swz(v, c) * kVec]);
    }
  };

#pragma unroll
  for (int s = 0; s < kStages - 1; ++s) {
    if (l_begin + s < l_end) issue(l_begin + s, stages[s]);
    cp_commit();
  }
  for (int l = l_begin; l < l_end; ++l) {
    const int it = l - l_begin;
    cp_wait<kStages - 2>();
    __syncthreads();
    if (fold && it > 0) copy_out(l - 1, stages[(it - 1) % kStages]);
    const int nl = l + kStages - 1;
    if (nl < l_end) issue(nl, stages[(it + kStages - 1) % kStages]);
    cp_commit();
    StageT& s = stages[it % kStages];
    // track first: it reads the pre-fold rows that the fold overwrites in place
    if (has_track) mma_store<kMaxL, kBV, false>(s, n_track, ksteps, lane, warp, tr_dst + l * p.s_state_layer);
    if (fold) mma_store<kMaxL, kBV, true>(s, n_fold, ksteps, lane, warp, nullptr);
  }
  cp_wait<0>();
  __syncthreads();
  if (fold && l_end > l_begin) copy_out(l_end - 1, stages[(l_end - 1 - l_begin) % kStages]);
  __syncthreads();  // the next item reuses the stages
}

}

using q35_gdn_commit_detail::Params;

// One CTA per work item, or a smaller grid striding over the items.
extern "C" __global__ void __launch_bounds__(q35_gdn_commit_detail::kThreads, (STAGES <= 2 ? 3 : 2))
    q35_gdn_commit(const __grid_constant__ Params p) {
  extern __shared__ __align__(16) unsigned char smem_raw[];
  auto* stages = reinterpret_cast<q35_gdn_commit_detail::Stage<MAX_L, BV>*>(smem_raw);
  for (int item = blockIdx.x; item < p.items; item += gridDim.x)
    q35_gdn_commit_detail::run_item<MAX_L, BV, STAGES>(p, stages, item);
}

extern "C" __global__ void q35_gdn_commit_finish(const __grid_constant__ Params p, int rows) {
  for (int i = threadIdx.x; i < rows; i += blockDim.x) {
    if (q35_gdn_commit_detail::ld_idx(p.state_idx, i, p.idx64 & 1) <= p.null_block) continue;
    const i64 r = q35_gdn_commit_detail::ld_idx(p.replay_idx, i, p.idx64 & 2);
    if (p.is_flush[r] == 0) continue;
    p.cache_base[r] = (p.cache_base[r] + p.write_pos[r]) & (p.L - 1);
    p.write_pos[r] = 0;
    p.is_flush[r] = 0;
  }
}
