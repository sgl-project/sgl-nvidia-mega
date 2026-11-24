// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
//
// Attention for the DFlash draft block: every request has kQ query tokens at the end of its sequence (positions
// seqlen - kQ .. seqlen - 1), attends without a causal mask to the keys of a paged bf16 KV cache within an optional
// window (key j is visible to query position p when p - left <= j <= p + right, a negative bound meaning unbounded),
// with grouped query heads (kGroup query heads per KV head) and head_dim 128.
//
// Work split: one thread-block cluster of kSplits CTAs per (request, KV head). The visible key range of the request
// is cut into kSplits contiguous parts (multiples of 16 keys); a CTA computes, for the kQ * kGroup query rows of its
// KV head, the online-softmax partial result over its part (row maximum, row sum, unnormalized output) with
// mma.sync.m16n8k16 (bf16 inputs, fp32 accumulation; probabilities rounded to bf16 for the second product); eight
// warps split it as 4 row slices x 2 halves of every 64-key block, and the two halves are combined in shared memory.
// The CTA partials are combined through distributed shared memory: every CTA stores its row maxima and row sums into
// every peer, and its output columns [s * 128 / kSplits, ...) into peer s (st.shared::cluster), after the cluster
// barrier phase that every CTA arrives on at entry (so all peers are running); then every thread arrives with
// release and waits with acquire, and CTA s normalizes its column slice from its own shared memory
// (PTX ISA: barrier.cluster, st.shared::cluster, memory consistency model).
//
// Numerics: same operand precisions as the stock kernel it replaces (bf16 Q, K, V; fp32 scores and accumulators;
// bf16 probabilities); the order of the softmax and output reductions differs.

#pragma once

#include <sgl_kernel/tensor.h>
#include <sgl_kernel/utils.h>

#include <sgl_kernel/utils.cuh>

#include <tvm/ffi/container/tensor.h>

#include <cuda_bf16.h>
#include <cstdint>

namespace muse_vr::draft_attn {

using namespace sglang;

constexpr int kD = 128;           // head dim
constexpr int kPad = 8;           // bf16 elements of row padding in shared tiles (conflict-free ldmatrix)
constexpr int kLd = kD + kPad;    // shared row stride (elements)
constexpr int kKeyBlock = 64;     // keys per inner block
constexpr int kResident = 3;      // key blocks loaded together (one load round covers up to 192 keys)
constexpr int kMaxHalf = kResident * kKeyBlock / 2;  // keys scored by one warp in one pass
constexpr int kWarps = 8;         // 4 row slices of 16 x 2 key halves
constexpr int kThreads = kWarps * 32;

struct Params {
  const bf16_t* q;  // [T, Hq, 128]
  int64_t q_row_stride, q_head_stride;
  const bf16_t* k_cache;
  const bf16_t* v_cache;
  int64_t k_page_stride, k_slot_stride, k_head_stride;
  int64_t v_page_stride, v_slot_stride, v_head_stride;
  const int32_t* page_table;
  int64_t pt_stride;
  const int32_t* seqlens;
  bf16_t* out;  // [T, Hq, 128]
  int64_t o_row_stride, o_head_stride;
  int page_size;
  int num_kv_heads;
  int left;   // < 0: unbounded
  int right;  // < 0: unbounded
  float scale_log2;  // softmax scale * log2(e)
};

SGL_DEVICE uint32_t smem_u32(const void* p) {
  return static_cast<uint32_t>(__cvta_generic_to_shared(p));
}

SGL_DEVICE void cp_async_16(uint32_t dst, const void* src, bool valid) {
  const int n = valid ? 16 : 0;
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;" ::"r"(dst), "l"(src), "r"(n) : "memory");
}

SGL_DEVICE void cp_async_wait_all() {
  asm volatile("cp.async.commit_group;\ncp.async.wait_group 0;" ::: "memory");
}

SGL_DEVICE void ldsm_x4(uint32_t addr, uint32_t& r0, uint32_t& r1, uint32_t& r2, uint32_t& r3) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];"
               : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3)
               : "r"(addr));
}

SGL_DEVICE void ldsm_x4_t(uint32_t addr, uint32_t& r0, uint32_t& r1, uint32_t& r2, uint32_t& r3) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0, %1, %2, %3}, [%4];"
               : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3)
               : "r"(addr));
}

SGL_DEVICE void mma16816(float* d, const uint32_t* a, uint32_t b0, uint32_t b1) {
  asm volatile(
      "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, "
      "{%0, %1, %2, %3};"
      : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}

SGL_DEVICE uint32_t pack_bf16(float lo, float hi) {
  const uint32_t l = __bfloat16_as_ushort(__float2bfloat16_rn(lo));
  const uint32_t h = __bfloat16_as_ushort(__float2bfloat16_rn(hi));
  return l | (h << 16);
}

SGL_DEVICE uint32_t cluster_ctarank() {
  uint32_t r;
  asm volatile("mov.u32 %0, %%cluster_ctarank;" : "=r"(r));
  return r;
}

SGL_DEVICE void cluster_sync_release_acquire() {
  asm volatile("barrier.cluster.arrive.release.aligned;\nbarrier.cluster.wait.acquire.aligned;" ::: "memory");
}

SGL_DEVICE uint32_t map_cta(uint32_t local_addr, uint32_t cta) {
  uint32_t remote;
  asm volatile("mapa.shared::cluster.u32 %0, %1, %2;" : "=r"(remote) : "r"(local_addr), "r"(cta));
  return remote;
}

template <int kQ, int kGroup, int kSplits>
struct Smem {
  static constexpr int kRows = kQ * kGroup;
  static constexpr int kDS = kD / kSplits;  // output columns combined by each CTA
  __nv_bfloat16 q[kRows * kLd];
  union {
    struct {
      __nv_bfloat16 k[kResident * kKeyBlock * kLd];
      __nv_bfloat16 v[kResident * kKeyBlock * kLd];
    } kv;
    struct {  // second key half's partial, handed to the first half's warps
      float o[kRows * kD];
      float m[kRows];
      float l[kRows];
    } half;
  } u;
  float2 rml[kSplits][kRows];      // (row maximum, row sum) of every CTA (pushed by the peers)
  float ro[kSplits][kRows][kDS];   // this CTA's output columns from every CTA
};

template <int kQ, int kGroup, int kSplits, bool kUsePDL>
__global__ void __launch_bounds__(kThreads, 1) draft_attn_kernel(const __grid_constant__ Params p) {
  using S = Smem<kQ, kGroup, kSplits>;
  constexpr int kRows = S::kRows;
  constexpr int kDS = S::kDS;
  static_assert(kRows == 4 * 16, "four 16-row slices");
  static_assert(kDS == 8, "one 8-column fragment tile per CTA");
  extern __shared__ __align__(16) uint8_t smem_raw[];
  S& sm = *reinterpret_cast<S*>(smem_raw);

  const int tid = threadIdx.x, warp = tid / 32, lane = tid % 32;
  const int rs = warp & 3, kh = warp >> 2;
  const int split = static_cast<int>(cluster_ctarank());
  const int cl = static_cast<int>(blockIdx.x) / kSplits;
  const int req = cl / p.num_kv_heads, kvh = cl % p.num_kv_heads;

  asm volatile("barrier.cluster.arrive.relaxed.aligned;" ::: "memory");
  device::PDLWaitPrimary<kUsePDL>();
  const int seqlen = p.seqlens[req];
  const int q0 = seqlen - kQ;  // position of the first query token
  const int kbeg = p.left < 0 ? 0 : max(0, q0 - p.left);
  const int kend = p.right < 0 ? seqlen : min(seqlen, seqlen - 1 + p.right + 1);
  const int span = max(0, kend - kbeg);
  const int per = ((span + kSplits * 16 - 1) / (kSplits * 16)) * 16;
  const int my0 = kbeg + split * per;
  const int my1 = min(kend, my0 + per);

  // Q rows: row r = token (r / kGroup) * kGroup + head (r % kGroup) of this KV head's group
  for (int c = tid; c < kRows * (kD / 8); c += kThreads) {
    const int r = c / (kD / 8), ch = c % (kD / 8);
    const int tok = r / kGroup, hh = r % kGroup;
    const bf16_t* src = p.q + static_cast<int64_t>(req * kQ + tok) * p.q_row_stride +
                        static_cast<int64_t>(kvh * kGroup + hh) * p.q_head_stride + ch * 8;
    cp_async_16(smem_u32(&sm.q[r * kLd + ch * 8]), src, true);
  }

  // per-thread rows: g = lane / 4 and g + 8 of this warp's slice
  const int r_lo = rs * 16 + lane / 4, r_hi = r_lo + 8;
  const int pos_lo = q0 + r_lo / kGroup, pos_hi = q0 + r_hi / kGroup;
  const int lo_lo = p.left < 0 ? 0 : pos_lo - p.left, lo_hi = p.left < 0 ? 0 : pos_hi - p.left;
  const int hi_lo = p.right < 0 ? seqlen - 1 : pos_lo + p.right, hi_hi = p.right < 0 ? seqlen - 1 : pos_hi + p.right;

  float o_acc[kD / 8][4];
#pragma unroll
  for (int n = 0; n < kD / 8; ++n)
#pragma unroll
    for (int e = 0; e < 4; ++e) o_acc[n][e] = 0.f;
  float m_lo = -INFINITY, m_hi = -INFINITY, l_lo = 0.f, l_hi = 0.f;

  for (int r0 = my0; r0 < my1; r0 += 2 * kMaxHalf) {
    // one load round: up to kResident key blocks (and, in the first round, the Q rows issued above)
    const int nr = min(2 * kMaxHalf, my1 - r0);
    if (r0 != my0) __syncthreads();  // previous round's K / V reads are done
    for (int c = tid; c < ((nr + 31) / 32) * 32 * (kD / 8); c += kThreads) {
      const int key = c / (kD / 8), ch = c % (kD / 8);
      const bool ok = key < nr;
      const int jj = ok ? r0 + key : r0;
      const int page = p.page_table[static_cast<int64_t>(req) * p.pt_stride + jj / p.page_size];
      const int slot = jj % p.page_size;
      const bf16_t* ks = p.k_cache + page * p.k_page_stride + slot * p.k_slot_stride + kvh * p.k_head_stride + ch * 8;
      const bf16_t* vs = p.v_cache + page * p.v_page_stride + slot * p.v_slot_stride + kvh * p.v_head_stride + ch * 8;
      cp_async_16(smem_u32(&sm.u.kv.k[key * kLd + ch * 8]), ks, ok);
      cp_async_16(smem_u32(&sm.u.kv.v[key * kLd + ch * 8]), vs, ok);
    }
    cp_async_wait_all();
    __syncthreads();
    {
      // this warp: keys [r0 + kh * half, r0 + kh * half + half) of the round, scored in one pass
      const int half = ((nr + 31) / 32) * 16;  // per key half, multiple of 16, <= kMaxHalf
      const int k0 = r0 + kh * half;
      const int nk = min(half, r0 + nr - k0);  // may be <= 0
      if (nk > 0) {
        const int kbase = (k0 - r0) * kLd;
        const int ng = (nk + 15) / 16;  // 16-key groups present
        float s[kMaxHalf / 8][4];
#pragma unroll
        for (int n = 0; n < kMaxHalf / 8; ++n)
#pragma unroll
          for (int e = 0; e < 4; ++e) s[n][e] = 0.f;
#pragma unroll
        for (int kk = 0; kk < kD / 16; ++kk) {
          uint32_t a[4];
          ldsm_x4(smem_u32(&sm.q[(rs * 16 + (lane % 16)) * kLd + kk * 16 + (lane / 16) * 8]), a[0], a[1], a[2], a[3]);
#pragma unroll
          for (int n2 = 0; n2 < kMaxHalf / 16; ++n2) {
            if (n2 < ng) {
              uint32_t b0, b1, b2, b3;
              const int key = n2 * 16 + (lane % 8) + (lane / 16) * 8;
              ldsm_x4(smem_u32(&sm.u.kv.k[kbase + key * kLd + kk * 16 + ((lane / 8) % 2) * 8]), b0, b1, b2, b3);
              mma16816(s[2 * n2], a, b0, b1);
              mma16816(s[2 * n2 + 1], a, b2, b3);
            }
          }
        }
        float bm_lo = -INFINITY, bm_hi = -INFINITY;
#pragma unroll
        for (int n = 0; n < kMaxHalf / 8; ++n) {
#pragma unroll
          for (int e = 0; e < 4; ++e) {
            const int key = n * 8 + 2 * (lane % 4) + (e % 2);
            const int j = k0 + key;
            const bool hi_row = e >= 2;
            const bool ok = key < nk && j >= (hi_row ? lo_hi : lo_lo) && j <= (hi_row ? hi_hi : hi_lo);
            s[n][e] = ok ? s[n][e] * p.scale_log2 : -INFINITY;
            if (hi_row)
              bm_hi = fmaxf(bm_hi, s[n][e]);
            else
              bm_lo = fmaxf(bm_lo, s[n][e]);
          }
        }
#pragma unroll
        for (int m = 1; m <= 2; m *= 2) {
          bm_lo = fmaxf(bm_lo, __shfl_xor_sync(0xffffffffu, bm_lo, m));
          bm_hi = fmaxf(bm_hi, __shfl_xor_sync(0xffffffffu, bm_hi, m));
        }
        const float mn_lo = fmaxf(m_lo, bm_lo), mn_hi = fmaxf(m_hi, bm_hi);
        const float sc_lo = mn_lo == -INFINITY ? 1.f : exp2f(m_lo - mn_lo);
        const float sc_hi = mn_hi == -INFINITY ? 1.f : exp2f(m_hi - mn_hi);
        m_lo = mn_lo;
        m_hi = mn_hi;
        float bl_lo = 0.f, bl_hi = 0.f;
#pragma unroll
        for (int n = 0; n < kMaxHalf / 8; ++n) {
#pragma unroll
          for (int e = 0; e < 4; ++e) {
            const bool hi_row = e >= 2;
            const float mm = hi_row ? m_hi : m_lo;
            const float pv = (s[n][e] == -INFINITY) ? 0.f : exp2f(s[n][e] - mm);
            s[n][e] = pv;
            if (hi_row)
              bl_hi += pv;
            else
              bl_lo += pv;
          }
        }
#pragma unroll
        for (int m = 1; m <= 2; m *= 2) {
          bl_lo += __shfl_xor_sync(0xffffffffu, bl_lo, m);
          bl_hi += __shfl_xor_sync(0xffffffffu, bl_hi, m);
        }
        l_lo = l_lo * sc_lo + bl_lo;
        l_hi = l_hi * sc_hi + bl_hi;
#pragma unroll
        for (int n = 0; n < kD / 8; ++n) {
          o_acc[n][0] *= sc_lo;
          o_acc[n][1] *= sc_lo;
          o_acc[n][2] *= sc_hi;
          o_acc[n][3] *= sc_hi;
        }
#pragma unroll
        for (int kj = 0; kj < kMaxHalf / 16; ++kj) {
          if (kj < ng) {
            uint32_t a[4];
            a[0] = pack_bf16(s[2 * kj][0], s[2 * kj][1]);
            a[1] = pack_bf16(s[2 * kj][2], s[2 * kj][3]);
            a[2] = pack_bf16(s[2 * kj + 1][0], s[2 * kj + 1][1]);
            a[3] = pack_bf16(s[2 * kj + 1][2], s[2 * kj + 1][3]);
#pragma unroll
            for (int n2 = 0; n2 < kD / 16; ++n2) {
              uint32_t b0, b1, b2, b3;
              const int key = kj * 16 + (lane % 8) + ((lane / 8) % 2) * 8;
              ldsm_x4_t(smem_u32(&sm.u.kv.v[kbase + key * kLd + n2 * 16 + (lane / 16) * 8]), b0, b1, b2, b3);
              mma16816(o_acc[2 * n2], a, b0, b1);
              mma16816(o_acc[2 * n2 + 1], a, b2, b3);
            }
          }
        }
      }
    }
  }

  // combine the two key halves (second half -> shared memory -> first half)
  __syncthreads();  // K / V no longer read; the union's other member is written below
  if (kh == 1) {
    if (lane % 4 == 0) {
      sm.u.half.m[r_lo] = m_lo;
      sm.u.half.m[r_hi] = m_hi;
      sm.u.half.l[r_lo] = l_lo;
      sm.u.half.l[r_hi] = l_hi;
    }
#pragma unroll
    for (int n = 0; n < kD / 8; ++n) {
      const int col = n * 8 + 2 * (lane % 4);
      *reinterpret_cast<float2*>(&sm.u.half.o[r_lo * kD + col]) = make_float2(o_acc[n][0], o_acc[n][1]);
      *reinterpret_cast<float2*>(&sm.u.half.o[r_hi * kD + col]) = make_float2(o_acc[n][2], o_acc[n][3]);
    }
  }
  __syncthreads();
  asm volatile("barrier.cluster.wait.aligned;" ::: "memory");  // every CTA of the cluster is running
  if (kh == 0) {
    const float m1_lo = sm.u.half.m[r_lo], m1_hi = sm.u.half.m[r_hi];
    const float M_lo = fmaxf(m_lo, m1_lo), M_hi = fmaxf(m_hi, m1_hi);
    const float a0_lo = M_lo == -INFINITY ? 0.f : exp2f(m_lo - M_lo), a1_lo = M_lo == -INFINITY ? 0.f : exp2f(m1_lo - M_lo);
    const float a0_hi = M_hi == -INFINITY ? 0.f : exp2f(m_hi - M_hi), a1_hi = M_hi == -INFINITY ? 0.f : exp2f(m1_hi - M_hi);
    const float L_lo = l_lo * a0_lo + sm.u.half.l[r_lo] * a1_lo;
    const float L_hi = l_hi * a0_hi + sm.u.half.l[r_hi] * a1_hi;
#pragma unroll
    for (int n = 0; n < kD / 8; ++n) {
      const int col = n * 8 + 2 * (lane % 4);
      const float2 h_lo = *reinterpret_cast<const float2*>(&sm.u.half.o[r_lo * kD + col]);
      const float2 h_hi = *reinterpret_cast<const float2*>(&sm.u.half.o[r_hi * kD + col]);
      // column tile n goes to CTA n (kDS == 8)
      const uint32_t dlo = map_cta(smem_u32(&sm.ro[split][r_lo][2 * (lane % 4)]), n);
      const uint32_t dhi = map_cta(smem_u32(&sm.ro[split][r_hi][2 * (lane % 4)]), n);
      const float x0 = o_acc[n][0] * a0_lo + h_lo.x * a1_lo, x1 = o_acc[n][1] * a0_lo + h_lo.y * a1_lo;
      const float x2 = o_acc[n][2] * a0_hi + h_hi.x * a1_hi, x3 = o_acc[n][3] * a0_hi + h_hi.y * a1_hi;
      asm volatile("st.shared::cluster.v2.f32 [%0], {%1, %2};" ::"r"(dlo), "f"(x0), "f"(x1) : "memory");
      asm volatile("st.shared::cluster.v2.f32 [%0], {%1, %2};" ::"r"(dhi), "f"(x2), "f"(x3) : "memory");
    }
#pragma unroll
    for (int c = lane % 4; c < kSplits; c += 4) {
      asm volatile("st.shared::cluster.v2.f32 [%0], {%1, %2};" ::"r"(map_cta(smem_u32(&sm.rml[split][r_lo]), c)),
                   "f"(M_lo), "f"(L_lo)
                   : "memory");
      asm volatile("st.shared::cluster.v2.f32 [%0], {%1, %2};" ::"r"(map_cta(smem_u32(&sm.rml[split][r_hi]), c)),
                   "f"(M_hi), "f"(L_hi)
                   : "memory");
    }
  }
  cluster_sync_release_acquire();

  // normalize this CTA's column slice of every row from its own shared memory
  for (int it = tid; it < kRows * (kDS / 2); it += kThreads) {
    const int r = it / (kDS / 2), c2 = (it % (kDS / 2)) * 2;
    float mx = -INFINITY;
#pragma unroll
    for (int c = 0; c < kSplits; ++c) mx = fmaxf(mx, sm.rml[c][r].x);
    float den = 0.f, ax = 0.f, ay = 0.f;
#pragma unroll
    for (int c = 0; c < kSplits; ++c) {
      const float2 ml = sm.rml[c][r];
      const float w = ml.x == -INFINITY ? 0.f : exp2f(ml.x - mx);
      den += w * ml.y;
      ax += w * sm.ro[c][r][c2];
      ay += w * sm.ro[c][r][c2 + 1];
    }
    const float inv = 1.f / den;
    const int tok = r / kGroup, hh = r % kGroup;
    bf16_t* dst = p.out + static_cast<int64_t>(req * kQ + tok) * p.o_row_stride +
                  static_cast<int64_t>(kvh * kGroup + hh) * p.o_head_stride + split * kDS + c2;
    *reinterpret_cast<uint32_t*>(dst) = pack_bf16(ax * inv, ay * inv);
  }
  device::PDLTriggerSecondary<kUsePDL>();
}

template <int kQ, int kGroup, int kSplits, bool kUsePDL>
struct DraftAttn {
  static void run(const tvm::ffi::TensorView q, const tvm::ffi::TensorView k_cache, const tvm::ffi::TensorView v_cache,
                  const tvm::ffi::TensorView page_table, const tvm::ffi::TensorView seqlens,
                  const tvm::ffi::TensorView out, int64_t left, int64_t right, double scale) {
    using namespace host;
    using S = Smem<kQ, kGroup, kSplits>;
    auto T = SymbolicSize{"t"};
    auto HQ = SymbolicSize{"hq"};
    auto B = SymbolicSize{"b"};
    auto device_ = SymbolicDevice{};
    device_.set_options<kDLCUDA>();
    TensorMatcher({T, HQ, kD}).with_dtype<bf16_t>().with_device(device_).verify(q);
    TensorMatcher({T, HQ, kD}).with_dtype<bf16_t>().with_device(device_).verify(out);
    TensorMatcher({B}).with_dtype<int32_t>().with_device(device_).verify(seqlens);
    RuntimeCheck(q.stride(2) == 1 && out.stride(2) == 1, "draft_attn: head_dim must be contiguous");
    RuntimeCheck(k_cache.dim() == 4 && v_cache.dim() == 4 && k_cache.size(3) == kD && k_cache.stride(3) == 1 &&
                     v_cache.stride(3) == 1,
                 "draft_attn: paged cache [pages, page, kv_heads, 128] expected");
    RuntimeCheck(page_table.dim() == 2 && page_table.stride(1) == 1, "draft_attn: page table [b, pages] expected");
    const int64_t t = T.unwrap(), b = B.unwrap(), hq = HQ.unwrap();
    const int64_t kvh = k_cache.size(2);
    RuntimeCheck(t == b * kQ, "draft_attn: every request must have the same query count");
    RuntimeCheck(hq == kvh * kGroup, "draft_attn: query heads must be kGroup per KV head");
    Params p{};
    p.q = static_cast<const bf16_t*>(q.data_ptr());
    p.q_row_stride = q.stride(0);
    p.q_head_stride = q.stride(1);
    p.k_cache = static_cast<const bf16_t*>(k_cache.data_ptr());
    p.v_cache = static_cast<const bf16_t*>(v_cache.data_ptr());
    p.k_page_stride = k_cache.stride(0);
    p.k_slot_stride = k_cache.stride(1);
    p.k_head_stride = k_cache.stride(2);
    p.v_page_stride = v_cache.stride(0);
    p.v_slot_stride = v_cache.stride(1);
    p.v_head_stride = v_cache.stride(2);
    p.page_table = static_cast<const int32_t*>(page_table.data_ptr());
    p.pt_stride = page_table.stride(0);
    p.seqlens = static_cast<const int32_t*>(seqlens.data_ptr());
    p.out = static_cast<bf16_t*>(out.data_ptr());
    p.o_row_stride = out.stride(0);
    p.o_head_stride = out.stride(1);
    p.page_size = static_cast<int>(k_cache.size(1));
    p.num_kv_heads = static_cast<int>(kvh);
    p.left = static_cast<int>(left);
    p.right = static_cast<int>(right);
    p.scale_log2 = static_cast<float>(scale * 1.4426950408889634);
    const auto dev = device_.unwrap();
    auto kernel = draft_attn_kernel<kQ, kGroup, kSplits, kUsePDL>;
    static bool configured = false;
    if (!configured) {
      RuntimeDeviceCheck(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, sizeof(S)));
      if (kSplits > 8) RuntimeDeviceCheck(cudaFuncSetAttribute(kernel, cudaFuncAttributeNonPortableClusterSizeAllowed, 1));
      configured = true;
    }
    auto lk = LaunchKernel(static_cast<unsigned>(b * kvh * kSplits), kThreads, dev, sizeof(S));
    lk.enable_pdl(kUsePDL);
    lk.enable_cluster(dim3(kSplits, 1, 1));
    lk(kernel, p);
  }
};

}  // namespace muse_vr::draft_attn
