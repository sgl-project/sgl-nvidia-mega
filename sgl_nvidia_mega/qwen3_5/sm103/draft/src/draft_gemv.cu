// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// Adapted from SGLang (Apache-2.0): python/sglang/kernels/jit/csrc/speculative/mkd_draft_gemv.cuh
// (branch nvidia/qwen35-agentx-gb300).
// One-row draft GEMV y[n] = bf16(sum_k W[n, k] * x[k]): bf16 operands, fp32 accumulation, output rounded
// to bf16 as the served GEMM writes it (only the fp32 summation order differs). One warp per row with the
// whole row in flight, the row loaded before the PDL wait (weights are static). With a hop site, every CTA
// owns one 8-row fragment of the output and pushes it into the decode hop (sm103/hop) before storing it. With a
// gate, the input is x * sigmoid(gate) rounded to bf16, as SGLang's attention output gate kernel writes it.
// The q35hop namespace below is hop.cu's, copied verbatim (the push must follow its mailbox protocol;
// test/kernels/test_qwen3_5_draft_hop.py compares the text).
//
// Built with NVRTC per reduction length Q35_DRAFT_GEMV_K: C-linkage entries q35_draft_gemv (4 rows a
// CTA) and q35_draft_gemv_push (8 rows a CTA, pushing), and their gated twins q35_draft_gemv_gate and
// q35_draft_gemv_push_gate.

#include <cuda_bf16.h>

// NVRTC has no <stdint.h>
typedef unsigned char uint8_t;
typedef unsigned short uint16_t;
typedef int int32_t;
typedef unsigned int uint32_t;
typedef unsigned long long uint64_t;

#ifndef Q35_HOP_DEADLINE_NS
#define Q35_HOP_DEADLINE_NS 0ull
#endif

namespace q35hop {


constexpr int kHidden = 4096;
constexpr int kVec = 8;
constexpr int kFrags = kHidden / kVec;
constexpr int kStages = 3;
constexpr int kRowCtas = 4;
constexpr int kRowThreads = 128;
constexpr int kRowWarps = kRowThreads / 32;
constexpr uint32_t kNegZeroPair = 0x80008000u;
static_assert(kRowCtas * kRowThreads == kFrags, "a row is one fragment per thread of the cluster");
static_assert(kRowWarps == 4, "row_sumsq_fold folds exactly four warp sums per CTA");

// Per-process control block in device memory.
struct Ctrl {
  uint32_t stage;   // stage of the next hop
  uint32_t arrive;  // CTA tickets of the running consumer launch
  uint32_t seq;     // hops completed
  uint32_t pad;
};

// Passed by value as a kernel parameter.
struct Site {
  void* mc;                     // multicast base of the mailbox [kStages][world][m_cap][kHidden] bf16
  void* uc;                     // this rank's unicast view of the same mailbox
  Ctrl* ctrl;
  int rank, world, m_cap;
};

// ---------------------------------------------------------------- PTX

__device__ __forceinline__ void pdl_wait() { asm volatile("griddepcontrol.wait;" ::: "memory"); }
__device__ __forceinline__ void pdl_trigger() { asm volatile("griddepcontrol.launch_dependents;" ::: "memory"); }
__device__ __forceinline__ unsigned long long gtimer() {
  unsigned long long t;
  asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
  return t;
}
__device__ __forceinline__ uint32_t ld_volatile_u32(const uint32_t* p) {
  uint32_t v;
  asm volatile("ld.volatile.global.u32 %0, [%1];" : "=r"(v) : "l"(p) : "memory");
  return v;
}
__device__ __forceinline__ uint4 ld_volatile_v4(const void* p) {
  uint4 v;
  asm volatile("ld.volatile.global.v4.u32 {%0,%1,%2,%3}, [%4];"
               : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w)
               : "l"(p)
               : "memory");
  return v;
}
__device__ __forceinline__ void st_v4(void* p, uint4 v) {
  asm volatile("st.global.v4.u32 [%0], {%1,%2,%3,%4};" ::"l"(p), "r"(v.x), "r"(v.y), "r"(v.z), "r"(v.w)
               : "memory");
}
__device__ __forceinline__ void mc_st_v4(void* mc, uint4 v) {
  asm volatile("multimem.st.relaxed.sys.global.v4.bf16x2 [%0], {%1,%2,%3,%4};" ::"l"(mc), "r"(v.x),
               "r"(v.y), "r"(v.z), "r"(v.w)
               : "memory");
}
__device__ __forceinline__ void mc_st_v2(void* mc, uint2 v) {
  asm volatile("multimem.st.relaxed.sys.global.v2.bf16x2 [%0], {%1,%2};" ::"l"(mc), "r"(v.x), "r"(v.y) : "memory");
}
__device__ __forceinline__ uint32_t cluster_ctarank() {
  uint32_t r;
  asm volatile("mov.u32 %0, %%cluster_ctarank;" : "=r"(r));
  return r;
}
__device__ __forceinline__ float div_full(float a, float b) {
  float r;
  asm("div.full.f32 %0, %1, %2;" : "=f"(r) : "f"(a), "f"(b));
  return r;
}
// Two f32 to an e4m3 pair; lo lands in the low byte.
__device__ __forceinline__ uint16_t cvt_e4m3x2(float lo, float hi) {
  uint16_t r;
  asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(r) : "f"(hi), "f"(lo));
  return r;
}

// ---------------------------------------------------------------- bf16

__device__ __forceinline__ float bf_lo(uint32_t w) { return __uint_as_float(w << 16); }
__device__ __forceinline__ float bf_hi(uint32_t w) { return __uint_as_float(w & 0xFFFF0000u); }
__device__ __forceinline__ uint32_t pack_bf16x2(float lo, float hi) {
  __nv_bfloat162 v = __floats2bfloat162_rn(lo, hi);
  return *reinterpret_cast<uint32_t*>(&v);
}
__device__ __forceinline__ void unpack8(uint4 v, float f[8]) {
  f[0] = bf_lo(v.x); f[1] = bf_hi(v.x); f[2] = bf_lo(v.y); f[3] = bf_hi(v.y);
  f[4] = bf_lo(v.z); f[5] = bf_hi(v.z); f[6] = bf_lo(v.w); f[7] = bf_hi(v.w);
}
__device__ __forceinline__ uint4 pack8(const float f[8]) {
  return make_uint4(pack_bf16x2(f[0], f[1]), pack_bf16x2(f[2], f[3]), pack_bf16x2(f[4], f[5]),
                    pack_bf16x2(f[6], f[7]));
}
__device__ __forceinline__ bool word_has_negzero(uint32_t w) {
  return (w & 0xFFFFu) == 0x8000u || (w >> 16) == 0x8000u;
}
__device__ __forceinline__ bool frag_has_negzero(uint4 v) {
  return word_has_negzero(v.x) | word_has_negzero(v.y) | word_has_negzero(v.z) | word_has_negzero(v.w);
}
__device__ __forceinline__ uint32_t sanitize_word(uint32_t w) {
  if ((w & 0xFFFFu) == 0x8000u) w &= 0xFFFF0000u;
  if ((w >> 16) == 0x8000u) w &= 0x0000FFFFu;
  return w;
}

// ---------------------------------------------------------------- stage protocol

__device__ __forceinline__ size_t slot(const Site& s, uint32_t stage, int src, int token, int frag) {
  return ((static_cast<size_t>(stage) * s.world + src) * s.m_cap + token) * kHidden +
         static_cast<size_t>(frag) * kVec;
}

struct Hop {
  uint32_t stage;
  uint32_t seq;
  uint32_t ticket;  // thread 0 only
};

// Consumer launch, every thread of every participating CTA, after griddepcontrol.wait. Thread
// 0 takes a ticket whose value is consumed only in hop_leave, so the atomic stays off the
// critical path.
__device__ __forceinline__ Hop hop_enter(const Site& s) {
  Hop h;
  h.stage = ld_volatile_u32(&s.ctrl->stage);
  h.seq = ld_volatile_u32(&s.ctrl->seq);
  h.ticket = threadIdx.x == 0 ? atomicAdd(&s.ctrl->arrive, 1u) : 0u;
  return h;
}

// Thread 0 of every participating CTA; the last arrival advances the stage. No CTA of this
// launch can read the advanced value, since every CTA read the stage before taking its ticket.
__device__ __forceinline__ void hop_leave(const Site& s, const Hop& h, uint32_t participants) {
  if (threadIdx.x == 0 && h.ticket == participants - 1) {
    s.ctrl->arrive = 0;
    s.ctrl->stage = (h.stage + 1) % kStages;
    s.ctrl->seq = h.seq + 1;
  }
}

// Producer-only launch, after griddepcontrol.wait.
__device__ __forceinline__ uint32_t stage_peek(const Site& s) { return ld_volatile_u32(&s.ctrl->stage); }

// ---------------------------------------------------------------- producer

// This rank's final bf16 partial for one fragment; exactly once per (hop, rank, token, frag).
// A launch that also stores its own outputs should push first: the push starts the flight.
__device__ __forceinline__ void push_frag(const Site& s, uint32_t stage, int token, int frag, uint4 v) {
  v = make_uint4(sanitize_word(v.x), sanitize_word(v.y), sanitize_word(v.z), sanitize_word(v.w));
  mc_st_v4(static_cast<uint16_t*>(s.mc) + slot(s, stage, s.rank, token, frag), v);
}

// Same for 4 consecutive bf16 (half a fragment, columns [4q, 4q + 4)); a fragment may be pushed
// as two halves by different threads.
__device__ __forceinline__ void push_quad(const Site& s, uint32_t stage, int token, int quad, uint2 v) {
  v = make_uint2(sanitize_word(v.x), sanitize_word(v.y));
  mc_st_v2(static_cast<uint16_t*>(s.mc) + slot(s, stage, s.rank, token, quad >> 1) + (quad & 1) * 4, v);
}

// ---------------------------------------------------------------- consumer

// A peer that missed the wait deadline: store to address 0x0, then trap (fail-stop).
__device__ __forceinline__ void fault_at_null() {
  asm volatile("{\n.reg .u64 a;\nmov.u64 a, 0;\nst.volatile.global.u32 [a], 0;\ntrap;\n}" ::: "memory");
}

// Sum over ranks 0..W-1 (fp32, from 0) of one fragment; spins on the sentinel. A wait past
// Q35_HOP_DEADLINE_NS (if not 0) stops the kernel.
template <int W>
__device__ __forceinline__ bool pull_frag(const Site& s, uint32_t stage, int token, int frag, float acc[8]) {
  const uint16_t* uc = static_cast<const uint16_t*>(s.uc);
  uint4 v[W];
#pragma unroll
  for (int r = 0; r < W; ++r) v[r] = ld_volatile_v4(uc + slot(s, stage, r, token, frag));
  uint32_t pending = 0;
#pragma unroll
  for (int r = 0; r < W; ++r) pending |= frag_has_negzero(v[r]) ? (1u << r) : 0u;
  unsigned long long t0 = 0;
  uint32_t spins = 0;
  while (pending) {
#pragma unroll
    for (int r = 0; r < W; ++r)
      if (pending & (1u << r)) v[r] = ld_volatile_v4(uc + slot(s, stage, r, token, frag));
    uint32_t still = 0;
#pragma unroll
    for (int r = 0; r < W; ++r)
      if ((pending & (1u << r)) && frag_has_negzero(v[r])) still |= 1u << r;
    pending = still;
    if (Q35_HOP_DEADLINE_NS != 0 && pending && ((++spins & 255u) == 0)) {
      const unsigned long long now = gtimer();
      if (t0 == 0) {
        t0 = now;
      } else if (now - t0 > Q35_HOP_DEADLINE_NS) {
        fault_at_null();
      }
    }
  }
#pragma unroll
  for (int e = 0; e < 8; ++e) acc[e] = 0.f;
#pragma unroll
  for (int r = 0; r < W; ++r) {
    float f[8];
    unpack8(v[r], f);
#pragma unroll
    for (int e = 0; e < 8; ++e) acc[e] = __fadd_rn(acc[e], f[e]);
  }
  return true;
}

// Write the sentinel back into the W slots this thread consumed (any time after pull_frag).
template <int W>
__device__ __forceinline__ void rearm_frag(const Site& s, uint32_t stage, int token, int frag) {
  uint16_t* uc = static_cast<uint16_t*>(s.uc);
  const uint4 z = make_uint4(kNegZeroPair, kNegZeroPair, kNegZeroPair, kNegZeroPair);
#pragma unroll
  for (int r = 0; r < W; ++r) st_v4(uc + slot(s, stage, r, token, frag), z);
}

// ---------------------------------------------------------------- row finish

// prenorm = bf16(acc + residual); pre[] gets the bf16-rounded values the norm uses.
__device__ __forceinline__ uint4 residual_add(const float acc[8], uint4 residual, float pre[8]) {
  float r[8];
  unpack8(residual, r);
#pragma unroll
  for (int e = 0; e < 8; ++e) pre[e] = __fadd_rn(acc[e], r[e]);
  const uint4 out = pack8(pre);
  unpack8(out, pre);
  return out;
}

__device__ __forceinline__ float sumsq8(const float x[8]) {
  float s = 0.f;
#pragma unroll
  for (int e = 0; e < 8; ++e) s = __fadd_rn(s, __fmul_rn(x[e], x[e]));
  return s;
}

struct RowSmem {
  float xs[kRowCtas][kRowWarps];
  uint64_t mbar;
};

// Prologue of every CTA of the row cluster, all threads, before griddepcontrol.wait: arm the
// exchange barrier and make it visible to the peers before any of them can send.
__device__ __forceinline__ void row_init(RowSmem& rs) {
  if (threadIdx.x == 0) {
    const uint32_t a = static_cast<uint32_t>(__cvta_generic_to_shared(&rs.mbar));
    asm volatile("mbarrier.init.shared::cta.b64 [%0], 1;" ::"r"(a) : "memory");
    asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;" ::"r"(a),
                 "r"(static_cast<uint32_t>(sizeof(rs.xs)))
                 : "memory");
    asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
  }
  asm volatile("barrier.cluster.arrive.relaxed.aligned;\n\tbarrier.cluster.wait.acquire.aligned;" ::: "memory");
}

// Row total of the per-thread values: lane butterfly per warp, then each warp sends its sum to
// every CTA of the cluster with st.async; each CTA waits on its own mbarrier (no release fence,
// so this thread's earlier remote stores are not waited for) and folds per CTA as
// (w0 + w2) + (w1 + w3), then the CTAs in rank order from 0.
__device__ __forceinline__ float row_sumsq_fold(float v, RowSmem& rs) {
  const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
#pragma unroll
  for (int off = 1; off < 32; off <<= 1) v = __fadd_rn(v, __shfl_xor_sync(0xffffffffu, v, off));
  if (lane < kRowCtas) {
    const uint32_t me = cluster_ctarank();
    const uint32_t slot_l = static_cast<uint32_t>(__cvta_generic_to_shared(&rs.xs[me][warp]));
    const uint32_t mbar_l = static_cast<uint32_t>(__cvta_generic_to_shared(&rs.mbar));
    uint32_t slot_r, mbar_r;
    asm volatile("mapa.shared::cluster.u32 %0, %1, %2;" : "=r"(slot_r) : "r"(slot_l), "r"(lane));
    asm volatile("mapa.shared::cluster.u32 %0, %1, %2;" : "=r"(mbar_r) : "r"(mbar_l), "r"(lane));
    asm volatile("st.async.shared::cluster.mbarrier::complete_tx::bytes.f32 [%0], %1, [%2];" ::"r"(slot_r),
                 "f"(v), "r"(mbar_r)
                 : "memory");
  }
  {
    const uint32_t a = static_cast<uint32_t>(__cvta_generic_to_shared(&rs.mbar));
    asm volatile(
        "{\n\t.reg .pred P;\n\tWAIT_ROW:\n\t"
        "mbarrier.try_wait.parity.acquire.cluster.shared::cta.b64 P, [%0], 0;\n\t"
        "@!P bra WAIT_ROW;\n\t}" ::"r"(a)
        : "memory");
  }
  float total = 0.f;
#pragma unroll
  for (int c = 0; c < kRowCtas; ++c)
    total = __fadd_rn(total, __fadd_rn(__fadd_rn(rs.xs[c][0], rs.xs[c][2]), __fadd_rn(rs.xs[c][1], rs.xs[c][3])));
  return total;
}

__device__ __forceinline__ float inv_rms(float row_sumsq, float eps) {
  return rsqrtf(__fadd_rn(row_sumsq / static_cast<float>(kHidden), eps));
}

// bf16((x * inv) * gamma), gamma being the applied multiplier.
__device__ __forceinline__ uint4 norm_apply(const float pre[8], float inv, const float gamma[8]) {
  float out[8];
#pragma unroll
  for (int e = 0; e < 8; ++e) out[e] = __fmul_rn(__fmul_rn(pre[e], inv), gamma[e]);
  return pack8(out);
}

// The reciprocal the static per-tensor FP8 quant multiplies by.
__device__ __forceinline__ float inv_scale_of(float scale) { return div_full(1.f, scale); }

// e4m3(clamp(x * inv_scale, +-448)) of the bf16 norm output, 8 bytes.
__device__ __forceinline__ uint2 quant_fp8(uint4 norm_bf16, float inv_scale) {
  float q[8];
  unpack8(norm_bf16, q);
  uint32_t w[4];
#pragma unroll
  for (int j = 0; j < 4; ++j) {
    const float a = fminf(fmaxf(__fmul_rn(q[2 * j], inv_scale), -448.f), 448.f);
    const float b = fminf(fmaxf(__fmul_rn(q[2 * j + 1], inv_scale), -448.f), 448.f);
    w[j] = cvt_e4m3x2(a, b);
  }
  return make_uint2(w[0] | (w[1] << 16), w[2] | (w[3] << 16));
}

}  // namespace q35hop

typedef long long int64_t;

#ifndef Q35_DRAFT_GEMV_K
#define Q35_DRAFT_GEMV_K 1024
#endif

namespace q35draft_gemv {

struct Params {
  const __nv_bfloat16* x;               // [K], written by the previous grid
  const __nv_bfloat16* gate;            // kGate: [K], applied to x as it is staged
  const __nv_bfloat16* __restrict__ w;  // [N, K]
  __nv_bfloat16* y;                     // [N]
  int32_t n;
  q35hop::Site hop;                     // kPush: the hop the fragments go to
  int32_t hop_token;                    // kPush: the output's row in the hop
};

// SGLang's device::warp::reduce_sum over the full warp: lane masks 16 down to 1, plain fp32 adds
__device__ __forceinline__ float warp_reduce_sum(float value) {
#pragma unroll
  for (uint32_t mask = 16; mask >= 1; mask >>= 1) value = value + __shfl_xor_sync(0xffffffffu, value, mask, 32);
  return value;
}

// Eight bf16(x * sigmoid(g)) in the instruction sequence Triton emits for _fused_sigmoid_mul_kernel (fp32,
// exp(-g) as ex2.approx of -g * log2(e), 1 / (1 + e) by div.full, product rounded to nearest-even bf16).
__device__ __forceinline__ uint4 gate_mul8(const uint4 xv, const uint4 gv) {
  const __nv_bfloat16* x = reinterpret_cast<const __nv_bfloat16*>(&xv);
  const __nv_bfloat16* g = reinterpret_cast<const __nv_bfloat16*>(&gv);
  uint4 out;
  __nv_bfloat16* o = reinterpret_cast<__nv_bfloat16*>(&out);
#pragma unroll
  for (int e = 0; e < 8; ++e) {
    float t;
    asm("mul.rn.f32 %0, %1, 0fBFB8AA3B;\n\t"
        "ex2.approx.f32 %0, %0;\n\t"
        "add.rn.f32 %0, %0, 0f3F800000;\n\t"
        "div.full.f32 %0, 0f3F800000, %0;\n\t"
        "mul.rn.f32 %0, %2, %0;"
        : "=&f"(t)
        : "f"(__bfloat162float(g[e])), "f"(__bfloat162float(x[e])));
    o[e] = __float2bfloat16_rn(t);
  }
  return out;
}

/**
 * \brief kRows warps per CTA; warp w of CTA c takes rows kRows * c + w, then a grid stride further on, the
 * next row loaded while the current one is reduced (as the draft head does).
 * \tparam kK    reduction length (a multiple of 256)
 * \tparam kRows warps per CTA; 8 when pushing, with one row per warp so a CTA owns one hop fragment
 * \tparam kGate stage x * sigmoid(gate) instead of x
 */
template <int kK, int kRows, bool kUsePDL, bool kPush, bool kGate>
__device__ __forceinline__ void mkd_gemv_kernel(const Params& p) {
  static_assert(kK % 256 == 0, "K must tile 32 lanes x 8 bf16");
  static_assert(!kPush || kRows == 8, "a pushing CTA owns exactly one 8-row fragment");
  constexpr int kVecs = kK / 8 / 32;
  __shared__ alignas(16) __nv_bfloat16 s_x[kK];
  __shared__ alignas(16) __nv_bfloat16 s_y[kRows];
  __shared__ uint32_t s_stage;
  const int warp = static_cast<int>(threadIdx.x / 32);
  const int lane = static_cast<int>(threadIdx.x % 32);
  const int stride = static_cast<int>(gridDim.x) * kRows;
  int row = static_cast<int>(blockIdx.x) * kRows + warp;
  uint4 wv[kVecs];
  if (row < p.n) {
    const uint4* wr = reinterpret_cast<const uint4*>(p.w + static_cast<int64_t>(row) * kK);
#pragma unroll
    for (int j = 0; j < kVecs; ++j) wv[j] = __ldg(wr + j * 32 + lane);
  }
  if (kUsePDL) q35hop::pdl_wait();
  for (int i = static_cast<int>(threadIdx.x); i < kK / 8; i += kRows * 32) {
    uint4 xv = reinterpret_cast<const uint4*>(p.x)[i];
    if constexpr (kGate) xv = gate_mul8(xv, reinterpret_cast<const uint4*>(p.gate)[i]);
    reinterpret_cast<uint4*>(s_x)[i] = xv;
  }
  if constexpr (kPush) {
    if (threadIdx.x == 0) s_stage = q35hop::stage_peek(p.hop);
  }
  __syncthreads();
  for (; row < p.n; row += stride) {
    float acc = 0.0f;
#pragma unroll
    for (int j = 0; j < kVecs; ++j) {
      const uint4 xv = reinterpret_cast<const uint4*>(s_x)[j * 32 + lane];
      const __nv_bfloat16* a = reinterpret_cast<const __nv_bfloat16*>(&wv[j]);
      const __nv_bfloat16* b = reinterpret_cast<const __nv_bfloat16*>(&xv);
#pragma unroll
      for (int e = 0; e < 8; ++e) acc = fmaf(__bfloat162float(a[e]), __bfloat162float(b[e]), acc);
    }
    const int next = row + stride;
    if (next < p.n) {
      const uint4* wr = reinterpret_cast<const uint4*>(p.w + static_cast<int64_t>(next) * kK);
#pragma unroll
      for (int j = 0; j < kVecs; ++j) wv[j] = __ldg(wr + j * 32 + lane);
    }
    acc = warp_reduce_sum(acc);
    const __nv_bfloat16 v = __float2bfloat16_rn(acc);
    if constexpr (kPush) {
      if (lane == 0) s_y[warp] = v;
      __syncthreads();
      // Push before the store: the push starts the flight to every rank.
      if (threadIdx.x == 0)
        q35hop::push_frag(p.hop, s_stage, p.hop_token, static_cast<int>(blockIdx.x),
                          *reinterpret_cast<const uint4*>(s_y));
    }
    if (lane == 0) p.y[row] = v;
  }
  if (kUsePDL) q35hop::pdl_trigger();
}

}  // namespace q35draft_gemv

using q35draft_gemv::Params;

extern "C" __global__ void __launch_bounds__(4 * 32) q35_draft_gemv(const Params __grid_constant__ p) {
  q35draft_gemv::mkd_gemv_kernel<Q35_DRAFT_GEMV_K, 4, true, false, false>(p);
}

extern "C" __global__ void __launch_bounds__(8 * 32) q35_draft_gemv_push(const Params __grid_constant__ p) {
  q35draft_gemv::mkd_gemv_kernel<Q35_DRAFT_GEMV_K, 8, true, true, false>(p);
}

extern "C" __global__ void __launch_bounds__(4 * 32) q35_draft_gemv_gate(const Params __grid_constant__ p) {
  q35draft_gemv::mkd_gemv_kernel<Q35_DRAFT_GEMV_K, 4, true, false, true>(p);
}

extern "C" __global__ void __launch_bounds__(8 * 32) q35_draft_gemv_push_gate(const Params __grid_constant__ p) {
  q35draft_gemv::mkd_gemv_kernel<Q35_DRAFT_GEMV_K, 8, true, true, true>(p);
}
