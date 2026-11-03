// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
/**
 * \brief Sliding-window paged attention for speculative verify: 8 query tokens per request, FP8 (e4m3) Q/K/V.
 *
 * Shapes (per call): q [R * 8, Hq, 64] e4m3 (heads and dims contiguous, any token stride); K/V pools
 * [slots, Hkv * 64] e4m3 (token-major paged pool, page size 2^page_shift); page_table [R, >= pages] int32;
 * seq_lens [R] int32 (KV length including the 8 verify tokens); out [R * 8, Hq, 64] bf16, contiguous.
 * Hq / Hkv == 8 (grouped-query attention).
 *
 * Masking: token t (0..7) of request r sits at position p = seq_len - 8 + t and attends keys [max(0, p - W), p],
 * where W = window_left.
 *
 * Arithmetic, in order:
 *   s   = fp32 dot product of e4m3 q and e4m3 k (tensor-core MMA, fp32 accumulation)
 *   x   = s * scale_log2                                   (scale_log2 = fp32(bmm1_scale * log2(e)), host side)
 *   p   = exp2(x - m)                                      (m: maximum of x over the warp's key range, fp32)
 *   l   = sum of p                                         (fp32)
 *   pq  = e4m3(p * 448)                                    (round to nearest even, satfinite)
 *   o   = sum of pq * v                                    (tensor-core MMA, fp32 accumulation)
 *   out = bf16(o * (bmm2_scale / 448) / l)
 * Partial results (o, m, l) of different key ranges are merged in fp32 with the usual max-rescaling: first the
 * warps of a block that share query rows (through shared memory), then the blocks of a cluster (through
 * distributed shared memory).
 * The MMAs take f16 operands: every e4m3 value (q, k, v, pq) converts to f16 exactly, so each product is the same
 * exact product of two e4m3 values and only the fp32 accumulation order differs from an e4m3-operand MMA.
 *
 * Work split: grid (kC, row blocks x Hkv, R), cluster (kC, 1, 1). One cluster serves one (request, KV head, row
 * block); its kC blocks own consecutive ranges of kChunk keys starting at the first key of the window. A block
 * holds kRG row groups of 16 query rows (row group w = tokens 2w and 2w + 1, all 8 heads of the KV head); each
 * row group is served by kSplit warps that split the block's keys evenly, using mma.sync.m16n8k16 (PTX ISA,
 * warp-level matrix multiply-accumulate). The kSplit partials of a row are merged in shared memory and the
 * result is sent to the block that finalizes the row (row r -> block r mod kC) with st.async (PTX ISA,
 * distributed shared memory), whose completion is counted in bytes on an mbarrier of the receiving block; the
 * receiver waits on that mbarrier (acquire, cluster scope) and merges its rows. One cluster barrier at kernel
 * start (after the mbarrier initialization, fence.mbarrier_init.release.cluster) orders the initialization
 * before any remote store.
 *
 * With kNormGate, the finalizing block also applies the attention output norm and gate to its rows: with x the bf16
 * attention output of one (token, head) row, ss = fp32 sum of x^2 (each of the row's 16 threads sums its 4 values
 * in order with fmaf, then a butterfly over the 16 threads, xor 8, 4, 2, 1), r = rsqrtf(ss / 64 + eps),
 * y = bf16((x * r) * (1 / (1 + exp(-g)))) with g the bf16 gate value, the reciprocal and exponential taken with
 * __fdividef and __expf. This is the arithmetic of SGLang's norm_gate kernel (no weight) for 16 threads per row.
 *
 * With kQk = E > 0 (query/key prologue), the kernel takes the bf16 QKV projection output instead of e4m3 q and does
 * the work of SGLang's fused QK-norm kernel (fused_qk_norm.py, _qk_rms_norm_fwd_kernel, FP8 KV-cache store) itself:
 * per (token, head) row of 64 values x, in fp32,
 *   ss = sum of x^2: each of the row's 64/E threads holds E consecutive columns and sums their squares in the
 *        fixed order of row_sumsq<E>, then a butterfly over the row's threads (xor 32/E, ..., 2, 1)
 *   r  = rsqrt.approx.ftz(eps + div.full(ss, 64))
 *   a  = f32(bf16(r * x))
 *   with RoPE (kRope), for the column pair (2i, 2i + 1), c = cos_sin[pos][i], s = cos_sin[pos][32 + i]:
 *        y_2i = f32(bf16(fma(a_2i, c, -(a_2i+1 * s)))), y_2i+1 = f32(bf16(fma(a_2i+1, c, a_2i * s)));  else y = a
 *   This is the form of the Triton kernel compiled with FP fusion. Compiled without it (kUnfused), every product
 *   and sum is rounded separately: the 2-column sum of squares is x0*x0 + x1*x1, the mean is ss * div.approx(1, 64),
 *   and RoPE is y_2i = a_2i*c - a_2i+1*s, y_2i+1 = a_2i*s + a_2i+1*c (each product rounded, then the sum).
 *   V rows are not normalized: y = x
 *   stored value = e4m3(min(max(y, -448), 448)), round to nearest even, satfinite.
 * The Triton kernel's per-thread column count follows its block size, which follows the row count
 * (T * (Hq + 2 Hkv)); the host picks E the same way, so the arithmetic, including the order of the sum of squares,
 * is the same for the row count of the call (tests/test_swa_verify_attn.py compares bitwise).
 * Each block computes the q rows of its row groups into shared memory; blocks whose key range holds keys of the new
 * tokens also compute the K/V rows of all 8 new tokens, and read those keys from shared memory instead of the pool.
 * The first block of row block 0 of every (request, KV head) stores the K/V rows of the 8 tokens into the pools at
 * cache_loc (rows with cache_loc == 0 are skipped, as in the Triton kernel); no block reads a pool row that this
 * kernel writes.
 *
 * Ordering: q (or the QKV output), the newest K/V rows, seq_lens and the page table may be written by the preceding
 * kernel, so the kernel reads them only after griddepcontrol.wait. With kPrefetch, the threads issue L2 prefetch hints for the
 * K/V rows of their key range before that wait; the seq_lens / page table values read for those hints only form
 * prefetch addresses (bounds-checked) and never reach the results.
 *
 * Operand layouts (no shared-memory staging of K and V: every lane loads its fragments from global memory):
 *  - QK^T: the 64 head dimensions are permuted identically for Q and K so that each lane's fragments for all four
 *    k16 steps come from one contiguous 16-byte group of its q row or k row (bytes 16*q .. 16*q+15, q = lane % 4).
 *  - PV: the probabilities feed the A operand straight from the QK accumulators. For V, lane (g, q) loads bytes
 *    16*q .. 16*q+15 of key g of each 8-key group; f16 pair i of those bytes forms 8x8 tile i, which movmatrix
 *    (PTX ISA, transpose) turns into a B fragment. Output column n of tile i is head dimension
 *    16 * (n / 2) + 2 * i + n % 2, so each lane's accumulators hold dimensions 16*q .. 16*q+15.
 */

#include <sgl_kernel/tensor.h>
#include <sgl_kernel/utils.h>

#include <sgl_kernel/runtime.cuh>
#include <sgl_kernel/utils.cuh>

#include <cooperative_groups.h>
#include <cuda_bf16.h>
#include <dlpack/dlpack.h>
#include <tvm/ffi/container/tensor.h>

#include <cmath>
#include <cstdint>

namespace muse_vr {

namespace cg = cooperative_groups;

namespace swa {

constexpr int kHeadDim = 64;
constexpr int kQPerKv = 8;
constexpr int kTokens = 8;
constexpr int kRec = 68;  // floats per partial-row record: 64 outputs, m, l, 2 pad
constexpr float kPScale = 448.0f;

struct Params {
  const uint8_t* __restrict__ q;
  const uint8_t* __restrict__ k_pool;
  const uint8_t* __restrict__ v_pool;
  const int32_t* __restrict__ page_table;
  const int32_t* __restrict__ seq_lens;
  __nv_bfloat16* __restrict__ out;
  int64_t pool_row_stride;  // bytes between consecutive token slots of the pools
  int64_t pool_slots;       // number of token slots in the pools
  int64_t page_table_stride;
  int64_t q_token_stride;   // bytes between consecutive query tokens
  int32_t num_q_heads;
  int32_t num_kv_heads;
  int32_t window_left;
  int32_t page_shift;
  float scale_log2;
  float out_scale;  // bmm2_scale / 448
  const __nv_bfloat16* __restrict__ gate;  // kNormGate: [R * 8, >= Hq * 64] bf16, row stride gate_token_stride
  int64_t gate_token_stride;                // elements
  float norm_eps;
  // kQk: QKV projection output [R * 8, >= (Hq + 2 Hkv) * 64] bf16 (q heads, then k heads, then v heads)
  const __nv_bfloat16* __restrict__ qkv;
  int64_t qkv_token_stride;                 // elements
  const int64_t* __restrict__ positions;    // [R * 8] (kRope)
  const float* __restrict__ cos_sin;        // [max position, 64] fp32: cos in columns 0..31, sin in 32..63 (kRope)
  const int64_t* __restrict__ cache_loc;    // [R * 8] pool row of each token (0: do not store)
  uint8_t* __restrict__ k_store;            // pools written by the prologue (the same rows k_pool / v_pool read)
  uint8_t* __restrict__ v_store;
  float qk_eps;
};

template <int kChunk, int kSplit, int kRG, int kC, int kQk = 0>
struct Cfg {
  static_assert(kRG == 1 || kRG == 2 || kRG == 4, "row groups per block");
  static_assert(kChunk % kSplit == 0 && (kChunk / kSplit) % 16 == 0, "each warp needs whole k16 steps");
  static_assert(kChunk / kSplit <= 64, "a warp's keys must fit in two pages of at least 64 keys");
  static_assert(kC >= 2 && kC <= 16, "cluster size (st.async needs a cluster of more than one block)");
  static_assert(kQk == 0 || kQk == 2 || kQk == 4 || kQk == 8, "prologue columns per thread");
  static constexpr int kWarps = kRG * kSplit;
  static constexpr int kThreads = 32 * kWarps;
  static constexpr int kKeysPerWarp = kChunk / kSplit;
  static constexpr int kNTiles = kKeysPerWarp / 8;
  static constexpr int kKSteps = kKeysPerWarp / 16;
  static constexpr int kRows = 16 * kRG;
  static constexpr int kRowBlocks = 4 / kRG;
  static constexpr int kOwn = (kRows + kC - 1) / kC;  // rows finalized per block, at most
  static constexpr int kPart = 0;
  static constexpr int kRecv = kPart + kSplit * kRows * kRec * 4;
  static constexpr int kBar = kRecv + kOwn * kC * kRec * 4;
  // kQk: e4m3 q rows of the block, then K and V rows of the 8 new tokens (64 bytes per row)
  static constexpr int kQs = kBar + 16;
  static constexpr int kNk = kQs + (kQk ? kRows * kHeadDim : 0);
  static constexpr int kNv = kNk + (kQk ? kTokens * kHeadDim : 0);
  static constexpr int kBytes = kNv + (kQk ? kTokens * kHeadDim : 0);
  static constexpr uint32_t kRecordBytes = (kHeadDim + 2) * 4;  // bytes received per (row, source block)
};

__device__ __forceinline__ uint4 ld_v4(const void* ptr) {
  uint4 v;
  asm volatile("ld.global.v4.u32 {%0, %1, %2, %3}, [%4];\n" : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w) : "l"(ptr));
  return v;
}

__device__ __forceinline__ int ld_cg_s32(const int32_t* ptr) {
  int v;
  asm volatile("ld.global.cg.s32 %0, [%1];\n" : "=r"(v) : "l"(ptr));
  return v;
}

__device__ __forceinline__ void prefetch_l2(const void* ptr) {
  asm volatile("prefetch.global.L2 [%0];\n" ::"l"(ptr));
}

__device__ __forceinline__ uint32_t f16x2_from_e4m3x2(uint32_t two_bytes) {
  uint32_t r;
  const uint16_t v = static_cast<uint16_t>(two_bytes);
  asm("cvt.rn.f16x2.e4m3x2 %0, %1;\n" : "=r"(r) : "h"(v));
  return r;
}

// e4m3(x0), e4m3(x1) (nearest even, satfinite), returned as f16x2 {x0 low, x1 high}
__device__ __forceinline__ uint32_t f16x2_of_e4m3(float x0, float x1) {
  uint16_t e;
  asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;\n" : "=h"(e) : "f"(x1), "f"(x0));
  return f16x2_from_e4m3x2(e);
}

__device__ __forceinline__ void mma_f16(float (&c)[4], uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3,
                                        uint32_t b0, uint32_t b1) {
  asm volatile(
      "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, "
      "{%0, %1, %2, %3};\n"
      : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
      : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}

__device__ __forceinline__ uint32_t movmatrix_trans(uint32_t a) {
  uint32_t d;
  asm volatile("movmatrix.sync.aligned.m8n8.trans.b16 %0, %1;\n" : "=r"(d) : "r"(a));
  return d;
}

__device__ __forceinline__ uint32_t smem_u32(const void* ptr) {
  return static_cast<uint32_t>(__cvta_generic_to_shared(ptr));
}

__device__ __forceinline__ uint32_t map_rank(uint32_t addr, int rank) {
  uint32_t d;
  asm volatile("mapa.shared::cluster.u32 %0, %1, %2;\n" : "=r"(d) : "r"(addr), "r"(rank));
  return d;
}

__device__ __forceinline__ void st_async_v4(uint32_t addr, float4 v, uint32_t bar) {
  asm volatile("st.async.shared::cluster.mbarrier::complete_tx::bytes.v4.f32 [%0], {%1, %2, %3, %4}, [%5];\n"
               ::"r"(addr), "f"(v.x), "f"(v.y), "f"(v.z), "f"(v.w), "r"(bar)
               : "memory");
}

__device__ __forceinline__ void st_async_v2(uint32_t addr, float a, float b, uint32_t bar) {
  asm volatile("st.async.shared::cluster.mbarrier::complete_tx::bytes.v2.f32 [%0], {%1, %2}, [%3];\n"
               ::"r"(addr), "f"(a), "f"(b), "r"(bar)
               : "memory");
}

__device__ __forceinline__ void mbar_init_expect(uint32_t bar, uint32_t bytes) {
  asm volatile("mbarrier.init.shared::cta.b64 [%0], 1;\n" ::"r"(bar) : "memory");
  asm volatile("fence.mbarrier_init.release.cluster;\n" ::: "memory");
  asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;\n" ::"r"(bar), "r"(bytes) : "memory");
}

// Upper bound on the number of try_wait calls of one receive-barrier wait. The arrivals come from the other blocks
// of the cluster, which run concurrently with this one; the bound turns a wait that can only hang through a defect
// into a trap (a CUDA error) instead of a spinning kernel.
inline constexpr uint32_t kMaxPolls = 1u << 24;

// Out of line: reached only when the bound is hit.
__device__ __noinline__ void wait_bound_exceeded() { __trap(); }

__device__ __forceinline__ void mbar_wait_phase0(uint32_t bar) {
  uint32_t polls = 0;
  while (true) {
    uint32_t done;
    asm volatile(
        "{\n"
        ".reg .pred p;\n"
        "mbarrier.try_wait.parity.acquire.cluster.shared::cta.b64 p, [%1], 0;\n"
        "selp.u32 %0, 1, 0, p;\n"
        "}\n"
        : "=r"(done)
        : "r"(bar)
        : "memory");
    if (done) break;
    if (++polls == kMaxPolls) [[unlikely]] wait_bound_exceeded();
  }
}

// Sum of squares of one thread's E consecutive columns, in the order of the Triton kernel for that column count.
template <int E, bool kUnfused = false>
__device__ __forceinline__ float row_sumsq(const float (&x)[E]) {
  if constexpr (E == 2) {
    if constexpr (kUnfused) return __fadd_rn(__fmul_rn(x[0], x[0]), __fmul_rn(x[1], x[1]));
    return __fmaf_rn(x[0], x[0], __fmul_rn(x[1], x[1]));
  } else if constexpr (E == 4) {
    return __fadd_rn(__fadd_rn(__fmul_rn(x[0], x[0]), __fmul_rn(x[2], x[2])),
                     __fadd_rn(__fmul_rn(x[1], x[1]), __fmul_rn(x[3], x[3])));
  } else {
    const float even = __fadd_rn(__fadd_rn(__fmul_rn(x[0], x[0]), __fmul_rn(x[2], x[2])),
                                 __fadd_rn(__fmul_rn(x[4], x[4]), __fmul_rn(x[6], x[6])));
    const float odd = __fadd_rn(__fadd_rn(__fmul_rn(x[1], x[1]), __fmul_rn(x[3], x[3])),
                                __fadd_rn(__fmul_rn(x[5], x[5]), __fmul_rn(x[7], x[7])));
    return __fadd_rn(even, odd);
  }
}

// Mean of the sum of squares over the 64 columns: div.full(ss, 64) (fused form), or ss times div.approx(1, 64)
// (unfused form: how the Triton kernel's "/ D" lowers without FP fusion).
template <bool kUnfused>
__device__ __forceinline__ float qk_mean(float ss) {
  float var;
  if constexpr (kUnfused) {
    float inv;
    asm("div.approx.f32 %0, %1, %2;\n" : "=f"(inv) : "f"(1.0f), "f"(64.0f));
    var = __fmul_rn(inv, ss);
  } else {
    asm("div.full.f32 %0, %1, %2;\n" : "=f"(var) : "f"(ss), "f"(64.0f));
  }
  return var;
}

template <bool kUnfused>
__device__ __forceinline__ float qk_rstd(float ss, float eps) {
  float r;
  const float var = qk_mean<kUnfused>(ss);
  const float t = __fadd_rn(eps, var);
  asm("rsqrt.approx.ftz.f32 %0, %1;\n" : "=f"(r) : "f"(t));
  return r;
}

__device__ __forceinline__ float bf16_round(float x) { return __bfloat162float(__float2bfloat16_rn(x)); }

// e4m3(x0), e4m3(x1) (nearest even, satfinite) as two bytes {x0 low, x1 high}
__device__ __forceinline__ uint16_t e4m3x2(float x0, float x1) {
  uint16_t e;
  asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;\n" : "=h"(e) : "f"(x1), "f"(x0));
  return e;
}

__device__ __forceinline__ float fast_exp2(float x) {
  float y;
  asm("ex2.approx.ftz.f32 %0, %1;\n" : "=f"(y) : "f"(x));
  return y;
}

__device__ __forceinline__ float quad_max(float v) {
  v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, 1));
  v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, 2));
  return v;
}

__device__ __forceinline__ float quad_sum(float v) {
  v += __shfl_xor_sync(0xffffffffu, v, 1);
  v += __shfl_xor_sync(0xffffffffu, v, 2);
  return v;
}

// Prologue loads are volatile so that they are issued in program order, ahead of the arithmetic that uses them.
template <int E>
__device__ __forceinline__ void ld_bf16_raw(const __nv_bfloat16* ptr, uint32_t (&w)[E / 2]) {
  if constexpr (E == 2) {
    asm volatile("ld.global.b32 %0, [%1];\n" : "=r"(w[0]) : "l"(ptr));
  } else if constexpr (E == 4) {
    asm volatile("ld.global.v2.b32 {%0, %1}, [%2];\n" : "=r"(w[0]), "=r"(w[1]) : "l"(ptr));
  } else {
    asm volatile("ld.global.v4.b32 {%0, %1, %2, %3}, [%4];\n"
                 : "=r"(w[0]), "=r"(w[1]), "=r"(w[2]), "=r"(w[3])
                 : "l"(ptr));
  }
}

template <int N>
__device__ __forceinline__ void ld_f32(const float* ptr, float* v) {
  if constexpr (N == 1) {
    asm volatile("ld.global.f32 %0, [%1];\n" : "=f"(v[0]) : "l"(ptr));
  } else if constexpr (N == 2) {
    asm volatile("ld.global.v2.f32 {%0, %1}, [%2];\n" : "=f"(v[0]), "=f"(v[1]) : "l"(ptr));
  } else {
    asm volatile("ld.global.v4.f32 {%0, %1, %2, %3}, [%4];\n" : "=f"(v[0]), "=f"(v[1]), "=f"(v[2]), "=f"(v[3]) : "l"(ptr));
  }
}

__device__ __forceinline__ int64_t ld_s64(const int64_t* ptr) {
  int64_t v;
  asm volatile("ld.global.s64 %0, [%1];\n" : "=l"(v) : "l"(ptr));
  return v;
}

__device__ __forceinline__ int ld_s32(const int32_t* ptr) {
  int v;
  asm volatile("ld.global.s32 %0, [%1];\n" : "=r"(v) : "l"(ptr));
  return v;
}

// Stores the E bytes of one thread (E = 2, 4 or 8).
template <int E>
__device__ __forceinline__ void st_bytes(void* ptr, const uint16_t (&b)[E / 2]) {
  if constexpr (E == 2) {
    *reinterpret_cast<uint16_t*>(ptr) = b[0];
  } else if constexpr (E == 4) {
    *reinterpret_cast<uint32_t*>(ptr) = static_cast<uint32_t>(b[0]) | (static_cast<uint32_t>(b[1]) << 16);
  } else {
    *reinterpret_cast<uint2*>(ptr) = make_uint2(static_cast<uint32_t>(b[0]) | (static_cast<uint32_t>(b[1]) << 16),
                                                static_cast<uint32_t>(b[2]) | (static_cast<uint32_t>(b[3]) << 16));
  }
}

// Rows of the query/key prologue held by one thread: kPasses rows (one per pass), E columns each, from kTok tokens
// (kTok == 1: every row of the thread belongs to one token and shares its position and RoPE row). Loads are issued
// for every pass before any arithmetic.
template <int E, bool kRope, int kPasses, int kTok, bool kUnfused>
struct QkRows {
  uint32_t w[kPasses][E / 2];  // bf16 pairs as loaded
  float cs[kTok][E];           // E / 2 cosines, then E / 2 sines
  int64_t pos[kTok];

  __device__ __forceinline__ void load(int ps, const __nv_bfloat16* src) { ld_bf16_raw<E>(src, w[ps]); }
  __device__ __forceinline__ void load_pos(int i, const int64_t* src) {
    if constexpr (kRope) pos[i] = ld_s64(src);
  }
  __device__ __forceinline__ void load_rope(const Params& p, int li) {
    if constexpr (kRope) {
#pragma unroll
      for (int i = 0; i < kTok; ++i) {
        const float* row = p.cos_sin + pos[i] * kHeadDim + (E / 2) * li;
        ld_f32<E / 2>(row, cs[i]);
        ld_f32<E / 2>(row + kHeadDim / 2, cs[i] + E / 2);
      }
    }
  }
  // e4m3 bytes of row ps; rope: apply RoPE (q and k rows); raw: no normalization (v rows).
  __device__ __forceinline__ void finish(int ps, int ti, float eps, bool rope, bool raw, uint16_t (&b)[E / 2]) const {
    float x[E];
#pragma unroll
    for (int i = 0; i < E / 2; ++i) {
      const float2 f = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&w[ps][i]));
      x[2 * i] = f.x;
      x[2 * i + 1] = f.y;
    }
    float ss = row_sumsq<E, kUnfused>(x);
    constexpr int kLanes = 64 / E;
#pragma unroll
    for (int m = kLanes / 2; m >= 1; m >>= 1) ss = __fadd_rn(ss, __shfl_xor_sync(0xffffffffu, ss, m));
    const float r = qk_rstd<kUnfused>(ss, eps);
    float y[E];
#pragma unroll
    for (int e = 0; e < E; ++e) y[e] = bf16_round(__fmul_rn(r, x[e]));
    if (kRope && rope) {
#pragma unroll
      for (int m = 0; m < E / 2; ++m) {
        const float c = cs[ti][m], s = cs[ti][E / 2 + m];
        const float a0 = y[2 * m], a1 = y[2 * m + 1];
        if constexpr (kUnfused) {
          y[2 * m] = bf16_round(__fsub_rn(__fmul_rn(a0, c), __fmul_rn(a1, s)));
          y[2 * m + 1] = bf16_round(__fadd_rn(__fmul_rn(a0, s), __fmul_rn(a1, c)));
        } else {
          y[2 * m] = bf16_round(__fmaf_rn(a0, c, -__fmul_rn(a1, s)));
          y[2 * m + 1] = bf16_round(__fmaf_rn(a1, c, __fmul_rn(a0, s)));
        }
      }
    }
    if (raw) {
#pragma unroll
      for (int e = 0; e < E; ++e) y[e] = x[e];
    }
#pragma unroll
    for (int m = 0; m < E / 2; ++m)
      b[m] = e4m3x2(fminf(fmaxf(y[2 * m], -448.0f), 448.0f), fminf(fmaxf(y[2 * m + 1], -448.0f), 448.0f));
  }
};

// Merge n partial records of one row (record s at rec + s * stride floats) for output columns 4*c4 .. 4*c4+3.
template <int n, int stride>
__device__ __forceinline__ void merge_records(const float* rec, int c4, float4& acc, float& mx, float& lsum) {
  float ms[n];
  mx = -INFINITY;
#pragma unroll
  for (int s = 0; s < n; ++s) {
    ms[s] = rec[s * stride + kHeadDim];
    mx = fmaxf(mx, ms[s]);
  }
  acc = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
  lsum = 0.0f;
#pragma unroll
  for (int s = 0; s < n; ++s) {
    if (ms[s] != -INFINITY) {
      const float w = fast_exp2(ms[s] - mx);
      const float4 v = *reinterpret_cast<const float4*>(rec + s * stride + 4 * c4);
      acc.x += w * v.x;
      acc.y += w * v.y;
      acc.z += w * v.z;
      acc.w += w * v.w;
      lsum += w * rec[s * stride + kHeadDim + 1];
    }
  }
}

template <int kChunk, int kSplit, int kRG, int kC, bool kUsePDL, bool kPrefetch, bool kNormGate, int kQk,
          bool kRope, bool kUnfused>
__global__ void __launch_bounds__(Cfg<kChunk, kSplit, kRG, kC, kQk>::kThreads)
    swa_verify_attn_kernel(const Params p) {
  using C = Cfg<kChunk, kSplit, kRG, kC, kQk>;
  extern __shared__ __align__(128) uint8_t smem[];
  float* const part = reinterpret_cast<float*>(smem + C::kPart);
  float* const recv = reinterpret_cast<float*>(smem + C::kRecv);
  const uint32_t bar = smem_u32(smem + C::kBar);

  cg::cluster_group cluster = cg::this_cluster();
  const int rank = static_cast<int>(cluster.block_rank());
  const int nown = (C::kRows - rank + kC - 1) / kC;  // rows rank, rank + kC, ... below kRows
  // Receive barrier: one arrival (here) plus the bytes of nown * kC records.
  if (threadIdx.x == 0) mbar_init_expect(bar, static_cast<uint32_t>(nown * kC) * C::kRecordBytes);
  // Cluster barrier arrival now, wait before the first remote store: every block of the cluster has started and
  // initialized its receive barrier before any block stores into another block's shared memory.
  asm volatile("barrier.cluster.arrive.relaxed.aligned;\n" ::: "memory");

  const int rb = blockIdx.y / p.num_kv_heads;
  const int kvh = blockIdx.y - rb * p.num_kv_heads;
  const int req = blockIdx.z;
  const int tid = threadIdx.x;
  const int warp = tid >> 5, lane = tid & 31;
  const int wrl = warp % kRG;     // row group within the block
  const int wk = warp / kRG;      // key sub-range within the block
  const int wr = rb * kRG + wrl;  // row group within the KV head: tokens 2*wr, 2*wr+1
  const int g = lane >> 2, q4 = lane & 3;
  const int64_t head_off = static_cast<int64_t>(kvh) * kHeadDim;
  const int32_t* pt = p.page_table + static_cast<int64_t>(req) * p.page_table_stride;
  const int page_mask = (1 << p.page_shift) - 1;

  if constexpr (kPrefetch) {
    // Hints only: values read here form prefetch addresses and are not used afterwards.
    const int len = ld_cg_s32(p.seq_lens + req);
    const int first = max(0, len - kTokens - p.window_left);
    const int k0 = first + rank * kChunk;
    const int n = max(0, min(kChunk, len - k0));
    for (int r = tid; r < n; r += C::kThreads) {
      const int key = k0 + r;
      const int64_t slot = (static_cast<int64_t>(ld_cg_s32(pt + (key >> p.page_shift))) << p.page_shift) +
                           (key & page_mask);
      if (slot >= 0 && slot < p.pool_slots) {
        const int64_t off = slot * p.pool_row_stride + head_off;
        prefetch_l2(p.k_pool + off);
        prefetch_l2(p.v_pool + off);
      }
    }
  }

  if constexpr (kUsePDL) asm volatile("griddepcontrol.wait;\n" ::: "memory");

  // Q rows (16 * wrl + g) and (+8): tokens t0 = 2*wr and t1 = 2*wr+1, head kvh*8 + g; 16 bytes each.
  const int t0 = 2 * wr, t1 = t0 + 1;
  uint4 qa, qb;
  if constexpr (kQk == 0) {
    const uint8_t* qrow = p.q + (static_cast<int64_t>(req) * kTokens + t0) * p.q_token_stride +
                          (kvh * kQPerKv + g) * kHeadDim + 16 * q4;
    qa = ld_v4(qrow);
    qb = ld_v4(qrow + p.q_token_stride);
  }

  // Prologue rows (kQk = E), 64 / E threads per row: the q rows of the block (row 16 * wrl + 8 * tt + h: token
  // 2 * (rb * kRG + wrl) + tt, head kvh * 8 + h) and the K/V rows of the 8 new tokens (row j: token j / 2, K for even
  // j, V for odd j). Warp w holds q rows w * kItemsQW .. + kItemsQW - 1 and K/V rows w * kItemsKW .. + kItemsKW - 1,
  // kRowsWarp per pass. All rows are loaded; the K/V rows are computed only in blocks that use them.
  constexpr int E = kQk ? kQk : 2;
  constexpr int kQkLanes = 64 / E, kQkRowsWarp = 32 / kQkLanes;
  constexpr int kItemsQW = C::kRows / C::kWarps, kItemsKW = 2 * kTokens / C::kWarps;
  constexpr int kPassQ = kQk ? kItemsQW / kQkRowsWarp : 1;
  constexpr int kPassKV = kQk ? kItemsKW / kQkRowsWarp : 1;
  static_assert(kQk == 0 || (kItemsQW * C::kWarps == C::kRows && kPassQ * kQkRowsWarp == kItemsQW &&
                             8 % kItemsQW == 0),
                "prologue: q rows must split evenly over warps, whole rows per pass, one token per warp");
  static_assert(kQk == 0 || (kItemsKW * C::kWarps == 2 * kTokens && kPassKV * kQkRowsWarp == kItemsKW),
                "prologue: K/V rows must split evenly over warps, whole rows per pass");
  const int qli = lane % kQkLanes, qrw = lane / kQkLanes;
  const int64_t tok0 = static_cast<int64_t>(req) * kTokens;
  const int wq = warp * kItemsQW;  // first q row of the warp
  const int qtok = 2 * (rb * kRG + (wq >> 4)) + ((wq >> 3) & 1);
  QkRows<E, kRope, kPassQ, 1, kUnfused> qrows;
  QkRows<E, kRope, kPassKV, kPassKV, kUnfused> kvrows;
  int64_t loc[kPassKV];
  const bool store = kQk > 0 && rb == 0 && rank == 0;  // stores the new K/V rows (every (request, KV head) once)
  const auto kv_token = [&](int ps) { return tok0 + ((warp * kItemsKW + ps * kQkRowsWarp + qrw) >> 1); };
  const auto load_qkv = [&]() {  // (one call; a lambda keeps the address math next to its loads)
    const __nv_bfloat16* qsrc = p.qkv + (tok0 + qtok) * p.qkv_token_stride +
                                ((kvh * kQPerKv + (wq & 7) + qrw) * kHeadDim + E * qli);
#pragma unroll
    for (int ps = 0; ps < kPassQ; ++ps) qrows.load(ps, qsrc + ps * kQkRowsWarp * kHeadDim);
#pragma unroll
    for (int ps = 0; ps < kPassKV; ++ps) {
      const int j = warp * kItemsKW + ps * kQkRowsWarp + qrw;
      kvrows.load(ps, p.qkv + kv_token(ps) * p.qkv_token_stride +
                          ((p.num_q_heads + (j & 1) * p.num_kv_heads + kvh) * kHeadDim + E * qli));
    }
  };
  // seq_len heads the longest chain (seq_len -> page table -> K/V rows): issued first.
  int seq_len = kQk > 0 ? ld_s32(p.seq_lens + req) : p.seq_lens[req];
  if constexpr (kQk > 0) {
    qrows.load_pos(0, p.positions + tok0 + qtok);
#pragma unroll
    for (int ps = 0; ps < kPassKV; ++ps) kvrows.load_pos(ps, p.positions + kv_token(ps));
    load_qkv();
  }

  // K and V fragments of this warp's keys: n-tile j holds key kbase + 8j + g; keys past the sequence end are zeros.
  // A warp's keys span at most two pages (kKeysPerWarp <= 64 <= page size): two page-table reads, then every load is
  // issued unconditionally (keys past the end read the warp's first key, a valid slot) and masked afterwards. With
  // kQk, the loads of keys >= tnew are skipped (those rows are written by this kernel).
  const int kbase = wk * C::kKeysPerWarp;
  uint4 kr[C::kNTiles], vr[C::kNTiles];
  if constexpr (kQk > 0) {
#pragma unroll
    for (int ps = 0; ps < kPassKV; ++ps) loc[ps] = store ? ld_s64(p.cache_loc + kv_token(ps)) : 0;
  }
  const int first = max(0, seq_len - kTokens - p.window_left);
  const int k0 = first + rank * kChunk;
  const int nvalid = max(0, min(kChunk, seq_len - k0));
  const int tnew = seq_len - kTokens;  // key index of token 0 (kQk: keys from tnew on come from the prologue)
  // Blocks whose key range holds new tokens use the computed K/V rows (and the storing block).
  const bool use_kv = kQk > 0 && (store || min(k0 + kChunk, seq_len) > max(k0, tnew));  // uniform over the block
  {
    const int kfirst = k0 + kbase;
    const int pfirst = kfirst >> p.page_shift;
    const int plast = max(seq_len - 1, 0) >> p.page_shift;
    const int32_t* const pte0 = pt + min(pfirst, plast);
    const int32_t* const pte1 = pt + min(pfirst + 1, plast);
    const int64_t pg0 = static_cast<int64_t>(kQk > 0 ? ld_s32(pte0) : *pte0) << p.page_shift;
    const int64_t pg1 = static_cast<int64_t>(kQk > 0 ? ld_s32(pte1) : *pte1) << p.page_shift;
    if constexpr (kQk > 0) {
      qrows.load_rope(p, qli);
      if (use_kv) kvrows.load_rope(p, qli);
    }
#pragma unroll
    for (int j = 0; j < C::kNTiles; ++j) {
      const int rel = kbase + 8 * j + g;
      const int key = rel < nvalid ? k0 + rel : kfirst;
      const int64_t off = (((key >> p.page_shift) == pfirst ? pg0 : pg1) + (key & page_mask)) * p.pool_row_stride +
                          head_off + 16 * q4;
      if (kQk == 0 || (rel < nvalid && key < tnew)) {
        kr[j] = ld_v4(p.k_pool + off);
        vr[j] = ld_v4(p.v_pool + off);
      } else {
        kr[j] = vr[j] = make_uint4(0u, 0u, 0u, 0u);
      }
    }
#pragma unroll
    for (int j = 0; j < C::kNTiles; ++j) {
      if (kbase + 8 * j + g >= nvalid) kr[j] = vr[j] = make_uint4(0u, 0u, 0u, 0u);
    }
  }

  if constexpr (kQk > 0) {
    uint8_t* const qs = smem + C::kQs;
    uint8_t* const nks = smem + C::kNk;
    uint8_t* const nvs = smem + C::kNv;
#pragma unroll
    for (int ps = 0; ps < kPassQ; ++ps) {
      uint16_t bq[E / 2];
      qrows.finish(ps, 0, p.qk_eps, true, false, bq);
      st_bytes<E>(qs + (wq + ps * kQkRowsWarp + qrw) * kHeadDim + E * qli, bq);
    }
    if (use_kv) {
#pragma unroll
      for (int ps = 0; ps < kPassKV; ++ps) {
        const int j = warp * kItemsKW + ps * kQkRowsWarp + qrw;
        const int t = j >> 1;
        uint16_t bkv[E / 2];
        kvrows.finish(ps, ps, p.qk_eps, (j & 1) == 0, (j & 1) == 1, bkv);
        st_bytes<E>(((j & 1) ? nvs : nks) + t * kHeadDim + E * qli, bkv);
        if (loc[ps] != 0)
          st_bytes<E>(((j & 1) ? p.v_store : p.k_store) + loc[ps] * p.pool_row_stride + kvh * kHeadDim + E * qli, bkv);
      }
    }
    __syncthreads();
    qa = *reinterpret_cast<const uint4*>(qs + (wrl * 16 + g) * kHeadDim + 16 * q4);
    qb = *reinterpret_cast<const uint4*>(qs + (wrl * 16 + 8 + g) * kHeadDim + 16 * q4);
#pragma unroll
    for (int j = 0; j < C::kNTiles; ++j) {
      const int rel = kbase + 8 * j + g;
      const int key = k0 + rel;
      if (rel < nvalid && key >= tnew) {
        kr[j] = *reinterpret_cast<const uint4*>(nks + (key - tnew) * kHeadDim + 16 * q4);
        vr[j] = *reinterpret_cast<const uint4*>(nvs + (key - tnew) * kHeadDim + 16 * q4);
      }
    }
  }

  // Q A fragments (f16) for the four k16 steps: step s uses bytes 4s..4s+3 of each lane's 16-byte group.
  uint32_t aq[4][4];
  {
    const uint32_t wa[4] = {qa.x, qa.y, qa.z, qa.w};
    const uint32_t wb[4] = {qb.x, qb.y, qb.z, qb.w};
#pragma unroll
    for (int sstep = 0; sstep < 4; ++sstep) {
      aq[sstep][0] = f16x2_from_e4m3x2(wa[sstep] & 0xffffu);
      aq[sstep][1] = f16x2_from_e4m3x2(wb[sstep] & 0xffffu);
      aq[sstep][2] = f16x2_from_e4m3x2(wa[sstep] >> 16);
      aq[sstep][3] = f16x2_from_e4m3x2(wb[sstep] >> 16);
    }
  }

  // S = Q K^T over this warp's keys kbase .. kbase + kKeysPerWarp - 1 (relative to k0); B fragments of step s are
  // bytes 4s..4s+3 of the lane's 16-byte group of its k row, in the same permuted order as Q.
  float s[C::kNTiles][4];
#pragma unroll
  for (int j = 0; j < C::kNTiles; ++j) {
    const uint32_t kw[4] = {kr[j].x, kr[j].y, kr[j].z, kr[j].w};
    s[j][0] = s[j][1] = s[j][2] = s[j][3] = 0.0f;
#pragma unroll
    for (int sstep = 0; sstep < 4; ++sstep)
      mma_f16(s[j], aq[sstep][0], aq[sstep][1], aq[sstep][2], aq[sstep][3], f16x2_from_e4m3x2(kw[sstep] & 0xffffu),
              f16x2_from_e4m3x2(kw[sstep] >> 16));
  }

  // Masked, scaled scores and row maxima. Accumulator elements 0,1 belong to token t0, elements 2,3 to t1.
  const int pos0 = seq_len - kTokens + t0;
  const int lo0 = max(0, pos0 - p.window_left) - k0, hi0 = min(nvalid, pos0 - k0 + 1);
  const int lo1 = max(0, pos0 + 1 - p.window_left) - k0, hi1 = min(nvalid, pos0 - k0 + 2);
  float m0 = -INFINITY, m1 = -INFINITY;
  if (kbase >= max(lo0, lo1) && kbase + C::kKeysPerWarp <= min(hi0, hi1)) {
    // Every key of this warp is visible to both tokens (warp-uniform): no masking.
#pragma unroll
    for (int j = 0; j < C::kNTiles; ++j) {
#pragma unroll
      for (int e = 0; e < 4; ++e) {
        const float x = s[j][e] * p.scale_log2;
        s[j][e] = x;
        if (e < 2) m0 = fmaxf(m0, x);
        else m1 = fmaxf(m1, x);
      }
    }
  } else {
#pragma unroll
    for (int j = 0; j < C::kNTiles; ++j) {
#pragma unroll
      for (int e = 0; e < 4; ++e) {
        const int key = kbase + 8 * j + 2 * q4 + (e & 1);
        const bool valid = (e < 2) ? (key >= lo0 && key < hi0) : (key >= lo1 && key < hi1);
        const float x = valid ? s[j][e] * p.scale_log2 : -INFINITY;
        s[j][e] = x;
        if (e < 2) m0 = fmaxf(m0, x);
        else m1 = fmaxf(m1, x);
      }
    }
  }
  m0 = quad_max(m0);
  m1 = quad_max(m1);
  const float mu0 = (m0 == -INFINITY) ? 0.0f : m0;
  const float mu1 = (m1 == -INFINITY) ? 0.0f : m1;
  float l0 = 0.0f, l1 = 0.0f;
#pragma unroll
  for (int j = 0; j < C::kNTiles; ++j) {
    const float p0 = fast_exp2(s[j][0] - mu0), p1 = fast_exp2(s[j][1] - mu0);
    const float p2 = fast_exp2(s[j][2] - mu1), p3 = fast_exp2(s[j][3] - mu1);
    l0 += p0 + p1;
    l1 += p2 + p3;
    s[j][0] = p0 * kPScale;
    s[j][1] = p1 * kPScale;
    s[j][2] = p2 * kPScale;
    s[j][3] = p3 * kPScale;
  }
  l0 = quad_sum(l0);
  l1 = quad_sum(l1);

  // O = P V over the warp's keys. o[i] = 8x8 output tile i: columns map to head dims 16 * (n / 2) + 2i + n % 2.
  float o[kHeadDim / 8][4];
#pragma unroll
  for (int i = 0; i < kHeadDim / 8; ++i) o[i][0] = o[i][1] = o[i][2] = o[i][3] = 0.0f;
#pragma unroll
  for (int kk = 0; kk < C::kKSteps; ++kk) {
    const uint32_t a0 = f16x2_of_e4m3(s[2 * kk][0], s[2 * kk][1]);
    const uint32_t a1 = f16x2_of_e4m3(s[2 * kk][2], s[2 * kk][3]);
    const uint32_t a2 = f16x2_of_e4m3(s[2 * kk + 1][0], s[2 * kk + 1][1]);
    const uint32_t a3 = f16x2_of_e4m3(s[2 * kk + 1][2], s[2 * kk + 1][3]);
    const uint32_t w0[4] = {vr[2 * kk].x, vr[2 * kk].y, vr[2 * kk].z, vr[2 * kk].w};          // keys 16kk + g
    const uint32_t w1[4] = {vr[2 * kk + 1].x, vr[2 * kk + 1].y, vr[2 * kk + 1].z, vr[2 * kk + 1].w};  // + 8
#pragma unroll
    for (int i = 0; i < kHeadDim / 8; ++i) {
      const uint32_t b0 = movmatrix_trans(f16x2_from_e4m3x2((w0[i >> 1] >> (16 * (i & 1))) & 0xffffu));
      const uint32_t b1 = movmatrix_trans(f16x2_from_e4m3x2((w1[i >> 1] >> (16 * (i & 1))) & 0xffffu));
      mma_f16(o[i], a0, a1, a2, a3, b0, b1);
    }
  }

  // This warp's partial rows -> part[wk][row].
  {
    float* rec0 = part + (wk * C::kRows + wrl * 16 + g) * kRec + 16 * q4;  // dims 16*q4 .. 16*q4+15
    float* rec1 = rec0 + 8 * kRec;
#pragma unroll
    for (int u = 0; u < 4; ++u) {
      *reinterpret_cast<float4*>(rec0 + 4 * u) = make_float4(o[2 * u][0], o[2 * u][1], o[2 * u + 1][0], o[2 * u + 1][1]);
      *reinterpret_cast<float4*>(rec1 + 4 * u) = make_float4(o[2 * u][2], o[2 * u][3], o[2 * u + 1][2], o[2 * u + 1][3]);
    }
    if (q4 == 0) {
      *reinterpret_cast<float2*>(rec0 + kHeadDim) = make_float2(m0, l0);
      *reinterpret_cast<float2*>(rec1 + kHeadDim) = make_float2(m1, l1);
    }
  }
  __syncthreads();

  // Merge the kSplit partials of each row; store the block's row record into the finalizing block.
  asm volatile("barrier.cluster.wait.aligned;\n" ::: "memory");
  for (int item = tid; item < C::kRows * 16; item += C::kThreads) {
    const int row = item >> 4, c4 = item & 15;
    float4 acc;
    float mx, lsum;
    merge_records<kSplit, C::kRows * kRec>(part + row * kRec, c4, acc, mx, lsum);
    const int owner = row % kC;
    const uint32_t dst = map_rank(smem_u32(recv + ((row / kC) * kC + rank) * kRec), owner);
    const uint32_t rbar = map_rank(bar, owner);
    st_async_v4(dst + 16 * c4, acc, rbar);
    if (c4 == 0) st_async_v2(dst + 4 * kHeadDim, mx, lsum, rbar);
  }
  if constexpr (kUsePDL) asm volatile("griddepcontrol.launch_dependents;\n" ::: "memory");
  mbar_wait_phase0(bar);

  // Finalize the rows this block owns (rank, rank + kC, ...), four output columns per item.
  for (int item = tid; item < nown * 16; item += C::kThreads) {
    const int sl = item >> 4, c4 = item & 15;
    float4 acc;
    float mx, lsum;
    merge_records<kC, kRec>(recv + sl * kC * kRec, c4, acc, mx, lsum);
    const int row = sl * kC + rank;  // block-local row
    const int t = 2 * (rb * kRG + row / 16) + ((row >> 3) & 1);
    const int h = kvh * kQPerKv + (row & 7);
    const float f = lsum > 0.0f ? p.out_scale / lsum : 0.0f;
    __nv_bfloat162 lo = __floats2bfloat162_rn(acc.x * f, acc.y * f);
    __nv_bfloat162 hi = __floats2bfloat162_rn(acc.z * f, acc.w * f);
    if constexpr (kNormGate) {
      const float2 x01 = __bfloat1622float2(lo), x23 = __bfloat1622float2(hi);
      float ss = fmaf(x01.x, x01.x, 0.0f);
      ss = fmaf(x01.y, x01.y, ss);
      ss = fmaf(x23.x, x23.x, ss);
      ss = fmaf(x23.y, x23.y, ss);
      const unsigned group = 0xffffu << (threadIdx.x & 16);  // the 16 threads of this row
#pragma unroll
      for (int mask = 8; mask >= 1; mask >>= 1) ss = ss + __shfl_xor_sync(group, ss, mask, 16);
      const float r = rsqrtf(ss / static_cast<float>(kHeadDim) + p.norm_eps);
      const uint2 graw = *reinterpret_cast<const uint2*>(
          p.gate + (static_cast<int64_t>(req) * kTokens + t) * p.gate_token_stride + h * kHeadDim + 4 * c4);
      const float2 g01 = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&graw.x));
      const float2 g23 = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&graw.y));
      const auto sig = [](float g) { return __fdividef(1.0f, 1.0f + __expf(-g)); };
      float y0 = x01.x * r, y1 = x01.y * r, y2 = x23.x * r, y3 = x23.y * r;
      y0 *= sig(g01.x);
      y1 *= sig(g01.y);
      y2 *= sig(g23.x);
      y3 *= sig(g23.y);
      lo = __floats2bfloat162_rn(y0, y1);
      hi = __floats2bfloat162_rn(y2, y3);
    }
    uint2 packed;
    packed.x = *reinterpret_cast<uint32_t*>(&lo);
    packed.y = *reinterpret_cast<uint32_t*>(&hi);
    *reinterpret_cast<uint2*>(p.out + ((static_cast<int64_t>(req) * kTokens + t) * p.num_q_heads + h) * kHeadDim +
                              4 * c4) = packed;
  }
}

}  // namespace swa

template <int kChunk, int kSplit, int kRG, int kC, bool kUsePDL, bool kPrefetch, bool kNormGate, int kQk, bool kRope,
          bool kUnfused>
struct SwaVerifyAttnKernel {
  static void run(const tvm::ffi::TensorView q,           // kQk == 0: [R*8, Hq, 64] e4m3, heads/dims contiguous
                  const tvm::ffi::TensorView k_pool,      // [slots, Hkv*64] e4m3 bytes, rows contiguous
                  const tvm::ffi::TensorView v_pool,      // [slots, Hkv*64]
                  const tvm::ffi::TensorView page_table,  // [R, pages] int32, inner contiguous
                  const tvm::ffi::TensorView seq_lens,    // [R] int32
                  const tvm::ffi::TensorView out,         // [R*8, Hq, 64] bf16, contiguous
                  int64_t window_left, int64_t page_size, double bmm1_scale, double bmm2_scale,
                  const tvm::ffi::TensorView gate,  // kNormGate: [R*8, >= Hq*64] bf16, inner contiguous
                  double norm_eps,
                  const tvm::ffi::TensorView qkv,        // kQk: [R*8, >= (Hq + 2 Hkv) * 64] bf16, inner contiguous
                  const tvm::ffi::TensorView positions,  // kRope: [>= R*8] int64
                  const tvm::ffi::TensorView cos_sin,    // kRope: [positions, 64] fp32, contiguous
                  const tvm::ffi::TensorView cache_loc,  // kQk: [>= R*8] int64
                  double qk_eps) {
    using namespace sglang::host;
    using Cfg = swa::Cfg<kChunk, kSplit, kRG, kC, kQk>;
    RuntimeCheck(k_pool.dtype().code == DLDataTypeCode::kDLFloat8_e4m3fn &&
                     v_pool.dtype().code == DLDataTypeCode::kDLFloat8_e4m3fn,
                 "k/v must be float8_e4m3fn");
    RuntimeCheck(out.dtype().code == DLDataTypeCode::kDLBfloat && out.dtype().bits == 16, "out must be bf16");
    RuntimeCheck(page_table.dtype().code == DLDataTypeCode::kDLInt && page_table.dtype().bits == 32 &&
                     seq_lens.dtype().code == DLDataTypeCode::kDLInt && seq_lens.dtype().bits == 32,
                 "page_table/seq_lens must be int32");
    RuntimeCheck(out.dim() == 3 && out.size(2) == swa::kHeadDim, "out must be [tokens, heads, 64]");
    const int64_t m = out.size(0), hq = out.size(1);
    RuntimeCheck(m % swa::kTokens == 0, "token count must be a multiple of 8");
    const int64_t reqs = m / swa::kTokens;
    RuntimeCheck(seq_lens.dim() == 1 && seq_lens.size(0) == reqs && seq_lens.stride(0) == 1, "seq_lens [R]");
    RuntimeCheck(page_table.dim() == 2 && page_table.size(0) >= reqs && page_table.stride(1) == 1, "page_table");
    RuntimeCheck(out.stride(2) == 1 && out.stride(1) == swa::kHeadDim && out.stride(0) == hq * swa::kHeadDim,
                 "out layout");
    RuntimeCheck(k_pool.dim() == 2 && v_pool.dim() == 2 && k_pool.stride(1) == 1 && v_pool.stride(1) == 1 &&
                     k_pool.stride(0) == v_pool.stride(0) && k_pool.size(0) == v_pool.size(0) &&
                     k_pool.size(1) == v_pool.size(1),
                 "pools must be [slots, row] with unit inner stride");
    const int64_t hkv = k_pool.size(1) / swa::kHeadDim;
    RuntimeCheck(hkv * swa::kHeadDim == k_pool.size(1) && hkv * swa::kQPerKv == hq, "expects 8 query heads per KV head");
    RuntimeCheck(window_left >= 0 && page_size >= 64 && (page_size & (page_size - 1)) == 0, "window/page size");
    RuntimeCheck((window_left + swa::kTokens + kChunk - 1) / kChunk == kC, "cluster size ", kC,
                 " does not cover window_left ", window_left);
    const auto aligned16 = [](const void* ptr) { return reinterpret_cast<uintptr_t>(ptr) % 16 == 0; };
    RuntimeCheck(aligned16(k_pool.data_ptr()) && aligned16(v_pool.data_ptr()) && aligned16(out.data_ptr()) &&
                     k_pool.stride(0) % 16 == 0,
                 "16-byte alignment");
    int page_shift = 0;
    while ((int64_t{1} << page_shift) < page_size) ++page_shift;

    swa::Params params{};
    if constexpr (kQk == 0) {
      RuntimeCheck(q.dim() == 3 && q.size(0) == m && q.size(1) == hq && q.size(2) == swa::kHeadDim &&
                       q.dtype().code == DLDataTypeCode::kDLFloat8_e4m3fn && q.stride(2) == 1 &&
                       q.stride(1) == swa::kHeadDim && q.stride(0) % 16 == 0 && aligned16(q.data_ptr()),
                   "q must be e4m3 [tokens, heads, 64], heads and dims contiguous, 16-byte aligned rows");
      params.q = static_cast<const uint8_t*>(q.data_ptr());
      params.q_token_stride = q.stride(0);
    } else {
      RuntimeCheck(qkv.dtype().code == DLDataTypeCode::kDLBfloat && qkv.dtype().bits == 16 && qkv.dim() == 2 &&
                       qkv.size(0) == m && qkv.size(1) >= (hq + 2 * hkv) * swa::kHeadDim && qkv.stride(1) == 1 &&
                       qkv.stride(0) % 8 == 0 && aligned16(qkv.data_ptr()),
                   "qkv must be bf16 [tokens, >= (Hq + 2 Hkv) * 64], inner contiguous, 16-byte aligned rows");
      RuntimeCheck(cache_loc.dtype().code == DLDataTypeCode::kDLInt && cache_loc.dtype().bits == 64 &&
                       cache_loc.dim() == 1 && cache_loc.size(0) >= m && cache_loc.stride(0) == 1,
                   "cache_loc must be int64 [>= tokens], contiguous");
      RuntimeCheck(k_pool.size(1) == hkv * swa::kHeadDim && k_pool.stride(0) == hkv * swa::kHeadDim,
                   "the prologue stores pool rows of Hkv * 64 contiguous bytes");
      params.qkv = static_cast<const __nv_bfloat16*>(qkv.data_ptr());
      params.qkv_token_stride = qkv.stride(0);
      params.cache_loc = static_cast<const int64_t*>(cache_loc.data_ptr());
      params.k_store = static_cast<uint8_t*>(k_pool.data_ptr());
      params.v_store = static_cast<uint8_t*>(v_pool.data_ptr());
      params.qk_eps = static_cast<float>(qk_eps);
      if constexpr (kRope) {
        RuntimeCheck(positions.dtype().code == DLDataTypeCode::kDLInt && positions.dtype().bits == 64 &&
                         positions.dim() == 1 && positions.size(0) >= m && positions.stride(0) == 1,
                     "positions must be int64 [>= tokens], contiguous");
        RuntimeCheck(cos_sin.dtype().code == DLDataTypeCode::kDLFloat && cos_sin.dtype().bits == 32 &&
                         cos_sin.dim() == 2 && cos_sin.size(1) == swa::kHeadDim && cos_sin.stride(1) == 1 &&
                         cos_sin.stride(0) == swa::kHeadDim && aligned16(cos_sin.data_ptr()),
                     "cos_sin must be fp32 [positions, 64], contiguous");
        params.positions = static_cast<const int64_t*>(positions.data_ptr());
        params.cos_sin = static_cast<const float*>(cos_sin.data_ptr());
      }
    }
    params.k_pool = static_cast<const uint8_t*>(k_pool.data_ptr());
    params.v_pool = static_cast<const uint8_t*>(v_pool.data_ptr());
    params.page_table = static_cast<const int32_t*>(page_table.data_ptr());
    params.seq_lens = static_cast<const int32_t*>(seq_lens.data_ptr());
    params.out = static_cast<__nv_bfloat16*>(out.data_ptr());
    params.pool_row_stride = k_pool.stride(0);
    params.pool_slots = k_pool.size(0);
    params.page_table_stride = page_table.stride(0);
    params.num_q_heads = static_cast<int32_t>(hq);
    params.num_kv_heads = static_cast<int32_t>(hkv);
    params.window_left = static_cast<int32_t>(window_left);
    params.page_shift = page_shift;
    params.scale_log2 = static_cast<float>(bmm1_scale * M_LOG2E);
    params.out_scale = static_cast<float>(bmm2_scale / static_cast<double>(swa::kPScale));
    if constexpr (kNormGate) {
      RuntimeCheck(gate.dtype().code == DLDataTypeCode::kDLBfloat && gate.dtype().bits == 16 && gate.dim() == 2 &&
                       gate.size(0) == m && gate.size(1) >= hq * swa::kHeadDim && gate.stride(1) == 1 &&
                       gate.stride(0) % 4 == 0 && reinterpret_cast<uintptr_t>(gate.data_ptr()) % 8 == 0,
                   "gate must be bf16 [tokens, >= Hq*64], inner contiguous, 8-byte aligned rows");
      params.gate = static_cast<const __nv_bfloat16*>(gate.data_ptr());
      params.gate_token_stride = gate.stride(0);
      params.norm_eps = static_cast<float>(norm_eps);
    }

    constexpr auto kernel =
        swa::swa_verify_attn_kernel<kChunk, kSplit, kRG, kC, kUsePDL, kPrefetch, kNormGate, kQk, kRope, kUnfused>;
    static bool configured = false;
    if (!configured) {
      RuntimeDeviceCheck(::cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, Cfg::kBytes));
      RuntimeDeviceCheck(::cudaFuncSetAttribute(kernel, cudaFuncAttributeNonPortableClusterSizeAllowed, 1));
      configured = true;
    }
    if (reqs == 0) return;
    const dim3 grid(kC, static_cast<unsigned>(Cfg::kRowBlocks * hkv), static_cast<unsigned>(reqs));
    LaunchKernel(grid, Cfg::kThreads, out.device(), Cfg::kBytes)
        .config({.use_pdl = kUsePDL, .cluster_dim = dim3(kC, 1, 1)})(kernel, params);
  }
};

}  // namespace muse_vr
