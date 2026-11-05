// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
/**
 * \brief Paged FP8 (e4m3) causal attention for prefill (context) forwards with tensor-core MMA (tcgen05): 8 query
 * heads per KV head, head_dim 64, page size 128, bf16 output; any number of requests, each with q_len new tokens at
 * the end of its KV range (positions kv_len - q_len .. kv_len - 1).
 *
 * Shapes (per call): q [tokens, Hq, 64] e4m3 (heads and dims contiguous; the tokens of request r are
 * cu_q[r] .. cu_q[r + 1] - 1); K and V caches [pages, Hkv, 128, 64] e4m3 with arbitrary page / head / token strides
 * (dims contiguous); page_table [R, >= pages] int32; seq_lens [R] int32 (KV length including the new tokens);
 * out [tokens, Hq, 64] bf16 (heads and dims contiguous).
 *
 * Masking: the query token at position p attends keys [0, p].
 *
 * Work split: one block per work unit (host-planned list, see ops/prefill_attn.py): 32 query tokens of one request and
 * the 8 query heads of one KV head (two query tiles of 128 rows: 16 tokens x 8 heads, row = 8 * token + head) over a
 * range of 128-key tiles (pages). A unit that covers all keys [0, last position of its tokens] writes the output; long
 * key ranges can be split into several units, which write unnormalized partial results (O, running maximum, row sum)
 * that prefill_attn_merge combines exactly. Units are launched largest first.
 * Unit lists per block (cta_units given): block b runs units cta_units[b] .. cta_units[b + 1] - 1 one after another. The
 * K/V stage ring, the scores buffer and the P buffers keep one block-wide tile count across the units, so the loads and
 * the scores of a unit's first tiles are issued while the previous unit finishes; each unit's arithmetic is the same as
 * when it runs alone in a block (the per-unit state m, L and O starts fresh; O is overwritten by the unit's first P V).
 *
 * Arithmetic, in order (per query row), the same operations as the stock FP8 context kernel's softmax:
 *   s   = fp32 dot product of e4m3 q and e4m3 k (tensor-core MMA, fp32 accumulation)
 *   (c  = fp32(bmm1_scale * log2(e)), host side; c > 0)
 *   for each tile of 128 keys, in key order:
 *     m'  = max(m, c * max of s over the tile)            (running maximum, fp32)
 *     f   = exp2(m - m') if m' > m and not the first tile, else 1   (ex2.approx.ftz.f32)
 *     O   = O * f (fp32, tensor memory, only when f != 1)
 *     y   = fma(s, c, log2(448) - m')                     (fp32; y <= log2(448) up to rounding)
 *     p   = exp2(y)                                       (fp32: ex2.approx.ftz.f32; optionally an fp32 polynomial
 *                                                          for kPolyPairs of every 8 key pairs, see exp2_poly2)
 *     P   = e4m3(p)                                       (round to nearest even, saturating; P <= 448)
 *     L   = L * f + (fp32 sum of p over the tile)         (the unquantized fp32 values)
 *     O  += P * v   (tensor-core MMA, fp32 accumulation)
 *   out = bf16(O * (bmm2_scale / L))
 * Half exp mode (kPolyPairs == kHalfExp, dev only): y is rounded to half (cvt.rn.f16x2.f32), p = ex2.approx.f16x2,
 * P = e4m3 of the half p, and the tile's sum of p is accumulated in half pairs (16 values each) before the fp32 L
 * update; a reduction-order / approximation numerics class, judged by the accuracy ladder.
 * Tensor-core row-sum mode (kPolyPairs == kHalfExpTc, dev only): the half exp as above without the half sums; L is
 * accumulated by the P V issuer as P(t) x ones (a second MMA, N = 16, B = a constant e4m3 1.0 tile in shared memory)
 * into tensor-memory columns next to O, rescaled with O, so L = sum of the e4m3 P values that multiply V. One P buffer
 * (the softmax threads wait for the MMAs of tile t - 1 before writing P(t)). Same numerics class as the half exp mode.
 * Option kOptTcSum32 (kernel template argument kOpt, default off): the same tensor-core row sum with the default fp32
 * exponentials (p = ex2.approx.ftz.f32, P = e4m3(p) as above); L = the sum of the e4m3 P values instead of the fp32 sum of
 * the unquantized p, so O / L is the weighted mean of V with exactly the weights the P V MMA used.
 * Split units: out = bf16((sum_i w_i O_i) * (bmm2_scale / sum_i w_i L_i)), w_i = exp2(m_i - max_i m_i).
 *
 * Block: 16 warps in four warpgroups (registers rebalanced with setmaxnreg: the softmax warpgroups grow, the others
 * shrink).
 *   - warps 4b .. 4b + 3 (b = 0, 1): softmax of query tile b, one thread per row (tensor-memory lane = row); they write
 *     Q_b to shared memory first.
 *   - warp 8 + 2b issues S(t) = Q_b K(t)^T (M = 128, N = 128, K = 64; A = Q_b in shared memory); warp 9 + 2b issues
 *     O_b += P_b(t) V(t) (M = 128, N = 64, two MMAs of K = 64; A = P_b in tensor memory). Separate issuing warps
 *     keep the scores issue of a tile from waiting behind its P V issue.
 *   - warp 12 (one thread issuing) loads the K and V tiles of each page with the tensor memory accelerator into a ring
 *     of kStages shared-memory stages; warps 13-15 only take part in the register rebalance.
 *
 * Pipeline of query tile b, per tile t:
 *   1. The group waits on a named barrier (sync) that the scores issuer arrives at once S(t) is in tensor memory and
 *      the P V MMA of tile t - 2 has completed (the issuer waits on the MMA-completion mbarriers itself, so the group
 *      only meets once per tile; P buffer t % 2 is then free).
 *   2. The group loads S(t) into registers and arrives at a second named barrier: the scores issuer may now issue
 *      S(t + 1) into the same columns.
 *   3. Exps, row sum, e4m3 P(t) into P buffer t % 2; O is rescaled (after the P V MMA of tile t - 1 completed) only
 *      when some row's maximum grew; the group arrives at the P barrier of buffer t % 2.
 *   4. The P V issuer issues P(t) V(t), commits to the buffer's mbarrier and to the stage's release mbarrier (one
 *      arrival per query tile).
 * Option kOptDirectS (kernel template argument kOpt, default off): in step 1 each softmax warp waits on the scores MMA's
 * completion mbarrier itself (no group meeting); the P-buffer condition (P V of tile t - 2 completed) is waited for by
 * the softmax threads right before step 3's P store. Option kOptHintWait: the load and issuing warps wait with a
 * suspend-time hint instead of re-polling. Option kOptWarpS: step 1 keeps the issuer's relay but meets each softmax warp
 * at its own named barrier (no group meeting). None of these options changes any arithmetic or its order.
 * Ordering of tensor-memory accesses across threads: tcgen05.wait, tcgen05.fence::before_thread_sync, barrier or
 * mbarrier arrive; the waiting thread issues tcgen05.fence::after_thread_sync before its tensor-memory access or MMA.
 * The P V MMAs of a query tile accumulate into O from one thread (tcgen05 pipeline order). Every issuer waits for the
 * TMA completion of a stage before its MMA reads it. Q is written to shared memory by the softmax threads (generic
 * proxy) and read by the MMA (async proxy): fence.proxy.async before the barrier.
 *
 * Shared-memory operand layouts (1024-byte aligned): K stages [128 keys][64 B] (K-major B operand), V stages
 * [128 keys][64 B] (MN-major B operand, dims contiguous), Q tiles [128 rows][64 B] (K-major A operand), all in the
 * 64-byte swizzle.
 *
 * Tensor memory (512 columns): query tile b at 256 b +: S [0, 128), O [128, 192), P buffers (e4m3, 4 per column)
 * [192, 224) and [224, 256).
 */

#if defined(__CUDACC_RTC__)
// Device-only build (NVRTC): no host headers; the launcher passes the tensor maps and the parameter block by value.
typedef unsigned char uint8_t;
typedef unsigned short uint16_t;
typedef unsigned int uint32_t;
typedef signed int int32_t;
typedef unsigned long long uint64_t;
typedef signed long long int64_t;
typedef unsigned long long uintptr_t;
typedef struct __align__(64) {
  uint64_t opaque[16];
} CUtensorMap;
#ifndef INFINITY
#define INFINITY __int_as_float(0x7f800000)
#endif
#include <cuda_bf16.h>
#else
#include <sgl_kernel/tensor.h>
#include <sgl_kernel/utils.h>

#include <sgl_kernel/runtime.cuh>
#include <sgl_kernel/utils.cuh>

#include <cuda.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <dlpack/dlpack.h>
#include <tvm/ffi/container/tensor.h>

#include <cmath>
#include <cstdint>
#include <type_traits>
#endif
static_assert(sizeof(CUtensorMap) == 128, "CUtensorMap ABI");

namespace muse_vr {

namespace pfa {

constexpr int kHeadDim = 64;
constexpr int kQPerKv = 8;
constexpr int kTile = 128;  // keys per tile = page size
constexpr int kRows = 128;  // rows per query tile: 16 tokens x 8 heads
constexpr int kTokPerQTile = kRows / kQPerKv;
constexpr int kQTiles = 2;
constexpr int kTokPerBlock = kTokPerQTile * kQTiles;
constexpr int kStages = 8;  // power of two
constexpr int kSoftmaxWarps = 4 * kQTiles;
constexpr int kMmaWarp = kSoftmaxWarps;  // warp kMmaWarp + 2 b: scores of query tile b; + 2 b + 1: its P V
constexpr int kLoadWarp = kSoftmaxWarps + 4;
constexpr int kThreads = 32 * (kSoftmaxWarps + 8);
// registers per thread after the warpgroups rebalance (setmaxnreg; the launch gives 65536 / kThreads): the helper
// warpgroups release down to kHelperRegs, the softmax warpgroups grow to kSoftmaxRegs (the pool stays within 65536)
constexpr int kHelperRegs = 40, kSoftmaxRegs = 216;
static_assert(2 * 128 * kHelperRegs + 2 * 128 * kSoftmaxRegs <= 65536, "register pool");
constexpr uint32_t kGroupThreads = 32 * 4 + 32;  // named barriers: a softmax group and one issuing warp
// P = e4m3(448 p): log2(448) folded into the exponent offset
constexpr float kLog2PScale = 8.80735492205760410749f;

// Shared memory (bytes from a 1024-aligned base): K stages [128 keys][64 B]; V stages; Q tiles [128 rows][64 B].
constexpr int kKvBytes = kTile * 64;
constexpr int kSmK = 0;
constexpr int kSmV = kSmK + kStages * kKvBytes;
constexpr int kSmQ = kSmV + kStages * kKvBytes;
constexpr int kQBytes = kRows * 64;
constexpr int kSmOnes = kSmQ + kQTiles * kQBytes;  // tensor-core row-sum mode: 16 rows x 64 B of e4m3 1.0
constexpr int kSmBar = kSmOnes + 1024;
constexpr int kSmBytes = kSmBar + 256 + 1024;  // + alignment slack
constexpr uint32_t kTmemCols = 512;
constexpr uint32_t kColS = 0, kColO = 128, kColP = 192, kColTile = 256;
// tensor-core row-sum mode: one P buffer at kColP1, the row-sum accumulator (16 columns, column 0 used) at kColRS
constexpr uint32_t kColRS = 192, kColP1 = 224;
constexpr uint32_t kIdescKDouble = 1u << 29;  // K = 64 for kind::f8f6f4 (PTX ISA, instruction descriptor)

struct Params {
  const int32_t* __restrict__ units;      // [n, 8]: request, KV head, first query token, tile begin, tile end, slot
  const int32_t* __restrict__ cta_units;  // [grid + 1] unit offsets per block (null: block b runs unit b)
  float* __restrict__ ws_o;               // split units: [slots, 256 rows, 64] unnormalized fp32 O
  float2* __restrict__ ws_ml;             // split units: [slots, 256 rows] (running maximum, row sum)
  const uint8_t* __restrict__ q;
  const int32_t* __restrict__ page_table;
  const int32_t* __restrict__ seq_lens;
  const int32_t* __restrict__ cu_q;
  __nv_bfloat16* __restrict__ out;
  int64_t q_token_stride;  // bytes
  int64_t page_table_stride;
  int64_t out_token_stride;  // elements
  float scale_log2;
  float out_scale;
  long long* trace;  // debug timeline of block (0, 0, 0) (null: off)
};
// debug timeline: clock64 of event k of tile t
#define PFA_TRACE(t, k) \
  do { \
    if constexpr (kTrace) { \
      if (blockIdx.x == 0 && blockIdx.y == 0 && blockIdx.z == 0 && (t) < 2048) p.trace[(t) * 48 + (k)] = clock64(); \
    } \
  } while (0)

// ---------------------------------------------------------------------------------------------- PTX wrappers
__device__ __forceinline__ uint32_t smem_u32(const void* p) {
  return static_cast<uint32_t>(__cvta_generic_to_shared(p));
}
__device__ __forceinline__ void mbar_init(uint32_t bar, uint32_t count) {
  asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;\n" ::"r"(bar), "r"(count) : "memory");
}
__device__ __forceinline__ void mbar_expect_tx(uint32_t bar, uint32_t bytes) {
  asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;\n" ::"r"(bar), "r"(bytes) : "memory");
}
// Upper bound on try_wait calls of one wait: every arrival comes from this block (or from its own MMAs and copies);
// the bound turns a wait that can only hang through a defect into a trap.
inline constexpr uint32_t kMaxPolls = 1u << 24;
__device__ __noinline__ void wait_bound_exceeded() { __trap(); }
__device__ __forceinline__ bool mbar_try(uint32_t bar, uint32_t parity) {
  uint32_t done;
  asm volatile(
      "{\n"
      ".reg .pred p;\n"
      "mbarrier.try_wait.parity.acquire.cta.shared::cta.b64 p, [%1], %2;\n"
      "selp.u32 %0, 1, 0, p;\n"
      "}\n"
      : "=r"(done)
      : "r"(bar), "r"(parity)
      : "memory");
  return done != 0;
}
__device__ __forceinline__ void mbar_wait(uint32_t bar, uint32_t parity) {
  uint32_t polls = 0;
  while (!mbar_try(bar, parity)) {
    if (++polls == kMaxPolls) [[unlikely]] wait_bound_exceeded();
  }
}
// The same for a thread with nothing else to do: after each failed try it sleeps for about kSleepNs.
template <uint32_t kSleepNs>
__device__ __forceinline__ void mbar_wait_sleep(uint32_t bar, uint32_t parity) {
  uint32_t polls = 0;
  while (!mbar_try(bar, parity)) {
    if (++polls == kMaxPolls) [[unlikely]] wait_bound_exceeded();
    __nanosleep(kSleepNs);
  }
}
// The same with a suspend-time hint (PTX ISA, mbarrier.try_wait timeHint): a waiting thread may stay suspended inside the
// instruction until the phase completes or the hint elapses, instead of re-polling (option kOptHintWait).
__device__ __forceinline__ bool mbar_try_hint(uint32_t bar, uint32_t parity, uint32_t hint_ns) {
  uint32_t done;
  asm volatile(
      "{\n"
      ".reg .pred p;\n"
      "mbarrier.try_wait.parity.acquire.cta.shared::cta.b64 p, [%1], %2, %3;\n"
      "selp.u32 %0, 1, 0, p;\n"
      "}\n"
      : "=r"(done)
      : "r"(bar), "r"(parity), "r"(hint_ns)
      : "memory");
  return done != 0;
}
template <uint32_t kHintNs>
__device__ __forceinline__ void mbar_wait_hint(uint32_t bar, uint32_t parity) {
  uint32_t polls = 0;
  while (!mbar_try_hint(bar, parity, kHintNs)) {
    if (++polls == kMaxPolls) [[unlikely]] wait_bound_exceeded();
  }
}
// named barriers (PTX barrier.cta): producers arrive, the consumer syncs
__device__ __forceinline__ void named_arrive(uint32_t id, uint32_t n) {
  asm volatile("barrier.cta.arrive.aligned %0, %1;\n" ::"r"(id), "r"(n) : "memory");
}
__device__ __forceinline__ void named_sync(uint32_t id, uint32_t n) {
  asm volatile("barrier.cta.sync.aligned %0, %1;\n" ::"r"(id), "r"(n) : "memory");
}
__device__ __forceinline__ void tma_load_4d(uint32_t dst, const CUtensorMap* map, int32_t c0, int32_t c1, int32_t c2,
                                            int32_t c3, uint32_t bar) {
  asm volatile(
      "cp.async.bulk.tensor.4d.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1, {%2, %3, %4, %5}], "
      "[%6];\n" ::"r"(dst),
      "l"(reinterpret_cast<uint64_t>(map)), "r"(c0), "r"(c1), "r"(c2), "r"(c3), "r"(bar)
      : "memory");
}
__device__ __forceinline__ void tc_fence_before() { asm volatile("tcgen05.fence::before_thread_sync;\n" ::: "memory"); }
__device__ __forceinline__ void tc_fence_after() { asm volatile("tcgen05.fence::after_thread_sync;\n" ::: "memory"); }
// MMA with A and B from shared memory
__device__ __forceinline__ void mma_ss(uint32_t d, uint64_t a, uint64_t b, uint32_t idesc, uint32_t accumulate) {
  asm volatile(
      "{\n"
      ".reg .pred p;\n"
      "setp.ne.b32 p, %4, 0;\n"
      "tcgen05.mma.cta_group::1.kind::f8f6f4 [%0], %1, %2, %3, p;\n"
      "}\n" ::"r"(d),
      "l"(a), "l"(b), "r"(idesc), "r"(accumulate)
      : "memory");
}
// MMA with A from tensor memory, B from shared memory
__device__ __forceinline__ void mma_ts(uint32_t d, uint32_t a_tmem, uint64_t b, uint32_t idesc, uint32_t accumulate) {
  asm volatile(
      "{\n"
      ".reg .pred p;\n"
      "setp.ne.b32 p, %4, 0;\n"
      "tcgen05.mma.cta_group::1.kind::f8f6f4 [%0], [%1], %2, %3, p;\n"
      "}\n" ::"r"(d),
      "r"(a_tmem), "l"(b), "r"(idesc), "r"(accumulate)
      : "memory");
}
__device__ __forceinline__ void mma_commit(uint32_t bar) {
  asm volatile("tcgen05.commit.cta_group::1.mbarrier::arrive::one.shared::cluster.b64 [%0];\n" ::"r"(bar) : "memory");
}

#define PFA_R8(i) "=r"(r[i]), "=r"(r[i + 1]), "=r"(r[i + 2]), "=r"(r[i + 3]), "=r"(r[i + 4]), "=r"(r[i + 5]), \
                  "=r"(r[i + 6]), "=r"(r[i + 7])
#define PFA_W8(i) "r"(r[i]), "r"(r[i + 1]), "r"(r[i + 2]), "r"(r[i + 3]), "r"(r[i + 4]), "r"(r[i + 5]), "r"(r[i + 6]), \
                  "r"(r[i + 7])
// 32 consecutive fp32 columns of this thread's tensor-memory lane, and their maximum
__device__ __forceinline__ float tmem_ld32_max(uint32_t taddr, float* v) {
  uint32_t r[32];
  float mx;
  asm volatile(
      "tcgen05.ld.red.sync.aligned.32x32b.x32.max.f32 {%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17,"
      "%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31}, %32, [%33];\n"
      : PFA_R8(0), PFA_R8(8), PFA_R8(16), PFA_R8(24), "=f"(mx)
      : "r"(taddr));
#pragma unroll
  for (int i = 0; i < 32; ++i) v[i] = __uint_as_float(r[i]);
  return mx;
}
__device__ __forceinline__ void tmem_ld32(uint32_t taddr, float* v) {
  uint32_t r[32];
  asm volatile(
      "tcgen05.ld.sync.aligned.32x32b.x32.b32 {%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17,%18,%19,"
      "%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31}, [%32];\n"
      : PFA_R8(0), PFA_R8(8), PFA_R8(16), PFA_R8(24)
      : "r"(taddr));
#pragma unroll
  for (int i = 0; i < 32; ++i) v[i] = __uint_as_float(r[i]);
}
__device__ __forceinline__ void tmem_ld16(uint32_t taddr, float* v) {
  uint32_t r[16];
  asm volatile(
      "tcgen05.ld.sync.aligned.32x32b.x16.b32 {%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15}, [%16];\n"
      : PFA_R8(0), PFA_R8(8)
      : "r"(taddr));
#pragma unroll
  for (int i = 0; i < 16; ++i) v[i] = __uint_as_float(r[i]);
}
__device__ __forceinline__ void tmem_st16(uint32_t taddr, const float* v) {
  uint32_t r[16];
#pragma unroll
  for (int i = 0; i < 16; ++i) r[i] = __float_as_uint(v[i]);
  asm volatile(
      "tcgen05.st.sync.aligned.32x32b.x16.b32 [%0], {%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16};\n" ::"r"(
          taddr),
      PFA_W8(0), PFA_W8(8)
      : "memory");
}
__device__ __forceinline__ void tmem_st16u(uint32_t taddr, const uint32_t* r) {
  asm volatile(
      "tcgen05.st.sync.aligned.32x32b.x16.b32 [%0], {%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16};\n" ::"r"(
          taddr),
      PFA_W8(0), PFA_W8(8)
      : "memory");
}
__device__ __forceinline__ float tmem_ld1(uint32_t taddr) {
  uint32_t r;
  asm volatile("tcgen05.ld.sync.aligned.32x32b.x1.b32 {%0}, [%1];\n" : "=r"(r) : "r"(taddr));
  return __uint_as_float(r);
}
__device__ __forceinline__ void tmem_st1(uint32_t taddr, float v) {
  asm volatile("tcgen05.st.sync.aligned.32x32b.x1.b32 [%0], {%1};\n" ::"r"(taddr), "r"(__float_as_uint(v)) : "memory");
}
__device__ __forceinline__ void tmem_wait_ld() { asm volatile("tcgen05.wait::ld.sync.aligned;\n" ::: "memory"); }
__device__ __forceinline__ void tmem_wait_st() { asm volatile("tcgen05.wait::st.sync.aligned;\n" ::: "memory"); }

// Shared-memory matrix descriptor: start >> 4 [0,14), leading byte offset >> 4 [16,30), stride byte offset >> 4
// [32,46), descriptor version 1 [46,48), swizzle mode [61,64) (PTX ISA, tcgen05 shared memory descriptor).
__device__ __forceinline__ uint64_t sdesc(uint32_t saddr, uint32_t sbo, uint32_t layout) {
  return static_cast<uint64_t>((saddr >> 4) & 0x3FFF) | (static_cast<uint64_t>(1) << 16) |
         (static_cast<uint64_t>((sbo >> 4) & 0x3FFF) << 32) | (static_cast<uint64_t>(1) << 46) |
         (static_cast<uint64_t>(layout & 7) << 61);
}
constexpr uint32_t kSw64 = 4;
// Instruction descriptor, kind::f8f6f4, e4m3 x e4m3 -> fp32 (PTX ISA, tcgen05 instruction descriptor).
constexpr uint32_t idesc_f8(int m, int n, int b_mn_major) {
  return (1u << 4) | (static_cast<uint32_t>(b_mn_major) << 16) | (static_cast<uint32_t>(n >> 3) << 17) |
         (static_cast<uint32_t>(m >> 4) << 24);
}

__device__ __forceinline__ float fast_exp2(float x) {
  float y;
  asm("ex2.approx.ftz.f32 %0, %1;\n" : "=f"(y) : "f"(x));
  return y;
}
__device__ __forceinline__ uint64_t f2_u64(float2 a) {
  return static_cast<uint64_t>(__float_as_uint(a.x)) | (static_cast<uint64_t>(__float_as_uint(a.y)) << 32);
}
__device__ __forceinline__ float2 u64_f2(uint64_t a) {
  return make_float2(__uint_as_float(static_cast<uint32_t>(a)), __uint_as_float(static_cast<uint32_t>(a >> 32)));
}
// Packed fp32 pairs (PTX ISA .f32x2): two independent IEEE fp32 operations, round to nearest even.
__device__ __forceinline__ float2 fma2(float2 a, float2 b, float2 c) {
  uint64_t d;
  asm("fma.rn.f32x2 %0, %1, %2, %3;\n" : "=l"(d) : "l"(f2_u64(a)), "l"(f2_u64(b)), "l"(f2_u64(c)));
  return u64_f2(d);
}
__device__ __forceinline__ float2 add2(float2 a, float2 b) {
  uint64_t d;
  asm("add.rn.f32x2 %0, %1, %2;\n" : "=l"(d) : "l"(f2_u64(a)), "l"(f2_u64(b)));
  return u64_f2(d);
}
__device__ __forceinline__ float2 sub2(float2 a, float2 b) {
  uint64_t d;
  asm("sub.rn.f32x2 %0, %1, %2;\n" : "=l"(d) : "l"(f2_u64(a)), "l"(f2_u64(b)));
  return u64_f2(d);
}
// exp2 of two fp32 values on the FMA pipe (optional, kPolyPairs > 0). y is clamped to >= -125.5 (results below
// 2^-125.5 become 2^-125.5: they are far below an e4m3 subnormal and below the fp32 resolution of a row sum that holds
// 448); y = n + r with n = round(y) (magic-constant addition, round to nearest even) and |r| <= 1/2 (exact); 2^r by a
// degree-5 polynomial (minimax for relative error on [-1/2, 1/2], fp32 coefficients, Horner with fused multiply-adds:
// max relative error below 2^-22 over every fp32 r in the range); 2^n applied by adding n to the exponent field.
__device__ __forceinline__ float2 exp2_poly2(float2 y) {
  y.x = fmaxf(y.x, -125.5f);
  y.y = fmaxf(y.y, -125.5f);
  const float2 magic = make_float2(12582912.0f, 12582912.0f);  // 1.5 * 2^23
  const float2 t = add2(y, magic);
  const float2 n = sub2(t, magic);
  const float2 r = sub2(y, n);
  float2 q = fma2(r, make_float2(0x1.5c08e6p-10f, 0x1.5c08e6p-10f), make_float2(0x1.3d0c52p-7f, 0x1.3d0c52p-7f));
  q = fma2(q, r, make_float2(0x1.c6b6e4p-5f, 0x1.c6b6e4p-5f));
  q = fma2(q, r, make_float2(0x1.ebf918p-3f, 0x1.ebf918p-3f));
  q = fma2(q, r, make_float2(0x1.62e428p-1f, 0x1.62e428p-1f));
  q = fma2(q, r, make_float2(0x1.000002p+0f, 0x1.000002p+0f));
  // the low bits of t hold n (two's complement after the shift)
  return make_float2(__uint_as_float(__float_as_uint(q.x) + (__float_as_uint(t.x) << 23)),
                     __uint_as_float(__float_as_uint(q.y) + (__float_as_uint(t.y) << 23)));
}
// two fp32 values -> two e4m3 bytes (round to nearest even, saturating), a in the low byte
__device__ __forceinline__ uint32_t e4m3x2(float a, float b) {
  uint16_t h;
  asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;\n" : "=h"(h) : "f"(b), "f"(a));
  return h;
}
// Half-precision exp mode (kPolyPairs == kHalfExp). PTX ISA: cvt.rn.f16x2.f32 (a in the low half), ex2.approx.f16x2,
// cvt.rn.satfinite.e4m3x2.f16x2 (low half -> low byte), add.rn.f16x2.
__device__ __forceinline__ uint32_t h2_from_f32(float a, float b) {
  uint32_t d;
  asm("cvt.rn.f16x2.f32 %0, %1, %2;\n" : "=r"(d) : "f"(b), "f"(a));
  return d;
}
__device__ __forceinline__ uint32_t ex2_h2(uint32_t a) {
  uint32_t d;
  asm("ex2.approx.f16x2 %0, %1;\n" : "=r"(d) : "r"(a));
  return d;
}
__device__ __forceinline__ uint32_t e4m3x2_h2(uint32_t a) {
  uint16_t h;
  asm("cvt.rn.satfinite.e4m3x2.f16x2 %0, %1;\n" : "=h"(h) : "r"(a));
  return h;
}
__device__ __forceinline__ uint32_t hadd2(uint32_t a, uint32_t b) {
  uint32_t d;
  asm("add.rn.f16x2 %0, %1, %2;\n" : "=r"(d) : "r"(a), "r"(b));
  return d;
}
__device__ __forceinline__ float2 h2_to_f2(uint32_t a) {
  float lo, hi;
  asm("{\n .reg .f16 l, h;\n mov.b32 {l, h}, %2;\n cvt.f32.f16 %0, l;\n cvt.f32.f16 %1, h;\n}\n"
      : "=f"(lo), "=f"(hi) : "r"(a));
  return make_float2(lo, hi);
}
constexpr int kHalfExp = 16;
constexpr int kHalfExpTc = 17;  // half exp, no row sum on the softmax threads: the P V issuer adds P x ones on the tensor core

// Kernel options (kOpt bits, default 0 = the original form). Synchronization only (no arithmetic change):
//   kOptDirectS: each softmax warp waits on the scores MMA's completion mbarrier itself, instead of a group-wide named
//     barrier that the scores issuer arrives at after waiting on that mbarrier; the P-buffer-free condition (the P V MMA
//     of tile t - 2 completed) is then checked by the softmax threads right before they write P(t).
//   kOptHintWait: the load warp and the issuing warps wait with mbarrier.try_wait plus a suspend-time hint instead of
//     re-polling (the load warp's try-and-sleep loop).
// Arithmetic (a different normalization sum, judged against a float64 reference):
//   kOptTcSum32: fp32 exp2 as in the default mode, and the row sum L taken by the tensor core as P x ones (the sum of the
//     e4m3 P values that multiply V, fp32 accumulation) instead of the fp32 sum of the unquantized p; the tensor-core
//     row-sum form of the half exp mode (one P buffer per query tile), with fp32 exponentials.
constexpr int kOptDirectS = 1;
constexpr int kOptHintWait = 2;
constexpr int kOptTcSum32 = 4;
// Synchronization only (no arithmetic change), exclusive with kOptDirectS:
//   kOptWarpS: the scores issuer's relay (it waits on the scores MMA's completion and on the P buffer as in the default form)
//     arrives at one named barrier per softmax warp (the warp's 32 threads + the issuer's 32) instead of one barrier of the
//     whole group, so each softmax warp waits only for the relay, not for the group's other warps.
constexpr int kOptWarpS = 8;
constexpr uint32_t kWaitHintNs = 8192;  // tunable

// ---------------------------------------------------------------------------------------------- main kernel
// exp2, row sum and e4m3 packing of the 128 keys of a tile: pk[i] = keys 4 i .. 4 i + 3 (lowest key in the lowest
// byte), acc[pair % 4] += the fp32 pairs; pairs with (pair % 8) < kPolyPairs use the polynomial
template <int kPolyPairs, bool kNoSum = false>
__device__ __forceinline__ void exp_tile(const float* x, float2 c2, float2 nb2, uint32_t* pk, float2* acc) {
  if constexpr (kNoSum) {
    // fp32 exp2 and e4m3 packing only (kOptTcSum32: the row sum comes from the tensor core)
    static_assert(kPolyPairs == 0, "the tensor-core row sum with fp32 exp uses the hardware exp2 only");
#pragma unroll
    for (int i = 0; i < 32; ++i) {
      float2 pr[2];
#pragma unroll
      for (int h = 0; h < 2; ++h) {
        const int pair = 2 * i + h;
        const float2 y = fma2(make_float2(x[2 * pair], x[2 * pair + 1]), c2, nb2);
        pr[h] = make_float2(fast_exp2(y.x), fast_exp2(y.y));
      }
      pk[i] = e4m3x2(pr[0].x, pr[0].y) | (e4m3x2(pr[1].x, pr[1].y) << 16);
    }
    return;
  } else if constexpr (kPolyPairs == kHalfExpTc) {
#pragma unroll
    for (int i = 0; i < 32; ++i) {
      uint32_t w[2];
#pragma unroll
      for (int h = 0; h < 2; ++h) {
        const int pair = 2 * i + h;
        const float2 y = fma2(make_float2(x[2 * pair], x[2 * pair + 1]), c2, nb2);
        w[h] = e4m3x2_h2(ex2_h2(h2_from_f32(y.x, y.y)));
      }
      pk[i] = w[0] | (w[1] << 16);
    }
    return;
  } else if constexpr (kPolyPairs == kHalfExp) {
    // y in fp32 as below, rounded to half; p = half exp2(y); P = e4m3(p) from the half value; the tile's row sum is
    // accumulated in four half pairs (16 values each) and added to acc in fp32
    uint32_t hacc[4] = {0u, 0u, 0u, 0u};
#pragma unroll
    for (int i = 0; i < 32; ++i) {
      uint32_t w[2];
#pragma unroll
      for (int h = 0; h < 2; ++h) {
        const int pair = 2 * i + h;
        const float2 y = fma2(make_float2(x[2 * pair], x[2 * pair + 1]), c2, nb2);
        const uint32_t ph = ex2_h2(h2_from_f32(y.x, y.y));
        hacc[pair & 3] = hadd2(hacc[pair & 3], ph);
        w[h] = e4m3x2_h2(ph);
      }
      pk[i] = w[0] | (w[1] << 16);
    }
#pragma unroll
    for (int j = 0; j < 4; ++j) acc[j] = add2(acc[j], h2_to_f2(hacc[j]));
    return;
  }
#pragma unroll
  for (int i = 0; i < 32; ++i) {
    float2 pr[2];
#pragma unroll
    for (int h = 0; h < 2; ++h) {
      const int pair = 2 * i + h;
      const float2 y = fma2(make_float2(x[2 * pair], x[2 * pair + 1]), c2, nb2);
      if ((pair & 7) < kPolyPairs) {
        pr[h] = exp2_poly2(y);
      } else {
        pr[h] = make_float2(fast_exp2(y.x), fast_exp2(y.y));
      }
      acc[pair & 3] = add2(acc[pair & 3], pr[h]);
    }
    pk[i] = e4m3x2(pr[0].x, pr[0].y) | (e4m3x2(pr[1].x, pr[1].y) << 16);
  }
}

// kPolyPairs: polynomial exp2 for this many of every 8 key pairs (the rest on ex2.approx.ftz.f32)
// kOpt: synchronization option bits (kOptDirectS, kOptHintWait; 0 = the original form)
template <bool kUsePDL, bool kTrace, int kPolyPairs, int kOpt = 0>
__global__ void __launch_bounds__(kThreads, 1)
    prefill_attn_kernel(const Params p, const __grid_constant__ CUtensorMap k_map,
                        const __grid_constant__ CUtensorMap v_map) {
  static_assert((kOpt & ~(kOptDirectS | kOptHintWait | kOptTcSum32 | kOptWarpS)) == 0, "unknown option bits");
  static_assert((kOpt & kOptDirectS) == 0 || (kOpt & kOptWarpS) == 0, "kOptDirectS and kOptWarpS are exclusive");
  constexpr bool kWarpS = (kOpt & kOptWarpS) != 0;
  static_assert((kOpt & kOptTcSum32) == 0 || kPolyPairs == 0, "kOptTcSum32 goes with the hardware fp32 exp2 only");
  constexpr bool kTcSum32 = (kOpt & kOptTcSum32) != 0;
  constexpr bool kDirectS = (kOpt & kOptDirectS) != 0;
  constexpr bool kHintWait = (kOpt & kOptHintWait) != 0;
  extern __shared__ uint8_t smem_raw[];
  uint8_t* const sm = reinterpret_cast<uint8_t*>((reinterpret_cast<uintptr_t>(smem_raw) + 1023) & ~uintptr_t{1023});
  uint64_t* const bars = reinterpret_cast<uint64_t*>(sm + kSmBar);
  // mbarriers: full[kStages] (TMA), empty[kStages] (stage released by both P V issuers), s_full[2] (S(t) MMA done),
  // pv_done[2][2] (P V MMA of the tile that used P buffer j done)
  const uint32_t bar0 = smem_u32(bars);
  const auto full_bar = [&](int s) { return bar0 + 8u * s; };
  const auto empty_bar = [&](int s) { return bar0 + 8u * (kStages + s); };
  const auto s_full_bar = [&](int b) { return bar0 + 8u * (2 * kStages + b); };
  const auto pv_done_bar = [&](int b, int j) { return bar0 + 8u * (2 * kStages + 2 + 2 * b + j); };
  uint32_t* const tmem_slot = reinterpret_cast<uint32_t*>(sm + kSmBar + 8 * (2 * kStages + 6));
  // named barriers of query tile b (128 softmax threads + 32 of one issuing warp): S columns read / Q written
  // (1 + 4 b, scores issuer syncs); P buffer j written (2 + 4 b + j, P V issuer syncs); S(t) ready and P buffer
  // t % 2 free (4 + 4 b, the group syncs)
  const auto sfree_id = [](int b) { return static_cast<uint32_t>(1 + 4 * b); };
  const auto pfull_id = [](int b, int j) { return static_cast<uint32_t>(2 + 4 * b + j); };
  const auto sready_id = [](int b) { return static_cast<uint32_t>(4 + 4 * b); };
  // kOptWarpS: one S-ready barrier per softmax warp (query tile b, warp wq of its group): ids 4 and 8 (the group barriers'
  // ids, unused in this form) and 9-14; 64 threads each (the softmax warp + the scores issuer)
  const auto sready_w_id = [](int b, int wq) {
    return static_cast<uint32_t>(wq == 0 ? 4 + 4 * b : 8 + 3 * b + wq);
  };
  constexpr uint32_t kWarpPairThreads = 64;

  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
  // work unit: key tiles [tile0, tile0 + ntiles) of the 32 query tokens tok0 .. tok0 + 31 of request req, KV head kvh.
  // Each role reads the unit fields and the tensor-memory base itself after the register rebalance, so that nothing
  // computed from them stays live across it (it would be parked in local memory and reloaded in the loops).
  // The block's units: [u0, u1) of p.units (one unit per block when p.cta_units is null).
  const auto unit_range = [&](int& u0, int& u1) {
    if (p.cta_units == nullptr) {
      u0 = static_cast<int>(blockIdx.x);
      u1 = u0 + 1;
    } else {
      u0 = p.cta_units[blockIdx.x];
      u1 = p.cta_units[blockIdx.x + 1];
    }
  };
  const auto unit_at = [&](int u) { return p.units + 8 * static_cast<int64_t>(u); };

  constexpr bool kTcSum = kPolyPairs == kHalfExpTc || kTcSum32;
  // Invariant: no global-memory access of any kind before this wait (the launch may overlap the predecessor's tail).
  if constexpr (kUsePDL) asm volatile("griddepcontrol.wait;\n" ::: "memory");

  if constexpr (kTcSum) {
    // B operand of the row-sum MMA: 16 rows (N) x 64 B (K) of e4m3 1.0, K-major 64-byte swizzle (any permutation of
    // equal bytes is the same tile); generic stores, ordered before the MMA reads by the proxy fence and the barrier below
    if (tid < 64) {
      reinterpret_cast<uint4*>(sm + kSmOnes)[tid] = make_uint4(0x38383838u, 0x38383838u, 0x38383838u, 0x38383838u);
      asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory");
    }
  }
  if (tid == 0) {
#pragma unroll
    for (int s = 0; s < kStages; ++s) {
      mbar_init(full_bar(s), 1);
      mbar_init(empty_bar(s), kQTiles);
    }
#pragma unroll
    for (int b = 0; b < kQTiles; ++b) {
      mbar_init(s_full_bar(b), 1);
      mbar_init(pv_done_bar(b, 0), 1);
      mbar_init(pv_done_bar(b, 1), 1);
    }
    asm volatile("fence.mbarrier_init.release.cluster;\n" ::: "memory");
  }
  if (warp == 0) {
    asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], %1;\n" ::"r"(smem_u32(tmem_slot)),
                 "r"(kTmemCols)
                 : "memory");
    asm volatile("tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned;\n" ::: "memory");
  }
  tc_fence_before();
  __syncthreads();
  tc_fence_after();

  if (warp > kLoadWarp) {
    asm volatile("setmaxnreg.dec.sync.aligned.u32 %0;\n" ::"n"(kHelperRegs) : "memory");
  } else if (warp == kLoadWarp) {
    asm volatile("setmaxnreg.dec.sync.aligned.u32 %0;\n" ::"n"(kHelperRegs) : "memory");
    // ------------------------------------------------------------------ loads
    if (lane == 0) {
      // K/V rows of the new tokens may have been written by a preceding kernel with generic-proxy stores; order the
      // proxies before the async-proxy reads (PTX ISA, memory consistency model: proxy fences).
      asm volatile("fence.proxy.async.global;\n" ::: "memory");
    }
    int u0, u1;
    unit_range(u0, u1);
    int g0 = 0;  // tiles of the block's earlier units (stage ring position)
    for (int u = u0; u < u1; ++u) {
      const int32_t* const unit = unit_at(u);
      const int req = unit[0], kvh = unit[1], ntiles = unit[4] - unit[3];
      const int32_t* const pt = p.page_table + static_cast<int64_t>(req) * p.page_table_stride + unit[3];
      int next_page = lane < ntiles ? pt[lane] : 0;
      for (int base = 0; base < ntiles; base += 32) {
        const int cur_page = next_page;
        next_page = base + 32 + lane < ntiles ? pt[base + 32 + lane] : 0;
        const int n = min(32, ntiles - base);
        for (int i = 0; i < n; ++i) {
          const int t = g0 + base + i;
          const int page = __shfl_sync(0xffffffffu, cur_page, i);
          if (lane == 0) {
            const int s = t & (kStages - 1);
            if (t >= kStages) {
              if constexpr (kHintWait) {
                mbar_wait_hint<kWaitHintNs>(empty_bar(s), ((t / kStages) - 1) & 1);
              } else {
                mbar_wait_sleep<256>(empty_bar(s), ((t / kStages) - 1) & 1);
              }
            }
            mbar_expect_tx(full_bar(s), 2 * kKvBytes);
            tma_load_4d(smem_u32(sm + kSmK + s * kKvBytes), &k_map, 0, 0, kvh, page, full_bar(s));
            tma_load_4d(smem_u32(sm + kSmV + s * kKvBytes), &v_map, 0, 0, kvh, page, full_bar(s));
          }
          __syncwarp();
        }
      }
      g0 += ntiles;
    }
  } else if (warp >= kMmaWarp) {
    asm volatile("setmaxnreg.dec.sync.aligned.u32 %0;\n" ::"n"(kHelperRegs) : "memory");
    // ------------------------------------------------------------------ MMA issue
    const uint32_t tmem = *tmem_slot;
    const int b = (warp - kMmaWarp) >> 1;
    const bool is_pv = ((warp - kMmaWarp) & 1) != 0;
    const uint32_t tb = tmem + kColTile * b;
    constexpr uint32_t id_qk = idesc_f8(128, 128, 0) | kIdescKDouble;
    constexpr uint32_t id_pv = idesc_f8(128, 64, 1) | kIdescKDouble;
    // descriptors of stage 0; stage s adds s * stage bytes / 16 to the start-address field (no carry: < 256 KB)
    const uint64_t dk0 = sdesc(smem_u32(sm + kSmK), 512, kSw64);
    const uint64_t dv0 = sdesc(smem_u32(sm + kSmV), 512, kSw64);
    const uint64_t dq = sdesc(smem_u32(sm + kSmQ + b * kQBytes), 512, kSw64);
    int u0, u1;
    unit_range(u0, u1);
    int g0 = 0;  // tiles of the block's earlier units: stage ring, scores and P-buffer phases run on across units
    for (int u = u0; u < u1; ++u) {
    const int ntiles = unit_at(u)[4] - unit_at(u)[3];
    for (int lt = 0; lt < ntiles; ++lt) {
      const int t = g0 + lt;
      const uint32_t st = static_cast<uint32_t>(t) & (kStages - 1);
      if (!is_pv) {
        // S(t): K(t) landed (waited for before parking at the barrier); the group has read S(t - 1) / written Q
        if (lane == 0) {
          if constexpr (kHintWait) {
            mbar_wait_hint<kWaitHintNs>(full_bar(st), (t / kStages) & 1);
          } else {
            mbar_wait(full_bar(st), (t / kStages) & 1);
          }
        }
        __syncwarp();
        named_sync(sfree_id(b), kGroupThreads);
        if (lane == 0) {
          PFA_TRACE(t, 8 + 4 * b);
          tc_fence_after();
          mma_ss(tb + kColS, dq, dk0 + st * (kKvBytes >> 4), id_qk, 0);
          mma_commit(s_full_bar(b));
          if constexpr (!kDirectS) {
            // relay: S(t) complete and the P V MMA of tile t - 2 (P buffer t % 2) complete
            mbar_wait(s_full_bar(b), t & 1);
            if (t >= 2) mbar_wait(pv_done_bar(b, t & 1), ((t >> 1) - 1) & 1);
            tc_fence_before();
          }
          PFA_TRACE(t, 9 + 4 * b);
        }
        __syncwarp();
        // kDirectS: the softmax threads wait on s_full themselves (S(t + 1) cannot be issued, and s_full cannot complete
        // again, before every softmax thread of the group has read S(t) and arrived at sfree)
        if constexpr (kWarpS) {
          // one relay arrival per softmax warp (barrier ids are per warp; each completes once per tile: the next arrival
          // follows S(t + 1), which needs every warp's sfree arrival after its sync for tile t)
#pragma unroll
          for (int w = 0; w < 4; ++w) named_arrive(sready_w_id(b, w), kWarpPairThreads);
        } else if constexpr (!kDirectS) {
          named_arrive(sready_id(b), kGroupThreads);
        }
      } else {
        // O += P(t) V(t) once the group has written P(t) (and rescaled O)
        const int j = t & 1;
        named_sync(pfull_id(b, j), kGroupThreads);
        if (lane == 0) {
          PFA_TRACE(t, 10 + 4 * b);
          mbar_wait(full_bar(st), (t / kStages) & 1);  // V(t) landed (completed long ago; orders the reads)
          tc_fence_after();
          const uint64_t dv = dv0 + st * (kKvBytes >> 4);
          if constexpr (kTcSum) {
            constexpr uint32_t id_rs = idesc_f8(128, 16, 0) | kIdescKDouble;
            const uint64_t d1 = sdesc(smem_u32(sm + kSmOnes), 512, kSw64);
            mma_ts(tb + kColO, tb + kColP1, dv, id_pv, lt > 0 ? 1u : 0u);
            mma_ts(tb + kColO, tb + kColP1 + 16, dv + (4096 >> 4), id_pv, 1u);
            mma_ts(tb + kColRS, tb + kColP1, d1, id_rs, lt > 0 ? 1u : 0u);
            mma_ts(tb + kColRS, tb + kColP1 + 16, d1, id_rs, 1u);
          } else {
            mma_ts(tb + kColO, tb + kColP + 32 * j, dv, id_pv, lt > 0 ? 1u : 0u);  // a unit's first P V starts O
            mma_ts(tb + kColO, tb + kColP + 32 * j + 16, dv + (4096 >> 4), id_pv, 1u);
          }
          mma_commit(pv_done_bar(b, j));
          mma_commit(empty_bar(st));
          PFA_TRACE(t, 11 + 4 * b);
        }
        __syncwarp();
      }
    }
    g0 += ntiles;
    }
  } else {
    asm volatile("setmaxnreg.inc.sync.aligned.u32 %0;\n" ::"n"(kSoftmaxRegs) : "memory");
    // ------------------------------------------------------------------ softmax (warps 0-7, one thread per row)
    const uint32_t tmem = *tmem_slot;
    const int b = warp >> 2, wq = warp & 3;
    const int row = 32 * wq + lane;  // tensor-memory lane
    const uint32_t tl = tmem + ((32u * wq) << 16) + kColTile * b;
    const float c = p.scale_log2;
    int u0, u1;
    unit_range(u0, u1);
    int g0 = 0;  // tiles of the block's earlier units (P-buffer and mbarrier phase indices run on across units)
    for (int u = u0; u < u1; ++u) {
    const int32_t* const unit = unit_at(u);
    const int req = unit[0], kvh = unit[1], tok0 = unit[2], tile0 = unit[3], ntiles = unit[4] - unit[3];
    const int q_begin = p.cu_q[req];
    const int q_len = p.cu_q[req + 1] - q_begin;
    const int prefix = p.seq_lens[req] - q_len;
    const int tok = tok0 + b * kTokPerQTile + (row >> 3);
    const int pos = prefix + min(tok, q_len - 1);  // rows past the request's tokens are computed but not stored
    {
      // Q row (64 e4m3) -> shared memory, K-major in the 64-byte swizzle (16-byte chunk j of row r at chunk
      // j ^ ((r >> 1) & 3)). For a later unit of the block, the scores MMAs of the earlier unit (async-proxy reads of
      // Q) have completed: the group has read their results after the issuer's mbarrier wait and the barrier relay
      // (kDirectS: after each thread's own wait on that mbarrier).
      // The proxy fence orders those reads before the generic writes below (PTX ISA, memory consistency model: proxy
      // fences); the second one orders the writes before the next scores MMA reads them.
      if (u > u0) asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory");
      const uint4* const src = reinterpret_cast<const uint4*>(
          p.q + static_cast<int64_t>(q_begin + min(tok, q_len - 1)) * p.q_token_stride +
          (kvh * kQPerKv + (row & 7)) * kHeadDim);
      uint8_t* const dst = sm + kSmQ + b * kQBytes + row * 64;
      uint4 w[4];
#pragma unroll
      for (int j = 0; j < 4; ++j) w[j] = src[j];
#pragma unroll
      for (int j = 0; j < 4; ++j) *reinterpret_cast<uint4*>(dst + ((j ^ ((row >> 1) & 3)) << 4)) = w[j];
      asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory");
      tc_fence_before();  // the earlier unit's last scores load (tcgen05.ld) before the next scores MMA overwrites them
      named_arrive(sfree_id(b), kGroupThreads);
    }
    // running maximum starts at -1e30 (finite): a row can have no key yet in a split unit's first tiles (the diagonal
    // straddles a tile boundary); its masked scores (-inf) still give p = 0 and L = 0, and the first real maximum
    // rescales by exp2(-1e30 - m') = 0. A unit with no key for a row leaves m = -1e30, L = 0 (merge weight 0). Rows
    // of an unsplit unit always have key 0 in their first tile.
    float m = -1e30f, l = 0.0f;
    for (int lt = 0; lt < ntiles; ++lt) {
      const int t = g0 + lt;  // block-wide tile index: P buffer t % 2 and the mbarrier phases
      const int j = t & 1;
      // S(t) is in tensor memory and P buffer j is free (kDirectS: S(t) only; the P buffer is checked before the P store)
      if constexpr (kDirectS) {
        mbar_wait(s_full_bar(b), t & 1);
      } else if constexpr (kWarpS) {
        named_sync(sready_w_id(b, wq), kWarpPairThreads);  // this warp and the relay only
      } else {
        named_sync(sready_id(b), kGroupThreads);
      }
      if (lane == 0) PFA_TRACE(t, 32 + 4 * b + wq);
      tc_fence_after();
      float x[128];
      float mx = tmem_ld32_max(tl + kColS, x);
      mx = fmaxf(mx, tmem_ld32_max(tl + kColS + 32, x + 32));
      mx = fmaxf(mx, tmem_ld32_max(tl + kColS + 64, x + 64));
      mx = fmaxf(mx, tmem_ld32_max(tl + kColS + 96, x + 96));
      tmem_wait_ld();
      if (lt + 1 < ntiles) {
        tc_fence_before();
        named_arrive(sfree_id(b), kGroupThreads);  // the scores are in registers: S(t + 1) may overwrite them
      }
      const int lim = pos - (tile0 + lt) * kTile;  // keys 0 .. lim of the tile take part
      if (__any_sync(0xffffffffu, lim < kTile - 1)) {  // warp-uniform: only tiles that cross a row's position
        mx = -INFINITY;
#pragma unroll
        for (int i = 0; i < 128; ++i) {
          x[i] = i <= lim ? x[i] : -INFINITY;
          mx = fmaxf(mx, x[i]);
        }
      }
      const float mnew = fmaxf(m, mx * c);
      const bool rescale = lt > 0 && mnew > m;
      const float nb = kLog2PScale - mnew;
      uint32_t pk[32];
      float2 acc[4] = {make_float2(0.f, 0.f), make_float2(0.f, 0.f), make_float2(0.f, 0.f), make_float2(0.f, 0.f)};
      exp_tile<kPolyPairs, kTcSum32>(x, make_float2(c, c), make_float2(nb, nb), pk, acc);
      const float f = rescale ? fast_exp2(m - mnew) : 1.0f;
      if constexpr (kTcSum) {
        // one P buffer: the P V and row-sum MMAs of tile t - 1 must have finished before P(t) overwrites it (and
        // before O and the row sum are rescaled)
        if (lt > 0) {
          mbar_wait(pv_done_bar(b, j ^ 1), ((t - 1) >> 1) & 1);
          tc_fence_after();
        }
        if (__any_sync(0xffffffffu, rescale)) {
#pragma unroll
          for (int h = 0; h < 4; ++h) {
            float o[16];
            tmem_ld16(tl + kColO + 16 * h, o);
            tmem_wait_ld();
#pragma unroll
            for (int i = 0; i < 16; ++i) o[i] *= f;
            tmem_st16(tl + kColO + 16 * h, o);
          }
          const float rs = tmem_ld1(tl + kColRS);
          tmem_wait_ld();
          tmem_st1(tl + kColRS, rs * f);
        }
        m = mnew;
        tmem_st16u(tl + kColP1, pk);
        tmem_st16u(tl + kColP1 + 16, pk + 16);
      } else {
        const float lt = ((acc[0].x + acc[2].x) + (acc[1].x + acc[3].x)) + ((acc[0].y + acc[2].y) + (acc[1].y + acc[3].y));
        if (__any_sync(0xffffffffu, rescale)) {
          // the P V MMA of tile t - 1 has finished: O holds the earlier tiles
          mbar_wait(pv_done_bar(b, j ^ 1), ((t - 1) >> 1) & 1);
          tc_fence_after();
#pragma unroll
          for (int h = 0; h < 4; ++h) {
            float o[16];
            tmem_ld16(tl + kColO + 16 * h, o);
            tmem_wait_ld();
#pragma unroll
            for (int i = 0; i < 16; ++i) o[i] *= f;
            tmem_st16(tl + kColO + 16 * h, o);
          }
        }
        l = fmaf(l, f, lt);
        m = mnew;
        if constexpr (kDirectS) {
          // P buffer j is free: the P V MMA of tile t - 2 (its last reader) has completed
          if (t >= 2) {
            mbar_wait(pv_done_bar(b, j), ((t >> 1) - 1) & 1);
            tc_fence_after();
          }
        }
        tmem_st16u(tl + kColP + 32 * j, pk);
        tmem_st16u(tl + kColP + 32 * j + 16, pk + 16);
      }
      tmem_wait_st();
      tc_fence_before();
      named_arrive(pfull_id(b, j), kGroupThreads);  // P(t) (and the O rescale) is in tensor memory
      if (lane == 0) PFA_TRACE(t, 40 + 4 * b + wq);
    }
    // epilogue: out = O * (bmm2_scale / L) after the last P V MMA of this query tile
    const int t_end = g0 + ntiles - 1;
    mbar_wait(pv_done_bar(b, t_end & 1), (t_end >> 1) & 1);
    tc_fence_after();
    float o[64];
    tmem_ld32(tl + kColO, o);
    tmem_ld32(tl + kColO + 32, o + 32);
    if constexpr (kTcSum) l = tmem_ld1(tl + kColRS);  // sum of the e4m3 P values of the row, in the O frame
    tmem_wait_ld();
    const int slot = unit[5];  // < 0: the unit covers all keys of its query tokens and writes the output
    if (slot >= 0) {
      // partial result of a split unit, in its own frame: O, m, l (combined by prefill_attn_merge)
      const int64_t r = static_cast<int64_t>(slot) * (kQTiles * kRows) + kRows * b + row;
      float4* const dst = reinterpret_cast<float4*>(p.ws_o + r * kHeadDim);
#pragma unroll
      for (int j = 0; j < 16; ++j) dst[j] = make_float4(o[4 * j], o[4 * j + 1], o[4 * j + 2], o[4 * j + 3]);
      p.ws_ml[r] = make_float2(m, l);
    } else if (tok < q_len) {
      const float inv = p.out_scale / l;
      __nv_bfloat16* const dst =
          p.out + static_cast<int64_t>(q_begin + tok) * p.out_token_stride + (kvh * kQPerKv + (row & 7)) * kHeadDim;
#pragma unroll
      for (int j = 0; j < 8; ++j) {
        uint4 w;
        __nv_bfloat162 h0 = __floats2bfloat162_rn(o[8 * j] * inv, o[8 * j + 1] * inv);
        __nv_bfloat162 h1 = __floats2bfloat162_rn(o[8 * j + 2] * inv, o[8 * j + 3] * inv);
        __nv_bfloat162 h2 = __floats2bfloat162_rn(o[8 * j + 4] * inv, o[8 * j + 5] * inv);
        __nv_bfloat162 h3 = __floats2bfloat162_rn(o[8 * j + 6] * inv, o[8 * j + 7] * inv);
        w.x = *reinterpret_cast<uint32_t*>(&h0);
        w.y = *reinterpret_cast<uint32_t*>(&h1);
        w.z = *reinterpret_cast<uint32_t*>(&h2);
        w.w = *reinterpret_cast<uint32_t*>(&h3);
        *reinterpret_cast<uint4*>(dst + 8 * j) = w;
      }
    }
    g0 += ntiles;
    }
  }
  tc_fence_before();
  __syncthreads();
  if constexpr (kUsePDL) asm volatile("griddepcontrol.launch_dependents;\n" ::: "memory");
  if (warp == 0) {
    tc_fence_after();
    asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, %1;\n" ::"r"(*tmem_slot), "r"(kTmemCols) : "memory");
  }
}

// Combines the partial results of split units, per row: m* = max m_i, w_i = exp2(m_i - m*) (ex2.approx.ftz.f32),
// out = bf16((sum w_i O_i) * (bmm2_scale / sum w_i l_i)). Block = one split unit (256 rows, one thread per row);
// munits [n, 8]: request, KV head, first query token, first slot, slot count.
__global__ void __launch_bounds__(kQTiles * kRows, 1)
    prefill_attn_merge(const int32_t* __restrict__ munits, const int32_t* __restrict__ cu_q,
                       const float* __restrict__ ws_o, const float2* __restrict__ ws_ml, __nv_bfloat16* __restrict__ out,
                       int64_t out_token_stride, float out_scale) {
  const int32_t* const u = munits + 8 * static_cast<int64_t>(blockIdx.x);
  const int req = u[0], kvh = u[1], tok0 = u[2], slot0 = u[3], nslots = u[4];
  const int r = threadIdx.x;  // row in the unit: query tile r / 128, row r % 128 = 8 * token + head
  const int q_begin = cu_q[req], q_len = cu_q[req + 1] - q_begin;
  const int tok = tok0 + (r / kRows) * kTokPerQTile + ((r % kRows) >> 3);
  if (tok >= q_len) return;
  float mstar = -INFINITY;
  for (int i = 0; i < nslots; ++i) mstar = fmaxf(mstar, ws_ml[static_cast<int64_t>(slot0 + i) * (kQTiles * kRows) + r].x);
  float o[64];
#pragma unroll
  for (int d = 0; d < 64; ++d) o[d] = 0.0f;
  float l = 0.0f;
  for (int i = 0; i < nslots; ++i) {
    const int64_t rr = static_cast<int64_t>(slot0 + i) * (kQTiles * kRows) + r;
    const float2 ml = ws_ml[rr];
    const float w = fast_exp2(ml.x - mstar);
    l = fmaf(w, ml.y, l);
    const float4* const src = reinterpret_cast<const float4*>(ws_o + rr * kHeadDim);
#pragma unroll
    for (int j = 0; j < 16; ++j) {
      const float4 v = src[j];
      o[4 * j] = fmaf(w, v.x, o[4 * j]);
      o[4 * j + 1] = fmaf(w, v.y, o[4 * j + 1]);
      o[4 * j + 2] = fmaf(w, v.z, o[4 * j + 2]);
      o[4 * j + 3] = fmaf(w, v.w, o[4 * j + 3]);
    }
  }
  const float inv = out_scale / l;
  __nv_bfloat16* const dst =
      out + static_cast<int64_t>(q_begin + tok) * out_token_stride + (kvh * kQPerKv + (r & 7)) * kHeadDim;
#pragma unroll
  for (int j = 0; j < 8; ++j) {
    uint4 w;
    __nv_bfloat162 h0 = __floats2bfloat162_rn(o[8 * j] * inv, o[8 * j + 1] * inv);
    __nv_bfloat162 h1 = __floats2bfloat162_rn(o[8 * j + 2] * inv, o[8 * j + 3] * inv);
    __nv_bfloat162 h2 = __floats2bfloat162_rn(o[8 * j + 4] * inv, o[8 * j + 5] * inv);
    __nv_bfloat162 h3 = __floats2bfloat162_rn(o[8 * j + 6] * inv, o[8 * j + 7] * inv);
    w.x = *reinterpret_cast<uint32_t*>(&h0);
    w.y = *reinterpret_cast<uint32_t*>(&h1);
    w.z = *reinterpret_cast<uint32_t*>(&h2);
    w.w = *reinterpret_cast<uint32_t*>(&h3);
    *reinterpret_cast<uint4*>(dst + 8 * j) = w;
  }
}

}  // namespace pfa

#if !defined(__CUDACC_RTC__)
// ---------------------------------------------------------------------------------------------- host side
namespace pfa_host {

using EncodeFn = CUresult (*)(CUtensorMap*, CUtensorMapDataType, cuuint32_t, void*, const cuuint64_t*,
                              const cuuint64_t*, const cuuint32_t*, const cuuint32_t*, CUtensorMapInterleave,
                              CUtensorMapSwizzle, CUtensorMapL2promotion, CUtensorMapFloatOOBfill);

inline EncodeFn encode_fn() {
  static EncodeFn fn = [] {
    void* ptr = nullptr;
    cudaDriverEntryPointQueryResult q;
    sglang::host::RuntimeDeviceCheck(
        ::cudaGetDriverEntryPointByVersion("cuTensorMapEncodeTiled", &ptr, 12000, cudaEnableDefault, &q));
    sglang::host::RuntimeCheck(q == cudaDriverEntryPointSuccess && ptr != nullptr, "cuTensorMapEncodeTiled not found");
    return reinterpret_cast<EncodeFn>(ptr);
  }();
  return fn;
}

// rank-n byte tensor map (dim 0 contiguous), 64-byte swizzle
inline CUtensorMap byte_map(const void* base, int rank, const cuuint64_t* dims, const cuuint64_t* strides,
                            const cuuint32_t* box) {
  CUtensorMap map;
  const cuuint32_t elem[4] = {1, 1, 1, 1};
  const CUresult r = encode_fn()(&map, CU_TENSOR_MAP_DATA_TYPE_UINT8, rank, const_cast<void*>(base), dims, strides, box,
                                 elem, CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_64B,
                                 CU_TENSOR_MAP_L2_PROMOTION_L2_256B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
  sglang::host::RuntimeCheck(r == CUDA_SUCCESS, "cuTensorMapEncodeTiled failed: ", static_cast<int>(r));
  return map;
}

}  // namespace pfa_host

template <bool kUsePDL, int kPolyPairs>
struct PrefillAttnKernel {
  static constexpr int kThreads = pfa::kThreads;
  static_assert((kPolyPairs >= 0 && kPolyPairs <= 8) || kPolyPairs == pfa::kHalfExp || kPolyPairs == pfa::kHalfExpTc,
                "polynomial pairs per 8 in [0, 8], or a half exp mode");
  static void run(const tvm::ffi::TensorView units,       // [n, 8] int32 work units (see pfa::Params)
                  const tvm::ffi::TensorView cta_units,   // [blocks + 1] int32 unit offsets per block, or empty
                  const tvm::ffi::TensorView munits,      // [m, 8] int32 split units to merge
                  const tvm::ffi::TensorView ws_o,        // [>= slots * 256 * 64] fp32
                  const tvm::ffi::TensorView ws_ml,       // [>= slots * 256 * 2] fp32
                  const tvm::ffi::TensorView q,           // [tokens, Hq, 64] e4m3, heads/dims contiguous
                  const tvm::ffi::TensorView k_cache,     // [pages, Hkv, 128, 64] e4m3, dims contiguous
                  const tvm::ffi::TensorView v_cache,     // [pages, Hkv, 128, 64] e4m3, dims contiguous
                  const tvm::ffi::TensorView page_table,  // [R, pages] int32, inner contiguous
                  const tvm::ffi::TensorView seq_lens,    // [R] int32
                  const tvm::ffi::TensorView cu_q,        // [R + 1] int32
                  const tvm::ffi::TensorView out,         // [tokens, Hq, 64] bf16, heads/dims contiguous
                  int64_t max_q_len, double bmm1_scale, double bmm2_scale, int64_t trace_ptr) {
    using namespace sglang::host;
    RuntimeCheck(q.dim() == 3 && q.size(2) == pfa::kHeadDim && q.dtype().code == DLDataTypeCode::kDLFloat8_e4m3fn &&
                     q.stride(2) == 1 && q.stride(1) == pfa::kHeadDim && q.stride(0) % 16 == 0,
                 "q must be e4m3 [tokens, heads, 64], heads and dims contiguous");
    const int64_t tokens = q.size(0), hq = q.size(1);
    for (const auto* c : {&k_cache, &v_cache}) {
      RuntimeCheck(c->dim() == 4 && c->size(2) == pfa::kTile && c->size(3) == pfa::kHeadDim && c->stride(3) == 1 &&
                       c->dtype().code == DLDataTypeCode::kDLFloat8_e4m3fn && c->stride(0) % 16 == 0 &&
                       c->stride(1) % 16 == 0 && c->stride(2) % 16 == 0,
                   "caches must be e4m3 [pages, Hkv, 128, 64], dims contiguous, 16-byte aligned strides");
    }
    const int64_t hkv = k_cache.size(1);
    RuntimeCheck(v_cache.size(1) == hkv && v_cache.size(0) == k_cache.size(0) && hkv * pfa::kQPerKv == hq,
                 "head layout");
    RuntimeCheck(out.dtype().code == DLDataTypeCode::kDLBfloat && out.dtype().bits == 16 && out.dim() == 3 &&
                     out.size(0) == tokens && out.size(1) == hq && out.size(2) == pfa::kHeadDim && out.stride(2) == 1 &&
                     out.stride(1) == pfa::kHeadDim && out.stride(0) % 8 == 0,
                 "out must be bf16 [tokens, Hq, 64], heads and dims contiguous");
    const int64_t reqs = seq_lens.size(0);
    RuntimeCheck(seq_lens.dim() == 1 && seq_lens.stride(0) == 1 && seq_lens.dtype().code == DLDataTypeCode::kDLInt &&
                     seq_lens.dtype().bits == 32,
                 "seq_lens [R] int32");
    RuntimeCheck(cu_q.dim() == 1 && cu_q.size(0) == reqs + 1 && cu_q.stride(0) == 1 &&
                     cu_q.dtype().code == DLDataTypeCode::kDLInt && cu_q.dtype().bits == 32,
                 "cu_q [R + 1] int32");
    RuntimeCheck(page_table.dim() == 2 && page_table.size(0) >= reqs && page_table.stride(1) == 1 &&
                     page_table.dtype().code == DLDataTypeCode::kDLInt && page_table.dtype().bits == 32,
                 "page_table [R, pages] int32");
    RuntimeCheck(bmm1_scale > 0.0, "bmm1_scale must be positive");
    const auto aligned16 = [](const void* ptr) { return reinterpret_cast<uintptr_t>(ptr) % 16 == 0; };
    RuntimeCheck(aligned16(q.data_ptr()) && aligned16(k_cache.data_ptr()) && aligned16(v_cache.data_ptr()) &&
                     aligned16(out.data_ptr()),
                 "16-byte alignment");

    RuntimeCheck(units.dim() == 2 && units.size(1) == 8 && units.stride(0) == 8 && units.stride(1) == 1 &&
                     units.dtype().code == DLDataTypeCode::kDLInt && units.dtype().bits == 32,
                 "units [n, 8] int32 contiguous");
    RuntimeCheck(cta_units.dim() == 1 && (cta_units.size(0) == 0 || cta_units.stride(0) == 1) &&
                     cta_units.dtype().code == DLDataTypeCode::kDLInt && cta_units.dtype().bits == 32,
                 "cta_units [blocks + 1] int32 contiguous (or empty)");
    RuntimeCheck(cta_units.size(0) == 0 || cta_units.size(0) >= 2, "cta_units needs at least one block");
    RuntimeCheck(munits.dim() == 2 && munits.size(1) == 8 && (munits.size(0) == 0 || munits.stride(0) == 8) &&
                     munits.dtype().code == DLDataTypeCode::kDLInt && munits.dtype().bits == 32,
                 "munits [m, 8] int32 contiguous");
    RuntimeCheck(ws_o.dtype().code == DLDataTypeCode::kDLFloat && ws_o.dtype().bits == 32 &&
                     ws_ml.dtype().code == DLDataTypeCode::kDLFloat && ws_ml.dtype().bits == 32,
                 "fp32 workspaces");
    pfa::Params params{};
    params.units = static_cast<const int32_t*>(units.data_ptr());
    params.cta_units = cta_units.size(0) > 0 ? static_cast<const int32_t*>(cta_units.data_ptr()) : nullptr;
    params.ws_o = static_cast<float*>(ws_o.data_ptr());
    params.ws_ml = static_cast<float2*>(ws_ml.data_ptr());
    params.q = static_cast<const uint8_t*>(q.data_ptr());
    params.q_token_stride = q.stride(0);
    params.page_table = static_cast<const int32_t*>(page_table.data_ptr());
    params.seq_lens = static_cast<const int32_t*>(seq_lens.data_ptr());
    params.cu_q = static_cast<const int32_t*>(cu_q.data_ptr());
    params.out = static_cast<__nv_bfloat16*>(out.data_ptr());
    params.page_table_stride = page_table.stride(0);
    params.out_token_stride = out.stride(0);
    params.scale_log2 = static_cast<float>(bmm1_scale * M_LOG2E);
    params.out_scale = static_cast<float>(bmm2_scale);
    params.trace = reinterpret_cast<long long*>(trace_ptr);

    // caches: dims {64 B, 128 tokens, Hkv, pages}, box {64, 128, 1, 1}
    const auto cache_map = [&](const tvm::ffi::TensorView& c) {
      const cuuint64_t d[4] = {64, pfa::kTile, static_cast<cuuint64_t>(hkv), static_cast<cuuint64_t>(c.size(0))};
      const cuuint64_t s[3] = {static_cast<cuuint64_t>(c.stride(2)), static_cast<cuuint64_t>(c.stride(1)),
                               static_cast<cuuint64_t>(c.stride(0))};
      const cuuint32_t bx[4] = {64, pfa::kTile, 1, 1};
      return pfa_host::byte_map(c.data_ptr(), 4, d, s, bx);
    };
    const CUtensorMap k_map = cache_map(k_cache);
    const CUtensorMap v_map = cache_map(v_cache);

    constexpr auto kernel = pfa::prefill_attn_kernel<kUsePDL, false, kPolyPairs>;
    constexpr auto kernel_trace = pfa::prefill_attn_kernel<kUsePDL, true, kPolyPairs>;
    int device = 0;
    RuntimeDeviceCheck(::cudaGetDevice(&device));
    static bool configured[64] = {};
    if (device < 0 || device >= 64 || !configured[device]) {
      RuntimeDeviceCheck(::cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, pfa::kSmBytes));
      RuntimeDeviceCheck(
          ::cudaFuncSetAttribute(kernel_trace, cudaFuncAttributeMaxDynamicSharedMemorySize, pfa::kSmBytes));
      if (device >= 0 && device < 64) configured[device] = true;
    }
    const int64_t nunits = units.size(0);
    if (reqs == 0 || max_q_len <= 0 || nunits == 0) return;
    const int64_t nblocks = cta_units.size(0) > 0 ? cta_units.size(0) - 1 : nunits;
    const dim3 grid(static_cast<unsigned>(nblocks));
    if (params.trace != nullptr) {
      LaunchKernel(grid, kThreads, q.device(), pfa::kSmBytes)
          .config({.use_pdl = kUsePDL})(kernel_trace, params, k_map, v_map);
    } else {
      LaunchKernel(grid, kThreads, q.device(), pfa::kSmBytes)
          .config({.use_pdl = kUsePDL})(kernel, params, k_map, v_map);
    }
    if (munits.size(0) > 0) {
      LaunchKernel(dim3(static_cast<unsigned>(munits.size(0))), pfa::kQTiles * pfa::kRows, q.device())(
          pfa::prefill_attn_merge, static_cast<const int32_t*>(munits.data_ptr()), params.cu_q, params.ws_o,
          params.ws_ml, params.out, params.out_token_stride, params.out_scale);
    }
  }
};

#endif  // !__CUDACC_RTC__

}  // namespace muse_vr
