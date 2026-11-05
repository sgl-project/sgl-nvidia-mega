// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
/**
 * \brief Paged FP8 (e4m3) full-attention verify for many requests per step: a persistent kernel whose thread blocks
 * split the summed key range of all requests evenly (one block per SM, one wave), read whole token rows of the K/V
 * pools (every KV head of the rank), and run the attention on tensor-core MMA (tcgen05); a second kernel merges the
 * pieces of requests that several blocks share.
 *
 * Shapes (per call): q [R * 8, Hq, 64] e4m3 (heads and dims contiguous, any token stride that is a multiple of 16 B);
 * K/V pools [slots, Hkv * 64] e4m3 (token-major, slot = page * 128 + offset); page_table [R, >= pages] int32; seq_lens
 * [R] int32 (KV length including the 8 verify tokens); out [R * 8, Hq, 64] bf16, contiguous. Hq = 8 Hkv, Hkv = 2 kPairs
 * (kPairs = 1 or 2: 2 or 4 KV heads per rank). Token t (0..7) of request r sits at position p = seq_len - 8 + t and
 * attends keys [0, p].
 *
 * Work split. A request has n_r = ceil(seq_len / kTileKeys) tiles of kTileKeys keys (kTileKeys = 128 / kPairs, so a K
 * or V tile is always 16 KB: kTileKeys token rows of Hkv * 64 bytes). With W = sum n_r and G = min(blocks, W), block b
 * takes the tiles [floor(b W / G), floor((b + 1) W / G)) of the concatenation of all requests (request-major). The part
 * of a request inside one block is a piece; a request that lies wholly inside one block is normalized and written by
 * it; otherwise each piece writes a partial record (unnormalized output, row maximum, row sum per query row) and
 * full_attn_wide_merge_kernel (launched next, PDL-chained) merges the request's records in block order. Norm + gate
 * variant (kGate, merge kernel only; the main kernel is the same): the merge kernel writes every request (a request
 * inside one block: the main kernel's bf16 output row read back) and applies the output norm (no weight) and the sigmoid
 * gate with SGLang's norm_gate arithmetic in the configuration that kernel would pick (kNgVec 8 or 16 bytes per thread;
 * see norm_gate4).
 *
 * Arithmetic per query row: s = fp32 dot product of e4m3 q and k (MMA, fp32 accumulation); per tile of kTileKeys keys,
 * in key order: mt = (max of s over the tile) * scale_log2; m' = max(m, mt); ar = exp2(m - m'); at = exp2(mt - m');
 * p448 = exp2(fma(s, scale_log2, log2(448) - mt)) (= 448 exp2(s scale_log2 - mt), the probability scaled for e4m3);
 * pq = e4m3(p448) (round to nearest even, satfinite); o = fma(o, ar, (sum of pq * v over the tile) * at) (MMA into
 * fp32, then fp32 arithmetic). Row sums (in x448 units): each of the row's two threads (half of every tile's keys)
 * keeps l_h = l_h * ar + (sum of its p448 over the tile) * at, its keys i in four interleaved partial sums (i % 4)
 * combined (0 + 2) + (1 + 3); l = l_0 + l_1 at the end of the piece. Merge of a request's pieces j = 0..k-1 (block
 * order): M = max m_j, O = sum exp2(m_j - M) o_j, L = sum exp2(m_j - M) l_j in order (pieces with m_j = -inf skipped);
 * out = bf16(O * (bmm2_scale / L)). A request inside one block: out = bf16(o * (bmm2_scale / l)). exp2 is
 * ex2.approx.ftz.f32 (or, with kPoly > 0, a degree-5 polynomial for kPoly of every 8 keys).
 *
 * Block roles: kSmWarps softmax warps (2 threads per query row: warp w -> half = w / (kSmWarps / 2) of every tile's
 * keys and of the output dims; the two threads of a row exchange tile maxima through shared memory and a named
 * barrier), one load warp (lane 0: per piece the q rows of the request with the tensor memory accelerator (one box of
 * 8 tokens x 8 heads x 64 B per KV head, 64-byte swizzle) into a Q buffer; per tile the K and V boxes (kTileKeys rows
 * x 128 B per KV-head pair, 128-byte swizzle; the pairs of a tile together cover whole token rows) into a ring of
 * kStages stages), one QK warp and one PV warp (the whole warp runs the loop with warp-uniform operands, one elected
 * lane issues: kind::f8f6f4, M = 64 rows per KV head, the two heads of a pair at tensor-memory lane offsets 0 and 16 of
 * every 32-lane group). Hand-offs per tile g: K/V landed -> QK(g) into score buffer g % 2 (issued two tiles ahead,
 * once the softmax warps hold tile g - 2's scores in registers) -> softmax -> probabilities in shared memory -> PV(g)
 * into output buffer g % 2 -> added into the registers of the softmax threads during tile g + 1.
 *
 * Ordering: q, seq_lens and the page table may be written by the preceding kernels; they are read only after
 * griddepcontrol.wait (use_pdl), followed by fence.proxy.async.global before the first tensor-memory-accelerator read.
 * Pool rows past the end of a request inside its last page are loaded and masked out of the probabilities (their
 * values are finite: the pools are allocated zero-filled and only finite values are stored into them). Every mbarrier
 * wait is bounded (trap after a fixed number of polls). Development probe modes (template argument kMode != 0) exist
 * for profiling only. Skeleton options (template argument kSkel, default 0) change instruction forms and hand-offs
 * only, never an output bit (bits listed in the kernel; units: tests/test_full_attn_wide_skel.py).
 */

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

namespace muse_vr {

namespace faw {

constexpr int kHeadDim = 64;
constexpr int kQPerKv = 8;
constexpr int kTokens = 8;
constexpr int kRows = 64;  // query rows per KV head: token * 8 + head
constexpr int kRec = 68;  // floats per partial row record: 64 outputs, m, l, 2 pad
constexpr float kPScale = 448.0f;
constexpr float kLog2PScale = 8.807354922057604f;  // log2(448)
constexpr int kMaxReqs = 256;
constexpr int kMaxPages = 1024;  // page numbers of a block's range held in shared memory (more: read from global)
constexpr int kTileBytes = 16384;  // one K or V tile
constexpr uint32_t kTmemCols = 512;  // S0 [0,128) S1 [128,256) O0 [256,384) O1 [384,512)

template <int kPairs, int kStages, int kTprT = 2>
struct Layout {
  static constexpr int kHeads = 2 * kPairs;
  static constexpr int kTileKeys = 128 / kPairs;
  static constexpr int kTpr = kTprT;  // softmax threads per query row (2: each half of a tile's keys and of the dims)
  static_assert(kTpr == 2 || (kTpr == 1 && kPairs == 2), "one thread per row only with 64-key tiles");
  static constexpr int kSmWarps = 4 * kPairs * kTpr;  // softmax warps: kTpr threads x (kHeads x 64 rows)
  static constexpr int kThreads = 32 * (kSmWarps + 3);  // + load warp + QK-MMA warp + PV-MMA warp
  static constexpr int kKpt = kTileKeys / kTpr;  // keys of a tile per softmax thread (32 or 64)
  static constexpr int kPRowBytes = kTileKeys;  // e4m3 probabilities of one row of a tile
  static constexpr int kSmK = 0;
  static constexpr int kSmV = kSmK + kStages * kTileBytes;
  static constexpr int kSmP = kSmV + kStages * kTileBytes;           // [2 buffers][heads][64 rows][kPRowBytes]
  static constexpr int kPBuf = kHeads * kRows * kPRowBytes;          // 16 KB
  static constexpr int kSmQ = kSmP + 2 * kPBuf;                      // [2 buffers][heads][64 rows][64 B]
  static constexpr int kQBuf = kHeads * kRows * 64;
  static constexpr int kQBufs = kStages >= 5 ? 1 : 2;  // Q buffers (one when the ring takes the shared memory)
  static constexpr int kSmRed = kSmQ + kQBufs * kQBuf;               // [3][heads * 64 rows][2] fp32
  static constexpr int kSmPre = kSmRed + 3 * kHeads * kRows * 2 * 4;  // prefix tile counts [kMaxReqs + 1]
  static constexpr int kSmPages = kSmPre + (kMaxReqs + 1) * 4 + 12;  // page numbers of the block's range
  static constexpr int kSmBar = (kSmPages + kMaxPages * 4 + 15) / 16 * 16;
  static constexpr int kSmBytes = kSmBar + 256 + 1024;
};

struct Params {
  const int32_t* __restrict__ page_table;
  const int32_t* __restrict__ seq_lens;
  float* __restrict__ partial;      // [G][2 slots][heads * 64 rows][kRec]
  uint32_t* __restrict__ counters;  // [kMaxReqs], zero between calls
  __nv_bfloat16* __restrict__ out;
  int64_t page_table_stride;
  int32_t num_reqs;
  int32_t num_q_heads;
  float scale_log2;
  float out_scale;  // bmm2_scale / 448
  // output norm + gate (kGate variants only): gate [R * 8, >= Hq * 64] bf16, row stride gate_token_stride
  const __nv_bfloat16* __restrict__ gate;
  int64_t gate_token_stride;
  float norm_eps;
};

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
__device__ __forceinline__ void mbar_arrive(uint32_t bar) {
  asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];\n" ::"r"(bar) : "memory");
}
// Upper bound on try_wait calls of one wait: every arrival comes from this block (its threads, MMAs and copies); the
// bound turns a wait that can only hang through a defect into a trap.
inline constexpr uint32_t kMaxPolls = 1u << 24;
__device__ __noinline__ void wait_bound_exceeded() { __trap(); }
// The poll loop is written in PTX (one asm statement) so the compiler sees convergent control flow around it: the
// role branches stay uniform, MMA descriptors stay in uniform registers and elect.sync needs no emulation. Bounded:
// after kMaxPolls failed polls the thread traps.
__device__ __forceinline__ void mbar_wait(uint32_t bar, uint32_t parity) {
  asm volatile(
      "{\n"
      ".reg .pred p;\n"
      ".reg .u32 n;\n"
      "mov.u32 n, 0;\n"
      "FAW_WAIT_%=:\n"
      "mbarrier.try_wait.parity.shared::cta.b64 p, [%0], %1;\n"
      "@p bra FAW_DONE_%=;\n"
      "add.u32 n, n, 1;\n"
      "setp.lt.u32 p, n, %2;\n"
      "@p bra FAW_WAIT_%=;\n"
      "trap;\n"
      "FAW_DONE_%=:\n"
      "}\n" ::"r"(bar),
      "r"(parity), "n"(kMaxPolls)
      : "memory");
}
// Wait with a suspend-time hint (ns): the thread may be suspended for up to the hint while the phase is incomplete and
// resumes when it completes (PTX ISA mbarrier.try_wait); used only by skeleton option bit 4 (the load warp's ring-refill
// wait). Bounded like mbar_wait.
template <uint32_t kHintNs>
__device__ __forceinline__ void mbar_wait_hint(uint32_t bar, uint32_t parity) {
  if constexpr (kHintNs == 0) {
    mbar_wait(bar, parity);
  } else {
    asm volatile(
        "{\n"
        ".reg .pred p;\n"
        ".reg .u32 n;\n"
        "mov.u32 n, 0;\n"
        "FAW_HWAIT_%=:\n"
        "mbarrier.try_wait.parity.shared::cta.b64 p, [%0], %1, %3;\n"
        "@p bra FAW_HDONE_%=;\n"
        "add.u32 n, n, 1;\n"
        "setp.lt.u32 p, n, %2;\n"
        "@p bra FAW_HWAIT_%=;\n"
        "trap;\n"
        "FAW_HDONE_%=:\n"
        "}\n" ::"r"(bar),
        "r"(parity), "n"((1u << 30) / (kHintNs < 64 ? 64 : kHintNs) + 1024), "n"(kHintNs)
        : "memory");
  }
}
// Pure polling (mbarrier.test_wait, never suspends); development variant of mbar_wait for barriers completed by thread
// arrivals. Bounded like mbar_wait.
__device__ __forceinline__ void mbar_wait_spin(uint32_t bar, uint32_t parity) {
  asm volatile(
      "{\n"
      ".reg .pred p;\n"
      ".reg .u32 n;\n"
      "mov.u32 n, 0;\n"
      "FAW_SPIN_%=:\n"
      "mbarrier.test_wait.parity.shared::cta.b64 p, [%0], %1;\n"
      "@p bra FAW_SDONE_%=;\n"
      "add.u32 n, n, 1;\n"
      "setp.lt.u32 p, n, %2;\n"
      "@p bra FAW_SPIN_%=;\n"
      "trap;\n"
      "FAW_SDONE_%=:\n"
      "}\n" ::"r"(bar),
      "r"(parity), "n"(kMaxPolls * 4)
      : "memory");
}
__device__ __forceinline__ void tma_load_2d(uint32_t dst, const CUtensorMap* map, int32_t c0, int32_t c1, uint32_t bar) {
  asm volatile(
      "cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1, {%2, %3}], [%4];\n" ::"r"(dst),
      "l"(reinterpret_cast<uint64_t>(map)), "r"(c0), "r"(c1), "r"(bar)
      : "memory");
}
__device__ __forceinline__ void tma_load_3d(uint32_t dst, const CUtensorMap* map, int32_t c0, int32_t c1, int32_t c2,
                                            uint32_t bar) {
  asm volatile(
      "cp.async.bulk.tensor.3d.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1, {%2, %3, %4}], [%5];\n" ::"r"(
          dst),
      "l"(reinterpret_cast<uint64_t>(map)), "r"(c0), "r"(c1), "r"(c2), "r"(bar)
      : "memory");
}
__device__ __forceinline__ void tma_prefetch_2d(const CUtensorMap* map, int32_t c0, int32_t c1) {
  asm volatile("cp.async.bulk.prefetch.tensor.2d.L2.global [%0, {%1, %2}];\n" ::"l"(reinterpret_cast<uint64_t>(map)),
               "r"(c0), "r"(c1)
               : "memory");
}
__device__ __forceinline__ void tc_fence_before() { asm volatile("tcgen05.fence::before_thread_sync;\n" ::: "memory"); }
__device__ __forceinline__ void tc_fence_after() { asm volatile("tcgen05.fence::after_thread_sync;\n" ::: "memory"); }
__device__ __forceinline__ void fence_proxy_async() { asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory"); }
__device__ __forceinline__ void mma_f8(uint32_t d, uint64_t a, uint64_t b, uint32_t idesc, uint32_t accumulate) {
  asm volatile(
      "{\n"
      ".reg .pred p;\n"
      "setp.ne.b32 p, %4, 0;\n"
      "tcgen05.mma.cta_group::1.kind::f8f6f4 [%0], %1, %2, %3, p;\n"
      "}\n" ::"r"(d),
      "l"(a), "l"(b), "r"(idesc), "r"(accumulate)
      : "memory");
}
// Warp-uniform variants: the whole warp executes them with uniform operands, one elected lane issues.
__device__ __forceinline__ void mma_f8_e(uint32_t d, uint64_t a, uint64_t b, uint32_t idesc, uint32_t accumulate) {
  asm volatile(
      "{\n"
      ".reg .pred e, p;\n"
      "elect.sync _|e, 0xffffffff;\n"
      "setp.ne.b32 p, %4, 0;\n"
      "@e tcgen05.mma.cta_group::1.kind::f8f6f4 [%0], %1, %2, %3, p;\n"
      "}\n" ::"r"(d),
      "l"(a), "l"(b), "r"(idesc), "r"(accumulate)
      : "memory");
}
__device__ __forceinline__ void mma_commit_e(uint32_t bar) {
  asm volatile(
      "{\n"
      ".reg .pred e;\n"
      "elect.sync _|e, 0xffffffff;\n"
      "@e tcgen05.commit.cta_group::1.mbarrier::arrive::one.shared::cluster.b64 [%0];\n"
      "}\n" ::"r"(bar)
      : "memory");
}
__device__ __forceinline__ void mma_commit(uint32_t bar) {
  asm volatile("tcgen05.commit.cta_group::1.mbarrier::arrive::one.shared::cluster.b64 [%0];\n" ::"r"(bar) : "memory");
}
// 32 consecutive fp32 columns of this thread's tensor-memory lane
__device__ __forceinline__ void tmem_ld32(uint32_t taddr, float (&v)[32]) {
  uint32_t r[32];
  asm volatile(
      "tcgen05.ld.sync.aligned.32x32b.x32.b32 {%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17,%18,%19,"
      "%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31}, [%32];\n"
      : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]), "=r"(r[4]), "=r"(r[5]), "=r"(r[6]), "=r"(r[7]), "=r"(r[8]),
        "=r"(r[9]), "=r"(r[10]), "=r"(r[11]), "=r"(r[12]), "=r"(r[13]), "=r"(r[14]), "=r"(r[15]), "=r"(r[16]),
        "=r"(r[17]), "=r"(r[18]), "=r"(r[19]), "=r"(r[20]), "=r"(r[21]), "=r"(r[22]), "=r"(r[23]), "=r"(r[24]),
        "=r"(r[25]), "=r"(r[26]), "=r"(r[27]), "=r"(r[28]), "=r"(r[29]), "=r"(r[30]), "=r"(r[31])
      : "r"(taddr));
#pragma unroll
  for (int i = 0; i < 32; ++i) v[i] = __uint_as_float(r[i]);
}
__device__ __forceinline__ void tmem_wait_ld() { asm volatile("tcgen05.wait::ld.sync.aligned;\n" ::: "memory"); }
// 32 consecutive fp32 columns of this thread's tensor-memory lane and their maximum (tcgen05.ld with .red .max, PTX ISA
// 8.8: the reduction of the loaded columns per lane, .NaN not set); the registers and the maximum are read only after
// tcgen05.wait::ld, as for the plain load (skeleton option bit 0)
__device__ __forceinline__ float tmem_ld32_max(uint32_t taddr, float (&v)[32]) {
  uint32_t r[32];
  float mx;
  asm volatile(
      "tcgen05.ld.red.sync.aligned.32x32b.x32.f32.max {%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17,%18,"
      "%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31}, %32, [%33];\n"
      : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]), "=r"(r[4]), "=r"(r[5]), "=r"(r[6]), "=r"(r[7]), "=r"(r[8]),
        "=r"(r[9]), "=r"(r[10]), "=r"(r[11]), "=r"(r[12]), "=r"(r[13]), "=r"(r[14]), "=r"(r[15]), "=r"(r[16]),
        "=r"(r[17]), "=r"(r[18]), "=r"(r[19]), "=r"(r[20]), "=r"(r[21]), "=r"(r[22]), "=r"(r[23]), "=r"(r[24]),
        "=r"(r[25]), "=r"(r[26]), "=r"(r[27]), "=r"(r[28]), "=r"(r[29]), "=r"(r[30]), "=r"(r[31]), "=f"(mx)
      : "r"(taddr));
#pragma unroll
  for (int i = 0; i < 32; ++i) v[i] = __uint_as_float(r[i]);
  return mx;
}
// 16 consecutive 32-bit columns of this thread's tensor-memory lane (skeleton option bit 5)
__device__ __forceinline__ void tmem_st16(uint32_t taddr, const uint32_t (&w)[16]) {
  asm volatile(
      "tcgen05.st.sync.aligned.32x32b.x16.b32 [%0], {%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16};\n" ::"r"(taddr),
      "r"(w[0]), "r"(w[1]), "r"(w[2]), "r"(w[3]), "r"(w[4]), "r"(w[5]), "r"(w[6]), "r"(w[7]), "r"(w[8]), "r"(w[9]), "r"(w[10]),
      "r"(w[11]), "r"(w[12]), "r"(w[13]), "r"(w[14]), "r"(w[15])
      : "memory");
}
__device__ __forceinline__ void tmem_wait_st() { asm volatile("tcgen05.wait::st.sync.aligned;\n" ::: "memory"); }
// MMA with A in tensor memory (warp-uniform operands, one elected lane issues; skeleton option bit 5)
__device__ __forceinline__ void mma_f8_ts_e(uint32_t d, uint32_t a, uint64_t b, uint32_t idesc, uint32_t accumulate) {
  asm volatile(
      "{\n"
      ".reg .pred e, p;\n"
      "elect.sync _|e, 0xffffffff;\n"
      "setp.ne.b32 p, %4, 0;\n"
      "@e tcgen05.mma.cta_group::1.kind::f8f6f4 [%0], [%1], %2, %3, p;\n"
      "}\n" ::"r"(d),
      "r"(a), "l"(b), "r"(idesc), "r"(accumulate)
      : "memory");
}

// Shared-memory matrix descriptor: start >> 4 [0,14), leading byte offset >> 4 [16,30), stride byte offset >> 4
// [32,46), descriptor version 1 [46,48), swizzle mode [61,64) (PTX ISA, tcgen05 shared memory descriptor).
__device__ __forceinline__ uint64_t sdesc(uint32_t saddr, uint32_t sbo, uint32_t layout) {
  return static_cast<uint64_t>((saddr >> 4) & 0x3FFF) | (static_cast<uint64_t>(1) << 16) |
         (static_cast<uint64_t>((sbo >> 4) & 0x3FFF) << 32) | (static_cast<uint64_t>(1) << 46) |
         (static_cast<uint64_t>(layout & 7) << 61);
}
constexpr uint32_t kSw128 = 2, kSw64 = 4;
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
// 2^y for a pair (y <= 0 or -inf) on the FMA pipes: y clamped to -126, y = n + f (n = round(y), |f| <= 1/2),
// 2^f by a degree-5 polynomial (Horner, fp32; max relative error 2.3e-7 over [-1/2, 1/2], the order of
// ex2.approx.ftz.f32's), then n added to the exponent field. Results below 2^-126 are not flushed to zero (they are
// at most 2^-126 and vanish in the e4m3 probabilities); ex2.approx.ftz.f32 returns 0 there.
__device__ __forceinline__ float2 fma2(float2 a, float2 b, float2 c);
__device__ __forceinline__ float2 add2(float2 a, float2 b);
__device__ __forceinline__ float2 exp2_poly2(float2 y) {
  constexpr float kMagic = 12582912.0f;  // 1.5 * 2^23: adding it rounds to an integer held in the low mantissa bits
  y = make_float2(fmaxf(y.x, -126.0f), fmaxf(y.y, -126.0f));
  const float2 t = add2(y, make_float2(kMagic, kMagic));
  const float2 n = add2(t, make_float2(-kMagic, -kMagic));
  const float2 f = fma2(n, make_float2(-1.0f, -1.0f), y);
  float2 q = fma2(f, make_float2(1.327764592e-03f, 1.327764592e-03f), make_float2(9.675552137e-03f, 9.675552137e-03f));
  q = fma2(q, f, make_float2(5.550709739e-02f, 5.550709739e-02f));
  q = fma2(q, f, make_float2(2.402211875e-01f, 2.402211875e-01f));
  q = fma2(q, f, make_float2(6.931469440e-01f, 6.931469440e-01f));
  q = fma2(q, f, make_float2(1.000000119e+00f, 1.000000119e+00f));
  return make_float2(__int_as_float(__float_as_int(q.x) + (__float_as_int(t.x) << 23)),
                     __int_as_float(__float_as_int(q.y) + (__float_as_int(t.y) << 23)));
}
__device__ __forceinline__ float2 fma2(float2 a, float2 b, float2 c) {
  uint64_t d;
  asm("fma.rn.f32x2 %0, %1, %2, %3;\n"
      : "=l"(d)
      : "l"(*reinterpret_cast<uint64_t*>(&a)), "l"(*reinterpret_cast<uint64_t*>(&b)), "l"(*reinterpret_cast<uint64_t*>(&c)));
  return *reinterpret_cast<float2*>(&d);
}
__device__ __forceinline__ float2 add2(float2 a, float2 b) {
  uint64_t d;
  asm("add.rn.f32x2 %0, %1, %2;\n"
      : "=l"(d)
      : "l"(*reinterpret_cast<uint64_t*>(&a)), "l"(*reinterpret_cast<uint64_t*>(&b)));
  return *reinterpret_cast<float2*>(&d);
}
__device__ __forceinline__ float2 mul2(float2 a, float2 b) {
  uint64_t d;
  asm("mul.rn.f32x2 %0, %1, %2;\n"
      : "=l"(d)
      : "l"(*reinterpret_cast<uint64_t*>(&a)), "l"(*reinterpret_cast<uint64_t*>(&b)));
  return *reinterpret_cast<float2*>(&d);
}
__device__ __forceinline__ float max3(float a, float b, float c) {
  float d;
  asm("max.f32 %0, %1, %2, %3;\n" : "=f"(d) : "f"(a), "f"(b), "f"(c));
  return d;
}
__device__ __forceinline__ uint32_t e4m3x4(float a, float b, float c, float d) {
  uint16_t lo, hi;
  asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;\n" : "=h"(lo) : "f"(b), "f"(a));
  asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;\n" : "=h"(hi) : "f"(d), "f"(c));
  return static_cast<uint32_t>(lo) | (static_cast<uint32_t>(hi) << 16);
}
// The same four conversions and byte positions, the two halves written into one 32-bit register inside one statement
// (skeleton option bit 1; same bytes as e4m3x4)
__device__ __forceinline__ uint32_t e4m3x4_joined(float a, float b, float c, float d) {
  uint32_t w;
  asm("{\n"
      ".reg .b16 lo, hi;\n"
      "cvt.rn.satfinite.e4m3x2.f32 lo, %2, %1;\n"
      "cvt.rn.satfinite.e4m3x2.f32 hi, %4, %3;\n"
      "mov.b32 %0, {lo, hi};\n"
      "}\n"
      : "=r"(w)
      : "f"(a), "f"(b), "f"(c), "f"(d));
  return w;
}
__device__ __forceinline__ void fence_gpu() { asm volatile("fence.acq_rel.gpu;\n" ::: "memory"); }
__device__ __forceinline__ uint32_t atom_add_acq_rel_gpu(uint32_t* p, uint32_t v) {
  uint32_t old;
  asm volatile("atom.acq_rel.gpu.global.add.u32 %0, [%1], %2;\n" : "=r"(old) : "l"(p), "r"(v) : "memory");
  return old;
}
__device__ __forceinline__ void st_relaxed_gpu(uint32_t* p, uint32_t v) {
  asm volatile("st.relaxed.gpu.global.u32 [%0], %1;\n" ::"l"(p), "r"(v) : "memory");
}
__device__ __forceinline__ float ld_relaxed_gpu_f32(const float* p) {
  float v;
  asm volatile("ld.relaxed.gpu.global.f32 %0, [%1];\n" : "=f"(v) : "l"(p) : "memory");
  return v;
}
__device__ __forceinline__ float2 ld_relaxed_gpu_v2(const float* p) {
  float2 v;
  asm volatile("ld.relaxed.gpu.global.v2.f32 {%0, %1}, [%2];\n" : "=f"(v.x), "=f"(v.y) : "l"(p) : "memory");
  return v;
}
__device__ __forceinline__ float4 ld_relaxed_gpu_v4(const float* p) {
  float4 v;
  asm volatile("ld.relaxed.gpu.global.v4.f32 {%0, %1, %2, %3}, [%4];\n"
               : "=f"(v.x), "=f"(v.y), "=f"(v.z), "=f"(v.w)
               : "l"(p)
               : "memory");
  return v;
}

// Block b of G owns tiles [first(b), first(b + 1)) of W.
__device__ __forceinline__ int range_first(int b, int G, int W) {
  return static_cast<int>((static_cast<int64_t>(b) * W) / G);
}
// The block whose range holds tile t (the largest b with range_first(b) <= t).
__device__ __forceinline__ int block_of(int t, int G, int W) {
  return static_cast<int>(((static_cast<int64_t>(t) + 1) * G - 1) / W);
}

// Skeleton options of the main kernel (template argument kSkel; 0 = the default code; every option keeps every output
// bit): bit 0 - an interior tile takes its score maximum from the score loads themselves (tcgen05.ld .red .max), edge
// tiles keep the masked maximum; bit 1 - the four e4m3 probabilities of a word are converted and joined in one statement;
// bit 2 - the combine loads every output column of the tile before one wait; bit 3 - the load warp keeps eight page-list
// loads per lane in flight before storing them; bit 4 - the load warp's ring-refill wait carries a 256 ns suspend-time hint
// (still bounded); bit 5 - the probabilities go to tensor memory over the thread's own score columns and the PV MMAs read
// them there, so the QK of tile g + 2 waits for PV(g) instead of for the scores in registers. (Kept at namespace scope: the
// default build's code is then that of the kernel without options.)
template <int kSkel, int kMode>
struct Skel {
  static_assert(kSkel >= 0 && kSkel <= 63, "skeleton options: bits 0-5");
  static constexpr bool kRedMax = (kSkel & 1) != 0 && kMode != 2;
  static constexpr bool kPackJoined = (kSkel & 2) != 0;
  static constexpr bool kOneWaitO = (kSkel & 4) != 0;
  static constexpr bool kPageFill = (kSkel & 8) != 0;
  static constexpr uint32_t kLoadHint = (kSkel & 16) != 0 ? 256u : 0u;
  static constexpr bool kPTmem = (kSkel & 32) != 0;
  static_assert(!kPTmem || kMode == 0 || kMode == 4, "skeleton option bit 5: production or timeline mode");
};

// ---------------------------------------------------------------------------------------------- main kernel
template <int kPairs, int kStages, bool kUsePDL, int kPoly, int kPf, int kMode, int kTpr, int kSkel>
__global__ void __launch_bounds__(Layout<kPairs, kStages, kTpr>::kThreads, 1)
    full_attn_wide_kernel(const Params p, const __grid_constant__ CUtensorMap q_map,
                          const __grid_constant__ CUtensorMap k_map, const __grid_constant__ CUtensorMap v_map) {
  using L = Layout<kPairs, kStages, kTpr>;
  constexpr int kHeads = L::kHeads, kTileKeys = L::kTileKeys, kSmWarps = L::kSmWarps, kKpt = L::kKpt;
  // development switch carried in kPf: kPf == 1 -> poll the thread-arrival barriers (no suspend), no L2 prefetch
  constexpr bool kSpinP = kPf == 1;
  // skeleton options (kSkel; see Skel above)
  using SK = Skel<kSkel, kMode>;
  extern __shared__ uint8_t smem_raw[];
  // 1024-byte aligned base by pointer arithmetic on the shared array (keeps the shared address space visible to the
  // compiler: st.shared / ld.shared instead of generic accesses)
  uint8_t* const sm = smem_raw + ((1024u - (smem_u32(smem_raw) & 1023u)) & 1023u);
  uint64_t* const bars = reinterpret_cast<uint64_t*>(sm + L::kSmBar);
  // bars: full[kStages], empty[kStages], s_full[2], p_full[2], o_full[2], q_full[2], q_empty[2], (tmem address)
  const uint32_t bar0 = smem_u32(bars);
  const auto full_bar = [&](int s) { return bar0 + 8u * s; };
  const auto empty_bar = [&](int s) { return bar0 + 8u * (kStages + s); };
  const auto s_full_bar = [&](int b) { return bar0 + 8u * (2 * kStages + b); };
  const auto p_full_bar = [&](int b) { return bar0 + 8u * (2 * kStages + 2 + b); };
  const auto o_full_bar = [&](int b) { return bar0 + 8u * (2 * kStages + 4 + b); };
  const auto q_full_bar = [&](int b) { return bar0 + 8u * (2 * kStages + 6 + b); };
  const auto q_empty_bar = [&](int b) { return bar0 + 8u * (2 * kStages + 8 + b); };
  const auto s_free_bar = [&](int b) { return bar0 + 8u * (2 * kStages + 10 + b); };  // scores of buffer b in registers
  uint32_t* const tmem_slot = reinterpret_cast<uint32_t*>(sm + L::kSmBar + 8 * (2 * kStages + 12));
  int32_t* const pre = reinterpret_cast<int32_t*>(sm + L::kSmPre);  // pre[r] = tiles of requests < r; pre[R] = W
  int32_t* const pages = reinterpret_cast<int32_t*>(sm + L::kSmPages);

  // warp index through a warp shuffle: the compiler then knows it is warp-uniform, so the role branches below are
  // convergent (the MMA warp's descriptors stay in uniform registers and elect.sync needs no emulation)
  const int tid = threadIdx.x, warp = __shfl_sync(0xffffffffu, tid >> 5, 0), lane = tid & 31;
  const int b = blockIdx.x;
  const int R = p.num_reqs;
  // development mode 4: clock stamps of every hand-off for blocks 0, G/2 and G-1 (tiles < 256), written after the
  // partial-record region of the workspace: [3 blocks][16 events][256 tiles] u64
  const int trace_slot = kMode != 4 ? -1 : (b == 0 ? 0 : b == static_cast<int>(gridDim.x) / 2 ? 1 : b == static_cast<int>(gridDim.x) - 1 ? 2 : -1);
  uint64_t* const trace = reinterpret_cast<uint64_t*>(
      p.partial + static_cast<int64_t>(gridDim.x) * 2 * (2 * kPairs * kRows) * kRec);
  const auto stamp = [&](int ev, int gg) {
    if (kMode == 4 && trace_slot >= 0 && gg >= 0 && gg < 256) {
      uint64_t c;
      asm volatile("mov.u64 %0, %%clock64;\n" : "=l"(c));
      trace[(trace_slot * 16 + ev) * 256 + gg] = c;
    }
  };

  // Barriers and tensor memory first: they touch nothing the preceding kernels write.
  if (tid == 0) {
#pragma unroll
    for (int s = 0; s < kStages; ++s) {
      mbar_init(full_bar(s), 1);
      // released by the PV MMAs of every KV-head pair (one commit each; load-only / MMA-only probes: one arrival)
      mbar_init(empty_bar(s), 1);
    }
#pragma unroll
    for (int i = 0; i < 2; ++i) {
      mbar_init(s_full_bar(i), 1);
      mbar_init(p_full_bar(i), kSmWarps);  // one elected lane per softmax warp
      mbar_init(o_full_bar(i), 1);
      mbar_init(q_full_bar(i), 1);
      mbar_init(q_empty_bar(i), 1);
      mbar_init(s_free_bar(i), kSmWarps);
    }
    asm volatile("fence.mbarrier_init.release.cluster;\n" ::: "memory");
  }
  if (warp == 0) {
    asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], %1;\n" ::"r"(smem_u32(tmem_slot)),
                 "r"(kTmemCols)
                 : "memory");
    asm volatile("tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned;\n" ::: "memory");
  }
  if constexpr (kUsePDL) asm volatile("griddepcontrol.wait;\n" ::: "memory");
  // Prefix tile counts (warp 9): pre[r] for r = 0..R.
  if (warp == kSmWarps + 1) {
    int carry = 0;
    for (int r0 = 0; r0 < R; r0 += 32) {
      const int r = r0 + lane;
      int n = 0;
      if (r < R) n = (max(p.seq_lens[r], 1) + kTileKeys - 1) / kTileKeys;
      int x = n;
#pragma unroll
      for (int o = 1; o < 32; o <<= 1) {
        const int y = __shfl_up_sync(0xffffffffu, x, o);
        if (lane >= o) x += y;
      }
      if (r < R) pre[r + 1] = carry + x;
      carry += __shfl_sync(0xffffffffu, x, 31);
    }
    if (lane == 0) pre[0] = 0;
  }
  tc_fence_before();
  __syncthreads();
  tc_fence_after();
  const uint32_t tmem = __shfl_sync(0xffffffffu, *tmem_slot, 0);  // warp-uniform
  if (tid == 0) stamp(15, 0);
  const int W = pre[R];
  // blocks used: at most one per tile, so that every used block owns at least one tile
  const int G = min(static_cast<int>(gridDim.x), W);
  const int T0 = b < G ? range_first(b, G, W) : W, T1 = b < G ? range_first(b + 1, G, W) : W;
  const int ntiles = __shfl_sync(0xffffffffu, T1 - T0, 0);  // warp-uniform (provably, for the MMA warps' code)
  // first request of the range: the largest r with pre[r] <= T0 (requests have >= 1 tile)
  int r_first = 0;
  {
    int lo = 0, hi = R - 1;
    while (lo < hi) {
      const int mid = (lo + hi + 1) >> 1;
      if (pre[mid] <= T0) lo = mid; else hi = mid - 1;
    }
    r_first = lo;
  }

  if (ntiles > 0) {
    if (warp == kSmWarps) {
      // ------------------------------------------------------------------ loads
      // page numbers of the range -> shared memory (all lanes, all loads in flight together)
      constexpr int kTilesPerPage = 128 / kTileKeys;
      {
        int r = r_first, base = 0;
        for (int t = T0; t < T1 && base < kMaxPages;) {
          const int lo = t - pre[r], hi = min(T1, pre[r + 1]) - pre[r];
          const int pg0 = lo / kTilesPerPage, pg1 = (hi - 1) / kTilesPerPage + 1;
          const int32_t* const pt = p.page_table + static_cast<int64_t>(r) * p.page_table_stride;
          if constexpr (SK::kPageFill) {
            // option bit 3: eight page loads per lane issued before any store (a loop with one load per iteration keeps
            // only one load in flight per lane: its store waits for it before the next load issues); same entries
            const int n = min(pg1 - pg0, kMaxPages - base);
            for (int i0 = 0; i0 < n; i0 += 256) {
              int v[8];
#pragma unroll
              for (int k = 0; k < 8; ++k) {
                const int i = i0 + lane + 32 * k;
                v[k] = i < n ? pt[pg0 + i] : 0;
              }
#pragma unroll
              for (int k = 0; k < 8; ++k) {
                const int i = i0 + lane + 32 * k;
                if (i < n) pages[base + i] = v[k];
              }
            }
          } else {
          for (int i = lane; i < pg1 - pg0 && base + i < kMaxPages; i += 32) pages[base + i] = pt[pg0 + i];
          }
          base += pg1 - pg0;
          t = pre[r] + hi;
          ++r;
        }
      }
      __syncwarp();
      if (lane == 0) {
        asm volatile("fence.proxy.async.global;\n" ::: "memory");
        int g = 0, k = 0, r = r_first, base = 0;
        for (int t = T0; t < T1; ++k) {
          const int lo = t - pre[r], hi = min(T1, pre[r + 1]) - pre[r];
          const int pg0 = lo / kTilesPerPage;
          // q rows of request r -> Q buffer k & 1 (free once the QK MMAs of piece k - 2 finished)
          constexpr int kQB = L::kQBufs;
          if ((kMode == 0 || kMode == 2 || kMode == 4 || kMode == 6) && k >= kQB)
            mbar_wait(q_empty_bar(k % kQB), ((k / kQB) - 1) & 1);
          mbar_expect_tx(q_full_bar(k % kQB), kHeads * kRows * 64);
#pragma unroll
          for (int h = 0; h < kHeads; ++h)
            tma_load_3d(smem_u32(sm + L::kSmQ + (k % kQB) * L::kQBuf + h * kRows * 64), &q_map, 0, h * kQPerKv,
                        r * kTokens, q_full_bar(k % kQB));
          const int32_t* const pt = p.page_table + static_cast<int64_t>(r) * p.page_table_stride;
          for (int i = lo; i < hi; ++i, ++g) {
            if constexpr (kMode == 9) continue;  // tensor-core rate probe: no tile loads
            const int s = g % kStages;
            if constexpr (SK::kLoadHint != 0) {  // option bit 4
              if (g >= kStages) mbar_wait_hint<SK::kLoadHint>(empty_bar(s), ((g / kStages) - 1) & 1);
            } else {
            if (g >= kStages) mbar_wait(empty_bar(s), ((g / kStages) - 1) & 1);
            }
            const int pi = base + i / kTilesPerPage - pg0;
            const int page = pi < kMaxPages ? pages[pi] : pt[i / kTilesPerPage];
            const int row0 = page * 128 + (i % kTilesPerPage) * kTileKeys;
            if constexpr (kPf > 1) {
              // L2 prefetch of the tile kPf ahead in this piece (deeper memory-level parallelism than the ring)
              const int ip = i + kPf;
              if (ip < hi) {
                const int pip = base + ip / kTilesPerPage - pg0;
                const int pagep = pip < kMaxPages ? pages[pip] : pt[ip / kTilesPerPage];
                const int rowp = pagep * 128 + (ip % kTilesPerPage) * kTileKeys;
#pragma unroll
                for (int c = 0; c < kPairs; ++c) {
                  tma_prefetch_2d(&k_map, c * 128, rowp);
                  tma_prefetch_2d(&v_map, c * 128, rowp);
                }
              }
            }
            mbar_expect_tx(full_bar(s), 2 * kTileBytes);
#pragma unroll
            for (int c = 0; c < kPairs; ++c) {
              tma_load_2d(smem_u32(sm + L::kSmK + s * kTileBytes + c * kTileKeys * 128), &k_map, c * 128, row0,
                          full_bar(s));
              tma_load_2d(smem_u32(sm + L::kSmV + s * kTileBytes + c * kTileKeys * 128), &v_map, c * 128, row0,
                          full_bar(s));
            }
            stamp(0, g);
          }
          base += (hi - 1) / kTilesPerPage + 1 - pg0;
          t = pre[r] + hi;
          ++r;
        }
      }
    } else if (warp == kSmWarps + 1) {
      // ------------------------------------------------------------------ MMA issue
      if (kMode == 1 && lane == 0) {
        for (int g = 0; g < ntiles; ++g) {  // load-only probe: release every stage as soon as it landed
          mbar_wait(full_bar(g % kStages), (g / kStages) & 1);
          if (g + kStages < ntiles) mbar_arrive(empty_bar(g % kStages));
        }
      } else if ((kMode == 3 || kMode == 9) && lane == 0) {
        // development probe: MMAs only (QK then PV of every tile, back to back, no softmax hand-offs); mode 9: no data
        // waits after the first stages (pure tensor-core issue/execute rate; kPf 1 = QK only, 2 = PV only)
        constexpr uint32_t id_qk = idesc_f8(64, kTileKeys, 0), id_pv = idesc_f8(64, 64, 1);
        const uint32_t sp = smem_u32(sm + L::kSmP), sq = smem_u32(sm + L::kSmQ);
        for (int g = 0; g < ntiles; ++g) {
          const int s = g % kStages, bb = g & 1;
          if (kMode == 3) mbar_wait(full_bar(s), (g / kStages) & 1);
          tc_fence_after();
          const uint32_t sk = smem_u32(sm + L::kSmK + s * kTileBytes), sv = smem_u32(sm + L::kSmV + s * kTileBytes);
#pragma unroll
          for (int c = 0; c < kPairs; ++c)
#pragma unroll
            for (int hh = 0; hh < 2; ++hh) {
#pragma unroll
              for (int kk = 0; kk < 2; ++kk)
                if (kMode == 3 || kPf != 2)
                mma_f8(tmem + ((16u * hh) << 16) + 128u * bb + 64u * c,
                       sdesc(sq + (2 * c + hh) * kRows * 64 + 32 * kk, 512, kSw64),
                       sdesc(sk + c * kTileKeys * 128 + 64 * hh + 32 * kk, 1024, kSw128), id_qk, kk);
#pragma unroll
              for (int kk = 0; kk < kTileKeys / 32; ++kk)
                if (kMode == 3 || kPf != 1)
                mma_f8(tmem + ((16u * hh) << 16) + 256u + 128u * bb + 64u * c,
                       sdesc(sp + (bb * kHeads + 2 * c + hh) * kRows * L::kPRowBytes + 32 * kk, kPairs == 2 ? 512 : 1024,
                             kPairs == 2 ? kSw64 : kSw128),
                       sdesc(sv + c * kTileKeys * 128 + 64 * hh + 4096 * kk, 1024, kSw128), id_pv, kk);
            }
          if (kMode == 3 && g + kStages < ntiles) mma_commit(empty_bar(s));
        }
        mma_commit(o_full_bar(0));
        mbar_wait(o_full_bar(0), 0);
      } else if (kMode != 1 && kMode != 3 && kMode != 9) {
        // QK issue (whole warp, warp-uniform operands, one elected lane issues). Descriptors: uniform bases plus
        // compile-time offsets (the start-address field is linear; shared-memory offsets stay inside it).
        constexpr uint32_t id_qk = idesc_f8(64, kTileKeys, 0);
        const uint64_t dq0 = sdesc(smem_u32(sm + L::kSmQ), 512, kSw64);
        const uint64_t dk0 = sdesc(smem_u32(sm + L::kSmK), 1024, kSw128);
        int r_q = r_first, k_q = 0, end_q = __shfl_sync(0xffffffffu, min(T1, pre[r_first + 1]), 0);
        const int T0u = __shfl_sync(0xffffffffu, T0, 0);
        const auto qk_tile = [&](int gq) {
          const int t = T0u + gq;
          if (t >= end_q) {  // first tile of the next piece: release the previous Q buffer, wait for this one
            mma_commit_e(q_empty_bar(k_q % L::kQBufs));
            ++r_q;
            ++k_q;
            end_q = __shfl_sync(0xffffffffu, min(T1, pre[r_q + 1]), 0);
            mbar_wait(q_full_bar(k_q % L::kQBufs), (k_q / L::kQBufs) & 1);
            __syncwarp();
            tc_fence_after();
          }
          const int s = gq % kStages, bq = gq & 1;
          if (lane == 0) stamp(10, gq);
          mbar_wait(full_bar(s), (gq / kStages) & 1);
          __syncwarp();
          if (lane == 0) stamp(1, gq);
          tc_fence_after();
          const uint64_t dq = dq0 + static_cast<uint64_t>(((k_q % L::kQBufs) * L::kQBuf) >> 4);
          const uint64_t dk = dk0 + static_cast<uint64_t>((s * kTileBytes) >> 4);
#pragma unroll
          for (int c = 0; c < kPairs; ++c)
#pragma unroll
            for (int hh = 0; hh < 2; ++hh)
#pragma unroll
              for (int kk = 0; kk < 2; ++kk)
                mma_f8_e(tmem + ((16u * hh) << 16) + 128u * bq + 64u * c,
                         dq + static_cast<uint64_t>(((2 * c + hh) * kRows * 64 + 32 * kk) >> 4),
                         dk + static_cast<uint64_t>((c * kTileKeys * 128 + 64 * hh + 32 * kk) >> 4), id_qk, kk);
          mma_commit_e(s_full_bar(bq));
          if (lane == 0) stamp(2, gq);
        };
        mbar_wait(q_full_bar(0), 0);
        __syncwarp();
        tc_fence_after();
        qk_tile(0);
        if (ntiles > 1) qk_tile(1);
        if constexpr (SK::kPTmem) {
          for (int g = 0; g + 2 < ntiles; ++g) {
            // option bit 5: tile g's probabilities sit in score buffer g & 1 until its PV finished (the scores-in-registers
            // barrier is still waited first, so every one of its phases has a waiter; it completes before PV(g) can)
            mbar_wait(s_free_bar(g & 1), (g >> 1) & 1);
            mbar_wait(o_full_bar(g & 1), (g >> 1) & 1);
            __syncwarp();
            tc_fence_after();
            qk_tile(g + 2);
          }
        } else {
        for (int g = 0; g + 2 < ntiles; ++g) {
          mbar_wait(s_free_bar(g & 1), (g >> 1) & 1);
          __syncwarp();
          tc_fence_after();
          qk_tile(g + 2);
        }
        }
      }
    } else if (warp == kSmWarps + 2) {
      // ------------------------------------------------------------------ PV issue (warp-uniform, elected lane)
      if (kMode != 1 && kMode != 3 && kMode != 9) {
        constexpr uint32_t id_pv = idesc_f8(64, 64, 1);
        constexpr uint32_t kPSbo = kPairs == 2 ? 512 : 1024, kPSw = kPairs == 2 ? kSw64 : kSw128;
        const uint64_t dp0 = sdesc(smem_u32(sm + L::kSmP), kPSbo, kPSw);
        const uint64_t dv0 = sdesc(smem_u32(sm + L::kSmV), 1024, kSw128);
        for (int g = 0; g < ntiles; ++g) {
          const int s = g % kStages, bb = g & 1;
          if (lane == 0) stamp(11, g);
          mbar_wait(p_full_bar(bb), (g >> 1) & 1);
          __syncwarp();
          if (lane == 0) stamp(6, g);
          tc_fence_after();
          const uint64_t dp = dp0 + static_cast<uint64_t>((bb * kHeads * kRows * L::kPRowBytes) >> 4);
          const uint64_t dv = dv0 + static_cast<uint64_t>((s * kTileBytes) >> 4);
          if constexpr (SK::kPTmem) {
#pragma unroll
            for (int c = 0; c < kPairs; ++c)
#pragma unroll
              for (int hh = 0; hh < 2; ++hh)
#pragma unroll
                for (int kk = 0; kk < kTileKeys / 32; ++kk)
                  // option bit 5: keys 32 kk .. of the row are the e4m3 words (4 keys each) over the score columns of
                  // the softmax thread that holds them: thread half 32 kk / kKpt, its word (32 kk % kKpt) / 4
                  mma_f8_ts_e(tmem + ((16u * hh) << 16) + 256u + 128u * bb + 64u * c,
                              tmem + ((16u * hh) << 16) + 128u * bb + 64u * c + (32 * kk / kKpt) * kKpt +
                                  (32 * kk % kKpt) / 4,
                              dv + static_cast<uint64_t>((c * kTileKeys * 128 + 64 * hh + 4096 * kk) >> 4), id_pv, kk);
          } else {
#pragma unroll
          for (int c = 0; c < kPairs; ++c)
#pragma unroll
            for (int hh = 0; hh < 2; ++hh)
#pragma unroll
              for (int kk = 0; kk < kTileKeys / 32; ++kk)
                mma_f8_e(tmem + ((16u * hh) << 16) + 256u + 128u * bb + 64u * c,
                         dp + static_cast<uint64_t>(((2 * c + hh) * kRows * L::kPRowBytes + 32 * kk) >> 4),
                         dv + static_cast<uint64_t>((c * kTileKeys * 128 + 64 * hh + 4096 * kk) >> 4), id_pv, kk);
          }
          mma_commit_e(o_full_bar(bb));
          if (lane == 0) stamp(7, g);
          if (g + kStages < ntiles) mma_commit_e(empty_bar(s));
        }
      }
    } else if (kMode == 0 || kMode == 2 || kMode == 4 || kMode == 6) {
      // ------------------------------------------------------------------ softmax (warps 0 .. kSmWarps-1)
      // Two threads per query row: warp w -> half = w / (kSmWarps / 2) (keys [kKpt half, +kKpt) of every tile, output
      // dims [32 half, +32)); wi = w % (kSmWarps / 2) -> lane group q4 = wi % 4 (tensor-memory lanes 32 q4 ..),
      // KV-head pair c = wi / 4; lane i -> head 2c + i / 16, row 16 q4 + i % 16. The two threads of a row (warps wi and
      // wi + kSmWarps / 2) exchange tile maxima and row sums through shared memory and named barrier 1 + wi.
      constexpr int kHalfWarps = kSmWarps / kTpr;
      const int half = warp / kHalfWarps, wi = warp % kHalfWarps;  // kTpr 1: half 0, wi = warp
      const int q4 = wi & 3, c = wi >> 2;
      const int hh = lane >> 4, row = 16 * q4 + (lane & 15);
      const int head = 2 * c + hh;  // KV head within the rank
      const int tok = row >> 3;
      constexpr int kDims = 64 / kTpr;  // output dims per thread
      const uint32_t lane_base = (32u * q4) << 16;
      const uint32_t s_col = 64u * c + kKpt * half;
      const uint32_t o_col = 64u * c + kDims * half;
      float* const red = reinterpret_cast<float*>(sm + L::kSmRed);
      const int rec_row = head * kRows + row;  // row of the block's record (0 .. kHeads * 64 - 1)
      constexpr int kRecRows = kHeads * kRows;
      const uint32_t xbar = 1 + wi;
      int g = 0;
      int r = r_first;
      for (int t = T0; t < T1; ++r) {
        const int lo = t - pre[r], hi = min(T1, pre[r + 1]) - pre[r];
        const int n_r = pre[r + 1] - pre[r];
        const int seq_len = p.seq_lens[r];
        const int pos = seq_len - kTokens + tok;
        float2 o2[kDims / 2];
#pragma unroll
        for (int d = 0; d < kDims / 2; ++d) o2[d] = make_float2(0.0f, 0.0f);
        float m = -INFINITY, l = 0.0f, ar_prev = 0.0f, at_prev = 0.0f;
        // rescale-and-add of an output tile into o (packed fp32 pairs: the same two roundings as fmaf(o, ar, v * at))
        const auto combine = [&](int gg) {
          const int bp = gg & 1;
          mbar_wait(o_full_bar(bp), (gg >> 1) & 1);
          if (tid == 0) stamp(8, gg);
          __syncwarp();  // each thread left the wait on its own: reconverge before the .sync.aligned tensor-memory loads
          tc_fence_after();
          const float2 a2 = make_float2(ar_prev, ar_prev), t2 = make_float2(at_prev, at_prev);
          if constexpr (SK::kOneWaitO) {
            // option bit 2: every output column of the tile loaded before one wait (one tensor-memory load latency per
            // combine instead of one per 32 columns); the same operations per element
            float v[kDims];
#pragma unroll
            for (int d0 = 0; d0 < kDims; d0 += 32)
              tmem_ld32(tmem + lane_base + 256u + 128u * bp + o_col + d0, *reinterpret_cast<float(*)[32]>(v + d0));
            tmem_wait_ld();
#pragma unroll
            for (int d = 0; d < kDims / 2; ++d) o2[d] = fma2(o2[d], a2, mul2(make_float2(v[2 * d], v[2 * d + 1]), t2));
          } else {
#pragma unroll
          for (int d0 = 0; d0 < kDims; d0 += 32) {
            float v[32];
            tmem_ld32(tmem + lane_base + 256u + 128u * bp + o_col + d0, v);
            tmem_wait_ld();
#pragma unroll
            for (int d = 0; d < 16; ++d)
              o2[d0 / 2 + d] = fma2(o2[d0 / 2 + d], a2, mul2(make_float2(v[2 * d], v[2 * d + 1]), t2));
          }
          }
          tc_fence_before();
          if (tid == 0) stamp(9, gg);
        };
        for (int i = lo; i < hi; ++i, ++g) {
          const int bb = g & 1;
          const int key0 = i * kTileKeys + kKpt * half;  // this thread's first key of the tile
          mbar_wait(s_full_bar(bb), (g >> 1) & 1);
          if (tid == 0) stamp(3, g);
          __syncwarp();  // each thread left the wait on its own: reconverge before the .sync.aligned tensor-memory loads
          tc_fence_after();
          float x[kKpt];
          float smax;
          if constexpr (SK::kRedMax) {
            // option bit 0: the maxima of the loaded score columns come with the loads (tcgen05.ld .red .max); an interior
            // tile (no masked key) takes them, the same maximum of the same values; edge tiles keep the masked maximum
            float rred[kKpt / 32];
#pragma unroll
            for (int j = 0; j < kKpt; j += 32)
              rred[j / 32] = tmem_ld32_max(tmem + lane_base + 128u * bb + s_col + j, *reinterpret_cast<float(*)[32]>(x + j));
            tmem_wait_ld();  // the loaded scores and the reductions are read only after this wait
            tc_fence_before();
            if (tid == 0) stamp(4, g);
            __syncwarp();
            if (lane == 0) mbar_arrive(s_free_bar(bb));  // score buffer bb may take the QK of tile g + 2
            if (key0 + kKpt - 1 > seq_len - kTokens) {
#pragma unroll
              for (int j = 0; j < kKpt; ++j)
                if (key0 + j > pos) x[j] = -INFINITY;
              float mx[kKpt / 3 + 1];
#pragma unroll
              for (int j = 0; j < kKpt / 3; ++j) mx[j] = max3(x[3 * j], x[3 * j + 1], x[3 * j + 2]);
              smax = kKpt % 3 == 2 ? fmaxf(x[kKpt - 2], x[kKpt - 1]) : x[kKpt - 1];
#pragma unroll
              for (int j = 0; j < kKpt / 3; j += 2)
                smax = j + 1 < kKpt / 3 ? max3(smax, mx[j], mx[j + 1]) : fmaxf(smax, mx[j]);
            } else {
              smax = rred[0];
#pragma unroll
              for (int j = 1; j < kKpt / 32; ++j) smax = fmaxf(smax, rred[j]);
            }
          } else {
#pragma unroll
          for (int j = 0; j < kKpt; j += 32)
            tmem_ld32(tmem + lane_base + 128u * bb + s_col + j, *reinterpret_cast<float(*)[32]>(x + j));
          tmem_wait_ld();
          tc_fence_before();
          if (tid == 0) stamp(4, g);
          __syncwarp();
          if (lane == 0) mbar_arrive(s_free_bar(bb));  // score buffer bb may take the QK of tile g + 2
          // tile edge: keys after the row's position do not take part. The test uses the smallest position of the request
          // (token 0), so it is warp-uniform: interior tiles branch around the masking instead of predicating it per key.
          if (key0 + kKpt - 1 > seq_len - kTokens) {
#pragma unroll
            for (int j = 0; j < kKpt; ++j)
              if (key0 + j > pos) x[j] = -INFINITY;
          }
          if constexpr (kMode == 2) {  // development probe: no softmax arithmetic, P = 0
#pragma unroll
            for (int j = 0; j < kKpt; ++j) x[j] = 0.0f;
          }
          // maximum of the raw scores (the scale is positive, so max(s) * scale == max(s * scale))
          float mx[kKpt / 3 + 1];
#pragma unroll
          for (int j = 0; j < kKpt / 3; ++j) mx[j] = max3(x[3 * j], x[3 * j + 1], x[3 * j + 2]);
          smax = kKpt % 3 == 2 ? fmaxf(x[kKpt - 2], x[kKpt - 1]) : x[kKpt - 1];
#pragma unroll
          for (int j = 0; j < kKpt / 3; j += 2) smax = j + 1 < kKpt / 3 ? max3(smax, mx[j], mx[j + 1]) : fmaxf(smax, mx[j]);
          }
          if constexpr (kTpr == 2) {
            red[(bb * kRecRows + rec_row) * 2 + half] = smax;
            asm volatile("bar.sync %0, %1;\n" ::"r"(xbar), "r"(64) : "memory");
            smax = fmaxf(red[(bb * kRecRows + rec_row) * 2], red[(bb * kRecRows + rec_row) * 2 + 1]);
          }
          // the tile's own maximum: the probabilities of the tile are quantized against it
          const float mtile = smax == -INFINITY ? -INFINITY : smax * p.scale_log2;
          const float mu = mtile == -INFINITY ? 0.0f : mtile;
          const float mnew = fmaxf(m, mtile);
          const float ar = m == -INFINITY ? 0.0f : fast_exp2(m - mnew);
          const float at = mtile == -INFINITY ? 0.0f : fast_exp2(mtile - mnew);
          // 448 * 2^y as 2^(y + log2 448): probabilities come out pre-scaled for the e4m3 conversion, and the row sums are
          // kept in the same (x 448) units; the output scale divides by them (bmm2_scale / l448)
          const float nmu448 = kLog2PScale - mu;
          const float2 sc = make_float2(p.scale_log2, p.scale_log2), nmu = make_float2(nmu448, nmu448);
          float2 ls[2] = {make_float2(0.0f, 0.0f), make_float2(0.0f, 0.0f)};
#pragma unroll
          for (int j = 0; j < kKpt && kMode != 2; j += 2) {
            const float2 y = fma2(make_float2(x[j], x[j + 1]), sc, nmu);
            // kPoly of every 8 keys take 2^y from the FMA pipes (exp2_poly2), the rest from ex2.approx
            const float2 pj = (j & 7) < kPoly ? exp2_poly2(y) : make_float2(fast_exp2(y.x), fast_exp2(y.y));
            ls[(j >> 1) & 1] = add2(ls[(j >> 1) & 1], pj);
            x[j] = pj.x;
            x[j + 1] = pj.y;
          }
          // this thread's part of the tile row sum (its keys in four interleaved partial sums, (0 + 2) + (1 + 3))
          const float lpart = (ls[0].x + ls[1].x) + (ls[0].y + ls[1].y);
          m = mnew;
          // probabilities -> P buffer bb. kPairs 2: rows of 64 B in the 64-byte swizzle (chunk q of row r at
          // q ^ (r / 2 % 4)); kPairs 1: rows of 128 B in the 128-byte swizzle (chunk q of row r at q ^ (r % 8)).
          // Buffer bb was last read by the MMAs of tile g - 2, finished before the output wait of the previous tile.
          if constexpr (SK::kPTmem || SK::kPackJoined) {
          // four e4m3 probabilities (key order) in one word; option bit 1 joins the two conversions in one statement
          const auto pack4 = [](float a, float b, float c, float d) -> uint32_t {
            if constexpr (SK::kPackJoined) {
              return e4m3x4_joined(a, b, c, d);
            } else {
              return e4m3x4(a, b, c, d);
            }
          };
          if constexpr (SK::kPTmem) {
            // option bit 5: this thread's kKpt e4m3 values (key order, four per column) over the first kKpt / 4 of its own
            // score columns of buffer bb (A operand of the PV MMA; the QK of tile g + 2 rewrites them only after PV(g)'s
            // commit); no shared-memory store, so no async-proxy fence
            static_assert(kKpt == 64, "tensor-memory probabilities: 64 keys per softmax thread");
            uint32_t pw[16];
#pragma unroll
            for (int j = 0; j < 16; ++j) pw[j] = pack4(x[4 * j], x[4 * j + 1], x[4 * j + 2], x[4 * j + 3]);
            tmem_st16(tmem + lane_base + 128u * bb + s_col, pw);
            tmem_wait_st();
          } else {
            uint8_t* const prow = sm + L::kSmP + (bb * kHeads + head) * kRows * L::kPRowBytes + row * L::kPRowBytes;
#pragma unroll
            for (int cc = 0; cc < kKpt / 16; ++cc) {
              uint4 w;
              w.x = pack4(x[16 * cc + 0], x[16 * cc + 1], x[16 * cc + 2], x[16 * cc + 3]);
              w.y = pack4(x[16 * cc + 4], x[16 * cc + 5], x[16 * cc + 6], x[16 * cc + 7]);
              w.z = pack4(x[16 * cc + 8], x[16 * cc + 9], x[16 * cc + 10], x[16 * cc + 11]);
              w.w = pack4(x[16 * cc + 12], x[16 * cc + 13], x[16 * cc + 14], x[16 * cc + 15]);
              const int q = (kKpt / 16) * half + cc;
              const int chunk = kPairs == 2 ? (q ^ ((row >> 1) & 3)) : (q ^ (row & 7));
              *reinterpret_cast<uint4*>(prow + (chunk << 4)) = w;
            }
            if constexpr (kMode != 6) fence_proxy_async();  // mode 6: timing probe only (P visibility not ordered)
          }
          } else {
          uint8_t* const prow = sm + L::kSmP + (bb * kHeads + head) * kRows * L::kPRowBytes + row * L::kPRowBytes;
#pragma unroll
          for (int cc = 0; cc < kKpt / 16; ++cc) {
            uint4 w;
            w.x = e4m3x4(x[16 * cc + 0], x[16 * cc + 1], x[16 * cc + 2], x[16 * cc + 3]);
            w.y = e4m3x4(x[16 * cc + 4], x[16 * cc + 5], x[16 * cc + 6], x[16 * cc + 7]);
            w.z = e4m3x4(x[16 * cc + 8], x[16 * cc + 9], x[16 * cc + 10], x[16 * cc + 11]);
            w.w = e4m3x4(x[16 * cc + 12], x[16 * cc + 13], x[16 * cc + 14], x[16 * cc + 15]);
            const int q = (kKpt / 16) * half + cc;
            const int chunk = kPairs == 2 ? (q ^ ((row >> 1) & 3)) : (q ^ (row & 7));
            *reinterpret_cast<uint4*>(prow + (chunk << 4)) = w;
          }
          if constexpr (kMode != 6) fence_proxy_async();  // mode 6: timing probe only (P visibility not ordered)
          }
          tc_fence_before();
          if (tid == 0) stamp(5, g);
          __syncwarp();
          if (lane == 0) mbar_arrive(p_full_bar(bb));
          // this thread's half of the row sum (the two halves are added once, at the end of the piece)
          if (i > lo) combine(g - 1);
          l = l * ar + lpart * at;
          ar_prev = ar;
          at_prev = at;
        }
        combine(g - 1);
        if constexpr (kTpr == 2) {
          red[(2 * kRecRows + rec_row) * 2 + half] = l;
          asm volatile("bar.sync %0, %1;\n" ::"r"(xbar), "r"(64) : "memory");
          l = red[(2 * kRecRows + rec_row) * 2] + red[(2 * kRecRows + rec_row) * 2 + 1];
          asm volatile("bar.sync %0, %1;\n" ::"r"(xbar), "r"(64) : "memory");  // red reused by the next piece
        }
        const int qh = head * kQPerKv + (row & 7);  // query head within the rank
        __nv_bfloat16* const orow =
            p.out + ((static_cast<int64_t>(r) * kTokens + tok) * p.num_q_heads + qh) * kHeadDim + kDims * half;
        const auto store_out = [&](const float2 (&acc)[kDims / 2], float f) {
#pragma unroll
          for (int d = 0; d < kDims / 2; d += 4) {
            uint4 w;
            __nv_bfloat162 h0 = __floats2bfloat162_rn(acc[d].x * f, acc[d].y * f);
            __nv_bfloat162 h1 = __floats2bfloat162_rn(acc[d + 1].x * f, acc[d + 1].y * f);
            __nv_bfloat162 h2 = __floats2bfloat162_rn(acc[d + 2].x * f, acc[d + 2].y * f);
            __nv_bfloat162 h3 = __floats2bfloat162_rn(acc[d + 3].x * f, acc[d + 3].y * f);
            w.x = *reinterpret_cast<uint32_t*>(&h0);
            w.y = *reinterpret_cast<uint32_t*>(&h1);
            w.z = *reinterpret_cast<uint32_t*>(&h2);
            w.w = *reinterpret_cast<uint32_t*>(&h3);
            *reinterpret_cast<uint4*>(orow + 2 * d) = w;
          }
        };
        if (lo == 0 && hi == n_r) {
          // the whole request in this block: normalize and store
          store_out(o2, l > 0.0f ? p.out_scale / l : 0.0f);
        } else {
          // partial record -> slot (block, 0) when the request started in an earlier block, else (block, 1)
          const int64_t idx = static_cast<int64_t>(b) * 2 + ((pre[r] < T0) ? 0 : 1);
          float* const rec = p.partial + (idx * kRecRows + rec_row) * kRec;
#pragma unroll
          for (int d = 0; d < kDims / 2; d += 2)
            *reinterpret_cast<float4*>(rec + kDims * half + 2 * d) = make_float4(o2[d].x, o2[d].y, o2[d + 1].x, o2[d + 1].y);
          if (half == 0) *reinterpret_cast<float2*>(rec + kHeadDim) = make_float2(m, l);
          // merged by full_attn_wide_merge_kernel (launched after this kernel)
        }
        t = pre[r] + hi;
      }
    }
  }
  tc_fence_before();
  __syncthreads();
  if (tid == 0) stamp(14, 0);
  if constexpr (kUsePDL) asm volatile("griddepcontrol.launch_dependents;\n" ::: "memory");
  if (warp == 0) {
    tc_fence_after();
    asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, %1;\n" ::"r"(tmem), "r"(kTmemCols) : "memory");
  }
}

// ---------------------------------------------------------------------------------------------- merge kernel
// Output norm (no weight) + sigmoid gate of 4 consecutive dims of a 64-dim row held by one of the row's 16 threads (a
// 16-lane half warp; thread `part` holds dims 4 part .. 4 part + 3), bitwise to SGLang's norm_gate kernel in its kVec-byte
// configuration: kVec 8 = 16 threads per row with 4 values each, kVec 16 = 8 threads per row with 8 values each. Per
// norm_gate thread: ss = fmaf over its values in order from 0.0f; then a butterfly over the row's norm_gate threads
// (xor masks from half the thread count down to 1; every lane ends with the same fp32 value); r = rsqrtf(ss / 64 + eps);
// y = bf16((x * r) * (1 / (1 + exp(-g)))) with __fdividef / __expf. With kVec 16, the even lane of a pair starts the
// chain on its 4 values and the odd lane continues it on its own 4 (the 8 values of one norm_gate thread, in order); the
// butterfly runs over the odd lanes (masks 8, 4, 2) and the result is shared with the even lane. `graw` = the thread's 4
// gate values (bf16), loaded by the caller.
template <int kVec>
__device__ __forceinline__ void norm_gate4(__nv_bfloat162& h0, __nv_bfloat162& h1, const uint2 graw, float eps) {
  static_assert(kVec == 8 || kVec == 16, "norm_gate vector width");
  const float2 x01 = __bfloat1622float2(h0), x23 = __bfloat1622float2(h1);
  const unsigned group = 0xffffu << (threadIdx.x & 16);  // the 16 threads of this row
  const auto chain = [&](float c) {
    c = fmaf(x01.x, x01.x, c);
    c = fmaf(x01.y, x01.y, c);
    c = fmaf(x23.x, x23.x, c);
    return fmaf(x23.y, x23.y, c);
  };
  float ss;
  if constexpr (kVec == 8) {
    ss = chain(0.0f);
#pragma unroll
    for (int mask = 8; mask >= 1; mask >>= 1) ss = ss + __shfl_xor_sync(group, ss, mask, 16);
  } else {
    const float e = chain(0.0f);
    const float pe = __shfl_up_sync(group, e, 1, 16);  // odd lane: its even partner's chain
    float t = (threadIdx.x & 1) ? chain(pe) : e;
#pragma unroll
    for (int mask = 8; mask >= 2; mask >>= 1) t = t + __shfl_xor_sync(group, t, mask, 16);
    ss = __shfl_sync(group, t, static_cast<int>((threadIdx.x & 15) | 1), 16);
  }
  const float rs = rsqrtf(ss / 64.0f + eps);
  const float2 g01 = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&graw.x));
  const float2 g23 = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&graw.y));
  const auto sig = [](float gv) { return __fdividef(1.0f, 1.0f + __expf(-gv)); };
  float y0 = x01.x * rs, y1 = x01.y * rs, y2 = x23.x * rs, y3 = x23.y * rs;
  y0 *= sig(g01.x);
  y1 *= sig(g01.y);
  y2 *= sig(g23.x);
  y3 *= sig(g23.y);
  h0 = __floats2bfloat162_rn(y0, y1);
  h1 = __floats2bfloat162_rn(y2, y3);
}

// Requests whose tiles span more than one block of the main kernel: grid (kHeads * 64 / kMergeRows, R), 256 threads;
// block (x, r) merges rows [16 x, 16 x + 16) of request r, 16 threads per row (thread: 4 output dims). The work split is
// recomputed from seq_lens exactly as the main kernel computes it (same G, same W). Records of the pieces j = b0..b1
// (b0: slot 1, others: slot 0), in block order: M = max m_j, O = sum exp2(m_j - M) o_j, L = sum exp2(m_j - M) l_j;
// out = bf16(O * (out_scale / L)). Latency rounds: one for the tile range (every lane issues all its seq_lens loads first),
// one for the pieces' maxima (spread over the 16 threads of a row) together with the first 8 pieces' outputs, one per
// further 8 pieces. The arithmetic (maximum, weights, fused multiply-adds in block order) is that of the earlier merge.
// kGate: every request is written here, and its rows also get the output norm and gate (norm_gate4<kNgVec>: SGLang's
// norm_gate arithmetic in the configuration it picks for the call, kNgVec 8 or 16 bytes per thread). A request inside one
// block: the main kernel's bf16 output row (already normalized) is read back and gated in place. The thread's gate values
// and that output row part are loaded at the head, with the seq_lens loads (the output part is used only for such a
// request; otherwise the main kernel did not write it and the value is dropped).
constexpr int kMergeRows = 16;
constexpr int kMergeBatch = 8;
template <int kPairs, bool kUsePDL, bool kGate = false, int kNgVec = 8>
__global__ void __launch_bounds__(256) full_attn_wide_merge_kernel(const Params p, int grid_blocks) {
  constexpr int kHeads = 2 * kPairs, kTileKeys = 128 / kPairs, kRecRows = kHeads * kRows;
  __shared__ int32_t range[3];  // tiles of requests < r, of requests <= r, of all requests
  if constexpr (kUsePDL) asm volatile("griddepcontrol.wait;\n" ::: "memory");
  const int R = p.num_reqs, r = blockIdx.y, tid = threadIdx.x, lane = tid & 31;
  const int row = kMergeRows * blockIdx.x + (tid >> 4), part = tid & 15;  // row of the request's records, 4-dim part
  const int head = row >> 6, r64 = row & 63, tok = r64 >> 3, qh = head * kQPerKv + (r64 & 7);
  __nv_bfloat16* const orow =
      p.out + ((static_cast<int64_t>(r) * kTokens + tok) * p.num_q_heads + qh) * kHeadDim + 4 * part;
  // kGate: the thread's 4 gate values and 4 output values do not depend on the work split; load them now, in flight
  // together with the seq_lens loads below (read after the wait, as before)
  uint2 graw = make_uint2(0u, 0u), oraw = make_uint2(0u, 0u);
  if constexpr (kGate) {
    graw = *reinterpret_cast<const uint2*>(p.gate + (static_cast<int64_t>(r) * kTokens + tok) * p.gate_token_stride +
                                           qh * kHeadDim + 4 * part);
    oraw = *reinterpret_cast<const uint2*>(orow);
  }
  if (tid < 32) {
    int n[kMaxReqs / 32];
#pragma unroll
    for (int k = 0; k < kMaxReqs / 32; ++k) {
      const int rr = lane + 32 * k;
      n[k] = rr < R ? p.seq_lens[rr] : 0;
    }
    int before = 0, upto = 0, all = 0;
#pragma unroll
    for (int k = 0; k < kMaxReqs / 32; ++k) {
      const int rr = lane + 32 * k;
      const int t = rr < R ? (max(n[k], 1) + kTileKeys - 1) / kTileKeys : 0;
      all += t;
      if (rr < r) before += t;
      if (rr <= r) upto += t;
    }
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) {
      before += __shfl_xor_sync(0xffffffffu, before, o);
      upto += __shfl_xor_sync(0xffffffffu, upto, o);
      all += __shfl_xor_sync(0xffffffffu, all, o);
    }
    if (lane == 0) {
      range[0] = before;
      range[1] = upto;
      range[2] = all;
    }
  }
  __syncthreads();
  const int W = range[2];
  const int G = min(grid_blocks, W);
  const int b0 = block_of(range[0], G, W), b1 = block_of(range[1] - 1, G, W);
  if (b0 == b1 && !kGate) return;  // the request lies in one block: written by the main kernel
  __nv_bfloat162 h0, h1;
  if (kGate && b0 == b1) {
    // a request inside one block: the main kernel's normalized bf16 output (loaded at the head)
    h0 = *reinterpret_cast<const __nv_bfloat162*>(&oraw.x);
    h1 = *reinterpret_cast<const __nv_bfloat162*>(&oraw.y);
  } else {
  float acc[4] = {0.0f, 0.0f, 0.0f, 0.0f};
  float lsum = 0.0f;
  const auto rec_of = [&](int j) {
    return p.partial + ((static_cast<int64_t>(j) * 2 + (j == b0 ? 1 : 0)) * kRecRows + row) * kRec;
  };
  // first batch of outputs and row sums in flight together with the maxima
  float2 ml[kMergeBatch];
  float4 v[kMergeBatch];
#pragma unroll
  for (int u = 0; u < kMergeBatch; ++u) {
    if (b0 + u > b1) continue;
    const float* const rj = rec_of(b0 + u);
    ml[u] = *reinterpret_cast<const float2*>(rj + kHeadDim);
    v[u] = *reinterpret_cast<const float4*>(rj + 4 * part);
  }
  // maximum over the pieces: thread `part` of the row reads pieces b0 + part, b0 + part + 16, ...; 16-lane max
  float mx = -INFINITY;
  for (int j = b0 + part; j <= b1; j += 16) mx = fmaxf(mx, rec_of(j)[kHeadDim]);
#pragma unroll
  for (int o = 8; o > 0; o >>= 1) mx = fmaxf(mx, __shfl_xor_sync(0xffffffffu, mx, o));
  // weighted sums in block order
  for (int j0 = b0;;) {
#pragma unroll
    for (int u = 0; u < kMergeBatch; ++u) {
      if (j0 + u > b1 || ml[u].x == -INFINITY) continue;
      const float w = fast_exp2(ml[u].x - mx);
      acc[0] += w * v[u].x;
      acc[1] += w * v[u].y;
      acc[2] += w * v[u].z;
      acc[3] += w * v[u].w;
      lsum += w * ml[u].y;
    }
    j0 += kMergeBatch;
    if (j0 > b1) break;
#pragma unroll
    for (int u = 0; u < kMergeBatch; ++u) {
      if (j0 + u > b1) continue;
      const float* const rj = rec_of(j0 + u);
      ml[u] = *reinterpret_cast<const float2*>(rj + kHeadDim);
      v[u] = *reinterpret_cast<const float4*>(rj + 4 * part);
    }
  }
  const float f = lsum > 0.0f ? p.out_scale / lsum : 0.0f;
  h0 = __floats2bfloat162_rn(acc[0] * f, acc[1] * f);
  h1 = __floats2bfloat162_rn(acc[2] * f, acc[3] * f);
  }
  if constexpr (kGate) {
    norm_gate4<kNgVec>(h0, h1, graw, p.norm_eps);
  }
  uint2 w;
  w.x = *reinterpret_cast<uint32_t*>(&h0);
  w.y = *reinterpret_cast<uint32_t*>(&h1);
  *reinterpret_cast<uint2*>(orow) = w;
}

}  // namespace faw

// ---------------------------------------------------------------------------------------------- host
namespace faw_host {

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

// Tensor map over a pool [slots, row_bytes] (bytes), box = 128 bytes x rows, 128-byte swizzle.
inline CUtensorMap pool_map(const void* base, int64_t slots, int64_t row_bytes, int rows) {
  CUtensorMap map;
  const cuuint64_t dims[2] = {static_cast<cuuint64_t>(row_bytes), static_cast<cuuint64_t>(slots)};
  const cuuint64_t strides[1] = {static_cast<cuuint64_t>(row_bytes)};
  const cuuint32_t box[2] = {128, static_cast<cuuint32_t>(rows)};
  const cuuint32_t elem[2] = {1, 1};
  const CUresult r = encode_fn()(&map, CU_TENSOR_MAP_DATA_TYPE_UINT8, 2, const_cast<void*>(base), dims, strides, box,
                                 elem, CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B,
                                 CU_TENSOR_MAP_L2_PROMOTION_L2_256B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
  sglang::host::RuntimeCheck(r == CUDA_SUCCESS, "cuTensorMapEncodeTiled (pool) failed: ", static_cast<int>(r));
  return map;
}

// Tensor map over q [tokens, heads, 64] (bytes; token stride in bytes), box = 64 B x 8 heads x 8 tokens, 64-byte swizzle.
inline CUtensorMap q_map(const void* base, int64_t tokens, int64_t heads, int64_t token_stride) {
  CUtensorMap map;
  const cuuint64_t dims[3] = {64, static_cast<cuuint64_t>(heads), static_cast<cuuint64_t>(tokens)};
  const cuuint64_t strides[2] = {64, static_cast<cuuint64_t>(token_stride)};
  const cuuint32_t box[3] = {64, 8, 8};
  const cuuint32_t elem[3] = {1, 1, 1};
  const CUresult r = encode_fn()(&map, CU_TENSOR_MAP_DATA_TYPE_UINT8, 3, const_cast<void*>(base), dims, strides, box,
                                 elem, CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_64B,
                                 CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
  sglang::host::RuntimeCheck(r == CUDA_SUCCESS, "cuTensorMapEncodeTiled (q) failed: ", static_cast<int>(r));
  return map;
}

}  // namespace faw_host

template <int kPairs, int kStages, bool kUsePDL, int kPoly, int kPf = 0, int kMode = 0, int kTpr = 2,
          bool kGate = false, int kNgVec = 8, int kSkel = 0>
struct FullAttnWideKernel {
  static void run(const tvm::ffi::TensorView q,           // [R*8, Hq, 64] e4m3, heads/dims contiguous
                  const tvm::ffi::TensorView k_pool,      // [slots, Hkv*64] e4m3, rows contiguous
                  const tvm::ffi::TensorView v_pool,      // [slots, Hkv*64]
                  const tvm::ffi::TensorView page_table,  // [R, pages] int32, inner contiguous
                  const tvm::ffi::TensorView seq_lens,    // [R] int32
                  const tvm::ffi::TensorView out,         // [R*8, Hq, 64] bf16, contiguous
                  const tvm::ffi::TensorView partial,     // fp32 workspace, >= blocks * 2 * Hkv * 64 * 68
                  const tvm::ffi::TensorView counters,    // int32 [>= 256], zero
                  int64_t blocks, double bmm1_scale, double bmm2_scale,
                  const tvm::ffi::TensorView gate,  // kGate: [R*8, >= Hq*64] bf16, inner contiguous; else 1 element (unused)
                  double norm_eps) {
    using namespace sglang::host;
    using L = faw::Layout<kPairs, kStages, kTpr>;
    RuntimeCheck(q.dim() == 3 && q.size(2) == faw::kHeadDim && q.dtype().code == DLDataTypeCode::kDLFloat8_e4m3fn &&
                     q.stride(2) == 1 && q.stride(1) == faw::kHeadDim && q.stride(0) % 16 == 0,
                 "q must be e4m3 [tokens, heads, 64], heads and dims contiguous");
    RuntimeCheck(k_pool.dtype().code == DLDataTypeCode::kDLFloat8_e4m3fn &&
                     v_pool.dtype().code == DLDataTypeCode::kDLFloat8_e4m3fn && k_pool.dim() == 2 && v_pool.dim() == 2 &&
                     k_pool.stride(1) == 1 && v_pool.stride(1) == 1 && k_pool.stride(0) == k_pool.size(1) &&
                     v_pool.stride(0) == v_pool.size(1) && k_pool.size(1) == v_pool.size(1),
                 "pools must be contiguous e4m3 [slots, Hkv*64]");
    RuntimeCheck(out.dtype().code == DLDataTypeCode::kDLBfloat && out.dtype().bits == 16, "out must be bf16");
    const int64_t m = out.size(0), hq = out.size(1), reqs = m / faw::kTokens;
    const int64_t hkv = k_pool.size(1) / faw::kHeadDim;
    RuntimeCheck(m % faw::kTokens == 0 && hkv == L::kHeads && hkv * faw::kQPerKv == hq && q.size(0) == m &&
                     q.size(1) == hq,
                 "head layout (KV heads per rank must match the kernel)");
    RuntimeCheck(reqs <= faw::kMaxReqs, "too many requests");
    RuntimeCheck(out.dim() == 3 && out.size(2) == faw::kHeadDim && out.stride(2) == 1 && out.stride(1) == faw::kHeadDim &&
                     out.stride(0) == hq * faw::kHeadDim,
                 "out layout");
    RuntimeCheck(seq_lens.dim() == 1 && seq_lens.size(0) == reqs && seq_lens.stride(0) == 1 &&
                     seq_lens.dtype().code == DLDataTypeCode::kDLInt && seq_lens.dtype().bits == 32,
                 "seq_lens [R] int32");
    RuntimeCheck(page_table.dim() == 2 && page_table.size(0) >= reqs && page_table.stride(1) == 1 &&
                     page_table.dtype().code == DLDataTypeCode::kDLInt && page_table.dtype().bits == 32,
                 "page_table [R, pages] int32");
    static_assert(!kGate || kMode == 0 || kMode == 4 || kMode == 6, "the norm + gate variant needs the merge kernel");
    RuntimeCheck(blocks >= 1 && partial.dtype().code == DLDataTypeCode::kDLFloat && partial.dtype().bits == 32 &&
                     partial.numel() >= blocks * 2 * hkv * faw::kRows * faw::kRec,
                 "partial workspace too small");
    RuntimeCheck(kGate == (gate.numel() > 1), "norm + gate: the module variant (kGate) must match the gate argument");
    RuntimeCheck(counters.dtype().code == DLDataTypeCode::kDLInt && counters.dtype().bits == 32 &&
                     counters.numel() >= faw::kMaxReqs,
                 "counters int32 [>= 256]");
    const auto aligned16 = [](const void* ptr) { return reinterpret_cast<uintptr_t>(ptr) % 16 == 0; };
    RuntimeCheck(aligned16(q.data_ptr()) && aligned16(k_pool.data_ptr()) && aligned16(v_pool.data_ptr()) &&
                     aligned16(out.data_ptr()) && aligned16(partial.data_ptr()),
                 "16-byte alignment");

    faw::Params params{};
    params.page_table = static_cast<const int32_t*>(page_table.data_ptr());
    params.seq_lens = static_cast<const int32_t*>(seq_lens.data_ptr());
    params.partial = static_cast<float*>(partial.data_ptr());
    params.counters = static_cast<uint32_t*>(counters.data_ptr());
    params.out = static_cast<__nv_bfloat16*>(out.data_ptr());
    params.page_table_stride = page_table.stride(0);
    params.num_reqs = static_cast<int32_t>(reqs);
    params.num_q_heads = static_cast<int32_t>(hq);
    params.scale_log2 = static_cast<float>(bmm1_scale * M_LOG2E);
    params.out_scale = static_cast<float>(bmm2_scale);  // row sums are kept in x448 units
    if constexpr (kGate) {
      RuntimeCheck(gate.dtype().code == DLDataTypeCode::kDLBfloat && gate.dtype().bits == 16 && gate.dim() == 2 &&
                       gate.size(0) == m && gate.size(1) >= hq * faw::kHeadDim && gate.stride(1) == 1 &&
                       gate.stride(0) % 4 == 0 && reinterpret_cast<uintptr_t>(gate.data_ptr()) % 8 == 0,
                   "gate must be bf16 [tokens, >= Hq*64], inner contiguous, 8-byte aligned rows");
      params.gate = static_cast<const __nv_bfloat16*>(gate.data_ptr());
      params.gate_token_stride = gate.stride(0);
      params.norm_eps = static_cast<float>(norm_eps);
    }
    const CUtensorMap qm = faw_host::q_map(q.data_ptr(), m, hq, q.stride(0));
    const CUtensorMap km = faw_host::pool_map(k_pool.data_ptr(), k_pool.size(0), k_pool.size(1), L::kTileKeys);
    const CUtensorMap vm = faw_host::pool_map(v_pool.data_ptr(), v_pool.size(0), v_pool.size(1), L::kTileKeys);

    static_assert(kPoly >= 0 && kPoly <= 8 && kPoly % 2 == 0, "kPoly: even, 0..8");
    constexpr auto kernel = faw::full_attn_wide_kernel<kPairs, kStages, kUsePDL, kPoly, kPf, kMode, kTpr, kSkel>;
    static bool configured = false;
    if (!configured) {
      RuntimeDeviceCheck(::cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, L::kSmBytes));
      configured = true;
    }
    if (reqs == 0) return;
    LaunchKernel(dim3(static_cast<unsigned>(blocks)), L::kThreads, out.device(), L::kSmBytes)
        .config({.use_pdl = kUsePDL})(kernel, params, qm, km, vm);
    if constexpr (kMode == 0 || kMode == 4 || kMode == 6) {
      constexpr auto merge = faw::full_attn_wide_merge_kernel<kPairs, kUsePDL, kGate, kNgVec>;
      LaunchKernel(dim3(static_cast<unsigned>(L::kHeads * faw::kRows / faw::kMergeRows), static_cast<unsigned>(reqs)),
                   256, out.device(), 0)
          .config({.use_pdl = kUsePDL})(merge, params, static_cast<int>(blocks));
    }
  }
};

}  // namespace muse_vr
