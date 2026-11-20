// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
/**
 * \brief Paged full-attention verify for many requests per step over a native NVFP4 KV cache (packed E2M1 values with
 * one E4M3 scale per 16 values): the persistent kernel of full_attn_wide.cuh with its K/V tiles produced in the kernel
 * from the NVFP4 pools. Thread blocks split the summed key range of all requests evenly (one block per SM, one wave),
 * read whole token rows of the packed pools and their block scales (every KV head of the rank) into a ring of raw
 * stages, turn each raw stage into the FP8 (e4m3) tile layout that full_attn_wide's tensor-memory-accelerator loads
 * produce, and run the attention on tensor-core MMA (tcgen05) exactly as full_attn_wide does; a second kernel merges the
 * pieces of requests that several blocks share.
 *
 * Shapes (per call): q [R * 8, Hq, 64] e4m3 (heads and dims contiguous, any token stride that is a multiple of 16 B);
 * K/V pools [slots, Hkv * 32] uint8 (token-major, slot = page * 128 + offset; two E2M1 values per byte, element 2i in the
 * low nibble of byte i); K/V scale pools [pages, Hkv, 128, 4] E4M3 (V scales four-token interleaved inside a page: the
 * scale of token t, block i sits at token (t & ~3) + i, column t & 3); page_table [R, >= pages] int32; seq_lens [R] int32
 * (KV length including the 8 verify tokens); out [R * 8, Hq, 64] bf16, contiguous. Hq = 8 Hkv, Hkv = 2 kPairs (kPairs = 1
 * or 2: 2 or 4 KV heads per rank). Token t (0..7) of request r sits at position p = seq_len - 8 + t and attends keys
 * [0, p].
 *
 * Dequantization: every K/V value becomes e4m3(f16(e2m1) * f16(scale)) (the product is exact in f16; one rounding to
 * e4m3, round to nearest even, satfinite) - the arithmetic of the plugin's NVFP4 verify kernel and of the prefill
 * dequantization. The kernel evaluates it through a table: for each of the 256 scale bytes, the 8 products of the E2M1
 * magnitudes are computed once per block with that same conversion chain (a table of 256 x 8 bytes); a packed value is then
 * looked up by its magnitude (byte permute) and its sign bit is applied to the e4m3 sign bit (the conversion is
 * sign-symmetric). The tile handed to the MMA is therefore byte-identical to the one full_attn_wide loads from an FP8 pool
 * that holds the dequantized values, and everything after it is full_attn_wide's arithmetic. One difference in data
 * movement only, with 4 KV heads: the e4m3 probabilities reach the PV MMA through tensor memory (each softmax thread
 * writes its row over its own scores of the tile; the MMA takes its A operand from tensor memory) instead of shared
 * memory - the same bytes in the same operand positions.
 *
 * Work split (full_attn_wide's). A request has n_r = ceil(seq_len / kTileKeys) tiles of kTileKeys keys (kTileKeys =
 * 128 / kPairs, so an FP8 K or V tile is always 16 KB: kTileKeys token rows of Hkv * 64 bytes; the packed tile is 8 KB).
 * With W = sum n_r and G = min(blocks, W), block b takes the tiles [floor(b W / G), floor((b + 1) W / G)) of the
 * concatenation of all requests (request-major). The part of a request inside one block is a piece; a request that lies
 * wholly inside one block is normalized and written by it; otherwise each piece writes a partial record (unnormalized
 * output, row maximum, row sum per query row) and kv4_attn_wide_merge_kernel (launched next, PDL-chained; the merge of
 * full_attn_wide) merges the request's records in block order.
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
 * ex2.approx.ftz.f32. (All of this is full_attn_wide's arithmetic; the global K/V scales are folded into bmm1 / bmm2 by
 * the caller, as for the stock NVFP4 path.)
 *
 * Block roles (16 warps, four warpgroups): warps 0-7 softmax (kTpr threads per query row as in full_attn_wide; with two,
 * the two threads of a row exchange tile maxima through shared memory and a named barrier among the softmax warps);
 * warp 8 load (lane 0: per piece the q rows of the request with the tensor memory accelerator (one box of 8 tokens x 8
 * heads x 64 B per KV head, 64-byte swizzle) into the Q buffer; per tile the packed K and V rows (kTileKeys rows x
 * Hkv * 32 B) and the tile's K and V scale rows (Hkv x kTileKeys x 4 B, as 128-byte rows in the 128-byte swizzle) into
 * a ring of kRaw raw stages); warp 9 QK MMA and warp 10 PV MMA (the whole warp runs the loop with warp-uniform
 * operands, one elected lane issues: kind::f8f6f4, M = 64 rows per KV head, the two heads of a pair at tensor-memory lane
 * offsets 0 and 16 of every 32-lane group); warp 11 idle; warps 12-15 dequantization (raw stage -> one of kF8 FP8
 * stages [pair][kTileKeys rows][128 B] in the 128-byte swizzle; each thread one 16-byte part of four consecutive rows,
 * steps ordered to keep its 128-bit shared-memory accesses free of bank conflicts, profiler-checked). Warpgroups 0-1
 * raise their register limit and 2-3 lower theirs (setmaxnreg, one instruction heading each side's branch).
 * Hand-offs per tile g: raw stage landed (complete_tx) -> dequantization of the K and V halves -> one
 * fence.proxy.async.shared::cta per thread, then the K half full, V half full and raw stage free arrivals (128 thread
 * arrivals each) -> QK(g) (waits for the K half) into score buffer g % 2 (issued two tiles ahead, once the softmax warps
 * hold tile g - 2's scores in registers and, with 4 KV heads, the PV of tile g - 2 has read its probabilities) ->
 * softmax -> probabilities in tensor memory over the tile's scores (4 KV heads) or in shared memory (2 KV heads, with
 * fence.proxy.async.shared::cta before the arrival) -> PV(g) (waits for the probabilities and the V half) into output
 * buffer g % 2 (its commit frees the FP8 stage) -> added into the registers of the softmax threads during tile g + 1.
 * Development probe modes (template argument kDq != 0) exist for profiling only; kDq = 0 is the kernel.
 *
 * Ordering: q, seq_lens and the page table may be written by the preceding kernels; they are read only after
 * griddepcontrol.wait (use_pdl), followed by fence.proxy.async.global before the first tensor-memory-accelerator read.
 * A raw stage is read with generic shared-memory loads and refilled by the tensor memory accelerator: each reading thread
 * executes fence.proxy.async.shared::cta after its last read of the stage and before its arrival on the stage's free
 * barrier. An FP8 stage is written with generic shared-memory stores and read by the MMA (async proxy): the same fence
 * precedes the arrival the MMA warp waits on; the stage is rewritten only after the PV commit of its tile. The
 * probabilities in tensor memory (4 KV heads): tcgen05.st, tcgen05.wait::st, tcgen05.fence::before_thread_sync, then
 * the arrival the PV warp waits on (it executes tcgen05.fence::after_thread_sync before its MMAs); their tensor-memory
 * columns are rewritten by the QK MMA of tile g + 2 only after the PV commit of tile g (the QK warp waits for it). Pool
 * rows past the end of a request inside its last page are loaded and masked out of the probabilities (their values are
 * finite: the pools are allocated zero-filled and only finite values are stored into them). Every mbarrier wait is
 * bounded (trap after a fixed number of polls).
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

namespace k4w {

constexpr int kHeadDim = 64;
constexpr int kQPerKv = 8;
constexpr int kTokens = 8;
constexpr int kRows = 64;  // query rows per KV head: token * 8 + head
constexpr int kRec = 68;  // floats per partial row record: 64 outputs, m, l, 2 pad
constexpr float kPScale = 448.0f;
constexpr float kLog2PScale = 8.807354922057604f;  // log2(448)
constexpr int kMaxReqs = 256;
constexpr int kMaxPages = 1024;  // page numbers of a block's range held in shared memory (more: read from global)
constexpr int kTileBytes = 16384;  // one FP8 K or V tile
constexpr uint32_t kTmemCols = 512;  // S0 [0,128) S1 [128,256) O0 [256,384) O1 [384,512)
constexpr int kPage = 128;
constexpr int kLutBytes = 256 * 8;  // dequantization table: 8 e4m3 products per scale byte
// warp roles
// dequantization warps: 4 (the kernel) or 8 (development build, -DK4W_DQ_WARPS=8: 20 warps, the other warpgroups'
// register limit lowered to fit)
#ifndef K4W_DQ_WARPS
#define K4W_DQ_WARPS 4
#endif
inline constexpr int kDqWarps = K4W_DQ_WARPS;
static_assert(kDqWarps == 4 || kDqWarps == 8, "4 or 8 dequantization warps");
constexpr int kNumWarps = 12 + kDqWarps;
constexpr int kThreads = 32 * kNumWarps;
constexpr int kLoadWarp = 8, kQkWarp = 9, kPvWarp = 10, kDqWarp0 = 12;
constexpr int kDqThreads = 32 * (kNumWarps - kDqWarp0);
// per-thread register limits after the prologue (setmaxnreg): softmax warpgroups kSmRegs, the others kAuxRegs; their
// sum must fit the block's launch allocation (kThreads x kLaunchRegs, checked on the host against the compiled count)
constexpr int kSmRegs = 168, kAuxRegs = kDqWarps == 4 ? 88 : 48;
constexpr int kLaunchRegs = (65536 / kThreads) / 8 * 8;
static_assert(8 * 32 * kSmRegs + (kNumWarps - 8) * 32 * kAuxRegs <= kThreads * kLaunchRegs, "register file");
// skeleton option bit 2: the load / MMA / spare warpgroup lowers its limit to kWg2Regs and the dequantization warpgroup to
// kDqRegs (4 dequantization warps; kWg2Regs8 / kDqRegs8 with 8), so the dequantization warps keep a whole tile's loads in
// registers
constexpr int kWg2Regs = 56, kDqRegs = 120;
static_assert(8 * 32 * kSmRegs + 4 * 32 * kWg2Regs + 4 * 32 * kDqRegs <= 512 * ((65536 / 512) / 8 * 8),
              "register file (skeleton option bit 2)");
constexpr int kWg2Regs8 = 32, kDqRegs8 = 56;
static_assert(8 * 32 * kSmRegs + 4 * 32 * kWg2Regs8 + 8 * 32 * kDqRegs8 <= 640 * ((65536 / 640) / 8 * 8),
              "register file (skeleton option bit 2, 8 dequantization warps)");
constexpr int kWg2RegsR = kDqWarps == 4 ? kWg2Regs : kWg2Regs8;
constexpr int kDqRegsR = kDqWarps == 4 ? kDqRegs : kDqRegs8;
// setmaxnreg runs per warpgroup (PTX ISA setmaxnreg): every role range starts and ends on a warpgroup boundary, and every
// lowered limit is at most / every raised limit at least the launch allocation (also checked on the host against the
// compiled count)
static_assert(kLoadWarp % 4 == 0 && kDqWarp0 % 4 == 0 && kDqWarps % 4 == 0 && kNumWarps % 4 == 0, "warpgroup-aligned roles");
static_assert(kAuxRegs <= kLaunchRegs && kWg2RegsR <= kLaunchRegs && kDqRegsR <= kLaunchRegs && kSmRegs >= kLaunchRegs,
              "setmaxnreg: lowered limits <= launch allocation <= raised limits");
// skeleton option bit 5 (with bit 2, 4 KV heads, one softmax thread per row, probabilities in tensor memory): a fifth
// warpgroup (warps 16-19, one per tensor-memory lane quarter) does the per-tile output rescale-and-add into an
// accumulator in tensor memory with the softmax's fp32 operations; the softmax warps keep no output registers between
// tiles. Register limits: softmax kSmRegsCr, load / MMA / spare kWg2Regs, dequantization kDqRegsCr, rescale kCrRegs.
constexpr int kCrWarp0 = 16;
constexpr int kSmRegsCr = 112, kDqRegsCr = 112, kCrRegs = 88;
constexpr int kLaunchRegsCr = (65536 / 640) / 8 * 8;
static_assert(8 * 32 * kSmRegsCr + 4 * 32 * kWg2Regs + 4 * 32 * kDqRegsCr + 4 * 32 * kCrRegs <= 640 * kLaunchRegsCr,
              "register file (skeleton option bit 5)");
static_assert(kWg2Regs <= kLaunchRegsCr && kCrRegs <= kLaunchRegsCr && kDqRegsCr >= kLaunchRegsCr && kSmRegsCr >= kLaunchRegsCr,
              "setmaxnreg (skeleton option bit 5): lowered limits <= launch allocation <= raised limits");
template <int kSkel>
constexpr int threads_of() {
  return (kSkel & 32) != 0 ? 640 : kThreads;
}

template <int kPairs, int kF8, int kRaw, int kTprT, int kPSm = 1>
struct Layout {
  static constexpr int kHeads = 2 * kPairs;
  static constexpr int kTileKeys = 128 / kPairs;
  static constexpr int kTpr = kTprT;  // softmax threads per query row (2: each half of a tile's keys and of the dims)
  static_assert(kTpr == 2 || (kTpr == 1 && kPairs == 2), "one thread per row only with 64-key tiles");
  static constexpr int kSmWarps = 4 * kPairs * kTpr;  // softmax warps: kTpr threads x (kHeads x 64 rows)
  static_assert(kSmWarps == 8, "two warpgroups of softmax warps");
  static constexpr int kKpt = kTileKeys / kTpr;  // keys of a tile per softmax thread (32 or 64)
  static constexpr int kPRowBytes = kTileKeys;  // e4m3 probabilities of one row of a tile
  // raw stage: packed K rows, packed V rows ([kTileKeys][Hkv * 32 B] each), K scales, V scales ([Hkv][kTileKeys][4])
  static constexpr int kRawData = kTileKeys * kHeads * 32;  // 8 KB
  static constexpr int kRawScale = kHeads * kTileKeys * 4;   // 1 KB
  static constexpr int kRawBytes = 2 * kRawData + 2 * kRawScale;
  static constexpr int kChunks = kRawData / 16;  // 16-byte packed chunks of a K (or V) tile
  static_assert(kChunks % kDqThreads == 0, "chunks per dequantization thread");
  static constexpr int kSmK = 0;
  static constexpr int kSmV = kSmK + kF8 * kTileBytes;
  static constexpr int kSmP = kSmV + kF8 * kTileBytes;               // [2 buffers][heads][64 rows][kPRowBytes]
  static constexpr int kPBuf = kHeads * kRows * kPRowBytes;          // 16 KB
  static constexpr int kSmQ = kSmP + 2 * kPBuf * kPSm;  // [heads][64 rows][64 B] (no P buffers: kPSm 0)
  static constexpr int kQBuf = kHeads * kRows * 64;
  static constexpr int kQBufs = 1;
  static constexpr int kSmRaw = kSmQ + kQBufs * kQBuf;               // [kRaw][kRawBytes]
  static constexpr int kSmLut = kSmRaw + kRaw * kRawBytes;           // [256][8] e4m3
  static constexpr int kSmRed = kSmLut + kLutBytes;                  // [3][heads * 64 rows][2] fp32
  static constexpr int kSmPre = kSmRed + 3 * kHeads * kRows * 2 * 4;  // prefix tile counts [kMaxReqs + 1]
  static constexpr int kSmPages = kSmPre + (kMaxReqs + 1) * 4 + 12;  // page numbers of the block's range
  static constexpr int kSmBar = (kSmPages + kMaxPages * 4 + 15) / 16 * 16;
  static constexpr int kNumBars = 3 * kF8 + 2 * kRaw + 12;
  static_assert(8 * kNumBars + 16 <= 512, "barrier region");
  static constexpr int kSmBytes = kSmBar + 512 + 1024;
  static_assert(kSmRaw % 1024 == 0 && kRawBytes % 1024 == 0 && kRawData % 1024 == 0, "raw stage alignment");
  static_assert(kSmLut % 8 == 0 && kLutBytes == 256 * 8, "8-byte table entries at an 8-byte aligned base (option bit 7 loads)");
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
// (sanitizer builds lower it with -DK4W_MAX_POLLS so an instrumented run that stalls traps instead of outliving its
// process timeout)
#ifndef K4W_MAX_POLLS
#define K4W_MAX_POLLS (1u << 24)
#endif
inline constexpr uint32_t kMaxPolls = K4W_MAX_POLLS;
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
// Wait used where every thread of a warp waits and the warp's next tensor-memory load (.sync.aligned) or other warp-wide
// operation needs the warp converged: each thread exits the spin on its own, so the warp reconverges before going on
// (PTX ISA: .aligned instructions in conditionally executed code need every thread of the warp to take the same path).
__device__ __forceinline__ void mbar_wait_warp(uint32_t bar, uint32_t parity) {
  mbar_wait(bar, parity);
  __syncwarp();
}
// Wait with a suspend-time hint (ns): the thread may be suspended for up to the hint while the phase is incomplete and
// resumes when it completes (PTX ISA mbarrier.try_wait); for warps whose waits are off the softmax's critical path, so
// they do not take issue slots from the warps that have work. Bounded like mbar_wait.
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
        "K4W_HWAIT_%=:\n"
        "mbarrier.try_wait.parity.shared::cta.b64 p, [%0], %1, %3;\n"
        "@p bra K4W_HDONE_%=;\n"
        "add.u32 n, n, 1;\n"
        "setp.lt.u32 p, n, %2;\n"
        "@p bra K4W_HWAIT_%=;\n"
        "trap;\n"
        "K4W_HDONE_%=:\n"
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
__device__ __forceinline__ void tma_load_4d(uint32_t dst, const CUtensorMap* map, int32_t c0, int32_t c1, int32_t c2,
                                            int32_t c3, uint32_t bar) {
  asm volatile(
      "cp.async.bulk.tensor.4d.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1, {%2, %3, %4, %5}], [%6];\n" ::"r"(
          dst),
      "l"(reinterpret_cast<uint64_t>(map)), "r"(c0), "r"(c1), "r"(c2), "r"(c3), "r"(bar)
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
// A operand from tensor memory (PTX ISA tcgen05.mma: A in tensor memory or shared memory, B in shared memory).
__device__ __forceinline__ void mma_f8_ts_e(uint32_t d, uint32_t a_tmem, uint64_t b, uint32_t idesc,
                                            uint32_t accumulate) {
  asm volatile(
      "{\n"
      ".reg .pred e, p;\n"
      "elect.sync _|e, 0xffffffff;\n"
      "setp.ne.b32 p, %4, 0;\n"
      "@e tcgen05.mma.cta_group::1.kind::f8f6f4 [%0], [%1], %2, %3, p;\n"
      "}\n" ::"r"(d),
      "r"(a_tmem), "l"(b), "r"(idesc), "r"(accumulate)
      : "memory");
}
// Block-scaled form (PTX ISA tcgen05.mma .kind::mxf8f6f4 .block_scale): A from tensor memory, B in shared memory, the
// UE8M0 scale factors of A and B in tensor memory (one per 32 values of K per row of A / column of B).
__device__ __forceinline__ void mma_bs_ts_e(uint32_t d, uint32_t a_tmem, uint64_t b, uint32_t idesc, uint32_t sfa,
                                            uint32_t sfb, uint32_t accumulate) {
  asm volatile(
      "{\n"
      ".reg .pred e, p;\n"
      "elect.sync _|e, 0xffffffff;\n"
      "setp.ne.b32 p, %6, 0;\n"
      "@e tcgen05.mma.cta_group::1.kind::mxf8f6f4.block_scale [%0], [%1], %2, %3, [%4], [%5], p;\n"
      "}\n" ::"r"(d),
      "r"(a_tmem), "l"(b), "r"(idesc), "r"(sfa), "r"(sfb), "r"(accumulate)
      : "memory");
}
// The same with two scale factors per row of A / column of B (scale_vec::2X: dense K = 64, one per 32 values of K; the
// scale-factor IDs select a 16-bit half of the scale-factor words).
__device__ __forceinline__ void mma_bs2_ts_e(uint32_t d, uint32_t a_tmem, uint64_t b, uint32_t idesc, uint32_t sfa,
                                             uint32_t sfb, uint32_t accumulate) {
  asm volatile(
      "{\n"
      ".reg .pred e, p;\n"
      "elect.sync _|e, 0xffffffff;\n"
      "setp.ne.b32 p, %6, 0;\n"
      "@e tcgen05.mma.cta_group::1.kind::mxf8f6f4.block_scale.scale_vec::2X [%0], [%1], %2, %3, [%4], [%5], p;\n"
      "}\n" ::"r"(d),
      "r"(a_tmem), "l"(b), "r"(idesc), "r"(sfa), "r"(sfb), "r"(accumulate)
      : "memory");
}
// Block-scaled FP4 form (PTX ISA tcgen05.mma .kind::mxf4nvf4 .block_scale .block16): A (packed E2M1) from tensor memory,
// B (packed E2M1, K-major) in shared memory, their UE4M3 scale factors (one per 16 values of K) in tensor memory.
__device__ __forceinline__ void mma_nvf4_ts_e(uint32_t d, uint32_t a_tmem, uint64_t b, uint32_t idesc, uint32_t sfa,
                                              uint32_t sfb, uint32_t accumulate) {
  asm volatile(
      "{\n"
      ".reg .pred e, p;\n"
      "elect.sync _|e, 0xffffffff;\n"
      "setp.ne.b32 p, %6, 0;\n"
      "@e tcgen05.mma.cta_group::1.kind::mxf4nvf4.block_scale.block16 [%0], [%1], %2, %3, [%4], [%5], p;\n"
      "}\n" ::"r"(d),
      "r"(a_tmem), "l"(b), "r"(idesc), "r"(sfa), "r"(sfb), "r"(accumulate)
      : "memory");
}
__device__ __forceinline__ void tmem_st2(uint32_t taddr, uint32_t w0, uint32_t w1) {
  asm volatile("tcgen05.st.sync.aligned.32x32b.x2.b32 [%0], {%1,%2};\n" ::"r"(taddr), "r"(w0), "r"(w1) : "memory");
}
__device__ __forceinline__ void tmem_st8(uint32_t taddr, const uint32_t (&w)[8]) {
  asm volatile("tcgen05.st.sync.aligned.32x32b.x8.b32 [%0], {%1,%2,%3,%4,%5,%6,%7,%8};\n" ::"r"(taddr), "r"(w[0]),
               "r"(w[1]), "r"(w[2]), "r"(w[3]), "r"(w[4]), "r"(w[5]), "r"(w[6]), "r"(w[7])
               : "memory");
}
// one / four consecutive 32-bit columns of this thread's tensor-memory lane
__device__ __forceinline__ void tmem_st1(uint32_t taddr, uint32_t w) {
  asm volatile("tcgen05.st.sync.aligned.32x32b.x1.b32 [%0], {%1};\n" ::"r"(taddr), "r"(w) : "memory");
}
__device__ __forceinline__ void tmem_st4(uint32_t taddr, uint32_t w0, uint32_t w1, uint32_t w2, uint32_t w3) {
  asm volatile("tcgen05.st.sync.aligned.32x32b.x4.b32 [%0], {%1,%2,%3,%4};\n" ::"r"(taddr), "r"(w0), "r"(w1), "r"(w2),
               "r"(w3)
               : "memory");
}
// 16 consecutive 32-bit columns of this thread's tensor-memory lane
__device__ __forceinline__ void tmem_st16(uint32_t taddr, const uint32_t (&w)[16]) {
  asm volatile(
      "tcgen05.st.sync.aligned.32x32b.x16.b32 [%0], {%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16};\n" ::"r"(
          taddr),
      "r"(w[0]), "r"(w[1]), "r"(w[2]), "r"(w[3]), "r"(w[4]), "r"(w[5]), "r"(w[6]), "r"(w[7]), "r"(w[8]), "r"(w[9]),
      "r"(w[10]), "r"(w[11]), "r"(w[12]), "r"(w[13]), "r"(w[14]), "r"(w[15])
      : "memory");
}
__device__ __forceinline__ void tmem_wait_st() { asm volatile("tcgen05.wait::st.sync.aligned;\n" ::: "memory"); }
// 32 consecutive 32-bit columns of this thread's tensor-memory lane
__device__ __forceinline__ void tmem_st32(uint32_t taddr, const uint32_t (&w)[32]) {
  asm volatile(
      "tcgen05.st.sync.aligned.32x32b.x32.b32 [%0], {%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17,%18,%19,"
      "%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31,%32};\n" ::"r"(taddr),
      "r"(w[0]), "r"(w[1]), "r"(w[2]), "r"(w[3]), "r"(w[4]), "r"(w[5]), "r"(w[6]), "r"(w[7]), "r"(w[8]), "r"(w[9]),
      "r"(w[10]), "r"(w[11]), "r"(w[12]), "r"(w[13]), "r"(w[14]), "r"(w[15]), "r"(w[16]), "r"(w[17]), "r"(w[18]),
      "r"(w[19]), "r"(w[20]), "r"(w[21]), "r"(w[22]), "r"(w[23]), "r"(w[24]), "r"(w[25]), "r"(w[26]), "r"(w[27]),
      "r"(w[28]), "r"(w[29]), "r"(w[30]), "r"(w[31])
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
// one / 16 consecutive 32-bit columns of this thread's tensor-memory lane
__device__ __forceinline__ uint32_t tmem_ld1(uint32_t taddr) {
  uint32_t r;
  asm volatile("tcgen05.ld.sync.aligned.32x32b.x1.b32 {%0}, [%1];\n" : "=r"(r) : "r"(taddr));
  return r;
}
__device__ __forceinline__ void tmem_ld16(uint32_t taddr, uint32_t (&r)[16]) {
  asm volatile(
      "tcgen05.ld.sync.aligned.32x32b.x16.b32 {%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15}, [%16];\n"
      : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]), "=r"(r[4]), "=r"(r[5]), "=r"(r[6]), "=r"(r[7]), "=r"(r[8]),
        "=r"(r[9]), "=r"(r[10]), "=r"(r[11]), "=r"(r[12]), "=r"(r[13]), "=r"(r[14]), "=r"(r[15])
      : "r"(taddr));
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
// 32 consecutive fp32 columns of this thread's tensor-memory lane and their maximum (tcgen05.ld with .red .max, PTX ISA
// 8.8: the reduction of the loaded columns per lane, .NaN not set); the registers and the maximum are read only after
// tcgen05.wait::ld, as for the plain load
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
__device__ __forceinline__ void tmem_wait_ld() { asm volatile("tcgen05.wait::ld.sync.aligned;\n" ::: "memory"); }

// Shared-memory matrix descriptor: start >> 4 [0,14), leading byte offset >> 4 [16,30), stride byte offset >> 4
// [32,46), descriptor version 1 [46,48), swizzle mode [61,64) (PTX ISA, tcgen05 shared memory descriptor).
__device__ __forceinline__ uint64_t sdesc(uint32_t saddr, uint32_t sbo, uint32_t layout) {
  return static_cast<uint64_t>((saddr >> 4) & 0x3FFF) | (static_cast<uint64_t>(1) << 16) |
         (static_cast<uint64_t>((sbo >> 4) & 0x3FFF) << 32) | (static_cast<uint64_t>(1) << 46) |
         (static_cast<uint64_t>(layout & 7) << 61);
}
constexpr uint32_t kSw128 = 2, kSw64 = 4;
// Instruction descriptor, kind::f8f6f4, e4m3 x e4m3 -> fp32 (PTX ISA, tcgen05 instruction descriptor).
// k64: the descriptor's K-dimension bit (dense K = 64 instead of 32; PTX ISA 9.4, instruction descriptor for
// .kind::f8f6f4, on the targets that list it).
constexpr uint32_t idesc_f8(int m, int n, int b_mn_major, bool k64 = false) {
  return (1u << 4) | (static_cast<uint32_t>(b_mn_major) << 16) | (static_cast<uint32_t>(n >> 3) << 17) |
         (static_cast<uint32_t>(m >> 4) << 24) | (k64 ? (1u << 29) : 0u);
}
// Instruction descriptor, kind::mxf8f6f4 block-scaled, e4m3 x e4m3 -> fp32, UE8M0 scales, scale factor A in the 128-lane
// layout, dense K = 32; a_sf_id selects the byte of the scale-factor A word (PTX ISA 9.4, instruction descriptor for
// .kind::mxf8f6f4), B's scale-factor ID 0. k64: dense K = 64 (the K-dimension bit; scale_vec::2X, a_sf_id 0 or 2
// selects the 16-bit half).
constexpr uint32_t idesc_bs_f8(int m, int n, int b_mn_major, int a_sf_id = 0, bool k64 = false) {
  return (static_cast<uint32_t>(b_mn_major) << 16) | (static_cast<uint32_t>(n >> 3) << 17) | (1u << 23) | (1u << 26) |
         (static_cast<uint32_t>(m >> 7) << 27) | (static_cast<uint32_t>(a_sf_id & 3) << 29) | (k64 ? (1u << 31) : 0u);
}
// Instruction descriptor, kind::mxf4nvf4 block16: E2M1 x E2M1 -> fp32, UE4M3 scales, dense K = 64, both K-major,
// scale factor A in the 128-lane layout, sparsity-version field 1 (PTX ISA 9.4: required value on this target),
// scale-factor IDs 0.
constexpr uint32_t idesc_nvf4(int m, int n) {
  return (1u << 7) | (1u << 10) | (1u << 12) | (static_cast<uint32_t>(n >> 3) << 17) | (1u << 26) |
         (static_cast<uint32_t>(m >> 7) << 27);
}
// Shared-memory matrix descriptor with an explicit leading byte offset (no-swizzle K-major layouts: the offset between
// the two 16-byte columns of the 8-row core matrix pair; PTX ISA tcgen05 matrix descriptors).
__device__ __forceinline__ uint64_t sdesc_lbo(uint32_t saddr, uint32_t lbo, uint32_t sbo, uint32_t layout) {
  return static_cast<uint64_t>((saddr >> 4) & 0x3FFF) | (static_cast<uint64_t>((lbo >> 4) & 0x3FFF) << 16) |
         (static_cast<uint64_t>((sbo >> 4) & 0x3FFF) << 32) | (static_cast<uint64_t>(1) << 46) |
         (static_cast<uint64_t>(layout & 7) << 61);
}

// Narrow-format conversions (PTX ISA cvt): an f16x2 word to two floats (low half first); four e4m3 bytes (byte 0 first)
// to floats; a float to its e4m3 byte (round to nearest even, satfinite); eight floats to eight E2M1 nibbles (value 0 in
// bits 0-3; round to nearest even, satfinite) and the nibbles back to floats (exact).
__device__ __forceinline__ float2 h2_to_f2(uint32_t h) {
  float a, b;
  asm("{\n.reg .b16 lo, hi;\nmov.b32 {lo, hi}, %2;\ncvt.f32.f16 %0, lo;\ncvt.f32.f16 %1, hi;\n}\n" : "=f"(a), "=f"(b) : "r"(h));
  return make_float2(a, b);
}
__device__ __forceinline__ void e4m3x4_to_f(uint32_t w, float* o) {
  uint32_t h0, h1;
  asm("{\n.reg .b16 lo, hi;\nmov.b32 {lo, hi}, %2;\ncvt.rn.f16x2.e4m3x2 %0, lo;\ncvt.rn.f16x2.e4m3x2 %1, hi;\n}\n"
      : "=r"(h0), "=r"(h1)
      : "r"(w));
  const float2 a = h2_to_f2(h0), b = h2_to_f2(h1);
  o[0] = a.x;
  o[1] = a.y;
  o[2] = b.x;
  o[3] = b.y;
}
__device__ __forceinline__ uint32_t f_to_e4m3(float x) {
  uint16_t h;
  asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;\n" : "=h"(h) : "f"(0.0f), "f"(x));
  return h & 0xFFu;
}
__device__ __forceinline__ float e4m3_to_f(uint32_t b) {
  float o[4];
  e4m3x4_to_f(b & 0xFFu, o);
  return o[0];
}
__device__ __forceinline__ uint32_t f8_to_e2m1x8(const float* v) {
  uint32_t w;
  asm("{\n.reg .b8 b0, b1, b2, b3;\n"
      "cvt.rn.satfinite.e2m1x2.f32 b0, %2, %1;\n"
      "cvt.rn.satfinite.e2m1x2.f32 b1, %4, %3;\n"
      "cvt.rn.satfinite.e2m1x2.f32 b2, %6, %5;\n"
      "cvt.rn.satfinite.e2m1x2.f32 b3, %8, %7;\n"
      "mov.b32 %0, {b0, b1, b2, b3};\n}\n"
      : "=r"(w)
      : "f"(v[0]), "f"(v[1]), "f"(v[2]), "f"(v[3]), "f"(v[4]), "f"(v[5]), "f"(v[6]), "f"(v[7]));
  return w;
}
__device__ __forceinline__ void e2m1x8_to_f(uint32_t w, float* o) {
  uint32_t h0, h1, h2, h3;
  asm("{\n.reg .b8 b0, b1, b2, b3;\nmov.b32 {b0, b1, b2, b3}, %4;\n"
      "cvt.rn.f16x2.e2m1x2 %0, b0;\ncvt.rn.f16x2.e2m1x2 %1, b1;\ncvt.rn.f16x2.e2m1x2 %2, b2;\ncvt.rn.f16x2.e2m1x2 %3, b3;\n}\n"
      : "=r"(h0), "=r"(h1), "=r"(h2), "=r"(h3)
      : "r"(w));
  const uint32_t h[4] = {h0, h1, h2, h3};
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    const float2 f = h2_to_f2(h[i]);
    o[2 * i] = f.x;
    o[2 * i + 1] = f.y;
  }
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
// 2^y of a pair in half precision (PTX ISA ex2.approx.f16x2): y rounded to f16x2 (value 0 in the low half), the f16x2
// result; and four such values (two f16x2 words, value 0 in the low half of the first) as four e4m3 bytes (key order).
__device__ __forceinline__ uint32_t exp2_f16x2(float2 y) {
  uint32_t h, r;
  asm("cvt.rn.f16x2.f32 %0, %1, %2;\n" : "=r"(h) : "f"(y.y), "f"(y.x));
  asm("ex2.approx.f16x2 %0, %1;\n" : "=r"(r) : "r"(h));
  return r;
}
__device__ __forceinline__ uint32_t e4m3x4_from_f16x2(uint32_t h01, uint32_t h23) {
  uint32_t w;
  asm("{\n.reg .b16 lo, hi;\ncvt.rn.satfinite.e4m3x2.f16x2 lo, %1;\ncvt.rn.satfinite.e4m3x2.f16x2 hi, %2;\n"
      "mov.b32 %0, {lo, hi};\n}\n"
      : "=r"(w)
      : "r"(h01), "r"(h23));
  return w;
}
__device__ __forceinline__ uint32_t e4m3x4(float a, float b, float c, float d) {
  uint16_t lo, hi;
  asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;\n" : "=h"(lo) : "f"(b), "f"(a));
  asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;\n" : "=h"(hi) : "f"(d), "f"(c));
  return static_cast<uint32_t>(lo) | (static_cast<uint32_t>(hi) << 16);
}
// The same four conversions and byte positions, the two halves written into one 32-bit register inside one statement
// (the compiler then needs no separate byte permute to join them).
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

// ---------------------------------------------------------------------------------------------- NVFP4 dequantization
__device__ __forceinline__ uint4 lds128(uint32_t a) {
  uint4 v;
  asm volatile("ld.shared.v4.u32 {%0, %1, %2, %3}, [%4];\n" : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w) : "r"(a)
               : "memory");
  return v;
}
__device__ __forceinline__ uint2 lds64(uint32_t a) {
  uint2 v;
  asm volatile("ld.shared.v2.u32 {%0, %1}, [%2];\n" : "=r"(v.x), "=r"(v.y) : "r"(a) : "memory");
  return v;
}
__device__ __forceinline__ uint32_t lds_u16(uint32_t a) {
  uint16_t v;
  asm volatile("ld.shared.u16 %0, [%1];\n" : "=h"(v) : "r"(a) : "memory");
  return v;
}
__device__ __forceinline__ uint32_t lds_u8(uint32_t a) {
  uint32_t v;
  asm volatile("ld.shared.u8 %0, [%1];\n" : "=r"(v) : "r"(a) : "memory");
  return v;
}
__device__ __forceinline__ void sts128(uint32_t a, uint4 v) {
  asm volatile("st.shared.v4.u32 [%0], {%1, %2, %3, %4};\n" ::"r"(a), "r"(v.x), "r"(v.y), "r"(v.z), "r"(v.w) : "memory");
}
__device__ __forceinline__ void sts64(uint32_t a, uint2 v) {
  asm volatile("st.shared.v2.u32 [%0], {%1, %2};\n" ::"r"(a), "r"(v.x), "r"(v.y) : "memory");
}
// The NVFP4 verify kernel's dequantization (nvfp4_verify_attn.cuh dequant8): four packed bytes = 8 E2M1 values
// (element 2i in the low nibble of byte i) times one E4M3 block scale -> 8 E4M3 values (two words, element order kept).
// f16(e2m1) * f16(scale) is exact; one rounding to e4m3. Here it only builds the table (lut_entry).
__device__ __forceinline__ uint2 dequant8_cvt(uint32_t packed, uint32_t scale2_f16x2) {
  uint32_t lo, hi;
  asm("{\n"
      ".reg .b8 b0, b1, b2, b3;\n"
      ".reg .b32 h0, h1, h2, h3;\n"
      ".reg .b16 e0, e1, e2, e3;\n"
      "mov.b32 {b0, b1, b2, b3}, %2;\n"
      "cvt.rn.f16x2.e2m1x2 h0, b0;\n"
      "cvt.rn.f16x2.e2m1x2 h1, b1;\n"
      "cvt.rn.f16x2.e2m1x2 h2, b2;\n"
      "cvt.rn.f16x2.e2m1x2 h3, b3;\n"
      "mul.rn.f16x2 h0, h0, %3;\n"
      "mul.rn.f16x2 h1, h1, %3;\n"
      "mul.rn.f16x2 h2, h2, %3;\n"
      "mul.rn.f16x2 h3, h3, %3;\n"
      "cvt.rn.satfinite.e4m3x2.f16x2 e0, h0;\n"
      "cvt.rn.satfinite.e4m3x2.f16x2 e1, h1;\n"
      "cvt.rn.satfinite.e4m3x2.f16x2 e2, h2;\n"
      "cvt.rn.satfinite.e4m3x2.f16x2 e3, h3;\n"
      "mov.b32 %0, {e0, e1};\n"
      "mov.b32 %1, {e2, e3};\n"
      "}\n"
      : "=r"(lo), "=r"(hi)
      : "r"(packed), "r"(scale2_f16x2));
  return make_uint2(lo, hi);
}
// One E4M3 byte -> the f16 value in both halves of an f16x2 word (exact).
__device__ __forceinline__ uint32_t scale_f16x2(uint32_t e4m3_byte) {
  uint32_t d;
  const uint16_t pair = static_cast<uint16_t>(e4m3_byte | (e4m3_byte << 8));
  asm("cvt.rn.f16x2.e4m3x2 %0, %1;\n" : "=r"(d) : "h"(pair));
  return d;
}
// Table entry of scale byte s: bytes k = 0..7 hold e4m3(f16(E2M1 magnitude k) * f16(s)) (magnitudes 0, 0.5, 1, 1.5, 2,
// 3, 4, 6), computed with dequant8_cvt on the packed magnitudes 0..7.
__device__ __forceinline__ uint2 lut_entry(uint32_t s) { return dequant8_cvt(0x76543210u, scale_f16x2(s)); }
// Eight packed E2M1 values (one word, element k in nibble k) -> 8 e4m3 values through a table entry t (two words,
// element order kept): the magnitude (nibble bits 0-2) selects the byte of t (prmt), and the nibble's sign bit
// (bit 3) flips the e4m3 sign bit: e4m3 rounding with satfinite is symmetric in the sign, so for every finite scale this
// equals dequant8_cvt (verified exhaustively by tests/test_kv4_attn_wide.py). Sign bytes come from prmt's sign-replicate
// mode: in w << 4 the sign of an even element is the msb of its byte, in w the sign of an odd element is.
__device__ __forceinline__ uint2 lut8(uint32_t w, uint2 t) {
  const uint32_t m = w & 0x77777777u;
  uint32_t lo, hi, slo, shi;
  const uint32_t x4 = w << 4;
  asm("prmt.b32 %0, %1, %2, %3;\n" : "=r"(lo) : "r"(t.x), "r"(t.y), "r"(m));
  asm("prmt.b32 %0, %1, %2, %3;\n" : "=r"(hi) : "r"(t.x), "r"(t.y), "r"(m >> 16));
  asm("prmt.b32 %0, %1, %2, 0xD9C8;\n" : "=r"(slo) : "r"(x4), "r"(w));
  asm("prmt.b32 %0, %1, %2, 0xFBEA;\n" : "=r"(shi) : "r"(x4), "r"(w));
  return make_uint2(lo ^ (slo & 0x80808080u), hi ^ (shi & 0x80808080u));
}
// lut8 with w << 4 taken as a funnel shift (rotate) of w: only the most significant bit of each byte of x4 is read
// (byte-permute sign replication), and those bits are the same in the shift and the rotate; same outputs as lut8
__device__ __forceinline__ uint2 lut8_rot(uint32_t w, uint2 t) {
  const uint32_t m = w & 0x77777777u;
  uint32_t lo, hi, slo, shi, x4;
  asm("shf.l.wrap.b32 %0, %1, %1, 4;\n" : "=r"(x4) : "r"(w));
  asm("prmt.b32 %0, %1, %2, %3;\n" : "=r"(lo) : "r"(t.x), "r"(t.y), "r"(m));
  asm("prmt.b32 %0, %1, %2, %3;\n" : "=r"(hi) : "r"(t.x), "r"(t.y), "r"(m >> 16));
  asm("prmt.b32 %0, %1, %2, 0xD9C8;\n" : "=r"(slo) : "r"(x4), "r"(w));
  asm("prmt.b32 %0, %1, %2, 0xFBEA;\n" : "=r"(shi) : "r"(x4), "r"(w));
  return make_uint2(lo ^ (slo & 0x80808080u), hi ^ (shi & 0x80808080u));
}
// lut8 with the high selector m >> 16 taken as the high word of m * 2^16 (an integer multiply instead of a shift; the
// same value for every m), so one of its integer operations runs as a multiply; same outputs as lut8
__device__ __forceinline__ uint2 lut8_mul(uint32_t w, uint2 t) {
  const uint32_t m = w & 0x77777777u;
  uint32_t lo, hi, slo, shi, mh;
  asm("mul.hi.u32 %0, %1, 65536;\n" : "=r"(mh) : "r"(m));
  const uint32_t x4 = w << 4;
  asm("prmt.b32 %0, %1, %2, %3;\n" : "=r"(lo) : "r"(t.x), "r"(t.y), "r"(m));
  asm("prmt.b32 %0, %1, %2, %3;\n" : "=r"(hi) : "r"(t.x), "r"(t.y), "r"(mh));
  asm("prmt.b32 %0, %1, %2, 0xD9C8;\n" : "=r"(slo) : "r"(x4), "r"(w));
  asm("prmt.b32 %0, %1, %2, 0xFBEA;\n" : "=r"(shi) : "r"(x4), "r"(w));
  return make_uint2(lo ^ (slo & 0x80808080u), hi ^ (shi & 0x80808080u));
}
template <int kRegs>
__device__ __forceinline__ void setmaxnreg_inc() {
  asm volatile("setmaxnreg.inc.sync.aligned.u32 %0;\n" ::"n"(kRegs));
}
template <int kRegs>
__device__ __forceinline__ void setmaxnreg_dec() {
  asm volatile("setmaxnreg.dec.sync.aligned.u32 %0;\n" ::"n"(kRegs));
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

// ---------------------------------------------------------------------------------------------- main kernel
template <int kPairs, int kF8, int kRaw, bool kUsePDL, int kTpr, int kDq, int kSkel>
__global__ void __launch_bounds__(threads_of<kSkel>(), 1)
    kv4_attn_wide_kernel(const Params p, const __grid_constant__ CUtensorMap q_map,
                         const __grid_constant__ CUtensorMap kd_map, const __grid_constant__ CUtensorMap vd_map,
                         const __grid_constant__ CUtensorMap ks_map, const __grid_constant__ CUtensorMap vs_map) {
  using L = Layout<kPairs, kF8, kRaw, kTpr, (((kDq >> 17) & 1) || kPairs != 2) ? 1 : 0>;
  constexpr int kHeads = L::kHeads, kTileKeys = L::kTileKeys, kSmWarps = L::kSmWarps, kKpt = L::kKpt;
  // the softmax below is full_attn_wide's code verbatim, in its served instantiation (no probe mode, no polynomial)
  // skeleton bits 18-20: kPoly of every 8 keys take 2^x from the FMA pipes (exp2_poly2, full_attn_wide's option; 0 =
  // the default code)
  constexpr int kMode = 0, kPoly = (kSkel >> 18) & 7;
  // skeleton options (kSkel, each bit keeps every output bit unchanged): bit 0 - the tile maximum of an interior tile from
  // the score loads (tcgen05.ld .red .max); bit 1 - e4m3 probability packing joined in one statement; bit 2 (4 or 8
  // dequantization warps) - register split kWg2RegsR / kDqRegsR, a tile's shared-memory loads issued before its stores,
  // table offsets by one byte permute each
  constexpr bool kRedMax = (kSkel & 1) != 0;
  constexpr bool kPackJoined = (kSkel & 2) != 0;
  constexpr bool kDqR = (kSkel & 4) != 0;
  // bit 3 (with bit 2): the table dequantization's 4-bit shift as a funnel shift (lut8_rot); bit 4: a 256 ns suspend-time
  // hint on the load warp's raw-stage refill wait only (still bounded)
  constexpr bool kRot = (kSkel & 8) != 0;
  // bit 6 (with bit 2): the table dequantization's high selector (m >> 16) as the high word of m * 2^16 (lut8_mul)
  constexpr bool kMulHi = (kSkel & 64) != 0;
  // bit 7 (with bit 2): the table entry address by an integer multiply-add instead of a shift-add
  constexpr bool kMadAddr = (kSkel & 128) != 0;
  // bit 8 (P in tensor memory, no rescale warpgroup): combine(g - 1) before the store of P(g)
  constexpr bool kEarlyComb = (kSkel & 256) != 0 && ((kDq >> 17) & 1) == 0 && kPairs == 2 && (kSkel & 32) == 0;
  // bit 9: every output column of a tile loaded before one wait in the combine (full_attn_wide's option 4); bit 10: the
  // load warp's page-list loads issued eight per lane before their stores (full_attn_wide's option 8); same outputs
  constexpr bool kOneWaitO = (kSkel & 512) != 0;
  constexpr bool kPageFill = (kSkel & 1024) != 0;
  // suspend hint (ns) of the waits of the load / MMA / dequantization warps (development tunable, kDq bits 8-15 x 64)
  constexpr uint32_t kAuxHint = ((static_cast<uint32_t>(kDq) >> 8) & 0xFFu) * 64u;
  constexpr uint32_t kLoadHint = (kSkel & 16) != 0 ? 256u : kAuxHint;
  // probabilities to the PV MMA through tensor memory with 4 KV heads (64-key tiles); through shared memory, as
  // full_attn_wide does, with 2 KV heads (128-key tiles: measured faster there, bench/bench_kv4_attn.py) or with
  // development bit 17
  constexpr bool kPTmem = ((kDq >> 17) & 1) == 0 && kPairs == 2;
  static_assert(!kPTmem || kKpt == 64, "tensor-memory probabilities: 64 keys per softmax thread");
  // option bit 5: rescale warpgroup (see kCrWarp0); one output buffer in tensor memory (columns 256..383) and the
  // accumulator (384..511); the per-tile (ar, at) of every row through shared memory (the kTpr-2 exchange area, unused at
  // one thread per row)
  constexpr bool kCr = (kSkel & 32) != 0;
  static_assert(!kCr || (kDqR && kDqWarps == 4 && kPTmem && kTpr == 1 && kPairs == 2),
                "option bit 5 needs bit 2, 4 dequantization warps, 4 KV heads, P in TMEM");
  // option 4096 (4 KV heads, one softmax thread per row, P in tensor memory): the output accumulates in tensor memory
  // across a piece's tiles. QK is unchanged; PV runs block-scaled (kind::mxf8f6f4, UE8M0) at M = 128 over the 128 lanes
  // of a KV-head pair as the scores lie (the two heads interleaved by 16 lanes): each K step covers 32 keys of one head,
  // and the rows of the other head take scale factor 2^-127 for it (their products vanish below the FP32 accumulator's
  // resolution). Each row's probabilities of a tile are e4m3(448 * 2^(s * scale_log2 - kt)), kt = ceil(tile maximum), and
  // its scale factor 2^(kt - m) (m = the row's reference, an integer) puts them on the accumulator's scale; no per-tile
  // output combine. A tile maximum more than 64 above m re-references the row (accumulator and row sum scaled by a power
  // of two). Option 8192 (with 4096): the row sum is accumulated by the same MMAs (a constant 1.0 B block), i.e. the sum
  // of the probabilities as applied.
  constexpr bool kPvAcc = (kSkel & 4096) != 0;
  constexpr bool kLmma = kPvAcc && (kSkel & 8192) != 0;
  // option 32768 (with 4096): the probabilities in their own tensor-memory columns instead of over the scores, so the QK of
  // tile g + 2 waits only for the softmax's score loads (not for PV(g)); the softmax waits for PV(g - 2) before it rewrites
  // P buffer g & 1. Columns with option 4096: S 0-255, O 256-383; without 32768: SFA 384-387, SFB 392-395, row sums 400-431;
  // with it: P 384-447 (32 bb + 16 c), SFA 448-451, SFB 452-455, row sums 456-487.
  constexpr bool kPsep = kPvAcc && (kSkel & 32768) != 0;
  // option 131072 (with 4096): the re-reference block rescales the accumulator 16 columns at a time (same products and
  // outputs), so the block keeps few registers live across the per-tile softmax path (tunable; bench/bench_kv4_attn.py)
  constexpr bool kSlimRef = kPvAcc && (kSkel & 131072) != 0;
  // option 2097152: the first piece's Q requested before the page list (see the load warp); same outputs
  constexpr bool kEarlyQ = (kSkel & 2097152) != 0;
  // option 4194304 (with 4096 + 8192, P in tensor memory): the probabilities' 2^y in half precision (see the softmax); a
  // numerics change (measured against float64 by the unit)
  constexpr bool kF16Exp = (kSkel & 4194304) != 0;
  // option 8388608 (with 16384): the Q terms' E2M1 codes by the conversion instruction on r x the rounded reciprocal of the
  // block scale (a numerics change at exact midpoints only; measured against float64 by the unit)
  constexpr bool kFastQ = (kSkel & 8388608) != 0;
  static_assert(!kF16Exp || (kLmma && kPTmem && kKpt == 64 && kPoly == 0), "option 4194304 needs 4096 + 8192");
  constexpr uint32_t kColSfa = kPsep ? 448u : 384u, kColSfb = kPsep ? 452u : 392u, kColL = kPsep ? 456u : 400u;
  const auto p_col = [](uint32_t bb_, uint32_t c_) -> uint32_t { return kPsep ? 384u + 32u * bb_ + 16u * c_ : 128u * bb_ + 64u * c_; };
  static_assert(!kPvAcc || (kPairs == 2 && kTpr == 1 && kPTmem && !kCr && !kEarlyComb),
                "option 4096 needs 4 KV heads, one softmax thread per row, P in tensor memory, no option 32 / 256");
  // option 65536: K = 64 per MMA (the instruction descriptor's K-dimension bit): one QK MMA per head and tile instead of
  // two, one PV MMA per 64 keys instead of two (option 4096: scale_vec::2X, the row's two scale factors in the 16-bit
  // half of its word that the scale-factor ID selects)
  constexpr bool kK64 = (kSkel & 65536) != 0;
  constexpr int kKStep = kK64 ? 64 : 32;  // K (e4m3 values) per MMA
  // option 16384 (with 4096, without 32768): QK block-scaled on the packed NVFP4 K (kind::mxf4nvf4, block 16, UE4M3
  // scales), no K dequantization. Q is kQTerms NVFP4 terms of the row's e4m3 query (each term quantizes the remaining
  // residual: block 16, scale = e4m3(amax / 6), E2M1 round-to-nearest-even, satfinite), relative to a per-row power of two
  // (folded into the row's score scale) that puts the row maximum at most 6 x 448; written by the softmax threads at
  // each piece's start (A operand in tensor memory, columns kColQa + 16 c + 8 t, each lane its own row). The MMAs run at
  // M = 128 over a pair's 128 lanes as they lie: the MMA of head 2c + hh takes the scale factors whose other-head rows
  // are 0, so those rows add exactly 0 (scale-factor A columns kColSfaQ + 4 c + 2 t + hh). The K scale words go from the
  // raw stage to tensor memory (the B scale factors, columns kColSfbK + 8 stage + 2 head, every lane quarter) by the
  // dequantization warps; the K data is read by the MMAs from the raw stage (128-byte swizzle), which the QK commit
  // releases with the dequantization threads.
  constexpr bool kD4b = (kSkel & 16384) != 0;
  constexpr int kQTerms = 2;
  constexpr uint32_t kColQa = 432u, kColSfaQ = kColQa + 16u * kPairs * kQTerms / 2, kColSfbK = kColSfaQ + 8u;
  static_assert(!kD4b || (kPvAcc && !kPsep && kPairs == 2 && kTpr == 1 && !kDqR),
                "option 16384 needs option 4096, not 32768 / 4");
  static_assert(!kD4b || kColQa >= kColL + 32u, "option 16384: tensor-memory columns after the row sums");
  static_assert(!kD4b || kColSfbK + 8u * kF8 <= kTmemCols, "option 16384: tensor-memory columns");
  extern __shared__ uint8_t smem_raw[];
  // 1024-byte aligned base by pointer arithmetic on the shared array (keeps the shared address space visible to the
  // compiler: st.shared / ld.shared instead of generic accesses)
  uint8_t* const sm = smem_raw + ((1024u - (smem_u32(smem_raw) & 1023u)) & 1023u);
  uint64_t* const bars = reinterpret_cast<uint64_t*>(sm + L::kSmBar);
  // bars: f8 full[kF8] (as full_attn_wide's full[]), f8 empty[kF8] (its empty[]), s_full[2], p_full[2], o_full[2],
  // q_full[2], q_empty[2], s_free[2], raw full[kRaw], raw empty[kRaw], (tmem address)
  const uint32_t bar0 = smem_u32(bars);
  const auto full_bar = [&](int s) { return bar0 + 8u * s; };
  const auto empty_bar = [&](int s) { return bar0 + 8u * (kF8 + s); };
  const auto s_full_bar = [&](int b) { return bar0 + 8u * (2 * kF8 + b); };
  const auto p_full_bar = [&](int b) { return bar0 + 8u * (2 * kF8 + 2 + b); };
  const auto o_full_bar = [&](int b) { return bar0 + 8u * (2 * kF8 + 4 + b); };
  const auto q_full_bar = [&](int b) { return bar0 + 8u * (2 * kF8 + 6 + b); };
  const auto q_empty_bar = [&](int b) { return bar0 + 8u * (2 * kF8 + 8 + b); };
  const auto s_free_bar = [&](int b) { return bar0 + 8u * (2 * kF8 + 10 + b); };  // scores of buffer b in registers
  const auto raw_full_bar = [&](int s) { return bar0 + 8u * (2 * kF8 + 12 + s); };
  const auto raw_empty_bar = [&](int s) { return bar0 + 8u * (2 * kF8 + 12 + kRaw + s); };
  // V half of FP8 stage s written (full_bar(s) then means: K half written; QK waits on it, PV on this one)
  const auto vfull_bar = [&](int s) { return bar0 + 8u * (2 * kF8 + 12 + 2 * kRaw + s); };
  uint32_t* const tmem_slot = reinterpret_cast<uint32_t*>(sm + L::kSmBar + 8 * L::kNumBars);
  // option bit 5 barriers, after the tensor-memory address word: arat_full[2] ((ar, at) of tile g in buffer g & 1: 256
  // softmax thread arrivals), arat_free[2] (the rescale threads read them: 128 arrivals), o_free[2] (the rescale warps
  // loaded O(g): one elected lane per warp), cr_done (a piece's accumulator complete: one per rescale warp), acc_free (the
  // softmax warps loaded it: one per softmax warp)
  const auto cr_bar = [&](int j) { return bar0 + 8u * (L::kNumBars + 1 + j); };
  const auto arat_full_bar = [&](int b) { return cr_bar(b); };
  const auto arat_free_bar = [&](int b) { return cr_bar(2 + b); };
  const auto o_free_bar = [&](int b) { return cr_bar(4 + b); };
  const auto cr_done_bar = [&]() { return cr_bar(6); };
  const auto acc_free_bar = [&]() { return cr_bar(7); };
  // option 16384: the piece's Q terms are in tensor memory (one elected lane per softmax warp)
  const auto qt_full_bar = [&]() { return cr_bar(8); };
  static_assert(8 * (L::kNumBars + 1 + 9) <= 512, "barrier region (option bits 5, 16384)");
  float2* const arat = reinterpret_cast<float2*>(sm + L::kSmRed);  // [2][heads * 64 rows] (ar, at)
  static_assert(!kCr || 2 * kHeads * kRows * 8 <= 3 * kHeads * kRows * 2 * 4, "(ar, at) buffers fit the exchange area");
  int32_t* const pre = reinterpret_cast<int32_t*>(sm + L::kSmPre);  // pre[r] = tiles of requests < r; pre[R] = W
  int32_t* const pages = reinterpret_cast<int32_t*>(sm + L::kSmPages);

  // warp index through a warp shuffle: the compiler then knows it is warp-uniform, so the role branches below are
  // convergent (the MMA warp's descriptors stay in uniform registers and elect.sync needs no emulation)
  const int tid = threadIdx.x, warp = __shfl_sync(0xffffffffu, tid >> 5, 0), lane = tid & 31;
  const int b = blockIdx.x;
  const int R = p.num_reqs;
  // development timeline (kDq bit 16): clock stamps of the hand-offs of blocks 0, G/2 and G-1 for tiles < 256, written
  // after the partial-record region of the workspace: [3 blocks][24 events][256 tiles] u64 (full_attn_wide's events
  // 0-15, dequantization 16-19)
  constexpr bool kTrace = (kDq & 16) != 0;
  const int trace_slot = !kTrace ? -1
                         : (b == 0 ? 0 : b == static_cast<int>(gridDim.x) / 2 ? 1 : b == static_cast<int>(gridDim.x) - 1 ? 2 : -1);
  uint64_t* const trace = reinterpret_cast<uint64_t*>(
      p.partial + static_cast<int64_t>(gridDim.x) * 2 * (2 * kPairs * kRows) * kRec);
  const auto stamp = [&](int ev, int gg) {
    if (kTrace && trace_slot >= 0 && gg >= 0 && gg < 256) {
      uint64_t c;
      asm volatile("mov.u64 %0, %%clock64;\n" : "=l"(c));
      trace[(trace_slot * 24 + ev) * 256 + gg] = c;
    }
  };

  // Barriers and tensor memory first: they touch nothing the preceding kernels write.
  if (tid == 0) {
#pragma unroll
    for (int s = 0; s < kF8; ++s) {
      mbar_init(full_bar(s), kDqThreads);   // K half: every dequantization thread arrives after its stores + fence
      mbar_init(vfull_bar(s), kDqThreads);  // V half: likewise
      mbar_init(empty_bar(s), 1);          // released by the commit after the tile's PV MMAs
    }
#pragma unroll
    for (int s = 0; s < kRaw; ++s) {
      mbar_init(raw_full_bar(s), 1);            // the load thread's expect_tx arrival + the copies' bytes
      // (option 16384: + the QK commit, the MMAs read the packed K from the raw stage)
      mbar_init(raw_empty_bar(s), kDqThreads + (kD4b ? 1 : 0));  // every dequantization thread arrives after its reads + proxy fence
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
    if constexpr (kCr) {
#pragma unroll
      for (int i = 0; i < 2; ++i) {
        mbar_init(arat_full_bar(i), 32 * kSmWarps);
        mbar_init(arat_free_bar(i), 128);
        mbar_init(o_free_bar(i), 4);
      }
      mbar_init(cr_done_bar(), 4);
      mbar_init(acc_free_bar(), kSmWarps);
    }
    // option 4096: acc_free = the softmax warps loaded a piece's accumulator (the next piece's first PV may overwrite it)
    if constexpr (kPvAcc) mbar_init(acc_free_bar(), kSmWarps);
    if constexpr (kD4b) mbar_init(qt_full_bar(), kSmWarps);
    asm volatile("fence.mbarrier_init.release.cluster;\n" ::: "memory");
  }
  if (warp == 0) {
    asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], %1;\n" ::"r"(smem_u32(tmem_slot)),
                 "r"(kTmemCols)
                 : "memory");
    asm volatile("tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned;\n" ::: "memory");
  }
  // Dequantization table (constants only, so it is built before the dependency wait); read after the barrier below.
  if (tid < 256) sts64(smem_u32(sm + L::kSmLut + 8 * tid), lut_entry(static_cast<uint32_t>(tid)));
  if constexpr (kLmma) {
    // option 8192: 4 KB of e4m3 1.0 (0x38) in the exchange area (unused at one softmax thread per row without option 32;
    // the descriptor's whole span lies inside): the B operand of the row-sum MMAs; generic stores read by the MMAs (async
    // proxy): each writing thread fences, and the block barrier below orders them before the PV warp's first MMA
    static_assert(3 * kHeads * kRows * 2 * 4 >= 4096, "exchange area holds the 1.0 block");
    if (tid < 256) sts128(smem_u32(sm + L::kSmRed + 16 * tid), make_uint4(0x38383838u, 0x38383838u, 0x38383838u, 0x38383838u));
    fence_proxy_async();
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

  // Registers: warpgroups 2-3 (load, MMA, dequantization, spare) give registers to the softmax warpgroups 0-1. Each
  // setmaxnreg heads the branch of its warpgroups (one static instruction per side, executed by every warp of those
  // warpgroups), so the compiler applies its limit to all of that branch's code. ntiles is the same for every warp.
  if (ntiles > 0) {
    if (kCr && warp >= kCrWarp0) {
      setmaxnreg_dec<kCrRegs>();
      // ------------------------------------------------------------------ output rescale-and-add (option bit 5)
      // Warp kCrWarp0 + q (tensor-memory lanes 32 q ..): lane i holds, for KV-head pair c (columns 64 c ..), the output
      // row of head 2 c + i / 16, row 16 q + i % 16 (the softmax thread of warp q + 4 c, lane i). Per tile g of a piece:
      // (ar, at) of both rows from shared memory, then O(g) (one buffer) and the accumulator from tensor memory, and
      // acc = fma(acc, ar, O * at) in packed fp32 pairs - the softmax's combine, the same two roundings per element in the
      // same tile order; the first tile of a piece starts from acc = 0 as the softmax's registers did.
      const int q = warp - kCrWarp0;
      const uint32_t lane_base = (32u * q) << 16;
      const int hh = lane >> 4, row = 16 * q + (lane & 15);
      const int rr0 = hh * kRows + row, rr1 = (2 + hh) * kRows + row;  // record rows of pair 0 / pair 1
      // g counts the tiles and kp the pieces of the block's whole range, continuous across pieces, as in every other role:
      // they select every barrier buffer and parity
      int g = 0, kp = 0;
      int r = r_first;
      for (int t = T0; t < T1; ++r, ++kp) {
        const int lo = t - pre[r], hi = min(T1, pre[r + 1]) - pre[r];
        for (int i = lo; i < hi; ++i, ++g) {
          const int b2 = g & 1;
          mbar_wait(arat_full_bar(b2), (g >> 1) & 1);
          const float2 aa0 = arat[b2 * (kHeads * kRows) + rr0], aa1 = arat[b2 * (kHeads * kRows) + rr1];
          mbar_arrive(arat_free_bar(b2));  // the softmax may refill buffer b2 for tile g + 2
          mbar_wait_warp(o_full_bar(b2), (g >> 1) & 1);
          if (lane == 0 && q == 0) stamp(8, g);
          tc_fence_after();
          if (i == lo && kp > 0) {
            // the softmax warps loaded the previous piece's accumulator before this piece overwrites it
            mbar_wait_warp(acc_free_bar(), (kp - 1) & 1);
            tc_fence_after();
          }
#pragma unroll
          for (int ch = 0; ch < 4; ++ch) {  // 32 columns at a time: pair ch / 2, dims 32 (ch % 2) ..
            const float2 aa = ch < 2 ? aa0 : aa1;
            const float2 a2 = make_float2(aa.x, aa.x), t2 = make_float2(aa.y, aa.y);
            float v[32], acc[32];
            tmem_ld32(tmem + lane_base + 256u + 32u * ch, v);
            if (i > lo) {
              tmem_ld32(tmem + lane_base + 384u + 32u * ch, acc);
            } else {
#pragma unroll
              for (int d = 0; d < 32; ++d) acc[d] = 0.0f;
            }
            tmem_wait_ld();
            if (ch == 3) {
              // every column of O(g) is in registers: PV(g + 1) may write the output buffer
              tc_fence_before();
              __syncwarp();
              if (lane == 0) mbar_arrive(o_free_bar(b2));
            }
            uint32_t w[32];
#pragma unroll
            for (int d = 0; d < 16; ++d) {
              const float2 o = fma2(make_float2(acc[2 * d], acc[2 * d + 1]), a2,
                                    mul2(make_float2(v[2 * d], v[2 * d + 1]), t2));
              w[2 * d] = __float_as_uint(o.x);
              w[2 * d + 1] = __float_as_uint(o.y);
            }
            tmem_st32(tmem + lane_base + 384u + 32u * ch, w);
            tmem_wait_st();
          }
          if (lane == 0 && q == 0) stamp(9, g);
          if (i == hi - 1) {
            // the piece's accumulator is complete: the softmax warps read it
            tc_fence_before();
            __syncwarp();
            if (lane == 0) mbar_arrive(cr_done_bar());
          }
        }
        t = pre[r] + hi;
      }
    } else if (kDqR && warp >= kDqWarp0) {
      if constexpr (kCr) {
        setmaxnreg_inc<kDqRegsCr>();
      } else {
        setmaxnreg_dec<kDqRegsR>();
      }
      // ------------------------------------------------------------------ dequantization, skeleton option bit 2
      // The same chunks, rows, step order and stores as the dequantization branch below (4 or 8 dequantization warps),
      // with the tile's shared-memory loads (both scale quads, both halves' packed chunks, the table entries) issued
      // before its first store, and each table offset taken by one byte permute of the scale word. Same arrivals per tile.
      const int d = tid - 32 * kDqWarp0;
      constexpr int kParts = kHeads * 2;
      constexpr int kSRows = kTileKeys / 32;
      constexpr int kRpt = kParts * kTileKeys / kDqThreads;
      // (this branch is taken only with bit 2; other builds compile it unused) rows per thread: 4 (4 dequantization warps)
      // or 2 (8 warps), the rows of the branch below
      static_assert(!kDqR || ((kRpt == 4 || kRpt == 2) && kRpt * kDqThreads == kParts * kTileKeys), "rows per thread");
      const int p = d % kParts, grp = d / kParts, h = p >> 1, half = p & 1;
      const int quad = kRpt == 4 ? grp : kRpt * grp / 4;
      const int tb = kRpt == 4 ? 0 : kRpt * grp % 4;  // the first row's place in its quad
      const int flip = kPairs == 2 ? (p >> 2) : (grp & 1);
      const int c0 = (h & 1) * 4 + half * 2;
      const int r128 = h * kSRows + (quad >> 3);
      const int s_off = r128 * 128 + (((quad & 7) ^ (r128 & 7)) << 4);
      const uint8_t* const lutb = sm + L::kSmLut;
      int roff[kRpt], doff0[kRpt], doff1[kRpt];
      uint32_t vsel[kRpt];  // byte permute selector of the V scale byte of step i (row place tb + (i ^ flip)), zero-extended
#pragma unroll
      for (int i = 0; i < kRpt; ++i) {
        const int row = kRpt * grp + (i ^ flip);
        roff[i] = row * (kHeads * 32) + h * 32 + half * 16;
        const int db = (h >> 1) * (kTileKeys * 128) + row * 128;
        doff0[i] = db + ((c0 ^ (row & 7)) << 4);
        doff1[i] = db + (((c0 + 1) ^ (row & 7)) << 4);
        vsel[i] = 0x4440u | static_cast<uint32_t>(kRpt == 4 ? (i ^ flip) : tb + (i ^ flip));
      }
      // K: the scale word of the step's row holds blocks 0..3; this thread's blocks are bytes 2 half and 2 half + 1
      const uint32_t ksel0 = 0x4440u | static_cast<uint32_t>(2 * half), ksel1 = 0x4440u | static_cast<uint32_t>(2 * half + 1);
      // option bit 7: the table entry address as one integer multiply-add by an opaque register 8 (same address)
      uint32_t eight = 8u;
      if constexpr (kMadAddr) asm volatile("mov.b32 %0, 8;\n" : "=r"(eight));
      const uint32_t lut_s = smem_u32(sm + L::kSmLut);
      const auto lut_at = [&](uint32_t byte) {  // table entry of scale byte `byte` (zero-extended)
        if constexpr (kMadAddr) {
          uint32_t a;
          uint2 t;
          asm("mad.lo.u32 %0, %1, %2, %3;\n" : "=r"(a) : "r"(byte), "r"(eight), "r"(lut_s));
          asm volatile("ld.shared.v2.u32 {%0, %1}, [%2];\n" : "=r"(t.x), "=r"(t.y) : "r"(a));
          return t;
        } else {
          return *reinterpret_cast<const uint2*>(lutb + 8u * byte);
        }
      };
      const auto lutq = [](uint32_t w, uint2 t) -> uint2 {  // option bit 3: funnel shift; bit 6: high-word multiply
        if constexpr (kRot) {
          return lut8_rot(w, t);
        } else if constexpr (kMulHi) {
          return lut8_mul(w, t);
        } else {
          return lut8(w, t);
        }
      };
      int rs = 0, fs = 0;
      uint32_t rph = 0, fbit = 0;
      for (int g = 0; g < ntiles; ++g) {
        if (d == 0) stamp(16, g);
        mbar_wait_hint<kAuxHint>(raw_full_bar(rs), rph);
        const uint8_t* const raw = sm + L::kSmRaw + rs * L::kRawBytes;
        const uint4 swk = *reinterpret_cast<const uint4*>(raw + 2 * L::kRawData + s_off);
        const uint4 swv = *reinterpret_cast<const uint4*>(raw + 2 * L::kRawData + L::kRawScale + s_off);
        uint4 ink[kRpt], inv[kRpt];
#pragma unroll
        for (int i = 0; i < kRpt; ++i) {
          ink[i] = *reinterpret_cast<const uint4*>(raw + roff[i]);
          inv[i] = *reinterpret_cast<const uint4*>(raw + L::kRawData + roff[i]);
        }
        uint2 tk0[kRpt], tk1[kRpt], tv0[kRpt], tv1[kRpt];
        {
          const uint32_t w[4] = {swk.x, swk.y, swk.z, swk.w};
          const uint32_t b0 = half ? swv.z : swv.x, b1 = half ? swv.w : swv.y;
          // 2 rows per thread: the words of row places tb, tb + 1
          const uint32_t wlo = tb ? swk.z : swk.x, whi = tb ? swk.w : swk.y;
#pragma unroll
          for (int i = 0; i < kRpt; ++i) {
            uint32_t wi;
            if constexpr (kRpt == 4) {
              wi = flip ? w[i ^ 1] : w[i];
            } else {
              wi = (i ^ flip) ? whi : wlo;
            }
            tk0[i] = lut_at(__byte_perm(wi, 0u, ksel0));
            tk1[i] = lut_at(__byte_perm(wi, 0u, ksel1));
            tv0[i] = lut_at(__byte_perm(b0, 0u, vsel[i]));
            tv1[i] = lut_at(__byte_perm(b1, 0u, vsel[i]));
          }
        }
        if (g >= kF8) mbar_wait_hint<kAuxHint>(empty_bar(fs), fbit ^ 1u);  // = ((g / kF8) - 1) & 1
        if (d == 0) stamp(17, g);
        uint8_t* const dk = sm + L::kSmK + fs * kTileBytes;
        uint8_t* const dv = sm + L::kSmV + fs * kTileBytes;
#pragma unroll
        for (int i = 0; i < kRpt; ++i) {
          const uint2 a = lutq(ink[i].x, tk0[i]), bq = lutq(ink[i].y, tk0[i]);
          const uint2 c = lutq(ink[i].z, tk1[i]), e = lutq(ink[i].w, tk1[i]);
          *reinterpret_cast<uint4*>(dk + doff0[i]) = make_uint4(a.x, a.y, bq.x, bq.y);
          *reinterpret_cast<uint4*>(dk + doff1[i]) = make_uint4(c.x, c.y, e.x, e.y);
        }
#pragma unroll
        for (int i = 0; i < kRpt; ++i) {
          const uint2 a = lutq(inv[i].x, tv0[i]), bq = lutq(inv[i].y, tv0[i]);
          const uint2 c = lutq(inv[i].z, tv1[i]), e = lutq(inv[i].w, tv1[i]);
          *reinterpret_cast<uint4*>(dv + doff0[i]) = make_uint4(a.x, a.y, bq.x, bq.y);
          *reinterpret_cast<uint4*>(dv + doff1[i]) = make_uint4(c.x, c.y, e.x, e.y);
        }
        // one fence per tile, then the K half, the V half and the raw stage (as the branch below, default form)
        fence_proxy_async();
        mbar_arrive(full_bar(fs));
        if (d == 0) stamp(18, g);
        mbar_arrive(vfull_bar(fs));
        mbar_arrive(raw_empty_bar(rs));
        if (d == 0) stamp(19, g);
        if (++rs == kRaw) {
          rs = 0;
          rph ^= 1u;
        }
        if (++fs == kF8) {
          fs = 0;
          fbit ^= 1u;
        }
      }
    } else if (warp >= kSmWarps) {
    setmaxnreg_dec<kDqR ? kWg2RegsR : kAuxRegs>();
    if (warp == kSmWarps) {
      // ------------------------------------------------------------------ loads
      // page numbers of the range -> shared memory (all lanes, all loads in flight together)
      constexpr int kTilesPerPage = 128 / kTileKeys;
      // option 2097152: the first piece's Q rows are requested before the page list is gathered (they need no page), so
      // they land while the page list loads (the option-16384 Q-term build of the first piece then overlaps it)
      if (kEarlyQ && lane == 0) {
        asm volatile("fence.proxy.async.global;\n" ::: "memory");
        mbar_expect_tx(q_full_bar(0), kHeads * kRows * 64);
#pragma unroll
        for (int h = 0; h < kHeads; ++h)
          tma_load_3d(smem_u32(sm + L::kSmQ + h * kRows * 64), &q_map, 0, h * kQPerKv, r_first * kTokens, q_full_bar(0));
      }
      {
        int r = r_first, base = 0;
        for (int t = T0; t < T1 && base < kMaxPages;) {
          const int lo = t - pre[r], hi = min(T1, pre[r + 1]) - pre[r];
          const int pg0 = lo / kTilesPerPage, pg1 = (hi - 1) / kTilesPerPage + 1;
          const int32_t* const pt = p.page_table + static_cast<int64_t>(r) * p.page_table_stride;
          if constexpr (kPageFill) {
            // option bit 10: eight page loads per lane issued before any store (one load per iteration keeps only one
            // load in flight per lane); same entries
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
          if (k >= kQB) mbar_wait_hint<kAuxHint>(q_empty_bar(k % kQB), ((k / kQB) - 1) & 1);
          if (!(kEarlyQ && k == 0)) {  // (option 2097152: piece 0's Q was requested above)
          mbar_expect_tx(q_full_bar(k % kQB), kHeads * kRows * 64);
#pragma unroll
          for (int h = 0; h < kHeads; ++h)
            tma_load_3d(smem_u32(sm + L::kSmQ + (k % kQB) * L::kQBuf + h * kRows * 64), &q_map, 0, h * kQPerKv,
                        r * kTokens, q_full_bar(k % kQB));
          }
          const int32_t* const pt = p.page_table + static_cast<int64_t>(r) * p.page_table_stride;
          for (int i = lo; i < hi; ++i, ++g) {
            // tile g -> raw stage g % kRaw (free once the dequantization threads released tile g - kRaw)
            const int s = g % kRaw;
            if (g >= kRaw) mbar_wait_hint<kLoadHint>(raw_empty_bar(s), ((g / kRaw) - 1) & 1);
            const int pi = base + i / kTilesPerPage - pg0;
            const int page = pi < kMaxPages ? pages[pi] : pt[i / kTilesPerPage];
            const int tok0 = (i % kTilesPerPage) * kTileKeys;  // first token of the tile inside its page
            const int row0 = page * kPage + tok0;
            const uint32_t rs = smem_u32(sm + L::kSmRaw + s * L::kRawBytes);
            mbar_expect_tx(raw_full_bar(s), L::kRawBytes);
            // packed rows: one [kTileKeys][Hkv * 32 B] box per pool; scale rows: one [Hkv][kTileKeys / 32][32 x 4 B] box
            // per pool (128-byte swizzle)
            tma_load_2d(rs, &kd_map, 0, row0, raw_full_bar(s));
            tma_load_2d(rs + L::kRawData, &vd_map, 0, row0, raw_full_bar(s));
            tma_load_4d(rs + 2 * L::kRawData, &ks_map, 0, tok0 >> 5, 0, page, raw_full_bar(s));
            tma_load_4d(rs + 2 * L::kRawData + L::kRawScale, &vs_map, 0, tok0 >> 5, 0, page, raw_full_bar(s));
            stamp(0, g);
          }
          base += (hi - 1) / kTilesPerPage + 1 - pg0;
          t = pre[r] + hi;
          ++r;
        }
      }
    } else if (warp == kSmWarps + 1) {
      // ------------------------------------------------------------------ MMA issue
      {
        // QK issue (whole warp, warp-uniform operands, one elected lane issues). Descriptors: uniform bases plus
        // compile-time offsets (the start-address field is linear; shared-memory offsets stay inside it).
        constexpr uint32_t id_qk = idesc_f8(64, kTileKeys, 0, kK64);
        const uint64_t dq0 = sdesc(smem_u32(sm + L::kSmQ), 512, kSw64);
        const uint64_t dk0 = sdesc(smem_u32(sm + L::kSmK), 1024, kSw128);
        constexpr uint32_t id_qk4 = idesc_nvf4(128, kTileKeys);
        const uint64_t dkr0 = sdesc(smem_u32(sm + L::kSmRaw), 1024, kSw128);
        int r_q = r_first, k_q = 0, end_q = __shfl_sync(0xffffffffu, min(T1, pre[r_first + 1]), 0);
        const int T0u = __shfl_sync(0xffffffffu, T0, 0);
        const auto qk_tile = [&](int gq) {
          const int t = T0u + gq;
          if (t >= end_q) {  // first tile of the next piece: release the previous Q buffer, wait for this one
            mma_commit_e(q_empty_bar(k_q % L::kQBufs));
            ++r_q;
            ++k_q;
            end_q = __shfl_sync(0xffffffffu, min(T1, pre[r_q + 1]), 0);
            mbar_wait_hint<kAuxHint>(q_full_bar(k_q % L::kQBufs), (k_q / L::kQBufs) & 1);
            // option 16384: and the piece's Q terms (the softmax threads wrote them after the previous piece's last
            // scores, i.e. after every QK MMA reading the previous terms completed)
            if constexpr (kD4b) mbar_wait_hint<kAuxHint>(qt_full_bar(), k_q & 1);
            __syncwarp();
            tc_fence_after();
          }
          const int s = gq % kF8, bq = gq & 1;
          if (lane == 0) stamp(10, gq);
          mbar_wait_hint<kAuxHint>(full_bar(s), (gq / kF8) & 1);  // option 16384: the tile's K scale factors written
          __syncwarp();
          if (lane == 0) stamp(1, gq);
          tc_fence_after();
          if constexpr (kD4b) {
            // M = 128, N = 64 keys, K = 64 dims (one MMA per Q term and head): A = Q term t of the pair's rows, B = head
            // 2c + hh's packed K rows of raw stage gq % kRaw (32 bytes at byte 32 (2c + hh) of each 128-byte row)
            const int sr = gq % kRaw;
            const uint64_t dkr = dkr0 + static_cast<uint64_t>((sr * L::kRawBytes) >> 4);
#pragma unroll
            for (int c = 0; c < kPairs; ++c)
#pragma unroll
              for (int tq = 0; tq < kQTerms; ++tq)
#pragma unroll
                for (int hh = 0; hh < 2; ++hh)
                  mma_nvf4_ts_e(tmem + 128u * bq + 64u * c, tmem + kColQa + 16u * c + 8u * tq,
                                dkr + static_cast<uint64_t>((64 * c + 32 * hh) >> 4), id_qk4,
                                tmem + kColSfaQ + 4u * c + 2u * tq + hh, tmem + kColSfbK + 8u * s + 2u * (2 * c + hh),
                                (tq == 0 && hh == 0) ? 0u : 1u);
            mma_commit_e(s_full_bar(bq));
            mma_commit_e(raw_empty_bar(sr));  // the raw stage's K data read
          } else {
          const uint64_t dq = dq0 + static_cast<uint64_t>(((k_q % L::kQBufs) * L::kQBuf) >> 4);
          const uint64_t dk = dk0 + static_cast<uint64_t>((s * kTileBytes) >> 4);
#pragma unroll
          for (int c = 0; c < kPairs; ++c)
#pragma unroll
            for (int hh = 0; hh < 2; ++hh)
#pragma unroll
              for (int kk = 0; kk < 64 / kKStep; ++kk)
                mma_f8_e(tmem + ((16u * hh) << 16) + 128u * bq + 64u * c,
                         dq + static_cast<uint64_t>(((2 * c + hh) * kRows * 64 + kKStep * kk) >> 4),
                         dk + static_cast<uint64_t>((c * kTileKeys * 128 + 64 * hh + kKStep * kk) >> 4), id_qk, kk);
          mma_commit_e(s_full_bar(bq));
          }
          if (lane == 0) stamp(2, gq);
        };
        mbar_wait_hint<kAuxHint>(q_full_bar(0), 0);
        if constexpr (kD4b) mbar_wait_hint<kAuxHint>(qt_full_bar(), 0);
        __syncwarp();
        tc_fence_after();
        qk_tile(0);
        if (ntiles > 1) qk_tile(1);
        for (int g = 0; g + 2 < ntiles; ++g) {
          mbar_wait_hint<kAuxHint>(s_free_bar(g & 1), (g >> 1) & 1);
          // PV(g) read P(g) over S buffer g & 1 (option 32768: P has its own columns, no wait)
          if constexpr (kPTmem && !kPsep) mbar_wait_hint<kAuxHint>(o_full_bar(g & 1), (g >> 1) & 1);
          __syncwarp();
          tc_fence_after();
          qk_tile(g + 2);
        }
      }
    } else if (warp == kSmWarps + 2) {
      // ------------------------------------------------------------------ PV issue (warp-uniform, elected lane)
      {
        constexpr uint32_t id_pv = idesc_f8(64, 64, 1, kK64);
        constexpr uint32_t kPSbo = kPairs == 2 ? 512 : 1024, kPSw = kPairs == 2 ? kSw64 : kSw128;
        const uint64_t dp0 = sdesc(smem_u32(sm + L::kSmP), kPSbo, kPSw);
        const uint64_t dv0 = sdesc(smem_u32(sm + L::kSmV), 1024, kSw128);
        // option 4096: pieces (requests of the block's range) start the accumulator afresh; the next piece's first PV
        // overwrites it only after the softmax warps loaded the previous piece's (acc_free)
        int kp_pv = 0, end_pv = __shfl_sync(0xffffffffu, min(T1, pre[r_first + 1]), 0), r_pv = r_first;
        const int T0p = __shfl_sync(0xffffffffu, T0, 0);
        for (int g = 0; g < ntiles; ++g) {
          const int s = g % kF8, bb = g & 1;
          bool piece_start = g == 0;
          if constexpr (kPvAcc) {
            if (T0p + g >= end_pv) {
              ++r_pv;
              ++kp_pv;
              end_pv = __shfl_sync(0xffffffffu, min(T1, pre[r_pv + 1]), 0);
              piece_start = true;
            }
            if (piece_start && kp_pv > 0) mbar_wait_hint<kAuxHint>(acc_free_bar(), (kp_pv - 1) & 1);
          }
          if (lane == 0) stamp(11, g);
          mbar_wait_hint<kAuxHint>(p_full_bar(bb), (g >> 1) & 1);
          mbar_wait_hint<kAuxHint>(vfull_bar(s), (g / kF8) & 1);  // the V half of the tile's FP8 stage
          // option bit 5: one output buffer; PV(g) overwrites it once the rescale warps loaded O(g - 1)
          if (kCr && g > 0) mbar_wait_hint<kAuxHint>(o_free_bar((g - 1) & 1), ((g - 1) >> 1) & 1);
          __syncwarp();
          if (lane == 0) stamp(6, g);
          tc_fence_after();
          const uint64_t dp = dp0 + static_cast<uint64_t>((bb * kHeads * kRows * L::kPRowBytes) >> 4);
          const uint64_t dv = dv0 + static_cast<uint64_t>((s * kTileBytes) >> 4);
          constexpr uint32_t kObuf = kCr ? 0u : 128u;  // output buffer stride (one buffer with option bit 5)
          // development probe (kDq bit 21, timing only, wrong outputs): no PV MMAs (the commits and hand-offs stay)
          if constexpr (((kDq >> 21) & 1) != 0) {
          } else if constexpr (kPvAcc) {
            // M = 128 over the pair's 128 lanes, block-scaled: K step j covers keys 32 (j & 1) .. + 31 of head
            // 2c + (j >> 1); A = this tile's P columns 8 (j & 1) .. + 7 of every lane (each lane holds its own head's
            // P); the row scale factor word (column 384 + 2 bb + c) has byte j = the row's factor when j / 2 is the
            // row's head, else 2^-127; the constant B scale factor 1 (columns 392 ..). Option 8192: the same A and
            // scale factors times a constant 1.0 block (N = 16) accumulate the row sums (columns 400 + 16 c ..).
            const uint64_t d1 = sdesc(smem_u32(sm + L::kSmRed), 128, 0);
            if constexpr (kK64) {
              // option 65536: one MMA per head (K = 64 = the tile's keys), scale-factor ID 2 hh = the 16-bit half
              // whose two bytes carry the row's factor when hh is the row's head (the factor word's bytes 2 hh, 2 hh + 1)
#pragma unroll
              for (int c = 0; c < kPairs; ++c)
#pragma unroll
                for (int hh = 0; hh < 2; ++hh) {
                  const uint32_t acc = (piece_start && hh == 0) ? 0u : 1u;
                  mma_bs2_ts_e(tmem + 256u + 64u * c, tmem + p_col(bb, c),
                               dv + static_cast<uint64_t>((c * kTileKeys * 128 + 64 * hh) >> 4),
                               idesc_bs_f8(128, 64, 1, 2 * hh, true), tmem + kColSfa + 2u * bb + c, tmem + kColSfb, acc);
                  if constexpr (kLmma)
                    mma_bs2_ts_e(tmem + kColL + 16u * c, tmem + p_col(bb, c), d1, idesc_bs_f8(128, 16, 1, 2 * hh, true),
                                 tmem + kColSfa + 2u * bb + c, tmem + kColSfb, acc);
                }
            } else {
#pragma unroll
            for (int c = 0; c < kPairs; ++c)
#pragma unroll
              for (int j = 0; j < 4; ++j) {
                const uint32_t acc = (piece_start && j == 0) ? 0u : 1u;
                mma_bs_ts_e(tmem + 256u + 64u * c, tmem + p_col(bb, c) + 8u * (j & 1),
                            dv + static_cast<uint64_t>((c * kTileKeys * 128 + 64 * (j >> 1) + 4096 * (j & 1)) >> 4),
                            idesc_bs_f8(128, 64, 1, j), tmem + kColSfa + 2u * bb + c, tmem + kColSfb, acc);
                if constexpr (kLmma)
                  mma_bs_ts_e(tmem + kColL + 16u * c, tmem + p_col(bb, c) + 8u * (j & 1), d1,
                              idesc_bs_f8(128, 16, 1, j), tmem + kColSfa + 2u * bb + c, tmem + kColSfb, acc);
              }
            }
          } else {
#pragma unroll
          for (int c = 0; c < kPairs; ++c)
#pragma unroll
            for (int hh = 0; hh < 2; ++hh)
#pragma unroll
              for (int kk = 0; kk < kTileKeys / kKStep; ++kk)
                if constexpr (kPTmem)
                  mma_f8_ts_e(tmem + ((16u * hh) << 16) + 256u + kObuf * bb + 64u * c,
                              tmem + ((16u * hh) << 16) + 128u * bb + 64u * c + (kKStep * kk / kKpt) * kKpt +
                                  (kKStep * kk % kKpt) / 4,
                              dv + static_cast<uint64_t>((c * kTileKeys * 128 + 64 * hh + kKStep * 128 * kk) >> 4), id_pv,
                              kk);
                else
                  mma_f8_e(tmem + ((16u * hh) << 16) + 256u + 128u * bb + 64u * c,
                           dp + static_cast<uint64_t>(((2 * c + hh) * kRows * L::kPRowBytes + kKStep * kk) >> 4),
                           dv + static_cast<uint64_t>((c * kTileKeys * 128 + 64 * hh + kKStep * 128 * kk) >> 4), id_pv,
                           kk);
          }
          mma_commit_e(o_full_bar(bb));
          if (lane == 0) stamp(7, g);
          // the FP8 stage of tile g (K read by QK(g), V by PV(g)) may take tile g + kF8 once these MMAs finished
          if (g + kF8 < ntiles) mma_commit_e(empty_bar(s));
        }
      }
    } else if (!kDqR && warp >= kDqWarp0) {
      // ------------------------------------------------------------------ dequantization (warps 12 .. 12 + kDqWarps - 1)
      // Per tile and K / V: kTileKeys packed rows of kParts 16-byte chunks (chunk part = 2 h + half: head h, dims
      // 32 half .. + 32 = scale blocks 2 half and 2 half + 1). Thread d takes part p = d % kParts of the four rows
      // 4 quad .. 4 quad + 3 (quad = d / kParts); each chunk becomes two 16-byte e4m3 chunks of the FP8 stage (pair
      // h / 2, the row, head (h % 2)'s dims at byte 64 (h % 2) + 32 half, 128-byte swizzle: chunk c of row r at
      // c ^ (r % 8)) - the layout of full_attn_wide's tensor-memory-accelerator box. Step i of the four takes row
      // 4 quad + (i ^ flip), flip = the KV-head pair (2 pairs) or quad % 2 (1 pair): neighbouring lanes then read
      // and write different rows of the two pairs, which kept the 128-bit packed reads and e4m3 writes free of bank
      // conflicts in the profiler (the plain lane order did not). The scale rows of a tile are in the 128-byte swizzle
      // as well (tensor-map box [head][tokens / 32][32 x 4 B]), so the four scale words of a quad are one 16-byte load
      // and the heads' words fall in different 16-byte columns.
      const int d = tid - 32 * kDqWarp0;
      constexpr int kParts = kHeads * 2;  // 16-byte chunks per packed token row
      constexpr int kSRows = kTileKeys / 32;  // 128-byte scale rows per head
      // rows per thread: 4 (4 dequantization warps) or 2 (8 warps: a thread takes half of a quad's rows)
      constexpr int kRpt = kParts * kTileKeys / kDqThreads;
      static_assert(kRpt * kDqThreads == kParts * kTileKeys && (kRpt == 4 || kRpt == 2), "rows per thread");
      const int p = d % kParts, grp = d / kParts, h = p >> 1, half = p & 1;
      const int quad = kRpt * grp / 4, tb = kRpt * grp % 4;  // the rows' quad and the first row's place in it
      const int flip = kPairs == 2 ? (p >> 2) : (grp & 1);
      const int c0 = (h & 1) * 4 + half * 2;  // e4m3 output chunk index in the 128-byte row
      const int r128 = h * kSRows + (quad >> 3);
      const int s_off = r128 * 128 + (((quad & 7) ^ (r128 & 7)) << 4);  // the quad's 16 scale bytes
      const uint8_t* const lutb = sm + L::kSmLut;
      // per-thread constants of the steps (rows kRpt grp + (i ^ flip)): packed-row offset, the two e4m3 chunk offsets
      // in the FP8 stage, the row's place in its quad (K: scale word; V: byte of the block words)
      int roff[kRpt], doff0[kRpt], doff1[kRpt], tq[kRpt];
#pragma unroll
      for (int i = 0; i < kRpt; ++i) {
        const int row = kRpt * grp + (i ^ flip);
        roff[i] = row * (kHeads * 32) + h * 32 + half * 16;
        const int db = (h >> 1) * (kTileKeys * 128) + row * 128;
        doff0[i] = db + ((c0 ^ (row & 7)) << 4);
        doff1[i] = db + (((c0 + 1) ^ (row & 7)) << 4);
        tq[i] = tb + (i ^ flip);
      }
      const int ksh = 16 * half;
      int rs = 0, fs = 0;
      uint32_t rph = 0, fbit = 0;  // (g / kRaw) & 1 and (g / kF8) & 1, kept incrementally
      for (int g = 0; g < ntiles; ++g) {
        if (d == 0) stamp(16, g);
        mbar_wait_hint<kAuxHint>(raw_full_bar(rs), rph);
        if (g >= kF8) mbar_wait_hint<kAuxHint>(empty_bar(fs), fbit ^ 1u);  // = ((g / kF8) - 1) & 1
        if (d == 0) stamp(17, g);
        const uint8_t* const raw = sm + L::kSmRaw + rs * L::kRawBytes;
#pragma unroll
        for (int kv = 0; kv < 2; ++kv) {
          if constexpr (kD4b) {
            if (kv == 0) {
              // option 16384: no K dequantization; the tile's K scale words to tensor memory as the QK's B scale
              // factors (scale_vec::4X: lane i, column j of a head = key 32 j + i's four block scales), in every lane
              // quarter: warp w covers lane quarter w % 4 for the heads of pair(s) (w - first dequantization warp) / 4 ..
              const int dw = warp - kDqWarp0;
              const uint32_t lb = (32u * static_cast<uint32_t>(dw & 3)) << 16;
              const uint8_t* const ksr = raw + 2 * L::kRawData;
              // the scale slot was last read by an earlier tile's QK MMAs (their completion observed through the waits
              // above): order this thread's tensor-memory stores after them (PTX ISA tcgen05 thread-sync rules)
              tc_fence_after();
#pragma unroll
              for (int hp = dw >> 2; hp < kPairs; hp += kDqWarps / 4) {
                uint32_t w4[4];
#pragma unroll
                for (int u = 0; u < 4; ++u) {  // u = 2 hh + j: head 2 hp + hh, keys 32 j + lane
                  const int r128 = (2 * hp + (u >> 1)) * kSRows + (u & 1);
                  w4[u] = *reinterpret_cast<const uint32_t*>(ksr + r128 * 128 + ((((lane >> 2) ^ (r128 & 7))) << 4) +
                                                             4 * (lane & 3));
                }
                tmem_st4(tmem + lb + kColSfbK + 8u * fs + 4u * hp, w4[0], w4[1], w4[2], w4[3]);
              }
              tmem_wait_st();
              tc_fence_before();
              mbar_arrive(full_bar(fs));
              if (d == 0) stamp(18, g);
              continue;
            }
          }
          const uint8_t* const src = raw + kv * L::kRawData;
          uint8_t* const dst = sm + (kv ? L::kSmV : L::kSmK) + fs * kTileBytes;
          constexpr int kDqm = kDq & 7;  // development: 1 no stores, 2 no work, 3 constant table, 4 no proxy fence,
                                         // 5 stores of constants only (no loads / arithmetic)
          if constexpr (kDqm == 5) {
#pragma unroll
            for (int i = 0; i < kRpt; ++i) {
              *reinterpret_cast<uint4*>(dst + doff0[i]) = make_uint4(0x38383838u, 0x38383838u, 0x38383838u, 0x38383838u);
              *reinterpret_cast<uint4*>(dst + doff1[i]) = make_uint4(0x38383838u, 0x38383838u, 0x38383838u, 0x38383838u);
            }
          }
          // development bit 20 (timing only, wrong outputs): no dequantization work for the K half (V converted as
          // usual); bounds a kernel whose QK consumes the packed K directly
          constexpr bool kSkipK = ((kDq >> 20) & 1) != 0;
          if (kSkipK && kv == 0) {
          } else if constexpr (kDqm != 2 && kDqm != 5) {
          // scale words of the quad's rows (K: word t = blocks 0..3 of row t; V, four-token interleaved: word b =
          // block b of rows t = 0..3, byte t) -> table byte offsets 8 x scale of blocks 2 half, 2 half + 1 per step
          const uint4 sw = *reinterpret_cast<const uint4*>(raw + 2 * L::kRawData + kv * L::kRawScale + s_off);
          uint32_t o0[kRpt], o1[kRpt];
          if (kv == 0) {
            const uint32_t w[4] = {sw.x, sw.y, sw.z, sw.w};
#pragma unroll
            for (int i = 0; i < kRpt; ++i) {
              uint32_t wi;
              if constexpr (kRpt == 4) {
                wi = flip ? w[i ^ 1] : w[i];
              } else {
                const int t = tq[i];
                wi = (t & 2) ? ((t & 1) ? sw.w : sw.z) : ((t & 1) ? sw.y : sw.x);
              }
              o0[i] = ((wi >> ksh) & 0xFFu) << 3;
              o1[i] = ((wi >> (ksh + 8)) & 0xFFu) << 3;
            }
          } else {
            const uint32_t b0 = half ? sw.z : sw.x, b1 = half ? sw.w : sw.y;
#pragma unroll
            for (int i = 0; i < kRpt; ++i) {
              o0[i] = ((b0 >> (8 * tq[i])) & 0xFFu) << 3;
              o1[i] = ((b1 >> (8 * tq[i])) & 0xFFu) << 3;
            }
          }
          uint4 in[kRpt];
#pragma unroll
          for (int i = 0; i < kRpt; ++i) in[i] = *reinterpret_cast<const uint4*>(src + roff[i]);
#pragma unroll
          for (int i = 0; i < kRpt; ++i) {
            uint2 t0, t1;
            if constexpr (kDqm == 3) {
              t0 = make_uint2(0x38302800u, 0x50484440u);
              t1 = t0;
            } else {
              t0 = *reinterpret_cast<const uint2*>(lutb + o0[i]);
              t1 = *reinterpret_cast<const uint2*>(lutb + o1[i]);
            }
            const uint2 a = lut8(in[i].x, t0), bq = lut8(in[i].y, t0);  // 16 values of block 2 half
            const uint2 c = lut8(in[i].z, t1), e = lut8(in[i].w, t1);   // 16 values of block 2 half + 1
            if constexpr (kDqm == 1) {
              asm volatile("" ::"r"(a.x), "r"(a.y), "r"(bq.x), "r"(bq.y), "r"(c.x), "r"(c.y), "r"(e.x), "r"(e.y));
              continue;
            }
            *reinterpret_cast<uint4*>(dst + doff0[i]) = make_uint4(a.x, a.y, bq.x, bq.y);
            *reinterpret_cast<uint4*>(dst + doff1[i]) = make_uint4(c.x, c.y, e.x, e.y);
          }
          }  // kDqm != 2
          // generic-proxy reads of the raw stage and writes of the FP8 stage, then the arrivals: the raw stage is
          // refilled by the tensor memory accelerator and the FP8 stage is read by the MMAs (both async proxy). One
          // fence per tile after the V half, then the K half (QK of the tile waits only on it), the V half and the raw
          // stage (development bit 18: a fence and an arrival after each half, K first)
          constexpr bool kOneFence = ((kDq >> 18) & 1) == 0;
          if (kOneFence && kv == 0) continue;
          if constexpr ((kDq & 7) != 4) fence_proxy_async();  // (development mode 4 omits it: timing only, unsafe)
          if (kv == 0) {
            mbar_arrive(full_bar(fs));
            if (d == 0) stamp(18, g);
          } else {
            if (kOneFence && !kD4b) mbar_arrive(full_bar(fs));
            mbar_arrive(vfull_bar(fs));
            mbar_arrive(raw_empty_bar(rs));
            if (d == 0) stamp(19, g);
          }
        }
        if (++rs == kRaw) {
          rs = 0;
          rph ^= 1u;
        }
        if (++fs == kF8) {
          fs = 0;
          fbit ^= 1u;
        }
      }
    }  // warp 11: idle
    } else {
      setmaxnreg_inc<kCr ? kSmRegsCr : kSmRegs>();
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
      // four e4m3 probabilities (key order) in one word; option bit 1 joins the two conversions in one statement
      const auto pack4 = [](float a, float b, float c, float d) -> uint32_t {
        if constexpr (kPackJoined) {
          return e4m3x4_joined(a, b, c, d);
        } else {
          return e4m3x4(a, b, c, d);
        }
      };
      int g = 0, kp = 0;  // kp: pieces done (option bit 5 accumulator hand-off parity)
      int r = r_first;
      if constexpr (kPvAcc) {
        // the constant B scale factor 1.0 (UE8M0 0x7f) in columns 392 .. 395 of every lane, before the first P hand-off
        // (whose tcgen05.wait::st + fence + arrival orders it before the first PV)
        if (c == 0) tmem_st4(tmem + lane_base + kColSfb, 0x7f7f7f7fu, 0x7f7f7f7fu, 0x7f7f7f7fu, 0x7f7f7f7fu);
      }
      for (int t = T0; t < T1; ++r, ++kp) {
        const int lo = t - pre[r], hi = min(T1, pre[r + 1]) - pre[r];
        const int n_r = pre[r + 1] - pre[r];
        const int seq_len = p.seq_lens[r];
        const int pos = seq_len - kTokens + tok;
        float2 o2[kDims / 2];
#pragma unroll
        for (int d = 0; d < kDims / 2; ++d) o2[d] = make_float2(0.0f, 0.0f);
        float m = -INFINITY, l = 0.0f, ar_prev = 0.0f, at_prev = 0.0f;
        float sl2 = p.scale_log2;  // the row's score scale (log2 units)
        if constexpr (kD4b) {
          // option 16384: this row's Q terms for the piece (see kD4b). The row's 64 e4m3 values from the piece's Q buffer
          // (64-byte swizzle: chunk q of row r at q ^ ((r >> 1) & 3)), once the tensor memory accelerator wrote it.
          if (tid == 0) stamp(20, kp);
          mbar_wait_warp(q_full_bar(kp % L::kQBufs), (kp / L::kQBufs) & 1);
          if (tid == 0) stamp(21, kp);
          const uint8_t* const qrow = sm + L::kSmQ + (kp % L::kQBufs) * L::kQBuf + (head * kRows + row) * 64;
          float rq[64];
#pragma unroll
          for (int q = 0; q < 4; ++q) {
            const uint4 w = *reinterpret_cast<const uint4*>(qrow + ((q ^ ((row >> 1) & 3)) << 4));
            e4m3x4_to_f(w.x, rq + 16 * q);
            e4m3x4_to_f(w.y, rq + 16 * q + 4);
            e4m3x4_to_f(w.z, rq + 16 * q + 8);
            e4m3x4_to_f(w.w, rq + 16 * q + 12);
          }
          float amax = 0.0f;
#pragma unroll
          for (int j = 0; j < 64; ++j) amax = fmaxf(amax, fabsf(rq[j]));
          // the row's power of two 2^ge >= amax / (6 x 448), the smallest (amax / 2688 = (an integer < 16) / 21 x 2^k for
          // e4m3 inputs: never within rounding of a power of two, so the product with the rounded reciprocal decides the
          // same exponent), folded into the score scale; exponent arithmetic only (amax = 0: 2^0)
          const float y = amax * (1.0f / 2688.0f);
          const int ge = y > 0.0f ? ((__float_as_int(y) >> 23) & 0xFF) - 126 : 0;
          const float ginv = __int_as_float((127 - ge) << 23);
          sl2 = p.scale_log2 * __int_as_float((127 + ge) << 23);
#pragma unroll
          for (int j = 0; j < 64; ++j) rq[j] *= ginv;
#pragma unroll
          for (int tq = 0; tq < kQTerms; ++tq) {
            uint32_t qa[8], sfw = 0u;
#pragma unroll
            for (int blk = 0; blk < 4; ++blk) {
              float bm = 0.0f;
#pragma unroll
              for (int k = 0; k < 16; ++k) bm = fmaxf(bm, fabsf(rq[16 * blk + k]));
              uint32_t sb = f_to_e4m3(bm * (1.0f / 6.0f));
              float sv = 1.0f;
              if (sb == 0u) {
                sb = 0x38u;  // 1.0: a block that is zero after rounding
              } else {
                sv = e4m3_to_f(sb);
              }
              // E2M1 round-to-nearest-even of r / sv without a division: |r| against the exact midpoints x sv (each
              // product exact in fp32); a tie goes to the even code (0, 1, 2, 4, 4, 6, 6 at 0.25, 0.75, 1.25, 1.75, 2.5,
              // 3.5, 5), values beyond 6 sv saturate at 6. The magnitude in units of sv: 0.5 per step up to 2, then 1, 1, 2.
              const float t0 = 0.25f * sv, t1 = 0.75f * sv, t2 = 1.25f * sv, t3 = 1.75f * sv, t4 = 2.5f * sv,
                          t5 = 3.5f * sv, t6 = 5.0f * sv;
              float v[16];
              if constexpr (kFastQ) {
                // option 8388608: r x (the rounded reciprocal of sv), the conversion's round-to-nearest-even (it can
                // differ from the exact rounding of r / sv only within an ulp of a midpoint), then the exact values back
                const float inv = __frcp_rn(sv);
#pragma unroll
                for (int k = 0; k < 16; ++k) v[k] = rq[16 * blk + k] * inv;
                qa[2 * blk] = f8_to_e2m1x8(v);
                qa[2 * blk + 1] = f8_to_e2m1x8(v + 8);
                e2m1x8_to_f(qa[2 * blk], v);
                e2m1x8_to_f(qa[2 * blk + 1], v + 8);
#pragma unroll
                for (int k = 0; k < 16; ++k) rq[16 * blk + k] = fmaf(-v[k], sv, rq[16 * blk + k]);  // the residual
                sfw |= sb << (8 * blk);
                continue;
              }
#pragma unroll
              for (int k = 0; k < 16; ++k) {
                const float r = rq[16 * blk + k], a = fabsf(r);
                const float mh = (a > t0 ? 0.5f : 0.0f) + (a >= t1 ? 0.5f : 0.0f) + (a > t2 ? 0.5f : 0.0f) +
                                 (a >= t3 ? 0.5f : 0.0f) + (a > t4 ? 1.0f : 0.0f) + (a >= t5 ? 1.0f : 0.0f) +
                                 (a > t6 ? 2.0f : 0.0f);
                v[k] = copysignf(mh, r);                          // on the E2M1 grid: the conversion is exact
                rq[16 * blk + k] = fmaf(-v[k], sv, r);            // the residual (exact)
              }
              qa[2 * blk] = f8_to_e2m1x8(v);
              qa[2 * blk + 1] = f8_to_e2m1x8(v + 8);
              sfw |= sb << (8 * blk);
            }
            tmem_st8(tmem + lane_base + kColQa + 16u * c + 8u * tq, qa);
            // scale factors of the MMAs of head 2c and 2c + 1: this row's word in its own head's column, 0 in the other
            tmem_st2(tmem + lane_base + kColSfaQ + 4u * c + 2u * tq, hh == 0 ? sfw : 0u, hh == 1 ? sfw : 0u);
          }
          tmem_wait_st();
          tc_fence_before();
          __syncwarp();
          if (lane == 0) mbar_arrive(qt_full_bar());
          if (tid == 0) stamp(22, kp);
        }
        // rescale-and-add of an output tile into o (packed fp32 pairs: the same two roundings as fmaf(o, ar, v * at))
        const auto combine = [&](int gg, bool last = false) {
          // development probe (kDq bit 19, timing only, wrong outputs): no per-tile combine (no output wait, loads or
          // arithmetic) except the piece's last tile (its wait keeps the output MMAs complete before the piece ends);
          // bounds what an accumulate-in-place output would save
          if constexpr (((kDq >> 19) & 1) != 0) {
            if (!last) return;
          }
          const int bp = gg & 1;
          mbar_wait_warp(o_full_bar(bp), (gg >> 1) & 1);
          if (tid == 0) stamp(8, gg);
          tc_fence_after();
          const float2 a2 = make_float2(ar_prev, ar_prev), t2 = make_float2(at_prev, at_prev);
          if constexpr (kOneWaitO) {
            // option bit 9: every output column of the tile loaded before one wait; the same operations per element
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
          mbar_wait_warp(s_full_bar(bb), (g >> 1) & 1);
          if (tid == 0) stamp(3, g);

          tc_fence_after();
          float x[kKpt];
          float rred[kKpt / 32];  // skeleton option bit 0: maxima of the loaded score columns, from the loads themselves
#pragma unroll
          for (int j = 0; j < kKpt; j += 32) {
            if constexpr (kRedMax) {
              rred[j / 32] = tmem_ld32_max(tmem + lane_base + 128u * bb + s_col + j, *reinterpret_cast<float(*)[32]>(x + j));
            } else {
              tmem_ld32(tmem + lane_base + 128u * bb + s_col + j, *reinterpret_cast<float(*)[32]>(x + j));
            }
          }
          tmem_wait_ld();  // the loaded scores and the reductions are read only after this wait
          tc_fence_before();
          if (tid == 0) stamp(4, g);
          __syncwarp();
          if (lane == 0) mbar_arrive(s_free_bar(bb));  // score buffer bb may take the QK of tile g + 2
          // tile edge: keys after the row's position do not take part. The test uses the smallest position of the request
          // (token 0), so it is warp-uniform: interior tiles branch around the masking instead of predicating it per key.
          const bool edge = key0 + kKpt - 1 > seq_len - kTokens;
          if (edge) {
#pragma unroll
            for (int j = 0; j < kKpt; ++j)
              if (key0 + j > pos) x[j] = -INFINITY;
          }
          if constexpr (kMode == 2) {  // development probe: no softmax arithmetic, P = 0
#pragma unroll
            for (int j = 0; j < kKpt; ++j) x[j] = 0.0f;
          }
          // maximum of the raw scores (the scale is positive, so max(s) * scale == max(s * scale)); with option bit 0 an
          // interior tile (no masked key) takes the loads' reduction, the same maximum of the same values
          float smax;
          if (kRedMax && !edge) {
            smax = rred[0];
#pragma unroll
            for (int j = 1; j < kKpt / 32; ++j) smax = fmaxf(smax, rred[j]);
          } else {
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
          const float mtile = smax == -INFINITY ? -INFINITY : smax * sl2;
          // option 4096: the tile's probabilities are quantized against kt = ceil(mtile) (an integer), and its scale
          // factor 2^(kt - m) (m = the row's reference, an integer) puts them on the accumulator's scale
          float kt = 0.0f, ft = 0.0f;
          uint32_t sfa_word = 0u;
          if constexpr (kPvAcc) {
            kt = mtile == -INFINITY ? -INFINITY : ceilf(mtile);
            const bool need = m != -INFINITY && kt != -INFINITY && kt > m + 64.0f;
            if (m == -INFINITY) m = kt;  // the piece's first tile with a finite maximum (earlier products are zeros)
            if (__any_sync(0xffffffffu, need)) {
              // re-reference: the accumulator rows (and row sums) of the rows whose maximum jumped are scaled by
              // 2^(m - kt) (exact); the other rows of the warp by 1. The output MMAs of tile g - 1 (the last ones
              // of this piece so far) are complete first; PV(g) starts only after this tile's hand-off below.
              mbar_wait_warp(o_full_bar((g - 1) & 1), ((g - 1) >> 1) & 1);
              tc_fence_after();
              // (a jump beyond 126 binades scales by 2^-126: the earlier terms are below that relative to the new ones)
              const float f = need ? __int_as_float((127 + max(static_cast<int>(m - kt), -126)) << 23) : 1.0f;
              if constexpr (kSlimRef) {
                // option 131072: 16 columns at a time, not unrolled (the same products)
#pragma unroll 1
                for (uint32_t d0 = 0; d0 < 64u; d0 += 16u) {
                  uint32_t ow[16];
                  tmem_ld16(tmem + lane_base + 256u + 64u * c + d0, ow);
                  tmem_wait_ld();
#pragma unroll
                  for (int d = 0; d < 16; ++d) ow[d] = __float_as_uint(__uint_as_float(ow[d]) * f);
                  tmem_st16(tmem + lane_base + 256u + 64u * c + d0, ow);
                }
                if constexpr (kLmma) {
                  const float lv0 = __uint_as_float(tmem_ld1(tmem + lane_base + kColL + 16u * c));
                  tmem_wait_ld();
                  tmem_st1(tmem + lane_base + kColL + 16u * c, __float_as_uint(lv0 * f));
                }
              } else {
              float ov[64];
              tmem_ld32(tmem + lane_base + 256u + 64u * c, *reinterpret_cast<float(*)[32]>(ov));
              tmem_ld32(tmem + lane_base + 256u + 64u * c + 32u, *reinterpret_cast<float(*)[32]>(ov + 32));
              tmem_wait_ld();
              uint32_t ow[32];
#pragma unroll
              for (int d = 0; d < 32; ++d) ow[d] = __float_as_uint(ov[d] * f);
              tmem_st32(tmem + lane_base + 256u + 64u * c, ow);
#pragma unroll
              for (int d = 0; d < 32; ++d) ow[d] = __float_as_uint(ov[32 + d] * f);
              tmem_st32(tmem + lane_base + 256u + 64u * c + 32u, ow);
              }
              if constexpr (kLmma && !kSlimRef) {
                // the row-sum accumulator (column 400 + 16 c; the only one of its 16 equal columns read back)
                float lv[32];
                tmem_ld32(tmem + lane_base + kColL + 16u * c, lv);
                tmem_wait_ld();
                tmem_st1(tmem + lane_base + kColL + 16u * c, __float_as_uint(lv[0] * f));
              }
              if (need) {
                l *= f;
                m = kt;
              }
            }
            if (kt != -INFINITY) {
              const int n = max(static_cast<int>(kt - m), -127);  // <= 64
              ft = n >= -126 ? __int_as_float((127 + n) << 23) : __int_as_float(0x00400000);
              // bytes 2 hh and 2 hh + 1 (the K steps of this row's head) carry the factor; the other head's K steps 2^-127
              sfa_word = static_cast<uint32_t>(127 + n) * (hh == 0 ? 0x00000101u : 0x01010000u);
            }
          }
          const float mu = mtile == -INFINITY ? 0.0f : (kPvAcc ? kt : mtile);
          const float mnew = kPvAcc ? m : fmaxf(m, mtile);
          const float ar = kPvAcc || m == -INFINITY ? 0.0f : fast_exp2(m - mnew);
          const float at = kPvAcc || mtile == -INFINITY ? 0.0f : fast_exp2(mtile - mnew);
          // 448 * 2^y as 2^(y + log2 448): probabilities come out pre-scaled for the e4m3 conversion, and the row sums are
          // kept in the same (x 448) units; the output scale divides by them (bmm2_scale / l448)
          const float nmu448 = kLog2PScale - mu;
          const float2 sc = make_float2(sl2, sl2), nmu = make_float2(nmu448, nmu448);
          float2 ls[2] = {make_float2(0.0f, 0.0f), make_float2(0.0f, 0.0f)};
          // option 4194304 (with 8192: the row sums come from the MMAs): 2^y in half precision, two per instruction, and
          // the e4m3 probabilities converted from the f16 pairs; y itself in fp32 as above
          uint32_t hp[kF16Exp ? kKpt / 2 : 1];
          if constexpr (kF16Exp) {
#pragma unroll
            for (int j = 0; j < kKpt; j += 2) hp[j / 2] = exp2_f16x2(fma2(make_float2(x[j], x[j + 1]), sc, nmu));
          }
#pragma unroll
          for (int j = 0; j < kKpt && kMode != 2 && !kF16Exp; j += 2) {
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
          if constexpr (kCr) {
            // (ar, at) of this row and tile for the rescale warps; buffer bb was read for tile g - 2 (arat_free)
            if (g >= 2) mbar_wait(arat_free_bar(bb), ((g >> 1) - 1) & 1);
            arat[bb * kRecRows + rec_row] = make_float2(ar, at);
            mbar_arrive(arat_full_bar(bb));
          }
          // probabilities -> tensor memory: this thread's kKpt e4m3 values (key order, four per column) over the first
          // kKpt / 4 columns of its own scores of buffer bb (A operand of the PV MMA; the QK of tile g + 2 rewrites them
          // only after the PV commit of tile g). Development bit 17: P buffer bb in shared memory instead - kPairs 2:
          // rows of 64 B in the 64-byte swizzle (chunk q of row r at q ^ (r / 2 % 4)); kPairs 1: rows of 128 B in the
          // 128-byte swizzle (chunk q of row r at q ^ (r % 8)); buffer bb was last read by the MMAs of tile g - 2.
          if constexpr (kPTmem) {
            uint32_t pw[16];
            if constexpr (kF16Exp) {
#pragma unroll
              for (int j = 0; j < 16; ++j) pw[j] = e4m3x4_from_f16x2(hp[2 * j], hp[2 * j + 1]);
            } else {
#pragma unroll
            for (int j = 0; j < 16; ++j) pw[j] = pack4(x[4 * j], x[4 * j + 1], x[4 * j + 2], x[4 * j + 3]);
            }
            // option bit 8: the previous tile's combine runs here, before this tile's probabilities are stored and PV(g)
            // starts reading them from tensor memory (independent values; the same per-element operations and order)
            if constexpr (kEarlyComb) {
              if (i > lo) combine(g - 1);
            }
            if constexpr (kPsep) {
              // option 32768: P buffer bb (own columns) was last read by PV(g - 2): wait for its commit first (it is
              // the latest commit on that barrier, so the parity cannot alias)
              if (g >= 2) {
                mbar_wait_warp(o_full_bar(bb), ((g - 2) >> 1) & 1);
                tc_fence_after();
              }
            }
            tmem_st16(tmem + lane_base + (kPsep ? p_col(bb, c) : 128u * bb + s_col), pw);
            // option 4096: the row's scale-factor word of this tile (one byte per PV K step)
            if constexpr (kPvAcc) tmem_st1(tmem + lane_base + kColSfa + 2u * bb + c, sfa_word);
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
          }  // kPTmem
          tc_fence_before();
          if (tid == 0) stamp(5, g);
          __syncwarp();
          if (lane == 0) mbar_arrive(p_full_bar(bb));
          // this thread's half of the row sum (the two halves are added once, at the end of the piece)
          if constexpr (!kCr && !kEarlyComb && !kPvAcc) {
            if (i > lo) combine(g - 1);
          }
          if constexpr (kPvAcc) {
            // the row sum on the accumulator's scale (ft = 2^(kt - m), exact); option 8192 takes it from the MMAs
            if constexpr (!kLmma) l = fmaf(lpart, ft, l);
          } else {
            l = l * ar + lpart * at;
          }
          ar_prev = ar;
          at_prev = at;
        }
        if constexpr (kCr) {
          // the piece's output accumulator from tensor memory (the rescale warps' last tile of the piece is complete)
          mbar_wait_warp(cr_done_bar(), kp & 1);
          tc_fence_after();
#pragma unroll
          for (int d0 = 0; d0 < kDims; d0 += 32) {
            float v[32];
            tmem_ld32(tmem + lane_base + 384u + o_col + d0, v);
            tmem_wait_ld();
#pragma unroll
            for (int d = 0; d < 16; ++d) o2[d0 / 2 + d] = make_float2(v[2 * d], v[2 * d + 1]);
          }
          tc_fence_before();
          __syncwarp();
          if (lane == 0) mbar_arrive(acc_free_bar());  // the next piece's first rescale may overwrite it
        } else if constexpr (kPvAcc) {
          // the piece's accumulator (this lane's row) once the output MMAs of its last tile completed; then the next
          // piece's first PV may overwrite it (acc_free: one arrival per softmax warp)
          mbar_wait_warp(o_full_bar((g - 1) & 1), ((g - 1) >> 1) & 1);
          tc_fence_after();
#pragma unroll
          for (int d0 = 0; d0 < kDims; d0 += 32) {
            float v[32];
            tmem_ld32(tmem + lane_base + 256u + 64u * c + d0, v);
            tmem_wait_ld();
#pragma unroll
            for (int d = 0; d < 16; ++d) o2[d0 / 2 + d] = make_float2(v[2 * d], v[2 * d + 1]);
          }
          if constexpr (kLmma) {
            // option 8192: the row sum accumulated by the MMAs (x 448 units, on the accumulator's scale) replaces the
            // register sum of the unquantized probabilities
            float lv[32];
            tmem_ld32(tmem + lane_base + kColL + 16u * c, lv);
            tmem_wait_ld();
            l = lv[0];
          }
          tc_fence_before();
          __syncwarp();
          if (lane == 0) mbar_arrive(acc_free_bar());
        } else {
          combine(g - 1, true);
        }
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
          const int slot = (pre[r] < T0) ? 0 : 1;
          float* const rec = p.partial + ((static_cast<int64_t>(b) * 2 + slot) * kRecRows + rec_row) * kRec;
#pragma unroll
          for (int d = 0; d < kDims / 2; d += 2)
            *reinterpret_cast<float4*>(rec + kDims * half + 2 * d) = make_float4(o2[d].x, o2[d].y, o2[d + 1].x, o2[d + 1].y);
          if (half == 0) *reinterpret_cast<float2*>(rec + kHeadDim) = make_float2(m, l);
          // merged by kv4_attn_wide_merge_kernel (launched after this kernel)
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
// Requests whose tiles span more than one block of the main kernel: grid (kHeads * 64 / kMergeRows, R), 256 threads;
// block (x, r) merges rows [16 x, 16 x + 16) of request r, 16 threads per row (thread: 4 output dims). The work split is
// recomputed from seq_lens exactly as the main kernel computes it (same G, same W). Records of the pieces j = b0..b1
// (b0: slot 1, others: slot 0), in block order: M = max m_j, O = sum exp2(m_j - M) o_j, L = sum exp2(m_j - M) l_j;
// out = bf16(O * (out_scale / L)). Latency rounds: one for the tile range (every lane issues all its seq_lens loads first),
// one for the pieces' maxima (spread over the 16 threads of a row) together with the first 8 pieces' outputs, one per
// further 8 pieces. The arithmetic (maximum, weights, fused multiply-adds in block order) is that of the earlier merge.
constexpr int kMergeRows = 16;
constexpr int kMergeBatch = 8;
template <int kPairs, bool kUsePDL>
__global__ void __launch_bounds__(256) kv4_attn_wide_merge_kernel(const Params p, int grid_blocks) {
  constexpr int kHeads = 2 * kPairs, kTileKeys = 128 / kPairs, kRecRows = kHeads * kRows;
  __shared__ int32_t range[3];  // tiles of requests < r, of requests <= r, of all requests
  if constexpr (kUsePDL) asm volatile("griddepcontrol.wait;\n" ::: "memory");
  const int R = p.num_reqs, r = blockIdx.y, tid = threadIdx.x, lane = tid & 31;
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
  if (b0 == b1) return;  // the request lies in one block: written by the main kernel
  const int row = kMergeRows * blockIdx.x + (tid >> 4), part = tid & 15;  // row of the request's records, 4-dim part
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
  float acc[4] = {0.0f, 0.0f, 0.0f, 0.0f};
  float lsum = 0.0f;
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
  const int head = row >> 6, rr = row & 63, tok = rr >> 3, qh = head * kQPerKv + (rr & 7);
  __nv_bfloat16* const orow =
      p.out + ((static_cast<int64_t>(r) * kTokens + tok) * p.num_q_heads + qh) * kHeadDim + 4 * part;
  __nv_bfloat162 h0 = __floats2bfloat162_rn(acc[0] * f, acc[1] * f);
  __nv_bfloat162 h1 = __floats2bfloat162_rn(acc[2] * f, acc[3] * f);
  uint2 w;
  w.x = *reinterpret_cast<uint32_t*>(&h0);
  w.y = *reinterpret_cast<uint32_t*>(&h1);
  *reinterpret_cast<uint2*>(orow) = w;
}

}  // namespace k4w

// ---------------------------------------------------------------------------------------------- host
namespace k4w_host {

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

// Tensor map over a packed pool [slots, row_bytes] (bytes), box = whole rows x `rows`, no swizzle (sw128: the 128-byte
// swizzle, rows of 128 bytes, for a K operand read by the MMA directly).
inline CUtensorMap packed_map(const void* base, int64_t slots, int64_t row_bytes, int rows, bool sw128 = false) {
  CUtensorMap map;
  const cuuint64_t dims[2] = {static_cast<cuuint64_t>(row_bytes), static_cast<cuuint64_t>(slots)};
  const cuuint64_t strides[1] = {static_cast<cuuint64_t>(row_bytes)};
  const cuuint32_t box[2] = {static_cast<cuuint32_t>(row_bytes), static_cast<cuuint32_t>(rows)};
  const cuuint32_t elem[2] = {1, 1};
  const CUresult r = encode_fn()(&map, CU_TENSOR_MAP_DATA_TYPE_UINT8, 2, const_cast<void*>(base), dims, strides, box,
                                 elem, CU_TENSOR_MAP_INTERLEAVE_NONE,
                                 sw128 ? CU_TENSOR_MAP_SWIZZLE_128B : CU_TENSOR_MAP_SWIZZLE_NONE,
                                 CU_TENSOR_MAP_L2_PROMOTION_L2_256B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
  sglang::host::RuntimeCheck(r == CUDA_SUCCESS, "cuTensorMapEncodeTiled (packed pool) failed: ", static_cast<int>(r));
  return map;
}

// Tensor map over a scale pool [pages, heads, 128 tokens, 4 B] as 32-bit elements {32 tokens, 4 token groups of 32,
// heads, pages}; box = {32, tokens of a tile / 32, all heads, 1 page} -> shared [head][tile tokens / 32][32 x 4 B], each
// 128-byte row in the 128-byte swizzle.
inline CUtensorMap scale_map(const void* base, int64_t pages, int64_t heads, int tile_tokens) {
  CUtensorMap map;
  const cuuint64_t dims[4] = {32, 4, static_cast<cuuint64_t>(heads), static_cast<cuuint64_t>(pages)};
  const cuuint64_t strides[3] = {128, 512, static_cast<cuuint64_t>(heads * 512)};
  const cuuint32_t box[4] = {32, static_cast<cuuint32_t>(tile_tokens / 32), static_cast<cuuint32_t>(heads), 1};
  const cuuint32_t elem[4] = {1, 1, 1, 1};
  const CUresult r = encode_fn()(&map, CU_TENSOR_MAP_DATA_TYPE_UINT32, 4, const_cast<void*>(base), dims, strides, box,
                                 elem, CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B,
                                 CU_TENSOR_MAP_L2_PROMOTION_L2_256B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
  sglang::host::RuntimeCheck(r == CUDA_SUCCESS, "cuTensorMapEncodeTiled (scale pool) failed: ", static_cast<int>(r));
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

}  // namespace k4w_host

template <int kPairs, int kF8, int kRaw, bool kUsePDL, int kTpr, int kDq, int kSkel = 0>
struct Kv4AttnWideKernel {
  static void run(const tvm::ffi::TensorView q,           // [R*8, Hq, 64] e4m3, heads/dims contiguous
                  const tvm::ffi::TensorView k_pool,      // [slots, Hkv*32] uint8 (packed E2M1), rows contiguous
                  const tvm::ffi::TensorView v_pool,      // [slots, Hkv*32]
                  const tvm::ffi::TensorView k_scales,    // [pages, Hkv, 128, 4] uint8 (E4M3), contiguous
                  const tvm::ffi::TensorView v_scales,    // [pages, Hkv, 128, 4] (four-token interleaved)
                  const tvm::ffi::TensorView page_table,  // [R, pages] int32, inner contiguous
                  const tvm::ffi::TensorView seq_lens,    // [R] int32
                  const tvm::ffi::TensorView out,         // [R*8, Hq, 64] bf16, contiguous
                  const tvm::ffi::TensorView partial,     // fp32 workspace, >= blocks * 2 * Hkv * 64 * 68
                  const tvm::ffi::TensorView counters,    // int32 [>= 256], zero
                  int64_t blocks, double bmm1_scale, double bmm2_scale) {
    using namespace sglang::host;
    using L = k4w::Layout<kPairs, kF8, kRaw, kTpr, (((kDq >> 17) & 1) || kPairs != 2) ? 1 : 0>;
    RuntimeCheck(q.dim() == 3 && q.size(2) == k4w::kHeadDim && q.dtype().code == DLDataTypeCode::kDLFloat8_e4m3fn &&
                     q.stride(2) == 1 && q.stride(1) == k4w::kHeadDim && q.stride(0) % 16 == 0,
                 "q must be e4m3 [tokens, heads, 64], heads and dims contiguous");
    const auto is_u8 = [](const tvm::ffi::TensorView& t) { return t.dtype().bits == 8; };
    RuntimeCheck(is_u8(k_pool) && is_u8(v_pool) && k_pool.dim() == 2 && v_pool.dim() == 2 && k_pool.stride(1) == 1 &&
                     v_pool.stride(1) == 1 && k_pool.stride(0) == k_pool.size(1) && v_pool.stride(0) == v_pool.size(1) &&
                     k_pool.size(1) == v_pool.size(1) && k_pool.size(0) == v_pool.size(0),
                 "packed pools must be contiguous uint8 [slots, Hkv*32]");
    const int64_t hkv = k_pool.size(1) / (k4w::kHeadDim / 2);
    const int64_t pages = k_pool.size(0) / k4w::kPage;
    RuntimeCheck(k_pool.size(0) % k4w::kPage == 0, "pool slots must be whole pages");
    RuntimeCheck(is_u8(k_scales) && is_u8(v_scales) && k_scales.dim() == 4 && v_scales.dim() == 4 &&
                     k_scales.stride(3) == 1 && v_scales.stride(3) == 1 && k_scales.stride(2) == 4 &&
                     v_scales.stride(2) == 4 && k_scales.stride(1) == 512 && v_scales.stride(1) == 512 &&
                     k_scales.stride(0) == hkv * 512 && v_scales.stride(0) == hkv * 512 && k_scales.size(0) >= pages &&
                     v_scales.size(0) >= pages && k_scales.size(1) == hkv && v_scales.size(1) == hkv &&
                     k_scales.size(2) == k4w::kPage && v_scales.size(2) == k4w::kPage && k_scales.size(3) == 4 &&
                     v_scales.size(3) == 4,
                 "scale pools must be contiguous uint8 [pages, Hkv, 128, 4]");
    RuntimeCheck(out.dtype().code == DLDataTypeCode::kDLBfloat && out.dtype().bits == 16, "out must be bf16");
    const int64_t m = out.size(0), hq = out.size(1), reqs = m / k4w::kTokens;
    RuntimeCheck(m % k4w::kTokens == 0 && hkv == L::kHeads && hkv * k4w::kQPerKv == hq && q.size(0) == m &&
                     q.size(1) == hq,
                 "head layout (KV heads per rank must match the kernel)");
    RuntimeCheck(reqs <= k4w::kMaxReqs, "too many requests");
    RuntimeCheck(out.dim() == 3 && out.size(2) == k4w::kHeadDim && out.stride(2) == 1 && out.stride(1) == k4w::kHeadDim &&
                     out.stride(0) == hq * k4w::kHeadDim,
                 "out layout");
    RuntimeCheck(seq_lens.dim() == 1 && seq_lens.size(0) == reqs && seq_lens.stride(0) == 1 &&
                     seq_lens.dtype().code == DLDataTypeCode::kDLInt && seq_lens.dtype().bits == 32,
                 "seq_lens [R] int32");
    RuntimeCheck(page_table.dim() == 2 && page_table.size(0) >= reqs && page_table.stride(1) == 1 &&
                     page_table.dtype().code == DLDataTypeCode::kDLInt && page_table.dtype().bits == 32,
                 "page_table [R, pages] int32");
    RuntimeCheck(blocks >= 1 && partial.dtype().code == DLDataTypeCode::kDLFloat && partial.dtype().bits == 32 &&
                     partial.numel() >= blocks * 2 * hkv * k4w::kRows * k4w::kRec,
                 "partial workspace too small");
    RuntimeCheck(counters.dtype().code == DLDataTypeCode::kDLInt && counters.dtype().bits == 32 &&
                     counters.numel() >= k4w::kMaxReqs,
                 "counters int32 [>= 256]");
    const auto aligned16 = [](const void* ptr) { return reinterpret_cast<uintptr_t>(ptr) % 16 == 0; };
    RuntimeCheck(aligned16(q.data_ptr()) && aligned16(k_pool.data_ptr()) && aligned16(v_pool.data_ptr()) &&
                     aligned16(k_scales.data_ptr()) && aligned16(v_scales.data_ptr()) && aligned16(out.data_ptr()) &&
                     aligned16(partial.data_ptr()),
                 "16-byte alignment");

    k4w::Params params{};
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
    const CUtensorMap qm = k4w_host::q_map(q.data_ptr(), m, hq, q.stride(0));
    // option 16384: the packed K rows land in the 128-byte swizzle (the QK MMA reads them directly)
    const CUtensorMap kdm = k4w_host::packed_map(k_pool.data_ptr(), k_pool.size(0), k_pool.size(1), L::kTileKeys,
                                                 (kSkel & 16384) != 0);
    const CUtensorMap vdm = k4w_host::packed_map(v_pool.data_ptr(), v_pool.size(0), v_pool.size(1), L::kTileKeys);
    const CUtensorMap ksm = k4w_host::scale_map(k_scales.data_ptr(), k_scales.size(0), hkv, L::kTileKeys);
    const CUtensorMap vsm = k4w_host::scale_map(v_scales.data_ptr(), v_scales.size(0), hkv, L::kTileKeys);

    constexpr auto kernel = k4w::kv4_attn_wide_kernel<kPairs, kF8, kRaw, kUsePDL, kTpr, kDq, kSkel>;
    static bool configured = false;
    if (!configured) {
      int dev = 0, optin = 0;
      RuntimeDeviceCheck(::cudaGetDevice(&dev));
      RuntimeDeviceCheck(::cudaDeviceGetAttribute(&optin, cudaDevAttrMaxSharedMemoryPerBlockOptin, dev));
      RuntimeCheck(L::kSmBytes <= optin, "kv4_attn_wide: shared memory per block exceeds the device limit");
      RuntimeDeviceCheck(::cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, L::kSmBytes));
      // the warpgroups' register limits (setmaxnreg) must fit the block's launch allocation, or the raising warps
      // would wait forever: refuse to launch instead
      cudaFuncAttributes fa{};
      RuntimeDeviceCheck(::cudaFuncGetAttributes(&fa, kernel));
      constexpr bool kSplitRegs = (kSkel & 4) != 0;
      constexpr bool kCrRole = (kSkel & 32) != 0;
      constexpr int kOthers = kCrRole      ? 4 * 32 * (k4w::kWg2Regs + k4w::kDqRegsCr + k4w::kCrRegs)
                              : kSplitRegs ? 4 * 32 * k4w::kWg2RegsR + k4w::kDqWarps * 32 * k4w::kDqRegsR
                                           : (k4w::kNumWarps - 8) * 32 * k4w::kAuxRegs;
      constexpr int kSm = kCrRole ? k4w::kSmRegsCr : k4w::kSmRegs;
      RuntimeCheck(8 * 32 * kSm + kOthers <= k4w::threads_of<kSkel>() * fa.numRegs,
                   "kv4_attn_wide: register limits exceed the launch allocation");
      // lowered limits at most, raised limits at least the compiled allocation (setmaxnreg .dec / .inc)
      constexpr int kLowMax = kCrRole      ? (k4w::kWg2Regs > k4w::kCrRegs ? k4w::kWg2Regs : k4w::kCrRegs)
                              : kSplitRegs ? (k4w::kWg2RegsR > k4w::kDqRegsR ? k4w::kWg2RegsR : k4w::kDqRegsR)
                                           : k4w::kAuxRegs;
      constexpr int kHighMin = kCrRole ? (k4w::kSmRegsCr < k4w::kDqRegsCr ? k4w::kSmRegsCr : k4w::kDqRegsCr) : kSm;
      RuntimeCheck(kLowMax <= fa.numRegs && kHighMin >= fa.numRegs,
                   "kv4_attn_wide: register limits on the wrong side of the launch allocation");
      configured = true;
    }
    if (reqs == 0) return;
    LaunchKernel(dim3(static_cast<unsigned>(blocks)), k4w::threads_of<kSkel>(), out.device(), L::kSmBytes)
        .config({.use_pdl = kUsePDL})(kernel, params, qm, kdm, vdm, ksm, vsm);
    constexpr auto merge = k4w::kv4_attn_wide_merge_kernel<kPairs, kUsePDL>;
    LaunchKernel(dim3(static_cast<unsigned>(L::kHeads * k4w::kRows / k4w::kMergeRows), static_cast<unsigned>(reqs)),
                 256, out.device(), 0)
        .config({.use_pdl = kUsePDL})(merge, params, static_cast<int>(blocks));
  }
};

// Development / test entry: the dequantization of a whole packed pool through the kernel's table path, so the unit test
// can compare it with the conversion chain exhaustively. dst [n, 8] e4m3 bytes from words [n] and scale bytes [n].
namespace k4w {
__global__ void kv4_lut_probe_kernel(const uint32_t* __restrict__ words, const uint8_t* __restrict__ scales,
                                     uint8_t* __restrict__ lut_out, uint8_t* __restrict__ cvt_out, int n) {
  __shared__ uint2 lut[256];
  for (int i = threadIdx.x; i < 256; i += blockDim.x) lut[i] = lut_entry(static_cast<uint32_t>(i));
  __syncthreads();
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x) {
    const uint32_t s = scales[i];
    const uint2 a = lut8(words[i], lut[s]);
    const uint2 c = dequant8_cvt(words[i], scale_f16x2(s));
    reinterpret_cast<uint2*>(lut_out)[i] = a;
    reinterpret_cast<uint2*>(cvt_out)[i] = c;
  }
}
}  // namespace k4w

struct Kv4LutProbe {
  static void run(const tvm::ffi::TensorView words, const tvm::ffi::TensorView scales,
                  const tvm::ffi::TensorView lut_out, const tvm::ffi::TensorView cvt_out) {
    using namespace sglang::host;
    const int64_t n = words.numel();
    RuntimeCheck(words.dim() == 1 && scales.dim() == 1 && words.stride(0) == 1 && scales.stride(0) == 1 &&
                     scales.numel() == n && lut_out.numel() == 8 * n && cvt_out.numel() == 8 * n,
                 "probe shapes (contiguous 1-D words / scales, 8 bytes per word out)");
    LaunchKernel(dim3(static_cast<unsigned>((n + 255) / 256)), 256, words.device(), 0)(
        k4w::kv4_lut_probe_kernel, static_cast<const uint32_t*>(words.data_ptr()),
        static_cast<const uint8_t*>(scales.data_ptr()), static_cast<uint8_t*>(lut_out.data_ptr()),
        static_cast<uint8_t*>(cvt_out.data_ptr()), static_cast<int>(n));
  }
};

}  // namespace muse_vr
