// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// MoE-LD: one-launch routed NVFP4 experts for small verify batches, with each locality domain streaming the expert
// weight halves placed on it (Muse Spark expert shapes, one expert-parallel rank).
//
// Replaces FlashInfer's trtllm-gen routed MoE sequence (routing kernel, FC1 + SwiGLU GEMM, FC2-input NVFP4
// quantization kernel, FC2 GEMM) for at most 16 token rows by one launch of one CTA per SM. Inputs are the ones SGLang
// hands to `trtllm_fp4_block_scale_routed_moe` (packed top-k ids/weights, NVFP4 activations with linear block scales
// and per-token scales, the shuffled NVFP4 expert weights and their 128x4-interleaved block scales, the per-expert
// output scalars); outputs are the three tensors that call returns with do_finalize=False: FC2 rows (compact,
// expert-major row order), the top-k weights, and the expanded-index -> row map consumed by the existing finalize.
//
// Work decomposition
//   Each CTA takes a slot in the locality domain of its SM (one ticket per domain and launch). Domain d owns, for every
//   active local expert, the FC1 tiles 32d..32d+31 (128 gate/up rows each = 64 intermediate columns of the shuffled
//   layout, so SwiGLU stays inside the domain) and the FC2 tiles 16d..16d+15 (128 output rows each, full K), the
//   latter split into two half pieces of 64 rows. With the ld_experts placement these are exactly the row halves
//   backed by domain d. FC1 tiles go round-robin over the domain's slots; FC2 half pieces follow each slot's FC1 tiles
//   and are dealt in readiness order (make_sched), so that a slot streams the weights of its next FC2 piece while the
//   FC1 work of that piece's expert completes elsewhere. Every tile or piece is computed whole (full K = 4096 in one
//   accumulator, K order 0..4095), so every output element is accumulated exactly as the stock kernel does.
//   All M tokens are the MMA N operand for every expert (N = 8 or 16): the FC1 token operand is loaded once per launch
//   (one TMA) and stays in shared memory; the FC2 token operand is built per FC2 piece from the expert's FC1 output.
//   Rows of tokens not routed to an expert are computed but never stored.
//   Weights stream through a ring of 40 KiB stages (32 KiB of weights + their block scales); an FC2 half piece is
//   exactly four stages, so a 4-stage ring holds a whole piece while its token operand is built. The MMA is always
//   M = 128 (the only height of the block-scaled FP4 MMA); for a half piece the other 64 rows are discarded.
//   Warp roles (each a separate non-inlined function): 0 weight copies, 1 MMA issue, 2 block-scale copies into tensor
//   memory, 3 control (token operand copy), 4-7 epilogue, 8-11 FC2 token-operand builders.
//
// Numerics (tests/test_moe_ld.py: bitwise against the stock call)
//   FC1 epilogue: x = acc_gate * s_tok; u = acc_up * s_tok (clamped at the stock limit);
//                 y = bf16_satfinite(((u * a_gate) * (x * rcp(1 + ex2(-(x * (a_gate * log2e)))))) * a_up).
//   FC2 input: FlashInfer's per-token NVFP4 quantization (row amax over the bf16 activation row, e4m3 block scales,
//              the same fast-math reciprocals), from FlashInfer's own header.
//   FC2: bf16_satfinite((acc * s_tok2) * a_out), s_tok2 = row_amax * (1 / (448 * 6)).
//
// Cross-CTA hand-off (PTX ISA memory consistency model, release / acquire)
//   FC1 output rows travel to the FC2 builders (of both domains) in 64-bit words of 4 bf16, stored into one copy per
//   domain (each backed by that domain's memory). Each epilogue warp, after its stores of a tile (ordered by
//   __syncwarp), adds 1 to its own shared-memory count with release semantics (cta); the control warp publishes tile mm
//   once all four warps' counts reach mm + 1 (the warps may be a tile apart, so a single CTA-wide count would not tell
//   which tile is complete), acquires them and adds 4 to the expert's global counter with release semantics (gpu).
//   A builder lane polls the counter
//   (relaxed until close to complete, then acquire loads); the acquire load that sees all 256 arrivals synchronizes with
//   every tile's release, and a named barrier orders the four builder warps' plain row loads after it. A NaN result is
//   written as the canonical 0x7FFF. The counters are per call site (layer) and re-armed by each expert's last FC2
//   reader (an atomic reader count): launches of one call site are ordered by completion (stream or graph order).
//   Inside a CTA: the usual tcgen05 pipelines (mbarriers, tcgen05.commit, tcgen05.fence around thread sync, proxy
//   fence before generic writes are read by the MMA).
//   Every wait is bounded; a timeout sets a sticky error word and traps.
// Slots: each CTA claims one schedule slot (slot_claim.cuh): the slot of the SM it runs on is only the preferred one, and
//   an atomic claim bitmap makes every slot covered exactly once per launch whatever %smid returns (the PTX ISA lets
//   %smid change during execution); a claim that fails or a launch that ends with a slot unclaimed sets the sticky error
//   word and traps. The previous form (slot taken directly from the SM) compiles only with -DMUSE_VR_SLOT_UNSAFE=1.
// Residency: grid = SM count, one CTA per SM; CTAs wait on other CTAs' FC1 tiles, so all CTAs of a launch must be
//   resident together. Each CTA therefore takes a start ticket on entry (one atomic increment of a per-call-site counter) and leaves only once every CTA of its launch has
//   taken one: no CTA can start on an SM that another CTA of the same launch has already left, whatever else holds SMs
//   when the grid launches or how early a CTA runs out of work. Tickets cycle modulo a multiple of the grid size, so a
//   launch's tickets are one aligned block and the counter never overflows. Progress needs every CTA of the grid to start
//   eventually (a kernel that holds SMs until this one ends would block it; the exit wait is bounded like every other).
// Ids board (flag 64, default off): a separate producer kernel of the routed ids may publish them before this grid's
//   programmatic predecessors complete. It stores the packed ids, then stores the site's tag (launch index modulo the ticket
//   cycle, plus one) to control word kBoardWord with release semantics (gpu). Warp 0 lane 0 takes this CTA's start ticket,
//   derives the expected tag from it and polls the word with acquire loads (gpu), a bounded number of times; on a match the
//   ids are read right away with relaxed loads (gpu), otherwise after griddepcontrol.wait (the producer is then a completed
//   predecessor) and the miss is counted. Either way the weight stream starts before the dependency wait, and every output
//   and the token operand are written / read after it, as with flag 1. The producer and this kernel must run 1:1 per site.

#pragma once

#include <cuda.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>

#include <cfloat>
#include <cstdint>

#include "ld_queue.cuh"
#include "slot_claim.cuh"
#include "tensorrt_llm/kernels/quantization_utils.cuh"

// Warp roles are inlined into the kernel body (measured faster in chain A/B than separate role functions);
// -DMUSE_VR_MOELD_ROLE=__noinline__ keeps them separate (e.g. to read per-role code size).
#ifndef MUSE_VR_MOELD_ROLE
#define MUSE_VR_MOELD_ROLE __forceinline__
#endif
// 1: warp 0 issues the first weight stages of its first FC1 tile at the end of its prologue and the producer continues
// after them (measured faster in chain A/B); 0: the producer issues every stage
#ifndef MUSE_VR_MOELD_PREISSUE
#define MUSE_VR_MOELD_PREISSUE 1
#endif
// FC1 tile publication: 1 = each epilogue warp adds to its expert's counter with release semantics itself (after it
// released the accumulator); 0 = through the control warp (shared-memory count, then one gpu-scope release per tile)
#ifndef MUSE_VR_MOELD_DIRECT_PUBLISH
#define MUSE_VR_MOELD_DIRECT_PUBLISH 0
#endif
#ifndef MUSE_VR_MOELD_EPI_BULK
#define MUSE_VR_MOELD_EPI_BULK 0  // epilogue: one accumulator load per item + selects (experiment)
#endif
#ifndef MUSE_VR_MOELD_ROW_PAIRS
#define MUSE_VR_MOELD_ROW_PAIRS 0  // FC2-operand rows two at a time (experiment)
#endif
#ifndef MUSE_VR_MOELD_SPIN_NS
#define MUSE_VR_MOELD_SPIN_NS 128  // epilogue back-off while waiting for an FC2 operand's row maxima
#endif
#ifndef MUSE_VR_MOELD_K128
#define MUSE_VR_MOELD_K128 0  // 1: K = 128 block-scaled MMAs (two 64-wide chunks per instruction)
#endif
#ifndef MOELD_BISECT
#define MOELD_BISECT 0
#endif
#ifndef MUSE_VR_MOELD_MAX_POLLS
#define MUSE_VR_MOELD_MAX_POLLS (1u << 24)
#endif

namespace muse_vr_moeld {

// ---------------------------------------------------------------- problem shape
constexpr int kHidden = 4096;  // FC1 K, FC2 N
constexpr int kInter = 4096;   // FC1 N / 2, FC2 K
constexpr int kTopK = 8;
constexpr int kMaxTokens = 16;
constexpr int kMaxRows = kMaxTokens * kTopK;  // 128 activation rows (routed local slots) per launch
constexpr int kMaxPool = 128;                 // local expert slots per rank
constexpr int kTile = 128;                    // MMA M
constexpr int kMmaK = 64;
// FC2 work unit: whole tiles (128 output rows, default) or half pieces (64 rows; the MMA still computes 128 rows, so a
// half piece costs a whole tile's MMAs: an experiment, measured slower in A/B (bench_moe_ld)).
#ifndef MUSE_VR_MOELD_FC2_HALF
#define MUSE_VR_MOELD_FC2_HALF 0
#endif
constexpr bool kFc2Half = MUSE_VR_MOELD_FC2_HALF != 0;
constexpr int kFc2Rows = kFc2Half ? 64 : 128;
// Weight stages: 32 KiB of e2m1 weights + their block scales (stock 128x4 layout, 512 bytes per 128 rows x 64 K).
//   FC1 stage: 128 rows x 512 K (8 chunks, 4 KiB of scales), 8 stages per tile.
//   FC2 stage: whole tiles as FC1; half pieces 64 rows x 1024 K (16 chunks; the scales of all 128 rows, 8 KiB).
constexpr int kStageW = 32 * 1024;
constexpr int kFc1Chunks = 8, kFc1Stages = kHidden / (64 * kFc1Chunks);
constexpr int kFc2Chunks = kFc2Half ? 16 : 8, kFc2Stages = kInter / (64 * kFc2Chunks);
constexpr int kStageBytes = kStageW + kFc2Chunks * 512;
constexpr int kFc1TilesPerExpert = 2 * kInter / kTile;  // 64
constexpr int kFc2TilesPerExpert = kHidden / kTile;     // 32
constexpr int kFc1TilesPerDomain = kFc1TilesPerExpert / 2;  // 32
constexpr int kFc2TilesPerDomain = kFc2TilesPerExpert / 2;  // 16
constexpr int kFc2PiecesPerDomain = (kFc2Half ? 2 : 1) * kFc2TilesPerDomain;
constexpr int kW1 = kFc2Half ? 2 : 1;  // schedule cost of an FC1 tile in FC2-piece units
constexpr int kMaxItems = 96;
constexpr int kActWords = kInter / 4;  // 64-bit words per activation row (4 bf16)
constexpr int kActHalf = kMaxRows * kActWords;  // words per half of the activation buffer
// Per half, one 32-bit completion count per local expert slot (FC1 epilogue warps that stored their part of the expert's
// rows; a hint that lets FC2 builders poll one word instead of the rows themselves).
constexpr int kCntWords = kMaxPool;  // 64-bit words holding 2 x kMaxPool counters (completions, readers)
constexpr int kFc1Arrivals = kFc1TilesPerExpert * 4;  // 64 tiles x 4 epilogue warps
#ifndef MUSE_VR_MOELD_EARLY_ARRIVALS
#define MUSE_VR_MOELD_EARLY_ARRIVALS kFc1Arrivals  // arrivals at which builders start loading an expert's rows
#endif
// Weight block scales in tensor memory: 0 = 32-lane layout written by tcgen05.cp (one copy per 64-wide K chunk, 4
// columns per chunk); 1 = 128-lane layout (instruction descriptor bit 26, PTX ISA tcgen05 block scaling) written by
// tcgen05.st from four scale-writer warps (warps 12-15), one column per chunk. Experiment, off: no gain in A/B.
#ifndef MUSE_VR_MOELD_SFA128
#define MUSE_VR_MOELD_SFA128 0
#endif
constexpr bool kSfa128 = MUSE_VR_MOELD_SFA128 != 0;
constexpr int kThreads = kSfa128 ? 512 : 384;
constexpr uint32_t kMaxPolls = MUSE_VR_MOELD_MAX_POLLS;
// tensor memory columns (scale-factor addresses must be 4-column aligned)
constexpr uint32_t kTmemCols = 512;
constexpr uint32_t kAccCol = 0;      // 2 accumulators x 16 columns
constexpr uint32_t kSfaCol = 32;     // weight scales: kTsfRegions regions of kTsfCols columns (4 per 64-wide K chunk)
constexpr int kTsfRegions = kFc2Half ? 3 : 5;  // capacity; whole tiles use one region per stage (kS <= 5)
constexpr uint32_t kTsfCols = (kSfa128 ? 1u : 4u) * kFc2Chunks;
constexpr uint32_t kSfbCol = 256;    // token-operand scales, 4 columns per 64-wide K chunk ({s, 0, 0, 0}), 256 columns
// flags
constexpr int kFlagIdsEarly = 1;    // routing prologue and weight stream before griddepcontrol.wait (info arm)
constexpr int kFlagEvictFirst = 2;  // L2 evict_first policy on the weight and scale copies
constexpr int kFlagSwapDomain = 4;  // diagnostic: every CTA works on the other domain's tiles (remote weights)
constexpr int kFlagEarlyTrigger = 8;  // A/B: launch_dependents after the prologue (default: when the grid exits)
constexpr int kFlagL2Prefetch = 16;   // FC2 tiles: L2 prefetch of the stages that do not fit the ring
constexpr int kFlagTailTrigger = 32;  // A/B: launch_dependents when the CTA's producer starts its last item
constexpr int kFlagIdsBoard = 64;     // ids-first: read the routed ids once the producer of the ids publishes them on this
                                      // site's board (Ids board in the header); weight stream before griddepcontrol.wait
// error codes (sticky error word)
constexpr int kErrPoll = 1, kErrMbar = 2, kErrSlot = 4, kErrStart = 8;
// start tickets: the counter cycles over kStartCycle launches' worth of tickets (a multiple of the grid size)
constexpr unsigned kStartCycle = 1u << 12;
constexpr int kStartWord = 0;  // index of the ticket counter in the control block's reserved words
// ids board (flag 64): reserved word kBoardWord holds the tag of the last ids published for this site, kBoardMissWord counts
// launches that gave up waiting for their tag and read the ids after the dependency wait instead
constexpr int kBoardWord = 2, kBoardMissWord = 4;
// slot claim area (slot_claim.cuh): reserved words kClaimWord .. kClaimWord + muse_vr_slot::kWords - 1
constexpr int kClaimWord = 14;
static_assert(kClaimWord + muse_vr_slot::kWords <= 25, "claim area inside the control block's reserved words");
#ifndef MUSE_VR_MOELD_BOARD_POLLS
#define MUSE_VR_MOELD_BOARD_POLLS (1u << 12)
#endif
constexpr uint32_t kBoardPolls = MUSE_VR_MOELD_BOARD_POLLS;
__host__ __device__ constexpr bool ids_before_dependency(int flags) { return (flags & (kFlagIdsEarly | kFlagIdsBoard)) != 0; }

struct Params {
  const int32_t* packed;      // [M, 8]  (expert << 16 | bf16 weight)
  const uint8_t* hs_sf;       // [M, 256] e4m3, linear
  const float* pts;           // [M]
  const uint8_t* sf13;        // [P, 8192, 256] e4m3, shuffled + 128x4 interleaved
  const uint8_t* sf2;         // [P, 4096, 256]
  const float* gate_alpha;    // [P]
  const float* up_alpha;      // [P]
  const float* out_alpha;     // [P]
  const int8_t* sm_domain;    // [SM count] slot of each SM: locality domain << 7 | rank among the domain's SMs
  __nv_bfloat16* out;         // [M * 8, 4096]
  uint16_t* expert_weights;   // [M, 8] bf16 bits
  int32_t* e2p;               // [M * 8]
  unsigned long long* act;    // [2 domain copies][2 halves][kMaxRows][kActWords], all-ones at construction; copy d is
                              // backed by domain d's memory (this site's)
  unsigned* cnt;              // [kMaxPool] per-expert FC1 completion counts, [kMaxPool] FC2 reader counts; zero at
                              // construction, re-armed by each expert's last reader
  unsigned long long* trace;  // timing builds only (-DMUSE_VR_MOELD_TRACE): [grid][64] globaltimer stamps / counts
  muse_vr_ldq::Ctl* ctl;      // this call site's control block
  int32_t* err;               // sticky error word
  int32_t m;
  int32_t pool;
  int32_t local_offset;
  int32_t n_dom0, n_dom1;     // CTA slots per domain (n_dom0 + n_dom1 == grid)
  float gsi;                  // FC2-input global scale inverse (1 / (448 * 6))
  int32_t flags;
  int32_t dbg;                // ablation (timing studies only; nonzero gives wrong outputs): 1 no MMAs, 128 no
                              // epilogue / builder roles, 256 no weight copies
};

// ---------------------------------------------------------------- PTX helpers
// Timing builds include a private stamp header (not part of this package); production builds compile the stamps out.
#ifdef MUSE_VR_MOELD_TRACE
#include "moe_ld_stamps.cuh"
#else
#define MOELD_TRACE(slot) ((void)0)
#define MOELD_TRACE_ADD(slot, v) ((void)0)
#endif
__device__ __forceinline__ uint32_t smem_u32(const void* p) {
  return static_cast<uint32_t>(__cvta_generic_to_shared(p));
}
#ifndef MUSE_VR_MOELD_FAIL
#define MUSE_VR_MOELD_FAIL __noinline__
#endif
__device__ MUSE_VR_MOELD_FAIL void fail(int32_t* err, int code) {
  atomicOr(err, code);
  __threadfence();
  __trap();
}
__device__ __forceinline__ bool elect_one() {
  uint32_t pred;
  asm volatile("{ .reg .pred P; elect.sync _|P, 0xffffffff; selp.u32 %0, 1, 0, P; }" : "=r"(pred));
  return pred != 0;
}
__device__ __forceinline__ void mbar_init(uint64_t* bar, uint32_t count) {
  asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" ::"r"(smem_u32(bar)), "r"(count) : "memory");
}
__device__ __forceinline__ void mbar_arrive(uint64_t* bar) {
  asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];" ::"r"(smem_u32(bar)) : "memory");
}
__device__ __forceinline__ void mbar_arrive_expect_tx(uint64_t* bar, uint32_t bytes) {
  asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;" ::"r"(smem_u32(bar)), "r"(bytes) : "memory");
}
#ifndef MUSE_VR_MOELD_WAIT
#define MUSE_VR_MOELD_WAIT 0
#endif
__device__ __forceinline__ bool mbar_try(uint64_t* bar, uint32_t parity) {
  uint32_t done;
#if MUSE_VR_MOELD_WAIT == 1
  asm volatile(
      "{ .reg .pred p; mbarrier.test_wait.parity.shared::cta.b64 p, [%1], %2; selp.u32 %0, 1, 0, p; }"
      : "=r"(done)
      : "r"(smem_u32(bar)), "r"(parity)
      : "memory");
#elif MUSE_VR_MOELD_WAIT == 2
  asm volatile(
      "{ .reg .pred p; mbarrier.try_wait.parity.shared::cta.b64 p, [%1], %2, %3; selp.u32 %0, 1, 0, p; }"
      : "=r"(done)
      : "r"(smem_u32(bar)), "r"(parity), "r"(32u)
      : "memory");
#else
  asm volatile(
      "{ .reg .pred p; mbarrier.try_wait.parity.shared::cta.b64 p, [%1], %2; selp.u32 %0, 1, 0, p; }"
      : "=r"(done)
      : "r"(smem_u32(bar)), "r"(parity)
      : "memory");
#endif
  return done != 0;
}
// Warp-uniform bounded wait (every lane of the warp calls it with the same barrier and parity).
#ifndef MUSE_VR_MOELD_ROLE_SLEEP_NS
#define MUSE_VR_MOELD_ROLE_SLEEP_NS 0
#endif
__device__ __forceinline__ void mbar_wait(uint64_t* bar, uint32_t parity, int32_t* err) {
  uint32_t polls = 0;
  while (!mbar_try(bar, parity)) {
    if (++polls > kMaxPolls) fail(err, kErrMbar);
    if (MUSE_VR_MOELD_ROLE_SLEEP_NS > 0) __nanosleep(MUSE_VR_MOELD_ROLE_SLEEP_NS);
  }
}
// Wait for roles off the weight-stream critical path (epilogue, builders): poll with a back-off so that idle waiters
// do not keep the barrier unit busy while the stream's copies complete on it.
#ifndef MUSE_VR_MOELD_IDLE_NS
#define MUSE_VR_MOELD_IDLE_NS 256
#endif
__device__ __forceinline__ void mbar_wait_idle(uint64_t* bar, uint32_t parity, int32_t* err) {
  uint32_t polls = 0;
  while (!mbar_try(bar, parity)) {
    if (++polls > kMaxPolls) fail(err, kErrMbar);
    if (MUSE_VR_MOELD_IDLE_NS > 0) __nanosleep(MUSE_VR_MOELD_IDLE_NS);
  }
}
__device__ __forceinline__ void fence_barrier_init() {
  asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
}
__device__ __forceinline__ void fence_proxy_async_smem() {
  asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
}
__device__ __forceinline__ void named_bar_sync(uint32_t id, uint32_t n) {
  asm volatile("bar.sync %0, %1;" ::"r"(id), "r"(n) : "memory");
}
__device__ __forceinline__ void named_bar_arrive(uint32_t id, uint32_t n) {
  asm volatile("bar.arrive %0, %1;" ::"r"(id), "r"(n) : "memory");
}
__device__ __forceinline__ void grid_dep_wait() {
  asm volatile("griddepcontrol.wait;" ::: "memory");
}
__device__ __forceinline__ uint64_t policy_evict_first() {
  uint64_t pol;
  asm volatile("createpolicy.fractional.L2::evict_first.b64 %0, 1.0;" : "=l"(pol));
  return pol;
}
__device__ __forceinline__ void tma_load_3d(uint32_t dst, const CUtensorMap* map, int32_t x, int32_t y, int32_t z,
                                            uint64_t* bar) {
  asm volatile(
      "cp.async.bulk.tensor.3d.shared::cta.global.tile.mbarrier::complete_tx::bytes [%0], [%1, {%2, %3, %4}], [%5];" ::"r"(dst),
      "l"(map), "r"(x), "r"(y), "r"(z), "r"(smem_u32(bar))
      : "memory");
}
__device__ __forceinline__ void tma_load_3d_hint(uint32_t dst, const CUtensorMap* map, int32_t x, int32_t y, int32_t z,
                                                 uint64_t* bar, uint64_t pol) {
  asm volatile(
      "cp.async.bulk.tensor.3d.shared::cta.global.tile.mbarrier::complete_tx::bytes.L2::cache_hint [%0], [%1, {%2, %3, %4}], [%5], %6;" ::"r"(dst),
      "l"(map), "r"(x), "r"(y), "r"(z), "r"(smem_u32(bar)), "l"(pol)
      : "memory");
}
__device__ __forceinline__ void bulk_load(uint32_t dst, const void* src, uint32_t bytes, uint64_t* bar) {
  asm volatile("cp.async.bulk.shared::cta.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];" ::"r"(dst), "l"(src),
               "r"(bytes), "r"(smem_u32(bar))
               : "memory");
}
__device__ __forceinline__ void bulk_load_hint(uint32_t dst, const void* src, uint32_t bytes, uint64_t* bar, uint64_t pol) {
  asm volatile(
      "cp.async.bulk.shared::cta.global.mbarrier::complete_tx::bytes.L2::cache_hint [%0], [%1], %2, [%3], %4;" ::"r"(dst),
      "l"(src), "r"(bytes), "r"(smem_u32(bar)), "l"(pol)
      : "memory");
}
__device__ __forceinline__ void l2_prefetch_3d(const CUtensorMap* map, int32_t x, int32_t y, int32_t z) {
  asm volatile("cp.async.bulk.prefetch.tensor.3d.L2.global.tile [%0, {%1, %2, %3}];" ::"l"(map), "r"(x), "r"(y), "r"(z)
               : "memory");
}
__device__ __forceinline__ void l2_prefetch(const void* src, uint32_t bytes) {
  asm volatile("cp.async.bulk.prefetch.L2.global [%0], %1;" ::"l"(src), "r"(bytes) : "memory");
}
__device__ __forceinline__ void prefetch_tensormap(const void* map) {
  asm volatile("prefetch.tensormap [%0];" ::"l"(map) : "memory");
}

// tcgen05 (PTX ISA, tensor core 5th generation)
__device__ __forceinline__ void tmem_alloc(uint32_t* dst_smem, uint32_t ncols) {
  asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], %1;" ::"r"(smem_u32(dst_smem)), "r"(ncols));
}
__device__ __forceinline__ void tmem_relinquish() {
  asm volatile("tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned;");
}
__device__ __forceinline__ void tmem_dealloc(uint32_t taddr, uint32_t ncols) {
  asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, %1;" ::"r"(taddr), "r"(ncols));
}
__device__ __forceinline__ void tc_commit(uint64_t* bar) {
  asm volatile("tcgen05.commit.cta_group::1.mbarrier::arrive::one.shared::cluster.b64 [%0];" ::"r"(smem_u32(bar)) : "memory");
}
__device__ __forceinline__ void tc_fence_after() {
  asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");
}
__device__ __forceinline__ void tc_fence_before() {
  asm volatile("tcgen05.fence::before_thread_sync;" ::: "memory");
}
__device__ __forceinline__ void tc_wait_ld() {
  asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
}
__device__ __forceinline__ void tc_wait_st() {
  asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
}
__device__ __forceinline__ void tc_cp_32x128b_x4(uint32_t taddr, uint64_t sdesc) {
  asm volatile("tcgen05.cp.cta_group::1.32x128b.warpx4 [%0], %1;" ::"r"(taddr), "l"(sdesc) : "memory");
}
__device__ __forceinline__ void tc_mma_nvf4(uint32_t d, uint64_t a, uint64_t b, uint32_t idesc, uint32_t sfa,
                                            uint32_t sfb, uint32_t accumulate) {
  asm volatile(
      "{ .reg .pred p; setp.ne.b32 p, %4, 0;\n"
      "  tcgen05.mma.cta_group::1.kind::mxf4nvf4.block_scale.block16 [%0], %1, %2, %3, [%5], [%6], p; }" ::"r"(d),
      "l"(a), "l"(b), "r"(idesc), "r"(accumulate), "r"(sfa), "r"(sfb)
      : "memory");
}
// Shared-memory matrix descriptor, K-major, 128-byte swizzle, 8-row groups of 1024 bytes (SBO), version 1.
__device__ __forceinline__ uint64_t sdesc_sw128(uint32_t addr) {
  return uint64_t((addr & 0x3FFFFu) >> 4) | (uint64_t(1024u >> 4) << 32) | (uint64_t(1) << 46) | (uint64_t(2) << 61);
}
// Scale-factor source descriptor for tcgen05.cp: no swizzle, 8 x 16-byte rows per core matrix (SBO 128 bytes).
__device__ __forceinline__ uint64_t sdesc_sf(uint32_t addr) {
  return uint64_t((addr & 0x3FFFFu) >> 4) | (uint64_t(128u >> 4) << 32) | (uint64_t(1) << 46);
}
// Instruction descriptor, kind::mxf4nvf4: dense, E2M1 operands, K-major, UE4M3 scales, scale ids 0, M = 128, N = n.
__host__ __device__ constexpr uint32_t idesc_nvf4(uint32_t n) {
  return (1u << 7) | (1u << 10) | ((n >> 3) << 17) | ((128u >> 4) << 24) | (kSfa128 ? (1u << 26) : 0u);
}
__device__ __forceinline__ void tmem_ld_x8(uint32_t taddr, uint32_t (&r)[8]) {
  asm volatile("tcgen05.ld.sync.aligned.32x32b.x8.b32 {%0,%1,%2,%3,%4,%5,%6,%7}, [%8];"
               : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]), "=r"(r[4]), "=r"(r[5]), "=r"(r[6]), "=r"(r[7])
               : "r"(taddr));
}
__device__ __forceinline__ void tmem_st_x16(uint32_t taddr, const uint32_t (&r)[16]) {
  asm volatile(
      "tcgen05.st.sync.aligned.32x32b.x16.b32 [%0], {%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16};" ::"r"(taddr),
      "r"(r[0]), "r"(r[1]), "r"(r[2]), "r"(r[3]), "r"(r[4]), "r"(r[5]), "r"(r[6]), "r"(r[7]), "r"(r[8]), "r"(r[9]),
      "r"(r[10]), "r"(r[11]), "r"(r[12]), "r"(r[13]), "r"(r[14]), "r"(r[15])
      : "memory");
}
__device__ __forceinline__ void sts_u64(uint32_t addr, uint64_t v) {
  asm volatile("st.shared.b64 [%0], %1;" ::"r"(addr), "l"(v) : "memory");
}
__device__ __forceinline__ void sts_u16(uint32_t addr, uint16_t v) {
  asm volatile("st.shared.u16 [%0], %1;" ::"r"(addr), "h"(v) : "memory");
}
__device__ __forceinline__ void sts_v4(uint32_t addr, uint4 v) {
  asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" ::"r"(addr), "r"(v.x), "r"(v.y), "r"(v.z), "r"(v.w) : "memory");
}
__device__ __forceinline__ void sts_v2(uint32_t addr, uint2 v) {
  asm volatile("st.shared.v2.b32 [%0], {%1,%2};" ::"r"(addr), "r"(v.x), "r"(v.y) : "memory");
}
__device__ __forceinline__ uint4 lds_v4(uint32_t addr) {
  uint4 v;
  asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];" : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w) : "r"(addr) : "memory");
  return v;
}
__device__ __forceinline__ void st_release_cta_shared(int32_t* p, int v) {
  asm volatile("st.release.cta.shared::cta.b32 [%0], %1;" ::"r"(smem_u32(p)), "r"(v) : "memory");
}
__device__ __forceinline__ int ld_acquire_cta_shared(const int32_t* p) {
  int v;
  asm volatile("ld.acquire.cta.shared::cta.b32 %0, [%1];" : "=r"(v) : "r"(smem_u32(p)) : "memory");
  return v;
}
__device__ __forceinline__ uint32_t shfl_idx(uint32_t v, int src) {
  uint32_t r;
  asm volatile("shfl.sync.idx.b32 %0, %1, %2, 0x1f, 0xffffffff;" : "=r"(r) : "r"(v), "r"(src));
  return r;
}
__device__ __forceinline__ void st_relaxed_u64(unsigned long long* p, unsigned long long v) {
  asm volatile("st.relaxed.gpu.global.b64 [%0], %1;" ::"l"(p), "l"(v) : "memory");
}
__device__ __forceinline__ ulonglong2 ld_ca_v2u64(const unsigned long long* p) {
  ulonglong2 v;
  asm volatile("ld.global.ca.v2.u64 {%0,%1}, [%2];" : "=l"(v.x), "=l"(v.y) : "l"(p));
  return v;
}
__device__ __forceinline__ uint32_t tmem_ld_x1(uint32_t taddr) {
  uint32_t r;
  asm volatile("tcgen05.ld.sync.aligned.32x32b.x1.b32 {%0}, [%1];\n\ttcgen05.wait::ld.sync.aligned;" : "=r"(r) : "r"(taddr) : "memory");
  return r;
}
__device__ __forceinline__ unsigned ld_acquire_gpu_u32(const unsigned* p) {
  unsigned v;
  asm volatile("ld.acquire.gpu.global.u32 %0, [%1];" : "=r"(v) : "l"(p) : "memory");
  return v;
}

// fp32 arithmetic with flush-to-zero (reproduces the stock results bitwise).
__device__ __forceinline__ float fmul_ftz(float a, float b) {
  float r;
  asm("mul.ftz.f32 %0, %1, %2;" : "=f"(r) : "f"(a), "f"(b));
  return r;
}
__device__ __forceinline__ float fadd_ftz(float a, float b) {
  float r;
  asm("add.ftz.f32 %0, %1, %2;" : "=f"(r) : "f"(a), "f"(b));
  return r;
}
__device__ __forceinline__ float ffma_ftz(float a, float b, float c) {
  float r;
  asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(r) : "f"(a), "f"(b), "f"(c));
  return r;
}
__device__ __forceinline__ float ex2_ftz(float x) {
  float r;
  asm("ex2.approx.ftz.f32 %0, %1;" : "=f"(r) : "f"(x));
  return r;
}
__device__ __forceinline__ float rcp_ftz(float x) {
  float r;
  asm("rcp.approx.ftz.f32 %0, %1;" : "=f"(r) : "f"(x));
  return r;
}
__device__ __forceinline__ uint32_t bf16_satfinite(float x) {
  uint32_t r;
  asm("cvt.rn.satfinite.bf16x2.f32 %0, %1, %2;" : "=r"(r) : "f"(0.0f), "f"(x));
  return r & 0xFFFFu;
}

// FC1 gated epilogue for one (gate, up) accumulator pair, with the stock operand clamp at its default limit.
__device__ __forceinline__ uint32_t swiglu_out(float acc_gate, float acc_up, float s_tok, float a_gate, float a_up) {
  constexpr float kLimit = 3.40282346638528859812e+38f;
  float x = fmul_ftz(acc_gate, s_tok);
  float u = fmul_ftz(acc_up, s_tok);
  const float x0 = x, u0 = u;
  x = x > kLimit ? kLimit : x;
  x = (x0 >= -kLimit || x0 != x0) ? x : -kLimit;
  u = u > kLimit ? kLimit : u;
  u = (u0 >= -kLimit || u0 != u0) ? u : -kLimit;
  const float k = fmul_ftz(a_gate, 1.4426950216293334961f);
  const float e = ex2_ftz(fmul_ftz(fmul_ftz(x, k), -1.0f));
  const float r = rcp_ftz(fadd_ftz(e, 1.0f));
  const float up = ffma_ftz(u, a_gate, 0.0f);
  return bf16_satfinite(fmul_ftz(fmul_ftz(up, fmul_ftz(x, r)), a_up));
}

// Byte offset of (row r, 16-byte chunk c) inside one 128-byte-swizzled K atom of rows.
__device__ __forceinline__ uint32_t sw128_off(int r, int c) {
  return uint32_t((r >> 3) * 1024 + (r & 7) * 128 + ((c ^ (r & 7)) << 4));
}

// ---------------------------------------------------------------- shared memory
template <int kN, int kS>
struct Smem {
  // token operands first: an upper-half FC2 piece's A descriptor starts 8 KiB below its stage (see role_mma)
  alignas(1024) uint8_t b[2][kN * (kHidden / 2)];     // token operands (full K), 128-byte-swizzled K atoms of kN rows
  alignas(1024) uint8_t st[kS][kStageBytes];          // weight stages: 32 KiB weights + up to 8 KiB block scales
  alignas(16) uint8_t sfst[kN][kHidden / 16];         // FC2 token-operand block scales [token][16-block]
  uint64_t full[kS], empty[kS], sf_full[kTsfRegions], tsf_empty[kTsfRegions];
  uint64_t tmem_full[2], tmem_empty[2];
  uint64_t bfc1_full, sfb1_full, sfst_full, pts_full, b_full[2], b_empty[2], ramax_free[4];
  alignas(16) uint16_t ostage[4][kN][32];  // FC2 epilogue: per epilogue warp, [token][32 output columns]
  float ramax[4][kN];     // FC2 input row maxima per FC2 piece (ring of 4)
  float pmax[4][kMaxTokens];  // builder: [warp][routed row] partial row maxima
  int32_t ramax_seq[4];   // 1 + the FC2 piece whose row maxima a ramax entry holds (st.release / ld.acquire, cta)
  uint32_t fc1_stored[4];  // per FC1 epilogue warp: tiles whose activation words it stored (red.release / ld.acquire, cta)
  float tok_pts[kN];      // FC1 per-token activation scales
  float alpha[3][kMaxPool];
  uint32_t tok_mask[kMaxPool];  // tokens routed to each local expert
  uint16_t base[kMaxPool];      // first activation / output row of each local expert
  uint8_t ord_expert[kMaxPool];  // active local experts in ascending id
  uint32_t items[kMaxItems];     // this CTA's work list (item_pack)
  int32_t num_active, num_rows, work_dom, rank, nr, half, n_items, n_fc1;
  int32_t pre;  // weight stages of item 0 issued by the prologue
  uint32_t tmem_base;
};

// Work item: FC1 tile (128 gate/up rows, full K) or FC2 half piece (64 output rows, full K).
struct Item {
  int fc2;   // 0 FC1, 1 FC2 half piece
  int e;     // local expert slot
  int tile;  // tile of the expert (FC1 0..63, FC2 0..31)
  int h;     // FC2: rows 64 h .. 64 h + 63 of the tile
};
__device__ __forceinline__ uint32_t item_pack(int fc2, int e, int tile, int h) {
  return uint32_t(e) | (uint32_t(tile) << 8) | (uint32_t(h) << 14) | (uint32_t(fc2) << 15);
}
__device__ __forceinline__ Item item_unpack(uint32_t v) {
  Item it;
  it.e = int(v & 0xFFu);
  it.tile = int((v >> 8) & 0x3Fu);
  it.h = int((v >> 14) & 1u);
  it.fc2 = int((v >> 15) & 1u);
  return it;
}
__device__ __forceinline__ int stages_of(const Item& it) { return it.fc2 ? kFc2Stages : kFc1Stages; }

// Per-domain schedule (identical arithmetic in every CTA of the domain; R slots, U active experts):
//   FC1: tile g (expert ordinal g / 32, tile g % 32 of the domain) goes to slot g % R, in list position g / R.
//   FC2: 32 half pieces per expert. Each slot takes T2 - 2 n1 of them after its n1 FC1 tiles (T2 = the smallest
//   per-slot load, in half-tile units, that covers all pieces; one fewer on the first D slots that have any), and
//   the pieces are dealt in readiness order (expert ordinal, then tile, then half) to the slot positions in order of
//   their start (2 n1 + k, then slot): the earliest experts' pieces go to the earliest positions.
struct Sched {
  int g0[4], g1[4], n1[4], n2[4];  // four contiguous slot ranges (possibly empty) with constant (FC1 count, FC2 count)
};
__device__ __forceinline__ Sched make_sched(int U, int R) {
  Sched sc;
  const int H = kFc2PiecesPerDomain * U;
  const int q = (kFc1TilesPerDomain * U) / R, s1 = (kFc1TilesPerDomain * U) % R;
  const int W = (kW1 * kFc1TilesPerDomain + kFc2PiecesPerDomain) * U;
  int T2 = (W + R - 1) / R;
  auto cap = [&](int T) { return s1 * max(0, T - kW1 * (q + 1)) + (R - s1) * max(0, T - kW1 * q); };
#pragma unroll 1
  while (cap(T2) < H) ++T2;
  const int cA = max(0, T2 - kW1 * (q + 1)), cB = max(0, T2 - kW1 * q);
  const int D = cap(T2) - H;
  const int dA = cA > 0 ? min(D, s1) : 0;
  const int dB = cB > 0 ? D - dA : 0;
  sc.g0[0] = 0;       sc.g1[0] = dA;      sc.n1[0] = q + 1; sc.n2[0] = max(cA - 1, 0);
  sc.g0[1] = dA;      sc.g1[1] = s1;      sc.n1[1] = q + 1; sc.n2[1] = cA;
  sc.g0[2] = s1;      sc.g1[2] = s1 + dB; sc.n1[2] = q;     sc.n2[2] = max(cB - 1, 0);
  sc.g0[3] = s1 + dB; sc.g1[3] = R;       sc.n1[3] = q;     sc.n2[3] = cB;
  return sc;
}
// Readiness rank of position k (0-based, after the FC1 tiles) of slot r.
__device__ __forceinline__ int sched_rank(const Sched& sc, int r, int n1, int k) {
  const int tau = kW1 * n1 + k;
  int before = 0;
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    const int sz = sc.g1[i] - sc.g0[i];  // empty ranges contribute nothing
    before += sz * min(max(tau - kW1 * sc.n1[i], 0), sc.n2[i]);
    if (kW1 * sc.n1[i] <= tau && tau < kW1 * sc.n1[i] + sc.n2[i]) before += max(0, min(sc.g1[i], r) - sc.g0[i]);
  }
  return before;
}

__device__ __forceinline__ int steps_of(int units, int rank, int nr) {
  return units > rank ? (units - 1 - rank) / nr + 1 : 0;
}
// Phase index of the b_empty barrier completed when the MMAs of FC2 piece f (of this CTA) are done; FC1 release = 0 on
// buffer 0. FC2 piece f uses buffer (f + 1) & 1.
__device__ __forceinline__ uint32_t rel_phase(int f) {
  return (f & 1) ? uint32_t(f >> 1) + 1u : uint32_t(f >> 1);
}

// One FC1 weight stage (stage j of FC1 tile it into ring slot pos): 128 rows x 512 K (two 128-row K atoms) + their
// block scales. The leader lane only.
template <int kN, int kS>
__device__ __forceinline__ void issue_fc1_stage(Smem<kN, kS>& s, const Params& p, const CUtensorMap* m13, const Item& it, int j,
                                                int pos, bool ef, uint64_t pol) {
  const uint32_t dst = smem_u32(&s.st[pos][0]);
  const int tile_row = it.e * kFc1TilesPerExpert + it.tile;
  const uint8_t* sf = p.sf13 + (size_t(tile_row) * (kHidden / 64) + size_t(j) * kFc1Chunks) * 512;
  mbar_arrive_expect_tx(&s.full[pos], uint32_t(kStageW + kFc1Chunks * 512));
  if (ef) {
    tma_load_3d_hint(dst, m13, 0, tile_row * kTile, j * (kFc1Chunks / 4), &s.full[pos], pol);
    bulk_load_hint(dst + kStageW, sf, uint32_t(kFc1Chunks * 512), &s.full[pos], pol);
  } else {
    tma_load_3d(dst, m13, 0, tile_row * kTile, j * (kFc1Chunks / 4), &s.full[pos]);
    bulk_load(dst + kStageW, sf, uint32_t(kFc1Chunks * 512), &s.full[pos]);
  }
}

// ---------------------------------------------------------------- roles
// warp 0: weight + weight-scale copies, one stage at a time
template <int kN, int kS>
__device__ MUSE_VR_MOELD_ROLE void role_producer(Smem<kN, kS>& s, const Params& p, const CUtensorMap* m13,
                                                 const CUtensorMap* m2h) {
  const bool leader = elect_one();
  const int n_items = s.n_items;
  const bool ef = (p.flags & kFlagEvictFirst) != 0;
  const bool pf = (p.flags & kFlagL2Prefetch) != 0;
  if (leader) MOELD_TRACE(34);
  if (n_items == 0 && (p.flags & kFlagTailTrigger)) asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
  const uint64_t pol = policy_evict_first();
  if (leader) MOELD_TRACE(42);
  const int pre = MUSE_VR_MOELD_PREISSUE ? s.pre : 0;
  int k = pre;
#pragma unroll 1
  for (int m = 0; m < n_items; ++m) {
    const Item it = item_unpack(s.items[m]);
    const int nst = stages_of(it);
    if (m + 1 == n_items && (p.flags & kFlagTailTrigger)) asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
    if (it.fc2 && pf && leader) {
      // the stages past the ring arrive only after the operand is built and the first MMAs free slots: start their
      // DRAM reads now so that the copies hit L2
      const int tile_row = it.e * kFc2TilesPerExpert + it.tile;
#pragma unroll 1
      for (int j = kS; j < nst; ++j) {
        l2_prefetch_3d(m2h, 0, tile_row * kTile + kFc2Rows * it.h, j * (kFc2Chunks / 4));
        l2_prefetch(p.sf2 + (size_t(tile_row) * (kInter / 64) + size_t(j) * kFc2Chunks) * 512, uint32_t(kFc2Chunks * 512));
      }
    }
#pragma unroll 1
    for (int j = m == 0 ? pre : 0; j < nst; ++j, ++k) {
      const int pos = k % kS;
      const uint32_t use = uint32_t(k / kS);
      // a slot's first use needs no wait
      if (k >= kS) mbar_wait(&s.empty[pos], (use & 1u) ^ 1u, p.err);
      if (leader && (p.dbg & 256)) {  // ablation: no copies (MMAs read whatever the stage holds): MMA-bound timing
        mbar_arrive(&s.full[pos]);
      } else if (leader) {
        const uint32_t dst = smem_u32(&s.st[pos][0]);
        if (k == 0) MOELD_TRACE(41);
        if (it.fc2) {
          // kFc2Rows rows x (64 kFc2Chunks) K + the block scales of all 128 rows of the tile for those chunks
          const int tile_row = it.e * kFc2TilesPerExpert + it.tile;
          const uint8_t* sf = p.sf2 + (size_t(tile_row) * (kInter / 64) + size_t(j) * kFc2Chunks) * 512;
          mbar_arrive_expect_tx(&s.full[pos], uint32_t(kStageW + kFc2Chunks * 512));
          if (ef) {
            tma_load_3d_hint(dst, m2h, 0, tile_row * kTile + kFc2Rows * it.h, j * (kFc2Chunks / 4), &s.full[pos], pol);
          } else {
            tma_load_3d(dst, m2h, 0, tile_row * kTile + kFc2Rows * it.h, j * (kFc2Chunks / 4), &s.full[pos]);
          }
          if (ef && !kFc2Half) {
            bulk_load_hint(dst + kStageW, sf, uint32_t(kFc2Chunks * 512), &s.full[pos], pol);
          } else {  // the other half piece of the tile reads the same scale bytes: no evict-first hint on them
            bulk_load(dst + kStageW, sf, uint32_t(kFc2Chunks * 512), &s.full[pos]);
          }
        } else {
          // 128 rows x 512 K (two 128-row K atoms) + their block scales
          const int tile_row = it.e * kFc1TilesPerExpert + it.tile;
          const uint8_t* sf = p.sf13 + (size_t(tile_row) * (kHidden / 64) + size_t(j) * kFc1Chunks) * 512;
          mbar_arrive_expect_tx(&s.full[pos], uint32_t(kStageW + kFc1Chunks * 512));
          if (ef) {
            tma_load_3d_hint(dst, m13, 0, tile_row * kTile, j * (kFc1Chunks / 4), &s.full[pos], pol);
            bulk_load_hint(dst + kStageW, sf, uint32_t(kFc1Chunks * 512), &s.full[pos], pol);
          } else {
            tma_load_3d(dst, m13, 0, tile_row * kTile, j * (kFc1Chunks / 4), &s.full[pos]);
            bulk_load(dst + kStageW, sf, uint32_t(kFc1Chunks * 512), &s.full[pos]);
          }
        }
        if (k == 0) MOELD_TRACE(3);
        if (m + 1 == n_items && j + 1 == nst) MOELD_TRACE(4);
      }
      __syncwarp();
    }
  }
}

// warp 2: weight block scales, shared memory -> tensor memory (a ring of kTsfRegions regions of kTsfCols columns; stage k
// uses region k % kTsfRegions once the MMAs of stage k - kTsfRegions are done)
template <int kN, int kS>
__device__ MUSE_VR_MOELD_ROLE void role_sfcopy(Smem<kN, kS>& s, const Params& p) {
  const bool leader = elect_one();
  const int n_items = s.n_items;
  const uint32_t tbase = s.tmem_base;
  int k = 0;
#pragma unroll 1
  for (int m = 0; m < n_items; ++m) {
    const Item it = item_unpack(s.items[m]);
    const int nst = stages_of(it);
    const int nch = it.fc2 ? kFc2Chunks : kFc1Chunks;
#pragma unroll 1
    for (int j = 0; j < nst; ++j, ++k) {
      const int pos = k % kS;
      const int r = kFc2Half ? k % kTsfRegions : pos;
      mbar_wait(&s.full[pos], uint32_t(k / kS) & 1u, p.err);
      if constexpr (kFc2Half) {
        if (k >= kTsfRegions) mbar_wait(&s.tsf_empty[r], (uint32_t(k / kTsfRegions) & 1u) ^ 1u, p.err);
      } else {
        if (k >= kS) mbar_wait(&s.empty[pos], (uint32_t(k / kS) & 1u) ^ 1u, p.err);  // the MMAs that read region pos last are done
      }
      tc_fence_after();
      if (leader) {
        const uint32_t col = tbase + kSfaCol + uint32_t(r) * kTsfCols;
        const uint32_t src = smem_u32(&s.st[pos][kStageW]);
#pragma unroll 1
        for (int c = 0; c < nch; ++c) tc_cp_32x128b_x4(col + 4u * c, sdesc_sf(src + uint32_t(c) * 512u));
        tc_commit(&s.sf_full[r]);
      }
      __syncwarp();
    }
  }
}

// warps 12-15 (kSfa128): weight block scales, shared memory -> tensor memory in the 128-lane layout. Warp q writes the
// scales of rows 32 q .. 32 q + 31 (its lane quadrant): lane i reads row 32 q + i's four scales of chunk c from the stock
// 128x4 interleaved layout (byte 16 i + 4 q of the chunk's 512 bytes) and stores them in column c of the stage's region.
template <int kN, int kS>
__device__ MUSE_VR_MOELD_ROLE void role_sfwrite(Smem<kN, kS>& s, const Params& p) {
  const int lane = threadIdx.x & 31, q = (threadIdx.x >> 5) & 3;
  const bool leader = elect_one();
  const int n_items = s.n_items;
  const uint32_t tq = s.tmem_base + (uint32_t(32 * q) << 16) + kSfaCol;
  int k = 0;
#pragma unroll 1
  for (int m = 0; m < n_items; ++m) {
    const Item it = item_unpack(s.items[m]);
    const int nst = stages_of(it);
#pragma unroll 1
    for (int j = 0; j < nst; ++j, ++k) {
      const int pos = k % kS;
      const int r = kFc2Half ? k % kTsfRegions : pos;
      mbar_wait(&s.full[pos], uint32_t(k / kS) & 1u, p.err);
      if constexpr (kFc2Half) {
        if (k >= kTsfRegions) mbar_wait(&s.tsf_empty[r], (uint32_t(k / kTsfRegions) & 1u) ^ 1u, p.err);
      } else {
        if (k >= kS) mbar_wait(&s.empty[pos], (uint32_t(k / kS) & 1u) ^ 1u, p.err);
      }
      tc_fence_after();
      const uint32_t src = smem_u32(&s.st[pos][kStageW]) + uint32_t(16 * lane + 4 * q);
      uint32_t v[8];
#pragma unroll
      for (int c = 0; c < 8; ++c)
        asm volatile("ld.shared.u32 %0, [%1];" : "=r"(v[c]) : "r"(src + uint32_t(c) * 512u) : "memory");
      asm volatile("tcgen05.st.sync.aligned.32x32b.x8.b32 [%0], {%1,%2,%3,%4,%5,%6,%7,%8};" ::"r"(tq + uint32_t(r) * kTsfCols),
                   "r"(v[0]), "r"(v[1]), "r"(v[2]), "r"(v[3]), "r"(v[4]), "r"(v[5]), "r"(v[6]), "r"(v[7])
                   : "memory");
      tc_wait_st();
      tc_fence_before();
      __syncwarp();
      if (leader) mbar_arrive(&s.sf_full[r]);
      __syncwarp();
    }
  }
}

// warp 1: MMA issue
template <int kN, int kS>
__device__ MUSE_VR_MOELD_ROLE void role_mma(Smem<kN, kS>& s, const Params& p) {
  const bool leader = elect_one();
  const uint32_t tbase = s.tmem_base;
  const int n_items = s.n_items, n_fc1 = s.n_fc1;
  constexpr uint32_t idesc = idesc_nvf4(uint32_t(kN));
  if (n_fc1 == 0 && leader) mbar_arrive(&s.b_empty[0]);  // no FC1 use of buffer 0
  __syncwarp();
  const bool solo = (p.dbg & 128) != 0;  // ablation: no epilogue / builder roles
  int f = 0, k = 0;
#pragma unroll 1
  for (int m = 0; m < n_items; ++m) {
    const Item it = item_unpack(s.items[m]);
    const int acc = m & 1;
    const uint32_t aph = uint32_t(m >> 1) & 1u;
    int bbuf = 0;
    if (!it.fc2) {
      if (m == 0 && !solo) {
        mbar_wait(&s.bfc1_full, 0u, p.err);
        mbar_wait(&s.sfb1_full, 0u, p.err);
      }
    } else {
      bbuf = (f + 1) & 1;
      if (f == 0 && leader) MOELD_TRACE(13);
      if (!solo) mbar_wait(&s.b_full[bbuf], uint32_t(f >> 1) & 1u, p.err);
      if (f == 0 && leader) MOELD_TRACE(8);
    }
    if (!solo && m >= 2) mbar_wait(&s.tmem_empty[acc], aph ^ 1u, p.err);  // first use of each accumulator: free
    tc_fence_after();
    if (leader && m < 6) MOELD_TRACE(16 + m);
    const uint32_t d = tbase + kAccCol + uint32_t(acc) * 16u;
    const uint32_t b0 = smem_u32(&s.b[bbuf][0]);
    const int nst = stages_of(it);
    // FC1: atoms of 128 rows (16 KiB); FC2 half piece: atoms of 64 rows (8 KiB). The MMA always reads 128 rows: an
    // FC2 lower half (h = 0) sits in rows 0..63 (rows 64..127 read the next atom, discarded); an upper half (h = 1)
    // starts 8 KiB below its atom so that it sits in rows 64..127, where the tile's own scale layout puts its scales.
    const uint32_t atom = it.fc2 ? uint32_t(kFc2Rows) * 128u : 16384u;
    const uint32_t shift = (kFc2Half && it.fc2 && it.h) ? 8192u : 0u;
    const int nch = it.fc2 ? kFc2Chunks : kFc1Chunks;
#pragma unroll 1
    for (int j = 0; j < nst; ++j, ++k) {
      const int pos = k % kS;
      const int r = kFc2Half ? k % kTsfRegions : pos;
      // the scale-copy role waited for this stage's bytes (full) before it copied the scales and committed sf_full: one
      // wait here orders both the weights and the scales before these MMAs (fewer waits per stage measured faster in A/B)
      if constexpr (kFc2Half) mbar_wait(&s.full[pos], uint32_t(k / kS) & 1u, p.err);
      mbar_wait(&s.sf_full[r], uint32_t(k / (kFc2Half ? kTsfRegions : kS)) & 1u, p.err);
      tc_fence_after();
      if (leader && k == 0) MOELD_TRACE(5);
      if (leader && f == 0 && it.fc2 && j == 0) MOELD_TRACE(7);
      if (leader) {
        const uint32_t a0 = smem_u32(&s.st[pos][0]) - shift;
        const uint32_t sfa = tbase + kSfaCol + uint32_t(r) * kTsfCols;
        if ((p.dbg & 1) == 0) {
#if MUSE_VR_MOELD_K128
          // K = 128 per MMA (instruction descriptor K field = 2: bit 3), two 64-wide chunks per instruction
#pragma unroll 4
          for (int c = 0; c < nch; c += 2) {
            const int g = j * nch + c;
            const uint64_t da = sdesc_sw128(a0 + uint32_t(c >> 2) * atom + uint32_t(c & 3) * 32);
            const uint64_t db = sdesc_sw128(b0 + uint32_t(g >> 2) * (kN * 128) + uint32_t(g & 3) * 32);
            tc_mma_nvf4(d, da, db, idesc | (1u << 3), sfa + 4u * c, tbase + kSfbCol + 4u * g, (j | c) != 0 ? 1u : 0u);
          }
#else
          if constexpr (!kFc2Half) {
            // whole tiles: 8 chunks per stage, descriptors = stage base + constant offsets (16-byte units)
            const uint64_t da0 = sdesc_sw128(a0);
            const uint64_t db0 = sdesc_sw128(b0 + uint32_t(j * 2) * (kN * 128));
            const uint32_t sfb0 = tbase + kSfbCol + 32u * uint32_t(j);
#pragma unroll
            for (int c = 0; c < 8; ++c) {
              const uint64_t da = da0 + uint64_t((uint32_t(c >> 2) * 16384u + uint32_t(c & 3) * 32u) >> 4);
              const uint64_t db = db0 + uint64_t((uint32_t(c >> 2) * uint32_t(kN * 128) + uint32_t(c & 3) * 32u) >> 4);
              tc_mma_nvf4(d, da, db, idesc, sfa + (kSfa128 ? 1u : 4u) * c, sfb0 + 4u * c, (j | c) != 0 ? 1u : 0u);
            }
          } else {
#pragma unroll 4
            for (int c = 0; c < nch; ++c) {
              const int g = j * nch + c;  // 64-wide K chunk of the tile
              const uint64_t da = sdesc_sw128(a0 + uint32_t(c >> 2) * atom + uint32_t(c & 3) * 32);
              const uint64_t db = sdesc_sw128(b0 + uint32_t(g >> 2) * (kN * 128) + uint32_t(g & 3) * 32);
              tc_mma_nvf4(d, da, db, idesc, sfa + (kSfa128 ? 1u : 4u) * c, tbase + kSfbCol + 4u * g, (j | c) != 0 ? 1u : 0u);
            }
          }
#endif
        }
        tc_commit(&s.empty[pos]);
        if constexpr (kFc2Half) tc_commit(&s.tsf_empty[r]);
      }
      __syncwarp();
    }
    if (leader) {
      if (m < 6) MOELD_TRACE(24 + m);
      if (m + 1 == n_items) MOELD_TRACE(6);
      tc_commit(&s.tmem_full[acc]);
      if (!it.fc2) {
        if (m == n_fc1 - 1) tc_commit(&s.b_empty[0]);
      } else {
        tc_commit(&s.b_empty[bbuf]);
      }
    }
    __syncwarp();
    if (it.fc2) ++f;
  }
}

// warp 3: per-token activation scales of the FC1 epilogue (the token operand copies were issued before the routing
// prologue, see the kernel)
template <int kN, int kS>
__device__ MUSE_VR_MOELD_ROLE void role_ctrl(Smem<kN, kS>& s, const Params& p, const CUtensorMap* mB) {
  const int lane = threadIdx.x & 31;
  if (ids_before_dependency(p.flags)) {
    grid_dep_wait();
    if (elect_one()) {
      mbar_arrive_expect_tx(&s.bfc1_full, uint32_t(kN * (kHidden / 2)));
      tma_load_3d(smem_u32(&s.b[0][0]), mB, 0, 0, 0, &s.bfc1_full);
      mbar_arrive_expect_tx(&s.sfst_full, uint32_t(p.m * (kHidden / 16)));
      bulk_load(smem_u32(&s.sfst[0][0]), p.hs_sf, uint32_t(p.m * (kHidden / 16)), &s.sfst_full);
    }
    __syncwarp();
  }
  if (lane < kN) s.tok_pts[lane] = lane < p.m ? p.pts[lane] : 0.0f;
  __syncwarp();
  if (elect_one()) mbar_arrive(&s.pts_full);  // release: tok_pts
  __syncwarp();
}

// warps 4-7: epilogue (TMEM lane quadrant = warp % 4). Loops run over the routed tokens only (one tensor-memory column
// each) and are not unrolled: the role's code stays small (hot-code budget per role; compact roles measured faster).
template <int kN, int kS>
__device__ MUSE_VR_MOELD_ROLE void role_epilogue(Smem<kN, kS>& s, const Params& p) {
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
  const int q = warp & 3;
  const bool leader = elect_one();
  const uint32_t tbase = s.tmem_base;
  const int n_items = s.n_items, n_fc1 = s.n_fc1;
  unsigned long long* const act = p.act;  // domain-0 copy; domain-1 copy at + 2 kActHalf
  for (int i = threadIdx.x - 4 * 32; i < p.pool; i += 128) {  // per-expert output scalars (static weights)
    s.alpha[0][i] = p.gate_alpha[i];
    s.alpha[1][i] = p.up_alpha[i];
    s.alpha[2][i] = p.out_alpha[i];
  }
  named_bar_sync(1, 128);  // the four epilogue warps
  if (n_fc1 > 0) mbar_wait_idle(&s.pts_full, 0u, p.err);  // acquire: tok_pts
  const bool up_row = (lane & 8) == 0;
  const float gsi = p.gsi;
  int f = 0;
#pragma unroll 1
  for (int m = 0; m < n_items; ++m) {
    const int acc = m & 1;
    const uint32_t aph = uint32_t(m >> 1) & 1u;
    const Item it = item_unpack(s.items[m]);
#if MUSE_VR_SLOT_TEST == 3
    if (!it.fc2 && q == 3)  // tests only: warp skew on FC1 tiles
      for (int z = 0; z < 64; ++z) __nanosleep(1000);
#endif
    const uint32_t mask = s.tok_mask[it.e];
    const int base = s.base[it.e];
    // FC2 half piece h holds valid rows in tensor-memory lane quadrants 2 h and 2 h + 1 only
    const bool live = !kFc2Half || !it.fc2 || ((q >> 1) == it.h);
    if (it.fc2) {
      // acquire the row maxima of this FC2 piece (a counter, not a parity: the builder may run two pieces ahead)
      uint32_t polls = 0;
      while (ld_acquire_cta_shared(&s.ramax_seq[f & 3]) != f + 1) {
        if (++polls > kMaxPolls) fail(p.err, kErrMbar);
        if (MUSE_VR_MOELD_SPIN_NS > 0) __nanosleep(MUSE_VR_MOELD_SPIN_NS);
      }
    }
    mbar_wait_idle(&s.tmem_full[acc], aph, p.err);
    tc_fence_after();
    if (leader && q == 0 && m + 1 == n_items) MOELD_TRACE(32);
    const uint32_t tacc = tbase + (uint32_t(q * 32) << 16) + kAccCol + uint32_t(acc) * 16u;
#if MUSE_VR_MOELD_EPI_BULK
    // all kN accumulator columns in one or two loads; a routed token's value is picked by selects (no local memory)
    uint32_t raw[kN];
    if (live) {
      uint32_t r0[8];
      tmem_ld_x8(tacc, r0);
#pragma unroll
      for (int x = 0; x < 8; ++x) raw[x] = r0[x];
      if constexpr (kN == 16) {
        uint32_t r1[8];
        tmem_ld_x8(tacc + 8u, r1);
#pragma unroll
        for (int x = 0; x < 8; ++x) raw[8 + x] = r1[x];
      }
      tc_wait_ld();
    }
    auto col_of = [&](int x) {
      uint32_t v = raw[0];
#pragma unroll
      for (int i = 1; i < kN; ++i) v = (x == i) ? raw[i] : v;
      return v;
    };
#else
    auto col_of = [&](int x) { return tmem_ld_x1(tacc + uint32_t(x)); };
#endif
    if (!it.fc2) {
      // ---- FC1: SwiGLU (stock arithmetic) -> activation words of 4 bf16 for the routed tokens
      const float a_gate = s.alpha[0][it.e], a_up = s.alpha[1][it.e];
      // up rows: lane = a + 16 b (a = lane & 7) holds intermediate column 16 q + 2 a + b of this tile; lanes 0, 2, 4, 6
      // gather columns 16 q + 2 a .. + 3 (word 16 tile + 4 q + a / 2)
      unsigned long long* dst = act + size_t(it.tile) * 16 + size_t(4 * q + ((lane & 7) >> 1));
      unsigned long long* dst1 = dst + 2 * size_t(kActHalf);  // the same word in the domain-1 copy
#pragma unroll 1
      for (uint32_t mm = mask; mm != 0u; mm &= mm - 1u) {  // routed tokens, ascending (warp-uniform)
        const int x = __ffs(int(mm)) - 1;
        const float v = __uint_as_float(col_of(x));
        const float gate = __shfl_xor_sync(0xffffffffu, v, 8);
        uint32_t y = 0;
        if (up_row) {
          y = swiglu_out(gate, v, s.tok_pts[x], a_gate, a_up);
          y = (y & 0x7FFFu) > 0x7F80u ? 0x7FFFu : y;  // canonical NaN
        }
        const uint32_t y1 = __shfl_sync(0xffffffffu, y, (lane + 16) & 31);
        const uint32_t y2 = __shfl_sync(0xffffffffu, y, (lane + 1) & 31);
        const uint32_t y3 = __shfl_sync(0xffffffffu, y, (lane + 17) & 31);
        if (lane < 8 && (lane & 1) == 0) {
          const int row = base + __popc(mask & ((1u << x) - 1u));
          const unsigned long long wv =
              (unsigned long long)(y | (y1 << 16)) | ((unsigned long long)(y2 | (y3 << 16)) << 32);
          st_relaxed_u64(dst + size_t(row) * kActWords, wv);
          st_relaxed_u64(dst1 + size_t(row) * kActWords, wv);
        }
      }
      tc_fence_before();
      __syncwarp();
      if (leader) {
        mbar_arrive(&s.tmem_empty[acc]);
#if MUSE_VR_MOELD_DIRECT_PUBLISH
        // release (gpu) after this warp's stores, ordered by the __syncwarp above
        asm volatile("red.release.gpu.global.add.u32 [%0], 1;" ::"l"(p.cnt + it.e) : "memory");
#else
        asm volatile("red.release.cta.shared::cta.add.u32 [%0], 1;" ::"r"(smem_u32(&s.fc1_stored[MUSE_VR_SLOT_UNSAFE ? 0 : q]))
                     : "memory");
#endif
      }
    } else {
      if (live) {
        // ---- FC2: output rows (stock epilogue arithmetic), one 64-byte segment per warp and token
        const float a_out = s.alpha[2][it.e];
        // lane holds output column 32 q + col of this tile (the shuffled row order of the FC2 weights); transpose
        // through this warp's staging row, then four lanes write the token's 64-byte segment
        const uint32_t col = uint32_t((lane & 7) * 4 + (lane >> 3));
        const uint32_t st0 = smem_u32(&s.ostage[q][0][0]);
        uint8_t* dst = reinterpret_cast<uint8_t*>(p.out) + (size_t(it.tile) * kTile + size_t(32 * q)) * 2;
#pragma unroll 1
        for (uint32_t mm = mask; mm != 0u; mm &= mm - 1u) {  // routed tokens, ascending (warp-uniform)
          const int x = __ffs(int(mm)) - 1;
          const float acc_v = __uint_as_float(col_of(x));
          const float pts2 = fmul_ftz(s.ramax[f & 3][x], gsi);
          sts_u16(st0 + col * 2u, uint16_t(bf16_satfinite(fmul_ftz(fmul_ftz(acc_v, pts2), a_out))));
          __syncwarp();
          if (lane < 4) {
            const int row = base + __popc(mask & ((1u << x) - 1u));
            *reinterpret_cast<uint4*>(dst + size_t(row) * kHidden * 2 + size_t(lane) * 16) = lds_v4(st0 + uint32_t(lane) * 16u);
          }
          __syncwarp();
        }
      }
      if (leader && q == 0 && m + 1 == n_items) MOELD_TRACE(36);
      tc_fence_before();
      __syncwarp();
      if (leader) {
        mbar_arrive(&s.tmem_empty[acc]);
        mbar_arrive(&s.ramax_free[f & 3]);
      }
      ++f;
    }
    if (leader && q == 0 && m + 1 == n_fc1) MOELD_TRACE(9);
    if (leader && q == 0 && m + 1 == n_items) MOELD_TRACE(33);
  }
}

// warps 8-11: token operands' block scales into tensor memory and the FC2 token operand of each FC2 tile.
// Token-operand scales: every lane quadrant holds token l's four scales of 64-wide K chunk g in lane l, column
// kSfbCol + 4 g (the 32-lane scale layout of PTX ISA tcgen05 block scaling, N <= 32); the other three columns of each
// group are written as zeros by the same 32-column stores (fewer, wider stores measured faster in A/B).
__device__ __forceinline__ void store_sfb(const uint32_t tbase, const uint32_t sfst_row, const bool has) {
#pragma unroll 1
  for (int g = 0; g < 8; ++g) {
    const uint4 v = has ? lds_v4(sfst_row + uint32_t(g) * 32u) : make_uint4(0, 0, 0, 0);
    const uint4 u = has ? lds_v4(sfst_row + uint32_t(g) * 32u + 16u) : make_uint4(0, 0, 0, 0);
    asm volatile(
        "tcgen05.st.sync.aligned.32x32b.x32.b32 [%0], {%1,%2,%2,%2,%3,%2,%2,%2,%4,%2,%2,%2,%5,%2,%2,%2,"
        "%6,%2,%2,%2,%7,%2,%2,%2,%8,%2,%2,%2,%9,%2,%2,%2};" ::"r"(tbase + 32u * uint32_t(g)),
        "r"(v.x), "r"(0u), "r"(v.y), "r"(v.z), "r"(v.w), "r"(u.x), "r"(u.y), "r"(u.z), "r"(u.w)
        : "memory");
  }
}
// |max| over one 16-element block (two 16-byte halves of bf16)
__device__ __forceinline__ float block_absmax(ulonglong2 a, ulonglong2 b) {
  const uint32_t w[8] = {uint32_t(a.x), uint32_t(a.x >> 32), uint32_t(a.y), uint32_t(a.y >> 32),
                         uint32_t(b.x), uint32_t(b.x >> 32), uint32_t(b.y), uint32_t(b.y >> 32)};
  float t[8];
#pragma unroll
  for (int i = 0; i < 8; ++i) t[i] = fmaxf(fabsf(__uint_as_float(w[i] << 16)), fabsf(__uint_as_float(w[i] & 0xFFFF0000u)));
#pragma unroll
  for (int st = 4; st > 0; st >>= 1)
#pragma unroll
    for (int i = 0; i < st; ++i) t[i] = fmaxf(t[i], t[i + st]);
  return t[0];
}

// FlashInfer's per-token NVFP4 quantization of one 16-element bf16 block (two 16-byte halves): returns 16 e2m1 values
// and writes the block's e4m3 scale byte
__device__ __forceinline__ uint64_t quant16(ulonglong2 a, ulonglong2 b, float enc, float amax, uint8_t* sf) {
  using PV = tensorrt_llm::kernels::PackedVec<__nv_bfloat16, 16>;
  const uint32_t w8[8] = {uint32_t(a.x), uint32_t(a.x >> 32), uint32_t(a.y), uint32_t(a.y >> 32),
                          uint32_t(b.x), uint32_t(b.x >> 32), uint32_t(b.y), uint32_t(b.y >> 32)};
  PV vec;
#pragma unroll
  for (int x = 0; x < 8; ++x)
    vec.elts[x] = __halves2bfloat162(__ushort_as_bfloat16(uint16_t(w8[x] & 0xFFFFu)), __ushort_as_bfloat16(uint16_t(w8[x] >> 16)));
  return tensorrt_llm::kernels::cvt_warp_fp16_to_fp4<__nv_bfloat16, 16, 16, false, false, std::false_type>(vec, enc, sf, amax);
}

template <int kN, int kS>
__device__ MUSE_VR_MOELD_ROLE void role_builder(Smem<kN, kS>& s, const Params& p) {
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
  const int w = warp & 3;
  const bool leader = elect_one();
  const uint32_t tbase = s.tmem_base + (uint32_t(w * 32) << 16) + kSfbCol;
  const int n_items = s.n_items, n_fc1 = s.n_fc1;
  // the copy allocated on this CTA's domain
  const unsigned long long* const act = p.act + size_t(2 * s.work_dom) * kActHalf;
  const uint32_t sfst0 = smem_u32(&s.sfst[0][0]);
  const float gsi = p.gsi;
  // rows 0..M-1 of sfst hold hs_sf (bulk copy issued before the routing prologue): wait for it in every case, since
  // sfst is rewritten below
  mbar_wait_idle(&s.sfst_full, 0u, p.err);
  if (n_fc1 > 0) {
    // FC1 token-operand scales into tensor memory
    store_sfb(tbase, sfst0 + uint32_t(lane) * (kHidden / 16), lane < p.m);
    tc_wait_st();
    tc_fence_before();
    __syncwarp();
    if (leader) mbar_arrive(&s.sfb1_full);
    __syncwarp();
  }
  named_bar_sync(2, 128);  // every builder warp read the FC1 scales before sfst is reused
  const int n_fc2 = n_items - n_fc1;
#pragma unroll 1
  for (int f = 0; f < n_fc2; ++f) {
    const Item it = item_unpack(s.items[n_fc1 + f]);
    const int bf = (f + 1) & 1;
    const uint32_t mask = s.tok_mask[it.e];
    const int base = s.base[it.e];
    // the previous user of this buffer is done (FC1 for f == 1, FC2 tile f - 2 for f >= 2), and its ramax entry is free
    if (f == 1) {
      mbar_wait_idle(&s.b_empty[0], 0u, p.err);
      // the FC1 token-operand copy into buffer 0 is issued in every CTA: it has landed before this tile's operand is
      // written there (with no FC1 tile in this CTA, no MMA waited for it)
      mbar_wait_idle(&s.bfc1_full, 0u, p.err);
    }
    if (f >= 2) mbar_wait_idle(&s.b_empty[bf], rel_phase(f - 2) & 1u, p.err);
    if (f >= 4) mbar_wait_idle(&s.ramax_free[f & 3], uint32_t((f >> 2) - 1) & 1u, p.err);
    const uint32_t bs = smem_u32(&s.b[bf][0]);
    // rows of tokens not routed to this expert: zero operand and scales (warp t % 4), before the expert is ready
#pragma unroll 1
    for (uint32_t mm = ~mask & ((1u << kN) - 1u); mm != 0u; mm &= mm - 1u) {
      const int t = __ffs(int(mm)) - 1;
      if ((t & 3) != w) continue;
#pragma unroll 1
      for (int i = 0; i < 4; ++i) {
        const int ch = lane * 4 + i;  // 16-byte chunk of the row: atom ch >> 3, chunk ch & 7
        sts_v4(bs + uint32_t(ch >> 3) * (kN * 128) + sw128_off(t, ch & 7), make_uint4(0, 0, 0, 0));
      }
      sts_v2(sfst0 + uint32_t(t) * (kHidden / 16) + uint32_t(lane) * 8, make_uint2(0, 0));
      if (lane == 0) s.ramax[f & 3][t] = 0.0f;
    }
    if (w == 0 && lane == 0) {
      // the expert's FC1 tiles are all published (release chains through the control warp): relaxed polls, then one
      // acquire fence (acquire pattern), then the named barrier orders every builder warp's row reads after it
      const unsigned* cnt = p.cnt + it.e;
      uint32_t polls = 0;
      // relaxed polls while far from complete (no fence per poll during the FC1 stream), acquire polls for the last few
      // tiles: the read that sees the full count is itself the acquire (no extra round trip after it)
      while (muse_vr_ldq::ld_relaxed_gpu(cnt) < unsigned(kFc1Arrivals - 16)) {
        if (++polls > kMaxPolls) fail(p.err, kErrPoll);
        __nanosleep(64);
      }
      while (ld_acquire_gpu_u32(cnt) < unsigned(kFc1Arrivals)) {
        if (++polls > kMaxPolls) fail(p.err, kErrPoll);
        __nanosleep(32);
      }
    }
    named_bar_sync(2, 128);
    if (leader && w == 0 && f == 0) MOELD_TRACE(14);
    if (leader && w == 0 && f == 1) MOELD_TRACE(22);
    const int n_rows = __popc(mask);
    if (leader && w == 0 && f <= 1) MOELD_TRACE_ADD(46 + f, (unsigned long long)n_rows);
    const unsigned long long* const rows = act + size_t(base) * kActWords + size_t(64 * w + lane) * 4;
    if (leader && w == 0 && f == 0) MOELD_TRACE(37);
#if MUSE_VR_MOELD_ROW_PAIRS
    // routed rows two at a time (the next pair in flight, in registers): every builder warp takes a quarter of each row
    // (lane: 16-element blocks 64 w + lane and 64 w + 32 + lane); row maxima across the warps through shared memory (one
    // barrier per pair); then FlashInfer's per-token NVFP4 recipe, two independent chains (one per row of the pair)
    ulonglong2 r0[4], r1[4], q0[4], q1[4];
    auto load_row = [&](ulonglong2 (&d)[4], int k) {
      const unsigned long long* r = rows + size_t(k) * kActWords;
      d[0] = ld_ca_v2u64(r);
      d[1] = ld_ca_v2u64(r + 2);
      d[2] = ld_ca_v2u64(r + 128);
      d[3] = ld_ca_v2u64(r + 130);
    };
    if (n_rows > 0) load_row(r0, 0);
    if (n_rows > 1) load_row(r1, 1);
#pragma unroll 1
    for (int k = 0; k < n_rows; k += 2) {
      const bool two = k + 1 < n_rows;
      if (k + 2 < n_rows) load_row(q0, k + 2);
      if (k + 3 < n_rows) load_row(q1, k + 3);
      // |x| >= 0: the fp32 bit pattern orders like the value, so one integer warp reduction gives the maximum
      const float l0 = fmaxf(block_absmax(r0[0], r0[1]), block_absmax(r0[2], r0[3]));
      const float l1 = two ? fmaxf(block_absmax(r1[0], r1[1]), block_absmax(r1[2], r1[3])) : 0.0f;
      const uint32_t m0 = __reduce_max_sync(0xffffffffu, __float_as_uint(l0));
      const uint32_t m1 = __reduce_max_sync(0xffffffffu, __float_as_uint(l1));
      if (lane == 0) {
        s.pmax[w][k] = __uint_as_float(m0);
        if (two) s.pmax[w][k + 1] = __uint_as_float(m1);
      }
      if (leader && w == 0 && f <= 1 && k == 0) MOELD_TRACE(f == 0 ? 12 : 23);
      named_bar_sync(2, 128);  // all four quarters' maxima of rows k, k + 1 written
      if (leader && w == 0 && f == 0 && k == 0) MOELD_TRACE(39);
      const int t0 = int(__fns(mask, 0u, k + 1)), t1 = two ? int(__fns(mask, 0u, k + 2)) : t0;
      const float a0 = fmaxf(fmaxf(s.pmax[0][k], s.pmax[1][k]), fmaxf(s.pmax[2][k], s.pmax[3][k]));
      const float a1 = two ? fmaxf(fmaxf(s.pmax[0][k + 1], s.pmax[1][k + 1]), fmaxf(s.pmax[2][k + 1], s.pmax[3][k + 1])) : a0;
      if (w == 0 && lane == 0) {
        s.ramax[f & 3][t0] = a0;
        if (two) s.ramax[f & 3][t1] = a1;
      }
      const float p0 = fmul_ftz(a0, gsi), p1 = fmul_ftz(a1, gsi);
      const float e0 = p0 != 0.0f ? tensorrt_llm::kernels::reciprocal_approximate_ftz(p0) : 0.0f;
      const float e1 = p1 != 0.0f ? tensorrt_llm::kernels::reciprocal_approximate_ftz(p1) : 0.0f;
#pragma unroll 1
      for (int j = 0; j < 2; ++j) {
        const int blk = 64 * w + 32 * j + lane;
        const int byte = blk * 8;
        const uint32_t boff = uint32_t(byte >> 7) * (kN * 128) + uint32_t(byte & 8);
        // chain A: row k; chain B: row k + 1 (computed only with a second row)
        const ulonglong2 xa = j ? r0[2] : r0[0], ya = j ? r0[3] : r0[1];
        const ulonglong2 xb = j ? r1[2] : r1[0], yb = j ? r1[3] : r1[1];
        uint8_t sfa_, sfb_;
        const uint64_t qa = quant16(xa, ya, e0, a0, &sfa_);
        s.sfst[t0][blk] = sfa_;
        sts_u64(bs + boff + sw128_off(t0, (byte & 127) >> 4), qa);
        if (two) {
          const uint64_t qb = quant16(xb, yb, e1, a1, &sfb_);
          s.sfst[t1][blk] = sfb_;
          sts_u64(bs + boff + sw128_off(t1, (byte & 127) >> 4), qb);
        }
      }
#pragma unroll
      for (int i = 0; i < 4; ++i) {
        r0[i] = q0[i];
        r1[i] = q1[i];
      }
    }
#else
    // routed rows one at a time (the next row in flight, in registers): every builder warp takes a quarter of each
    // row (lane: 16-element blocks 64 w + lane and 64 w + 32 + lane); row maximum across the warps through shared
    // memory; then FlashInfer's per-token NVFP4 recipe on the lane's two blocks (two independent chains)
    ulonglong2 c[4], n1[4];
    auto load_row = [&](ulonglong2 (&d)[4], int k) {
      const unsigned long long* r = rows + size_t(k) * kActWords;
      d[0] = ld_ca_v2u64(r);
      d[1] = ld_ca_v2u64(r + 2);
      d[2] = ld_ca_v2u64(r + 128);
      d[3] = ld_ca_v2u64(r + 130);
    };
    if (n_rows > 0) load_row(c, 0);
#pragma unroll 1
    for (int k = 0; k < n_rows; ++k) {
      if (k + 1 < n_rows) load_row(n1, k + 1);
      // |x| >= 0: the fp32 bit pattern orders like the value, so one integer warp reduction gives the maximum
      const float lm = fmaxf(block_absmax(c[0], c[1]), block_absmax(c[2], c[3]));
      const uint32_t am = __reduce_max_sync(0xffffffffu, __float_as_uint(lm));
      if (lane == 0) s.pmax[w][k] = __uint_as_float(am);
      if (leader && w == 0 && f <= 1 && k == 0) MOELD_TRACE(f == 0 ? 12 : 23);
      named_bar_sync(2, 128);  // all four quarters' maxima of row k written
      if (leader && w == 0 && f == 0 && k == 0) MOELD_TRACE(39);
      const int t = int(__fns(mask, 0u, k + 1));  // routed row k belongs to the k-th routed token
      const float amax = fmaxf(fmaxf(s.pmax[0][k], s.pmax[1][k]), fmaxf(s.pmax[2][k], s.pmax[3][k]));
      if (w == 0 && lane == 0) s.ramax[f & 3][t] = amax;
      const float pts2 = fmul_ftz(amax, gsi);
      const float enc = pts2 != 0.0f ? tensorrt_llm::kernels::reciprocal_approximate_ftz(pts2) : 0.0f;
#pragma unroll
      for (int j = 0; j < 2; ++j) {
        uint8_t sf8;
        const uint64_t q8 = quant16(c[2 * j], c[2 * j + 1], enc, amax, &sf8);
        const int blk = 64 * w + 32 * j + lane;
        s.sfst[t][blk] = sf8;
        const int byte = blk * 8;
        sts_u64(bs + uint32_t(byte >> 7) * (kN * 128) + sw128_off(t, (byte & 127) >> 4) + uint32_t(byte & 8), q8);
      }
#pragma unroll
      for (int i = 0; i < 4; ++i) c[i] = n1[i];
    }
#endif
    fence_proxy_async_smem();
    if (leader && w == 0 && f == 0) MOELD_TRACE(11);
    if (leader && w == 0 && f == 1) MOELD_TRACE(30);
    named_bar_sync(2, 128);  // all four builder warps wrote this tile's operand, scales and row maxima
    if (leader && w == 0 && f == 0) MOELD_TRACE(48);
    if (threadIdx.x == 8 * 32) st_release_cta_shared(&s.ramax_seq[f & 3], f + 1);
    // the previous tile's MMAs (which read the token scales in tensor memory) are done
    if (f == 0) {
      if (n_fc1 > 0) mbar_wait_idle(&s.b_empty[0], 0u, p.err);
    } else {
      mbar_wait_idle(&s.b_empty[f & 1], rel_phase(f - 1) & 1u, p.err);  // tile f - 1 used buffer f & 1
    }
    tc_fence_after();
    if (leader && w == 0 && f == 0) MOELD_TRACE(49);
    store_sfb(tbase, sfst0 + uint32_t(lane) * (kHidden / 16), lane < kN);
    tc_wait_st();
    if (leader && w == 0 && f == 0) MOELD_TRACE(50);
    tc_fence_before();
    __syncwarp();
    if (leader) mbar_arrive(&s.b_full[bf]);  // release: token operand
    if (w == 0 && lane == 0) {
      // the expert's last reader (of its 2 x kFc2PiecesPerDomain FC2 pieces) re-arms its counters for the next launch
      // of this call site. Every reader's increment is acq_rel: it releases that reader's earlier read of the count, and
      // the last reader's increment acquires all of them before its reset stores (PTX memory model; launches of a site
      // are ordered by completion)
      unsigned* const rd = p.cnt + kMaxPool + it.e;
      if (muse_vr_ldq::atom_add_acq_rel_gpu(rd, 1u) == unsigned(2 * kFc2PiecesPerDomain - 1)) {
        muse_vr_ldq::st_relaxed_gpu(p.cnt + it.e, 0u);
        muse_vr_ldq::st_relaxed_gpu(rd, 0u);
      }
    }
    if (leader && w == 0 && f == 0) MOELD_TRACE(10);
    if (leader && w == 0 && f == 1) MOELD_TRACE(31);
    named_bar_sync(2, 128);  // every builder warp read sfst before the next tile rewrites it
  }
}

// ---------------------------------------------------------------- kernel
template <int kN, int kS>
__global__ void __launch_bounds__(kThreads, 1)
    moe_ld_kernel(const __grid_constant__ CUtensorMap m13, const __grid_constant__ CUtensorMap m2h,
                  const __grid_constant__ CUtensorMap mB, const __grid_constant__ Params p) {
  using S = Smem<kN, kS>;
  extern __shared__ __align__(1024) uint8_t smem_raw[];
  S& s = *reinterpret_cast<S*>((reinterpret_cast<uintptr_t>(smem_raw) + 1023) & ~uintptr_t(1023));
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
  if (tid == 0) MOELD_TRACE(0);

  // ---------------- setup + routing prologue, on the producer's critical path:
  //   warp 3 initializes the barriers (then issues the token-operand copies), warp 1 allocates tensor memory, warp 0
  //   takes this CTA's domain slot (one ticket) and reads the routed ids in the same round trip, builds the work list
  //   and starts the weight stream; every other warp waits for the work list on named barrier 3.
  const bool early = ids_before_dependency(p.flags);
  const bool board = (p.flags & kFlagIdsBoard) != 0;
  const int m = p.m;
  __shared__ unsigned start_base;  // first ticket of this launch's block (see Residency in the header)
  if (warp == 3) {
    if (elect_one()) {
      if (!board) {  // with the ids board, warp 0 takes the ticket (its launch index names the expected tag)
        const unsigned t = atomicInc(&p.ctl->reserved[kStartWord], gridDim.x * kStartCycle - 1u);
        start_base = t - t % gridDim.x;
      }
      for (int i = 0; i < kS; ++i) {
        mbar_init(&s.full[i], 1);
        mbar_init(&s.empty[i], 1);
      }
      for (int i = 0; i < kTsfRegions; ++i) {
        mbar_init(&s.sf_full[i], kSfa128 ? 4 : 1);
        mbar_init(&s.tsf_empty[i], 1);
      }
      for (int i = 0; i < 2; ++i) {
        mbar_init(&s.tmem_full[i], 1);
        mbar_init(&s.tmem_empty[i], 4);
        mbar_init(&s.b_full[i], 4);
        mbar_init(&s.b_empty[i], 1);
      }
      mbar_init(&s.bfc1_full, 1);
      mbar_init(&s.sfb1_full, 4);
      mbar_init(&s.sfst_full, 1);
      mbar_init(&s.pts_full, 1);
      for (int i = 0; i < 4; ++i) {
        mbar_init(&s.ramax_free[i], 4);
        s.ramax_seq[i] = 0;
      }
      for (int i = 0; i < 4; ++i) s.fc1_stored[i] = 0u;
      fence_barrier_init();
      prefetch_tensormap(&m13);
      prefetch_tensormap(&m2h);
      prefetch_tensormap(&mB);
    }
    __syncwarp();
    named_bar_arrive(4, 64);  // barriers initialized (warp 0 waits before its first copy)
    if (!early) {
      grid_dep_wait();
      // FC1 token operand (all tokens, full K; rows past M are zero-filled) and its block scales
      if (elect_one()) {
        mbar_arrive_expect_tx(&s.bfc1_full, uint32_t(kN * (kHidden / 2)));
        tma_load_3d(smem_u32(&s.b[0][0]), &mB, 0, 0, 0, &s.bfc1_full);
        mbar_arrive_expect_tx(&s.sfst_full, uint32_t(p.m * (kHidden / 16)));
        bulk_load(smem_u32(&s.sfst[0][0]), p.hs_sf, uint32_t(p.m * (kHidden / 16)), &s.sfst_full);
      }
      __syncwarp();
    }
  } else if (warp == 1) {
    tmem_alloc(&s.tmem_base, kTmemCols);
    tmem_relinquish();
    tc_fence_before();
  } else if (warp == 0) {
    // the routed ids and this CTA's slot: the SM's domain slot is the hint, the claim (slot_claim.cuh) is issued now and
    // settled once the ids are in, so its round trip overlaps theirs
    unsigned slot = 0;
#if !MUSE_VR_SLOT_UNSAFE
    unsigned* const claim_area = &p.ctl->reserved[kClaimWord];
    unsigned claim_old = 0;
#endif
#if MUSE_VR_SLOT_TRACE
    unsigned smid_claim = 0;
#endif
    if (lane == 0) {
      unsigned smid;
      asm volatile("mov.u32 %0, %%smid;" : "=r"(smid));
      slot = unsigned(reinterpret_cast<const uint8_t*>(p.sm_domain)[smid]);  // domain << 7 | rank in the domain
#if !MUSE_VR_SLOT_UNSAFE
      claim_old = muse_vr_slot::try_hint(claim_area, slot, p.n_dom0, p.n_dom1);
#endif
#if MUSE_VR_SLOT_TRACE
      smid_claim = smid;
#endif
    }
    int32_t pk[kMaxRows / 32];
    if (board) {
      // Ids board (header): the expected tag follows from this launch's index; poll, then read the ids (relaxed, gpu)
      unsigned got = 0;
      if (lane == 0) {
        const unsigned t = atomicInc(&p.ctl->reserved[kStartWord], gridDim.x * kStartCycle - 1u);
        start_base = t - t % gridDim.x;
        const unsigned want = start_base / gridDim.x + 1u;
#pragma unroll 1
        for (uint32_t n = 0; n < kBoardPolls; ++n) {
          if (ld_acquire_gpu_u32(&p.ctl->reserved[kBoardWord]) == want) {
            got = 1;
            break;
          }
          __nanosleep(64);
        }
        if (!got && blockIdx.x == 0) atomicAdd(&p.ctl->reserved[kBoardMissWord], 1u);
      }
      __syncwarp();  // memory ordering among the warp's threads: every lane's id loads follow lane 0's acquire
      got = __shfl_sync(0xffffffffu, got, 0);
      if (!got) grid_dep_wait();
#pragma unroll
      for (int k = 0; k < kMaxRows / 32; ++k)
        pk[k] = (lane + 32 * k < m * kTopK)
                    ? int32_t(muse_vr_ldq::ld_relaxed_gpu(reinterpret_cast<const unsigned*>(p.packed) + lane + 32 * k))
                    : 0;
    } else {
      if (!early) grid_dep_wait();
#pragma unroll
      for (int k = 0; k < kMaxRows / 32; ++k) pk[k] = (lane + 32 * k < m * kTopK) ? p.packed[lane + 32 * k] : 0;
    }
#pragma unroll
    for (int k = 0; k < kMaxPool / 32; ++k) s.tok_mask[lane + 32 * k] = 0u;
    __syncwarp();
#pragma unroll
    for (int k = 0; k < kMaxRows / 32; ++k) {
      const int i = lane + 32 * k;
      const int e = int((pk[k] >> 16) & 0xFFFF) - p.local_offset;
      if (i < m * kTopK && e >= 0 && e < p.pool) atomicOr(&s.tok_mask[e], 1u << (i / kTopK));
    }
    __syncwarp();
#if !MUSE_VR_SLOT_UNSAFE
    if (lane == 0) {
      const unsigned hint = slot;
      slot = muse_vr_slot::settle(claim_area, hint, claim_old, p.n_dom0, p.n_dom1, p.err);
#if MUSE_VR_SLOT_TRACE
      muse_vr_slot::trace_claim(p.trace ? p.trace + 64 * blockIdx.x + 60 : nullptr, smid_claim, hint, slot);
#endif
    }
#endif
    slot = shfl_idx(slot, 0);
    if (lane == 0) MOELD_TRACE(44);
    const unsigned dom = (slot >> 7) & 1u;
    const int wdom = (p.flags & kFlagSwapDomain) ? int(dom ^ 1u) : int(dom);  // domain whose tiles this CTA computes
    const int R = dom ? p.n_dom1 : p.n_dom0, r = int(slot & 127u);
    int run = 0, ord = 0;
#pragma unroll 1
    for (int c = 0; c < p.pool; c += 32) {
      const int e = c + lane;
      const int n = e < p.pool ? __popc(s.tok_mask[e]) : 0;
      int incl = n;
#pragma unroll
      for (int o = 1; o < 32; o <<= 1) {
        const int y = __shfl_up_sync(0xffffffffu, incl, o);
        if (lane >= o) incl += y;
      }
      if (e < p.pool) s.base[e] = uint16_t(run + incl - n);
      const uint32_t act = __ballot_sync(0xffffffffu, n > 0);
      if (n > 0) s.ord_expert[ord + __popc(act & ((1u << lane) - 1u))] = uint8_t(e);
      run += __shfl_sync(0xffffffffu, incl, 31);
      ord += __popc(act);
    }
    __syncwarp();  // ord_expert
    if (lane == 0) MOELD_TRACE(45);
    // this CTA's work list. Whole FC2 tiles: one round-robin over the domain list FC1(all experts) then FC2(all
    // experts) (list position = time step; FC2 tiles of earlier experts first). Half pieces: make_sched.
    const int U = ord;
    const int n1 = R > 0 ? steps_of(kFc1TilesPerDomain * U, r, R) : 0;
    int n2 = 0;
    Sched sc{};
    if (R > 0 && U > 0) {
      if constexpr (kFc2Half) {
        sc = make_sched(U, R);
#pragma unroll
        for (int i = 0; i < 4; ++i)
          if (sc.g0[i] <= r && r < sc.g1[i]) n2 = sc.n2[i];
      } else {
        n2 = steps_of((kFc1TilesPerDomain + kFc2PiecesPerDomain) * U, r, R) - n1;
      }
    }
    if (n1 + n2 > kMaxItems) fail(p.err, kErrSlot);
#pragma unroll 1
    for (int i = lane; i < n1 + n2; i += 32) {
      uint32_t v;
      if (i < n1) {
        const int g = r + i * R;
        v = item_pack(0, s.ord_expert[g >> 5], kFc1TilesPerDomain * wdom + (g & 31), 0);
      } else {
        int pc;
        if constexpr (kFc2Half) {
          pc = sched_rank(sc, r, n1, i - n1);
          const int j = pc & 31;
          v = item_pack(1, s.ord_expert[pc >> 5], kFc2TilesPerDomain * wdom + (j >> 1), j & 1);
        } else {
          pc = r + i * R - kFc1TilesPerDomain * U;  // index in the domain's FC2 list
          v = item_pack(1, s.ord_expert[pc >> 4], kFc2TilesPerDomain * wdom + (pc & 15), 0);
        }
      }
      s.items[i] = v;
    }
    if (lane == 0) {
      s.num_active = ord;
      s.num_rows = run;
      s.work_dom = wdom;
      s.rank = r;
      s.nr = R;
      s.n_fc1 = n1;
      s.n_items = n1 + n2;
    }
    __syncwarp();
#if MUSE_VR_MOELD_PREISSUE
    // the first weight stages of the first FC1 tile, issued right here (the producer continues after them)
    named_bar_sync(4, 64);  // barriers initialized (warp 3)
    int pre = 0;
    if (n1 > 0 && !early && (p.dbg & 256) == 0) {
      pre = kS < kFc1Stages ? kS : kFc1Stages;
      const Item it0 = item_unpack(s.items[0]);
      if (elect_one()) {
        const bool ef = (p.flags & kFlagEvictFirst) != 0;
        const uint64_t pol = policy_evict_first();
        MOELD_TRACE(41);
#pragma unroll 1
        for (int j = 0; j < pre; ++j) issue_fc1_stage<kN, kS>(s, p, &m13, it0, j, j, ef, pol);
        MOELD_TRACE(3);
      }
      __syncwarp();
    }
    if (lane == 0) s.pre = pre;
    __syncwarp();
#endif
  }
  // the work list (warp 0) and tensor memory (warp 1) for everyone; warp 0 does not wait (bar.arrive)
  if (warp == 0) {
    named_bar_arrive(3, kThreads);
  } else {
    named_bar_sync(3, kThreads);
  }
  tc_fence_after();
  if (tid == 0) MOELD_TRACE(1);

  // ---------------- roles
  if (tid == 0) MOELD_TRACE(2);
  // Opt-in early trigger (flag 8): the dependent grid (finalize) may launch while this one runs; it reads our outputs only
  // after its griddepcontrol.wait (this grid's completion and the visibility of its memory operations). Measured slower in
  // chain than triggering at each CTA's last item (flag 32, the ops default; see role_producer) or at grid exit.
  if (p.flags & kFlagEarlyTrigger) asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
  if (warp == 0) {
    if (!MUSE_VR_MOELD_PREISSUE) named_bar_sync(4, 64);  // barriers initialized (warp 3)
    if (!(MOELD_BISECT & 1)) role_producer<kN, kS>(s, p, &m13, &m2h);
  } else if (warp == 1) {
    if (!(MOELD_BISECT & 2)) role_mma<kN, kS>(s, p);
  } else if (warp == 2) {
    if constexpr (!kSfa128) role_sfcopy<kN, kS>(s, p);
  } else if (warp == 3) {
    if (!(MOELD_BISECT & 8)) role_ctrl<kN, kS>(s, p, &mB);
    if (!MUSE_VR_MOELD_DIRECT_PUBLISH && (p.dbg & 128) == 0 && lane == 0) {
      // publish each FC1 tile to its expert's counter once the four epilogue warps stored their words
      unsigned* const cnt = p.cnt;
#pragma unroll 1
      for (int mm = 0; mm < s.n_fc1; ++mm) {
        uint32_t polls = 0, done;
        while (true) {
          #if MUSE_VR_SLOT_UNSAFE
          // previous form: one CTA-wide count (assumes the four epilogue warps finish FC1 tiles in step)
          asm volatile("ld.relaxed.cta.shared::cta.u32 %0, [%1];" : "=r"(done) : "r"(smem_u32(&s.fc1_stored[0])) : "memory");
          if (done >= uint32_t(4 * (mm + 1))) break;
          #else
          // tile mm is stored once each of the four epilogue warps counted it (exact whatever their skew; the fence below
          // then acquires all four warps' releases)
          done = 0xFFFFFFFFu;
          #pragma unroll
          for (int w4 = 0; w4 < 4; ++w4) {
            uint32_t d;
            asm volatile("ld.relaxed.cta.shared::cta.u32 %0, [%1];" : "=r"(d) : "r"(smem_u32(&s.fc1_stored[w4])) : "memory");
            done = d < done ? d : done;
          }
          if (done >= uint32_t(mm + 1)) break;
          #endif
          if (++polls > kMaxPolls) fail(p.err, kErrMbar);
          __nanosleep(64);
        }
        asm volatile("fence.acq_rel.cta;" ::: "memory");  // acquire pattern (cta)
        const Item it = item_unpack(s.items[mm]);
        asm volatile("red.release.gpu.global.add.u32 [%0], 4;" ::"l"(cnt + it.e) : "memory");
      }
    }
    if (lane == 0) {
#if !MUSE_VR_SLOT_UNSAFE
      // this CTA's claim (warp 0, before named barrier 3) is complete: the last CTA checks and clears the claim bitmap
      muse_vr_slot::finish(&p.ctl->reserved[kClaimWord], p.err);
#endif
      // leave only once every CTA of this launch has taken its start ticket (Residency in the header); waited here,
      // where this warp's work ends early, so the check overlaps the other roles. A relaxed load suffices: nothing is
      // read on the strength of it. Measured slower when waited after the final block barrier (bench_moe_ld A/B).
      const unsigned cycle = gridDim.x * kStartCycle;
      uint32_t polls = 0;
      for (;;) {
        const unsigned v = muse_vr_ldq::ld_relaxed_gpu(&p.ctl->reserved[kStartWord]);
        const unsigned d = v >= start_base ? v - start_base : v + cycle - start_base;
        if (d >= gridDim.x) break;
        if (++polls > kMaxPolls) fail(p.err, kErrStart);
        __nanosleep(256);
      }
    }
    __syncwarp();
    if (blockIdx.x == 0) {
      // top-k weights and the expanded-index -> row map for the finalize (after the dependency wait: our outputs)
      if (early) grid_dep_wait();
      for (int i = lane; i < m * kTopK; i += 32) {
        const int32_t v = p.packed[i];
        const int e = int((v >> 16) & 0xFFFF) - p.local_offset;
        int row = -1;
        if (e >= 0 && e < p.pool) {
          const int t = i / kTopK;
          row = s.base[e] + __popc(s.tok_mask[e] & ((1u << t) - 1u));
        }
        p.e2p[i] = row;
        p.expert_weights[i] = uint16_t(v & 0xFFFF);
      }
    }
  } else if (warp < 8) {
    if (early) grid_dep_wait();
    if ((p.dbg & 128) == 0) role_epilogue<kN, kS>(s, p);
  } else if (warp < 12) {
    if ((p.dbg & 128) == 0) role_builder<kN, kS>(s, p);
  } else {
    if constexpr (kSfa128) role_sfwrite<kN, kS>(s, p);
  }

  __syncthreads();  // no thread leaves before the control warp saw every start ticket of this launch
  if (tid == 0) MOELD_TRACE(15);
#if MUSE_VR_SLOT_TRACE && !MUSE_VR_SLOT_UNSAFE
  if (tid == 0) muse_vr_slot::trace_exit(p.trace ? p.trace + 64 * blockIdx.x + 60 : nullptr);
#endif
  if (warp == 1) tmem_dealloc(s.tmem_base, kTmemCols);
}

}  // namespace muse_vr_moeld

// ---------------------------------------------------------------- host
#ifndef MUSE_VR_MOELD_DEVICE_ONLY
#include <sgl_kernel/tensor.h>
#include <sgl_kernel/type.cuh>
#include <sgl_kernel/utils.cuh>
#include <sgl_kernel/utils.h>

#include <dlpack/dlpack.h>
#include <tvm/ffi/container/tensor.h>

#include <mutex>
#include <unordered_map>

namespace muse_vr_moeld {
namespace {  // internal linkage: every JIT module keeps its own encoder and map caches

using PFN_encodeTiled = CUresult (*)(CUtensorMap*, CUtensorMapDataType, cuuint32_t, void*, const cuuint64_t*,
                                      const cuuint64_t*, const cuuint32_t*, const cuuint32_t*, CUtensorMapInterleave,
                                      CUtensorMapSwizzle, CUtensorMapL2promotion, CUtensorMapFloatOOBfill);

PFN_encodeTiled get_encode_fn() {
  static PFN_encodeTiled fn = [] {
    void* ptr = nullptr;
    cudaDriverEntryPointQueryResult q;
    sglang::host::RuntimeDeviceCheck(
        cudaGetDriverEntryPointByVersion("cuTensorMapEncodeTiled", &ptr, 12000, cudaEnableDefault, &q));
    sglang::host::RuntimeCheck(q == cudaDriverEntryPointSuccess && ptr != nullptr, "cuTensorMapEncodeTiled unavailable");
    return reinterpret_cast<PFN_encodeTiled>(ptr);
  }();
  return fn;
}

// [rows, 2048] uint8 rows viewed as (128-byte K atom, row, atom index); a box of box_rows rows x box_atoms atoms lands as
// box_atoms consecutive 128-byte-swizzled atoms of box_rows rows.
CUtensorMap make_rows_map(const void* base, uint64_t rows, uint32_t box_rows, uint32_t box_atoms) {
  CUtensorMap map{};
  const cuuint64_t dim[3] = {128, rows, kHidden / 2 / 128};
  const cuuint64_t stride[2] = {kHidden / 2, 128};
  const cuuint32_t box[3] = {128, box_rows, box_atoms};
  const cuuint32_t es[3] = {1, 1, 1};
  const CUresult rc = get_encode_fn()(&map, CU_TENSOR_MAP_DATA_TYPE_UINT8, 3, const_cast<void*>(base), dim, stride, box,
                                      es, CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B,
                                      CU_TENSOR_MAP_L2_PROMOTION_L2_256B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
  sglang::host::RuntimeCheck(rc == CUDA_SUCCESS, "cuTensorMapEncodeTiled failed");
  return map;
}

const CUtensorMap& cached_map(const void* base, uint64_t rows, uint32_t box_rows, uint32_t box_atoms) {
  static std::mutex mu;
  static std::unordered_map<uint64_t, CUtensorMap> cache;
  uint64_t key = reinterpret_cast<uint64_t>(base);
  key = key * 1000003ull ^ rows;
  key = key * 1000003ull ^ (uint64_t(box_rows) << 8 | box_atoms);
  std::lock_guard<std::mutex> lock(mu);
  auto it = cache.find(key);
  if (it == cache.end()) it = cache.emplace(key, make_rows_map(base, rows, box_rows, box_atoms)).first;
  return it->second;
}

}  // namespace

template <int kN, int kS>
struct MuseVrMoeLd {
  static_assert(kN == 8 || kN == 16);
  static_assert(kS >= 3 && kS <= 5);
  static_assert(kMaxPool == 128, "counter reset uses one epilogue thread per local expert slot");

  static void run(const tvm::ffi::TensorView packed, const tvm::ffi::TensorView hs, const tvm::ffi::TensorView hs_sf,
                  const tvm::ffi::TensorView pts, const tvm::ffi::TensorView w13, const tvm::ffi::TensorView sf13,
                  const tvm::ffi::TensorView w2, const tvm::ffi::TensorView sf2, const tvm::ffi::TensorView gate_alpha,
                  const tvm::ffi::TensorView up_alpha, const tvm::ffi::TensorView out_alpha,
                  const tvm::ffi::TensorView sm_domain, const tvm::ffi::TensorView out,
                  const tvm::ffi::TensorView expert_weights, const tvm::ffi::TensorView e2p,
                  const tvm::ffi::TensorView act, const tvm::ffi::TensorView aux, const tvm::ffi::TensorView ctl,
                  const tvm::ffi::TensorView err,
                  int64_t local_offset, int64_t n_dom0, int64_t n_dom1, double gsi, int64_t flags, int64_t dbg,
                  int64_t pdl) {
    using namespace sglang;
    using namespace sglang::host;
    auto M = SymbolicSize{"num_tokens"};
    auto P = SymbolicSize{"pool"};
    auto NSM = SymbolicSize{"sm_count"};
    auto device_ = SymbolicDevice{};
    device_.set_options<kDLCUDA>();
    TensorMatcher({M, kTopK}).with_dtype<int32_t>().with_device(device_).verify(packed);
    TensorMatcher({M, kHidden / 2}).with_dtype<uint8_t>().with_device(device_).verify(hs);
    TensorMatcher({M, kHidden / 16}).with_dtype<uint8_t>().with_device(device_).verify(hs_sf);
    TensorMatcher({M}).with_dtype<float>().with_device(device_).verify(pts);
    TensorMatcher({P, 2 * kInter, kHidden / 2}).with_dtype<uint8_t>().with_device(device_).verify(w13);
    TensorMatcher({P, 2 * kInter, kHidden / 16}).with_dtype<uint8_t>().with_device(device_).verify(sf13);
    TensorMatcher({P, kHidden, kInter / 2}).with_dtype<uint8_t>().with_device(device_).verify(w2);
    TensorMatcher({P, kHidden, kInter / 16}).with_dtype<uint8_t>().with_device(device_).verify(sf2);
    TensorMatcher({P}).with_dtype<float>().with_device(device_).verify(gate_alpha);
    TensorMatcher({P}).with_dtype<float>().with_device(device_).verify(up_alpha);
    TensorMatcher({P}).with_dtype<float>().with_device(device_).verify(out_alpha);
    TensorMatcher({NSM}).with_dtype<int8_t>().with_device(device_).verify(sm_domain);
    auto MR = SymbolicSize{"rows"};
    TensorMatcher({MR, kHidden}).with_dtype<bf16_t>().with_device(device_).verify(out);
    TensorMatcher({M, kTopK}).with_dtype<int16_t>().with_device(device_).verify(expert_weights);
    TensorMatcher({MR}).with_dtype<int32_t>().with_device(device_).verify(e2p);
    TensorMatcher({4 * int64_t(kActHalf)}).with_dtype<int64_t>().with_device(device_).verify(act);
    auto NX = SymbolicSize{"aux_words"};
    TensorMatcher({NX}).with_dtype<int64_t>().with_device(device_).verify(aux);
    TensorMatcher({32}).with_dtype<int32_t>().with_device(device_).verify(ctl);
    TensorMatcher({1}).with_dtype<int32_t>().with_device(device_).verify(err);
    const int64_t m = M.unwrap(), pool = P.unwrap();
    RuntimeCheck(m >= 1 && m <= kN, "token count outside 1..N of this build");
    RuntimeCheck(MR.unwrap() == m * kTopK, "output rows must be num_tokens * top_k");
    RuntimeCheck(pool >= 1 && pool <= kMaxPool, "pool outside 1..128");
    RuntimeCheck(local_offset >= 0 && local_offset + pool <= 32768, "bad local expert offset");
    RuntimeCheck(n_dom0 >= 0 && n_dom1 >= 0 && n_dom0 + n_dom1 == NSM.unwrap(), "domain slots must cover the grid");
    RuntimeCheck(NX.unwrap() >= int64_t(kCntWords), "auxiliary buffer too small");  // counters + reader counts
    // aux words after the counters, when present, receive per-CTA timestamps in timing builds
    const bool has_trace = NX.unwrap() >= int64_t(kCntWords) + NSM.unwrap() * 64;

    const Params params{
        .packed = static_cast<const int32_t*>(packed.data_ptr()),
        .hs_sf = static_cast<const uint8_t*>(hs_sf.data_ptr()),
        .pts = static_cast<const float*>(pts.data_ptr()),
        .sf13 = static_cast<const uint8_t*>(sf13.data_ptr()),
        .sf2 = static_cast<const uint8_t*>(sf2.data_ptr()),
        .gate_alpha = static_cast<const float*>(gate_alpha.data_ptr()),
        .up_alpha = static_cast<const float*>(up_alpha.data_ptr()),
        .out_alpha = static_cast<const float*>(out_alpha.data_ptr()),
        .sm_domain = static_cast<const int8_t*>(sm_domain.data_ptr()),
        .out = static_cast<__nv_bfloat16*>(out.data_ptr()),
        .expert_weights = static_cast<uint16_t*>(expert_weights.data_ptr()),
        .e2p = static_cast<int32_t*>(e2p.data_ptr()),
        .act = static_cast<unsigned long long*>(act.data_ptr()),
        .cnt = static_cast<unsigned*>(aux.data_ptr()),
        .trace = has_trace ? static_cast<unsigned long long*>(aux.data_ptr()) + kCntWords : nullptr,
        .ctl = static_cast<muse_vr_ldq::Ctl*>(ctl.data_ptr()),
        .err = static_cast<int32_t*>(err.data_ptr()),
        .m = static_cast<int32_t>(m),
        .pool = static_cast<int32_t>(pool),
        .local_offset = static_cast<int32_t>(local_offset),
        .n_dom0 = static_cast<int32_t>(n_dom0),
        .n_dom1 = static_cast<int32_t>(n_dom1),
        .gsi = static_cast<float>(gsi),
        .flags = static_cast<int32_t>(flags),
        .dbg = static_cast<int32_t>(dbg),
    };
    RuntimeCheck(NSM.unwrap() <= 256 && n_dom0 <= 128 && n_dom1 <= 128, "slot table holds 128 ranks per domain");
    const uint64_t r13 = uint64_t(pool) * 2 * kInter, r2 = uint64_t(pool) * kHidden;
    const CUtensorMap& m13 = cached_map(w13.data_ptr(), r13, kTile, kFc1Chunks / 4);  // FC1 stage
    const CUtensorMap& m2h = cached_map(w2.data_ptr(), r2, kFc2Rows, kFc2Chunks / 4);  // FC2 stage
    const CUtensorMap& mB = cached_map(hs.data_ptr(), uint64_t(m), kN, 16);
    constexpr auto kernel = moe_ld_kernel<kN, kS>;
    constexpr std::size_t kSmem = sizeof(Smem<kN, kS>) + 1024;
    [[maybe_unused]] static const bool configured = [] {
      int dev = 0, optin = 0;
      RuntimeDeviceCheck(cudaGetDevice(&dev));
      RuntimeDeviceCheck(cudaDeviceGetAttribute(&optin, cudaDevAttrMaxSharedMemoryPerBlockOptin, dev));
      RuntimeCheck(kSmem <= std::size_t(optin), "shared memory per block above the device's opt-in limit");
      RuntimeDeviceCheck(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, int(kSmem)));
      return true;
    }();
    LaunchKernel(dim3(static_cast<uint32_t>(NSM.unwrap())), dim3(kThreads), device_.unwrap(), kSmem)
        .enable_pdl(pdl != 0)(kernel, m13, m2h, mB, params);
  }
};

}  // namespace muse_vr_moeld
#endif  // MUSE_VR_MOELD_DEVICE_ONLY
