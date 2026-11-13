// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// MoE-LD-W BF16: routed BF16 experts of one expert-parallel rank for verify batches of 1..512 token rows, in one launch of
// one CTA per SM that streams every active local expert's weights once per pass of up to kPassRows routed rows. The NVFP4
// kernel (moe_ldw.cuh) with BF16 operands: the same prologue, work lists, slot claim, residency, ring and counters.
//
// Weights: the stock BF16 expert layout (gated row interleave + block-major K in 128-byte blocks), read by 4-D tensor maps
// (128-byte block, row, K block, expert). Every ring stage carries 32 KiB of weights (128 rows x 128 K) and the pass's token
// rows (16 / 32 / 48 / 64 / 128 rows x 128 K); the MMA is kind::f16 (BF16 x BF16 -> FP32), M = 128 x N = the bucket height,
// K = 16 per instruction; every output element is accumulated over the full K in K order.
// Prologue: as in moe_ldw.cuh. Copy phase (warps 4-11): the BF16 token rows of this CTA's slots into the expert-major
// buffer, the expanded-index map and top-k weights, then one release of the grid's rows-ready count.
// FC1 epilogue: SwiGLU (stock arithmetic, no scales) -> BF16 activation rows, published per tile by warp 3 (one count per
// tile and epilogue warp). FC2: the pass's activation rows ARE the FC2 operand (BF16): the producer acquires the pass's
// FC1 count, fences the proxies and loads them by tensor map; no build step. FC2 epilogue: BF16 output rows.
// Every wait is bounded (poll count, sticky error word); a CTA whose wait expired traps (MUSE_VR_LDW_TRAP, default 1).
// Build variants as in moe_ldw.cuh (-DMUSE_VR_LDW_PASS_ROWS=32 / 48 with -DMUSE_VR_LDW_STAGES=4); no head-copy path.

#pragma once

#include <cuda.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>

#include <cfloat>
#include <cstddef>
#include <cstdint>

#include "ld_queue.cuh"
#include "slot_claim.cuh"
#include "tensorrt_llm/kernels/quantization_utils.cuh"

// Warp roles are inlined into the kernel body (measured faster in chain A/B than separate role functions);
// -DMUSE_VR_LDW_ROLE=__noinline__ keeps them separate (e.g. to read per-role code size).
// weight stages (beyond the ring) whose bytes are prefetched into L2 right after the first stages go out: the ring cannot
// take them before the token rows of the first pass are copied, so their DRAM reads would otherwise wait (0: off)
#ifndef MUSE_VR_LDW_HEAD_PF
#define MUSE_VR_LDW_HEAD_PF 0
#endif
#ifndef MUSE_VR_LDW_ROLE
#define MUSE_VR_LDW_ROLE __forceinline__
#endif
#ifndef MUSE_VR_LDW_IDLE_NS
#define MUSE_VR_LDW_IDLE_NS 256
#endif
#ifndef MUSE_VR_LDW_TRAP
#define MUSE_VR_LDW_TRAP 1
#endif
// Tests only: 1 = the start ticket and the slot claim are issued BEFORE griddepcontrol.wait (the earlier placement; the
// negative control of tests/test_moe_ldwb_order.py). Default 0: both are issued right after the wait (prologue).
#ifndef MUSE_VR_LDW_TEST_PREWAIT_CLAIM
#define MUSE_VR_LDW_TEST_PREWAIT_CLAIM 0
#endif
#ifndef MUSE_VR_LDW_MAX_POLLS
#define MUSE_VR_LDW_MAX_POLLS (1u << 24)
#endif

// Every build variant (compile flags) gets its own symbol names (the ops layer sets MUSE_VR_LDW_ABI per variant), so
// variants loaded in one process never share a kernel or a launch-configuration static.
#ifndef MUSE_VR_LDW_ABI
#define MUSE_VR_LDW_ABI v0
#endif
namespace muse_vr_ldwb {
inline namespace MUSE_VR_LDW_ABI {

constexpr uint32_t kMaxPolls = MUSE_VR_LDW_MAX_POLLS;
// ---------------------------------------------------------------- MoE-LD-W BF16 problem shape
constexpr int kHidden = 4096;  // FC1 K, FC2 N
constexpr int kInter = 4096;   // FC1 N / 2, FC2 K
constexpr int kTopK = 8;
constexpr int kMaxTokens = 512;
constexpr int kMaxSlots = kMaxTokens * kTopK;  // routed slots per launch (all ranks' ids are read; local ones kept)
constexpr int kMaxPool = 128;
#ifdef MUSE_VR_LDW_PASS_ROWS
constexpr int kPassRows = MUSE_VR_LDW_PASS_ROWS;
static_assert(kPassRows == 32 || kPassRows == 48 || kPassRows == 64 || kPassRows == 128, "pass rows");
#else
constexpr int kPassRows = 128;
#endif
constexpr int kMaxPasses = kMaxPool + kMaxSlots / kPassRows;
static_assert(kMaxPasses <= 256, "pass index bits of a work item");
constexpr int kTile = 128;
constexpr int kRowBytes = kHidden * 2;              // one BF16 row of 4096 elements (token, activation, weight)
constexpr int kStageW = 32 * 1024;                  // 128 weight rows x 128 K
constexpr int kChunks = 8, kStageK = 16 * kChunks;  // 8 MMAs of K = 16 per stage
constexpr int kStagesPerTile = kHidden / kStageK;   // FC1 K = FC2 K = 4096
// -DMUSE_VR_LDW_RELAY=1 (development): each tile's K is split in two halves streamed by CTAs of the two locality domains (the
// host places each expert's K-block halves on the domains, alternating by expert parity); the first half's CTA hands its exact
// fp32 accumulator tile to the second half's CTA, which loads it into tensor memory and continues the same MMA sequence.
#ifndef MUSE_VR_LDW_RELAY
#define MUSE_VR_LDW_RELAY 0
#endif
#if MUSE_VR_LDW_RELAY
constexpr int kStagesPerItem = kStagesPerTile / 2;  // one K half per item
constexpr int kRelayPasses = kMaxPasses;            // relay slots: passes x (64 FC1 + 32 FC2 tiles)
constexpr int kRelayCols = kPassRows;               // fp32 accumulator columns per relay row (the largest bucket)
constexpr int kRelaySlots = kRelayPasses * 96;
#else
constexpr int kStagesPerItem = kStagesPerTile;
#endif
constexpr int kStageA = kStageW;
constexpr int kStageB = kPassRows * 256;            // token rows x 128 K
constexpr int kStageBytes = kStageA + kStageB;
#ifdef MUSE_VR_LDW_STAGES
constexpr int kStages = MUSE_VR_LDW_STAGES;
#else
constexpr int kStages = 3;
#endif
#define MUSE_VR_LDW_GATHER 0  // no head-copy path in this kernel
constexpr int kHeadItems = 1, kHeadRows = 1;
constexpr int kFc1TilesPerExpert = 2 * kInter / kTile, kFc2TilesPerExpert = kHidden / kTile;
constexpr int kFc1TilesPerDomain = kFc1TilesPerExpert / 2, kFc2TilesPerDomain = kFc2TilesPerExpert / 2;
constexpr int kMaxItems = MUSE_VR_LDW_RELAY ? 192 : 96;
constexpr int kActWords = kInter / 4;  // 64-bit words per activation row (4 bf16)
constexpr int kFc1Arrivals = kFc1TilesPerExpert * 4;
constexpr int kThreads = 384;
constexpr uint32_t kTmemCols = 256;
constexpr uint32_t kAccCol = 0;  // 2 accumulators x 128 columns
constexpr int kErrPoll = 1, kErrMbar = 2, kErrSlot = 4, kErrStart = 8, kErrDrain = 16, kErrShape = 32;
// slot claim area (slot_claim.cuh): 32-bit words kClaimWord .. kClaimWord + muse_vr_slot::kWords - 1 of the control
// tensor (word 0: start tickets)
constexpr int kClaimWord = 21;
static_assert(kClaimWord + muse_vr_slot::kWords <= 32, "claim area inside the 32-word control tensor");
constexpr unsigned kStartCycle = 1u << 12;
constexpr int kNumBuckets = 4;  // token-operand heights 16 / 32 / (48 | 64) / 128
#if MUSE_VR_LDW_PASS_ROWS == 48
__host__ __device__ constexpr int bucket_of(int rows) { return rows <= 16 ? 0 : rows <= 32 ? 1 : rows <= 48 ? 2 : 3; }
__host__ __device__ constexpr int bucket_rows(int b) { return b == 2 ? 48 : 16 << b; }
#else
__host__ __device__ constexpr int bucket_of(int rows) { return rows <= 16 ? 0 : rows <= 32 ? 1 : rows <= 64 ? 2 : 3; }
__host__ __device__ constexpr int bucket_rows(int b) { return 16 << b; }
#endif
// ---------------------------------------------------------------- PTX helpers
__device__ __forceinline__ uint32_t smem_u32(const void* p) {
  return static_cast<uint32_t>(__cvta_generic_to_shared(p));
}
// CTA abort flag (shared memory; set on the first expired wait of any thread of the CTA)
__shared__ int g_abort;
__device__ __forceinline__ bool aborted() { return *reinterpret_cast<volatile int*>(&g_abort) != 0; }
__device__ __noinline__ void fail(int32_t* err, int code) {
  atomicOr(err, code);
  *reinterpret_cast<volatile int*>(&g_abort) = 1;
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
__device__ __forceinline__ void mbar_expect_tx(uint64_t* bar, uint32_t bytes) {
  asm volatile("mbarrier.expect_tx.relaxed.cta.shared::cta.b64 [%0], %1;" ::"r"(smem_u32(bar)), "r"(bytes) : "memory");
}
__device__ __forceinline__ bool mbar_try(uint64_t* bar, uint32_t parity) {
  uint32_t done;
  asm volatile(
      "{ .reg .pred p; mbarrier.try_wait.parity.shared::cta.b64 p, [%1], %2; selp.u32 %0, 1, 0, p; }"
      : "=r"(done)
      : "r"(smem_u32(bar)), "r"(parity)
      : "memory");
  return done != 0;
}
// Non-blocking test of a phase (warp-uniform: lane 0's observation)
__device__ __forceinline__ bool mbar_test_warp(uint64_t* bar, uint32_t parity) {
  uint32_t done;
  asm volatile(
      "{ .reg .pred p; mbarrier.test_wait.parity.shared::cta.b64 p, [%1], %2; selp.u32 %0, 1, 0, p; }"
      : "=r"(done)
      : "r"(smem_u32(bar)), "r"(parity)
      : "memory");
  return __shfl_sync(0xffffffffu, done, 0) != 0;
}
// Bounded wait (every lane of the calling warp calls it with the same barrier and parity). Returns false when the CTA
// aborted or the bound ran out (which aborts the CTA).
__device__ __forceinline__ bool mbar_wait(uint64_t* bar, uint32_t parity, int32_t* err, uint32_t sleep_ns = 0) {
  uint32_t polls = 0;
  while (!mbar_try(bar, parity)) {
    ++polls;
    if ((polls & 63u) == 0u && aborted()) return false;
    if (polls > kMaxPolls) {
      fail(err, kErrMbar);
      return false;
    }
    if (sleep_ns > 0) __nanosleep(sleep_ns);
  }
  return true;
}
// Waits off the weight-stream critical path (epilogue, builders, control): one inline test, then an out-of-line polling
// loop shared by every call site (hot-code budget of those roles).
__device__ __noinline__ bool mbar_wait_slow(uint64_t* bar, uint32_t parity, int32_t* err) {
  return mbar_wait(bar, parity, err, MUSE_VR_LDW_IDLE_NS);
}
__device__ __forceinline__ bool mbar_wait_idle(uint64_t* bar, uint32_t parity, int32_t* err) {
  return mbar_try(bar, parity) || mbar_wait_slow(bar, parity, err);
}
// Bounded wait for asynchronous work this CTA issued (completes on its own): not cut short by the abort flag.
__device__ __noinline__ void mbar_drain(uint64_t* bar, uint32_t parity, int32_t* err) {
  uint32_t polls = 0;
  while (!mbar_try(bar, parity)) {
    if (++polls > kMaxPolls) {
      atomicOr(err, kErrDrain);
      return;
    }
  }
}
__device__ __forceinline__ void fence_barrier_init() {
  asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
}
__device__ __forceinline__ void fence_proxy_async_smem() {
  asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
}
__device__ __forceinline__ void fence_proxy_async_global() {
  asm volatile("fence.proxy.async.global;" ::: "memory");
}
__device__ __forceinline__ void named_bar_sync(uint32_t id, uint32_t n) {
  // non-aligned form: correct even if the calling warp is not converged at this point (PTX ISA barrier.sync)
  asm volatile("barrier.sync %0, %1;" ::"r"(id), "r"(n) : "memory");
}
__device__ __forceinline__ void named_bar_arrive(uint32_t id, uint32_t n) {
  asm volatile("barrier.arrive %0, %1;" ::"r"(id), "r"(n) : "memory");
}
__device__ __forceinline__ void grid_dep_wait() {
  asm volatile("griddepcontrol.wait;" ::: "memory");
}
__device__ __forceinline__ uint64_t policy_evict_first() {
  uint64_t pol;
  asm volatile("createpolicy.fractional.L2::evict_first.b64 %0, 1.0;" : "=l"(pol));
  return pol;
}
// asynchronous copies global -> shared of 16 / 4 bytes (generic proxy); completion tracked by cp_async_arrive
__device__ __forceinline__ void cp_async_16(uint32_t dst, const void* src) {
  asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16;" ::"r"(dst), "l"(src) : "memory");
}
__device__ __forceinline__ void cp_async_4(uint32_t dst, const void* src) {
  asm volatile("cp.async.ca.shared::cta.global [%0], [%1], 4;" ::"r"(dst), "l"(src) : "memory");
}
// cp.async groups: one per head stage; before a ring slot is written again, the issuing thread waits for the group that
// last wrote it (complete by then: its stage was consumed, which the empty barrier already showed), so the reuse is
// ordered after the copy by this thread too
__device__ __forceinline__ void cp_async_commit() { asm volatile("cp.async.commit_group;" ::: "memory"); }
__device__ __forceinline__ void cp_async_wait_newer(int n) {  // at most n newer groups pending (n = 0, 1, 2)
  if (n <= 0) asm volatile("cp.async.wait_group 0;" ::: "memory");
  else if (n == 1) asm volatile("cp.async.wait_group 1;" ::: "memory");
  else asm volatile("cp.async.wait_group 2;" ::: "memory");
}
// arrive-on of bar once every prior cp.async of this thread has completed; the pending count is raised by one first, so
// the phase cannot complete before those copies (PTX ISA cp.async.mbarrier.arrive)
__device__ __forceinline__ void cp_async_arrive(uint64_t* bar) {
  asm volatile("cp.async.mbarrier.arrive.shared::cta.b64 [%0];" ::"r"(smem_u32(bar)) : "memory");
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
__device__ __forceinline__ void tma_load_4d(uint32_t dst, const CUtensorMap* map, int32_t x, int32_t y, int32_t z, int32_t w,
                                            uint64_t* bar) {
  asm volatile(
      "cp.async.bulk.tensor.4d.shared::cta.global.tile.mbarrier::complete_tx::bytes [%0], [%1, {%2, %3, %4, %5}], [%6];" ::"r"(dst),
      "l"(map), "r"(x), "r"(y), "r"(z), "r"(w), "r"(smem_u32(bar))
      : "memory");
}
__device__ __forceinline__ void tma_load_4d_hint(uint32_t dst, const CUtensorMap* map, int32_t x, int32_t y, int32_t z,
                                                 int32_t w, uint64_t* bar, uint64_t pol) {
  asm volatile(
      "cp.async.bulk.tensor.4d.shared::cta.global.tile.mbarrier::complete_tx::bytes.L2::cache_hint [%0], [%1, {%2, %3, %4, %5}], [%6], %7;" ::"r"(dst),
      "l"(map), "r"(x), "r"(y), "r"(z), "r"(w), "r"(smem_u32(bar)), "l"(pol)
      : "memory");
}
__device__ __forceinline__ void l2_prefetch_3d(const CUtensorMap* map, int32_t x, int32_t y, int32_t z) {
  asm volatile("cp.async.bulk.prefetch.tensor.3d.L2.global.tile [%0, {%1, %2, %3}];" ::"l"(map), "r"(x), "r"(y), "r"(z)
               : "memory");
}
__device__ __forceinline__ void l2_prefetch(const void* src, uint32_t bytes) {
  asm volatile("cp.async.bulk.prefetch.L2.global [%0], %1;" ::"l"(src), "r"(bytes) : "memory");
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
__device__ __forceinline__ void tc_wait_st() {
  asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
}
__device__ __forceinline__ void tc_mma_bf16(uint32_t d, uint64_t a, uint64_t b, uint32_t idesc, uint32_t accumulate) {
  asm volatile(
      "{ .reg .pred p; setp.ne.b32 p, %4, 0;\n"
      "  tcgen05.mma.cta_group::1.kind::f16 [%0], %1, %2, %3, p; }" ::"r"(d), "l"(a), "l"(b), "r"(idesc), "r"(accumulate)
      : "memory");
}
// Shared-memory matrix descriptor, K-major, 128-byte swizzle, 8-row groups of 1024 bytes (SBO), version 1.
__device__ __forceinline__ uint64_t sdesc_sw128(uint32_t addr) {
  return uint64_t((addr & 0x3FFFFu) >> 4) | (uint64_t(1024u >> 4) << 32) | (uint64_t(1) << 46) | (uint64_t(2) << 61);
}
// Instruction descriptor, kind::f16 (PTX ISA tcgen05 instruction descriptor): D F32, A and B BF16, dense, K-major,
// M = 128, N = n.
__host__ __device__ constexpr uint32_t idesc_bf16(uint32_t n) {
  return (1u << 4) | (1u << 7) | (1u << 10) | ((n >> 3) << 17) | ((128u >> 4) << 24);
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
__device__ __forceinline__ unsigned long long ld_relaxed_u64(const unsigned long long* p) {
  unsigned long long v;
  asm volatile("ld.relaxed.gpu.global.b64 %0, [%1];" : "=l"(v) : "l"(p) : "memory");
  return v;
}
__device__ __forceinline__ ulonglong2 ld_relaxed_v2u64(const unsigned long long* p) {
  ulonglong2 v;
  asm volatile("ld.relaxed.gpu.global.v2.b64 {%0,%1}, [%2];" : "=l"(v.x), "=l"(v.y) : "l"(p) : "memory");
  return v;
}
__device__ __forceinline__ unsigned long long ld_acquire_u64(const unsigned long long* p) {
  unsigned long long v;
  asm volatile("ld.acquire.gpu.global.b64 %0, [%1];" : "=l"(v) : "l"(p) : "memory");
  return v;
}
__device__ __forceinline__ int ld_relaxed_s32(const int32_t* p) {
  int v;
  asm volatile("ld.relaxed.gpu.global.b32 %0, [%1];" : "=r"(v) : "l"(p) : "memory");
  return v;
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

// FC1 gated epilogue for one (gate, up) accumulator pair: the NVFP4 kernel's stock arithmetic with unit scales (no
// per-token or per-expert factors in the BF16 expert path) and the stock operand clamp at its default limit.
__device__ __forceinline__ uint32_t swiglu_out(float acc_gate, float acc_up) {
  constexpr float kLimit = 3.40282346638528859812e+38f;
  float x = acc_gate;
  float u = acc_up;
  const float x0 = x, u0 = u;
  x = x > kLimit ? kLimit : x;
  x = (x0 >= -kLimit || x0 != x0) ? x : -kLimit;
  u = u > kLimit ? kLimit : u;
  u = (u0 >= -kLimit || u0 != u0) ? u : -kLimit;
  const float e = ex2_ftz(fmul_ftz(fmul_ftz(x, 1.4426950216293334961f), -1.0f));
  const float r = rcp_ftz(fadd_ftz(e, 1.0f));
  return bf16_satfinite(fmul_ftz(u, fmul_ftz(x, r)));
}
// Byte offset of (row r, 16-byte chunk c) inside one 128-byte-swizzled K atom of rows.
__device__ __forceinline__ uint32_t sw128_off(int r, int c) {
  return uint32_t((r >> 3) * 1024 + (r & 7) * 128 + ((c ^ (r & 7)) << 4));
}


constexpr int kMaxMySlots = 32;                  // routed slots whose token row one CTA copies (slots b, b + G, ...)
#if MUSE_VR_LDW_RELAY
static_assert(kPassRows <= kRelayCols, "relay rows hold the largest bucket");
constexpr int kRelayCnt = 2 * kMaxPasses + 32;   // relay counts [kRelaySlots] (one per epilogue warp of the first half)
constexpr int kCntSet = kRelayCnt + kRelaySlots;
#else
constexpr int kCntSet = 2 * kMaxPasses + 32;     // per launch parity: FC1 arrivals [kMaxPasses], FC2-operand rows
#endif
constexpr int kRowsReady = 2 * kMaxPasses;       // [kMaxPasses], rows-ready count of the grid (one per CTA)
constexpr int kChunkSlots = 256, kNumChunks = kMaxSlots / kChunkSlots;

struct Params {
  const int32_t* packed;     // [M, 8] (expert << 16 | bf16 weight)
  const uint8_t* hs;         // [M, 8192] BF16 token rows
  const int8_t* sm_domain;
  uint8_t* hs_perm;          // [kMaxSlots, 8192] FC1 token rows, expert-major
  unsigned long long* act;   // [kMaxSlots][kActWords] BF16 activation rows (the FC2 operand)
  unsigned* cnt;             // [2][kCntSet] counters, one set per launch parity
  __nv_bfloat16* out;        // [M * 8, 4096]
  int32_t* e2p;              // [M * 8]
  uint16_t* expert_weights;  // [M * 8]
  unsigned* ticket;
  int32_t* err;
  int32_t m, pool, local_offset;
  int32_t n_dom0, n_dom1;
  int32_t flags;
  int32_t dbg;               // ablation (timing studies only): 1 no MMAs, 128 no epilogue roles, 256 no token copies,
                             // 512 no work items
  unsigned long long* stamps;  // [grid][kStampWords] per-CTA timestamps (timing builds only), or nullptr
#if MUSE_VR_LDW_RELAY
  float* relay;              // [kRelaySlots][128][kRelayCols] first-half fp32 accumulator tiles
#endif
};
constexpr int kFlagEvictFirst = 2, kFlagTailTrigger = 32;
constexpr int kStampWords = 24;
// Timing builds (-DMUSE_VR_LDW_STAMPS) include a stamp header that is not part of this package; production builds compile
// the stamps out.
#ifdef MUSE_VR_LDW_STAMPS
#include "ldw_stamps.cuh"
#else
#define LDW_STAMP(p, slot) ((void)0)
#define LDW_STAMPV(p, slot, v) ((void)0)
#endif
constexpr int kStStart = 0, kStPro = 1, kStCopy = 2, kStRows = 3, kStFirstFull = 4, kStFc1Issued = 5, kStFc2Dep = 6,
              kStAllIssued = 7, kStMma = 8, kStEpi = 9, kStBuild = 10, kStEnd = 11, kStEpiFc1 = 12, kStCopyRows = 13,
              kStCopyImg = 14, kStCopyBar = 15, kStItems = 16, kStCost = 17, kStSm = 18, kStCopyIn = 21, kStCopyLd = 22;

// ---------------------------------------------------------------- shared memory
struct Smem {
  alignas(1024) uint8_t st[kStages][kStageBytes];  // [weights 32K | token rows (up to kPassRows) x 256 B]
  uint64_t full[kStages], empty[kStages];
  uint64_t tmem_full[2], tmem_empty[2], drain_mma;
#if MUSE_VR_LDW_RELAY
  uint64_t acc_ready[2];             // second-half items: the relayed accumulator is in tensor memory (4 epilogue warps)
  int16_t qord[kMaxPasses];          // this domain's passes: first-half passes (expert parity == domain) first
#endif
  alignas(16) uint16_t ostage[4][8][32];  // FC2 epilogue: 8 rows x 32 output columns per epilogue warp
  int16_t q_expert[kMaxPasses], q_row[kMaxPasses], q_rows[kMaxPasses];
  int16_t base[kMaxPool], qbase[kMaxPool];  // first permuted row / first pass of each local expert
  int16_t my_slot[kMaxMySlots], my_pos[kMaxMySlots], my_exp[kMaxMySlots];
  int16_t ptok[kHeadItems][kHeadRows];  // unused (no head-copy path); kept for the shared prologue
  int32_t head_n;
  uint32_t items[kMaxItems];
  int32_t n_items, n_fc1, n_pass, n_my, set;
  uint32_t fc1_stored[4];  // per FC1 epilogue warp: tiles whose activation words it stored (red.release / ld.acquire, cta)
  uint32_t tmem_base;
  unsigned start_base;
  unsigned slot;  // this CTA's claimed slot (domain << 7 | rank), set in the prologue
};
static_assert(kStageBytes % 1024 == 0 && kStageA % 1024 == 0, "stage parts stay 1024-byte aligned");
// prologue scratch (the last ring stage, before its first copy): expert of each slot, per-chunk expert counts
struct Scratch {
  int16_t exp[kMaxSlots];
  uint32_t ccnt[kNumChunks][kMaxPool];
  uint32_t total[kMaxPool];
};
static_assert(sizeof(Scratch) <= kStageBytes, "prologue scratch fits one ring stage");

__device__ __forceinline__ uint32_t item_pack(int fc2, int q, int tile) {
  return uint32_t(q) | (uint32_t(tile) << 8) | (uint32_t(fc2) << 14);
}
struct Item {
  int fc2, q, tile, half;
};
__device__ __forceinline__ Item item_unpack(uint32_t v) {
  Item it;
  it.q = int(v & 0xFFu);
  it.tile = int((v >> 8) & 0x3Fu);
  it.fc2 = int((v >> 14) & 1u);
  it.half = int((v >> 15) & 1u);
  return it;
}
__device__ __forceinline__ int steps_of(int units, int rank, int nr) {
  return units > rank ? (units - 1 - rank) / nr + 1 : 0;
}
// Bounded acquire poll of a counter (one lane); false when the bound ran out or the CTA aborted.
// Relaxed polls until the value is reached, then one acquire load (the acquire pattern of PTX ISA: the acquire reads
// the same location after the relaxed observations).
__device__ __forceinline__ bool acquire_at_least(const unsigned* c, unsigned want, int32_t* err) {
  uint32_t polls = 0;
  while (muse_vr_ldq::ld_relaxed_gpu(c) < want) {
    if (aborted()) return false;
    if (++polls > kMaxPolls) {
      fail(err, kErrPoll);
      return false;
    }
    __nanosleep(64);
  }
  while (ld_acquire_gpu_u32(c) < want) {  // counts only grow: the first acquire normally observes it
    if (aborted() || ++polls > kMaxPolls) {
      if (!aborted()) fail(err, kErrPoll);
      return false;
    }
  }
  return true;
}
// Two 16-lane x 16-column loads (lane bases t0 and t1), then one wait. Fragment of .16x256b (PTX ISA tcgen05 matrix
// fragments): thread t holds lane t / 4 (registers 0, 1) and lane t / 4 + 8 (registers 2, 3), columns 2 (t % 4) and
// 2 (t % 4) + 1; .x2 repeats it for the next 8 columns (registers 4..7).
__device__ __forceinline__ void tmem_ld_16x256b_x2_pair(uint32_t t0, uint32_t t1, uint32_t (&a)[8], uint32_t (&b)[8]) {
  asm volatile(
      "tcgen05.ld.sync.aligned.16x256b.x2.b32 {%0,%1,%2,%3,%4,%5,%6,%7}, [%16];\n\t"
      "tcgen05.ld.sync.aligned.16x256b.x2.b32 {%8,%9,%10,%11,%12,%13,%14,%15}, [%17];\n\t"
      "tcgen05.wait::ld.sync.aligned;"
      : "=r"(a[0]), "=r"(a[1]), "=r"(a[2]), "=r"(a[3]), "=r"(a[4]), "=r"(a[5]), "=r"(a[6]), "=r"(a[7]), "=r"(b[0]),
        "=r"(b[1]), "=r"(b[2]), "=r"(b[3]), "=r"(b[4]), "=r"(b[5]), "=r"(b[6]), "=r"(b[7])
      : "r"(t0), "r"(t1)
      : "memory");
}
__device__ __forceinline__ void tmem_ld_x4(uint32_t taddr, uint32_t (&r)[4]) {
  asm volatile("tcgen05.ld.sync.aligned.32x32b.x4.b32 {%0,%1,%2,%3}, [%4];\n\ttcgen05.wait::ld.sync.aligned;"
               : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3])
               : "r"(taddr)
               : "memory");
}
__device__ __forceinline__ void tmem_ld_x16(uint32_t taddr, uint32_t (&r)[16]) {
  asm volatile(
      "tcgen05.ld.sync.aligned.32x32b.x16.b32 {%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15}, [%16];\n\t"
      "tcgen05.wait::ld.sync.aligned;"
      : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]), "=r"(r[4]), "=r"(r[5]), "=r"(r[6]), "=r"(r[7]), "=r"(r[8]),
        "=r"(r[9]), "=r"(r[10]), "=r"(r[11]), "=r"(r[12]), "=r"(r[13]), "=r"(r[14]), "=r"(r[15])
      : "r"(taddr)
      : "memory");
}
__device__ __forceinline__ void tmem_ld_x8(uint32_t taddr, uint32_t (&r)[8]) {
  asm volatile("tcgen05.ld.sync.aligned.32x32b.x8.b32 {%0,%1,%2,%3,%4,%5,%6,%7}, [%8];\n\ttcgen05.wait::ld.sync.aligned;"
               : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]), "=r"(r[4]), "=r"(r[5]), "=r"(r[6]), "=r"(r[7])
               : "r"(taddr)
               : "memory");
}

#if MUSE_VR_LDW_RELAY
__device__ __forceinline__ void tmem_st_x16(uint32_t taddr, const uint32_t (&r)[16]) {
  asm volatile(
      "tcgen05.st.sync.aligned.32x32b.x16.b32 [%0], {%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16};" ::"r"(taddr),
      "r"(r[0]), "r"(r[1]), "r"(r[2]), "r"(r[3]), "r"(r[4]), "r"(r[5]), "r"(r[6]), "r"(r[7]), "r"(r[8]), "r"(r[9]), "r"(r[10]),
      "r"(r[11]), "r"(r[12]), "r"(r[13]), "r"(r[14]), "r"(r[15])
      : "memory");
}
__device__ __forceinline__ int relay_slot(const Item& it) {
  return it.fc2 ? kRelayPasses * 64 + it.q * 32 + it.tile : it.q * 64 + it.tile;
}
#endif
// ---------------------------------------------------------------- roles
// warp 0: weight and token-operand copies, one stage at a time
__device__ MUSE_VR_LDW_ROLE void role_producer(Smem& s, const Params& p, const CUtensorMap* m13, const CUtensorMap* m2,
                                               const CUtensorMap* b1, const CUtensorMap* b2) {
  const bool leader = elect_one();
  const int n_items = s.n_items;
  const bool ef = (p.flags & kFlagEvictFirst) != 0;
  const bool tok = (p.dbg & 256) == 0;
  const uint64_t pol = policy_evict_first();
  unsigned* const cset = p.cnt + s.set * kCntSet;
  if (n_items == 0 && (p.flags & kFlagTailTrigger)) asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
  int k = 0, issued = 0;
  bool rows_ready = false;
  // the last ring stage held the prologue scratch (generic accesses, ordered before this point by the CTA barrier): a
  // proxy fence before any asynchronous copy into the ring (PTX ISA memory consistency model, proxies)
  fence_proxy_async_smem();
  // the weights of the first kStages stages go out at once (free slots); their token rows wait for the rows-ready count
  int pre = 0;
  if (tok && leader && !aborted()) {
    pre = min(kStages, n_items * kStagesPerItem);
#pragma unroll 1
    for (int x = 0; x < pre; ++x) {
      const Item it = item_unpack(s.items[x / kStagesPerItem]);
      const int e = s.q_expert[it.q], j = x % kStagesPerItem + (kStagesPerTile / 2) * it.half;
      const uint32_t dst = smem_u32(&s.st[x][0]);
      mbar_expect_tx(&s.full[x], uint32_t(kStageA));
      if (ef)
        tma_load_4d_hint(dst, it.fc2 ? m2 : m13, 0, it.tile * kTile, j * 2, e, &s.full[x], pol);
      else
        tma_load_4d(dst, it.fc2 ? m2 : m13, 0, it.tile * kTile, j * 2, e, &s.full[x]);
    }
  }
#pragma unroll 1
  for (int m = 0; m < n_items; ++m) {
    const Item it = item_unpack(s.items[m]);
    const int e = s.q_expert[it.q], rows = s.q_rows[it.q], qrow = s.q_row[it.q];
    const int bk = bucket_of(rows), nb = bucket_rows(bk);
    if (m + 1 == n_items && (p.flags & kFlagTailTrigger)) asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
    bool fc2_ready = !it.fc2 || (p.dbg & 128) != 0;
    const CUtensorMap* wmap = it.fc2 ? m2 : m13;
    const CUtensorMap* bmap = (it.fc2 ? b2 : b1) + bk;
    const uint32_t tok_tx = uint32_t(nb * 256);
#pragma unroll 1
    for (int jj = 0; jj < kStagesPerItem; ++jj, ++k) {
      const int j = jj + (kStagesPerTile / 2) * it.half;  // K stage within the tile
      const int pos = k % kStages;
      const uint32_t use = uint32_t(k / kStages);
      if (k >= kStages) mbar_wait(&s.empty[pos], (use & 1u) ^ 1u, p.err);
      if (leader && issued == k && !aborted()) {
        const uint32_t dst = smem_u32(&s.st[pos][0]);
        if (k >= pre) {
          mbar_arrive_expect_tx(&s.full[pos], uint32_t(kStageA) + (tok ? tok_tx : 0u));
          // weights first: they depend on nothing
          if (ef)
            tma_load_4d_hint(dst, wmap, 0, it.tile * kTile, j * 2, e, &s.full[pos], pol);
          else
            tma_load_4d(dst, wmap, 0, it.tile * kTile, j * 2, e, &s.full[pos]);
        } else {
          mbar_arrive_expect_tx(&s.full[pos], tok ? tok_tx : 0u);  // weights already out (expect-tx above)
        }
        if (tok) {
          // token rows (FC1) / activation rows (FC2) are stored by other CTAs with generic stores: acquire their counts,
          // then a proxy fence orders the asynchronous-proxy copies after them (FC1: the grid's rows-ready count; FC2:
          // the pass's FC1 tiles, published by every CTA's warp 3)
          if (!it.fc2 && !rows_ready) {
            rows_ready = acquire_at_least(cset + kRowsReady, gridDim.x, p.err) || true;
            LDW_STAMP(p, kStRows);
          }
          if (!fc2_ready) fc2_ready = acquire_at_least(cset + it.q, unsigned(kFc1Arrivals), p.err) || true;
          fence_proxy_async_global();
          tma_load_3d(dst + kStageA, bmap, 0, qrow, j * 2, &s.full[pos]);
        }
        ++issued;
      }
      __syncwarp();
    }
  }
  if (leader) LDW_STAMP(p, kStAllIssued);
  if (leader && aborted()) {
    // pre-issued weight stages without their arrival: arrive, so the phase completes once the weights land
#pragma unroll 1
    for (int kk = issued; kk < pre; ++kk) mbar_arrive(&s.full[kk]);
    const int out = issued > pre ? issued : pre;
#pragma unroll 1
    for (int kk = out > kStages ? out - kStages : 0; kk < out; ++kk)
      mbar_drain(&s.full[kk % kStages], uint32_t(kk / kStages) & 1u, p.err);
  }
  __syncwarp();
}

// warp 1: MMA issue (N = the pass's token-operand height)
__device__ MUSE_VR_LDW_ROLE void role_mma(Smem& s, const Params& p) {
  const bool leader = elect_one();
  const uint32_t tbase = s.tmem_base;
  const int n_items = s.n_items;
  int k = 0;
#if MUSE_VR_LDW_RELAY
  uint32_t rph[2] = {0u, 0u};
#endif
#pragma unroll 1
  for (int m = 0; m < n_items; ++m) {
    const Item it = item_unpack(s.items[m]);
    const int nb = bucket_rows(bucket_of(s.q_rows[it.q]));
    const uint32_t idesc = idesc_bf16(uint32_t(nb));
    const int acc = m & 1;
    const uint32_t aph = uint32_t(m >> 1) & 1u;
    if ((p.dbg & 128) == 0 && m >= 2) mbar_wait(&s.tmem_empty[acc], aph ^ 1u, p.err);
#if MUSE_VR_LDW_RELAY
    if (it.half && (p.dbg & 128) == 0) {  // the first half's accumulator, relayed into this tensor-memory accumulator
      mbar_wait(&s.acc_ready[acc], rph[acc] & 1u, p.err);
      ++rph[acc];
    }
#endif
    tc_fence_after();
    const uint32_t d = tbase + kAccCol + uint32_t(acc) * 128u;
#pragma unroll 1
    for (int j = 0; j < kStagesPerItem; ++j, ++k) {
      const int pos = k % kStages;
      mbar_wait(&s.full[pos], uint32_t(k / kStages) & 1u, p.err);
      if (m == 0 && j == 0 && leader) LDW_STAMP(p, kStFirstFull);
      tc_fence_after();
      if (leader) {
        if ((p.dbg & 1) == 0 && !aborted()) {
          const uint64_t da0 = sdesc_sw128(smem_u32(&s.st[pos][0]));
          const uint64_t db0 = sdesc_sw128(smem_u32(&s.st[pos][kStageA]));
          // 8 MMAs of K = 16 (32 bytes of each row): 128-byte K atoms of 128 weight rows (16 KiB) / nb token rows
#pragma unroll
          for (int c = 0; c < kChunks; ++c) {
            const uint64_t da = da0 + uint64_t((uint32_t(c >> 2) * 16384u + uint32_t(c & 3) * 32u) >> 4);
            const uint64_t db = db0 + uint64_t((uint32_t(c >> 2) * uint32_t(nb * 128) + uint32_t(c & 3) * 32u) >> 4);
            tc_mma_bf16(d, da, db, idesc, (it.half != 0 || (j | c) != 0) ? 1u : 0u);
          }
        }
        tc_commit(&s.empty[pos]);
      }
      __syncwarp();
    }
    if (leader) tc_commit(&s.tmem_full[acc]);
    __syncwarp();
  }
  if (leader) tc_commit(&s.drain_mma);
  __syncwarp();
  mbar_drain(&s.drain_mma, 0u, p.err);
  if (leader) LDW_STAMP(p, kStMma);
}

// Warps 4-11 (epilogue + idle warps, before their roles): this CTA's share of the permuted token rows (slots b, b + G, ...):
// BF16 rows of 8 KiB, one warp per row, 16-byte pieces with several loads in flight per thread, the expanded-index map and
// top-k weights of those slots; then one release of the grid's rows-ready count on behalf of the eight warps.
__device__ __noinline__ void copy_rows(Smem& s, const Params& p, int wt) {  // one shared copy of this cold code
  constexpr int kW = 256;  // worker threads
  const int n_my = s.n_my;
  const int lane = wt & 31, wi = wt >> 5;
  const bool tail = wt < n_my;
  const int sl_t = tail ? s.my_slot[wt] : 0;
  const int32_t pk_t = tail ? p.packed[sl_t] : 0;
  if (wt == 0) LDW_STAMP(p, kStCopyIn);
#pragma unroll 1
  for (int j = wi; j < n_my; j += kW / 32) {
    const int pos = s.my_pos[j];
    if (pos < 0) continue;  // warp-uniform
    const int t = s.my_slot[j] / kTopK;
    const uint4* src = reinterpret_cast<const uint4*>(p.hs + size_t(t) * kRowBytes);
    uint4* dst = reinterpret_cast<uint4*>(p.hs_perm + size_t(pos) * kRowBytes);
#pragma unroll
    for (int h = 0; h < 2; ++h) {  // 512 pieces of 16 bytes: two rounds of 8 loads in flight per lane
      uint4 v[8];
#pragma unroll
      for (int u = 0; u < 8; ++u) v[u] = src[lane + 32 * (8 * h + u)];
#pragma unroll
      for (int u = 0; u < 8; ++u) dst[lane + 32 * (8 * h + u)] = v[u];
    }
  }
  if (wt == 0) LDW_STAMP(p, kStCopyRows);
  if (tail) {
    p.e2p[sl_t] = s.my_pos[wt];
    p.expert_weights[sl_t] = uint16_t(pk_t & 0xFFFF);
  }
  named_bar_sync(7, kW);  // every worker's stores, before one release on their behalf
  if (wt == 0) {
    asm volatile("red.release.gpu.global.add.u32 [%0], 1;" ::"l"(p.cnt + s.set * kCntSet + kRowsReady) : "memory");
    LDW_STAMP(p, kStCopy);
  }
}

// warps 4-7: epilogue (TMEM lane quadrant = warp % 4): 16 (FC1) / 8 (FC2) token columns per tensor-memory load
__device__ MUSE_VR_LDW_ROLE void role_epilogue(Smem& s, const Params& p) {
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
  const int q = warp & 3;
  const bool leader = elect_one();
  const uint32_t tbase = s.tmem_base;
  const int n_items = s.n_items;
  copy_rows(s, p, threadIdx.x - 4 * 32);
  if (p.dbg & 128) return;
#if MUSE_VR_LDW_RELAY
  unsigned* const cset = p.cnt + s.set * kCntSet;
  // second-half item mi: acquire the first half's four warp counts, load this warp's 32 rows of the fp32 accumulator tile into
  // the item's tensor-memory accumulator (whose previous item's epilogue already ran), then one arrival per warp
  auto preload = [&](int mi) {
    const Item jt = item_unpack(s.items[mi]);
    if (!jt.half) return;
    const int nbj = bucket_rows(bucket_of(s.q_rows[jt.q]));
    const int slot = relay_slot(jt);
    if (lane == 0) acquire_at_least(cset + kRelayCnt + slot, 4u, p.err);
    __syncwarp();  // memory ordering among the warp's threads: the loads below follow lane 0's acquire
    const float* src = p.relay + (size_t(slot) * 128 + size_t(q * 32 + lane)) * kRelayCols;
    const uint32_t ta = tbase + (uint32_t(q * 32) << 16) + kAccCol + uint32_t(mi & 1) * 128u;
#pragma unroll 1
    for (int c0 = 0; c0 < nbj; c0 += 16) {
      uint32_t r[16];
#pragma unroll
      for (int u = 0; u < 4; ++u) {
        const uint4 v = reinterpret_cast<const uint4*>(src + c0)[u];
        r[4 * u] = v.x;
        r[4 * u + 1] = v.y;
        r[4 * u + 2] = v.z;
        r[4 * u + 3] = v.w;
      }
      tmem_st_x16(ta + uint32_t(c0), r);
    }
    tc_wait_st();
    tc_fence_before();
    __syncwarp();
    if (leader) mbar_arrive(&s.acc_ready[mi & 1]);
  };
  // a second-half item's preload may wait only once every earlier item's epilogue ran: a wait ahead of an earlier FC1
  // epilogue could wait on a first half that itself waits for that FC1 publication. Item done + 2 is tried without waiting
  // (one relaxed read); when not yet available it is deferred to the next item's end.
  auto relay_ready = [&](int mi) -> bool {
    unsigned v = 0;
    if (lane == 0) v = muse_vr_ldq::ld_relaxed_gpu(cset + kRelayCnt + relay_slot(item_unpack(s.items[mi])));
    return __shfl_sync(0xffffffffu, v, 0) >= 4u;
  };
  int pend = (n_items > 0 && item_unpack(s.items[0]).half) ? 0 : -1;
  auto advance = [&](int done) {  // after item done's epilogue (done = -1: before the first item)
    if (pend == done + 1) {
      preload(pend);
      pend = -1;
    }
    const int mi = done + 2;
    if (mi < n_items && item_unpack(s.items[mi]).half) {
      if (relay_ready(mi))
        preload(mi);
      else
        pend = mi;
    }
  };
  advance(-1);
#endif
#pragma unroll 1
  for (int m = 0; m < n_items; ++m) {
    const int acc = m & 1;
    const uint32_t aph = uint32_t(m >> 1) & 1u;
    const Item it = item_unpack(s.items[m]);
    const int rows = s.q_rows[it.q], qrow = s.q_row[it.q];
    mbar_wait_idle(&s.tmem_full[acc], aph, p.err);
    tc_fence_after();
    const uint32_t tacc = tbase + (uint32_t(q * 32) << 16) + kAccCol + uint32_t(acc) * 128u;
#if MUSE_VR_LDW_RELAY
    if (!it.half) {
      // first half: this warp's 32 rows of the exact fp32 accumulator tile to the relay buffer, then one release per warp
      const int nbi = bucket_rows(bucket_of(rows));
      float* dst = p.relay + (size_t(relay_slot(it)) * 128 + size_t(q * 32 + lane)) * kRelayCols;
#pragma unroll 1
      for (int c0 = 0; c0 < nbi; c0 += 16) {
        uint32_t r[16];
        tmem_ld_x16(tacc + uint32_t(c0), r);
#pragma unroll
        for (int u = 0; u < 4; ++u)
          reinterpret_cast<uint4*>(dst + c0)[u] = make_uint4(r[4 * u], r[4 * u + 1], r[4 * u + 2], r[4 * u + 3]);
      }
      tc_fence_before();
      __syncwarp();  // the warp's stores before lane 0's release
      if (lane == 0) asm volatile("red.release.gpu.global.add.u32 [%0], 1;" ::"l"(cset + kRelayCnt + relay_slot(it)) : "memory");
      if (leader) mbar_arrive(&s.tmem_empty[acc]);
      advance(m);
      continue;
    }
#endif
    if (!it.fc2) {
      // FC1: SwiGLU -> activation words of 4 bf16, one routed row per column (layout as in moe_ldw.cuh)
      const int ta = lane >> 2, tx = 2 * (lane & 3);
      uint32_t* dst = reinterpret_cast<uint32_t*>(p.act + size_t(qrow) * kActWords + size_t(it.tile) * 16) + (8 * q + ta);
      const uint32_t lb = tbase + (uint32_t(q * 32) << 16) + kAccCol + uint32_t(acc) * 128u;
#pragma unroll 1
      for (int x0 = 0; x0 < rows; x0 += 16) {
        uint32_t v0[8], v1[8];
        tmem_ld_16x256b_x2_pair(lb + uint32_t(x0), lb + (16u << 16) + uint32_t(x0), v0, v1);
#pragma unroll
        for (int h = 0; h < 2; ++h) {
#pragma unroll
          for (int c = 0; c < 2; ++c) {
            const int x = x0 + 8 * h + tx + c;
            if (x < rows) {
              uint32_t y0 = swiglu_out(__uint_as_float(v0[4 * h + 2 + c]), __uint_as_float(v0[4 * h + c]));
              uint32_t y1 = swiglu_out(__uint_as_float(v1[4 * h + 2 + c]), __uint_as_float(v1[4 * h + c]));
              y0 = (y0 & 0x7FFFu) > 0x7F80u ? 0x7FFFu : y0;  // canonical NaN
              y1 = (y1 & 0x7FFFu) > 0x7F80u ? 0x7FFFu : y1;
              asm volatile("st.relaxed.gpu.global.b32 [%0], %1;" ::"l"(dst + size_t(x) * (kActWords * 2)), "r"(y0 | (y1 << 16))
                           : "memory");
            }
          }
        }
      }
      tc_fence_before();
      __syncwarp();
      if (leader) {
        mbar_arrive(&s.tmem_empty[acc]);
        asm volatile("red.release.cta.shared::cta.add.u32 [%0], 1;" ::"r"(smem_u32(&s.fc1_stored[MUSE_VR_SLOT_UNSAFE ? 0 : q]))
                     : "memory");
        if (q == 0 && m + 1 == s.n_fc1) LDW_STAMP(p, kStEpiFc1);
      }
    } else {
      // FC2: output rows (fp32 accumulator -> bf16); 8 rows staged per warp, then one 16-byte store per lane
      const uint32_t col = uint32_t((lane & 7) * 4 + (lane >> 3));
      const uint32_t st0 = smem_u32(&s.ostage[q][0][0]);
      uint8_t* dst = reinterpret_cast<uint8_t*>(p.out) + (size_t(it.tile) * kTile + size_t(32 * q)) * 2;
#pragma unroll 1
      for (int x0 = 0; x0 < rows; x0 += 8) {
        uint32_t v[8];
        tmem_ld_x8(tacc + uint32_t(x0), v);
#pragma unroll
        for (int i = 0; i < 8; ++i) sts_u16(st0 + uint32_t(i * 64) + col * 2u, uint16_t(bf16_satfinite(__uint_as_float(v[i]))));
        __syncwarp();
        const int i = lane >> 2;  // row of this lane's 16 bytes
        if (x0 + i < rows)
          *reinterpret_cast<uint4*>(dst + size_t(qrow + x0 + i) * kHidden * 2 + size_t(lane & 3) * 16) =
              lds_v4(st0 + uint32_t(i * 64) + uint32_t(lane & 3) * 16u);
        __syncwarp();
      }
      tc_fence_before();
      __syncwarp();
      if (leader) mbar_arrive(&s.tmem_empty[acc]);
    }
#if MUSE_VR_LDW_RELAY
    advance(m);
#endif
  }
  if (q == 0 && leader) LDW_STAMP(p, kStEpi);
}
// warp 3 after the prologue: FC1 publication per tile, residency exit wait
__device__ MUSE_VR_LDW_ROLE void role_ctrl(Smem& s, const Params& p) {
  const int lane = threadIdx.x & 31;
  unsigned* const cset = p.cnt + s.set * kCntSet;
  if ((p.dbg & 128) == 0 && lane == 0) {
    int stored = 0;  // FC1 items whose activation words the epilogue stores (every FC1 item; second halves only with the relay)
#pragma unroll 1
    for (int mm = 0; mm < s.n_fc1; ++mm) {
      if (MUSE_VR_LDW_RELAY && item_unpack(s.items[mm]).half == 0) continue;
      const int need = ++stored;
      uint32_t polls = 0, done = 0;
      bool ok = true;
      while (true) {
        #if MUSE_VR_SLOT_UNSAFE
        // previous form: one CTA-wide count (assumes the four epilogue warps finish FC1 tiles in step)
        asm volatile("ld.relaxed.cta.shared::cta.u32 %0, [%1];" : "=r"(done) : "r"(smem_u32(&s.fc1_stored[0])) : "memory");
        if (done >= uint32_t(4 * need)) break;
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
        if (done >= uint32_t(need)) break;
        #endif
        if (aborted()) {
          ok = false;
          break;
        }
        if (++polls > kMaxPolls) {
          fail(p.err, kErrMbar);
          ok = false;
          break;
        }
        __nanosleep(64);
      }
      if (!ok) break;
      asm volatile("fence.acq_rel.cta;" ::: "memory");  // acquire pattern (cta)
      const Item it = item_unpack(s.items[mm]);
      asm volatile("red.release.gpu.global.add.u32 [%0], 4;" ::"l"(cset + it.q) : "memory");
    }
  }
  __syncwarp();
#if !MUSE_VR_SLOT_UNSAFE && !MUSE_VR_SLOT_EPOCH
  // this CTA's claim (prologue, before named barrier 3) is complete: the last CTA checks and clears the claim bitmap
  if (lane == 0) muse_vr_slot::finish(p.ticket + kClaimWord, p.err);
#endif
  if (lane == 0 && !aborted()) {
    const unsigned cycle = gridDim.x * kStartCycle;
    uint32_t polls = 0;
    for (;;) {
      const unsigned v = muse_vr_ldq::ld_relaxed_gpu(p.ticket);
      const unsigned d = v >= s.start_base ? v - s.start_base : v + cycle - s.start_base;
      if (d >= gridDim.x) break;
      if (aborted()) break;
      if (++polls > kMaxPolls) {
        fail(p.err, kErrStart);
        break;
      }
      __nanosleep(256);
    }
  }
  __syncwarp();
}

// Prologue, every thread of the CTA: counting the local routed slots per expert (per 256-slot chunk), expert bases and
// passes, this CTA's work list (warp 0) and the permuted positions of the slots whose rows this CTA copies (warps 8-11).
// The scratch lives in the last ring stage, which no copy targets before the work list is published (named barrier 3).
__device__ MUSE_VR_LDW_ROLE void prologue(Smem& s, const Params& p) {
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
  Scratch& x = *reinterpret_cast<Scratch*>(&s.st[kStages - 1][0]);
  const int nslot = p.m * kTopK;
  // Start ticket (residency, counter parity) and slot claim: both are atomic RMWs on control words that the previous
  // launch of this call site updates (its last CTA clears the claim bitmap in finish()). They are issued right AFTER
  // griddepcontrol.wait: under programmatic dependent launch the wait is the documented edge that orders them after the
  // completion of every prerequisite grid (PTX ISA, griddepcontrol), so the previous launch's clear is ordered before
  // this launch's claim. Issued before the routed-ids loads below, so their latency still overlaps those loads and the
  // ccnt zeroing (= moe_ldw.cuh's placement). -DMUSE_VR_LDW_TEST_PREWAIT_CLAIM=1 (tests only: the negative
  // control of tests/test_moe_ldwb_order.py) keeps the earlier pre-wait placement.
  unsigned ticket = 0;
#if !MUSE_VR_SLOT_UNSAFE && MUSE_VR_SLOT_EPOCH
  // launch-counter claim (slot_claim.cuh): issued by thread 0 after its routed-ids loads below (an RMW issued at
  // launch delays those loads), settled when the work list needs it
  unsigned ep_smid = 0, ep_old = 0, ep_hint = 0;
#endif
#if !MUSE_VR_SLOT_UNSAFE && !MUSE_VR_SLOT_EPOCH
  unsigned claim_hint = 0, claim_old = 0;
#endif
#if MUSE_VR_LDW_TEST_PREWAIT_CLAIM
  if (tid == 0) ticket = atomicInc(p.ticket, gridDim.x * kStartCycle - 1u);
#if !MUSE_VR_SLOT_UNSAFE && !MUSE_VR_SLOT_EPOCH
  if (tid == 0) {
    claim_hint = unsigned(reinterpret_cast<const uint8_t*>(p.sm_domain)[muse_vr_slot::read_smid()]);
    claim_old = muse_vr_slot::try_hint(p.ticket + kClaimWord, claim_hint, p.n_dom0, p.n_dom1);
  }
#endif
#endif
  grid_dep_wait();  // ids and token rows come from predecessors
#if !MUSE_VR_LDW_TEST_PREWAIT_CLAIM
  // start ticket: its value is first needed at the parity step below
  if (tid == 0) ticket = atomicInc(p.ticket, gridDim.x * kStartCycle - 1u);
#if !MUSE_VR_SLOT_UNSAFE && !MUSE_VR_SLOT_EPOCH
  // slot claim (slot_claim.cuh): the SM's slot is the hint; settled when the work list needs it
  if (tid == 0) {
    claim_hint = unsigned(reinterpret_cast<const uint8_t*>(p.sm_domain)[muse_vr_slot::read_smid()]);
    claim_old = muse_vr_slot::try_hint(p.ticket + kClaimWord, claim_hint, p.n_dom0, p.n_dom1);
  }
#endif
#endif
  for (int i = tid; i < kNumChunks * kMaxPool; i += kThreads) (&x.ccnt[0][0])[i] = 0u;
  named_bar_sync(6, kThreads);
  {
    // all of this thread's ids in flight at once, then the counts
    constexpr int kPer = (kMaxSlots + kThreads - 1) / kThreads;
    int32_t v[kPer];
#pragma unroll
    for (int u = 0; u < kPer; ++u) {
      const int sl = tid + u * kThreads;
      v[u] = sl < nslot ? p.packed[sl] : -1;
    }
#if !MUSE_VR_SLOT_UNSAFE && MUSE_VR_SLOT_EPOCH
    if (tid == 0) {
      const uint8_t* const table = reinterpret_cast<const uint8_t*>(p.sm_domain);
      ep_smid = muse_vr_slot::epoch_test_smid(table, muse_vr_slot::read_smid());
      ep_old = muse_vr_slot::epoch_try_sm(p.ticket + 32, ep_smid, kStartCycle);  // counters: words 32.. of ctl
      ep_hint = ep_smid < gridDim.x ? unsigned(table[ep_smid]) : muse_vr_slot::kInvalid;
    }
#endif
#pragma unroll
    for (int u = 0; u < kPer; ++u) {
      const int sl = tid + u * kThreads;
      if (sl < nslot) {
        const int e = int((v[u] >> 16) & 0xFFFF) - p.local_offset;
        const bool loc = e >= 0 && e < p.pool;
        x.exp[sl] = int16_t(loc ? e : -1);
        if (loc) atomicAdd(&x.ccnt[sl / kChunkSlots][e], 1u);
      }
    }
  }
  named_bar_sync(6, kThreads);
  if (tid < kMaxPool) {  // per expert: exclusive prefix over chunks (in place) and the total
    uint32_t run = 0;
    for (int c = 0; c < kNumChunks; ++c) {
      const uint32_t n = x.ccnt[c][tid];
      x.ccnt[c][tid] = run;
      run += n;
    }
    x.total[tid] = tid < p.pool ? run : 0u;
  }
  named_bar_sync(6, kThreads);
  if (warp == 0) {  // expert bases and passes (ascending expert order), pass tables
    int rows = 0, q = 0;
#pragma unroll 1
    for (int c = 0; c < kMaxPool; c += 32) {
      const int e = c + lane;
      const int n = int(x.total[e]);
      const int np = (n + kPassRows - 1) / kPassRows;
      int incl = n, inclq = np;
#pragma unroll
      for (int o = 1; o < 32; o <<= 1) {
        const int y = __shfl_up_sync(0xffffffffu, incl, o), yq = __shfl_up_sync(0xffffffffu, inclq, o);
        if (lane >= o) {
          incl += y;
          inclq += yq;
        }
      }
      const int b = rows + incl - n, qb = q + inclq - np;
      s.base[e] = int16_t(b);
      s.qbase[e] = int16_t(qb);
      for (int pq = 0; pq < np; ++pq) {
        if (qb + pq < kMaxPasses) {
          s.q_expert[qb + pq] = int16_t(e);
          s.q_row[qb + pq] = int16_t(b + pq * kPassRows);
          s.q_rows[qb + pq] = int16_t(min(kPassRows, n - pq * kPassRows));
        }
      }
      rows += __shfl_sync(0xffffffffu, incl, 31);
      q += __shfl_sync(0xffffffffu, inclq, 31);
    }
    if (lane == 0) {
      s.n_pass = q;
      s.head_n = 0;
      s.start_base = ticket - ticket % gridDim.x;
      s.set = int((s.start_base / gridDim.x) & 1u);
#if !MUSE_VR_SLOT_UNSAFE && MUSE_VR_SLOT_EPOCH
      s.slot = muse_vr_slot::epoch_settle_sm(p.ticket + 32, p.ticket + kClaimWord + muse_vr_slot::kMissWord,
                                             reinterpret_cast<const uint8_t*>(p.sm_domain), ep_smid, ep_hint, ep_old,
                                             ticket / gridDim.x, kStartCycle, p.n_dom0, p.n_dom1, p.err);
#elif !MUSE_VR_SLOT_UNSAFE
      s.slot = muse_vr_slot::settle(p.ticket + kClaimWord, claim_hint, claim_old, p.n_dom0, p.n_dom1, p.err);
#endif
    }
  }
  named_bar_sync(6, kThreads);
  if (warp == 0) {  // this CTA's work list: one round-robin over the domain list FC1(all passes) then FC2(all passes)
    unsigned slot = 0;
#if MUSE_VR_SLOT_UNSAFE
    if (lane == 0) {
      unsigned smid;
      asm volatile("mov.u32 %0, %%smid;" : "=r"(smid));
      slot = unsigned(reinterpret_cast<const uint8_t*>(p.sm_domain)[smid]);
    }
    slot = shfl_idx(slot, 0);
#else
    slot = s.slot;  // claimed in the prologue (named barrier 6 above)
#endif
    int nq = s.n_pass;
    if (nq > kMaxPasses) {
      if (lane == 0) fail(p.err, kErrShape);
      nq = 0;
    }
    const unsigned dom = (slot >> 7) & 1u;
    const int R = dom ? p.n_dom1 : p.n_dom0, r = int(slot & 127u);
#if MUSE_VR_LDW_RELAY
    // every tile's two K halves: the domain holding an expert's first K half (expert parity) runs the first halves; this
    // domain's pass order = its first-half passes, then the others (their first halves run on the other domain meanwhile)
    if (nq > kRelayPasses) {
      if (lane == 0) fail(p.err, kErrShape);
      nq = 0;
    }
    int na = 0;
#pragma unroll 1
    for (int b = 0; b < nq; b += 32) {
      const int qq = b + lane;
      const bool a = qq < nq && (int(s.q_expert[qq]) & 1) == int(dom);
      const unsigned bal = __ballot_sync(0xffffffffu, a);
      if (a) s.qord[na + __popc(bal & ((1u << lane) - 1u))] = int16_t(qq);
      na += __popc(bal);
    }
    int nbq = na;
#pragma unroll 1
    for (int b = 0; b < nq; b += 32) {
      const int qq = b + lane;
      const bool o = qq < nq && (int(s.q_expert[qq]) & 1) != int(dom);
      const unsigned bal = __ballot_sync(0xffffffffu, o);
      if (o) s.qord[nbq + __popc(bal & ((1u << lane) - 1u))] = int16_t(qq);
      nbq += __popc(bal);
    }
    __syncwarp();
    const int n1 = R > 0 ? steps_of(kFc1TilesPerExpert * nq, r, R) : 0;
    const int n2 = R > 0 ? steps_of((kFc1TilesPerExpert + kFc2TilesPerExpert) * nq, r, R) - n1 : 0;
#else
    const int n1 = R > 0 ? steps_of(kFc1TilesPerDomain * nq, r, R) : 0;
    const int n2 = R > 0 ? steps_of((kFc1TilesPerDomain + kFc2TilesPerDomain) * nq, r, R) - n1 : 0;
#endif
    int nt = (p.dbg & 512) ? 0 : n1 + n2;
    if (nt > kMaxItems) {
      if (lane == 0) fail(p.err, kErrSlot);
      nt = 0;
    }
#pragma unroll 1
    for (int i = lane; i < nt; i += 32) {
      uint32_t v;
#if MUSE_VR_LDW_RELAY
      if (i < n1) {
        const int g = r + i * R, ps = g / kFc1TilesPerExpert;
        v = item_pack(0, s.qord[ps], g % kFc1TilesPerExpert) | (uint32_t(ps >= na) << 15);
      } else {
        const int g = r + i * R - kFc1TilesPerExpert * nq, ps = g / kFc2TilesPerExpert;
        v = item_pack(1, s.qord[ps], g % kFc2TilesPerExpert) | (uint32_t(ps >= na) << 15);
      }
#else
      if (i < n1) {
        const int g = r + i * R;
        v = item_pack(0, g / kFc1TilesPerDomain, kFc1TilesPerDomain * int(dom) + (g % kFc1TilesPerDomain));
      } else {
        const int pc = r + i * R - kFc1TilesPerDomain * nq;
        v = item_pack(1, pc / kFc2TilesPerDomain, kFc2TilesPerDomain * int(dom) + (pc % kFc2TilesPerDomain));
      }
#endif
      s.items[i] = v;
    }
    if (lane == 0) {
      s.n_fc1 = nt > 0 ? n1 : 0;
      s.n_items = nt;
    }
  } else if (MUSE_VR_LDW_GATHER && warp == 3 && (p.dbg & 256) == 0 && s.n_pass <= kMaxPasses) {
    // head copies: the token of each row of the passes of this CTA's first FC1 items (item h computes tile g = r + h R of
    // pass g / 32), while a pass has at most kHeadRows rows. Row i of pass qq of expert e is e's slot of rank
    // (qq - qbase[e]) * 128 + i in slot order - the copy phase's stable expert-major order: the per-chunk prefix counts
    // give the chunk holding the pass's first row, then an ordered scan from there (8 independent ballots per chunk).
    unsigned slot = 0;
#if MUSE_VR_SLOT_UNSAFE
    if (lane == 0) {
      unsigned smid;
      asm volatile("mov.u32 %0, %%smid;" : "=r"(smid));
      slot = unsigned(reinterpret_cast<const uint8_t*>(p.sm_domain)[smid]);
    }
    slot = shfl_idx(slot, 0);
#else
    slot = s.slot;  // the slot warp 0's work list uses (claimed in the prologue)
#endif
    const int R = ((slot >> 7) & 1u) ? p.n_dom1 : p.n_dom0, r = int(slot & 127u);
    const int nq = s.n_pass;
    int hn = 0;
#pragma unroll 1
    for (int h = 0; h < kHeadItems; ++h) {
      const int g = r + h * R;
      if (R <= 0 || g >= kFc1TilesPerDomain * nq) break;  // warp-uniform
      const int qq = g / kFc1TilesPerDomain, e = s.q_expert[qq], rows = s.q_rows[qq];
      if (rows > kHeadRows) break;
      const int r0 = (qq - int(s.qbase[e])) * kPassRows;
      const int nch = (nslot + kChunkSlots - 1) / kChunkSlots;
      int c0 = 0;  // last chunk whose prefix count of e is <= r0: it holds e's slot of rank r0
#pragma unroll
      for (int st = kNumChunks / 2; st > 0; st >>= 1)
        if (c0 + st < nch && int(x.ccnt[c0 + st][e]) <= r0) c0 += st;
      int seen = int(x.ccnt[c0][e]);
#pragma unroll 1
      for (int c = c0; c < nch && seen < r0 + rows; ++c) {
        // one chunk at a time: its 8 groups of 32 slots loaded and balloted independently, then ranked in order
        int ex[kChunkSlots / 32];
        unsigned bal[kChunkSlots / 32];
#pragma unroll
        for (int u = 0; u < kChunkSlots / 32; ++u) {
          const int sl = c * kChunkSlots + 32 * u + lane;
          ex[u] = sl < nslot ? int(x.exp[sl]) : -2;
        }
#pragma unroll
        for (int u = 0; u < kChunkSlots / 32; ++u) bal[u] = __ballot_sync(0xffffffffu, ex[u] == e);
#pragma unroll
        for (int u = 0; u < kChunkSlots / 32; ++u) {
          const int rk = seen + __popc(bal[u] & ((1u << lane) - 1u));
          if (ex[u] == e && rk >= r0 && rk < r0 + rows) s.ptok[h][rk - r0] = int16_t((c * kChunkSlots + 32 * u + lane) / kTopK);
          seen += __popc(bal[u]);
        }
      }
      hn = h + 1;
    }
    if (lane == 0) s.head_n = hn;
  } else if (warp >= 8) {  // permuted positions of this CTA's slots (one warp per slot)
    const int w = warp & 3;
    const int G = int(gridDim.x);
    int n_my = (nslot > int(blockIdx.x)) ? (nslot - 1 - int(blockIdx.x)) / G + 1 : 0;
    if (n_my > kMaxMySlots) {
      if (tid == 8 * 32) fail(p.err, kErrShape);
      n_my = 0;
    }
#pragma unroll 1
    for (int j = w; j < n_my; j += 4) {
      const int sl = int(blockIdx.x) + G * j;
      const int e = x.exp[sl];
      int pos = -1;
      if (e >= 0) {
        // stable rank: earlier chunks (prefix) + earlier slots of the same expert in this chunk
        const int c0 = (sl / kChunkSlots) * kChunkSlots;
        int before = 0;
#pragma unroll 1
        for (int c = c0; c < sl; c += 32) {
          const int idx = c + lane;
          before += __popc(__ballot_sync(0xffffffffu, idx < sl && x.exp[idx] == e));
        }
        pos = int(s.base[e]) + int(x.ccnt[sl / kChunkSlots][e]) + before;
      }
      if (lane == 0) {
        s.my_slot[j] = int16_t(sl);
        s.my_pos[j] = int16_t(pos);
        s.my_exp[j] = int16_t(e);
      }
    }
    if (tid == 8 * 32) s.n_my = n_my;
  } else if (warp >= 4) {  // the other launch parity's counters: zero this CTA's share for the next launch
    unsigned* const other = p.cnt + (1 - s.set) * kCntSet;
    for (int i = int(blockIdx.x) + int(gridDim.x) * (tid - 4 * 32); i < kCntSet; i += int(gridDim.x) * 128)
      muse_vr_ldq::st_relaxed_gpu(other + i, 0u);
  }
}

struct alignas(64) Maps {
  CUtensorMap m13, m2;                           // FC1 / FC2 weight stages
  CUtensorMap b1[kNumBuckets], b2[kNumBuckets];  // FC1 / FC2 token rows, one box height per bucket
};

__global__ void __launch_bounds__(kThreads, 1) ldw_kernel(const __grid_constant__ Maps maps, const __grid_constant__ Params p) {
  extern __shared__ __align__(1024) uint8_t smem_raw[];
  Smem& s = *reinterpret_cast<Smem*>((reinterpret_cast<uintptr_t>(smem_raw) + 1023) & ~uintptr_t(1023));
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
  if (tid == 0) {
    g_abort = 0;
    LDW_STAMP(p, kStStart);
  }
  if (warp == 3) {
    if (elect_one()) {
      for (int i = 0; i < kStages; ++i) {
        mbar_init(&s.full[i], 1);
        mbar_init(&s.empty[i], 1);
      }
      for (int i = 0; i < 2; ++i) {
        mbar_init(&s.tmem_full[i], 1);
        mbar_init(&s.tmem_empty[i], 4);
#if MUSE_VR_LDW_RELAY
        mbar_init(&s.acc_ready[i], 4);
#endif
      }
      mbar_init(&s.drain_mma, 1);
      for (int i = 0; i < 4; ++i) s.fc1_stored[i] = 0u;
      fence_barrier_init();
      prefetch_tensormap(&maps.m13);
      prefetch_tensormap(&maps.m2);
    }
    __syncwarp();
  } else if (warp == 1) {
    tmem_alloc(&s.tmem_base, kTmemCols);
    tmem_relinquish();
    tc_fence_before();
  }
  named_bar_sync(6, kThreads);  // barrier initialization, tensor memory
  prologue(s, p);
  named_bar_sync(3, kThreads);  // work list, pass tables, positions, counter parity for every role
  if (tid == 0) {
    LDW_STAMP(p, kStPro);
    unsigned smid;
    asm volatile("mov.u32 %0, %%smid;" : "=r"(smid));
    int cost = 0;
    for (int i = 0; i < s.n_items; ++i) cost += bucket_rows(bucket_of(s.q_rows[item_unpack(s.items[i]).q]));
    LDW_STAMPV(p, kStItems, (uint64_t(s.n_items) << 16) | uint64_t(s.n_fc1));
    LDW_STAMPV(p, kStCost, uint64_t(cost));
    LDW_STAMPV(p, kStSm, uint64_t(smid));
    (void)cost;
  }
  tc_fence_after();
  if (warp == 0) {
    role_producer(s, p, &maps.m13, &maps.m2, maps.b1, maps.b2);
  } else if (warp == 1) {
    role_mma(s, p);
  } else if (warp == 2) {
    // idle: no scale copies in the BF16 kernel
  } else if (warp == 3) {
    role_ctrl(s, p);
  } else if (warp < 8) {
    role_epilogue(s, p);
  } else {
    copy_rows(s, p, threadIdx.x - 4 * 32);  // warps 8-11: copy phase only
  }
  named_bar_sync(5, kThreads);
  if (tid == 0) LDW_STAMP(p, kStEnd);
  if (warp == 1) tmem_dealloc(s.tmem_base, kTmemCols);
#if MUSE_VR_LDW_TRAP
  if (tid == 0 && aborted()) __trap();  // after the drains and the tensor-memory release: the error word is already set
#endif
}

}  // inline namespace MUSE_VR_LDW_ABI
}  // namespace muse_vr_ldwb

// ---------------------------------------------------------------- host
#ifndef MUSE_VR_LDW_DEVICE_ONLY
#include <sgl_kernel/tensor.h>
#include <sgl_kernel/type.cuh>
#include <sgl_kernel/utils.cuh>
#include <sgl_kernel/utils.h>

#include <dlpack/dlpack.h>
#include <tvm/ffi/container/tensor.h>

#include <mutex>
#include <unordered_map>

namespace muse_vr_ldwb {
inline namespace MUSE_VR_LDW_ABI {
namespace {

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

// [rows, 8192] uint8 rows (BF16 rows of 4096) viewed as (128-byte K atom, row, atom index); a box of box_rows rows x 2
// atoms lands as 2 consecutive 128-byte-swizzled atoms of box_rows rows.
CUtensorMap make_rows_map(const void* base, uint64_t rows, uint32_t box_rows) {
  CUtensorMap map{};
  const cuuint64_t dim[3] = {128, rows, kRowBytes / 128};
  const cuuint64_t stride[2] = {kRowBytes, 128};
  const cuuint32_t box[3] = {128, box_rows, 2};
  const cuuint32_t es[3] = {1, 1, 1};
  const CUresult rc = get_encode_fn()(&map, CU_TENSOR_MAP_DATA_TYPE_UINT8, 3, const_cast<void*>(base), dim, stride, box,
                                      es, CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B,
                                      CU_TENSOR_MAP_L2_PROMOTION_L2_256B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
  sglang::host::RuntimeCheck(rc == CUDA_SUCCESS, "cuTensorMapEncodeTiled failed");
  return map;
}

// Block-major-K expert stack [P][K / 64][rows][64] (BF16; 128-byte K blocks, the stock BF16 expert layout) viewed as
// (128-byte block, row, K block, expert); a box of 128 rows x 2 K blocks lands as 2 consecutive 128-byte-swizzled atoms.
CUtensorMap make_weight_map(const void* base, uint64_t rows, uint64_t pool) {
  CUtensorMap map{};
  const cuuint64_t dim[4] = {128, rows, kRowBytes / 128, pool};
  const cuuint64_t stride[3] = {128, rows * 128, rows * kRowBytes};
  const cuuint32_t box[4] = {128, uint32_t(kTile), 2, 1};
  const cuuint32_t es[4] = {1, 1, 1, 1};
  const CUresult rc = get_encode_fn()(&map, CU_TENSOR_MAP_DATA_TYPE_UINT8, 4, const_cast<void*>(base), dim, stride, box,
                                      es, CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B,
                                      CU_TENSOR_MAP_L2_PROMOTION_L2_256B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
  sglang::host::RuntimeCheck(rc == CUDA_SUCCESS, "cuTensorMapEncodeTiled failed (weights)");
  return map;
}

template <class F>
const CUtensorMap& cached(uint64_t key, F make) {
  static std::mutex mu;
  static std::unordered_map<uint64_t, CUtensorMap> cache;
  std::lock_guard<std::mutex> lock(mu);
  auto it = cache.find(key);
  if (it == cache.end()) it = cache.emplace(key, make()).first;
  return it->second;
}

}  // namespace

struct MuseVrLdwb {
  static constexpr std::size_t kSmem = sizeof(Smem) + 1024;

  static void preload() {
    using namespace sglang::host;
    int dev = 0, optin = 0;
    RuntimeDeviceCheck(cudaGetDevice(&dev));
    RuntimeDeviceCheck(cudaDeviceGetAttribute(&optin, cudaDevAttrMaxSharedMemoryPerBlockOptin, dev));
    RuntimeCheck(kSmem <= std::size_t(optin), "shared memory per block above the device's opt-in limit");
    RuntimeDeviceCheck(cudaFuncSetAttribute(ldw_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, int(kSmem)));
    cudaFuncAttributes attr;
    RuntimeDeviceCheck(cudaFuncGetAttributes(&attr, ldw_kernel));
  }

  static void run(const tvm::ffi::TensorView packed, const tvm::ffi::TensorView hs, const tvm::ffi::TensorView w13,
                  const tvm::ffi::TensorView w2, const tvm::ffi::TensorView sm_domain, const tvm::ffi::TensorView out,
                  const tvm::ffi::TensorView expert_weights, const tvm::ffi::TensorView e2p,
                  const tvm::ffi::TensorView hs_perm, const tvm::ffi::TensorView act, const tvm::ffi::TensorView cnt,
                  const tvm::ffi::TensorView ctl, const tvm::ffi::TensorView err,
                  const tvm::ffi::Optional<tvm::ffi::TensorView> stamps, int64_t local_offset, int64_t n_dom0,
                  int64_t n_dom1, int64_t flags, int64_t dbg, int64_t pdl
#if MUSE_VR_LDW_RELAY
                  , const tvm::ffi::TensorView relay
#endif
                  ) {
    using namespace sglang;
    using namespace sglang::host;
    auto M = SymbolicSize{"num_tokens"};
    auto P = SymbolicSize{"pool"};
    auto NSM = SymbolicSize{"sm_count"};
    auto device_ = SymbolicDevice{};
    device_.set_options<kDLCUDA>();
    TensorMatcher({M, kTopK}).with_dtype<int32_t>().with_device(device_).verify(packed);
    TensorMatcher({M, kHidden}).with_dtype<bf16_t>().with_device(device_).verify(hs);
    // the stock BF16 expert stacks: block-major K [P][K / 64][rows][64]
    TensorMatcher({P, int64_t(kRowBytes / 128), 2 * kInter, 64}).with_dtype<bf16_t>().with_device(device_).verify(w13);
    TensorMatcher({P, int64_t(kRowBytes / 128), kHidden, 64}).with_dtype<bf16_t>().with_device(device_).verify(w2);
    TensorMatcher({NSM}).with_dtype<int8_t>().with_device(device_).verify(sm_domain);
    auto MR = SymbolicSize{"rows"};
    TensorMatcher({MR, kHidden}).with_dtype<bf16_t>().with_device(device_).verify(out);
    TensorMatcher({M, kTopK}).with_dtype<int16_t>().with_device(device_).verify(expert_weights);
    TensorMatcher({MR}).with_dtype<int32_t>().with_device(device_).verify(e2p);
    TensorMatcher({int64_t(kMaxSlots), int64_t(kRowBytes)}).with_dtype<uint8_t>().with_device(device_).verify(hs_perm);
    TensorMatcher({int64_t(kMaxSlots) * kActWords}).with_dtype<int64_t>().with_device(device_).verify(act);
    TensorMatcher({int64_t(2 * kCntSet)}).with_dtype<int32_t>().with_device(device_).verify(cnt);
#if !MUSE_VR_SLOT_UNSAFE && MUSE_VR_SLOT_EPOCH
    TensorMatcher({32 + muse_vr_slot::kEpochWords}).with_dtype<int32_t>().with_device(device_).verify(ctl);
#else
    TensorMatcher({32}).with_dtype<int32_t>().with_device(device_).verify(ctl);
#endif
    TensorMatcher({1}).with_dtype<int32_t>().with_device(device_).verify(err);
    if (stamps.has_value())
      TensorMatcher({NSM.unwrap() * int64_t(kStampWords)}).with_dtype<int64_t>().with_device(device_).verify(stamps.value());
    const int64_t m = M.unwrap(), pool = P.unwrap();
    RuntimeCheck(m >= 1 && m <= kMaxTokens, "token count outside 1..512");
    RuntimeCheck(MR.unwrap() == m * kTopK, "output rows must be num_tokens * top_k");
    RuntimeCheck(pool >= 1 && pool <= kMaxPool, "pool outside 1..128");
    RuntimeCheck(local_offset >= 0 && local_offset + pool <= 32768, "bad local expert offset");
    RuntimeCheck(n_dom0 >= 0 && n_dom1 >= 0 && n_dom0 + n_dom1 == NSM.unwrap(), "domain slots must cover the grid");
    RuntimeCheck(NSM.unwrap() <= 256 && n_dom0 <= 128 && n_dom1 <= 128, "slot table holds 128 ranks per domain");
    RuntimeCheck(m * kTopK <= int64_t(kMaxMySlots) * NSM.unwrap(), "routed slots per CTA above the copy table");

    Params p{};
    p.packed = static_cast<const int32_t*>(packed.data_ptr());
    p.hs = static_cast<const uint8_t*>(hs.data_ptr());
    p.sm_domain = static_cast<const int8_t*>(sm_domain.data_ptr());
    p.hs_perm = static_cast<uint8_t*>(hs_perm.data_ptr());
    p.act = static_cast<unsigned long long*>(act.data_ptr());
    p.cnt = static_cast<unsigned*>(cnt.data_ptr());
    p.out = static_cast<__nv_bfloat16*>(out.data_ptr());
    p.e2p = static_cast<int32_t*>(e2p.data_ptr());
    p.expert_weights = static_cast<uint16_t*>(expert_weights.data_ptr());
    p.ticket = static_cast<unsigned*>(ctl.data_ptr());
    p.err = static_cast<int32_t*>(err.data_ptr());
    p.m = static_cast<int32_t>(m);
    p.pool = static_cast<int32_t>(pool);
    p.local_offset = static_cast<int32_t>(local_offset);
    p.n_dom0 = static_cast<int32_t>(n_dom0);
    p.n_dom1 = static_cast<int32_t>(n_dom1);
    p.flags = static_cast<int32_t>(flags);
    p.dbg = static_cast<int32_t>(dbg);
    p.stamps = stamps.has_value() ? static_cast<unsigned long long*>(stamps.value().data_ptr()) : nullptr;
#if MUSE_VR_LDW_RELAY
    TensorMatcher({int64_t(kRelaySlots) * 128 * kRelayCols}).with_dtype<float>().with_device(device_).verify(relay);
    p.relay = static_cast<float*>(relay.data_ptr());
#endif

    Maps maps;
    const void* w13p = w13.data_ptr();
    const void* w2p = w2.data_ptr();
    const void* hp = hs_perm.data_ptr();
    const void* ap = act.data_ptr();
    maps.m13 = cached(reinterpret_cast<uint64_t>(w13p) * 1000003ull ^ uint64_t(pool) ^ 0x13,
                      [&] { return make_weight_map(w13p, uint64_t(2 * kInter), uint64_t(pool)); });
    maps.m2 = cached(reinterpret_cast<uint64_t>(w2p) * 1000003ull ^ uint64_t(pool) ^ 0x2,
                     [&] { return make_weight_map(w2p, uint64_t(kHidden), uint64_t(pool)); });
    for (int b = 0; b < kNumBuckets; ++b) {
      const uint32_t br = uint32_t(bucket_rows(b));
      maps.b1[b] = cached(reinterpret_cast<uint64_t>(hp) * 1000003ull ^ (uint64_t(br) << 8 | 1),
                          [&] { return make_rows_map(hp, uint64_t(kMaxSlots), br); });
      maps.b2[b] = cached(reinterpret_cast<uint64_t>(ap) * 1000003ull ^ (uint64_t(br) << 8 | 2),
                          [&] { return make_rows_map(ap, uint64_t(kMaxSlots), br); });
    }
    [[maybe_unused]] static const bool configured = [] {
      preload();
      return true;
    }();
    LaunchKernel(dim3(static_cast<uint32_t>(NSM.unwrap())), dim3(kThreads), device_.unwrap(), kSmem)
        .enable_pdl(pdl != 0)(ldw_kernel, maps, p);
  }
};

}  // inline namespace MUSE_VR_LDW_ABI
}  // namespace muse_vr_ldwb
#endif  // MUSE_VR_LDW_DEVICE_ONLY
