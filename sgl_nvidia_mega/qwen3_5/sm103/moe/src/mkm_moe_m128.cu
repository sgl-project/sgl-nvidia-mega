// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// Fused small-M NVFP4 MoE expert phase for Qwen3.5 TP8 decode verify (M <= 128 tokens, top-10, 512 experts,
// hidden 4096, intermediate 128 per rank): the top-10 of the router logits in the trtllm-gen routing's order and
// weights, FC1, SwiGLU, the NVFP4 requantization and FC2 in one launch over the trtllm-gen weight layout. Writes the
// deferred-finalize rows (row t * 10 + k, identity permutation) and the expert weights.
//
// The 64-token engine (mkm_moe_m64.cu) for up to 128 tokens, on a given routing only (route_topk's ids): each item
// carries its rows as a token list. As there, the work is (expert, row group of at most 16 of its
// tokens) units, so an expert that more than 16 tokens chose runs as several units on different CTAs. CTA c speculates
// on slot c's expert and runs its first unit when slot c is that expert's first; every other unit (experts first seen
// past the grid, further row groups) goes to the CTAs whose slot repeats an earlier expert, then round robin to
// CTAs 0, 1, ... Per unit the arithmetic and summation order are the 16-token engine's: rows bit for bit
// trtllm-gen's.
//
// Built with NVRTC: the kernel has C linkage (q35_mkm_moe_m128), and the launcher's dynamic shared memory size
// (Q35_MKM_SMEM_BYTES) is checked against the layout at compile time.
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
#include <cuda_bf16.h>

// NVRTC has no <cstdint>, <cuda.h> or <math.h>
typedef unsigned char uint8_t;
typedef unsigned short uint16_t;
typedef int int32_t;
typedef unsigned int uint32_t;
typedef unsigned long long uint64_t;
typedef unsigned long long uintptr_t;
typedef struct __align__(64) {
  uint64_t opaque[16];
} CUtensorMap;
#ifndef INFINITY
#define INFINITY __int_as_float(0x7f800000)
#endif
// Minimal SM100 (tcgen05 / TMA / mbarrier) PTX helpers for the small-M NVFP4 MoE engine.

namespace qwen35 {

__device__ __forceinline__ uint32_t smem_u32(const void* p) {
  return static_cast<uint32_t>(__cvta_generic_to_shared(p));
}

// ---------------------------------------------------------------- mbarrier
__device__ __forceinline__ void mbar_init(uint64_t* bar, uint32_t count) {
  asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" ::"r"(smem_u32(bar)), "r"(count) : "memory");
}
__device__ __forceinline__ void fence_mbar_init() {
  asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
}
__device__ __forceinline__ void mbar_expect_tx(uint64_t* bar, uint32_t bytes) {
  asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;" ::"r"(smem_u32(bar)), "r"(bytes) : "memory");
}
__device__ __forceinline__ void mbar_arrive(uint64_t* bar) {
  asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];" ::"r"(smem_u32(bar)) : "memory");
}
__device__ __forceinline__ bool mbar_try_wait(uint64_t* bar, uint32_t phase) {
  uint32_t ok;
  asm volatile(
      "{\n .reg .pred p;\n mbarrier.try_wait.parity.shared::cta.b64 p, [%1], %2;\n selp.u32 %0, 1, 0, p;\n}\n"
      : "=r"(ok)
      : "r"(smem_u32(bar)), "r"(phase)
      : "memory");
  return ok != 0;
}
__device__ __forceinline__ void mbar_wait(uint64_t* bar, uint32_t phase) {
  while (!mbar_try_wait(bar, phase)) {
  }
}

// ---------------------------------------------------------------- TMA / bulk copies
__device__ __forceinline__ void tma_load_2d(void* dst, const void* tmap, uint64_t* bar, int c0, int c1) {
  asm volatile(
      "cp.async.bulk.tensor.2d.shared::cluster.global.tile.mbarrier::complete_tx::bytes [%0], [%1, {%3, %4}], [%2];" ::
          "r"(smem_u32(dst)),
      "l"(tmap), "r"(smem_u32(bar)), "r"(c0), "r"(c1)
      : "memory");
}
__device__ __forceinline__ void bulk_load(void* dst, const void* src, uint32_t bytes, uint64_t* bar) {
  asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];" ::"r"(
                   smem_u32(dst)),
               "l"(src), "r"(bytes), "r"(smem_u32(bar))
               : "memory");
}
__device__ __forceinline__ void prefetch_tmap(const void* tmap) {
  asm volatile("prefetch.tensormap [%0];" ::"l"(tmap) : "memory");
}
// generic-proxy smem writes -> visible to the async proxy (UMMA / UTCCP operand reads)
__device__ __forceinline__ void fence_proxy_async_smem() {
  asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
}

// ---------------------------------------------------------------- tcgen05
__device__ __forceinline__ void tmem_alloc(uint32_t* dst_smem, uint32_t ncols) {
  asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], %1;" ::"r"(smem_u32(dst_smem)),
               "r"(ncols)
               : "memory");
}
__device__ __forceinline__ void tmem_relinquish() {
  asm volatile("tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned;" ::: "memory");
}
__device__ __forceinline__ void tmem_dealloc(uint32_t taddr, uint32_t ncols) {
  asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, %1;" ::"r"(taddr), "r"(ncols) : "memory");
}
__device__ __forceinline__ void tc_fence_before() {
  asm volatile("tcgen05.fence::before_thread_sync;" ::: "memory");
}
__device__ __forceinline__ void tc_fence_after() {
  asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");
}
// arrive on `bar` when all prior tcgen05 async ops issued by this thread complete
__device__ __forceinline__ void tc_commit(uint64_t* bar) {
  asm volatile("tcgen05.commit.cta_group::1.mbarrier::arrive::one.shared::cluster.b64 [%0];" ::"r"(smem_u32(bar))
               : "memory");
}
// smem -> TMEM copy of one 512-byte scale-factor chunk (32 rows x 16 B), broadcast to the 4 lane quarters
__device__ __forceinline__ void utccp_32x128b_warpx4(uint32_t taddr, uint64_t sdesc) {
  asm volatile("tcgen05.cp.cta_group::1.32x128b.warpx4 [%0], %1;" ::"r"(taddr), "l"(sdesc) : "memory");
}
// D(tmem) (+)= A(smem) * B(smem)^T with NVFP4 block-16 scales from TMEM
__device__ __forceinline__ void umma_nvf4(uint32_t tmem_d, uint64_t adesc, uint64_t bdesc, uint32_t idesc,
                                          uint32_t tmem_sfa, uint32_t tmem_sfb, uint32_t accumulate) {
  asm volatile(
      "{\n .reg .pred p;\n setp.ne.b32 p, %4, 0;\n"
      " tcgen05.mma.cta_group::1.kind::mxf4nvf4.block_scale.block16 [%0], %1, %2, %3, [%5], [%6], p;\n}\n" ::"r"(
          tmem_d),
      "l"(adesc), "l"(bdesc), "r"(idesc), "r"(accumulate), "r"(tmem_sfa), "r"(tmem_sfb)
      : "memory");
}
// each warp reads its 32-lane quarter: 32 lanes x 16 consecutive 32-bit columns
__device__ __forceinline__ void tmem_ld_32x32b_x16(uint32_t taddr, uint32_t (&r)[16]) {
  asm volatile(
      "tcgen05.ld.sync.aligned.32x32b.x16.b32 {%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15}, [%16];"
      : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]), "=r"(r[4]), "=r"(r[5]), "=r"(r[6]), "=r"(r[7]),
        "=r"(r[8]), "=r"(r[9]), "=r"(r[10]), "=r"(r[11]), "=r"(r[12]), "=r"(r[13]), "=r"(r[14]), "=r"(r[15])
      : "r"(taddr)
      : "memory");
}
// L2 prefetch of a contiguous global range (bytes multiple of 16, 16-byte aligned)
__device__ __forceinline__ void bulk_prefetch_l2(const void* gptr, uint32_t bytes) {
  asm volatile("cp.async.bulk.prefetch.L2.global [%0], %1;" ::"l"(gptr), "r"(bytes) : "memory");
}
__device__ __forceinline__ void tmem_ld_wait() { asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory"); }
// each warp writes its 32-lane quarter: 32 lanes x 16 consecutive 32-bit columns
__device__ __forceinline__ void tmem_st_32x32b_x16(uint32_t taddr, const uint32_t (&r)[16]) {
  asm volatile(
      "tcgen05.st.sync.aligned.32x32b.x16.b32 [%0], {%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16};" ::"r"(
          taddr),
      "r"(r[0]), "r"(r[1]), "r"(r[2]), "r"(r[3]), "r"(r[4]), "r"(r[5]), "r"(r[6]), "r"(r[7]), "r"(r[8]), "r"(r[9]),
      "r"(r[10]), "r"(r[11]), "r"(r[12]), "r"(r[13]), "r"(r[14]), "r"(r[15])
      : "memory");
}
__device__ __forceinline__ void tmem_st_wait() { asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory"); }

// ---------------------------------------------------------------- descriptors
// SM100 shared-memory matrix descriptor (version 1). layout: 0 none, 2 SW128, 4 SW64, 6 SW32.
__device__ __forceinline__ uint64_t make_sdesc(uint32_t saddr, uint32_t lbo, uint32_t sbo, uint32_t layout) {
  uint64_t d = 0;
  d |= (uint64_t)((saddr >> 4) & 0x3FFF);
  d |= (uint64_t)((lbo >> 4) & 0x3FFF) << 16;
  d |= (uint64_t)((sbo >> 4) & 0x3FFF) << 32;
  d |= (uint64_t)1 << 46;  // version
  d |= (uint64_t)(layout & 0x7) << 61;
  return d;
}
// Block-scaled instruction descriptor: E2M1 x E2M1, UE4M3 scales, both K-major, F32 accumulate.
__host__ __device__ constexpr uint32_t make_idesc_nvf4(uint32_t M, uint32_t N) {
  return (1u << 7)                 // a_format E2M1
         | (1u << 10)              // b_format E2M1
         | ((N >> 3) << 17)        // n_dim
         | (0u << 23)              // scale format UE4M3
         | ((M >> 4) << 24);       // m_dim
}

}  // namespace qwen35

namespace sglang {

namespace qwen_mkm_moe_m128 {

using namespace qwen35;


constexpr int kH = 4096, kE = 512, kTopK = 10, kMaxTok = 128;
constexpr int kGroups = kMaxTok / 32;  // 32-token groups: lane l holds tokens l + 32 h
constexpr int kRows = 16;     // A rows (tokens) per item
// per CTA: at most 1 + ceil(units / CTAs) items; units <= 560 at 128 tokens (each expert's ceil(tokens / 16)) over
// >= 107 CTAs (the launcher's bound)
constexpr int kMaxItems = 8;
constexpr int kI = 128;             // intermediates per expert per rank (TP8)
constexpr int kN1 = 2 * kI;         // W13 rows per expert (gate + up)
constexpr int kStageBytes = 36864;  // 32 KB weights + 4 KB scale factors
constexpr int kStages = 4;
constexpr int kW13BoxRows = 128;    // rows of one W13 TMA box; the host tensor map must use the same
constexpr int kFc1Stages = 16;      // the expert's 256 W13 rows, 4 k-blocks each (full K = 4096)
constexpr int kFc2Stages = 8;       // two tiles of 256 W2 rows x 64 B each
constexpr int kTotStages = kFc1Stages + kFc2Stages;
constexpr int kThreads = 256;
// TMEM columns. FC1: accumulator [0,256), x SFA ring [256,320) (16 slots x 4), SFB ring [384,512) (16 slots x 8).
// FC2: accumulator [0,256), h SFA [256,320) (16 atoms x 4), SFB ring [384,512) (16 slots x 8).
constexpr uint32_t kColSfaX = 256, kColSfa = 256, kColSfb = 384;
constexpr int kSfbSlots = 16, kSfaSlots = 16;
constexpr int kARow = 260;  // FC1 accumulator row stride in floats: 16 B aligned, rows on distinct banks
#ifndef L2_EVICT_FIRST  // stream expert weights with an L2 evict-first policy (they are read once per layer)
#define L2_EVICT_FIRST 1
#endif
#ifndef TOPK_RANK
#define TOPK_RANK 0
#endif
#ifndef SF_UTCCP_FC1
#define SF_UTCCP_FC1 0
#endif
#ifndef SPEC_STAGES  // ring stages issued for the speculated expert before the routing verdict (rest: after it)
#define SPEC_STAGES 4
#endif
#ifndef STAGE_STAMPS  // DIAG: per-stage issue (slots 64 + st) and landing (16 + st, seen by warp 4) stamps
#define STAGE_STAMPS 0
#endif
#ifndef DIAG_SPEC  // DIAG: warp 1's speculative ranking, stamps 16 = logits loaded, 17 = keys, 18 = top-10 (stride 32)
#define DIAG_SPEC 0
#endif
#ifndef DIAG_ROUTE  // DIAG: stamps 13 = warp 0's routing rounds done, 15 = their barrier, 14 = item 0's rows gathered
#define DIAG_ROUTE 0
#endif
constexpr int kTraceStride = STAGE_STAMPS ? 128 : (DIAG_SPEC || DIAG_ROUTE) ? 32 : 16;
#ifndef SMEM_SPACE  // 1: shared-memory accesses through Smem compile to shared-space instructions
#define SMEM_SPACE 1
#endif
#ifndef PF_SET  // L2 prefetch of the stages that wait for a ring slot at: 1 = start, 4 = FC2, 8 = FC2 pass 1
// (none: from 56 rows every CTA streams and HBM is the bound, so these prefetches only compete with the ring's loads)
#define PF_SET 0
#endif
#ifndef PF_N1  // stages prefetched at the start trigger
#define PF_N1 4
#endif
#ifndef PF_N2  // stages prefetched at the FC2 trigger
#define PF_N2 4
#endif
#ifndef TAILV  // FC2 drains: bit 0 = last pass on all 8 warps; bit 1 = TMEM loads double-buffered; bit 2 = coalesced stores
#define TAILV 3
#endif
#define TAIL8 (TAILV & 1)
#define DRAIN_PIPE ((TAILV >> 1) & 1)
#define DRAIN_SMEM ((TAILV >> 2) & 1)
static_assert(DRAIN_SMEM == 0, "x's shared memory holds the next item's rows while the drains run");
#ifndef EARLY_SF2  // FC2 stages whose scale factors are staged before the h epilogue (old order: 2)
#define EARLY_SF2 0
#endif
#ifndef GATHER_BATCH  // x row granules each gathering thread loads before storing them
#define GATHER_BATCH 8
#endif
#ifndef GATHER_ASYNC  // 1: the gather copies by cp.async (every granule in flight at once, no register staging)
#define GATHER_ASYNC 1
#endif
#ifndef SPEC_PF  // stages of the speculated expert past the first ring fill prefetched to L2 while the routing runs
#define SPEC_PF 4
#endif
#ifndef FC2_PF  // FC2 tiles (beyond the first kStages) prefetched to L2 ahead of the ring; 0 = off
#define FC2_PF 0
#endif
#ifndef FC2_PF_LEAD  // issue the FC2 prefetch when the producer reaches FC1 stage kFc1Stages - FC2_PF_LEAD
#define FC2_PF_LEAD 4
#endif

struct Params {
  const uint16_t* logits;    // [M, 512] bf16 router logits (router_gemm_quant output)
  uint16_t* expert_weights;  // [M, 10] bf16 out: DynBlock softmax-over-top-10 weights (the last CTA writes)
  int32_t* topk_ids;         // [M, 10] out (the last CTA writes)
  const uint8_t* x_sf;       // [M, 256] linear e4m3
  const uint8_t* w13_sf;     // [E, 2 row blocks x 64 k blocks x 512 B] (128x4 interleaved)
  const uint8_t* w2_sf;      // [E, 32 row blocks x 2 k blocks x 512 B]
  const float* g1_alpha;     // [E] gate dequant
  const float* g1_scale_c;   // [E] up dequant * FC2 input quant scale
  const float* g2_alpha;     // [E]
  __nv_bfloat16* out;        // [M * 10, 4096]
  long long* trace;          // optional [grid, 16] %globaltimer stamps
  int num_tokens;
  const uint8_t* x_fp4;      // [M, 2048] NVFP4 activations (each item gathers its rows)
  // [M, 10] the routing's expert ids (route_topk's, bit for bit the trtllm-gen routing): when set, the engine takes them
  // in place of ranking the logits (and writes neither expert_weights nor topk_ids, which route_topk wrote)
  const int32_t* ids_in;
};

struct __align__(1024) Smem {
  uint8_t ring[kStages][kStageBytes];
  alignas(1024) uint8_t xa[32 * 1024];  // x: per 128-B K chunk c, 2 KB = SW128 atoms of item rows 0..7 | 8..15
  alignas(1024) uint8_t ha[16 * 1024];  // h: 8 copies of (rows 0..7 | rows 8..15) x 128 B (K = 128: 64 B), SW128
  float acc1[16 * kARow];               // FC1 accumulator rows 0..15 (item rows) x 256 columns
  uint8_t xsf[16][256];                 // x scale bytes of the item's rows, full K
  uint32_t hsf[16][2];                  // h scale words: row r, k-block kb (4 e4m3 bytes)
  uint64_t full[kStages], empty[kStages], sf_full[kStages];
  uint64_t x_full, fc1_done, acc_ready, h_ready, fc2_pass[2], fc2_free, acc_drained, route_done;
  uint32_t tmem;
  int nitems, spec_ok;
  int item_e[kMaxItems];
  uint8_t item_rows[kMaxItems][kRows];  // each item's tokens, ascending
  int item_n[kMaxItems];
  int nrows[2], rtok[2][kRows], rkslot[2][kRows];  // rows of the current / next item (by item parity)
  int first[kE];                        // first slot of each expert (atomicMin)
  int cnt[kE];                          // tokens of each expert (atomicAdd)
  uint32_t fbits[kMaxTok];              // token t: its top-k slots that are their expert's first (bit k)
  int ounits[kMaxTok];                  // token t: the units its first-occurrence slots hand out (not a CTA's own)
  uint8_t uslot[kMaxTok][kTopK];        // per slot: those units (0 unless a first occurrence)
  int32_t ids[kMaxTok][kTopK];
};

__device__ __forceinline__ uint64_t gtime() {
  uint64_t t;
  asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t)::"memory");
  return t;
}
__device__ __forceinline__ uint32_t cluster_rank() {
  uint32_t r;
  asm volatile("mov.u32 %0, %%cluster_ctarank;" : "=r"(r));
  return r;
}
__device__ __forceinline__ void cluster_arrive() {
  asm volatile("barrier.cluster.arrive.release.aligned;" ::: "memory");
}
// no release: the phase only orders DSMEM traffic that mbarriers already completed, so pending global stores
// (stamps, routing outputs, output rows) need not drain before the arrive
__device__ __forceinline__ void cluster_arrive_relaxed() {
  asm volatile("barrier.cluster.arrive.relaxed.aligned;" ::: "memory");
}
__device__ __forceinline__ void cluster_wait() {
  asm volatile("barrier.cluster.wait.acquire.aligned;" ::: "memory");
}
__device__ __forceinline__ uint32_t mapa(uint32_t saddr, uint32_t rank) {
  uint32_t r;
  asm volatile("mapa.shared::cluster.u32 %0, %1, %2;" : "=r"(r) : "r"(saddr), "r"(rank));
  return r;
}
// smem (this CTA) -> smem (peer CTA in the cluster), completes tx bytes on the peer's mbarrier
__device__ __forceinline__ void bulk_s2s(uint32_t dst_cluster, uint32_t src, uint32_t bytes, uint32_t bar_cluster) {
  asm volatile("cp.async.bulk.shared::cluster.shared::cta.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];" ::"r"(
                   dst_cluster),
               "r"(src), "r"(bytes), "r"(bar_cluster)
               : "memory");
}
// registers -> the peer CTA's shared memory; completes the bytes on the peer's mbarrier
__device__ __forceinline__ void st_async_v4(uint32_t dst, const uint4& v, uint32_t bar) {
  asm volatile("st.async.shared::cluster.mbarrier::complete_tx::bytes.v4.b32 [%0], {%1, %2, %3, %4}, [%5];" ::"r"(dst),
               "r"(v.x), "r"(v.y), "r"(v.z), "r"(v.w), "r"(bar)
               : "memory");
}
__device__ __forceinline__ void st_async_b32(uint32_t dst, uint32_t v, uint32_t bar) {
  asm volatile("st.async.shared::cluster.mbarrier::complete_tx::bytes.b32 [%0], %1, [%2];" ::"r"(dst), "r"(v), "r"(bar)
               : "memory");
}
__device__ __forceinline__ void tmem_st_x4(uint32_t taddr, const uint4& v) {
  asm volatile("tcgen05.st.sync.aligned.32x32b.x4.b32 [%0], {%1, %2, %3, %4};" ::"r"(taddr), "r"(v.x), "r"(v.y),
               "r"(v.z), "r"(v.w)
               : "memory");
}
// bar.sync counts arrivals per warp execution: reconverge first, so a warp whose lane 0 just ran a stamp or an
// mbarrier arrive cannot arrive twice and release the barrier early
__device__ __forceinline__ void named_bar(int id, int n) {
  __syncwarp();
  asm volatile("bar.sync %0, %1;" ::"r"(id), "r"(n) : "memory");
}
__device__ __forceinline__ void tmem_ld_x32(uint32_t taddr, uint32_t (&r)[32]) {
  asm volatile(
      "tcgen05.ld.sync.aligned.32x32b.x32.b32 {%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17,%18,%19,"
      "%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31}, [%32];"
      : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]), "=r"(r[4]), "=r"(r[5]), "=r"(r[6]), "=r"(r[7]), "=r"(r[8]),
        "=r"(r[9]), "=r"(r[10]), "=r"(r[11]), "=r"(r[12]), "=r"(r[13]), "=r"(r[14]), "=r"(r[15]), "=r"(r[16]),
        "=r"(r[17]), "=r"(r[18]), "=r"(r[19]), "=r"(r[20]), "=r"(r[21]), "=r"(r[22]), "=r"(r[23]), "=r"(r[24]),
        "=r"(r[25]), "=r"(r[26]), "=r"(r[27]), "=r"(r[28]), "=r"(r[29]), "=r"(r[30]), "=r"(r[31])
      : "r"(taddr)
      : "memory");
}
__device__ __forceinline__ void tmem_ld_x64(uint32_t taddr, uint32_t (&r)[64]) {
  asm volatile(
      "tcgen05.ld.sync.aligned.32x32b.x64.b32 {%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17,%18,%19,"
      "%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31,%32,%33,%34,%35,%36,%37,%38,%39,%40,%41,%42,%43,%44,%45,%46,%47,"
      "%48,%49,%50,%51,%52,%53,%54,%55,%56,%57,%58,%59,%60,%61,%62,%63}, [%64];"
      : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]), "=r"(r[4]), "=r"(r[5]), "=r"(r[6]), "=r"(r[7]), "=r"(r[8]),
        "=r"(r[9]), "=r"(r[10]), "=r"(r[11]), "=r"(r[12]), "=r"(r[13]), "=r"(r[14]), "=r"(r[15]), "=r"(r[16]),
        "=r"(r[17]), "=r"(r[18]), "=r"(r[19]), "=r"(r[20]), "=r"(r[21]), "=r"(r[22]), "=r"(r[23]), "=r"(r[24]),
        "=r"(r[25]), "=r"(r[26]), "=r"(r[27]), "=r"(r[28]), "=r"(r[29]), "=r"(r[30]), "=r"(r[31]), "=r"(r[32]),
        "=r"(r[33]), "=r"(r[34]), "=r"(r[35]), "=r"(r[36]), "=r"(r[37]), "=r"(r[38]), "=r"(r[39]), "=r"(r[40]),
        "=r"(r[41]), "=r"(r[42]), "=r"(r[43]), "=r"(r[44]), "=r"(r[45]), "=r"(r[46]), "=r"(r[47]), "=r"(r[48]),
        "=r"(r[49]), "=r"(r[50]), "=r"(r[51]), "=r"(r[52]), "=r"(r[53]), "=r"(r[54]), "=r"(r[55]), "=r"(r[56]),
        "=r"(r[57]), "=r"(r[58]), "=r"(r[59]), "=r"(r[60]), "=r"(r[61]), "=r"(r[62]), "=r"(r[63])
      : "r"(taddr)
      : "memory");
}
__device__ __forceinline__ void tmem_st_x8(uint32_t taddr, const uint32_t (&r)[8]) {
  asm volatile("tcgen05.st.sync.aligned.32x32b.x8.b32 [%0], {%1,%2,%3,%4,%5,%6,%7,%8};" ::"r"(taddr), "r"(r[0]),
               "r"(r[1]), "r"(r[2]), "r"(r[3]), "r"(r[4]), "r"(r[5]), "r"(r[6]), "r"(r[7])
               : "memory");
}
__device__ __forceinline__ void tmem_st_x16(uint32_t taddr, const uint32_t (&r)[16]) {
  asm volatile("tcgen05.st.sync.aligned.32x32b.x16.b32 [%0], {%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16};" ::
                   "r"(taddr),
               "r"(r[0]), "r"(r[1]), "r"(r[2]), "r"(r[3]), "r"(r[4]), "r"(r[5]), "r"(r[6]), "r"(r[7]), "r"(r[8]), "r"(r[9]),
               "r"(r[10]), "r"(r[11]), "r"(r[12]), "r"(r[13]), "r"(r[14]), "r"(r[15])
               : "memory");
}
__device__ __forceinline__ void tmem_st_x32(uint32_t taddr, const uint32_t (&r)[32]) {
  asm volatile(
      "tcgen05.st.sync.aligned.32x32b.x32.b32 [%0], {%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17,%18,"
      "%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31,%32};" ::"r"(taddr),
      "r"(r[0]), "r"(r[1]), "r"(r[2]), "r"(r[3]), "r"(r[4]), "r"(r[5]), "r"(r[6]), "r"(r[7]), "r"(r[8]), "r"(r[9]),
      "r"(r[10]), "r"(r[11]), "r"(r[12]), "r"(r[13]), "r"(r[14]), "r"(r[15]), "r"(r[16]), "r"(r[17]), "r"(r[18]),
      "r"(r[19]), "r"(r[20]), "r"(r[21]), "r"(r[22]), "r"(r[23]), "r"(r[24]), "r"(r[25]), "r"(r[26]), "r"(r[27]),
      "r"(r[28]), "r"(r[29]), "r"(r[30]), "r"(r[31])
      : "memory");
}
__device__ __forceinline__ float ex2_approx(float x) {
  float y;
  asm("ex2.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x));
  return y;
}
__device__ __forceinline__ float mul_ftz(float a, float b) {
  float y;
  asm("mul.ftz.f32 %0, %1, %2;" : "=f"(y) : "f"(a), "f"(b));
  return y;
}
__device__ __forceinline__ float rcp_approx(float x) {
  float y;
  asm("rcp.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x));
  return y;
}
__device__ __forceinline__ uint32_t cvt_e4m3(float a) {
  uint32_t out;
  asm("{\n .reg .b16 pair;\n .reg .f32 zero;\n mov.f32 zero, 0f00000000;\n"
      " cvt.rn.satfinite.e4m3x2.f32 pair, zero, %1;\n cvt.u32.u16 %0, pair;\n}"
      : "=r"(out)
      : "f"(a));
  return out & 0xFFu;
}
__device__ __forceinline__ float e4m3_to_f32(uint32_t v) {
  float out;
  asm("{\n .reg .b16 pair;\n .reg .b32 h2;\n .reg .b16 lo, hi;\n cvt.u16.u32 pair, %1;\n"
      " cvt.rn.f16x2.e4m3x2 h2, pair;\n mov.b32 {lo, hi}, h2;\n cvt.f32.f16 %0, lo;\n}"
      : "=f"(out)
      : "r"(v));
  return out;
}
__device__ __forceinline__ uint32_t cvt_e2m1x8(const float* v) {
  uint32_t out;
  asm("{\n .reg .b8 b0, b1, b2, b3;\n"
      " cvt.rn.satfinite.e2m1x2.f32 b0, %2, %1;\n cvt.rn.satfinite.e2m1x2.f32 b1, %4, %3;\n"
      " cvt.rn.satfinite.e2m1x2.f32 b2, %6, %5;\n cvt.rn.satfinite.e2m1x2.f32 b3, %8, %7;\n"
      " mov.b32 %0, {b0, b1, b2, b3};\n}"
      : "=r"(out)
      : "f"(v[0]), "f"(v[1]), "f"(v[2]), "f"(v[3]), "f"(v[4]), "f"(v[5]), "f"(v[6]), "f"(v[7]));
  return out;
}
// ---- router top-k, bit-exact with FlashInfer's single-block routing (routingCustom, RenormalizeNaive on bf16):
// key = bf16 order (cub float twiddle) above 65535 - expert, i.e. the higher score first, the smaller expert on ties.
__device__ __forceinline__ uint32_t topk_key(uint32_t bits, uint32_t expert) {
  const uint32_t twiddled = bits ^ ((bits & 0x8000u) ? 0xFFFFu : 0x8000u);
  return (twiddled << 16) | (65535u - expert);
}
__device__ __forceinline__ void topk_unkey(uint32_t key, uint32_t& bits, uint32_t& expert) {
  expert = 65535u - (key & 0xFFFFu);
  const uint32_t twiddled = key >> 16;
  bits = twiddled ^ ((twiddled & 0x8000u) ? 0x8000u : 0xFFFFu);
}
__device__ __forceinline__ uint32_t max16(const uint32_t (&k)[16]) {
  const uint32_t a = max(max(max(k[0], k[1]), max(k[2], k[3])), max(max(k[4], k[5]), max(k[6], k[7])));
  const uint32_t b = max(max(max(k[8], k[9]), max(k[10], k[11])), max(max(k[12], k[13]), max(k[14], k[15])));
  return max(a, b);
}
__device__ __forceinline__ uint32_t warp_max_u32(uint32_t v) {
  uint32_t r;
  asm volatile("redux.sync.max.u32 %0, %1, 0xffffffff;" : "=r"(r) : "r"(v));
  return r;
}
__device__ __forceinline__ uint32_t warp_sum_u32(uint32_t v) {
  uint32_t r;
  asm volatile("redux.sync.add.u32 %0, %1, 0xffffffff;" : "=r"(r) : "r"(v));
  return r;
}
__device__ __forceinline__ uint32_t warp_topk_of_lanes(uint32_t v, uint32_t lane) {
  uint32_t mine = 0;
#pragma unroll
  for (uint32_t k = 0; k < kTopK; ++k) {
    const uint32_t top = warp_max_u32(v);
    v = v == top ? 0u : v;
    mine = lane == k ? top : mine;
  }
  return mine;
}
__device__ __forceinline__ uint32_t count_greater(const uint32_t* s_cand, uint32_t v) {
  uint32_t r = 0;
#pragma unroll
  for (int j = 0; j < 32; j += 4) {
    const uint4 q = *reinterpret_cast<const uint4*>(s_cand + j);
    r += (q.x > v) + (q.y > v) + (q.z > v) + (q.w > v);
  }
  return r;
}
// Top-10 keys of 512 (16 per lane), lane k < 10 holding the k-th largest (keys unique and nonzero). Ranks come from
// all-pairs compares over shared memory rather than chains of warp reductions.
__device__ __forceinline__ uint32_t warp_top10(uint32_t (&keys)[16], uint32_t lane, uint32_t* s_cand) {
  // bound = the 10th largest lane maximum; every top-10 key is >= bound
  const uint32_t m = max16(keys);
  s_cand[lane] = m;
  __syncwarp();
  const uint32_t r = count_greater(s_cand, m);
  const uint32_t bound = __shfl_sync(0xffffffffu, m, __ffs(__ballot_sync(0xffffffffu, r == kTopK - 1)) - 1);
  uint32_t count = 0;
#pragma unroll
  for (uint32_t i = 0; i < 16; ++i) count += keys[i] >= bound;
  // exclusive prefix of count over the lanes, from the ballots of its bits (count <= 16)
  const uint32_t lt = (1u << lane) - 1u;
  uint32_t offset = 0, total = 0;
#pragma unroll
  for (int b = 0; b < 5; ++b) {
    const uint32_t bb = __ballot_sync(0xffffffffu, (count >> b) & 1u);
    offset += (uint32_t)__popc(bb & lt) << b;
    total += (uint32_t)__popc(bb) << b;
  }
  if (total <= 32) {
    __syncwarp();  // every lane has read the lane maxima
#pragma unroll
    for (uint32_t i = 0; i < 16; ++i) {
      if (keys[i] >= bound) s_cand[offset++] = keys[i];
    }
    if (lane >= total) s_cand[lane] = 0u;
    __syncwarp();
    const uint32_t cand = s_cand[lane];
    const uint32_t rk = count_greater(s_cand, cand);
    __syncwarp();
    if (lane < total && rk < kTopK) s_cand[rk] = cand;
    __syncwarp();
    return lane < kTopK ? s_cand[lane] : 0u;
  }
  uint32_t mine = 0;
#pragma unroll 1
  for (uint32_t k = 0; k < kTopK; ++k) {
    const uint32_t top = warp_max_u32(max16(keys));
#pragma unroll
    for (uint32_t i = 0; i < 16; ++i) keys[i] = keys[i] == top ? 0u : keys[i];
    mine = lane == k ? top : mine;
  }
  return mine;
}
// Top-10 keys of 512 (16 per lane), lane k holding the k-th largest (keys unique and nonzero).
__device__ __forceinline__ uint32_t warp_top10_redux(uint32_t (&keys)[16], uint32_t lane, uint32_t* s_cand) {
  uint32_t v = max16(keys);
  uint32_t bound = 0;
#pragma unroll
  for (uint32_t k = 0; k < kTopK; ++k) {
    bound = warp_max_u32(v);
    v = v == bound ? 0u : v;
  }
  uint32_t count = 0;
#pragma unroll
  for (uint32_t i = 0; i < 16; ++i) count += keys[i] >= bound;
  const uint32_t total = warp_sum_u32(count);
  if (total <= 32) {
    uint32_t offset = count;
#pragma unroll
    for (uint32_t d = 1; d < 32; d <<= 1) {
      const uint32_t up = __shfl_up_sync(0xffffffffu, offset, d);
      offset += lane >= d ? up : 0u;
    }
    offset -= count;
#pragma unroll
    for (uint32_t i = 0; i < 16; ++i) {
      if (keys[i] >= bound) s_cand[offset++] = keys[i];
    }
    __syncwarp();
    const uint32_t cand = lane < total ? s_cand[lane] : 0u;
    __syncwarp();
    return warp_topk_of_lanes(cand, lane);
  }
  uint32_t mine = 0;
#pragma unroll 1
  for (uint32_t k = 0; k < kTopK; ++k) {
    const uint32_t top = warp_max_u32(max16(keys));
#pragma unroll
    for (uint32_t i = 0; i < 16; ++i) keys[i] = keys[i] == top ? 0u : keys[i];
    mine = lane == k ? top : mine;
  }
  return mine;
}
// The top-10 keys of one logits row (16 bf16 a lane in v0, v1), lane k < 10 holding the k-th largest
__device__ __forceinline__ uint32_t row_top10(const uint4 v0, const uint4 v1, uint32_t lane, uint32_t* s_cand) {
  uint32_t keys[16];
#pragma unroll
  for (int i = 0; i < 8; ++i) {
    const uint32_t w = i < 4 ? (&v0.x)[i] : (&v1.x)[i - 4];
    keys[2 * i] = topk_key(w & 0xFFFFu, lane * 16 + 2 * i);
    keys[2 * i + 1] = topk_key(w >> 16, lane * 16 + 2 * i + 1);
  }
  return TOPK_RANK ? warp_top10(keys, lane, s_cand) : warp_top10_redux(keys, lane, s_cand);
}
// FlashInfer calcSoftmax (bf16 scores) as built with -use_fast_math: expf -> ex2.approx.ftz of x * log2(e),
// division -> div.full.ftz; reductions are cooperative_groups::reduce like the reference.
__device__ __forceinline__ float fast_expf(float x) {
  float y;
  asm("{\n .reg .f32 t;\n mul.ftz.f32 t, %1, 0f3FB8AA3B;\n ex2.approx.ftz.f32 %0, t;\n}" : "=f"(y) : "f"(x));
  return y;
}
__device__ __forceinline__ float fast_div(float a, float b) {
  float y;
  asm("div.full.ftz.f32 %0, %1, %2;" : "=f"(y) : "f"(a), "f"(b));
  return y;
}
__device__ __forceinline__ uint16_t softmax_topk(uint16_t score_bits, uint32_t lane) {
  namespace cg = cooperative_groups;
  const auto warp = cg::tiled_partition<32>(cg::this_thread_block());
  const float si = __uint_as_float((uint32_t)score_bits << 16);
  float max_score = -INFINITY;
  if (lane < kTopK) max_score = si >= max_score ? si : max_score;
  max_score = cg::reduce(warp, max_score, cg::greater<float>());
  float sum_score = 0.f, new_score = 0.f;
  if (lane < kTopK) {
    new_score = fast_expf(__fsub_rn(si, max_score));
    sum_score = __fadd_rn(sum_score, new_score);
  }
  sum_score = cg::reduce(warp, sum_score, cg::plus<float>());
  uint16_t out = score_bits;
  if (lane < kTopK) {
    const __nv_bfloat16 b = __float2bfloat16_rn(fast_div(new_score, sum_score));
    out = *reinterpret_cast<const uint16_t*>(&b);
  }
  return out;
}

__device__ __forceinline__ uint64_t policy_evict_first() {
  uint64_t pol;
  asm volatile("createpolicy.fractional.L2::evict_first.b64 %0, 1.0;" : "=l"(pol));
  return pol;
}
__device__ __forceinline__ void tma_load_2d_w(void* dst, const void* tmap, uint64_t* bar, int c0, int c1, uint64_t pol) {
  if (L2_EVICT_FIRST)
    asm volatile(
        "cp.async.bulk.tensor.2d.shared::cluster.global.tile.mbarrier::complete_tx::bytes.L2::cache_hint"
        " [%0], [%1, {%3, %4}], [%2], %5;" ::"r"(smem_u32(dst)),
        "l"(tmap), "r"(smem_u32(bar)), "r"(c0), "r"(c1), "l"(pol)
        : "memory");
  else
    tma_load_2d(dst, tmap, bar, c0, c1);
}
__device__ __forceinline__ void bulk_load_w(void* dst, const void* src, uint32_t bytes, uint64_t* bar, uint64_t pol) {
  if (L2_EVICT_FIRST)
    asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes.L2::cache_hint [%0], [%1], %2, [%3], %4;" ::"r"(
                     smem_u32(dst)),
                 "l"(src), "r"(bytes), "r"(smem_u32(bar)), "l"(pol)
                 : "memory");
  else
    bulk_load(dst, src, bytes, bar);
}

__device__ __forceinline__ void stamp(const Params& p, int slot) {
  if (p.trace) p.trace[blockIdx.x * kTraceStride + slot] = (long long)gtime();
}
// stamp after `dep` is available (the branch needs its value; keys are never 0)
__device__ __forceinline__ void stamp_dep(const Params& p, int slot, uint32_t dep) {
  if (dep == 0u) __trap();
  stamp(p, slot);
}
__device__ __forceinline__ void tma_prefetch_2d(const void* tmap, int c0, int c1) {
  asm volatile("cp.async.bulk.prefetch.tensor.2d.L2.global.tile [%0, {%1, %2}];" ::"l"(tmap), "r"(c0), "r"(c1) : "memory");
}
// Physical W13 column (accumulator column) holding reordered row q: block-32 shuffle of FlashInfer's
// get_shuffle_matrix_a_row_indices (epilogue_tile_m = 128); reordered row 2j is up_j, 2j + 1 is gate_j.
__device__ __forceinline__ int w13_col(int q) { return (q & ~31) + ((q & 31) % 4) * 8 + (q & 31) / 4; }
__host__ __device__ constexpr int w13_logical(int p) {
  return ((p & ~31) + (p % 8) * 4 + (p & 31) / 8) % 2 ? ((p & ~31) + (p % 8) * 4 + (p & 31) / 8) / 2
                                                     : 128 + ((p & ~31) + (p % 8) * 4 + (p & 31) / 8) / 2;
}


// FC2 drain of one pass: TMEM lane quarter q (tiles 2 q, 2 q + 1 of the pass), 256 accumulator columns
__device__ __forceinline__ void reg_fence64(uint32_t (&r)[64]) {  // keep uses of r after the tcgen05.wait::ld
#pragma unroll
  for (int i = 0; i < 64; ++i) asm volatile("" : "+r"(r[i]));
}

// FC2 accumulator columns [64 C0, 64 C1) of this warp's lane quarter -> bf16 gemm2_out rows (row r of a tile = the
// r-th row of the item, row info buffer rb)
template <int C0, int C1>
__device__ __forceinline__ void fc2_drain16(const Params& p, Smem& s, uint32_t tm, int q, int lane, int pass, float a2,
                                            int rb) {
  const int row = q * 32 + lane, tile = pass * 8 + row / 16, r = row % 16;
  const bool valid = r < s.nrows[rb];
  __nv_bfloat16* orow = valid ? p.out + (size_t)(s.rtok[rb][r] * kTopK + s.rkslot[rb][r]) * kH + tile * 256 : nullptr;
  const uint32_t base = tm + ((uint32_t)(q * 32) << 16);
  auto put = [&](const uint32_t (&r)[64], int c) {
    if (valid) {
#pragma unroll
      for (int b32 = 0; b32 < 2; ++b32) {
        uint32_t pk[16];
#pragma unroll
        for (int l = 0; l < 32; l += 2) {
          const float v0 = __uint_as_float(r[b32 * 32 + (l % 4) * 8 + l / 4]) * a2;
          const float v1 = __uint_as_float(r[b32 * 32 + ((l + 1) % 4) * 8 + (l + 1) / 4]) * a2;
          const __nv_bfloat162 b2 = __floats2bfloat162_rn(v0, v1);
          pk[l / 2] = *reinterpret_cast<const uint32_t*>(&b2);
        }
#pragma unroll
        for (int i = 0; i < 4; ++i)
          *reinterpret_cast<uint4*>(orow + c * 64 + b32 * 32 + i * 8) =
              make_uint4(pk[4 * i], pk[4 * i + 1], pk[4 * i + 2], pk[4 * i + 3]);
      }
    }
  };
  if (DRAIN_PIPE) {  // the next chunk's TMEM load is in flight while this chunk converts and stores
    uint32_t ra[64], rb2[64];
    tmem_ld_x64(base + C0 * 64, ra);
    tmem_ld_wait();
    reg_fence64(ra);
#pragma unroll
    for (int c = C0; c < C1; c += 2) {
      if (c + 1 < C1) tmem_ld_x64(base + (uint32_t)((c + 1) * 64), rb2);
      put(ra, c);
      if (c + 1 < C1) {
        tmem_ld_wait();
        reg_fence64(rb2);
        if (c + 2 < C1) tmem_ld_x64(base + (uint32_t)((c + 2) * 64), ra);
        put(rb2, c + 1);
        if (c + 2 < C1) { tmem_ld_wait(); reg_fence64(ra); }
      }
    }
  } else {
#pragma unroll 1
    for (int c = C0; c < C1; ++c) {
      uint32_t r[64];
      tmem_ld_x64(base + (uint32_t)(c * 64), r);
      tmem_ld_wait();
      put(r, c);
    }
  }
}

extern "C" __global__ void __launch_bounds__(kThreads, 1)
q35_mkm_moe_m128(const __grid_constant__ CUtensorMap tm13, const __grid_constant__ CUtensorMap tm2, const Params p) {
  extern __shared__ __align__(1024) uint8_t raw[];
#if SMEM_SPACE  // pointer arithmetic on the __shared__ array keeps the shared address space (STS/LDS, not generic)
  Smem& s = *reinterpret_cast<Smem*>(raw + ((1024u - (smem_u32(raw) & 1023u)) & 1023u));
#else
  Smem& s = *reinterpret_cast<Smem*>((reinterpret_cast<uintptr_t>(raw) + 1023) & ~uintptr_t(1023));
#endif
  const int warp = threadIdx.x / 32, lane = threadIdx.x % 32;
  const int slot = blockIdx.x;  // the (token, top-k) slot this CTA speculates on; gridDim.x <= M * 10
  const int M = p.num_tokens;
  for (int i = threadIdx.x; i < kE; i += kThreads) {
    s.first[i] = 0x7FFFFFFF;
    s.cnt[i] = 0;
  }
  if (threadIdx.x < kMaxTok) {
    s.fbits[threadIdx.x] = 0u;
    s.ounits[threadIdx.x] = 0;
  }
  if (threadIdx.x == 0) {
    stamp(p, 0);
    for (int i = 0; i < kStages; ++i) {
      mbar_init(&s.full[i], 1);
      mbar_init(&s.empty[i], 1);
      mbar_init(&s.sf_full[i], 4);
    }
    // every barrier below completes once per item, so item i waits parity i & 1
    mbar_init(&s.x_full, 1);
    mbar_init(&s.fc1_done, 1);
    mbar_init(&s.acc_ready, 2);  // warps 0 and 4 drained their columns into acc1
    mbar_init(&s.h_ready, 4);
    mbar_init(&s.fc2_pass[0], 1);
    mbar_init(&s.fc2_pass[1], 1);
    mbar_init(&s.fc2_free, 4);
    mbar_init(&s.acc_drained, 4);  // warps 4..7 read the last FC2 pass: the next item's FC1 may overwrite columns 0..255
    mbar_init(&s.route_done, 1);
    fence_mbar_init();
  }
  if (warp == 2) { tmem_alloc(&s.tmem, 512); tmem_relinquish(); tc_fence_before(); }
  if (warp == 1 && lane == 0) { prefetch_tmap(&tm13); prefetch_tmap(&tm2); }
  __syncthreads();
  tc_fence_after();
  const uint32_t tm = s.tmem;
  const int my_t = slot / kTopK, my_k = slot % kTopK;
  // ring stage g (global over the CTA's items) of item-relative stage st of expert e
  auto issue_stage = [&](int g, int st, int e, uint64_t pol) {
    const int b = g % kStages;
    mbar_expect_tx(&s.full[b], kStageBytes);
    uint8_t* dst = s.ring[b];
    if (st < kFc1Stages) {
      // the expert's rows as two boxes of 128 rows, K bytes [128 st, 128 st + 128)
      tma_load_2d_w(dst, &tm13, &s.full[b], st * 128, e * kN1, pol);
      tma_load_2d_w(dst + 16384, &tm13, &s.full[b], st * 128, e * kN1 + 128, pol);
      // scale atoms of row blocks 0 and 1, k-blocks [4 st, 4 st + 4)
      const uint8_t* sf = p.w13_sf + (size_t)e * (kN1 * kH / 16);
      bulk_load_w(dst + 32768, sf + (size_t)(st * 4) * 512, 2048, &s.full[b], pol);
      bulk_load_w(dst + 32768 + 2048, sf + (size_t)(64 + st * 4) * 512, 2048, &s.full[b], pol);
    } else {
      // W2 tiles 2 f and 2 f + 1 (256 rows x 64 B each) and their scale atoms (2 row blocks x 2 k-blocks)
      const int f = st - kFc1Stages;
#pragma unroll
      for (int tl = 0; tl < 2; ++tl) {
        const int row0 = (2 * f + tl) * 256;
        tma_load_2d_w(dst + tl * 16384, &tm2, &s.full[b], 0, e * kH + row0, pol);
        bulk_load_w(dst + 32768 + tl * 2048, p.w2_sf + (size_t)e * (kH * 8) + (size_t)(row0 / 128) * 2 * 512, 2048,
                    &s.full[b], pol);
      }
    }
  };
  auto prefetch_stage = [&](int st, int e) {  // the same bytes as issue_stage, to L2 only
    if (st < kFc1Stages) {
      tma_prefetch_2d(&tm13, st * 128, e * kN1);
      tma_prefetch_2d(&tm13, st * 128, e * kN1 + 128);
      const uint8_t* sf = p.w13_sf + (size_t)e * (kN1 * kH / 16);
      bulk_prefetch_l2(sf + (size_t)(st * 4) * 512, 2048);
      bulk_prefetch_l2(sf + (size_t)(64 + st * 4) * 512, 2048);
    } else {
      const int row0 = (st - kFc1Stages) * 512;
      tma_prefetch_2d(&tm2, 0, e * kH + row0);
      tma_prefetch_2d(&tm2, 0, e * kH + row0 + 256);
      bulk_prefetch_l2(p.w2_sf + (size_t)e * (kH * 8) + (size_t)(row0 / 128) * 2 * 512, 4096);
    }
  };
  // Rows of an item (the tokens of mask in increasing order, at most 16) into row buffer rb, by one warp
  auto set_rows = [&](int rb, int it) {
    const int n = s.item_n[it], ei = s.item_e[it];
    if (lane < n) {
      const int t = s.item_rows[it][lane];
      int ks = 0;
#pragma unroll
      for (int k = 0; k < kTopK; ++k)
        if (s.ids[t][k] == ei) ks = k;
      s.rtok[rb][lane] = t;
      s.rkslot[rb][lane] = ks;
    }
    if (lane == 0) s.nrows[rb] = n;
  };
  // their x rows and scale bytes into xa / xsf, by threads tid = 0 .. nt - 1
  auto copy_rows = [&](int rb, int tid, int nt) {
    const int n1 = s.nrows[rb] * 128, n = s.nrows[rb] * 144;  // 16-B granules: x rows, then their scale bytes
    if (GATHER_ASYNC) {
      for (int idx = tid; idx < n; idx += nt) {
        const uint8_t* src;
        uint8_t* dst;
        if (idx < n1) {
          const int r = idx >> 7, c = (idx >> 3) & 15, q = idx & 7;
          src = p.x_fp4 + (size_t)s.rtok[rb][r] * (kH / 2) + c * 128 + q * 16;
          dst = s.xa + c * 2048 + (r / 8) * 1024 + (r % 8) * 128 + ((q ^ (r % 8)) * 16);
        } else {
          const int r = (idx - n1) >> 4, q = (idx - n1) & 15;
          src = p.x_sf + (size_t)s.rtok[rb][r] * (kH / 16) + q * 16;
          dst = &s.xsf[r][q * 16];
        }
        asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" ::"r"(smem_u32(dst)), "l"(src) : "memory");
      }
      asm volatile("cp.async.wait_all;" ::: "memory");
      fence_proxy_async_smem();
      return;
    }
    // GATHER_BATCH loads a thread ahead of their stores: the global latency is paid once a batch
    for (int i0 = tid; i0 < n; i0 += GATHER_BATCH * nt) {
      uint4 v[GATHER_BATCH];
#pragma unroll
      for (int j = 0; j < GATHER_BATCH; ++j) {
        const int idx = i0 + j * nt;
        if (idx < n1) {  // row r, K chunk c (128 B), 16-B granule q of the chunk
          const int r = idx >> 7, c = (idx >> 3) & 15, q = idx & 7;
          v[j] = *reinterpret_cast<const uint4*>(p.x_fp4 + (size_t)s.rtok[rb][r] * (kH / 2) + c * 128 + q * 16);
        } else if (idx < n) {
          const int r = (idx - n1) >> 4, q = (idx - n1) & 15;
          v[j] = *reinterpret_cast<const uint4*>(p.x_sf + (size_t)s.rtok[rb][r] * (kH / 16) + q * 16);
        }
      }
#pragma unroll
      for (int j = 0; j < GATHER_BATCH; ++j) {
        const int idx = i0 + j * nt;
        if (idx < n1) {
          const int r = idx >> 7, c = (idx >> 3) & 15, q = idx & 7;
          *reinterpret_cast<uint4*>(s.xa + c * 2048 + (r / 8) * 1024 + (r % 8) * 128 + ((q ^ (r % 8)) * 16)) = v[j];
        } else if (idx < n) {
          const int r = (idx - n1) >> 4, q = (idx - n1) & 15;
          *reinterpret_cast<uint4*>(&s.xsf[r][q * 16]) = v[j];
        }
      }
    }
    fence_proxy_async_smem();  // the tensor core reads xa through the async proxy
  };
  // the given routing: warps 0, 2..7 load every slot's expert; warp 1 starts this CTA's weight stream at once on its
  // own slot's expert (a wrong guess only wastes its first ring fill)
  asm volatile("griddepcontrol.wait;" ::: "memory");
  if (threadIdx.x == 32) stamp(p, 10);
  if (warp == 1) {
    // speculate on this CTA's own slot: its expert's weight stream starts before the work split
    const uint32_t espec = (uint32_t)p.ids_in[slot];
    if (lane == 0) {
      stamp(p, 9);
      const uint64_t pol = policy_evict_first();
      for (int st = 0; st < SPEC_STAGES; ++st) {
        issue_stage(st, st, (int)espec, pol);
        if (STAGE_STAMPS) stamp(p, 64 + st);
      }
      // HBM is otherwise idle until the routing is done: the next stages go to L2 now (another CTA streams the
      // expert when this slot is not its first)
      for (int st = SPEC_STAGES; st < SPEC_STAGES + SPEC_PF && st < kTotStages; ++st) prefetch_stage(st, (int)espec);
      stamp(p, 2);
    }
    mbar_wait(&s.route_done, 0);
  } else {
    // the routing as given: the 224 routing threads load every slot's expert at once, then record and count them
    const int tid = (warp == 0 ? 0 : warp - 1) * 32 + lane, S = M * kTopK;
    constexpr int kPer = (kMaxTok * kTopK + 223) / 224;
    int ex[kPer];
#pragma unroll
    for (int j = 0; j < kPer; ++j) ex[j] = tid + 224 * j < S ? p.ids_in[tid + 224 * j] : 0;
#pragma unroll
    for (int j = 0; j < kPer; ++j) {
      const int s2 = tid + 224 * j;
      if (s2 < S) {
        s.ids[s2 / kTopK][s2 % kTopK] = ex[j];
        atomicMin(&s.first[ex[j]], s2);
        atomicAdd(&s.cnt[ex[j]], 1);
      }
    }
  }
  if (warp != 1) {
    const int base = warp == 0 ? 0 : warp - 1;
    if (DIAG_ROUTE && threadIdx.x == 0) stamp(p, 13);
    named_bar(4, 224);
    if (DIAG_ROUTE && threadIdx.x == 0) stamp(p, 15);
    // each routing warp marks its tokens' slots that are their expert's first and the units they hand out (s.first and
    // s.cnt are final after the barrier): a first occurrence at slot s2 of an expert with n tokens hands out its
    // ceil(n / 16) row groups, less the first one when s2 < G (that CTA runs it)
    const int G = (int)gridDim.x;
    {  // every slot at once over the 224 routing threads (per-token sums by shared-memory atomics)
      const int tid = base * 32 + lane, S = M * kTopK;
#pragma unroll
      for (int j = 0; j < (kMaxTok * kTopK + 223) / 224; ++j) {
        const int s2 = tid + 224 * j;
        if (s2 < S) {
          const int t = s2 / kTopK, k = s2 % kTopK, e2 = s.ids[t][k];
          const bool f = s.first[e2] == s2;
          const int u = f ? (s.cnt[e2] + kRows - 1) / kRows - (s2 < G ? 1 : 0) : 0;
          s.uslot[t][k] = (uint8_t)u;
          if (f) {
            atomicOr(&s.fbits[t], 1u << k);
            atomicAdd(&s.ounits[t], u);
          }
        }
      }
    }
    named_bar(4, 224);
    if (DIAG_ROUTE && threadIdx.x == 0) stamp(p, 20);
    if (warp == 0) {
      // Work items: (expert, row group) units. Slot c's expert's first unit belongs to CTA c when c is the expert's first
      // slot (its speculative stream already carries it). The other units, in order of their expert's first slot then
      // row group, go to the CTAs whose own slot repeats an earlier expert, in order; units beyond those CTAs become
      // further items of CTAs 0, 1, ... Every CTA derives the same assignment from the same routing.
      // Per-token prefixes over 128 tokens: lane l holds tokens l + 32 h, h < kGroups.
      auto scan = [&](int v) -> int {  // inclusive prefix over the lanes
#pragma unroll
        for (int d = 1; d < 32; d <<= 1) {
          const int up = __shfl_up_sync(0xffffffffu, v, d);
          v += lane >= d ? up : 0;
        }
        return v;
      };
      uint32_t fb[kGroups];
      int ou[kGroups], fi[kGroups], oi[kGroups];
      int fcarry = 0, ocarry = 0;
#pragma unroll
      for (int h = 0; h < kGroups; ++h) {
        const int t = lane + 32 * h;
        fb[h] = t < M ? s.fbits[t] : 0u;
        ou[h] = t < M ? s.ounits[t] : 0;
        fi[h] = scan(__popc(fb[h])) + fcarry;
        oi[h] = scan(ou[h]) + ocarry;
        fcarry = __shfl_sync(0xffffffffu, fi[h], 31);
        ocarry = __shfl_sync(0xffffffffu, oi[h], 31);
      }
      const int Uf = fcarry, O = ocarry;  // distinct experts, units handed out
      // token t's exclusive prefixes and first bits (t < M; h = t / 32 is uniform over the warp)
      auto tok = [&](int t, int& fex, int& oex, uint32_t& ftb) {
        const int h = t >> 5, l = t & 31;
        int fin = 0, oin = 0, our = 0;
        uint32_t f = 0;
#pragma unroll
        for (int g = 0; g < kGroups; ++g) {
          if (g == h) {
            fin = fi[g];
            oin = oi[g];
            our = ou[g];
            f = fb[g];
          }
        }
        fin = __shfl_sync(0xffffffffu, fin, l);
        oin = __shfl_sync(0xffffffffu, oin, l);
        our = __shfl_sync(0xffffffffu, our, l);
        ftb = __shfl_sync(0xffffffffu, f, l);
        oex = oin - our;
        fex = fin - __popc(ftb);
      };
      auto before = [&](int s2) -> int {  // first-occurrence slots before slot s2 (s2 <= M * 10)
        const int t = s2 / kTopK;
        if (t >= M) return Uf;
        int fex, oex;
        uint32_t ftb;
        tok(t, fex, oex, ftb);
        return fex + __popc(ftb & ((1u << (s2 % kTopK)) - 1u));
      };
      // the n-th (0-based) handed-out unit: (expert, row group)
      auto nth_unit = [&](int n, int& e, int& grp) {
        int t = 0;
#pragma unroll
        for (int h = 0; h < kGroups; ++h) t += __popc(__ballot_sync(0xffffffffu, lane + 32 * h < M && oi[h] <= n));
        int fex, oex;
        uint32_t ftb;
        tok(t, fex, oex, ftb);
        int k = 0, left = n - oex;
        while (left >= (int)s.uslot[t][k]) left -= s.uslot[t][k++];
        e = s.ids[t][k];
        grp = left + (t * kTopK + k < G ? 1 : 0);  // the first group of a slot below G is its own CTA's
      };
      const int nf_c = before(slot), nf_g = before(G);
      if (DIAG_ROUTE && lane == 0) stamp(p, 21);
      const int xc = s.ids[my_t][my_k];
      const bool owner = s.first[xc] == slot;
      const int D = G - nf_g, dc = slot - nf_c;  // duplicate CTAs; this one's rank among them
      // the whole warp walks the items (nth_unit is warp-collective); the lanes write each item's token list
      int ni = 0;
      auto add = [&](int e, int grp) {
        if (ni >= kMaxItems) __trap();
        int before_g = 0;  // the expert's tokens in earlier 32-token groups
#pragma unroll
        for (int h = 0; h < kGroups; ++h) {
          const int t = lane + 32 * h;
          bool has = false;
#pragma unroll
          for (int k = 0; k < kTopK; ++k) has |= t < M && s.ids[t][k] == e;
          const uint32_t bal = __ballot_sync(0xffffffffu, has);
          const int r = before_g + __popc(bal & ((1u << lane) - 1u)) - grp * kRows;  // row within the group
          if (has && r >= 0 && r < kRows) s.item_rows[ni][r] = (uint8_t)t;
          before_g += __popc(bal);
        }
        if (lane == 0) {
          s.item_e[ni] = e;
          s.item_n[ni] = min(kRows, before_g - grp * kRows);
        }
        ++ni;
      };
      if (owner) {
        add(xc, 0);
      } else if (dc < O) {
        int e, grp;
        nth_unit(dc, e, grp);
        add(e, grp);
      }
      for (int o = D + slot; o < O; o += G) {
        int e, grp;
        nth_unit(o, e, grp);
        add(e, grp);
      }
      __syncwarp();
      const int nall = ni;
      if (lane == 0) {
        s.nitems = nall;
        s.spec_ok = owner ? 1 : 0;
        stamp(p, 1);
        if (DIAG_ROUTE && p.trace) {  // DIAG values (not times): 22 = items, 23 = speculation kept
          p.trace[blockIdx.x * kTraceStride + 22] = nall;
          p.trace[blockIdx.x * kTraceStride + 23] = owner ? 1 : 0;
        }
        mbar_arrive(&s.route_done);
      }
    }
    named_bar(4, 224);
  }
  const int nitems = s.nitems;
  if (nitems == 0) {  // a duplicate slot with no work: let the speculative loads land, then leave
    if (warp == 1) {
      if (lane == 0)
        for (int b = 0; b < SPEC_STAGES; ++b) mbar_wait(&s.full[b], 0);
      __syncwarp();
    }
    named_bar(7, kThreads);
    if (warp == 2) tmem_dealloc(tm, 512);
    return;
  }
  tc_fence_after();
  const bool spec_ok = s.spec_ok != 0;
  // ring stage of item 0's stage 0: the speculative fill (stages 0 .. SPEC_STAGES - 1) is item 0's own only if spec_ok
  const int base0 = spec_ok ? 0 : SPEC_STAGES;
  float ag = 0.f, sc = 0.f, a2 = 0.f;
  auto load_scales = [&](int it) {
    const int e = s.item_e[it];
    ag = p.g1_alpha[e];
    sc = p.g1_scale_c[e];
    a2 = p.g2_alpha[e];
  };
  // FC1 drain of quarter-0 accumulator columns [64 c0, 64 c0 + 128) (TMEM lanes 0..15 hold the item rows) into acc1
  auto drain_fc1 = [&](int c0) {
#pragma unroll 1
    for (int c = c0; c < c0 + 2; ++c) {
      uint32_t r[64];
      tmem_ld_x64(tm + (uint32_t)(c * 64), r);
      tmem_ld_wait();
      if (lane < 16) {
#pragma unroll
        for (int i = 0; i < 16; ++i)
          *reinterpret_cast<uint4*>(s.acc1 + lane * kARow + c * 64 + 4 * i) =
              make_uint4(r[4 * i], r[4 * i + 1], r[4 * i + 2], r[4 * i + 3]);
      }
    }
    tc_fence_before();
    __syncwarp();
  };
  // SwiGLU + NVFP4 epilogue (h = 0: the one FC1 pass), tasks (row, 16-wide block, half) = 16 x 8 x 2 = 256,
  // task = et + k * nt.
  // Accumulator columns of each 32-block: [0, 8) up 2m, [8, 16) gate 2m, [16, 24) up 2m + 1, [24, 32) gate 2m + 1
  auto epi_tasks = [&](int h, int et, int nt) {
#pragma unroll 1
    for (int task = et; task < 256; task += nt) {
      const int row = task / 16, blk = (task / 2) % 8, hf = task % 2;
      const float4* ar = reinterpret_cast<const float4*>(s.acc1 + row * kARow + 32 * blk + 4 * hf);
      const float4 u_even = ar[0], g_even = ar[2], u_odd = ar[4], g_odd = ar[6];
      const float ga[8] = {g_even.x, g_odd.x, g_even.y, g_odd.y, g_even.z, g_odd.z, g_even.w, g_odd.w};
      const float ua[8] = {u_even.x, u_odd.x, u_even.y, u_odd.y, u_even.z, u_odd.z, u_even.w, u_odd.w};
      // trtllm-gen's SwiGLU + NVFP4 epilogue op for op (from its SASS); a = FC1 dequant, c = a x FC2 input scale:
      // h' = (u a)(g sigmoid(g a)), sigmoid = rcp(ex2(-(g (a log2e))) + 1), block scale e4m3(max|h'| c / 6), FP4 (h' / scale) c
      const float a_l2e = mul_ftz(ag, 1.4426950216293334961f);
      float hv[8];
      float amax = 0.f;
#pragma unroll
      for (int i = 0; i < 8; ++i) {
        const float ex = ex2_approx(-(ga[i] * a_l2e));
        const float sg = rcp_approx(ex + 1.0f);
        const float v = fmaf(ua[i], ag, 0.0f) * (ga[i] * sg);
        hv[i] = v;
        amax = fmaxf(amax, fabsf(v));
      }
      amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, 1));
      const uint32_t sf = cvt_e4m3(fmaxf(0.f, fabsf(amax * sc)) * 0.16666667163372039795f);
      const float sff = e4m3_to_f32(sf);
      const float osc = sff == 0.f ? 0.f : rcp_approx(sff);
      float qv[8];
#pragma unroll
      for (int i = 0; i < 8; ++i) qv[i] = mul_ftz(hv[i], osc) * sc;
      const uint32_t packed = cvt_e2m1x8(qv);
      const int gblk = 8 * h + blk;  // 16-wide block over the 128 intermediates; k-blocks 0, 1
      const int chunk = gblk / 2, r8 = row % 8;
      const uint32_t off = (uint32_t)((row / 8) * 1024 + r8 * 128 + ((chunk ^ r8) * 16) + (gblk % 2) * 8 + hf * 4);
#pragma unroll
      for (int cp = 0; cp < 8; ++cp) *reinterpret_cast<uint32_t*>(s.ha + cp * 2048 + off) = packed;
      if (hf == 0) reinterpret_cast<uint8_t*>(&s.hsf[row][gblk / 4])[gblk % 4] = (uint8_t)sf;
    }
  };
  // Rows of item it (its tokens in increasing order, at most 16) and their x rows / scale bytes, by warps 0 and 3;
  // row info is double-buffered by item parity (the previous item's FC2 drains still read theirs).
  auto gather = [&](int it) {
    const int rb = it & 1;
    if (warp == 0) set_rows(rb, it);
    named_bar(5, 64);
    copy_rows(rb, (warp == 0 ? 0 : 32) + lane, 64);
    named_bar(5, 64);
    if (warp == 0 && lane == 0) {
      mbar_arrive(&s.x_full);
      if (DIAG_ROUTE && it == 0) stamp(p, 14);
    }
  };

  if (warp == 1) {
    if (lane == 0) {
      // ---------------- TMA producer: every item's weight stream through the ring
      const uint64_t pol = policy_evict_first();
      for (int it = 0; it < nitems; ++it) {
        const int e = s.item_e[it], base = base0 + it * kTotStages;
        const int st0 = (it == 0 && spec_ok) ? SPEC_STAGES : 0;
        for (int st = st0; st < kTotStages; ++st) {
          const int g = base + st;
          if (FC2_PF > 0 && st == kFc1Stages - FC2_PF_LEAD) {
            // the FC2 tiles beyond the ring go to L2 now, while the last FC1 stages and h are in progress
            for (int j = kStages; j < kStages + FC2_PF; ++j) prefetch_stage(kFc1Stages + j, e);
          }
          if (PF_SET != 0 && ((st == st0 && (PF_SET & 1)) || (st == kFc1Stages + 4 && (PF_SET & 4)) ||
                              (st == kFc1Stages + 6 && (PF_SET & 8)))) {
            // these stages wait for MMAs that are slow to start after a phase change; let HBM fetch them meanwhile
            const int n = st == st0 ? PF_N1 : st == kFc1Stages + 4 ? PF_N2 : 4;
            for (int k = st; k < st + n && k < kTotStages; ++k) prefetch_stage(k, e);
          }
          if (g >= kStages) mbar_wait(&s.empty[g % kStages], ((g / kStages) - 1) & 1);
          issue_stage(g, st, e, pol);
          if (STAGE_STAMPS && it == 0) stamp(p, 64 + st);
        }
      }
    }
    __syncwarp();
  } else if (warp == 2) {
    if (lane == 0) {
      // ---------------- MMA issuer (tensor pipe: MMAs only)
      const uint32_t idesc = make_idesc_nvf4(128u, 256u);
      if (!spec_ok) {  // the speculative fill was another CTA's expert: recycle its ring stages without MMAs
        for (int g = 0; g < SPEC_STAGES; ++g) {
          mbar_wait(&s.full[g % kStages], (g / kStages) & 1);
          mbar_wait(&s.sf_full[g % kStages], (g / kStages) & 1);
          mbar_arrive(&s.empty[g % kStages]);
        }
      }
      for (int it = 0; it < nitems; ++it) {
        const int base = base0 + it * kTotStages, par = it & 1;
        if (it > 0) mbar_wait(&s.acc_drained, (it - 1) & 1);
        mbar_wait(&s.x_full, par);
        tc_fence_after();
        int n = 0;
        for (int st = 0; st < kFc1Stages; ++st) {
          const int g = base + st, b = g % kStages, ks = st;
          mbar_wait(&s.full[b], (g / kStages) & 1);
          mbar_wait(&s.sf_full[b], (g / kStages) & 1);
          tc_fence_after();
          if (st == 0 && it == 0) stamp(p, 3);
          if (st == 0 && it > 0 && it == nitems - 1) stamp(p, 11);
          if (DIAG_ROUTE && st == 0 && it < 4) stamp(p, 24 + it);  // DIAG: item it's first FC1 MMA
          if (DIAG_ROUTE && st == kFc1Stages - 1 && it < 4) stamp(p, 28 + it);  // DIAG: its last FC1 stage landed
#pragma unroll
          for (int q = 0; q < 4; ++q, ++n) {
            const int kb = ks * 4 + q;  // k-blocks in K order: one accumulator, the stock summation order
            umma_nvf4(tm, make_sdesc(smem_u32(s.xa + (kb / 4) * 2048) + (kb % 4) * 32, 0, 1024, 2),
                      make_sdesc(smem_u32(s.ring[b]) + q * 32, 0, 1024, 2), idesc,
                      tm + kColSfaX + (uint32_t)(kb % kSfaSlots) * 4, tm + kColSfb + (uint32_t)(n % kSfbSlots) * 8,
                      kb > 0);
          }
          tc_commit(&s.empty[b]);
        }
        tc_commit(&s.fc1_done);
        mbar_wait(&s.h_ready, par);
        tc_fence_after();
        if (it == 0) stamp(p, 6);
        for (int st = kFc1Stages; st < kTotStages; ++st) {
          const int g = base + st, b = g % kStages, f = st - kFc1Stages;
          if (f == 4) { mbar_wait(&s.fc2_free, par); tc_fence_after(); }
          mbar_wait(&s.full[b], (g / kStages) & 1);
          mbar_wait(&s.sf_full[b], (g / kStages) & 1);
          tc_fence_after();
#pragma unroll
          for (int tl = 0; tl < 2; ++tl) {
            const int j = 2 * f + tl, jt = j % 8;
#pragma unroll
            for (int kb = 0; kb < 2; ++kb) {
              const int n2 = 64 + j * 2 + kb;  // scale-factor slot, numbered on from FC1's 64 k-blocks
              umma_nvf4(tm, make_sdesc(smem_u32(s.ha) + kb * 32, 0, 1024, 2),
                        make_sdesc(smem_u32(s.ring[b] + tl * 16384) + kb * 32, 0, 512, 4), idesc,
                        tm + kColSfa + (uint32_t)(jt * 2 + kb) * 4, tm + kColSfb + (uint32_t)(n2 % kSfbSlots) * 8,
                        (jt | kb) != 0);
            }
          }
          tc_commit(&s.empty[b]);
          if (f % 4 == 3) tc_commit(&s.fc2_pass[f / 4]);
        }
      }
    }
    __syncwarp();
  } else if (warp == 0 || warp == 3) {
    // ---------------- x gather, FC1 drains (warp 0: columns [0, 128); warp 4 drains [128, 256)) and the epilogue
    const int et = (warp == 0 ? 0 : 32) + lane;  // 0..63
    gather(0);
    for (int it = 0; it < nitems; ++it) {
      const int par = it & 1;
      load_scales(it);
      if (warp == 0) {
        mbar_wait(&s.fc1_done, par);
        tc_fence_after();
        if (lane == 0 && it == 0) stamp(p, 4);
        drain_fc1(0);
        if (lane == 0) {
          mbar_arrive(&s.acc_ready);
          if (it == 0 && !DIAG_ROUTE) stamp(p, 13);
        }
      }
      mbar_wait(&s.acc_ready, par);
      epi_tasks(0, et, 192);
      if (warp == 3 && lane == 0 && it == 0 && !DIAG_ROUTE) stamp(p, 14);
      fence_proxy_async_smem();
      named_bar(6, 192);  // h and its scale words are complete for the TMEM warpgroup
      if (it + 1 < nitems) gather(it + 1);  // FC1 of item it is done: x and its scale bytes are free
    }
  } else {
    // ---------------- TMEM-side warpgroup (warps 4..7): q = lane quarter (TMEM lanes 32 q .. 32 q + 31)
    const int q = warp - 4;
    const uint32_t lane_addr = (uint32_t)(q * 32) << 16;
    if (!spec_ok) {  // the speculative fill is recycled without MMAs: release its scale-factor side
      for (int g = 0; g < SPEC_STAGES; ++g) {
        mbar_wait(&s.full[g % kStages], (g / kStages) & 1);
        __syncwarp();
        if (lane == 0) mbar_arrive(&s.sf_full[g % kStages]);
      }
    }
    for (int it = 0; it < nitems; ++it) {
      const int base = base0 + it * kTotStages, par = it & 1, rb = it & 1;
      load_scales(it);
      mbar_wait(&s.x_full, par);
      const int nrows = s.nrows[rb];
      auto stage_sf2 = [&](int st) {  // FC2 tiles 2 f, 2 f + 1: k-blocks 0, 1 of their row blocks (rows 0..127 | 128..255)
        const int g = base + st, b = g % kStages;
        mbar_wait(&s.full[b], (g / kStages) & 1);
        if (STAGE_STAMPS && threadIdx.x == 128 && it == 0) stamp(p, 16 + st);
        if (g >= 4) mbar_wait(&s.empty[(g - 4) % kStages], ((g - 4) / kStages) & 1);
        const uint8_t* sf = s.ring[b] + 32768;
        const int n0 = 64 + (st - kFc1Stages) * 4;
        for (int i = 0; i < 4; ++i) {
          const int tl = i / 2, kb = i % 2;
          const uint4 v0 = *reinterpret_cast<const uint4*>(sf + tl * 2048 + kb * 512 + lane * 16);
          const uint4 v1 = *reinterpret_cast<const uint4*>(sf + tl * 2048 + 1024 + kb * 512 + lane * 16);
          const uint32_t r[8] = {v0.x, v0.y, v0.z, v0.w, v1.x, v1.y, v1.z, v1.w};
          tmem_st_x8(tm + lane_addr + kColSfb + (uint32_t)((n0 + i) % kSfbSlots) * 8, r);
        }
        tmem_st_wait();
        tc_fence_before();
        __syncwarp();
        if (lane == 0) mbar_arrive(&s.sf_full[b]);
      };
      // FC1 stage st: weight scale factors of k-blocks [4 ks, 4 ks + 4) (8 columns each: row blocks 2 h | 2 h + 1) and
      // the x scale factors of the same k-blocks (row m -> lane m % 32, column m / 32 of each 4-column atom, every
      // quarter holds a copy; rows >= nrows, including the A rows read past the 16 real ones, get zero scales)
      for (int st = 0; st < kFc1Stages; ++st) {
        const int g = base + st, b = g % kStages, ks = st;
        mbar_wait(&s.full[b], (g / kStages) & 1);
        if (STAGE_STAMPS && threadIdx.x == 128 && it == 0) stamp(p, 16 + st);
        if (g >= 4) mbar_wait(&s.empty[(g - 4) % kStages], ((g - 4) / kStages) & 1);
        const uint8_t* sf = s.ring[b] + 32768;
#pragma unroll
        for (int i = 0; i < 4; ++i) {
          const uint4 v0 = *reinterpret_cast<const uint4*>(sf + i * 512 + lane * 16);
          const uint4 v1 = *reinterpret_cast<const uint4*>(sf + 2048 + i * 512 + lane * 16);
          const uint32_t r[8] = {v0.x, v0.y, v0.z, v0.w, v1.x, v1.y, v1.z, v1.w};
          tmem_st_x8(tm + lane_addr + kColSfb + (uint32_t)((st * 4 + i) % kSfbSlots) * 8, r);
        }
        uint32_t xr[16];
#pragma unroll
        for (int i = 0; i < 16; ++i) {
          const int kb = ks * 4 + i / 4;
          xr[i] = (i % 4 == 0 && lane < nrows) ? *reinterpret_cast<const uint32_t*>(&s.xsf[lane][kb * 4]) : 0u;
        }
        tmem_st_x16(tm + lane_addr + kColSfaX + (uint32_t)((ks * 4) % kSfaSlots) * 4, xr);
        tmem_st_wait();
        tc_fence_before();
        __syncwarp();
        if (lane == 0) mbar_arrive(&s.sf_full[b]);
      }
      // EARLY_SF2: stage the first FC2 scale factors before h (they wait for FC2 data to land, which delays h)
      for (int st = kFc1Stages; st < kFc1Stages + EARLY_SF2; ++st) stage_sf2(st);
      if (q == 0) {  // drain the upper 128 columns
        mbar_wait(&s.fc1_done, par);
        tc_fence_after();
        drain_fc1(2);
        if (lane == 0) mbar_arrive(&s.acc_ready);
      }
      mbar_wait(&s.acc_ready, par);
      epi_tasks(0, 64 + (warp - 4) * 32 + lane, 192);
      fence_proxy_async_smem();
      named_bar(6, 192);  // h complete
      if (threadIdx.x == 128 && it == 0) stamp(p, 5);
      // h scale factors: 16 atoms (tile jt, k-block kb) of 4 columns; row m = 32 c + lane is nonzero only in its tile
      {
        uint32_t r[32];
#pragma unroll 1
        for (int half = 0; half < 2; ++half) {
#pragma unroll
          for (int i = 0; i < 32; ++i) {
            const int a = half * 8 + i / 4, col = i % 4;  // atom a = 2 jt + kb
            const int jt = a / 2, kb = a % 2;
            const int row = 32 * col + lane;
            r[i] = (row / 16 == jt && row % 16 < nrows) ? s.hsf[row % 16][kb] : 0u;
          }
          tmem_st_x32(tm + lane_addr + kColSfa + half * 32, r);
        }
        tmem_st_wait();
      }
      tc_fence_before();
      __syncwarp();
      if (lane == 0) mbar_arrive(&s.h_ready);
      if (q == 0 && lane == 0 && it == 0 && !DIAG_ROUTE) stamp(p, 15);
      for (int st = kFc1Stages + EARLY_SF2; st < kTotStages; ++st) {
        if (st == kFc1Stages + 4) {
          // pass 0 is fully staged; drain it once its MMAs are done, then free the accumulator for pass 1
          mbar_wait(&s.fc2_pass[0], par);
          tc_fence_after();
          if (q == 0 && lane == 0 && it == 0) stamp(p, 7);
          fc2_drain16<0, 4>(p, s, tm, q, lane, 0, a2, rb);
          tc_fence_before();
          __syncwarp();
          if (lane == 0) mbar_arrive(&s.fc2_free);
        }
        stage_sf2(st);
      }
      mbar_wait(&s.fc2_pass[1], par);
      tc_fence_after();
      if (q == 0 && lane == 0 && it == 0) stamp(p, 12);
      if (TAIL8 && it + 1 == nitems) fc2_drain16<0, 2>(p, s, tm, q, lane, 1, a2, rb);
      else fc2_drain16<0, 4>(p, s, tm, q, lane, 1, a2, rb);
      tc_fence_before();
      __syncwarp();
      if (lane == 0) mbar_arrive(&s.acc_drained);
    }
  }
  if (TAIL8 && warp < 4) {  // the last item's last pass, upper columns: every warp is free now (TMEM lane quarter = warp)
    const int it = nitems - 1;
    load_scales(it);
    mbar_wait(&s.fc2_pass[1], it & 1);
    tc_fence_after();
    fc2_drain16<2, 4>(p, s, tm, warp, lane, 1, a2, it & 1);
  }
  tc_fence_before();
  named_bar(7, kThreads);  // every warp, including the drains, before TMEM is released
  if (threadIdx.x == 128) stamp(p, 8);
  if (warp == 2) tmem_dealloc(tm, 512);
}

static_assert(sizeof(Smem) + 1024 == Q35_MKM_SMEM_BYTES, "the launcher's dynamic shared memory size");

}  // namespace qwen_mkm_moe_m128


}  // namespace sglang
