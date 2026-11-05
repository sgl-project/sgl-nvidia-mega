// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
/**
 * \brief SWA-T: paged FP8 (e4m3) sliding-window verify attention for mid / large batches (``ops.swa_t``), one thread-block
 * cluster of kC CTAs per request (grid (kC, R), cluster (kC, 1, 1)). CTA c of request r takes chunk c of the request's
 * window tiles (tiles [floor(c n / kC), floor((c + 1) n / kC)) of its n window tiles) and runs full_attn_wide's per-tile
 * machinery and arithmetic on them (tcgen05 MMA, fp32 softmax as described below; kDiet only with its bitwise
 * scheduling bits 4 and 16 - see SwaTKernel). Its partial row records (64 unnormalized outputs, row maximum, row sum) go
 * to the row owners of the cluster over distributed shared memory (row i -> CTA i % kC; kPf bit 256: one bulk copy per
 * owner from a staging area in the CTA's own K ring, else st.async per 16 bytes); each owner waits on its receive
 * barrier, merges its rows' records in chunk order with the merge kernel's arithmetic below, applies the optional fused
 * output norm + gate (SGLang's norm_gate arithmetic, 16 threads per row) and writes bf16 rows. No merge kernel, no
 * global partials. A cluster barrier at entry (receive barriers initialised before any remote write) and, with bulk
 * records, one before exit (no CTA exits while a copy may still read its shared memory).
 *
 * QK prologue (kPf bits 16-17 = column class E 2 / 4 / 8; kPf bit 262144 = the unfused arithmetic form, below;
 * SwaTKernel::run_qk): the kernel takes the bf16 QKV projection
 * output instead of e4m3 q and does the work of SGLang's fused QK-norm call (Triton _qk_rms_norm_fwd_kernel with the FP8
 * KV-cache store) itself, with the arithmetic and per-thread column order of that kernel for the call's row count (as the
 * plugin's swa_verify_attn prologue): every CTA computes its request's q rows straight into its Q buffer (the layout the
 * q tensor map would load), and each of this step's K / V rows is computed and stored at cache_loc by the one CTA of the
 * cluster whose chunk holds that key's tile, which then reads it through its own tile copy (fence.proxy.async.global and
 * an mbarrier between the store and the copy). kDiet bit 32 is also available with fp32 exponentials (P in tensor memory).
 *
 * Ordering (use_pdl): with kPf bit 32 the CTA reads seq_lens, its page-table entries and the first ring stages of K/V
 * tiles that hold only keys older than this step's 8 tokens BEFORE griddepcontrol.wait (graph inputs and rows of earlier
 * steps: the late-writer class, valid in decode graph replays where a full stream dependency orders their writes before
 * the kernel); q and every tile holding this step's keys are read after the wait. The CTA lets its dependents launch
 * right after its wait (kPf bit 64: at its end instead). Every mbarrier wait is bounded (trap after a fixed number of
 * polls); every cluster-level wait is inside the CTA's own cluster.
 *
 * The documentation below is full_attn_wide's (the machinery and the per-tile arithmetic are the same; the work split and
 * the merge paths it describes for kC == 0 are not used by SWA-T).
 */
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
 * full_attn_wide_merge_kernel (launched next, PDL-chained) merges the request's records in block order.
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
 * for profiling only.
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

namespace swat {

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
  static constexpr int kSmBytes = kSmBar + 512 + 1024;  // barriers + tensor-memory slot + small words, alignment slack
};

struct Params {
  const int32_t* __restrict__ page_table;
  const int32_t* __restrict__ seq_lens;
  float* __restrict__ partial;      // [G][2 slots][heads * 64 rows][kRec] (localized: [2][G + kMaxReqs][rows][kRec])
  uint32_t* __restrict__ counters;  // [kMaxReqs], zero between calls
  __nv_bfloat16* __restrict__ out;
  int64_t page_table_stride;
  int32_t num_reqs;
  int32_t num_q_heads;
  float scale_log2;
  float out_scale;  // bmm2_scale / 448
  // localized variant (kLoc): pools placed per locality domain in 2^page_shift-page blocks (domain of page p =
  // ((p >> page_shift) & 1) ^ domain_flip); plan = full_attn_wide_plan_kernel's output; sm_tab[smid] = domain << 16 | rank
  // of the SM within its domain; blocks_dom[d] = SMs of domain d (= blocks of the main kernel on it)
  const int32_t* __restrict__ plan;
  const int32_t* __restrict__ sm_tab;
  int32_t plan_stride;  // page positions per (domain, request) row of the plan
  int32_t page_shift;
  int32_t domain_flip;
  int32_t blocks_dom[2];
  // sliding window (-1: full attention): token p attends keys [max(0, p - window_left), p]; a request's tiles start at
  // the tile of its first token's first key
  int32_t window_left;
  // output norm + gate in the merge kernel (null: off): gate [R * 8, >= Hq * 64] bf16, row stride gate_token_stride; with it
  // every request goes through the merge kernel (a request inside one block writes its record like any piece)
  const __nv_bfloat16* __restrict__ gate;
  int64_t gate_token_stride;
  float norm_eps;
  // SWA-T kPf bit 1024: bytes [0, l2pf_bytes) at l2pf (the next GEMM's weights) are prefetched into L2 at entry, spread
  // over the CTAs (static data: no dependency on the preceding kernels)
  const char* l2pf;
  int64_t l2pf_bytes;
  // SWA-T QK prologue (kPf bits 16-17 = E class): the bf16 QKV projection output [tokens, >= (Hq + 2 Hkv) * 64] (q heads,
  // then k heads, then v heads; row stride qkv_stride elements), positions [tokens] int64, cos_sin [max pos, 64] fp32,
  // cache_loc [tokens] int64 (pool rows of this step's tokens; 0 = skip), the QK-norm eps; K / V rows are stored into the
  // pools (k_store / v_store, rows of pool_row_bytes)
  const __nv_bfloat16* qkv;
  int64_t qkv_stride;
  const int64_t* positions;
  const float* cos_sin;
  const int64_t* cache_loc;
  float qk_eps;
  int32_t num_kv_heads;
  int64_t pool_row_bytes;
  uint8_t* k_store;
  uint8_t* v_store;
};

// SWA-T: receive region of a block of a kC-cluster (one receive barrier, then the records of its owned rows from every
// chunk of the request): [16 B barrier][ceil(rows / kC) x kC x kRec floats].
template <int kRecRows, int kC>
struct SwatRecv {
  static constexpr int kOwnMax = (kRecRows + kC - 1) / kC;
  static constexpr int kBytes = kC > 0 ? 16 + kOwnMax * kC * kRec * 4 : 0;
};
__device__ __forceinline__ uint32_t swat_mapa(uint32_t addr, uint32_t rank) {
  uint32_t out;
  asm volatile("mapa.shared::cluster.u32 %0, %1, %2;\n" : "=r"(out) : "r"(addr), "r"(rank));
  return out;
}
__device__ __forceinline__ void swat_st_async_v4(uint32_t addr, float4 v, uint32_t bar) {
  asm volatile("st.async.shared::cluster.mbarrier::complete_tx::bytes.v4.f32 [%0], {%1, %2, %3, %4}, [%5];\n"
               ::"r"(addr), "f"(v.x), "f"(v.y), "f"(v.z), "f"(v.w), "r"(bar) : "memory");
}
__device__ __forceinline__ void swat_st_async_v2(uint32_t addr, float a, float b2, uint32_t bar) {
  asm volatile("st.async.shared::cluster.mbarrier::complete_tx::bytes.v2.f32 [%0], {%1, %2}, [%3];\n"
               ::"r"(addr), "f"(a), "f"(b2), "r"(bar) : "memory");
}

// First tile (of kTileKeys keys) a request of KV length L reads, and its tile count.
template <int kTileKeys>
__device__ __forceinline__ int tile_lo(int L, int window_left) {
  return window_left < 0 ? 0 : max(0, L - kTokens - window_left) / kTileKeys;
}
template <int kTileKeys>
__device__ __forceinline__ int tiles_of(int L, int window_left) {
  return (max(L, 1) + kTileKeys - 1) / kTileKeys - tile_lo<kTileKeys>(L, window_left);
}

// Plan (localized variant), int32: tiles[2][kMaxReqs] (tiles of request r whose page is on domain d), then
// pos[2][kMaxReqs][plan_stride] (positions in the page table of request r's pages on domain d, ascending).
__device__ __forceinline__ int plan_tiles_off(int d, int r) { return d * kMaxReqs + r; }
__device__ __forceinline__ int64_t plan_pos_off(int d, int r, int stride) {
  return 2 * kMaxReqs + (static_cast<int64_t>(d) * kMaxReqs + r) * stride;
}

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
// (SWAT_MAX_POLLS: a small cap for compute-sanitizer builds, so an instrumented wait that cannot complete traps within
// seconds; the ops' MUSE_VR_*_SAN_POLLS env builds such a module under its own name)
#ifdef SWAT_MAX_POLLS
inline constexpr uint32_t kMaxPolls = SWAT_MAX_POLLS;
#else
inline constexpr uint32_t kMaxPolls = 1u << 24;
#endif
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
      "SWAT_WAIT_%=:\n"
      "mbarrier.try_wait.parity.shared::cta.b64 p, [%0], %1;\n"
      "@p bra SWAT_DONE_%=;\n"
      "add.u32 n, n, 1;\n"
      "setp.lt.u32 p, n, %2;\n"
      "@p bra SWAT_WAIT_%=;\n"
      "trap;\n"
      "SWAT_DONE_%=:\n"
      "}\n" ::"r"(bar),
      "r"(parity), "n"(kMaxPolls)
      : "memory");
}
// Pure polling (mbarrier.test_wait, never suspends); development variant of mbar_wait for barriers completed by thread
// arrivals. Bounded like mbar_wait.
__device__ __forceinline__ void mbar_wait_spin(uint32_t bar, uint32_t parity) {
  asm volatile(
      "{\n"
      ".reg .pred p;\n"
      ".reg .u32 n;\n"
      "mov.u32 n, 0;\n"
      "SWAT_SPIN_%=:\n"
      "mbarrier.test_wait.parity.shared::cta.b64 p, [%0], %1;\n"
      "@p bra SWAT_SDONE_%=;\n"
      "add.u32 n, n, 1;\n"
      "setp.lt.u32 p, n, %2;\n"
      "@p bra SWAT_SPIN_%=;\n"
      "trap;\n"
      "SWAT_SDONE_%=:\n"
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
// 16 consecutive 32-bit columns of this thread's tensor-memory lane
__device__ __forceinline__ void tmem_st16(uint32_t taddr, const uint32_t (&v)[16]) {
  asm volatile(
      "tcgen05.st.sync.aligned.32x32b.x16.b32 [%0], {%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16};\n" ::"r"(taddr),
      "r"(v[0]), "r"(v[1]), "r"(v[2]), "r"(v[3]), "r"(v[4]), "r"(v[5]), "r"(v[6]), "r"(v[7]), "r"(v[8]), "r"(v[9]), "r"(v[10]),
      "r"(v[11]), "r"(v[12]), "r"(v[13]), "r"(v[14]), "r"(v[15])
      : "memory");
}
__device__ __forceinline__ void tmem_wait_st() { asm volatile("tcgen05.wait::st.sync.aligned;\n" ::: "memory"); }
// MMA with A in tensor memory (warp-uniform operands, one elected lane issues)
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
// ---- SWA-T QK prologue: the arithmetic of SGLang's fused QK-norm Triton kernel (_qk_rms_norm_fwd_kernel with the FP8
// KV-cache store), as the plugin's swa_verify_attn prologue reproduces it bitwise per column class E (columns per thread,
// following the Triton kernel's block size for the call's row count): per (token, head) row of 64 values x, fp32,
// ss = sum of x^2 (each of the row's 64 / E threads sums its E columns in qk_sumsq<E>'s order, then a butterfly over the
// row's threads), r = rsqrt.approx.ftz(eps + div.full(ss, 64)), a = f32(bf16(r x)); RoPE on the pairs (2i, 2i+1) with
// c = cos_sin[pos][i], s = cos_sin[pos][32 + i]: y_2i = f32(bf16(fma(a_2i, c, -(a_2i+1 s)))), y_2i+1 = f32(bf16(fma(a_2i+1, c,
// a_2i s))); V rows y = x; stored e4m3(clamp(y, -448, 448)), nearest even, satfinite.
// That is the form of the Triton kernel compiled with FP fusion. Compiled without it (kUnfused), every product and sum is
// rounded separately: the 2-column sum of squares is x0*x0 + x1*x1, the mean is ss * div.approx(1, 64), and RoPE is
// y_2i = a_2i*c - a_2i+1*s, y_2i+1 = a_2i*s + a_2i+1*c (each product rounded, then the sum).
template <int E, bool kUnfused = false>
__device__ __forceinline__ float qk_sumsq(const float (&x)[E]) {
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
template <bool kUnfused = false>
__device__ __forceinline__ float qk_rstd(float ss, float eps) {
  float var, r;
  if constexpr (kUnfused) {
    float inv;
    asm("div.approx.f32 %0, %1, %2;\n" : "=f"(inv) : "f"(1.0f), "f"(64.0f));
    var = __fmul_rn(inv, ss);
  } else {
    asm("div.full.f32 %0, %1, %2;\n" : "=f"(var) : "f"(ss), "f"(64.0f));
  }
  const float t = __fadd_rn(eps, var);
  asm("rsqrt.approx.ftz.f32 %0, %1;\n" : "=f"(r) : "f"(t));
  return r;
}
__device__ __forceinline__ float qk_bf16_round(float x) { return __bfloat162float(__float2bfloat16_rn(x)); }
__device__ __forceinline__ uint16_t qk_e4m3x2(float x0, float x1) {
  uint16_t e;
  asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;\n" : "=h"(e) : "f"(x1), "f"(x0));
  return e;
}
// One prologue row slice (E columns of one (token, head) row) of one thread: loads first (issued in program order), then
// finish() gives the e4m3 bytes. rope: q / k rows; raw: v rows.
template <int E, bool kUnfused = false>
struct QkSlice {
  uint32_t w[E / 2];
  float cs[E];
  __device__ __forceinline__ void load(const __nv_bfloat16* src) {
    if constexpr (E == 2) {
      asm volatile("ld.global.b32 %0, [%1];\n" : "=r"(w[0]) : "l"(src));
    } else if constexpr (E == 4) {
      asm volatile("ld.global.v2.b32 {%0, %1}, [%2];\n" : "=r"(w[0]), "=r"(w[1]) : "l"(src));
    } else {
      asm volatile("ld.global.v4.b32 {%0, %1, %2, %3}, [%4];\n" : "=r"(w[0]), "=r"(w[1]), "=r"(w[2]), "=r"(w[3]) : "l"(src));
    }
  }
  __device__ __forceinline__ void load_rope(const float* row, int li) {  // row = cos_sin + pos * 64
#pragma unroll
    for (int m = 0; m < E / 2; ++m) {
      cs[m] = row[(E / 2) * li + m];
      cs[E / 2 + m] = row[32 + (E / 2) * li + m];
    }
  }
  __device__ __forceinline__ void finish(float eps, bool rope, bool raw, uint16_t (&b)[E / 2]) const {
    float x[E];
#pragma unroll
    for (int i = 0; i < E / 2; ++i) {
      const float2 f = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&w[i]));
      x[2 * i] = f.x;
      x[2 * i + 1] = f.y;
    }
    float ss = qk_sumsq<E, kUnfused>(x);
    constexpr int kLanes = 64 / E;
#pragma unroll
    for (int m = kLanes / 2; m >= 1; m >>= 1) ss = __fadd_rn(ss, __shfl_xor_sync(0xffffffffu, ss, m));
    const float r = qk_rstd<kUnfused>(ss, eps);
    float y[E];
#pragma unroll
    for (int e = 0; e < E; ++e) y[e] = qk_bf16_round(__fmul_rn(r, x[e]));
    if (rope) {
#pragma unroll
      for (int m = 0; m < E / 2; ++m) {
        const float c = cs[m], sn = cs[E / 2 + m];
        const float a0 = y[2 * m], a1 = y[2 * m + 1];
        if constexpr (kUnfused) {
          y[2 * m] = qk_bf16_round(__fsub_rn(__fmul_rn(a0, c), __fmul_rn(a1, sn)));
          y[2 * m + 1] = qk_bf16_round(__fadd_rn(__fmul_rn(a0, sn), __fmul_rn(a1, c)));
        } else {
          y[2 * m] = qk_bf16_round(__fmaf_rn(a0, c, -__fmul_rn(a1, sn)));
          y[2 * m + 1] = qk_bf16_round(__fmaf_rn(a1, c, __fmul_rn(a0, sn)));
        }
      }
    }
    if (raw) {
#pragma unroll
      for (int e = 0; e < E; ++e) y[e] = x[e];
    }
#pragma unroll
    for (int m = 0; m < E / 2; ++m)
      b[m] = qk_e4m3x2(fminf(fmaxf(y[2 * m], -448.0f), 448.0f), fminf(fmaxf(y[2 * m + 1], -448.0f), 448.0f));
  }
};
// kDiet bit 0 (paired half-precision exponentials): 2^y of two keys by one ex2.approx.f16x2, their row-sum adds as f16x2,
// their e4m3 probabilities straight from the f16 pair. h2_of(a, b): f16x2 with a in the low half, b in the high half.
__device__ __forceinline__ uint32_t h2_of(float a, float b) {
  uint32_t r;
  asm("cvt.rn.f16x2.f32 %0, %1, %2;\n" : "=r"(r) : "f"(b), "f"(a));
  return r;
}
__device__ __forceinline__ uint32_t ex2_h2(uint32_t x) {
  uint32_t y;
  asm("ex2.approx.f16x2 %0, %1;\n" : "=r"(y) : "r"(x));
  return y;
}
__device__ __forceinline__ uint32_t hmul2_u(uint32_t a, uint32_t b) {
  uint32_t d;
  asm("mul.rn.f16x2 %0, %1, %2;\n" : "=r"(d) : "r"(a), "r"(b));
  return d;
}
__device__ __forceinline__ uint32_t hadd2_u(uint32_t a, uint32_t b) {
  uint32_t d;
  asm("add.rn.f16x2 %0, %1, %2;\n" : "=r"(d) : "r"(a), "r"(b));
  return d;
}
// e4m3 pair (low byte = low half) of an f16x2 value
__device__ __forceinline__ uint32_t e4m3x2_h2(uint32_t h) {
  uint16_t r;
  asm("cvt.rn.satfinite.e4m3x2.f16x2 %0, %1;\n" : "=h"(r) : "r"(h));
  return r;
}
// e4m3 x4 (low byte first) of two f16x2 values: keys (lo.lo, lo.hi, hi.lo, hi.hi)
__device__ __forceinline__ uint32_t e4m3x4_h2(uint32_t lo, uint32_t hi) {
  uint32_t r;
  asm("{\n.reg .b16 a, b;\ncvt.rn.satfinite.e4m3x2.f16x2 a, %1;\ncvt.rn.satfinite.e4m3x2.f16x2 b, %2;\nmov.b32 %0, {a, b};\n}\n"
      : "=r"(r)
      : "r"(lo), "r"(hi));
  return r;
}
__device__ __forceinline__ float2 f2_of_h2(uint32_t h) {
  float a, b;
  asm("{\n.reg .f16 lo, hi;\nmov.b32 {lo, hi}, %2;\ncvt.f32.f16 %0, lo;\ncvt.f32.f16 %1, hi;\n}\n" : "=f"(a), "=f"(b) : "r"(h));
  return make_float2(a, b);
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

// Named-barrier OR reduction over `count` threads (all of them pass the same barrier id).
__device__ __forceinline__ bool bar_red_or(uint32_t id, uint32_t count, bool v) {
  uint32_t r;
  asm volatile(
      "{\n"
      ".reg .pred p, q;\n"
      "setp.ne.u32 p, %1, 0;\n"
      "bar.red.or.pred q, %2, %3, p;\n"
      "selp.u32 %0, 1, 0, q;\n"
      "}\n"
      : "=r"(r)
      : "r"(static_cast<uint32_t>(v)), "r"(id), "r"(count)
      : "memory");
  return r != 0;
}

// Named barrier `id` over `count` threads: wait (bar.sync) / signal without waiting (bar.arrive).
__device__ __forceinline__ void nbar_sync(uint32_t id, uint32_t count) {
  asm volatile("bar.sync %0, %1;\n" ::"r"(id), "r"(count) : "memory");
}
__device__ __forceinline__ void nbar_arrive(uint32_t id, uint32_t count) {
  asm volatile("bar.arrive %0, %1;\n" ::"r"(id), "r"(count) : "memory");
}

__device__ __forceinline__ uint32_t sm_id() {
  uint32_t v;
  asm volatile("mov.u32 %0, %%smid;\n" : "=r"(v));
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

template <int kPairs, int kStages, bool kUsePDL, int kPoly, int kPf, int kMode, int kTpr, bool kLoc, bool kWin, bool kPipe,
          bool kSkipRs, bool kLazy, int kPp, int kDiet, bool kGateW, int kC = 0, int kNgVec = 8>
__global__ void __launch_bounds__(Layout<kPairs, kStages, kTpr>::kThreads, 1)
    full_attn_wide_kernel(const Params p, const __grid_constant__ CUtensorMap q_map,
                          const __grid_constant__ CUtensorMap k_map, const __grid_constant__ CUtensorMap v_map) {
  using L = Layout<kPairs, kStages, kTpr>;
  constexpr int kHeads = L::kHeads, kTileKeys = L::kTileKeys, kSmWarps = L::kSmWarps, kKpt = L::kKpt;
  // development switch carried in kPf: kPf == 1 -> poll the thread-arrival barriers (no suspend), no L2 prefetch
  constexpr bool kSpinP = (kPf & 31) == 1;
  // SWA-T QK prologue (kPf bits 16-17: 1 / 2 / 3 = column class E 2 / 4 / 8): the kernel takes the bf16 QKV output and does
  // the fused QK-norm call's work itself (q rows into the Q buffer, this step's K / V rows into the pools)
  constexpr int kQkSel = (kPf >> 16) & 3;
  constexpr bool kQkUnfused = (kPf & 262144) != 0;  // the prologue's unfused arithmetic form (see qk_sumsq)
  constexpr int kQkE = kQkSel == 0 ? 0 : kQkSel == 1 ? 2 : kQkSel == 2 ? 4 : 8;
  static_assert(kQkE == 0 || (kC > 0 && kPairs == 1 && kTpr == 2), "QK prologue: SWA-T, one KV-head pair per CTA");
  extern __shared__ uint8_t smem_raw[];
  // 1024-byte aligned base by pointer arithmetic on the shared array (keeps the shared address space visible to the
  // compiler: st.shared / ld.shared instead of generic accesses)
  uint8_t* const sm = smem_raw + ((1024u - (smem_u32(smem_raw) & 1023u)) & 1023u);
  uint64_t* const bars = reinterpret_cast<uint64_t*>(sm + L::kSmBar);
  // SWA-T receive region: right after the barrier words (inside the allocation whatever the 1024-byte alignment shift of
  // `sm`; the layout's alignment slack then sits after it, at the end of the dynamic allocation)
  constexpr int kRecvOff = L::kSmBar + 512;
  // bars: full[kStages], empty[kStages], s_full[kNb], p_full[kNb], o_full[kNb], q_full[2], q_empty[2], s_free[kNb],
  // (tmem address). kPp (ping-pong KV-head pairs): the score / probability / output hand-offs are per pair (index
  // 2 * pair + buffer), so the two pairs' softmax warps do not share a barrier; otherwise one per buffer (pair ignored).
  constexpr int kNb = kPp ? 4 : 2;
  const uint32_t bar0 = smem_u32(bars);
  const auto pidx = [](int b, int c) { return kPp ? 2 * c + b : b; };
  const auto full_bar = [&](int s) { return bar0 + 8u * s; };
  const auto empty_bar = [&](int s) { return bar0 + 8u * (kStages + s); };
  const auto s_full_bar = [&](int b, int c = 0) { return bar0 + 8u * (2 * kStages + pidx(b, c)); };
  const auto p_full_bar = [&](int b, int c = 0) { return bar0 + 8u * (2 * kStages + kNb + pidx(b, c)); };
  const auto o_full_bar = [&](int b, int c = 0) { return bar0 + 8u * (2 * kStages + 2 * kNb + pidx(b, c)); };
  const auto q_full_bar = [&](int b) { return bar0 + 8u * (2 * kStages + 3 * kNb + b); };
  const auto q_empty_bar = [&](int b) { return bar0 + 8u * (2 * kStages + 3 * kNb + 2 + b); };
  // scores of buffer b (pair c) in registers
  const auto s_free_bar = [&](int b, int c = 0) { return bar0 + 8u * (2 * kStages + 3 * kNb + 4 + pidx(b, c)); };
  // SWA-T QK prologue: this step's K/V rows stored (softmax warps -> load warp, before tiles that hold them)
  const uint32_t kv_bar = bar0 + 8u * (2 * kStages + 4 * kNb + 4);
  constexpr int kBarWords = 2 * kStages + 4 * kNb + 5;
  static_assert(8 * kBarWords + 48 <= 512, "barrier region");
  uint32_t* const tmem_slot = reinterpret_cast<uint32_t*>(sm + L::kSmBar + 8 * kBarWords);
  // localized: end tile (block coordinates) of piece k at qend[k & 3], written by the load warp before the piece's Q
  // arrive (release) and read by the QK warp after its Q wait (acquire), so the QK warp needs no request walk
  int32_t* const qend = reinterpret_cast<int32_t*>(sm + L::kSmBar + 8 * kBarWords + 16);
  // kLazy: per (P buffer, KV-head pair) the PV target output buffer << 1 | accumulate flag, written by the pair's first
  // softmax lane before its p_full arrive (release) and read by the PV warp after its p_full wait (acquire)
  int32_t* const pvflag = qend + 4;
  int32_t* const pre = reinterpret_cast<int32_t*>(sm + L::kSmPre);  // pre[r] = tiles of requests < r; pre[R] = W
  int32_t* const pages = reinterpret_cast<int32_t*>(sm + L::kSmPages);

  // warp index through a warp shuffle: the compiler then knows it is warp-uniform, so the role branches below are
  // convergent (the MMA warp's descriptors stay in uniform registers and elect.sync needs no emulation)
  const int tid = threadIdx.x, warp = __shfl_sync(0xffffffffu, tid >> 5, 0), lane = tid & 31;
  // SWA-T (kC > 0): grid (kC, R, P), cluster (kC, 1, 1): block c = cluster rank of request blockIdx.y; blockIdx.z = the
  // KV-head pair of the rank this cluster takes (P = KV heads per rank / (2 kPairs): more pairs than the kernel's
  // layout holds are spread over clusters)
  const int po = kC > 0 ? static_cast<int>(blockIdx.z) : 0;  // first KV-head pair of this block, in units of kPairs pairs
  const int b = kC > 0 ? (static_cast<int>(blockIdx.z) * static_cast<int>(gridDim.y) + static_cast<int>(blockIdx.y)) * kC +
                             static_cast<int>(blockIdx.x)
                       : static_cast<int>(blockIdx.x);
  const int R = p.num_reqs;
  // localized variant: the block works on the tiles of one locality domain, as block `jb` of `Gd` there (claimed below,
  // after the dependency wait)
  int dom = 0, jb = b, Gd = static_cast<int>(gridDim.x);
  const int win = (kWin && !kLoc) ? p.window_left : -1;  // sliding window (compile-time off unless kWin)
  // partial records: [G][2] (whole-range split) or [2 domains][G + kMaxReqs] (localized: slot jb + r)
  // (SWA-T with global records: [R][kC] records, so the stamps go after those)
  const int64_t n_records = kLoc ? 2 * (static_cast<int64_t>(gridDim.x) + kMaxReqs)
                            : kC > 0 ? (static_cast<int64_t>(gridDim.y) * gridDim.z + 2) * static_cast<int64_t>(gridDim.x)
                                     : 2 * static_cast<int64_t>(gridDim.x);
  // development mode 4: clock stamps of every hand-off for blocks 0, G/2 and G-1 (tiles < 256), written after the
  // partial-record region of the workspace: [3 blocks][16 events][256 tiles] u64
  const int trace_slot = kMode != 4 ? -1 : (b == 0 ? 0 : b == static_cast<int>(gridDim.x) / 2 ? 1 : b == static_cast<int>(gridDim.x) - 1 ? 2 : -1);
  uint64_t* const trace = reinterpret_cast<uint64_t*>(p.partial + n_records * (2 * kPairs * kRows) * kRec);
  const auto stamp = [&](int ev, int gg) {
    if (kMode == 4 && trace_slot >= 0 && gg >= 0 && gg < 256) {
      uint64_t c;
      asm volatile("mov.u64 %0, %%clock64;\n" : "=l"(c));
      trace[(trace_slot * 16 + ev) * 256 + gg] = c;
    }
  };

  if (tid == 0) stamp(15, 1);  // mode 4: kernel entry
  if constexpr (kC > 0 && (kPf & 1024) != 0) {
    if (warp == kSmWarps + 2 && lane == 0 && p.l2pf != nullptr) {
      const int64_t nb = static_cast<int64_t>(gridDim.x) * gridDim.y * gridDim.z;
      const int64_t chunk = ((p.l2pf_bytes + nb - 1) / nb + 15) / 16 * 16;
      const int64_t beg = static_cast<int64_t>(b) * chunk, end = min(p.l2pf_bytes, beg + chunk);
      for (int64_t a = beg; a < end; a += 65536) {
        const int64_t rem = end - a;
        const uint32_t sz = static_cast<uint32_t>(rem < 65536 ? rem : 65536) & ~15u;
        if (sz) asm volatile("cp.async.bulk.prefetch.L2.global [%0], %1;\n" ::"l"(p.l2pf + a), "r"(sz) : "memory");
      }
    }
  }
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
#pragma unroll
      for (int c = 0; c < (kPp ? 2 : 1); ++c) {
        mbar_init(s_full_bar(i, c), 1);
        mbar_init(p_full_bar(i, c), kPp ? kSmWarps / 2 : kSmWarps);  // one elected lane per softmax warp (of the pair)
        mbar_init(o_full_bar(i, c), 1);
        mbar_init(s_free_bar(i, c), kPp ? kSmWarps / 2 : kSmWarps);
      }
      mbar_init(q_full_bar(i), 1);
      mbar_init(q_empty_bar(i), 1);
    }
    if constexpr (kC > 0) mbar_init(smem_u32(sm + kRecvOff), 1);  // SWA-T receive barrier
    if constexpr (kQkE > 0) mbar_init(kv_bar, 1);
    asm volatile("fence.mbarrier_init.release.cluster;\n" ::: "memory");
  }
  // SWA-T: the cluster barrier that publishes the receive barriers' initialisation completes here, at entry (arrive, then
  // wait at once). kPf bit 32768 = the split form instead (arrive here, wait before the first remote record store; ~0.1-0.2
  // us faster per call), which compute-sanitizer racecheck cannot follow (it stalls until the self-exit), so the default is
  // the form racecheck checks. (Completing it just before the dependency wait instead measured ~1 us slower.)
  constexpr bool kSplitCB = kC > 0 && (kPf & 32768) != 0;
  if constexpr (kC > 0) {
    if constexpr (kSplitCB) {
      asm volatile("barrier.cluster.arrive.relaxed.aligned;\n" ::: "memory");
    } else {
      asm volatile("barrier.cluster.arrive.release.aligned;\n" ::: "memory");
      asm volatile("barrier.cluster.wait.acquire.aligned;\n" ::: "memory");
    }
  }
  if (warp == 0) {
    asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], %1;\n" ::"r"(smem_u32(tmem_slot)),
                 "r"(kTmemCols)
                 : "memory");
    asm volatile("tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned;\n" ::: "memory");
  }
  // SWA-T staging (kPf bit 32): the prefix tile counts (seq_lens), this chunk's page list (page table) and its first
  // ring stages of K/V tiles holding only keys older than this step's 8 tokens are read BEFORE the dependency wait
  // (late-writer class: graph inputs and rows of earlier steps); q and this step's rows after it.
  constexpr bool kStageT = kC > 0 && (kPf & 32) != 0;
  // SWA-T bulk records (kPf bit 256): records are written to this CTA's shared memory (the K ring, free after the last
  // tile) grouped by owner, then one bulk copy per owner (instead of one st.async per 16 bytes); layout at the owner
  // [chunk][row slot][kRec]; a cluster barrier before exit keeps every source alive until its copies completed
  constexpr bool kBulkT = kC > 0 && (kPf & 256) != 0 && (kPf & 512) == 0;
  // SWA-T global records (kPf bit 512): records go to the workspace (L2) like full_attn_wide's partials, one cluster
  // barrier (release / acquire) orders them before the owners read them; no DSMEM traffic
  constexpr bool kGRecT = kC > 0 && (kPf & 512) != 0;
  int pre_issued = 0;  // load warp lane 0: tiles issued before the wait
  if constexpr (kStageT) {
    if (warp == kSmWarps + 1) {
      int carry = 0;
      for (int r0 = 0; r0 < R; r0 += 32) {
        const int r = r0 + lane;
        int n = 0;
        if (r < R) n = tiles_of<kTileKeys>(p.seq_lens[r], win);
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
    __syncthreads();  // barriers initialised, pre[] visible
    if (warp == kSmWarps) {
      constexpr int kTilesPerPage = 128 / kTileKeys;
      const int rr = static_cast<int>(blockIdx.y), nr = pre[rr + 1] - pre[rr], cc = static_cast<int>(blockIdx.x);
      const int lo = (nr * cc) / kC, hi = (nr * (cc + 1)) / kC;
      if (hi > lo) {
        const int seq_len = p.seq_lens[rr];
        const int tlo = tile_lo<kTileKeys>(seq_len, win);
        const int pg0 = (lo + tlo) / kTilesPerPage, pg1 = (hi - 1 + tlo) / kTilesPerPage + 1;
        const int32_t* const pt = p.page_table + static_cast<int64_t>(rr) * p.page_table_stride;
        for (int i = lane; i < pg1 - pg0 && i < kMaxPages; i += 32) pages[i] = pt[pg0 + i];
        __syncwarp();
        if (lane == 0) {
          asm volatile("fence.proxy.async.global;\n" ::: "memory");
          for (int i = lo; i < hi && i - lo < kStages; ++i) {
            if ((i + tlo + 1) * kTileKeys > seq_len - kTokens) break;  // holds this step's keys: after the wait
            const int s = i - lo;
            const int pi = (i + tlo) / kTilesPerPage - pg0;
            const int page = pi < kMaxPages ? pages[pi] : pt[(i + tlo) / kTilesPerPage];
            const int row0 = page * 128 + ((i + tlo) % kTilesPerPage) * kTileKeys;
            mbar_expect_tx(full_bar(s), 2 * kTileBytes);
#pragma unroll
            for (int c = 0; c < kPairs; ++c) {
              tma_load_2d(smem_u32(sm + L::kSmK + s * kTileBytes + c * kTileKeys * 128), &k_map, (po * kPairs + c) * 128, row0,
                          full_bar(s));
              tma_load_2d(smem_u32(sm + L::kSmV + s * kTileBytes + c * kTileKeys * 128), &v_map, (po * kPairs + c) * 128, row0,
                          full_bar(s));
            }
            ++pre_issued;
          }
          if constexpr ((kPf & 128) != 0) {
            // kPf bit 128: the chunk's remaining old-key tiles (beyond the ring) go to L2 before the wait as well
            for (int i = lo + pre_issued; i < hi; ++i) {
              if ((i + tlo + 1) * kTileKeys > seq_len - kTokens) break;
              const int pi = (i + tlo) / kTilesPerPage - pg0;
              const int page = pi < kMaxPages ? pages[pi] : pt[(i + tlo) / kTilesPerPage];
              const int row0 = page * 128 + ((i + tlo) % kTilesPerPage) * kTileKeys;
#pragma unroll
              for (int c = 0; c < kPairs; ++c) {
                tma_prefetch_2d(&k_map, (po * kPairs + c) * 128, row0);
                tma_prefetch_2d(&v_map, (po * kPairs + c) * 128, row0);
              }
            }
          }
        }
      }
    }
  }
  if constexpr (kUsePDL) asm volatile("griddepcontrol.wait;\n" ::: "memory");
  // SWA-T: the dependents (the output projection) may launch now and stream their weights during this kernel
  if constexpr (kUsePDL && kC > 0 && (kPf & 64) == 0) asm volatile("griddepcontrol.launch_dependents;\n" ::: "memory");
  if (tid == 0) stamp(15, 2);  // mode 4: after the dependency wait
  // Localized: claim the rank of this SM in its domain (sm_tab, from the SM census) in the per-call bitmask
  // counters[0..15] ([2 domains][8 words]); a block that lands on an SM whose rank is already taken (an SM that ran an
  // earlier block of this grid while another SM was busy) takes the first free rank, own domain first. Grid = ranks, so
  // every rank is taken exactly once whatever the placement; the merge kernel checks that and clears the bitmask.
  if constexpr (kLoc) {
    // warp 0 claims, every lane walking the same (lane-0-shuffled) decisions, lane 0 doing the atomics; no trap here
    // (a divergent trap region costs the MMA warps their provably uniform issue path): the merge kernel checks that
    // every rank was taken
    if (warp == 0) {
      const int e0 = __shfl_sync(0xffffffffu, p.sm_tab[sm_id()], 0);
      int d = e0 >> 16, j = e0 & 0xffff;
      uint32_t* const claim = p.counters;
      uint32_t old = 0;
      if (lane == 0) old = atomicOr(&claim[d * 8 + (j >> 5)], 1u << (j & 31));
      old = __shfl_sync(0xffffffffu, old, 0);
      if (old & (1u << (j & 31))) {
        int got = 0;
        for (int k = 0; k < 2 && !got; ++k) {
          const int dd = k ? 1 - d : d;
          const int g = dd ? p.blocks_dom[1] : p.blocks_dom[0];
          for (int jj = 0; jj < g && !got; ++jj) {
            const uint32_t bit = 1u << (jj & 31);
            uint32_t o2 = 0;
            if (lane == 0) o2 = atomicOr(&claim[dd * 8 + (jj >> 5)], bit);
            o2 = __shfl_sync(0xffffffffu, o2, 0);
            if (!(o2 & bit)) {
              d = dd;
              j = jj;
              got = 1;
            }
          }
        }
        if (!got) j = 0xffff;  // no free rank (more blocks than ranks): take no work
      }
      if (lane == 0) pre[kMaxReqs + 1] = (d << 16) | j;
    }
    __syncthreads();
    // through a lane-0 shuffle, so the compiler knows the value is warp-uniform (keeps the MMA warps' issue path uniform)
    const int e = __shfl_sync(0xffffffffu, pre[kMaxReqs + 1], 0);
    dom = e >> 16;
    jb = e & 0xffff;
    Gd = dom ? p.blocks_dom[1] : p.blocks_dom[0];  // (no dynamic index into the parameter struct: no local copy)
  }
  // Prefix tile counts (warp 9): pre[r] for r = 0..R (localized: tiles on the block's domain; a request can have none).
  if (!kStageT && warp == kSmWarps + 1) {
    int carry = 0;
    for (int r0 = 0; r0 < R; r0 += 32) {
      const int r = r0 + lane;
      int n = 0;
      if (r < R) n = kLoc ? p.plan[plan_tiles_off(dom, r)] : tiles_of<kTileKeys>(p.seq_lens[r], win);
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
  const int G = min(Gd, W);
  int T0c, T1c;
  if constexpr (kC > 0) {  // SWA-T: chunk blockIdx.x of request blockIdx.y's tiles
    const int rr = static_cast<int>(blockIdx.y), nr = pre[rr + 1] - pre[rr], cc = static_cast<int>(blockIdx.x);
    T0c = pre[rr] + (nr * cc) / kC;
    T1c = pre[rr] + (nr * (cc + 1)) / kC;
  } else {
    T0c = jb < G ? range_first(jb, G, W) : W;
    T1c = jb < G ? range_first(jb + 1, G, W) : W;
  }
  const int T0 = T0c, T1 = T1c;
  if constexpr (kC > 0) {
    // arm the receive barrier: the records of the rows this block owns from every chunk that holds tiles
    if (!kGRecT && tid == 0) {
      const int rr = static_cast<int>(blockIdx.y), nr = pre[rr + 1] - pre[rr];
      const int own = (kHeads * kRows - static_cast<int>(blockIdx.x) + kC - 1) / kC;
      mbar_expect_tx(smem_u32(sm + kRecvOff),
                     static_cast<uint32_t>(own * min(nr, kC) * (kBulkT ? kRec : kHeadDim + 2) * 4));
    }
  }
  const int ntiles = __shfl_sync(0xffffffffu, T1 - T0, 0);  // warp-uniform (provably, for the MMA warps' code)
  // the first request at or after r with tiles (localized: requests can have none on this domain); the loop condition
  // goes through a lane-0 shuffle so that, for a warp-uniform r, the loop is provably uniform (MMA warps' issue path)
  const auto nonempty = [&](int r) {
    if constexpr (kLoc)
      while (__shfl_sync(0xffffffffu, (r < R && pre[r + 1] == pre[r]) ? 1 : 0, 0)) ++r;
    return r;
  };
  // the same for code that one lane runs alone (no warp collective)
  const auto nonempty1 = [&](int r) {
    if constexpr (kLoc)
      while (r < R && pre[r + 1] == pre[r]) ++r;
    return r;
  };
  // page positions of request r on this domain (localized)
  const auto pos_of = [&](int r) { return p.plan + plan_pos_off(dom, r, p.plan_stride); };
  // first request of the range: the largest r with pre[r] <= T0 (it has tiles: an empty request shares its pre with
  // the next one)
  int r_first = 0;
  {
    int lo = 0, hi = R - 1;
    while (lo < hi) {
      const int mid = (lo + hi + 1) >> 1;
      if (pre[mid] <= T0) lo = mid; else hi = mid - 1;
    }
    r_first = kLoc ? __shfl_sync(0xffffffffu, lo, 0) : lo;
  }

  if (ntiles > 0) {
    if (warp == kSmWarps) {
      // ------------------------------------------------------------------ loads
      // page numbers of the range -> shared memory (all lanes, all loads in flight together)
      constexpr int kTilesPerPage = 128 / kTileKeys;
      if constexpr (!kStageT) {
        int r = r_first, base = 0;
        for (int t = T0; t < T1 && base < kMaxPages;) {
          r = nonempty(r);
          const int lo = t - pre[r], hi = min(T1, pre[r + 1]) - pre[r];
          const int tlo = win < 0 ? 0 : tile_lo<kTileKeys>(p.seq_lens[r], win);
          const int pg0 = (lo + tlo) / kTilesPerPage, pg1 = (hi - 1 + tlo) / kTilesPerPage + 1;
          const int32_t* const pt = p.page_table + static_cast<int64_t>(r) * p.page_table_stride;
          const int32_t* const ps = pos_of(r);
          if constexpr ((kDiet & 16) != 0) {
            // kDiet bit 4: eight page loads per lane issued before any store (a loop with one load per iteration keeps
            // only one load in flight per lane: its store waits for it before the next load issues)
            const int n = min(pg1 - pg0, kMaxPages - base);
            for (int i0 = 0; i0 < n; i0 += 256) {
              int v[8];
#pragma unroll
              for (int k = 0; k < 8; ++k) {
                const int i = i0 + lane + 32 * k;
                v[k] = i < n ? (kLoc ? pt[ps[pg0 + i]] : pt[pg0 + i]) : 0;
              }
#pragma unroll
              for (int k = 0; k < 8; ++k) {
                const int i = i0 + lane + 32 * k;
                if (i < n) pages[base + i] = v[k];
              }
            }
          } else {
          for (int i = lane; i < pg1 - pg0 && base + i < kMaxPages; i += 32)
            pages[base + i] = kLoc ? pt[ps[pg0 + i]] : pt[pg0 + i];
          }
          base += pg1 - pg0;
          t = pre[r] + hi;
          ++r;
        }
      }
      __syncwarp();
      if (lane == 0) {
        asm volatile("fence.proxy.async.global;\n" ::: "memory");
        stamp(15, 3);  // mode 4: page list filled, first loads next
        int g = 0, k = 0, r = r_first, base = 0;
        for (int t = T0; t < T1; ++k) {
          r = nonempty1(r);
          const int lo = t - pre[r], hi = min(T1, pre[r + 1]) - pre[r];
          const int tlo = win < 0 ? 0 : tile_lo<kTileKeys>(p.seq_lens[r], win);
          const int pg0 = (lo + tlo) / kTilesPerPage;
          // q rows of request r -> Q buffer k & 1 (free once the QK MMAs of piece k - 2 finished)
          constexpr int kQB = L::kQBufs;
          if ((kMode == 0 || kMode == 2 || kMode == 4 || kMode == 6) && k >= kQB)
            mbar_wait(q_empty_bar(k % kQB), ((k / kQB) - 1) & 1);
          if constexpr (kLoc) qend[k & 3] = pre[r] + hi;
          if constexpr (kQkE == 0) {  // (QK prologue: the softmax warps write the q rows)
            mbar_expect_tx(q_full_bar(k % kQB), kHeads * kRows * 64);
#pragma unroll
            for (int h = 0; h < kHeads; ++h)
              tma_load_3d(smem_u32(sm + L::kSmQ + (k % kQB) * L::kQBuf + h * kRows * 64), &q_map, 0,
                          (po * kHeads + h) * kQPerKv, r * kTokens, q_full_bar(k % kQB));
          }
          const int seq_len_r = p.seq_lens[r];
          bool kv_seen = false;
          const int32_t* const pt = p.page_table + static_cast<int64_t>(r) * p.page_table_stride;
          const int32_t* const ps = pos_of(r);
          // page-table index of tile i's page (localized: the i-th tile of the request on this domain)
          const auto pt_index = [&](int i) { return kLoc ? ps[i / kTilesPerPage] : (i + tlo) / kTilesPerPage; };
          for (int i = lo; i < hi; ++i, ++g) {
            if constexpr (kMode == 9) continue;  // tensor-core rate probe: no tile loads
            if (kStageT && g < pre_issued) continue;  // issued before the dependency wait
            if constexpr (kQkE > 0) {  // a tile with this step's keys: after the prologue stored those rows
              if (!kv_seen && (i + tlo + 1) * kTileKeys > seq_len_r - kTokens) {
                mbar_wait(kv_bar, 0);
                asm volatile("fence.proxy.async.global;\n" ::: "memory");
                kv_seen = true;
              }
            }
            const int s = g % kStages;
            if (g >= kStages) mbar_wait(empty_bar(s), ((g / kStages) - 1) & 1);
            const int pi = base + (i + tlo) / kTilesPerPage - pg0;
            const int page = pi < kMaxPages ? pages[pi] : pt[pt_index(i)];
            const int row0 = page * 128 + ((i + tlo) % kTilesPerPage) * kTileKeys;
            if constexpr ((kPf & 31) > 1) {
              // L2 prefetch of the tile kPf ahead in this piece (deeper memory-level parallelism than the ring);
              // SWA-T flag bits 32 (staging) and 64 (late trigger) are not part of the distance
              const int ip = i + (kPf & 31);
              if (ip < hi) {
                const int pip = base + (ip + tlo) / kTilesPerPage - pg0;
                const int pagep = pip < kMaxPages ? pages[pip] : pt[pt_index(ip)];
                const int rowp = pagep * 128 + ((ip + tlo) % kTilesPerPage) * kTileKeys;
#pragma unroll
                for (int c = 0; c < kPairs; ++c) {
                  tma_prefetch_2d(&k_map, (po * kPairs + c) * 128, rowp);
                  tma_prefetch_2d(&v_map, (po * kPairs + c) * 128, rowp);
                }
              }
            }
            mbar_expect_tx(full_bar(s), 2 * kTileBytes);
#pragma unroll
            for (int c = 0; c < kPairs; ++c) {
              tma_load_2d(smem_u32(sm + L::kSmK + s * kTileBytes + c * kTileKeys * 128), &k_map, (po * kPairs + c) * 128, row0,
                          full_bar(s));
              tma_load_2d(smem_u32(sm + L::kSmV + s * kTileBytes + c * kTileKeys * 128), &v_map, (po * kPairs + c) * 128, row0,
                          full_bar(s));
            }
            stamp(0, g);
          }
          base += (hi - 1 + tlo) / kTilesPerPage + 1 - pg0;
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
            ++k_q;
            if constexpr (!kLoc) {
              ++r_q;
              end_q = __shfl_sync(0xffffffffu, min(T1, pre[r_q + 1]), 0);
            }
            mbar_wait(q_full_bar(k_q % L::kQBufs), (k_q / L::kQBufs) & 1);
            if constexpr (kLoc) end_q = __shfl_sync(0xffffffffu, qend[k_q & 3], 0);
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
          for (int c = 0; c < kPairs; ++c) {
            if constexpr (kPp != 0) {
              // per pair: score buffer bq of this pair is free once its softmax warps hold tile gq - 2's scores
              if (gq >= 2) {
                mbar_wait(s_free_bar(bq, c), ((gq - 2) >> 1) & 1);
                __syncwarp();
                tc_fence_after();
              }
            }
#pragma unroll
            for (int hh = 0; hh < 2; ++hh)
#pragma unroll
              for (int kk = 0; kk < 2; ++kk)
                mma_f8_e(tmem + ((16u * hh) << 16) + 128u * bq + 64u * c,
                         dq + static_cast<uint64_t>(((2 * c + hh) * kRows * 64 + 32 * kk) >> 4),
                         dk + static_cast<uint64_t>((c * kTileKeys * 128 + 64 * hh + 32 * kk) >> 4), id_qk, kk);
            if constexpr (kPp != 0) mma_commit_e(s_full_bar(bq, c));
          }
          if constexpr (kPp == 0) mma_commit_e(s_full_bar(bq));
          if (lane == 0) stamp(2, gq);
        };
        mbar_wait(q_full_bar(0), 0);
        if constexpr (kLoc) end_q = __shfl_sync(0xffffffffu, qend[0], 0);
        __syncwarp();
        tc_fence_after();
        qk_tile(0);
        if (ntiles > 1) qk_tile(1);
        for (int g = 0; g + 2 < ntiles; ++g) {
          if constexpr (kPp == 0) {
            // kDiet bit 5: tile g's probabilities sit in score buffer g & 1 until its PV finished
            if constexpr ((kDiet & 32) != 0)
              mbar_wait(o_full_bar(g & 1), (g >> 1) & 1);
            else
              mbar_wait(s_free_bar(g & 1), (g >> 1) & 1);
            __syncwarp();
            tc_fence_after();
          }
          qk_tile(g + 2);  // kPp: per-pair score-buffer waits inside
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
          if constexpr (kPp == 0) {
            mbar_wait(p_full_bar(bb), (g >> 1) & 1);
            __syncwarp();
            if (lane == 0) stamp(6, g);
            tc_fence_after();
          }
          const uint64_t dp = dp0 + static_cast<uint64_t>((bb * kHeads * kRows * L::kPRowBytes) >> 4);
          const uint64_t dv = dv0 + static_cast<uint64_t>((s * kTileBytes) >> 4);
#pragma unroll
          for (int c = 0; c < kPairs; ++c) {
            if constexpr (kPp != 0) {  // per pair: its probabilities of tile g are in shared memory
              mbar_wait(p_full_bar(bb, c), (g >> 1) & 1);
              __syncwarp();
              if (lane == 0 && c == 0) stamp(6, g);
              tc_fence_after();
            }
            // output buffer and first-step accumulate flag of this pair (kLazy: from the softmax; else buffer bb, fresh)
            uint32_t ob = bb, acc0 = 0;
            if constexpr (kLazy) {
              const int f = __shfl_sync(0xffffffffu, pvflag[bb * 2 + c], 0);
              ob = static_cast<uint32_t>(f >> 1);
              acc0 = static_cast<uint32_t>(f & 1);
            }
#pragma unroll
            for (int hh = 0; hh < 2; ++hh)
#pragma unroll
              for (int kk = 0; kk < kTileKeys / 32; ++kk)
                if constexpr ((kDiet & 32) != 0)
                  mma_f8_ts_e(tmem + ((16u * hh) << 16) + 256u + 128u * ob + 64u * c,
                              tmem + ((16u * hh) << 16) + 128u * bb + 64u * c + 8u * kk,
                              dv + static_cast<uint64_t>((c * kTileKeys * 128 + 64 * hh + 4096 * kk) >> 4), id_pv,
                              kk > 0 ? 1u : acc0);
                else
                mma_f8_e(tmem + ((16u * hh) << 16) + 256u + 128u * ob + 64u * c,
                         dp + static_cast<uint64_t>(((2 * c + hh) * kRows * L::kPRowBytes + 32 * kk) >> 4),
                         dv + static_cast<uint64_t>((c * kTileKeys * 128 + 64 * hh + 4096 * kk) >> 4), id_pv,
                         kk > 0 ? 1u : acc0);
            if constexpr (kPp != 0) mma_commit_e(o_full_bar(bb, c));
          }
          if constexpr (kPp == 0) mma_commit_e(o_full_bar(bb));
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
      if constexpr (kQkE > 0) {
        // QK prologue (all softmax threads, uniform): q rows of the request -> Q buffer 0 (the TMA box layout: per KV head
        // h, row token * 8 + q head, 64-byte rows in the 64-byte swizzle); K / V rows of this step's tokens whose key lies
        // in this CTA's chunk -> the pools at cache_loc (each row stored by exactly one CTA of the cluster)
        constexpr int E = kQkE, kLn = 64 / E, kRpp = 32 * kSmWarps / kLn;  // threads per row, rows per pass
        const int li = tid % kLn, rp = tid / kLn;
        const int rq = static_cast<int>(blockIdx.y), seq_len = p.seq_lens[rq];
        const int64_t tok0 = static_cast<int64_t>(rq) * kTokens;
        const int hq = p.num_q_heads, hkv = p.num_kv_heads;
        constexpr int kQPass = kHeads * kRows / kRpp;
        static_assert(kQPass * kRpp == kHeads * kRows, "q prologue passes");
        QkSlice<E, kQkUnfused> qsl[kQPass];
#pragma unroll
        for (int ps = 0; ps < kQPass; ++ps) {
          const int j = ps * kRpp + rp, h = j / kRows, r64 = j % kRows;
          const int qh = (po * kHeads + h) * kQPerKv + (r64 & 7);
          qsl[ps].load(p.qkv + (tok0 + (r64 >> 3)) * p.qkv_stride + qh * kHeadDim + E * li);
        }
        constexpr int kKvRows = 2 * kHeads * kTokens, kKvPass = (kKvRows + kRpp - 1) / kRpp;
        QkSlice<E, kQkUnfused> kvs[kKvPass];
        int64_t kv_loc[kKvPass];
        bool kv_own[kKvPass];
        {
          const int nr = pre[rq + 1] - pre[rq], cq = static_cast<int>(blockIdx.x);
          const int clo = (nr * cq) / kC, chi = (nr * (cq + 1)) / kC;
          const int tlo_q = tile_lo<kTileKeys>(seq_len, win);
#pragma unroll
          for (int ps = 0; ps < kKvPass; ++ps) {
            const int j = min(ps * kRpp + rp, kKvRows - 1);
            const int t = j / (2 * kHeads), h = (j >> 1) % kHeads, isv = j & 1;
            const int tile = (seq_len - kTokens + t) / kTileKeys - tlo_q;
            kv_own[ps] = ps * kRpp + rp < kKvRows && tile >= clo && tile < chi;
            kv_loc[ps] = p.cache_loc[tok0 + t];
            const int col = (isv ? hq + hkv : hq) + po * kHeads + h;
            kvs[ps].load(p.qkv + (tok0 + t) * p.qkv_stride + col * kHeadDim + E * li);
          }
        }
#pragma unroll
        for (int ps = 0; ps < kQPass; ++ps) {
          const int r64 = (ps * kRpp + rp) % kRows;
          qsl[ps].load_rope(p.cos_sin + p.positions[tok0 + (r64 >> 3)] * kHeadDim, li);
        }
#pragma unroll
        for (int ps = 0; ps < kKvPass; ++ps) {
          const int j = min(ps * kRpp + rp, kKvRows - 1);
          if (!(j & 1)) kvs[ps].load_rope(p.cos_sin + p.positions[tok0 + j / (2 * kHeads)] * kHeadDim, li);
        }
#pragma unroll
        for (int ps = 0; ps < kQPass; ++ps) {
          const int j = ps * kRpp + rp, h = j / kRows, r64 = j % kRows;
          uint16_t bq[E / 2];
          qsl[ps].finish(p.qk_eps, true, false, bq);
          const int byte = E * li;
          uint8_t* const dst = sm + L::kSmQ + h * kRows * 64 + r64 * 64 + ((((byte >> 4) ^ ((r64 >> 1) & 3))) << 4) + (byte & 15);
          if constexpr (E == 2) {
            *reinterpret_cast<uint16_t*>(dst) = bq[0];
          } else if constexpr (E == 4) {
            *reinterpret_cast<uint32_t*>(dst) = static_cast<uint32_t>(bq[0]) | (static_cast<uint32_t>(bq[1]) << 16);
          } else {
            *reinterpret_cast<uint2*>(dst) = make_uint2(static_cast<uint32_t>(bq[0]) | (static_cast<uint32_t>(bq[1]) << 16),
                                                        static_cast<uint32_t>(bq[2]) | (static_cast<uint32_t>(bq[3]) << 16));
          }
        }
#pragma unroll
        for (int ps = 0; ps < kKvPass; ++ps) {
          const int j = min(ps * kRpp + rp, kKvRows - 1);
          const int h = (j >> 1) % kHeads, isv = j & 1;
          uint16_t bk[E / 2];
          kvs[ps].finish(p.qk_eps, !isv, isv, bk);  // every lane runs the row butterfly (owned or not)
          if (kv_own[ps] && kv_loc[ps] != 0) {
            uint8_t* const dst = (isv ? p.v_store : p.k_store) + kv_loc[ps] * p.pool_row_bytes + (po * kHeads + h) * kHeadDim +
                                 E * li;
            if constexpr (E == 2) {
              *reinterpret_cast<uint16_t*>(dst) = bk[0];
            } else if constexpr (E == 4) {
              *reinterpret_cast<uint32_t*>(dst) = static_cast<uint32_t>(bk[0]) | (static_cast<uint32_t>(bk[1]) << 16);
            } else {
              *reinterpret_cast<uint2*>(dst) = make_uint2(static_cast<uint32_t>(bk[0]) | (static_cast<uint32_t>(bk[1]) << 16),
                                                          static_cast<uint32_t>(bk[2]) | (static_cast<uint32_t>(bk[3]) << 16));
            }
          }
        }
        // generic writes (q in shared memory, K / V rows in global) before the async-proxy readers (QK MMAs, tile copies)
        asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory");
        asm volatile("fence.proxy.async.global;\n" ::: "memory");
        nbar_sync(1 + kSmWarps / kTpr, 32 * kSmWarps);
        if (tid == 0) {
          mbar_arrive(q_full_bar(0));
          mbar_arrive(kv_bar);
        }
      }
      int g = 0;
      int r = r_first;
      int cur = 1;  // kLazy: the output buffer the pair's PVs currently accumulate into (switches at every fold)
      for (int t = T0; t < T1; ++r) {
        r = nonempty(r);
        const int lo = t - pre[r], hi = min(T1, pre[r + 1]) - pre[r];
        const int n_r = pre[r + 1] - pre[r];
        const int32_t* const ps = pos_of(r);
        const int seq_len = p.seq_lens[r];
        const int pos = seq_len - kTokens + tok;
        const int tlo = win < 0 ? 0 : tile_lo<kTileKeys>(seq_len, win);
        float2 o2[kDims / 2];
#pragma unroll
        for (int d = 0; d < kDims / 2; ++d) o2[d] = make_float2(0.0f, 0.0f);
        float m = -INFINITY, l = 0.0f, ar_prev = 0.0f, at_prev = 0.0f, ar_prev2 = 0.0f, at_prev2 = 0.0f;
        // kDiet bit 1 (unit combine): the output and row sum are kept in units of 2^ru, ru = the reference the last tile's
        // probabilities were quantized against (its own maximum, floored at the running maximum - 64)
        float ru = -INFINITY;
        // rescale-and-add of an output tile into o (packed fp32 pairs: the same two roundings as fmaf(o, ar, v * at)):
        // wait for the tile's PV, then load it and add
        const auto combine_wait = [&](int gg) {
          mbar_wait(o_full_bar(gg & 1, c), (gg >> 1) & 1);
          if (tid == 0) stamp(8, gg);
          __syncwarp();  // each thread left the wait on its own: reconverge before the .sync.aligned tensor-memory accesses
          tc_fence_after();
        };
        const auto combine_math = [&](int gg, float ar_c, float at_c) {
          const int bp = gg & 1;
          const float2 a2 = make_float2(ar_c, ar_c), t2 = make_float2(at_c, at_c);
          if constexpr (kSkipRs) {
            // no row of this warp grew its maximum on tile gg (ar == 1 exactly): o += v * at, one fused rounding (the
            // two-rounding form o * 1 + (v * at) is skipped; a different, not less exact, rounding)
            if (__all_sync(0xffffffffu, ar_c == 1.0f)) {
#pragma unroll
              for (int d0 = 0; d0 < kDims; d0 += 32) {
                float v[32];
                tmem_ld32(tmem + lane_base + 256u + 128u * bp + o_col + d0, v);
                tmem_wait_ld();
#pragma unroll
                for (int d = 0; d < 16; ++d) o2[d0 / 2 + d] = fma2(make_float2(v[2 * d], v[2 * d + 1]), t2, o2[d0 / 2 + d]);
              }
              tc_fence_before();
              if (tid == 0) stamp(9, gg);
              return;
            }
          }
          if constexpr ((kDiet & 4) != 0) {
            // kDiet bit 2: every output column of the tile loaded before one wait (one tensor-memory load latency per
            // combine instead of one per 32 columns)
            float v[kDims];
#pragma unroll
            for (int d0 = 0; d0 < kDims; d0 += 32)
              tmem_ld32(tmem + lane_base + 256u + 128u * bp + o_col + d0, *reinterpret_cast<float(*)[32]>(v + d0));
            tmem_wait_ld();
#pragma unroll
            for (int d = 0; d < kDims / 2; ++d)
              o2[d] = (kDiet & 2) ? fma2(o2[d], a2, make_float2(v[2 * d], v[2 * d + 1]))
                                  : fma2(o2[d], a2, mul2(make_float2(v[2 * d], v[2 * d + 1]), t2));
          } else {
#pragma unroll
          for (int d0 = 0; d0 < kDims; d0 += 32) {
            float v[32];
            tmem_ld32(tmem + lane_base + 256u + 128u * bp + o_col + d0, v);
            tmem_wait_ld();
#pragma unroll
            for (int d = 0; d < 16; ++d)
              o2[d0 / 2 + d] = (kDiet & 2) ? fma2(o2[d0 / 2 + d], a2, make_float2(v[2 * d], v[2 * d + 1]))
                                           : fma2(o2[d0 / 2 + d], a2, mul2(make_float2(v[2 * d], v[2 * d + 1]), t2));
          }
          }
          tc_fence_before();
          if (tid == 0) stamp(9, gg);
        };
        const auto combine = [&](int gg, float ar_c, float at_c) {
          combine_wait(gg);
          combine_math(gg, ar_c, at_c);
        };
        // kLazy fold: wait for the last PV into buffer `buf` (tile gg), then o = (o + that buffer) * f
        const auto lazy_fold = [&](int gg, int buf, float f) {
          mbar_wait(o_full_bar(gg & 1, c), (gg >> 1) & 1);
          __syncwarp();  // each thread left the wait on its own: reconverge before the .sync.aligned tensor-memory accesses
          tc_fence_after();
          const float2 f2 = make_float2(f, f);
  #pragma unroll
          for (int d0 = 0; d0 < kDims; d0 += 32) {
            float v[32];
            tmem_ld32(tmem + lane_base + 256u + 128u * static_cast<uint32_t>(buf) + o_col + d0, v);
            tmem_wait_ld();
  #pragma unroll
            for (int d = 0; d < 16; ++d)
              o2[d0 / 2 + d] = mul2(add2(o2[d0 / 2 + d], make_float2(v[2 * d], v[2 * d + 1])), f2);
          }
          tc_fence_before();
        };
        for (int i = lo; i < hi; ++i, ++g) {
          const int bb = g & 1;
          // this thread's first key of the tile (localized: tile i of the request's tiles on this domain; only the
          // tiles of the list's last two pages can reach past position seq_len - 8 (page positions ascend), so only
          // they load their page position - no dependent global load on the other tiles' critical path)
          constexpr int kTpp = 128 / kTileKeys;
          const int key0 = (kLoc ? (i >= n_r - 2 * kTpp ? ps[i / kTpp] * 128 + (i % kTpp) * kTileKeys : 0)
                                 : (i + tlo) * kTileKeys) +
                           kKpt * half;
          mbar_wait(s_full_bar(bb, c), (g >> 1) & 1);
          if (tid == 0) stamp(3, g);
          __syncwarp();  // each thread left the wait on its own: reconverge before the .sync.aligned tensor-memory accesses
          tc_fence_after();
          float x[kKpt];
#pragma unroll
          for (int j = 0; j < kKpt; j += 32)
            tmem_ld32(tmem + lane_base + 128u * bb + s_col + j, *reinterpret_cast<float(*)[32]>(x + j));
          tmem_wait_ld();
          tc_fence_before();
          if (tid == 0) stamp(4, g);
          __syncwarp();
          if constexpr ((kDiet & 32) == 0)
            if (lane == 0) mbar_arrive(s_free_bar(bb, c));  // score buffer bb may take the QK of tile g + 2
          // tile edge: keys after the row's position do not take part. The test uses the smallest position of the request
          // (token 0), so it is warp-uniform: interior tiles branch around the masking instead of predicating it per key.
          if (key0 + kKpt - 1 > seq_len - kTokens) {
#pragma unroll
            for (int j = 0; j < kKpt; ++j)
              if (key0 + j > pos) x[j] = -INFINITY;
          }
          // window start (sliding window): keys before the row's first key do not take part; warp-uniform test against
          // the largest first key of the request's rows (token 7)
          if (win >= 0 && key0 < seq_len - 1 - win) {
#pragma unroll
            for (int j = 0; j < kKpt; ++j)
              if (key0 + j < pos - win) x[j] = -INFINITY;
          }
          if constexpr (kMode == 2) {  // development probe: no softmax arithmetic, P = 0
#pragma unroll
            for (int j = 0; j < kKpt; ++j) x[j] = 0.0f;
          }
          // maximum of the raw scores (the scale is positive, so max(s) * scale == max(s * scale))
          float mx[kKpt / 3 + 1];
#pragma unroll
          for (int j = 0; j < kKpt / 3; ++j) mx[j] = max3(x[3 * j], x[3 * j + 1], x[3 * j + 2]);
          float smax = kKpt % 3 == 2 ? fmaxf(x[kKpt - 2], x[kKpt - 1]) : x[kKpt - 1];
#pragma unroll
          for (int j = 0; j < kKpt / 3; j += 2) smax = j + 1 < kKpt / 3 ? max3(smax, mx[j], mx[j + 1]) : fmaxf(smax, mx[j]);
          if constexpr (kTpr == 2) {
            red[(bb * kRecRows + rec_row) * 2 + half] = smax;
            asm volatile("bar.sync %0, %1;\n" ::"r"(xbar), "r"(64) : "memory");
            smax = fmaxf(red[(bb * kRecRows + rec_row) * 2], red[(bb * kRecRows + rec_row) * 2 + 1]);
          }
          // the tile's own maximum: the probabilities of the tile are quantized against it
          const float mtile = smax == -INFINITY ? -INFINITY : smax * p.scale_log2;
          const float mnew = fmaxf(m, mtile);
          // kLazy: probabilities against the running maximum (the row's maximum key still maps to exactly 448 on the tile
          // that sets it); the pair's PVs accumulate in tensor memory across tiles, and the output is folded into the
          // registers (rescaled) only on tiles where some row of the KV-head pair grows its maximum (and at piece ends)
          bool fold_now = false;
          if constexpr (kLazy) fold_now = bar_red_or(9u + static_cast<uint32_t>(c), 128u * kTpr, mtile > m) || i == lo;
          // kDiet bit 1 (unit combine): the tile's probabilities are quantized against ru_new = max(tile maximum, running
          // maximum - 64) (= the tile maximum except on tiles more than 2^64 below the row's maximum, whose probabilities then
          // flush to zero), and the output / row sum are carried in units of 2^ru: o = o * 2^(ru - ru_new) + PV, one FMA per
          // element (o * ar + PV * at otherwise; one of ar, at is 1 for every row, the unit form needs no second factor)
          constexpr bool kUnit = (kDiet & 2) != 0, kExp16 = (kDiet & 1) != 0;
          const float ru_new = kUnit ? fmaxf(mtile, mnew - 64.0f) : 0.0f;
          const float mu = kUnit ? (ru_new == -INFINITY ? 0.0f : ru_new)
                                 : (kLazy ? (mnew == -INFINITY ? 0.0f : mnew) : (mtile == -INFINITY ? 0.0f : mtile));
          const float ar = kUnit ? (ru == -INFINITY ? 0.0f : fast_exp2(ru - ru_new))
                                 : (m == -INFINITY ? 0.0f : fast_exp2(m - mnew));
          const float at = (kLazy || kUnit) ? 1.0f : (mtile == -INFINITY ? 0.0f : fast_exp2(mtile - mnew));
          // 448 * 2^y as 2^(y + log2 448): probabilities come out pre-scaled for the e4m3 conversion, and the row sums are
          // kept in the same (x 448) units; the output scale divides by them (bmm2_scale / l448). kDiet bit 0: 2^(y + 4) in
          // f16 (in (0, 16]: the keys near the maximum keep ~11 bits, nothing that survives e4m3 is subnormal), then x 28 in
          // f16 -> the same x 448 probabilities (the maximum exactly 448) as the f32 path up to f16 rounding
          const float nmu448 = (kExp16 ? 4.0f : kLog2PScale) - mu;
          const float2 sc = make_float2(p.scale_log2, p.scale_log2), nmu = make_float2(nmu448, nmu448);
          float2 ls[2] = {make_float2(0.0f, 0.0f), make_float2(0.0f, 0.0f)};
#pragma unroll
          // kPipe: the output combine of two tiles back runs between the two halves of this tile's exponentials (its PV
          // is done by then), so its tensor-memory loads and FMAs overlap exponential work instead of following the
          // probability hand-off; same operations per element, combines in the same tile order
          constexpr int kJSplit = kPipe ? kKpt / 2 : kKpt;
          // probabilities -> P buffer bb. kPairs 2: rows of 64 B in the 64-byte swizzle (chunk q of row r at
          // q ^ (r / 2 % 4)); kPairs 1: rows of 128 B in the 128-byte swizzle (chunk q of row r at q ^ (r % 8)).
          // Buffer bb was last read by the MMAs of tile g - 2, finished before the output wait of the previous tile.
          uint8_t* const prow = sm + L::kSmP + (bb * kHeads + head) * kRows * L::kPRowBytes + row * L::kPRowBytes;
          // kDiet bit 0: e4m3 probabilities of keys 4k..4k+3 (low byte first) and four f16x2 partial row sums
          uint32_t pw[kKpt / 4];
          uint32_t hs[4] = {0u, 0u, 0u, 0u};
#pragma unroll
          for (int k = 0; k < kKpt / 4; ++k) pw[k] = 0u;
          const auto store_p = [&](int cc) {
            uint4 w;
            if constexpr (kExp16) {
              w = make_uint4(pw[4 * cc], pw[4 * cc + 1], pw[4 * cc + 2], pw[4 * cc + 3]);
            } else {
            w.x = e4m3x4(x[16 * cc + 0], x[16 * cc + 1], x[16 * cc + 2], x[16 * cc + 3]);
            w.y = e4m3x4(x[16 * cc + 4], x[16 * cc + 5], x[16 * cc + 6], x[16 * cc + 7]);
            w.z = e4m3x4(x[16 * cc + 8], x[16 * cc + 9], x[16 * cc + 10], x[16 * cc + 11]);
            w.w = e4m3x4(x[16 * cc + 12], x[16 * cc + 13], x[16 * cc + 14], x[16 * cc + 15]);
            }
            const int q = (kKpt / 16) * half + cc;
            const int chunk = kPairs == 2 ? (q ^ ((row >> 1) & 3)) : (q ^ (row & 7));
            *reinterpret_cast<uint4*>(prow + (chunk << 4)) = w;
          };
          // kPp 1: the two pairs' exponential phases alternate (named barriers 13 / 14 over both pairs' softmax warps):
          // pair 1 starts tile g's after pair 0 finished them, pair 0 starts tile g + 1's after pair 1 finished tile g's
          // (development variant; slower in the microbench)
          if constexpr (kPp == 1) {
            if (c == 1)
              nbar_sync(13, 32 * kSmWarps);
            else if (g > 0)
              nbar_sync(14, 32 * kSmWarps);
          }
          if constexpr (kExp16) {
            // two keys per ex2.approx.f16x2 (half the exponential instructions), x 28 in f16, f16 row-sum partials
            // (8 values <= 448 per half), e4m3 straight from the f16 pair
            constexpr uint32_t k28x2 = 0x4f004f00u;  // f16x2 (28, 28)
#pragma unroll
            for (int j = 0; j < kKpt && kMode != 2; j += 4) {
              const float2 y0 = fma2(make_float2(x[j], x[j + 1]), sc, nmu);
              const float2 y1 = fma2(make_float2(x[j + 2], x[j + 3]), sc, nmu);
              const uint32_t e0 = hmul2_u(ex2_h2(h2_of(y0.x, y0.y)), k28x2);
              const uint32_t e1 = hmul2_u(ex2_h2(h2_of(y1.x, y1.y)), k28x2);
              hs[(j >> 1) & 3] = hadd2_u(hs[(j >> 1) & 3], e0);
              hs[((j >> 1) + 1) & 3] = hadd2_u(hs[((j >> 1) + 1) & 3], e1);
              pw[j >> 2] = e4m3x4_h2(e0, e1);
            }
          }
#pragma unroll
          for (int j = 0; j < kJSplit && kMode != 2 && !kExp16; j += 2) {
            const float2 y = fma2(make_float2(x[j], x[j + 1]), sc, nmu);
            // kPoly of every 8 keys take 2^y from the FMA pipes (exp2_poly2), the rest from ex2.approx
            const float2 pj = (j & 7) < kPoly ? exp2_poly2(y) : make_float2(fast_exp2(y.x), fast_exp2(y.y));
            ls[(j >> 1) & 1] = add2(ls[(j >> 1) & 1], pj);
            x[j] = pj.x;
            x[j + 1] = pj.y;
          }
          if constexpr (kPipe) {
            // lag 2: the output tile of two tiles back (its PV finished during the previous tile) is added here, between
            // the halves of this tile's exponentials; its wait also proves that PV of two tiles back no longer reads P
            // buffer bb, so the first half's probabilities go out right after it (frees their registers)
            if (i >= lo + 2) combine_wait(g - 2);
#pragma unroll
            for (int cc = 0; cc < kJSplit / 16; ++cc) store_p(cc);
            if (i >= lo + 2) combine_math(g - 2, ar_prev2, at_prev2);
          }
#pragma unroll
          for (int j = kJSplit; j < kKpt && kMode != 2 && !kExp16; j += 2) {
            const float2 y = fma2(make_float2(x[j], x[j + 1]), sc, nmu);
            const float2 pj = (j & 7) < kPoly ? exp2_poly2(y) : make_float2(fast_exp2(y.x), fast_exp2(y.y));
            ls[(j >> 1) & 1] = add2(ls[(j >> 1) & 1], pj);
            x[j] = pj.x;
            x[j + 1] = pj.y;
          }
          if constexpr (kPp == 1) {
            if (c == 0)
              nbar_arrive(13, 32 * kSmWarps);
            else if (g + 1 < ntiles)
              nbar_arrive(14, 32 * kSmWarps);
          }
          // this thread's part of the tile row sum (its keys in four interleaved partial sums, (0 + 2) + (1 + 3);
          // kDiet bit 0: f16x2 partials (0 + 1), (2 + 3) to f32, then (lo + lo) + (hi + hi))
          float lpart;
          if constexpr (kExp16) {
            const float2 f01 = f2_of_h2(hadd2_u(hs[0], hs[1])), f23 = f2_of_h2(hadd2_u(hs[2], hs[3]));
            lpart = (f01.x + f23.x) + (f01.y + f23.y);
          } else {
            lpart = (ls[0].x + ls[1].x) + (ls[0].y + ls[1].y);
          }
          m = mnew;
          if constexpr (kLazy) {
            // P buffer bb was last read by PV of two tiles back (no per-tile output wait proves it here)
            if (g >= 2) mbar_wait(o_full_bar(bb, c), ((g - 2) >> 1) & 1);
            __syncwarp();  // each thread left the wait on its own: reconverge before the .sync.aligned tensor-memory accesses
          }
          if constexpr ((kDiet & 32) != 0) {
            // kDiet bit 5: the probabilities go to tensor memory over this tile's scores (already in registers): row lane,
            // e4m3 x4 per column, keys in order from the pair's first score column; the PV MMAs read them from there
            uint32_t pv16[16];
            if constexpr (!kExp16) {
              // fp32 exponentials: the same e4m3 conversion as the shared-memory path (store_p), keys 4k .. 4k+3 per word
#pragma unroll
              for (int k = 0; k < kKpt / 4; ++k) pw[k] = e4m3x4(x[4 * k], x[4 * k + 1], x[4 * k + 2], x[4 * k + 3]);
            }
#pragma unroll
            for (int k = 0; k < 16; ++k) pv16[k] = pw[k];
            tmem_st16(tmem + lane_base + 128u * bb + 64u * c + (kKpt / 4) * half, pv16);
            tmem_wait_st();
          } else {
#pragma unroll
          for (int cc = kPipe ? kJSplit / 16 : 0; cc < kKpt / 16; ++cc) store_p(cc);
          }
          if constexpr (kLazy) {
            if (q4 == 0 && half == 0 && lane == 0) pvflag[bb * 2 + c] = ((fold_now ? (cur ^ 1) : cur) << 1) | (fold_now ? 0 : 1);
          }
          if constexpr (kMode != 6 && (kDiet & 32) == 0) fence_proxy_async();  // mode 6: timing probe only (P visibility not ordered)
          tc_fence_before();
          if (tid == 0) stamp(5, g);
          __syncwarp();
          if (lane == 0) mbar_arrive(p_full_bar(bb, c));
          // this thread's half of the row sum (the two halves are added once, at the end of the piece)
          if constexpr (kLazy) {
            if (fold_now) {
              if (i > lo) lazy_fold(g - 1, cur, ar);
              cur ^= 1;
            }
          } else if constexpr (!kPipe) {
            if (i > lo) combine(g - 1, ar_prev, at_prev);
          }
          l = l * ar + lpart * at;
          ar_prev2 = ar_prev;
          at_prev2 = at_prev;
          ar_prev = ar;
          at_prev = at;
          if constexpr (kUnit) ru = ru_new;
        }
        // the piece's remaining output tiles, in tile order (kPipe: two, plain: one; kLazy: the current accumulator)
        if constexpr (kLazy) {
          lazy_fold(g - 1, cur, 1.0f);
        } else {
          if constexpr (kPipe) {
            if (hi - lo >= 2) combine(g - 2, ar_prev2, at_prev2);
          }
          combine(g - 1, ar_prev, at_prev);
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
        const bool whole = kC == 0 && lo == 0 && hi == n_r && (!kLoc || p.plan[plan_tiles_off(1 - dom, r)] == 0);
        if constexpr (kGRecT) {
          float* const grec =
              p.partial + (((static_cast<int64_t>(blockIdx.z) * gridDim.y + blockIdx.y) * kC + blockIdx.x) * kRecRows + rec_row) * kRec;
#pragma unroll
          for (int d = 0; d < kDims / 2; d += 2)
            *reinterpret_cast<float4*>(grec + kDims * half + 2 * d) = make_float4(o2[d].x, o2[d].y, o2[d + 1].x, o2[d + 1].y);
          if (half == 0) *reinterpret_cast<float2*>(grec + kHeadDim) = make_float2((kDiet & 2) ? ru : m, l);
        } else if constexpr (kBulkT) {
          // local record of row rec_row at [owner][row slot][kRec] in the K ring: every copy into the ring landed and
          // every MMA reading it finished (each tile's scores and output were consumed above); the proxy fence orders
          // those async-proxy accesses before these generic stores
          asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory");
          constexpr int kOwnMaxT = (kRecRows + kC - 1) / kC;
          // rows this CTA owns go straight into its own receive slots [chunk = own rank][row slot] (a bulk copy into the
          // issuing CTA's own shared memory is undefined: its destination must be another CTA's); the others are staged
          float* const lrec = (rec_row % kC == static_cast<int>(blockIdx.x))
                                  ? reinterpret_cast<float*>(sm + kRecvOff + 16) +
                                        (static_cast<int>(blockIdx.x) * kOwnMaxT + rec_row / kC) * kRec
                                  : reinterpret_cast<float*>(sm + L::kSmK) + ((rec_row % kC) * kOwnMaxT + rec_row / kC) * kRec;
#pragma unroll
          for (int d = 0; d < kDims / 2; d += 2)
            *reinterpret_cast<float4*>(lrec + kDims * half + 2 * d) = make_float4(o2[d].x, o2[d].y, o2[d + 1].x, o2[d + 1].y);
          if (half == 0) *reinterpret_cast<float2*>(lrec + kHeadDim) = make_float2((kDiet & 2) ? ru : m, l);
        } else if constexpr (kC > 0) {
          // SWA-T: this chunk's record of row rec_row -> its owner (rank rec_row % kC), slot (rec_row / kC, chunk)
          if constexpr (kSplitCB) asm volatile("barrier.cluster.wait.aligned;\n" ::: "memory");
          const uint32_t owner = static_cast<uint32_t>(rec_row % kC);
          const uint32_t rbase = smem_u32(sm + kRecvOff);
          const uint32_t dst = swat_mapa(rbase + 16u + static_cast<uint32_t>(((rec_row / kC) * kC + static_cast<int>(blockIdx.x)) * kRec * 4),
                                         owner);
          const uint32_t rbar = swat_mapa(rbase, owner);
#pragma unroll
          for (int d = 0; d < kDims / 2; d += 2)
            swat_st_async_v4(dst + 4u * static_cast<uint32_t>(kDims * half + 2 * d),
                             make_float4(o2[d].x, o2[d].y, o2[d + 1].x, o2[d + 1].y), rbar);
          if (half == 0) swat_st_async_v2(dst + 4u * kHeadDim, (kDiet & 2) ? ru : m, l, rbar);
        } else if (whole && !kGateW) {
          // the whole request in this block: normalize and store
          store_out(o2, l > 0.0f ? p.out_scale / l : 0.0f);
        } else {
          // partial record -> slot (block, 0) when the request started in an earlier block, else (block, 1); with the
          // norm + gate (not localized) a request inside this block writes its record to its own slot 2 G + r
          const int slot = (pre[r] < T0) ? 0 : 1;
          const int64_t rec_idx = kLoc ? static_cast<int64_t>(dom) * (gridDim.x + kMaxReqs) + jb + r
                                  : (kGateW && whole) ? 2 * static_cast<int64_t>(gridDim.x) + r
                                          : static_cast<int64_t>(b) * 2 + slot;
          float* const rec = p.partial + (rec_idx * kRecRows + rec_row) * kRec;
#pragma unroll
          for (int d = 0; d < kDims / 2; d += 2)
            *reinterpret_cast<float4*>(rec + kDims * half + 2 * d) = make_float4(o2[d].x, o2[d].y, o2[d + 1].x, o2[d + 1].y);
          if (half == 0) *reinterpret_cast<float2*>(rec + kHeadDim) = make_float2((kDiet & 2) ? ru : m, l);
          // merged by full_attn_wide_merge_kernel (launched after this kernel)
        }
        t = pre[r] + hi;
      }
      if constexpr (kBulkT) {
        // all of this CTA's records are in shared memory -> one bulk copy per owner (lane o of warp 0 -> owner o)
        asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory");  // generic record stores -> the bulk copies
        if constexpr (kSplitCB) asm volatile("barrier.cluster.wait.aligned;\n" ::: "memory");
        nbar_sync(1 + kSmWarps / kTpr, 32 * kSmWarps);  // next id after the pair exchanges (1 .. kSmWarps / kTpr)
        if (tid == kC) {
          // the self-owned rows (written above, ordered by the block-level barrier for every reader of this CTA): their
          // bytes on the local receive barrier
          const int own_s = (kRecRows - static_cast<int>(blockIdx.x) + kC - 1) / kC;
          asm volatile("mbarrier.complete_tx.relaxed.cta.shared::cta.b64 [%0], %1;\n" ::"r"(smem_u32(sm + kRecvOff)),
                       "r"(static_cast<uint32_t>(own_s * kRec * 4))
                       : "memory");
        }
        if (tid < kC && tid != static_cast<int>(blockIdx.x)) {
          constexpr int kOwnMaxT = (kRecRows + kC - 1) / kC;
          const int o = tid, own_o = (kRecRows - o + kC - 1) / kC;
          const uint32_t rbase = smem_u32(sm + kRecvOff);
          const uint32_t src = smem_u32(sm + L::kSmK) + static_cast<uint32_t>(o * kOwnMaxT * kRec * 4);
          const uint32_t dst = swat_mapa(rbase + 16u + static_cast<uint32_t>(static_cast<int>(blockIdx.x) * kOwnMaxT * kRec * 4),
                                         static_cast<uint32_t>(o));
          const uint32_t rb = swat_mapa(rbase, static_cast<uint32_t>(o));
          asm volatile("cp.async.bulk.shared::cluster.shared::cta.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];\n"
                       ::"r"(dst), "r"(src), "r"(static_cast<uint32_t>(own_o * kRec * 4)), "r"(rb) : "memory");
        }
      }
    }
  }
  if constexpr (kGRecT) {
    // every record of the cluster is in global memory before any owner reads (release / acquire)
    if constexpr (kSplitCB) asm volatile("barrier.cluster.wait.aligned;\n" ::: "memory");
    asm volatile("barrier.cluster.arrive.release.aligned;\n" ::: "memory");
    asm volatile("barrier.cluster.wait.acquire.aligned;\n" ::: "memory");
  }
  if constexpr (kC > 0) {
    // SWA-T owner merge (the softmax warps; full_attn_wide_merge_kernel's arithmetic): rows rank, rank + kC, ... of the
    // request; 16 threads per row (4 output dims each); pieces = the chunks with tiles, in chunk order
    constexpr int kRecRowsT = kHeads * kRows;
    if (warp < kSmWarps) {
      const int rr = static_cast<int>(blockIdx.y), nr = pre[rr + 1] - pre[rr], rank = static_cast<int>(blockIdx.x);
      const uint32_t rbar = smem_u32(sm + kRecvOff);
      const float* const recv = reinterpret_cast<const float*>(sm + kRecvOff + 16);
      if constexpr (!kGRecT) mbar_wait(rbar, 0);
      __syncwarp();  // reconverge before the merge's group shuffles
      const int own = (kRecRowsT - rank + kC - 1) / kC;
      const int part = tid & 15;
      for (int i = tid >> 4; i < own; i += kSmWarps * 2) {
        const int row = i * kC + rank;
        const auto has_tiles = [&](int cc) { return (nr * (cc + 1)) / kC > (nr * cc) / kC; };
        // record of chunk cc for row slot i: [i][cc] (st.async records) or [cc][i] (bulk records)
        constexpr int kOwnMaxM = (kRecRowsT + kC - 1) / kC;
        const auto rec_of = [&](int cc) {
          return kGRecT ? p.partial + (((static_cast<int64_t>(blockIdx.z) * gridDim.y + rr) * kC + cc) * kRecRowsT + row) * kRec
                        : recv + (kBulkT ? cc * kOwnMaxM + i : i * kC + cc) * kRec;
        };
        const auto ld1 = [&](const float* q) { if constexpr (kGRecT) return __ldcg(q); else return *q; };
        const auto ld4 = [&](const float* q) {
          if constexpr (kGRecT) return __ldcg(reinterpret_cast<const float4*>(q));
          else return *reinterpret_cast<const float4*>(q);
        };
        // every chunk's (m, l, 4 outputs) loaded up front, unconditionally (one load latency for the row instead of one
        // per chunk; a slot of a chunk without tiles holds stale values that the has_tiles tests below never use); the
        // arithmetic and its order are unchanged
        const int head = row >> 6, r64 = row & 63, tok = r64 >> 3, qh = (po * kHeads + head) * kQPerKv + (r64 & 7);
        // kGateW: the row's 4 gate values are loaded together with the records (same latency round)
        uint2 graw = make_uint2(0u, 0u);
        if constexpr (kGateW)
          graw = *reinterpret_cast<const uint2*>(p.gate + (static_cast<int64_t>(rr) * kTokens + tok) * p.gate_token_stride +
                                                 qh * kHeadDim + 4 * part);
        float mjs[kC], ljs[kC];
        float4 vjs[kC];
#pragma unroll
        for (int cc = 0; cc < kC; ++cc) {
          const float* const rj = rec_of(cc);
          mjs[cc] = ld1(rj + kHeadDim);
          ljs[cc] = ld1(rj + kHeadDim + 1);
          vjs[cc] = ld4(rj + 4 * part);
        }
        float mx = -INFINITY;
#pragma unroll
        for (int cc = 0; cc < kC; ++cc)
          if (has_tiles(cc)) mx = fmaxf(mx, mjs[cc]);
        float acc[4] = {0.0f, 0.0f, 0.0f, 0.0f};
        float lsum = 0.0f;
#pragma unroll
        for (int cc = 0; cc < kC; ++cc) {
          if (!has_tiles(cc)) continue;
          const float mj = mjs[cc];
          if (mj == -INFINITY) continue;
          const float w = fast_exp2(mj - mx);
          const float4 v = vjs[cc];
          acc[0] += w * v.x;
          acc[1] += w * v.y;
          acc[2] += w * v.z;
          acc[3] += w * v.w;
          lsum += w * ljs[cc];
        }
        const float f = lsum > 0.0f ? p.out_scale / lsum : 0.0f;
        __nv_bfloat16* const orow =
            p.out + ((static_cast<int64_t>(rr) * kTokens + tok) * p.num_q_heads + qh) * kHeadDim + 4 * part;
        __nv_bfloat162 h0 = __floats2bfloat162_rn(acc[0] * f, acc[1] * f);
        __nv_bfloat162 h1 = __floats2bfloat162_rn(acc[2] * f, acc[3] * f);
        if constexpr (kGateW) {
          norm_gate4<kNgVec>(h0, h1, graw, p.norm_eps);
        }
        uint2 wv;
        wv.x = *reinterpret_cast<uint32_t*>(&h0);
        wv.y = *reinterpret_cast<uint32_t*>(&h1);
        *reinterpret_cast<uint2*>(orow) = wv;
      }
    }
  }
  tc_fence_before();
  __syncthreads();
  if (tid == 0) stamp(14, 0);
  if constexpr (kUsePDL && (kC == 0 || (kPf & 64) != 0)) asm volatile("griddepcontrol.launch_dependents;\n" ::: "memory");
  if constexpr (kBulkT) {
    // no CTA exits while a bulk copy may still read its shared memory: every owner has waited for all of its records
    // (above) before it arrives here (split form: threads that have not waited on the entry phase do so first)
    if constexpr (kSplitCB)
      if (!(warp < kSmWarps && ntiles > 0)) asm volatile("barrier.cluster.wait.aligned;\n" ::: "memory");
    asm volatile("barrier.cluster.arrive.release.aligned;\n" ::: "memory");
    asm volatile("barrier.cluster.wait.acquire.aligned;\n" ::: "memory");
  }
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
// one for the pieces' maxima (spread over the 16 threads of a row) together with the first kMergeBatch pieces' outputs, one
// per further kMergeBatch pieces. The host takes kMergeBatch = 24 only at 2 KV heads per rank when a request can span more
// than 8 blocks (fewer requests than blocks / 8): the 24-piece round needs ~250 registers, which halves the merge blocks per SM
// and loses with larger merge grids. The arithmetic (maximum, weights, fused multiply-adds in block order) is that of the
// earlier merge, whatever the batch. kGate: the output norm + gate (p.gate) and the single-block requests' own record slots.
// With p.gate the merged rows also get the output norm and gate (SGLang's norm_gate arithmetic for 16 threads per row:
// ss = fp32 sum of the bf16 outputs' squares, each thread's 4 values in order with fmaf, then a butterfly over the row's
// 16 threads (xor 8, 4, 2, 1); r = rsqrtf(ss / 64 + eps); y = bf16((x * r) * (1 / (1 + exp(-g)))) with __fdividef / __expf),
// and requests with a single piece are written here too.
constexpr int kMergeRows = 16;
template <int kPairs, bool kUsePDL, bool kLoc, int kMergeBatch = 8, bool kGate = false, bool kEarlyTrigger = false>
__global__ void __launch_bounds__(256) full_attn_wide_merge_kernel(const Params p, int grid_blocks) {
  constexpr int kHeads = 2 * kPairs, kTileKeys = 128 / kPairs, kRecRows = kHeads * kRows;
  constexpr int kDoms = kLoc ? 2 : 1;
  // per domain: tiles of requests < r, of requests <= r, of all requests
  __shared__ int32_t range[2][3];
  if constexpr (kUsePDL) asm volatile("griddepcontrol.wait;\n" ::: "memory");
  // kEarlyTrigger: the dependents (the output projection) may launch now and start their weight stream during the
  // merge; they read this kernel's output only after their own dependency wait. Otherwise they launch as blocks exit.
  if constexpr (kUsePDL && kEarlyTrigger) asm volatile("griddepcontrol.launch_dependents;\n" ::: "memory");
  const int R = p.num_reqs, r = blockIdx.y, tid = threadIdx.x, lane = tid & 31;
  if constexpr (kLoc) {
    // every rank of both domains was taken by exactly one block of the main kernel; then clear the bitmask
    if (tid == 0 && blockIdx.x == 0 && blockIdx.y == 0) {
      for (int d = 0; d < 2; ++d) {
        const int g = d ? p.blocks_dom[1] : p.blocks_dom[0];
        for (int w = 0; w < 8; ++w) {
          const int lo = 32 * w, n = max(0, min(32, g - lo));
          const uint32_t want = n == 32 ? 0xffffffffu : ((1u << n) - 1u);
          if (p.counters[d * 8 + w] != want) __trap();
          p.counters[d * 8 + w] = 0u;
        }
      }
    }
  }
  if (tid < 32 * kDoms) {
    const int d = tid >> 5;
    int n[kMaxReqs / 32];
#pragma unroll
    for (int k = 0; k < kMaxReqs / 32; ++k) {
      const int rr = lane + 32 * k;
      n[k] = rr < R ? (kLoc ? p.plan[plan_tiles_off(d, rr)] : p.seq_lens[rr]) : 0;
    }
    int before = 0, upto = 0, all = 0;
#pragma unroll
    for (int k = 0; k < kMaxReqs / 32; ++k) {
      const int rr = lane + 32 * k;
      const int t = rr < R ? (kLoc ? n[k] : tiles_of<kTileKeys>(n[k], p.window_left)) : 0;
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
      range[d][0] = before;
      range[d][1] = upto;
      range[d][2] = all;
    }
  }
  __syncthreads();
  // pieces of request r: per domain the blocks j0[d] .. j0[d] + cnt[d] - 1 of that domain's split, domain 0 first
  int j0[2] = {0, 0}, cnt[2] = {0, 0};
#pragma unroll
  for (int d = 0; d < kDoms; ++d) {
    const int W = range[d][2];
    if (range[d][1] == range[d][0]) continue;  // no tiles of r on this domain
    const int G = min(kLoc ? p.blocks_dom[d] : grid_blocks, W);
    j0[d] = block_of(range[d][0], G, W);
    cnt[d] = block_of(range[d][1] - 1, G, W) - j0[d] + 1;
  }
  const int npieces = cnt[0] + cnt[1];
  if (npieces <= 1 && !kGate) return;  // the request lies in one block: written by the main kernel
  const int b0 = 0, b1 = npieces - 1;  // piece indices
  const int row = kMergeRows * blockIdx.x + (tid >> 4), part = tid & 15;  // row of the request's records, 4-dim part
  const auto rec_of = [&](int q) {
    int64_t idx;
    if constexpr (kLoc)
      idx = q < cnt[0] ? static_cast<int64_t>(j0[0] + q + r)
                       : static_cast<int64_t>(grid_blocks + kMaxReqs) + j0[1] + (q - cnt[0]) + r;
    else
      idx = (kGate && npieces == 1) ? 2 * static_cast<int64_t>(grid_blocks) + r  // gate: a request inside one block
                         : static_cast<int64_t>(j0[0] + q) * 2 + (q == 0 ? 1 : 0);
    return p.partial + (idx * kRecRows + row) * kRec;
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
  // weighted sums in piece order (block order; localized: domain 0's pieces, then domain 1's)
  float acc[4] = {0.0f, 0.0f, 0.0f, 0.0f};
  float lsum = 0.0f;
  for (int qb = b0;;) {
#pragma unroll
    for (int u = 0; u < kMergeBatch; ++u) {
      if (qb + u > b1 || ml[u].x == -INFINITY) continue;
      const float w = fast_exp2(ml[u].x - mx);
      acc[0] += w * v[u].x;
      acc[1] += w * v[u].y;
      acc[2] += w * v[u].z;
      acc[3] += w * v[u].w;
      lsum += w * ml[u].y;
    }
    qb += kMergeBatch;
    if (qb > b1) break;
#pragma unroll
    for (int u = 0; u < kMergeBatch; ++u) {
      if (qb + u > b1) continue;
      const float* const rj = rec_of(qb + u);
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
  if constexpr (kGate) {
    const float2 x01 = __bfloat1622float2(h0), x23 = __bfloat1622float2(h1);
    float ss = fmaf(x01.x, x01.x, 0.0f);
    ss = fmaf(x01.y, x01.y, ss);
    ss = fmaf(x23.x, x23.x, ss);
    ss = fmaf(x23.y, x23.y, ss);
    const unsigned group = 0xffffu << (threadIdx.x & 16);  // the 16 threads of this row
#pragma unroll
    for (int mask = 8; mask >= 1; mask >>= 1) ss = ss + __shfl_xor_sync(group, ss, mask, 16);
    const float rs = rsqrtf(ss / static_cast<float>(kHeadDim) + p.norm_eps);
    const uint2 graw = *reinterpret_cast<const uint2*>(
        p.gate + (static_cast<int64_t>(r) * kTokens + tok) * p.gate_token_stride + qh * kHeadDim + 4 * part);
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
  uint2 w;
  w.x = *reinterpret_cast<uint32_t*>(&h0);
  w.y = *reinterpret_cast<uint32_t*>(&h1);
  *reinterpret_cast<uint2*>(orow) = w;
}

// ---------------------------------------------------------------------------------------------- localized plan
// One block (256 threads) per request r: the request's pages (page-table positions 0 .. np - 1) are split by the
// locality domain of their page number, d(page) = ((page >> page_shift) & 1) ^ domain_flip, into pos[d][r][...] in
// ascending position order (stream compaction: warp ballots + a prefix over the 8 warps), and tiles[d][r] = tiles of
// the request in list d (whole pages give 128 / kTileKeys tiles; the last page gives the tiles its keys reach).
// Spill (spill_from = 0 or 1, spill_q = fraction x 65536): the last round(n x fraction) of the request's pages on domain
// spill_from go to the other list instead (its blocks read them remotely), to even out per-block work when the domains
// have different SM counts; the lists stay in ascending position order.
template <int kPairs, bool kUsePDL>
__global__ void __launch_bounds__(256)
    full_attn_wide_plan_kernel(const int32_t* __restrict__ page_table, int64_t page_table_stride,
                               const int32_t* __restrict__ seq_lens, int32_t* __restrict__ plan, int plan_stride,
                               int page_shift, int domain_flip, int spill_from, int spill_q) {
  constexpr int kTileKeys = 128 / kPairs, kTpp = 128 / kTileKeys;
  __shared__ int32_t wcnt[4][8];  // per warp: virtual domain 0 / 1, true domain 0 / 1
  __shared__ int32_t carry[4];
  if constexpr (kUsePDL) asm volatile("griddepcontrol.wait;\n" ::: "memory");
  const int r = blockIdx.x, tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
  const int seq = max(seq_lens[r], 1), np = (seq + 127) / 128;
  const int32_t* const pt = page_table + static_cast<int64_t>(r) * page_table_stride;
  int32_t* const pos0 = plan + plan_pos_off(0, r, plan_stride);
  int32_t* const pos1 = plan + plan_pos_off(1, r, plan_stride);
  const auto dom_of = [&](int k) { return ((pt[k] >> page_shift) & 1) ^ domain_flip; };
  const uint32_t lt = (1u << lane) - 1u;
  // pages of the giving domain that move (pass 1 counts them when spilling)
  int keep = 1 << 30;  // ranks (within the giving domain) >= keep move to the other list
  if (spill_from == 0 || spill_from == 1) {
    if (tid == 0) carry[0] = 0;
    __syncthreads();
    int mine = 0;
    for (int k = tid; k < np; k += 256) mine += dom_of(k) == spill_from;
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) mine += __shfl_xor_sync(0xffffffffu, mine, o);
    if (lane == 0) atomicAdd(&carry[0], mine);
    __syncthreads();
    const int n = carry[0];
    keep = n - static_cast<int>((static_cast<int64_t>(n) * spill_q + 32768) >> 16);
    __syncthreads();
  }
  if (tid < 4) carry[tid] = 0;
  __syncthreads();
  for (int base = 0; base < np; base += 256) {
    const int k = base + tid;
    const bool valid = k < np;
    const int dt = valid ? dom_of(k) : 0;  // true domain
    const uint32_t t0 = __ballot_sync(0xffffffffu, valid && dt == 0), t1 = __ballot_sync(0xffffffffu, valid && dt == 1);
    if (lane == 0) {
      wcnt[2][warp] = __popc(t0);
      wcnt[3][warp] = __popc(t1);
    }
    __syncthreads();
    int tb = carry[2 + dt];  // rank of this page within its true domain
#pragma unroll
    for (int w = 0; w < 8; ++w)
      if (w < warp) tb += wcnt[2 + dt][w];
    tb += __popc((dt ? t1 : t0) & lt);
    const int d = (valid && dt == spill_from && tb >= keep) ? 1 - dt : dt;  // list (virtual domain)
    const uint32_t m1 = __ballot_sync(0xffffffffu, valid && d == 1), m0 = __ballot_sync(0xffffffffu, valid && d == 0);
    if (lane == 0) {
      wcnt[0][warp] = __popc(m0);
      wcnt[1][warp] = __popc(m1);
    }
    __syncthreads();
    int before0 = carry[0], before1 = carry[1], tot[4] = {0, 0, 0, 0};
#pragma unroll
    for (int w = 0; w < 8; ++w) {
      if (w < warp) {
        before0 += wcnt[0][w];
        before1 += wcnt[1][w];
      }
#pragma unroll
      for (int c = 0; c < 4; ++c) tot[c] += wcnt[c][w];
    }
    if (valid) {
      if (d == 0) pos0[before0 + __popc(m0 & lt)] = k;
      else pos1[before1 + __popc(m1 & lt)] = k;
    }
    __syncthreads();
    if (tid == 0) {
#pragma unroll
      for (int c = 0; c < 4; ++c) carry[c] += tot[c];
    }
    __syncthreads();
  }
  if (tid == 0) {
    // list of the last page: the true domain, or the other one when the last page moved (it is the giving domain's
    // highest-ranked page)
    const int dt = dom_of(np - 1);
    const int dl = (dt == spill_from && carry[2 + dt] - 1 >= keep) ? 1 - dt : dt;
    const int tl = (seq - 128 * (np - 1) + kTileKeys - 1) / kTileKeys;  // tiles of the last page (1 .. kTpp)
#pragma unroll
    for (int d = 0; d < 2; ++d) plan[plan_tiles_off(d, r)] = carry[d] * kTpp - (d == dl ? kTpp - tl : 0);
  }
  if constexpr (kUsePDL) asm volatile("griddepcontrol.launch_dependents;\n" ::: "memory");
}

// SM -> locality domain census for the localized variant: one block per SM (the caller forces it with dynamic shared
// memory and grid = SM count); thread 0 times dependent atomics on a word of domain 0's memory and of domain 1's memory
// (distinct 128-byte lines per block). out[3 b .. 3 b + 2] = smid, cycles on domain 0, cycles on domain 1.
__global__ void full_attn_wide_sm_probe_kernel(unsigned long long* buf0, unsigned long long* buf1, int32_t* out) {
  if (threadIdx.x != 0) return;
  unsigned long long* const a0 = buf0 + 16 * blockIdx.x;
  unsigned long long* const a1 = buf1 + 16 * blockIdx.x;
  long long t[2];
#pragma unroll
  for (int d = 0; d < 2; ++d) {
    unsigned long long* const a = d == 0 ? a0 : a1;
    unsigned long long v = atomicAdd(a, 0ull);
    const long long c0 = clock64();
    for (int i = 0; i < 32; ++i) v = atomicAdd(a, v >> 63);  // dependent chain, adds 0
    t[d] = clock64() - c0 + static_cast<long long>(v & 0ull);
  }
  out[3 * blockIdx.x] = static_cast<int32_t>(sm_id());
  out[3 * blockIdx.x + 1] = static_cast<int32_t>(min(t[0], 2000000000ll));
  out[3 * blockIdx.x + 2] = static_cast<int32_t>(min(t[1], 2000000000ll));
}

}  // namespace swat

// ---------------------------------------------------------------------------------------------- host
namespace swat_host {

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

}  // namespace swat_host

template <int kPairs, int kStages, bool kUsePDL, int kPoly, int kPf = 0, int kMode = 0, int kTpr = 2, bool kLoc = false,
          bool kWin = false, bool kPipe = false, bool kSkipRs = false, bool kLazy = false, int kPp = 0, int kDiet = 0,
          bool kGateW = false, bool kEarlyTrigger = false, int kC = 0, int kNgVec = 8>
struct SwaTKernel {
  // SWA-T: clusters of this instantiation that can be resident at once on the current device (runtime occupancy query
  // with this kernel's cluster size, block size and shared memory) -> out[0] (int32, host tensor). The op picks kC per
  // request count from it, so no device constant is built in.
  static void max_clusters(const tvm::ffi::TensorView out) {
    using namespace sglang::host;
    using L = swat::Layout<kPairs, kStages, kTpr>;
    static_assert(kC > 0, "SWA-T instantiations only");
    constexpr auto kernel =
        swat::full_attn_wide_kernel<kPairs, kStages, kUsePDL, kPoly, kPf, kMode, kTpr, kLoc, kWin, kPipe, kSkipRs, kLazy,
                                   kPp, kDiet, kGateW, kC, kNgVec>;
    constexpr int kSmem = L::kSmBytes + swat::SwatRecv<L::kHeads * swat::kRows, kC>::kBytes;
    RuntimeDeviceCheck(::cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, kSmem));
    if constexpr (kC > 8)
      RuntimeDeviceCheck(::cudaFuncSetAttribute(kernel, cudaFuncAttributeNonPortableClusterSizeAllowed, 1));
    cudaLaunchConfig_t cfg = {};
    cfg.gridDim = dim3(static_cast<unsigned>(kC), 1, 1);
    cfg.blockDim = dim3(L::kThreads, 1, 1);
    cfg.dynamicSmemBytes = kSmem;
    cudaLaunchAttribute attr[1];
    attr[0].id = cudaLaunchAttributeClusterDimension;
    attr[0].val.clusterDim.x = static_cast<unsigned>(kC);
    attr[0].val.clusterDim.y = 1;
    attr[0].val.clusterDim.z = 1;
    cfg.attrs = attr;
    cfg.numAttrs = 1;
    int n = 0;
    RuntimeDeviceCheck(::cudaOccupancyMaxActiveClusters(&n, kernel, &cfg));
    RuntimeCheck(out.dtype().code == DLDataTypeCode::kDLInt && out.dtype().bits == 32 && out.numel() >= 1,
                 "max_clusters: int32 host tensor");
    static_cast<int32_t*>(out.data_ptr())[0] = n;
  }

  static_assert(!(kDiet & 32) || (kPp == 0 && !kLazy && !kPipe && (kMode == 0 || kMode == 4)),
                "kDiet bit 5 (probabilities in tensor memory): shared hand-offs, no kPipe, production or timeline mode");
  static_assert(kDiet >= 0 && kDiet <= 63 && (kDiet & 8) == 0 &&
                    ((kDiet & 7) == 0 || (!kLazy && !kPipe && !kSkipRs && kPoly == 0)),
                "kDiet (bit 0 f16x2 exponentials, bit 1 unit combine, bit 2 one-wait output loads, bit 4 page-list loads in "
                "flight together): bits 0-2 not with kLazy / kPipe / kSkipRs / kPoly");
  static_assert(!(kLazy && (kPipe || kSkipRs)), "kLazy replaces the per-tile combine");
  static_assert(kPp >= 0 && kPp <= 2 && (kPp == 0 || (kPairs == 2 && kTpr == 1 && !kLazy && !kPipe)),
                "kPp (ping-pong KV-head pairs): 4 KV heads, one thread per row, not with kLazy / kPipe");
  // SWA-T with the QK prologue (kPf bits 16-17 = column class): takes the bf16 QKV projection output instead of e4m3 q and
  // stores this step's K / V rows into the pools itself (the fused QK-norm call's work)
  static void run_qk(const tvm::ffi::TensorView qkv,        // [R*8, >= (Hq + 2 Hkv) * 64] bf16, inner contiguous
                     const tvm::ffi::TensorView k_pool, const tvm::ffi::TensorView v_pool,  // [slots, Hkv*64] e4m3
                     const tvm::ffi::TensorView page_table, const tvm::ffi::TensorView seq_lens,
                     const tvm::ffi::TensorView out,        // [R*8, Hq, 64] bf16
                     const tvm::ffi::TensorView partial, const tvm::ffi::TensorView counters,
                     const tvm::ffi::TensorView positions,  // [R*8] int64
                     const tvm::ffi::TensorView cos_sin,    // [max pos, 64] fp32
                     const tvm::ffi::TensorView cache_loc,  // [R*8] int64 (pool rows of this step's tokens; 0 = skip)
                     double bmm1_scale, double bmm2_scale, int64_t window_left, const tvm::ffi::TensorView gate,
                     double norm_eps, double qk_eps) {
    using namespace sglang::host;
    using L = swat::Layout<kPairs, kStages, kTpr>;
    static_assert(kC > 0 && ((kPf >> 16) & 3) != 0, "run_qk: SWA-T with the QK prologue");
    RuntimeCheck(out.dtype().code == DLDataTypeCode::kDLBfloat && out.dtype().bits == 16 && out.dim() == 3 &&
                     out.size(2) == swat::kHeadDim && out.stride(2) == 1 && out.stride(1) == swat::kHeadDim &&
                     out.stride(0) == out.size(1) * swat::kHeadDim,
                 "out must be bf16 [tokens, Hq, 64], contiguous");
    const int64_t m = out.size(0), hq = out.size(1), reqs = m / swat::kTokens;
    const int64_t hkv = k_pool.size(1) / swat::kHeadDim;
    RuntimeCheck(m % swat::kTokens == 0 && hkv % L::kHeads == 0 && hkv * swat::kQPerKv == hq && reqs <= swat::kMaxReqs,
                 "head layout / request count");
    RuntimeCheck(k_pool.dtype().code == DLDataTypeCode::kDLFloat8_e4m3fn &&
                     v_pool.dtype().code == DLDataTypeCode::kDLFloat8_e4m3fn && k_pool.dim() == 2 && v_pool.dim() == 2 &&
                     k_pool.stride(1) == 1 && v_pool.stride(1) == 1 && k_pool.stride(0) == k_pool.size(1) &&
                     v_pool.stride(0) == v_pool.size(1) && k_pool.size(1) == v_pool.size(1),
                 "pools must be contiguous e4m3 [slots, Hkv*64]");
    RuntimeCheck(qkv.dtype().code == DLDataTypeCode::kDLBfloat && qkv.dtype().bits == 16 && qkv.dim() == 2 &&
                     qkv.size(0) == m && qkv.size(1) >= (hq + 2 * hkv) * swat::kHeadDim && qkv.stride(1) == 1 &&
                     qkv.stride(0) % 8 == 0 && reinterpret_cast<uintptr_t>(qkv.data_ptr()) % 16 == 0,
                 "qkv must be bf16 [tokens, >= (Hq + 2 Hkv) * 64], inner contiguous, 16-byte aligned rows");
    RuntimeCheck(positions.dim() == 1 && positions.size(0) == m && positions.stride(0) == 1 &&
                     positions.dtype().code == DLDataTypeCode::kDLInt && positions.dtype().bits == 64 &&
                     cache_loc.dim() == 1 && cache_loc.size(0) == m && cache_loc.stride(0) == 1 &&
                     cache_loc.dtype().code == DLDataTypeCode::kDLInt && cache_loc.dtype().bits == 64,
                 "positions / cache_loc: int64 [tokens], contiguous");
    RuntimeCheck(cos_sin.dim() == 2 && cos_sin.size(1) == swat::kHeadDim && cos_sin.stride(1) == 1 &&
                     cos_sin.stride(0) == swat::kHeadDim && cos_sin.dtype().code == DLDataTypeCode::kDLFloat &&
                     cos_sin.dtype().bits == 32,
                 "cos_sin: fp32 [positions, 64], contiguous");
    RuntimeCheck(seq_lens.dim() == 1 && seq_lens.size(0) == reqs && seq_lens.stride(0) == 1 &&
                     seq_lens.dtype().code == DLDataTypeCode::kDLInt && seq_lens.dtype().bits == 32,
                 "seq_lens [R] int32");
    RuntimeCheck(page_table.dim() == 2 && page_table.size(0) >= reqs && page_table.stride(1) == 1 &&
                     page_table.dtype().code == DLDataTypeCode::kDLInt && page_table.dtype().bits == 32,
                 "page_table [R, pages] int32");
    RuntimeCheck(partial.dtype().code == DLDataTypeCode::kDLFloat && partial.dtype().bits == 32 &&
                     counters.dtype().code == DLDataTypeCode::kDLInt && counters.dtype().bits == 32 &&
                     counters.numel() >= swat::kMaxReqs,
                 "workspace");
    RuntimeCheck(window_left >= 0, "window_left >= 0");
    swat::Params params{};
    params.page_table = static_cast<const int32_t*>(page_table.data_ptr());
    params.seq_lens = static_cast<const int32_t*>(seq_lens.data_ptr());
    params.partial = static_cast<float*>(partial.data_ptr());
    params.counters = static_cast<uint32_t*>(counters.data_ptr());
    params.out = static_cast<__nv_bfloat16*>(out.data_ptr());
    params.page_table_stride = page_table.stride(0);
    params.num_reqs = static_cast<int32_t>(reqs);
    params.num_q_heads = static_cast<int32_t>(hq);
    params.scale_log2 = static_cast<float>(bmm1_scale * M_LOG2E);
    params.out_scale = static_cast<float>(bmm2_scale);
    params.window_left = static_cast<int32_t>(window_left);
    RuntimeCheck(kGateW == (gate.numel() > 1), "norm + gate: the module variant (kGateW) must match the gate argument");
    if constexpr (kGateW) {
      RuntimeCheck(gate.dtype().code == DLDataTypeCode::kDLBfloat && gate.dtype().bits == 16 && gate.dim() == 2 &&
                       gate.size(0) == m && gate.size(1) >= hq * swat::kHeadDim && gate.stride(1) == 1 &&
                       gate.stride(0) % 4 == 0 && reinterpret_cast<uintptr_t>(gate.data_ptr()) % 8 == 0,
                   "gate must be bf16 [tokens, >= Hq*64], inner contiguous, 8-byte aligned rows");
      params.gate = static_cast<const __nv_bfloat16*>(gate.data_ptr());
      params.gate_token_stride = gate.stride(0);
      params.norm_eps = static_cast<float>(norm_eps);
    }
    params.qkv = static_cast<const __nv_bfloat16*>(qkv.data_ptr());
    params.qkv_stride = qkv.stride(0);
    params.positions = static_cast<const int64_t*>(positions.data_ptr());
    params.cos_sin = static_cast<const float*>(cos_sin.data_ptr());
    params.cache_loc = static_cast<const int64_t*>(cache_loc.data_ptr());
    params.qk_eps = static_cast<float>(qk_eps);
    params.num_kv_heads = static_cast<int32_t>(hkv);
    params.pool_row_bytes = k_pool.size(1);
    params.k_store = static_cast<uint8_t*>(k_pool.data_ptr());
    params.v_store = static_cast<uint8_t*>(v_pool.data_ptr());
    const CUtensorMap qm = {};  // unused: the prologue writes the q rows
    const CUtensorMap km = swat_host::pool_map(k_pool.data_ptr(), k_pool.size(0), k_pool.size(1), L::kTileKeys);
    const CUtensorMap vm = swat_host::pool_map(v_pool.data_ptr(), v_pool.size(0), v_pool.size(1), L::kTileKeys);
    constexpr auto kernel =
        swat::full_attn_wide_kernel<kPairs, kStages, kUsePDL, kPoly, kPf, kMode, kTpr, kLoc, kWin, kPipe, kSkipRs, kLazy,
                                   kPp, kDiet, kGateW, kC, kNgVec>;
    constexpr int kSmem = L::kSmBytes + swat::SwatRecv<L::kHeads * swat::kRows, kC>::kBytes;
    static bool configured = false;
    if (!configured) {
      RuntimeDeviceCheck(::cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, kSmem));
      if constexpr (kC > 8)
        RuntimeDeviceCheck(::cudaFuncSetAttribute(kernel, cudaFuncAttributeNonPortableClusterSizeAllowed, 1));
      configured = true;
    }
    if (reqs == 0) return;
    static_assert(!kLoc && kWin && (kMode == 0 || kMode == 4), "SWA-T: windowed, plain, production mode (4: stamps)");
    LaunchKernel(dim3(static_cast<unsigned>(kC), static_cast<unsigned>(reqs), static_cast<unsigned>(hkv / L::kHeads)),
                 L::kThreads, out.device(), kSmem)
        .config({.use_pdl = kUsePDL, .cluster_dim = dim3(static_cast<unsigned>(kC), 1, 1)})(kernel, params, qm, km, vm);
  }

  static void run(const tvm::ffi::TensorView q,           // [R*8, Hq, 64] e4m3, heads/dims contiguous
                  const tvm::ffi::TensorView k_pool,      // [slots, Hkv*64] e4m3, rows contiguous
                  const tvm::ffi::TensorView v_pool,      // [slots, Hkv*64]
                  const tvm::ffi::TensorView page_table,  // [R, pages] int32, inner contiguous
                  const tvm::ffi::TensorView seq_lens,    // [R] int32
                  const tvm::ffi::TensorView out,         // [R*8, Hq, 64] bf16, contiguous
                  const tvm::ffi::TensorView partial,     // fp32 workspace, >= blocks * 2 * Hkv * 64 * 68
                  const tvm::ffi::TensorView counters,    // int32 [>= 256], zero
                  int64_t blocks, double bmm1_scale, double bmm2_scale,
                  // localized variant only (kLoc); ignored otherwise
                  const tvm::ffi::TensorView plan,        // int32, full_attn_wide_plan output for this page table
                  const tvm::ffi::TensorView sm_tab,      // int32 [SM count]: domain << 16 | rank in domain
                  int64_t plan_stride, int64_t page_shift, int64_t domain_flip, int64_t blocks_dom0,
                  int64_t blocks_dom1, int64_t window_left,
                  const tvm::ffi::TensorView gate,  // output norm + gate in the merge kernel: [R*8, >= Hq*64] bf16 (1 element: off)
                  double norm_eps) {
    using namespace sglang::host;
    using L = swat::Layout<kPairs, kStages, kTpr>;
    RuntimeCheck(q.dim() == 3 && q.size(2) == swat::kHeadDim && q.dtype().code == DLDataTypeCode::kDLFloat8_e4m3fn &&
                     q.stride(2) == 1 && q.stride(1) == swat::kHeadDim && q.stride(0) % 16 == 0,
                 "q must be e4m3 [tokens, heads, 64], heads and dims contiguous");
    RuntimeCheck(k_pool.dtype().code == DLDataTypeCode::kDLFloat8_e4m3fn &&
                     v_pool.dtype().code == DLDataTypeCode::kDLFloat8_e4m3fn && k_pool.dim() == 2 && v_pool.dim() == 2 &&
                     k_pool.stride(1) == 1 && v_pool.stride(1) == 1 && k_pool.stride(0) == k_pool.size(1) &&
                     v_pool.stride(0) == v_pool.size(1) && k_pool.size(1) == v_pool.size(1),
                 "pools must be contiguous e4m3 [slots, Hkv*64]");
    RuntimeCheck(out.dtype().code == DLDataTypeCode::kDLBfloat && out.dtype().bits == 16, "out must be bf16");
    const int64_t m = out.size(0), hq = out.size(1), reqs = m / swat::kTokens;
    const int64_t hkv = k_pool.size(1) / swat::kHeadDim;
    RuntimeCheck(m % swat::kTokens == 0 && (kC > 0 ? hkv % L::kHeads == 0 : hkv == L::kHeads) && hkv * swat::kQPerKv == hq &&
                     q.size(0) == m &&
                     q.size(1) == hq,
                 "head layout (KV heads per rank must match the kernel)");
    RuntimeCheck(reqs <= swat::kMaxReqs, "too many requests");
    RuntimeCheck(out.dim() == 3 && out.size(2) == swat::kHeadDim && out.stride(2) == 1 && out.stride(1) == swat::kHeadDim &&
                     out.stride(0) == hq * swat::kHeadDim,
                 "out layout");
    RuntimeCheck(seq_lens.dim() == 1 && seq_lens.size(0) == reqs && seq_lens.stride(0) == 1 &&
                     seq_lens.dtype().code == DLDataTypeCode::kDLInt && seq_lens.dtype().bits == 32,
                 "seq_lens [R] int32");
    RuntimeCheck(page_table.dim() == 2 && page_table.size(0) >= reqs && page_table.stride(1) == 1 &&
                     page_table.dtype().code == DLDataTypeCode::kDLInt && page_table.dtype().bits == 32,
                 "page_table [R, pages] int32");
    const int64_t records = kLoc ? 2 * (blocks + swat::kMaxReqs) : 2 * blocks;
    RuntimeCheck(blocks >= 1 && partial.dtype().code == DLDataTypeCode::kDLFloat && partial.dtype().bits == 32 &&
                     partial.numel() >= records * hkv * swat::kRows * swat::kRec,
                 "partial workspace too small");
    if constexpr (kLoc) {
      RuntimeCheck(blocks_dom0 >= 1 && blocks_dom1 >= 1 && blocks_dom0 + blocks_dom1 == blocks && blocks_dom0 <= 256 &&
                       blocks_dom1 <= 256,
                   "localized: blocks must equal the ranks (blocks_dom0 + blocks_dom1, at most 256 per domain)");
      RuntimeCheck(sm_tab.dtype().code == DLDataTypeCode::kDLInt && sm_tab.dtype().bits == 32 && sm_tab.numel() >= blocks,
                   "localized: sm_tab int32 [SM count]");
      RuntimeCheck(plan.dtype().code == DLDataTypeCode::kDLInt && plan.dtype().bits == 32 &&
                       plan_stride >= page_table.size(1) &&
                       plan.numel() >= 2 * swat::kMaxReqs + 2 * swat::kMaxReqs * plan_stride,
                   "localized: plan int32 [2 * 256 + 2 * 256 * plan_stride], plan_stride >= page-table width");
    }
    RuntimeCheck(counters.dtype().code == DLDataTypeCode::kDLInt && counters.dtype().bits == 32 &&
                     counters.numel() >= swat::kMaxReqs,
                 "counters int32 [>= 256]");
    const auto aligned16 = [](const void* ptr) { return reinterpret_cast<uintptr_t>(ptr) % 16 == 0; };
    RuntimeCheck(aligned16(q.data_ptr()) && aligned16(k_pool.data_ptr()) && aligned16(v_pool.data_ptr()) &&
                     aligned16(out.data_ptr()) && aligned16(partial.data_ptr()),
                 "16-byte alignment");

    swat::Params params{};
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
    RuntimeCheck(kWin ? (window_left >= 0 && !kLoc) : window_left == -1,
                 "window_left: >= 0 in the windowed variant (not localized), -1 otherwise");
    params.window_left = static_cast<int32_t>(window_left);
    RuntimeCheck(kGateW == (gate.numel() > 1), "norm + gate: the module variant (kGateW) must match the gate argument");
    if constexpr (kGateW) {
      RuntimeCheck(!kLoc && kMode == 0, "norm + gate: plain variant, production mode");
      if constexpr (kC == 0)  // SWA-T keeps its records in shared memory (no per-request global records)
        RuntimeCheck(partial.numel() >= (2 * blocks + swat::kMaxReqs) * hkv * swat::kRows * swat::kRec,
                     "norm + gate: workspace needs 2 * blocks + 256 records");
      RuntimeCheck(gate.dtype().code == DLDataTypeCode::kDLBfloat && gate.dtype().bits == 16 && gate.dim() == 2 &&
                       gate.size(0) == m && gate.size(1) >= hq * swat::kHeadDim && gate.stride(1) == 1 &&
                       gate.stride(0) % 4 == 0 && reinterpret_cast<uintptr_t>(gate.data_ptr()) % 8 == 0,
                   "gate must be bf16 [tokens, >= Hq*64], inner contiguous, 8-byte aligned rows");
      params.gate = static_cast<const __nv_bfloat16*>(gate.data_ptr());
      params.gate_token_stride = gate.stride(0);
      params.norm_eps = static_cast<float>(norm_eps);
    }
    if constexpr (kC > 0 && (kPf & 1024) != 0) {  // SWA-T: `plan` carries the tensor to prefetch into L2 (1 element: none)
      if (plan.numel() > 1) {
        params.l2pf = static_cast<const char*>(plan.data_ptr());
        params.l2pf_bytes = static_cast<int64_t>(plan.numel()) * (plan.dtype().bits / 8);
      }
    }
    if constexpr (kLoc) {
      params.plan = static_cast<const int32_t*>(plan.data_ptr());
      params.sm_tab = static_cast<const int32_t*>(sm_tab.data_ptr());
      params.plan_stride = static_cast<int32_t>(plan_stride);
      params.page_shift = static_cast<int32_t>(page_shift);
      params.domain_flip = static_cast<int32_t>(domain_flip);
      params.blocks_dom[0] = static_cast<int32_t>(blocks_dom0);
      params.blocks_dom[1] = static_cast<int32_t>(blocks_dom1);
    }
    const CUtensorMap qm = swat_host::q_map(q.data_ptr(), m, hq, q.stride(0));
    const CUtensorMap km = swat_host::pool_map(k_pool.data_ptr(), k_pool.size(0), k_pool.size(1), L::kTileKeys);
    const CUtensorMap vm = swat_host::pool_map(v_pool.data_ptr(), v_pool.size(0), v_pool.size(1), L::kTileKeys);

    static_assert(kPoly >= 0 && kPoly <= 8 && kPoly % 2 == 0, "kPoly: even, 0..8");
    constexpr auto kernel =
        swat::full_attn_wide_kernel<kPairs, kStages, kUsePDL, kPoly, kPf, kMode, kTpr, kLoc, kWin, kPipe, kSkipRs, kLazy,
                                   kPp, kDiet, kGateW, kC, kNgVec>;
    constexpr int kSmem = L::kSmBytes + swat::SwatRecv<L::kHeads * swat::kRows, kC>::kBytes;
    static bool configured = false;
    if (!configured) {
      RuntimeDeviceCheck(::cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, kSmem));
      if constexpr (kC > 8)
        RuntimeDeviceCheck(::cudaFuncSetAttribute(kernel, cudaFuncAttributeNonPortableClusterSizeAllowed, 1));
      configured = true;
    }
    if (reqs == 0) return;
    if constexpr (kC > 0) {  // SWA-T: grid (kC, R), cluster (kC, 1, 1); the owners merge in-cluster (no merge kernel)
      static_assert(!kLoc && kWin && (kMode == 0 || kMode == 4), "SWA-T: windowed, plain, production mode (4: stamps)");
      LaunchKernel(dim3(static_cast<unsigned>(kC), static_cast<unsigned>(reqs), static_cast<unsigned>(hkv / L::kHeads)),
                   L::kThreads, out.device(), kSmem)
          .config({.use_pdl = kUsePDL, .cluster_dim = dim3(static_cast<unsigned>(kC), 1, 1)})(kernel, params, qm, km, vm);
      return;
    }
    LaunchKernel(dim3(static_cast<unsigned>(blocks)), L::kThreads, out.device(), L::kSmBytes)
        .config({.use_pdl = kUsePDL})(kernel, params, qm, km, vm);
    // localized: the merge also checks and re-arms the placement guard, so it runs in every mode
    if constexpr (kMode == 0 || kMode == 4 || kMode == 6 || kLoc) {
      const dim3 mgrid(static_cast<unsigned>(L::kHeads * swat::kRows / swat::kMergeRows), static_cast<unsigned>(reqs));
      constexpr bool gated = kGateW;
      const bool wide_batch = kPairs == 1 && !kLoc && reqs * 8 < blocks;  // a request may span more than 8 blocks
      if constexpr (gated) {
        if constexpr (!kLoc) {
          if (wide_batch)
            LaunchKernel(mgrid, 256, out.device(), 0)
                .config({.use_pdl = kUsePDL})(swat::full_attn_wide_merge_kernel<kPairs, kUsePDL, false, 24, true, kEarlyTrigger>, params,
                                              static_cast<int>(blocks));
          else
            LaunchKernel(mgrid, 256, out.device(), 0)
                .config({.use_pdl = kUsePDL})(swat::full_attn_wide_merge_kernel<kPairs, kUsePDL, false, 8, true, kEarlyTrigger>, params,
                                              static_cast<int>(blocks));
        }
      } else if (wide_batch) {
        LaunchKernel(mgrid, 256, out.device(), 0)
            .config({.use_pdl = kUsePDL})(swat::full_attn_wide_merge_kernel<kPairs, kUsePDL, kLoc, 24, false, kEarlyTrigger>, params,
                                          static_cast<int>(blocks));
      } else {
        LaunchKernel(mgrid, 256, out.device(), 0)
            .config({.use_pdl = kUsePDL})(swat::full_attn_wide_merge_kernel<kPairs, kUsePDL, kLoc, 8, false, kEarlyTrigger>, params,
                                          static_cast<int>(blocks));
      }
    }
  }
};


template <int kPairs, bool kUsePDL>
struct SwaTPlan {
  static void run(const tvm::ffi::TensorView page_table,  // [>= R, pages] int32, inner contiguous
                  const tvm::ffi::TensorView seq_lens,    // [R] int32
                  const tvm::ffi::TensorView plan,        // int32 [2 * 256 + 2 * 256 * plan_stride]
                  int64_t plan_stride, int64_t page_shift, int64_t domain_flip, int64_t spill_from, int64_t spill_q) {
    using namespace sglang::host;
    const int64_t reqs = seq_lens.size(0);
    RuntimeCheck(reqs <= swat::kMaxReqs && page_table.dim() == 2 && page_table.size(0) >= reqs &&
                     page_table.stride(1) == 1 && plan_stride >= page_table.size(1) &&
                     plan.numel() >= 2 * swat::kMaxReqs + 2 * swat::kMaxReqs * plan_stride,
                 "full_attn_wide_plan: shapes");
    if (reqs == 0) return;
    LaunchKernel(dim3(static_cast<unsigned>(reqs)), 256, page_table.device(), 0)
        .config({.use_pdl = kUsePDL})(swat::full_attn_wide_plan_kernel<kPairs, kUsePDL>,
                                      static_cast<const int32_t*>(page_table.data_ptr()), page_table.stride(0),
                                      static_cast<const int32_t*>(seq_lens.data_ptr()),
                                      static_cast<int32_t*>(plan.data_ptr()), static_cast<int>(plan_stride),
                                      static_cast<int>(page_shift), static_cast<int>(domain_flip),
                                      static_cast<int>(spill_from), static_cast<int>(spill_q));
  }
};

struct SwaTSmProbe {
  static void run(const tvm::ffi::TensorView buf0, const tvm::ffi::TensorView buf1, const tvm::ffi::TensorView out,
                  int64_t blocks, int64_t smem_bytes) {
    using namespace sglang::host;
    RuntimeCheck(buf0.numel() * buf0.dtype().bits / 8 >= blocks * 128 && buf1.numel() * buf1.dtype().bits / 8 >= blocks * 128 &&
                     out.numel() >= 3 * blocks,
                 "full_attn_wide_sm_probe: sizes");
    RuntimeDeviceCheck(::cudaFuncSetAttribute(swat::full_attn_wide_sm_probe_kernel,
                                              cudaFuncAttributeMaxDynamicSharedMemorySize, static_cast<int>(smem_bytes)));
    LaunchKernel(dim3(static_cast<unsigned>(blocks)), 32, out.device(), static_cast<size_t>(smem_bytes))(
        swat::full_attn_wide_sm_probe_kernel, static_cast<unsigned long long*>(buf0.data_ptr()),
        static_cast<unsigned long long*>(buf1.data_ptr()), static_cast<int32_t*>(out.data_ptr()));
  }
};

// Accuracy probe of the kernel's two exp2 forms (tests only): out_hw = ex2.approx.ftz.f32(y), out_poly = exp2_poly2(y)
// (the FMA-pipe polynomial the kPoly option uses), for every y of a float32 tensor.
namespace swat {
__global__ void exp2_probe_kernel(const float* __restrict__ y, float* __restrict__ hw, float* __restrict__ poly, int64_t n) {
  const int64_t i = (static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x) * 2;
  if (i + 1 < n) {
    const float2 v = make_float2(y[i], y[i + 1]);
    const float2 p = exp2_poly2(v);
    hw[i] = fast_exp2(v.x);
    hw[i + 1] = fast_exp2(v.y);
    poly[i] = p.x;
    poly[i + 1] = p.y;
  }
}
}  // namespace swat
struct SwaTExp2Probe {
  static void run(const tvm::ffi::TensorView y, const tvm::ffi::TensorView hw, const tvm::ffi::TensorView poly) {
    using namespace sglang::host;
    const int64_t n = y.numel();
    RuntimeCheck(n % 2 == 0 && hw.numel() == n && poly.numel() == n && y.dtype().code == DLDataTypeCode::kDLFloat &&
                     y.dtype().bits == 32,
                 "exp2 probe: float32 tensors of one even length");
    const unsigned blocks = static_cast<unsigned>((n / 2 + 255) / 256);
    LaunchKernel(dim3(blocks), 256, y.device())(swat::exp2_probe_kernel, static_cast<const float*>(y.data_ptr()),
                                                 static_cast<float*>(hw.data_ptr()), static_cast<float*>(poly.data_ptr()), n);
  }
};

}  // namespace muse_vr
