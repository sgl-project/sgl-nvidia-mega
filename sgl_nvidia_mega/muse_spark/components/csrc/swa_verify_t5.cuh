// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
/**
 * \brief Sliding-window paged attention for speculative verify on the 5th-generation tensor core (tcgen05):
 * 8 query tokens per request, FP8 (e4m3) Q/K/V, bf16 output. Same shapes, masking and rounding points as
 * swa_verify_attn.cuh (see there); a different work split and MMA path.
 *
 * Arithmetic, in order (the rounding points of swa_verify_attn and of the stock trtllm-gen kernel):
 *   s   = fp32 dot product of e4m3 q and e4m3 k (tcgen05.mma .kind::f8f6f4, fp32 accumulation in tensor memory)
 *   m   = running maximum of x = s * scale_log2 over the keys this thread has seen (fp32; each key half of a row
 *         keeps its own; scale_log2 = fp32(bmm1_scale * log2(e)), host side; the maximum is taken over s, scaled once)
 *   p'  = exp2(fma(s, scale_log2, log2(448) - m)) (ex2.approx.ftz) = 448 * exp2(x - m) up to rounding;
 *   pq  = e4m3(p') (round to nearest even, satfinite)
 *   o   = o * exp2(m_old - m) + sum of pq * v over the page (tcgen05.mma, fp32 accumulation in tensor memory; the
 *         rescale is a tensor-memory read / multiply / write by the row's threads before the page's MMA)
 *   l   = l * exp2(m_old - m) + sum of p' over the page (fp32; each key half of a row sums in a fixed tree order)
 *   the two key halves of a row merge with max-rescaling (fp32) into the block's partial (o, m, l)
 *   partial (o, m, l) per block -> global workspace; the last block of a (request, KV head) merges the partials in
 *   split order with max-rescaling (fp32) and writes out = bf16(o * bmm2_scale / l)
 *
 * Work split: grid (S, Hkv, R), no clusters. Block s serves request r, KV head h and the s-th run of whole KV pages
 * (page size 128) of the window [max(0, seq_len - 8 - window_left), seq_len). The A operand of every MMA is [Q; Q]:
 * the 64 query rows of (r, h) (row = 8 * token + head-in-group) in tensor-memory lanes 0..63 and again in 64..127.
 * Lane L = 64 h + r takes key half h of each page (64 keys) for row r: S and PV are issued as two lane-masked halves
 * (tcgen05.mma disable-output-lane: keys 0..63 update lanes 0..63 only, keys 64..127 lanes 64..127 only), so a page's
 * scores take 64 tensor-memory columns and its p 16; O lanes r and 64 + r hold the two key-half sums of row r (merged
 * in the epilogue).
 *
 * Roles (8 warps):
 *   warps 0-3  softmax, one tensor-memory lane quarter each (all 4 SM sub-partitions): load the q row into tensor
 *              memory; per page: read the half's 64 scores, mask, tree max (the half's own running maximum), exp2,
 *              tree sum, quantize p into tensor memory, rescale the lane's O when the max grew.
 *   warps 4-7  copy the block's K / V pages into shared memory (cp.async 16 bytes, zero-filled outside the window),
 *              one commit group per page; warp 7 also issues the MMAs (one elected lane): S of page j as soon as its K
 *              is in and page j-1's scores are in registers, then PV of page j-1.
 *   With kPrefetch, warps 4-7 first issue L2 prefetch hints for the block's K / V rows before griddepcontrol.wait;
 *   the seq_lens / page-table values read for them only form bounds-checked prefetch addresses.
 *
 * Shared-memory operand layouts (PTX ISA, tcgen05 canonical layouts, no swizzling, 16-byte core-matrix rows):
 *   K page (B of S = Q K^T, K-major, N = keys): key i, bytes 16c..16c+15 at (i / 8) * 512 + c * 128 + (i % 8) * 16
 *     -> leading byte offset 128 (next 16 bytes of K), stride byte offset 512 (next 8 keys).
 *   V page (B of O = P V, MN-major = transposed B, N = head dims, K = keys): the same byte placement read as
 *     ((16, 1, 4), (8, 16)) : ((1, 16, 128), (16, 512)) -> leading byte offset 512 (next 8 keys), stride byte offset
 *     128 (next 16 dims).
 * Tensor-memory columns: S [0, 64), O [64, 128), P [128, 144) (4 e4m3 per 32-bit column, keys in order),
 * Q [144, 160) (4 e4m3 per column, dims in order).
 *
 * Ordering (memory-model manifest in the private notes): every global read whose value is used happens after
 * griddepcontrol.wait.
 * cp.async writes -> cp.async.wait_group -> fence.proxy.async.shared::cta -> mbarrier arrive (release) -> wait
 * (acquire) by the issuing warp -> MMA. tcgen05.st -> tcgen05.wait::st -> tcgen05.fence::before_thread_sync ->
 * mbarrier arrive -> wait -> tcgen05.fence::after_thread_sync -> MMA. MMA -> tcgen05.commit -> wait ->
 * tcgen05.fence::after_thread_sync -> tcgen05.ld; a page's scores are read (tcgen05.wait::ld) before the softmax
 * threads arrive on the barrier that lets the next page's S overwrite them. Partials: each block counts its arrival
 * first (atom.relaxed.gpu on the pair's counter); a block that is not last stores its record (st.global) -> bar.sync
 * -> one thread: red.release.gpu on the pair's ready counter; the last block's thread 0 polls the ready counter with
 * ld.acquire.gpu until S - 1 -> bar.sync -> ld.global.cg of the other records (its own stays in shared memory) ->
 * merge -> output; it resets both counters to 0 for the next launch (launches on the stream are ordered).
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
#include <cstdlib>
#include <cstdint>

namespace muse_vr {

namespace swa_t5 {

constexpr int kHeadDim = 64;
constexpr int kQPerKv = 8;
constexpr int kTokens = 8;
constexpr int kRows = 64;           // real query rows per (request, KV head)
constexpr int kPage = 128;          // keys per page = keys per MMA tile
constexpr int kHalf = 64;           // keys per softmax thread per page
constexpr int kThreads = 256;
constexpr int kPageBytes = 2 * kPage * kHeadDim;  // K then V, 16 KiB
constexpr int kRecFloats = 68;      // 64 outputs, m, l (key half 0), l (key half 1), pad
constexpr float kPScale = 448.0f;
constexpr float kLog2PScale = 8.807354922057604f;  // log2(448)
constexpr uint32_t kTmemCols = 256;
constexpr uint32_t kColS = 0, kColO = 64, kColP = 128, kColQ = 144;
constexpr int kMaxSplits = 16;
// instruction descriptors (PTX ISA, tcgen05 instruction descriptor, .kind::f8f6f4): fp32 D (bit 4), e4m3 A and B
// (format 0), M = 128 (bits 24-28 = M >> 4), N at bits 17-22 (= N >> 3); PV transposes B (bit 16)
constexpr uint32_t kIdescS = (1u << 4) | (uint32_t(kHalf >> 3) << 17) | (uint32_t(128 >> 4) << 24);
constexpr uint32_t kIdescPV = (1u << 4) | (1u << 16) | (uint32_t(kHeadDim >> 3) << 17) | (uint32_t(128 >> 4) << 24);

struct Params {
  const uint8_t* __restrict__ q;
  const uint8_t* __restrict__ k_pool;
  const uint8_t* __restrict__ v_pool;
  const int32_t* __restrict__ page_table;
  const int32_t* __restrict__ seq_lens;
  __nv_bfloat16* __restrict__ out;
  float* __restrict__ part;      // [R * Hkv * S, 64, kRecFloats] fp32 partial records
  uint32_t* __restrict__ count;  // [2 * R * Hkv]: arrivals, then published records (0 between launches)
  int64_t pool_row_stride;  // bytes between consecutive token slots of the pools
  int64_t pool_slots;       // token slots in the pools (bounds the prefetch hints)
  int32_t pt_cols;          // page-table columns (bounds the prefetch hints)
  int64_t page_table_stride;
  int64_t q_token_stride;   // bytes between consecutive query tokens
  int32_t num_q_heads;
  int32_t num_kv_heads;
  int32_t window_left;
  uint32_t test_sender_delay_ns;  // tests only: cluster block 1 waits this long before its pushes (0 = off)
  int32_t test_pf_seq_force;      // tests only: the kernel-top page-table prefetch uses test_pf_seq_value as seq_lens
  int32_t test_pf_seq_value;
  float scale_log2;
  float out_scale;  // bmm2_scale (l sums p * 448, so the 448 of pq cancels)
};

template <int kPages>
struct Cfg {
  // the page region also holds the half-1 sums (64 x 65 floats at 0) and the block's record (at kRecS) at the end
  static constexpr int kRecS = 16896;
  static constexpr int kTiles = kPages * kPageBytes > kRecS + kRows * kRecFloats * 4 ? kPages * kPageBytes
                                                                                     : kRecS + kRows * kRecFloats * 4;
  static constexpr int kCtl = kTiles;  // barriers, half maxima, tensor-memory address, last flag
  static constexpr int kBars = 5 + kPages;
  static constexpr int kMx = kCtl + 8 * kBars;      // float [2 parities][2 halves][64 rows]
  static constexpr int kSlot = kMx + 4 * 2 * 2 * kRows;
  static constexpr int kBytes = kSlot + 16;
  static_assert(kPages >= 1 && kPages <= 6, "pages per block");
};

// Upper bound on try_wait polls of one mbarrier wait (a wait that can only hang through a defect traps instead).
inline constexpr uint32_t kMaxPolls = 1u << 24;
__device__ __noinline__ void t5_wait_bound_exceeded() { __trap(); }

__device__ __forceinline__ uint32_t su(const void* ptr) {
  return static_cast<uint32_t>(__cvta_generic_to_shared(ptr));
}
__device__ __forceinline__ void mbar_init(uint32_t bar, uint32_t count) {
  asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;\n" ::"r"(bar), "r"(count) : "memory");
}
__device__ __forceinline__ void mbar_arrive(uint32_t bar) {
  asm volatile("mbarrier.arrive.release.cta.shared::cta.b64 _, [%0];\n" ::"r"(bar) : "memory");
}
__device__ __forceinline__ void mbar_wait(uint32_t bar, uint32_t parity) {
  uint32_t polls = 0;
  while (true) {
    uint32_t done;
    asm volatile(
        "{\n.reg .pred p;\nmbarrier.try_wait.parity.acquire.cta.shared::cta.b64 p, [%1], %2;\nselp.u32 %0, 1, 0, p;\n}\n"
        : "=r"(done)
        : "r"(bar), "r"(parity)
        : "memory");
    if (done) break;
    if (++polls == kMaxPolls) [[unlikely]] t5_wait_bound_exceeded();
  }
}
__device__ __forceinline__ void cp_async16(uint32_t dst, const void* src, uint32_t src_bytes) {
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" ::"r"(dst), "l"(src), "r"(src_bytes) : "memory");
}
__device__ __forceinline__ void cp_commit() { asm volatile("cp.async.commit_group;\n" ::: "memory"); }
template <int N>
__device__ __forceinline__ void cp_wait() {
  asm volatile("cp.async.wait_group %0;\n" ::"n"(N) : "memory");
}
// wait until at most n of this thread's copy groups are pending (n is a constant once the caller's loop is unrolled)
__device__ __forceinline__ void cp_wait_n(int n) {
  switch (n) {
    case 0: cp_wait<0>(); break;
    case 1: cp_wait<1>(); break;
    case 2: cp_wait<2>(); break;
    case 3: cp_wait<3>(); break;
    case 4: cp_wait<4>(); break;
    default: cp_wait<5>(); break;
  }
}
__device__ __forceinline__ void fence_proxy_async() {
  asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory");
}
__device__ __forceinline__ void tc_fence_before() {
  asm volatile("tcgen05.fence::before_thread_sync;\n" ::: "memory");
}
__device__ __forceinline__ void tc_fence_after() {
  asm volatile("tcgen05.fence::after_thread_sync;\n" ::: "memory");
}
__device__ __forceinline__ void tc_wait_ld() { asm volatile("tcgen05.wait::ld.sync.aligned;\n" ::: "memory"); }
__device__ __forceinline__ void tc_wait_st() { asm volatile("tcgen05.wait::st.sync.aligned;\n" ::: "memory"); }
__device__ __forceinline__ void softmax_sync() { asm volatile("bar.sync 1, 128;\n" ::: "memory"); }

// shared-memory matrix descriptor (PTX ISA, tcgen05 shared memory descriptor): start address, leading / stride byte
// offsets (16-byte units), fixed 0b001 at bits 46-48, base offset 0, relative leading offset, no swizzling
__device__ __forceinline__ uint64_t sdesc(uint32_t saddr, uint32_t lbo, uint32_t sbo) {
  return static_cast<uint64_t>((saddr >> 4) & 0x3FFFu) | (static_cast<uint64_t>((lbo >> 4) & 0x3FFFu) << 16) |
         (static_cast<uint64_t>((sbo >> 4) & 0x3FFFu) << 32) | (1ull << 46);
}

// S = Q K^T for one page as two lane-masked N = 64 halves into the same 64 columns: keys 0..63 update lanes 0..63 only,
// keys 64..127 lanes 64..127 only (disable-output-lane); each half is two K32 MMAs (dims 0..31, 32..63). A = q in
// tensor memory ([Q; Q]), B = the K page (keys 64..127 start 4096 bytes in).
__device__ __forceinline__ void mma_s(uint32_t d, uint32_t a, uint64_t b) {
  asm volatile(
      "{\n\t.reg .pred e, f, t;\n\t.reg .b32 a1, ml, mh;\n\t.reg .b64 b1, b2, b3;\n\telect.sync _|e, 0xffffffff;\n\t"
      "setp.ne.b32 f, 0, 0;\n\tsetp.eq.b32 t, 1, 1;\n\tmov.b32 ml, 0;\n\tmov.b32 mh, 0xffffffff;\n\t"
      "add.u32 a1, %1, 8;\n\tadd.s64 b1, %2, 16;\n\tadd.s64 b2, %2, 256;\n\tadd.s64 b3, %2, 272;\n\t"
      "@e tcgen05.mma.cta_group::1.kind::f8f6f4 [%0], [%1], %2, %3, {ml, ml, mh, mh}, f;\n\t"
      "@e tcgen05.mma.cta_group::1.kind::f8f6f4 [%0], [a1], b1, %3, {ml, ml, mh, mh}, t;\n\t"
      "@e tcgen05.mma.cta_group::1.kind::f8f6f4 [%0], [%1], b2, %3, {mh, mh, ml, ml}, f;\n\t"
      "@e tcgen05.mma.cta_group::1.kind::f8f6f4 [%0], [a1], b3, %3, {mh, mh, ml, ml}, t;\n\t}" ::"r"(d),
      "r"(a), "l"(b), "r"(kIdescS)
      : "memory");
}
// O += P V for one page, lane-masked like S: lanes 0..63 take keys 0..63 (V rows 0..63), lanes 64..127 keys 64..127;
// A = each lane's 64 p values (16 columns), B = the V page (transposed); acc0 = 0: the first overwrites
__device__ __forceinline__ void mma_pv(uint32_t d, uint32_t a, uint64_t b, uint32_t acc0) {
  asm volatile(
      "{\n\t.reg .pred f, t, e;\n\t.reg .b32 a1, ml, mh;\n\t.reg .b64 b1, b2, b3;\n\telect.sync _|e, 0xffffffff;\n\t"
      "setp.ne.b32 f, %4, 0;\n\tsetp.eq.b32 t, 1, 1;\n\tmov.b32 ml, 0;\n\tmov.b32 mh, 0xffffffff;\n\t"
      "add.u32 a1, %1, 8;\n\tadd.s64 b1, %2, 128;\n\tadd.s64 b2, %2, 256;\n\tadd.s64 b3, %2, 384;\n\t"
      "@e tcgen05.mma.cta_group::1.kind::f8f6f4 [%0], [%1], %2, %3, {ml, ml, mh, mh}, f;\n\t"
      "@e tcgen05.mma.cta_group::1.kind::f8f6f4 [%0], [a1], b1, %3, {ml, ml, mh, mh}, t;\n\t"
      "@e tcgen05.mma.cta_group::1.kind::f8f6f4 [%0], [%1], b2, %3, {mh, mh, ml, ml}, f;\n\t"
      "@e tcgen05.mma.cta_group::1.kind::f8f6f4 [%0], [a1], b3, %3, {mh, mh, ml, ml}, t;\n\t}" ::"r"(d),
      "r"(a), "l"(b), "r"(kIdescPV), "r"(acc0)
      : "memory");
}
__device__ __forceinline__ void commit(uint32_t bar) {
  asm volatile(
      "{\n\t.reg .pred e;\n\telect.sync _|e, 0xffffffff;\n\t"
      "@e tcgen05.commit.cta_group::1.mbarrier::arrive::one.shared::cluster.b64 [%0];\n\t}" ::"r"(bar)
      : "memory");
}

__device__ __forceinline__ void tmem_ld32(uint32_t taddr, uint32_t (&r)[32]) {
  asm volatile(
      "tcgen05.ld.sync.aligned.32x32b.x32.b32 {%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17,%18,%19,"
      "%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31}, [%32];\n"
      : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]), "=r"(r[4]), "=r"(r[5]), "=r"(r[6]), "=r"(r[7]), "=r"(r[8]),
        "=r"(r[9]), "=r"(r[10]), "=r"(r[11]), "=r"(r[12]), "=r"(r[13]), "=r"(r[14]), "=r"(r[15]), "=r"(r[16]),
        "=r"(r[17]), "=r"(r[18]), "=r"(r[19]), "=r"(r[20]), "=r"(r[21]), "=r"(r[22]), "=r"(r[23]), "=r"(r[24]),
        "=r"(r[25]), "=r"(r[26]), "=r"(r[27]), "=r"(r[28]), "=r"(r[29]), "=r"(r[30]), "=r"(r[31])
      : "r"(taddr)
      : "memory");
}
__device__ __forceinline__ void tmem_st32(uint32_t taddr, const uint32_t (&r)[32]) {
  asm volatile(
      "tcgen05.st.sync.aligned.32x32b.x32.b32 [%0], {%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17,%18,"
      "%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31,%32};\n" ::"r"(taddr),
      "r"(r[0]), "r"(r[1]), "r"(r[2]), "r"(r[3]), "r"(r[4]), "r"(r[5]), "r"(r[6]), "r"(r[7]), "r"(r[8]), "r"(r[9]),
      "r"(r[10]), "r"(r[11]), "r"(r[12]), "r"(r[13]), "r"(r[14]), "r"(r[15]), "r"(r[16]), "r"(r[17]), "r"(r[18]),
      "r"(r[19]), "r"(r[20]), "r"(r[21]), "r"(r[22]), "r"(r[23]), "r"(r[24]), "r"(r[25]), "r"(r[26]), "r"(r[27]),
      "r"(r[28]), "r"(r[29]), "r"(r[30]), "r"(r[31])
      : "memory");
}
__device__ __forceinline__ void tmem_ld16(uint32_t taddr, uint32_t (&r)[16]) {
  asm volatile(
      "tcgen05.ld.sync.aligned.32x32b.x16.b32 {%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15}, [%16];\n"
      : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]), "=r"(r[4]), "=r"(r[5]), "=r"(r[6]), "=r"(r[7]), "=r"(r[8]),
        "=r"(r[9]), "=r"(r[10]), "=r"(r[11]), "=r"(r[12]), "=r"(r[13]), "=r"(r[14]), "=r"(r[15])
      : "r"(taddr)
      : "memory");
}
__device__ __forceinline__ void tmem_st16(uint32_t taddr, const uint32_t (&r)[16]) {
  asm volatile(
      "tcgen05.st.sync.aligned.32x32b.x16.b32 [%0], {%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16};\n" ::"r"(
          taddr),
      "r"(r[0]), "r"(r[1]), "r"(r[2]), "r"(r[3]), "r"(r[4]), "r"(r[5]), "r"(r[6]), "r"(r[7]), "r"(r[8]), "r"(r[9]),
      "r"(r[10]), "r"(r[11]), "r"(r[12]), "r"(r[13]), "r"(r[14]), "r"(r[15])
      : "memory");
}

__device__ __forceinline__ void tmem_st8(uint32_t taddr, const uint32_t (&r)[8]) {
  asm volatile("tcgen05.st.sync.aligned.32x32b.x8.b32 [%0], {%1,%2,%3,%4,%5,%6,%7,%8};\n" ::"r"(taddr), "r"(r[0]),
               "r"(r[1]), "r"(r[2]), "r"(r[3]), "r"(r[4]), "r"(r[5]), "r"(r[6]), "r"(r[7])
               : "memory");
}

__device__ __forceinline__ void prefetch_l2(const void* ptr) {
  asm volatile("prefetch.global.L2 [%0];\n" ::"l"(ptr));
}

__device__ __forceinline__ float fast_exp2(float x) {
  float y;
  asm("ex2.approx.ftz.f32 %0, %1;\n" : "=f"(y) : "f"(x));
  return y;
}
// e4m3(x0) in the low byte, e4m3(x1) in the high byte (nearest even, satfinite)
__device__ __forceinline__ uint32_t e4m3x2(float x0, float x1) {
  uint16_t e;
  asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;\n" : "=h"(e) : "f"(x1), "f"(x0));
  return e;
}
__device__ __forceinline__ float4 ld_cg_v4(const float* ptr) {
  float4 v;
  asm volatile("ld.global.cg.v4.f32 {%0, %1, %2, %3}, [%4];\n"
               : "=f"(v.x), "=f"(v.y), "=f"(v.z), "=f"(v.w)
               : "l"(ptr)
               : "memory");
  return v;
}
__device__ __forceinline__ float ld_cg_f32(const float* ptr) {
  float v;
  asm volatile("ld.global.cg.f32 %0, [%1];\n" : "=f"(v) : "l"(ptr) : "memory");
  return v;
}

template <int kPages, bool kUsePDL, bool kPrefetch>
__global__ void __launch_bounds__(kThreads, 2) swa_t5_kernel(const __grid_constant__ Params p) {
  using C = Cfg<kPages>;
  extern __shared__ __align__(128) uint8_t smem[];
  const int tid = static_cast<int>(threadIdx.x);
  const int warp = tid >> 5, lane = tid & 31;
  const int split = static_cast<int>(blockIdx.x), nsplit = static_cast<int>(gridDim.x);
  const int kvh = static_cast<int>(blockIdx.y), req = static_cast<int>(blockIdx.z);

  const uint32_t tiles = su(smem);
  const uint32_t ctl = su(smem + C::kCtl);
  const uint32_t bar_q = ctl, bar_s = ctl + 8, bar_o = ctl + 16, bar_p = ctl + 24, bar_sf = ctl + 32;
  const uint32_t bar_kv0 = ctl + 40;
  float* const mxs = reinterpret_cast<float*>(smem + C::kMx);
  uint32_t* const tmem_slot = reinterpret_cast<uint32_t*>(smem + C::kSlot);
  uint32_t* const last_flag = tmem_slot + 1;

  if (tid == 0) {
    mbar_init(bar_q, 128);
    mbar_init(bar_s, 1);
    mbar_init(bar_o, 1);
    mbar_init(bar_p, 128);
    mbar_init(bar_sf, 128);
#pragma unroll
    for (int j = 0; j < kPages; ++j) mbar_init(bar_kv0 + 8 * j, 128);
    asm volatile("fence.mbarrier_init.release.cluster;\n" ::: "memory");
  }
  if (warp == 0) {
    asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], %1;\n" ::"r"(su(tmem_slot)),
                 "r"(kTmemCols)
                 : "memory");
    asm volatile("tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned;\n" ::: "memory");
  }
  tc_fence_before();
  __syncthreads();
  tc_fence_after();
  const uint32_t tbase = *tmem_slot;

  if constexpr (kPrefetch) {
    // L2 prefetch hints for this block's K / V rows, issued before the wait: seq_lens and the page table may still be
    // written by the preceding kernel, so the values read here only form (bounds-checked) prefetch addresses and never
    // reach the results; every value the kernel uses is read again after griddepcontrol.wait
    if (warp >= 4) {
      const int ct = tid - 128;
      const int sl = min(max(p.seq_lens[req], 0), p.pt_cols * kPage);
      const int f0 = max(0, sl - kTokens - p.window_left);
      const int pf = f0 / kPage, pe = (sl + kPage - 1) / kPage;
      const int pp = (max(0, pe - pf) + nsplit - 1) / nsplit;
      const int q0 = pf + split * pp;
      const int qn = max(0, min(min(pp, kPages), pe - q0));
      const int32_t* const ptp = p.page_table + static_cast<int64_t>(req) * p.page_table_stride;
      for (int idx = ct; idx < qn * kPage; idx += 128) {
        const int64_t slot = static_cast<int64_t>(ptp[min(q0 + (idx >> 7), p.pt_cols - 1)]) * kPage + (idx & 127);
        if (slot >= 0 && slot < p.pool_slots) {
          prefetch_l2(p.k_pool + slot * p.pool_row_stride + kvh * kHeadDim);
          prefetch_l2(p.v_pool + slot * p.pool_row_stride + kvh * kHeadDim);
        }
      }
    }
  }
  if constexpr (kUsePDL) asm volatile("griddepcontrol.wait;\n" ::: "memory");
  const int seq_len = p.seq_lens[req];
  const int first = max(0, seq_len - kTokens - p.window_left);
  const int pg_first = first / kPage;
  const int pg_end = (seq_len + kPage - 1) / kPage;  // one past the last page holding a key
  const int npages = max(0, pg_end - pg_first);
  const int per = (npages + nsplit - 1) / nsplit;
  const int my_pg0 = pg_first + split * per;
  const int my_np = max(0, min(min(per, kPages), pg_end - my_pg0));
  const int pair = req * p.num_kv_heads + kvh;
  float* const part_blk = p.part + (static_cast<int64_t>(pair) * nsplit + split) * (kRows * kRecFloats);

  if (warp < 4) {
    // ---- softmax: thread = tensor-memory lane L = query row r (= L % 64) x key half h (= L / 64) ----
    const int h = tid >> 6;
    const int r = tid & 63;
    const int t = r >> 3, g = r & 7;
    const int pos = seq_len - kTokens + t;
    const int lo = max(0, pos - p.window_left);
    const uint32_t lane_off = static_cast<uint32_t>(warp * 32) << 16;
    {
      const uint4* qrow = reinterpret_cast<const uint4*>(
          p.q + static_cast<int64_t>(req * kTokens + t) * p.q_token_stride + (kvh * kQPerKv + g) * kHeadDim);
      uint32_t qv[16];
#pragma unroll
      for (int c = 0; c < 4; ++c) {
        const uint4 v = qrow[c];
        qv[4 * c] = v.x;
        qv[4 * c + 1] = v.y;
        qv[4 * c + 2] = v.z;
        qv[4 * c + 3] = v.w;
      }
      tmem_st16(tbase + lane_off + kColQ, qv);
      tc_wait_st();
      tc_fence_before();
      mbar_arrive(bar_q);
    }
    float m = -INFINITY, l = 0.0f;
    for (int j = 0; j < my_np; ++j) {
      mbar_wait(bar_s, static_cast<uint32_t>(j & 1));
      tc_fence_after();
      float sv[kHalf];
      {
        uint32_t s0[32], s1[32];
        tmem_ld32(tbase + lane_off + kColS, s0);
        tmem_ld32(tbase + lane_off + kColS + 32, s1);
        tc_wait_ld();
        // the scores are in registers: the next page's S may overwrite the columns
        tc_fence_before();
        mbar_arrive(bar_sf);
#pragma unroll
        for (int i = 0; i < 32; ++i) {
          sv[i] = __uint_as_float(s0[i]);
          sv[32 + i] = __uint_as_float(s1[i]);
        }
      }
      const int kh0 = (my_pg0 + j) * kPage + kHalf * h;
      if (!(kh0 >= lo && kh0 + kHalf - 1 <= pos)) {
#pragma unroll
        for (int i = 0; i < kHalf; ++i) {
          const int key = kh0 + i;
          if (key < lo || key > pos) sv[i] = -INFINITY;
        }
      }
      float mt;
      {
        float a[8];
#pragma unroll
        for (int u = 0; u < 8; ++u) a[u] = sv[u];
#pragma unroll
        for (int i = 8; i < kHalf; ++i) a[i & 7] = fmaxf(a[i & 7], sv[i]);
        mt = fmaxf(fmaxf(fmaxf(a[0], a[1]), fmaxf(a[2], a[3])), fmaxf(fmaxf(a[4], a[5]), fmaxf(a[6], a[7])));
      }
      mt *= p.scale_log2;  // max of x = s * scale_log2 over this key half (scale_log2 > 0)
      const float m_new = fmaxf(m, mt);
      const float mu = (m_new == -INFINITY) ? 0.0f : m_new;
      const float alpha = fast_exp2(m - mu);  // m = -inf -> 0
      // p * 448 directly: exp2(x - m + log2 448); l sums the same scaled values (the 448 cancels in o / l)
      const float bias = kLog2PScale - mu;
      float sa[4] = {0.f, 0.f, 0.f, 0.f};
      uint32_t pw[16];
#pragma unroll
      for (int w = 0; w < 16; ++w) {
        const float p0 = fast_exp2(fmaf(sv[4 * w], p.scale_log2, bias));
        const float p1 = fast_exp2(fmaf(sv[4 * w + 1], p.scale_log2, bias));
        const float p2 = fast_exp2(fmaf(sv[4 * w + 2], p.scale_log2, bias));
        const float p3 = fast_exp2(fmaf(sv[4 * w + 3], p.scale_log2, bias));
        sa[0] += p0;
        sa[1] += p1;
        sa[2] += p2;
        sa[3] += p3;
        pw[w] = e4m3x2(p0, p1) | (e4m3x2(p2, p3) << 16);
      }
      if (j > 0) {
        // PV of the previous page has completed: o (its lane's key-half sum) is stable and the p columns are free
        mbar_wait(bar_o, static_cast<uint32_t>((j - 1) & 1));
        tc_fence_after();
        if (__any_sync(0xffffffffu, alpha != 1.0f)) {
#pragma unroll
          for (int c = 0; c < 2; ++c) {
            uint32_t ov[32];
            tmem_ld32(tbase + lane_off + kColO + 32 * c, ov);
            tc_wait_ld();
#pragma unroll
            for (int i = 0; i < 32; ++i) ov[i] = __float_as_uint(__uint_as_float(ov[i]) * alpha);
            tmem_st32(tbase + lane_off + kColO + 32 * c, ov);
          }
        }
      }
      tmem_st16(tbase + lane_off + kColP, pw);
      l = l * alpha + ((sa[0] + sa[1]) + (sa[2] + sa[3]));
      m = m_new;
      tc_wait_st();
      tc_fence_before();
      mbar_arrive(bar_p);
    }
    if constexpr (kUsePDL) asm volatile("griddepcontrol.launch_dependents;\n" ::: "memory");
    // ---- partial record: o = half 0 + half 1 of the row (lanes r and 64 + r) ----
    float* const xo = reinterpret_cast<float*>(smem);  // [64 rows][66] half-1 o, m, l; aliases page 0 (MMAs done)
    uint32_t ov[64];
    if (my_np > 0) {
      mbar_wait(bar_o, static_cast<uint32_t>((my_np - 1) & 1));
      tc_fence_after();
      uint32_t (&o0)[32] = *reinterpret_cast<uint32_t(*)[32]>(&ov[0]);
      uint32_t (&o1)[32] = *reinterpret_cast<uint32_t(*)[32]>(&ov[32]);
      tmem_ld32(tbase + lane_off + kColO, o0);
      tmem_ld32(tbase + lane_off + kColO + 32, o1);
      tc_wait_ld();
    } else {
#pragma unroll
      for (int i = 0; i < 64; ++i) ov[i] = 0u;
    }
    fence_proxy_async();  // generic writes into page 0 after the MMAs' async-proxy reads of it
    if (h == 1) {
#pragma unroll
      for (int i = 0; i < 64; ++i) xo[r * 66 + i] = __uint_as_float(ov[i]);
      xo[r * 66 + 64] = m;
      xo[r * 66 + 65] = l;
    }
    softmax_sync();
    if (h == 0) {
      // the two key halves keep their own running maxima: merge them with max-rescaling (half 0 first)
      const float m1 = xo[r * 66 + 64];
      const float mm = fmaxf(m, m1);
      const float w0 = m == -INFINITY ? 0.0f : fast_exp2(m - mm);
      const float w1 = m1 == -INFINITY ? 0.0f : fast_exp2(m1 - mm);
      float* const rr = reinterpret_cast<float*>(smem + C::kRecS) + r * kRecFloats;
#pragma unroll
      for (int i = 0; i < 64; i += 4)
        *reinterpret_cast<float4*>(rr + i) = make_float4(
            __uint_as_float(ov[i]) * w0 + xo[r * 66 + i] * w1, __uint_as_float(ov[i + 1]) * w0 + xo[r * 66 + i + 1] * w1,
            __uint_as_float(ov[i + 2]) * w0 + xo[r * 66 + i + 2] * w1,
            __uint_as_float(ov[i + 3]) * w0 + xo[r * 66 + i + 3] * w1);
      rr[64] = mm;
      rr[65] = l * w0;
      rr[66] = xo[r * 66 + 65] * w1;
      rr[67] = 0.0f;
    }
  } else {
    // ---- copies: warps 4-7, 4 x 16 bytes of K and of V per thread per page; warp 7 issues the MMAs ----
    const int ct = tid - 128;
    const int32_t* const pt = p.page_table + static_cast<int64_t>(req) * p.page_table_stride;
#pragma unroll
    for (int j = 0; j < kPages; ++j) {
      if (j < my_np) {
        const int pg = my_pg0 + j;
        const int64_t slot0 = static_cast<int64_t>(pt[pg]) * kPage;
        const uint32_t kdst = tiles + j * kPageBytes, vdst = kdst + kPage * kHeadDim;
#pragma unroll
        for (int u = 0; u < 4; ++u) {
          const int idx = u * 128 + ct;  // 0..511 = key * 4 + 16-byte chunk
          const int i = idx >> 2, c = idx & 3;
          const int key = pg * kPage + i;
          const bool valid = key >= first && key < seq_len;
          const int64_t off = (slot0 + i) * p.pool_row_stride + kvh * kHeadDim + 16 * c;
          const uint32_t o = static_cast<uint32_t>((i >> 3) * 512 + c * 128 + (i & 7) * 16);
          cp_async16(kdst + o, valid ? p.k_pool + off : p.k_pool, valid ? 16u : 0u);
          cp_async16(vdst + o, valid ? p.v_pool + off : p.v_pool, valid ? 16u : 0u);
        }
      }
      cp_commit();
    }
    if (warp == 7) {
      mbar_wait(bar_q, 0);
      tc_fence_after();
    }
#pragma unroll
    for (int j = 0; j < kPages; ++j) {
      cp_wait_n(kPages - 1 - j);
      fence_proxy_async();
      mbar_arrive(bar_kv0 + 8 * j);
      if (warp == 7 && j < my_np) {
        // S of page j as soon as its K is in and the softmax warps hold page j-1's scores in registers, so it runs
        // under page j-1's exp2 work; then PV of page j-1 once its p is in tensor memory
        mbar_wait(bar_kv0 + 8 * j, 0);
        if (j > 0) {
          mbar_wait(bar_sf, static_cast<uint32_t>((j - 1) & 1));
          tc_fence_after();
        }
        mma_s(tbase + kColS, tbase + kColQ, sdesc(tiles + j * kPageBytes, 128, 512));
        commit(bar_s);
        if (j > 0) {
          mbar_wait(bar_p, static_cast<uint32_t>((j - 1) & 1));
          tc_fence_after();
          mma_pv(tbase + kColO, tbase + kColP, sdesc(tiles + (j - 1) * kPageBytes + kPage * kHeadDim, 512, 128),
                 j > 1 ? 1u : 0u);
          commit(bar_o);
        }
      }
    }
    if (warp == 7 && my_np > 0) {
      mbar_wait(bar_p, static_cast<uint32_t>((my_np - 1) & 1));
      tc_fence_after();
      mma_pv(tbase + kColO, tbase + kColP, sdesc(tiles + (my_np - 1) * kPageBytes + kPage * kHeadDim, 512, 128),
             my_np > 1 ? 1u : 0u);
      commit(bar_o);
    }
  }

  // ---- arrival first: the block that is last to arrive merges; the others publish their record and leave ----
  tc_fence_before();
  __syncthreads();
  const int npairs = static_cast<int>(gridDim.y * gridDim.z);
  if (tid == 0) {
    uint32_t old;
    asm volatile("atom.relaxed.gpu.global.add.u32 %0, [%1], 1;\n" : "=r"(old) : "l"(p.count + pair) : "memory");
    *last_flag = old == static_cast<uint32_t>(nsplit - 1);
  }
  __syncthreads();
  if (warp == 0) {
    tc_fence_after();
    asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, %1;\n" ::"r"(tbase), "r"(kTmemCols) : "memory");
  }
  const float* const rec_s = reinterpret_cast<const float*>(smem + C::kRecS);
  if (*last_flag == 0u) {
    // not last: the record goes to the workspace, then one release add on the pair's ready counter
    for (int i = tid; i < kRows * kRecFloats / 4; i += kThreads)
      reinterpret_cast<float4*>(part_blk)[i] = reinterpret_cast<const float4*>(rec_s)[i];
    __syncthreads();
    if (tid == 0)
      asm volatile("red.release.gpu.global.add.u32 [%0], 1;\n" ::"l"(p.count + npairs + pair) : "memory");
    return;
  }
  if (tid == 0) {
    // the other S - 1 blocks have counted their arrival, so each is running and only stores its record before its
    // release add: the wait is bounded by their store latency (a defect traps)
    uint32_t polls = 0;
    while (true) {
      uint32_t ready;
      asm volatile("ld.acquire.gpu.global.u32 %0, [%1];\n" : "=r"(ready) : "l"(p.count + npairs + pair) : "memory");
      if (ready == static_cast<uint32_t>(nsplit - 1)) break;
      if (++polls == kMaxPolls) [[unlikely]] t5_wait_bound_exceeded();
    }
  }
  __syncthreads();
  {
    // thread = (row r = tid / 4, dims 16 * (tid % 4) .. + 15); loads batched four splits at a time
    const int r = tid >> 2, q4 = tid & 3;
    const float* const part_pair = p.part + static_cast<int64_t>(pair) * nsplit * (kRows * kRecFloats);
    float ms[kMaxSplits];
#pragma unroll
    for (int s = 0; s < kMaxSplits; ++s)
      ms[s] = s >= nsplit ? -INFINITY
              : s == split ? rec_s[r * kRecFloats + 64]
                           : __ldcg(part_pair + (s * kRows + r) * kRecFloats + 64);
    float mx = -INFINITY;
#pragma unroll
    for (int s = 0; s < kMaxSplits; ++s) mx = fmaxf(mx, ms[s]);
    float acc[16];
#pragma unroll
    for (int i = 0; i < 16; ++i) acc[i] = 0.0f;
    float lsum = 0.0f;
    if (mx != -INFINITY) {
#pragma unroll
      for (int s0 = 0; s0 < kMaxSplits; s0 += 4) {
        if (s0 >= nsplit) break;
        float4 v[4][4];
        float l2[4][2];
#pragma unroll
        for (int u = 0; u < 4; ++u) {
          const int s = s0 + u;
          if (s == split) {
            const float* const rr = rec_s + r * kRecFloats;
#pragma unroll
            for (int c = 0; c < 4; ++c) v[u][c] = *reinterpret_cast<const float4*>(rr + 16 * q4 + 4 * c);
            l2[u][0] = rr[65];
            l2[u][1] = rr[66];
          } else {
            const float* const rr = part_pair + (min(s, nsplit - 1) * kRows + r) * kRecFloats;
#pragma unroll
            for (int c = 0; c < 4; ++c) v[u][c] = __ldcg(reinterpret_cast<const float4*>(rr + 16 * q4 + 4 * c));
            l2[u][0] = __ldcg(rr + 65);
            l2[u][1] = __ldcg(rr + 66);
          }
        }
#pragma unroll
        for (int u = 0; u < 4; ++u) {
          const int s = s0 + u;
          if (s < nsplit && ms[s] != -INFINITY) {
            const float w = fast_exp2(ms[s] - mx);
#pragma unroll
            for (int c = 0; c < 4; ++c) {
              acc[4 * c] += w * v[u][c].x;
              acc[4 * c + 1] += w * v[u][c].y;
              acc[4 * c + 2] += w * v[u][c].z;
              acc[4 * c + 3] += w * v[u][c].w;
            }
            lsum += w * (l2[u][0] + l2[u][1]);
          }
        }
      }
    }
    const float sc = lsum > 0.0f ? p.out_scale / lsum : 0.0f;
    const int t = r >> 3, g = r & 7;
    uint32_t packed[8];
#pragma unroll
    for (int i = 0; i < 8; ++i) {
      __nv_bfloat162 b = __floats2bfloat162_rn(acc[2 * i] * sc, acc[2 * i + 1] * sc);
      packed[i] = *reinterpret_cast<uint32_t*>(&b);
    }
    uint4* const dst = reinterpret_cast<uint4*>(
        p.out + (static_cast<int64_t>(req * kTokens + t) * p.num_q_heads + kvh * kQPerKv + g) * kHeadDim + 16 * q4);
    dst[0] = make_uint4(packed[0], packed[1], packed[2], packed[3]);
    dst[1] = make_uint4(packed[4], packed[5], packed[6], packed[7]);
    if (tid == 0) {  // the next launch starts from 0 (launches are stream-ordered)
      p.count[pair] = 0u;
      p.count[npairs + pair] = 0u;
    }
  }
}

// ---------------------------------------------------------------------------------------------------------------
// t6: one block per (request, KV head, split) with kNPipe page pipelines (pipeline q takes pages q, q + kNPipe, ...
// of the block's run) sharing a kRing-page shared-memory ring that one copy warp keeps full (a slot is refilled once
// the PV that last read it has completed: tcgen05.commit on the slot's free barrier). Per pipeline: 4 softmax warps
// (warps 4q..4q+3, one lane quarter each; the t5 page step) and one MMA warp (warp 12 + q). Lane-masked S / PV as t5.
// The block merges its pipelines (tensor memory -> registers) and key halves (shared memory). With S > 1 blocks per
// pair: kCluster (S = 2..3, one cluster per pair) - blocks 1..S-1 push their merged rows into block 0's shared memory
// (st.async, byte-counted on block 0's barrier) and block 0 merges in block order and writes the output; otherwise
// the arrival-first split-K merge of t5 through the global workspace.
// ---------------------------------------------------------------------------------------------------------------
constexpr int kNPipe = 3;
constexpr int kRing = 12;
constexpr int kT6Copy = 1;                                   // copy warps
constexpr int kT6Threads = (4 * kNPipe + kNPipe + kT6Copy) * 32;  // 512
constexpr uint32_t kT6Col = 160;                             // per pipeline: S 64, O 64, P 16
constexpr uint32_t kT6ColQ = 480;
constexpr int kT6Ring = kRing * kPageBytes;                  // 144 KiB
constexpr int kT6RecF = 68;                                  // per lane: 64 o, m, l, 2 pad (16-byte rows)
constexpr int kT6RecS = kNPipe * 128 * kT6RecF * 4;          // pipeline records (alias the ring at the end)
constexpr int kT6BlockRec = kT6RecS;                         // the block record (64 rows x kRecFloats) after them
static_assert(kT6RecS + kRows * kRecFloats * 4 <= kT6Ring, "epilogue records fit in the ring");
constexpr int kT6Ctl = kT6Ring;
// barriers: q, per slot K-loaded / free, per pipeline s / o0 / p0 / sf / o1 / p1, recv (cluster merge), per slot
// V-loaded
constexpr int kT6Bars = 1 + 2 * kRing + 6 * kNPipe + 1 + kRing;
constexpr int kT6VLag = 3;  // a page's V box is requested this many pages after its K box (the first S MMAs need K only)
constexpr int kT6Slot = kT6Ctl + 8 * kT6Bars;
constexpr int kT6Bytes = kT6Slot + 16;
// cluster merge (kCluster): blocks 1..S-1 of a pair push their merged rows (64 o, m, l, 2 pad) into block 0's
// receive area with st.async (completion counted in bytes on block 0's recv barrier)
constexpr int kT6MaxClSplits = 3;
constexpr int kT6RecvRow = kRecFloats * 4;
constexpr int kT6Recv = (kT6Bytes + 127) & ~127;
constexpr int kT6BytesCl = kT6Recv + (kT6MaxClSplits - 1) * kRows * kT6RecvRow;

// shared-memory descriptor, 64-byte swizzle (layout type 4): start, stride byte offset (8 rows of 64 bytes)
__device__ __forceinline__ uint64_t sdesc64(uint32_t saddr) {
  return static_cast<uint64_t>((saddr >> 4) & 0x3FFFu) | (1ull << 16) | (static_cast<uint64_t>(512 >> 4) << 32) |
         (1ull << 46) | (4ull << 61);
}
__device__ __forceinline__ void tma_load_2d(uint32_t dst, const CUtensorMap* map, int32_t c0, int32_t c1, uint32_t bar) {
  asm volatile(
      "cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1, {%2, %3}], [%4];\n" ::"r"(
          dst),
      "l"(reinterpret_cast<uint64_t>(map)), "r"(c0), "r"(c1), "r"(bar)
      : "memory");
}
// L2 prefetch of the same box (no shared-memory destination, no completion)
__device__ __forceinline__ void tma_prefetch_2d(const CUtensorMap* map, int32_t c0, int32_t c1) {
  asm volatile("cp.async.bulk.prefetch.tensor.2d.L2.global.tile [%0, {%1, %2}];\n" ::"l"(reinterpret_cast<uint64_t>(map)),
               "r"(c0), "r"(c1)
               : "memory");
}
__device__ __forceinline__ void mbar_expect(uint32_t bar, uint32_t bytes) {
  asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;\n" ::"r"(bar), "r"(bytes) : "memory");
}
// S for one page from a 64-byte-swizzled K tile (TMA layout): K steps advance 32 bytes inside the swizzled row
__device__ __forceinline__ void mma_s_sw(uint32_t d, uint32_t a, uint64_t b) {
  asm volatile(
      "{\n\t.reg .pred e, f, t;\n\t.reg .b32 a1, ml, mh;\n\t.reg .b64 b1, b2, b3;\n\telect.sync _|e, 0xffffffff;\n\t"
      "setp.ne.b32 f, 0, 0;\n\tsetp.eq.b32 t, 1, 1;\n\tmov.b32 ml, 0;\n\tmov.b32 mh, 0xffffffff;\n\t"
      "add.u32 a1, %1, 8;\n\tadd.s64 b1, %2, 2;\n\tadd.s64 b2, %2, 256;\n\tadd.s64 b3, %2, 258;\n\t"
      "@e tcgen05.mma.cta_group::1.kind::f8f6f4 [%0], [%1], %2, %3, {ml, ml, mh, mh}, f;\n\t"
      "@e tcgen05.mma.cta_group::1.kind::f8f6f4 [%0], [a1], b1, %3, {ml, ml, mh, mh}, t;\n\t"
      "@e tcgen05.mma.cta_group::1.kind::f8f6f4 [%0], [%1], b2, %3, {mh, mh, ml, ml}, f;\n\t"
      "@e tcgen05.mma.cta_group::1.kind::f8f6f4 [%0], [a1], b3, %3, {mh, mh, ml, ml}, t;\n\t}" ::"r"(d),
      "r"(a), "l"(b), "r"(kIdescS)
      : "memory");
}

__device__ __forceinline__ uint32_t map_rank(uint32_t addr, uint32_t rank) {
  uint32_t d;
  asm volatile("mapa.shared::cluster.u32 %0, %1, %2;\n" : "=r"(d) : "r"(addr), "r"(rank));
  return d;
}
// 16 bytes into another block's shared memory (shared::cluster address); the destination block's barrier `rbar`
// (shared::cluster address in the same block) counts the bytes on completion
__device__ __forceinline__ void st_async_v4(uint32_t addr, float a, float b, float c, float d, uint32_t rbar) {
  asm volatile("st.async.shared::cluster.mbarrier::complete_tx::bytes.v4.b32 [%0], {%1, %2, %3, %4}, [%5];\n" ::"r"(addr),
               "r"(__float_as_uint(a)), "r"(__float_as_uint(b)), "r"(__float_as_uint(c)), "r"(__float_as_uint(d)),
               "r"(rbar)
               : "memory");
}

// wait with acquire at cluster scope (the arrivals are complete-tx from other blocks of the cluster)
__device__ __forceinline__ void mbar_wait_cluster(uint32_t bar, uint32_t parity) {
  uint32_t polls = 0;
  while (true) {
    uint32_t done;
    asm volatile(
        "{\n.reg .pred p;\nmbarrier.try_wait.parity.acquire.cluster.shared::cta.b64 p, [%1], %2;\nselp.u32 %0, 1, 0, p;\n}\n"
        : "=r"(done)
        : "r"(bar), "r"(parity)
        : "memory");
    if (done) break;
    if (++polls == kMaxPolls) [[unlikely]] t5_wait_bound_exceeded();
  }
}
__device__ __forceinline__ void cluster_arrive_release() {
  asm volatile("barrier.cluster.arrive.release.aligned;\n" ::: "memory");
}
__device__ __forceinline__ void cluster_arrive_relaxed() {
  asm volatile("barrier.cluster.arrive.relaxed.aligned;\n" ::: "memory");
}
__device__ __forceinline__ void cluster_wait_acquire() {
  asm volatile("barrier.cluster.wait.acquire.aligned;\n" ::: "memory");
}
__device__ __forceinline__ bool mbar_test(uint32_t bar, uint32_t parity) {
  uint32_t ok;
  asm volatile(
      "{\n\t.reg .pred p;\n\tmbarrier.test_wait.parity.acquire.cta.shared::cta.b64 p, [%1], %2;\n\tselp.u32 %0, 1, 0, p;\n\t}"
      : "=r"(ok)
      : "r"(bar), "r"(parity)
      : "memory");
  return ok != 0;
}
// packed fp32 pairs (PTX ISA 8.6+, sm_100+): the same per-element rounding as the scalar instructions
__device__ __forceinline__ uint64_t f2pack(float a, float b) {
  uint64_t d;
  asm("mov.b64 %0, {%1, %2};\n" : "=l"(d) : "f"(a), "f"(b));
  return d;
}
// (d0, d1) = (a0 * b + c, a1 * b + c), b2 = {b, b}, c2 = {c, c}
__device__ __forceinline__ void fma2(float a0, float a1, uint64_t b2, uint64_t c2, float& d0, float& d1) {
  asm("{\n\t.reg .b64 ra, rd;\n\tmov.b64 ra, {%2, %3};\n\tfma.rn.f32x2 rd, ra, %4, %5;\n\tmov.b64 {%0, %1}, rd;\n\t}"
      : "=f"(d0), "=f"(d1)
      : "f"(a0), "f"(a1), "l"(b2), "l"(c2));
}
// (s0, s1) += (a0, a1)
__device__ __forceinline__ void add2(float& s0, float& s1, float a0, float a1) {
  asm("{\n\t.reg .b64 rs, ra;\n\tmov.b64 rs, {%0, %1};\n\tmov.b64 ra, {%2, %3};\n\tadd.rn.f32x2 rs, rs, ra;\n\t"
      "mov.b64 {%0, %1}, rs;\n\t}"
      : "+f"(s0), "+f"(s1)
      : "f"(a0), "f"(a1));
}
// three-input maximum (PTX ISA 8.8+, sm_100+); exact, so the grouping does not change the result
__device__ __forceinline__ float fmax3(float a, float b, float c) {
  float d;
  asm("max.f32 %0, %1, %2, %3;\n" : "=f"(d) : "f"(a), "f"(b), "f"(c));
  return d;
}

// 64 consecutive fp32 columns of this warp's lanes plus their maximum (tcgen05.ld.red, PTX ISA 8.8+): the maximum is
// exact, so it equals a maximum taken in registers
__device__ __forceinline__ void tmem_ld64_max(uint32_t taddr, float (&v)[64], float& mx) {
  asm volatile(
      "tcgen05.ld.red.sync.aligned.32x32b.x64.max.f32 {%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31,%32,%33,%34,%35,%36,%37,%38,%39,%40,%41,%42,%43,%44,%45,%46,%47,%48,%49,%50,%51,%52,%53,%54,%55,%56,%57,%58,%59,%60,%61,%62,%63}, %64, [%65];\n"
      : "=f"(v[0]), "=f"(v[1]), "=f"(v[2]), "=f"(v[3]), "=f"(v[4]), "=f"(v[5]), "=f"(v[6]), "=f"(v[7]), "=f"(v[8]), "=f"(v[9]), "=f"(v[10]), "=f"(v[11]), "=f"(v[12]), "=f"(v[13]), "=f"(v[14]), "=f"(v[15]), "=f"(v[16]), "=f"(v[17]), "=f"(v[18]), "=f"(v[19]), "=f"(v[20]), "=f"(v[21]), "=f"(v[22]), "=f"(v[23]), "=f"(v[24]), "=f"(v[25]), "=f"(v[26]), "=f"(v[27]), "=f"(v[28]), "=f"(v[29]), "=f"(v[30]), "=f"(v[31]), "=f"(v[32]), "=f"(v[33]), "=f"(v[34]), "=f"(v[35]), "=f"(v[36]), "=f"(v[37]), "=f"(v[38]), "=f"(v[39]), "=f"(v[40]), "=f"(v[41]), "=f"(v[42]), "=f"(v[43]), "=f"(v[44]), "=f"(v[45]), "=f"(v[46]), "=f"(v[47]), "=f"(v[48]), "=f"(v[49]), "=f"(v[50]), "=f"(v[51]), "=f"(v[52]), "=f"(v[53]), "=f"(v[54]), "=f"(v[55]), "=f"(v[56]), "=f"(v[57]), "=f"(v[58]), "=f"(v[59]), "=f"(v[60]), "=f"(v[61]), "=f"(v[62]), "=f"(v[63]), "=f"(mx)
      : "r"(taddr)
      : "memory");
}
__device__ __forceinline__ void tmem_ld64(uint32_t taddr, uint32_t (&r)[64]) {
  asm volatile(
      "tcgen05.ld.sync.aligned.32x32b.x64.b32 {%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31,%32,%33,%34,%35,%36,%37,%38,%39,%40,%41,%42,%43,%44,%45,%46,%47,%48,%49,%50,%51,%52,%53,%54,%55,%56,%57,%58,%59,%60,%61,%62,%63}, [%64];\n"
      : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]), "=r"(r[4]), "=r"(r[5]), "=r"(r[6]), "=r"(r[7]), "=r"(r[8]), "=r"(r[9]), "=r"(r[10]), "=r"(r[11]), "=r"(r[12]), "=r"(r[13]), "=r"(r[14]), "=r"(r[15]), "=r"(r[16]), "=r"(r[17]), "=r"(r[18]), "=r"(r[19]), "=r"(r[20]), "=r"(r[21]), "=r"(r[22]), "=r"(r[23]), "=r"(r[24]), "=r"(r[25]), "=r"(r[26]), "=r"(r[27]), "=r"(r[28]), "=r"(r[29]), "=r"(r[30]), "=r"(r[31]), "=r"(r[32]), "=r"(r[33]), "=r"(r[34]), "=r"(r[35]), "=r"(r[36]), "=r"(r[37]), "=r"(r[38]), "=r"(r[39]), "=r"(r[40]), "=r"(r[41]), "=r"(r[42]), "=r"(r[43]), "=r"(r[44]), "=r"(r[45]), "=r"(r[46]), "=r"(r[47]), "=r"(r[48]), "=r"(r[49]), "=r"(r[50]), "=r"(r[51]), "=r"(r[52]), "=r"(r[53]), "=r"(r[54]), "=r"(r[55]), "=r"(r[56]), "=r"(r[57]), "=r"(r[58]), "=r"(r[59]), "=r"(r[60]), "=r"(r[61]), "=r"(r[62]), "=r"(r[63])
      : "r"(taddr)
      : "memory");
}
__device__ __forceinline__ void tmem_st64(uint32_t taddr, const uint32_t (&r)[64]) {
  asm volatile(
      "tcgen05.st.sync.aligned.32x32b.x64.b32 [%0], {%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31,%32,%33,%34,%35,%36,%37,%38,%39,%40,%41,%42,%43,%44,%45,%46,%47,%48,%49,%50,%51,%52,%53,%54,%55,%56,%57,%58,%59,%60,%61,%62,%63,%64};\n" ::"r"(taddr),
      "r"(r[0]), "r"(r[1]), "r"(r[2]), "r"(r[3]), "r"(r[4]), "r"(r[5]), "r"(r[6]), "r"(r[7]), "r"(r[8]), "r"(r[9]), "r"(r[10]), "r"(r[11]), "r"(r[12]), "r"(r[13]), "r"(r[14]), "r"(r[15]), "r"(r[16]), "r"(r[17]), "r"(r[18]), "r"(r[19]), "r"(r[20]), "r"(r[21]), "r"(r[22]), "r"(r[23]), "r"(r[24]), "r"(r[25]), "r"(r[26]), "r"(r[27]), "r"(r[28]), "r"(r[29]), "r"(r[30]), "r"(r[31]), "r"(r[32]), "r"(r[33]), "r"(r[34]), "r"(r[35]), "r"(r[36]), "r"(r[37]), "r"(r[38]), "r"(r[39]), "r"(r[40]), "r"(r[41]), "r"(r[42]), "r"(r[43]), "r"(r[44]), "r"(r[45]), "r"(r[46]), "r"(r[47]), "r"(r[48]), "r"(r[49]), "r"(r[50]), "r"(r[51]), "r"(r[52]), "r"(r[53]), "r"(r[54]), "r"(r[55]), "r"(r[56]), "r"(r[57]), "r"(r[58]), "r"(r[59]), "r"(r[60]), "r"(r[61]), "r"(r[62]), "r"(r[63])
      : "memory");
}
// (a0, a1) *= b2 = {b, b} (mul.rn.f32x2: the same rounding as two scalar multiplies)
__device__ __forceinline__ void mul2(uint32_t& a0, uint32_t& a1, uint64_t b2) {
  asm("{\n\t.reg .b64 ra;\n\tmov.b64 ra, {%0, %1};\n\tmul.rn.f32x2 ra, ra, %2;\n\tmov.b64 {%0, %1}, ra;\n\t}"
      : "+r"(a0), "+r"(a1)
      : "l"(b2));
}

// two fp32 columns of this warp's lanes
__device__ __forceinline__ void tmem_st2(uint32_t taddr, float a, float b) {
  asm volatile("tcgen05.st.sync.aligned.32x32b.x2.b32 [%0], {%1, %2};\n" ::"r"(taddr), "r"(__float_as_uint(a)),
               "r"(__float_as_uint(b))
               : "memory");
}
__device__ __forceinline__ void tmem_ld2(uint32_t taddr, float& a, float& b) {
  uint32_t x, y;
  asm volatile("tcgen05.ld.sync.aligned.32x32b.x2.b32 {%0, %1}, [%2];\n" : "=r"(x), "=r"(y) : "r"(taddr) : "memory");
  a = __uint_as_float(x);
  b = __uint_as_float(y);
}

__device__ __forceinline__ void cp_wait_n8(int n) {
  switch (n) {
    case 0: cp_wait<0>(); break;
    case 1: cp_wait<1>(); break;
    case 2: cp_wait<2>(); break;
    case 3: cp_wait<3>(); break;
    case 4: cp_wait<4>(); break;
    case 5: cp_wait<5>(); break;
    case 6: cp_wait<6>(); break;
    case 7: cp_wait<7>(); break;
    default: cp_wait<8>(); break;
  }
}

template <bool kUsePDL, bool kCluster, bool kEarly>
__global__ void __launch_bounds__(kT6Threads, 1) swa_t6_kernel(const __grid_constant__ Params p,
                                                                const __grid_constant__ CUtensorMap k_map,
                                                                const __grid_constant__ CUtensorMap v_map) {
  extern __shared__ __align__(128) uint8_t smem[];
  const int tid = static_cast<int>(threadIdx.x);
  const int warp = tid >> 5, lane = tid & 31;
  const int split = static_cast<int>(blockIdx.x), nsplit = static_cast<int>(gridDim.x);
  const int kvh = static_cast<int>(blockIdx.y), req = static_cast<int>(blockIdx.z);

  const uint32_t ring = su(smem);
  const uint32_t ctl = su(smem + kT6Ctl);
  const uint32_t bar_q = ctl;
  const uint32_t bar_loaded0 = ctl + 8, bar_free0 = ctl + 8 + 8 * kRing, bar_pipe0 = ctl + 8 + 16 * kRing;
  const uint32_t bar_recv = ctl + 8 * (1 + 2 * kRing + 6 * kNPipe);
  const uint32_t bar_vload0 = bar_recv + 8;
  uint32_t* const tmem_slot = reinterpret_cast<uint32_t*>(smem + kT6Slot);
  uint32_t* const last_flag = tmem_slot + 1;

  int pf_sl = 0;
  if constexpr (kUsePDL && !kEarly) {
    // first thing, before the barrier setup: an L2 prefetch of the page-table line(s) the copy warp reads right after
    // griddepcontrol.wait. seq_lens may still be written by the preceding kernel, so the value read here only forms a
    // bounds-checked prefetch address and never reaches a result; every value used is read again after the wait.
    // One 32-bit load of this block's own request's entry (the producer's element width); the prefetch is issued
    // after the copy warp's tensor-memory allocation below, which overlaps this load
    if (warp == 4 * kNPipe + kNPipe && lane < 2) pf_sl = p.test_pf_seq_force ? p.test_pf_seq_value : p.seq_lens[req];
  }
  if (tid == 0) {
    mbar_init(bar_q, 128);
#pragma unroll
    for (int s = 0; s < kRing; ++s) {
      mbar_init(bar_loaded0 + 8 * s, 1);
      mbar_init(bar_vload0 + 8 * s, 1);
      mbar_init(bar_free0 + 8 * s, 1);
    }
#pragma unroll
    for (int q = 0; q < kNPipe; ++q) {
      mbar_init(bar_pipe0 + 48 * q, 1);        // s
      mbar_init(bar_pipe0 + 48 * q + 8, 1);    // o0: PV of the pipeline's even steps (P buffer 0)
      mbar_init(bar_pipe0 + 48 * q + 16, 128); // p0: P of the even steps written
      mbar_init(bar_pipe0 + 48 * q + 24, 128); // sf
      mbar_init(bar_pipe0 + 48 * q + 32, 1);   // o1: PV of the odd steps (P buffer 1)
      mbar_init(bar_pipe0 + 48 * q + 40, 128); // p1: P of the odd steps written
    }
    if constexpr (kCluster) mbar_init(bar_recv, 1);
    asm volatile("fence.mbarrier_init.release.cluster;\n" ::: "memory");
    if constexpr (kCluster) {
      // block 0 expects every other block's 64 rows; its own arrival is this one
      if (split == 0)
        asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;\n" ::"r"(bar_recv),
                     "r"(static_cast<uint32_t>((nsplit - 1) * kRows * kT6RecvRow))
                     : "memory");
    }
  }
  if (warp == 4 * kNPipe + kNPipe - 1) {
    // pipeline 2's MMA warp allocates (and, in the epilogue, deallocates) the tensor memory: one warp owns both
    asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], %1;\n" ::"r"(su(tmem_slot)), "r"(512)
                 : "memory");
    asm volatile("tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned;\n" ::: "memory");
  }
  if (warp == 4 * kNPipe + kNPipe) {
    if constexpr (kUsePDL && !kEarly) {
      if (lane < 2) {
        // the pre-wait value, clamped to the block's own table row
        const int sl = min(max(pf_sl, 0), p.pt_cols * kPage);
        const int f0 = max(0, sl - kTokens - p.window_left);
        const int pf = f0 / kPage, pe = (sl + kPage - 1) / kPage;
        const int pp = (max(0, pe - pf) + nsplit - 1) / nsplit;
        const int q0 = min(pf + split * pp, p.pt_cols - 1);
        const int q1 = min(max(q0, pf + split * pp + pp - 1), p.pt_cols - 1);
        prefetch_l2(p.page_table + static_cast<int64_t>(req) * p.page_table_stride + (lane == 0 ? q0 : q1));
      }
    }
  }
  tc_fence_before();
  __syncthreads();
  tc_fence_after();
  const uint32_t tbase = *tmem_slot;
  // cluster: every block's barriers are initialized (and block 0's byte count set) before any block can push
  if constexpr (kCluster) {
    asm volatile("barrier.cluster.arrive.relaxed.aligned;\n" ::: "memory");
    // every thread but the copy warp (which starts loading at once and completes it after its last load) observes this
    // first phase now, while the pages are still in flight, so it may arrive again at the senders' exit barrier
    if (warp != 4 * kNPipe + kNPipe) cluster_wait_acquire();
  }
  if constexpr (kUsePDL && kEarly) {
    // the next kernel may launch now (it still waits for this grid's completion before reading anything this grid
    // writes); the copy warp issues L2 prefetches of this block's K / V boxes before griddepcontrol.wait. seq_lens
    // and the page table may still be written by the preceding kernel, so the values read here only form (bounds-
    // checked) prefetch coordinates and never reach a result; every value used is read again after the wait
    asm volatile("griddepcontrol.launch_dependents;\n" ::: "memory");
    if (warp == 4 * kNPipe + kNPipe) {
      const int sl = min(max(p.seq_lens[req], 0), p.pt_cols * kPage);
      const int f0 = max(0, sl - kTokens - p.window_left);
      const int pf = f0 / kPage, pe = (sl + kPage - 1) / kPage;
      const int pp = (max(0, pe - pf) + nsplit - 1) / nsplit;
      const int q0 = pf + split * pp;
      const int qn = max(0, min(pp, pe - q0));
      const int32_t* const ptp = p.page_table + static_cast<int64_t>(req) * p.page_table_stride;
      for (int j = lane; j < qn; j += 32) {
        const int32_t pg = ptp[min(q0 + j, p.pt_cols - 1)];
        if (pg >= 0 && static_cast<int64_t>(pg) * kPage < p.pool_slots) {
          tma_prefetch_2d(&k_map, kvh * kHeadDim, pg * kPage);
          tma_prefetch_2d(&v_map, kvh * kHeadDim, pg * kPage);
        }
      }
    }
  }

  if constexpr (kUsePDL) asm volatile("griddepcontrol.wait;\n" ::: "memory");
  const int seq_len = p.seq_lens[req];
  const int first = max(0, seq_len - kTokens - p.window_left);
  const int pg_first = first / kPage;
  const int pg_end = (seq_len + kPage - 1) / kPage;
  const int npages = max(0, pg_end - pg_first);
  const int per = (npages + nsplit - 1) / nsplit;
  const int my_pg0 = pg_first + split * per;
  const int my_np = max(0, min(per, pg_end - my_pg0));
  const int pair = req * p.num_kv_heads + kvh;
  float* const part_blk = p.part + (static_cast<int64_t>(pair) * nsplit + split) * (kRows * kRecFloats);

  float m = -INFINITY, l = 0.0f;
  if (warp < 4 * kNPipe) {
    // ---- softmax of pipeline q: lane L = 64 h + r of quarter warp % 4 ----
    const int q = warp >> 2;
    const int L = (warp & 3) * 32 + lane;
    const int h = L >> 6, r = L & 63;
    const int t = r >> 3, g = r & 7;
    const int pos = seq_len - kTokens + t;
    const int lo = max(0, pos - p.window_left);
    const uint32_t lane_off = static_cast<uint32_t>((warp & 3) * 32) << 16;
    const uint32_t cS = kT6Col * q, cO = cS + 64, cP = cS + 128;
    const uint32_t bar_s = bar_pipe0 + 48 * q, bar_sf = bar_s + 24;
    const auto bar_ob = [bar_s](int b) { return bar_s + (b ? 32u : 8u); };  // PV of the steps with parity b
    // P of the steps with parity b written: one barrier per P buffer, so with double-buffered P a softmax that runs a
    // step ahead of the MMA warp cannot complete a second phase of the barrier the MMA warp is about to wait on
    const auto bar_pb = [bar_s](int b) { return bar_s + (b ? 40u : 16u); };
    const uint64_t sc2 = f2pack(p.scale_log2, p.scale_log2);
    if (q == 0) {
      const uint4* qrow = reinterpret_cast<const uint4*>(
          p.q + static_cast<int64_t>(req * kTokens + t) * p.q_token_stride + (kvh * kQPerKv + g) * kHeadDim);
      uint32_t qv[16];
#pragma unroll
      for (int c = 0; c < 4; ++c) {
        const uint4 v = qrow[c];
        qv[4 * c] = v.x;
        qv[4 * c + 1] = v.y;
        qv[4 * c + 2] = v.z;
        qv[4 * c + 3] = v.w;
      }
      tmem_st16(tbase + lane_off + kT6ColQ, qv);
      tc_wait_st();
      tc_fence_before();
      mbar_arrive(bar_q);
    }
    const int n_q = my_np > q ? (my_np - q + kNPipe - 1) / kNPipe : 0;
    for (int i = 0; i < n_q; ++i) {
      mbar_wait(bar_s, static_cast<uint32_t>(i & 1));
      tc_fence_after();
      float sv[kHalf];
      float mt;
      static_assert(kHalf == 64, "one 64-column load per page half");
      tmem_ld64_max(tbase + lane_off + cS, sv, mt);  // the scores and their maximum
      tc_wait_ld();
      tc_fence_before();
      mbar_arrive(bar_sf);
      const int kh0 = (my_pg0 + q + kNPipe * i) * kPage + kHalf * h;
      if (!(kh0 >= lo && kh0 + kHalf - 1 <= pos)) {
        // edge half-page: mask the keys outside [lo, pos] (they may hold any bytes), then take the maximum again
        // keys kh0 + k of this half are valid for the row iff a <= k < b: one 64-bit validity mask, one bit test per key
        const int va = min(max(lo - kh0, 0), kHalf), vb = min(max(pos + 1 - kh0, 0), kHalf);
        const auto lowmask = [](int n) { return n >= 64 ? ~0ull : ((1ull << n) - 1ull); };
        const uint64_t vm = lowmask(vb) & ~lowmask(va);
        const uint32_t vm0 = static_cast<uint32_t>(vm), vm1 = static_cast<uint32_t>(vm >> 32);
#pragma unroll
        for (int k = 0; k < kHalf; ++k) {
          if (!(((k < 32) ? vm0 : vm1) & (1u << (k & 31)))) sv[k] = -INFINITY;
        }
        // exact maximum (grouping does not matter): 8 interleaved chains of three-input maxima
        float a[8];
#pragma unroll
        for (int u = 0; u < 8; ++u) {
          a[u] = fmax3(sv[u], sv[8 + u], sv[16 + u]);
          a[u] = fmax3(a[u], sv[24 + u], sv[32 + u]);
          a[u] = fmax3(a[u], sv[40 + u], sv[48 + u]);
          a[u] = fmaxf(a[u], sv[56 + u]);
        }
        mt = fmax3(fmax3(a[0], a[1], a[2]), fmax3(a[3], a[4], a[5]), fmaxf(a[6], a[7]));
      }
      mt *= p.scale_log2;
      // the running maximum m moves when the page maximum exceeds it; only then is O rescaled (alpha < 1)
      const bool grow = mt > m;  // m = -inf: any finite page maximum
      const float m_new = grow ? mt : m;
      const float mu = (m_new == -INFINITY) ? 0.0f : m_new;
      const float alpha = grow ? fast_exp2(m - mu) : 1.0f;
      const float bias = kLog2PScale - mu;
      const uint64_t bi2 = f2pack(bias, bias);
      float sa0 = 0.f, sa1 = 0.f, sa2 = 0.f, sa3 = 0.f;
      uint32_t pw[16];
#pragma unroll
      for (int w = 0; w < 16; ++w) {
        float x0, x1, x2, x3;
        fma2(sv[4 * w], sv[4 * w + 1], sc2, bi2, x0, x1);
        fma2(sv[4 * w + 2], sv[4 * w + 3], sc2, bi2, x2, x3);
        const float p0 = fast_exp2(x0), p1 = fast_exp2(x1), p2 = fast_exp2(x2), p3 = fast_exp2(x3);
        add2(sa0, sa1, p0, p1);
        add2(sa2, sa3, p2, p3);
        pw[w] = e4m3x2(p0, p1) | (e4m3x2(p2, p3) << 16);
      }
      // WAR on P: buffer i & 1 was last read by the PV of step i - 2
      if (i >= 2) {
        mbar_wait(bar_ob(i & 1), static_cast<uint32_t>(((i - 2) >> 1) & 1));
        tc_fence_after();
      }
      // a rescaling step observes every PV still outstanding on O: step i - 1's (its commit also tracks every earlier
      // MMA of the MMA warp) and step i - 2's (observed above)
      if (i > 0 && __any_sync(0xffffffffu, grow)) {
        mbar_wait(bar_ob((i - 1) & 1), static_cast<uint32_t>(((i - 1) >> 1) & 1));
        tc_fence_after();
        {
          uint32_t ox[64];
          tmem_ld64(tbase + lane_off + cO, ox);
          tc_wait_ld();
          const uint64_t al2 = f2pack(alpha, alpha);
#pragma unroll
          for (int k = 0; k < 64; k += 2) mul2(ox[k], ox[k + 1], al2);
          tmem_st64(tbase + lane_off + cO, ox);
        }
      }
      tmem_st16(tbase + lane_off + cP + 16 * (i & 1), pw);
      l = l * alpha + ((sa0 + sa1) + (sa2 + sa3));
      m = m_new;
      tc_wait_st();
      tc_fence_before();
      mbar_arrive(bar_pb(i & 1));
    }
    if constexpr (kUsePDL && !kEarly) asm volatile("griddepcontrol.launch_dependents;\n" ::: "memory");
    if (n_q > 0) mbar_wait(bar_ob((n_q - 1) & 1), static_cast<uint32_t>(((n_q - 1) >> 1) & 1));  // O is final
    // park (m, l) in the first two of this pipeline's score columns (no S MMA writes them any more: the last one was
    // read above) for the block epilogue, which reads them per lane quarter after the block barrier
    tmem_st2(tbase + lane_off + cS, m, l);
    tc_wait_st();
  } else if (warp < 4 * kNPipe + kNPipe) {
    // ---- MMA warp of pipeline q ----
    const int q = warp - 4 * kNPipe;
    const uint32_t cS = kT6Col * q, cO = cS + 64, cP = cS + 128;
    const uint32_t bar_s = bar_pipe0 + 48 * q, bar_sf = bar_s + 24;
    const auto bar_ob = [bar_s](int b) { return bar_s + (b ? 32u : 8u); };  // PV of the steps with parity b
    // P of the steps with parity b written: one barrier per P buffer, so with double-buffered P a softmax that runs a
    // step ahead of the MMA warp cannot complete a second phase of the barrier the MMA warp is about to wait on
    const auto bar_pb = [bar_s](int b) { return bar_s + (b ? 40u : 16u); };
    const int n_q = my_np > q ? (my_np - q + kNPipe - 1) / kNPipe : 0;
    // V of page jp in slot sp: wait for its box, then (edge pages only) zero the V rows of keys outside [first,
    // seq_len), which may hold any bytes incl. NaN (generic stores after the TMA completed; each storing lane fences
    // them for the async proxy before the warp sync that precedes the elected MMA issue), so 0 * V stays 0 in PV
    const auto v_ready = [&](int jp, int sp) {
      mbar_wait(bar_vload0 + 8 * sp, static_cast<uint32_t>((jp / kRing) & 1));
      const int kb = (my_pg0 + jp) * kPage;
      const int r0 = max(0, min(kPage, first - kb)), r1 = max(0, min(kPage, seq_len - kb));
      if (r0 > 0 || r1 < kPage) {
        uint8_t* const vt = smem + sp * kPageBytes + kPage * kHeadDim;
        for (int idx = lane; idx < kPage * 4; idx += 32) {
          const int row = idx >> 2;
          if (row < r0 || row >= r1) *reinterpret_cast<uint4*>(vt + row * 64 + (idx & 3) * 16) = make_uint4(0u, 0u, 0u, 0u);
        }
        fence_proxy_async();
        __syncwarp();
      }
    };
    mbar_wait(bar_q, 0);
    tc_fence_after();
    for (int i = 0; i < n_q; ++i) {
      const int j = q + kNPipe * i;
      const int slot = j % kRing;
      mbar_wait(bar_loaded0 + 8 * slot, static_cast<uint32_t>((j / kRing) & 1));  // K of page j
      if (i > 0) {
        mbar_wait(bar_sf, static_cast<uint32_t>((i - 1) & 1));
        tc_fence_after();
      }
      mma_s_sw(tbase + cS, tbase + kT6ColQ, sdesc64(ring + slot * kPageBytes));
      commit(bar_s);
      if (i > 0) {
        const int jp = j - kNPipe, sp = jp % kRing;
        v_ready(jp, sp);
        mbar_wait(bar_pb((i - 1) & 1), static_cast<uint32_t>(((i - 1) >> 1) & 1));
        tc_fence_after();
        mma_pv(tbase + cO, tbase + cP + 16 * ((i - 1) & 1), sdesc64(ring + sp * kPageBytes + kPage * kHeadDim),
               i - 1 > 0 ? 1u : 0u);
        commit(bar_ob((i - 1) & 1));
        if (jp + kRing < my_np) commit(bar_free0 + 8 * sp);  // only slots the copy warp refills
      }
    }
    if (n_q > 0) {
      const int jp = q + kNPipe * (n_q - 1), sp = jp % kRing;
      v_ready(jp, sp);
      mbar_wait(bar_pb((n_q - 1) & 1), static_cast<uint32_t>(((n_q - 1) >> 1) & 1));
      tc_fence_after();
      mma_pv(tbase + cO, tbase + cP + 16 * ((n_q - 1) & 1), sdesc64(ring + sp * kPageBytes + kPage * kHeadDim),
             n_q > 1 ? 1u : 0u);
      commit(bar_ob((n_q - 1) & 1));
      if (jp + kRing < my_np) commit(bar_free0 + 8 * sp);
    }
    // no free-slot commit is issued for the block's last kRing pages (no slot is refilled after them), so every
    // commit arrival of the block is waited on by some thread before the block's final barrier: bar_o by the softmax
    // threads, free[] by the copy warp
  } else {
    // ---- copy warp: keeps the ring full with TMA page loads (K and V boxes of 128 rows x 64 bytes of this KV head) ----
    if (warp == 4 * kNPipe + kNPipe) {
      // the page-table entries of 32 pages at a time are read by the 32 lanes at once (one round trip instead of one
      // per page before its load can issue), then broadcast per page
      const int32_t* const pt = p.page_table + static_cast<int64_t>(req) * p.page_table_stride;
      int32_t pg_lane = 0, pg_prev = 0;
      for (int t = 0; t < my_np + kT6VLag; ++t) {
        if (t < my_np && (t & 31) == 0) {
          pg_prev = pg_lane;
          pg_lane = t + lane < my_np ? pt[my_pg0 + t + lane] : 0;
        }
        if (t < my_np) {
          // K of page t (its slot is free once the PV of page t - kRing completed, which read both of its boxes)
          const int32_t pg = __shfl_sync(0xffffffffu, pg_lane, t & 31);
          const int slot = t % kRing;
          if (t >= kRing) mbar_wait(bar_free0 + 8 * slot, static_cast<uint32_t>(((t / kRing) - 1) & 1));
          if (lane == 0) {
            const uint32_t bar = bar_loaded0 + 8 * slot;
            mbar_expect(bar, kPageBytes / 2);
            tma_load_2d(ring + slot * kPageBytes, &k_map, kvh * kHeadDim, pg * kPage, bar);
          }
        }
        const int jv = t - kT6VLag;
        if (jv >= 0 && jv < my_np) {
          // V of page jv, into the slot its K box went to
          // pg_lane holds the 32-entry chunk last read (that of min(t, my_np - 1)); jv is at most kT6VLag behind it
          const bool cur = (jv >> 5) == (min(t, my_np - 1) >> 5);
          const int32_t pg = __shfl_sync(0xffffffffu, cur ? pg_lane : pg_prev, jv & 31);
          const int slot = jv % kRing;
          if (lane == 0) {
            const uint32_t bar = bar_vload0 + 8 * slot;
            mbar_expect(bar, kPageBytes / 2);
            tma_load_2d(ring + slot * kPageBytes + kPage * kHeadDim, &v_map, kvh * kHeadDim, pg * kPage, bar);
          }
        }
        __syncwarp();
      }
      // cluster: this warp (idle from here) completes the cluster barrier for the block, off the pipelines' path;
      // the epilogue's bar.sync orders it before any of this block's pushes into block 0
      if constexpr (kCluster) asm volatile("barrier.cluster.wait.acquire.aligned;\n" ::: "memory");
    }
  }

  // ---- block merge on all 16 warps: warp w takes tensor-memory lane quarter w % 4 (lane L = 64 h + r) and O columns
  // 16 (w / 4) .. +15; the 3 pipelines merge in registers (pipeline order), the two key halves of a row through
  // shared memory; one block per pair writes the output, otherwise the block's rows go to brec ----
  float* const xh = reinterpret_cast<float*>(smem + 4096);           // [64 rows][kRecFloats] key half 1, merged
  float* const brec = reinterpret_cast<float*>(smem + kT6BlockRec);
  static_assert(kT6Threads == 16 * 32 && kNPipe == 3, "epilogue: 4 column groups x 4 lane quarters, 3 pipelines");
  tc_fence_before();
  __syncthreads();  // every pipeline's last PV has completed (each softmax warp waited on it); MMAs are done
  fence_proxy_async();  // generic writes into the ring (key half 1 rows below) after the MMAs' async-proxy reads
  {
    tc_fence_after();
    const int cg = warp >> 2;
    const int L = (warp & 3) * 32 + lane;
    const int h = L >> 6, r = L & 63;
    const uint32_t lane_off = static_cast<uint32_t>((warp & 3) * 32) << 16;
    float mq[kNPipe], lq[kNPipe], wq[kNPipe];
    float mx = -INFINITY, lsum = 0.0f;
#pragma unroll
    for (int q = 0; q < kNPipe; ++q) tmem_ld2(tbase + lane_off + kT6Col * q, mq[q], lq[q]);
    uint32_t t0[16], t1[16], t2[16];
    tmem_ld16(tbase + lane_off + 64 + 16 * cg, t0);
    tmem_ld16(tbase + lane_off + kT6Col + 64 + 16 * cg, t1);
    tmem_ld16(tbase + lane_off + 2 * kT6Col + 64 + 16 * cg, t2);
    tc_wait_ld();
#pragma unroll
    for (int q = 0; q < kNPipe; ++q) mx = fmaxf(mx, mq[q]);
#pragma unroll
    for (int q = 0; q < kNPipe; ++q) {
      wq[q] = (mq[q] == -INFINITY) ? 0.0f : fast_exp2(mq[q] - mx);
      lsum += wq[q] * lq[q];
    }
    float o[16];
#pragma unroll
    for (int k = 0; k < 16; ++k) {
      o[k] = 0.0f + wq[0] * __uint_as_float(t0[k]);
      o[k] = o[k] + wq[1] * __uint_as_float(t1[k]);
      o[k] = o[k] + wq[2] * __uint_as_float(t2[k]);
    }
    if (h == 1) {
      float* const xr = xh + r * kRecFloats;
#pragma unroll
      for (int k = 0; k < 16; k += 4)
        *reinterpret_cast<float4*>(xr + 16 * cg + k) = make_float4(o[k], o[k + 1], o[k + 2], o[k + 3]);
      if (cg == 0) *reinterpret_cast<float2*>(xr + 64) = make_float2(mx, lsum);
    }
    tc_fence_before();
    __syncthreads();  // key half 1 in xh; every tensor-memory read of the block is done
    // the deallocation runs on pipeline 2's MMA warp (the allocating warp; a key-half-1 warp with nothing left to do),
    // off the path of warp 0, which pushes the record (blocks 1..S-1) or merges (block 0)
    if (warp == 4 * kNPipe + kNPipe - 1) {
      tc_fence_after();
      asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, %1;\n" ::"r"(tbase), "r"(512) : "memory");
    }
    if (h == 0) {
      const float* const xr = xh + r * kRecFloats;
      const float2 ml1 = *reinterpret_cast<const float2*>(xr + 64);
      const float mm = fmaxf(mx, ml1.x);
      const float w0 = mx == -INFINITY ? 0.0f : fast_exp2(mx - mm);
      const float w1 = ml1.x == -INFINITY ? 0.0f : fast_exp2(ml1.x - mm);
#pragma unroll
      for (int k = 0; k < 16; k += 4) {
        const float4 v = *reinterpret_cast<const float4*>(xr + 16 * cg + k);
        o[k] = o[k] * w0 + v.x * w1;
        o[k + 1] = o[k + 1] * w0 + v.y * w1;
        o[k + 2] = o[k + 2] * w0 + v.z * w1;
        o[k + 3] = o[k + 3] * w0 + v.w * w1;
      }
      const float lt = lsum * w0 + ml1.y * w1;
      if (nsplit == 1) {
        const float sc = lt > 0.0f ? p.out_scale / lt : 0.0f;
        const int t = r >> 3, g = r & 7;
        uint4* const dst = reinterpret_cast<uint4*>(
            p.out + (static_cast<int64_t>(req * kTokens + t) * p.num_q_heads + kvh * kQPerKv + g) * kHeadDim + 16 * cg);
#pragma unroll
        for (int c = 0; c < 2; ++c) {
          uint32_t pk[4];
#pragma unroll
          for (int k = 0; k < 4; ++k) {
            __nv_bfloat162 b2 = __floats2bfloat162_rn(o[8 * c + 2 * k] * sc, o[8 * c + 2 * k + 1] * sc);
            pk[k] = *reinterpret_cast<uint32_t*>(&b2);
          }
          dst[c] = make_uint4(pk[0], pk[1], pk[2], pk[3]);
        }
      } else if (kCluster && split > 0) {
        // blocks 1..S-1: stage this row's 16 columns (and m, l, 2 pad from column group 0) in this block's own (unused)
        // receive area with generic stores, fenced for the async proxy; after the key-half-0 threads' barrier one thread
        // pushes the 64-row record into block 0's slot split - 1 with one bulk copy (complete-tx in bytes on block 0's
        // recv barrier); nothing in this block is read remotely, and the record stays valid until block 0 has received it
        // (exit barrier below)
        float* const sr = reinterpret_cast<float*>(smem + kT6Recv) + r * kRecFloats;
#pragma unroll
        for (int k = 0; k < 16; k += 4)
          *reinterpret_cast<float4*>(sr + 16 * cg + k) = make_float4(o[k], o[k + 1], o[k + 2], o[k + 3]);
        if (cg == 0) *reinterpret_cast<float4*>(sr + 64) = make_float4(mm, lt, 0.0f, 0.0f);
        fence_proxy_async();
        asm volatile("bar.sync 2, 256;\n" ::: "memory");  // the 256 key-half-0 threads (8 whole warps)
        if (warp == 0 && lane == 0) {
          if (p.test_sender_delay_ns != 0u && split == 1) {
            for (uint32_t w = 0; w < p.test_sender_delay_ns; w += 1000u) __nanosleep(1000u);
          }
          const uint32_t dsta =
              map_rank(su(smem + kT6Recv) + static_cast<uint32_t>((split - 1) * kRows * kT6RecvRow), 0);
          const uint32_t rb = map_rank(bar_recv, 0);
          asm volatile(
              "cp.async.bulk.shared::cluster.shared::cta.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];\n" ::"r"(dsta),
              "r"(su(smem + kT6Recv)), "r"(static_cast<uint32_t>(kRows * kT6RecvRow)), "r"(rb)
              : "memory");
        }
        __syncwarp();
        // a sender stays until block 0 has received every pushed byte (exit barrier below)
        cluster_arrive_release();
      } else if (kCluster) {
        // block 0: once every pushed byte has landed, merge this row's 16 columns over the S blocks in block order
        // (its own from registers first) and write them
        mbar_wait_cluster(bar_recv, 0);
        cluster_arrive_relaxed();  // every pushed byte has landed: the senders may exit
        float ms[kT6MaxClSplits], ls[kT6MaxClSplits];
        ms[0] = mm;
        ls[0] = lt;
        float mx2 = mm;
#pragma unroll
        for (int s2 = 1; s2 < kT6MaxClSplits; ++s2) {
          const float2 v = s2 < nsplit ? *reinterpret_cast<const float2*>(
                                             reinterpret_cast<const float*>(smem + kT6Recv) +
                                             ((s2 - 1) * kRows + r) * kRecFloats + 64)
                                       : make_float2(-INFINITY, 0.0f);
          ms[s2] = v.x;
          ls[s2] = v.y;
          mx2 = fmaxf(mx2, v.x);
        }
        float lsum2 = 0.0f;
        if (mx2 == -INFINITY) {
#pragma unroll
          for (int k = 0; k < 16; ++k) o[k] = 0.0f;
        } else {
          const float w0s = ms[0] == -INFINITY ? 0.0f : fast_exp2(ms[0] - mx2);
#pragma unroll
          for (int k = 0; k < 16; ++k) o[k] = 0.0f + w0s * o[k];
          lsum2 = w0s * ls[0];
#pragma unroll
          for (int s2 = 1; s2 < kT6MaxClSplits; ++s2) {
            if (s2 < nsplit && ms[s2] != -INFINITY) {
              const float w = fast_exp2(ms[s2] - mx2);
              const float* const src =
                  reinterpret_cast<const float*>(smem + kT6Recv) + ((s2 - 1) * kRows + r) * kRecFloats + 16 * cg;
#pragma unroll
              for (int k = 0; k < 16; k += 4) {
                const float4 v = *reinterpret_cast<const float4*>(src + k);
                o[k] += w * v.x;
                o[k + 1] += w * v.y;
                o[k + 2] += w * v.z;
                o[k + 3] += w * v.w;
              }
              lsum2 += w * ls[s2];
            }
          }
        }
        const float sc = lsum2 > 0.0f ? p.out_scale / lsum2 : 0.0f;
        const int t = r >> 3, g = r & 7;
        uint4* const dst = reinterpret_cast<uint4*>(
            p.out + (static_cast<int64_t>(req * kTokens + t) * p.num_q_heads + kvh * kQPerKv + g) * kHeadDim + 16 * cg);
#pragma unroll
        for (int c = 0; c < 2; ++c) {
          uint32_t pk[4];
#pragma unroll
          for (int k = 0; k < 4; ++k) {
            __nv_bfloat162 b2 = __floats2bfloat162_rn(o[8 * c + 2 * k] * sc, o[8 * c + 2 * k + 1] * sc);
            pk[k] = *reinterpret_cast<uint32_t*>(&b2);
          }
          dst[c] = make_uint4(pk[0], pk[1], pk[2], pk[3]);
        }
      } else {
        float* const rr = brec + r * kRecFloats;
#pragma unroll
        for (int k = 0; k < 16; k += 4)
          *reinterpret_cast<float4*>(rr + 16 * cg + k) = make_float4(o[k], o[k + 1], o[k + 2], o[k + 3]);
        if (cg == 0) *reinterpret_cast<float4*>(rr + 64) = make_float4(mm, lt, 0.0f, 0.0f);
      }
    }
    if constexpr (kCluster) {
      if (h == 1) cluster_arrive_release();  // threads that push / merge nothing
    }
  }
  if (nsplit == 1) return;
  if constexpr (kCluster) {
    // pushed (blocks 1..S-1) or merged and written (block 0) above; no thread exits before block 0 has received
    // every pushed byte (block 0 arrives after its receive wait)
    cluster_wait_acquire();
    return;
  }

  // ---- S > 1: arrival-first split-K merge (as t5) ----
  __syncthreads();
  const int npairs = static_cast<int>(gridDim.y * gridDim.z);
  if (tid == 0) {
    uint32_t old;
    asm volatile("atom.relaxed.gpu.global.add.u32 %0, [%1], 1;\n" : "=r"(old) : "l"(p.count + pair) : "memory");
    *last_flag = old == static_cast<uint32_t>(nsplit - 1);
  }
  __syncthreads();
  if (*last_flag == 0u) {
    for (int i = tid; i < kRows * kRecFloats / 4; i += kT6Threads)
      reinterpret_cast<float4*>(part_blk)[i] = reinterpret_cast<const float4*>(brec)[i];
    __syncthreads();
    if (tid == 0)
      asm volatile("red.release.gpu.global.add.u32 [%0], 1;\n" ::"l"(p.count + npairs + pair) : "memory");
    return;
  }
  if (tid == 0) {
    uint32_t polls = 0;
    while (true) {
      uint32_t ready;
      asm volatile("ld.acquire.gpu.global.u32 %0, [%1];\n" : "=r"(ready) : "l"(p.count + npairs + pair) : "memory");
      if (ready == static_cast<uint32_t>(nsplit - 1)) break;
      if (++polls == kMaxPolls) [[unlikely]] t5_wait_bound_exceeded();
    }
  }
  __syncthreads();
  if (tid < 256) {
    const int r = tid >> 2, q4 = tid & 3;
    const float* const part_pair = p.part + static_cast<int64_t>(pair) * nsplit * (kRows * kRecFloats);
    float ms[kMaxSplits];
#pragma unroll
    for (int s = 0; s < kMaxSplits; ++s)
      ms[s] = s >= nsplit ? -INFINITY
              : s == split ? brec[r * kRecFloats + 64]
                           : __ldcg(part_pair + (s * kRows + r) * kRecFloats + 64);
    float mx = -INFINITY;
#pragma unroll
    for (int s = 0; s < kMaxSplits; ++s) mx = fmaxf(mx, ms[s]);
    float acc[16];
#pragma unroll
    for (int k = 0; k < 16; ++k) acc[k] = 0.0f;
    float lsum = 0.0f;
    if (mx != -INFINITY) {
      for (int s = 0; s < nsplit; ++s) {
        if (ms[s] == -INFINITY) continue;
        const float w = fast_exp2(ms[s] - mx);
        const float* const rr = s == split ? brec + r * kRecFloats : part_pair + (s * kRows + r) * kRecFloats;
#pragma unroll
        for (int c = 0; c < 4; ++c) {
          const float4 v = s == split ? *reinterpret_cast<const float4*>(rr + 16 * q4 + 4 * c)
                                      : __ldcg(reinterpret_cast<const float4*>(rr + 16 * q4 + 4 * c));
          acc[4 * c] += w * v.x;
          acc[4 * c + 1] += w * v.y;
          acc[4 * c + 2] += w * v.z;
          acc[4 * c + 3] += w * v.w;
        }
        lsum += w * (s == split ? rr[65] : __ldcg(rr + 65));
      }
    }
    const float sc = lsum > 0.0f ? p.out_scale / lsum : 0.0f;
    const int t = r >> 3, g = r & 7;
    uint32_t packed[8];
#pragma unroll
    for (int k = 0; k < 8; ++k) {
      __nv_bfloat162 b2 = __floats2bfloat162_rn(acc[2 * k] * sc, acc[2 * k + 1] * sc);
      packed[k] = *reinterpret_cast<uint32_t*>(&b2);
    }
    uint4* const dst = reinterpret_cast<uint4*>(
        p.out + (static_cast<int64_t>(req * kTokens + t) * p.num_q_heads + kvh * kQPerKv + g) * kHeadDim + 16 * q4);
    dst[0] = make_uint4(packed[0], packed[1], packed[2], packed[3]);
    dst[1] = make_uint4(packed[4], packed[5], packed[6], packed[7]);
    if (tid == 0) {
      p.count[pair] = 0u;
      p.count[npairs + pair] = 0u;
    }
  }
}

}  // namespace swa_t5

template <int kPages, bool kUsePDL, bool kPrefetch>
struct SwaVerifyT5Kernel {
  static void run(const tvm::ffi::TensorView q,           // [R*8, Hq, 64] e4m3, heads/dims contiguous
                  const tvm::ffi::TensorView k_pool,      // [slots, Hkv*64] e4m3, rows contiguous
                  const tvm::ffi::TensorView v_pool,      // [slots, Hkv*64]
                  const tvm::ffi::TensorView page_table,  // [R, pages] int32, inner contiguous
                  const tvm::ffi::TensorView seq_lens,    // [R] int32
                  const tvm::ffi::TensorView out,         // [R*8, Hq, 64] bf16, contiguous
                  const tvm::ffi::TensorView part,        // fp32 workspace, >= R*Hkv*S*64*68 floats
                  const tvm::ffi::TensorView count,       // int32 [>= R*Hkv], zero between launches
                  int64_t window_left, int64_t page_size, double bmm1_scale, double bmm2_scale, int64_t splits) {
    using namespace sglang::host;
    using C = swa_t5::Cfg<kPages>;
    RuntimeCheck(k_pool.dtype().code == DLDataTypeCode::kDLFloat8_e4m3fn &&
                     v_pool.dtype().code == DLDataTypeCode::kDLFloat8_e4m3fn &&
                     q.dtype().code == DLDataTypeCode::kDLFloat8_e4m3fn,
                 "q/k/v must be float8_e4m3fn");
    RuntimeCheck(out.dtype().code == DLDataTypeCode::kDLBfloat && out.dtype().bits == 16, "out must be bf16");
    RuntimeCheck(page_table.dtype().code == DLDataTypeCode::kDLInt && page_table.dtype().bits == 32 &&
                     seq_lens.dtype().code == DLDataTypeCode::kDLInt && seq_lens.dtype().bits == 32,
                 "page_table/seq_lens must be int32");
    RuntimeCheck(out.dim() == 3 && out.size(2) == swa_t5::kHeadDim, "out must be [tokens, heads, 64]");
    const int64_t m = out.size(0), hq = out.size(1);
    RuntimeCheck(m % swa_t5::kTokens == 0, "token count must be a multiple of 8");
    const int64_t reqs = m / swa_t5::kTokens;
    RuntimeCheck(seq_lens.dim() == 1 && seq_lens.size(0) == reqs && seq_lens.stride(0) == 1, "seq_lens [R]");
    RuntimeCheck(page_table.dim() == 2 && page_table.size(0) >= reqs && page_table.stride(1) == 1, "page_table");
    RuntimeCheck(out.stride(2) == 1 && out.stride(1) == swa_t5::kHeadDim && out.stride(0) == hq * swa_t5::kHeadDim,
                 "out layout");
    RuntimeCheck(k_pool.dim() == 2 && v_pool.dim() == 2 && k_pool.stride(1) == 1 && v_pool.stride(1) == 1 &&
                     k_pool.stride(0) == v_pool.stride(0) && k_pool.size(0) == v_pool.size(0) &&
                     k_pool.size(1) == v_pool.size(1),
                 "pools must be [slots, row] with unit inner stride");
    const int64_t hkv = k_pool.size(1) / swa_t5::kHeadDim;
    RuntimeCheck(hkv * swa_t5::kHeadDim == k_pool.size(1) && hkv * swa_t5::kQPerKv == hq,
                 "expects 8 query heads per KV head");
    RuntimeCheck(page_size == swa_t5::kPage, "page size must be 128");
    RuntimeCheck(window_left >= 0 && splits >= 1 && splits <= swa_t5::kMaxSplits, "window / splits");
    const int64_t max_pages = (window_left + swa_t5::kTokens + swa_t5::kPage - 1) / swa_t5::kPage + 1;
    RuntimeCheck((max_pages + splits - 1) / splits <= kPages, "pages per block ", (max_pages + splits - 1) / splits,
                 " exceed the build's ", kPages);
    RuntimeCheck(q.dim() == 3 && q.size(0) == m && q.size(1) == hq && q.size(2) == swa_t5::kHeadDim &&
                     q.stride(2) == 1 && q.stride(1) == swa_t5::kHeadDim && q.stride(0) % 16 == 0,
                 "q must be [tokens, heads, 64], heads and dims contiguous, 16-byte aligned rows");
    RuntimeCheck(part.dtype().code == DLDataTypeCode::kDLFloat && part.dtype().bits == 32 &&
                     part.numel() >= reqs * hkv * splits * swa_t5::kRows * swa_t5::kRecFloats,
                 "partial workspace too small");
    RuntimeCheck(count.dtype().code == DLDataTypeCode::kDLInt && count.dtype().bits == 32 &&
                     count.numel() >= 2 * reqs * hkv,
                 "counter workspace too small");
    const auto aligned16 = [](const void* ptr) { return reinterpret_cast<uintptr_t>(ptr) % 16 == 0; };
    RuntimeCheck(aligned16(k_pool.data_ptr()) && aligned16(v_pool.data_ptr()) && aligned16(out.data_ptr()) &&
                     aligned16(q.data_ptr()) && aligned16(part.data_ptr()) && k_pool.stride(0) % 16 == 0,
                 "16-byte alignment");

    swa_t5::Params params{};
    params.q = static_cast<const uint8_t*>(q.data_ptr());
    params.k_pool = static_cast<const uint8_t*>(k_pool.data_ptr());
    params.v_pool = static_cast<const uint8_t*>(v_pool.data_ptr());
    params.page_table = static_cast<const int32_t*>(page_table.data_ptr());
    params.seq_lens = static_cast<const int32_t*>(seq_lens.data_ptr());
    params.out = static_cast<__nv_bfloat16*>(out.data_ptr());
    params.part = static_cast<float*>(part.data_ptr());
    params.count = static_cast<uint32_t*>(count.data_ptr());
    params.pool_row_stride = k_pool.stride(0);
    params.pool_slots = k_pool.size(0);
    params.pt_cols = static_cast<int32_t>(page_table.size(1));
    params.page_table_stride = page_table.stride(0);
    params.q_token_stride = q.stride(0);
    params.num_q_heads = static_cast<int32_t>(hq);
    params.num_kv_heads = static_cast<int32_t>(hkv);
    params.window_left = static_cast<int32_t>(window_left);
    params.scale_log2 = static_cast<float>(bmm1_scale * M_LOG2E);
    params.out_scale = static_cast<float>(bmm2_scale);

    constexpr auto kernel = swa_t5::swa_t5_kernel<kPages, kUsePDL, kPrefetch>;
    static bool configured = false;
    if (!configured) {
      RuntimeDeviceCheck(::cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, C::kBytes));
      configured = true;
    }
    if (reqs == 0) return;
    const dim3 grid(static_cast<unsigned>(splits), static_cast<unsigned>(hkv), static_cast<unsigned>(reqs));
    LaunchKernel(grid, swa_t5::kThreads, out.device(), C::kBytes).config({.use_pdl = kUsePDL})(kernel, params);
  }
};

namespace swa_t5_host {
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
// pool [slots, row_bytes] (bytes), box = 64 bytes (one KV head) x 128 rows (one page), 64-byte swizzle
inline CUtensorMap pool_map64(const void* base, int64_t slots, int64_t row_bytes) {
  CUtensorMap map;
  const cuuint64_t dims[2] = {static_cast<cuuint64_t>(row_bytes), static_cast<cuuint64_t>(slots)};
  const cuuint64_t strides[1] = {static_cast<cuuint64_t>(row_bytes)};
  const cuuint32_t box[2] = {64, 128};
  const cuuint32_t elem[2] = {1, 1};
  const CUresult r = encode_fn()(&map, CU_TENSOR_MAP_DATA_TYPE_UINT8, 2, const_cast<void*>(base), dims, strides, box,
                                 elem, CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_64B,
                                 CU_TENSOR_MAP_L2_PROMOTION_L2_128B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
  sglang::host::RuntimeCheck(r == CUDA_SUCCESS, "cuTensorMapEncodeTiled failed: ", static_cast<int>(r));
  return map;
}
}  // namespace swa_t5_host

template <bool kUsePDL, bool kCluster, bool kEarly>
struct SwaVerifyT6Kernel {
  // blocks of this kernel that can be resident on the device at once with `splits` blocks per cluster (cluster
  // instantiations) or without clusters; written into out[0] (int32, host)
  static void capacity(const tvm::ffi::TensorView out, int64_t splits) {
    using namespace sglang::host;
    RuntimeCheck(out.dtype().code == DLDataTypeCode::kDLInt && out.dtype().bits == 32 && out.numel() >= 1 &&
                     out.device().device_type == kDLCPU,
                 "capacity output must be a host int32 tensor");
    RuntimeCheck(splits >= 1 && splits <= swa_t5::kT6MaxClSplits, "splits");
    constexpr auto kernel = swa_t5::swa_t6_kernel<kUsePDL, kCluster, kEarly>;
    const int smem = kCluster ? swa_t5::kT6BytesCl : swa_t5::kT6Bytes;
    RuntimeDeviceCheck(::cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem));
    int dev = 0, sms = 0, n = 0;
    RuntimeDeviceCheck(::cudaGetDevice(&dev));
    RuntimeDeviceCheck(::cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, dev));
    if (kCluster && splits > 1) {
      cudaLaunchConfig_t cfg{};
      cfg.gridDim = dim3(static_cast<unsigned>(splits), 1, static_cast<unsigned>(sms));
      cfg.blockDim = dim3(swa_t5::kT6Threads, 1, 1);
      cfg.dynamicSmemBytes = static_cast<size_t>(smem);
      cudaLaunchAttribute attr[1];
      attr[0].id = cudaLaunchAttributeClusterDimension;
      attr[0].val.clusterDim.x = static_cast<unsigned>(splits);
      attr[0].val.clusterDim.y = 1;
      attr[0].val.clusterDim.z = 1;
      cfg.attrs = attr;
      cfg.numAttrs = 1;
      RuntimeDeviceCheck(::cudaOccupancyMaxActiveClusters(&n, kernel, &cfg));
      n *= static_cast<int>(splits);
    } else {
      RuntimeDeviceCheck(::cudaOccupancyMaxActiveBlocksPerMultiprocessor(&n, kernel, swa_t5::kT6Threads, smem));
      n *= sms;
    }
    static_cast<int32_t*>(out.data_ptr())[0] = n;
  }

  static void run(const tvm::ffi::TensorView q,           // [R*8, Hq, 64] e4m3, heads/dims contiguous
                  const tvm::ffi::TensorView k_pool,      // [slots, Hkv*64] e4m3, rows contiguous
                  const tvm::ffi::TensorView v_pool,      // [slots, Hkv*64]
                  const tvm::ffi::TensorView page_table,  // [R, pages] int32, inner contiguous
                  const tvm::ffi::TensorView seq_lens,    // [R] int32
                  const tvm::ffi::TensorView out,         // [R*8, Hq, 64] bf16, contiguous
                  const tvm::ffi::TensorView part,        // fp32 workspace, >= R*Hkv*S*64*68 floats
                  const tvm::ffi::TensorView count,       // int32 [>= R*Hkv], zero between launches
                  int64_t window_left, int64_t page_size, double bmm1_scale, double bmm2_scale, int64_t splits) {
    using namespace sglang::host;
    RuntimeCheck(k_pool.dtype().code == DLDataTypeCode::kDLFloat8_e4m3fn &&
                     v_pool.dtype().code == DLDataTypeCode::kDLFloat8_e4m3fn &&
                     q.dtype().code == DLDataTypeCode::kDLFloat8_e4m3fn,
                 "q/k/v must be float8_e4m3fn");
    RuntimeCheck(out.dtype().code == DLDataTypeCode::kDLBfloat && out.dtype().bits == 16, "out must be bf16");
    RuntimeCheck(page_table.dtype().code == DLDataTypeCode::kDLInt && page_table.dtype().bits == 32 &&
                     seq_lens.dtype().code == DLDataTypeCode::kDLInt && seq_lens.dtype().bits == 32,
                 "page_table/seq_lens must be int32");
    RuntimeCheck(out.dim() == 3 && out.size(2) == swa_t5::kHeadDim, "out must be [tokens, heads, 64]");
    const int64_t m = out.size(0), hq = out.size(1);
    RuntimeCheck(m % swa_t5::kTokens == 0, "token count must be a multiple of 8");
    const int64_t reqs = m / swa_t5::kTokens;
    RuntimeCheck(seq_lens.dim() == 1 && seq_lens.size(0) == reqs && seq_lens.stride(0) == 1, "seq_lens [R]");
    RuntimeCheck(page_table.dim() == 2 && page_table.size(0) >= reqs && page_table.stride(1) == 1, "page_table");
    RuntimeCheck(page_table.size(1) >= 1 &&
                     page_table.size(1) <= (int64_t{2147483647} - swa_t5::kPage) / swa_t5::kPage,
                 "page_table width must be in [1, (INT32_MAX - 128) / 128]");
    RuntimeCheck(out.stride(2) == 1 && out.stride(1) == swa_t5::kHeadDim && out.stride(0) == hq * swa_t5::kHeadDim,
                 "out layout");
    RuntimeCheck(k_pool.dim() == 2 && v_pool.dim() == 2 && k_pool.stride(1) == 1 && v_pool.stride(1) == 1 &&
                     k_pool.stride(0) == v_pool.stride(0) && k_pool.size(0) == v_pool.size(0) &&
                     k_pool.size(1) == v_pool.size(1),
                 "pools must be [slots, row] with unit inner stride");
    const int64_t hkv = k_pool.size(1) / swa_t5::kHeadDim;
    RuntimeCheck(hkv * swa_t5::kHeadDim == k_pool.size(1) && hkv * swa_t5::kQPerKv == hq,
                 "expects 8 query heads per KV head");
    RuntimeCheck(page_size == swa_t5::kPage, "page size must be 128");
    RuntimeCheck(window_left >= 0 && splits >= 1 && splits <= swa_t5::kMaxSplits, "window / splits");
    RuntimeCheck(q.dim() == 3 && q.size(0) == m && q.size(1) == hq && q.size(2) == swa_t5::kHeadDim &&
                     q.stride(2) == 1 && q.stride(1) == swa_t5::kHeadDim && q.stride(0) % 16 == 0,
                 "q must be [tokens, heads, 64], heads and dims contiguous, 16-byte aligned rows");
    RuntimeCheck(part.dtype().code == DLDataTypeCode::kDLFloat && part.dtype().bits == 32 &&
                     part.numel() >= reqs * hkv * splits * swa_t5::kRows * swa_t5::kRecFloats,
                 "partial workspace too small");
    RuntimeCheck(count.dtype().code == DLDataTypeCode::kDLInt && count.dtype().bits == 32 &&
                     count.numel() >= 2 * reqs * hkv,
                 "counter workspace too small");
    const auto aligned16 = [](const void* ptr) { return reinterpret_cast<uintptr_t>(ptr) % 16 == 0; };
    RuntimeCheck(aligned16(k_pool.data_ptr()) && aligned16(v_pool.data_ptr()) && aligned16(out.data_ptr()) &&
                     aligned16(q.data_ptr()) && aligned16(part.data_ptr()) && k_pool.stride(0) % 16 == 0,
                 "16-byte alignment");

    swa_t5::Params params{};
    params.q = static_cast<const uint8_t*>(q.data_ptr());
    params.k_pool = static_cast<const uint8_t*>(k_pool.data_ptr());
    params.v_pool = static_cast<const uint8_t*>(v_pool.data_ptr());
    params.page_table = static_cast<const int32_t*>(page_table.data_ptr());
    params.seq_lens = static_cast<const int32_t*>(seq_lens.data_ptr());
    params.out = static_cast<__nv_bfloat16*>(out.data_ptr());
    params.part = static_cast<float*>(part.data_ptr());
    params.count = static_cast<uint32_t*>(count.data_ptr());
    params.pool_row_stride = k_pool.stride(0);
    params.pool_slots = k_pool.size(0);
    params.pt_cols = static_cast<int32_t>(page_table.size(1));
    params.page_table_stride = page_table.stride(0);
    params.q_token_stride = q.stride(0);
    params.num_q_heads = static_cast<int32_t>(hq);
    params.num_kv_heads = static_cast<int32_t>(hkv);
    params.window_left = static_cast<int32_t>(window_left);
    params.scale_log2 = static_cast<float>(bmm1_scale * M_LOG2E);
    params.out_scale = static_cast<float>(bmm2_scale);
    {
      const char* d = std::getenv("MUSE_VR_SWA_T6_TEST_SENDER_DELAY_NS");  // tests only
      params.test_sender_delay_ns = d ? static_cast<uint32_t>(std::strtoul(d, nullptr, 10)) : 0u;
      const char* e = std::getenv("MUSE_VR_SWA_T6_TEST_PF_SEQ");  // tests only: planted pre-wait seq_lens value
      params.test_pf_seq_force = e ? 1 : 0;
      params.test_pf_seq_value = e ? static_cast<int32_t>(std::strtol(e, nullptr, 10)) : 0;
    }

    constexpr auto kernel = swa_t5::swa_t6_kernel<kUsePDL, kCluster, kEarly>;
    const CUtensorMap k_map = swa_t5_host::pool_map64(k_pool.data_ptr(), k_pool.size(0), k_pool.stride(0));
    const CUtensorMap v_map = swa_t5_host::pool_map64(v_pool.data_ptr(), v_pool.size(0), v_pool.stride(0));
    static bool configured = false;
    if (!configured) {
      RuntimeDeviceCheck(::cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
                                                kCluster ? swa_t5::kT6BytesCl : swa_t5::kT6Bytes));
      configured = true;
    }
    if (reqs == 0) return;
    const dim3 grid(static_cast<unsigned>(splits), static_cast<unsigned>(hkv), static_cast<unsigned>(reqs));
    if constexpr (kCluster) {
      RuntimeCheck(splits >= 2 && splits <= swa_t5::kT6MaxClSplits, "cluster merge needs 2..3 blocks per pair");
      LaunchKernel(grid, swa_t5::kT6Threads, out.device(), swa_t5::kT6BytesCl)
          .config({.use_pdl = kUsePDL, .cluster_dim = dim3(static_cast<unsigned>(splits), 1, 1)})(kernel, params, k_map,
                                                                                                  v_map);
    } else {
      LaunchKernel(grid, swa_t5::kT6Threads, out.device(), swa_t5::kT6Bytes)
          .config({.use_pdl = kUsePDL})(kernel, params, k_map, v_map);
    }
  }
};

}  // namespace muse_vr
