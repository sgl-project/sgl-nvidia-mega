// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
/**
 * \brief Paged FP8 (e4m3) attention for speculative verify with tensor-core MMA (tcgen05): 8 query tokens per
 * request, 8 query heads per KV head, head_dim 64, page size 128. The key range of a request (all keys for full
 * attention, or a sliding window) is split over thread blocks in proportion to its length; every block writes a
 * partial result (unnormalized output, row maximum, row sum) and a second kernel merges them.
 *
 * Shapes (per call): q [R * 8, Hq, 64] e4m3 (heads and dims contiguous); K/V pools [slots, Hkv * 64] e4m3
 * (token-major, slot = page * 128 + offset); page_table [R, >= pages] int32; seq_lens [R] int32 (KV length including
 * the 8 verify tokens); out [R * 8, Hq, 64] bf16, contiguous. Hkv must be even.
 *
 * Masking: token t (0..7) of request r sits at position p = seq_len - 8 + t and attends keys [lo, p], where
 * lo = max(0, p - W) with W = window_left >= 0, and lo = 0 for window_left < 0 (full attention).
 *
 * Arithmetic, in order (per query row):
 *   s   = fp32 dot product of e4m3 q and e4m3 k (tensor-core MMA, fp32 accumulation)
 *   (scale_log2 = fp32(bmm1_scale * log2(e)), host side)
 *   for each tile of 128 keys, in key order:
 *     mt  = max of s over the tile * scale_log2           (the tile's maximum, fp32; the scale is positive)
 *     m'  = max(m, mt)                                    (running maximum)
 *     ar  = exp2(m - m'), at = exp2(mt - m')              (rescale of the earlier tiles, weight of this tile)
 *     p   = exp2(fma(s, scale_log2, -mt))                 (against the tile's own maximum; one rounding)
 *     l   = l * ar + (sum of p over the tile) * at        (fp32; each half of the tile's keys summed in four
 *                                                          interleaved partial sums: keys i with i % 4 = 0, 1, 2,
 *                                                          3 into sums 0, 1, 2, 3, then (0 + 2) + (1 + 3))
 *     pq  = e4m3(p * 448)                                 (round to nearest even, satfinite)
 *     o   = fma(o, ar, (sum of pq * v over the tile) * at) (tensor-core MMA into fp32, then fp32 arithmetic)
 *   l of the row = l of the first half of the keys + l of the second half
 *   merge of the blocks of a request and KV head (see the merge kernel): per group g = 0..G-1 of blocks
 *     b = g, g + G, ...: M_g = max m_b, O_g = sum exp2(m_b - M_g) o_b, L_g likewise (in block order, in batches of
 *     16 with the running maximum updated per batch); then M = max M_g, O = sum exp2(M_g - M) O_g, L likewise
 *   out = bf16(O * (bmm2_scale / 448) / L)
 * exp2 is ex2.approx.ftz.f32. With kNormGate the merge kernel also applies the attention output norm and gate to
 * each (token, head) row (SGLang's norm_gate arithmetic for 16 threads per row; see the merge kernel).
 *
 * Work split: grid (S, Hkv / 2, R). Block (s, j, r) serves KV heads 2j and 2j + 1 of request r for pages
 * [first + s * n, first + (s + 1) * n) of the request's key range, n = ceil(pages / S). Each block: 10 warps.
 * Warp 8 (one thread) loads the K and V tiles of each page with the tensor memory accelerator (a 128-key x 128-byte
 * box: the two KV heads' 64 bytes of each key) into a ring of kStages shared-memory stages. Warp 9 (one thread)
 * issues the MMAs (tcgen05.mma, kind::f8f6f4, M = 64 rows per KV head, the two heads' results at tensor-memory
 * lane offsets 0 and 16 of every 32-lane group): S = Q K^T into one of two 128-column score buffers, and, once the
 * probabilities of a tile are in shared memory, P V into one of two 64-column output buffers. Warps 0-7 do the
 * softmax arithmetic above with two threads per query row (warp w, lane i: KV head 2j + i / 16, row
 * 16 (w % 4) + i % 16, row = token * 8 + head; w / 4 selects the half of the tile's keys and of the output dims);
 * the two threads of a row exchange their partial maxima through shared memory and a named barrier. Hand-offs use
 * mbarriers: TMA completion (complete_tx) for full stages, tcgen05.commit for finished MMAs (scores ready, output
 * tile ready, stage free), thread arrivals for probabilities written (each thread stores its part of the row, then
 * fence.proxy.async.shared::cta, then arrives).
 *
 * Shared-memory operand layouts (all 1024-byte aligned): K and V stages [128 keys][128 B] in the 128-byte swizzle
 * (as written by the tensor memory accelerator); Q [64 rows][64 B] per head in the 64-byte swizzle; P [64 rows]
 * [128 B] per head in the 128-byte swizzle. Q and P are K-major A operands, K is a K-major B operand, V is an
 * MN-major B operand (dims contiguous).
 *
 * With kQk = E > 0 (query/key prologue; layers without RoPE), the kernel takes the bf16 QKV projection output instead
 * of e4m3 q and does the work of SGLang's fused QK-norm kernel (fused_qk_norm.py, _qk_rms_norm_fwd_kernel with an FP8
 * KV-cache store) itself, with that kernel's arithmetic for E columns per thread (the host picks E from the row count
 * T * (Hq + 2 Hkv) as the Triton kernel picks its block size): per (token, head) row of 64 values x, in fp32,
 *   ss = sum of x^2: each of the row's 64/E threads sums the squares of its E consecutive columns in the fixed order
 *        of row_sumsq<E>, then a butterfly over the row's threads (xor 32/E, ..., 2, 1)
 *   r  = rsqrt.approx.ftz(eps + div.full(ss, 64)),  y = f32(bf16(r * x))  (V rows: y = x)
 *   (the Triton kernel's form with FP fusion; without it (kUnfused) the 2-column sum of squares is x0*x0 + x1*x1 and
 *   the mean is ss * div.approx(1, 64), each operation rounded separately)
 *   stored value = e4m3(min(max(y, -448), 448)), round to nearest even, satfinite.
 * Every block computes the 128 q rows of its KV-head pair into shared memory. A block whose pages hold keys of the 8
 * new tokens (positions seq_len - 8 .. seq_len - 1) also computes their K and V rows for its pair, writes them into
 * its shared-memory K/V tiles after the tile's load has completed (before the tile's MMAs; the loaded pool rows of
 * those keys are replaced) and stores them into the pools at cache_loc (rows with cache_loc == 0 are skipped, as in
 * the Triton kernel). The pages of a (request, KV-head pair) are split over its blocks without overlap, so only the
 * storing block reads (by TMA, completed before the store) the pool bytes it stores.
 *
 * Ordering: q, seq_lens and the page table may be written by the preceding kernels, so the kernel reads them only
 * after griddepcontrol.wait (with use_pdl); before that wait it only initializes its barriers and tensor memory. The K/V pool rows read here are written before this kernel by the
 * preceding kernels of the step or by earlier steps. Pool rows past the end of a request inside its last page,
 * and rows before the window, are loaded with the page and masked out of the probabilities (their values are
 * finite: the pools are allocated zero-filled and only finite values are stored into them).
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

namespace fva {

constexpr int kHeadDim = 64;
constexpr int kQPerKv = 8;
constexpr int kTokens = 8;
constexpr int kRows = 64;  // query rows per KV head: token * 8 + head
constexpr int kTile = 128;  // keys per tile = page size
constexpr int kStages = 4;
constexpr int kSoftmaxWarps = 8;  // two threads per query row
constexpr int kThreads = 32 * (kSoftmaxWarps + 2);
constexpr int kRec = 68;  // floats per partial row record: 64 outputs, m, l, 2 pad
constexpr float kPScale = 448.0f;

// Shared memory (bytes from a 1024-aligned base)
constexpr int kStageBytes = kTile * 128;  // one K or V stage: 128 keys x (2 heads x 64 B)
constexpr int kSmK = 0;
constexpr int kSmV = kSmK + kStages * kStageBytes;
constexpr int kSmP = kSmV + kStages * kStageBytes;  // [2 buffers][2 heads][64 rows][128 B]
constexpr int kSmQ = kSmP + 2 * 2 * kRows * 128;    // [2 heads][64 rows][64 B]
constexpr int kSmRed = kSmQ + 2 * kRows * 64;       // [2 tile parities + row sums][128 rows][2 halves] fp32
constexpr int kSmPages = kSmRed + 3 * 128 * 2 * 4;  // page numbers of the split (int32), up to kMaxTiles
constexpr int kMaxTiles = 2048;
constexpr int kMaxCluster = 16;
constexpr int kSmRecv = kSmPages + kMaxTiles * 4;   // kClusterMerge: [ceil(128 / S) rows][S sources][kRec] fp32
constexpr int kRecvBytes = (128 + kMaxCluster) * kRec * 4;
constexpr int kSmBar = kSmRecv + kRecvBytes;        // mbarriers + tensor-memory address
constexpr int kSmBytes = kSmBar + 256 + 1024;       // + alignment slack
constexpr int kSmNew = kSmBar + 256;                // kQk: K and V rows of the 8 new tokens [2][8 tokens][128 B]
constexpr int kSmBytesQk = kSmNew + 2 * kTokens * 128 + 1024;
constexpr uint32_t kTmemCols = 512;                 // S0 [0,128) S1 [128,256) O0 [256,320) O1 [320,384)

struct Params {
  const uint8_t* __restrict__ q;
  const int32_t* __restrict__ page_table;
  const int32_t* __restrict__ seq_lens;
  float* __restrict__ partial;  // [R][Hkv][64 rows][S][kRec]
  __nv_bfloat16* __restrict__ out;
  int64_t q_token_stride;  // bytes
  int64_t page_table_stride;
  int32_t num_q_heads;
  int32_t num_kv_heads;
  int32_t num_splits;
  int32_t window_left;  // < 0: full attention
  float scale_log2;
  float out_scale;  // bmm2_scale / 448
  const __nv_bfloat16* __restrict__ gate;  // kNormGate: [R * 8, >= Hq * 64] bf16, row stride gate_token_stride
  int64_t gate_token_stride;                // elements
  float norm_eps;
  // kQk: [R * 8, >= (2 Hq + 2 Hkv) * 64] bf16 QKV output (q, k, v, gate), row stride qkv_token_stride elements;
  // the pool slot of each new token (0: not stored); the pools for generic stores (row stride pool_row_bytes);
  // the QK-norm eps
  const __nv_bfloat16* __restrict__ qkv;
  int64_t qkv_token_stride;
  const int64_t* __restrict__ cache_loc;
  uint8_t* k_store;
  uint8_t* v_store;
  int64_t pool_row_bytes;
  float qk_eps;
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
// Upper bound on try_wait calls of one wait: every arrival comes from this block (or from its own MMAs and copies);
// the bound turns a wait that can only hang through a defect into a trap.
inline constexpr uint32_t kMaxPolls = 1u << 24;
__device__ __noinline__ void wait_bound_exceeded() { __trap(); }
__device__ __forceinline__ void mbar_wait(uint32_t bar, uint32_t parity) {
  uint32_t polls = 0;
  while (true) {
    uint32_t done;
    asm volatile(
        "{\n"
        ".reg .pred p;\n"
        "mbarrier.try_wait.parity.shared::cta.b64 p, [%1], %2;\n"
        "selp.u32 %0, 1, 0, p;\n"
        "}\n"
        : "=r"(done)
        : "r"(bar), "r"(parity)
        : "memory");
    if (done) break;
    if (++polls == kMaxPolls) [[unlikely]] wait_bound_exceeded();
  }
}
// Wait used where every thread of a warp waits and the warp's next tensor-memory load (.sync.aligned) or other warp-wide
// operation needs the warp converged: each thread exits the spin on its own, so the warp reconverges before going on
// (PTX ISA: .aligned instructions in conditionally executed code need every thread of the warp to take the same path).
__device__ __forceinline__ void mbar_wait_warp(uint32_t bar, uint32_t parity) {
  mbar_wait(bar, parity);
  __syncwarp();
}
__device__ __forceinline__ void tma_load_2d(uint32_t dst, const CUtensorMap* map, int32_t c0, int32_t c1, uint32_t bar) {
  asm volatile(
      "cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1, {%2, %3}], [%4];\n" ::"r"(dst),
      "l"(reinterpret_cast<uint64_t>(map)), "r"(c0), "r"(c1), "r"(bar)
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
// Packed fp32 pairs (PTX ISA: fma/add/mul .f32x2): two independent IEEE fp32 operations each.
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

// ---------------------------------------------------------------- query/key prologue (kQk)
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
// Loads are volatile so that they are issued in program order, ahead of the arithmetic that uses them.
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
// e4m3 bytes of one row's E columns held by this thread (all 64/E threads of the row call it together): QK-norm of
// the row, or (raw) the values unchanged; see the header comment.
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

template <int E, bool kUnfused>
__device__ __forceinline__ void qk_finish(const uint32_t (&w)[E / 2], float eps, bool raw, uint16_t (&b)[E / 2]) {
  float x[E];
#pragma unroll
  for (int i = 0; i < E / 2; ++i) {
    const float2 f = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&w[i]));
    x[2 * i] = f.x;
    x[2 * i + 1] = f.y;
  }
  float ss = row_sumsq<E, kUnfused>(x);
#pragma unroll
  for (int m = 32 / E; m >= 1; m >>= 1) ss = __fadd_rn(ss, __shfl_xor_sync(0xffffffffu, ss, m));
  float r;
  const float var = qk_mean<kUnfused>(ss);
  const float t = __fadd_rn(eps, var);
  asm("rsqrt.approx.ftz.f32 %0, %1;\n" : "=f"(r) : "f"(t));
  float y[E];
#pragma unroll
  for (int e = 0; e < E; ++e) y[e] = raw ? x[e] : __bfloat162float(__float2bfloat16_rn(__fmul_rn(r, x[e])));
#pragma unroll
  for (int m = 0; m < E / 2; ++m) {
    uint16_t e;
    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;\n"
        : "=h"(e)
        : "f"(fminf(fmaxf(y[2 * m + 1], -448.0f), 448.0f)), "f"(fminf(fmaxf(y[2 * m], -448.0f), 448.0f)));
    b[m] = e;
  }
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

// Pages of the split: [p0, p1) of the request's logical pages.
__device__ __forceinline__ void split_pages(int seq_len, int window_left, int num_splits, int split, int& p0, int& p1,
                                            int& lo) {
  lo = window_left >= 0 ? max(0, seq_len - kTokens - window_left) : 0;
  const int first = lo / kTile, end = (max(seq_len, 1) - 1) / kTile + 1;
  const int n = (end - first + num_splits - 1) / num_splits;
  p0 = min(end, first + split * n);
  p1 = min(end, p0 + n);
}

// ---------------------------------------------------------------------------------------------- main kernel
__device__ __forceinline__ uint32_t map_rank(uint32_t addr, int rank) {
  uint32_t d;
  asm volatile("mapa.shared::cluster.u32 %0, %1, %2;\n" : "=r"(d) : "r"(addr), "r"(rank));
  return d;
}
__device__ __forceinline__ void st_async_v4(uint32_t addr, float4 v, uint32_t bar) {
  asm volatile("st.async.shared::cluster.mbarrier::complete_tx::bytes.v4.f32 [%0], {%1, %2, %3, %4}, [%5];\n" ::"r"(addr),
               "f"(v.x), "f"(v.y), "f"(v.z), "f"(v.w), "r"(bar)
               : "memory");
}
__device__ __forceinline__ void mbar_wait_cluster(uint32_t bar, uint32_t parity) {
  uint32_t polls = 0;
  while (true) {
    uint32_t done;
    asm volatile(
        "{\n"
        ".reg .pred p;\n"
        "mbarrier.try_wait.parity.acquire.cluster.shared::cta.b64 p, [%1], %2;\n"
        "selp.u32 %0, 1, 0, p;\n"
        "}\n"
        : "=r"(done)
        : "r"(bar), "r"(parity)
        : "memory");
    if (done) break;
    if (++polls == kMaxPolls) [[unlikely]] wait_bound_exceeded();
  }
}

// Output norm and gate of one (token, head) row from 16 threads (thread c holds dims 4c .. 4c + 3 in x[4], the bf16
// attention output): SGLang's norm_gate arithmetic (no weight) for 16 threads per row (see the merge kernel).
__device__ __forceinline__ void norm_gate16(float (&x)[4], const __nv_bfloat16* gate, float eps) {
  float ss = fmaf(x[0], x[0], 0.0f);
  ss = fmaf(x[1], x[1], ss);
  ss = fmaf(x[2], x[2], ss);
  ss = fmaf(x[3], x[3], ss);
  const unsigned group = 0xffffu << (threadIdx.x & 16);
#pragma unroll
  for (int mask = 8; mask >= 1; mask >>= 1) ss = ss + __shfl_xor_sync(group, ss, mask, 16);
  const float r = rsqrtf(ss / static_cast<float>(kHeadDim) + eps);
  const uint2 graw = *reinterpret_cast<const uint2*>(gate);
  const float2 g01 = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&graw.x));
  const float2 g23 = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&graw.y));
  const auto sig = [](float g) { return __fdividef(1.0f, 1.0f + __expf(-g)); };
  float y[4] = {x[0] * r, x[1] * r, x[2] * r, x[3] * r};
  y[0] *= sig(g01.x);
  y[1] *= sig(g01.y);
  y[2] *= sig(g23.x);
  y[3] *= sig(g23.y);
#pragma unroll
  for (int i = 0; i < 4; ++i) x[i] = y[i];
}

template <bool kUsePDL, bool kClusterMerge, bool kNormGate, int kQk, bool kUnfused>
__global__ void __launch_bounds__(kThreads, 1)
    full_verify_attn_kernel(const Params p, const __grid_constant__ CUtensorMap k_map,
                            const __grid_constant__ CUtensorMap v_map) {
  extern __shared__ uint8_t smem_raw[];
  // 1024-byte aligned base by pointer arithmetic on the shared array (keeps the shared address space visible to the
  // compiler: st.shared / ld.shared instead of generic accesses)
  uint8_t* const sm = smem_raw + ((1024u - (smem_u32(smem_raw) & 1023u)) & 1023u);
  uint64_t* const bars = reinterpret_cast<uint64_t*>(sm + kSmBar);
  // bars: full[kStages], empty[kStages], s_full[2], p_full[2], o_full[2], recv, (tensor-memory address), patch
  const uint32_t bar0 = smem_u32(bars);
  const auto full_bar = [&](int s) { return bar0 + 8u * s; };
  const auto empty_bar = [&](int s) { return bar0 + 8u * (kStages + s); };
  const auto s_full_bar = [&](int b) { return bar0 + 8u * (2 * kStages + b); };
  const auto p_full_bar = [&](int b) { return bar0 + 8u * (2 * kStages + 2 + b); };
  const auto o_full_bar = [&](int b) { return bar0 + 8u * (2 * kStages + 4 + b); };
  const uint32_t recv_bar = bar0 + 8u * (2 * kStages + 6);
  uint32_t* const tmem_slot = reinterpret_cast<uint32_t*>(sm + kSmBar + 8 * (2 * kStages + 7));
  const uint32_t patch_bar = bar0 + 8u * (2 * kStages + 8);  // kQk: new-token rows written into a K/V tile
  // kClusterMerge: the blocks of a (request, KV-head pair) form one cluster (rank = split); row R (of the block's
  // 128: KV head R / 64, row R % 64) is merged by rank R % S into slot R / S of its receive region.
  const int S = p.num_splits;
  const int n_own = kClusterMerge ? (128 - static_cast<int>(blockIdx.x) + S - 1) / S : 0;

  const int split = blockIdx.x, pair = blockIdx.y, req = blockIdx.z;
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;

  // Barriers and tensor memory first: they touch nothing the preceding kernels write.
  if (tid == 0) {
#pragma unroll
    for (int s = 0; s < kStages; ++s) {
      mbar_init(full_bar(s), 1);
      mbar_init(empty_bar(s), 1);
    }
#pragma unroll
    for (int b = 0; b < 2; ++b) {
      mbar_init(s_full_bar(b), 1);
      mbar_init(p_full_bar(b), 32 * kSoftmaxWarps);
      mbar_init(o_full_bar(b), 1);
    }
    if constexpr (kQk > 0) mbar_init(patch_bar, 32 * kSoftmaxWarps);
    if constexpr (kClusterMerge) {
      mbar_init(recv_bar, 1);
      asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;\n" ::"r"(recv_bar),
                   "r"(static_cast<uint32_t>(n_own * S * kRec * 4))
                   : "memory");
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
  const uint32_t tmem = *tmem_slot;
  // Cluster barrier arrival now, wait before the first remote store: every block of the cluster has started and
  // initialized its receive barrier before any block stores into another block's shared memory.
  if constexpr (kClusterMerge) asm volatile("barrier.cluster.arrive.relaxed.aligned;\n" ::: "memory");

  if constexpr (kUsePDL) asm volatile("griddepcontrol.wait;\n" ::: "memory");
  const int seq_len = p.seq_lens[req];
  int p0, p1, lo;
  split_pages(seq_len, p.window_left, p.num_splits, split, p0, p1, lo);
  const int ntiles = p1 - p0;
  // kQk: tiles [tn0, tn0 + ntn) of this block hold keys of the new tokens (keys tnew .. seq_len - 1)
  const int tnew = seq_len - kTokens;
  int tn0 = 0, ntn = 0;
  if constexpr (kQk > 0) {
    const int a = max(max(tnew, 0) / kTile, p0), b = min((max(seq_len, 1) - 1) / kTile, p1 - 1);
    if (a <= b) {
      tn0 = a - p0;
      ntn = b - a + 1;
    }
  }
  if (!kClusterMerge && ntiles <= 0) {  // uniform over the block: no partial for this split
    if (warp == 0)
      asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, %1;\n" ::"r"(tmem), "r"(kTmemCols) : "memory");
    return;
  }
  // Warp 8: page numbers of the split's tiles -> shared memory (one load per page, all in flight together), then the
  // first kStages tile loads. Warps 0-7: Q rows. Warp 9 starts once Q is in shared memory (named barrier 5).
  int32_t* const pages = reinterpret_cast<int32_t*>(sm + kSmPages);
  const int32_t* const pt = p.page_table + static_cast<int64_t>(req) * p.page_table_stride + p0;
  const auto issue_tile = [&](int t) {
    const int s = t % kStages;
    const int page = t < kMaxTiles ? pages[t] : pt[t];
    mbar_expect_tx(full_bar(s), 2 * kStageBytes);
    tma_load_2d(smem_u32(sm + kSmK + s * kStageBytes), &k_map, pair * 128, page * kTile, full_bar(s));
    tma_load_2d(smem_u32(sm + kSmV + s * kStageBytes), &v_map, pair * 128, page * kTile, full_bar(s));
  };
  if (warp == kSoftmaxWarps) {
    for (int t = lane; t < min(ntiles, kMaxTiles); t += 32) pages[t] = pt[t];
    __syncwarp();
    if (lane == 0) {
      // The pool rows may have been written by a preceding kernel with generic-proxy stores; with programmatic
      // dependent launch only griddepcontrol.wait separates them from these async-proxy reads, so order the proxies
      // (PTX ISA, memory consistency model: proxy fences).
      asm volatile("fence.proxy.async.global;\n" ::: "memory");
      for (int t = 0; t < min(ntiles, kStages); ++t) issue_tile(t);
    }
  } else if (warp < kSoftmaxWarps) {
    // Q rows of the two KV heads -> [head][row][64 B] in the 64-byte swizzle (16-byte chunk c of row r at
    // c ^ (r/2 % 4)).
    uint8_t* const sq = sm + kSmQ;
    if constexpr (kQk == 0) {
#pragma unroll
      for (int i = tid; i < 2 * kRows * 4; i += 32 * kSoftmaxWarps) {
        const int hh = i >> 8, r = (i >> 2) & 63, c = i & 3;
        const int t = r >> 3, g = r & 7;
        const uint4 v = *reinterpret_cast<const uint4*>(
            p.q + (static_cast<int64_t>(req) * kTokens + t) * p.q_token_stride +
            ((2 * pair + hh) * kQPerKv + g) * kHeadDim + 16 * c);
        *reinterpret_cast<uint4*>(sq + hh * kRows * 64 + r * 64 + ((c ^ ((r >> 1) & 3)) << 4)) = v;
      }
    } else {
      // q rows from the QKV output: row R (0..127: head 2 * pair + R / 64, row R % 64 = token * 8 + head-in-group),
      // 64 / E threads per row (consecutive lanes), E columns each; all loads first, then the arithmetic.
      constexpr int E = kQk, kLanes = 64 / E, kRowsPass = 32 * kSoftmaxWarps / kLanes, kPasses = 2 * kRows / kRowsPass;
      const int li = lane % kLanes;
      uint32_t w[kPasses][E / 2];
#pragma unroll
      for (int ps = 0; ps < kPasses; ++ps) {
        const int R = ps * kRowsPass + tid / kLanes, hh = R >> 6, r = R & 63;
        ld_bf16_raw<E>(p.qkv + (static_cast<int64_t>(req) * kTokens + (r >> 3)) * p.qkv_token_stride +
                           ((2 * pair + hh) * kQPerKv + (r & 7)) * kHeadDim + E * li,
                       w[ps]);
      }
#pragma unroll
      for (int ps = 0; ps < kPasses; ++ps) {
        const int R = ps * kRowsPass + tid / kLanes, hh = R >> 6, r = R & 63;
        uint16_t b[E / 2];
        qk_finish<E, kUnfused>(w[ps], p.qk_eps, false, b);
        const int c = E * li;  // byte column of the 64-byte row
        st_bytes<E>(sq + hh * kRows * 64 + r * 64 + ((((c >> 4) ^ ((r >> 1) & 3)) << 4) | (c & 15)), b);
      }
    }
    fence_proxy_async();
    asm volatile("bar.arrive 5, %0;\n" ::"r"(32 * (kSoftmaxWarps + 1)) : "memory");
    if constexpr (kQk > 0) {
      if (ntn > 0) {  // uniform over the block
        // K and V rows of the 8 new tokens for the pair -> [K/V][token][head-in-pair * 64 + column] (row J of 32:
        // V if J >= 16, head-in-pair (J >> 3) & 1, token J & 7)
        constexpr int E = kQk, kLanes = 64 / E, kRowsPass = 32 * kSoftmaxWarps / kLanes, kPasses = 32 / kRowsPass;
        const int li = lane % kLanes;
        uint32_t w[kPasses][E / 2];
#pragma unroll
        for (int ps = 0; ps < kPasses; ++ps) {
          const int J = ps * kRowsPass + tid / kLanes, kv = J >> 4, hh = (J >> 3) & 1, t = J & 7;
          ld_bf16_raw<E>(p.qkv + (static_cast<int64_t>(req) * kTokens + t) * p.qkv_token_stride +
                             (p.num_q_heads + kv * p.num_kv_heads + 2 * pair + hh) * kHeadDim + E * li,
                         w[ps]);
        }
        uint8_t* const snew = sm + kSmNew;
#pragma unroll
        for (int ps = 0; ps < kPasses; ++ps) {
          const int J = ps * kRowsPass + tid / kLanes, kv = J >> 4, hh = (J >> 3) & 1, t = J & 7;
          uint16_t b[E / 2];
          qk_finish<E, kUnfused>(w[ps], p.qk_eps, kv == 1, b);
          st_bytes<E>(snew + kv * kTokens * 128 + t * 128 + hh * 64 + E * li, b);
        }
        asm volatile("bar.sync 6, %0;\n" ::"r"(32 * kSoftmaxWarps) : "memory");
      }
    }
  } else {
    asm volatile("bar.sync 5, %0;\n" ::"r"(32 * (kSoftmaxWarps + 1)) : "memory");
  }

  if (warp == kSoftmaxWarps) {
    // ------------------------------------------------------------------ loads
    if (lane == 0) {
      for (int t = kStages; t < ntiles; ++t) {
        mbar_wait(empty_bar(t % kStages), (t / kStages - 1) & 1);
        issue_tile(t);
      }
    }
  } else if (warp == kSoftmaxWarps + 1) {
    // ------------------------------------------------------------------ MMA issue
    if (lane == 0) {
      constexpr uint32_t id_qk = idesc_f8(64, 128, 0), id_pv = idesc_f8(64, 64, 1);
      const uint32_t sq = smem_u32(sm + kSmQ), sp = smem_u32(sm + kSmP);
      const auto qk = [&](int t) {
        const int s = t % kStages, b = t & 1;
        mbar_wait(full_bar(s), (t / kStages) & 1);
        if constexpr (kQk > 0)
          if (t >= tn0 && t < tn0 + ntn) mbar_wait(patch_bar, (t - tn0) & 1);
        tc_fence_after();
        const uint32_t sk = smem_u32(sm + kSmK + s * kStageBytes);
#pragma unroll
        for (int hh = 0; hh < 2; ++hh) {
#pragma unroll
          for (int k = 0; k < 2; ++k)
            mma_f8(tmem + ((16u * hh) << 16) + 128u * b, sdesc(sq + hh * kRows * 64 + 32 * k, 512, kSw64),
                   sdesc(sk + 64 * hh + 32 * k, 1024, kSw128), id_qk, k);
        }
        mma_commit(s_full_bar(b));
      };
      if (ntiles > 0) qk(0);
      for (int t = 0; t < ntiles; ++t) {
        const int s = t % kStages, b = t & 1;
        if (t + 1 < ntiles) qk(t + 1);  // its score buffer was released by the arrival waited below for tile t - 1
        mbar_wait(p_full_bar(b), (t >> 1) & 1);
        tc_fence_after();
        const uint32_t sv = smem_u32(sm + kSmV + s * kStageBytes);
#pragma unroll
        for (int hh = 0; hh < 2; ++hh) {
#pragma unroll
          for (int k = 0; k < 4; ++k)
            mma_f8(tmem + ((16u * hh) << 16) + 256u + 64u * b,
                   sdesc(sp + (b * 2 + hh) * kRows * 128 + 32 * k, 1024, kSw128),
                   sdesc(sv + 64 * hh + 4096 * k, 1024, kSw128), id_pv, k);
        }
        mma_commit(o_full_bar(b));
        mma_commit(empty_bar(s));
      }
    }
  } else {
    // ------------------------------------------------------------------ softmax (warps 0-7, two threads per row)
    // Warp w, lane i: row of KV head 2 * pair + i / 16, row 16 * (w % 4) + i % 16 (tensor-memory lane 32 * (w % 4) + i);
    // half = w / 4 takes keys [64 half, 64 half + 64) of every tile and output dims [32 half, 32 half + 32).
    const int quarter = warp & 3, half = warp >> 2;
    const int hh = lane >> 4, row = 16 * quarter + (lane & 15);
    const int lrow = 32 * quarter + lane;  // tensor-memory lane = slot in the row-max exchange
    const int tok = row >> 3;
    const int pos = seq_len - kTokens + tok;  // the row's position: keys <= pos
    const int klo = p.window_left >= 0 ? max(lo, pos - p.window_left) : 0;
    const uint32_t lane_base = (32u * quarter) << 16;
    float* const red = reinterpret_cast<float*>(sm + kSmRed);
    uint8_t* const prow0 = sm + kSmP + hh * kRows * 128 + row * 128;
    float o[32];
#pragma unroll
    for (int d = 0; d < 32; ++d) o[d] = 0.0f;
    float m = -INFINITY, l = 0.0f, ar_prev = 0.0f, at_prev = 0.0f;
    for (int t = 0; t < ntiles; ++t) {
      const int b = t & 1;
      const int key0 = (p0 + t) * kTile + 64 * half;  // this thread's first key of the tile
      if constexpr (kQk > 0) {
        if (t >= tn0 && t < tn0 + ntn) {  // uniform over the block
          // the new tokens' keys of this tile: rows of the loaded K/V tile replaced (after the load completed) and
          // stored into the pools; then the MMA warp may issue the tile's MMAs
          const int st = t % kStages, kp = (p0 + t) * kTile;
          const int kb = max(tnew, kp), nk = min(seq_len, kp + kTile) - kb;
          mbar_wait_warp(full_bar(st), (t / kStages) & 1);
          const uint8_t* const snew = sm + kSmNew;
          for (int i = tid; i < 2 * nk * 8; i += 32 * kSoftmaxWarps) {
            const int kv = i / (nk * 8), rem = i - kv * nk * 8, key = kb + (rem >> 3), c = rem & 7;
            const int row = key - kp, tt = key - tnew;
            const uint4 v = *reinterpret_cast<const uint4*>(snew + kv * kTokens * 128 + tt * 128 + 16 * c);
            uint8_t* const stage = sm + (kv ? kSmV : kSmK) + st * kStageBytes;
            *reinterpret_cast<uint4*>(stage + row * 128 + ((c ^ (row & 7)) << 4)) = v;
            const int64_t loc = p.cache_loc[static_cast<int64_t>(req) * kTokens + tt];
            uint8_t* const pool = kv ? p.v_store : p.k_store;
            if (loc != 0) *reinterpret_cast<uint4*>(pool + loc * p.pool_row_bytes + pair * 128 + 16 * c) = v;
          }
          fence_proxy_async();
          mbar_arrive(patch_bar);
        }
      }
      mbar_wait_warp(s_full_bar(b), (t >> 1) & 1);
      tc_fence_after();
      float x[64];
      tmem_ld32(tmem + lane_base + 128u * b + 64u * half, *reinterpret_cast<float(*)[32]>(x));
      tmem_ld32(tmem + lane_base + 128u * b + 64u * half + 32u, *reinterpret_cast<float(*)[32]>(x + 32));
      tmem_wait_ld();
      if (!(key0 >= klo && key0 + 63 <= pos)) {  // tile edge: keys outside [klo, pos] do not take part
#pragma unroll
        for (int i = 0; i < 64; ++i) {
          const int key = key0 + i;
          if (key < klo || key > pos) x[i] = -INFINITY;
        }
      }
      // maximum of the raw scores (the scale is positive, so max(s) * scale == max(s * scale))
      float mx[22];
#pragma unroll
      for (int i = 0; i < 21; ++i) mx[i] = max3(x[3 * i], x[3 * i + 1], x[3 * i + 2]);
      mx[21] = x[63];
#pragma unroll
      for (int i = 0; i < 7; ++i) mx[i] = max3(mx[3 * i], mx[3 * i + 1], mx[3 * i + 2]);
      mx[0] = max3(max3(mx[0], mx[1], mx[2]), max3(mx[3], mx[4], mx[5]), fmaxf(mx[6], mx[21]));
      red[(b * 128 + lrow) * 2 + half] = mx[0];
      asm volatile("bar.sync %0, %1;\n" ::"r"(1 + quarter), "r"(64) : "memory");
      const float smax = fmaxf(red[(b * 128 + lrow) * 2], red[(b * 128 + lrow) * 2 + 1]);
      // the tile's own maximum: the probabilities of the tile are quantized against it
      const float mtile = smax == -INFINITY ? -INFINITY : smax * p.scale_log2;
      const float mu = mtile == -INFINITY ? 0.0f : mtile;
      const float mnew = fmaxf(m, mtile);
      const float ar = m == -INFINITY ? 0.0f : fast_exp2(m - mnew);         // rescale of the earlier tiles
      const float at = mtile == -INFINITY ? 0.0f : fast_exp2(mtile - mnew);  // weight of this tile
      // p = exp2(s * scale - mu) (one rounding: fused multiply-add), pq = e4m3(p * 448)
      const float2 sc = make_float2(p.scale_log2, p.scale_log2), nmu = make_float2(-mu, -mu);
      const float2 k448 = make_float2(kPScale, kPScale);
      float2 ls[2] = {make_float2(0.0f, 0.0f), make_float2(0.0f, 0.0f)};
#pragma unroll
      for (int i = 0; i < 64; i += 2) {
        const float2 y = fma2(make_float2(x[i], x[i + 1]), sc, nmu);
        const float2 pi = make_float2(fast_exp2(y.x), fast_exp2(y.y));
        ls[(i >> 1) & 1] = add2(ls[(i >> 1) & 1], pi);
        const float2 pq = mul2(pi, k448);
        x[i] = pq.x;
        x[i + 1] = pq.y;
      }
      l = l * ar + ((ls[0].x + ls[1].x) + (ls[0].y + ls[1].y)) * at;
      m = mnew;
      // probabilities -> P buffer b (128-byte swizzle: chunk c of row r at c ^ (r % 8)); buffer b was last read by the
      // MMAs of tile t - 2, finished before the output-tile wait of the previous iteration.
      uint8_t* const prow = prow0 + b * 2 * kRows * 128;
#pragma unroll
      for (int c = 0; c < 4; ++c) {
        uint4 w;
        w.x = e4m3x4(x[16 * c + 0], x[16 * c + 1], x[16 * c + 2], x[16 * c + 3]);
        w.y = e4m3x4(x[16 * c + 4], x[16 * c + 5], x[16 * c + 6], x[16 * c + 7]);
        w.z = e4m3x4(x[16 * c + 8], x[16 * c + 9], x[16 * c + 10], x[16 * c + 11]);
        w.w = e4m3x4(x[16 * c + 12], x[16 * c + 13], x[16 * c + 14], x[16 * c + 15]);
        *reinterpret_cast<uint4*>(prow + (((4 * half + c) ^ (row & 7)) << 4)) = w;
      }
      fence_proxy_async();
      tc_fence_before();
      mbar_arrive(p_full_bar(b));
      if (t > 0) {
        const int bp = (t - 1) & 1;
        mbar_wait_warp(o_full_bar(bp), ((t - 1) >> 1) & 1);
        tc_fence_after();
        float v[32];
        tmem_ld32(tmem + lane_base + 256u + 64u * bp + 32u * half, v);
        tmem_wait_ld();
#pragma unroll
        for (int i = 0; i < 32; ++i) o[i] = fmaf(o[i], ar_prev, v[i] * at_prev);
        tc_fence_before();
      }
      ar_prev = ar;
      at_prev = at;
    }
    if (ntiles > 0) {
      const int bp = (ntiles - 1) & 1;
      mbar_wait_warp(o_full_bar(bp), ((ntiles - 1) >> 1) & 1);
      tc_fence_after();
      float v[32];
      tmem_ld32(tmem + lane_base + 256u + 64u * bp + 32u * half, v);
      tmem_wait_ld();
#pragma unroll
      for (int i = 0; i < 32; ++i) o[i] = fmaf(o[i], ar_prev, v[i] * at_prev);
    }
    // row sum: half 0 + half 1
    red[(256 + lrow) * 2 + half] = l;
    asm volatile("bar.sync %0, %1;\n" ::"r"(1 + quarter), "r"(64) : "memory");
    const float lrowsum = red[(256 + lrow) * 2] + red[(256 + lrow) * 2 + 1];
    const int kvh = 2 * pair + hh;
    if constexpr (!kClusterMerge) {
      // partial record of this row: [req][kvh][row][split][kRec]
      float* rec = p.partial +
                   (((static_cast<int64_t>(req) * p.num_kv_heads + kvh) * kRows + row) * p.num_splits + split) * kRec;
#pragma unroll
      for (int d = 0; d < 32; d += 4)
        *reinterpret_cast<float4*>(rec + 32 * half + d) = make_float4(o[d], o[d + 1], o[d + 2], o[d + 3]);
      if (half == 0) *reinterpret_cast<float2*>(rec + kHeadDim) = make_float2(m, lrowsum);
    } else {
      // record of this row -> slot R / S, source rank `split`, of rank R % S (R = 64 * head-in-pair + row)
      const int R = 64 * hh + row;
      const int owner = R % S;
      float* const recv = reinterpret_cast<float*>(sm + kSmRecv);
      const uint32_t dst = map_rank(smem_u32(recv + ((R / S) * S + split) * kRec + 32 * half), owner);
      const uint32_t rbar = map_rank(recv_bar, owner);
      asm volatile("barrier.cluster.wait.aligned;\n" ::: "memory");
#pragma unroll
      for (int d = 0; d < 32; d += 4) st_async_v4(dst + 4 * d, make_float4(o[d], o[d + 1], o[d + 2], o[d + 3]), rbar);
      if (half == 0) st_async_v4(dst + 4 * kHeadDim, make_float4(m, lrowsum, 0.0f, 0.0f), rbar);
    }
  }
  if constexpr (kClusterMerge) {
    if (warp >= kSoftmaxWarps) asm volatile("barrier.cluster.wait.aligned;\n" ::: "memory");
    if (warp < kSoftmaxWarps) {
      // merge of the rows this block owns: 16 threads per row (thread c: dims 4c .. 4c + 3), sources in rank order
      // against the maximum over the sources; then the output norm and gate (kNormGate)
      mbar_wait_cluster(recv_bar, 0);
      __syncwarp();  // reconverge before the warp-wide merge (shuffles)
      const float* const recv = reinterpret_cast<const float*>(sm + kSmRecv);
      for (int item = tid; item < n_own * 16; item += 32 * kSoftmaxWarps) {
        const int sl = item >> 4, c4 = item & 15;
        const float* rec = recv + sl * S * kRec;
        float mx = -INFINITY;
        for (int sr = 0; sr < S; ++sr) mx = fmaxf(mx, rec[sr * kRec + kHeadDim]);
        float acc[4] = {0.0f, 0.0f, 0.0f, 0.0f};
        float lsum = 0.0f;
        for (int sr = 0; sr < S; ++sr) {
          const float ms = rec[sr * kRec + kHeadDim];
          if (ms == -INFINITY) continue;
          const float w = fast_exp2(ms - mx);
          const float4 v = *reinterpret_cast<const float4*>(rec + sr * kRec + 4 * c4);
          acc[0] += w * v.x;
          acc[1] += w * v.y;
          acc[2] += w * v.z;
          acc[3] += w * v.w;
          lsum += w * rec[sr * kRec + kHeadDim + 1];
        }
        const int R = sl * S + static_cast<int>(blockIdx.x);
        const int kvh = 2 * pair + R / 64, row = R % 64;
        const int t = row >> 3, h = kvh * kQPerKv + (row & 7);
        const float f = lsum > 0.0f ? p.out_scale / lsum : 0.0f;
        __nv_bfloat162 lo = __floats2bfloat162_rn(acc[0] * f, acc[1] * f);
        __nv_bfloat162 hi = __floats2bfloat162_rn(acc[2] * f, acc[3] * f);
        if constexpr (kNormGate) {
          const float2 x01 = __bfloat1622float2(lo), x23 = __bfloat1622float2(hi);
          float x[4] = {x01.x, x01.y, x23.x, x23.y};
          norm_gate16(x, p.gate + (static_cast<int64_t>(req) * kTokens + t) * p.gate_token_stride + h * kHeadDim + 4 * c4,
                      p.norm_eps);
          lo = __floats2bfloat162_rn(x[0], x[1]);
          hi = __floats2bfloat162_rn(x[2], x[3]);
        }
        uint2 packed;
        packed.x = *reinterpret_cast<uint32_t*>(&lo);
        packed.y = *reinterpret_cast<uint32_t*>(&hi);
        *reinterpret_cast<uint2*>(p.out + ((static_cast<int64_t>(req) * kTokens + t) * p.num_q_heads + h) * kHeadDim +
                                  4 * c4) = packed;
      }
    }
  }
  tc_fence_before();
  __syncthreads();
  if constexpr (kUsePDL) asm volatile("griddepcontrol.launch_dependents;\n" ::: "memory");
  if (warp == 0) {
    tc_fence_after();
    asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, %1;\n" ::"r"(tmem), "r"(kTmemCols) : "memory");
  }
}

// ---------------------------------------------------------------------------------------------- merge kernel
// grid (64 / (8 / G), Hkv, R), 256 threads, G = merge_groups (1, 2, 4 or 8) warps per row. Warp w: row
// blockIdx.x * (8 / G) + w / G, group g = w % G merges the splits s = g, g + G, ... (lane j: output dims 2j, 2j + 1)
// against its own maximum M_g: O_g = sum exp2(m_s - M_g) o_s, L_g = sum exp2(m_s - M_g) l_s, in split order; then with
// M = max M_g: O = sum exp2(M_g - M) O_g, L likewise, in group order.
template <bool kUsePDL, bool kNormGate>
__global__ void __launch_bounds__(256) full_verify_attn_merge_kernel(const Params p, int merge_groups) {
  __shared__ float sums[8][kHeadDim + 2];
  if constexpr (kUsePDL) asm volatile("griddepcontrol.wait;\n" ::: "memory");
  const int G = merge_groups;
  const int kvh = blockIdx.y, req = blockIdx.z;
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
  const int row = blockIdx.x * (8 / G) + warp / G, grp = warp % G;
  const int seq_len = p.seq_lens[req];
  int nsplit = 0;
  {
    int p0, p1, lo;
    split_pages(seq_len, p.window_left, p.num_splits, 0, p0, p1, lo);
    const int n = p1 - p0;  // pages per split
    const int first = lo / kTile, end = (max(seq_len, 1) - 1) / kTile + 1;
    nsplit = n > 0 ? (end - first + n - 1) / n : 0;
  }
  const float* base = p.partial + ((static_cast<int64_t>(req) * p.num_kv_heads + kvh) * kRows + row) *
                                      static_cast<int64_t>(p.num_splits) * kRec;
  float mw = -INFINITY;
  float2 acc = make_float2(0.0f, 0.0f);
  float lw = 0.0f;
  constexpr int kBatch = 16;  // splits whose loads are issued together
  for (int s0 = grp; s0 < nsplit; s0 += G * kBatch) {
    float2 ml[kBatch], v[kBatch];
#pragma unroll
    for (int i = 0; i < kBatch; ++i) {
      const int s = s0 + G * i;
      if (s < nsplit) {
        const float* rec = base + s * kRec;
        ml[i] = *reinterpret_cast<const float2*>(rec + kHeadDim);
        v[i] = *reinterpret_cast<const float2*>(rec + 2 * lane);
      } else {
        ml[i] = make_float2(-INFINITY, 0.0f);
        v[i] = make_float2(0.0f, 0.0f);
      }
    }
    float mb = mw;
#pragma unroll
    for (int i = 0; i < kBatch; ++i) mb = fmaxf(mb, ml[i].x);
    if (mb != -INFINITY) {
      const float r = mw == -INFINITY ? 0.0f : fast_exp2(mw - mb);
      acc.x *= r;
      acc.y *= r;
      lw *= r;
#pragma unroll
      for (int i = 0; i < kBatch; ++i) {
        const float ws = ml[i].x == -INFINITY ? 0.0f : fast_exp2(ml[i].x - mb);
        acc.x += ws * v[i].x;
        acc.y += ws * v[i].y;
        lw += ws * ml[i].y;
      }
      mw = mb;
    }
  }
  if (G > 1) {
    sums[warp][2 * lane] = acc.x;
    sums[warp][2 * lane + 1] = acc.y;
    if (lane == 0) {
      sums[warp][kHeadDim] = lw;
      sums[warp][kHeadDim + 1] = mw;
    }
    __syncthreads();
    if (grp != 0) return;
    float mx = -INFINITY;
    for (int g = 0; g < G; ++g) mx = fmaxf(mx, sums[warp + g][kHeadDim + 1]);
    float o0 = 0.0f, o1 = 0.0f, lt = 0.0f;
    for (int g = 0; g < G; ++g) {
      const float m_g = sums[warp + g][kHeadDim + 1];
      const float r = m_g == -INFINITY ? 0.0f : fast_exp2(m_g - mx);
      o0 += r * sums[warp + g][2 * lane];
      o1 += r * sums[warp + g][2 * lane + 1];
      lt += r * sums[warp + g][kHeadDim];
    }
    acc = make_float2(o0, o1);
    lw = lt;
  }
  const float f = lw > 0.0f ? p.out_scale / lw : 0.0f;
  const int t = row >> 3, h = kvh * kQPerKv + (row & 7);
  __nv_bfloat162 v = __floats2bfloat162_rn(acc.x * f, acc.y * f);
  if constexpr (kNormGate) {
    // Output norm and gate of the row (x: the bf16 attention output; lane j holds dims 2j, 2j + 1). The arithmetic of
    // SGLang's norm_gate kernel (no weight) with 16 threads per row: thread i sums the squares of dims 4i .. 4i + 3
    // in order with fmaf, then a butterfly over the 16 threads (xor 8, 4, 2, 1); r = rsqrtf(ss / 64 + eps);
    // y = bf16((x * r) * (1 / (1 + exp(-g)))) with __fdividef and __expf. Here thread i of that kernel is the lane
    // pair (2i, 2i + 1): the even lane starts the chain, the odd lane finishes it and holds the row's partial.
    const float2 x = __bfloat1622float2(v);
    const float a = fmaf(x.y, x.y, fmaf(x.x, x.x, 0.0f));
    const float prev = __shfl_xor_sync(0xffffffffu, a, 1);
    float ss = fmaf(x.y, x.y, fmaf(x.x, x.x, prev));  // meaningful on odd lanes
#pragma unroll
    for (int mask = 16; mask >= 2; mask >>= 1) ss = ss + __shfl_xor_sync(0xffffffffu, ss, mask);
    ss = __shfl_sync(0xffffffffu, ss, lane | 1);
    const float r = rsqrtf(ss / static_cast<float>(kHeadDim) + p.norm_eps);
    const float2 g = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(
        p.gate + (static_cast<int64_t>(req) * kTokens + t) * p.gate_token_stride + h * kHeadDim + 2 * lane));
    const auto sig = [](float z) { return __fdividef(1.0f, 1.0f + __expf(-z)); };
    float y0 = x.x * r, y1 = x.y * r;
    y0 *= sig(g.x);
    y1 *= sig(g.y);
    v = __floats2bfloat162_rn(y0, y1);
  }
  *reinterpret_cast<__nv_bfloat162*>(
      p.out + ((static_cast<int64_t>(req) * kTokens + t) * p.num_q_heads + h) * kHeadDim + 2 * lane) = v;
}

}  // namespace fva

// ---------------------------------------------------------------------------------------------- host
namespace fva_host {

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

// Tensor map over a pool [slots, row_bytes] (bytes), box = 128 bytes x 128 rows, 128-byte swizzle.
inline CUtensorMap pool_map(const void* base, int64_t slots, int64_t row_bytes) {
  CUtensorMap map;
  const cuuint64_t dims[2] = {static_cast<cuuint64_t>(row_bytes), static_cast<cuuint64_t>(slots)};
  const cuuint64_t strides[1] = {static_cast<cuuint64_t>(row_bytes)};
  const cuuint32_t box[2] = {128, static_cast<cuuint32_t>(fva::kTile)};
  const cuuint32_t elem[2] = {1, 1};
  const CUresult r = encode_fn()(&map, CU_TENSOR_MAP_DATA_TYPE_UINT8, 2, const_cast<void*>(base), dims, strides, box,
                                 elem, CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B,
                                 CU_TENSOR_MAP_L2_PROMOTION_L2_256B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
  sglang::host::RuntimeCheck(r == CUDA_SUCCESS, "cuTensorMapEncodeTiled failed: ", static_cast<int>(r));
  return map;
}

}  // namespace fva_host

template <bool kUsePDL, bool kClusterMerge, bool kNormGate, int kQk, bool kUnfused>
struct FullVerifyAttnKernel {
  static void run(const tvm::ffi::TensorView q,           // [R*8, Hq, 64] e4m3, heads/dims contiguous (kQk: unused)
                  const tvm::ffi::TensorView k_pool,      // [slots, Hkv*64] e4m3, rows contiguous
                  const tvm::ffi::TensorView v_pool,      // [slots, Hkv*64]
                  const tvm::ffi::TensorView page_table,  // [R, pages] int32, inner contiguous
                  const tvm::ffi::TensorView seq_lens,    // [R] int32
                  const tvm::ffi::TensorView out,         // [R*8, Hq, 64] bf16, contiguous
                  const tvm::ffi::TensorView partial,     // fp32 workspace, >= R * Hkv * splits * 64 * 68
                  int64_t window_left, int64_t page_size, int64_t num_splits, double bmm1_scale, double bmm2_scale,
                  const tvm::ffi::TensorView gate,  // kNormGate: [R*8, >= Hq*64] bf16, inner contiguous
                  double norm_eps,
                  const tvm::ffi::TensorView qkv,        // kQk: [R*8, >= (2 Hq + 2 Hkv) * 64] bf16, inner contiguous
                  const tvm::ffi::TensorView cache_loc,  // kQk: [R*8] int64
                  double qk_eps) {
    using namespace sglang::host;
    if constexpr (kQk == 0)
      RuntimeCheck(q.dim() == 3 && q.size(2) == fva::kHeadDim && q.dtype().code == DLDataTypeCode::kDLFloat8_e4m3fn &&
                       q.stride(2) == 1 && q.stride(1) == fva::kHeadDim && q.stride(0) % 16 == 0,
                   "q must be e4m3 [tokens, heads, 64], heads and dims contiguous");
    static_assert(kQk == 0 || kQk == 2 || kQk == 4 || kQk == 8, "prologue columns per thread");
    RuntimeCheck(k_pool.dtype().code == DLDataTypeCode::kDLFloat8_e4m3fn &&
                     v_pool.dtype().code == DLDataTypeCode::kDLFloat8_e4m3fn && k_pool.dim() == 2 && v_pool.dim() == 2 &&
                     k_pool.stride(1) == 1 && v_pool.stride(1) == 1 && k_pool.stride(0) == k_pool.size(1) &&
                     v_pool.stride(0) == v_pool.size(1) && k_pool.size(1) == v_pool.size(1),
                 "pools must be contiguous e4m3 [slots, Hkv*64]");
    RuntimeCheck(out.dtype().code == DLDataTypeCode::kDLBfloat && out.dtype().bits == 16, "out must be bf16");
    RuntimeCheck(page_size == fva::kTile, "page size must be 128");
    const int64_t m = out.size(0), hq = out.size(1), reqs = m / fva::kTokens;
    const int64_t hkv = k_pool.size(1) / fva::kHeadDim;
    RuntimeCheck(m % fva::kTokens == 0 && hkv * fva::kQPerKv == hq && hkv % 2 == 0, "head layout");
    RuntimeCheck(out.dim() == 3 && out.size(0) == m && out.size(1) == hq && out.size(2) == fva::kHeadDim &&
                     out.stride(2) == 1 && out.stride(1) == fva::kHeadDim && out.stride(0) == hq * fva::kHeadDim,
                 "out layout");
    RuntimeCheck(seq_lens.dim() == 1 && seq_lens.size(0) == reqs && seq_lens.stride(0) == 1 &&
                     seq_lens.dtype().code == DLDataTypeCode::kDLInt && seq_lens.dtype().bits == 32,
                 "seq_lens [R] int32");
    RuntimeCheck(page_table.dim() == 2 && page_table.size(0) >= reqs && page_table.stride(1) == 1 &&
                     page_table.dtype().code == DLDataTypeCode::kDLInt && page_table.dtype().bits == 32,
                 "page_table [R, pages] int32");
    if constexpr (kClusterMerge) {
      // st.async (the record exchange) is defined only for clusters of more than one block (PTX ISA)
      RuntimeCheck(num_splits >= 2 && num_splits <= fva::kMaxCluster, "cluster merge needs 2..16 splits");
    } else {
      RuntimeCheck(num_splits >= 1 && partial.dtype().code == DLDataTypeCode::kDLFloat && partial.dtype().bits == 32 &&
                       partial.numel() >= reqs * hkv * num_splits * fva::kRows * fva::kRec,
                   "partial workspace too small");
    }
    const auto aligned16 = [](const void* ptr) { return reinterpret_cast<uintptr_t>(ptr) % 16 == 0; };
    RuntimeCheck((kQk > 0 || aligned16(q.data_ptr())) && aligned16(k_pool.data_ptr()) && aligned16(v_pool.data_ptr()) &&
                     aligned16(out.data_ptr()) && aligned16(partial.data_ptr()),
                 "16-byte alignment");
    if constexpr (kQk > 0) {
      RuntimeCheck(qkv.dim() == 2 && qkv.size(0) == m && qkv.size(1) >= (2 * hq + 2 * hkv) * fva::kHeadDim &&
                       qkv.stride(1) == 1 && qkv.dtype().code == DLDataTypeCode::kDLBfloat && qkv.dtype().bits == 16 &&
                       qkv.stride(0) % 8 == 0 && reinterpret_cast<uintptr_t>(qkv.data_ptr()) % 16 == 0,
                   "qkv must be bf16 [tokens, >= (2 Hq + 2 Hkv) * 64], inner contiguous, 16-byte aligned rows");
      RuntimeCheck(cache_loc.dim() == 1 && cache_loc.size(0) == m && cache_loc.stride(0) == 1 &&
                       cache_loc.dtype().code == DLDataTypeCode::kDLInt && cache_loc.dtype().bits == 64,
                   "cache_loc [tokens] int64");
    }

    fva::Params params{};
    params.q = kQk > 0 ? nullptr : static_cast<const uint8_t*>(q.data_ptr());
    params.page_table = static_cast<const int32_t*>(page_table.data_ptr());
    params.seq_lens = static_cast<const int32_t*>(seq_lens.data_ptr());
    params.partial = static_cast<float*>(partial.data_ptr());
    params.out = static_cast<__nv_bfloat16*>(out.data_ptr());
    params.q_token_stride = kQk > 0 ? 0 : q.stride(0);
    params.page_table_stride = page_table.stride(0);
    params.num_q_heads = static_cast<int32_t>(hq);
    params.num_kv_heads = static_cast<int32_t>(hkv);
    params.num_splits = static_cast<int32_t>(num_splits);
    params.window_left = static_cast<int32_t>(window_left);
    params.scale_log2 = static_cast<float>(bmm1_scale * M_LOG2E);
    params.out_scale = static_cast<float>(bmm2_scale / static_cast<double>(fva::kPScale));
    if constexpr (kNormGate) {
      RuntimeCheck(gate.dtype().code == DLDataTypeCode::kDLBfloat && gate.dtype().bits == 16 && gate.dim() == 2 &&
                       gate.size(0) == m && gate.size(1) >= hq * fva::kHeadDim && gate.stride(1) == 1 &&
                       gate.stride(0) % 2 == 0 && reinterpret_cast<uintptr_t>(gate.data_ptr()) % 4 == 0,
                   "gate must be bf16 [tokens, >= Hq*64], inner contiguous, 4-byte aligned rows");
      params.gate = static_cast<const __nv_bfloat16*>(gate.data_ptr());
      params.gate_token_stride = gate.stride(0);
      params.norm_eps = static_cast<float>(norm_eps);
    }
    if constexpr (kQk > 0) {
      params.qkv = static_cast<const __nv_bfloat16*>(qkv.data_ptr());
      params.qkv_token_stride = qkv.stride(0);
      params.cache_loc = static_cast<const int64_t*>(cache_loc.data_ptr());
      params.k_store = static_cast<uint8_t*>(k_pool.data_ptr());
      params.v_store = static_cast<uint8_t*>(v_pool.data_ptr());
      params.pool_row_bytes = k_pool.size(1);
      params.qk_eps = static_cast<float>(qk_eps);
    }
    const CUtensorMap k_map = fva_host::pool_map(k_pool.data_ptr(), k_pool.size(0), k_pool.size(1));
    const CUtensorMap v_map = fva_host::pool_map(v_pool.data_ptr(), v_pool.size(0), v_pool.size(1));

    constexpr auto kernel = fva::full_verify_attn_kernel<kUsePDL, kClusterMerge, kNormGate, kQk, kUnfused>;
    constexpr int kSmem = kQk > 0 ? fva::kSmBytesQk : fva::kSmBytes;
    constexpr auto merge = fva::full_verify_attn_merge_kernel<kUsePDL, kNormGate>;
    static bool configured = false;
    if (!configured) {
      RuntimeDeviceCheck(::cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, kSmem));
      if constexpr (kClusterMerge)
        RuntimeDeviceCheck(::cudaFuncSetAttribute(kernel, cudaFuncAttributeNonPortableClusterSizeAllowed, 1));
      configured = true;
    }
    if (reqs == 0) return;
    const dim3 grid(static_cast<unsigned>(num_splits), static_cast<unsigned>(hkv / 2), static_cast<unsigned>(reqs));
    if constexpr (kClusterMerge) {
      LaunchKernel(grid, fva::kThreads, out.device(), kSmem)
          .config({.use_pdl = kUsePDL, .cluster_dim = dim3(static_cast<unsigned>(num_splits), 1, 1)})(kernel, params,
                                                                                                    k_map, v_map);
      return;
    }
    LaunchKernel(grid, fva::kThreads, out.device(), kSmem)
        .config({.use_pdl = kUsePDL})(kernel, params, k_map, v_map);
    // warps per row in the merge: enough warps in total for the row count (tunable, bench/bench_full_verify_attn.py)
    const int64_t rows = reqs * hkv * fva::kRows;
    const int groups = rows <= 256 ? 8 : rows <= 512 ? 4 : rows <= 1024 ? 2 : 1;
    LaunchKernel(dim3(static_cast<unsigned>(fva::kRows * groups / 8), static_cast<unsigned>(hkv),
                      static_cast<unsigned>(reqs)),
                 256, out.device(), 0)
        .config({.use_pdl = kUsePDL})(merge, params, groups);
  }
};

}  // namespace muse_vr
