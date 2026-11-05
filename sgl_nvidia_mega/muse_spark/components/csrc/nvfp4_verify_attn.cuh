// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
/**
 * \brief Paged NVFP4-KV attention for speculative verify with tensor-core MMA (tcgen05): 8 query tokens per request,
 * 8 query heads per KV head, head_dim 64, page size 128. Same structure as the plugin's FP8 verify kernel
 * (full_verify_attn): the key range of a request (all keys, or a sliding window) is split over thread blocks in
 * proportion to its length; every block writes a partial result (unnormalized output, row maximum, row sum) and a
 * second kernel merges them. The difference: the K/V pool rows are packed E2M1 with one E4M3 block scale per 16 values
 * (SGLang's native NVFP4 layout for trtllm-gen), and a stage of dequantization warps turns each tile into the FP8
 * operands the MMA reads.
 *
 * Shapes (per call): q [R * 8, Hq, 64] bf16 (heads and dims contiguous); K/V pools [slots, Hkv * 32] uint8 (two E2M1
 * values per byte, element 2i in the low nibble; slot = page * 128 + offset); K/V scale pools [pages, Hkv, 128, 4]
 * E4M3 (V scales four-token interleaved inside a page: the scale of token t, block i sits at token (t & ~3) + i,
 * column t & 3); page_table [R, >= pages] int32; seq_lens [R] int32 (KV length including the 8 verify tokens);
 * out [R * 8, Hq, 64] bf16, contiguous. Hkv must be even.
 *
 * Masking: token t (0..7) of request r sits at position p = seq_len - 8 + t and attends keys [lo, p], where
 * lo = max(0, p - W) with W = window_left >= 0, and lo = 0 for window_left < 0 (full attention).
 *
 * Arithmetic, in order (per query row):
 *   q   = e4m3(q_bf16)                                   (round to nearest even, satfinite; the stock path rounds q
 *                                                          to e4m3 the same way before its kernel)
 *   k   = e4m3(f16(e2m1) * f16(scale))                   (the product is exact in f16; one rounding to e4m3, round to
 *                                                          nearest even, satfinite); v likewise
 *   s   = fp32 dot product of e4m3 q and e4m3 k (tensor-core MMA, fp32 accumulation)
 *   then exactly the FP8 verify kernel's online softmax, e4m3 probabilities (p * 448), P V MMA, split merge, and
 *   out = bf16(O * (bmm2_scale / 448) / L)
 * The global KV scale is folded into bmm1_scale / bmm2_scale by the caller (as for the stock kernel).
 *
 * Work split: grid (S, Hkv / 2, R). Block (s, j, r) serves KV heads 2j and 2j + 1 of request r for pages
 * [first + s * n, first + (s + 1) * n) of the request's key range, n = ceil(pages / S). 14 warps:
 *   warp 8 (one thread): per tile, the packed K and V rows of the two KV heads (two 128-row x 64-byte boxes, tensor
 *     memory accelerator) and their two 1024-byte scale blocks (bulk copies) into a ring of kRawStages raw stages;
 *   warps 10-13: dequantize a raw stage into one of kF8Stages FP8 stages ([128 keys][128 B], 128-byte swizzle: the
 *     layout the FP8 kernel's tensor-memory-accelerator loads produce), then release the raw stage;
 *   warp 9 (one thread): the MMAs (tcgen05.mma kind::f8f6f4, M = 64 rows per KV head at tensor-memory lane offsets 0
 *     and 16): S = Q K^T and P V, as in the FP8 kernel;
 *   warps 0-7: Q (bf16 -> e4m3 into shared memory) and the softmax, two threads per query row, as in the FP8 kernel.
 * Hand-offs use mbarriers: raw stage full (TMA complete_tx), raw stage free (128 dequantization-thread arrivals),
 * FP8 stage full (128 arrivals after fence.proxy.async.shared::cta), FP8 stage free (tcgen05.commit), scores ready /
 * probabilities written / output tile ready as in the FP8 kernel.
 *
 * Ordering: q, seq_lens and the page table may be written by the preceding kernels, so the kernel reads them only
 * after griddepcontrol.wait (with use_pdl). The pool rows are written by preceding kernels with generic-proxy stores;
 * the loading thread executes fence.proxy.async.global before its first copy. Pool rows past the end of a request
 * inside its last page, and rows before the window, are loaded with the page and masked out of the probabilities
 * (their values are finite: the pools are allocated zero-filled and only finite values are stored into them).
 */

#include <sgl_kernel/tensor.h>
#include <sgl_kernel/utils.h>

#include <sgl_kernel/runtime.cuh>
#include <sgl_kernel/utils.cuh>

#include <cuda.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <dlpack/dlpack.h>
#include <tvm/ffi/container/tensor.h>

#include <cmath>
#include <cstdint>

namespace muse_vr {

namespace nva {

constexpr int kHeadDim = 64;
constexpr int kQPerKv = 8;
constexpr int kTokens = 8;
constexpr int kRows = 64;  // query rows per KV head: token * 8 + head
constexpr int kTile = 128;  // keys per tile = page size
constexpr int kF8Stages = 3;
constexpr int kRawStages = 3;
constexpr int kSoftmaxWarps = 8;  // two threads per query row
constexpr int kDequantWarps = 4;
constexpr int kLoadWarp = kSoftmaxWarps, kMmaWarp = kSoftmaxWarps + 1, kDqWarp0 = kSoftmaxWarps + 2;
constexpr int kDqThreads = 32 * kDequantWarps;
constexpr int kThreads = 32 * (kSoftmaxWarps + 2 + kDequantWarps);
constexpr int kRec = 68;  // floats per partial row record: 64 outputs, m, l, 2 pad
constexpr float kPScale = 448.0f;

// Shared memory (bytes from a 1024-aligned base)
constexpr int kF8StageBytes = kTile * 128;   // FP8 K or V stage: 128 keys x (2 heads x 64 B)
constexpr int kRawBytes = kTile * 64;        // packed K or V: 128 keys x (2 heads x 32 B)
constexpr int kScaleBytes = 2 * kTile * 4;   // block scales of the two heads: [2][128][4]
constexpr int kSmK = 0;
constexpr int kSmV = kSmK + kF8Stages * kF8StageBytes;
constexpr int kSmRawK = kSmV + kF8Stages * kF8StageBytes;
constexpr int kSmRawV = kSmRawK + kRawStages * kRawBytes;
constexpr int kSmRawKs = kSmRawV + kRawStages * kRawBytes;
constexpr int kSmRawVs = kSmRawKs + kRawStages * kScaleBytes;
constexpr int kSmP = kSmRawVs + kRawStages * kScaleBytes;  // [2 buffers][2 heads][64 rows][128 B]
constexpr int kSmQ = kSmP + 2 * 2 * kRows * 128;          // [2 heads][64 rows][64 B]
constexpr int kSmRed = kSmQ + 2 * kRows * 64;             // [2 tile parities + row sums][128 rows][2 halves] fp32
constexpr int kSmPages = kSmRed + 3 * 128 * 2 * 4;        // page numbers of the split (int32), up to kMaxTiles
constexpr int kMaxTiles = 2048;
constexpr int kSmBar = kSmPages + kMaxTiles * 4;          // mbarriers + tensor-memory address
constexpr int kNumBars = 4 * kRawStages + 2 * kF8Stages + 6;  // raw full/free, f8 full/free, s/p/o x2
constexpr int kSmBytes = kSmBar + 8 * kNumBars + 16 + 1024;  // + tmem slot + alignment slack
constexpr uint32_t kTmemCols = 512;                        // S0 [0,128) S1 [128,256) O0 [256,320) O1 [320,384)
constexpr uint32_t kStageTx = 2 * kRawBytes + 2 * kScaleBytes;

struct Params {
  const __nv_bfloat16* __restrict__ q;
  const int32_t* __restrict__ page_table;
  const int32_t* __restrict__ seq_lens;
  const uint8_t* __restrict__ k_scales;  // [pages][Hkv][128][4]
  const uint8_t* __restrict__ v_scales;
  float* __restrict__ partial;  // [R][Hkv][64 rows][S][kRec]
  __nv_bfloat16* __restrict__ out;
  int64_t q_token_stride;  // elements
  int64_t page_table_stride;
  int32_t num_q_heads;
  int32_t num_kv_heads;
  int32_t num_splits;
  int32_t window_left;  // < 0: full attention
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
__device__ __forceinline__ void tma_load_2d(uint32_t dst, const CUtensorMap* map, int32_t c0, int32_t c1, uint32_t bar) {
  asm volatile(
      "cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1, {%2, %3}], [%4];\n" ::"r"(dst),
      "l"(reinterpret_cast<uint64_t>(map)), "r"(c0), "r"(c1), "r"(bar)
      : "memory");
}
__device__ __forceinline__ void bulk_load(uint32_t dst, const void* src, uint32_t bytes, uint32_t bar) {
  asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];\n" ::"r"(dst),
               "l"(reinterpret_cast<uint64_t>(src)), "r"(bytes), "r"(bar)
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

// Shared-memory accesses through 32-bit shared-window addresses (the generic pointer into dynamic shared memory loses its
// address space after the alignment arithmetic).
__device__ __forceinline__ uint4 lds128(uint32_t a) {
  uint4 v;
  asm volatile("ld.shared.v4.u32 {%0, %1, %2, %3}, [%4];\n" : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w) : "r"(a));
  return v;
}
__device__ __forceinline__ uint32_t lds_u8(uint32_t a) {
  uint16_t v;
  asm volatile("ld.shared.u8 %0, [%1];\n" : "=h"(v) : "r"(a));
  return v;
}
__device__ __forceinline__ void sts_f32(uint32_t a, float v) {
  asm volatile("st.shared.f32 [%0], %1;\n" ::"r"(a), "f"(v) : "memory");
}
__device__ __forceinline__ float2 lds_f32x2(uint32_t a) {
  float2 v;
  asm volatile("ld.shared.v2.f32 {%0, %1}, [%2];\n" : "=f"(v.x), "=f"(v.y) : "r"(a) : "memory");
  return v;
}
__device__ __forceinline__ void sts128(uint32_t a, uint4 v) {
  asm volatile("st.shared.v4.u32 [%0], {%1, %2, %3, %4};\n" ::"r"(a), "r"(v.x), "r"(v.y), "r"(v.z), "r"(v.w) : "memory");
}

// Four packed bytes = 8 E2M1 values (element 2i in the low nibble of byte i) times one E4M3 block scale -> 8 E4M3
// values (two words, element order kept). f16(e2m1) * f16(scale) is exact; one rounding to e4m3.
__device__ __forceinline__ uint2 dequant8(uint32_t packed, uint32_t scale2_f16x2) {
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
template <bool kUsePDL>
__global__ void __launch_bounds__(kThreads, 1)
    nvfp4_verify_attn_kernel(const Params p, const __grid_constant__ CUtensorMap k_map,
                             const __grid_constant__ CUtensorMap v_map) {
  extern __shared__ uint8_t smem_raw[];
  uint8_t* const sm = reinterpret_cast<uint8_t*>((reinterpret_cast<uintptr_t>(smem_raw) + 1023) & ~uintptr_t{1023});
  uint64_t* const bars = reinterpret_cast<uint64_t*>(sm + kSmBar);
  const uint32_t bar0 = smem_u32(bars);
  const auto raw_full = [&](int s) { return bar0 + 8u * s; };
  const auto raw_free = [&](int s) { return bar0 + 8u * (kRawStages + s); };
  const auto f8_full = [&](int s) { return bar0 + 8u * (2 * kRawStages + s); };
  const auto f8_free = [&](int s) { return bar0 + 8u * (2 * kRawStages + kF8Stages + s); };
  constexpr int kB = 2 * kRawStages + 2 * kF8Stages;
  const auto s_full_bar = [&](int b) { return bar0 + 8u * (kB + b); };
  const auto p_full_bar = [&](int b) { return bar0 + 8u * (kB + 2 + b); };
  const auto o_full_bar = [&](int b) { return bar0 + 8u * (kB + 4 + b); };
  uint32_t* const tmem_slot = reinterpret_cast<uint32_t*>(sm + kSmBar + 8 * kNumBars);

  const int split = blockIdx.x, pair = blockIdx.y, req = blockIdx.z;
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;

  // Barriers and tensor memory first: they touch nothing the preceding kernels write.
  if (tid == 0) {
#pragma unroll
    for (int s = 0; s < kRawStages; ++s) {
      mbar_init(raw_full(s), 1);
      mbar_init(raw_free(s), kDqThreads);
    }
#pragma unroll
    for (int s = 0; s < kF8Stages; ++s) {
      mbar_init(f8_full(s), kDqThreads);
      mbar_init(f8_free(s), 1);
    }
#pragma unroll
    for (int b = 0; b < 2; ++b) {
      mbar_init(s_full_bar(b), 1);
      mbar_init(p_full_bar(b), 32 * kSoftmaxWarps);
      mbar_init(o_full_bar(b), 1);
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

  if constexpr (kUsePDL) asm volatile("griddepcontrol.wait;\n" ::: "memory");
  const int seq_len = p.seq_lens[req];
  int p0, p1, lo;
  split_pages(seq_len, p.window_left, p.num_splits, split, p0, p1, lo);
  const int ntiles = p1 - p0;
  if (ntiles <= 0) {  // uniform over the block: no partial for this split
    if (p.num_splits == 1 && warp < kSoftmaxWarps) {  // no merge either: zero rows, as the merge writes them
      const int hh = lane >> 4, row = 16 * (warp & 3) + (lane & 15), kvh = 2 * pair + hh;
      __nv_bfloat16* const dst =
          p.out + ((static_cast<int64_t>(req) * kTokens + (row >> 3)) * p.num_q_heads + kvh * kQPerKv + (row & 7)) *
                      kHeadDim +
          32 * (warp >> 2);
#pragma unroll
      for (int dd = 0; dd < 32; dd += 8) *reinterpret_cast<uint4*>(dst + dd) = make_uint4(0u, 0u, 0u, 0u);
    }
    if (warp == 0)
      asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, %1;\n" ::"r"(tmem), "r"(kTmemCols) : "memory");
    return;
  }
  int32_t* const pages = reinterpret_cast<int32_t*>(sm + kSmPages);
  const int32_t* const pt = p.page_table + static_cast<int64_t>(req) * p.page_table_stride + p0;
  const int64_t scale_pair_bytes = static_cast<int64_t>(kScaleBytes);
  const auto issue_tile = [&](int t) {
    const int s = t % kRawStages;
    const int page = t < kMaxTiles ? pages[t] : pt[t];
    const int64_t sc = (static_cast<int64_t>(page) * p.num_kv_heads + 2 * pair) * (kTile * 4);
    mbar_expect_tx(raw_full(s), kStageTx);
    tma_load_2d(smem_u32(sm + kSmRawK + s * kRawBytes), &k_map, pair * 64, page * kTile, raw_full(s));
    tma_load_2d(smem_u32(sm + kSmRawV + s * kRawBytes), &v_map, pair * 64, page * kTile, raw_full(s));
    bulk_load(smem_u32(sm + kSmRawKs + s * kScaleBytes), p.k_scales + sc, scale_pair_bytes, raw_full(s));
    bulk_load(smem_u32(sm + kSmRawVs + s * kScaleBytes), p.v_scales + sc, scale_pair_bytes, raw_full(s));
  };
  if (warp == kLoadWarp) {
    for (int t = lane; t < min(ntiles, kMaxTiles); t += 32) pages[t] = pt[t];
    __syncwarp();
    if (lane == 0) {
      // Pool rows written by preceding kernels with generic-proxy stores; order them before these async-proxy reads.
      asm volatile("fence.proxy.async.global;\n" ::: "memory");
      for (int t = 0; t < min(ntiles, kRawStages); ++t) issue_tile(t);
    }
  } else if (warp < kSoftmaxWarps) {
    // Q rows of the two KV heads, bf16 -> e4m3 -> [head][row][64 B] in the 64-byte swizzle (16-byte chunk c of row r
    // at c ^ (r/2 % 4)). Item i: head hh, row r, 16-byte output chunk c (= 16 dims).
    uint8_t* const sq = sm + kSmQ;
    for (int i = tid; i < 2 * kRows * 4; i += 32 * kSoftmaxWarps) {
      const int hh = i >> 8, r = (i >> 2) & 63, c = i & 3;
      const int t = r >> 3, g = r & 7;
      const __nv_bfloat16* src = p.q + (static_cast<int64_t>(req) * kTokens + t) * p.q_token_stride +
                                 ((2 * pair + hh) * kQPerKv + g) * kHeadDim + 16 * c;
      const uint4 a = *reinterpret_cast<const uint4*>(src);
      const uint4 b = *reinterpret_cast<const uint4*>(src + 8);
      float f[16];
      const uint32_t w[8] = {a.x, a.y, a.z, a.w, b.x, b.y, b.z, b.w};
#pragma unroll
      for (int j = 0; j < 8; ++j) {
        const float2 v = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&w[j]));
        f[2 * j] = v.x;
        f[2 * j + 1] = v.y;
      }
      uint4 o;
      o.x = e4m3x4(f[0], f[1], f[2], f[3]);
      o.y = e4m3x4(f[4], f[5], f[6], f[7]);
      o.z = e4m3x4(f[8], f[9], f[10], f[11]);
      o.w = e4m3x4(f[12], f[13], f[14], f[15]);
      *reinterpret_cast<uint4*>(sq + hh * kRows * 64 + r * 64 + ((c ^ ((r >> 1) & 3)) << 4)) = o;
    }
    fence_proxy_async();
    asm volatile("bar.arrive 5, %0;\n" ::"r"(32 * (kSoftmaxWarps + 1)) : "memory");
  } else if (warp == kMmaWarp) {
    asm volatile("bar.sync 5, %0;\n" ::"r"(32 * (kSoftmaxWarps + 1)) : "memory");
  }

  if (warp == kLoadWarp) {
    // ------------------------------------------------------------------ loads
    if (lane == 0) {
      for (int t = kRawStages; t < ntiles; ++t) {
        mbar_wait(raw_free(t % kRawStages), (t / kRawStages - 1) & 1);
        issue_tile(t);
      }
    }
  } else if (warp >= kDqWarp0) {
    // ------------------------------------------------------------------ dequantization (warps 10-13)
    // Per tile and K/V: 512 input chunks of 16 bytes (row = key, chunk q of row: head q / 2, dims 32 (q % 2) ..
    // +32). Thread d handles chunks d, d + 128, d + 256, d + 384; each becomes two 16-byte e4m3 chunks of the FP8
    // stage row (head hh's 64 bytes at 64 hh, 128-byte swizzle: chunk c of row r at c ^ (r % 8)).
    const int d = tid - 32 * kDqWarp0;
    for (int t = 0; t < ntiles; ++t) {
      const int rs = t % kRawStages, fs = t % kF8Stages;
      mbar_wait(raw_full(rs), (t / kRawStages) & 1);
      if (t >= kF8Stages) mbar_wait(f8_free(fs), (t / kF8Stages - 1) & 1);
#pragma unroll
      for (int kv = 0; kv < 2; ++kv) {
        const uint32_t raw = smem_u32(sm + (kv ? kSmRawV : kSmRawK) + rs * kRawBytes);
        const uint32_t scl = smem_u32(sm + (kv ? kSmRawVs : kSmRawKs) + rs * kScaleBytes);
        const uint32_t dst = smem_u32(sm + (kv ? kSmV : kSmK) + fs * kF8StageBytes);
#pragma unroll
        for (int j = 0; j < 4; ++j) {
          const int q = d + 128 * j;
          const int row = q >> 2, part = q & 3, hh = part >> 1, blk0 = (part & 1) * 2;
          const uint4 in = lds128(raw + 16 * q);
          // block scales blk0, blk0 + 1 of (head hh, key row): K plain, V four-token interleaved inside the page
          const uint32_t sbase = scl + hh * kTile * 4;
          const uint32_t a0 = kv == 0 ? sbase + row * 4 + blk0 : sbase + ((row & ~3) + blk0) * 4 + (row & 3);
          const uint32_t a1 = kv == 0 ? a0 + 1 : a0 + 4;
          const uint32_t f0 = scale_f16x2(lds_u8(a0)), f1 = scale_f16x2(lds_u8(a1));
          const uint2 a = dequant8(in.x, f0), b = dequant8(in.y, f0);  // 16 values of block blk0
          const uint2 c = dequant8(in.z, f1), e = dequant8(in.w, f1);  // 16 values of block blk0 + 1
          const int c0 = hh * 4 + (part & 1) * 2;                      // output chunk index in the 128-byte row
          const uint32_t drow = dst + row * 128;
          sts128(drow + ((c0 ^ (row & 7)) << 4), make_uint4(a.x, a.y, b.x, b.y));
          sts128(drow + (((c0 + 1) ^ (row & 7)) << 4), make_uint4(c.x, c.y, e.x, e.y));
        }
      }
      fence_proxy_async();
      mbar_arrive(f8_full(fs));
      mbar_arrive(raw_free(rs));
    }
  } else if (warp == kMmaWarp) {
    // ------------------------------------------------------------------ MMA issue
    if (lane == 0) {
      constexpr uint32_t id_qk = idesc_f8(64, 128, 0), id_pv = idesc_f8(64, 64, 1);
      const uint32_t sq = smem_u32(sm + kSmQ), sp = smem_u32(sm + kSmP);
      const auto qk = [&](int t) {
        const int s = t % kF8Stages, b = t & 1;
        mbar_wait(f8_full(s), (t / kF8Stages) & 1);
        tc_fence_after();
        const uint32_t sk = smem_u32(sm + kSmK + s * kF8StageBytes);
#pragma unroll
        for (int hh = 0; hh < 2; ++hh) {
#pragma unroll
          for (int k = 0; k < 2; ++k)
            mma_f8(tmem + ((16u * hh) << 16) + 128u * b, sdesc(sq + hh * kRows * 64 + 32 * k, 512, kSw64),
                   sdesc(sk + 64 * hh + 32 * k, 1024, kSw128), id_qk, k);
        }
        mma_commit(s_full_bar(b));
      };
      qk(0);
      for (int t = 0; t < ntiles; ++t) {
        const int s = t % kF8Stages, b = t & 1;
        if (t + 1 < ntiles) qk(t + 1);  // its score buffer was released by the arrival waited below for tile t - 1
        mbar_wait(p_full_bar(b), (t >> 1) & 1);
        tc_fence_after();
        const uint32_t sv = smem_u32(sm + kSmV + s * kF8StageBytes);
#pragma unroll
        for (int hh = 0; hh < 2; ++hh) {
#pragma unroll
          for (int k = 0; k < 4; ++k)
            mma_f8(tmem + ((16u * hh) << 16) + 256u + 64u * b,
                   sdesc(sp + (b * 2 + hh) * kRows * 128 + 32 * k, 1024, kSw128),
                   sdesc(sv + 64 * hh + 4096 * k, 1024, kSw128), id_pv, k);
        }
        mma_commit(o_full_bar(b));
        mma_commit(f8_free(s));
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
    const uint32_t red_s = smem_u32(sm + kSmRed);
    uint8_t* const prow0 = sm + kSmP + hh * kRows * 128 + row * 128;
    float o[32];
#pragma unroll
    for (int dd = 0; dd < 32; ++dd) o[dd] = 0.0f;
    float m = -INFINITY, l = 0.0f, ar_prev = 0.0f, at_prev = 0.0f;
    for (int t = 0; t < ntiles; ++t) {
      const int b = t & 1;
      const int key0 = (p0 + t) * kTile + 64 * half;  // this thread's first key of the tile
      mbar_wait(s_full_bar(b), (t >> 1) & 1);
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
      float mx[22];
#pragma unroll
      for (int i = 0; i < 21; ++i) mx[i] = max3(x[3 * i], x[3 * i + 1], x[3 * i + 2]);
      mx[21] = x[63];
#pragma unroll
      for (int i = 0; i < 7; ++i) mx[i] = max3(mx[3 * i], mx[3 * i + 1], mx[3 * i + 2]);
      mx[0] = max3(max3(mx[0], mx[1], mx[2]), max3(mx[3], mx[4], mx[5]), fmaxf(mx[6], mx[21]));
      sts_f32(red_s + 4u * ((b * 128 + lrow) * 2 + half), mx[0]);
      asm volatile("bar.sync %0, %1;\n" ::"r"(1 + quarter), "r"(64) : "memory");
      const float2 pm = lds_f32x2(red_s + 4u * ((b * 128 + lrow) * 2));
      const float smax = fmaxf(pm.x, pm.y);
      const float mtile = smax == -INFINITY ? -INFINITY : smax * p.scale_log2;
      const float mu = mtile == -INFINITY ? 0.0f : mtile;
      const float mnew = fmaxf(m, mtile);
      const float ar = m == -INFINITY ? 0.0f : fast_exp2(m - mnew);
      const float at = mtile == -INFINITY ? 0.0f : fast_exp2(mtile - mnew);
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
      uint8_t* const prow = prow0 + b * 2 * kRows * 128;
#pragma unroll
      for (int c = 0; c < 4; ++c) {
        uint4 w;
        w.x = e4m3x4(x[16 * c + 0], x[16 * c + 1], x[16 * c + 2], x[16 * c + 3]);
        w.y = e4m3x4(x[16 * c + 4], x[16 * c + 5], x[16 * c + 6], x[16 * c + 7]);
        w.z = e4m3x4(x[16 * c + 8], x[16 * c + 9], x[16 * c + 10], x[16 * c + 11]);
        w.w = e4m3x4(x[16 * c + 12], x[16 * c + 13], x[16 * c + 14], x[16 * c + 15]);
        sts128(smem_u32(prow) + (((4 * half + c) ^ (row & 7)) << 4), w);
      }
      fence_proxy_async();
      tc_fence_before();
      mbar_arrive(p_full_bar(b));
      if (t > 0) {
        const int bp = (t - 1) & 1;
        mbar_wait(o_full_bar(bp), ((t - 1) >> 1) & 1);
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
    {
      const int bp = (ntiles - 1) & 1;
      mbar_wait(o_full_bar(bp), ((ntiles - 1) >> 1) & 1);
      tc_fence_after();
      float v[32];
      tmem_ld32(tmem + lane_base + 256u + 64u * bp + 32u * half, v);
      tmem_wait_ld();
#pragma unroll
      for (int i = 0; i < 32; ++i) o[i] = fmaf(o[i], ar_prev, v[i] * at_prev);
    }
    sts_f32(red_s + 4u * ((256 + lrow) * 2 + half), l);
    asm volatile("bar.sync %0, %1;\n" ::"r"(1 + quarter), "r"(64) : "memory");
    const float2 pl = lds_f32x2(red_s + 4u * ((256 + lrow) * 2));
    const float lrowsum = pl.x + pl.y;
    const int kvh = 2 * pair + hh;
    if (p.num_splits == 1) {
      // One split: normalize and write the bf16 output here (no merge launch), with the merge kernel's arithmetic
      // for a single record - accumulators 0 + exp2(m - m) * o and 0 + exp2(m - m) * l - so the bytes are the merged
      // output's.
      const float w1 = m == -INFINITY ? 0.0f : fast_exp2(m - m);
      const float lw = fmaf(w1, lrowsum, 0.0f);
      const float f = lw > 0.0f ? p.out_scale / lw : 0.0f;
      __nv_bfloat16* const dst =
          p.out + ((static_cast<int64_t>(req) * kTokens + (row >> 3)) * p.num_q_heads + kvh * kQPerKv + (row & 7)) *
                      kHeadDim +
          32 * half;
#pragma unroll
      for (int dd = 0; dd < 32; dd += 8) {
        __nv_bfloat162 b[4];
#pragma unroll
        for (int j = 0; j < 4; ++j)
          b[j] = __floats2bfloat162_rn(fmaf(w1, o[dd + 2 * j], 0.0f) * f, fmaf(w1, o[dd + 2 * j + 1], 0.0f) * f);
        *reinterpret_cast<uint4*>(dst + dd) = *reinterpret_cast<const uint4*>(b);
      }
    } else {
      float* rec = p.partial +
                   (((static_cast<int64_t>(req) * p.num_kv_heads + kvh) * kRows + row) * p.num_splits + split) * kRec;
#pragma unroll
      for (int dd = 0; dd < 32; dd += 4)
        *reinterpret_cast<float4*>(rec + 32 * half + dd) = make_float4(o[dd], o[dd + 1], o[dd + 2], o[dd + 3]);
      if (half == 0) *reinterpret_cast<float2*>(rec + kHeadDim) = make_float2(m, lrowsum);
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
template <bool kUsePDL>
__global__ void __launch_bounds__(256) nvfp4_verify_attn_merge_kernel(const Params p, int merge_groups) {
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
  const __nv_bfloat162 v = __floats2bfloat162_rn(acc.x * f, acc.y * f);
  *reinterpret_cast<__nv_bfloat162*>(
      p.out + ((static_cast<int64_t>(req) * kTokens + t) * p.num_q_heads + h) * kHeadDim + 2 * lane) = v;
}

}  // namespace nva

// ---------------------------------------------------------------------------------------------- host
namespace nva_host {

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

// Tensor map over a packed pool [slots, row_bytes] (bytes), box = 64 bytes x 128 rows, no swizzle.
inline CUtensorMap pool_map(const void* base, int64_t slots, int64_t row_bytes) {
  CUtensorMap map;
  const cuuint64_t dims[2] = {static_cast<cuuint64_t>(row_bytes), static_cast<cuuint64_t>(slots)};
  const cuuint64_t strides[1] = {static_cast<cuuint64_t>(row_bytes)};
  const cuuint32_t box[2] = {64, static_cast<cuuint32_t>(nva::kTile)};
  const cuuint32_t elem[2] = {1, 1};
  const CUresult r = encode_fn()(&map, CU_TENSOR_MAP_DATA_TYPE_UINT8, 2, const_cast<void*>(base), dims, strides, box,
                                 elem, CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_NONE,
                                 CU_TENSOR_MAP_L2_PROMOTION_L2_256B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
  sglang::host::RuntimeCheck(r == CUDA_SUCCESS, "cuTensorMapEncodeTiled failed: ", static_cast<int>(r));
  return map;
}

}  // namespace nva_host

template <bool kUsePDL>
struct Nvfp4VerifyAttnKernel {
  static void run(const tvm::ffi::TensorView q,           // [R*8, Hq, 64] bf16, heads/dims contiguous
                  const tvm::ffi::TensorView k_pool,      // [slots, Hkv*32] uint8, rows contiguous
                  const tvm::ffi::TensorView v_pool,      // [slots, Hkv*32] uint8
                  const tvm::ffi::TensorView k_scales,    // [pages, Hkv, 128, 4] e4m3 (uint8 view), contiguous
                  const tvm::ffi::TensorView v_scales,    // same
                  const tvm::ffi::TensorView page_table,  // [R, pages] int32, inner contiguous
                  const tvm::ffi::TensorView seq_lens,    // [R] int32
                  const tvm::ffi::TensorView out,         // [R*8, Hq, 64] bf16, contiguous
                  const tvm::ffi::TensorView partial,     // fp32 workspace, >= R * Hkv * splits * 64 * 68
                  int64_t window_left, int64_t page_size, int64_t num_splits, double bmm1_scale, double bmm2_scale) {
    using namespace sglang::host;
    RuntimeCheck(q.dim() == 3 && q.size(2) == nva::kHeadDim && q.dtype().code == DLDataTypeCode::kDLBfloat &&
                     q.dtype().bits == 16 && q.stride(2) == 1 && q.stride(1) == nva::kHeadDim && q.stride(0) % 8 == 0,
                 "q must be bf16 [tokens, heads, 64], heads and dims contiguous");
    RuntimeCheck(k_pool.dim() == 2 && v_pool.dim() == 2 && k_pool.dtype().bits == 8 && v_pool.dtype().bits == 8 &&
                     k_pool.stride(1) == 1 && v_pool.stride(1) == 1 && k_pool.stride(0) == k_pool.size(1) &&
                     v_pool.stride(0) == v_pool.size(1) && k_pool.size(1) == v_pool.size(1) &&
                     k_pool.size(0) == v_pool.size(0),
                 "pools must be contiguous bytes [slots, Hkv*32]");
    RuntimeCheck(out.dtype().code == DLDataTypeCode::kDLBfloat && out.dtype().bits == 16, "out must be bf16");
    RuntimeCheck(page_size == nva::kTile, "page size must be 128");
    const int64_t m = q.size(0), hq = q.size(1), reqs = m / nva::kTokens;
    const int64_t hkv = k_pool.size(1) / (nva::kHeadDim / 2);
    RuntimeCheck(m % nva::kTokens == 0 && hkv * nva::kQPerKv == hq && hkv % 2 == 0, "head layout");
    RuntimeCheck(k_scales.dim() == 4 && v_scales.dim() == 4 && k_scales.size(1) == hkv && k_scales.size(2) == nva::kTile &&
                     k_scales.size(3) == nva::kHeadDim / 16 && k_scales.dtype().bits == 8 && v_scales.dtype().bits == 8 &&
                     k_scales.size(0) == v_scales.size(0) && k_scales.size(0) * nva::kTile == k_pool.size(0) &&
                     k_scales.stride(3) == 1 && k_scales.stride(0) == hkv * nva::kTile * 4 &&
                     v_scales.stride(3) == 1 && v_scales.stride(0) == hkv * nva::kTile * 4,
                 "scale pools must be contiguous [pages, Hkv, 128, 4] bytes covering the slots");
    RuntimeCheck(out.dim() == 3 && out.size(0) == m && out.size(1) == hq && out.size(2) == nva::kHeadDim &&
                     out.stride(2) == 1 && out.stride(1) == nva::kHeadDim && out.stride(0) == hq * nva::kHeadDim,
                 "out layout");
    RuntimeCheck(seq_lens.dim() == 1 && seq_lens.size(0) == reqs && seq_lens.stride(0) == 1 &&
                     seq_lens.dtype().code == DLDataTypeCode::kDLInt && seq_lens.dtype().bits == 32,
                 "seq_lens [R] int32");
    RuntimeCheck(page_table.dim() == 2 && page_table.size(0) >= reqs && page_table.stride(1) == 1 &&
                     page_table.dtype().code == DLDataTypeCode::kDLInt && page_table.dtype().bits == 32,
                 "page_table [R, pages] int32");
    RuntimeCheck(num_splits >= 1 && partial.dtype().code == DLDataTypeCode::kDLFloat && partial.dtype().bits == 32 &&
                     partial.numel() >= reqs * hkv * num_splits * nva::kRows * nva::kRec,
                 "partial workspace too small");
    const auto aligned16 = [](const void* ptr) { return reinterpret_cast<uintptr_t>(ptr) % 16 == 0; };
    RuntimeCheck(aligned16(q.data_ptr()) && aligned16(k_pool.data_ptr()) && aligned16(v_pool.data_ptr()) &&
                     aligned16(k_scales.data_ptr()) && aligned16(v_scales.data_ptr()) && aligned16(out.data_ptr()) &&
                     aligned16(partial.data_ptr()),
                 "16-byte alignment");

    nva::Params params{};
    params.q = static_cast<const __nv_bfloat16*>(q.data_ptr());
    params.page_table = static_cast<const int32_t*>(page_table.data_ptr());
    params.seq_lens = static_cast<const int32_t*>(seq_lens.data_ptr());
    params.k_scales = static_cast<const uint8_t*>(k_scales.data_ptr());
    params.v_scales = static_cast<const uint8_t*>(v_scales.data_ptr());
    params.partial = static_cast<float*>(partial.data_ptr());
    params.out = static_cast<__nv_bfloat16*>(out.data_ptr());
    params.q_token_stride = q.stride(0);
    params.page_table_stride = page_table.stride(0);
    params.num_q_heads = static_cast<int32_t>(hq);
    params.num_kv_heads = static_cast<int32_t>(hkv);
    params.num_splits = static_cast<int32_t>(num_splits);
    params.window_left = static_cast<int32_t>(window_left);
    params.scale_log2 = static_cast<float>(bmm1_scale * M_LOG2E);
    params.out_scale = static_cast<float>(bmm2_scale / static_cast<double>(nva::kPScale));
    const CUtensorMap k_map = nva_host::pool_map(k_pool.data_ptr(), k_pool.size(0), k_pool.size(1));
    const CUtensorMap v_map = nva_host::pool_map(v_pool.data_ptr(), v_pool.size(0), v_pool.size(1));

    constexpr auto kernel = nva::nvfp4_verify_attn_kernel<kUsePDL>;
    constexpr auto merge = nva::nvfp4_verify_attn_merge_kernel<kUsePDL>;
    static bool configured = false;
    if (!configured) {
      RuntimeDeviceCheck(::cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, nva::kSmBytes));
      configured = true;
    }
    if (reqs == 0) return;
    const dim3 grid(static_cast<unsigned>(num_splits), static_cast<unsigned>(hkv / 2), static_cast<unsigned>(reqs));
    LaunchKernel(grid, nva::kThreads, q.device(), nva::kSmBytes)
        .config({.use_pdl = kUsePDL})(kernel, params, k_map, v_map);
    if (num_splits == 1) return;  // the kernel wrote the output
    const int64_t rows = reqs * hkv * nva::kRows;
    const int groups = rows <= 256 ? 8 : rows <= 512 ? 4 : rows <= 1024 ? 2 : 1;
    LaunchKernel(dim3(static_cast<unsigned>(nva::kRows * groups / 8), static_cast<unsigned>(hkv),
                      static_cast<unsigned>(reqs)),
                 256, q.device(), 0)
        .config({.use_pdl = kUsePDL})(merge, params, groups);
  }
};

}  // namespace muse_vr
