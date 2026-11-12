// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// K3.1 DSA index query as one PDL kernel (replaces the stock decode chain sgl_kernel rmsnorm -> torch.mm wq_b ->
// act_quant -> bf16_head_gate_q_scale, plus the weights_proj torch.mm of prepare_gate). Same method and rounding
// points as stock (kimi_k31/dsa/indexer.py, sglang head_gate.py / act_quant_triton.py):
//   xn    = bf16(x * rsqrt(mean(x^2) + eps) * w_qa)           (x = projected[:, :1536], FP32 math)
//   q     = bf16(xn @ wq_b^T)                                  (FP32 accumulation; one BF16 rounding)
//   scale = exp2(ceil(log2(max(amax_128(q), 1e-4) * (1/448))))   (ue8m0; per row and head)
//   q8    = e4m3_rn(clamp(q / scale, +-448))
//   gate  = bf16(hidden @ weights_proj^T)                      (FP32 accumulation)
//   w     = float(bf16(gate * 32^-0.5)) * scale * 128^-0.5
// Only FP32 summation orders differ from stock. The scale is stock's act_quant(query, 128, scale_fmt="ue8m0") (the
// Indexer gets scale_fmt="ue8m0" from deepseek_v2.py), with the instructions Triton emits for it: libdevice log2f,
// ceil, ex2.approx.ftz, div.full.
//
// Grid: 32 index heads x kSplit CTAs (one cluster per head). Each CTA owns 128/kSplit rows of its head's wq_b
// block (staged into SMEM before griddepcontrol.wait: weights only) and 7168/kSplit columns of the gate dot. The
// head's amax and gate partials meet through DSMEM with one cluster barrier.
//
// Query-front form (kQF, K31MK_DSA_QF): the input projection also computed the index-K rows (indexer.wk) and the
// head-gate logits (indexer.weights_proj) as extra BF16 columns of `projected`, so this kernel reads the gate from
// there (no hidden-state loads, no gate dot) and also stores the new tokens' index-K rows, replacing the side-stream
// writer (projection GEMM, LayerNorm, fused_store_index_k_cache). The store keeps stock's method and rounding points:
//   k     = bf16(LayerNorm128(projected[:, k_col : k_col + 128]) * gamma + beta)   (FP32 math, FP32 affine)
//   scale = max(amax_128(k), 1e-4) / 448                                           (FP32, correctly rounded)
//   page[slot / 64]: k8 = e4m3_rn(clamp(k * (1 / scale), +-448)) at (slot % 64) * 128, scale at 8192 + (slot % 64) * 4
// Row b's index-K is stored by warp 7 of the CTA (head b, rank kSplit - 1) between the exchange's arrive and wait.
// Its readers (the index logits) run after this grid completes, as they already do for the query codes.
//
// MXFP4 index-K (kKFP4,; K31MK_DSA_IDX_FP4_FP8Q with the host's 68 B pool [pages, 64 * 68]): the same k, stored as
// idx_fp4.store_index_k / the host prefill writer store it (sglang fp4_indexer_quant.quantize_fp4_indexer_row, RNE off), with
// the instructions Triton emits for that row math (div.full):
//   per 32-group g: sf = max(amax_32(k) / 6, 1e-4); e = ceil_ue8m0(sf) (exponent + (mantissa != 0), clamped to [1, 254])
//   code = e2m1 threshold code of k / 2^(e - 127) (|x| > 0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5 after min(|x|, 6); sign unless 0)
//   page[slot / 64] (4352 B): codes at (slot % 64) * 64 (low nibble = even element), e at 4096 + (slot % 64) * 4 + g.
// kKFP4 = false instances take the unchanged FP8 store (the MXFP4 tail is an if constexpr branch).
//
// MXFP4 index query (kQ4,; K31MK_DSA_IDX_Q=fp4 with the MXFP4 index-K pages): the same BF16 q, quantized per 32-element
// group as idx_fp4.quantize_query does it (sglang fp4_indexer_quant row math, RNE off; the instructions Triton emits, div.full):
//   group g of head h = this cluster's ranks 2g and 2g + 1 (16 dims each): amax_32 = max of their exchanged amax
//   sf = max(amax_32 / 6, 1e-4); e = ceil_ue8m0(sf) (clamped to [1, 254]); code = e2m1 threshold code of q / 2^(e - 127)
//   q4[row, h, 64] (low nibble = even element; rank r writes bytes 8 r .. 8 r + 7), qsf[row, h, 4] (byte g = e of group g, written by
//   rank 2g), w = float(bf16(gate * 32^-0.5)) * 128^-0.5 (no q scale: the MX block scales feed the logits UMMA).
// kQ4 = false instances are the FP8 kernel unchanged (if constexpr branches; launched by run / run_copy / run_qf as before).
//
// 9-16 rows (query-front form only, K31MK_DSA_QF_MAX_ROWS=16): kR = 16 treats the rows as two n8 tiles of the m16n8k16
// product (one wq_b A fragment, two MMAs per k-step); warp w loads and normalizes rows w and w + 8; every per-row array is
// sized kR; two CTAs per SM (Smem<8, 16> ~110 KB). Each row's arithmetic is the 8-row kernel's, so the outputs are bitwise
// those of two 8-row launches on the row halves. The kR = 8 instances compile to the same SASS as before.
#include <sgl_kernel/tensor.h>
#include <sgl_kernel/utils.h>

#include <sgl_kernel/runtime.cuh>
#include <sgl_kernel/utils.cuh>

#include <dlpack/dlpack.h>
#include <tvm/ffi/container/tensor.h>

#include <cuda_bf16.h>
#include <cstdint>

namespace k31_index_q {

constexpr int kHeads = 32, kDim = 128, kLora = 1536, kHidden = 7168, kRows = 8;
constexpr int kThreads = 256;
constexpr int kStride = kLora * 2 + 16;  // padded SMEM row (bytes): conflict-free fragment loads


__device__ __forceinline__ uint32_t smem_addr(const void* p) {
  return static_cast<uint32_t>(__cvta_generic_to_shared(p));
}
__device__ __forceinline__ void cp_async16(void* dst, const void* src) {
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" ::"r"(smem_addr(dst)), "l"(src) : "memory");
}
__device__ __forceinline__ void cp_async_commit() { asm volatile("cp.async.commit_group;\n" ::: "memory"); }
__device__ __forceinline__ void cp_async_wait_all() { asm volatile("cp.async.wait_group 0;\n" ::: "memory"); }
__device__ __forceinline__ void pdl_wait() { asm volatile("griddepcontrol.wait;" ::: "memory"); }
__device__ __forceinline__ void pdl_trigger() { asm volatile("griddepcontrol.launch_dependents;" ::: "memory"); }
__device__ __forceinline__ uint32_t cluster_rank() {
  uint32_t r;
  asm volatile("mov.u32 %0, %%cluster_ctarank;\n" : "=r"(r));
  return r;
}
__device__ __forceinline__ void cluster_arrive_release() {
  asm volatile("barrier.cluster.arrive.release.aligned;\n" ::: "memory");
}
__device__ __forceinline__ void cluster_wait_acquire() {
  asm volatile("barrier.cluster.wait.acquire.aligned;\n" ::: "memory");
}
__device__ __forceinline__ void cluster_arrive_relaxed() {
  asm volatile("barrier.cluster.arrive.relaxed.aligned;\n" ::: "memory");
}
__device__ __forceinline__ void st_peer_f32(const void* p, uint32_t rank, float v) {
  uint32_t a;
  asm volatile("mapa.shared::cluster.u32 %0, %1, %2;\n" : "=r"(a) : "r"(smem_addr(p)), "r"(rank));
  asm volatile("st.relaxed.cluster.shared::cluster.f32 [%0], %1;\n" ::"r"(a), "f"(v) : "memory");
}
__device__ __forceinline__ void mma_bf16(float (&d)[4], uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3,
                                         uint32_t b0, uint32_t b1) {
  asm volatile(
      "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, "
      "{%0, %1, %2, %3};\n"
      : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
      : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}
__device__ __forceinline__ uint8_t to_e4m3(float x) {
  uint16_t pair;
  asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;\n" : "=h"(pair) : "f"(0.f), "f"(x));
  return static_cast<uint8_t>(pair & 0xff);
}
__device__ __forceinline__ float bf16_round(float x) { return __bfloat162float(__float2bfloat16_rn(x)); }
__device__ __forceinline__ float div_full(float a, float b) {
  float r;
  asm("div.full.f32 %0, %1, %2;\n" : "=f"(r) : "f"(a), "f"(b));
  return r;
}
// act_quant_triton.py round_scale: exp2(ceil(log2(max(amax, 1e-4) * (1 / 448)))).
__device__ __forceinline__ float ue8m0_scale(float amax) {
  const float x = fmaxf(amax, 1e-4f) * (1.0f / 448.0f);
  const float e = ceilf(log2f(x));
  float scale;
  asm("ex2.approx.ftz.f32 %0, %1;\n" : "=f"(scale) : "f"(e));
  return scale;
}

struct Params {
  const __nv_bfloat16* projected;  // [rows, pitch]; q_a = columns [0, 1536)
  const __nv_bfloat16* hidden;     // [rows, hidden_stride]
  const __nv_bfloat16* w_qa;       // [1536] q_a RMSNorm weight
  const __nv_bfloat16* wq_b;       // [4096, 1536]
  const __nv_bfloat16* w_proj;     // [32, 7168]
  uint8_t* q8;                     // [rows, 32, 128] FP8 E4M3
  float* w;                        // [rows, 32]
  int64_t pitch, hidden_stride;
  int rows;
  float eps;
  const uint8_t* mq_src;  // optional: MLA query [rows, H, 576] FP8 -> sparse core's padded [rows, 16, 576]
  uint8_t* mq_dst;        // (null: off)
  int mq_heads;           // H: the local MLA heads (TP8 12, TP16 6)
  // Query-front form only (kQF): index-K / head-gate column offsets in `projected`, the write slots, the index-K pages
  // [pages, 64 * 132] and the index-K LayerNorm.
  int k_col, g_col;
  const int64_t* slots;  // [rows]; a negative slot skips the store
  uint8_t* kcache;
  const float* kgamma;  // [128] FP32
  const float* kbeta;   // [128] FP32
  float keps;
  uint8_t* qsf;  // kQ4 only: [rows, 32, 4] ue8m0 bytes of the MXFP4 query (q8 then holds the packed e2m1 [rows, 32, 64])
};

// kQ4: the ue8m0 exponent of a 32-group from its amax (idx_fp4.quantize_query: e = ceil_ue8m0(max(amax / 6, 1e-4)), clamped to [1, 254]).
__device__ __forceinline__ uint32_t q4_exp(float amax32) {
  const float sfx = fmaxf(div_full(amax32, 6.0f), 1.0e-4f);
  const uint32_t bx = __float_as_uint(sfx);
  const uint32_t ex = ((bx >> 23) & 0xffu) + ((bx & 0x7fffffu) != 0u);
  return ex < 1u ? 1u : (ex > 254u ? 254u : ex);
}

// e2m1 threshold code of x (fp4_indexer_quant._fp4_e2m1_code: strict thresholds, sign only on a nonzero code).
__device__ __forceinline__ uint32_t e2m1_code(float x) {
  const float ax = fminf(fabsf(x), 6.0f);
  uint32_t idx = (ax > 0.25f) + (ax > 0.75f) + (ax > 1.25f) + (ax > 1.75f) + (ax > 2.5f) + (ax > 3.5f) + (ax > 5.0f);
  return idx | ((x < 0.f && idx != 0u) ? 8u : 0u);
}

// One index-K row (warp-wide; lane owns columns 4 lane .. 4 lane + 3), as stock's LayerNorm + fused_store_index_k_cache
// (kKFP4: as the MXFP4 writer, header note).
template <bool kKFP4 = false>
__device__ __forceinline__ void store_index_k(const Params& p, uint2 raw, int64_t slot, int lane) {
  const __nv_bfloat16* e = reinterpret_cast<const __nv_bfloat16*>(&raw);
  float v[4], sum = 0.f;
#pragma unroll
  for (int j = 0; j < 4; ++j) {
    v[j] = __bfloat162float(e[j]);
    sum += v[j];
  }
#pragma unroll
  for (int d = 16; d; d >>= 1) sum += __shfl_xor_sync(0xffffffffu, sum, d);
  const float mean = sum * (1.0f / 128.0f);
  float var = 0.f;
#pragma unroll
  for (int j = 0; j < 4; ++j) {
    v[j] -= mean;
    var = fmaf(v[j], v[j], var);
  }
#pragma unroll
  for (int d = 16; d; d >>= 1) var += __shfl_xor_sync(0xffffffffu, var, d);
  const float rstd = rsqrtf(var * (1.0f / 128.0f) + p.keps);
  float amax = 0.f;
#pragma unroll
  for (int j = 0; j < 4; ++j) {
    const int col = lane * 4 + j;
    v[j] = bf16_round(v[j] * rstd * p.kgamma[col] + p.kbeta[col]);
    amax = fmaxf(amax, fabsf(v[j]));
  }
  if constexpr (kKFP4) {
    // the lane's 32-group = lanes 8g .. 8g + 7
#pragma unroll
    for (int d = 4; d; d >>= 1) amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, d));
    if (slot < 0) return;
    const float sf = fmaxf(div_full(amax, 6.0f), 1.0e-4f);
    const uint32_t bits = __float_as_uint(sf);
    uint32_t e = ((bits >> 23) & 0xffu) + ((bits & 0x7fffffu) != 0u);
    e = e < 1u ? 1u : (e > 254u ? 254u : e);
    const float scale = __uint_as_float(e << 23);
    uint32_t packed = 0;
#pragma unroll
    for (int j = 0; j < 4; ++j) packed |= e2m1_code(div_full(v[j], scale)) << (4 * j);
    uint8_t* page = p.kcache + (slot >> 6) * static_cast<int64_t>(68 * 64);
    const int off = static_cast<int>(slot & 63);
    reinterpret_cast<uint16_t*>(page + off * 64)[lane] = static_cast<uint16_t>(packed);
    if ((lane & 7) == 0) page[64 * 64 + off * 4 + (lane >> 3)] = static_cast<uint8_t>(e);
    return;
  }
#pragma unroll
  for (int d = 16; d; d >>= 1) amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, d));
  if (slot < 0) return;
  const float scale = __fdiv_rn(fmaxf(1e-4f, amax), 448.0f);
  const float inv = __fdiv_rn(1.0f, scale);
  uint8_t* page = p.kcache + (slot >> 6) * static_cast<int64_t>(132 * 64);
  const int off = static_cast<int>(slot & 63);
  uint32_t packed = 0;
#pragma unroll
  for (int j = 0; j < 4; ++j)
    packed |= static_cast<uint32_t>(to_e4m3(fminf(fmaxf(v[j] * inv, -448.f), 448.f))) << (8 * j);
  reinterpret_cast<uint32_t*>(page + off * 128)[lane] = packed;
  if (lane == 0) *reinterpret_cast<float*>(page + 128 * 64 + off * 4) = scale;
}

template <int kSplit, int kR = kRows>
struct Smem {
  static constexpr int kOwn = kDim / kSplit;  // wq_b rows per CTA (16 for kSplit 8)
  static constexpr int kGate = kHidden / kSplit;
  alignas(16) uint8_t wq[kOwn][kStride];
  alignas(16) uint8_t x[kR][kStride];
  alignas(16) __nv_bfloat16 wqa[kLora];
  alignas(16) __nv_bfloat16 wp[kR == kRows ? kGate : 8];  // kR 16 is kQF only: the gate weights are not read
  float part[8][kOwn][kR];  // per-warp K-slice partials
  float val[kOwn][kR];
  float red[8][kR];
  float ex_amax[kSplit][kR];
  float ex_gate[kSplit][kR];
  float scale[kR];
};

// kMqHeads: the MLA query heads of the optional side copy (TP8 12, TP16 6; compile-time so TP8 code is unchanged).
// kQF: the query-front form (header comment): gate from `projected`, index-K rows stored here.
// kKFP4: the query-front index-K store writes MXFP4 rows into [pages, 64 * 68] pages (header note).
// kQ4: the index query is stored as MXFP4 (header note).
template <int kSplit, int kMqHeads = 12, bool kQF = false, int kR = kRows, bool kKFP4 = false, bool kQ4 = false>
__global__ void __launch_bounds__(kThreads, kR == kRows ? 1 : 2) index_q_kernel(const __grid_constant__ Params p) {
  using S = Smem<kSplit, kR>;
  constexpr int kOwn = S::kOwn, kGate = S::kGate;
  constexpr int kRW = kR / 8;  // rows per warp (warp w: rows w, w + 8)
  static_assert(kOwn == 16, "one m16 tile of wq_b rows per CTA");
  static_assert(kR == kRows || (kR == 16 && kQF), "16 rows: query-front form only");
  extern __shared__ __align__(16) uint8_t smem_raw[];
  S& s = *reinterpret_cast<S*>(smem_raw);
  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5, g = lane >> 2, qd = lane & 3;
  const int head = blockIdx.x / kSplit;
  const uint32_t rank = cluster_rank();
  const int row0 = head * kDim + static_cast<int>(rank) * kOwn;  // first wq_b row of this CTA

  // Every peer has started (its shared memory exists) before anyone stores into it: arrive now, wait before the
  // first DSMEM store (CUDA PG, distributed shared memory).
  cluster_arrive_relaxed();
  // Weights only before the wait.
  for (int i = tid; i < kOwn * (kLora / 8); i += kThreads) {
    const int r = i / (kLora / 8), c = i % (kLora / 8);
    cp_async16(&s.wq[r][c * 16], p.wq_b + static_cast<int64_t>(row0 + r) * kLora + c * 8);
  }
  for (int i = tid; i < kLora / 8; i += kThreads) cp_async16(&s.wqa[i * 8], p.w_qa + i * 8);
  if constexpr (!kQF) {
    for (int i = tid; i < kGate / 8; i += kThreads)
      cp_async16(&s.wp[i * 8], p.w_proj + static_cast<int64_t>(head) * kHidden + rank * kGate + i * 8);
  }
  cp_async_commit();
  pdl_wait();
  pdl_trigger();

  // Optional side job: the sparse core's H -> 16 head-padded query copy (replaces SparseMla.prepare_query's
  // separate copy kernel); the MLA query kernel wrote mq_src before our wait. Pad heads stay zero.
  if (p.mq_src != nullptr) {
    const int chunks = p.rows * kMqHeads * 36;  // 16-B chunks
    for (int i = blockIdx.x * kThreads + tid; i < chunks; i += gridDim.x * kThreads) {
      const int r = i / (kMqHeads * 36), rem = i % (kMqHeads * 36);
      reinterpret_cast<uint4*>(p.mq_dst + static_cast<int64_t>(r) * 16 * 576)[rem] =
          reinterpret_cast<const uint4*>(p.mq_src + static_cast<int64_t>(r) * kMqHeads * 576)[rem];
    }
  }
  // Issue every activation load at once: warp w's q_a row (6 x 16 B per lane) and this CTA's hidden columns.
  constexpr int kQaChunks = kLora / 8 / 32;  // 16-B chunks per lane
  constexpr int kGateChunks = kGate / 8;     // 16-B chunks per row
  constexpr int kGatePer = (kR * kGateChunks + kThreads - 1) / kThreads;
  uint4 xa[kRW * kQaChunks];
  uint4 hv[kQF ? 1 : kGatePer];
  float gq = 0.f;          // kQF: rank 0, thread b < rows: row b's head-gate logit (BF16 value)
  uint2 kraw = {0u, 0u};   // kQF: CTA (head b, rank kSplit - 1), warp 7: lane's 4 index-K values of row b
  int64_t kslot = -1;
  const bool k_owner = kQF && rank == kSplit - 1 && warp == 7 && head < p.rows;
  {
#pragma unroll
    for (int j = 0; j < kRW; ++j) {
      const int r = warp + 8 * j;
      const uint4* src = reinterpret_cast<const uint4*>(p.projected + static_cast<int64_t>(r) * p.pitch);
#pragma unroll
      for (int i = 0; i < kQaChunks; ++i)
        xa[j * kQaChunks + i] = r < p.rows ? src[i * 32 + lane] : make_uint4(0, 0, 0, 0);
    }
    if constexpr (kQF) {
      const int gcol = p.g_col + head;
      if (rank == 0 && tid < p.rows) gq = __bfloat162float(p.projected[static_cast<int64_t>(tid) * p.pitch + gcol]);
      if (k_owner) {
        kraw = reinterpret_cast<const uint2*>(p.projected + static_cast<int64_t>(head) * p.pitch + p.k_col)[lane];
        kslot = p.slots[head];
      }
    } else {
#pragma unroll
      for (int i = 0; i < kGatePer; ++i) {
        const int c = tid + i * kThreads, b = c / kGateChunks, cc = c % kGateChunks;
        hv[i] = b < p.rows ? reinterpret_cast<const uint4*>(p.hidden + b * p.hidden_stride + rank * kGate)[cc]
                           : make_uint4(0, 0, 0, 0);
      }
    }
  }
  cp_async_wait_all();
  __syncthreads();  // w_qa, wq_b and wp staged
  // RMSNorm of q_a (rows >= p.rows are zero): warp w normalizes row w (and w + 8 at kR 16).
  float rr0 = 0.f;
#pragma unroll
  for (int jr = 0; jr < kRW; ++jr) {
    const int r = warp + 8 * jr;
    const uint4* xr = xa + jr * kQaChunks;
    float ss = 0.f;
#pragma unroll
    for (int i = 0; i < kQaChunks; ++i) {
      const __nv_bfloat16* e = reinterpret_cast<const __nv_bfloat16*>(&xr[i]);
#pragma unroll
      for (int j = 0; j < 8; ++j) ss = fmaf(__bfloat162float(e[j]), __bfloat162float(e[j]), ss);
    }
#pragma unroll
    for (int d = 16; d; d >>= 1) ss += __shfl_xor_sync(0xffffffffu, ss, d);
    const float rr = rsqrtf(ss / static_cast<float>(kLora) + p.eps);
    if (jr == 0) rr0 = rr;
#pragma unroll
    for (int i = 0; i < kQaChunks; ++i) {
      const int k0 = (i * 32 + lane) * 8;
      const __nv_bfloat16* e = reinterpret_cast<const __nv_bfloat16*>(&xr[i]);
      __align__(16) __nv_bfloat16 o[8];
#pragma unroll
      for (int j = 0; j < 8; ++j)
        o[j] = r < p.rows ? __float2bfloat16_rn(__bfloat162float(e[j]) * rr * __bfloat162float(s.wqa[k0 + j]))
                          : __float2bfloat16_rn(0.f);
      *reinterpret_cast<uint4*>(&s.x[r][k0 * 2]) = *reinterpret_cast<const uint4*>(o);
    }
  }
  // Gate partial: this CTA's kGate columns of hidden @ weights_proj[head]^T, fixed reduction order (kQF: none, the
  // input projection computed the gate).
  if constexpr (!kQF) {
    float acc[kR];
#pragma unroll
    for (int b = 0; b < kR; ++b) acc[b] = 0.f;
#pragma unroll
    for (int i = 0; i < kGatePer; ++i) {
      const int c = tid + i * kThreads, b = c / kGateChunks, cc = c % kGateChunks;
      if (b < kR) {
        const __nv_bfloat16* e = reinterpret_cast<const __nv_bfloat16*>(&hv[i]);
        float t = 0.f;
#pragma unroll
        for (int j = 0; j < 8; ++j) t = fmaf(__bfloat162float(e[j]), __bfloat162float(s.wp[cc * 8 + j]), t);
#pragma unroll
        for (int bb = 0; bb < kR; ++bb)
          if (bb == b) acc[bb] += t;
      }
    }
#pragma unroll
    for (int b = 0; b < kR; ++b) {
#pragma unroll
      for (int d = 16; d; d >>= 1) acc[b] += __shfl_xor_sync(0xffffffffu, acc[b], d);
      if (lane == 0) s.red[warp][b] = acc[b];
    }
  }
  __syncthreads();
  // wq_b slice: warp w reduces K columns [192 w, 192 w + 192) of the 16 x 8 product (16 x 16 at kR 16: two n8 tiles).
  if constexpr (kRW == 2) {
    float d0[4] = {0.f, 0.f, 0.f, 0.f}, d1[4] = {0.f, 0.f, 0.f, 0.f};
#pragma unroll
    for (int ks = 0; ks < kLora / 8 / 16; ++ks) {
      const int k0 = warp * (kLora / 8) + ks * 16;
      auto wq = [&](int r, int k) { return *reinterpret_cast<const uint32_t*>(&s.wq[r][k * 2]); };
      auto xs = [&](int r, int k) { return *reinterpret_cast<const uint32_t*>(&s.x[r][k * 2]); };
      const uint32_t a0 = wq(g, k0 + 2 * qd), a1 = wq(g + 8, k0 + 2 * qd), a2 = wq(g, k0 + 8 + 2 * qd),
                     a3 = wq(g + 8, k0 + 8 + 2 * qd);
      mma_bf16(d0, a0, a1, a2, a3, xs(g, k0 + 2 * qd), xs(g, k0 + 8 + 2 * qd));
      mma_bf16(d1, a0, a1, a2, a3, xs(g + 8, k0 + 2 * qd), xs(g + 8, k0 + 8 + 2 * qd));
    }
    s.part[warp][g][2 * qd] = d0[0];
    s.part[warp][g][2 * qd + 1] = d0[1];
    s.part[warp][g + 8][2 * qd] = d0[2];
    s.part[warp][g + 8][2 * qd + 1] = d0[3];
    s.part[warp][g][8 + 2 * qd] = d1[0];
    s.part[warp][g][8 + 2 * qd + 1] = d1[1];
    s.part[warp][g + 8][8 + 2 * qd] = d1[2];
    s.part[warp][g + 8][8 + 2 * qd + 1] = d1[3];
  } else {
    float d[4] = {0.f, 0.f, 0.f, 0.f};
#pragma unroll
    for (int ks = 0; ks < kLora / 8 / 16; ++ks) {
      const int k0 = warp * (kLora / 8) + ks * 16;
      auto wq = [&](int r, int k) { return *reinterpret_cast<const uint32_t*>(&s.wq[r][k * 2]); };
      auto xs = [&](int r, int k) { return *reinterpret_cast<const uint32_t*>(&s.x[r][k * 2]); };
      mma_bf16(d, wq(g, k0 + 2 * qd), wq(g + 8, k0 + 2 * qd), wq(g, k0 + 8 + 2 * qd), wq(g + 8, k0 + 8 + 2 * qd),
               xs(g, k0 + 2 * qd), xs(g, k0 + 8 + 2 * qd));
    }
    s.part[warp][g][2 * qd] = d[0];
    s.part[warp][g][2 * qd + 1] = d[1];
    s.part[warp][g + 8][2 * qd] = d[2];
    s.part[warp][g + 8][2 * qd + 1] = d[3];
  }
  __syncthreads();
  if (tid < kOwn * kR) {  // fixed-order sum over the 8 K slices, one BF16 rounding
    const int r = tid / kR, b = tid % kR;
    float acc = 0.f;
#pragma unroll
    for (int w8 = 0; w8 < 8; ++w8) acc += s.part[w8][r][b];
    s.val[r][b] = bf16_round(acc);
  }
  __syncthreads();
  cluster_wait_acquire();  // pairs with the arrive at kernel start: every peer is running
  if (tid < kR) {  // local amax over this CTA's 16 rows, gate partial: to the head's cluster
    const int b = tid;
    float amax = 0.f, gate = 0.f;
#pragma unroll
    for (int r = 0; r < kOwn; ++r) amax = fmaxf(amax, fabsf(s.val[r][b]));
#pragma unroll
    for (int c = 0; c < kSplit; ++c) st_peer_f32(&s.ex_amax[rank][b], c, amax);
    if constexpr (!kQF) {
#pragma unroll
      for (int w8 = 0; w8 < 8; ++w8) gate += s.red[w8][b];
      st_peer_f32(&s.ex_gate[rank][b], 0, gate);
    }
  }
  cluster_arrive_release();
  // kQF: the index-K row of this CTA's head (rows < p.rows), off the exchange: between arrive and wait.
  if constexpr (kQF) {
    if (k_owner) store_index_k<kKFP4>(p, kraw, kslot, lane);
  }
  cluster_wait_acquire();
  if (tid < kR) {
    const int b = tid;
    if constexpr (kQ4) {  // this CTA's 32-group: ranks 2 (rank / 2) and 2 (rank / 2) + 1
      const float gmax = fmaxf(s.ex_amax[rank & ~1u][b], s.ex_amax[rank | 1u][b]);
      const float gsf = fmaxf(div_full(gmax, 6.0f), 1.0e-4f);
      const uint32_t gbits = __float_as_uint(gsf);
      uint32_t ge = ((gbits >> 23) & 0xffu) + ((gbits & 0x7fffffu) != 0u);
      ge = ge < 1u ? 1u : (ge > 254u ? 254u : ge);
      s.scale[b] = __uint_as_float(ge << 23);
      // kQ4 v2: rank 0 writes the head's four group exponents as ONE 32-bit word (v1: four byte stores from four CTAs into one word)
      if (rank == 0 && b < p.rows) {
        uint32_t word = 0;
#pragma unroll
        for (int g4 = 0; g4 < 4; ++g4) word |= q4_exp(fmaxf(s.ex_amax[2 * g4][b], s.ex_amax[2 * g4 + 1][b])) << (8 * g4);
        reinterpret_cast<uint32_t*>(p.qsf)[static_cast<int64_t>(b) * kHeads + head] = word;
      }
    } else {
      float amax = 0.f;
#pragma unroll
      for (int c = 0; c < kSplit; ++c) amax = fmaxf(amax, s.ex_amax[c][b]);
      s.scale[b] = ue8m0_scale(amax);
    }
    if (rank == 0 && b < p.rows) {
      float gate = 0.f;
      if constexpr (kQF) {
        gate = gq;  // already the BF16 GEMM output: the bf16_round below is exact
      } else {
#pragma unroll
        for (int c = 0; c < kSplit; ++c) gate += s.ex_gate[c][b];
      }
      const float g2 = bf16_round(bf16_round(gate) * 0.17677669529663687f);  // 32^-0.5
      if constexpr (kQ4) {
        p.w[b * kHeads + head] = g2 * 0.08838834764831845f;  // 128^-0.5; no q scale (MXFP4 query)
      } else {
      p.w[b * kHeads + head] = g2 * s.scale[b] * 0.08838834764831845f;  // 128^-0.5
      }
    }
  }
  __syncthreads();
  if constexpr (kQ4) {  // kQ4 v2: one code per thread (the FP8 form's thread map), the odd column's code pulled from the next lane;
    // thread (b, r) with r even writes byte 8 rank + r / 2 of row b, head `head` (low nibble = even element). kOwn * kR is a multiple of 32.
    static_assert((kOwn * kR) % 32 == 0 && kOwn % 2 == 0,
                  "kQ4 v2: whole warps run the full-mask pair shuffle; every warp starts at an even element of a row");
    if (tid < kOwn * kR) {
      const int r = tid % kOwn, b = tid / kOwn;
      const uint32_t code = e2m1_code(div_full(s.val[r][b], s.scale[b]));
      const uint32_t hi = __shfl_down_sync(0xffffffffu, code, 1);
      if ((r & 1) == 0 && b < p.rows)
        p.q8[(static_cast<int64_t>(b) * kHeads + head) * (kDim / 2) + rank * (kOwn / 2) + r / 2] = static_cast<uint8_t>(code | (hi << 4));
    }
  } else if (tid < kOwn * kR) {
    const int r = tid % kOwn, b = tid / kOwn;
    if (b < p.rows) {
      const float y = fminf(fmaxf(div_full(s.val[r][b], s.scale[b]), -448.f), 448.f);
      p.q8[(static_cast<int64_t>(b) * kHeads + head) * kDim + rank * kOwn + r] = to_e4m3(y);
    }
  }
}

struct Binding {
  static void run(const tvm::ffi::TensorView projected, const tvm::ffi::TensorView hidden,
                  const tvm::ffi::TensorView w_qa, const tvm::ffi::TensorView wq_b, const tvm::ffi::TensorView w_proj,
                  const tvm::ffi::TensorView q8, const tvm::ffi::TensorView w, const double eps) {
    run_impl(projected, hidden, w_qa, wq_b, w_proj, q8, w, eps, nullptr, nullptr, 0);
  }
  static void run_copy(const tvm::ffi::TensorView projected, const tvm::ffi::TensorView hidden,
                       const tvm::ffi::TensorView w_qa, const tvm::ffi::TensorView wq_b,
                       const tvm::ffi::TensorView w_proj, const tvm::ffi::TensorView q8, const tvm::ffi::TensorView w,
                       const double eps, const tvm::ffi::TensorView mq_src, const tvm::ffi::TensorView mq_dst) {
    using namespace sglang::host;
    const int64_t heads = mq_src.stride(0) / 576;  // TP8 12, TP16 6
    RuntimeCheck((heads == 12 || heads == 6) && mq_src.stride(0) == heads * 576 && mq_src.size(0) >= projected.size(0),
                 "MLA query [rows, H, 576], H = 12 (TP8) or 6 (TP16)");
    RuntimeCheck(mq_dst.size(0) >= projected.size(0) && mq_dst.stride(0) == 16 * 576, "padded query [rows, 16, 576]");
    run_impl(projected, hidden, w_qa, wq_b, w_proj, q8, w, eps, static_cast<const uint8_t*>(mq_src.data_ptr()),
             static_cast<uint8_t*>(mq_dst.data_ptr()), static_cast<int>(heads));
  }
  static void run_impl(const tvm::ffi::TensorView projected, const tvm::ffi::TensorView hidden,
                       const tvm::ffi::TensorView w_qa, const tvm::ffi::TensorView wq_b,
                       const tvm::ffi::TensorView w_proj, const tvm::ffi::TensorView q8, const tvm::ffi::TensorView w,
                       const double eps, const uint8_t* mq_src, uint8_t* mq_dst, int mq_heads) {
    using namespace sglang::host;
    const int64_t rows = projected.size(0);
    RuntimeCheck(rows >= 1 && rows <= kRows, "index-Q kernel serves 1-8 rows");
    RuntimeCheck(projected.stride(1) == 1 && hidden.stride(1) == 1 && hidden.size(0) == rows, "row layouts");
    RuntimeCheck(projected.stride(0) % 8 == 0 && hidden.stride(0) % 8 == 0, "16-byte aligned rows");
    RuntimeCheck(wq_b.size(0) == kHeads * kDim && wq_b.size(1) == kLora && wq_b.stride(0) == kLora, "wq_b");
    RuntimeCheck(w_proj.size(0) == kHeads && w_proj.size(1) == kHidden && w_proj.stride(0) == kHidden, "w_proj");
    RuntimeCheck(w_qa.size(0) == kLora, "w_qa");
    RuntimeCheck(q8.size(0) == rows && w.size(0) == rows, "outputs");
    Params p{static_cast<const __nv_bfloat16*>(projected.data_ptr()),
             static_cast<const __nv_bfloat16*>(hidden.data_ptr()),
             static_cast<const __nv_bfloat16*>(w_qa.data_ptr()),
             static_cast<const __nv_bfloat16*>(wq_b.data_ptr()),
             static_cast<const __nv_bfloat16*>(w_proj.data_ptr()),
             static_cast<uint8_t*>(q8.data_ptr()),
             static_cast<float*>(w.data_ptr()),
             projected.stride(0),
             hidden.stride(0),
             static_cast<int>(rows),
             static_cast<float>(eps),
             mq_src,
             mq_dst,
             mq_heads};
    constexpr int kSplit = 8;
    auto kernel = mq_heads == 6 ? index_q_kernel<kSplit, 6> : index_q_kernel<kSplit, 12>;
    const size_t bytes = sizeof(Smem<kSplit>);
    CHECK_CUDA(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, static_cast<int>(bytes)));
    LaunchKernel(kHeads * kSplit, kThreads, projected.device(), bytes)
        .config({.use_pdl = true, .cluster_dim = dim3(kSplit, 1, 1)})(kernel, p);
  }
  // Query-front form: `projected` [rows, pitch] carries q_a at [0, 1536), the index-K rows at [k_col, k_col + 128) and
  // the 32 head-gate logits at [g_col, g_col + 32); the new tokens' index-K rows go to `kcache` at `slots`.
  static void run_qf(const tvm::ffi::TensorView projected, const tvm::ffi::TensorView w_qa,
                     const tvm::ffi::TensorView wq_b, const tvm::ffi::TensorView q8, const tvm::ffi::TensorView w,
                     const double eps, const int64_t k_col, const int64_t g_col, const tvm::ffi::TensorView slots,
                     const tvm::ffi::TensorView kcache, const tvm::ffi::TensorView kgamma,
                     const tvm::ffi::TensorView kbeta, const double keps) {
    using namespace sglang::host;
    const int64_t rows = projected.size(0);
    RuntimeCheck(rows >= 1 && rows <= 2 * kRows, "query-front index-Q kernel serves 1-16 rows");
    RuntimeCheck(projected.stride(1) == 1 && projected.stride(0) % 8 == 0, "16-byte aligned projected rows");
    RuntimeCheck(k_col >= kLora && k_col % 4 == 0 && k_col + kDim <= projected.size(1), "index-K columns");
    RuntimeCheck(g_col >= kLora && g_col + kHeads <= projected.size(1), "head-gate columns");
    RuntimeCheck(wq_b.size(0) == kHeads * kDim && wq_b.size(1) == kLora && wq_b.stride(0) == kLora, "wq_b");
    RuntimeCheck(w_qa.size(0) == kLora, "w_qa");
    RuntimeCheck(q8.size(0) == rows && w.size(0) == rows, "outputs");
    RuntimeCheck(slots.size(0) >= rows && slots.stride(0) == 1 && slots.dtype().bits == 64, "int64 write slots");
    // [pages, 64 * 132] FP8 pages, or the MXFP4 pages [pages, 64 * 68] of K31MK_DSA_IDX_FP4_FP8Q
    const bool kfp4 = kcache.ndim() == 2 && kcache.size(1) == 64 * 68 && kcache.stride(0) == 64 * 68 && kcache.stride(1) == 1;
    RuntimeCheck(kfp4 || (kcache.ndim() == 2 && kcache.size(1) == 64 * 132 && kcache.stride(0) == 64 * 132 && kcache.stride(1) == 1),
                 "index-K pages [pages, 64 * 132] (or MXFP4 [pages, 64 * 68])");
    RuntimeCheck(kgamma.size(0) == kDim && kbeta.size(0) == kDim && kgamma.dtype().bits == 32 && kbeta.dtype().bits == 32,
                 "FP32 index-K LayerNorm");
    Params p{};
    p.projected = static_cast<const __nv_bfloat16*>(projected.data_ptr());
    p.w_qa = static_cast<const __nv_bfloat16*>(w_qa.data_ptr());
    p.wq_b = static_cast<const __nv_bfloat16*>(wq_b.data_ptr());
    p.q8 = static_cast<uint8_t*>(q8.data_ptr());
    p.w = static_cast<float*>(w.data_ptr());
    p.pitch = projected.stride(0);
    p.rows = static_cast<int>(rows);
    p.eps = static_cast<float>(eps);
    p.k_col = static_cast<int>(k_col);
    p.g_col = static_cast<int>(g_col);
    p.slots = static_cast<const int64_t*>(slots.data_ptr());
    p.kcache = static_cast<uint8_t*>(kcache.data_ptr());
    p.kgamma = static_cast<const float*>(kgamma.data_ptr());
    p.kbeta = static_cast<const float*>(kbeta.data_ptr());
    p.keps = static_cast<float>(keps);
    constexpr int kSplit = 8;
    if (rows <= kRows) {  // the served 8-row instance, unchanged (kfp4: its MXFP4-store twin)
      auto kernel = kfp4 ? index_q_kernel<kSplit, 12, true, kRows, true> : index_q_kernel<kSplit, 12, true>;
      const size_t bytes = sizeof(Smem<kSplit>);
      CHECK_CUDA(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, static_cast<int>(bytes)));
      LaunchKernel(kHeads * kSplit, kThreads, projected.device(), bytes)
          .config({.use_pdl = true, .cluster_dim = dim3(kSplit, 1, 1)})(kernel, p);
      return;
    }
    // QF16: 9-16 rows. Two CTAs per SM need the full shared-memory carveout (Smem<8, 16> ~112 KB each).
    auto kernel = kfp4 ? index_q_kernel<kSplit, 12, true, 2 * kRows, true> : index_q_kernel<kSplit, 12, true, 2 * kRows>;
    const size_t bytes = sizeof(Smem<kSplit, 2 * kRows>);
    CHECK_CUDA(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, static_cast<int>(bytes)));
    CHECK_CUDA(cudaFuncSetAttribute(kernel, cudaFuncAttributePreferredSharedMemoryCarveout, 100));
    LaunchKernel(kHeads * kSplit, kThreads, projected.device(), bytes)
        .config({.use_pdl = true, .cluster_dim = dim3(kSplit, 1, 1)})(kernel, p);
  }

  // ---- MXFP4 index query (kQ4;, K31MK_DSA_IDX_Q=fp4): q4 int8 [rows, 32, 64] packed e2m1, qsf int32 [rows, 32] (four
  // ue8m0 bytes), w FP32 [rows, 32] without the q scale. Same grids, shared memory and PDL / cluster launch as the FP8 entries.
  static void check_q4(const tvm::ffi::TensorView q4, const tvm::ffi::TensorView qsf, const tvm::ffi::TensorView w, const int64_t rows) {
    using namespace sglang::host;
    RuntimeCheck(q4.ndim() == 3 && q4.size(0) == rows && q4.size(1) == kHeads && q4.size(2) == kDim / 2 && q4.stride(0) == kHeads * kDim / 2 &&
                     q4.stride(1) == kDim / 2 && q4.stride(2) == 1 && q4.dtype().bits == 8,
                 "MXFP4 query int8 [rows, 32, 64] contiguous");
    RuntimeCheck(qsf.ndim() == 2 && qsf.size(0) == rows && qsf.size(1) == kHeads && qsf.stride(0) == kHeads && qsf.stride(1) == 1 &&
                     qsf.dtype().bits == 32,
                 "MXFP4 query scales int32 [rows, 32] contiguous");
    RuntimeCheck(reinterpret_cast<uintptr_t>(qsf.data_ptr()) % 4 == 0, "MXFP4 query scales 4-byte aligned (one 32-bit word store per row and head)");
    RuntimeCheck(w.size(0) == rows, "outputs");
  }
  static void run_q4(const tvm::ffi::TensorView projected, const tvm::ffi::TensorView hidden, const tvm::ffi::TensorView w_qa,
                     const tvm::ffi::TensorView wq_b, const tvm::ffi::TensorView w_proj, const tvm::ffi::TensorView q4,
                     const tvm::ffi::TensorView qsf, const tvm::ffi::TensorView w, const double eps) {
    run_q4_impl(projected, hidden, w_qa, wq_b, w_proj, q4, qsf, w, eps, nullptr, nullptr, 0);
  }
  static void run_copy_q4(const tvm::ffi::TensorView projected, const tvm::ffi::TensorView hidden, const tvm::ffi::TensorView w_qa,
                          const tvm::ffi::TensorView wq_b, const tvm::ffi::TensorView w_proj, const tvm::ffi::TensorView q4,
                          const tvm::ffi::TensorView qsf, const tvm::ffi::TensorView w, const double eps,
                          const tvm::ffi::TensorView mq_src, const tvm::ffi::TensorView mq_dst) {
    using namespace sglang::host;
    const int64_t heads = mq_src.stride(0) / 576;  // TP8 12, TP16 6
    RuntimeCheck((heads == 12 || heads == 6) && mq_src.stride(0) == heads * 576 && mq_src.size(0) >= projected.size(0),
                 "MLA query [rows, H, 576], H = 12 (TP8) or 6 (TP16)");
    RuntimeCheck(mq_dst.size(0) >= projected.size(0) && mq_dst.stride(0) == 16 * 576, "padded query [rows, 16, 576]");
    run_q4_impl(projected, hidden, w_qa, wq_b, w_proj, q4, qsf, w, eps, static_cast<const uint8_t*>(mq_src.data_ptr()),
                static_cast<uint8_t*>(mq_dst.data_ptr()), static_cast<int>(heads));
  }
  static void run_q4_impl(const tvm::ffi::TensorView projected, const tvm::ffi::TensorView hidden, const tvm::ffi::TensorView w_qa,
                          const tvm::ffi::TensorView wq_b, const tvm::ffi::TensorView w_proj, const tvm::ffi::TensorView q4,
                          const tvm::ffi::TensorView qsf, const tvm::ffi::TensorView w, const double eps, const uint8_t* mq_src,
                          uint8_t* mq_dst, int mq_heads) {
    using namespace sglang::host;
    const int64_t rows = projected.size(0);
    RuntimeCheck(rows >= 1 && rows <= kRows, "index-Q kernel serves 1-8 rows");
    RuntimeCheck(projected.stride(1) == 1 && hidden.stride(1) == 1 && hidden.size(0) == rows, "row layouts");
    RuntimeCheck(projected.stride(0) % 8 == 0 && hidden.stride(0) % 8 == 0, "16-byte aligned rows");
    RuntimeCheck(wq_b.size(0) == kHeads * kDim && wq_b.size(1) == kLora && wq_b.stride(0) == kLora, "wq_b");
    RuntimeCheck(w_proj.size(0) == kHeads && w_proj.size(1) == kHidden && w_proj.stride(0) == kHidden, "w_proj");
    RuntimeCheck(w_qa.size(0) == kLora, "w_qa");
    check_q4(q4, qsf, w, rows);
    Params p{};
    p.projected = static_cast<const __nv_bfloat16*>(projected.data_ptr());
    p.hidden = static_cast<const __nv_bfloat16*>(hidden.data_ptr());
    p.w_qa = static_cast<const __nv_bfloat16*>(w_qa.data_ptr());
    p.wq_b = static_cast<const __nv_bfloat16*>(wq_b.data_ptr());
    p.w_proj = static_cast<const __nv_bfloat16*>(w_proj.data_ptr());
    p.q8 = static_cast<uint8_t*>(q4.data_ptr());
    p.w = static_cast<float*>(w.data_ptr());
    p.pitch = projected.stride(0);
    p.hidden_stride = hidden.stride(0);
    p.rows = static_cast<int>(rows);
    p.eps = static_cast<float>(eps);
    p.mq_src = mq_src;
    p.mq_dst = mq_dst;
    p.mq_heads = mq_heads;
    p.qsf = static_cast<uint8_t*>(qsf.data_ptr());
    constexpr int kSplit = 8;
    auto kernel = mq_heads == 6 ? index_q_kernel<kSplit, 6, false, kRows, false, true> : index_q_kernel<kSplit, 12, false, kRows, false, true>;
    const size_t bytes = sizeof(Smem<kSplit>);
    CHECK_CUDA(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, static_cast<int>(bytes)));
    LaunchKernel(kHeads * kSplit, kThreads, projected.device(), bytes)
        .config({.use_pdl = true, .cluster_dim = dim3(kSplit, 1, 1)})(kernel, p);
  }
  // Query-front form with the MXFP4 query: kcache must be the MXFP4 index-K pages [pages, 64 * 68] (the FP4 query pairs only with them).
  static void run_qf_q4(const tvm::ffi::TensorView projected, const tvm::ffi::TensorView w_qa, const tvm::ffi::TensorView wq_b,
                        const tvm::ffi::TensorView q4, const tvm::ffi::TensorView qsf, const tvm::ffi::TensorView w, const double eps,
                        const int64_t k_col, const int64_t g_col, const tvm::ffi::TensorView slots, const tvm::ffi::TensorView kcache,
                        const tvm::ffi::TensorView kgamma, const tvm::ffi::TensorView kbeta, const double keps) {
    using namespace sglang::host;
    const int64_t rows = projected.size(0);
    RuntimeCheck(rows >= 1 && rows <= 2 * kRows, "query-front index-Q kernel serves 1-16 rows");
    RuntimeCheck(projected.stride(1) == 1 && projected.stride(0) % 8 == 0, "16-byte aligned projected rows");
    RuntimeCheck(k_col >= kLora && k_col % 4 == 0 && k_col + kDim <= projected.size(1), "index-K columns");
    RuntimeCheck(g_col >= kLora && g_col + kHeads <= projected.size(1), "head-gate columns");
    RuntimeCheck(wq_b.size(0) == kHeads * kDim && wq_b.size(1) == kLora && wq_b.stride(0) == kLora, "wq_b");
    RuntimeCheck(w_qa.size(0) == kLora, "w_qa");
    check_q4(q4, qsf, w, rows);
    RuntimeCheck(slots.size(0) >= rows && slots.stride(0) == 1 && slots.dtype().bits == 64, "int64 write slots");
    RuntimeCheck(kcache.ndim() == 2 && kcache.size(1) == 64 * 68 && kcache.stride(0) == 64 * 68 && kcache.stride(1) == 1,
                 "MXFP4 query: index-K pages must be MXFP4 [pages, 64 * 68]");
    RuntimeCheck(kgamma.size(0) == kDim && kbeta.size(0) == kDim && kgamma.dtype().bits == 32 && kbeta.dtype().bits == 32,
                 "FP32 index-K LayerNorm");
    Params p{};
    p.projected = static_cast<const __nv_bfloat16*>(projected.data_ptr());
    p.w_qa = static_cast<const __nv_bfloat16*>(w_qa.data_ptr());
    p.wq_b = static_cast<const __nv_bfloat16*>(wq_b.data_ptr());
    p.q8 = static_cast<uint8_t*>(q4.data_ptr());
    p.w = static_cast<float*>(w.data_ptr());
    p.pitch = projected.stride(0);
    p.rows = static_cast<int>(rows);
    p.eps = static_cast<float>(eps);
    p.k_col = static_cast<int>(k_col);
    p.g_col = static_cast<int>(g_col);
    p.slots = static_cast<const int64_t*>(slots.data_ptr());
    p.kcache = static_cast<uint8_t*>(kcache.data_ptr());
    p.kgamma = static_cast<const float*>(kgamma.data_ptr());
    p.kbeta = static_cast<const float*>(kbeta.data_ptr());
    p.keps = static_cast<float>(keps);
    p.qsf = static_cast<uint8_t*>(qsf.data_ptr());
    constexpr int kSplit = 8;
    if (rows <= kRows) {
      auto kernel = index_q_kernel<kSplit, 12, true, kRows, true, true>;
      const size_t bytes = sizeof(Smem<kSplit>);
      CHECK_CUDA(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, static_cast<int>(bytes)));
      LaunchKernel(kHeads * kSplit, kThreads, projected.device(), bytes)
          .config({.use_pdl = true, .cluster_dim = dim3(kSplit, 1, 1)})(kernel, p);
      return;
    }
    auto kernel = index_q_kernel<kSplit, 12, true, 2 * kRows, true, true>;
    const size_t bytes = sizeof(Smem<kSplit, 2 * kRows>);
    CHECK_CUDA(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, static_cast<int>(bytes)));
    CHECK_CUDA(cudaFuncSetAttribute(kernel, cudaFuncAttributePreferredSharedMemoryCarveout, 100));
    LaunchKernel(kHeads * kSplit, kThreads, projected.device(), bytes)
        .config({.use_pdl = true, .cluster_dim = dim3(kSplit, 1, 1)})(kernel, p);
  }
};

}  // namespace k31_index_q
