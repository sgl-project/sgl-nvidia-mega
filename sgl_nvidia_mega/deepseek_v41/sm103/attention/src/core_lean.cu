// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//
// DeepSeek-V4.1 decode attention core at 1-8 rows and attention TP8's 8 local heads (slot
// attn_core_lean): one launch per layer, grid (splits, rows), one cluster of `splits` CTAs
// per row, 64 keys per CTA (10 splits: 128 window + 512 compressed keys; 2 splits: the
// 128-key window of a window-only layer), the merge with the attention sink and the
// output's inverse RoPE inside the cluster, BF16 output [rows, 8 * 512].
//
// Derived from muLLM v0.4.1 (NVIDIA Corporation, Apache-2.0): its batched DeepSeek-V4.1
// sparse attention kernel (BatchedAttention and its lean levers; the lean gathered
// sparse entry with its shared-memory pad, and the lean window entry, as its batch-1
// world-8 decode serves them) and its window-KV format device primitives (below), in
// its attention-TP8 build. This file has been modified from those; the arithmetic, the
// key chunking, the summation and merge orders are the original's. The modifications
// are interface adaptations to SGLang's buffers:
//   * keys are addressed through SGLang's per-row token indices into its paged KV pool
//     (SGLANG_DSV4_KV_LAYOUT=v41_new: 528-byte window rows and 288-byte FP4 compressed
//     rows, the bytes of muLLM's ring entries and compressed entries), in place of
//     muLLM's slot ring and block tables; window key i is the row's window index i
//     (oldest first), compressed key j its sparse index j. So the ring addressing
//     (RingEntry, kLeanGather's arithmetic decode-lane path), the block-table staging
//     (kLeanTable, kLeanShift) and muLLM's step staging and ring commit are gone: the
//     core reads the keys SGLang's store kernel wrote;
//   * the row's key counts are SGLang's window and sparse lengths, not its position;
//     SGLang pads its index rows with -1, so a key whose index is negative is masked
//     like a key past the count (a per-key flag in shared memory), and a split whose
//     keys are all masked takes 0 probabilities rather than exp(-inf - -inf);
//   * the query's RoPE (cos, sin) pairs come from SGLang's freqs_cis row of the
//     position ([S, 64] fp32, cos and sin interleaved) instead of muLLM's rotation table;
//   * the chain-stamp macros, the floor (bounded replay) and wide entries, the default
//     (non-lean) key path and the window key, commit and insert kernels are dropped.
// Host side: ../core_lean.py.
//
// Build switch: CORE_POS_INT64 (0/1: positions int64 or int32).

#if !defined(CORE_POS_INT64)
#error "core_lean.cu: the launcher passes CORE_POS_INT64"
#endif

typedef unsigned char uint8_t;
typedef unsigned short uint16_t;
typedef unsigned int uint32_t;
typedef signed int int32_t;
#if defined(__CUDACC_RTC__)
typedef unsigned long long uint64_t;
typedef signed long long int64_t;
#else
typedef unsigned long uint64_t;
typedef signed long int64_t;
#endif

#include <cooperative_groups.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>

#if CORE_POS_INT64
typedef int64_t position_t;
#else
typedef int32_t position_t;
#endif

#define DSV41_ATTN_WORLD 8
#ifndef INFINITY
#define INFINITY __int_as_float(0x7f800000)
#endif

// ---------------------------------------------------------------- window-KV formats

namespace d41attn {

constexpr int kHeadDim = 512;
constexpr int kRopeDim = 64;
constexpr int kRopePairs = 32;
constexpr int kWindow = 128;
constexpr int kWindowEntryBytes = 528;
constexpr int kCompressedEntryBytes = 288;
constexpr int kTopK = 512;

__device__ __forceinline__ void GridDependencyWait() {
  asm volatile("griddepcontrol.wait;" ::: "memory");
}

__device__ __forceinline__ void LaunchDependents() {
  asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
}

__device__ __forceinline__ void ClusterSyncShared() {
  asm volatile(
      "fence.release.sync_restrict::shared::cta.cluster;\n\t"
      "barrier.cluster.arrive.relaxed.aligned;\n\t"
      "barrier.cluster.wait.aligned;\n\t"
      "fence.acquire.sync_restrict::shared::cluster.cluster;" ::: "memory");
}

__device__ __forceinline__ void ClusterSyncRelaxed() {
  asm volatile("barrier.cluster.arrive.relaxed.aligned;\n\tbarrier.cluster.wait.aligned;"
               ::: "memory");
}

__device__ __forceinline__ float RoundBf16(float value) {
  return __bfloat162float(__float2bfloat16_rn(value));
}

__device__ __forceinline__ float WarpSum(float value) {
#pragma unroll
  for (int offset = 16; offset > 0; offset >>= 1) {
    value += __shfl_xor_sync(0xffffffffU, value, offset);
  }
  return value;
}

__device__ __forceinline__ float WarpMax(float value) {
#pragma unroll
  for (int offset = 16; offset > 0; offset >>= 1) {
    value = fmaxf(value, __shfl_xor_sync(0xffffffffU, value, offset));
  }
  return value;
}

__device__ __forceinline__ void Rotate(float x0, float x1, float c, float s, bool inverse,
                                       float& y0, float& y1) {
  const float sine = inverse ? -s : s;
  y0 = __fmaf_rn(x0, c, -__fmul_rn(x1, sine));
  y1 = __fmaf_rn(x1, c, __fmul_rn(x0, sine));
}

__device__ __forceinline__ float Pow2(int exponent) {
  return __uint_as_float(static_cast<uint32_t>(exponent + 127) << 23);
}

__device__ __forceinline__ void MmaBf16(float (&d)[4], uint32_t a0, uint32_t a1,
                                        uint32_t a2, uint32_t a3, uint32_t b0,
                                        uint32_t b1) {
  asm volatile(
      "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
      "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};"
      : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
      : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}

__device__ __forceinline__ uint32_t PackBf16(float low, float high) {
  const __nv_bfloat162 pair = __floats2bfloat162_rn(low, high);
  return *reinterpret_cast<const uint32_t*>(&pair);
}

}  // namespace d41attn

// ---------------------------------------------------------------- batched attention

namespace cg = cooperative_groups;

namespace {

constexpr int kThreads = 512;
constexpr int kWarps = kThreads / 32;
constexpr int kHeads = 16;
constexpr int kLocalHeads = 64 / DSV41_ATTN_WORLD;
static_assert(kHeads % kLocalHeads == 0, "the local heads tile the 16 MMA rows");
constexpr int kDim = d41attn::kHeadDim;
constexpr int kFeaturesPerWarp = kDim / kWarps;
constexpr int kOutputTiles = kFeaturesPerWarp / 8;
constexpr int kKeys = 64;
constexpr int kCompressedSplits = 10;
constexpr int kWindowSplits = 2;
constexpr int kStride = kDim + 8;
constexpr int kScoreStride = kKeys + 4;
constexpr int kProbStride = kKeys + 8;

// muLLM's lean levers. kLeanShift and kLeanTable addressed muLLM's block tables, which
// SGLang's per-row token indices replace, and kLeanRingLate placed muLLM's ring commit,
// which is not carried (file comment); the others keep their meaning.
constexpr int kLeanMerge = 2;
constexpr int kLeanKeys = 4;
constexpr int kLeanSwap = 16;
constexpr int kLeanMergeStage = 64;
constexpr int kLeanGather = 256;
// The lean gathered sparse entry's levers (kLean4 | kLeanGather) and the lean window
// entry's (kLeanWindow), less the ones named above as gone.
constexpr int kLean4Gather = kLeanMerge | kLeanKeys | kLeanSwap | kLeanMergeStage | kLeanGather;
constexpr int kLeanWindow = kLeanKeys | kLeanSwap | kLeanGather;

template <int kOStride>
struct SharedStorageOf {
  __nv_bfloat16 q[kHeads][kStride];
  union {
    __nv_bfloat16 k[kKeys][kStride];
    float o[kHeads][kOStride];
  } kv;
  float s[kHeads][kScoreStride];
  __nv_bfloat16 p[kHeads][kProbStride];
  float stats[kHeads][2];
  float sink[kHeads];
  float rotation[2 * d41attn::kRopePairs];
};
using SharedStorage = SharedStorageOf<kDim>;
static_assert(sizeof(SharedStorage) == 90304, "core_lean.py restates the shared-memory size");
static_assert(sizeof(SharedStorageOf<kDim + 8>) == sizeof(SharedStorage), "padded O tile");

__device__ __forceinline__ uint32_t SharedAddress(const void* pointer) {
  return static_cast<uint32_t>(__cvta_generic_to_shared(pointer));
}

__device__ __forceinline__ void LdMatrixX4(uint32_t (&r)[4], const void* pointer) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];"
               : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3])
               : "r"(SharedAddress(pointer)));
}

__device__ __forceinline__ void LdMatrixX2(uint32_t (&r)[2], const void* pointer) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0, %1}, [%2];"
               : "=r"(r[0]), "=r"(r[1])
               : "r"(SharedAddress(pointer)));
}

__device__ __forceinline__ void LdMatrixX4Trans(uint32_t (&r)[4], const void* pointer) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0, %1, %2, %3}, [%4];"
               : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3])
               : "r"(SharedAddress(pointer)));
}

struct RawChunk {
  uint2 bytes;
  uint32_t scale;
};

__device__ __forceinline__ RawChunk LoadWindowChunk(const uint8_t* entry, int chunk) {
  return {*reinterpret_cast<const uint2*>(entry + chunk * 8),
          static_cast<uint32_t>(entry[kDim + chunk / 4])};
}

__device__ __forceinline__ RawChunk LoadCompressedChunk(const uint8_t* entry, int chunk) {
  return {make_uint2(*reinterpret_cast<const uint32_t*>(entry + chunk * 4), 0U),
          static_cast<uint32_t>(entry[kDim / 2 + chunk / 2])};
}

__device__ __forceinline__ uint4 ScaledE4m3Chunk(uint32_t low, uint32_t high, float scale) {
  uint32_t words[4];
  const uint32_t parts[2] = {low, high};
#pragma unroll
  for (int index = 0; index < 4; ++index) {
    const uint32_t pair = (parts[index / 2] >> (16 * (index % 2))) & 0xFFFFU;
    const __half2_raw half =
        __nv_cvt_fp8x2_to_halfraw2(static_cast<__nv_fp8x2_storage_t>(pair), __NV_E4M3);
    const float2 values = __half22float2(__half2(half));
    words[index] = d41attn::PackBf16(values.x * scale, values.y * scale);
  }
  return make_uint4(words[0], words[1], words[2], words[3]);
}

__device__ __forceinline__ uint4 WindowChunk(const RawChunk& raw) {
  return ScaledE4m3Chunk(raw.bytes.x, raw.bytes.y,
                         d41attn::Pow2(static_cast<int>(raw.scale) - 127));
}

__device__ __forceinline__ uint4 CompressedChunkLean(const RawChunk& raw) {
  const __half scale_half(__nv_cvt_fp8_to_halfraw(static_cast<__nv_fp8_storage_t>(raw.scale), __NV_E4M3));
  const __half2 scale = __half2half2(scale_half);
  uint32_t halves[4];
  asm("{\n\t.reg .b8 b0, b1, b2, b3;\n\tmov.b32 {b0, b1, b2, b3}, %4;\n\t"
      "cvt.rn.f16x2.e2m1x2 %0, b0;\n\tcvt.rn.f16x2.e2m1x2 %1, b1;\n\t"
      "cvt.rn.f16x2.e2m1x2 %2, b2;\n\tcvt.rn.f16x2.e2m1x2 %3, b3;\n\t}"
      : "=r"(halves[0]), "=r"(halves[1]), "=r"(halves[2]), "=r"(halves[3])
      : "r"(raw.bytes.x));
  uint32_t words[4];
#pragma unroll
  for (int index = 0; index < 4; ++index) {
    const float2 values =
        __half22float2(__hmul2(*reinterpret_cast<const __half2*>(&halves[index]), scale));
    words[index] = d41attn::PackBf16(values.x, values.y);
  }
  return make_uint4(words[0], words[1], words[2], words[3]);
}

#if defined(__CUDACC_VER_MAJOR__) &&                                                  \
    (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2)) && \
    (defined(__CUDA_ARCH_FEAT_SM100_ALL) || defined(__CUDA_ARCH_FEAT_SM103_ALL) ||    \
     defined(__CUDA_ARCH_FEAT_SM107_ALL) || defined(__CUDA_ARCH_FEAT_SM120_ALL) ||    \
     defined(__CUDA_ARCH_FEAT_SM121_ALL))
#define CORE_LEAN_GATHER_BF16_CVT 1
#else
#define CORE_LEAN_GATHER_BF16_CVT 0
#endif

#if CORE_LEAN_GATHER_BF16_CVT
template <int kHalf>
__device__ __forceinline__ uint32_t E4m3PairBf16(uint32_t word) {
  uint32_t pair;
  if constexpr (kHalf == 0) {
    asm("{\n\t.reg .b16 lo, hi;\n\tmov.b32 {lo, hi}, %1;\n\tcvt.rn.bf16x2.e4m3x2 %0, lo;\n\t}"
        : "=r"(pair)
        : "r"(word));
  } else {
    asm("{\n\t.reg .b16 lo, hi;\n\tmov.b32 {lo, hi}, %1;\n\tcvt.rn.bf16x2.e4m3x2 %0, hi;\n\t}"
        : "=r"(pair)
        : "r"(word));
  }
  return pair;
}

__device__ __forceinline__ uint32_t MulBf16x2(uint32_t a, uint32_t b) {
  uint32_t product;
  asm("mul.rn.bf16x2 %0, %1, %2;" : "=r"(product) : "r"(a), "r"(b));
  return product;
}
#endif

__device__ __forceinline__ uint4 WindowChunkBf16(const RawChunk& raw) {
#if CORE_LEAN_GATHER_BF16_CVT
  const uint32_t scale = raw.scale * 0x00800080U;  // (s << 7) in both halves
  return make_uint4(MulBf16x2(E4m3PairBf16<0>(raw.bytes.x), scale),
                    MulBf16x2(E4m3PairBf16<1>(raw.bytes.x), scale),
                    MulBf16x2(E4m3PairBf16<0>(raw.bytes.y), scale),
                    MulBf16x2(E4m3PairBf16<1>(raw.bytes.y), scale));
#else
  return WindowChunk(raw);
#endif
}

__device__ __forceinline__ uint4 CompressedChunkBf16(const RawChunk& raw) {
#if CORE_LEAN_GATHER_BF16_CVT
  const uint32_t scale = E4m3PairBf16<0>(raw.scale * 0x0101U);
  uint32_t pairs[4];
  asm("{\n\t.reg .b8 b0, b1, b2, b3;\n\tmov.b32 {b0, b1, b2, b3}, %4;\n\t"
      "cvt.rn.bf16x2.e2m1x2 %0, b0;\n\tcvt.rn.bf16x2.e2m1x2 %1, b1;\n\t"
      "cvt.rn.bf16x2.e2m1x2 %2, b2;\n\tcvt.rn.bf16x2.e2m1x2 %3, b3;\n\t}"
      : "=r"(pairs[0]), "=r"(pairs[1]), "=r"(pairs[2]), "=r"(pairs[3])
      : "r"(raw.bytes.x));
  return make_uint4(MulBf16x2(pairs[0], scale), MulBf16x2(pairs[1], scale),
                    MulBf16x2(pairs[2], scale), MulBf16x2(pairs[3], scale));
#else
  return CompressedChunkLean(raw);
#endif
}

__device__ __forceinline__ void ClusterArriveAfter(float a, float b, float c) {
  asm volatile("barrier.cluster.arrive.relaxed.aligned;" ::"f"(a), "f"(b), "f"(c) : "memory");
}

__device__ __forceinline__ void ClusterArrive() {
  asm volatile("barrier.cluster.arrive.relaxed.aligned;" ::: "memory");
}

__device__ __forceinline__ void ClusterWait() {
  asm volatile("barrier.cluster.wait.aligned;" ::: "memory");
}

// Adaptation: one KV row of SGLang's paged pool, token `index` of a pool whose pages
// hold 2^page_shift tokens of `row_bytes` each, `page_stride` bytes apart.
__device__ __forceinline__ const uint8_t* PoolRow(const uint8_t* pool, int64_t page_stride,
                                                  int page_shift, int row_bytes, int index) {
  return pool + static_cast<int64_t>(index >> page_shift) * page_stride +
         static_cast<int64_t>(index & ((1 << page_shift) - 1)) * row_bytes;
}

// The kernel's parameters (adapted: SGLang's pool, indices and lengths in place of
// muLLM's step descriptor, control words, ring and block tables).
struct Pool {
  const uint8_t* rows;
  int64_t page_stride;
  int page_shift;
};

// kExtra: the layer reads compressed keys (muLLM's ratio > 0). kEarlyIndex (adapted): the
// row's window indices and window length are host-written before the replay (the
// caller guarantees it, as for swap-AB's pre-wait window gather), so they are read
// before the grid-dependency wait, as muLLM reads its descriptor and block table there.
//
// kWindowKeys (adapted): the most window keys of a row, muLLM's kWindow (128) for the
// target; the DSpark draft's window-only attention reads up to 192 (three splits of the
// window entry; muLLM serves its draft with its own DSpark attention, not ported).
template <int kSplits, bool kExtra, bool kEarlyIndex, int kLean, int kWindowKeys = d41attn::kWindow>
__device__ __forceinline__ void BatchedAttention(
    const __nv_bfloat16* q, int q_stride, const float* freqs, const position_t* positions,
    Pool window_pool, const int32_t* window_indices, int window_index_stride,
    const int32_t* window_lengths, Pool extra_pool, const int32_t* extra_indices,
    int extra_index_stride, int extra_width, const int32_t* extra_lengths,
    const float* __restrict__ sink, __nv_bfloat16* __restrict__ output, float softmax_scale) {
  constexpr bool kSwap = (kLean & kLeanSwap) != 0 && kLocalHeads <= 8;
  constexpr bool kMergeStage = (kLean & kLeanMergeStage) != 0;
  static_assert(!kMergeStage || (kLean & kLeanMerge) != 0,
                "kLeanMergeStage reads kLeanMerge's padded O tile");
  constexpr bool kGather = (kLean & kLeanGather) != 0;
  static_assert(!kGather || (kLean & kLeanKeys) != 0, "kLeanGather needs kLeanKeys");
  using Storage = SharedStorageOf<(kLean & kLeanMerge) != 0 ? kDim + 8 : kDim>;
  extern __shared__ __align__(16) uint8_t shared_bytes[];
  Storage& smem = *reinterpret_cast<Storage*>(shared_bytes);
  // Adaptation: a flag per key of the split, 0 where SGLang's index is negative.
  uint8_t* key_valid = shared_bytes + sizeof(Storage);
  cg::cluster_group cluster = cg::this_cluster();
  const int split = static_cast<int>(blockIdx.x);
  const int row = static_cast<int>(blockIdx.y);
  const int warp = static_cast<int>(threadIdx.x) >> 5;
  const int lane = static_cast<int>(threadIdx.x) & 31;
  if (threadIdx.x < kHeads) smem.sink[threadIdx.x] = sink[threadIdx.x % kLocalHeads];
  const int begin = split * kKeys;
  constexpr int kChunks = kDim / 8;
  constexpr int kKeyStep = kThreads / kChunks;
  constexpr int kKeysPerThread = kKeys / kKeyStep;
  const int chunk = static_cast<int>(threadIdx.x) % kChunks;
  const int key_base = static_cast<int>(threadIdx.x) / kChunks;
  d41attn::LaunchDependents();
  // The row's position (written before the replay, as the fused wo_a reads it) and, with
  // kEarlyIndex, its window length and the window indices of this thread's keys.
  const int position = static_cast<int>(positions[row]);
  const int32_t* window_row = window_indices + static_cast<int64_t>(row) * window_index_stride;
  int window_early = 0;
  int window_slots[kKeysPerThread];
  if constexpr (kEarlyIndex) {
    window_early = min(kWindowKeys, window_lengths[row]);
#pragma unroll
    for (int i = 0; i < kKeysPerThread; ++i) {
      const int index = begin + key_base + i * kKeyStep;
      window_slots[i] = index < window_early ? window_row[index] : -1;
    }
  }
  d41attn::GridDependencyWait();
  // As in the batch-1 kernel, the top-k indices head the longest dependent chain
  // (index, entry): they are requested first, assuming a full window.
  int entries[kKeysPerThread];
  const int32_t* extra_row = extra_indices + static_cast<int64_t>(row) * extra_index_stride;
#pragma unroll
  for (int i = 0; i < kKeysPerThread; ++i) {
    const int index = begin + key_base + i * kKeyStep;
    entries[i] = kExtra && index >= kWindowKeys && index - kWindowKeys < extra_width
                     ? extra_row[index - kWindowKeys]
                     : -1;
  }
  const int window = kEarlyIndex ? window_early : min(kWindowKeys, window_lengths[row]);
  const int compressed_count = kExtra ? min(min(d41attn::kTopK, extra_width), extra_lengths[row]) : 0;
  const int count = min(kKeys, window + compressed_count - begin);
  if (window != kWindowKeys) {
#pragma unroll
    for (int i = 0; i < kKeysPerThread; ++i) {
      const int key = key_base + i * kKeyStep;
      const int index = begin + key;
      entries[i] = kExtra && key < count && index >= window ? extra_row[index - window] : -1;
    }
  }
  if constexpr (!kEarlyIndex) {
#pragma unroll
    for (int i = 0; i < kKeysPerThread; ++i) {
      const int index = begin + key_base + i * kKeyStep;
      window_slots[i] = index < window ? window_row[index] : -1;
    }
  }
  constexpr int kQueryChunks = kHeads * kChunks / kThreads;
  // Adaptation: the RoPE (cos, sin) pairs from freqs_cis[position] (interleaved).
  const float* rotation = freqs + static_cast<int64_t>(position) * 2 * d41attn::kRopePairs;
  const __nv_bfloat16* q_row = q + static_cast<int64_t>(row) * q_stride;
  uint4 query[kQueryChunks];
  float4 query_cos[kQueryChunks], query_sin[kQueryChunks];
#pragma unroll
  for (int i = 0; i < kQueryChunks; ++i) {
    const int item = static_cast<int>(threadIdx.x) + i * kThreads;
    const int dim = (item % kChunks) * 8;
    query[i] =
        *reinterpret_cast<const uint4*>(q_row + (item / kChunks % kLocalHeads) * kDim + dim);
    if (dim >= kDim - d41attn::kRopeDim) {
      const int rope_pair = (dim - (kDim - d41attn::kRopeDim)) / 2;
      const float4 low = *reinterpret_cast<const float4*>(rotation + 2 * rope_pair);
      const float4 high = *reinterpret_cast<const float4*>(rotation + 2 * rope_pair + 4);
      query_cos[i] = make_float4(low.x, low.z, high.x, high.z);
      query_sin[i] = make_float4(low.y, low.w, high.y, high.w);
    }
  }
  if (threadIdx.x < 2 * d41attn::kRopePairs) {
    const int pair = static_cast<int>(threadIdx.x) % d41attn::kRopePairs;
    smem.rotation[threadIdx.x] = rotation[2 * pair + static_cast<int>(threadIdx.x) / d41attn::kRopePairs];
  }

  if (count > 0) {
    RawChunk raw[kKeysPerThread];
    bool valid[kKeysPerThread];
    bool all_window = false, all_compressed = false;
    if constexpr ((kLean & kLeanKeys) != 0) {
      all_window = begin + kKeys <= window;
      all_compressed = begin >= window && count == kKeys;
    }
    auto load_window = [&](int i) {
      valid[i] = window_slots[i] >= 0;
      raw[i] = {make_uint2(0U, 0U), 0U};
      if (valid[i]) {
        raw[i] = LoadWindowChunk(PoolRow(window_pool.rows, window_pool.page_stride,
                                         window_pool.page_shift, d41attn::kWindowEntryBytes,
                                         window_slots[i]),
                                 chunk);
      }
    };
    auto load_compressed = [&](int i) {
      valid[i] = entries[i] >= 0;
      raw[i] = {make_uint2(0U, 0U), 0U};
      if (valid[i]) {
        raw[i] = LoadCompressedChunk(PoolRow(extra_pool.rows, extra_pool.page_stride,
                                             extra_pool.page_shift,
                                             d41attn::kCompressedEntryBytes, entries[i]),
                                     chunk);
      }
    };
    if (all_compressed) {
#pragma unroll
      for (int i = 0; i < kKeysPerThread; ++i) load_compressed(i);
    } else if (all_window) {
#pragma unroll
      for (int i = 0; i < kKeysPerThread; ++i) load_window(i);
    } else {
#pragma unroll
      for (int i = 0; i < kKeysPerThread; ++i) {
        const int key = key_base + i * kKeyStep;
        raw[i] = {make_uint2(0U, 0U), 0U};
        valid[i] = false;
        if (key < count) {
          if (begin + key < window) {
            load_window(i);
          } else {
            load_compressed(i);
          }
        }
      }
    }
#pragma unroll
    for (int i = 0; i < kQueryChunks; ++i) {
      const int item = static_cast<int>(threadIdx.x) + i * kThreads;
      const int head = item / kChunks;
      const int dim = (item % kChunks) * 8;
      uint4 value = query[i];
      if (dim >= kDim - d41attn::kRopeDim) {
        uint32_t words[4] = {value.x, value.y, value.z, value.w};
        const float cosines[4] = {query_cos[i].x, query_cos[i].y, query_cos[i].z, query_cos[i].w};
        const float sines[4] = {query_sin[i].x, query_sin[i].y, query_sin[i].z, query_sin[i].w};
#pragma unroll
        for (int index = 0; index < 4; ++index) {
          const __nv_bfloat162 pair = *reinterpret_cast<const __nv_bfloat162*>(&words[index]);
          float y0, y1;
          d41attn::Rotate(__low2float(pair), __high2float(pair), cosines[index], sines[index],
                          false, y0, y1);
          words[index] = d41attn::PackBf16(y0, y1);
        }
        value = make_uint4(words[0], words[1], words[2], words[3]);
      }
      *reinterpret_cast<uint4*>(&smem.q[head][dim]) = value;
    }
    {
      auto compressed_chunk = [&](int i) -> uint4 {
        if constexpr (kGather) {
          return CompressedChunkBf16(raw[i]);
        } else {
          return CompressedChunkLean(raw[i]);
        }
      };
      auto window_chunk = [&](int i) -> uint4 {
        if constexpr (kGather) {
          return WindowChunkBf16(raw[i]);
        } else {
          return WindowChunk(raw[i]);
        }
      };
      auto store_key = [&](int i, const uint4& value) {
        const int key = key_base + i * kKeyStep;
        *reinterpret_cast<uint4*>(&smem.kv.k[key][chunk * 8]) = value;
        if (chunk == 0) key_valid[key] = valid[i] ? 1 : 0;
      };
      if (all_compressed) {
#pragma unroll
        for (int i = 0; i < kKeysPerThread; ++i) store_key(i, compressed_chunk(i));
      } else if (all_window) {
#pragma unroll
        for (int i = 0; i < kKeysPerThread; ++i) store_key(i, window_chunk(i));
      } else {
#pragma unroll
        for (int i = 0; i < kKeysPerThread; ++i) {
          const int key = key_base + i * kKeyStep;
          uint4 value = make_uint4(0U, 0U, 0U, 0U);
          if (key < count) {
            value = begin + key < window ? window_chunk(i) : compressed_chunk(i);
          }
          store_key(i, value);
        }
      }
    }
    __syncthreads();

    if constexpr (kSwap) {
      if (warp < kKeys / 8) {
        const int tile = warp >> 1;
        const int upper = warp & 1;
        float partial[2][4] = {};
        const int matrix = lane >> 3;
        const int r = lane & 7;
#pragma unroll 2
        for (int k0 = 0; k0 < kDim; k0 += 16 * 4) {
#pragma unroll
          for (int c = 0; c < 2; ++c) {
            const int column = k0 + 16 * (2 * upper + c);
            uint32_t a[4];
            LdMatrixX4(a, &smem.kv.k[tile * 16 + (matrix & 1) * 8 + r][column + (matrix >> 1) * 8]);
            uint32_t b[2];
            LdMatrixX2(b, &smem.q[r][column + ((lane >> 3) & 1) * 8]);
            d41attn::MmaBf16(partial[c], a[0], a[1], a[2], a[3], b[0], b[1]);
          }
        }
        const int g = lane >> 2;
        const int t = lane & 3;
#pragma unroll
        for (int index = 0; index < 4; ++index) {
          smem.s[upper * 8 + 2 * t + (index & 1)][tile * 16 + g + (index >> 1) * 8] =
              partial[0][index] + partial[1][index];
        }
      }
    } else if (warp < kKeys / 8) {
      constexpr int kChains = 4;
      float partial[kChains][4] = {};
      const int matrix = lane >> 3;
      const int r = lane & 7;
#pragma unroll 2
      for (int k0 = 0; k0 < kDim; k0 += 16 * kChains) {
#pragma unroll
        for (int c = 0; c < kChains; ++c) {
          uint32_t a[4];
          LdMatrixX4(a, &smem.q[(matrix & 1) * 8 + r][k0 + 16 * c + (matrix >> 1) * 8]);
          uint32_t b[2];
          LdMatrixX2(b, &smem.kv.k[warp * 8 + r][k0 + 16 * c + ((lane >> 3) & 1) * 8]);
          d41attn::MmaBf16(partial[c], a[0], a[1], a[2], a[3], b[0], b[1]);
        }
      }
      float acc[4];
#pragma unroll
      for (int index = 0; index < 4; ++index) {
        acc[index] = (partial[0][index] + partial[1][index]) + (partial[2][index] + partial[3][index]);
      }
      const int g = lane >> 2;
      const int t = lane & 3;
#pragma unroll
      for (int index = 0; index < 4; ++index) {
        const int head = g + (index >> 1) * 8;
        const int key = warp * 8 + 2 * t + (index & 1);
        smem.s[head][key] =
            key < count && key_valid[key] ? __fmul_rn(acc[index], softmax_scale) : -INFINITY;
      }
    }
    __syncthreads();
    if (!kSwap || warp < 8) {
      const int head = warp;
      float scores[kKeys / 32];
      float top = -INFINITY;
#pragma unroll
      for (int i = 0; i < kKeys / 32; ++i) {
        if constexpr (kSwap) {
          const int key = lane + 32 * i;
          const float sum = smem.s[head][key] + smem.s[head + 8][key];
          scores[i] = key < count && key_valid[key] ? __fmul_rn(sum, softmax_scale) : -INFINITY;
        } else {
          scores[i] = smem.s[head][lane + 32 * i];
        }
        top = fmaxf(top, scores[i]);
      }
      top = d41attn::WarpMax(top);
      // Adaptation: a split of masked keys only (SGLang's -1 indices) takes p = 0.
      const float shift = top == -INFINITY ? 0.0F : top;
      float sum = 0.0F;
#pragma unroll
      for (int i = 0; i < kKeys / 32; ++i) {
        const float probability = expf(scores[i] - shift);
        sum += probability;
        smem.p[head][lane + 32 * i] = __float2bfloat16_rn(probability);
      }
      sum = d41attn::WarpSum(sum);
      if (lane == 0) {
        smem.stats[head][0] = top;
        smem.stats[head][1] = sum;
      }
    }
    __syncthreads();
    if constexpr (kSwap) {
      float pv[2][4] = {};
      const int matrix = lane >> 3;
      const int r = lane & 7;
#pragma unroll
      for (int k0 = 0; k0 < kKeys; k0 += 16) {
        uint32_t b[2];
        LdMatrixX2(b, &smem.p[r][k0 + ((lane >> 3) & 1) * 8]);
#pragma unroll
        for (int m = 0; m < 2; ++m) {
          const int m0 = warp * kFeaturesPerWarp + 16 * m;
          uint32_t a[4];
          LdMatrixX4Trans(a, &smem.kv.k[k0 + (matrix >> 1) * 8 + r][m0 + (matrix & 1) * 8]);
          d41attn::MmaBf16(pv[m], a[0], a[1], a[2], a[3], b[0], b[1]);
        }
      }
      __syncthreads();
      const int g = lane >> 2;
      const int t = lane & 3;
#pragma unroll
      for (int m = 0; m < 2; ++m) {
#pragma unroll
        for (int index = 0; index < 4; ++index) {
          smem.kv.o[2 * t + (index & 1)][warp * kFeaturesPerWarp + 16 * m + g + (index >> 1) * 8] =
              pv[m][index];
        }
      }
    } else {
      float acc[kOutputTiles][4];
#pragma unroll
      for (int tile = 0; tile < kOutputTiles; ++tile) {
#pragma unroll
        for (int index = 0; index < 4; ++index) acc[tile][index] = 0.0F;
      }
      {
        const int matrix = lane >> 3;
        const int r = lane & 7;
#pragma unroll
        for (int k0 = 0; k0 < kKeys; k0 += 16) {
          uint32_t a[4];
          LdMatrixX4(a, &smem.p[(matrix & 1) * 8 + r][k0 + (matrix >> 1) * 8]);
#pragma unroll
          for (int pair = 0; pair < kOutputTiles / 2; ++pair) {
            uint32_t b[4];
            const int n0 = warp * kFeaturesPerWarp + pair * 16 + (matrix >> 1) * 8;
            LdMatrixX4Trans(b, &smem.kv.k[k0 + (matrix & 1) * 8 + r][n0]);
            d41attn::MmaBf16(acc[2 * pair], a[0], a[1], a[2], a[3], b[0], b[1]);
            d41attn::MmaBf16(acc[2 * pair + 1], a[0], a[1], a[2], a[3], b[2], b[3]);
          }
        }
      }
      __syncthreads();
      const int g = lane >> 2;
      const int t = lane & 3;
#pragma unroll
      for (int tile = 0; tile < kOutputTiles; ++tile) {
        const int dim = warp * kFeaturesPerWarp + tile * 8 + 2 * t;
        *reinterpret_cast<float2*>(&smem.kv.o[g][dim]) = make_float2(acc[tile][0], acc[tile][1]);
        if constexpr ((kLean & kLeanMerge) == 0 || kLocalHeads > 8) {
          *reinterpret_cast<float2*>(&smem.kv.o[g + 8][dim]) =
              make_float2(acc[tile][2], acc[tile][3]);
        }
      }
    }
  } else {
    if (threadIdx.x < kHeads) {
      smem.stats[threadIdx.x][0] = -INFINITY;
      smem.stats[threadIdx.x][1] = 0.0F;
    }
    for (int index = static_cast<int>(threadIdx.x); index < kHeads * kDim; index += kThreads) {
      smem.kv.o[index / kDim][index % kDim] = 0.0F;
    }
  }
  d41attn::ClusterSyncShared();

  __nv_bfloat16* out = output + static_cast<int64_t>(row) * kLocalHeads * kDim;
  if constexpr ((kLean & kLeanMerge) == 0) {
    constexpr int kPairs =
        (kLocalHeads * kDim / 2 + kSplits * kThreads - 1) / (kSplits * kThreads);
    float2 parts[kPairs][kSplits];
    float2 stats[kPairs][kSplits];
#pragma unroll
    for (int i = 0; i < kPairs; ++i) {
      const int pair = split * kThreads + static_cast<int>(threadIdx.x) + i * kSplits * kThreads;
      const int head = pair / (kDim / 2);
      const int dim = (pair % (kDim / 2)) * 2;
#pragma unroll
      for (int rank = 0; rank < kSplits; ++rank) {
        if (pair < kLocalHeads * kDim / 2 && rank < kSplits) {
          const Storage* peer = cluster.map_shared_rank(&smem, rank);
          parts[i][rank] = *reinterpret_cast<const float2*>(&peer->kv.o[head][dim]);
          stats[i][rank] = *reinterpret_cast<const float2*>(&peer->stats[head][0]);
        }
      }
    }
#pragma unroll
    for (int i = 0; i < kPairs; ++i) {
      const int pair = split * kThreads + static_cast<int>(threadIdx.x) + i * kSplits * kThreads;
      if (pair >= kLocalHeads * kDim / 2) continue;
      const int head = pair / (kDim / 2);
      const int dim = (pair % (kDim / 2)) * 2;
      float top = -INFINITY;
#pragma unroll
      for (int rank = 0; rank < kSplits; ++rank) {
        if (rank < kSplits) top = fmaxf(top, stats[i][rank].x);
      }
      float denominator = 0.0F;
      float2 sum = make_float2(0.0F, 0.0F);
#pragma unroll
      for (int rank = 0; rank < kSplits; ++rank) {
        if (rank < kSplits) {
          const float weight = expf(stats[i][rank].x - top);
          denominator += stats[i][rank].y * weight;
          sum.x = __fmaf_rn(parts[i][rank].x, weight, sum.x);
          sum.y = __fmaf_rn(parts[i][rank].y, weight, sum.y);
        }
      }
      denominator += expf(smem.sink[head] - top);
      float y0 = d41attn::RoundBf16(__fdiv_rn(sum.x, denominator));
      float y1 = d41attn::RoundBf16(__fdiv_rn(sum.y, denominator));
      if (dim >= kDim - d41attn::kRopeDim) {
        const int rope_pair = (dim - (kDim - d41attn::kRopeDim)) / 2;
        float r0, r1;
        d41attn::Rotate(y0, y1, smem.rotation[rope_pair],
                        smem.rotation[d41attn::kRopePairs + rope_pair], true, r0, r1);
        y0 = d41attn::RoundBf16(r0);
        y1 = d41attn::RoundBf16(r1);
      }
      *reinterpret_cast<__nv_bfloat162*>(out + head * kDim + dim) = __floats2bfloat162_rn(y0, y1);
    }
  } else if constexpr (!kMergeStage) {
    static_assert(kSplits <= 32, "one lane per peer");
    for (int head = split; head < kLocalHeads; head += kSplits) {
      if (threadIdx.x < kDim / 2) {
        const int dim = 2 * static_cast<int>(threadIdx.x);
        float2 peer_stats = make_float2(-INFINITY, 0.0F);
        if (lane < kSplits) {
          peer_stats = *reinterpret_cast<const float2*>(
              &cluster.map_shared_rank(&smem, lane)->stats[head][0]);
        }
        float2 parts[kSplits];
        float2 stats[kSplits];
#pragma unroll
        for (int rank = 0; rank < kSplits; ++rank) {
          const Storage* peer = cluster.map_shared_rank(&smem, rank);
          parts[rank] = *reinterpret_cast<const float2*>(&peer->kv.o[head][dim]);
        }
#pragma unroll
        for (int rank = 0; rank < kSplits; ++rank) {
          stats[rank] = make_float2(__shfl_sync(0xFFFFFFFFU, peer_stats.x, rank),
                                    __shfl_sync(0xFFFFFFFFU, peer_stats.y, rank));
        }
        float top = -INFINITY;
#pragma unroll
        for (int rank = 0; rank < kSplits; ++rank) top = fmaxf(top, stats[rank].x);
        float denominator = 0.0F;
        float2 sum = make_float2(0.0F, 0.0F);
#pragma unroll
        for (int rank = 0; rank < kSplits; ++rank) {
          const float weight = expf(stats[rank].x - top);
          denominator += stats[rank].y * weight;
          sum.x = __fmaf_rn(parts[rank].x, weight, sum.x);
          sum.y = __fmaf_rn(parts[rank].y, weight, sum.y);
        }
        denominator += expf(smem.sink[head] - top);
        float y0 = d41attn::RoundBf16(__fdiv_rn(sum.x, denominator));
        float y1 = d41attn::RoundBf16(__fdiv_rn(sum.y, denominator));
        if (dim >= kDim - d41attn::kRopeDim) {
          const int rope_pair = (dim - (kDim - d41attn::kRopeDim)) / 2;
          float r0, r1;
          d41attn::Rotate(y0, y1, smem.rotation[rope_pair],
                          smem.rotation[d41attn::kRopePairs + rope_pair], true, r0, r1);
          y0 = d41attn::RoundBf16(r0);
          y1 = d41attn::RoundBf16(r1);
        }
        *reinterpret_cast<__nv_bfloat162*>(out + head * kDim + dim) =
            __floats2bfloat162_rn(y0, y1);
      }
    }
  } else {
    constexpr int kPer = 2;
    constexpr int kMergers = kDim / kPer;
    static_assert(kMergers <= kThreads && kMergers % 32 == 0, "merging threads");
    static_assert(kSplits < 32, "one lane per peer and one for the sink");
    static_assert((kDim - d41attn::kRopeDim) % (32 * kPer) == 0, "RoPE elements: whole warps");
    const bool merges = kMergers == kThreads || static_cast<int>(threadIdx.x) < kMergers;
    const int dim = static_cast<int>(threadIdx.x) * kPer;
    if (split >= kLocalHeads || !merges) ClusterArrive();
    for (int head = split; merges && head < kLocalHeads; head += kSplits) {
      float peer_max = -INFINITY;
      float peer_sum = 0.0F;
      if (lane < kSplits) {
        const float2 peer_stats = *reinterpret_cast<const float2*>(
            &cluster.map_shared_rank(&smem, lane)->stats[head][0]);
        peer_max = peer_stats.x;
        peer_sum = peer_stats.y;
      }
      float parts[kSplits][kPer];
#pragma unroll
      for (int rank = 0; rank < kSplits; ++rank) {
        const float* peer_o = &cluster.map_shared_rank(&smem, rank)->kv.o[head][dim];
        const float2 pair = *reinterpret_cast<const float2*>(peer_o);
        parts[rank][0] = pair.x;
        parts[rank][1] = pair.y;
      }
      const float top = d41attn::WarpMax(peer_max);
      const float lane_weight = expf((lane < kSplits ? peer_max : smem.sink[head]) - top);
      float denominator = 0.0F;
      float sum[kPer];
#pragma unroll
      for (int e = 0; e < kPer; ++e) sum[e] = 0.0F;
#pragma unroll
      for (int rank = 0; rank < kSplits; ++rank) {
        const float weight = __shfl_sync(0xFFFFFFFFU, lane_weight, rank);
        denominator += __shfl_sync(0xFFFFFFFFU, peer_sum, rank) * weight;
#pragma unroll
        for (int e = 0; e < kPer; ++e) sum[e] = __fmaf_rn(parts[rank][e], weight, sum[e]);
      }
      denominator += __shfl_sync(0xFFFFFFFFU, lane_weight, kSplits);
      if (head + kSplits >= kLocalHeads) ClusterArriveAfter(sum[0], sum[kPer - 1], denominator);
      float y[kPer];
#pragma unroll
      for (int e = 0; e < kPer; ++e) y[e] = d41attn::RoundBf16(__fdiv_rn(sum[e], denominator));
      if (dim >= kDim - d41attn::kRopeDim) {
        const int rope_pair = (dim - (kDim - d41attn::kRopeDim)) / 2;
        float r0, r1;
        d41attn::Rotate(y[0], y[1], smem.rotation[rope_pair],
                        smem.rotation[d41attn::kRopePairs + rope_pair], true, r0, r1);
        y[0] = d41attn::RoundBf16(r0);
        y[1] = d41attn::RoundBf16(r1);
      }
      *reinterpret_cast<__nv_bfloat162*>(out + head * kDim + dim) =
          __floats2bfloat162_rn(y[0], y[1]);
    }
  }
  if constexpr (kMergeStage) {
    ClusterWait();
  } else {
    d41attn::ClusterSyncRelaxed();
  }
}

}  // namespace

#define CORE_PARAMETERS                                                                     \
  const __nv_bfloat16 *q, int q_stride, const float *freqs, const position_t *positions,    \
      const uint8_t *window_rows, int64_t window_page_stride, int window_page_shift,        \
      const int32_t *window_indices, int window_index_stride, const int32_t *window_lengths, \
      const uint8_t *extra_rows, int64_t extra_page_stride, int extra_page_shift,           \
      const int32_t *extra_indices, int extra_index_stride, int extra_width,                \
      const int32_t *extra_lengths, const float *__restrict__ sink,                        \
      __nv_bfloat16 *__restrict__ output, float softmax_scale
#define CORE_ARGUMENTS                                                                      \
  q, q_stride, freqs, positions, Pool{window_rows, window_page_stride, window_page_shift},  \
      window_indices, window_index_stride, window_lengths,                                  \
      Pool{extra_rows, extra_page_stride, extra_page_shift}, extra_indices,                 \
      extra_index_stride, extra_width, extra_lengths, sink, output, softmax_scale

// The lean gathered sparse entry: grid (10, rows), cluster (10, 1, 1), 512 threads, two
// CTAs per SM by registers; the launch's dynamic shared memory is muLLM's pad (one CTA per
// SM).
extern "C" __global__ __launch_bounds__(kThreads, 2) void dsv41_core_lean_sparse(CORE_PARAMETERS) {
  BatchedAttention<kCompressedSplits, true, false, kLean4Gather>(CORE_ARGUMENTS);
}

extern "C" __global__ __launch_bounds__(kThreads, 2) void dsv41_core_lean_sparse_early(
    CORE_PARAMETERS) {
  BatchedAttention<kCompressedSplits, true, true, kLean4Gather>(CORE_ARGUMENTS);
}

// The lean window entry: grid (2, rows), cluster (2, 1, 1), 512 threads.
extern "C" __global__ __launch_bounds__(kThreads) void dsv41_core_lean_window(CORE_PARAMETERS) {
  BatchedAttention<kWindowSplits, false, false, kLeanWindow>(CORE_ARGUMENTS);
}

extern "C" __global__ __launch_bounds__(kThreads) void dsv41_core_lean_window_early(
    CORE_PARAMETERS) {
  BatchedAttention<kWindowSplits, false, true, kLeanWindow>(CORE_ARGUMENTS);
}

// The DSpark draft's window-only attention (adapted): the window entry over three splits,
// up to 192 window keys.
extern "C" __global__ __launch_bounds__(kThreads) void dsv41_core_lean_window3(CORE_PARAMETERS) {
  BatchedAttention<3, false, false, kLeanWindow, 3 * kKeys>(CORE_ARGUMENTS);
}

extern "C" __global__ __launch_bounds__(kThreads) void dsv41_core_lean_window3_early(
    CORE_PARAMETERS) {
  BatchedAttention<3, false, true, kLeanWindow, 3 * kKeys>(CORE_ARGUMENTS);
}
