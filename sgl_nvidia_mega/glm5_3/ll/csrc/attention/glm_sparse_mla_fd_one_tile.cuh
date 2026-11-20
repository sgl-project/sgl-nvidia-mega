// Sparse MLA decode as flash decoding over the top-k gathered KV rows, for DSA
// models (DeepSeek-V3.2 / GLM-5.x) with the raw 576 B FP8 KV layout of the
// `trtllm` DSA decode backend: 8 or 16 query heads per rank, top-k 2048.
//
// Adapted from TensorRT-LLM (Apache-2.0):
//   cpp/tensorrt_llm/kernels/sparseMlaFlashDecode.cu (Dongfeng Yu; 2808056e, 9bea245e, f5052d41)
//   cpp/tensorrt_llm/kernels/glmLowLatency/w4/sparseMlaDecode.cu (Xiao Wang; GB300 hardening)
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "glm_sparse_mla_fd_common.cuh"

namespace sglang {

namespace glm_ll_sparse_mla_fd {

/// \brief Shared-memory layout of one CTA for `kHeads` query heads.
template <int kHeads>
struct Layout {
  static_assert(kHeads == 8 || kHeads == 16, "8 or 16 query heads (one or two n8 MMA tiles)");
  static constexpr int kHeadTiles = kHeads / 8;
  static constexpr int kKTile = 0;
  static constexpr int kQTile = kKTile + kKeysPerCta * kKRowBytes;
  static constexpr int kPTile = kQTile + kHeads * kQRowHalves * 2;
  // [source split][head][dim within this CTA's 32 output dims] FP32 partials.
  static constexpr int kRecv = kPTile + kHeads * kPRowHalves * 2;
  // [maxima of the heads, then sums of the heads][source split] FP32.
  static constexpr int kRecvStats = kRecv + kSplits * kHeads * kDimsPerCta * 4;
  static constexpr int kRedMax = kRecvStats + 2 * kHeads * kSplits * 4;
  static constexpr int kRedSum = kRedMax + kWarps * kHeads * 4;
  // mbarriers: one per warp for its row gather, one for the pushed partials.
  static constexpr int kBarriers = kRedSum + kWarps * kHeads * 4;
  static constexpr int kBytes = kBarriers + (kWarps + 1) * 8;
  static constexpr uint32_t kRecvBytes = kSplits * (kHeads * kDimsPerCta + 2 * kHeads) * 4;
  static constexpr int kQChunks = kHeads * kHeadDimQk / 16;
  static constexpr int kQChunksPerThread = (kQChunks + kThreads - 1) / kThreads;
  static constexpr int kStatsPerThread = 2 * kHeads * kSplits / kThreads;
  static constexpr int kMergeHeadsPerWarp = kHeads / kWarps;

  static_assert(kQTile % 16 == 0 && kPTile % 16 == 0 && kRecv % 16 == 0, "16 B aligned tiles");
  static_assert(kBarriers % 8 == 0, "mbarrier alignment");
  static_assert(kStatsPerThread * kThreads == 2 * kHeads * kSplits, "whole statistics stores per thread");
  static_assert(kMergeHeadsPerWarp * kWarps == kHeads, "merge: whole heads per warp, one lane per dim");
  static_assert(kBytes <= 227 * 1024, "shared memory per CTA");
};

// v_b fold (MLA W_UV): CTA s of a row's cluster also computes columns [16 s, 16 s + 16) of
// every head of the o_proj input, vb_out[r, h * 256 + 16 s + v] =
// bf16(sum_d bf16(O[r, h, d]) * vb_proj[h, 16 s + v, d]).
inline constexpr int kVHeadDim = 256;
inline constexpr int kVPerCta = kVHeadDim / kSplits;
inline constexpr int kVbRowCopyBytes = kHeadDimV * 2;
// Staged v_b rows are padded by 16 B: rows shift by 4 banks, so the eight rows of an
// ldmatrix phase hit disjoint 16 B bank groups.
inline constexpr int kVbRowBytes = kVbRowCopyBytes + 16;
inline constexpr int kVbSliceBytes = kVPerCta * kVbRowBytes;
inline constexpr int kVbKSteps = kHeadDimV / 16;
static_assert(kVPerCta == 16, "one m16 row tile of v per head and CTA");
static_assert(kVbRowBytes / 16 % 8 == 1, "bank-conflict-free v_b pitch");

/**
 * \brief Shared-memory layout of the one-tile kernel with the v_b fold (8 heads).
 *
 * The base layout, then the v_b slices of heads 0 .. kHeads - 2 (staged with the
 * attention) and the fold's barriers. The last head's slice lands on the K rows once
 * every warp's P.V is done; the row's gathered BF16 output lands on the query tile,
 * which is dead before any CTA of the cluster can merge (a merge needs every CTA's
 * partials, which a CTA sends only after its QK).
 */
template <int kHeads>
struct FoldLayout {
  using Base = Layout<kHeads>;
  static constexpr int kEarlyHeads = kHeads - 1;
  static constexpr int kVb = (Base::kBytes + 127) / 128 * 128;
  static constexpr int kVbLate = Base::kKTile;
  static constexpr int kGatheredO = Base::kQTile;
  // mbarriers: early v_b slices, the last head's slice, the gathered output.
  static constexpr int kBarriers = kVb + kEarlyHeads * kVbSliceBytes;
  static constexpr int kBytes = kBarriers + 3 * 8;
  static constexpr int kEarlyCopies = kEarlyHeads * kVPerCta;
  static constexpr uint32_t kEarlyVbBytes = kEarlyCopies * kVbRowCopyBytes;
  static constexpr uint32_t kLateVbBytes = kVPerCta * kVbRowCopyBytes;
  static constexpr uint32_t kGatheredOBytes = kHeads * kHeadDimV * 2;

  static_assert(kGatheredOBytes <= kHeads * kQRowHalves * 2, "the gathered output fits on the query tile");
  static_assert(kVbSliceBytes <= kKeysPerCta * kKRowBytes, "the last head's slice fits on the K rows");
  static_assert(kEarlyCopies <= kThreads, "one early v_b row copy per thread");
  static_assert(kVb % 128 == 0 && kBarriers % 8 == 0, "alignment");
  static_assert(kBytes <= 227 * 1024, "shared memory per CTA");
};

/// \brief Kernel arguments (passed as one `__grid_constant__` struct).
struct Params {
  const uint8_t* q;         // [rows, kHeads, 576] e4m3, rows contiguous
  const uint8_t* kv;        // [num_slots, 576] e4m3: 512 latent + 64 rope per slot
  const int32_t* indices;   // [rows, indices_stride] global slots; < 0 or >= num_slots masks
  const int32_t* seq_lens;  // [>= rows] causal KV length of each row
  __nv_bfloat16* out;       // [rows, kHeads, 512]; with the v_b fold [rows, out_stride]
  int64_t indices_stride;
  int64_t num_slots;
  float scale_log2;  // softmax scale * dequant scales * log2(e)
  float out_scale;   // applied to the normalized P.V
  // v_b fold only: [kHeads, 256, 512] bf16, and the row stride of `out` in elements.
  const __nv_bfloat16* vb_proj;
  int64_t out_stride;
};

#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)

/**
 * One head's 16 v rows of the v_b product: d[0] = row g, d[2] = row g + 8 (any column),
 * from the staged rows at `a_addr` (lane's ldmatrix row address) and the head's BF16
 * output `o_words` (u32 pairs). Step j holds k in [16 j, 16 j + 16), accumulated in one
 * chain in k order: the result equals cuBLAS's BF16 BMM bit for bit.
 */
SGL_DEVICE void vb_rows(float (&d)[4], uint32_t a_addr, const uint32_t* o_words, int tig) {
  float acc[4] = {};
#pragma unroll
  for (int j = 0; j < kVbKSteps; ++j) {
    uint32_t a[4];
    ldmatrix_x4(a, a_addr + j * 32);
    mma_bf16(acc, a[0], a[1], a[2], a[3], o_words[8 * j + tig], o_words[8 * j + tig + 4]);
  }
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    d[i] = acc[i];
  }
}

#endif  // defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)

/**
 * \brief Sparse MLA decode of one query row per 16-CTA cluster.
 *
 * Semantics (TRTLLM-Gen generation kernel, plus two masks): position j of row r
 * is used iff j < min(seq_lens[r], 2048), 0 <= indices[r, j] < num_slots.
 * Scores are FP32 sums of exact e4m3 products times `scale_log2` (base 2);
 * P = exp2(S - CTA max) enters P.V as e4m3(448 P), the row sums use the FP32 P;
 * partials merge in FP32. Repeated slots count per occurrence; a row without a
 * usable position outputs zeros.
 *
 * PDL: before `griddepcontrol.wait` the kernel touches only its own shared
 * memory and its cluster barrier. It triggers its dependents once its gathers
 * are issued; its only global write is `out`, so a dependent must wait before
 * reading `out` or writing any buffer read here.
 *
 * kFuseVb (8 heads): `out` is the o_proj input [rows, out_stride] instead, the
 * v_b product of the BF16-rounded output (see FoldLayout). Before the wait the
 * kernel also stages its slices of `vb_proj`, a weight never written in the forward.
 *
 * \tparam kHeads       Query heads per row: 8 or 16 (one or two n8 MMA tiles)
 * \tparam kUsePDL      Emit the PDL wait and trigger
 * \tparam kFuseVb      Write the v_b product (8 heads)
 */
template <int kHeads, bool kUsePDL, bool kFuseVb = false>
// 8 heads cap registers at 128 so that two CTAs can share an SM (the fold fills one).
__global__ void __launch_bounds__(kThreads, kHeads == 8 && !kFuseVb ? 2 : 1)
    glm_sparse_mla_fd_kernel(const __grid_constant__ Params params) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
  static_assert(!kFuseVb || kHeads == 8, "the v_b fold maps warp h to head h");
  using L = Layout<kHeads>;
  constexpr int kHeadTiles = L::kHeadTiles;
  extern __shared__ __align__(128) uint8_t smem[];

  const int tid = threadIdx.x;
  const int warp = tid >> 5;
  const int lane = tid & 31;
  const int group = lane >> 2;
  const int tig = lane & 3;
  // The cluster spans gridDim.x, so the split is the CTA's rank in the cluster.
  const int split = blockIdx.x;
  const int row = blockIdx.y;

  uint8_t* k_tile = smem + L::kKTile;
  __half* q_tile = reinterpret_cast<__half*>(smem + L::kQTile);
  __half* p_tile = reinterpret_cast<__half*>(smem + L::kPTile);
  float* recv = reinterpret_cast<float*>(smem + L::kRecv);
  float* recv_stats = reinterpret_cast<float*>(smem + L::kRecvStats);
  float* red_max = reinterpret_cast<float*>(smem + L::kRedMax);
  float* red_sum = reinterpret_cast<float*>(smem + L::kRedSum);
  const uint32_t gather_bar = smem_addr(smem + L::kBarriers + warp * 8);
  const uint32_t recv_bar = smem_addr(smem + L::kBarriers + kWarps * 8);
  [[maybe_unused]] uint32_t vb_bar = 0;
  [[maybe_unused]] uint32_t vb_late_bar = 0;
  [[maybe_unused]] uint32_t gathered_bar = 0;
  if constexpr (kFuseVb) {
    vb_bar = smem_addr(smem + FoldLayout<kHeads>::kBarriers);
    vb_late_bar = vb_bar + 8;
    gathered_bar = vb_bar + 16;
  }

  if (lane == 0) {
    mbar_init(gather_bar, kKeysPerWarp);
    if (warp == 0) {
      mbar_init(recv_bar, 1);
      if constexpr (kFuseVb) {
        mbar_init(vb_bar, 1);
        mbar_init(vb_late_bar, 1);
        mbar_init(gathered_bar, 1);
      }
    }
    // The initializations must be visible to the bulk-copy engine and to the peers' st.async.
    fence_mbar_init();
    if (warp == 0) {
      // Every CTA of the cluster sends this CTA its partials and statistics exactly once.
      mbar_arrive_expect_tx(recv_bar, L::kRecvBytes);
      if constexpr (kFuseVb) {
        // ... and its 32 output dims of every head in BF16.
        mbar_arrive_expect_tx(gathered_bar, FoldLayout<kHeads>::kGatheredOBytes);
        mbar_arrive_expect_tx(vb_bar, FoldLayout<kHeads>::kEarlyVbBytes);
        mbar_arrive_expect_tx(vb_late_bar, FoldLayout<kHeads>::kLateVbBytes);
      }
    }
  }
  __syncthreads();
  // Paired with the cluster wait before the first remote store.
  cluster_arrive_relaxed();

  // v_b rows [16 s, 16 s + 16) of heads 0 .. kHeads - 2, one 1 KiB row per thread, issued
  // before the PDL wait (a weight).
  if constexpr (kFuseVb) {
    using F = FoldLayout<kHeads>;
    if (tid < F::kEarlyCopies) {
      const int head = tid / kVPerCta;
      const int v = tid - head * kVPerCta;
      const __nv_bfloat16* src =
          params.vb_proj + (static_cast<int64_t>(head) * kVHeadDim + split * kVPerCta + v) * kHeadDimV;
      bulk_copy_g2s(smem_addr(smem + F::kVb + head * kVbSliceBytes + v * kVbRowBytes), src, kVbRowCopyBytes, vb_bar);
    }
  }

  device::PDLWaitPrimary<kUsePDL>();

  // The query does not depend on the index list: start its loads first.
  const uint8_t* q_row = params.q + static_cast<int64_t>(row) * kHeads * kHeadDimQk;
  uint4 q_raw[L::kQChunksPerThread];
#pragma unroll
  for (int i = 0; i < L::kQChunksPerThread; ++i) {
    const int chunk = tid + i * kThreads;
    if (chunk < L::kQChunks) {
      q_raw[i] = ld_global_v4(q_row + chunk * 16);
    }
  }

  // Lane l < 16 of warp w owns position 128 s + 16 w + l; index and length share one round trip.
  bool valid = false;
  int32_t slot = 0;
  uint8_t* my_row = k_tile + (warp * kKeysPerWarp + lane) * kKRowBytes;
  if (lane < kKeysPerWarp) {
    const int key = split * kKeysPerCta + warp * kKeysPerWarp + lane;
    slot = ld_global_s32(params.indices + static_cast<int64_t>(row) * params.indices_stride + key);
    const int32_t seq_len = ld_global_s32(params.seq_lens + row);
    valid = key < seq_len && slot >= 0 && static_cast<int64_t>(slot) < params.num_slots;
  }
  if (valid) {
    mbar_arrive_expect_tx(gather_bar, kHeadDimQk);
    bulk_copy_g2s(smem_addr(my_row), params.kv + static_cast<int64_t>(slot) * kHeadDimQk, kHeadDimQk, gather_bar);
  }
  const uint32_t valid_bits = __ballot_sync(0xffffffffU, valid);

  // FP16 copy of the query, shared by all warps.
#pragma unroll
  for (int i = 0; i < L::kQChunksPerThread; ++i) {
    const int chunk = tid + i * kThreads;
    if (chunk < L::kQChunks) {
      const int head = chunk / (kHeadDimQk / 16);
      const int col = (chunk - head * (kHeadDimQk / 16)) * 16;
      uint4 lo;
      uint4 hi;
      lo.x = e4m3x2_to_half2(q_raw[i].x);
      lo.y = e4m3x2_to_half2(q_raw[i].x >> 16);
      lo.z = e4m3x2_to_half2(q_raw[i].y);
      lo.w = e4m3x2_to_half2(q_raw[i].y >> 16);
      hi.x = e4m3x2_to_half2(q_raw[i].z);
      hi.y = e4m3x2_to_half2(q_raw[i].z >> 16);
      hi.z = e4m3x2_to_half2(q_raw[i].w);
      hi.w = e4m3x2_to_half2(q_raw[i].w >> 16);
      uint4* dst = reinterpret_cast<uint4*>(q_tile + head * kQRowHalves + col);
      dst[0] = lo;
      dst[1] = hi;
    }
  }

  // Every global input is read or in flight; nothing below depends on overlap for progress.
  device::PDLTriggerSecondary<kUsePDL>();

  // A CTA whose 128 positions are all masked (kv_len < 128 s, the N1 case) skips
  // its gathers' zero-fill, QK and P.V, and contributes max -inf, sum 0, zero partials.
  const bool cta_active = __syncthreads_or(valid ? 1 : 0) != 0;

  // Masked rows of an active CTA are zeroed so that zero probabilities never meet
  // non-finite bytes in P.V.
  if (cta_active && lane < kKeysPerWarp && !valid) {
#pragma unroll 4
    for (int c = 0; c < kHeadDimQk / 16; ++c) {
      *reinterpret_cast<uint4*>(my_row + c * 16) = make_uint4(0U, 0U, 0U, 0U);
    }
    mbar_arrive(gather_bar);
  }

  if (cta_active) {
    mbar_wait(gather_bar, 0);
    __syncwarp();

    // S^T = K Q^T for this warp's 16 keys: keys on the M side, heads on N. Within
    // each 16-value k chunk the MMA k slots are permuted so that the two halves of
    // an ldmatrix register map straight onto the A fragment: slots 2t, 2t+1 hold
    // k = 4t, 4t+1 and slots 2t+8, 2t+9 hold k = 4t+2, 4t+3; the query is read in
    // that order.
    float s_acc[kHeadTiles][4];
    {
      float acc[kHeadTiles][2][4] = {};
      const uint32_t a_addr = smem_addr(k_tile + (warp * kKeysPerWarp + (lane & 15)) * kKRowBytes + (lane >> 4) * 16);
#pragma unroll
      for (int kk = 0; kk < kQkSteps; ++kk) {
        uint32_t r[4];
        ldmatrix_x4(r, a_addr + kk * 32);
        const uint32_t a00 = e4m3x2_to_half2(r[0]);
        const uint32_t a01 = e4m3x2_to_half2(r[1]);
        const uint32_t a02 = e4m3x2_to_half2(r[0] >> 16);
        const uint32_t a03 = e4m3x2_to_half2(r[1] >> 16);
        const uint32_t a10 = e4m3x2_to_half2(r[2]);
        const uint32_t a11 = e4m3x2_to_half2(r[3]);
        const uint32_t a12 = e4m3x2_to_half2(r[2] >> 16);
        const uint32_t a13 = e4m3x2_to_half2(r[3] >> 16);
#pragma unroll
        for (int ht = 0; ht < kHeadTiles; ++ht) {
          const __half* q_frag = q_tile + (ht * 8 + group) * kQRowHalves + 4 * tig + kk * 32;
          const uint2 b0 = *reinterpret_cast<const uint2*>(q_frag);
          const uint2 b1 = *reinterpret_cast<const uint2*>(q_frag + 16);
          mma_f16(acc[ht][0], a00, a01, a02, a03, b0.x, b0.y);
          mma_f16(acc[ht][1], a10, a11, a12, a13, b1.x, b1.y);
        }
      }
#pragma unroll
      for (int ht = 0; ht < kHeadTiles; ++ht) {
#pragma unroll
        for (int i = 0; i < 4; ++i) {
          s_acc[ht][i] = acc[ht][0][i] + acc[ht][1][i];
        }
      }
    }

    // Fragment element (key, head): 0 -> (group, 2 tig), 1 -> (group, 2 tig + 1), 2/3 -> key + 8.
    const bool valid0 = (valid_bits >> group) & 1U;
    const bool valid1 = (valid_bits >> (group + 8)) & 1U;
    float s[kHeadTiles][4];
#pragma unroll
    for (int ht = 0; ht < kHeadTiles; ++ht) {
      s[ht][0] = valid0 ? s_acc[ht][0] * params.scale_log2 : -INFINITY;
      s[ht][1] = valid0 ? s_acc[ht][1] * params.scale_log2 : -INFINITY;
      s[ht][2] = valid1 ? s_acc[ht][2] * params.scale_log2 : -INFINITY;
      s[ht][3] = valid1 ? s_acc[ht][3] * params.scale_log2 : -INFINITY;
      float max_a = fmaxf(s[ht][0], s[ht][2]);
      float max_b = fmaxf(s[ht][1], s[ht][3]);
#pragma unroll
      for (int offset = 4; offset < 32; offset <<= 1) {
        max_a = fmaxf(max_a, __shfl_xor_sync(0xffffffffU, max_a, offset));
        max_b = fmaxf(max_b, __shfl_xor_sync(0xffffffffU, max_b, offset));
      }
      if (lane < 4) {
        red_max[warp * kHeads + ht * 8 + 2 * tig] = max_a;
        red_max[warp * kHeads + ht * 8 + 2 * tig + 1] = max_b;
      }
    }
    __syncthreads();

#pragma unroll
    for (int ht = 0; ht < kHeadTiles; ++ht) {
      float head_max_a = -INFINITY;
      float head_max_b = -INFINITY;
#pragma unroll
      for (int w = 0; w < kWarps; ++w) {
        head_max_a = fmaxf(head_max_a, red_max[w * kHeads + ht * 8 + 2 * tig]);
        head_max_b = fmaxf(head_max_b, red_max[w * kHeads + ht * 8 + 2 * tig + 1]);
      }
      float p[4];
      p[0] = s[ht][0] == -INFINITY ? 0.f : fast_exp2(s[ht][0] - head_max_a);
      p[1] = s[ht][1] == -INFINITY ? 0.f : fast_exp2(s[ht][1] - head_max_b);
      p[2] = s[ht][2] == -INFINITY ? 0.f : fast_exp2(s[ht][2] - head_max_a);
      p[3] = s[ht][3] == -INFINITY ? 0.f : fast_exp2(s[ht][3] - head_max_b);

      float sum_a = p[0] + p[2];
      float sum_b = p[1] + p[3];
#pragma unroll
      for (int offset = 4; offset < 32; offset <<= 1) {
        sum_a += __shfl_xor_sync(0xffffffffU, sum_a, offset);
        sum_b += __shfl_xor_sync(0xffffffffU, sum_b, offset);
      }
      if (lane < 4) {
        red_sum[warp * kHeads + ht * 8 + 2 * tig] = sum_a;
        red_sum[warp * kHeads + ht * 8 + 2 * tig + 1] = sum_b;
      }
      const int key = warp * kKeysPerWarp + group;
      const int head = ht * 8 + 2 * tig;
      p_tile[head * kPRowHalves + key] = round_probability(p[0]);
      p_tile[(head + 1) * kPRowHalves + key] = round_probability(p[1]);
      p_tile[head * kPRowHalves + key + 8] = round_probability(p[2]);
      p_tile[(head + 1) * kPRowHalves + key + 8] = round_probability(p[3]);
    }
    // Every warp observed its own gather barrier before its QK; this barrier orders
    // all key rows and the probabilities before the P.V reads of the other warps.
    __syncthreads();
  }

  // Every peer has initialized its barriers once the cluster barrier completes.
  cluster_wait();

  // This CTA's statistics to every CTA of the cluster: value t / 16 (maxima of the
  // heads, then sums) to rank t % 16.
#pragma unroll
  for (int i = 0; i < L::kStatsPerThread; ++i) {
    const int item = tid + i * kThreads;
    const int index = item / kSplits;
    const int dest = item % kSplits;
    const int head = index % kHeads;
    const bool is_max = index < kHeads;
    float value = is_max ? -INFINITY : 0.f;
    if (cta_active) {
#pragma unroll
      for (int w = 0; w < kWarps; ++w) {
        value = is_max ? fmaxf(value, red_max[w * kHeads + head]) : value + red_sum[w * kHeads + head];
      }
    }
    store_async(map_to_rank(smem_addr(recv_stats + index * kSplits + split), dest), value, map_to_rank(recv_bar, dest));
  }

  // O^T = V^T P^T for this warp's 64 output dims, one 16-dim tile at a time; each
  // finished tile goes to the CTA that finalizes its dims while the next one is
  // computed. Transposing the e4m3 V rows with b16 ldmatrix gives keys (2t, 2t+1)
  // x dims (2g, 2g+1) per register; a byte permute splits the two dims, so M row g
  // of a tile is dim 2g and M row g + 8 is dim 2g + 1.
  {
    constexpr float kInvProbScale = 1.f / kProbScale;
    const uint32_t recv_base = smem_addr(recv);
    if (cta_active) {
      uint32_t p_frag[kHeadTiles][kPvSteps][2][2];
#pragma unroll
      for (int ht = 0; ht < kHeadTiles; ++ht) {
        const __half* p_row = p_tile + (ht * 8 + group) * kPRowHalves + 2 * tig;
#pragma unroll
        for (int ks = 0; ks < kPvSteps; ++ks) {
#pragma unroll
          for (int half = 0; half < 2; ++half) {
            p_frag[ht][ks][half][0] = *reinterpret_cast<const uint32_t*>(p_row + ks * 32 + half * 16);
            p_frag[ht][ks][half][1] = *reinterpret_cast<const uint32_t*>(p_row + ks * 32 + half * 16 + 8);
          }
        }
      }
      const uint32_t v_addr = smem_addr(k_tile + lane * kKRowBytes + warp * kDimsPerWarp);
#pragma unroll
      for (int mt = 0; mt < kPvTilesPerWarp; ++mt) {
        // Independent accumulators per MMA keep the tensor pipe busy within one tile.
        float part[kHeadTiles][kPvSteps][2][4] = {};
#pragma unroll
        for (int ks = 0; ks < kPvSteps; ++ks) {
          uint32_t r[4];
          ldmatrix_x4_trans(r, v_addr + ks * 32 * kKRowBytes + mt * 16);
          uint32_t dim_even[4];
          uint32_t dim_odd[4];
#pragma unroll
          for (int j = 0; j < 4; ++j) {
            const uint32_t pairs = __byte_perm(r[j], 0, 0x3120);
            dim_even[j] = e4m3x2_to_half2(pairs);
            dim_odd[j] = e4m3x2_to_half2(pairs >> 16);
          }
#pragma unroll
          for (int ht = 0; ht < kHeadTiles; ++ht) {
            mma_f16(
                part[ht][ks][0],
                dim_even[0],
                dim_odd[0],
                dim_even[1],
                dim_odd[1],
                p_frag[ht][ks][0][0],
                p_frag[ht][ks][0][1]);
            mma_f16(
                part[ht][ks][1],
                dim_even[2],
                dim_odd[2],
                dim_even[3],
                dim_odd[3],
                p_frag[ht][ks][1][0],
                p_frag[ht][ks][1][1]);
          }
        }
        const int dim = warp * kDimsPerWarp + mt * 16 + 2 * group;
        const uint32_t dest = dim / kDimsPerCta;
        const int local_dim = dim - static_cast<int>(dest) * kDimsPerCta;
        const uint32_t remote_bar = map_to_rank(recv_bar, dest);
#pragma unroll
        for (int ht = 0; ht < kHeadTiles; ++ht) {
          float o[4];
#pragma unroll
          for (int i = 0; i < 4; ++i) {
            o[i] = ((part[ht][0][0][i] + part[ht][0][1][i]) + (part[ht][1][0][i] + part[ht][1][1][i])) +
                   ((part[ht][2][0][i] + part[ht][2][1][i]) + (part[ht][3][0][i] + part[ht][3][1][i]));
          }
          const uint32_t addr_a =
              recv_base + ((split * kHeads + ht * 8 + 2 * tig) * kDimsPerCta + local_dim) * sizeof(float);
          const uint32_t addr_b = addr_a + kDimsPerCta * sizeof(float);
          store_async2(map_to_rank(addr_a, dest), o[0] * kInvProbScale, o[2] * kInvProbScale, remote_bar);
          store_async2(map_to_rank(addr_b, dest), o[1] * kInvProbScale, o[3] * kInvProbScale, remote_bar);
        }
      }
    } else {
      // Same addresses and byte counts as above, zero partials.
#pragma unroll
      for (int mt = 0; mt < kPvTilesPerWarp; ++mt) {
        const int dim = warp * kDimsPerWarp + mt * 16 + 2 * group;
        const uint32_t dest = dim / kDimsPerCta;
        const int local_dim = dim - static_cast<int>(dest) * kDimsPerCta;
        const uint32_t remote_bar = map_to_rank(recv_bar, dest);
#pragma unroll
        for (int ht = 0; ht < kHeadTiles; ++ht) {
          const uint32_t addr_a =
              recv_base + ((split * kHeads + ht * 8 + 2 * tig) * kDimsPerCta + local_dim) * sizeof(float);
          const uint32_t addr_b = addr_a + kDimsPerCta * sizeof(float);
          store_async2(map_to_rank(addr_a, dest), 0.f, 0.f, remote_bar);
          store_async2(map_to_rank(addr_b, dest), 0.f, 0.f, remote_bar);
        }
      }
    }
  }

  if constexpr (kFuseVb) {
    using F = FoldLayout<kHeads>;
    // Every warp's P.V reads of the K rows precede the last head's v_b slice copy onto them.
    __syncthreads();
    if (warp == 0 && lane < kVPerCta) {
      fence_proxy_async_smem();
      const __nv_bfloat16* src =
          params.vb_proj + (static_cast<int64_t>(kHeads - 1) * kVHeadDim + split * kVPerCta + lane) * kHeadDimV;
      bulk_copy_g2s(smem_addr(smem + F::kVbLate + lane * kVbRowBytes), src, kVbRowCopyBytes, vb_late_bar);
    }
  }

  // No CTA leaves before its own receive barrier completes, which needs every
  // peer's bytes, so no st.async targets an exited CTA.
  mbar_wait_cluster(recv_bar, 0);

  // Finalize 32 of the 512 output dims of every head from the pushed partials: warp
  // w handles heads w (and w + 8 at 16 heads), lane i < 16 the statistics of
  // source split i, every lane one output dim.
#pragma unroll
  for (int hh = 0; hh < L::kMergeHeadsPerWarp; ++hh) {
    const int head = warp + hh * kWarps;
    const float peer_max = lane < kSplits ? recv_stats[head * kSplits + lane] : -INFINITY;
    const float peer_sum = lane < kSplits ? recv_stats[(kHeads + head) * kSplits + lane] : 0.f;
    float global_max = peer_max;
#pragma unroll
    for (int offset = kSplits / 2; offset > 0; offset >>= 1) {
      global_max = fmaxf(global_max, __shfl_xor_sync(0xffffffffU, global_max, offset));
    }
    const float weight = peer_max == -INFINITY ? 0.f : fast_exp2(peer_max - global_max);
    float sum = weight * peer_sum;
#pragma unroll
    for (int offset = kSplits / 2; offset > 0; offset >>= 1) {
      sum += __shfl_xor_sync(0xffffffffU, sum, offset);
    }
    sum = __shfl_sync(0xffffffffU, sum, 0);
    float acc = 0.f;
#pragma unroll
    for (int i = 0; i < kSplits; ++i) {
      acc += __shfl_sync(0xffffffffU, weight, i) * recv[(i * kHeads + head) * kDimsPerCta + lane];
    }
    const float result = sum > 0.f ? acc * (params.out_scale / sum) : 0.f;
    if constexpr (!kFuseVb) {
      params.out[(static_cast<int64_t>(row) * kHeads + head) * kHeadDimV + split * kDimsPerCta + lane] =
          __float2bfloat16_rn(result);
    } else {
      // The BF16 output (what a separate v_b BMM reads) to every CTA of the cluster: lane
      // l sends 16 B chunk l % 4 of this CTA's 32 dims of the head to ranks l / 4 and l / 4 + 8.
      const uint32_t mine = __bfloat16_as_ushort(__float2bfloat16_rn(result));
      const int chunk = lane & 3;
      uint32_t words[4];
#pragma unroll
      for (int i = 0; i < 4; ++i) {
        const uint32_t lo = __shfl_sync(0xffffffffU, mine, 8 * chunk + 2 * i);
        const uint32_t hi = __shfl_sync(0xffffffffU, mine, 8 * chunk + 2 * i + 1);
        words[i] = (lo & 0xffffU) | (hi << 16);
      }
      const uint32_t local = smem_addr(
          smem + FoldLayout<kHeads>::kGatheredO + (head * kHeadDimV + split * kDimsPerCta + 8 * chunk) * 2);
#pragma unroll
      for (int k = 0; k < 2; ++k) {
        const uint32_t dest = static_cast<uint32_t>((lane >> 2) + 8 * k);
        store_async_v4(map_to_rank(local, dest), words, map_to_rank(gathered_bar, dest));
      }
    }
  }

  if constexpr (kFuseVb) {
    using F = FoldLayout<kHeads>;
    // As for the partials: no CTA leaves before every peer's output chunks have landed.
    mbar_wait_cluster(gathered_bar, 0);
    const int head = warp;
    if (head < F::kEarlyHeads) {
      mbar_wait(vb_bar, 0);
    } else {
      mbar_wait(vb_late_bar, 0);
    }
    const uint8_t* w_head = head < F::kEarlyHeads ? smem + F::kVb + head * kVbSliceBytes : smem + F::kVbLate;
    const uint32_t a_addr = smem_addr(w_head + (lane & 15) * kVbRowBytes + (lane >> 4) * 16);
    const uint32_t* o_words = reinterpret_cast<const uint32_t*>(smem + F::kGatheredO + head * kHeadDimV * 2);
    float d[4];
    vb_rows(d, a_addr, o_words, tig);
    // Every column of D holds the same product: thread (g, 0) stores v rows g and g + 8.
    if (tig == 0) {
      __nv_bfloat16* dst = params.out + static_cast<int64_t>(row) * params.out_stride + head * kVHeadDim +
                           split * kVPerCta + group;
      dst[0] = __float2bfloat16_rn(d[0]);
      dst[8] = __float2bfloat16_rn(d[2]);
    }
  }
#else
  __trap();
#endif
}

}  // namespace glm_ll_sparse_mla_fd

/**
 * \brief Host launcher of `glm_sparse_mla_fd_kernel` for one head count.
 *
 * \tparam kHeads  Query heads per row: 8 or 16
 * \tparam kUsePDL Launch with programmatic stream serialization and emit the PDL wait/trigger
 * \tparam kFuseVb The v_b fold instance (8 heads; `run_fold` only)
 */
template <int kHeads, bool kUsePDL, bool kFuseVb = false>
struct GlmLlSparseMlaFdOneTile {
  using Layout = glm_ll_sparse_mla_fd::Layout<kHeads>;
  static constexpr auto kKernel = glm_ll_sparse_mla_fd::glm_sparse_mla_fd_kernel<kHeads, kUsePDL, kFuseVb>;
  static constexpr int64_t kSmemBytes = [] {
    if constexpr (kFuseVb) {
      return static_cast<int64_t>(glm_ll_sparse_mla_fd::FoldLayout<kHeads>::kBytes);
    } else {
      return static_cast<int64_t>(Layout::kBytes);
    }
  }();

  /// \brief Dynamic shared memory the kernel needs; launches may request more (to spread CTAs).
  static int64_t smem_bytes() {
    return kSmemBytes;
  }

  /**
   * \brief Dynamic shared memory that keeps a second CTA of this kernel off the SM
   * (half the SM's shared memory), or `smem_bytes()` when one CTA already fills it.
   */
  static int64_t spread_smem_bytes() {
    int device = 0;
    int per_sm = 0;
    int per_block = 0;
    CHECK_CUDA(cudaGetDevice(&device));
    CHECK_CUDA(cudaDeviceGetAttribute(&per_sm, cudaDevAttrMaxSharedMemoryPerMultiprocessor, device));
    CHECK_CUDA(cudaDeviceGetAttribute(&per_block, cudaDevAttrMaxSharedMemoryPerBlockOptin, device));
    const int64_t half = (static_cast<int64_t>(per_sm) / 2 + 127) / 128 * 128;
    return std::min<int64_t>(std::max<int64_t>(kSmemBytes, half), per_block);
  }

  /**
   * \brief Raise the kernel's dynamic shared-memory limit on `device_id` to `bytes` (never lowers
   * it) and allow its non-portable 16-CTA cluster there. Function attributes are per device, so
   * the memo is too.
   */
  static void ensure_smem_limit(int device_id, int64_t bytes) {
    static std::mutex mutex;
    static std::map<int, int64_t> configured;
    const std::lock_guard<std::mutex> lock(mutex);
    int64_t& limit = configured[device_id];
    if (bytes <= limit) return;
    // cudaFuncSetAttribute acts on the current device.
    tvm::ffi::CUDADeviceGuard guard(device_id);
    CHECK_CUDA(cudaFuncSetAttribute(kKernel, cudaFuncAttributeMaxDynamicSharedMemorySize, static_cast<int>(bytes)));
    CHECK_CUDA(cudaFuncSetAttribute(kKernel, cudaFuncAttributeNonPortableClusterSizeAllowed, 1));
    limit = bytes;
  }

  /**
   * \brief Clusters of 16 CTAs of this kernel that can be resident at once on the current device
   * with `smem` bytes of dynamic shared memory per CTA (`cudaOccupancyMaxActiveClusters`).
   */
  static int64_t max_active_clusters(int64_t smem) {
    using namespace host;
    CHECK_HOST(smem >= kSmemBytes) << "smem " << smem << " below the kernel's " << kSmemBytes;
    int device = 0;
    CHECK_CUDA(cudaGetDevice(&device));
    ensure_smem_limit(device, smem);
    cudaLaunchConfig_t config = {};
    config.gridDim = dim3(glm_ll_sparse_mla_fd::kSplits, 1, 1);
    config.blockDim = dim3(glm_ll_sparse_mla_fd::kThreads, 1, 1);
    config.dynamicSmemBytes = static_cast<size_t>(smem);
    cudaLaunchAttribute attr = {};
    attr.id = cudaLaunchAttributeClusterDimension;
    attr.val.clusterDim.x = glm_ll_sparse_mla_fd::kSplits;
    attr.val.clusterDim.y = 1;
    attr.val.clusterDim.z = 1;
    config.attrs = &attr;
    config.numAttrs = 1;
    int num_clusters = 0;
    CHECK_CUDA(cudaOccupancyMaxActiveClusters(&num_clusters, kKernel, &config));
    return num_clusters;
  }

  /**
   * \brief Launch one 16-CTA cluster per query row.
   *
   * \param out        [rows, kHeads, 512] bf16, contiguous
   * \param q          [rows, kHeads, 576] e4m3, contiguous
   * \param kv         [num_slots, 576] e4m3 (or uint8) KV rows, row stride 576 B
   * \param indices    [rows, >= 2048] int32 global slots, unit inner stride
   * \param seq_lens   [>= rows] int32 causal KV length of each row
   * \param scale_log2 softmax scale times the dequantization scales times log2(e)
   * \param out_scale  scale applied to the normalized output
   * \param smem       dynamic shared memory per CTA, >= `smem_bytes()`
   */
  static void
  run(tvm::ffi::TensorView out,
      tvm::ffi::TensorView q,
      tvm::ffi::TensorView kv,
      tvm::ffi::TensorView indices,
      tvm::ffi::TensorView seq_lens,
      double scale_log2,
      double out_scale,
      int64_t smem) {
    static_assert(!kFuseVb, "the v_b fold instance launches through run_fold");
    using namespace host;
    namespace fd = glm_ll_sparse_mla_fd;
    auto R = SymbolicSize{"rows"};
    auto device = SymbolicDevice{};
    device.set_options<kDLCUDA>();
    TensorMatcher({R, kHeads, fd::kHeadDimV})  //
        .with_dtype<bf16_t>()
        .with_device(device)
        .verify(out);
    launch(q, kv, indices, seq_lens, R, device, out.data_ptr(), 0, nullptr, scale_log2, out_scale, smem);
  }

  /**
   * \brief `run` with the v_b fold: every CTA also writes its 16 columns of every head's
   * v_b product, out[r, h * 256 + v] = bf16(sum_d bf16(O[r, h, d]) * vb_proj[h, v, d]).
   *
   * \param out     [rows, >= kHeads * 256] bf16, unit inner stride: the o_proj input
   * \param vb_proj [kHeads, 256, 512] bf16 W_UV, contiguous, 16 B aligned
   * \param smem    dynamic shared memory per CTA, >= `smem_bytes()` (one CTA per SM)
   * Other parameters as `run`.
   */
  static void run_fold(
      tvm::ffi::TensorView out,
      tvm::ffi::TensorView q,
      tvm::ffi::TensorView kv,
      tvm::ffi::TensorView indices,
      tvm::ffi::TensorView seq_lens,
      tvm::ffi::TensorView vb_proj,
      double scale_log2,
      double out_scale,
      int64_t smem) {
    static_assert(kFuseVb && kHeads == 8, "run_fold is the 8-head v_b fold instance");
    using namespace host;
    namespace fd = glm_ll_sparse_mla_fd;
    auto R = SymbolicSize{"rows"};
    auto OS = SymbolicSize{"out_row_stride"};
    auto OW = SymbolicSize{"out_width"};
    auto device = SymbolicDevice{};
    device.set_options<kDLCUDA>();
    TensorMatcher({R, OW})  //
        .with_strides({OS, 1})
        .with_dtype<bf16_t>()
        .with_device(device)
        .verify(out);
    TensorMatcher({kHeads, fd::kVHeadDim, fd::kHeadDimV})  //
        .with_strides({fd::kVHeadDim * fd::kHeadDimV, fd::kHeadDimV, 1})
        .with_dtype<bf16_t>()
        .with_device(device)
        .verify(vb_proj);
    CHECK_HOST(OW.unwrap() >= kHeads * fd::kVHeadDim) << "out width " << OW.unwrap() << " < " << kHeads * fd::kVHeadDim;
    CHECK_HOST(reinterpret_cast<uintptr_t>(vb_proj.data_ptr()) % 16 == 0) << "vb_proj must be 16 B aligned";
    launch(
        q,
        kv,
        indices,
        seq_lens,
        R,
        device,
        out.data_ptr(),
        OS.unwrap(),
        static_cast<const __nv_bfloat16*>(vb_proj.data_ptr()),
        scale_log2,
        out_scale,
        smem);
  }

 private:
  static void launch(
      tvm::ffi::TensorView q,
      tvm::ffi::TensorView kv,
      tvm::ffi::TensorView indices,
      tvm::ffi::TensorView seq_lens,
      host::SymbolicSize& R,
      host::SymbolicDevice& device,
      void* out,
      int64_t out_stride,
      const __nv_bfloat16* vb_proj,
      double scale_log2,
      double out_scale,
      int64_t smem) {
    using namespace host;
    namespace fd = glm_ll_sparse_mla_fd;
    auto S = SymbolicSize{"num_slots"};
    auto W = SymbolicSize{"indices_width"};
    auto IS = SymbolicSize{"indices_row_stride"};
    auto B = SymbolicSize{"seq_lens_len"};
    TensorMatcher({R, kHeads, fd::kHeadDimQk})  //
        .with_dtype<fp8_e4m3_t, uint8_t>()
        .with_device(device)
        .ensure_alignment(16)
        .verify(q);
    TensorMatcher({S, fd::kHeadDimQk})  //
        .with_dtype<fp8_e4m3_t, uint8_t>()
        .with_device(device)
        .ensure_alignment(16)
        .verify(kv);
    TensorMatcher({R, W})  //
        .with_strides({IS, 1})
        .with_dtype<int32_t>()
        .with_device(device)
        .verify(indices);
    TensorMatcher({B})  //
        .with_dtype<int32_t>()
        .with_device(device)
        .verify(seq_lens);

    const int64_t rows = R.unwrap();
    CHECK_HOST(W.unwrap() >= fd::kTopK) << "indices width " << W.unwrap() << " < " << fd::kTopK;
    CHECK_HOST(B.unwrap() >= rows) << "seq_lens holds " << B.unwrap() << " rows, need " << rows;
    CHECK_HOST(rows <= 65535) << "rows " << rows << " exceed gridDim.y";
    CHECK_HOST(smem >= kSmemBytes) << "smem " << smem << " below the kernel's " << kSmemBytes;
    if (rows == 0) return;
    ensure_smem_limit(device.unwrap().device_id, smem);

    const fd::Params params = {
        .q = static_cast<const uint8_t*>(q.data_ptr()),
        .kv = static_cast<const uint8_t*>(kv.data_ptr()),
        .indices = static_cast<const int32_t*>(indices.data_ptr()),
        .seq_lens = static_cast<const int32_t*>(seq_lens.data_ptr()),
        .out = static_cast<__nv_bfloat16*>(out),
        .indices_stride = IS.unwrap(),
        .num_slots = S.unwrap(),
        .scale_log2 = static_cast<float>(scale_log2),
        .out_scale = static_cast<float>(out_scale),
        .vb_proj = vb_proj,
        .out_stride = out_stride,
    };
    LaunchKernel(
        dim3(fd::kSplits, static_cast<uint32_t>(rows), 1),
        dim3(fd::kThreads, 1, 1),
        device.unwrap(),
        static_cast<std::size_t>(smem))
        .config({.use_pdl = kUsePDL, .cluster_dim = dim3(fd::kSplits, 1, 1)})(kKernel, params);
  }
};

}  // namespace sglang
