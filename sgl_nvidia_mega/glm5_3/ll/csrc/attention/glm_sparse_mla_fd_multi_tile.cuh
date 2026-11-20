// Sparse MLA flash decode, multi-tile mode: more query rows than one wave of the
// one-tile kernel's 16-CTA clusters holds (glm_sparse_mla_fd_common.cuh).
//
// Adapted from TensorRT-LLM (Apache-2.0):
//   cpp/tensorrt_llm/kernels/sparseMlaFlashDecode.cu (Dongfeng Yu; 2808056e, 9bea245e, f5052d41)
//   cpp/tensorrt_llm/kernels/glmLowLatency/w4/sparseMlaDecode.cu (Xiao Wang; GB300 hardening)
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "glm_sparse_mla_fd_common.cuh"

namespace sglang {

namespace glm_ll_sparse_mla_fd {

// ---------------------------------------------------------------------------
// Multi-tile mode, for more rows than one wave of 16-CTA clusters holds: query
// row r is one cluster of kSplits (2-8) CTAs. CTA s attends to the 128-position
// tiles s, s + kSplits, s + 2 kSplits, ... of the row (interleaved, so that a
// short row spreads its active tiles over the cluster), streams them through a
// two-stage bulk-gather ring with an online softmax, and pushes its partials once.
// QK and P.V run as e4m3 mma.sync (m16n8k32, FP32 accumulation; SM10x lowers it to
// e4m3 unpacks and FP16 HMMAs, which bound a tile at ~2.4k clocks at 8 heads): the query
// stays FP8 in registers, which measured faster than an FP16 query tile in shared memory.
// The output dims merge in 16-dim blocks: block b belongs to rank b % kSplits.
// ---------------------------------------------------------------------------
inline constexpr int kTilesPerRow = kTopK / kKeysPerCta;
inline constexpr int kDimBlock = 16;
inline constexpr int kDimBlocks = kHeadDimV / kDimBlock;
inline constexpr int kMtStages = 2;
// Byte pitches of the e4m3 query and probability tiles: rows g = 0..7 of a 32-bit
// fragment load start 20 g and 4 g banks apart, so the eight rows never collide.
inline constexpr int kQRowBytes = kHeadDimQk + 16;
inline constexpr int kPRowBytes = kKeysPerCta + 16;
// P of a tile is rounded against max(tile maximum, running maximum - kRefWindow)
// (log2 units): the one-tile kernel's per-tile reference, with the accumulators'
// rescaling bounded by 2^kRefWindow. A tile that far below the maximum weighs < 2^-20.
inline constexpr float kRefWindow = 20.f;

static_assert(kDimsPerWarp == 4 * kDimBlock, "each P.V warp produces four 16-dim blocks");

/// \brief Shared-memory layout of one multi-tile CTA.
template <int kHeads, int kSplits>
struct MtLayout {
  static_assert(kHeads == 8 || kHeads == 16, "8 or 16 query heads (one or two n8 MMA tiles)");
  static_assert(kSplits >= 2 && kSplits <= 8, "portable clusters of 2-8 CTAs");
  static constexpr int kHeadTiles = kHeads / 8;
  // Tiles of CTA 0 (the most of any rank) and output blocks of rank 0.
  static constexpr int kTiles = (kTilesPerRow + kSplits - 1) / kSplits;
  static constexpr int kLocalBlocks = (kDimBlocks + kSplits - 1) / kSplits;
  static constexpr int kStageBytes = kKeysPerCta * kKRowBytes;
  static constexpr int kRing = 0;
  // e4m3 query [head][576], e4m3 probabilities [head][128 keys in P.V k order].
  static constexpr int kQTile = kRing + kMtStages * kStageBytes;
  static constexpr int kPTile = kQTile + kHeads * kQRowBytes;
  // [source split][head][local block][16 dims] FP32 partials.
  static constexpr int kRecv = kPTile + kHeads * kPRowBytes;
  // [maxima of the heads, then sums of the heads][source split] FP32.
  static constexpr int kRecvStats = kRecv + kSplits * kHeads * kLocalBlocks * kDimBlock * 4;
  static constexpr int kRedMax = kRecvStats + 2 * kHeads * kSplits * 4;
  static constexpr int kRedSum = kRedMax + kWarps * kHeads * 4;
  // This CTA's final (max, sum) of the heads, then the merge weights [head][source split].
  static constexpr int kStats = kRedSum + kWarps * kHeads * 4;
  static constexpr int kWeights = kStats + 2 * kHeads * 4;
  // [tile][position] usable slot or -1, [tile][warp] ballot of usable positions, active-tile mask.
  static constexpr int kSlots = kWeights + kHeads * kSplits * 4;
  static constexpr int kValid = kSlots + kTiles * kKeysPerCta * 4;
  static constexpr int kActive = kValid + kTiles * kWarps * 4;
  // mbarriers: [stage][warp] row gathers, then the pushed partials.
  static constexpr int kBarriers = (kActive + 4 + 7) / 8 * 8;
  static constexpr int kBytes = kBarriers + (kMtStages * kWarps + 1) * 8;
  static constexpr int kQChunks = kHeads * kHeadDimQk / 16;
  static constexpr int kQChunksPerThread = (kQChunks + kThreads - 1) / kThreads;

  static_assert(kQRowBytes / 4 % 32 == 20 && kPRowBytes / 4 % 32 == 4, "bank-conflict-free pitches");
  static_assert(kQTile % 16 == 0 && kPTile % 16 == 0 && kRecv % 16 == 0, "16 B aligned tiles");
  static_assert(kBytes <= 227 * 1024, "shared memory per CTA");

  SGL_DEVICE_HOST static constexpr int local_blocks(int rank) {
    return (kDimBlocks - rank + kSplits - 1) / kSplits;
  }
  // Every CTA of the cluster sends rank `rank` its statistics and its partials of the rank's blocks.
  SGL_DEVICE_HOST static constexpr uint32_t recv_bytes(int rank) {
    return static_cast<uint32_t>(kSplits * (kHeads * local_blocks(rank) * kDimBlock + 2 * kHeads) * 4);
  }
};

/// \brief Kernel arguments of the multi-tile mode.
struct MtParams {
  const uint8_t* q;
  const uint8_t* kv;
  const int32_t* indices;
  const int32_t* seq_lens;
  __nv_bfloat16* out;
  int64_t indices_stride;
  int64_t num_slots;
  float scale_log2;
  float out_scale;
};

#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)

// D += A * B, A 16x32 e4m3 (row), B 32x8 e4m3 (col), FP32 accumulation.
SGL_DEVICE void mma_e4m3(float (&d)[4], uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3, uint32_t b0, uint32_t b1) {
  asm volatile(
      "mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, "
      "{%0, %1, %2, %3};\n"
      : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
      : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}

// e4m3(448 * p), the probability rounding of the TRTLLM-Gen generation kernels.
SGL_DEVICE uint8_t probability_e4m3(float p) {
  return __nv_cvt_float_to_fp8(p * kProbScale, __NV_SATFINITE, __NV_E4M3);
}

// P.V k order of key j of a 32-key block: thread t of a transposed ldmatrix receives keys
// 2t, 2t + 1 of each 8-key matrix, so k slots 4t..4t+3 hold keys 2t, 2t+1, 2t+8, 2t+9.
SGL_DEVICE int pv_k_slot(int key) {
  const int matrix = key >> 3;
  const int r = key & 7;
  return (matrix >> 1) * 16 + (r >> 1) * 4 + (matrix & 1) * 2 + (r & 1);
}

#endif  // defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)

/**
 * \brief Sparse MLA decode of one query row per cluster of kSplits CTAs, each CTA
 * streaming 16 / kSplits (rounded up) tiles of 128 positions.
 *
 * Semantics are those of `glm_sparse_mla_fd_kernel` (scores from exact e4m3
 * products accumulated in FP32, P rounded to e4m3(448 P), FP32 softmax and
 * merge), with P of each tile rounded against max(tile maximum, running maximum
 * - kRefWindow) instead of the 128-key CTA maximum. PDL and DSMEM ownership as
 * there: nothing global before the wait, dependents triggered once the first
 * gathers are issued, `out` is the only global write, and no CTA leaves before
 * its receive barrier has every peer's bytes.
 *
 * \tparam kHeads  Query heads per row: 8 or 16
 * \tparam kSplits CTAs per row (cluster size), 2-8
 * \tparam kUsePDL Emit the PDL wait and trigger
 */
template <int kHeads, int kSplits, bool kUsePDL>
__global__ void __launch_bounds__(kThreads, 1) glm_sparse_mla_fd_mt_kernel(const __grid_constant__ MtParams params) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
  using L = MtLayout<kHeads, kSplits>;
  constexpr int kHeadTiles = L::kHeadTiles;
  constexpr int kTiles = L::kTiles;
  // QK accumulator chains per head tile (alternating k steps) to hide the MMA latency.
  constexpr int kQkChains = kHeads == 8 ? 4 : 2;
  extern __shared__ __align__(128) uint8_t smem[];

  const int tid = threadIdx.x;
  const int warp = tid >> 5;
  const int lane = tid & 31;
  const int group = lane >> 2;
  const int tig = lane & 3;
  // The cluster spans gridDim.x, so the split is the CTA's rank in the cluster.
  const int split = blockIdx.x;
  const int row = blockIdx.y;
  const int num_tiles = (kTilesPerRow - split + kSplits - 1) / kSplits;

  uint8_t* ring = smem + L::kRing;
  uint8_t* q_tile = smem + L::kQTile;
  uint8_t* p_tile = smem + L::kPTile;
  float* recv = reinterpret_cast<float*>(smem + L::kRecv);
  float* recv_stats = reinterpret_cast<float*>(smem + L::kRecvStats);
  float* red_max = reinterpret_cast<float*>(smem + L::kRedMax);
  float* red_sum = reinterpret_cast<float*>(smem + L::kRedSum);
  float* stats = reinterpret_cast<float*>(smem + L::kStats);
  float* weights = reinterpret_cast<float*>(smem + L::kWeights);
  int32_t* slot_tab = reinterpret_cast<int32_t*>(smem + L::kSlots);
  uint32_t* valid_tab = reinterpret_cast<uint32_t*>(smem + L::kValid);
  uint32_t* active_word = reinterpret_cast<uint32_t*>(smem + L::kActive);
  const uint32_t bar_base = smem_addr(smem + L::kBarriers);
  const uint32_t recv_bar = bar_base + kMtStages * kWarps * 8;
  const auto gather_bar = [&](int stage) { return bar_base + (stage * kWarps + warp) * 8; };

  if (lane == 0) {
#pragma unroll
    for (int stage = 0; stage < kMtStages; ++stage) {
      mbar_init(gather_bar(stage), kKeysPerWarp);
    }
    if (warp == 0) {
      mbar_init(recv_bar, 1);
      *active_word = 0U;
    }
    fence_mbar_init();
    if (warp == 0) {
      mbar_arrive_expect_tx(recv_bar, L::recv_bytes(split));
    }
  }
  __syncthreads();
  cluster_arrive_relaxed();

  const uint8_t* q_row = params.q + static_cast<int64_t>(row) * kHeads * kHeadDimQk;
  uint4 q_raw[L::kQChunksPerThread];
  const auto load_q = [&]() {
#pragma unroll
    for (int i = 0; i < L::kQChunksPerThread; ++i) {
      const int chunk = tid + i * kThreads;
      if (chunk < L::kQChunks) {
        q_raw[i] = ld_global_v4(q_row + chunk * 16);
      }
    }
  };

  // Lane l < 16 of warp w owns position 16 w + l of each of this CTA's tiles.
  const auto build_index_table = [&]() {
    const int32_t* idx_row = params.indices + static_cast<int64_t>(row) * params.indices_stride;
    int32_t slots[kTiles];
#pragma unroll
    for (int i = 0; i < kTiles; ++i) slots[i] = -1;
    int32_t seq_len = 0;
    if (lane < kKeysPerWarp) {
      seq_len = ld_global_s32(params.seq_lens + row);
#pragma unroll
      for (int i = 0; i < kTiles; ++i) {
        if (i < num_tiles) {
          slots[i] = ld_global_s32(idx_row + (split + i * kSplits) * kKeysPerCta + warp * kKeysPerWarp + lane);
        }
      }
    }
#pragma unroll
    for (int i = 0; i < kTiles; ++i) {
      const int key = (split + i * kSplits) * kKeysPerCta + warp * kKeysPerWarp + lane;
      const bool valid = lane < kKeysPerWarp && i < num_tiles && key < seq_len && slots[i] >= 0 &&
                         static_cast<int64_t>(slots[i]) < params.num_slots;
      const uint32_t bits = __ballot_sync(0xffffffffU, valid);
      if (lane < kKeysPerWarp && i < num_tiles) {
        slot_tab[i * kKeysPerCta + warp * kKeysPerWarp + lane] = valid ? slots[i] : -1;
      }
      if (lane == 0 && i < num_tiles) {
        valid_tab[i * kWarps + warp] = bits;
        if (bits != 0U) atomicOr(active_word, 1U << i);
      }
    }
  };

  device::PDLWaitPrimary<kUsePDL>();
  load_q();
  build_index_table();
  __syncthreads();
  const uint32_t active = *active_word;

  // Gather tile `tile` into `stage`: one bulk copy per usable row; masked rows of an active
  // tile are zeroed so that zero probabilities never meet non-finite bytes in P.V. Lane l
  // of warp w always owns row 16 w + l, so its own proxy fence orders its zero-fill before
  // a later bulk copy into the same row.
  const auto issue_row = [&](int32_t slot, int stage) {
    uint8_t* dst = ring + stage * L::kStageBytes + (warp * kKeysPerWarp + lane) * kKRowBytes;
    const uint32_t bar = gather_bar(stage);
    mbar_arrive_expect_tx(bar, kHeadDimQk);
    bulk_copy_g2s(smem_addr(dst), params.kv + static_cast<int64_t>(slot) * kHeadDimQk, kHeadDimQk, bar);
  };
  const auto issue_gather = [&](int tile, int stage) {
    const int32_t slot = lane < kKeysPerWarp ? slot_tab[tile * kKeysPerCta + warp * kKeysPerWarp + lane] : -1;
    if (lane < kKeysPerWarp) {
      if (slot >= 0) {
        issue_row(slot, stage);
      } else {
        uint8_t* dst = ring + stage * L::kStageBytes + (warp * kKeysPerWarp + lane) * kKRowBytes;
#pragma unroll 4
        for (int c = 0; c < kHeadDimQk / 16; ++c) {
          *reinterpret_cast<uint4*>(dst + c * 16) = make_uint4(0U, 0U, 0U, 0U);
        }
        mbar_arrive(gather_bar(stage));
      }
    }
  };
  uint32_t to_issue = active;
#pragma unroll
  for (int stage = 0; stage < kMtStages; ++stage) {
    if (to_issue != 0U) {
      issue_gather(__ffs(to_issue) - 1, stage);
      to_issue &= to_issue - 1;
    }
  }

  // e4m3 copy of the query in shared memory, then this thread's B fragments of every k step.
#pragma unroll
  for (int i = 0; i < L::kQChunksPerThread; ++i) {
    const int chunk = tid + i * kThreads;
    if (chunk < L::kQChunks) {
      const int head = chunk / (kHeadDimQk / 16);
      const int col = (chunk - head * (kHeadDimQk / 16)) * 16;
      *reinterpret_cast<uint4*>(q_tile + head * kQRowBytes + col) = q_raw[i];
    }
  }

  // Every global input is read or in flight once the first gathers are issued.
  device::PDLTriggerSecondary<kUsePDL>();
  __syncthreads();
  uint32_t q_frag[kHeadTiles][kQkSteps][2];
#pragma unroll
  for (int ht = 0; ht < kHeadTiles; ++ht) {
    const uint8_t* q_head = q_tile + (ht * 8 + group) * kQRowBytes + 4 * tig;
#pragma unroll
    for (int kk = 0; kk < kQkSteps; ++kk) {
      q_frag[ht][kk][0] = *reinterpret_cast<const uint32_t*>(q_head + kk * 32);
      q_frag[ht][kk][1] = *reinterpret_cast<const uint32_t*>(q_head + kk * 32 + 16);
    }
  }

  // Online softmax state of this thread's heads ht * 8 + 2 tig (+1): the running maximum,
  // the reference the accumulators and sums are scaled to, the sum; and the P.V
  // accumulators O^T[dims of block 4 warp + mt][heads of tile ht].
  float m_run[kHeadTiles][2];
  float m_ref[kHeadTiles][2];
  float l_run[kHeadTiles][2];
  float o_acc[4][kHeadTiles][4];
#pragma unroll
  for (int ht = 0; ht < kHeadTiles; ++ht) {
#pragma unroll
    for (int i = 0; i < 2; ++i) {
      m_run[ht][i] = -INFINITY;
      m_ref[ht][i] = -INFINITY;
      l_run[ht][i] = 0.f;
    }
#pragma unroll
    for (int mt = 0; mt < 4; ++mt) {
#pragma unroll
      for (int i = 0; i < 4; ++i) o_acc[mt][ht][i] = 0.f;
    }
  }

  // P.V k slots of this thread's keys 16 warp + group (+8) in the probability tile.
  const int p_col = (warp >> 1) * 32 + pv_k_slot((warp & 1) * 16 + group);
  const int p_col_hi = (warp >> 1) * 32 + pv_k_slot((warp & 1) * 16 + group + 8);

  uint32_t pending = active;
  for (int a = 0; pending != 0U; ++a) {
    const int tile = __ffs(pending) - 1;
    pending &= pending - 1;
    const int stage = a & 1;
    const uint8_t* k_tile = ring + stage * L::kStageBytes;
    mbar_wait(gather_bar(stage), static_cast<uint32_t>(a >> 1) & 1U);
    __syncwarp();

    // S^T = K Q^T for this warp's 16 keys: keys on M, heads on N. The ldmatrix registers
    // are the e4m3 A fragment as loaded (rows g / g + 8, k bytes 4 tig.. and 16 + 4 tig..).
    float s_acc[kHeadTiles][4];
    {
      float acc[kHeadTiles][kQkChains][4] = {};
      const uint32_t a_addr = smem_addr(k_tile + (warp * kKeysPerWarp + (lane & 15)) * kKRowBytes + (lane >> 4) * 16);
#pragma unroll
      for (int kk = 0; kk < kQkSteps; ++kk) {
        uint32_t r[4];
        ldmatrix_x4(r, a_addr + kk * 32);
#pragma unroll
        for (int ht = 0; ht < kHeadTiles; ++ht) {
          mma_e4m3(acc[ht][kk % kQkChains], r[0], r[1], r[2], r[3], q_frag[ht][kk][0], q_frag[ht][kk][1]);
        }
      }
#pragma unroll
      for (int ht = 0; ht < kHeadTiles; ++ht) {
#pragma unroll
        for (int i = 0; i < 4; ++i) {
          float sum = acc[ht][0][i];
#pragma unroll
          for (int c = 1; c < kQkChains; ++c) sum += acc[ht][c][i];
          s_acc[ht][i] = sum;
        }
      }
    }

    const uint32_t valid_bits = valid_tab[tile * kWarps + warp];
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

    // An active tile has a usable position, so every head's tile maximum is finite.
    float alpha[kHeadTiles][2];
#pragma unroll
    for (int ht = 0; ht < kHeadTiles; ++ht) {
      float ref[2];
#pragma unroll
      for (int i = 0; i < 2; ++i) {
        float tile_max = -INFINITY;
#pragma unroll
        for (int w = 0; w < kWarps; ++w) {
          tile_max = fmaxf(tile_max, red_max[w * kHeads + ht * 8 + 2 * tig + i]);
        }
        m_run[ht][i] = fmaxf(m_run[ht][i], tile_max);
        ref[i] = fmaxf(tile_max, m_run[ht][i] - kRefWindow);
        alpha[ht][i] = m_ref[ht][i] == -INFINITY ? 0.f : fast_exp2(m_ref[ht][i] - ref[i]);
        m_ref[ht][i] = ref[i];
      }
      float p[4];
      p[0] = s[ht][0] == -INFINITY ? 0.f : fast_exp2(s[ht][0] - ref[0]);
      p[1] = s[ht][1] == -INFINITY ? 0.f : fast_exp2(s[ht][1] - ref[1]);
      p[2] = s[ht][2] == -INFINITY ? 0.f : fast_exp2(s[ht][2] - ref[0]);
      p[3] = s[ht][3] == -INFINITY ? 0.f : fast_exp2(s[ht][3] - ref[1]);
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
      uint8_t* p_head = p_tile + (ht * 8 + 2 * tig) * kPRowBytes;
      p_head[p_col] = probability_e4m3(p[0]);
      p_head[kPRowBytes + p_col] = probability_e4m3(p[1]);
      p_head[p_col_hi] = probability_e4m3(p[2]);
      p_head[kPRowBytes + p_col_hi] = probability_e4m3(p[3]);
    }
    // Orders all key rows of the stage and the probabilities before the P.V reads of the other warps.
    __syncthreads();

#pragma unroll
    for (int ht = 0; ht < kHeadTiles; ++ht) {
#pragma unroll
      for (int i = 0; i < 2; ++i) {
        float tile_sum = 0.f;
#pragma unroll
        for (int w = 0; w < kWarps; ++w) tile_sum += red_sum[w * kHeads + ht * 8 + 2 * tig + i];
        l_run[ht][i] = l_run[ht][i] * alpha[ht][i] + tile_sum;
      }
#pragma unroll
      for (int mt = 0; mt < 4; ++mt) {
        o_acc[mt][ht][0] *= alpha[ht][0];
        o_acc[mt][ht][2] *= alpha[ht][0];
        o_acc[mt][ht][1] *= alpha[ht][1];
        o_acc[mt][ht][3] *= alpha[ht][1];
      }
    }

    // O^T += V^T P^T for this warp's 64 output dims. A transposed b16 ldmatrix of 32 keys
    // x 16 dims gives thread (g, t) keys 2t, 2t + 1 of each 8-key matrix x dims 2g, 2g + 1;
    // byte permutes split the dims (M row g = dim 2g, row g + 8 = dim 2g + 1) and P is
    // stored in the matching k order (pv_k_slot).
    {
      uint32_t p_frag[kHeadTiles][kPvSteps][2];
#pragma unroll
      for (int ht = 0; ht < kHeadTiles; ++ht) {
        const uint8_t* p_row = p_tile + (ht * 8 + group) * kPRowBytes + 4 * tig;
#pragma unroll
        for (int ks = 0; ks < kPvSteps; ++ks) {
          p_frag[ht][ks][0] = *reinterpret_cast<const uint32_t*>(p_row + ks * 32);
          p_frag[ht][ks][1] = *reinterpret_cast<const uint32_t*>(p_row + ks * 32 + 16);
        }
      }
      const uint32_t v_addr = smem_addr(k_tile + lane * kKRowBytes + warp * kDimsPerWarp);
#pragma unroll
      for (int ks = 0; ks < kPvSteps; ++ks) {
#pragma unroll
        for (int mt = 0; mt < 4; ++mt) {
          uint32_t r[4];
          ldmatrix_x4_trans(r, v_addr + ks * 32 * kKRowBytes + mt * 16);
          const uint32_t a0 = __byte_perm(r[0], r[1], 0x6420);
          const uint32_t a1 = __byte_perm(r[0], r[1], 0x7531);
          const uint32_t a2 = __byte_perm(r[2], r[3], 0x6420);
          const uint32_t a3 = __byte_perm(r[2], r[3], 0x7531);
#pragma unroll
          for (int ht = 0; ht < kHeadTiles; ++ht) {
            mma_e4m3(o_acc[mt][ht], a0, a1, a2, a3, p_frag[ht][ks][0], p_frag[ht][ks][1]);
          }
        }
      }
    }

    if (to_issue != 0U) {
      // Every warp is done with this stage (QK rows and P.V columns) before it is refilled.
      __syncthreads();
      fence_proxy_async_smem();
      issue_gather(__ffs(to_issue) - 1, stage);
      to_issue &= to_issue - 1;
    }
  }

  // Lanes 0-3 of every warp hold the statistics of heads ht * 8 + 2 tig (+1).
  if (warp == 0 && lane < 4) {
#pragma unroll
    for (int ht = 0; ht < kHeadTiles; ++ht) {
#pragma unroll
      for (int i = 0; i < 2; ++i) {
        stats[ht * 8 + 2 * tig + i] = m_ref[ht][i];
        stats[kHeads + ht * 8 + 2 * tig + i] = l_run[ht][i];
      }
    }
  }
  __syncthreads();

  // Every peer has initialized its barriers once the cluster barrier completes.
  cluster_wait();

  // Statistics to every CTA of the cluster: value t / kSplits (references, then sums) to rank t % kSplits.
  for (int item = tid; item < 2 * kHeads * kSplits; item += kThreads) {
    const int index = item / kSplits;
    const int dest = item - index * kSplits;
    store_async(
        map_to_rank(smem_addr(recv_stats + index * kSplits + split), dest), stats[index], map_to_rank(recv_bar, dest));
  }
  {
    constexpr float kInvProbScale = 1.f / kProbScale;
    const uint32_t recv_base = smem_addr(recv);
#pragma unroll
    for (int mt = 0; mt < 4; ++mt) {
      const int block = warp * 4 + mt;
      const uint32_t dest = static_cast<uint32_t>(block % kSplits);
      const int local_block = block / kSplits;
      const uint32_t remote_bar = map_to_rank(recv_bar, dest);
#pragma unroll
      for (int ht = 0; ht < kHeadTiles; ++ht) {
        const int head = ht * 8 + 2 * tig;
        const uint32_t addr_a =
            recv_base +
            (((split * kHeads + head) * L::kLocalBlocks + local_block) * kDimBlock + 2 * group) * sizeof(float);
        const uint32_t addr_b = addr_a + L::kLocalBlocks * kDimBlock * sizeof(float);
        store_async2(
            map_to_rank(addr_a, dest), o_acc[mt][ht][0] * kInvProbScale, o_acc[mt][ht][2] * kInvProbScale, remote_bar);
        store_async2(
            map_to_rank(addr_b, dest), o_acc[mt][ht][1] * kInvProbScale, o_acc[mt][ht][3] * kInvProbScale, remote_bar);
      }
    }
  }

  // No CTA leaves before its own receive barrier completes, which needs every
  // peer's bytes, so no st.async targets an exited CTA.
  mbar_wait_cluster(recv_bar, 0);

  if (tid < kHeads) {
    float global_ref = -INFINITY;
#pragma unroll
    for (int i = 0; i < kSplits; ++i) global_ref = fmaxf(global_ref, recv_stats[tid * kSplits + i]);
    float w[kSplits];
    float sum = 0.f;
#pragma unroll
    for (int i = 0; i < kSplits; ++i) {
      const float peer_ref = recv_stats[tid * kSplits + i];
      w[i] = peer_ref == -INFINITY ? 0.f : fast_exp2(peer_ref - global_ref);
      sum += w[i] * recv_stats[(kHeads + tid) * kSplits + i];
    }
    const float scale = sum > 0.f ? params.out_scale / sum : 0.f;
#pragma unroll
    for (int i = 0; i < kSplits; ++i) weights[tid * kSplits + i] = w[i] * scale;
  }
  __syncthreads();

  const int local_blocks = L::local_blocks(split);
  for (int item = tid; item < kHeads * local_blocks * kDimBlock; item += kThreads) {
    const int dim = item & (kDimBlock - 1);
    const int head_block = item / kDimBlock;
    const int head = head_block / local_blocks;
    const int local_block = head_block - head * local_blocks;
    float acc = 0.f;
#pragma unroll
    for (int i = 0; i < kSplits; ++i) {
      acc += weights[head * kSplits + i] *
             recv[((i * kHeads + head) * L::kLocalBlocks + local_block) * kDimBlock + dim];
    }
    params.out
        [(static_cast<int64_t>(row) * kHeads + head) * kHeadDimV + (local_block * kSplits + split) * kDimBlock + dim] =
        __float2bfloat16_rn(acc);
  }
#else
  __trap();
#endif
}

}  // namespace glm_ll_sparse_mla_fd

/**
 * \brief Host launcher of the multi-tile mode (`glm_sparse_mla_fd_mt_kernel`) for one head
 * count, with every supported cluster size in one module.
 *
 * \tparam kHeads  Query heads per row: 8 or 16
 * \tparam kUsePDL Launch with programmatic stream serialization and emit the PDL wait/trigger
 */
template <int kHeads, bool kUsePDL>
struct GlmLlSparseMlaFdMultiTile {
  template <int kSplits>
  static constexpr auto kKernel = glm_ll_sparse_mla_fd::glm_sparse_mla_fd_mt_kernel<kHeads, kSplits, kUsePDL>;

  // Tiles per CTA 8, 6, 4, 3 and 2; 5 and 7 CTAs would need as many tiles as 4 and 6.
  template <typename F>
  static auto with_splits(int64_t splits, F&& f) {
    using namespace host;
    switch (splits) {
      case 2:
        return f(std::integral_constant<int, 2>{});
      case 3:
        return f(std::integral_constant<int, 3>{});
      case 4:
        return f(std::integral_constant<int, 4>{});
      case 6:
        return f(std::integral_constant<int, 6>{});
      case 8:
        return f(std::integral_constant<int, 8>{});
      default:
        panic(DebugInfo{}, "sparse MLA flash decode: unsupported split count ", splits);
    }
  }

  /// \brief Dynamic shared memory of the `splits`-CTA instance.
  static int64_t smem_bytes(int64_t splits) {
    return with_splits(
        splits, [](auto s) -> int64_t { return glm_ll_sparse_mla_fd::MtLayout<kHeads, decltype(s)::value>::kBytes; });
  }

  template <int kSplits>
  static void ensure_smem_limit(int device_id) {
    static std::mutex mutex;
    static std::map<int, bool> configured;
    const std::lock_guard<std::mutex> lock(mutex);
    if (configured[device_id]) return;
    tvm::ffi::CUDADeviceGuard guard(device_id);
    CHECK_CUDA(cudaFuncSetAttribute(
        kKernel<kSplits>,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        glm_ll_sparse_mla_fd::MtLayout<kHeads, kSplits>::kBytes));
    configured[device_id] = true;
  }

  /// \brief Clusters of `splits` CTAs of this kernel resident at once on the current device.
  static int64_t max_active_clusters(int64_t splits) {
    return with_splits(splits, [](auto s) -> int64_t {
      constexpr int kSplits = decltype(s)::value;
      int device = 0;
      CHECK_CUDA(cudaGetDevice(&device));
      ensure_smem_limit<kSplits>(device);
      cudaLaunchConfig_t config = {};
      config.gridDim = dim3(kSplits, 1, 1);
      config.blockDim = dim3(glm_ll_sparse_mla_fd::kThreads, 1, 1);
      config.dynamicSmemBytes = glm_ll_sparse_mla_fd::MtLayout<kHeads, kSplits>::kBytes;
      cudaLaunchAttribute attr = {};
      attr.id = cudaLaunchAttributeClusterDimension;
      attr.val.clusterDim.x = kSplits;
      attr.val.clusterDim.y = 1;
      attr.val.clusterDim.z = 1;
      config.attrs = &attr;
      config.numAttrs = 1;
      int num_clusters = 0;
      CHECK_CUDA(cudaOccupancyMaxActiveClusters(&num_clusters, kKernel<kSplits>, &config));
      return num_clusters;
    });
  }

  /**
   * \brief Launch one cluster of `splits` CTAs per query row.
   *
   * Arguments as `GlmLlSparseMlaFdOneTile::run`, with `splits` in {2, 3, 4, 6, 8} instead of
   * the shared-memory size.
   */
  static void
  run(tvm::ffi::TensorView out,
      tvm::ffi::TensorView q,
      tvm::ffi::TensorView kv,
      tvm::ffi::TensorView indices,
      tvm::ffi::TensorView seq_lens,
      double scale_log2,
      double out_scale,
      int64_t splits) {
    using namespace host;
    namespace fd = glm_ll_sparse_mla_fd;
    auto R = SymbolicSize{"rows"};
    auto S = SymbolicSize{"num_slots"};
    auto W = SymbolicSize{"indices_width"};
    auto IS = SymbolicSize{"indices_row_stride"};
    auto B = SymbolicSize{"seq_lens_len"};
    auto device = SymbolicDevice{};
    device.set_options<kDLCUDA>();
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
    TensorMatcher({R, kHeads, fd::kHeadDimV})  //
        .with_dtype<bf16_t>()
        .with_device(device)
        .verify(out);

    const int64_t rows = R.unwrap();
    CHECK_HOST(W.unwrap() >= fd::kTopK) << "indices width " << W.unwrap() << " < " << fd::kTopK;
    CHECK_HOST(B.unwrap() >= rows) << "seq_lens holds " << B.unwrap() << " rows, need " << rows;
    CHECK_HOST(rows <= 65535) << "rows " << rows << " exceed gridDim.y";
    if (rows == 0) return;

    const fd::MtParams params = {
        .q = static_cast<const uint8_t*>(q.data_ptr()),
        .kv = static_cast<const uint8_t*>(kv.data_ptr()),
        .indices = static_cast<const int32_t*>(indices.data_ptr()),
        .seq_lens = static_cast<const int32_t*>(seq_lens.data_ptr()),
        .out = static_cast<__nv_bfloat16*>(out.data_ptr()),
        .indices_stride = IS.unwrap(),
        .num_slots = S.unwrap(),
        .scale_log2 = static_cast<float>(scale_log2),
        .out_scale = static_cast<float>(out_scale),
    };
    const DLDevice dl_device = device.unwrap();
    with_splits(splits, [&](auto s) {
      constexpr int kSplits = decltype(s)::value;
      ensure_smem_limit<kSplits>(dl_device.device_id);
      LaunchKernel(
          dim3(kSplits, static_cast<uint32_t>(rows), 1),
          dim3(fd::kThreads, 1, 1),
          dl_device,
          static_cast<std::size_t>(fd::MtLayout<kHeads, kSplits>::kBytes))
          .config({.use_pdl = kUsePDL, .cluster_dim = dim3(kSplits, 1, 1)})(kKernel<kSplits>, params);
    });
  }
};

}  // namespace sglang
