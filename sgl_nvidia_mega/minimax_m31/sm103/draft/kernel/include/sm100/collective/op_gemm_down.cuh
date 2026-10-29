#pragma once

// OP gemm_down: down: [7, 3072] x [6144, 3072]^T, pushes this rank's TP4 partial (bf16 pairs) to
// every rank.

#include "include/sm100/collective/op_gemm_common.cuh"

namespace m31_draft {
namespace {

// Schedule: units of 16 weight rows x the full K (one 16-row box),
// stages of 16 rows x 2048 K bytes (the last: 1024), stored as 128-byte swizzled K blocks.
struct GemmDown {
  static constexpr int kId = kGemmDown;
  __host__ __device__ static constexpr int units() { return gemm_n(kId) / kTallRows; }
  __device__ static int stages(int) { return (gemm_k(kId) + kStageRow - 1) / kStageRow; }
  // The consumers of this GEMM's output gate on token 0's first word of the last unit.
  __device__ static int gate_word(int grid) {
    return last_gemm_unit(units(), grid) * (kTallRows / 2);
  }
  // Producer side: issue stage st of unit u (lane 0 of the weight-stream warp).
  __device__ static void issue(LayerArgs const &args, WeightMaps const &maps, int layer, int u,
                               int st, uint32_t dst, uint32_t full, uint64_t policy) {
    int const y0 = u * kTallRows;
    int const k = gemm_k(kId), ksf = k / kScaleBlock;
    int const kc = k - st * kStageRow < kStageRow ? k - st * kStageRow : kStageRow;
    mbar_expect_tx(full, uint32_t(kTallRows * kc + kTallRows * ksf));
    int const kb = st * (kStageRow / kKBlock);
    uint8_t const *sf = args.down_sf;
    int const map = kc == kStageRow ? 0 : 1; // the full box, or the 8-block tail box
    tma_load_3d(dst, &maps.map[layer][kId][map], 0, y0, kb, full, policy);
    bulk_load(dst + kSlotW, sf + size_t(y0) * ksf, uint32_t(kTallRows * ksf), full, policy);
  }
  // Host side: this GEMM's tensor maps of one layer (dims (128 bytes, rows, K / 128 blocks),
  // byte strides (K, 128), 128-byte swizzle).
  static bool encode(PFN_cuTensorMapEncodeTiled_v12000 encode, LayerArgs const &args, int layer,
                     WeightMaps *maps) {
    cuuint64_t const dims[3] = {cuuint64_t(kKBlock), cuuint64_t(kHidden),
                                cuuint64_t(kInter / kKBlock)};
    cuuint64_t const strides[2] = {cuuint64_t(kInter), cuuint64_t(kKBlock)};
    cuuint32_t const elements[3] = {1, 1, 1};
    {
      cuuint32_t const box[3] = {cuuint32_t(kKBlock), cuuint32_t(kTallRows),
                                 cuuint32_t(kStageRow / kKBlock)};
      if (encode(&maps->map[layer][kId][0], CU_TENSOR_MAP_DATA_TYPE_UINT8, 3,
                 const_cast<uint8_t *>(args.down), dims, strides, box, elements,
                 CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B,
                 CU_TENSOR_MAP_L2_PROMOTION_L2_256B,
                 CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE) != CUDA_SUCCESS)
        return false;
    }
    {
      cuuint32_t const box[3] = {cuuint32_t(kKBlock), cuuint32_t(kTallRows),
                                 cuuint32_t(kStageRow / kKBlock / 2)};
      if (encode(&maps->map[layer][kId][1], CU_TENSOR_MAP_DATA_TYPE_UINT8, 3,
                 const_cast<uint8_t *>(args.down), dims, strides, box, elements,
                 CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B,
                 CU_TENSOR_MAP_L2_PROMOTION_L2_256B,
                 CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE) != CUDA_SUCCESS)
        return false;
    }
    return true;
  }
};

__device__ void down_unit(Ctx &c, int layer, int u, int parity) {
  float *red = reinterpret_cast<float *>(c.smem + kSmemRed) + parity * kConsumerWarps * 32 * 4;
  int const lane = c.lane;
  // rows g8, g8 + 8; tokens 2 tig, + 1
  float2 acc01 = make_float2(0.0f, 0.0f), acc23 = make_float2(0.0f, 0.0f);
  gemm_mainloop<kGemmDown>(c, acc01, acc23);
  *reinterpret_cast<float4 *>(red + (c.warp * 32 + lane) * 4) =
      make_float4(acc01.x, acc01.y, acc23.x, acc23.y);
  consumer_sync();
  // Thread (t, pair) of warps 0-1 sums token t's columns 2 pair, 2 pair + 1; even pairs push their
  // word and the next one with one 16-byte store per rank (half the NVLink writes of 8-byte
  // stores). No barrier: the next unit reduces into the other buffer, and the one after it writes
  // this buffer only after that unit's barrier, which follows these reads.
  if (c.tid < 2 * 32) {
    int const t = c.tid / (kTallRows / 2), pair = c.tid % (kTallRows / 2);
    uint32_t const gen = sum_generation(c, 2 * layer + 2);
    uint64_t word = 0;
    if (t < kRows)
      word = uint64_t(pack_bf16(tall_sum(red, t, 2 * pair), tall_sum(red, t, 2 * pair + 1))) |
             (uint64_t(gen + 1u) << 32);
    uint64_t const next = __shfl_down_sync(0xffffffffu, word, 1);
    if (t < kRows && (pair & 1) == 0)
      for (int receiver = 0; receiver < kTp; ++receiver)
        st_ll2(ll_row(c.a.arena[receiver], int(gen & 1u), c.a.rank, t) + u * (kTallRows / 2) + pair,
               word, next);
  }
}

// Consumer side: every unit of this GEMM on this CTA.
__device__ void down_phase(Ctx &c, int layer) {
  constexpr int units = GemmDown::units();
  if (!gemm_prologue<kGemmDown>(c, layer, units))
    return;
  int parity = 0;
  for (int u = first_gemm_unit(c.cta, c.grid); u < units; u += c.grid, parity ^= 1)
    down_unit(c, layer, u, parity);
  consumer_sync(); // the phase's shared memory is reused by the next phase
}

} // namespace
} // namespace m31_draft
