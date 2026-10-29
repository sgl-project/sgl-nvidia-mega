#pragma once

// OP gemm_qkv: qkv: [7, 6144] x [2304, 6144]^T, publishes the raw q | k | v rows (LL6 words, bf16
// pairs) for attention.

#include "include/sm100/collective/op_gemm_common.cuh"

namespace m31_draft {
namespace {

// Schedule: units of 16 weight rows x the full K (one 16-row box),
// stages of 16 rows x 2048 K bytes, stored as 128-byte swizzled K blocks.
struct GemmQkv {
  static constexpr int kId = kGemmQkv;
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
    uint8_t const *sf = args.qkv_sf;
    tma_load_3d(dst, &maps.map[layer][kId][0], 0, y0, kb, full, policy);
    bulk_load(dst + kSlotW, sf + size_t(y0) * ksf, uint32_t(kTallRows * ksf), full, policy);
  }
  // Host side: this GEMM's tensor maps of one layer (dims (128 bytes, rows, K / 128 blocks),
  // byte strides (K, 128), 128-byte swizzle).
  static bool encode(PFN_cuTensorMapEncodeTiled_v12000 encode, LayerArgs const &args, int layer,
                     WeightMaps *maps) {
    cuuint64_t const dims[3] = {cuuint64_t(kKBlock), cuuint64_t(kQkvRows),
                                cuuint64_t(kHidden / kKBlock)};
    cuuint64_t const strides[2] = {cuuint64_t(kHidden), cuuint64_t(kKBlock)};
    cuuint32_t const elements[3] = {1, 1, 1};
    {
      cuuint32_t const box[3] = {cuuint32_t(kKBlock), cuuint32_t(kTallRows),
                                 cuuint32_t(kStageRow / kKBlock)};
      if (encode(&maps->map[layer][kId][0], CU_TENSOR_MAP_DATA_TYPE_UINT8, 3,
                 const_cast<uint8_t *>(args.qkv), dims, strides, box, elements,
                 CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B,
                 CU_TENSOR_MAP_L2_PROMOTION_L2_256B,
                 CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE) != CUDA_SUCCESS)
        return false;
    }
    return true;
  }
};

__device__ void qkv_unit(Ctx &c, int layer, int u, int parity) {
  float *red = reinterpret_cast<float *>(c.smem + kSmemRed) + parity * kConsumerWarps * 32 * 4;
  int const lane = c.lane;
  // rows g8, g8 + 8; tokens 2 tig, + 1
  float2 acc01 = make_float2(0.0f, 0.0f), acc23 = make_float2(0.0f, 0.0f);
  gemm_mainloop<kGemmQkv>(c, acc01, acc23);
  *reinterpret_cast<float4 *>(red + (c.warp * 32 + lane) * 4) =
      make_float4(acc01.x, acc01.y, acc23.x, acc23.y);
  consumer_sync();
  // Publish the BF16 rows as LL6 words that attention polls directly. No barrier: the next unit
  // reduces into the other buffer, and the one after it writes this buffer only after that unit's
  // barrier, which follows these reads.
  if (c.tid < kRows * kTallRows / 2) {
    int const t = c.tid / (kTallRows / 2), pair = c.tid % (kTallRows / 2);
    uint64_t const word = pack_bf16(tall_sum(red, t, 2 * pair), tall_sum(red, t, 2 * pair + 1));
    st_relaxed_gpu_u64(qkv_ll(c.arena, layer) + t * kQkvLLRowWords + u * (kTallRows / 2) + pair,
                       ll_word(word, ll_tag(c.epoch)));
  }
}

// Consumer side: every unit of this GEMM on this CTA.
__device__ void qkv_phase(Ctx &c, int layer) {
  constexpr int units = GemmQkv::units();
  if (!gemm_prologue<kGemmQkv>(c, layer, units))
    return;
  int parity = 0;
  for (int u = first_gemm_unit(c.cta, c.grid); u < units; u += c.grid, parity ^= 1)
    qkv_unit(c, layer, u, parity);
  consumer_sync(); // the phase's shared memory is reused by the next phase
}

} // namespace
} // namespace m31_draft
