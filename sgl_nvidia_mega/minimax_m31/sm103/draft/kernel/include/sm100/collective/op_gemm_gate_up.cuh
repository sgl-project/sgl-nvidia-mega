#pragma once

// OP gemm_gate_up: gate_up: [7, 6144] x [6144, 6144]^T (gate rows 0-3071 | up rows 3072-6143),
// SwiGLU-OAI epilogue, publishes the BF16 SwiGLU rows (LL6 words, 3 per 8 channels) for down.

#include "include/sm100/collective/op_gemm_common.cuh"

namespace m31_draft {
namespace {

// Schedule: units of 16 weight rows x the full K (8 gate rows and the matching 8 up rows),
// stages of 16 rows x 2048 K bytes, stored as 128-byte swizzled K blocks.
struct GemmGateUp {
  static constexpr int kId = kGemmGateUp;
  __host__ __device__ static constexpr int units() { return gemm_n(kId) / kTallRows; }
  __device__ static int stages(int) { return (gemm_k(kId) + kStageRow - 1) / kStageRow; }
  // The consumers of this GEMM's output gate on token 0's first word of the last unit.
  __device__ static int gate_word(int grid) { return swi_word(last_gemm_unit(units(), grid), 0); }
  // Producer side: issue stage st of unit u (lane 0 of the weight-stream warp).
  __device__ static void issue(LayerArgs const &args, WeightMaps const &maps, int layer, int u,
                               int st, uint32_t dst, uint32_t full, uint64_t policy) {
    int const y0 = u * kPairRows, y1 = kInter + u * kPairRows;
    int const k = gemm_k(kId), ksf = k / kScaleBlock;
    int const kc = k - st * kStageRow < kStageRow ? k - st * kStageRow : kStageRow;
    mbar_expect_tx(full, uint32_t(kTallRows * kc + kTallRows * ksf));
    int const kb = st * (kStageRow / kKBlock);
    uint8_t const *sf = args.gate_up_sf;
    // rows 0-7: gate, rows 8-15: the matching up rows
    tma_load_3d(dst, &maps.map[layer][kId][0], 0, y0, kb, full, policy);
    tma_load_3d(dst + kSlotW / 2, &maps.map[layer][kId][0], 0, y1, kb, full, policy);
    bulk_load(dst + kSlotW, sf + size_t(y0) * ksf, uint32_t(kPairRows * ksf), full, policy);
    bulk_load(dst + kSlotW + kPairRows * ksf, sf + size_t(y1) * ksf, uint32_t(kPairRows * ksf),
              full, policy);
  }
  // Host side: this GEMM's tensor maps of one layer (dims (128 bytes, rows, K / 128 blocks),
  // byte strides (K, 128), 128-byte swizzle).
  static bool encode(PFN_cuTensorMapEncodeTiled_v12000 encode, LayerArgs const &args, int layer,
                     WeightMaps *maps) {
    cuuint64_t const dims[3] = {cuuint64_t(kKBlock), cuuint64_t(kGateUpRows),
                                cuuint64_t(kHidden / kKBlock)};
    cuuint64_t const strides[2] = {cuuint64_t(kHidden), cuuint64_t(kKBlock)};
    cuuint32_t const elements[3] = {1, 1, 1};
    {
      cuuint32_t const box[3] = {cuuint32_t(kKBlock), cuuint32_t(kPairRows),
                                 cuuint32_t(kStageRow / kKBlock)};
      if (encode(&maps->map[layer][kId][0], CU_TENSOR_MAP_DATA_TYPE_UINT8, 3,
                 const_cast<uint8_t *>(args.gate_up), dims, strides, box, elements,
                 CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B,
                 CU_TENSOR_MAP_L2_PROMOTION_L2_256B,
                 CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE) != CUDA_SUCCESS)
        return false;
    }
    return true;
  }
};

__device__ void gate_up_unit(Ctx &c, int layer, int u, int parity) {
  float *red = reinterpret_cast<float *>(c.smem + kSmemRed) + parity * kConsumerWarps * 32 * 4;
  int const lane = c.lane;
  // rows g8, g8 + 8; tokens 2 tig, + 1
  float2 acc01 = make_float2(0.0f, 0.0f), acc23 = make_float2(0.0f, 0.0f);
  gemm_mainloop<kGemmGateUp>(c, acc01, acc23);
  *reinterpret_cast<float4 *>(red + (c.warp * 32 + lane) * 4) =
      make_float4(acc01.x, acc01.y, acc23.x, acc23.y);
  consumer_sync();
  // SwiGLU-OAI on the BF16 gate/up rows, published as LL6 words that down polls directly. Thread
  // t * 8 + i (warps 0-1) computes value i of token t; lane 3k of each token's eight lanes gathers
  // values 3k .. 3k + 2 (word 2: 6, 7) with shuffles and stores word k. No barrier: the next unit
  // reduces into the other buffer, and the one after it writes this buffer only after that unit's
  // barrier, which follows these reads.
  if (c.tid < 64) {
    int const t = c.tid / kPairRows, i = c.tid % kPairRows;
    uint32_t bits = 0;
    if (t < kRows) {
      float const gv = fminf(bf16r(tall_sum(red, t, i)), kSwigluLimit);
      float const uv =
          fminf(fmaxf(bf16r(tall_sum(red, t, kPairRows + i)), -kSwigluLimit), kSwigluLimit);
      float const sigmoid = 1.0f / (1.0f + expf(-gv * kSwigluAlpha));
      bits = f2bf(gv * sigmoid * (uv + 1.0f));
    }
    uint32_t const next1 = __shfl_down_sync(0xffffffffu, bits, 1);
    uint32_t const next2 = __shfl_down_sync(0xffffffffu, bits, 2);
    if (t < kRows && i % 3 == 0) {
      uint64_t const payload = uint64_t(bits) | (uint64_t(next1) << 16) |
                               (i + 2 < kPairRows ? uint64_t(next2) << 32 : 0);
      st_relaxed_gpu_u64(swi_ll(c.arena, layer) + t * kSwiLLRowWords + swi_word(u, i / 3),
                         ll_word(payload, ll_tag(c.epoch)));
    }
  }
}

// Consumer side: every unit of this GEMM on this CTA.
__device__ void gate_up_phase(Ctx &c, int layer) {
  constexpr int units = GemmGateUp::units();
  if (!gemm_prologue<kGemmGateUp>(c, layer, units))
    return;
  int parity = 0;
  for (int u = first_gemm_unit(c.cta, c.grid); u < units; u += c.grid, parity ^= 1)
    gate_up_unit(c, layer, u, parity);
  consumer_sync(); // the phase's shared memory is reused by the next phase
}

} // namespace
} // namespace m31_draft
