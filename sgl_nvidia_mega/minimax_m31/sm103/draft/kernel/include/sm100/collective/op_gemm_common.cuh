#pragma once

// Shared by the four GEMM OPs: the MXFP8 mainloop, the reduction readout, the activation loaders
// (each GEMM's input contract) and the per-phase prologue.

#include "include/sm100/collective/op_weight_stream.cuh"

#include <type_traits>

namespace m31_draft {
namespace {

// ---------------------------------------------------------------------------
// MXFP8 GEMM unit u: out[token][n] = sum_k act[token][k] * w[n][k] for 16 weight
// rows and the full K. Warp w takes the 32-wide scale groups 8w..8w+7 of every
// stage (4w..4w+3 of down's 1024-byte last stage): A = weights via ldmatrix from
// the swizzled stage, B = activation^T, the e4m3 product of each group as two
// exact f16 MMAs, then the group's two UE8M0 scales applied to the FP32 result
// before accumulating in FP32; the eight warp sums are added in warp order. qkv
// publishes the BF16 rows, gate_up applies SwiGLU-OAI to the BF16 gate/up rows
// and publishes the BF16 result, o_proj and down push their BF16 rows (this
// rank's TP4 partial) to every rank.
__device__ __forceinline__ float tall_sum(float const *red, int t, int r) {
  int const lane = (r & 7) * 4 + (t >> 1), j = (r >> 3) * 2 + (t & 1);
  float v = red[lane * 4 + j];
#pragma unroll
  for (int w = 1; w < kConsumerWarps; ++w)
    v += red[(w * 32 + lane) * 4 + j];
  return v;
}

// The unit's MXFP8 product over the full K: every stage of the unit in ring order, warp w taking
// the scale groups gw * w .. gw * w + gw - 1 of each stage (gw = 8 for the 2048-byte stages, 4 for
// down's 1024-byte last stage). The two stage shapes are separate instantiations, so every group's
// shared-memory offsets, scale bytes and activation scales are compile-time constants.
template <int G>
__device__ __forceinline__ void gemm_mainloop(Ctx &c, float2 &acc01, float2 &acc23) {
  constexpr int k = gemm_k(G), ksf = k / kScaleBlock;
  constexpr int kFull = k / kStageRow;                                    // 2048-byte stages
  constexpr int kTail = (k % kStageRow) / (kConsumerWarps * kScaleBlock); // groups per warp: 0, 4
  constexpr bool pair = G == kGemmGateUp;
  // one K block of a box
  constexpr uint32_t block_bytes = pair ? kPairRows * kKBlock : kTallRows * kKBlock;
  int const lane = c.lane, g8 = lane >> 2, tig = lane & 3;
  // ldmatrix: lane supplies row lrow (rows 0-7 / 8-15 for matrices 0,2 / 1,3) and the 16-byte half
  // lhalf of a 32-byte group; a box is [K block][rows][128 B] with chunk c of row r at c ^ (r % 8).
  int const lrow = (lane & 7) + ((lane >> 3) & 1) * 8, lhalf = lane >> 4;
  uint32_t const row_off =
      pair ? (lrow >> 3) * (kSlotW / 2) + (lrow & 7) * kKBlock : lrow * kKBlock;
  // the swizzled 16-byte chunk of the lane's half for each 32-byte group position q of a K block
  uint32_t chunk[4];
#pragma unroll
  for (int q = 0; q < 4; ++q)
    chunk[q] = row_off + (uint32_t(((q * 2 + lhalf) ^ (lrow & 7))) << 4);
  // B fragments (activation^T): token g8's bytes; token 7 is the zero row. Plain loads, so the
  // per-group offsets fold into the instructions.
  uint32_t const *b_row =
      reinterpret_cast<uint32_t const *>(c.smem + kSmemAct + g8 * kActRow + tig * 4);
  float2 const *xf = reinterpret_cast<float2 const *>(c.smem + kSmemXf) + tig * kXfStride;
  auto stage = [&](auto gw_const, int st) {
    constexpr int gw = decltype(gw_const)::value;
    uint32_t const slot_base = c.weights.acquire();
    int const j0 = c.warp * gw;                          // first group of this warp in the stage
    int const grp = st * (kStageRow / kScaleBlock) + j0; // first group of this warp in K
    uint32_t sw0[2], sw1[2];
    float2 xs[gw];
    uint32_t const sw_base = slot_base + kSlotW;
    if constexpr (gw == 8) {
      lds_u64(sw_base + g8 * ksf + grp, sw0);
      lds_u64(sw_base + (g8 + 8) * ksf + grp, sw1);
    } else {
      sw0[0] = lds_u32(sw_base + g8 * ksf + grp);
      sw1[0] = lds_u32(sw_base + (g8 + 8) * ksf + grp);
      sw0[1] = sw1[1] = 0u;
    }
#pragma unroll
    for (int q = 0; q < gw / 2; ++q) {
      float4 const v = *reinterpret_cast<float4 const *>(xf + grp + 2 * q);
      xs[2 * q] = make_float2(v.x, v.y);
      xs[2 * q + 1] = make_float2(v.z, v.w);
    }
    // this warp's first K block of the stage (j0 is a multiple of 4)
    uint32_t const a_warp = slot_base + (j0 >> 2) * block_bytes;
    uint32_t const *b_grp = b_row + grp * (kScaleBlock / 4);
#pragma unroll
    for (int jj = 0; jj < gw; ++jj) {
      // group jj of the warp: K block jj / 4, 32-byte group jj % 4 of its 128 bytes
      uint32_t a[4];
      ldsm_x4(a, a_warp + (jj >> 2) * block_bytes + chunk[jj & 3]);
      uint32_t const b[2] = {b_grp[jj * (kScaleBlock / 4)], b_grp[jj * (kScaleBlock / 4) + 4]};
      float d[4];
      mma_e4m3_f16(d, a, b);
      // FP32 scale products and accumulation as packed pairs (IEEE, same as the scalar ops).
      float const w0 = ue8m0_byte(sw0[jj >> 2], jj & 3), w1 = ue8m0_byte(sw1[jj >> 2], jj & 3);
      float2 const p0 = __fmul2_rn(make_float2(w0, w0), xs[jj]);
      float2 const p1 = __fmul2_rn(make_float2(w1, w1), xs[jj]);
      acc01 = __ffma2_rn(make_float2(d[0], d[1]), p0, acc01);
      acc23 = __ffma2_rn(make_float2(d[2], d[3]), p1, acc23);
    }
    c.weights.retire(lane);
  };
  for (int st = 0; st < kFull; ++st)
    stage(std::integral_constant<int, 8>{}, st);
  if constexpr (kTail > 0)
    stage(std::integral_constant<int, kTail>{}, kFull);
}

// The decoded activation scale of token row `row`, group `grp` in the GEMM's float-pair table
// (pair (tig, grp) = tokens 2 tig, 2 tig + 1), written by the loader that holds the code.
__device__ __forceinline__ void put_act_scale(Ctx &c, int row, int grp, uint32_t code) {
  reinterpret_cast<float *>(c.smem + kSmemXf)[((row >> 1) * kXfStride + grp) * 2 + (row & 1)] =
      ue8m0(code);
}

// down's input: the BF16 SwiGLU output from its LL6 words, MXFP8-quantized here
// (SGLang's quantizer: UE8M0 per 32 values, E4M3 RNE). Thread i < 224 owns the 32-value
// groups i % 32 + 32 m (m < 3) of token row i / 32: word e of a group sits at e * 96 + group
// (swi_word), so each poll instruction of a warp reads 32 consecutive words; every load is in
// flight at once, and every address is a per-thread base plus a constant.
__device__ void load_swiglu_ll(Ctx &c, int layer) {
  constexpr int kGroupsRow = kInter / kScaleBlock; // 96
  constexpr int kLanes = 32;
  constexpr int kPer = kGroupsRow / kLanes;                // 3
  constexpr int kGroupWords = kScaleBlock / kPairRows * 3; // 12
  static_assert(kGroupsRow % kLanes == 0 && kRows * kLanes <= kConsumerThreads);
  // A CTA whose last gate_up unit is in the last wave arrives here about when the SwiGLU rows are
  // complete and polls every word directly; a CTA that finished earlier first waits (one thread) on
  // a word of the last wave, so its full poll does not load L2 while the last wave streams.
  if (!in_last_wave(c.cta, c.grid, gemm_n(kGemmGateUp) / kTallRows))
    gate_ll6(c, swi_ll(c.arena, layer) + gemm_gate_word(kGemmGateUp, c.grid));
  if (c.tid < kRows * kLanes) {
    int const t = c.tid / kLanes, g0 = c.tid % kLanes;
    // word e of group g0 + 32 k at src + e * 96 + 32 k: lanes poll consecutive words
    uint64_t const *src = swi_ll(c.arena, layer) + t * kSwiLLRowWords + g0;
    uint32_t const tag = ll_tag(c.epoch);
    uint64_t w[kPer][kGroupWords];
#pragma unroll
    for (int k = 0; k < kPer; ++k)
#pragma unroll
      for (int e = 0; e < kGroupWords; ++e)
        w[k][e] = ld_relaxed_gpu_u64(src + e * kGroupsRow + k * kLanes);
    unsigned long long start = 0;
    for (;;) {
      bool ready = true;
#pragma unroll
      for (int k = 0; k < kPer; ++k)
#pragma unroll
        for (int e = 0; e < kGroupWords; ++e) {
          if (!ll_ok(w[k][e], tag)) {
            ready = false;
            w[k][e] = ld_relaxed_gpu_u64(src + e * kGroupsRow + k * kLanes);
          }
        }
      if (ready)
        break;
      if (start == 0)
        start = globaltimer();
      else
        check_deadline(start);
    }
    uint8_t *s_act = c.smem + kSmemAct + t * kActRow + g0 * kScaleBlock;
#pragma unroll
    for (int k = 0; k < kPer; ++k) {
      // Value j of the group (unit j / 8, value i = j % 8) is half i % 3 of word 3 (j / 8) + i / 3.
      auto value = [&](int j) {
        int const unit = j / kPairRows, i = j % kPairRows;
        return bf2f(uint16_t(w[k][3 * unit + i / 3] >> (16 * (i % 3))));
      };
      float amax = 0.0f;
#pragma unroll
      for (int j = 0; j < kScaleBlock; ++j)
        amax = fmaxf(amax, fabsf(value(j)));
      float inv;
      int const scale = e8m0_from_amax(amax, &inv);
      uint32_t b[kScaleBlock / 4];
#pragma unroll
      for (int i = 0; i < kScaleBlock / 4; ++i)
        b[i] = to_e4m3x2(value(4 * i) * inv, value(4 * i + 1) * inv) |
               (to_e4m3x2(value(4 * i + 2) * inv, value(4 * i + 3) * inv) << 16);
      uint4 *dst = reinterpret_cast<uint4 *>(s_act + k * kLanes * kScaleBlock);
      dst[0] = make_uint4(b[0], b[1], b[2], b[3]);
      dst[1] = make_uint4(b[4], b[5], b[6], b[7]);
      put_act_scale(c, t, g0 + k * kLanes, uint32_t(scale));
    }
  }
}

// The normed MXFP8 activation of norm stage `stage` from its LL6 words, polled directly
// and unpacked into the GEMM's shared-memory rows. Thread i owns the act word pairs i + 256 m
// (m < 2: words 2p, 2p + 1, one 16-byte load, 12 payload bytes = three aligned 32-bit stores)
// of every token row and, for i < 224, scale word i % 32 of row i / 32, so every address is a
// per-thread base plus a constant.
__device__ void load_act_ll(Ctx &c, int stage) {
  constexpr int kActWords = kHidden / 6;               // 1024 per row
  constexpr int kPairs = kActWords / 2;                // 512 per row
  constexpr int kPerRow = kPairs / kConsumerThreads;   // 2
  constexpr int kSfWords = kActLLRowWords - kActWords; // 32 per row
  static_assert(kPairs % kConsumerThreads == 0 && kRows * kSfWords <= kConsumerThreads);
  static_assert(kActLLRowWords % 2 == 0 && kActRow % 4 == 0);
  gate_tp4(c, stage); // the norm CTAs publish these words shortly after their TP4 sum is complete
  uint64_t const *src = act_ll(c.arena, stage) + 2 * c.tid;
  uint64_t const *sf_src =
      act_ll(c.arena, stage) + (c.tid / kSfWords) * kActLLRowWords + kActWords + c.tid % kSfWords;
  bool const has_sf = c.tid < kRows * kSfWords;
  uint32_t const tag = ll_tag(c.epoch);
  uint64_t w[kRows][kPerRow][2], ws;
#pragma unroll
  for (int t = 0; t < kRows; ++t)
#pragma unroll
    for (int m = 0; m < kPerRow; ++m)
      ld_relaxed_gpu_v2(src + t * kActLLRowWords + 2 * m * kConsumerThreads, w[t][m][0],
                        w[t][m][1]);
  ws = has_sf ? ld_relaxed_gpu_u64(sf_src) : ll_word(0, tag);
  unsigned long long start = 0;
  for (;;) {
    bool ready = ll_ok(ws, tag);
#pragma unroll
    for (int t = 0; t < kRows; ++t)
#pragma unroll
      for (int m = 0; m < kPerRow; ++m)
        ready &= ll_ok(w[t][m][0], tag) && ll_ok(w[t][m][1], tag);
    if (ready)
      break;
    if (start == 0)
      start = globaltimer();
    else
      check_deadline(start);
#pragma unroll
    for (int t = 0; t < kRows; ++t)
#pragma unroll
      for (int m = 0; m < kPerRow; ++m)
        if (!ll_ok(w[t][m][0], tag) || !ll_ok(w[t][m][1], tag))
          ld_relaxed_gpu_v2(src + t * kActLLRowWords + 2 * m * kConsumerThreads, w[t][m][0],
                            w[t][m][1]);
    if (!ll_ok(ws, tag))
      ws = ld_relaxed_gpu_u64(sf_src);
  }
  // pair p of a row: bytes 12 p .. 12 p + 11 = word 2p's six payload bytes, then word 2p + 1's
  uint8_t *act = c.smem + kSmemAct + 12 * c.tid;
#pragma unroll
  for (int t = 0; t < kRows; ++t)
#pragma unroll
    for (int m = 0; m < kPerRow; ++m) {
      uint32_t const a_lo = uint32_t(w[t][m][0]), a_hi = uint32_t(w[t][m][0] >> 32);
      uint32_t const b_lo = uint32_t(w[t][m][1]), b_hi = uint32_t(w[t][m][1] >> 32);
      uint32_t *dst = reinterpret_cast<uint32_t *>(act + t * kActRow + 12 * m * kConsumerThreads);
      dst[0] = a_lo;
      dst[1] = __byte_perm(a_hi, b_lo, 0x5410);
      dst[2] = __byte_perm(b_lo, b_hi, 0x5432);
    }
  if (has_sf) {
    // scale word w of row r: the codes of groups 6 w .. 6 w + 5
    int const r = c.tid / kSfWords, g = 6 * (c.tid % kSfWords);
#pragma unroll
    for (int b = 0; b < 6; ++b)
      put_act_scale(c, r, g + b, uint32_t(ws >> (8 * b)) & 0xffu);
  }
}

// The o_proj activation from its LL words: every word is polled until it carries this
// call's generation (the combine's writes need no separate counter).
__device__ void load_attn_act(Ctx &c, int layer) {
  constexpr int kPer = (kAttnLLWords + kConsumerThreads - 1) / kConsumerThreads;
  // A CTA without a combine row reaches this poll during the attention phase: it first waits for
  // the attention partials (eight light counter polls, as the combine CTAs do), so that its full
  // poll of the combine's words does not load L2 during attention.
  if (first_combine_row(c.cta) >= kAttnRows)
    wait_signal(c, kSigAttn, layer, uint32_t(c.grid));
  uint64_t const *src = at<uint64_t>(c.arena, kAttnLL) + size_t(layer) * kAttnLLWords;
  uint64_t w[kPer];
#pragma unroll
  for (int k = 0; k < kPer; ++k) {
    int const i = c.tid + k * kConsumerThreads;
    w[k] = i < kAttnLLWords ? ld_relaxed_gpu_u64(src + i) : uint64_t(c.epoch) << 32;
  }
  unsigned long long start = 0;
  for (;;) {
    bool ready = true;
#pragma unroll
    for (int k = 0; k < kPer; ++k)
      ready &= uint32_t(w[k] >> 32) == c.epoch;
    if (ready)
      break;
    if (start == 0)
      start = globaltimer();
    else
      check_deadline(start);
#pragma unroll
    for (int k = 0; k < kPer; ++k) {
      int const i = c.tid + k * kConsumerThreads;
      if (uint32_t(w[k] >> 32) != c.epoch)
        w[k] = ld_relaxed_gpu_u64(src + i);
    }
  }
#pragma unroll
  for (int k = 0; k < kPer; ++k) {
    int const i = c.tid + k * kConsumerThreads;
    if (i < kAttnActWords) {
      int const t = i / (kQSize / 4), j = i % (kQSize / 4);
      *reinterpret_cast<uint32_t *>(c.smem + kSmemAct + t * kActRow + j * 4) = uint32_t(w[k]);
    } else if (i < kAttnLLWords) {
      int const si = i - kAttnActWords, t = si / (kQSize / kScaleBlock),
                grp = si % (kQSize / kScaleBlock);
      put_act_scale(c, t, grp, uint32_t(w[k]) & 0xffu);
    }
  }
}

// The per-phase prologue shared by every GEMM: this CTA's activation rows and their decoded
// scales in shared memory (the loader is the GEMM's input contract) and the padding token's zero
// row. Returns false when the CTA has no unit of this GEMM.
template <int G> __device__ bool gemm_prologue(Ctx &c, int layer, int units) {
  if (first_gemm_unit(c.cta, c.grid) >= units)
    return false;
  // Padding token 7 (it does not depend on the input): its B fragments read zeros and its
  // scales are code 0x7f = 1.0 (its column is discarded).
  for (int i = c.tid; i < gemm_k(G) / 16; i += kConsumerThreads)
    reinterpret_cast<int4 *>(c.smem + kSmemAct + kRows * kActRow)[i] = make_int4(0, 0, 0, 0);
  for (int i = c.tid; i < gemm_k(G) / kScaleBlock; i += kConsumerThreads)
    put_act_scale(c, kRows, i, 0x7fu);
  // Every input arrives as self-validating words, polled directly; the loader writes the
  // activation rows and their decoded scales (float pairs per group, decoded once per phase).
  if constexpr (G == kGemmO)
    load_attn_act(c, layer);
  else if constexpr (G == kGemmDown)
    load_swiglu_ll(c, layer);
  else
    load_act_ll(c, G == kGemmQkv ? 2 * layer : 2 * layer + 1);
  consumer_sync();
  return true;
}

} // namespace
} // namespace m31_draft
