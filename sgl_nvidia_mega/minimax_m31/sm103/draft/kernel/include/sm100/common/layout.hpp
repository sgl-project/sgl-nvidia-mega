#pragma once

// Launch geometry, arena layout and shared-memory layout of the draft backbone kernel,
// shared by every OP: no OP chooses its own offsets.

#include "include/draft_args.hpp"
#include <cstddef>
#include <cuda.h>
#include <cudaTypedefs.h>

namespace m31_draft {
namespace {

constexpr int kConsumerWarps = 8;
constexpr int kConsumerThreads = kConsumerWarps * 32;
constexpr int kThreads = kConsumerThreads + 32; // + the weight-streaming warp
constexpr int kGemms = 4;                       // qkv, o, gate_up, down
constexpr int kTallRows = 16;                   // weight rows of a unit
// gate_up unit: 8 gate rows + the matching 8 up rows
constexpr int kPairRows = kTallRows / 2;
// K bytes of a weight row per stage (down's second stage: 1024)
constexpr int kStageRow = 2048;
constexpr int kKBlock = 128; // K bytes of one swizzled box row
// [K block][16 rows][128 B] (gate_up: two 8-row boxes)
constexpr int kSlotW = kTallRows * kStageRow;
constexpr int kMaxSfRow = kHidden / kScaleBlock; // scale bytes of a weight row (largest K)
// + the tile's scale rows
constexpr int kSlotBytes = (kSlotW + kTallRows * kMaxSfRow + 1023) / 1024 * 1024;
constexpr int kRing = 4;   // stages in shared memory
constexpr int kChunk = 64; // keys of one attention work item
constexpr int kMaxChunks = kWindow / kChunk;
constexpr int kAttnRows = kRows * kQHeads; // 112: row = token * 16 + head
// An attention work item takes one chunk for half of the heads: 56 query rows (local row
// token * 8 + head % 8) in four 16-row tiles, the last padded with 8 zero rows.
constexpr int kHalfHeads = kQHeads / 2;
constexpr int kItemRows = kRows * kHalfHeads;     // 56
constexpr int kItemTiles = (kItemRows + 15) / 16; // 4
// TP4 sums of one call: embedding, then o_proj, down per layer
constexpr int kSums = 2 * kLayers + 1;
constexpr int kLLWords = kHidden / 2; // TP4 words of one token row
constexpr float kSoftmaxScaleLog2 = 0.08838834764831845f * 1.4426950408889634f;
constexpr float kNegInf = -__builtin_huge_valf();
// A wait on another CTA or rank longer than this traps; the host may pass its own bound.
#ifndef M31_DRAFT_WAIT_TIMEOUT_NS
#define M31_DRAFT_WAIT_TIMEOUT_NS 10000000000ull
#endif
constexpr unsigned long long kWaitTimeoutNs = M31_DRAFT_WAIT_TIMEOUT_NS;

enum : int { kGemmQkv = 0, kGemmO = 1, kGemmGateUp = 2, kGemmDown = 3 };
// Per-call hand-over counters (only the attention partials use one); each receives a
// fixed number of arrivals per call.
enum : int { kSigAttn = 0, kSigKinds };

// Arena layout (bytes). The arena is zeroed once; counters and TP4 words carry the call generation.
constexpr size_t kCallWord = 0;    // u32: completed calls
constexpr size_t kStartWord = 128; // u32: CTAs that read the generation, every call
constexpr size_t kSignals = 4096;  // u32 per 128-byte line: [kind][16][kSignalSubs]
constexpr int kSignalSlots = 16;
// arrivals spread over 8 counters (arrival id mod 8), waited on in parallel
constexpr int kSignalSubs = 8;
// u64 [2][kTp sender][kRows][kLLWords]
constexpr size_t kLL = size_t(1) << 20;
constexpr size_t kResidual = size_t(3) << 20; // bf16 [kRows][kHidden]
// Self-validating LL6 words (low 48 bits payload, high 16 bits the call's tag), one buffer
// per use, each written once per call and polled directly by its readers:
//  - the normed MXFP8 activation of norm stage s < 2 * kLayers (stage 2l: qkv input of layer l,
//    stage 2l + 1: gate_up input): [kRows][1024 words of 6 e4m3 bytes | 32 words of 6 scale bytes];
//  - the raw qkv output of a layer: [kRows][kQkvRows / 2 words of a bf16 pair];
//  - the SwiGLU output of a layer: [kRows][swi_word(unit, k)], word k = bf16 values 0-2 | 3-5 |
//  6-7.
constexpr int kActLLRowWords = kHidden / 6 + kHidden / kScaleBlock / 6; // 1056
constexpr int kActLLWords = kRows * kActLLRowWords;
constexpr int kQkvLLRowWords = kQkvRows / 2;
constexpr int kQkvLLWords = kRows * kQkvLLRowWords;
constexpr int kSwiLLRowWords = kInter / kPairRows * 3;
// Word k (0-2) of gate_up unit u in a token row of SwiGLU words: [12 positions of a 32-channel
// group][96 groups] (a group is 4 units of 8 channels), so the down loaders' lanes (one group
// each) poll consecutive words (coalesced) instead of 96-byte strided ones.
__host__ __device__ constexpr int swi_word(int u, int k) {
  return ((u % (kScaleBlock / kPairRows)) * 3 + k) * (kInter / kScaleBlock) +
         u / (kScaleBlock / kPairRows);
}
constexpr int kSwiLLWords = kRows * kSwiLLRowWords;
constexpr size_t kActLL = kResidual + (size_t(128) << 10);
constexpr size_t kQkvLL = kActLL + size_t(2 * kLayers) * kActLLWords * 8;
constexpr size_t kSwiLL = kQkvLL + size_t(kLayers) * kQkvLLWords * 8;
// f32 [kMaxChunks][kAttnRows][kHeadDim]
constexpr size_t kAttnPartial = size_t(8) << 20;
constexpr size_t kAttnStats = size_t(12) << 20; // f32 [kAttnRows][kMaxChunks][2]
// The o_proj activation as LL words (4 e4m3 bytes or one scale code, plus the call
// generation), one buffer per layer: u64 [kRows][kQSize / 4] then [kRows][kQSize / 32].
constexpr int kAttnActWords = kRows * (kQSize / 4);
constexpr int kAttnLLWords = kAttnActWords + kRows * (kQSize / kScaleBlock);
constexpr size_t kAttnLL = size_t(13) << 20;
static_assert(kSignals + size_t(kSigKinds) * kSignalSlots * kSignalSubs * 128 <= kLL);
static_assert(kLL + size_t(2) * kTp * kRows * kLLWords * 8 <= kResidual);
static_assert(kResidual + size_t(kRows) * kHidden * 2 <= kActLL);
static_assert(kHidden % 6 == 0 && kHidden / kScaleBlock % 6 == 0 && kQkvRows % 2 == 0);
static_assert(kSwiLL + size_t(kLayers) * kSwiLLWords * 8 <= kAttnPartial);
static_assert(kAttnPartial + size_t(kMaxChunks) * kAttnRows * kHeadDim * 4 <= kAttnStats);
static_assert(kAttnStats + size_t(kMaxChunks) * kAttnRows * 2 * 4 <= kAttnLL);
static_assert(kAttnLL + size_t(kLayers) * kAttnLLWords * 8 <= kArenaBytes);

// Shared memory. The ring is private to the weight stream; the OPs share one union region,
// each OP owning one slice (only one OP runs on a CTA at a time; every OP ends with a
// consumer barrier before the next one reuses the region).
constexpr int kSmemRing = 0; // [kRing][kSlotBytes], 1024-byte aligned (128-byte swizzle)
constexpr int kSmemUnion = kSmemRing + kRing * kSlotBytes;
// op_attention slice.
constexpr int kARow = kHeadDim + 8; // attention row (bf16 elements)
constexpr int kSmemQ = kSmemUnion;
constexpr int kSmemK = kSmemQ + kItemTiles * 16 * kARow * 2;
constexpr int kSmemV = kSmemK + kChunk * kARow * 2;
// bf16 [2][kRows][kHeadDim]: block K | V
constexpr int kSmemFresh = kSmemV + kChunk * kARow * 2;
constexpr int kSmemAttnEnd = kSmemFresh + 2 * kRows * kHeadDim * 2;
// op_tp_norm slice: e4m3 [kHidden] then ue8m0 [kHidden / 32] of the row, then the bf16 norm
// weight and residual rows staged during the TP4 poll.
constexpr int kSmemNormRow = kSmemUnion;
constexpr int kSmemNormW = kSmemNormRow + 8192; // after the 6336-byte row slice
constexpr int kSmemNormRes = kSmemNormW + kHidden * 2;
constexpr int kSmemNormEnd = kSmemNormRes + kHidden * 2;
// op_gemm slice. Activation row (bytes); 16 mod 128. The activation scales are decoded by the
// loaders straight into the float-pair table at kSmemXf.
constexpr int kActRow = kHidden + 16;
constexpr int kSmemAct = kSmemUnion; // [kRows + 1][kActRow]; row kRows: zeros (padding token 7)
constexpr int kSmemRed = (kSmemAct + (kRows + 1) * kActRow + 15) / 16 * 16; // f32 [8 warps][32][4]
// float2 per group; +16 bytes keeps the four tig rows on distinct banks
constexpr int kXfStride = kMaxSfRow + 2;
constexpr int kSmemXf = kSmemRed + 2 * kConsumerWarps * 32 * 4 * 4; // float2 [4 tig][kXfStride]
constexpr int kSmemActEnd = kSmemXf + 4 * kXfStride * 8;
// op_combine slice.
constexpr int kSmemCombAcc = kSmemUnion; // f32 [8][kHeadDim]
constexpr int kSmemCombEnd = kSmemCombAcc + 8 * kHeadDim * 4;
constexpr int kSmemUnionEnd = kSmemAttnEnd > kSmemActEnd
                                  ? (kSmemAttnEnd > kSmemCombEnd ? kSmemAttnEnd : kSmemCombEnd)
                                  : (kSmemActEnd > kSmemCombEnd ? kSmemActEnd : kSmemCombEnd);
constexpr int kSmemMisc = (kSmemUnionEnd + 127) / 128 * 128;
static_assert(kSlotBytes % 1024 == 0 && kSlotW % 1024 == 0);
static_assert(kSmemNormEnd <= kSmemUnionEnd, "norm staging inside the union region");
static_assert(kSmemK % 16 == 0 && kSmemV % 16 == 0 && kSmemFresh % 16 == 0 && kSmemRed % 128 == 0 &&
              kSmemXf % 128 == 0);

struct SmemMisc {
  unsigned long long full[kRing];  // stage landed (TMA transaction bytes)
  unsigned long long empty[kRing]; // stage released by the eight consumer warps
  // the latency-critical inputs are on their way: start the weight stream
  unsigned long long go;
  int64_t slot_mapping[kRows];
  int64_t positions[kRows];
  int64_t input_ids[kRows];
  uint32_t epoch;
  int kv_len;
  float red[kConsumerWarps];
  float comb_l;
  float comb_stat[kMaxChunks][2];
  float comb_w[kMaxChunks];
  int fix[kChunk]; // block row whose fresh K/V replaces the window row, or -1
  // Layer-independent attention inputs, staged once per call (op_attention.cuh): the window slots
  // of this CTA's first chunk and the block row whose fresh K/V replaces each of them (or -1),
  // every layer's 1 + q/k norm weights and the block rows' cos | sin.
  int64_t win_slot[kChunk];
  int win_fix[kChunk];
  float qkw[kLayers][2][kHeadDim];
  float cs[kRows][kRotaryDim];
  int win_any[kChunk / 32]; // per warp of the staging threads: some win_fix row >= 0 (kept last:
                            // the float4-accessed rows above keep their alignment)
};
static_assert(kChunk / 32 == 2, "attention_phase reads win_any[0] and win_any[1]");
static_assert(offsetof(SmemMisc, cs) % 16 == 0 && offsetof(SmemMisc, qkw) % 16 == 0 &&
                  offsetof(SmemMisc, win_slot) % 8 == 0,
              "vector-accessed SmemMisc rows");
// + alignment of the dynamic base
constexpr int kSmemBytes = kSmemMisc + int(sizeof(SmemMisc)) + 1024;
static_assert(kSmemBytes <= 232448, "shared memory");

// Weight tensor maps: kMapsPerGemm slots per GEMM and layer, encoded and read only by that
// GEMM's OP (op_gemm_<name>.cuh).
constexpr int kMapsPerGemm = 2;
struct WeightMaps {
  CUtensorMap map[kLayers][kGemms][kMapsPerGemm];
};
struct PlanState {
  int grid;                                   // one CTA per SM
  uint8_t maps_raw[sizeof(WeightMaps) + 128]; // the maps, 128-byte aligned inside
};
static_assert(sizeof(PlanState) <= kPlanStateBytes, "plan state");

// GEMM shapes (the weights' K and output rows); how a GEMM cuts them into units and
// stages is private to its OP.
__host__ __device__ constexpr int gemm_k(int g) {
  return g == kGemmO ? kQSize : g == kGemmDown ? kInter : kHidden;
}
__host__ __device__ constexpr int gemm_n(int g) {
  return g == kGemmQkv ? kQkvRows : g == kGemmGateUp ? kGateUpRows : kHidden;
}

// CTA roles of the OPs, in one place: the producer's weight schedule and the consumer OPs
// derive their work from these, so a role change cannot desynchronize them.
// tp_norm / output: CTA t < kRows owns token row t.
__host__ __device__ constexpr bool owns_row(int cta) { return cta < kRows; }
// gemm: units are dealt from the last CTA backwards (the first CTAs run the row OPs); every
// GEMM keeps this dealing so the weight schedule and the consumers agree.
__host__ __device__ constexpr int first_gemm_unit(int cta, int grid) { return grid - 1 - cta; }
// The unit dealt last (the final unit of the CTA whose first unit is 0); polling gates wait on it.
__host__ __device__ constexpr int last_gemm_unit(int units, int grid) {
  return (units - 1) / grid * grid;
}
// Whether the CTA's last unit of a GEMM is in the GEMM's last wave: such a CTA finishes about when
// the GEMM's output is complete, so its consumers poll the output words directly (a gate would
// only add a round trip); CTAs that finish earlier gate first (no full poll while the last wave
// streams its weights).
__host__ __device__ constexpr bool in_last_wave(int cta, int grid, int units) {
  return first_gemm_unit(cta, grid) < units &&
         (units - 1 - first_gemm_unit(cta, grid)) / grid == (units - 1) / grid;
}
// attention: item i (chunk i / 2, head half i % 2) on CTA i mod grid (the entry prefetch warms
// only each CTA's first chunk, all of them while grid >= 2 * kMaxChunks); the block's cache write
// on CTA 0.
__host__ __device__ constexpr int first_attention_item(int cta) { return cta; }
__host__ __device__ constexpr int item_chunk(int item) { return item >> 1; }
__host__ __device__ constexpr int item_half(int item) { return item & 1; }
__host__ __device__ constexpr int first_attention_chunk(int cta) {
  return item_chunk(first_attention_item(cta));
}
__host__ __device__ constexpr bool writes_cache(int cta) { return cta == 0; }
// combine: query row r on CTA r mod grid.
__host__ __device__ constexpr int first_combine_row(int cta) { return cta; }

} // namespace
} // namespace m31_draft
