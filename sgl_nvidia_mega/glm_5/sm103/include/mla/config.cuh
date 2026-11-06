#pragma once

// Compile-time MLA layout and all persistent resource sizes.
// The argument ABI and fixed model dimensions live in args.h.
// These variant offsets preserve the validated shared-memory lifetimes.

#include <mla/args.h>

namespace glm5::mla {
inline constexpr int kThreads = 384;
inline constexpr int kConsumerThreads = 256;
inline constexpr int kOutputCtas = 128;
inline constexpr int kAttentionCtasPerToken = 16;
inline constexpr int kAttentionPasses =
    (kTokens == 8 && kTpWorldSize == 4) ? 1 : 2;
inline constexpr int kAttentionCtas = kTokens * kAttentionCtasPerToken;
inline constexpr int kMainCtas = kOutputCtas;
inline constexpr int kAttentionPartialCtas =
    kTokens == 8 ? kAttentionCtas * kAttentionPasses : kMainCtas;
inline constexpr int kAuxCtas = 20;
inline constexpr int kGridCtas = kMainCtas + kAuxCtas;
inline constexpr int kProjXActiveCtas = 82;
inline constexpr int kProjQActiveCtas = kQueryHeads * (kQkNopeDim + kRopeDim) / 32;
inline constexpr int kQAbsorbActiveCtas = kQueryHeads * kKvLatentDim / 64;
inline constexpr int kOAbsorbActiveCtas = kQueryHeads * kHeadDim / 32;

inline constexpr std::size_t kStaticSmemBytes = 0x4000;
// Attention and projection phases reuse shared storage.
inline constexpr std::size_t kDynamicSmemBytes = 0x32400;
inline constexpr std::size_t kMinimumWorkspaceBytes = 4 * 1024 * 1024;

// Partition each rank's mailbox by token, source rank, phase and CTA.
inline constexpr std::size_t kArCtaStride = 0x80;
inline constexpr std::size_t kArPhaseStride = 0x4000;
inline constexpr std::size_t kArSourceStride = 0x8000;
inline constexpr std::size_t kArTokenStride = kArSourceStride * (kTpWorldSize - 1);
inline constexpr std::size_t kPeerWorkspaceBytes = kTokens * kArTokenStride;

enum class Phase : int {
  kProjX = 0,
  kProjQ = 1,
  kRope = 2,
  kKv = 3,
  kQAbsorb = 4,
  kAttentionPartials = 5,
  kCombine = 6,
  kOAbsorb = 7,
  kCount = 8,
};

enum class RopeLayout : int {
  // Legacy ABI: [token, cos0, sin0, cos1, sin1, ...].
  kTokenInterleaved = 0,
  // SGLang ABI: [position, cos[0:32], sin[0:32]].
  kPositionCosSin = 1,
};

constexpr std::size_t align_up(std::size_t value, std::size_t alignment) {
  return (value + alignment - 1) / alignment * alignment;
}

// Store projection intermediates in token-major order.
struct WorkspaceLayout {
  static constexpr std::size_t kProjX = 0;
  static constexpr std::size_t kProjXBytes =
      kTokens * (2048 + kKvLatentDim + kRopeDim) * sizeof(std::uint16_t);

  static constexpr std::size_t kProjQ = align_up(kProjX + kProjXBytes, 128);
  static constexpr std::size_t kProjQBytes =
      kTokens * (kQueryHeads * kQkNopeDim + kQueryHeads * kRopeDim) * sizeof(std::uint16_t);

  static constexpr std::size_t kKvNorm = align_up(kProjQ + kProjQBytes, 128);
  static constexpr std::size_t kKvNormBytes = kTokens * kKvLatentDim * sizeof(std::uint16_t);

  static constexpr std::size_t kQAbsorb = align_up(kKvNorm + kKvNormBytes, 128);
  static constexpr std::size_t kQAbsorbBytes =
      kTokens * kQueryHeads * kKvLatentDim * sizeof(std::uint16_t);

  static constexpr std::size_t kPartialO = align_up(kQAbsorb + kQAbsorbBytes, 128);
  static constexpr std::size_t kPartialOBytes =
      kAttentionPartialCtas * kQueryHeads * kValueDim * sizeof(std::uint16_t);

  static constexpr std::size_t kPartialLse = align_up(kPartialO + kPartialOBytes, 128);
  static constexpr std::size_t kPartialLseBytes = kAttentionPartialCtas * kQueryHeads * sizeof(float);

  static constexpr std::size_t kCombined = align_up(kPartialLse + kPartialLseBytes, 128);
  static constexpr std::size_t kCombinedBytes =
      kTokens * kQueryHeads * kValueDim * sizeof(std::uint16_t);

  static constexpr std::size_t kOAbsorb = align_up(kCombined + kCombinedBytes, 128);
  static constexpr std::size_t kOAbsorbBytes =
      kTokens * kQueryHeads * kHeadDim * sizeof(std::uint16_t);

  static constexpr std::size_t kPhaseStates = align_up(kOAbsorb + kOAbsorbBytes, 128);
  static constexpr std::size_t kPhaseStateBytes =
      static_cast<int>(Phase::kCount) * sizeof(std::uint64_t);
  // Keep attention completion state separate from ordinary phase counters.
  static constexpr std::size_t kAttentionPartialAsyncState = kPhaseStates + kPhaseStateBytes;
  static constexpr std::size_t kAttentionPartialAsyncStateBytes = 16;
  static constexpr std::size_t kEnd =
      align_up(kAttentionPartialAsyncState + kAttentionPartialAsyncStateBytes, 128);
};

static_assert(WorkspaceLayout::kAttentionPartialAsyncState % 16 == 0);
static_assert(WorkspaceLayout::kEnd <= kMinimumWorkspaceBytes);
static_assert(kArCtaStride * kOutputCtas == kArPhaseStride);
static_assert(kArPhaseStride * 2 == kArSourceStride);
static_assert(kArSourceStride * (kTpWorldSize - 1) == kArTokenStride);


}  // namespace glm5::mla

#include <mla/pipeline.cuh>

namespace glm5::mla::detail {

#if MEGAMLA_USE_NATIVE_BF16X2_E4M3X2_CVT
using Hmma = HmmaM16N8K16F32Bf16;
#else
using Hmma = HmmaM16N8K16F32F16;
#endif
using InternalActivation = Hmma::Element;
static_assert(sizeof(InternalActivation) == sizeof(__nv_bfloat16));

inline constexpr int kProjXRows = 2048 + kKvLatentDim + kRopeDim;
inline constexpr int kProjQRows = kQueryHeads * kQkNopeDim + kQueryHeads * kRopeDim;
#if GLM5_MEGAMLA_TOKENS == 4
inline constexpr int kKvAuxCtaBegin = 128;
#elif GLM5_MEGAMLA_TOKENS == 8
inline constexpr int kKvAuxCtaBegin = kMainCtas;
#endif
#if GLM5_MEGAMLA_TOKENS == 4
inline constexpr int kKvAuxCtaEnd = 132;
#elif GLM5_MEGAMLA_TOKENS == 8
inline constexpr int kKvAuxCtaEnd = kKvAuxCtaBegin + kTokens;
#endif
#if GLM5_MEGAMLA_TOKENS == 4
inline constexpr int kRopeAuxCtaBegin = 132;
#elif GLM5_MEGAMLA_TOKENS == 8
inline constexpr int kRopeAuxCtaBegin = kKvAuxCtaEnd;
#endif
#if GLM5_MEGAMLA_TOKENS == 4
inline constexpr int kRopeAuxCtaEnd = 136;
#elif GLM5_MEGAMLA_TOKENS == 8
inline constexpr int kRopeAuxCtaEnd = kRopeAuxCtaBegin + kTokens;
#endif
inline constexpr int kProducerThreads = kThreads - kConsumerThreads;
inline constexpr int kProducerWarps = kProducerThreads / 32;
inline constexpr int kAttentionProducerWarpBegin = kConsumerThreads / 32;
#if GLM5_MEGAMLA_TOKENS == 4
inline constexpr int kAttentionQkWarps = kTpWorldSize == 4 ? 8 : 4;
#elif GLM5_MEGAMLA_TOKENS == 8
inline constexpr int kAttentionQkWarps = kTpWorldSize == 4 ? 8 : 4;
#endif
inline constexpr int kAttentionQkThreads = kAttentionQkWarps * 32;
inline constexpr int kAttentionStageBytes = 0x9000;
inline constexpr int kAttentionStagePayloadBytes = 0x8000;
#if GLM5_MEGAMLA_TOKENS == 4
inline constexpr int kAttentionKvStages = kFp8KvCache ? 1 : 2;
#elif GLM5_MEGAMLA_TOKENS == 8
inline constexpr int kAttentionKvStages = kTpWorldSize == 4 ? 2 : 1;
#endif
inline constexpr int kAttentionBarrierOffset = 0x3900;
inline constexpr int kAttentionPartialContributionOffset = 0x39a0;
inline constexpr int kProjectionRmsScratchOffset = 0x0;
inline constexpr int kGenerationScratchOffset = 0x100;
inline constexpr int kProjectionActivationOffset = 0x24000;
// Pad activation rows to avoid bank conflicts during FP8 matrix loads.
inline constexpr int kProjXActivationRowPadding = 16;
inline constexpr int kProjQActivationRowPadding = 16;
inline constexpr int kAbsorbActivationRowPadding =
    kTokens == 8 && kTpWorldSize == 4 ? 8 : 0;
inline constexpr int kProjectionPartialTokenPadding =
    kTokens == 8 && kTpWorldSize == 4 ? 4 : 0;
inline constexpr int kProjXReducedOffset =
    kTokens == 8 && kTpWorldSize == 4 ? 0x2e000 : 0x30080;
#if GLM5_MEGAMLA_TOKENS == 4
inline constexpr int kProjXPartialsOffset = 0x30180;
#elif GLM5_MEGAMLA_TOKENS == 8
inline constexpr int kProjXPartialsOffset = kTpWorldSize == 4 ? 0x2e200 : 0x30280;
#endif
#if GLM5_MEGAMLA_TOKENS == 4
inline constexpr int kProjQReducedOffset = 0x28080;
#elif GLM5_MEGAMLA_TOKENS == 8
inline constexpr int kProjQReducedOffset = 0x2c100;
#endif
#if GLM5_MEGAMLA_TOKENS == 4
inline constexpr int kProjQPartialsOffset = 0x28180;
#elif GLM5_MEGAMLA_TOKENS == 8
inline constexpr int kProjQPartialsOffset = 0x2c300;
#endif
#if GLM5_MEGAMLA_TOKENS == 4
inline constexpr int kQAbsorbScratchOffset = 0x24600;
#elif GLM5_MEGAMLA_TOKENS == 8
inline constexpr int kQAbsorbScratchOffset = kTpWorldSize == 4 ? 0x24d00 : 0x24c00;
#endif
inline constexpr int kCombineWeightsOffset = 0x2b800;
inline constexpr int kCombineSplitsPerStage = 16;
inline constexpr int kCombineElementsPerSplit = kValueDim;
#if GLM5_MEGAMLA_TOKENS == 4
inline constexpr int kOAbsorbReducedOffset = 0x25000;
#elif GLM5_MEGAMLA_TOKENS == 8
inline constexpr int kOAbsorbReducedOffset = kTpWorldSize == 4 ? 0x26100 : 0x26000;
#endif
#if GLM5_MEGAMLA_TOKENS == 4
inline constexpr int kOAbsorbPartialsOffset = 0x25100;
#elif GLM5_MEGAMLA_TOKENS == 8
inline constexpr int kOAbsorbPartialsOffset = kTpWorldSize == 4 ? 0x26400 : 0x26200;
#endif
#if GLM5_MEGAMLA_TOKENS == 4
inline constexpr int kOProjResidualOffset = kTpWorldSize == 4 ? 0x2c000 : 0x28000;
#elif GLM5_MEGAMLA_TOKENS == 8
inline constexpr int kOProjResidualOffset = 0x2c000;
#endif
#if GLM5_MEGAMLA_TOKENS == 4
inline constexpr int kOProjReducedOffset = kTpWorldSize == 4 ? 0x2c180 : 0x28180;
#elif GLM5_MEGAMLA_TOKENS == 8
inline constexpr int kOProjReducedOffset = 0x2c300;
#endif
#if GLM5_MEGAMLA_TOKENS == 4
inline constexpr int kOProjPartialsOffset = kTpWorldSize == 4 ? 0x2c300 : 0x28300;
#elif GLM5_MEGAMLA_TOKENS == 8
inline constexpr int kOProjPartialsOffset = 0x2c600;
#endif
inline constexpr int kOProjPages = kQueryHeads * kHeadDim / 512;
inline constexpr int kAttentionQueryLatentOffset = 0x24000;
inline constexpr int kAttentionPartialOStagingOffset = kAttentionQueryLatentOffset;
inline constexpr int kAttentionQueryPeOffset = 0x28000;
inline constexpr int kAttentionKeyPeOffset = 0x28800;
inline constexpr int kAttentionProbabilityOffset = 0x2a800;
#if GLM5_MEGAMLA_TOKENS == 4
inline constexpr int kAttentionSelectedOffset = 0x2b000;
#elif GLM5_MEGAMLA_TOKENS == 8
inline constexpr int kAttentionSelectedOffset = kTpWorldSize == 4 ? 0x2e800 : 0x2e400;
#endif
inline constexpr int kAttentionSoftmaxScratchOffset =
    kTokens == 8 && kTpWorldSize == 4 ? 0x2b800 : 0x2b400;
inline constexpr int kAttentionSoftmaxWarpStride = 0x40 / sizeof(float);
inline constexpr int kAttentionSoftmaxHeadStride = 0x8 / sizeof(float);
#if GLM5_MEGAMLA_TOKENS == 4
inline constexpr int kAttentionLseStagingOffset = kTpWorldSize == 4 ? 0x2b600 : 0x2b500;
#elif GLM5_MEGAMLA_TOKENS == 8
inline constexpr int kAttentionLseStagingOffset = kTpWorldSize == 4 ? 0x2ba00 : 0x2b500;
#endif
#if GLM5_MEGAMLA_TOKENS == 4
inline constexpr int kAttentionScratchEnd = 0x2b800;
#elif GLM5_MEGAMLA_TOKENS == 8
inline constexpr int kAttentionScratchEnd = kTpWorldSize == 4 ? 0x2ea00 : 0x2e500;
#endif
inline constexpr int kAttentionPartialOBytes =
    kQueryHeads * kValueDim * static_cast<int>(sizeof(InternalActivation));
inline constexpr int kAttentionPartialLseBytes = kQueryHeads * static_cast<int>(sizeof(float));
#if GLM5_MEGAMLA_TOKENS == 4
inline constexpr int kAttentionGatherTransactionBytes = kFp8KvCache ? 0x8000 : 0x10000;
#elif GLM5_MEGAMLA_TOKENS == 8
inline constexpr int kAttentionGatherTransactionBytes = kTpWorldSize == 4 ? 0x10000 : 0x8000;
#endif
inline constexpr int kGatherRows = 4;
#if GLM5_MEGAMLA_TOKENS == 4
inline constexpr int kGatherRowGroupsPerProducer = 4;
#elif GLM5_MEGAMLA_TOKENS == 8
inline constexpr int kGatherRowGroupsPerProducer = kTpWorldSize == 4 ? 8 : 4;
#endif
#if GLM5_MEGAMLA_TOKENS == 4
inline constexpr int kGatherColumns = kFp8KvCache ? 128 : 64;
#elif GLM5_MEGAMLA_TOKENS == 8
inline constexpr int kGatherColumns = 128;
#endif
inline constexpr int kGatherChunksPerKvStage =
    kTokens == 8 && kTpWorldSize == 4 ? 2 : 4;
inline constexpr int kGatherChunkBytes = kAttentionStagePayloadBytes / kGatherChunksPerKvStage;
inline constexpr int kGatherChunks = kAttentionKvStages * kGatherChunksPerKvStage;
inline constexpr int kGatherRowBytes = kGatherColumns * kKvCacheElementBytes;
inline constexpr int kGatherInstructionBytes = kGatherRows * kGatherRowBytes;
inline constexpr int kGatherRowGroupBytes = kProducerWarps * kGatherInstructionBytes;
inline constexpr int kGatherInstructionsPerProducer = kGatherRowGroupsPerProducer * kGatherChunks;
inline constexpr int kGatherInstructionsPerCta = kProducerWarps * kGatherInstructionsPerProducer;
inline constexpr int kProjPageK = 1024;
inline constexpr int kProjPageKTiles = kProjPageK / 16;
inline constexpr int kProjMatrixBytes = 32768;
inline constexpr int kProjPageBytes = 32896;
inline constexpr int kQAbsorbMatrixBytes = 64 * kQkNopeDim;
inline constexpr int kQAbsorbPageBytes = 12416;
inline constexpr int kOAbsorbMatrixBytes = 32 * kValueDim;
inline constexpr int kOAbsorbPageBytes = 16512;
inline constexpr int kOProjMatrixBytes = 48 * 512;
inline constexpr int kOProjPageBytes = 24704;


// The active JIT translation unit instantiates exactly one token/TP pair.
// Keep actual Attention scheduling explicit: MTP314 uses 32 CTAs per token.
template <int Tokens, int Tp>
struct MlaConfig {
  static constexpr std::size_t kWorkspaceRequiredBytes = kMinimumWorkspaceBytes;
  static constexpr std::size_t kPeerBytes = Tokens * 0x8000 * (Tp - 1);
  static constexpr int kWorkspaceFill = 0;
  static_assert(Tokens == 4 || Tokens == 8);
  static_assert(Tp == 4 || Tp == 8);
  static constexpr int kTokens = Tokens;
  static constexpr int kTpSize = Tp;
  static constexpr int kHeads = 64 / Tp;
  static constexpr int kMainCtas = 128;
  static constexpr int kOutputCtas = 128;
  static constexpr int kGridCtas = 148;
  static constexpr int kEpochCount = kGridCtas;
  static constexpr int kRequiredSms = kGridCtas;
  static constexpr int kAttentionCtasPerToken = Tokens == 4 ? 32 : 16;
  static constexpr int kKvPerPass = Tokens == 8 && Tp == 4 ? 128 : 64;
  static constexpr int kAttentionPasses = Tokens == 8 && Tp == 8 ? 2 : 1;
  static constexpr int kPartialSplitsPerToken = kAttentionCtasPerToken * kAttentionPasses;
  static constexpr bool kSharedProjX = Tokens == 8;
  static constexpr bool kStaticOProjTail = Tokens == 8 && Tp == 4;
  static constexpr int kCombineUnroll = Tokens == 8 && Tp == 4 ? 4 : 32;
  static constexpr int kOProjLocalStaticOffset = 0x2000;
  static constexpr int kOProjResidualDynamicOffset = 0x32000;
  static constexpr int kOProjPartialStaticOffset = 0x2300;
  static constexpr int kOProjWarpPartialStride = Tokens * 20 + 1;
};

// Validated stage lifetimes and resource limits for the template specializations.
#if GLM5_MEGAMLA_TOKENS == 4 || GLM5_MEGAMLA_TOKENS == 8
#if GLM5_MEGAMLA_TOKENS == 4
static_assert(kProducerWarps == 4);
static_assert(kGatherChunks * kGatherColumns == kKvLatentDim);
static_assert(kGatherChunkBytes == 0x2000);
static_assert(kGatherRowBytes == 0x80);
static_assert(kGatherInstructionBytes == 0x200);
static_assert(kGatherRowGroupBytes == 0x800);
static_assert(kGatherInstructionsPerProducer == (kFp8KvCache ? 16 : 32));
static_assert(kGatherInstructionsPerCta == (kFp8KvCache ? 64 : 128));
static_assert(kAttentionGatherTransactionBytes ==
              kGatherInstructionsPerCta * kGatherInstructionBytes);
static_assert(4 * kAttentionStageBytes == kAttentionQueryLatentOffset);
static_assert(kAttentionPartialOStagingOffset + kAttentionPartialOBytes <= kAttentionQueryPeOffset);
static_assert(kAttentionPartialOStagingOffset % 16 == 0);
static_assert(kAttentionPartialOBytes % 16 == 0);
static_assert(kAttentionSoftmaxScratchOffset + kAttentionQkWarps * kAttentionSoftmaxWarpStride *
                                                   static_cast<int>(sizeof(float)) ==
              kAttentionLseStagingOffset);
static_assert(kAttentionLseStagingOffset % 16 == 0);
static_assert(kAttentionPartialLseBytes % 16 == 0);
static_assert(kAttentionLseStagingOffset + kAttentionPartialLseBytes <= kAttentionScratchEnd);
static_assert(kAttentionQueryLatentOffset % 1024 == 0);
static_assert(kAttentionQueryPeOffset % 1024 == 0);
static_assert(kAttentionKeyPeOffset % 1024 == 0);
static_assert(kAttentionProbabilityOffset % 1024 == 0);
static_assert(kAttentionSelectedOffset % 1024 == 0);
static_assert(kAttentionScratchEnd <= kDynamicSmemBytes);
static_assert(kProjectionActivationOffset + kTokens * (kHidden + kProjXActivationRowPadding) *
                                                static_cast<int>(sizeof(InternalActivation)) <=
              kProjXReducedOffset);
static_assert(kProjectionActivationOffset + kTokens * (2048 + kProjQActivationRowPadding) *
                                                static_cast<int>(sizeof(InternalActivation)) ==
              kProjQReducedOffset);
static_assert(kProjXPartialsOffset + 8 * 0x404 <= kDynamicSmemBytes);
static_assert(2 * kCombineSplitsPerStage * kCombineElementsPerSplit *
                  static_cast<int>(sizeof(InternalActivation)) ==
              kAttentionStagePayloadBytes);
static_assert(kCombineWeightsOffset + 32 * sizeof(float) <= kDynamicSmemBytes);
static_assert(kOProjPartialsOffset + 8 * 0x304 <= kDynamicSmemBytes);
static_assert(kAttentionBarrierOffset + sizeof(PipelineControl) <= kStaticSmemBytes);
static_assert(kAttentionBarrierOffset + sizeof(PipelineControl) ==
              kAttentionPartialContributionOffset);
static_assert(kAttentionPartialContributionOffset + sizeof(uint4) <= kStaticSmemBytes);
static_assert(kGenerationScratchOffset + sizeof(std::uint32_t) < kAttentionBarrierOffset);

#elif GLM5_MEGAMLA_TP_SIZE == 4
static_assert(kProducerWarps == 4);
static_assert(kGatherChunks * kGatherColumns == kKvLatentDim);
static_assert(kGatherChunkBytes == 0x4000);
static_assert(kGatherRowBytes == 0x80);
static_assert(kGatherInstructionBytes == 0x200);
static_assert(kGatherRowGroupBytes == 0x800);
static_assert(kGatherInstructionsPerProducer == 32);
static_assert(kGatherInstructionsPerCta == 128);
static_assert(kAttentionGatherTransactionBytes ==
              kGatherInstructionsPerCta * kGatherInstructionBytes);
static_assert(4 * kAttentionStageBytes == kAttentionQueryLatentOffset);
static_assert(kAttentionPartialOStagingOffset + kAttentionPartialOBytes <= kAttentionQueryPeOffset);
static_assert(kAttentionPartialOStagingOffset % 16 == 0);
static_assert(kAttentionPartialOBytes % 16 == 0);
static_assert(kAttentionSoftmaxScratchOffset + kAttentionQkWarps * kAttentionSoftmaxWarpStride *
                                                   static_cast<int>(sizeof(float)) ==
              kAttentionLseStagingOffset);
static_assert(kAttentionLseStagingOffset % 16 == 0);
static_assert(kAttentionPartialLseBytes % 16 == 0);
static_assert(kAttentionLseStagingOffset + kAttentionPartialLseBytes <= kAttentionScratchEnd);
static_assert(kAttentionQueryLatentOffset % 1024 == 0);
static_assert(kAttentionQueryPeOffset % 1024 == 0);
static_assert(kAttentionKeyPeOffset % 1024 == 0);
static_assert(kAttentionProbabilityOffset % 1024 == 0);
static_assert(kAttentionSelectedOffset % 1024 == 0);
static_assert(kProjQPartialsOffset +
                  8 * (2 * kTokens * 16 + kTokens * kProjectionPartialTokenPadding + 1) *
                      sizeof(float) <=
              kAttentionSelectedOffset);
static_assert(kAttentionSelectedOffset + 128 * sizeof(std::int32_t) <= kAttentionScratchEnd);
static_assert(kProjXReducedOffset + kTokens * 32 * sizeof(InternalActivation) <=
              kProjXPartialsOffset);
static_assert(kProjQReducedOffset + kTokens * 32 * sizeof(InternalActivation) <=
              kProjQPartialsOffset);
static_assert(kProjectionActivationOffset + kTokens * (kQkNopeDim + kAbsorbActivationRowPadding) *
                                                sizeof(InternalActivation) <=
              kQAbsorbScratchOffset);
static_assert(kProjectionActivationOffset + kTokens * (kValueDim + kAbsorbActivationRowPadding) *
                                                sizeof(InternalActivation) <=
              kOAbsorbReducedOffset);
static_assert(kOAbsorbReducedOffset + kTokens * 32 * sizeof(InternalActivation) <=
              kOAbsorbPartialsOffset);
static_assert(kProjectionActivationOffset + kTokens * 512 * sizeof(InternalActivation) <=
              kOProjResidualOffset);
static_assert(kOProjResidualOffset + kTokens * 48 * sizeof(__nv_bfloat16) <= kOProjReducedOffset);
static_assert(kOProjReducedOffset + kTokens * 48 * sizeof(__nv_bfloat16) <= kOProjPartialsOffset);
static_assert(kAttentionScratchEnd <= kDynamicSmemBytes);
static_assert(kProjectionActivationOffset +
                  kTokens * (1024 + kProjXActivationRowPadding) * sizeof(InternalActivation) <=
              kProjXReducedOffset);
static_assert(kProjectionActivationOffset +
                  kTokens * (2048 + kProjQActivationRowPadding) * sizeof(InternalActivation) <=
              kProjQReducedOffset);
static_assert(kProjXPartialsOffset +
                  8 * (2 * 128 + kTokens * kProjectionPartialTokenPadding + 1) * sizeof(float) <=
              kDynamicSmemBytes);
static_assert(kProjQPartialsOffset +
                  8 * (2 * kTokens * 16 + kTokens * kProjectionPartialTokenPadding + 1) *
                      sizeof(float) <=
              kDynamicSmemBytes);
static_assert(kQAbsorbScratchOffset + kTokens * 64 * sizeof(InternalActivation) <=
              kDynamicSmemBytes);
static_assert(kOAbsorbPartialsOffset +
                  8 * (2 * kTokens * 16 + kTokens * kProjectionPartialTokenPadding + 1) *
                      sizeof(float) <=
              kDynamicSmemBytes);
static_assert(2 * kCombineSplitsPerStage * kCombineElementsPerSplit *
                  static_cast<int>(sizeof(InternalActivation)) ==
              kAttentionStagePayloadBytes);
static_assert(kCombineWeightsOffset + 32 * sizeof(float) <= kDynamicSmemBytes);
static_assert(kOProjPartialsOffset + 8 * (3 * kTokens * 16 + 1) * sizeof(float) <=
              kDynamicSmemBytes);
static_assert(kAttentionBarrierOffset + sizeof(PipelineControl) <= kStaticSmemBytes);
static_assert(kAttentionBarrierOffset + sizeof(PipelineControl) ==
              kAttentionPartialContributionOffset);
static_assert(kAttentionPartialContributionOffset + sizeof(uint4) <= kStaticSmemBytes);
static_assert(kGenerationScratchOffset + sizeof(std::uint32_t) < kAttentionBarrierOffset);

static_assert(kTokens == 8 && kQueryHeads * kHeadDim == 4096);
static_assert(kProjectionActivationOffset + (kTokens - 1) * 4096 * sizeof(InternalActivation) <=
              kDynamicSmemBytes);
static_assert(4096 * sizeof(InternalActivation) ==
              MlaConfig<kTokens, kTpWorldSize>::kOProjLocalStaticOffset);
static_assert(MlaConfig<kTokens, kTpWorldSize>::kOProjLocalStaticOffset +
                  kTokens * 48 * sizeof(__nv_bfloat16) ==
              MlaConfig<kTokens, kTpWorldSize>::kOProjPartialStaticOffset);
static_assert(MlaConfig<kTokens, kTpWorldSize>::kOProjResidualDynamicOffset +
                  kTokens * 48 * sizeof(__nv_bfloat16) <=
              kDynamicSmemBytes);
static_assert(MlaConfig<kTokens, kTpWorldSize>::kOProjPartialStaticOffset +
                  8 * MlaConfig<kTokens, kTpWorldSize>::kOProjWarpPartialStride * sizeof(float) <=
              kAttentionBarrierOffset);

#else
static_assert(kProducerWarps == 4);
static_assert(kGatherChunks * kGatherColumns == kKvLatentDim);
static_assert(kGatherChunkBytes == 0x2000);
static_assert(kGatherRowBytes == 0x80);
static_assert(kGatherInstructionBytes == 0x200);
static_assert(kGatherRowGroupBytes == 0x800);
static_assert(kGatherInstructionsPerProducer == 16);
static_assert(kGatherInstructionsPerCta == 64);
static_assert(kAttentionGatherTransactionBytes ==
              kGatherInstructionsPerCta * kGatherInstructionBytes);
static_assert(4 * kAttentionStageBytes == kAttentionQueryLatentOffset);
static_assert(kAttentionPartialOStagingOffset + kAttentionPartialOBytes <= kAttentionQueryPeOffset);
static_assert(kAttentionPartialOStagingOffset % 16 == 0);
static_assert(kAttentionPartialOBytes % 16 == 0);
static_assert(kAttentionSoftmaxScratchOffset + kAttentionQkWarps * kAttentionSoftmaxWarpStride *
                                                   static_cast<int>(sizeof(float)) ==
              kAttentionLseStagingOffset);
static_assert(kAttentionLseStagingOffset % 16 == 0);
static_assert(kAttentionPartialLseBytes % 16 == 0);
static_assert(kAttentionLseStagingOffset + kAttentionPartialLseBytes <= kAttentionScratchEnd);
static_assert(kAttentionQueryLatentOffset % 1024 == 0);
static_assert(kAttentionQueryPeOffset % 1024 == 0);
static_assert(kAttentionKeyPeOffset % 1024 == 0);
static_assert(kAttentionProbabilityOffset % 1024 == 0);
static_assert(kAttentionSelectedOffset % 1024 == 0);
static_assert(kProjQPartialsOffset + 8 * (2 * kTokens * 16 + 1) * sizeof(float) <=
              kAttentionSelectedOffset);
static_assert(kAttentionSelectedOffset + 64 * sizeof(std::int32_t) <= kAttentionScratchEnd);
static_assert(kProjXReducedOffset + kTokens * 32 * sizeof(InternalActivation) <=
              kProjXPartialsOffset);
static_assert(kProjQReducedOffset + kTokens * 32 * sizeof(InternalActivation) <=
              kProjQPartialsOffset);
static_assert(kProjectionActivationOffset + kTokens * kQkNopeDim * sizeof(InternalActivation) <=
              kQAbsorbScratchOffset);
static_assert(kProjectionActivationOffset + kTokens * kValueDim * sizeof(InternalActivation) <=
              kOAbsorbReducedOffset);
static_assert(kOAbsorbReducedOffset + kTokens * 32 * sizeof(InternalActivation) <=
              kOAbsorbPartialsOffset);
static_assert(kProjectionActivationOffset +
                  kTokens * kQueryHeads * kHeadDim * sizeof(InternalActivation) <=
              kOProjResidualOffset);
static_assert(kOProjResidualOffset + kTokens * 48 * sizeof(__nv_bfloat16) <= kOProjReducedOffset);
static_assert(kOProjReducedOffset + kTokens * 48 * sizeof(__nv_bfloat16) <= kOProjPartialsOffset);
static_assert(kAttentionScratchEnd <= kDynamicSmemBytes);
static_assert(kProjectionActivationOffset +
                  kTokens * (1024 + kProjXActivationRowPadding) * sizeof(InternalActivation) <=
              kProjXReducedOffset);
static_assert(kProjectionActivationOffset +
                  kTokens * (2048 + kProjQActivationRowPadding) * sizeof(InternalActivation) <=
              kProjQReducedOffset);
static_assert(kProjXPartialsOffset + 8 * (2 * 128 + 1) * sizeof(float) <= kDynamicSmemBytes);
static_assert(kProjQPartialsOffset + 8 * (2 * kTokens * 16 + 1) * sizeof(float) <=
              kDynamicSmemBytes);
static_assert(kQAbsorbScratchOffset + kTokens * 64 * sizeof(InternalActivation) <=
              kDynamicSmemBytes);
static_assert(kOAbsorbPartialsOffset + 8 * (2 * kTokens * 16 + 1) * sizeof(float) <=
              kDynamicSmemBytes);
static_assert(2 * kCombineSplitsPerStage * kCombineElementsPerSplit *
                  static_cast<int>(sizeof(InternalActivation)) ==
              kAttentionStagePayloadBytes);
static_assert(kCombineWeightsOffset + 32 * sizeof(float) <= kDynamicSmemBytes);
static_assert(kOProjPartialsOffset + 8 * (3 * kTokens * 16 + 1) * sizeof(float) <=
              kDynamicSmemBytes);
static_assert(kAttentionBarrierOffset + sizeof(PipelineControl) <= kStaticSmemBytes);
static_assert(kAttentionBarrierOffset + sizeof(PipelineControl) ==
              kAttentionPartialContributionOffset);
static_assert(kAttentionPartialContributionOffset + sizeof(uint4) <= kStaticSmemBytes);
static_assert(kGenerationScratchOffset + sizeof(std::uint32_t) < kAttentionBarrierOffset);

#endif
#endif

}  // namespace glm5::mla::detail
