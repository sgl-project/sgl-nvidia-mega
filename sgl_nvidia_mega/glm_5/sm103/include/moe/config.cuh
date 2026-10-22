#pragma once

// GLM-5.3 native FP8 configuration. One JIT translation unit binds one Config.
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cstddef>
#include <cstdint>
#include <common/hmma_b16.cuh>
#include <moe/pipeline.cuh>
#include <comm/oneshot_allreduce.cuh>

namespace glm5::moe::detail {

template <int Tokens, int Tp>
struct MoeCommonConfig {
  static_assert(Tokens == 4 || Tokens == 8);
  static_assert(Tp == 4 || Tp == 8);
  static constexpr int kArElementsPerVirtualTile = 48;
  static constexpr int kArMaxVirtualTilesPerCta = 8;
  static constexpr std::size_t kArPayloadBytes = 0x180;
  static constexpr int kArPayloadRowsPerColumn = 6;
  static constexpr std::size_t kArPhaseStride = 0x4000;
  static constexpr int kArReductionLanesPerWarp = 12;
  static constexpr int kArReductionValuesPerThread = 4;
  static constexpr int kArReductionWarps = 4;
  static constexpr int kArRemoteSources = Tp - 1;
  static constexpr std::size_t kArSourceStride = 0x8000;
  static constexpr int kArTileElements = 192;
  static constexpr std::size_t kArTileStride = 0x200;
  static constexpr int kArTilesPerToken = 32;
  static constexpr std::size_t kArTokenStride = (Tp - 1) * 0x8000;
  static constexpr int kArValuesPerVector = 8;
  static constexpr int kArVectorBytes = 16;
  static constexpr int kArVectorsPerColumn = 8;
  static constexpr std::size_t kArVirtualTileStride = 0x80;
  static constexpr int kArVirtualTilesPerPass = 4;
  static constexpr std::uint16_t kBf16NotReady = 0xffffU;
  static constexpr int kConsumerThreads = 256;
  static constexpr int kDownActivationSlots = 9;
  static constexpr int kDownKPerTask = 256;
  static constexpr int kDownM16PerVirtualTile = 3;
  static constexpr int kDownM16Rows = 16;
  static constexpr int kDownMaxM16PerCta = 24;
  static constexpr int kDownNativeTransactionBytes = 0x3020;
  static constexpr int kDownResetProducerWarp = 1;
  static constexpr int kDownVirtualTilesPerCta = 4;
  static constexpr int kDownWeightTmaBytes = 0x3000;
  static constexpr int kDynamicSmemBytes = 0x34400;
  static constexpr float kEps = 1.0e-5F;
  static constexpr unsigned kFullWarpMask = 0xffffffffU;
  static constexpr std::size_t kGateUpCachePolicyBytes = sizeof(std::uint64_t);
  static constexpr int kGateUpL2EvictLastSlots = 2;
  static constexpr int kGateUpL2RetainedExpertMask = 3;
  static constexpr int kGateUpPagesPerGroup = 6;
  static constexpr std::size_t kGateUpScaleOffset = 0x8000;
  static constexpr int kGateUpScratchElements = Tokens == 4 ? 128 : (Tp == 4 ? 64 : 32);
  static constexpr int kGateUpTmaKAtomsPerChunk = 2;
  static constexpr std::uint32_t kGlobalNotReady = 0xffffffffU;
  static constexpr std::uint32_t kGlobalReady = 0U;
  static constexpr int kGridCtas = Tokens == 4 ? 144 : 148;
  static constexpr int kHidden = 6144;
  static constexpr int kLocalInter = 2048 / Tp;
  static constexpr float kNegativeFloatMax = -3.402823466e+38F;
  static constexpr std::size_t kNormOffset = 0x00000;
  static constexpr int kPipelineStages = 4;
  static constexpr int kProducerThreads = 128;
  static constexpr int kRequiredSms = 148;
  static constexpr std::size_t kWorkspaceCapacityBytes = 4 * 1024 * 1024;
  static constexpr int kRmsGammaTransactionBytes = kHidden * sizeof(NormGain);
  static constexpr int kRmsInputCopiesPerConsumer = 3;
  static constexpr int kRmsInputCopyBytes = 16;
  static constexpr float kRouteScale = 2.5F;
  static constexpr int kRoutedCtasPerToken = Tokens == 4 ? 32 : 18;
  static constexpr int kRoutedExperts = 256;
  static constexpr int kRouterExpertsPerCta = 8;
  static constexpr int kRouterMaxExpertsPerCta = 15;
  static constexpr int kRouterPipelineTaskBase = 1;
  static constexpr int kSharedActivationPageBuffers = 2;
  static constexpr int kSharedFollowerGateUpPipelineTaskBase = 0;
  static constexpr int kSharedOwnerGateUpPipelineTaskBase = 1;
  static constexpr int kSharedTokensPerCta = 2;
  static constexpr std::size_t kStageBytes = 0x9000;
  static constexpr int kStaticSmemBytes = 0x3c00;
  static constexpr std::size_t kStorageControlBytes =
      Tokens == 4 ? 0x400 : (Tp == 4 ? 0x500 : 0x580);
  static constexpr int kThreads = 384;
  static constexpr int kTmaAtomK = 128;
  static constexpr int kTokens = Tokens;
  static constexpr int kTopK = 8;
  static constexpr bool kTopkFrequency = Tokens == 4 && Tp == 4;
  static constexpr std::size_t kTopkHandoffBytes = sizeof(std::uint64_t);
  static constexpr int kTpSize = Tp;
  static constexpr int kWarpSize = 32;
  static constexpr int kArMaxTileElements = kArMaxVirtualTilesPerCta * kArElementsPerVirtualTile;
  static constexpr int kArPayloadVectors = kArTileElements / kArValuesPerVector;
  static constexpr int kArPayloadVectorsPerProducer = kWarpSize / kArValuesPerVector;
  static constexpr int kArProducerWarps = kArTileElements / kWarpSize;
  static constexpr int kConsumerWarps = kConsumerThreads / kWarpSize;
  static constexpr int kDownActivationBytes =
      kDownActivationSlots * kLocalInter * sizeof(__nv_bfloat16);
  static constexpr int kDownExpertPairs = (kDownActivationSlots + 1) / 2;
  static constexpr int kDownK32Partitions = kLocalInter / 32;
  static constexpr int kDownK32PartitionsPerTask = kDownKPerTask / 32;
  static constexpr int kDownKTasks = kLocalInter / kDownKPerTask;
  static constexpr int kDownM16Tiles = kHidden / kDownM16Rows;
  static constexpr int kDownPhysicalTmaKAtoms = kLocalInter / kTmaAtomK;
  static constexpr int kDownScaleBlocks = kLocalInter / kTmaAtomK;
  static constexpr int kDownStagedTmaKAtoms = kDownKPerTask / kTmaAtomK;
  static constexpr std::size_t kDownTmaAtomTileBytes =
      kDownM16PerVirtualTile * kDownM16Rows * kTmaAtomK;
  static constexpr int kDownTmaKAtoms = kLocalInter / kTmaAtomK;
  static constexpr int kGateUpLayoutFactor = kLocalInter / 256;
  static constexpr int kGateUpRowGroupsPerExpert = kLocalInter / 16;
  static constexpr int kGateUpScaleBlocksPerMatrix = kLocalInter / kTmaAtomK;
  static constexpr std::size_t kGateUpTmaAtomBytes = 16 * kTmaAtomK;
  static constexpr int kGateUpTmaAtoms = 1024 / kTmaAtomK;
  static constexpr int kGateUpTmaK = kTmaAtomK * kGateUpTmaKAtomsPerChunk;
  static constexpr int kHotExpertBudget = 2 * kTokens;
  static constexpr int kHotExpertMaskWords = kRoutedExperts / kWarpSize;
  static constexpr std::size_t kLocalTopkBytes = kTopK * (sizeof(std::int32_t) + sizeof(float));
  static constexpr int kProducerWarps = kProducerThreads / kWarpSize;
  static constexpr int kRoutedCtas = kTokens * kRoutedCtasPerToken;
  static constexpr int kRoutedGateUpRowsPerCta = kLocalInter / 4;
  static constexpr int kRoutedGateUpSentinelsPerToken = kTopK * 4;
  static constexpr int kRouterMaxPairsPerCta = (kRouterMaxExpertsPerCta + 1) / 2;
  static constexpr int kRouterPackedValues = kHidden / 2;
  static constexpr int kRouterPairExpertBytes = kHidden * sizeof(__nv_bfloat16);
  static constexpr int kRouterPairsPerCta = kRouterExpertsPerCta / 2;
  static constexpr int kSharedActivationBytes = kTokens * kHidden * sizeof(__nv_bfloat16);
  static constexpr int kSharedActivationPageElements = kTokens * 1024;
  static constexpr int kSharedCtaEnd = kGridCtas;
  static constexpr int kSharedGateUpPipelineTaskBase = kSharedTokensPerCta;
  static constexpr int kSharedGateUpRowGroupsPerCta = kLocalInter / (16 * 16);
  static constexpr int kSharedGateUpSentinelsPerToken = kLocalInter / 16;
  static constexpr int kTopkItems = kTokens * kTopK;
  static constexpr std::size_t kArMaxPayloadBytes = kArMaxTileElements * sizeof(__nv_bfloat16);
  static constexpr int kArMaxPayloadVectors = kArMaxTileElements / kArValuesPerVector;
  static constexpr int kDownActivationCopyVectors = kDownActivationBytes / kRmsInputCopyBytes;
  static constexpr int kDownPipelineTasks =
      kDownVirtualTilesPerCta * kDownExpertPairs * kDownKTasks;
  static constexpr std::size_t kGateUpTmaAtomPairBytes = 2 * kGateUpTmaAtomBytes;
  static constexpr int kGateUpTmaKChunks = 1024 / kGateUpTmaK;
  static constexpr int kGateUpTmaWeightCopies = kGateUpTmaAtoms;
  static constexpr std::size_t kOriGateUpScaleMatrixStride =
      kGateUpScaleBlocksPerMatrix * 8 * sizeof(float);
  static constexpr int kRmsGammaBytesPerProducer = kRmsGammaTransactionBytes / kProducerWarps;
  static constexpr int kRoutedGateUpRowGroupsPerCta = kRoutedGateUpRowsPerCta / 16;
  static constexpr int kRoutedPrefixPipelineTasks = kRouterPipelineTaskBase + kRouterPairsPerCta;
  static constexpr int kRouterPairChunksPerThread = kRouterPackedValues / kConsumerThreads;
  static constexpr int kRouterPairMaxTransactionBytes = 2 * kRouterPairExpertBytes;
  static constexpr int kRouterPairTransactionBytes = 2 * kRouterPairExpertBytes;
  static constexpr int kRouterPartialElements = kRouterMaxExpertsPerCta * kConsumerWarps;
  static constexpr int kRouterReductionExpertsPerWarp = kWarpSize / kConsumerWarps;
  static constexpr int kSharedActivationCopyVectors = kSharedActivationBytes / kRmsInputCopyBytes;
  static constexpr int kSharedActivationPageBytes =
      kSharedActivationPageElements * sizeof(__nv_bfloat16);
  static constexpr int kSharedCtaBegin = kRoutedCtas;
  static constexpr int kUpGateSentinelsPerToken =
      kSharedGateUpSentinelsPerToken + kRoutedGateUpSentinelsPerToken;
  static constexpr std::size_t kGateUpTmaChunkBytes =
      kGateUpTmaKAtomsPerChunk * kGateUpTmaAtomPairBytes;
  static constexpr int kRoutedGateUpPipelineTaskBase = kRoutedPrefixPipelineTasks;
  static constexpr int kRoutedGateUpPipelineTasks = kRoutedGateUpRowGroupsPerCta * 6;
  static constexpr int kSharedActivationPageCopyVectors =
      kSharedActivationPageBytes / kRmsInputCopyBytes;
  static constexpr std::size_t kSharedActivationStorageBytes =
      kSharedActivationPageBuffers * kSharedActivationPageBytes;
  static constexpr int kSharedCtas = kSharedCtaEnd - kSharedCtaBegin;
  static constexpr int kTopkStatsCta = kSharedCtaBegin + kTokens;
  static constexpr int kRoutedDownPipelineTaskBase =
      kRoutedGateUpPipelineTaskBase + kRoutedGateUpPipelineTasks;
  static constexpr int kEpochCount = kRoutedCtas;
  static constexpr std::size_t kPeerBytes = kTokens * kArTokenStride;
  static constexpr int kWorkspaceFill = 0xff;

};

template <int Tokens, int Tp>
struct MoeConfig;
template <int Tp>
struct MoeConfig<4, Tp> : MoeCommonConfig<4, Tp> {
  using Base = MoeCommonConfig<4, Tp>;
  static constexpr std::size_t kArLocalOffset = 0x28980;
  static constexpr std::size_t kArRemoteBytes = Base::kArRemoteSources * Base::kArPayloadBytes;
  static constexpr std::size_t kArRemoteOffset = 0x2a480;
  static constexpr std::size_t kArResidualOffset = 0x28b00;
  static constexpr std::size_t kAuxOffset = Base::kPipelineStages * Base::kStageBytes;
  static constexpr std::size_t kDownExpertStageStride = 0x3400;
  static constexpr int kDownNativeScaleTransactionBytes =
      4 * Base::kDownScaleBlocks * sizeof(float);
  static constexpr std::size_t kDownReductionBytes =
      Base::kConsumerWarps * Base::kDownVirtualTilesPerCta * 48 * sizeof(float);
  static constexpr std::size_t kDownStageScaleOffset = 0x3000;
  static constexpr int kDownTmaIssueWarps = 2;
  static constexpr int kDownWeightTransactionBytes = 48 * Base::kDownKPerTask;
  static constexpr int kGateUpL2EvictFirstSlot = 6;
  static constexpr std::size_t kGateUpScaleMatrixStride = 0x40;
  static constexpr int kGateUpScaleTmaBytes =
      (Base::kGateUpScaleBlocksPerMatrix + 1) * 8 * sizeof(float);
  static constexpr std::size_t kPeerWorkspaceCacheBytes = kFusedMoeMaxWorldSize * sizeof(void*);
  static constexpr int kRoutedGateUpMaxGroups = 8;
  static constexpr int kRoutedGateUpResultElements = 2 * 16;
  static constexpr std::size_t kRouterLogitsOffset = 0x0c000;
  static constexpr int kRouterReductionWarps =
      Base::kRouterExpertsPerCta / Base::kRouterReductionExpertsPerWarp;
  static constexpr int kSharedConsumerWarpsPerGroup = 2;
  static constexpr int kSharedGateUpPipelineTasks = 6;
  static constexpr int kSharedRowGroups = 16;
  static constexpr int kSharedRowGroupsPerCta = 4;
  static constexpr std::size_t kUpGateOffset = 0x0d000;
  static constexpr std::size_t kDownReductionOffset = kAuxOffset + Base::kDownActivationBytes;
  static constexpr std::size_t kDownStageExpertStride =
      (kDownWeightTransactionBytes + kDownNativeScaleTransactionBytes + 0x3ff) & ~0x3ff;
  static constexpr std::size_t kGateUpNativeTransactionBytes =
      Base::kGateUpTmaWeightCopies * Base::kGateUpTmaAtomPairBytes + kGateUpScaleTmaBytes;
  static constexpr std::size_t kNormalizedReadyOffset =
      kUpGateOffset +
      Base::kTokens * Base::kDownActivationSlots * Base::kLocalInter * sizeof(__nv_bfloat16);
  static constexpr int kOriGateUpTransactionBytes = 0x8000 + kGateUpScaleTmaBytes;
  static constexpr int kRoutedGateUpMaxPipelineTasks =
      kRoutedGateUpMaxGroups * Base::kGateUpPagesPerGroup;
  static constexpr std::size_t kSharedActivationOffset = kAuxOffset + 0x800;
  static constexpr int kSharedConsumerThreadsPerGroup =
      kSharedConsumerWarpsPerGroup * Base::kWarpSize;
  static constexpr int kSharedGateUpResultElements =
      kSharedRowGroupsPerCta * 2 * Base::kTokens * 16;
  static constexpr std::size_t kSharedPartialBytes =
      kSharedRowGroupsPerCta * 2 * 2 * Base::kTokens * 16 * sizeof(float);
  static constexpr std::size_t kSharedSentinelBytes =
      kSharedRowGroupsPerCta * Base::kTokens * sizeof(std::uint16_t);
  static constexpr std::size_t kTopkRawOffset = kAuxOffset;
  static constexpr std::size_t kSharedPartialOffset =
      kSharedActivationOffset + Base::kSharedActivationStorageBytes;
  static constexpr std::size_t kSharedResultBytes = kSharedGateUpResultElements * sizeof(float);
  static constexpr std::size_t kTopkBiasOffset =
      kTopkRawOffset + Base::kRoutedExperts * sizeof(float);
  static constexpr std::size_t kTopkClaimOffset =
      kNormalizedReadyOffset + Base::kTokens * sizeof(std::uint32_t);
  static constexpr std::size_t kRoutedActivationOffset = kSharedPartialOffset;
  static constexpr std::size_t kSharedResultOffset = kSharedPartialOffset + kSharedPartialBytes;
  static constexpr std::size_t kTopkEndOffset =
      kTopkBiasOffset + Base::kRoutedExperts * sizeof(float);
  static constexpr std::size_t kTopkReadyOffset =
      kTopkClaimOffset + Base::kTokens * sizeof(std::uint32_t);
  static constexpr std::size_t kHotExpertMaskOffset =
      kTopkReadyOffset + Base::kTokens * sizeof(std::uint32_t);
  static constexpr std::size_t kSharedSentinelOffset = kSharedResultOffset + kSharedResultBytes;
  static constexpr std::size_t kTopkFrequencyReadyOffset =
      kHotExpertMaskOffset + Base::kHotExpertMaskWords * sizeof(std::uint32_t);
  static constexpr std::size_t kWorkspaceRequiredBytes =
      Tp == 4 ? (kTopkFrequencyReadyOffset + sizeof(std::uint32_t)) : (0x11b10);
};

template <int Tp>
struct MoeConfig<8, Tp> : MoeCommonConfig<8, Tp> {
  using Base = MoeCommonConfig<8, Tp>;
  static constexpr std::size_t kArLocalOffset = 0x30000;
  static constexpr std::size_t kArRemoteBytes = Base::kArRemoteSources * Base::kArMaxPayloadBytes;
  static constexpr std::size_t kAuxOffset = 0x24000;
  static constexpr auto kDownNativeScaleTransactionBytes = [] {
    if constexpr (Tp == 4)
      return static_cast<std::size_t>(4 * Base::kDownScaleBlocks * sizeof(float));
    else
      return static_cast<int>(4 * Base::kDownScaleBlocks * sizeof(float));
  }();
  static constexpr std::size_t kDownReductionBytes =
      Base::kConsumerWarps * Base::kDownMaxM16PerCta * Base::kDownM16Rows * sizeof(float);
  static constexpr int kDownTmaIssueWarps = 3;
  static constexpr auto kDownWeightTransactionBytes = [] {
    if constexpr (Tp == 4)
      return static_cast<std::size_t>(Base::kDownStagedTmaKAtoms * Base::kDownTmaAtomTileBytes);
    else
      return static_cast<int>(48 * Base::kDownKPerTask);
  }();
  static constexpr int kGateUpL2EvictFirstSlot = Tp == 4 ? (6) : (4);
  static constexpr std::size_t kGateUpScaleMatrixStride =
      Tp == 4 ? (Base::kGateUpScaleBlocksPerMatrix * 8 * sizeof(float)) : (0x40);
  static constexpr std::size_t kGateUpScaleTmaBytes =
      Tp == 4 ? ((Base::kGateUpScaleBlocksPerMatrix + 1) * 8 * sizeof(float))
              : (3 * 8 * sizeof(float));
  static constexpr std::size_t kPeerWorkspaceCacheBytes =
      Tp == 4 ? (kFusedMoeMaxWorldSize * sizeof(void*)) : (kFusedMoeTp8WorldSize * sizeof(void*));
  static constexpr int kRoutedGateUpMaxGroups = Tp == 4 ? (8 * Base::kGateUpLayoutFactor) : (8);
  static constexpr int kRoutedGateUpResultElements = Tp == 4 ? (4 * 16) : (2 * 16);
  static constexpr std::size_t kRouterLogitsOffset = 0x18000;
  static constexpr int kRouterReductionWarps =
      (Base::kRouterMaxExpertsPerCta + Base::kRouterReductionExpertsPerWarp - 1) /
      Base::kRouterReductionExpertsPerWarp;
  static constexpr int kSharedConsumerWarpsPerGroup = Tp == 4 ? (1) : (2);
  static constexpr int kSharedRowGroups = Tp == 4 ? (Base::kGateUpRowGroupsPerExpert) : (16);
  static constexpr int kSharedRowGroupsPerCta = Tp == 4 ? (8) : (4);
  static constexpr std::size_t kUpGateOffset = 0x1a000;
  static constexpr std::size_t kArResidualOffset =
      Tp == 4 ? (kArLocalOffset + Base::kArMaxPayloadBytes) : (0x30300);
  static constexpr std::size_t kDownExpertStageStride =
      Tp == 4 ? ((kDownWeightTransactionBytes + kDownNativeScaleTransactionBytes + 0x3ff) & ~0x3ff)
              : (0x3400);
  static constexpr std::size_t kDownReductionOffset = kAuxOffset + Base::kDownActivationBytes;
  static constexpr std::size_t kDownStageExpertStride =
      (kDownWeightTransactionBytes + kDownNativeScaleTransactionBytes + 0x3ff) & ~0x3ff;
  static constexpr std::size_t kDownStageScaleOffset =
      Tp == 4 ? (kDownWeightTransactionBytes) : (0x3000);
  static constexpr std::size_t kGateUpNativeTransactionBytes =
      Base::kGateUpTmaWeightCopies * Base::kGateUpTmaAtomPairBytes + kGateUpScaleTmaBytes;
  static constexpr std::size_t kNormalizedReadyOffset =
      Tp == 4 ? (kUpGateOffset + Base::kTokens * Base::kDownActivationSlots * Base::kLocalInter *
                                     sizeof(__nv_bfloat16))
              : (0x23000);
  static constexpr int kOriGateUpTransactionBytes = 0x8000 + kGateUpScaleTmaBytes;
  static constexpr int kRoutedGateUpMaxPipelineTasks =
      kRoutedGateUpMaxGroups * Base::kGateUpPagesPerGroup;
  static constexpr std::size_t kSharedActivationOffset = kAuxOffset;
  static constexpr int kSharedConsumerThreadsPerGroup =
      kSharedConsumerWarpsPerGroup * Base::kWarpSize;
  static constexpr int kSharedGateUpPipelineTasks =
      kSharedRowGroupsPerCta * Base::kGateUpPagesPerGroup;
  static constexpr int kSharedGateUpResultElements =
      kSharedRowGroupsPerCta * 2 * Base::kTokens * 16;
  static constexpr std::size_t kSharedPartialBytes =
      kSharedRowGroupsPerCta * 2 * 2 * Base::kTokens * 16 * sizeof(float);
  static constexpr std::size_t kSharedSentinelBytes =
      kSharedRowGroupsPerCta * Base::kTokens * sizeof(std::uint16_t);
  static constexpr std::size_t kTopkRawOffset = kAuxOffset;
  static constexpr std::size_t kArRemoteOffset =
      Tp == 4 ? (kArResidualOffset + Base::kArMaxPayloadBytes) : (0x30600);
  static constexpr std::size_t kSharedPartialOffset =
      kSharedActivationOffset + Base::kSharedActivationStorageBytes;
  static constexpr std::size_t kSharedResultBytes = kSharedGateUpResultElements * sizeof(float);
  static constexpr std::size_t kTopkBiasOffset =
      kTopkRawOffset + Base::kRoutedExperts * sizeof(float);
  static constexpr std::size_t kTopkClaimOffset =
      kNormalizedReadyOffset + Base::kTokens * sizeof(std::uint32_t);
  static constexpr std::size_t kWorkspaceRequiredBytes =
      Tp == 4 ? (kNormalizedReadyOffset + Base::kTokens * sizeof(std::uint32_t)) : (0x23020);
  static constexpr std::size_t kSharedResultOffset = kSharedPartialOffset + kSharedPartialBytes;
  static constexpr std::size_t kTopkEndOffset =
      kTopkBiasOffset + Base::kRoutedExperts * sizeof(float);
  static constexpr std::size_t kTopkReadyOffset =
      kTopkClaimOffset + Base::kTokens * sizeof(std::uint32_t);
  static constexpr std::size_t kHotExpertMaskOffset =
      kTopkReadyOffset + Base::kTokens * sizeof(std::uint32_t);
  static constexpr std::size_t kSharedSentinelOffset =
      Tp == 4 ? (kSharedActivationOffset + Base::kSharedActivationStorageBytes)
              : (kSharedResultOffset + kSharedResultBytes);
  static constexpr std::size_t kRoutedActivationOffset =
      Tp == 4 ? (kSharedSentinelOffset + kSharedSentinelBytes) : (kSharedPartialOffset);
  static constexpr std::size_t kTopkFrequencyReadyOffset =
      kHotExpertMaskOffset + Base::kHotExpertMaskWords * sizeof(std::uint32_t);
};

using ActiveMoeConfig = MoeConfig<kFusedMoeSeqLen, kFusedMoeTpSize>;

// Private aliases keep the selected instruction bodies readable.
// Entry instantiation checks that Config equals this selected specialization.
inline constexpr auto kArElementsPerVirtualTile = ActiveMoeConfig::kArElementsPerVirtualTile;
inline constexpr auto kArLocalOffset = ActiveMoeConfig::kArLocalOffset;
inline constexpr auto kArMaxPayloadBytes = ActiveMoeConfig::kArMaxPayloadBytes;
inline constexpr auto kArMaxPayloadVectors = ActiveMoeConfig::kArMaxPayloadVectors;
inline constexpr auto kArMaxTileElements = ActiveMoeConfig::kArMaxTileElements;
inline constexpr auto kArMaxVirtualTilesPerCta = ActiveMoeConfig::kArMaxVirtualTilesPerCta;
inline constexpr auto kArPayloadBytes = ActiveMoeConfig::kArPayloadBytes;
inline constexpr auto kArPayloadRowsPerColumn = ActiveMoeConfig::kArPayloadRowsPerColumn;
inline constexpr auto kArPayloadVectors = ActiveMoeConfig::kArPayloadVectors;
inline constexpr auto kArPayloadVectorsPerProducer = ActiveMoeConfig::kArPayloadVectorsPerProducer;
inline constexpr auto kArPhaseStride = ActiveMoeConfig::kArPhaseStride;
inline constexpr auto kArProducerWarps = ActiveMoeConfig::kArProducerWarps;
inline constexpr auto kArReductionLanesPerWarp = ActiveMoeConfig::kArReductionLanesPerWarp;
inline constexpr auto kArReductionValuesPerThread = ActiveMoeConfig::kArReductionValuesPerThread;
inline constexpr auto kArReductionWarps = ActiveMoeConfig::kArReductionWarps;
inline constexpr auto kArRemoteBytes = ActiveMoeConfig::kArRemoteBytes;
inline constexpr auto kArRemoteOffset = ActiveMoeConfig::kArRemoteOffset;
inline constexpr auto kArRemoteSources = ActiveMoeConfig::kArRemoteSources;
inline constexpr auto kArResidualOffset = ActiveMoeConfig::kArResidualOffset;
inline constexpr auto kArSourceStride = ActiveMoeConfig::kArSourceStride;
inline constexpr auto kArTileElements = ActiveMoeConfig::kArTileElements;
inline constexpr auto kArTileStride = ActiveMoeConfig::kArTileStride;
inline constexpr auto kArTilesPerToken = ActiveMoeConfig::kArTilesPerToken;
inline constexpr auto kArTokenStride = ActiveMoeConfig::kArTokenStride;
inline constexpr auto kArValuesPerVector = ActiveMoeConfig::kArValuesPerVector;
inline constexpr auto kArVectorBytes = ActiveMoeConfig::kArVectorBytes;
inline constexpr auto kArVectorsPerColumn = ActiveMoeConfig::kArVectorsPerColumn;
inline constexpr auto kArVirtualTileStride = ActiveMoeConfig::kArVirtualTileStride;
inline constexpr auto kArVirtualTilesPerPass = ActiveMoeConfig::kArVirtualTilesPerPass;
inline constexpr auto kAuxOffset = ActiveMoeConfig::kAuxOffset;
inline constexpr auto kBf16NotReady = ActiveMoeConfig::kBf16NotReady;
inline constexpr auto kConsumerThreads = ActiveMoeConfig::kConsumerThreads;
inline constexpr auto kConsumerWarps = ActiveMoeConfig::kConsumerWarps;
inline constexpr auto kDownActivationBytes = ActiveMoeConfig::kDownActivationBytes;
inline constexpr auto kDownActivationCopyVectors = ActiveMoeConfig::kDownActivationCopyVectors;
inline constexpr auto kDownActivationSlots = ActiveMoeConfig::kDownActivationSlots;
inline constexpr auto kDownExpertPairs = ActiveMoeConfig::kDownExpertPairs;
inline constexpr auto kDownExpertStageStride = ActiveMoeConfig::kDownExpertStageStride;
inline constexpr auto kDownK32Partitions = ActiveMoeConfig::kDownK32Partitions;
inline constexpr auto kDownK32PartitionsPerTask = ActiveMoeConfig::kDownK32PartitionsPerTask;
inline constexpr auto kDownKPerTask = ActiveMoeConfig::kDownKPerTask;
inline constexpr auto kDownKTasks = ActiveMoeConfig::kDownKTasks;
inline constexpr auto kDownM16PerVirtualTile = ActiveMoeConfig::kDownM16PerVirtualTile;
inline constexpr auto kDownM16Rows = ActiveMoeConfig::kDownM16Rows;
inline constexpr auto kDownM16Tiles = ActiveMoeConfig::kDownM16Tiles;
inline constexpr auto kDownMaxM16PerCta = ActiveMoeConfig::kDownMaxM16PerCta;
inline constexpr auto kDownNativeScaleTransactionBytes =
    ActiveMoeConfig::kDownNativeScaleTransactionBytes;
inline constexpr auto kDownNativeTransactionBytes = ActiveMoeConfig::kDownNativeTransactionBytes;
inline constexpr auto kDownPhysicalTmaKAtoms = ActiveMoeConfig::kDownPhysicalTmaKAtoms;
inline constexpr auto kDownPipelineTasks = ActiveMoeConfig::kDownPipelineTasks;
inline constexpr auto kDownReductionBytes = ActiveMoeConfig::kDownReductionBytes;
inline constexpr auto kDownReductionOffset = ActiveMoeConfig::kDownReductionOffset;
inline constexpr auto kDownResetProducerWarp = ActiveMoeConfig::kDownResetProducerWarp;
inline constexpr auto kDownScaleBlocks = ActiveMoeConfig::kDownScaleBlocks;
inline constexpr auto kDownStageExpertStride = ActiveMoeConfig::kDownStageExpertStride;
inline constexpr auto kDownStageScaleOffset = ActiveMoeConfig::kDownStageScaleOffset;
inline constexpr auto kDownStagedTmaKAtoms = ActiveMoeConfig::kDownStagedTmaKAtoms;
inline constexpr auto kDownTmaAtomTileBytes = ActiveMoeConfig::kDownTmaAtomTileBytes;
inline constexpr auto kDownTmaIssueWarps = ActiveMoeConfig::kDownTmaIssueWarps;
inline constexpr auto kDownTmaKAtoms = ActiveMoeConfig::kDownTmaKAtoms;
inline constexpr auto kDownVirtualTilesPerCta = ActiveMoeConfig::kDownVirtualTilesPerCta;
inline constexpr auto kDownWeightTmaBytes = ActiveMoeConfig::kDownWeightTmaBytes;
inline constexpr auto kDownWeightTransactionBytes = ActiveMoeConfig::kDownWeightTransactionBytes;
inline constexpr auto kDynamicSmemBytes = ActiveMoeConfig::kDynamicSmemBytes;
inline constexpr auto kEps = ActiveMoeConfig::kEps;
inline constexpr auto kFullWarpMask = ActiveMoeConfig::kFullWarpMask;
inline constexpr auto kGateUpCachePolicyBytes = ActiveMoeConfig::kGateUpCachePolicyBytes;
inline constexpr auto kGateUpL2EvictFirstSlot = ActiveMoeConfig::kGateUpL2EvictFirstSlot;
inline constexpr auto kGateUpL2EvictLastSlots = ActiveMoeConfig::kGateUpL2EvictLastSlots;
inline constexpr auto kGateUpL2RetainedExpertMask = ActiveMoeConfig::kGateUpL2RetainedExpertMask;
inline constexpr auto kGateUpLayoutFactor = ActiveMoeConfig::kGateUpLayoutFactor;
inline constexpr auto kGateUpNativeTransactionBytes =
    ActiveMoeConfig::kGateUpNativeTransactionBytes;
inline constexpr auto kGateUpPagesPerGroup = ActiveMoeConfig::kGateUpPagesPerGroup;
inline constexpr auto kGateUpRowGroupsPerExpert = ActiveMoeConfig::kGateUpRowGroupsPerExpert;
inline constexpr auto kGateUpScaleBlocksPerMatrix = ActiveMoeConfig::kGateUpScaleBlocksPerMatrix;
inline constexpr auto kGateUpScaleMatrixStride = ActiveMoeConfig::kGateUpScaleMatrixStride;
inline constexpr auto kGateUpScaleOffset = ActiveMoeConfig::kGateUpScaleOffset;
inline constexpr auto kGateUpScaleTmaBytes = ActiveMoeConfig::kGateUpScaleTmaBytes;
inline constexpr auto kGateUpScratchElements = ActiveMoeConfig::kGateUpScratchElements;
inline constexpr auto kGateUpTmaAtomBytes = ActiveMoeConfig::kGateUpTmaAtomBytes;
inline constexpr auto kGateUpTmaAtomPairBytes = ActiveMoeConfig::kGateUpTmaAtomPairBytes;
inline constexpr auto kGateUpTmaAtoms = ActiveMoeConfig::kGateUpTmaAtoms;
inline constexpr auto kGateUpTmaChunkBytes = ActiveMoeConfig::kGateUpTmaChunkBytes;
inline constexpr auto kGateUpTmaK = ActiveMoeConfig::kGateUpTmaK;
inline constexpr auto kGateUpTmaKAtomsPerChunk = ActiveMoeConfig::kGateUpTmaKAtomsPerChunk;
inline constexpr auto kGateUpTmaKChunks = ActiveMoeConfig::kGateUpTmaKChunks;
inline constexpr auto kGateUpTmaWeightCopies = ActiveMoeConfig::kGateUpTmaWeightCopies;
inline constexpr auto kGlobalNotReady = ActiveMoeConfig::kGlobalNotReady;
inline constexpr auto kGlobalReady = ActiveMoeConfig::kGlobalReady;
inline constexpr auto kGridCtas = ActiveMoeConfig::kGridCtas;
inline constexpr auto kHidden = ActiveMoeConfig::kHidden;
inline constexpr auto kHotExpertBudget = ActiveMoeConfig::kHotExpertBudget;
inline constexpr auto kHotExpertMaskOffset = ActiveMoeConfig::kHotExpertMaskOffset;
inline constexpr auto kHotExpertMaskWords = ActiveMoeConfig::kHotExpertMaskWords;
inline constexpr auto kLocalInter = ActiveMoeConfig::kLocalInter;
inline constexpr auto kLocalTopkBytes = ActiveMoeConfig::kLocalTopkBytes;
inline constexpr auto kNegativeFloatMax = ActiveMoeConfig::kNegativeFloatMax;
inline constexpr auto kNormOffset = ActiveMoeConfig::kNormOffset;
inline constexpr auto kNormalizedReadyOffset = ActiveMoeConfig::kNormalizedReadyOffset;
inline constexpr auto kOriGateUpScaleMatrixStride = ActiveMoeConfig::kOriGateUpScaleMatrixStride;
inline constexpr auto kOriGateUpTransactionBytes = ActiveMoeConfig::kOriGateUpTransactionBytes;
inline constexpr auto kPeerWorkspaceCacheBytes = ActiveMoeConfig::kPeerWorkspaceCacheBytes;
inline constexpr auto kPipelineStages = ActiveMoeConfig::kPipelineStages;
inline constexpr auto kProducerThreads = ActiveMoeConfig::kProducerThreads;
inline constexpr auto kProducerWarps = ActiveMoeConfig::kProducerWarps;
inline constexpr auto kRequiredSms = ActiveMoeConfig::kRequiredSms;
inline constexpr auto kRmsGammaBytesPerProducer = ActiveMoeConfig::kRmsGammaBytesPerProducer;
inline constexpr auto kRmsGammaTransactionBytes = ActiveMoeConfig::kRmsGammaTransactionBytes;
inline constexpr auto kRmsInputCopiesPerConsumer = ActiveMoeConfig::kRmsInputCopiesPerConsumer;
inline constexpr auto kRmsInputCopyBytes = ActiveMoeConfig::kRmsInputCopyBytes;
inline constexpr auto kRouteScale = ActiveMoeConfig::kRouteScale;
inline constexpr auto kRoutedActivationOffset = ActiveMoeConfig::kRoutedActivationOffset;
inline constexpr auto kRoutedCtas = ActiveMoeConfig::kRoutedCtas;
inline constexpr auto kRoutedCtasPerToken = ActiveMoeConfig::kRoutedCtasPerToken;
inline constexpr auto kRoutedDownPipelineTaskBase = ActiveMoeConfig::kRoutedDownPipelineTaskBase;
inline constexpr auto kRoutedExperts = ActiveMoeConfig::kRoutedExperts;
inline constexpr auto kRoutedGateUpMaxGroups = ActiveMoeConfig::kRoutedGateUpMaxGroups;
inline constexpr auto kRoutedGateUpMaxPipelineTasks =
    ActiveMoeConfig::kRoutedGateUpMaxPipelineTasks;
inline constexpr auto kRoutedGateUpPipelineTaskBase =
    ActiveMoeConfig::kRoutedGateUpPipelineTaskBase;
inline constexpr auto kRoutedGateUpPipelineTasks = ActiveMoeConfig::kRoutedGateUpPipelineTasks;
inline constexpr auto kRoutedGateUpResultElements = ActiveMoeConfig::kRoutedGateUpResultElements;
inline constexpr auto kRoutedGateUpRowGroupsPerCta = ActiveMoeConfig::kRoutedGateUpRowGroupsPerCta;
inline constexpr auto kRoutedGateUpRowsPerCta = ActiveMoeConfig::kRoutedGateUpRowsPerCta;
inline constexpr auto kRoutedGateUpSentinelsPerToken =
    ActiveMoeConfig::kRoutedGateUpSentinelsPerToken;
inline constexpr auto kRoutedPrefixPipelineTasks = ActiveMoeConfig::kRoutedPrefixPipelineTasks;
inline constexpr auto kRouterExpertsPerCta = ActiveMoeConfig::kRouterExpertsPerCta;
inline constexpr auto kRouterLogitsOffset = ActiveMoeConfig::kRouterLogitsOffset;
inline constexpr auto kRouterMaxExpertsPerCta = ActiveMoeConfig::kRouterMaxExpertsPerCta;
inline constexpr auto kRouterMaxPairsPerCta = ActiveMoeConfig::kRouterMaxPairsPerCta;
inline constexpr auto kRouterPackedValues = ActiveMoeConfig::kRouterPackedValues;
inline constexpr auto kRouterPairChunksPerThread = ActiveMoeConfig::kRouterPairChunksPerThread;
inline constexpr auto kRouterPairExpertBytes = ActiveMoeConfig::kRouterPairExpertBytes;
inline constexpr auto kRouterPairMaxTransactionBytes =
    ActiveMoeConfig::kRouterPairMaxTransactionBytes;
inline constexpr auto kRouterPairTransactionBytes = ActiveMoeConfig::kRouterPairTransactionBytes;
inline constexpr auto kRouterPairsPerCta = ActiveMoeConfig::kRouterPairsPerCta;
inline constexpr auto kRouterPartialElements = ActiveMoeConfig::kRouterPartialElements;
inline constexpr auto kRouterPipelineTaskBase = ActiveMoeConfig::kRouterPipelineTaskBase;
inline constexpr auto kRouterReductionExpertsPerWarp =
    ActiveMoeConfig::kRouterReductionExpertsPerWarp;
inline constexpr auto kRouterReductionWarps = ActiveMoeConfig::kRouterReductionWarps;
inline constexpr auto kSharedActivationBytes = ActiveMoeConfig::kSharedActivationBytes;
inline constexpr auto kSharedActivationCopyVectors = ActiveMoeConfig::kSharedActivationCopyVectors;
inline constexpr auto kSharedActivationOffset = ActiveMoeConfig::kSharedActivationOffset;
inline constexpr auto kSharedActivationPageBuffers = ActiveMoeConfig::kSharedActivationPageBuffers;
inline constexpr auto kSharedActivationPageBytes = ActiveMoeConfig::kSharedActivationPageBytes;
inline constexpr auto kSharedActivationPageCopyVectors =
    ActiveMoeConfig::kSharedActivationPageCopyVectors;
inline constexpr auto kSharedActivationPageElements =
    ActiveMoeConfig::kSharedActivationPageElements;
inline constexpr auto kSharedActivationStorageBytes =
    ActiveMoeConfig::kSharedActivationStorageBytes;
inline constexpr auto kSharedConsumerThreadsPerGroup =
    ActiveMoeConfig::kSharedConsumerThreadsPerGroup;
inline constexpr auto kSharedConsumerWarpsPerGroup = ActiveMoeConfig::kSharedConsumerWarpsPerGroup;
inline constexpr auto kSharedCtaBegin = ActiveMoeConfig::kSharedCtaBegin;
inline constexpr auto kSharedCtaEnd = ActiveMoeConfig::kSharedCtaEnd;
inline constexpr auto kSharedCtas = ActiveMoeConfig::kSharedCtas;
inline constexpr auto kSharedFollowerGateUpPipelineTaskBase =
    ActiveMoeConfig::kSharedFollowerGateUpPipelineTaskBase;
inline constexpr auto kSharedGateUpPipelineTaskBase =
    ActiveMoeConfig::kSharedGateUpPipelineTaskBase;
inline constexpr auto kSharedGateUpPipelineTasks = ActiveMoeConfig::kSharedGateUpPipelineTasks;
inline constexpr auto kSharedGateUpResultElements = ActiveMoeConfig::kSharedGateUpResultElements;
inline constexpr auto kSharedGateUpRowGroupsPerCta = ActiveMoeConfig::kSharedGateUpRowGroupsPerCta;
inline constexpr auto kSharedGateUpSentinelsPerToken =
    ActiveMoeConfig::kSharedGateUpSentinelsPerToken;
inline constexpr auto kSharedOwnerGateUpPipelineTaskBase =
    ActiveMoeConfig::kSharedOwnerGateUpPipelineTaskBase;
inline constexpr auto kSharedPartialBytes = ActiveMoeConfig::kSharedPartialBytes;
inline constexpr auto kSharedPartialOffset = ActiveMoeConfig::kSharedPartialOffset;
inline constexpr auto kSharedResultBytes = ActiveMoeConfig::kSharedResultBytes;
inline constexpr auto kSharedResultOffset = ActiveMoeConfig::kSharedResultOffset;
inline constexpr auto kSharedRowGroups = ActiveMoeConfig::kSharedRowGroups;
inline constexpr auto kSharedRowGroupsPerCta = ActiveMoeConfig::kSharedRowGroupsPerCta;
inline constexpr auto kSharedSentinelBytes = ActiveMoeConfig::kSharedSentinelBytes;
inline constexpr auto kSharedSentinelOffset = ActiveMoeConfig::kSharedSentinelOffset;
inline constexpr auto kSharedTokensPerCta = ActiveMoeConfig::kSharedTokensPerCta;
inline constexpr auto kStageBytes = ActiveMoeConfig::kStageBytes;
inline constexpr auto kStaticSmemBytes = ActiveMoeConfig::kStaticSmemBytes;
inline constexpr auto kStorageControlBytes = ActiveMoeConfig::kStorageControlBytes;
inline constexpr auto kThreads = ActiveMoeConfig::kThreads;
inline constexpr auto kTmaAtomK = ActiveMoeConfig::kTmaAtomK;
inline constexpr auto kTokens = ActiveMoeConfig::kTokens;
inline constexpr auto kTopK = ActiveMoeConfig::kTopK;
inline constexpr auto kTopkBiasOffset = ActiveMoeConfig::kTopkBiasOffset;
inline constexpr auto kTopkClaimOffset = ActiveMoeConfig::kTopkClaimOffset;
inline constexpr auto kTopkEndOffset = ActiveMoeConfig::kTopkEndOffset;
inline constexpr auto kTopkFrequency = ActiveMoeConfig::kTopkFrequency;
inline constexpr auto kTopkFrequencyReadyOffset = ActiveMoeConfig::kTopkFrequencyReadyOffset;
inline constexpr auto kTopkHandoffBytes = ActiveMoeConfig::kTopkHandoffBytes;
inline constexpr auto kTopkItems = ActiveMoeConfig::kTopkItems;
inline constexpr auto kTopkRawOffset = ActiveMoeConfig::kTopkRawOffset;
inline constexpr auto kTopkReadyOffset = ActiveMoeConfig::kTopkReadyOffset;
inline constexpr auto kTopkStatsCta = ActiveMoeConfig::kTopkStatsCta;
inline constexpr auto kTpSize = ActiveMoeConfig::kTpSize;
inline constexpr auto kUpGateOffset = ActiveMoeConfig::kUpGateOffset;
inline constexpr auto kUpGateSentinelsPerToken = ActiveMoeConfig::kUpGateSentinelsPerToken;
inline constexpr auto kWarpSize = ActiveMoeConfig::kWarpSize;
inline constexpr auto kWorkspaceRequiredBytes = ActiveMoeConfig::kWorkspaceRequiredBytes;

#if MEGAMOE_USE_NATIVE_E4M3X2_TO_BF16X2
using Hmma = HmmaM16N8K16F32Bf16;
using StagedActivation = __nv_bfloat16;
#else
using Hmma = HmmaM16N8K16F32F16;
using StagedActivation = __half;
#endif
using PipelineControl = PipelineState<4, 0x9000, 384, 256, 4>;
struct Layout {
  using Args = Glm5MegaMoeArgs;
};
struct RoutedCtaDescriptor {
  int router_expert_base;
  int router_expert_count;
  int gate_flat_base;
  int gate_group_count;
  int down_m16_base;
  int down_m16_count;
};

template <class Config, bool Frequency = Config::kTopkFrequency>
struct alignas(16) MoeSharedStorage {
  __nv_bfloat16 normalized[Config::kHidden];
  float scratch[Config::kThreads];
  float gate_up[Config::kGateUpScratchElements];
  PipelineControl pipeline;
  std::int32_t local_selected_indices[Config::kTopK];
  float local_selected_probs[Config::kTopK];
  alignas(8) std::uint64_t topk_handoff;
  void* peer_workspaces[kFusedMoeMaxWorldSize];
  std::uint8_t padding[Config::kStorageControlBytes - sizeof(PipelineControl) -
                       Config::kLocalTopkBytes - Config::kTopkHandoffBytes -
                       kFusedMoeMaxWorldSize * sizeof(void*)];
};
template <class Config>
struct alignas(16) MoeSharedStorage<Config, true> {
  __nv_bfloat16 normalized[Config::kHidden];
  float scratch[Config::kThreads];
  float gate_up[Config::kGateUpScratchElements];
  PipelineControl pipeline;
  std::int32_t local_selected_indices[Config::kTopK];
  float local_selected_probs[Config::kTopK];
  alignas(8) std::uint64_t topk_handoff;
  std::uint64_t gate_up_cache_policy;
  void* peer_workspaces[kFusedMoeMaxWorldSize];
  std::uint8_t padding[Config::kStorageControlBytes - sizeof(PipelineControl) -
                       Config::kLocalTopkBytes - Config::kTopkHandoffBytes -
                       Config::kGateUpCachePolicyBytes - kFusedMoeMaxWorldSize * sizeof(void*)];
};
using SharedStorage = MoeSharedStorage<ActiveMoeConfig>;

template <class Config>
struct MoeAllReduceConfig {
  static constexpr int kWorldSize = Config::kTpSize;
  static constexpr int kHidden = Config::kHidden;
  static constexpr int kConsumerThreads = Config::kConsumerThreads;
  static constexpr int kWarpSize = Config::kWarpSize;
  static constexpr int kElementsPerVirtualTile = Config::kArElementsPerVirtualTile;
  static constexpr int kValuesPerVector = Config::kArValuesPerVector;
  static constexpr int kPayloadRowsPerColumn = Config::kArPayloadRowsPerColumn;
  static constexpr int kVectorsPerColumn = Config::kArVectorsPerColumn;
  static constexpr int kVirtualTilesPerPass = Config::kArVirtualTilesPerPass;
  static constexpr int kReductionValuesPerThread = Config::kArReductionValuesPerThread;
  static constexpr int kMaxVirtualTilesPerCta = Config::kArMaxVirtualTilesPerCta;
  static constexpr int kMaxTileElements = Config::kArMaxTileElements;
  static constexpr int kMaxPayloadVectors = Config::kArMaxPayloadVectors;
  static constexpr int kRemoteSources = Config::kArRemoteSources;
  static constexpr std::size_t kVirtualTileStride = Config::kArVirtualTileStride;
  static constexpr std::size_t kPhaseStride = Config::kArPhaseStride;
  static constexpr std::size_t kSourceStride = Config::kArSourceStride;
  static constexpr std::size_t kTokenStride = Config::kArTokenStride;
  static constexpr std::size_t kLocalOffset = Config::kArLocalOffset;
  static constexpr std::size_t kResidualOffset = Config::kArResidualOffset;
  static constexpr std::size_t kRemoteOffset = Config::kArRemoteOffset;
};
using Tp8AllReduceConfig = MoeAllReduceConfig<ActiveMoeConfig>;
using TpAllReduceConfig = MoeAllReduceConfig<ActiveMoeConfig>;
}  // namespace glm5::moe::detail

namespace glm5::moe {
inline constexpr auto kDefaultWorkspaceBytes = detail::ActiveMoeConfig::kWorkspaceCapacityBytes;
inline constexpr auto kMinimumWorkspaceBytes = detail::ActiveMoeConfig::kWorkspaceRequiredBytes;
inline constexpr auto kFusedMoeEpochCounterCount = detail::ActiveMoeConfig::kEpochCount;
inline constexpr auto kFusedMoeTp8EpochCounterCount = kFusedMoeEpochCounterCount;
inline constexpr auto kFusedMoeRoutedCtasPerToken = detail::ActiveMoeConfig::kRoutedCtasPerToken;
inline constexpr auto kFusedMoePeerWorkspaceBytes = detail::ActiveMoeConfig::kPeerBytes;
inline constexpr auto kFusedMoeTp8PeerWorkspaceBytes = kFusedMoePeerWorkspaceBytes;
}  // namespace glm5::moe
