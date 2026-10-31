#pragma once

// Shared phase orchestration for the GLM-5.3 4/8-token specializations.
#include <moe/expert.cuh>

namespace glm5::moe::detail {
#if GLM5_MEGAMOE_SEQ_LEN == 4 && GLM5_MEGAMOE_TP_SIZE == 8
static_assert(sizeof(StagedActivation) == sizeof(__nv_bfloat16));
static_assert(sizeof(PipelineControl) <= 0x400);
static_assert(kLocalTopkBytes == 0x40);
static_assert(kTopkHandoffBytes == 8);
static_assert(kPeerWorkspaceCacheBytes == 0x40);
static_assert(sizeof(PipelineControl) + kLocalTopkBytes + kTopkHandoffBytes +
                  kPeerWorkspaceCacheBytes <=
              0x400);
static_assert(sizeof(SharedStorage) == kStaticSmemBytes);
static_assert(kWorkspaceRequiredBytes <= kDefaultWorkspaceBytes);
static_assert(kFusedMoeSeqLen == kTokens);
static_assert(kMinimumWorkspaceBytes == kWorkspaceRequiredBytes);
static_assert(kFusedMoeTpSize == kTpSize,
              "MTP314 TP8 source requires GLM5_MEGAMOE_TP_SIZE=8");
static_assert(kFusedMoeLocalInter == kLocalInter);
static_assert(kRoutedCtas == kFusedMoeEpochCounterCount);
static_assert(kRoutedCtas / kTokens == kFusedMoeRoutedCtasPerToken);
static_assert(kArProducerWarps * kArPayloadVectorsPerProducer == kArPayloadVectors);
static_assert(kNormalizedReadyOffset ==
              kUpGateOffset + kTokens * 9 * kLocalInter * sizeof(__nv_bfloat16));
static_assert(kPipelineStages * kStageBytes == kAuxOffset);
static_assert(kDownActivationBytes % kRmsInputCopyBytes == 0);
static_assert(kSharedActivationBytes % kRmsInputCopyBytes == 0);
static_assert(kAuxOffset + kDownActivationBytes <= kDynamicSmemBytes);
static_assert(kDownReductionOffset + kDownReductionBytes <= kDynamicSmemBytes);
static_assert(kDownReductionOffset + kDownReductionBytes <= kArLocalOffset);
static_assert(kArLocalOffset + kArPayloadBytes == kArResidualOffset);
static_assert(kArResidualOffset + kArPayloadBytes <= kDynamicSmemBytes);
static_assert(kArPayloadBytes == kArTileElements * sizeof(__nv_bfloat16));
static_assert(kArPayloadVectors * kArVectorBytes == kArPayloadBytes);
static_assert(kArLocalOffset % alignof(uint4) == 0);
static_assert(kArResidualOffset % alignof(uint4) == 0);
static_assert(kArRemoteOffset % alignof(uint4) == 0);
static_assert(kArTilesPerToken * kArTileStride == kArPhaseStride);
static_assert(2 * kArPhaseStride == kArSourceStride);
static_assert(kArRemoteSources * kArSourceStride == kArTokenStride);
static_assert(kTokens * kArTokenStride == kFusedMoePeerWorkspaceBytes);
static_assert(kArResidualOffset + kArPayloadBytes <= kArRemoteOffset);
static_assert(kArRemoteOffset + kArRemoteBytes <= kDynamicSmemBytes);
static_assert(kArRemoteSources * kArPayloadVectors <= kConsumerThreads);
static_assert(kArReductionWarps * kArReductionLanesPerWarp * kArReductionValuesPerThread ==
              kArTileElements);
static_assert(kArReductionWarps * sizeof(float) == sizeof(float4));
static_assert(kSharedActivationOffset + kSharedActivationBytes <= kDynamicSmemBytes);
static_assert((kPipelineStages & (kPipelineStages - 1)) == 0);
static_assert(kAuxOffset < kDynamicSmemBytes);
static_assert(kTopkBiasOffset % kRmsInputCopyBytes == 0);
static_assert(kTopkEndOffset <= kDynamicSmemBytes);
static_assert(kRoutedExperts % 4 == 0);
static_assert(offsetof(SharedStorage, scratch) % alignof(uint4) == 0);
static_assert(kRouterLogitsOffset % alignof(uint4) == 0);
static_assert(kThreads == kConsumerThreads + kProducerThreads);
static_assert(kConsumerWarps == kRouterExpertsPerCta);
static_assert(kProducerWarps == 4);
static_assert(kGateUpTmaKChunks * kGateUpTmaChunkBytes == kGateUpScaleOffset);
static_assert(kGateUpScaleOffset + kOriGateUpScaleMatrixStride + 0x20 <= kStageBytes);
static_assert(kGateUpTmaWeightCopies % kProducerWarps == 0);
static_assert(kGateUpTmaWeightCopies * kGateUpTmaAtomPairBytes + kGateUpScaleTmaBytes ==
              kOriGateUpTransactionBytes);
static_assert(kLocalInter % kDownKPerTask == 0);
static_assert(kDownKTasks == 1);
static_assert(kDownPhysicalTmaKAtoms == kDownKTasks * kDownStagedTmaKAtoms);
static_assert(kDownStagedTmaKAtoms == 2);
static_assert(kDownK32Partitions == kDownKTasks * kDownK32PartitionsPerTask);
static_assert(kDownK32PartitionsPerTask == kConsumerWarps);
static_assert(48 * kDownKPerTask == kDownWeightTransactionBytes);
static_assert(kDownStageExpertStride % 1024 == 0);
static_assert(2 * kDownStageExpertStride <= kStageBytes);
static_assert(kDownActivationSlots == 1 + kTopK);
static_assert(kDownExpertPairs == 5);
static_assert(kDownVirtualTilesPerCta * 48 == 192);
static_assert(kRouterPairsPerCta == 4);
static_assert(kRouterPairExpertBytes == 0x3000);
static_assert(kRouterPairTransactionBytes == 0x6000);
static_assert(kRouterPairTransactionBytes <= kStageBytes);
static_assert(kRouterPackedValues % kConsumerThreads == 0);
static_assert(kRouterPairChunksPerThread == 12);
static_assert(kWarpSize % kConsumerWarps == 0);
static_assert(kRouterReductionExpertsPerWarp == 4);
static_assert(kRouterReductionWarps == 2);
static_assert(kRouterExpertsPerCta * kConsumerWarps <= kThreads);
static_assert(kRoutedGateUpPipelineTasks == 24);
static_assert(kSharedGateUpPipelineTasks == 6);
static_assert(kDownPipelineTasks == 20);
static_assert(kRoutedGateUpPipelineTaskBase == 5);
static_assert(kRoutedDownPipelineTaskBase == 29);
static_assert(kRmsGammaTransactionBytes <= kStageBytes);
static_assert(kRmsGammaBytesPerProducer * kProducerWarps == kRmsGammaTransactionBytes);
static_assert(kRmsGammaBytesPerProducer % kRmsInputCopyBytes == 0);
static_assert(kConsumerThreads * kRmsInputCopiesPerConsumer * kRmsInputCopyBytes ==
              kHidden * static_cast<int>(sizeof(__nv_bfloat16)));
#elif GLM5_MEGAMOE_SEQ_LEN == 4 && GLM5_MEGAMOE_TP_SIZE == 4
static_assert(sizeof(StagedActivation) == sizeof(__nv_bfloat16));
static_assert(sizeof(PipelineControl) <= 0x400);
static_assert(kLocalTopkBytes == 0x40);
static_assert(kTopkHandoffBytes == 8);
static_assert(kGateUpCachePolicyBytes == 8);
static_assert(kPeerWorkspaceCacheBytes == 0x40);
static_assert(sizeof(PipelineControl) + kLocalTopkBytes + kTopkHandoffBytes +
                  kGateUpCachePolicyBytes + kPeerWorkspaceCacheBytes <=
              0x400);
static_assert(sizeof(SharedStorage) == kStaticSmemBytes);
static_assert(kWorkspaceRequiredBytes <= kDefaultWorkspaceBytes);
static_assert(kFusedMoeSeqLen == kTokens);
static_assert(kMinimumWorkspaceBytes == kWorkspaceRequiredBytes);
static_assert(kFusedMoeTpSize == kTpSize,
              "MTP314 TP4 source requires GLM5_MEGAMOE_TP_SIZE=4");
static_assert(kFusedMoeLocalInter == kLocalInter);
static_assert(kRoutedCtas == kFusedMoeEpochCounterCount);
static_assert(kRoutedCtas / kTokens == kFusedMoeRoutedCtasPerToken);
static_assert(kArProducerWarps * kArPayloadVectorsPerProducer == kArPayloadVectors);
static_assert(kNormalizedReadyOffset ==
              kUpGateOffset + kTokens * 9 * kLocalInter * sizeof(__nv_bfloat16));
static_assert(kTopkItems == kWarpSize);
static_assert(kHotExpertMaskWords == 8);
static_assert(kHotExpertBudget == 8);
static_assert(kTopkStatsCta == 132);
static_assert(kTopkStatsCta < kSharedCtaEnd);
static_assert(kTopkClaimOffset == 0x16010);
static_assert(kTopkReadyOffset == 0x16020);
static_assert(kHotExpertMaskOffset == 0x16030);
static_assert(kTopkFrequencyReadyOffset == 0x16050);
static_assert(kWorkspaceRequiredBytes == 0x16054);
static_assert(kPipelineStages * kStageBytes == kAuxOffset);
static_assert(kDownActivationBytes % kRmsInputCopyBytes == 0);
static_assert(kSharedActivationBytes % kRmsInputCopyBytes == 0);
static_assert(kAuxOffset + kDownActivationBytes <= kDynamicSmemBytes);
static_assert(kDownReductionOffset + kDownReductionBytes <= kDynamicSmemBytes);
static_assert(kDownReductionOffset + kDownReductionBytes <= kArLocalOffset);
static_assert(kArLocalOffset + kArPayloadBytes == kArResidualOffset);
static_assert(kArResidualOffset + kArPayloadBytes <= kDynamicSmemBytes);
static_assert(kArPayloadBytes == kArTileElements * sizeof(__nv_bfloat16));
static_assert(kArPayloadVectors * kArVectorBytes == kArPayloadBytes);
static_assert(kArLocalOffset % alignof(uint4) == 0);
static_assert(kArResidualOffset % alignof(uint4) == 0);
static_assert(kArRemoteOffset % alignof(uint4) == 0);
static_assert(kArTilesPerToken * kArTileStride == kArPhaseStride);
static_assert(2 * kArPhaseStride == kArSourceStride);
static_assert(kArRemoteSources * kArSourceStride == kArTokenStride);
static_assert(kTokens * kArTokenStride == kFusedMoePeerWorkspaceBytes);
static_assert(kArResidualOffset + kArPayloadBytes <= kArRemoteOffset);
static_assert(kArRemoteOffset + kArRemoteBytes <= kDynamicSmemBytes);
static_assert(kArRemoteSources * kArPayloadVectors <= kConsumerThreads);
static_assert(kArReductionWarps * kArReductionLanesPerWarp * kArReductionValuesPerThread ==
              kArTileElements);
static_assert(kArReductionWarps * sizeof(float) == sizeof(float4));
static_assert(kSharedActivationOffset + kSharedActivationBytes <= kDynamicSmemBytes);
static_assert((kPipelineStages & (kPipelineStages - 1)) == 0);
static_assert(kAuxOffset < kDynamicSmemBytes);
static_assert(kTopkBiasOffset % kRmsInputCopyBytes == 0);
static_assert(kTopkEndOffset <= kDynamicSmemBytes);
static_assert(kRoutedExperts % 4 == 0);
static_assert(offsetof(SharedStorage, scratch) % alignof(uint4) == 0);
static_assert(kRouterLogitsOffset % alignof(uint4) == 0);
static_assert(kThreads == kConsumerThreads + kProducerThreads);
static_assert(kConsumerWarps == kRouterExpertsPerCta);
static_assert(kProducerWarps == 4);
static_assert(kGateUpTmaKChunks * kGateUpTmaChunkBytes == kGateUpScaleOffset);
static_assert(kGateUpScaleOffset + kOriGateUpScaleMatrixStride + 0x20 <= kStageBytes);
static_assert(kGateUpTmaWeightCopies % kProducerWarps == 0);
static_assert(kGateUpTmaWeightCopies * kGateUpTmaAtomPairBytes + kGateUpScaleTmaBytes ==
              kOriGateUpTransactionBytes);
static_assert(kLocalInter % kDownKPerTask == 0);
static_assert(kDownKTasks == 2);
static_assert(kDownPhysicalTmaKAtoms == kDownKTasks * kDownStagedTmaKAtoms);
static_assert(kDownStagedTmaKAtoms == 2);
static_assert(kDownK32Partitions == kDownKTasks * kDownK32PartitionsPerTask);
static_assert(kDownK32PartitionsPerTask == kConsumerWarps);
static_assert(48 * kDownKPerTask == kDownWeightTransactionBytes);
static_assert(kDownStageExpertStride % 1024 == 0);
static_assert(2 * kDownStageExpertStride <= kStageBytes);
static_assert(kDownActivationSlots == 1 + kTopK);
static_assert(kDownExpertPairs == 5);
static_assert(kDownVirtualTilesPerCta * 48 == 192);
static_assert(kRouterPairsPerCta == 4);
static_assert(kRouterPairExpertBytes == 0x3000);
static_assert(kRouterPairTransactionBytes == 0x6000);
static_assert(kRouterPairTransactionBytes <= kStageBytes);
static_assert(kRouterPackedValues % kConsumerThreads == 0);
static_assert(kRouterPairChunksPerThread == 12);
static_assert(kWarpSize % kConsumerWarps == 0);
static_assert(kRouterReductionExpertsPerWarp == 4);
static_assert(kRouterReductionWarps == 2);
static_assert(kRouterExpertsPerCta * kConsumerWarps <= kThreads);
static_assert(kRoutedGateUpPipelineTasks == 48);
static_assert(kSharedGateUpPipelineTasks == 6);
static_assert(kDownPipelineTasks == 40);
static_assert(kRoutedGateUpPipelineTaskBase == 5);
static_assert(kRoutedDownPipelineTaskBase == 53);
static_assert(kRmsGammaTransactionBytes <= kStageBytes);
static_assert(kRmsGammaBytesPerProducer * kProducerWarps == kRmsGammaTransactionBytes);
static_assert(kRmsGammaBytesPerProducer % kRmsInputCopyBytes == 0);
static_assert(kConsumerThreads * kRmsInputCopiesPerConsumer * kRmsInputCopyBytes ==
              kHidden * static_cast<int>(sizeof(__nv_bfloat16)));
#elif GLM5_MEGAMOE_SEQ_LEN == 8 && GLM5_MEGAMOE_TP_SIZE == 8
static_assert(sizeof(StagedActivation) == sizeof(__nv_bfloat16));
static_assert(sizeof(PipelineControl) <= 0x400);
static_assert(sizeof(PipelineControl) == 2 * kPipelineStages * sizeof(std::uint64_t));
static_assert(kLocalTopkBytes == 0x40);
static_assert(kTopkHandoffBytes == 8);
static_assert(kPeerWorkspaceCacheBytes == 0x40);
static_assert(sizeof(PipelineControl) + kLocalTopkBytes + kTopkHandoffBytes +
                  kPeerWorkspaceCacheBytes <=
              0x580);
static_assert(sizeof(SharedStorage) == kStaticSmemBytes);
static_assert(kWorkspaceRequiredBytes <= kDefaultWorkspaceBytes);
static_assert(kFusedMoeSeqLen == kTokens);
static_assert(kMinimumWorkspaceBytes == kWorkspaceRequiredBytes);
static_assert(kRouterLogitsOffset == kNormOffset + kTokens * kHidden * sizeof(__nv_bfloat16));
static_assert(kUpGateOffset == kRouterLogitsOffset + kTokens * kRoutedExperts * sizeof(float));
static_assert(kNormalizedReadyOffset ==
              kUpGateOffset + kTokens * 9 * kLocalInter * sizeof(__nv_bfloat16));
static_assert(kWorkspaceRequiredBytes >= kNormalizedReadyOffset + kTokens * sizeof(std::uint32_t));
static_assert(kPipelineStages * kStageBytes == kAuxOffset);
static_assert(kDownActivationBytes % kRmsInputCopyBytes == 0);
static_assert(kSharedActivationPageBytes % kRmsInputCopyBytes == 0);
static_assert(kAuxOffset + kDownActivationBytes <= kDynamicSmemBytes);
static_assert(kDownReductionOffset + kDownReductionBytes <= kDynamicSmemBytes);
static_assert(kDownReductionOffset + kDownReductionBytes <= kArLocalOffset);
static_assert(kSharedSentinelOffset + kSharedSentinelBytes <= kArLocalOffset);
static_assert(kArLocalOffset + kArMaxPayloadBytes == kArResidualOffset);
static_assert(kArResidualOffset + kArMaxPayloadBytes == kArRemoteOffset);
static_assert(kArRemoteOffset + kArRemoteBytes <= kDynamicSmemBytes);
static_assert(kArRemoteOffset + sizeof(uint4) + kArMaxPayloadBytes <= kDynamicSmemBytes);
static_assert(kArMaxPayloadVectors * sizeof(uint4) == kArMaxPayloadBytes);
static_assert(kArMaxVirtualTilesPerCta * kArVirtualTileStride <= 0x400);
static_assert(128 * kArVirtualTileStride == kArPhaseStride);
static_assert(2 * kArPhaseStride == kArSourceStride);
static_assert(kArRemoteSources * kArSourceStride == kArTokenStride);
static_assert(kTokens * kArTokenStride == kFusedMoeTp8PeerWorkspaceBytes);
static_assert(kTokens * kRoutedCtasPerToken == kFusedMoeTp8EpochCounterCount);
static_assert(kRoutedCtasPerToken == kFusedMoeRoutedCtasPerToken);
static_assert(kSharedActivationPageBytes == 0x4000);
static_assert(kSharedActivationStorageBytes == 0x8000);
static_assert(kSharedPartialBytes == 0x2000);
static_assert(kSharedResultBytes == 0x1000);
static_assert(kSharedSentinelBytes == 0x40);
static_assert(kSharedActivationOffset == 0x24000);
static_assert(kSharedPartialOffset == 0x2c000);
static_assert(kRoutedActivationOffset + kHidden * sizeof(__nv_bfloat16) <= kArLocalOffset);
static_assert(kSharedResultOffset == 0x2e000);
static_assert(kSharedSentinelOffset + kSharedSentinelBytes <= kDynamicSmemBytes);
static_assert((kPipelineStages & (kPipelineStages - 1)) == 0);
static_assert(kAuxOffset < kDynamicSmemBytes);
static_assert(kTopkBiasOffset % kRmsInputCopyBytes == 0);
static_assert(kTopkEndOffset <= kDynamicSmemBytes);
static_assert(kRoutedExperts % 4 == 0);
static_assert(offsetof(SharedStorage, scratch) % alignof(uint4) == 0);
static_assert(kRouterLogitsOffset % alignof(uint4) == 0);
static_assert(kThreads == kConsumerThreads + kProducerThreads);
static_assert(kProducerWarps == 4);
static_assert(kGateUpTmaKChunks * kGateUpTmaChunkBytes == kGateUpScaleOffset);
static_assert(kGateUpScaleOffset + kGateUpScaleMatrixStride + 0x20 <= kStageBytes);
static_assert(kGateUpTmaWeightCopies % kProducerWarps == 0);
static_assert(kGateUpNativeTransactionBytes == 0x8060);
static_assert(48 * kLocalInter == 0x3000);
static_assert(kDownWeightTmaBytes + 0x20 == kDownNativeTransactionBytes);
static_assert(kDownTmaIssueWarps + 1 == kProducerWarps);
static_assert(kDownExpertStageStride % 1024 == 0);
static_assert(kDownActivationSlots == 1 + kTopK);
static_assert(kDownExpertPairs == 5);
static_assert(kDownM16Tiles == 384);
static_assert(kDownM16Tiles % kDownM16PerVirtualTile == 0);
static_assert(kDownMaxM16PerCta == 24);
static_assert(kRouterMaxPairsPerCta == 8);
static_assert(kRouterPairExpertBytes == 0x3000);
static_assert(kRouterPairMaxTransactionBytes == 0x6000);
static_assert(kRouterPairMaxTransactionBytes <= kStageBytes);
static_assert(kRouterPackedValues % kConsumerThreads == 0);
static_assert(kRouterPairChunksPerThread == 12);
static_assert(kWarpSize % kConsumerWarps == 0);
static_assert(kRouterPartialElements <= kThreads);
static_assert(kRouterReductionExpertsPerWarp == 4);
static_assert(kRouterReductionWarps == 4);
static_assert(kRoutedGateUpMaxPipelineTasks == 48);
static_assert(kSharedGateUpPipelineTasks == 24);
static_assert(kSharedGateUpPipelineTaskBase == 2);
static_assert(kRmsGammaTransactionBytes <= kStageBytes);
static_assert(kRmsGammaBytesPerProducer * kProducerWarps == kRmsGammaTransactionBytes);
static_assert(kRmsGammaBytesPerProducer % kRmsInputCopyBytes == 0);
static_assert(kConsumerThreads * kRmsInputCopiesPerConsumer * kRmsInputCopyBytes ==
              kHidden * static_cast<int>(sizeof(__nv_bfloat16)));
static_assert(kRoutedCtas == 144);
static_assert(kSharedCtaBegin == 144);
static_assert(kSharedCtas == 4);
static_assert(kSharedCtaEnd == kGridCtas);
static_assert(kRoutedGateUpResultElements == 32);
static_assert(kSharedGateUpResultElements == 1024);
static_assert(kDownReductionBytes == 0x3000);
static_assert(kDownExpertStageStride * 2 <= kStageBytes);
static_assert(validate_routed_descriptors<ActiveMoeConfig>());
#elif GLM5_MEGAMOE_SEQ_LEN == 8 && GLM5_MEGAMOE_TP_SIZE == 4
static_assert(sizeof(StagedActivation) == sizeof(__nv_bfloat16));
static_assert(sizeof(PipelineControl) <= 0x400);
static_assert(sizeof(PipelineControl) == 2 * kPipelineStages * sizeof(std::uint64_t));
static_assert(kLocalTopkBytes == 0x40);
static_assert(kTopkHandoffBytes == 8);
static_assert(kPeerWorkspaceCacheBytes == 0x40);
static_assert(sizeof(PipelineControl) + kLocalTopkBytes + kTopkHandoffBytes +
                  kPeerWorkspaceCacheBytes <=
              0x500);
static_assert(sizeof(SharedStorage) == kStaticSmemBytes);
static_assert(kWorkspaceRequiredBytes <= kDefaultWorkspaceBytes);
static_assert(kFusedMoeSeqLen == kTokens);
static_assert(kMinimumWorkspaceBytes == kWorkspaceRequiredBytes);
static_assert(kRouterLogitsOffset == kNormOffset + kTokens * kHidden * sizeof(__nv_bfloat16));
static_assert(kUpGateOffset == kRouterLogitsOffset + kTokens * kRoutedExperts * sizeof(float));
static_assert(kNormalizedReadyOffset ==
              kUpGateOffset + kTokens * 9 * kLocalInter * sizeof(__nv_bfloat16));
static_assert(kWorkspaceRequiredBytes == kNormalizedReadyOffset + kTokens * sizeof(std::uint32_t));
static_assert(kPipelineStages * kStageBytes == kAuxOffset);
static_assert(kDownActivationBytes % kRmsInputCopyBytes == 0);
static_assert(kSharedActivationPageBytes % kRmsInputCopyBytes == 0);
static_assert(kAuxOffset + kDownActivationBytes <= kDynamicSmemBytes);
static_assert(kDownReductionOffset + kDownReductionBytes <= kDynamicSmemBytes);
static_assert(kDownReductionOffset + kDownReductionBytes <= kArLocalOffset);
static_assert(kSharedSentinelOffset + kSharedSentinelBytes <= kArLocalOffset);
static_assert(kRoutedActivationOffset + kHidden * sizeof(__nv_bfloat16) <= kArLocalOffset);
static_assert(kArLocalOffset + kArMaxPayloadBytes == kArResidualOffset);
static_assert(kArResidualOffset + kArMaxPayloadBytes == kArRemoteOffset);
static_assert(kArRemoteOffset + kArRemoteBytes <= kDynamicSmemBytes);
static_assert(kArRemoteOffset + sizeof(uint4) + kArMaxPayloadBytes <= kDynamicSmemBytes);
static_assert(kArMaxPayloadVectors * sizeof(uint4) == kArMaxPayloadBytes);
static_assert(kArMaxVirtualTilesPerCta * kArVirtualTileStride <= 0x400);
static_assert(128 * kArVirtualTileStride == kArPhaseStride);
static_assert(2 * kArPhaseStride == kArSourceStride);
static_assert(kArRemoteSources * kArSourceStride == kArTokenStride);
static_assert(kTokens * kArTokenStride == kFusedMoePeerWorkspaceBytes);
static_assert(kTokens * kRoutedCtasPerToken == kFusedMoeEpochCounterCount);
static_assert(kRoutedCtasPerToken == kFusedMoeRoutedCtasPerToken);
static_assert(kSharedActivationPageBytes == 0x4000);
static_assert(kSharedActivationStorageBytes == 0x8000);
static_assert(kSharedSentinelBytes == 0x80);
static_assert(kSharedActivationOffset == 0x24000);
static_assert(kSharedSentinelOffset == 0x2c000);
static_assert(kSharedSentinelOffset + kSharedSentinelBytes <= kDynamicSmemBytes);
static_assert((kPipelineStages & (kPipelineStages - 1)) == 0);
static_assert(kAuxOffset < kDynamicSmemBytes);
static_assert(kTopkBiasOffset % kRmsInputCopyBytes == 0);
static_assert(kTopkEndOffset <= kDynamicSmemBytes);
static_assert(kRoutedExperts % 4 == 0);
static_assert(offsetof(SharedStorage, scratch) % alignof(uint4) == 0);
static_assert(kRouterLogitsOffset % alignof(uint4) == 0);
static_assert(kThreads == kConsumerThreads + kProducerThreads);
static_assert(kFusedMoeTpSize == kTpSize,
              "MTP718 TP4 source requires GLM5_MEGAMOE_TP_SIZE=4");
static_assert(kFusedMoeLocalInter == kLocalInter);
static_assert(kProducerWarps == 4);
static_assert(kGateUpTmaKChunks * kGateUpTmaChunkBytes == kGateUpScaleOffset);
static_assert(kGateUpScaleOffset + kGateUpScaleMatrixStride + 0x20 <= kStageBytes);
static_assert(kGateUpTmaWeightCopies % kProducerWarps == 0);
static_assert(kGateUpLayoutFactor == 2);
static_assert(kGateUpRowGroupsPerExpert == 32);
static_assert(kGateUpScaleBlocksPerMatrix == 4);
static_assert(kGateUpScaleTmaBytes == 0xa0);
static_assert(kGateUpNativeTransactionBytes == 0x80a0);
static_assert(kDownTmaAtomTileBytes == 0x1800);
static_assert(kLocalInter % kDownKPerTask == 0);
static_assert(kDownKTasks == 2);
static_assert(kDownPhysicalTmaKAtoms == kDownKTasks * kDownStagedTmaKAtoms);
static_assert(kDownStagedTmaKAtoms == 2);
static_assert(kDownK32PartitionsPerTask == kConsumerWarps);
static_assert(kDownWeightTransactionBytes == 0x3000);
static_assert(kDownNativeScaleTransactionBytes == 0x40);
static_assert(kDownStageScaleOffset == kDownWeightTransactionBytes);
static_assert(kDownExpertStageStride == 0x3400);
static_assert(kDownTmaIssueWarps + 1 == kProducerWarps);
static_assert(kDownExpertStageStride % 1024 == 0);
static_assert(kDownActivationSlots == 1 + kTopK);
static_assert(kDownExpertPairs == 5);
static_assert(kDownM16Tiles == 384);
static_assert(kDownM16Tiles % kDownM16PerVirtualTile == 0);
static_assert(kDownMaxM16PerCta == 24);
static_assert(kRouterMaxPairsPerCta == 8);
static_assert(kRouterPairExpertBytes == 0x3000);
static_assert(kRouterPairMaxTransactionBytes == 0x6000);
static_assert(kRouterPairMaxTransactionBytes <= kStageBytes);
static_assert(kRouterPackedValues % kConsumerThreads == 0);
static_assert(kRouterPairChunksPerThread == 12);
static_assert(kWarpSize % kConsumerWarps == 0);
static_assert(kRouterPartialElements <= kThreads);
static_assert(kRouterReductionExpertsPerWarp == 4);
static_assert(kRouterReductionWarps == 4);
static_assert(kRoutedGateUpMaxPipelineTasks == 96);
static_assert(kSharedGateUpPipelineTasks == 48);
static_assert(kSharedGateUpPipelineTaskBase == 2);
static_assert(kRmsGammaTransactionBytes <= kStageBytes);
static_assert(kRmsGammaBytesPerProducer * kProducerWarps == kRmsGammaTransactionBytes);
static_assert(kRmsGammaBytesPerProducer % kRmsInputCopyBytes == 0);
static_assert(kConsumerThreads * kRmsInputCopiesPerConsumer * kRmsInputCopyBytes ==
              kHidden * static_cast<int>(sizeof(__nv_bfloat16)));
static_assert(kRoutedCtas == 144);
static_assert(kSharedCtaBegin == 144);
static_assert(kSharedCtas == 4);
static_assert(kSharedCtaEnd == kGridCtas);
static_assert(kSharedRowGroups == 32);
static_assert(kSharedRowGroupsPerCta == 8);
static_assert(kSharedConsumerThreadsPerGroup == kWarpSize);
static_assert(kRoutedGateUpResultElements == 64);
static_assert(kDownReductionBytes == 0x3000);
static_assert(kDownExpertStageStride * 2 <= kStageBytes);
static_assert(validate_routed_descriptors<ActiveMoeConfig>());
#endif

template <class Config>
  requires(Config::kTokens == 4)
__device__ __forceinline__ void run_megamoe(Glm5MegaMoeArgs const& args,
                                            MoeSharedStorage<Config>* shared,
                                            std::uint8_t* dynamic_storage) {
  static_assert(Config::kTokens == kFusedMoeSeqLen && Config::kTpSize == kFusedMoeTpSize);
  auto* workspace = static_cast<std::uint8_t*>(args.workspace);
  auto* shared_normalized = reinterpret_cast<__nv_bfloat16*>(workspace + kNormOffset);
  auto* router_logits = reinterpret_cast<float*>(workspace + kRouterLogitsOffset);
  auto* up_gate = reinterpret_cast<__nv_bfloat16*>(workspace + kUpGateOffset);
  auto* normalized_ready = reinterpret_cast<std::uint32_t*>(workspace + kNormalizedReadyOffset);
  auto* topk_claim = reinterpret_cast<std::uint32_t*>(workspace + kTopkClaimOffset);
  auto* topk_ready = reinterpret_cast<std::uint32_t*>(workspace + kTopkReadyOffset);
  auto* hot_expert_mask = reinterpret_cast<std::uint32_t*>(workspace + kHotExpertMaskOffset);
  auto* frequency_ready = reinterpret_cast<std::uint32_t*>(workspace + kTopkFrequencyReadyOffset);
  auto const* x = static_cast<__nv_bfloat16 const*>(args.x);
  auto const* router_weight = static_cast<__nv_bfloat16 const*>(args.router_weight);
  auto* local_output = static_cast<__nv_bfloat16*>(args.local_output);
  Layout layout{};
  int const cta = static_cast<int>(blockIdx.x);
  if (cta < kRoutedCtas && threadIdx.x == 0)
    cutlass::arch::ClusterBarrier::init(&shared->topk_handoff, 1);
  if (cta < kRoutedCtas && args.world_size == kFusedMoeWorldSize &&
      threadIdx.x < kFusedMoeWorldSize)
    shared->peer_workspaces[threadIdx.x] = args.peer_workspaces[threadIdx.x];
  {
    if (cta < kSharedCtaEnd) initialize_pipeline(&shared->pipeline);
  }

  // Keep RMS and router descriptor lifetimes scoped to their respective phases.
  if (cta < kRoutedCtas) {
    int const token = cta / kRoutedCtasPerToken;
    int const local = cta % kRoutedCtasPerToken;
    {
      routed_rms_norm_and_router_projection<Config>(
          x + static_cast<std::size_t>(token) * kHidden, args.rms_gamma, router_weight, token,
          local, shared->normalized, router_logits, shared->scratch, &shared->pipeline,
          dynamic_storage);
    }
  } else {
    if (cta >= kSharedCtaBegin && cta < kSharedCtaBegin + kTokens) {
      int const token = cta - kSharedCtaBegin;
      shared_rms_norm_and_publish<Config>(
          x + static_cast<std::size_t>(token) * kHidden, args.rms_gamma, shared->normalized,
          shared_normalized + static_cast<std::size_t>(token) * kHidden, normalized_ready, token,
          shared->scratch, &shared->pipeline, dynamic_storage);
    }
  }

  // Four-token TP4's first consumer warp aggregates cross-token expert reuse.
  if constexpr (Config::kTopkFrequency) {
    if (cta == kTopkStatsCta && threadIdx.x < kWarpSize)
      collect_topk_frequency<Config>(args.selected_indices, topk_ready, hot_expert_mask,
                                     frequency_ready, args.ar_epoch_counters, dynamic_storage);
  }
  if (cta < kRoutedCtas) {
    int const token = cta / kRoutedCtasPerToken;
    int const local = cta % kRoutedCtasPerToken;
    if constexpr (Config::kTopkFrequency) {
      int const counter = token * kArTilesPerToken + local;
      select_topk<Config>(router_logits, args.router_bias, token, counter,
                          shared->local_selected_indices, shared->local_selected_probs,
                          args.selected_indices, args.selected_probs, topk_claim, topk_ready,
                          args.ar_epoch_counters, dynamic_storage);
      if (threadIdx.x == 0)
        prepare_routed_gate_up_cache<Config>(
            token, local, shared->local_selected_indices, hot_expert_mask, frequency_ready,
            args.ar_epoch_counters, &shared->gate_up_cache_policy, &shared->topk_handoff);
    } else {
      select_topk<Config>(router_logits, args.router_bias, token, local >> 2,
                          shared->local_selected_indices, shared->local_selected_probs,
                          args.selected_indices, args.selected_probs, local == 0,
                          &shared->topk_handoff, dynamic_storage);
    }
  }

  // GateUp + SiLU: keep routed/shared staging and producer roles separate.
  if (cta < kRoutedCtas) {
    int const token = cta / kRoutedCtasPerToken;
    int const local = cta % kRoutedCtasPerToken;
    if constexpr (Config::kTopkFrequency) {
      routed_gate_up<Config>(layout, args, shared->normalized, token, local,
                             shared->local_selected_indices, &shared->gate_up_cache_policy,
                             &shared->topk_handoff, up_gate, shared->gate_up, &shared->pipeline,
                             dynamic_storage);
    } else {
      routed_gate_up<Config>(layout, args, shared->normalized, token, local,
                             shared->local_selected_indices, nullptr, &shared->topk_handoff, up_gate,
                             shared->gate_up, &shared->pipeline, dynamic_storage);
    }
  } else {
    if (cta >= kSharedCtaBegin && cta < kSharedCtaEnd) {
      int const base = cta - kSharedCtaBegin;
#pragma unroll 1
      for (int pass = 0; pass < kSharedGateUpRowGroupsPerCta; ++pass) {
        if (pass != 0) initialize_pipeline(&shared->pipeline);
        shared_gate_up<Config>(layout, args, shared_normalized, normalized_ready, base + pass * 16,
                               up_gate, shared->gate_up, &shared->pipeline, dynamic_storage);
      }
    }
  }

  // Run Down projection, then finalize local or TP output with BF16 rounding.
  if (cta < kRoutedCtas) {
    int const token = cta / kRoutedCtasPerToken;
    int const local = cta % kRoutedCtasPerToken;
    auto* partial = args.world_size == 0
                        ? nullptr
                        : reinterpret_cast<__nv_bfloat16*>(dynamic_storage + kArLocalOffset);
    {
      routed_down<Config>(
          layout, args, up_gate,
          args.world_size == 0 ? nullptr : static_cast<__nv_bfloat16 const*>(args.residual),
          shared->local_selected_indices, shared->local_selected_probs, token, local, local_output,
          partial, &shared->pipeline, dynamic_storage);
      if (args.world_size != 0)
        tp_allreduce_and_finalize<Config>(args, shared->peer_workspaces, token, local,
                                          dynamic_storage);
    }
  }
}

}  // namespace glm5::moe::detail

namespace glm5::moe {

#if GLM5_MEGAMOE_SEQ_LEN == 8
#ifndef GLM5_MEGAMOE_KERNEL_NAME
#error "The eight-token entry must name its global kernel before including this header"
#endif

// Keep the grid-constant argument in the global entry and use shared device
// templates for each stage to preserve scheduling across specializations.
extern "C" __global__ __launch_bounds__(
    detail::ActiveMoeConfig::kThreads,
    1) void GLM5_MEGAMOE_KERNEL_NAME(__grid_constant__ Glm5MegaMoeArgs const args) {
  using namespace detail;
  using Glm5Config = detail::ActiveMoeConfig;

  __shared__ __align__(16) std::uint8_t static_storage[kStaticSmemBytes];
  extern __shared__ __align__(16) std::uint8_t dynamic_storage[];

  auto* shared = reinterpret_cast<SharedStorage*>(static_storage);
  auto* workspace = static_cast<std::uint8_t*>(args.workspace);
  auto* shared_normalized = reinterpret_cast<__nv_bfloat16*>(workspace + kNormOffset);
  auto* router_logits = reinterpret_cast<float*>(workspace + kRouterLogitsOffset);
  auto* up_gate = reinterpret_cast<__nv_bfloat16*>(workspace + kUpGateOffset);
  auto* normalized_ready = reinterpret_cast<std::uint32_t*>(workspace + kNormalizedReadyOffset);

  auto const* x = static_cast<__nv_bfloat16 const*>(args.x);
  auto const* router_weight = static_cast<__nv_bfloat16 const*>(args.router_weight);
  auto* local_output = static_cast<__nv_bfloat16*>(args.local_output);

  int const cta = static_cast<int>(blockIdx.x);
  if (cta < kRoutedCtas && threadIdx.x == 0) {
    cutlass::arch::ClusterBarrier::init(&shared->topk_handoff, 1);
  }
  if (cta < kRoutedCtas && args.world_size == kFusedMoeWorldSize &&
      threadIdx.x < kFusedMoeWorldSize) {
    shared->peer_workspaces[threadIdx.x] = args.peer_workspaces[threadIdx.x];
  }
  initialize_pipeline(&shared->pipeline,
                      cta < kRoutedCtas ? kConsumerThreads : kSharedConsumerThreadsPerGroup);

  // Phase 1: 18 Routed CTAs compute disjoint Router expert tiles for each
  // token. Each Shared CTA normalizes and publishes two tokens concurrently.
  if (cta < kRoutedCtas) {
    int const token = cta / kRoutedCtasPerToken;
    int const routed_local = cta % kRoutedCtasPerToken;
    RoutedCtaDescriptor const descriptor = routed_cta_descriptor<Glm5Config>(routed_local);
    routed_rms_norm_and_router_projection<Glm5Config>(
        x + static_cast<std::size_t>(token) * kHidden, args.rms_gamma, router_weight, token,
        descriptor.router_expert_base, descriptor.router_expert_count, shared->normalized,
        router_logits, shared->scratch, &shared->pipeline, dynamic_storage);
  } else {
    int const shared_local = cta - kSharedCtaBegin;
    int const first_token = shared_local * kSharedTokensPerCta;
    shared_rms_norm_and_publish_pair<Glm5Config>(
        x + static_cast<std::size_t>(first_token) * kHidden, args.rms_gamma, shared->normalized,
        shared_normalized + static_cast<std::size_t>(first_token) * kHidden, normalized_ready,
        first_token, shared->scratch, &shared->pipeline, dynamic_storage);
  }

  // Phase 2: every Routed CTA polls the token's 256 sentinel-backed
  // logits, then redundantly computes the same CTA-local Top-8. Only tile 0
  // publishes the public debug tensors.
  if (cta < kRoutedCtas) {
    int const token = cta / kRoutedCtasPerToken;
    int const routed_local = cta % kRoutedCtasPerToken;
    RoutedCtaDescriptor const descriptor = routed_cta_descriptor<Glm5Config>(routed_local);
    int const first_selected_slot = descriptor.gate_flat_base / kGateUpRowGroupsPerExpert;
    int const last_selected_slot =
        (descriptor.gate_flat_base + descriptor.gate_group_count - 1) / kGateUpRowGroupsPerExpert;
    select_topk<Glm5Config>(
        router_logits, args.router_bias, token, first_selected_slot, last_selected_slot,
        shared->local_selected_indices, shared->local_selected_probs, args.selected_indices,
        args.selected_probs, routed_local == 0, &shared->topk_handoff, dynamic_storage);
  }

  // Phase 3: all Routed CTAs compute their selected expert tiles. Each of the
  // four Shared CTAs computes four row groups with two warps per row group.
  if (cta < kRoutedCtas) {
    int const token = cta / kRoutedCtasPerToken;
    int const routed_local = cta % kRoutedCtasPerToken;
    RoutedCtaDescriptor const descriptor = routed_cta_descriptor<Glm5Config>(routed_local);
    int const gate_task_base = kRouterPipelineTaskBase + (descriptor.router_expert_count + 1) / 2;
    if (threadIdx.x == 0 && args.ar_epoch_counters != nullptr) {
      int const counter = token * kRoutedCtasPerToken + routed_local;
      auto* generation_scratch =
          reinterpret_cast<std::uint32_t*>(dynamic_storage + kArRemoteOffset);
      generation_scratch[0] =
          comm::next_generation(comm::load_relaxed_gpu_u32(args.ar_epoch_counters + counter));
    }
    routed_gate_up<Glm5Config>(args, shared->normalized, token, descriptor.gate_flat_base,
                               descriptor.gate_group_count, gate_task_base,
                               shared->local_selected_indices, &shared->topk_handoff, up_gate,
                               shared->gate_up, &shared->pipeline, dynamic_storage);
  } else {
    int const shared_local = cta - kSharedCtaBegin;
    shared_gate_up<Glm5Config>(args, shared_normalized, normalized_ready,
                               shared_local * kSharedRowGroupsPerCta, kSharedGateUpPipelineTaskBase,
                               up_gate, &shared->pipeline, dynamic_storage);
  }

  // Phase 4: Routed CTAs consume all nine Gate/Up activation slots and write
  // disjoint 16-row Down tiles. Shared CTAs have completed their role.
  if (cta < kRoutedCtas) {
    int const token = cta / kRoutedCtasPerToken;
    int const routed_local = cta % kRoutedCtasPerToken;
    RoutedCtaDescriptor const descriptor = routed_cta_descriptor<Glm5Config>(routed_local);
    int const gate_task_base = kRouterPipelineTaskBase + (descriptor.router_expert_count + 1) / 2;
    int const down_task_base = gate_task_base + descriptor.gate_group_count * kGateUpPagesPerGroup;
    int const world_size = args.world_size;
    auto* ar_local_partial =
        world_size == 0 ? nullptr
                        : reinterpret_cast<__nv_bfloat16*>(dynamic_storage + kArLocalOffset);
    down_and_finalize_k32_partitioned<Glm5Config>(
        args, up_gate, shared->local_selected_indices, shared->local_selected_probs, token,
        routed_local, descriptor.down_m16_base, descriptor.down_m16_count, down_task_base,
        local_output, ar_local_partial, &shared->pipeline, dynamic_storage);
    if (world_size != 0) {
      comm::allreduce_and_finalize<TpAllReduceConfig>(
          args, shared->peer_workspaces, token, descriptor.down_m16_base / kDownM16PerVirtualTile,
          down_virtual_tile_count<Glm5Config>(descriptor.down_m16_count),
          descriptor.down_m16_base * kDownM16Rows, dynamic_storage);
    }
  }
}
#endif

}  // namespace glm5::moe
