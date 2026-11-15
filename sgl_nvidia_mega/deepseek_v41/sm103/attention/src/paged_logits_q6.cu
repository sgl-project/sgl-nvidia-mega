// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// Adapted from DeepGEMM (https://github.com/deepseek-ai/DeepGEMM),
// deep_gemm/include/deep_gemm/legacy/impls/sm100_mqa_logits.cuh (sm100_mqa_logits_core_impl,
// sm100_paged_mqa_logits) and deep_gemm/legacy/scheduler/sm100_paged_mqa_logits.cuh
// (SM100PagedMQALogitsScheduler) as SGLang's 2026-11-10 nightly ships them; modified by
// NVIDIA (one Q block of all of a request's rows, two TMEM stages when three do not fit, the
// paged MXFP4 path only). Its license:
//
//   MIT License
//
//   Copyright (c) 2025 DeepSeek
//
//   Permission is hereby granted, free of charge, to any person obtaining a copy
//   of this software and associated documentation files (the "Software"), to deal
//   in the Software without restriction, including without limitation the rights
//   to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
//   copies of the Software, and to permit persons to whom the Software is
//   furnished to do so, subject to the following conditions:
//
//   The above copyright notice and this permission notice shall be included in all
//   copies or substantial portions of the Software.
//
//   THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
//   IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
//   FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
//   AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
//   LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
//   OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
//   SOFTWARE.
//
// DeepGEMM's paged MXFP4 MQA logits with bf16 weights and logits (no clean, grouped rows,
// 2D context lengths), bit for bit: DeepGEMM fixes BLOCK_Q = 128 / heads, so 32 heads take
// 4 rows per Q block and 6 verify rows run as two Q blocks of 3, each streaming the whole
// KV range and issuing its own MMAs. Here a request's Q6_ROWS rows (up to 7) are one Q
// block, UMMA_N = rows * heads: the KV streams once per SM range. Every logit is the same
// UMMA output column (a row-head of B) reduced by the same thread with the same bf16 FMAs,
// so the bits do not change. TMEM holds two accumulator stages when three do not fit; the
// math warps then release a stage before they reduce it.
// DeepGEMM's own metadata (get_paged_mqa_logits_metadata) partitions the work.
#include <deep_gemm/legacy/impls/sm100_mqa_logits.cuh>

#if !defined(Q6_ROWS) || !defined(Q6_PAGE_KV) || !defined(Q6_KV_STAGES)
#error "define Q6_ROWS, Q6_PAGE_KV and Q6_KV_STAGES"
#endif
#ifndef Q6_EARLY_RELEASE
#define Q6_EARLY_RELEASE 1
#endif

namespace deep_gemm::legacy::dsv41 {

// SM100PagedMQALogitsScheduler with BLOCK_Q a parameter instead of 128 / kNumHeads
template <uint32_t kNextN, uint32_t BLOCK_Q_, uint32_t SPLIT_KV, uint32_t PAGE_KV, uint32_t kSplitsPerChunk>
struct PagedScheduler {
    static constexpr bool kIsPaged = true;
    static constexpr bool kIsVarlenLayout = false;
    static constexpr bool kIsContextLens2DLayout = true;
    static constexpr uint32_t kTokensPerRequest = kNextN;
    static constexpr bool kHasPartialBlock = true;
    static constexpr uint32_t kPageKV = PAGE_KV;
    static constexpr uint32_t kNumPagesPerSplit = SPLIT_KV / PAGE_KV;
    static constexpr uint32_t BLOCK_Q = BLOCK_Q_;

    using Info = sched::RequestInfo<kNextN, true, false, BLOCK_Q, SPLIT_KV, PAGE_KV>;

    const uint32_t* context_lens;
    const uint32_t* block_table;
    uint32_t block_table_stride;
    uint32_t num_q_tokens_total;

    Info cur;
    uint32_t cur_kv_split_base;
    uint32_t cur_q_block_in_request;
    uint32_t end_q_token_idx, end_kv_split_idx;
    bool done;

    uint32_t cur_block_table_row;
    uint32_t cur_q_block_token_base;
    uint32_t cur_num_block_tokens;
    uint32_t cur_request_num_kv_pages;

    CUTLASS_DEVICE PagedScheduler(const uint32_t& sm_idx,
                                  const uint32_t* context_lens,
                                  const uint32_t* schedule_meta,
                                  const uint32_t* block_table,
                                  const uint32_t& block_table_stride,
                                  const uint32_t& num_q_tokens_total) {
        this->context_lens = context_lens;
        this->block_table = block_table;
        this->block_table_stride = block_table_stride;
        this->num_q_tokens_total = num_q_tokens_total;

        const auto start = reinterpret_cast<const uint2*>(schedule_meta)[sm_idx];
        const auto end = reinterpret_cast<const uint2*>(schedule_meta)[sm_idx + 1];
        end_q_token_idx = end.x;
        end_kv_split_idx = end.y;

        cur_kv_split_base = start.y;
        cur_q_block_in_request = 0;
        done = (start.x >= num_q_tokens_total) or
               (start.x == end_q_token_idx and start.y >= end_kv_split_idx);
        if (not done)
            cur = Info::from_q_token(start.x, num_q_tokens_total, context_lens, nullptr);

        cur_block_table_row = 0;
        cur_q_block_token_base = 0;
        cur_num_block_tokens = 1;
        cur_request_num_kv_pages = 0;
    }

    CUTLASS_DEVICE uint32_t get_cur_kv_split_upper() const {
        return (cur.q_token_start == end_q_token_idx) ? end_kv_split_idx : cur.num_kv_splits;
    }

    template <bool kIsCompressedLogits = false>
    CUTLASS_DEVICE bool next_q_block(uint32_t& q_block_idx, uint32_t& kv_split_base, uint32_t& num_kv_splits,
                                     uint32_t* = nullptr, uint32_t* = nullptr) {
        q_block_idx = 0;
        if (done)
            return false;

        const uint32_t upper = get_cur_kv_split_upper();
        cur_block_table_row = cur.get_block_table_row();
        uint32_t q_block_token_offset, q_block_num_tokens;
        cur.get_q_block_span(cur_q_block_in_request, q_block_token_offset, q_block_num_tokens);
        cur_q_block_token_base = cur.q_token_start + q_block_token_offset;
        cur_num_block_tokens = q_block_num_tokens;
        cur_request_num_kv_pages = cur.num_kv_pages;
        kv_split_base = cur_kv_split_base;
        const uint32_t remaining = upper - cur_kv_split_base;
        num_kv_splits = (cur.num_q_blocks == 1) ? remaining
                                               : (remaining < kSplitsPerChunk ? remaining : kSplitsPerChunk);

        ++ cur_q_block_in_request;
        if (cur_q_block_in_request == cur.num_q_blocks) {
            cur_q_block_in_request = 0;
            cur_kv_split_base += num_kv_splits;
            if (cur_kv_split_base >= upper) {
                if (cur.q_token_start == end_q_token_idx) {
                    done = true;
                } else {
                    const uint32_t next_q_token = cur.q_token_start + cur.num_q_tokens;
                    cur_kv_split_base = 0;
                    if (next_q_token >= num_q_tokens_total)
                        done = true;
                    else {
                        cur = Info::from_q_token(next_q_token, num_q_tokens_total, context_lens, nullptr);
                        if (cur.q_token_start == end_q_token_idx and end_kv_split_idx == 0)
                            done = true;
                    }
                }
            }
        }
        return true;
    }

    CUTLASS_DEVICE uint32_t get_num_block_tokens(const uint32_t&) const { return cur_num_block_tokens; }

    CUTLASS_DEVICE uint32_t get_q_tma_token_base(const uint32_t&) const { return cur_q_block_token_base; }

    CUTLASS_DEVICE uint32_t get_kv_page_coord_by_page_offset(const uint32_t& page_offset) const {
        if (page_offset >= cur_request_num_kv_pages)
            return 0;
        const auto block_table_offset = cur_block_table_row * static_cast<uint64_t>(block_table_stride);
        return block_table[block_table_offset + page_offset];
    }

    CUTLASS_DEVICE uint32_t get_logits_row(const uint32_t&, const uint32_t& token_idx) const {
        return cur_q_block_token_base + token_idx;
    }

    CUTLASS_DEVICE uint32_t get_logits_col(const uint32_t& kv_split_base,
                                           const uint32_t& kv_split_idx,
                                           const uint32_t& math_thread_idx) const {
        return (kv_split_base + kv_split_idx) * SPLIT_KV + math_thread_idx;
    }
};

// sm100_mqa_logits_core_impl, paged MXFP4 / bf16 reduce path, with kNumTmemStages a parameter
template <uint32_t kNumHeads, uint32_t kHeadDim,
          uint32_t BLOCK_Q, uint32_t SPLIT_KV,
          uint32_t kNumQStages, uint32_t kNumKVStages, uint32_t kNumTmemStages,
          uint32_t kNumSpecializedThreads, uint32_t kNumMathThreads,
          typename qk_dtype_t, typename logits_dtype_t, typename reduce_dtype_t, typename MakeScheduler,
          uint32_t kNumMathWarpGroups = kNumMathThreads / 128>
CUTLASS_DEVICE void paged_core(const uint32_t logits_stride,
                               logits_dtype_t* logits,
                               const cute::TmaDescriptor& tensor_map_q,
                               const cute::TmaDescriptor& tensor_map_sf_q,
                               const cute::TmaDescriptor& tensor_map_kv,
                               const cute::TmaDescriptor& tensor_map_sf_kv,
                               const cute::TmaDescriptor& tensor_map_weights,
                               const MakeScheduler& make_scheduler) {
    constexpr bool kIsMXSF = true;
    constexpr bool kIsFP4 = cute::is_same_v<qk_dtype_t, cutlass::float_e2m1_t>;
    DG_STATIC_ASSERT(kIsFP4 and not cute::is_same_v<reduce_dtype_t, float>, "paged MXFP4 with bf16 reduce only");

    const auto sm_idx = blockIdx.x;
    using scheduler_t = decltype(make_scheduler(0));
    constexpr bool kPagedWideLoad = use_paged_wide_tmem_load<scheduler_t, kIsMXSF,
        kNumHeads, kHeadDim, logits_dtype_t, reduce_dtype_t>();
    const auto warp_idx = cutlass::canonical_warp_idx_sync();
    const auto lane_idx = ptx::get_lane_idx();
    constexpr uint32_t kSpecWarpStart = kNumMathWarpGroups * 4;

    if (warp_idx == kSpecWarpStart) {
        cute::prefetch_tma_descriptor(&tensor_map_q);
        cute::prefetch_tma_descriptor(&tensor_map_sf_q);
        cute::prefetch_tma_descriptor(&tensor_map_weights);
        cute::prefetch_tma_descriptor(&tensor_map_kv);
        cute::prefetch_tma_descriptor(&tensor_map_sf_kv);
    }

    static constexpr uint32_t kNumUTCCPAlignedElems = 128;
    static constexpr uint32_t UMMA_M = 128;
    static constexpr uint32_t BLOCK_QH = BLOCK_Q * kNumHeads;
    static constexpr uint32_t UMMA_N = math::constexpr_align(BLOCK_QH, 8u);
    static constexpr uint32_t UMMA_K = kIsFP4 ? 64 : 32;
    static constexpr uint32_t kNumSFQ  = math::constexpr_align(UMMA_N, kNumUTCCPAlignedElems);
    static constexpr uint32_t kNumSFKV = math::constexpr_align(SPLIT_KV, kNumUTCCPAlignedElems);
    static constexpr uint32_t kNumQKBytesPerToken = kIsFP4 ? (kHeadDim / 2) : kHeadDim;
    static constexpr uint32_t SMEM_Q_SIZE_PER_STAGE = BLOCK_QH * kNumQKBytesPerToken;
    static constexpr uint32_t SMEM_KV_SIZE_PER_STAGE = SPLIT_KV * kNumQKBytesPerToken;
    static constexpr uint32_t SMEM_SF_Q_SIZE_PER_STAGE = BLOCK_QH * sizeof(int);
    static constexpr uint32_t SMEM_SF_KV_SIZE_PER_STAGE = kNumSFKV * sizeof(int);
    static constexpr uint32_t kNumWeightBytesPerRow = math::constexpr_align(
        kNumHeads * static_cast<uint32_t>(sizeof(reduce_dtype_t)), 16u);
    static constexpr uint32_t kNumWeightElementsPerRow =
        kNumWeightBytesPerRow / static_cast<uint32_t>(sizeof(reduce_dtype_t));
    static constexpr uint32_t SMEM_WEIGHT_SIZE_PER_STAGE = BLOCK_Q * kNumWeightBytesPerRow;

    DG_STATIC_ASSERT(kNumSpecializedThreads == 128 and kNumMathThreads % 128 == 0, "Invalid threads");
    DG_STATIC_ASSERT(SPLIT_KV == kNumMathWarpGroups * UMMA_M and SPLIT_KV % kNumUTCCPAlignedElems == 0, "Invalid `SPLIT_KV`");

    using SharedStorage = layout::MQALogitsSharedStorage<kNumHeads, kHeadDim, kIsMXSF, BLOCK_Q, SPLIT_KV,
                                                         kNumQStages, kNumKVStages, kNumTmemStages, qk_dtype_t, reduce_dtype_t>;
    extern __shared__ __align__(SharedStorage::kSwizzleAlignment) uint8_t smem_buffer[];
    auto& smem = *reinterpret_cast<SharedStorage*>(smem_buffer);

    constexpr uint32_t kNumAccumTmemCols = UMMA_N * kNumTmemStages;
    constexpr uint32_t kNumTmemCols = utils::get_num_aligned_tmem_cols<kNumAccumTmemCols + kNumSFQ / 32 + kNumSFKV / 32>();
    constexpr uint32_t kTmemStartColOfSFQ = kNumAccumTmemCols;
    constexpr uint32_t kTmemStartColOfSFKV = kNumAccumTmemCols + kNumSFQ / 32;
    DG_STATIC_ASSERT(kNumTmemCols <= 512, "Too many tensor memory");

    if (warp_idx == kSpecWarpStart + 1 and cute::elect_one_sync()) {
        #pragma unroll
        for (uint32_t i = 0; i < kNumQStages; ++ i) {
            smem.full_q_barriers[i].init(1);
            smem.empty_q_barriers[i].init(kNumMathThreads + 32);
        }
        #pragma unroll
        for (uint32_t i = 0; i < kNumKVStages; ++ i) {
            smem.full_kv_barriers[i].init(1);
            smem.empty_kv_barriers[i].init(1);
        }
        #pragma unroll
        for (uint32_t i = 0; i < kNumTmemStages; ++i) {
            smem.full_tmem_barriers[i].init(1);
            smem.empty_tmem_barriers[i].init(128);
        }
        cutlass::arch::fence_barrier_init();
    }
    __syncwarp();

    if (warp_idx == kSpecWarpStart + 2)
        cute::TMEM::Allocator1Sm().allocate(kNumTmemCols, &smem.tmem_ptr_in_smem);
    __syncthreads();

    RingPipeline<kNumQStages> q_pipeline;
    RingPipeline<kNumKVStages> kv_pipeline;
    RingPipeline<kNumTmemStages> tmem_pipeline;

    constexpr uint32_t kNumSpecializedRegisters = 56;
    constexpr uint32_t kNumEntryRegisters = (512 / (kNumMathWarpGroups + 1) / 8) * 8;
    constexpr uint32_t kNumMathRegisters = kNumMathWarpGroups == 2 ? 224 :
        ((kNumEntryRegisters * (kNumMathWarpGroups + 1) - kNumSpecializedRegisters) / kNumMathWarpGroups / 8) * 8;
    DG_STATIC_ASSERT(kNumMathRegisters * kNumMathWarpGroups + kNumSpecializedRegisters <= 512,
                     "Register reconfiguration exceeds the CTA pool");

    cudaGridDependencySynchronize();

    if (warp_idx == kSpecWarpStart) {
        cutlass::arch::warpgroup_reg_dealloc<kNumSpecializedRegisters>();
        if (cute::elect_one_sync()) {
            auto scheduler = make_scheduler(sm_idx);
            uint32_t q_block_idx, kv_base, num_kv_splits;
            while (scheduler.next_q_block(q_block_idx, kv_base, num_kv_splits)) {
                CUTE_TIE_DECL(q_pipeline.advance(), q_stage_idx, q_phase);
                smem.empty_q_barriers[q_stage_idx].wait(q_phase ^ 1);

                const uint32_t q_token_base = scheduler.get_q_tma_token_base(q_block_idx);
                tma::copy<kHeadDim, BLOCK_Q * kNumHeads, 0>(
                    &tensor_map_q, &smem.full_q_barriers[q_stage_idx],
                    smem.smem_q[q_stage_idx], 0, q_token_base * kNumHeads);
                tma::copy<BLOCK_Q * kNumHeads, 1, 0>(&tensor_map_sf_q, &smem.full_q_barriers[q_stage_idx], smem.smem_sf_q[q_stage_idx], 0, q_token_base);
                tma::copy<kNumWeightElementsPerRow, BLOCK_Q, 0>(&tensor_map_weights, &smem.full_q_barriers[q_stage_idx], smem.smem_weights[q_stage_idx], 0, q_token_base);
                smem.full_q_barriers[q_stage_idx].arrive_and_expect_tx(SMEM_Q_SIZE_PER_STAGE + SMEM_SF_Q_SIZE_PER_STAGE + SMEM_WEIGHT_SIZE_PER_STAGE);
            }
        }
        __syncwarp();
    } else if (warp_idx == kSpecWarpStart + 1) {
        cutlass::arch::warpgroup_reg_dealloc<kNumSpecializedRegisters>();

        auto scheduler = make_scheduler(sm_idx);
        uint32_t cached_kv_page_base = 0;
        uint32_t cached_kv_page_coord = 0;
        uint32_t q_block_idx, kv_base, num_kv_splits;
        while (scheduler.next_q_block(q_block_idx, kv_base, num_kv_splits)) {
            cached_kv_page_base = cute::numeric_limits<uint32_t>::max();
            #pragma unroll 1
            for (uint32_t kv_split_idx = 0; kv_split_idx < num_kv_splits; ++ kv_split_idx) {
                constexpr uint32_t kPageKV = scheduler_t::kPageKV;
                constexpr uint32_t kNumPagesPerSplit = scheduler_t::kNumPagesPerSplit;
                DG_STATIC_ASSERT(kNumPagesPerSplit <= 32, "Split spans more pages than a warp can cache");

                const uint32_t kv_page_base = (kv_base + kv_split_idx) * kNumPagesPerSplit;
                if (kv_page_base < cached_kv_page_base or kv_page_base + kNumPagesPerSplit > cached_kv_page_base + 32) {
                    cached_kv_page_base = (kv_page_base / 32) * 32;
                    cached_kv_page_coord = scheduler.get_kv_page_coord_by_page_offset(cached_kv_page_base + lane_idx);
                }

                CUTE_TIE_DECL(kv_pipeline.advance(), kv_stage_idx, kv_phase);
                if (cute::elect_one_sync())
                    smem.empty_kv_barriers[kv_stage_idx].wait(kv_phase ^ 1);
                __syncwarp();

                int page_coords[kNumPagesPerSplit];
                #pragma unroll
                for (uint32_t page_idx = 0; page_idx < kNumPagesPerSplit; ++ page_idx) {
                    const auto src_lane = static_cast<int>(kv_page_base - cached_kv_page_base + page_idx);
                    page_coords[page_idx] = __shfl_sync(0xffffffff, cached_kv_page_coord, src_lane);
                }

                if (cute::elect_one_sync()) {
                    #pragma unroll
                    for (uint32_t page_idx = 0; page_idx < kNumPagesPerSplit; ++ page_idx) {
                        tma::copy<kHeadDim, kPageKV, 0, qk_dtype_t, true>(
                            &tensor_map_kv, &smem.full_kv_barriers[kv_stage_idx],
                            smem.smem_kv[kv_stage_idx] + page_idx * kPageKV * kNumQKBytesPerToken,
                            0, 0, 1, page_coords[page_idx]);
                        tma::copy<kPageKV, 1, 0>(&tensor_map_sf_kv, &smem.full_kv_barriers[kv_stage_idx],
                                                 smem.smem_sf_kv[kv_stage_idx] + page_idx * kPageKV,
                                                 0, page_coords[page_idx]);
                    }
                    smem.full_kv_barriers[kv_stage_idx].arrive_and_expect_tx(SMEM_KV_SIZE_PER_STAGE + SMEM_SF_KV_SIZE_PER_STAGE);
                }
                __syncwarp();
            }
        }
    } else if (warp_idx == kSpecWarpStart + 2) {
        cutlass::arch::warpgroup_reg_dealloc<kNumSpecializedRegisters>();
        DG_TRAP_ONLY_DEVICE_ASSERT(ptx::ld_shared(&smem.tmem_ptr_in_smem) == 0);

        auto utccp_required_smem_warp_transpose = [&](const uint32_t* smem_ptr) {
            DG_STATIC_ASSERT(kNumUTCCPAlignedElems == 128, "Invalid aligned elements");
            uint32_t values[4];
            #pragma unroll
            for (uint32_t i = 0; i < 4; ++ i)
                values[i] = ptx::ld_shared(smem_ptr + i * 32 + lane_idx);
            __syncwarp();
            ptx::st_shared(smem_ptr + lane_idx * 4, values[0], values[1], values[2], values[3]);
        };

        auto sf_desc = mma::sm100::make_sf_desc(nullptr);

        auto scheduler = make_scheduler(sm_idx);
        uint32_t q_block_idx, kv_base, num_kv_splits;
        while (scheduler.next_q_block(q_block_idx, kv_base, num_kv_splits)) {
            CUTE_TIE_DECL(q_pipeline.advance(), q_stage_idx, q_phase);
            smem.full_q_barriers[q_stage_idx].wait(q_phase);

            #pragma unroll
            for (uint32_t i = 0; i < kNumSFQ / kNumUTCCPAlignedElems; ++ i) {
                auto smem_ptr = smem.smem_sf_q[q_stage_idx] + i * kNumUTCCPAlignedElems;
                utccp_required_smem_warp_transpose(smem_ptr);
            }
            cutlass::arch::fence_view_async_shared();
            #pragma unroll
            for (uint32_t i = 0; i < kNumSFQ / kNumUTCCPAlignedElems; ++ i) {
                auto smem_ptr = smem.smem_sf_q[q_stage_idx] + i * kNumUTCCPAlignedElems;
                mma::sm100::replace_smem_desc_addr(sf_desc, smem_ptr);
                if (cute::elect_one_sync())
                    cute::SM100_UTCCP_4x32dp128bit_1cta::copy(sf_desc, kTmemStartColOfSFQ + i * 4);
                __syncwarp();
            }

            for (uint32_t kv_split_idx = 0; kv_split_idx < num_kv_splits; ++ kv_split_idx) {
                CUTE_TIE_DECL(kv_pipeline.advance(), kv_stage_idx, kv_phase);
                smem.full_kv_barriers[kv_stage_idx].wait(kv_phase);

                #pragma unroll
                for (uint32_t i = 0; i < kNumSFKV / kNumUTCCPAlignedElems; ++ i) {
                    auto smem_ptr = smem.smem_sf_kv[kv_stage_idx] + i * kNumUTCCPAlignedElems;
                    utccp_required_smem_warp_transpose(smem_ptr);
                }
                cutlass::arch::fence_view_async_shared();

                if (cute::elect_one_sync()) {
                    #pragma unroll
                    for (uint32_t i = 0; i < kNumSFKV / kNumUTCCPAlignedElems; ++ i) {
                        auto smem_ptr = smem.smem_sf_kv[kv_stage_idx] + i * kNumUTCCPAlignedElems;
                        mma::sm100::replace_smem_desc_addr(sf_desc, smem_ptr);
                        cute::SM100_UTCCP_4x32dp128bit_1cta::copy(sf_desc, kTmemStartColOfSFKV + i * 4);
                    }
                    #pragma unroll
                    for (uint32_t i = 0; i < kNumMathWarpGroups; ++ i) {
                        CUTE_TIE_DECL(tmem_pipeline.advance(), tmem_stage_idx, tmem_phase);
                        uint32_t tmem_addr = tmem_stage_idx * UMMA_N;

                        smem.empty_tmem_barriers[tmem_stage_idx].wait(tmem_phase ^ 1);
                        ptx::tcgen05_after_thread_sync();

                        constexpr uint32_t kPackFactor = get_smem_pack_factor<qk_dtype_t>();
                        constexpr uint32_t kQKSwizzleMode = kHeadDim / kPackFactor;

                        using mma_op_t = ptx::SM100_MMA_MXF4_SS;
                        auto instr_desc = cute::UMMA::make_instr_desc_block_scaled<qk_dtype_t, qk_dtype_t, float, cutlass::float_ue8m0_t,
                                                                                   UMMA_M, UMMA_N, cute::UMMA::Major::K, cute::UMMA::Major::K>();
                        #pragma unroll
                        for (uint32_t k = 0; k < kHeadDim / UMMA_K; ++ k) {
                            auto runtime_instr_desc = mma::sm100::make_runtime_instr_desc_with_sf_id(instr_desc, k * kPackFactor, k * kPackFactor);
                            auto a_desc = mma::sm100::make_umma_desc<cute::UMMA::Major::K, 0, kHeadDim, kQKSwizzleMode>(
                                smem.smem_kv[kv_stage_idx], i * UMMA_M, k * UMMA_K);
                            auto b_desc = mma::sm100::make_umma_desc<cute::UMMA::Major::K, 0, kHeadDim, kQKSwizzleMode>(
                                smem.smem_q[q_stage_idx], 0, k * UMMA_K);
                            mma_op_t::fma(
                                a_desc, b_desc, tmem_addr, k, runtime_instr_desc,
                                kTmemStartColOfSFKV + i * 4, kTmemStartColOfSFQ);
                        }

                        ptx::umma_arrive_no_elect(smem.full_tmem_barriers[tmem_stage_idx]);
                    }
                }
                __syncwarp();
                cutlass::arch::umma_arrive(reinterpret_cast<uint64_t*>(&smem.empty_kv_barriers[kv_stage_idx]));
            }
            smem.empty_q_barriers[q_stage_idx].arrive();
        }
    } else if (warp_idx == kSpecWarpStart + 3) {
        cutlass::arch::warpgroup_reg_dealloc<kNumSpecializedRegisters>();
    } else if (warp_idx < kSpecWarpStart) {
        cutlass::arch::warpgroup_reg_alloc<kNumMathRegisters>();

        const auto local_math_warp_idx = get_mqa_math_warp_idx<kPagedWideLoad>(warp_idx);
        const auto math_warpgroup_idx = local_math_warp_idx / 4;
        const auto math_thread_idx = local_math_warp_idx * 32 + lane_idx;
        DG_STATIC_ASSERT(kNumMathWarpGroups <= kNumTmemStages, "Math warp groups exceed TMEM stages");
        tmem_pipeline.advance(math_warpgroup_idx);

        DG_STATIC_ASSERT(kNumHeads % 4 == 0, "Head count must be a multiple of 4");
        DG_STATIC_ASSERT(8 <= UMMA_N and UMMA_N <= 256, "Invalid UMMA_N for MMA");
        using weights_reg_dtype_t = nv_bfloat162;
        constexpr uint32_t kNumWeightsRegPerToken = kNumHeads / 2;
        constexpr bool kEarlyRelease = Q6_EARLY_RELEASE and kNumTmemStages < 3 and BLOCK_Q <= 6 and kNumHeads == 32 and
            not kPagedWideLoad;
        weights_reg_dtype_t weights[kEarlyRelease ? 1 : BLOCK_Q][kNumWeightsRegPerToken];
        float accum[kNumHeads];

        auto scheduler = make_scheduler(sm_idx);
        uint32_t q_block_idx, kv_base, num_kv_splits;
        uint32_t seq_k_start[1], seq_k_end[1];
        while (scheduler.template next_q_block<false>(
                q_block_idx, kv_base, num_kv_splits, seq_k_start, seq_k_end)) {
            CUTE_TIE_DECL(q_pipeline.advance(), q_stage_idx, q_phase);
            smem.full_q_barriers[q_stage_idx].wait(q_phase);

            const auto process_q_block = [&](auto num_valid_tokens_t) {
                constexpr uint32_t kNumValidTokens = decltype(num_valid_tokens_t)::value;

                #pragma unroll
                for (uint32_t i = 0; i < (kEarlyRelease ? 0 : kNumValidTokens); ++ i) {
                    const auto smem_weights_row = smem.smem_weights[q_stage_idx] + i * kNumWeightElementsPerRow;
                    const auto packed_row = reinterpret_cast<const uint32_t*>(smem_weights_row);
                    #pragma unroll
                    for (uint32_t j = 0; j < kNumHeads / 2; ++ j) {
                        const auto packed = ptx::ld_shared(packed_row + j);
                        weights[i][j] = ptx::exchange(*reinterpret_cast<const nv_bfloat162*>(&packed), 0);
                    }
                }

                for (uint32_t kv_split_idx = 0; kv_split_idx < num_kv_splits; ++ kv_split_idx) {
                    auto kv_offset = scheduler.get_logits_col(kv_base, kv_split_idx, math_thread_idx);

                    CUTE_TIE_DECL(tmem_pipeline.advance(kNumMathWarpGroups), tmem_stage_idx, tmem_phase);
                    smem.full_tmem_barriers[tmem_stage_idx].wait(tmem_phase);
                    ptx::tcgen05_after_thread_sync();

                    // Two TMEM stages leave the MMA no slack: take every row's ReLU'd bf16 pairs
                    // out of TMEM, release the stage, then reduce (the same operations in the same
                    // order, the weights read from shared memory to keep the registers)
                    if constexpr (kEarlyRelease) {
                        nv_bfloat162 act[kNumValidTokens][kNumHeads / 2];
                        #pragma unroll
                        for (uint32_t i = 0; i < kNumValidTokens; ++ i) {
                            uint32_t tmem_addr = tmem_stage_idx * UMMA_N + i * kNumHeads;
                            ptx::tmem_load_32dp32b<kNumHeads / 2>(tmem_addr, reinterpret_cast<uint32_t*>(accum));
                            cutlass::arch::fence_view_async_tmem_load();
                            ptx::tmem_load_32dp32b<kNumHeads / 2>(tmem_addr + kNumHeads / 2,
                                                                  reinterpret_cast<uint32_t*>(accum + kNumHeads / 2));
                            cutlass::arch::fence_view_async_tmem_load();
                            #pragma unroll
                            for (uint32_t j = 0; j < kNumHeads; j += 2)
                                act[i][j / 2] = ptx::cvt_relu_bf16x2_f32(make_float2(accum[j], accum[j + 1]));
                        }
                        ptx::tcgen05_before_thread_sync();
                        smem.empty_tmem_barriers[tmem_stage_idx].arrive();

                        #pragma unroll
                        for (uint32_t i = 0; i < kNumValidTokens; ++ i) {
                            const auto packed_row = reinterpret_cast<const uint32_t*>(
                                smem.smem_weights[q_stage_idx] + i * kNumWeightElementsPerRow);
                            const auto transform = [&](const uint32_t& j, const nv_bfloat162& sum) {
                                const auto packed = ptx::ld_shared(packed_row + j / 2);
                                const auto b = *reinterpret_cast<const nv_bfloat162*>(&packed);
                                return __hfma2(act[i][j / 2], b, sum);
                            };
                            auto sum_0 = __floats2bfloat162_rn(0.0f, 0.0f);
                            auto sum_1 = __floats2bfloat162_rn(0.0f, 0.0f);
                            #pragma unroll
                            for (uint32_t j = 0; j < kNumHeads; j += 4) {
                                sum_0 = transform(j, sum_0);
                                sum_1 = transform(j + 2, sum_1);
                            }
                            auto sum = __hadd2_rn(sum_0, sum_1);
                            const reduce_dtype_t reduced = __hadd_rn(sum.x, sum.y);
                            const auto q_offset = scheduler.get_logits_row(q_block_idx, i) * static_cast<uint64_t>(logits_stride);
                            logits[q_offset + kv_offset] = static_cast<logits_dtype_t>(reduced);
                        }
                    } else {
                    #pragma unroll
                    for (uint32_t i = 0; i < kNumValidTokens; ++ i) {
                        uint32_t tmem_addr = tmem_stage_idx * UMMA_N + i * kNumHeads;
                        if constexpr (kNumHeads == 8) {
                            ptx::tmem_load_32dp32b<kNumHeads>(tmem_addr, reinterpret_cast<uint32_t*>(accum));
                            cutlass::arch::fence_view_async_tmem_load();
                        } else if constexpr (kNumHeads == 16) {
                            ptx::tmem_load_32dp32b<kNumHeads / 2>(tmem_addr, reinterpret_cast<uint32_t*>(accum));
                            ptx::tmem_load_32dp32b<kNumHeads / 2>(tmem_addr + kNumHeads / 2,
                                                                  reinterpret_cast<uint32_t*>(accum + kNumHeads / 2));
                            cutlass::arch::fence_view_async_tmem_load();
                        } else if constexpr (kPagedWideLoad) {
                            ptx::tmem_load_32dp32b<64>(tmem_addr, reinterpret_cast<uint32_t*>(accum));
                            cutlass::arch::fence_view_async_tmem_load();
                        } else if constexpr (kNumHeads == 32 or kNumHeads == 64) {
                            ptx::tmem_load_32dp32b<kNumHeads / 2>(tmem_addr, reinterpret_cast<uint32_t*>(accum));
                            cutlass::arch::fence_view_async_tmem_load();
                            ptx::tmem_load_32dp32b<kNumHeads / 2>(tmem_addr + kNumHeads / 2,
                                                                  reinterpret_cast<uint32_t*>(accum + kNumHeads / 2));
                            cutlass::arch::fence_view_async_tmem_load();
                        } else {
                            load_tmem_heads_decomposed<kNumHeads>(tmem_addr, accum);
                            cutlass::arch::fence_view_async_tmem_load();
                        }

                        if (i == kNumValidTokens - 1) {
                            ptx::tcgen05_before_thread_sync();
                            smem.empty_tmem_barriers[tmem_stage_idx].arrive();
                        }

                        reduce_dtype_t reduced;
                        auto sum_0 = __floats2bfloat162_rn(0.0f, 0.0f);
                        auto sum_1 = __floats2bfloat162_rn(0.0f, 0.0f);
                        const auto transform = [&](const uint32_t& j, const nv_bfloat162& sum) {
                            const auto a = ptx::cvt_relu_bf16x2_f32(make_float2(accum[j], accum[j + 1]));
                            const auto b = weights[i][j / 2];
                            return __hfma2(a, b, sum);
                        };

                        #pragma unroll
                        for (uint32_t j = 0; j < kNumHeads; j += 4) {
                            sum_0 = transform(j, sum_0);
                            sum_1 = transform(j + 2, sum_1);
                        }

                        auto sum = __hadd2_rn(sum_0, sum_1);
                        reduced = __hadd_rn(sum.x, sum.y);
                        auto result = static_cast<logits_dtype_t>(reduced);
                        const auto q_offset = scheduler.get_logits_row(q_block_idx, i) * static_cast<uint64_t>(logits_stride);
                        logits[q_offset + kv_offset] = result;
                    }
                    }
                }
            };

            dispatch_num_block_tokens<BLOCK_Q>(scheduler.get_num_block_tokens(q_block_idx), process_q_block);

            cutlass::arch::fence_view_async_shared();
            smem.empty_q_barriers[q_stage_idx].arrive();
        }

        cutlass::arch::NamedBarrier(kNumMathThreads, 0).sync();
        if (get_mqa_math_warp_idx<kPagedWideLoad>(warp_idx) == 0)
            cute::TMEM::Allocator1Sm().free(0, kNumTmemCols);
    }
}

constexpr uint32_t kRows = Q6_ROWS;
constexpr uint32_t kHeads = 32;
constexpr uint32_t kHeadDim = 128;
constexpr uint32_t kSplitKV = 256;
constexpr uint32_t kQStages = 3;
constexpr uint32_t kKVStages = Q6_KV_STAGES;
constexpr uint32_t kSpecThreads = 128;
constexpr uint32_t kMathThreads = 256;
constexpr uint32_t kUmmaN = math::constexpr_align(kRows * kHeads, 8u);
constexpr uint32_t kTmemStages =
    kUmmaN * 3 + math::constexpr_align(kUmmaN, 128u) / 32 + kSplitKV / 32 <= 512 ? 3 : 2;
DG_STATIC_ASSERT(kRows >= 1 and kRows * kHeads <= 256, "one Q block of at most 256 row-heads");

using Storage = layout::MQALogitsSharedStorage<kHeads, kHeadDim, true, kRows, kSplitKV,
                                               kQStages, kKVStages, kTmemStages,
                                               cutlass::float_e2m1_t, __nv_bfloat16>;

}  // namespace deep_gemm::legacy::dsv41

extern "C" __device__ const uint32_t dsv41_paged_logits_q6_smem = sizeof(deep_gemm::legacy::dsv41::Storage);

extern "C" __global__ __launch_bounds__(384, 1)
void dsv41_paged_logits_q6(const uint32_t num_q_tokens_total,
                           const uint32_t logits_stride, const uint32_t block_table_stride,
                           const uint32_t* context_lens, cutlass::bfloat16_t* logits,
                           const uint32_t* block_table, const uint32_t* schedule_meta,
                           const __grid_constant__ cute::TmaDescriptor tensor_map_q,
                           const __grid_constant__ cute::TmaDescriptor tensor_map_sf_q,
                           const __grid_constant__ cute::TmaDescriptor tensor_map_kv,
                           const __grid_constant__ cute::TmaDescriptor tensor_map_sf_kv,
                           const __grid_constant__ cute::TmaDescriptor tensor_map_weights) {
    using namespace deep_gemm::legacy::dsv41;
    const auto make_scheduler = [&](const uint32_t& sm_idx) {
        return PagedScheduler<kRows, kRows, kSplitKV, Q6_PAGE_KV, 16>(
            sm_idx, context_lens, schedule_meta, block_table, block_table_stride, num_q_tokens_total);
    };
    paged_core<kHeads, kHeadDim, kRows, kSplitKV, kQStages, kKVStages, kTmemStages,
               kSpecThreads, kMathThreads, cutlass::float_e2m1_t, cutlass::bfloat16_t, __nv_bfloat16,
               decltype(make_scheduler)>(
        logits_stride, logits,
        tensor_map_q, tensor_map_sf_q, tensor_map_kv, tensor_map_sf_kv, tensor_map_weights,
        make_scheduler);
}
