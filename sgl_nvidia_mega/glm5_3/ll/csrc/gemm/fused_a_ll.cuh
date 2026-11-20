// SPDX-License-Identifier: Apache-2.0
/*
 * Adapted from
 * https://github.com/NVIDIA/TensorRT-LLM/blob/619709fc33bd5dc268f19d6a741fe7ed51c0f8f5/cpp/tensorrt_llm/kernels/dsv3MinLatencyKernels/dsv3FusedAGemm.cu
 *
 * Copyright (c) 2019-2024, NVIDIA CORPORATION.  All rights reserved.
 * Copyright (c) 2021, NAVER Corp.  Authored by CLOVA.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// Portions Copyright 2023-2024 SGLang Team (Apache-2.0)
// Adapted from SGLang (Apache-2.0) python/sglang/kernels/jit/csrc/gemm/dsv3_fused_a_gemm.cuh, whose
// header is kept unchanged above.

// The GLM-5 low-latency fused-A GEMM forms, for the MLA fused q_a / kv_a projection and
// q_b of a TP4 rank (bf16, out = x @ W^T):
// - DSV3FusedAGemmLLKernel (1-16 rows): SGLang's fused_a_gemm_kernel with a pipelined
//   MMA main loop (kIlp) and a 224 KB stage ring (kDeep). Each option gives the same
//   bits as SGLang's dsv3_fused_a_gemm kernel: every accumulator sees the same HMMA
//   sequence and the epilogue is unchanged.
// - DSV3FusedAGemmWideKernel (17-48 rows): a split-K kernel, a cluster of kSplitK CTAs
//   per tile_m output rows summed through DSMEM in split order. Deterministic, not
//   bit-identical to the 1-16-row kernel or to the layer's own GEMM.
// SGLang's own fused_a_gemm_kernel and DSV3FusedAGemmKernel are not repeated here;
// the loaders and the MMA computer are SGLang's, unchanged.

#include <sgl_kernel/tensor.h>
#include <sgl_kernel/utils.h>

#include <sgl_kernel/runtime.cuh>
#include <sgl_kernel/utils.cuh>

#include <tvm/ffi/container/tensor.h>

#include <algorithm>
#include <cooperative_groups.h>
#include <cuda_bf16.h>
#include <utility>

// Inside sglang:: for the sgl_kernel helpers (host::, device::, bf16_t), as spec_step's sources.
namespace sglang::glm_ll {

using bf16_t = __nv_bfloat16;

__device__ void hmma_16_8_16_f32acc_bf16ab(
    float (&d_reg)[4], const bf16_t (&a_reg)[8], const bf16_t (&b_reg)[4], float const (&c_reg)[4]) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
  uint32_t a0 = *reinterpret_cast<uint32_t const*>(a_reg + 0);
  uint32_t a1 = *reinterpret_cast<uint32_t const*>(a_reg + 2);
  uint32_t a2 = *reinterpret_cast<uint32_t const*>(a_reg + 4);
  uint32_t a3 = *reinterpret_cast<uint32_t const*>(a_reg + 6);
  uint32_t b0 = *reinterpret_cast<uint32_t const*>(b_reg + 0);
  uint32_t b1 = *reinterpret_cast<uint32_t const*>(b_reg + 2);
  asm volatile(
      "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
      "{%0,  %1,  %2,  %3},"
      "{%4,  %5,  %6,  %7},"
      "{%8,  %9},"
      "{%10, %11, %12, %13};\n"
      : "=f"(d_reg[0]), "=f"(d_reg[1]), "=f"(d_reg[2]), "=f"(d_reg[3])
      : "r"(a0),
        "r"(a1),
        "r"(a2),
        "r"(a3),
        "r"(b0),
        "r"(b1),
        "f"(d_reg[0]),
        "f"(d_reg[1]),
        "f"(d_reg[2]),
        "f"(d_reg[3]));
#endif
}

extern "C" {
__device__ uint32_t __nvvm_get_smem_pointer(void*);
}

__device__ void ldgsts_128(void const* gPtr, void* sPtr, uint32_t pred) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
  if (pred) {
    uint32_t smemPtrAsUint32 = __nvvm_get_smem_pointer(sPtr);
    asm volatile("cp.async.cg.shared.global.L2::128B [%0], [%1], %2;\n" ::"r"(smemPtrAsUint32), "l"(gPtr), "n"(16));
  }
#endif
}

__device__ void ldsm_x4(void* smem_ptr, uint32_t* reg_ptr) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
  asm volatile("ldmatrix.sync.aligned.x4.m8n8.shared.b16 {%0, %1, %2, %3}, [%4];\n"
               : "=r"(reg_ptr[0]), "=r"(reg_ptr[1]), "=r"(reg_ptr[2]), "=r"(reg_ptr[3])
               : "r"(__nvvm_get_smem_pointer(smem_ptr)));
#endif
}

template <class Type>
__device__ int apply_swizzle_343_on_elem_row_col(int row_idx_, int col_idx_) {
  uint32_t row_idx = *reinterpret_cast<uint32_t*>(&row_idx_);
  uint32_t col_idx = *reinterpret_cast<uint32_t*>(&col_idx_);
  row_idx = row_idx % 8;
  row_idx = row_idx * (16 / sizeof(Type));
  col_idx = col_idx ^ row_idx;
  return *reinterpret_cast<int*>(&col_idx);
}

__device__ void initialize_barrier(uint64_t* smem_barrier, int thread_count = 1) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
  uint32_t smem_int_ptr = __nvvm_get_smem_pointer(smem_barrier);
  asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;\n" ::"r"(smem_int_ptr), "r"(thread_count));
#endif
}

__device__ void wait_barrier(uint64_t* smem_barrier, int phase_bit) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
  uint32_t smem_int_ptr = __nvvm_get_smem_pointer(smem_barrier);
  asm volatile(
      "{\n"
      ".reg .pred                P1;\n"
      "LAB_WAIT:\n"
      "mbarrier.try_wait.parity.shared::cta.b64 P1, [%0], %1;\n"
      "@P1                       bra DONE;\n"
      "bra                   LAB_WAIT;\n"
      "DONE:\n"
      "}\n" ::"r"(smem_int_ptr),
      "r"(phase_bit));
#endif
}

__device__ bool try_wait_barrier(uint64_t* smem_ptr, int phase_bit) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
  uint32_t wait_complete;
  uint32_t smem_int_ptr = __nvvm_get_smem_pointer(smem_ptr);
  asm volatile(
      "{\n\t"
      ".reg .pred P1; \n\t"
      "mbarrier.try_wait.parity.shared::cta.b64 P1, [%1], %2; \n\t"
      "selp.b32 %0, 1, 0, P1; \n\t"
      "}"
      : "=r"(wait_complete)
      : "r"(smem_int_ptr), "r"(phase_bit));
  return static_cast<bool>(wait_complete);
#endif
  return false;
}

// Never suspends, unlike try_wait_barrier.
__device__ bool test_wait_barrier(uint64_t* smem_ptr, int phase_bit) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
  uint32_t wait_complete;
  uint32_t smem_int_ptr = __nvvm_get_smem_pointer(smem_ptr);
  asm volatile(
      "{\n\t"
      ".reg .pred P1; \n\t"
      "mbarrier.test_wait.parity.shared::cta.b64 P1, [%1], %2; \n\t"
      "selp.b32 %0, 1, 0, P1; \n\t"
      "}"
      : "=r"(wait_complete)
      : "r"(smem_int_ptr), "r"(phase_bit));
  return static_cast<bool>(wait_complete);
#endif
  return false;
}

__device__ void arrive_barrier(uint64_t* smem_barrier) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
  uint32_t smem_int_ptr = __nvvm_get_smem_pointer(smem_barrier);
  asm volatile(
      "{\n"
      ".reg .b64 state; \n"
      "mbarrier.arrive.shared::cta.b64   state, [%0];\n"
      "}\n" ::"r"(smem_int_ptr));
#endif
}

__device__ void ldgsts_arrive(uint64_t* smem_barrier) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
  uint32_t smem_int_ptr = __nvvm_get_smem_pointer(smem_barrier);
  asm volatile("cp.async.mbarrier.arrive.noinc.shared.b64 [%0];" : : "r"(smem_int_ptr));
#endif
}

template <int gemm_k, int tile_m, int tile_k, int stage_cnt>
struct GmemLoaderA {
  static constexpr int elem_bytes = 2;
  static constexpr int vec_bytes = 16;
  static constexpr int vec_elems = vec_bytes / elem_bytes;
  static constexpr int thread_cnt = 64;
  static_assert((tile_m * tile_k) % (vec_elems * thread_cnt) == 0);
  static constexpr int a_inst_cnt_per_iter = (tile_m * tile_k) / (vec_elems * thread_cnt);
  static_assert(gemm_k % tile_k == 0);
  static constexpr int k_iter_cnt = gemm_k / tile_k;

  static constexpr int mma_warp_cnt = 4;
  static constexpr int per_mma_warp_k = tile_k / mma_warp_cnt;
  static constexpr int k_each_chunk = gemm_k / mma_warp_cnt;

 private:
  __device__ int k_project(int tile_k_idx) {
    return (tile_k_idx / per_mma_warp_k * k_each_chunk) + (tile_k_idx % per_mma_warp_k);
  }

 public:
  __device__ GmemLoaderA(bf16_t const* gmem_a_local_, bf16_t* smem_a_, uint64_t* smem_barrier_)
      : gmem_a(gmem_a_local_), smem_a(smem_a_), smem_barrier(smem_barrier_), local_tid(threadIdx.x % thread_cnt) {}

  __device__ void prepare() {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
#pragma unroll
    for (int i = 0; i < a_inst_cnt_per_iter; i++) {
      int linear_idx = local_tid * vec_elems + i * thread_cnt * vec_elems;
      int m_idx = linear_idx / tile_k;
      int k_idx = linear_idx % tile_k;
      k_idx = apply_swizzle_343_on_elem_row_col<bf16_t>(m_idx, k_idx);
      a_smem_offsets[i] = m_idx * tile_k + k_idx;
    }
#endif
  }

  __device__ void issue_mainloop() {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
#pragma unroll 1
    for (int loop_idx = 0; loop_idx < k_iter_cnt; loop_idx++) {
      if (need_wait) {
        wait_barrier(smem_barrier + 1 + stage_idx * 2, phase_bit);
      }
      int next_stage_idx = stage_idx + 1;
      int next_phase_bit = next_stage_idx == stage_cnt ? phase_bit ^ 1 : phase_bit;
      next_stage_idx = next_stage_idx == stage_cnt ? 0 : next_stage_idx;
      if (loop_idx != k_iter_cnt - 1) {
        need_wait = !try_wait_barrier(smem_barrier + 1 + next_stage_idx * 2, next_phase_bit);
      }

#pragma unroll
      for (int i = 0; i < a_inst_cnt_per_iter; i++) {
        int smem_offset = a_smem_offsets[i];
        bf16_t* smem_ptr_this_iter = smem_a + stage_idx * tile_m * tile_k + smem_offset;
        int linear_idx = local_tid * vec_elems + i * thread_cnt * vec_elems;
        int m_idx = linear_idx / tile_k;
        int k_idx = linear_idx % tile_k;
        int gmem_offset = m_idx * gemm_k + k_project(k_idx);
        bf16_t const* gmem_ptr_this_iter = gmem_a + gmem_offset;
        ldgsts_128(gmem_ptr_this_iter, smem_ptr_this_iter, true);
      }
      ldgsts_arrive(smem_barrier + stage_idx * 2);

      stage_idx = next_stage_idx;
      phase_bit = next_phase_bit;
      gmem_a += per_mma_warp_k;
    }
#endif
  }

  bf16_t const* gmem_a;
  bf16_t* smem_a;
  uint64_t* smem_barrier;
  int local_tid;
  int stage_idx = 0;
  int phase_bit = 1;
  bool need_wait = true;

  int a_smem_offsets[a_inst_cnt_per_iter];
};

template <int gemm_k, int tile_n, int tile_k, int stage_cnt>
struct GmemLoaderB {
  static constexpr int elem_bytes = 2;
  static constexpr int vec_bytes = 16;
  static constexpr int vec_elems = vec_bytes / elem_bytes;
  static constexpr int thread_cnt = 64;
  static_assert((tile_n * tile_k) % (vec_elems * thread_cnt) == 0);
  static constexpr int b_inst_cnt_per_iter = (tile_n * tile_k) / (vec_elems * thread_cnt);
  static_assert(gemm_k % tile_k == 0);
  static constexpr int k_iter_cnt = gemm_k / tile_k;

  static constexpr int mma_warp_cnt = 4;
  static constexpr int per_mma_warp_k = tile_k / mma_warp_cnt;
  static constexpr int k_each_chunk = gemm_k / mma_warp_cnt;

 private:
  __device__ int k_project(int tile_k_idx) {
    return (tile_k_idx / per_mma_warp_k * k_each_chunk) + (tile_k_idx % per_mma_warp_k);
  }

 public:
  __device__ GmemLoaderB(bf16_t const* gmem_b_local_, bf16_t* smem_b_, uint64_t* smem_barrier_, int gemm_n_)
      : gmem_b(gmem_b_local_),
        smem_b(smem_b_),
        smem_barrier(smem_barrier_),
        gemm_n(gemm_n_),
        local_tid(threadIdx.x % thread_cnt) {}

  __device__ void prepare() {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
#pragma unroll
    for (int i = 0; i < b_inst_cnt_per_iter; i++) {
      int linear_idx = local_tid * vec_elems + i * thread_cnt * vec_elems;
      int n_idx = linear_idx / tile_k;
      int k_idx = linear_idx % tile_k;
      k_idx = apply_swizzle_343_on_elem_row_col<bf16_t>(n_idx, k_idx);
      b_smem_offsets[i] = n_idx * tile_k + k_idx;
      preds[i] = n_idx < gemm_n;
    }
#endif
  }

  __device__ void issue_mainloop() {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
    cudaGridDependencySynchronize();
#pragma unroll 1
    for (int loop_idx = 0; loop_idx < k_iter_cnt; loop_idx++) {
      if (need_wait) {
        wait_barrier(smem_barrier + 1 + stage_idx * 2, phase_bit);
      }
      int next_stage_idx = stage_idx + 1;
      int next_phase_bit = next_stage_idx == stage_cnt ? phase_bit ^ 1 : phase_bit;
      next_stage_idx = next_stage_idx == stage_cnt ? 0 : next_stage_idx;
      if (loop_idx != k_iter_cnt - 1) {
        need_wait = !try_wait_barrier(smem_barrier + 1 + next_stage_idx * 2, next_phase_bit);
      }
#pragma unroll
      for (int i = 0; i < b_inst_cnt_per_iter; i++) {
        int smem_offset = b_smem_offsets[i];
        bf16_t* smem_ptr_this_iter = smem_b + stage_idx * tile_n * tile_k + smem_offset;
        int linear_idx = local_tid * vec_elems + i * thread_cnt * vec_elems;
        int n_idx = linear_idx / tile_k;
        int k_idx = linear_idx % tile_k;
        int gmem_offset = n_idx * gemm_k + k_project(k_idx);
        bf16_t const* gmem_ptr_this_iter = gmem_b + gmem_offset;
        ldgsts_128(gmem_ptr_this_iter, smem_ptr_this_iter, preds[i]);
      }
      ldgsts_arrive(smem_barrier + stage_idx * 2);

      stage_idx = next_stage_idx;
      phase_bit = next_phase_bit;
      gmem_b += per_mma_warp_k;
    }
#endif
  }

  bf16_t const* gmem_b;
  bf16_t* smem_b;
  uint64_t* smem_barrier;
  int gemm_n;
  int local_tid;
  int stage_idx = 0;
  int phase_bit = 1;
  bool need_wait = true;

  int b_smem_offsets[b_inst_cnt_per_iter];
  uint32_t preds[b_inst_cnt_per_iter];
};

template <int gemm_m, int gemm_k, int tile_m, int tile_n, int tile_k, int stage_cnt>
struct MmaComputer {
  static constexpr int elem_bytes = 2;
  static constexpr int thread_cnt = 128;
  static_assert(gemm_k % tile_k == 0);
  static_assert(tile_k % (thread_cnt / 32) == 0);
  static constexpr int per_warp_tile_k = tile_k / (thread_cnt / 32);
  static constexpr int k_iter_cnt = gemm_k / tile_k;
  static constexpr int k_phase_cnt = per_warp_tile_k / 16;
  static constexpr int m_iter_cnt = (tile_m + 15) / 16;
  static constexpr int n_iter_cnt = (tile_n + 7) / 8;
  static_assert(m_iter_cnt == 1 || m_iter_cnt == 2);
  static_assert(n_iter_cnt == 1 || n_iter_cnt == 2);

  __device__ MmaComputer(
      bf16_t* gmem_c_local_, bf16_t* smem_a_, bf16_t* smem_b_, uint64_t* smem_barrier_, int warp_idx_, int gemm_n_)
      : gmem_c(gmem_c_local_),
        smem_a(smem_a_),
        smem_b(smem_b_),
        smem_barrier(smem_barrier_),
        warp_idx(warp_idx_ - (thread_cnt / 32)),
        gemm_n(gemm_n_) {}

 private:
  __device__ constexpr int internal_b_atom_func(int tid) {
    if constexpr (tile_n < 8) {
      return (tid % tile_n) + ((tid % 8) / tile_n * 0) + tid / 8 * 8 * tile_n;
    } else {
      return (tid % 8) + ((tid % 32) / 8 * (tile_n * 8));
    }
  }

 public:
  __device__ void prepare() {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
#pragma unroll
    for (int m = 0; m < m_iter_cnt; m++) {
#pragma unroll
      for (int i = 0; i < k_phase_cnt; i++) {
        int linear_idx = (lane_idx % 16) + (lane_idx / 16) * 128 + i * 256;
        int m_idx = linear_idx % 16 + m * 16;
        int k_idx = linear_idx / 16 + warp_k_offset_in_tile_k;
        k_idx = apply_swizzle_343_on_elem_row_col<bf16_t>(m_idx, k_idx);
        a_smem_offsets[m][i] = m_idx * tile_k + k_idx;
      }
    }
#pragma unroll
    for (int n_iter_idx = 0; n_iter_idx < n_iter_cnt; n_iter_idx++) {
#pragma unroll
      for (int i = 0; i < k_phase_cnt; i += 2) {
        int linear_idx = internal_b_atom_func(lane_idx) + i * tile_n * 16 + n_iter_idx * 8;
        int n_idx = linear_idx % tile_n;
        int k_idx = linear_idx / tile_n + warp_k_offset_in_tile_k;
        k_idx = apply_swizzle_343_on_elem_row_col<bf16_t>(n_idx, k_idx);
        b_smem_offsets[n_iter_idx][i] = n_idx * tile_k + k_idx;
      }
    }
#endif
  }

  __device__ void issue_mainloop() {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
#pragma unroll 1
    for (int loop_idx = 0; loop_idx < k_iter_cnt; loop_idx++) {
      wait_barrier(smem_barrier + 0 + stage_idx * 2, phase_bit);

#pragma unroll
      for (int m = 0; m < m_iter_cnt; m++) {
#pragma unroll
        for (int i = 0; i < k_phase_cnt; i++) {
          int smem_offset = a_smem_offsets[m][i];
          bf16_t* smem_ptr_this_iter = smem_a + stage_idx * tile_m * tile_k + smem_offset;
          ldsm_x4(smem_ptr_this_iter, reinterpret_cast<uint32_t*>(a_reg[m][i]));
        }
      }

#pragma unroll
      for (int n_iter_idx = 0; n_iter_idx < n_iter_cnt; n_iter_idx++) {
#pragma unroll
        for (int i = 0; i < k_phase_cnt; i += 2) {
          int smem_offset = b_smem_offsets[n_iter_idx][i];
          bf16_t* smem_ptr_this_iter = smem_b + stage_idx * tile_n * tile_k + smem_offset;
          ldsm_x4(smem_ptr_this_iter, reinterpret_cast<uint32_t*>(b_reg[n_iter_idx][i]));
        }
      }

#pragma unroll
      for (int k_iter_idx = 0; k_iter_idx < k_phase_cnt; k_iter_idx++) {
#pragma unroll
        for (int n_iter_idx = 0; n_iter_idx < n_iter_cnt; n_iter_idx++) {
#pragma unroll
          for (int m = 0; m < m_iter_cnt; m++) {
            hmma_16_8_16_f32acc_bf16ab(
                acc_reg[m][n_iter_idx], a_reg[m][k_iter_idx], b_reg[n_iter_idx][k_iter_idx], acc_reg[m][n_iter_idx]);
          }
        }
      }
      arrive_barrier(smem_barrier + 1 + stage_idx * 2);
      stage_idx += 1;
      phase_bit = stage_idx == stage_cnt ? phase_bit ^ 1 : phase_bit;
      stage_idx = stage_idx == stage_cnt ? 0 : stage_idx;
    }
#endif
  }

  __device__ void epi() {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
    asm volatile("bar.sync %0, %1;" : : "r"(1), "r"(thread_cnt));
    constexpr int thread_m = 2 * m_iter_cnt;
    constexpr int thread_n = 2 * n_iter_cnt;
    constexpr int cta_mma_n = n_iter_cnt * 8;
    float acc_reg_reorg[thread_m][thread_n];

    for (int i = 0; i < thread_m; i++) {
      for (int j = 0; j < thread_n; j++) {
        acc_reg_reorg[i][j] = acc_reg[i / 2][j / 2][(j % 2) + (i % 2) * 2];
      }
    }

    float* smem_c = reinterpret_cast<float*>(smem_a);
    auto smem_c_index_func = [&](int m_idx, int n_idx) {
      int group_rows = 32 / cta_mma_n;
      int group_cnt = 2;
      return (m_idx % group_rows * cta_mma_n) + (m_idx / group_rows * (32 + group_cnt)) + n_idx;
    };
    constexpr int cosize_smem_c = ((tile_m * cta_mma_n) / 32) * (32 + 2);

#pragma unroll
    for (int m_idx_thread = 0; m_idx_thread < thread_m; m_idx_thread++) {
#pragma unroll
      for (int n_idx_thread = 0; n_idx_thread < thread_n; n_idx_thread++) {
        int m_idx = (lane_idx / 4) + (m_idx_thread % 2) * 8 + (m_idx_thread / 2) * 16;
        int n_idx = ((lane_idx % 4) * 2) + (n_idx_thread % 2) + (n_idx_thread / 2) * 8;
        smem_c[cosize_smem_c * warp_idx + smem_c_index_func(m_idx, n_idx)] = acc_reg_reorg[m_idx_thread][n_idx_thread];
      }
    }
    asm volatile("bar.sync %0, %1;" : : "r"(1), "r"(thread_cnt));

    if (warp_idx == 0) {
      constexpr int final_acc_reg_cnt = (tile_m * tile_n + 31) / 32;
      float acc_final[final_acc_reg_cnt]{};

#pragma unroll
      for (int reg_idx = 0; reg_idx < final_acc_reg_cnt; reg_idx++) {
        int linear_idx = reg_idx * 32 + lane_idx;
        int m_idx = linear_idx % tile_m;
        int n_idx = linear_idx / tile_m;
        acc_final[reg_idx] += smem_c[smem_c_index_func(m_idx, n_idx) + 0 * cosize_smem_c] +
                              smem_c[smem_c_index_func(m_idx, n_idx) + 1 * cosize_smem_c] +
                              smem_c[smem_c_index_func(m_idx, n_idx) + 2 * cosize_smem_c] +
                              smem_c[smem_c_index_func(m_idx, n_idx) + 3 * cosize_smem_c];
      }

#pragma unroll
      for (int reg_idx = 0; reg_idx < final_acc_reg_cnt; reg_idx++) {
        int linear_idx = reg_idx * 32 + lane_idx;
        int m_idx = linear_idx % tile_m;
        int n_idx = linear_idx / tile_m;
        if (m_idx < tile_m && n_idx < gemm_n) {
          gmem_c[n_idx * gemm_m + m_idx] = __float2bfloat16(acc_final[reg_idx]);
        }
      }
    }
#endif
  }

  bf16_t* gmem_c;
  bf16_t* smem_a;
  bf16_t* smem_b;
  uint64_t* smem_barrier;
  int warp_idx;
  int gemm_n;
  int stage_idx = 0;
  int phase_bit = 0;
  int lane_idx = threadIdx.x % 32;
  int warp_k_offset_in_tile_k = warp_idx * per_warp_tile_k;

  int a_smem_offsets[m_iter_cnt][k_phase_cnt];
  int b_smem_offsets[n_iter_cnt][k_phase_cnt];

  bf16_t a_reg[m_iter_cnt][k_phase_cnt][8];
  bf16_t b_reg[n_iter_cnt][k_phase_cnt][4];
  float acc_reg[m_iter_cnt][n_iter_cnt][4]{};
};

// MmaComputer with a pipelined main loop: the next stage's fragments go into a second register
// buffer while the current stage's HMMAs issue, so neither the full-barrier test nor the ldmatrix
// latency sits between the dependent HMMAs of consecutive stages. Every accumulator sees the same
// HMMA sequence on the same fragments as in MmaComputer::issue_mainloop, so outputs are bit-identical.
template <int gemm_m, int gemm_k, int tile_m, int tile_n, int tile_k, int stage_cnt>
struct MmaComputerIlp : MmaComputer<gemm_m, gemm_k, tile_m, tile_n, tile_k, stage_cnt> {
  using Base = MmaComputer<gemm_m, gemm_k, tile_m, tile_n, tile_k, stage_cnt>;

  __device__ MmaComputerIlp(
      bf16_t* gmem_c_local_, bf16_t* smem_a_, bf16_t* smem_b_, uint64_t* smem_barrier_, int warp_idx_, int gemm_n_)
      : Base(gmem_c_local_, smem_a_, smem_b_, smem_barrier_, warp_idx_, gemm_n_) {}

  static constexpr int k_iter_cnt = Base::k_iter_cnt;
  static constexpr int k_phase_cnt = Base::k_phase_cnt;
  static constexpr int m_iter_cnt = Base::m_iter_cnt;
  static constexpr int n_iter_cnt = Base::n_iter_cnt;
  static_assert(k_iter_cnt % 2 == 0, "the pipelined main loop takes k iterations in pairs");
  static_assert(k_phase_cnt % 2 == 0, "the pipelined main loop splits each stage's k phases in halves");

  __device__ void issue_mainloop() {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
    wait_barrier(this->smem_barrier + 0 + this->stage_idx * 2, this->phase_bit);
    load_fragments<0>(this->stage_idx);
#pragma unroll 1
    for (int loop_idx = 0; loop_idx < k_iter_cnt; loop_idx += 2) {
      issue_stage<0>(loop_idx);
      issue_stage<1>(loop_idx + 1);
    }
#endif
  }

 private:
  template <int buf>
  __device__ auto& a_frag() {
    if constexpr (buf == 0) {
      return this->a_reg;
    } else {
      return a_reg_next;
    }
  }

  template <int buf>
  __device__ auto& b_frag() {
    if constexpr (buf == 0) {
      return this->b_reg;
    } else {
      return b_reg_next;
    }
  }

  // One k iteration; its fragments are already in buffer buf.
  template <int buf>
  __device__ void issue_stage(int loop_idx) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
    // This stage's fragments are already in registers. Release its slot now, not after the next
    // stage lands: arriving later costs the loaders one stage of pipeline depth when HBM-bound.
    // The arrive's release semantics order the ldmatrix reads before the slot is refilled.
    arrive_barrier(this->smem_barrier + 1 + this->stage_idx * 2);
    this->stage_idx += 1;
    this->phase_bit = this->stage_idx == stage_cnt ? this->phase_bit ^ 1 : this->phase_bit;
    this->stage_idx = this->stage_idx == stage_cnt ? 0 : this->stage_idx;
    const bool has_next = loop_idx + 1 < k_iter_cnt;
    const bool next_ready =
        !has_next || test_wait_barrier(this->smem_barrier + 0 + this->stage_idx * 2, this->phase_bit);
    issue_mma<buf, 0, k_phase_cnt / 2>();
    if (has_next) {
      if (!next_ready) {
        wait_barrier(this->smem_barrier + 0 + this->stage_idx * 2, this->phase_bit);
      }
      load_fragments<buf ^ 1>(this->stage_idx);
    }
    issue_mma<buf, k_phase_cnt / 2, k_phase_cnt>();
#endif
  }

  template <int buf>
  __device__ void load_fragments(int stage) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
    auto& a = a_frag<buf>();
    auto& b = b_frag<buf>();
#pragma unroll
    for (int m = 0; m < m_iter_cnt; m++) {
#pragma unroll
      for (int i = 0; i < k_phase_cnt; i++) {
        bf16_t* smem_ptr_this_iter = this->smem_a + stage * tile_m * tile_k + this->a_smem_offsets[m][i];
        ldsm_x4(smem_ptr_this_iter, reinterpret_cast<uint32_t*>(a[m][i]));
      }
    }
#pragma unroll
    for (int n_iter_idx = 0; n_iter_idx < n_iter_cnt; n_iter_idx++) {
#pragma unroll
      for (int i = 0; i < k_phase_cnt; i += 2) {
        bf16_t* smem_ptr_this_iter = this->smem_b + stage * tile_n * tile_k + this->b_smem_offsets[n_iter_idx][i];
        ldsm_x4(smem_ptr_this_iter, reinterpret_cast<uint32_t*>(b[n_iter_idx][i]));
      }
    }
#endif
  }

  // k phases [k_begin, k_end) in MmaComputer::issue_mainloop's order.
  template <int buf, int k_begin, int k_end>
  __device__ void issue_mma() {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
    auto& a = a_frag<buf>();
    auto& b = b_frag<buf>();
#pragma unroll
    for (int k_iter_idx = k_begin; k_iter_idx < k_end; k_iter_idx++) {
#pragma unroll
      for (int n_iter_idx = 0; n_iter_idx < n_iter_cnt; n_iter_idx++) {
#pragma unroll
        for (int m = 0; m < m_iter_cnt; m++) {
          hmma_16_8_16_f32acc_bf16ab(
              this->acc_reg[m][n_iter_idx], a[m][k_iter_idx], b[n_iter_idx][k_iter_idx], this->acc_reg[m][n_iter_idx]);
        }
      }
    }
#endif
  }

  bf16_t a_reg_next[m_iter_cnt][k_phase_cnt][8];
  bf16_t b_reg_next[n_iter_cnt][k_phase_cnt][4];
};

// SGLang's fused_a_gemm_kernel (batch_size 1); kIlp: the MMA warps run MmaComputerIlp. The stage
// count is a template argument, so the deep ring is a Tiling choice. Same outputs either way.
template <int gemm_m, int gemm_k, int tile_m, int tile_n, int tile_k, int stage_cnt, bool kIlp>
__global__ __launch_bounds__(256, 1) void fused_a_gemm_ll_kernel(
    bf16_t* output, bf16_t const* mat_a, bf16_t const* mat_b, int gemm_n) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
  constexpr int load_thread_cnt = 128;
  constexpr int compute_thread_cnt = 128;
  constexpr int k_iter_cnt = gemm_k / tile_k;
  static_assert(gemm_m % 16 == 0);
  static_assert(gemm_k % tile_k == 0);
  static_assert(gemm_m % tile_m == 0);
  static_assert(tile_k == 128 || tile_k == 256 || tile_k == 512 || tile_k == 1024);
  static_assert(tile_m == 16 || tile_m == 32);
  static_assert(stage_cnt <= k_iter_cnt);
  constexpr int g2s_vec_bytes = 16;
  constexpr int a_elem_bytes = 2;
  constexpr int b_elem_bytes = 2;
  static_assert((tile_m * a_elem_bytes + tile_n * b_elem_bytes) * tile_k * stage_cnt <= 225 * 1024);
  static_assert((tile_m * tile_k * a_elem_bytes) % (load_thread_cnt * g2s_vec_bytes) == 0);
  static_assert((tile_n * tile_k * b_elem_bytes) % (load_thread_cnt * g2s_vec_bytes) == 0);

  extern __shared__ char smem[];
  uint64_t* smem_barrier = reinterpret_cast<uint64_t*>(smem);
  bf16_t* smem_a = reinterpret_cast<bf16_t*>(smem + (stage_cnt * 8 * 2 + 1024) / 1024 * 1024);
  bf16_t* smem_b = smem_a + tile_m * tile_k * stage_cnt;

  int cta_m_idx = tile_m * blockIdx.x;
  int cta_n_idx = tile_n * blockIdx.y;
  bf16_t const* gmem_a_local = mat_a + cta_m_idx * gemm_k;
  bf16_t const* gmem_b_local = mat_b + cta_n_idx * gemm_k;
  bf16_t* gmem_c_local = output + cta_n_idx * gemm_m + cta_m_idx;

  int warp_idx = __shfl_sync(0xffffffff, threadIdx.x / 32, 0);

  if (warp_idx == 4) {
    for (int i = 0; i < stage_cnt; i++) {
      initialize_barrier(smem_barrier + i * 2 + 0, load_thread_cnt);
      initialize_barrier(smem_barrier + i * 2 + 1, compute_thread_cnt);
    }
  }
  __syncthreads();

  if (warp_idx < 2) {
    // Weights only: no kernel writes them during a forward, so nothing here waits on the producer grid.
    GmemLoaderA<gemm_k, tile_m, tile_k, stage_cnt> a_loader(gmem_a_local, smem_a, smem_barrier);
    a_loader.prepare();
    a_loader.issue_mainloop();
  } else if (warp_idx < 4) {
    GmemLoaderB<gemm_k, tile_n, tile_k, stage_cnt> b_loader(gmem_b_local, smem_b, smem_barrier, gemm_n);
    b_loader.prepare();
    b_loader.issue_mainloop();
  } else if constexpr (kIlp) {
    MmaComputerIlp<gemm_m, gemm_k, tile_m, tile_n, tile_k, stage_cnt> mma_computer(
        gmem_c_local, smem_a, smem_b, smem_barrier, warp_idx, gemm_n);
    mma_computer.prepare();
    mma_computer.issue_mainloop();
    mma_computer.epi();
  } else {
    MmaComputer<gemm_m, gemm_k, tile_m, tile_n, tile_k, stage_cnt> mma_computer(
        gmem_c_local, smem_a, smem_b, smem_barrier, warp_idx, gemm_n);
    mma_computer.prepare();
    mma_computer.issue_mainloop();
    mma_computer.epi();
  }
  cudaTriggerProgrammaticLaunchCompletion();
#endif
}

// kStageBudgetKB: shared memory for the stage ring on SM10x (192 in the image's kernel).
template <int kHdIn, int kHdOut, int kTileN, int kTileM, int kStageBudgetKB = 192>
struct FusedAGemmTiling {
  static constexpr int gemm_m = kHdOut;
  static constexpr int gemm_k = kHdIn;
  static constexpr int tile_m = kTileM;
  static constexpr int tile_n = kTileN;                        // 8 or 16
  static constexpr int tile_k = std::max(256, 1024 / tile_n);  // 256
#if defined(SGL_CUDA_ARCH) && SGL_CUDA_ARCH >= 1200
  static constexpr int smem_stage_budget = 96 * 1024;
#else
  static constexpr int smem_stage_budget = kStageBudgetKB * 1024;
#endif
  static constexpr int max_stage_cnt = smem_stage_budget / ((tile_m + tile_n) * tile_k * sizeof(bf16_t));
  static constexpr int k_iter_cnt = gemm_k / tile_k;
  static constexpr int stage_cnt = k_iter_cnt > max_stage_cnt ? max_stage_cnt : k_iter_cnt;
  static constexpr int barrier_bytes = (stage_cnt * 16 + 1023) / 1024 * 1024;  // 4096
  static constexpr int smem_bytes =
      ((tile_m * 2 + tile_n * 2) * tile_k * stage_cnt + barrier_bytes + 1023) / 1024 * 1024;
};

// Stage ring of the deep variant: 11 stages at 8 rows, 9 at 16, against SGLang's 9 and 8 at 192 KB.
inline constexpr int kFusedADeepStageBudgetKB = 224;

template <
    typename T,
    int kHdIn,
    int kHdOut,
    int kTileN,
    int kTileM,
    bool kUsePDL,
    bool kIlp,
    bool kDeep>
void invokeFusedAGemmLL(T* output, T const* mat_a, T const* mat_b, int num_tokens, DLDevice device) {
  using Tiling = FusedAGemmTiling<kHdIn, kHdOut, kTileN, kTileM, kDeep ? kFusedADeepStageBudgetKB : 192>;
  int const gemm_n = num_tokens;
  std::swap(mat_a, mat_b);
  int cta_m_cnt = Tiling::gemm_m / Tiling::tile_m;
  int cta_n_cnt = (gemm_n + Tiling::tile_n - 1) / Tiling::tile_n;
  constexpr int smem_bytes = Tiling::smem_bytes;

  dim3 grid(cta_m_cnt, cta_n_cnt, 1);
  dim3 block_size(256);

  auto kernel = fused_a_gemm_ll_kernel<
      Tiling::gemm_m,
      Tiling::gemm_k,
      Tiling::tile_m,
      Tiling::tile_n,
      Tiling::tile_k,
      Tiling::stage_cnt,
      kIlp>;
  if (smem_bytes >= (48 * 1024)) {
    host::RuntimeDeviceCheck(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_bytes));
  }
  host::LaunchKernel(grid, block_size, device, smem_bytes).enable_pdl(kUsePDL)(kernel, output, mat_a, mat_b, gemm_n);
}

// tile_m=32 halves the CTA count so these shapes fit in one SM wave instead of two (SGLang's choice).
constexpr int pick_tile_m(int hd_in, int hd_out) {
  if (hd_out == 2624 && hd_in == 6144) return 32;
  if (hd_out == 4096 && hd_in == 2048) return 32;
  return 16;
}

// SGLang's DSV3FusedAGemmKernel contract (1-16 rows) on fused_a_gemm_ll_kernel.
template <int kHdIn, int kHdOut, bool kUsePDL, bool kIlp, bool kDeep>
struct DSV3FusedAGemmLLKernel {
  static constexpr int kTileM = pick_tile_m(kHdIn, kHdOut);
  static_assert(kHdOut % kTileM == 0, "hd_out must be a multiple of tile_m");

  template <int kTileN>
  static void launch(bf16_t* out, bf16_t const* a, bf16_t const* b, int num_tokens, DLDevice dev) {
    invokeFusedAGemmLL<bf16_t, kHdIn, kHdOut, kTileN, kTileM, kUsePDL, kIlp, kDeep>(out, a, b, num_tokens, dev);
  }

  static void
  run(const tvm::ffi::TensorView mat_a, const tvm::ffi::TensorView mat_b, const tvm::ffi::TensorView output) {
    using namespace host;

    auto M = SymbolicSize{"num_tokens"};
    auto K = SymbolicSize{"hd_in"};
    auto N = SymbolicSize{"hd_out"};
    auto device = SymbolicDevice{};
    K.set_value(kHdIn);
    N.set_value(kHdOut);
    device.set_options<kDLCUDA>();

    TensorMatcher({M, K}).with_dtype<bf16_t>().with_device(device).verify(mat_a);
    TensorMatcher({K, N}).with_dtype<bf16_t>().with_device(device).with_strides({1, kHdIn}).verify(mat_b);
    TensorMatcher({M, N}).with_dtype<bf16_t>().with_device(device).verify(output);

    const int num_tokens = static_cast<int>(M.unwrap());
    RuntimeCheck(
        num_tokens >= 1 && num_tokens <= 16,
        "glm_ll fused_a: num_tokens must be in [1, 16], got ",
        num_tokens);

    const DLDevice dev = device.unwrap();
    auto* out_ptr = static_cast<bf16_t*>(output.data_ptr());
    auto* a_ptr = static_cast<bf16_t const*>(mat_a.data_ptr());
    auto* b_ptr = static_cast<bf16_t const*>(mat_b.data_ptr());

    if (num_tokens <= 8) {
      launch<8>(out_ptr, a_ptr, b_ptr, num_tokens, dev);
    } else {
      launch<16>(out_ptr, a_ptr, b_ptr, num_tokens, dev);
    }
  }
};

// ---------------------------------------------------------------------------------------------
// Wide fused-A for 17-48 rows. fused_a_gemm_kernel holds a whole K row per CTA, so above 16 rows
// each of its CTAs also streams up to 48 activation rows and runs 3-6x the HMMAs on one SM.
// Here a cluster of kSplitK CTAs splits K: tile_m 64 x 3 splits is 123 CTAs for 2624 outputs,
// against 82 for fused_a_gemm_kernel. Partial tiles are summed through DSMEM in split order,
// so outputs are deterministic (not bit-identical to the <= 16-row kernel, which sums in another
// order). kSplitK == 1 is a plain CTA-per-tile variant of the same main loop.
// Warps 0-1 stream weights (they never wait on the PDL producer), warps 2-3 stream activations
// after the grid dependency wait, kMmaWarps warps (4 or 8) from warp 4 on run HMMAs on an equal
// share of each stage's k range.

__device__ __forceinline__ void hmma_16816_f32_bf16(float (&d)[4], uint32_t const (&a)[4], uint32_t b0, uint32_t b1) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
  asm volatile(
      "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
      "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
      : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
#endif
}

template <int kThreads>
__device__ __forceinline__ void mma_warps_sync() {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
  asm volatile("bar.sync 1, %0;\n" ::"n"(kThreads) : "memory");
#endif
}

// Two warps copy a kRows x kTileK bf16 tile per k iteration (row stride kRowStride elements) into a
// ring of kStages swizzled stages. Rows >= valid_rows are not copied; they only feed output columns
// that are never stored.
template <int kRowStride, int kRows, int kTileK, int kStages, int kIters>
struct WideStageLoader {
  static constexpr int kThreads = 64;
  static constexpr int kVecsPerRow = kTileK / 8;
  static_assert((kRows * kVecsPerRow) % kThreads == 0);
  static constexpr int kVecsPerThread = kRows * kVecsPerRow / kThreads;

  __device__ static void run(bf16_t const* gmem, bf16_t* smem, uint64_t* barriers, int valid_rows, int local_tid) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
    int stage = 0;
    int phase = 1;
#pragma unroll 1
    for (int iter = 0; iter < kIters; iter++) {
      wait_barrier(barriers + 1 + stage * 2, phase);
      bf16_t* stage_smem = smem + stage * kRows * kTileK;
#pragma unroll
      for (int i = 0; i < kVecsPerThread; i++) {
        const int vec = local_tid + i * kThreads;
        const int row = vec / kVecsPerRow;
        const int col = (vec % kVecsPerRow) * 8;
        ldgsts_128(
            gmem + row * kRowStride + iter * kTileK + col,
            stage_smem + row * kTileK + apply_swizzle_343_on_elem_row_col<bf16_t>(row, col),
            row < valid_rows);
      }
      ldgsts_arrive(barriers + stage * 2);
      if (++stage == kStages) {
        stage = 0;
        phase ^= 1;
      }
    }
#endif
  }
};

template <int kTileM, int kTileN, int kTileK, int kStages, int kIters, int kMmaWarps>
struct WideMma {
  static constexpr int kThreads = kMmaWarps * 32;
  static constexpr int kMIter = kTileM / 16;
  static constexpr int kNIter = kTileN / 8;
  static constexpr int kWarpK = kTileK / kMmaWarps;
  static constexpr int kKPhase = kWarpK / 16;
  static_assert(kTileM % 16 == 0 && kTileN % 8 == 0 && kKPhase % 2 == 0);
  // Row stride (floats) of the epilogue tiles; +4 keeps the fragment stores conflict-free.
  static constexpr int kLdm = kTileM + 4;
  static constexpr int kSliceFloats = kTileN * kLdm;

  float acc[kMIter][kNIter][4];

  __device__ void mainloop(bf16_t const* smem_a, bf16_t const* smem_b, uint64_t* barriers, int mma_warp, int lane) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
#pragma unroll
    for (int m = 0; m < kMIter; m++)
#pragma unroll
      for (int n = 0; n < kNIter; n++)
#pragma unroll
        for (int j = 0; j < 4; j++)
          acc[m][n][j] = 0.f;
    // ldmatrix x4 lane addresses: A rows 0-15 at k +0 / +8; B rows 0-7 at k +0, +8, +16, +24.
    const int a_row = lane % 16;
    const int a_col = mma_warp * kWarpK + (lane / 16) * 8;
    const int b_row = lane % 8;
    const int b_col = mma_warp * kWarpK + (lane / 8) * 8;
    int stage = 0;
    int phase = 0;
#pragma unroll 1
    for (int iter = 0; iter < kIters; iter++) {
      wait_barrier(barriers + stage * 2, phase);
      bf16_t const* stage_a = smem_a + stage * kTileM * kTileK;
      bf16_t const* stage_b = smem_b + stage * kTileN * kTileK;
#pragma unroll
      for (int kp = 0; kp < kKPhase; kp += 2) {
        uint32_t a[kMIter][2][4];
        uint32_t b[kNIter][4];
#pragma unroll
        for (int m = 0; m < kMIter; m++) {
#pragma unroll
          for (int h = 0; h < 2; h++) {
            const int row = m * 16 + a_row;
            const int col = a_col + (kp + h) * 16;
            ldsm_x4(
                const_cast<bf16_t*>(stage_a + row * kTileK + apply_swizzle_343_on_elem_row_col<bf16_t>(row, col)),
                a[m][h]);
          }
        }
#pragma unroll
        for (int n = 0; n < kNIter; n++) {
          const int row = n * 8 + b_row;
          const int col = b_col + kp * 16;
          ldsm_x4(
              const_cast<bf16_t*>(stage_b + row * kTileK + apply_swizzle_343_on_elem_row_col<bf16_t>(row, col)), b[n]);
        }
        if (kp + 2 == kKPhase) {
          // Release semantics order the ldmatrix reads above before the loaders refill the stage.
          arrive_barrier(barriers + 1 + stage * 2);
        }
#pragma unroll
        for (int h = 0; h < 2; h++)
#pragma unroll
          for (int n = 0; n < kNIter; n++)
#pragma unroll
            for (int m = 0; m < kMIter; m++)
              hmma_16816_f32_bf16(acc[m][n], a[m][h], b[n][2 * h], b[n][2 * h + 1]);
      }
      if (++stage == kStages) {
        stage = 0;
        phase ^= 1;
      }
    }
#endif
  }

  // Sums the MMA warps' tiles, in warp order, into red[n * kLdm + m] (fp32).
  __device__ void reduce_warps(float* red, int mma_warp, int lane, int mma_tid) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
    mma_warps_sync<kThreads>();  // every MMA warp is done reading the stages that red overlays
    const int g = lane / 4;
    const int t = lane % 4;
#pragma unroll
    for (int m = 0; m < kMIter; m++)
#pragma unroll
      for (int n = 0; n < kNIter; n++)
#pragma unroll
        for (int j = 0; j < 4; j++) {
          const int row = m * 16 + g + (j >> 1) * 8;
          const int col = n * 8 + t * 2 + (j & 1);
          red[mma_warp * kSliceFloats + col * kLdm + row] = acc[m][n][j];
        }
    mma_warps_sync<kThreads>();
#pragma unroll 4
    for (int e = mma_tid; e < kTileM * kTileN; e += kThreads) {
      const int idx = (e / kTileM) * kLdm + e % kTileM;
      float v = red[idx];
#pragma unroll
      for (int w = 1; w < kMmaWarps; w++) {
        v += red[idx + w * kSliceFloats];
      }
      red[idx] = v;
    }
#endif
  }
};

template <int kGemmM, int kGemmK, int kTileM, int kTileN, int kTileK, int kStages, int kSplitK, int kMmaWarps>
__global__ __launch_bounds__(128 + kMmaWarps * 32, 1) void fused_a_gemm_wide_kernel(
    bf16_t* output, bf16_t const* weight, bf16_t const* act, int num_tokens) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
  constexpr int kKPerCta = kGemmK / kSplitK;
  constexpr int kIters = kKPerCta / kTileK;
  constexpr int kThreads = 128 + kMmaWarps * 32;
  static_assert(kGemmK % (kSplitK * kTileK) == 0 && kGemmM % kTileM == 0);
  static_assert(kSplitK >= 1 && kSplitK <= 8 && kStages <= kIters && kStages * 16 <= 1024);
  static_assert(kMmaWarps == 4 || kMmaWarps == 8);
  using Mma = WideMma<kTileM, kTileN, kTileK, kStages, kIters, kMmaWarps>;
  static_assert(
      kMmaWarps * Mma::kSliceFloats * 4 <= kStages * (kTileM + kTileN) * kTileK * 2, "epilogue overlays the stages");

  extern __shared__ __align__(1024) char smem[];
  uint64_t* barriers = reinterpret_cast<uint64_t*>(smem);
  bf16_t* smem_a = reinterpret_cast<bf16_t*>(smem + 1024);
  bf16_t* smem_b = smem_a + kStages * kTileM * kTileK;
  float* red = reinterpret_cast<float*>(smem_a);

  int split_idx = 0;
  if constexpr (kSplitK > 1) {
    split_idx = static_cast<int>(cooperative_groups::this_cluster().block_rank());
  }
  const int m_base = (blockIdx.x / kSplitK) * kTileM;
  const int warp_idx = __shfl_sync(0xffffffff, threadIdx.x / 32, 0);
  const int lane = threadIdx.x % 32;

  if (threadIdx.x == 0) {
    for (int i = 0; i < kStages; i++) {
      initialize_barrier(barriers + i * 2 + 0, 128);  // full: 64 weight + 64 activation loader arrivals
      initialize_barrier(barriers + i * 2 + 1, Mma::kThreads);  // empty: every MMA thread
    }
  }
  __syncthreads();

  if (warp_idx < 2) {
    WideStageLoader<kGemmK, kTileM, kTileK, kStages, kIters>::run(
        weight + m_base * kGemmK + split_idx * kKPerCta, smem_a, barriers, kTileM, threadIdx.x % 64);
    // Same release point as fused_a_gemm_kernel: once the weight loader has issued its last stage.
    cudaTriggerProgrammaticLaunchCompletion();
  } else if (warp_idx < 4) {
    cudaGridDependencySynchronize();
    WideStageLoader<kGemmK, kTileN, kTileK, kStages, kIters>::run(
        act + split_idx * kKPerCta, smem_b, barriers, num_tokens, threadIdx.x % 64);
  } else {
    Mma mma;
    mma.mainloop(smem_a, smem_b, barriers, warp_idx - 4, lane);
    mma.reduce_warps(red, warp_idx - 4, lane, threadIdx.x - 128);
    if constexpr (kSplitK == 1) {
#pragma unroll 4
      for (int e = threadIdx.x - 128; e < kTileM * kTileN; e += Mma::kThreads) {
        const int n = e / kTileM;
        const int m = e % kTileM;
        if (n < num_tokens) {
          output[n * kGemmM + m_base + m] = __float2bfloat16(red[n * Mma::kLdm + m]);
        }
      }
    }
  }

  if constexpr (kSplitK > 1) {
    // Every CTA's partial tile is in its red; each CTA then finishes a 1/kSplitK slice of the
    // tile, summing the partials in split order. The second sync keeps every CTA's shared memory
    // alive until the cluster has read it.
    auto cluster = cooperative_groups::this_cluster();
    cluster.sync();
    float const* parts[kSplitK];
#pragma unroll
    for (int r = 0; r < kSplitK; r++) {
      parts[r] = cluster.map_shared_rank(red, r);
    }
    constexpr int kElems = kTileM * kTileN;
    constexpr int kChunk = (kElems + kSplitK - 1) / kSplitK;
    const int end = min(kElems, (split_idx + 1) * kChunk);
    for (int e = split_idx * kChunk + static_cast<int>(threadIdx.x); e < end; e += kThreads) {
      const int n = e / kTileM;
      const int m = e % kTileM;
      if (n < num_tokens) {
        const int idx = n * Mma::kLdm + m;
        float v = parts[0][idx];
#pragma unroll
        for (int r = 1; r < kSplitK; r++) {
          v += parts[r][idx];
        }
        output[n * kGemmM + m_base + m] = __float2bfloat16(v);
      }
    }
    cluster.sync();
  }
#endif
}

// Max rows of the wide kernel; it is meant for 17..48, but serves 1..48 for testing.
inline constexpr int kFusedAWideMaxRows = 48;

template <int kHdIn, int kHdOut, int kTileM, int kTileN, int kTileK, int kSplitK>
struct FusedAWideTiling {
  static constexpr int stage_bytes = (kTileM + kTileN) * kTileK * 2;
  static constexpr int iters = kHdIn / kSplitK / kTileK;
  // Same stage budget as fused_a_gemm_kernel.
  static constexpr int max_stages = 192 * 1024 / stage_bytes;
  static constexpr int stages = iters < max_stages ? iters : max_stages;
  static constexpr int smem_bytes = 1024 + stages * stage_bytes;
};

template <int kHdIn, int kHdOut, int kTileM, int kTileN, int kTileK, int kSplitK, int kMmaWarps, bool kUsePDL>
void invokeFusedAGemmWide(
    bf16_t* output, bf16_t const* act, bf16_t const* weight, int num_tokens, DLDevice device) {
  using Tiling = FusedAWideTiling<kHdIn, kHdOut, kTileM, kTileN, kTileK, kSplitK>;
  auto kernel =
      fused_a_gemm_wide_kernel<kHdOut, kHdIn, kTileM, kTileN, kTileK, Tiling::stages, kSplitK, kMmaWarps>;
  host::RuntimeDeviceCheck(
      cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, Tiling::smem_bytes));
  host::LaunchKernel launch(
      dim3(kHdOut / kTileM * kSplitK), dim3(128 + kMmaWarps * 32), device, Tiling::smem_bytes);
  launch.enable_pdl(kUsePDL);
  if constexpr (kSplitK > 1) {
    launch.enable_cluster(dim3(kSplitK, 1, 1));
  }
  launch(kernel, output, weight, act, num_tokens);
}

// Clusters (CTAs when kSplitK is 1) of invokeFusedAGemmWide's launch that fit on the current
// device at once; one wave needs kHdOut / kTileM. A cluster's CTAs must share a GPC.
template <int kHdIn, int kHdOut, int kTileM, int kTileN, int kTileK, int kSplitK, int kMmaWarps>
int fusedAGemmWideMaxActiveClusters() {
  using Tiling = FusedAWideTiling<kHdIn, kHdOut, kTileM, kTileN, kTileK, kSplitK>;
  auto kernel =
      fused_a_gemm_wide_kernel<kHdOut, kHdIn, kTileM, kTileN, kTileK, Tiling::stages, kSplitK, kMmaWarps>;
  host::RuntimeDeviceCheck(
      cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, Tiling::smem_bytes));
  constexpr int kThreads = 128 + kMmaWarps * 32;
  if constexpr (kSplitK > 1) {
    cudaLaunchConfig_t config = {};
    config.gridDim = dim3(kHdOut / kTileM * kSplitK);
    config.blockDim = dim3(kThreads);
    config.dynamicSmemBytes = Tiling::smem_bytes;
    cudaLaunchAttribute attr = {};
    attr.id = cudaLaunchAttributeClusterDimension;
    attr.val.clusterDim.x = kSplitK;
    attr.val.clusterDim.y = 1;
    attr.val.clusterDim.z = 1;
    config.attrs = &attr;
    config.numAttrs = 1;
    int num_clusters = 0;
    host::RuntimeDeviceCheck(cudaOccupancyMaxActiveClusters(&num_clusters, kernel, &config));
    return num_clusters;
  } else {
    int device = 0;
    int num_sms = 0;
    int blocks_per_sm = 0;
    host::RuntimeDeviceCheck(cudaGetDevice(&device));
    host::RuntimeDeviceCheck(cudaDeviceGetAttribute(&num_sms, cudaDevAttrMultiProcessorCount, device));
    host::RuntimeDeviceCheck(
        cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocks_per_sm, kernel, kThreads, Tiling::smem_bytes));
    return blocks_per_sm * num_sms;
  }
}

// SGLang's DSV3FusedAGemmKernel contract for 1..48 rows on fused_a_gemm_wide_kernel; tile_n is the
// smallest of 24, 32 and 48 that covers the rows.
template <int kHdIn, int kHdOut, bool kUsePDL, int kTileM, int kSplitK, int kTileK, int kMmaWarps = 4>
struct DSV3FusedAGemmWideKernel {
  static_assert(kHdOut % kTileM == 0, "hd_out must be a multiple of tile_m");

  static void
  run(const tvm::ffi::TensorView mat_a, const tvm::ffi::TensorView mat_b, const tvm::ffi::TensorView output) {
    using namespace host;

    auto M = SymbolicSize{"num_tokens"};
    auto K = SymbolicSize{"hd_in"};
    auto N = SymbolicSize{"hd_out"};
    auto device = SymbolicDevice{};
    K.set_value(kHdIn);
    N.set_value(kHdOut);
    device.set_options<kDLCUDA>();

    TensorMatcher({M, K}).with_dtype<bf16_t>().with_device(device).verify(mat_a);
    TensorMatcher({K, N}).with_dtype<bf16_t>().with_device(device).with_strides({1, kHdIn}).verify(mat_b);
    TensorMatcher({M, N}).with_dtype<bf16_t>().with_device(device).verify(output);

    const int num_tokens = static_cast<int>(M.unwrap());
    RuntimeCheck(
        num_tokens >= 1 && num_tokens <= kFusedAWideMaxRows,
        "glm_ll fused_a_wide: num_tokens must be in [1, ",
        kFusedAWideMaxRows,
        "], got ",
        num_tokens);

    const DLDevice dev = device.unwrap();
    auto* out_ptr = static_cast<bf16_t*>(output.data_ptr());
    auto* a_ptr = static_cast<bf16_t const*>(mat_a.data_ptr());
    auto* b_ptr = static_cast<bf16_t const*>(mat_b.data_ptr());
    if (num_tokens <= 24) {
      invokeFusedAGemmWide<kHdIn, kHdOut, kTileM, 24, kTileK, kSplitK, kMmaWarps, kUsePDL>(
          out_ptr, a_ptr, b_ptr, num_tokens, dev);
    } else if (num_tokens <= 32) {
      invokeFusedAGemmWide<kHdIn, kHdOut, kTileM, 32, kTileK, kSplitK, kMmaWarps, kUsePDL>(
          out_ptr, a_ptr, b_ptr, num_tokens, dev);
    } else {
      invokeFusedAGemmWide<kHdIn, kHdOut, kTileM, 48, kTileK, kSplitK, kMmaWarps, kUsePDL>(
          out_ptr, a_ptr, b_ptr, num_tokens, dev);
    }
  }

  // fusedAGemmWideMaxActiveClusters for the tile_n that run() picks for num_tokens.
  static int64_t max_active_clusters(int64_t num_tokens) {
    host::RuntimeCheck(
        num_tokens >= 1 && num_tokens <= kFusedAWideMaxRows,
        "glm_ll fused_a_wide: num_tokens must be in [1, ",
        kFusedAWideMaxRows,
        "], got ",
        num_tokens);
    if (num_tokens <= 24) {
      return fusedAGemmWideMaxActiveClusters<kHdIn, kHdOut, kTileM, 24, kTileK, kSplitK, kMmaWarps>();
    } else if (num_tokens <= 32) {
      return fusedAGemmWideMaxActiveClusters<kHdIn, kHdOut, kTileM, 32, kTileK, kSplitK, kMmaWarps>();
    }
    return fusedAGemmWideMaxActiveClusters<kHdIn, kHdOut, kTileM, 48, kTileK, kSplitK, kMmaWarps>();
  }
};

}  // namespace sglang::glm_ll
