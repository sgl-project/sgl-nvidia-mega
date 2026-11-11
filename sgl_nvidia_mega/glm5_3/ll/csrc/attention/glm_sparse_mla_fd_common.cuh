// Sparse MLA decode as flash decoding over the top-k gathered KV rows, for GLM-5 DSA
// layers with the raw 576 B FP8 KV row of the `trtllm` DSA decode backend: 8 or 16
// query heads per rank, top-k 2048. Shared constants and PTX helpers of the one-tile
// kernel (glm_sparse_mla_fd_one_tile.cuh) and the multi-tile mode
// (glm_sparse_mla_fd_multi_tile.cuh).
//
// Adapted from TensorRT-LLM (Apache-2.0):
//   cpp/tensorrt_llm/kernels/sparseMlaFlashDecode.cu (Dongfeng Yu; 2808056e, 9bea245e, f5052d41)
//   cpp/tensorrt_llm/kernels/glmLowLatency/w4/sparseMlaDecode.cu (Xiao Wang; GB300 hardening)
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <sgl_kernel/tensor.h>
#include <sgl_kernel/utils.h>

#include <sgl_kernel/utils.cuh>

#include <dlpack/dlpack.h>
#include <tvm/ffi/container/tensor.h>
#include <tvm/ffi/extra/cuda/device_guard.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <map>
#include <mutex>
#include <type_traits>

namespace sglang {

namespace glm_ll_sparse_mla_fd {

// Work split: query row r is one cluster of kSplits CTAs; CTA s owns index-list
// positions [128 s, 128 s + 128). Each CTA pushes its unnormalized partial output
// and softmax statistics into the shared memory of the CTA that finalizes those
// output dims (st.async completing on that CTA's mbarrier).
inline constexpr int kTopK = 2048;
inline constexpr int kHeadDimQk = 576;
inline constexpr int kHeadDimV = 512;
inline constexpr int kKeysPerCta = 128;
inline constexpr int kSplits = kTopK / kKeysPerCta;
inline constexpr int kWarps = 8;
inline constexpr int kThreads = kWarps * 32;
inline constexpr int kKeysPerWarp = kKeysPerCta / kWarps;
inline constexpr int kDimsPerWarp = kHeadDimV / kWarps;
inline constexpr int kDimsPerCta = kHeadDimV / kSplits;
inline constexpr int kQkSteps = kHeadDimQk / 32;
inline constexpr int kPvSteps = kKeysPerCta / 32;
inline constexpr int kPvTilesPerWarp = kDimsPerWarp / 16;
// P enters P.V as e4m3(448 * p), the rounding of the TRTLLM-Gen generation kernels.
inline constexpr float kProbScale = 448.f;

// Padded shared pitches, bank-conflict free: K rows shift by 20 banks (the eight
// rows of an ldmatrix phase hit disjoint 16 B bank groups); the 64-bit query
// fragment loads of rows g = 0..3 cover banks 8g..8g+7; the 32-bit probability
// fragment loads of rows g = 0..7 cover banks 4g..4g+3.
inline constexpr int kKRowBytes = kHeadDimQk + 16;
inline constexpr int kQRowHalves = kHeadDimQk + 16;
inline constexpr int kPRowHalves = kKeysPerCta + 8;

static_assert(kSplits == 16 && kKeysPerWarp == 16, "one m16 key tile per warp, 16-CTA clusters");
static_assert(kDimsPerCta == 32 && kDimsPerWarp == 2 * kDimsPerCta, "each P.V warp feeds two merge CTAs");
static_assert(
    kKRowBytes / 4 % 32 == 20 && kQRowHalves / 2 % 32 == 8 && kPRowHalves / 2 % 32 == 4, "bank-conflict-free pitches");

#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)

// Raw PTX: cluster-scope mbarrier waits, DSMEM st.async and bulk copies have no
// sgl_kernel wrapper; the generic `ptx::mbar_*` helpers lack the clobbers these
// ordering points need.
SGL_DEVICE uint32_t smem_addr(const void* ptr) {
  return static_cast<uint32_t>(__cvta_generic_to_shared(ptr));
}

SGL_DEVICE void mbar_init(uint32_t bar, uint32_t count) {
  asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;\n" ::"r"(bar), "r"(count) : "memory");
}

SGL_DEVICE void fence_mbar_init() {
  asm volatile("fence.mbarrier_init.release.cluster;\n" ::: "memory");
}

SGL_DEVICE void mbar_arrive_expect_tx(uint32_t bar, uint32_t bytes) {
  asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;\n" ::"r"(bar), "r"(bytes) : "memory");
}

SGL_DEVICE void mbar_arrive(uint32_t bar) {
  asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];\n" ::"r"(bar) : "memory");
}

SGL_DEVICE void mbar_wait(uint32_t bar, uint32_t parity) {
  asm volatile(
      "{\n"
      ".reg .pred done;\n"
      "GLM_FD_WAIT_%=:\n"
      "mbarrier.try_wait.parity.shared::cta.b64 done, [%0], %1;\n"
      "@!done bra GLM_FD_WAIT_%=;\n"
      "}\n" ::"r"(bar),
      "r"(parity)
      : "memory");
}

// Cluster-scope acquire: the bytes this barrier counts come from peers' st.async.
SGL_DEVICE void mbar_wait_cluster(uint32_t bar, uint32_t parity) {
  asm volatile(
      "{\n"
      ".reg .pred done;\n"
      "GLM_FD_WAIT_CLUSTER_%=:\n"
      "mbarrier.try_wait.parity.acquire.cluster.shared::cta.b64 done, [%0], %1;\n"
      "@!done bra GLM_FD_WAIT_CLUSTER_%=;\n"
      "}\n" ::"r"(bar),
      "r"(parity)
      : "memory");
}

SGL_DEVICE void bulk_copy_g2s(uint32_t dst, const void* src, uint32_t bytes, uint32_t bar) {
  asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];\n" ::"r"(dst),
               "l"(src),
               "r"(bytes),
               "r"(bar)
               : "memory");
}

SGL_DEVICE void cluster_arrive_relaxed() {
  asm volatile("barrier.cluster.arrive.relaxed.aligned;\n" ::: "memory");
}

SGL_DEVICE void cluster_wait() {
  asm volatile("barrier.cluster.wait.acquire.aligned;\n" ::: "memory");
}

SGL_DEVICE uint32_t map_to_rank(uint32_t local_addr, uint32_t rank) {
  uint32_t remote;
  asm volatile("mapa.shared::cluster.u32 %0, %1, %2;\n" : "=r"(remote) : "r"(local_addr), "r"(rank));
  return remote;
}

SGL_DEVICE void store_async(uint32_t remote_addr, float value, uint32_t remote_bar) {
  asm volatile("st.async.shared::cluster.mbarrier::complete_tx::bytes.f32 [%0], %1, [%2];\n" ::"r"(remote_addr),
               "f"(value),
               "r"(remote_bar)
               : "memory");
}

SGL_DEVICE void store_async2(uint32_t remote_addr, float a, float b, uint32_t remote_bar) {
  asm volatile(
      "st.async.shared::cluster.mbarrier::complete_tx::bytes.v2.f32 [%0], {%1, %2}, [%3];\n" ::"r"(remote_addr),
      "f"(a),
      "f"(b),
      "r"(remote_bar)
      : "memory");
}

SGL_DEVICE void store_async_v4(uint32_t remote_addr, const uint32_t (&v)[4], uint32_t remote_bar) {
  asm volatile(
      "st.async.shared::cluster.mbarrier::complete_tx::bytes.v4.b32 [%0], {%1, %2, %3, %4}, [%5];\n" ::"r"(
          remote_addr),
      "r"(v[0]),
      "r"(v[1]),
      "r"(v[2]),
      "r"(v[3]),
      "r"(remote_bar)
      : "memory");
}

// Coherent (not .nc) loads: the kernel may be resident while its PDL primary
// still writes these buffers, so the read-only path could serve stale lines.
SGL_DEVICE uint4 ld_global_v4(const void* ptr) {
  uint4 v;
  asm volatile("ld.global.v4.u32 {%0, %1, %2, %3}, [%4];\n" : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w) : "l"(ptr));
  return v;
}

SGL_DEVICE int32_t ld_global_s32(const int32_t* ptr) {
  int32_t v;
  asm volatile("ld.global.s32 %0, [%1];\n" : "=r"(v) : "l"(ptr));
  return v;
}

SGL_DEVICE void ldmatrix_x4(uint32_t (&r)[4], uint32_t addr) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
               : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3])
               : "r"(addr));
}

SGL_DEVICE void ldmatrix_x4_trans(uint32_t (&r)[4], uint32_t addr) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0, %1, %2, %3}, [%4];\n"
               : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3])
               : "r"(addr));
}

// D += A * B, A 16x16 f16 (row), B 16x8 f16 (col), FP32 accumulation.
SGL_DEVICE void mma_f16(float (&d)[4], uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3, uint32_t b0, uint32_t b1) {
  asm volatile(
      "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, "
      "{%0, %1, %2, %3};\n"
      : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
      : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}

// D += A * B, A 16x16 bf16 (row), B 16x8 bf16 (col), FP32 accumulation.
SGL_DEVICE void mma_bf16(float (&d)[4], uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3, uint32_t b0, uint32_t b1) {
  asm volatile(
      "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, "
      "{%0, %1, %2, %3};\n"
      : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
      : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}

// Orders this thread's generic-proxy accesses of shared memory before later bulk copies into it.
SGL_DEVICE void fence_proxy_async_smem() {
  asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory");
}

SGL_DEVICE float fast_exp2(float x) {
  float y;
  asm volatile("ex2.approx.ftz.f32 %0, %1;\n" : "=f"(y) : "f"(x));
  return y;
}

// Two e4m3 values (low byte first, low 16 bits of `pair`) as two FP16 values; exact.
SGL_DEVICE uint32_t e4m3x2_to_half2(uint32_t pair) {
  const __half2_raw h = __nv_cvt_fp8x2_to_halfraw2(static_cast<__nv_fp8x2_storage_t>(pair & 0xffffU), __NV_E4M3);
  return static_cast<uint32_t>(h.x) | (static_cast<uint32_t>(h.y) << 16);
}

// e4m3(448 * p) as an exact FP16 value.
SGL_DEVICE __half round_probability(float p) {
  const __nv_fp8_storage_t q = __nv_cvt_float_to_fp8(p * kProbScale, __NV_SATFINITE, __NV_E4M3);
  return __half(__nv_cvt_fp8_to_halfraw(q, __NV_E4M3));
}

#endif  // defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)

}  // namespace glm_ll_sparse_mla_fd

}  // namespace sglang
