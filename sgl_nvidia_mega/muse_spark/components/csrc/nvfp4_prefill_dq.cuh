// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
/**
 * \brief Dequantize the native NVFP4 KV pages of one prefill batch into an FP8 (E4M3) workspace, so the stock FP8-KV
 * context kernel can serve prefill over an NVFP4 KV cache.
 *
 * The E2M1-KV context kernel converts every K/V tile once per query tile that reads it. Converting each page once per
 * layer here and running the FP8-KV kernel over the workspace removes that repeated conversion; this kernel is a single
 * memory-bound pass over the batch's pages.
 *
 * Shapes (per call): packed pools viewed [pages, H, page, head_dim / 2] uint8 (any page / head / token strides, bytes of
 * a token row contiguous), element 2i of a row in the low nibble of byte i; scale pools [pages, H, page, head_dim / 16]
 * E4M3 (token and block dims contiguous; V scales four-token interleaved inside a page: the scale of token t, block i
 * sits at token (t & ~3) + i, column t & 3, as trtllm-gen reads them); block_tables [B, >= pages of each request] int32;
 * seq_lens [B] int32 (KV length of each request, cached prefix plus the tokens of this pass). Workspace: K and V viewed
 * [workspace pages, H, page, head_dim] E4M3 (any page / head / token strides, head_dim contiguous) and ws_table [B, >=
 * pages of each request] int32. Request b's pages are written to workspace pages off(b) + j, off(b) = sum over b' < b
 * of ceil(seq_lens[b'] / page), and ws_table[b][j] = off(b) + j. The caller sizes the workspace for the batch.
 *
 * Arithmetic: e4m3 = cvt.rn.satfinite(f16(e2m1) * f16(block scale)); the product is exact in f16, so this is one
 * round-to-nearest-even rounding of the exact dequantized value (the plugin verify kernel's dequant8). Global K/V scales
 * are not applied here (the caller keeps them in bmm1 / bmm2 scales, as the stock NVFP4 path does). Rows of the last
 * page past a request's length are converted as stored (the pools hold only finite values; the attention kernel masks
 * those keys).
 *
 * Work split: grid (max pages per request, B, 2), one block per (request, page, K or V). The block stages the page's
 * scale tile ([H][page][4] bytes) in shared memory, then each thread loads all of its 16-byte packed items (32 values =
 * two scale blocks of one token-head row) before converting and storing them (2 x 16 bytes each); consecutive threads
 * cover consecutive bytes of the token rows. Requests past their last page exit at once. With use_pdl the kernel reads
 * its inputs only after griddepcontrol.wait and signals dependents after its stores.
 */

#include <sgl_kernel/tensor.h>
#include <sgl_kernel/utils.h>

#include <sgl_kernel/runtime.cuh>
#include <sgl_kernel/utils.cuh>

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cstdint>

namespace muse_vr {
namespace npd {

constexpr int kThreads = 256;
constexpr int kHeadDim = 64;
constexpr int kBlocks = kHeadDim / 16;  // scale blocks per head row
constexpr int kItems = 4;               // 16-byte items per thread per launch step (page 128, up to 4 KV heads)

struct Params {
  const uint8_t* k4;
  const uint8_t* v4;
  const uint8_t* ksf;
  const uint8_t* vsf;
  const int32_t* block_tables;
  const int32_t* seq_lens;
  uint8_t* k8;
  uint8_t* v8;
  int32_t* ws_table;
  // element strides (bytes, all pools are 1-byte elements)
  int64_t k4_page, k4_head, k4_tok, v4_page, v4_head, v4_tok;
  int64_t ksf_page, ksf_head, vsf_page, vsf_head;
  int64_t k8_page, k8_head, k8_tok, v8_page, v8_head, v8_tok;
  int64_t bt_stride, ws_stride;
  int32_t batch, num_heads, page_size, ws_pages;
};

// Four packed bytes = 8 E2M1 values (element 2i in the low nibble of byte i) times one E4M3 block scale -> 8 E4M3
// values (two words, element order kept). f16(e2m1) * f16(scale) is exact; one rounding to e4m3.
__device__ __forceinline__ uint2 dequant8(uint32_t packed, uint32_t scale2_f16x2) {
  uint32_t lo, hi;
  asm("{\n"
      ".reg .b8 b0, b1, b2, b3;\n"
      ".reg .b32 h0, h1, h2, h3;\n"
      ".reg .b16 e0, e1, e2, e3;\n"
      "mov.b32 {b0, b1, b2, b3}, %2;\n"
      "cvt.rn.f16x2.e2m1x2 h0, b0;\n"
      "cvt.rn.f16x2.e2m1x2 h1, b1;\n"
      "cvt.rn.f16x2.e2m1x2 h2, b2;\n"
      "cvt.rn.f16x2.e2m1x2 h3, b3;\n"
      "mul.rn.f16x2 h0, h0, %3;\n"
      "mul.rn.f16x2 h1, h1, %3;\n"
      "mul.rn.f16x2 h2, h2, %3;\n"
      "mul.rn.f16x2 h3, h3, %3;\n"
      "cvt.rn.satfinite.e4m3x2.f16x2 e0, h0;\n"
      "cvt.rn.satfinite.e4m3x2.f16x2 e1, h1;\n"
      "cvt.rn.satfinite.e4m3x2.f16x2 e2, h2;\n"
      "cvt.rn.satfinite.e4m3x2.f16x2 e3, h3;\n"
      "mov.b32 %0, {e0, e1};\n"
      "mov.b32 %1, {e2, e3};\n"
      "}\n"
      : "=r"(lo), "=r"(hi)
      : "r"(packed), "r"(scale2_f16x2));
  return make_uint2(lo, hi);
}

// One E4M3 byte -> the f16 value in both halves of an f16x2 word (exact).
__device__ __forceinline__ uint32_t scale_f16x2(uint32_t e4m3_byte) {
  uint32_t d;
  const uint16_t pair = static_cast<uint16_t>(e4m3_byte | (e4m3_byte << 8));
  asm("cvt.rn.f16x2.e4m3x2 %0, %1;\n" : "=r"(d) : "h"(pair));
  return d;
}

template <bool kUsePDL>
__global__ void __launch_bounds__(kThreads) nvfp4_prefill_dq_kernel(const Params p) {
  extern __shared__ __align__(16) uint8_t s_sf[];  // the page's scale tile of this tensor: [H][page][4]
  __shared__ int32_t s_part[kThreads / 32];
  const int b = blockIdx.y;
  const int j = blockIdx.x;
  const int kv = blockIdx.z;
  const int tid = threadIdx.x;
  if constexpr (kUsePDL) asm volatile("griddepcontrol.wait;\n" ::: "memory");

  const int ps = p.page_size;
  const int npages = (p.seq_lens[b] + ps - 1) / ps;
  if (j < npages) {
    const int64_t src = p.block_tables[static_cast<int64_t>(b) * p.bt_stride + j];
    // workspace offset of request b: pages of the requests before it
    int part = 0;
    for (int i = tid; i < b; i += kThreads) part += (p.seq_lens[i] + ps - 1) / ps;
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) part += __shfl_xor_sync(0xffffffffu, part, o);
    if ((tid & 31) == 0) s_part[tid >> 5] = part;

    const int H = p.num_heads;
    // stage the scale tile: H rows of page * 4 contiguous bytes (16-byte chunks)
    const uint8_t* sf = kv ? p.vsf + src * p.vsf_page : p.ksf + src * p.ksf_page;
    const int64_t sf_head = kv ? p.vsf_head : p.ksf_head;
    const int chunks_per_head = ps * kBlocks / 16;
    for (int c = tid; c < H * chunks_per_head; c += kThreads) {
      const int h = c / chunks_per_head;
      const int cc = c - h * chunks_per_head;
      *reinterpret_cast<uint4*>(s_sf + (h * ps * kBlocks) + cc * 16) =
          *reinterpret_cast<const uint4*>(sf + h * sf_head + cc * 16);
    }
    // packed rows: one item = 16 bytes = 32 values (two scale blocks) of one (token, head) row half
    const uint8_t* in = kv ? p.v4 + src * p.v4_page : p.k4 + src * p.k4_page;
    const int64_t in_head = kv ? p.v4_head : p.k4_head, in_tok = kv ? p.v4_tok : p.k4_tok;
    const int items = ps * H * 2;
    uint4 raw[kItems];
#pragma unroll
    for (int u = 0; u < kItems; ++u) {
      const int it = tid + u * kThreads;
      raw[u] = make_uint4(0u, 0u, 0u, 0u);
      if (it < items) {
        const int t = it / (H * 2), rh = it - t * (H * 2), h = rh >> 1, half = rh & 1;
        raw[u] = *reinterpret_cast<const uint4*>(in + h * in_head + t * in_tok + half * 16);
      }
    }
    __syncthreads();  // scale tile + offset parts visible
    int off = 0;
#pragma unroll
    for (int w = 0; w < kThreads / 32; ++w) off += s_part[w];
    const int64_t dst = off + j;
    if (dst >= p.ws_pages) return;  // never past the workspace (the host sizes it from the same lengths)
    if (kv == 0 && tid == 0) p.ws_table[static_cast<int64_t>(b) * p.ws_stride + j] = static_cast<int32_t>(dst);

    uint8_t* out = kv ? p.v8 + dst * p.v8_page : p.k8 + dst * p.k8_page;
    const int64_t out_head = kv ? p.v8_head : p.k8_head, out_tok = kv ? p.v8_tok : p.k8_tok;
#pragma unroll
    for (int u = 0; u < kItems; ++u) {
      const int it = tid + u * kThreads;
      if (it < items) {
        const int t = it / (H * 2), rh = it - t * (H * 2), h = rh >> 1, half = rh & 1;
        const int blk0 = half * 2;
        uint32_t sb0, sb1;
        if (kv == 0) {
          sb0 = s_sf[(h * ps + t) * kBlocks + blk0];
          sb1 = s_sf[(h * ps + t) * kBlocks + blk0 + 1];
        } else {  // four-token interleave: block i of token t at token (t & ~3) + i, column t & 3
          sb0 = s_sf[(h * ps + (t & ~3) + blk0) * kBlocks + (t & 3)];
          sb1 = s_sf[(h * ps + (t & ~3) + blk0 + 1) * kBlocks + (t & 3)];
        }
        const uint32_t s0 = scale_f16x2(sb0), s1 = scale_f16x2(sb1);
        const uint2 a = dequant8(raw[u].x, s0), c = dequant8(raw[u].y, s0);
        const uint2 d = dequant8(raw[u].z, s1), e = dequant8(raw[u].w, s1);
        uint4* o = reinterpret_cast<uint4*>(out + h * out_head + t * out_tok + half * 32);
        o[0] = make_uint4(a.x, a.y, c.x, c.y);
        o[1] = make_uint4(d.x, d.y, e.x, e.y);
      }
    }
  }
  if constexpr (kUsePDL) asm volatile("griddepcontrol.launch_dependents;\n" ::: "memory");
}

}  // namespace npd

template <bool kUsePDL>
struct Nvfp4PrefillDqKernel {
  static void run(const tvm::ffi::TensorView k4,        // [pages, H, page, 32] uint8 view
                  const tvm::ffi::TensorView v4,        // same
                  const tvm::ffi::TensorView ksf,       // [pages, H, page, 4] e4m3 (1-byte)
                  const tvm::ffi::TensorView vsf,       // same, four-token interleaved
                  const tvm::ffi::TensorView block_tables,  // [B, >= max pages] int32
                  const tvm::ffi::TensorView seq_lens,  // [B] int32
                  const tvm::ffi::TensorView k8,        // [ws pages, H, page, 64] e4m3 (1-byte) view
                  const tvm::ffi::TensorView v8,        // same
                  const tvm::ffi::TensorView ws_table,  // [B, >= max pages] int32
                  int64_t max_pages,                    // max pages of one request (host, from the KV lengths)
                  int64_t total_pages) {                // pages of the batch (host); the workspace holds them
    using namespace sglang::host;
    const auto is_i32 = [](const tvm::ffi::TensorView& t) {
      return t.dtype().code == DLDataTypeCode::kDLInt && t.dtype().bits == 32;
    };
    RuntimeCheck(k4.dim() == 4 && v4.dim() == 4 && ksf.dim() == 4 && vsf.dim() == 4 && k8.dim() == 4 && v8.dim() == 4,
                 "pools must be 4-D [pages, heads, page, row] views");
    const int64_t H = k4.size(1), ps = k4.size(2);
    RuntimeCheck(k4.dtype().bits == 8 && v4.dtype().bits == 8 && ksf.dtype().bits == 8 && vsf.dtype().bits == 8 &&
                     k8.dtype().bits == 8 && v8.dtype().bits == 8,
                 "pools must have 1-byte elements");
    RuntimeCheck(k4.size(3) == npd::kHeadDim / 2 && v4.size(3) == npd::kHeadDim / 2 && k4.stride(3) == 1 &&
                     v4.stride(3) == 1 && v4.size(1) == H && v4.size(2) == ps,
                 "packed pools: head_dim 64 rows, bytes contiguous");
    RuntimeCheck(ksf.size(1) == H && ksf.size(2) == ps && ksf.size(3) == npd::kBlocks && ksf.stride(3) == 1 &&
                     ksf.stride(2) == npd::kBlocks && vsf.size(1) == H && vsf.size(2) == ps &&
                     vsf.size(3) == npd::kBlocks && vsf.stride(3) == 1 && vsf.stride(2) == npd::kBlocks,
                 "scale pools: [pages, H, page, 4], token and block dims contiguous");
    RuntimeCheck(k8.size(1) == H && k8.size(2) == ps && k8.size(3) == npd::kHeadDim && k8.stride(3) == 1 &&
                     v8.size(1) == H && v8.size(2) == ps && v8.size(3) == npd::kHeadDim && v8.stride(3) == 1,
                 "workspace: [pages, H, page, 64], head_dim contiguous");
    RuntimeCheck(ps % 4 == 0 && ps * H * 2 <= npd::kItems * npd::kThreads, "page size x heads too large for one block");
    RuntimeCheck(ksf.stride(1) % 16 == 0 && vsf.stride(1) % 16 == 0 && ksf.stride(0) % 16 == 0 && vsf.stride(0) % 16 == 0 &&
                     reinterpret_cast<uintptr_t>(ksf.data_ptr()) % 16 == 0 &&
                     reinterpret_cast<uintptr_t>(vsf.data_ptr()) % 16 == 0,
                 "scale tiles must be 16-byte aligned");
    const auto aligned = [](const tvm::ffi::TensorView& t, int64_t a) {
      return reinterpret_cast<uintptr_t>(t.data_ptr()) % a == 0;
    };
    RuntimeCheck(aligned(k4, 16) && aligned(v4, 16) && k4.stride(2) % 16 == 0 && k4.stride(1) % 16 == 0 &&
                     k4.stride(0) % 16 == 0 && v4.stride(2) % 16 == 0 && v4.stride(1) % 16 == 0 &&
                     v4.stride(0) % 16 == 0,
                 "packed rows must be 16-byte aligned");
    RuntimeCheck(aligned(k8, 16) && aligned(v8, 16) && k8.stride(2) % 16 == 0 && k8.stride(1) % 16 == 0 &&
                     k8.stride(0) % 16 == 0 && v8.stride(2) % 16 == 0 && v8.stride(1) % 16 == 0 &&
                     v8.stride(0) % 16 == 0,
                 "workspace rows must be 16-byte aligned");
    RuntimeCheck(is_i32(block_tables) && is_i32(seq_lens) && is_i32(ws_table) && block_tables.dim() == 2 &&
                     ws_table.dim() == 2 && seq_lens.dim() == 1 && block_tables.stride(1) == 1 &&
                     ws_table.stride(1) == 1 && seq_lens.stride(0) == 1,
                 "tables must be int32, rows contiguous");
    const int64_t B = seq_lens.size(0);
    RuntimeCheck(block_tables.size(0) >= B && ws_table.size(0) >= B && block_tables.size(1) >= max_pages &&
                     ws_table.size(1) >= max_pages,
                 "tables too small");
    RuntimeCheck(k8.size(0) >= total_pages && v8.size(0) >= total_pages, "workspace too small");
    if (B == 0 || max_pages == 0 || total_pages == 0) return;
    npd::Params p{};
    p.k4 = static_cast<const uint8_t*>(k4.data_ptr());
    p.v4 = static_cast<const uint8_t*>(v4.data_ptr());
    p.ksf = static_cast<const uint8_t*>(ksf.data_ptr());
    p.vsf = static_cast<const uint8_t*>(vsf.data_ptr());
    p.block_tables = static_cast<const int32_t*>(block_tables.data_ptr());
    p.seq_lens = static_cast<const int32_t*>(seq_lens.data_ptr());
    p.k8 = static_cast<uint8_t*>(k8.data_ptr());
    p.v8 = static_cast<uint8_t*>(v8.data_ptr());
    p.ws_table = static_cast<int32_t*>(ws_table.data_ptr());
    p.k4_page = k4.stride(0);
    p.k4_head = k4.stride(1);
    p.k4_tok = k4.stride(2);
    p.v4_page = v4.stride(0);
    p.v4_head = v4.stride(1);
    p.v4_tok = v4.stride(2);
    p.ksf_page = ksf.stride(0);
    p.ksf_head = ksf.stride(1);
    p.vsf_page = vsf.stride(0);
    p.vsf_head = vsf.stride(1);
    p.k8_page = k8.stride(0);
    p.k8_head = k8.stride(1);
    p.k8_tok = k8.stride(2);
    p.v8_page = v8.stride(0);
    p.v8_head = v8.stride(1);
    p.v8_tok = v8.stride(2);
    p.bt_stride = block_tables.stride(0);
    p.ws_stride = ws_table.stride(0);
    p.batch = static_cast<int32_t>(B);
    p.num_heads = static_cast<int32_t>(H);
    p.page_size = static_cast<int32_t>(ps);
    p.ws_pages = static_cast<int32_t>(k8.size(0) < v8.size(0) ? k8.size(0) : v8.size(0));
    LaunchKernel(dim3(static_cast<unsigned>(max_pages), static_cast<unsigned>(B), 2u), npd::kThreads, k4.device(),
                 static_cast<size_t>(H * ps * npd::kBlocks))
        .config({.use_pdl = kUsePDL})(npd::nvfp4_prefill_dq_kernel<kUsePDL>, p);
  }
};

}  // namespace muse_vr
