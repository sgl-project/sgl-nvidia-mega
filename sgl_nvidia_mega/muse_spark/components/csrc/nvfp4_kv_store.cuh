// SPDX-License-Identifier: Apache-2.0
/*
 * Copyright (c) 2025 by FlashInfer team.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
// Derived from FlashInfer (https://github.com/flashinfer-ai/flashinfer) csrc/fp4_kv_quantization.cu as shipped in
// flashinfer-python 0.7.1 (flashinfer/data/csrc/fp4_kv_quantization.cu); see ../include/LICENSE.flashinfer.
// Modified by NVIDIA: FlashInfer's per-block NVFP4 quantization is kept (the rcp.approx, bf16 abs / max and E2M1
// packing helpers and the E4M3 block-scale and output-scale arithmetic of quantize_fp16_to_e2m1_with_scaling); the
// kernel around it quantizes one layer's K and V rows read from strided views and scatters the packed rows and block
// scales into SGLang's native NVFP4 pool layout in one launch, with optional programmatic dependent launch.
//
// NVFP4 KV write in one launch: quantize one layer's K and V rows (bf16, token rows may be strided views of the
// projection output) to E2M1 with one E4M3 scale per 16 values, and scatter the packed rows and the block scales into
// SGLang's native NVFP4 pool layout (packed rows [slots, heads, head_dim / 2]; scales [pages, heads, page, head_dim /
// 16], V scales four-token interleaved inside the page as trtllm-gen reads them).
//
// The stock path runs FlashInfer's nvfp4_kv_quantize for K, a copy (.contiguous()) and the same quantization for V,
// then SGLang's scatter kernel: four launches per layer. The per-block arithmetic here is the one of FlashInfer's
// fp4_kv_quantization.cu (Apache-2.0): amax of the 16 values, sf = rcp(gs) * (amax * rcp(6)) rounded to E4M3,
// out_scale = rcp(sf * gs) (0 when amax == 0), values * out_scale rounded to E2M1 (rcp = rcp.approx.ftz.f32), and the
// module is compiled with -use_fast_math as FlashInfer compiles its kernel, so the bytes written are the stock bytes
// (checked bit for bit in tests/test_nvfp4_kv_store.py).
//
// Work split: one thread per 8 values (16 input bytes -> 4 output bytes); the two threads of a 16-value block are
// neighbouring lanes and exchange their maxima with one shuffle (as the stock kernel does). The even thread writes the
// block scale. Items past the last token run the shuffle and store nothing.

#include <sgl_kernel/tensor.h>
#include <sgl_kernel/utils.h>

#include <cuda_bf16.h>
#include <cuda_fp8.h>

#include <cstdint>

namespace muse_vr {
namespace nks {

constexpr int kThreads = 256;

struct Params {
  const __nv_bfloat16* k;
  const __nv_bfloat16* v;
  const void* loc;
  const float* k_gs;
  const float* v_gs;
  uint8_t* k_out;
  uint8_t* v_out;
  uint8_t* k_sf;
  uint8_t* v_sf;
  int64_t k_tok_stride, k_head_stride, v_tok_stride, v_head_stride, loc_stride;
  int32_t num_tokens, num_heads, head_dim, page_size;
};

__device__ __forceinline__ float rcp_approx_ftz(float a) {
  float b;
  asm volatile("rcp.approx.ftz.f32 %0, %1;\n" : "=f"(b) : "f"(a));
  return b;
}

__device__ __forceinline__ __nv_bfloat162 abs_bf162(__nv_bfloat162 a) {
  __nv_bfloat162 r;
  r.x = __float2bfloat16(fabsf(__bfloat162float(a.x)));
  r.y = __float2bfloat16(fabsf(__bfloat162float(a.y)));
  return r;
}

__device__ __forceinline__ __nv_bfloat162 max_bf162(__nv_bfloat162 a, __nv_bfloat162 b) {
  __nv_bfloat162 r;
  r.x = __bfloat162float(a.x) > __bfloat162float(b.x) ? a.x : b.x;
  r.y = __bfloat162float(a.y) > __bfloat162float(b.y) ? a.y : b.y;
  return r;
}

// 8 floats -> 8 E2M1 values in one word, value 2i in the low nibble of byte i.
__device__ __forceinline__ uint32_t to_e2m1x8(const float2 (&f)[4]) {
  uint32_t val;
  asm volatile(
      "{\n"
      ".reg .b8 b0, b1, b2, b3;\n"
      "cvt.rn.satfinite.e2m1x2.f32 b0, %2, %1;\n"
      "cvt.rn.satfinite.e2m1x2.f32 b1, %4, %3;\n"
      "cvt.rn.satfinite.e2m1x2.f32 b2, %6, %5;\n"
      "cvt.rn.satfinite.e2m1x2.f32 b3, %8, %7;\n"
      "mov.b32 %0, {b0, b1, b2, b3};\n"
      "}\n"
      : "=r"(val)
      : "f"(f[0].x), "f"(f[0].y), "f"(f[1].x), "f"(f[1].y), "f"(f[2].x), "f"(f[2].y), "f"(f[3].x), "f"(f[3].y));
  return val;
}

template <bool kLoc64, bool kUsePDL>
__global__ void __launch_bounds__(kThreads) nvfp4_kv_store_kernel(const Params p) {
  const int per_head = p.head_dim / 8;  // 8-value items per head
  const int per_kv = p.num_heads * per_head;
  const int per_tok = 2 * per_kv;
  const int64_t item = static_cast<int64_t>(blockIdx.x) * kThreads + threadIdx.x;
  const int64_t tok = item / per_tok;
  const int w = static_cast<int>(item - tok * per_tok);
  const int kv = w / per_kv, r = w - kv * per_kv, h = r / per_head, part = r - h * per_head;
  const bool valid = tok < p.num_tokens;

  if constexpr (kUsePDL) asm volatile("griddepcontrol.wait;\n" ::: "memory");
  uint4 raw = make_uint4(0u, 0u, 0u, 0u);
  float gs = 1.0f;
  int64_t slot = 0;
  if (valid) {
    const __nv_bfloat16* src = kv ? p.v + tok * p.v_tok_stride + h * p.v_head_stride
                                  : p.k + tok * p.k_tok_stride + h * p.k_head_stride;
    raw = *reinterpret_cast<const uint4*>(src + part * 8);
    gs = *(kv ? p.v_gs : p.k_gs);
    slot = kLoc64 ? static_cast<const int64_t*>(p.loc)[tok * p.loc_stride]
                  : static_cast<int64_t>(static_cast<const int32_t*>(p.loc)[tok * p.loc_stride]);
  }
  __nv_bfloat162 vec[4];
  vec[0] = *reinterpret_cast<__nv_bfloat162*>(&raw.x);
  vec[1] = *reinterpret_cast<__nv_bfloat162*>(&raw.y);
  vec[2] = *reinterpret_cast<__nv_bfloat162*>(&raw.z);
  vec[3] = *reinterpret_cast<__nv_bfloat162*>(&raw.w);

  __nv_bfloat162 m = abs_bf162(vec[0]);
#pragma unroll
  for (int i = 1; i < 4; ++i) m = max_bf162(m, abs_bf162(vec[i]));
  m = max_bf162(__shfl_xor_sync(0xffffffffu, m, 1), m);
  const float amax = __bfloat162float(__bfloat162float(m.x) > __bfloat162float(m.y) ? m.x : m.y);

  float sf = rcp_approx_ftz(gs) * (amax * rcp_approx_ftz(6.0f));
  const __nv_fp8_e4m3 sf8 = __nv_fp8_e4m3(sf);
  sf = static_cast<float>(sf8);
  const float out_scale = amax != 0 ? rcp_approx_ftz(sf * gs) : 0.0f;
  float2 f[4];
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    f[i] = __bfloat1622float2(vec[i]);
    f[i].x *= out_scale;
    f[i].y *= out_scale;
  }
  const uint32_t packed = to_e2m1x8(f);

  if (valid) {
    uint8_t* dst = (kv ? p.v_out : p.k_out) + (slot * p.num_heads + h) * (p.head_dim / 2) + part * 4;
    *reinterpret_cast<uint32_t*>(dst) = packed;
    if ((part & 1) == 0) {
      const int sd = p.head_dim / 16, blk = part >> 1;
      const int64_t page = slot / p.page_size;
      const int t = static_cast<int>(slot - page * p.page_size);
      const int64_t base = (page * p.num_heads + h) * p.page_size * sd;
      int64_t off;
      if (kv == 0) {
        off = base + static_cast<int64_t>(t) * sd + blk;
      } else {  // four-token interleave: token (t & ~3) + blk / g, column (blk % g) * 4 + t % 4, g = sd / 4
        const int g = sd / 4;
        off = base + static_cast<int64_t>((t / 4) * 4 + blk / g) * sd + (blk % g) * 4 + t % 4;
      }
      (kv ? p.v_sf : p.k_sf)[off] = sf8.__x;
    }
  }
  if constexpr (kUsePDL) asm volatile("griddepcontrol.launch_dependents;\n" ::: "memory");
}

}  // namespace nks

template <bool kLoc64, bool kUsePDL>
struct Nvfp4KvStoreKernel {
  static void run(const tvm::ffi::TensorView k,      // [N, H, hd] bf16; dims contiguous, rows/heads 16-byte aligned
                  const tvm::ffi::TensorView v,      // same shape
                  const tvm::ffi::TensorView loc,    // [N] int64 (kLoc64) or int32 slots
                  const tvm::ffi::TensorView k_gs,   // [1] fp32 global scale
                  const tvm::ffi::TensorView v_gs,   // [1] fp32
                  const tvm::ffi::TensorView k_out,  // [slots, H, hd / 2] bytes, contiguous
                  const tvm::ffi::TensorView v_out,  // same
                  const tvm::ffi::TensorView k_sf,   // [pages, H, page, hd / 16] bytes, contiguous
                  const tvm::ffi::TensorView v_sf,   // same
                  int64_t page_size) {
    using namespace sglang::host;
    const auto bf16_rows = [](const tvm::ffi::TensorView& t) {
      return t.dim() == 3 && t.dtype().code == DLDataTypeCode::kDLBfloat && t.dtype().bits == 16 && t.stride(2) == 1 &&
             t.stride(1) % 8 == 0 && t.stride(0) % 8 == 0 && reinterpret_cast<uintptr_t>(t.data_ptr()) % 16 == 0;
    };
    RuntimeCheck(bf16_rows(k) && bf16_rows(v), "k/v must be bf16 [tokens, heads, head_dim], dims contiguous, aligned");
    const int64_t n = k.size(0), hn = k.size(1), hd = k.size(2);
    RuntimeCheck(v.size(0) == n && v.size(1) == hn && v.size(2) == hd, "k/v shapes differ");
    RuntimeCheck(hd % 64 == 0, "head_dim must be a multiple of 64");
    RuntimeCheck(loc.dim() == 1 && loc.size(0) == n && loc.dtype().code == DLDataTypeCode::kDLInt &&
                     loc.dtype().bits == (kLoc64 ? 64 : 32),
                 "loc must be [tokens] int64/int32 as instantiated");
    RuntimeCheck(k_gs.numel() >= 1 && v_gs.numel() >= 1 && k_gs.dtype().code == DLDataTypeCode::kDLFloat &&
                     k_gs.dtype().bits == 32 && v_gs.dtype().code == DLDataTypeCode::kDLFloat && v_gs.dtype().bits == 32,
                 "global scales must be fp32");
    const auto packed_ok = [&](const tvm::ffi::TensorView& t) {
      return t.dim() == 3 && t.dtype().bits == 8 && t.size(1) == hn && t.size(2) == hd / 2 && t.stride(2) == 1 &&
             t.stride(1) == hd / 2 && t.stride(0) == hn * hd / 2 && reinterpret_cast<uintptr_t>(t.data_ptr()) % 4 == 0;
    };
    RuntimeCheck(packed_ok(k_out) && packed_ok(v_out) && k_out.size(0) == v_out.size(0), "packed pools layout");
    const auto scale_ok = [&](const tvm::ffi::TensorView& t) {
      return t.dim() == 4 && t.dtype().bits == 8 && t.size(1) == hn && t.size(2) == page_size && t.size(3) == hd / 16 &&
             t.stride(3) == 1 && t.stride(2) == hd / 16 && t.stride(1) == page_size * hd / 16 &&
             t.stride(0) == hn * page_size * hd / 16 && t.size(0) * page_size == k_out.size(0);
    };
    RuntimeCheck(page_size % 4 == 0 && scale_ok(k_sf) && scale_ok(v_sf), "native scale pools layout");
    if (n == 0) return;
    nks::Params p{};
    p.k = static_cast<const __nv_bfloat16*>(k.data_ptr());
    p.v = static_cast<const __nv_bfloat16*>(v.data_ptr());
    p.loc = loc.data_ptr();
    p.k_gs = static_cast<const float*>(k_gs.data_ptr());
    p.v_gs = static_cast<const float*>(v_gs.data_ptr());
    p.k_out = static_cast<uint8_t*>(k_out.data_ptr());
    p.v_out = static_cast<uint8_t*>(v_out.data_ptr());
    p.k_sf = static_cast<uint8_t*>(k_sf.data_ptr());
    p.v_sf = static_cast<uint8_t*>(v_sf.data_ptr());
    p.k_tok_stride = k.stride(0);
    p.k_head_stride = k.stride(1);
    p.v_tok_stride = v.stride(0);
    p.v_head_stride = v.stride(1);
    p.loc_stride = loc.stride(0);
    p.num_tokens = static_cast<int32_t>(n);
    p.num_heads = static_cast<int32_t>(hn);
    p.head_dim = static_cast<int32_t>(hd);
    p.page_size = static_cast<int32_t>(page_size);
    const int64_t items = n * 2 * hn * (hd / 8);
    const unsigned blocks = static_cast<unsigned>((items + nks::kThreads - 1) / nks::kThreads);
    LaunchKernel(dim3(blocks), nks::kThreads, k.device(), 0)
        .config({.use_pdl = kUsePDL})(nks::nvfp4_kv_store_kernel<kLoc64, kUsePDL>, p);
  }
};

}  // namespace muse_vr
