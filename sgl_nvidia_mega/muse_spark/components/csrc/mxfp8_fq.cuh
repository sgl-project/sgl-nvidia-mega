// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
/**
 * \brief MXFP8 quantization of BF16 operands (E4M3 values, one E8M0 scale per 32 elements along K, scales in the
 * 128 x 4 swizzled layout) with several independent loads in flight per thread, for one tensor or a group of up to
 * four tensors in one launch.
 *
 * Arithmetic, per block of 32 consecutive elements of a row (the rule of the MXFP8 quantizer SGLang's MXFP8 linear
 * method calls, reproduced operation by operation so the bytes are identical; checked by tests/test_mxfp8_fq.py):
 *   amax = max over the block of |x| (bf16 absolute value and bf16 maximum, pairwise; a 16-element half per thread,
 *          the two halves combined across the thread pair), then converted to fp32;
 *   s    = amax * rcp.approx.ftz(448)  (fp32 multiply; kFtz selects the flush-to-zero form);
 *   e    = E8M0 of s rounded toward +inf, saturating to the largest finite code (the stored scale byte);
 *   r    = (float(e) != 0) ? rcp.approx.ftz(float(e)) : 0, the "!= 0" test flush-to-zero in the kFtz form (so the
 *          subnormal 2^-127 of code 0x00 counts as zero there);
 *   q    = E4M3 (round to nearest even, saturating) of clamp(float(x) * r, -448, 448). kLean computes the same bytes with
 *          packed pair multiplies and the saturating conversion alone, then maps NaN codes to +448 as the clamp does.
 * Rows of the scale layout past the tensor's rows (up to the next multiple of 128) get scale bytes 0. K must be a
 * multiple of 128 (whole scale tiles) and at most 8192; the Python wrapper falls back to the stock quantizer otherwise.
 *
 * Work split: rows of all tensors (each padded to a multiple of 128) are numbered in order; a block takes U rows per
 * step (rows blockIdx.x + u * gridDim.x), and thread t handles the 16-element items t and t + 256 of each row, so the two
 * halves of every scale block sit in one lane pair. Two register stages: the loads of the next step are issued before the
 * current step converts and stores, so they are in flight during the conversion. A padding row only writes zero scale
 * bytes. With use_pdl the kernel reads its inputs only after griddepcontrol.wait and signals dependents after its stores.
 */

#include <sgl_kernel/tensor.h>
#include <sgl_kernel/utils.h>

#include <sgl_kernel/runtime.cuh>
#include <sgl_kernel/utils.cuh>

#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>

#include <cstdint>

namespace muse_vr {
namespace fq {

constexpr int kThreads = 256;
constexpr int kItemsPerThread = 2;  // items per thread per row: K <= 2 * 256 * 16 = 8192
constexpr int kMaxK = kThreads * kItemsPerThread * 16;
constexpr int kMaxSeg = 4;

struct Seg {
  const uint4* in;  // [rows, K] bf16 as 16-byte words
  uint4* out;       // [rows, K] e4m3 as 16-byte words
  uint8_t* sf;      // swizzled scales
  int32_t row0;     // first row of this tensor in the concatenated (padded) row numbering
  int32_t rows;     // data rows
  int32_t items_per_row;  // K / 16
  int32_t sf_cols;        // K / 32
};

struct Params {
  Seg seg[kMaxSeg];
  int32_t nseg;
  int32_t total_rows;  // sum over tensors of rows padded to 128
};

__device__ __forceinline__ float rcp_approx_ftz(float a) {
  float b;
  asm volatile("rcp.approx.ftz.f32 %0, %1;\n" : "=f"(b) : "f"(a));
  return b;
}

template <bool kFtz>
__device__ __forceinline__ float fmul(float a, float b) {
  float c;
  if constexpr (kFtz) {
    asm("mul.ftz.f32 %0, %1, %2;\n" : "=f"(c) : "f"(a), "f"(b));
  } else {
    asm("mul.rn.f32 %0, %1, %2;\n" : "=f"(c) : "f"(a), "f"(b));
  }
  return c;
}

template <bool kFtz>
__device__ __forceinline__ bool nonzero(float x) {
  if constexpr (kFtz) {
    uint32_t p;
    asm("{\n .reg .pred q;\n setp.ne.ftz.f32 q, %1, 0f00000000;\n selp.u32 %0, 1, 0, q;\n}\n" : "=r"(p) : "f"(x));
    return p != 0u;
  } else {
    return x != 0.0f;
  }
}

// Byte offset of scale (row m, scale column k) in the 128 x 4 swizzled layout: tiles of 128 rows x 4 columns (512 bytes),
// row tiles outer; inside a tile, [32][4][4] = (m % 32, (m % 128) / 32, k % 4).
__device__ __forceinline__ int64_t sf_offset(int32_t m, int32_t k, int32_t sf_cols) {
  const int32_t k_tiles = (sf_cols + 3) >> 2;
  return static_cast<int64_t>(m >> 7) * k_tiles * 512 + (k >> 2) * 512 + (m & 31) * 16 + ((m & 127) >> 5) * 4 + (k & 3);
}

__device__ __forceinline__ int find_seg(const Params& p, int32_t row) {
  int s = 0;
#pragma unroll
  for (int i = 1; i < kMaxSeg; ++i)
    if (i < p.nseg && row >= p.seg[i].row0) s = i;
  return s;
}

// Packed fp32 pair multiply (PTX mul .f32x2: two independent IEEE fp32 multiplies, the same results as two scalar ones).
template <bool kFtz>
__device__ __forceinline__ float2 fmul2(float2 a, float2 b) {
  uint64_t d;
  const uint64_t x = *reinterpret_cast<uint64_t*>(&a), y = *reinterpret_cast<uint64_t*>(&b);
  if constexpr (kFtz) {
    asm("mul.rn.ftz.f32x2 %0, %1, %2;\n" : "=l"(d) : "l"(x), "l"(y));
  } else {
    asm("mul.rn.f32x2 %0, %1, %2;\n" : "=l"(d) : "l"(x), "l"(y));
  }
  return *reinterpret_cast<float2*>(&d);
}

// E4M3 bytes whose low 7 bits are all ones (the NaN codes 0x7f / 0xff) -> 0x7e (+448): what clamp(NaN, -448, 448) gives
// (min / max return the non-NaN operand, so a NaN product clamps to +448).
__device__ __forceinline__ uint32_t nan_to_448(uint32_t w) {
  const uint32_t m = __vcmpeq4(w & 0x7f7f7f7fu, 0x7f7f7f7fu);
  return (w & ~m) | (0x7e7e7e7eu & m);
}

// Quantize one 16-element item held by this lane; the partner lane (lane ^ 1) holds the other half of the scale block.
// Every lane of the warp must call it (the pair maximum is a shuffle). Returns the 16 E4M3 bytes and the scale byte.
// kLean: packed pair multiplies and no explicit clamp. The satfinite conversion already maps finite values beyond +-448 and
// +-inf to +-448, so the clamp only changes NaN products, which nan_to_448 maps as the clamp does (the edge units check this
// byte for byte; the clamped form stays selectable).
template <bool kFtz, bool kLean>
__device__ __forceinline__ uint4 quant_item(const uint4 a, const uint4 b, uint8_t& scale_byte) {
  __nv_bfloat162 v[8];
  *reinterpret_cast<uint4*>(&v[0]) = a;
  *reinterpret_cast<uint4*>(&v[4]) = b;
  __nv_bfloat162 m = __habs2(v[0]);
#pragma unroll
  for (int i = 1; i < 8; ++i) m = __hmax2(m, __habs2(v[i]));
  uint32_t mu = *reinterpret_cast<uint32_t*>(&m);
  uint32_t ou = __shfl_xor_sync(0xffffffffu, mu, 1);
  m = __hmax2(*reinterpret_cast<__nv_bfloat162*>(&ou), m);
  const float amax = __bfloat162float(__hmax(m.x, m.y));
  const float s = fmul<kFtz>(amax, rcp_approx_ftz(448.0f));
  __nv_fp8_e8m0 e;
  e.__x = __nv_cvt_float_to_e8m0(s, __NV_SATFINITE, cudaRoundPosInf);
  const float sv = static_cast<float>(e);
  const float r = nonzero<kFtz>(sv) ? rcp_approx_ftz(sv) : 0.0f;
  scale_byte = e.__x;
  union {
    uint4 u;
    uint32_t w[4];
    __nv_fp8x2_e4m3 q[8];
  } o;
  if constexpr (kLean) {
    const float2 rr = make_float2(r, r);
#pragma unroll
    for (int i = 0; i < 8; ++i) o.q[i] = __nv_fp8x2_e4m3(fmul2<kFtz>(__bfloat1622float2(v[i]), rr));
#pragma unroll
    for (int j = 0; j < 4; ++j) o.w[j] = nan_to_448(o.w[j]);
  } else {
#pragma unroll
    for (int i = 0; i < 8; ++i) {
      float2 f = __bfloat1622float2(v[i]);
      f.x = fmaxf(fminf(fmul<kFtz>(f.x, r), 448.0f), -448.0f);
      f.y = fmaxf(fminf(fmul<kFtz>(f.y, r), 448.0f), -448.0f);
      o.q[i] = __nv_fp8x2_e4m3(f);
    }
  }
  return o.u;
}

// One stage = U rows (rows r0 + u * gridDim.x). load_stage issues the stage's loads (zeros past the data); store_stage converts
// and stores them.
template <int U>
__device__ __forceinline__ void load_stage(const Params& p, int32_t r0, int t, uint4 (&a)[U][kItemsPerThread],
                                           uint4 (&b)[U][kItemsPerThread]) {
#pragma unroll
  for (int u = 0; u < U; ++u) {
    const int32_t r = r0 + u * static_cast<int32_t>(gridDim.x);
#pragma unroll
    for (int i = 0; i < kItemsPerThread; ++i) {
      a[u][i] = make_uint4(0u, 0u, 0u, 0u);
      b[u][i] = a[u][i];
    }
    if (r >= p.total_rows) continue;
    const Seg& sg = p.seg[find_seg(p, r)];
    const int32_t row = r - sg.row0;
    if (row >= sg.rows) continue;  // padding row: no data
    const uint4* const src_row = sg.in + 2 * static_cast<int64_t>(row) * sg.items_per_row;
#pragma unroll
    for (int i = 0; i < kItemsPerThread; ++i) {
      const int32_t it = t + i * kThreads;
      if (it < sg.items_per_row) {
        // Coherent global loads (ld.global.cg), not the read-only path: with use_pdl the input may be written by the
        // preceding kernel while this one is already resident, and is read here after griddepcontrol.wait.
        a[u][i] = __ldcg(src_row + 2 * it);
        b[u][i] = __ldcg(src_row + 2 * it + 1);
      }
    }
  }
}

template <int U, bool kFtz, bool kLean>
__device__ __forceinline__ void store_stage(const Params& p, int32_t r0, int t, int warp_base,
                                            const uint4 (&a)[U][kItemsPerThread], const uint4 (&b)[U][kItemsPerThread]) {
#pragma unroll
  for (int u = 0; u < U; ++u) {
    const int32_t r = r0 + u * static_cast<int32_t>(gridDim.x);
    if (r >= p.total_rows) break;  // block-uniform
    const Seg& sg = p.seg[find_seg(p, r)];
    const int32_t row = r - sg.row0;
    if (row >= sg.rows) {  // padding row of the scale layout (block-uniform): zero scale bytes
      for (int32_t k = t; k < sg.sf_cols; k += kThreads) sg.sf[sf_offset(row, k, sg.sf_cols)] = 0;
      continue;
    }
    // per-row address terms, hoisted (sf_offset(row, k) = sf_row + (k >> 2) * 512 + (k & 3))
    uint4* const out_row = sg.out + static_cast<int64_t>(row) * sg.items_per_row;
    uint8_t* const sf_row = sg.sf + sf_offset(row, 0, sg.sf_cols);
#pragma unroll
    for (int i = 0; i < kItemsPerThread; ++i) {
      if (warp_base + i * kThreads >= sg.items_per_row) continue;  // warp-uniform
      const int32_t it = t + i * kThreads;
      uint8_t sb;
      const uint4 q = quant_item<kFtz, kLean>(a[u][i], b[u][i], sb);
      if (it < sg.items_per_row) {
        out_row[it] = q;
        const int32_t k = it >> 1;
        if ((it & 1) == 0) sf_row[(k >> 2) * 512 + (k & 3)] = sb;
      }
    }
  }
}

// Two register stages: the next stage's loads are in flight while the current stage converts and stores.
template <int U, bool kUsePDL, bool kFtz, bool kLean>
__global__ void __launch_bounds__(kThreads) mxfp8_fq_kernel(const __grid_constant__ Params p) {
  if constexpr (kUsePDL) asm volatile("griddepcontrol.wait;\n" ::: "memory");
  const int t = threadIdx.x;
  const int warp_base = t & ~31;
  const int32_t step = static_cast<int32_t>(gridDim.x) * U;
  uint4 a0[U][kItemsPerThread], b0[U][kItemsPerThread], a1[U][kItemsPerThread], b1[U][kItemsPerThread];
  int32_t r0 = blockIdx.x;
  load_stage<U>(p, r0, t, a0, b0);
  while (r0 < p.total_rows) {
    const int32_t r1 = r0 + step;
    load_stage<U>(p, r1, t, a1, b1);
    store_stage<U, kFtz, kLean>(p, r0, t, warp_base, a0, b0);
    if (r1 >= p.total_rows) break;
    const int32_t r2 = r1 + step;
    load_stage<U>(p, r2, t, a0, b0);
    store_stage<U, kFtz, kLean>(p, r1, t, warp_base, a1, b1);
    r0 = r2;
  }
  if constexpr (kUsePDL) asm volatile("griddepcontrol.launch_dependents;\n" ::: "memory");
}

template <int U, bool kUsePDL, bool kFtz, bool kLean>
struct Mxfp8FqKernel {
  // Up to four (in, out, sf) triples; triples past nseg are ignored (pass any valid tensors there).
  static void run(const tvm::ffi::TensorView in0, const tvm::ffi::TensorView out0, const tvm::ffi::TensorView sf0,
                  const tvm::ffi::TensorView in1, const tvm::ffi::TensorView out1, const tvm::ffi::TensorView sf1,
                  const tvm::ffi::TensorView in2, const tvm::ffi::TensorView out2, const tvm::ffi::TensorView sf2,
                  const tvm::ffi::TensorView in3, const tvm::ffi::TensorView out3, const tvm::ffi::TensorView sf3,
                  int64_t nseg, int64_t grid) {
    using namespace sglang::host;
    RuntimeCheck(nseg >= 1 && nseg <= kMaxSeg, "1 to 4 tensors");
    const tvm::ffi::TensorView* ins[kMaxSeg] = {&in0, &in1, &in2, &in3};
    const tvm::ffi::TensorView* outs[kMaxSeg] = {&out0, &out1, &out2, &out3};
    const tvm::ffi::TensorView* sfs[kMaxSeg] = {&sf0, &sf1, &sf2, &sf3};
    Params p{};
    p.nseg = static_cast<int32_t>(nseg);
    int64_t total = 0;
    for (int i = 0; i < nseg; ++i) {
      const auto& x = *ins[i];
      const auto& y = *outs[i];
      const auto& f = *sfs[i];
      RuntimeCheck(x.dim() == 2 && y.dim() == 2 && x.dtype().bits == 16 && y.dtype().bits == 8 && f.dtype().bits == 8,
                   "in [rows, K] 16-bit, out [rows, K] 8-bit, scales 8-bit");
      const int64_t rows = x.size(0), k = x.size(1);
      RuntimeCheck(y.size(0) == rows && y.size(1) == k && x.stride(1) == 1 && x.stride(0) == k && y.stride(1) == 1 &&
                       y.stride(0) == k,
                   "contiguous [rows, K] operands");
      RuntimeCheck(k % 128 == 0 && k > 0 && k <= kMaxK, "K must be a multiple of 128, at most 8192");
      const int64_t rows_pad = (rows + 127) / 128 * 128;
      RuntimeCheck(f.numel() >= rows_pad * (k / 32), "scale buffer too small");
      RuntimeCheck(reinterpret_cast<uintptr_t>(x.data_ptr()) % 16 == 0 && reinterpret_cast<uintptr_t>(y.data_ptr()) % 16 == 0,
                   "16-byte aligned operands");
      Seg& s = p.seg[i];
      s.in = static_cast<const uint4*>(x.data_ptr());
      s.out = static_cast<uint4*>(y.data_ptr());
      s.sf = static_cast<uint8_t*>(f.data_ptr());
      s.row0 = static_cast<int32_t>(total);
      s.rows = static_cast<int32_t>(rows);
      s.items_per_row = static_cast<int32_t>(k / 16);
      s.sf_cols = static_cast<int32_t>(k / 32);
      total += rows_pad;
    }
    RuntimeCheck(total < (int64_t(1) << 31), "too many rows");
    p.total_rows = static_cast<int32_t>(total);
    if (total == 0) return;
    RuntimeCheck(grid >= 1, "grid >= 1");
    LaunchKernel(dim3(static_cast<unsigned>(grid)), kThreads, in0.device(), 0)
        .config({.use_pdl = kUsePDL})(mxfp8_fq_kernel<U, kUsePDL, kFtz, kLean>, p);
  }
};

}  // namespace fq
}  // namespace muse_vr
