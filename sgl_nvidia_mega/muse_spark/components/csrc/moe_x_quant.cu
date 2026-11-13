// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// Row kernels of the routed-expert path: per-token NVFP4 quantization of the expert-sorted rows (the input of the second
// expert GEMM), and the copy of the call's NVFP4 activations into sorted-row order (the input of the first expert GEMM).
//
// Arithmetic: the per-row and per-16-element-block computation of FlashInfer's nvfp4QuantAndPerTokenScaleKernel in
// its default configuration (fast math, no 4over6 search): row amax over the BF16 row, per-token scale
// rowAmax * globalScaleInv, encode scale = rcp.approx.ftz(per-token scale) (0 for an all-zero row), then FlashInfer's
// own per-block conversion cvt_warp_fp16_to_fp4 (block scale, E4M3 rounding, E2M1 packing) and its 128x4 swizzled
// scale-factor offset. The row-sorted and region forms call those device functions; the row-block form writes the same
// per-block operations out in explicit PTX (see quant_row_blocks_kernel) and is tested bitwise against them.
//
// Work split (the only difference): one warp per row, each lane holding num_cols / 512 blocks of 16 values in
// registers, so every row is read once; rows are visited by a grid-stride loop over the sorted rows of the valid tiles
// only (row < tile_limit[row / tile_m] for row < num_tiles * tile_m, both read from device memory). The row maximum is
// an exact reduction (max), so the visiting order does not change any result.
//
// Second form (quant_row_blocks): the row maximum is supplied per row (the first expert GEMM's epilogue reduces it
// into row_amax as the FP32 bit pattern of the largest |BF16 output| of the row), so each thread converts independent
// 16-value blocks with the same per-row and per-block arithmetic and no row-wide reduction.

#include <cuda_bf16.h>

#include <cstdint>

#include "tensorrt_llm/kernels/quantization.cuh"
#include "tensorrt_llm/kernels/quantization_utils.cuh"
#include "tvm_ffi_utils.h"

namespace muse_vr {
namespace {

using tensorrt_llm::kernels::cvt_warp_fp16_to_fp4;
using tensorrt_llm::kernels::get_sf_out_offset_128x4;
using tensorrt_llm::kernels::loadPackedVec;
using tensorrt_llm::kernels::PackedVec;
using tensorrt_llm::kernels::reciprocal_approximate_ftz;

constexpr int kSfVec = 16;
constexpr int kWarps = 8;
constexpr int kMinBlocks = 2;  // resident blocks per SM the register budget must allow (tunable; bench)

// 32-byte global accesses
struct alignas(32) Block32 {
  uint4 a, b;
};
__device__ __forceinline__ Block32 ld_block32(void const* p) {
  Block32 v;
  asm("ld.global.v8.b32 {%0, %1, %2, %3, %4, %5, %6, %7}, [%8];"
      : "=r"(v.a.x), "=r"(v.a.y), "=r"(v.a.z), "=r"(v.a.w), "=r"(v.b.x), "=r"(v.b.y), "=r"(v.b.z), "=r"(v.b.w)
      : "l"(p));
  return v;
}
__device__ __forceinline__ void st_block32(void* p, Block32 const& v) {
  asm volatile("st.global.v8.b32 [%0], {%1, %2, %3, %4, %5, %6, %7, %8};" ::"l"(p), "r"(v.a.x), "r"(v.a.y), "r"(v.a.z),
               "r"(v.a.w), "r"(v.b.x), "r"(v.b.y), "r"(v.b.z), "r"(v.b.w)
               : "memory");
}

template <int kVecsPerLane>
__global__ void __launch_bounds__(kWarps * 32, kMinBlocks)
    quant_sorted_rows_kernel(__nv_bfloat16 const* __restrict__ in, int num_cols, int max_rows, int tile_m,
                             int const* __restrict__ num_tiles, int const* __restrict__ tile_limit,
                             float global_scale_inv, uint8_t* __restrict__ out, uint8_t* __restrict__ out_sf,
                             float* __restrict__ out_scale) {
  using InType = PackedVec<__nv_bfloat16, kSfVec>;
  using PackedFp4Type = uint64_t;
  int const lane = threadIdx.x & 31;
  int const nwarps = gridDim.x * kWarps;
  int const rows_end = min(num_tiles[0] * tile_m, max_rows);
  int const num_vecs = num_cols / kSfVec;
  for (int row = blockIdx.x * kWarps + (threadIdx.x >> 5); row < rows_end; row += nwarps) {
    if (row >= tile_limit[row / tile_m]) continue;  // padding row of its tile (warp-uniform)
    InType vec[kVecsPerLane];
    float local_amax = 0.f;
#pragma unroll
    for (int i = 0; i < kVecsPerLane; ++i) {
      int64_t const off = static_cast<int64_t>(row) * num_vecs + lane + 32 * i;
      loadPackedVec(vec[i], reinterpret_cast<InType const*>(in) + off);
      __nv_bfloat162 m(0.f, 0.f);
#pragma unroll
      for (int j = 0; j < kSfVec / 2; ++j) {
        m = __hmax2(m, __habs2(vec[i].elts[j]));
      }
      local_amax = fmaxf(local_amax, static_cast<float>(__hmax(m.x, m.y)));
    }
#pragma unroll
    for (int s = 16; s > 0; s >>= 1) {
      local_amax = fmaxf(local_amax, __shfl_xor_sync(0xffffffffu, local_amax, s));
    }
    float const row_amax = local_amax;
    float const per_token_scale = row_amax * global_scale_inv;
    if (lane == 0) out_scale[row] = per_token_scale;
    float const encode_scale = per_token_scale != 0.0f ? reciprocal_approximate_ftz(per_token_scale) : 0.0f;
#pragma unroll
    for (int i = 0; i < kVecsPerLane; ++i) {
      int const v = lane + 32 * i;
      uint8_t sf;
      PackedFp4Type q = cvt_warp_fp16_to_fp4<__nv_bfloat16, kSfVec, kSfVec, false, false, std::false_type>(
          vec[i], encode_scale, &sf, row_amax);
      reinterpret_cast<PackedFp4Type*>(out)[static_cast<int64_t>(row) * num_vecs + v] = q;
      out_sf[get_sf_out_offset_128x4(row, v, num_vecs)] = sf;
    }
  }
}

// Copies each valid sorted row's NVFP4 activation (payload, block scales, per-token scale) from its token row: the payload
// row as is, the linear block scales into the 128x4 swizzled layout of the sorted rows, the per-token scale per sorted
// row; zeroes the row's entry of row_amax (if given) and the ctl words (if given; block 0) for the first GEMM's epilogue.
// One warp per item = the eight sorted rows whose block scales share 32-byte pieces of the swizzled layout (rows 2j,
// 2j + 1, 2j + 32, 2j + 33, ... of a 128-row block) x one half of the row: all loads of the item are issued before its
// stores, and each lane stores one whole 32-byte scale piece (padding rows of a partly valid group get zero scales).
// Rows of valid tiles below their mn_limit only; groups with no valid row are skipped.
constexpr int kPermWarps = 4;  // warps per block of the row copy (tunable; bench)

template <int kRowBytes, int kSfWordsPerRow>
__global__ void __launch_bounds__(kPermWarps * 32)
    permute_rows_kernel(uint8_t const* __restrict__ x, uint8_t const* __restrict__ x_sf,
                        float const* __restrict__ token_scale, int const* __restrict__ row_to_expanded, int top_k,
                        int max_rows, int tile_m, int const* __restrict__ num_tiles,
                        int const* __restrict__ tile_limit, uint8_t* __restrict__ out, uint8_t* __restrict__ out_sf,
                        float* __restrict__ out_scale, uint32_t* __restrict__ row_amax, uint32_t* __restrict__ ctl,
                        int ctl_words) {
  constexpr int kHalves = 2;
  constexpr int kHalfBytes = kRowBytes / kHalves;
  static_assert(kHalfBytes == 32 * 32, "one 32-byte payload piece per lane per row half");
  static_assert(kSfWordsPerRow == 32 * kHalves, "one scale word (four blocks) per lane per row half");
  constexpr unsigned kFull = 0xffffffffu;
  if (blockIdx.x == 0 && ctl != nullptr) {
    for (int i = threadIdx.x; i < ctl_words; i += blockDim.x) ctl[i] = 0u;
  }
  int const lane = threadIdx.x & 31;
  int const nwarps = gridDim.x * kPermWarps;
  int const rows_end = min(num_tiles[0] * tile_m, max_rows);
  int const total = (rows_end >> 7) * 16 * kHalves;  // items: (128-row block, row pair j, half)
  int const lr = lane & 7;                           // group row resolved by lanes 0 .. 7
  for (int item = blockIdx.x * kPermWarps + (threadIdx.x >> 5); item < total; item += nwarps) {
    int const h = item & 1;
    int const grp = item >> 1;
    int const base_row = ((grp >> 4) << 7) + 2 * (grp & 15);
    int const limit = tile_limit[base_row / tile_m];  // the group lies in one tile (tile_m is a multiple of 128)
    if (base_row >= limit) continue;                  // no valid row in the group (warp-uniform)
    int const my_row = base_row + (lr & 1) + 32 * (lr >> 1);
    int my_token = -1;
    if (lane < 8 && my_row < limit) my_token = row_to_expanded[my_row] / top_k;
    int tok[8];
#pragma unroll
    for (int r = 0; r < 8; ++r) tok[r] = __shfl_sync(kFull, my_token, r);
    Block32 v[8];
    uint32_t w[8];
#pragma unroll
    for (int r = 0; r < 8; ++r) {
      w[r] = 0u;
      if (tok[r] >= 0) {
        v[r] = ld_block32(x + static_cast<int64_t>(tok[r]) * kRowBytes + h * kHalfBytes + lane * 32);
        w[r] = reinterpret_cast<uint32_t const*>(x_sf)[static_cast<int64_t>(tok[r]) * kSfWordsPerRow + h * 32 + lane];
      }
    }
    float scale = 0.0f;
    if (h == 0 && my_token >= 0) scale = token_scale[my_token];
#pragma unroll
    for (int r = 0; r < 8; ++r) {
      if (tok[r] >= 0) {
        int const row = base_row + (r & 1) + 32 * (r >> 1);
        st_block32(out + static_cast<int64_t>(row) * kRowBytes + h * kHalfBytes + lane * 32, v[r]);
      }
    }
    // scale piece of k tile h * 32 + lane: word (r & 1) * 4 + (r >> 1) belongs to group row r
    Block32 piece;
    piece.a = make_uint4(w[0], w[2], w[4], w[6]);
    piece.b = make_uint4(w[1], w[3], w[5], w[7]);
    int64_t const sf_off = static_cast<int64_t>(base_row >> 7) * kSfWordsPerRow * 512 +
                           static_cast<int64_t>(h * 32 + lane) * 512 + (base_row & 31) * 16;
    st_block32(out_sf + sf_off, piece);
    if (h == 0 && my_token >= 0) {
      out_scale[my_row] = scale;
      if (row_amax != nullptr) row_amax[my_row] = 0u;
    }
  }
}

// Quantizes the regions (128 sorted rows x 128 columns) the first GEMM left unconverted: the ctl words are
// [row-block counters (num_blocks) | per-region done flags (num_blocks x regions_per_block) | error word]; a region whose flag is 0
// is converted here from the BF16 rows and the row maxima, with the same per-row and per-block arithmetic. One block
// of 128 threads per region (one row per thread); rows of valid tiles below their mn_limit only.
__global__ void __launch_bounds__(128)
    quant_regions_kernel(__nv_bfloat16 const* __restrict__ in, uint32_t const* __restrict__ row_amax,
                         uint32_t const* __restrict__ ctl, int num_cols, int max_rows, int tile_m,
                         int const* __restrict__ num_tiles, int const* __restrict__ tile_limit, float global_scale_inv,
                         uint8_t* __restrict__ out, uint8_t* __restrict__ out_sf, float* __restrict__ out_scale) {
  using InType = PackedVec<__nv_bfloat16, kSfVec>;
  using PackedFp4Type = uint64_t;
  constexpr int kRegionCols = 128;
  constexpr int kBlocks = kRegionCols / kSfVec;  // 8
  int const num_vecs = num_cols / kSfVec;
  int const regions_per_block = num_cols / kRegionCols;
  int const num_blocks = max_rows / 128;
  int const rows_end = min(num_tiles[0] * tile_m, max_rows);
  int const used_blocks = (rows_end + 127) / 128;
  uint32_t const* done = ctl + num_blocks;
  for (int reg = blockIdx.x; reg < used_blocks * regions_per_block; reg += gridDim.x) {
    if (done[reg] != 0u) continue;  // block-uniform
    int const rb = reg / regions_per_block;
    int const n = reg - rb * regions_per_block;
    int const row = rb * 128 + threadIdx.x;
    if (row >= rows_end || row >= tile_limit[row / tile_m]) continue;
    InType vec[kBlocks];
    int64_t const base = static_cast<int64_t>(row) * num_vecs + n * kBlocks;
#pragma unroll
    for (int i = 0; i < kBlocks; ++i) loadPackedVec(vec[i], reinterpret_cast<InType const*>(in) + base + i);
    float const amax = __uint_as_float(row_amax[row]);
    float const per_token_scale = amax * global_scale_inv;
    if (n == 0) out_scale[row] = per_token_scale;
    float const encode_scale = per_token_scale != 0.0f ? reciprocal_approximate_ftz(per_token_scale) : 0.0f;
    uint32_t words[2] = {0u, 0u};
#pragma unroll
    for (int i = 0; i < kBlocks; ++i) {
      uint8_t sf;
      PackedFp4Type q = cvt_warp_fp16_to_fp4<__nv_bfloat16, kSfVec, kSfVec, false, false, std::false_type>(
          vec[i], encode_scale, &sf, amax);
      reinterpret_cast<PackedFp4Type*>(out)[base + i] = q;
      words[i / 4] |= static_cast<uint32_t>(sf) << (8 * (i % 4));
    }
    *reinterpret_cast<uint32_t*>(out_sf + get_sf_out_offset_128x4(row, n * kBlocks, num_vecs)) = words[0];
    *reinterpret_cast<uint32_t*>(out_sf + get_sf_out_offset_128x4(row, n * kBlocks + 4, num_vecs)) = words[1];
  }
}

// Per-token NVFP4 quantization of the valid sorted rows from a given row maximum (second form, see the header): each
// lane converts four 16-value blocks per item (one 32-byte load each) and stores their four scale bytes as one word;
// the warp issues the loads of its next item before converting the current one (two items in flight, in registers).
// The work split over rows and blocks is described at the kernel.
//
// The per-block arithmetic is written out with explicit PTX here: the same operations, operand order, rounding and
// flush-to-zero behaviour as the per-block function used above (abs / max in BF16, scale-factor product, RN-satfinite
// E4M3 scale, approximate reciprocals, per-element products, RN-satfinite E2M1 codes in the same packing), with the
// independent products of two elements or two blocks issued as one two-wide multiply. Outputs are bitwise equal to
// that function's (tests).
__device__ __forceinline__ float mul_ftz(float a, float b) {
  float d;
  asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(d) : "f"(a), "f"(b));
  return d;
}
__device__ __forceinline__ uint64_t pack2(float lo, float hi) {
  uint64_t d;
  asm("mov.b64 %0, {%1, %2};" : "=l"(d) : "f"(lo), "f"(hi));
  return d;
}
__device__ __forceinline__ uint64_t mul_ftz_x2(uint64_t a, uint64_t b) {
  uint64_t d;
  asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(d) : "l"(a), "l"(b));
  return d;
}
__device__ __forceinline__ float lo_f32(uint64_t v) { return __uint_as_float(static_cast<uint32_t>(v)); }
__device__ __forceinline__ float hi_f32(uint64_t v) { return __uint_as_float(static_cast<uint32_t>(v >> 32)); }

// Largest |value| of a 16-value block in BF16 (same reduction as the per-block function), as FP32.
__device__ __forceinline__ float block_abs_max(uint4 const& a, uint4 const& b) {
  using tensorrt_llm::common::cuda_abs;
  using tensorrt_llm::common::cuda_max;
  uint32_t const w[8] = {a.x, a.y, a.z, a.w, b.x, b.y, b.z, b.w};
  __nv_bfloat162 m = cuda_abs(*reinterpret_cast<__nv_bfloat162 const*>(&w[0]));
#pragma unroll
  for (int j = 1; j < 8; ++j) m = cuda_max(m, cuda_abs(*reinterpret_cast<__nv_bfloat162 const*>(&w[j])));
  return float(cuda_max(m.x, m.y));
}

// Sixteen BF16 values (eight bf16x2 words, element x in the low half) times one scale -> sixteen E2M1 codes, byte j =
// code(y_j) << 4 | code(x_j), byte 0 lowest (the stock packing).
__device__ __forceinline__ uint64_t block_to_e2m1(uint4 const& a, uint4 const& b, float scale) {
  uint32_t const w[8] = {a.x, a.y, a.z, a.w, b.x, b.y, b.z, b.w};
  uint64_t const s2 = pack2(scale, scale);
  float f[16];
#pragma unroll
  for (int j = 0; j < 8; ++j) {
    uint64_t const p = mul_ftz_x2(pack2(__uint_as_float(w[j] << 16), __uint_as_float(w[j] & 0xffff0000u)), s2);
    f[2 * j] = lo_f32(p);
    f[2 * j + 1] = hi_f32(p);
  }
  uint32_t lo, hi;
  asm("{\n"
      ".reg .b8 b0, b1, b2, b3, b4, b5, b6, b7;\n"
      "cvt.rn.satfinite.e2m1x2.f32 b0, %3, %2;\n"
      "cvt.rn.satfinite.e2m1x2.f32 b1, %5, %4;\n"
      "cvt.rn.satfinite.e2m1x2.f32 b2, %7, %6;\n"
      "cvt.rn.satfinite.e2m1x2.f32 b3, %9, %8;\n"
      "cvt.rn.satfinite.e2m1x2.f32 b4, %11, %10;\n"
      "cvt.rn.satfinite.e2m1x2.f32 b5, %13, %12;\n"
      "cvt.rn.satfinite.e2m1x2.f32 b6, %15, %14;\n"
      "cvt.rn.satfinite.e2m1x2.f32 b7, %17, %16;\n"
      "mov.b32 %0, {b0, b1, b2, b3};\n"
      "mov.b32 %1, {b4, b5, b6, b7};\n"
      "}"
      : "=r"(lo), "=r"(hi)
      : "f"(f[0]), "f"(f[1]), "f"(f[2]), "f"(f[3]), "f"(f[4]), "f"(f[5]), "f"(f[6]), "f"(f[7]), "f"(f[8]),
        "f"(f[9]), "f"(f[10]), "f"(f[11]), "f"(f[12]), "f"(f[13]), "f"(f[14]), "f"(f[15]));
  return (static_cast<uint64_t>(hi) << 32) | lo;
}

// Scale factors of two blocks with largest magnitudes m0, m1: returns the two E4M3 bytes (block 0 in the low byte) and
// the two output scales.
__device__ __forceinline__ uint32_t pair_scales(float m0, float m1, float rcp6, float enc, float rcp_enc, float& o0,
                                                float& o1) {
  uint64_t const t = mul_ftz_x2(pack2(m0, m1), pack2(rcp6, rcp6));
  uint64_t const s = mul_ftz_x2(pack2(enc, enc), t);
  uint16_t sf2;
  asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(sf2) : "f"(hi_f32(s)), "f"(lo_f32(s)));
  uint32_t h2;
  asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(h2) : "h"(sf2));
  float const d0 = __half2float(__ushort_as_half(static_cast<unsigned short>(h2 & 0xffffu)));
  float const d1 = __half2float(__ushort_as_half(static_cast<unsigned short>(h2 >> 16)));
  uint64_t const q = mul_ftz_x2(pack2(d0, d1), pack2(rcp_enc, rcp_enc));
  o0 = m0 != 0.0f ? reciprocal_approximate_ftz(lo_f32(q)) : 0.0f;
  o1 = m1 != 0.0f ? reciprocal_approximate_ftz(hi_f32(q)) : 0.0f;
  return sf2;
}

constexpr int kQuantWarps = 4;  // warps per block of the row-block quantizer (tunable; bench)

// One warp item = 4 x 32-byte blocks per lane plus the lane's row maximum. Two item kinds share one grid-stride index
// space: group items cover the 128-row blocks whose rows are all valid, row items cover the valid rows of the (at most
// one) partly valid 128-row block of each tile.
//  - group item: the eight sorted rows whose block scales share 32-byte pieces of the swizzled layout (rows 2j, 2j + 1,
//    2j + 32, 2j + 33, ... of the 128-row block) x 16 consecutive blocks; lane 4r + q converts blocks 4q .. 4q + 3 of
//    group row r, so its four scale bytes form one word and the eight lanes with the same q store one whole piece.
//  - row item: 128 consecutive blocks of one row; lane l converts blocks l + 32 i and the scale bytes of four
//    consecutive blocks (held by four neighbouring lanes) are regrouped with two shuffles into one word per lane.
struct QuantItem {
  Block32 blk[4];
  uint32_t amax_bits;
};

template <int kNumCols>
__global__ void __launch_bounds__(kQuantWarps * 32)
    quant_row_blocks_kernel(__nv_bfloat16 const* __restrict__ in, uint32_t const* __restrict__ row_amax,
                            int max_rows, int tile_shift, int const* __restrict__ num_tiles,
                            int const* __restrict__ tile_limit, float global_scale_inv, uint8_t* __restrict__ out,
                            uint8_t* __restrict__ out_sf, float* __restrict__ out_scale) {
  constexpr int num_vecs = kNumCols / kSfVec;    // blocks per row
  constexpr int kSegBlocks = 16;                 // group item: blocks of one row
  constexpr int segs = num_vecs / kSegBlocks;    // group items per group
  constexpr int kChunk = 128;                    // row item: blocks of one row
  constexpr int chunks = num_vecs / kChunk;      // row items per row
  static_assert(num_vecs % kChunk == 0, "row width");
  constexpr unsigned kFull = 0xffffffffu;
  int const lane = threadIdx.x & 31;
  int const r = lane >> 2;
  int const q = lane & 3;
  int const row_in_group = (r & 1) + 32 * (r >> 1);
  int const nwarps = gridDim.x * kQuantWarps;
  int const ntiles = min(num_tiles[0], max_rows >> tile_shift);
  int const blocks_per_tile = 1 << (tile_shift - 7);  // 128-row blocks per tile (tile_m is a multiple of 128)
  int const n_group_items = (ntiles * blocks_per_tile) * 16 * segs;
  int const total = n_group_items + ntiles * 128 * chunks;
  // Resolves an item index: returns false if it has no valid work; otherwise row0 / lim are set (row0 = group base row
  // for a group item, the row for a row item).
  auto resolve = [&](int item, int& row0, int& lim) {
    if (item < n_group_items) {
      int const grp = item / segs;
      int const blk = grp >> 4;
      lim = tile_limit[(blk << 7) >> tile_shift];
      row0 = (blk << 7) + 2 * (grp & 15);
      return (blk << 7) + 127 < lim;  // whole 128-row block valid
    }
    int const k = item - n_group_items;
    int const tile = k / (128 * chunks);
    int const rr = (k / chunks) & 127;
    lim = tile_limit[tile];
    int const tail = (lim - (tile << tile_shift)) & 127;  // valid rows of the partly valid block (0: none)
    row0 = lim - tail + rr;
    return rr < tail;
  };
  auto next_valid = [&](int item, int& row0, int& lim) {
    while (item < total && !resolve(item, row0, lim)) item += nwarps;
    return item;
  };
  auto load = [&](QuantItem& d, int item, int row0) {
    if (item < n_group_items) {
      int const row = row0 + row_in_group;
      int const seg = item % segs;
      Block32 const* src =
          reinterpret_cast<Block32 const*>(in) + static_cast<int64_t>(row) * num_vecs + seg * kSegBlocks + 4 * q;
#pragma unroll
      for (int i = 0; i < 4; ++i) d.blk[i] = ld_block32(src + i);
      d.amax_bits = row_amax[row];
    } else {
      int const chunk = (item - n_group_items) % chunks;
      Block32 const* src =
          reinterpret_cast<Block32 const*>(in) + static_cast<int64_t>(row0) * num_vecs + chunk * kChunk + lane;
#pragma unroll
      for (int i = 0; i < 4; ++i) d.blk[i] = ld_block32(src + 32 * i);
      d.amax_bits = row_amax[row0];
    }
  };
  float const rcp6 = reciprocal_approximate_ftz(6.0f);
  // shuffle selectors of the 4 x 4 byte transpose inside each group of four lanes (row items)
  uint32_t const sel_a = q < 2 ? 0x5410u : 0x3276u;
  uint32_t const sel_b = (q & 1) ? 0x3715u : 0x6240u;
  auto convert = [&](QuantItem const& d, int item, int row0) {
    bool const group_item = item < n_group_items;
    int const row = group_item ? row0 + row_in_group : row0;
    int const part = group_item ? item % segs : (item - n_group_items) % chunks;
    float const amax = __uint_as_float(d.amax_bits);
    float const per_token_scale = mul_ftz(amax, global_scale_inv);
    if (part == 0 && (group_item ? q == 0 : lane == 0)) out_scale[row] = per_token_scale;
    float const enc = per_token_scale != 0.0f ? reciprocal_approximate_ftz(per_token_scale) : 0.0f;
    float const rcp_enc = reciprocal_approximate_ftz(enc);
    float o[4];
    uint32_t const sf01 = pair_scales(block_abs_max(d.blk[0].a, d.blk[0].b), block_abs_max(d.blk[1].a, d.blk[1].b),
                                      rcp6, enc, rcp_enc, o[0], o[1]);
    uint32_t const sf23 = pair_scales(block_abs_max(d.blk[2].a, d.blk[2].b), block_abs_max(d.blk[3].a, d.blk[3].b),
                                      rcp6, enc, rcp_enc, o[2], o[3]);
    uint64_t e[4];
#pragma unroll
    for (int i = 0; i < 4; ++i) e[i] = block_to_e2m1(d.blk[i].a, d.blk[i].b, o[i]);
    uint32_t word = sf01 | (sf23 << 16);
    int v;
    if (group_item) {
      v = part * kSegBlocks + 4 * q;
      Block32 packed;
      packed.a = make_uint4(static_cast<uint32_t>(e[0]), static_cast<uint32_t>(e[0] >> 32),
                            static_cast<uint32_t>(e[1]), static_cast<uint32_t>(e[1] >> 32));
      packed.b = make_uint4(static_cast<uint32_t>(e[2]), static_cast<uint32_t>(e[2] >> 32),
                            static_cast<uint32_t>(e[3]), static_cast<uint32_t>(e[3] >> 32));
      st_block32(out + (static_cast<int64_t>(row) * num_vecs + v) * 8, packed);
    } else {
      uint64_t* const out_row = reinterpret_cast<uint64_t*>(out) + static_cast<int64_t>(row) * num_vecs + part * kChunk;
#pragma unroll
      for (int i = 0; i < 4; ++i) out_row[lane + 32 * i] = e[i];
      // byte i of `word` is the scale of block lane + 32 i; after the transpose lane 4g + q holds the scales of blocks
      // 32 q + 4 g .. 32 q + 4 g + 3 (byte k = block 32 q + 4 g + k)
      word = __byte_perm(word, __shfl_xor_sync(kFull, word, 2), sel_a);
      word = __byte_perm(word, __shfl_xor_sync(kFull, word, 1), sel_b);
      v = part * kChunk + 32 * q + 4 * r;
    }
    *reinterpret_cast<uint32_t*>(out_sf + get_sf_out_offset_128x4(row, v, num_vecs)) = word;
  };
  // two items in flight per warp: the next item's loads are issued before the current item is converted
  QuantItem x, y;
  int rx = 0, ry = 0, lx = 0, ly = 0;
  int item = next_valid(blockIdx.x * kQuantWarps + (threadIdx.x >> 5), rx, lx);
  if (item < total) load(x, item, rx);
  while (item < total) {
    int next = next_valid(item + nwarps, ry, ly);
    if (next < total) load(y, next, ry);
    convert(x, item, rx);
    item = next;
    if (item >= total) break;
    next = next_valid(item + nwarps, rx, lx);
    if (next < total) load(x, next, rx);
    convert(y, item, ry);
    item = next;
  }
}

}  // namespace

// x [T, H / 2] uint8, x_sf [T, H / 16] uint8 (linear), token_scale [T] float32, row_to_expanded [R] int32 (token * top_k +
// slot), num_tiles [1], tile_limit [>= num_tiles]; out [R, H / 2] uint8, out_sf [R, H / 16] uint8 (128x4 swizzled),
// out_scale [R] float32; row_amax [R] uint32 (zeroed for the valid rows) or an empty tensor. H = 4096 (template).
void permute_rows(TensorView x, TensorView x_sf, TensorView token_scale, TensorView row_to_expanded, int64_t top_k,
                  TensorView num_tiles, TensorView tile_limit, int64_t tile_m, TensorView out, TensorView out_sf,
                  TensorView out_scale, TensorView row_amax, TensorView ctl, int64_t num_blocks) {
  int const rows = static_cast<int>(out.size(0));
  int const hidden = static_cast<int>(x.size(1)) * 2;
  TVM_FFI_ICHECK_EQ(hidden, 4096) << "hidden width";
  TVM_FFI_ICHECK_EQ(rows % 128, 0) << "rows must be a multiple of 128";
  bool const zero_amax = row_amax.size(0) > 0;
  if (zero_amax) TVM_FFI_ICHECK_EQ(row_amax.size(0), rows) << "row_amax rows";
  cudaStream_t stream = get_stream(x.device());
  TVM_FFI_ICHECK(tile_m > 0 && tile_m % 128 == 0) << "tile_m must be a multiple of 128";
  permute_rows_kernel<4096 / 2, 4096 / kSfVec / 4><<<static_cast<int>(num_blocks), kPermWarps * 32, 0, stream>>>(
      reinterpret_cast<uint8_t const*>(x.data_ptr()), reinterpret_cast<uint8_t const*>(x_sf.data_ptr()),
      reinterpret_cast<float const*>(token_scale.data_ptr()), reinterpret_cast<int const*>(row_to_expanded.data_ptr()),
      static_cast<int>(top_k), rows, static_cast<int>(tile_m), reinterpret_cast<int const*>(num_tiles.data_ptr()),
      reinterpret_cast<int const*>(tile_limit.data_ptr()), reinterpret_cast<uint8_t*>(out.data_ptr()),
      reinterpret_cast<uint8_t*>(out_sf.data_ptr()), reinterpret_cast<float*>(out_scale.data_ptr()),
      zero_amax ? reinterpret_cast<uint32_t*>(row_amax.data_ptr()) : nullptr,
      ctl.size(0) > 0 ? reinterpret_cast<uint32_t*>(ctl.data_ptr()) : nullptr, static_cast<int>(ctl.size(0)));
}

// in [R, C] bf16, row_amax [R] uint32, ctl [R / 128 * (1 + C / 128)] uint32 (see the kernel), num_tiles [1] int32,
// tile_limit [>= num_tiles] int32; out [R, C / 2] uint8, out_sf [R, C / 16] uint8 (128x4 swizzled), out_scale [R].
void quant_regions(TensorView in, TensorView row_amax, TensorView ctl, TensorView num_tiles, TensorView tile_limit,
                   int64_t tile_m, double scale_inv, TensorView out, TensorView out_sf, TensorView out_scale,
                   int64_t num_blocks) {
  int const rows = static_cast<int>(in.size(0));
  int const cols = static_cast<int>(in.size(1));
  TVM_FFI_ICHECK_EQ(cols % 128, 0) << "row width";
  TVM_FFI_ICHECK_EQ(rows % 128, 0) << "rows must be a multiple of 128";
  TVM_FFI_ICHECK_EQ(ctl.size(0), rows / 128 * (1 + cols / 128) + 1) << "ctl words";
  cudaStream_t stream = get_stream(in.device());
  quant_regions_kernel<<<static_cast<int>(num_blocks), 128, 0, stream>>>(
      reinterpret_cast<__nv_bfloat16 const*>(in.data_ptr()), reinterpret_cast<uint32_t const*>(row_amax.data_ptr()),
      reinterpret_cast<uint32_t const*>(ctl.data_ptr()), cols, rows, static_cast<int>(tile_m),
      reinterpret_cast<int const*>(num_tiles.data_ptr()), reinterpret_cast<int const*>(tile_limit.data_ptr()),
      static_cast<float>(scale_inv), reinterpret_cast<uint8_t*>(out.data_ptr()),
      reinterpret_cast<uint8_t*>(out_sf.data_ptr()), reinterpret_cast<float*>(out_scale.data_ptr()));
}

// in [R, C] bf16 (C = 4096), row_amax [R] uint32 (FP32 bits of the row's largest |value|), num_tiles [1] int32,
// tile_limit [>= num_tiles] int32; out [R, C / 2] uint8, out_sf [R, C / 16] uint8 (128x4 swizzled, R a multiple of
// 128), out_scale [R] float32.
void quant_row_blocks(TensorView in, TensorView row_amax, TensorView num_tiles, TensorView tile_limit, int64_t tile_m,
                      double scale_inv, TensorView out, TensorView out_sf, TensorView out_scale, int64_t num_blocks) {
  int const rows = static_cast<int>(in.size(0));
  int const cols = static_cast<int>(in.size(1));
  TVM_FFI_ICHECK_EQ(cols, 4096) << "row width";
  TVM_FFI_ICHECK(tile_m > 0 && (tile_m & (tile_m - 1)) == 0) << "tile_m must be a power of two";
  int const tile_shift = __builtin_ctzll(static_cast<unsigned long long>(tile_m));
  TVM_FFI_ICHECK_EQ(rows % 128, 0) << "rows must be a multiple of 128";
  TVM_FFI_ICHECK_EQ(row_amax.size(0), rows);
  TVM_FFI_ICHECK_EQ(out.size(0), rows);
  TVM_FFI_ICHECK_EQ(out_scale.size(0), rows);
  cudaStream_t stream = get_stream(in.device());
  quant_row_blocks_kernel<4096><<<static_cast<int>(num_blocks), kQuantWarps * 32, 0, stream>>>(
      reinterpret_cast<__nv_bfloat16 const*>(in.data_ptr()), reinterpret_cast<uint32_t const*>(row_amax.data_ptr()),
      rows, tile_shift, reinterpret_cast<int const*>(num_tiles.data_ptr()),
      reinterpret_cast<int const*>(tile_limit.data_ptr()), static_cast<float>(scale_inv),
      reinterpret_cast<uint8_t*>(out.data_ptr()), reinterpret_cast<uint8_t*>(out_sf.data_ptr()),
      reinterpret_cast<float*>(out_scale.data_ptr()));
}

// in [R, C] bf16 (C = 32 * 16 * 8), num_tiles [1] int32, tile_limit [>= num_tiles] int32; out [R, C / 2] uint8,
// out_sf [R, C / 16] uint8 (128x4 swizzled, R a multiple of 128), out_scale [R] float32.
void quant_sorted_rows(TensorView in, TensorView num_tiles, TensorView tile_limit, int64_t tile_m, double scale_inv,
                       TensorView out, TensorView out_sf, TensorView out_scale, int64_t num_blocks) {
  int const rows = static_cast<int>(in.size(0));
  int const cols = static_cast<int>(in.size(1));
  TVM_FFI_ICHECK_EQ(cols, 32 * kSfVec * 8) << "row width";
  TVM_FFI_ICHECK_EQ(rows % 128, 0) << "rows must be a multiple of 128";
  TVM_FFI_ICHECK_EQ(out.size(0), rows);
  TVM_FFI_ICHECK_EQ(out_scale.size(0), rows);
  cudaStream_t stream = get_stream(in.device());
  quant_sorted_rows_kernel<8><<<static_cast<int>(num_blocks), kWarps * 32, 0, stream>>>(
      reinterpret_cast<__nv_bfloat16 const*>(in.data_ptr()), cols, rows, static_cast<int>(tile_m),
      reinterpret_cast<int const*>(num_tiles.data_ptr()), reinterpret_cast<int const*>(tile_limit.data_ptr()),
      static_cast<float>(scale_inv), reinterpret_cast<uint8_t*>(out.data_ptr()),
      reinterpret_cast<uint8_t*>(out_sf.data_ptr()), reinterpret_cast<float*>(out_scale.data_ptr()));
}

}  // namespace muse_vr

TVM_FFI_DLL_EXPORT_TYPED_FUNC(quant_sorted_rows, muse_vr::quant_sorted_rows);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(permute_rows, muse_vr::permute_rows);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(quant_row_blocks, muse_vr::quant_row_blocks);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(quant_regions, muse_vr::quant_regions);
