/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
// Tensor-core fragments and exact conversions. The MMA helpers follow the msa_v2
// high_tps m3_mma.cuh; the NVFP4/MXFP8 conversions reproduce the msa_v1 MegaAttention
// contract (sm100_m3_decode_attention_projection_numeric.hpp, ..._indexer_config.hpp,
// ..._attention_output_epilogue.hpp).
#pragma once

#include <cuda_fp16.h>

#include "ring_sync.cuh"

namespace attn_ring {

// ---- fragment loads ------------------------------------------------------------------
__device__ __forceinline__ uint4 ldsm_x4(uint32_t a) {
  uint4 v;
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];"
               : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w)
               : "r"(a));
  return v;
}
// Transpose 32 rows x 16 bytes into two adjacent m16n8k32 B fragments: {x,z} is the
// low n8 fragment and {y,w} the high one.
__device__ __forceinline__ uint4 ldsm_32x16_trans_b8(uint32_t a) {
  uint4 v;
  asm volatile("ldmatrix.sync.aligned.m16n16.x2.trans.shared::cta.b8 {%0,%1,%2,%3}, [%4];"
               : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w)
               : "r"(a));
  return v;
}

// A fragment of a 16-row x 32-K E4M3 block from a 128B-swizzled K-major tile laid out
// [kc][16 rows][128 B] (kc = 128-byte K chunk). k32 selects the 32-byte quarter of kc.
__device__ __forceinline__ uint4 weight_a_frag(uint32_t tile, int kc, int k32, int lane) {
  const int row = (lane & 7) + 8 * ((lane >> 3) & 1);
  const int chunk = 2 * k32 + (lane >> 4);
  return ldsm_x4(tile + uint32_t(kc) * 2048u + uint32_t(row) * 128u +
                 (uint32_t(chunk ^ (row & 7)) << 4));
}

// UE8M0 byte k (compile-time 0..3) of a packed word as an FP32 power of two: the byte shifted into
// the exponent field; a zero byte gives 0 < 0x00400000, so the unsigned max selects 2^-127.
template <int k> __device__ __forceinline__ float ue8m0_byte_to_f32(uint32_t w) {
  const uint32_t bits = (k < 3 ? (w << (23 - 8 * k)) : (w >> 1)) & 0x7f800000u;
  return __uint_as_float(max(bits, 0x00400000u));
}

// ---- E4M3 MMA through explicit f16 halves -----------------------------------------
// sm_100 has no FP8 mma.sync: ptxas lowers m16n8k32.e4m3 to F2FP unpacks + two HMMA.16816 (k
// elements 4s, 4s+1 / 16+4s, 17+4s in the first, 4s+2, 4s+3 / 18+4s, 19+4s in the second) and a
// zero-add. The helpers below issue the same two HMMAs with operands unpacked once by the caller.
__device__ __forceinline__ uint32_t e4m3x2_to_f16x2(uint32_t two) {
  uint32_t packed;
  asm("{\n\t.reg .b16 v;\n\tmov.b16 v, %1;\n\tcvt.rn.f16x2.e4m3x2 %0, v;\n\t}"
      : "=r"(packed)
      : "h"(uint16_t(two)));
  return packed;
}
__device__ __forceinline__ uint32_t f16_lo(uint32_t w) { return e4m3x2_to_f16x2(w & 0xffffu); }
__device__ __forceinline__ uint32_t f16_hi(uint32_t w) { return e4m3x2_to_f16x2(w >> 16); }
__device__ __forceinline__ void hmma16816(float (&d)[4], uint32_t a0, uint32_t a1, uint32_t a2,
                                          uint32_t a3, uint32_t b0, uint32_t b1) {
  asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3},"
               "{%4,%5,%6,%7},{%8,%9},{%0,%1,%2,%3};"
               : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
               : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}
// B operand (two e4m3x4 registers) as the f16 halves of both HMMAs.
struct HalfB {
  uint32_t lo[2], hi[2];
};
__device__ __forceinline__ HalfB half_b(uint32_t b0, uint32_t b1) {
  HalfB h;
  h.lo[0] = f16_lo(b0);
  h.lo[1] = f16_lo(b1);
  h.hi[0] = f16_hi(b0);
  h.hi[1] = f16_hi(b1);
  return h;
}
// d (+)= A . B for one K32 group: A = e4m3 m16n8k32 A fragment, B pre-unpacked.
__device__ __forceinline__ void qmma_h(float (&d)[4], uint4 a, const HalfB &b) {
  hmma16816(d, f16_lo(a.x), f16_lo(a.y), f16_lo(a.z), f16_lo(a.w), b.lo[0], b.lo[1]);
  hmma16816(d, f16_hi(a.x), f16_hi(a.y), f16_hi(a.z), f16_hi(a.w), b.hi[0], b.hi[1]);
}

// ---- E4M3 / E2M1 / UE8M0 conversions -----------------------------------------------
__device__ __forceinline__ uint16_t f32x2_to_e4m3x2(float lo, float hi) {
  uint16_t r;
  asm volatile("cvt.rn.satfinite.e4m3x2.f32 %0, %2, %1;" : "=h"(r) : "f"(lo), "f"(hi));
  return r;
}
__device__ __forceinline__ uint32_t f32x4_to_e4m3x4(float a, float b, float c, float d) {
  return uint32_t(f32x2_to_e4m3x2(a, b)) | (uint32_t(f32x2_to_e4m3x2(c, d)) << 16);
}
__device__ __forceinline__ float e4m3_to_f32(uint32_t byte) {
  uint32_t packed;
  asm volatile("{\n\t.reg .b16 v;\n\tmov.b16 v, %1;\n\tcvt.rn.f16x2.e4m3x2 %0, v;\n\t}"
               : "=r"(packed)
               : "h"(uint16_t(byte & 0xffu)));
  return __half2float(__ushort_as_half(uint16_t(packed & 0xffffu)));
}
// Four FP32 values -> four E2M1 codes (low nibble first), round-to-nearest-even.
__device__ __forceinline__ uint16_t f32x4_to_e2m1x4(float a, float b, float c, float d) {
  uint16_t r;
  asm volatile("{\n\t.reg .b8 lo, hi;\n\t"
               "cvt.rn.satfinite.e2m1x2.f32 lo, %2, %1;\n\t"
               "cvt.rn.satfinite.e2m1x2.f32 hi, %4, %3;\n\t"
               "mov.b16 %0, {lo, hi};\n\t}"
               : "=h"(r)
               : "f"(a), "f"(b), "f"(c), "f"(d));
  return r;
}
// QMUL4: four E2M1 codes times a broadcast E4M3 scale -> four E4M3 bytes (RN, satfinite).
__device__ __forceinline__ uint32_t qmul4(uint16_t packed, uint32_t broadcast_scale) {
  uint32_t out;
  asm volatile("{ .reg .b16 value;\n  mov.b16 value, %1;\n"
               "  mul.rn.satfinite.e4m3x4.e2m1x4.e4m3x4 %0, value, %2; }\n"
               : "=r"(out)
               : "h"(packed), "r"(broadcast_scale));
  return out;
}
__device__ __forceinline__ uint32_t broadcast_scale_byte(uint2 scales, int index) {
  const uint32_t word = index < 4 ? scales.x : scales.y;
  return __byte_perm(word, 0, (index & 3) * 0x1111);
}

// Dynamic MXFP8 scale code: UE8M0 round-up of amax / 448 with the frozen subnormal rule.
__device__ __forceinline__ uint32_t ue8m0_round_up(float value) {
  if (!(value > 0.f))
    return 0u;
  const uint32_t bits = __float_as_uint(value);
  const uint32_t exponent = (bits >> 23) & 0xffu, mantissa = bits & 0x7fffffu;
  uint32_t bump = mantissa != 0u ? 1u : 0u;
  if (exponent == 0u && mantissa <= 0x400000u)
    bump = 0u;
  const uint32_t code = exponent + bump;
  return code > 254u ? 254u : code;
}
// 2^(127 - code); code 0 gives +0 (an all-zero group).
__device__ __forceinline__ float ue8m0_inverse(uint32_t code) {
  if (code == 0u)
    return 0.f;
  const int e = 254 - int(code);
  return __uint_as_float(uint32_t(e > 0 ? e : 0) << 23);
}
// Attention-output MXFP8: scale = trunc_pow2(amax) / 256, stored exactly; inverse is exact.
__device__ __forceinline__ void attn_out_scale(float amax, uint32_t &code, float &inverse) {
  const float pow2 = __uint_as_float(__float_as_uint(amax) & 0x7f800000u);
  const float sf = pow2 * (1.f / 256.f);
  const uint32_t e = (__float_as_uint(sf) >> 23) & 0xffu;
  code = sf != 0.f ? e : 0u;
  inverse = sf != 0.f ? __uint_as_float(0x7f000000u - __float_as_uint(sf)) : 0.f;
}

// 2^x as the MegaAttention softmax evaluates it (ex2.approx.ftz).
__device__ __forceinline__ float ex2_approx(float x) {
  float y;
  asm("ex2.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x));
  return y;
}

__device__ __forceinline__ float bf16_round(float v) {
  return __bfloat162float(__float2bfloat16_rn(v));
}
__device__ __forceinline__ float bf16_lo(uint32_t w) { return __uint_as_float(w << 16); }
__device__ __forceinline__ float bf16_hi(uint32_t w) { return __uint_as_float(w & 0xffff0000u); }

} // namespace attn_ring
