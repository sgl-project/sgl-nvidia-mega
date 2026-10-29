#pragma once

// BF16 / UE8M0 / E4M3 conversions and the mma.sync fragments of the GEMM and attention OPs.

#include "include/sm100/common/layout.hpp"
#include <cuda_bf16.h>
#include <cuda_fp8.h>

namespace m31_draft {
namespace {

__device__ __forceinline__ float bf2f(uint16_t v) { return __uint_as_float(uint32_t(v) << 16); }
__device__ __forceinline__ uint16_t f2bf(float x) {
  return __bfloat16_as_ushort(__float2bfloat16_rn(x));
}
__device__ __forceinline__ float bf16r(float x) { return bf2f(f2bf(x)); }
// A bf16 pair (lo: bits 0-15, hi: bits 16-31) as an FP32 pair.
__device__ __forceinline__ float2 bf2x2(uint32_t v) {
  return make_float2(__uint_as_float(v << 16), __uint_as_float(v & 0xffff0000u));
}
// Two FP32 values rounded to bf16 (RNE, each as __float2bfloat16_rn) in one instruction: lo in bits
// 0-15.
__device__ __forceinline__ uint32_t f2bf2(float lo, float hi) {
  uint32_t r;
  asm("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(r) : "f"(hi), "f"(lo));
  return r;
}
// One packed conversion (the same RNE bits as two __float2bfloat16_rn).
__device__ __forceinline__ uint32_t pack_bf16(float lo, float hi) { return f2bf2(lo, hi); }
// 2^(code - 127) of a UE8M0 scale code 0..254 (code 0 is the subnormal 2^-127). The product of two
// such powers of two is exactly 2^(a + b - 254), rounded like ldexpf where it under- or overflows.
__device__ __forceinline__ float ue8m0(uint32_t code) {
  return __int_as_float(code ? code << 23 : 0x00400000u);
}
// The same value from byte `byte` (0-3) of a packed word: max(code << 23, 2^-127) is code << 23 for
// codes 1..254 and the subnormal 2^-127 (0x00400000) for code 0.
__device__ __forceinline__ float ue8m0_byte(uint32_t word, int byte) {
  int const shifted = int(byte == 3 ? word >> 1 : word << (23 - 8 * byte)) & 0x7f800000;
  return __int_as_float(max(shifted, 0x00400000));
}

// MXFP8 scale of a 32-value group: the smallest power of two that brings
// amax within 448, as SGLang's MXFP8 quantizers choose it.
__device__ __forceinline__ int e8m0_from_amax(float amax, float *inv) {
  int const bits = __float_as_int(amax);
  int const e0 = ((bits >> 23) & 0xff) - 135;
  float const threshold = __int_as_float(((e0 + 135) << 23) | 0x600000);
  int e = e0 + (amax > threshold ? 1 : 0);
  e = e < -127 ? -127 : e;
  *inv = __int_as_float((127 - e) << 23);
  return e + 127;
}
// Two values to e4m3 (RNE, saturating, as SGLang's quantizer) in one instruction: lo in bits 0-7,
// hi in bits 8-15.
__device__ __forceinline__ uint32_t to_e4m3x2(float lo, float hi) {
  uint16_t r;
  asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(r) : "f"(hi), "f"(lo));
  return r;
}

__device__ __forceinline__ uint32_t e4m3x2_to_f16x2(uint32_t v) {
  uint32_t r;
  asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(r) : "h"(uint16_t(v)));
  return r;
}
__device__ __forceinline__ void mma_f16(float (&d)[4], uint32_t a0, uint32_t a1, uint32_t a2,
                                        uint32_t a3, uint32_t b0, uint32_t b1) {
  asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, "
               "{%8,%9}, {%0,%1,%2,%3};"
               : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
               : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}
// d = A (16 x 32 e4m3) x B (32 x 8 e4m3), the m16n8k32 fragments: every e4m3 pair converts
// exactly to f16 and the low / high pairs of each register form two m16n8k16 products
// with the same k assignment in A and B, so every product enters the FP32 sum once.
__device__ __forceinline__ void mma_e4m3_f16(float (&d)[4], uint32_t const (&a)[4],
                                             uint32_t const (&b)[2]) {
  d[0] = d[1] = d[2] = d[3] = 0.0f;
  mma_f16(d, e4m3x2_to_f16x2(a[0]), e4m3x2_to_f16x2(a[1]), e4m3x2_to_f16x2(a[2]),
          e4m3x2_to_f16x2(a[3]), e4m3x2_to_f16x2(b[0]), e4m3x2_to_f16x2(b[1]));
  mma_f16(d, e4m3x2_to_f16x2(a[0] >> 16), e4m3x2_to_f16x2(a[1] >> 16), e4m3x2_to_f16x2(a[2] >> 16),
          e4m3x2_to_f16x2(a[3] >> 16), e4m3x2_to_f16x2(b[0] >> 16), e4m3x2_to_f16x2(b[1] >> 16));
}
__device__ __forceinline__ void mma_bf16(float (&d)[4], uint32_t const (&a)[4], uint32_t b0,
                                         uint32_t b1) {
  asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, "
               "{%8,%9}, {%0,%1,%2,%3};"
               : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
               : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}

} // namespace
} // namespace m31_draft
