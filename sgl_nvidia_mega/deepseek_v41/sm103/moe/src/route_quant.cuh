// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
//
// MXFP8 quant of the routed MoE input (slot gate_route_quant): one BF16 row of 5120 into
// E4M3 values and UE8M0 scales in the linear [rows, 160] layout, as SGLang's
// quantize_routed_input asks FlashInfer's mxfp8_quantize (cute-dsl, linear) for it. Four
// adjacent lanes hold the four 8-element fragments of a 32-element scale block.
//
// Bit for bit FlashInfer's: the block's absolute maximum over BF16 pairs (sign bits
// cleared, max.bf16x2, then the larger half as FP32, then two xor shuffles; a maximum is
// independent of the order it is taken in), times 1 / 448, rounded up to a power of two
// (UE8M0), the inverse scale built from the exponent, each element times it, clamped to
// +-448 and converted with cvt.rn.satfinite. The arithmetic is inline PTX without .ftz, so
// the router build's options (--ftz for the Triton reference) do not reach it.

#pragma once

namespace route_quant {

constexpr int kHidden = 5120;
constexpr int kFragment = 8;                     // BF16 elements a lane converts
constexpr int kFragments = kHidden / kFragment;  // 640 a row
constexpr int kScaleCols = kHidden / 32;         // 160 a row

__device__ __forceinline__ unsigned int AbsMax4(uint4 v) {
  unsigned int a0, a1, a2, a3, m01, m23, m;
  asm("and.b32 %0, %1, 0x7FFF7FFF;" : "=r"(a0) : "r"(v.x));
  asm("and.b32 %0, %1, 0x7FFF7FFF;" : "=r"(a1) : "r"(v.y));
  asm("and.b32 %0, %1, 0x7FFF7FFF;" : "=r"(a2) : "r"(v.z));
  asm("and.b32 %0, %1, 0x7FFF7FFF;" : "=r"(a3) : "r"(v.w));
  asm("max.bf16x2 %0, %1, %2;" : "=r"(m01) : "r"(a0), "r"(a1));
  asm("max.bf16x2 %0, %1, %2;" : "=r"(m23) : "r"(a2), "r"(a3));
  asm("max.bf16x2 %0, %1, %2;" : "=r"(m) : "r"(m01), "r"(m23));
  return m;
}

__device__ __forceinline__ float HalvesMax(unsigned int x) {
  float r;
  asm("{\n"
      ".reg .b32 lo, hi;\n"
      ".reg .f32 f0, f1;\n"
      "and.b32 lo, %1, 0xFFFF;\n"
      "shr.b32 hi, %1, 16;\n"
      "shl.b32 lo, lo, 16;\n"
      "shl.b32 hi, hi, 16;\n"
      "mov.b32 f0, lo;\n"
      "mov.b32 f1, hi;\n"
      "max.f32 %0, f0, f1;\n"
      "}\n"
      : "=f"(r)
      : "r"(x));
  return r;
}

__device__ __forceinline__ float MaxF32(float a, float b) {
  float r;
  asm("max.f32 %0, %1, %2;" : "=f"(r) : "f"(a), "f"(b));
  return r;
}

__device__ __forceinline__ float MulRn(float a, float b) {
  float r;
  asm("mul.rn.f32 %0, %1, %2;" : "=f"(r) : "f"(a), "f"(b));
  return r;
}

__device__ __forceinline__ unsigned int Ue8m0(float value) {
  unsigned int r;
  asm("{\n"
      ".reg .pred p_zero, p_has_mant, p_exp_zero, p_tiny_sub, p_ovf;\n"
      ".reg .u32 bits, exp_biased, mantissa, bump, result;\n"
      "setp.le.f32 p_zero, %1, 0f00000000;\n"
      "mov.b32 bits, %1;\n"
      "shr.b32 exp_biased, bits, 23;\n"
      "and.b32 exp_biased, exp_biased, 255;\n"
      "and.b32 mantissa, bits, 0x7FFFFF;\n"
      "setp.ne.u32 p_has_mant, mantissa, 0;\n"
      "selp.u32 bump, 1, 0, p_has_mant;\n"
      "setp.eq.u32 p_exp_zero, exp_biased, 0;\n"
      "setp.le.u32 p_tiny_sub, mantissa, 0x400000;\n"
      "and.pred p_tiny_sub, p_exp_zero, p_tiny_sub;\n"
      "@p_tiny_sub mov.u32 bump, 0;\n"
      "add.u32 result, exp_biased, bump;\n"
      "setp.gt.u32 p_ovf, result, 254;\n"
      "selp.u32 result, 254, result, p_ovf;\n"
      "selp.u32 %0, 0, result, p_zero;\n"
      "}\n"
      : "=r"(r)
      : "f"(value));
  return r;
}

__device__ __forceinline__ float InvScale(unsigned int ue8m0) {
  float r;
  asm("{\n"
      ".reg .s32 new_exp;\n"
      ".reg .b32 float_bits;\n"
      ".reg .pred p_zero;\n"
      "setp.eq.u32 p_zero, %1, 0;\n"
      "sub.s32 new_exp, 254, %1;\n"
      "max.s32 new_exp, new_exp, 0;\n"
      "shl.b32 float_bits, new_exp, 23;\n"
      "mov.b32 %0, float_bits;\n"
      "@p_zero mov.b32 %0, 0;\n"
      "}\n"
      : "=f"(r)
      : "r"(ue8m0));
  return r;
}

__device__ __forceinline__ unsigned int Fp8Pair(unsigned int bf2, float inv) {
  unsigned int r;
  asm("{\n"
      ".reg .b32 lo, hi;\n"
      ".reg .f32 f0, f1;\n"
      ".reg .b16 fp8_pair;\n"
      "and.b32 lo, %1, 0xFFFF;\n"
      "shr.b32 hi, %1, 16;\n"
      "shl.b32 lo, lo, 16;\n"
      "shl.b32 hi, hi, 16;\n"
      "mov.b32 f0, lo;\n"
      "mov.b32 f1, hi;\n"
      "mul.f32 f0, f0, %2;\n"
      "mul.f32 f1, f1, %2;\n"
      "min.f32 f0, f0, 0f43E00000;\n"
      "max.f32 f0, f0, 0fC3E00000;\n"
      "min.f32 f1, f1, 0f43E00000;\n"
      "max.f32 f1, f1, 0fC3E00000;\n"
      "cvt.rn.satfinite.e4m3x2.f32 fp8_pair, f1, f0;\n"
      "cvt.u32.u16 %0, fp8_pair;\n"
      "}\n"
      : "=r"(r)
      : "r"(bf2), "f"(inv));
  return r;
}

// Load fragment f (< kFragments) of `row` (16 bytes).
__device__ __forceinline__ uint4 Load(const unsigned short* x, int row, int f) {
  return *reinterpret_cast<const uint4*>(x + static_cast<long long>(row) * kHidden + f * kFragment);
}

// Quantize fragment f of `row` from its loaded words; every lane of the warp calls it
// (the shuffles), the four lanes of a scale block on its four fragments.
__device__ __forceinline__ void Store(uint4 v, unsigned char* __restrict__ q, unsigned char* __restrict__ scales,
                                      int row, int f) {
  float amax = HalvesMax(AbsMax4(v));
  amax = MaxF32(amax, __shfl_xor_sync(0xffffffffU, amax, 1));
  amax = MaxF32(amax, __shfl_xor_sync(0xffffffffU, amax, 2));
  const unsigned int scale = Ue8m0(MulRn(amax, __uint_as_float(0x3B124925U)));  // fp32(1 / 448)
  const float inv = InvScale(scale);
  const unsigned int p01 = Fp8Pair(v.x, inv) & 0xFFFFU;
  const unsigned int p23 = Fp8Pair(v.y, inv) & 0xFFFFU;
  const unsigned int p45 = Fp8Pair(v.z, inv) & 0xFFFFU;
  const unsigned int p67 = Fp8Pair(v.w, inv) & 0xFFFFU;
  uint2 packed;
  packed.x = p01 | (p23 << 16);
  packed.y = p45 | (p67 << 16);
  *reinterpret_cast<uint2*>(q + static_cast<long long>(row) * kHidden + f * kFragment) = packed;
  if ((f & 3) == 0) scales[row * kScaleCols + (f >> 2)] = static_cast<unsigned char>(scale);
}

}  // namespace route_quant
