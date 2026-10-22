#pragma once

#include <cuda_bf16.h>
#include <cuda_fp16.h>

#include <cstdint>
#include <type_traits>

namespace glm5 {

// Warp-level D += A * B atom for either BF16 or FP16 operands. LDMATRIX and
// STMATRIX use the common b16 storage layout; only conversion and MMA opcode
// depend on the operand type.
template <bool kFp16>
struct HmmaM16N8K16F32B16 {
  using Element = std::conditional_t<kFp16, __half, __nv_bfloat16>;

  struct alignas(4) AFragment {
    Element element[8];
  };

  struct alignas(4) BFragment {
    Element element[4];
  };

  struct Accumulator {
    float element[4];
  };

  __device__ __forceinline__ static std::uint32_t from_bf16x2(
      std::uint32_t packed_bf16) {
    if constexpr (!kFp16) {
      return packed_bf16;
    } else {
      std::uint32_t packed_fp16;
      asm volatile(
          "{\n\t"
          ".reg .b16 bf16_low, bf16_high, fp16_low, fp16_high;\n\t"
          "mov.b32 {bf16_low, bf16_high}, %1;\n\t"
          "cvt.rn.f16.bf16 fp16_low, bf16_low;\n\t"
          "cvt.rn.f16.bf16 fp16_high, bf16_high;\n\t"
          "mov.b32 %0, {fp16_low, fp16_high};\n\t"
          "}\n"
          : "=r"(packed_fp16)
          : "r"(packed_bf16));
      return packed_fp16;
    }
  }

  __device__ __forceinline__ static std::uint32_t to_bf16x2(
      std::uint32_t packed_mma) {
    if constexpr (!kFp16) {
      return packed_mma;
    } else {
      std::uint32_t packed_bf16;
      asm volatile(
          "{\n\t"
          ".reg .b16 fp16_low, fp16_high, bf16_low, bf16_high;\n\t"
          "mov.b32 {fp16_low, fp16_high}, %1;\n\t"
          "cvt.rn.bf16.f16 bf16_low, fp16_low;\n\t"
          "cvt.rn.bf16.f16 bf16_high, fp16_high;\n\t"
          "mov.b32 %0, {bf16_low, bf16_high};\n\t"
          "}\n"
          : "=r"(packed_bf16)
          : "r"(packed_mma));
      return packed_bf16;
    }
  }

  // Match the existing BF16 rounding before placing an FP32 value in an MMA
  // fragment. This keeps attention probabilities and combine weights stable.
  __device__ __forceinline__ static Element from_bf16_rounded_float(
      float value) {
    if constexpr (!kFp16) {
      return __float2bfloat16_rn(value);
    } else {
      std::uint16_t result;
      asm volatile(
          "{\n\t"
          ".reg .b16 bf16_value;\n\t"
          "cvt.rn.bf16.f32 bf16_value, %1;\n\t"
          "cvt.rn.f16.bf16 %0, bf16_value;\n\t"
          "}\n"
          : "=h"(result)
          : "f"(value));
      return __ushort_as_half(result);
    }
  }

  __device__ __forceinline__ static std::uint32_t pack_bf16_rounded_float2(
      float low, float high) {
    std::uint32_t packed_bf16;
    asm volatile("cvt.rn.bf16x2.f32 %0, %2, %1;"
                 : "=r"(packed_bf16)
                 : "f"(low), "f"(high));
    return from_bf16x2(packed_bf16);
  }

  __device__ __forceinline__ static void clear(Accumulator& accumulator) {
#pragma unroll
    for (float& value : accumulator.element) {
      value = 0.0F;
    }
  }

  __device__ __forceinline__ static void load_a_shared(
      AFragment& fragment, void const* row_address) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    std::uint32_t raw[4];
    std::uint32_t const shared_address =
        static_cast<std::uint32_t>(__cvta_generic_to_shared(row_address));
    asm volatile(
        "ldmatrix.sync.aligned.m8n8.x4.shared.b16 "
        "{%0, %1, %2, %3}, [%4];"
        : "=r"(raw[0]), "=r"(raw[1]), "=r"(raw[2]), "=r"(raw[3])
        : "r"(shared_address)
        : "memory");
    auto* words = reinterpret_cast<std::uint32_t*>(fragment.element);
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      words[i] = raw[i];
    }
#else
    (void)fragment;
    (void)row_address;
#endif
  }

  template <bool kSourceIsBf16 = true>
  __device__ __forceinline__ static void load_b_shared(
      BFragment& fragment, void const* row_address) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    std::uint32_t raw[2];
    std::uint32_t const shared_address =
        static_cast<std::uint32_t>(__cvta_generic_to_shared(row_address));
    asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0, %1}, [%2];"
                 : "=r"(raw[0]), "=r"(raw[1])
                 : "r"(shared_address)
                 : "memory");
    auto* words = reinterpret_cast<std::uint32_t*>(fragment.element);
    if constexpr (kSourceIsBf16) {
      words[0] = from_bf16x2(raw[0]);
      words[1] = from_bf16x2(raw[1]);
    } else {
      words[0] = raw[0];
      words[1] = raw[1];
    }
#else
    (void)fragment;
    (void)row_address;
#endif
  }

  template <bool kSourceIsBf16 = true>
  __device__ __forceinline__ static void load_b_shared_transposed(
      BFragment& fragment, void const* row_address) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    std::uint32_t raw[2];
    std::uint32_t const shared_address =
        static_cast<std::uint32_t>(__cvta_generic_to_shared(row_address));
    asm volatile("ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16 {%0, %1}, [%2];"
                 : "=r"(raw[0]), "=r"(raw[1])
                 : "r"(shared_address)
                 : "memory");
    auto* words = reinterpret_cast<std::uint32_t*>(fragment.element);
    if constexpr (kSourceIsBf16) {
      words[0] = from_bf16x2(raw[0]);
      words[1] = from_bf16x2(raw[1]);
    } else {
      words[0] = raw[0];
      words[1] = raw[1];
    }
#else
    (void)fragment;
    (void)row_address;
#endif
  }

  __device__ __forceinline__ static void store_a_shared(
      AFragment const& fragment, void* row_address) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
    auto const* words = reinterpret_cast<std::uint32_t const*>(fragment.element);
    std::uint32_t const shared_address =
        static_cast<std::uint32_t>(__cvta_generic_to_shared(row_address));
    asm volatile(
        "stmatrix.sync.aligned.m8n8.x4.shared.b16 "
        "[%0], {%1, %2, %3, %4};"
        :
        : "r"(shared_address), "r"(words[0]), "r"(words[1]), "r"(words[2]),
          "r"(words[3])
        : "memory");
#else
    (void)fragment;
    (void)row_address;
#endif
  }

  __device__ __forceinline__ static void fma(
      Accumulator& accumulator, AFragment const& a_fragment,
      BFragment const& b_fragment) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    auto const* a = reinterpret_cast<std::uint32_t const*>(a_fragment.element);
    auto const* b = reinterpret_cast<std::uint32_t const*>(b_fragment.element);
    if constexpr (kFp16) {
      asm volatile(
          "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
          "{%0, %1, %2, %3}, "
          "{%4, %5, %6, %7}, "
          "{%8, %9}, "
          "{%0, %1, %2, %3};\n"
          : "+f"(accumulator.element[0]), "+f"(accumulator.element[1]),
            "+f"(accumulator.element[2]), "+f"(accumulator.element[3])
          : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]),
            "r"(b[1]));
    } else {
      asm volatile(
          "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
          "{%0, %1, %2, %3}, "
          "{%4, %5, %6, %7}, "
          "{%8, %9}, "
          "{%0, %1, %2, %3};\n"
          : "+f"(accumulator.element[0]), "+f"(accumulator.element[1]),
            "+f"(accumulator.element[2]), "+f"(accumulator.element[3])
          : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]),
            "r"(b[1]));
    }
#else
    (void)accumulator;
    (void)a_fragment;
    (void)b_fragment;
#endif
  }
};

using HmmaM16N8K16F32Bf16 = HmmaM16N8K16F32B16<false>;
using HmmaM16N8K16F32F16 = HmmaM16N8K16F32B16<true>;

static_assert(sizeof(HmmaM16N8K16F32Bf16::AFragment) == 16);
static_assert(sizeof(HmmaM16N8K16F32Bf16::BFragment) == 8);
static_assert(sizeof(HmmaM16N8K16F32Bf16::Accumulator) == 16);
static_assert(sizeof(HmmaM16N8K16F32F16::AFragment) == 16);
static_assert(sizeof(HmmaM16N8K16F32F16::BFragment) == 8);
static_assert(sizeof(HmmaM16N8K16F32F16::Accumulator) == 16);

}  // namespace glm5
