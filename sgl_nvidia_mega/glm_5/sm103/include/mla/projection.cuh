#pragma once

// Native FP8 weight tiles, RMS normalization and projection fragments.
// CTA ownership, K128/K64 scale groups and MMA shapes are preserved.
#include <mla/config.cuh>
#include <common/norm.cuh>
#include <cutlass/arch/reg_reconfig.h>

namespace glm5::mla::detail {

// Native [out,in] E4M3 tiles are staged only in shared memory.  The
// row-major LDMATRIX/K pairing matches MegaMoE's native FP8 path.
__device__ __forceinline__ uint2 load_native_fp8_rows(
    std::uint8_t const* address, bool transpose = false) {
  uint2 rows;
  std::uint32_t const shared_address =
      static_cast<std::uint32_t>(__cvta_generic_to_shared(address));
  if (transpose) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16 {%0, %1}, [%2];"
                 : "=r"(rows.x), "=r"(rows.y)
                 : "r"(shared_address) : "memory");
  } else {
    asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0, %1}, [%2];"
                 : "=r"(rows.x), "=r"(rows.y)
                 : "r"(shared_address) : "memory");
  }
  return rows;
}

__device__ __forceinline__ void decode_native_fp8_rows(
    uint2 rows, Hmma::AFragment& fragment) {
  auto* words = reinterpret_cast<std::uint32_t*>(fragment.element);
  words[0] = e4m3x2_to_mma_x2(static_cast<std::uint16_t>(rows.x));
  words[1] = e4m3x2_to_mma_x2(static_cast<std::uint16_t>(rows.y));
  words[2] = e4m3x2_to_mma_x2(static_cast<std::uint16_t>(rows.x >> 16));
  words[3] = e4m3x2_to_mma_x2(static_cast<std::uint16_t>(rows.y >> 16));
}

__device__ __forceinline__ uint4 load_native_fp8_row_pair(
    std::uint8_t const* address) {
  uint4 rows;
  std::uint32_t const shared_address =
      static_cast<std::uint32_t>(__cvta_generic_to_shared(address));
  asm volatile(
      "ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];"
      : "=r"(rows.x), "=r"(rows.y), "=r"(rows.z), "=r"(rows.w)
      : "r"(shared_address) : "memory");
  return rows;
}

// Read packed scales in their native layout and pad transfers
// to match FP32 scale barrier accounting.
template <int kColumns>
__device__ __forceinline__ void issue_native_scales(
    NativeWeightSource const& source, int row, int first_column,
    std::uint8_t* destination, std::uint64_t* ready) {
  static_assert(kColumns % 4 == 0);
  if (source.packed_scale_col_stride != 0) {
#pragma unroll
    for (int packet = 0; packet < kColumns / 4; ++packet) {
      auto const* pointer = reinterpret_cast<std::uint32_t const*>(source.scales) +
          row + (first_column / 4 + packet) * source.packed_scale_col_stride;
      cute::SM90_BULK_COPY_G2S::copy(
          pointer, ready, destination + packet * 16, 16);
    }
  } else {
    auto const* pointer =
        source.scales + (row / 128) * source.scale_row_stride + first_column;
    cute::SM90_BULK_COPY_G2S::copy(pointer, ready, destination,
                                 kColumns * sizeof(float));
  }
}

__device__ __forceinline__ float staged_native_scale(
    NativeWeightSource const& source, std::uint8_t const* data, int column) {
  if (source.packed_scale_col_stride != 0) {
    std::uint32_t const word =
        reinterpret_cast<std::uint32_t const*>(data)[(column / 4) * 4];
    return __uint_as_float(((word >> ((column & 3) * 8)) & 255U) << 23);
  }
  return reinterpret_cast<float const*>(data)[column];
}

template <int kRows, int kPageK>
struct NativeRowWeightAccessor {
  static constexpr int kMatrixBytes = kRows * kPageK;
  static constexpr int kScaleColumns = kPageK / 128;
  static constexpr int kScaleRows = kRows == 48 ? 3 : 1;
  static constexpr int kPageBytes =
      kMatrixBytes + kScaleRows * kScaleColumns * sizeof(float);

  NativeWeightSource const& source;
  int row;

  __device__ __forceinline__ void issue_page(
      int page, std::uint8_t* destination, std::uint64_t* ready) const {
    // Tile weights along K to avoid shared-memory bank aliasing.
    cute::SM90_TMA_LOAD_3D::copy(
        &source.tensor_map, ready,
        static_cast<std::uint64_t>(cute::TMA::CacheHintSm100::EVICT_NORMAL),
        destination, 0, row, page * kScaleColumns);
#pragma unroll
    for (int m_tile = 0; m_tile < kScaleRows; ++m_tile) {
      issue_native_scales<kScaleColumns>(
          source, row + m_tile * 16, page * kScaleColumns,
          destination + kMatrixBytes + m_tile * kScaleColumns * sizeof(float), ready);
    }
    // FP8 and scale transactions share the ready barrier. Neither payload
    // blocks the producer warp while the four-stage pipeline is filling.
  }

  __device__ __forceinline__ void load_page(
      std::uint8_t const* page_base, int m_tile, int page_k_tile, int lane,
      Hmma::AFragment& fragment) const {
    int const swizzled = (page_k_tile / 8) * (kRows * 128) +
                         (m_tile * 16 + (lane & 15)) * 128 +
                         ((page_k_tile ^ lane) & 7) * 16;
    decode_native_fp8_rows(load_native_fp8_rows(page_base + swizzled), fragment);
  }

  // One x4 load supplies two adjacent M16 fragments, preserving their
  // K pairing while sharing address calculation and warp load issue.
  __device__ __forceinline__ void load_pair(
      std::uint8_t const* page_base, int m_tile, int page_k_tile, int lane,
      Hmma::AFragment (&fragments)[2]) const {
    int const swizzled = (page_k_tile / 8) * (kRows * 128) +
                         (m_tile * 16 + (lane & 31)) * 128 +
                         ((page_k_tile ^ lane) & 7) * 16;
    uint4 const raw = load_native_fp8_row_pair(page_base + swizzled);
    decode_native_fp8_rows(uint2{raw.x, raw.y}, fragments[0]);
    decode_native_fp8_rows(uint2{raw.z, raw.w}, fragments[1]);
  }

  __device__ __forceinline__ float scale_page(
      std::uint8_t const* page_base, int m_tile, int page_k_tile) const {
    int const scale_row = kScaleRows == 1 ? 0 : m_tile;
    return staged_native_scale(source, page_base + kMatrixBytes,
                               scale_row * kScaleColumns + page_k_tile / 8);
  }
};

struct ProjXWeightAccessor : NativeRowWeightAccessor<32, 1024> {
  __device__ __forceinline__ ProjXWeightAccessor(
      Glm5MegaMlaArgs const& args, int cta)
      : NativeRowWeightAccessor{
            cta < 64 ? args.q_a_weight : args.kv_a_weight,
            (cta < 64 ? cta : cta - 64) * 32} {}
};

struct ProjQWeightAccessor : NativeRowWeightAccessor<32, 1024> {
  __device__ __forceinline__ ProjQWeightAccessor(
      Glm5MegaMlaArgs const& args, int cta)
      : NativeRowWeightAccessor{
            args.q_b_weight,
            cta * 32 < kQueryHeads * kQkNopeDim
                ? (cta * 32 / kQkNopeDim) * 256 + cta * 32 % kQkNopeDim
                : ((cta * 32 - kQueryHeads * kQkNopeDim) / kRopeDim) * 256 +
                      kQkNopeDim +
                      (cta * 32 - kQueryHeads * kQkNopeDim) % kRopeDim} {}
};

struct QAbsorbWeightAccessor {
  static constexpr int kMatrixBytes = kQAbsorbMatrixBytes;
  static constexpr int kPageBytes = kMatrixBytes + 3 * 4 * sizeof(float);
  NativeWeightSource const& source;
  int head;
  int latent_column;

  __device__ __forceinline__ QAbsorbWeightAccessor(
      Glm5MegaMlaArgs const& args, int cta)
      : source(args.k_absorb_weight), head(cta / 8),
        latent_column((cta & 7) * 64) {}

  __device__ __forceinline__ void issue_page(
      int, std::uint8_t* destination, std::uint64_t* ready) const {
    cute::SM90_TMA_LOAD_2D::copy(
        &source.tensor_map, ready,
        static_cast<std::uint64_t>(cute::TMA::CacheHintSm100::EVICT_NORMAL),
        destination, latent_column, head * (kQkNopeDim + kHeadDim));
#pragma unroll
    for (int block = 0; block < 3; ++block) {
      issue_native_scales<4>(
          source, head * (kQkNopeDim + kHeadDim) + block * 64, 0,
          destination + kMatrixBytes + block * 4 * sizeof(float), ready);
    }
  }

  __device__ __forceinline__ void load_page(
      std::uint8_t const* page_base, int m_tile, int page_k_tile, int lane,
      Hmma::AFragment& fragment) const {
    int const swizzled = (page_k_tile * 16 + (lane & 15)) * 64 +
                         ((m_tile ^ (lane >> 1)) & 3) * 16;
    uint2 const raw = load_native_fp8_rows(page_base + swizzled, true);
    // Transposed b16 loads pair adjacent latent bytes. Restore K pairs
    // with PRMT; the cheap M permutation is undone at the output store.
    std::uint32_t const even_odd_low = __byte_perm(raw.x, raw.x, 0x3120);
    std::uint32_t const even_odd_high = __byte_perm(raw.y, raw.y, 0x3120);
    auto* words = reinterpret_cast<std::uint32_t*>(fragment.element);
    words[0] = e4m3x2_to_mma_x2(static_cast<std::uint16_t>(even_odd_low));
    words[1] = e4m3x2_to_mma_x2(static_cast<std::uint16_t>(even_odd_low >> 16));
    words[2] = e4m3x2_to_mma_x2(static_cast<std::uint16_t>(even_odd_high));
    words[3] = e4m3x2_to_mma_x2(static_cast<std::uint16_t>(even_odd_high >> 16));
  }

  __device__ __forceinline__ float scale_page(
      std::uint8_t const* page_base, int, int page_k_tile) const {
    return staged_native_scale(source, page_base + kMatrixBytes,
                               (page_k_tile / 4) * 4 + latent_column / 128);
  }

  __device__ __forceinline__ static int output_row(int mma_row) {
    return 2 * (mma_row & 7) + (mma_row >> 3);
  }
};

struct OAbsorbWeightAccessor : NativeRowWeightAccessor<32, 512> {
  __device__ __forceinline__ OAbsorbWeightAccessor(
      Glm5MegaMlaArgs const& args, int cta)
      : NativeRowWeightAccessor{
            args.v_absorb_weight,
            (cta / 8) * (kQkNopeDim + kHeadDim) + kQkNopeDim +
                (cta & 7) * 32} {}
};

struct OProjWeightAccessor : NativeRowWeightAccessor<48, 512> {
  __device__ __forceinline__ OProjWeightAccessor(
      Glm5MegaMlaArgs const& args, int cta)
      : NativeRowWeightAccessor{args.o_proj_weight, cta * 48} {}
};

__device__ __forceinline__ int clamp_int(int value, int lower, int upper) {
  return value < lower ? lower : value > upper ? upper : value;
}

__device__ __forceinline__ int load_position(void const* positions, int element_bytes,
                                             int token) {
  if (element_bytes == static_cast<int>(sizeof(std::int64_t))) {
    return static_cast<int>(static_cast<std::int64_t const*>(positions)[token]);
  }
  return static_cast<std::int32_t const*>(positions)[token];
}

__device__ __forceinline__ void configure_warp_specialized_registers() {
  int const warp = static_cast<int>(threadIdx.x) >> 5;
  if (warp >= 8) {
    cutlass::arch::warpgroup_reg_dealloc<72>();
  } else {
    cutlass::arch::warpgroup_reg_alloc<216>();
  }
}

template <class Element>
__device__ __forceinline__ Element b16_from_bits(std::uint16_t bits) {
  if constexpr (std::is_same_v<Element, __nv_bfloat16>) {
    return __ushort_as_bfloat16(bits);
  } else {
    static_assert(std::is_same_v<Element, __half>);
    return __ushort_as_half(bits);
  }
}

template <class Element>
__device__ __forceinline__ float b16_to_float(Element value) {
  if constexpr (std::is_same_v<Element, __nv_bfloat16>) {
    return bf16_to_float(value);
  } else {
    static_assert(std::is_same_v<Element, __half>);
    return __half2float(value);
  }
}

template <class Element>
__device__ __forceinline__ float b16x2_square_sum(std::uint32_t packed, float sum) {
  if constexpr (std::is_same_v<Element, __nv_bfloat16> && kTokens == 8) {
    asm volatile(
        "{\n\t"
        ".reg .b16 low, high;\n\t"
        "mov.b32 {low, high}, %1;\n\t"
        "fma.rn.f32.bf16 %0, low, low, %0;\n\t"
        "fma.rn.f32.bf16 %0, high, high, %0;\n\t"
        "}\n"
        : "+f"(sum) : "r"(packed));
    return sum;
  } else {
    float const low = b16_to_float(b16_from_bits<Element>(static_cast<std::uint16_t>(packed)));
    float const high =
        b16_to_float(b16_from_bits<Element>(static_cast<std::uint16_t>(packed >> 16)));
    return fmaf(high, high, fmaf(low, low, sum));
  }
}

__device__ __forceinline__ std::uint32_t pack_internal_x2(float low, float high) {
#if MEGAMLA_USE_NATIVE_BF16X2_E4M3X2_CVT
  return pack_bf16x2(low, high);
#else
  std::uint32_t packed;
  asm volatile("cvt.rn.f16x2.f32 %0, %2, %1;"
               : "=r"(packed)
               : "f"(low), "f"(high));
  return packed;
#endif
}

template <class Element>
__device__ __forceinline__ Element float_to_b16(float value) {
  if constexpr (std::is_same_v<Element, __nv_bfloat16>) {
    return float_to_bf16(value);
  } else {
    static_assert(std::is_same_v<Element, __half>);
    return __float2half_rn(value);
  }
}

__device__ __forceinline__ InternalActivation float_to_internal(float value) {
  return float_to_b16<InternalActivation>(value);
}

template <class InputElement>
__device__ __forceinline__ std::uint32_t normalize_to_internal_x2(
    std::uint32_t packed, float gamma_low, float gamma_high, float inverse) {
  float const low =
      b16_to_float(b16_from_bits<InputElement>(static_cast<std::uint16_t>(packed)));
  float const high =
      b16_to_float(b16_from_bits<InputElement>(static_cast<std::uint16_t>(packed >> 16)));
  return pack_internal_x2(low * gamma_low * inverse, high * gamma_high * inverse);
}

template <int kElements, int kRowPaddingElements, class InputElement>
__device__ __forceinline__ void stage_rms_normalized_activation(InputElement const* input,
                                                                int input_stride,
                                                                NormGain const* gamma,
                                                                InternalActivation* shared_activation,
                                                                float* scratch) {
  static_assert(kElements % (kConsumerThreads * 8) == 0);
  static_assert(kRowPaddingElements % 8 == 0);
  constexpr int kVectorsPerThread = kElements / (kConsumerThreads * 8);
  constexpr int kSharedRowStride = kElements + kRowPaddingElements;
  int const tid = static_cast<int>(threadIdx.x);
  int const warp = tid >> 5;
  int const lane = tid & 31;

  uint4 values[kTokens][kVectorsPerThread];
  float4 gamma_low[kVectorsPerThread];
  float4 gamma_high[kVectorsPerThread];
#pragma unroll
  for (int token = 0; token < kTokens; ++token) {
#pragma unroll
    for (int vector = 0; vector < kVectorsPerThread; ++vector) {
      int const element = (vector * kConsumerThreads + tid) * 8;
      values[token][vector] =
          *reinterpret_cast<uint4 const*>(input + token * input_stride + element);
    }
  }
#pragma unroll
  for (int token = 0; token < kTokens; ++token) {
    float even_sum = 0.0F;
    float odd_sum = 0.0F;
#pragma unroll
    for (int vector = 0; vector < kVectorsPerThread; ++vector) {
      uint4 const value = values[token][vector];
      even_sum = b16x2_square_sum<InputElement>(value.x, even_sum);
      odd_sum = b16x2_square_sum<InputElement>(value.y, odd_sum);
      even_sum = b16x2_square_sum<InputElement>(value.z, even_sum);
      odd_sum = b16x2_square_sum<InputElement>(value.w, odd_sum);
    }
    float sum = warp_sum(even_sum + odd_sum);
    if (lane == 0) {
      scratch[token * 8 + warp] = sum;
    }
  }
#pragma unroll
  for (int vector = 0; vector < kVectorsPerThread; ++vector) {
    int const element = (vector * kConsumerThreads + tid) * 8;
    gamma_low[vector] = load_norm4(gamma + element);
    gamma_high[vector] = load_norm4(gamma + element + 4);
  }
  consumer_group_sync();

  float inverse[kTokens];
  // Reduce each token in one lane, then broadcast its RMS inverse.
  // Keep the summation order fixed.
  float lane_inverse = 0.0F;
  if (lane < kTokens) {
    float sum = 0.0F;
#pragma unroll
    for (int source_warp = 0; source_warp < 8; ++source_warp) {
      sum += scratch[lane * 8 + source_warp];
    }
    lane_inverse = rsqrtf(sum / static_cast<float>(kElements) + kRmsEpsilon);
  }
#pragma unroll
  for (int token = 0; token < kTokens; ++token) {
    inverse[token] = __shfl_sync(kFullWarpMask, lane_inverse, token);
  }

#pragma unroll
  for (int vector = 0; vector < kVectorsPerThread; ++vector) {
    int const element = (vector * kConsumerThreads + tid) * 8;
    float4 const low_gamma = gamma_low[vector];
    float4 const high_gamma = gamma_high[vector];
#pragma unroll
    for (int token = 0; token < kTokens; ++token) {
      uint4 const value = values[token][vector];
      uint4 normalized;
      normalized.x =
          normalize_to_internal_x2<InputElement>(value.x, low_gamma.x, low_gamma.y, inverse[token]);
      normalized.y =
          normalize_to_internal_x2<InputElement>(value.y, low_gamma.z, low_gamma.w, inverse[token]);
      normalized.z = normalize_to_internal_x2<InputElement>(value.z, high_gamma.x, high_gamma.y,
                                                             inverse[token]);
      normalized.w = normalize_to_internal_x2<InputElement>(value.w, high_gamma.z, high_gamma.w,
                                                             inverse[token]);
      *reinterpret_cast<uint4*>(shared_activation + token * kSharedRowStride + element) =
          normalized;
    }
  }
  consumer_group_sync();
}

template <int kElements, int kRowPaddingElements = 0>
__device__ __forceinline__ Hmma::BFragment load_q_absorb_b_fragment(
    InternalActivation const* shared_activation, int k_base, int lane) {
  Hmma::BFragment fragment{};
  int const token = lane >> 2;
  int const lane_in_group = lane & 3;
  if (token < kTokens) {
    constexpr int kSharedRowStride = kElements + kRowPaddingElements;
    auto const* source = reinterpret_cast<std::uint32_t const*>(
        shared_activation + token * kSharedRowStride + k_base + 2 * lane_in_group);
    auto* destination = reinterpret_cast<std::uint32_t*>(fragment.element);
    destination[0] = source[0];
    destination[1] = source[4];
  }
  return fragment;
}

template <int kElements, int kRowPaddingElements = 0>
__device__ __forceinline__ Hmma::BFragment load_projection_b_fragment(
    InternalActivation const* shared_activation, int k_base, int lane) {
  Hmma::BFragment fragment{};
  int const token = lane >> 2;
  int const lane_in_group = lane & 3;
  if (token < kTokens) {
    constexpr int kSharedRowStride = kElements + kRowPaddingElements;
    auto const* source = reinterpret_cast<std::uint32_t const*>(
        shared_activation + token * kSharedRowStride + k_base + 4 * lane_in_group);
    auto* destination = reinterpret_cast<std::uint32_t*>(fragment.element);
    destination[0] = source[0];
    destination[1] = source[1];
  }
  return fragment;
}

template <int kMTiles, int kActivationElements, int kRowPaddingElements = 0,
          class WeightAccessor>
__device__ __forceinline__ void accumulate_projection_k64_page(
    WeightAccessor const& weight, std::uint8_t const* page_base, int page,
    InternalActivation const* shared_activation, Hmma::Accumulator (&accumulated)[kMTiles]) {
  int const warp = static_cast<int>(threadIdx.x) >> 5;
  int const lane = static_cast<int>(threadIdx.x) & 31;
  Hmma::Accumulator block[kMTiles];
#pragma unroll
  for (int m_tile = 0; m_tile < kMTiles; ++m_tile) {
    Hmma::clear(block[m_tile]);
  }

#pragma unroll
  for (int k16 = 0; k16 < 4; ++k16) {
    int const page_k_tile = warp * 4 + k16;
    int const k_base =
        (kActivationElements == 512 ? 0 : page * 512) + page_k_tile * 16;
    Hmma::BFragment const b_fragment =
        load_projection_b_fragment<kActivationElements, kRowPaddingElements>(shared_activation,
                                                                              k_base, lane);
#pragma unroll
    for (int m_tile = 0; m_tile + 1 < kMTiles; m_tile += 2) {
      Hmma::AFragment fragments[2];
      weight.load_pair(page_base, m_tile, page_k_tile, lane, fragments);
      Hmma::fma(block[m_tile], fragments[0], b_fragment);
      Hmma::fma(block[m_tile + 1], fragments[1], b_fragment);
    }
    if constexpr (kMTiles % 2 != 0) {
      Hmma::AFragment fragment;
      weight.load_page(page_base, kMTiles - 1, page_k_tile, lane, fragment);
      Hmma::fma(block[kMTiles - 1], fragment, b_fragment);
    }
  }

#pragma unroll
  for (int m_tile = 0; m_tile < kMTiles; ++m_tile) {
    float const scale = weight.scale_page(page_base, m_tile, warp * 4);
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      accumulated[m_tile].element[i] =
          fmaf(block[m_tile].element[i], scale, accumulated[m_tile].element[i]);
    }
  }
}

template <int kRows, class Element>
__device__ __forceinline__ void store_reduced_projection_global(Element const* shared_reduced,
                                                                Element* output,
                                                                int output_stride,
                                                                int output_row_base) {
  static_assert(kRows % 8 == 0);
  constexpr int kVectorsPerToken = kRows / 8;
  int const tid = static_cast<int>(threadIdx.x);
  if (tid < kTokens * kVectorsPerToken) {
    int const token = tid / kVectorsPerToken;
    int const vector = tid % kVectorsPerToken;
    uint4 const value =
        *reinterpret_cast<uint4 const*>(shared_reduced + token * kRows + vector * 8);
    *reinterpret_cast<uint4*>(output + token * output_stride + output_row_base + vector * 8) =
        value;
  }
}

__device__ __forceinline__ void bulk_store_proj_x(InternalActivation const* shared_reduced,
                                                  InternalActivation* output,
                                                  int output_row_base) {
  if (threadIdx.x == 0) {
    publish_shared_writes_to_async_proxy();
#pragma unroll
    for (int token = 0; token < kTokens; ++token) {
      copy_bulk_shared_to_global(output + token * kProjXRows + output_row_base,
                                 shared_reduced + token * 32,
                                 32 * sizeof(InternalActivation));
    }
    commit_bulk_copy_group();
    wait_bulk_copy_group();
  }
}

template <int kPages, class WeightAccessor>
__device__ __forceinline__ void produce_projection_pages(WeightAccessor const& weight,
                                                         PipelineControl* pipeline,
                                                         std::uint8_t* dynamic_storage,
                                                         int task_base) {
  int const tid = static_cast<int>(threadIdx.x);
#pragma unroll
  for (int page = 0; page < kPages; ++page) {
    int const task = task_base + page;
    int const stage = task & (kPipelineStages - 1);
    producer_wait_stage(pipeline, task);
    if (tid == kConsumerThreads) {
      producer_expect_transaction(pipeline->ready + stage, WeightAccessor::kPageBytes);
      weight.issue_page(page, pipeline_stage(dynamic_storage, task, kAttentionStageBytes),
                        pipeline->ready + stage);
    }
    // Lane 0 of each producer warp signals readiness. The copy leader
    // issues expect_tx before its own arrival.
    producer_arrive_stage(pipeline, task);
  }
}

__device__ __forceinline__ void copy_8b_g2s(void const* global_source, void* shared_destination) {
  std::uint32_t const shared_address =
      static_cast<std::uint32_t>(__cvta_generic_to_shared(shared_destination));
  std::uint64_t const global_address = reinterpret_cast<std::uint64_t>(global_source);
  asm volatile("cp.async.ca.shared.global [%0], [%1], 8;"
               :
               : "r"(shared_address), "l"(global_address)
               : "memory");
}

__device__ __forceinline__ std::uint64_t* phase_state(Glm5MegaMlaArgs const& args, Phase phase) {
  return workspace_pointer<std::uint64_t>(args, WorkspaceLayout::kPhaseStates) +
         static_cast<int>(phase);
}

template <int kActiveCtas>
__device__ __forceinline__ std::uint32_t contiguous_phase_contribution(int cta) {
  static_assert(kActiveCtas > 0 && kActiveCtas <= 256);
  constexpr int kBaseContribution = 256 / kActiveCtas;
  constexpr int kExtraContributionCtas = 256 % kActiveCtas;
  if (cta >= kActiveCtas) {
    return 0U;
  }
  return static_cast<std::uint32_t>(kBaseContribution + (cta < kExtraContributionCtas));
}

// Page-only ProjX activation for MTP718; MTP314 stages all 6144 columns.
// Each specialization defines its indexing and reduction order.
#if GLM5_MEGAMLA_TOKENS == 4
template <int kMTiles, int kActivationElements, int kRowPaddingElements = 0,
          class WeightAccessor>
__device__ __forceinline__ void accumulate_projection_k128_page(
    WeightAccessor const& weight, std::uint8_t const* page_base, int page,
    InternalActivation const* shared_activation, Hmma::Accumulator (&accumulated)[kMTiles]) {
  int const warp = static_cast<int>(threadIdx.x) >> 5;
  int const lane = static_cast<int>(threadIdx.x) & 31;
  Hmma::Accumulator block[kMTiles];
#pragma unroll
  for (int m_tile = 0; m_tile < kMTiles; ++m_tile) {
    Hmma::clear(block[m_tile]);
  }

#pragma unroll
  for (int k16 = 0; k16 < 8; ++k16) {
    int const page_k_tile = warp * 8 + k16;
    int const k_base = page * kProjPageK + page_k_tile * 16;
    Hmma::BFragment const b_fragment =
        load_projection_b_fragment<kActivationElements, kRowPaddingElements>(shared_activation,
                                                                              k_base, lane);
#pragma unroll
    for (int m_tile = 0; m_tile + 1 < kMTiles; m_tile += 2) {
      Hmma::AFragment fragments[2];
      weight.load_pair(page_base, m_tile, page_k_tile, lane, fragments);
      Hmma::fma(block[m_tile], fragments[0], b_fragment);
      Hmma::fma(block[m_tile + 1], fragments[1], b_fragment);
    }
    if constexpr (kMTiles % 2 != 0) {
      Hmma::AFragment fragment;
      weight.load_page(page_base, kMTiles - 1, page_k_tile, lane, fragment);
      Hmma::fma(block[kMTiles - 1], fragment, b_fragment);
    }
  }

#pragma unroll
  for (int m_tile = 0; m_tile < kMTiles; ++m_tile) {
    float const scale = weight.scale_page(page_base, m_tile, warp * 8);
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      accumulated[m_tile].element[i] =
          fmaf(block[m_tile].element[i], scale, accumulated[m_tile].element[i]);
    }
  }
}

template <int kMTiles, bool kStoreFullFragment, class OutputElement>
__device__ __forceinline__ void reduce_projection_warp_partials(
    Hmma::Accumulator const (&accumulated)[kMTiles], float* shared_partials,
    OutputElement* shared_reduced) {
  int const tid = static_cast<int>(threadIdx.x);
  int const warp = tid >> 5;
  int const lane = tid & 31;
  int const lane_in_group = lane & 3;
  constexpr int kFloatsPerWarp = kMTiles * (kStoreFullFragment ? 128 : 64);
  constexpr int kWarpStride = kFloatsPerWarp + 1;

  if (kStoreFullFragment || lane_in_group < 2) {
#pragma unroll
    for (int m_tile = 0; m_tile < kMTiles; ++m_tile) {
#pragma unroll
      for (int i = 0; i < 4; ++i) {
        int const row = (lane >> 2) + 8 * (i >> 1);
        int const token = 2 * lane_in_group + (i & 1);
        int const output = token * (kMTiles * 16) + m_tile * 16 + row;
        shared_partials[warp * kWarpStride + output] = accumulated[m_tile].element[i];
      }
    }
  }
  consumer_group_sync();

  constexpr int kOutputs = kTokens * kMTiles * 16;
  if (tid < kOutputs) {
    float sum = 0.0F;
#pragma unroll
    for (int source_warp = 0; source_warp < 8; ++source_warp) {
      sum += shared_partials[source_warp * kWarpStride + tid];
    }
    shared_reduced[tid] = float_to_b16<OutputElement>(sum);
  }
  consumer_group_sync();
}

#else
template <int kMTiles, int kActivationElements, int kRowPaddingElements = 0,
          class WeightAccessor>
__device__ __forceinline__ void accumulate_projection_k128_page(
    WeightAccessor const& weight, std::uint8_t const* page_base, int page,
    InternalActivation const* shared_activation,
    Hmma::Accumulator (&accumulated)[kMTiles]) {
  int const warp = static_cast<int>(threadIdx.x) >> 5;
  int const lane = static_cast<int>(threadIdx.x) & 31;
  Hmma::Accumulator block[kMTiles];
#pragma unroll
  for (int m_tile = 0; m_tile < kMTiles; ++m_tile) {
    Hmma::clear(block[m_tile]);
  }

#pragma unroll
  // 8 x k16 = K128 page per warp
  for (int k16 = 0; k16 < 8; ++k16) {
    int const page_k_tile = warp * 8 + k16;
    int const k_base =
        (kActivationElements == kProjPageK ? 0 : page * kProjPageK) +
        page_k_tile * 16;
    Hmma::BFragment const b_fragment =
        load_projection_b_fragment<kActivationElements, kRowPaddingElements>(
            shared_activation, k_base, lane);
#pragma unroll
    for (int m_tile = 0; m_tile + 1 < kMTiles; m_tile += 2) {
      Hmma::AFragment fragments[2];
      weight.load_pair(page_base, m_tile, page_k_tile, lane, fragments);
      Hmma::fma(block[m_tile], fragments[0], b_fragment);
      Hmma::fma(block[m_tile + 1], fragments[1], b_fragment);
    }
    if constexpr (kMTiles % 2 != 0) {
      Hmma::AFragment fragment;
      weight.load_page(page_base, kMTiles - 1, page_k_tile, lane, fragment);
      Hmma::fma(block[kMTiles - 1], fragment, b_fragment);
    }
  }

#pragma unroll
  for (int m_tile = 0; m_tile < kMTiles; ++m_tile) {
    float const scale = weight.scale_page(page_base, m_tile, warp * 8);
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      accumulated[m_tile].element[i] =
          fmaf(block[m_tile].element[i], scale, accumulated[m_tile].element[i]);
    }
  }
}

template <int kMTiles, bool kStoreFullFragment, class OutputElement>
__device__ __forceinline__ void reduce_projection_warp_partials(
    Hmma::Accumulator const (&accumulated)[kMTiles], float* shared_partials,
    OutputElement* shared_reduced) {
  int const tid = static_cast<int>(threadIdx.x);
  int const warp = tid >> 5;
  int const lane = tid & 31;
  int const lane_in_group = lane & 3;
  constexpr int kFloatsPerWarp =
      kMTiles * (kStoreFullFragment ? 128 : kTokens * 16);
  constexpr int kWarpStride =
      kFloatsPerWarp + kTokens * kProjectionPartialTokenPadding + 1;

  if (kStoreFullFragment || lane_in_group < (kTokens + 1) / 2) {
#pragma unroll
    for (int m_tile = 0; m_tile < kMTiles; ++m_tile) {
#pragma unroll
      for (int i = 0; i < 4; ++i) {
        int const row = (lane >> 2) + 8 * (i >> 1);
        int const token = 2 * lane_in_group + (i & 1);
        int const output = token * (kMTiles * 16 + kProjectionPartialTokenPadding) +
                           m_tile * 16 + row;
        shared_partials[warp * kWarpStride + output] =
            accumulated[m_tile].element[i];
      }
    }
  }
  consumer_group_sync();

  constexpr int kOutputs = kTokens * kMTiles * 16;
  for (int output = tid; output < kOutputs; output += kConsumerThreads) {
    int partial_index = output;
    if constexpr (kProjectionPartialTokenPadding != 0) {
      partial_index += (output / (kMTiles * 16)) * kProjectionPartialTokenPadding;
    }
    float sum = 0.0F;
#pragma unroll
    for (int source_warp = 0; source_warp < 8; ++source_warp) {
      sum += shared_partials[source_warp * kWarpStride + partial_index];
    }
    shared_reduced[output] = float_to_b16<OutputElement>(sum);
  }
  consumer_group_sync();
}

#endif

__device__ __forceinline__ void run_proj_q_consumer(Glm5MegaMlaArgs const& args, int cta,
                                                    std::uint8_t* static_storage,
                                                    std::uint8_t* dynamic_storage,
                                                    PipelineControl* pipeline, int task_base) {
  auto const* q_a = workspace_pointer_const<InternalActivation>(args, WorkspaceLayout::kProjX);
  auto* projection = workspace_pointer<InternalActivation>(args, WorkspaceLayout::kProjQ);
  auto* scratch = reinterpret_cast<float*>(static_storage + kProjectionRmsScratchOffset);
  auto* shared_activation =
      reinterpret_cast<InternalActivation*>(dynamic_storage + kProjectionActivationOffset);
  auto* shared_partials = reinterpret_cast<float*>(dynamic_storage + kProjQPartialsOffset);
  auto* shared_reduced =
      reinterpret_cast<InternalActivation*>(dynamic_storage + kProjQReducedOffset);

  stage_rms_normalized_activation<2048, kProjQActivationRowPadding>(
      q_a, kProjXRows, args.q_gamma, shared_activation, scratch);
  ProjQWeightAccessor const weight{args, cta};
  Hmma::Accumulator accumulated[2];
#pragma unroll
  for (int m_tile = 0; m_tile < 2; ++m_tile) {
    Hmma::clear(accumulated[m_tile]);
  }

#pragma unroll
  for (int page = 0; page < 2; ++page) {
    int const task = task_base + page;
    consumer_wait_stage(pipeline, task);
    auto const* page_base = pipeline_stage(dynamic_storage, task, kAttentionStageBytes);
    accumulate_projection_k128_page<2, 2048, kProjQActivationRowPadding>(
        weight, page_base, page, shared_activation, accumulated);
    consumer_release_stage(pipeline, task);
  }

  reduce_projection_warp_partials<2, false>(accumulated, shared_partials, shared_reduced);
  store_reduced_projection_global<32>(shared_reduced, projection, kProjQRows, cta * 32);
}

// Auxiliary CTAs normalize inputs while producers load ProjX weights.
// Publish the normalized inputs before signaling readiness.
struct ProjXRmsWorkspace {
  static constexpr std::size_t kInverse = WorkspaceLayout::kEnd;
  static constexpr std::size_t kNormalized =
      align_up(kInverse + kTokens * sizeof(float), 128);
  static constexpr std::size_t kEnd =
      kNormalized + kTokens * kHidden * sizeof(InternalActivation);
};
static_assert(ProjXRmsWorkspace::kEnd <= kMinimumWorkspaceBytes);

__device__ __forceinline__ void precompute_projx_normalized_input(
    Glm5MegaMlaArgs const& args, int token, std::uint32_t generation,
    std::uint8_t* static_storage, std::uint8_t* dynamic_storage) {
  int const tid = static_cast<int>(threadIdx.x);
  int const warp = tid >> 5;
  int const lane = tid & 31;
  auto const* input =
      static_cast<__nv_bfloat16 const*>(args.x) + token * kHidden;
  auto* scratch =
      reinterpret_cast<float*>(static_storage + kProjectionRmsScratchOffset);
  auto* shared_inverse = reinterpret_cast<float*>(static_storage + 0x200);
  constexpr int kVectorsPerThread = kHidden / (kConsumerThreads * 8);
  uint4 values[kVectorsPerThread];
  float sum = 0.0F;
#pragma unroll
  for (int vector = 0; vector < kVectorsPerThread; ++vector) {
    uint4 const packed = *reinterpret_cast<uint4 const*>(
        input + (vector * kConsumerThreads + tid) * 8);
    values[vector] = packed;
    std::uint32_t const words[4] = {packed.x, packed.y, packed.z, packed.w};
#pragma unroll
    for (int word = 0; word < 4; ++word) {
      std::uint32_t const bits = words[word];
      asm volatile(
          "{\n\t"
          ".reg .b16 low, high;\n\t"
          "mov.b32 {low, high}, %1;\n\t"
          "fma.rn.f32.bf16 %0, low, low, %0;\n\t"
          "fma.rn.f32.bf16 %0, high, high, %0;\n\t"
          "}\n"
          : "+f"(sum) : "r"(bits));
    }
  }
  sum = warp_sum(sum);
  if (lane == 0) {
    scratch[warp] = sum;
  }
  consumer_group_sync();
  if (tid == 0) {
    sum = 0.0F;
#pragma unroll
    for (int source_warp = 0; source_warp < 8; ++source_warp) {
      sum += scratch[source_warp];
    }
    *shared_inverse =
        rsqrtf(sum / static_cast<float>(kHidden) + kRmsEpsilon);
    workspace_pointer<float>(args, ProjXRmsWorkspace::kInverse)[token] =
        *shared_inverse;
  }
  consumer_group_sync();
  float const inverse = *shared_inverse;
  auto* shared_normalized = reinterpret_cast<InternalActivation*>(
      dynamic_storage + kProjectionActivationOffset);
#pragma unroll
  for (int vector = 0; vector < kVectorsPerThread; ++vector) {
    int const column = (vector * kConsumerThreads + tid) * 8;
    float4 const low_gamma =
        load_norm4(args.x_gamma + column);
    float4 const high_gamma =
        load_norm4(args.x_gamma + column + 4);
    uint4 const value = values[vector];
    uint4 normalized;
    normalized.x = normalize_to_internal_x2<__nv_bfloat16>(
        value.x, low_gamma.x, low_gamma.y, inverse);
    normalized.y = normalize_to_internal_x2<__nv_bfloat16>(
        value.y, low_gamma.z, low_gamma.w, inverse);
    normalized.z = normalize_to_internal_x2<__nv_bfloat16>(
        value.z, high_gamma.x, high_gamma.y, inverse);
    normalized.w = normalize_to_internal_x2<__nv_bfloat16>(
        value.w, high_gamma.z, high_gamma.w, inverse);
    *reinterpret_cast<uint4*>(shared_normalized + column) = normalized;
  }
  consumer_group_sync();
  if (tid == 0) {
    publish_shared_writes_to_async_proxy();
    auto* normalized =
        workspace_pointer<InternalActivation>(args, ProjXRmsWorkspace::kNormalized);
    copy_bulk_shared_to_global(normalized + token * kHidden, shared_normalized,
                               kHidden * sizeof(InternalActivation));
    commit_bulk_copy_group();
    wait_bulk_copy_group();
    auto* ready =
        workspace_pointer<std::uint64_t>(args, WorkspaceLayout::kPhaseStates) +
        static_cast<int>(Phase::kAttentionPartials);
    phase_release_add(ready, generation,
                      256 / kTokens + (token < 256 % kTokens));
  }
}

__device__ __forceinline__ void stage_projx_normalized_page(
    InternalActivation const* normalized_x, int page,
    InternalActivation* shared_activation) {
  constexpr int kRowStride = kProjPageK + kProjXActivationRowPadding;
  constexpr int kVectorsPerPage = kProjPageK / 8;
  int const tid = static_cast<int>(threadIdx.x);
  int const first_token = tid / kVectorsPerPage;
  int const column = (tid % kVectorsPerPage) * 8;
  int const source_column = page * kProjPageK + column;
#pragma unroll
  for (int token_offset = 0; token_offset < kTokens / 2; ++token_offset) {
    int const token = first_token + 2 * token_offset;
    uint4 const value = *reinterpret_cast<uint4 const*>(
        normalized_x + token * kHidden + source_column);
    *reinterpret_cast<uint4*>(shared_activation + token * kRowStride + column) =
        value;
  }
  consumer_group_sync();
}

__device__ __forceinline__ void run_projx_normalized_consumer(
    Glm5MegaMlaArgs const& args, int cta, std::uint8_t* static_storage,
    std::uint8_t* dynamic_storage, PipelineControl* pipeline, int task_base) {
  auto const* x =
      workspace_pointer_const<InternalActivation>(args, ProjXRmsWorkspace::kNormalized);
  auto* projection =
      workspace_pointer<InternalActivation>(args, WorkspaceLayout::kProjX);
  auto* shared_activation = reinterpret_cast<InternalActivation*>(
      dynamic_storage + kProjectionActivationOffset);
  auto* shared_partials =
      reinterpret_cast<float*>(dynamic_storage + kProjXPartialsOffset);
  auto* shared_reduced =
      reinterpret_cast<InternalActivation*>(dynamic_storage + kProjXReducedOffset);

  if (threadIdx.x == 0) {
    auto* ready =
        workspace_pointer<std::uint64_t>(args, WorkspaceLayout::kPhaseStates) +
        static_cast<int>(Phase::kAttentionPartials);
    std::uint32_t const generation =
        *reinterpret_cast<std::uint32_t*>(static_storage +
                                         kGenerationScratchOffset);
    prepare_phase(ready, generation);
    wait_phase_count(ready, generation, 256);
  }
  consumer_group_sync();
  ProjXWeightAccessor const weight{
      args, cta};
  Hmma::Accumulator accumulated[2];
#pragma unroll
  for (int m_tile = 0; m_tile < 2; ++m_tile) {
    Hmma::clear(accumulated[m_tile]);
  }

  // Reuse each warp's MMA accumulation across weight pages.
#pragma unroll 1
  for (int page = 0; page < 6; ++page) {
    stage_projx_normalized_page(x, page, shared_activation);
    int const task = task_base + page;
    consumer_wait_stage(pipeline, task);
    auto const* page_base =
        pipeline_stage(dynamic_storage, task, kAttentionStageBytes);
    accumulate_projection_k128_page<2, kProjPageK, kProjXActivationRowPadding>(
        weight, page_base, page, shared_activation, accumulated);
    consumer_group_sync();
    consumer_release_stage(pipeline, task);
  }

  reduce_projection_warp_partials<2, true>(accumulated, shared_partials,
                                           shared_reduced);
  bulk_store_proj_x(shared_reduced, projection, cta * 32);
}


// Configured projection and cache stages with explicit rounding boundaries.
template <class Config>
__device__ __forceinline__ void run_proj_x_consumer(Glm5MegaMlaArgs const& args, int cta,
                                                    std::uint8_t* static_storage,
                                                    std::uint8_t* dynamic_storage,
                                                    PipelineControl* pipeline, int task_base) {
  auto const* x = static_cast<__nv_bfloat16 const*>(args.x);
  auto* projection = workspace_pointer<InternalActivation>(args, WorkspaceLayout::kProjX);
  auto* scratch = reinterpret_cast<float*>(static_storage + kProjectionRmsScratchOffset);
  auto* shared_activation =
      reinterpret_cast<InternalActivation*>(dynamic_storage + kProjectionActivationOffset);
  auto* shared_partials = reinterpret_cast<float*>(dynamic_storage + kProjXPartialsOffset);
  auto* shared_reduced =
      reinterpret_cast<InternalActivation*>(dynamic_storage + kProjXReducedOffset);

  stage_rms_normalized_activation<kHidden, kProjXActivationRowPadding>(x, kHidden, args.x_gamma,
                                                                       shared_activation, scratch);
  ProjXWeightAccessor const weight{args, cta};
  Hmma::Accumulator accumulated[2];
#pragma unroll
  for (int m_tile = 0; m_tile < 2; ++m_tile) {
    Hmma::clear(accumulated[m_tile]);
  }

  // Reuse each warp's MMA accumulation across weight pages.
#pragma unroll 1
  for (int page = 0; page < 6; ++page) {
    int const task = task_base + page;
    consumer_wait_stage(pipeline, task);
    auto const* page_base = pipeline_stage(dynamic_storage, task, kAttentionStageBytes);
    accumulate_projection_k128_page<2, kHidden, kProjXActivationRowPadding>(
        weight, page_base, page, shared_activation, accumulated);
    consumer_release_stage(pipeline, task);
  }

  reduce_projection_warp_partials<2, true>(accumulated, shared_partials, shared_reduced);
  bulk_store_proj_x(shared_reduced, projection, cta * 32);
}

template <class Config>
__device__ __forceinline__ void run_kv_norm(Glm5MegaMlaArgs const& args, int token,
                                            std::uint8_t* dynamic_storage) {
  int const tid = static_cast<int>(threadIdx.x);
  int const warp = tid >> 5;
  int const lane = tid & 31;
  auto const* projection =
      workspace_pointer_const<InternalActivation>(args, WorkspaceLayout::kProjX);
  auto const* input = projection + token * kProjXRows + 2048;
  auto* shared_input =
      reinterpret_cast<InternalActivation*>(dynamic_storage + kProjectionActivationOffset);
  auto* shared_gamma = reinterpret_cast<NormGain*>(dynamic_storage + kOProjResidualOffset);
  auto* scratch = reinterpret_cast<float*>(dynamic_storage + kAttentionKeyPeOffset);

  if (tid < 128) {
#if GLM5_NORM_BF16
    copy_8b_g2s(args.kv_gamma + tid * 4, shared_gamma + tid * 4);
#else
    copy_8b_g2s(args.kv_gamma + tid * 2, shared_gamma + tid * 2);
    copy_8b_g2s(args.kv_gamma + 256 + tid * 2, shared_gamma + 256 + tid * 2);
#endif
    copy_8b_g2s(input + tid * 4, shared_input + tid * 4);
  }
  cutlass::arch::cp_async_fence();
  wait_cp_async_group_0();
  consumer_group_sync();

  std::uint32_t const packed_for_sum = reinterpret_cast<std::uint32_t const*>(shared_input)[tid];
  float sum = b16x2_square_sum<InternalActivation>(packed_for_sum, 0.0F);
  sum = warp_sum(sum);
  if (lane == 0) {
    scratch[warp] = sum;
  }
  consumer_group_sync();
  if (tid < 8) {
    float total = scratch[tid];
    total += __shfl_down_sync(0xffU, total, 4, 8);
    total += __shfl_down_sync(0xffU, total, 2, 8);
    total += __shfl_down_sync(0xffU, total, 1, 8);
    scratch[tid] = total;
  }
  consumer_group_sync();
  float const inverse = rsqrtf(scratch[0] / static_cast<float>(kKvLatentDim) + kRmsEpsilon);
  int const cache_position = load_position(args.cache_positions, args.cache_position_bytes, token);
  std::uint32_t const packed_input = reinterpret_cast<std::uint32_t const*>(shared_input)[tid];
  float2 const gamma = load_norm2(shared_gamma + tid * 2);
  std::uint32_t const normalized =
      normalize_to_internal_x2<InternalActivation>(packed_input, gamma.x, gamma.y, inverse);
  if constexpr (kFp8KvCache) {
    auto* cache = static_cast<std::uint8_t*>(args.kv_cache) +
                  static_cast<std::size_t>(cache_position) * args.kv_cache_row_stride;
    reinterpret_cast<std::uint16_t*>(cache)[tid] = mma_x2_to_e4m3x2(normalized);
  } else {
    auto* cache = static_cast<__nv_bfloat16*>(args.kv_cache) +
                  static_cast<std::size_t>(cache_position) * args.kv_cache_row_stride;
    reinterpret_cast<std::uint32_t*>(cache)[tid] = Hmma::to_bf16x2(normalized);
  }
  publish_global_writes_to_async_proxy();
  consumer_group_sync();
}

template <class Config>
__device__ __forceinline__ void run_rope(Glm5MegaMlaArgs const& args, int token) {
  int const tid = static_cast<int>(threadIdx.x);
  auto* proj_q = workspace_pointer<InternalActivation>(args, WorkspaceLayout::kProjQ);
  auto const* proj_x = workspace_pointer_const<InternalActivation>(args, WorkspaceLayout::kProjX);
  bool const sglang_layout = args.rope_layout == static_cast<int>(RopeLayout::kPositionCosSin);
  int const rope_row =
      sglang_layout ? load_position(args.rope_positions, args.rope_position_bytes, token) : token;
  float const* frequencies = args.rope_freqs + static_cast<std::size_t>(rope_row) * kRopeDim;
  int const cache_position = load_position(args.cache_positions, args.cache_position_bytes, token);

  // Eight Q heads plus the one shared K positional vector, 32 pairs each.
  if (tid < kConsumerThreads) {
    for (int task = tid; task < (kQueryHeads + 1) * (kRopeDim / 2); task += kConsumerThreads) {
      int const vector = task / (kRopeDim / 2);
      int const pair = task % (kRopeDim / 2);
      InternalActivation const* source;
      if (vector < kQueryHeads) {
        auto* destination =
            proj_q + token * kProjQRows + kQueryHeads * kQkNopeDim + vector * kRopeDim;
        source = destination;
      } else {
        source = proj_x + token * kProjXRows + 2048 + kKvLatentDim;
      }
      float const x0 = b16_to_float(source[pair * 2]);
      float const x1 = b16_to_float(source[pair * 2 + 1]);
      float const cosine = frequencies[sglang_layout ? pair : pair * 2];
      float const sine = frequencies[sglang_layout ? pair + kRopeDim / 2 : pair * 2 + 1];
      std::uint32_t const rotated =
          pack_internal_x2(fmaf(-x1, sine, x0 * cosine), fmaf(x0, sine, x1 * cosine));
      if (vector < kQueryHeads) {
        auto* destination =
            proj_q + token * kProjQRows + kQueryHeads * kQkNopeDim + vector * kRopeDim;
        reinterpret_cast<std::uint32_t*>(destination)[pair] = rotated;
      } else if constexpr (kFp8KvCache) {
        auto* destination = static_cast<std::uint8_t*>(args.pe_cache) +
                            static_cast<std::size_t>(cache_position) * args.pe_cache_row_stride;
        reinterpret_cast<std::uint16_t*>(destination)[pair] = mma_x2_to_e4m3x2(rotated);
      } else {
        auto* destination = static_cast<__nv_bfloat16*>(args.pe_cache) +
                            static_cast<std::size_t>(cache_position) * args.pe_cache_row_stride;
        reinterpret_cast<std::uint32_t*>(destination)[pair] = Hmma::to_bf16x2(rotated);
      }
    }
  }
}

template <class Config>
__device__ __forceinline__ void run_q_absorb_consumer(Glm5MegaMlaArgs const& args, int cta,
                                                      std::uint8_t* dynamic_storage,
                                                      PipelineControl* pipeline, int task_base) {
  int const tid = static_cast<int>(threadIdx.x);
  int const warp = tid >> 5;
  int const lane = tid & 31;
  int const head = cta / 8;
  int const row_tile = cta & 7;
  auto const* proj_q = workspace_pointer_const<InternalActivation>(args, WorkspaceLayout::kProjQ);
  auto* output = workspace_pointer<InternalActivation>(args, WorkspaceLayout::kQAbsorb);
  auto* shared_activation =
      reinterpret_cast<InternalActivation*>(dynamic_storage + kProjectionActivationOffset);
  auto* shared_output =
      reinterpret_cast<InternalActivation*>(dynamic_storage + kQAbsorbScratchOffset);

  // Stage q_nope once per CTA for reuse across MMA tiles.
  if (tid < kTokens * (kQkNopeDim / 8)) {
    int const token = tid / (kQkNopeDim / 8);
    int const vector = tid % (kQkNopeDim / 8);
    uint4 const value = *reinterpret_cast<uint4 const*>(proj_q + token * kProjQRows +
                                                        head * kQkNopeDim + vector * 8);
    *reinterpret_cast<uint4*>(shared_activation +
                              token * (kQkNopeDim + kAbsorbActivationRowPadding) + vector * 8) =
        value;
  }
  consumer_group_sync();
  InternalActivation const* mma_activation = shared_activation;

  QAbsorbWeightAccessor const weight{args, cta};
  int const row_base = head * kKvLatentDim + row_tile * 64;
  consumer_wait_stage(pipeline, task_base);
  auto const* page_base = pipeline_stage(dynamic_storage, task_base, kAttentionStageBytes);

  Hmma::Accumulator accumulated;
  Hmma::clear(accumulated);
  if (warp < 4) {
#pragma unroll
    for (int scale_block = 0; scale_block < 3; ++scale_block) {
      Hmma::Accumulator block;
      Hmma::clear(block);
#pragma unroll
      for (int k16 = 0; k16 < 4; ++k16) {
        int const page_k_tile = scale_block * 4 + k16;
        Hmma::BFragment const b_fragment =
            load_q_absorb_b_fragment<kQkNopeDim, kAbsorbActivationRowPadding>(
                mma_activation, page_k_tile * 16, lane);
        Hmma::AFragment a_fragment;
        weight.load_page(page_base, warp, page_k_tile, lane, a_fragment);
        Hmma::fma(block, a_fragment, b_fragment);
      }
      float const scale = weight.scale_page(page_base, warp, scale_block * 4);
#pragma unroll
      for (int i = 0; i < 4; ++i) {
        accumulated.element[i] = fmaf(block.element[i], scale, accumulated.element[i]);
      }
    }
  }
  consumer_release_stage(pipeline, task_base);

  // Transpose the projection result into shared memory.
  int const lane_in_group = lane & 3;
  if (warp < 4 && lane_in_group < (kTokens + 1) / 2) {
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      int const row = (lane >> 2) + 8 * (i >> 1);
      int const token = 2 * lane_in_group + (i & 1);
      shared_output[token * 64 + warp * 16 + QAbsorbWeightAccessor::output_row(row)] =
          float_to_internal(accumulated.element[i]);
    }
  }

  // Synchronize the transposed tile before global stores.
  // The caller orders these stores before publishing QAbsorb readiness.
  consumer_group_sync();
  if (tid < kTokens * (64 / 8)) {
    int const token = tid / (64 / 8);
    int const vector = tid % (64 / 8);
    uint4 const value = *reinterpret_cast<uint4 const*>(shared_output + token * 64 + vector * 8);
    *reinterpret_cast<uint4*>(output + token * (kQueryHeads * kKvLatentDim) + row_base +
                              vector * 8) = value;
  }
}

template <class Config>
__device__ __forceinline__ void run_o_absorb_consumer(Glm5MegaMlaArgs const& args, int cta,
                                                      std::uint8_t* dynamic_storage,
                                                      PipelineControl* pipeline, int task_base) {
  int const tid = static_cast<int>(threadIdx.x);
  int const head = cta / 8;
  int const row_tile = cta & 7;
  auto const* combined =
      workspace_pointer_const<InternalActivation>(args, WorkspaceLayout::kCombined);
  auto* output = workspace_pointer<InternalActivation>(args, WorkspaceLayout::kOAbsorb);
  auto* shared_activation =
      reinterpret_cast<InternalActivation*>(dynamic_storage + kProjectionActivationOffset);
  auto* shared_partials = reinterpret_cast<float*>(dynamic_storage + kOAbsorbPartialsOffset);
  auto* shared_reduced =
      reinterpret_cast<InternalActivation*>(dynamic_storage + kOAbsorbReducedOffset);

  if (tid < kValueDim / 8) {
#pragma unroll
    for (int token = 0; token < kTokens; ++token) {
      copy_16b_g2s(combined + (token * kQueryHeads + head) * kValueDim + tid * 8,
                   shared_activation + token * (kValueDim + kAbsorbActivationRowPadding) + tid * 8);
    }
  }
  cutlass::arch::cp_async_fence();

  OAbsorbWeightAccessor const weight{args, cta};
  int const row_base = head * kHeadDim + row_tile * 32;
  consumer_wait_stage(pipeline, task_base);
  cutlass::arch::cp_async_wait<0>();
  consumer_group_sync();
  InternalActivation const* mma_activation = shared_activation;

  Hmma::Accumulator accumulated[2];
#pragma unroll
  for (int m_tile = 0; m_tile < 2; ++m_tile) {
    Hmma::clear(accumulated[m_tile]);
  }
  auto const* page_base = pipeline_stage(dynamic_storage, task_base, kAttentionStageBytes);
  accumulate_projection_k64_page<2, kValueDim, kAbsorbActivationRowPadding>(
      weight, page_base, 0, mma_activation, accumulated);
  consumer_release_stage(pipeline, task_base);

  reduce_projection_warp_partials<2, false>(accumulated, shared_partials, shared_reduced);
  store_reduced_projection_global<32>(shared_reduced, output, kQueryHeads * kHeadDim, row_base);
}

template <class Config>
__device__ __forceinline__ void accumulate_o_proj_page(OProjWeightAccessor const& weight,
                                                       std::uint8_t const* page_base, int page,
                                                       InternalActivation const* token_activation,
                                                       Hmma::Accumulator (&accumulated)[3]) {
  int const warp = static_cast<int>(threadIdx.x) >> 5;
  int const lane = static_cast<int>(threadIdx.x) & 31;
  Hmma::Accumulator block[3];
#pragma unroll
  for (int m_tile = 0; m_tile < 3; ++m_tile) {
    Hmma::clear(block[m_tile]);
  }
#pragma unroll
  for (int k16 = 0; k16 < 4; ++k16) {
    int const page_k_tile = warp * 4 + k16;
    int const k_base = page * 512 + page_k_tile * 16;
    auto const* source =
        reinterpret_cast<std::uint32_t const*>(token_activation + k_base + 4 * (lane & 3));
    Hmma::BFragment b_fragment;
    auto* words = reinterpret_cast<std::uint32_t*>(b_fragment.element);
    words[0] = source[0];
    words[1] = source[1];
#pragma unroll
    for (int m_tile = 0; m_tile < 3; ++m_tile) {
      Hmma::AFragment a_fragment;
      weight.load_page(page_base, m_tile, page_k_tile, lane, a_fragment);
      Hmma::fma(block[m_tile], a_fragment, b_fragment);
    }
  }
#pragma unroll
  for (int m_tile = 0; m_tile < 3; ++m_tile) {
    float const scale = weight.scale_page(page_base, m_tile, warp * 4);
#pragma unroll
    for (int element = 0; element < 4; ++element) {
      accumulated[m_tile].element[element] =
          fmaf(block[m_tile].element[element], scale, accumulated[m_tile].element[element]);
    }
  }
}

template <class Config>
__device__ __forceinline__ void reduce_o_proj_partials(Hmma::Accumulator const (&accumulated)[3],
                                                       float* shared_partials,
                                                       __nv_bfloat16* shared_reduced) {
  int const tid = static_cast<int>(threadIdx.x);
  int const warp = tid >> 5;
  int const lane = tid & 31;
  // Reduce one output tile at a time to reuse scratch storage.
  // Keep warp summation order fixed before BF16 rounding.
#pragma unroll 1
  for (int m_tile = 0; m_tile < 3; ++m_tile) {
#pragma unroll
    for (int element = 0; element < 4; ++element) {
      int const row = (lane >> 2) + 8 * (element >> 1);
      int const token = 2 * (lane & 3) + (element & 1);
      shared_partials[warp * Config::kOProjWarpPartialStride + token * 20 + row] =
          accumulated[m_tile].element[element];
    }
    consumer_group_sync();
    if (tid < kTokens * 16) {
      int const partial_index = (tid >> 4) * 20 + (tid & 15);
      float sum = 0.0F;
#pragma unroll
      for (int source_warp = 0; source_warp < 8; ++source_warp) {
        sum += shared_partials[source_warp * Config::kOProjWarpPartialStride + partial_index];
      }
      int const token = tid >> 4;
      int const row = tid & 15;
      shared_reduced[token * 48 + m_tile * 16 + row] = float_to_bf16(sum);
    }
    consumer_group_sync();
  }
}

template <class Config>
__device__ __forceinline__ void run_split_storage_o_proj_consumer(
    Glm5MegaMlaArgs const& args, int cta, std::uint8_t* static_storage,
    std::uint8_t* dynamic_storage, PipelineControl* pipeline, int task_base) {
  int const tid = static_cast<int>(threadIdx.x);
  int const lane = tid & 31;
  auto const* o_absorb =
      workspace_pointer_const<InternalActivation>(args, WorkspaceLayout::kOAbsorb);
  auto const* residual = static_cast<__nv_bfloat16 const*>(args.residual);
  auto* shared_activation =
      reinterpret_cast<InternalActivation*>(dynamic_storage + kProjectionActivationOffset);
  auto* last_token_activation = reinterpret_cast<InternalActivation*>(static_storage);
  auto* shared_residual =
      reinterpret_cast<__nv_bfloat16*>(dynamic_storage + Config::kOProjResidualDynamicOffset);
  auto* shared_reduced =
      reinterpret_cast<__nv_bfloat16*>(static_storage + Config::kOProjLocalStaticOffset);
  auto* shared_partials =
      reinterpret_cast<float*>(static_storage + Config::kOProjPartialStaticOffset);
#pragma unroll 1
  for (int page = 0; page < kOProjPages; ++page) {
    for (int index = tid; index < kTokens * 64; index += kConsumerThreads) {
      int const token = index >> 6;
      int const vector = index & 63;
      auto* destination =
          token == kTokens - 1 ? last_token_activation : shared_activation + token * 4096;
      copy_16b_g2s(o_absorb + token * 4096 + page * 512 + vector * 8,
                   destination + page * 512 + vector * 8);
    }
  }
  cutlass::arch::cp_async_fence();
  cutlass::arch::cp_async_wait<0>();
  consumer_group_sync();
  if (tid < kTokens * 6) {
    int const token = tid / 6;
    int const vector = tid % 6;
    copy_16b_g2s(residual + token * kHidden + cta * 48 + vector * 8,
                 shared_residual + token * 48 + vector * 8);
  }
  cutlass::arch::cp_async_fence();

  int const token = lane >> 2;
  auto const* token_activation =
      token == kTokens - 1 ? last_token_activation : shared_activation + token * 4096;
  OProjWeightAccessor const weight{args, cta};
  Hmma::Accumulator accumulated[3];
#pragma unroll
  for (int m_tile = 0; m_tile < 3; ++m_tile) {
    Hmma::clear(accumulated[m_tile]);
  }
#pragma unroll 1
  for (int page = 0; page < kOProjPages; ++page) {
    int const task = task_base + page;
    consumer_wait_stage(pipeline, task);
    auto const* page_base = pipeline_stage(dynamic_storage, task, kAttentionStageBytes);
    accumulate_o_proj_page<Config>(weight, page_base, page, token_activation, accumulated);
    consumer_release_stage(pipeline, task);
  }
  reduce_o_proj_partials<Config>(accumulated, shared_partials, shared_reduced);
}

template <class Config>
__device__ __forceinline__ void run_dynamic_o_proj_consumer(Glm5MegaMlaArgs const& args, int cta,
                                                            std::uint8_t* static_storage,
                                                            std::uint8_t* dynamic_storage,
                                                            PipelineControl* pipeline,
                                                            int task_base) {
  int const tid = static_cast<int>(threadIdx.x);
  auto const* o_absorb =
      workspace_pointer_const<InternalActivation>(args, WorkspaceLayout::kOAbsorb);
  auto const* residual = static_cast<__nv_bfloat16 const*>(args.residual);
  auto* shared_activation =
      reinterpret_cast<InternalActivation*>(dynamic_storage + kProjectionActivationOffset);
  auto* shared_residual = reinterpret_cast<__nv_bfloat16*>(dynamic_storage + kOProjResidualOffset);
  auto* shared_reduced = reinterpret_cast<__nv_bfloat16*>(dynamic_storage + kOProjReducedOffset);
  auto* shared_partials = reinterpret_cast<float*>(dynamic_storage + kOProjPartialsOffset);

  // Stage activation pages for reuse during output projection.
  if constexpr (Config::kTokens == 4) {
#pragma unroll 1
    for (int page = 0; page < kOProjPages; ++page) {
      int const token = tid >> 6;
      int const vector = tid & 63;
      copy_16b_g2s(o_absorb + token * (kQueryHeads * kHeadDim) + page * 512 + vector * 8,
                   shared_activation + token * (kQueryHeads * kHeadDim) + page * 512 + vector * 8);
    }
  } else {
#pragma unroll 1
    for (int page = 0; page < kOProjPages; ++page) {
      for (int index = tid; index < kTokens * 64; index += kConsumerThreads) {
        int const token = index >> 6;
        int const vector = index & 63;
        copy_16b_g2s(
            o_absorb + token * (kQueryHeads * kHeadDim) + page * 512 + vector * 8,
            shared_activation + token * (kQueryHeads * kHeadDim) + page * 512 + vector * 8);
      }
    }
  }
  cutlass::arch::cp_async_fence();

  OProjWeightAccessor const weight{args, cta};
  Hmma::Accumulator accumulated[3];
#pragma unroll
  for (int m_tile = 0; m_tile < 3; ++m_tile) {
    Hmma::clear(accumulated[m_tile]);
  }
  cutlass::arch::cp_async_wait<0>();
  consumer_group_sync();

  // Overlap residual staging with output projection.
  // Wait before the epilogue reads the staged residual.
  if (tid < kTokens * 6) {
    int const token = tid / 6;
    int const vector = tid % 6;
    copy_16b_g2s(residual + token * kHidden + cta * 48 + vector * 8,
                 shared_residual + token * 48 + vector * 8);
  }
  cutlass::arch::cp_async_fence();

  InternalActivation const* mma_activation = shared_activation;

#pragma unroll 1
  for (int page = 0; page < kOProjPages; ++page) {
    int const task = task_base + page;
    consumer_wait_stage(pipeline, task);
    auto const* page_base = pipeline_stage(dynamic_storage, task, kAttentionStageBytes);
    accumulate_projection_k64_page<3, kQueryHeads * kHeadDim>(weight, page_base, page,
                                                              mma_activation, accumulated);
    consumer_release_stage(pipeline, task);
  }

  reduce_projection_warp_partials<3, false>(accumulated, shared_partials, shared_reduced);
}

template <class Config>
__device__ __forceinline__ void run_o_proj_consumer(Glm5MegaMlaArgs const& args, int cta,
                                                    std::uint8_t* static_storage,
                                                    std::uint8_t* dynamic_storage,
                                                    PipelineControl* pipeline, int task_base) {
  if constexpr (Config::kStaticOProjTail) {
    run_split_storage_o_proj_consumer<Config>(args, cta, static_storage, dynamic_storage, pipeline,
                                              task_base);
  } else {
    run_dynamic_o_proj_consumer<Config>(args, cta, static_storage, dynamic_storage, pipeline,
                                        task_base);
  }
}

template <class Config>
__device__ __forceinline__ void run_vector_local_epilogue(Glm5MegaMlaArgs const& args, int cta,
                                                          __nv_bfloat16 const* shared_local,
                                                          __nv_bfloat16 const* shared_residual) {
  int const tid = static_cast<int>(threadIdx.x);
  constexpr int kRowsPerCta = 48;
  constexpr int kValuesPerVector = 8;
  constexpr int kVectorsPerToken = kRowsPerCta / kValuesPerVector;
  constexpr int kActiveThreads = kTokens * kVectorsPerToken;
  auto* local_output = reinterpret_cast<uint4*>(args.local_output);
  auto* final_output = reinterpret_cast<uint4*>(args.final_output);

  uint4 local{};
  int output_vector = 0;
  if (tid < kActiveThreads) {
    int const token = tid / kVectorsPerToken;
    int const vector = tid % kVectorsPerToken;
    output_vector = token * (kHidden / kValuesPerVector) + cta * kVectorsPerToken + vector;
    local = reinterpret_cast<uint4 const*>(shared_local)[tid];
    local_output[output_vector] = local;
  }
  cutlass::arch::cp_async_wait<0>();
  consumer_group_sync();
  if (tid < kActiveThreads) {
    uint4 const residual = reinterpret_cast<uint4 const*>(shared_residual)[tid];
    uint4 const result =
        make_uint4(add_bf16x2(local.x, residual.x), add_bf16x2(local.y, residual.y),
                   add_bf16x2(local.z, residual.z), add_bf16x2(local.w, residual.w));
    final_output[output_vector] = result;
  }
}

template <class Config>
__device__ __forceinline__ void run_scalar_local_epilogue(Glm5MegaMlaArgs const& args, int cta,
                                                          __nv_bfloat16 const* shared_local,
                                                          __nv_bfloat16 const* shared_residual) {
  int const tid = static_cast<int>(threadIdx.x);
  constexpr int kRowsPerCta = 48;
  constexpr int kValuesPerVector = 8;
  constexpr int kVectorsPerToken = kRowsPerCta / kValuesPerVector;
  auto* local_output = static_cast<__nv_bfloat16*>(args.local_output);
  auto* final_output = static_cast<__nv_bfloat16*>(args.final_output);

  // Preserve local OProj while the early residual copy may still be in flight.
  if (tid < kTokens * kVectorsPerToken) {
    int const token = tid / kVectorsPerToken;
    int const vector = tid % kVectorsPerToken;
    uint4 const value = *reinterpret_cast<uint4 const*>(shared_local + token * kRowsPerCta +
                                                        vector * kValuesPerVector);
    auto* destination =
        reinterpret_cast<uint4*>(local_output + token * kHidden + cta * kRowsPerCta);
    destination[vector] = value;
  }

  cutlass::arch::cp_async_wait<0>();
  consumer_group_sync();

  if (tid < kRowsPerCta) {
#pragma unroll
    for (int token = 0; token < kTokens; ++token) {
      int const local_index = token * kRowsPerCta + tid;
      int const output_index = token * kHidden + cta * kRowsPerCta + tid;
      float const local = bf16_to_float(shared_local[local_index]);
      float const residual = bf16_to_float(shared_residual[local_index]);
      final_output[output_index] = float_to_bf16(local + residual);
    }
  }
}

template <class Config>
__device__ __forceinline__ void run_single_rank_epilogue(Glm5MegaMlaArgs const& args, int cta,
                                                         __nv_bfloat16 const* shared_local,
                                                         __nv_bfloat16 const* shared_residual) {
  if constexpr (Config::kStaticOProjTail) {
    run_vector_local_epilogue<Config>(args, cta, shared_local, shared_residual);
  } else {
    run_scalar_local_epilogue<Config>(args, cta, shared_local, shared_residual);
  }
}
}  // namespace glm5::mla::detail
