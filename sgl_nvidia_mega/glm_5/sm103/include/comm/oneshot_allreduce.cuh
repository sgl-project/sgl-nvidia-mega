#pragma once

#include <common/wait.cuh>

#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cutlass/arch/barrier.h>
#include <cutlass/arch/memory_sm80.h>

#include <cstddef>
#include <cstdint>

namespace glm5::comm {

inline constexpr unsigned kFullWarpMask = 0xffffffffU;

__device__ __forceinline__ void store_relaxed_system_u32x4(uint4* address, uint4 value) {
  asm volatile(
      "st.L1::no_allocate.relaxed.sys.v4.u32 "
      "[%0], {%1, %2, %3, %4};"
      :
      : "l"(address), "r"(value.x), "r"(value.y), "r"(value.z), "r"(value.w)
      : "memory");
}

__device__ __forceinline__ uint4 load_relaxed_system_u32x4(uint4 const* address) {
  uint4 value;
  asm volatile(
      "ld.L1::no_allocate.relaxed.sys.v4.u32 "
      "{%0, %1, %2, %3}, [%4];"
      : "=r"(value.x), "=r"(value.y), "=r"(value.z), "=r"(value.w)
      : "l"(address)
      : "memory");
  return value;
}

__device__ __forceinline__ uint2 load_relaxed_system_u32x2(uint2 const* address) {
  uint2 value;
  asm volatile(
      "ld.L1::no_allocate.relaxed.sys.v2.u32 "
      "{%0, %1}, [%2];"
      : "=r"(value.x), "=r"(value.y)
      : "l"(address)
      : "memory");
  return value;
}

__device__ __forceinline__ void store_relaxed_gpu_u32x4(uint4* address, uint4 value) {
  asm volatile(
      "st.global.relaxed.gpu.v4.u32 "
      "[%0], {%1, %2, %3, %4};"
      :
      : "l"(address), "r"(value.x), "r"(value.y), "r"(value.z), "r"(value.w)
      : "memory");
}

__device__ __forceinline__ std::uint32_t load_relaxed_gpu_u32(
    std::uint32_t const* address) {
  std::uint32_t value;
  asm volatile("ld.global.relaxed.gpu.u32 %0, [%1];"
               : "=r"(value)
               : "l"(address)
               : "memory");
  return value;
}

__device__ __forceinline__ std::uint16_t load_relaxed_gpu_u16(
    std::uint16_t const* address) {
  std::uint16_t value;
  asm volatile("ld.global.relaxed.gpu.u16 %0, [%1];"
               : "=h"(value)
               : "l"(address)
               : "memory");
  return value;
}

__device__ __forceinline__ void store_relaxed_gpu_u16(std::uint16_t* address,
                                                       std::uint16_t value) {
  asm volatile("st.global.relaxed.gpu.u16 [%0], %1;"
               :
               : "l"(address), "h"(value)
               : "memory");
}

__device__ __forceinline__ void acquire_fence_gpu() {
  asm volatile("fence.acquire.gpu;" : : : "memory");
}

__device__ __forceinline__ uint4 load_shared_u32x4(uint4 const* address) {
  uint4 value;
  std::uint32_t const shared_address =
      static_cast<std::uint32_t>(__cvta_generic_to_shared(address));
  asm volatile("ld.shared.v4.u32 {%0, %1, %2, %3}, [%4];"
               : "=r"(value.x), "=r"(value.y), "=r"(value.z), "=r"(value.w)
               : "r"(shared_address)
               : "memory");
  return value;
}

__device__ __forceinline__ uint2 load_shared_u32x2(uint2 const* address) {
  uint2 value;
  std::uint32_t const shared_address =
      static_cast<std::uint32_t>(__cvta_generic_to_shared(address));
  asm volatile("ld.shared.v2.u32 {%0, %1}, [%2];"
               : "=r"(value.x), "=r"(value.y)
               : "r"(shared_address)
               : "memory");
  return value;
}

__device__ __forceinline__ void store_shared_u32x4(uint4* address, uint4 value) {
  std::uint32_t const shared_address =
      static_cast<std::uint32_t>(__cvta_generic_to_shared(address));
  asm volatile("st.shared.v4.u32 [%0], {%1, %2, %3, %4};"
               :
               : "r"(shared_address), "r"(value.x), "r"(value.y), "r"(value.z), "r"(value.w)
               : "memory");
}

__device__ __forceinline__ uint2 load_shared_u32x2(std::uint32_t shared_address) {
  uint2 value;
  asm volatile("ld.shared.v2.u32 {%0, %1}, [%2];"
               : "=r"(value.x), "=r"(value.y)
               : "r"(shared_address)
               : "memory");
  return value;
}

__device__ __forceinline__ void* load_shared_pointer(void* const* address) {
  std::uint64_t value;
  std::uint32_t const shared_address =
      static_cast<std::uint32_t>(__cvta_generic_to_shared(address));
  asm volatile("ld.shared.u64 %0, [%1];"
               : "=l"(value)
               : "r"(shared_address)
               : "memory");
  return reinterpret_cast<void*>(value);
}

__device__ __forceinline__ std::uint32_t next_generation(std::uint32_t generation) {
  generation += 1U;
  return generation == 0U ? 1U : generation;
}

__device__ __forceinline__ std::uint16_t bf16_bits_from_u32x2(uint2 value, int index) {
  std::uint32_t const word = index < 2 ? value.x : value.y;
  return static_cast<std::uint16_t>(word >> ((index & 1) * 16));
}

__device__ __forceinline__ std::uint32_t pack_bf16x2(float low, float high) {
  std::uint32_t packed;
  asm("cvt.rn.bf16x2.f32 %0, %2, %1;" : "=r"(packed) : "f"(low), "f"(high));
  return packed;
}

__device__ __forceinline__ float2 bf16x2_bits_to_float2(std::uint32_t value) {
  return make_float2(__uint_as_float(value << 16),
                     __uint_as_float(value & 0xffff0000U));
}

__device__ __forceinline__ float2 add_f32x2(float2 lhs, float2 rhs) {
  std::uint64_t const lhs_bits = static_cast<std::uint64_t>(__float_as_uint(lhs.x)) |
                                 (static_cast<std::uint64_t>(__float_as_uint(lhs.y)) << 32);
  std::uint64_t const rhs_bits = static_cast<std::uint64_t>(__float_as_uint(rhs.x)) |
                                 (static_cast<std::uint64_t>(__float_as_uint(rhs.y)) << 32);
  std::uint64_t result_bits;
  asm("add.f32x2 %0, %1, %2;" : "=l"(result_bits) : "l"(lhs_bits), "l"(rhs_bits));
  return make_float2(__uint_as_float(static_cast<std::uint32_t>(result_bits)),
                     __uint_as_float(static_cast<std::uint32_t>(result_bits >> 32)));
}

__device__ __forceinline__ void accumulate_direct_rank(
    uint4 values, float2& sum01, float2& sum23, float2& sum45, float2& sum67) {
  sum01 = add_f32x2(sum01, bf16x2_bits_to_float2(values.x));
  sum23 = add_f32x2(sum23, bf16x2_bits_to_float2(values.y));
  sum45 = add_f32x2(sum45, bf16x2_bits_to_float2(values.z));
  sum67 = add_f32x2(sum67, bf16x2_bits_to_float2(values.w));
}

__device__ __forceinline__ std::uint32_t finalize_bf16x2(std::uint32_t reduced,
                                                         std::uint32_t residual) {
  float const reduced0 =
      __bfloat162float(__ushort_as_bfloat16(static_cast<std::uint16_t>(reduced)));
  float const reduced1 =
      __bfloat162float(__ushort_as_bfloat16(static_cast<std::uint16_t>(reduced >> 16)));
  float const residual0 =
      __bfloat162float(__ushort_as_bfloat16(static_cast<std::uint16_t>(residual)));
  float const residual1 =
      __bfloat162float(__ushort_as_bfloat16(static_cast<std::uint16_t>(residual >> 16)));
  return pack_bf16x2(reduced0 + residual0, reduced1 + residual1);
}

__device__ __forceinline__ std::uint32_t add_bf16x2(std::uint32_t lhs,
                                                     std::uint32_t rhs) {
  std::uint32_t result;
  asm volatile("add.rn.bf16x2 %0, %1, %2;" : "=r"(result) : "r"(lhs), "r"(rhs));
  return result;
}

__device__ __forceinline__ uint2 pack_bf16x4(float const (&values)[4]) {
  return make_uint2(pack_bf16x2(values[0], values[1]),
                    pack_bf16x2(values[2], values[3]));
}

__device__ __forceinline__ int compact_source_slot(int source_rank, int destination_rank) {
  return source_rank < destination_rank ? source_rank : source_rank - 1;
}

__device__ __forceinline__ int expand_peer_slot(int peer_slot, int local_rank) {
  return peer_slot < local_rank ? peer_slot : peer_slot + 1;
}

template <class Config>
__device__ __forceinline__ std::uint8_t* virtual_tile_base(void* peer_workspace, int token,
                                                           int source_slot, int phase,
                                                           int virtual_tile) {
  return static_cast<std::uint8_t*>(peer_workspace) +
         static_cast<std::size_t>(token) * Config::kTokenStride +
         static_cast<std::size_t>(source_slot) * Config::kSourceStride +
         static_cast<std::size_t>(phase) * Config::kPhaseStride +
         static_cast<std::size_t>(virtual_tile) * Config::kVirtualTileStride;
}

template <int kSource, class Config, class Args>
__device__ __forceinline__ uint4 load_direct_rank(
    Args const& args, uint4 const* local_vectors, int vector, int rank, int token, int phase,
    int virtual_tile, int lane) {
  static_assert(kSource >= 0 && kSource < Config::kWorldSize);
  if (rank == kSource) {
    return load_shared_u32x4(local_vectors + vector);
  }
  int const source_slot =
      kSource == 0 ? 0 : kSource - static_cast<int>(kSource >= rank);
  auto const* source_vectors = reinterpret_cast<uint4 const*>(virtual_tile_base<Config>(
      args.peer_workspaces[rank], token, source_slot, phase, virtual_tile));
  return load_relaxed_system_u32x4(source_vectors + lane);
}

__device__ __forceinline__ void poll_backoff() { asm volatile("" : : : "memory"); }

__device__ __forceinline__ void wait_marker(uint4 const* address, std::uint32_t expected) {
  uint4 observed;
  do {
    observed = load_relaxed_system_u32x4(address);
    poll_backoff();
  } while (observed.z != expected || observed.w != expected);
}

template <class Config>
__device__ __forceinline__ void consumer_group_sync() {
  cutlass::arch::NamedBarrier::sync(
      Config::kConsumerThreads, cutlass::arch::ReservedNamedBarriers::TransposeBarrier);
}

template <class Config, class Args>
__device__ __forceinline__ void store_deferred_local_output(
    Args const& args, int token, int output_element_base, int payload_vectors,
    uint4 const* local_vectors) {
  int const tid = static_cast<int>(threadIdx.x);
  if (args.local_output == nullptr || tid >= payload_vectors) {
    return;
  }
  auto* output = reinterpret_cast<uint4*>(
      static_cast<__nv_bfloat16*>(args.local_output) +
      static_cast<std::size_t>(token) * Config::kHidden + output_element_base);
  store_relaxed_gpu_u32x4(output + tid, load_shared_u32x4(local_vectors + tid));
}

// kBoundOutput opts into zero-padded tail masking; legacy seven-argument calls
// retain their full-tile path at compile time. The optional bound is
// expressed in BF16 elements and must be a multiple of kValuesPerVector.
template <class Config, bool kDirectRemoteReduce = false,
          bool kAddResidual = true, bool kBoundOutput = false, class Args>
__device__ void allreduce_and_finalize(Args const& args, void* const* peer_workspaces, int token,
                                       int virtual_tile_base_index, int virtual_tile_count,
                                       int output_element_base,
                                       std::uint8_t* dynamic_storage,
                                       int valid_output_elements = -1) {
  static_assert(Config::kWorldSize == 4 || Config::kWorldSize == 8);
  static_assert(Config::kRemoteSources == Config::kWorldSize - 1);
  static_assert(Config::kConsumerThreads == 256);
  static_assert(Config::kWarpSize == 32);
  static_assert(Config::kPayloadRowsPerColumn > 0 &&
                Config::kPayloadRowsPerColumn < Config::kVectorsPerColumn);
  static_assert(Config::kVectorsPerColumn == 8);
  static_assert(Config::kValuesPerVector == 8);
  static_assert(Config::kReductionValuesPerThread == 4);
  static_assert(Config::kVirtualTilesPerPass == 4);
  static_assert(Config::kMaxVirtualTilesPerCta <= Config::kVirtualTilesPerPass * 2);

  int const tid = static_cast<int>(threadIdx.x);
  if (tid >= Config::kConsumerThreads) {
    return;
  }

  int const world_size = args.world_size;
  int const rank = args.rank;
  int const payload_vectors = virtual_tile_count * Config::kPayloadRowsPerColumn;
  int const tile_elements = virtual_tile_count * Config::kElementsPerVirtualTile;
  int const output_elements =
      kBoundOutput && valid_output_elements >= 0 ? valid_output_elements : tile_elements;
  auto* local_vectors = reinterpret_cast<uint4*>(dynamic_storage + Config::kLocalOffset);
  auto* remote_vectors = reinterpret_cast<uint4*>(dynamic_storage + Config::kRemoteOffset);

  int phase;
  std::uint32_t marker;
  if (args.ar_epoch_counters != nullptr) {
    marker = reinterpret_cast<std::uint32_t const*>(remote_vectors)[0];
    phase = static_cast<int>(marker & 1U);
  } else {
    phase = static_cast<int>(args.epoch & 1U);
    marker = static_cast<std::uint32_t>(args.epoch) + 1U;
    if (marker == 0U) {
      marker = 1U;
    }
  }

  if (world_size == Config::kWorldSize) {
    int const warp = tid >> 5;
    int const lane = tid & (Config::kWarpSize - 1);
    int const peer_slot =
        Config::kWorldSize == 4
            ? ((warp & 1) != 0 ? 2 + (lane >> 3) : lane >> 3)
            : ((warp & 1) << 2) | (lane & 3);
    int const row = Config::kWorldSize == 4 ? lane & 7 : lane >> 2;
    bool const payload_lane = row < Config::kPayloadRowsPerColumn;
    bool const marker_lane = row == Config::kVectorsPerColumn - 1;
    bool const selected_peer =
        peer_slot < Config::kRemoteSources &&
        (Config::kWorldSize != 4 || (warp & 1) != 0 || (lane >> 3) < 2);

#pragma unroll
    for (int pass = 0; pass < 2; ++pass) {
      int const column = (warp >> 1) + pass * Config::kVirtualTilesPerPass;
      bool const active_column = column < virtual_tile_count;
      if (selected_peer && active_column) {
        int const destination = expand_peer_slot(peer_slot, rank);
        int const source_slot = compact_source_slot(rank, destination);
        auto* destination_vectors = reinterpret_cast<uint4*>(virtual_tile_base<Config>(
            load_shared_pointer(peer_workspaces + destination), token, source_slot, phase,
            virtual_tile_base_index + column));
        int const payload_vector = column * Config::kPayloadRowsPerColumn + row;
        uint4 const values = payload_lane ? load_shared_u32x4(local_vectors + payload_vector)
                                          : make_uint4(marker, marker, marker, marker);
        store_relaxed_system_u32x4(destination_vectors + row, values);
      }
    }
    if constexpr (kDirectRemoteReduce) {
      cutlass::arch::cp_async_wait<0>();
    }
    consumer_group_sync<Config>();

    if constexpr (kBoundOutput) {
      store_deferred_local_output<Config>(
          args, token, output_element_base,
          min(payload_vectors, output_elements / Config::kValuesPerVector), local_vectors);
    } else {
      store_deferred_local_output<Config>(
          args, token, output_element_base, payload_vectors, local_vectors);
    }

    if constexpr (kDirectRemoteReduce) {
      if (warp < virtual_tile_count) {
        if (lane < Config::kRemoteSources) {
          auto* source_vectors = reinterpret_cast<uint4*>(virtual_tile_base<Config>(
              args.peer_workspaces[rank], token, lane, phase,
              virtual_tile_base_index + warp));
          wait_marker(source_vectors + Config::kVectorsPerColumn - 1, marker);
        }
        __syncwarp();

        if (lane < Config::kPayloadRowsPerColumn &&
            (!kBoundOutput || warp * Config::kPayloadRowsPerColumn + lane <
                                 output_elements / Config::kValuesPerVector)) {
          int const vector = warp * Config::kPayloadRowsPerColumn + lane;
          float2 sum01 = make_float2(0.0F, 0.0F);
          float2 sum23 = make_float2(0.0F, 0.0F);
          float2 sum45 = make_float2(0.0F, 0.0F);
          float2 sum67 = make_float2(0.0F, 0.0F);
          int const virtual_tile = virtual_tile_base_index + warp;
          uint4 current = load_direct_rank<0, Config>(
              args, local_vectors, vector, rank, token, phase, virtual_tile, lane);
          uint4 next = load_direct_rank<1, Config>(
              args, local_vectors, vector, rank, token, phase, virtual_tile, lane);
          accumulate_direct_rank(current, sum01, sum23, sum45, sum67);
          current = next;
          next = load_direct_rank<2, Config>(
              args, local_vectors, vector, rank, token, phase, virtual_tile, lane);
          accumulate_direct_rank(current, sum01, sum23, sum45, sum67);
          current = next;
          next = load_direct_rank<3, Config>(
              args, local_vectors, vector, rank, token, phase, virtual_tile, lane);
          accumulate_direct_rank(current, sum01, sum23, sum45, sum67);
          if constexpr (Config::kWorldSize == 4) {
            accumulate_direct_rank(next, sum01, sum23, sum45, sum67);
          } else {
            current = next;
            next = load_direct_rank<4, Config>(
                args, local_vectors, vector, rank, token, phase, virtual_tile, lane);
            accumulate_direct_rank(current, sum01, sum23, sum45, sum67);
            current = next;
            next = load_direct_rank<5, Config>(
                args, local_vectors, vector, rank, token, phase, virtual_tile, lane);
            accumulate_direct_rank(current, sum01, sum23, sum45, sum67);
            current = next;
            next = load_direct_rank<6, Config>(
                args, local_vectors, vector, rank, token, phase, virtual_tile, lane);
            accumulate_direct_rank(current, sum01, sum23, sum45, sum67);
            current = next;
            next = load_direct_rank<7, Config>(
                args, local_vectors, vector, rank, token, phase, virtual_tile, lane);
            accumulate_direct_rank(current, sum01, sum23, sum45, sum67);
            accumulate_direct_rank(next, sum01, sum23, sum45, sum67);
          }

          uint4 const rounded =
              make_uint4(pack_bf16x2(sum01.x, sum01.y), pack_bf16x2(sum23.x, sum23.y),
                         pack_bf16x2(sum45.x, sum45.y), pack_bf16x2(sum67.x, sum67.y));
          uint4 finalized = rounded;
          if constexpr (kAddResidual) {
            auto const* staged_residual = reinterpret_cast<uint4 const*>(
                dynamic_storage + Config::kResidualOffset);
            uint4 const residual = load_shared_u32x4(staged_residual + vector);
            finalized = make_uint4(add_bf16x2(rounded.x, residual.x),
                                   add_bf16x2(rounded.y, residual.y),
                                   add_bf16x2(rounded.z, residual.z),
                                   add_bf16x2(rounded.w, residual.w));
          }
          auto* final_output = reinterpret_cast<uint4*>(
              static_cast<__nv_bfloat16*>(args.final_output) +
              static_cast<std::size_t>(token) * Config::kHidden + output_element_base);
          final_output[vector] = finalized;
        }
      }
      return;
    }

    if constexpr (!kDirectRemoteReduce) {
#pragma unroll
      for (int pass = 0; pass < 2; ++pass) {
        int const column = (warp >> 1) + pass * Config::kVirtualTilesPerPass;
        bool const active_column = column < virtual_tile_count;
        uint4* source_vectors =
            selected_peer && active_column
                ? reinterpret_cast<uint4*>(virtual_tile_base<Config>(
                      args.peer_workspaces[rank], token, peer_slot, phase,
                      virtual_tile_base_index + column))
                : nullptr;
        if (source_vectors != nullptr && marker_lane) {
          wait_marker(source_vectors + row, marker);
        }
        __syncwarp();
        if (source_vectors != nullptr && payload_lane) {
          int const payload_vector = column * Config::kPayloadRowsPerColumn + row;
          uint4 const values = load_relaxed_system_u32x4(source_vectors + row);
          store_shared_u32x4(remote_vectors + peer_slot * Config::kMaxPayloadVectors +
                                 payload_vector,
                             values);
        }
      }
    }
  }

  cutlass::arch::cp_async_wait<0>();
  consumer_group_sync<Config>();

  int const reduction_threads = output_elements / Config::kReductionValuesPerThread;
  if (tid < reduction_threads) {
    int const vector = tid;
    std::uint32_t const local_address =
        static_cast<std::uint32_t>(__cvta_generic_to_shared(local_vectors)) +
        vector * sizeof(uint2);
    uint2 rounded;
    if (world_size == Config::kWorldSize) {
      std::uint32_t const remote_address =
          static_cast<std::uint32_t>(__cvta_generic_to_shared(remote_vectors));
      float sums[Config::kReductionValuesPerThread] = {0.0F, 0.0F, 0.0F, 0.0F};
#pragma unroll
      for (int source = 0; source < Config::kWorldSize; ++source) {
        int const source_slot = source == 0 ? 0 : source - static_cast<int>(source >= rank);
        std::uint32_t const source_address =
            source == rank
                ? local_address
                : remote_address +
                      (source_slot * Config::kMaxTileElements +
                       vector * Config::kReductionValuesPerThread) *
                          sizeof(__nv_bfloat16);
        uint2 const values = load_shared_u32x2(source_address);
#pragma unroll
        for (int element = 0; element < Config::kReductionValuesPerThread; ++element) {
          sums[element] +=
              __bfloat162float(__ushort_as_bfloat16(bf16_bits_from_u32x2(values, element)));
        }
      }
      rounded = pack_bf16x4(sums);
    } else {
      rounded = load_shared_u32x2(local_address);
    }

    uint2 finalized = rounded;
    if constexpr (kAddResidual) {
      std::uint32_t const residual_address =
          static_cast<std::uint32_t>(
              __cvta_generic_to_shared(dynamic_storage + Config::kResidualOffset)) +
          vector * sizeof(uint2);
      uint2 const residual_values = load_shared_u32x2(residual_address);
      finalized = make_uint2(finalize_bf16x2(rounded.x, residual_values.x),
                             finalize_bf16x2(rounded.y, residual_values.y));
    }
    auto* final_output = reinterpret_cast<uint2*>(
        static_cast<__nv_bfloat16*>(args.final_output) +
        static_cast<std::size_t>(token) * Config::kHidden + output_element_base);
    final_output[vector] = finalized;
  }
}

// One CTA owns 48 BF16 output elements per token. Four-token mode uses warp
// pairs to broadcast; six- and eight-token modes give one warp per token. Each token
// is reduced directly from the seven peer mailboxes.
template <class Config, class Args>
__device__ __forceinline__ void allreduce_48_rows_and_add_residual(
    Args const& args, std::uint32_t generation, __nv_bfloat16* shared_local,
    __nv_bfloat16 const* shared_residual) {
  static_assert(Config::kWorldSize == 4 || Config::kWorldSize == 8);
  static_assert(Config::kTokens == 4 || Config::kTokens == 8);
  static_assert(Config::kConsumerThreads == 256);
  static_assert(Config::kPayloadVectors == 6);

  int const tid = static_cast<int>(threadIdx.x);
  int const cta = static_cast<int>(blockIdx.x);
  int const phase = static_cast<int>(generation & 1U);
  int const warp = tid >> 5;
  int const lane = tid & 31;
  constexpr int kRemoteSources = Config::kWorldSize - 1;

  auto mailbox_base = [&](void* peer_workspace, int token, int source_slot) {
    return static_cast<std::uint8_t*>(peer_workspace) +
           static_cast<std::size_t>(token) * Config::kTokenStride +
           static_cast<std::size_t>(source_slot) * Config::kSourceStride +
           static_cast<std::size_t>(phase) * Config::kPhaseStride +
           static_cast<std::size_t>(cta) * Config::kCtaStride;
  };

  auto* local_vectors = reinterpret_cast<uint4*>(shared_local);
  auto* local_output = static_cast<__nv_bfloat16*>(args.local_output);
  auto* output = static_cast<__nv_bfloat16*>(args.final_output);

  if constexpr (Config::kTokens >= 6) {
    if (warp < Config::kTokens) {
      int const token = warp;
      int const row = lane >> 2;
#pragma unroll
      for (int peer_group = 0; peer_group < 2; ++peer_group) {
        int const peer_slot = (lane & 3) + peer_group * 4;
        if (peer_slot < kRemoteSources) {
          int const destination = expand_peer_slot(peer_slot, args.rank);
          int const source_slot = compact_source_slot(args.rank, destination);
          auto* destination_vectors = reinterpret_cast<uint4*>(
              mailbox_base(args.peer_workspaces[destination], token, source_slot));
          uint4 const value =
              row < Config::kPayloadVectors
                  ? load_shared_u32x4(local_vectors + token * Config::kPayloadVectors + row)
                  : make_uint4(generation, generation, generation, generation);
          store_relaxed_system_u32x4(destination_vectors + row, value);
        }
      }
    }
  } else {
    int const token = warp >> 1;
    int const peer_slot = ((warp & 1) << 2) | (lane & 3);
    int const row = lane >> 2;
    if (peer_slot < kRemoteSources) {
      int const destination = expand_peer_slot(peer_slot, args.rank);
      int const source_slot = compact_source_slot(args.rank, destination);
      auto* destination_vectors = reinterpret_cast<uint4*>(
          mailbox_base(args.peer_workspaces[destination], token, source_slot));
      uint4 const value =
          row < Config::kPayloadVectors
              ? load_shared_u32x4(local_vectors + token * Config::kPayloadVectors + row)
              : make_uint4(generation, generation, generation, generation);
      store_relaxed_system_u32x4(destination_vectors + row, value);
    }
  }

  if (local_output != nullptr && tid < Config::kTokens * Config::kPayloadVectors) {
    int const output_token = tid / Config::kPayloadVectors;
    int const vector = tid % Config::kPayloadVectors;
    auto* destination = reinterpret_cast<uint4*>(
        local_output + output_token * Config::kHidden + cta * 48);
    destination[vector] = load_shared_u32x4(
        local_vectors + output_token * Config::kPayloadVectors + vector);
  }

  cutlass::arch::cp_async_wait<0>();
  cutlass::arch::NamedBarrier::sync(Config::kConsumerThreads,
                                    Config::kNamedBarrier);

  constexpr int kReductionValuesPerLane = 4;
  constexpr int kReductionLanes = 48 / kReductionValuesPerLane;
  if (warp < Config::kTokens) {
    int const reduce_token = warp;
    if (lane < kRemoteSources) {
      auto const* source_vectors = reinterpret_cast<uint4 const*>(
          mailbox_base(args.peer_workspaces[args.rank], reduce_token, lane));
      wait_marker(source_vectors + 7, generation);
    }
    __syncwarp();

    if (lane >= kReductionLanes) {
      return;
    }
    float sum[kReductionValuesPerLane] = {};
#pragma unroll
    for (int source = 0; source < Config::kWorldSize; ++source) {
      uint2 values;
      if (source == args.rank) {
        auto const* local_halves = reinterpret_cast<uint2 const*>(shared_local);
        values = load_shared_u32x2(
            local_halves + reduce_token * kReductionLanes + lane);
      } else {
        int const source_slot = compact_source_slot(source, args.rank);
        auto const* source_vectors = reinterpret_cast<uint4 const*>(
            mailbox_base(args.peer_workspaces[args.rank], reduce_token, source_slot));
        values = load_relaxed_system_u32x2(
            reinterpret_cast<uint2 const*>(source_vectors) + lane);
      }
#pragma unroll
      for (int element = 0; element < kReductionValuesPerLane; ++element) {
        std::uint32_t const packed = element < 2 ? values.x : values.y;
        std::uint16_t const bits =
            static_cast<std::uint16_t>(packed >> (16 * (element & 1)));
        sum[element] += __bfloat162float(__ushort_as_bfloat16(bits));
      }
    }
    uint2 const rounded =
        make_uint2(pack_bf16x2(sum[0], sum[1]), pack_bf16x2(sum[2], sum[3]));
    auto const* residual_halves = reinterpret_cast<uint2 const*>(shared_residual);
    uint2 const residual = load_shared_u32x2(
        residual_halves + reduce_token * kReductionLanes + lane);
    uint2 const finalized =
        make_uint2(finalize_bf16x2(rounded.x, residual.x),
                   finalize_bf16x2(rounded.y, residual.y));
    auto* output_halves = reinterpret_cast<uint2*>(
        output + reduce_token * Config::kHidden + cta * 48);
    output_halves[lane] = finalized;
  }
}

}  // namespace glm5::comm
