// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// Portions Copyright 2023-2024 SGLang Team (Apache-2.0)
// Derived from SGLang (https://github.com/sgl-project/sglang), Apache License 2.0; see ../include/LICENSE.sglang.
// Sources: SGLang python/sglang/kernels/jit/csrc/distributed/nvlink_comm.cuh (nvlink_pull_kernel and the pull half of
// NVLinkComm) and python/sglang/kernels/jit/include/sgl_kernel/distributed/communicator.cuh (McBarrier) at revision
// a55ab7d.
// Modified by NVIDIA: the two barrier waits of the pull kernel (entry and exit) are bounded. Each wait polls the same
// semaphore with the same load (scope and semantics unchanged) and counts its unsuccessful polls; at the caller's
// limit the kernel prints one diagnostic line and executes a trap, which the host sees as a CUDA error, instead of
// polling forever. The data path, the launch shape (grid, block, barrier windows) and every load, store, reduction
// and fence of the original are kept as they are; only the all-gather entry point is exported.

#include <sgl_kernel/ffi.h>
#include <sgl_kernel/tensor.h>
#include <sgl_kernel/utils.h>

#include <sgl_kernel/math.cuh>
#include <sgl_kernel/runtime.cuh>
#include <sgl_kernel/type.cuh>
#include <sgl_kernel/utils.cuh>
#include <sgl_kernel/vec.cuh>

#include <sgl_kernel/distributed/communicator.cuh>

#include <dlpack/dlpack.h>
#include <tvm/ffi/extra/stl.h>
#include <tvm/ffi/object.h>

#include <algorithm>
#include <bit>
#include <cstdint>
#include <cstdio>

namespace sglang {

namespace muse_vr_pull {

using device::distributed::Semaphore;

struct BoundedPullParams {
  const void* __restrict__ input;
  const void* __restrict__ residual;
  void* __restrict__ output;
  uint32_t num_vecs;
  // multicast buffer
  uint8_t* input_mc;
  uint8_t* output_mc;
  Semaphore* sem_local;
  Semaphore* sem_mc;
  uint32_t rank;
  uint32_t world_size;
  uint32_t max_polls;  // unsuccessful polls after which a barrier wait ends the kernel with a trap
};

inline constexpr uint32_t kPullCTASize = 512;  // max value

#define MUSE_VR_PULL_KERNEL __global__ __launch_bounds__(kPullCTASize, 1)

enum Primitive {
  RS = 0b01,     // Reduce-Scatter
  AG = 0b10,     // All-Gather
  AR = RS | AG,  // All-Reduce = RS + AG
};

template <typename vec_t>
SGL_DEVICE vec_t reduce_vec(vec_t x, vec_t y) {
  vec_t arr[2] = {x, y};
  return device::reduce_vec(arr);
}

// Reached only when a wait has used up its polls. Out of line so the poll loops keep their shape.
__device__ __noinline__ void barrier_wait_exceeded(
    uint32_t rank, uint32_t barrier, uint32_t arrived, uint32_t world_size, uint32_t polls) {
  printf(
      "muse_vr_plugin pull all-gather: rank %u block %u barrier %u (0 = entry, 1 = exit) still waiting after %u "
      "polls with %u of %u ranks arrived; trap\n",
      rank,
      blockIdx.x,
      barrier,
      polls,
      arrived,
      world_size);
  __trap();
}

/// SGLang's McBarrier (same window reservation, same signal and the same poll), with each wait bounded to
/// `max_polls` unsuccessful polls. One thread drives the whole barrier.
struct BoundedMcBarrier {
 public:
  SGL_DEVICE BoundedMcBarrier(Semaphore* local, Semaphore* mc, uint32_t world_size, uint32_t num_arrives)
      : m_counter(0), m_world_size(world_size), m_local(local), m_mc(mc) {
    if (threadIdx.x == 0) {
      m_counter = local[blockIdx.x].counter_ptr()->inc(num_arrives * world_size);
    }
  }

  template <bool kNeedFence>
  SGL_DEVICE void arrive(uint32_t n, uint32_t max_polls, uint32_t rank) const {
    if (threadIdx.x != 0) return;
    const auto window = m_counter + n * m_world_size;
    const auto bx = blockIdx.x;
    const auto semaphore = &m_local[bx];
    const auto mc_semaphore = &m_mc[bx];
    uint32_t polls = 0;
    if constexpr (kNeedFence) {
      mc_semaphore->put_release_multicast();
      while (semaphore->get_acquire() - window < m_world_size) {
        if (++polls == max_polls) [[unlikely]]
          barrier_wait_exceeded(rank, n, semaphore->get_relaxed() - window, m_world_size, polls);
      }
    } else {
      mc_semaphore->put_relaxed_multicast();
      while (semaphore->get_relaxed() - window < m_world_size) {
        if (++polls == max_polls) [[unlikely]]
          barrier_wait_exceeded(rank, n, semaphore->get_relaxed() - window, m_world_size, polls);
      }
    }
  }

  SGL_DEVICE void arrive_relaxed(uint32_t n, uint32_t max_polls, uint32_t rank) const {
    return this->arrive<false>(n, max_polls, rank);
  }

  SGL_DEVICE void arrive_rel_acq(uint32_t n, uint32_t max_polls, uint32_t rank) const {
    return this->arrive<true>(n, max_polls, rank);
  }

 private:
  uint32_t m_counter;  // window base; meaningful in thread 0, the sole poller
  uint32_t m_world_size;
  Semaphore* m_local;
  Semaphore* m_mc;
};

template <typename T, bool kHasResidual, Primitive kPrim, bool kUsePDL, uint32_t kPullUnroll>
MUSE_VR_PULL_KERNEL void muse_vr_pull_bounded_kernel(const __grid_constant__ BoundedPullParams params) {
  using namespace device;
  constexpr uint32_t kVecSize = 16 / sizeof(T);  // 16 bytes per vector
  using vec_t = device::AlignedVector<packed_t<T>, kVecSize / 2>;
  constexpr uint32_t kNumWarpVecs = kPullUnroll * kWarpThreads;

  // Round-robin chunks to blocks: the global warp index runs block-fastest, so
  // neighbouring chunks are driven by different CTAs.
  const auto warp_in_block = threadIdx.x / kWarpThreads;
  const auto global_warp_id = blockIdx.x + gridDim.x * warp_in_block;
  const auto lane_id = threadIdx.x % kWarpThreads;
  const auto num_warps = gridDim.x * (kPullCTASize / kWarpThreads);

  PDLWaitPrimary<kUsePDL>();
  const auto barrier = BoundedMcBarrier{params.sem_local, params.sem_mc, params.world_size, 2};
  barrier.arrive_relaxed(/*n=*/0, params.max_polls, params.rank);
  __syncthreads();

  const auto num_whole_chunks = params.num_vecs / kNumWarpVecs;
  // warp uniform unrolled path, 0 predicate
  for (auto chunk = global_warp_id; chunk < num_whole_chunks; chunk += num_warps) {
    vec_t vecs[kPullUnroll];
    const auto base = chunk * kNumWarpVecs + lane_id;

#pragma unroll
    for (uint32_t i = 0; i < kPullUnroll; ++i) {
      const auto vid = base + i * kWarpThreads;
      if constexpr (kPrim & Primitive::RS) {
        ptx::ld_multimem_16B(vecs[i], params.input_mc, vid);
      } else {
        ptx::ld_global_16B(vecs[i], params.input, vid);
      }
    }

    if constexpr (kHasResidual) {
      vec_t residuals[kPullUnroll];
#pragma unroll
      for (uint32_t i = 0; i < kPullUnroll; ++i) {
        residuals[i].load(params.residual, base + i * kWarpThreads);
      }
#pragma unroll
      for (uint32_t i = 0; i < kPullUnroll; ++i) {
        vecs[i] = reduce_vec(vecs[i], residuals[i]);
      }
    }

#pragma unroll
    for (uint32_t i = 0; i < kPullUnroll; ++i) {
      if constexpr (kPrim & Primitive::AG) {
        ptx::st_multimem_16B(vecs[i], params.output_mc, base + i * kWarpThreads);
      } else {
        ptx::st_global_16B(vecs[i], params.output, base + i * kWarpThreads);
      }
    }
  }

  const auto chunk_offset = num_whole_chunks * kNumWarpVecs;
  const auto global_tid = global_warp_id * kWarpThreads + lane_id;
  const auto global_threads = num_warps * kWarpThreads;
  for (auto vid = chunk_offset + global_tid; vid < params.num_vecs; vid += global_threads) {
    vec_t vec;
    if constexpr (kPrim & Primitive::RS) {
      ptx::ld_multimem_16B(vec, params.input_mc, vid);
    } else {
      ptx::ld_global_16B(vec, params.input, vid);
    }
    if constexpr (kHasResidual) {
      vec_t res;
      res.load(params.residual, vid);
      vec = reduce_vec(vec, res);
    }
    if constexpr (kPrim & Primitive::AG) {
      ptx::st_multimem_16B(vec, params.output_mc, vid);
    } else {
      ptx::st_global_16B(vec, params.output, vid);
    }
  }

  PDLTriggerSecondary<kUsePDL>();
  __syncthreads();
  if constexpr (kPrim & Primitive::AG) {
    barrier.arrive_rel_acq(/*n=*/1, params.max_polls, params.rank);
  } else {  // no store multimem, only local store
    barrier.arrive_relaxed(/*n=*/1, params.max_polls, params.rank);
  }
}

template <typename T, bool kUsePDL>
struct MuseVrPullBounded {
 private:
  using TensorView = tvm::ffi::TensorView;
  using PullPlaneObj = host::distributed::PullPlaneObj;
  using CommunicatorRef = host::distributed::CommunicatorRef;
  static constexpr uint32_t kVecBytes = 16;
  static constexpr uint32_t kVecSize = kVecBytes / sizeof(T);

 public:
  struct RouteInfo {
    uint32_t prefix_tokens;    // exclusive prefix sum
    uint32_t num_rank_tokens;  // current rank
  };

  static RouteInfo get_routing(uint32_t num_tokens, uint32_t rank, uint32_t world_size) {
    const auto avg = num_tokens / world_size;
    const auto rem = num_tokens % world_size;
    return {rank * avg + std::min(rank, rem), avg + (rank < rem ? 1 : 0)};
  }

  struct HostParams {
    int64_t hidden_size;
    DLDevice device;
  };

  static const void* get_residual_ptr(
      const tvm::ffi::Optional<TensorView>& residual,
      uint32_t domain_tokens,
      uint32_t total_tokens,
      uint32_t prefix_bytes) {
    if (!residual.has_value()) return nullptr;
    const auto tokens = static_cast<uint32_t>(residual.value().size(0));
    const auto* base = static_cast<const uint8_t*>(residual.value().data_ptr());
    if (tokens == domain_tokens) return base;
    CHECK_HOST(tokens == total_tokens) << "residual has " << tokens << " tokens, expected " << domain_tokens
                                       << " (this rank's shard) or " << total_tokens << " (the whole tensor)";
    return base + prefix_bytes;
  }

  static HostParams check_params(
      const TensorView in,
      const TensorView out,
      const tvm::ffi::Optional<TensorView>& residual = {},
      host::DebugInfo info = {}) {
    using namespace host;
    auto D = SymbolicSize{"hidden_size"};
    auto device_ = SymbolicDevice{};
    auto dtype_ = SymbolicDType{};
    if constexpr (!std::is_same_v<T, void>) dtype_.set_options<T>();
    device_.set_options<kDLCUDA>();
    TensorMatcher({-1, D})  //
        .with_dtype(dtype_)
        .with_device(device_)
        .verify(in, info);
    TensorMatcher({-1, D})  //
        .with_dtype(dtype_)
        .with_device(device_)
        .verify(out, info);
    if (residual.has_value()) {
      TensorMatcher({-1, D})  //
          .with_dtype(dtype_)
          .with_device(device_)
          .verify(residual.value(), info);
    }
    return {D.unwrap(), device_.unwrap()};
  }

 private:
  template <Primitive kPrim, uint32_t kPullUnroll>
  static void run_pull(
      const PullPlaneObj& pull,
      const TensorView in,
      const TensorView out,
      const tvm::ffi::Optional<TensorView> residual,
      uintptr_t in_mc_ptr,
      uintptr_t out_mc_ptr,
      uint32_t num_blocks_hint,
      uint32_t max_polls) {
    CHECK_HOST(pull.mc_semaphore != nullptr);
    const auto [hidden_size, device] = check_params(in, out, residual);
    const auto rank = pull.rank;
    const auto world_size = pull.world_size;
    const auto num_tokens = static_cast<uint32_t>(in.size(0));
    const auto out_tokens = static_cast<uint32_t>(out.size(0));
    const auto total_tokens = kPrim == Primitive::AG ? out_tokens : num_tokens;
    const auto routing = get_routing(total_tokens, rank, world_size);
    const auto num_vecs_per_token = static_cast<uint32_t>(hidden_size / kVecSize);
    const auto bytes_per_token = static_cast<uint32_t>(num_vecs_per_token * kVecBytes);
    const auto prefix_bytes = static_cast<uint32_t>(routing.prefix_tokens * bytes_per_token);

    // 0 = no hint, autotune; > 0 always use hint but clip to upper bound
    if constexpr (kPrim == Primitive::AR) {
      CHECK_HOST(num_tokens == out_tokens && in_mc_ptr != 0 && out_mc_ptr != 0);
      in_mc_ptr += prefix_bytes;
      out_mc_ptr += prefix_bytes;
      if (num_blocks_hint == 0) num_blocks_hint = host::div_ceil(256u, kPullUnroll * world_size);
    } else if constexpr (kPrim == Primitive::RS) {
      CHECK_HOST(out_tokens == routing.num_rank_tokens && in_mc_ptr != 0);
      in_mc_ptr += prefix_bytes;
      if (num_blocks_hint == 0) num_blocks_hint = pull.num_blocks;  // use all the blocks for RS
    } else {
      static_assert(kPrim == Primitive::AG);
      CHECK_HOST(num_tokens == routing.num_rank_tokens && out_mc_ptr != 0);
      out_mc_ptr += prefix_bytes;
      if (num_blocks_hint == 0) num_blocks_hint = host::div_ceil(128u, kPullUnroll * world_size);
    }
    /// NOTE: hard limit upper bound is `pull.num_blocks`
    num_blocks_hint = std::min(num_blocks_hint, pull.num_blocks);

    // Every pull primitive works on this rank's shard, so a full-length
    // residual is sliced onto it.
    const auto residual_ptr = get_residual_ptr(residual, routing.num_rank_tokens, total_tokens, prefix_bytes);
    const auto params = BoundedPullParams{
        .input = in.data_ptr(),
        .residual = residual_ptr,
        .output = out.data_ptr(),
        .num_vecs = static_cast<uint32_t>(routing.num_rank_tokens * num_vecs_per_token),
        .input_mc = std::bit_cast<uint8_t*>(in_mc_ptr),
        .output_mc = std::bit_cast<uint8_t*>(out_mc_ptr),
        .sem_local = pull.semaphores[rank],
        .sem_mc = pull.mc_semaphore,
        .rank = rank,
        .world_size = pull.world_size,
        .max_polls = max_polls,
    };

    /// NOTE: the final num_blocks resolution must be world unified, otherwise may deadlock
    const auto max_vecs_in_world = host::div_ceil(total_tokens, world_size) * num_vecs_per_token;
    const auto max_num_blocks = host::div_ceil(max_vecs_in_world, kPullUnroll * kPullCTASize);
    const auto num_blocks = std::max(1u, std::min(max_num_blocks, num_blocks_hint));
    const auto kernel = residual.has_value() ? muse_vr_pull_bounded_kernel<T, true, kPrim, kUsePDL, kPullUnroll>
                                             : muse_vr_pull_bounded_kernel<T, false, kPrim, kUsePDL, kPullUnroll>;
    host::LaunchKernel(num_blocks, kPullCTASize, device).enable_pdl(kUsePDL)(kernel, params);
  }

 public:
  template <uint32_t kPullUnroll>
  static void all_gather_pull(
      CommunicatorRef comm,  // only pull is needed
      TensorView in,
      TensorView out,
      tvm::ffi::Optional<TensorView> residual,
      int64_t out_mc_ptr,
      uint32_t num_blocks_hint,
      uint32_t max_polls) {
    CHECK_HOST(max_polls > 0) << "the barrier poll limit must be positive";
    return run_pull<Primitive::AG, kPullUnroll>(
        comm->get_pull_obj(), in, out, residual, 0, out_mc_ptr, num_blocks_hint, max_polls);
  }
};

#undef MUSE_VR_PULL_KERNEL

}  // namespace muse_vr_pull

}  // namespace sglang
