// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
//
// Adapted from SGLang (https://github.com/sgl-project/sglang),
// python/sglang/kernels/jit/csrc/gemm/tiny_gemm/mma.cuh (Apache License 2.0, Copyright SGLang
// Team): tiny_mma_kernel, the tensor-core router GEMM SGLang's MoEGate runs at decode widths
// (BF16 x and w, FP32 logits). Modified by NVIDIA for slot gate_route_early:
//
//   * a standalone NVRTC build of one schedule (ROUTER_N, ROUTER_K, ROUTER_WARPS,
//     ROUTER_CLUSTER, ROUTER_MTILES: SGLang's own _mma_schedule for the router's shape, read at
//     build), Blackwell's 32-byte lane groups, the parameters as plain arguments;
//   * the same loads, MMAs (m16n8k16, one chunk per accumulator, chunks added in FP32 from the
//     first), warp sums in warp order and cluster sums in rank order, so the logits are
//     SGLang's bit for bit;
//   * with ROUTER_EARLY_RELEASE the grid releases its programmatic dependents once its weight
//     loads are issued (before its own griddepcontrol.wait, as TensorRT-LLM's low-M router GEMM
//     does) instead of after its cluster reduction, so the router's top-k (gate_route.cuh)
//     launches while this GEMM runs and does its setup before its own griddepcontrol.wait,
//     which still returns only once this grid has completed and its stores are visible.

#ifndef ROUTER_EARLY_RELEASE
#define ROUTER_EARLY_RELEASE 1
#endif

namespace {

constexpr unsigned int kN = ROUTER_N;
constexpr unsigned int kK = ROUTER_K;
constexpr unsigned int kNumWarps = ROUTER_WARPS;
constexpr unsigned int kClusterSize = ROUTER_CLUSTER;
constexpr unsigned int kMTiles = ROUTER_MTILES;
constexpr unsigned int kWarpThreads = 32;
constexpr unsigned int kBlockSize = kNumWarps * kWarpThreads;
constexpr unsigned int kTileN = 16;
constexpr unsigned int kTileM = 8;
constexpr unsigned int kChunkK = 64;
constexpr unsigned int kNumChunks = kK / kChunkK;
constexpr unsigned int kNumSplits = kNumWarps * kClusterSize;
constexpr unsigned int kMaxChunks = (kNumChunks + kNumSplits - 1) / kNumSplits;
constexpr bool kEvenSplit = kNumChunks % kNumSplits == 0;
// Blackwell: a lane's 16 elements of a chunk are one 32-byte group (kVecBytes 32, kGroup 16)
constexpr unsigned int kWords = 8;
constexpr unsigned int kCTAsPerSM = kBlockSize <= 256 ? 2 : 1;
constexpr unsigned int kNumTokens = kMTiles * kTileM;
constexpr unsigned int kNumOutputs = kNumTokens * kTileN;
constexpr unsigned int kSlice = (kNumOutputs + kClusterSize - 1) / kClusterSize;

static_assert(kN % kTileN == 0 && kK % kChunkK == 0);
static_assert(kNumSplits <= kNumChunks, "every warp needs at least one chunk");
static_assert(kClusterSize >= 1 && kClusterSize <= 8, "portable cluster sizes only");

struct Words {
  unsigned int w[kWords];
};

__device__ __forceinline__ void LoadWords(Words& r, const unsigned short* p) {
  const uint4* q = reinterpret_cast<const uint4*>(p);
  const uint4 a = q[0];
  const uint4 b = q[1];
  r.w[0] = a.x, r.w[1] = a.y, r.w[2] = a.z, r.w[3] = a.w;
  r.w[4] = b.x, r.w[5] = b.y, r.w[6] = b.z, r.w[7] = b.w;
}

__device__ __forceinline__ void Mma(float (&d)[4], const unsigned int (&a)[4], const unsigned int (&b)[2]) {
  asm("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, "
      "{%0, %1, %2, %3};"
      : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
}

// sums[i] += x[token tile i] . w over one 64-element chunk (SGLang's mma_chunk)
__device__ __forceinline__ void MmaChunk(float (&sums)[kMTiles][4], const Words (&w)[2], const Words (&x)[kMTiles]) {
#pragma unroll
  for (unsigned int s = 0; s < 4; ++s) {
    const unsigned int a[4] = {w[0].w[2 * s], w[1].w[2 * s], w[0].w[2 * s + 1], w[1].w[2 * s + 1]};
#pragma unroll
    for (unsigned int i = 0; i < kMTiles; ++i) {
      const unsigned int b[2] = {x[i].w[2 * s], x[i].w[2 * s + 1]};
      Mma(sums[i], a, b);
    }
  }
}

__device__ __forceinline__ unsigned int ToShared(const void* p) {
  return static_cast<unsigned int>(__cvta_generic_to_shared(p));
}

__device__ __forceinline__ unsigned int Mapa(unsigned int address, unsigned int rank) {
  unsigned int r;
  asm volatile("mapa.shared::cluster.u32 %0, %1, %2;" : "=r"(r) : "r"(address), "r"(rank));
  return r;
}

__device__ __forceinline__ void ReleaseDependents() {
  asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
}

}  // namespace

extern "C" __global__ __launch_bounds__(kBlockSize, kCTAsPerSM) void dsv41_router_mma(
    const unsigned short* __restrict__ x, const unsigned short* __restrict__ w, float* __restrict__ out,
    long long stride_x, unsigned int num_tokens) {
  __shared__ __align__(16) float s_partial[kNumWarps][kNumOutputs];
  __shared__ float s_inbox[kClusterSize][kClusterSize > 1 ? kSlice : 1];
  __shared__ __align__(8) unsigned long long s_bar;

  const unsigned int warp_id = threadIdx.x / kWarpThreads;
  const unsigned int lane_id = threadIdx.x % kWarpThreads;
  const unsigned int g = lane_id / 4;
  const unsigned int t = lane_id % 4;
  const unsigned int rank = blockIdx.x % kClusterSize;
  const unsigned int tile = blockIdx.x / kClusterSize;
  const unsigned int token_begin = blockIdx.y * kNumTokens;
  const unsigned int split = rank * kNumWarps + warp_id;
  const unsigned int chunk_begin = split * kNumChunks / kNumSplits;
  const unsigned int num_chunks = (split + 1) * kNumChunks / kNumSplits - chunk_begin;

  // the weight does not depend on the producer: all of it before the wait
  const unsigned short* w_row = w + static_cast<unsigned long long>(tile * kTileN + g) * kK + chunk_begin * 64 + t * 16;
  Words w_reg[kMaxChunks][2];
#pragma unroll
  for (unsigned int c = 0; c < kMaxChunks; ++c) {
    if (kEvenSplit || c < num_chunks) {
#pragma unroll
      for (unsigned int h = 0; h < 2; ++h) LoadWords(w_reg[c][h], w_row + h * 8 * kK + c * 64);
    }
  }
#if ROUTER_EARLY_RELEASE
  ReleaseDependents();  // the weight loads are issued
#endif

  if constexpr (kClusterSize > 1) {
    if (threadIdx.x == 0) {
      const unsigned int bar = ToShared(&s_bar);
      asm volatile("mbarrier.init.shared.b64 [%0], 1;" ::"r"(bar) : "memory");
      asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
      const unsigned int slice = min(kSlice, kNumOutputs - rank * kSlice);
      asm volatile("mbarrier.arrive.expect_tx.shared.b64 _, [%0], %1;" ::"r"(bar),
                   "r"(kClusterSize * slice * 4u)
                   : "memory");
    }
    asm volatile("barrier.cluster.arrive.relaxed.aligned;" ::: "memory");
  }

  asm volatile("griddepcontrol.wait;" ::: "memory");

  const unsigned short* x_row[kMTiles];
#pragma unroll
  for (unsigned int i = 0; i < kMTiles; ++i) {
    const unsigned int m = min(token_begin + i * 8 + g, num_tokens - 1);
    x_row[i] = x + static_cast<long long>(m) * stride_x + chunk_begin * 64 + t * 16;
  }

  float acc[kMTiles][4];
#pragma unroll
  for (unsigned int c = 0; c < kMaxChunks; ++c) {
    if (kEvenSplit || c < num_chunks) {
      Words x_reg[kMTiles];
#pragma unroll
      for (unsigned int i = 0; i < kMTiles; ++i) LoadWords(x_reg[i], x_row[i] + c * 64);
      // the tensor cores truncate while accumulating: their sums stay one chunk, added in FP32
      float chunk_acc[kMTiles][4] = {};
      MmaChunk(chunk_acc, w_reg[c], x_reg);
#pragma unroll
      for (unsigned int i = 0; i < kMTiles; ++i) {
#pragma unroll
        for (unsigned int r = 0; r < 4; ++r) acc[i][r] = c == 0 ? chunk_acc[i][r] : acc[i][r] + chunk_acc[i][r];
      }
    }
  }

  float* const out_tile = out + tile * kTileN;
  const auto store = [&](unsigned int o, float value) {
    const unsigned int lane = o / 4 % kWarpThreads;
    const unsigned int token = token_begin + o / 128 * 8 + lane % 4 * 2 + o % 2;
    const unsigned int row = lane / 4 + o % 4 / 2 * 8;
    if (token < num_tokens) out_tile[token * kN + row] = value;
  };
  if constexpr (kNumWarps == 1 && kClusterSize == 1) {
#if !ROUTER_EARLY_RELEASE
    ReleaseDependents();
#endif
#pragma unroll
    for (unsigned int i = 0; i < kMTiles; ++i) {
#pragma unroll
      for (unsigned int r = 0; r < 4; ++r) store((i * kWarpThreads + lane_id) * 4 + r, acc[i][r]);
    }
    return;
  }

#pragma unroll
  for (unsigned int i = 0; i < kMTiles; ++i) {
    *reinterpret_cast<float4*>(&s_partial[warp_id][(i * kWarpThreads + lane_id) * 4]) =
        make_float4(acc[i][0], acc[i][1], acc[i][2], acc[i][3]);
  }
  __syncthreads();

  const auto cta_sum = [&](unsigned int o) {
    float sum = s_partial[0][o];
#pragma unroll
    for (unsigned int k = 1; k < kNumWarps; ++k) sum += s_partial[k][o];
    return sum;
  };

  if constexpr (kClusterSize == 1) {
#if !ROUTER_EARLY_RELEASE
    ReleaseDependents();
#endif
    for (unsigned int o = threadIdx.x; o < kNumOutputs; o += kBlockSize) store(o, cta_sum(o));
  } else {
    asm volatile("barrier.cluster.wait.aligned;" ::: "memory");  // the peers' mbarriers are initialized
    const unsigned int inbox = ToShared(&s_inbox[rank][0]);
    const unsigned int bar = ToShared(&s_bar);
    for (unsigned int o = threadIdx.x; o < kNumOutputs; o += kBlockSize) {
      const unsigned int dst = o / kSlice;
      const unsigned int byte_offset = (o % kSlice) * 4u;
      asm volatile("st.async.shared::cluster.mbarrier::complete_tx::bytes.b32 [%0], %1, [%2];" ::"r"(
                       Mapa(inbox + byte_offset, dst)),
                   "f"(cta_sum(o)), "r"(Mapa(bar, dst))
                   : "memory");
    }
    asm volatile(
        "{\n\t.reg .pred p;\n\t"
        "WAIT_%=: mbarrier.try_wait.parity.shared.b64 p, [%0], 0;\n\t"
        "@!p bra WAIT_%=;\n\t}\n" ::"r"(bar)
        : "memory");
#if !ROUTER_EARLY_RELEASE
    ReleaseDependents();
#endif
    const unsigned int slice = min(kSlice, kNumOutputs - rank * kSlice);
    for (unsigned int o = threadIdx.x; o < slice; o += kBlockSize) {
      float sum = s_inbox[0][o];
#pragma unroll
      for (unsigned int r = 1; r < kClusterSize; ++r) sum += s_inbox[r][o];
      store(rank * kSlice + o, sum);
    }
  }
}
