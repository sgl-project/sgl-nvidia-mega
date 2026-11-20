// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
//
// Derived from muLLM v0.4.1 (NVIDIA Corporation, Apache-2.0): its DeepSeek-V4.1 Engram
// kernels' TP all-gather of the Engram projection's shards (the publish kernel, and the
// mailbox read and epoch advance of the gate-collect kernel). Changes: the world, the shard
// width and the row count are arguments; the collect writes the gathered BF16 rows
// out for SGLang's own gate instead of running a fused gate; every wait has a
// deadline (the plugin's fail-stop); one arena region per Engram layer.
//
// Slot engram_publish: the Engram prefetch's WKV rows reach every rank without a
// collective. Per Engram layer, inside a symmetric-memory buffer on every rank:
//   payload[stage 2][row max_rows][column n]  u32 words (tag << 16) | BF16 bits
//   control[0] epoch, control[1] arrivals     this rank's own words
// A word is valid for epoch e when its high half is the tag 1 + e % 65535 (an empty
// arena is all zero, which no tag matches), so every BF16 bit pattern travels as is.
// Epoch e uses stage e % 2; a rank cannot publish epoch e + 2 before every rank has
// consumed epoch e, since its epoch e + 1 collect waits for all ranks' e + 1 shards.
//
//   dsv41_engram_publish (the Engram side stream, after the rank's WKV columns): stores
//     the rank's columns [rows, n / world] as tagged words into every rank's mailbox
//     (itself included). It only stores; it never waits.
//   dsv41_engram_collect (the main stream, at the layer's gate): reads [rows, n] from
//     this rank's mailbox, waiting until every word carries this call's tag, and writes
//     the BF16 bits out. Its last CTA advances this rank's epoch for the next call.
//
// A kernel that spins on its peers off the main chain can join a cross-rank cycle, so
// the only waiting kernel is the collect, on the main stream, in the same order on
// every rank. Both kernels read the epoch from this rank's control word; the collect
// advances it only after every collect CTA has read every word, which needs every
// publish CTA of this rank to have stored (so to have read the epoch) first, and the
// next publish on the arena is in the next step. A wait past timeout_ns stores to
// address 0 and traps, so a rank whose peer stopped faults instead of hanging.
#include <cuda_bf16.h>

using u32 = unsigned int;
using u64 = unsigned long long;
using i64 = long long;

namespace {

constexpr int kThreads = 256;

__device__ __forceinline__ void StoreWords(u64 base, i64 word, uint4 value) {
  asm volatile("st.global.relaxed.sys.v4.u32 [%0], {%1, %2, %3, %4};" ::"l"(
                   reinterpret_cast<u32*>(base) + word),
               "r"(value.x), "r"(value.y), "r"(value.z), "r"(value.w)
               : "memory");
}

__device__ __forceinline__ uint4 LoadWords(const u32* address) {
  uint4 value;
  asm volatile("ld.global.relaxed.sys.v4.u32 {%0, %1, %2, %3}, [%4];"
               : "=r"(value.x), "=r"(value.y), "=r"(value.z), "=r"(value.w)
               : "l"(address)
               : "memory");
  return value;
}

__device__ __forceinline__ u64 GlobalTimer() {
  u64 t;
  asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
  return t;
}

// A missed wait deadline: store to address 0x0, then trap (the plugin's fail-stop).
__device__ __noinline__ void FaultAtNull() {
  asm volatile(
      "{\n\t.reg .u64 a;\n\tmov.u64 a, 0;\n\tst.volatile.global.u32 [a], 0;\n\ttrap;\n\t}" ::: "memory");
}

__device__ __forceinline__ u32 ReadEpoch(const u32* control) {
  __shared__ u32 s_epoch;
  if (threadIdx.x == 0) s_epoch = *reinterpret_cast<const volatile u32*>(control);
  __syncthreads();
  return s_epoch;
}

__device__ __forceinline__ u32 EpochTag(u32 epoch) {
  return (1U + epoch % 65535U) << 16;
}

__device__ __forceinline__ bool Tagged(uint4 w, u32 tag) {
  return (w.x & 0xFFFF0000U) == tag && (w.y & 0xFFFF0000U) == tag &&
         (w.z & 0xFFFF0000U) == tag && (w.w & 0xFFFF0000U) == tag;
}

}  // namespace

// kv_local [rows, cols] BF16 (contiguous; cols = n / world, a multiple of 4); peers: the
// world ranks' arena bases (device array); payload / control: byte offsets of this
// layer's region inside every arena.
extern "C" __global__ void __launch_bounds__(kThreads)
dsv41_engram_publish(const __nv_bfloat16* __restrict__ kv_local,
                     const u64* __restrict__ peers, i64 payload_offset,
                     i64 control_offset, int rank, int world, int rows, int n,
                     int max_rows) {
  const u32* const control = reinterpret_cast<const u32*>(
      peers[rank] + static_cast<u64>(control_offset));
  const u32 epoch = ReadEpoch(control);
  const u32 tag = EpochTag(epoch);
  const int cols = n / world;
  const i64 stage = static_cast<i64>(epoch & 1U) * max_rows * n;
  const int groups = rows * (cols / 4);
  const int stride = static_cast<int>(gridDim.x * blockDim.x);
  for (int group = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x); group < groups;
       group += stride) {
    const int row = group / (cols / 4);
    const int column = (group % (cols / 4)) * 4;
    const uint2 raw = *reinterpret_cast<const uint2*>(
        kv_local + static_cast<i64>(row) * cols + column);
    const uint4 words = make_uint4(tag | (raw.x & 0xFFFFU), tag | (raw.x >> 16),
                                   tag | (raw.y & 0xFFFFU), tag | (raw.y >> 16));
    const i64 word =
        stage + static_cast<i64>(row) * n + static_cast<i64>(rank) * cols + column;
    for (int peer = 0; peer < world; ++peer) {
      StoreWords(peers[peer] + static_cast<u64>(payload_offset), word, words);
    }
  }
}

// out [rows, n] BF16 (contiguous): every rank's columns from this rank's mailbox.
extern "C" __global__ void __launch_bounds__(kThreads)
dsv41_engram_collect(const u64* __restrict__ peers, i64 payload_offset,
                     i64 control_offset, int rank, int rows, int n, int max_rows,
                     __nv_bfloat16* __restrict__ out, i64 timeout_ns) {
  u32* const control =
      reinterpret_cast<u32*>(peers[rank] + static_cast<u64>(control_offset));
  const u32 epoch = ReadEpoch(control);
  const u32 tag = EpochTag(epoch);
  const u32* const payload =
      reinterpret_cast<const u32*>(peers[rank] + static_cast<u64>(payload_offset)) +
      static_cast<i64>(epoch & 1U) * max_rows * n;
  const int groups = rows * (n / 4);
  const int stride = static_cast<int>(gridDim.x * blockDim.x);
  for (int group = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x); group < groups;
       group += stride) {
    const i64 word = static_cast<i64>(group) * 4;
    uint4 w = LoadWords(payload + word);
    if (!Tagged(w, tag)) {
      const u64 start = GlobalTimer();
      u32 spins = 0;
      do {
        w = LoadWords(payload + word);
        if ((++spins & 1023U) == 0 && GlobalTimer() - start > static_cast<u64>(timeout_ns)) {
          FaultAtNull();
        }
      } while (!Tagged(w, tag));
    }
    const uint2 bits = make_uint2((w.x & 0xFFFFU) | (w.y << 16), (w.z & 0xFFFFU) | (w.w << 16));
    *reinterpret_cast<uint2*>(out + word) = bits;
  }
  // Advance the epoch once every CTA has read its words of this stage; the next publish
  // on this arena is in the next step (stream ordered after this grid).
  __syncthreads();
  if (threadIdx.x == 0) {
    __threadfence();
    const u32 ctas = gridDim.x;
    const u32 previous = atomicAdd(control + 1, 1U);
    if (previous == ctas - 1) {
      control[1] = 0U;
      __threadfence();
      atomicExch(control, epoch + 1U);
    }
  }
}
