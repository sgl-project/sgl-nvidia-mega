// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
//
// Benchmark only (runtime/bench.py, SGLANG_NVIDIA_MEGAKERNEL_BENCH_STEP_TIMES): one
// timestamp per covered graph replay, captured into the graph on a forked stream, so a
// served A/B gets every step's period without a profiler. Kept short (one CTA, no host
// reads after the first replay): it shares the GPU with the step's first kernels.
//
// Thread 0 reads %globaltimer first, then takes the next event index from a device
// counter. ``table`` (pinned host words, filled by the host when the capture ends:
// {count, (pointer, n | int64 << 32 | half << 40) ...}) lists the routed-expert id
// tensors of the same graph, which still hold its previous replay's choice; the first
// replay copies it into ``dev_table`` (device) and later replays read that copy. Per
// tensor a warp counts the distinct ids (>= 0) and those below ``half`` (the first
// expert-parallel half) with a shared-memory bitmap, into the counts ring; with
// ``ids_ring`` the raw ids are copied as int16. Thread 0 then writes the event {time,
// meta, position, seq} with seq (index + 1) last, after a system-scope fence, so a host
// reader that sees seq == index + 1 sees the whole event.

constexpr int kMaxEntries = 64;
constexpr int kMaxIds = 64;      // per tensor: rows * top_k
constexpr int kTableWords = 1 + 2 * kMaxEntries;
constexpr int kMaxExperts = 512;  // bitmap bits per warp
constexpr int kWarps = 8;

__device__ __forceinline__ long long LoadId(const void* p, bool is64, int i) {
  return is64 ? static_cast<const long long*>(p)[i]
              : static_cast<long long>(static_cast<const int*>(p)[i]);
}

extern "C" __global__ void __launch_bounds__(kWarps * 32) dsv41_bench_stamp(
    unsigned long long* __restrict__ events,             // pinned [cap][4]
    short* __restrict__ counts_ring,                     // pinned [cap][2 * kMaxEntries], or null
    short* __restrict__ ids_ring,                        // pinned [cap][stride], or null
    unsigned int* __restrict__ counter,                  // device, one word
    const unsigned long long* __restrict__ table,        // pinned row, or null
    unsigned long long* __restrict__ dev_table,          // device row (cache of table)
    int* __restrict__ dev_valid,                         // device word: dev_table holds table
    const void* __restrict__ positions, int positions64, unsigned int cap, int stride,
    unsigned long long meta) {
  __shared__ unsigned int s_idx;
  __shared__ unsigned long long s_time;
  __shared__ int s_valid;
  __shared__ unsigned int s_bits[kWarps][kMaxExperts / 32];
  if (threadIdx.x == 0) {
    unsigned long long t;
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
    s_time = t;
    s_idx = atomicAdd(counter, 1U);
    s_valid = table != nullptr ? *dev_valid : 0;
  }
  __syncthreads();
  const unsigned int idx = s_idx;
  const unsigned int slot = idx % cap;
  if (table != nullptr) {
    const unsigned long long* row = dev_table;
    if (!s_valid) {  // first replay: cache the host row on the device
      for (int i = threadIdx.x; i < kTableWords; i += blockDim.x) dev_table[i] = table[i];
      __syncthreads();
      if (threadIdx.x == 0) {
        __threadfence();
        *dev_valid = 1;
      }
    }
    const int entries = min(static_cast<int>(row[0]), kMaxEntries);
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    if (counts_ring != nullptr) {
      short* out = counts_ring + static_cast<unsigned long long>(slot) * (2 * kMaxEntries);
      unsigned int* bits = s_bits[warp];
      for (int j = warp; j < kMaxEntries; j += kWarps) {
        int distinct = -1, low = -1;
        if (j < entries) {
          const void* p = reinterpret_cast<const void*>(row[1 + 2 * j]);
          const unsigned long long m = row[2 + 2 * j];
          const int n = min(static_cast<int>(m & 0xffffffffULL), kMaxIds);
          const bool is64 = ((m >> 32) & 0xff) != 0;
          const int half = static_cast<int>((m >> 40) & 0xffff);
          for (int w = lane; w < kMaxExperts / 32; w += 32) bits[w] = 0U;
          __syncwarp();
          for (int i = lane; i < n; i += 32) {
            const long long v = LoadId(p, is64, i);
            if (v >= 0 && v < kMaxExperts) atomicOr(&bits[v >> 5], 1U << (v & 31));
          }
          __syncwarp();
          int d = 0, l = 0;
          if (lane < kMaxExperts / 32) {
            const unsigned int word = bits[lane];
            d = __popc(word);
            const int below = half - 32 * lane;  // bits of this word below half
            l = below >= 32 ? d : (below > 0 ? __popc(word & ((1U << below) - 1U)) : 0);
          }
#pragma unroll
          for (int o = 16; o > 0; o >>= 1) {
            d += __shfl_xor_sync(0xffffffffU, d, o);
            l += __shfl_xor_sync(0xffffffffU, l, o);
          }
          distinct = d;
          low = l;
          __syncwarp();
        }
        if (lane == 0) {
          out[2 * j] = static_cast<short>(distinct);
          out[2 * j + 1] = static_cast<short>(low);
        }
      }
    }
    if (ids_ring != nullptr) {
      short* out = ids_ring + static_cast<unsigned long long>(slot) * stride;
      int base = 0;
      for (int j = 0; j < entries; ++j) {
        const void* p = reinterpret_cast<const void*>(row[1 + 2 * j]);
        const unsigned long long m = row[2 + 2 * j];
        const int n = static_cast<int>(m & 0xffffffffULL);
        const bool is64 = ((m >> 32) & 0xff) != 0;
        for (int i = threadIdx.x; i < n && base + i < stride; i += blockDim.x) {
          out[base + i] = static_cast<short>(LoadId(p, is64, i));
        }
        base += n;
      }
    }
    __threadfence_system();
  }
  __syncthreads();
  if (threadIdx.x == 0) {
    long long pos = -1;
    if (positions != nullptr) {
      pos = positions64 ? static_cast<const long long*>(positions)[0]
                        : static_cast<long long>(static_cast<const int*>(positions)[0]);
    }
    unsigned long long* e = events + static_cast<unsigned long long>(slot) * 4;
    e[0] = s_time;
    e[1] = meta;
    e[2] = static_cast<unsigned long long>(pos);
    __threadfence_system();
    reinterpret_cast<volatile unsigned long long*>(e)[3] = static_cast<unsigned long long>(idx) + 1ULL;
  }
}
