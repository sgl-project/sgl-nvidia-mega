// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
//
// The embedding rows of a multi-row MTP draft step's tokens without the vocabulary-parallel
// embedding's all-reduce (draft_embed_xchg): the rank whose vocabulary shard holds a row's token
// writes that row of its embedding table to every rank through the multicast mapping
// (mkd_xchg's transport: epoch-tagged 64-bit words, each carrying two bf16, three slots rotating
// by epoch, bounded waits that trap); every rank then reads the row from its own copy.
//
// The rows equal SGLang's: there every other rank contributes a zero row to the all-reduce, so
// the sum is the owner's row with -0.0 read as +0.0, which is what is written here.
//
// A rank may own every row of a call and so wait on no one; an arrival word per rank (written at
// entry, awaited by one CTA before the launch completes) keeps every rank within one call of the
// others, so a slot is never rewritten before every rank has read it.
//
// Grid (chunks, m): chunk c of row r moves 8-byte words [c * kThreads * kWords, ...) of the row.
// Built with NVRTC.

#include "mkd_common.cuh"

namespace sglang {

namespace draft_embed {

inline constexpr int kSlots = 3;
inline constexpr int kMaxRows = 128;
inline constexpr int kThreads = 256;
inline constexpr int kWords = 4;  // 8-byte words a thread moves
inline constexpr uint32_t kSpinLimit = 1u << 27;

// -0.0 as +0.0 in each bf16 half (zero rows added to it by an all-reduce)
SGL_DEVICE uint32_t plus_zero(uint32_t v) {
  if ((v & 0xFFFFu) == 0x8000u) v &= 0xFFFF0000u;
  if ((v >> 16) == 0x8000u) v &= 0x0000FFFFu;
  return v;
}

SGL_DEVICE uint32_t poll(const uint64_t* src, uint32_t tag) {
  uint64_t w;
  uint32_t spins = 0;
  do {
    asm volatile("ld.relaxed.sys.global.u64 %0, [%1];" : "=l"(w) : "l"(src) : "memory");
    if (++spins > kSpinLimit) __trap();
  } while (static_cast<uint32_t>(w >> 32) != tag);
  return static_cast<uint32_t>(w);
}

}  // namespace draft_embed

struct DraftEmbedParams {
  const int64_t* tokens;   // the token of row r at tokens[r * tokens_stride], written by an earlier grid
  const uint32_t* table;   // this rank's embedding shard, [vocab_local, hidden] bf16 as bf16 pairs
  uint32_t* out;           // [m, hidden] bf16 as bf16 pairs
  uint32_t* epoch;         // [2]: call counter shared by all ranks (from 1), CTA arrivals
  uint64_t* mc;            // multicast base of [kSlots, kMaxRows, hidden / 2] words
  const uint64_t* uc;      // this rank's own copy of it
  int64_t tokens_stride;
  int32_t m;
  int32_t hidden;
  int32_t vocab_start;     // global id of this rank's table row 0
  int32_t vocab_local;     // the ids this rank holds: [vocab_start, vocab_start + vocab_local)
  int32_t rank;
  int32_t world;
};

}  // namespace sglang

using sglang::DraftEmbedParams;

extern "C" __global__ void __launch_bounds__(256) draft_embed_xchg(const DraftEmbedParams __grid_constant__ p) {
  using namespace sglang::draft_embed;
  sglang::device::PDLWaitPrimary<true>();
  const int row = static_cast<int>(blockIdx.y);
  const int words = p.hidden / 2;
  // every CTA reads the call's tag before it arrives; the last arrival advances it
  const uint32_t tag = *reinterpret_cast<volatile uint32_t*>(p.epoch);
  const int64_t base = (static_cast<int64_t>(tag % kSlots) * kMaxRows + row) * words;
  const int64_t local = p.tokens[row * p.tokens_stride] - p.vocab_start;
  const int w0 = static_cast<int>(blockIdx.x) * kThreads * kWords + static_cast<int>(threadIdx.x);
  // the arrival words follow the kSlots * kMaxRows rows
  const int64_t arrive = static_cast<int64_t>(kSlots) * kMaxRows * words;
  const bool barrier = blockIdx.x == 0 && blockIdx.y == 0;
  if (barrier && threadIdx.x == 0) {
    asm volatile("multimem.st.relaxed.sys.global.b64 [%0], %1;" ::"l"(p.mc + arrive + p.rank),
                 "l"(static_cast<uint64_t>(tag)) : "memory");
  }
  if (local >= 0 && local < p.vocab_local) {
    const uint32_t* src = p.table + local * words;
#pragma unroll
    for (int j = 0; j < kWords; ++j) {
      const int w = w0 + j * kThreads;
      if (w < words) {
        const uint64_t word = static_cast<uint64_t>(plus_zero(src[w])) | (static_cast<uint64_t>(tag) << 32);
        asm volatile("multimem.st.relaxed.sys.global.b64 [%0], %1;" ::"l"(p.mc + base + w), "l"(word) : "memory");
      }
    }
  }
  // the thread's words in flight together, then each re-polled only while its tag is stale
  uint64_t v[kWords];
#pragma unroll
  for (int j = 0; j < kWords; ++j) {
    const int w = w0 + j * kThreads;
    if (w < words) asm volatile("ld.relaxed.sys.global.u64 %0, [%1];" : "=l"(v[j]) : "l"(p.uc + base + w) : "memory");
  }
#pragma unroll
  for (int j = 0; j < kWords; ++j) {
    const int w = w0 + j * kThreads;
    if (w < words)
      p.out[static_cast<int64_t>(row) * words + w] =
          static_cast<uint32_t>(v[j] >> 32) == tag ? static_cast<uint32_t>(v[j]) : poll(p.uc + base + w, tag);
  }
  if (barrier && static_cast<int>(threadIdx.x) < p.world) {
    // every rank has entered this call (tags only grow: a rank one call ahead also passes)
    uint64_t w;
    uint32_t spins = 0;
    do {
      asm volatile("ld.relaxed.sys.global.u64 %0, [%1];" : "=l"(w) : "l"(p.uc + arrive + threadIdx.x) : "memory");
      if (++spins > kSpinLimit) __trap();
    } while (static_cast<uint32_t>(w) - tag >= 0x80000000u);
  }
  __syncthreads();
  if (threadIdx.x == 0) {
    __threadfence();
    if (atomicAdd(p.epoch + 1, 1u) == gridDim.x * gridDim.y - 1) {
      p.epoch[1] = 0u;
      p.epoch[0] = tag + 1;
    }
  }
}
