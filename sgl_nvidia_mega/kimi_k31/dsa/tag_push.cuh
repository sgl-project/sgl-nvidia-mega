// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// K31MK_DSA_REQ_SHARD_TKPUSH (default off; dsa/req_shard.py, dsa/tkpush.py): the store half of dsa/tag_gather.py's
// flag-free gather, done by the kernel that produces the owner rows (the x47v2 small-batch cluster top-k,
// dsa/x47960/topk_v2_tkpush.cuh) instead of a separate _push_kernel launch.
//
// Word format and layout are tag_gather.py's: slot id v (-1 <= v < 2**25 - 1) travels as (tag << 25) | (v + 1),
// tag = epoch & 0x7F, stored with st.relaxed.sys through each rank's symmetric-memory unicast mapping into copy c of
// that rank's mailbox at word (c * world + rank) * n_own + row * row_words + col (= _push_kernel's
// c * (W * n_own) + RANK * n_own + offs). The consumer (_poll_kernel, or the fused sparse core) is unchanged.
//
// Epoch: the producer reads the consumer's own use counter of copy c (poll CTA 0, or the fused core's row 0 / split 0;
// every counter of a copy advances once per use on every rank) and stores tag (counter + 1) & 0x7F, i.e. the epoch the
// consumers of this call wait for. It keeps no counter of its own, so a call that pushes with tag_gather.py's
// _push_ext_kernel instead (a launch path without the in-kernel push) uses the same epoch source.
//
// Release: every storing thread executes fence.acq_rel.sys after the producer's griddepcontrol.wait and before its first
// store (begin()); that fence followed by the strong (.relaxed.sys) stores is the release pattern the consumer's
// ld.relaxed.sys + fence.acq_rel.sys acquire synchronizes with (PTX ISA 8.8 release / acquire patterns), exactly the
// _push_kernel's fence-then-st.relaxed.sys. Ordering argument and reuse chain: see tag_gather.py.
#pragma once

#include <cstdint>

#ifndef K31_TKP_NOFENCE
#define K31_TKP_NOFENCE 0  // 1 = K31MK_DSA_REQ_SHARD_TAG_NOFENCE (TAGNF, fails closed): no release fence
#endif

namespace k31_tkpush {

inline constexpr uint32_t kMaxPeers = 16;  // TP16; the host refuses a larger world

struct TagPush {
  uint64_t peers[kMaxPeers];  // each rank's mailbox base address (its symmetric-memory unicast mapping), int32 words
  const int32_t* epoch;       // the consumer's use counter of copy c; this call's tag = (*epoch + 1) & 0x7F
  int64_t base;               // word offset of this rank's block in copy c: c * (world * n_own) + rank * n_own
  int64_t row_words;          // words per owner row in the block (the top-k output row stride, 2048)
  uint32_t world;             // ranks (<= kMaxPeers)
};

__device__ __forceinline__ void fence_acq_rel_sys() {
  asm volatile("fence.acq_rel.sys;" ::: "memory");
}

__device__ __forceinline__ void st_relaxed_sys(uint32_t* p, uint32_t v) {
  asm volatile("st.relaxed.sys.global.b32 [%0], %1;" ::"l"(p), "r"(v) : "memory");
}

/// Call once per thread, after griddepcontrol.wait (every grid this one depends on is complete, so the counter holds
/// the last use's value) and before the thread's first store(). Returns the tag bits (tag << 25).
__device__ __forceinline__ uint32_t tag_only(const TagPush& p) {  // the tag bits without the fence (caller fences)
  uint32_t e = 0;
  asm volatile("ld.relaxed.gpu.global.b32 %0, [%1];" : "=r"(e) : "l"(p.epoch) : "memory");
  return ((e + 1u) & 0x7Fu) << 25;
}

__device__ __forceinline__ uint32_t begin(const TagPush& p) {
  const uint32_t tagv = tag_only(p);
  if constexpr (K31_TKP_NOFENCE) {
    // TAGNF: copy reuse rests on the served layer chain; a lost order shows the next use's tag -> the readers trap
  } else {
    fence_acq_rel_sys();  // release: this rank's earlier reads of its mailboxes precede the stores below
  }
  return tagv;
}

__device__ __forceinline__ uint32_t encode(int32_t v, uint32_t tagv) {
  return tagv | (static_cast<uint32_t>(v + 1) & 0x1FFFFFFu);
}

/// Store slot id ``v`` of owner row ``row``, column ``col`` into copy c of every rank's mailbox (this rank included).
__device__ __forceinline__ void store(const TagPush& p, uint32_t tagv, uint32_t row, uint32_t col, int32_t v) {
  const uint32_t w = encode(v, tagv);
  const int64_t off = p.base + static_cast<int64_t>(row) * p.row_words + col;
#pragma unroll
  for (uint32_t j = 0; j < kMaxPeers; ++j) {
    if (j < p.world) {
      st_relaxed_sys(reinterpret_cast<uint32_t*>(p.peers[j]) + off, w);
    }
  }
}

}  // namespace k31_tkpush
