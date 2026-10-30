#pragma once

// OP embed: vocab-parallel embedding as TP4 sum 0.

#include "include/sm100/common/sync.cuh"

namespace m31_draft {
namespace {

// Vocab-parallel embedding: each rank pushes its shard's row of every token it
// owns and zeros for the rest, so TP4 sum 0 is the embedding.
__device__ void embedding_phase(Ctx &c) {
  uint32_t const gen = sum_generation(c, 0);
  uint64_t const flag = uint64_t(gen + 1u) << 32;
  int const buf = int(gen & 1u);
  constexpr int kChunks = kHidden / 8;
  for (int i = c.cta * kConsumerThreads + c.tid; i < kRows * kChunks;
       i += c.grid * kConsumerThreads) {
    int const t = i / kChunks, ch = i - t * kChunks;
    int64_t const local = c.misc->input_ids[t] - c.a.vocab_start;
    int4 v = make_int4(0, 0, 0, 0);
    if (local >= 0 && local < c.a.vocab_rows)
      v = __ldg(reinterpret_cast<int4 const *>(c.a.embed + size_t(local) * kHidden) + ch);
    uint64_t const w0 = uint64_t(uint32_t(v.x)) | flag, w1 = uint64_t(uint32_t(v.y)) | flag;
    uint64_t const w2 = uint64_t(uint32_t(v.z)) | flag, w3 = uint64_t(uint32_t(v.w)) | flag;
    for (int receiver = 0; receiver < kTp; ++receiver) {
      uint64_t *dst = ll_row(c.a.arena[receiver], buf, c.a.rank, t) + ch * 4;
      st_ll2(dst, w0, w1);
      st_ll2(dst + 2, w2, w3);
    }
  }
}

} // namespace
} // namespace m31_draft
