#pragma once

// The weight stream: the per-GEMM schedule dispatch and the producer loop.

#include "include/sm100/collective/op_gemm_down.cuh"
#include "include/sm100/collective/op_gemm_gate_up.cuh"
#include "include/sm100/collective/op_gemm_o.cuh"
#include "include/sm100/collective/op_gemm_qkv.cuh"

namespace m31_draft {
namespace {

// The per-GEMM schedule and gate words, from the four GEMM OPs.
__host__ __device__ constexpr int gemm_units(int g) {
  return g == kGemmQkv      ? GemmQkv::units()
         : g == kGemmO      ? GemmO::units()
         : g == kGemmGateUp ? GemmGateUp::units()
                            : GemmDown::units();
}
__device__ int gemm_stages(int g, int u) {
  return g == kGemmQkv      ? GemmQkv::stages(u)
         : g == kGemmO      ? GemmO::stages(u)
         : g == kGemmGateUp ? GemmGateUp::stages(u)
                            : GemmDown::stages(u);
}
__device__ int gemm_gate_word(int g, int grid) {
  return g == kGemmQkv      ? GemmQkv::gate_word(grid)
         : g == kGemmO      ? GemmO::gate_word(grid)
         : g == kGemmGateUp ? GemmGateUp::gate_word(grid)
                            : GemmDown::gate_word(grid);
}

// gate_up and down end with a last wave of units on part of the CTAs, whose remaining stages still
// stream from memory. A CTA outside that wave finished its units earlier: it issues the next GEMM's
// first stages (not needed before the next phase) only after the last wave's gate word is visible,
// so they do not compete for memory with the last wave's stages. The word's writer is a last-wave
// CTA's consumer, which never waits on this producer; this CTA's consumers need the next GEMM only
// after the whole last wave, which includes that word.
__device__ void hold_next_gemm(uint8_t *arena, int rank, uint32_t epoch, int layer, int gemm,
                               int cta, int grid, int lane) {
  if ((gemm != kGemmGateUp && gemm != kGemmDown) || in_last_wave(cta, grid, gemm_units(gemm)))
    return;
  if (lane == 0) {
    uint64_t const *word;
    uint32_t want;
    if (gemm == kGemmGateUp) {
      word = swi_ll(arena, layer) + gemm_gate_word(kGemmGateUp, grid);
      want = ll_tag(epoch);
    } else {
      // this rank's own TP4 word of the last unit (sum 2 layer + 2), in this rank's arena
      uint32_t const gen = (epoch - 1u) * uint32_t(kSums) + uint32_t(2 * layer + 2);
      word = ll_row(arena, int(gen & 1u), rank, 0) + gemm_gate_word(kGemmDown, grid);
      want = gen + 1u;
    }
    unsigned long long start = 0;
    for (uint32_t spin = 0;; ++spin) {
      uint64_t const v = ld_relaxed_sys_u64(word);
      if (gemm == kGemmGateUp ? ll_ok(v, want) : uint32_t(v >> 32) == want)
        break;
      if ((spin & 63u) == 0u) {
        if (start == 0)
          start = globaltimer();
        else
          check_deadline(start);
      }
    }
  }
  __syncwarp();
}

// The weight stream: the last warp issues every stage of the call in consumption
// order; each stage's copies are issued by its GEMM's OP.
__device__ void producer(Args const &a, WeightMaps const &maps, Producer stream, SmemMisc *misc,
                         int cta, int grid, int lane) {
  uint64_t const policy = evict_first_policy();

  StageIter ld;
  ld.start(cta, grid);
  mbar_wait(smem_u32(&misc->go), 0u);
  uint8_t *arena = a.arena[a.rank];
  uint32_t const epoch = misc->epoch;
  int prev_step = -1; // the GEMM step (layer * kGemms + gemm) of the last issued stage
  for (; ld.valid; stream.advance()) {
    int const step = ld.layer * kGemms + ld.gemm;
    if (prev_step >= 0 && step != prev_step)
      hold_next_gemm(arena, a.rank, epoch, prev_step / kGemms, prev_step % kGemms, cta, grid, lane);
    prev_step = step;
    uint32_t const dst = stream.acquire();
    if (lane == 0) {
      uint32_t const full = stream.full();
      LayerArgs const &layer = a.layer[ld.layer];
      switch (ld.gemm) {
      case kGemmQkv:
        GemmQkv::issue(layer, maps, ld.layer, ld.u, ld.st, dst, full, policy);
        break;
      case kGemmO:
        GemmO::issue(layer, maps, ld.layer, ld.u, ld.st, dst, full, policy);
        break;
      case kGemmGateUp:
        GemmGateUp::issue(layer, maps, ld.layer, ld.u, ld.st, dst, full, policy);
        break;
      default:
        GemmDown::issue(layer, maps, ld.layer, ld.u, ld.st, dst, full, policy);
        break;
      }
    }
    __syncwarp();
    ld.next(cta, grid);
  }
}

} // namespace
} // namespace m31_draft
