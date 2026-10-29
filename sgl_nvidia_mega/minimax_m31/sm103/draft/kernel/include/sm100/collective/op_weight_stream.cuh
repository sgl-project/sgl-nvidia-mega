#pragma once

// The weight stream's generic part: the order of weight stages one CTA consumes in a call.
// Each GEMM's unit count, stage count and stage issue come from its OP (see
// op_weight_producer.cuh for the dispatch).

#include "include/sm100/common/sync.cuh"

namespace m31_draft {
namespace {

// Per-GEMM schedule, defined in op_weight_producer.cuh from the four GEMM OPs.
__host__ __device__ constexpr int gemm_units(int g);
__device__ int gemm_stages(int g, int u);

// The sequence of weight stages one CTA consumes in a call: layer, GEMM, unit, stage.
// GEMM step s = layer * kGemms + gemm; a call walks steps [0, kSteps).
constexpr int kSteps = kLayers * kGemms;
struct StageIter {
  int layer, gemm, u, st, units;
  int nst; // stages of the current unit
  bool valid;
  __device__ void seek(int cta, int grid) {
    while (layer * kGemms + gemm < kSteps) {
      units = gemm_units(gemm);
      if (first_gemm_unit(cta, grid) < units) {
        u = first_gemm_unit(cta, grid);
        st = 0;
        nst = gemm_stages(gemm, u);
        valid = true;
        return;
      }
      if (++gemm == kGemms) {
        gemm = 0;
        ++layer;
      }
    }
    valid = false;
  }
  __device__ void start(int cta, int grid) {
    layer = 0;
    gemm = 0;
    seek(cta, grid);
  }
  __device__ void next(int cta, int grid) {
    if (++st < nst)
      return;
    st = 0;
    u += grid;
    if (u < units) {
      nst = gemm_stages(gemm, u);
      return;
    }
    if (++gemm == kGemms) {
      gemm = 0;
      ++layer;
    }
    seek(cta, grid);
  }
};

} // namespace
} // namespace m31_draft
