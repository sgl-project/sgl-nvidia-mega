// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
//
// Router top-k + route record + the routed input's MXFP8 quant in one cluster launch
// (gate_route.cuh, kQuant; slot gate_route_quant) for the DeepSeek-V4.1 target (384
// experts, top-6); grid = cluster = rows (1-8), block = 384.

#include "gate_route.cuh"

extern "C" __global__ __launch_bounds__(384) void dsv41_gate_route_quant(
    const float* logits, const float* bias, const float* bias_alt, const void* input_ids, int ids64,
    long long alt_token_id, const int* num_valid, float* __restrict__ weights, int* __restrict__ ids,
    float scale, float eps, int* __restrict__ permuted, int* __restrict__ record, int experts,
    int expert_offset, const unsigned short* x, unsigned char* __restrict__ x_q, unsigned char* __restrict__ x_s) {
  GateRoute<384, 6, true>(logits, bias, bias_alt, input_ids, ids64, alt_token_id, num_valid, weights, ids, scale,
                           eps, permuted, record, experts, expert_offset, x, x_q, x_s);
}
