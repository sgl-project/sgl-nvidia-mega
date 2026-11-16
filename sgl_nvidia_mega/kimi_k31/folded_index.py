# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Packed MLA/index-K projection and a query-kernel index-cache side output."""

from __future__ import annotations

import cutlass
import cutlass.cute as cute
import torch
from cutlass import BFloat16, Float32, Uint8, Uint32

from sgl_nvidia_mega.kimi_k31.k3.cutedsl.common.primitives import (
    float_bits,
    fp8_bits,
    warp_max_abs,
    warp_sum,
)
from sgl_nvidia_mega.kimi_k31.k3.cutedsl.mla.prepared import PreparedMla
from sgl_nvidia_mega.kimi_k31.k3.cutedsl.mla.query import MlaQuery

from .projection_storage import pitch as packed_pitch


class IndexMlaQuery(MlaQuery):
    @cute.jit
    def finish(self, args, tid, part, head, tok0):
        cache, gamma, beta, epsilon = args[11:15]
        lane, warp = tid % 32, tid // 32
        relative = part * 8 + warp
        row = tok0 + relative
        # Head zero's cluster partitions the tile into independent warp rows.
        # This covers both the small FMAs and the 16/32/64-row tiled kernels.
        # Packed rows [q_a | kv_a | gate | index-K]: index-K at 2112 + 128 H (TP8 3648, TP16 2880).
        pitch: cutlass.Constexpr = packed_pitch(self.heads)
        index_col: cutlass.Constexpr = pitch - 128
        if (head == 0) & (relative < self.tokens) & (row < self.batch):
            slot = args[6][row]
            if slot >= 0:
                projected = cute.make_tensor(
                    args[0].iterator,
                    cute.make_layout((self.batch, pitch), stride=(pitch, 1)),
                )
                values = cute.make_rmem_tensor(4, Float32)
                total = Float32(0)
                for i in cutlass.range_constexpr(4):
                    values[i] = Float32(projected[row, index_col + lane * 4 + i])
                    total += values[i]
                mean = warp_sum(total) * Float32(1.0 / 128)
                variance = Float32(0)
                for i in cutlass.range_constexpr(4):
                    values[i] -= mean
                    variance += values[i] * values[i]
                variance = warp_sum(variance) * Float32(1.0 / 128)
                inv_std = cute.math.rsqrt(variance + epsilon[0], fastmath=True)
                maximum = Float32(0)
                for i in cutlass.range_constexpr(4):
                    col = lane * 4 + i
                    value = values[i] * inv_std * gamma[col] + beta[col]
                    # Stock LayerNorm returns BF16 before the FP8 quantizer.
                    values[i] = Float32(BFloat16(value))
                    maximum = max(maximum, abs(values[i]))
                maximum = warp_max_abs(maximum)
                scale = max(maximum, Float32(1e-4)) / Float32(448)
                inverse = Float32(1) / scale
                page, offset = slot // 64, slot % 64
                for i in cutlass.range_constexpr(4):
                    cache[page, offset * 128 + lane * 4 + i] = fp8_bits(
                        values[i] * inverse
                    )
                if lane == 0:
                    bits = Uint32(float_bits(scale))
                    for i in cutlass.range_constexpr(4):
                        cache[page, 8192 + offset * 4 + i] = Uint8(bits >> (8 * i))


class PreparedIndexMla(PreparedMla):
    def __init__(self, attention_and_indexer, *args, **kwargs):
        attention, indexer = attention_and_indexer
        super().__init__(attention, *args, **kwargs)
        if not indexer.skip_rope or indexer.apply_hadamard:
            raise ValueError("folded K3.1 index-K requires NoPE without Hadamard")
        norm = indexer.k_norm
        if any(
            w.shape != (128,) or w.dtype != torch.float32
            for w in (norm.weight, norm.bias)
        ):
            raise ValueError("folded K3.1 index-K requires FP32 affine LayerNorm128")
        self.index_norm = (norm.weight, norm.bias)
        self.index_epsilon = torch.tensor(
            [norm.variance_epsilon], dtype=torch.float32, device=self.device
        )
        self.index_cache = None
        self.query_type = IndexMlaQuery

    def bind_index_cache(self, cache, page_size):
        from .dsa.idx_fp4_knob import refuse

        refuse("the folded index-K MLA query (folded_index.PreparedIndexMla)")
        if (
            page_size != 64
            or cache.dtype != torch.uint8
            or cache.ndim != 2
            or cache.shape[1] != 8448
            or not cache.is_contiguous()
        ):
            raise ValueError("folded K3.1 requires uint8 index-K pages of 8448 bytes")
        if (
            self.index_cache is not None
            and self.index_cache.data_ptr() != cache.data_ptr()
        ):
            raise RuntimeError("K3.1 index-K cache changed after preparation")
        self.index_cache = cache
        self.query_buffers = (cache, *self.index_norm, self.index_epsilon)

    def prepare_cache(self, cache):
        if self.index_cache is None:
            raise RuntimeError("bind the SGLang index-K cache before preparing MLA")
        return super().prepare_cache(cache)
