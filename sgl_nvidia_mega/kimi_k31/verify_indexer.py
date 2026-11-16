# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""R-row K31 index-K writer with weights-only preparation."""

from dataclasses import dataclass
import torch
import torch.nn.functional as F

# SGLang names this module reads (test/unit/test_contract.py pins them)
SGLANG_READS = {"kernels/ops/kvcache/dsa_fused_store_index_cache.py": ("fused_store_index_k_cache",)}


@dataclass(frozen=True)
class IndexKeyWeights:
    projection: torch.Tensor
    norm: torch.Tensor
    bias: torch.Tensor
    eps: float


class PreparedVerifyIndexKey:
    def __init__(self, weights: IndexKeyWeights):
        if weights.projection.dtype != torch.bfloat16 or weights.projection.shape != (
            128,
            7168,
        ):
            raise ValueError("K31 verify index-K needs BF16 [128,7168] weights")
        if weights.norm.dtype != torch.float32 or weights.bias.dtype != torch.float32:
            raise ValueError("K31 verify index-K needs stock FP32 LayerNorm parameters")
        self.weights = weights
        self.cache = None

    def bind(self, cache, page_size, rows=(8,)):
        from .dsa.idx_fp4_knob import ENABLED as IDX_FP4

        if IDX_FP4:  # K31MK_DSA_IDX_FP4: the host's MXFP4 pool, uint8 [pages, 64 * 68]
            from .dsa import idx_fp4

            if page_size != 64:
                raise ValueError("K31 verify index-K pool needs 64-token pages")
            idx_fp4.check_cache(cache)
        elif (
            page_size != 64
            or cache.dtype != torch.uint8
            or cache.ndim != 2
            or cache.shape[1] != 64 * 132
        ):
            raise ValueError("K31 verify index-K pool must be uint8 [pages,64*132]")
        self.cache = cache
        for r in rows:
            self.run(
                torch.zeros(r, 7168, dtype=torch.bfloat16, device=cache.device),
                torch.zeros(r, dtype=torch.int64, device=cache.device),
            )

    def run(self, hidden, slots):
        from flashinfer.norm import layernorm
        from sglang.kernels.ops.kvcache.dsa_fused_store_index_cache import (
            fused_store_index_k_cache,
        )

        w = self.weights
        projected = F.linear(hidden, w.projection)
        key = layernorm(projected, w.norm, w.bias, w.eps)
        from .dsa.idx_fp4_knob import ENABLED as IDX_FP4

        if IDX_FP4:
            from .dsa import idx_fp4

            idx_fp4.store_index_k(key, self.cache, slots)
        else:
            fused_store_index_k_cache(key, self.cache, slots, 64)
