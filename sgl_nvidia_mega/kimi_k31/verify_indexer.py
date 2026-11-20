# SPDX-License-Identifier: Apache-2.0
"""R-row K31 index-K writer with weights-only preparation."""

from dataclasses import dataclass

import torch
import torch.nn.functional as F

# Names read from SGLang. A contract test checks these before GPU use.
SGLANG_READS = {
    "kernels/ops/kvcache/dsa_fused_store_index_cache.py": (
        "fused_store_index_k_cache",
    ),
}


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
        if (
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

        from .dsa.verify import dcp_store_args

        w = self.weights
        projected = F.linear(hidden, w.projection)
        key = layernorm(projected, w.norm, w.bias, w.eps)
        fused_store_index_k_cache(key, self.cache, slots, 64, *dcp_store_args())
