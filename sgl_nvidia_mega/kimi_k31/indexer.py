# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Maintain K3.1's index-K cache without computing an unused all-token ranking."""

from __future__ import annotations

from typing import Any

import torch

from sgl_nvidia_mega.kimi_k31.k3.executor import KimiK3MegakernelExecutor

SGLANG_READS = {
    "kernels/ops/kvcache/dsa_fused_store_index_cache.py": (
        "fused_store_index_k_cache",
    ),
}


class PreparedIndexKey:
    def __init__(self, indexer: Any):
        if not indexer.skip_rope or indexer.apply_hadamard:
            raise ValueError("Kimi K3.1 index-K writer requires NoPE without Hadamard")
        self.projection = indexer.wk
        self.norm = indexer.k_norm
        self.cache = None

    def bind(
        self, cache: torch.Tensor, page_size: int, rows: tuple[int, ...] = (1,)
    ) -> None:
        from .dsa.idx_fp4_knob import refuse

        refuse("the non-verify decode index-K writer (indexer.PreparedIndexKey)")
        if page_size != 64 or cache.dtype != torch.uint8:
            raise ValueError("Kimi K3.1 needs uint8 index-K cache with 64-token pages")
        if cache.ndim != 2 or cache.shape[1] != 64 * 132 or not cache.is_contiguous():
            raise ValueError(
                f"unsupported Kimi K3.1 index-K cache layout: {cache.shape}"
            )
        if self.cache is not None and self.cache.data_ptr() != cache.data_ptr():
            raise RuntimeError("Kimi K3.1 index-K cache changed after preparation")
        self.cache = cache
        # Compile the host-provided fused store before CUDA graph capture. The reserved
        # padding slot zero is safe to initialize before any live request.
        for width in rows:
            self.run(
                cache.new_zeros((width, 7168), dtype=torch.bfloat16),
                cache.new_zeros((width,), dtype=torch.int64),
            )

    def run(self, hidden: torch.Tensor, slots: torch.Tensor) -> None:
        from sglang.kernels.ops.kvcache.dsa_fused_store_index_cache import (
            fused_store_index_k_cache,
        )

        if self.cache is None:
            raise RuntimeError("Kimi K3.1 index-K cache was not prepared")
        key = self.norm(self.projection(hidden)[0])
        fused_store_index_k_cache(key, self.cache, slots, 64)


class KimiK31MegakernelExecutor(KimiK3MegakernelExecutor):
    def __init__(self, resources: Any, indexers: dict[int, PreparedIndexKey]):
        super().__init__(resources)
        self.indexers = indexers

    def before_attention(self, layer_id: int, hidden: torch.Tensor, view: Any) -> None:
        if layer_id in self.indexers:
            self.indexers[layer_id].run(hidden, view.mla.write_slots)
