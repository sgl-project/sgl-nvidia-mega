# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Ordinary decode chain with a tensor/numeric sparse-indexer step."""

from sgl_nvidia_mega.kimi_k31.k3.executor import KimiK3MegakernelExecutor


class SparseDecodeExecutor(KimiK3MegakernelExecutor):
    def before_attention(self, layer_id, hidden, view):
        attention = self.resources.attention[layer_id]
        if attention.kind == "mla":
            attention.core.set_step(view.sparse, view.index)
            attention.core.prepare_gate(hidden)
