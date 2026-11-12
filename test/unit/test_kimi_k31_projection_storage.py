# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""The folded projection preserves stock Parameter views and invalidates its old alias."""

from types import SimpleNamespace
from unittest.mock import patch

import torch

from sgl_nvidia_mega.kimi_k31.k3 import projection_storage as base
from sgl_nvidia_mega.kimi_k31 import projection_storage as folded


def test_folded_projection_keeps_live_stock_weights_and_is_idempotent():
    def linear(rows):
        weight = torch.nn.Parameter(torch.randn(rows, 8), requires_grad=False)
        weight.loader_marker = object()
        return SimpleNamespace(weight=weight)

    attn = SimpleNamespace(
        fused_qkv_a_proj_with_mqa=linear(16),
        g_proj=linear(8),
        indexer=SimpleNamespace(wk=linear(128), _wk_gate_weight=torch.empty(160, 8)),
    )
    layout = base._Layout(
        name="test K31",
        columns=8,
        dtype=torch.float32,
        parts=(
            base._Part("fused_qkv_a_proj_with_mqa.weight", 16),
            base._Part("g_proj.weight", 8),
            base._Part("indexer.wk.weight", 128),
        ),
    )
    params = [
        attn.fused_qkv_a_proj_with_mqa.weight,
        attn.g_proj.weight,
        attn.indexer.wk.weight,
    ]
    old = torch.cat([p.detach() for p in params])

    def cpu_coalesce(owner, spec, **kwargs):
        return base._coalesce_projection_storage(owner, spec, device_type="cpu")

    with (
        patch.object(folded, "_LAYOUT", layout),
        patch.object(folded, "_coalesce_projection_storage", side_effect=cpu_coalesce),
    ):
        packed = folded.coalesce_index_projection(attn)
        assert torch.equal(packed, old)
        assert attn.indexer._wk_gate_weight is None
        assert attn.indexer.wk.weight is params[2]
        assert attn.indexer.wk.weight.data_ptr() == packed[24:].data_ptr()
        assert folded.coalesce_index_projection(attn) is packed
        packed[24, 0] = 7
        assert attn.indexer.wk.weight[0, 0] == 7
        assert all(hasattr(p, "loader_marker") for p in params)
