# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""K3.1 MLA and index-K Parameter views over one packed allocation."""

import torch

from sgl_nvidia_mega.kimi_k31.k3.projection_storage import (
    _coalesce_projection_storage,
    _Layout,
    _Part,
    local_mla_heads,
)

PITCH = 3776  # TP8; pitch(heads) in general


def pitch(heads: int = 12) -> int:
    """Packed rows [q_a | kv_a | gate | index-K]: TP8 (12 heads) 3776, TP16 (6 heads) 3008."""
    return 2112 + heads * 128 + 128


def layout(heads: int = 12) -> _Layout:
    return _Layout(
        name="K3.1 MLA/index-K",
        parts=(
            _Part("fused_qkv_a_proj_with_mqa.weight", 2112),
            _Part("g_proj.weight", heads * 128),
            _Part("indexer.wk.weight", 128),
        ),
        columns=7168,
        dtype=torch.bfloat16,
    )


_LAYOUT = layout(12)


def coalesce_index_projection(attention):
    """Preserve Parameter identities; stock decode uses the repointed Wk too."""
    root = _coalesce_projection_storage(attention, layout(local_mla_heads(attention)), device_type="cuda")
    # The stock [Wk | head-gate] alias predates plugin preparation. Its Wk rows
    # now live here; disable that optional alias so stock paths read live weights.
    attention.indexer._wk_gate_weight = None
    return root
