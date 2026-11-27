# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Draft sparse MLA (CPU): opt-in by SGLANG_NVIDIA_MEGAKERNEL_GLM_DRAFT_SPARSE_MLA=1 (default
off, no slot), and the draft attention group's dispatch: the family's sparse MLA and
value up-projection with the knob on and the top-k ids in the kernel's layout (int32
[rows, top-k], contiguous), SGLang's calls otherwise."""

import pytest
import torch

from sgl_nvidia_mega.glm5_3 import draft_attention as da
from sgl_nvidia_mega.glm5_3.attention.fused import TOPK, FusedAttention
from sgl_nvidia_mega.glm5_3.attention.step import AttentionStep
from sgl_nvidia_mega.glm5_3.runtime import SHIPPED_SLOTS


def test_opt_in(monkeypatch):
    assert "draft_sparse_mla" not in SHIPPED_SLOTS
    monkeypatch.delenv(da.DRAFT_SPARSE_MLA_ENV, raising=False)
    assert not da.draft_sparse_mla_on()
    monkeypatch.setenv(da.DRAFT_SPARSE_MLA_ENV, "1")
    assert da.draft_sparse_mla_on()


def _group(fused: bool):
    group = object.__new__(da.DraftFusedAttention)
    group.fused_sparse = fused
    return group


@pytest.fixture
def calls(monkeypatch):
    seen = []
    for cls, tag in ((FusedAttention, "fused"), (AttentionStep, "stock")):
        monkeypatch.setattr(
            cls, "sparse_mla", lambda self, *a, _t=tag: seen.append(("mla", _t)) or _t
        )
        monkeypatch.setattr(
            cls, "uv_proj", lambda self, *a, _t=tag: seen.append(("uv", _t)) or _t
        )
    return seen


def test_sparse_mla_dispatch(calls):
    q = torch.empty(2, 8, 576)
    ids = torch.zeros(2, TOPK, dtype=torch.int32)
    assert _group(True).sparse_mla(None, q, ids, None) == "fused"
    assert _group(False).sparse_mla(None, q, ids, None) == "stock"
    # fewer id rows than query rows: the family pads them (pad_topk_indices)
    assert _group(True).sparse_mla(None, q, ids[:1], None) == "fused"
    # not the kernel's layout: SGLang's call
    assert _group(True).sparse_mla(None, q, ids.long(), None) == "stock"
    assert _group(True).sparse_mla(None, q, ids[:, : TOPK // 2], None) == "stock"
    assert _group(True).sparse_mla(None, q, ids.t().contiguous().t(), None) == "stock"
    assert _group(True).sparse_mla(None, q[:1], ids, None) == "stock"
    assert _group(True).sparse_mla(None, q, None, None) == "stock"


def test_uv_proj_dispatch(calls):
    out = torch.empty(2, 8, 512)
    assert _group(True).uv_proj(None, out) == "fused"
    assert _group(False).uv_proj(None, out) == "stock"
    assert calls == [("uv", "fused"), ("uv", "stock")]
