# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""The 8-row MLA prologue's overlapped issue order (``MlaPrologue(8, overlap8=True)``),
on CPU.

- Only the 8-row form takes it; its shared-memory layout differs from ``layout(8)`` only
  by the larger partials region (the padded q_b / absorb partial rows), the regions stay
  disjoint and inside the budget.
- ``compile_specs`` builds the 8-row prologues (both trigger forms) with it only on
  request, under a distinct compile key, and never the 16-row ones.
"""

import pytest

from sgl_nvidia_mega.glm5_3.sm103.attention import mla_prologue as mp

SMEM_LIMIT = 227 * 1024


def _numerics():
    from sgl_nvidia_mega.glm5_3.attention import fused

    return fused._prologue_numerics()


def test_overlap8_only_at_8_rows():
    num = _numerics()
    with pytest.raises(ValueError):
        mp.MlaPrologue(16, num, overlap8=True)
    assert mp.MlaPrologue(8, num, overlap8=True).overlap8
    assert not mp.MlaPrologue(8, num).overlap8


def test_overlap8_layout():
    base = mp.layout(8)
    ov = mp.MlaPrologue(8, _numerics(), overlap8=True).off
    assert ov["red"] == base["red"]
    assert ov["ssq"] - ov["red"] == mp.P8_RED_BYTES >= base["ssq"] - base["red"]
    # q_b partials: 8 warps x 8 tokens padded rows; absorb: 2 halves x 8 tokens
    assert mp.P8_RED_BYTES >= 8 * 8 * mp.P8_RED_QB_ROW * 4
    assert mp.P8_RED_BYTES >= 2 * 8 * mp.P8_RED_UK_ROW * 4
    keys = ["wqb", "wuk", "xs", "pub", "qn", "red", "ssq", "mbar", "total"]
    assert [ov[k] for k in keys] == sorted(ov[k] for k in keys)
    for k in ("wqb", "wuk", "xs", "pub", "qn"):
        assert ov[k] == base[k]
    assert ov["mbar"] % 8 == 0 and ov["total"] <= SMEM_LIMIT
    # the padded rows: writer lanes (8 t + g) on distinct banks, reader lanes consecutive
    banks = {((2 * t) * mp.P8_RED_QB_ROW + g) % 32 for t in range(4) for g in range(8)}
    assert len(banks) == 32
    banks = {((2 * t) * mp.P8_RED_UK_ROW + g) % 32 for t in range(4) for g in range(8)}
    assert len(banks) == 32


@pytest.mark.parametrize("overlap", [False, True])
def test_compile_specs_overlap8(overlap):
    from sgl_nvidia_mega.glm5_3.attention import fused
    from sgl_nvidia_mega.glm5_3.sm103.attention import api

    specs = api.compile_specs(
        (8, 16),
        numerics=_numerics(),
        index_q_rope_form=fused.INDEX_Q_ROPE_FORM,
        warps=(16,),
        prologue_end=True,
        prologue8_overlap=overlap,
    )
    for name in ("prologue.8", "prologue_end.8"):
        k, key, _ = specs[name]
        assert k.overlap8 is overlap
        assert key.get("overlap8", False) is overlap
    for name in ("prologue.16", "prologue_end.16"):
        k, key, _ = specs[name]
        assert not k.overlap8 and "overlap8" not in key
    assert fused.PROLOGUE8_OVERLAP_ENV == "SGLANG_NVIDIA_MEGAKERNEL_GLM_PROLOGUE8_OVERLAP"
