# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""The indexer-layer prologue split (``SGLANG_NVIDIA_MEGAKERNEL_GLM_IDX_SPLIT``), on CPU.

- ``compile_specs`` builds ``front.<cap>`` (``IndexerFront``) only on request, keyed by
  the prologue numerics, and leaves every other unit's kernel options and compile key
  as they are without it.
- ``IndexerFront`` traces the prologue's own index-K method (one ``MlaPrologue`` of the
  same numerics) and refuses a q_a sum order it does not implement.
- The stock steps never split.
"""

import dataclasses

import pytest


def _numerics(**kw):
    from sgl_nvidia_mega.glm5_3.attention import fused

    return dataclasses.replace(fused._prologue_numerics(), **kw)


def _specs(**kw):
    from sgl_nvidia_mega.glm5_3.attention import fused
    from sgl_nvidia_mega.glm5_3.sm103.attention import api

    return api.compile_specs(
        (8, 16),
        numerics=_numerics(),
        index_q_rope_form=fused.INDEX_Q_ROPE_FORM,
        warps=(8, 16),
        prologue_end=True,
        **kw,
    )


@pytest.mark.parametrize(
    "knobs",
    [
        {},
        dict(
            prologue16_overlap=True,
            prologue16_kv_split=True,
            prologue8_overlap=True,
            prologue_push=True,
            prologue_f32x2=True,
        ),
    ],
)
def test_front_specs_only_on_request(knobs):
    off = _specs(**knobs)
    on = _specs(idx_front=True, **knobs)
    assert not any(k.startswith("front.") for k in off)
    assert sorted(k for k in on if k.startswith("front.")) == ["front.16", "front.8"]
    assert set(on) - set(off) == {"front.8", "front.16"}
    for name, (kern, key, args) in off.items():
        kern2, key2, args2 = on[name]
        assert type(kern2) is type(kern) and key2 == key and len(args2) == len(args)
        assert vars(kern2) == vars(kern)
    for cap in (8, 16):
        kern, key, args = on[f"front.{cap}"]
        assert kern.max_rows == cap and key["max_rows"] == cap
        assert key["norm_order"] == "strided_8x256"
        assert len(args) == 21  # 20 values + the stream


def test_front_traces_the_prologue_numerics():
    from sgl_nvidia_mega.glm5_3.sm103.attention.indexer_front import IndexerFront
    from sgl_nvidia_mega.glm5_3.sm103.attention.mla_prologue import MlaPrologue

    num = _numerics()
    f = IndexerFront(16, num, index_rope_cache_fp32=False)
    assert isinstance(f.pro, MlaPrologue)
    assert f.pro.num == num and f.pro.max_rows == 16
    assert f.pro.index_rope_cache_fp32 is False
    with pytest.raises(ValueError):
        IndexerFront(16, _numerics(norm_order="lane_contiguous"))
    with pytest.raises(ValueError):
        IndexerFront(12, num)


def test_stock_steps_never_split():
    from sgl_nvidia_mega.glm5_3.attention.fused import IDX_SPLIT_ENV
    from sgl_nvidia_mega.glm5_3.attention.step import AttentionStep

    assert AttentionStep().indexer_split(None, None, True) is None
    assert IDX_SPLIT_ENV == "SGLANG_NVIDIA_MEGAKERNEL_GLM_IDX_SPLIT"
