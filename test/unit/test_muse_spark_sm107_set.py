# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Muse Spark's two component sets: the Rubin set lives in muse_spark/sm107, the GB300 set in muse_spark/components, and
the loader registers the Rubin set only on an architecture spec.COMPONENT_ISAS lists for it."""

from __future__ import annotations

from pathlib import Path

import pytest

from sgl_nvidia_mega.muse_spark import components

MUSE_SPARK = Path(components.__file__).resolve().parents[1]


def test_rubin_set_lives_in_sm107():
    assert components.SM107 <= set(components.COMPONENTS)
    for name in components.SM107:
        assert (MUSE_SPARK / "sm107" / f"{name}.py").is_file(), name
        assert not (MUSE_SPARK / "components" / f"{name}.py").exists(), name


def test_gb300_set_stays_in_components():
    for name in set(components.COMPONENTS) - components.SM107:
        assert (MUSE_SPARK / "components" / f"{name}.py").is_file(), name
        assert not (MUSE_SPARK / "sm107" / f"{name}.py").exists(), name


@pytest.mark.parametrize(
    "capability, admitted",
    [((10, 7), True), ((10, 3), False), (None, False)],
)
def test_rubin_set_registers_only_on_its_architecture(
    monkeypatch, capability, admitted
):
    pytest.importorskip("sglang")
    from sgl_nvidia_mega.muse_spark.spec import COMPONENT_ISAS

    assert COMPONENT_ISAS["sm107"] == frozenset({"sm_107"})
    monkeypatch.setattr(components, "_capability", lambda: capability)
    reason = components._sm107_refusal()
    if admitted:
        assert reason is None
    else:
        assert reason is not None and "sm_107" in reason


@pytest.mark.parametrize(
    "backend_source, reachable",
    [
        ("from fa import flash_attn_with_kvcache\n", True),
        (
            "try:\n    from fa import flash_attn_with_kvcache\nexcept ImportError:\n    flash_attn_with_kvcache = None\n",
            True,
        ),
        (
            "class B:\n    def __init__(self):\n        from fa import flash_attn_with_kvcache\n",
            False,
        ),
    ],
)
def test_draft_attn_refuses_a_backend_without_its_module_level_hook_target(
    monkeypatch, tmp_path, backend_source, reachable
):
    """draft_attn hooks the FlashAttention backend module's flash_attn_with_kvcache; a backend that binds the function
    per instance (SGLang after 306ccf6) never calls the hook, so the component refuses instead of reporting engaged."""
    import importlib.util
    import types

    from sgl_nvidia_mega.muse_spark.sm107 import draft_attn

    backend = (
        tmp_path
        / "sglang"
        / "srt"
        / "layers"
        / "attention"
        / "flashattention_backend.py"
    )
    backend.parent.mkdir(parents=True)
    backend.write_text(backend_source)
    spec = types.SimpleNamespace(submodule_search_locations=[str(tmp_path / "sglang")])
    monkeypatch.setattr(importlib.util, "find_spec", lambda name: spec)
    assert draft_attn._target_reachable() is reachable

    records = []
    monkeypatch.setenv("MUSE_VR_DRAFT_ATTN", "1")
    monkeypatch.setattr(
        draft_attn.bed, "report", lambda name, ok, **kw: records.append((name, ok, kw))
    )
    if not reachable:
        assert draft_attn.register() is False
        assert records == [
            (
                "draft_attn",
                False,
                {"reason": f"no module-level hook target {draft_attn.TARGET}"},
            )
        ]
