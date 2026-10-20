# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""The plugin entry point on CPU: it registers each spec with SGLang's megakernels
registry and no hook on SGLang's code, except the parity test's own."""

import pytest
from sglang.srt.megakernels import spec as registry

from sgl_nvidia_mega import plugin


@pytest.fixture
def hooks(monkeypatch):
    from sglang.srt.plugins.hook_registry import HookRegistry

    monkeypatch.setattr(registry, "_REGISTRY", {})
    seen = []
    monkeypatch.setattr(
        HookRegistry, "register", lambda target, hook, kind: seen.append(target)
    )
    return seen


def test_register_registers_every_spec_and_no_hook(hooks, monkeypatch):
    monkeypatch.delenv(plugin.PARITY_SORT_ENV, raising=False)
    plugin.register()
    assert registry._REGISTRY == dict(plugin.SPECS)
    assert hooks == []
    # the registered path resolves to the spec SGLang checks
    spec, why = registry.find_spec(architectures=[plugin.SPECS[0][0]], isa="sm_103")
    assert spec is not None, why


def test_the_entry_point_is_the_one_sglang_checks():
    import tomllib
    from pathlib import Path

    project = tomllib.loads((Path(__file__).parents[2] / "pyproject.toml").read_text())
    entry_points = project["project"]["entry-points"]["sglang.srt.plugins"]
    assert entry_points == {registry.ENTRY_POINT: "sgl_nvidia_mega.plugin:register"}
