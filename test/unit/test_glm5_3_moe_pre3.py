# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""g1 / g2 ``pre`` = 3 form (CPU): the knob parsing, the forms compile_specs builds (stock and
die-local), and their compile-cache keys (the stock and die-local keys are unchanged with the
knob off; the die-local forms ignore it)."""

import pytest

from sgl_nvidia_mega.glm5_3.sm103.moe import api
from sgl_nvidia_mega.glm5_3.sm103.moe.engine import Nvfp4MoeSmallM


def test_knob_default(monkeypatch):
    monkeypatch.delenv(api.MOE_PRE3_ENV, raising=False)
    assert api.g12_knobs() == {"pre": 1} and not api.g12_pre3_requested()
    for raw in ("0", " ", ""):
        monkeypatch.setenv(api.MOE_PRE3_ENV, raw)
        assert api.g12_knobs() == {"pre": 1}


def test_knob_on(monkeypatch):
    monkeypatch.setenv(api.MOE_PRE3_ENV, "1")
    assert api.g12_pre3_requested()
    assert api.g12_knobs() == {"pre": 3}
    assert api.g12_knobs(localize=True) == {"pre": 1}


@pytest.mark.parametrize("raw", ["2", "3", "yes", "-1"])
def test_knob_rejects(monkeypatch, raw):
    monkeypatch.setenv(api.MOE_PRE3_ENV, raw)
    with pytest.raises(ValueError):
        api.g12_knobs()


def test_engine_forms():
    for mode in ("g1", "g2"):
        assert Nvfp4MoeSmallM(mode).knobs() == {}
        assert Nvfp4MoeSmallM(mode, pre=1).knobs() == {}
        assert Nvfp4MoeSmallM(mode, pre=3).knobs() == {"pre": 3}
    with pytest.raises(ValueError):
        Nvfp4MoeSmallM("g2", merge=False, pre=3)
    with pytest.raises(ValueError):
        Nvfp4MoeSmallM("g1", pre=2)


@pytest.mark.parametrize("on", [False, True])
def test_compile_specs(monkeypatch, on):
    pytest.importorskip("cutlass")
    if on:
        monkeypatch.setenv(api.MOE_PRE3_ENV, "1")
    else:
        monkeypatch.delenv(api.MOE_PRE3_ENV, raising=False)
    specs = api.compile_specs(shared=True)
    want = 3 if on else 1
    for mode in ("g1", "g2"):
        kernel = specs[mode][0]
        assert kernel.mode == mode and kernel.pre == want
        assert kernel.knobs() == ({"pre": 3} if want == 3 else {})
