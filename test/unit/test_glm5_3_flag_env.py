# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""GLM-5.3 / GLM-5.2: ``--enable-nvidia-megakernel`` turns on the embedding replication
slot ``draft`` needs, on CPU.

The spec pins its environment per TP size (``ENV`` at TP8 / TP4, ``glm5_3/sm107``'s at
TP16), so its ``Requirements.env`` is empty and SGLang's own ``FLAG_SETS_ENV`` sets
nothing for it. ``config_disable_reasons`` sets ``SGLANG_ENABLE_EMBED_REPLICATION`` at
TP8 / TP4 when it is unset, keeps and checks a value the user set, leaves the
environment as it was when the setup is refused, and leaves TP16 (whose draft reads the
vocab-sharded embedding) alone.
"""

import logging
import os
from types import SimpleNamespace

import pytest

from sgl_nvidia_mega.glm5_3 import draft, runtime, sm107, spec

NAME = "SGLANG_ENABLE_EMBED_REPLICATION"


@pytest.fixture(autouse=True)
def _unset(monkeypatch):
    # setenv first so that monkeypatch records the variable even when it is absent:
    # teardown then restores its original state, including the value the spec writes
    # straight to os.environ during a test, so it does not reach later test files.
    monkeypatch.setenv(NAME, "")
    monkeypatch.delenv(NAME)


@pytest.fixture
def accepted(monkeypatch):
    """Every check of a TP8 / TP4 setup passes except the pinned environment, which
    pins only the flag's variable."""
    for name in (
        "toolchain_reasons",
        "binding_reasons",
        "draft_capture_reasons",
        "deadline_reasons",
        "profile_reasons",
        "pd_reasons",
    ):
        monkeypatch.setattr(spec, name, lambda *a, **k: [])
    monkeypatch.setattr(runtime, "slot_reasons", lambda: [])
    monkeypatch.setattr(spec, "SERVER_ARGS", {})
    monkeypatch.setattr(spec, "ENV", {NAME: True})


def _cfg(tp):
    graphs = SimpleNamespace(decode=SimpleNamespace(backend="full"))
    return SimpleNamespace(tp_size=tp, cuda_graph_config=graphs)


def test_the_draft_slot_pins_it_per_tp_size():
    if "draft" not in spec.SLOTS:
        pytest.skip("slot draft is off in this environment")
    assert spec.ENV[NAME] is True
    assert NAME in spec.FLAG_SETS_ENV
    # not in Requirements.env: SGLang would then set and require it at TP16 too
    assert NAME not in spec.REQUIREMENTS.env
    assert NAME not in draft.SPEC.requires.env
    assert NAME not in sm107.ENV


@pytest.mark.parametrize("tp", [4, 8])
def test_the_flag_sets_it_at_tp8_and_tp4(accepted, caplog, tp):
    caplog.set_level(logging.INFO, logger=spec.__name__)
    assert spec._config_disable_reasons(_cfg(tp), None) == []
    assert os.environ[NAME] == "True"
    assert f"set {NAME}=True for {spec.SPEC.name}" in caplog.text


@pytest.mark.parametrize("value, ok", [("1", True), ("0", False)])
def test_a_value_the_user_set_is_kept_and_checked(accepted, monkeypatch, value, ok):
    monkeypatch.setenv(NAME, value)
    reasons = spec._config_disable_reasons(_cfg(4), None)
    assert reasons == ([] if ok else [f"{NAME} False is not True"])
    assert os.environ[NAME] == value


def test_a_refused_setup_leaves_it_unset(accepted, monkeypatch):
    monkeypatch.setattr(spec, "pd_reasons", lambda cfg: ["refused"])
    assert spec._config_disable_reasons(_cfg(4), None) == ["refused"]
    assert NAME not in os.environ


def test_tp16_leaves_it_alone(monkeypatch):
    monkeypatch.setattr(sm107, "config_reasons", lambda cfg, hf_config: [])
    assert spec._config_disable_reasons(_cfg(spec.SM107_TP_SIZE), None) == []
    assert NAME not in os.environ


def test_only_the_flags_variables_are_set():
    assert spec.set_flag_env({"SGLANG_DSA_FUSE_TOPK": True}) == []
    assert "SGLANG_DSA_FUSE_TOPK" not in os.environ
