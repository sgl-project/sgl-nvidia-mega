# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the SGLang project
"""Public admission and option wiring; no kernel execution or GPU qualification."""

from types import SimpleNamespace as NS
from unittest.mock import Mock

import pytest

from sgl_nvidia_mega.kimi_k3 import compat, providers, spec
from sgl_nvidia_mega.kimi_k3.shape_policy import (
    DECODE_WIDTHS,
    MAX_CONTEXT,
    baseline_capture_widths,
    mla_max_seq_len,
    prepared_widths,
    token_widths,
)

OPTIONS = ("SGLANG_KIMI_K3_MEGAKERNEL_PREFETCH", "SGLANG_KIMI_K3_MEGAKERNEL_EMBEDDING")


@pytest.mark.parametrize("q", (1, 4, 6, 8))
def test_width_and_context_contract(q):
    expected = DECODE_WIDTHS if q == 1 else tuple(range(q, 65, q))
    assert token_widths(q) == prepared_widths(q) == expected
    subset = (1, 17, 64) if q == 1 else (q, 2 * q)
    assert prepared_widths(q, tuple(reversed(subset))) == subset
    captures = baseline_capture_widths(q, (1, 7, 17, 65))
    assert all(0 < rows <= 64 and rows % q == 0 for rows in captures)
    assert set(expected).issuperset(captures) if q > 1 else 17 in captures
    assert mla_max_seq_len(MAX_CONTEXT, q) == MAX_CONTEXT + (q if q > 1 else 0)
    for widths in ((), (0,), (65,), (True,), (q, q)):
        with pytest.raises(ValueError):
            prepared_widths(q, widths)
    for context in (0, MAX_CONTEXT + 1, True):
        with pytest.raises(ValueError):
            mla_max_seq_len(context, q)


def test_config_does_not_admit_other_speculation_or_extend_model_context():
    cfg = NS(
        cuda_graph_config=NS(decode=NS(backend="full")),
        context_length=MAX_CONTEXT,
        speculative_algorithm="DSPARK",
        speculative_num_draft_tokens=6,
        speculative_adaptive=False,
    )
    hf = NS(max_position_embeddings=1048576)
    assert spec._config_disable_reasons(cfg, hf) == []
    for field, value in (
        ("speculative_algorithm", "EAGLE"),
        ("speculative_num_draft_tokens", 3),
        ("speculative_adaptive", True),
        ("context_length", MAX_CONTEXT + 1),
    ):
        assert spec._config_disable_reasons(NS(**(vars(cfg) | {field: value})), hf)
    assert spec._config_disable_reasons(cfg, NS(max_position_embeddings=8192))
    assert hf.max_position_embeddings == 1048576


@pytest.fixture
def registered(monkeypatch):
    for name in OPTIONS:
        monkeypatch.delenv(name, raising=False)
    validate, memory, build = Mock(), Mock(return_value={}), Mock()
    monkeypatch.setattr(compat, "validate_runtime_compatibility", validate)
    monkeypatch.setattr(providers, "_cuda_memory", memory)
    monkeypatch.setattr(providers, "build_resources", build)
    monkeypatch.setattr(spec, "get_model", lambda *_: NS(context_length=MAX_CONTEXT))
    build.side_effect = lambda *a, **k: NS(tokens_per_request=k["tokens_per_request"])
    return validate, memory, build


@pytest.mark.parametrize("q", (1, 6))
def test_registered_options_are_independent_default_off(monkeypatch, registered, q):
    from sglang.srt.kernel_plugins.nvidia import binding

    _, _, build = registered
    monkeypatch.setattr(
        binding, "_gather_tp", Mock(side_effect=AssertionError("unexpected collective"))
    )
    for values in ((None, None), ("0", "0"), ("1", "0"), ("0", "1"), ("1", "1")):
        for name, value in zip(OPTIONS, values):
            if value is None:
                monkeypatch.delenv(name, raising=False)
            else:
                monkeypatch.setenv(name, value)
        built = spec.SPEC.build(
            NS(language_model=object()),
            None,
            spec.BuildContext(tuple(range(1, 65)), q, 30.0, object()),
        )
        kwargs = build.call_args.kwargs
        assert built.widths == kwargs["widths"] == token_widths(q)
        assert kwargs["tokens_per_request"] == q
        for key, value in zip(("enable_prefetch", "enable_embedding"), values):
            assert kwargs.get(key, False) is (value == "1")
        assert "dcp" not in kwargs and "mla_core" not in kwargs


@pytest.mark.parametrize("name", OPTIONS)
def test_invalid_options_fail_before_preparation(monkeypatch, registered, name):
    monkeypatch.setenv(name, "true")
    with pytest.raises(ValueError, match="must be 0 or 1"):
        spec.SPEC.build(
            NS(language_model=object()),
            None,
            spec.BuildContext((6,), 6, 30.0, object()),
        )
    for boundary in registered:
        boundary.assert_not_called()


@pytest.mark.parametrize(
    "q,captures,expected",
    (
        (1, None, (2, 4)),
        (1, (17,), (*range(1, 9), 16, 17, 32, 64)),
        (6, (1, 7, 13, 61), (6, 12, 18)),
    ),
)
def test_capture_widths_agree_before_preparing(
    monkeypatch, registered, q, captures, expected
):
    from sglang.srt.kernel_plugins.nvidia import binding

    _, _, build = registered
    events = []

    def agree(value, group=None):
        events.append("agree")
        return [value] * 8

    def prepare(*args, **kwargs):
        events.append("prepare")
        return NS(tokens_per_request=q)

    monkeypatch.setattr(binding, "_gather_tp", agree)
    build.side_effect = prepare
    context = spec.BuildContext((2, 4), q, 30.0, object())
    built = spec.build_with_capture_widths(
        NS(language_model=object()), None, context, capture_token_rows=captures
    )
    assert events == ["agree", "prepare"]
    assert built.widths == build.call_args.kwargs["widths"] == expected
    build.reset_mock()
    monkeypatch.setattr(
        binding, "_gather_tp", lambda value, group=None: [value] * 7 + [("different",)]
    )
    with pytest.raises(RuntimeError, match="ranks.*differ"):
        spec.build_with_capture_widths(
            NS(language_model=object()), None, context, capture_token_rows=captures
        )
    build.assert_not_called()
