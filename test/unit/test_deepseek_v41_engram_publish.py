# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Slot engram_publish on CPU: opt-in, needs engram, reported built only with the
prefetch's exchange; the later Engram layers' WKV columns start LATE_LEAD layers
before them (the first layer's right after the reduce); and before_layer hands every
layer to the exchange's schedule, the all-reduce gather only without it."""

from types import SimpleNamespace

from sgl_nvidia_mega.deepseek_v41 import engram, runtime
from sgl_nvidia_mega.deepseek_v41.engram import prefetch as ep


def test_the_slot_is_opt_in_and_needs_engram(monkeypatch):
    assert "engram_publish" in runtime.OPT_IN_SLOTS
    assert "engram_publish" not in runtime.SHIPPED_SLOTS
    monkeypatch.setattr(runtime, "SLOTS", frozenset({"engram_publish"}))
    assert any("needs slot 'engram'" in r for r in runtime.slot_reasons())
    monkeypatch.setattr(runtime, "SLOTS", frozenset({"engram", "engram_publish"}))
    assert runtime.slot_reasons() == []


def _built(prefetch):
    bound = {
        "engram": engram.Bound(prefetch, ""),
        "attention": {},
        "embedding": None,
        "moe": SimpleNamespace(),
    }
    facts = SimpleNamespace(routed=False)
    return runtime.built_slots({}, bound, facts)


def test_built_only_with_the_exchange():
    assert _built(None)["engram_publish"] is False
    assert _built(SimpleNamespace(exchange=None))["engram_publish"] is False
    built = _built(SimpleNamespace(exchange=object()))
    assert built["engram_publish"] is True and built["engram"] is True


def test_the_later_layers_start_their_columns_late():
    # DeepSeek-V4.1-Flash: Engram layers 1 and 14
    assert ep.project_points([14, 1], 1) == {1: None, 14: 14 - ep.LATE_LEAD}
    assert ep.LATE_LEAD == 4
    # a layer within LATE_LEAD of the first starts right after the reduce
    assert ep.project_points([2, 5, 6], 2) == {2: None, 5: None, 6: None}
    assert ep.project_points([2, 7], 2) == {2: None, 7: 3}


class _Prefetch:
    def __init__(self, exchange):
        self.exchange = exchange
        self.gather_at = 1
        self.calls = []

    def at_layer(self, i):
        self.calls.append(("at", i))

    def gather(self):
        self.calls.append(("gather",))


def _st(prefetch):
    return SimpleNamespace(
        slots=SimpleNamespace(engram_prefetch=True),
        hash_ids=object(),
        bound={"engram": engram.Bound(prefetch, "")},
    )


def test_before_layer_dispatch():
    with_exchange = _Prefetch(object())
    st = _st(with_exchange)
    for i in range(0, 16):
        engram.before_layer(i, st)
    assert with_exchange.calls == [("at", i) for i in range(16)]
    without = _Prefetch(None)
    st = _st(without)
    for i in range(0, 16):
        engram.before_layer(i, st)
    assert without.calls == [("gather",)]
