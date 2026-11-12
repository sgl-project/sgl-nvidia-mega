# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Slot gate_route_quant: opt-in, needs gate_route; the MoE slot takes the quant only on
the fast top-k without a pre-quantized input, and then forks no routed-quant stream.
The launch itself is tested bitwise in test/kernels/test_deepseek_v41_moe.py (one GPU)
and test/kernels/test_deepseek_v41_moe_ep.py (world 8)."""

from types import SimpleNamespace

import pytest
import torch

from sgl_nvidia_mega.deepseek_v41 import runtime
from sgl_nvidia_mega.deepseek_v41.moe import routed


def test_the_slots_run_by_default():
    for slot in ("gate_route", "gate_route_quant", "gemm2_pipe"):
        assert slot in runtime.DEFAULT_ON_SLOTS and slot in runtime.SHIPPED_SLOTS
        assert slot not in runtime.OPT_IN_SLOTS


def test_the_slot_needs_gate_route(monkeypatch):
    monkeypatch.setattr(
        runtime, "SLOTS", frozenset({"moe", "gate", "gate_route_quant"})
    )
    assert any(
        "'gate_route_quant' needs slot 'gate_route'" in r
        for r in runtime.slot_reasons()
    )
    monkeypatch.setattr(
        runtime, "SLOTS", frozenset({"moe", "gate", "gate_route", "gate_route_quant"})
    )
    assert not any("gate_route_quant" in r for r in runtime.slot_reasons())


def test_built_only_with_the_quant_kernel():
    def bound(kernel, router=None):
        gemm = SimpleNamespace(
            gate_route_kernel=object(), gemm2_pipe=None, gate_route_quant_kernel=kernel
        )
        slot = SimpleNamespace(gemm=gemm, shared=None, gate=True, router=router)
        return {
            "moe": SimpleNamespace(slots={3: slot}),
            "attention": {},
            "embedding": None,
            "engram": SimpleNamespace(prefetch=None),
            "logits": None,
        }

    facts = SimpleNamespace(routed=True)
    assert runtime.built_slots({}, bound(object()), facts)["gate_route_quant"]
    assert not runtime.built_slots({}, bound(None), facts)["gate_route_quant"]
    # slot gate_route_early: built where a layer's early router built
    assert not runtime.built_slots({}, bound(object()), facts)["gate_route_early"]
    assert runtime.built_slots({}, bound(object(), object()), facts)["gate_route_early"]


def test_gate_route_early_is_opt_in_and_needs_gate_route(monkeypatch):
    assert "gate_route_early" in runtime.OPT_IN_SLOTS
    assert "gate_route_early" not in runtime.SHIPPED_SLOTS
    monkeypatch.setattr(
        runtime, "SLOTS", frozenset({"moe", "gate", "gate_route_early"})
    )
    assert any(
        "'gate_route_early' needs slot 'gate_route'" in r
        for r in runtime.slot_reasons()
    )
    monkeypatch.setattr(
        runtime, "SLOTS", frozenset({"moe", "gate", "gate_route", "gate_route_early"})
    )
    assert not any("gate_route_early" in r for r in runtime.slot_reasons())


class _Gemm:
    def __init__(self, serves):
        self.serves = serves
        self.calls = []

    def serves_gate_route_quant(self, rows, input_ids=None, x=None):
        self.calls.append((rows, input_ids, x))
        return self.serves


def _slot(gate, serves, hidden=5120):
    slot = routed.MoeSlot.__new__(routed.MoeSlot)
    slot.gate = gate
    slot.gemm = _Gemm(serves)
    slot.mlp = SimpleNamespace(experts=SimpleNamespace(hidden=hidden))
    return slot


@pytest.fixture
def hidden_size(monkeypatch):
    import sglang.srt.layers.quantization.mxfp4_flashinfer_trtllm_moe as m

    monkeypatch.setattr(m, "routed_hidden_size", lambda experts: experts.hidden)


def test_quant_in_gate_follows_the_pair(hidden_size):
    x = torch.zeros(6, 5120, dtype=torch.bfloat16)
    ids = torch.zeros(6, dtype=torch.int64)
    assert _slot(True, True).quant_in_gate(x, ids, False)
    assert not _slot(True, False).quant_in_gate(x, ids, False)
    assert not _slot(False, True).quant_in_gate(x, ids, False)
    # a padded routed hidden size is not the row the kernel quantizes
    assert not _slot(True, True, hidden=5248).quant_in_gate(x, ids, False)
    # the token ids reach the kernel only with vision routing, as gate_route reads them
    slot = _slot(True, True)
    slot.quant_in_gate(x, ids, False)
    slot.quant_in_gate(x, ids, True)
    assert slot.gemm.calls[0][1] is None and slot.gemm.calls[1][1] is ids


def test_run_refuses_the_quant_without_the_fast_top_k(hidden_size):
    slot = _slot(True, True)
    x = torch.zeros(6, 5120, dtype=torch.bfloat16)
    with pytest.raises(RuntimeError, match="gate_route_quant"):
        slot.run(x, object(), None, quant_in_gate=True)
    with pytest.raises(RuntimeError, match="gate_route_quant"):
        slot.run(x, object(), (x, x, None), quant_in_gate=True)
