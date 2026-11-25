# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Slot moe_balanced on CPU: on by default, a numerics change, needs moe; the pair's
weights per SGLang storage (range split, balanced storage without the slot: the rank's EP
window, balanced storage with it: every expert, a 384-entry clamp and the rank's EP group
as deal group); reported built only when a pair deals, so a default start without the
balanced storage leaves it off."""

from types import SimpleNamespace

import pytest
import torch

from sgl_nvidia_mega.deepseek_v41 import runtime
from sgl_nvidia_mega.deepseek_v41.moe import routed

E, LOCAL = 384, 192


def test_the_slot_runs_by_default_labelled_numerics_and_needs_moe(monkeypatch):
    assert "moe_balanced" in runtime.DEFAULT_ON_SLOTS
    assert "moe_balanced" in runtime.NUMERICS_CHANGE_SLOTS
    assert "moe_balanced" not in runtime.OPT_IN_SLOTS
    monkeypatch.setattr(runtime, "SLOTS", frozenset({"moe_balanced"}))
    assert any("needs slot 'moe'" in r for r in runtime.slot_reasons())
    monkeypatch.setattr(runtime, "SLOTS", frozenset({"moe", "moe_balanced"}))
    assert runtime.slot_reasons() == []


def _experts(balanced: bool, ep_rank: int):
    stored = E if balanced else LOCAL
    t = {n: torch.arange(stored).view(stored, 1) for n in ("w13", "s13", "w2", "s2")}
    experts = SimpleNamespace(
        w13_weight=t["w13"],
        w13_weight_scale_inv=t["s13"],
        w2_weight=t["w2"],
        w2_weight_scale_inv=t["s2"],
        moe_ep_rank=ep_rank,
        moe_ep_size=2,
        num_local_experts=LOCAL,
        num_experts=E,
        balanced_expert_storage=balanced,
    )
    if balanced:
        experts.expert_window = lambda w: w[ep_rank * LOCAL : (ep_rank + 1) * LOCAL]
    method = SimpleNamespace(_gemm1_clamp_limit_tensor=torch.full((LOCAL,), 7.0))
    return experts, method


@pytest.mark.parametrize("ep_rank", [0, 1])
def test_layer_weights(ep_rank):
    experts, method = _experts(False, ep_rank)
    w, clamp, offset, group = routed.layer_weights(experts, method, deal=True)
    assert w[0].shape[0] == LOCAL and offset == ep_rank * LOCAL and group == -1
    experts, method = _experts(True, ep_rank)
    w, clamp, offset, group = routed.layer_weights(experts, method, deal=False)
    assert int(w[0][0]) == ep_rank * LOCAL and w[0].shape[0] == LOCAL
    assert offset == ep_rank * LOCAL and group == -1 and clamp.numel() == LOCAL
    w, clamp, offset, group = routed.layer_weights(experts, method, deal=True)
    assert w[0].shape[0] == E and offset == 0 and group == ep_rank
    assert clamp.numel() == E and bool((clamp == 7.0).all())


def test_built_only_when_a_pair_deals():
    def built(groups):
        slots = {
            i: SimpleNamespace(gemm=SimpleNamespace(deal_group=g), shared=None)
            for i, g in enumerate(groups)
        }
        bound = {
            "engram": SimpleNamespace(prefetch=None),
            "attention": {},
            "embedding": None,
            "moe": SimpleNamespace(slots=slots),
        }
        return runtime.built_slots({}, bound, SimpleNamespace(routed=True))

    assert built([-1, -1])["moe_balanced"] is False
    assert built([0, 0])["moe_balanced"] is True
