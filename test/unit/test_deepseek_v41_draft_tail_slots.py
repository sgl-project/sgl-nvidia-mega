# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Slots markov_tail and draft_final_collapse on CPU: opt-in draft slots with their
prerequisites, the draft hop built with the collapse only when asked, the startup
summary's state of each, and the proposal step's launches with and without the tail.
The kernels are tested bitwise in test/kernels/test_deepseek_v41_markov_tail.py and
test/kernels/test_deepseek_v41_draft_hop.py."""

from types import SimpleNamespace
from unittest import mock

import pytest
import torch

from sgl_nvidia_mega.deepseek_v41 import runtime

pytest.importorskip("sglang.srt.models.deepseek_v4_mhc")

from sgl_nvidia_mega.deepseek_v41 import draft_slots  # noqa: E402
from sgl_nvidia_mega.deepseek_v41.boundary import boundaries, draft_hop  # noqa: E402

SLOTS = ("markov_tail", "draft_final_collapse")


@pytest.mark.parametrize("slot", SLOTS)
def test_the_slots_are_draft_slots(slot):
    assert slot in runtime.DRAFT_SLOTS
    # markov_tail runs by default; draft_final_collapse has no resolved served gain
    if slot == "markov_tail":
        assert slot in runtime.DEFAULT_ON_SLOTS and slot not in runtime.OPT_IN_SLOTS
    else:
        assert slot in runtime.OPT_IN_SLOTS and slot not in runtime.SHIPPED_SLOTS


@pytest.mark.parametrize(
    "slot, needs", [("markov_tail", "markov"), ("draft_final_collapse", "draft_hop")]
)
def test_the_slots_need_their_prerequisite(monkeypatch, slot, needs):
    monkeypatch.setattr(runtime, "SLOTS", frozenset({slot}))
    assert any(f"'{slot}' needs slot '{needs}'" in r for r in runtime.slot_reasons())
    monkeypatch.setattr(
        runtime, "SLOTS", frozenset({slot, needs, "draft_post", "draft"})
    )
    assert not any(slot in r for r in runtime.slot_reasons())


def _gather(values):
    def gather(out, value, group=None):
        out[:] = values

    return gather


@pytest.mark.parametrize("final_collapse", [False, True])
def test_the_draft_hop_builds_the_collapse_only_when_asked(final_collapse):
    made = []

    def mailbox(stage, top_k, group, deadline_s, max_rows, final_collapse=False):
        made.append(final_collapse)
        return SimpleNamespace(capacity=max_rows)

    group = SimpleNamespace(world_size=8, device_group=None, cpu_group=None)
    parallel = SimpleNamespace(tp_size=8, moe_ep_size=2, tp_group=group)
    stage = SimpleNamespace(
        layer_id=43, mlp=SimpleNamespace(experts=SimpleNamespace(top_k=3))
    )
    with (
        mock.patch("sglang.srt.runtime_context.get_parallel", return_value=parallel),
        mock.patch.object(boundaries, "_fusable", return_value=True),
        mock.patch.object(draft_hop, "_mailbox", mailbox),
        mock.patch.object(torch.distributed, "all_gather_object", _gather([""] * 8)),
    ):
        hop, why = draft_hop.build(
            [stage], deadline_s=1.0, max_rows=48, final_collapse=final_collapse
        )
    assert why == "" and made == [final_collapse]
    assert hop.final_collapse is final_collapse


def test_the_hop_forwards_the_collapse_call():
    calls = []
    inner = SimpleNamespace(
        capacity=48, finalize=lambda *a, **kw: calls.append(kw) or ("s", "c")
    )
    hop = draft_hop.DraftHop(inner, final_collapse=True)
    assert hop.finalize(1, 2, rows=5, collapse=True) == ("s", "c")
    assert calls == [{"mxfp8_out": None, "collapse": True}]


def _built(hop):
    return SimpleNamespace(
        draft_sampler=object(),
        low_m=True,
        widths=(5,),
        _view=SimpleNamespace(model=SimpleNamespace(layers=[])),
        _bound={"boundary": SimpleNamespace(draft_hop=hop)},
        _hop_reason="no fused post",
    )


def test_the_summary_states_each_slot():
    on = SimpleNamespace(capacity=48, final_collapse=True)
    off = SimpleNamespace(capacity=48, final_collapse=False)
    slots = frozenset({"draft", "markov", "markov_tail", "draft_final_collapse"})
    with mock.patch.object(draft_slots, "_forward", wraps=draft_slots._forward):
        got = draft_slots.states(_built(on), slots)
        assert got["markov_tail"][0] == draft_slots.BUILT
        assert got["draft_final_collapse"][0] == draft_slots.BUILT
        assert draft_slots.states(_built(off), slots)["draft_final_collapse"][0] == (
            draft_slots.DECLINED
        )
        assert draft_slots.states(_built(None), slots)["draft_final_collapse"][0] == (
            draft_slots.DECLINED
        )
    built = _built(on)
    built.draft_sampler = None
    assert draft_slots.states(built, slots)["markov_tail"][0] == draft_slots.DECLINED


class _Chain:
    """Records what the proposal step asks of the chain."""

    def __init__(self, tail):
        self.tail, self.pdl, self.calls = tail, False, []

    def fork_noise(self, **kw):
        pass

    def join_noise(self):
        pass

    def sample(self, **kw):
        self.calls.append(kw)
        kw["draft_tokens"].fill_(7)


@pytest.mark.parametrize("tail", [False, True])
@pytest.mark.parametrize("greedy", [False, True])
@pytest.mark.parametrize("with_ids", [False, True])
def test_the_step_hands_the_tail_to_the_chain_or_runs_it(tail, greedy, with_ids):
    from sgl_nvidia_mega.deepseek_v41 import sampler

    bs, gamma, hidden = 2, 3, 4
    lm = SimpleNamespace(
        collapse_hc_head=lambda x: x,
        stages=[SimpleNamespace(norm=lambda x: x)],
    )
    chain = _Chain(tail)
    noise = SimpleNamespace(key=1, offset=torch.tensor([5], dtype=torch.int64))
    tokens = torch.zeros(4 * gamma, dtype=torch.int64)
    verify_ids = torch.full((4, gamma + 1), -1, dtype=torch.int64) if with_ids else None
    step = sampler._Step(
        lm=lm,
        chain=chain,
        lm_head=lambda x: torch.zeros(x.shape[0], 8),
        bs=bs,
        gamma=gamma,
        greedy_draft=greedy,
        noise=noise,
        draft_tokens=tokens,
        step_logits=None,
        step_temperatures=torch.zeros(4),
        verify_ids=verify_ids,
    )
    step.is_logged = True
    anchor = torch.tensor([11, 12])
    step(torch.zeros(bs * gamma, hidden), anchor)
    (call,) = chain.calls
    if tail:
        assert (call["verify_ids"] is not None) == with_ids
        assert (call["noise_offset"] is not None) == (not greedy)
        assert int(noise.offset) == 5  # the chain advances it
    else:
        assert call["verify_ids"] is None and call["noise_offset"] is None
        assert int(noise.offset) == (5 if greedy else 6)
        if with_ids:
            assert verify_ids[:bs].tolist() == [[11, 7, 7, 7], [12, 7, 7, 7]]
    if with_ids:
        assert verify_ids[bs:].eq(-1).all()
