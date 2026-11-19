# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""CPU checks of slot markov against SGLang's DSpark model and graph runner (the
slot's other CPU checks are test_minimax_m31_draft_slots.py): the Markov heads the
chain refuses, with the reason, and the engagement check, which stops a server whose
folded proposal never asks for the draft sampler at its first step after graph
capture."""

from types import SimpleNamespace

import pytest

pytest.importorskip("triton")
pytest.importorskip("sglang.srt.models.dspark")

import torch  # noqa: E402

from sgl_nvidia_mega.minimax_m31 import draft, sampler  # noqa: E402
from sgl_nvidia_mega.minimax_m31.sm103.draft import markov  # noqa: E402


def _head(vocab=1024, w2_dtype=torch.bfloat16):
    from sglang.srt.models.dspark import VanillaMarkov

    head = VanillaMarkov(vocab_size=vocab, markov_rank=markov.RANK_DIM)
    head.to(torch.bfloat16)
    head.markov_w2.weight.data = head.markov_w2.weight.data.to(w2_dtype)
    return head


def test_the_sampler_names_why_the_chain_cannot_serve_a_markov_head():
    """Checked at the draft spec's load, before the LM head is attached: the chain
    computes VanillaMarkov's bias, markov_w2 . markov_w1[previous], from BF16 tables."""

    def reason(head):
        return sampler.head_reason(SimpleNamespace(markov_head=head), gamma=7, world=4)

    assert "no Markov head" in reason(None)
    # a head of another class, whatever its tables
    assert "needs VanillaMarkov" in reason(SimpleNamespace(vocab_size=1024))
    assert "FP32 markov_w2" in reason(_head(w2_dtype=torch.float32))


def test_a_server_that_never_asks_for_the_draft_sampler_stops(monkeypatch):
    """The target's executor calls the check on every forward: it waits for SGLang's
    graph capture, then, at the first forward outside it, raises unless the folded
    proposal asked for the draft sampler (the DSpark draft graph, captured after the
    target's)."""
    from sglang.srt.model_executor import runner

    capturing = {"on": False}
    monkeypatch.setattr(runner, "get_is_capture_mode", lambda: capturing["on"])
    monkeypatch.setattr(draft, "_COUNTS", {"built": 1, "sampler": 0})
    check = draft.EngagementCheck(frozenset({"markov"}))
    assert check() is False  # before any capture: an autotune pass, say
    capturing["on"] = True
    assert check() is False
    capturing["on"] = False
    with pytest.raises(RuntimeError, match="never asked for the draft sampler"):
        check()
    monkeypatch.setattr(draft, "_COUNTS", {"built": 0, "sampler": 0})
    with pytest.raises(RuntimeError, match="no DSpark draft spec was built"):
        check()
    monkeypatch.setattr(draft, "_COUNTS", {"built": 1, "sampler": 1})
    assert check() is True
