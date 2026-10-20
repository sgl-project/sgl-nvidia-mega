# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Slot target_lm_head through SGLang's own LogitsProcessor._compute_lm_head: the
stand-in's quant method runs the low-M kernel at 1-8 rows and SGLang's matmul bit for
bit above; build refuses the heads SGLang would compute another way."""

import logging
import sys
from contextlib import nullcontext
from types import SimpleNamespace

import pytest
import torch

from sgl_nvidia_mega.deepseek_v41 import logits, runtime

DIM, VOCAB = 64, 96


@pytest.fixture
def low_m(monkeypatch):
    """FlashInfer's low-M GEMM on CPU: records each call's rows."""
    calls = []

    def mm_bf16(x, weight_t, *, pdl, backend):
        calls.append(x.shape[0])
        return (x.float() @ weight_t.float()).to(torch.bfloat16)

    monkeypatch.setitem(sys.modules, "flashinfer", SimpleNamespace(mm_bf16=mm_bf16))
    monkeypatch.setitem(
        sys.modules,
        "flashinfer.autotuner",
        SimpleNamespace(autotune=lambda on: nullcontext()),
    )
    monkeypatch.setattr(logits, "low_m_lm_head_supported", lambda: True)
    return calls


def _processor(**over):
    from sglang.srt.layers.logits_processor import LogitsProcessor

    proc = LogitsProcessor.__new__(LogitsProcessor)
    torch.nn.Module.__init__(proc)
    proc.use_fp32_lm_head, proc.rl_on_policy_target = False, None
    for k, v in over.items():
        setattr(proc, k, v)
    return proc


def _lm_head(dtype=torch.bfloat16):
    from sglang.srt.layers.quantization.unquant import UnquantizedEmbeddingMethod

    torch.manual_seed(0)
    weight = torch.randn(VOCAB, DIM).to(dtype)
    return SimpleNamespace(weight=weight, quant_method=UnquantizedEmbeddingMethod())


def _model(**proc):
    return SimpleNamespace(lm_head=_lm_head(), logits_processor=_processor(**proc))


def test_sglang_routes_the_stand_in_to_its_quant_method_and_its_own_head_not():
    from sglang.srt.layers.logits_processor import should_apply_lm_head_quant_method

    head = _lm_head()
    stand_in = logits.TargetLmHead(head)
    assert should_apply_lm_head_quant_method(stand_in, stand_in.quant_method)
    assert not should_apply_lm_head_quant_method(head, head.quant_method)


@pytest.mark.parametrize("rows", [1, 6, 8, 9, 12, 16, 24])
def test_the_low_m_kernel_serves_1_to_8_rows_and_sglangs_matmul_above(low_m, rows):
    model = _model()
    stand_in = logits.TargetLmHead(model.lm_head)
    x = torch.randn(rows, DIM).to(torch.bfloat16)
    ours = model.logits_processor._compute_lm_head(x, stand_in)
    theirs = model.logits_processor._compute_lm_head(x, model.lm_head)
    assert ours.dtype == theirs.dtype and ours.shape == theirs.shape
    if rows <= 8:
        assert low_m == [rows]
        torch.testing.assert_close(ours.float(), theirs.float(), rtol=2**-7, atol=0.1)
    else:
        assert low_m == [] and torch.equal(ours, theirs)


@pytest.mark.parametrize(
    "change, want",
    [
        (lambda m: setattr(m.logits_processor, "use_fp32_lm_head", True), "FP32"),
        (
            lambda m: setattr(m.logits_processor, "rl_on_policy_target", "fsdp"),
            "on-policy",
        ),
        (
            lambda m: (
                setattr(m.lm_head, "set_lora", 1) or setattr(m.lm_head, "apply_lora", 1)
            ),
            "LoRA",
        ),  # fmt: skip
        (
            lambda m: setattr(m.lm_head, "quant_method", SimpleNamespace(apply=print)),
            "quant method",
        ),  # fmt: skip
        (lambda m: setattr(m.lm_head, "weight", m.lm_head.weight.half()), "float16"),
    ],
    ids=["fp32", "on_policy", "lora", "quantized", "fp16"],
)
def test_build_refuses_a_head_sglang_computes_another_way(low_m, change, want):
    model = _model()
    assert logits.target_lm_head_reason(model) == ""
    change(model)
    assert want in logits.target_lm_head_reason(model)


def test_build_tunes_and_returns_the_stand_in_only_when_listed(low_m, monkeypatch):
    model = _model()
    monkeypatch.setattr(runtime, "SLOTS", frozenset())
    assert logits.build(model, None, None) is None and low_m == []
    monkeypatch.setattr(runtime, "SLOTS", frozenset({"target_lm_head"}))
    head = logits.build(model, None, None)
    assert isinstance(head, logits.TargetLmHead) and head.lm_head is model.lm_head
    # the tuning rows, then build's value check through SGLang's _compute_lm_head
    assert low_m == [*logits.TUNING_ROWS, *logits.CHECK_ROWS]
    model.logits_processor.use_fp32_lm_head = True
    assert logits.build(model, None, None) is None


def test_build_refuses_logits_outside_the_tolerance(low_m, monkeypatch, caplog):
    """Values, not only dtype and shape: a kernel off by more than spec.NUMERICS at
    any served row count is refused (zeros would match on both heads)."""
    monkeypatch.setattr(runtime, "SLOTS", frozenset({"target_lm_head"}))
    good = sys.modules["flashinfer"].mm_bf16

    def off_at_6(x, weight_t, **kw):
        out = good(x, weight_t, **kw)
        return out * 1.05 if x.shape[0] == 6 else out

    monkeypatch.setitem(sys.modules, "flashinfer", SimpleNamespace(mm_bf16=off_at_6))
    assert logits.build(_model(), None, None) is None
    assert "at 6 rows" in caplog.text


@pytest.mark.parametrize("others_ok", [True, False])
def test_every_rank_agrees_on_the_verdict(low_m, monkeypatch, others_ok):
    """A rank whose own check passes still refuses when another rank's failed."""
    import torch.distributed as dist

    monkeypatch.setattr(runtime, "SLOTS", frozenset({"target_lm_head"}))
    groups = []

    def all_reduce(flag, op, group):
        groups.append((op, group))
        flag.fill_(int(bool(flag.item()) and others_ok))

    monkeypatch.setattr(dist, "all_reduce", all_reduce)
    tp = SimpleNamespace(world_size=4, cpu_group="cpu")
    head = logits.build(_model(), None, tp)
    assert (head is not None) == others_ok
    assert groups == [(dist.ReduceOp.MIN, "cpu")]


def test_a_capture_logs_each_row_count_once_and_tuning_none(low_m, monkeypatch, caplog):
    method = logits.TargetLmHead(_lm_head()).quant_method
    caplog.set_level(logging.INFO, logger=logits.__name__)
    method.tune()
    monkeypatch.setattr(logits, "_capturing", lambda: True)
    for rows in (6, 6, 8, 12):
        method.apply(None, torch.zeros(rows, DIM, dtype=torch.bfloat16))
    lines = [
        r.getMessage() for r in caplog.records if "target_lm_head" in r.getMessage()
    ]
    assert lines == [
        "NVIDIA megakernel: target_lm_head captured at 6 rows",
        "NVIDIA megakernel: target_lm_head captured at 8 rows",
    ]


@pytest.mark.parametrize("bound", [{}, {"logits": None}, {"logits": "stand-in"}])
def test_the_covered_forward_hands_the_logits_processor_the_built_head(bound):
    seen = []
    lm = SimpleNamespace(
        lm_head="SGLang's head",
        logits_processor=lambda ids, hidden, head, fb, aux, **kw: seen.append(head),
    )
    st = SimpleNamespace(lm=lm, input_ids=None, fb=None, bound=bound)
    logits.logits(st, None, None, None)
    assert seen == [bound.get("logits") or "SGLang's head"]


def test_the_slot_is_opt_in_and_in_the_built_table():
    assert "target_lm_head" in runtime.OPT_IN_SLOTS
    assert "target_lm_head" not in runtime.SHIPPED_SLOTS
    facts = SimpleNamespace(routed=False)
    bound = {
        "moe": None,
        "attention": {},
        "embedding": None,
        "engram": SimpleNamespace(prefetch=None),
    }
    assert (
        runtime.built_slots({}, {**bound, "logits": None}, facts)["target_lm_head"]
        is False
    )
    assert runtime.built_slots({}, {**bound, "logits": "head"}, facts)["target_lm_head"]
