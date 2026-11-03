# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""The Qwen3.5 megakernel's runtime on CPU: the steps it covers, the binding and the
EAGLE entries at build, the MTP draft spec, the step state and the parity check."""

from types import SimpleNamespace

import pytest
import torch
from sglang.kernels.plugin_kernels.nvidia import spec as registry
from sglang.kernels.plugin_kernels.nvidia.spec import Step

from sgl_nvidia_mega import plugin
from sgl_nvidia_mega.qwen3_5 import (
    runtime,
)
from sgl_nvidia_mega.qwen3_5.runtime import executor, parity
from sgl_nvidia_mega.qwen3_5.runtime.state import StepState


@pytest.fixture(autouse=True)
def registered(monkeypatch):
    monkeypatch.setattr(registry, "_REGISTRY", {})
    monkeypatch.delenv(plugin.PARITY_SORT_ENV, raising=False)
    plugin.register()


# ---------------------------------------------------------------- the covered steps


def _verify(widths=(7,), binding="binding"):
    """An executor over a model with no layers (plan reads only host state), built
    with SGLang's EAGLE binding."""
    lm = SimpleNamespace(model=SimpleNamespace(start_layer=0, end_layer=0, layers=[]))
    return executor.Qwen35Verify(lm, {}, {}, widths, binding=binding)


def _batch(verify=True):
    return SimpleNamespace(
        forward_mode=SimpleNamespace(is_target_verify=lambda: verify)
    )


def test_only_target_verify_steps_of_built_widths_are_covered():
    built = _verify()
    ids = torch.zeros(7, dtype=torch.int64)
    assert built.plan(ids, _batch()) == Step(mode="target_verify", rows=7)
    assert built.plan(ids, _batch(), pp_proxy_tensors=None) == Step(
        mode="target_verify", rows=7
    )
    assert built.plan(torch.zeros(14, dtype=torch.int64), _batch()) is None
    assert built.plan(ids, _batch(verify=False)) is None
    assert built.plan(ids, _batch(), get_embedding=True) is None
    assert built.plan(ids, _batch(), pp_proxy_tensors=object()) is None
    # an argument the executor does not know leaves the step to SGLang
    assert built.plan(ids, _batch(), input_embeds=None) is None


def test_the_covered_widths_are_the_ones_the_build_returns():
    built = _verify(widths=(7, 14, 28))
    for rows in (7, 14, 28):
        ids = torch.zeros(rows, dtype=torch.int64)
        assert built.plan(ids, _batch()) == Step(mode="target_verify", rows=rows)
    assert built.plan(torch.zeros(21, dtype=torch.int64), _batch()) is None


@pytest.mark.parametrize("tp_size, want", [(8, (7,)), (4, (7,))])
def test_the_build_covers_more_requests_only_where_a_slot_runs(
    monkeypatch, tp_size, want
):
    monkeypatch.setattr(runtime, "SLOTS", frozenset())
    monkeypatch.setattr(
        "sglang.srt.runtime_context.get_parallel",
        lambda: SimpleNamespace(tp_size=tp_size),
    )
    model = SimpleNamespace(
        model=SimpleNamespace(start_layer=0, end_layer=0, layers=[])
    )
    # what SGLang offers for the spec's MAX_ROWS with 7 rows a request
    ctx = SimpleNamespace(
        widths=(7, 14, 21, 28),
        tokens_per_request=7,
        deadline_s=60.0,
        binding="EagleBinding",
    )
    built = runtime.build(model, ctx)
    assert built.widths == want and built.summary() == "slots none (stock)"


def test_the_build_hands_over_the_binding_and_every_entry_stays_sglangs():
    from sglang.kernels.plugin_kernels.nvidia.spec import ENTRIES

    from sgl_nvidia_mega.qwen3_5 import draft

    built = _verify()
    assert built.binding == "binding"
    # covered from the first step on: the binding comes with the build
    ids = torch.zeros(7, dtype=torch.int64)
    assert built.plan(ids, _batch()) == Step(mode="target_verify", rows=7)
    target, mtp = built, draft.SPEC.build("lm", None, _draft_ctx())
    assert [n for n in ENTRIES if getattr(target, n, None) is not None] == []
    assert [n for n in ENTRIES if getattr(mtp, n, None) is not None] == []


def test_verify_sample_is_refused_with_the_switch_that_replaces_it(monkeypatch):
    monkeypatch.setattr(runtime, "SLOTS", frozenset({"route", "verify_sample"}))
    assert [r.split(": ", 1)[1] for r in runtime.slot_reasons()] == [
        "'verify_sample' is no longer an entry: drop it; an SGLang with post-verify "
        "graphs turns them on with SGLANG_ENABLE_EAGLE_POST_VERIFY_CUDA_GRAPH=1"
    ]


def test_the_runtime_gives_the_executor_the_build_contexts_binding(monkeypatch):
    monkeypatch.setattr(runtime, "SLOTS", frozenset())
    model = SimpleNamespace(
        model=SimpleNamespace(start_layer=0, end_layer=0, layers=[])
    )
    ctx = SimpleNamespace(
        widths=(7,), tokens_per_request=7, deadline_s=60.0, binding="EagleBinding"
    )
    built = runtime.build(model, ctx)
    assert built.binding == "EagleBinding" and built.summary() == "slots none (stock)"


def _draft_ctx():
    return SimpleNamespace(
        widths=(), tokens_per_request=7, deadline_s=60.0, binding="EagleBinding"
    )


def test_the_mtp_draft_covers_no_step():
    from sglang.kernels.plugin_kernels.nvidia.spec import ENTRIES

    from sgl_nvidia_mega.qwen3_5 import draft

    built = draft.SPEC.build("lm", SimpleNamespace(context_len=4096), _draft_ctx())
    assert (
        draft.SPEC.max_rows == 0 and built.widths == () and built.draft_sampler is None
    )
    assert built.plan(torch.zeros(7, dtype=torch.int64), _batch()) is None
    assert (draft.ARCHITECTURE, "sgl_nvidia_mega.qwen3_5.draft:SPEC") in plugin.SPECS
    # without a request every draft phase stays SGLang's
    assert [n for n in ENTRIES if getattr(built, n, None) is not None] == []


def test_the_draft_entries_are_offered_only_on_request(monkeypatch):
    from sglang.kernels.plugin_kernels.nvidia.spec import ENTRIES

    from sgl_nvidia_mega.qwen3_5 import draft

    monkeypatch.setattr(runtime, "SLOTS", frozenset({"draft_loop", "draft_extend"}))
    assert runtime.slot_reasons() == []
    built = draft.SPEC.build("lm", SimpleNamespace(context_len=4096), _draft_ctx())
    assert [n for n in ENTRIES if getattr(built, n, None) is not None] == [
        "draft_loop",
        "draft_extend",
        "begin_draft_round",
    ]
    assert built.context_len == 4096 and built.binding == "EagleBinding"
    # the target's build ignores the draft's names
    model = SimpleNamespace(
        model=SimpleNamespace(start_layer=0, end_layer=0, layers=[])
    )
    target_ctx = SimpleNamespace(
        widths=(7,), tokens_per_request=7, deadline_s=60.0, binding="EagleBinding"
    )
    assert runtime.build(model, target_ctx).summary() == "slots none (stock)"


@pytest.mark.parametrize(
    "batch_size, lower_bound, seq_lens, want",
    [
        (1, 16384, None, "q1_long"),
        (1, 16383, None, "q1_stock"),
        (1, None, [20000], "q1_long"),
        (1, None, [100], "q1_stock"),
        (1, None, None, "q1_stock"),
        (2, 50000, None, "q1_stock"),
    ],
)
def test_the_draft_loop_routes_q1_by_kv_length_as_2_2(
    monkeypatch, batch_size, lower_bound, seq_lens, want
):
    from sgl_nvidia_mega.qwen3_5 import draft

    monkeypatch.setattr(runtime, "SLOTS", frozenset({"draft_loop", "draft_q1"}))
    built = draft.SPEC.build("lm", SimpleNamespace(context_len=4096), _draft_ctx())
    variants = built.graph_variants["draft_loop"]
    assert variants.capture_labels == ("q1_stock", "q1_long")
    batch = SimpleNamespace(
        batch_size=batch_size,
        attention_kv_len_lower_bound=lower_bound,
        seq_lens_cpu=None if seq_lens is None else torch.tensor(seq_lens),
    )
    assert variants.select(batch) == want


def test_the_draft_extend_takes_the_split_kv_variants_with_slot_draft_skv2(
    monkeypatch,
):
    from sgl_nvidia_mega.qwen3_5 import draft
    from sgl_nvidia_mega.qwen3_5.attention import route

    # inside entry draft_extend only: the draft loop is not needed
    monkeypatch.setattr(runtime, "SLOTS", frozenset({"draft_extend", "draft_skv2"}))
    assert runtime.slot_reasons() == []
    built = draft.SPEC.build("lm", SimpleNamespace(context_len=4096), _draft_ctx())
    assert built.graph_variants == {"draft_extend": route.EXTEND}
    assert route.EXTEND.min_length == 32768
    # with draft_q1 the draft offers each runner its own variants
    monkeypatch.setattr(
        runtime,
        "SLOTS",
        frozenset({"draft_loop", "draft_q1", "draft_extend", "draft_skv2"}),
    )
    built = draft.SPEC.build("lm", SimpleNamespace(context_len=4096), _draft_ctx())
    assert sorted(built.graph_variants) == ["draft_extend", "draft_loop"]
    assert built.graph_variants["draft_extend"] is route.EXTEND


@pytest.mark.parametrize(
    "slots, want",
    [
        ({"draft_moe"}, "slot 'draft_moe' runs inside entry 'draft_loop'"),
        ({"draft_loop", "draft_hop"}, "slot 'draft_hop' runs on slot 'hop''s mailbox"),
        (
            {"draft_loop", "draft_skv2"},
            "slot 'draft_skv2' runs inside entry 'draft_extend'",
        ),
    ],
)
def test_the_draft_slots_name_what_they_need(monkeypatch, slots, want):
    monkeypatch.setattr(runtime, "SLOTS", frozenset(slots))
    assert [r.split(": ", 1)[1] for r in runtime.slot_reasons()] == [want]


class _DraftParts:
    """Parts with no slot built: the stock model step."""

    def __init__(self, slots=frozenset(), tp_size=8):
        self.slots, self.mkd, self.engaged = slots, None, []
        self.binding = SimpleNamespace(tp_size=tp_size)

    def note_engaged(self, entry, bs):
        self.engaged.append((entry, bs))


def _stub_model_step(monkeypatch, calls):
    """The draft's model step as a log: fc_input -> x, layer -> hidden, top1 -> token."""
    from sgl_nvidia_mega.qwen3_5.draft import layer

    def fc_input(parts, token, hidden):
        calls.append(("fc", token.tolist(), hidden.tolist()))
        return hidden + 1

    def run_layer(parts, x, positions, cache_loc, meta, kv, *, rows_per_request):
        calls.append(("layer", positions.tolist(), cache_loc.tolist(), meta, kv))
        return x * 10

    def top1(parts, hidden):
        calls.append(("top1", hidden.tolist()))
        return hidden[:, 0].to(torch.int64) % 1000

    monkeypatch.setattr(layer, "fc_input", fc_input)
    monkeypatch.setattr(layer, "layer", run_layer)
    monkeypatch.setattr(layer, "top1", top1)


def test_the_draft_loop_writes_each_steps_token_and_restores_the_positions(
    monkeypatch,
):
    from sgl_nvidia_mega.qwen3_5.draft import SERVED_BS
    from sgl_nvidia_mega.qwen3_5.draft.loop import DraftLoop

    calls = []
    _stub_model_step(monkeypatch, calls)
    parts = _DraftParts()
    loop = DraftLoop(parts, num_steps=4)
    assert loop.step_for(SERVED_BS) == loop.step
    io = SimpleNamespace(
        num_requests=1,
        num_steps=4,
        topk_index=torch.tensor([[7]]),
        hidden_states=torch.tensor([[1.0, 2.0]]),
        positions=torch.tensor([100]),
        mrope_positions=torch.tensor([[100], [100], [100]]),
        # request-major: step i writes slot 40 + i
        out_cache_loc=torch.tensor([40, 41, 42, 43]),
        steps=("m0", "m1", "m2"),
        kv="kv",
        draft_tokens_out=torch.full((1, 4), -1, dtype=torch.int64),
    )
    loop.step(io)
    assert parts.engaged == [("draft_loop", 1)]
    # three draft steps at positions 100..102 on slots 40..42 with step i's metadata
    layers = [c for c in calls if c[0] == "layer"]
    assert [c[1:] for c in layers] == [
        ([100], [40], "m0", "kv"),
        ([101], [41], "m1", "kv"),
        ([102], [42], "m2", "kv"),
    ]
    # step i + 1 embeds step i's token after its final-norm hidden state
    fcs = [c for c in calls if c[0] == "fc"]
    assert [c[1] for c in fcs] == [[7], [20], [210]]
    assert io.draft_tokens_out.tolist() == [[7, 20, 210, 110]]
    assert io.positions.tolist() == [100] and io.mrope_positions.tolist() == [
        [100],
        [100],
        [100],
    ]


@pytest.mark.parametrize(
    "tp_size, slots, served",
    [
        (8, {"multi_request"}, (1, 2)),
        (8, set(), (1,)),
        (4, {"multi_request"}, (1,)),
    ],
)
def test_the_draft_loop_serves_two_requests_at_tp8_on_request(
    monkeypatch, tp_size, slots, served
):
    from sgl_nvidia_mega.qwen3_5.draft.loop import DraftLoop

    monkeypatch.setattr(runtime, "SLOTS", frozenset(slots))
    loop = DraftLoop(_DraftParts(tp_size=tp_size), num_steps=4)
    assert tuple(bs for bs in (1, 2, 4) if loop.step_for(bs) == loop.step) == served


@pytest.mark.parametrize("split_fc", [False, True])
def test_a_two_request_draft_loop_runs_each_step_on_dense_rows(monkeypatch, split_fc):
    from sgl_nvidia_mega.qwen3_5.draft import layer
    from sgl_nvidia_mega.qwen3_5.draft.loop import DraftLoop

    calls = []
    _stub_model_step(monkeypatch, calls)
    stub_layer = layer.layer

    def run_layer(parts, x, positions, cache_loc, meta, kv, *, rows_per_request):
        calls.append(("dense", cache_loc.is_contiguous()))
        return stub_layer(
            parts, x, positions, cache_loc, meta, kv, rows_per_request=rows_per_request
        )

    monkeypatch.setattr(layer, "layer", run_layer)
    parts = _DraftParts()
    if split_fc:
        # slot draft_xchg's row-split fc, never its one-request exchange steps
        parts.model = SimpleNamespace(embed_tokens=lambda t: t.to(torch.float32))
        parts.mkd = SimpleNamespace(
            split_fc=lambda e, h: (calls.append(("split", e.tolist())), h + 1)[1]
        )
    loop = DraftLoop(parts, num_steps=4)
    io = SimpleNamespace(
        num_requests=2,
        num_steps=4,
        topk_index=torch.tensor([[7], [8]]),
        hidden_states=torch.tensor([[1.0, 2.0], [3.0, 4.0]]),
        positions=torch.tensor([100, 200]),
        mrope_positions=None,
        # request-major: request r's step i writes slot 40 + 10 r + i
        out_cache_loc=torch.tensor([40, 41, 42, 43, 50, 51, 52, 53]),
        steps=("m0", "m1", "m2"),
        kv="kv",
        draft_tokens_out=torch.full((2, 4), -1, dtype=torch.int64),
    )
    loop.step(io)
    assert parts.engaged == [("draft_loop", 2)]
    assert [c[1:] for c in calls if c[0] == "layer"] == [
        ([100, 200], [40, 50], "m0", "kv"),
        ([101, 201], [41, 51], "m1", "kv"),
        ([102, 202], [42, 52], "m2", "kv"),
    ]
    assert [c[1] for c in calls if c[0] == "dense"] == [True] * 3
    fc = "split" if split_fc else "fc"
    assert [c[1] for c in calls if c[0] == fc] == [[7, 8], [20, 40], [210, 410]]
    assert io.draft_tokens_out.tolist() == [[7, 20, 210, 110], [8, 40, 410, 110]]
    assert io.positions.tolist() == [100, 200]


def test_the_draft_extend_writes_the_selected_rows_outputs(monkeypatch):
    from sgl_nvidia_mega.qwen3_5.draft.extend import DraftExtend

    calls = []
    _stub_model_step(monkeypatch, calls)
    parts = _DraftParts()
    extend = DraftExtend(parts, tokens_per_request=3)
    assert extend.step_for(1) == extend.step and extend.step_for(4) is None
    io = SimpleNamespace(
        num_requests=1,
        tokens_per_request=3,
        input_ids=torch.tensor([5, 6, 7]),
        positions=torch.tensor([10, 11, 12]),
        hidden_states=torch.tensor([[1.0], [2.0], [3.0]]),
        out_cache_loc=torch.tensor([20, 21, 22]),
        select_index=torch.tensor([1]),
        attn="meta",
        kv="kv",
        topk_index_out=torch.zeros(1, 1, dtype=torch.int64),
        topk_p_out=torch.zeros(1, 1),
        hidden_out=torch.zeros(1, 1),
    )
    extend.step(io)
    assert parts.engaged == [("draft_extend", 1)]
    assert [c[1:] for c in calls if c[0] == "layer"] == [
        ([10, 11, 12], [20, 21, 22], "meta", "kv")
    ]
    # the selected row 1: (2 + 1) * 10
    assert io.hidden_out.tolist() == [[30.0]]
    assert io.topk_index_out.tolist() == [[30]] and io.topk_p_out.tolist() == [[1.0]]


# ------------------------------------------------------------------- the parity check


@pytest.fixture
def eager(monkeypatch):
    """Outside CUDA graph capture (a CPU build of torch has no capture to query), with
    an attention backend that has no linear-attention states."""
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)
    monkeypatch.setattr(
        "sglang.srt.model_executor.forward_context.get_attn_backend",
        lambda: SimpleNamespace(),
    )


def test_the_second_forward_sees_the_conv_windows_the_first_saw(eager, monkeypatch):
    conv = torch.zeros(2, 3, 4)  # [layers, slots, ...]
    cache = SimpleNamespace(conv=[conv])
    linear = SimpleNamespace(
        req_to_token_pool=SimpleNamespace(
            mamba_pool=SimpleNamespace(mamba_cache=cache)
        ),
        forward_metadata=SimpleNamespace(mamba_cache_indices=torch.tensor([1, -1])),
    )
    monkeypatch.setattr(
        "sglang.srt.model_executor.forward_context.get_attn_backend",
        lambda: SimpleNamespace(linear_attn_backend=linear),
    )

    def forward():
        # reads slot 1, then rewrites it in place, as a verify forward's conv update does
        out = conv[:, 1].sum().reshape(1, 1).clone()
        conv[:, 1] += 1.0
        return SimpleNamespace(next_token_logits=out, hidden_states=None)

    got = parity.Parity((7,), "cpu").run(Step("target_verify", 7), forward, forward)
    assert got.next_token_logits.item() == 0.0  # the restored state, not 8.0
    assert conv[:, 1].eq(1.0).all() and conv[:, 0].eq(0.0).all()


def test_parity_compares_bits_not_values():
    a = torch.tensor([0.0, 1.0, float("nan")])
    assert int(parity.differing(a, a.clone())) == 0
    assert int(parity.differing(a, torch.tensor([-0.0, 1.0, float("nan")]))) == 1
    assert (
        int(parity.differing(a, torch.tensor([0.0, 1.0 + 2**-20, float("nan")]))) == 1
    )
    assert parity.differing(None, None) == 0
    with pytest.raises(RuntimeError):
        parity.differing(a, a[:2])
    with pytest.raises(RuntimeError):
        parity.differing(a, None)


def test_parity_counts_the_requests_own_rows_apart():
    # two requests of 2 rows; the second is a padded one (the pad slot)
    valid = parity.valid_rows(torch.tensor([3, -1]), 4)
    assert valid.tolist() == [True, True, False, False]
    a = torch.zeros(4, 3)
    b = a.clone()
    b[3] = 1.0  # a padded row
    assert int(parity.differing(a, b)) == 3 and int(parity.differing(a, b, valid)) == 0
    b[0, 1] = 1.0  # a request's row
    assert int(parity.differing(a, b, valid)) == 1


@pytest.mark.parametrize("mode", ["1", "control"])
def test_an_eager_parity_step_runs_sglangs_forward_first(
    eager, monkeypatch, caplog, mode
):
    monkeypatch.setattr(parity, "MODE", mode)
    order = []

    def output(name, value):
        def run():
            order.append(name)
            return SimpleNamespace(
                next_token_logits=torch.full((7, 3), value), hidden_states=None
            )

        return run

    checker = parity.Parity((7,), "cpu")
    with caplog.at_level("INFO"):
        got = checker.run(
            Step("target_verify", 7), output("sglang", 1.0), output("plugin", 2.0)
        )
    assert order == (["sglang", "plugin"] if mode == "1" else ["sglang", "sglang"])
    want = 0 if mode == "control" else 21
    assert (
        f"rows=7 logits_differ={want} hidden_differ=0 logits_compared=21 "
        "hidden_compared=0" in caplog.text
    )
    assert got.next_token_logits[0, 0] == (1.0 if mode == "control" else 2.0)


def test_a_capture_warmup_is_logged_apart_from_the_eager_steps(
    eager, monkeypatch, caplog
):
    monkeypatch.setattr(
        "sglang.srt.model_executor.runner.get_is_capture_mode", lambda: True
    )

    def output():
        return SimpleNamespace(next_token_logits=torch.zeros(7, 3), hidden_states=None)

    with caplog.at_level("INFO"):
        parity.Parity((7,), "cpu").run(Step("target_verify", 7), output, output)
    assert "parity capture-warmup target_verify rows=7 logits_differ=0" in caplog.text
    assert " eager " not in caplog.text


@torch.no_grad()
def test_trace_names_the_first_module_call_that_differs(eager, monkeypatch, caplog):
    monkeypatch.setattr(parity, "MODE", "trace")
    model = torch.nn.Sequential(torch.nn.Linear(4, 4), torch.nn.Linear(4, 4))
    x = torch.ones(7, 4)

    def forward(shift):
        def run():
            h = model[0](x) + shift  # an op between two modules
            return SimpleNamespace(next_token_logits=model[1](h), hidden_states=None)

        return run

    checker = parity.Parity((7,), "cpu", model)
    with caplog.at_level("INFO"):
        checker.run(Step("target_verify", 7), forward(0.0), forward(0.0))
        checker.run(Step("target_verify", 7), forward(0.0), forward(1.0))
    traces = [r.message for r in caplog.records if "parity trace" in r.message]
    assert traces == [
        "NVIDIA megakernel parity trace target_verify rows=7: no diverging call "
        "among 2 common module calls",
        "NVIDIA megakernel parity trace target_verify rows=7: 1#0 input: 28 of 28 "
        "elements differ",
    ]


def test_graph_parity_counts_are_logged_once_each(eager, caplog):
    checker = parity.Parity((7,), "cpu")
    checker.counters[7] += torch.tensor([3, 0, 5, 63, 0, 0, 2])
    checker.distance[7] += torch.tensor(
        [21.0, 2.0, 0.42, 0.25, 0.5], dtype=torch.float64
    )
    with caplog.at_level("INFO"):
        checker.flush()
        checker.flush()
    lines = [r.message for r in caplog.records if "parity graph" in r.message]
    assert lines == [
        "NVIDIA megakernel parity graph target_verify rows=7 steps=3 "
        "logits_differ=0 hidden_differ=5 logits_compared=63 hidden_compared=0 "
        "valid_logits_differ=0 valid_hidden_differ=2 "
        "rows_compared=21 top1_differ=2 dlp_mean=0.02 dlp_max=0.25 hidden_dist_max=0.5"
    ]
    # the distance restarts with every flush
    assert checker.distance[7].tolist() == [0.0] * 5


def test_the_parity_distance_reads_top1_flips_and_log_prob_gaps():
    a = torch.tensor([[2.0, 0.0, 0.0], [0.0, 3.0, 0.0]])
    b = torch.tensor([[2.0, 0.0, 0.0], [0.0, 1.0, 2.0]])  # row 1's top-1 moves
    rows, top1, total, worst, hidden = parity.distance(
        a, b, torch.zeros(2, 4), torch.full((2, 4), 0.25)
    ).tolist()
    gap = torch.log_softmax(a[1], -1)[1] - torch.log_softmax(b[1], -1)[1]
    assert (rows, top1, hidden) == (2.0, 1.0, 0.25)
    assert total == pytest.approx(gap.item()) and worst == pytest.approx(gap.item())
    assert parity.distance(None, None, None, None) is None


def test_the_step_state_carries_the_binding():
    lm = SimpleNamespace(model="model")
    st = StepState(
        lm=lm,
        forward_batch="fb",
        positions="pos",
        input_ids="ids",
        slots="slots",
        bound={},
        binding="EagleBinding",
        view="EagleForwardView",
    )
    assert (st.model, st.binding, st.view) == (
        "model",
        "EagleBinding",
        "EagleForwardView",
    )
