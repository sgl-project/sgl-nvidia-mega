# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""The executor and SGLang's binding on CPU: the layer loop every package codes
against, the binding reaching every reader, the slots' reads in SGLang's real init
order (scheduler_init.py), and the Engram prefetch's host table."""

import contextlib
from types import SimpleNamespace
from unittest import mock

import pytest
import torch

pytest.importorskip("sglang.srt.speculative.dspark_components.dspark_worker_v2")

from scheduler_init import (  # noqa: E402
    STAGES,
    EarlyRead,
    Scenario,
    SchedulerInit,
)
from sglang.srt.megakernels.spec import Step  # noqa: E402

from sgl_nvidia_mega.deepseek_v41 import runtime, spec  # noqa: E402
from sgl_nvidia_mega.deepseek_v41.engram import prefetch  # noqa: E402
from sgl_nvidia_mega.deepseek_v41.runtime import executor  # noqa: E402
from sgl_nvidia_mega.deepseek_v41.runtime import state  # noqa: E402
from sgl_nvidia_mega.deepseek_v41.runtime.state import StepState  # noqa: E402

# ------------------------------------------------------------------------------------
# The executor's layer loop on CPU: the order in which it calls the step functions,
# which every package codes against.


@pytest.fixture
def calls(monkeypatch):
    """Every step function replaced by a recorder."""
    seen = []

    def record(name, ret=None):
        def fn(*args, **kwargs):
            layer = next((a for a in args if isinstance(a, SimpleNamespace)), None)
            sub = next((a for a in args if a in ("attn", "ffn")), None)
            index = next((a for a in args if type(a) is int), None)
            seen.append((name, getattr(layer, "i", index), sub))
            return ret

        return fn

    for pkg, names in {
        "boundary": ["start", "materialize", "pre", "post"],
        "attention": ["step"],
        "moe": ["step"],
        "engram": ["hashes", "apply", "before_layer", "before_moe"],
    }.items():
        for name in names:
            monkeypatch.setattr(
                getattr(executor, pkg), name, record(f"{pkg}.{name}", "x")
            )
    monkeypatch.setattr(executor.boundary, "final", record("boundary.final", (1, 2)))

    @contextlib.contextmanager
    def with_current_layer(i):
        yield

    recorder = SimpleNamespace(with_current_layer=with_current_layer)
    monkeypatch.setattr(
        "sglang.srt.eplb.expert_distribution.get_global_expert_distribution_recorder",
        lambda: recorder,
    )
    return seen


def _state(engram_layers=(), capture=None):
    layers = [
        SimpleNamespace(i=i, engram=object() if i in engram_layers else None)
        for i in range(3)
    ]
    model = SimpleNamespace(
        start_layer=0,
        end_layer=3,
        layers=layers,
        dspark_layers_to_capture=capture,
    )
    st = StepState(
        lm=SimpleNamespace(model=model),
        forward_batch=None,
        positions=None,
        input_ids=torch.zeros(6, dtype=torch.int64),
        input_ids_global=None,
        slots=None,
        bound={},
    )
    st.aux = [] if capture is not None else None
    return st


LAYER = [
    ("engram.before_layer", None),
    ("boundary.pre", "attn"),
    ("attention.step", None),
    ("boundary.post", "attn"),
    ("engram.before_moe", None),
    ("moe.step", None),
    ("boundary.post", "ffn"),
]


def test_each_layer_runs_pre_attention_post_moe_post(calls):
    out = executor.MegakernelDecode(None, {}, {}, ())._layers(
        _state(), torch.zeros(6, 4, 8)
    )
    assert out == (1, 2)
    assert calls[:2] == [("engram.hashes", None, None), ("boundary.start", None, None)]
    body = [(name, sub) for name, _, sub in calls[2:-1]]
    assert body == LAYER * 3
    assert [i for _, i, _ in calls[2:-1]] == [i for i in range(3) for _ in LAYER]
    assert calls[-1][0] == "boundary.final"


def test_engram_and_capture_layers_read_the_materialized_streams(calls, monkeypatch):
    # the capture mean on the main stream (slot aux_mean_stats off)
    monkeypatch.setattr(executor, "AUX_MEAN_STATS", False)
    st = _state(engram_layers=(1,), capture=[2])
    st_residual = torch.ones(6, 4, 8)

    def materialize(layer, st):
        calls.append(("boundary.materialize", layer.i, None))
        st.state = SimpleNamespace(residual=st_residual)

    monkeypatch.setattr(executor.boundary, "materialize", materialize)
    executor.MegakernelDecode(None, {}, {}, ())._layers(st, torch.zeros(6, 4, 8))
    names = [(name, i) for name, i, _ in calls]
    # the prefetch's gather point before layer 1's streams; Engram before its attention
    assert names.index(("engram.before_layer", 1)) < names.index(
        ("boundary.materialize", 1)
    )
    assert names.index(("boundary.materialize", 1)) < names.index(("engram.apply", 1))
    assert names.index(("engram.apply", 1)) < names.index(("boundary.pre", 1))
    assert names.index(("boundary.materialize", 2)) < names.index(("boundary.pre", 2))
    assert ("engram.apply", 2) not in names and ("boundary.materialize", 0) not in names
    assert len(st.aux) == 1 and torch.equal(st.aux[0], st_residual.mean(dim=1))


def test_the_embedding_is_sglangs_unless_its_slot_is_on():
    from sgl_nvidia_mega.deepseek_v41 import plan
    from sgl_nvidia_mega.deepseek_v41.runtime import embedding

    looked_up = []
    st = _state()
    st.model.embed_tokens = lambda ids: looked_up.append(ids) or "stock"
    st.slots = plan.slots(6, plan.Facts())
    st.bound = {"embedding": torch.zeros(4, 4)}
    assert embedding.embed(st) == "stock" and looked_up == [st.input_ids]
    # a built table without the slot, or the slot without a table: still stock
    st.slots = plan.slots(6, plan.Facts(embed=True))
    st.bound = {"embedding": None}
    assert embedding.embed(st) == "stock"


def _engram(layout="per_rank", backend="cutedsl", columns=25600):
    from sglang.srt.layers.quantization.fp8 import Fp8LinearMethod
    from sglang.srt.layers.quantization.fp8_utils import Mxfp8DenseGemmBackend

    method = object.__new__(Fp8LinearMethod)
    method.use_mxfp8, method.block_fp8_as_mxfp8 = False, True
    method.mxfp8_dense_backend = Mxfp8DenseGemmBackend(f"flashinfer_{backend}")
    table = None if layout is None else SimpleNamespace(layout=layout)
    wkv = SimpleNamespace(
        quant_method=method,
        block_fp8_mxfp8_ready=True,
        weight=torch.empty(columns, 6144, dtype=torch.float8_e4m3fn),
    )
    return SimpleNamespace(embed=SimpleNamespace(host_table=table), wkv=wkv)


def test_the_engram_prefetch_takes_only_the_per_rank_table():
    from sgl_nvidia_mega.deepseek_v41.engram import prefetch as ep

    def reason(engram):
        # the binding hands the host table (MegakernelBinding.engram_host_table)
        return ep.reason(engram, 4, engram.embed.host_table)

    assert reason(_engram()) == ""
    assert reason(_engram(backend="cutlass")) == ""
    # the shared table and the device table keep SGLang's inline lookup + WKV
    for layout in (None, "shared"):
        assert "SGLANG_DSV41_ENGRAM_HOST_TABLE_LAYOUT=per_rank" in reason(
            _engram(layout)
        )
    assert "not a FlashInfer MXFP8" in reason(_engram(backend="trtllm"))
    assert "scale tiles" in reason(_engram(columns=25600 + 128))


def test_the_engram_prefetch_holds_the_reduced_rows_until_gather(monkeypatch):
    """reduce()'s all-reduced rows are read on the side stream after reduce() returns.
    An out-of-place all-reduce's record_stream does not keep them (the allocator ignores
    it on DLPack tensors), so the prefetch holds them until gather() joins wkv_done."""
    import gc
    import weakref

    waits, refs = [], []

    class Stream:
        def wait_event(self, event):
            waits.append((event, refs[-1]() is not None if refs else None))

        def wait_stream(self, other):
            pass

    class Event:
        def record(self, stream=None):
            pass

    class Rows:  # an out-of-place all-reduce result
        def record_stream(self, stream):
            pass  # what the caching allocator does for a tensor it did not allocate

        def __getitem__(self, j):
            return self

    def reduce_owned_rows(looked):
        out = Rows()
        refs.append(weakref.ref(out))
        return out

    main = Stream()
    monkeypatch.setattr(prefetch.torch.cuda, "current_stream", lambda: main)
    monkeypatch.setattr(
        prefetch.torch.cuda, "stream", lambda s: contextlib.nullcontext()
    )
    engram = SimpleNamespace(
        embed=SimpleNamespace(_reduce_owned_rows=reduce_owned_rows)
    )
    layer = SimpleNamespace(
        engram=engram, n=4, cols=slice(0, 4), quarter=lambda rows: torch.ones(2, 4)
    )
    p = object.__new__(prefetch.EngramPrefetch)
    p.layers, p.stream, p.rows, p.hash_cols, p.dim = {3: layer}, Stream(), 2, 1, 4
    p.looked, p.parts = torch.zeros(8), torch.zeros(8)
    p.looked_up, p.reduced, p.wkv_done = Event(), Event(), Event()
    p.group, p.kv, p._emb = SimpleNamespace(all_reduce=lambda x: x), None, None

    p.reduce()
    gc.collect()
    assert refs[0]() is not None  # held past reduce(), while the side stream reads it
    p.gather()
    gc.collect()
    # still held when main waits for the side stream's WKV; released after
    assert (p.wkv_done, True) in waits and refs[0]() is None


def _batch(mode="TARGET_VERIFY"):
    is_verify = mode == "TARGET_VERIFY"
    return SimpleNamespace(
        forward_mode=SimpleNamespace(name=mode, is_target_verify=lambda: is_verify)
    )


def _capturing(monkeypatch, on=True):
    import sglang.srt.model_executor.runner as runner

    monkeypatch.setattr(runner, "get_is_capture_mode", lambda: on)


def test_only_verify_steps_of_built_widths_are_covered(monkeypatch):
    _capturing(monkeypatch)
    decode = executor.MegakernelDecode(None, {}, {}, (6, 12, 48))
    ids = torch.zeros(12, dtype=torch.int64)
    assert decode.plan(ids, _batch()) == Step(mode="target_verify", rows=12)
    assert decode.plan(torch.zeros(18, dtype=torch.int64), _batch()) is None
    assert decode.plan(ids, _batch("DECODE")) is None
    assert decode.plan(ids, _batch(), None, pp_proxy_tensors=object()) is None


@pytest.mark.parametrize("which", ["target", "draft"])
@pytest.mark.parametrize(
    "capturing, switch, served",
    [
        (False, None, False),
        (True, None, True),
        (False, ("SGLANG_NVIDIA_MEGAKERNEL_PARITY", "1"), True),
        (False, ("SGLANG_NVIDIA_MEGAKERNEL_FAULT_INJECT", "1:hop_a:5000"), True),
    ],
)
def test_only_captured_steps_are_served(monkeypatch, which, capturing, switch, served):
    """Under FULL decode CUDA graphs SGLang may still step eagerly (an uncaptured
    batch); such a step runs SGLang's forward, and only the graph capture (its
    warm-up runs included) or a test-only switch runs the covered step."""
    from sgl_nvidia_mega.deepseek_v41 import draft_step

    _capturing(monkeypatch, capturing)
    for name in spec.EAGER_TEST_SWITCHES:
        monkeypatch.delenv(name, raising=False)
    if switch:
        monkeypatch.setenv(*switch)
    if which == "target":
        built = executor.MegakernelDecode(None, {}, {}, (6, 12))
    else:
        built = draft_step.DraftDecode.__new__(draft_step.DraftDecode)
        built.widths = (6, 12)
    step = built.plan(torch.zeros(12, dtype=torch.int64), _batch())
    assert step == (Step(mode="target_verify", rows=12) if served else None)


def test_a_two_shot_width_is_a_covered_step(monkeypatch):
    """With the two-shot slot built, a 384-row verify step (64 requests) runs the
    executor, whose boundary then takes the two-shot posts at that width."""
    from sgl_nvidia_mega.deepseek_v41 import plan as slot_plan

    _capturing(monkeypatch)

    widths = tuple(range(6, slot_plan.max_rows(True) + 1, 6))
    decode = executor.MegakernelDecode(None, {}, {}, widths)
    ids = torch.zeros(384, dtype=torch.int64)
    assert decode.plan(ids, _batch()) == Step(mode="target_verify", rows=384)
    assert slot_plan.slots(384, slot_plan.Facts(two_shot=True)).two_shot
    narrow = executor.MegakernelDecode(None, {}, {}, tuple(range(6, 49, 6)))
    assert narrow.plan(ids, _batch()) is None


def test_slot_selection_is_validated(monkeypatch):
    from sgl_nvidia_mega.deepseek_v41 import runtime

    monkeypatch.delenv(runtime.SLOTS_ENV, raising=False)
    assert runtime._requested_slots() == frozenset(runtime.SHIPPED_SLOTS)
    monkeypatch.setenv(runtime.SLOTS_ENV, "")
    assert runtime._requested_slots() == frozenset()
    monkeypatch.setenv(runtime.SLOTS_ENV, " moe , gate ")
    assert runtime._requested_slots() == {"moe", "gate"}
    for raw, want in (
        ("moe,gatee", "unknown slot 'gatee'"),
        ("gate", "needs slot 'moe'"),
        ("draft_lm_head", "needs slot 'markov'"),
        ("moe,hop_b", "needs slot 'hop_a'"),
        ("early_trigger", "needs slot 'hop_a'"),
        ("hop_a_batch2", "needs slot 'hop_a'"),
        ("hop_a,hop_b_start_trigger", "needs slot 'hop_b'"),
        ("attn_indexer", "needs slot 'attn'"),
        ("shared_ll", "needs slot 'moe'"),
    ):
        monkeypatch.setattr(runtime, "SLOTS", frozenset(raw.split(",")))
        assert any(want in r for r in runtime.slot_reasons()), raw
    monkeypatch.setattr(runtime, "SLOTS", frozenset(runtime.SHIPPED_SLOTS))
    assert runtime.slot_reasons() == []
    opt_in = frozenset(runtime.SHIPPED_SLOTS) | set(runtime.OPT_IN_SLOTS)
    monkeypatch.setattr(runtime, "SLOTS", opt_in)
    assert runtime.slot_reasons() == []


def test_the_engram_prefetch_is_built_only_when_named(monkeypatch):
    """The Engram prefetch builds only when the requested slots name it (the default
    does), never from SGLang's host-table settings alone."""
    from sgl_nvidia_mega.deepseek_v41 import runtime

    monkeypatch.delenv(runtime.SLOTS_ENV, raising=False)
    assert "engram" in runtime._requested_slots()
    monkeypatch.setenv(runtime.SLOTS_ENV, "moe")
    assert "engram" not in runtime._requested_slots()
    monkeypatch.setenv(runtime.SLOTS_ENV, "moe,engram")
    monkeypatch.setattr(runtime, "SLOTS", runtime._requested_slots())
    assert "engram" in runtime.SLOTS and runtime.slot_reasons() == []


def test_the_engram_prefetch_collectives_run_at_their_layers_only():
    """The reduce runs before the MoE of the layer before the first Engram layer and
    the gather before the first Engram layer, both on the main stream, only with
    the prefetch on for the step."""
    from sgl_nvidia_mega.deepseek_v41 import engram, plan

    ran = []
    prefetch = SimpleNamespace(
        reduce_at=0,
        gather_at=1,
        reduce=lambda: ran.append("reduce"),
        gather=lambda: ran.append("gather"),
    )
    st = _state(engram_layers=(1, 2))
    st.bound = {"engram": SimpleNamespace(prefetch=prefetch)}
    st.hash_ids = torch.zeros(6, 2, 24, dtype=torch.int64)
    st.slots = plan.slots(6, plan.Facts(engram_prefetch=True))
    for i in range(3):
        engram.before_layer(i, st)
        ran.append(f"layer {i}")
        engram.before_moe(i, st)
    assert ran == ["layer 0", "reduce", "gather", "layer 1", "layer 2"]
    ran.clear()
    for slots, hash_ids in (
        (plan.slots(18, plan.Facts(engram_prefetch=True)), st.hash_ids),
        (st.slots, None),
    ):
        st.slots, st.hash_ids = slots, hash_ids
        for i in range(3):
            engram.before_layer(i, st)
            engram.before_moe(i, st)
    assert ran == []


def test_the_engram_prefetch_needs_a_layer_before_the_first_engram_layer():
    from sgl_nvidia_mega.deepseek_v41.engram import prefetch as ep

    layers = [
        SimpleNamespace(engram=object() if i in (0, 2) else None) for i in range(3)
    ]
    model = SimpleNamespace(
        model=SimpleNamespace(start_layer=0, end_layer=3, layers=layers)
    )
    prefetch, why = ep.build(model, SimpleNamespace(widths=(5, 10)), group=None)
    assert prefetch is None and "layer 0 is the first layer" in why


# ------------------------------------------------------------------------------------
# SGLang's binding (BuildContext.binding) reaches every reader on the real build and
# forward paths: each package's build context, the executor and its per-step state, and
# the draft's built object. A package handed none builds at no width (10-18 boot crash).


class _Stop(Exception):
    pass


def test_runtime_build_hands_every_package_the_binding(monkeypatch):
    """runtime.build gives each package of PACKAGES the same PackageContext, and it
    carries the BuildContext's binding."""
    seen = {}

    def recorder(name, last):
        def build(model, ctx, group):
            seen[name] = ctx
            if last:
                raise _Stop  # the rest of the build needs real packages
            return None

        return SimpleNamespace(build=build)

    names = list(executor.PACKAGES)
    monkeypatch.setattr(
        executor,
        "PACKAGES",
        {n: recorder(n, i == len(names) - 1) for i, n in enumerate(names)},
    )
    monkeypatch.setattr(
        "sglang.srt.runtime_context.get_parallel",
        lambda: SimpleNamespace(tp_group=None),
    )
    from sglang.srt.megakernels.binding import MegakernelBinding

    model = SimpleNamespace()
    binding = MegakernelBinding(model=model)  # SGLang's own, as install builds it
    ctx = SimpleNamespace(
        widths=(6,), tokens_per_request=6, deadline_s=1.0, binding=binding
    )
    with pytest.raises(_Stop):
        runtime.build(model, ctx)
    assert sorted(seen) == sorted(names)
    assert all(c.binding is binding for c in seen.values()), seen


def test_the_executor_hands_each_step_its_view_and_the_binding(monkeypatch):
    """A covered forward's StepState carries Step.view and the build's binding."""
    binding = SimpleNamespace(forward_view=lambda: "from the binding")
    decode = executor.MegakernelDecode(
        SimpleNamespace(prepare_language_model_inputs=lambda ids, fb, emb: (ids, emb)),
        slots={6: "slots-6"},
        bound={},
        widths=(6,),
        binding=binding,
    )
    assert decode.binding is binding
    made = []

    def record(**kw):
        made.append(kw)
        raise _Stop

    monkeypatch.setattr(executor, "StepState", record)
    step = executor.Step(mode="target_verify", rows=6, view="the step's view")
    with (
        mock.patch("sglang.srt.layers.layer_boundary.get_attn_tp_context"),
        pytest.raises(_Stop),
    ):
        decode._forward(step, "ids", "pos", SimpleNamespace(), None)
    assert made[0]["view"] == "the step's view" and made[0]["binding"] is binding


@pytest.mark.parametrize("view, want", [("the step's view", "the step's view"),
                                        (None, "from the binding")])  # fmt: skip
def test_a_step_without_a_view_asks_the_binding(view, want):
    st = state.StepState.__new__(state.StepState)
    st.view = view
    st.binding = SimpleNamespace(forward_view=lambda: "from the binding")
    assert st.forward_view() == want


def test_the_draft_build_keeps_the_binding():
    """The draft spec's build gets its own BuildContext.binding; the draft forward's
    ported attention reads its view from it."""
    from sgl_nvidia_mega.deepseek_v41 import draft, draft_step

    binding = object()
    with (
        mock.patch.object(draft_step, "DraftDecode") as decode,
        mock.patch.object(runtime, "SLOTS", frozenset({"draft"})),
    ):
        draft._build("model", None, SimpleNamespace(deadline_s=1.0, binding=binding))
    assert decode.call_args.kwargs["binding"] is binding


# ------------------------------------------------------------------------------------
# Every slot that reads model or runner state after load, in SGLang's real init order
# (scheduler_init.py): no read before the stage that sets it, and the slot engages.

# FU2's draft forward slots (draft_embed apart: v5 drops it; its scenario adds it back)
DRAFT_SLOTS = frozenset(
    {
        "markov",
        "draft_lm_head",
        "draft",
        "draft_attn",
        "draft_moe",
        "draft_gate",
        "draft_tail",
        "draft_post",
        "draft_seam",
        "draft_qkv_a",
        "draft_qfront",
        "draft_wo_a",
        "draft_swapab",
        "draft_prenorm",
        "draft_qkv_a_ll",
        "draft_wo_b_ll",
    }
)


def _binding(init: SchedulerInit) -> SimpleNamespace:
    """SGLang's MegakernelBinding, built at install: it reads nothing until asked; its
    forward view reads the running forward's attention backend (here the draft runner's)."""

    def forward_view():
        backend = init.draft_runner.attn_backend
        return SimpleNamespace(kv_pool=SimpleNamespace(pool=backend.token_to_kv_pool))

    return SimpleNamespace(forward_view=forward_view)


def _draft_decode_at_install(init: SchedulerInit) -> None:
    """install: with slot draft the draft spec builds DraftDecode when the draft model
    loads, before attach_shared_modules. Its stage builders only see the stages (they
    compile kernels on GPU), so they are recorded here instead of run; so is
    draft_prenorm's hardware check."""
    from sgl_nvidia_mega.deepseek_v41 import draft, draft_step, runtime

    built = []

    def record(name):
        return lambda stages, *a, **k: (
            built.append(name)
            or SimpleNamespace(slots={}, reason="", summary=lambda: "")
        )

    with (
        mock.patch.object(runtime, "SLOTS", init.slots),
        mock.patch.object(draft_step.moe.routed, "build_layers", record("moe")),
        mock.patch.object(draft_step.boundary, "Boundaries", record("boundary")),
        mock.patch.object(draft_step.draft_attention, "build", record("attention")),
        mock.patch("torch.cuda.is_available", return_value=False),
        mock.patch(
            "sgl_nvidia_mega.deepseek_v41.boundary.prenorm.reason", return_value=""
        ),
    ):
        init.state.draft_built = draft._build(
            init.draft_model,
            None,
            SimpleNamespace(deadline_s=1.0, binding=_binding(init)),
        )
    assert isinstance(init.state.draft_built, draft_step.DraftDecode)
    assert sorted(built) == ["attention", "boundary", "moe"]


def _draft_decode_engages(init: SchedulerInit) -> None:
    """draft_graphs: the built widths are gamma rows a request, and draft_embed's table
    (deferred at install) binds to the attached embedding only now."""
    from sgl_nvidia_mega.deepseek_v41 import draft_step

    built = init.state.draft_built
    assert built.widths[0] == init.gamma
    # the draft forward's view resolves only once the draft has its attention backend
    assert built._binding.forward_view().kv_pool.pool is init.draft_pool
    if "draft_embed" in init.slots:
        assert built._embed_table is draft_step._SHARED_TABLE
        assert init.draft_model.embed_tokens is not None


DRAFT_DECODE = Scenario(
    "draft_decode",
    DRAFT_SLOTS,
    (
        ("install", _draft_decode_at_install),
        ("draft_graphs", _draft_decode_engages),
        # the first decode after the scheduler's init: the same reads again
        ("overlap", _draft_decode_engages),
    ),
)
DRAFT_DECODE_EMBED = DRAFT_DECODE._replace(
    name="draft_decode_embed", slots=DRAFT_SLOTS | {"draft_embed", "embed"}
)


SCENARIOS = [DRAFT_DECODE, DRAFT_DECODE_EMBED]


@pytest.mark.parametrize("scenario", SCENARIOS, ids=lambda s: s.name)
def test_slot_reads_nothing_before_sglang_sets_it(scenario):
    SchedulerInit().run(scenario)


# The fixture itself: an early read fails however the slot reads it.


def test_attributes_appear_at_their_stage():
    init = SchedulerInit()
    init.advance("draft_worker")
    assert init.draft_model.lm_head is not None
    with pytest.raises(EarlyRead, match="token_to_kv_pool.*'memory_pools'"):
        init.draft_runner.token_to_kv_pool
    init.advance("memory_pools")
    assert init.draft_runner.token_to_kv_pool.swa_req_ring_size is None
    with pytest.raises(AttributeError):
        init.draft_runner.no_such_attribute
    # set at install: the runner's config and the target's head, shared with the draft
    assert init.target_runner.model_config.vocab_size > 0
    assert init.draft_model.lm_head is init.target_model.lm_head
    with pytest.raises(EarlyRead, match="attn_backend.*'attention_backends'"):
        init.draft_runner.attn_backend
    init.advance("attention_backends")
    assert init.draft_runner.attn_backend.token_to_kv_pool is init.draft_pool
    with pytest.raises(EarlyRead, match="decode_cuda_graph_runner"):
        init.draft_runner.decode_cuda_graph_runner


def test_install_facts_exist_before_any_stage():
    """The draft model's own load-time attributes and SGLang's runtime context (the
    parallel groups, spec args, graph config) exist at install."""
    import sglang.srt.runtime_context as context

    init = SchedulerInit()
    assert init.draft_model.sample_from_anchor and len(init.draft_model.stages) == 2
    # read through the module: plugin code imports these inside its functions
    with init.runtime_context():
        assert context.get_parallel().tp_group.world_size == 4
        assert context.get_spec().speculative_num_draft_tokens == init.gamma + 1
        assert context.get_exec().graph.cuda_graph_config.decode.bs == list(
            init.capture_bs
        )


def test_draft_embed_pattern_fails_at_install():
    """draft_embed read the draft's embedding at install, before attach_shared_modules."""
    init = SchedulerInit()
    with pytest.raises(EarlyRead, match="embed_tokens"):
        init.draft_model.embed_tokens


def test_draft_decode_catches_the_draft_embed_bug():
    """The 10-18 02:49 crash: draft_embed's table read from the draft's embedding when
    DraftDecode built, at install. The draft forward's scenario must fail on it."""
    from sgl_nvidia_mega.deepseek_v41 import draft_step

    with mock.patch.object(
        draft_step, "_embed_table", lambda lm, slots: lm.embed_tokens.weight
    ):
        with pytest.raises(EarlyRead, match="embed_tokens.*'install'"):
            SchedulerInit().run(DRAFT_DECODE_EMBED)


def test_mid_fold_pattern_fails_through_getattr_default():
    """MID_FOLD's first version went inert: getattr(runner, name, None) at the phases'
    init returned None before the target graphs existed. EarlyRead is no
    AttributeError, so neither getattr nor hasattr hides it."""
    init = SchedulerInit()
    init.advance("draft_worker")
    with pytest.raises(EarlyRead, match="decode_cuda_graph_runner"):
        getattr(init.worker.model_runner, "decode_cuda_graph_runner", None)
    with pytest.raises(EarlyRead):
        hasattr(init.worker.model_runner, "decode_cuda_graph_runner")


def test_engram_history_pattern_fails_before_the_memory_pools():
    """P1's first version read the Engram history at the phases' init."""
    init = SchedulerInit()
    init.advance("draft_worker")
    with pytest.raises(EarlyRead, match="ngram_embedding_manager.*'memory_pools'"):
        init.worker.model_runner.ngram_embedding_manager.engram_hasher.history
    init.advance("memory_pools")
    assert init.worker.model_runner.ngram_embedding_manager is not None


def test_stages_only_move_forward():
    init = SchedulerInit()
    init.advance("target_graphs")
    with pytest.raises(AssertionError):
        init.advance("draft_worker")
    assert init.stage == "target_graphs" and STAGES[-1] == "overlap"


# ------------------------------------------------------------------------------------
# The Engram prefetch takes each layer's host table from SGLang's binding, never from
# the layer itself, and stays off on an SGLang that offers no binding.


def _model():
    engram = SimpleNamespace(layer_hash_index=0)
    layers = [SimpleNamespace(engram=None), SimpleNamespace(engram=engram)]
    return SimpleNamespace(
        model=SimpleNamespace(layers=layers, start_layer=0, end_layer=2)
    )


def _ctx(binding=None):
    """The context the runtime hands each package (not SGLang's BuildContext)."""
    from sgl_nvidia_mega.deepseek_v41.runtime import PackageContext

    return PackageContext(
        widths=(6,), tokens_per_request=6, deadline_s=1.0, copies={}, binding=binding
    )


def test_without_a_binding_the_prefetch_stays_off():
    built, why = prefetch.build(_model(), _ctx(), group=None)
    assert built is None and "binding" in why


def test_the_host_table_comes_from_the_binding(monkeypatch):
    """SGLang's own MegakernelBinding hands the layer's host table (a shared one here,
    which the prefetch refuses)."""
    from sglang.srt.megakernels.binding import MegakernelBinding

    model = _model()
    engram = model.model.layers[1].engram
    engram.embed = SimpleNamespace(host_table=SimpleNamespace(layout="shared"))
    binding = MegakernelBinding(model=model)
    asked = []
    real = binding.engram_host_table
    monkeypatch.setattr(
        binding, "engram_host_table", lambda e: asked.append(e) or real(e)
    )
    ctx = _ctx(binding)
    # one rank: the gathered error is this rank's own
    monkeypatch.setattr(prefetch, "_any_error", lambda group, error: error)
    built, why = prefetch.build(model, ctx, SimpleNamespace(world_size=4))
    assert asked == [model.model.layers[1].engram]
    assert built is None and "per-rank host Engram table" in why
