# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""The Qwen3.5 megakernel's step packages on CPU, against SGLang's own forward on
stubbed modules (the same ops in the same order, on the same streams, with the same
arguments): the decoder layers, the GatedDeltaNet, the attention, the MoE, the layer
loop, and the slots route, mkm, gdn_verify and hop."""

import contextlib
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
from sglang.kernels.plugin_kernels.nvidia import spec as registry
from sglang.kernels.plugin_kernels.nvidia.spec import Step
from sglang.srt.models import qwen2_moe, qwen3_5

from sgl_nvidia_mega import plugin
from sgl_nvidia_mega.qwen3_5 import (
    attention,
    gdn,
    moe,
    plan,
    runtime,
)
from sgl_nvidia_mega.qwen3_5.boundary import hop as boundary_hop
from sgl_nvidia_mega.qwen3_5.runtime import executor


@pytest.fixture(autouse=True)
def registered(monkeypatch):
    monkeypatch.setattr(registry, "_REGISTRY", {})
    monkeypatch.delenv(plugin.PARITY_SORT_ENV, raising=False)
    plugin.register()


# ------------------------------------------------- the stock step against SGLang's


def _verify(widths=(7,), binding="binding"):
    """An executor over a model with no layers (plan reads only host state), built
    with SGLang's EAGLE binding."""
    lm = SimpleNamespace(model=SimpleNamespace(start_layer=0, end_layer=0, layers=[]))
    return executor.Qwen35Verify(lm, {}, {}, widths, binding=binding)


class _Log(list):
    def tag(self, value):
        """A short stand-in for a value: a known tensor's name and the shape it is
        viewed in, a known object's name, or the value itself."""
        if isinstance(value, torch.Tensor):
            for name, known in self.known.items():
                if (
                    isinstance(known, torch.Tensor)
                    and value.data_ptr() == known.data_ptr()
                ):
                    return f"{name}{tuple(value.shape)}"
            return f"?{tuple(value.shape)}"
        for name, known in self.known.items():
            if value is known:
                return name
        return value if isinstance(value, (str, bool, int, type(None))) else "?"

    def op(self, name, out=None):
        def run(*args, **kwargs):
            kw = tuple(sorted((k, self.tag(v)) for k, v in kwargs.items()))
            self.append((name, self.stream, tuple(map(self.tag, args)), kw))
            return out

        return run


def _log(**known):
    log = _Log()
    log.known, log.stream = dict(known), "main"
    return log


class _Stream:
    def __init__(self, name, log):
        self.name, self.log = name, log

    def wait_stream(self, other):
        self.log.append(("wait", self.name, other.name))


@contextlib.contextmanager
def _streams(log, capture):
    main = _Stream("main", log)

    @contextlib.contextmanager
    def on(stream):
        prev, log.stream = log.stream, stream.name
        try:
            yield
        finally:
            log.stream = prev

    with (
        patch.object(torch.cuda, "current_stream", return_value=main),
        patch.object(torch.cuda, "stream", side_effect=on),
        patch.object(qwen2_moe, "get_is_capture_mode", return_value=capture),
        patch(
            "sglang.srt.model_executor.runner.get_is_capture_mode", return_value=capture
        ),
    ):
        yield


class _Callable:
    def __init__(self, fn, **attrs):
        self._fn = fn
        self.__dict__.update(attrs)

    def __call__(self, *args, **kwargs):
        return self._fn(*args, **kwargs)


def _moe_block(log, x, *, alt_stream=True, shared=True, deferred_ok=True):
    """A Qwen2MoeSparseMoeBlock whose gate, top-k, experts and shared expert log."""
    mlp = qwen2_moe.Qwen2MoeSparseMoeBlock.__new__(qwen2_moe.Qwen2MoeSparseMoeBlock)
    torch.nn.Module.__init__(mlp)
    bypassed = SimpleNamespace(format="bypassed")
    routed = torch.zeros(7, 4)
    log.known["routed"] = routed
    experts = _Callable(
        log.op("experts", routed),
        supports_deferred_finalize=deferred_ok,
        forward_deferred_finalize=log.op("experts_deferred", "routed_d"),
    )
    mlp.__dict__.update(
        alt_stream=_Stream("alt", log) if alt_stream else None,
        reduce_results=False,
        enable_shared_expert_fusion=False,
        _use_mega_moe=False,
        layer_id=3,
        gate=_Callable(log.op("gate", ("logits", None))),
        topk=_Callable(log.op("topk", bypassed)),
        experts=experts,
        shared_expert=_Callable(log.op("shared", "shared_out")) if shared else None,
        shared_expert_gate=_Callable(
            log.op("shared_gate", "gate_out"), weight=torch.ones(1, 4)
        ),
    )
    return mlp


@contextlib.contextmanager
def _moe_kernels(log):
    with (
        patch.object(
            qwen2_moe, "fused_gate_sigmoid_mul", log.op("gate_mul", "gated_shared")
        ),
        patch.object(qwen2_moe, "fused_gate_sigmoid_mul_add", log.op("gate_mul_add")),
        patch(
            "sglang.kernels.ops.elementwise.elementwise.fused_gate_sigmoid_mul_add",
            log.op("gate_mul_add"),
        ),
        patch.object(qwen2_moe, "is_npu", return_value=False),
        # SGLang pages experts on one GPU only, never under the spec's TP8
        patch.object(
            qwen2_moe,
            "get_exec",
            return_value=SimpleNamespace(
                moe=SimpleNamespace(enable_paged_experts=False)
            ),
        ),
        patch.object(qwen2_moe, "use_intel_amx_backend", return_value=False),
        patch.object(qwen2_moe, "_use_aiter", False),
        patch.object(qwen2_moe.TopKOutputChecker, "format_is_bypassed", lambda o: True),
        patch.object(
            qwen2_moe.TopKOutputChecker, "format_is_standard", lambda o: False
        ),
        patch(
            "sglang.srt.layers.layer_boundary.fusions.cutedsl.MoeDeferredFinalize."
            "from_flashinfer",
            log.op("deferred_finalize", "handoff"),
        ),
        patch.object(
            qwen2_moe, "get_moe_a2a_backend", return_value=_Callable(None, **_NO_A2A)
        ),
    ):
        yield


_NO_A2A = {
    name: (lambda: False)
    for name in ("is_deepep", "is_deepep_v2", "is_mori", "is_ascend_fuseep")
}


@pytest.mark.parametrize("capture", [False, True])
@pytest.mark.parametrize("defer", [False, True])
@pytest.mark.parametrize("alt_stream", [False, True])
def test_the_moe_step_issues_what_sglang_issues(capture, defer, alt_stream):
    runs = []
    for which in ("sglang", "plugin"):
        x = torch.zeros(7, 4)
        log = _log(x=x)
        mlp = _moe_block(log, x, alt_stream=alt_stream)
        with _streams(log, capture), _moe_kernels(log):
            if which == "sglang":
                out = mlp.forward(x, None, defer_finalize=defer)
            else:
                out = moe.step(SimpleNamespace(mlp=mlp), x, _st(), defer=defer)
        runs.append((log, log.tag(out) if not isinstance(out, torch.Tensor) else out))
    (want_log, want), (got_log, got) = runs
    assert got_log == want_log
    assert want == got if not isinstance(want, torch.Tensor) else torch.equal(want, got)


def _st(route=False):
    return SimpleNamespace(slots=plan.Slots(route=route))


def _router_front(log, mlp):
    """The block's experts as the route slot reads them (the dispatcher, the runner
    core, the input scale), and its router front's outputs."""
    from sgl_nvidia_mega.qwen3_5.sm103.moe import router

    class _Dispatched:
        def _replace(self, **kwargs):
            return log.op("replace", "dispatched_fp4")(**kwargs)

    log.known.update(
        gate_w=torch.zeros(router.EXPERTS, router.HIDDEN, dtype=torch.bfloat16),
        scale=torch.ones(()),
        logits=torch.zeros(7, router.EXPERTS, dtype=torch.bfloat16),
        x_fp4=torch.zeros(7, router.HIDDEN // 2, dtype=torch.uint8),
        x_sf=torch.zeros(7, router.HIDDEN // 16, dtype=torch.uint8),
    )
    k = log.known
    mlp.gate.weight = k["gate_w"]
    mlp.experts.__dict__.update(
        w13_input_scale_quant=k["scale"],
        dispatcher=SimpleNamespace(
            dispatch=log.op("dispatch", _Dispatched()),
            combine=log.op("combine", "routed_d"),
        ),
        run_moe_core=log.op("run_moe_core", "combine_input"),
    )

    @contextlib.contextmanager
    def deferred():
        log.append(("deferred_context", log.stream))
        yield

    def front(x, w, scale):
        log.op("router_front")(x, w, scale)
        assert scale.shape == (1,) and scale.data_ptr() == k["scale"].data_ptr()
        return k["logits"], k["x_fp4"], k["x_sf"]

    return (
        contextlib.ExitStack(),
        patch.object(router, "router_gemm_quant", front),
        patch(
            "sglang.srt.layers.moe.moe_runner.flashinfer_trtllm."
            "flashinfer_trtllm_deferred_finalize_context",
            deferred,
        ),
    )


@pytest.mark.parametrize("capture", [False, True])
def test_the_route_slot_replaces_only_the_gate_and_the_routed_experts(capture):
    """Slot route: the router front in place of the gate, then FusedMoE's
    deferred-finalize path on its quantized input; every other op is the stock one."""
    runs = []
    for route in (False, True):
        x = torch.zeros(7, 4)
        log = _log(x=x)
        mlp = _moe_block(log, x)
        _, front, context = _router_front(log, mlp)
        with _streams(log, capture), _moe_kernels(log), front, context:
            out = moe.step(SimpleNamespace(mlp=mlp), x, _st(route), defer=True)
        runs.append((log, log.tag(out)))
    (stock, stock_out), (got, got_out) = runs
    start = [e[0] for e in stock].index("gate")
    assert [e[0] for e in stock[start : start + 3]] == [
        "gate",
        "topk",
        "experts_deferred",
    ]
    on_front = [
        ("router_front", "main", ("x(7, 4)", "gate_w(512, 4096)", "scale(1,)"), ()),
        ("topk", "main", ("x(7, 4)", "logits(7, 512)"), ()),
        ("dispatch", "main", (), (("hidden_states", "x(7, 4)"), ("topk_output", "?"))),
        (
            "replace",
            "main",
            (),
            (
                ("hidden_states", "x_fp4(7, 2048)"),
                ("hidden_states_scale", "x_sf(7, 256)"),
            ),
        ),
        ("deferred_context", "main"),
        ("run_moe_core", "main", (), (("dispatch_output", "dispatched_fp4"),)),
        ("combine", "main", (), (("combine_input", "combine_input"),)),
    ]
    assert got == stock[:start] + on_front + stock[start + 3 :]
    assert got_out == stock_out == "handoff"


class _GateUp:
    """The shared gate/up rows the router front returns, logging the stream they are
    recorded on."""

    def __init__(self, log):
        self.log = log

    def record_stream(self, stream):
        self.log.append(("record_stream", stream.name))


def test_the_route_shared_slot_starts_the_shared_expert_after_the_router_launch():
    """Slot route_shared at graph capture with the deferred finalize: the router launch
    also computes the shared gate/up; the alternate stream waits for that launch only (an
    event recorded right after it), then runs the activation, down_proj and the gate,
    while the routed experts run on the main stream."""
    from sgl_nvidia_mega.qwen3_5.sm103.moe import router

    x = torch.zeros(7, 4)
    log = _log(x=x)
    mlp = _moe_block(log, x)
    _, front, context = _router_front(log, mlp)
    k = log.known
    k.update(
        # ModelOpt keeps the weight as the [K, N] transpose of a row-major [N, K]
        sh_w=torch.zeros(256, 4096, dtype=torch.float8_e4m3fn).t(),
        sh_in=torch.ones(()),
        sh_ws=torch.ones(()),
    )
    gate_up = _GateUp(log)
    k["gate_up"] = gate_up
    mlp.shared_expert = SimpleNamespace(
        gate_up_proj=SimpleNamespace(
            weight=k["sh_w"], input_scale=k["sh_in"], weight_scale=k["sh_ws"]
        ),
        _enable_silu_fp4_quant_fusion=False,
        act_fn=log.op("act", "act_out"),
        down_proj=log.op("down_proj", ("down_out", None)),
    )
    k["act_out"] = "act_out"

    def front_shared(x_, w, scale, sh_w, sh_in, sh_ws):
        log.op("router_front_shared")(x_, w, scale, sh_w, sh_in, sh_ws)
        assert sh_w.shape == (256, 4096) and sh_w.is_contiguous()
        assert sh_in.shape == sh_ws.shape == (1,)
        return k["logits"], k["x_fp4"], k["x_sf"], gate_up

    class _Event:
        def record(self):
            log.append(("event_record", log.stream))

    def wait_event(self, event):
        self.log.append(("wait_event", self.name))

    with (
        _streams(log, True),
        _moe_kernels(log),
        front,
        context,
        patch.object(router, "router_gemm_quant_shared", front_shared),
        patch.object(torch.cuda, "Event", _Event),
        patch.object(_Stream, "wait_event", wait_event, create=True),
    ):
        st = SimpleNamespace(slots=plan.Slots(route=True, route_shared=True))
        out = moe.step(SimpleNamespace(mlp=mlp), x, st, defer=True)
    assert log.tag(out) == "handoff"
    ops = [(e[0], e[1]) for e in log]
    assert ops == [
        ("wait", "alt"),
        ("router_front_shared", "main"),
        ("event_record", "main"),
        ("topk", "main"),
        ("dispatch", "main"),
        ("replace", "main"),
        ("deferred_context", "main"),
        ("run_moe_core", "main"),
        ("combine", "main"),
        ("record_stream", "alt"),
        ("wait_event", "alt"),
        ("act", "alt"),
        ("down_proj", "alt"),
        ("gate_mul", "alt"),
        ("wait", "main"),
        ("deferred_finalize", "main"),
    ]
    assert log[1][2] == (
        "x(7, 4)",
        "gate_w(512, 4096)",
        "scale(1,)",
        "sh_w(256, 4096)",
        "sh_in(1,)",
        "sh_ws(1,)",
    )
    assert log[11][2] == ("gate_up",) and log[12][2] == ("act_out",)


def test_the_router_fronts_shared_stage_fits_shared_memory():
    from sgl_nvidia_mega.qwen3_5.sm103.moe import router

    # GemvSmem<max_m, true>::kBytes: + 8 and max_m e4m3 rows of 4112 B, 2 KiB of
    # partials and a third mbarrier
    assert router.shared_bytes(8) == 131328 + 1024 + 16
    assert router.shared_bytes(8, shared=True) == 200216
    assert router.shared_bytes(1, shared=True) == 113976


@pytest.mark.parametrize(
    "change, want",
    [
        ({}, ""),
        ({"act_fn": object()}, "shared expert activation object"),
        ({"bias": torch.zeros(256)}, "not a static per-tensor FP8 linear"),
        ({"rows": 1024}, "not a static per-tensor FP8 linear"),
    ],
)
def test_the_route_shared_slot_names_what_a_layer_lacks(change, want):
    from sglang.srt.layers.activation import SiluAndMul
    from sglang.srt.layers.quantization.modelopt_quant import ModelOptFp8LinearMethod

    from sgl_nvidia_mega.qwen3_5.moe import route

    method = object.__new__(ModelOptFp8LinearMethod)
    method.use_marlin = method.use_sm120_fp8 = False
    expert = qwen2_moe.Qwen2MoeMLP.__new__(qwen2_moe.Qwen2MoeMLP)
    torch.nn.Module.__init__(expert)
    rows = change.get("rows", 256)
    expert.__dict__.update(
        gate_up_proj=SimpleNamespace(
            quant_method=method,
            bias=change.get("bias"),
            weight=torch.zeros(rows, 4096, dtype=torch.float8_e4m3fn).t(),
            input_scale=torch.ones(1),
            weight_scale=torch.ones(1),
        ),
        act_fn=change.get("act_fn", object.__new__(SiluAndMul)),
    )
    mlp = SimpleNamespace(shared_expert=expert, shared_expert_gate="gate")
    got = route.shared_unsupported(mlp)
    assert (want in got) if want else got == ""


@pytest.mark.parametrize("capture", [False, True])
def test_the_mkm_slot_runs_the_routed_experts_on_the_router_fronts_outputs(capture):
    """Slot mkm: the router front, then one fused launch whose rows and weights reach
    the deferred finalize in expert order; every other op is the stock one."""
    from sgl_nvidia_mega.qwen3_5.sm103.moe import mkm as kernel

    runs = []
    for slots in (plan.Slots(), plan.Slots(route=True, mkm=True)):
        x = torch.zeros(7, 4)
        log = _log(x=x)
        mlp = _moe_block(log, x)
        _, front, context = _router_front(log, mlp)
        k = log.known
        k.update(
            w13_sf=torch.zeros(4, dtype=torch.float8_e4m3fn),
            w2_sf=torch.zeros(8, dtype=torch.float8_e4m3fn),
            g1=torch.ones(512),
            c=torch.ones(512),
            g2=torch.ones(512),
            rows=torch.zeros(70, 4, dtype=torch.bfloat16),
            weights=torch.zeros(7, 10, dtype=torch.bfloat16),
        )
        mlp.experts.__dict__.update(
            w13_weight_scale=k["w13_sf"],
            w2_weight_scale=k["w2_sf"],
            g1_alphas=k["g1"],
            g1_scale_c=k["c"],
            g2_alphas=k["g2"],
        )
        combined = []

        def combine(combine_input, log=log, combined=combined):
            log.append(("combine", log.stream))
            combined.append(combine_input)
            return "routed_d"

        mlp.experts.dispatcher.combine = combine
        bound = SimpleNamespace(
            mkm={3: "maps"}, identity=torch.arange(70, dtype=torch.int32)
        )
        st = SimpleNamespace(slots=slots, bound={"moe": bound})
        fused = patch.object(
            kernel, "mkm_moe", log.op("mkm", (k["rows"], k["weights"]))
        )
        with _streams(log, capture), _moe_kernels(log), front, context, fused:
            out = moe.step(SimpleNamespace(mlp=mlp), x, st, defer=True)
        runs.append((log, log.tag(out), combined))
    (stock, stock_out, _), (got, got_out, combined) = runs
    start = [e[0] for e in stock].index("gate")
    assert [e[0] for e in stock[start : start + 3]] == [
        "gate",
        "topk",
        "experts_deferred",
    ]
    on_mkm = [
        ("router_front", "main", ("x(7, 4)", "gate_w(512, 4096)", "scale(1,)"), ()),
        (
            "mkm",
            "main",
            (
                "maps",
                "logits(7, 512)",
                "x_fp4(7, 2048)",
                "x_sf(7, 256)",
                "w13_sf(4,)",
                "w2_sf(8,)",
                "g1(512,)",
                "c(512,)",
                "g2(512,)",
            ),
            (),
        ),
        ("combine", "main"),
    ]
    assert got == stock[:start] + on_mkm + stock[start + 3 :]
    assert got_out == stock_out == "handoff"
    (deferred,) = (c.hidden_states for c in combined)
    assert deferred.gemm2_out is log.known["rows"]
    assert deferred.expert_weights is log.known["weights"]
    assert torch.equal(
        deferred.expanded_idx_to_permuted_idx, torch.arange(70, dtype=torch.int32)
    )
    assert deferred.top_k == 10


def test_the_mkm_slot_names_what_a_layer_lacks():
    from sglang.srt.layers.moe.utils import RoutingMethodType

    from sgl_nvidia_mega.qwen3_5.moe import mkm

    def meta(dtype, *shape):
        return torch.empty(shape, dtype=dtype, device="meta")

    fp8 = torch.float8_e4m3fn
    experts = SimpleNamespace(
        w13_weight=meta(torch.uint8, 512, 256, 2048),
        w2_weight=meta(torch.uint8, 512, 4096, 64),
        w13_weight_scale=meta(fp8, 512, 256, 256),
        w2_weight_scale=meta(fp8, 512, 4096, 8),
        g1_alphas=meta(torch.float32, 512),
        g1_scale_c=meta(torch.float32, 512),
        g2_alphas=meta(torch.float32, 512),
        num_experts=512,
        num_local_experts=512,
        moe_ep_rank=0,
        top_k=10,
        routing_method_type=RoutingMethodType.RenormalizeNaive,
        moe_runner_config=SimpleNamespace(
            activation="silu",
            is_gated=True,
            routed_scaling_factor=None,
            gemm1_alpha=None,
            gemm1_clamp_limit=None,
            swiglu_limit=None,
        ),
    )
    cfg = SimpleNamespace(
        top_k=10,
        renormalize=True,
        scoring_func="softmax",
        correction_bias=None,
        use_grouped_topk=False,
        num_fused_shared_experts=0,
    )
    mlp = SimpleNamespace(experts=experts, topk=SimpleNamespace(topk_config=cfg))
    assert mkm.unsupported(mlp) == ""
    experts.w13_weight = meta(torch.uint8, 512, 512, 2048)  # TP4's intermediate
    assert mkm.unsupported(mlp) == "w13_weight torch.uint8 (512, 512, 2048)"
    experts.w13_weight = meta(torch.uint8, 512, 256, 2048)
    cfg.scoring_func = "sigmoid"
    assert mkm.unsupported(mlp).startswith("routing other than")
    cfg.scoring_func = "softmax"
    experts.moe_runner_config.swiglu_limit = 7.0
    assert mkm.unsupported(mlp).startswith("an expert activation")


def test_the_route_slot_is_not_built_for_a_layer_it_cannot_serve(monkeypatch):
    from sgl_nvidia_mega.qwen3_5.moe import route

    mlp = _moe_block(_log(), torch.zeros(7, 4))
    mlp.gate.weight = torch.zeros(512, 4096, dtype=torch.float16)
    layers = [SimpleNamespace(mlp=mlp)]
    model = SimpleNamespace(
        model=SimpleNamespace(start_layer=0, end_layer=1, layers=layers)
    )
    monkeypatch.setattr(runtime, "SLOTS", frozenset({"route"}))
    ctx = SimpleNamespace(widths=(7,), tokens_per_request=7, deadline_s=60.0)
    want = "layer 0: router gate torch.float16 (512, 4096)"
    assert route.build(model, 7, fused=False).reason == want
    with pytest.raises(
        RuntimeError, match="requested slots \\['route'\\] built no width"
    ):
        runtime.build(model, ctx)


def _gdn_module(log, x):
    """A Qwen3_5GatedDeltaNet whose projections, backend call, norm and out_proj log."""
    mod = qwen3_5.Qwen3_5GatedDeltaNet.__new__(qwen3_5.Qwen3_5GatedDeltaNet)
    torch.nn.Module.__init__(mod)
    qkvz, ba = torch.zeros(7, 10), torch.zeros(7, 2)
    z = torch.zeros(7, 2, 3)
    core = torch.zeros(7, 2, 3)
    log.known.update(qkvz=qkvz, ba=ba, z=z, core=core)
    mod.__dict__.update(
        num_k_heads=16,
        num_v_heads=64,
        attn_tp_size=8,
        head_k_dim=128,
        head_v_dim=128,
        _forward_input_proj=log.op("in_proj", (qkvz, ba)),
        attn=log.op("linear_attention", core),
        norm=log.op("gated_norm", torch.ones(14, 3)),
        out_proj=log.op("out_proj", ("out", None)),
    )
    split = log.op("split", ("mixed", z, "b", "a"))
    return mod, split


@pytest.mark.parametrize("packed_verify", [False, True])
def test_the_gdn_step_issues_what_sglang_issues(packed_verify):
    runs = []
    for which in ("sglang", "plugin"):
        x = torch.zeros(7, 4)
        log = _log(x=x)
        mod, split = _gdn_module(log, x)
        backend = SimpleNamespace(
            linear_attn_backend=SimpleNamespace(supports_packed_verify=packed_verify)
        )
        mode = SimpleNamespace(
            is_target_verify=lambda: True,
            is_decode=lambda: False,
            is_extend_without_speculative=lambda: False,
        )
        # a verify batch is never relabeled: its logical mode is its label
        batch = SimpleNamespace(forward_mode=mode, logical_forward_mode=mode)
        with (
            patch.object(qwen3_5, "_is_xpu", False),
            patch.object(qwen3_5, "_is_cpu", False),
            patch.object(qwen3_5, "_is_cuda", True),
            patch.object(qwen3_5, "_gdn_decode_fused_proj_conv", True),
            patch.object(qwen3_5, "fused_qkvzba_split_reshape_cat_contiguous", split),
            patch(
                "sglang.kernels.ops.attention.triton_gdn_fused_proj."
                "fused_qkvzba_split_reshape_cat_contiguous",
                split,
            ),
            patch.object(qwen3_5, "_fp8_tuple_input", return_value=False),
            patch(
                "sglang.srt.model_executor.forward_context.get_attn_backend",
                return_value=backend,
            ),
        ):
            try:
                if which == "sglang":
                    out = mod.forward(x, batch)
                else:
                    st = SimpleNamespace(fb=batch, slots=plan.Slots())
                    out = gdn.step(SimpleNamespace(linear_attn=mod), x, st)
            except (NotImplementedError, RuntimeError, AttributeError) as e:
                out = type(e).__name__
        runs.append((log, out))
    (want_log, want), (got_log, got) = runs
    if packed_verify:
        # SGLang folds the split and conv into such a backend; the step refuses it
        assert got == "NotImplementedError" and want != got
        return
    assert got_log == want_log and want == got == "out"


def _gdn_verify_setup(log, *, ring_len=16, temporal_dtype=torch.bfloat16, draft=7):
    """A GatedDeltaNet whose projections, norm and out_proj log, and the binding of a
    one-request verify step (layer 2 is the pools' third GDN layer)."""
    from sgl_nvidia_mega.qwen3_5.gdn import verify

    H, K, HV, V = 2, 4, 4, 4
    D = 2 * H * K + HV * V
    k = log.known
    k.update(
        qkvz=torch.zeros(draft, D + HV * V),
        ba=torch.zeros(draft, 2 * HV),
        scale=torch.ones(1),
        states=torch.tensor([5]),
        inter=torch.tensor([0]),
        qsl=torch.tensor([0, draft]),
        req=torch.tensor([3]),
        conv=torch.zeros(4, 9, D, 3),
        window=torch.zeros(4, 9, draft, D, 3),
        temporal=torch.zeros(4, 9, HV, K, V, dtype=temporal_dtype),
        ring_d=torch.zeros(4, 9, HV, ring_len, V),
        ring_k=torch.zeros(4, 9, H, ring_len, K),
        ring_g=torch.zeros(4, 9, HV, ring_len),
        write_pos=torch.zeros(9, dtype=torch.int32),
        cache_base=torch.zeros(9, dtype=torch.int32),
        is_flush=torch.zeros(9, dtype=torch.int32),
    )
    # layer 2's slices: the pools' third GDN layer
    k.update({f"{n}2": k[n][2] for n in ("conv", "window", "temporal", "ring_d")})
    attn = SimpleNamespace(
        layer_id=2,
        conv_weights=torch.zeros(D, 4),
        bias=None,
        activation="silu",
        q_dim=H * K,
        k_dim=H * K,
        v_dim=HV * V,
        num_q_heads=H,
        num_k_heads=H,
        num_v_heads=HV,
        head_q_dim=K,
        head_k_dim=K,
        head_v_dim=V,
        A_log=torch.zeros(HV),
        dt_bias=torch.zeros(HV),
    )
    norm = SimpleNamespace(weight=torch.ones(V), eps=1e-6, activation="silu")
    k.update(out_w=torch.zeros(HV * V, 8), out_s=torch.ones(1))
    out_proj = SimpleNamespace(
        weight=k["out_w"], weight_scale=k["out_s"], input_scale=k["scale"]
    )
    gdn = SimpleNamespace(
        attn=attn,
        norm=norm,
        out_proj=out_proj,
        num_k_heads=H,
        num_v_heads=HV,
        attn_tp_size=1,
        head_k_dim=K,
        head_v_dim=V,
        _forward_input_proj=lambda x: (k["qkvz"], k["ba"]),
    )
    meta = SimpleNamespace(
        state_indices=k["states"],
        intermediate_indices=k["inter"],
        query_start_loc=k["qsl"],
    )
    pools = SimpleNamespace(
        layer_ids=(0, 1, 2, 4),
        conv=(k["conv"],),
        intermediate_conv_window=(k["window"],),
        temporal=k["temporal"],
        ring_d=k["ring_d"],
        ring_k=k["ring_k"],
        ring_g=k["ring_g"],
        ring_rawv=None,
        ring_rawk=None,
        ring_beta=None,
        ring_write_pos=k["write_pos"],
        ring_cache_base=k["cache_base"],
        ring_is_flush=k["is_flush"],
    )
    st = SimpleNamespace(
        fb=SimpleNamespace(req_pool_indices=k["req"]),
        binding=SimpleNamespace(tokens_per_request=draft),
        view=SimpleNamespace(gdn=meta, mamba=pools),
        scratch={},
    )
    return verify, gdn, st


@pytest.mark.parametrize("push", [False, True])
def test_the_gdn_verify_slot_runs_the_backend_chain_on_the_forward_views_rows(push):
    """With ``push`` (slot gdn_push) the norm releases its dependents early and
    out_proj runs as the hop's push GEMM on the layer's packed weights."""
    import msgspec

    from sgl_nvidia_mega.qwen3_5.gdn import conv, norm_fp8, replayssm

    log = _log()
    verify, gdn, st = _gdn_verify_setup(log)
    k = log.known
    k["packed2"] = torch.zeros(16, dtype=torch.uint8)
    hop = SimpleNamespace(out_proj_push=log.op("out_proj_push", torch.zeros(7, 8)))
    st.bound = {
        "boundary": SimpleNamespace(hop=hop),
        "gdn": SimpleNamespace(packed={2: k["packed2"]}),
    }
    opts = msgspec.structs.replace(verify.SLOT, push=push)

    def views(qkvz, ba, nk, nv, hk, hv):
        log.append(("views", log.tag(qkvz), log.tag(ba), nk, nv, hk, hv))
        return qkvz[:, :32], qkvz[:, 32:].view(7, nv, hv), ba[:, :4], ba[:, 4:]

    def conv_update(x, state, weight, bias, act, states, window, inter, **kw):
        log.append(
            (
                "conv",
                tuple(x.shape),
                log.tag(state),
                log.tag(states),
                log.tag(window),
                log.tag(inter),
                tuple(sorted(kw.items())),
            )
        )
        return torch.zeros(1, 7, 32).transpose(1, 2)

    names = (
        "q",
        "k",
        "v",
        "a",
        "b",
        "A_log",
        "dt_bias",
        "out",
        "checkpoint_state",
        "d_cache",
        "k_cache",
        "g_cache",
        "rawv_cache",
        "rawk_cache",
        "beta_cache",
        "query_start_loc",
        "ssm_state_indices",
        "replay_indices",
        "write_pos",
        "cache_base",
        "is_flush",
    )

    def spec_decode(*args, **kw):
        tags = {n: log.tag(v) for n, v in zip(names, args)}
        log.append(("verify", tags, kw))

    def norm(x, weight, eps, z, scale, **kw):
        log.append(
            (
                "norm_fp8",
                tuple(x.shape),
                tuple(z.shape),
                log.tag(scale),
                tuple(sorted(kw.items())),
            )
        )
        return torch.zeros(28, 4)

    @contextlib.contextmanager
    def symmetric(group, disabled=False):
        log.append(("symmetric",))
        yield

    with (
        patch(
            "sglang.kernels.ops.attention.triton_gdn_fused_proj.qwen3_5_gdn_prefill_projection_views",
            views,
        ),
        patch.object(conv, "conv_update", conv_update),
        patch.object(replayssm, "verify", spec_decode),
        patch.object(norm_fp8, "rms_norm_gated_fp8", norm),
        patch(
            "sglang.srt.distributed.device_communicators.pynccl_allocator.use_symmetric_memory",
            symmetric,
        ),
        patch("sglang.srt.layers.dp_attention.is_allocation_symmetric", lambda: True),
        patch(
            "sglang.srt.runtime_context.get_parallel",
            lambda: SimpleNamespace(tp_group="tp"),
        ),
        patch(
            "sglang.srt.layers.quantization.fp8_utils.flashinfer_bmm_fp8",
            log.op("bmm_fp8"),
            create=True,
        ),
    ):
        for _ in range(2):  # two layers of one step
            verify.step(gdn, torch.zeros(7, 8), st, opts)
    proj = ["out_proj_push"] if push else ["symmetric", "bmm_fp8"]
    one_layer = ["conv", "verify", "norm_fp8", *proj, "views"]
    assert [e[0] for e in log] == ["views", *one_layer, *one_layer[:-1]]
    _, shape, state, states, window, inter, kw = log[1]
    assert (shape, states, inter) == ((1, 32, 7), "states(1,)", "inter(1,)")
    assert state == "conv2(9, 32, 3)" and window == "window2(9, 7, 32, 3)"
    assert dict(kw) == {"block_n": 128, "token_major_out": True}
    tags, options = log[2][1], log[2][2]
    assert (
        tags["ssm_state_indices"] == "states(1,)"
        and tags["replay_indices"] == "req(1,)"
    )
    assert tags["query_start_loc"] == "qsl(2,)" and tags["write_pos"] == "write_pos(9,)"
    assert (
        tags["rawv_cache"] is None
        and tags["checkpoint_state"] == "temporal2(9, 4, 4, 4)"
    )
    assert tags["d_cache"] == "ring_d2(9, 4, 16, 4)"
    assert options == dict(
        max_cache_len=16,
        max_spec_len=7,
        scale=0.5,
        null_block_id=-1,
        block_v=32,
        num_warps=2,
        pdl=False,
    )
    assert log[3] == (
        "norm_fp8",
        (28, 4),
        (7, 4, 4),
        "scale(1,)",
        (("activation", "silu"), ("early_dependents", push)),
    )
    if push:
        # the push GEMM on the FP8 rows and the layer's packed weights and scales
        assert log[4] == (
            "out_proj_push",
            "main",
            (),
            (
                ("input_scale", "scale(1,)"),
                ("packed_w", "packed2(16,)"),
                ("weight_scale", "out_s(1,)"),
                ("x_fp8", "?(7, 16)"),
            ),
        )
        return
    # out_proj's BMM on the FP8 rows with its static input scale, output in the activation dtype
    assert log[5] == (
        "bmm_fp8",
        "main",
        ("?(7, 16)", "out_w(16, 8)", "scale(1,)", "out_s(1,)", "?"),
        (),
    )


@pytest.mark.parametrize("takes", [False, True])
def test_the_gdn_push_slot_pushes_only_where_the_moe_input_consumes(
    monkeypatch, caplog, takes
):
    import msgspec

    from sgl_nvidia_mega.qwen3_5 import gdn
    from sgl_nvidia_mega.qwen3_5.gdn import verify

    monkeypatch.setattr(verify, "_ENGAGED", set())
    seen = []
    monkeypatch.setattr(boundary_hop, "takes_moe_input", lambda hop, *a: takes)
    monkeypatch.setattr(verify, "step", lambda mod, x, st, opts: seen.append(opts))
    layer = SimpleNamespace(linear_attn="gdn")
    st = SimpleNamespace(
        slots=plan.Slots(gdn_verify=True, hop=True, gdn_push=True),
        bound={"boundary": boundary_hop.Bound(hop="hop")},
    )
    with caplog.at_level("INFO"):
        gdn.step(layer, torch.zeros(7, 4), st)
    assert seen == [msgspec.structs.replace(verify.SLOT, push=takes)]
    # the push is logged once where it is chosen, for slot gdn_verify's and gdn_front's paths
    logged = "NVIDIA megakernel: Qwen3.5 slot gdn_push runs at 7 rows" in caplog.text
    assert logged == takes


def _gdn_front_setup(log):
    """A Qwen3.5 TP8 GatedDeltaNet (layer 2, the pools' third GDN layer) for slot
    gdn_front, with the binding of a one-request verify step of 7 rows."""
    H, K, HV, V, D = 2, 128, 8, 128, 1536
    k = log.known
    k.update(
        x=torch.zeros(7, 4096, dtype=torch.bfloat16),
        packed2=torch.zeros(16, dtype=torch.uint8),
        front2=torch.zeros(16, dtype=torch.uint8),
        error=torch.zeros(1, dtype=torch.int32),
        in_s=torch.ones(1),
        in_ws=torch.ones(1),
        ba_w=torch.zeros(16, 4096, dtype=torch.bfloat16),
        conv_w=torch.zeros(D, 4, dtype=torch.bfloat16),
        out_s=torch.ones(1),
        out_ws=torch.ones(1),
        norm_w=torch.ones(V),
        states=torch.tensor([5]),
        inter=torch.tensor([0]),
        qsl=torch.tensor([0, 7]),
        req=torch.tensor([2]),
        conv=torch.zeros(4, 3, D, 3),
        window=torch.zeros(4, 3, 7, D, 3),
        temporal=torch.zeros(4, 3, HV, K, V),
        ring_d=torch.zeros(4, 3, HV, 16, V),
        ring_k=torch.zeros(4, 3, H, 16, K),
        ring_g=torch.zeros(4, 3, HV, 16),
        write_pos=torch.zeros(3, dtype=torch.int32),
        cache_base=torch.zeros(3, dtype=torch.int32),
        is_flush=torch.zeros(3, dtype=torch.int32),
    )
    k.update({f"{n}2": k[n][2] for n in ("conv", "window", "temporal")})
    k.update({f"{n}2": k[n][2] for n in ("ring_d", "ring_k", "ring_g")})
    attn = SimpleNamespace(
        layer_id=2,
        conv_weights=k["conv_w"],
        q_dim=H * K,
        k_dim=H * K,
        v_dim=HV * V,
        num_k_heads=H,
        num_v_heads=HV,
        head_k_dim=K,
        head_v_dim=V,
        A_log=torch.zeros(HV),
        dt_bias=torch.zeros(HV),
    )
    gdn = SimpleNamespace(
        attn=attn,
        norm=SimpleNamespace(weight=k["norm_w"], eps=1e-6),
        in_proj_qkvz=SimpleNamespace(input_scale=k["in_s"], weight_scale=k["in_ws"]),
        in_proj_ba=SimpleNamespace(weight=k["ba_w"]),
        out_proj=SimpleNamespace(input_scale=k["out_s"], weight_scale=k["out_ws"]),
    )
    pools = SimpleNamespace(
        layer_ids=(0, 1, 2, 4),
        conv=(k["conv"],),
        intermediate_conv_window=(k["window"],),
        temporal=k["temporal"],
        ring_d=k["ring_d"],
        ring_k=k["ring_k"],
        ring_g=k["ring_g"],
        ring_rawv=None,
        ring_rawk=None,
        ring_beta=None,
        ring_write_pos=k["write_pos"],
        ring_cache_base=k["cache_base"],
        ring_is_flush=k["is_flush"],
    )
    meta = SimpleNamespace(
        state_indices=k["states"],
        intermediate_indices=k["inter"],
        query_start_loc=k["qsl"],
        pad_slot_id=-1,
    )
    st = SimpleNamespace(
        fb=SimpleNamespace(req_pool_indices=k["req"]),
        view=SimpleNamespace(gdn=meta, mamba=pools),
        bound={
            "gdn": SimpleNamespace(
                front={2: k["front2"]}, packed={2: k["packed2"]}, error=k["error"]
            )
        },
    )
    return gdn, st


@pytest.mark.parametrize("push", [False, True])
def test_the_gdn_front_slot_runs_the_front_the_verify_the_norm_and_out_proj(push):
    """The branch's fused path in its order: the front (projections, conv, b / a) on
    the layer's pools at the step's slots, the ReplaySSM verify on its conv rows waiting
    for it before the gates, the norm into out_proj's FP8 rows, then out_proj (with
    slot gdn_push, the hop's push GEMM)."""
    import msgspec

    from sgl_nvidia_mega.qwen3_5.gdn import front, front_norm, replayssm, verify
    from sgl_nvidia_mega.qwen3_5.sm103.gdn import front as kernel

    log = _log()
    gdn, st = _gdn_front_setup(log)
    k = log.known
    hop = SimpleNamespace(out_proj_push=log.op("out_proj_push", "pushed"))
    st.bound["boundary"] = SimpleNamespace(hop=hop)
    seen = {}

    def front_launch(*args, **kw):
        seen["projected"], seen["conv"], seen["ba"] = args[9], args[10], kw["ba_out"]
        log.op("front")(*args, **kw)

    def spec_decode(q, k_, v, a, b, *args, **kw):
        seen["core"] = args[2]
        seen["qkv"] = (q, k_, v)
        seen["ab"] = (a, b)
        log.append(("verify", tuple(map(log.tag, args[3:])), kw))

    def norm(x, z, weight, scale, quant, rstd, eps):
        seen["norm"] = (x, z, quant, rstd)
        log.append(("norm", log.tag(weight), log.tag(scale), eps))

    opts = msgspec.structs.replace(verify.SLOT_PDL, push=push)
    with (
        patch.object(kernel, "front", front_launch),
        patch.object(replayssm, "verify", spec_decode),
        patch.object(front_norm, "norm", norm),
        patch.object(
            verify,
            "_out_proj",
            lambda linear, fp8, dtype: (
                log.append(("out_proj", linear, fp8, dtype)) or "partial"
            ),
        ),
    ):
        out = front.step(gdn, k["x"], st, opts)
    assert [e[0] for e in log] == [
        "front",
        "verify",
        "norm",
        "out_proj_push" if push else "out_proj",
    ]
    _, _, args, kw = log[0]
    assert args == (
        "x(7, 4096)",
        "front2(16,)",
        "in_s(1,)",
        "in_ws(1,)",
        "conv_w(1536, 4)",
        "conv2(3, 1536, 3)",
        "window2(3, 7, 1536, 3)",
        "states(1,)",
        "inter(1,)",
        "?(7, 2560)",
        "?(7, 1536)",
        "error(1,)",
    )
    assert dict(kw) == {
        "ba_weight": "ba_w(16, 4096)",
        "ba_out": "?(7, 16)",
        "checkpoint": "temporal2(3, 8, 128, 128)",
        "history_d": "ring_d2(3, 8, 16, 128)",
        "history_k": "ring_k2(3, 2, 16, 128)",
        "history_g": "ring_g2(3, 8, 16)",
        "ring_indices": "req(1,)",
        "pad_slot": -1,
    }
    # q / k / v are the conv rows' views; a / b the front's b8 / a8 halves
    q, k_, v = seen["qkv"]
    conv, ba = seen["conv"], seen["ba"]
    assert q.data_ptr() == conv.data_ptr() and q.shape == (7, 2, 128)
    assert v.shape == (7, 8, 128) and v.data_ptr() == conv[:, 512:].data_ptr()
    a, b = seen["ab"]
    assert a.data_ptr() == ba[:, 8:].data_ptr() and b.data_ptr() == ba.data_ptr()
    assert a.shape == b.shape == (7, 8) and a.stride() == (16, 1)
    tags, options = log[1][1], log[1][2]
    assert tags[:4] == (
        "temporal2(3, 8, 128, 128)",
        "ring_d2(3, 8, 16, 128)",
        "ring_k2(3, 2, 16, 128)",
        "ring_g2(3, 8, 16)",
    )
    assert tags[4:10] == (None, None, None, "qsl(2,)", "states(1,)", "req(1,)")
    assert options == dict(
        max_cache_len=16,
        max_spec_len=7,
        scale=128**-0.5,
        null_block_id=-1,
        block_v=32,
        num_warps=2,
        pdl=True,
        wait_for_gate_producer=True,
    )
    # the norm reads the verify's output and the front's z, into FP8 [56, 128]
    x_norm, z, quant, rstd = seen["norm"]
    assert x_norm.data_ptr() == seen["core"].data_ptr() and x_norm.shape == (56, 128)
    assert z.data_ptr() == seen["projected"][:, 1536:].data_ptr()
    assert z.shape == (7, 8, 128) and z.stride() == (2560, 128, 1)
    assert quant.dtype == torch.float8_e4m3fn and quant.shape == (56, 128)
    assert rstd.shape == (56,) and log[2][1:] == ("norm_w(128,)", "out_s(1,)", 1e-6)
    if push:
        assert out == "pushed"
        assert dict(log[3][3]) == {
            "input_scale": "out_s(1,)",
            "packed_w": "packed2(16,)",
            "weight_scale": "out_ws(1,)",
            "x_fp8": "?(7, 1024)",
        }
    else:
        assert out == "partial"
        _, linear, fp8, dtype = log[3]
        assert linear is gdn.out_proj and dtype == torch.bfloat16
        assert fp8.data_ptr() == quant.data_ptr() and fp8.shape == (7, 1024)


@pytest.mark.parametrize("front_slot", [False, True])
def test_the_gdn_step_runs_the_front_where_the_plan_gives_it(monkeypatch, front_slot):
    from sgl_nvidia_mega.qwen3_5 import gdn
    from sgl_nvidia_mega.qwen3_5.gdn import front, verify

    seen = []
    monkeypatch.setattr(verify, "step", lambda mod, x, st, opts: seen.append("verify"))
    monkeypatch.setattr(front, "step", lambda mod, x, st, opts: seen.append(opts))
    st = SimpleNamespace(
        slots=plan.Slots(gdn_verify=True, gdn_pdl=True, gdn_front=front_slot)
    )
    gdn.step(SimpleNamespace(linear_attn="gdn"), torch.zeros(7, 4), st)
    assert seen == ([verify.SLOT_PDL] if front_slot else ["verify"])


def _front_gdn(**change):
    """A Qwen3.5 TP8 GatedDeltaNet as SGLang loads the NVFP4 V2 checkpoint (static FP8
    in_proj_qkvz and out_proj, BF16 in_proj_ba), with ``change`` applied."""
    from sglang.srt.layers.quantization.modelopt_quant import ModelOptFp8LinearMethod
    from sglang.srt.layers.quantization.unquant import UnquantizedLinearMethod

    def fp8(shape):
        method = object.__new__(ModelOptFp8LinearMethod)
        method.use_marlin = method.use_sm120_fp8 = False
        return SimpleNamespace(
            quant_method=method,
            weight=torch.zeros(shape, dtype=torch.float8_e4m3fn),
            input_scale=torch.ones(1),
            weight_scale=torch.ones(1),
        )

    gdn = SimpleNamespace(
        attn_tp_size=8,
        head_k_dim=128,
        head_v_dim=128,
        conv_kernel_size=4,
        activation="silu",
        attn=SimpleNamespace(
            num_k_heads=2,
            num_v_heads=8,
            bias=None,
            conv_weights=torch.zeros(1536, 4, dtype=torch.bfloat16),
        ),
        norm=SimpleNamespace(
            activation="silu", norm_before_gate=True, bias=None, group_size=None
        ),
        in_proj_qkvz=fp8((4096, 2560)),
        out_proj=fp8((1024, 4096)),
        in_proj_ba=SimpleNamespace(
            quant_method=object.__new__(UnquantizedLinearMethod),
            bias=None,
            weight=torch.zeros(16, 4096, dtype=torch.bfloat16),
        ),
    )
    for path, value in change.items():
        obj, *names = path.split("__")
        target = gdn if obj == "gdn" else getattr(gdn, obj)
        for name in names[:-1]:
            target = getattr(target, name)
        setattr(target, names[-1], value)
    return gdn


@pytest.mark.parametrize(
    "change, want",
    [
        ({}, ""),
        ({"gdn__attn_tp_size": 4}, "TP 4"),
        ({"norm__group_size": 2}, "a grouped, biased or gate-first norm"),
        ({"attn__bias": torch.zeros(1536)}, "a conv bias"),
        ({"in_proj_qkvz__weight": torch.zeros(4096, 2560)}, "in_proj_qkvz is not"),
        ({"in_proj_ba__bias": torch.zeros(16)}, "in_proj_ba is not"),
        ({"out_proj__input_scale": torch.ones(2)}, "out_proj is not"),
    ],
)
def test_the_gdn_front_slot_names_what_a_layer_lacks(change, want):
    from sgl_nvidia_mega.qwen3_5.gdn import front

    got = front.unsupported(_front_gdn(**change))
    assert (want in got) if want else got == ""


def test_the_gdn_front_slot_names_the_pools_it_cannot_serve():
    from sgl_nvidia_mega.qwen3_5.gdn import front

    def pools(window):
        return SimpleNamespace(
            conv=(torch.zeros(2, 3, 1536, 3, dtype=torch.bfloat16),),
            intermediate_conv_window=(window,),
        )

    phys = torch.zeros(2, 3, 1536, 9, dtype=torch.bfloat16)
    dedup = phys.as_strided((2, 3, 7, 1536, 3), (*phys.stride()[:2], 1, 9, 1))
    assert front.pool_reasons(pools(dedup)) == []
    dense = torch.zeros(2, 3, 7, 1536, 3, dtype=torch.bfloat16)
    assert "deduplicated" in front.pool_reasons(pools(dense))[0]


def test_in_proj_qkvz_packs_into_the_fronts_fragments():
    from sgl_nvidia_mega.qwen3_5.sm103.gdn import front

    w = torch.arange(2560 * 4096, dtype=torch.int64).remainder(251).to(torch.uint8)
    packed = front.pack(w.view(2560, 4096).view(torch.float8_e4m3fn))
    assert packed.dtype == torch.uint8 and packed.numel() == 2560 * 4096
    # unit 0, rank 0, warp 0, step 0, block 0, lane 0: the first 16 bytes the CTA loads
    want = w.view(2560, 4096)
    first = packed[:16].view(4, 4)
    assert torch.equal(first[0], want[0, 0:4]) and torch.equal(first[1], want[8, 0:4])
    assert torch.equal(first[2], want[0, 16:20]) and torch.equal(
        first[3], want[8, 16:20]
    )
    with pytest.raises(ValueError, match="the GDN front packs"):
        front.pack(torch.zeros(2560, 4096))


def test_the_fronts_launch_arguments_have_the_kernels_layout():
    import ctypes

    from sgl_nvidia_mega.qwen3_5.sm103.gdn import front

    # gdn_pdl_front_ba_mkm_detail::Params and gdn_ba::Params on a 64-bit target
    assert ctypes.sizeof(front.Prefetch) == 32
    assert ctypes.sizeof(front.BaParams) == 208 and front.BaParams.prefetch.offset == 40
    assert front.BaParams.pad_slot.offset == 200
    assert ctypes.sizeof(front.Params) == 376 and front.Params.ba.offset == 168
    assert (
        front.Params.state_stride.offset == 112 and front.Params.pad_slot.offset == 160
    )


def test_the_fronts_l2_prefetch_takes_contiguous_slots_only():
    from sgl_nvidia_mega.qwen3_5.sm103.gdn import front

    t = torch.zeros(3, 8, 16, 128, dtype=torch.bfloat16)
    region = front._prefetch(t, 1)
    assert (region.slot_stride, region.bytes, region.ring) == (
        8 * 16 * 128 * 2,
        8 * 16 * 128 * 2,
        1,
    )
    assert front._prefetch(t.transpose(2, 3), 0) is None
    assert front._prefetch(torch.zeros(3, 7, dtype=torch.bfloat16), 0) is None


@pytest.mark.parametrize(
    "kw, want",
    [
        (dict(ring_len=8), "ring length 8"),
        (dict(temporal_dtype=torch.float32), "torch.float32 checkpoints"),
        (dict(draft=9), "9 rows a request"),
    ],
)
def test_the_gdn_verify_slot_names_the_pools_it_cannot_serve(kw, want):
    verify, _, st = _gdn_verify_setup(_log(), **kw)
    assert verify.pool_reasons(st.view.mamba, st.binding.tokens_per_request) == [want]
    # the first covered step refuses, before it runs anything
    built = _verify(binding=st.binding)
    built.slots = {7: plan.Slots(gdn_verify=True)}
    step = Step(mode="target_verify", rows=7, view=st.view)
    with pytest.raises(RuntimeError, match=f"slot gdn_verify cannot run: {want}"):
        built.forward(step, None, torch.zeros(7, dtype=torch.int64), None, None)


def test_the_pools_are_checked_once_at_the_first_covered_step(monkeypatch):
    verify, _, st = _gdn_verify_setup(_log())
    built = _verify(binding=st.binding)
    built.slots = {7: plan.Slots(gdn_verify=True)}
    checks = []
    real = verify.pool_reasons
    monkeypatch.setattr(verify, "pool_reasons", lambda *a: checks.append(a) or real(*a))
    monkeypatch.setattr(built, "_forward", lambda *a: "logits")
    step = Step(mode="target_verify", rows=7, view=st.view)
    for _ in range(2):
        assert built.forward(step, None, "ids", "pos", "fb") == "logits"
    assert checks == [(st.view.mamba, 7)]


def _attention_layer(log, x):
    layer = qwen3_5.Qwen3_5AttentionDecoderLayer.__new__(
        qwen3_5.Qwen3_5AttentionDecoderLayer
    )
    torch.nn.Module.__init__(layer)
    q, k, v, gate, o = (torch.zeros(7, 4) for _ in range(5))
    log.known.update(q=q, k=k, v=v, gate=gate, o=o)
    layer.__dict__.update(
        attn_output_gate=True,
        forward_prepare_cuda_fused=log.op("prepare", (q, k, v, gate)),
        attn=log.op("attention", o),
        o_proj=log.op("o_proj", ("out", None)),
    )
    return layer


def test_the_attention_step_issues_what_sglang_issues():
    runs = []
    for which in ("sglang", "plugin"):
        x = torch.zeros(7, 4)
        log = _log(x=x, positions="pos", batch="fb")
        layer = _attention_layer(log, x)
        mul = log.op("sigmoid_mul", "gated")
        log.known["gated"] = "gated"
        with (
            patch.object(qwen3_5, "_is_cuda", True),
            patch.object(qwen3_5, "_is_npu", False),
            patch.object(qwen3_5, "_fp8_tuple_input", return_value=False),
            patch.object(qwen3_5, "fused_sigmoid_mul", mul),
            patch("sglang.kernels.ops.elementwise.elementwise.fused_sigmoid_mul", mul),
        ):
            if which == "sglang":
                out = layer.self_attention(
                    positions="pos", hidden_states=x, forward_batch="fb"
                )
            else:
                st = SimpleNamespace(positions="pos", fb="fb", slots=plan.Slots())
                out = attention.step(layer, x, st)
        runs.append((log, out))
    (want_log, want), (got_log, got) = runs
    assert got_log == want_log and want == got == "out"


# ------------------------------------------------------------------- slot skv2


class _Plan:
    """The SKV2 plan's stand-in: logs its build and its runs."""

    log = None

    def __init__(self, query, kv, table, lengths):
        self.log.append(("plan", self.log.tag(query), *map(self.log.tag, kv)))
        self.log.append(("strides", *(c.stride() for c in kv)))
        self.log.append(("plan_inputs", self.log.tag(table), self.log.tag(lengths)))

    def run(self, query, kv, table, lengths, *, qk_scale, v_scale, out):
        self.log.append(
            ("skv2", self.log.tag(query), self.log.tag(table), qk_scale, v_scale)
        )
        self.log.known["skv2_out"] = out
        return out


@contextlib.contextmanager
def _skv2_kernels(log, label=None, capture=True):
    import sys

    from sglang.kernels.ops.kvcache import fused_fp8_qkv_kv_cache as fused_module

    q8 = log.known.setdefault("q8", torch.zeros(7, 1024, dtype=torch.float8_e4m3fn))

    def fused(**kw):
        log.append(
            (
                "fused_fp8_qkv_kv_cache",
                tuple(sorted((k, log.tag(v)) for k, v in kw.items())),
            )
        )
        return q8

    _Plan.log = log
    with (
        patch.object(fused_module, "fused_fp8_qkv_kv_cache", fused),
        patch.dict(
            sys.modules,
            {
                "sgl_nvidia_mega.qwen3_5.sm103.attention.skv2": SimpleNamespace(
                    StockSplitPlan=_Plan
                )
            },
        ),
        patch(
            "sglang.srt.model_executor.runner_utils.capture_mode."
            "get_capture_attention_variant",
            return_value=label,
        ),
        patch(
            "sglang.srt.model_executor.runner_utils.capture_mode.get_is_capture_mode",
            return_value=capture,
        ),
    ):
        yield


def _skv2_setup(log, k_scale=None, **fb):
    """Layer 7's RadixAttention (TP8: 4 query heads, 1 KV head of 256), the forward's
    view of a pool of 3 pages for layers 3 and 7, and the verify step's state."""
    from sgl_nvidia_mega.qwen3_5.attention import split_kv

    split_kv._PLANS.clear()
    pool = {
        name: torch.zeros(3 * 64, 1, 256, dtype=torch.float8_e4m3fn)
        for name in ("k3", "v3", "k7", "v7")
    }
    attn = _Callable(
        log.op("attention", "stock_o"),
        layer_id=7,
        k_scale=k_scale,
        v_scale=torch.full((1,), 0.5),
        tp_q_head_num=4,
        tp_k_head_num=1,
        tp_v_head_num=1,
        head_dim=256,
        scaling=0.0625,
        sliding_window_size=-1,
    )
    view = SimpleNamespace(
        kv=SimpleNamespace(
            layer_ids=(3, 7),
            k=(pool["k3"], pool["k7"]),
            v=(pool["v3"], pool["v7"]),
            k_scale=(1.0, 0.75),
            v_scale=(1.0, 0.5),
        ),
        attn=SimpleNamespace(
            page_table=torch.zeros(1, 4, dtype=torch.int32),
            cache_seqlens=torch.full((1,), 130, dtype=torch.int32),
        ),
    )
    batch = SimpleNamespace(
        out_cache_loc=torch.zeros(7, dtype=torch.int64), batch_size=1, **fb
    )
    log.known.update(
        pool,
        loc=batch.out_cache_loc,
        table=view.attn.page_table,
        lengths=view.attn.cache_seqlens,
        v_scale=attn.v_scale,
        fb=batch,
    )
    return attn, SimpleNamespace(view=view, fb=batch)


def test_the_skv2_slot_writes_kv_as_the_backend_does_then_runs_skv2_on_the_view():
    from sgl_nvidia_mega.qwen3_5.attention import split_kv

    log = _log()
    attn, st = _skv2_setup(log)
    q, k, v = (torch.zeros(7, n, dtype=torch.bfloat16) for n in (1024, 256, 256))
    log.known.update(q=q, k=k, v=v)
    with _skv2_kernels(log):
        outs = [split_kv.attend(attn, q, k, v, st) for _ in range(2)]  # two steps
    fused = log[0]
    assert fused[0] == "fused_fp8_qkv_kv_cache"
    kw = dict(fused[1])
    # TRT-LLM MHA's fused FP8 write of this layer's pool rows, its default scale for
    # a missing checkpoint scale
    assert (kw["q"], kw["k"], kw["v"]) == ("q(7, 1024)", "k(7, 256)", "v(7, 256)")
    assert (kw["k_cache"], kw["v_cache"], kw["cache_loc"]) == (
        "k7(192, 1, 256)",
        "v7(192, 1, 256)",
        "loc(7,)",
    )
    assert kw["k_scale"] == "?(1,)" and kw["v_scale"] == "v_scale(1,)"
    # the query as (rows, heads, head_dim) and the pool as HND pages; one plan
    assert log[1] == ("plan", "q8(7, 4, 256)", "k7(3, 1, 64, 256)", "v7(3, 1, 64, 256)")
    assert log[2] == ("strides", (16384, 16384, 256, 1), (16384, 16384, 256, 1))
    assert log[3] == ("plan_inputs", "table(1, 4)", "lengths(1,)")
    # the BMM scales TRT-LLM MHA passes: k_scale * scaling and v_scale
    assert log[4] == ("skv2", "q8(7, 4, 256)", "table(1, 4)", 0.75 * 0.0625, 0.5)
    assert [e[0] for e in log[5:]] == ["fused_fp8_qkv_kv_cache", "skv2"]
    assert all(o.shape == (7, 1024) and o.dtype == torch.bfloat16 for o in outs)


@pytest.mark.parametrize(
    "label, capture, fb, want",
    [
        ("split_kv_long", True, {}, "skv2"),
        ("split_kv_stock", True, {}, "attention"),
        # SGLang captures no label of ours (no runner takes the variants yet)
        (None, True, {}, "skv2"),
        (None, False, {}, "skv2"),
        # an eager step routed by the request's committed KV length
        (None, False, dict(attention_kv_len_lower_bound=8192), "skv2"),
        (None, False, dict(attention_kv_len_lower_bound=8191), "attention"),
        (
            None,
            False,
            dict(attention_kv_len_lower_bound=None, seq_lens_cpu=torch.tensor([9000])),
            "skv2",
        ),
        (
            None,
            False,
            dict(attention_kv_len_lower_bound=None, seq_lens_cpu=None),
            "attention",
        ),
    ],
)
def test_the_skv2_slot_takes_split_kv_where_the_branch_routes_it(
    label, capture, fb, want
):
    from sgl_nvidia_mega.qwen3_5.attention import split_kv

    log = _log()
    attn, st = _skv2_setup(log, **{"seq_lens_cpu": None, **fb})
    q8 = torch.zeros(7, 1024, dtype=torch.float8_e4m3fn)
    log.known["q8"] = q8
    with _skv2_kernels(log, label=label, capture=capture):
        split_kv.core(attn, q8, st)
    assert log[-1][0] == want
    if want == "attention":
        # SGLang's attention on the written rows: no K / V to write
        assert log[-1][2:] == (("q8(7, 1024)", None, None, "fb"), ())


def test_the_split_kv_variants_select_on_host_state_only():
    from sgl_nvidia_mega.qwen3_5.attention import route

    assert route.SplitKVGraphVariants.capture_labels == (
        "split_kv_stock",
        "split_kv_long",
    )
    assert (route.VERIFY.min_length, route.EXTEND.min_length) == (8192, 32768)

    def batch(bs=1, bound=None, lens=None):
        return SimpleNamespace(
            batch_size=bs, attention_kv_len_lower_bound=bound, seq_lens_cpu=lens
        )

    assert route.VERIFY.select(batch(bound=8192)) == "split_kv_long"
    assert route.VERIFY.select(batch(bound=8191)) == "split_kv_stock"
    assert route.EXTEND.select(batch(bound=32767)) == "split_kv_stock"
    assert route.VERIFY.select(batch(bs=2, bound=10**6)) == "split_kv_stock"
    assert route.VERIFY.select(batch(lens=torch.tensor([8192]))) == "split_kv_long"
    assert route.VERIFY.select(batch(lens=torch.tensor([9000, 9000]))) == (
        "split_kv_stock"
    )
    assert route.VERIFY.select(batch()) == "split_kv_stock"


@pytest.mark.parametrize(
    "kw, want",
    [
        (dict(), ""),
        (dict(rows=4), "4 rows a request (SKV2 runs 7)"),
        (dict(tp_q_head_num=2), "2 query / 1 KV heads a rank"),
        (dict(tp_k_head_num=2), "4 query / 2 KV heads a rank"),
        (dict(head_dim=128), "head_dim 128"),
        (dict(sliding_window_size=4096), "a sliding window of 4096"),
    ],
)
def test_the_skv2_slot_names_what_a_layer_lacks(kw, want):
    from sgl_nvidia_mega.qwen3_5.attention import split_kv

    rows = kw.pop("rows", 7)
    attn = SimpleNamespace(
        **{
            "tp_q_head_num": 4,
            "tp_k_head_num": 1,
            "head_dim": 256,
            "sliding_window_size": -1,
            **kw,
        }
    )
    with patch.object(torch.cuda, "get_device_capability", return_value=(10, 3)):
        got = split_kv.unsupported(attn, rows)
    assert want in got if want else got == ""
    with patch.object(torch.cuda, "get_device_capability", return_value=(9, 0)):
        assert (
            split_kv.unsupported(
                SimpleNamespace(
                    tp_q_head_num=4,
                    tp_k_head_num=1,
                    head_dim=256,
                    sliding_window_size=-1,
                ),
                7,
            )
            == "SKV2 runs on SM10x"
        )


def test_the_skv2_slot_names_the_pools_it_cannot_read():
    from sgl_nvidia_mega.qwen3_5.attention import split_kv

    def kv(*bufs):
        return SimpleNamespace(k=bufs, v=bufs)

    fp8 = torch.zeros(3 * 64, 1, 256, dtype=torch.float8_e4m3fn)
    assert split_kv.pool_reasons(kv(fp8), 64) == []
    assert split_kv.pool_reasons(kv(fp8), 32) == [
        "page size 32 (SKV2 reads pages of 64)"
    ]
    bf16 = torch.zeros(3 * 64, 1, 256, dtype=torch.bfloat16)
    assert "not contiguous FP8" in split_kv.pool_reasons(kv(bf16), 64)[0]
    odd = torch.zeros(65, 1, 256, dtype=torch.float8_e4m3fn)
    assert split_kv.pool_reasons(kv(odd), 64) == [
        "a KV pool buffer of shape (65, 1, 256)"
    ]


def test_the_skv2_slot_runs_the_attention_step_through_the_slot():
    runs = []
    for slots in (plan.Slots(), plan.Slots(skv2=True)):
        x = torch.zeros(7, 4)
        log = _log(x=x, positions="pos", batch="fb")
        layer = _attention_layer(log, x)
        mul = log.op("sigmoid_mul", "gated")
        st = SimpleNamespace(positions="pos", fb="fb", slots=slots)
        with (
            patch.object(qwen3_5, "_is_cuda", True),
            patch("sglang.kernels.ops.elementwise.elementwise.fused_sigmoid_mul", mul),
            patch(
                "sgl_nvidia_mega.qwen3_5.attention.split_kv.attend",
                log.op("skv2_attend", log.known["o"]),
            ),
        ):
            attention.step(layer, x, st)
        runs.append([e[0] for e in log])
    stock, slot = runs
    assert stock == ["prepare", "attention", "sigmoid_mul", "o_proj"]
    assert slot == ["prepare", "skv2_attend", "sigmoid_mul", "o_proj"]


def test_the_executor_offers_the_verify_split_kv_variants_with_slot_skv2():
    from sgl_nvidia_mega.qwen3_5.attention import route

    lm = SimpleNamespace(model=SimpleNamespace(start_layer=0, end_layer=0, layers=[]))
    stock = executor.Qwen35Verify(lm, {7: plan.Slots()}, {}, (7,))
    slot = executor.Qwen35Verify(lm, {7: plan.Slots(skv2=True)}, {}, (7,))
    # keyed by the runner that captures them; a runner without a key captures one graph
    assert stock.graph_variants == {}
    assert slot.graph_variants == {"target_verify": route.VERIFY}


# -------------------------------------------------------------------- slot mka


def _fp8_linear(log, name):
    """A static per-tensor FP8 ModelOpt linear (its method matched by name, as the branch does)."""
    method = type("ModelOptFp8LinearMethod", (), {"use_marlin": False})()
    linear = SimpleNamespace(
        quant_method=method,
        weight=torch.zeros(8, 4, dtype=torch.float8_e4m3fn),
        input_scale=torch.ones(1),
        weight_scale=torch.ones(1),
    )
    log.known.update(
        {f"{name}_sx": linear.input_scale, f"{name}_sw": linear.weight_scale}
    )
    return linear


def _mka_layer(log, attn, **kw):
    """A Qwen3.5 TP8 full-attention layer as slot mka reads it."""
    from sglang.srt.layers.rotary_embedding.mrope import MRotaryEmbedding

    rope = MRotaryEmbedding.__new__(MRotaryEmbedding)
    torch.nn.Module.__init__(rope)
    rope.__dict__.update(
        rotary_dim=64,
        cos_sin_cache=torch.zeros(16, 64),
        axis_map=torch.zeros(32, dtype=torch.int64),
    )
    layer = SimpleNamespace(
        attn=attn,
        attn_output_gate=True,
        num_heads=4,
        num_kv_heads=1,
        head_dim=256,
        hidden_size=4096,
        rotary_emb=rope,
        qkv_proj=_fp8_linear(log, "qkv"),
        o_proj=_fp8_linear(log, "o"),
        q_norm=SimpleNamespace(weight=torch.zeros(256), variance_epsilon=1e-6),
        k_norm=SimpleNamespace(weight=torch.zeros(256)),
    )
    layer.__dict__.update(kw)
    log.known.update(
        cos_sin=rope.cos_sin_cache,
        axis=rope.axis_map,
        qnw=layer.q_norm.weight,
        knw=layer.k_norm.weight,
    )
    return layer


@pytest.mark.parametrize("skv2", [False, True])
def test_the_mka_slot_runs_the_prologue_the_attention_and_the_epilogue(skv2):
    from sgl_nvidia_mega.qwen3_5.attention import mka, split_kv
    from sgl_nvidia_mega.qwen3_5.sm103.attention import mka as kernels

    log = _log()
    attn, st = _skv2_setup(log, k_scale=torch.full((1,), 0.75))
    log.known["k_scale"] = attn.k_scale
    layer = _mka_layer(log, attn)
    x = torch.zeros(7, 4096, dtype=torch.bfloat16)
    positions = torch.zeros(3, 7, dtype=torch.int64)
    st.positions, st.slots = positions, plan.Slots(skv2=skv2, mka=True)
    log.known.update(x=x, positions=positions, w_qkv="w_qkv", w_o="w_o")
    mka.WEIGHTS[7] = ("w_qkv", "w_o")
    attn._fn = lambda *a, **k: log.op("attention")(*a, **k) or torch.zeros(7, 4, 256)

    def prologue(**kw):
        log.known.update(q=kw["q_out"], gate=kw["gate_out"])
        log.append(("prologue", tuple(sorted((k, log.tag(v)) for k, v in kw.items()))))
        log.known["eps"] = kw["eps"]

    def epilogue(**kw):
        log.append(("epilogue", tuple(sorted((k, log.tag(v)) for k, v in kw.items()))))

    def core(attn_, q, st_):
        log.append(("skv2_core", log.tag(q)))
        return torch.zeros(7, 1024, dtype=torch.bfloat16)

    @contextlib.contextmanager
    def symmetric(group, disabled=False):
        log.append(("symmetric", group, disabled))
        yield

    with (
        patch.object(kernels, "prologue", prologue),
        patch.object(kernels, "epilogue", epilogue),
        patch.object(split_kv, "core", core),
        patch(
            "sglang.srt.distributed.device_communicators.pynccl_allocator.use_symmetric_memory",
            symmetric,
        ),
        patch("sglang.srt.layers.dp_attention.is_allocation_symmetric", lambda: True),
        patch(
            "sglang.srt.runtime_context.get_parallel",
            lambda: SimpleNamespace(tp_group="tp"),
        ),
    ):
        out = attention.step(layer, x, st)
    del mka.WEIGHTS[7]
    assert [e[0] for e in log] == [
        "prologue",
        "skv2_core" if skv2 else "attention",
        "symmetric",
        "epilogue",
    ]
    assert dict(log[0][1]) == {
        "x": "x(7, 4096)",
        "w_packed": "w_qkv",
        "sx": "qkv_sx(1,)",
        "sw": "qkv_sw(1,)",
        "q_norm_w": "qnw(256,)",
        "k_norm_w": "knw(256,)",
        "cos_sin": "cos_sin(16, 64)",
        "positions": "positions(3, 7)",
        "axis_map": "axis(32,)",
        # this layer's pool rows of the forward's view, at the step's slots
        "cache_loc": "loc(7,)",
        "k_cache": "k7(192, 1, 256)",
        "v_cache": "v7(192, 1, 256)",
        "k_scale": "k_scale(1,)",
        "v_scale": "v_scale(1,)",
        "q_out": "q(7, 1024)",
        "gate_out": "gate(7, 1024)",
        "nq": 4,
        "eps": "?",
    }
    assert log.known["eps"] == 1e-6  # the q norm's epsilon
    if skv2:
        assert log[1] == ("skv2_core", "q(7, 1024)")
    else:
        # SGLang's attention on the rows the prologue wrote: no K / V to write
        assert log[1][2:] == (("q(7, 1024)", None, None, "fb"), ())
    assert log[2] == ("symmetric", "tp", False)
    kw = dict(log[3][1])
    assert (kw["gate"], kw["w_packed"], kw["sx"], kw["sw"]) == (
        "gate(7, 1024)",
        "w_o",
        "o_sx(1,)",
        "o_sw(1,)",
    )
    assert kw["attn"] == "?(7, 1024)" and kw["nq"] == 4 and kw["site"] is None
    assert out.shape == (7, 4096) and out.dtype == torch.bfloat16


@pytest.mark.parametrize("consumes", [False, True])
def test_the_mka_push_slot_pushes_where_the_moe_input_only_consumes(consumes):
    from sgl_nvidia_mega.qwen3_5.attention import mka
    from sgl_nvidia_mega.qwen3_5.boundary import hop as boundary_hop
    from sgl_nvidia_mega.qwen3_5.sm103.attention import mka as kernels

    log = _log()
    attn, st = _skv2_setup(log, k_scale=torch.full((1,), 0.75))
    layer = _mka_layer(log, attn)
    x = torch.zeros(7, 4096, dtype=torch.bfloat16)
    st.positions = torch.zeros(3, 7, dtype=torch.int64)
    st.slots = plan.Slots(skv2=True, mka=True, mka_push=True, hop=True)
    hop = SimpleNamespace(
        site="site", mark_pushed=lambda t: log.append(("mark_pushed", log.tag(t)))
    )
    st.bound = {"boundary": SimpleNamespace(hop=hop)}
    mka.WEIGHTS[7] = ("w_qkv", "w_o")

    def epilogue(**kw):
        log.known["out"] = kw["out"]
        log.append(("epilogue", kw["site"]))

    def takes(hop_, layer_, st_, rows):
        log.append(("takes_moe_input", hop_ is hop, layer_ is layer, rows))
        return consumes

    @contextlib.contextmanager
    def symmetric(group, disabled=False):
        yield

    with (
        patch.object(kernels, "prologue", lambda **kw: None),
        patch.object(kernels, "epilogue", epilogue),
        patch(
            "sgl_nvidia_mega.qwen3_5.attention.split_kv.core",
            lambda *a: torch.zeros(7, 1024, dtype=torch.bfloat16),
        ),
        patch.object(boundary_hop, "takes_moe_input", takes),
        patch(
            "sglang.srt.distributed.device_communicators.pynccl_allocator.use_symmetric_memory",
            symmetric,
        ),
        patch("sglang.srt.layers.dp_attention.is_allocation_symmetric", lambda: True),
        patch(
            "sglang.srt.runtime_context.get_parallel",
            lambda: SimpleNamespace(tp_group="tp"),
        ),
    ):
        out = attention.step(layer, x, st)
    del mka.WEIGHTS[7]
    assert log[0] == ("takes_moe_input", True, True, 7)
    if consumes:
        # the epilogue pushes through the hop's site, and its all-reduce only consumes
        assert log[1:] == [("epilogue", "site"), ("mark_pushed", "out(7, 4096)")]
    else:
        assert log[1:] == [("epilogue", None)]
    assert out is log.known["out"]


@pytest.mark.parametrize(
    "kw, want",
    [
        (dict(), ""),
        (dict(attn_output_gate=False), "no attention output gate"),
        (dict(num_heads=2), "2 query / 1 KV heads a rank"),
        (dict(num_kv_heads=2), "4 query / 2 KV heads a rank"),
        (dict(head_dim=128), "head_dim 128, hidden 4096"),
        (dict(rotary_emb=SimpleNamespace(rotary_dim=64)), "no M-RoPE axis map"),
        (dict(rotary_dim=128), "rotary_dim 128 with a torch.float32 cos/sin cache"),
        (dict(cos_sin_dtype=torch.bfloat16), "rotary_dim 64 with a torch.bfloat16"),
        (dict(qkv_marlin=True), "not a static per-tensor FP8 linear"),
        (dict(o_scales=2), "not a static per-tensor FP8 linear"),
    ],
)
def test_the_mka_slot_names_what_a_layer_lacks(kw, want):
    from sgl_nvidia_mega.qwen3_5.attention import mka

    log = _log()
    special = {
        k: kw.pop(k)
        for k in list(kw)
        if k in ("rotary_dim", "cos_sin_dtype", "qkv_marlin", "o_scales")
    }
    layer = _mka_layer(log, None, **kw)
    if "rotary_dim" in special:
        layer.rotary_emb.rotary_dim = special["rotary_dim"]
    if "cos_sin_dtype" in special:
        layer.rotary_emb.cos_sin_cache = torch.zeros(
            16, 64, dtype=special["cos_sin_dtype"]
        )
    if "qkv_marlin" in special:
        layer.qkv_proj.quant_method.use_marlin = True
    if "o_scales" in special:
        layer.o_proj.weight_scale = torch.ones(special["o_scales"])
    got = mka.unsupported(layer)
    assert (want in got) if want else got == ""


def test_the_mka_slot_is_built_with_its_weights_packed_for_every_attention_layer(
    monkeypatch,
):
    from sgl_nvidia_mega.qwen3_5.attention import mka

    def layer(kind):
        made = kind.__new__(kind)
        torch.nn.Module.__init__(made)
        return made

    layers = [
        layer(qwen3_5.Qwen3_5LinearDecoderLayer),
        layer(qwen3_5.Qwen3_5AttentionDecoderLayer),
        layer(qwen3_5.Qwen3_5AttentionDecoderLayer),
    ]
    model = SimpleNamespace(
        is_mrope_enabled=True,
        model=SimpleNamespace(start_layer=0, end_layer=3, layers=layers),
    )
    packed, why = [], {}
    monkeypatch.setattr(runtime, "SLOTS", frozenset({"mka"}))
    monkeypatch.setattr(mka, "unsupported", lambda l: why.get(id(l), ""))
    monkeypatch.setattr(mka, "pack", packed.append)
    ctx = SimpleNamespace(tokens_per_request=7)
    assert attention.build(model, ctx) == attention.Bound(mka=True)
    assert packed == layers[1:]
    # a layer the slot cannot serve leaves the slot unbuilt, packs nothing more
    why[id(layers[2])] = "no attention output gate"
    assert attention.build(model, ctx) == attention.Bound(
        reason="layer 2: no attention output gate"
    )
    assert len(packed) == 2
    model.is_mrope_enabled = False
    assert attention.build(model, ctx).reason == (
        "layer 1: no M-RoPE positions (MKA reads them)"
    )


def test_the_mka_slot_names_the_pools_it_cannot_write():
    from sgl_nvidia_mega.qwen3_5.attention import mka

    def kv(*bufs):
        return SimpleNamespace(k=bufs, v=bufs)

    assert (
        mka.pool_reasons(kv(torch.zeros(64, 1, 256, dtype=torch.float8_e4m3fn))) == []
    )
    assert mka.pool_reasons(kv(torch.zeros(64, 2, 256, dtype=torch.float8_e4m3fn))) == [
        "a torch.float8_e4m3fn KV pool buffer of strides (512, 256, 1)"
    ]
    assert "torch.float32" in mka.pool_reasons(kv(torch.zeros(64, 1, 256)))[0]


class _Exit:
    def __init__(self, log, defer):
        self.log, self.defer_moe_finalize = log, defer

    def __enter__(self):
        self.log.append(("ffn_exit.enter",))
        return self

    def __exit__(self, *exc):
        self.log.append(("ffn_exit.exit",))

    def finish(self, out):
        self.log.append(("ffn_exit.finish", self.log.tag(out)))
        return "layer_out"


class _MoeBlock(qwen2_moe.Qwen2MoeSparseMoeBlock):
    def __init__(self, log):
        torch.nn.Module.__init__(self)
        self.__dict__["log"] = log

    def forward(self, x, forward_batch=None, defer_finalize=False):
        self.log.append(("moe", self.log.tag(x), defer_finalize))
        return "moe_out"


def _decoder_layer(kind, log, defer):
    layer = kind.__new__(kind)
    torch.nn.Module.__init__(layer)
    a_in, m_in = torch.zeros(7, 4), torch.zeros(7, 4)
    log.known.update(a_in=a_in, m_in=m_in)
    layer.mlp = _MoeBlock(log)
    layer.__dict__.update(
        attn_boundary=SimpleNamespace(
            prepare=log.op("attn.prepare", a_in), finish=log.op("attn.finish", "h1")
        ),
        ffn_boundary=SimpleNamespace(
            prepare=log.op("ffn.prepare", m_in), exit=lambda fb: _Exit(log, defer)
        ),
        linear_attn=lambda x, fb: log.append(("mixer", log.tag(x))) or "part",
        self_attention=lambda positions, hidden_states, forward_batch: (
            log.append(("mixer", log.tag(hidden_states))) or "part"
        ),
    )
    return layer


@pytest.mark.parametrize(
    "kind", [qwen3_5.Qwen3_5LinearDecoderLayer, qwen3_5.Qwen3_5AttentionDecoderLayer]
)
@pytest.mark.parametrize("defer", [False, True])
def test_a_layer_runs_what_sglangs_decoder_layer_runs(kind, defer):
    batch = SimpleNamespace(forward_mode=SimpleNamespace(is_idle=lambda: False))
    runs = []
    for which in ("sglang", "plugin"):
        log = _log(hidden="hidden", batch=batch, positions="pos", h1="h1", part="part")
        layer = _decoder_layer(kind, log, defer)
        if which == "sglang":
            out = layer.forward(
                positions="pos",
                hidden_states="hidden",
                forward_batch=batch,
                captured_last_layer_outputs=None,
            )
        else:
            st = SimpleNamespace(fb=batch, positions="pos", slots=plan.Slots())
            built = _verify()
            built.mixers = {
                0: lambda layer, x, st: log.append(("mixer", log.tag(x))) or "part"
            }
            with patch.object(
                executor.moe,
                "step",
                lambda layer, x, st, defer: (
                    log.append(("moe", log.tag(x), defer)) or "moe_out"
                ),
            ):
                out = built._layer(0, layer, "hidden", st, None)
        runs.append((log, out))
    (want_log, want), (got_log, got) = runs
    assert got_log == want_log and want == got == "layer_out"


def test_the_layer_loop_opens_the_stream_scopes_each_layer_and_closes_it(monkeypatch):
    log = []
    layers = [
        kind.__new__(kind)
        for kind in (
            qwen3_5.Qwen3_5LinearDecoderLayer,
            qwen3_5.Qwen3_5AttentionDecoderLayer,
        )
    ]
    for layer in layers:
        torch.nn.Module.__init__(layer)
    model = SimpleNamespace(start_layer=0, end_layer=2, layers=layers)
    lm = SimpleNamespace(
        model=model, is_mrope_enabled=True, capture_aux_hidden_states=False
    )
    built = executor.Qwen35Verify(lm, {7: plan.Slots()}, {}, (7,))
    assert built.mixers == {0: gdn.step, 1: attention.step}
    built.mixers = {i: (lambda i: lambda *a: f"part{i}")(i) for i in range(2)}

    @contextlib.contextmanager
    def scope(i):
        log.append(("layer", i))
        yield

    recorder = SimpleNamespace(with_current_layer=scope)
    monkeypatch.setattr(
        "sglang.srt.eplb.expert_distribution.get_global_expert_distribution_recorder",
        lambda: recorder,
    )
    monkeypatch.setattr(executor.embedding, "embed", lambda st: "embeds")
    monkeypatch.setattr(
        executor.boundary, "open_stream", lambda st: log.append(("open",))
    )
    monkeypatch.setattr(
        executor.boundary,
        "close_stream",
        lambda st, h: log.append(("close", h)) or "final",
    )
    monkeypatch.setattr(
        executor.Qwen35Verify,
        "_layer",
        lambda self, i, layer, h, st, capture: (
            log.append(("run", i, h, capture)) or f"h{i}"
        ),
    )
    monkeypatch.setattr(
        executor.logits, "logits", lambda st, h, aux: (st.positions, st.slots, h, aux)
    )
    batch = SimpleNamespace(mrope_positions="mrope")
    out = built.forward(Step("target_verify", 7), None, "ids", "pos", batch)
    assert out == ("mrope", plan.Slots(), "final", None)
    assert log == [
        ("open",),
        ("layer", 0),
        ("run", 0, "embeds", None),
        ("layer", 1),
        ("run", 1, "h0", None),
        ("close", "h1"),
    ]


# ------------------------------------------------------------------------- slot hop


class _Fused:
    """SGLang's CuTe DSL workspace, or the hop: the same outputs from the same operands
    (fresh tensors), and a log of the operands."""

    def __init__(self, log, rows=7):
        self.log, self.rows = log, rows

    def supports(self, m):
        return m <= self.rows

    def finalize(self, handoff, residual, gamma):
        return self.moe_finalize_all_reduce_rms_norm(
            routed_output=handoff.routed_output,
            expert_weights=handoff.expert_weights,
            permuted_indices=handoff.permuted_indices,
            gated_shared_output=handoff.gated_shared_output,
            residual=residual,
            gamma=gamma,
        )

    def moe_finalize_all_reduce_rms_norm(self, *, residual, gamma, **rows):
        self.log.append(("finalize", *rows.values(), residual, gamma))
        return residual + gamma, residual * 2

    def all_reduce_residual_rms_norm(self, *, local_contribution, residual, gamma):
        self.log.append(("all_reduce", local_contribution, residual, gamma))
        return residual + gamma, residual * 2


@pytest.fixture
def cutedsl_gates(monkeypatch):
    """SGLang's CuTe DSL eligibility as at TP8 for a verify step: the workspace's rows
    (``gates["open"] = False`` closes it)."""
    from sglang.srt.layers.layer_boundary.fusions import cutedsl
    from sglang.test.boundary_fixtures import parallel_of

    gates = {"open": True}
    monkeypatch.setattr(
        cutedsl.CuteDSLFusion,
        "_eligible",
        lambda self, plan, fb, m: gates["open"] and self.service.supports(m),
    )
    monkeypatch.setattr(
        cutedsl, "get_parallel", lambda: parallel_of(attn_dp=1, attn_tp=8)
    )
    comm = SimpleNamespace(comm=SimpleNamespace(enable_quant_communications=False))
    monkeypatch.setattr(cutedsl, "get_exec", lambda: comm)
    monkeypatch.setattr("sglang.srt.runtime_context.get_exec", lambda: comm)
    return gates


def _hop_layer(service, fusions=True):
    """A Qwen3.5 MoE layer's two stages as SGLang binds them at TP8, with the CuTe DSL
    fusion on ``service`` unless not ``fusions``."""
    from sglang.srt.layers.layer_boundary import declare_attn, declare_ffn
    from sglang.srt.layers.layer_boundary.fusions import cutedsl
    from sglang.srt.layers.layer_boundary.residual.add_norm import (
        Fp8Input,
        NormQuantReadout,
    )
    from sglang.srt.layers.layernorm import GemmaRMSNorm
    from sglang.test.boundary_fixtures import build_stages, parallel_of, planning

    fusion = cutedsl.CuteDSLFusion() if fusions else None
    if fusions:
        fusion.install(service, defers_finalize=True, terminal_finalize=True)
    moe_ffn = dict(sparse=True, next_layer_sparse=True)
    parallel = parallel_of(attn_dp=1, attn_tp=8)
    with planning(parallel), patch.object(cutedsl, "get_parallel", lambda: parallel):
        attn, ffn = build_stages(
            (
                declare_attn(read=NormQuantReadout(fp8_input=Fp8Input.TUPLE_AND_BF16)),
                GemmaRMSNorm(16),
                {"fusions": fusion},
            ),
            (declare_ffn(**moe_ffn), GemmaRMSNorm(16), {"fusions": fusion}),
            previous=declare_ffn(**moe_ffn),
        )
    return SimpleNamespace(attn_boundary=attn, ffn_boundary=ffn, layer_id=3)


def _handoff(rows=7, weights=torch.bfloat16):
    from sglang.srt.layers.layer_boundary.fusions.cutedsl import MoeDeferredFinalize

    return MoeDeferredFinalize(
        routed_output=torch.randn(rows * 10, 16).bfloat16(),
        expert_weights=torch.rand(rows, 10).to(weights),
        permuted_indices=torch.arange(rows * 10, dtype=torch.int32).view(rows, 10),
        gated_shared_output=torch.randn(rows, 16).bfloat16(),
        m=rows,
        finish=lambda: pytest.fail("the handoff was completed unfused"),
    )


def _verify_batch(stream, rows=7):
    from sglang.srt.model_executor.forward_batch_info import ForwardMode

    return SimpleNamespace(
        forward_mode=ForwardMode.TARGET_VERIFY,
        input_ids=torch.zeros(rows),
        residual_stream=stream,
    )


def _same_calls(a, b):
    """The same calls on the same tensors (by identity)."""
    assert len(a) == len(b) == 1
    assert a[0][0] == b[0][0] and all(x is y for x, y in zip(a[0][1:], b[0][1:]))


def test_out_proj_packs_into_the_push_gemms_fragments():
    from sgl_nvidia_mega.qwen3_5.sm103 import hop

    w = torch.arange(4096 * 1024, dtype=torch.int64).remainder(251).to(torch.uint8)
    packed = hop.pack_out_proj(w.view(4096, 1024).view(torch.float8_e4m3fn))
    assert packed.dtype == torch.uint8 and packed.numel() == 4096 * 1024
    # CTA 0, warp 0, k-step 0, block 0, lane 0: rows 0 and 8, K bytes 0-3 and 16-19
    frag = packed[:16].view(4, 4)
    want = w.view(4096, 1024)
    assert torch.equal(frag[0], want[0, 0:4]) and torch.equal(frag[1], want[8, 0:4])
    assert torch.equal(frag[2], want[0, 16:20]) and torch.equal(frag[3], want[8, 16:20])
    with pytest.raises(ValueError, match="out_proj packing"):
        hop.pack_out_proj(torch.zeros(4096, 512, dtype=torch.float8_e4m3fn))


def _pushing_hop(log):
    """sm103's Hop without its mailbox: the launches it would issue, logged."""
    from sgl_nvidia_mega.qwen3_5.sm103 import hop as sm103_hop

    hop = object.__new__(sm103_hop.Hop)
    hop.max_m, hop._pushed = 8, None
    hop._args = lambda m, residual, gamma, scale: (SimpleNamespace(m=m), ("out",))
    hop._launch = lambda name, args, ctas, cluster=None: log.append(name)
    hop._consume = lambda args: log.append("consume")
    return hop


def test_a_pushed_partials_all_reduce_only_consumes():
    log = []
    hop = _pushing_hop(log)
    pushed, other = torch.zeros(7, 16), torch.zeros(7, 16)
    hop.mark_pushed(pushed)
    assert hop.has_pushed(pushed) and not hop.has_pushed(other)
    for partial in (pushed, other):  # the push consumed, the next one publishes
        hop.all_reduce_residual_rms_norm(
            local_contribution=partial, residual="r", gamma="g"
        )
    assert log == ["consume", "q35hop_publish", "consume"]
    assert hop._pushed is None


def test_a_push_left_unconsumed_stops_the_next_hop():
    hop = _pushing_hop([])
    pushed, other = torch.zeros(7, 16), torch.zeros(7, 16)
    hop.mark_pushed(pushed)
    with pytest.raises(RuntimeError, match="never consumed"):
        hop.mark_pushed(other)
    with pytest.raises(RuntimeError, match="never consumed"):
        hop.all_reduce_residual_rms_norm(
            local_contribution=other, residual="r", gamma="g"
        )
    with pytest.raises(RuntimeError, match="never consumed"):
        hop.moe_finalize_all_reduce_rms_norm(
            routed_output=None,
            expert_weights=None,
            permuted_indices=torch.zeros(7, 10, dtype=torch.int32),
            gated_shared_output=None,
            residual="r",
            gamma="g",
        )


@pytest.mark.parametrize("point", ["layer input", "MoE input", "final norm"])
def test_the_hop_slot_runs_where_sglang_runs_its_cutedsl_kernel(cutedsl_gates, point):
    """At each point, SGLang's stage boundary with the CuTe DSL workspace and the
    slot's mirror with the hop make the same call on the same tensors and leave the
    residual stream alike."""
    from sglang.srt.layers.layer_boundary import PLAIN_ADD
    from sglang.srt.layers.layer_boundary.residual import batch as residual_batch
    from sglang.srt.layers.layer_boundary.residual.stream import ResidualStream
    from sglang.srt.layers.layernorm import GemmaRMSNorm

    logs = {"sglang": [], "plugin": []}
    service, hop = _Fused(logs["sglang"]), _Fused(logs["plugin"])
    layer, owed = _hop_layer(service), _handoff()
    residual, part = torch.randn(7, 16).bfloat16(), torch.randn(7, 16).bfloat16()
    final = GemmaRMSNorm(16)
    runs = {}
    for run in ("sglang", "plugin"):
        stream = ResidualStream(residual)
        fb = _verify_batch(stream)
        st = SimpleNamespace(fb=fb, model=SimpleNamespace(norm=final))
        if point == "MoE input":
            handle = layer.attn_boundary.finish(part, fb)
            x = (
                layer.ffn_boundary.prepare(handle, fb)
                if run == "sglang"
                else boundary_hop.moe_input(hop, layer, handle, st)
            )
        else:
            handle = stream.record(owed, PLAIN_ADD)
            if point == "layer input":
                x = (
                    layer.attn_boundary.prepare(handle, fb)
                    if run == "sglang"
                    else boundary_hop.layer_input(hop, layer, handle, st)
                )
            else:
                x = (
                    residual_batch.final_norm(
                        handle, fb, final, finalize_norm=service, skip_empty=True
                    )
                    if run == "sglang"
                    else boundary_hop.final_norm(hop, handle, st)
                )
        runs[run] = (x, fb.residual_stream, stream.residual, stream.pending)
    _same_calls(logs["sglang"], logs["plugin"])
    (x, stream, written, pending), (y, stream2, written2, pending2) = runs.values()
    assert torch.equal(x, y) and pending is pending2 is None
    if point == "final norm":
        assert stream is stream2 is None
    else:
        assert torch.equal(written, written2) and written is not residual


@pytest.mark.parametrize("why", ["sglang declines", "fp32 weights", "too many rows"])
def test_the_hop_slot_leaves_to_sglang_a_finalize_it_cannot_stand_in_for(
    cutedsl_gates, why
):
    from sglang.srt.layers.layer_boundary import PLAIN_ADD
    from sglang.srt.layers.layer_boundary.residual.stream import ResidualStream

    log = []
    layer = _hop_layer(_Fused([]))
    cutedsl_gates["open"] = why != "sglang declines"
    owed = _handoff(weights=torch.float32 if why == "fp32 weights" else torch.bfloat16)
    hop = _Fused(log, rows=6 if why == "too many rows" else 7)
    stream = ResidualStream(torch.zeros(7, 16))
    handle = stream.record(owed, PLAIN_ADD)
    st = SimpleNamespace(fb=_verify_batch(stream), model=SimpleNamespace(norm=None))
    assert boundary_hop.layer_input(hop, layer, handle, st) is None
    if why != "sglang declines":
        assert boundary_hop.final_norm(hop, handle, st) is None
    assert log == [] and stream.pending.owed is owed


def test_the_hop_slot_leaves_to_sglang_a_mixer_sum_it_declines(cutedsl_gates):
    from sglang.srt.layers.layer_boundary.residual.stream import ResidualStream

    log = []
    layer = _hop_layer(_Fused([]))
    cutedsl_gates["open"] = False
    stream = ResidualStream(torch.zeros(7, 16))
    st = SimpleNamespace(fb=_verify_batch(stream))
    handle = layer.attn_boundary.finish(torch.ones(7, 16).bfloat16(), st.fb)
    assert boundary_hop.moe_input(_Fused(log), layer, handle, st) is None
    assert log == [] and stream.pending is not None


def test_the_hop_slot_mirrors_the_paths_sglang_binds_for_qwen3_5():
    assert boundary_hop.path_refusals(_hop_layer(SimpleNamespace())) == []
    assert boundary_hop.path_refusals(_hop_layer(None, fusions=False)) == [
        "its attention input runs no CuTe DSL fusion",
        "its ffn input runs no CuTe DSL fusion",
    ]


def test_the_hop_slot_names_what_it_cannot_serve():
    service = SimpleNamespace(hidden_size=4096, top_k=10)
    layers = [_hop_layer(SimpleNamespace()), _hop_layer(None, fusions=False)]
    m = SimpleNamespace(
        flashinfer_mnnvl_cutedsl_fusion=service,
        layers=layers,
        start_layer=0,
        end_layer=1,
    )
    assert boundary_hop.refusals(m, 8) == []
    assert boundary_hop.refusals(m, 16) == ["TP 16 (the hop serves TP (2, 4, 8))"]
    service.hidden_size = 5120
    assert boundary_hop.refusals(m, 8) == [
        "hidden 5120, top-10 (the hop serves hidden 4096, up to top-16)"
    ]
    service.hidden_size, m.end_layer = 4096, 2
    assert boundary_hop.refusals(m, 8) == [
        "layer 3: its attention input runs no CuTe DSL fusion (and 1 more)"
    ]
    m.flashinfer_mnnvl_cutedsl_fusion = None
    assert boundary_hop.refusals(m, 8) == [
        "SGLang runs no FlashInfer MNNVL CuTe DSL fusion"
    ]


@pytest.mark.parametrize("taken", [False, True])
def test_with_slot_hop_each_point_falls_back_to_sglang_where_the_hop_declines(
    monkeypatch, taken
):
    from sgl_nvidia_mega.qwen3_5 import boundary

    log = []
    for point in ("layer_input", "moe_input", "final_norm"):
        monkeypatch.setattr(
            boundary_hop,
            point,
            lambda hop, *a, point=point: (
                log.append((point, hop)) or (point if taken else None)
            ),
        )
    monkeypatch.setattr(
        "sglang.srt.layers.layer_boundary.residual.batch.final_norm",
        lambda *a, **k: "sglang final norm",
    )
    layer = SimpleNamespace(
        attn_boundary=SimpleNamespace(
            prepare=lambda h, fb, capture_gathered: "sglang layer input",
            finish=lambda part, fb: "handle",
        ),
        ffn_boundary=SimpleNamespace(prepare=lambda h, fb: "sglang MoE input"),
    )
    st = SimpleNamespace(
        fb="fb",
        slots=plan.Slots(hop=True),
        bound={"boundary": boundary_hop.Bound(hop="the hop")},
        model=SimpleNamespace(norm="norm", flashinfer_mnnvl_cutedsl_fusion=None),
    )
    got = (
        boundary.pre(layer, "h", st),
        boundary.post_attn(layer, "part", st),
        boundary.close_stream(st, "h"),
    )
    if taken:
        assert got == ("layer_input", "moe_input", "final_norm")
    else:
        assert got == ("sglang layer input", "sglang MoE input", "sglang final norm")
    assert log == [(p, "the hop") for p in ("layer_input", "moe_input", "final_norm")]
    # an auxiliary capture reads SGLang's own prepare
    assert boundary.pre(layer, "h", st, capture=[]) == "sglang layer input"
