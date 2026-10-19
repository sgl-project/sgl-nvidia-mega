# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""The MoE step and the mHC boundaries on CPU, against SGLang's own MoE forward and
stage boundary: the same ops in the same order with the same arguments, the routed
pair's and hops' gates, and the routed-expert GEMM's host-side bounds."""

import importlib
import math
from contextlib import contextmanager
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest
import torch

mhc = pytest.importorskip("sglang.srt.models.deepseek_v4_mhc")

from sglang.srt.models import deepseek_v2  # noqa: E402

from sgl_nvidia_mega.deepseek_v41 import moe  # noqa: E402
from sgl_nvidia_mega.deepseek_v41.boundary.boundaries import Boundaries  # noqa: E402
from sgl_nvidia_mega.deepseek_v41.moe.step import dual_stream  # noqa: E402
from sgl_nvidia_mega.deepseek_v41.sm103.moe import (  # noqa: E402
    routed_expert_gemm as reg,
)

# ------------------------------------------------------------------------------------
# The stock MoE step on CPU, against SGLang's own ``DeepseekV2MoE.forward`` with
# ``leave_output_sum`` on the same stubbed block: the same ops in the same order, on the
# same streams, with the same arguments, and the same pieces out, in the eager and the
# CUDA graph capture branches.


class _Stream:
    def __init__(self, name, log):
        self.name, self.log = name, log

    def wait_stream(self, other):
        self.log.append(("wait", self.name, other.name))

    def record_event(self):
        self.log.append(("record", self.name))
        return f"event:{self.name}"


def _block(log, *, inplace, quant_once, quant_routed, vision=False, hash_=False):
    """A DeepseekV2MoE whose gate, top-k, experts and shared expert log their calls."""
    cur = {"stream": "main"}
    x = torch.arange(12.0).reshape(3, 4)

    def tag(v):
        return "x" if v is x else ("tensor" if isinstance(v, torch.Tensor) else v)

    def op(name, out):
        def run(*args, **kwargs):
            kw = tuple(sorted((k, tag(v)) for k, v in kwargs.items()))
            log.append((name, cur["stream"], tuple(map(tag, args)), kw))
            return out

        return run

    mlp = deepseek_v2.DeepseekV2MoE.__new__(deepseek_v2.DeepseekV2MoE)
    torch.nn.Module.__init__(mlp)
    topk_out = SimpleNamespace(format="standard")
    quant = SimpleNamespace(quantize_routed_input=op("quantize_routed", ("xq", "xsf")))
    experts = SimpleNamespace(
        moe_runner_config=SimpleNamespace(inplace=inplace),
        supports_deferred_finalize=True,
        quant_method=quant,
        forward_deferred_finalize=op("experts_deferred", "routed_d"),
        forward_impl=op("experts_impl", "routed_i"),
        w13_weight=torch.zeros(1, 8, 2, dtype=torch.uint8),
    )
    experts_call = op("experts", "routed")
    mlp.__dict__.update(
        alt_stream=_Stream("alt", log),
        routed_quant_stream=_Stream("rq", log),
        num_fused_shared_experts=0,
        _enable_a2a_moe=False,
        _fuse_shared_experts_inside_sbo=False,
        _shared_expert_tp1=False,
        _deferred_finalize_max_tokens=0,
        _moe_quant_once=None,
        routed_scaling_factor=1.5,
        reduce_results=False,
        layer_id=7,
        is_nextn=False,
        is_hash=hash_,
    )
    mlp.__dict__["gate"] = _Callable(
        op("gate", "logits"), e_score_correction_bias_vl=1 if vision else None
    )
    mlp.__dict__["topk"] = _Callable(
        op("topk", topk_out),
        topk_config="cfg",
        empty_topk_output=op("empty_topk", topk_out),
    )
    mlp.__dict__["experts"] = _Callable(experts_call, **experts.__dict__)
    mlp.__dict__["shared_experts"] = _Callable(
        op("shared", "shared_out"), gate_up_proj=SimpleNamespace()
    )
    mlp.__dict__["_maybe_quant_moe_input_once"] = op(
        "quant_once", ("q", "s") if quant_once else None
    )
    mlp.__dict__["_should_quant_routed_input_mxfp8"] = op(
        "should_quant_routed", quant_routed
    )
    return mlp, x, cur


class _Callable:
    def __init__(self, fn, **attrs):
        self._fn = fn
        self.__dict__.update(attrs)

    def __call__(self, *args, **kwargs):
        return self._fn(*args, **kwargs)


@contextmanager
def _sglang(log, cur, *, capture, defer_finalize=False):
    main = _Stream("main", log)

    @contextmanager
    def on(stream):
        prev, cur["stream"] = cur["stream"], stream.name
        try:
            yield
        finally:
            cur["stream"] = prev

    fwd = SimpleNamespace(
        flashinfer_trtllm_bypass=False,
        defer_moe_finalize=defer_finalize,
        mlp_reduce_scatter=False,
    )
    exec_ = SimpleNamespace(moe=SimpleNamespace(enable_eplb=False))
    with (
        patch.object(torch.cuda, "current_stream", return_value=main),
        patch.object(torch.cuda, "stream", side_effect=on),
        patch.object(deepseek_v2, "get_forward", return_value=fwd),
        patch.object(deepseek_v2, "get_exec", return_value=exec_),
        patch.object(deepseek_v2, "_is_cuda", True),
        patch.object(deepseek_v2, "_is_hip", False),
        patch.object(deepseek_v2, "get_is_capture_mode", return_value=capture),
        patch.object(deepseek_v2, "is_in_breakable_cuda_graph", return_value=False),
        patch(
            "sglang.srt.model_executor.runner.get_is_capture_mode", return_value=capture
        ),
        patch("sglang.srt.runtime_context.get_forward", return_value=fwd),
        patch("sglang.srt.layers.moe.mega_moe.should_use_mega_moe", return_value=False),
        patch.object(
            deepseek_v2.DeepseekV2MoE, "_can_dual_stream_graph", return_value=False
        ),
    ):
        yield


def _run_both(capture, defers_sum, skip_shared=False, **block):
    runs = []
    for which in ("sglang", "plugin"):
        log = []
        mlp, x, cur = _block(log, **block)
        batch = SimpleNamespace(moe_num_token_non_padded=lambda: "nnp")
        with _sglang(log, cur, capture=capture):
            if which == "sglang":
                out = mlp.forward(
                    x,
                    batch,
                    input_ids="ids",
                    input_ids_global="ids_g",
                    skip_shared_experts=skip_shared,
                    return_moe_output=defers_sum,
                    leave_output_sum=True,
                )
            else:
                transport = SimpleNamespace(
                    input_ids="ids",
                    input_ids_global="ids_g",
                    do_shared_local=skip_shared,
                    return_moe_output=defers_sum,
                )
                st = SimpleNamespace(
                    fb=batch,
                    transport=transport,
                    slots=SimpleNamespace(routed=False, fast_gate=False),
                    bound={"moe": None},
                )
                out = moe.step(SimpleNamespace(mlp=mlp), x, st)
        runs.append((log, out))
    return runs


@pytest.mark.parametrize("capture", [False, True])
@pytest.mark.parametrize("defers_sum", [False, True])
@pytest.mark.parametrize(
    "block",
    [
        dict(inplace=False, quant_once=False, quant_routed=False),
        dict(inplace=True, quant_once=True, quant_routed=False),
        dict(inplace=False, quant_once=False, quant_routed=True),
        dict(inplace=False, quant_once=False, quant_routed=False, vision=True),
        dict(inplace=False, quant_once=False, quant_routed=False, hash_=True),
        dict(inplace=True, quant_once=True, quant_routed=False, skip_shared=True),
    ],
)
def test_the_stock_step_issues_what_sglang_issues(capture, defers_sum, block):
    if block.get("vision"):
        pytest.importorskip("sglang.srt.multimodal.dsv41.vl_routing")
    with (
        patch("sglang.srt.multimodal.dsv41.vl_routing.vision_topk") as vt,
        patch.object(deepseek_v2, "vision_topk", vt),
    ):
        vt.side_effect = lambda *a, **k: SimpleNamespace(format="standard")
        (want_log, want), (got_log, got) = _run_both(capture, defers_sum, **block)
    assert got_log == want_log
    assert type(got) is type(want)
    assert got._replace(experts=None) == want._replace(experts=None)


def test_the_branch_follows_capture_mode():
    log = []
    mlp, x, cur = _block(log, inplace=False, quant_once=False, quant_routed=False)
    for capture in (False, True):
        with _sglang(log, cur, capture=capture):
            assert dual_stream(mlp, 3) is capture
            assert not dual_stream(mlp, 0)


@pytest.mark.parametrize("capture", [False, True])
@pytest.mark.parametrize("fast_gate", [False, True])
def test_the_routed_pair_replaces_only_the_top_k_and_the_routed_call(
    capture, fast_gate, caplog, monkeypatch
):
    # the module, not moe.step (the package's step function)
    step_mod = importlib.import_module("sgl_nvidia_mega.deepseek_v41.moe.step")
    monkeypatch.setattr(step_mod, "_FAST_GATE_SERVED", set())
    log = []
    mlp, x, cur = _block(log, inplace=False, quant_once=False, quant_routed=False)

    def logged(name, out):
        def run(*args):
            log.append((name, cur["stream"]))
            return out

        return run

    slot = SimpleNamespace(
        gate=True,
        select=logged("fast_gate", "gate_topk"),
        run=logged("pair", "deferred"),
        shared=None,
    )
    st = SimpleNamespace(
        fb=SimpleNamespace(moe_num_token_non_padded=lambda: "nnp"),
        transport=SimpleNamespace(
            input_ids_global="ids_g", do_shared_local=False, return_moe_output=True
        ),
        slots=SimpleNamespace(routed=True, fast_gate=fast_gate),
        bound={"moe": SimpleNamespace(slots={7: slot})},
    )
    with _sglang(log, cur, capture=capture), caplog.at_level("INFO"):
        out = moe.step(SimpleNamespace(mlp=mlp, layer_id=7), x, st)
    names = [e[:2] if e[0] in ("fast_gate", "pair") else e[0] for e in log]
    # engagement from the dispatch itself, as merge.sh's REQUIRE lines read it
    served = [r.getMessage() for r in caplog.records if "fast gate" in r.getMessage()]
    rows = f"NVIDIA megakernel: fast gate top-k serves {x.shape[0]} rows"
    assert served == ([rows] if fast_gate else [])
    assert ("pair", "main") in names and "experts" not in names
    assert (("fast_gate", "main") in names) is fast_gate
    assert ("topk" in names) is not fast_gate
    assert ("shared", "alt", ("x",), (("gemm_output_zero_allocator", None),)) in log
    assert out.routed == "deferred" and out.shared == "shared_out"


@pytest.mark.parametrize("fast_gate", [False, True])
@pytest.mark.parametrize("ready", [None, "event"])
def test_the_slot_calls_the_pair_with_its_signature(fast_gate, ready):
    from unittest.mock import create_autospec

    from sgl_nvidia_mega.deepseek_v41.moe import routed
    from sgl_nvidia_mega.deepseek_v41.sm103.moe.routed_expert_gemm import (
        RoutedExpertGemm,
    )

    gemm = create_autospec(RoutedExpertGemm, instance=True)
    gemm.top_k = 6
    pair_out = (
        torch.zeros(3, 4),
        torch.zeros(3, 6),
        torch.zeros(18, dtype=torch.int32),
    )
    gemm.supports_gate_topk.return_value = True
    config = SimpleNamespace(
        routed_scaling_factor=1.5,
        apply_routed_scaling_factor_on_output=True,
        correction_bias_for_dtype=lambda dtype: torch.zeros(384, dtype=dtype),
    )
    mlp = SimpleNamespace(
        topk=SimpleNamespace(topk_config=config),
        gate=SimpleNamespace(e_score_correction_bias_vl=None),
        experts=SimpleNamespace(quant_method=None),
    )
    slot = routed.MoeSlot(mlp, gemm, gate=True)
    topk = (
        slot.select(torch.zeros(3, 384), None, None, False)
        if fast_gate
        else SimpleNamespace(topk_ids="ids", topk_weights="w")
    )
    order = []
    stream = SimpleNamespace(wait_event=lambda e: order.append(("join", e)))
    gemm.gate_topk_and_gemms.side_effect = lambda *a, **k: (
        order.append(("gate", k.get("input_ready"))) or (*pair_out, None)
    )
    gemm.side_effect = lambda *a, **k: order.append("pair") or pair_out
    with patch.object(torch.cuda, "current_stream", return_value=stream):
        out = slot.run(torch.zeros(3, 4), topk, ("xq", "xs", ready))
    # the slot joins the routed input before the top-k in both cases
    first = ("gate", None) if fast_gate else "pair"
    assert order == ([("join", ready)] if ready else []) + [first]
    if fast_gate:
        assert gemm.gate_topk_and_gemms.call_args.args[5] is None  # no fill kernel
    assert out.top_k == 6 and out.gemm2_out is pair_out[0]


# ------------------------------------------------------------------------------------
# Slot shared_ll: the shared expert's gate_up on the low-latency GEMM, from the routed
# input quant, then SGLang's clamped SwiGLU and down_proj.


def _shared_mlp(log, n=256, k=256, **overrides):
    def down_proj(h):
        log.append(("down_proj", tuple(h.shape)))
        return "shared_out", None

    gate_up = SimpleNamespace(
        weight=torch.zeros(n, k, dtype=torch.float8_e4m3fn),
        weight_scale_inv_swizzled=torch.zeros(n * k // 32, dtype=torch.uint8),
    )
    shared = SimpleNamespace(
        swiglu_limit=10.0,
        use_fused_clamp_act_mul=False,
        allow_fused_down=True,
        gate_up_proj=gate_up,
        down_proj=_Callable(
            down_proj, weight=torch.zeros(k, n // 2, dtype=torch.float8_e4m3fn)
        ),
    )
    shared.__dict__.update(overrides)
    return SimpleNamespace(shared_experts=shared, num_fused_shared_experts=0)


def test_shared_ll_runs_sglangs_clamped_mlp_on_its_own_quant(monkeypatch):
    from sglang.kernels.ops.moe import dsv4

    from sgl_nvidia_mega.deepseek_v41.moe import shared_ll

    log = []
    layer = shared_ll.SharedLL(_shared_mlp(log), max_rows=48)
    x = torch.zeros(12, 256, dtype=torch.bfloat16)
    gate_up_out = torch.zeros(12, 256, dtype=torch.bfloat16)
    monkeypatch.setattr(
        shared_ll,
        "mxfp8_row_tiles",
        lambda rows: log.append(("quant", rows is x)) or ("q", "s"),
    )
    monkeypatch.setattr(
        shared_ll.mxfp8_ll,
        "linear",
        lambda a, a_scale, w, w_scale: (
            log.append(("gate_up", a, a_scale)) or gate_up_out
        ),
    )
    monkeypatch.setattr(
        dsv4,
        "silu_and_mul_clamp",
        lambda x, out, limit: log.append(("clamp", x is gate_up_out, out.shape, limit)),
    )
    assert layer(x) == "shared_out"
    assert log == [
        ("quant", True),
        ("gate_up", "q", "s"),
        ("clamp", True, (12, 128), 10.0),
        ("down_proj", (12, 128)),
    ]


def test_shared_ll_applies_only_to_sglangs_cuda_clamped_path(monkeypatch):
    from sgl_nvidia_mega.deepseek_v41.moe import shared_ll

    monkeypatch.setattr(shared_ll.mxfp8_ll, "supported", lambda: True)
    assert shared_ll.why_not(_shared_mlp([])) == ""
    uint8_down = _Callable(
        None, weight=torch.zeros(1, dtype=torch.uint8), weight_scale_inv=None
    )
    no_view = SimpleNamespace(weight=torch.zeros(256, 256, dtype=torch.float8_e4m3fn))
    for overrides, why in (
        (dict(swiglu_limit=None), "clamped SwiGLU"),
        (dict(use_fused_clamp_act_mul=True), "clamped SwiGLU"),
        (dict(_enable_nvfp4_gemm_swiglu_fusion=True), "clamped SwiGLU"),
        (dict(down_proj=uint8_down), "fused down_proj"),
        (dict(gate_up_proj=no_view), "no MXFP8 view"),
    ):
        assert why in shared_ll.why_not(_shared_mlp([], **overrides)), overrides
    assert "multiples of 128" in shared_ll.why_not(_shared_mlp([], n=192))
    fused = _shared_mlp([])
    fused.num_fused_shared_experts = 1
    assert "no separate shared expert" in shared_ll.why_not(fused)


@pytest.mark.parametrize("capture", [False, True])
@pytest.mark.parametrize("quant_routed", [False, True])
@pytest.mark.parametrize("quant_once", [False, True])
def test_shared_ll_takes_the_shared_expert_at_the_fork(
    capture, quant_routed, quant_once, caplog, monkeypatch
):
    """The shared expert on the alternate stream from the layer's rows, with no wait on
    the routed quant (which would launch it behind the routed GEMMs); the routed pair
    keeps SGLang's pre-quant."""
    step_mod = importlib.import_module("sgl_nvidia_mega.deepseek_v41.moe.step")
    monkeypatch.setattr(step_mod, "_SHARED_LL_SERVED", set())
    log = []
    mlp, x, cur = _block(
        log, inplace=False, quant_once=quant_once, quant_routed=quant_routed
    )

    def pair(x_, topk, pre_quant):
        pre = None if pre_quant is None else tuple(pre_quant)
        log.append(("pair", cur["stream"], pre))
        return "deferred"

    def shared(rows):
        log.append(("shared_ll", cur["stream"], rows is x))
        return "shared_ll_out"

    slot = SimpleNamespace(gate=False, run=pair, shared=shared)
    st = SimpleNamespace(
        fb=SimpleNamespace(moe_num_token_non_padded=lambda: "nnp"),
        transport=SimpleNamespace(
            input_ids_global="ids_g", do_shared_local=False, return_moe_output=True
        ),
        slots=SimpleNamespace(routed=True, fast_gate=False),
        bound={"moe": SimpleNamespace(slots={7: slot})},
    )
    with _sglang(log, cur, capture=capture), caplog.at_level("INFO"):
        out = moe.step(SimpleNamespace(mlp=mlp, layer_id=7), x, st)
    names = [e[0] for e in log]
    served = [r.getMessage() for r in caplog.records if "low-latency" in r.getMessage()]
    if quant_once:  # SGLang's fp8 input for both: its shared expert, as before
        assert "shared_ll" not in names and "shared" in names and not served
        assert out.shared == "shared_out"[: x.shape[0]]  # SGLang's padded-row slice
        return
    assert "shared" not in names
    assert ("shared_ll", "alt", True) in log
    # the alternate stream forks before the routed branch and joins nothing else
    alt_waits = [e for e in log if e[0] == "wait" and e[1] == "alt"]
    assert alt_waits == [("wait", "alt", "main")]
    assert log.index(("wait", "alt", "main")) < names.index("shared_ll")
    assert names.count("quantize_routed") == (1 if quant_routed else 0)
    pre = ("xq", "xsf", "event:rq") if quant_routed else None
    assert ("pair", "main", pre) in log
    assert served == [
        "NVIDIA megakernel: shared expert gate_up on the low-latency GEMM serves 3 rows"
    ]
    assert out.routed == "deferred" and out.shared == "shared_ll_out"


def test_shared_ll_binds_where_it_applies_and_falls_back_whole(monkeypatch):
    from sgl_nvidia_mega.deepseek_v41.moe import routed, shared_ll

    built = []

    class Built:
        def __init__(self, mlp, max_rows):
            self.mlp, self.max_rows = mlp, max_rows

        def precompile(self, widths):
            built.append((self.mlp, self.max_rows, tuple(widths)))

    monkeypatch.setattr(shared_ll, "SharedLL", Built)
    monkeypatch.setattr(
        shared_ll, "why_not", lambda mlp: "" if mlp != "draft" else "no MXFP8 view"
    )
    slots = {i: routed.MoeSlot.__new__(routed.MoeSlot) for i in range(3)}
    for i, slot in slots.items():
        slot.mlp, slot.shared = ("draft" if i == 2 else f"mlp{i}"), None
    assert routed._bind_shared(slots, (6, 12)) == "layer 2: no MXFP8 view"
    assert built == [("mlp0", 48, (6, 12)), ("mlp1", 48, (6, 12))]
    assert slots[2].shared is None and slots[0].shared.mlp == "mlp0"

    def boom(widths):
        raise RuntimeError("no kernel")

    monkeypatch.setattr(Built, "precompile", lambda self, widths: boom(widths))
    assert routed._bind_shared(slots, (6,)) == "RuntimeError: no kernel"
    assert all(s.shared is None for s in slots.values())


# ------------------------------------------------------------------------------------
# The stock mHC boundaries on CPU: they call SGLang's mHC functions with the arguments
# SGLang's stage boundary passes, in its order.

ROWS, HC, HIDDEN = 6, 4, 8
# the fixture stubs defers_sum; the tests of its two-shot widths restore this one
REAL_DEFERS_SUM = Boundaries.defers_sum


def _layer(layer_id, *, engram=None, mxfp8=True):
    norm = SimpleNamespace(weight=torch.ones(HIDDEN), variance_epsilon=1e-6)
    return SimpleNamespace(
        layer_id=layer_id,
        engram=engram,
        boundary_mhc_fusions=True,
        boundary_pre_from_prev=True,
        _can_fuse_attn_mhc=True,
        _can_fuse_ffn_mhc=True,
        hc_stats_stream=object(),
        hc_cfg=SimpleNamespace(hidden=HIDDEN),
        attn_hc=SimpleNamespace(name=f"attn{layer_id}"),
        ffn_hc=SimpleNamespace(name=f"ffn{layer_id}"),
        input_layernorm=SimpleNamespace(norm=norm, name=f"in{layer_id}"),
        post_attention_layernorm=SimpleNamespace(norm=norm, name=f"post{layer_id}"),
        self_attn=SimpleNamespace(
            attn_tp_size=4, accepts_mxfp8_swizzled_input=lambda: mxfp8
        ),
        mlp=SimpleNamespace(tp_size=4),
    )


def _boundary(fake, layers, **kw):
    def make_boundary(norm, *, accepts_mxfp8, hc=None):
        return ("next", norm.name, accepts_mxfp8, hc.name)

    fake.make_boundary.side_effect = make_boundary
    with patch("sglang.srt.models.deepseek_v4_mhc", fake):
        return Boundaries(layers, **kw)


@pytest.fixture
def fake():
    calls = []
    m = Mock()
    m.HcState = mhc.HcState
    m.AttnOutput = mhc.AttnOutput
    m.use_stats_stream.return_value = True
    m.fork_stats_stream.side_effect = lambda s: calls.append(("fork", s))

    def combine(hc, state, quantized):
        calls.append(("combine", hc.name, quantized is not None))
        if quantized is not None:
            quantized.append("q")
        return "x"

    m.combine.side_effect = combine
    m.run_attn_post.side_effect = lambda *a, **k: mhc.HcState(torch.ones(1), "pre")
    m.run_moe_post.side_effect = lambda *a, **k: mhc.HcState(torch.zeros(1), "pre")
    m.calls = calls
    # the fused collective's widths unless a test says otherwise
    with patch.object(Boundaries, "defers_sum", return_value=True):
        yield m


def test_a_layer_runs_sglangs_mhc_calls_in_order(fake):
    layers = [_layer(0), _layer(1), _layer(2, engram=object())]
    b = _boundary(fake, layers)
    streams = torch.randn(ROWS, HC, HIDDEN)
    st = b.start(streams)
    batch = object()
    layer = layers[0]

    x, xq = b.pre(layer, "attn", st, batch)
    assert (x, xq) == ("x", "q")
    assert fake.calls == [("fork", layer.hc_stats_stream), ("combine", "attn0", True)]
    fake.use_stats_stream.assert_called_with(layer.hc_cfg, batch, streams)

    partial = torch.randn(ROWS, HIDDEN)
    b.post(layer, "attn", partial, st)
    args, kw = fake.run_attn_post.call_args
    assert args[0] is layer.attn_hc and args[2] is streams
    assert isinstance(args[1], mhc.AttnOutput) and args[1].partial is partial
    assert kw == dict(
        stats_stream=layer.hc_stats_stream,
        next=("next", "post0", False, "ffn0"),
        world_size=4,
        precomputed=None,
    )

    fake.calls.clear()
    x, xq = b.pre(layer, "ffn", st, batch)
    assert xq is None and fake.calls[1] == ("combine", "ffn0", False)
    moe_out = object()
    b.post(layer, "ffn", moe_out, st)
    args, kw = fake.run_moe_post.call_args
    assert args[0] is layer.ffn_hc and args[1] is moe_out
    assert kw["next"] == ("next", "in1", True, "attn1") and kw["world_size"] == 4

    # the MoE post before an Engram layer and the last one leave the streams only
    for layer in layers[1:]:
        b.pre(layer, "attn", st, batch)
        b.post(layer, "attn", partial, st)
        b.pre(layer, "ffn", st, batch)
        b.post(layer, "ffn", moe_out, st)
        assert fake.run_moe_post.call_args.kwargs["next"] is None


def test_the_attention_post_takes_sglangs_attention_output_as_is(fake):
    layers = [_layer(0)]
    b = _boundary(fake, layers)
    st = b.start(torch.randn(ROWS, HC, HIDDEN))
    b.pre(layers[0], "attn", st, object())
    partial = torch.randn(ROWS, HIDDEN)
    b.post(layers[0], "attn", mhc.AttnOutput(partial), st)
    assert fake.run_attn_post.call_args.args[1].partial is partial


def test_without_cross_layer_fusion_no_moe_post_builds_the_next_input(fake):
    layers = [_layer(0), _layer(1)]
    b = _boundary(fake, layers, cross_layer=False)
    st = b.start(torch.randn(ROWS, HC, HIDDEN))
    b.pre(layers[0], "ffn", st, object())
    b.post(layers[0], "ffn", object(), st)
    assert fake.run_moe_post.call_args.kwargs["next"] is None


def test_an_attention_without_mxfp8_input_gets_no_quantized_rows(fake):
    layers = [_layer(0, mxfp8=False), _layer(1, mxfp8=False)]
    b = _boundary(fake, layers)
    st = b.start(torch.randn(ROWS, HC, HIDDEN))
    assert b.pre(layers[0], "attn", st, object()) == ("x", None)
    b.post(layers[0], "attn", torch.randn(ROWS, HIDDEN), st)
    b.pre(layers[0], "ffn", st, object())
    b.post(layers[0], "ffn", object(), st)
    assert fake.run_moe_post.call_args.kwargs["next"][2] is False


def test_a_post_needs_its_pre(fake):
    b = _boundary(fake, [_layer(0)])
    st = b.start(torch.randn(ROWS, HC, HIDDEN))
    with pytest.raises(RuntimeError):
        b.post(_layer(0), "attn", torch.randn(ROWS, HIDDEN), st)
    b.pre(_layer(0), "attn", st, object())
    with pytest.raises(RuntimeError):
        b.pre(_layer(0), "ffn", st, object())


def test_past_the_push_plane_the_attention_sum_runs_before_a_plain_post(fake):
    """Above the fused collective's widths SGLang's transport all-reduces the wo_b
    rows on the attention TP group; handing the post the partial would run the
    collective kernel past its push plane."""
    layers = [_layer(0)]
    b = _boundary(fake, layers)
    st = b.start(torch.randn(200, HC, HIDDEN))
    b.pre(layers[0], "attn", st, object())
    partial, summed = torch.randn(200, HIDDEN), torch.randn(200, HIDDEN)
    group = SimpleNamespace(all_reduce=Mock(return_value=summed))
    with (
        patch.object(Boundaries, "defers_sum", return_value=False) as defers,
        patch(
            "sglang.srt.runtime_context.get_parallel",
            return_value=SimpleNamespace(attn_tp_group=group),
        ),
    ):
        b.post(layers[0], "attn", partial, st)
    defers.assert_called_once_with(layers[0], "attn", 200)
    group.all_reduce.assert_called_once_with(partial)
    assert fake.run_attn_post.call_args.args[1] is summed


def test_hop_a_collects_a_published_partial_with_sglangs_triplet_and_norm(fake, caplog):
    """A published wo_b partial is collected by the Lamport consumer with the
    triplet SGLang's post would compute and the FFN norm; anything else takes
    SGLang's post."""
    triplet = (torch.ones(ROWS, HC), torch.ones(ROWS, HC), torch.ones(ROWS, HC, HC))
    fake.mix_stats.return_value = triplet
    fake.HcNormed = mhc.HcNormed
    updated, normed = torch.zeros(ROWS, HC, HIDDEN), torch.ones(ROWS, HIDDEN)
    published = torch.randn(ROWS, HIDDEN)
    hop = SimpleNamespace(
        published=lambda t: t is published,
        collect=Mock(return_value=(updated, normed)),
    )
    layers = [_layer(0)]
    with patch("sglang.srt.models.deepseek_v4_mhc", fake):
        b = Boundaries(layers, hop_a=hop)
    fake.make_boundary.return_value = None
    b._ffn_next[0] = SimpleNamespace(
        norm=SimpleNamespace(weight=torch.ones(HIDDEN), variance_epsilon=1e-6)
    )
    streams = torch.randn(ROWS, HC, HIDDEN)
    st = b.start(streams)
    b.pre(layers[0], "attn", st, object())
    with patch.object(torch.cuda, "current_stream") as current, caplog.at_level("INFO"):
        b.post(layers[0], "attn", mhc.AttnOutput(published), st)
    current.return_value.wait_stream.assert_called_once_with(layers[0].hc_stats_stream)
    # engagement from the dispatch itself, as merge.sh's REQUIRE lines read it
    served = [r.getMessage() for r in caplog.records if "hop A" in r.getMessage()]
    assert served == [f"NVIDIA megakernel: hop A attention post serves {ROWS} rows"]
    fake.mix_stats.assert_called_once_with(
        layers[0].attn_hc, streams, layers[0].hc_stats_stream
    )
    args = hop.collect.call_args.args
    assert args[0] is published and args[1] is streams
    assert args[2:5] == (triplet[1], triplet[2], triplet[0])
    assert args[6] == 1e-6
    assert st.hc.streams is updated and st.hc.input.rows is normed
    assert st.hc.pre is triplet[0]
    fake.run_attn_post.assert_not_called()

    b.pre(layers[0], "ffn", st, object())
    b.post(layers[0], "ffn", object(), st)
    b.pre(layers[0], "attn", st, object())
    b.post(layers[0], "attn", torch.randn(ROWS, HIDDEN), st)
    fake.run_attn_post.assert_called_once()


def _deferred(rows, *, replicated=False):
    from sglang.srt.layers.moe.moe_runner.flashinfer_trtllm import (
        FlashInferTrtllmDeferredFinalizeOutput,
    )
    from sglang.srt.models.deepseek_v2 import MoEOutput

    routed = FlashInferTrtllmDeferredFinalizeOutput(
        torch.ones(rows * 6, HIDDEN), torch.ones(rows, 6), torch.zeros(rows, 6), 6
    )
    return MoEOutput(routed, torch.ones(rows, HIDDEN), None, 1.0, replicated)


def _hop_boundary(fake, rows, **kw):
    """A boundary at ``rows`` with SGLang's triplet and next-input types real."""
    triplet = (torch.ones(rows, HC), torch.ones(rows, HC), torch.ones(rows, HC, HC))
    fake.mix_stats.return_value = triplet
    fake.HcNormed, fake.HcQuantized = mhc.HcNormed, mhc.HcQuantized
    fake.is_deferred_finalize = mhc.is_deferred_finalize
    layers = [_layer(0), _layer(1)]
    with patch("sglang.srt.models.deepseek_v4_mhc", fake):
        b = Boundaries(layers, **kw)
    norm = SimpleNamespace(weight=torch.ones(HIDDEN), variance_epsilon=1e-6)
    b._ffn_next[0] = SimpleNamespace(norm=norm)
    return b, layers, triplet, norm


@pytest.mark.parametrize(
    "receiver,rows",
    [("closed", 64), ("unfusable", 64), ("normed", 64), ("mxfp8", 64), ("mxfp8", 192)],
)
def test_two_shot_runs_both_posts_and_builds_sglangs_next_input(fake, receiver, rows):
    """Both posts take the two-shot hop; the MoE post hands on what SGLang's
    post would: MXFP8 only where its fused quant runs (up to 128 rows), else the
    streams only, for the next combine to collapse."""
    updated, normed = torch.zeros(rows, HC, HIDDEN), torch.ones(rows, HIDDEN)
    hop = SimpleNamespace(
        fits=lambda r, h: r >= 49 and h == HIDDEN,
        attn_post=Mock(return_value=(updated, normed)),
        moe_post=Mock(return_value=(updated, normed)),
        moe_post_only=Mock(return_value=updated),
        scale_bytes=lambda r: 160 * 128,
    )
    with patch.object(Boundaries, "defers_sum", REAL_DEFERS_SUM):
        b, layers, triplet, norm = _hop_boundary(fake, rows, two_shot=hop)
        assert b.defers_sum(layers[0], "attn", rows)
        assert b.defers_sum(layers[0], "ffn", rows)
    pre, post_mix, comb = triplet
    streams = torch.randn(rows, HC, HIDDEN)
    st = b.start(streams)
    b.pre(layers[0], "attn", st, object())
    partial = torch.randn(rows, HIDDEN)
    with (
        patch.object(torch.cuda, "current_stream"),
        patch(
            "sglang.srt.layers.quantization.mxfp4_flashinfer_trtllm_moe.can_fuse_all_reduce",
            side_effect=lambda r, h: r <= 152,
        ),
    ):
        b.post(layers[0], "attn", partial, st)
    assert hop.attn_post.call_args.args == (
        partial,
        streams,
        post_mix,
        comb,
        pre,
        norm.weight,
        1e-6,
    )
    # past SGLang's fused collective its own combine norms the streams
    if rows <= 152:
        assert isinstance(st.hc.input, mhc.HcNormed)
    else:
        assert st.hc.input is None and st.hc.streams is updated
    fake.run_attn_post.assert_not_called()

    b._attn_next[0] = {
        "closed": None,
        "unfusable": SimpleNamespace(norm=norm, norm_fusable=False, accepts_mxfp8=True),
        "normed": SimpleNamespace(norm=norm, norm_fusable=True, accepts_mxfp8=False),
        "mxfp8": SimpleNamespace(norm=norm, norm_fusable=True, accepts_mxfp8=True),
    }[receiver]
    b.pre(layers[0], "ffn", st, object())
    out = _deferred(rows)
    with patch.object(torch.cuda, "current_stream"):
        b.post(layers[0], "ffn", out, st)
    fake.run_moe_post.assert_not_called()
    if receiver != "mxfp8" or rows > 128:
        assert hop.moe_post_only.call_args.args[4] is out.shared
        assert st.hc.input is None
        hop.moe_post.assert_not_called()
    else:
        q, sf = hop.moe_post.call_args.kwargs["mxfp8_out"]
        assert q.shape == (rows, HIDDEN) and sf.numel() == 160 * 128
        assert st.hc.input.swizzled.data is q


def test_hop_b_takes_the_moe_post_only_where_sglang_would_fuse_it(fake):
    """Hop B mirrors SGLang's one-shot post at hop-A widths: MXFP8 only for an
    MXFP8 receiver whose norm folds, else the streams; a replicated shared expert
    stays on SGLang's post (the finalize would add it on every rank)."""
    rows = 6
    updated = torch.zeros(rows, HC, HIDDEN)
    hop_a = SimpleNamespace(
        rows=(6,),
        published=lambda t: False,
        finalize=Mock(return_value=(updated, torch.ones(rows, HIDDEN))),
    )
    b, layers, triplet, norm = _hop_boundary(fake, rows, hop_a=hop_a, hop_b=True)
    fused = SimpleNamespace(norm=norm, norm_fusable=True, accepts_mxfp8=True)
    st = b.start(torch.randn(rows, HC, HIDDEN))
    for nxt, quant in ((None, False), (fused, True)):
        b._attn_next[0] = nxt
        b.pre(layers[0], "ffn", st, object())
        with patch.object(torch.cuda, "current_stream"):
            b.post(layers[0], "ffn", _deferred(rows), st)
        call = hop_a.finalize.call_args
        assert ("mxfp8_out" in call.kwargs) is quant
        assert (
            isinstance(st.hc.input, mhc.HcQuantized) if quant else st.hc.input is None
        )
    fake.run_moe_post.assert_not_called()
    b.pre(layers[0], "ffn", st, object())
    b.post(layers[0], "ffn", _deferred(rows, replicated=True), st)
    fake.run_moe_post.assert_called_once()


def test_two_shot_needs_sglangs_static_fusion_gate(fake):
    """A layer SGLang would not fuse keeps SGLang's post at two-shot widths."""
    hop = SimpleNamespace(fits=lambda r, h: True, attn_post=Mock())
    with patch.object(Boundaries, "defers_sum", REAL_DEFERS_SUM):
        b, layers, _, _ = _hop_boundary(fake, 64, two_shot=hop)
        layers[0]._can_fuse_attn_mhc = False
        with patch(
            "sglang.srt.layers.quantization.mxfp4_flashinfer_trtllm_moe.can_fuse_all_reduce",
            return_value=False,
        ):
            assert not b.defers_sum(layers[0], "attn", 64)
    st = b.start(torch.randn(64, HC, HIDDEN))
    b.pre(layers[0], "attn", st, object())
    with patch(
        "sglang.srt.runtime_context.get_parallel",
        return_value=SimpleNamespace(
            attn_tp_group=SimpleNamespace(all_reduce=lambda t: t)
        ),
    ):
        b.post(layers[0], "attn", torch.randn(64, HIDDEN), st)
    hop.attn_post.assert_not_called()
    fake.run_attn_post.assert_called_once()


def test_the_prenorm_hands_sglangs_posts_their_triplet(fake, monkeypatch):
    from sgl_nvidia_mega.deepseek_v41.boundary import prenorm

    triplet = ("pre", "post", "comb")
    stats = Mock(return_value=triplet)
    monkeypatch.setattr(prenorm, "mix_stats", stats)
    fake.use_stats_stream.return_value = False  # no stats stream to join on CPU
    layers = [_layer(0), _layer(1)]
    b = _boundary(fake, layers, prenorm=True)
    streams = torch.randn(ROWS, HC, HIDDEN)
    # the MoE post's pre reads the attention post's streams: real ones, not the stub's
    fake.run_attn_post.side_effect = lambda *a, **k: mhc.HcState(streams, None)
    st = b.start(streams)
    layer = layers[0]
    b.pre(layer, "attn", st, object())
    b.post(layer, "attn", torch.randn(ROWS, HIDDEN), st)
    assert fake.run_attn_post.call_args.kwargs["precomputed"] == triplet
    stats.assert_called_with(layer.attn_hc, streams, None)
    b.pre(layer, "ffn", st, object())
    b.post(layer, "ffn", object(), st)
    assert fake.run_moe_post.call_args.kwargs["precomputed"] == triplet
    assert stats.call_args.args[0] is layer.ffn_hc
    fake.mix_stats.assert_not_called()
    assert "DeepGEMM TF32" in b.summary()


def test_the_prenorm_takes_only_bf16_rows_up_to_its_width(monkeypatch):
    from sgl_nvidia_mega.deepseek_v41.boundary import prenorm

    sglang_stats = Mock(return_value="sglang")
    monkeypatch.setattr(mhc, "mix_stats", sglang_stats)
    monkeypatch.setattr(prenorm, "_coefficients", Mock(return_value="prenorm"))
    hc = object()
    small = torch.zeros(prenorm.MAX_ROWS, HC, HIDDEN, dtype=torch.bfloat16)
    wide = torch.zeros(prenorm.MAX_ROWS + 1, HC, HIDDEN, dtype=torch.bfloat16)
    assert prenorm.mix_stats(hc, small) == "prenorm"
    assert prenorm.mix_stats(hc, wide) == "sglang"
    assert prenorm.mix_stats(hc, small.float()) == "sglang"
    assert prenorm.takes(small) and not prenorm.takes(wide)


def test_the_prenorm_logs_only_the_rows_it_serves(fake, monkeypatch, caplog):
    from sgl_nvidia_mega.deepseek_v41.boundary import prenorm

    monkeypatch.setattr(prenorm, "mix_stats", Mock(return_value=("p", "q", "c")))
    fake.use_stats_stream.return_value = False
    layers = [_layer(0)]
    b = _boundary(fake, layers, prenorm=True)
    for rows, dtype in ((6, torch.bfloat16), (prenorm.MAX_ROWS + 6, torch.bfloat16)):
        st = b.start(torch.zeros(rows, HC, HIDDEN, dtype=dtype))
        b.pre(layers[0], "attn", st, object())
        with caplog.at_level("INFO"):
            b.post(layers[0], "attn", torch.zeros(rows, HIDDEN), st)
    served = [r.getMessage() for r in caplog.records if "mHC prenorm" in r.getMessage()]
    assert served == ["NVIDIA megakernel: mHC prenorm serves 6 rows"]


def test_a_capture_layer_appends_the_streams_mean_without_a_stats_stream(fake):
    """Slot aux_mean_stats where the layer has no stats stream: the mean runs on the
    caller's stream, of the streams the attention reads (after an Engram injection)."""
    fake.use_stats_stream.return_value = False
    layers = [_layer(0)]
    b = _boundary(fake, layers)
    st = b.start(torch.randn(ROWS, HC, HIDDEN))
    injected = torch.randn(ROWS, HC, HIDDEN)
    b.with_residual(st, injected)
    aux = []
    b.pre(layers[0], "attn", st, object(), aux)
    assert len(aux) == 1 and torch.equal(aux[0], injected.mean(dim=1))
    assert st.aux_done == []
    b.join_aux(st)
    assert st.aux_done == []


def test_a_wide_capture_layer_keeps_the_mean_on_the_main_stream(fake):
    """Slot aux_mean_stats past AUX_MEAN_MAX_ROWS: the stats stream is not used."""
    from sgl_nvidia_mega.deepseek_v41.boundary.boundaries import AUX_MEAN_MAX_ROWS

    fake.use_stats_stream.return_value = True
    layers = [_layer(0)]
    b = _boundary(fake, layers)
    streams = torch.randn(AUX_MEAN_MAX_ROWS + 1, HC, HIDDEN)
    st = b.start(streams)
    aux = []
    b.pre(layers[0], "attn", st, object(), aux)
    assert torch.equal(aux[0], streams.mean(dim=1))
    assert st.aux_done == []


# ------------------------------------------------------------------------------------
# The DeepSeek-V4.1 SM103 kernels' host-side bounds on CPU: the routed-expert record's
# tile bound and the served row bound.


@pytest.mark.parametrize(
    "rows,top_k,experts", [(12, 6, 384), (48, 6, 384), (64, 8, 32), (40, 3, 128)]
)
def test_max_ctas_bounds_any_routing(rows, top_k, experts):
    generator = torch.Generator().manual_seed(rows * top_k)
    for distinct in (top_k, top_k + 1, 2 * top_k, 64, experts):
        distinct = min(distinct, experts)
        pool = torch.randperm(experts, generator=generator)[:distinct]
        ids = torch.stack(
            [
                pool[torch.randperm(distinct, generator=generator)[:top_k]]
                for _ in range(rows)
            ]
        )
        counts = torch.bincount(ids.flatten()).tolist()
        tiles = sum(math.ceil(c / reg.TILE) for c in counts if c)
        assert tiles <= reg.max_ctas(rows, top_k, experts)


def test_the_served_rows_fit_the_wide_route():
    assert reg.MAX_ROWS < reg.SERVED_MAX_ROWS <= reg.WIDE_MAX_ROWS
