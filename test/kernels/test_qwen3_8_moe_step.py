# SPDX-License-Identifier: Apache-2.0
"""Slot moe_finalize's decode-size MoE step (runtime/moe.py) under a real CUDA graph, at
Qwen3.8's TP1 MoE geometry (512 experts, top-10, hidden 2560, expert intermediate 640, a
gated BF16 shared expert of intermediate 640) with random NVFP4 expert weights and the
fused routing, on an SM10x GPU with FlashInfer 0.7.3 or 0.7.4.

The captured step runs the source's dual stream: the shared expert first on the current
stream, on the block input itself (SHARED_NO_CLONE) or on a clone of it, and the router
GEMM, the routing and the routed GEMMs on the side stream, with no wait between the
two branches before the finalize. Bitwise against the order it replaces (the shared
expert after the router GEMM, through an event) and against the eager step, from 4 to
320 rows; a one-element input change is the control that must differ. The graph's
edges: the two branches share only the fork (in the replaced order the shared expert
depends on the router GEMM, the control), the clone is one copy node off the fork, and
the finalize joins both branches.
"""

import ctypes as c
import os

# Rows above 16 take FlashInfer's single-cluster routing (the math the fused kernel
# reproduces) only with its block-per-token split off, as the tuned runs set it;
# FlashInfer reads this once per process.
os.environ["FLASHINFER_ROUTING_FORCE_BLOCK_PER_TOKEN"] = "0"

from types import SimpleNamespace
from unittest import mock

import pytest
import torch
import torch.nn.functional as F

from sgl_nvidia_mega.qwen3_8.runtime import (
    bf16_gemm_pins,
    capture_fork,
    moe,
    moe_routing,
)


def _available() -> bool:
    if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 10:
        return False
    try:
        import flashinfer
    except ImportError:
        return False
    return flashinfer.__version__ in moe_routing.FLASHINFER_VERSIONS


pytestmark = pytest.mark.skipif(
    not _available(), reason="the fused routing needs SM10x and FlashInfer 0.7.3/0.7.4"
)

HIDDEN, EXPERTS, TOP_K, INTER, SHARED_INTER = 2560, 512, 10, 640, 640
# the c13 points' verify, decode and draft rows, and the c176 points' verify rows
ROWS = (4, 8, 12, 16, 20, 32, 64, 176, 192, 240, 256, 272, 320)
CU_GRAPH_NODE_TYPE_MEMCPY = 1


@pytest.fixture(scope="module")
def runner():
    # isort: off
    # mxfp4 first: importing flashinfer_trtllm first hits a circular import.
    from sglang.srt.layers.quantization import mxfp4  # noqa: F401
    from sglang.srt.layers.moe.moe_runner import flashinfer_trtllm

    # isort: on
    return flashinfer_trtllm


@pytest.fixture(autouse=True)
def tp1(runner, monkeypatch):
    from sglang.srt import runtime_context

    parallel = SimpleNamespace(tp_size=1)
    monkeypatch.setattr(runtime_context, "get_parallel", lambda: parallel)
    monkeypatch.setattr(runner, "get_parallel", lambda: parallel)
    monkeypatch.setattr(runner, "get_forward", lambda: SimpleNamespace(sp_active=False))
    assert moe_routing.fused_reasons() == ""
    monkeypatch.setattr(moe_routing, "_fused_built", True)
    # SGLang's projection calls (no BF16 pins, no SM107 small-M routes)
    monkeypatch.setattr(bf16_gemm_pins, "routed", lambda: False)


@pytest.fixture(scope="module")
def block(runner):
    return _Block(runner, seed=0)


class _Experts:
    """The FusedMoE members the step reads, over main's stock runner function (at TP1
    and EP1 the dispatch passes the input through)."""

    def __init__(self, runner, g):
        from sglang.srt.layers.moe.moe_runner.base import MoeRunnerConfig
        from sglang.srt.layers.moe.utils import RoutingMethodType

        def u8(*shape, lo=0, hi=256):
            return torch.randint(
                lo, hi, shape, generator=g, device="cuda", dtype=torch.uint8
            )

        def per_expert(value):
            return torch.full((EXPERTS,), value, dtype=torch.float32, device="cuda")

        # Random NVFP4 bytes; E4M3 block scales of 2^-3 .. 2^0 keep the GEMMs finite.
        fp8 = torch.float8_e4m3fn
        weights = dict(
            w13_weight=u8(EXPERTS, 2 * INTER, HIDDEN // 2),
            w2_weight=u8(EXPERTS, HIDDEN, INTER // 2),
            w13_weight_scale=u8(
                EXPERTS, 2 * INTER, HIDDEN // 16, lo=0x20, hi=0x39
            ).view(fp8),
            w2_weight_scale=u8(EXPERTS, HIDDEN, INTER // 16, lo=0x20, hi=0x39).view(
                fp8
            ),
            g1_scale_c=per_expert(1 / 256),
            g1_alphas=per_expert(1 / 256),
            g2_alphas=per_expert(1 / 256),
            w13_input_scale_quant=torch.tensor([64.0], device="cuda"),
        )
        self.__dict__.update(weights)
        self.runner = runner
        self.num_experts = self.num_local_experts = EXPERTS
        self.moe_ep_rank = 0
        self.intermediate_size_per_partition = INTER
        self.routing_method_type = RoutingMethodType.RenormalizeNaive
        # FusedMoE sets inplace=False for FlashInfer's TRT-LLM runner
        self.moe_runner_config = MoeRunnerConfig(activation="silu", inplace=False)
        self.quant_info = runner.FlashInferTrtllmFp4MoeQuantInfo(
            **weights,
            global_num_experts=EXPERTS,
            local_expert_offset=0,
            local_num_experts=EXPERTS,
            intermediate_size_per_partition=INTER,
            routing_method_type=RoutingMethodType.RenormalizeNaive,
        )
        self.dispatcher = SimpleNamespace(
            combine=lambda combine_input: combine_input.hidden_states
        )

    def run_moe_core(self, dispatch_output):
        return self.runner.fused_experts_none_to_flashinfer_trtllm_fp4(
            dispatch_output, self.quant_info, self.moe_runner_config
        )


class _TopK:
    def __init__(self):
        from sglang.srt.layers.moe.topk import TopKConfig

        self.topk_config = TopKConfig(top_k=TOP_K)

    def __call__(self, x, logits):
        from sglang.srt.layers.moe.topk import BypassedTopKOutput

        return BypassedTopKOutput(x, logits, self.topk_config)


class _Linear:  # a BF16 projection: module(x) -> (out, None)
    def __init__(self, out_features, in_features, g):
        self.weight = (
            torch.randn(out_features, in_features, generator=g, device="cuda") * 0.02
        ).to(torch.bfloat16)

    def __call__(self, x):
        return F.linear(x, self.weight), None


class _Block:
    """The Qwen2MoeSparseMoeBlock members the step reads: the router, the NVFP4 experts
    with the fused routing, and an ungated BF16 shared expert (gate_up, SiLU-and-mul,
    down_proj) whose gate the finalize applies."""

    supports_deferred_finalize = True
    shared_expert = "shared expert"

    def __init__(self, runner, seed):
        g = torch.Generator(device="cuda").manual_seed(seed)
        self.experts = _Experts(runner, g)
        self.topk = _TopK()
        self.gate = _Linear(EXPERTS, HIDDEN, g)
        self.gate_up = _Linear(2 * SHARED_INTER, HIDDEN, g)
        self.down = _Linear(HIDDEN, SHARED_INTER, g)
        self.shared_expert_gate = _Linear(1, HIDDEN, g)
        self.alt_stream = torch.cuda.Stream()

    def _forward_shared_experts(self, x, apply_gate):
        assert not apply_gate
        gate_up = self.gate_up(x)[0]
        act = F.silu(gate_up[:, :SHARED_INTER]) * gate_up[:, SHARED_INTER:]
        return self.down(act)[0]


def _executor():
    return SimpleNamespace(
        moe_finalize_ready=True,
        moe_routing=moe_routing.DEFAULT,
        smallm_gemm=None,
        draft=False,
    )


def _st(ops):
    return SimpleNamespace(ran=lambda op, impl: ops.append((op, impl)), fb=None)


def _router_first(executor, mlp, x, st):
    """The order the step ran before: the router GEMM on the side stream, the shared
    expert on the current stream after it (an event), then the routing and the routed
    GEMMs on the side stream, the join and the same finalize launch."""
    from sglang.kernels.jit.utils import is_arch_support_pdl

    current, side = torch.cuda.current_stream(), mlp.alt_stream
    side.wait_stream(current)
    with torch.cuda.stream(side):
        logits = bf16_gemm_pins.linear(mlp.gate, x)
        router_done = side.record_event()
    current.wait_event(router_done)
    shared = moe._shared(executor, mlp, x, st)
    with torch.cuda.stream(side):
        deferred = moe_routing.fused_from_logits(mlp, x, logits, st)
    current.wait_stream(side)
    for t in (
        deferred.gemm2_out,
        deferred.expert_weights,
        deferred.expanded_idx_to_permuted_idx,
    ):
        t.record_stream(current)
    pdl = is_arch_support_pdl()
    return moe._finalize(
        mlp,
        x,
        shared,
        deferred,
        chunk=moe.FINALIZE_COLUMN_CHUNK if x.shape[0] <= 64 else 0,
        USE_PDL=pdl,
        num_warps=moe.FINALIZE_NUM_WARPS,
        **({"launch_pdl": True} if pdl else {}),
    )


def _step(executor, mlp, x, st):
    return moe.step(executor, SimpleNamespace(mlp=mlp), x, st)


def _eager(mlp, x):
    ops = []
    with mock.patch(
        "sglang.srt.model_executor.runner.get_is_capture_mode", lambda: False
    ):
        out = _step(_executor(), mlp, x, _st(ops))
    torch.cuda.synchronize()
    assert (moe_routing.OP, "fused_routing") in ops
    return out


def _captured(fn, mlp, x, no_clone=False, probe=None):
    """``fn`` captured in a CUDA graph on a static copy of ``x`` and replayed; returns
    the replay's output and the ops it recorded."""
    executor, ops = _executor(), []
    static_x = x.clone()
    main = torch.cuda.Stream()
    # warm outside the capture: tuner lookups, JIT builds, each stream's cuBLAS workspace
    with torch.cuda.stream(main):
        fn(executor, mlp, static_x, _st([]))
    with torch.cuda.stream(mlp.alt_stream):
        mlp.gate(static_x)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with (
        mock.patch(
            "sglang.srt.model_executor.runner.get_is_capture_mode", lambda: True
        ),
        mock.patch.object(moe, "SHARED_NO_CLONE", no_clone),
        # the step's FusedMoE check sees the fake experts as FusedMoE
        mock.patch("sglang.srt.layers.moe.fused_moe_triton.layer.FusedMoE", _Experts),
    ):
        with torch.cuda.graph(graph, stream=main):
            # a node for the block input, as the layer's previous kernel would be
            block_in = static_x + 0
            if probe is not None:
                probe["entry"] = capture_fork.projection_frontier(
                    torch.cuda.current_stream()
                )
            out = fn(executor, mlp, block_in, _st(ops))
    graph.replay()
    torch.cuda.synchronize()
    return out.clone(), ops


def _one_element_changed(x):
    y = x.clone()
    y[0, 0] += 0.5
    return y


@pytest.mark.parametrize("rows", ROWS)
def test_captured_step_bitwise_to_the_replaced_order(block, rows):
    g = torch.Generator(device="cuda").manual_seed(rows)
    x = torch.randn(rows, HIDDEN, generator=g, device="cuda").to(torch.bfloat16)
    ref, _ = _captured(_router_first, block, x)
    assert bool(torch.isfinite(ref).all())
    for no_clone in (False, True):
        out, ops = _captured(_step, block, x, no_clone=no_clone)
        assert ("moe.shared_input", "no_clone" if no_clone else "stock") in ops
        assert (moe_routing.OP, "fused_routing") in ops
        assert ("moe", "finalize_permuted") in ops
        assert torch.equal(out, ref), (rows, no_clone)
    # the control: changing one input element changes the output
    changed, _ = _captured(_step, block, _one_element_changed(x), no_clone=True)
    assert not torch.equal(changed, ref)


@pytest.mark.parametrize("rows", ROWS)
def test_captured_step_bitwise_to_the_eager_step(block, rows):
    g = torch.Generator(device="cuda").manual_seed(1000 + rows)
    x = torch.randn(rows, HIDDEN, generator=g, device="cuda").to(torch.bfloat16)
    ref = _eager(block, x)
    assert bool(torch.isfinite(ref).all())
    for no_clone in (False, True):
        out, _ = _captured(_step, block, x, no_clone=no_clone)
        assert torch.equal(out, ref), (rows, no_clone)
    assert not torch.equal(_eager(block, _one_element_changed(x)), ref)


def _dependencies(node):
    driver = c.CDLL("libcuda.so.1")
    count = c.c_size_t()
    try:  # cuGraphNodeGetDependencies_v2(node, deps, edge data, count); PDL edges carry
        # edge data, so a query without a CUgraphEdgeData (8 bytes) buffer is lossy (402)
        get = driver.cuGraphNodeGetDependencies_v2
        args = lambda deps, edges: (c.c_void_p(node), deps, edges, c.byref(count))
    except AttributeError:
        get = driver.cuGraphNodeGetDependencies
        args = lambda deps, edges: (c.c_void_p(node), deps, c.byref(count))
    assert get(*args(None, None)) == 0
    if count.value == 0:  # a root node
        return set()
    deps = (c.c_void_p * count.value)()
    edges = (c.c_uint64 * count.value)()
    assert get(*args(deps, edges)) == 0
    return {d for d in deps}


def _ancestors(nodes):
    seen, todo = set(), list(nodes)
    while todo:
        for dep in _dependencies(todo.pop()):
            if dep not in seen:
                seen.add(dep)
                todo.append(dep)
    return seen


def _node_type(node):
    kind = c.c_int()
    driver = c.CDLL("libcuda.so.1")
    assert driver.cuGraphNodeGetType(c.c_void_p(node), c.byref(kind)) == 0
    return kind.value


def _probed(fn, block, x, no_clone):
    """Capture ``fn`` with the capture frontiers around the shared expert, the router
    GEMM, the routed half and the finalize recorded (read while the graph exists)."""
    probe = {}
    shared, fused, finalize = moe._shared, moe_routing.fused_from_logits, moe._finalize
    linear = bf16_gemm_pins.linear

    def frontier():  # empty outside the capture (the warm-up call)
        nodes = capture_fork.projection_frontier(torch.cuda.current_stream())
        return set(nodes or ())

    def shared_probe(*args):
        probe["shared_in"] = frontier()
        probe["shared_in_deps"] = [_dependencies(n) for n in probe["shared_in"]]
        probe["shared_in_types"] = {_node_type(n) for n in probe["shared_in"]}
        out = shared(*args)
        probe["shared_out"] = frontier()
        probe["shared_branch"] = probe["shared_out"] | _ancestors(probe["shared_out"])
        return out

    def linear_probe(module, x):
        out = linear(module, x)
        if module is block.gate:
            probe["router"] = frontier()
            probe["router_branch"] = probe["router"] | _ancestors(probe["router"])
        return out

    def fused_probe(*args):
        out = fused(*args)
        probe["routed_out"] = frontier()
        return out

    def finalize_probe(*args, **kw):
        probe["finalize_in"] = frontier()
        return finalize(*args, **kw)

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(moe, "_shared", shared_probe)
        mp.setattr(bf16_gemm_pins, "linear", linear_probe)
        mp.setattr(moe_routing, "fused_from_logits", fused_probe)
        mp.setattr(moe, "_finalize", finalize_probe)
        _captured(fn, block, x, no_clone=no_clone, probe=probe)
    return probe


@pytest.mark.parametrize("rows", (16, 256))
def test_graph_runs_the_shared_expert_beside_the_router_gemm(block, rows):
    x = torch.randn(rows, HIDDEN, device="cuda").to(torch.bfloat16)
    for no_clone in (True, False):
        p = _probed(_step, block, x, no_clone)
        entry = set(p["entry"])
        if no_clone:
            # the shared expert's first kernel depends on the block input's node only
            assert p["shared_in"] == entry
        else:
            # one copy node off the fork, then the shared expert
            assert p["shared_in_types"] == {CU_GRAPH_NODE_TYPE_MEMCPY}
            assert p["shared_in_deps"] == [entry]
        # the shared expert and the router GEMM share only the fork
        assert p["shared_branch"] & p["router_branch"] == entry
        # the finalize waits for both branches
        assert p["finalize_in"] == p["shared_out"] | p["routed_out"]
        assert p["shared_out"].isdisjoint(p["routed_out"])
    # the control: in the replaced order the shared expert depends on the router GEMM
    p = _probed(_router_first, block, x, False)
    assert p["router"] <= p["shared_in"]
    assert p["router"] <= p["shared_branch"]
