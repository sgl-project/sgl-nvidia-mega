# SPDX-License-Identifier: Apache-2.0
"""The MoE routing slot (runtime/moe_routing.py) against SGLang main's stock deferred
call, ``fused_experts_none_to_flashinfer_trtllm_fp4`` on a logits-bypassed top-10
(FlashInfer's own routing kernel and CuTe-DSL ``fp4_quantize``), at Qwen3.8's TP1 MoE
geometry (512 experts, hidden 2560) with random NVFP4 weights, on an SM10x GPU with
FlashInfer 0.7.3 or 0.7.4.

Bitwise, as the source claims: the fused kernel's NVFP4 activations and block scales
against the stock quant; the routing weights, and every routed GEMM2 row read through
each side's permutation (the row order inside an expert may differ), for 1..32 rows (one
CTA) and 33..512 rows (the cluster), at the default tactic and after autotuning the
stock call (the fused path must look up the same tactic); and the prequant call's
deferred output, from 1 to 1024 rows.
"""

import os

# Rows above 16 take FlashInfer's single-cluster routing (the math the fused kernel
# reproduces) only with its block-per-token split off, as the tuned runs set it;
# FlashInfer reads this once per process.
os.environ["FLASHINFER_ROUTING_FORCE_BLOCK_PER_TOKEN"] = "0"

from types import SimpleNamespace
from unittest import mock

import pytest
import torch

from sgl_nvidia_mega.qwen3_8.runtime import moe_routing


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

HIDDEN, EXPERTS, TOP_K, INTER = 2560, 512, 10, 512
# 1-32 rows run the single-CTA kernel, 33-512 the cluster kernel
ROWS = (1, 2, 4, 8, 16, 32, 33, 128, 512)
PREQUANT_ROWS = (1, 64, 600, 1024)


@pytest.fixture(scope="module")
def runner():
    # isort: off
    # mxfp4 first: importing flashinfer_trtllm first hits a circular import.
    from sglang.srt.layers.quantization import mxfp4  # noqa: F401
    from sglang.srt.layers.moe.moe_runner import flashinfer_trtllm

    # isort: on
    return flashinfer_trtllm


@pytest.fixture(autouse=True)
def tp1(runner):
    from sglang.srt import runtime_context

    parallel = SimpleNamespace(tp_size=1)
    with (
        mock.patch.object(runtime_context, "get_parallel", lambda: parallel),
        mock.patch.object(runner, "get_parallel", lambda: parallel),
        mock.patch.object(
            runner, "get_forward", lambda: SimpleNamespace(sp_active=False)
        ),
    ):
        yield


@pytest.fixture
def fused(monkeypatch):
    assert moe_routing.fused_reasons() == ""
    monkeypatch.setattr(moe_routing, "_fused_built", True)


def _weights(seed: int) -> dict:
    g = torch.Generator(device="cuda").manual_seed(seed)

    def u8(*shape: int, lo: int = 0, hi: int = 256) -> torch.Tensor:
        return torch.randint(
            lo, hi, shape, generator=g, device="cuda", dtype=torch.uint8
        )

    def per_expert(value: float) -> torch.Tensor:
        return torch.full((EXPERTS,), value, dtype=torch.float32, device="cuda")

    # Random NVFP4 bytes: both sides read the same weights, so parity needs no
    # meaningful values. E4M3 block scales of 2^-3 .. 2^0 keep the GEMMs finite.
    fp8 = torch.float8_e4m3fn
    return dict(
        w13_weight=u8(EXPERTS, 2 * INTER, HIDDEN // 2),
        w2_weight=u8(EXPERTS, HIDDEN, INTER // 2),
        w13_weight_scale=u8(EXPERTS, 2 * INTER, HIDDEN // 16, lo=0x20, hi=0x39).view(
            fp8
        ),
        w2_weight_scale=u8(EXPERTS, HIDDEN, INTER // 16, lo=0x20, hi=0x39).view(fp8),
        g1_scale_c=per_expert(1 / 256),
        g1_alphas=per_expert(1 / 256),
        g2_alphas=per_expert(1 / 256),
        w13_input_scale_quant=torch.tensor([64.0], device="cuda"),
    )


class _Experts:
    """The FusedMoE members the slot reads, over main's stock runner function (at TP1
    and EP1 the dispatch passes the input through)."""

    def __init__(self, runner, weights):
        from sglang.srt.layers.moe.moe_runner.base import MoeRunnerConfig
        from sglang.srt.layers.moe.utils import RoutingMethodType

        self.__dict__.update(weights)
        self.runner = runner
        self.num_experts = self.num_local_experts = EXPERTS
        self.moe_ep_rank = 0
        self.intermediate_size_per_partition = INTER
        self.routing_method_type = RoutingMethodType.RenormalizeNaive
        self.moe_runner_config = MoeRunnerConfig(activation="silu")
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

    def forward_deferred_finalize(self, x, topk_output):
        from sglang.srt.layers.moe.token_dispatcher.standard import (
            StandardDispatchOutput,
        )

        dispatch = StandardDispatchOutput(
            hidden_states=x, hidden_states_scale=None, topk_output=topk_output
        )
        with self.runner.flashinfer_trtllm_deferred_finalize_context():
            return self.run_moe_core(dispatch).hidden_states


class _TopK:
    def __init__(self):
        from sglang.srt.layers.moe.topk import TopKConfig

        self.topk_config = TopKConfig(top_k=TOP_K)

    def __call__(self, x, logits):
        from sglang.srt.layers.moe.topk import BypassedTopKOutput

        return BypassedTopKOutput(x, logits, self.topk_config)


class _Mlp:  # the Qwen2MoeSparseMoeBlock members the slot reads
    def __init__(self, experts, logits):
        self.experts, self.logits = experts, logits
        self.topk = _TopK()
        self.alt_stream = torch.cuda.Stream()

    def gate(self, x):
        return self.logits, None


def _inputs(m: int):
    g = torch.Generator(device="cuda").manual_seed(m)
    logits = torch.randn(m, EXPERTS, generator=g, device="cuda").to(torch.bfloat16)
    x = torch.randn(m, HIDDEN, generator=g, device="cuda").to(torch.bfloat16)
    return x, logits


def _stock(experts, x, logits):
    out = experts.forward_deferred_finalize(x, _TopK()(x, logits))
    torch.cuda.synchronize()
    return out


def _slot(experts, x, logits, options, *, capture=False):
    ops = []
    st = SimpleNamespace(ran=lambda op, impl: ops.append(impl))
    executor = SimpleNamespace(moe_routing=options)
    with mock.patch(
        "sglang.srt.model_executor.runner.get_is_capture_mode", lambda: capture
    ):
        out = moe_routing.routed(executor, _Mlp(experts, logits), x, st)
    torch.cuda.synchronize()
    return out, ops


def _assert_same(out, ref):
    assert torch.equal(out.expert_weights.float(), ref.expert_weights.float())
    out_perm = out.expanded_idx_to_permuted_idx.view(-1, TOP_K).long()
    ref_perm = ref.expanded_idx_to_permuted_idx.view(-1, TOP_K).long()
    assert bool((ref_perm >= 0).all()) and bool((out_perm >= 0).all())
    # Only routed rows are written: the padding rows of each expert's tile in
    # gemm2_out are uninitialized scratch, which finalize never reads.
    assert bool(torch.isfinite(ref.gemm2_out[ref_perm]).all())
    # Rows inside an expert may be ordered differently; each routed row must match.
    assert torch.equal(out.gemm2_out[out_perm], ref.gemm2_out[ref_perm])


def test_fused_routing_matches_the_stock_call(runner, fused):
    from flashinfer.autotuner import autotune

    experts = _Experts(runner, _weights(0))
    # Default tactic first: autotuning caches a tactic per M bucket.
    for tuned in (False, True):
        for m in ROWS:
            x, logits = _inputs(m)
            if tuned:
                with autotune(True):
                    _stock(experts, x, logits)
            ref = _stock(experts, x, logits)
            out, ops = _slot(experts, x, logits, moe_routing.DEFAULT)
            assert ops == ["fused_routing"], (tuned, m)
            _assert_same(out, ref)


@pytest.mark.parametrize("m", ROWS)
def test_fused_quant_matches_fp4_quantize(runner, fused, m):
    from sgl_nvidia_mega.qwen3_8.kernels.ops.moe import fused_routing as op

    scale = torch.tensor([64.0], device="cuda")
    x, logits = _inputs(m)
    x_fp4, x_sf = op.alloc_quant_outputs(x)
    ids = torch.empty(m, TOP_K, dtype=torch.int32, device="cuda")
    weights = torch.empty(m, TOP_K, dtype=torch.bfloat16, device="cuda")
    op.fused_routing(
        logits,
        x,
        scale,
        x_fp4=x_fp4,
        x_sf=x_sf,
        ids=ids,
        weights=weights,
        tile_tokens=8,
    )
    ref_fp4, ref_sf = runner.quantize_hidden_states_fp4(x, scale)
    torch.cuda.synchronize()
    assert torch.equal(x_fp4, ref_fp4)
    assert torch.equal(x_sf, ref_sf.view(torch.uint8))


def test_prequant_matches_the_stock_call(runner):
    experts = _Experts(runner, _weights(2))
    options = moe_routing.Options(fused_routing=False)
    for m in PREQUANT_ROWS:
        x, logits = _inputs(m)
        ref = _stock(experts, x, logits)
        out, ops = _slot(experts, x, logits, options, capture=True)
        assert ops == ["prequant"], m
        _assert_same(out, ref)


@pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 10,
    reason="needs an SM10x GPU",
)
@pytest.mark.parametrize("rows", [1, 4, 13, 64])
def test_chunked_finalize_matches_the_row_finalize(rows):
    """Slot moe_finalize's decode-size launch: the (row, 1024-column) grid writes what one
    row a program writes, at the same block and warp count (the gate's one reduction)."""
    from sgl_nvidia_mega.qwen3_8.kernels.ops.elementwise import moe_finalize as fin

    torch.manual_seed(rows)
    hidden, top_k, dev = 2560, 10, "cuda"
    x = torch.randn(rows, hidden, device=dev, dtype=torch.bfloat16)
    gate = torch.randn(hidden, device=dev, dtype=torch.bfloat16)
    shared = torch.randn(rows, hidden, device=dev, dtype=torch.bfloat16)
    gemm2 = torch.randn(rows * top_k + 7, hidden + 64, device=dev, dtype=torch.bfloat16)
    perm = torch.randperm(rows * top_k + 7, device=dev)[: rows * top_k].to(torch.int32)
    perm[::7] = -1
    perm = perm.view(rows, top_k).contiguous()
    weights = torch.rand(rows, top_k, device=dev, dtype=torch.float32)
    outs = []
    for chunk in (0, 1024):
        out = torch.empty_like(x)
        args = (x, gate, shared, out, gemm2, perm, weights, gemm2.stride(0))
        kw = dict(hidden_dim=hidden, TOP_K=top_k, BLOCK_SIZE=4096, num_warps=16)
        if chunk:
            fin._fused_finalize_gate_sigmoid_mul_add_chunked_kernel[
                (rows, -(-hidden // chunk))
            ](*args, CHUNK=chunk, **kw)
        else:
            fin._fused_finalize_gate_sigmoid_mul_add_kernel[(rows,)](*args, **kw)
        outs.append(out)
    torch.cuda.synchronize()
    assert torch.equal(outs[0], outs[1])
