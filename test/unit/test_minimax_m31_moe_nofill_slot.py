# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""CPU checks of slot moe_nofill of the MiniMax M3.1 megakernel (moe/moe_nofill.py).

SGLang's own MoE code runs on small fake pieces: MiniMaxM3MoE.forward_normal, the
experts' deferred-finalize path, the compressed-tensors method, MiniMaxM3Nvfp4MoE's
apply_weights, the runner and fused_experts_none_to_flashinfer_trtllm_fp4 are SGLang's;
the shared expert, router, top-k and dispatcher record their calls; FlashInfer is a CPU
fake whose per-token quantize host code mirrors FlashInfer 0.7.4 and whose kernels
compute deterministic bytes from all their inputs. Checked: the slot's step makes
SGLang's launches on SGLang's streams with SGLang's arguments, minus the fill and the
quantize's allocations, and returns the same handoff; its build (with the check that
FlashInfer's own call compiles the kernel the slot launches, and the refusal of hooked
modules) and fallbacks. On GPU the slot is checked by the composition check
(SGLANG_NVIDIA_MEGAKERNEL_PARITY=1) at every width it serves.
"""

import contextlib
import enum
import importlib.machinery
import os
import sys
import types
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

# The quantization package before the MoE runner module (the gate imports it): the
# reverse order is a circular import on CUDA.
pytest.importorskip("sglang.srt.layers.quantization.minimax_m3_nvfp4")

# isort: off
from sglang.srt.layers.quantization.minimax_m3_nvfp4 import MiniMaxM3Nvfp4MoE  # noqa: E402
from sglang.srt.layers.moe.moe_runner import flashinfer_trtllm as fit  # noqa: E402

# isort: on
from sglang.srt.layers.moe.fused_moe_triton.layer import FusedMoE  # noqa: E402
from sglang.srt.layers.moe.moe_runner.base import MoeRunnerConfig  # noqa: E402
from sglang.srt.layers.moe.moe_runner.runner import MoeRunner  # noqa: E402
from sglang.srt.layers.moe.token_dispatcher.standard import (  # noqa: E402
    StandardDispatchOutput,
)
from sglang.srt.layers.moe.topk import StandardTopKOutput  # noqa: E402
from sglang.srt.layers.quantization.compressed_tensors.compressed_tensors import (  # noqa: E402
    CompressedTensorsFusedMoEMethod,
)
from sglang.srt.models import minimax_m3 as m3  # noqa: E402
from sglang.srt.models.minimax_m3 import (  # noqa: E402
    MiniMaxM3MoE,
)
from sglang.test.test_utils import CustomTestCase  # noqa: E402

from sgl_nvidia_mega.minimax_m31 import moe as m31_moe  # noqa: E402
from sgl_nvidia_mega.minimax_m31 import plan as m31_plan  # noqa: E402
from sgl_nvidia_mega.minimax_m31.moe import moe_nofill as mn  # noqa: E402

# the widths of the fake MoE's covered steps (8 rows a request)
WIDTHS = (8, 16, 24)

H, E, I, TOP_K = 64, 8, 32, 4
ENVS = (
    "FLASHINFER_DISABLE_FP4_QUANT_FAST_MATH",
    "FLASHINFER_NVFP4_4OVER6",
    "FLASHINFER_NVFP4_4OVER6_E4M3_USE_256",
)


class Trace:
    def __init__(self):
        self.calls = []

    def add(self, *call):
        self.calls.append(call)


class Stream:
    def __init__(self, name, trace):
        self.name, self.trace = name, trace

    def wait_stream(self, other):
        self.trace.add("wait_stream", self.name, other.name)


class Streams:
    """torch.cuda's current stream and stream context, on CPU."""

    def __init__(self, trace):
        self.trace = trace
        self.main, self.alt = Stream("main", trace), Stream("alt", trace)
        self.current = self.main

    def current_stream(self, device=None):
        return self.current

    @contextlib.contextmanager
    def stream(self, s):
        self.trace.add("enter", s.name)
        prev, self.current = self.current, s
        try:
            yield
        finally:
            self.current = prev
            self.trace.add("leave", s.name)


# ------------------------------------------------------------------ fake FlashInfer


class SfLayout(enum.IntEnum):
    layout_128x4 = 0
    layout_8x4 = 1
    layout_linear = 2


class ActivationType(enum.IntEnum):
    Gelu = 0
    Relu = 1
    Silu = 2
    Swiglu = 3
    Geglu = 4
    Relu2 = 6
    Situ = 10


def _module(name, *, package=False, **attrs):
    mod = types.ModuleType(name)
    mod.__spec__ = importlib.machinery.ModuleSpec(name, None, is_package=package)
    if package:
        mod.__path__ = []
    mod.__dict__.update(attrs)
    return mod


def _unexpected(*args, **kwargs):
    raise AssertionError("the routed chain must not reach this FlashInfer entry")


# FlashInfer's "argument not given" marker for nvfp4_4over6 (read the environment)
_UNSET = object()


class FakeFlashInfer:
    """FlashInfer's entry points the routed chain reaches. The host code of
    nvfp4_quantize's per-token CuTe-DSL path is FlashInfer 0.7.4's; the kernels write
    deterministic bytes from every input and the key's numerics flags (PDL, the CTA
    width and the grid, as in FlashInfer's kernel, change no byte)."""

    SMS = 148

    def __init__(self, trace, streams, *, pdl=True):
        self.trace, self.streams = trace, streams
        self.compiled, self.launches, self.moe_calls = [], [], []
        self.nq = _module(
            "flashinfer.quantization.kernels.nvfp4_quantize",
            SF_LAYOUT_LINEAR=2,
            _env_flag_enabled=lambda name: os.environ.get(name) == "1",
            resolve_nvfp4_4over6=self._resolve_4over6,
            _per_token_cta_threads=self._cta_threads,
            _get_compiled_kernel_nvfp4_per_token=self._compile,
            nvfp4_quantize_per_token_cute_dsl=self._per_token_cute_dsl,
            get_device_sm_count=lambda device: self.SMS,
            _MAX_THREADS_PER_BLOCK=1024,
        )
        kernels = _module(
            "flashinfer.quantization.kernels", package=True, nvfp4_quantize=self.nq
        )
        quantization = _module("flashinfer.quantization", package=True, kernels=kernels)
        self.utils = _module("flashinfer.utils", device_support_pdl=lambda device: pdl)
        core = _module("flashinfer.fused_moe.core", ActivationType=ActivationType)
        fused_moe = _module(
            "flashinfer.fused_moe",
            package=True,
            core=core,
            trtllm_fp4_block_scale_routed_moe=self._routed_moe,
            trtllm_fp4_block_scale_moe=_unexpected,
        )
        root = _module(
            "flashinfer",
            package=True,
            SfLayout=SfLayout,
            nvfp4_quantize=self._nvfp4_quantize,
            utils=self.utils,
            quantization=quantization,
            fused_moe=fused_moe,
        )
        mods = (root, self.utils, quantization, kernels, self.nq, fused_moe, core)
        self.modules = {m.__name__: m for m in mods}

    def installed(self):
        return patch.dict(sys.modules, self.modules)

    @staticmethod
    def _resolve_4over6(setting=_UNSET):
        """resolve_nvfp4_4over6: omitted reads the environment, a config is itself."""
        if setting is not _UNSET:
            return setting
        if os.environ.get("FLASHINFER_NVFP4_4OVER6") != "1":
            return None
        use_256 = os.environ.get("FLASHINFER_NVFP4_4OVER6_E4M3_USE_256") == "1"
        return ("4over6", 256 if use_256 else 448, "MAE", False)

    @staticmethod
    def _cta_threads(k, m):
        """_per_token_cta_threads: the CTA width for a row of k elements (few rows)."""
        blocks = k // 16
        threads = min(128, max(1, (blocks + 31) // 32) * 32)
        while threads < 512 and threads < blocks:
            threads *= 2
        return threads

    def _compile(
        self,
        dtype_key,
        K,
        sf_layout=0,
        enable_pdl=False,
        disable_fp4_quant_fast_math=False,
        nvfp4_4over6_config=None,
        fold_out_scale=False,
        threads=None,
        has_valid_rows=False,
        rows_per_group=0,
        sf_rows_per_group=0,
    ):
        key = tuple(v for name, v in locals().items() if name != "self")
        self.compiled.append(key)

        def launch(x, fp4, sf, pts, m, padded_m, gsi, out_scale, num_blocks, *rows):
            self.trace.add("quant_kernel", self.streams.current.name, key, m)
            self.launches.append(
                dict(key=key, x=x, fp4=fp4, sf=sf, pts=pts, m=m, gsi=gsi)
                | dict(padded_m=padded_m, num_blocks=num_blocks, rows=rows)
            )
            xf = x.float()
            bias = 2 * disable_fp4_quant_fast_math
            bias += 0 if nvfp4_4over6_config is None else nvfp4_4over6_config[1]
            pts.copy_(xf.abs().amax(1) * gsi[0])
            fp4.copy_(
                ((xf[:, ::2] * 4).round() + bias).long().remainder(256).to(torch.uint8)
            )
            blocks = xf.reshape(m, K // 16, 16).abs().amax(-1)
            sf.copy_(
                ((blocks * 8).round() + bias)
                .long()
                .remainder(256)
                .to(torch.uint8)
                .reshape(-1)
            )

        return launch

    def _per_token_cute_dsl(
        self, input, global_scale_inv, sf_layout=0, enable_pdl=None, nvfp4_4over6=_UNSET
    ):
        """nvfp4_quantize_per_token_cute_dsl of FlashInfer 0.7.4 as SGLang's call
        reaches it (the linear layout, every row, no folded output scale), on CPU."""
        nq = self.nq
        assert sf_layout == nq.SF_LAYOUT_LINEAR
        if enable_pdl is not False:
            enable_pdl = self.utils.device_support_pdl(input.device)
        input = input.contiguous()
        m, k = input.shape
        gsi = global_scale_inv.float().reshape(1).contiguous()
        threads = nq._per_token_cta_threads(k, m)
        sms = nq.get_device_sm_count(input.device)
        kernel_fn = nq._get_compiled_kernel_nvfp4_per_token(
            {torch.float16: "float16", torch.bfloat16: "bfloat16"}[input.dtype],
            k,
            sf_layout,
            enable_pdl,
            nq._env_flag_enabled("FLASHINFER_DISABLE_FP4_QUANT_FAST_MATH"),
            nq.resolve_nvfp4_4over6(nvfp4_4over6),
            False,
            threads,
            False,
            0,
            0,
        )
        fp4_output = torch.empty(m, k // 2, dtype=torch.uint8)
        scale_output = torch.empty(m * (k // 16), dtype=torch.uint8)
        per_token_scale = torch.empty(m, dtype=torch.float32)
        num_blocks = min(m, sms * (nq._MAX_THREADS_PER_BLOCK // threads))
        kernel_fn(
            input,
            fp4_output,
            scale_output,
            per_token_scale,
            m,
            m,
            gsi,
            gsi,
            num_blocks,
            None,
            None,
        )
        return fp4_output, scale_output.reshape(m, k // 16), per_token_scale

    def _nvfp4_quantize(
        self,
        a,
        a_global_sf,
        sfLayout=SfLayout.layout_128x4,
        backend="cuda",
        per_token_activation=False,
        nvfp4_4over6=_UNSET,
    ):
        self.trace.add("nvfp4_quantize", self.streams.current.name)
        assert per_token_activation and backend == "cute-dsl"
        assert sfLayout == SfLayout.layout_linear
        return self.nq.nvfp4_quantize_per_token_cute_dsl(
            a,
            a_global_sf,
            sf_layout=self.nq.SF_LAYOUT_LINEAR,
            nvfp4_4over6=self.nq.resolve_nvfp4_4over6(nvfp4_4over6),
        )

    def _routed_moe(self, **kw):
        self.trace.add("routed_moe", self.streams.current.name)
        self.moe_calls.append(kw)
        ids, weights = kw["topk_ids"]
        hs, sf, pts = (
            kw["hidden_states"],
            kw["hidden_states_scale"],
            kw["per_token_scale"],
        )
        m, top_k = ids.shape
        rows = hs.float().repeat_interleave(2, 1)
        rows = (rows + sf.view(torch.uint8).float().repeat_interleave(16, 1)) * pts[
            :, None
        ]
        w = kw["gemm1_weights"].float().mean() + kw["gemm2_weights"].float().mean()
        w = w + kw["output1_scale_scalar"].sum() + kw["gemm1_clamp_limit"].sum()
        gemm2 = rows.repeat_interleave(top_k, 0) + ids.reshape(-1, 1).float() + w
        idx = torch.arange(m * top_k, dtype=torch.int32).flip(0)
        return [gemm2.to(torch.bfloat16), weights, idx]


# ------------------------------------------------------------ fake MoE pieces


class Shared(torch.nn.Module):
    def __init__(self, trace, streams):
        super().__init__()
        self.trace, self.streams = trace, streams

    def forward(self, x):
        self.trace.add("shared", self.streams.current.name, tuple(x.shape))
        return x * 0.25 + 1


class TopK(torch.nn.Module):
    def __init__(self, trace, streams):
        super().__init__()
        self.trace, self.streams = trace, streams

    def forward(self, x, router_logits):
        self.trace.add("topk", self.streams.current.name)
        weights, ids = router_logits.topk(TOP_K, dim=-1)
        return StandardTopKOutput(
            topk_weights=weights.softmax(-1) * 2,
            topk_ids=ids.to(torch.int32),
            router_logits=router_logits,
        )


class Dispatcher:
    def __init__(self, trace, streams):
        self.trace, self.streams = trace, streams

    def dispatch(self, hidden_states, topk_output):
        self.trace.add("dispatch", self.streams.current.name)
        return StandardDispatchOutput(
            hidden_states=hidden_states,
            hidden_states_scale=None,
            topk_output=topk_output,
        )

    def combine(self, combine_input):
        self.trace.add("combine", self.streams.current.name)
        (hidden_states,) = combine_input
        return hidden_states


class Scheme(MiniMaxM3Nvfp4MoE):
    """MiniMaxM3Nvfp4MoE on the TRT-LLM routed runner, without its platform checks."""

    def __init__(self, runner):
        self.use_marlin, self.use_flashinfer_trtllm = False, True
        self.use_mega_moe_prefill = False
        self.requantizes_experts = False
        self.runner = runner


class Experts(FusedMoE):
    """A FusedMoE as MiniMaxM3Nvfp4MoE's TRT-LLM processing leaves it (no
    FusedMoE.__init__)."""

    def __init__(self, trace, streams, *, seed=0, weight_hidden=H):
        torch.nn.Module.__init__(self)
        g = torch.Generator().manual_seed(seed)

        def p(t):
            return torch.nn.Parameter(t, requires_grad=False)

        def u8(*shape):
            return torch.randint(0, 256, shape, generator=g, dtype=torch.uint8)

        self.w13_weight = p(u8(E, 2 * I, weight_hidden // 2))
        self.w2_weight = p(u8(E, weight_hidden, I // 2))
        self.w13_weight_scale = p(
            torch.rand(E, 2 * I, weight_hidden // 16, generator=g).to(
                torch.float8_e4m3fn
            )
        )
        self.w2_weight_scale = p(
            torch.rand(E, weight_hidden, I // 16, generator=g).to(torch.float8_e4m3fn)
        )
        for name in ("g1_scale_c", "g1_alphas", "g2_alphas", "w13_input_scale_quant"):
            setattr(self, name, p(torch.rand(E, generator=g)))
        self.gemm1_alpha = torch.rand(E, generator=g) + 1
        self.gemm1_beta = torch.rand(E, generator=g)
        self.gemm1_clamp_limit = torch.rand(E, generator=g) * 7
        self.num_experts = self.num_local_experts = E
        self.moe_ep_size, self.moe_ep_rank = 1, 0
        self.hidden_size = H
        self.intermediate_size_per_partition = I
        self.routing_method_type = None
        self.supports_deferred_finalize = True
        self.moe_runner_config = MoeRunnerConfig(
            num_experts=E,
            num_local_experts=E,
            hidden_size=H,
            intermediate_size_per_partition=I,
            top_k=TOP_K,
            activation="silu",
            is_gated=True,
            gemm1_alpha=1.702,
            gemm1_beta=1.0,
            gemm1_clamp_limit=7.0,
            layer=self,
        )
        runner = object.__new__(MoeRunner)
        runner.config = self.moe_runner_config
        runner.fused_func = fit.fused_experts_none_to_flashinfer_trtllm_routed
        runner.lora_enabled = False
        self.quant_method = CompressedTensorsFusedMoEMethod(None)
        self.scheme = Scheme(runner)
        self.dispatcher = Dispatcher(trace, streams)


class Moe(MiniMaxM3MoE):
    """A MiniMaxM3MoE (no MiniMaxM3MoE.__init__): SGLang's forward over fakes."""

    def __init__(self, trace, streams, *, alt=True, seed=0, **experts_kw):
        torch.nn.Module.__init__(self)
        self.trace, self.streams = trace, streams
        self.alt_stream = streams.alt if alt else None
        self.num_fused_shared_experts = 0
        self.shared_experts = Shared(trace, streams)
        self.experts = Experts(trace, streams, seed=seed, **experts_kw)
        self.topk = TopK(trace, streams)
        self.supports_deferred_finalize = True
        g = torch.Generator().manual_seed(seed + 1)
        self.router = torch.randn(H, E, generator=g)

    def _compute_router_logits(self, x):
        self.trace.add("router", self.streams.current.name)
        return x.float() @ self.router


class Dense(torch.nn.Module):
    def forward(self, x, forward_batch=None):
        return x + 1


def fake_model(trace, streams, *, layers=4, dense=1, **moe_kw):
    mods = [
        SimpleNamespace(
            layer_id=i,
            is_layer_sparse=i >= dense,
            mlp=Moe(trace, streams, seed=i, **moe_kw) if i >= dense else Dense(),
        )
        for i in range(layers)
    ]
    return SimpleNamespace(
        model=SimpleNamespace(layers=mods, start_layer=0, end_layer=layers)
    )


def rows(m, seed=0, dtype=torch.bfloat16):
    g = torch.Generator().manual_seed(seed)
    return (torch.randn(m, H, generator=g) * 3).to(dtype)


@contextlib.contextmanager
def runtime(streams, *, capturing=True, defer=True, env=None):
    """The queries SGLang's MoE and the slot both make, faked alike; ``env`` the only
    FlashInfer NVFP4 variables set."""
    environ = {k: v for k, v in os.environ.items() if k not in ENVS}
    with contextlib.ExitStack() as stack:
        stack.enter_context(
            patch.dict(os.environ, {**environ, **(env or {})}, clear=True)
        )
        for target, name, value in (
            (m3, "get_is_capture_mode", lambda: capturing),
            (m3, "is_in_breakable_cuda_graph", lambda: False),
            (
                m3,
                "get_forward",
                lambda: SimpleNamespace(defer_moe_finalize=defer),
            ),
            (fit, "trtllm_moe_enable_pdl", lambda n: n <= 8192),
            (torch.cuda, "current_stream", streams.current_stream),
            (torch.cuda, "stream", streams.stream),
        ):
            stack.enter_context(patch.object(target, name, value))
        yield


class Allocations:
    """Counts torch.full / torch.zeros / torch.empty calls (the fills and buffers)."""

    def __init__(self):
        self.counts = {"full": 0, "zeros": 0, "empty": 0}

    @contextlib.contextmanager
    def counting(self):
        with contextlib.ExitStack() as stack:
            for name in self.counts:
                real = getattr(torch, name)

                def counted(*a, _real=real, _name=name, **k):
                    self.counts[_name] += 1
                    return _real(*a, **k)

                stack.enter_context(patch.object(torch, name, counted))
            yield


def _same(a, b):
    """Two arguments of a call: the same object, or tensors of equal dtype, shape and
    bytes, or equal values (recursively for tuples)."""
    if a is b:
        return True
    if isinstance(a, torch.Tensor) and isinstance(b, torch.Tensor):
        return (
            a.dtype == b.dtype
            and a.shape == b.shape
            and a.stride() == b.stride()
            and torch.equal(
                a.contiguous().reshape(-1).view(torch.uint8),
                b.contiguous().reshape(-1).view(torch.uint8),
            )
        )
    if isinstance(a, tuple) and isinstance(b, tuple):
        return len(a) == len(b) and all(_same(x, y) for x, y in zip(a, b))
    return type(a) is type(b) and a == b


def _moe_call_equal(test, got, want):
    """Two routed MoE calls with the same arguments: the ignored output buffer (an
    uninitialized allocation, flashinfer#6077) by its shape, dtype and strides."""
    test.assertEqual(sorted(got), sorted(want))
    for name in want:
        a, b = got[name], want[name]
        if name == "output":
            test.assertEqual(
                (a.dtype, a.shape, a.stride()), (b.dtype, b.shape, b.stride())
            )
        else:
            test.assertTrue(_same(a, b), name)


def _handoff_equal(test, got, want):
    for name in ("routed_output", "expert_weights", "permuted_indices"):
        test.assertTrue(_same(getattr(got, name), getattr(want, name)), name)
    test.assertTrue(torch.equal(got.gated_shared_output, want.gated_shared_output))
    test.assertEqual(got.m, want.m)


def _quant(max_rows=16):
    return mn.PerTokenQuant(max_rows=max_rows, hidden=H, device="cpu")


def _bound(model, quant):
    m = model.model
    return mn.Bound(
        widths=tuple(range(1, quant.max_rows + 1)),
        quant=quant,
        mlps={layer.layer_id: layer.mlp for layer in m.layers if layer.is_layer_sparse},
    )


class TestMoeStep(CustomTestCase):
    """The slot's MoE step against SGLang's MiniMaxM3MoE.forward on the same pieces."""

    def _run(self, *, capturing, m=8, env=None, alt=True):
        out = {}
        for path in ("stock", "slot"):
            trace = Trace()
            streams = Streams(trace)
            fi = FakeFlashInfer(trace, streams)
            mlp = Moe(trace, streams, alt=alt)
            quant = _quant()
            x = rows(m)
            allocations = Allocations()
            with fi.installed(), runtime(streams, capturing=capturing, env=env):
                e4m3 = mn.e4m3_max()
                with allocations.counting():
                    if path == "stock":
                        result = mlp(x, forward_batch=None)
                    else:
                        result = mn.forward(mlp, x, quant)
            out[path] = SimpleNamespace(
                trace=trace.calls,
                result=result,
                fi=fi,
                quant=quant,
                allocations=allocations.counts,
                e4m3=e4m3,
            )
        return out["stock"], out["slot"]

    def _check(self, stock, slot):
        # SGLang's launches in SGLang's order on SGLang's streams, minus FlashInfer's
        # nvfp4_quantize wrapper (whose allocations hold the fills)
        self.assertEqual(
            slot.trace, [c for c in stock.trace if c[0] != "nvfp4_quantize"]
        )
        self.assertIn("nvfp4_quantize", [c[0] for c in stock.trace])
        _handoff_equal(self, slot.result, stock.result)
        # the same kernel under the same key, the same global scale
        (want,), (got,) = stock.fi.launches, slot.fi.launches
        self.assertEqual(got["key"], want["key"])
        self.assertEqual(got["m"], want["m"])
        for name in ("x", "fp4", "sf", "pts", "gsi"):
            self.assertTrue(_same(got[name], want[name]), name)
        # on FlashInfer's grid, every row, no row groups
        for name in ("padded_m", "num_blocks", "rows"):
            self.assertEqual(got[name], want[name], name)
        # into the persistent buffers
        q = slot.quant
        self.assertEqual(got["fp4"].data_ptr(), q.fp4.data_ptr())
        self.assertEqual(got["sf"].data_ptr(), q.block_scale.data_ptr())
        self.assertEqual(got["pts"].data_ptr(), q.per_token_scale.data_ptr())
        self.assertIs(got["gsi"], q.global_scale_inv[slot.e4m3])
        # the routed MoE call with the stock arguments
        (want,), (got,) = stock.fi.moe_calls, slot.fi.moe_calls
        _moe_call_equal(self, got, want)
        self.assertEqual(got["hidden_states"].data_ptr(), q.fp4.data_ptr())
        # no fill and no quantize allocation of the slot's own: only the MoE's ignored
        # output buffer, which the stock call allocates too (flashinfer#6077)
        self.assertEqual(stock.allocations, {"full": 1, "zeros": 0, "empty": 4})
        self.assertEqual(slot.allocations, {"full": 0, "zeros": 0, "empty": 1})

    def test_capture_order_routed_experts_on_the_alt_stream(self):
        stock, slot = self._run(capturing=True)
        self._check(stock, slot)
        calls = [c[:2] for c in slot.trace]
        self.assertEqual(
            calls,
            [
                ("wait_stream", "alt"),
                ("shared", "main"),
                ("enter", "alt"),
                ("router", "alt"),
                ("topk", "alt"),
                ("dispatch", "alt"),
                ("quant_kernel", "alt"),
                ("routed_moe", "alt"),
                ("combine", "alt"),
                ("leave", "alt"),
                ("wait_stream", "main"),
            ],
        )

    def test_eager_order_one_stream(self):
        for capturing, alt in ((False, True), (True, False)):
            stock, slot = self._run(capturing=capturing, alt=alt)
            self._check(stock, slot)
            self.assertEqual(
                {c[1] for c in slot.trace if c[0] != "wait_stream"}, {"main"}
            )
            self.assertNotIn("enter", [c[0] for c in slot.trace])

    def test_environment_choices_follow_the_stock_call(self):
        cases = (
            ({}, False, None, 448.0),
            ({"FLASHINFER_DISABLE_FP4_QUANT_FAST_MATH": "1"}, True, None, 448.0),
            ({"FLASHINFER_NVFP4_4OVER6": "1"}, False, 448, 448.0),
            (
                {
                    "FLASHINFER_NVFP4_4OVER6": "1",
                    "FLASHINFER_NVFP4_4OVER6_E4M3_USE_256": "1",
                },
                False,
                256,
                256.0,
            ),
            # SGLang reads "true" as set, FlashInfer only "1": a 256 global scale with
            # FlashInfer's plain kernel, as the stock chain does
            (
                {
                    "FLASHINFER_NVFP4_4OVER6": "true",
                    "FLASHINFER_NVFP4_4OVER6_E4M3_USE_256": "true",
                },
                False,
                None,
                256.0,
            ),
        )
        for env, exact, four_over_six, e4m3 in cases:
            stock, slot = self._run(capturing=True, env=env)
            self._check(stock, slot)
            key = slot.fi.launches[0]["key"]
            self.assertEqual(key[:5], ("bfloat16", H, 2, True, exact), env)
            self.assertEqual(None if key[5] is None else key[5][1], four_over_six)
            gsi = slot.fi.launches[0]["gsi"]
            self.assertEqual(gsi.item(), torch.tensor(1 / (e4m3 * 6)).item(), env)

    def test_every_width_reuses_one_set_of_buffers(self):
        trace = Trace()
        streams = Streams(trace)
        fi = FakeFlashInfer(trace, streams)
        quant = _quant(max_rows=16)
        with fi.installed(), runtime(streams):
            for m in (1, 5, 16):
                fp4, sf, pts = quant(rows(m, seed=m))
                self.assertEqual(
                    (fp4.shape, sf.shape, sf.dtype, pts.shape),
                    ((m, H // 2), (m, H // 16), torch.float8_e4m3fn, (m,)),
                )
                self.assertEqual(fp4.data_ptr(), quant.fp4.data_ptr())
                self.assertEqual(sf.data_ptr(), quant.block_scale.data_ptr())
                self.assertEqual(pts.data_ptr(), quant.per_token_scale.data_ptr())
            for bad in (rows(17), rows(4)[:, : H - 16].contiguous(), rows(8).t()):
                with self.assertRaises(ValueError):
                    quant(bad)
        self.assertEqual(
            {g.item() for g in quant.global_scale_inv.values()},
            {torch.tensor(1 / (448 * 6)).item(), torch.tensor(1 / (256 * 6)).item()},
        )


class TestStepFallback(CustomTestCase):
    """moe.step: the slot when planned and its preconditions hold, else SGLang's
    module call."""

    def _step(self, *, slot=True, layer_id=1, x=None, defer=True, model=None, **kw):
        trace = Trace()
        streams = Streams(trace)
        fi = FakeFlashInfer(trace, streams)
        model = model or fake_model(trace, streams)
        quant = _quant()
        bound = kw.pop("bound", _bound(model, quant))
        layer = model.model.layers[layer_id]
        for f in kw.pop("mutate", ()):
            f(layer)
        st = SimpleNamespace(
            slots=m31_plan.Slots(moe_nofill=slot),
            bound={"moe": bound},
            fb="fb",
        )
        x = rows(8) if x is None else x
        called = []
        real = MiniMaxM3MoE.forward

        def stock(self, hidden_states, forward_batch):
            called.append(forward_batch)
            return "stock"

        with fi.installed(), runtime(streams, defer=defer):
            why = mn.fallback_reason(bound, layer, x)
            with patch.object(MiniMaxM3MoE, "forward", stock):
                out = m31_moe.step(layer, x, st)
        self.assertIs(MiniMaxM3MoE.forward, real)
        return out, called, why, fi

    def test_the_slot_runs_when_planned(self):
        out, called, why, fi = self._step()
        self.assertEqual((called, why), ([], ""))
        self.assertEqual(type(out).__name__, "MoeDeferredFinalize")
        self.assertEqual(len(fi.launches), 1)

    def test_slot_off_or_dense_layer_run_sglangs_module(self):
        out, called, _, fi = self._step(slot=False)
        self.assertEqual((out, called, fi.launches), ("stock", ["fb"], []))
        out, _, _, _ = self._step(layer_id=0)
        self.assertTrue(torch.equal(out, rows(8) + 1))

    def test_a_missed_precondition_runs_sglangs_module(self):
        cases = {
            "not deferred": dict(defer=False),
            "too many rows": dict(x=rows(17)),
            "not contiguous": dict(x=rows(16)[::2]),
            "fp16": dict(x=rows(8, dtype=torch.float16)),
            "not built": dict(bound=None),
            "built for no width": dict(bound=mn.Bound()),
            "replaced module": dict(
                mutate=[lambda layer: setattr(layer, "mlp", Moe(*_pieces()))]
            ),
            "hooked": dict(
                mutate=[
                    lambda layer: layer.mlp.register_forward_pre_hook(lambda *a: None)
                ]
            ),
        }
        for name, kw in cases.items():
            out, called, why, fi = self._step(**kw)
            self.assertEqual((out, called), ("stock", ["fb"]), name)
            self.assertTrue(why, name)
            self.assertEqual(fi.launches, [], name)


def _pieces():
    trace = Trace()
    return trace, Streams(trace)


class TestBuild(CustomTestCase):
    def _build(self, *, widths=(8, 16, 24), gathered=None, mutate=(), fi_mutate=()):
        trace = Trace()
        streams = Streams(trace)
        fi = FakeFlashInfer(trace, streams)
        for f in fi_mutate:
            f(fi)
        model = fake_model(trace, streams)
        for f in mutate:
            f(model)

        def gather(obj):
            return [obj] + list(gathered or [])

        with (
            fi.installed(),
            runtime(streams),
            patch.object(mn, "_gather", gather),
        ):
            return mn.build(model, widths), model, fi

    def test_every_width_when_every_layer_takes_the_chain(self):
        bound, model, fi = self._build()
        self.assertEqual(bound.reason, "")
        self.assertEqual(bound.widths, (8, 16, 24))
        self.assertTrue(all(m31_moe.serves(bound, w) for w in (8, 16, 24)))
        self.assertFalse(m31_moe.serves(bound, 32))
        moe = [layer for layer in model.model.layers if layer.is_layer_sparse]
        self.assertEqual(bound.mlps, {layer.layer_id: layer.mlp for layer in moe})
        q = bound.quant
        self.assertEqual(
            (q.fp4.shape, q.block_scale.shape, q.per_token_scale.shape),
            ((24, H // 2), (24 * H // 16,), (24,)),
        )
        # verify ran the kernel against FlashInfer's own call at 1 and 24 rows
        self.assertEqual([launch["m"] for launch in fi.launches], [1, 1, 24, 24])
        self.assertIsNone(m31_moe.build(model, (8,), frozenset()))
        self.assertFalse(m31_moe.serves(None, 8))

    def test_none_when_a_layer_misses_the_chain(self):
        def on_layer(f, i=2):
            return [lambda model: f(model.model.layers[i])]

        def experts(name, value):
            return on_layer(lambda layer: setattr(layer.mlp.experts, name, value))

        def scheme(name, value):
            return on_layer(
                lambda layer: setattr(layer.mlp.experts.scheme, name, value)
            )

        def runner(name, value):
            return on_layer(
                lambda layer: setattr(layer.mlp.experts.scheme.runner, name, value)
            )

        trace = Trace()
        cases = {
            "expert parallelism": experts("moe_ep_size", 2),
            "not CompressedTensorsFusedMoEMethod": experts("quant_method", object()),
            "TRT-LLM routed runner alone": scheme("use_flashinfer_trtllm", False),
            "does not run fused_experts": runner("lora_enabled", True),
            "padded past hidden size": on_layer(
                lambda layer: setattr(
                    layer.mlp,
                    "experts",
                    Experts(trace, Streams(trace), weight_hidden=H + 32),
                )
            ),
            "not MiniMaxM3MoE": on_layer(lambda layer: setattr(layer, "mlp", Dense())),
            "hooks, which the slot would skip": on_layer(
                lambda layer: layer.mlp.register_forward_hook(lambda *a: None)
            ),
            "replaces MiniMaxM3MoE.forward_normal": on_layer(
                lambda layer: setattr(layer.mlp, "forward_normal", lambda x: x)
            ),
            "AttributeError": on_layer(lambda layer: delattr(layer.mlp, "experts")),
        }
        for needle, mutate in cases.items():
            bound, _, _ = self._build(mutate=mutate)
            self.assertEqual(bound.widths, (), needle)
            self.assertIn(needle, bound.reason)
            self.assertIn("rank 0: ", bound.reason)
            if needle != "AttributeError":
                self.assertIn("layer 2: ", bound.reason)
            self.assertFalse(m31_moe.serves(bound, 8))

    def test_none_when_flashinfer_lacks_the_kernel_or_disagrees(self):
        def without_cute_dsl_quantize(fi):
            # a FlashInfer without flashinfer.quantization.kernels.nvfp4_quantize
            delattr(fi.modules["flashinfer.quantization.kernels"], "nvfp4_quantize")
            fi.modules[fi.nq.__name__] = None  # importing it raises ImportError

        def without_getter(fi):
            delattr(fi.nq, "_get_compiled_kernel_nvfp4_per_token")

        def other_bytes(fi):
            # FlashInfer's own call compiles another kernel than the slot's key
            real = fi.nq._get_compiled_kernel_nvfp4_per_token
            stock = fi.nq.nvfp4_quantize_per_token_cute_dsl

            def other(*key):
                return real(*key[:4], not key[4], key[5])

            def drifted(*a, **k):
                with patch.object(fi.nq, "_get_compiled_kernel_nvfp4_per_token", other):
                    return stock(*a, **k)

            fi.nq.nvfp4_quantize_per_token_cute_dsl = drifted

        def without_pdl(fi):
            # FlashInfer's own call turns PDL off: the same bytes (PDL only orders
            # launches), another kernel than the slot's key launches
            stock = fi.nq.nvfp4_quantize_per_token_cute_dsl
            fi.nq.nvfp4_quantize_per_token_cute_dsl = lambda *a, **k: stock(
                *a, **{**k, "enable_pdl": False}
            )

        variant = (
            f"under [('bfloat16', {H}, 2, False, False, None, False, 32, False, 0, 0)]"
        )
        for mutate, needle in (
            (without_cute_dsl_quantize, "does not import"),
            (without_getter, "has no _get_compiled_kernel_nvfp4_per_token"),
            (other_bytes, "differ from FlashInfer's nvfp4_quantize"),
            (without_pdl, variant),
        ):
            bound, _, fi = self._build(fi_mutate=[mutate])
            self.assertEqual(bound.widths, (), needle)
            self.assertIn(needle, bound.reason)
        # a variant of the same bytes shows only in the key FlashInfer compiles under;
        # the build leaves FlashInfer's getter as it found it
        self.assertNotIn("differ from FlashInfer's nvfp4_quantize", bound.reason)
        self.assertEqual(fi.nq._get_compiled_kernel_nvfp4_per_token, fi._compile)

    def test_none_on_every_rank_when_one_rank_misses(self):
        bound, _, _ = self._build(gathered=[["layer 7: the experts are odd"]])
        self.assertEqual(bound.widths, ())
        self.assertEqual(bound.reason, "rank 1: layer 7: the experts are odd")


if __name__ == "__main__":
    unittest.main()
