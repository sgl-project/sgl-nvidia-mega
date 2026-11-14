# SPDX-License-Identifier: Apache-2.0
"""The Qwen3.8 executor's MoE routing slot on CPU: which path a call takes (fused
routing, prequant, stock), what each path launches and in which order, the stock call's
arguments the fused path rebuilds, and the build refusals; and slot moe_finalize's
extend-size step, which orders the routed half the source's way. SGLang's and
FlashInfer's modules are replaced by fakes and the kernel launches by spies, so no GPU,
no kernel build and no model weights are needed."""

import enum
import logging
import os
import sys
import types
import unittest
from contextlib import contextmanager
from types import SimpleNamespace
from typing import NamedTuple
from unittest import mock

import torch

from sgl_nvidia_mega.qwen3_8.kernels.ops.moe import fused_routing as op
from sgl_nvidia_mega.qwen3_8.runtime import moe
from sgl_nvidia_mega.qwen3_8.runtime import moe_routing as mr

HIDDEN, EXPERTS, TOP_K, INTER = 2560, 512, 10, 64
FP8 = torch.float8_e4m3fn


class _Routing(enum.IntEnum):  # the fake RoutingMethodType
    Default = 0
    RenormalizeNaive = 4


class _InputMode(enum.IntEnum):  # the fake RoutingInputMode
    FromLogits = 1


class _FusedMoE:
    """The fake FusedMoE: ModelOpt NVFP4 experts on FlashInfer's TRT-LLM runner."""

    def __init__(self, log):
        self.log = log
        self.quant_method = _NvFp4Method()
        self.runner_backend = SimpleNamespace(is_flashinfer_trtllm=lambda: True)
        self.trtllm_padded_hidden_size = None
        self._mega_moe_nvfp4 = False
        self.num_experts = self.num_local_experts = EXPERTS
        self.moe_ep_rank = 0
        self.intermediate_size_per_partition = INTER
        self.routing_method_type = _Routing.RenormalizeNaive
        self.moe_runner_config = SimpleNamespace(
            activation="silu", is_gated=True, routed_scaling_factor=None, inplace=False
        )
        self.w13_weight = torch.zeros(
            EXPERTS, 2 * INTER, HIDDEN // 2, dtype=torch.uint8
        )
        self.w13_weight_scale = torch.zeros(
            EXPERTS, 2 * INTER, HIDDEN // 16, dtype=torch.uint8
        )
        self.w2_weight = torch.zeros(EXPERTS, HIDDEN, INTER // 2, dtype=torch.uint8)
        self.w2_weight_scale = torch.zeros(
            EXPERTS, HIDDEN, INTER // 16, dtype=torch.uint8
        )
        self.g1_scale_c, self.g1_alphas, self.g2_alphas = (
            torch.full((EXPERTS,), v) for v in (1.0, 2.0, 3.0)
        )
        self.w13_input_scale_quant = torch.tensor([64.0])
        self.dispatcher = SimpleNamespace(combine=self.combine)

    def forward_deferred_finalize(self, x, topk_output):
        self.log.append(("stock_call", x, topk_output))
        return "stock"

    def run_moe_core(self, dispatch_output):
        self.log.append(("run_moe_core", dispatch_output, _STATE["deferred"]))
        return SimpleNamespace(hidden_states="prequant")

    def combine(self, combine_input):
        self.log.append(("combine",))
        return combine_input.hidden_states


class _NvFp4Method:  # the fake ModelOptNvFp4FusedMoEMethod
    quant_config = SimpleNamespace(use_per_token_activation=False)


class _Stream:
    def __init__(self, name, log):
        self.name, self.log = name, log

    def wait_stream(self, other):
        self.log.append(("wait", self.name, other.name))

    def record_event(self):
        self.log.append(("record_event", self.name))
        return ("event", self.name)

    def wait_event(self, event):
        self.log.append(("wait_event", self.name, event[1]))


class _Quantized:  # one of the prequant's outputs
    def __init__(self, name, log):
        self.name, self.log = name, log

    def record_stream(self, stream):
        self.log.append(("record_stream", self.name, stream.name))


class _Inputs(NamedTuple):  # the fake MoeRunnerInputs
    output: object
    routing_logits: object
    topk_ids: object
    expert_weights: object
    hidden_states: object
    hidden_states_scale: object
    gemm1_lora_delta: object
    per_token_scale: object

    def to_list(self):
        return list(self)


class _Runner:  # the fake FlashInfer MoERunner
    def __init__(self, moe_op, **kw):
        _STATE["runner"] = self
        self.moe_op, self.kw = moe_op, kw
        self.forwards = []

    def _make_tuning_config(self, inputs, **kw):
        return ("tuning", kw)

    def forward(self, inputs, tactic, **kw):
        self.forwards.append((inputs, list(tactic), kw))
        _STATE["log"].append(("body_prep" if kw.get("do_preparation") else "body",))
        if kw.get("do_preparation"):
            return "workspace"
        return [torch.ones(4, HIDDEN, dtype=torch.bfloat16)]


class _Tuner:
    def __init__(self):
        self.is_tuning_mode = False
        self.tactic = [16, 3]
        self.calls = []

    def choose_one(self, op_name, runners, config, inputs, **kw):
        self.calls.append((op_name, runners, config, inputs, kw))
        _STATE["log"].append(("choose_one",))
        return None, self.tactic


_STATE = {}


@contextmanager
def _deferred():
    _STATE["deferred"] = True
    try:
        yield
    finally:
        _STATE["deferred"] = False


@contextmanager
def _on(stream):
    previous, _STATE["on"] = _STATE["on"], stream
    try:
        yield
    finally:
        _STATE["on"] = previous


def _quantize(x, scale):
    log = _STATE["log"]
    log.append(("quant", _STATE["on"].name, x, scale))
    return _Quantized("x_fp4", log), _Quantized("x_sf", log)


def _install(test):
    """SGLang's and FlashInfer's modules the slot imports, as fakes, restored after."""
    import sglang.srt.utils as sglang_utils  # real; patched below, before the fakes go in

    log = []
    _STATE.update(
        log=log,
        deferred=False,
        current=_Stream("current", log),
        capture=False,
        tp=1,
        lora=False,
        a2a_none=True,
        tuner=_Tuner(),
        runner=None,
    )
    _STATE["on"] = _STATE["current"]
    fakes = {
        "sglang.srt.environ": dict(
            envs=SimpleNamespace(
                SGLANG_EXPERIMENTAL_LORA_OPTI=SimpleNamespace(
                    get=lambda: _STATE["lora"]
                )
            )
        ),
        "sglang.srt.layers.moe.fused_moe_triton.layer": dict(FusedMoE=_FusedMoE),
        "sglang.srt.layers.moe.utils": dict(
            RoutingMethodType=_Routing,
            get_moe_a2a_backend=lambda: SimpleNamespace(
                is_none=lambda: _STATE["a2a_none"]
            ),
        ),
        "sglang.srt.layers.quantization.modelopt_quant": dict(
            ModelOptNvFp4FusedMoEMethod=_NvFp4Method
        ),
        "sglang.srt.layers.quantization.fp4_utils": dict(
            _flashinfer_fp4_quantize_backend="cute-dsl"
        ),
        "sglang.srt.runtime_context": dict(
            get_parallel=lambda: SimpleNamespace(tp_size=_STATE["tp"])
        ),
        # moe.step reads only this; the real package imports SGLang's distributed
        # stack, which needs the real runtime_context the fake above replaces
        "sglang.srt.batch_invariant_ops": dict(
            is_batch_invariant_mode_enabled=lambda: False
        ),
        "sglang.srt.model_executor.runner": dict(
            get_is_capture_mode=lambda: _STATE["capture"]
        ),
        "sglang.srt.layers.moe.moe_runner.flashinfer_trtllm": dict(
            quantize_hidden_states_fp4=_quantize,
            flashinfer_trtllm_deferred_finalize_context=_deferred,
            get_activation_type=lambda act, is_gated: ("act", act, is_gated),
            trtllm_moe_enable_pdl=lambda rows: rows <= 32,
            FlashInferTrtllmDeferredFinalizeOutput=SimpleNamespace,
        ),
        "sglang.srt.layers.moe.token_dispatcher.standard": dict(
            StandardDispatchOutput=SimpleNamespace
        ),
        "flashinfer": dict(__version__="0.7.3"),
        "flashinfer.autotuner": dict(
            AutoTuner=SimpleNamespace(get=lambda: _STATE["tuner"])
        ),
        "flashinfer.fused_moe.shared.inputs": dict(
            MoeRunnerInputs=_Inputs,
            alloc_trtllm_moe_output=lambda n, h, fin, dev: torch.empty(
                n, h if fin else 0, dtype=torch.bfloat16
            ),
        ),
        "flashinfer.fused_moe.backends.trtllm": dict(MoERunner=_Runner),
        "flashinfer.fused_moe.core": dict(
            get_trtllm_moe_sm100_module=lambda: SimpleNamespace(moe_op="moe_op")
        ),
        "flashinfer.tllm_enums": dict(
            Fp8QuantizationType=SimpleNamespace(NoneFp8="none_fp8"),
            RoutingInputMode=_InputMode,
            WeightLayout=SimpleNamespace(MajorK="major_k"),
            deduce_trtllm_gen_tensor_dtype=lambda t, s: ("dtype", t.dtype, s.dtype),
        ),
        "flashinfer.utils": dict(device_support_pdl=lambda device: True),
    }
    for name, attrs in fakes.items():
        module = types.ModuleType(name)
        module.__dict__.update(attrs)
        old = sys.modules.get(name)
        sys.modules[name] = module
        test.addCleanup(_restore, name, old)
    for target, name, value in (
        (torch.cuda, "current_stream", lambda: _STATE["on"]),
        (torch.cuda, "stream", _on),
        (mr, "_fused_built", True),
        (sglang_utils, "next_power_of_2", lambda n: 1 << (n - 1).bit_length()),
    ):
        patcher = mock.patch.object(target, name, value)
        patcher.start()
        test.addCleanup(patcher.stop)
    mr._log_once.cache_clear()


def _restore(name, old):
    if old is None:
        sys.modules.pop(name, None)
    else:
        sys.modules[name] = old


class _TopK:
    def __init__(self, log):
        self.log = log
        self.topk_config = SimpleNamespace(
            top_k=TOP_K, correction_bias=None, num_expert_group=None, topk_group=None
        )

    def __call__(self, x, logits):
        self.log.append(("topk",))
        return SimpleNamespace(router_logits=logits, topk_config=self.topk_config)


class _Mlp:  # the fake Qwen2MoeSparseMoeBlock
    def __init__(self, log):
        self.log = log
        self.experts = _FusedMoE(log)
        self.topk = _TopK(log)
        self.alt_stream = _Stream("alt", log)
        self.enable_shared_expert_fusion = False

    def gate(self, x):
        self.log.append(("gate", _STATE["on"].name))
        return torch.zeros(x.shape[0], EXPERTS, dtype=torch.bfloat16), None


class _St:
    def __init__(self):
        self.ran_ops = []

    def ran(self, op_name, impl):
        self.ran_ops.append((op_name, impl))


class _Case(unittest.TestCase):
    def setUp(self):
        _install(self)
        self.calls = []

        def spy(logits, x, scale, **kw):
            self.calls.append(dict(kw, logits=logits, x=x, scale=scale))
            _STATE["log"].append(("fused_routing",))
            rows = x.shape[0]
            metadata = SimpleNamespace(
                tensors=lambda: ("metadata",),
                expanded_idx_to_permuted_idx=torch.arange(rows * TOP_K),
            )
            return op.FusedRouting(kw["ids"], metadata)

        patcher = mock.patch.object(op, "fused_routing", spy)
        patcher.start()
        self.addCleanup(patcher.stop)

    def run_step(self, rows=8, options=mr.DEFAULT, mlp=None):
        self.mlp = mlp or _Mlp(_STATE["log"])
        self.x = torch.zeros(rows, HIDDEN, dtype=torch.bfloat16)
        self.st = _St()
        executor = SimpleNamespace(moe_routing=options)
        return mr.routed(executor, self.mlp, self.x, self.st)

    def names(self):
        return [entry[0] for entry in _STATE["log"]]


class TestFusedRouting(_Case):
    def test_slot_off_declines(self):
        executor = SimpleNamespace(moe_routing=None)
        mlp = _Mlp(_STATE["log"])
        self.assertIsNone(mr.routed(executor, mlp, torch.zeros(4, HIDDEN), _St()))
        self.assertEqual(_STATE["log"], [])

    def test_kernel_then_the_prepared_routing_body(self):
        out = self.run_step(rows=8)
        self.assertEqual(
            self.names(),
            ["gate", "topk", "choose_one", "fused_routing", "body_prep", "body"],
        )
        self.assertEqual(self.st.ran_ops, [("moe.routing", "fused_routing")])
        (kernel,) = self.calls
        runner = _STATE["runner"]
        # the tuned tactic's tile, and the tactic both body calls run
        self.assertEqual(kernel["tile_tokens"], 16)
        self.assertEqual([f[1] for f in runner.forwards], [[16, 3], [16, 3]])
        prep, body = (f[2] for f in runner.forwards)
        self.assertEqual(prep["da_routing_metadata"], ["metadata"])
        self.assertTrue(prep["do_preparation"])
        self.assertEqual(body["da_body_workspace"], "workspace")
        # the deferred output: the body's FC2 rows, the kernel's weights and permutation
        self.assertTrue(
            torch.equal(out.gemm2_out, torch.ones(4, HIDDEN, dtype=torch.bfloat16))
        )
        self.assertIs(out.expert_weights, kernel["weights"])
        self.assertEqual(out.expanded_idx_to_permuted_idx.numel(), 8 * TOP_K)
        self.assertEqual(out.top_k, TOP_K)
        # the kernel writes the stock op's buffers: [8, 10] int32 ids, bf16 weights
        self.assertEqual(kernel["ids"].dtype, torch.int32)
        self.assertEqual(tuple(kernel["weights"].shape), (8, TOP_K))
        self.assertEqual(kernel["weights"].dtype, torch.bfloat16)
        self.assertEqual(tuple(kernel["x_fp4"].shape), (8, HIDDEN // 2))
        self.assertEqual(kernel["x_sf"].dtype, FP8)
        self.assertIs(kernel["scale"], self.mlp.experts.w13_input_scale_quant)

    def test_the_lookup_is_the_stock_calls(self):
        self.run_step(rows=8)
        runner = _STATE["runner"]
        ((op_name, runners, config, inputs, kw),) = _STATE["tuner"].calls
        self.assertEqual(op_name, "flashinfer::trtllm_fp4_block_scale_moe")
        self.assertEqual(runners, [runner])
        # the stock deferred call passes no output: the op's (rows, 0) placeholder
        self.assertEqual(tuple(inputs[0].shape), (8, 0))
        self.assertEqual(
            config[1],
            dict(
                tune_max_num_tokens=8,
                routing_input_mode=_InputMode.FromLogits,
                hidden_states_scale_layout=None,
                use_cold_l2_cache=True,
                use_cuda_graph=True,
            ),
        )
        self.assertEqual(kw["routing_input_mode"], 1)
        self.assertFalse(kw["do_finalize"])
        self.assertTrue(kw["enable_pdl"])
        self.assertEqual(kw["activation_type"], ("act", "silu", True))
        self.assertEqual(kw["routing_method_type"], _Routing.RenormalizeNaive)
        self.assertEqual(runner.moe_op, "moe_op")
        self.assertEqual(
            (
                runner.kw["hidden_size"],
                runner.kw["top_k"],
                runner.kw["intermediate_size"],
            ),
            (HIDDEN, TOP_K, INTER),
        )
        self.assertEqual(runner.kw["dtype_act"], ("dtype", torch.uint8, FP8))

    def test_call_kwargs(self):
        experts = _FusedMoE([])
        topk = SimpleNamespace(
            router_logits="logits", topk_config=_TopK([]).topk_config
        )
        x_fp4 = torch.zeros(5, HIDDEN // 2, dtype=torch.uint8)
        kw = mr._call_kwargs(experts, topk, x_fp4, "x_sf")
        self.assertEqual(kw["routing_logits"], "logits")
        self.assertIs(kw["hidden_states"], x_fp4)
        self.assertEqual(kw["hidden_states_scale"], "x_sf")
        self.assertEqual(kw["gemm1_weights_scale"].dtype, FP8)
        self.assertEqual(kw["gemm2_weights_scale"].dtype, FP8)
        for name, field in (
            ("output1_scale_scalar", "g1_scale_c"),
            ("output1_scale_gate_scalar", "g1_alphas"),
            ("output2_scale_scalar", "g2_alphas"),
            ("gemm1_weights", "w13_weight"),
            ("gemm2_weights", "w2_weight"),
        ):
            self.assertEqual(kw[name].data_ptr(), getattr(experts, field).data_ptr())
        self.assertEqual(
            (kw["gemm1_alpha"], kw["gemm1_beta"], kw["gemm1_clamp_limit"]),
            (None, None, None),
        )
        self.assertEqual((kw["local_expert_offset"], kw["local_num_experts"]), (0, 512))
        self.assertEqual((kw["tune_max_num_tokens"], kw["enable_pdl"]), (8, True))
        self.assertFalse(kw["do_finalize"])
        self.assertNotIn("output", kw)

    def test_default_tactic_takes_the_default_tile(self):
        _STATE["tuner"].tactic = -1
        self.run_step(rows=8)
        self.assertEqual(self.calls[0]["tile_tokens"], 8)
        self.assertEqual(_STATE["runner"].forwards[0][1], [-1, -1])
        self.assertEqual(mr._default_tile_n(1000, EXPERTS), 16)
        self.assertEqual(mr._default_tile_n(8192, EXPERTS), 128)

    def test_layout_miss_runs_the_stock_call_on_the_same_routing(self):
        with (
            mock.patch.object(op, "covered", return_value=False),
            self.assertLogs(mr.logger, logging.INFO) as logs,
        ):
            out = self.run_step(rows=8)
        self.assertEqual(out, "stock")
        self.assertEqual(self.names(), ["gate", "topk", "stock_call"])
        self.assertIs(_STATE["log"][-1][1], self.x)
        self.assertEqual(self.st.ran_ops, [("moe.routing", "stock")])
        self.assertIn("tensor layout", logs.output[0])


class TestRows(_Case):
    def setUp(self):
        super().setUp()
        # FlashInfer's block-per-token routing split off, as the tuned runs set it
        patcher = mock.patch.dict(
            "os.environ", {"FLASHINFER_ROUTING_FORCE_BLOCK_PER_TOKEN": "0"}
        )
        patcher.start()
        self.addCleanup(patcher.stop)

    def assert_not_fused(self, reason=None, **kw):
        if reason is None:
            out = self.run_step(**kw)
        else:
            with self.assertLogs(mr.logger, logging.INFO) as logs:
                out = self.run_step(**kw)
            self.assertIn(reason, logs.output[0])
        self.assertIsNone(out)  # eager: no prequant, so the caller runs stock
        self.assertEqual(_STATE["log"], [])
        _STATE["log"].clear()

    def test_row_cap(self):
        self.assertIsNotNone(self.run_step(rows=512, options=mr.Options(max_rows=512)))
        _STATE["log"].clear()
        self.assert_not_fused(rows=9, options=mr.Options(max_rows=8))
        self.assert_not_fused(rows=513, options=mr.Options(max_rows=4096))

    def test_above_16_rows_needs_the_block_per_token_split_off(self):
        self.assertIsNotNone(self.run_step(rows=17))
        _STATE["log"].clear()
        self.assertIsNotNone(self.run_step(rows=16))
        _STATE["log"].clear()
        with mock.patch.dict("os.environ"):
            del os.environ["FLASHINFER_ROUTING_FORCE_BLOCK_PER_TOKEN"]
            self.assert_not_fused("above 16 rows", rows=17)

    def test_autotuning_runs_stock_without_a_log(self):
        _STATE["tuner"].is_tuning_mode = True
        with mock.patch.object(mr.logger, "info") as info:
            self.assert_not_fused(rows=8)
        info.assert_not_called()

    def test_other_layouts(self):
        for name, change in (
            ("routing config", lambda m: setattr(m.topk.topk_config, "top_k", 8)),
            (
                "routing config",
                lambda m: setattr(m.topk.topk_config, "correction_bias", 1),
            ),
            ("grouped", lambda m: setattr(m.topk.topk_config, "topk_group", 4)),
            (
                "routing method",
                lambda m: setattr(m.experts, "routing_method_type", _Routing.Default),
            ),
            ("expert layout", lambda m: setattr(m.experts, "num_local_experts", 256)),
            ("expert layout", lambda m: setattr(m.experts, "moe_ep_rank", 1)),
        ):
            with self.subTest(name):
                mr._log_once.cache_clear()
                mlp = _Mlp(_STATE["log"])
                change(mlp)
                self.assert_not_fused(name, rows=8, mlp=mlp)

    def test_not_built_or_off(self):
        self.assert_not_fused(rows=8, options=mr.Options(fused_routing=False))
        with mock.patch.object(mr, "_fused_built", False):
            self.assert_not_fused(rows=8)


class TestPrequant(_Case):
    def setUp(self):
        super().setUp()
        _STATE["capture"] = True

    def test_quant_beside_the_router_gemv(self):
        out = self.run_step(rows=600)
        self.assertEqual(out, "prequant")
        log = _STATE["log"]
        width = dict(wait=3, record_stream=3, quant=2, gate=2)
        self.assertEqual(
            [e[: width.get(e[0], 1)] for e in log],
            [
                ("wait", "alt", "current"),
                ("quant", "alt"),
                ("gate", "current"),
                ("topk",),
                ("wait", "current", "alt"),
                ("record_stream", "x_fp4", "current"),
                ("record_stream", "x_sf", "current"),
                ("run_moe_core",),
                ("combine",),
            ],
        )
        self.assertIs(log[1][2], self.x)
        self.assertIs(log[1][3], self.mlp.experts.w13_input_scale_quant)
        _, dispatch, deferred = log[7]
        self.assertTrue(deferred)
        self.assertEqual(dispatch.hidden_states.name, "x_fp4")
        self.assertEqual(dispatch.hidden_states_scale.name, "x_sf")
        self.assertEqual(dispatch.topk_output.topk_config, self.mlp.topk.topk_config)
        self.assertEqual(self.st.ran_ops, [("moe.routing", "prequant")])

    def test_takes_rows_the_fused_path_does_not(self):
        with mock.patch.object(mr, "_fused_built", False):
            self.assertEqual(self.run_step(rows=8), "prequant")
        _STATE["log"].clear()
        self.assertIsNotNone(self.run_step(rows=8).gemm2_out)  # fused when built

    def test_needs_capture_an_alt_stream_and_the_option(self):
        self.assertIsNone(
            self.run_step(rows=600, options=mr.Options(prequant_overlap=False))
        )
        mlp = _Mlp(_STATE["log"])
        mlp.alt_stream = None
        self.assertIsNone(self.run_step(rows=600, mlp=mlp))
        _STATE["capture"] = False
        self.assertIsNone(self.run_step(rows=600))
        self.assertEqual(_STATE["log"], [])


class _Block(_Mlp):
    """The fake Qwen2MoeSparseMoeBlock as moe.step drives it: the deferred output, and
    SGLang's router and shared expert, each logging the stream it runs on."""

    shared_expert = "shared expert"

    def __init__(self, log, rows):
        super().__init__(log)
        self.shared_expert_gate = SimpleNamespace(
            weight=torch.zeros(1, HIDDEN, dtype=torch.bfloat16)
        )
        self.experts.dispatcher = SimpleNamespace(combine=self.combine)
        self.out = SimpleNamespace(
            top_k=TOP_K,
            gemm2_out=torch.zeros(rows * TOP_K, HIDDEN, dtype=torch.bfloat16),
            expert_weights=torch.zeros(rows, TOP_K),
            expanded_idx_to_permuted_idx=torch.zeros(rows * TOP_K, dtype=torch.int32),
        )

    def combine(self, combine_input):
        self.log.append(("combine",))
        return self.out

    def _forward_router_experts(self, x, defer_finalize):
        self.log.append(("routed", _STATE["on"].name, defer_finalize))
        return self.out

    def _forward_shared_experts(self, x, apply_gate):
        self.log.append(("shared", _STATE["on"].name, apply_gate, x))
        return torch.zeros_like(x)


class _Kernel:  # the finalize kernel: logs its launch
    def __getitem__(self, grid):
        return lambda *args, **kw: _STATE["log"].append(
            ("finalize", _STATE["on"].name, grid, kw)
        )


class TestExtendStep(_Case):
    """Slot moe_finalize from EXTEND_MIN_ROWS rows: the source's extend-size MoE, in a
    breakable prefill graph's capture as eagerly."""

    WIDTH = dict(
        wait=3, quant=2, record_event=2, shared=3, gate=2, wait_event=3, routed=3,
        record_stream=3, finalize=3,
    )  # fmt: skip

    def setUp(self):
        super().setUp()
        _STATE["capture"] = True
        self.pdl = True
        kernel = types.ModuleType(
            "sgl_nvidia_mega.qwen3_8.kernels.ops.elementwise.moe_finalize"
        )
        kernel._fused_finalize_gate_sigmoid_mul_add_kernel = _Kernel()
        qwen2 = types.ModuleType("sglang.srt.models.qwen2_moe")
        qwen2.Qwen2MoeSparseMoeBlock = _Block
        for module in (kernel, qwen2):
            self.addCleanup(_restore, module.__name__, sys.modules.get(module.__name__))
            sys.modules[module.__name__] = module

        def record_stream(t, stream):
            _STATE["log"].append(("record_stream", tuple(t.shape), stream.name))

        for patcher in (
            mock.patch(
                "sglang.kernels.jit.utils.is_arch_support_pdl", lambda: self.pdl
            ),
            mock.patch.object(torch.Tensor, "record_stream", record_stream),
        ):
            patcher.start()
            self.addCleanup(patcher.stop)

    def run_moe(self, rows=moe.EXTEND_MIN_ROWS, options=mr.DEFAULT, inplace=False):
        self.mlp = _Block(_STATE["log"], rows)
        self.mlp.experts.moe_runner_config.inplace = inplace
        self.x = torch.randn(rows, HIDDEN).to(torch.bfloat16)
        self.st = _St()
        executor = SimpleNamespace(
            moe_finalize_ready=True, moe_routing=options, smallm_gemm=None
        )
        out = moe.step(executor, SimpleNamespace(mlp=self.mlp), self.x, self.st)
        self.assertEqual(tuple(out.shape), (rows, HIDDEN))
        log = _STATE["log"]
        return [e[: self.WIDTH.get(e[0], 1)] for e in log], log[-1][3]

    def test_capture_quant_and_shared_beside_the_routed_stream(self):
        rows = moe.EXTEND_MIN_ROWS
        order, launch = self.run_moe()
        self.assertEqual(
            order,
            [
                ("wait", "alt", "current"),
                ("quant", "current"),
                ("record_event", "current"),
                ("shared", "current", False),
                ("gate", "alt"),
                ("topk",),
                ("wait_event", "alt", "current"),
                ("run_moe_core",),
                ("combine",),
                ("wait", "current", "alt"),
                ("record_stream", (rows * TOP_K, HIDDEN), "current"),
                ("record_stream", (rows, TOP_K), "current"),
                ("record_stream", (rows * TOP_K,), "current"),
                ("finalize", "current", (rows,)),
            ],
        )
        log = _STATE["log"]
        self.assertIs(log[1][2], self.x)
        self.assertIs(log[1][3], self.mlp.experts.w13_input_scale_quant)
        _, dispatch, deferred = log[7]
        self.assertTrue(deferred)
        self.assertEqual(dispatch.hidden_states.name, "x_fp4")
        self.assertEqual(dispatch.hidden_states_scale.name, "x_sf")
        self.assertEqual(
            {k: launch[k] for k in ("BLOCK_SIZE", "num_warps", "maxnreg", "USE_PDL")},
            dict(BLOCK_SIZE=4096, num_warps=8, maxnreg=64, USE_PDL=True),
        )
        self.assertTrue(launch["launch_pdl"])
        self.assertEqual(
            self.st.ran_ops, [("moe.routing", "prequant"), ("moe", "extend_finalize")]
        )

    def test_eager_shared_then_routed(self):
        _STATE["capture"] = False
        order, launch = self.run_moe(rows=4096)
        self.assertEqual(
            order,
            [
                ("shared", "current", False),
                ("routed", "current", True),
                ("finalize", "current", (4096,)),
            ],
        )
        self.assertEqual((launch["maxnreg"], launch["USE_PDL"]), (64, True))
        self.assertEqual(self.st.ran_ops, [("moe", "extend_finalize")])

    def test_capture_without_the_prequant_routes_on_the_side_stream(self):
        for options in (None, mr.Options(prequant_overlap=False)):
            with self.subTest(options=options):
                _STATE["log"].clear()
                order, _ = self.run_moe(options=options)
                self.assertEqual(
                    order[:4],
                    [
                        ("wait", "alt", "current"),
                        ("shared", "current", False),
                        ("routed", "alt", True),
                        ("wait", "current", "alt"),
                    ],
                )
                self.assertEqual(self.st.ran_ops, [("moe", "extend_finalize")])

    def test_no_pdl_launch_without_pdl(self):
        self.pdl = False
        _, launch = self.run_moe()
        self.assertFalse(launch["USE_PDL"])
        self.assertNotIn("launch_pdl", launch)
        self.assertEqual(launch["maxnreg"], 64)

    def test_fewer_rows_prequant_on_the_shared_stream(self):
        # rows the fused routing does not take: the source's prequant on the current
        # stream before the shared expert, the router and routed experts on the side stream
        rows = moe.EXTEND_MIN_ROWS - 1
        with mock.patch.object(mr, "_takes_fused", lambda *a: False):
            order, launch = self.run_moe(rows=rows)
        # nothing orders the shared expert after the router GEMM
        self.assertEqual(
            order[:8],
            [
                ("wait", "alt", "current"),
                ("quant", "current"),
                ("record_event", "current"),
                ("shared", "current", False),
                ("gate", "alt"),
                ("topk",),
                ("wait_event", "alt", "current"),
                ("run_moe_core",),
            ],
        )
        self.assertEqual(order[-1], ("finalize", "current", (rows,)))
        self.assertEqual(
            self.st.ran_ops[:2], [("moe.shared_input", "stock"), (mr.OP, "prequant")]
        )

    def run_fused(self, rows, no_clone, inplace=False):
        """The decode-size step on rows the fused routing takes (its routed half a spy)."""

        def fused_from_logits(mlp, x, logits, st):
            _STATE["log"].append(("fused_from_logits", _STATE["on"].name))
            st.ran(mr.OP, "fused_routing")
            return mlp.out

        with (
            mock.patch.object(mr, "_takes_fused", lambda *a: True),
            mock.patch.object(mr, "fused_from_logits", fused_from_logits),
            mock.patch.object(moe, "SHARED_NO_CLONE", no_clone),
        ):
            return self.run_moe(rows=rows, inplace=inplace)

    def test_fused_rows_shared_expert_beside_the_router_gemm(self):
        # the source's forward_normal_dual_stream with its fused routing: the shared expert
        # first on the current stream, without waiting for the router GEMM; the router
        # GEMM, the routing and the routed GEMMs on the side stream; the finalize after
        # the join
        for rows in (4, 16, 64, 176, 256, 320):
            for no_clone in (True, False):
                with self.subTest(rows=rows, no_clone=no_clone):
                    _STATE["log"].clear()
                    order, _ = self.run_fused(rows, no_clone)
                    self.assertEqual(
                        order,
                        [
                            ("wait", "alt", "current"),
                            ("shared", "current", False),
                            ("gate", "alt"),
                            ("fused_from_logits",),
                            ("wait", "current", "alt"),
                            ("record_stream", (rows * TOP_K, HIDDEN), "current"),
                            ("record_stream", (rows, TOP_K), "current"),
                            ("record_stream", (rows * TOP_K,), "current"),
                            ("finalize", "current", (rows,)),
                        ],
                    )
                    log = _STATE["log"]
                    self.assertEqual(log[3], ("fused_from_logits", "alt"))
                    # the shared expert reads the input itself, or SGLang's clone of it
                    self.assertIs(log[1][3] is self.x, no_clone)
                    self.assertTrue(torch.equal(log[1][3], self.x))
                    impl = "no_clone" if no_clone else "stock"
                    self.assertEqual(
                        self.st.ran_ops,
                        [
                            ("moe.shared_input", impl),
                            (mr.OP, "fused_routing"),
                            ("moe", "finalize_permuted"),
                        ],
                    )

    def test_inplace_runner_keeps_the_clone(self):
        # the source's condition: no clone only for a routed runner that does not write
        # its input in place
        self.run_fused(256, no_clone=True, inplace=True)
        self.assertIsNot(_STATE["log"][1][3], self.x)
        self.assertEqual(self.st.ran_ops[0], ("moe.shared_input", "stock"))

    def test_eager_has_no_clone(self):
        # the source's eager branch: the shared expert on the input, then the routed half
        _STATE["capture"] = False
        with mock.patch.object(moe, "SHARED_NO_CLONE", False):
            order, _ = self.run_moe(rows=8, options=None)
        self.assertEqual(
            order,
            [
                ("shared", "current", False),
                ("routed", "current", True),
                ("finalize", "current", (8,)),
            ],
        )
        self.assertIs(_STATE["log"][0][3], self.x)
        self.assertEqual(self.st.ran_ops, [("moe", "finalize_permuted")])

    def test_fewer_rows_take_the_source_dual_stream(self):
        # the source's forward_normal_dual_stream at decode sizes: the shared expert on the
        # current stream, the routed experts on the side stream, the finalize after the join
        rows = moe.EXTEND_MIN_ROWS - 1
        order, launch = self.run_moe(rows=rows, options=None)
        self.assertEqual(
            order,
            [
                ("wait", "alt", "current"),
                ("shared", "current", False),
                ("routed", "alt", True),
                ("wait", "current", "alt"),
                ("record_stream", (rows * TOP_K, HIDDEN), "current"),
                ("record_stream", (rows, TOP_K), "current"),
                ("record_stream", (rows * TOP_K,), "current"),
                ("finalize", "current", (rows,)),
            ],
        )
        # the source's decode-size launch: its warp rule (16) and PDL; one row a program
        # above 64 rows
        self.assertEqual((launch["USE_PDL"], launch["num_warps"]), (self.pdl, 16))
        self.assertNotIn("maxnreg", launch)
        self.assertNotIn("CHUNK", launch)
        self.assertEqual(
            self.st.ran_ops,
            [("moe.shared_input", "stock"), ("moe", "finalize_permuted")],
        )


_NO_CLONE_SWITCH = r"""
import importlib
import json
import os

from sgl_nvidia_mega.qwen3_8.runtime import moe

out = {}
for value in (None, "1", "true", "True", "0", "false", "yes"):
    os.environ.pop(moe.SHARED_NO_CLONE_ENV, None)
    if value is not None:
        os.environ[moe.SHARED_NO_CLONE_ENV] = value
    importlib.reload(moe)
    out[str(value)] = moe.SHARED_NO_CLONE
print(json.dumps(out))
"""


class TestSharedNoCloneSwitch(unittest.TestCase):
    def test_read_as_the_source_reads_its_switch(self):
        """get_bool_env_var's parsing, default off, under the recipes' name for the
        source's SGLANG_QWEN_MOE_SHARED_NO_CLONE."""
        import json
        import subprocess

        self.assertEqual(
            moe.SHARED_NO_CLONE_ENV, "SGLANG_OPT_MOE_SHARED_EXPERT_NO_CLONE"
        )
        env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1")
        done = subprocess.run(
            [sys.executable, "-c", _NO_CLONE_SWITCH],
            env=env,
            capture_output=True,
            text=True,
            timeout=300,
        )
        self.assertEqual(done.returncode, 0, done.stderr[-2000:])
        got = json.loads(done.stdout.strip().splitlines()[-1])
        self.assertEqual(
            got,
            {
                "None": False,
                "1": True,
                "true": True,
                "True": True,
                "0": False,
                "false": False,
                "yes": False,
            },
        )


class TestOptions(unittest.TestCase):
    def test_defaults_are_the_tuned_values(self):
        self.assertEqual(
            mr.DEFAULT,
            mr.Options(fused_routing=True, max_rows=512, prequant_overlap=True),
        )


class TestBuild(unittest.TestCase):
    def setUp(self):
        _install(self)
        mr._fused_built = False  # the patch restores it
        _STATE["kernel"] = True
        self.available = []

        def available():
            self.available.append(1)
            return _STATE["kernel"]

        patcher = mock.patch.object(op, "available", available)
        patcher.start()
        self.addCleanup(patcher.stop)

    def model(self, mlps=None):
        mlps = mlps if mlps is not None else [_Mlp([]), _Mlp([])]
        layers = [SimpleNamespace(mlp=mlp) for mlp in mlps]
        m = SimpleNamespace(layers=layers, start_layer=0, end_layer=len(layers))
        return SimpleNamespace(model=m)

    def build(self, model=None, options=mr.DEFAULT, capability=(10, 0)):
        with (
            mock.patch.object(torch.cuda, "is_available", return_value=True),
            mock.patch.object(
                torch.cuda, "get_device_capability", return_value=capability
            ),
        ):
            return mr.build(model or self.model(), options)

    def test_layer_reasons(self):
        self.assertEqual(mr.layer_reasons(_Mlp([])), "")
        for part, change in (
            ("LoRA", lambda m: _STATE.update(lora=True)),
            ("all-to-all", lambda m: _STATE.update(a2a_none=False)),
            ("FusedMoE", lambda m: setattr(m, "experts", SimpleNamespace())),
            ("ModelOpt", lambda m: setattr(m.experts, "quant_method", object())),
            (
                "TRT-LLM",
                lambda m: setattr(
                    m.experts,
                    "runner_backend",
                    SimpleNamespace(is_flashinfer_trtllm=lambda: False),
                ),
            ),
            (
                "per-token",
                lambda m: setattr(
                    m.experts.quant_method,
                    "quant_config",
                    SimpleNamespace(use_per_token_activation=True),
                ),
            ),
            ("padded", lambda m: setattr(m.experts, "trtllm_padded_hidden_size", 2688)),
            ("MegaMoE", lambda m: setattr(m.experts, "_mega_moe_nvfp4", True)),
            (
                "shared expert",
                lambda m: setattr(m, "enable_shared_expert_fusion", True),
            ),
        ):
            with self.subTest(part):
                _STATE.update(lora=False, a2a_none=True)
                mlp = _Mlp([])
                change(mlp)
                self.assertIn(part, mr.layer_reasons(mlp))

    def test_refusals(self):
        self.assertIn("SM10x", self.build(capability=(9, 0)))
        _STATE["tp"] = 2
        self.assertIn("TP2", self.build())
        _STATE["tp"] = 1
        bad = _Mlp([])
        bad.enable_shared_expert_fusion = True
        reason = self.build(self.model([_Mlp([]), bad]))
        self.assertTrue(reason.startswith("layer 1: "), reason)
        self.assertFalse(mr._fused_built)

    def test_builds_the_fused_path_once(self):
        self.assertEqual(self.build(), "")
        self.assertTrue(mr._fused_built)
        self.assertEqual(self.build(), "")
        self.assertEqual(self.available, [1])

    def assert_without_fused(self, part):
        with self.assertLogs(mr.logger, logging.WARNING) as logs:
            self.assertEqual(self.build(), "")
        self.assertFalse(mr._fused_built)
        self.assertIn(part, logs.output[0])

    def test_builds_on_each_mirrored_flashinfer(self):
        self.assertEqual(mr.FLASHINFER_VERSIONS, ("0.7.3", "0.7.4"))
        for version in mr.FLASHINFER_VERSIONS:
            with (
                self.subTest(version),
                mock.patch.object(sys.modules["flashinfer"], "__version__", version),
            ):
                mr._fused_built = False
                self.assertEqual(self.build(), "")
                self.assertTrue(mr._fused_built)

    def test_other_flashinfer_keeps_the_slot_without_fused_routing(self):
        for version in ("0.7.2", "0.7.5rc1"):
            with (
                self.subTest(version),
                mock.patch.object(sys.modules["flashinfer"], "__version__", version),
            ):
                self.assert_without_fused(f"FlashInfer {version}")

    def test_other_quant_recipe_keeps_the_slot_without_fused_routing(self):
        fp4_utils = sys.modules["sglang.srt.layers.quantization.fp4_utils"]
        with mock.patch.object(fp4_utils, "_flashinfer_fp4_quantize_backend", "cuda"):
            self.assert_without_fused("recipe")
        with mock.patch.dict("os.environ", {"FLASHINFER_NVFP4_4OVER6": "1"}):
            self.assert_without_fused("recipe")

    def test_kernel_that_does_not_build_keeps_the_slot(self):
        _STATE["kernel"] = False
        self.assert_without_fused("does not build")

    def test_fused_off_builds_nothing(self):
        self.assertEqual(self.build(options=mr.Options(fused_routing=False)), "")
        self.assertFalse(mr._fused_built)
        self.assertEqual(self.available, [])


if __name__ == "__main__":
    unittest.main()
