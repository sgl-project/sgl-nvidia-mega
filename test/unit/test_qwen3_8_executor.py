# SPDX-License-Identifier: Apache-2.0
"""The Qwen3.8 executor's host-side decisions on CPU: which forwards plan() covers,
the startup refusals and the engagement log. SGLang's model module and graph
helpers are replaced by fakes, so no GPU and no model weights are needed."""

import contextlib
import enum
import logging
import sys
import types
import unittest
from types import SimpleNamespace
from unittest import mock

import torch


class _Mode(enum.Enum):
    EXTEND = 1
    DECODE = 2
    MIXED = 3
    IDLE = 4
    TARGET_VERIFY = 5
    DRAFT_EXTEND_V2 = 6
    PREBUILT = 7

    def is_idle(self):
        return self is _Mode.IDLE

    def is_target_verify(self):
        return self is _Mode.TARGET_VERIFY

    def is_decode(self):
        return self is _Mode.DECODE

    def is_draft_extend_v2(self):
        return self is _Mode.DRAFT_EXTEND_V2

    def is_extend_or_draft_extend_or_mixed(self):
        return self in (_Mode.EXTEND, _Mode.MIXED)


class _Linear:  # the fake GDN layer type
    ple = None


class _Attention:
    ple = None


_STATE = {"breakable": False, "cp": False}


class _Holder:  # the fake _BreakablePLEBatch
    def __init__(self):
        self.batch = None


def _install_fakes():
    qwen4 = types.ModuleType("sglang.srt.models.qwen4_exp")
    qwen4.Qwen4ExpLinearDecoderLayer = _Linear
    qwen4.Qwen4ExpAttentionDecoderLayer = _Attention
    qwen4._shards_rows_over_cp = lambda fb: _STATE["cp"]
    qwen4._BreakablePLEBatch = _Holder
    qwen4._MixStreams = lambda mixer, normalized: ("mix", normalized)
    sys.modules["sglang.srt.models.qwen4_exp"] = qwen4
    bcg = types.ModuleType(
        "sglang.srt.model_executor.runner_backend_utils.breakable_cuda_graph"
    )
    bcg.is_in_breakable_cuda_graph = lambda: _STATE["breakable"]
    # outside a capture a break runs its body (tests of the real decorator: TestBreaks)
    bcg.eager_on_graph = lambda enable: lambda fn: fn
    sys.modules[bcg.__name__] = bcg


_install_fakes()

from sgl_nvidia_mega.qwen3_8 import runtime  # noqa: E402
from sgl_nvidia_mega.qwen3_8.runtime.engagement import Engagement  # noqa: E402
from sgl_nvidia_mega.qwen3_8.runtime.executor import (  # noqa: E402
    Qwen3_8Executor,
    Qwen3_8UcacheExecutor,
)


def _model(layers=(_Linear, _Linear, _Attention)):
    m = SimpleNamespace(
        layers=[cls() for cls in layers], start_layer=0, end_layer=len(layers)
    )
    return SimpleNamespace(model=m)


def _batch(mode, rows):
    return SimpleNamespace(forward_mode=mode, input_ids=torch.zeros(rows))


def _executor(draft=False):
    widths = tuple(range(1, 1025)) if draft else tuple(range(4, 1025, 4))
    return Qwen3_8Executor(_model(), widths=widths, draft=draft, slots=frozenset())


class TestPlan(unittest.TestCase):
    def setUp(self):
        _STATE.update(breakable=False, cp=False)

    def plan(self, ex, mode, rows, **kw):
        step = ex.plan(torch.zeros(rows), _batch(mode, rows), **kw)
        return None if step is None else (step.mode, step.rows)

    def test_target(self):
        ex = _executor()
        self.assertEqual(self.plan(ex, _Mode.TARGET_VERIFY, 8), ("target_verify", 8))
        self.assertEqual(
            self.plan(ex, _Mode.TARGET_VERIFY, 1024), ("target_verify", 1024)
        )
        self.assertIsNone(self.plan(ex, _Mode.TARGET_VERIFY, 6))  # not a built width
        self.assertIsNone(self.plan(ex, _Mode.TARGET_VERIFY, 1028))
        self.assertEqual(self.plan(ex, _Mode.DECODE, 4), ("decode", 4))
        self.assertIsNone(self.plan(ex, _Mode.DRAFT_EXTEND_V2, 8))
        self.assertEqual(self.plan(ex, _Mode.EXTEND, 16384), ("prefill", 16384))
        self.assertEqual(self.plan(ex, _Mode.MIXED, 37), ("prefill", 37))

    def test_draft(self):
        ex = _executor(draft=True)
        self.assertEqual(self.plan(ex, _Mode.DECODE, 3), ("decode", 3))
        self.assertEqual(self.plan(ex, _Mode.DRAFT_EXTEND_V2, 12), ("draft_extend", 12))
        self.assertIsNone(self.plan(ex, _Mode.TARGET_VERIFY, 8))
        self.assertEqual(self.plan(ex, _Mode.EXTEND, 5), ("prefill", 5))

    def test_declined(self):
        ex = _executor()
        self.assertIsNone(self.plan(ex, _Mode.IDLE, 0))
        self.assertIsNone(self.plan(ex, _Mode.PREBUILT, 4))
        self.assertIsNone(self.plan(ex, _Mode.TARGET_VERIFY, 8, get_embedding=True))
        self.assertIsNone(self.plan(ex, _Mode.TARGET_VERIFY, 8, pp_proxy_tensors=1))
        self.assertIsNone(self.plan(ex, _Mode.TARGET_VERIFY, 8, unknown=1))
        _STATE["breakable"] = True
        self.assertIsNone(self.plan(ex, _Mode.EXTEND, 64))
        _STATE.update(breakable=False, cp=True)
        self.assertIsNone(self.plan(ex, _Mode.EXTEND, 64))

    def test_mixers(self):
        ex = _executor()
        names = [ex.mixers[i].__name__ for i in range(3)]
        self.assertEqual(names, ["_gdn", "_gdn", "_qsa"])


class _St:
    def __init__(self, mode, rows=8):
        self.fb = SimpleNamespace(
            forward_mode=mode, spec_info=SimpleNamespace(draft_token_num=4)
        )
        self.positions = torch.zeros(rows)
        self.step = SimpleNamespace(
            mode="prefill" if mode in (_Mode.EXTEND, _Mode.MIXED) else str(mode),
            rows=rows,
        )
        self.ran_ops = []
        self.scratch = {}

    def ran(self, op, impl):
        self.ran_ops.append((op, impl))


class TestStepFallbacks(unittest.TestCase):
    """Each step function runs SGLang's module when its slot is off or the step is
    outside what the kernel serves, and records that it ran stock."""

    def setUp(self):
        qwen2 = types.ModuleType("sglang.srt.models.qwen2_moe")

        class Block:
            def __call__(self, x, fb=None, defer_finalize=None):
                return ("moe", defer_finalize)

        qwen2.Qwen2MoeSparseMoeBlock = Block
        sys.modules["sglang.srt.models.qwen2_moe"] = qwen2
        self.Block = Block

    def test_moe_stock(self):
        from sgl_nvidia_mega.qwen3_8.runtime import moe

        ex = SimpleNamespace(moe_finalize_ready=False)
        layer = SimpleNamespace(mlp=self.Block())
        st = _St(_Mode.TARGET_VERIFY)
        self.assertEqual(moe.step(ex, layer, torch.zeros(8, 2560), st), ("moe", False))
        self.assertEqual(st.ran_ops, [("moe", "stock")])

    def test_moe_rows_bound(self):
        """No rows, or another hidden size: SGLang's block (any row count above runs the
        slot, the extend sizes too: test_qwen3_8_moe_routing.TestExtendStep)."""
        from sgl_nvidia_mega.qwen3_8.runtime import moe

        ex = SimpleNamespace(moe_finalize_ready=True)
        layer = SimpleNamespace(mlp=self.Block())
        for x in (torch.zeros(0, 2560), torch.zeros(moe.EXTEND_MIN_ROWS, 4096)):
            st = _St(_Mode.EXTEND)
            moe.step(ex, layer, x, st)
            self.assertEqual(st.ran_ops, [("moe", "stock")])

    def test_moe_shared_before_routed(self):
        """Eager (not capturing): the source's order, the shared expert then the routed
        experts, then the finalize kernel."""
        from sgl_nvidia_mega.qwen3_8.runtime import moe

        calls = []
        runner = types.ModuleType("sglang.srt.model_executor.runner")
        runner.get_is_capture_mode = lambda: False
        fin = types.ModuleType(
            "sgl_nvidia_mega.qwen3_8.kernels.ops.elementwise.moe_finalize"
        )

        launches = []

        class Kernel:
            def __init__(self, name):
                self.name = name

            def __getitem__(self, grid):
                def launch(*a, **k):
                    calls.append("finalize")
                    launches.append((self.name, grid, k))

                return launch

        fin._fused_finalize_gate_sigmoid_mul_add_kernel = Kernel("rows")
        fin._fused_finalize_gate_sigmoid_mul_add_chunked_kernel = Kernel("chunked")
        saved = {n: sys.modules.get(n) for n in (runner.__name__, fin.__name__)}
        sys.modules.update({runner.__name__: runner, fin.__name__: fin})
        self.addCleanup(
            lambda: [
                sys.modules.pop(n) if v is None else sys.modules.__setitem__(n, v)
                for n, v in saved.items()
            ]
        )
        rows = 8

        class Block(self.Block):
            alt_stream = None
            shared_expert = None
            shared_expert_gate = SimpleNamespace(weight=torch.zeros(1, 2560))

            def _forward_router_experts(self, x, defer_finalize):
                calls.append("routed")
                return SimpleNamespace(
                    top_k=10,
                    gemm2_out=torch.zeros(rows * 10, 2560),
                    expanded_idx_to_permuted_idx=torch.zeros(
                        rows, 10, dtype=torch.int32
                    ),
                    expert_weights=torch.zeros(rows, 10),
                )

            def _forward_shared_experts(self, x, apply_gate):
                calls.append("shared")
                return torch.zeros_like(x)

        ex = SimpleNamespace(
            moe_finalize_ready=True, moe_routing=None, smallm_gemm=None
        )
        layer = SimpleNamespace(mlp=Block())
        st = _St(_Mode.TARGET_VERIFY)
        self.enterContext(mock.patch.object(moe, "FINALIZE_COLUMN_CHUNK", 1024))
        moe.step(ex, layer, torch.zeros(rows, 2560), st)
        self.assertEqual(calls, ["shared", "routed", "finalize"])
        self.assertEqual(st.ran_ops, [("moe", "finalize_permuted")])
        # the source's decode-size launch: 16 warps, PDL where supported, and up to 64
        # rows the (row, 1024-column) grid
        from sglang.kernels.jit.utils import is_arch_support_pdl

        pdl = is_arch_support_pdl()
        name, grid, kw = launches[0]
        self.assertEqual((name, grid), ("chunked", (rows, 3)))
        self.assertEqual((kw["CHUNK"], kw["num_warps"], kw["USE_PDL"]), (1024, 16, pdl))
        self.assertEqual(kw.get("launch_pdl", False), pdl)
        # the recipe's chunk 0 (the source's default): one row a program
        with mock.patch.object(moe, "FINALIZE_COLUMN_CHUNK", 0):
            moe.step(ex, layer, torch.zeros(rows, 2560), _St(_Mode.TARGET_VERIFY))
        name, grid, kw = launches[1]
        self.assertEqual((name, grid, "CHUNK" in kw), ("rows", (rows,), False))

    def test_gdn_stock_outside_verify(self):
        from sgl_nvidia_mega.qwen3_8.runtime import gdn

        ex = SimpleNamespace(gdn_pdl_ready=True, gdn_ucache=None, gdn_prefill=None)
        layer = SimpleNamespace(linear_attn=lambda x, fb: "gdn-stock")
        st = _St(_Mode.EXTEND)
        self.assertEqual(gdn.step(ex, layer, torch.zeros(8, 4), st), "gdn-stock")
        self.assertEqual(st.ran_ops, [("gdn", "stock")])

    def test_gdn_glue_declines_one_request(self):
        """The glue serves 2-8 requests; a 1-request verify falls through."""
        from sgl_nvidia_mega.qwen3_8.runtime import gdn

        ex = SimpleNamespace(
            gdn_glue_ready=True,
            gdn_pdl_ready=False,
            gdn_ucache=None,
            gdn_prefill=None,
            gdn_front_ready=False,
        )
        layer = SimpleNamespace(linear_attn=lambda x, fb: "gdn-stock")
        st = _St(_Mode.TARGET_VERIFY, rows=4)
        self.assertEqual(gdn.step(ex, layer, torch.zeros(4, 8), st), "gdn-stock")
        self.assertEqual(st.ran_ops, [("gdn", "stock")])

    def test_gdn_glue_runs_the_projections_around_one_launch(self):
        from unittest import mock

        from sgl_nvidia_mega.qwen3_8.runtime import gdn

        rows, seen = 8, {}
        kmod = types.ModuleType(
            "sgl_nvidia_mega.qwen3_8.kernels.ops.attention.linear.gdn_fused_verify_glue"
        )

        def glue(**kw):
            seen.update(kw)
            return torch.ones(rows, 6)

        kmod.gdn_fused_verify_glue = glue
        umod = types.ModuleType("sglang.srt.layers.attention.linear.utils")
        selected = []
        umod.select_verify_intermediate_state_indices = lambda idx, req, valid, size: (
            selected.append(valid) or idx
        )
        umod.pp_spec_stable_rows_enabled = lambda: False
        cache = SimpleNamespace(
            conv=[torch.zeros(1)],
            intermediate_conv_window=[torch.zeros(1)],
            temporal="ckpt",
            replayssm_d="d",
            replayssm_k="k",
            replayssm_g="g",
            replayssm_rawv=None,
            replayssm_rawk=None,
            replayssm_beta=None,
        )
        backend = SimpleNamespace(
            forward_metadata=SimpleNamespace(
                retrieve_next_token=None,
                retrieve_next_sibling=None,
                retrieve_parent_token=None,
                mamba_cache_indices=torch.arange(2),
            ),
            state_layer_cache=lambda layer_id: cache,
            state_pool=SimpleNamespace(
                replayssm_spec_write_pos="wp",
                replayssm_cache_base="cb",
                replayssm_is_flush="fl",
            ),
            verify_intermediate_state_indices=torch.arange(2),
            req_to_token_pool=SimpleNamespace(size=16),
        )
        attn = SimpleNamespace(
            layer_id=0,
            conv_weights="w",
            activation="silu",
            A_log="al",
            dt_bias="dt",
            num_k_heads=4,
            head_k_dim=128,
        )
        linear = SimpleNamespace(
            attn=attn,
            norm=SimpleNamespace(weight="nw", eps=1e-6, activation="swish"),
            _forward_input_proj=lambda x: ("qkvz", "ba"),
            out_proj=lambda core: (core * 2, None),
        )
        # slot gdn_front declines 8 rows before it imports its kernels
        ex = SimpleNamespace(
            gdn_glue_ready=True,
            gdn_pdl_ready=False,
            gdn_ucache=None,
            gdn_prefill=None,
            smallm_gemm=None,
            gdn_front_ready=True,
        )
        st = _St(_Mode.TARGET_VERIFY, rows=rows)
        st.fb.req_pool_indices = torch.arange(2)
        with (
            mock.patch.dict(sys.modules, {kmod.__name__: kmod, umod.__name__: umod}),
            mock.patch.object(gdn, "linear_backend", lambda: backend),
        ):
            out = gdn.step(
                ex, SimpleNamespace(linear_attn=linear), torch.zeros(rows, 8), st
            )
        self.assertTrue(torch.equal(out, torch.full((rows, 6), 2.0)))
        self.assertEqual(st.ran_ops, [("gdn", "verify_glue")])
        self.assertEqual((seen["mixed_qkvz"], seen["mixed_ba"]), ("qkvz", "ba"))
        self.assertEqual(
            (seen["draft_token_num"], seen["norm_activation"]), (4, "swish")
        )
        self.assertIsNone(seen["conv_bias"])
        # without PP stable rows, the source's call: no mask, the backend's own indices
        self.assertEqual(selected, [])
        self.assertIs(
            seen["intermediate_state_indices"],
            backend.verify_intermediate_state_indices,
        )

    def test_gdn_front_runs_one_request(self):
        """4 rows: the front's projection + conv, then the core with the fused norm on
        the packed [q|k|v|z|b|a] columns, then the out_proj; the window rows are
        selected once a step."""
        from unittest import mock

        from sgl_nvidia_mega.qwen3_8.runtime import gdn

        rows, hv, vd, qk = 4, 3, 2, 4  # packed: q 4 | k 4 | v 6 | z 6 | b 3 | a 3
        width = 2 * qk + 2 * hv * vd + 2 * hv
        seen, selects = {}, []
        pkg = "sgl_nvidia_mega.qwen3_8.kernels.ops.attention.gdn_front."
        front, core = (
            types.ModuleType(pkg + "mk_front"),
            types.ModuleType(pkg + "mk_core"),
        )

        def run(
            x, weight, conv_w, conv_b, conv_state, state_idx, window, w_rows, pad_slot
        ):
            seen["run"] = (weight, conv_b, state_idx, w_rows, pad_slot)
            return torch.arange(width, dtype=torch.float32).repeat(rows, 1), "qkv"

        def core_fn(qkv, a, b, *pools, z, norm_weight, normalized, eps):
            seen["core"] = (qkv, a[0].tolist(), b[0].tolist(), z[0].tolist(), pools)
            return normalized.fill_(1.0)

        front.run, core.core = run, core_fn
        conv = types.ModuleType("sglang.kernels.ops.mamba.causal_conv1d_triton")
        conv.PAD_SLOT_ID = -1
        umod = types.ModuleType("sglang.srt.layers.attention.linear.utils")
        umod.select_verify_intermediate_state_indices = lambda idx, req, valid, size: (
            selects.append(size) or idx
        )
        cache = SimpleNamespace(
            conv=["conv"],
            intermediate_conv_window=["window"],
            temporal="ckpt",
            replayssm_d="d",
            replayssm_k="k",
            replayssm_g="g",
            replayssm_rawv="dlo",
            replayssm_rawk="klo",
        )
        backend = SimpleNamespace(
            forward_metadata=SimpleNamespace(
                retrieve_next_token=None,
                retrieve_next_sibling=None,
                retrieve_parent_token=None,
                mamba_cache_indices=torch.tensor([5, -1]),
            ),
            state_layer_cache=lambda layer_id: cache,
            state_pool=SimpleNamespace(
                replayssm_spec_write_pos="wp",
                replayssm_cache_base="cb",
                replayssm_is_flush="fl",
            ),
            verify_intermediate_state_indices=torch.tensor([7, 8]),
            req_to_token_pool=SimpleNamespace(size=16),
        )
        attn = SimpleNamespace(
            layer_id=0,
            conv_weights="w",
            A_log="al",
            dt_bias="dt",
            q_dim=qk,
            k_dim=qk,
            v_dim=hv * vd,
            num_v_heads=hv,
            head_v_dim=vd,
        )
        linear = SimpleNamespace(
            attn=attn,
            norm=SimpleNamespace(weight="nw", eps=1e-6),
            _fused_in_proj_weight="packed_w",
            out_proj=lambda core: (core * 2, None),
        )
        ex = SimpleNamespace(
            gdn_front_ready=True,
            gdn_glue_ready=True,
            gdn_pdl_ready=False,
            gdn_ucache=None,
            gdn_prefill=None,
            smallm_gemm=None,
        )
        st = _St(_Mode.TARGET_VERIFY, rows=rows)
        st.fb.req_pool_indices = torch.tensor([3])
        mods = {m.__name__: m for m in (front, core, conv, umod)}
        with (
            mock.patch.dict(sys.modules, mods),
            mock.patch.object(gdn, "linear_backend", lambda: backend),
        ):
            for _ in range(2):  # two GDN layers of one step
                out = gdn.step(
                    ex, SimpleNamespace(linear_attn=linear), torch.zeros(rows, 8), st
                )
        self.assertTrue(torch.equal(out, torch.full((rows, hv * vd), 2.0)))
        self.assertEqual(st.ran_ops, [("gdn", "front")] * 2)
        self.assertEqual(selects, [16])
        weight, conv_b, state_idx, w_rows, pad_slot = seen["run"]
        self.assertEqual((weight, conv_b, pad_slot), ("packed_w", None, -1))
        self.assertEqual((state_idx.tolist(), w_rows.tolist()), ([5], [7]))
        qkv, a, b, z, pools = seen["core"]
        self.assertEqual(qkv, "qkv")
        # mk_core.core's order: A_log ... rings, state / replay indices, cursors, out
        self.assertEqual(pools[:8], ("al", "dt", "ckpt", "d", "k", "g", "dlo", "klo"))
        self.assertEqual(pools[8].tolist() + pools[9].tolist(), [5, -1, 3])
        self.assertEqual(pools[10:13], ("wp", "cb", "fl"))
        self.assertEqual(z, [float(c) for c in range(14, 20)])
        self.assertEqual((b, a), ([20.0, 21.0, 22.0], [23.0, 24.0, 25.0]))
        self.assertEqual(tuple(pools[13].shape), (rows, hv, vd))

    def test_qsa_prefill_slot_or_stock(self):
        from unittest import mock

        from sgl_nvidia_mega.qwen3_8.runtime import attention, qsa_prefill

        layer = SimpleNamespace(is_qsa=True, self_attention=lambda **kw: "qsa-stock")
        ex = SimpleNamespace(
            qsa_pool_ready=False,
            qsa_union=None,
            qsa_indexer=None,
            qsa_prefill="o",
            qsa_draft=None,
        )
        for served, want in ((None, "qsa-stock"), ("qsa-prefill", "qsa-prefill")):
            with mock.patch.object(qsa_prefill, "extend", return_value=served) as ext:
                out = attention.step(ex, layer, torch.zeros(8, 4), _St(_Mode.EXTEND))
                self.assertEqual(out, want)
                out = attention.step(ex, layer, torch.zeros(8, 4), _St(_Mode.DECODE))
                self.assertEqual(out, "qsa-stock")
                ext.assert_called_once()

    def test_draft_prefill_qsa_takes_the_prefill_path(self):
        """The draft's slots for its decode and draft extend leave its prefill (the
        body a breakable prefill graph captures) to qsa_prefill, as the target's."""
        from unittest import mock

        from sgl_nvidia_mega.qwen3_8.runtime import attention, qsa_prefill

        layer = SimpleNamespace(is_qsa=True, self_attention=lambda **kw: "qsa-stock")
        ex = SimpleNamespace(
            qsa_pool_ready=False,
            qsa_union="union",
            qsa_indexer="indexer",
            qsa_prefill="o",
            qsa_draft="glue",
        )
        with (
            mock.patch.object(qsa_prefill, "extend", return_value="qsa-prefill") as ext,
            mock.patch.object(attention, "_decomposed", side_effect=AssertionError),
            mock.patch.object(attention, "_draft_extend", side_effect=AssertionError),
        ):
            out = attention.step(ex, layer, torch.zeros(8, 4), _St(_Mode.EXTEND))
        self.assertEqual(out, "qsa-prefill")
        ext.assert_called_once()

    def test_gdn_prefill_slot_or_stock(self):
        from unittest import mock

        from sgl_nvidia_mega.qwen3_8.runtime import gdn, gdn_prefill

        ex = SimpleNamespace(gdn_pdl_ready=False, gdn_ucache=None, gdn_prefill="opts")
        layer = SimpleNamespace(linear_attn=lambda x, fb: "gdn-stock")
        for served, want in ((None, "gdn-stock"), ("gdn-prefill", "gdn-prefill")):
            with mock.patch.object(gdn_prefill, "extend", return_value=served) as ext:
                st = _St(_Mode.EXTEND)
                self.assertEqual(gdn.step(ex, layer, torch.zeros(8, 4), st), want)
                ext.assert_called_once()
                # a verify never asks the prefill kernels
                st = _St(_Mode.DECODE)
                self.assertEqual(
                    gdn.step(ex, layer, torch.zeros(8, 4), st), "gdn-stock"
                )
                ext.assert_called_once()

    def test_qsa_stock_when_off_or_prefill(self):
        from sgl_nvidia_mega.qwen3_8.runtime import attention

        calls = []
        layer = SimpleNamespace(
            is_qsa=True, self_attention=lambda **kw: calls.append(kw) or "qsa-stock"
        )
        for ready, mode in ((False, _Mode.TARGET_VERIFY), (True, _Mode.EXTEND)):
            st = _St(mode, 8)
            ex = SimpleNamespace(
                qsa_pool_ready=ready,
                qsa_union=None,
                qsa_indexer=None,
                qsa_prefill=None,
                qsa_draft=None,
            )
            out = attention.step(ex, layer, torch.zeros(8, 4), st)
            self.assertEqual(out, "qsa-stock")
            self.assertEqual(st.ran_ops, [("qsa", "stock")])
        self.assertEqual(len(calls), 2)


def _prefill_cfg(backend):
    return SimpleNamespace(
        cuda_graph_config=SimpleNamespace(prefill=SimpleNamespace(backend=backend))
    )


class TestReasons(unittest.TestCase):
    def test_breakable_prefill_graphs_need_the_binding_body(self):
        from unittest import mock

        with mock.patch.object(runtime, "gdn_graph_tables", return_value=False):
            with mock.patch.object(
                runtime, "prefill_graph_supported", return_value=True
            ):
                for backend in ("breakable", "Backend.BREAKABLE", "disabled"):
                    self.assertEqual(
                        runtime.config_reasons(_prefill_cfg(backend), None), []
                    )
                self.assertEqual(runtime.config_reasons(SimpleNamespace(), None), [])
                for backend in ("full", "tc_piecewise"):  # capture the whole step
                    self.assertTrue(runtime.config_reasons(_prefill_cfg(backend), None))
            # an SGLang without MegakernelSpec.prefill_graph
            with mock.patch.object(
                runtime, "prefill_graph_supported", return_value=False
            ):
                (reason,) = runtime.config_reasons(_prefill_cfg("breakable"), None)
                self.assertIn("prefill_graph", reason)
                self.assertEqual(
                    runtime.config_reasons(_prefill_cfg("disabled"), None), []
                )

    def test_flashinfer_gdn_tables_need_slot_gdn_prefill(self):
        from unittest import mock

        cfg = _prefill_cfg("breakable")
        with (
            mock.patch.object(runtime, "gdn_graph_tables", return_value=True),
            mock.patch.object(runtime, "prefill_graph_supported", return_value=True),
        ):
            with mock.patch.dict(runtime.os.environ, {runtime.SLOTS_ENV: "ple"}):
                (reason,) = runtime.config_reasons(cfg, None)
                self.assertIn("gdn_prefill", reason)
            with mock.patch.dict(
                runtime.os.environ, {runtime.SLOTS_ENV: "gdn_prefill"}
            ):
                self.assertEqual(runtime.config_reasons(cfg, None), [])

    def test_flashinfer_gdn_tables_refuse_attention_cp(self):
        from unittest import mock

        ctx = types.ModuleType("sglang.srt.runtime_context")
        ctx.get_parallel = lambda: SimpleNamespace(attn_cp_size=2)
        with (
            mock.patch.object(runtime, "gdn_graph_tables", return_value=True),
            mock.patch.object(runtime, "prefill_graphs_on", return_value=True),
            mock.patch.dict(sys.modules, {ctx.__name__: ctx}),
        ):
            self.assertTrue(runtime.model_reasons(SimpleNamespace(), None))
            ctx.get_parallel = lambda: SimpleNamespace(attn_cp_size=1)
            self.assertEqual(runtime.model_reasons(SimpleNamespace(), None), [])

    def test_model_reasons(self):
        pp = SimpleNamespace(is_first_rank=True, is_last_rank=False)
        self.assertTrue(runtime.model_reasons(SimpleNamespace(pp_group=pp), None))
        model = SimpleNamespace(capture_aux_hidden_states=True)
        self.assertTrue(runtime.model_reasons(model, None))
        self.assertEqual(runtime.model_reasons(SimpleNamespace(), None), [])


def _gdn_front_model():
    """One GDN layer of the TP1 shape slot gdn_front takes."""
    attn = SimpleNamespace(
        q_dim=2048,
        k_dim=2048,
        v_dim=6144,
        num_v_heads=48,
        num_q_heads=16,
        head_q_dim=128,
        num_k_heads=16,
        head_k_dim=128,
        bias=None,
        activation="silu",
        A_log=torch.zeros(48),
        dt_bias=torch.zeros(48, dtype=torch.bfloat16),
    )
    norm = SimpleNamespace(
        group_size=None,
        norm_before_gate=True,
        bias=None,
        activation="sigmoid",
        weight=torch.zeros(128, dtype=torch.bfloat16),
    )
    layer = _Linear()
    layer.linear_attn = SimpleNamespace(
        attn=attn,
        norm=norm,
        attn_tp_size=1,
        num_k_heads=16,
        num_v_heads=48,
        head_k_dim=128,
        head_v_dim=128,
        _fused_in_proj_weight=SimpleNamespace(
            shape=(16480, 2560), dtype=torch.bfloat16
        ),
    )
    m = SimpleNamespace(layers=[layer, _Attention()], start_layer=0, end_layer=2)
    return SimpleNamespace(model=m), layer.linear_attn


def _gdn_front_backend():
    bf16 = torch.bfloat16
    cache = SimpleNamespace(
        replayssm_d=torch.zeros(2, 1, 16, 1, dtype=bf16),
        replayssm_rawv="dlo",
        replayssm_rawk="klo",
        replayssm_beta=None,
        temporal=torch.zeros(1, dtype=bf16),
        conv=[torch.zeros(2, 4, 3, dtype=bf16)],
        intermediate_conv_window=[torch.zeros(2, 4, 4, 3, dtype=bf16)],
    )
    pool = SimpleNamespace(
        replayssm_cache_base="cb", replayssm_spec_fold=False, replayssm_is_kda=False
    )
    return SimpleNamespace(state_pool=pool, state_layer_cache=lambda i: cache), cache


class TestGdnFront(unittest.TestCase):
    def test_model_reasons(self):
        from sgl_nvidia_mega.qwen3_8.runtime import gdn

        model, linear = _gdn_front_model()
        self.assertEqual(gdn.front_reasons(model), "")
        for change in (
            lambda: setattr(linear, "_fused_in_proj_weight", None),
            lambda: setattr(linear._fused_in_proj_weight, "shape", (16480, 4096)),
            lambda: setattr(
                linear.norm, "activation", "swish"
            ),  # the core's is sigmoid
            lambda: setattr(linear, "attn_tp_size", 4),
            lambda: setattr(linear.attn, "bias", torch.zeros(1)),
            lambda: setattr(
                linear.attn, "A_log", torch.zeros(48, dtype=torch.bfloat16)
            ),
        ):
            model, linear = _gdn_front_model()
            change()
            self.assertTrue(gdn.front_reasons(model))

    def test_pool_reasons(self):
        from sgl_nvidia_mega.qwen3_8.runtime import gdn

        backend, _ = _gdn_front_backend()
        self.assertEqual(gdn.front_pool_reasons(backend, 0, 4), "")
        self.assertTrue(gdn.front_pool_reasons(backend, 0, 8))
        for name, value in (
            ("replayssm_beta", "beta"),  # the exact-raw rings
            ("replayssm_rawv", None),
            ("replayssm_d", torch.zeros(2, 1, 16, 1)),
            ("temporal", torch.zeros(1)),
        ):
            backend, cache = _gdn_front_backend()
            setattr(cache, name, value)
            self.assertTrue(gdn.front_pool_reasons(backend, 0, 4), name)

    def test_pool_check_turns_the_slot_off(self):
        from unittest import mock

        from sgl_nvidia_mega.qwen3_8.runtime import gdn

        ex = _executor()
        attn = SimpleNamespace(layer_id=3)
        ex.lm.model.layers[0].linear_attn = SimpleNamespace(attn=attn)
        ex.gdn_front_ready = True
        fb = SimpleNamespace(spec_info=SimpleNamespace(draft_token_num=4))
        with (
            mock.patch.object(gdn, "linear_backend", lambda: "backend"),
            mock.patch.object(gdn, "front_pool_reasons", return_value="no rings") as r,
        ):
            ex._check_pools(fb)
        r.assert_called_once_with("backend", 3, 4)
        self.assertFalse(ex.gdn_front_ready)

    def test_front_compiles_eagerly_never_while_capturing(self):
        """A layout compiles at its first eager call and a capture reuses it; a cold
        layout while capturing raises before any compile or launch."""
        import importlib
        from unittest import mock

        import sgl_nvidia_mega.qwen3_8.kernels.ops.attention.gdn_front as package

        calls, capturing = [], [False]
        cuda, bindings, driver = (
            types.ModuleType(n)
            for n in ("cuda", "cuda.bindings", "cuda.bindings.driver")
        )
        cuda.bindings, bindings.driver, driver.CUstream = bindings, driver, lambda s: s
        cutlass, cute, runtime = (
            types.ModuleType(n)
            for n in ("cutlass", "cutlass.cute", "cutlass.cute.runtime")
        )
        cutlass.cute, cute.runtime, cutlass.BFloat16 = cute, runtime, "bf16"
        cute.experimental = SimpleNamespace(
            compile=lambda op, *args: (
                calls.append("compile") or (lambda *a: calls.append("launch"))
            )
        )
        runtime.from_dlpack = lambda t, assumed_align: t
        kernel = types.ModuleType(package.__name__ + ".mk_front_kernel")
        kernel.MkGdnFront = lambda **kw: SimpleNamespace(**kw)
        mods = (cuda, bindings, driver, cutlass, cute, runtime, kernel)
        mods = {m.__name__: m for m in mods}
        bf = lambda *shape: torch.empty(shape, dtype=torch.bfloat16, device="meta")
        idx = lambda: torch.empty(1, dtype=torch.int32, device="meta")
        args = lambda: (
            bf(4, 2560),
            bf(16480, 2560),
            bf(10240, 4),
            None,
            bf(3, 10240, 3),
            idx(),
            bf(2, 4, 10240, 3),
            idx(),
        )
        cuda_fns = dict(
            is_current_stream_capturing=lambda: capturing[0],
            current_stream=lambda d: SimpleNamespace(cuda_stream=7),
            get_device_capability=lambda d: (10, 7),
        )
        self.addCleanup(lambda: vars(package).pop("mk_front", None))
        with (
            mock.patch.dict(sys.modules, mods),
            mock.patch.multiple(torch.cuda, **cuda_fns),
        ):
            sys.modules.pop(package.__name__ + ".mk_front", None)
            front = importlib.import_module(package.__name__ + ".mk_front")
            packed, conv = front.run(*args(), pad_slot=-1)
            self.assertEqual(
                tuple(packed.shape) + tuple(conv.shape), (4, 16480, 4, 10240)
            )
            capturing[0] = True
            front.run(*args(), pad_slot=-1)
            self.assertEqual(calls, ["compile", "launch", "launch"])
            with self.assertRaises(RuntimeError):
                front.run(*args(), pad_slot=0)  # a layout no eager step compiled
            self.assertEqual(calls, ["compile", "launch", "launch"])


class _QsaAttn:  # the layer's RadixAttention: SGLang's sparse attention when called
    layer_id, scaling = 7, 0.0625

    def __init__(self, calls):
        self.calls = calls

    def __call__(self, q, k, v, fb, topk_indices=None):
        self.calls.append("stock")
        return torch.full((q.shape[0], 24 * 256), 5.0)


class _KvRouter:
    def __init__(self, dtype, shape=(64, 2, 256)):
        self.k = torch.empty(shape, dtype=dtype)
        self.v = torch.empty(shape, dtype=dtype)
        self.reads = []

    def layer(self, layer_id):
        return SimpleNamespace(key=lambda: self.k, value=lambda: self.v)

    def read(self, layer_id):
        self.reads.append(layer_id)
        return self.layer(layer_id)


def _qsa_backend(dtype, positions="qpos"):
    md = SimpleNamespace(
        indexer_metadata=SimpleNamespace(decode_logical_positions=positions),
        row_req_pool_indices="row_requests",
        sequence_lengths="lengths",
    )
    return SimpleNamespace(
        token_to_kv_pool=SimpleNamespace(layer_router=_KvRouter(dtype)),
        req_to_token_pool=SimpleNamespace(req_to_token="req_to_token"),
        _resolve_metadata=lambda fb: md,
        _uses_block_indices=lambda indices: indices.shape[1] == 512,
        _fused_kv_pool_eligible=True,
        compress_ratio=4,
    )


class _Kernels:  # PoolKernels / UnionKernels doubles that record their inputs
    def __init__(self, out, impl=None):
        self.out, self.impl, self.seen = out, impl, []

    def run(self, **kw):
        self.seen.append(kw)
        rows = kw["q"].shape[0]
        out = torch.full((rows, 24 * 256), self.out)
        return out if self.impl is None else (out, self.impl)


class TestQsaPoolFp8(unittest.TestCase):
    """Slot qsa_pool_fp8: which verify widths and pools run the union kernels, what
    they read, and the build / pool checks that turn the slot off."""

    def setUp(self):
        from unittest import mock

        elementwise = types.ModuleType("sglang.kernels.ops.elementwise.elementwise")
        self.gated = []
        elementwise.fused_sigmoid_mul = lambda x, gate, inplace: (
            self.gated.append(float(x[0, 0])) or x + 1
        )
        patch = mock.patch.dict(sys.modules, {elementwise.__name__: elementwise})
        patch.start()
        self.addCleanup(patch.stop)

    def run_step(
        self,
        dtype,
        rows,
        *,
        union=True,
        split_k=True,
        mode=None,
        draft=4,
        positions="qpos",
        blocks=512,
        q_dtype=torch.bfloat16,
    ):
        from unittest import mock

        from sgl_nvidia_mega.qwen3_8.runtime import attention

        calls = []
        layer = SimpleNamespace(
            is_qsa=True,
            alt_stream=None,
            attn=_QsaAttn(calls),
            _prepare_qkv_gate=lambda positions, hidden_states, forward_batch: (
                torch.zeros(rows, 24 * 256, dtype=q_dtype),
                torch.zeros(rows, 2 * 256),
                torch.zeros(rows, 2 * 256),
                torch.zeros(rows, 24 * 256),
            ),
            _compute_qsa_topk_indices=lambda x, positions, fb: torch.zeros(
                rows, blocks, dtype=torch.int64
            ),
            o_proj=lambda x: (x, None),
        )
        ex = SimpleNamespace(
            qsa_pool_ready=split_k,
            qsa_pool=_Kernels(3.0),
            qsa_core=None,
            qsa_union=_Kernels(1.0, "union_ws_fp8") if union else None,
            qsa_indexer=None,
            qsa_draft=None,
            qsa_prefill=None,
            smallm_gemm=None,
        )
        st = _St(mode or _Mode.TARGET_VERIFY, rows)
        st.fb.spec_info.draft_token_num = draft
        st.fb.req_pool_indices = torch.zeros(rows // 4)
        st.fb.out_cache_loc = "out_cache_loc"
        backend = _qsa_backend(dtype, positions)
        with mock.patch.object(attention, "sparse_backend", lambda: backend):
            out = attention.step(ex, layer, torch.zeros(rows, 8), st)
        impl = [i for op, i in st.ran_ops if op == "qsa"]
        return impl, out, ex, calls, backend

    def test_widths_and_pools(self):
        fp8, bf16 = torch.float8_e4m3fn, torch.bfloat16
        for dtype, rows, union, want in (
            (bf16, 8, False, "pool_splitk"),
            (bf16, 64, False, "pool_splitk"),
            (bf16, 128, False, "stock"),  # the pool check turned the union off
            (fp8, 8, True, "stock"),
            (fp8, 32, True, "stock"),
            (fp8, 52, True, "stock"),  # below the measured bound (56)
            (fp8, 56, True, "union_ws_fp8"),
            (fp8, 64, True, "union_ws_fp8"),  # the split-K kernels read BF16 only
            (fp8, 288, True, "union_ws_fp8"),
            (fp8, 512, True, "union_ws_fp8"),
            (fp8, 576, True, "union_ws_fp8"),
            (fp8, 580, True, "stock"),
        ):
            impl, _, ex, calls, _ = self.run_step(dtype, rows, union=union)
            self.assertEqual(impl, [want], (dtype, rows))
            self.assertEqual(calls, ["stock"] if want == "stock" else [], (dtype, rows))
            if union:
                self.assertEqual(len(ex.qsa_union.seen), want != "stock", rows)

    def test_union_reads_and_gate(self):
        rows = 288
        impl, out, ex, _, backend = self.run_step(torch.float8_e4m3fn, rows)
        self.assertEqual(impl, ["union_ws_fp8"])
        (kw,) = ex.qsa_union.seen
        self.assertEqual(tuple(kw["q"].shape), (rows, 24, 256))
        self.assertEqual(tuple(kw["new_k"].shape), (rows, 2, 256))
        self.assertEqual(tuple(kw["new_v"].shape), (rows, 2, 256))
        self.assertEqual(kw["indices"].dtype, torch.int32)
        self.assertEqual(tuple(kw["indices"].shape), (rows, 512))
        router = backend.token_to_kv_pool.layer_router
        self.assertIs(kw["key_cache"], router.k)
        self.assertIs(kw["value_cache"], router.v)
        self.assertEqual(router.reads, [7])  # waits for a layer-wise load-back
        self.assertEqual(
            (kw["locations"], kw["slot_map"], kw["row_requests"]),
            ("out_cache_loc", "req_to_token", "row_requests"),
        )
        self.assertEqual(kw["sequence_lengths"], "lengths")
        self.assertEqual(kw["query_positions"], "qpos")
        self.assertEqual(kw["scale"], 0.0625)
        self.assertNotIn("gate", kw)
        # SGLang's gate on the union output, then o_proj on the gated rows
        self.assertEqual(self.gated, [1.0])
        self.assertTrue(torch.equal(out, torch.full((rows, 24 * 256), 2.0)))

    def test_other_steps_run_stock(self):
        fp8 = torch.float8_e4m3fn
        for kw in (
            dict(mode=_Mode.DECODE),
            dict(draft=3),
            dict(positions=None),  # no logical positions: nothing launched
            dict(blocks=2048),  # token indices, not blocks
            dict(q_dtype=torch.float16),
        ):
            impl, _, ex, calls, _ = self.run_step(fp8, 128, **kw)
            self.assertEqual(impl, ["stock"], kw)
            self.assertEqual(calls, ["stock"], kw)
            self.assertEqual(ex.qsa_union.seen, [], kw)

    def test_union_kernels_workspace_and_launch(self):
        from unittest import mock

        from sgl_nvidia_mega.qwen3_8.qsa import pool_union_ws
        from sgl_nvidia_mega.qwen3_8.runtime import attention

        launches = []
        launch = lambda *args, **kw: launches.append((args, kw))
        for rows, served in ((36, True), (288, True), (512, False)):
            launches.clear()
            requests = rows // 4
            with (
                mock.patch.object(pool_union_ws, "qsa_pool_union_decode_ws", launch),
                mock.patch.object(pool_union_ws, "supported", lambda q, k, w: served),
            ):
                out, impl = attention.UnionKernels().run(
                    q=torch.zeros(rows, 24, 256, dtype=torch.bfloat16),
                    key_cache="k",
                    value_cache="v",
                    new_k="nk",
                    new_v="nv",
                    locations="loc",
                    slot_map="map",
                    row_requests="req",
                    indices="idx",
                    sequence_lengths="len",
                    query_positions="pos",
                    scale=0.0625,
                )
            self.assertEqual(impl, "union_ws_fp8" if served else "union_fp8")
            self.assertEqual(tuple(out.shape), (rows, 6144))
            self.assertEqual(out.dtype, torch.bfloat16)
            ((args, kw),) = launches
            names = ("k", "v", "nk", "nv", "loc", "map", "req", "idx", "len", "pos")
            self.assertEqual(args[1:], names + (4, 0.0625))
            self.assertIs(kw["out"], out)
            self.assertEqual(
                (kw["splits"], kw["union_warps"], kw["pdl"], kw["ws_pdl"], kw["ratio"]),
                (2, 16, True, True, 4),
            )
            # token slots, row masks, sort scratch, counts a request, a row; m, l, acc
            stats = requests * 2 * 2 * 64
            sizes = (requests * 8208,) * 2 + (requests * 2048, requests, rows)
            sizes += (stats, stats, stats * 256)
            self.assertEqual(tuple(t.numel() for t in kw["workspace"]), sizes)
            for t in kw["workspace"]:
                self.assertEqual(t.storage_offset() * t.element_size() % 16, 0, rows)

    def test_build_reasons(self):
        from unittest import mock

        from sgl_nvidia_mega.qwen3_8.qsa import pool_union_ws
        from sgl_nvidia_mega.qwen3_8.runtime import attention

        with mock.patch.object(torch.cuda, "is_available", lambda: False):
            self.assertIn("SM10x", attention.union_build())
        sm10x = dict(
            is_available=lambda: True, get_device_capability=lambda *a: (10, 7)
        )
        with mock.patch.multiple(torch.cuda, **sm10x):
            with mock.patch.object(pool_union_ws, "HAS_GLUON", False):
                self.assertIn("Gluon", attention.union_build())
            with mock.patch.object(pool_union_ws, "HAS_GLUON", True):
                self.assertEqual(attention.union_build(), "")
                with mock.patch.object(attention, "UNION_SPLITS", 0):
                    self.assertIn(attention.UNION_SPLITS_ENV, attention.union_build())

    def test_build_agrees_and_checks_the_layers(self):
        from unittest import mock

        from sgl_nvidia_mega.qwen3_8.runtime import attention

        binding = types.ModuleType("sglang.srt.kernel_plugins.nvidia.binding")
        agreed = []
        binding.megakernel_agree = lambda reason, what: agreed.append((reason, what))
        self.assertIn("qsa_pool_fp8", runtime.KNOWN_SLOTS)
        with (
            mock.patch.dict(sys.modules, {binding.__name__: binding}),
            mock.patch.object(attention, "union_build", return_value=""),
        ):
            for heads, built in (((24, 2, 256), True), ((6, 1, 256), False)):
                agreed.clear()
                model = _model((_Linear, _Attention))
                layer = model.model.layers[1]
                layer.is_qsa = True
                layer.attn = SimpleNamespace(
                    tp_q_head_num=heads[0], tp_k_head_num=heads[1], head_dim=heads[2]
                )
                kernels = runtime._build_qsa_union(model)
                self.assertEqual(isinstance(kernels, attention.UnionKernels), built)
                ((reason, what),) = agreed
                self.assertEqual(what, "Qwen3.8 slot qsa_pool_fp8")
                self.assertEqual(bool(reason), not built)

    def test_pool_reasons(self):
        from unittest import mock

        from sgl_nvidia_mega.qwen3_8.qsa import pool_decode
        from sgl_nvidia_mega.qwen3_8.runtime import attention

        fp8 = torch.float8_e4m3fn
        with mock.patch.object(pool_decode, "write_new_kv_e4m3_ok", return_value=True):
            self.assertEqual(attention.union_pool_reasons(_qsa_backend(fp8), 7), "")
            reason = attention.union_pool_reasons(_qsa_backend(torch.bfloat16), 7)
            self.assertIn("bfloat16", reason)
            strided = torch.empty(64, 256, 2, dtype=fp8).transpose(1, 2)
            for change in (
                # a scaled or non-token pool
                lambda b: setattr(b, "_fused_kv_pool_eligible", False),
                lambda b: setattr(b, "compress_ratio", 8),
                lambda b: setattr(b.token_to_kv_pool.layer_router, "k", strided),
                lambda b: setattr(
                    b.token_to_kv_pool, "layer_router", _KvRouter(fp8, (64, 1, 256))
                ),
            ):
                backend = _qsa_backend(fp8)
                change(backend)
                self.assertTrue(attention.union_pool_reasons(backend, 7))
        with mock.patch.object(pool_decode, "write_new_kv_e4m3_ok", return_value=False):
            self.assertIn("cast", attention.union_pool_reasons(_qsa_backend(fp8), 7))

    def test_pool_check_turns_the_slot_off(self):
        from unittest import mock

        from sgl_nvidia_mega.qwen3_8.runtime import attention

        for reason, draft in (
            ("KV pool dtype torch.bfloat16", False),
            ("", False),
            ("KV pool dtype torch.bfloat16", True),  # the draft's own pool
            ("", True),
        ):
            ex = _executor(draft=draft)
            ex.lm.model.layers[2].attn = SimpleNamespace(layer_id=7)
            ex.qsa_union = "kernels"
            with (
                mock.patch.object(attention, "sparse_backend", lambda: "backend"),
                mock.patch.object(
                    attention, "union_pool_reasons", return_value=reason
                ) as r,
            ):
                fb = SimpleNamespace(spec_info=SimpleNamespace(draft_token_num=4))
                ex._check_pools(fb)
            r.assert_called_once_with("backend", 7)
            self.assertEqual(ex.qsa_union, None if reason else "kernels")

    def test_pools_are_checked_at_the_first_verify_or_draft_extend(self):
        from sglang.kernels.plugin_kernels.nvidia.spec import Step

        for draft, modes, first in (
            (False, ("decode", "target_verify", "target_verify"), 1),
            (True, ("decode", "draft_extend", "draft_extend"), 1),
        ):
            ex = _executor(draft=draft)
            checked = []

            def check(fb, ex=ex, checked=checked):
                checked.append(fb)
                ex._pools_checked = True

            ex._check_pools = check
            ex._draft = ex._target = lambda st: st.step.mode
            batches = [SimpleNamespace(forward_mode=m) for m in modes]
            for mode, fb in zip(modes, batches):
                ex._forward(Step(mode=mode, rows=4), None, "pos", fb)
            self.assertEqual(len(checked), 1, draft)
            self.assertIs(checked[0], batches[first], draft)

    def draft_extend(
        self,
        rows,
        *,
        indexer=False,
        glue=True,
        union=True,
        tokens=None,
        q_dtype=torch.bfloat16,
        requests=None,
        loc=None,
        **backend_kw,
    ):
        """One draft extend step of the draft's QSA layer, with the indexer's expanded
        token indices from SGLang's indexer (with or without slot qsa_draft's capture)
        or the plugin's."""
        from unittest import mock

        from sgl_nvidia_mega.qwen3_8.runtime import attention, qsa_draft, qsa_indexer

        calls = []
        if tokens is None:
            tokens, _, _ = _expanded(rows)
        layer = SimpleNamespace(
            is_qsa=True,
            layer_id=7,
            alt_stream=None,
            attn=_QsaAttn(calls),
            _prepare_qkv_gate=lambda positions, hidden_states, forward_batch: (
                torch.zeros(rows, 24 * 256, dtype=q_dtype),
                torch.zeros(rows, 2 * 256),
                torch.zeros(rows, 2 * 256),
                torch.zeros(rows, 24 * 256),
            ),
            o_proj=lambda x: (x, None),
        )
        ex = SimpleNamespace(
            qsa_pool_ready=False,
            qsa_pool=None,
            qsa_core=None,
            qsa_union=_Kernels(1.0, "union_ws_fp8") if union else None,
            qsa_indexer="kernels" if indexer else None,
            qsa_draft="glue" if glue else None,
            qsa_prefill=None,
            smallm_gemm=None,
        )
        st = _St(_Mode.DRAFT_EXTEND_V2, rows)
        st.fb.req_pool_indices = torch.zeros(requests or rows // 4)
        st.fb.out_cache_loc = torch.arange(rows) if loc is None else loc
        backend = _draft_extend_backend(rows, **backend_kw)
        runner = types.ModuleType("sglang.srt.model_executor.runner")
        runner.get_is_capture_mode = lambda: True
        selected = lambda *a: tokens
        with (
            mock.patch.dict(sys.modules, {runner.__name__: runner}),
            mock.patch.object(
                sys.modules["sglang.srt.models.qwen4_exp"],
                "_QSA_INDEXER_OVERLAP_TOKEN_THRESHOLD",
                1024,
                create=True,
            ),
            mock.patch.object(attention, "sparse_backend", lambda: backend),
            mock.patch.object(qsa_draft, "indices", selected),
            mock.patch.object(qsa_indexer, "select", selected),
        ):
            out = attention.step(ex, layer, torch.zeros(rows, 8), st)
        impl = [i for op, i in st.ran_ops if op == "qsa"]
        return impl, out, ex, calls, backend

    def test_draft_extend_widths(self):
        for indexer, glue in ((False, True), (True, True), (False, False)):
            for rows, want in (
                (8, "stock"),
                (52, "stock"),  # below the verify's bound
                (56, "union_ws_fp8_draft"),
                (288, "union_ws_fp8_draft"),
                (576, "union_ws_fp8_draft"),
                (580, "stock"),
            ):
                impl, _, ex, calls, _ = self.draft_extend(
                    rows, indexer=indexer, glue=glue
                )
                self.assertEqual(impl, [want], (indexer, glue, rows))
                self.assertEqual(calls, ["stock"] if want == "stock" else [], rows)
                self.assertEqual(len(ex.qsa_union.seen), want != "stock", rows)
            impl, _, _, calls, _ = self.draft_extend(288, indexer=indexer, union=False)
            self.assertEqual((impl, calls), (["stock"], ["stock"]))

    def test_draft_extend_reads_blocks_and_gate(self):
        rows = 288
        tokens, blocks, valid = _expanded(rows)
        impl, out, ex, _, backend = self.draft_extend(rows, tokens=tokens)
        self.assertEqual(impl, ["union_ws_fp8_draft"])
        (kw,) = ex.qsa_union.seen
        got = kw["indices"]
        self.assertEqual((tuple(got.shape), got.dtype), ((rows, 512), torch.int32))
        self.assertTrue(got.is_contiguous())
        for r in range(rows):
            n = int(valid[r])
            # each row's selected blocks; past them (and the tail's column) -1
            self.assertTrue(torch.equal(got[r, :n], blocks[r, :n]), r)
            self.assertTrue(bool((got[r, n + 1 :] == -1).all()), r)
        self.assertEqual(tuple(kw["q"].shape), (rows, 24, 256))
        self.assertEqual(tuple(kw["new_k"].shape), (rows, 2, 256))
        router = backend.token_to_kv_pool.layer_router
        self.assertIs(kw["key_cache"], router.k)
        self.assertEqual(router.reads, [7])
        md = backend._resolve_metadata(None)
        self.assertIs(kw["row_requests"], md.row_req_pool_indices)
        self.assertIs(kw["sequence_lengths"], md.sequence_lengths)
        self.assertEqual(kw["query_positions"], "qpos")
        self.assertEqual((kw["slot_map"], kw["scale"]), ("req_to_token", 0.0625))
        self.assertEqual(kw["locations"].tolist(), list(range(rows)))
        self.assertEqual(self.gated, [1.0])  # SGLang's gate, then o_proj
        self.assertTrue(torch.equal(out, torch.full((rows, 24 * 256), 2.0)))

    def test_draft_extend_declines_launching_nothing(self):
        rows = 288
        for kw in (
            dict(graph=False),  # eager metadata
            dict(profile=None),
            dict(profile=(256, 1024)),
            dict(tokens=torch.zeros(rows, 512, dtype=torch.int32)),  # block indices
            dict(requests=rows),  # one row a request
            dict(row_requests=torch.zeros(rows // 4)),
            dict(lengths=torch.zeros(rows - 1)),
            dict(positions=None),
            dict(loc=torch.arange(2 * rows)[::2]),
            dict(q_dtype=torch.float16),
        ):
            impl, _, ex, calls, _ = self.draft_extend(rows, **kw)
            self.assertEqual((impl, calls), (["stock"], ["stock"]), kw)
            self.assertEqual(ex.qsa_union.seen, [], kw)


def _core_backend(
    rows=4,
    *,
    graph=True,
    lengths=None,
    positions="qpos",
    requests=True,
    dtype=torch.bfloat16,
    **kw,
):
    """The verify's QSA backend as the tcgen05 core's gates read it."""
    md = SimpleNamespace(
        is_cuda_graph=graph,
        sequence_lengths=torch.arange(rows) + 100 if lengths is None else lengths,
        indexer_metadata=SimpleNamespace(
            decode_logical_positions=None
            if positions is None
            else torch.arange(rows) + 99
        ),
        row_req_pool_indices=torch.zeros(rows, dtype=torch.int32) if requests else None,
    )
    backend = SimpleNamespace(
        token_to_kv_pool=SimpleNamespace(layer_router=_KvRouter(dtype)),
        req_to_token_pool=SimpleNamespace(req_to_token="req_to_token"),
        _resolve_metadata=lambda fb: md,
        _uses_block_indices=lambda indices: indices.shape[1] == 512,
        _fused_kv_pool_eligible=True,
        compress_ratio=4,
        md=md,
    )
    backend.__dict__.update(kw)
    return backend


class TestQsaCore(unittest.TestCase):
    """Slot qsa_pool with SGLANG_NVIDIA_MEGAKERNEL_QWEN3_8_ATTN_CORE_V1=1: a one-request
    verify in a CUDA graph on a BF16 pool runs the tcgen05 core under the source's gates,
    before the split-K pool kernels; anything else keeps the pool kernels (or SGLang's
    attention)."""

    def setUp(self):
        elementwise = types.ModuleType("sglang.kernels.ops.elementwise.elementwise")
        self.gated = []
        elementwise.fused_sigmoid_mul = lambda x, gate, inplace: (
            self.gated.append(float(x[0, 0])) or x + 1
        )
        patch = mock.patch.dict(sys.modules, {elementwise.__name__: elementwise})
        patch.start()
        self.addCleanup(patch.stop)

    def run_core(
        self,
        rows=4,
        *,
        core=True,
        topk=1,
        draft=4,
        mode=_Mode.TARGET_VERIFY,
        loc=None,
        blocks=512,
        k=None,
        v=None,
        backend=None,
        gate_v2=False,
        merge_gate=False,
    ):
        from sgl_nvidia_mega.qwen3_8.runtime import attention

        calls = []
        qkv = torch.zeros(rows, 13312, dtype=torch.bfloat16)
        self.gate = torch.zeros(rows, 24 * 256)
        layer = SimpleNamespace(
            is_qsa=True,
            alt_stream=None,
            attn=_QsaAttn(calls),
            _prepare_qkv_gate=lambda positions, hidden_states, forward_batch: (
                torch.zeros(rows, 24 * 256, dtype=torch.bfloat16),
                torch.zeros(rows, 2 * 256, dtype=torch.bfloat16) if k is None else k,
                qkv[:, -512:] if v is None else v,  # the QKV's strided V view
                self.gate,
            ),
            _compute_qsa_topk_indices=lambda x, positions, fb: torch.zeros(
                rows, blocks, dtype=torch.int64
            ),
            o_proj=lambda x: (x, None),
        )
        ex = SimpleNamespace(
            qsa_pool_ready=True,
            qsa_pool=_Kernels(3.0),
            qsa_core=_Kernels(9.0) if core else None,
            qsa_union=None,
            qsa_indexer=None,
            qsa_draft=None,
            qsa_prefill=None,
            smallm_gemm=None,
        )
        st = _St(mode, rows)
        st.fb.spec_info = SimpleNamespace(draft_token_num=draft, topk=topk)
        st.fb.req_pool_indices = torch.zeros(rows // 4)
        st.fb.out_cache_loc = torch.arange(rows) + 1 if loc is None else loc
        backend = backend or _core_backend(rows)
        with (
            mock.patch.object(attention, "sparse_backend", lambda: backend),
            mock.patch.object(attention, "ATTN_GATE_V2", gate_v2),
            mock.patch.object(attention, "POOL_MERGE_GATE", merge_gate),
        ):
            out = attention.step(ex, layer, torch.zeros(rows, 8), st)
        impl = [i for op, i in st.ran_ops if op == "qsa"]
        return impl, out, ex, calls, backend, st

    def test_one_request_verify_runs_the_core(self):
        impl, out, ex, calls, backend, st = self.run_core()
        self.assertEqual((impl, calls, ex.qsa_pool.seen), (["tcgen05_v1"], [], []))
        (kw,) = ex.qsa_core.seen
        self.assertEqual(tuple(kw["q"].shape), (4, 24, 256))
        for name in ("new_k", "new_v"):
            self.assertEqual(tuple(kw[name].shape), (4, 2, 256))
            self.assertEqual(kw[name].stride()[1:], (256, 1))
        self.assertEqual(kw["new_v"].stride(0), 13312)  # the strided view, no copy
        router = backend.token_to_kv_pool.layer_router
        self.assertIs(kw["key_cache"], router.k)
        self.assertIs(kw["value_cache"], router.v)
        self.assertEqual(router.reads, [7])
        self.assertIs(kw["locations"], st.fb.out_cache_loc)
        self.assertIs(kw["row_requests"], backend.md.row_req_pool_indices)
        self.assertEqual(
            (kw["indices"].dtype, tuple(kw["indices"].shape)), (torch.int32, (4, 512))
        )
        self.assertIs(kw["sequence_lengths"], backend.md.sequence_lengths)
        self.assertIs(
            kw["query_positions"], backend.md.indexer_metadata.decode_logical_positions
        )
        self.assertEqual((kw["slot_map"], kw["scale"]), ("req_to_token", 0.0625))
        self.assertIsNone(kw["gate"])
        # SGLang's gate on the core's output, then o_proj on the gated rows
        self.assertEqual(self.gated, [9.0])
        self.assertTrue(torch.equal(out, torch.full((4, 24 * 256), 10.0)))
        # no row requests in the metadata: the source falls back to the batch's, one a
        # request, which a 4-row verify does not match: nothing launched
        impl, _, ex, _, _, _ = self.run_core(backend=_core_backend(requests=False))
        self.assertEqual((impl, ex.qsa_core.seen), (["stock"], []))

    def test_gate_in_the_merge(self):
        """SGLANG_NVIDIA_MEGAKERNEL_QWEN3_8_ATTN_GATE_V2, or
        SGLANG_ENABLE_QSA_POOL_MERGE_GATE, hands the gate to the core: its epilogue merge
        applies it and nothing gates after it."""
        for kw in (
            dict(gate_v2=True),
            dict(merge_gate=True),
            dict(gate_v2=True, merge_gate=True),
        ):
            self.gated.clear()
            impl, out, ex, _, _, _ = self.run_core(**kw)
            self.assertEqual(impl, ["tcgen05_gate_v2"], kw)
            self.assertIs(ex.qsa_core.seen[0]["gate"], self.gate)
            self.assertEqual(self.gated, [], kw)
            self.assertTrue(torch.equal(out, torch.full((4, 24 * 256), 9.0)))
        # the pool kernels keep their own merge gate
        impl, _, ex, _, _, _ = self.run_core(rows=8, gate_v2=True)
        self.assertEqual((impl, len(ex.qsa_pool.seen)), (["pool_splitk"], 1))

    def test_other_steps_keep_the_pool_kernels(self):
        strided_k = torch.zeros(4, 2, 512, dtype=torch.bfloat16)[:, :, :256]
        for name, kw, want in (
            ("off", dict(core=False), "pool_splitk"),
            ("two requests", dict(rows=8), "pool_splitk"),
            ("topk 2", dict(topk=2), "pool_splitk"),
            ("eager metadata", dict(backend=_core_backend(graph=False)), "pool_splitk"),
            (
                "lengths",
                dict(backend=_core_backend(lengths=torch.zeros(3))),
                "pool_splitk",
            ),
            (
                "not the plain pool",
                dict(backend=_core_backend(_fused_kv_pool_eligible=False)),
                "pool_splitk",
            ),
            ("ratio 8", dict(backend=_core_backend(compress_ratio=8)), "pool_splitk"),
            ("strided loc", dict(loc=torch.arange(8)[::2]), "pool_splitk"),
            ("strided k", dict(k=strided_k.reshape(4, 2, 256)), "pool_splitk"),
            ("three draft tokens", dict(draft=3), "stock"),
            ("no positions", dict(backend=_core_backend(positions=None)), "stock"),
            ("token indices", dict(blocks=2051), "stock"),
            (
                "FP8 pool",
                dict(backend=_core_backend(dtype=torch.float8_e4m3fn)),
                "stock",
            ),
        ):
            impl, _, ex, calls, _, _ = self.run_core(**kw)
            self.assertEqual(impl, [want], name)
            if ex.qsa_core is not None:
                self.assertEqual(ex.qsa_core.seen, [], name)

    def test_core_kernels_launch(self):
        from sgl_nvidia_mega.qwen3_8.kernels.ops.attention import qsa_tcgen05_v1
        from sgl_nvidia_mega.qwen3_8.runtime import attention

        names = (
            "q",
            "key_cache",
            "value_cache",
            "new_k",
            "new_v",
            "locations",
            "slot_map",
            "row_requests",
            "indices",
            "sequence_lengths",
            "query_positions",
            "scale",
        )
        with mock.patch.object(qsa_tcgen05_v1, "run", lambda *a, **kw: (a, kw)):
            args, kw = attention.CoreKernels().run(**{n: n for n in names})
        self.assertEqual(args, names)
        self.assertEqual(kw, dict(width=4, store_new_kv=True))
        from sgl_nvidia_mega.qwen3_8.kernels.ops.attention import qsa_gate_epilogue_v2

        for fix in (False, True):  # the QSA PDL fix: the merge waits for the partials
            with (
                mock.patch.object(
                    qsa_gate_epilogue_v2, "run_gated", lambda *a, **kw: (a, kw)
                ),
                mock.patch.object(attention, "QSA_PDL_FIX", fix),
            ):
                args, kw = attention.CoreKernels().run(
                    **{n: n for n in names}, gate="gate"
                )
            self.assertEqual(args, names + ("gate",))
            self.assertEqual(kw, dict(width=4, store_new_kv=True, wait_primary=fix))

    def test_build(self):
        from sgl_nvidia_mega.qwen3_8.runtime import attention

        binding = types.ModuleType("sglang.srt.kernel_plugins.nvidia.binding")
        agreed = []
        binding.megakernel_agree = lambda reason, what: agreed.append((reason, what))
        with mock.patch.dict(sys.modules, {binding.__name__: binding}):
            with mock.patch.object(attention, "ATTN_CORE_V1", False):
                self.assertIsNone(runtime._build_qsa_core())
            self.assertEqual(agreed, [])
            with mock.patch.object(attention, "ATTN_CORE_V1", True):
                with mock.patch.object(attention, "core_build", return_value=""):
                    self.assertIsInstance(
                        runtime._build_qsa_core(), attention.CoreKernels
                    )
                with mock.patch.object(
                    attention, "core_build", return_value="no SM10x"
                ):
                    self.assertIsNone(runtime._build_qsa_core())
        self.assertEqual(
            agreed,
            [
                ("", "Qwen3.8 slot qsa_pool tcgen05 core"),
                ("no SM10x", "Qwen3.8 slot qsa_pool tcgen05 core"),
            ],
        )
        with mock.patch.object(torch.cuda, "is_available", lambda: False):
            self.assertIn("SM10x", attention.core_build())


def _expanded(rows):
    """Expanded token indices [rows, 2051] as SGLang's Triton expansion lays them out:
    each row's selected blocks' tokens, its uncompressed tail, then -1; the blocks and
    each row's count."""
    gen = torch.Generator().manual_seed(rows)
    valid = torch.randint(1, 513, (rows,), generator=gen)
    blocks = torch.full((rows, 512), -1, dtype=torch.int32)
    tokens = torch.full((rows, 2051), -1, dtype=torch.int32)
    for r in range(rows):
        n = int(valid[r])
        blocks[r, :n] = torch.randperm(4096, generator=gen)[:n].int()
        tokens[r, : 4 * n] = (blocks[r, :n, None] * 4 + torch.arange(4)).flatten()
        tokens[r, 4 * n : 4 * n + 2] = 40000 + r
    return tokens, blocks, valid


def _draft_extend_backend(
    rows,
    *,
    graph=True,
    profile=(512, 2048),
    row_requests=None,
    lengths=None,
    positions="qpos",
):
    md = SimpleNamespace(
        is_cuda_graph=graph,
        indexer_metadata=SimpleNamespace(decode_logical_positions=positions),
        row_req_pool_indices=(
            torch.arange(rows) // 4 if row_requests is None else row_requests
        ),
        sequence_lengths=torch.arange(rows) + 100 if lengths is None else lengths,
    )
    backend = _qsa_backend(torch.float8_e4m3fn, positions)
    backend._resolve_metadata = lambda fb: md
    if profile is not None:
        profile = SimpleNamespace(block_topk=profile[0], budget=profile[1])
    backend.qsa_profile = profile
    return backend


class _State:  # a UcacheState double that records the calls
    state_bytes_per_slot = 0
    state_bytes_per_request_row = 123
    keeps_draft_ssm = tracks_decode_state = backs_up_state = False
    tokens_per_request = 4
    buffers = "rings"

    def __init__(self):
        self.calls = []

    def reset_state_slots(self, slots, request_rows):
        self.calls.append(("reset", slots, request_rows))

    def commit(self, io):
        self.calls.append(("commit", io))
        return True

    def prepare(self, view):
        self.calls.append(("prepare", view))


class TestOwnUcache(unittest.TestCase):
    def _executor(self):
        ex = Qwen3_8UcacheExecutor(
            _model(), widths=tuple(range(4, 1025, 4)), draft=False, slots=frozenset()
        )
        ex.ucache_state = _State()
        return ex

    def test_binding_protocols(self):
        try:
            from sglang.kernels.plugin_kernels.nvidia.spec import (
                SupportsRecurrentCommit,
                SupportsStateSlotReset,
            )
        except ImportError:
            self.skipTest("binding v2 (SupportsStateSlotReset) not in this SGLang")
        ex = self._executor()
        self.assertIsInstance(ex, SupportsStateSlotReset)
        self.assertIsInstance(ex, SupportsRecurrentCommit)
        self.assertNotIsInstance(_executor(), SupportsStateSlotReset)
        self.assertNotIsInstance(_executor(), SupportsRecurrentCommit)

    def test_delegates_to_state(self):
        ex = self._executor()
        self.assertEqual(
            (ex.state_bytes_per_slot, ex.state_bytes_per_request_row), (0, 123)
        )
        self.assertFalse(ex.keeps_draft_ssm or ex.tracks_decode_state)
        self.assertFalse(ex.backs_up_state)
        ex.reset_state_slots("slots", "rows")
        self.assertTrue(ex.commit_recurrent_state("io"))
        self.assertEqual(
            ex.ucache_state.calls, [("reset", "slots", "rows"), ("commit", "io")]
        )

    def test_prepare_builds_the_verify_once(self):
        ex = self._executor()
        built = []
        ex._build_ucache_verify = lambda mamba, buffers, tokens: (
            built.append((mamba, buffers, tokens)),
            setattr(ex, "gdn_ucache", "verify"),
        )
        view = SimpleNamespace(mamba="pools")
        runtime.prepare(ex, view)
        runtime.prepare(ex, view)  # a recapture repeats prepare
        self.assertEqual(built, [("pools", "rings", 4)])
        self.assertEqual(ex.ucache_state.calls, [("prepare", view), ("prepare", view)])
        runtime.prepare(_executor(), view)  # no own u-cache: nothing to prepare

    def test_config_reasons(self):
        cfg = SimpleNamespace(
            mamba_ssm_dtype="float16",
            disable_radix_cache=False,
            mamba_radix_cache_strategy="extra_buffer_lazy",
            speculative_algorithm="NEXTN",
            speculative_eagle_topk=1,
            speculative_num_draft_tokens=4,
            max_running_requests=8,
        )
        env = runtime.UCACHE_ENV
        old = runtime.os.environ.get(env)
        runtime.os.environ[env] = "1"
        try:
            self.assertEqual(runtime.config_reasons(cfg, None), [])
            cfg.mamba_ssm_dtype = "bfloat16"
            cfg.max_running_requests = 512  # 512 x 4 rows > MAX_ROWS
            self.assertEqual(len(runtime.config_reasons(cfg, None)), 2)
        finally:
            if old is None:
                del runtime.os.environ[env]
            else:
                runtime.os.environ[env] = old


class TestSpecs(unittest.TestCase):
    def test_target_and_draft(self):
        from sgl_nvidia_mega.qwen3_8 import spec

        self.assertIn("forward", spec.SPEC.name)
        self.assertIn("draft", spec.DRAFT_SPEC.name)
        self.assertIsNone(getattr(spec.DRAFT_SPEC, "prepare", None))
        if "prepare" in spec.MegakernelSpec.__struct_fields__:
            self.assertIs(spec.SPEC.prepare, runtime.prepare)
        if "prefill_graph" in spec.MegakernelSpec.__struct_fields__:
            # breakable prefill graphs may capture both prefill bodies
            self.assertTrue(spec.SPEC.prefill_graph)
            self.assertTrue(spec.DRAFT_SPEC.prefill_graph)


class _LogitsOutput:  # the fake LogitsProcessorOutput
    hidden_states = None


def _target_lm():
    """A target model whose layer module's forward is a spy."""
    lm = _model()
    lm.is_mrope_enabled = False
    lm.language_model_only = True
    lm.lm_head = "head"
    lm.logits_processor = lambda ids, hidden, head, fb, aux: _LogitsOutput()
    lm.calls = []

    def forward(input_ids, positions, forward_batch, input_embeds=None):
        lm.calls.append((input_ids, positions, forward_batch, input_embeds))
        lm.model.last_hc_hidden_states = "hc"
        return "hidden"

    lm.model.forward = forward
    return lm


class Qwen4ExpForCausalLMMTP(SimpleNamespace):  # build tells the draft by its name
    pass


def _draft_lm():
    """An MTP draft model whose layer module's forward (Qwen4ExpModel's) is a spy."""
    lm = Qwen4ExpForCausalLMMTP(model=_model(layers=(_Attention,)).model)
    lm.lm_head = "head"
    lm.logits_processor = lambda ids, hidden, head, fb: _LogitsOutput()
    lm._prepare_input_embeds = lambda ids, fb, embeds: "embeds"
    lm._mtp_input_fusion = lambda embeds, hidden: ("streams", embeds, hidden)
    lm._set_hc_logits_hidden_states = lambda out, hc, fb: setattr(
        out, "hidden_states", hc
    )
    lm.calls = []

    def forward(input_ids, positions, forward_batch, inputs_embeds=None):
        lm.calls.append((input_ids, positions, forward_batch, inputs_embeds))
        return "hidden", "hc"

    lm.model.forward = forward
    return lm


class TestPrefillBody(unittest.TestCase):
    """SupportsPrefillBody: the layer loop of a prefill as the layer module's forward,
    which breakable prefill graphs capture."""

    def setUp(self):
        _STATE.update(breakable=False, cp=False)

    def body(self, ex, mode, rows, *args, ids=True, **kw):
        fb = _batch(mode, rows)
        step = ex.plan_body(fb.input_ids if ids else None, fb, *args, **kw)
        return None if step is None else (step.mode, step.rows)

    def test_off_without_prefill_graphs(self):
        for ex in (_executor(), _executor(draft=True)):
            _STATE["breakable"] = False
            self.assertIsNone(self.body(ex, _Mode.EXTEND, 64))
            self.assertIsNone(self.body(ex, _Mode.EXTEND, 64, inputs_embeds="streams"))
            _STATE["breakable"] = True
            self.assertIsNone(ex.plan(torch.zeros(64), _batch(_Mode.EXTEND, 64)))

    def test_covers_what_plan_covers_as_prefill(self):
        ex = _executor()
        ex.prefill_graph = True
        for breakable in (False, True):  # captured or eager alike
            _STATE["breakable"] = breakable
            self.assertEqual(self.body(ex, _Mode.EXTEND, 8192), ("prefill", 8192))
            self.assertEqual(self.body(ex, _Mode.MIXED, 37), ("prefill", 37))
            self.assertEqual(
                self.body(ex, _Mode.EXTEND, 64, input_embeds="slot"), ("prefill", 64)
            )
            self.assertEqual(
                self.body(ex, _Mode.EXTEND, 64, ids=False), ("prefill", 64)
            )
            # the outer step around a replay
            step = ex.plan(torch.zeros(64), _batch(_Mode.EXTEND, 64))
            self.assertEqual((step.mode, step.rows), ("prefill", 64))
        for mode in (
            _Mode.TARGET_VERIFY,
            _Mode.DECODE,
            _Mode.DRAFT_EXTEND_V2,
            _Mode.IDLE,
            _Mode.PREBUILT,
        ):
            self.assertIsNone(self.body(ex, mode, 8), mode)
        self.assertIsNone(self.body(ex, _Mode.EXTEND, 8, pp_proxy_tensors=1))
        self.assertIsNone(self.body(ex, _Mode.EXTEND, 8, input_deepstack_embeds=1))
        self.assertIsNone(self.body(ex, _Mode.EXTEND, 8, unknown=1))
        self.assertIsNone(self.body(ex, _Mode.EXTEND, 8, "embeds", 1))  # positional pp
        _STATE["cp"] = True
        self.assertIsNone(self.body(ex, _Mode.EXTEND, 8))

    def test_draft_body_takes_the_mtp_streams(self):
        ex = _executor(draft=True)
        ex.prefill_graph = True
        for breakable in (False, True):
            _STATE["breakable"] = breakable
            # the graph runner's capture names Qwen4ExpModel.forward's parameter
            self.assertEqual(
                self.body(ex, _Mode.EXTEND, 8192, inputs_embeds="slot"),
                ("prefill", 8192),
            )
            # the draft's outer step passes the streams positionally
            self.assertEqual(
                self.body(ex, _Mode.EXTEND, 64, "streams"), ("prefill", 64)
            )
            self.assertEqual(self.body(ex, _Mode.MIXED, 37), ("prefill", 37))
            # the outer step around a replay
            step = ex.plan(torch.zeros(64), _batch(_Mode.EXTEND, 64))
            self.assertEqual((step.mode, step.rows), ("prefill", 64))
        # draft extend and decode stay in the draft's own graphs (plan)
        for mode in (
            _Mode.DRAFT_EXTEND_V2,
            _Mode.DECODE,
            _Mode.TARGET_VERIFY,
            _Mode.IDLE,
            _Mode.PREBUILT,
        ):
            self.assertIsNone(self.body(ex, mode, 8, inputs_embeds="slot"), mode)
        self.assertIsNone(self.body(ex, _Mode.EXTEND, 8, "streams", 1))  # positional pp
        self.assertIsNone(self.body(ex, _Mode.EXTEND, 8, inputs_embeds="s", unknown=1))
        _STATE["cp"] = True
        self.assertIsNone(self.body(ex, _Mode.EXTEND, 8, inputs_embeds="slot"))
        # the target's layer module takes no inputs_embeds: its own forward runs
        _STATE["cp"] = False
        target = _executor()
        target.prefill_graph = True
        self.assertIsNone(self.body(target, _Mode.EXTEND, 8, inputs_embeds="slot"))

    def test_draft_forward_body_returns_the_streams(self):
        from sglang.kernels.plugin_kernels.nvidia.spec import Step

        ex = _executor(draft=True)
        ex.prefill_graph = True
        seen, logged = [], []
        ex._language_model = lambda st, ids, embeds: (
            seen.append((st.graph, st.positions, ids, embeds)) or ("h", "hc")
        )
        ex.engagement.begin = lambda: logged.append("begin")
        ex.engagement.end = lambda mode, rows: logged.append(("end", mode, rows))
        fb = _batch(_Mode.EXTEND, 16)
        step = Step(mode="prefill", rows=16, view="pview")
        for breakable in (False, True):
            _STATE["breakable"] = breakable
            out = ex.forward_body(step, None, None, "pos", fb, inputs_embeds="slot")
            self.assertEqual(out, ("h", "hc"))  # Qwen4ExpModel.forward's pair
            out = ex.forward_body(step, None, fb.input_ids, "pos", fb, "streams")
            self.assertEqual(out, ("h", "hc"))
        self.assertEqual(
            seen,
            [
                (graph, "pos", fb.input_ids, embeds)
                for graph in (False, True)
                for embeds in ("slot", "streams")
            ],
        )
        # the draft's layer module has no HC side channel
        self.assertFalse(hasattr(ex.lm.model, "last_hc_hidden_states"))
        # eager, the outer prefill step records the body; a capture records itself
        self.assertEqual(logged, ["begin", ("end", "prefill", 16)] * 2)

    def test_draft_capture_logs_its_engagement(self):
        from unittest import mock

        from sglang.kernels.plugin_kernels.nvidia.spec import Step

        ex = _executor(draft=True)
        ex.prefill_graph = True

        def layers(st, ids, embeds):
            st.ran("qsa", "break")
            st.ran("moe", "extend_finalize")
            return "h", "hc"

        ex._language_model = layers
        _STATE["breakable"] = True
        fb = _batch(_Mode.EXTEND, 16384)
        with (
            mock.patch.object(torch.cuda, "is_available", return_value=True),
            mock.patch.object(
                torch.cuda, "is_current_stream_capturing", return_value=True
            ),
            self.assertLogs(
                "sgl_nvidia_mega.qwen3_8.runtime.engagement", logging.INFO
            ) as logs,
        ):
            step = Step(mode="prefill", rows=16384)
            ex.forward_body(step, None, None, "pos", fb, inputs_embeds="slot")
        (line,) = logs.output
        self.assertIn(
            "engagement: draft prefill rows=16384 graph: moe=extend_finalize; qsa=break",
            line,
        )

    def run_draft(self, ex, mode=_Mode.EXTEND, breakable=False):
        from unittest import mock

        from sglang.kernels.plugin_kernels.nvidia.spec import Step

        logits = types.ModuleType("sglang.srt.layers.logits_processor")
        logits.LogitsProcessorOutput = _LogitsOutput
        eplb = types.ModuleType("sglang.srt.eplb.expert_distribution")
        eplb.get_global_expert_distribution_recorder = lambda: SimpleNamespace(
            disable_this_region=contextlib.nullcontext
        )
        _STATE["breakable"] = breakable
        fb = SimpleNamespace(
            forward_mode=mode,
            input_ids=torch.zeros(8),
            spec_info=SimpleNamespace(hidden_states="target"),
        )
        name = "prefill" if mode is _Mode.EXTEND else "draft_extend"
        with mock.patch.dict(sys.modules, {m.__name__: m for m in (logits, eplb)}):
            out = ex._forward(Step(mode=name, rows=8), fb.input_ids, "pos", fb)
        return out, fb

    def test_draft_outer_prefill_calls_the_layer_module(self):
        lm = _draft_lm()
        ex = Qwen3_8Executor(lm, widths=(4,), draft=True, slots=frozenset())
        ex.prefill_graph = True
        ex._language_model = lambda *a: self.fail("the body runs the layers")
        out, fb = self.run_draft(ex)
        # Qwen4ExpForCausalLMMTP.forward's call: the MTP streams as inputs_embeds
        self.assertEqual(
            lm.calls, [(fb.input_ids, "pos", fb, ("streams", "embeds", "target"))]
        )
        self.assertEqual(out.hidden_states, "hc")
        # around a replay of the captured body
        out, _ = self.run_draft(ex, breakable=True)
        self.assertEqual((len(lm.calls), out.hidden_states), (2, "hc"))
        recorded = [item for ops in ex.engagement._seen.values() for item in ops]
        self.assertIn(("layers", "prefill_graph"), recorded)

    def test_draft_runs_the_layers_outside_the_prefill_body(self):
        lm = _draft_lm()
        ex = Qwen3_8Executor(lm, widths=(8,), draft=True, slots=frozenset())
        ex._language_model = lambda st, ids, streams: ("hidden2", "hc2")
        # no prefill graphs: the prefill's layers run here
        out, _ = self.run_draft(ex)
        self.assertEqual((lm.calls, out.hidden_states), ([], "hc2"))
        # a draft extend is never a prefill body
        ex.prefill_graph = True
        out, _ = self.run_draft(ex, mode=_Mode.DRAFT_EXTEND_V2)
        self.assertEqual((lm.calls, out.hidden_states), ([], "hc2"))

    def test_build_sets_the_body_on_target_and_draft(self):
        from unittest import mock

        context = types.ModuleType("sglang.srt.runtime_context")
        context.get_parallel = lambda: SimpleNamespace(tp_size=1)
        context.get_exec = lambda: SimpleNamespace(mamba=None)
        ctx = SimpleNamespace(widths=(4, 8), binding=None, tokens_per_request=4)
        env = {
            k: v for k, v in runtime.os.environ.items() if not k.startswith("SGLANG_")
        }
        with (
            mock.patch.dict(sys.modules, {context.__name__: context}),
            mock.patch.dict(runtime.os.environ, env, clear=True),
            mock.patch.object(runtime, "gdn_graph_tables", return_value=True),
        ):
            for on in (False, True):
                with mock.patch.object(runtime, "prefill_graphs_on", return_value=on):
                    built = runtime.build(_draft_lm(), None, ctx)
                    self.assertEqual((built.draft, built.prefill_graph), (True, on))
                    if not on:
                        built = runtime.build(_model(), None, ctx)
                        self.assertEqual(
                            (built.draft, built.prefill_graph), (False, on)
                        )
            # FlashInfer's GDN tables need slot gdn_prefill on the target only: the MTP
            # draft has no GDN layer
            with mock.patch.object(runtime, "prefill_graphs_on", return_value=True):
                with self.assertRaisesRegex(RuntimeError, "gdn_prefill"):
                    runtime.build(_model(), None, ctx)

    def run_prefill(self, ex, breakable=False):
        from unittest import mock

        from sglang.kernels.plugin_kernels.nvidia.spec import Step

        logits = types.ModuleType("sglang.srt.layers.logits_processor")
        logits.LogitsProcessorOutput = _LogitsOutput
        _STATE["breakable"] = breakable
        fb = SimpleNamespace(forward_mode=_Mode.EXTEND, input_ids=torch.zeros(8))
        with mock.patch.dict(sys.modules, {logits.__name__: logits}):
            return ex._forward(Step(mode="prefill", rows=8), fb.input_ids, "pos", fb)

    def test_outer_prefill_calls_the_layer_module(self):
        lm = _target_lm()
        ex = Qwen3_8Executor(lm, widths=(4,), draft=False, slots=frozenset())
        ex.prefill_graph = True
        ex._language_model = lambda *a: self.fail("the body runs the layers")
        out = self.run_prefill(ex)
        ((ids, positions, fb, embeds),) = lm.calls
        self.assertEqual((positions, embeds), ("pos", None))
        self.assertIs(ids, fb.input_ids)
        self.assertEqual(out.hidden_states, "hc")
        # around a replay of the captured body
        out = self.run_prefill(ex, breakable=True)
        self.assertEqual((len(lm.calls), out.hidden_states), (2, "hc"))
        recorded = [item for ops in ex.engagement._seen.values() for item in ops]
        self.assertIn(("layers", "prefill_graph"), recorded)

    def test_outer_prefill_without_prefill_graphs_runs_the_layers(self):
        lm = _target_lm()
        ex = Qwen3_8Executor(lm, widths=(4,), draft=False, slots=frozenset())
        ex._language_model = lambda st, ids, embeds: ("hidden2", "hc2")
        out = self.run_prefill(ex)
        self.assertEqual(lm.calls, [])
        self.assertEqual(
            (out.hidden_states, lm.model.last_hc_hidden_states), ("hc2", "hc2")
        )

    def test_forward_body_sets_the_streams(self):
        from sglang.kernels.plugin_kernels.nvidia.spec import Step

        ex = _executor()
        ex.prefill_graph = True
        seen, logged = [], []
        ex._language_model = lambda st, ids, embeds: (
            seen.append((st.graph, st.positions, st.view, ids, embeds)) or ("h", "hc")
        )
        ex.engagement.begin = lambda: logged.append("begin")
        ex.engagement.end = lambda mode, rows: logged.append(("end", mode, rows))
        fb = _batch(_Mode.EXTEND, 16)
        step = Step(mode="prefill", rows=16, view="pview")
        for breakable in (False, True):
            _STATE["breakable"] = breakable
            ex.lm.model.last_hc_hidden_states = "stale"
            out = ex.forward_body(step, None, None, "pos", fb, input_embeds="slot")
            self.assertEqual((out, ex.lm.model.last_hc_hidden_states), ("h", "hc"))
        self.assertEqual(
            seen,
            [
                (False, "pos", "pview", fb.input_ids, "slot"),
                (True, "pos", "pview", fb.input_ids, "slot"),
            ],
        )
        # eager, the outer prefill step records the body; a capture records itself
        self.assertEqual(logged, ["begin", ("end", "prefill", 16)])


class _PleLayer:
    def __init__(self, ple):
        self.ple = ple
        self.attn_hyper_connection = SimpleNamespace(
            hc_norm="norm", block_inject_weight=SimpleNamespace(weight="inject")
        )

    def hc_norms(self, normalized_input, next_norm):
        return contextlib.nullcontext(SimpleNamespace(output="normed"))


class TestLayerLoopInGraph(unittest.TestCase):
    """Inside a breakable prefill graph the PLE prepare, prefetch, add and commit go
    through prefill_graph's breaks; eagerly through ple's step functions."""

    def setUp(self):
        from unittest import mock

        _STATE.update(breakable=False, cp=False)
        recorder = SimpleNamespace(
            with_current_layer=lambda i: contextlib.nullcontext()
        )
        eplb = types.ModuleType("sglang.srt.eplb.expert_distribution")
        eplb.get_global_expert_distribution_recorder = lambda: recorder
        pkg = types.ModuleType("sglang.srt.layers.layer_boundary.residual")
        pkg.batch = SimpleNamespace(
            start=lambda fb: None,
            final_norm=lambda hidden, fb, mix: hidden,
            clear=lambda fb: None,
        )
        patcher = mock.patch.dict(sys.modules, {eplb.__name__: eplb, pkg.__name__: pkg})
        patcher.start()
        self.addCleanup(patcher.stop)
        from sgl_nvidia_mega.qwen3_8.runtime import ple
        from sgl_nvidia_mega.qwen3_8.runtime import prefill_graph as pg

        self.calls = []
        spied = (
            (ple, "ple", ("prepare", "prefetch", "commit", "add")),
            (
                pg,
                "prefill_graph",
                ("ple_prepare", "ple_prefetch", "ple_commit", "ple_add"),
            ),
        )
        for module, label, names in spied:
            for name in names:
                patcher = mock.patch.object(module, name, self.spy(f"{label}.{name}"))
                patcher.start()
                self.addCleanup(patcher.stop)

    def spy(self, name):
        def call(*args, **kwargs):
            self.calls.append(name)
            return "holder" if name.endswith("prepare") else None

        return call

    def run_loop(self, graph):
        from sglang.kernels.plugin_kernels.nvidia.spec import Step

        from sgl_nvidia_mega.qwen3_8.runtime.state import StepState

        layers = (_PleLayer("ple0"), _PleLayer("ple1"), _PleLayer(None))
        m = SimpleNamespace(
            layers=list(layers),
            start_layer=0,
            end_layer=3,
            has_ple=True,
            pp_group=SimpleNamespace(is_last_rank=True),
            hyper_connection_mixer=SimpleNamespace(hc_norm="final"),
        )
        ex = Qwen3_8Executor(
            SimpleNamespace(model=m), widths=(4,), draft=False, slots=frozenset()
        )
        ex._layer = lambda i, layer, hidden, st: hidden + 1
        fb = _batch(_Mode.EXTEND, 4)
        st = StepState(
            lm=ex.lm,
            step=Step(mode="prefill", rows=4),
            forward_batch=fb,
            positions=None,
            input_ids=fb.input_ids,
            slots=frozenset(),
            binding=None,
            engagement=ex.engagement,
        )
        st.graph = graph
        hidden, hc = ex._language_model(st, fb.input_ids, torch.zeros(4, 2))
        self.assertTrue(torch.equal(hc, torch.full((4, 2), 3.0)))
        self.assertEqual(st.ple_batch, "holder")
        return ex, st

    def test_breaks_inside_a_graph(self):
        self.run_loop(graph=True)
        # layer 1's prefetch while layer 0 runs; layer 2 has no PLE
        self.assertEqual(
            self.calls,
            [
                "prefill_graph.ple_prepare",
                "prefill_graph.ple_prefetch",
                "prefill_graph.ple_commit",
            ],
        )

    def test_step_functions_eagerly(self):
        self.run_loop(graph=False)
        self.assertEqual(self.calls, ["ple.prepare", "ple.prefetch", "ple.commit"])

    def test_decomposed_add(self):
        ex, st = self.run_loop(graph=True)
        self.calls.clear()
        ex._ple_add(_PleLayer("ple0"), torch.zeros(4, 2), st)
        st.graph = False
        ex._ple_add(_PleLayer("ple0"), torch.zeros(4, 2), st)
        self.assertEqual(self.calls, ["prefill_graph.ple_add", "ple.add"])


class _LiveSt:
    def __init__(self, fb):
        self.fb = fb
        self.positions = None
        self.ran_ops = []

    def ran(self, op, impl):
        self.ran_ops.append((op, impl))


class TestBreaks(unittest.TestCase):
    """prefill_graph's breaks: SGLang's eager_on_graph functions that run the step
    functions on the live batch and write the caller's buffers."""

    def test_breaks_are_eager_on_graph_functions(self):
        import importlib
        from unittest import mock

        from sgl_nvidia_mega.qwen3_8 import runtime as package

        name = "sgl_nvidia_mega.qwen3_8.runtime.prefill_graph"
        old = vars(package).get("prefill_graph")
        self.addCleanup(
            lambda: (
                setattr(package, "prefill_graph", old)
                if old is not None
                else vars(package).pop("prefill_graph", None)
            )
        )
        bcg = "sglang.srt.model_executor.runner_backend_utils.breakable_cuda_graph"
        with mock.patch.dict(sys.modules):
            for module in [m for m in sys.modules if m.startswith(bcg) or m == name]:
                del sys.modules[module]
            pg = importlib.import_module(name)
            for fn in (
                "_qsa",
                "_gdn_core",
                "_ple_prepare",
                "_ple_prefetch",
                "_ple_add",
                "_ple_commit",
            ):
                self.assertTrue(hasattr(getattr(pg, fn), "run_eager"), fn)

    def setUp(self):
        from sgl_nvidia_mega.qwen3_8.runtime import prefill_graph

        self.pg = prefill_graph
        self.fb = SimpleNamespace(forward_mode=_Mode.EXTEND)
        self.live = _LiveSt(self.fb)
        self.ex = SimpleNamespace(ple_options="opts")

    def patched(self, **fns):
        from unittest import mock

        from sgl_nvidia_mega.qwen3_8.runtime import ple

        stack = contextlib.ExitStack()
        stack.enter_context(mock.patch.object(self.pg, "_live", lambda ex: self.live))
        for name, fn in fns.items():
            stack.enter_context(mock.patch.object(ple, name, fn))
        return stack

    def test_ple_add_on_the_live_batch(self):
        holder, seen = _Holder(), []

        def stock_ple(hidden_states, forward_batch, batch):
            seen.append(("stock", batch))
            return torch.full_like(hidden_states, 2.0)

        stock_ple.forward_idle = lambda fb: seen.append(("idle", fb))
        layer = SimpleNamespace(ple=stock_ple)
        residual, output = torch.ones(4, 2), torch.zeros(4, 2)
        # an idle batch replayed as a dummy extend adds nothing
        with self.patched():
            self.pg._ple_add(self.ex, layer, residual, holder, output)
        self.assertEqual(seen, [("idle", self.fb)])
        self.assertTrue(torch.equal(output, residual))
        holder.batch = "batch"
        with self.patched(
            served=lambda options, layer_, batch, fb, in_break: (
                seen.append(("served", options, batch, in_break)) or "gate"
            ),
            forward=lambda options, ple_, hidden, *a: torch.full_like(hidden, 3.0),
        ):
            self.pg._ple_add(self.ex, layer, residual, holder, output)
        self.assertEqual(seen[-1], ("served", "opts", "batch", True))
        self.assertTrue(torch.equal(output, torch.full((4, 2), 4.0)))
        with self.patched(served=lambda *a, **k: ""):
            self.pg._ple_add(self.ex, layer, residual, holder, output)
        self.assertEqual(seen[-1], ("stock", "batch"))
        self.assertTrue(torch.equal(output, torch.full((4, 2), 3.0)))
        self.assertEqual(self.live.ran_ops, [("ple.add", "stock")])

    def test_ple_prepare_prefetch_commit_on_the_live_batch(self):
        holder, seen = _Holder(), []

        def prepare(st, ids, options):
            seen.append(("prepare", st, ids, options))
            return "batch"

        def prefetch(layer, batch, st, options):
            seen.append(("prefetch", layer, batch, options))

        with self.patched(
            prepare=prepare,
            prefetch=prefetch,
            commit=lambda batch, st: seen.append(("commit", batch, st)),
        ):
            self.pg._ple_prepare(self.ex, "ids", holder)
            self.pg._ple_prefetch(self.ex, "layer", holder)
            self.pg._ple_commit(self.ex, holder)
        self.assertEqual(
            seen,
            [
                ("prepare", self.live, "ids", "opts"),
                ("prefetch", "layer", "batch", "opts"),
                ("commit", "batch", self.live),
            ],
        )
        self.assertEqual(holder.batch, "batch")

    def test_capture_side_records_breaks_and_allocates_outputs(self):
        from unittest import mock

        st = _LiveSt(self.fb)
        st.ple_batch = "holder"
        breaks = []
        record = lambda name, at: lambda *a: breaks.append((name, a[at]))
        with (
            mock.patch.object(self.pg, "_ple_add", record("add", -2)),
            mock.patch.object(self.pg, "_ple_prefetch", record("prefetch", -1)),
            mock.patch.object(self.pg, "_ple_commit", record("commit", -1)),
            mock.patch.object(self.pg, "_ple_prepare", record("prepare", -1)),
        ):
            out = self.pg.ple_add(self.ex, "layer", torch.ones(4, 2), st)
            self.pg.ple_prefetch(self.ex, "layer", st)
            self.pg.ple_commit(self.ex, st)
            holder = self.pg.ple_prepare(self.ex, "ids", st)
        self.assertEqual(tuple(out.shape), (4, 2))
        self.assertIsInstance(holder, _Holder)
        self.assertEqual(
            breaks,
            [
                ("add", "holder"),
                ("prefetch", "holder"),
                ("commit", "holder"),
                ("prepare", holder),
            ],
        )
        self.assertEqual(
            st.ran_ops,
            [
                ("ple.add", "break"),
                ("ple.prefetch", "break"),
                ("ple.commit", "break"),
                ("ple.prepare", "break"),
            ],
        )


def _draft_state(tail=4, width=2051):
    """SGLang's QSAMTPSharedSparseIndices of two QSA layers (3, 7), 6 request slots."""
    return SimpleNamespace(
        layer_slots={3: 0, 7: 1},
        tail_width=tail,
        trash_row=6,
        indices=torch.zeros(2, 7, width + tail, dtype=torch.int32),
        captured_len=torch.ones(2, 7, dtype=torch.int32),
    )


def _draft_backend(rows, **kw):
    """The draft step's QSA backend, with its MTP selection state and the helpers its
    fused K/V path calls (recorded)."""
    calls = []
    md = SimpleNamespace(
        indexer_metadata=SimpleNamespace(
            decode_logical_positions=kw.pop("positions", torch.arange(rows) + 40)
        ),
        sequence_lengths=torch.arange(rows) + 41,
        is_cuda_graph=kw.pop("graph", True),
        fa2_valid_counts=kw.pop("counts", torch.zeros(16, dtype=torch.int32)),
        row_req_pool_indices=kw.pop("row_requests", None),
    )
    scratch = lambda capacity, heads, dim, dtype, device: (
        calls.append(("scratch", capacity, heads, dim, dtype))
        or (torch.zeros(capacity, heads, dim, dtype=dtype),) * 2
    )
    backend = SimpleNamespace(
        _mtp_shared_sparse_indices=kw.pop("state", _draft_state()),
        should_reuse_mtp_sparse_indices=lambda fb: kw.get("reuse", True),
        _resolve_metadata=lambda fb: md,
        _fused_kv_pool_eligible=kw.pop("eligible", True),
        _uses_block_indices=lambda indices: indices.shape[1] == 512,
        token_to_kv_pool=SimpleNamespace(
            layer_router=_KvRouter(kw.pop("kv_dtype", torch.bfloat16))
        ),
        req_to_token_pool=SimpleNamespace(req_to_token="req_to_token"),
        _get_trtllm_sparse_tables=lambda bs, pages, page, device: (
            calls.append(("tables", bs, pages, page)) or ("cu", "block_tables")
        ),
        _get_fa2_scratch=scratch,
        _cuda_graph_max_tokens=16,
        _trtllm_workspace=kw.pop("workspace", "backend_workspace"),
        calls=calls,
        md=md,
    )
    return backend


class TestQsaDraft(unittest.TestCase):
    """Slot qsa_draft: the draft decode's lookup pack and the draft extend's seed
    capture; what each takes from SGLang's backend, the stock fallbacks, the stream
    order of the draft extend and the build gates. The kernels themselves run under
    Triton's interpreter against SGLang's own code (TestQsaDraftKernels)."""

    def setUp(self):
        from unittest import mock

        # imported before sys.modules is patched (its restore would unload Triton)
        from sgl_nvidia_mega.qwen3_8.qsa import (  # noqa: F401
            fused_kv_lookup,
            mtp_capture,
        )
        from sgl_nvidia_mega.qwen3_8.runtime import qsa_draft  # noqa: F401

        self.trtllm = []
        qsab = types.ModuleType("sglang.srt.layers.attention.qwen_sparse_attn_backend")
        qsab._TRTLLM_SPARSE_PAGE_SIZE = 64
        qsab._resolve_trtllm_sparse_decode = lambda: self.decode_fn
        self.decode_fn = lambda **kw: (
            self.trtllm.append(kw) or torch.full((kw["query"].shape[0], 6144), 2.0)
        )
        flashinfer = types.ModuleType("flashinfer.utils")
        flashinfer.get_trtllm_gen_multi_ctas_kv_counter_bytes = lambda b, h, sms: 8
        patch = mock.patch.dict(
            sys.modules, {m.__name__: m for m in (qsab, flashinfer)}
        )
        patch.start()
        self.addCleanup(patch.stop)
        sm100 = mock.patch("sglang.srt.utils.is_sm100_supported", lambda: True)
        sm100.start()
        self.addCleanup(sm100.stop)
        props = mock.patch.object(
            torch.cuda,
            "get_device_properties",
            lambda d: SimpleNamespace(multi_processor_count=148),
        )
        props.start()
        self.addCleanup(props.stop)

    def decode(self, rows=3, *, glue=None, q_dtype=torch.bfloat16, **kw):
        from unittest import mock

        from sgl_nvidia_mega.qwen3_8.qsa import fused_kv_lookup
        from sgl_nvidia_mega.qwen3_8.runtime import qsa_draft

        backend = _draft_backend(rows, **kw)
        glue = glue or qsa_draft.DraftGlue()
        layer = SimpleNamespace(
            layer_id=7,
            attn=SimpleNamespace(
                layer_id=7, scaling=0.0625, tp_q_head_num=24, head_dim=256
            ),
        )
        fb = SimpleNamespace(
            req_pool_indices=torch.arange(rows) + 1,
            out_cache_loc=torch.arange(rows) + 100,
        )
        packs = []
        q = torch.zeros(rows, 24 * 256, dtype=q_dtype)
        k = v = torch.zeros(rows, 2 * 256)
        with mock.patch.object(
            fused_kv_lookup, "fused_kv_lookup", lambda *a: packs.append(a)
        ):
            out = glue.decode(layer, backend, q, k, v, fb)
        return out, packs, backend, fb

    def test_decode_packs_with_the_lookup_then_trtllm(self):
        out, packs, backend, fb = self.decode(rows=3)
        self.assertEqual(tuple(out.shape), (3, 6144))
        ((kc, vc, k, v, loc, r2t, row_requests, lookup, lengths, counts, pk, pv),) = (
            packs
        )
        router = backend.token_to_kv_pool.layer_router
        self.assertEqual(router.reads, [7])
        self.assertIs(kc, router.k)
        self.assertEqual((tuple(k.shape), tuple(v.shape)), ((3, 2, 256),) * 2)
        self.assertIs(loc, fb.out_cache_loc)
        self.assertEqual(r2t, "req_to_token")
        self.assertIs(row_requests, fb.req_pool_indices)  # no row requests: the batch's
        state = backend._mtp_shared_sparse_indices
        # the decoder layer 7's slot, the batch's requests and logical positions
        self.assertEqual(lookup.indices.data_ptr(), state.indices[1].data_ptr())
        self.assertEqual(
            lookup.captured_len.data_ptr(), state.captured_len[1].data_ptr()
        )
        self.assertIs(lookup.requests, fb.req_pool_indices)
        self.assertIs(
            lookup.positions, backend.md.indexer_metadata.decode_logical_positions
        )
        self.assertEqual((lookup.shape, lookup.tail_width), ((3, 2055), 4))
        self.assertIs(lengths, backend.md.sequence_lengths)
        self.assertIs(counts, backend.md.fa2_valid_counts)
        # 2055 columns in 33 pages of 64; the graph's scratch capacity, BF16 pool
        self.assertEqual(tuple(pk.shape), (3 * 2112, 2, 256))
        self.assertEqual(
            backend.calls,
            [("tables", 3, 33, 64), ("scratch", 16 * 2112, 2, 256, torch.bfloat16)],
        )
        (kw,) = self.trtllm
        self.assertEqual(tuple(kw["query"].shape), (3, 24, 256))
        self.assertEqual(tuple(kw["kv_cache"][0].shape), (99, 2, 64, 256))
        self.assertEqual(kw["block_tables"], "block_tables")
        self.assertIs(kw["seq_lens"], counts)
        self.assertEqual(
            (kw["max_seq_len"], kw["bmm1_scale"], kw["bmm2_scale"]), (2112, 0.0625, 1.0)
        )
        self.assertEqual(kw["workspace_buffer"], "backend_workspace")
        self.assertEqual(kw["multi_ctas_kv_counter_buffer"].numel(), 8)

    def test_decode_eager_and_own_workspace(self):
        from unittest import mock

        from sgl_nvidia_mega.qwen3_8.runtime import qsa_draft

        with mock.patch.object(qsa_draft, "WORKSPACE_BYTES", 32):
            glue = qsa_draft.DraftGlue()
            _, packs, backend, _ = self.decode(
                rows=2,
                glue=glue,
                graph=False,
                workspace=None,
                row_requests=torch.tensor([5, 6]),
            )
            self.decode(rows=2, glue=glue, workspace=None)
        counts, row_requests = packs[0][9], packs[0][6]
        self.assertEqual((tuple(counts.shape), counts.dtype), ((2,), torch.int32))
        self.assertEqual(row_requests.tolist(), [5, 6])
        self.assertEqual(backend.calls[1][1], 2 * 2112)  # eager: the step's rows only
        first, second = self.trtllm
        self.assertIs(first["workspace_buffer"], glue._workspace)
        self.assertIs(second["workspace_buffer"], glue._workspace)  # one, shared
        self.assertEqual(glue._workspace.numel(), 32)
        self.assertIs(
            first["multi_ctas_kv_counter_buffer"],
            second["multi_ctas_kv_counter_buffer"],
        )

    def test_decode_declines_launching_nothing(self):
        from sgl_nvidia_mega.qwen3_8.runtime import qsa_draft

        for kw in (
            dict(glue=qsa_draft.DraftGlue(lookup=False)),  # the source's gate is off
            dict(state=None),  # no MTP selection sharing
            dict(reuse=False),
            dict(eligible=False),
            dict(positions=None),
            dict(positions=torch.arange(4)),
            dict(graph=True, counts=None),
            dict(state=_draft_state(tail=4, width=508)),  # 512 columns: block indices
            dict(kv_dtype=torch.float32, q_dtype=torch.float16),
        ):
            out, packs, backend, _ = self.decode(rows=3, **kw)
            self.assertIsNone(out, kw)
            self.assertEqual((packs, self.trtllm, backend.calls), ([], [], []), kw)
        self.decode_fn = None  # no TRT-LLM decode
        out, packs, _, _ = self.decode(rows=3)
        self.assertIsNone(out)
        self.assertEqual(packs, [])

    def capture(
        self,
        *,
        graph=True,
        mode=_Mode.DRAFT_EXTEND_V2,
        bs=3,
        accept=True,
        topk_dtype=torch.int32,
    ):
        from unittest import mock

        from sgl_nvidia_mega.qwen3_8.qsa import mtp_capture
        from sgl_nvidia_mega.qwen3_8.runtime import qsa_draft

        state = _draft_state()
        spec = SimpleNamespace(num_front_tokens=1)
        if accept:
            spec.num_accept_tokens = torch.tensor([2, 3, 1, 9], dtype=torch.int32)
        fb = SimpleNamespace(
            forward_mode=mode,
            batch_size=bs,
            req_pool_indices=torch.tensor([3, 0, 5, 1]),
            spec_info=spec,
        )
        extend = torch.tensor([4, 4, 4, 4], dtype=torch.int32)
        backend = SimpleNamespace(
            _mtp_shared_sparse_indices=state, _capture_extend_seq_lens=lambda fb: extend
        )
        md = SimpleNamespace(
            is_cuda_graph=graph,
            sequence_lengths=torch.arange(12, dtype=torch.int32),
            get_token_to_batch_idx=lambda: torch.arange(12, dtype=torch.int32),
        )
        topk = torch.zeros(12, 2051, dtype=topk_dtype)
        launches = []
        with mock.patch.object(mtp_capture, "capture", lambda *a: launches.append(a)):
            done = qsa_draft.DraftGlue().capture(backend, topk, fb, md, 7)
        return done, launches, state, md, topk

    def test_capture_one_launch(self):
        done, (args,), state, md, topk = self.capture()
        self.assertTrue(done)
        t, extend, accept, front, req, lengths, rows, seed, captured, trash = args
        self.assertIs(t, topk)
        self.assertEqual((extend.tolist(), accept.tolist()), ([4, 4, 4], [2, 3, 1]))
        self.assertEqual((front, req.tolist(), trash), (1, [3, 0, 5], 6))
        self.assertIs(lengths, md.sequence_lengths)
        self.assertEqual(rows.tolist(), list(range(12)))
        self.assertEqual(seed.data_ptr(), state.indices[1].data_ptr())
        self.assertEqual(captured.data_ptr(), state.captured_len[1].data_ptr())
        # an extend without accept counts ends each request at its seed row
        done, (args,), *_ = self.capture(accept=False, mode=_Mode.EXTEND)
        self.assertTrue(done)
        self.assertEqual((args[2], args[3]), (None, 0))

    def test_capture_declines_to_sglang(self):
        for kw in (
            dict(graph=False, mode=_Mode.EXTEND),  # an eager plain extend: torch's own
            dict(bs=0),
            dict(bs=5),  # more requests than the batch has slots for
            dict(topk_dtype=torch.int64),  # the kernel's input check
        ):
            done, launches, *_ = self.capture(**kw)
            self.assertFalse(done, kw)
            self.assertEqual(launches, [], kw)

    def test_capture_records_and_falls_back(self):
        from sgl_nvidia_mega.qwen3_8.runtime import qsa_draft

        calls = []
        backend = SimpleNamespace(
            capture_mtp_sparse_indices=lambda topk, fb, layer_id, metadata=None: (
                calls.append((layer_id, metadata))
            )
        )
        for glue, want in (
            (SimpleNamespace(capture=lambda *a: True), "batched"),
            (SimpleNamespace(capture=lambda *a: False), "stock"),
            (None, "stock"),
        ):
            calls.clear()
            st = _St(_Mode.DRAFT_EXTEND_V2)
            ex = SimpleNamespace(qsa_draft=glue)
            qsa_draft.capture(ex, backend, "topk", st.fb, "md", 7, st)
            self.assertEqual(st.ran_ops, [("qsa.mtp_capture", want)])
            self.assertEqual(calls, [] if want == "batched" else [(7, "md")])

    def test_indices_runs_the_indexer_then_the_capture(self):
        from unittest import mock

        from sgl_nvidia_mega.qwen3_8.runtime import qsa_draft

        order = []
        backend = SimpleNamespace(
            should_capture_mtp_sparse_indices=lambda fb: (
                order.append("should_capture") or True
            )
        )
        glue = types.ModuleType("sglang.srt.layers.attention.qsa.glue")
        glue.get_qsa_indexer_metadata = lambda b, layer_id, fb: (
            order.append(("metadata", layer_id)) or "md"
        )
        glue.resolve_qsa_sparse_backend = lambda b: b
        context = types.ModuleType("sglang.srt.model_executor.forward_context")
        context.get_attn_backend = lambda: backend
        layer = SimpleNamespace(
            layer_id=7,
            indexer=lambda x, positions, fb, md: (
                order.append(("indexer", md)) or "topk"
            ),
            _compute_qsa_topk_indices=lambda x, positions, fb: (
                order.append("stock") or "stock-topk"
            ),
        )
        ex = SimpleNamespace(qsa_draft="glue")
        capture = lambda *a: order.append(("capture",) + a[1:6])
        with (
            mock.patch.dict(
                sys.modules, {glue.__name__: glue, context.__name__: context}
            ),
            mock.patch.object(qsa_draft, "capture", capture),
        ):
            st = _St(_Mode.DRAFT_EXTEND_V2)
            self.assertEqual(qsa_draft.indices(ex, layer, "x", "pos", st), "topk")
            self.assertEqual(
                order,
                [
                    ("metadata", 7),
                    ("indexer", "md"),
                    "should_capture",
                    ("capture", backend, "topk", st.fb, "md", 7),
                ],
            )
            order.clear()
            _STATE["breakable"] = True
            try:
                out = qsa_draft.indices(ex, layer, "x", "pos", st)
            finally:
                _STATE["breakable"] = False
            self.assertEqual((out, order), ("stock-topk", ["stock"]))

    def attention_step(
        self, mode, *, glue_out=None, alt_stream=None, capture_mode=True
    ):
        from unittest import mock

        from sgl_nvidia_mega.qwen3_8.runtime import attention, qsa_draft

        order = []
        rows = 4
        layer = SimpleNamespace(
            is_qsa=True,
            layer_id=7,
            alt_stream=alt_stream,
            attn=lambda q, k, v, fb, topk_indices=None: (
                order.append(("attn", topk_indices)) or torch.zeros(rows, 6144)
            ),
            _prepare_qkv_gate=lambda positions, hidden_states, forward_batch: (
                order.append("qkv") or (torch.zeros(rows, 6144), "k", "v", None)
            ),
            _compute_qsa_topk_indices=lambda x, positions, fb: (
                order.append("lookup") or "stock-indices"
            ),
            o_proj=lambda x: (x, None),
        )
        glue = SimpleNamespace(
            decode=lambda *a: order.append("lookup_pack") or glue_out
        )
        ex = SimpleNamespace(
            qsa_pool_ready=False,
            qsa_core=None,
            qsa_union=None,
            qsa_indexer=None,
            qsa_draft=glue,
            qsa_prefill=None,
            smallm_gemm=None,
        )
        runner = types.ModuleType("sglang.srt.model_executor.runner")
        runner.get_is_capture_mode = lambda: capture_mode
        elementwise = types.ModuleType("sglang.kernels.ops.elementwise.elementwise")
        elementwise.fused_sigmoid_mul = lambda x, gate, inplace: x
        st = _St(mode, rows)
        indices = mock.Mock(
            side_effect=lambda *a: order.append("indices") or _Indices(order)
        )
        with (
            mock.patch.dict(
                sys.modules, {m.__name__: m for m in (runner, elementwise)}
            ),
            mock.patch.object(
                sys.modules["sglang.srt.models.qwen4_exp"],
                "_QSA_INDEXER_OVERLAP_TOKEN_THRESHOLD",
                1024,
                create=True,
            ),
            mock.patch.object(attention, "sparse_backend", lambda: "backend"),
            mock.patch.object(qsa_draft, "indices", indices),
        ):
            attention.step(ex, layer, torch.zeros(rows, 8), st)
        return order, st.ran_ops

    def test_draft_decode_serves_or_falls_back(self):
        order, ran = self.attention_step(_Mode.DECODE, glue_out=torch.zeros(4, 6144))
        self.assertEqual(
            order, ["qkv", "lookup_pack"]
        )  # no torch lookup, no stock attn
        self.assertEqual(
            ran, [("qsa.indexer", "mtp_lookup_in_pack"), ("qsa", "fused_kv_lookup")]
        )
        order, ran = self.attention_step(_Mode.DECODE, glue_out=None)
        self.assertEqual(
            order, ["qkv", "lookup_pack", "lookup", ("attn", "stock-indices")]
        )
        self.assertEqual(ran, [("qsa.indexer", "stock"), ("qsa", "stock")])

    def test_draft_extend_keeps_sglang_stream_order(self):
        from unittest import mock

        order, ran = self.attention_step(_Mode.DRAFT_EXTEND_V2)  # no side stream
        self.assertEqual(order[:2], ["qkv", "indices"])
        self.assertEqual(ran, [("qsa.indexer", "stock"), ("qsa", "stock")])
        # under capture the indexer and the seed capture run on the layer's side stream
        alt, current = _Stream("alt", []), None
        current = _Stream("main", alt.log)
        entered = contextlib.contextmanager(
            lambda s: (
                alt.log.append(f"enter {s.name}"),
                (yield),
                alt.log.append(f"exit {s.name}"),
            )
        )
        with (
            mock.patch.object(torch.cuda, "current_stream", lambda: current),
            mock.patch.object(torch.cuda, "stream", entered),
        ):
            order, _ = self.attention_step(_Mode.DRAFT_EXTEND_V2, alt_stream=alt)
        self.assertEqual(
            alt.log,
            [
                "alt waits main",
                "enter alt",
                "exit alt",
                "main waits alt",
                "record main",
            ],
        )
        self.assertEqual(order[:2], ["qkv", "indices"])
        order, _ = self.attention_step(
            _Mode.DRAFT_EXTEND_V2, alt_stream=alt, capture_mode=False
        )
        self.assertEqual(order[:2], ["qkv", "indices"])

    def test_build_gates(self):
        from unittest import mock

        from sgl_nvidia_mega.qwen3_8.runtime import qsa_draft

        self.assertIn("qsa_draft", runtime.KNOWN_SLOTS)
        cfg = dict(tp=1, steps=3, topk=1)
        context = types.ModuleType("sglang.srt.runtime_context")
        context.get_parallel = lambda: SimpleNamespace(tp_size=cfg["tp"])
        context.get_spec = lambda: SimpleNamespace(
            speculative_num_steps=cfg["steps"], speculative_eagle_topk=cfg["topk"]
        )
        binding = types.ModuleType("sglang.srt.kernel_plugins.nvidia.binding")
        agreed = []
        binding.megakernel_agree = lambda reason, what: agreed.append((reason, what))
        model = _model((_Linear, _Attention))
        model.model.layers[1].is_qsa = True
        model.hidden_size, model.hc_count = 2560, 4
        sm10x = dict(
            is_available=lambda: True, get_device_capability=lambda *a: (10, 7)
        )
        with (
            mock.patch.dict(sys.modules, {m.__name__: m for m in (context, binding)}),
            mock.patch.multiple(torch.cuda, **sm10x),
        ):
            self.assertEqual(qsa_draft.build(model), "")
            glue = runtime._build_qsa_draft(model)
            self.assertTrue(glue.lookup_ready)
            self.assertEqual(agreed, [("", "Qwen3.8 slot qsa_draft")])
            for change, slot, lookup in (
                (dict(tp=4), "TP4", None),
                (dict(steps=4), "", "4-step"),
                (dict(topk=2), "", "topk-2"),
            ):
                cfg.update(change)
                self.assertIn(slot, qsa_draft.build(model))
                if lookup:
                    self.assertIn(lookup, qsa_draft.lookup_reason(model))
                    self.assertFalse(runtime._build_qsa_draft(model).lookup_ready)
                else:
                    self.assertIsNone(runtime._build_qsa_draft(model))
                cfg.update(tp=1, steps=3, topk=1)
            model.hc_count = 1
            self.assertIn("HC", qsa_draft.lookup_reason(model))
            model.model.layers[1].is_qsa = False
            self.assertIn("no QSA layer", qsa_draft.build(model))
        with mock.patch.object(torch.cuda, "is_available", lambda: False):
            self.assertIn("SM10x", qsa_draft.build(model))


class _Indices:  # the indexer's output, recording record_stream
    def __init__(self, log):
        self.log = log

    def record_stream(self, stream):
        stream.log.append(f"record {stream.name}")


class _Stream:
    def __init__(self, name, log):
        self.name, self.log = name, log

    def wait_stream(self, other):
        self.log.append(f"{self.name} waits {other.name}")


SEL = ("select", None, None)  # slot qsa_indexer's select, with no step or projection


class TestQsaIndexerStreams(unittest.TestCase):
    """The source's streams for a decomposed QSA step's indexer
    (``attention._qkv_and_indices``): under capture, not in a breakable graph and below
    1024 rows, the indexer runs on the layer's side stream beside the QKV / gate
    projection and the main stream joins it before the attention; otherwise everything
    runs on the main stream."""

    SIDE = [
        "alt waits main",
        "qkv",
        "enter alt",
        SEL,
        "exit alt",
        "main waits alt",
        "record main",
        "attn",
    ]

    def run_step(
        self,
        mode=_Mode.TARGET_VERIFY,
        rows=8,
        *,
        capture=True,
        breakable=False,
        alt=True,
        indexer=True,
        glue=None,
        patches=(),
        layer_kw=None,
        dtype=torch.float32,
    ):
        """One decomposed QSA step; returns (the stream log, the engagement record).
        ``patches``: (target, name, value) applied for the step."""
        from sgl_nvidia_mega.qwen3_8.runtime import attention, qsa_draft, qsa_indexer

        log = self.log = []
        main = _Stream("main", log)
        entered = contextlib.contextmanager(
            lambda s: (
                log.append(f"enter {s.name}"),
                (yield),
                log.append(f"exit {s.name}"),
            )
        )
        layer = SimpleNamespace(
            is_qsa=True,
            layer_id=7,
            alt_stream=_Stream("alt", log) if alt else None,
            attn_output_gate=True,
            attn=lambda q, k, v, fb, topk_indices=None: (
                log.append("attn") or torch.zeros(rows, 6144)
            ),
            _prepare_qkv_gate=lambda positions, hidden_states, forward_batch: (
                log.append("qkv") or (torch.zeros(rows, 6144), "k", "v", None)
            ),
            _compute_qsa_topk_indices=lambda x, positions, fb: (
                log.append("sglang") or _Indices(log)
            ),
            o_proj=lambda x: (x, None),
        )
        layer.__dict__.update(layer_kw or {})
        ex = SimpleNamespace(
            # the union never takes these widths: SGLang's attention runs
            qsa_pool_ready=False,
            qsa_pool=None,
            qsa_union="union",
            qsa_core=None,
            qsa_indexer=indexer or None,
            qsa_draft=glue,
            qsa_prefill=None,
            smallm_gemm=None,
        )
        runner = types.ModuleType("sglang.srt.model_executor.runner")
        runner.get_is_capture_mode = lambda: capture
        elementwise = types.ModuleType("sglang.kernels.ops.elementwise.elementwise")
        elementwise.fused_sigmoid_mul = lambda x, gate, inplace: x
        st = _St(mode, rows)
        st.fb.req_pool_indices = torch.zeros(rows // 4)
        select = lambda *a: log.append(("select",) + a[5:]) or _Indices(log)
        draft = lambda *a: log.append("draft_indices") or _Indices(log)
        _STATE["breakable"] = breakable
        with contextlib.ExitStack() as stack:
            stack.callback(_STATE.update, breakable=False)
            modules = {m.__name__: m for m in (runner, elementwise)}
            stack.enter_context(mock.patch.dict(sys.modules, modules))
            stack.enter_context(
                mock.patch.object(
                    sys.modules["sglang.srt.models.qwen4_exp"],
                    "_QSA_INDEXER_OVERLAP_TOKEN_THRESHOLD",
                    1024,
                    create=True,
                )
            )
            for target, name, value in (
                (attention, "sparse_backend", lambda: "backend"),
                (qsa_indexer, "select", select),
                (qsa_draft, "indices", draft),
                (torch.cuda, "current_stream", lambda: main),
                (torch.cuda, "stream", entered),
                *patches,
            ):
                stack.enter_context(mock.patch.object(target, name, value))
            attention.step(ex, layer, torch.zeros(rows, 8, dtype=dtype), st)
        return log, st.ran_ops

    def test_verify_indexer_beside_the_qkv(self):
        log, ran = self.run_step()
        self.assertEqual(log, self.SIDE)
        self.assertIn(("qsa.index_stream", "side"), ran)
        # SGLang's indexer (slot qsa_indexer off) takes the same streams
        log, ran = self.run_step(indexer=False)
        self.assertEqual(log, [s if s != SEL else "sglang" for s in self.SIDE])
        self.assertIn(("qsa.indexer", "stock"), ran)

    def test_draft_extend_and_decode_take_the_side_stream(self):
        for mode in (_Mode.DRAFT_EXTEND_V2, _Mode.DECODE):
            log, ran = self.run_step(mode)
            self.assertEqual(log, self.SIDE, mode)
            self.assertIn(("qsa.index_stream", "side"), ran)
        # a draft extend without slot qsa_indexer: SGLang's indexer and the seed capture
        log, _ = self.run_step(_Mode.DRAFT_EXTEND_V2, indexer=False, glue="glue")
        want = [s if s != SEL else "draft_indices" for s in self.SIDE]
        self.assertEqual(log, want)

    def test_main_stream_outside_the_overlap(self):
        for kw in (
            dict(capture=False),  # eager
            dict(breakable=True),  # graph breaks stay off the overlap stream
            dict(rows=1024),  # the source's bound
            dict(alt=False),
        ):
            log, ran = self.run_step(**kw)
            self.assertEqual(log, ["qkv", SEL, "attn"], kw)
            self.assertNotIn("qsa.index_stream", dict(ran), kw)

    def test_early_indexer(self):
        """The early indexer: the side stream's indexer is enqueued before the QKV."""
        from sgl_nvidia_mega.qwen3_8.runtime import attention

        fused = []
        indexer = SimpleNamespace(_use_fused_prep=lambda x: fused.append(x) or True)
        early = (attention, "QSA_EARLY_INDEXER", True)
        cuda = (torch.Tensor, "is_cuda", property(lambda t: True))
        kw = dict(layer_kw=dict(indexer=indexer), dtype=torch.bfloat16)
        log, ran = self.run_step(patches=(early, cuda), **kw)
        self.assertEqual(
            log,
            [
                "alt waits main",
                "enter alt",
                SEL,
                "exit alt",
                "qkv",
                "main waits alt",
                "record main",
                "attn",
            ],
        )
        self.assertIn(("qsa.index_stream", "side_early"), ran)
        self.assertEqual(len(fused), 1)
        for name, patches, step_kw in (
            ("off", (cuda,), kw),
            ("float32", (early, cuda), dict(kw, dtype=torch.float32)),
            (
                "unfused prep",
                (early, cuda),
                dict(
                    kw,
                    layer_kw=dict(
                        indexer=SimpleNamespace(_use_fused_prep=lambda x: False)
                    ),
                ),
            ),
            (
                "no output gate",
                (early, cuda),
                dict(kw, layer_kw=dict(indexer=indexer, attn_output_gate=False)),
            ),
        ):
            log, ran = self.run_step(patches=patches, **step_kw)
            self.assertEqual(log, self.SIDE, name)
            self.assertIn(("qsa.index_stream", "side"), ran)
        # eager: no side stream, so no early indexer
        log, _ = self.run_step(patches=(early, cuda), capture=False, **kw)
        self.assertEqual(log, ["qkv", SEL, "attn"])

    def pdl_step(
        self,
        mode=_Mode.TARGET_VERIFY,
        rows=8,
        *,
        frontier=("node",),
        prepared=True,
        backend="cutedsl",
        quant=None,
        env=True,
        dtype=torch.bfloat16,
        **kw,
    ):
        """A step with the PDL-index branch's conditions met unless changed."""
        from sgl_nvidia_mega.qwen3_8.runtime import (
            attention,
            bf16_gemm_pins,
            capture_fork,
        )

        class Unquant:
            pass

        projection = SimpleNamespace(
            quant_method=(quant or Unquant)(),
            weight=torch.empty(640, 2560, dtype=torch.bfloat16),
        )
        fused = kw.pop("fused", True)
        layer_kw = kw.pop("layer_kw", {})
        layer_kw.setdefault(
            "indexer",
            SimpleNamespace(index_qk_proj=projection, _use_fused_prep=lambda x: fused),
        )
        unquant = types.ModuleType("sglang.srt.layers.quantization.unquant")
        unquant.UnquantizedLinearMethod = Unquant
        unquant.get_bf16_gemm_backend = lambda: SimpleNamespace(
            is_cutedsl=lambda: backend == "cutedsl"
        )
        # the source's split-K table at (N, K) = (640, 2560): M 1-4 and 8
        unquant.use_bf16_splitk_gemm = lambda m, n, k: (
            (n, k) == (640, 2560) and m in (1, 2, 3, 4, 8)
        )
        self.step = SimpleNamespace(rows=rows)
        kernels = SimpleNamespace(
            prepare=lambda layer, x, positions, fb: (
                self.log.append("prepare") or (self.step if prepared else None)
            )
        )
        self.projected = []

        def linear(module, x):
            if module is not projection:
                return module(x)[0]
            self.log.append(("index_qk", x.shape[0]))
            self.projected.append(_Indices(self.log))
            return self.projected[-1]

        patches = (
            (attention, "QSA_PDL_INDEX", env),
            (torch.Tensor, "is_cuda", property(lambda t: True)),
            (bf16_gemm_pins, "linear", linear),
            (
                capture_fork,
                "projection_frontier",
                lambda s: self.log.append(f"frontier {s.name}") or frontier,
            ),
            (
                capture_fork,
                "fork_after_projection",
                lambda s, nodes: self.log.append((f"{s.name} after", nodes)),
            ),
        )
        with mock.patch.dict(sys.modules, {unquant.__name__: unquant}):
            return self.run_step(
                mode,
                rows,
                patches=patches,
                layer_kw=layer_kw,
                dtype=dtype,
                indexer=kw.pop("indexer", kernels),
                **kw,
            )

    def test_pdl_index_projects_on_the_main_stream(self):
        """The PDL-index branch: index_qk_proj before the QKV on the main stream, no event
        between them; the side stream depends on the projection's completion only and
        runs the rest of the indexer on it."""
        for mode, rows in (
            (_Mode.TARGET_VERIFY, 4),
            (_Mode.TARGET_VERIFY, 8),
            (_Mode.DRAFT_EXTEND_V2, 8),
        ):
            log, ran = self.pdl_step(mode, rows)
            (qk,) = self.projected
            self.assertEqual(
                log,
                [
                    "prepare",
                    ("index_qk", rows),
                    "frontier main",
                    "qkv",
                    "alt waits main",
                    ("alt after", ("node",)),
                    "enter alt",
                    ("select", self.step, qk),
                    "exit alt",
                    "record alt",
                    "main waits alt",
                    "record main",
                    "attn",
                ],
                (mode, rows),
            )
            self.assertIn(("qsa.index_stream", "pdl_index"), ran)
        # not capturing (a warmup): the side stream waits for the whole QKV
        log, _ = self.pdl_step(frontier=None)
        self.assertNotIn(("alt after", None), log)
        self.assertEqual(
            log[:6],
            [
                "prepare",
                ("index_qk", 8),
                "frontier main",
                "qkv",
                "alt waits main",
                "enter alt",
            ],
        )

    def test_pdl_index_declines_to_the_side_stream(self):
        prepared = ["prepare"] + self.SIDE
        for name, kw, want in (
            ("off", dict(env=False), self.SIDE),
            ("decode", dict(mode=_Mode.DECODE), self.SIDE),
            ("float32", dict(dtype=torch.float32), self.SIDE),
            ("no gate", dict(layer_kw=dict(attn_output_gate=False)), self.SIDE),
            ("plugin indexer declines", dict(prepared=False), prepared),
            ("12 rows: no split-K", dict(rows=12), prepared),
            ("20 rows", dict(rows=20), prepared),
            ("not CuTe DSL", dict(backend="cublas"), prepared),
            ("quantized", dict(quant=object), prepared),
            ("unfused prep", dict(fused=False), prepared),
        ):
            log, ran = self.pdl_step(**kw)
            self.assertEqual(log, want, name)
            self.assertIn(("qsa.index_stream", "side"), ran)
            self.assertEqual(self.projected, [], name)
        # SGLang's indexer takes no projection
        log, _ = self.pdl_step(indexer=False)
        self.assertEqual(log, [s if s != SEL else "sglang" for s in self.SIDE])
        # eager: no overlap, no projection
        log, _ = self.pdl_step(capture=False)
        self.assertEqual(log, ["qkv", SEL, "attn"])

    def test_draft_decode_glue_keeps_the_main_stream(self):
        out = torch.zeros(8, 6144)
        packs = []
        glue = SimpleNamespace(decode=lambda *a: packs.append("pack") or out)
        log, ran = self.run_step(_Mode.DECODE, glue=glue)
        self.assertEqual((log, packs), (["qkv"], ["pack"]))  # no indexer, no fork
        glue.decode = lambda *a: packs.append("pack")  # declines: SGLang's lookup
        log, ran = self.run_step(_Mode.DECODE, glue=glue)
        self.assertEqual(log, ["qkv", SEL, "attn"])
        self.assertNotIn("qsa.index_stream", dict(ran))


_SWITCHES = r"""
import importlib
import json
import os

from sgl_nvidia_mega.qwen3_8.runtime import attention

names = {
    "QSA_EARLY_INDEXER": "SGLANG_NVIDIA_MEGAKERNEL_QWEN3_8_QSA_EARLY_INDEXER",
    "QSA_PDL_INDEX": "SGLANG_NVIDIA_MEGAKERNEL_QWEN3_8_QSA_PDL_INDEX",
    "ATTN_CORE_V1": "SGLANG_NVIDIA_MEGAKERNEL_QWEN3_8_ATTN_CORE_V1",
    "ATTN_GATE_V2": "SGLANG_NVIDIA_MEGAKERNEL_QWEN3_8_ATTN_GATE_V2",
    "POOL_MERGE_GATE": "SGLANG_ENABLE_QSA_POOL_MERGE_GATE",
    "QSA_PDL_FIX": "SGLANG_NVIDIA_MEGAKERNEL_QWEN3_8_QSA_PDL_FIX",
}
out = {}
for value in (None, "1", "true", "True", "0", "yes"):
    for name in names.values():
        os.environ.pop(name, None)
        if value is not None:
            os.environ[name] = value
    importlib.reload(attention)
    out[str(value)] = {attr: getattr(attention, attr) for attr in names}
print(json.dumps(out))
"""


class TestSourceSwitches(unittest.TestCase):
    def test_read_as_the_source_reads_them(self):
        """get_bool_env_var's parsing (default off) for the self_attention switches and
        the gate's, an exact "1" for the tcgen05-core and QSA PDL-fix switches."""
        import json
        import os
        import subprocess

        env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1")
        done = subprocess.run(
            [sys.executable, "-c", _SWITCHES],
            env=env,
            capture_output=True,
            text=True,
            timeout=300,
        )
        self.assertEqual(done.returncode, 0, done.stderr[-2000:])
        got = json.loads(done.stdout.strip().splitlines()[-1])
        boolean = (
            "QSA_EARLY_INDEXER",
            "QSA_PDL_INDEX",
            "ATTN_GATE_V2",
            "POOL_MERGE_GATE",
        )
        for value, on in (
            ("None", ()),
            ("1", "all"),
            ("true", boolean),
            ("True", boolean),
            ("0", ()),
            ("yes", ()),
        ):
            want = {a: on == "all" or a in on for a in got[value]}
            self.assertEqual(got[value], want, value)


_INTERPRETED_DRAFT = r"""
import json
from types import SimpleNamespace as NS

import torch
from sglang.srt.layers.attention.qsa.metadata import QSAIndexerMetadata
from sglang.srt.layers.attention.qwen_sparse_attn_backend import (
    QSAMTPSharedSparseIndices,
    QwenSparseAttnBackend,
)

from sgl_nvidia_mega.qwen3_8.qsa import fused_kv_lookup as L
from sgl_nvidia_mega.qwen3_8.qsa import mtp_capture as C

torch.manual_seed(0)
out = {"lookup": [], "capture": []}
N = 6


def state():
    s = QSAMTPSharedSparseIndices(
        layer_ids=[3, 7], num_requests=N, token_topk=12, tail_width=4, device="cpu"
    )
    s.indices.copy_(torch.randint(-1, 44, s.indices.shape, dtype=torch.int32))
    s.captured_len.copy_(torch.randint(30, 41, s.captured_len.shape, dtype=torch.int32))
    return s


for req, pos in (([2], [41]), ([2, 5, 0], [41, 33, 0]), ([1, 2, 3, 4, 5], [40, 30, 44, 36, 31])):
    s = state()
    req, pos = torch.tensor(req), torch.tensor(pos)
    slot = s.layer_slots[7]
    lookup = L.MtpLookup(s.indices[slot], s.captured_len[slot], req, pos, 4)
    out["lookup"].append(torch.equal(lookup.materialize(), s.lookup(req, pos, 7)))

for extend, accept, req, front in (
    ([4], [2], [3], 0),
    ([4, 4, 0], [2, 4, 1], [2, 0, 5], 0),
    ([4, 4, 4, 4], [1, 0, 4, 3], [1, 2, 3, 4], 1),
    ([3, 5, 2], None, [5, 1, 2], 0),
):
    bs = len(req)
    extend = torch.tensor(extend, dtype=torch.int32)
    rows = int(extend.sum())
    topk = torch.randint(-1, 100, (rows, 10), dtype=torch.int32)
    seq = torch.randint(10, 50, (rows,), dtype=torch.int32)
    t2b = torch.arange(rows, dtype=torch.int32)
    req = torch.tensor(req)
    acc = None if accept is None else torch.tensor(accept, dtype=torch.int32)
    a, b = state(), state()
    b.indices.copy_(a.indices)
    b.captured_len.copy_(a.captured_len)
    md = QSAIndexerMetadata(
        sequence_lengths=seq, token_to_batch_idx=t2b, token_slot_table=None,
        out_cache_loc=None, token_to_kv_pool=None, compress_ratio=4, block_topk=512,
        is_cuda_graph=True,
    )
    spec = NS(extend_seq_lens_tensor=extend, num_front_tokens=front)
    if acc is not None:
        spec.num_accept_tokens = acc
    fb = NS(batch_size=bs, req_pool_indices=req, spec_info=spec)
    me = NS(
        _mtp_shared_sparse_indices=b,
        _capture_extend_seq_lens=QwenSparseAttnBackend._capture_extend_seq_lens,
    )
    QwenSparseAttnBackend._capture_mtp_sparse_indices_from_extend_lens(me, topk, fb, md, 7)
    slot = a.layer_slots[7]
    seed, captured = a.indices[slot], a.captured_len[slot]
    reason = C.reason(topk, extend, acc, req, seq, t2b, seed, captured, a.trash_row)
    C.capture(topk, extend, acc, front, req, seq, t2b, seed, captured, a.trash_row)
    keep = slice(0, N)  # the trash row's content is unspecified in both
    out["capture"].append(
        [
            reason,
            torch.equal(a.indices[:, keep], b.indices[:, keep]),
            torch.equal(a.captured_len[:, keep], b.captured_len[:, keep]),
        ]
    )
print(json.dumps(out))
"""


class TestQsaDraftKernels(unittest.TestCase):
    """Slot qsa_draft's lookup rows and seed capture, under Triton's interpreter,
    against SGLang's own lookup and torch capture (the pack kernel is SGLang's with
    the lookup's index load, which the interpreter does not run: GPU test)."""

    def test_lookup_rows_and_capture_match_sglang(self):
        import json
        import os
        import subprocess

        env = dict(os.environ, TRITON_INTERPRET="1", PYTHONDONTWRITEBYTECODE="1")
        done = subprocess.run(
            [sys.executable, "-c", _INTERPRETED_DRAFT],
            env=env,
            capture_output=True,
            text=True,
            timeout=600,
        )
        self.assertEqual(done.returncode, 0, done.stderr[-2000:])
        got = json.loads(done.stdout.strip().splitlines()[-1])
        self.assertEqual(got["lookup"], [True] * 3)
        # one request, a padded slot and an empty extend, front rows, no accept counts
        self.assertEqual(got["capture"], [["", True, True]] * 4)


_INTERPRETED_FOLD = r"""
import json
import torch
from sglang.kernels.ops.speculative.topk1 import draft_topk1_postprocess
from sgl_nvidia_mega.qwen3_8.kernels.ops.speculative import draft_step_fold as F

torch.manual_seed(0)
out = []
for bs, vocab, tie in ((1, 20000, False), (3, 17000, True), (4, 8192, False)):
    weight = torch.randn(vocab, 2560).to(torch.bfloat16)
    logits = torch.randn(bs, vocab).to(torch.bfloat16)
    if tie:  # a tie across splits: the first split's index in both
        logits[1, 100] = logits[1, 9000] = 50.0
    pos0, mrope0 = torch.randint(10, 100, (bs,)), torch.randint(10, 100, (3, bs))
    for column in (1, 2):  # a step that feeds another, then the last
        tokens_a = torch.zeros(bs, 3, dtype=torch.int64)
        tokens_b = tokens_a.clone()
        pa, ma = pos0.clone(), mrope0.clone()
        if column == 1:
            slot = F.make_slot(weight, bs)
            p_a, i_a = F.finalize_next_input(logits, pa, tokens_a, column, slot, ma)
        else:
            p_a, i_a = F.draft_topk1_postprocess(logits, pa, tokens_a, column, ma)
        # SGLang: its top-1 on the FP32 logits, the M-RoPE + 1, the next step's embedding
        pb, mb = pos0.clone(), mrope0.clone()
        p_b, i_b = draft_topk1_postprocess(logits.float(), pb, tokens_b, column)
        mb.add_(1)
        same = [
            torch.equal(i_a, i_b), torch.equal(p_a, p_b), torch.equal(pa, pb),
            torch.equal(ma, mb), torch.equal(tokens_a, tokens_b),
        ]
        if column == 1:
            same.append(torch.equal(slot.embedding, weight[i_b.flatten()]))
        out.append(same)
print(json.dumps(out))
"""


class _LogitsProcessor:  # the draft's LogitsProcessor fields the BF16 logits read
    def __init__(self, **kw):
        self.final_logit_softcapping = kw.get("softcap")
        self.do_tensor_parallel_all_gather = kw.get("gather", False)
        self.do_tensor_parallel_all_gather_dp_attn = False
        self.use_fp32_lm_head = kw.get("fp32", False)
        self.use_fp32_lm_head_output = False
        self.logit_scale = kw.get("scale")
        self.vocab_size = 6
        self.calls = []

    def _compute_lm_head(self, hidden, lm_head, bias):
        self.calls.append(("lm_head", hidden, lm_head, bias))
        return torch.ones(hidden.shape[0], 8, dtype=torch.bfloat16)

    def __call__(self, *args):
        self.calls.append("stock")
        return _LogitsOutput()


class TestDraftFold(unittest.TestCase):
    """Slot draft_fold through SGLang's draft_step_fold entry: the entry's gate, the
    finalize of each draft step, the round's draft forwards taking the folded
    embedding and returning the LM head's BF16 logits; the kernels under Triton's
    interpreter against SGLang's top-1 on the FP32 logits."""

    method = types.ModuleType("sglang.srt.layers.vocab_parallel_embedding")
    method.UnquantizedEmbeddingMethod = type("UnquantizedEmbeddingMethod", (), {})

    def model(self, **kw):
        from sgl_nvidia_mega.qwen3_8.kernels.ops.speculative import draft_step_fold

        method = self.method
        weight = kw.pop("weight", torch.zeros(20000, 2560, dtype=torch.bfloat16))
        embed = SimpleNamespace(
            weight=SimpleNamespace(
                is_cuda=True,
                dtype=weight.dtype,
                ndim=weight.ndim,
                shape=weight.shape,
                is_contiguous=lambda: True,
                device="cpu",
            ),
            tp_size=kw.pop("tp", 1),
            quant_method=method.UnquantizedEmbeddingMethod(),
            output_dtype=None,
            num_embeddings=20000,
            org_vocab_size=20000,
        )
        model = SimpleNamespace(
            model=SimpleNamespace(embed_tokens=embed),
            hc_count=kw.pop("hc", 4),
            logits_processor=_LogitsProcessor(**kw),
            lm_head=SimpleNamespace(weight=torch.zeros(1, dtype=torch.bfloat16)),
        )
        return model, method, draft_step_fold

    def test_gate_and_slot(self):
        from unittest import mock

        from sgl_nvidia_mega.qwen3_8.runtime import draft_fold

        model, method, kernels = self.model()
        ex = SimpleNamespace(lm=model)
        with mock.patch.dict(sys.modules, {method.__name__: method}):
            with mock.patch.object(
                kernels, "make_slot", lambda w, bs: SimpleNamespace(bs=bs, w=w)
            ):
                fold = draft_fold.make(ex, batch_size=4, num_steps=3)
                self.assertEqual(
                    (fold.slot.bs, fold.num_steps, fold.bf16), (4, 3, True)
                )
                self.assertIs(fold.slot.w, model.model.embed_tokens.weight)
                # logits the FP32 copy also transforms: the fold reads the FP32 ones
                for kw in (dict(softcap=30.0), dict(gather=True), dict(fp32=True)):
                    m, *_ = self.model(**kw)
                    self.assertFalse(draft_fold.make(SimpleNamespace(lm=m), 2, 3).bf16)
            for bs, steps in ((5, 3), (0, 3), (2, 4)):
                self.assertIsNone(draft_fold.make(ex, bs, steps))
            for kw in (dict(hc=1), dict(tp=4)):
                m, *_ = self.model(**kw)
                self.assertIsNone(draft_fold.make(SimpleNamespace(lm=m), 2, 3))
        self.assertIn("draft_fold", runtime.KNOWN_SLOTS)
        # the capture's own scratch: 3 splits of SGLang's 8192-column top-1 blocks
        slot = kernels.make_slot(torch.zeros(20000, 2560, dtype=torch.bfloat16), 2)
        self.assertEqual(tuple(slot.embedding.shape), (2, 2560))
        self.assertEqual(tuple(slot.partial_values.shape), (2, 3))

    def test_bf16_logits_are_the_lm_head_scaled_and_cut(self):
        from sgl_nvidia_mega.qwen3_8.runtime import draft_fold

        for scale, want in ((None, 1.0), (0.5, 0.5)):
            lm, *_ = self.model(scale=scale)
            logits = draft_fold.bf16_logits(lm, torch.zeros(3, 4))
            self.assertEqual(
                (tuple(logits.shape), logits.dtype), ((3, 6), torch.bfloat16)
            )
            self.assertTrue(torch.all(logits == want))
            ((_, _, head, bias),) = lm.logits_processor.calls
            self.assertIs(head, lm.lm_head)
            self.assertIsNone(bias)

    def test_finalize_each_step(self):
        from unittest import mock

        from sgl_nvidia_mega.qwen3_8.kernels.ops.speculative import draft_step_fold
        from sgl_nvidia_mega.qwen3_8.runtime import draft_fold

        ex = _executor(draft=True)
        slot = SimpleNamespace(embedding=torch.ones(2, 2560))
        launched = []
        with (
            mock.patch.object(
                draft_step_fold,
                "finalize_next_input",
                lambda *a: launched.append(("fold",) + a) or ("p", "i"),
            ),
            mock.patch.object(
                draft_step_fold,
                "draft_topk1_postprocess",
                lambda *a: launched.append(("last",) + a) or ("p2", "i2"),
            ),
        ):
            fold = draft_fold.StepFold(ex, slot, 3, True)
            self.assertEqual(fold.finalize("lg", "pos", "tok", 1, "mrope"), ("p", "i"))
            self.assertIs(ex.step_embedding, slot.embedding)
            self.assertEqual(fold.finalize("lg2", "pos", "tok", 2, None), ("p2", "i2"))
            self.assertIsNone(ex.step_embedding)
        self.assertEqual(
            launched,
            [
                ("fold", "lg", "pos", "tok", 1, slot, "mrope"),
                ("last", "lg2", "pos", "tok", 2, None),
            ],
        )

    def test_the_rounds_forwards_take_the_embedding_and_bf16_logits(self):
        from unittest import mock

        from sgl_nvidia_mega.qwen3_8.runtime import draft_fold

        ex = _executor(draft=True)
        recorder = types.ModuleType("sglang.srt.eplb.expert_distribution")
        recorder.get_global_expert_distribution_recorder = lambda: SimpleNamespace(
            disable_this_region=contextlib.nullcontext
        )
        logits_module = types.ModuleType("sglang.srt.layers.logits_processor")
        logits_module.LogitsProcessorOutput = lambda next_token_logits: SimpleNamespace(
            next_token_logits=next_token_logits, hidden_states=None
        )
        seen = []
        lm, *_ = self.model()
        lm._prepare_input_embeds = lambda ids, fb, e: seen.append("embed") or "stock"
        lm._mtp_input_fusion = lambda embeds, hidden: seen.append(embeds) or "streams"
        lm._set_hc_logits_hidden_states = lambda out, hc, fb: seen.append(("hc", hc))
        ex._language_model = lambda st, ids, streams: (torch.zeros(2, 4), "hc")
        ex.lm = lm
        embedding = torch.ones(2, 2560)
        fold = draft_fold.StepFold(ex, SimpleNamespace(embedding=embedding), 3, True)
        stock_fold = draft_fold.StepFold(ex, None, 3, False)
        other = draft_fold.StepFold(_executor(draft=True), None, 3, True)

        def run(fb_fold, pending=None):
            ex.step_embedding = pending
            seen.clear()
            lm.logits_processor.calls.clear()
            st = _St(_Mode.DECODE, 2)
            st.lm, st.input_ids, st.step.mode = lm, torch.zeros(2), "decode"
            st.fb.spec_info.hidden_states = "h"
            if fb_fold != "absent":
                st.fb.draft_step_fold = fb_fold
            with mock.patch.dict(
                sys.modules, {m.__name__: m for m in (recorder, logits_module)}
            ):
                out = ex._draft(st)
            return out, list(seen), dict(st.ran_ops), lm.logits_processor.calls

        out, seen_ops, ran, calls = run(fold, pending=embedding)
        self.assertIs(seen_ops[0], embedding)  # the folded embedding, no embed
        self.assertEqual(ran["logits"], "lm_head_bf16")
        self.assertEqual(out.next_token_logits.dtype, torch.bfloat16)
        self.assertEqual(seen_ops[-1], ("hc", "hc"))
        self.assertIsNone(ex.step_embedding)
        out, seen_ops, ran, _ = run(fold)  # the round's first step embeds its tokens
        self.assertEqual((seen_ops[0], ran["embed"]), ("embed", "stock"))
        self.assertEqual(ran["logits"], "lm_head_bf16")
        # no fold of this executor this round: SGLang's embedding and logits
        for fb_fold in (None, "absent", other):
            out, seen_ops, ran, calls = run(fb_fold, pending=embedding)
            self.assertEqual(seen_ops[0], "embed", fb_fold)
            self.assertEqual((ran["logits"], calls), ("stock", ["stock"]), fb_fold)
        # a fold over FP32 logits still takes the folded embedding
        _, seen_ops, ran, calls = run(stock_fold, pending=embedding)
        self.assertEqual((ran["embed"], ran["logits"]), ("draft_fold", "stock"))
        self.assertEqual(calls, ["stock"])

    def test_kernels_match_sglang_top1(self):
        import json
        import os
        import subprocess

        env = dict(os.environ, TRITON_INTERPRET="1", PYTHONDONTWRITEBYTECODE="1")
        done = subprocess.run(
            [sys.executable, "-c", _INTERPRETED_FOLD],
            env=env,
            capture_output=True,
            text=True,
            timeout=600,
        )
        self.assertEqual(done.returncode, 0, done.stderr[-2000:])
        got = json.loads(done.stdout.strip().splitlines()[-1])
        # token, topk_p, positions, M-RoPE, draft tokens (and the embedding) on the
        # BF16 logits, for a step that feeds another and for the last
        self.assertEqual(got, [[True] * 6, [True] * 5] * 3)


class TestEngagement(unittest.TestCase):
    def test_logs_each_new_set_once(self):
        e = Engagement("target")
        with self.assertLogs(
            "sgl_nvidia_mega.qwen3_8.runtime.engagement", logging.INFO
        ) as logs:
            for _ in range(3):
                e.begin()
                e.note("target_verify", 8, "gdn", "stock")
                e.note("target_verify", 8, "qsa", "stock")
                e.end("target_verify", 8)
            e.begin()
            e.note("target_verify", 8, "gdn", "gdn_core")
            e.end("target_verify", 8)
        self.assertEqual(len(logs.records), 2)
        self.assertIn("gdn=stock; qsa=stock", logs.output[0])
        self.assertIn("gdn=gdn_core", logs.output[1])


if __name__ == "__main__":
    unittest.main()
