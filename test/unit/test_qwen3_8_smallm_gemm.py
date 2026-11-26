# SPDX-License-Identifier: Apache-2.0
"""The Qwen3.8 executor's small-M GEMM routes on CPU (runtime/smallm_gemm.py): which
calls each route serves and which it leaves to SGLang, the options, the build
refusals and the build checks. SGLang's modules are fakes and the kernels CPU
stand-ins with the same arithmetic, so no GPU and no kernel build are needed."""

import sys
import types
import unittest
from types import SimpleNamespace
from unittest import mock

import torch
import torch.nn.functional as F

from sgl_nvidia_mega.qwen3_8.kernels.ops.gemm import splitk_act_mul
from sgl_nvidia_mega.qwen3_8.runtime import smallm_gemm as sg

BF16 = torch.bfloat16
N, K = sg.OUT_PROJ
INTER, HIDDEN = sg.INTER, sg.HIDDEN
LOGGER = "sgl_nvidia_mega.qwen3_8.runtime.smallm_gemm"
LOW, MID = range(4, 17), range(20, 33)


def _gemm(x, w):
    return x @ w.T


def _silu_mul(gate_up):
    g, u = gate_up.float().chunk(2, dim=-1)
    return (F.silu(g) * u).to(BF16)


def _unpermute(w_perm):
    """The inverse of permute_gate_up."""
    return w_perm.view(INTER // 32, 2, 32, -1).transpose(0, 1).reshape(2 * INTER, -1)


class _Unquant:  # UnquantizedLinearMethod
    pass


class _Fp8:
    pass


class _Linear:
    """A fake SGLang linear; calling it runs x @ weight.T."""

    def __init__(self, n, k, seed=0, quant=_Unquant, bias=None, gather_output=False):
        g = torch.Generator().manual_seed(seed)
        self.weight = (torch.randn(n, k, generator=g) * 0.02).to(BF16)
        self.quant_method = quant()
        self.bias = bias
        self.gather_output = gather_output
        self.calls = 0

    def __call__(self, x):
        self.calls += 1
        return _gemm(x, self.weight), None


class _SiluAndMul:
    def __init__(self, native=False):
        self._forward_method = None
        self.native = native
        self.calls = 0

    def forward_cuda(self, x):
        return _silu_mul(x)

    def forward_native(self, x):
        return _silu_mul(x)

    def _resolve_forward_method(self):
        return self.forward_native if self.native else self.forward_cuda

    def __call__(self, x):
        self.calls += 1
        return self._resolve_forward_method()(x)


class _MLP:  # the shared expert's Qwen2MoeMLP
    def __init__(self, seed=1):
        self.gate_up_proj = _Linear(2 * INTER, HIDDEN, seed)
        self.down_proj = _Linear(HIDDEN, INTER, seed + 1)
        self.act_fn = _SiluAndMul()
        self._enable_silu_fp4_quant_fusion = False


def _stock_mlp(mlp, x):
    return _gemm(_silu_mul(_gemm(x, mlp.gate_up_proj.weight)), mlp.down_proj.weight)


class _Block:  # Qwen2MoeSparseMoeBlock
    def __init__(self, shared):
        self.shared_expert = shared


class _GdnLayer:  # Qwen4ExpLinearDecoderLayer
    def __init__(self, out_proj, mlp):
        self.linear_attn = SimpleNamespace(out_proj=out_proj)
        self.mlp = mlp


class _St:
    def __init__(self):
        self.ran_ops = []

    def ran(self, op, impl):
        self.ran_ops.append((op, impl))


_STATE = {}


def _restore(name, old):
    if old is None:
        sys.modules.pop(name, None)
    else:
        sys.modules[name] = old


class _Case(unittest.TestCase):
    """Fake SGLang modules, CPU kernels and a fresh process-wide state per test."""

    def setUp(self):
        _STATE.update(tp=1, cutedsl=True, fused_backend=None, capturing=False)
        fakes = {
            "sglang.srt.layers.quantization.unquant": dict(
                UnquantizedLinearMethod=_Unquant,
                get_bf16_gemm_backend=lambda: SimpleNamespace(
                    is_cutedsl=lambda: _STATE["cutedsl"]
                ),
                _bf16_gemm_dispatch_impl=lambda x, w, bias: _gemm(x, w),
            ),
            "sglang.srt.runtime_context": dict(
                get_parallel=lambda: SimpleNamespace(tp_size=_STATE["tp"])
            ),
            "sglang.srt.models.qwen2_moe": dict(Qwen2MoeSparseMoeBlock=_Block),
            "sglang.srt.models.qwen4_exp": dict(Qwen4ExpLinearDecoderLayer=_GdnLayer),
            "sglang.srt.layers.activation": dict(
                SiluAndMul=_SiluAndMul, silu_and_mul=_silu_mul
            ),
            "sglang.kernels.fused_op": dict(
                get_fused_op_backend=lambda: _STATE["fused_backend"]
            ),
        }
        for name, attrs in fakes.items():
            module = types.ModuleType(name)
            module.__dict__.update(attrs)
            self.addCleanup(_restore, name, sys.modules.get(name))
            sys.modules[name] = module
        self.splitk_calls, self.act_calls, self.tgv_calls = [], [], []
        self.splitk_scale = {}  # tactic -> factor on the fake split-K's output
        self.act_flip = False
        kernels = SimpleNamespace(
            epi=SimpleNamespace(
                SplitKTactic=lambda *t: t, validate_tactic=self._validate
            ),
            act=SimpleNamespace(
                permute_gate_up=splitk_act_mul.permute_gate_up, run=self._act_run
            ),
            tgv=self._tgv,
            stream=None,
        )
        for name, value in (
            ("_k", kernels),
            ("_passed", {}),
            ("_projs", sg.weakref.WeakSet()),
            ("_act_passed", None),
            ("_prepared_weights", sg.weakref.WeakKeyDictionary()),
            ("_anchors", {}),
            ("_tgv_downs", sg.weakref.WeakSet()),
            ("_splitk", self._splitk),
        ):
            patcher = mock.patch.object(sg, name, value)
            patcher.start()
            self.addCleanup(patcher.stop)
        patcher = mock.patch.object(
            torch.cuda, "is_current_stream_capturing", lambda: _STATE["capturing"]
        )
        patcher.start()
        self.addCleanup(patcher.stop)

    @staticmethod
    def _validate(tactic, m, n, k):
        if tactic[1] not in (8, 16, 32):
            raise ValueError(f"unsupported mma_n={tactic[1]}")

    def _splitk(self, x_2d, weight, out, tactic):
        self.splitk_calls.append((x_2d.shape[0], tactic))
        out.copy_(_gemm(x_2d, weight) * self.splitk_scale.get(tactic, 1.0))

    def _act_run(self, x, w_perm, out, anchor):
        self.act_calls.append((x.shape[0], w_perm, anchor))
        out.copy_(_silu_mul(_gemm(x, _unpermute(w_perm))))
        if self.act_flip:
            out.view(-1).view(torch.int16)[0] ^= 1
        return out

    def _tgv(self, x, w):
        self.tgv_calls.append(x.shape[0])
        return _gemm(x, w)

    def install(self, projs=(), mlps=(), options=sg.DEFAULT):
        return sg.install(
            list(projs),
            list(mlps),
            options,
            device=torch.device("cpu"),
            stock_gemm=lambda x, w: _gemm(x, w),
        )


def _x(rows, k, seed=2):
    g = torch.Generator().manual_seed(seed)
    return torch.randn(rows, k, generator=g).to(BF16)


def _executor(options=sg.DEFAULT, draft=False):
    return SimpleNamespace(smallm_gemm=options, draft=draft)


class TestOptions(unittest.TestCase):
    def test_defaults_are_the_source_routes(self):
        d = sg.DEFAULT
        self.assertEqual(d.rel_l2_max, 1e-3)
        self.assertTrue(d.shared_fused_act)
        self.assertEqual(d.shared_down_tgv_rows, (4, 8, 16))
        tactics = sg._out_proj_tactics(d)
        self.assertEqual(sorted(tactics), [*LOW, *MID])
        self.assertEqual({tactics[m] for m in range(4, 9)}, {(64, 8, 4, 8)})
        self.assertEqual({tactics[m] for m in range(9, 17)}, {(64, 16, 4, 10)})
        self.assertEqual({tactics[m] for m in MID}, {(64, 32, 4, 8)})

    def test_recipe_switches_with_the_source_gating(self):
        envs = {
            "SGLANG_OPT_SM107_BF16_SMALLM_ROUTES": "1",
            sg.NONBITWISE_LOWM_ENV: "1",
            sg.NONBITWISE_ENV: "1",
            sg.FUSED_ACT_ENV: "1",
            sg.DOWN_TGV_ENV: "1",
        }
        with mock.patch.dict("os.environ", envs, clear=False):
            self.assertEqual(sg.recipe_options(), sg.DEFAULT)
        # the source's defaults: everything off
        with mock.patch.dict("os.environ", {}, clear=True):
            self.assertEqual(
                sg.recipe_options(),
                sg.Options(
                    out_proj_low_m=(),
                    out_proj_mid_m=(),
                    shared_fused_act=False,
                    shared_down_tgv_rows=(),
                ),
            )
        # the non-bitwise routes and the fused act need the bitwise routes on
        off = dict(envs, SGLANG_OPT_SM107_BF16_SMALLM_ROUTES="0")
        with mock.patch.dict("os.environ", off, clear=True):
            o = sg.recipe_options()
        self.assertEqual(
            (o.out_proj_low_m, o.out_proj_mid_m, o.shared_fused_act), ((), (), False)
        )
        self.assertEqual(o.shared_down_tgv_rows, (4, 8, 16))

    def test_overlapping_ranges_are_refused(self):
        with self.assertRaises(ValueError):
            sg.Options(out_proj_mid_m=(((16, 32), (64, 32, 4, 8)),))


class TestOutProj(_Case):
    def setUp(self):
        super().setUp()
        self.proj = _Linear(N, K)
        self.assertEqual(self.install(projs=[self.proj]), "")
        self.splitk_calls.clear()

    def test_routed_rows_run_split_k_with_their_tactic(self):
        tactics = sg._out_proj_tactics(sg.DEFAULT)
        for rows in (4, 8, 9, 16, 20, 32):
            st = _St()
            x = _x(rows, K)
            out = sg.out_proj(_executor(), self.proj, x, st, "gdn.out_proj")
            self.assertTrue(torch.equal(out, _gemm(x, self.proj.weight)))
            self.assertEqual(self.splitk_calls[-1], (rows, tactics[rows]))
            self.assertEqual(st.ran_ops, [("gdn.out_proj", "splitk")])
        self.assertEqual(self.proj.calls, 0)

    def test_leading_dims_are_kept(self):
        out = sg.out_proj(_executor(), self.proj, _x(4, K).view(2, 2, K), _St(), "o")
        self.assertEqual(tuple(out.shape), (2, 2, N))

    def test_other_calls_stay_stock(self):
        other = _Linear(N, K)
        x8 = _x(8, K)
        cases = [(_executor(), self.proj, _x(rows, K)) for rows in (1, 3, 17, 19, 33)]
        cases += [
            (_executor(None), self.proj, x8),  # slot off
            (_executor(), other, x8),  # a projection build did not take
            (_executor(), self.proj, x8.half()),
            (_executor(), self.proj, _x(K, 8).t()),  # not contiguous
        ]
        for ex, proj, x in cases:
            st = _St()
            self.assertIsNone(sg.out_proj(ex, proj, x, st, "gdn.out_proj"))
            self.assertEqual(st.ran_ops, [])
        self.assertEqual(self.splitk_calls, [])


class TestSharedExpert(_Case):
    def setUp(self):
        super().setUp()
        self.mlp = _MLP()
        self.assertEqual(self.install(mlps=[self.mlp]), "")
        self.act_calls.clear()
        self.tgv_calls.clear()

    def run_step(self, rows, draft=False, options=sg.DEFAULT):
        st = _St()
        x = _x(rows, HIDDEN, seed=rows)
        out = sg.shared_expert(_executor(options, draft), self.mlp, x, st)
        if out is not None:
            self.assertTrue(torch.equal(out, _stock_mlp(self.mlp, x)), rows)
        return out, st.ran_ops

    def test_fused_act_at_1_to_16_rows(self):
        w_perm = splitk_act_mul.permute_gate_up(self.mlp.gate_up_proj.weight)
        for rows in range(1, 17):
            _, ran = self.run_step(rows, draft=True)
            self.assertEqual(ran, [("moe.shared_expert", "act_splitk")])
            got_rows, got_w, anchor = self.act_calls[-1]
            self.assertEqual(got_rows, rows)
            self.assertTrue(torch.equal(got_w, w_perm))
            self.assertEqual(tuple(anchor.shape), (rows, 2 * INTER))
        self.assertEqual((self.mlp.gate_up_proj.calls, self.mlp.act_fn.calls), (0, 0))
        self.assertEqual(self.mlp.down_proj.calls, 16)
        self.assertEqual(self.tgv_calls, [])

    def test_tgv_down_on_the_target_at_4_8_16(self):
        for rows in range(1, 17):
            _, ran = self.run_step(rows)
            impl = "act_splitk+down_tgv" if rows in (4, 8, 16) else "act_splitk"
            self.assertEqual(ran, [("moe.shared_expert", impl)])
        self.assertEqual(self.tgv_calls, [4, 8, 16])

    def test_tgv_alone_where_the_act_does_not_serve(self):
        sg._act_passed = False
        for rows, impl in ((8, "down_tgv"), (3, None)):
            out, ran = self.run_step(rows)
            self.assertEqual(ran, [("moe.shared_expert", impl)] if impl else [])
            self.assertEqual(out is None, impl is None)
        self.assertEqual(self.act_calls, [])
        self.assertEqual((self.mlp.gate_up_proj.calls, self.mlp.act_fn.calls), (1, 1))

    def test_other_calls_stay_stock(self):
        off = sg.Options(shared_fused_act=False, shared_down_tgv_rows=())
        for rows, draft, options in (
            (17, False, sg.DEFAULT),
            (32, False, sg.DEFAULT),
            (4, False, off),
        ):
            out, ran = self.run_step(rows, draft, options)
            self.assertIsNone(out)
            self.assertEqual(ran, [])
        x = _x(4, HIDDEN)
        for ex, mlp, x in (
            (_executor(None), self.mlp, x),
            (_executor(), _MLP(seed=5), x),  # a shared expert build did not take
            (_executor(), self.mlp, x.half()),
        ):
            self.assertIsNone(sg.shared_expert(ex, mlp, x, _St()))
        self.assertEqual((self.act_calls, self.tgv_calls), ([], []))

    def test_a_new_weight_is_permuted_again_outside_capture_only(self):
        gate_up = self.mlp.gate_up_proj
        gate_up.weight = gate_up.weight.clone() * 2
        _STATE["capturing"] = True
        out, ran = self.run_step(3, draft=True)
        self.assertIsNone(out)
        _STATE["capturing"] = False
        _, ran = self.run_step(3, draft=True)
        self.assertEqual(ran, [("moe.shared_expert", "act_splitk")])
        self.assertEqual(sg._prepared_weights[gate_up][0], gate_up.weight.data_ptr())


class TestBuild(_Case):
    def build(self, model, capability=(10, 7)):
        with (
            mock.patch.object(torch.cuda, "is_available", lambda: True),
            mock.patch.object(torch.cuda, "get_device_capability", lambda: capability),
            mock.patch.object(torch.cuda, "current_device", lambda: 0),
        ):
            return sg.build(model)

    def model(self):
        attn = SimpleNamespace(o_proj=_Linear(N, K, seed=7), mlp=_Block(None))
        gdn = _GdnLayer(_Linear(N, K, seed=8), _Block(_MLP()))
        dense = SimpleNamespace(o_proj=_Linear(N, K, seed=9), mlp=object())
        m = SimpleNamespace(layers=[attn, gdn, dense], start_layer=0, end_layer=3)
        return SimpleNamespace(model=m), (attn, gdn, dense)

    def test_refusals(self):
        model, _ = self.model()
        with mock.patch.object(torch.cuda, "is_available", lambda: False):
            self.assertIn("SM107", sg.build(model))
        self.assertIn("SM107", self.build(model, capability=(10, 0)))
        _STATE["tp"] = 2
        self.assertIn("TP1", self.build(model))
        _STATE.update(tp=1, cutedsl=False)
        self.assertIn("cutedsl", self.build(model))

    def test_hands_the_layers_and_sglang_dispatch_to_install(self):
        model, (attn, gdn, dense) = self.model()
        seen = {}

        def install(projs, mlps, options, *, device, stock_gemm):
            seen.update(projs=projs, mlps=mlps, device=device, stock=stock_gemm)
            return ""

        with mock.patch.object(sg, "install", install):
            self.assertEqual(self.build(model), "")
        self.assertEqual(
            seen["projs"], [attn.o_proj, gdn.linear_attn.out_proj, dense.o_proj]
        )
        self.assertEqual(seen["mlps"], [gdn.mlp.shared_expert])
        self.assertEqual(seen["device"], torch.device("cuda", 0))
        x, w = _x(4, K), attn.o_proj.weight
        self.assertTrue(torch.equal(seen["stock"](x, w), _gemm(x, w)))

    def test_layer_reasons(self):
        self.assertEqual(sg._proj_reason(_Linear(N, K)), "")
        for proj in (
            _Linear(N, K, quant=_Fp8),
            _Linear(N, K, bias=torch.zeros(N)),
            _Linear(N, 4096),
        ):
            self.assertTrue(sg._proj_reason(proj))
        self.assertEqual((sg._act_reason(_MLP()), sg._down_reason(_MLP())), ("", ""))
        changes = {
            "fp4": lambda m: setattr(m, "_enable_silu_fp4_quant_fusion", True),
            "gather": lambda m: setattr(m.gate_up_proj, "gather_output", True),
            "quant": lambda m: setattr(m.gate_up_proj, "quant_method", _Fp8()),
            "shape": lambda m: setattr(m, "gate_up_proj", _Linear(2 * 512, HIDDEN)),
            "act": lambda m: setattr(m, "act_fn", object()),
            "native": lambda m: setattr(m, "act_fn", _SiluAndMul(native=True)),
            "backend": lambda m: _STATE.update(fused_backend="triton"),
        }
        for name, change in changes.items():
            mlp = _MLP()
            change(mlp)
            self.assertTrue(sg._act_reason(mlp), name)
            _STATE["fused_backend"] = None
        mlp = _MLP()
        mlp.down_proj = _Linear(HIDDEN, 512)
        self.assertTrue(sg._down_reason(mlp))

    def test_each_check_runs_once_a_process(self):
        proj, mlp = _Linear(N, K), _MLP()
        for _ in range(2):
            self.assertEqual(self.install([proj], [mlp]), "")
        self.assertEqual(sorted(r for r, _ in self.splitk_calls), [*LOW, *MID])
        self.assertEqual(len(self.act_calls), 17)  # 1-16 rows and the edge case
        self.assertIn(proj, sg._projs)
        self.assertIn(mlp.gate_up_proj, sg._prepared_weights)
        self.assertIn(mlp.down_proj, sg._tgv_downs)
        # build compiles TGV's pick for each of its rows
        self.assertEqual(self.tgv_calls, [4, 8, 16, 4, 8, 16])

    def test_a_route_past_the_bound_keeps_sglang(self):
        self.splitk_scale[(64, 16, 4, 10)] = 1.01  # relative L2 0.01
        proj = _Linear(N, K)
        with self.assertLogs(LOGGER, "WARNING") as logs:
            self.assertEqual(self.install([proj]), "")
        self.assertEqual(len(logs.records), 8)
        for rows in (*LOW, *MID):
            served = sg.out_proj(_executor(), proj, _x(rows, K), _St(), "o")
            self.assertEqual(served is None, 9 <= rows <= 16, rows)

    def test_an_invalid_tactic_keeps_sglang(self):
        options = sg.Options(out_proj_low_m=(((4, 4), (64, 12, 4, 8)),))
        proj = _Linear(N, K)
        with self.assertLogs(LOGGER, "WARNING"):
            self.assertEqual(self.install([proj], options=options), "")
        self.assertIsNone(sg.out_proj(_executor(options), proj, _x(4, K), _St(), "o"))
        self.assertIsNotNone(
            sg.out_proj(_executor(options), proj, _x(20, K), _St(), "o")
        )

    def test_an_act_mismatch_turns_only_the_act_off(self):
        self.act_flip = True
        mlp = _MLP()
        with self.assertLogs(LOGGER, "WARNING"):
            self.assertEqual(self.install([_Linear(N, K)], [mlp]), "")
        self.assertFalse(sg._act_passed)
        self.assertEqual(len(self.act_calls), 1)
        self.assertNotIn(mlp.gate_up_proj, sg._prepared_weights)
        self.assertIn(mlp.down_proj, sg._tgv_downs)

    def test_nothing_served_gives_the_reasons(self):
        reason = self.install([_Linear(N, K, quant=_Fp8)], [])
        self.assertIn("out_proj: not an unquantized linear", reason)
        off = sg.Options(shared_fused_act=False, shared_down_tgv_rows=())
        self.assertEqual(self.install([], [_MLP()], off), "no route serves this model")

    def test_kernels_that_do_not_import(self):
        def missing():
            raise ImportError("No module named 'cutlass'")

        with (
            mock.patch.object(sg, "_k", None),
            mock.patch.object(sg, "_kernels", missing),
        ):
            self.assertIn("cutlass", self.install([_Linear(N, K)]))


class TestKernelCopy(unittest.TestCase):
    def test_permuted_tile_holds_gate_then_up_of_the_same_columns(self):
        rows = torch.arange(2 * INTER, dtype=torch.float32)
        w = rows[:, None].expand(2 * INTER, 8).contiguous()
        got = splitk_act_mul.permute_gate_up(w)[:, 0].view(INTER // 32, 64)
        for t in range(INTER // 32):
            want = torch.cat([rows[32 * t : 32 * t + 32], rows[INTER + 32 * t :][:32]])
            self.assertTrue(torch.equal(got[t], want), t)
        self.assertTrue(torch.equal(_unpermute(splitk_act_mul.permute_gate_up(w)), w))

    def test_epilogue_is_the_fast_math_sequence(self):
        ops = [
            line.split()[0]
            for line in splitk_act_mul.SILU_MUL_PTX.replace(";", "\n").splitlines()
            if line.strip() and not line.strip().startswith(("{", "}", ".reg"))
        ]
        self.assertEqual(
            ops,
            [
                "mul.ftz.f32",
                "ex2.approx.ftz.f32",
                "add.ftz.f32",
                "div.approx.ftz.f32",
                "mul.ftz.f32",
            ],
        )


if __name__ == "__main__":
    unittest.main()
