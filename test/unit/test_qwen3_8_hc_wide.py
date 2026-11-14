# SPDX-License-Identifier: Apache-2.0
"""The wide HC dispatch (runtime/hc_wide.py) on CPU: which kernel each row count and
limit reaches, and what each kernel is handed. The kernel modules and SGLang's
permute_pad_up_weight are spies installed through sys.modules, the HC modules are
stand-ins with GatedResidual's attributes and the tensors are meta tensors that
report CUDA, so no GPU, compiler or SGLang kernel runs."""

import inspect
import sys
import types
import unittest
from types import SimpleNamespace
from unittest import mock

import torch

from sgl_nvidia_mega.qwen3_8.runtime import hc_wide
from sgl_nvidia_mega.qwen3_8.runtime.hc_wide import (
    DEFAULT,
    HIGH_CONCURRENCY,
    MID_CONCURRENCY,
    Limits,
)

BF16 = torch.bfloat16
HC, HS, RANK = 4, 2560, 320
WIDTH = HC * HS
_OPS = "sgl_nvidia_mega.qwen3_8.kernels.ops"


class _Cuda(torch.Tensor):
    @property
    def is_cuda(self):
        return True


def _t(*shape, dtype=BF16, cuda=True):
    t = torch.empty(shape, dtype=dtype, device="meta")
    return t.as_subclass(_Cuda) if cuda else t


class _Norm:
    def __init__(self, group_size=HS, dtype=BF16, width=WIDTH):
        self.group_size = group_size
        self.weight = torch.empty(width, dtype=dtype, device="meta")
        self.variance_epsilon = 1e-6
        self.calls = 0

    def __call__(self, x):
        self.calls += 1
        return _t(*x.shape, dtype=x.dtype)


class _HC:
    """GatedResidual's attributes (main): per-branch norm, mix and inject weights."""

    params_dtype = BF16

    def __init__(self, combine=True, use_mix=True, hc=HC, hs=HS, per_branch=True):
        self.hc_count, self.hidden_size = hc, hs
        width = hc * hs
        self.hc_norm = _Norm(hs if per_branch else None, width=width)
        self._mix_up_weight_padded = None
        if use_mix:
            self.input_mix_weight_down = SimpleNamespace(
                weight=torch.empty(RANK, width, dtype=BF16, device="meta")
            )
            self.input_mix_weight_up = SimpleNamespace(
                weight=torch.empty(width, RANK, dtype=BF16, device="meta")
            )
        if combine:
            self.block_inject_weight = SimpleNamespace(
                weight=torch.empty(hc, width, dtype=BF16, device="meta")
            )
        self.mix_calls, self.combine_calls = [], []

    def mix(self, hyper_input, normalized_input=None):
        self.mix_calls.append((hyper_input, normalized_input))
        return "stock mixed", (hyper_input, normalized_input, None)

    def combine(self, block_output, residuals):
        self.combine_calls.append((block_output, residuals))
        return "stock combined"


class _Spies:
    def __init__(self):
        self.calls = []

    def __call__(self, name, result=None):
        def spy(*args, **kwargs):
            self.calls.append((name, args, kwargs))
            if isinstance(result, BaseException):
                raise result
            return result(*args, **kwargs) if callable(result) else result

        return spy

    def named(self, name):
        return [(a, k) for n, a, k in self.calls if n == name]


def _modules(spies):
    def module(name, **attrs):
        m = types.ModuleType(name)
        m.__dict__.update(attrs)
        return m

    def ks_launch(block, residual, **kw):
        normed = _t(*residual.shape) if kw["write_normed"] else None
        totals = (
            _t(residual.shape[0], HC, dtype=torch.float32)
            if kw["next_inject_weight"] is not None
            else None
        )
        return "ks combined", normed, totals, "ks mixed"

    return {
        f"{_OPS}.gemm.hc_mix_wide": module(
            f"{_OPS}.gemm.hc_mix_wide",
            WIDE_MAX_ROWS=512,
            hc_mix_wide=spies(
                "hc_mix_wide", lambda x, *a: _t(x.shape[0], HS, dtype=x.dtype)
            ),
        ),
        f"{_OPS}.gemm.hc_boundary_ks": module(
            f"{_OPS}.gemm.hc_boundary_ks",
            MAX_ROWS=320,
            FLAG_WRITE_NORMED=4,
            supported=spies("supported", True),
            hc_boundary_ks=spies("hc_boundary_ks", ks_launch),
            warmup=spies("ks.warmup"),
        ),
        f"{_OPS}.elementwise.hc_combine_extend": module(
            f"{_OPS}.elementwise.hc_combine_extend",
            hc_combine_extend=spies(
                "hc_combine_extend", ("ext combined", "ext normed", "ext totals")
            ),
            warmup=spies("ext.warmup"),
        ),
        f"{_OPS}.elementwise.hc_combine_decode_wide": module(
            f"{_OPS}.elementwise.hc_combine_decode_wide",
            WIDE_MAX_ROWS=128,
            hc_combine_gate=spies(
                "hc_combine_gate",
                lambda normed, w: _t(normed.shape[0], 8, HC, dtype=torch.float32),
            ),
            hc_combine_apply=spies("hc_combine_apply", "applied"),
            hc_combine_apply_norm_wide=spies(
                "hc_combine_apply_norm_wide", ("wide combined", "wide normed")
            ),
            _jit_module=spies("decode._jit_module"),
        ),
        # SGLang's split-K epilogue GEMM with its max_m (sgl-project/sglang#50194)
        "sglang.kernels.ops.gemm.dense_bf16_gemm_sm100_splitk_epilogue": module(
            "sglang.kernels.ops.gemm.dense_bf16_gemm_sm100_splitk_epilogue",
            validate_tactic=lambda tactic, m, n, k, *, max_m=32: None,
        ),
        f"{_OPS}.gemm.hc_mix_up_gate": module(f"{_OPS}.gemm.hc_mix_up_gate"),
        "sglang.kernels.ops.gemm.hc_mix_splitk": module(
            "sglang.kernels.ops.gemm.hc_mix_splitk",
            permute_pad_up_weight=spies("permute_pad_up_weight", "padded up"),
        ),
    }


class _Case(unittest.TestCase):
    def setUp(self):
        self.spies = _Spies()
        for patcher in (
            mock.patch.dict(sys.modules, _modules(self.spies)),
            mock.patch.object(hc_wide, "_beside", self._beside),
            mock.patch.object(hc_wide, "_join", self.spies("join")),
            mock.patch.object(torch.cuda, "current_device", return_value=0),
        ):
            patcher.start()
            self.addCleanup(patcher.stop)

    def _beside(self, fn, normed, limits):
        self.spies.calls.append(("beside", (normed, limits), {}))
        return fn()


class TestLimits(unittest.TestCase):
    def test_recipe_values(self):
        # high concurrency: SGLANG_HC_MIX_SPLITK_MAX_ROWS=512, SGLANG_X_HC_EXTEND_MIN_ROWS
        # =33, SGLANG_HC_BOUNDARY_KS_MAX_ROWS=320, SGLANG_FORWARD_HELPER_STREAM_PRIORITY=-1
        self.assertEqual(HIGH_CONCURRENCY, Limits(512, 33, 320, 3, 0, -1))
        # mid concurrency: SGLANG_X_HC_EXTEND_MIN_ROWS=33, SGLANG_HC_FUSED_COMBINE_MAX_ROWS=128
        self.assertEqual(MID_CONCURRENCY, Limits(0, 33, 0, 3, 128, None))
        self.assertIs(DEFAULT, HIGH_CONCURRENCY)

    def test_reads_no_environment(self):
        source = inspect.getsource(hc_wide)
        for read in ("os.environ", "getenv(", "envs.", "srt.environ", "EnvBool"):
            self.assertNotIn(read, source)


class TestRoutes(_Case):
    def test_high_concurrency(self):
        hc = _HC()
        mix, ext, ks = "hc_mix_wide", "hc_combine_extend", "hc_boundary_ks"
        for rows, expected in (
            (32, ()),
            (33, (mix, ext, ks)),
            (320, (mix, ext, ks)),
            (321, (mix, ext)),
            (512, (mix, ext)),
            (513, (ext,)),
            (4096, (ext,)),
        ):
            with self.subTest(rows=rows):
                self.assertEqual(hc_wide.routes(hc, rows), expected)
                self.assertEqual(hc_wide.supported(hc, rows) == "", bool(expected))

    def test_mid_concurrency(self):
        hc = _HC()
        ext, wide = "hc_combine_extend", "hc_combine_apply_norm_wide"
        for rows, expected in ((33, (ext, wide)), (128, (ext, wide)), (129, (ext,))):
            with self.subTest(rows=rows):
                self.assertEqual(
                    hc_wide.routes(hc, rows, limits=MID_CONCURRENCY), expected
                )

    def test_final_mixer_reads_only(self):
        mixer = _HC(combine=False)
        self.assertEqual(hc_wide.routes(mixer, 64), ("hc_mix_wide", "hc_boundary_ks"))
        self.assertEqual(hc_wide.routes(mixer, 400), ("hc_mix_wide",))
        self.assertTrue(hc_wide.supported(mixer, 600))

    def test_reasons(self):
        self.assertIn("runtime/hc.py", hc_wide.supported(_HC(), 32))
        self.assertIn("per-branch", hc_wide.supported(_HC(per_branch=False), 64))
        # The MTP draft's extra stream: 5 x 2560 fits none of the kernels
        self.assertIn("5 x 2560", hc_wide.supported(_HC(hc=5), 64))

    def test_limits_clamp_to_the_kernels(self):
        hc = _HC()
        wide = Limits(
            mix_wide_max_rows=4096,
            extend_min_rows=1,
            boundary_ks_max_rows=4096,
            fused_combine_max_rows=4096,
        )
        self.assertEqual(len(hc_wide.routes(hc, 33, limits=wide)), 4)
        self.assertEqual(hc_wide.routes(hc, 4096, limits=wide), ("hc_combine_extend",))
        # The one-launch boundary needs the extend boundary; caps below 33 are off
        self.assertEqual(
            hc_wide.routes(hc, 64, limits=Limits(boundary_ks_max_rows=320)), ()
        )
        self.assertEqual(
            hc_wide.routes(hc, 33, limits=Limits(fused_combine_max_rows=32)), ()
        )


class TestMix(_Case):
    def test_wide_mix(self):
        hc = _HC()
        streams, totals = _t(64, WIDTH), _t(64, HC, dtype=torch.float32)
        for _ in range(2):
            mixed, normed, partials = hc_wide.mix(hc, streams, None, totals)
        self.assertEqual(hc.hc_norm.calls, 2)
        self.assertIs(partials, totals)
        self.assertEqual(tuple(mixed.shape), (64, HS))
        self.assertEqual(mixed.dtype, BF16)
        calls = self.spies.named("hc_mix_wide")
        self.assertEqual(len(calls), 2)
        args, _ = calls[-1]
        self.assertIs(args[0], normed)
        self.assertEqual(tuple(args[1].shape), (RANK, WIDTH))  # the down weight
        self.assertEqual(args[2:], ("padded up", HC, HS))
        # The padded up weight is made once and kept on the module
        [(perm, _)] = self.spies.named("permute_pad_up_weight")
        self.assertEqual(perm, (hc.input_mix_weight_up.weight, HC))
        self.assertEqual(hc._mix_up_weight_padded, "padded up")
        self.assertEqual(hc.mix_calls, [])
        self.assertEqual(hc_wide.mix_route(hc, normed), "hc_mix_wide")

    def test_handed_normalized_skips_the_norm(self):
        hc = _HC()
        normalized = _t(64, WIDTH)
        _, normed, _ = hc_wide.mix(hc, _t(64, WIDTH), normalized)
        self.assertIs(normed, normalized)
        self.assertEqual(hc.hc_norm.calls, 0)

    def test_stock_mix(self):
        for rows, cuda, compiling in (
            (600, True, False),
            (64, False, False),
            (64, True, True),
        ):
            with (
                self.subTest(rows=rows, cuda=cuda, compiling=compiling),
                mock.patch.object(
                    torch.compiler, "is_compiling", return_value=compiling
                ),
            ):
                hc = _HC()
                streams = _t(rows, WIDTH, cuda=cuda)
                normalized = _t(rows, WIDTH, cuda=cuda)
                mixed, normed, partials = hc_wide.mix(hc, streams, normalized)
                self.assertEqual(mixed, "stock mixed")
                self.assertIsNone(partials)
                self.assertEqual(hc.mix_calls, [(streams, normalized)])
        self.assertEqual(self.spies.named("hc_mix_wide"), [])

    def test_side_stream_gate(self):
        hc = _HC()
        mixed, normed, partials = hc_wide.mix(hc, _t(64, WIDTH), limits=MID_CONCURRENCY)
        self.assertEqual(mixed, "stock mixed")  # MID keeps the stock mix
        [((beside_normed, limits), _)] = self.spies.named("beside")
        self.assertIs(beside_normed, normed)
        self.assertIs(limits, MID_CONCURRENCY)
        [(gate, _)] = self.spies.named("hc_combine_gate")
        self.assertIs(gate[0], normed)
        self.assertIs(gate[1], hc.block_inject_weight.weight)
        self.assertEqual(tuple(partials.shape), (64, 8, HC))

    def test_no_side_gate(self):
        carried = _t(64, HC, dtype=torch.float32)
        for rows, partials, limits in (
            (64, carried, MID_CONCURRENCY),
            (129, None, MID_CONCURRENCY),
            (64, None, HIGH_CONCURRENCY),
        ):
            with self.subTest(rows=rows, carried=partials is not None, limits=limits):
                _, _, out = hc_wide.mix(
                    _HC(), _t(rows, WIDTH), None, partials, limits=limits
                )
                self.assertIs(out, partials)
        self.assertEqual(self.spies.named("beside"), [])


class TestWrite(_Case):
    def args(self, rows=64):
        hc, nxt = _HC(), _HC()
        return hc, nxt, _t(rows, HS), _t(rows, WIDTH), _t(rows, WIDTH)

    def test_extend_without_partials(self):
        hc, nxt, block, residual, normed = self.args()
        inject = nxt.block_inject_weight.weight
        self.assertEqual(
            hc_wide.write_route(hc, block, residual, normed, None, nxt.hc_norm),
            "hc_combine_extend",
        )
        out = hc_wide.write(hc, block, residual, normed, None, nxt.hc_norm, inject)
        self.assertEqual(out, ("ext combined", "ext normed", "ext totals"))
        [(args, kw)] = self.spies.named("hc_combine_extend")
        self.assertEqual(args, (block, residual, HC, HS))
        self.assertIsNone(kw["partials"])
        self.assertIs(kw["normed_residual"], normed)
        self.assertIs(kw["inject_weight"], hc.block_inject_weight.weight)
        self.assertIs(kw["norm_weight"], nxt.hc_norm.weight)
        self.assertEqual(kw["eps"], 1e-6)
        self.assertIs(kw["next_inject_weight"], inject)

    def test_extend_reads_the_totals(self):
        hc, nxt, block, residual, normed = self.args()
        totals = _t(64, HC, dtype=torch.float32)
        hc_wide.write(hc, block, residual, normed, totals, nxt.hc_norm)
        [(_, kw)] = self.spies.named("hc_combine_extend")
        self.assertIs(kw["partials"], totals)
        self.assertIsNone(kw["normed_residual"])
        self.assertIsNone(kw["inject_weight"])
        self.assertIsNone(kw["next_inject_weight"])

    def test_extend_drops_a_next_norm_or_gate_it_cannot_take(self):
        hc, nxt, block, residual, normed = self.args()
        inject = nxt.block_inject_weight.weight
        fp32_norm = _Norm(dtype=torch.float32)
        wrong_inject = torch.empty(HC, HS, dtype=BF16, device="meta")
        for norm, gate, norm_kept, gate_kept in (
            (fp32_norm, inject, False, False),
            (None, inject, False, False),  # a PLE layer reads next
            (nxt.hc_norm, wrong_inject, True, False),
        ):
            self.spies.calls.clear()
            hc_wide.write(hc, block, residual, normed, None, norm, gate)
            [(_, kw)] = self.spies.named("hc_combine_extend")
            self.assertEqual(kw["norm_weight"] is not None, norm_kept)
            self.assertEqual(kw["eps"], 1e-6 if norm_kept else 0.0)
            self.assertEqual(kw["next_inject_weight"] is not None, gate_kept)

    def test_wide_combine_norm(self):
        hc, nxt, block, residual, normed = self.args()
        partials = _t(64, 8, HC, dtype=torch.float32)
        mid = MID_CONCURRENCY
        out = hc_wide.write(
            hc, block, residual, normed, partials, nxt.hc_norm, limits=mid
        )
        self.assertEqual(out, ("wide combined", "wide normed", None))
        [((joined, limits), _)] = self.spies.named("join")
        self.assertIs(joined, partials)
        self.assertIs(limits, mid)
        [(args, _)] = self.spies.named("hc_combine_apply_norm_wide")
        self.assertEqual(args, (block, residual, partials, nxt.hc_norm.weight, 1e-6))
        # Before a PLE layer: the partials applied, no norm
        out = hc_wide.write(hc, block, residual, normed, partials, None, limits=mid)
        self.assertEqual(out, ("applied", None, None))
        self.assertEqual(len(self.spies.named("join")), 2)
        [(args, _)] = self.spies.named("hc_combine_apply")
        self.assertEqual(args, (block, residual, partials))
        self.assertEqual(self.spies.named("hc_combine_extend"), [])

    def test_stock_write_back(self):
        totals = _t(64, HC, dtype=torch.float32)
        for rows, partials, limits in (
            (32, None, HIGH_CONCURRENCY),
            (64, totals, Limits(extend_min_rows=1024)),  # totals no kernel reads
            (64, None, Limits()),
        ):
            with self.subTest(rows=rows, limits=limits):
                hc, nxt, block, residual, normed = self.args(rows)
                out = hc_wide.write(
                    hc, block, residual, normed, partials, nxt.hc_norm, limits=limits
                )
                self.assertEqual(out, ("stock combined", None, None))
                self.assertEqual(hc.combine_calls, [(block, (residual, normed, None))])
        self.assertEqual(self.spies.named("hc_combine_extend"), [])
        self.assertEqual(self.spies.named("join"), [])


class TestBoundary(_Case):
    def args(self, rows=64):
        writer, reader = _HC(), _HC()
        totals = _t(rows, HC, dtype=torch.float32)
        return writer, reader, _t(rows, HS), _t(rows, WIDTH), totals

    def test_one_launch(self):
        for flags, gate, written in (
            (3, True, False),
            (7, True, True),
            (3, False, True),
        ):
            with self.subTest(flags=flags, gate=gate):
                self.spies.calls.clear()
                writer, reader, block, residual, totals = self.args()
                inject = reader.block_inject_weight.weight if gate else None
                limits = Limits(512, 33, 320, flags)
                self.assertEqual(
                    hc_wide.boundary_reason(
                        writer, block, residual, totals, reader, inject, limits=limits
                    ),
                    "",
                )
                self.assertEqual(self.spies.named("supported"), [((64, 0), {})])
                combined, normalized, next_totals, mixed = hc_wide.boundary(
                    writer, block, residual, totals, reader, inject, limits=limits
                )
                [(args, kw)] = self.spies.named("hc_boundary_ks")
                self.assertEqual(args, (block, residual))
                self.assertIs(kw["partials"], totals)
                self.assertIs(kw["norm_weight"], reader.hc_norm.weight)
                self.assertEqual(kw["eps"], 1e-6)
                self.assertIs(kw["next_inject_weight"], inject)
                self.assertEqual(tuple(kw["w_down"].shape), (RANK, WIDTH))
                self.assertEqual(kw["w_up_permuted"], "padded up")
                self.assertEqual(kw["write_normed"], written)
                self.assertEqual(kw["flags"], flags)
                self.assertEqual((combined, mixed), ("ks combined", "ks mixed"))
                self.assertEqual(normalized.numel() > 0, written)
                self.assertEqual(next_totals is not None, gate)
                [(perm, _)] = self.spies.named("permute_pad_up_weight")
                self.assertIs(perm[0], reader.input_mix_weight_up.weight)

    def test_the_final_mixer_reads(self):
        writer, _, block, residual, totals = self.args()
        mixer = _HC(combine=False)
        self.assertEqual(
            hc_wide.boundary_reason(writer, block, residual, totals, mixer), ""
        )

    def test_declined(self):
        cases = {
            "decode partials": dict(partials=_t(64, 8, HC, dtype=torch.float32)),
            "no partials": dict(partials=None),
            "rows above the cap": dict(rows=321),
            "rows of hc.py": dict(rows=32),
            "MID limits": dict(limits=MID_CONCURRENCY),
            "a foreign next gate": dict(inject=_HC().block_inject_weight.weight),
            "a writer without the extend boundary": dict(writer=_HC(combine=False)),
            "a reader without mix weights": dict(reader=_HC(use_mix=False)),
            "strided totals": dict(
                partials=_t(64, 2 * HC, dtype=torch.float32)[:, :HC]
            ),
            "a grid that is not resident": dict(resident=False),
        }
        for name, case in cases.items():
            with self.subTest(name):
                rows = case.get("rows", 64)
                w, r, b, res, tot = self.args(rows)
                w = case.get("writer", w)
                r = case.get("reader", r)
                gate = case.get("inject", r.block_inject_weight.weight)
                part = case.get("partials", tot)
                with mock.patch.object(
                    sys.modules[f"{_OPS}.gemm.hc_boundary_ks"],
                    "supported",
                    return_value=case.get("resident", True),
                ):
                    reason = hc_wide.boundary_reason(
                        w, b, res, part, r, gate, limits=case.get("limits", DEFAULT)
                    )
                self.assertTrue(reason, name)
        self.assertEqual(self.spies.named("hc_boundary_ks"), [])


class TestBuild(_Case):
    def build(self, limits, capability=(10, 0)):
        with (
            mock.patch.object(torch.cuda, "is_available", return_value=True),
            mock.patch.object(
                torch.cuda, "get_device_capability", return_value=capability
            ),
            mock.patch.object(hc_wide, "gate_stream", self.spies("gate_stream")),
        ):
            return hc_wide.build(limits=limits)

    def test_needs_an_sm10x_gpu(self):
        self.assertIn("SM10x", self.build(DEFAULT, capability=(9, 0)))
        with mock.patch.object(torch.cuda, "is_available", return_value=False):
            self.assertIn("SM10x", hc_wide.build())

    def test_builds_the_enabled_kernels(self):
        self.assertEqual(self.build(HIGH_CONCURRENCY), "")
        self.assertEqual(self.spies.named("ext.warmup"), [((HC, HS, BF16), {})])
        self.assertEqual(len(self.spies.named("ks.warmup")), 1)
        self.assertEqual(self.spies.named("decode._jit_module"), [])
        self.spies.calls.clear()
        self.assertEqual(self.build(MID_CONCURRENCY), "")
        self.assertEqual(len(self.spies.named("ext.warmup")), 1)
        self.assertEqual(self.spies.named("ks.warmup"), [])
        self.assertEqual(len(self.spies.named("decode._jit_module")), 1)
        [((device, priority), _)] = self.spies.named("gate_stream")
        self.assertEqual((device.type, priority), ("cuda", None))
        self.spies.calls.clear()
        self.assertEqual(self.build(Limits()), "")
        self.assertEqual(self.spies.calls, [])

    def test_reports_a_failed_kernel(self):
        ext = sys.modules[f"{_OPS}.elementwise.hc_combine_extend"]
        with mock.patch.object(ext, "warmup", side_effect=RuntimeError("nvcc")):
            reason = self.build(HIGH_CONCURRENCY)
        self.assertEqual(reason, "hc_combine_extend: RuntimeError: nvcc")
        self.assertEqual(len(self.spies.named("ks.warmup")), 1)

    def test_wide_mix_needs_the_split_k_max_m(self):
        gemm = sys.modules[
            "sglang.kernels.ops.gemm.dense_bf16_gemm_sm100_splitk_epilogue"
        ]
        with mock.patch.object(gemm, "validate_tactic", lambda tactic, m, n, k: None):
            reason = self.build(HIGH_CONCURRENCY)
        self.assertIn("hc_mix_wide", reason)
        self.assertIn("sgl-project/sglang#50194", reason)


if __name__ == "__main__":
    unittest.main()
