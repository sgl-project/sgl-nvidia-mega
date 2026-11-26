# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the SGLang project
"""CPU checks for the K3.1 megakernel on Rubin (VR200: SM107, 212 SMs). No GPU, no SGLang import,
no kernel compile.

- ``targets.select`` picks the plans from the architecture: the package's own (``GB300``, no
  overrides) on SM103 and SM100, Rubin's (``sm107.tp8.RUBIN``) on SM107, none elsewhere, where
  the build stops. Every ISA the spec admits has a target.
- Rubin's KDA verify schedules at TP8 (12 heads) and 5 rows a request, and nothing for another
  head count or row count.
- The schedules of the build's target reach the KDA verify kernels through the model-level code
  (``verify_spec`` -> ``verify_kda`` -> ``sm103/k3/kda/verify.py``) and take precedence over that
  file's own choice for the keys they have; on SM103 and SM100 the choice is that file's alone.
- Rubin's MoE plans at TP8: the shared expert's down rows take 12 SMs at 9-12 rows and 8 at
  25-64 on verify steps, and the routed experts copy the plan at 5-7 rows as well as 8. They
  reach the split table (``split_table``) and the experts' compile (``plan_b8``) through
  ``build_resources``; GB300's tables are unchanged on SM103 and SM100.
- ``sm103/`` takes those plans as values: none of its files imports ``targets`` or ``sm107``.
"""

from __future__ import annotations

import ast
import sys
import types
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

from sgl_nvidia_mega.kimi_k31 import targets, verify_binding, verify_contract
from sgl_nvidia_mega.kimi_k31.sm103.k3.common.policy import batch_table
from sgl_nvidia_mega.kimi_k31.sm107 import tp8

K31 = Path(targets.__file__).parent
PACKAGE = "sgl_nvidia_mega.kimi_k31"
DEFAULT = (0, 8, 2, 8)  # sm103/k3/kda/verify.py verify_schedule below 9 rows a request


def _defined(node):
    """The names a top-level statement defines: a def, a class or an assignment."""
    targets = getattr(node, "targets", [getattr(node, "target", None)])
    names = {n.id for t in targets if t for n in ast.walk(t) if isinstance(n, ast.Name)}
    return names | {getattr(node, "name", None)}


def _nodes(path, names):
    """The named top-level definitions of ``path`` (under kimi_k31)."""
    body = ast.parse((K31 / path).read_text()).body
    return [n for n in body if _defined(n) & set(names)]


def _load(path, names, **scope):
    """Run the named top-level definitions of ``path`` without its GPU imports."""
    body = _nodes(path, names)
    exec(compile(ast.Module(body=body, type_ignores=[]), path, "exec"), scope)
    return scope


def _method(path, cls, name):
    """Method ``name`` of top-level class ``cls`` in ``path``, as a function of the package."""
    (node,) = _nodes(path, {cls})
    method = next(n for n in node.body if getattr(n, "name", None) == name)
    scope = {
        "__package__": PACKAGE,
        "KdaVerifyWeights": verify_binding.KdaVerifyWeights,
    }
    exec(compile(ast.Module(body=[method], type_ignores=[]), path, "exec"), scope)
    return scope[name]


def _spec_isas() -> set[str]:
    for node in ast.walk(ast.parse((K31 / "k3/spec.py").read_text())):
        if (
            isinstance(node, ast.Call)
            and getattr(node.func, "id", None) == "Requirements"
        ):
            isas = next(kw.value for kw in node.keywords if kw.arg == "isas")
            return {elt.value for elt in isas.args[0].elts}
    raise AssertionError("no Requirements(...) call in k3/spec.py")


class Targets(unittest.TestCase):
    def test_select_by_architecture(self):
        # Catches: Rubin served with GB300's plans, or GB300 / B300 / B200 with Rubin's.
        self.assertIs(targets.select((10, 3)), targets.GB300)
        self.assertIs(targets.select((10, 0)), targets.GB300)
        self.assertIs(targets.select((10, 7)), tp8.RUBIN)
        self.assertIs(targets.select([10, 7]), tp8.RUBIN)
        # No plans for another architecture: build_resources stops there instead of running
        # GB300's plans on a GPU nobody chose them for.
        for sm in ((9, 0), (10, 1), (12, 0)):
            self.assertIsNone(targets.select(sm))

    def test_rubin_serves_tp8_only(self):
        # The spec turns these into startup refusals, before the weights load.
        (reason,) = targets.tp_reasons((10, 7), 16)
        self.assertIn("serves TP [8], not TP16", reason)
        self.assertEqual(targets.tp_reasons((10, 7), 8), [])
        for tp in (8, 16):
            self.assertEqual(targets.tp_reasons((10, 3), tp), [])
            self.assertEqual(targets.tp_reasons((10, 0), tp), [])
        self.assertEqual(
            targets.tp_reasons((9, 0), 16), []
        )  # no target: not a TP matter

    def test_every_admitted_isa_has_a_target(self):
        # Catches: an ISA added to the spec that then stops in build_resources.
        isas = _spec_isas()
        self.assertEqual(isas, {"sm_100", "sm_103", "sm_107"})
        for isa in isas:
            major, minor = divmod(int(isa.removeprefix("sm_")), 10)
            self.assertIsNotNone(targets.select((major, minor)), isa)

    def test_rubin_kda_verify_schedules(self):
        table = tp8.RUBIN.kda_verify_schedules(12)
        self.assertEqual(
            dict(table),
            {
                (4, 5): (0, 8, 2, 4),
                (5, 5): (0, 8, 2, 4),
                (6, 5): (0, 8, 2, 4),
                (7, 5): (0, 8, 2, 4),
                (8, 5): (0, 8, 3, 4),
            },
        )
        # 1 to 3 requests, and every row count but 5, keep the package's schedule
        self.assertFalse({r for r, _ in table} & {1, 2, 3})
        self.assertEqual({q for _, q in table}, {5})
        # Catches: the schedule that faults on SM107 from 4 requests at Q5.
        self.assertNotIn(DEFAULT, table.values())
        # TP16 (6 heads) has no Rubin plans
        self.assertEqual(dict(tp8.RUBIN.kda_verify_schedules(6)), {})


class KdaVerifySchedules(unittest.TestCase):
    """The KDA verify kernels a TP8 build compiles, by the schedule each (requests, rows a
    request) is handed: ``BuiltK31Verify._prepare_front_kda`` over the real adapter and the
    real front provider, with the compile (``executable``) recorded instead of run."""

    HEADS, LAYER = 12, 1

    def _built(self, capability, queries, heads=HEADS):
        compiled = {}

        def executable(specs, heads, requests, queries, mode, has_bias, device, sched):
            compiled[requests, queries] = sched
            return object()

        front = _load(
            "sm103/k3/kda/verify.py",
            {
                "SM103_TP8_HEADS",
                "SM103_TP8_VERIFY",
                "verify_schedule",
                "PreparedKdaVerify",
            },
            torch=torch,
            cutlass=types.SimpleNamespace(Float32=float),
            decode_conv_weight=lambda weight: weight,
            layout=lambda tensor: (tuple(tensor.shape), tensor.dtype),
            executable=executable,
        )
        adapter = types.ModuleType(f"{PACKAGE}.verify_kda")
        adapter.PreparedKdaVerify = _load(
            "verify_kda.py",
            {"PreparedKdaVerify"},
            FrontKdaVerify=front["PreparedKdaVerify"],
        )["PreparedKdaVerify"]

        d = heads * 128
        pitch = 4 * d + 128 + heads + (-(4 * d + 128 + heads) % 16)
        bf16, f32 = torch.bfloat16, torch.float32
        e = lambda shape, dtype=bf16: torch.empty(shape, dtype=dtype, device="meta")
        slots, rows, ring = 19, 17, 16
        pools = types.SimpleNamespace(
            conv=e((slots, 3 * d, 3)),
            temporal=e((slots, heads, 128, 128), f32),
            rawv=e((slots, heads, ring, 128)),
            rawk=e((slots, heads, ring, 128)),
            log_g=e((slots, heads, ring, 128), f32),
            beta=e((slots, heads, ring), f32),
            conv_windows=e((rows, queries, 3, 3 * d)).transpose(-1, -2),
        )
        core = types.SimpleNamespace(
            weights=(
                e((3, 4, d), f32),
                e((d,), f32),
                e((d, 128)),
                e((heads,), f32),
                e((d,), f32),
                e((128,), f32),
            ),
            has_bias=False,
            device=types.SimpleNamespace(index=0),
        )
        layer = types.SimpleNamespace(
            layer_idx=self.LAYER,
            self_attn=types.SimpleNamespace(
                attn=types.SimpleNamespace(lower_bound=-5.0),
                o_norm=types.SimpleNamespace(eps=1e-6),
            ),
        )
        built = types.SimpleNamespace(
            keys=verify_contract.verify_keys(queries, range(1, 9)),
            resources=types.SimpleNamespace(
                target=targets.select(capability),
                scratch={"kda.projected": e((64, pitch))},
                layers=(layer,),
                attention={self.LAYER: types.SimpleNamespace(core=core)},
            ),
            executor=types.SimpleNamespace(kda_pools={self.LAYER: pools}, kda_cores={}),
        )
        prepare = _method("verify_spec.py", "BuiltK31Verify", "_prepare_front_kda")
        with (
            patch.dict(sys.modules, {adapter.__name__: adapter}),
            patch.object(torch.cuda, "get_device_capability", return_value=capability),
        ):
            prepare(built)
        self.assertIn(self.LAYER, built.executor.kda_cores)
        self.assertEqual(set(compiled), {(r, queries) for r in range(1, 9)})
        return {r: compiled[r, queries] for r in range(1, 9)}

    def test_rubin_q5_runs_rubin_schedules_from_four_requests(self):
        self.assertEqual(
            self._built((10, 7), 5),
            {
                1: DEFAULT,
                2: DEFAULT,
                3: DEFAULT,
                4: (0, 8, 2, 4),
                5: (0, 8, 2, 4),
                6: (0, 8, 2, 4),
                7: (0, 8, 2, 4),
                8: (0, 8, 3, 4),
            },
        )

    def test_rubin_has_no_plans_for_q8_or_tp16(self):
        # Neither is a Rubin plan and GB300's TP8 table is GB300's: the package default.
        self.assertEqual(set(self._built((10, 7), 8).values()), {DEFAULT})
        self.assertEqual(set(self._built((10, 7), 5, heads=6).values()), {DEFAULT})

    def test_gb300_and_gb200_keep_the_schedules_of_sm103(self):
        # Catches: a Rubin schedule, or any override, reaching SM103 / SM100.
        sm103 = _load("sm103/k3/kda/verify.py", {"SM103_TP8_VERIFY"}, torch=torch)[
            "SM103_TP8_VERIFY"
        ]
        for queries in (5, 8):
            want = {r: sm103.get((r, queries), DEFAULT) for r in range(1, 9)}
            self.assertEqual(self._built((10, 3), queries), want)
            self.assertEqual(set(self._built((10, 0), queries).values()), {DEFAULT})
        self.assertEqual(self._built((10, 3), 5)[3], (0, 8, 2, 4))

    def test_the_callers_schedule_wins_for_its_keys_only(self):
        compiled = {}

        def executable(*args):
            compiled[args[2], args[3]] = args[-1]

        front = _load(
            "sm103/k3/kda/verify.py",
            {
                "SM103_TP8_HEADS",
                "SM103_TP8_VERIFY",
                "verify_schedule",
                "PreparedKdaVerify",
            },
            torch=torch,
            cutlass=types.SimpleNamespace(Float32=float),
            decode_conv_weight=lambda weight: weight,
            layout=lambda tensor: None,
            executable=executable,
        )["PreparedKdaVerify"]
        heads, d = 12, 12 * 128
        e = lambda shape, dtype=torch.bfloat16: torch.empty(
            shape, dtype=dtype, device="meta"
        )
        f32 = torch.float32
        provider = front(
            conv_weight=e((3, d, 4), f32),
            f_b_weight=e((d, 128)),
            a_log=e((heads,), f32),
            dt_bias=e((d,), f32),
            norm_weight=e((128,), f32),
            lower_bound=-5.0,
            eps=1e-6,
            policies={(3, 5): (6288, 0), (2, 5): (6288, 0)},
            conv_dim_first=True,
            # over SM103's own table entry at (3, 5); (2, 5) is not the caller's
            schedules={(3, 5): (0, 8, 1, 8)},
        )
        with patch.object(torch.cuda, "get_device_capability", return_value=(10, 3)):
            provider.prepare_caches(
                (e((9, 3 * d, 3)), e((9, heads, 128, 128), f32)),
                conv_window=e((7, 5, 3, 3 * d)),
                rings=(
                    e((9, heads, 16, 128)),
                    e((9, heads, 16, 128)),
                    e((9, heads, 16, 128), f32),
                    e((9, heads, 16), f32),
                ),
            )
        self.assertEqual(compiled, {(3, 5): (0, 8, 1, 8), (2, 5): DEFAULT})


class MoePlans(unittest.TestCase):
    """Rubin's plans for the shared expert's down rows and the experts' plan copy."""

    FRONT, EXPERTS = "sm103/k3/moe/front.py", "sm103/k3/moe/experts.py"

    def _split(self, capability, tp, verify):
        """Down SMs by batch at 9-64 rows, as ``SharedFront`` reads them."""
        table = _load(
            self.FRONT, {"SPLIT", "SPLITS", "split_table"}, batch_table=batch_table
        )["split_table"]
        overrides = targets.select(capability).down_splits(tp, verify)
        split = table(tp, verify, overrides)
        return {b: split.get(b, 0) for b in range(9, 65)}

    def test_rubin_plan_values(self):
        rubin = tp8.RUBIN
        self.assertEqual(rubin.down_splits(8, True), "9-12:12,25-64:8")
        self.assertEqual(rubin.down_splits(8, False), "")  # q = 1 steps: GB300's
        self.assertEqual(rubin.plan_copy(8), frozenset({5, 6, 7}))
        # TP16 has no Rubin plans
        self.assertEqual(rubin.down_splits(16, True), "")
        self.assertEqual(rubin.plan_copy(16), frozenset())

    def test_rubin_verify_down_rows_take_more_sms(self):
        want = {b: 12 for b in range(9, 13)}
        want |= {b: 6 for b in range(13, 17)} | {b: 4 for b in range(17, 25)}
        want |= {b: 8 for b in range(25, 65)}
        self.assertEqual(self._split((10, 7), 8, True), want)

    def test_other_split_tables_are_gb300s(self):
        # Catches: Rubin's SMs reaching q = 1 steps, TP16, SM103 or SM100.
        gb300 = {b: 6 for b in range(9, 17)} | {b: 4 for b in range(17, 33)}
        gb300 |= {b: 3 for b in range(33, 65)}
        self.assertEqual(self._split((10, 3), 8, True), gb300)
        self.assertEqual(self._split((10, 0), 8, True), gb300)
        for tp, verify in ((8, False), (16, False), (16, True)):
            want = self._split((10, 3), tp, verify)
            self.assertEqual(self._split((10, 7), tp, verify), want)
            self.assertEqual(self._split((10, 0), tp, verify), want)

    def test_the_plan_copy_rows(self):
        plan_b8 = _load(self.EXPERTS, {"plan_b8"})["plan_b8"]

        def rows(capability, tp):
            extra = targets.select(capability).plan_copy(tp)
            return [b for b in range(1, 65) if plan_b8(tp, b, extra)]

        self.assertEqual(rows((10, 7), 8), [5, 6, 7, 8])
        self.assertEqual(rows((10, 3), 8), [8])
        self.assertEqual(rows((10, 0), 8), [8])
        for capability in ((10, 7), (10, 3), (10, 0)):
            self.assertEqual(rows(capability, 16), [])
        # without the caller's rows: 8 only
        self.assertEqual([b for b in range(1, 65) if plan_b8(8, b)], [8])
        # the copy runs on the capacity-8 engine only (the ExpertsKernel assert)
        self.assertLessEqual(set(tp8.PLAN_COPY_ROWS), {5, 6, 7})


if __name__ == "__main__":
    unittest.main()
