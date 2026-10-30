# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the SGLang project
"""Host-only coverage/key tests for the private, still admission-OFF DCP waits."""

import ast
import copy
import math
import unittest
from contextlib import nullcontext
from functools import cache
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import Mock

ROOT = Path(__file__).resolve().parents[2]
MLA = ROOT / "sgl_nvidia_mega/kimi_k3/cutedsl/mla"


def definition(filename, *names):
    node = ast.parse((MLA / filename).read_text())
    for name in names:
        node = next(item for item in node.body if getattr(item, "name", None) == name)
    return copy.deepcopy(node)


def execute(nodes, namespace):
    exec(
        compile(ast.Module(body=nodes, type_ignores=[]), "<host audit>", "exec"),
        namespace,
    )


def constructor(filename, name):
    node = definition(filename, name)
    node.body = [definition(filename, name, "__init__")]
    namespace = {"TILED_FROM": {12: 4, 6: 6}}
    execute([node], namespace)
    return namespace[name]


def calls(node, name):
    return [
        item
        for item in ast.walk(node)
        if isinstance(item, ast.Call)
        and getattr(item.func, "id", getattr(item.func, "attr", None)) == name
    ]


class TestDcpDeadline(unittest.TestCase):
    def test_exact_three_remote_polls_have_lazy_start_and_retry_check(self):
        for filename, names, count in (
            ("query.py", ("MlaQuery", "gathered"), 1),
            ("dcp.py", ("DcpValue", "merge"), 2),
        ):
            node = definition(filename, *names)
            loops = [item for item in ast.walk(node) if isinstance(item, ast.While)]
            self.assertEqual(len(loops), count)
            self.assertEqual(len(calls(node, "wait_deadline_start")), count)
            self.assertEqual(len(calls(node, "wait_deadline_check")), count)
            for loop in loops:
                self.assertEqual(len(calls(loop.body[0], "wait_deadline_check")), 1)
                self.assertIn("self.deadline_ns", ast.unparse(loop.body[0]))
                # The identical pending predicate guards timer startup, keeping
                # already-published reads on the existing timer-free fast path.
                starts = [
                    item
                    for item in ast.walk(node)
                    if isinstance(item, ast.If)
                    and ast.dump(item.test) == ast.dump(loop.test)
                    and len(calls(item, "wait_deadline_start")) == 1
                ]
                self.assertEqual(len(starts), 1)

    def test_deadline_is_positive_integer_and_dcp_off_stays_zero(self):
        query = constructor("query.py", "MlaQuery")
        value = constructor("dcp.py", "DcpValue")
        self.assertEqual(query(12, 1).deadline_ns, 0)
        for invalid in (None, True, False, 0, -1, 1.5, math.nan, math.inf, 1 << 63):
            with self.subTest(invalid=invalid):
                with self.assertRaisesRegex(ValueError, "positive int64"):
                    query(12, 1, dcp=8, deadline_ns=invalid)
                with self.assertRaisesRegex(ValueError, "positive int64"):
                    value(12, 8, 1, deadline_ns=invalid)
        with self.assertRaisesRegex(ValueError, "only supported with DCP"):
            query(12, 1, deadline_ns=10)
        for valid in (1, 50_000_000, (1 << 63) - 1):
            self.assertEqual(query(12, 1, dcp=8, deadline_ns=valid).deadline_ns, valid)
            self.assertEqual(value(12, 8, 1, deadline_ns=valid).deadline_ns, valid)

    def compiler_namespace(self):
        compiled = Mock(side_effect=lambda kernel, *args: NS(kernel=kernel))
        namespace = {
            "cache": cache,
            "fake": lambda *args: object(),
            "make_fake_stream": object,
            "Int32": int,
            "cutlass": NS(Float32=float, Int32=int, Int64=int),
            "torch": NS(
                float32="float32",
                int32="int32",
                int64="int64",
                uint8="uint8",
                accelerator=NS(device_index=lambda device: nullcontext()),
            ),
            "cute": NS(compile=compiled),
            "MlaQuery": constructor("query.py", "MlaQuery"),
            "DcpValue": constructor("dcp.py", "DcpValue"),
            "MlaValue": lambda *args: NS(value=True),
            "prepare_shared_carveout": lambda code, *args: code,
        }
        execute([definition("dcp.py", "compile_value")], namespace)
        execute([definition("prepared.py", "compile_core")], namespace)
        return namespace, compiled

    def test_actual_cached_compilers_distinguish_dcp_deadlines(self):
        namespace, compiled = self.compiler_namespace()
        core, value = namespace["compile_core"], namespace["compile_value"]
        specs = (((1,), (1,), "bf16"),) * 8
        args = (1, 12, 3648, specs, 0)
        off = core(*args)
        self.assertIs(core(*args), off)
        self.assertEqual(off[0].kernel.deadline_ns, 0)
        self.assertEqual(value.cache_info().misses, 0)
        first = core(*args, dcp=8, deadline_ns=50)
        self.assertIs(core(*args, dcp=8, deadline_ns=50), first)
        second = core(*args, dcp=8, deadline_ns=100)
        self.assertIsNot(first, second)
        self.assertEqual(first[0].kernel.deadline_ns, 50)
        self.assertEqual(first[1].kernel.deadline_ns, 50)
        self.assertEqual(second[0].kernel.deadline_ns, 100)
        self.assertEqual(second[1].kernel.deadline_ns, 100)
        self.assertEqual(value.cache_info().misses, 2)
        self.assertEqual(compiled.call_count, 6)
        with self.assertRaisesRegex(ValueError, "positive int64"):
            core(*args, dcp=8)
        with self.assertRaisesRegex(ValueError, "only supported with DCP"):
            core(*args, deadline_ns=50)

    def test_preparation_forwards_deadline_only_for_dcp(self):
        method = definition("prepared.py", "PreparedMla", "prepare_cache")
        compile_core = Mock(return_value=("query-code", "value-code"))
        namespace = {
            "layout": lambda tensor: ((), (), "bf16"),
            "to_cute": lambda tensor: tensor,
            "compile_core": compile_core,
            "torch": NS(
                uint8="uint8",
                cuda=NS(
                    get_device_properties=lambda device: NS(multi_processor_count=148)
                ),
            ),
        }
        execute([method], namespace)
        kv = NS(view=lambda dtype: object(), is_cuda=False)
        for dcp in (None, NS(size=8)):
            owner = NS(
                weights=(object(),) * 7,
                cores={(1, 1): "tokenspeed"},
                device=NS(index=0),
                dcp=dcp,
                pitches={1: 3648},
                query_scratch=[object()],
                latent=[object()],
                heads=12,
                deadline_ns=50 if dcp else 0,
                kernels={},
                prepare_dcp=Mock(),
            )
            namespace["prepare_cache"](owner, kv)
            self.assertEqual(
                compile_core.call_args.kwargs, {"deadline_ns": 50} if dcp else {}
            )
            self.assertEqual(compile_core.call_args.args[5], 8 if dcp else 0)
        self.assertEqual(compile_core.call_count, 2)

    def test_prepared_constructor_validates_before_allocating(self):
        node = definition("prepared.py", "PreparedMla", "__init__")
        # Invalid deadlines must raise before the attention object is accessed.
        namespace = {}
        execute([node], namespace)
        for dcp, deadline in ((object(), 0), (object(), True), (None, 1)):
            with self.assertRaises(ValueError):
                namespace["__init__"](
                    NS(), None, None, None, dcp=dcp, deadline_ns=deadline
                )


if __name__ == "__main__":
    unittest.main()
