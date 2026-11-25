# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""CPU checks for the K3.1 megakernel's GB200 (SM100) admission. No GPU, no SGLang import."""

from __future__ import annotations

import ast
import os
import unittest
from pathlib import Path
from unittest import mock

from sgl_nvidia_mega.kimi_k31.k3 import shape_policy

SPEC_PY = Path(shape_policy.__file__).with_name("spec.py")


def _spec_requirements() -> dict[str, ast.expr]:
    """The keyword arguments of the Requirements(...) call in k3/spec.py, read without importing SGLang."""
    for node in ast.walk(ast.parse(SPEC_PY.read_text())):
        if (
            isinstance(node, ast.Call)
            and getattr(node.func, "id", None) == "Requirements"
        ):
            return {kw.arg: kw.value for kw in node.keywords}
    raise AssertionError("no Requirements(...) call in k3/spec.py")


class Sm100Admission(unittest.TestCase):
    def test_sm148_lowers_the_sm_floor_by_name(self):
        # Catches: lowering resident_blocks for every ISA instead of behind the opt-in.
        with mock.patch.dict(os.environ, {shape_policy.SLOTS_ENV: ""}):
            self.assertEqual(shape_policy.resident_blocks(), 152)
        with mock.patch.dict(os.environ, {shape_policy.SLOTS_ENV: "sm148"}):
            self.assertEqual(shape_policy.resident_blocks(), 148)
        with mock.patch.dict(
            os.environ, {shape_policy.SLOTS_ENV: "draft, sm148 ,target_lm_head"}
        ):
            self.assertEqual(shape_policy.resident_blocks(), 148)
        with mock.patch.dict(os.environ, {shape_policy.SLOTS_ENV: "sm1480"}):
            self.assertEqual(shape_policy.resident_blocks(), 152)

    def test_spec_admits_sm100_and_reads_the_floor(self):
        # Catches: the spec still refusing sm_100, or hard-coding 152 again.
        req = _spec_requirements()
        isas = {elt.value for elt in req["isas"].args[0].elts}
        self.assertEqual(isas, {"sm_100", "sm_103"})
        self.assertEqual(ast.unparse(req["resident_blocks"]), "resident_blocks()")


if __name__ == "__main__":
    unittest.main()
