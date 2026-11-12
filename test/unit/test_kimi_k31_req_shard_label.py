# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""CPU tests of the request-sharded selection's graph label (dsa/req_shard.py req_shard_label) and its wiring in
dsa/verify.py (tsi_graph_label passes K31MK_DSA_REQ_SHARD_SUM_CONTEXT). Expected values are hand-derived from the rule:
  off (sum_context 0): requests >= min_requests AND min(context) >= min_context  (the served rule, unchanged);
  on  (sum_context > 0): requests >= min_requests AND sum(context) >= sum_context.
The module files are loaded directly (no package import), so the tests run without a GPU. K31_TEST_REQ_SHARD_PY /
K31_TEST_VERIFY_PY point them at other copies (mutation runs)."""

import ast
import importlib.util
import os
import random
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2] / "sgl_nvidia_mega" / "kimi_k31" / "dsa"
REQ_SHARD_PY = Path(os.environ.get("K31_TEST_REQ_SHARD_PY", ROOT / "req_shard.py"))
VERIFY_PY = Path(os.environ.get("K31_TEST_VERIFY_PY", ROOT / "verify.py"))


def _label():
    spec = importlib.util.spec_from_file_location("k31_req_shard_under_test", REQ_SHARD_PY)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.req_shard_label


def _t(xs):
    return torch.tensor(xs, dtype=torch.int32)


def test_off_is_the_shortest_request_rule():
    label = _label()
    rng = random.Random(1106)
    for _ in range(2000):  # independent re-statement of the served rule on random steps
        n = rng.randint(1, 16)
        lens = [rng.choice([1000, 2049, 65535, 65536, 65537, 200000, 600000]) for _ in range(n)]
        want = n >= 4 and min(lens) >= 65536
        assert label(_t(lens), 4, 65536) is want, lens
        assert label(_t(lens), 4, 65536, 0) is want, lens


def test_sum_rule_engages_a_step_with_one_short_request():
    label = _label()
    lens = [70000, 70000, 70000, 1000]  # min 1,000 < 65,536 (off: False); sum 211,000
    assert label(_t(lens), 4, 65536) is False
    assert label(_t(lens), 4, 65536, 200000) is True
    assert label(_t(lens), 4, 65536, 211001) is False  # max 70,000 alone would never reach these thresholds


def test_sum_rule_boundary_is_inclusive():
    label = _label()
    lens = [100000, 100000, 100000, 100000, 100000, 100000, 100000, 100000, 100000]  # sum 900,000
    assert label(_t(lens), 4, 65536, 900000) is True
    assert label(_t(lens), 4, 65536, 900001) is False


def test_sum_rule_keeps_the_request_floor():
    label = _label()
    assert label(_t([500000, 500000, 500000]), 4, 65536, 1000) is False  # 3 requests < 4, any sum
    assert label(_t([500000] * 4), 4, 65536, 1000) is True
    assert label(_t([]), 4, 65536, 1000) is False
    assert label(None, 4, 65536, 1000) is False


def test_verify_passes_the_sum_knob_to_the_label():
    tree = ast.parse(VERIFY_PY.read_text())
    env_default = None
    for node in tree.body:  # module-level REQ_SHARD_SUM_CONTEXT = int(os.environ.get("K31MK_DSA_REQ_SHARD_SUM_CONTEXT", "0"))
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "REQ_SHARD_SUM_CONTEXT" for t in node.targets):
            call = node.value.args[0]
            assert call.args[0].value == "K31MK_DSA_REQ_SHARD_SUM_CONTEXT"
            env_default = call.args[1].value
    assert env_default == "0", "the knob must default to off"
    fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "tsi_graph_label")
    calls = [n for n in ast.walk(fn) if isinstance(n, ast.Call) and getattr(n.func, "id", None) == "req_shard_label"]
    assert len(calls) == 1
    args = [a.id for a in calls[0].args if isinstance(a, ast.Name)]
    assert args == ["seq_lens", "REQ_SHARD_MIN_REQ", "REQ_SHARD_MIN_CONTEXT", "REQ_SHARD_SUM_CONTEXT"], args
