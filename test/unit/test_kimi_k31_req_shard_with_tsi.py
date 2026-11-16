# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""CPU tests of K31MK_DSA_REQ_SHARD_WITH_TSI (default off): TSI at rows <= 32 and request-sharded selection (R1) above, in one
third verify graph. Covers dsa/verify.py's knob block + tsi_graph_label (executed from the file's own source, without importing
cutlass), dsa/req_shard.py SplitSelection, and the BuiltSparseVerify wiring (AST). Expected values are restated from the rules:
  TSI rule  : rows in (8, 16) and min(context) >= K31MK_TSI_MIN_CONTEXT, or under K31MK_TSI_WIDE with >= 2 requests rows in
              (16, 24, 32) and sum(context) >= K31MK_TSI_SUM_CONTEXT;
  R1 rule   : requests >= K31MK_DSA_REQ_SHARD_MIN_REQ and min(context) >= K31MK_DSA_REQ_SHARD_MIN_CONTEXT;
  WITH_TSI  : rows <= 32 -> the TSI rule only; rows > 32 -> the R1 rule only. Off: R1 alone (if on) else TSI alone, and
              K31MK_TSI + K31MK_DSA_REQ_SHARD together is an error.
K31_TEST_REQ_SHARD_PY / K31_TEST_VERIFY_PY point the tests at other copies (mutation runs)."""

import ast
import importlib.util
import os
import random
import sys
import types
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2] / "sgl_nvidia_mega" / "kimi_k31" / "dsa"
REQ_SHARD_PY = Path(os.environ.get("K31_TEST_REQ_SHARD_PY", ROOT / "req_shard.py"))
VERIFY_PY = Path(os.environ.get("K31_TEST_VERIFY_PY", ROOT / "verify.py"))
PKG = "k31dsa_with_tsi_under_test"
BASE_ENV = {"K31MK_TSI_MIN_CONTEXT": "262144", "K31MK_TSI_WIDE": "1", "K31MK_TSI_SUM_CONTEXT": "393216",
            "K31MK_DSA_REQ_SHARD_MIN_REQ": "4", "K31MK_DSA_REQ_SHARD_MIN_CONTEXT": "65536"}
BOTH = {**BASE_ENV, "K31MK_TSI": "1", "K31MK_DSA_REQ_SHARD": "1", "K31MK_DSA_REQ_SHARD_WITH_TSI": "1"}


def _package():
    """A package whose path holds REQ_SHARD_PY, so verify.py's 'from .req_shard import ...' loads the copy under test."""
    for key in [k for k in sys.modules if k == PKG or k.startswith(PKG + ".")]:
        del sys.modules[key]
    pkg = types.ModuleType(PKG)
    pkg.__path__ = []
    sys.modules[PKG] = pkg
    spec = importlib.util.spec_from_file_location(PKG + ".req_shard", REQ_SHARD_PY)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[PKG + ".req_shard"] = mod
    spec.loader.exec_module(mod)
    return mod


def _verify_knobs(env):
    """Executes verify.py's top-level statements from 'TSI_KERNEL = ...' through 'def tsi_graph_label' with os.environ = env."""
    _package()
    tree = ast.parse(VERIFY_PY.read_text())
    names = [i for i, n in enumerate(tree.body) if isinstance(n, ast.Assign) and any(getattr(t, "id", None) == "TSI_KERNEL" for t in n.targets)]
    ends = [i for i, n in enumerate(tree.body) if isinstance(n, ast.FunctionDef) and n.name == "tsi_graph_label"]
    assert len(names) == 1 and len(ends) == 1 and names[0] < ends[0]
    body = ast.Module(body=tree.body[names[0]: ends[0] + 1], type_ignores=[])
    # verify.py imports IDX_FP4 (dsa/idx_fp4_knob.py, prec-l2mk) above the knob block, which reads it; the switch-on branch imports
    # dsa/idx_fp4 and is not exercised here
    g = {"os": types.SimpleNamespace(environ=dict(env)), "SHORT_QUERY_WIDTHS": (), "__name__": PKG + ".verify", "__package__": PKG,
         "IDX_FP4": env.get("K31MK_DSA_IDX_FP4", "0") == "1", "IDX_FP4_FP8Q": False, "IDX_FP4_FP4Q": False}
    exec(compile(body, str(VERIFY_PY), "exec"), g)
    return g


def _t(xs):
    return torch.tensor(xs, dtype=torch.int32)


def _tsi_rule(lens, q=8):
    n, rows = len(lens), len(lens) * q
    if n == 0:
        return False
    if n >= 2:
        return rows in (16, 24, 32) and sum(lens) >= 393216
    return rows in (8, 16) and min(lens) >= 262144


def _rs_rule(lens):
    return len(lens) >= 4 and len(lens) > 0 and min(lens) >= 65536


def test_default_keeps_the_exclusivity():
    raised = False
    try:
        _verify_knobs({**BASE_ENV, "K31MK_TSI": "1", "K31MK_DSA_REQ_SHARD": "1"})
    except ValueError:
        raised = True
    assert raised, "K31MK_TSI + K31MK_DSA_REQ_SHARD without WITH_TSI must stay an error"


def test_with_tsi_needs_both_selections():
    for env in ({**BASE_ENV, "K31MK_DSA_REQ_SHARD_WITH_TSI": "1", "K31MK_TSI": "1"},
                {**BASE_ENV, "K31MK_DSA_REQ_SHARD_WITH_TSI": "1", "K31MK_DSA_REQ_SHARD": "1"}):
        raised = False
        try:
            _verify_knobs(env)
        except ValueError:
            raised = True
        assert raised, env


def test_with_tsi_routes_the_label_by_rows():
    label = _verify_knobs(BOTH)["tsi_graph_label"]
    assert label(_t([70000] * 4), 8) is False   # rows 32: TSI only (sum 280,000 < 393,216), although R1's rule holds
    assert label(_t([200000] * 4), 8) is True   # rows 32: TSI wide rule (sum 800,000)
    assert label(_t([70000] * 8), 8) is True    # rows 64: R1's rule (TSI has no 64-row form)
    assert label(_t([70000] * 7 + [1000]), 8) is False  # rows 64: R1's shortest-request rule fails
    assert label(_t([300000]), 8) is True       # rows 8: TSI rule
    assert label(_t([100000, 100000]), 8) is False  # rows 16: TSI wide sum 200,000 < 393,216
    assert label(None, 8) is False
    assert label(_t([]), 8) is False
    rng = random.Random(1107)
    for _ in range(3000):
        n = rng.randint(1, 16)
        lens = [rng.choice([1000, 65535, 65536, 70000, 131072, 200000, 262143, 262144, 600000]) for _ in range(n)]
        want = _tsi_rule(lens) if 8 * n <= 32 else _rs_rule(lens)
        assert label(_t(lens), 8) is want, lens


def test_off_leaves_each_selection_alone():
    rs_only = _verify_knobs({**BASE_ENV, "K31MK_DSA_REQ_SHARD": "1"})["tsi_graph_label"]
    tsi_only = _verify_knobs({**BASE_ENV, "K31MK_TSI": "1"})["tsi_graph_label"]
    assert rs_only(_t([70000] * 4), 8) is True   # R1 alone still acts at rows 32
    assert tsi_only(_t([70000] * 8), 8) is False
    rng = random.Random(1108)
    for _ in range(2000):
        n = rng.randint(1, 16)
        lens = [rng.choice([1000, 65536, 70000, 200000, 262144, 600000]) for _ in range(n)]
        assert rs_only(_t(lens), 8) is _rs_rule(lens), lens
        assert tsi_only(_t(lens), 8) is _tsi_rule(lens), lens


class _State:
    def __init__(self, name, log):
        self.name, self.log = name, log

    def prepare(self, table, contexts, causal):
        self.log.append((self.name, "prepare", int(causal.shape[0])))

    def select(self, ordinal, query, gates, cache, out):
        self.log.append((self.name, "select", ordinal))
        return (self.name, ordinal, query, gates, cache, out)


def test_split_selection_picks_by_rows_per_forward():
    rs_mod = _package()
    log = []
    s = rs_mod.SplitSelection(_State("tsi", log), _State("rs", log), 32)
    for rows, want in ((8, "tsi"), (32, "tsi"), (40, "rs"), (128, "rs"), (16, "tsi")):
        log.clear()
        s.prepare("tab", "ctx", torch.zeros(rows))
        outs = [s.select(o, "q", "g", "c", "o") for o in range(24)]
        assert log[0] == (want, "prepare", rows)
        assert all(entry[0] == want for entry in log), log
        assert outs[5] == (want, 5, "q", "g", "c", "o")


def test_built_sparse_verify_wires_both_states():
    tree = ast.parse(VERIFY_PY.read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "BuiltSparseVerify")
    init = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "__init__")
    first_if = next(n for n in init.body if isinstance(n, ast.If) and "TSI_KERNEL" in ast.unparse(n.test))
    assert ast.unparse(first_if.test) == "TSI_KERNEL and REQ_SHARD", ast.unparse(first_if.test)
    calls = [n for n in ast.walk(ast.Module(body=first_if.body, type_ignores=[]))
             if isinstance(n, ast.Call) and getattr(n.func, "id", None) == "SplitSelection"]
    assert len(calls) == 1
    assert [ast.unparse(a) for a in calls[0].args] == ["tsi", "rs", "TSI_RS_SPLIT_ROWS"]
    built = [getattr(n.func, "id", None) for n in ast.walk(ast.Module(body=first_if.body, type_ignores=[])) if isinstance(n, ast.Call)]
    assert "_build_tsi" in built and "_build_req_shard" in built
