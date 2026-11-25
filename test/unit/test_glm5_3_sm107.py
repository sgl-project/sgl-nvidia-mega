# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import ast
import hashlib
import os
import random
import shutil
import struct
import subprocess
import sys
import types
from pathlib import Path

import pytest

PKG = Path(__file__).resolve().parents[2] / "sgl_nvidia_mega" / "glm5_3" / "sm107"


def test_the_package_imports_without_sglang_torch_or_cute_dsl():
    code = (
        "import sys, sgl_nvidia_mega.glm5_3.sm107\n"
        "bad = sorted(m for m in sys.modules if m.split('.')[0] in ('sglang', 'torch', 'cutlass'))\n"
        "assert not bad, bad\n"
    )
    env = dict(os.environ, PYTHONPATH=str(PKG.parents[2]))
    subprocess.run([sys.executable, "-c", code], check=True, env=env)


@pytest.fixture
def mk():
    import sgl_nvidia_mega.glm5_3.sm107 as mk

    return mk


def test_the_flag_is_the_only_switch():
    """--enable-nvidia-megakernel at TP16 is the only way in: glm5_3's specs dispatch
    to sm107; the plugin has no switch of its own and installs no import hooks."""
    sources = [PKG.parents[1] / "plugin.py", *PKG.glob("*.py")]
    for path in sources:
        text = path.read_text()
        assert "GLM5MK" not in text and "meta_path" not in text, path
    spec_src = (PKG.parent / "spec.py").read_text()
    draft_src = (PKG.parent / "draft.py").read_text()
    assert "sm107.config_reasons(cfg, hf_config)" in spec_src
    assert "sm107.build_target(model, hf_config, ctx)" in spec_src
    assert "sm107.build_draft(model, hf_config, ctx)" in draft_src
    assert not (PKG / "root").exists()


def test_the_cubin_directory_is_refused_when_unset_or_incomplete(
    mk, monkeypatch, tmp_path
):
    monkeypatch.delenv(mk.ARTIFACTS_ENV, raising=False)
    (reason,) = mk.artifact_reasons()
    assert mk.ARTIFACTS_ENV in reason and "not set" in reason
    monkeypatch.setenv(mk.ARTIFACTS_ENV, str(tmp_path))
    for part in mk.ARTIFACT_PARTS:
        (tmp_path / part).mkdir()
        for name in mk.ARTIFACT_FILES:
            (tmp_path / part / name).write_bytes(b"x")
    assert mk.artifact_reasons() == []
    (tmp_path / "tail" / "CALL.json").unlink()
    (reason,) = mk.artifact_reasons()
    assert "tail/CALL.json" in reason


class _Mode:
    def __init__(self, name):
        self.name = name

    def is_target_verify(self):
        return self.name == "target_verify"

    def is_idle(self):
        return self.name == "idle"


class _Rows:
    def __init__(self, n):
        self.shape = (n,)


def _batch(bs, mode):
    return types.SimpleNamespace(batch_size=bs, forward_mode=_Mode(mode))


def test_the_built_plans_one_request_only(mk, monkeypatch):
    monkeypatch.setattr(mk, "_step_class", lambda: lambda **k: k)
    target, draft = mk.Built("t", is_draft=False), mk.Built("d", is_draft=True)
    assert target.widths == draft.widths == (mk.TOKENS_PER_REQUEST,)
    assert target.plan(_Rows(8), _batch(1, "target_verify")) == {
        "mode": "target_verify",
        "rows": 8,
    }
    assert target.plan(_Rows(16), _batch(2, "target_verify")) is None
    assert target.plan(_Rows(1), _batch(1, "decode")) is None
    assert draft.plan(_Rows(8), _batch(1, "draft_extend")) == {
        "mode": "decode",
        "rows": 8,
    }
    assert draft.plan(_Rows(1), _batch(1, "decode")) == {"mode": "decode", "rows": 1}
    assert draft.plan(_Rows(2), _batch(2, "decode")) is None
    assert draft.plan(_Rows(0), _batch(1, "idle")) is None
    assert target.forward(None, lambda *a, **k: (a, k), 1, x=2) == ((1,), {"x": 2})


def test_the_runner_binds_before_its_graphs_once(mk, monkeypatch):
    calls = []
    runner = type(
        "ModelRunner", (), {"init_cuda_graphs": lambda self: calls.append("graphs")}
    )
    module = types.ModuleType("sglang.srt.model_executor.model_runner")
    module.ModelRunner = runner
    for name in ("sglang", "sglang.srt", "sglang.srt.model_executor"):
        monkeypatch.setitem(sys.modules, name, types.ModuleType(name))
    monkeypatch.setitem(sys.modules, module.__name__, module)
    fake_target = types.SimpleNamespace(bind_runner=lambda r: calls.append("target"))
    fake_draft = types.SimpleNamespace(bind_runner=lambda r: calls.append("draft"))
    monkeypatch.setitem(sys.modules, "sgl_nvidia_mega.glm5_3.sm107.target", fake_target)
    monkeypatch.setitem(sys.modules, "sgl_nvidia_mega.glm5_3.sm107.draft", fake_draft)
    monkeypatch.setattr(mk, "target", fake_target, raising=False)
    monkeypatch.setattr(mk, "draft", fake_draft, raising=False)
    mk._bind_before_graphs()
    wrapped = runner.init_cuda_graphs
    mk._bind_before_graphs()
    assert runner.init_cuda_graphs is wrapped and runner._glm5_3_sm107 is True
    runner().init_cuda_graphs()
    draft_runner = runner()
    draft_runner.is_draft_worker = True
    draft_runner.init_cuda_graphs()
    assert calls == ["target", "graphs", "draft", "graphs"]


def test_the_context_must_be_one_request_of_eight_tokens(mk):
    mk._check_context(types.SimpleNamespace(tokens_per_request=8, widths=(8, 16)))
    with pytest.raises(ValueError):
        mk._check_context(types.SimpleNamespace(tokens_per_request=4, widths=(4, 8)))


def test_no_binaries_and_no_cpp():
    # __pycache__ (git-ignored) is written by any test that imported the package first
    found = [
        p
        for p in PKG.rglob("*")
        if p.suffix in (".so", ".o", ".cubin", ".cu", ".cc", ".cuh", ".h", ".pyc")
        and "__pycache__" not in p.parts
    ]
    assert not found, found


KERNELS = [("chain_n148sq", "entry"), ("dd", "draft_bind"), ("dd", "d012_bind")]


def _candidate(tmp_path, tree):
    out = tmp_path / "candidate"
    shutil.copytree(
        PKG / "kernels" / tree, out, ignore=shutil.ignore_patterns("__pycache__")
    )
    return out


@pytest.mark.parametrize("tree, entry", KERNELS)
def test_every_kernel_tree_import_resolves(tmp_path, tree, entry):
    out = _candidate(tmp_path, tree)
    trees = {p.stem: ast.parse(p.read_text()) for p in out.glob("*.py")}
    assert entry in trees
    bound = {}
    for m, t in trees.items():
        names = set()
        for n in ast.walk(t):
            if isinstance(n, (ast.FunctionDef, ast.ClassDef)):
                names.add(n.name)
            elif isinstance(n, ast.Assign):
                names |= {
                    x.id
                    for tg in n.targets
                    for x in ast.walk(tg)
                    if isinstance(x, ast.Name)
                }
            elif isinstance(n, (ast.Import, ast.ImportFrom)):
                names |= {(a.asname or a.name).split(".")[0] for a in n.names}
        bound[m] = names
    for m, t in trees.items():
        for n in ast.walk(t):
            if isinstance(n, ast.ImportFrom) and (
                n.level or (n.module or "").startswith("candidate.")
            ):
                mod = (n.module or "").split(".")[-1] if n.module else None
                for a in n.names:
                    assert (
                        (a.name in trees)
                        if mod is None
                        else (mod in trees and a.name in bound[mod])
                    ), (m, mod, a.name)


def _load_order_errors(tree, entry):
    bodies = {p.stem: ast.parse(p.read_text()).body for p in tree.glob("*.py")}
    state, defined, errors = {}, {m: set() for m in bodies}, []

    def run(m):
        state[m] = "loading"
        for n in bodies[m]:
            if isinstance(n, ast.ImportFrom) and (
                n.level or (n.module or "").startswith("candidate")
            ):
                src = (n.module or "").split(".")[-1] if n.module else None
                for a in n.names:
                    target = src or a.name
                    if target in bodies and target not in state:
                        run(target)
                    if src is not None and a.name not in defined.get(src, set()):
                        errors.append(f"{m}: {a.name} not yet defined in {src}")
                    defined[m].add(a.asname or a.name)
            elif isinstance(n, (ast.FunctionDef, ast.ClassDef)):
                defined[m].add(n.name)
            elif isinstance(n, (ast.Assign, ast.AnnAssign)):
                for t in n.targets if isinstance(n, ast.Assign) else [n.target]:
                    defined[m] |= {x.id for x in ast.walk(t) if isinstance(x, ast.Name)}
            elif isinstance(n, (ast.Import, ast.ImportFrom)):
                defined[m] |= {(a.asname or a.name).split(".")[0] for a in n.names}
        state[m] = "done"

    run(entry)
    return errors


@pytest.mark.parametrize("tree, entry", KERNELS)
def test_the_kernel_trees_load_in_import_order(tmp_path, tree, entry):
    assert _load_order_errors(_candidate(tmp_path, tree), entry) == []


@pytest.mark.parametrize("tree, entry", KERNELS)
def test_the_kernel_entry_modules_import_with_cute_dsl(tmp_path, tree, entry):
    pytest.importorskip("cutlass")
    _candidate(tmp_path, tree)
    code = (
        f"import sys; sys.path.insert(0, {str(tmp_path)!r}); import candidate.{entry}"
    )
    subprocess.run([sys.executable, "-c", code], check=True)


def test_the_fc2_lists_follow_the_kernel_work_split():
    from sgl_nvidia_mega.glm5_3.sm107.kernels.chain_n148sq import schedule

    payload = schedule.fc2_lists_payload()
    # pins the device split (schedule constants + moe.precompute_cuts): a split change must change this on purpose
    assert (
        hashlib.sha256(payload).hexdigest()
        == "bca9d1e2e686eb265b5ac608ecf8c9cec09e77899a6e48027639d0e30e442018"
    )
    words = struct.unpack(f"<{3 * schedule.LIST_ROWS}I", payload)
    for experts in range(1, schedule.MAX_EXPERTS + 1):
        rows = range(experts * (experts - 1), experts * (experts + 1))
        assert all(0 < words[3 * r] & 15 <= 8 for r in rows)


try:
    import torch
except ImportError:
    torch = None
needs_torch = pytest.mark.skipif(torch is None, reason="needs torch")


def _div64(v):
    return torch.div(v, 64, rounding_mode="trunc")


def fill_step_all_ref(r2t, req, pos, bt, kvl):
    last = int(pos[7])
    n = min((last >> 6) + 1, bt.numel())
    bt[:n] = _div64(r2t[int(req[0]), : 64 * n : 64]).to(torch.int32)
    kvl[0] = max(last + 1, 8)


def post_verify_ref(
    acc,
    na,
    am,
    cand,
    seq_lens,
    sim_l,
    predict,
    ai,
    al,
    nsl,
    at,
    bonus,
    s0_acc,
    s0_na,
    sel,
):
    p = acc.clamp(min=0).clone()
    n = int(na[0])
    if sim_l > 0:
        n = sim_l
        p[: n - 1] = cand.view(-1)[1:n].to(torch.int32)
        p[n - 1] = am[n - 1]
    idx = torch.arange(8, dtype=torch.int32, device=acc.device)
    a = torch.where(idx < n, idx, torch.full_like(idx, -1))
    predict.copy_(p)
    ai.view(-1).copy_(a)
    at.view(-1).copy_(p[torch.where(a >= 0, a, torch.full_like(a, 7)).long()])
    s0_acc.copy_(p)
    al[0], s0_na[0], nsl[0], bonus[0], sel[0] = (
        n,
        n,
        seq_lens[0] + n,
        p[n - 1],
        min(max(n - 1, 0), 7),
    )


def draft_pages_ref(r2t, req, sl, bt):
    for i in range(8):
        page = (int(sl[0]) + i) >> 6
        if 0 <= page < bt.numel():
            bt[page] = _div64(r2t[int(req[0]), page * 64])


def chain_tree_ref(bonus, drafts, sl, mask, mode, positions, rbuf, draft_out):
    s = int(sl[0])
    for i in range(8):
        base = (s + 8) * i + s if mode == 0 else 8 * i
        mask.view(-1)[base : base + 8] = torch.arange(8, device=mask.device) <= i
    positions.copy_(s + torch.arange(8, dtype=torch.int64, device=positions.device))
    rbuf.view(3, 8).copy_(
        torch.tensor([list(range(8)), [1, 2, 3, 4, 5, 6, 7, -1], [-1] * 8])
    )
    draft_out[0] = bonus.view(-1)[0]
    draft_out[1:8] = drafts.view(-1)[:7]


def _post_verify_expected(acc, na, am, cand, seq_lens, sim_l):
    p = [a if a > 0 else 0 for a in acc]
    al = sim_l if sim_l > 0 else na
    if sim_l > 0:
        for i in range(al - 1):
            p[i] = cand[i + 1]
        p[al - 1] = am[al - 1]
    ai = [i if i < al else -1 for i in range(8)]
    return (
        p,
        ai,
        al,
        seq_lens + al,
        [p[a if a >= 0 else 7] for a in ai],
        p[al - 1],
        min(max(al - 1, 0), 7),
    )


def _pv_case(rng, device="cpu"):
    i32, i64 = (
        dict(dtype=torch.int32, device=device),
        dict(dtype=torch.int64, device=device),
    )
    n, sim = rng.randint(1, 8), rng.choice([0, 0, rng.randint(1, 8)])
    acc = (
        [rng.randint(1, 150000) for _ in range(n - 1)]
        + [rng.randint(-1, 150000)]
        + [rng.choice([0, -1])] * (8 - n)
    )
    am, cand, sl = (
        [rng.randint(0, 150000) for _ in range(8)],
        [rng.randint(0, 150000) for _ in range(8)],
        rng.randint(8, 10**6),
    )
    ins = (
        torch.tensor(acc, **i32),
        torch.tensor([n], **i32),
        torch.tensor(am, **i32),
        torch.tensor(cand, **i64),
        torch.tensor([sl], **i64),
        sim,
    )
    outs = lambda: (
        torch.empty(8, **i32),
        torch.empty((1, 8), **i32),
        torch.empty(1, **i32),
        torch.empty(1, **i64),
        torch.empty((1, 8), **i32),
        torch.empty(1, **i32),
        torch.empty(8, **i32),
        torch.empty(1, **i32),
        torch.empty(1, **i64),
    )
    return (acc, n, am, cand, sl, sim), ins, outs


@needs_torch
def test_the_post_verify_reference_gives_the_expected_values():
    rng = random.Random(0)
    for _ in range(4000):
        raw, ins, outs = _pv_case(rng)
        o = outs()
        post_verify_ref(*ins, *o)
        p, ai, al, nsl, at, bonus, sel = _post_verify_expected(*raw)
        assert (
            o[0].tolist() == p
            and o[1].view(-1).tolist() == ai
            and int(o[2]) == al
            and int(o[3]) == nsl
        )
        assert (
            o[4].view(-1).tolist() == at
            and int(o[5]) == bonus
            and o[6].tolist() == p
            and int(o[7]) == al
        )
        assert int(o[8]) == sel


@needs_torch
def test_the_page_table_references_give_the_expected_values():
    rng = random.Random(1)
    r2t = torch.randint(-(1 << 20), 1 << 20, (3, 4096), dtype=torch.int32)
    for _ in range(300):
        req, last = rng.randint(0, 2), rng.randint(0, 4095)
        bt, kvl = torch.zeros(48, dtype=torch.int32), torch.zeros(1, dtype=torch.int32)
        fill_step_all_ref(
            r2t, torch.tensor([req]), torch.arange(last - 7, last + 1), bt, kvl
        )
        n = min((last >> 6) + 1, 48)
        want = [
            int(r2t[req, 64 * k]) // 64
            if r2t[req, 64 * k] >= 0
            else -((-int(r2t[req, 64 * k])) // 64)
            for k in range(n)
        ]
        assert (
            bt[:n].tolist() == want
            and bt[n:].eq(0).all()
            and int(kvl) == max(last + 1, 8)
        )
        bt.zero_()
        sl = rng.randint(0, 4000)
        draft_pages_ref(r2t, torch.tensor([req]), torch.tensor([sl]), bt)
        pages = {(sl + i) >> 6 for i in range(8)} & set(range(48))
        assert all(
            int(bt[p]) == int(torch.div(r2t[req, 64 * p], 64, rounding_mode="trunc"))
            for p in pages
        )


@needs_torch
def test_the_chain_tree_reference_gives_the_expected_values():
    rng = random.Random(2)
    for _ in range(300):
        s, mode = rng.randint(1, 5000), rng.choice([0, 1])
        mask = torch.ones((s * 8 + 64) if mode == 0 else 64, dtype=torch.bool)
        pos, rbuf, dout = (
            torch.empty(8, dtype=torch.int64),
            torch.empty((3, 1, 8), dtype=torch.int64),
            torch.empty(8, dtype=torch.int64),
        )
        bonus, drafts = (
            torch.tensor([rng.randint(0, 150000)]),
            torch.randint(0, 150000, (1, 7)),
        )
        chain_tree_ref(bonus, drafts, torch.tensor([s]), mask, mode, pos, rbuf, dout)
        for i in range(8):
            base = (s + 8) * i + s if mode == 0 else 8 * i
            assert mask[base : base + 8].tolist() == [k <= i for k in range(8)]
        assert pos.tolist() == [s + i for i in range(8)] and rbuf[1, 0].tolist() == [
            1,
            2,
            3,
            4,
            5,
            6,
            7,
            -1,
        ]
        assert rbuf[0, 0].tolist() == list(range(8)) and rbuf[2, 0].tolist() == [-1] * 8
        assert dout.tolist() == [int(bonus)] + drafts.view(-1).tolist()


@pytest.mark.skipif(
    torch is None or not torch.cuda.is_available(), reason="needs a GPU with CuTe DSL"
)
def test_the_cute_dsl_ops_are_bitwise_equal_to_the_references():
    pytest.importorskip("cutlass")
    from sgl_nvidia_mega.glm5_3.sm107 import ops

    d = "cuda"
    rng = random.Random(3)
    for _ in range(4000):
        raw, ins, outs = _pv_case(rng, d)
        for dt in (torch.int64, torch.int32):
            args = (*ins[:3], ins[3].to(dt), *ins[4:])
            a, b = outs(), outs()
            ops.post_verify(*args, *a)
            post_verify_ref(*args, *b)
            assert all(torch.equal(x, y) for x, y in zip(a, b))
    w = 1100000
    r2t = (
        torch.arange(4 * w, dtype=torch.int64, device=d).view(4, w) * 3 % (1 << 20)
    ).to(torch.int32)
    for last in [19, 70, 335000, 335063] + [
        rng.randint(20, 1048000) for _ in range(500)
    ]:
        req = torch.tensor([rng.randint(0, 3)], device=d)
        pos = torch.arange(last - 7, last + 1, device=d)
        a = (
            torch.zeros(16388, dtype=torch.int32, device=d),
            torch.zeros(1, dtype=torch.int32, device=d),
        )
        b = (a[0].clone(), a[1].clone())
        ops.fill_step_all(r2t, req, pos, *a)
        fill_step_all_ref(r2t, req, pos, *b)
        assert torch.equal(a[0], b[0]) and torch.equal(a[1], b[1])
        sl = torch.tensor([rng.randint(1, 1048000)], device=d)
        ops.draft_pages(r2t, req, sl, a[0])
        draft_pages_ref(r2t, req, sl, b[0])
        assert torch.equal(a[0], b[0])
    for _ in range(300):
        s, mode = rng.randint(1, 5000), rng.choice([0, 1])
        for bdt in (torch.int32, torch.int64):
            bonus = torch.tensor([rng.randint(0, 150000)], dtype=bdt, device=d)
            drafts, sl = (
                torch.randint(0, 150000, (7,), device=d),
                torch.tensor([s], device=d),
            )
            n = (s * 8 + 64) if mode == 0 else 64
            res = []
            for f in (ops.chain_tree, chain_tree_ref):
                o = (
                    torch.ones(n, dtype=torch.bool, device=d),
                    torch.empty(8, dtype=torch.int64, device=d),
                    torch.full((3, 1, 8), -7, dtype=torch.int64, device=d),
                    torch.empty(8, dtype=torch.int64, device=d),
                )
                f(bonus, drafts, sl, o[0], mode, o[1], o[2], o[3])
                res.append(o)
            assert all(torch.equal(x, y) for x, y in zip(*res))
