# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""K31MK_DSA_QFP_PAR_MAX_ROWS routing (dsa/verify.py _enable_qfp): which QFP profiles fork the index-K store.

Expected values come from the knob's specification, not from the code: a profile of ``rows`` rows forks iff
K31MK_DSA_QFP == 2 and (K31MK_DSA_QFP_PAR_MAX_ROWS == 0 or rows <= K31MK_DSA_QFP_PAR_MAX_ROWS); an explicit
``parallel`` argument overrides both. Host-only (no kernel runs): the profile is a stand-in with the packed pitch.
"""

import os
import subprocess
import sys
from types import SimpleNamespace

import pytest

ROWS = (40, 64, 72, 88, 96, 104, 112, 128)


def spec(mode, limit, rows):
    return mode == 2 and (limit == 0 or rows <= limit)


def profile(V, rows, heads=6):
    _, _, pitch = V.qf_columns(heads)
    base = SimpleNamespace(heads=heads, pitches={rows: pitch})
    ip = SimpleNamespace(folded=object(), weights=None)  # fold already built (as BuiltSparseVerify leaves it)
    return SimpleNamespace(base=base, index_projection=ip, qfp=None, qfp_par=None)


KW = SimpleNamespace(norm=None, bias=None, eps=1e-6)


@pytest.mark.parametrize("mode,limit", [(2, 0), (2, 96), (2, 104), (2, 40), (1, 0), (1, 96)])
def test_qfp_par_route(monkeypatch, mode, limit):
    from sgl_nvidia_mega.kimi_k31.dsa import verify as V

    monkeypatch.setattr(V, "DSA_QFP", mode)
    monkeypatch.setattr(V, "DSA_QFP_PAR_MAX_ROWS", limit)
    got = {}
    for rows in ROWS:
        p = profile(V, rows)
        V._enable_qfp(p, rows, KW)
        assert p.qfp is not None, rows
        got[rows] = p.qfp_par
    want = {rows: spec(mode, limit, rows) for rows in ROWS}
    assert got == want, f"mode {mode} limit {limit}: got {got}, want {want}"


@pytest.mark.parametrize("parallel", [True, False])
def test_qfp_par_explicit(monkeypatch, parallel):
    from sgl_nvidia_mega.kimi_k31.dsa import verify as V

    monkeypatch.setattr(V, "DSA_QFP", 2)
    monkeypatch.setattr(V, "DSA_QFP_PAR_MAX_ROWS", 96)
    for rows in ROWS:
        p = profile(V, rows)
        V._enable_qfp(p, rows, KW, parallel=parallel)
        assert p.qfp_par is parallel, (rows, parallel, p.qfp_par)


@pytest.mark.parametrize("value,ok", [("96", True), ("0", True), ("-1", False)])
def test_qfp_par_env(value, ok):
    env = dict(os.environ, K31MK_DSA_QFP_PAR_MAX_ROWS=value)
    code = "from sgl_nvidia_mega.kimi_k31.dsa import verify as V; print('PAR', V.DSA_QFP_PAR_MAX_ROWS)"
    r = subprocess.run([sys.executable, "-c", code], env=env, capture_output=True, text=True, timeout=600)
    if ok:
        assert r.returncode == 0 and f"PAR {int(value)}" in r.stdout, r.stderr[-2000:]
    else:
        assert r.returncode != 0 and "K31MK_DSA_QFP_PAR_MAX_ROWS must be >= 0" in r.stderr, r.stderr[-2000:]
