# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""K-22a fix in MlaQuery.tiled (K31MK_MLAQ_K22 / MlaQuery(k22=...)).

The fix changes WHERE the peers' DSMEM bulk copies land (mode C: own SMEM for qn) or how the first cluster arrive is
ordered (mode A: fence.proxy.async.shared::cta + .release arrive); the arithmetic and its order are unchanged, so the
claim is BITWISE equality with the as-built kernel (mode 0) on identical inputs. The memory-model property itself is
not observable here (the race is not shown to fail in practice): separate PTX checks
read it from the PTX (CPU), including a restore-the-alias mutant.

One GPU, SM100 family. K31_TEST_HEADS=6 (TP16: T = 16 at rows 8 / 16, T = 32 at rows 32) or 12 (TP8: T = 16 at rows
16, T = 64 at rows 64, where C applies A because x + own qn would exceed 232,448 B per CTA). Each case builds the real
serving class (PreparedIndexMla -> IndexMlaQuery, or PreparedMla -> MlaQuery) with the query type pinned to a mode,
runs the dense path (query -> trtllm-gen dense core -> MlaValue) A-B-A on the same projected rows, KV pool and index
cache, and compares (gates fixed before any run):
- query bytes [rows, H, 576], the whole KV pool, the whole index-K cache: mode X == mode 0, bitwise (the query kernel's
  direct outputs). Control: mode 0 == mode 0 (second A run) bitwise, else the case is invalid (test construction).
- layer output: mode X == mode 0 bitwise when the control's outputs are bitwise (the core's own determinism), else
  recorded only.
- engagement, read from the PTX of the query kernel that ran (CUTE_DSL_KEEP=ptx, as run_shards.sh sets; no PTX = fail):
  "C" when the peers' bulk-copy destination (the constant of the `add.s32` that defines the mapa source) lies past the x
  tile (>= T x 1568 x 2 B), "A" when the first cluster arrive is a non-relaxed `barrier.cluster.arrive` directly after
  `fence.proxy.async.shared::cta`, else "0"; it must equal the mode applied (C at T = 64 applies A). (In the first run the
  driver attribute CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES reads the device maximum for every arm - CuTe sets the device
  maximum, not the kernel's size - so it cannot show engagement.)
- knob: with K31MK_MLAQ_K22 set in the environment (the task's knob-on runs), the serving build (no pinned type) runs
  that mode's kernel; unset: mode 0 (default off).

Mutations (``K31_MUTATION=<name>`` via _harness / the fixed run_suite.sh; each must FAIL this file):
  k22c_alias_back      C allocates nothing and aliases x again (the K-22a layout)   -> engagement (C cases)
  k22c_copy_to_alias   C's readers use the own qn but the peers still copy into x  -> bitwise (C cases)
  k22_kwarg_ignored    MlaQuery ignores k22= (env default "0" everywhere)          -> engagement (C cases)
"""

from __future__ import annotations

import functools
import math
import os
import re

import pytest
import torch

import _harness as H

Q_FILE = "kimi_k31/k3/cutedsl/mla/query.py"
MUTATIONS = {
    "k22c_alias_back": (
        Q_FILE,
        "qn = alloc(smem, BFloat16, (senders, T, R), 128)",
        "qn = view(xq.iterator, 9 * T * R * 4, BFloat16, (senders, T, R), (T * R, R, 1))",
    ),
    "k22c_copy_to_alias": (
        Q_FILE,
        "qn.iterator + part * T * R, qblk.iterator, barq, tid, size=T * R * 2",
        "xq.iterator + 9 * T * R * 4 + part * T * R, qblk.iterator, barq, tid, size=T * R * 2",
    ),
    "k22_kwarg_ignored": (
        Q_FILE,
        "self.k22 = K22_FIX if k22 is None else k22",
        "self.k22 = K22_FIX",
    ),
}
H.load_plugin(MUTATIONS)

import test_k31_mla as M  # noqa: E402

HEADS = M.HEADS
CASES = {
    6: [("index", 8, "C"), ("mla", 8, "C"), ("index", 8, "A"), ("index", 16, "C"), ("index", 32, "C"),
        ("mla", 32, "A")],
    12: [("index", 16, "C"), ("mla", 16, "A"), ("index", 64, "C"), ("mla", 64, "A")],
}[HEADS]
CTX = 1000

needs_gpu = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 10,
    reason="needs an SM100-family GPU",
)


@functools.cache  # one class per (type, mode): compile_core caches on it, so the control reruns the same kernel
def _pinned(base, mode):
    """``base`` with the K-22a mode fixed (compile_core builds query_type(heads, batch, q_b_cp_async=, dcp=))."""

    def __init__(self, *args, **kwargs):
        base.__init__(self, *args, k22=mode, **kwargs)

    return type(f"{base.__name__}K22{mode}", (base,), {"__init__": __init__})


def _build(kind, w, rows, kv, idx, mode):
    """The serving object of ``kind`` with its query kernel pinned to ``mode`` (None: the environment's knob)."""
    from sgl_nvidia_mega.kimi_k31.k3.cutedsl.mla.prepared import PreparedMla
    from sgl_nvidia_mega.kimi_k31.k3.cutedsl.mla.query import MlaQuery
    from sgl_nvidia_mega.kimi_k31.folded_index import IndexMlaQuery, PreparedIndexMla

    scratch = torch.empty((rows, HEADS, 576), dtype=torch.uint8, device="cuda")
    attention, _ = M.fakes(w)
    if kind == "index":
        obj = PreparedIndexMla(M.fakes(w), {rows: M.PITCH}, scratch, max_context=1 << 17, core="trtllm", dcp=None,
                               queries=(1,))
        if mode is not None:
            obj.query_type = _pinned(IndexMlaQuery, mode)
        obj.bind_index_cache(idx, M.PAGE)
    else:
        obj = PreparedMla(attention, {rows: M.PITCH}, scratch, max_context=1 << 17, core="trtllm", dcp=None,
                          queries=(1,))
        if mode is not None:
            obj.query_type = _pinned(MlaQuery, mode)
    obj.prepare_cache(kv)
    return obj


def _ptx(obj, rows):
    """PTX of the query kernel ``obj`` runs at ``rows``: the compiled object's kept artifact, else the newest dump of
    this query class at this batch in CUTE_DSL_DUMP_DIR; None when neither exists (the caller fails)."""
    kernel = obj.kernels[rows, 1][0]
    ptx = getattr(getattr(kernel, "compiled", kernel), "__ptx__", None)
    if ptx:
        return ptx
    dump = os.environ.get("CUTE_DSL_DUMP_DIR")
    name = f"{obj.query_type.__name__}_object_at"
    if not dump or not os.path.isdir(dump):
        return None
    files = [os.path.join(dump, f) for f in os.listdir(dump)
             if f.endswith(".ptx") and name in f and f"_{rows}_2112_" in f]
    return open(max(files, key=os.path.getmtime)).read() if files else None


def _engaged(obj, rows, tokens):
    """The K-22a form of the query kernel that ran: "C" (bulk destination past the x tile), "A" (proxy fence directly
    before a release first arrive) or "0"."""
    text = _ptx(obj, rows)
    assert text, f"no PTX of the rows-{rows} query kernel (CUTE_DSL_KEEP=ptx unset?): engagement unknown"
    lines = text.split("\n")
    b = next(i for i, ln in enumerate(lines) if "cp.async.bulk.shared::cluster.shared::cta" in ln)
    dst = re.search(r"mapa\.shared::cluster\.u32 dst,(%r\d+),", lines[b]).group(1)
    define = next(ln for ln in reversed(lines[:b]) if re.match(rf"\s*add\.s32\s+{dst},", ln))
    offset = int(define.rsplit(",", 1)[1].strip(" ;"))
    arrive = next(i for i, ln in enumerate(lines) if "barrier.cluster.arrive" in ln)
    fenced = ".relaxed" not in lines[arrive] and "fence.proxy.async.shared::cta" in lines[arrive - 1]
    return "C" if offset >= tokens * 1568 * 2 else "A" if fenced else "0", offset


def _case(seed, rows):
    g = H.generator(seed)
    w = M.make_weights(7)
    pages = rows * math.ceil(CTX / M.PAGE) + 2
    kv = M.kv_pool(pages, g)
    idx = torch.randint(0, 120, (pages, M.PAGE * 132), generator=g, device="cuda", dtype=torch.uint8)
    table, slots, lengths = M.requests(rows, CTX, pages, g)
    projected, _ = M.activations(rows, g)
    return w, kv, idx, table, slots, lengths, projected


def _run(kind, case, rows, mode):
    w, kv0, idx0, table, slots, lengths, projected = case
    kv, idx = kv0.clone(), idx0.clone()
    obj = _build(kind, w, rows, kv, idx, mode)
    out = torch.empty((rows, HEADS * 128), dtype=torch.bfloat16, device="cuda")
    H.poison(out, obj.query_scratch)
    obj.run(projected, out, cache=kv, slots=slots, block_table=table, lengths=lengths)
    torch.cuda.synchronize()
    q = obj.query_scratch[:rows].clone()
    tokens = _applied(rows, "0")[1]
    return {"query": q, "kv": kv, "index": idx, "out": out.clone(), "form": _engaged(obj, rows, tokens)}


def _applied(rows, mode):
    from sgl_nvidia_mega.kimi_k31.k3.cutedsl.mla.query import MlaQuery

    tokens = MlaQuery(HEADS, rows, k22="0").tokens  # the tile height T (the expected fallback is derived here)
    return ("A" if mode == "C" and tokens > 32 else mode), tokens


def _equal(a, b):
    return torch.equal(a.view(torch.uint8), b.view(torch.uint8))


@needs_gpu
@pytest.mark.parametrize("kind,rows,mode", CASES, ids=[f"{k}_b{r}_{m}" for k, r, m in CASES])
def test_k22_bitwise(kind, rows, mode):
    tag = f"{kind}_b{rows}_{mode}_h{HEADS}"
    case = _case(9100 + rows + 3 * (kind == "mla"), rows)
    base, fixed, again = (_run(kind, case, rows, m) for m in ("0", mode, "0"))
    for key in ("query", "kv", "index"):
        assert _equal(base[key], again[key]), f"{tag}: control mode 0 vs mode 0 differs in {key} (test construction)"
        diff = int((base[key].view(torch.uint8) != fixed[key].view(torch.uint8)).sum())
        H.record({"kind": "bitwise", "item": f"k22.{tag}.{key}", "differ_bytes": diff})
        assert diff == 0, f"{tag}: {key} differs from mode 0 in {diff} bytes"
    assert not bool((base["query"] == 0xFF).all(-1).any()), f"{tag}: query rows left poisoned (kernel did not run)"
    control_out = _equal(base["out"], again["out"])
    out_equal = _equal(base["out"], fixed["out"])
    H.record({"kind": "bitwise", "item": f"k22.{tag}.out", "control_bitwise": control_out, "equal": out_equal})
    if control_out:
        assert out_equal, f"{tag}: layer output differs from mode 0 (control bitwise)"
    applied, tokens = _applied(rows, mode)
    print("K22", tag, "T", tokens, "applied", applied, "form base", base["form"], "fixed", fixed["form"], "out_ctl",
          control_out, flush=True)
    assert base["form"][0] == "0", f"{tag}: mode 0 kernel is not the as-built form: {base['form']}"
    assert fixed["form"][0] == applied, f"{tag}: mode {mode} ran form {fixed['form']}, expected {applied} (not engaged)"


@needs_gpu
def test_k22_env_knob():
    """The serving build (query type not pinned) follows K31MK_MLAQ_K22 (unset: mode 0, default off)."""
    env = os.environ.get("K31MK_MLAQ_K22", "0")
    rows = 8 if HEADS == 6 else 16
    case = _case(9300, rows)
    served = _run("index", case, rows, None)
    pinned = _run("index", case, rows, env)
    base = _run("index", case, rows, "0")
    print("K22_ENV", env, "form served", served["form"], "pinned", pinned["form"], "base", base["form"], flush=True)
    assert served["form"][0] == _applied(rows, env)[0], f"serving build ran {served['form']}, knob says {env}"
    assert served["form"] == pinned["form"], f"serving build {served['form']} != mode {env} {pinned['form']}"
    for key in ("query", "kv", "index"):
        assert _equal(served[key], base[key]), f"serving build (K31MK_MLAQ_K22={env}) {key} != mode 0"
