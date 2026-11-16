# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""K3.1 persistent experts with the compliant X3-WAR release (``K31MK_EXPERTS_X3_RELEASE=2``, default off;
K-10). After each weight stage's MMAs the MMA warp runs ``bar.sync 12, 32`` and lane 0 arrives (release)
on the slot's empty barrier, which counts 2 (that arrive + the stage's tcgen05.commit), so the producer's metadata
rewrite is ordered after every lane's read. Loads, MMAs and stores are unchanged: outputs must be bit-identical.

Checks (stated before any run; never loosened):
  X1  x3 vs shipped PersistentExperts at TP16 (same host-layout weights, x, ids, one process): 0 differing BF16 output
      cells on every call (2 eager + 2 CUDA-graph replays with new inputs) at B4 and B8 unpacked, B8 unpacked with K-17
      lean (strong), and B8 packed K64 (6 slots, SM107 only) with and without strong.
  X2  engagement: at B4 / B8 the x3 engine runs a different compiled kernel from the shipped one; at B1, B2 and B16
      (not stream calls) the knob is inert: the same compiled kernel object as the shipped engine.
  X3  ``K31MK_EXPERTS_X3_RELEASE`` accepts 0 / 2 only (=1, the lane-0 form, is refused).
  T1  x3 rows vs the FP64 reference (``_ref_moe16.expert_rows``), the T1 of test_k31_experts_tp16.
  X4  the kernel a server builds: PersistentExperts constructed from the environment (the Run
      form: STRONG_SYNC=1, PACKED_K64=1, EXP_SITU_LEAN=1, GRID_SMS=H.bundle_grid_sms()) and compiled THROUGH ``prepare()``, at B8 and B4.
      Its PTX with K31MK_EXPERTS_X3_RELEASE=2, against the same build with 0, must show exactly the deferred form: one
      ``bar.sync 12,32``; one more ``mbarrier.arrive`` and no other sync-class change; no added ``shfl.sync`` (form =1's
      lane-0 broadcast); init counts 1 -> 2 on exactly the ``stages`` empty barriers [stages, 2 x stages) (6 packed).
      The expected form is the design, not a recording of this code.
  x3 meta (K31MK_EXPERTS_X3_RELEASE=3 = R4 meta barrier; K31MK_EXPERTS_HFULL_BAR=1 = bar.sync 13,32 in the hidden
  loader):
  X1m / X2m  X1 + X2 + T1 for env 3 + HFULL_BAR=1 (B4, B8 unpacked; B8 packed 6 weak / strong) and HFULL_BAR=1 alone (B8 p6).
  X4m  prepare()-path PTX (Run form, B8 / B4), env 3 + HFULL_BAR=1 against env 0: meta barriers [misc + 8(2S+50), + 8S) all
       init count 1, every other init unchanged (empty stays 1); one bar.sync 12,32; exactly one arrive on the meta barriers,
       addressed by the register of the consumer's full[slot] wait; every producer wait on empty[slot] (add.s32 R, S, misc+8S)
       paired with a later wait on meta[slot] with the same slot and parity registers; no other use of the meta literals;
       hfull: one bar.sync 13,32 per poll_hidden copy in place of its bar.warp.sync, directly after the proxy fence and
       before lane 0's arrive on hfull (misc + 16S + 256); no shfl change.

Mutations (fixed harness; each must make a base-passing test fail):
  x3_count1    the empty barriers keep count 1 while lane 0 still arrives: a phase completes on the first of the two
               arrivals (lane 0 arrives right after the MMA issue, before the MMAs finish reading the slot) and the
               other spills into the next phase, so the producer refills slots early        [X1 / T1, or HANG]
  x3_inert     prepare never passes the knob to the kernel (knob read, not engaged)            [X2]
  x3_all       prepare passes the knob at every TP16 batch (B16 grouped calls build an x3 kernel or refuse it)  [X2]
  x3_noarrive  bar.sync stays, lane 0's arrive is dropped (count 2): the producer waits forever on a reused slot
               (run alone with a short shard timeout: a labelled HANG catch)                   [X1 HANG; X4 statically]
  m_noarrive   lane 0's arrive on the meta barrier is dropped: the producer waits forever    [X4m statically; X1m HANG]
  m_nowait     the producer's wait on meta[slot] in issue() is dropped (ordering only, no runtime effect)  [X4m]
  m_empty2     the empty barriers also count 2 under env 3 (no second arrival): hang        [X4m statically; X1m HANG]
  hb_inert     the hidden loader keeps bar.warp.sync with HFULL_BAR=1                        [X4m]
  m_as_bar     env 3 maps to the "bar" form                                                  [X4m, X2m kernel check]
  x3_rf4plumb  prepare passes ``self.x3_release and <stream call>`` (the earlier plumbing): "bar" and True is True,
               which the earlier plumbing compiled as form =1; here ExpertsKernel refuses True                [X4, X1]
"""

import collections
import glob
import os
import re

import pytest
import torch

import _harness as H
import _ref_moe as R
import _ref_moe16 as M
import test_k31_experts_tp16 as T16

EXPERTS = "kimi_k31/k3/cutedsl/moe/experts.py"
RUNTIME = "kimi_k31/k3/cutedsl/moe/experts_runtime.py"
MUTATIONS = {
    "x3_count1": (EXPERTS,
                  "commit + lane-0 release arrive\n                    count = 2",
                  "commit + lane-0 release arrive\n                    count = 1"),
    "x3_inert": (RUNTIME,
                 "situ_lean=self.situ_lean, strong=self.strong, x3_release=x3,",
                 "situ_lean=self.situ_lean, strong=self.strong, x3_release=False,"),
    "x3_all": (RUNTIME,
               "x3 = self.x3_release if self.tp == 16 and batch <= 8 and streamed(batch) and not mode else False",
               "x3 = self.x3_release if self.tp == 16 else False"),
    "x3_rf4plumb": (RUNTIME,
                    "x3 = self.x3_release if self.tp == 16 and batch <= 8 and streamed(batch) and not mode else False",
                    "x3 = self.x3_release and (self.tp == 16 and batch <= 8 and streamed(batch) and not mode)"),
    "m_noarrive": (EXPERTS,
                   "                        cute.arch.mbarrier_arrive(metabar.iterator + slot)\n",
                   "                        pass\n"),
    "m_nowait": (EXPERTS,
                 "            if cutlass.const_expr(self.x3_meta):  # cand-09b: every MMA-warp lane's read of meta[slot] is done\n",
                 "            if cutlass.const_expr(False):\n"),
    "m_empty2": (EXPERTS,
                 "        self.x3_count2 = x3_release == \"bar\"",
                 "        self.x3_count2 = bool(x3_release)"),
    "hb_inert": (EXPERTS,
                 "        if cutlass.const_expr(self.hfull_bar):  # cand-09b",
                 "        if cutlass.const_expr(False):  # cand-09b"),
    "m_as_bar": (RUNTIME,
                 '    return {"2": "bar", "3": "meta"}.get(value, False)',
                 '    return {"2": "bar", "3": "bar"}.get(value, False)'),
    "x3_noarrive": (EXPERTS,
                    "                    if lane == 0:\n                        cute.arch.mbarrier_arrive(empty.iterator + slot)\n"
                    "                if cutlass.const_expr(self.x3_meta):",
                    "                if cutlass.const_expr(self.x3_meta):"),
}

needs_gpu = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 10,
    reason="needs an SM100-family GPU",
)
needs_sm107 = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() != (10, 7),
    reason="packed K64 experts need SM107 (VR200)",
)


def _er():
    from sgl_nvidia_mega.kimi_k31.k3.cutedsl.moe import experts_runtime as ER

    return ER


def engine(prepared, batch, x3, packed=0, strong=False, hfull=False):
    """``x3``: False / True (= "bar", env 2) / "meta" (env 3); ``hfull``: K31MK_EXPERTS_HFULL_BAR=1."""
    ER = _er()
    keys = ("K31MK_EXPERTS_X3_RELEASE", "K31MK_EXPERTS_HFULL_BAR")
    old = {k: os.environ.get(k) for k in keys}
    os.environ[keys[0]] = {False: "0", True: "2", "bar": "2", "meta": "3"}[x3]
    os.environ[keys[1]] = "1" if hfull else "0"
    try:
        ex = ER.PersistentExperts(*prepared, beta=R.BETA, linear_beta=R.LIN, activation_floor=None, packed=packed,
                                  strong=strong)
    finally:
        for k, v in old.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
    assert ex.x3_release == {False: False, True: "bar", "bar": "bar", "meta": "meta"}[x3]
    assert getattr(ex, "hfull_bar", False) == hfull
    out = torch.empty((batch * 16, R.H_LAT), dtype=torch.bfloat16, device="cuda")
    ex.prepare(batch, ER.scratch(ex.tp, torch.device("cuda")), out)
    return ex, out


def bitwise(name, a, b):
    differ = int((a.view(torch.int16) != b.view(torch.int16)).sum())
    H.record({"kind": "check", "item": name, "cells_differ": differ, "cells": a.numel()})
    print("CHECK", name, "cells_differ", differ, flush=True)
    assert differ == 0, f"{name}: x3 and shipped differ in {differ} cells"


def run_pair(batch, packed, seed, strong=False, x3=True, hfull=False):
    H.load_plugin(MUTATIONS)
    logical, prepared = T16.weights16()
    (ex_a, out_a), (ex_x, out_x) = (engine(prepared, batch, False, packed, strong),
                                    engine(prepared, batch, x3, packed, strong, hfull))
    same = ex_a.prepared[batch][3] is ex_x.prepared[batch][3]
    on_stream = 4 <= batch <= 8
    H.record({"kind": "check", "item": f"x3.engaged.b{batch}.p{packed}", "same_kernel": same, "stream": on_stream})
    assert same != on_stream, f"b{batch}: x3 engine shares the shipped kernel = {same} (stream call {on_stream})"
    g = H.generator(seed + batch)
    x = torch.empty((batch, R.H_LAT), device="cuda")
    ids = torch.empty((batch, 16), dtype=torch.int32, device="cuda")

    def call():
        ex_a.run(x, ids)
        ex_x.run(x, ids)

    graph = None
    tag = f"x3{'' if x3 is True else '.' + str(x3)}{'.hb' if hfull else ''}.{'k.' if strong else ''}{'p6.' if packed else ''}b{batch}"
    for n in range(4):
        xv = T16.latent(batch, g)
        i, _ = R.random_route(batch, g)
        x.copy_(xv)
        ids.copy_(i)
        H.poison(out_a, out_x)
        if n < 2:
            call()
        else:
            graph = graph or T16.graph_of(call)
            H.poison(out_a, out_x)
            graph.replay()
        torch.cuda.synchronize()
        bitwise(f"{tag}.call{n}", out_x, out_a)
        T16.check_rows(f"{tag}.call{n}", out_x.view(batch, 16, -1), M.expert_rows(logical, xv, i))


@needs_gpu
@pytest.mark.parametrize("batch", (4, 8), ids=lambda b: f"b{b}")
def test_x3_routed(batch):
    """X1 + X2 + T1, unpacked, weak hand-offs."""
    run_pair(batch, 0, 1370)


@needs_gpu
def test_x3_routed_strong():
    """X1 + X2 + T1, unpacked B8 with K-17 lean (``strong=True``)."""
    run_pair(8, 0, 1470, strong=True)


@needs_sm107
@pytest.mark.parametrize("strong", (False, True), ids=("weak", "strong"))
def test_x3_routed_packed(strong):
    """X1 + X2 + T1, packed K64 (6 slots), B8 (the C3 / Run form with strong)."""
    run_pair(8, 6, 1570, strong=strong)


@needs_gpu
@pytest.mark.parametrize("batch", (1, 2, 16), ids=lambda b: f"b{b}")
def test_x3_inert(batch):
    """X2 inert side + X1: not a stream call, so the x3 engine must reuse the shipped kernel (bitwise trivially)."""
    run_pair(batch, 0, 1670)


def test_x3_env_values():
    """X3: only 0 and 2 are accepted (no GPU)."""
    H.load_plugin(MUTATIONS)
    ER = _er()
    old = os.environ.get("K31MK_EXPERTS_X3_RELEASE")
    try:
        for value, want in (("0", False), ("2", "bar"), ("3", "meta")):
            os.environ["K31MK_EXPERTS_X3_RELEASE"] = value
            assert ER.x3_release_enabled() == want
        for value in ("1", "4", "bar", "meta"):
            os.environ["K31MK_EXPERTS_X3_RELEASE"] = value
            with pytest.raises(ValueError):
                ER.x3_release_enabled()
    finally:
        if old is None:
            os.environ.pop("K31MK_EXPERTS_X3_RELEASE", None)
        else:
            os.environ["K31MK_EXPERTS_X3_RELEASE"] = old


RUN_FORM = {"K31MK_EXPERTS_STRONG_SYNC": "1", "K31MK_EXPERTS_PACKED_K64": "1", "K31MK_EXP_SITU_LEAN": "1"}
# + K31MK_EXPERTS_GRID_SMS = H.bundle_grid_sms() (added in prepare_ptx)
SYNC_OPS = ("mbarrier", "tcgen05", "cp.async", "fence", "bar.", "barrier", "griddepcontrol")


def census(ptx):
    """PTX opcode counts (guards and labels stripped)."""
    c = collections.Counter()
    for line in ptx.splitlines():
        for t in line.replace("{", ";").replace("}", ";").split(";"):
            t = t.strip()
            if not t or t.startswith(("//", ".", "$", "%")) or t.endswith(":"):
                continue
            t = re.sub(r"^@!?%?\w+\s+", "", t)
            op = t.split()[0]
            if re.match(r"^[a-z][A-Za-z0-9_.:]*$", op):
                c[op] += 1
    return c


def inits(ptx):
    """mbarrier.init: SMEM offset -> count (the count register's mov.b32 immediate, else "reg")."""
    regs, out = {}, {}
    for line in ptx.splitlines():
        m = re.match(r"\s*mov\.b32\s+(%r\d+),\s*(-?\d+);", line)
        if m:
            regs[m.group(1)] = int(m.group(2))
        m = re.match(r"\s*mbarrier\.init[\w.:]*\s+\[%r\d+\+(\d+)\],\s*(%r\d+|\d+);", line)
        if m:
            v = m.group(2)
            out[int(m.group(1))] = int(v) if v.isdigit() else regs.get(v, "reg")
    return out


def prepare_ptx(monkeypatch, tmp_path, prepared, batch, env):
    """The experts kernel PersistentExperts.prepare() compiles for ``batch`` with the Run form + ``env`` read from the
    environment (as the server constructs it), PTX kept through the compile options."""
    ER = _er()
    for k, v in {**RUN_FORM, "K31MK_EXPERTS_GRID_SMS": str(H.bundle_grid_sms()), **env}.items():
        monkeypatch.setenv(k, v)
    out_dir = tmp_path / ("ptx_b%d_%s" % (batch, "_".join(f"{k[-8:]}{v}" for k, v in sorted(env.items())) or "off"))
    out_dir.mkdir()
    real = ER.cute.compile

    def keep(*args, **kwargs):
        assert "options" not in kwargs, kwargs
        return real(*args, options=f"--keep-ptx --dump-dir={out_dir}", **kwargs)

    ER.compile_experts.cache_clear()
    monkeypatch.setattr(ER.cute, "compile", keep)
    try:
        ex = ER.PersistentExperts(*prepared, beta=R.BETA, linear_beta=R.LIN, activation_floor=None)
        out = torch.empty((batch * 16, R.H_LAT), dtype=torch.bfloat16, device="cuda")
        ex.prepare(batch, ER.scratch(ex.tp, torch.device("cuda")), out)
    finally:
        monkeypatch.setattr(ER.cute, "compile", real)
        ER.compile_experts.cache_clear()
    assert ex.packed == 6 and ex.strong is True and ex.situ_lean is True and ex.grid_for(batch) == H.bundle_grid_sms(), (
        ex.packed, ex.strong, ex.situ_lean, ex.grid_for(batch))
    ptx = getattr(getattr(ex.prepared[batch][3], "artifacts", None), "PTX", None)
    if ptx and len(ptx) < 4096 and os.path.exists(ptx):
        ptx = open(ptx).read()
    if not ptx:
        files = glob.glob(str(out_dir / "**" / "*.ptx"), recursive=True)
        assert len(files) == 1, files
        ptx = open(files[0]).read()
    assert ".target sm_10" in ptx, ptx[:200]
    return ex, ptx


def form_bar(name, a, b, stages):
    """X4: B (env 2) against A (env 0) is exactly the deferred release form."""
    ca, cb = census(a), census(b)
    diff = {k: cb[k] - ca[k] for k in set(ca) | set(cb) if cb[k] != ca[k]}
    sync = {k: v for k, v in diff.items() if k.startswith(SYNC_OPS)}
    shfl = sum(v for k, v in diff.items() if k.startswith("shfl.sync"))
    ia, ib = inits(a), inits(b)
    base = min(ia)
    changed = sorted(o for o in set(ia) | set(ib) if ia.get(o) != ib.get(o))
    bar12 = len(re.findall(r"bar(?:rier)?\.sync(?:\.aligned)?\s+12,\s*32;", b))
    H.record({"kind": "check", "item": name, "sync_changes": sync, "shfl_delta": shfl, "bar12": bar12,
              "init_changed": [o - base for o in changed]})
    print("CHECK", name, "sync", sync, "shfl", shfl, "bar12", bar12, "inits", [o - base for o in changed], flush=True)
    assert bar12 == 1 and len(re.findall(r"bar(?:rier)?\.sync(?:\.aligned)?\s+12,\s*32;", a)) == 0, (name, bar12)
    arrive = {k: v for k, v in sync.items() if k.startswith("mbarrier.arrive")}
    bars = {k: v for k, v in sync.items() if k.startswith(("bar.", "barrier"))}
    assert len(sync) == 2 and list(bars.values()) == [1] and list(arrive.values()) == [1], (name, sync)
    assert "expect_tx" not in list(arrive)[0], (name, arrive)
    assert shfl == 0, (name, "form =1's shfl broadcast present", shfl)
    assert changed == [base + 8 * (stages + i) for i in range(stages)], (name, [o - base for o in changed])
    assert all(ia[o] == 1 and ib[o] == 2 for o in changed) and len(ia) == len(ib), (name, ia, ib)


@needs_sm107
@pytest.mark.parametrize("batch", (8, 4), ids=lambda b: f"b{b}")
def test_x3_prepare_form(batch, monkeypatch, tmp_path):
    """X4: the prepare()-path kernel under K31MK_EXPERTS_X3_RELEASE=2 is the deferred form."""
    H.load_plugin(MUTATIONS)
    _, prepared = T16.weights16()
    ex_a, a = prepare_ptx(monkeypatch, tmp_path, prepared, batch, {"K31MK_EXPERTS_X3_RELEASE": "0"})
    ex_b, b = prepare_ptx(monkeypatch, tmp_path, prepared, batch, {"K31MK_EXPERTS_X3_RELEASE": "2"})
    assert ex_a.x3_release is False and ex_b.x3_release == "bar"
    form_bar(f"x4.bar.b{batch}", a, b, 6)


def waits(ptx, lit):
    """(slot register, parity register, position) of each try_wait whose address register is ``add.s32 R, S, lit``."""
    adds = {m.group(1): m.group(2) for m in re.finditer(r"add\.s32\s+(%r\d+),\s*(%r\d+),\s*" + str(lit) + r";", ptx)}
    return [(adds[m.group(1)], m.group(2), m.start())
            for m in re.finditer(r"mbarrier\.try_wait[\w.:]*\s+p,\s*\[(%r\d+)\],\s*(%r\d+);", ptx) if m.group(1) in adds]


def form_meta(name, a, b, stages):
    """X4m: B (env 3 + HFULL_BAR=1) against A (env 0) is exactly the x3 meta form."""
    ia, ib = inits(a), inits(b)
    base = min(ia)
    E, M, HF = base + 8 * stages, base + 8 * (2 * stages + 50), base + 16 * stages + 256
    meta = [M + 8 * i for i in range(stages)]
    arrives = re.findall(r"mbarrier\.arrive[\w.:]*\s+%rd\d+,\s*\[(%r\d+)\+(\d+)\]", b)
    on_meta = [r for r, o in arrives if M <= int(o) < M + 8 * stages]
    full_regs = {s for s, _, _ in waits(b, base)}
    ew, mw = waits(b, E), waits(b, M)
    uses = sum(1 for x in re.findall(r"[+,]\s*(\d{5,7})[\];]", b) if M <= int(x) < M + 8 * stages)
    adds = len(re.findall(r"add\.s32\s+%r\d+,\s*%r\d+,\s*" + str(M) + r";", b))
    bar12 = len(re.findall(r"bar\.sync\s+12,\s*32;", b))
    bar13 = [m.end() for m in re.finditer(r"bar\.sync\s+13,\s*32;", b)]
    wa, wb = len(re.findall(r"bar\.warp\.sync", a)), len(re.findall(r"bar\.warp\.sync", b))
    shfl = sum(census(b)[k] - census(a)[k] for k in set(census(a)) | set(census(b)) if k.startswith("shfl.sync"))
    rec = {"meta_inits": [ib.get(o) for o in meta], "on_meta": on_meta, "empty_waits": len(ew), "meta_waits": len(mw),
           "bar12": bar12, "bar13": len(bar13), "warp_sync": (wa, wb), "shfl": shfl, "meta_uses": uses}
    H.record({"kind": "check", "item": name, **rec})
    print("CHECK", name, rec, flush=True)
    assert all(ib.get(o) == 1 for o in meta), (name, "meta inits", rec)
    assert all(ib.get(o) == c for o, c in ia.items()), (name, "an existing barrier's count changed (empty must stay 1)")
    assert sorted(set(ib) - set(ia)) == [o for o in meta if o not in ia], (name, "unexpected new inits")
    assert bar12 == 1 and len(on_meta) == 1 and on_meta[0] in full_regs, (name, rec)
    assert len(ew) > 0 and len(ew) == len(mw) and all(
        any(ms == es and mp == ep and mpos > epos for ms, mp, mpos in mw) for es, ep, epos in ew), (name, "waits", rec)
    assert uses == stages + 1 + adds, (name, "other uses of the meta literals", rec)
    assert len(bar13) >= 1 and wa - wb == len(bar13) and shfl == 0, (name, rec)
    for end in bar13:
        nxt = re.search(r"(mbarrier\.[\w.:]+[^;]*;|bar\.[\w.]+[^;]*;|fence[\w.:]*;)", b[end:])
        tgt = re.search(r"\[%r\d+\+(\d+)\]", nxt.group(1)) if nxt else None
        assert nxt and nxt.group(1).startswith("mbarrier.arrive") and tgt and int(tgt.group(1)) == HF, (name, "after bar 13")
        assert "fence.proxy.async.shared::cta" in "".join(b[:end].rstrip().splitlines()[-3:]), (name, "before bar 13")


@needs_sm107
@pytest.mark.parametrize("batch", (8, 4), ids=lambda b: f"b{b}")
def test_x3m_prepare_form(batch, monkeypatch, tmp_path):
    """X4m: the prepare()-path kernel under K31MK_EXPERTS_X3_RELEASE=3 + K31MK_EXPERTS_HFULL_BAR=1 is the x3 meta form."""
    H.load_plugin(MUTATIONS)
    _, prepared = T16.weights16()
    off = {"K31MK_EXPERTS_X3_RELEASE": "0", "K31MK_EXPERTS_HFULL_BAR": "0"}
    on = {"K31MK_EXPERTS_X3_RELEASE": "3", "K31MK_EXPERTS_HFULL_BAR": "1"}
    ex_a, a = prepare_ptx(monkeypatch, tmp_path, prepared, batch, off)
    ex_b, b = prepare_ptx(monkeypatch, tmp_path, prepared, batch, on)
    assert ex_a.x3_release is False and ex_b.x3_release == "meta" and ex_b.hfull_bar is True
    form_meta(f"x4m.b{batch}", a, b, 6)


@needs_gpu
@pytest.mark.parametrize("batch", (4, 8), ids=lambda b: f"b{b}")
def test_x3m_routed(batch):
    """X1m + X2m + T1, unpacked (5 slots), weak hand-offs, env 3 + HFULL_BAR=1."""
    run_pair(batch, 0, 1770, x3="meta", hfull=True)


@needs_sm107
@pytest.mark.parametrize("strong", (False, True), ids=("weak", "strong"))
def test_x3m_routed_packed(strong):
    """X1m + X2m + T1, packed K64 (6 slots), B8, env 3 + HFULL_BAR=1 (the x3 meta form with strong)."""
    run_pair(8, 6, 1870, strong=strong, x3="meta", hfull=True)


@needs_sm107
def test_hb_routed_packed():
    """X1m + X2m + T1, packed K64 (6 slots), B8 strong, HFULL_BAR=1 alone (x3 off)."""
    run_pair(8, 6, 1970, strong=True, x3=False, hfull=True)


@needs_gpu
@pytest.mark.parametrize("batch", (1, 2, 16), ids=lambda b: f"b{b}")
def test_x3m_inert(batch):
    """X2m inert side: not a stream call, so env 3 + HFULL_BAR=1 must reuse the shipped kernel."""
    run_pair(batch, 0, 2070, x3="meta", hfull=True)


def test_hfull_env_values():
    """K31MK_EXPERTS_HFULL_BAR accepts 0 / 1 only (no GPU)."""
    H.load_plugin(MUTATIONS)
    ER = _er()
    old = os.environ.get("K31MK_EXPERTS_HFULL_BAR")
    try:
        for value, want in (("0", False), ("1", True)):
            os.environ["K31MK_EXPERTS_HFULL_BAR"] = value
            assert ER.hfull_bar_enabled() is want
        for value in ("2", "true", "yes"):
            os.environ["K31MK_EXPERTS_HFULL_BAR"] = value
            with pytest.raises(ValueError):
                ER.hfull_bar_enabled()
    finally:
        if old is None:
            os.environ.pop("K31MK_EXPERTS_HFULL_BAR", None)
        else:
            os.environ["K31MK_EXPERTS_HFULL_BAR"] = old
