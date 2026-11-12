# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""K3.1 TP16 routed experts taking x as the MXFP8 plan by TMA (``K31MK_EXP_XPLAN``, default off).

The shipped TP16 B<=8 ``ExpertsKernel`` has every CTA load the whole FP32 latent (114 KB) and quantize it to MXFP8 into
its SMEM x image. With the knob, ``planned.EarlyPlanKernel`` writes the MXFP8 plan once per call (the stock input rule,
``experts.plan_chunk``) and each CTA fetches the same SMEM image with two TMAs (28 atoms of 8 rows x 128 B, 128B swizzle,
and the scale words; rows >= batch out of bounds, zero-filled) right after its dependency wait. Same SMEM bytes, so the
claim is bitwise.

Checks (stated before any run; never loosened):
  XP  ``EarlyPlanKernel`` plan bytes equal ``_ref_moe16.mx8_bytes(x)`` (E4M3 codes and UE8M0 scales, the stock rule)
      byte for byte (B1, B8; 3 calls, the last with an all-zero block and a 1e-30-scaled block).
  X1  xplan engine vs shipped engine (same weights, x, ids; separate scratch), ``PersistentExperts`` TP16 B1/2/4/8 and
      B16 (grouped: the knob must leave it unchanged), unpacked and packed K64 (6 slots, SM107 only): 0 differing BF16
      cells on every call (2 eager, the second with a zero block and a 1e-30 block, then 2 CUDA-graph replays with new
      inputs); outputs NaN-poisoned before each call.
  T1  xplan rows vs the FP64 reference ``_ref_moe16.expert_rows`` (Frobenius rel-L2 <= 1e-2, every row <= 3e-2, all
      cells finite): independent of the shipped engine (X1 alone passes if both share a bug).
  XR  engagement: the xplan engine prepared with ``planned=True`` (caller-written plan) and given a plan written for
      x' while ``run`` gets x: its rows equal the shipped rows of x' bit for bit and differ from those of x (>= 1000
      cells), so the engine reads x from the plan, not from the FP32 argument.
  XS  scope: the knob engages at B8 only (the c=1 verify batch); at B1/B2/B4 the xplan engine
      is prepared exactly as the shipped one (no plan, no plan maps): ``engine`` asserts it, so X1 at B1/2/4 is the inert side.
  XK16 the prepare()-path PTX of the Run form + XPLAN=1 at B8 (packed 6, strong, lean, bundle grid, x3 meta, hfull, pace
      240): after the first ``griddepcontrol.wait`` and before the first ``cp.async.bulk.tensor`` the TMA-issuing thread
      executes ``fence.proxy.async.global`` and then ``mbarrier.arrive.expect_tx`` (K-16 consumer side).
  XQ  the routing kernel's copy role with the front hook (``K31MK_EXP_XPLAN_ROUTE``: plan + plan_tma at B<=8), alone
      on a mailbox holding FP32 x: its latent equals x bit for bit, its plan bytes equal ``mx8_bytes(x)`` (codes,
      scales, zero scale padding), the mailbox latent words are re-armed, and the xplan engine prepared ``planned`` on
      that plan equals the shipped engine on x bit for bit (B1, B8).

Mutations (fixed harness; each must be CAUGHT):
  xp_row      the x TMA starts at plan row 1 (row offset)                                        [X1, T1]
  xp_atom     the x TMA starts at atom 1 (atoms shifted, the last one zero)                      [X1, T1]
  xp_sfgroup  the scale TMA starts at atom 1 (scale words of the wrong group)                    [X1, T1]
  xp_plan     EarlyPlanKernel quantizes chunk c ^ 1 into chunk c (plan writer bug)                [XP, X1, T1]
  xp_writer   prepare never runs the plan writer (stale plan)                                    [X1, T1]
  xp_bytes    xready[0] expects 16 bytes more than the TMAs deliver: the MMA warp never starts    HANG (own run)
  xr_chunk    the copy role quantizes its chunk into position c ^ 1 of the plan                  [XQ]
  xk16_nofence  the consumer fence after griddepcontrol.wait is dropped (K-16 regresses)           [XK16]
  xs_scope    prepare engages the knob at every B<=8 again (B1-B7 would take OOB-filled TMA rows) [XS via X1 b1/b2/b4]

Timing (``K31_TIME=1``, Rubin; pre-registered): per-call device time with cold weights.
"""

import os
import statistics

import pytest
import torch

import _harness as H
import _ref_moe as R
import _ref_moe16 as M
import test_k31_experts_tp16 as T16

EX = "kimi_k31/k3/cutedsl/moe/experts.py"
ER_FILE = "kimi_k31/k3/cutedsl/moe/experts_runtime.py"
PL = "kimi_k31/k3/cutedsl/moe/planned.py"
FR = "kimi_k31/k3/cutedsl/moe/front.py"
_X_TMA = "load_tma3_keep(sm.iterator + self.off_x, maps.iterator + 256, bar, 0, 0)"
_SF_TMA = "load_tma3_keep(sm.iterator + self.off_xsf, maps.iterator + 384, bar, 0, 0)"
_WRITER = "compile_early_plan(batch, self.device.index))\n            plan = plan[:1] if planned else plan"
MUTATIONS = {
    "xp_row": (EX, _X_TMA, _X_TMA.replace("bar, 0, 0)", "bar, 1, 0)")),
    "xp_atom": (EX, _X_TMA, _X_TMA.replace("bar, 0, 0)", "bar, 0, 1)")),
    "xp_sfgroup": (EX, _SF_TMA, _SF_TMA.replace("bar, 0, 0)", "bar, 0, 1)")),
    "xp_plan": (PL, "                source = x.iterator + token * 3584 + chunk * 8\n                words = (*load_words",
                "                source = x.iterator + token * 3584 + (chunk ^ 1) * 8\n                words = (*load_words"),
    "xp_writer": (ER_FILE, _WRITER, _WRITER.replace("plan[:1] if planned else plan", "plan[:1]")),
    "xp_bytes": (EX, "cute.arch.mbarrier_arrive_and_expect_tx(bar, 28672 + 28 * 128)",
                 "cute.arch.mbarrier_arrive_and_expect_tx(bar, 28672 + 28 * 128 + 16)"),
    "xk16_nofence": (EX, "            fence_proxy_async_global()  # K-16: the plan is the predecessor grid's generic stores (after the wait)\n",
                     ""),
    "xs_scope": (ER_FILE, "        xtma = self.xtma if batch == 8 else False\n", "        xtma = self.xtma if batch <= 8 else False\n"),
    "xr_chunk": (FR, "plan_chunk(plan, token, tid + 256 * i, words, fp32=self.fp32)",
                 "plan_chunk(plan, token, (tid + 256 * i) ^ 1, words, fp32=self.fp32)"),
}

needs_gpu = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 10,
    reason="needs an SM100-family GPU",
)
PACKED = int(os.environ.get("XP_PACKED_STAGES", "6"))


def _er():
    from sgl_nvidia_mega.kimi_k31.k3.cutedsl.moe import experts_runtime as ER

    return ER


def engine(prepared, batch, xplan, packed=0, planned=False):
    """A PersistentExperts at TP16 with its own scratch, prepared for ``batch``; returns (engine, out, buffers)."""
    ER = _er()
    ex = ER.PersistentExperts(*prepared, beta=R.BETA, linear_beta=R.LIN, activation_floor=None, packed=packed,
                              xplan=xplan)
    assert ex.xtma == xplan and ex.packed == packed, (ex.xtma, ex.packed, xplan, packed)
    out = torch.empty((batch * 16, R.H_LAT), dtype=torch.bfloat16, device="cuda")
    buffers = ER.scratch(ex.tp, torch.device("cuda"))
    ex.prepare(batch, buffers, out, planned=planned)
    plan = ex.prepared[batch][5]
    if xplan and batch == 8:  # engagement (B8 only, XS): the plan tuple carries the writer unless the caller writes it
        assert plan is not None and len(plan) == (1 if planned else 2), plan
        assert ex.xtma_maps[-1].numel() == 512, ex.xtma_maps[-1].numel()
    elif batch <= 8:
        assert plan is None, plan
    return ex, out, buffers


def check_bitwise(name, got, want):
    differ = int((got.view(torch.int16) != want.view(torch.int16)).sum())
    H.record({"kind": "check", "item": name, "x1_cells_differ": differ, "cells": got.numel()})
    print("CHECK", name, "x1_cells_differ", differ, flush=True)
    assert differ == 0, f"{name}: xplan and shipped rows differ in {differ} cells"


def special(x):
    """An all-zero block (the 1e-10 clamp, zero codes) and a 1e-30-scaled block in every token."""
    x = x.clone()
    x[:, :32] = 0.0
    x[:, 32:64] *= 1e-30
    return M.canon(x)


def write_plan(x, plan):
    from sgl_nvidia_mega.kimi_k31.k3.cutedsl.common.host import stream, to_cute
    from sgl_nvidia_mega.kimi_k31.k3.cutedsl.moe import planned as PLN

    code = PLN.compile_early_plan(x.shape[0], torch.cuda.current_device())
    code(to_cute(x, 16), to_cute(plan, 16), stream(x.device))


@needs_gpu
@pytest.mark.parametrize("batch", (1, 8), ids=lambda b: f"b{b}")
def test_plan_early(batch):
    """XP: EarlyPlanKernel's MXFP8 bytes equal the stock rule's, byte for byte."""
    H.load_plugin(MUTATIONS)
    from sgl_nvidia_mega.kimi_k31.k3.cutedsl.moe import experts as EX_

    plan = torch.empty(EX_.PLAN_BYTES, dtype=torch.uint8, device="cuda")
    g = H.generator(1360 + batch)
    for n in range(3):
        x = T16.latent(batch, g)
        if n == 2:
            x = special(x)
        plan.fill_(0xA5)
        write_plan(x, plan)
        torch.cuda.synchronize()
        q_ref, s_ref = M.mx8_bytes(x)
        q = plan[: batch * R.H_LAT].view(batch, R.H_LAT)
        words = plan[EX_.PLAN_XSF: EX_.PLAN_XSF + batch * 448].view(batch, 28, 16)
        scales = words[:, :, :4].reshape(batch, 112)
        bad_q, bad_s = int((q != q_ref).sum()), int((scales != s_ref).sum())
        bad_pad = int((words[:, :, 4:] != 0).sum())  # bytes 4-15 of every scale word: zero (the shipped clear_x image)
        H.record({"kind": "check", "item": f"xp.b{batch}.call{n}", "bad_codes": bad_q, "bad_scales": bad_s,
                  "bad_scale_padding": bad_pad})
        print("CHECK xp", batch, n, bad_q, bad_s, bad_pad, flush=True)
        assert bad_q == 0 and bad_s == 0 and bad_pad == 0, (
            f"xp.b{batch}.call{n}: {bad_q} E4M3 bytes, {bad_s} scale bytes, {bad_pad} padding bytes differ")


@needs_gpu
@pytest.mark.parametrize("packed", (0, PACKED), ids=lambda p: f"p{p}")
@pytest.mark.parametrize("batch", (1, 2, 4, 8, 16), ids=lambda b: f"b{b}")
def test_xplan(batch, packed):
    """X1 + T1: xplan vs shipped engine (same packing), B1-8 (B16: the unchanged grouped path)."""
    H.load_plugin(MUTATIONS)
    if packed and not H.is_sm107():
        H.record({"kind": "skip", "item": f"xplan.b{batch}.p{packed}", "reason": "packed K64 needs SM107"})
        pytest.skip("packed K64 experts need SM107 (VR200)")
    logical, prepared = T16.weights16()
    ex_a, out_a, _ = engine(prepared, batch, False, packed)
    ex_x, out_x, _ = engine(prepared, batch, True, packed)
    g = H.generator(1460 + 10 * batch + packed)
    x = torch.empty((batch, R.H_LAT), device="cuda")
    ids = torch.empty((batch, 16), dtype=torch.int32, device="cuda")

    def call():
        ex_a.run(x, ids)
        ex_x.run(x, ids)

    graph = None
    for n in range(4):
        xv = T16.latent(batch, g)
        if n == 1:
            xv = special(xv)
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
        name = f"xplan.b{batch}.p{packed}.call{n}"
        check_bitwise(name, out_x, out_a)
        T16.check_rows(name, out_x.view(batch, 16, -1), M.expert_rows(logical, xv, i))
    H.record({"kind": "resources", "item": f"xplan.b{batch}.p{packed}", "grid": ex_x.grid_for(batch),
              "xtma": str(ex_x.xtma), "packed": ex_x.packed, **H.device_facts()})


@needs_gpu
def test_xplan_reads_plan():
    """XR: with a caller-written plan the engine's x comes from the plan, not from its FP32 argument."""
    H.load_plugin(MUTATIONS)
    batch = 8
    _, prepared = T16.weights16()
    ex_a, out_a, _ = engine(prepared, batch, False)
    ex_x, out_x, buffers = engine(prepared, batch, True, planned=True)
    g = H.generator(1561)
    x, x2 = T16.latent(batch, g), T16.latent(batch, g)
    ids, _ = R.random_route(batch, g)
    H.poison(out_a, out_x)
    write_plan(x2, buffers["plan"])
    ex_x.run(x, ids)
    torch.cuda.synchronize()
    got = out_x.clone()
    ex_a.run(x2, ids)
    torch.cuda.synchronize()
    want = out_a.clone()
    ex_a.run(x, ids)
    torch.cuda.synchronize()
    other = int((got.view(torch.int16) != out_a.view(torch.int16)).sum())
    H.record({"kind": "check", "item": "xr.b8", "cells_vs_rows_of_x": other})
    check_bitwise("xr.b8.rows_of_plan_x2", got, want)
    assert other >= 1000, f"xr.b8: rows from the plan of x' differ from the rows of x in only {other} cells"


@needs_gpu
@pytest.mark.parametrize("batch", (1, 8), ids=lambda b: f"b{b}")
def test_route_plan(batch):
    """XQ: the front hook's copy role writes the B<=8 plan with the stock rule; the TMA-fed experts on it == shipped."""
    H.load_plugin(MUTATIONS)
    from sgl_nvidia_mega.kimi_k31.k3.cutedsl.common.host import stream, to_cute
    from sgl_nvidia_mega.kimi_k31.k3.cutedsl.moe import experts as EX_
    from sgl_nvidia_mega.kimi_k31.k3.cutedsl.moe import front as F
    from cutlass import Float32

    dev = torch.device("cuda")
    code = F.compile_route(batch, 384, True, (False, True, False), dev.index, True, True, 0, bounded=False, fp32=True,
                           plan_tma=True)
    _, prepared = T16.weights16()
    ex_a, out_a, _ = engine(prepared, batch, False)
    ex_x, out_x, buffers = engine(prepared, batch, True, planned=True)
    plan = buffers["plan"]
    f32 = torch.float32
    logits, bias = torch.zeros((batch, 896), dtype=f32, device=dev), torch.zeros(896, dtype=f32, device=dev)
    ids0 = torch.zeros((batch, 16), dtype=torch.int32, device=dev)
    weights = torch.zeros((batch, 16), dtype=torch.bfloat16, device=dev)
    mailbox = torch.empty((1, 64, 7168 + 2 * 896), dtype=torch.bfloat16, device=dev)
    latent = torch.empty((batch, R.H_LAT), dtype=f32, device=dev)
    gate_up = torch.zeros((batch, 768), dtype=f32, device=dev)
    hidden = torch.zeros((batch, 384), dtype=torch.bfloat16, device=dev)
    words = mailbox.view(torch.int32)  # [1, 64, 4480]: FP32 latent words [0, 3584)
    g = H.generator(1660 + batch)
    for n in range(2):
        x = T16.latent(batch, g)
        if n == 1:
            x = special(x)
        i, _ = R.random_route(batch, g)
        words.fill_(-(1 << 31))  # the 0x80000000 sentinel everywhere ...
        words[0, :batch, : R.H_LAT].copy_(x.view(torch.int32))  # ... except this call's latent words
        H.poison(latent)
        plan.fill_(0xA5)
        args = (logits, bias, ids0, weights, mailbox, latent, gate_up, hidden)
        code(*map(to_cute, args), Float32(1), Float32(R.BETA), Float32(R.LIN), to_cute(plan), stream(dev))
        torch.cuda.synchronize()
        q_ref, s_ref = M.mx8_bytes(x)
        q = plan[: batch * R.H_LAT].view(batch, R.H_LAT)
        sw = plan[EX_.PLAN_XSF: EX_.PLAN_XSF + batch * 448].view(batch, 28, 16)
        rec = {"kind": "check", "item": f"xq.b{batch}.call{n}",
               "latent_cells_differ": int((latent.view(torch.int32) != x.view(torch.int32)).sum()),
               "bad_codes": int((q != q_ref).sum()), "bad_scales": int((sw[:, :, :4].reshape(batch, 112) != s_ref).sum()),
               "bad_scale_padding": int((sw[:, :, 4:] != 0).sum()),
               "unarmed_words": int((words[0, :batch, : R.H_LAT] != -(1 << 31)).sum())}
        H.record(rec)
        print("CHECK", rec, flush=True)
        assert all(v == 0 for k, v in rec.items() if k not in ("kind", "item")), rec
        H.poison(out_a, out_x)
        ex_x.run(x, i)
        ex_a.run(x, i)
        torch.cuda.synchronize()
        check_bitwise(f"xq.b{batch}.call{n}.experts", out_x, out_a)


# ------------------------------------------------------------------------------ timing (Rubin)
ARMS = {  # name: (xplan, packed, bundle grid (H.bundle_grid_sms()) or the device's, planned (plan pre-written), valid outputs)
    "a16": (False, 0, None, False, True),
    "x16": (True, 0, None, False, True),
    "a16g": (False, 0, True, False, True),
    "x16g": (True, 0, True, False, True),
    "p16g": (False, PACKED, True, False, True),
    "xp16g": (True, PACKED, True, False, True),
    "xn16": (True, 0, None, True, True),  # control: plan pre-written, no plan kernel per call (call 0 valid only)
}
BITWISE_PAIRS = {"x16": "a16", "x16g": "a16g", "xp16g": "p16g", "xn16": "a16"}


def timing_arm(name, prepared, batch):
    xplan, packed, bundle, planned, _ = ARMS[name]
    saved = os.environ.pop("K31MK_EXPERTS_GRID_SMS", None)
    if bundle:
        os.environ["K31MK_EXPERTS_GRID_SMS"] = str(H.bundle_grid_sms())
    try:
        return engine(prepared, batch, xplan, packed, planned)
    finally:
        os.environ.pop("K31MK_EXPERTS_GRID_SMS", None)
        if saved is not None:
            os.environ["K31MK_EXPERTS_GRID_SMS"] = saved


@needs_gpu
@pytest.mark.parametrize("batch", (8,), ids=lambda b: f"b{b}")
def test_time_xplan(batch):
    """Pre-registered timing: arms of ``XP_TIME_ARMS``, A..Z Z..A A..Z Z..A on this GPU. Valid
    arms first pass T1 on the first timing route and equal their bitwise partner's rows; controls are labelled."""
    if os.environ.get("K31_TIME") != "1" or os.environ.get("K31_MUTATION"):
        pytest.skip("timing")
    H.load_plugin(MUTATIONS)
    names = os.environ.get("XP_TIME_ARMS", ",".join(ARMS)).split(",")
    if not H.is_sm107():
        names = [n for n in names if not ARMS[n][1]]
    logical, prepared = T16.weights16()
    gt = H.generator(1950 + batch)
    size = int(os.environ.get("XP_POOL", "112"))  # 112 -> ~79 distinct at B8 (the shared timing routes)
    pools = torch.randperm(R.E_ALL, generator=gt, device="cuda")[: 4 * size].view(4, size)
    routes = [R.random_route(batch, gt, pool=pools[i % 4]) for i in range(16)]
    xs = [T16.latent(batch, gt) for _ in routes]
    runners, rows = {}, {}
    for name in names:
        ex, out, buffers = timing_arm(name, prepared, batch)
        if ARMS[name][3]:  # plan pre-written for the first route's x (later calls read it stale: timing only)
            write_plan(xs[0], buffers["plan"])
        H.poison(out)
        ex.run(xs[0], routes[0][0])
        torch.cuda.synchronize()
        if ARMS[name][4]:
            T16.check_rows(f"time.{name}.b{batch}", out.view(batch, 16, -1),
                           M.expert_rows(logical, xs[0], routes[0][0]))
            rows[name] = out.clone()
            partner = BITWISE_PAIRS.get(name)
            if partner in rows:
                check_bitwise(f"time.{name}.b{batch}.vs_{partner}", rows[name], rows[partner])
        else:
            H.record({"kind": "control", "item": f"time.{name}.b{batch}", "outputs": "invalid by design"})
        runners[name] = ex
    distinct = sum(int(torch.unique(r[0]).numel()) for r in routes) / len(routes)
    calls = {k: [lambda ex=ex, x=x, r=r: ex.run(x, r[0]) for x, r in zip(xs, routes)] for k, ex in runners.items()}
    got = {k: [] for k in names}
    for k in names + names[::-1] + names + names[::-1]:
        got[k].append(H.cold_time_us(calls[k])["us_median"])
    for k in names:
        ex = runners[k]
        row = {"kind": "time", "item": f"xplan.time.b{batch}.{k}", "us_median": statistics.median(got[k]),
               "us_runs": got[k], "us_spread": max(got[k]) - min(got[k]), "distinct_mean": distinct,
               "grid": ex.grid_for(batch), "xtma": str(ex.xtma), "packed": ex.packed,
               "plan_writer": ex.prepared[batch][5] is not None and len(ex.prepared[batch][5]) > 1,
               "control": not ARMS[k][4] or ARMS[k][3], **H.device_facts()}
        H.record(row)
        print("TIME", row, flush=True)


@needs_gpu
def test_xplan_k16_fence(monkeypatch, tmp_path):
    """XK16: the Run form + XPLAN=1 prepare()-path PTX at B8 fences generic -> async proxy after the PDL wait."""
    H.load_plugin(MUTATIONS)
    if not H.is_sm107():
        pytest.skip("the Run D form is the packed K64 engine (SM107)")
    import test_k31_experts_x3_release as X3

    from sgl_nvidia_mega.kimi_k31.k3.cutedsl.moe import planned as PLN

    PLN.compile_early_plan(8, torch.cuda.current_device())  # cached first: the kept-PTX compile then dumps the experts kernel only
    _, prepared = T16.weights16()
    env = {"K31MK_EXP_XPLAN": "1", "K31MK_EXPERTS_X3_RELEASE": "3", "K31MK_EXPERTS_HFULL_BAR": "1",
           "K31MK_EXPERTS_PRODUCER_PACE": "240"}
    ex, ptx = X3.prepare_ptx(monkeypatch, tmp_path, prepared, 8, env)
    assert ex.xtma is True and ex.prepared[8][5] is not None, (ex.xtma, ex.prepared[8][5])
    w = ptx.find("griddepcontrol.wait")
    t = ptx.find("cp.async.bulk.tensor", w)
    seg = ptx[w:t] if 0 <= w < t else ""
    f, a = seg.find("fence.proxy.async.global;"), seg.find("mbarrier.arrive.expect_tx")
    rec = {"kind": "check", "item": "xk16.b8", "wait_at": w, "tma_at": t, "fence_in_seg": f, "expect_tx_in_seg": a,
           "tmas_after_wait": ptx.count("cp.async.bulk.tensor", w)}
    H.record(rec)
    print("CHECK", rec, flush=True)
    assert 0 <= w < t and 0 <= f < a, rec
