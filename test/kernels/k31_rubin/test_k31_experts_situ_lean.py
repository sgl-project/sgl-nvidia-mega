# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""K3.1 SiTU with the parameter reciprocals hoisted out of the per-element loop (``K31MK_EXP_SITU_LEAN``, default
off; the ``stock_lean`` form). Routed: the persistent experts' FC1 epilogue takes
rcp.approx(beta) and rcp.approx(linear_beta) once per thread and drops the identity FLT_MAX clamp. Shared: RouteSitu's
situ role takes div.rn(1, beta) and div.rn(1, linear_beta) once per thread. Same instructions on the same operands per
element, so the outputs must be bit-identical to the shipped form.

Checks (stated before any run; never loosened):
  L1  routed: lean vs shipped PersistentExperts at TP16 (same host-layout weights, x, ids, one process): 0 differing
      BF16 output cells on every call (2 eager + 2 CUDA-graph replays with new inputs), B1, 2, 4, 8, 16.
  T1  lean rows vs the FP64 reference (``_ref_moe16.expert_rows``): the T1 of test_k31_experts_tp16.
  L2  shared: lean vs shipped RouteSitu situ role (TP16 shared width 384, FP32 gate_up), B1, 2, 8, 16: 0 differing
      BF16 hidden cells; gate/up inputs N(0, s^2), s in {1, 4, 16} plus planted zeros and +-80.
  TS  lean shared hidden vs the FP64 SiTU (beta 4, linear_beta 25): every cell with |ref| >= 2^-60 within 1 BF16 ulp
      of BF16(RN(ref)) (ideas/approx g6: stock's approximate form is <= 0.503 BF16 ulp from the truth before the
      final BF16 rounding). Independent of the shipped kernel.

Mutations (fixed harness; each must make a base-passing test fail):
  sl_swap     routed hoisted form uses rcp(linear_beta) for the gate and rcp(beta) for the up branch   [L1, T1]
  sl_shared   shared hoisted form uses div.rn(1, linear_beta) for the gate branch                          [L2, TS]

Timing (``K31_TIME=1``, Rubin): experts per call at B8 (cold weights, 16 calls per graph) and the
RouteSitu situ role at B8 (16 calls per graph), arms shipped vs lean.
"""

import os
import statistics

import pytest
import torch

import _harness as H
import _ref_moe as R
import _ref_moe16 as M
import test_k31_experts_tp16 as T16

MATH = "kimi_k31/k3/cutedsl/moe/math.py"
SHARED = "kimi_k31/k3/cutedsl/moe/shared.py"
MUTATIONS = {
    "sl_swap": (MATH, "    g = situ_gate(gate, beta, inv_beta)\n    u = situ_up(up, linear_beta, inv_lb)\n",
                "    g = situ_gate(gate, beta, inv_lb)\n    u = situ_up(up, linear_beta, inv_beta)\n"),
    "sl_shared": (SHARED, "    \"\"\"``situ`` with ``inv_beta`` = div_rn(1, beta) and ``inv_lb`` = div_rn(1, linear_beta) computed once by the\n"
                          "    caller (the same IEEE division of the same operands: bitwise equal to ``situ``).\"\"\"\n"
                          "    g = situ_gate(gate, beta, inv_beta)\n",
                  "    \"\"\"``situ`` with ``inv_beta`` = div_rn(1, beta) and ``inv_lb`` = div_rn(1, linear_beta) computed once by the\n"
                  "    caller (the same IEEE division of the same operands: bitwise equal to ``situ``).\"\"\"\n"
                  "    g = situ_gate(gate, beta, inv_lb)\n"),
}

needs_gpu = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 10,
    reason="needs an SM100-family GPU",
)

SHARED_W = 384  # TP16 shared-expert intermediate per rank (6144 / 16)


def _er():
    from sgl_nvidia_mega.kimi_k31.k3.cutedsl.moe import experts_runtime as ER

    return ER


def engine(prepared, batch, lean):
    ER = _er()
    ex = ER.PersistentExperts(*prepared, beta=R.BETA, linear_beta=R.LIN, activation_floor=None, situ_lean=lean)
    assert ex.situ_lean == lean
    out = torch.empty((batch * 16, R.H_LAT), dtype=torch.bfloat16, device="cuda")
    ex.prepare(batch, ER.scratch(ex.tp, torch.device("cuda")), out)
    return ex, out


def bitwise(name, a, b):
    differ = int((a.view(torch.int16) != b.view(torch.int16)).sum())
    H.record({"kind": "check", "item": name, "cells_differ": differ, "cells": a.numel()})
    print("CHECK", name, "cells_differ", differ, flush=True)
    assert differ == 0, f"{name}: lean and shipped differ in {differ} cells"


@needs_gpu
@pytest.mark.parametrize("batch", (1, 2, 4, 8, 16), ids=lambda b: f"b{b}")
def test_lean_routed(batch):
    """L1 + T1."""
    H.load_plugin(MUTATIONS)
    logical, prepared = T16.weights16()
    (ex_s, out_s), (ex_l, out_l) = engine(prepared, batch, False), engine(prepared, batch, True)
    assert ex_s.prepared[batch][3] is not ex_l.prepared[batch][3], "lean and shipped share one compiled kernel"
    g = H.generator(760 + batch)
    x = torch.empty((batch, R.H_LAT), device="cuda")
    ids = torch.empty((batch, 16), dtype=torch.int32, device="cuda")

    def call():
        ex_s.run(x, ids)
        ex_l.run(x, ids)

    graph = None
    for n in range(4):
        xv = T16.latent(batch, g)
        i, _ = R.random_route(batch, g)
        x.copy_(xv)
        ids.copy_(i)
        H.poison(out_s, out_l)
        if n < 2:
            call()
        else:
            graph = graph or T16.graph_of(call)
            H.poison(out_s, out_l)
            graph.replay()
        torch.cuda.synchronize()
        bitwise(f"lean.routed.b{batch}.call{n}", out_l, out_s)
        T16.check_rows(f"lean.routed.b{batch}.call{n}", out_l.view(batch, 16, -1), M.expert_rows(logical, xv, i))


class SituRole:
    """RouteSitu's situ role alone (TP16 shared width), launched with every argument of the routing kernel."""

    def __init__(self, batch, lean):
        from sgl_nvidia_mega.kimi_k31.k3.cutedsl.common.host import stream, to_cute
        from sgl_nvidia_mega.kimi_k31.k3.cutedsl.moe import front as F

        dev = torch.cuda.current_device()
        self.code = F.compile_route(batch, SHARED_W, True, (False, False, True), dev, deadline_ns=0, fp32=True,
                                    situ_lean=lean)
        f32 = dict(dtype=torch.float32, device="cuda")
        self.gate_up = torch.zeros((batch, 2 * SHARED_W), **f32)
        self.hidden = torch.zeros((batch, SHARED_W), dtype=torch.bfloat16, device="cuda")
        keep = (torch.zeros((batch, 896), **f32), torch.zeros(896, **f32),
                torch.zeros((batch, 16), dtype=torch.int32, device="cuda"),
                torch.zeros((batch, 16), dtype=torch.bfloat16, device="cuda"),
                torch.zeros((1, 64, 7168 + 2 * 896), dtype=torch.bfloat16, device="cuda"),
                torch.zeros((batch, 3584), **f32))
        self.keep = keep
        self.args = tuple(map(to_cute, keep)) + (to_cute(self.gate_up), to_cute(self.hidden))
        import cutlass as c

        self.scalars = (c.Float32(1.0), c.Float32(R.BETA), c.Float32(R.LIN))
        self.stream = stream

    def __call__(self):
        self.code(*self.args, *self.scalars, self.stream(self.hidden.device))


def situ_inputs(batch, g, scale):
    v = torch.randn((batch, 2 * SHARED_W), generator=g, device="cuda") * scale
    v[:, ::53] = 0.0
    v[:, 7::97] = 80.0
    v[:, 11::89] = -80.0
    return v


def situ_ref(gate_up):
    gate, up = gate_up[:, :SHARED_W].double(), gate_up[:, SHARED_W:].double()
    return R.situ(gate, up)


def bf16_ulp_distance(a, b):
    """Ordered distance of BF16 bit patterns (monotone integer line)."""
    def ordered(x):
        i = x.view(torch.int16).to(torch.int32)
        return torch.where(i < 0, -32768 - i, i)
    return (ordered(a) - ordered(b)).abs()


@needs_gpu
@pytest.mark.parametrize("batch", (1, 2, 8, 16), ids=lambda b: f"b{b}")
def test_lean_shared(batch):
    """L2 + TS on RouteSitu's situ role."""
    H.load_plugin(MUTATIONS)
    roles = SituRole(batch, False), SituRole(batch, True)
    assert roles[0].code is not roles[1].code, "lean and shipped share one compiled kernel"
    g = H.generator(860 + batch)
    for n, scale in enumerate((1.0, 4.0, 16.0)):
        v = situ_inputs(batch, g, scale)
        for r in roles:
            r.gate_up.copy_(v)
            H.poison(r.hidden)
            r()
        torch.cuda.synchronize()
        bitwise(f"lean.shared.b{batch}.s{scale:g}", roles[1].hidden, roles[0].hidden)
        ref = situ_ref(v)
        keep = ref.abs() >= 2.0 ** -60
        dist = bf16_ulp_distance(roles[1].hidden, ref.float().to(torch.bfloat16))
        worst = int(dist[keep].max()) if bool(keep.any()) else 0
        rec = {"kind": "check", "item": f"lean.shared.b{batch}.s{scale:g}.ts", "max_bf16_ulp": worst,
               "cells": int(keep.sum())}
        H.record(rec)
        print("CHECK", rec, flush=True)
        assert bool(torch.isfinite(roles[1].hidden.float()).all()), "non-finite shared hidden"
        assert worst <= 1, f"TS: shared hidden {worst} BF16 ulp from the FP64 SiTU"


# ------------------------------------------------------------------------------ timing (Rubin)
@needs_gpu
def test_time_lean():
    """Pre-registered timing: experts B8 (cold weights) and RouteSitu situ role B8, arms
    shipped / lean, A B B A A B B A on this GPU."""
    if os.environ.get("K31_TIME") != "1" or os.environ.get("K31_MUTATION"):
        pytest.skip("timing")
    H.load_plugin(MUTATIONS)
    import test_k31_experts_depad as D

    batch = 8
    logical, prepared = T16.weights16()
    gt = H.generator(990)
    routes = D.time_routes(batch, gt)
    xs = [T16.latent(batch, gt) for _ in routes]
    runners = {}
    for name, lean in (("s16", False), ("l16", True)):
        ex, out = engine(prepared, batch, lean)
        H.poison(out)
        ex.run(xs[0], routes[0][0])
        torch.cuda.synchronize()
        T16.check_rows(f"time.{name}", out.view(batch, 16, -1), M.expert_rows(logical, xs[0], routes[0][0]))
        runners[name] = ex
    roles = {"s_situ": SituRole(batch, False), "l_situ": SituRole(batch, True)}
    v = situ_inputs(batch, gt, 4.0)
    for r in roles.values():
        r.gate_up.copy_(v)
    calls = {k: [lambda ex=ex, x=x, r=r: ex.run(x, r[0]) for x, r in zip(xs, routes)] for k, ex in runners.items()}
    calls.update({k: [r] * 16 for k, r in roles.items()})
    order = ["s16", "l16", "l16", "s16"] * 2 + ["s_situ", "l_situ", "l_situ", "s_situ"] * 2
    got = {k: [] for k in calls}
    for k in order:
        got[k].append(H.cold_time_us(calls[k])["us_median"])
    for k in calls:
        row = {"kind": "time", "item": f"lean.time.b{batch}.{k}", "us_median": statistics.median(got[k]),
               "us_runs": got[k], "us_spread": max(got[k]) - min(got[k]), **H.device_facts()}
        H.record(row)
        print("TIME", row, flush=True)
