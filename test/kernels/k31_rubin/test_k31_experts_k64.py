# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""K3.1 TP16 routed experts with packed FP4 weights on Rubin's K64 MMAs (``K31MK_EXPERTS_PACKED_K64``, default off;
TP16 only). The packed path is the vendored ``RubinExpertsKernel(packed=True)``: the MXFP4
weight bytes land as plain UINT8 TMA boxes in 16 KB slots and feed dense K=64 ``kind::mxf8f6f4.block_scale`` MMAs
(instruction-descriptor bit 31), which read packed E2M1 from SMEM and apply two per-32 scales per row (bytes ID and
ID + 1; checked on the target GPU by a hardware control). Same inputs and FP32 accumulator as the shipped K32
engine: only the accumulation grouping inside the MMA changes (K64 instead of K32 groups).

Checks (stated before any run; never loosened):
  K1  packed vs shipped (same weights, x, ids, one process): >= 99% of the BF16 cells of a call bitwise equal and
      every cell within 2 BF16 ulp (ordered int16 distance). Basis: a reorder of FP32 partial sums moves an FC1 sum by
      ~1e-7 relative; a hidden E4M3 code flips only when an FC1 value sits on a rounding midpoint.
  K2  packed vs the FP64 reference (``_ref_moe16.expert_rows``, logical weights): T1 of test_k31_experts_tp16
      (Frobenius rel-L2 <= 1e-2, every row <= 3e-2, all cells finite). Outputs NaN-poisoned before every call.

Mutations (fixed harness; each must make a base-passing test fail):
  k64_sfid   SFA ID 0 <-> 2 on the K64 MMAs (the other K64 group's scale byte pair)              [K1, K2]
  k64_aoff   p8's padded A offset (64 B per K64) on the packed slot                                 [K1, K2]
  k64_chunk  the streamed producer loads FC1 stage 2*ks of the packed map (B4-8)                    [K1, K2]
  k64_chunk_sched  the same in the scheduled producer (B1-3)                                        [K1, K2]
"""

import os
import statistics

import pytest
import torch

import _harness as H
import _ref_moe as R
import _ref_moe16 as M
import test_k31_experts_tp16 as T16

PE = "kimi_k31/k3/cutedsl/moe/experts.py"
MUTATIONS = {
    "k64_sfid": (PE, 'code += [f"mov.b32 w{k},0x{142738048 | 1 << 31 | k << 29 | k << 4:08x};" for k in (0, 2)]',
                 'code += [f"mov.b32 w{k},0x{142738048 | 1 << 31 | (k ^ 2) << 29 | k << 4:08x};" for k in (0, 2)]'),
    "k64_aoff": (PE, "        if packed:\n            a = k\n",
                 "        if packed:\n            a = k // 4 * 1024 + k % 4 * 2\n"),
    "k64_chunk": (PE, "tma = (w13, w13_scales, row, ks if self.packed else 2 * ks, offset, Int32(1024))",
                  "tma = (w13, w13_scales, row, 2 * ks, offset, Int32(1024))"),
    "k64_chunk_sched": (PE, "row, chunk = (expert * self.w13_rows + r * 128, ks if self.packed else 2 * ks)",
                        "row, chunk = (expert * self.w13_rows + r * 128, 2 * ks)"),
}

# One validated slot count drives the compiled kernel and the maps.
MUTATIONS["k64_noslotcheck"] = ("kimi_k31/k3/cutedsl/moe/experts_runtime.py",
                                "    if slots not in PACKED_STAGES:\n", "    if False:\n")
MUTATIONS["k64_noarchcheck"] = ("kimi_k31/k3/cutedsl/moe/experts_runtime.py",
                                "    if torch.cuda.get_device_capability(device) != (10, 7):\n", "    if False:\n")

needs_sm107 = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() != (10, 7),
    reason="packed K64 MMAs need VR200 (sm_107)",
)

K1_BITWISE, K1_ULP = 0.99, 2


def engine(prepared, batch, packed):
    from sgl_nvidia_mega.kimi_k31.k3.cutedsl.moe import experts_runtime as ER

    ex = ER.PersistentExperts(*prepared, beta=R.BETA, linear_beta=R.LIN, activation_floor=None, packed=packed)
    assert ex.packed == packed, (ex.packed, packed)
    out = torch.empty((batch * 16, R.H_LAT), dtype=torch.bfloat16, device="cuda")
    ex.prepare(batch, ER.scratch(ex.tp, torch.device("cuda")), out)
    return ex, out


def ordered(bits):
    """BF16 bit patterns (int16) on a monotone integer line: ulp distance = |ordered(a) - ordered(b)|."""
    b = bits.to(torch.int32)
    return torch.where(b < 0, -32768 - b, b)


def check_k1(name, packed_rows, shipped_rows):
    a, b = packed_rows.view(torch.int16), shipped_rows.view(torch.int16)
    same = float((a == b).double().mean())
    ulp = int((ordered(a) - ordered(b)).abs().max())
    stats = {"kind": "check", "item": name, "k1_bitwise_frac": same, "k1_max_ulp": ulp,
             "cells": a.numel(), "cells_differ": int((a != b).sum())}
    print("CHECK", stats, flush=True)
    H.record(stats)
    assert bool(torch.isfinite(packed_rows.float()).all()), f"{name}: non-finite packed rows"
    assert same >= K1_BITWISE and ulp <= K1_ULP, f"{name}: K1 bitwise {same:.5f} (>= {K1_BITWISE}) max ulp {ulp} (<= 2)"


CASES = [(b, 6) for b in (1, 2, 3, 4, 5, 8)] + [(8, 5)]


@needs_sm107
@pytest.mark.parametrize("batch,stages", CASES, ids=lambda v: str(v))
def test_k64(batch, stages):
    """K1 and K2 on the packed K64 engine vs the shipped K32 engine: two eager calls, then two CUDA-graph replays."""
    H.load_plugin(MUTATIONS)
    logical, prepared = T16.weights16()
    ship, out_s = engine(prepared, batch, 0)
    pack, out_p = engine(prepared, batch, stages)
    g = H.generator(560 + 10 * batch + stages)
    x = torch.empty((batch, R.H_LAT), device="cuda")
    ids = torch.empty((batch, 16), dtype=torch.int32, device="cuda")

    def call():
        ship.run(x, ids)
        pack.run(x, ids)

    graph = None
    for n in range(4):
        xv = T16.latent(batch, g)
        i, _ = R.random_route(batch, g, distinct=batch == 1)
        x.copy_(xv)
        ids.copy_(i)
        H.poison(out_s, out_p)
        if n < 2:
            call()
        else:
            graph = graph or T16.graph_of(call)
            H.poison(out_s, out_p)
            graph.replay()
        torch.cuda.synchronize()
        name = f"k64.b{batch}.s{stages}.call{n}"
        rows = out_p.view(batch, 16, -1)
        check_k1(name, out_p, out_s)
        T16.check_rows(name, rows, M.expert_rows(logical, xv, i))
    H.record({"kind": "resources", "item": f"k64.b{batch}.s{stages}", "grid": pack.grid_for(batch),
              "distinct": int(torch.unique(ids).numel()), **H.function_resources(pack.prepared[batch][3]),
              **H.device_facts()})


# ------------------------------------------------------------------------------ timing (Rubin)
def time_routes(batch, gt):
    size = int(os.environ.get("K64_POOL", "112"))  # 112 -> ~79 distinct at B8 (the shared timing routes)
    pools = torch.randperm(R.E_ALL, generator=gt, device="cuda")[: 4 * size].view(4, size)
    return [R.random_route(batch, gt, pool=pools[i % 4]) for i in range(16)]


def timing_arm(name, prepared, batch):
    """a16: shipped TP16 engine as the MK runs it at B8; p16 / p16s5: packed with 6 / 5 slots. Suffix g: the grid
    derived for H.bundle_grid_sms() SMs (``K31MK_EXPERTS_GRID_SMS``, arm G)."""
    if name.endswith("g"):
        os.environ["K31MK_EXPERTS_GRID_SMS"] = str(H.bundle_grid_sms())
        try:
            return timing_arm(name[:-1], prepared, batch)
        finally:
            del os.environ["K31MK_EXPERTS_GRID_SMS"]
    return engine(prepared, batch, {"a16": 0, "p16": 6, "p16s5": 5}[name])


@needs_sm107
def test_time_k64():
    """Pre-registered timing: arms of ``K64_TIME_ARMS``, A..Z Z..A A..Z Z..A on this GPU; each arm first
    passes T1 on the first timing route."""
    if os.environ.get("K31_TIME") != "1" or os.environ.get("K31_MUTATION"):
        pytest.skip("timing")
    H.load_plugin(MUTATIONS)
    batch = 8
    names = os.environ.get("K64_TIME_ARMS", "a16,p16,p16s5,a16g,p16g").split(",")
    logical, prepared = T16.weights16()
    gt = H.generator(900 + batch)
    routes = time_routes(batch, gt)
    xs = [T16.latent(batch, gt) for _ in routes]
    runners = {}
    for name in names:
        ex, out = timing_arm(name, prepared, batch)
        H.poison(out)
        ex.run(xs[0], routes[0][0])
        torch.cuda.synchronize()
        T16.check_rows(f"time.{name}.b{batch}", out.view(batch, 16, -1), M.expert_rows(logical, xs[0], routes[0][0]))
        runners[name] = ex
    distinct = sum(int(torch.unique(r[0]).numel()) for r in routes) / len(routes)
    calls = {k: [lambda ex=ex, x=x, r=r: ex.run(x, r[0]) for x, r in zip(xs, routes)] for k, ex in runners.items()}
    got = {k: [] for k in names}
    for k in names + names[::-1] + names + names[::-1]:
        got[k].append(H.cold_time_us(calls[k])["us_median"])
    for k in names:
        row = {"kind": "time", "item": f"k64.time.b{batch}.{k}", "us_median": statistics.median(got[k]),
               "us_runs": got[k], "us_spread": max(got[k]) - min(got[k]), "distinct_mean": distinct,
               "grid": runners[k].grid_for(batch), **H.device_facts()}
        H.record(row)
        print("TIME", row, flush=True)


# ------------------------------------------------------------------------------ NV1 (config validation)
@pytest.mark.skipif(not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 10,
                    reason="needs an SM100-family GPU")
def test_packed_slots_validated():
    """NV1: a slot count outside (5, 6, 7) (e.g. -1, which compiled the unpacked kernel but bound the packed maps and
    hung on the FC2 expect_tx) and packed on a non-SM107 device must be refused at construction; 0 means shipped."""
    H.load_plugin(MUTATIONS)
    from sgl_nvidia_mega.kimi_k31.k3.cutedsl.moe import experts_runtime as ER

    _, prepared = T16.weights16()
    for bad in (-1, 1, 8):
        with pytest.raises(ValueError):
            ER.PersistentExperts(*prepared, beta=R.BETA, linear_beta=R.LIN, activation_floor=None, packed=bad)
    assert ER.PersistentExperts(*prepared, beta=R.BETA, linear_beta=R.LIN, activation_floor=None, packed=0).packed == 0
    if torch.cuda.get_device_capability() == (10, 7):
        ex = ER.PersistentExperts(*prepared, beta=R.BETA, linear_beta=R.LIN, activation_floor=None, packed=6)
        assert ex.packed == 6
    else:
        with pytest.raises(ValueError):
            ER.PersistentExperts(*prepared, beta=R.BETA, linear_beta=R.LIN, activation_floor=None, packed=6)
