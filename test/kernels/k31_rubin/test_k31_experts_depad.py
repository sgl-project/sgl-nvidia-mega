# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""K3.1 TP16 routed experts reading FC2 weights from the de-padded load-time copy (``K31MK_EXPERTS_W2_DEPAD``,
default off). The host pads the TP16 per-rank intermediate 192 -> 256 for FlashInfer, so every FC2 row
is 96 live bytes plus 32 zero bytes. The engine's FC2 map reads two 64-byte chunks at bytes 0 and 32 of a row, so it
never requests the padding, but the padding still sits in the 128-byte lines the reads touch. The relayout copies the
live bytes into 96-byte rows and points map 1 at them; kernel code is unchanged.

Checks (stated before any run; never loosened):
  D0  relayout round trip on the host layout (``_ref_moe16.host_layout16`` of random logical weights):
      ``repad_w2(depad_w2(W2))`` equals W2 byte for byte; ``depad_w2`` keeps exactly the logical FC2 bytes
      (``live[e]`` equals the logical w2 rows in the host's row order, an independent derivation from
      ``_ref_moe16``); and ``depad_w2`` refuses a W2 with one planted nonzero padding byte.
  D1  de-padded vs shipped engine (same host-layout weights, x, ids, one process): every BF16 output cell bitwise
      equal on every call (2 eager calls + 2 CUDA-graph replays with new inputs); PersistentExperts B1, 2, 4, 8, 16
      (B16: grouped plan path) and ClusterExpertsTP16 B1, 2. Basis: the boxes land the same bytes in SMEM, so every
      later instruction sees identical operands.
  T1  de-padded rows vs the FP64 reference (``_ref_moe16.expert_rows``, logical weights), the T1 of
      test_k31_experts_tp16 (Frobenius rel-L2 <= 1e-2, every row <= 3e-2, all cells finite; outputs NaN-poisoned).
      Independent of the shipped engine (D1 alone would pass if both engines shared a bug).

Mutations (fixed harness; host-side code: the kernel PTX is unchanged by
design, so each is shown live by a base-passing test that fails under it):
  dp_stride   map 1 keeps the 128-byte row stride on the 96-byte copy (rows read across row boundaries)  [D1, T1]
  dp_offset   depad_w2 keeps bytes 32-127 instead of 0-95                                                [D0, D1, T1]
  dp_nocheck  depad_w2 drops its zero-padding check                                                      [D0]
  dp_repad    repad_w2 writes the live bytes at offset 32                                                [D0]

Timing (``K31_TIME=1``, Rubin, pre-registered): B8 (and B16 info) per-call device time with
cold weights, arms a16 (shipped), d16 (de-padded), a16g / d16g (grid derived for the bundle's SM count, H.bundle_grid_sms()).
"""

import os
import statistics

import pytest
import torch

import _harness as H
import _ref_moe as R
import _ref_moe16 as M
import test_k31_experts_tp16 as T16

ER_FILE = "kimi_k31/k3/cutedsl/moe/experts_runtime.py"
MUTATIONS = {
    "dp_stride": (ER_FILE, "fc2 = (self.w2_live, 3584, W2_LIVE_BYTES, 2, 32, 2, False)",
                  "fc2 = (self.w2_live, 3584, W2_ROW_BYTES, 2, 32, 2, False)"),
    "dp_offset": (ER_FILE, "return rows[..., :W2_LIVE_BYTES].contiguous()",
                  "return rows[..., W2_ROW_BYTES - W2_LIVE_BYTES:].contiguous()"),
    "dp_nocheck": (ER_FILE, "    if bool(rows[..., W2_LIVE_BYTES:].any()):\n",
                   "    if False:\n"),
    "dp_repad": (ER_FILE, "out[..., :W2_LIVE_BYTES] = live", "out[..., W2_ROW_BYTES - W2_LIVE_BYTES:] = live"),
}

needs_gpu = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 10,
    reason="needs an SM100-family GPU",
)


def _er():
    from sgl_nvidia_mega.kimi_k31.k3.cutedsl.moe import experts_runtime as ER

    return ER


def engines(prepared, batch):
    """(shipped, de-padded) PersistentExperts on the same weights, each prepared for ``batch``."""
    ER = _er()
    pair = []
    for depad in (False, True):
        ex = ER.PersistentExperts(*prepared, beta=R.BETA, linear_beta=R.LIN, activation_floor=None, w2_depad=depad)
        assert ex.w2_depad == depad, (ex.w2_depad, depad)
        out = torch.empty((batch * 16, R.H_LAT), dtype=torch.bfloat16, device="cuda")
        ex.prepare(batch, ER.scratch(ex.tp, torch.device("cuda")), out)
        pair.append((ex, out))
    # Engagement: the de-padded engine's FC2 descriptor differs from the shipped one (base address and stride).
    assert not torch.equal(pair[0][0].maps[128:256], pair[1][0].maps[128:256]), "map 1 unchanged by w2_depad"
    assert torch.equal(pair[0][0].maps[:128], pair[1][0].maps[:128]), "map 0 (w13) must not change"
    return pair


def check_bitwise(name, a, b):
    differ = int((a.view(torch.int16) != b.view(torch.int16)).sum())
    H.record({"kind": "check", "item": name, "d1_cells_differ": differ, "cells": a.numel()})
    print("CHECK", name, "d1_cells_differ", differ, flush=True)
    assert differ == 0, f"{name}: de-padded and shipped rows differ in {differ} cells"


@needs_gpu
def test_depad_roundtrip():
    """D0 on the host layout of random logical weights (all 896 experts)."""
    H.load_plugin(MUTATIONS)
    ER = _er()
    logical, prepared = T16.weights16()
    W2 = prepared[2]
    assert W2.dtype == torch.uint8 and tuple(W2.shape) == (R.E_ALL, R.H_LAT, 128), (W2.dtype, W2.shape)
    live = ER.depad_w2(W2)
    assert tuple(live.shape) == (R.E_ALL, R.H_LAT, 96) and live.is_contiguous()
    back = ER.repad_w2(live)
    rt_bad = int((back != W2).sum())
    # Independent derivation (not through depad_w2): the host permutes FC2 rows only (FlashInfer's epilogue-tile
    # row permutation, as ``_ref_moe16.host_layout16`` applies it), so the live bytes are the logical rows
    # [E, 3584, 96] in that order.
    from flashinfer.fused_moe.core import get_w2_permute_indices_with_cache

    order = get_w2_permute_indices_with_cache({}, W2[0], 128).to("cuda")
    logical_bad = int((live != logical[2][:, order, :]).sum())
    planted = W2.clone()
    planted[R.E_ALL // 2, 1234, 100] = 1
    refused = False
    try:
        ER.depad_w2(planted)
    except ValueError:
        refused = True
    rec = {"kind": "check", "item": "d0", "roundtrip_bytes_differ": rt_bad, "live_vs_logical_bytes_differ": logical_bad,
           "planted_padding_refused": refused}
    H.record(rec)
    print("CHECK", rec, flush=True)
    assert rt_bad == 0, f"D0: repad(depad(W2)) differs from W2 in {rt_bad} bytes"
    assert logical_bad == 0, f"D0: live bytes differ from the logical FC2 rows in {logical_bad} bytes"
    assert refused, "D0: depad_w2 accepted a nonzero padding byte"


@needs_gpu
@pytest.mark.parametrize("batch", (1, 2, 4, 8, 16), ids=lambda b: f"b{b}")
def test_depad_persistent(batch):
    """D1 + T1 on PersistentExperts at TP16."""
    H.load_plugin(MUTATIONS)
    logical, prepared = T16.weights16()
    (ex_a, out_a), (ex_d, out_d) = engines(prepared, batch)
    g = H.generator(560 + batch)
    x = torch.empty((batch, R.H_LAT), device="cuda")
    ids = torch.empty((batch, 16), dtype=torch.int32, device="cuda")

    def call():
        ex_a.run(x, ids)
        ex_d.run(x, ids)

    graph = None
    for n in range(4):
        xv = T16.latent(batch, g)
        i, _ = R.random_route(batch, g)
        x.copy_(xv)
        ids.copy_(i)
        H.poison(out_a, out_d)
        if n < 2:
            call()
        else:
            graph = graph or T16.graph_of(call)
            H.poison(out_a, out_d)
            graph.replay()
        torch.cuda.synchronize()
        check_bitwise(f"depad.persistent16.b{batch}.call{n}", out_d, out_a)
        T16.check_rows(f"depad.persistent16.b{batch}.call{n}", out_d.view(batch, 16, -1), M.expert_rows(logical, xv, i))


@needs_gpu
@pytest.mark.parametrize("batch", (1, 2), ids=lambda b: f"b{b}")
def test_depad_cluster(batch):
    """D1 + T1 on ClusterExpertsTP16 (the B1/B2 chain reads FC2 through the same map 1)."""
    H.load_plugin(MUTATIONS)
    ER = _er()
    logical, prepared = T16.weights16()
    engines_ = [ER.PersistentExperts(*prepared, beta=R.BETA, linear_beta=R.LIN, activation_floor=None, w2_depad=d)
                for d in (False, True)]
    g = H.generator(660 + batch)
    src = torch.empty((batch, R.H_LAT), device="cuda")
    ids_src = torch.empty((batch, 16), dtype=torch.int32, device="cuda")
    xs = [torch.empty((batch, R.H_LAT), device="cuda") for _ in range(2)]
    ids = [torch.empty((batch, 16), dtype=torch.int32, device="cuda") for _ in range(2)]
    outs = [torch.empty((batch * 16, R.H_LAT), dtype=torch.bfloat16, device="cuda") for _ in range(2)]
    runs = [T16.Cluster16(e).bind(xs[k], ids[k], outs[k]) for k, e in enumerate(engines_)]

    def call():  # the kernel re-arms its ids to -1: refill before every launch
        for k in range(2):
            xs[k].copy_(src)
            ids[k].copy_(ids_src)
            runs[k]()

    graph = None
    for n in range(4):
        xv = T16.latent(batch, g)
        i, _ = R.random_route(batch, g, distinct=batch == 1)
        src.copy_(xv)
        ids_src.copy_(i)
        H.poison(*outs)
        if n < 2:
            call()
        else:
            graph = graph or T16.graph_of(call)
            H.poison(*outs)
            graph.replay()
        torch.cuda.synchronize()
        check_bitwise(f"depad.cluster16.b{batch}.call{n}", outs[1], outs[0])
        T16.check_rows(f"depad.cluster16.b{batch}.call{n}", outs[1].view(batch, 16, -1), M.expert_rows(logical, xv, i))


# ------------------------------------------------------------------------------ timing (Rubin)
def time_routes(batch, gt):
    size = int(os.environ.get("DP_POOL", "112"))  # 112 -> ~79 distinct at B8 (the shared timing routes)
    pools = torch.randperm(R.E_ALL, generator=gt, device="cuda")[: 4 * size].view(4, size)
    return [R.random_route(batch, gt, pool=pools[i % 4]) for i in range(16)]


def timing_arm(name, prepared, batch):
    """a16: shipped; d16: de-padded FC2; suffix g: grid derived for H.bundle_grid_sms() SMs (arm G)."""
    ER = _er()
    if name.endswith("g"):
        os.environ["K31MK_EXPERTS_GRID_SMS"] = str(H.bundle_grid_sms())
        try:
            return timing_arm(name[:-1], prepared, batch)
        finally:
            del os.environ["K31MK_EXPERTS_GRID_SMS"]
    ex = ER.PersistentExperts(*prepared, beta=R.BETA, linear_beta=R.LIN, activation_floor=None,
                              w2_depad={"a16": False, "d16": True}[name])
    out = torch.empty((batch * 16, R.H_LAT), dtype=torch.bfloat16, device="cuda")
    ex.prepare(batch, ER.scratch(ex.tp, torch.device("cuda")), out)
    return ex, out


@needs_gpu
@pytest.mark.parametrize("batch", (8, 16), ids=lambda b: f"b{b}")
def test_time_depad(batch):
    """Pre-registered timing: arms of ``DP_TIME_ARMS``, A..Z Z..A A..Z Z..A on this GPU; each
    arm first passes T1 on the first timing route, and every arm's rows equal the first arm's bit for bit."""
    if os.environ.get("K31_TIME") != "1" or os.environ.get("K31_MUTATION"):
        pytest.skip("timing")
    H.load_plugin(MUTATIONS)
    names = os.environ.get("DP_TIME_ARMS", "a16,d16,a16g,d16g").split(",")
    logical, prepared = T16.weights16()
    gt = H.generator(950 + batch)
    routes = time_routes(batch, gt)
    xs = [T16.latent(batch, gt) for _ in routes]
    runners, first = {}, None
    for name in names:
        ex, out = timing_arm(name, prepared, batch)
        H.poison(out)
        ex.run(xs[0], routes[0][0])
        torch.cuda.synchronize()
        T16.check_rows(f"time.{name}.b{batch}", out.view(batch, 16, -1), M.expert_rows(logical, xs[0], routes[0][0]))
        if first is None:
            first = out.clone()
        else:
            check_bitwise(f"time.{name}.b{batch}.vs_{names[0]}", out, first)
        runners[name] = ex
    distinct = sum(int(torch.unique(r[0]).numel()) for r in routes) / len(routes)
    calls = {k: [lambda ex=ex, x=x, r=r: ex.run(x, r[0]) for x, r in zip(xs, routes)] for k, ex in runners.items()}
    got = {k: [] for k in names}
    for k in names + names[::-1] + names + names[::-1]:
        got[k].append(H.cold_time_us(calls[k])["us_median"])
    for k in names:
        row = {"kind": "time", "item": f"depad.time.b{batch}.{k}", "us_median": statistics.median(got[k]),
               "us_runs": got[k], "us_spread": max(got[k]) - min(got[k]), "distinct_mean": distinct,
               "grid": runners[k].grid_for(batch), "w2_depad": runners[k].w2_depad, **H.device_facts()}
        H.record(row)
        print("TIME", row, flush=True)
