# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""The re-planned routed-expert engine (``moe/planned.py``, ``PlannedExperts``) at the TP8
per-rank shapes, B1-B8, against the same independent references and tolerances as
test_k31_experts.py (``_ref_moe``: FP64 stock method from the logical weights; FlashInfer
``trtllm_fp4_block_scale_routed_moe`` as the stock yardstick).

Tolerances (copied unchanged from test_k31_experts.py, stated before any run):
  T1 per-expert BF16 rows vs FP64: Frobenius rel-L2 <= 1e-2, every row <= 3e-2, all finite
     (outputs NaN-poisoned before each call, so a row nobody wrote fails).
  T2 weighted sum vs FP64, against FlashInfer on the same inputs: rel(MK) <= 1.25 rel(FI)
     + 1e-3 and max|MK - ref| <= 2 max|FI - ref| + rms(ref) / 256.
  Shipped parity (reported, not gated): the shipped engine splits FC1 K across CTAs and
     adds the partials in FP32 in another order, so a few hidden E4M3 roundings may differ;
     the count of differing BF16 output cells is recorded.
Every case runs 2 eager calls then 2 CUDA-graph replays, each with new inputs, so the
cross-call protocol (parities, claim counters, hidden images) is exercised 4 times.

Routing cases: random top-16 (B1-B8; ~99 distinct experts at B8), all-distinct (B8: 128
units, 768 FC1 tiles, more tiles than CTAs: 3.6 waves), and few-distinct (B8, every
token on the same 16 experts: 96 FC1 tiles, fewer than CTAs).

Mutations (each must fail; the bug class each stands for):
  pl_drop_kstep    FC1 tiles skip K step 13 (dropped stage / lost K part)
  pl_drop_pair     the last claimed FC2 pair is never issued (dropped tile: rows stay NaN)
  pl_fc1_slice     FC1 reads the next N tile's rows (wrong expert slice)
  pl_fc2_expert    FC2 streams the next distinct expert's weights (wrong expert)
  pl_stale_claim   the claim counter is not re-armed (stale flag: call 3 finds it spent)
  pl_stale_hidden  hidden images are not re-armed (stale flag: FC2 reads 2-calls-old hidden)
  pl_stale_parity  the planned call counter does not advance (parity never flips)
  pl_stale_tileclaim  the FC1 tile counter is not re-armed (call 3 processes only the first
                   wave of tiles; its FC2 then waits forever on missing hidden images: caught by
                   the run's timeout, so it is run alone with K31_TIMEOUT=300)
  pe_situ_swap     shared SiTU operand order (shows T1 sees the shared epilogue)

Timing (``K31_TIME=1``, unmutated): cold weights (calls cycle 4 disjoint 224-expert pools;
B1: 56 calls over disjoint 16-expert sets), CUDA-graph replay, median of 20, ABBA x2 on
one GPU: shipped PersistentExperts at its grid, planned at the same grid, planned at one CTA per SM.
"""

import os

import pytest
import torch

import _harness as H
import _ref_moe as R
import test_k31_experts as TE

PL = "kimi_k31/k3/cutedsl/moe/planned.py"
PE = "kimi_k31/k3/cutedsl/moe/experts.py"
MUTATIONS = {
    "pl_drop_kstep": (PL, "        for ks in cutlass.range(lo, hi, unroll=1):\n            c = Int32(ks == 0) | Int32(ks == 13) << 1",
                      "        for ks in cutlass.range(lo, min(hi, 13), unroll=1):\n            c = Int32(ks == 0) | Int32(ks == 12) << 1"),
    "pl_drop_pair": (PL, "            active = q < pairs\n", "            active = q < pairs - 1\n"),
    "pl_fc1_slice": (PL, "        row = expert * self.w13_rows + r * 128\n",
                     "        row = expert * self.w13_rows + (r + 1) % 6 * 128\n"),
    "pl_fc2_expert": (PL, "                expert = experts[unit]\n                work = (maps.iterator, expert, unit, 2 * pair)",
                      "                expert = experts[(unit + 1) % distinct]\n                work = (maps.iterator, expert, unit, 2 * pair)"),
    "pl_stale_claim": (PL, "            store_words(self.claim_counter(workspace, parity ^ 1), (Uint32(4294967295),))\n",
                       "            pass\n"),
    "pl_stale_hidden": (PL, "        for i in range(lo + lane, hi, 32):\n            store_words(other + i * 16, (Uint32(4294967295),) * 4)\n",
                        "        for i in range(lo + lane, lo + lane, 32):\n            store_words(other + i * 16, (Uint32(4294967295),) * 4)\n"),
    "pl_stale_parity": (PL, "        calls[cta] = call + 1\n        if cta < MAX_CTAS - self.grid:\n            calls[self.grid + cta] = call + 1",
                        "        calls[cta] = call"),
    "pl_stale_tileclaim": (PL, "            store_words(self.claim_counter(workspace, 2 + (parity ^ 1)), (Uint32(4294967295),))\n",
                           "            pass\n"),
    "pe_situ_swap": TE.MUTATIONS["pe_situ_swap"],
}

needs_gpu = TE.needs_gpu


def planned(prepared, batches, grid=None, stages=None, xplan=None, wide=False, chain2=None):
    """``W4_STAGES`` / ``W4_XPLAN`` select the variant the correctness suite runs."""
    from sgl_nvidia_mega.kimi_k31.k3.cutedsl.moe import planned as PLN

    stages = stages or int(os.environ.get("W4_STAGES", "5"))
    xplan = bool(int(os.environ.get("W4_XPLAN", "0"))) if xplan is None else xplan
    chain2 = bool(int(os.environ.get("W4_CHAIN2", "0"))) if chain2 is None else chain2
    ex = PLN.PlannedExperts(*prepared, beta=R.BETA, linear_beta=R.LIN, activation_floor=None, stages=stages,
                            xplan=xplan, batches="1-8:1", wide=wide, chain2=chain2)
    if grid is not None:
        for b in batches:
            ex.grids[b] = grid
    buffers = PLN.scratch(8, torch.device("cuda"))
    outs = {}
    for b in batches:
        outs[b] = torch.empty((b * 16, R.H_LAT), dtype=torch.bfloat16, device="cuda")
        ex.prepare(b, buffers, outs[b])
    return ex, outs


def route(batch, g, case):
    if case == "random":
        return R.random_route(batch, g)
    if case == "distinct":
        return R.random_route(batch, g, distinct=True)
    pool = torch.randperm(R.E_ALL, generator=g, device="cuda")[:16]  # "few": all tokens on one set
    ids = pool.view(1, 16).expand(batch, 16).contiguous().int()
    w = torch.rand(batch, 16, generator=g, device="cuda") + 0.1
    return ids, (w / w.sum(-1, keepdim=True)).float().contiguous()


CASES = [(b, "random") for b in (1, 2, 3, 4, 5, 8)] + [(8, "distinct"), (8, "few")]


@needs_gpu
@pytest.mark.parametrize("batch,case", CASES, ids=lambda v: str(v))
def test_planned(batch, case):
    H.load_plugin(MUTATIONS)
    logical, prepared = R.shared_weights()
    ex, outs = planned(prepared, (batch,))
    ship, souts = TE.persistent(prepared, (batch,))
    out, sout = outs[batch], souts[batch]
    g = H.generator(300 + 10 * batch + len(case))
    x = torch.empty((batch, R.H_LAT), device="cuda")
    ids = torch.empty((batch, 16), dtype=torch.int32, device="cuda")
    run = lambda: ex.run(x, ids)  # noqa: E731
    graph = None
    for call in range(4):
        x.copy_(TE.latent(batch, g))
        i, w = route(batch, g, case)
        ids.copy_(i)
        H.poison(out)
        if call < 2:
            run()
        else:
            graph = graph or TE.graph_of(run)
            H.poison(out)
            graph.replay()
        torch.cuda.synchronize()
        ref = R.expert_rows(logical, x, ids)
        rows = out.view(batch, 16, -1)
        name = f"planned.b{batch}.{case}.call{call}"
        TE.check_rows(name, rows, ref)
        fi, fo = TE.fi_call(prepared, x, ids, w)
        fi()
        torch.cuda.synchronize()
        TE.check_sum(name + ".sum", rows, w, ref, fo)
        if not os.environ.get("K31_MUTATION"):
            H.poison(sout)
            ship.run(x, ids)
            torch.cuda.synchronize()
            diff = int((out.view(torch.int16) != sout.view(torch.int16)).sum())
            H.record({"kind": "parity", "item": name, "cells_differ": diff, "cells": out.numel(),
                      "distinct": int(torch.unique(ids).numel())})
    H.record({"kind": "resources", "item": f"planned.b{batch}.{case}", "grid": ex.grid_for(batch),
              **H.function_resources(ex.prepared[batch][3]), **H.device_facts()})


def ex_scratch():
    from sgl_nvidia_mega.kimi_k31.k3.cutedsl.moe import planned as PLN

    return PLN.scratch(8, torch.device("cuda"))


def time_routes(batch, gt):
    if batch == 1:
        n = 56
        perm = torch.randperm(R.E_ALL, generator=gt, device="cuda").int().view(56, 1, 16)
        return [(perm[i].contiguous(), R.random_route(1, gt)[1]) for i in range(n)]
    pools = torch.randperm(R.E_ALL, generator=gt, device="cuda").view(4, 224)
    return [R.random_route(batch, gt, pool=pools[i % 4]) for i in range(16)]


@needs_gpu
@pytest.mark.parametrize("batch", (1, 2, 4, 8), ids=lambda b: f"b{b}")
def test_time(batch):
    """ABBA x2: shipped PersistentExperts (its grid) vs planned with ring depth 5 / 6 (6 in the
    large-SMEM mode), FP32 x quantized per CTA or the MXFP8 plan (``xplan``: plan kernel +
    x TMAs; the plan kernel is inside the timed calls), at the shipped grid and at one CTA per SM."""
    if os.environ.get("K31_TIME") != "1" or os.environ.get("K31_MUTATION"):
        pytest.skip("timing")
    H.load_plugin(MUTATIONS)
    logical, prepared = R.shared_weights()
    ship, _ = TE.persistent(prepared, (batch,))
    grid = ship.grid_for(batch)
    engines = {"ship": (ship, grid)}
    variants = os.environ.get("W4_TIME_ARMS", "s5,s6x,s6x_ext")
    for v in variants.split(","):
        stages, xplan, ext, c2 = int(v[1]), "x" in v, v.endswith("_ext"), "c2" in v
        ex, _ = planned(prepared, (), stages=stages, xplan=xplan, chain2=c2)
        ex.prepare(batch, ex_scratch(), torch.empty((batch * 16, R.H_LAT), dtype=torch.bfloat16, device="cuda"),
                   (), ext)
        engines[f"planned_{v}"] = (ex, grid)
    gt = H.generator(900 + batch)
    routes = time_routes(batch, gt)
    xs = [TE.latent(batch, gt) for _ in routes]
    # every arm once on the first route: T1 (a timing arm must compute the right thing)
    for name, (ex, _) in engines.items():
        out = ex.prepared[batch][0]
        H.poison(out)
        if name.endswith("_ext"):  # external plan: written here, outside the timed calls
            from sgl_nvidia_mega.kimi_k31.k3.cutedsl.moe import experts_runtime as ER

            plan = ex.prepared[batch][5][0]
            ER.compile_plan(batch, 0, True)(ER.to_cute(xs[0], 16), plan, ER.stream(xs[0].device))
        ex.run(xs[0], routes[0][0])
        torch.cuda.synchronize()
        TE.check_rows(f"time.{name}.b{batch}", out.view(batch, 16, -1), R.expert_rows(logical, xs[0], routes[0][0]))
    distinct = sum(int(torch.unique(r[0]).numel()) for r in routes) / len(routes)
    arms = {k: [lambda ex=ex, x=x, r=r: ex.run(x, r[0]) for x, r in zip(xs, routes)] for k, (ex, _) in engines.items()}
    fl = R.floor_us(round(distinct))
    fl["distinct_mean"] = distinct
    names = list(arms)
    got = {k: [] for k in names}
    for k in names + names[::-1] + names:  # 3 runs per arm, ABC CBA ABC
        got[k].append(H.cold_time_us(arms[k]))
    for k in names:
        meds = [t["us_median"] for t in got[k]]
        row = {"kind": "time", "item": f"q8.b{batch}.{k}", "us_median": sorted(meds)[1], "us_runs": meds,
               "us_spread": max(meds) - min(meds), "grid": engines[k][1], **fl}
        row["tbps_eff"] = fl["bytes"] / (row["us_median"] * 1e-6) / 1e12
        H.record(row)
        print("TIME", row, flush=True)


@needs_gpu
@pytest.mark.parametrize("batch", (4, 8), ids=lambda b: f"b{b}")
def test_wide(batch):
    """Experiment: sm_107f dense K=64 MMAs (idesc bit 31, scale IDs 0 / 2) on MX block-32 data.
    PTX 9.4 Table 68 lists one scale per row for kind::mxf8f6f4 at every K, so this may apply one
    scale per 64 elements; T1 against the FP64 reference (independent scales per 32-block)
    decides. Records the verdict; times the variant only if T1 passes."""
    if os.environ.get("K31_MUTATION"):
        pytest.skip("experiment")
    H.load_plugin(MUTATIONS)
    logical, prepared = R.shared_weights()
    g = H.generator(1300 + batch)
    x = TE.latent(batch, g)
    ids, w = R.random_route(batch, g)
    ref = R.expert_rows(logical, x, ids)
    ex, outs = planned(prepared, (batch,), stages=6, xplan=True, wide=True)
    H.poison(outs[batch])
    ex.run(x, ids)
    torch.cuda.synchronize()
    rows = outs[batch].view(batch, 16, -1).float()
    fro = R.rel_l2(rows, ref) if bool(torch.isfinite(rows).all()) else float("inf")
    ok = fro <= TE.T1_FRO
    H.record({"kind": "check", "item": f"wide.b{batch}", "t1_fro": fro, "pass": ok})
    print("WIDE", batch, fro, ok, flush=True)
    if not ok or os.environ.get("K31_TIME") != "1":
        return
    base, _ = planned(prepared, (batch,), stages=6, xplan=True)
    gt = H.generator(1400 + batch)
    routes = time_routes(batch, gt)
    xs = [TE.latent(batch, gt) for _ in routes]
    arms = {"s6x": [lambda x=x, r=r: base.run(x, r[0]) for x, r in zip(xs, routes)],
            "s6x_wide": [lambda x=x, r=r: ex.run(x, r[0]) for x, r in zip(xs, routes)]}
    got = {k: [] for k in arms}
    for k in list(arms) + list(arms)[::-1]:
        got[k].append(H.cold_time_us(arms[k]))
    for k in arms:
        meds = [t["us_median"] for t in got[k]]
        H.record({"kind": "time", "item": f"wide.b{batch}.{k}", "us_median": sum(meds) / len(meds), "us_runs": meds})
        print("TIME", k, meds, flush=True)


@needs_gpu
@pytest.mark.parametrize("batch", (5, 8), ids=lambda b: f"b{b}")
def test_publish(batch):
    """Publish mode (the MK's B5-B8 path: experts stage BF16 rows, ``finalize`` sums the 16
    slots with the BF16 routing weights in FP32 and pushes each 8-column chunk to its owner's
    two-shot receive slot) on one GPU: all 8 owners' receive buffers are local. Checks, every
    call (2 eager + 2 graph replays, new inputs): planned == shipped bitwise on every pushed
    cell; every expected cell written (buffers pre-filled with the -0 sentinel, which the
    pushes never produce); |pushed - FP64 weighted sum of the FP64 rows| <= 2^-7 |ref| + 2^-9 rms(ref)
    (BF16 per-expert rows, FP32 sum, one BF16 rounding; the same budget T1/T2 allow)."""
    H.load_plugin(MUTATIONS)
    from sgl_nvidia_mega.kimi_k31.k3.cutedsl.moe import experts_runtime as ER

    logical, prepared = R.shared_weights()
    tp, slot = 8, 3  # receive slot = parity 0 x TP + "rank" 3
    engines = {}
    for name in ("ship", "planned"):
        if name == "ship":
            ex = ER.PersistentExperts(*prepared, beta=R.BETA, linear_beta=R.LIN)
            buffers = ER.scratch(8, torch.device("cuda"))
        else:
            ex, _ = planned(prepared, ())
            from sgl_nvidia_mega.kimi_k31.k3.cutedsl.moe import planned as PLN

            buffers = PLN.scratch(8, torch.device("cuda"))
        recv = torch.empty((tp, 3 * tp, 64, 3584 // tp), dtype=torch.bfloat16, device="cuda")
        peers = torch.tensor([recv[o].data_ptr() for o in range(tp)], dtype=torch.int64, device="cuda")
        box = (ER.to_cute(recv[0], 16), ER.to_cute(peers, 8), __import__("cutlass").Int32(slot))
        out = torch.empty((batch * 16,), dtype=torch.bfloat16, device="cuda")  # unused in publish mode
        ex.prepare(batch, buffers, out, box, False, True)
        engines[name] = (ex, recv, peers, buffers)
    g = H.generator(1500 + batch)
    x = torch.empty((batch, R.H_LAT), device="cuda")
    ids = torch.empty((batch, 16), dtype=torch.int32, device="cuda")
    w = torch.empty((batch, 16), dtype=torch.bfloat16, device="cuda")
    runs = {k: (lambda ex=v[0]: ex.run(x, ids, w.view(-1))) for k, v in engines.items()}
    graphs = {}
    for call in range(4):
        x.copy_(TE.latent(batch, g))
        i, wf = R.random_route(batch, g)
        ids.copy_(i)
        w.copy_(wf.to(torch.bfloat16))
        got = {}
        for k, (ex, recv, _, _) in engines.items():
            recv.view(torch.int16).fill_(-0x8000)
            if call < 2:
                runs[k]()
            else:
                graphs[k] = graphs.get(k) or TE.graph_of(runs[k])
                recv.view(torch.int16).fill_(-0x8000)
                graphs[k].replay()
            torch.cuda.synchronize()
            # owner o's slot holds columns [o*448, (o+1)*448) of tokens < batch
            got[k] = torch.cat([recv[o, slot, :batch] for o in range(tp)], dim=-1).clone()
        ref = R.weighted(R.expert_rows(logical, x, ids), w.float())
        name = f"publish.b{batch}.call{call}"
        unwritten = int((got["planned"].view(torch.int16) == -0x8000).sum())
        same = bool(torch.equal(got["planned"].view(torch.int16), got["ship"].view(torch.int16)))
        finite = bool(torch.isfinite(got["planned"].float()).all())
        rel = R.rel_l2(got["planned"], ref)
        stats = {"kind": "check", "item": name, "unwritten": unwritten, "bitwise_vs_ship": same,
                 "rel_l2": rel, "finite": finite, "cells": got["planned"].numel()}
        print("CHECK", stats, flush=True)
        H.record(stats)
        assert unwritten == 0, stats
        assert same, stats
        assert finite and rel <= TE.T1_FRO, stats
