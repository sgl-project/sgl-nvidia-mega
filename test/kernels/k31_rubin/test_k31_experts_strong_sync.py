# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""K3.1 persistent experts with strong intra-grid hand-offs (``K31MK_EXPERTS_STRONG_SYNC``, default off;
K-17). The hidden images, split-K partials, staged FC2 rows and their 0xFFFFFFFF re-arms are stored
with ``st.relaxed.gpu`` instead of weak ``st.global``; the hidden image's two block scales of a (tile, token) move as
one 16-bit store (warp ew 0 computes block 1's scale from the same two warp maxima warp ew 2 uses) and are polled as two
16-bit loads, instead of byte stores polled as 32-bit words (a mixed-size data race, PTX 8.7.2). The bytes written and
the arithmetic are unchanged, so the outputs must be bit-identical to the shipped form.

Checks (stated before any run; never loosened):
  K1  strong vs shipped PersistentExperts at TP16 (same host-layout weights, x, ids, one process): 0 differing BF16
      output cells on every call (2 eager + 2 CUDA-graph replays with new inputs), B1, 2, 4, 8, 16 unpacked and B8
      packed K64 (6 slots, SM107 only).
  T1  strong rows vs the FP64 reference (``_ref_moe16.expert_rows``): the T1 of test_k31_experts_tp16.

Cost split: strong=True is
the lean form (block 1's scale computed once per thread outside the token loop). K1 + T1 also run for the
constructor-only mode "inloop_DEBUG" (the in-loop scale form).

Mutations (fixed harness; each must make a base-passing test fail):
  ss_high   in-loop form: warp ew 0 takes block 1's scale from warps 0 and 1 instead of 2 and 3  [K1 inloop, T1]
  ss_pair   the 16-bit scale pair stores block 1's scale in the low byte and block 0's in the high byte  [K1, T1]
  ss_poll   the second 16-bit scale poll reads the first half again                               [K1, T1]
  lean_hi   lean form (strong=True): block 1's scale from warps 0 and 1 instead of 2 and 3         [K1 full, T1]
  lean_tok  lean form (strong=True): block 1's scale of token lane + 1 (wrong token; inert at B1)  [K1 full, T1]
The scope changes themselves (weak -> relaxed.gpu) cannot change outputs; a PTX scan checks them.

Timing (``K31_TIME=1``, Rubin): per call, cold weights (16 calls per graph), arms shipped (a)
vs strong (k): B8 unpacked (decision), B8 packed K64, B16 (info); order a k k a a k k a per pair, one GPU.
"""

import os
import statistics

import pytest
import torch

import _harness as H
import _ref_moe as R
import _ref_moe16 as M
import test_k31_experts_tp16 as T16

EXPERTS = "kimi_k31/k3/cutedsl/moe/experts.py"
MUTATIONS = {
    "ss_high": (EXPERTS,
                "high = cute.arch.fmax(shared_max[(eg * 4 + 2) * 8 + t], shared_max[(eg * 4 + 3) * 8 + t])",
                "high = cute.arch.fmax(shared_max[(eg * 4 + 0) * 8 + t], shared_max[(eg * 4 + 1) * 8 + t])"),
    "ss_pair": (EXPERTS,
                "pair = (Uint32(scale_token) & 255) | (Uint32(scale_high) & 255) << 8",
                "pair = (Uint32(scale_high) & 255) | (Uint32(scale_token) & 255) << 8"),
    "ss_poll": (EXPERTS,
                "scale = Uint32(load_u16_volatile(at)) | Uint32(load_u16_volatile(at + 2)) << 16",
                "scale = Uint32(load_u16_volatile(at)) | Uint32(load_u16_volatile(at)) << 16"),
    "lean_hi": (EXPERTS,
                "high_lean = cute.arch.fmax(shared_max[(eg * 4 + 2) * 8 + token], shared_max[(eg * 4 + 3) * 8 + token])",
                "high_lean = cute.arch.fmax(shared_max[(eg * 4 + 0) * 8 + token], shared_max[(eg * 4 + 1) * 8 + token])"),
    "lean_tok": (EXPERTS,
                 "token = lane & (self.width - 1)",
                 "token = (lane + 1) & (self.width - 1)"),
}
MODES = {"full": True, "inloop": "inloop_DEBUG"}
ARMS = {"a": False, "k": True, "d": "inloop_DEBUG"}

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


def engine(prepared, batch, strong, packed=0):
    ER = _er()
    ex = ER.PersistentExperts(*prepared, beta=R.BETA, linear_beta=R.LIN, activation_floor=None, packed=packed,
                              strong=strong)
    assert ex.strong == strong
    out = torch.empty((batch * 16, R.H_LAT), dtype=torch.bfloat16, device="cuda")
    ex.prepare(batch, ER.scratch(ex.tp, torch.device("cuda")), out)
    return ex, out


def bitwise(name, a, b):
    differ = int((a.view(torch.int16) != b.view(torch.int16)).sum())
    H.record({"kind": "check", "item": name, "cells_differ": differ, "cells": a.numel()})
    print("CHECK", name, "cells_differ", differ, flush=True)
    assert differ == 0, f"{name}: strong and shipped differ in {differ} cells"


def run_pair(batch, packed, seed, mode="full"):
    H.load_plugin(MUTATIONS)
    logical, prepared = T16.weights16()
    (ex_a, out_a), (ex_k, out_k) = engine(prepared, batch, False, packed), engine(prepared, batch, MODES[mode], packed)
    assert ex_a.prepared[batch][3] is not ex_k.prepared[batch][3], "strong and shipped share one compiled kernel"
    g = H.generator(seed + batch)
    x = torch.empty((batch, R.H_LAT), device="cuda")
    ids = torch.empty((batch, 16), dtype=torch.int32, device="cuda")

    def call():
        ex_a.run(x, ids)
        ex_k.run(x, ids)

    graph = None
    tag = f"strong.{'' if mode == 'full' else mode + '.'}{'p6.' if packed else ''}b{batch}"
    for n in range(4):
        xv = T16.latent(batch, g)
        i, _ = R.random_route(batch, g)
        x.copy_(xv)
        ids.copy_(i)
        H.poison(out_a, out_k)
        if n < 2:
            call()
        else:
            graph = graph or T16.graph_of(call)
            H.poison(out_a, out_k)
            graph.replay()
        torch.cuda.synchronize()
        bitwise(f"{tag}.call{n}", out_k, out_a)
        T16.check_rows(f"{tag}.call{n}", out_k.view(batch, 16, -1), M.expert_rows(logical, xv, i))


@needs_gpu
@pytest.mark.parametrize("batch", (1, 2, 4, 8, 16), ids=lambda b: f"b{b}")
@pytest.mark.parametrize("mode", tuple(MODES))
def test_strong_routed(mode, batch):
    """K1 + T1, unpacked."""
    run_pair(batch, 0, 1170, mode)


@needs_sm107
@pytest.mark.parametrize("mode", tuple(MODES))
def test_strong_routed_packed(mode):
    """K1 + T1, packed K64 (6 slots), B8."""
    run_pair(8, 6, 1270, mode)


# ------------------------------------------------------------------------------ timing (Rubin)
@needs_gpu
def test_time_strong():
    """Pre-registered timing: per-call device time, cold weights, arms shipped / strong,
    a k k a a k k a on this GPU, for B8 unpacked, B8 packed K64 (SM107) and B16. ``K31_STRONG_ARMS`` (default "a,k"):
    arms from ``ARMS``, order arms + reversed(arms), twice (a,k,d -> a k d d k a x2)."""
    if os.environ.get("K31_TIME") != "1" or os.environ.get("K31_MUTATION"):
        pytest.skip("timing")
    H.load_plugin(MUTATIONS)
    import test_k31_experts_depad as D

    logical, prepared = T16.weights16()
    sm107 = torch.cuda.get_device_capability() == (10, 7)
    cases = [("b8", 8, 0)] + ([("b8p6", 8, 6)] if sm107 else []) + [("b16", 16, 0)]
    for label, batch, packed in cases:
        gt = H.generator(1990 + batch + packed)
        routes = D.time_routes(batch, gt)
        xs = [T16.latent(batch, gt) for _ in routes]
        calls = {}
        arms = os.environ.get("K31_STRONG_ARMS", "a,k").split(",")
        for arm, strong in ((arm, ARMS[arm]) for arm in arms):
            ex, out = engine(prepared, batch, strong, packed)
            H.poison(out)
            ex.run(xs[0], routes[0][0])
            torch.cuda.synchronize()
            T16.check_rows(f"time.{label}.{arm}", out.view(batch, 16, -1),
                           M.expert_rows(logical, xs[0], routes[0][0]))
            calls[arm] = [lambda ex=ex, x=x, r=r: ex.run(x, r[0]) for x, r in zip(xs, routes)]
        got = {k: [] for k in calls}
        for k in (arms + arms[::-1]) * 2:
            got[k].append(H.cold_time_us(calls[k])["us_median"])
        for k in calls:
            row = {"kind": "time", "item": f"strong.time.{label}.{k}", "us_median": statistics.median(got[k]),
                   "us_runs": got[k], "us_spread": max(got[k]) - min(got[k]), **H.device_facts()}
            H.record(row)
            print("TIME", row, flush=True)
