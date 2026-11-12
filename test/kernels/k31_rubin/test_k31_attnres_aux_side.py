# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""K31MK_AUX_FOLD=3: the drafter's aux mixture on a side stream (``attn_res.aux_side``).

A non-final capture consumer compiled under mode 3 stores only the updated prefix row into
``AuxSideMixture.rows``; ``fork`` launches the stock AuxMixture score + combine passes on the
side stream over that stored row and the saved blocks, and ``join`` makes the current stream
wait for them. The final AttnRes and a block write inside rows 0..nb-1 keep the in-line
passes (``side_mixture is None``).

Contract checked at the serving capture shapes (``test_k31_attnres.FOLD_VARIANTS``: nb 1, 2,
5, 6, 7, the final AttnRes over 8 blocks, block writes at row nb and inside 0..nb-1, Lamport
ring input) and rows 1 / 8 / 16 / 64:
- BITWISE equal to the unfused in-line passes (K31MK_AUX_FOLD=0, the stock form) on the aux
  mixture, the normalized output, the new prefix and the bank, for the same inputs.
- The aux mixture also within |aux - ref| <= 1e-3 + 2^-7 |ref| of the independent FP32
  reference ``_ref_attnkda.attnres_aux_mixture`` (tolerance stated before any run, as in
  ``test_k31_attnres``).
Three schedules per case, all on one GPU: ``main_late`` (a ~10 ms spin on the main stream
before the consumer: a fork that does not wait for the consumer mixes the poisoned row
buffer), ``side_late`` (a ~10 ms spin on the side stream before the mixture, then the live
prefix is overwritten on the main stream: the mixture must read the stored row, and the
copy taken after ``join`` must see the finished mixture), and ``graph`` (side_late captured
in one CUDA graph, replayed on three fresh inputs).

Mutations (each must make the suite fail): ``side_nowait`` (fork does not wait for the
consumer), ``side_nojoin`` (join does not wait for the side stream), ``side_store`` (the
side consumer keeps the in-line mixture, so it stores the mixture, not the prefix row).
"""

import pytest
import torch

import _harness as H
import _ref_attnkda as R
from test_k31_attnres import (
    DEADLINE_NS,
    EPS,
    FOLD_VARIANTS,
    SENTINEL,
    _inputs,
    _prepared_aux,
    _target,
    _weights,
)

MUTATIONS = {
    "side_nowait": (
        "kimi_k31/k3/cutedsl/attn_res/aux_side.py",
        "        self.stream.wait_event(self.ready)\n",
        "",
    ),
    "side_nojoin": (
        "kimi_k31/k3/cutedsl/attn_res/aux_side.py",
        "            torch.cuda.current_stream(self.stream.device).wait_event(self.done)\n",
        "            pass\n",
    ),
    "side_store": (
        "kimi_k31/k3/cutedsl/attn_res/prepared.py",
        "        mixture = mixture and not side\n",
        "        mixture = mixture\n",
    ),
}

ROWS = (1, 8, 16, 64)
SPIN = 20_000_000  # torch.cuda._sleep cycles: several ms at the SM clock

needs_gpu = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 10,
    reason="needs an SM100-family GPU",
)


def _bits(t):
    return t.contiguous().view(torch.int16)


def _differ(a, b):
    return int((_bits(a) != _bits(b)).sum())


class _Case:
    def __init__(self, variant, rows):
        nb, has_delta, write, final, ring = FOLD_VARIANTS[variant]
        self.nb, self.has_delta, self.write, self.final, self.ring = nb, has_delta, write, final, ring
        self.rows = rows
        g = H.generator(9100 + rows * 37 + sorted(FOLD_VARIANTS).index(variant))
        self.g = g
        target = _target(g, nb, write, final)
        self.norm, self.query, self.out_norm = _weights(target, final)
        self.mailbox = None
        if ring:
            self.mailbox = torch.empty((1, 64, 7168), dtype=torch.bfloat16, device="cuda")
            self.mailbox.view(torch.int32).fill_(SENTINEL)
        self.stock = _prepared_aux(target, final, self.mailbox, 0)
        self.side = _prepared_aux(target, final, self.mailbox, 3)

    def inputs(self):
        return _inputs(self.g, self.rows, self.nb)

    def launch(self, prep, delta, prefix, blocks, out, aux):
        if self.ring:
            self.mailbox[0, : self.rows].copy_(delta)
            prep.run_ring(prefix, blocks, out, aux=aux)
        else:
            prep.run(delta, prefix, blocks, out, aux=aux)

    def reference(self, delta, prefix0, blocks0):
        """The unfused in-line passes (K31MK_AUX_FOLD=0) on copies of the inputs."""
        prefix, blocks = prefix0.clone(), blocks0.clone()
        out = torch.empty((self.rows, 7168), dtype=torch.bfloat16, device="cuda")
        aux = torch.empty_like(out)
        H.poison(out, aux)
        self.launch(self.stock, delta, prefix, blocks, out, aux)
        torch.cuda.synchronize()
        return out, prefix, blocks, aux


def _compare(item, got, ref, case, prefix_ref):
    names = ("output", "prefix", "bank", "aux mixture")
    for name, a, b in zip(names, got, ref):
        diff = _differ(a, b)
        H.record({"kind": "aux_side_bitwise", "item": f"{item}.{name}", "differing": diff, "cells": a.numel()})
        assert diff == 0, f"{item}: {name} differs from the unfused passes in {diff}/{a.numel()} cells"
    ref_aux = R.attnres_aux_mixture(
        prefix_ref, ref[2], nb=case.nb, norm_w=case.norm.weight, query_w=case.query.weight, eps=EPS
    )
    H.check(item + ".mixture", got[3], ref_aux, 1e-3, 2**-7)


def _side_eager(case, side_mix, schedule, delta, prefix0, blocks0):
    """One eager side-stream run; returns (out, prefix, blocks, aux) as seen after join."""
    rows = case.rows
    prefix, blocks = prefix0.clone(), blocks0.clone()
    out = torch.empty((rows, 7168), dtype=torch.bfloat16, device="cuda")
    aux = torch.empty_like(out)
    H.poison(out, aux, side_mix.rows)
    torch.cuda.synchronize()
    if schedule == "main_late":
        torch.cuda._sleep(SPIN)
    case.launch(case.side, delta, prefix, blocks, out, side_mix.row(0, rows))
    prefix_after, blocks_after = prefix.clone(), blocks.clone()
    if schedule == "side_late":
        with torch.cuda.stream(side_mix.stream):
            torch.cuda._sleep(SPIN)
    side_mix.fork(case.side, 0, blocks, aux)
    if schedule == "side_late":
        H.poison(prefix)  # the live prefix changes again in the next o-projection
    side_mix.join()
    seen = aux.clone()  # on the current stream, right after join
    torch.cuda.synchronize()
    return out, prefix_after, blocks_after, seen


@needs_gpu
@pytest.mark.parametrize("rows", ROWS)
@pytest.mark.parametrize("variant", list(FOLD_VARIANTS))
def test_aux_side(variant, rows):
    H.load_plugin(MUTATIONS)
    from sgl_nvidia_mega.kimi_k31.k3.cutedsl.attn_res.aux_side import AuxSideMixture

    case = _Case(variant, rows)
    item = f"aux_side.{variant}.b{rows}"
    expect_side = (not case.final) and case.nb > 0 and not 0 <= case.write < case.nb
    assert (case.side.side_mixture is not None) == expect_side, f"{item}: side_mixture binding"
    delta, prefix0, blocks0 = case.inputs()
    ref = case.reference(delta, prefix0, blocks0)
    prefix_ref = ref[1]
    if not expect_side:
        # Mode 3 keeps the in-line passes here (folded for the final AttnRes).
        prefix, blocks = prefix0.clone(), blocks0.clone()
        out = torch.empty((rows, 7168), dtype=torch.bfloat16, device="cuda")
        aux = torch.empty_like(out)
        H.poison(out, aux)
        case.launch(case.side, delta, prefix, blocks, out, aux)
        torch.cuda.synchronize()
        if 0 <= case.write < case.nb:
            # Outside ClusterAttnRes's contract (see test_attnres_fold): fallback only, bitwise to stock.
            for name, a, b in zip(("output", "prefix", "bank", "aux mixture"), (out, prefix, blocks, aux), ref):
                assert _differ(a, b) == 0, f"{item}: in-line {name} differs from stock"
            return
        _compare(item + ".inline", (out, prefix, blocks, aux), ref, case, prefix_ref)
        return

    side_mix = AuxSideMixture(1, rows, torch.device("cuda"))
    for schedule in ("main_late", "side_late"):
        got = _side_eager(case, side_mix, schedule, delta, prefix0, blocks0)
        _compare(f"{item}.{schedule}", got, ref, case, prefix_ref)
        if case.ring:
            assert bool((case.mailbox[0, :rows].view(torch.int32) == SENTINEL).all()), f"{item}: ring not re-armed"

    # side_late captured in one CUDA graph; static buffers refilled before each replay.
    s_delta, s_prefix, s_blocks = delta.clone(), prefix0.clone(), blocks0.clone()
    s_out = torch.empty((rows, 7168), dtype=torch.bfloat16, device="cuda")
    s_aux, s_seen = torch.empty_like(s_out), torch.empty_like(s_out)
    s_prefix_after, s_blocks_after = torch.empty_like(s_prefix), torch.empty_like(s_blocks)
    graph = torch.cuda.CUDAGraph()
    capture = torch.cuda.Stream()
    capture.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(capture):
        with torch.cuda.graph(graph, stream=capture):
            case.launch(case.side, s_delta, s_prefix, s_blocks, s_out, side_mix.row(0, rows))
            s_prefix_after.copy_(s_prefix)
            s_blocks_after.copy_(s_blocks)
            # The spin must be captured: the side stream joins the capture first.
            side_mix.stream.wait_stream(capture)
            with torch.cuda.stream(side_mix.stream):
                torch.cuda._sleep(SPIN)
            side_mix.fork(case.side, 0, s_blocks, s_aux)
            s_prefix.view(torch.uint8).fill_(0xFF)
            side_mix.join()
            s_seen.copy_(s_aux)
    torch.cuda.current_stream().wait_stream(capture)
    torch.cuda.synchronize()
    for replay in range(3):
        delta, prefix0, blocks0 = case.inputs()
        ref = case.reference(delta, prefix0, blocks0)
        s_delta.copy_(delta)
        s_prefix.copy_(prefix0)
        s_blocks.copy_(blocks0)
        H.poison(s_out, s_aux, s_seen, side_mix.rows)
        torch.cuda.synchronize()
        graph.replay()
        torch.cuda.synchronize()
        got = (s_out, s_prefix_after, s_blocks_after, s_seen)
        _compare(f"{item}.graph{replay}", got, ref, case, ref[1])
