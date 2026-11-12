# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""K3.1 TP16 o-projection all-reduce + attn-res prefix: the MK's rounding points against stock's fused kernel.

Stock TP16 (``sglang::all_reduce_push_res_kernel`` on Rubin for 8 / 16 / 32 rows, ``ar_fusion.cuh`` +
``vec.cuh reduce_vec``) computes, per element, ``bf16_rn(((P_0 + P_1) + ... + P_15) + prefix_old)`` in FP32 with ONE
rounding (P_q = rank q's BF16 o_proj partial). This test emulates that in PyTorch FP32 from the same inputs (independent
of the MK and of stock's kernel) and requires the MK's new prefix to equal it bit for bit.

Paths (``oproj.prepared`` at TP16): rows 8 = ``OprojFused`` (consumer groups 4), rows 16 = producer + split consumer
(groups 4), rows 32 = two-shot (owner ``common.twoshot.reduce`` + one-slot consumer). Over W = 4 GPUs with real
symm_mem multicast: GPU g plays TP rank g of 16 (its producer is real); ranks 4-15's partials are injected into this
GPU's mailbox slots (rows 8 / 16), or into owner g's receive slots 4-15 (rows 32), and the gather columns of owners
4-15 (no GPU plays them) are written by the test from the stock emulation (they test nothing; the 4 x 448 columns of
owners 0-3 are reduced by the kernel and are reported separately).

Datasets (fixed before any run):
  E (calls 0, 2): every rank's x and W are small integers (x in {-2..2}, W in {-2..2} x 2^-6), so each S_q = x_q W_q^T
    is exact in FP32 in any order, P_q = bf16_rn(S_q) is known exactly, and the FP32 sum of the 16 P_q is exact in any
    order: the MK's grouped order and stock's rank order give the same FP32 sum, so the only possible difference from
    stock is the rounding points. GATE G1: new prefix == stock emulation, bitwise (-0 mapped to +0), every cell.
  W (calls 1, 3): ranks 0-3 as E; injected ranks 4-15 have wide-range BF16 values (|.| ~ 2^-20..2^5), so FP32 sums
    depend on order. GATE G2: new prefix == the emulation with the MK's documented order (rows 8/16: four groups of 4
    ranks summed in rank order, then the groups in order; rows 32: rank order) and stock's rounding points, bitwise.
    REPORT R1 (not a gate): cells that differ from the stock rank-order emulation (a reorder of accumulation).
  Every call: EXERCISED = cells where the double-rounding emulation bf16(old + bf16(sum)) differs from the stock
    emulation, recorded per case (dataset E, has_delta layers must have EXERCISED > 0, else the case cannot tell the
    two apart: a test failure); out (AttnRes + RMSNorm, downstream) vs ``_ref_comm.reference`` with its tolerances;
    mailbox / gather / receive slots re-armed; calls 0-1 eager, 2-3 from one CUDA graph per pair with new inputs.

Bugs caught (``MUTATIONS``): the base-D double rounding (``unfixed``), a dropped prefix, the two-shot owner rounding
before adding the prefix, the two-shot prefix added twice, the prefix lanes swapped in the owner's add.

``--opr`` (package (b), the ported Rubin o-proj options; every one bitwise identical to the
path it replaces, so every gate above applies unchanged): the knobs ``OPR_ENV`` are set before the kernels are built
and each case first checks what each kernel was built with (``PreparedOproj.opr``) against ``opr_expected`` (the tables
written out by hand). Rows 8: fused + vote; 16: one-shot split, resident tcgen05 producer (+ carve-out 80) + vote
consumer; 32: two-shot, the producer reduces the owned columns (+ the prefix on has_delta layers, single rounding) with
a resident ring (+ carve-out 80), the consumer polls the gather with batched owner polls off; 128: none engaged.
Bugs caught with ``--opr`` (``MUTATIONS`` opr_*): the producer owner drops the prefix, the vote poll leaves before
every slot is published, the batched owner poll leaves before every slot is published, the options built but not
passed to the kernels.

Usage (one node, 4 GPUs): python3 -m torch.distributed.run --nproc-per-node 4 k31_oproj_round16.py --out DIR
  [--rows 8,16,32] [--max-pairs N] [--deadline-s S]. Exit status 0 only when every check on every rank passed (``TP_RESULT ... failures=0``).
"""

import argparse
import json
import os
import sys
import time
import traceback
import zlib

# The single rounding is a default-off knob; this test checks the knob-on arithmetic.
os.environ.setdefault("K31MK_OPROJ_SINGLE_ROUND", "1")

import torch
import torch.distributed as dist

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _harness as H  # noqa: E402
import _ref_comm as C  # noqa: E402

TP, K, HIDDEN, CAP = 16, 768, C.HIDDEN, C.CAP
OWN = HIDDEN // TP  # two-shot columns per owner (448)
# rows -> layer pairs ((even layer, mla), (odd layer, mla)); (nb, has_delta) from _ref_comm.layer_facts:
# 92 (8, delta), 5 (1, delta), 84 (8, no delta), 41 (4, delta), 12 (2, no delta).
CASES = {
    8: [((92, False), (5, True)), ((84, True), (41, False))],
    16: [((92, True), (41, False)), ((12, False), (5, True))],
    32: [((92, False), (41, True)), ((84, True), (5, False))],
    # Two-shot rows the L3 own Breq 3 / 5 / 6 / 7 graphs (24 / 40 / 48 / 56) and Breq 8 (64) build (package (b)
    # tables 17-64 cover them; the earlier gate ran only 8 / 16 / 32 / 128). Same layer pairs as 32.
    24: [((92, False), (41, True)), ((84, True), (5, False))],
    40: [((92, False), (41, True)), ((84, True), (5, False))],
    48: [((92, False), (41, True)), ((84, True), (5, False))],
    56: [((92, False), (41, True)), ((84, True), (5, False))],
    64: [((92, False), (41, True)), ((84, True), (5, False))],
    # Breq 16 verify (CAP-128 #3): two-shot over 128-row receive / gather buffers (oproj.prepared.capacity).
    128: [((92, False), (41, True)), ((84, True), (5, False))],
    # Wide tier (Breq 9 / 12 / 15 (72 / 96 / 120 rows) run their own graphs, two-shot (TWOSHOT[16] 17-128) inside the boot's
    # 128-row buffers. Run them with --cap 128 (and every smaller row count with --cap 128 = a <= 64-row call of a graphs-1..16 boot).
    72: [((92, False), (41, True)), ((84, True), (5, False))],
    96: [((92, False), (41, True)), ((84, True), (5, False))],
    120: [((92, False), (41, True)), ((84, True), (5, False))],
}
GROUPS = {8: 4, 16: 4, 32: 1, 24: 1, 40: 1, 48: 1, 56: 1, 64: 1, 128: 1, 72: 1, 96: 1, 120: 1}  # the consumer's polling groups per rows (oproj.prepared.compile_oproj)

CO = "kimi_k31/k3/cutedsl/oproj/consumer.py"
PD = "kimi_k31/k3/cutedsl/oproj/producer.py"
# Package (b) at the ported tables: see the module docstring.
OPR_ENV = {
    "K31MK_OPROJ_BATCHED": "17-64:1",
    "K31MK_OPROJ_PRODUCER_REDUCE": "17-64:1",
    "K31MK_OPROJ_RESIDENT": "9-64:1",
    "K31MK_OPROJ_PROD_CARVEOUT": "9-40:80",
    "K31MK_OPROJ_VOTE": "1-16:1",
}


def opr_expected(rows, has_delta):
    """{kind: options} each kernel at ``rows`` must be built with under OPR_ENV (() = the unchanged kernel)."""
    return {
        8: {"fused": (True,)},
        16: {"producer": (False, False, True), "consumer": (False, False, True)},
        32: {"producer": (True, has_delta, True), "consumer": (True, True, False)},
        # 24-64: two-shot (TWOSHOT 17-128), so BATCHED / PRODUCER_REDUCE 17-64 and RESIDENT 9-64 apply, VOTE (one-shot only) not.
        **{r: {"producer": (True, has_delta, True), "consumer": (True, True, False)} for r in (24, 40, 48, 56, 64)},
        128: {"producer": (), "consumer": ()},
    }[rows]
TS = "kimi_k31/k3/cutedsl/common/twoshot.py"
PR = "kimi_k31/k3/cutedsl/oproj/prepared.py"
_RESIDUAL_ADD = (
    "        if cutlass.const_expr(residual is not None):\n"
    "            low = add_rn(low, residual[2 * j])\n"
    "            high = add_rn(high, residual[2 * j + 1])\n"
)
MUTATIONS = {
    # The base-D arithmetic: bf16(old + bf16(sum)) in the consumer, and the two-shot owner rounds the sum
    # alone (no prefix) while the consumer adds the prefix with a second rounding.
    # Since the knob: the knob read as off, i.e. the default path is the base-D arithmetic.
    "unfixed": (CO, 'SINGLE_ROUND = os.environ.get("K31MK_OPROJ_SINGLE_ROUND", "0") == "1"',
                "SINGLE_ROUND = False"),
    # The one-/split-shot consumer drops the prefix: new prefix = bf16(sum).
    "drop_prefix": (CO, "                        acc = pair_op(two(old, j), acc)\n",
                    "                        acc = (acc[0], acc[1])\n"),
    # The two-shot owner rounds the FP32 sum to BF16 before adding the prefix (a second rounding inside the owner).
    "ts_round_first": (TS, _RESIDUAL_ADD,
                       "        if cutlass.const_expr(residual is not None):\n"
                       "            low, high = unpack_bf16(pack_pair(low, high))\n"
                       "            low = add_rn(low, residual[2 * j])\n"
                       "            high = add_rn(high, residual[2 * j + 1])\n"),
    # The consumer's guard misses the two-shot case: the owner and the consumer both add the prefix.
    "ts_double_add": (CO, "if cutlass.const_expr(self.has_delta and not self.twoshot):",
                      "if cutlass.const_expr(self.has_delta):"),
    # The owner adds the prefix of the neighbouring column (lanes swapped within a pair).
    "ts_lane_swap": (TS, _RESIDUAL_ADD,
                     "        if cutlass.const_expr(residual is not None):\n"
                     "            low = add_rn(low, residual[2 * j + 1])\n"
                     "            high = add_rn(high, residual[2 * j])\n"),
    # CAP-128 #3: the kernels are built for 64-row buffers while the boot's buffers hold 128 rows (the push / gather
    # offsets of rows 128 use the wrong slot stride). Equivalent at rows <= 64 (cap is 64 there).
    # --opr: the producer's owner reduce drops the prefix (the gather row = bf16(sum)): G1 fails on has_delta layers.
    "opr_noresid": (PD, "reduce(recv, gather, gather_mc, rank, token, column, self.deadline_ns, old, True)",
                    "reduce(recv, gather, gather_mc, rank, token, column, self.deadline_ns, None, True)"),
    # --opr: the vote poll never loops (sums whatever the first fetch read, sentinels included).
    "opr_vote_early": (CO, "            while not cute.arch.vote_all_sync(ready):\n",
                       "            while not cute.arch.vote_all_sync(True):\n"),
    # --opr: the batched owner poll reports every slot published after one pass.
    "opr_batched_early": (TS, "            pending |= 1 << slot\n    return pending\n",
                          "            pending |= 0\n    return pending\n"),
    # --opr: the options are computed but the kernels are built without them.
    "opr_inert": (PR, "                    **(dict(opr=by_kind[kind]) if by_kind[kind] else {}),\n",
                  "                    **{},\n"),
    "cap64_build": (PR, "                    publish and kind != \"producer\",\n                    cap,\n",
                    "                    publish and kind != \"producer\",\n                    64,\n"),
}


OPR = False  # --opr
CAP_FORCE = 0  # --cap


def _gen(device, *key):
    """The same stream in every process (same seed, same GPU model)."""
    g = torch.Generator(device=device)
    g.manual_seed(zlib.crc32(repr(key).encode()))
    return g


def small_int(shape, scale, g, device):
    """Integers in {-2..2} times ``scale`` (a power of two), BF16 (exact)."""
    return (torch.randint(-2, 3, shape, generator=g, device=device).float() * scale).to(torch.bfloat16)


def canon(t):
    return C.canon(t.contiguous())


def bits(t):
    return canon(t).view(torch.int16)


def stock_sum(partials, old):
    """Stock's reduce_vec: FP32 adds in rank order, then the residual, ONE rounding (not rounded here)."""
    acc = partials[0].float()
    for p in partials[1:]:
        acc = acc + p.float()
    if old is not None:
        acc = acc + old.float()
    return acc


def grouped_sum(partials, groups):
    """The MK consumer's order: ``groups`` groups of TP / groups ranks, each in rank order, then the groups in order."""
    n = TP // groups
    parts = []
    for k in range(groups):
        a = partials[k * n].float()
        for p in partials[k * n + 1 : (k + 1) * n]:
            a = a + p.float()
        parts.append(a)
    acc = parts[0]
    for a in parts[1:]:
        acc = acc + a
    return acc


class TwoShot16:
    """``common.twoshot.TwoShot`` with TP16 geometry over 4 GPUs: owners 4-15 (no consumer) get local stand-in receive
    buffers (written by the real producers, never read) and their gather columns are written by ``inject``."""

    def __init__(self, group, device, world, parities=2, cap=CAP):
        import torch.distributed._symmetric_memory as symm_mem

        bf16 = dict(dtype=torch.bfloat16, device=device)
        self.recv = symm_mem.empty((parities, TP, cap, OWN), **bf16)
        self.gather = symm_mem.empty((parities, 1, cap, HIDDEN), **bf16)
        handles = [symm_mem.rendezvous(t, group) for t in (self.recv, self.gather)]
        size = self.recv[0].numel() * 2
        ptrs = handles[0].buffer_ptrs
        self.standins = {o: torch.empty((parities, TP, cap, OWN), **bf16) for o in range(world, TP)}
        bases = []
        for p in range(parities):
            row = []
            for o in range(TP):
                if o < world:
                    row.append(int(ptrs[o]) + p * size)
                else:
                    row.append(self.standins[o].data_ptr() + p * size)
            bases.append(row)
        self.peers = torch.tensor(bases, dtype=torch.int64, device=device)
        self.gather_mc = int(handles[1].multicast_ptr or 0)
        if not self.gather_mc:
            raise RuntimeError("two-shot stand-in needs multicast memory")
        for t in (self.recv, self.gather, *self.standins.values()):
            t.view(torch.int32).fill_(C.SENTINEL_I32)
        self.world = world
        torch.cuda.synchronize(device)
        dist.barrier(group=group)

    def views(self, parity):
        gather_mc = self.gather_mc + parity * self.gather[0].numel() * 2
        return self.recv[parity], self.peers[parity], self.gather[parity], gather_mc


class Case:
    def __init__(self, P, group, g, world, device, rows, pair, deadline_ns):
        self.P, self.group, self.g, self.world, self.device, self.rows = P, group, g, world, device, rows
        self.layers = tuple(layer for layer, _ in pair)
        self.name = f"round16_tp{world}.r{rows}.l{self.layers[0]}-{self.layers[1]}"
        # as build_resources sizes both buffers; --cap: the boot's capacity (a graphs-1..16 boot sizes every call's buffers at 128 rows)
        cap = CAP_FORCE or P.capacity((rows,))
        assert cap >= rows, (cap, rows)
        self.mailbox = P.OprojMailbox(group, device, TP, g, (2, TP, cap, HIDDEN))
        self.twoshot = TwoShot16(group, device, world, cap=cap) if rows in P.twoshot_batches(TP) else None
        self.groups = GROUPS[rows]
        self.weights, self.decoders, self.preps = {}, {}, {}
        for layer, mla in pair:
            gw = _gen(device, "w", layer)
            self.weights[layer] = [small_int((HIDDEN, K), 2.0**-6, gw, device) for _ in range(TP)]
            dec = C.make_decoder(layer, self.weights[layer][g], device)
            self.decoders[layer] = dec
            prep = P.PreparedOproj(dec, self.mailbox, (rows,), mla, None, self.twoshot, deadline_ns=deadline_ns)
            if OPR:
                want = opr_expected(rows, self.has_delta(layer))
                got = prep.opr.get(rows)
                H.record({"kind": "check", "item": f"{self.name}.l{layer}.g{g}.opr_built", "want": repr(want),
                          "got": repr(got)})
                assert got == want, f"package (b) build: {got} != {want}"
            kinds = len(prep.codes[rows])
            want = 1 if rows <= 8 else 2
            assert kinds == want, f"rows {rows}: {kinds} kernels, expected {want}"
            assert (rows in prep.twoshots) == (self.twoshot is not None), "two-shot path not as expected"
            self.preps[layer] = prep

    def inputs(self, layer, call):
        """(x of this GPU's rank, prefix_old, bank, partials[16], dataset)."""
        b, d = self.rows, self.device
        dataset = "E" if call % 2 == 0 else "W"
        gx = _gen(d, "x", layer, call)
        xs = [small_int((b, K), 1.0, gx, d) for _ in range(TP)]
        partials = []
        for q in range(TP):
            exact = xs[q].double() @ self.weights[layer][q].double().T  # exact: |S| <= 48 in steps of 2^-6
            partials.append(canon(exact.float().to(torch.bfloat16)))
        if dataset == "W":
            gp = _gen(d, "wide", layer, call)
            for q in range(self.world, TP):
                mag = torch.exp2(torch.empty((b, HIDDEN), device=d).uniform_(-20.0, 5.0, generator=gp))
                partials[q] = canon((torch.randn((b, HIDDEN), generator=gp, device=d) * mag).to(torch.bfloat16))
        prefix_old, bank = C.state_inputs(layer, call, b, d)
        return xs[self.g], prefix_old, bank, partials, dataset

    def has_delta(self, layer):
        return not C.layer_facts(layer)[0]

    def expected(self, layer, partials, prefix_old):
        """Stock emulation, MK-order emulation and the double-rounding emulation (all BF16, -0 -> +0)."""
        old = prefix_old if self.has_delta(layer) else None
        stock = canon(stock_sum(partials, old).to(torch.bfloat16))
        if self.groups == 1:
            mk = stock  # two-shot: the owner sums in rank order, then the prefix
            plain = stock_sum(partials, None)
        else:
            plain = grouped_sum(partials, self.groups)
            mk = canon((plain + old.float()).to(torch.bfloat16)) if old is not None else canon(plain.to(torch.bfloat16))
        if old is not None:
            double = canon((old.float() + plain.to(torch.bfloat16).float()).to(torch.bfloat16))
        else:
            double = canon(plain.to(torch.bfloat16))
        return stock, mk, double

    def stage(self, layer, partials, prefix_old):
        """Injected data: mailbox slots 4-15 (rows 8 / 16), or owner g's receive slots 4-15 and owners 4-15's gather
        columns = the stock emulation of their reduce (rows 32). Returns the tensors to copy and their targets."""
        b, w, parity = self.rows, self.world, layer % 2
        if self.twoshot is None:
            return [(self.mailbox.buffer[parity, w:TP, :b], torch.stack(partials[w:]))]
        lo, hi = self.g * OWN, (self.g + 1) * OWN
        recv = torch.stack([p[:, lo:hi] for p in partials[w:]])  # [12, B, 448]
        old = prefix_old if self.has_delta(layer) else None
        owners = canon(stock_sum(partials, old).to(torch.bfloat16))[:, w * OWN :]
        return [(self.twoshot.recv[parity, w:TP, :b], recv), (self.twoshot.gather[parity, 0, :b, w * OWN :], owners)]

    def verify(self, tag, layer, dataset, prefix, out, prefix_old, bank, partials, failures):
        name = f"{self.name}.l{layer}.g{self.g}.{tag}.{dataset}"
        parity = layer % 2
        try:
            stock, mk, double = self.expected(layer, partials, prefix_old)
            got = bits(prefix)
            exercised = int((bits(double) != bits(stock)).sum())
            vs_stock = got != bits(stock)
            vs_mk = got != bits(mk)
            real = slice(0, self.world * OWN)  # two-shot columns the kernel reduced (owners 0-3)
            row = {"kind": "check", "item": f"{name}.prefix", "cells": got.numel(), "has_delta": self.has_delta(layer),
                   "exercised_double_vs_single": exercised, "mismatch_vs_stock": int(vs_stock.sum()),
                   "mismatch_vs_mk_order": int(vs_mk.sum()), "mismatch_vs_double": int((got != bits(double)).sum()),
                   "mk_order_vs_stock": int((bits(mk) != bits(stock)).sum())}
            if self.twoshot is not None:
                row["real_owner_cells"] = got[:, real].numel()
                row["real_owner_mismatch_vs_stock"] = int(vs_stock[:, real].sum())
                row["real_owner_exercised"] = int((bits(double)[:, real] != bits(stock)[:, real]).sum())
            H.record(row)
            print("CHECK", json.dumps(row), flush=True)
            if dataset == "E":
                assert row["mk_order_vs_stock"] == 0, "dataset E sums are not order-free (test data bug)"
                if self.has_delta(layer):
                    assert exercised > 0, "dataset E does not exercise the double rounding"
                    if self.twoshot is not None:
                        assert row["real_owner_exercised"] > 0, "real-owner columns not exercised"
                assert row["mismatch_vs_stock"] == 0, (
                    f"G1: {row['mismatch_vs_stock']}/{got.numel()} prefix cells differ from the stock emulation"
                    f" (vs double-rounding emulation: {row['mismatch_vs_double']})")
            else:
                assert row["mismatch_vs_mk_order"] == 0, (
                    f"G2: {row['mismatch_vs_mk_order']}/{got.numel()} prefix cells differ from the MK-order emulation"
                    f" with stock rounding points")
            ref = C.reference(self.decoders[layer], partials, prefix_old, bank)
            rms = float(ref[1].float().pow(2).mean().sqrt())
            H.check(f"{name}.out", out, ref[1], C.OUT_TOL_ABS_SCALE * rms, C.OUT_TOL_REL)
            if self.twoshot is None:
                C.check_rearmed(name, self.mailbox.buffer[parity], self.rows)
            else:
                C.check_rearmed(f"{name}.gather", self.twoshot.gather[parity], self.rows)
                C.check_rearmed(f"{name}.recv", self.twoshot.recv[parity], self.rows)
        except AssertionError as e:
            failures.append(f"{name}: {e}")
            print("FAIL", name, str(e)[:400], flush=True)

    def sync(self):
        torch.cuda.synchronize()
        dist.barrier(group=self.group)

    def eager(self, failures):
        b, d = self.rows, self.device
        out = torch.empty((b, HIDDEN), dtype=torch.bfloat16, device=d)
        for call in (0, 1):
            for layer in self.layers:
                x, prefix_old, bank, partials, dataset = self.inputs(layer, call)
                self.sync()
                for dst, src in self.stage(layer, partials, prefix_old):
                    dst.copy_(src)
                prefix = prefix_old.clone()
                H.poison(out)
                self.sync()
                self.preps[layer].run(x, prefix, bank, out)
                self.sync()
                self.verify(f"call{call}", layer, dataset, prefix, out, prefix_old, bank, partials, failures)

    def graph(self, failures):
        b, d, w = self.rows, self.device, self.world
        st = {}
        for layer in self.layers:
            s = {"x": torch.empty((b, K), dtype=torch.bfloat16, device=d),
                 "prefix": torch.empty((b, HIDDEN), dtype=torch.bfloat16, device=d),
                 "bank": torch.empty((b, 8, HIDDEN), dtype=torch.bfloat16, device=d),
                 "out": torch.empty((b, HIDDEN), dtype=torch.bfloat16, device=d)}
            if self.twoshot is None:
                s["inj"] = [torch.empty((TP - w, b, HIDDEN), dtype=torch.bfloat16, device=d)]
            else:
                s["inj"] = [torch.empty((TP - w, b, OWN), dtype=torch.bfloat16, device=d),
                            torch.empty((b, HIDDEN - w * OWN), dtype=torch.bfloat16, device=d)]
            st[layer] = s
        self.sync()
        graph, stream = torch.cuda.CUDAGraph(), torch.cuda.Stream()
        with torch.cuda.graph(graph, stream=stream):
            for layer in self.layers:
                s, parity = st[layer], layer % 2
                if self.twoshot is None:
                    self.mailbox.buffer[parity, w:TP, :b].copy_(s["inj"][0])
                else:
                    self.twoshot.recv[parity, w:TP, :b].copy_(s["inj"][0])
                    self.twoshot.gather[parity, 0, :b, w * OWN :].copy_(s["inj"][1])
                self.preps[layer].run(s["x"], s["prefix"], s["bank"], s["out"])
        self.sync()
        for call in (2, 3):
            ins = {layer: self.inputs(layer, call) for layer in self.layers}
            for layer in self.layers:
                x, prefix_old, bank, partials, _ = ins[layer]
                s = st[layer]
                s["x"].copy_(x)
                s["prefix"].copy_(prefix_old)
                s["bank"].copy_(bank)
                for (_, src), buf in zip(self.stage(layer, partials, prefix_old), s["inj"]):
                    buf.copy_(src)
                H.poison(s["out"])
            self.sync()
            graph.replay()
            self.sync()
            for layer in self.layers:
                x, prefix_old, bank, partials, dataset = ins[layer]
                s = st[layer]
                self.verify(f"graph{call}", layer, dataset, s["prefix"], s["out"], prefix_old, bank, partials,
                            failures)
        del graph


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--rows", default=",".join(map(str, CASES)))
    ap.add_argument("--deadline-s", type=float, default=2.0)
    ap.add_argument("--max-pairs", type=int, default=0, help="layer pairs per rows value (0 = all)")
    ap.add_argument("--opr", action="store_true", help="package (b): set OPR_ENV before building")
    ap.add_argument("--cap", type=int, default=0, help="mailbox / two-shot rows for every case (0 = capacity((rows,)); a graphs-1..16 boot: 128)")
    args = ap.parse_args()
    global OPR, CAP_FORCE
    OPR = args.opr
    CAP_FORCE = args.cap
    if OPR:
        os.environ.update(OPR_ENV)
    local = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local)
    device = torch.device("cuda", local)
    dist.init_process_group("nccl", device_id=device)
    g, world = dist.get_rank(), dist.get_world_size()
    assert world == 4, "TP ranks 0-3 real, 4-15 injected"
    tag = (os.environ.get("K31_MUTATION") or "base") + ("_opr" if OPR else "") + (f"_cap{CAP_FORCE}" if CAP_FORCE else "")
    os.makedirs(args.out, exist_ok=True)
    os.environ["K31_RESULTS"] = os.path.join(args.out, f"round16_{tag}_g{g}.jsonl")
    H.load_plugin(MUTATIONS)
    from sgl_nvidia_mega.kimi_k31.k3.cutedsl.oproj import prepared as P

    H.record({"kind": "env", **H.device_facts(), "world": world})
    failures, cases = [], 0
    group = dist.group.WORLD
    for rows in map(int, args.rows.split(",")):
        for pair in CASES[rows][: args.max_pairs or None]:
            t0 = time.time()
            try:
                case = Case(P, group, g, world, device, rows, pair, int(args.deadline_s * 1e9))
                case.eager(failures)
                case.graph(failures)
                cases += 1
                print(f"CASE {case.name} g{g} done in {time.time() - t0:.1f}s", flush=True)
            except Exception as e:  # noqa: BLE001 - report and continue with the next case
                failures.append(f"r{rows} {pair}: {e!r}")
                print("CASE_ERROR", rows, pair, repr(e)[:600], flush=True)
                traceback.print_exc()
    t = torch.tensor([len(failures)], device=device)
    dist.all_reduce(t, group=group)
    with open(os.path.join(args.out, f"round16_{tag}_g{g}.failures.json"), "w") as f:
        json.dump(failures, f, indent=1)
    print(f"[rank{g}] TP_RESULT world={world} cases={cases} failures={int(t)}", flush=True)
    dist.destroy_process_group()
    sys.exit(1 if int(t) else 0)


if __name__ == "__main__":
    main()
