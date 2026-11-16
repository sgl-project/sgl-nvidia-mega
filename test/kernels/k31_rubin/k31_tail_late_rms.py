# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Late RMS in the TP16 MoE tail (``K31MK_MOE_TAIL_LATE_RMS``; the ported late_rms /
owner_squares) at Breq 1 (B8: front-mode ``MoeTail``) and Breq 2 (B16: published
``WaveMoeTail``, reduce, early), in-chain through ``moe.prepared.PreparedMoE`` over W = 4 GPUs (ranks 4-15 injected).
The rig is ``k31_tail_opts_b8.py`` with the batch made a
parameter, the published-tail path (push + WaveMoeTail) restored for B16, per-arm numerics orders and an L1 mode.

Arms (each with its own mailboxes; env flags are read at construction):
  B8:  A, A2 today; L late RMS ("8:1"); L2 second instance; LO late + owner-only squares ("8:3");
       Z  K31MK_MOE_TAIL_POLL_NS=0 (the adopted Z); ZL, ZLO  Z + L / LO; Z2, ZL2 second instances.
  B16: A, A2 today (RMS over the BF16 reduced rows: K-G3-03); L late RMS ("16:1"); L2;
       F  K31MK_MOE_TAIL_FP32_RMS=16:1 (FP32 reduced rows, RMS of the FP32 sums: the K-G3-03 fix); FL  F + L;
       F2, FL2 second instances.

Checks (tolerances stated before any run):
  ids / weights / latent / hg / shared / partial / multicast / RS / rank_bitwise / graph: as the source rig
  (expert rows T1 and partial-vs-rows only on unpublished arms, as the source rig);
  output   rows of ranks 0-3 vs an FP64 emulation of the arm's OWN rounding order, within 2 BF16 ulp of
           max(|out|, |y|, |shared|):  stock  BF16 x, RMS of the FP32 sums, BF16(x inv gamma), up, BF16, + shared, BF16;
           k3  as stock with the RMS over BF16(x) (K-G3-03, B16 today);  late / late_k3  BF16(x gamma), up, x inv, BF16,
           + shared, BF16 (RMS as stock / k3);
  engaged  the arm's compiled tail IS compile_tail(...) for its intended options; a late arm's is not the default;
  reorder  every late arm's outputs differ from A's in >= 1 cell over the run (the reorder is in effect);
  bitwise  ids, latent partials, shared rows (and expert rows) equal A's; outputs equal the arm's output reference
           (A for A2/Z, L for L2/LO/ZL/ZLO);
  path     ExpertsKernel + MoeTail(448) without push (B8) / WaveMoeTail(112), push_kernel only if the shared expert is
           not the fused kernel (B16: fused, it peer-stores its rows; its S is read back from the RS mailboxes).
L1 (``--l1 N``, ``--l1-modes shared,noshared``): N input sets per mode (layers in chain order, fresh seeded x,
  injected partials, shards and a per-set scale), at the tail output (shared) and at the routed output y (noshared: shared rows zero);
  per set and arm, mean over tokens of rel-L2(out - REF) over the 1792 output columns of ranks 0-3, REF = FP64 exact
  math (x = sum of the 16 BF16 partials, x / rms(x) * gamma, up, + sum of the 16 BF16 shared rows); comparators on
  the same mailbox contents: S = stock (FP32 rank-order all-reduce, BF16, FP32-sum RMS rsqrt, (x inv) gamma BF16,
  stock's cuBLAS up GEMM F.linear, + FP32-summed shared, BF16), Lt = late order in torch (FP32 GEMM), Pt = PLANTED
  control (Lt with the FP32 GEMM sum rounded to BF16 before the scale). Records only; bin/l1_verdict.py decides.
Mutations (each must fail): late_twice, late_noscale, late_token, late_half_peers, late_gamma_twice, wave_late_token,
wave_late_noscale, wave_late_group, flag_inert; K-G3-03 fix (B16 F / FL): fp32_sq_bf16, fp32_half, fp32_rearm_half
(race-type), fp32_flag_inert. ``plant_bf16`` (both sites) is the kernel-planted L1 control.

Timing (--time R): R replicates, arm order alternating forward / reverse. Per run: 20 warm replays; M2 = CUDA-event
time of 100 back-to-back graph replays / 600 layers; from a torch-profiler trace of 20 replays, per layer T = t_end(tail)
- t_end(preceding experts kernel) and M1 = t_end(tail) - t_end(RouteSitu grid 2B) (medians on this GPU).

Usage: python3 -m torch.distributed.run --nproc-per-node 4 k31_tail_late_rms.py --out DIR --batch 8|16
       [--arms ...] [--time 6] [--l1 64 --l1-arms A,L --l1-seed0 0]
"""

import argparse
import hashlib
import json
import os
import statistics
import sys
import tempfile
import time
import traceback
from types import SimpleNamespace

import torch
import torch.distributed as dist

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _harness as H  # noqa: E402
import _ref_moe as R  # noqa: E402
import _ref_moe16 as M  # noqa: E402
from k31_moe_tp16 import (  # noqa: E402
    EPS, HID, LAT, LOGITS, LROWS, SENT, SHARD, TP, Checker, Model, bf, canon_bf16, gen, randn, route_ref, ulp_bf16)

B = 8  # set from --batch in main()
NLAYERS = 6  # multiple of 3: see k31_wavetail_tp.py
LAYER_IDS = tuple(21 + i for i in range(NLAYERS))
TARGET_ED = 79.0  # per 8-row request group (B16 = two requests)

TL = "kimi_k31/k3/cutedsl/moe/tail.py"
TR_ = "kimi_k31/k3/cutedsl/moe/tail_runtime.py"
MUTATIONS = {
    # MoeTail late: the leader applies the RMS reciprocal twice.
    "late_twice": (TL, "                    value = value * inverse\n",
                   "                    value = value * inverse * inverse\n"),
    # MoeTail late: the leader never scales (the GEMV of BF16(x gamma) is published unnormalized).
    "late_noscale": (TL, "                    value = value * inverse\n", "                    value = value\n"),
    # MoeTail late: the leader takes the other token's squares.
    "late_token": (TL, "                        total += squares[peer * local_tokens + warp]",
                   "                        total += squares[peer * local_tokens + (local_tokens - 1 - warp)]"),
    # MoeTail late: the leader sums only half of the 16 (CTA, warp) square partials.
    "late_half_peers": (TL, "                    for peer in cutlass.range_constexpr(2 * self.cluster):\n"
                            "                        total += squares[peer * local_tokens + warp]",
                        "                    for peer in cutlass.range_constexpr(self.cluster):\n"
                        "                        total += squares[peer * local_tokens + warp]"),
    # MoeTail late: x is scaled by gamma twice before the GEMV.
    "late_gamma_twice": (TL, "scaled = lambda k, h: bf16(xx[k], h) * bf16(gx[k], h)",
                         "scaled = lambda k, h: bf16(xx[k], h) * bf16(gx[k], h) * bf16(gx[k], h)"),
    # WaveMoeTail late: an owner scales its line by the next token's RMS reciprocal.
    "wave_late_token": (TL, "            token = 16 // self.lines * s + own_token\n",
                        "            token = 16 // self.lines * s + (own_token + 1) % 16\n"),
    # WaveMoeTail late: the owners never scale.
    "wave_late_noscale": (TL, "value = Float32(BFloat16(sum(parts, Float32(0)) * inverse))",
                          "value = Float32(BFloat16(sum(parts, Float32(0))))"),
    # WaveMoeTail late: the owner picks the other group's reciprocal (token +- 8's).
    "wave_late_group": (TL, "        return shuffle(inverse, (token // 8) << shift)",
                        "        return shuffle(inverse, (1 - token // 8) << shift)"),
    # The env flag is read but never reaches the kernel.
    "flag_inert": (TR_, '    return batch_table(os.environ.get("K31MK_MOE_TAIL_LATE_RMS", "")).get(batch, 0)',
                   "    return 0"),
    # WaveMoeTail fp32_rms: the reader takes the RMS squares from the BF16-rounded words again (K-G3-03 itself).
    "fp32_sq_bf16": (TL, "                s2[g] = total\n", "                s2[g] = self.square_sum(xwords[g])\n"),
    # WaveMoeTail fp32_rms: the reader takes the low 16-byte half of its eight FP32 sums twice.
    "fp32_half": (TL, "                sums += [bits_float(high[g, k]) for k in range(4)]",
                  "                sums += [bits_float(low[g, k]) for k in range(4)]"),
    # WaveMoeTail fp32_rms: the previous layer's FP32 reduced rows are re-armed only up to the BF16 rows' extent (a
    # reader can take a stale row three layers later instead of waiting: race-type, stated before any run).
    "fp32_rearm_half": (TL, "                for i in cutlass.range(start, batch * 896, step, unroll=1):",
                        "                for i in cutlass.range(start, batch * 448, step, unroll=1):"),
    # The FP32-RMS env flag is read but never reaches the kernel.
    "fp32_flag_inert": (TR_, '    return bool(batch_table(os.environ.get("K31MK_MOE_TAIL_FP32_RMS", "")).get(batch, 0))',
                        "    return False"),
    # L1 PLANTED control (kernel): the FP32 GEMV / MMA sum is rounded to BF16 before the late scale (a BF16 rounding
    # of an FP32 intermediate). Must FAIL L1; never a suite mutant.
    "plant_bf16": [(TL, "                    value = value * inverse\n",
                    "                    value = Float32(BFloat16(value)) * inverse\n"),
                   (TL, "value = Float32(BFloat16(sum(parts, Float32(0)) * inverse))",
                    "value = Float32(BFloat16(Float32(BFloat16(sum(parts, Float32(0)))) * inverse))")],
}

FLAGS = ("K31MK_MOE_TAIL_POLL_NS", "K31MK_MOE_TAIL_GATE", "K31MK_MOE_TAIL_RELOAD", "K31MK_MOE_TAIL_LATE_RMS",
         "K31MK_MOE_TAIL_FP32_RMS")


def arms_for(batch):
    """name: (env, intended (poll_ns, late, fp32_rms), upstream reference arm, output reference arm, numerics order).
    F / FL (B16): KNOWN K-G3-03's fix (``K31MK_MOE_TAIL_FP32_RMS``: FP32 reduced rows, RMS of the FP32 sums)."""
    late = lambda v: {FLAGS[3]: f"{batch}:{v}"}
    z, f32 = {FLAGS[0]: "0"}, {FLAGS[4]: f"{batch}:1"}
    if batch == 8:
        return {
            "A": ({}, (None, 0, False), "A", "A", "stock"),
            "A2": ({}, (None, 0, False), "A", "A", "stock"),
            "L": (late(1), (None, 1, False), "A", "L", "late"),
            "L2": (late(1), (None, 1, False), "A", "L", "late"),
            "LO": (late(3), (None, 3, False), "A", "L", "late"),
            "Z": (z, (0, 0, False), "A", "A", "stock"),
            "ZL": ({**z, **late(1)}, (0, 1, False), "A", "L", "late"),
            "ZLO": ({**z, **late(3)}, (0, 3, False), "A", "L", "late"),
            "Z2": (z, (0, 0, False), "A", "A", "stock"),
            "ZL2": ({**z, **late(1)}, (0, 1, False), "A", "L", "late"),
        }
    return {
        "A": ({}, (None, 0, False), "A", "A", "k3"),
        "A2": ({}, (None, 0, False), "A", "A", "k3"),
        "L": (late(1), (None, 1, False), "A", "L", "late_k3"),
        "L2": (late(1), (None, 1, False), "A", "L", "late_k3"),
        "F": (f32, (None, 0, True), "A", "F", "stock"),
        "FL": ({**f32, **late(1)}, (None, 1, True), "A", "FL", "late"),
        "F2": (f32, (None, 0, True), "A", "F", "stock"),
        "FL2": ({**f32, **late(1)}, (None, 1, True), "A", "FL", "late"),
    }


ARMS = arms_for(8)


def build_consumer(device_index, tokens, words, row_words, deadline_ns):
    """One CTA per token polls the first ``words`` 32-bit words of its output row (relaxed sys-scope loads, as the MK
    polls) until none is the 0x80000000 sentinel, saves them and re-arms them (test stand-in for the next AttnRes)."""
    import cutlass
    import cutlass.cute as cute
    from cutlass import Int32, Int64, Uint32
    from cutlass.cute.runtime import make_fake_stream
    from sgl_nvidia_mega.kimi_k31.k3.cutedsl.common import primitives as P
    from sgl_nvidia_mega.kimi_k31.k3.cutedsl.common.host import fake

    class Consumer:
        @cute.jit
        def __call__(self, *args):
            P.launch(self.kernel, args, tokens, 256)

        @cute.kernel
        def kernel(self, args):
            ring, save = args
            tid, _, _, token = P.thread()
            cute.arch.griddepcontrol_wait()
            for v in cutlass.range(tid, words // 4, 256, unroll=1):
                src = ring.iterator + token * row_words + 4 * v
                got = P.load_words(src, count=4, scope="sys")
                started, polls = Int64(0), Int32(0)
                if not P.published(got):
                    started = P.wait_deadline_start(deadline_ns)
                while not P.published(got):
                    polls = P.wait_deadline_check(started, polls, deadline_ns)
                    got = P.load_words(src, count=4, scope="sys")
                P.store_words(save.iterator + token * words + 4 * v, got)
                P.store_words(src, (Uint32(0x80000000),) * 4, scope="gpu")

    args = (fake((tokens * row_words,), None, torch.int32), fake((tokens * words,), None, torch.int32))
    with torch.accelerator.device_index(device_index):
        return cute.compile(Consumer(), *args, make_fake_stream())


def layer_inputs(device):
    """x of layer i = BF16(base_r + shift_i + s roll(noise_r, i)) per 8-row request group r (common to every GPU);
    ``seed`` > 0 (L1 sets) adds a fresh seeded N(0, 0.5 s) term to every row."""
    groups = B // 8
    base = [randn((HID,), 1.0, 777 + 10 * r, device).double() for r in range(groups)]
    noise = [randn((8, HID), 1.0, 778 + 10 * r, device).double() for r in range(groups)]
    shifts = [randn((HID,), 0.3, 900 + layer, device).double() for layer in LAYER_IDS]

    def make(i, s, seed=0):
        rows = torch.cat([base[r] + shifts[i] + s * torch.roll(noise[r], i, 0) for r in range(groups)])
        if seed:
            rows = rows + randn((B, HID), 0.5 * s, 50000 + seed, device).double()
        return canon_bf16(rows.to(torch.bfloat16)).contiguous()

    return make


def choose_spread(models, make_x):
    """The spread whose FP64 routing gives a median distinct-expert count per 8-row group closest to TARGET_ED."""
    best = None
    for s in (0.1, 0.2, 0.3, 0.5, 0.7, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0, 2.5, 3.0, 4.0):
        eds = []
        for i, m in enumerate(models):
            x = make_x(i, s)
            for r in range(B // 8):
                ids = set()
                for t in range(8 * r, 8 * r + 8):
                    ids.update(route_ref(m, x[t])[1].tolist())
                eds.append(len(ids))
        med = statistics.median(eds)
        if best is None or abs(med - TARGET_ED) < abs(best[1] - TARGET_ED):
            best = (s, med, eds)
    return best


def injected(lay, x, world, device, seed=0, scale=1.0, shared=True):
    """Ranks world..15's inputs for layer ``lay`` and rows ``x``: FP32 latent / logits shards (from FP64), BF16 latent
    partials N(0, 0.5 scale) and shared rows N(0, 0.5), or +0 without ``shared`` (seeded identically on every GPU)."""
    layer, m = lay.layer, lay.m
    k = 2000 + 100 * layer + 7919 * seed
    lay.lat_ref = x.double() @ m.down.double().T
    logits_ref = x.double() @ m.router.double().T
    lay.inj_s = torch.stack([randn((B, HID), 0.5, k + 1000 + q, device) for q in range(world, TP)])
    if not shared:
        lay.inj_s = torch.zeros_like(lay.inj_s)
    lay.inj_lat = M.canon(lay.lat_ref.float()[:, world * LROWS:])
    lay.inj_logits = M.canon(logits_ref.float()[:, world * LOGITS:])
    lay.inj_partials = torch.stack([canon_bf16(randn((B, LAT), 0.5 * scale, k + q, device))
                                    for q in range(world, TP)])
    lay.inj_rs = canon_bf16(lay.inj_s[:, :, lay.g * SHARD:(lay.g + 1) * SHARD]).contiguous()


class Arm:
    def __init__(self, name, rig):
        env = ARMS[name][0]
        mods = rig.mods
        self.name, self.rig = name, rig
        g, d, group = rig.g, rig.device, rig.group
        saved = {k: os.environ.get(k) for k in FLAGS}
        for k in FLAGS:
            os.environ.pop(k, None)
        os.environ.update(env)
        try:
            self.front_box = mods.OP.OprojMailbox(group, d, TP, g, (1, 64, 2 * LAT + 2 * 896))
            self.state = mods.PM.MoEScratch(TP, d, (B,))
            tb = self.tail_box = mods.TR.MoeTailBox(group, d, TP, g)
            self.scratch = [torch.empty_like(tb.rs.buffer).view(torch.int32).fill_(SENT).view(torch.bfloat16)
                            for _ in range(TP - rig.world)]
            ptrs = list(tb.rs.handle.buffer_ptrs) + [s.data_ptr() for s in self.scratch]
            self.peers = torch.tensor(ptrs, dtype=torch.int64, device=d)
            tb.tail = (mods.host.to_cute(self.peers, 16), mods.TR.flat(tb.out.buffer),
                       rig.Int64(tb.latent.mc), rig.Int64(tb.out.mc))
            op = SimpleNamespace(rank=g, contract=SimpleNamespace(tp_size=TP, rms_eps=EPS))
            self.layers = []
            for i, lay in enumerate(rig.layers):
                experts = rig.experts_cls(*lay.host_w, beta=R.BETA, linear_beta=R.LIN, activation_floor=False,
                                          **rig.planned_options)
                tail = mods.TR.PreparedTail(op, lay.m.gamma, lay.m.up, (B,), (tb, i, NLAYERS),
                                            deadline_ns=rig.deadline_ns)
                front = mods.FR.SharedFront(lay.m.moe(), lay.packed, self.front_box, self.state.front, (B,), R.BETA,
                                            R.LIN, plan=self.state.plan, counters=self.state.shared, q=8,
                                            deadline_ns=rig.deadline_ns)
                pm = mods.PM.PreparedMoE(front, experts, tail, self.state, prefetch=None, layer_idx=lay.layer)
                self.layers.append(SimpleNamespace(pm=pm, tail=tail, experts=experts, parity=i % 3))
            self.grid = self.layers[0].experts.grid_for(B)
            self.published = B in self.layers[0].tail.pushers
            # B9+: the fused / split shared expert peer-stores its rows into the RS mailboxes itself and leaves
            # storage[5] (and the FP32 gate/up storage[1]) unwritten (SharedFront.shared_expert).
            self.fused = self.layers[0].pm.front.plans[B][6] is not None
        finally:
            for k, v in saved.items():
                if v is None:
                    os.environ.pop(k, None)
                else:
                    os.environ[k] = v
        self.saves = [torch.full((B, rig.world * SHARD // 2), SENT, dtype=torch.int32, device=d)
                      for _ in range(NLAYERS)]
        H.record({"kind": "arm", "arm": name, "env": env, "grid": self.grid, "published": self.published,
                  "fused_shared": self.fused, "g": g})

    def inject(self, i):
        lay, w = self.rig.layers[i], self.rig.world
        row = self.front_box.buffer[0, :B].view(torch.float32)  # [B, 4480]: latent 3584 | logits 896
        row[:, w * LROWS: LAT].copy_(lay.inj_lat)
        row[:, LAT + w * LOGITS:].copy_(lay.inj_logits)
        t = i % 3
        self.tail_box.latent.buffer[t, w:TP, :B].copy_(lay.inj_partials)
        self.tail_box.rs.buffer[t, w:TP, :B].copy_(lay.inj_rs)

    def run_layer(self, i):
        rig = self.rig
        out = self.tail_box.out.buffer[0, :B]
        self.layers[i].pm.run(rig.x[i], out, consume=lambda o: o)
        ring = self.tail_box.out.buffer[0].view(torch.int32)
        rig.consumer(rig.mods.host.to_cute(ring.reshape(-1), 16), rig.mods.host.to_cute(self.saves[i].reshape(-1), 16),
                     rig.mods.host.stream(rig.device))

    def chain(self):
        for i in range(NLAYERS):
            self.inject(i)
            self.run_layer(i)


def order_ref(order, Lsum, gamma, up_n, sh_n):
    """FP64 emulation of an arm's rounding order (module docstring) for one token: (out, y) over the first n columns."""
    xb = bf(Lsum)
    ms = (xb ** 2).mean() if order.endswith("k3") else (Lsum ** 2).mean()
    inv = 1.0 / torch.sqrt(ms + EPS)
    if order.startswith("late"):
        y = bf(inv * (up_n @ bf(xb * gamma)))
    else:
        y = bf(up_n @ bf(xb * inv * gamma))
    return bf(y + sh_n), y


class Rig:
    def __init__(self, mods, group, g, world, device, deadline_ns):
        from cutlass import Int64

        self.mods, self.group, self.g, self.world, self.device = mods, group, g, world, device
        self.Int64, self.deadline_ns = Int64, deadline_ns
        self.sms = torch.cuda.get_device_properties(device).multi_processor_count
        self.experts_cls, self.planned_options = mods.ER.PersistentExperts, {}
        if mods.PL.enabled():  # as the providers build it (falls back to the shipped engine at TP16)
            self.experts_cls, self.planned_options = mods.PL.PlannedExperts, mods.PL.options()
        models = [Model(g, device, layer) for layer in LAYER_IDS]
        self.make_x = make_x = layer_inputs(device)
        self.spread, med, eds = choose_spread(models, make_x)
        H.record({"kind": "routing", "spread": self.spread, "ed_median": med, "ed_layers": eds, "g": g})
        print(f"ROUTING spread {self.spread} E_d median {med} groups {eds}", flush=True)
        self.layers, self.x = [], []
        for i, (layer, m) in enumerate(zip(LAYER_IDS, models)):
            x = make_x(i, self.spread)
            self.x.append(x)
            logical = M.make_logical16(R.E_ALL, gen(1600 + 37 * layer + g, device))
            router = m.router[g * LOGITS:(g + 1) * LOGITS]
            down = m.down[g * LROWS:(g + 1) * LROWS]
            packed = torch.cat((router, router.new_zeros((-LOGITS % 16, HID)), down)).contiguous()
            lay = SimpleNamespace(layer=layer, m=m, g=g, logical=logical, host_w=M.host_layout16(*logical),
                                  packed=packed)
            injected(lay, x, world, device)
            self.layers.append(lay)
        self.consumer = build_consumer(device.index, B, world * SHARD // 2, HID // 2, deadline_ns)
        self.arms = {}
        self.next_layer = 0  # every chain (eager checks, graph replays, timing) ends at layer NLAYERS - 1

    def gather(self, t):
        out = [torch.empty_like(t) for _ in range(self.world)]
        dist.all_gather(out, t.contiguous(), group=self.group)
        return out

    def sync(self):
        torch.cuda.synchronize()
        dist.barrier(group=self.group)

    def shared_rows(self, arm, t):
        """This rank's shared-expert rows [B, 7168] as stored in the owners' RS mailboxes (parity t): owner o < W's
        shard from its GPU (collective: every rank calls this), owners W..15's from the local scratch. The fused
        shared expert (B9+) writes only these; the unfused path's storage[5] holds the same values (-0 as +0)."""
        g, w = self.g, self.world
        got = self.gather(arm.tail_box.rs.buffer[t, :w, :B].clone())  # got[o][q] = rank q's shard for owner o
        parts = [got[o][g] for o in range(w)] + [arm.scratch[o - w][t, g, :B] for o in range(w, TP)]
        return torch.cat(parts, -1)

    # ------------------------------------------------------------------ correctness
    def check_layer(self, arm, i, fails, ref_rows):
        g, w, lay = self.g, self.world, self.layers[i]
        m, t, st, tb = lay.m, i % 3, arm.state, arm.tail_box
        order = ARMS[arm.name][4]
        ids = st.front[2][:B].clone()
        wts = st.front[3][:B].clone()
        latent = st.front[6][:B].clone()
        hg = st.front[1][:B].clone()
        h_all = st.front[4][:B].clone()
        S = self.shared_rows(arm, t) if arm.fused else st.front[5][:B].clone()
        lat_gpus = self.gather(latent)
        slots = tb.latent.buffer[t, :TP, :B].clone()  # [16, B, 3584]
        allslots = self.gather(slots[:w].contiguous())
        S_all = self.gather(S)
        saves_all = self.gather(arm.saves[i])
        rows = None if arm.published else st.expanded[: B * 16].clone().view(B, 16, LAT)
        x = self.x[i]
        G_ref = M.expert_rows(lay.logical, latent, ids)  # [B, 16, 3584] FP64 on the MK's own latent and ids
        ref_rows[(arm.name, i)] = (ids, latent, slots[g].clone(), S, rows)
        n = w * SHARD
        up_n = m.up[:n].double()
        gamma = m.gamma.double()
        for tok in range(B):
            ck = Checker(f"b{B}.{arm.name}.l{lay.layer}.t{tok}.g{g}", fails)
            ck.route(m, x[tok], ids[tok], wts[tok])
            ck.close("latent", latent[tok], lay.lat_ref[tok], slice(0, w * LROWS))
            if not torch.equal(latent[tok, w * LROWS:].view(torch.int32), lay.inj_lat[tok].view(torch.int32)):
                ck.fail("latent_inj", "injected latent rows changed")
            if not all(torch.equal(o[tok].view(torch.int32), latent[tok].view(torch.int32)) for o in lat_gpus):
                ck.fail("latent_multicast", "latent words differ across GPUs")
            h_mk = h_all[tok].double()
            if arm.fused:
                # The fused gate/up keeps its FP32 output on chip (storage[1] unwritten): h vs BF16(SiTU_FP64) of
                # the FP64 gate/up of x, within 2 BF16 ulp + 2^-8 rms(h) (FP32 accumulation + stock's approximate
                # SiTU; near a zero of the gate or up the FP32 gate/up error is amplified, so a pure ulp bound cannot
                # hold without the MK's own gate/up: GB300 g2 at 2 ulp alone, every arm alike, up to 5 of 384 cells).
                hg64 = m.gate_up.double() @ x[tok].double()
                h_ref = bf(R.situ(hg64[:384], hg64[384:]).float())
                lim_h = 2 * ulp_bf16(torch.maximum(h_mk.abs(), h_ref.abs())) + 2 ** -8 * float(h_ref.pow(2).mean().sqrt())
            else:
                ck.close("hg", hg[tok], m.gate_up.double() @ x[tok].double())
                # S1: the MK's BF16 hidden within 1 BF16 ulp of BF16(SiTU_FP64) of its own gate/up (stock SiTU
                # is approx).
                h_ref = bf(R.situ(hg[tok, :384].double(), hg[tok, 384:].double()).float())
                lim_h = ulp_bf16(torch.maximum(h_mk.abs(), h_ref.abs()))
            nh = int(((h_mk - h_ref).abs() > lim_h).sum())
            # S2: S vs FP64 shared_down @ h_mk (the source rig's tolerance).
            s_ref = m.shared_down.double() @ h_mk
            tol = 2 ** -7 * s_ref.abs() + 2 ** -8 * float(s_ref.pow(2).mean().sqrt())
            ns = int(((S[tok].double() - s_ref).abs() > tol).sum())
            H.record({"kind": "check", "item": f"{ck.name}.shared", "h_violations": nh, "s_violations": ns,
                      "h_diff_cells": int((h_mk != h_ref).sum())})
            if nh or not torch.isfinite(h_mk).all():
                ck.fail("shared_h", f"{nh} hidden cells beyond the h bound of BF16(SiTU_FP64)")
            if ns or not torch.isfinite(S[tok].float()).all():
                ck.fail("shared", f"{ns} cells vs down @ h_mk")
            wd = wts[tok].double().view(16, 1)
            part_ref = (wd * G_ref[tok]).sum(0)
            mine = slots[g, tok].double()
            fro = float((mine - part_ref).norm() / part_ref.norm())
            H.record({"kind": "check", "item": f"{ck.name}.partial_t1", "rel_l2": fro})
            if not (torch.isfinite(mine).all() and fro <= 3e-2):
                ck.fail("partial_t1", f"row rel-L2 {fro:.3e} > 3e-2")
            if rows is not None:
                G = rows[tok]
                gf = R.rel_l2(G.view(1, 16, LAT), G_ref[tok].view(1, 16, LAT))
                grow = float(((G.double() - G_ref[tok]).norm(dim=-1) / G_ref[tok].norm(dim=-1)).max())
                H.record({"kind": "check", "item": f"{ck.name}.experts", "t1_fro": gf, "t1_row_max": grow})
                if not (torch.isfinite(G.float()).all() and gf <= 1e-2 and grow <= 3e-2):
                    ck.fail("experts", f"T1 {gf:.3e} {grow:.3e}")
                terms = wd * G.double()
                lim = ulp_bf16(torch.maximum(terms.sum(0).abs(), terms.abs().max(0).values))
                nb = int(((mine - terms.sum(0)).abs() > lim).sum())
                if nb:
                    ck.fail("partial", f"{nb} cells beyond 1 ulp of BF16(sum w G)")
            for q in range(w):
                if not torch.equal(allslots[q][q, tok].view(torch.int16), slots[q, tok].view(torch.int16)):
                    ck.fail("multicast", f"latent partial slot {q} differs from rank {q}'s own copy")
            # Fused path: S IS the RS mailbox contents, so its routing is checked by the shared check above (every
            # owner's shard vs the FP64 row); the unfused path checks the shards against storage[5] bit for bit.
            for q in range(0 if arm.fused else w):
                want = canon_bf16(S_all[q][tok, g * SHARD:(g + 1) * SHARD])
                if not torch.equal(tb.rs.buffer[t, q, tok].view(torch.int16), want.view(torch.int16)):
                    ck.fail("rs", f"shard from rank {q} differs")
            for o in range(TP if arm.fused else w, TP):
                got = arm.scratch[o - w][t, g, tok]
                want = canon_bf16(S[tok, o * SHARD:(o + 1) * SHARD])
                if not torch.equal(got.view(torch.int16), want.view(torch.int16)):
                    ck.fail("rs_out", f"shard for owner {o} differs")
            Lsum = slots[:, tok].double().sum(0)
            S_full = [S_all[q][tok].double() for q in range(w)] + [lay.inj_s[q - w][tok].double() for q in range(w, TP)]
            sh = bf(sum(S_full))
            out_ref, y = order_ref(order, Lsum, gamma, up_n, sh[:n])
            out = arm.saves[i][tok].view(torch.bfloat16).double()
            lim = 2 * ulp_bf16(torch.maximum(torch.maximum(out_ref.abs(), y.abs()), sh[:n].abs()))
            err = (out - out_ref).abs()
            nbad = int((err > lim).sum())
            H.record({"kind": "check", "item": f"{ck.name}.output", "order": order, "max_abs": float(err.max()),
                      "max_ulps": float((err / ulp_bf16(out_ref.abs())).max()), "violations": nbad})
            if nbad or not torch.isfinite(out).all():
                for c in torch.topk(err - lim, min(4, n)).indices.tolist():
                    H.record({"kind": "diag", "item": f"{ck.name}.output", "col": c, "out": float(out[c]),
                              "ref": float(out_ref[c]), "y": float(y[c]), "sh": float(sh[c]), "lim": float(lim[c])})
                ck.fail("output", f"{nbad} cells beyond 2 ulp of the {order} order; max {float(err.max()):.3e}")
            if not all(torch.equal(o[tok], arm.saves[i][tok]) for o in saves_all):
                ck.fail("rank_bitwise", "output rows differ across GPUs")
        part_ref = (wts.double().view(B, 16, 1) * G_ref).sum(1)
        fro = R.rel_l2(slots[g], part_ref)
        H.record({"kind": "check", "item": f"b{B}.{arm.name}.l{lay.layer}.g{g}.partial_fro", "rel_l2": fro})
        if not fro <= 1e-2:
            fails.append(f"b{B}.{arm.name}.l{lay.layer}.g{g}.partial_fro: {fro:.3e} > 1e-2")
            print("FAIL partial_fro", arm.name, lay.layer, fro, flush=True)

    def correctness(self, arm, fails, ref_rows):
        for s in arm.saves:
            s.fill_(SENT)
        for i in range(NLAYERS):
            self.sync()
            arm.inject(i)
            arm.run_layer(i)
            self.sync()
            self.check_layer(arm, i, fails, ref_rows)
        eager = [s.clone() for s in arm.saves]
        graph, stream = torch.cuda.CUDAGraph(), torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        self.sync()
        with torch.cuda.graph(graph, stream=stream):
            arm.chain()
        self.sync()
        for rep in range(2):
            for s in arm.saves:
                s.fill_(SENT)
            self.sync()
            graph.replay()
            self.sync()
            for i in range(NLAYERS):
                same = torch.equal(arm.saves[i], eager[i])
                H.record({"kind": "check", "item": f"b{B}.{arm.name}.graph{rep}.l{i}.g{self.g}",
                          "bitwise_vs_eager": same})
                if not same:
                    fails.append(f"b{B}.{arm.name}.graph{rep}.l{i}.g{self.g}: differs from eager")
                    print("FAIL graph", arm.name, rep, i, flush=True)
        arm.graph, arm.eager = graph, eager

    def intended_tail(self, ns, late, fp32=False):
        """compile_tail for the given options, called exactly as MoeTailWeights calls it (same cache key)."""
        TRm, dev, dl = self.mods.TR, self.device.index, self.deadline_ns
        extra = dict(late_rms=late) if late else {}
        if fp32:
            extra["fp32_rms"] = True
        if B <= 8:
            return TRm.compile_tail(TP, B, dev, "front", early=False, umma=False, deadline_ns=dl, poll_ns=ns,
                                    gate=False, reload=False, **extra)
        return TRm.compile_tail(TP, B, dev, "publish", early=True, umma=False, deadline_ns=dl, poll_ns=None,
                                gate=False, reload=False, **extra)

    def engaged(self, arm, fails):
        """The arm's tail is compile_tail's object for its INTENDED options (ARMS), never what the code chose."""
        ns, late, fp32 = ARMS[arm.name][1]
        code = arm.layers[0].tail.tail_weights.codes[B][0]
        want, base = self.intended_tail(ns, late, fp32), self.intended_tail(ns, 0)
        bad = []
        if code is not want:
            bad.append("tail code is not compile_tail(intended options)")
        if (late or fp32) and code is base:
            bad.append("option arm runs the default tail")
        H.record({"kind": "check", "item": f"b{B}.{arm.name}.engaged.g{self.g}", "intended": [ns, late, fp32],
                  "bad": bad})
        if bad:
            fails.append(f"b{B}.{arm.name}.engaged.g{self.g}: {bad}")
            print("FAIL engaged", arm.name, bad, flush=True)

    # ------------------------------------------------------------------ profiling / timing
    def profile(self, arm, replays):
        from torch.profiler import ProfilerActivity, profile

        self.sync()
        with profile(activities=[ProfilerActivity.CUDA]) as prof:
            for _ in range(replays):
                arm.graph.replay()
            torch.cuda.synchronize()
        path = os.path.join(tempfile.mkdtemp(prefix="k31l17_tail_"), "trace.json")
        prof.export_chrome_trace(path)
        with open(path) as f:
            events = json.load(f)["traceEvents"]
        os.remove(path)
        ks = sorted((e for e in events if e.get("cat") == "kernel"), key=lambda e: e["ts"])
        seg_t, seg_m1, route_end, experts_end, kinds = [], [], None, None, {}
        for e in ks:
            name, grid = e["name"], tuple(e.get("args", {}).get("grid") or (0, 0, 0))
            end = e["ts"] + e["dur"]
            key = None
            if "RouteSitu" in name and grid[0] == 2 * B:
                route_end = end
            elif "ExpertsGroup" in name:
                key = "group"
            elif "Experts" in name and "Plan" not in name:
                key = f"experts{grid[0]}"
            elif "push_kernel" in name:
                key = "push"
            elif "WaveMoeTail" in name:
                key = f"wave{grid[0]}"
            elif "MoeTail" in name:
                key = f"moetail{grid[0]}"
            if key:
                kinds.setdefault(key, []).append(e["dur"])
            if key and key.startswith("experts"):
                experts_end = end
            if key and key.startswith(("wave", "moetail")):
                if experts_end is not None:
                    seg_t.append(end - experts_end)
                if route_end is not None:
                    seg_m1.append(end - route_end)
                route_end, experts_end = None, None
        return seg_t, seg_m1, {k: (len(v), statistics.median(v)) for k, v in kinds.items()}

    def path_check(self, arm, kinds, fails):
        want_e = f"experts{arm.grid}"
        want_t = "wave112" if B > 8 else "moetail448"
        bad = []
        if want_e not in kinds or any(k.startswith("experts") and k != want_e for k in kinds):
            bad.append(f"experts kernels {sorted(k for k in kinds if k.startswith('experts'))} != {want_e}")
        if want_t not in kinds or any(k.startswith(("wave", "moetail")) and k != want_t for k in kinds):
            bad.append(f"tail kernels {sorted(k for k in kinds if k.startswith(('wave', 'moetail')))} != {want_t}")
        push = arm.published and not arm.fused  # the fused shared expert peer-stores its rows itself
        if ("push" in kinds) != push:
            bad.append(f"push kernel present={'push' in kinds}, expected {push}")
        H.record({"kind": "check", "item": f"b{B}.{arm.name}.path.g{self.g}", "kinds": kinds, "bad": bad})
        if bad:
            fails.append(f"b{B}.{arm.name}.path.g{self.g}: {bad}")
            print("FAIL path", arm.name, bad, flush=True)

    def time_arm(self, arm, rep, order, replays=20):
        for _ in range(20):
            arm.graph.replay()
        self.sync()
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(100):
            arm.graph.replay()
        end.record()
        torch.cuda.synchronize()
        m2 = start.elapsed_time(end) * 1000.0 / (100 * NLAYERS)
        m2_all = self.gather(torch.tensor([m2], device=self.device))
        seg_t, seg_m1, kinds = self.profile(arm, replays)
        med = lambda s: statistics.median(s) if s else float("nan")
        pct = lambda s, p: sorted(s)[int(p * len(s))] if s else None

        def tmean(s):  # mean of the central 80% (CUPTI timestamps are quantized, so medians jump in quanta)
            s = sorted(s)
            k = len(s) // 10
            return statistics.fmean(s[k:len(s) - k]) if len(s) > 2 * k else float("nan")

        rec = {"kind": "time", "arm": arm.name, "rep": rep, "order": order, "g": self.g, "batch": B,
               "t_us": med(seg_t), "t_n": len(seg_t), "t_p10": pct(seg_t, 0.1), "t_p90": pct(seg_t, 0.9),
               "t_tmean": tmean(seg_t), "m1_tmean": tmean(seg_m1), "t_samples": [round(v, 3) for v in seg_t],
               "m1_us": med(seg_m1), "m1_n": len(seg_m1), "m2_us": m2, "m2_max_us": max(float(v) for v in m2_all),
               "kernels": kinds}
        H.record(rec)
        print("TIME", json.dumps({k: v for k, v in rec.items() if k != "t_samples"}), flush=True)
        return rec

    # ------------------------------------------------------------------ L1 (kernel vs FP64)
    def l1_sets(self, arms, n_sets, seed0, fails, shared=True):
        """L1 sets at the tail output (``shared``: + the shared rows) or at the routed output y (shared rows zero:
        the shared-expert down weights of ranks 0-3 zeroed in place for these sets, injected shared rows +0; the
        weights are restored afterwards). Pre-registered."""
        g, w = self.g, self.world
        mode = "shared" if shared else "noshared"
        saved = []
        if not shared:
            for lay in self.layers:
                saved.append(lay.m.shared_down.clone())
                lay.m.shared_down.zero_()  # the prepared projections / TMA maps read this memory
            torch.cuda.synchronize()
        n = w * SHARD
        F = torch.nn.functional
        torch.backends.cuda.matmul.allow_tf32 = False
        for s in range(n_sets):
            # Layers run in chain order across modes: a layer re-arms its predecessor's parity, so after layer j only
            # layer j + 1 may run (GB300 g2: restarting at layer 0 after 64 sets read layer 3's stale slots).
            i, seed = self.next_layer, seed0 + s + 1
            self.next_layer = (i + 1) % NLAYERS
            lay = self.layers[i]
            # Per-set scale of the injected partials, log-uniform in [1/8, 8] (seeded; common to every GPU).
            u = torch.rand((1,), generator=gen(90000 + seed, self.device), device=self.device)
            scale = float(torch.exp2(6.0 * u - 3.0))
            self.x[i] = self.make_x(i, self.spread, seed)
            injected(lay, self.x[i], w, self.device, seed, scale, shared)
            outs, slots_by, S_by = {}, {}, {}
            for name in arms:
                arm = self.arms[name]
                arm.saves[i].fill_(SENT)
                self.sync()
                arm.inject(i)
                arm.run_layer(i)
                self.sync()
                outs[name] = arm.saves[i].view(torch.bfloat16).clone()
                slots_by[name] = arm.tail_box.latent.buffer[i % 3, :TP, :B].clone()
                own = self.shared_rows(arm, i % 3) if arm.fused else arm.state.front[5][:B].clone()
                S_by[name] = torch.cat(self.gather(own), 0).view(w, B, HID)
            ref_arm = arms[0]
            same = all(torch.equal(slots_by[a].view(torch.int16), slots_by[ref_arm].view(torch.int16)) and
                       torch.equal(S_by[a].view(torch.int16), S_by[ref_arm].view(torch.int16)) for a in arms)
            if not same:
                fails.append(f"l1.set{s}.g{g}: arms saw different tail inputs")
                print("FAIL l1 inputs", s, flush=True)
            if g != 0:
                continue
            m = lay.m
            slots = slots_by[ref_arm]  # [16, B, 3584] BF16
            S16 = torch.cat([S_by[ref_arm], lay.inj_s], 0)  # [16, B, 7168] BF16 (ranks 0-3 real, 4-15 injected)
            digest = hashlib.sha256(slots.view(torch.int16).cpu().numpy().tobytes() +
                                    S16[:, :, :n].contiguous().view(torch.int16).cpu().numpy().tobytes()).hexdigest()
            # REF: FP64 exact math.
            L64 = slots.double().sum(0)
            S64 = S16[:, :, :n].double().sum(0)
            inv64 = 1.0 / torch.sqrt((L64 ** 2).mean(-1, keepdim=True) + EPS)
            ref = (L64 * inv64 * m.gamma.double()) @ m.up[:n].double().T + S64
            # S: stock (ar_fusion order, cuBLAS up GEMM over all 7168 rows as the replicated stock layer).
            acc = torch.zeros(B, LAT, dtype=torch.float32, device=self.device)
            for r in range(TP):
                acc = acc + slots[r].float()
            xb = acc.to(torch.bfloat16)
            inv = torch.rsqrt((acc * acc).sum(-1, keepdim=True) / LAT + EPS)
            normed = ((xb.float() * inv) * m.gamma.float()).to(torch.bfloat16)
            y_s = F.linear(normed, m.up)[:, :n]
            sha = torch.zeros(B, n, dtype=torch.float32, device=self.device)
            for r in range(TP):
                sha = sha + S16[r, :, :n].float()
            shb = sha.to(torch.bfloat16)
            emu = {"S": (y_s.float() + shb.float()).to(torch.bfloat16)}
            # Lt (late order in torch) and Pt (PLANTED: BF16 rounding of the FP32 GEMM sum before the scale).
            xg = (xb.float() * m.gamma.float()).to(torch.bfloat16)
            yl = xg.float() @ m.up[:n].float().T
            emu["Lt"] = ((yl * inv).to(torch.bfloat16).float() + shb.float()).to(torch.bfloat16)
            emu["Pt"] = ((yl.to(torch.bfloat16).float() * inv).to(torch.bfloat16).float() + shb.float()).to(
                torch.bfloat16)
            if not shared and int((S16.view(torch.int16) & 0x7FFF).max()) != 0:  # +0 / -0 only
                fails.append(f"l1.{mode}.set{s}.g{g}: shared rows not zero")
                print("FAIL l1 noshared rows", s, flush=True)
            rec = {"kind": "l1", "mode": mode, "set": s, "layer": i, "seed": seed, "scale": scale, "batch": B,
                   "digest": digest,
                   "inputs_same": same, "x_rms": float((L64 ** 2).mean().sqrt()), "err": {}, "maxabs": {},
                   "p999": {}, "max_ulp": {}, "bitwise_vs_S": {}}
            rn = ref.norm(dim=-1)
            rms = (ref ** 2).mean(-1, keepdim=True).sqrt()
            ref_bf = bf(ref.float())
            for name, out in list(outs.items()) + list(emu.items()):
                e = out.double() - ref
                rec["err"][name] = float((e.norm(dim=-1) / rn).mean())
                rec["maxabs"][name] = float(e.abs().max())
                rec["p999"][name] = float(torch.quantile((e.abs() / rms).flatten().float(), 0.999))
                rec["max_ulp"][name] = float(((out.double() - ref_bf).abs() / ulp_bf16(ref_bf.abs())).max())
                rec["bitwise_vs_S"][name] = float((out.view(torch.int16) == emu["S"].view(torch.int16)).float().mean())
            H.record(rec)
            if s % 16 == 0 or s == n_sets - 1:
                print("L1", json.dumps({k: rec[k] for k in ("mode", "set", "layer", "scale", "err")}), flush=True)
        if saved:
            for lay, wgt in zip(self.layers, saved):
                lay.m.shared_down.copy_(wgt)
            torch.cuda.synchronize()


def cross_arm(rig, ref_rows, fails):
    """Gate terms: upstream values equal the upstream reference arm's, outputs the output reference arm's, bit for
    bit; and every late arm's outputs differ from A's somewhere (the reorder is in effect)."""
    for name, arm in rig.arms.items():
        _, (_, late, fp32), up_ref, out_ref, _ = ARMS[name]
        for ref, what in ((up_ref, "upstream"), (out_ref, "output")):
            if ref not in rig.arms:
                fails.append(f"b{B}.{name}.g{rig.g}.bitwise: reference arm {ref} not in the run")
                continue
            a = rig.arms[ref]
            for i in range(NLAYERS):
                ra, rx = ref_rows[(ref, i)], ref_rows[(name, i)]
                if what == "upstream":
                    same = {"ids": torch.equal(ra[0], rx[0]),
                            "partial": torch.equal(ra[2].view(torch.int16), rx[2].view(torch.int16)),
                            "shared": torch.equal(ra[3].view(torch.int16), rx[3].view(torch.int16))}
                    if ra[4] is not None and rx[4] is not None:
                        same["experts"] = torch.equal(ra[4].view(torch.int16), rx[4].view(torch.int16))
                else:
                    same = {"output": torch.equal(a.eager[i], arm.eager[i])}
                av, bv = a.eager[i].view(torch.bfloat16).double(), arm.eager[i].view(torch.bfloat16).double()
                ulps = float(((av - bv).abs() / ulp_bf16(av.abs())).max())
                H.record({"kind": "cross", "arm": name, "ref": ref, "what": what, "layer": i, "g": rig.g,
                          "output_max_ulp": ulps, **same})
                bad = [k for k, v in same.items() if not v]
                if bad:
                    fails.append(f"b{B}.{name}.l{i}.g{rig.g}.bitwise_vs_{ref}: {bad} (output max {ulps:.2f} ulp)")
                    print(f"FAIL bitwise_vs_{ref}", name, i, bad, ulps, flush=True)
        if (late or fp32) and "A" in rig.arms:
            a = rig.arms["A"]
            diff = sum(int((a.eager[i] != arm.eager[i]).sum()) for i in range(NLAYERS))  # int32 words differing
            cells = sum(int((a.eager[i].view(torch.int16) != arm.eager[i].view(torch.int16)).sum())
                        for i in range(NLAYERS))
            H.record({"kind": "check", "item": f"b{B}.{name}.reorder.g{rig.g}", "cells_differing_vs_A": cells,
                      "words": diff, "cells_total": NLAYERS * B * rig.world * SHARD})
            if cells == 0:
                fails.append(f"b{B}.{name}.reorder.g{rig.g}: outputs bitwise equal to A (option not in effect)")
                print("FAIL reorder", name, flush=True)


def main():
    global B, ARMS
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--batch", type=int, default=8, choices=(8, 16))
    ap.add_argument("--deadline-s", type=float, default=2.0)
    ap.add_argument("--arms", default="")
    ap.add_argument("--time", type=int, default=0, help="timing replicates (0: correctness only)")
    ap.add_argument("--replays", type=int, default=20, help="profiled graph replays per timing replicate")
    ap.add_argument("--time-always", action="store_true", help="time even when a check failed (screening only)")
    ap.add_argument("--no-profile", action="store_true", help="skip the profiler path check (compute-sanitizer runs)")
    ap.add_argument("--l1", type=int, default=0, help="L1 input sets (0: none)")
    ap.add_argument("--l1-arms", default="A,L")
    ap.add_argument("--l1-seed0", type=int, default=0)
    ap.add_argument("--l1-modes", default="shared,noshared", help="L1 points: tail output (shared), routed y (noshared)")
    args = ap.parse_args()
    B, ARMS = args.batch, arms_for(args.batch)
    arms = tuple((args.arms or ",".join(ARMS)).split(","))
    if arms[0] != "A" or any(a not in ARMS for a in arms):
        raise SystemExit(f"arm A (today) must be first and every arm one of {list(ARMS)}")
    local = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local)
    device = torch.device("cuda", local)
    dist.init_process_group("nccl", device_id=device)
    g, world = dist.get_rank(), dist.get_world_size()
    tag = os.environ.get("K31_MUTATION") or "base"
    os.makedirs(args.out, exist_ok=True)
    os.environ["K31_RESULTS"] = os.path.join(args.out, f"b{B}_{tag}_g{g}.jsonl")
    H.load_plugin(MUTATIONS)
    from sgl_nvidia_mega.kimi_k31.k3.cutedsl.common import host
    from sgl_nvidia_mega.kimi_k31.k3.cutedsl.moe import experts_runtime as ER
    from sgl_nvidia_mega.kimi_k31.k3.cutedsl.moe import front as FR
    from sgl_nvidia_mega.kimi_k31.k3.cutedsl.moe import planned as PL
    from sgl_nvidia_mega.kimi_k31.k3.cutedsl.moe import prepared as PM
    from sgl_nvidia_mega.kimi_k31.k3.cutedsl.moe import tail_runtime as TR
    from sgl_nvidia_mega.kimi_k31.k3.cutedsl.oproj import prepared as OP

    mods = SimpleNamespace(host=host, ER=ER, FR=FR, PL=PL, PM=PM, TR=TR, OP=OP)
    H.record({"kind": "env", **H.device_facts(), "world": world, "arms": list(arms), "mutation": tag, "batch": B})
    fails = []
    t0 = time.time()
    try:
        rig = Rig(mods, dist.group.WORLD, g, world, device, int(args.deadline_s * 1e9))
        ref_rows = {}
        for name in arms:
            t = time.time()
            arm = rig.arms[name] = Arm(name, rig)
            print(f"ARM {name} built grid {arm.grid} published {arm.published} ({time.time() - t:.0f}s)", flush=True)
            rig.correctness(arm, fails, ref_rows)
            rig.engaged(arm, fails)
            kinds = None
            if not args.no_profile:
                _, _, kinds = rig.profile(arm, 3)
                rig.path_check(arm, kinds, fails)
            print(f"ARM {name} checked: fails so far {len(fails)} kinds {kinds}", flush=True)
        cross_arm(rig, ref_rows, fails)
        total = torch.tensor([len(fails)], device=device)
        dist.all_reduce(total)  # every rank takes the same branch (the chain polls all four)
        if args.time and (args.time_always or int(total) == 0):
            for rep in range(args.time):
                order = list(arms) if rep % 2 == 0 else list(reversed(arms))
                for k, name in enumerate(order):
                    rig.sync()
                    rig.time_arm(rig.arms[name], rep, k, args.replays)
        if args.l1:
            l1_arms = tuple(args.l1_arms.split(","))
            if any(a not in rig.arms for a in l1_arms):
                raise RuntimeError(f"L1 arms {l1_arms} must be built arms {list(rig.arms)}")
            for mode in args.l1_modes.split(","):
                rig.l1_sets(l1_arms, args.l1, args.l1_seed0, fails, shared=(mode == "shared"))
    except Exception as e:  # noqa: BLE001
        fails.append(f"g{g}: {type(e).__name__}: {e}")
        print(f"CASE_ERROR g{g}: {type(e).__name__}: {e}", flush=True)
        traceback.print_exc()
    with open(os.path.join(args.out, f"b{B}_{tag}_g{g}.failures.json"), "w") as f:
        json.dump(fails, f, indent=1)
    n = torch.tensor([len(fails)], device=device)
    try:
        dist.all_reduce(n)
    except Exception:  # noqa: BLE001
        pass
    if g == 0:
        print(f"TP_RESULT failures={int(n)} ({time.time() - t0:.0f}s)", flush=True)
    dist.destroy_process_group()
    sys.exit(1 if int(n) else 0)


if __name__ == "__main__":
    main()
