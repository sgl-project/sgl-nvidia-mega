# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""The cost of the compliant mailbox re-arm fence on the MoE hops H2 (RouteSitu poll),
H3/HS (MoeTail poll) and H4 (MoeTail publish), TP16 B8 (Q8 verify), through ``moe.prepared.PreparedMoE`` over W = 4 GPUs
(ranks 4-15 injected). A copy of ``k31_tail_opts_b8.py`` whose arms differ only
in ``common.mailbox.REARM_FENCE`` (compiled per arm, caches cleared): A, A2 "gpu" (shipped mc_rf); FN "0" (no fence: an
unsafe cost control, never served); FS "sys"; FA "sys_alias" (fence.proxy.alias + fence.acq_rel.sys, the form a
Dat3M litmus test forbids the lost publish with). Experts grid = all SMs, MoeTail front, as production TP16 B8.
The text below is the source rig's; its option arms are replaced by the fence arms (engaged = PTX fence counts per arm).

Arms (each with its own mailboxes; env flags are read at construction):
  A, A2  today (A2: identical control);  Z  K31MK_MOE_TAIL_POLL_NS=0;  R  K31MK_MOE_TAIL_RELOAD=1-64:1;
  Gt  K31MK_MOE_TAIL_GATE=3-8:1;  ZR  Z + R;  ZRG  Z + R + Gt;  ZG  Z + Gt;
  G, GZRG  (info) experts grid SMs - 16 (the adopted arm G) without / with Z + R + Gt; bitwise vs G;
  <name>2  a second instance of each config (Z2, R2, ..., GZRG2).

Checks (tolerances stated before any run):
  ids / weights / latent / hg / experts / partial / multicast / RS / output / rank_bitwise / graph: as the source rig;
  shared  (re-registered) S1: h_mk within 1 BF16 ulp of BF16(SiTU_FP64(hg)); S2: S vs FP64 shared_down @ h_mk;
  path    ExpertsKernel(SMs or SMs - 16) + MoeTail(448), no push / WaveMoeTail;
  engaged the arm's compiled tail IS compile_tail(...) for the arm's intended options (cache identity);
  bitwise every arm's outputs, latent partial, S, ids and expert rows equal its reference arm's (A or G) bit for bit.
Mutations (each must fail): reload_assign, flag_inert.

Timing (--time R): R replicates, arm order alternating forward / reverse. Per run: 20 warm replays; M2 = CUDA-event
time of 100 back-to-back graph replays / 600 layers; from a torch-profiler trace of 20 replays, per layer T = t_end(MoeTail)
- t_end(preceding ExpertsKernel) and M1 = t_end(MoeTail) - t_end(RouteSitu grid 16) (medians on this GPU).

Usage: python3 -m torch.distributed.run --nproc-per-node 4 k31_tail_opts_b8.py --out DIR
       [--arms A,A2,Z,R,Gt,ZR,ZRG,G,GZRG] [--time 6]
"""

import argparse
import json
import os
import statistics
import sys
import tempfile
import time
import traceback
from types import SimpleNamespace

# PTX text on every compiled kernel (the fence path check), set before cutlass is imported.
_out = sys.argv[sys.argv.index("--out") + 1] if "--out" in sys.argv else "."
os.environ["CUTE_DSL_KEEP"] = "ptx"
os.environ["CUTE_DSL_DUMP_DIR"] = os.path.join(_out, f"ptx_g{os.environ.get('LOCAL_RANK', '0')}")
os.makedirs(os.environ["CUTE_DSL_DUMP_DIR"], exist_ok=True)  # the DSL silently skips the dump into a missing directory

import torch
import torch.distributed as dist

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _harness as H  # noqa: E402
import _ref_moe as R  # noqa: E402
import _ref_moe16 as M  # noqa: E402
from k31_moe_tp16 import (  # noqa: E402
    EPS, HID, LAT, LOGITS, LROWS, SENT, SHARD, TP, Checker, Model, bf, canon_bf16, gen, randn, route_ref, ulp_bf16)

B = 8
NLAYERS = 6  # multiple of 3: see k31_wavetail_tp.py
LAYER_IDS = tuple(21 + i for i in range(NLAYERS))
TARGET_ED = 79.0

TL = "kimi_k31/k3/cutedsl/moe/tail.py"
TR_ = "kimi_k31/k3/cutedsl/moe/tail_runtime.py"
MB_ = "kimi_k31/k3/cutedsl/common/mailbox.py"
MUTATIONS = {
    # The sys_alias mode drops its alias fence (only the PTX path check can see it: fences change no value).
    "fa_noalias": (MB_, "        _fence_proxy_alias()\n", "        pass\n"),
    # The sys modes emit the gpu fence instead.
    "fs_gpu": (MB_, "        _fence_acq_rel_sys()\n", "        cute.arch.fence_acq_rel_gpu()\n"),
}

FLAGS = ("K31MK_MOE_TAIL_POLL_NS", "K31MK_MOE_TAIL_GATE", "K31MK_MOE_TAIL_RELOAD")
ARMS = {  # name: (env, intended tail options, bitwise reference arm, experts SMs left free, re-arm fence mode)
    "A": ({}, (None, False, False), "A", 0, "gpu"),
    "A2": ({}, (None, False, False), "A", 0, "gpu"),
    "FN": ({}, (None, False, False), "A", 0, ""),
    "FS": ({}, (None, False, False), "A", 0, "sys"),
    "FA": ({}, (None, False, False), "A", 0, "sys_alias"),
}
# Expected (gpu, sys, alias) fence counts in the PTX per mode: MoeTail front (TP16 B8) and RouteSitu(16) route + copy.
TAIL_FENCES = {"gpu": (1, 0, 0), "": (0, 0, 0), "sys": (0, 1, 0), "sys_alias": (0, 1, 1)}
ROUTE_FENCES = {"gpu": (3, 0, 0), "": (0, 0, 0), "sys": (0, 3, 0), "sys_alias": (0, 3, 3)}


def ptx_text(code):
    """PTX text of a compiled kernel: the DSL returns the text, or the dumped file's path when it keeps it on disk.
    ``common.host.PreparedLaunchAttributes`` (the tail after its carve-out fix) wraps the compiled function."""
    code = getattr(code, "compiled", code)
    text = code.__ptx__
    if isinstance(text, str) and ".version" not in text and os.path.isfile(text):
        with open(text) as f:
            text = f.read()
    if not isinstance(text, str) or ".version" not in text:
        raise RuntimeError(f"no PTX text on the compiled kernel: {str(text)[:120]!r}")
    return text


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
    """x of layer i = BF16(base + shift_i + s roll(noise, i)) for a spread s (common to every GPU)."""
    base = randn((HID,), 1.0, 777, device).double()
    noise = randn((B, HID), 1.0, 778, device).double()
    shifts = [randn((HID,), 0.3, 900 + layer, device).double() for layer in LAYER_IDS]
    return lambda i, s: canon_bf16((base + shifts[i] + s * torch.roll(noise, i, 0)).to(torch.bfloat16)).contiguous()


def choose_spread(models, make_x):
    """The spread whose FP64 routing gives a median distinct-expert count closest to TARGET_ED."""
    best = None
    for s in (0.1, 0.2, 0.3, 0.5, 0.7, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0, 2.5, 3.0, 4.0):
        eds = []
        for i, m in enumerate(models):
            x, ids = make_x(i, s), set()
            for t in range(B):
                ids.update(route_ref(m, x[t])[1].tolist())
            eds.append(len(ids))
        med = statistics.median(eds)
        if best is None or abs(med - TARGET_ED) < abs(best[1] - TARGET_ED):
            best = (s, med, eds)
    return best


class Arm:
    def __init__(self, name, rig):
        env, _, _, free, fence = ARMS[name]
        env = {**env, "K31MK_EXPERTS_GRID_SMS": str(rig.sms - free)} if free else env
        from sgl_nvidia_mega.kimi_k31.k3.cutedsl.common import mailbox as MB

        def set_fence(mode):  # the mode is not a cache key: clear the compile caches when it changes
            if MB.REARM_FENCE != mode:
                MB.REARM_FENCE = mode
                for fn in (rig.mods.TR.compile_tail, rig.mods.FR.compile_route, rig.mods.FR.compile_publish):
                    fn.cache_clear()
        mods = rig.mods
        self.name, self.rig = name, rig
        g, d, group = rig.g, rig.device, rig.group
        keys = FLAGS + ("K31MK_EXPERTS_GRID_SMS",)
        saved = {k: os.environ.get(k) for k in keys}
        for k in keys:
            os.environ.pop(k, None)
        os.environ.update(env)
        set_fence(fence)
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
                                            R.LIN, plan=self.state.plan, counters=self.state.shared, q=B,
                                            deadline_ns=rig.deadline_ns)
                pm = mods.PM.PreparedMoE(front, experts, tail, self.state, prefetch=None, layer_idx=lay.layer)
                self.layers.append(SimpleNamespace(pm=pm, tail=tail, experts=experts, parity=i % 3))
            self.grid = self.layers[0].experts.grid_for(B)
            self.published = B in self.layers[0].tail.pushers
        finally:
            set_fence("gpu")
            for k, v in saved.items():
                if v is None:
                    os.environ.pop(k, None)
                else:
                    os.environ[k] = v
        self.saves = [torch.full((B, rig.world * SHARD // 2), SENT, dtype=torch.int32, device=d)
                      for _ in range(NLAYERS)]
        H.record({"kind": "arm", "arm": name, "env": env, "grid": self.grid, "published": self.published, "g": g})

    def inject(self, i):
        lay, w = self.rig.layers[i], self.rig.world
        row = self.front_box.buffer[0, :B].view(torch.float32)  # [8, 4480]: latent 3584 | logits 896
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
        make_x = layer_inputs(device)
        self.spread, med, eds = choose_spread(models, make_x)
        H.record({"kind": "routing", "spread": self.spread, "ed_median": med, "ed_layers": eds, "g": g})
        print(f"ROUTING spread {self.spread} E_d median {med} layers {eds}", flush=True)
        self.layers, self.x = [], []
        for i, (layer, m) in enumerate(zip(LAYER_IDS, models)):
            x = make_x(i, self.spread)
            self.x.append(x)
            logical = M.make_logical16(R.E_ALL, gen(1600 + 37 * layer + g, device))
            router = m.router[g * LOGITS:(g + 1) * LOGITS]
            down = m.down[g * LROWS:(g + 1) * LROWS]
            packed = torch.cat((router, router.new_zeros((-LOGITS % 16, HID)), down)).contiguous()
            lat_ref = x.double() @ m.down.double().T
            logits_ref = x.double() @ m.router.double().T
            inj_s = torch.stack([randn((B, HID), 0.5, 3000 + 100 * layer + q, device) for q in range(world, TP)])
            lay = SimpleNamespace(
                layer=layer, m=m, logical=logical, host_w=M.host_layout16(*logical), packed=packed,
                lat_ref=lat_ref, inj_s=inj_s,
                inj_lat=M.canon(lat_ref.float()[:, world * LROWS:]),
                inj_logits=M.canon(logits_ref.float()[:, world * LOGITS:]),
                inj_partials=torch.stack([canon_bf16(randn((B, LAT), 0.5, 2000 + 100 * layer + q, device))
                                          for q in range(world, TP)]),
                inj_rs=canon_bf16(inj_s[:, :, g * SHARD:(g + 1) * SHARD]).contiguous())
            self.layers.append(lay)
        self.consumer = build_consumer(device.index, B, world * SHARD // 2, HID // 2, deadline_ns)
        self.arms = {}

    def gather(self, t):
        out = [torch.empty_like(t) for _ in range(self.world)]
        dist.all_gather(out, t.contiguous(), group=self.group)
        return out

    def sync(self):
        torch.cuda.synchronize()
        dist.barrier(group=self.group)

    # ------------------------------------------------------------------ correctness
    def check_layer(self, arm, i, fails, ref_rows):
        g, w, lay = self.g, self.world, self.layers[i]
        m, t, st, tb = lay.m, i % 3, arm.state, arm.tail_box
        ids = st.front[2][:B].clone()
        wts = st.front[3][:B].clone()
        latent = st.front[6][:B].clone()
        hg = st.front[1][:B].clone()
        h_all = st.front[4][:B].clone()
        S = st.front[5][:B].clone()
        lat_gpus = self.gather(latent)
        slots = tb.latent.buffer[t, :TP, :B].clone()  # [16, 8, 3584]
        allslots = self.gather(slots[:w].contiguous())
        S_all = self.gather(S)
        saves_all = self.gather(arm.saves[i])
        rows = st.expanded[: B * 16].clone().view(B, 16, LAT)
        x = self.x[i]
        G_ref = M.expert_rows(lay.logical, latent, ids)  # [8, 16, 3584] FP64 on the MK's own latent and ids
        ref_rows[(arm.name, i)] = (ids, latent, slots[g].clone(), S, rows)
        for tok in range(B):
            ck = Checker(f"b8.{arm.name}.l{lay.layer}.t{tok}.g{g}", fails)
            ck.route(m, x[tok], ids[tok], wts[tok])
            ck.close("latent", latent[tok], lay.lat_ref[tok], slice(0, w * LROWS))
            if not torch.equal(latent[tok, w * LROWS:].view(torch.int32), lay.inj_lat[tok].view(torch.int32)):
                ck.fail("latent_inj", "injected latent rows changed")
            if not all(torch.equal(o[tok].view(torch.int32), latent[tok].view(torch.int32)) for o in lat_gpus):
                ck.fail("latent_multicast", "latent words differ across GPUs")
            ck.close("hg", hg[tok], m.gate_up.double() @ x[tok].double())
            # S1: the MK's BF16 hidden within 1 BF16 ulp of BF16(SiTU_FP64) of its own gate/up (stock SiTU is approx).
            h_mk = h_all[tok].double()
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
                ck.fail("shared_h", f"{nh} hidden cells beyond 1 ulp of BF16(SiTU_FP64)")
            if ns or not torch.isfinite(S[tok].float()).all():
                ck.fail("shared", f"{ns} cells vs down @ h_mk")
            wd = wts[tok].double().view(16, 1)
            part_ref = (wd * G_ref[tok]).sum(0)
            mine = slots[g, tok].double()
            fro = float((mine - part_ref).norm() / part_ref.norm())
            H.record({"kind": "check", "item": f"{ck.name}.partial_t1", "rel_l2": fro})
            if not (torch.isfinite(mine).all() and fro <= 3e-2):
                ck.fail("partial_t1", f"row rel-L2 {fro:.3e} > 3e-2")
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
            for q in range(w):
                want = canon_bf16(S_all[q][tok, g * SHARD:(g + 1) * SHARD])
                if not torch.equal(tb.rs.buffer[t, q, tok].view(torch.int16), want.view(torch.int16)):
                    ck.fail("rs", f"shard from rank {q} differs")
            for o in range(w, TP):
                got = arm.scratch[o - w][t, g, tok]
                want = canon_bf16(S[tok, o * SHARD:(o + 1) * SHARD])
                if not torch.equal(got.view(torch.int16), want.view(torch.int16)):
                    ck.fail("rs_out", f"shard for owner {o} differs")
            Lsum = slots[:, tok].double().sum(0)
            inv = 1.0 / torch.sqrt((Lsum ** 2).mean() + EPS)
            xn = bf(bf(Lsum) * inv * m.gamma.double())
            y = bf(m.up.double() @ xn)
            S_full = [S_all[q][tok].double() for q in range(w)] + [lay.inj_s[q - w][tok].double() for q in range(w, TP)]
            sh = bf(sum(S_full))
            out_ref = bf(y + sh)
            n = w * SHARD
            out = arm.saves[i][tok].view(torch.bfloat16).double()
            lim = 2 * ulp_bf16(torch.maximum(torch.maximum(out_ref[:n].abs(), y[:n].abs()), sh[:n].abs()))
            err = (out - out_ref[:n]).abs()
            nbad = int((err > lim).sum())
            H.record({"kind": "check", "item": f"{ck.name}.output", "max_abs": float(err.max()),
                      "max_ulps": float((err / ulp_bf16(out_ref[:n].abs())).max()), "violations": nbad})
            if nbad or not torch.isfinite(out).all():
                ck.fail("output", f"{nbad} cells beyond 2 ulp; max {float(err.max()):.3e}")
            if not all(torch.equal(o[tok], arm.saves[i][tok]) for o in saves_all):
                ck.fail("rank_bitwise", "output rows differ across GPUs")
        part_ref = (wts.double().view(B, 16, 1) * G_ref).sum(1)
        fro = R.rel_l2(slots[g], part_ref)
        H.record({"kind": "check", "item": f"b8.{arm.name}.l{lay.layer}.g{g}.partial_fro", "rel_l2": fro})
        if not fro <= 1e-2:
            fails.append(f"b8.{arm.name}.l{lay.layer}.g{g}.partial_fro: {fro:.3e} > 1e-2")
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
                H.record({"kind": "check", "item": f"b8.{arm.name}.graph{rep}.l{i}.g{self.g}", "bitwise_vs_eager": same})
                if not same:
                    fails.append(f"b8.{arm.name}.graph{rep}.l{i}.g{self.g}: differs from eager")
                    print("FAIL graph", arm.name, rep, i, flush=True)
        arm.graph, arm.eager = graph, eager

    def engaged(self, arm, fails):
        """The arm's compiled tail and routing kernels carry the fences of its INTENDED mode (ARMS), read from the PTX."""
        mode = ARMS[arm.name][4]
        tail_code = arm.layers[0].tail.tail_weights.codes[B][0]
        route_code = arm.layers[0].pm.front.plans[B][2]
        count = lambda t: tuple(t.count(f) for f in ("fence.acq_rel.gpu", "fence.acq_rel.sys", "fence.proxy.alias"))
        got = (count(ptx_text(tail_code)), count(ptx_text(route_code)))
        want = (TAIL_FENCES[mode], ROUTE_FENCES[mode])
        bad = [] if got == want else [f"fence counts (tail, route) {got} != {want} for mode {mode!r}"]
        H.record({"kind": "check", "item": f"b8.{arm.name}.engaged.g{self.g}", "mode": mode, "got": got, "bad": bad})
        if bad:
            fails.append(f"b8.{arm.name}.engaged.g{self.g}: {bad}")
            print("FAIL engaged", arm.name, bad, flush=True)

    # ------------------------------------------------------------------ profiling / timing
    def profile(self, arm, replays):
        from torch.profiler import ProfilerActivity, profile

        self.sync()
        with profile(activities=[ProfilerActivity.CUDA]) as prof:
            for _ in range(replays):
                arm.graph.replay()
            torch.cuda.synchronize()
        path = os.path.join(tempfile.mkdtemp(prefix="k31x_tail_"), "trace.json")
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
            elif "ExpertsKernel" in name:
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
        want_e, want_t = f"experts{self.sms - ARMS[arm.name][3]}", "moetail448"
        bad = []
        if want_e not in kinds or any(k.startswith("experts") and k != want_e for k in kinds):
            bad.append(f"experts kernels {sorted(k for k in kinds if k.startswith('experts'))} != {want_e}")
        if want_t not in kinds or any(k.startswith(("wave", "moetail")) and k != want_t for k in kinds):
            bad.append(f"tail kernels {sorted(k for k in kinds if k.startswith(('wave', 'moetail')))} != {want_t}")
        if "push" in kinds:
            bad.append("push kernel present")
        H.record({"kind": "check", "item": f"b8.{arm.name}.path.g{self.g}", "kinds": kinds, "bad": bad})
        if bad:
            fails.append(f"b8.{arm.name}.path.g{self.g}: {bad}")
            print("FAIL path", arm.name, bad, flush=True)

    def time_arm(self, arm, rep, order):
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
        seg_t, seg_m1, kinds = self.profile(arm, 20)
        med = lambda s: statistics.median(s) if s else float("nan")
        pct = lambda s, p: sorted(s)[int(p * len(s))] if s else None
        rec = {"kind": "time", "arm": arm.name, "rep": rep, "order": order, "g": self.g,
               "t_us": med(seg_t), "t_n": len(seg_t), "t_p10": pct(seg_t, 0.1), "t_p90": pct(seg_t, 0.9),
               "m1_us": med(seg_m1), "m1_n": len(seg_m1), "m2_us": m2, "m2_max_us": max(float(v) for v in m2_all),
               "kernels": kinds}
        H.record(rec)
        print("TIME", json.dumps(rec), flush=True)
        return rec


def cross_arm(rig, ref_rows, fails):
    """Gate term: every arm's outputs, latent partial, S, ids and expert rows equal its reference arm's bit for bit."""
    for name, arm in rig.arms.items():
        ref = ARMS[name][2]
        if ref not in rig.arms:
            fails.append(f"b8.{name}.g{rig.g}.bitwise: reference arm {ref} not in the run")
            continue
        a = rig.arms[ref]
        for i in range(NLAYERS):
            ra, rx = ref_rows[(ref, i)], ref_rows[(name, i)]
            same = {"output": torch.equal(a.eager[i], arm.eager[i]),
                    "ids": torch.equal(ra[0], rx[0]),
                    "partial": torch.equal(ra[2].view(torch.int16), rx[2].view(torch.int16)),
                    "shared": torch.equal(ra[3].view(torch.int16), rx[3].view(torch.int16)),
                    "experts": torch.equal(ra[4].view(torch.int16), rx[4].view(torch.int16))}
            av, bv = a.eager[i].view(torch.bfloat16).double(), arm.eager[i].view(torch.bfloat16).double()
            ulps = float(((av - bv).abs() / ulp_bf16(av.abs())).max())
            H.record({"kind": "cross", "arm": name, "ref": ref, "layer": i, "g": rig.g, "output_max_ulp": ulps, **same})
            bad = [k for k, v in same.items() if not v]
            if bad:
                fails.append(f"b8.{name}.l{i}.g{rig.g}.bitwise_vs_{ref}: {bad} (output max {ulps:.2f} ulp)")
                print(f"FAIL bitwise_vs_{ref}", name, i, bad, ulps, flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--deadline-s", type=float, default=2.0)
    ap.add_argument("--arms", default="A,A2,FN,FS,FA")
    ap.add_argument("--time", type=int, default=0, help="timing replicates (0: correctness only)")
    ap.add_argument("--time-always", action="store_true", help="time even when a check failed (screening only)")
    ap.add_argument("--no-profile", action="store_true", help="skip the profiler path check (compute-sanitizer runs)")
    args = ap.parse_args()
    arms = tuple(args.arms.split(","))
    if arms[0] != "A":
        raise SystemExit("arm A (today) must be first: every arm is compared with it")
    local = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local)
    device = torch.device("cuda", local)
    dist.init_process_group("nccl", device_id=device)
    g, world = dist.get_rank(), dist.get_world_size()
    tag = os.environ.get("K31_MUTATION") or "base"
    os.makedirs(args.out, exist_ok=True)
    os.environ["K31_RESULTS"] = os.path.join(args.out, f"b8_{tag}_g{g}.jsonl")
    H.load_plugin(MUTATIONS)
    from sgl_nvidia_mega.kimi_k31.k3.cutedsl.common import host
    from sgl_nvidia_mega.kimi_k31.k3.cutedsl.moe import experts_runtime as ER
    from sgl_nvidia_mega.kimi_k31.k3.cutedsl.moe import front as FR
    from sgl_nvidia_mega.kimi_k31.k3.cutedsl.moe import planned as PL
    from sgl_nvidia_mega.kimi_k31.k3.cutedsl.moe import prepared as PM
    from sgl_nvidia_mega.kimi_k31.k3.cutedsl.moe import tail_runtime as TR
    from sgl_nvidia_mega.kimi_k31.k3.cutedsl.oproj import prepared as OP

    mods = SimpleNamespace(host=host, ER=ER, FR=FR, PL=PL, PM=PM, TR=TR, OP=OP)
    H.record({"kind": "env", **H.device_facts(), "world": world, "arms": list(arms), "mutation": tag})
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
                    rig.time_arm(rig.arms[name], rep, k)
    except Exception as e:  # noqa: BLE001
        fails.append(f"g{g}: {type(e).__name__}: {e}")
        print(f"CASE_ERROR g{g}: {type(e).__name__}: {e}", flush=True)
        traceback.print_exc()
    with open(os.path.join(args.out, f"b8_{tag}_g{g}.failures.json"), "w") as f:
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
