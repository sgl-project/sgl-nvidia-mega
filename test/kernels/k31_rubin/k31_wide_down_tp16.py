# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Wide decode tier M1(a): the shared expert's down rows at 65-128 rows after the folded front, into the MoE
tail's reduce-scatter (RS) mailbox of a 128-row tail box, over W = 4 GPUs with real symm_mem (GPU g = TP rank g of 16;
ranks 4-15's front words are injected; this GPU's RS shards for owners 4-15 land in local scratch). Seeded weights as
k31_moe_tp16.py (``Model``).

Form under test (T): K31MK_MOE_WIDE=65-128:1 (whole process), K31MK_MOE_FRONT_TMEM=65-128:1 + K31MK_MOE_WIDE_FOLD=65-128:1
(set around ``SharedFront`` construction): ``route`` runs the MoeFrontTmem64 grid (front rows + shared gate/up + SiTU,
h left in the scratch) and the routing kernel (plan blocks into the wide engine's plan), then ``shared_expert`` returns
the ``SharedDown`` launch (on ``WIDE_DOWN`` SMs, 4) that ``PreparedMoE`` runs beside the wide experts.

Per (layer 11 / 12, call 0 / 1 / 2, batch 72 / 96 / 128; call 2 repeats call 0's inputs, tail parity = call % 3): arm every
RS word (this GPU's RS buffer and the 12 scratch slots) with 0x5A5A, inject ranks 4-15's front words, ``route`` then
``shared_expert`` + the returned launch on a side stream (the order ``PreparedMoE.run`` uses for merged batches), sync,
barrier. Checks (tolerances written before any run, MOE-wideM1b.2):
  path     ``shared_expert`` returns a launch (FTM batch: no gate kernel) and the batch is in ``front.front_tmem`` and in
           ``front.split`` with 4 SMs (``WIDE_DOWN``; K31MK_MOE_SPLIT_SMS unset).
  h        this GPU's BF16 h vs FP64 bf16(SiTU(bf16(g), bf16(u))), g / u = x @ gate_up.T halves: |d| <= 2^-6 |ref| +
           2^-8 rms(ref) (k31_moe_front_tmem_tp16's bound).
  rs       every real sender q in 0-3: RS slot (parity, q, token) on this GPU vs FP64 W_q[g*448:(g+1)*448] @ h_q[token]
           (W_q = sender q's shared down shard, h_q = sender q's own BF16 h, gathered): |d| <= 2^-8 |ref| +
           2 * 384 * 2^-24 * (|W_q| @ |h_q|) (one BF16 output rounding + FP32 accumulation over K = 384: the down GEMM
           alone, k31_moe_tp16's 'k' bound); no -0 word (the push writes +0).
  rs_out   this GPU's shards for owners 4-15 (scratch slot (parity, g, token)) vs FP64 W_g[o*448:(o+1)*448] @ h_g: same bound.
  canary   every RS / scratch word outside the written region (parity, senders 0-3 (scratch: g), tokens 0..b-1) still
           holds 0x5A5A: rows b..127, the other two parities, senders 4-15 (a 64-row stride or a token wrap writes there).
  determ   call 2 (parity 2) received bytes == call 0 (parity 0) received bytes, bit for bit (scratch too).
  graph    route + shared_expert + the launch captured in one CUDA graph per batch (layer 11, parity 0), replayed on call
           0's inputs (arming + injection outside): received RS bytes and scratch == eager call 0, bit for bit.
Mutations (each must fail at >= 72 rows with rs / canary lines, not by a crash): down_cap64 (the RS row stride passed to
the down launch is 64, not the tail box's 128), down_row_wrap (the peer store puts token t at row t % 64).

Timing (--time, only if every check passed; Tier 0, ranks a rig, not a gate): arms S (served wide form: no FTM / fold:
Fp32FrontGemm publish + routing kernel + WidePlanKernel on the main stream, the split-K shared chain gate/up -> SiTU ->
down on the side stream beside the front) and T (above: FTM + routing on main, SharedDown on the side stream after the
route), one CUDA graph per (arm, batch) holding layers 11 and 12, each layer: injection, a 2 us spin kernel (the o-proj
stand-in; it does NOT trigger dependents, so FTM's pre-wait weight staging cannot overlap it: biased against T), the
front, the side stream joined before the next layer. Schedule S T T S x reps on the same GPUs; per replay and layer, CUPTI
END times relative to the spin END: F = front ready (S: WidePlanKernel END, T: routing kernel END), fe = front GEMM END
(Fp32FrontGemm / MoeFrontTmem64), sd = shared down END (S: last SplitKGemm of the side chain, T: SharedDown).
Reported per (arm, batch): median over replays x layers per GPU, then the median of the 4 GPUs; paired d = T - S per
ABBA block. Prediction (written before data,  + this rig's non-PDL producer): F_T at 128 rows
10 [6, 15] us, F_S 24 [17, 32] us, d(F) at 128 -14 [-8, -22] us; at 72 rows d(F) -9 [-4, -15] us.

Usage (one node, 4 GPUs): python3 -m torch.distributed.run --nproc-per-node 4 k31_wide_down_tp16.py --out DIR [--quick]
[--time] [--reps 20]   (--quick: layer 11, call 0, no graph / determ / time; for mutation runs)
Exit status 0 only when every check on every rank passed (``TP_RESULT ... failures=0``).
"""

import argparse
import gzip
import json
import os
import statistics
import sys
import time
import traceback
from types import SimpleNamespace as NS

import torch
import torch.distributed as dist

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _harness as H  # noqa: E402
import _ref_moe as R  # noqa: E402
import _ref_moe16 as M  # noqa: E402
from k31_moe_tp16 import Checker, Model, canon_bf16, randn  # noqa: E402

TP, HID, LAT = 16, 7168, 3584
SHARD, LROWS, LOGITS = HID // TP, LAT // TP, 896 // TP
INTER = 384  # shared expert intermediate per rank
BATCHES = (72, 96, 128)
CAP = 128
ROWS = "65-128:1"
FILL16 = 0x5A5A
WIDE_DOWN_SMS = 4

FR_PATH = "kimi_k31/k3/cutedsl/moe/front.py"
SH_PATH = "kimi_k31/k3/cutedsl/moe/shared.py"
MUTATIONS = {
    "down_cap64": (FR_PATH, "mailbox = (tail.peers, *tail.ring, Int32(tail.tail_box.cap))  # RS row stride",
                   "mailbox = (tail.peers, *tail.ring, Int32(64))  # RS row stride"),
    "down_row_wrap": (SH_PATH, "row = (parity * self.tp + rank) * capacity + t\n",
                      "row = (parity * self.tp + rank) * capacity + t % 64\n"),
}
ARM_ENV = {"S": {}, "T": {"K31MK_MOE_FRONT_TMEM": ROWS, "K31MK_MOE_WIDE_FOLD": ROWS}}


class Rig:
    def __init__(self, P, group, g, world, device, deadline_ns, layers, arms):
        host, OP, FR, PM, TRB, ER = P
        from cutlass import Int32

        self.I32, self.host, self.FR, self.ER = Int32, host, FR, ER
        self.g, self.world, self.device, self.group = g, world, device, group
        self.front_box = OP.OprojMailbox(group, device, TP, g, (1, CAP, 2 * LAT + 2 * 896))
        self.state = PM.MoEScratch(TP, device, BATCHES)
        self.side = torch.cuda.Stream(device=device)
        self.tb = TRB(group, device, TP, g, cap=CAP)
        self.scratch = [torch.empty_like(self.tb.rs.buffer) for _ in range(TP - world)]
        ptrs = list(self.tb.rs.handle.buffer_ptrs) + [s.data_ptr() for s in self.scratch]
        self.peers = torch.tensor(ptrs, dtype=torch.int64, device=device)
        self.splan = torch.empty_like(self.state.experts["rubin_wide"]["plan"])  # S arm's WidePlanKernel target
        self.layers = []
        for layer in layers:
            m = Model(g, device, layer)
            router = m.router[g * LOGITS:(g + 1) * LOGITS]
            down = m.down[g * LROWS:(g + 1) * LROWS]
            packed = torch.cat((router, router.new_zeros((-LOGITS % 16, HID)), down)).contiguous()
            fronts = {}
            for arm in arms:
                saved = {k: os.environ.pop(k, None) for k in ("K31MK_MOE_FRONT_TMEM", "K31MK_MOE_WIDE_FOLD")}
                os.environ.update(ARM_ENV[arm])
                wide = {"wide_plan": self.state.experts["rubin_wide"]["plan"]} if arm == "T" else {}
                try:
                    fronts[arm] = FR.SharedFront(m.moe(), packed, self.front_box, self.state.front, BATCHES, R.BETA,
                                                 R.LIN, plan=self.state.plan, counters=self.state.shared, q=8,
                                                 deadline_ns=deadline_ns, **wide)
                finally:
                    for k, v in saved.items():
                        os.environ.pop(k, None)
                        if v is not None:
                            os.environ[k] = v
            # Sender q's shared down shard (Model seeds it per rank: 1000 * layer + 200 + q).
            downs = [randn((HID, INTER), INTER ** -0.5, 1000 * layer + 200 + q, device) for q in range(world)]
            self.layers.append(NS(m=m, fronts=fronts, layer=layer, downs=downs))
        self.wide_plan_code = {b: ER.compile_wide_plan(b, device.index) for b in BATCHES} if "S" in arms else {}
        self.saved = {}

    # ------------------------------------------------------------------ helpers
    def tail(self, parity):
        return NS(peers=self.host.to_cute(self.peers, 16), ring=(self.I32(parity), self.I32(self.g)),
                  tail_box=NS(cap=self.tb.cap), push=lambda rows: None)

    def inputs(self, lay, call, b):
        seed_call = 0 if call == 2 else call
        x = randn((b, HID), 1.0, 9000 + 100 * seed_call + 10 * lay.layer + b, self.device)
        m = lay.m
        return x, x.double() @ m.router.double().T, x.double() @ m.down.double().T

    def inject(self, b, logits_ref, lat_ref):
        w = self.world
        row = self.front_box.buffer[0, :b].view(torch.float32)  # [b, 4480]: latent 3584 | logits 896
        row[:, w * LROWS: LAT].copy_(M.canon(lat_ref.float()[:, w * LROWS:]))
        row[:, LAT + w * LOGITS:].copy_(M.canon(logits_ref.float()[:, w * LOGITS:]))

    def arm_rs(self):
        self.tb.rs.buffer.view(torch.int16).fill_(FILL16)
        for s in self.scratch:
            s.view(torch.int16).fill_(FILL16)

    def sync(self):
        torch.cuda.synchronize()
        dist.barrier(group=self.group)

    def run_t(self, front, xc, tail):
        """``PreparedMoE.run``'s merged order: route on main; the down launch on the side stream after it."""
        main = torch.cuda.current_stream(self.device)
        front.route(xc)
        st = self.state
        st.begin.record(main)
        with torch.cuda.stream(self.side):
            self.side.wait_event(st.begin)
            _, beside = front.shared_expert(xc, tail)
            launched = beside is not None
            if launched:
                beside()
            st.shared_done.record(self.side)
        main.wait_event(st.shared_done)
        return launched

    def run_s(self, front, xc, tail, b):
        """Served wide form: the split-K shared chain on the side stream beside the front; WidePlanKernel after route."""
        main = torch.cuda.current_stream(self.device)
        st = self.state
        st.begin.record(main)
        with torch.cuda.stream(self.side):
            self.side.wait_event(st.begin)
            front.shared_expert(xc, tail)
            st.shared_done.record(self.side)
        _, _, latent = front.route(xc)
        plan = self.splan[: -(-b // 64) * self.FR.PLAN_BYTES]  # exactly the blocks WidePlanKernel(b) was compiled for
        self.wide_plan_code[b](self.host.to_cute(latent, 16), self.host.to_cute(plan), self.host.stream(self.device))
        main.wait_event(st.shared_done)

    # ------------------------------------------------------------------ checks
    def down_bound(self, W, h):
        """FP64 reference W @ h and the bound 2^-8 |ref| + 2 * 384 * 2^-24 (|W| @ |h|) (rows of W, h [n, 384])."""
        ref = h.double() @ W.double().T
        acc = h.double().abs() @ W.double().abs().T
        return ref, 2 ** -8 * ref.abs() + 2 * INTER * 2 ** -24 * acc

    def check_call(self, lay, call, b, fails):
        front = lay.fronts["T"]
        x, logits_ref, lat_ref = self.inputs(lay, call, b)
        xc = canon_bf16(x)
        parity = call % 3
        ck = Checker(f"wdown.b{b}.l{lay.layer}.call{call}.g{self.g}", fails)
        if b not in front.front_tmem or front.split.get(b) != WIDE_DOWN_SMS:
            ck.fail("path", f"front_tmem {sorted(front.front_tmem)} split {front.split.get(b)}")
            return
        self.state.front[4][:b].fill_(float("nan"))
        self.arm_rs()
        self.inject(b, logits_ref, lat_ref)
        self.sync()
        launched = self.run_t(front, xc, self.tail(parity))
        self.sync()
        if not launched:
            ck.fail("path", "shared_expert returned no down launch")
            return
        w, g = self.world, self.g
        h = self.state.front[4][:b].clone()
        # h vs FP64 (own rank)
        hg = x.double() @ lay.m.gate_up.double().T
        g64, u64 = hg[:, :INTER].to(torch.bfloat16).double(), hg[:, INTER:].to(torch.bfloat16).double()
        h_ref = R.situ(g64, u64).to(torch.bfloat16).double()
        tol = 2 ** -6 * h_ref.abs() + 2 ** -8 * float(h_ref.pow(2).mean().sqrt())
        bad = int((~((h.double() - h_ref).abs() <= tol)).sum())
        H.record({"kind": "check", "item": f"{ck.name}.h", "violations": bad})
        if bad:
            ck.fail("h", f"{bad} cells")
        hs = [torch.empty_like(h) for _ in range(w)]
        dist.all_gather(hs, h.contiguous(), group=self.group)
        rs = self.tb.rs.buffer  # [3, 16, CAP, 448]
        worst = {"rs": 0.0, "rs_out": 0.0}
        # received shards from senders 0-3
        for q in range(w):
            ref, lim = self.down_bound(lay.downs[q][g * SHARD:(g + 1) * SHARD], hs[q])
            got = rs[parity, q, :b].double()
            err = (got - ref).abs()
            nbad = int((~(err <= lim)).sum())
            neg0 = int((rs[parity, q, :b].view(torch.int16) == -0x8000).sum())
            worst["rs"] = max(worst["rs"], float((err / lim.clamp(min=1e-30)).nan_to_num(1e30).max()))
            if nbad or neg0:
                ck.fail("rs", f"sender {q}: {nbad} cells beyond the bound, {neg0} -0 words; max {float(err.nan_to_num(1e30).max()):.3e}")
        # this GPU's shards for owners 4-15
        for o in range(w, TP):
            ref, lim = self.down_bound(lay.downs[g][o * SHARD:(o + 1) * SHARD], h)
            got = self.scratch[o - w][parity, g, :b].double()
            err = (got - ref).abs()
            nbad = int((~(err <= lim)).sum())
            worst["rs_out"] = max(worst["rs_out"], float((err / lim.clamp(min=1e-30)).nan_to_num(1e30).max()))
            if nbad:
                ck.fail("rs_out", f"owner {o}: {nbad} cells beyond the bound")
        # canary: everything outside the written region still 0x5A5A
        mask = torch.ones(rs.shape[:3], dtype=torch.bool, device=self.device)
        mask[parity, :w, :b] = False
        spill = int((rs.view(torch.int16)[mask] != FILL16).sum())
        smask = torch.ones(rs.shape[:3], dtype=torch.bool, device=self.device)
        smask[parity, g, :b] = False
        spill += sum(int((s.view(torch.int16)[smask] != FILL16).sum()) for s in self.scratch)
        H.record({"kind": "check", "item": ck.name, "worst_err_over_bound": worst, "canary_changed": spill})
        print("CHECK", ck.name, json.dumps({"worst_err_over_bound": worst, "canary_changed": spill}), flush=True)
        if spill:
            ck.fail("canary", f"{spill} words outside the written region changed")
        recv = (rs[parity, :w, :b].clone(), torch.stack([s[parity, g, :b] for s in self.scratch]).clone())
        self.saved[(lay.layer, call, b)] = recv
        if call == 2:
            r0 = self.saved[(lay.layer, 0, b)]
            if not all(torch.equal(a.view(torch.int16), c.view(torch.int16)) for a, c in zip(recv, r0)):
                ck.fail("determ", "call 2 (parity 2) received bytes differ from call 0 (parity 0)")

    def graph(self, lay, b, fails):
        front = lay.fronts["T"]
        ck = Checker(f"wdown.graph.b{b}.l{lay.layer}.g{self.g}", fails)
        x, logits_ref, lat_ref = self.inputs(lay, 0, b)
        xc = canon_bf16(x).clone()
        tail = self.tail(0)
        self.arm_rs()
        self.inject(b, logits_ref, lat_ref)
        self.sync()
        self.run_t(front, xc, tail)  # warm once (consumes this injection)
        self.sync()
        graph, cap = torch.cuda.CUDAGraph(), torch.cuda.Stream(device=self.device)
        with torch.cuda.stream(cap):
            with torch.cuda.graph(graph, stream=cap):
                self.run_t(front, xc, tail)
        self.sync()
        self.arm_rs()
        self.inject(b, logits_ref, lat_ref)
        self.sync()
        graph.replay()
        self.sync()
        rs, g, w = self.tb.rs.buffer, self.g, self.world
        recv = (rs[0, :w, :b], torch.stack([s[0, g, :b] for s in self.scratch]))
        r0 = self.saved[(lay.layer, 0, b)]
        if not all(torch.equal(a.view(torch.int16), c.view(torch.int16)) for a, c in zip(recv, r0)):
            ck.fail("graph", "received RS bytes differ from the eager call 0")
        H.record({"kind": "check", "item": ck.name, "ok": True})
        del graph

    # ------------------------------------------------------------------ timing (Tier 0)
    def build_timed(self, arm, b):
        items = []
        for lay in self.layers:
            x, logits_ref, lat_ref = self.inputs(lay, 0, b)
            row = self.front_box.buffer[0, :b].view(torch.float32)
            w = self.world
            src = (M.canon(lat_ref.float()[:, w * LROWS:]).clone(), M.canon(logits_ref.float()[:, w * LOGITS:]).clone())
            items.append((lay, canon_bf16(x).clone(), row, src))
        tail = self.tail(0)

        def chain():
            for lay, xc, row, (lat_src, log_src) in items:
                row[:, self.world * LROWS: LAT].copy_(lat_src)
                row[:, LAT + self.world * LOGITS:].copy_(log_src)
                torch.cuda._sleep(4000)  # ~2 us at ~2 GHz: the o-proj stand-in (no dependents trigger)
                if arm == "T":
                    self.run_t(lay.fronts["T"], xc, tail)
                else:
                    self.run_s(lay.fronts["S"], xc, tail, b)

        chain()  # warm
        self.sync()
        graph, cap = torch.cuda.CUDAGraph(), torch.cuda.Stream(device=self.device)
        with torch.cuda.stream(cap):
            with torch.cuda.graph(graph, stream=cap):
                chain()
        self.sync()
        return graph


FAMILIES = (("spin", "spin_kernel"), ("ftm", "MoeFrontTmem64"), ("fpf", "Fp32FrontGemm"), ("route", "RouteSitu"),
            ("wplan", "WidePlan"), ("sdown", "SharedDown"), ("splitk", "SplitK"))


def segments(trace_path, schedule, nlayers):
    d = json.load(gzip.open(trace_path))
    ks = sorted((e for e in d["traceEvents"] if e.get("cat") == "kernel"), key=lambda e: e["ts"])
    fam = []
    for e in ks:
        f = next((n for n, p in FAMILIES if p in e["name"]), None)
        if f is not None:
            fam.append((f, e))
    spins = [i for i, (f, _) in enumerate(fam) if f == "spin"]
    if len(spins) != nlayers * len(schedule):
        raise RuntimeError(f"{len(spins)} spin kernels, expected {nlayers * len(schedule)}")
    end = lambda e: e["ts"] + e["dur"]  # noqa: E731
    grid = lambda e: tuple(e.get("args", {}).get("grid", (0, 0, 0)))  # noqa: E731
    out = []
    for k, i in enumerate(spins):
        arm, b = schedule[k // nlayers]
        seg = [x for x in fam[i + 1: spins[k + 1] if k + 1 < len(spins) else len(fam)]]
        by = {n: [e for f, e in seg if f == n] for n, _ in FAMILIES}
        s_end = end(fam[i][1])
        routes = sorted(by["route"], key=lambda e: grid(e)[0])
        if arm == "T":
            assert len(by["ftm"]) == 1 and len(routes) == 1 and len(by["sdown"]) == 1, (arm, b, {n: len(v) for n, v in by.items()})
            fe, fr, sd = by["ftm"][0], routes[0], by["sdown"][0]
        else:
            assert len(by["fpf"]) == 1 and len(by["wplan"]) == 1 and len(routes) == 2 and len(by["splitk"]) >= 2, \
                (arm, b, {n: len(v) for n, v in by.items()})
            assert grid(routes[0])[0] < grid(routes[1])[0], "S: route and SiTU RouteSitu grids not distinct"
            fe, fr, sd = by["fpf"][0], by["wplan"][0], max(by["splitk"], key=end)
        out.append((k // nlayers, arm, b, {"F": end(fr) - s_end, "fe": end(fe) - s_end, "sd": end(sd) - s_end}))
    return out


def med(v):
    return round(statistics.median(v), 3) if v else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--deadline-s", type=float, default=2.0)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--time", action="store_true")
    ap.add_argument("--reps", type=int, default=20)
    args = ap.parse_args()
    os.environ["K31MK_MOE_WIDE"] = ROWS  # before MoEScratch: the scratch then holds the wide engine's plan
    for k in ("K31MK_MOE_FRONT_TMEM", "K31MK_MOE_WIDE_FOLD", "K31MK_MOE_SPLIT_SMS"):
        os.environ.pop(k, None)
    local = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local)
    device = torch.device("cuda", local)
    dist.init_process_group("nccl", device_id=device)
    g, world = dist.get_rank(), dist.get_world_size()
    assert world == 4, "TP ranks 0-3 real, 4-15 injected"
    tag = os.environ.get("K31_MUTATION") or "base"
    os.makedirs(args.out, exist_ok=True)
    os.environ["K31_RESULTS"] = os.path.join(args.out, f"wdown_{tag}_g{g}.jsonl")
    H.load_plugin(MUTATIONS)
    from sgl_nvidia_mega.kimi_k31.k3.cutedsl.common import host
    from sgl_nvidia_mega.kimi_k31.k3.cutedsl.moe import experts_runtime as ER
    from sgl_nvidia_mega.kimi_k31.k3.cutedsl.moe import front as FR
    from sgl_nvidia_mega.kimi_k31.k3.cutedsl.moe import prepared as PM
    from sgl_nvidia_mega.kimi_k31.k3.cutedsl.moe.tail_runtime import MoeTailBox
    from sgl_nvidia_mega.kimi_k31.k3.cutedsl.oproj import prepared as OP

    H.record({"kind": "env", **H.device_facts(), "world": world, "mutation": tag})
    fails, res = [], {}
    t0 = time.time()
    try:
        layers = (11,) if args.quick else (11, 12)
        arms = ("T", "S") if args.time else ("T",)
        rig = Rig((host, OP, FR, PM, MoeTailBox, ER), dist.group.WORLD, g, world, device, int(args.deadline_s * 1e9),
                  layers, arms)
        print(f"g{g} rig built {time.time() - t0:.0f}s", flush=True)
        for call in range(1 if args.quick else 3):
            for lay in rig.layers:
                for b in BATCHES:
                    rig.check_call(lay, call, b, fails)
        if not args.quick:
            for b in BATCHES:
                rig.graph(rig.layers[0], b, fails)
        print(f"g{g} checks done {time.time() - t0:.0f}s failures={len(fails)}", flush=True)
        n = torch.tensor([len(fails)], device=device)
        dist.all_reduce(n)
        if args.time and int(n) == 0:
            res = timing(rig, args, g)
    except Exception as e:  # noqa: BLE001
        fails.append(f"g{g}: {type(e).__name__}: {e}")
        traceback.print_exc()
    with open(os.path.join(args.out, f"wdown_{tag}_g{g}.failures.json"), "w") as f:
        json.dump({"failures": fails, "timing": res}, f, indent=1)
    n = torch.tensor([len(fails)], device=device)
    try:
        dist.all_reduce(n)
    except Exception:  # noqa: BLE001
        pass
    if g == 0:
        print(f"TP_RESULT failures={int(n)} ({time.time() - t0:.0f}s)", flush=True)
    dist.destroy_process_group()
    sys.exit(1 if int(n) else 0)


def timing(rig, args, g):
    """Tier 0 timing; an error here is reported as TIMING_ERROR (the correctness verdict stands)."""
    res = {}
    try:
        graphs = {(arm, b): rig.build_timed(arm, b) for b in BATCHES for arm in ("S", "T")}
        for gr in graphs.values():
            for _ in range(3):
                gr.replay()
        rig.sync()
        schedule = [(arm, b) for _ in range(args.reps) for b in BATCHES for arm in ("S", "T", "T", "S")]
        from torch.profiler import ProfilerActivity, profile

        with profile(activities=[ProfilerActivity.CUDA]) as prof:
            for key in schedule:
                graphs[key].replay()
            torch.cuda.synchronize()
        raw = os.path.join(args.out, f"trace_wdown_g{g}.json")
        prof.export_chrome_trace(raw)
        with open(raw, "rb") as f, gzip.open(raw + ".gz", "wb") as z:
            z.write(f.read())
        os.remove(raw)
        seg = segments(raw + ".gz", schedule, len(rig.layers))
        per = {}
        for _, arm, b, vals in seg:
            for k, v in vals.items():
                per.setdefault(f"{arm}.b{b}.{k}", []).append(v)
        # paired d = T - S per ABBA block (block = 4 consecutive replays of one batch: S T T S)
        blocks = {}
        for r, arm, b, vals in seg:
            blocks.setdefault((r // 4, b), {}).setdefault(arm, []).append(vals["F"])
        pairs = {}
        for (blk, b), v in blocks.items():
            pairs.setdefault(b, []).append(statistics.mean(v["T"]) - statistics.mean(v["S"]))
        res = {k: med(v) for k, v in per.items()}
        res.update({f"dF.b{b}": med(v) for b, v in pairs.items()})
        res["n_blocks"] = {b: len(v) for b, v in pairs.items()}
        print("TIMING", g, json.dumps(res), flush=True)
        H.record({"kind": "timing", "item": f"wdown.g{g}", **res})
    except Exception as e:  # noqa: BLE001
        res["timing_error"] = f"{type(e).__name__}: {e}"[:2000]
        try:  # which kernel names the trace held (diagnosis of a name / count mismatch)
            d = json.load(gzip.open(os.path.join(args.out, f"trace_wdown_g{g}.json.gz")))
            names = sorted({e["name"][:120] for e in d["traceEvents"] if e.get("cat") == "kernel"})
            res["kernel_names"] = names[:60]
        except Exception:  # noqa: BLE001
            pass
        print("TIMING_ERROR", g, res["timing_error"], flush=True)
        traceback.print_exc()
    return res


if __name__ == "__main__":
    main()
