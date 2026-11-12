# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Wide-tier M0: the TP16 B8 (Q8 verify) MoE layer inside a 128-row tail box, the box a graphs-1..16 boot
builds (``providers``: o-proj / front / tail boxes follow the largest batch). ``K31MK_MOE_RUBIN8`` (the ported
``RubinExpertsKernel``) used to publish with a fixed 64-row slot stride, so the served c=8 env could not boot with
that box; it now takes the experts' ``box_rows``. Rig: ``k31_tail_opts_b8.Rig`` (PreparedMoE over W = 4 GPUs, ranks
4-15 injected, six layers, stand-in ring consumer), arms rebuilt here with a box size.

Arms (env flags read at construction; each with its own mailboxes):
  A     today's default B8 path, 64-row boxes;     AC    the same, 128-row boxes (front, tail, experts box_rows);
  R8    F's served B8 MoE env (F_B8 below: RUBIN8 + XPLAN_ROUTE + TP16_PUBLISH + TAIL_WAVE8 ...), 64-row boxes = the
        served c=8 build;                          R8C   F_B8, 128-row boxes;   R8C2  a second R8C instance.
  Controls (r2, after wm0r8box_r1: R8 = the served build at 64-row boxes failed hg / shared_h / output exactly as R8C):
  R8nS   F_B8 without K31MK_EXP_SHARED_SKS (gate/up FP32 into the front's hg + RouteSitu, as the rig reads it), 64 rows;
  R8nL   F_B8 with K31MK_MOE_TAIL_WAVE8 minus "late1" (early RMS, the rig's output reference order), 64 rows;
  R8nSL  both changes, 64 rows;                    R8CnSL  both changes, 128-row boxes (bitwise vs R8nSL).
  F's die placement (K31MK_MOE_DIES / DIES8) and its 9-64-row knobs are left out (not B8; placement needs the providers).

Checks:
  rig     ``k31_tail_opts_b8.Rig.check_layer`` per layer and arm (FP64 route / latent / hg / shared / partial_t1 / output vs
          the stock method, rank bitwise, RS shards) and graph replay == eager. Published arms (R8*) write no expert rows
          (``experts`` / ``partial`` items read them): those two items are recorded as INFO for published arms only.
          Two more items are INFO, decided by the arm's ENV (``info_items``), never by its name:
          - K31MK_EXP_SHARED_SKS=1: B<=8 gate/up runs as SplitKSituGemm with h BF16 out (front.py ``self.sks.run``); the
            FP32 gate/up the rig reads as ``hg`` (front views[1]) is not the full product, and ``shared_h`` is derived
            from it -> ``hg`` / ``shared_h`` INFO (``shared`` = S vs FP64 down @ h_mk stays a gate).
          - WAVE8 "late1": the one-group WaveMoeTail's late RMS (MMA on BF16(x * gamma), 1/rms after; tail.py WaveMoeTail)
            while the rig's ``output`` reference is the early-RMS order bf(up @ bf(bf(L) * inv * gamma)) -> ``output`` INFO.
          The controls R8nS / R8nL / R8nSL keep every other F_B8 key, so those items stay gates there.
  engaged R8 / R8C / R8C2: every layer's B8 call runs ``RubinExpertsKernel`` (``8 in experts.rubin8.rubin``) and the tail
          is published (B8 in ``tail.pushers``); A / AC: no RUBIN8.
  bitwise AC == A and R8C, R8C2 == R8 on eager outputs, graph outputs, ids, latent partial slot, shared S (box rows
          change only addresses). A check item failing in R8C but not in R8 fails; an item failing equally in R8 and
          R8C is a rig / env mismatch of the served env, reported (it fails the run; the read rule in the runner says how
          it is read).
Mutations (must fail): r8_stride64 (RubinExpertsKernel publishes at the old 64-row stride), r8_nobox (the RUBIN8 engine
  is built without the layer's box_rows).
Timing (--time R, info): CUDA-event time of 100 graph replays / 6 layers per arm, arm order alternating.

Usage: python3 -m torch.distributed.run --nproc-per-node 4 k31_rubin8_box.py --out DIR [--arms A,AC,R8,R8C,R8C2] [--time 4]
       python3 k31_rubin8_box.py --selftest   (CPU, stdlib only: arm envs and the INFO rule vs hand-written expectations)
"""

import argparse
import json
import os
import sys
import time
import traceback
from types import SimpleNamespace

# the served decode env, the keys that act at B8 (DIES / DIES8 left out); "bundle" = H.bundle_grid_sms() (set in BoxArm).
F_B8 = {
    "K31MK_EXPERTS_GRID_SMS": "bundle", "K31MK_EXPERTS_HFULL_BAR": "1", "K31MK_EXPERTS_PACKED_K64": "1",
    "K31MK_EXPERTS_PRODUCER_PACE": "240", "K31MK_EXPERTS_STRONG_SYNC": "1", "K31MK_EXPERTS_X3_RELEASE": "3",
    "K31MK_EXP_SHARED_SKS": "1", "K31MK_EXP_SITU_LEAN": "1", "K31MK_EXP_W8": "1", "K31MK_EXP_W8_GC": "1",
    "K31MK_EXP_W8_PUBLISH": "f2spfcw", "K31MK_EXP_XPLAN": "1", "K31MK_EXP_XPLAN_ROUTE": "1",
    "K31MK_MOE_COPY_HALVES": "8:1", "K31MK_MOE_EF2": "w", "K31MK_MOE_ROUTE_LEAN8": "w", "K31MK_MOE_RUBIN8": "1",
    "K31MK_MOE_SITU_EARLY": "1-8:1", "K31MK_MOE_TAIL_LATE_RMS": "8:1,16:1", "K31MK_MOE_TAIL_POLL_NS": "0",
    "K31MK_MOE_TAIL_WAVE8": "direct+early+late1+lean_owner+poll0+reload+store16+sxbar+tfence",
    "K31MK_MOE_TP16_PUBLISH": "8:1",
}
SKS, WAVE8 = "K31MK_EXP_SHARED_SKS", "K31MK_MOE_TAIL_WAVE8"


def without_sks(env):
    return {k: v for k, v in env.items() if k != SKS}


def without_late1(env):
    return {**env, WAVE8: "+".join(o for o in env[WAVE8].split("+") if o != "late1")}


ARMS = {  # name: (env, box rows, bitwise reference arm, RUBIN8 expected)
    "A": ({}, 64, "A", False),
    "AC": ({}, 128, "A", False),
    "R8": (F_B8, 64, "R8", True),
    "R8C": (F_B8, 128, "R8", True),
    "R8C2": (F_B8, 128, "R8", True),
    "R8nS": (without_sks(F_B8), 64, "R8nS", True),
    "R8nL": (without_late1(F_B8), 64, "R8nL", True),
    "R8nSL": (without_late1(without_sks(F_B8)), 64, "R8nSL", True),
    "R8CnSL": (without_late1(without_sks(F_B8)), 128, "R8nSL", True),
}


def info_items(name):
    """Check items of arm ``name`` recorded as INFO instead of failing (see the module docstring), from its env."""
    env, _, _, published = ARMS[name]
    items = {"experts", "partial"} if published else set()
    if env.get(SKS) == "1":
        items |= {"hg", "shared_h"}
    if "late1" in env.get(WAVE8, "").split("+"):
        items.add("output")
    return items


def classify(name, fail):
    """'info' or 'fail' for one per-token failure line 'b8.<arm>.l<L>.t<T>.g<G>.<item>: ...' of arm ``name``."""
    head = fail.split(":", 1)[0].split(".")
    tokenwise = len(head) == 6 and head[0] == "b8" and head[1] == name and head[3].startswith("t")
    return "info" if tokenwise and head[5] in info_items(name) else "fail"


def selftest():
    """CPU: expectations written by hand from F's env and the docstring rule."""
    bad = []
    want_wave = "direct+early+lean_owner+poll0+reload+store16+sxbar+tfence"
    for name in ("R8nS", "R8nSL", "R8CnSL"):
        if SKS in ARMS[name][0]:
            bad.append(f"{name} keeps {SKS}")
    for name in ("R8nL", "R8nSL", "R8CnSL"):
        if ARMS[name][0].get(WAVE8) != want_wave:
            bad.append(f"{name} WAVE8 {ARMS[name][0].get(WAVE8)!r}")
    for name, drop in (("R8nS", {SKS}), ("R8nL", {WAVE8}), ("R8nSL", {SKS, WAVE8}), ("R8CnSL", {SKS, WAVE8})):
        env = ARMS[name][0]
        same = {k: v for k, v in F_B8.items() if k not in drop}
        if {k: v for k, v in env.items() if k != WAVE8 or WAVE8 not in drop} != same:
            bad.append(f"{name} differs from F_B8 beyond {sorted(drop)}")
        if env.get("K31MK_MOE_RUBIN8") != "1":
            bad.append(f"{name} lost RUBIN8")
    want = {"A": set(), "AC": set(), "R8": {"experts", "partial", "hg", "shared_h", "output"},
            "R8C": {"experts", "partial", "hg", "shared_h", "output"},
            "R8C2": {"experts", "partial", "hg", "shared_h", "output"}, "R8nS": {"experts", "partial", "output"},
            "R8nL": {"experts", "partial", "hg", "shared_h"}, "R8nSL": {"experts", "partial"},
            "R8CnSL": {"experts", "partial"}}
    for name, items in want.items():
        if info_items(name) != items:
            bad.append(f"info_items({name}) = {sorted(info_items(name))}, want {sorted(items)}")
    cases = [("R8", "b8.R8.l21.t0.g0.output: 8 cells beyond 2 ulp", "info"),
             ("R8nSL", "b8.R8nSL.l21.t0.g0.output: 8 cells beyond 2 ulp", "fail"),
             ("R8nL", "b8.R8nL.l21.t0.g0.output: 8 cells beyond 2 ulp", "fail"),
             ("R8nL", "b8.R8nL.l21.t0.g0.hg: 768 cells; max 9.4", "info"),
             ("R8nS", "b8.R8nS.l21.t0.g0.shared_h: 384 hidden cells", "fail"),
             ("R8", "b8.R8.l21.t0.g0.shared: 3 cells vs down @ h_mk", "fail"),
             ("R8", "b8.R8.l21.g0.partial_fro: 2.0e-02 > 1e-2", "fail"),
             ("R8C", "b8.R8C.l2.g0.bitwise_vs_R8: ['output']", "fail"),
             ("R8C", "b8.R8C.graph0.l2.g0: differs from eager", "fail"),
             ("R8C", "b8.R8C.engaged.g0: ['tail not published at B8']", "fail"),
             ("A", "b8.A.l21.t0.g0.hg: 768 cells", "fail"),
             ("R8C", "b8.R8.l21.t0.g0.output: other arm's line", "fail")]
    for name, line, w in cases:
        if classify(name, line) != w:
            bad.append(f"classify({name}, {line!r}) = {classify(name, line)}, want {w}")
    if ARMS["R8CnSL"][2] != "R8nSL" or ARMS["R8CnSL"][1] != 128 or ARMS["R8nSL"][1] != 64:
        bad.append("R8CnSL must be the 128-row box compared bitwise with R8nSL (64)")
    for b in bad:
        print("SELFTEST_FAIL", b)
    print(f"SELFTEST {'PASS' if not bad else 'FAIL'} ({len(bad)} problems)")
    return 1 if bad else 0


if __name__ == "__main__" and sys.argv[1:2] == ["--selftest"]:
    sys.exit(selftest())

import torch  # noqa: E402
import torch.distributed as dist  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _harness as H  # noqa: E402
import _ref_moe as R  # noqa: E402
import k31_tail_opts_b8 as T  # noqa: E402
from k31_moe_tp16 import LAT, SENT, SHARD, TP  # noqa: E402

B, NLAYERS = T.B, T.NLAYERS
ENV_KEYS = tuple(sorted(set(T.FLAGS) | set(F_B8) | {"K31MK_EXPERTS_GRID_SMS"}))
RUBIN = "kimi_k31/k3/cutedsl/rubin/moe/experts_rubin.py"
ER_ = "kimi_k31/k3/cutedsl/moe/experts_runtime.py"
MUTATIONS = {
    "r8_stride64": (RUBIN, "(parity * self.tp + rank) * self.box_rows + token) * 3584 + column  # box_rows: 64 unless",
                    "(parity * self.tp + rank) * 64 + token) * 3584 + column  # box_rows: 64 unless"),
    "r8_nobox": (ER_, "dies=dies8, q=2,\n                **self.box_kw,\n", "dies=dies8, q=2,\n"),
}


class BoxArm(T.Arm):
    """``k31_tail_opts_b8.Arm`` with the box rows of a wide boot (front box, tail box, experts box_rows)."""

    def __init__(self, name, rig):  # noqa: D107 (same body as T.Arm.__init__ plus the box)
        env, cap, _, _ = ARMS[name]
        mods = rig.mods
        self.name, self.rig, self.cap = name, rig, cap
        g, d, group = rig.g, rig.device, rig.group
        saved = {k: os.environ.get(k) for k in ENV_KEYS}
        for k in ENV_KEYS:
            os.environ.pop(k, None)
        os.environ.update({k: str(H.bundle_grid_sms()) if v == "bundle" else v for k, v in env.items()})
        try:
            self.front_box = mods.OP.OprojMailbox(group, d, TP, g, (1, cap, 2 * LAT + 2 * 896))
            self.state = mods.PM.MoEScratch(TP, d, (B,))
            tb = self.tail_box = mods.TR.MoeTailBox(group, d, TP, g, cap=cap)
            self.scratch = [torch.empty_like(tb.rs.buffer).view(torch.int32).fill_(SENT).view(torch.bfloat16)
                            for _ in range(TP - rig.world)]
            ptrs = list(tb.rs.handle.buffer_ptrs) + [s.data_ptr() for s in self.scratch]
            self.peers = torch.tensor(ptrs, dtype=torch.int64, device=d)
            tb.tail = (mods.host.to_cute(self.peers, 16), mods.TR.flat(tb.out.buffer),
                       rig.Int64(tb.latent.mc), rig.Int64(tb.out.mc))
            op = SimpleNamespace(rank=g, contract=SimpleNamespace(tp_size=TP, rms_eps=T.EPS))
            box_kw = {} if cap == 64 else {"box_rows": cap}  # as providers passes it
            self.layers = []
            for i, lay in enumerate(rig.layers):
                experts = rig.experts_cls(*lay.host_w, beta=R.BETA, linear_beta=R.LIN, activation_floor=False,
                                          **rig.planned_options, **box_kw)
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
            for k, v in saved.items():
                if v is None:
                    os.environ.pop(k, None)
                else:
                    os.environ[k] = v
        self.saves = [torch.full((B, rig.world * SHARD // 2), SENT, dtype=torch.int32, device=d)
                      for _ in range(NLAYERS)]
        H.record({"kind": "arm", "arm": name, "env": env, "cap": cap, "grid": self.grid, "published": self.published,
                  "tail_cap": tb.cap, "g": g})


def engaged(rig, arm, fails):
    want = ARMS[arm.name][3]
    r8 = [bool(getattr(lay.experts, "rubin8", None) is not None and B in lay.experts.rubin8.rubin) for lay in arm.layers]
    bad = []
    if want and not all(r8):
        bad.append(f"RUBIN8 not on every layer: {r8}")
    if not want and any(r8):
        bad.append(f"RUBIN8 on a default arm: {r8}")
    if want and not arm.published:
        bad.append("tail not published at B8")
    if arm.tail_box.cap != ARMS[arm.name][1]:
        bad.append(f"tail box cap {arm.tail_box.cap}")
    H.record({"kind": "check", "item": f"b8.{arm.name}.engaged.g{rig.g}", "rubin8": r8, "published": arm.published,
              "bad": bad})
    if bad:
        fails.append(f"b8.{arm.name}.engaged.g{rig.g}: {bad}")
        print("FAIL engaged", arm.name, bad, flush=True)


def graph_outputs(rig, arm):
    for s in arm.saves:
        s.fill_(SENT)
    rig.sync()
    arm.graph.replay()
    rig.sync()
    return [s.clone() for s in arm.saves]


def cross(rig, ref_rows, graph_out, fails):
    for name, arm in rig.arms.items():
        ref = ARMS[name][2]
        if ref == name:
            continue
        if ref not in rig.arms:
            fails.append(f"b8.{name}.g{rig.g}.bitwise: reference arm {ref} not in the run")
            continue
        a = rig.arms[ref]
        published = ARMS[name][3]
        for i in range(NLAYERS):
            ra, rx = ref_rows[(ref, i)], ref_rows[(name, i)]
            same = {"output": torch.equal(a.eager[i], arm.eager[i]),
                    "graph_output": torch.equal(graph_out[ref][i], graph_out[name][i]),
                    "ids": torch.equal(ra[0], rx[0]),
                    "partial": torch.equal(ra[2].view(torch.int16), rx[2].view(torch.int16)),
                    "shared": torch.equal(ra[3].view(torch.int16), rx[3].view(torch.int16))}
            if not published:
                same["experts"] = torch.equal(ra[4].view(torch.int16), rx[4].view(torch.int16))
            H.record({"kind": "cross", "arm": name, "ref": ref, "layer": i, "g": rig.g, **same})
            bad = [k for k, v in same.items() if not v]
            if bad:
                fails.append(f"b8.{name}.l{i}.g{rig.g}.bitwise_vs_{ref}: {bad}")
                print(f"FAIL bitwise_vs_{ref}", name, i, bad, flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--deadline-s", type=float, default=2.0)
    ap.add_argument("--arms", default="A,AC,R8,R8C,R8C2")
    ap.add_argument("--time", type=int, default=0, help="timing replicates (info; 0: none)")
    args = ap.parse_args()
    arms = tuple(args.arms.split(","))
    local = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local)
    device = torch.device("cuda", local)
    dist.init_process_group("nccl", device_id=device)
    g, world = dist.get_rank(), dist.get_world_size()
    tag = os.environ.get("K31_MUTATION") or "base"
    os.makedirs(args.out, exist_ok=True)
    os.environ["K31_RESULTS"] = os.path.join(args.out, f"box_{tag}_g{g}.jsonl")
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
    fails, info = [], []
    t0 = time.time()
    try:
        rig = T.Rig(mods, dist.group.WORLD, g, world, device, int(args.deadline_s * 1e9))
        ref_rows, graph_out = {}, {}
        for name in arms:
            t = time.time()
            arm = rig.arms[name] = BoxArm(name, rig)
            print(f"ARM {name} built cap {arm.cap} grid {arm.grid} published {arm.published} ({time.time() - t:.0f}s)",
                  flush=True)
            got = []
            rig.correctness(arm, got, ref_rows)
            for f in got:
                (info if classify(name, f) == "info" else fails).append(f)
            engaged(rig, arm, fails)
            graph_out[name] = graph_outputs(rig, arm)
            print(f"ARM {name} checked: fails so far {len(fails)} info {len(info)}", flush=True)
        cross(rig, ref_rows, graph_out, fails)
        for rep in range(args.time):
            order = list(arms) if rep % 2 == 0 else list(reversed(arms))
            for k, name in enumerate(order):
                rig.sync()
                rig.time_arm(rig.arms[name], rep, k)
    except Exception as e:  # noqa: BLE001
        fails.append(f"g{g}: {type(e).__name__}: {e}")
        print(f"CASE_ERROR g{g}: {type(e).__name__}: {e}", flush=True)
        traceback.print_exc()
    with open(os.path.join(args.out, f"box_{tag}_g{g}.failures.json"), "w") as f:
        json.dump({"fails": fails, "info_published": info}, f, indent=1)
    n = torch.tensor([len(fails)], device=device)
    try:
        dist.all_reduce(n)
    except Exception:  # noqa: BLE001
        pass
    if g == 0:
        print(f"INFO_PUBLISHED {len(info)} items on rank 0 (env-classified items, see info_items)", flush=True)
        print(f"TP_RESULT failures={int(n)} ({time.time() - t0:.0f}s)", flush=True)
    dist.destroy_process_group()
    sys.exit(1 if int(n) else 0)


if __name__ == "__main__":
    main()
