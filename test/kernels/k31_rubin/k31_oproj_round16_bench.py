# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""In-chain A/B of the TP16 o-projection with the old (two roundings) and new (one rounding, as stock) consumer
arithmetic. One process holds both versions: "before" is the base-D plugin copied under the
package name ``sgl_nvidia_mega_before`` (directory ``K31_BEFORE_ROOT``; every import inside ``oproj`` / ``common`` is
relative), "after" is this tree.

Per rows (8 = OprojFused, 16 = split consumer, 32 = two-shot), a chain of L distinct has_delta layers (distinct
weights, L chosen so a replay streams >= 3x L2; parities alternate), each layer = [memcpy of ranks 4-15's injected
partials] + PreparedOproj.run, captured as one CUDA graph per arm, over W = 4 GPUs (GPU g = TP rank g, real symm_mem
multicast; one shared mailbox and two-shot stand-in).

Path check (before timing, per arm and rows, eager, dataset E of k31_oproj_round16): "after" must give the stock
emulation bit for bit, "before" the double-rounding emulation bit for bit (two-shot: the kernel-reduced owner
columns 0-1791); else the arm is VOID.
Metric (pre-registered): per-layer us = CUDA-event wall of one replay / L; per cycle and
arm the median of ``--reps`` replays; the max over the 4 GPUs. Order A B B A per cycle (A = before, B = after),
``--cycles`` cycles after warm replays. Resolved iff |median(B) - median(A)| > 2 x max(spread A, spread B), spread =
max - min of the arm's per-cycle values. Secondary: CUPTI durations of the o-proj kernels per arm (torch.profiler).

Usage (one node, 4 GPUs): K31_BEFORE_ROOT=DIR python3 -m torch.distributed.run --nproc-per-node 4
  k31_oproj_round16_bench.py --out DIR [--rows 8,16,32] [--cycles 8] [--reps 5]
"""

import argparse
import importlib
import json
import os
import statistics
import sys
import traceback

# "after" = the knob-on single rounding; "before" = the base-D copy (double rounding).
os.environ.setdefault("K31MK_OPROJ_SINGLE_ROUND", "1")

import torch
import torch.distributed as dist

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _harness as H  # noqa: E402
import _ref_comm as C  # noqa: E402
import k31_oproj_round16 as R  # noqa: E402

TP, K, HIDDEN, OWN = R.TP, R.K, R.HIDDEN, R.OWN
# has_delta layer ids (nb 1..8) with alternating parity: (layer % 12 != 0)
DELTA_LAYERS = [l for l in range(1, 93) if l % 12 != 0]


def chain_layers(n):
    """n layer ids alternating even / odd parity, all has_delta."""
    evens = [l for l in DELTA_LAYERS if l % 2 == 0]
    odds = [l for l in DELTA_LAYERS if l % 2 == 1]
    out = []
    for i in range(n):
        src = evens if i % 2 == 0 else odds
        out.append(src[(i // 2) % len(src)])
    return out


class Bench:
    def __init__(self, mods, group, g, world, device, rows, n_layers, deadline_ns):
        self.mods, self.group, self.g, self.world, self.device, self.rows = mods, group, g, world, device, rows
        after = mods["B"]
        self.mailbox = after.OprojMailbox(group, device, TP, g)
        self.twoshot = R.TwoShot16(group, device, world) if rows in after.twoshot_batches(TP) else None
        self.layers = chain_layers(n_layers)
        b, d = rows, device
        self.weights, self.preps, self.bufs = [], {"A": [], "B": []}, []
        for i, layer in enumerate(self.layers):
            gw = R._gen(d, "bw", i)
            w = R.small_int((HIDDEN, K), 2.0**-6, gw, d)
            self.weights.append(w)
            for arm, mod in mods.items():
                dec = C.make_decoder(layer, w, d)
                self.preps[arm].append(mod.PreparedOproj(dec, self.mailbox, (b,), i % 3 == 0, None, self.twoshot,
                                                         deadline_ns=deadline_ns))
            gx = R._gen(d, "bx", i)
            prefix_old, bank = C.state_inputs(layer, 100 + i, b, d)
            parity = layer % 2
            if self.twoshot is None:
                inj = [(self.mailbox.buffer[parity, world:TP, :b],
                        R.canon(torch.randn((TP - world, b, HIDDEN), generator=gx, device=d).to(torch.bfloat16)))]
            else:
                inj = [(self.twoshot.recv[parity, world:TP, :b],
                        R.canon(torch.randn((TP - world, b, OWN), generator=gx, device=d).to(torch.bfloat16))),
                       (self.twoshot.gather[parity, 0, :b, world * OWN:],
                        R.canon(torch.randn((b, HIDDEN - world * OWN), generator=gx, device=d).to(torch.bfloat16)))]
            self.bufs.append({"x": R.small_int((b, K), 1.0, gx, d), "prefix0": prefix_old, "prefix": prefix_old.clone(),
                              "bank": bank, "out": torch.empty((b, HIDDEN), dtype=torch.bfloat16, device=d),
                              "inj": inj})

    def sync(self):
        torch.cuda.synchronize()
        dist.barrier(group=self.group)

    def run_chain(self, arm):
        for prep, buf in zip(self.preps[arm], self.bufs):
            for dst, src in buf["inj"]:
                dst.copy_(src)
            prep.run(buf["x"], buf["prefix"], buf["bank"], buf["out"])

    def path_check(self, arm, rec):
        """Eager, layer 0 only, dataset E of k31_oproj_round16 (all 16 partials integer-exact)."""
        b, d, w = self.rows, self.device, self.world
        layer = self.layers[0]
        gx = R._gen(d, "px", layer)
        xs = [R.small_int((b, K), 1.0, gx, d) for _ in range(TP)]
        gw = R._gen(d, "pw", layer)
        ws = [R.small_int((HIDDEN, K), 2.0**-6, gw, d) for _ in range(TP)]
        partials = [R.canon((xs[q].double() @ ws[q].double().T).float().to(torch.bfloat16)) for q in range(TP)]
        prefix_old, bank = C.state_inputs(layer, 7, b, d)
        dec = C.make_decoder(layer, ws[self.g], d)
        prep = self.mods[arm].PreparedOproj(dec, self.mailbox, (b,), False, None, self.twoshot, deadline_ns=2_000_000_000)
        stock = R.canon(R.stock_sum(partials, prefix_old).to(torch.bfloat16))
        plain = R.stock_sum(partials, None)
        double = R.canon((prefix_old.float() + plain.to(torch.bfloat16).float()).to(torch.bfloat16))
        parity = layer % 2
        self.sync()
        if self.twoshot is None:
            self.mailbox.buffer[parity, w:TP, :b].copy_(torch.stack(partials[w:]))
        else:
            lo, hi = self.g * OWN, (self.g + 1) * OWN
            self.twoshot.recv[parity, w:TP, :b].copy_(torch.stack([p[:, lo:hi] for p in partials[w:]]))
            conv = stock if arm == "B" else R.canon(plain.to(torch.bfloat16))  # each arm's gather-row convention
            self.twoshot.gather[parity, 0, :b, w * OWN:].copy_(conv[:, w * OWN:])
        prefix, out = prefix_old.clone(), torch.empty((b, HIDDEN), dtype=torch.bfloat16, device=d)
        self.sync()
        prep.run(xs[self.g], prefix, bank, out)
        self.sync()
        cols = slice(0, w * OWN) if self.twoshot is not None else slice(None)
        got = R.bits(prefix)[:, cols]
        n_stock = int((got != R.bits(stock)[:, cols]).sum())
        n_double = int((got != R.bits(double)[:, cols]).sum())
        exercised = int((R.bits(double)[:, cols] != R.bits(stock)[:, cols]).sum())
        ok = (n_stock == 0) if arm == "B" else (n_double == 0 and n_stock > 0)
        row = {"kind": "path", "rows": b, "arm": arm, "cells": got.numel(), "mismatch_vs_stock": n_stock,
               "mismatch_vs_double": n_double, "exercised": exercised, "ok": ok}
        rec.append(row)
        print("PATH", json.dumps(row), flush=True)
        return ok

    def time(self, cycles, reps, rec):
        graphs = {}
        for arm in ("A", "B"):
            self.sync()
            graphs[arm], stream = torch.cuda.CUDAGraph(), torch.cuda.Stream()
            with torch.cuda.graph(graphs[arm], stream=stream):
                self.run_chain(arm)
        self.sync()

        def replay(arm, n):
            vals = []
            for _ in range(n):
                dist.barrier(group=self.group)
                a, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                a.record()
                graphs[arm].replay()
                e.record()
                e.synchronize()
                vals.append(a.elapsed_time(e) * 1e3 / len(self.layers))
            return vals

        for arm in ("A", "B"):
            replay(arm, 3)
        per = {"A": [], "B": []}
        for _ in range(cycles):
            for arm in ("A", "B", "B", "A"):
                mine = torch.tensor([statistics.median(replay(arm, reps))], device=self.device)
                allv = [torch.zeros_like(mine) for _ in range(self.world)]
                dist.all_gather(allv, mine, group=self.group)
                per[arm].append(max(float(v) for v in allv))
        res = {}
        for arm in ("A", "B"):
            v = per[arm]
            res[arm] = {"median": statistics.median(v), "spread": max(v) - min(v), "n": len(v), "values": v}
        delta = res["B"]["median"] - res["A"]["median"]
        bound = 2 * max(res["A"]["spread"], res["B"]["spread"])
        row = {"kind": "time", "rows": self.rows, "layers": len(self.layers), "us_per_layer": res,
               "delta_B_minus_A": delta, "resolution": bound, "resolved": abs(delta) > bound}
        rec.append(row)
        if self.g == 0:
            print("TIME", json.dumps({k: v for k, v in row.items() if k != "us_per_layer"}),
                  json.dumps({a: {k: res[a][k] for k in ("median", "spread", "n")} for a in res}), flush=True)

        from torch.profiler import ProfilerActivity, profile

        for arm in ("A", "B"):
            self.sync()
            with profile(activities=[ProfilerActivity.CUDA]) as prof:
                for _ in range(3):
                    graphs[arm].replay()
                torch.cuda.synchronize()
            durs = {}
            for evt in prof.events():
                if evt.device_type.name != "CUDA" or "oproj" not in evt.name.lower():
                    continue
                dur = evt.device_time if hasattr(evt, "device_time") else evt.cuda_time
                # Group by kernel kind, never by a name prefix: the two packages' mangled names differ in length, and a
                # 60-character prefix merged the "before" producer and consumer into one median (Rubin r1, rows 16 / 32).
                name = evt.name.lower()
                kind = next((k for k in ("fused", "consumer", "umma", "producer") if f"oproj{k}" in name), "other")
                durs.setdefault(kind, []).append(dur)
            kern = {k: {"median_us": statistics.median(v), "n": len(v)} for k, v in durs.items()}
            row = {"kind": "kernels", "rows": self.rows, "arm": arm, "g": self.g, "kernels": kern}
            rec.append(row)
            print("KERNELS", json.dumps(row), flush=True)
        del graphs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--rows", default="8,16,32")
    ap.add_argument("--cycles", type=int, default=8)
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--layers", type=int, default=0, help="0: enough distinct weights to stream >= 3x L2")
    ap.add_argument("--tag", default="run")
    args = ap.parse_args()
    local = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local)
    device = torch.device("cuda", local)
    dist.init_process_group("nccl", device_id=device)
    g, world = dist.get_rank(), dist.get_world_size()
    assert world == 4
    os.makedirs(args.out, exist_ok=True)
    os.environ["K31_RESULTS"] = os.path.join(args.out, f"bench_{args.tag}_g{g}.jsonl")
    H.load_plugin()
    before_root = os.path.realpath(os.environ["K31_BEFORE_ROOT"])
    sys.path.insert(0, before_root)
    mods = {"A": importlib.import_module("sgl_nvidia_mega_before.kimi_k3.cutedsl.oproj.prepared"),
            "B": importlib.import_module("sgl_nvidia_mega.kimi_k31.k3.cutedsl.oproj.prepared")}
    files = {a: os.path.realpath(m.__file__) for a, m in mods.items()}
    assert files["A"] != files["B"] and before_root in files["A"], files
    H.hold_fp8_clock()
    H.record({"kind": "env", **H.device_facts(), "world": world, "files": files})
    n_layers = args.layers or H.copies_for(HIDDEN * K * 2)
    n_layers += n_layers % 2
    rec, failures = [], []
    for rows in map(int, args.rows.split(",")):
        try:
            bench = Bench(mods, dist.group.WORLD, g, world, device, rows, n_layers, 2_000_000_000)
            oks = [bench.path_check(arm, rec) for arm in ("A", "B")]
            ok = torch.tensor([int(all(oks))], device=device)
            dist.all_reduce(ok, op=dist.ReduceOp.MIN)
            if not int(ok):
                failures.append(f"rows {rows}: path check failed (arm VOID)")
                print("FAIL path", rows, flush=True)
                continue
            bench.time(args.cycles, args.reps, rec)
            del bench
        except Exception as e:  # noqa: BLE001
            failures.append(f"rows {rows}: {e!r}")
            print("CASE_ERROR", rows, repr(e)[:600], flush=True)
            traceback.print_exc()
    for row in rec:
        H.record(row)
    if g == 0:
        times = [r for r in rec if r["kind"] == "time"]
        print("CO_BENCH_RESULT", json.dumps({"tag": args.tag, "layers": n_layers, "failures": failures,
                                             "rows": {r["rows"]: {"A": r["us_per_layer"]["A"]["median"],
                                                                  "B": r["us_per_layer"]["B"]["median"],
                                                                  "spreadA": r["us_per_layer"]["A"]["spread"],
                                                                  "spreadB": r["us_per_layer"]["B"]["spread"],
                                                                  "delta": r["delta_B_minus_A"],
                                                                  "resolution": r["resolution"],
                                                                  "resolved": r["resolved"]} for r in times}}),
              flush=True)
    t = torch.tensor([len(failures)], device=device)
    dist.all_reduce(t)
    print(f"[rank{g}] BENCH_RESULT failures={int(t)}", flush=True)
    dist.destroy_process_group()
    sys.exit(1 if int(t) else 0)


if __name__ == "__main__":
    main()
