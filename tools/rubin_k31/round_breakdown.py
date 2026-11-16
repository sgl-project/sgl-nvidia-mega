# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Per-round time breakdown of a decode / MTP trace (SGLang /start_profile output of one rank, CPU+GPU).

GPU work is grouped by launch (kernels, memcpys and memsets sharing the CUDA correlation id of one
cudaGraphLaunch or one eager launch). Each group is labelled by the innermost SGLang Python frames that
issued the launch. The target forward (decode step or MTP verify) is the largest graph group; a round runs
from the end of one target group to the end of the next, so it contains the draft, the verify and whatever
the host does in between (accept, metadata, sampling copies). Per round the report gives:

  round   wall time between consecutive target ends
  target  wall span of the target group (decode or verify graph); before_target: previous target end to
          this target's start (the draft, host work and launch gaps of the round)
  <label> GPU busy time (union of kernel intervals) of every other group label
  idle    round wall time with no kernel running (host-bound gaps)

Usage: python3 round_breakdown.py TRACE.json[.gz] [--frames 3] [--top 12]
"""

import argparse
import bisect
import collections
import gzip
import json
import statistics

GPU_CATS = ("kernel", "gpu_memcpy", "gpu_memset")
SKIP_FRAMES = ("torch/", "contextlib", "threading.py", "<built-in", "nn/modules/module.py", "utils/_contextlib")


def load(path):
    with (gzip.open(path, "rt") if path.endswith(".gz") else open(path)) as f:
        return json.load(f)["traceEvents"]


def union(intervals):
    total, end = 0.0, -1e30
    for s, e in sorted(intervals):
        if e <= end:
            continue
        total += e - max(s, end)
        end = e
    return total


def frames_at(stacks, tid, ts, depth):
    """Innermost SGLang Python frames on CPU thread ``tid`` that enclose time ``ts``."""
    starts, events = stacks.get(tid, ([], []))
    i = bisect.bisect_right(starts, ts)
    found = []
    for e in reversed(events[max(0, i - 4000):i]):  # enclosing frames start before ts and end after it
        if e["ts"] + e["dur"] >= ts and not any(s in e["name"] for s in SKIP_FRAMES):
            found.append(e)
    found.sort(key=lambda e: -e["ts"])  # innermost first
    names = []
    for e in found:
        n = e["name"].split("/")[-1]
        if n not in names:
            names.append(n)
        if len(names) == depth:
            break
    return " < ".join(names) or "?"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("trace")
    ap.add_argument("--frames", type=int, default=2, help="Python frames in a group label")
    ap.add_argument("--top", type=int, default=12)
    a = ap.parse_args()
    ev = load(a.trace)

    py = collections.defaultdict(list)
    for e in ev:
        if e.get("cat") == "python_function" and e.get("ph") == "X":
            py[e["tid"]].append(e)
    stacks = {}
    for tid, es in py.items():
        es.sort(key=lambda e: e["ts"])
        stacks[tid] = ([e["ts"] for e in es], es)

    launches = {}
    for e in ev:
        if e.get("cat") in ("cuda_runtime", "cuda_driver") and "correlation" in e.get("args", {}):
            launches[e["args"]["correlation"]] = e

    groups = collections.defaultdict(list)
    for e in ev:
        if e.get("cat") in GPU_CATS and e.get("ph") == "X":
            groups[e["args"].get("correlation")].append(e)

    info = []
    for corr, ks in groups.items():
        s = min(k["ts"] for k in ks)
        t = max(k["ts"] + k["dur"] for k in ks)
        lr = launches.get(corr)
        graph = bool(lr and "Graph" in lr["name"])
        label = frames_at(stacks, lr["tid"], lr["ts"], a.frames) if lr else "?"
        info.append(dict(corr=corr, start=s, end=t, n=len(ks), graph=graph, label=label,
                         busy=union([(k["ts"], k["ts"] + k["dur"]) for k in ks]), kernels=ks))
    info.sort(key=lambda g: g["start"])
    big = max(g["n"] for g in info)
    targets = [g for g in info if g["graph"] and g["n"] >= 0.5 * big]
    print(f"groups {len(info)}; target graph groups {len(targets)} (>= {big // 2} kernels; largest {big})")

    rounds = []
    for prev, cur in zip(targets, targets[1:]):
        lo, hi = prev["end"], cur["end"]
        inside = [g for g in info if g["start"] >= lo and g["end"] <= hi and g is not cur]
        per = collections.defaultdict(list)
        for g in inside:
            per[g["label"] + (" [graph]" if g["graph"] else "")].extend(
                (k["ts"], k["ts"] + k["dur"]) for k in g["kernels"])
        allk = [iv for ivs in per.values() for iv in ivs] + [(k["ts"], k["ts"] + k["dur"]) for k in cur["kernels"]]
        r = {"round": hi - lo, "target": cur["end"] - cur["start"], "target_busy": cur["busy"],
             "before_target": cur["start"] - lo,
             "idle": (hi - lo) - union(allk)}
        for lab, ivs in per.items():
            r[lab] = union(ivs)
            if lab.endswith("[graph]"):
                r[lab + " span"] = max(e for _, e in ivs) - min(b for b, _ in ivs)
        rounds.append(r)
    if not rounds:
        print("fewer than two target groups")
        return
    keys = collections.Counter(k for r in rounds for k in r)
    print(f"rounds {len(rounds)}; median us (mean) over rounds; labels = issuing Python frames, innermost first")
    order = ["round", "target", "target_busy", "before_target", "idle"]
    order += sorted((k for k in keys if k not in order), key=lambda k: -statistics.mean(r.get(k, 0) for r in rounds))
    for k in order[: len(order) if a.top <= 0 else a.top + 5]:
        vals = [r.get(k, 0.0) for r in rounds]
        print(f"  {statistics.median(vals):10.1f} ({statistics.mean(vals):10.1f})  {k}  [in {keys[k]} rounds]")
    t = targets[len(targets) // 2]
    print(f"target label: {t['label']}  kernels {t['n']}")


if __name__ == "__main__":
    main()
