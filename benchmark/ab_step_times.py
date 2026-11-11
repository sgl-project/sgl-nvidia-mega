# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Served A/B of the DeepSeek-V4.1 decode step from in-graph step stamps.

Two arms (environment / slot changes) launch alternately in palindrome order (A B B A
...) on the same node(s) of a SLURM allocation, each a fresh server with the benchmark
step stamps on (``runtime/bench.py``, SGLANG_NVIDIA_MEGAKERNEL_BENCH_STEP_TIMES) and,
by default, fixed routing (SGLANG_NVIDIA_MEGAKERNEL_BENCH_FIXED_ROUTING, benchmark
only), so both arms do the same expert GEMMs every step. Each launch runs a warm-up
pass and ``--passes`` static_table.py passes (batch 1, ISL 8K / OSL 1K, forced
acceptance 3.51). ``analyze`` turns the stamps into one period per verify step (start
of one verify forward to the next), drops each request's first and last steps and
the warm-up pass, and reports arm B - arm A with a standard error and a 95 %
interval from the launch means (and, for information, from pass means and steps);
with real routing it first removes, within each launch, the part of the period
explained by the step's distinct-expert count (and position).

Run (login node, inside an allocation; stdlib only):

    python3 benchmark/ab_step_times.py run --job <jobid> --topology tp4 --node 0 \\
        --image <sqsh> --model <model dir> --src <this checkout> --out <dir> \\
        --arm off "" --arm pf "AB_SLOTS_ADD=prologue_fork" --launches 2 --passes 4

Analyze (anywhere with numpy):

    python3 benchmark/ab_step_times.py analyze <dir> [--covariate auto|none|dsum|dmax]

Record a routing table from a run with SGLANG_NVIDIA_MEGAKERNEL_BENCH_STEP_TIMES_IDS=1:

    python3 benchmark/ab_step_times.py table <launch dir> --out table.npz
"""

from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
VERIFY, DRAFT, VERIFY_END, DRAFT_END = 0, 1, 2, 3
MAX_ENTRIES = 64
IDS_STRIDE = 2048


# ------------------------------------------------------------------------------- run


def _nodes(job: str) -> list[str]:
    out = subprocess.run(
        f"squeue -j {job} -h -o %N | xargs scontrol show hostnames",
        shell=True,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.split()
    if not out:
        raise SystemExit(f"job {job} has no nodes")
    return out


def _addr(host: str) -> str:
    out = subprocess.run(
        ["getent", "hosts", host], capture_output=True, text=True
    ).stdout
    return out.split()[0] if out else host


def _order(arms: list[str], launches: int) -> list[str]:
    """A B B A A B B A ... (``launches`` per arm)."""
    seq = []
    for i in range(launches):
        seq += arms if i % 2 == 0 else arms[::-1]
    return seq


def _stop(job: str, hosts: list[str]) -> None:
    subprocess.run(
        [
            "srun",
            f"--jobid={job}",
            "--overlap",
            f"-N{len(hosts)}",
            "--ntasks-per-node=1",
            "-w",
            ",".join(hosts),
            "--gres=gpu:0",
            "bash",
            "-c",
            'for i in $(seq 1 30); do pkill -9 -f "[s]glang|[l]aunch_server"; sleep 3; '
            'n=$(pgrep -fc "[s]glang|[l]aunch_server"); [ "$n" = 0 ] && break; done; '
            'echo "$(hostname) left: procs=$n"',
        ],
        check=False,
    )


def cmd_run(a) -> None:
    hosts_all = _nodes(a.job)
    want = 2 if a.topology == "tp8" else 1
    idx = [int(x) for x in a.node.split(",")]
    hosts = [hosts_all[i] for i in idx][:want]
    if len(hosts) != want:
        raise SystemExit(f"{a.topology} needs {want} node(s), got {hosts}")
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    arms = {name: env for name, env in a.arm}
    if len(arms) not in (1, 2):
        raise SystemExit("give two --arm NAME ENV (one for a single-arm recording)")
    order = _order(list(arms), a.launches)
    mounts = [f"{a.model}:/ab_model:ro", f"{a.src}:{a.src}", f"{out}:{out}"]
    if a.cache:
        mounts.append(f"{a.cache}:/root/.cache")
    mounts += a.mount
    common = [
        f"--jobid={a.job}",
        "--overlap",
        f"--container-image={a.image}",
        f"--container-mounts={','.join(mounts)}",
        "--container-remap-root",
        "--no-container-mount-home",
    ]
    meta = {
        "topology": a.topology,
        "arms": arms,
        "order": order,
        "hosts": hosts,
        "fixed": a.fixed,
        "passes": a.passes,
        "image": os.path.basename(a.image),
        "runtime_opt": a.runtime_opt,
        "isl": a.isl,
        "osl": a.osl,
    }
    (out / "run.json").write_text(json.dumps(meta, indent=1))
    head = _addr(hosts[0])
    for i, arm in enumerate(order):
        tag = f"{i:02d}-{arm}"
        d = out / tag
        if (d / "passes.jsonl").exists() and a.resume:
            continue
        (d / "stamps").mkdir(parents=True, exist_ok=True)
        _stop(a.job, hosts)
        env_bench = [f"SGLANG_NVIDIA_MEGAKERNEL_BENCH_STEP_TIMES={d / 'stamps'}"]
        if a.fixed not in ("", "off"):
            env_bench.append(f"SGLANG_NVIDIA_MEGAKERNEL_BENCH_FIXED_ROUTING={a.fixed}")
        if a.ids:
            env_bench.append("SGLANG_NVIDIA_MEGAKERNEL_BENCH_STEP_TIMES_IDS=1")
        if a.no_stamps:
            env_bench = env_bench[1:]
        kvs = env_bench + shlex.split(a.common_env) + shlex.split(arms[arm])
        ab_env = " ".join(kvs)
        # the arm's variables in a file: srun --export cannot carry commas
        (d / "arm.env").write_text(
            "".join(
                f"export {k}={shlex.quote(v)}\n"
                for k, _, v in (kv.partition("=") for kv in kvs)
            )
        )
        export = (
            f"--export=ALL,AB_TOPO={a.topology},AB_SRC={a.src},AB_MODEL=/ab_model,"
            f"HEAD_ADDR={head},DIST_PORT={29000 + (os.getpid() + i) % 900},"
            f"AB_ENVFILE={d / 'arm.env'},AB_RUNTIME_OPT={1 if a.runtime_opt == 'on' else 0}"
        )
        sel = ["-N2", "--ntasks-per-node=1"] if want == 2 else ["-N1", "-n1"]
        print(f"{time.strftime('%F %T')} launch {tag} on {hosts}: {ab_env}", flush=True)
        with open(d / "server.log", "w") as log:
            server = subprocess.Popen(
                [
                    "srun",
                    *common,
                    *sel,
                    "-w",
                    ",".join(hosts),
                    "--gres=gpu:4",
                    export,
                    "bash",
                    str(HERE / "ab_serve.sh"),
                ],
                stdout=log,
                stderr=subprocess.STDOUT,
            )
        with open(d / "client.log", "w") as log:
            rc = subprocess.run(
                [
                    "srun",
                    *common,
                    "-N1",
                    "-n1",
                    "-w",
                    hosts[0],
                    "--gres=gpu:0",
                    "bash",
                    str(HERE / "ab_client.sh"),
                    "http://127.0.0.1:30000",
                    str(d),
                    str(a.passes),
                    str(a.isl),
                    str(a.osl),
                ],
                stdout=log,
                stderr=subprocess.STDOUT,
            ).returncode
        server.terminate()
        try:
            server.wait(60)
        except subprocess.TimeoutExpired:
            server.kill()
        _stop(a.job, hosts)
        print(f"{time.strftime('%F %T')} {tag} client rc={rc}", flush=True)
        if rc != 0 and not a.keep_going:
            raise SystemExit(f"{tag}: client failed, see {d}/client.log and server.log")
    print("RUN DONE", flush=True)


# --------------------------------------------------------------------------- analyze


def _load_launch(d: Path, rank: int = 0):
    import numpy as np

    ev = np.fromfile(d / "stamps" / f"events-r{rank}.bin", dtype=np.int64).reshape(
        -1, 4
    )
    ev = ev[np.argsort(ev[:, 3])]
    layout = json.loads((d / "stamps" / f"layout-r{rank}.json").read_text())
    counts = None
    cpath = d / "stamps" / f"counts-r{rank}.bin"
    if cpath.exists() and cpath.stat().st_size:
        raw = np.fromfile(cpath, dtype=np.int16).reshape(-1, 4 + 2 * MAX_ENTRIES)
        seq = raw[:, :4].copy().view(np.int64).ravel()
        counts = dict(zip(seq.tolist(), raw[:, 4:]))
    return ev, layout, counts


def _features(layout_slot, row, topology):
    """(sum of distinct experts, sum over layers of the larger half) of the target's
    MoE layers in one counts row."""
    dsum = dmax = 0
    for j, e in enumerate(layout_slot["entries"]):
        dist, low = int(row[2 * j]), int(row[2 * j + 1])
        if dist < 0:
            continue
        dsum += dist
        dmax += max(low, dist - low)
    return dsum, dmax


def steps_of_launch(d: Path, a):
    """Per verify step: period, segments, position, covariates, request and pass index."""
    import numpy as np

    ev, layout, counts = _load_launch(d)
    t, meta, pos, seq = ev[:, 0], ev[:, 1], ev[:, 2], ev[:, 3] - 1
    kind = meta & 0xFF
    slot = meta >> 16
    captured = np.array(
        [layout[s]["captured"] if s < len(layout) else False for s in slot]
    )
    keep = captured
    t, kind, slot, pos, seq = t[keep], kind[keep], slot[keep], pos[keep], seq[keep]
    vi = np.nonzero(kind == VERIFY)[0]
    di = np.nonzero(kind == DRAFT)[0]
    rows = []
    req = -1
    prev_same = False
    # the draft start after each verify start
    dpos = np.searchsorted(di, vi)
    for n in range(len(vi) - 1):
        i, j = vi[n], vi[n + 1]
        dt = (t[j] - t[i]) / 1e3  # us
        adv = pos[j] - pos[i]
        same = 0 < adv <= 8 and dt < 50_000
        if not prev_same:
            req += 1
            k = 0
        prev_same = same
        if not same:
            continue
        r = {
            "req": req,
            "k": k,
            "period": dt,
            "pos": int(pos[i]),
            "adv": int(adv),
            "t": int(t[i]),
        }
        k += 1
        if dpos[n] < len(di) and di[dpos[n]] < j:
            dd = di[dpos[n]]
            r["seg_verify"] = (t[dd] - t[i]) / 1e3
            r["seg_draft"] = (t[j] - t[dd]) / 1e3
        # step n's ids are written by the next replay's stamp of the same graph
        if counts is not None and int(seq[j]) in counts and slot[j] == slot[i]:
            r["dsum"], r["dmax"] = _features(
                layout[int(slot[j])], counts[int(seq[j])], a
            )
        rows.append(r)
    # steps per request, to drop the last ones
    per_req = {}
    for r in rows:
        per_req[r["req"]] = per_req.get(r["req"], 0) + 1
    for r in rows:
        r["k_from_end"] = per_req[r["req"]] - 1 - r["k"]
    nreq = req + 1
    return rows, nreq


def _assign_passes(d: Path, rows) -> None:
    """Each step's client pass (passes.jsonl), by its request's middle stamp on the
    host clock (offset: the smallest drain-time minus stamp); -1 outside every pass
    (SGLang's own warm-up, health checks). Pass 0 is the client's warm-up pass."""
    import numpy as np

    clock = np.fromfile(d / "stamps" / "clock-r0.bin", dtype=np.int64).reshape(-1, 2)
    offset = (clock[:, 0] - clock[:, 1]).min() / 1e9
    passes = [
        json.loads(x)
        for x in (d / "passes.jsonl").read_text().splitlines()
        if x.strip()
    ]
    first = {}
    for r in rows:
        lo, hi = first.get(r["req"], (r["t"], r["t"]))
        first[r["req"]] = (min(lo, r["t"]), max(hi, r["t"]))
    which = {}
    for req, (lo, hi) in first.items():
        mid = (lo + hi) / 2e9 + offset
        which[req] = next(
            (p["pass"] for p in passes if p["start"] <= mid <= p["end"]), -1
        )
    for r in rows:
        r["pass"] = which[r["req"]]


def _mean_se(xs):
    import numpy as np

    xs = np.asarray(xs, dtype=float)
    if len(xs) < 2:
        return float(xs.mean()) if len(xs) else float("nan"), float("nan")
    return float(xs.mean()), float(xs.std(ddof=1) / np.sqrt(len(xs)))


def _tq(dof: float) -> float:
    """The two-sided 95 % t quantile."""
    from math import isfinite

    try:
        from scipy.stats import t as tdist

        return float(tdist.ppf(0.975, dof)) if isfinite(dof) else 1.96
    except Exception:  # no scipy
        table = [
            (1, 12.71),
            (2, 4.30),
            (3, 3.18),
            (4, 2.78),
            (5, 2.57),
            (6, 2.45),
            (7, 2.36),
            (8, 2.31),
            (10, 2.23),
            (12, 2.18),
            (15, 2.13),
            (20, 2.09),
            (30, 2.04),
        ]
        return next((v for k, v in table if dof <= k), 1.96)


def _welch(a_vals, b_vals):
    """B - A from unit means, pooled two-sample t (launches of both arms share one
    launch-to-launch variance): difference, SE, dof, 95 % half-width."""
    import numpy as np

    a_vals, b_vals = np.asarray(a_vals, float), np.asarray(b_vals, float)
    na, nb = len(a_vals), len(b_vals)
    diff = float(b_vals.mean() - a_vals.mean())
    dof = na + nb - 2
    pooled = ((na - 1) * a_vals.var(ddof=1) + (nb - 1) * b_vals.var(ddof=1)) / dof
    se = float(np.sqrt(pooled * (1 / na + 1 / nb)))
    return diff, se, dof, _tq(dof) * se


def cmd_analyze(a) -> None:
    import numpy as np

    root = Path(a.dir)
    meta = json.loads((root / "run.json").read_text())
    arms = list(meta["arms"])
    topo = meta["topology"]
    launches = sorted(
        p for p in root.iterdir() if p.is_dir() and (p / "passes.jsonl").exists()
    )
    data = []  # (arm, launch, rows)
    for d in launches:
        arm = d.name.split("-", 1)[1]
        try:
            rows, nreq = steps_of_launch(d, topo)
        except FileNotFoundError as e:
            print(f"{d.name}: no stamps ({e})")
            continue
        _assign_passes(d, rows)
        sel = [
            r
            for r in rows
            if r["pass"] >= 1
            and r["k"] >= a.skip_head
            and r["k_from_end"] >= a.skip_tail
            and (a.page <= 0 or r["pos"] // a.page == (r["pos"] + r["adv"]) // a.page)
        ]
        # host stalls: steps far above the launch's median (median + max(k * MAD, floor))
        # are dropped and counted; they come in episodes of seconds on a node and are
        # not a property of either arm
        if sel:
            y = np.array([r["period"] for r in sel])
            med = float(np.median(y))
            mad = float(np.median(np.abs(y - med)))
            cut = med + max(a.mad * 1.4826 * mad, a.floor)
            out = y > cut
            if out.any():
                print(
                    f"{d.name}: dropped {int(out.sum())} of {len(y)} steps above {cut:.0f} us "
                    f"(median {med:.0f}, MAD {mad:.1f})"
                )
            sel = [r for r, o in zip(sel, out) if not o]
        data.append((arm, d.name, sel, nreq, rows))
    if not data:
        print("no launches with stamps")
    cov = a.covariate
    if cov == "auto":
        has = bool(data) and all(
            "dsum" in r for _, _, sel, _, _ in data for r in sel[:50]
        )
        spread = (
            np.std([r.get("dsum", 0) for _, _, sel, _, _ in data for r in sel])
            if has
            else 0
        )
        cov = ("dmax" if topo == "tp8" else "dsum") if has and spread > 0.5 else "none"
    # within-launch regression of the period on the covariate (+ position)
    beta = None
    if cov != "none":
        xs, ys = [], []
        for _, _, sel, _, _ in data:
            x = np.array([[r[cov], r["pos"]] for r in sel], float)
            y = np.array([r["period"] for r in sel], float)
            xs.append(x - x.mean(0))
            ys.append(y - y.mean())
        X, Y = np.concatenate(xs), np.concatenate(ys)
        beta, *_ = np.linalg.lstsq(X, Y, rcond=None)
        resid = Y - X @ beta
        r2 = 1 - resid.var() / Y.var()
        xbar = np.concatenate(
            [
                np.array([[r[cov], r["pos"]] for r in sel], float)
                for _, _, sel, _, _ in data
            ]
        ).mean(0)
    print(
        f"# {root}  topology {topo}  fixed routing {meta.get('fixed')!r}  order {' '.join(meta['order'])}"
        f"  runtime optimization {meta.get('runtime_opt', 'off (not recorded)')}"
    )
    print(
        f"# steps kept: pass >= 1, step >= {a.skip_head} from a request's start, >= {a.skip_tail} from its end"
        + (f", no {a.page}-token page crossing" if a.page > 0 else "")
    )
    if beta is not None:
        print(
            f"# covariate {cov}: {beta[0]:.3f} us per distinct expert, {beta[1] * 1e3:.3f} us per 1000 positions "
            f"(within-launch R^2 {r2:.3f})"
        )
    print(
        f"# per request: {a.stat} of its step periods; a launch: the mean over its requests"
    )
    print(
        f"{'launch':<16}{'steps':>7}{'reqs':>6}{'mean':>10}{'adj':>10}{'sd adj':>8}{'p50':>9}"
        f"{'verify':>9}{'draft':>9}{'dsum':>7}{'dmax':>7}{'LAUNCH':>10}"
    )
    launch_means = {arm: [] for arm in arms}
    pass_means = {arm: [] for arm in arms}
    step_vals = {arm: [] for arm in arms}
    req_stats = {}
    summary = []
    for arm, name, sel, nreq, rows in data:
        y = np.array([r["period"] for r in sel], float)
        if beta is not None:
            x = np.array([[r[cov], r["pos"]] for r in sel], float)
            yadj = y - (x - xbar) @ beta
        else:
            yadj = y
        sv = np.array([r.get("seg_verify", np.nan) for r in sel], float)
        sd_ = np.array([r.get("seg_draft", np.nan) for r in sel], float)
        ds = np.array([r.get("dsum", np.nan) for r in sel], float)
        dm = np.array([r.get("dmax", np.nan) for r in sel], float)
        line = (
            f"{name:<16}{len(y):>7}{nreq:>6}{y.mean():>10.2f}{yadj.mean():>10.2f}{yadj.std():>8.2f}"
            f"{np.median(y):>9.1f}{np.nanmean(sv):>9.1f}{np.nanmean(sd_):>9.1f}"
            f"{np.nanmean(ds):>7.1f}{np.nanmean(dm):>7.1f}"
        )
        # per request: the median (default) or mean of its adjusted step periods; a
        # launch / pass is the mean over its requests
        reqs = np.array([r["req"] for r in sel])
        passes_of = {r["req"]: r["pass"] for r in sel}
        stat = np.median if a.stat == "median" else np.mean
        per_req = {q: float(stat(yadj[reqs == q])) for q in sorted(set(reqs.tolist()))}
        # slow episodes: whole requests tens of us slower (host interference on the node);
        # dropped relative to the launch's own median request, so an arm's effect (on
        # every request) is kept
        mid = float(np.median(list(per_req.values())))
        slow = {q: v for q, v in per_req.items() if abs(v - mid) > a.req_cut}
        if slow:
            print(
                f"{name}: dropped {len(slow)} of {len(per_req)} requests more than {a.req_cut:.0f} us "
                f"from the launch's median request ({mid:.1f}): {[round(v - mid) for v in slow.values()]}"
            )
            per_req = {q: v for q, v in per_req.items() if q not in slow}
        launch_means[arm].append(float(np.mean(list(per_req.values()))))
        step_vals[arm].append(yadj)
        for p in sorted({passes_of[q] for q in per_req}):
            pass_means[arm].append(
                float(np.mean([v for q, v in per_req.items() if passes_of[q] == p]))
            )
        req_stats[name] = per_req
        print(line + f"{launch_means[arm][-1]:>10.2f}")
        summary.append(
            {
                "launch": name,
                "arm": arm,
                "steps": int(len(y)),
                "period_us": float(y.mean()),
                "launch_stat_us": launch_means[arm][-1],
                "adjusted_us": float(yadj.mean()),
                "sd_us": float(y.std()),
                "seg_verify_us": float(np.nanmean(sv)),
                "seg_draft_us": float(np.nanmean(sd_)),
                "dsum": float(np.nanmean(ds)),
                "dmax": float(np.nanmean(dm)),
            }
        )
    A, B = arms
    print(f"\n# {B} - {A} (us per step; 95 % interval = t(dof) x SE)")
    res = {}
    for unit, vals in (("launch", launch_means), ("pass", pass_means)):
        if min(len(vals[A]), len(vals[B])) < 2:
            print(f"{unit:>7}: needs >= 2 {unit}es per arm")
            continue
        diff, se, dof, hw = _welch(vals[A], vals[B])
        res[unit] = {"diff": diff, "se": se, "dof": dof, "ci95": hw}
        print(
            f"{unit:>7}: {diff:+8.2f}  SE {se:6.2f}  dof {dof:5.1f}  95% [{diff - hw:+.2f}, {diff + hw:+.2f}]"
            f"   ({len(vals[A])} vs {len(vals[B])} {unit}es"
            + (")" if unit == "launch" else "; ignores launch effects, not a result)")
        )
    # variance components of the pass means: within a launch vs between launches
    within, between = [], []
    for arm, name, sel, _, _ in data:
        by_pass = {}
        for r in sel:
            by_pass.setdefault(r["pass"], set()).add(r["req"])
        pm = [
            np.mean([req_stats[name][q] for q in qs if q in req_stats[name]])
            for qs in by_pass.values()
            if any(q in req_stats[name] for q in qs)
        ]
        if len(pm) > 1:
            within.append(np.var(pm, ddof=1))
    for arm in arms:
        if len(launch_means[arm]) > 1:
            between.append(np.var(launch_means[arm], ddof=1))
    if within and between:
        print(
            f"# sd of a pass mean within a launch {np.sqrt(np.mean(within)):.2f} us; "
            f"sd of a launch mean within an arm {np.sqrt(np.mean(between)):.2f} us"
        )
    # the client's view (static_table.py step p50 per pass), for comparison
    static = {arm: [] for arm in arms}
    for d in launches:
        arm = d.name.split("-", 1)[1]
        f = d / "static" / "summary.jsonl"
        if f.exists():
            vals = [
                json.loads(x)["step_p50_ms"] * 1e3
                for x in f.read_text().splitlines()[1:]
                if x.strip()
            ]
            if vals:
                static[arm].append(float(np.mean(vals)))
    if min(len(static[A]), len(static[B])) >= 2:
        diff, se, dof, hw = _welch(static[A], static[B])
        res["static_p50"] = {"diff": diff, "se": se, "dof": dof, "ci95": hw}
        print(
            f" static: {diff:+8.2f}  SE {se:6.2f}  dof {dof:5.1f}  95% [{diff - hw:+.2f}, {diff + hw:+.2f}]"
            f"   (client step p50, launch means: {A} {np.round(static[A], 1).tolist()} {B} {np.round(static[B], 1).tolist()})"
        )
    if not step_vals[A] or not step_vals[B]:
        return
    ya, yb = np.concatenate(step_vals[A]), np.concatenate(step_vals[B])
    diff = yb.mean() - ya.mean()
    se = np.sqrt(ya.var(ddof=1) / len(ya) + yb.var(ddof=1) / len(yb))
    res["step_naive"] = {"diff": float(diff), "se": float(se)}
    print(
        f"  steps: {diff:+8.2f}  SE {se:6.2f}  (every step independent: ignores launch effects)"
    )
    # variance components of the pass means: within-launch vs between-launch
    print(
        json.dumps({"summary": summary, "result": res, "covariate": cov}),
        file=open(root / "analysis.json", "w"),
    )
    if a.crossings:
        _crossings(data)


def _crossings(data) -> None:
    import numpy as np

    print(
        "\n# page crossings: mean period excess of steps whose positions cross a multiple of P"
    )
    rows = [
        r
        for _, _, _, _, rs in data
        for r in rs
        if r["pass"] >= 1 and r["k"] >= 8 and r["k_from_end"] >= 2
    ]
    y = np.array([r["period"] for r in rows])
    for page in (4, 64, 128, 256, 512):
        c = np.array([r["pos"] // page != (r["pos"] + r["adv"]) // page for r in rows])
        if c.any() and (~c).any():
            print(
                f"  P={page:>4}: {c.mean():6.1%} of steps, excess {y[c].mean() - y[~c].mean():+7.2f} us"
            )


# ----------------------------------------------------------------------------- table


def cmd_table(a) -> None:
    """A fixed-routing table from one recorded step (raw ids, rank 0)."""
    import numpy as np

    d = Path(a.launch)
    layout = json.loads((d / "stamps" / "layout-r0.json").read_text())
    raw = np.fromfile(d / "stamps" / "ids-r0.bin", dtype=np.int16).reshape(
        -1, 4 + IDS_STRIDE
    )
    seq = raw[:, :4].copy().view(np.int64).ravel()
    ev, _, _ = _load_launch(d)
    by_seq = {
        int(s) - 1: (int(m) & 0xFF, int(m) >> 16) for m, s in zip(ev[:, 1], ev[:, 3])
    }
    picks = {}
    for kind_name, kind in (("target", VERIFY), ("draft", DRAFT)):
        cands = []
        for i, s in enumerate(seq.tolist()):
            k, sl = by_seq.get(s, (None, None))
            if k != kind or sl >= len(layout) or not layout[sl]["captured"]:
                continue
            ent = layout[sl]["entries"]
            off, layers, dist = 0, {}, 0
            for e in ent:
                ids = (
                    raw[i, 4 + off : 4 + off + e["n"]]
                    .astype(np.int64)
                    .reshape(-1, e["k"])
                )
                off += e["n"]
                layers[e["layer"]] = ids
                dist += len(np.unique(ids[ids >= 0]))
            if layers:
                cands.append((dist, i, layers))
        if not cands:
            continue
        cands.sort(key=lambda c: c[0])
        q = {"median": 0.5, "p10": 0.1, "p90": 0.9}.get(a.step, 0.5)
        dist, i, layers = cands[int(q * (len(cands) - 1))]
        print(
            f"{kind_name}: {len(cands)} recorded steps, picked one with {dist} distinct experts over "
            f"{len(layers)} layers ({a.step}); range {cands[0][0]}-{cands[-1][0]}"
        )
        for layer, ids in layers.items():
            picks[f"{kind_name}_{layer}"] = ids
    np.savez(a.out, **picks)
    print(f"wrote {a.out}: {len(picks)} layers")


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser(
        "run", help="launch the arms in palindrome order and collect stamps"
    )
    r.add_argument("--job", required=True)
    r.add_argument("--topology", choices=("tp4", "tp8"), default="tp4")
    r.add_argument(
        "--node",
        default="0",
        help="node index (tp4) or two (tp8: 0,1) in the allocation",
    )
    r.add_argument("--image", required=True)
    r.add_argument("--model", required=True)
    r.add_argument(
        "--src", default=str(HERE.parent), help="plugin checkout (cluster path)"
    )
    r.add_argument("--out", required=True)
    r.add_argument("--cache", default="", help="host directory mounted at /root/.cache")
    r.add_argument("--mount", action="append", default=[])
    r.add_argument(
        "--arm",
        nargs=2,
        action="append",
        metavar=("NAME", "ENV"),
        required=True,
        help="arm name and its 'K=V K=V' (AB_SLOTS_ADD=..., AB_SLOTS=..., any variable)",
    )
    r.add_argument("--common-env", default="", help="'K=V ...' for both arms")
    r.add_argument(
        "--fixed",
        default="synth",
        help="fixed routing for both arms: synth[:...], table:<npz>, off",
    )
    r.add_argument(
        "--runtime-opt",
        choices=("on", "off"),
        default="on",
        help="SGLANG_ENABLE_NV_MEGA_RUNTIME_OPTIMIZATION=1 for both arms (default on)",
    )
    r.add_argument(
        "--ids", action="store_true", help="also record raw expert ids (rank 0)"
    )
    r.add_argument(
        "--no-stamps", action="store_true", help="no step stamps (overhead check)"
    )
    r.add_argument("--launches", type=int, default=2, help="launches per arm")
    r.add_argument("--passes", type=int, default=4, help="measured passes per launch")
    r.add_argument("--isl", type=int, default=8192, help="static_table input length")
    r.add_argument("--osl", type=int, default=1024, help="static_table output length")
    r.add_argument("--resume", action="store_true")
    r.add_argument("--keep-going", action="store_true")
    r.set_defaults(func=cmd_run)
    z = sub.add_parser("analyze", help="B - A with standard errors")
    z.add_argument("dir")
    z.add_argument(
        "--covariate", choices=("auto", "none", "dsum", "dmax"), default="auto"
    )
    z.add_argument("--skip-head", type=int, default=8)
    z.add_argument("--skip-tail", type=int, default=2)
    z.add_argument(
        "--page",
        type=int,
        default=0,
        help="drop steps that cross a multiple of this position",
    )
    z.add_argument(
        "--crossings",
        action="store_true",
        help="print the period excess at page crossings",
    )
    z.add_argument(
        "--mad",
        type=float,
        default=8.0,
        help="drop steps above median + this many robust sd (MAD x 1.4826)",
    )
    z.add_argument(
        "--floor",
        type=float,
        default=60.0,
        help="... but never closer than this to the median (us)",
    )
    z.add_argument(
        "--stat",
        choices=("median", "mean"),
        default="median",
        help="per-request statistic",
    )
    z.add_argument(
        "--req-cut",
        type=float,
        default=25.0,
        help="drop requests whose statistic is further than this from the launch's median request (us)",
    )
    z.set_defaults(func=cmd_analyze)
    t = sub.add_parser(
        "table", help="a fixed-routing table from a launch recorded with --ids"
    )
    t.add_argument("launch")
    t.add_argument("--out", required=True)
    t.add_argument("--step", choices=("median", "p10", "p90"), default="median")
    t.set_defaults(func=cmd_table)
    a = ap.parse_args()
    a.func(a)


if __name__ == "__main__":
    sys.exit(main())
