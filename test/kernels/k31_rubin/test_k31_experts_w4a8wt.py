# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""K3.1 TP16 W4A8 whole-tile routed experts (``K31MK_EXPERTS_W4A8WT``, default off): the W4A4 engine's
structure with the shipped W4A8 numerics (MXFP8 x plan of the FP32 latent, packed E2M1 weights on K64
``kind::mxf8f6f4`` MMAs, the shipped SiTU / MXFP8 hidden / hidden images). VR200 only (packed K64 MMAs).

Checks (stated before any run; never loosened):
  T1   W8 rows vs the FP64 reference (``_ref_moe16.expert_rows``, logical weights): Frobenius rel-L2 <= 1e-2, every row
       <= 3e-2, every cell finite; outputs NaN-poisoned before every call (2 eager + 2 CUDA-graph replays with new
       inputs), B1 / 2 / 4 / 8, random and distinct routes.
  ACC  over each case's calls: W8's mean Frobenius rel-L2 vs FP64 <= 1.02 x the shipped engine's on the same inputs, and
       W8's worst row <= 1.10 x the shipped engine's worst row (a K64 / whole-tile reorder must not be less accurate).
  K1   (info, recorded only) share of BF16 cells bitwise equal to the shipped engine and the max ordered-ulp distance.

Mutations (fixed harness; each must make a base-passing test fail; the bug each stands for):
  w8_sid    the second K64 MMA of a 128-K block reads scale bytes 0-1 instead of 2-3          [T1, ACC]
  w8_boff   FC1 x advances 32 B per K64 (MXFP4 stride) instead of 64 B                        [T1, ACC]
  w8_hatom  FC2 reads hidden K 128-191 from atom 0's second half instead of atom 1            [T1, ACC]
  w8_xswz   the x descriptor drops the 128B swizzle                                           [T1, ACC]
  served form (MUTATIONS below): w8i_k16, w8i_strong, w8i_rearm, w8i_meta, w8i_metawait, w8i_queue,
  w8i_xexit [served_ptx]; w8i_grid, w8i_dispatch [served_engaged]; w8i_failopen, w8i_guard [served_failclosed].

Timing (``K31_TIME=1``, one GPU): B8 cold weights, arms a16 (shipped, every SM), a16g / p16g
(packed K64) at the bundle's grid SM count, w8 at 52 / 36 / 16 SMs left free and every SM; order A..Z Z..A A..Z Z..A.
"""

import os
import statistics

import pytest
import torch

import _harness as H
import _ref_moe as R
import _ref_moe16 as M
import test_k31_experts_tp16 as T16

W8F = "kimi_k31/k3/cutedsl/mkv2/experts_w4a8.py"
W4F = "kimi_k31/k3/cutedsl/mkv2/experts_w4a4.py"
ERF = "kimi_k31/k3/cutedsl/moe/experts_runtime.py"
MUTATIONS = {
    "w8_sid": (W8F, "    return 142738048 | 1 << 31 | sid << 29 | sid << 4",
               "    return 142738048 | 1 << 31 | (sid & 1) << 29 | sid << 4"),
    "w8_boff": (W8F, "(j // 2) * 64 + (j % 2) * 4, 4 * (j // 2)", "(j // 2) * 64 + (j % 2) * 2, 4 * (j // 2)"),
    "w8_hatom": (W8F, "        steps += [(1024 + 2, 64, 4, 4, 0)]", "        steps += [(1024 + 2, 8, 4, 4, 0)]"),
    "w8_xswz": (W8F, "        x0, h0 = (smem_desc(sm.iterator + o, True) for o in (self.off_x, self.off_h))",
                "        x0, h0 = smem_desc(sm.iterator + self.off_x, False), smem_desc(sm.iterator + self.off_h, True)"),
    # Served form (K31MK_EXP_W8): each stands for a plausible integration bug; the named test must fail.
    # K-16 consumer fence dropped                                                                    [served_ptx]
    "w8i_k16": (W8F, "        if cutlass.const_expr(self.k16):\n            PLN.fence_proxy_async_global()",
                "        if cutlass.const_expr(False):\n            PLN.fence_proxy_async_global()"),
    # the served path compiles W8 without the Run form's K-17 strong hand-off                       [served_ptx]
    "w8i_strong": (ERF, '("strong", self.strong), ("hfull_bar"', '("strong", False), ("hfull_bar"'),
    # the strong hidden re-arm falls back to the weak form                                          [served_ptx]
    "w8i_rearm": (W8F, "                store_words(other + i * 16, (Uint32(4294967295),) * 4, scope=EX.handoff_scope(self))",
                  "                store_words(other + i * 16, (Uint32(4294967295),) * 4)"),
    # grid_for(8) does not report the W8 grid (the MK front would serialize the shared expert)       [served_engaged]
    "w8i_grid": (ERF, "        if self.w8 and batch == 8:\n            return self.w8_grid",
                 "        if False:\n            return self.w8_grid"),
    # the served B8 call silently keeps the shipped engine                                           [served_engaged]
    "w8i_dispatch": (ERF, "            return self.prepare_w8(batch, buffers, out, planned)",
                     "            pass"),
    # Memory-model audit findings: each drops one fix; the named test must fail.
    # A: the MMA warp's meta join is a bar.warp.sync again (K-10b)                                    [served_ptx]
    "w8i_meta": (W8F, "                named_barrier(12, 32)  # the 32 lanes' metadata reads happen before lane 0's release (PTX 8.9.4)",
                 "                cute.arch.sync_warp()"),
    # A: the producer rewrites meta[slot] without waiting on the meta barrier                        [served_ptx]
    "w8i_metawait": (W4F, """            if cutlass.const_expr(self.meta_bar):  # R4: every MMA-warp lane's read of meta[slot] (use - 3) is done""",
                     """            if cutlass.const_expr(False):  # R4: every MMA-warp lane's read of meta[slot] (use - 3) is done"""),
    # B: queue-empty counts 1 again (one lane's arrive completes the phase: the S-2 class)              [served_ptx]
    "w8i_queue": (W4F, "                cute.arch.mbarrier_init(queue.iterator + i, 32 if (i >= 8 and self.queue_bar) else 1)",
                  "                cute.arch.mbarrier_init(queue.iterator + i, 1)"),
    # C: a CTA with no FC1 op exits without waiting on its x TMA                                      [served_ptx]
    "w8i_xexit": (W8F, "                if cutlass.const_expr(self.xexit):  # C:",
                  "                if cutlass.const_expr(False):  # C:"),
    # E: K31MK_EXP_W8 compiles W8 with the weak hand-off / K-10b join when the env lacks them           [served_failclosed]
    "w8i_failopen": (ERF, "        if self.w8 and (self.strong is not True or self.hfull_bar is not True):",
                     "        if False:"),
    # E: the hfull join compiles outside the served form (the guard bypass V1 noted)                  [served_failclosed]
    "w8i_guard": (W8F, "        if hfull_bar and not served:", "        if False:"),
}

needs_sm107 = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() != (10, 7),
    reason="W4A8 whole-tile experts need VR200 (sm_107)",
)

ACC_MEAN, ACC_ROW = 1.02, 1.10
# W4A8WTExperts.prepare's knobs on VR200 (the standalone form) and what prepare_w8 adds (the served form)
STANDALONE_KNOBS = (("ablate", ()), ("slots", 3), ("xone", True), ("deadline_ns", 0))
SERVED_EXTRA = {"strong": True, "hfull_bar": True, "k16": True, "situ_lean": True, "served": True}


def w8_engine(prepared, batch, grid=None):
    from sgl_nvidia_mega.kimi_k31.k3.cutedsl.mkv2 import experts_w4a8 as W8
    from sgl_nvidia_mega.kimi_k31.k3.cutedsl.moe import planned as PLN

    saved = os.environ.get("K31MK_EXPERTS_W4A8WT_GRID")
    if grid is not None:
        os.environ["K31MK_EXPERTS_W4A8WT_GRID"] = str(grid)
    try:
        ex = W8.W4A8WTExperts(*prepared, beta=R.BETA, linear_beta=R.LIN, activation_floor=None)
    finally:
        if saved is None:
            os.environ.pop("K31MK_EXPERTS_W4A8WT_GRID", None)
        else:
            os.environ["K31MK_EXPERTS_W4A8WT_GRID"] = saved
    out = torch.empty((batch * 16, R.H_LAT), dtype=torch.bfloat16, device="cuda")
    ex.prepare(batch, PLN.scratch(16, torch.device("cuda")), out)
    assert ex.prepared[batch][3] is not None
    return ex, out


def ordered(bits):
    i = bits.view(torch.int16).to(torch.int32)
    return torch.where(i < 0, -32768 - i, i)


def stats(rows, ref):
    fro = R.rel_l2(rows, ref)
    per_row = ((rows.double() - ref.double()).norm(dim=-1) / ref.double().norm(dim=-1)).flatten()
    return fro, float(per_row.max())


@needs_sm107
@pytest.mark.parametrize("case", ("random", "distinct"))
@pytest.mark.parametrize("batch", (1, 2, 4, 8), ids=lambda b: f"b{b}")
def test_w4a8wt(batch, case):
    """T1 + ACC (+ K1 info) against the shipped engine on the same x, ids."""
    H.load_plugin(MUTATIONS)
    logical, prepared = T16.weights16()
    w8, out_w = w8_engine(prepared, batch)
    ship, out_s = T16.persistent(prepared, batch)
    g = H.generator(1300 + 10 * batch + (case == "distinct"))
    x = torch.empty((batch, R.H_LAT), device="cuda")
    ids = torch.empty((batch, 16), dtype=torch.int32, device="cuda")

    def call():
        w8.run(x, ids)
        ship.run(x, ids)

    graph, fros, rows_max = None, {"w8": [], "ship": []}, {"w8": 0.0, "ship": 0.0}
    for n in range(4):
        xv = T16.latent(batch, g)
        i, _ = R.random_route(batch, g, distinct=case == "distinct")
        x.copy_(xv)
        ids.copy_(i)
        H.poison(out_w, out_s)
        if n < 2:
            call()
        else:
            graph = graph or T16.graph_of(call)
            H.poison(out_w, out_s)
            graph.replay()
        torch.cuda.synchronize()
        ref = M.expert_rows(logical, xv, i)
        T16.check_rows(f"w8.b{batch}.{case}.call{n}", out_w.view(batch, 16, -1), ref)
        for k, o in (("w8", out_w), ("ship", out_s)):
            f, r = stats(o.view(batch, 16, -1), ref)
            fros[k].append(f)
            rows_max[k] = max(rows_max[k], r)
        same = float((out_w.view(torch.int16) == out_s.view(torch.int16)).float().mean())
        ulp = int((ordered(out_w) - ordered(out_s)).abs().max())
        H.record({"kind": "info", "item": f"w8.b{batch}.{case}.call{n}.k1", "bitwise_frac": same, "max_ulp": ulp})
    mean_w, mean_s = statistics.mean(fros["w8"]), statistics.mean(fros["ship"])
    rec = {"kind": "check", "item": f"w8.b{batch}.{case}.acc", "mean_fro_w8": mean_w, "mean_fro_ship": mean_s,
           "row_max_w8": rows_max["w8"], "row_max_ship": rows_max["ship"], "grid": w8.grid_for(batch)}
    H.record(rec)
    print("CHECK", rec, flush=True)
    assert mean_w <= ACC_MEAN * mean_s and rows_max["w8"] <= ACC_ROW * rows_max["ship"], rec


# ------------------------------------------------------------------------------ timing (Rubin, one GPU)
def timing_arm(name, prepared, batch):
    from sgl_nvidia_mega.kimi_k31.k3.cutedsl.moe import experts_runtime as ER

    if name.startswith("w8@"):
        return w8_engine(prepared, batch, int(name[3:]))
    env = {"a16": {}, "a16g": {"K31MK_EXPERTS_GRID_SMS": str(H.bundle_grid_sms())},
           "p16g": {"K31MK_EXPERTS_GRID_SMS": str(H.bundle_grid_sms())}}[name]
    saved = {k: os.environ.get(k) for k in env}
    os.environ.update(env)
    try:
        ex = ER.PersistentExperts(*prepared, beta=R.BETA, linear_beta=R.LIN, activation_floor=None,
                                  packed=6 if name == "p16g" else 0)
        out = torch.empty((batch * 16, R.H_LAT), dtype=torch.bfloat16, device="cuda")
        ex.prepare(batch, ER.scratch(16, torch.device("cuda")), out)
    finally:
        for k, v in saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
    return ex, out


@needs_sm107
def test_time_w4a8wt():
    if os.environ.get("K31_TIME") != "1" or os.environ.get("K31_MUTATION"):
        pytest.skip("timing")
    H.load_plugin(MUTATIONS)
    batch = 8
    sms = torch.cuda.get_device_properties(0).multi_processor_count  # w8 arms: 52 / 36 / 16 SMs left free
    names = os.environ.get("W8_TIME_ARMS", f"a16,a16g,p16g,w8@{sms - 52},w8@{sms - 36},w8@{sms - 16}").split(",")
    logical, prepared = T16.weights16()
    gt = H.generator(int(os.environ.get("W8_TIME_SEED", 1990)))
    size = int(os.environ.get("W8_POOL", "112"))
    pools = torch.randperm(R.E_ALL, generator=gt, device="cuda")[: 4 * size].view(4, size)
    routes = [R.random_route(batch, gt, pool=pools[i % 4]) for i in range(16)]
    xs = [T16.latent(batch, gt) for _ in routes]
    runners = {}
    for name in names:
        ex, out = timing_arm(name, prepared, batch)
        H.poison(out)
        ex.run(xs[0], routes[0][0])
        torch.cuda.synchronize()
        T16.check_rows(f"time.{name}.b{batch}", out.view(batch, 16, -1), M.expert_rows(logical, xs[0], routes[0][0]))
        runners[name] = ex
    distinct = sum(int(torch.unique(r[0]).numel()) for r in routes) / len(routes)
    calls = {k: [lambda ex=ex, x=x, r=r: ex.run(x, r[0]) for x, r in zip(xs, routes)] for k, ex in runners.items()}
    got = {k: [] for k in names}
    for k in names + names[::-1] + names + names[::-1]:
        got[k].append(H.cold_time_us(calls[k])["us_median"])
    for k in names:
        row = {"kind": "time", "item": f"w8.time.b{batch}.{k}", "us_median": statistics.median(got[k]),
               "us_runs": got[k], "us_spread": max(got[k]) - min(got[k]), "distinct_mean": distinct,
               "grid": runners[k].grid_for(batch), **H.device_facts()}
        H.record(row)
        print("TIME", row, flush=True)


# ------------------------------------------------------------------------------ the served form (K31MK_EXP_W8)
def w8_grid():
    """The W8 engine's default grid on this device: every SM but ``W8_FREE_SMS`` (moe/experts_runtime.py)."""
    from sgl_nvidia_mega.kimi_k31.k3.cutedsl.moe import experts_runtime as ER

    return torch.cuda.get_device_properties(0).multi_processor_count - ER.W8_FREE_SMS


# The Run form's experts env with the W8 knob; the served class is ``PersistentExperts`` (providers).
RUN_FORM = {"K31MK_EXPERTS_STRONG_SYNC": "1", "K31MK_EXPERTS_PACKED_K64": "1", "K31MK_EXP_SITU_LEAN": "1",
            "K31MK_EXPERTS_X3_RELEASE": "3", "K31MK_EXPERTS_HFULL_BAR": "1", "K31MK_EXPERTS_PRODUCER_PACE": "240",
            "K31MK_EXP_XPLAN": "1"}  # + K31MK_EXPERTS_GRID_SMS = H.bundle_grid_sms() (added in served())


def served(prepared, batch, w8=True, over=None):
    """PersistentExperts as the providers build it, with RUN_FORM (+ K31MK_EXP_W8=1, + ``over``); scratch from the
    same env."""
    from sgl_nvidia_mega.kimi_k31.k3.cutedsl.moe import experts_runtime as ER

    env = dict(RUN_FORM, K31MK_EXPERTS_GRID_SMS=str(H.bundle_grid_sms()), **({"K31MK_EXP_W8": "1"} if w8 else {}), **(over or {}))
    saved = {k: os.environ.get(k) for k in list(env) + ["K31MK_EXP_W8"]}
    os.environ.pop("K31MK_EXP_W8", None)
    os.environ.update(env)
    try:
        ex = ER.PersistentExperts(*prepared, beta=R.BETA, linear_beta=R.LIN, activation_floor=None)
        out = torch.empty((batch * 16, R.H_LAT), dtype=torch.bfloat16, device="cuda")
        ex.prepare(batch, ER.scratch(16, torch.device("cuda")), out)
        ptx = ex.prepared[batch][3].artifacts.PTX  # read now: a later compile may reuse the KEEP dump file name
        if ptx and len(ptx) < 4096 and os.path.exists(ptx):
            ptx = open(ptx).read()
    finally:
        for k, v in saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
    return ex, out, ptx


def _offset(lines, defs, line):
    """The SMEM offset of an mbarrier operand: ``[%rA+N]`` -> N; ``[%rA]`` -> N of ``add.s32 %rA, %rB, N``."""
    inner = line.split("[", 1)[1].split("]", 1)[0].replace(" ", "")
    if "+" in inner:
        return int(inner.split("+", 1)[1])
    return defs.get(inner)


def _bar_before(lines, i):
    for j in range(i - 1, -1, -1):
        t = lines[j].replace(" ", "").replace("\t", "")
        if t.startswith("bar.") or t.startswith("barrier."):
            return t
    return None


def sync_census(ptx, k):
    """The served kernel's forms of edges A / B / C, from the PTX text. ``k`` = a
    W4A8ExpertsKernel with the same layout (offsets only). A: no lane-0 generic arrive on empty[]; every producer
    try_wait on empty[slot] is paired with one on the meta barrier; exactly one meta arrive and its nearest preceding
    barrier is ``bar.sync 12,32``. B: every queue-empty arrive's nearest preceding barrier is ``bar.sync 14,32``.
    C: try_wait lines on xready[0] (counted; the served minus the standalone count = the end-marker site)."""
    lines = [ln.strip() for ln in (ptx or "").splitlines()]
    defs = {}
    for ln in lines:
        p = ln.replace(",", " ").replace(";", " ").split()
        if len(p) == 4 and p[0] == "add.s32" and p[3].isdigit():
            defs[p[1]] = int(p[3])
    misc = k.off_misc
    empty, meta = misc + k.bar_empty, misc + 8 * (2 * k.stages + 50)
    qempty, xready = misc + k.bar_queue + 64, misc + k.bar_xready
    waits, arrives = {}, []
    for i, ln in enumerate(lines):
        if "mbarrier.try_wait" in ln:
            off = _offset(lines, defs, ln)
            waits[off] = waits.get(off, 0) + 1
        elif ln.startswith("mbarrier.arrive") and "expect_tx" not in ln and "[" in ln:
            arrives.append((_offset(lines, defs, ln), _bar_before(lines, i)))
    in_ = lambda o, base, n: o is not None and base <= o < base + 8 * n  # noqa: E731
    movs = {}
    for ln in lines:
        p = ln.replace(",", " ").replace(";", " ").split()
        if len(p) == 3 and p[0] == "mov.b32" and p[2].isdigit():
            movs[p[1]] = int(p[2])
    qinit = []
    for ln in lines:
        if ln.startswith("mbarrier.init") and "[" in ln:
            off = _offset(lines, defs, ln)
            if in_(off, qempty, 8):
                qinit.append(movs.get(ln.replace(";", "").split(",")[-1].strip()))
    return {
        "empty_waits": sum(v for o, v in waits.items() if in_(o, empty, k.stages)),
        "meta_waits": sum(v for o, v in waits.items() if in_(o, meta, k.stages)),
        "empty_generic_arrives": sum(in_(o, empty, k.stages) for o, _ in arrives),
        "meta_arrives": [b for o, b in arrives if in_(o, meta, k.stages)],
        "queue_arrives": [b for o, b in arrives if in_(o, qempty, 8)],
        "xready_waits": waits.get(xready, 0),
        "queue_empty_init_counts": sorted(set(qinit)), "queue_empty_inits": len(qinit),
        "bar_ids": sorted({ln.replace(" ", "").replace("\t", "")[8:].split(",")[0].rstrip(";") for ln in lines
                           if ln.replace(" ", "").replace("\t", "").startswith("bar.sync")}),
    }


def census(ptx):
    lines = [ln.strip() for ln in (ptx or "").splitlines()]
    wait = next(i for i, ln in enumerate(lines) if ln.startswith("griddepcontrol.wait"))
    xarm = [i for i, ln in enumerate(lines) if "mbarrier.arrive.expect_tx" in ln]
    fences = [i for i, ln in enumerate(lines) if ln.startswith("fence.proxy.async.global")]
    return {"relaxed_gpu_st": sum(ln.startswith("st.relaxed.gpu.global") for ln in lines),
            "u16_polls": sum(ln.startswith("ld.volatile.global.u16") for ln in lines),
            "hfull_join": sum(ln.replace(" ", "").replace("\t", "") == "bar.sync13,32;" for ln in lines),
            # K-16: a fence.proxy.async.global after the dependency wait and before the first expect_tx (= x's)
            "k16": bool(fences and xarm and wait < fences[0] < xarm[0]), "fences": len(fences)}


@needs_sm107
def test_served_ptx():
    """F1 lesson: the PTX of the kernel the served path compiles (CUTE_DSL_KEEP=ptx) carries the Run-form fixes.
    Expected forms from the shipped engine's K-17 / hfull-bar / K-16 sites, not from W8:
    hidden data v4 + scale u16 stores and the v4 re-arm as st.relaxed.gpu (>= 7: 1 data, 1 scale, 5 re-arm loop
    stores of this grid's unroll), 16-bit scale polls (2), one bar.sync 13,32, the K-16 fence between the wait and
    x's expect_tx. Bug caught: any of the five mutants (the served kernel silently keeping a weak or unfenced form)."""
    H.load_plugin(MUTATIONS)
    if os.environ.get("CUTE_DSL_KEEP") != "ptx":
        raise AssertionError("run with CUTE_DSL_KEEP=ptx (the census reads the served kernel's PTX)")
    from sgl_nvidia_mega.kimi_k31.k3.cutedsl.mkv2 import experts_w4a8 as W8

    _, prepared = T16.weights16()
    ex, _, ptx = served(prepared, 8)
    c = census(ptx)
    H.record({"kind": "check", "item": "w8i.served_ptx", **c})
    print("CHECK served_ptx", c, flush=True)
    assert c["relaxed_gpu_st"] >= 7 and c["u16_polls"] == 2 and c["hfull_join"] == 1 and c["k16"], c
    # Edges A / B / C. Expected forms from the shipped engine's x3 meta R4 site (A),
    # the K-10d H1 shape (B) and ExpertsKernel.consume's xtma exit wait (C); the control is the standalone W8 compiled
    # fresh in this process (same FC1 / producer code, none of the fixes). Bugs caught: w8i_meta, w8i_metawait,
    # w8i_queue, w8i_xexit (each drops one fix); the pre-fix tree fails every A / B / C line.
    knobs = dict(STANDALONE_KNOBS)
    lay = W8.W4A8ExpertsKernel(16, 8, ex.experts, w8_grid(), False, ex.activation_floor, **dict(knobs, **SERVED_EXTRA))
    code = W8.compile_w4a8.__wrapped__(16, ex.experts, 8, w8_grid(), torch.cuda.current_device(), ex.activation_floor,
                                       tuple(knobs.items()))[0]
    base = code.artifacts.PTX  # read now (a later compile may reuse the KEEP dump file name)
    if base and len(base) < 4096 and os.path.exists(base):
        base = open(base).read()
    s, b = sync_census(ptx, lay), sync_census(base, lay)
    rec = {"served": s, "standalone": b, "standalone_hfull_join": census(base)["hfull_join"]}
    H.record({"kind": "check", "item": "w8i.sync_census", **rec})
    print("CHECK sync_census", rec, flush=True)
    assert rec["standalone_hfull_join"] == 0 and b["meta_waits"] == 0, rec  # the control is the unfixed engine
    # A: R4 (count-1 meta barrier after bar.sync 12,32; empty[] = the commit only)
    assert s["empty_generic_arrives"] == 0 and s["meta_arrives"] == ["bar.sync12,32;"], rec
    assert s["meta_waits"] == s["empty_waits"] >= 2, rec
    # B: every loader lane releases its read: the 8 queue-empty barriers count 32 (standalone: 1) and no join is needed
    assert s["queue_empty_inits"] == 8 and s["queue_empty_init_counts"] == [32], rec
    assert b["queue_empty_init_counts"] == [1], rec
    # C: one more xready[0] wait site (2 lines: try + retry) than the standalone = the end-marker wait
    assert s["xready_waits"] == b["xready_waits"] + 2, rec
    # named-barrier IDs: 0 (CTA), 1 (routing / epilogue group 0), %r = 1 + 3 eg, 12 meta, 13 hfull
    assert set(s["bar_ids"]) - {"0", "1", "12", "13"} <= {x for x in s["bar_ids"] if x.startswith("%")}, rec


@needs_sm107
def test_served_engaged():
    """The served B8 call runs the W8 engine at the W8 grid and the front sees that grid; B4 keeps the Run form's grid.
    Bug caught: w8i_dispatch (shipped engine under the knob), w8i_grid (front told the bundle's grid while W8 runs its own)."""
    from sgl_nvidia_mega.kimi_k31.k3.cutedsl.mkv2 import experts_w4a8 as W8

    H.load_plugin(MUTATIONS)
    _, prepared = T16.weights16()
    ex, _, _ = served(prepared, 8)
    rd, _, _ = served(prepared, 8, w8=False)
    knobs = (("ablate", ()), ("slots", 3), ("xone", True), ("deadline_ns", 0),
             ("strong", ex.strong), ("hfull_bar", bool(ex.hfull_bar)), ("k16", True), ("situ_lean", bool(ex.situ_lean)),
             ("served", True))
    want = W8.compile_w4a8(16, ex.experts, 8, w8_grid(), torch.cuda.current_device(), ex.activation_floor, knobs)[0]
    rec = {"w8": ex.w8, "grid8": ex.grid_for(8), "code_is_w8": ex.prepared[8][3] is want, "grid4": ex.grid_for(4), "grid4_runform": rd.grid_for(4)}
    H.record({"kind": "check", "item": "w8i.engaged", **rec})
    print("CHECK engaged", rec, flush=True)
    assert rec["w8"] and rec["grid8"] == w8_grid() and rec["code_is_w8"] and rec["grid4"] == rec["grid4_runform"], rec


@needs_sm107
def test_served_failclosed():
    """E: under K31MK_EXP_W8 a Run env without K31MK_EXPERTS_STRONG_SYNC=1 or without
    K31MK_EXPERTS_HFULL_BAR=1 must raise at PersistentExperts construction with the env message (never compile the K-17 (b)
    / K-10b forms), and the kernel refuses the hfull join outside the served form and the served form without strong /
    hfull. Bugs caught: w8i_failopen (the env guard gone: the error moves to the kernel's check at prepare), w8i_guard."""
    from sgl_nvidia_mega.kimi_k31.k3.cutedsl.mkv2 import experts_w4a8 as W8

    H.load_plugin(MUTATIONS)
    _, prepared = T16.weights16()
    got = {}
    for name, over in (("strong0", {"K31MK_EXPERTS_STRONG_SYNC": "0"}), ("hfull0", {"K31MK_EXPERTS_HFULL_BAR": "0"})):
        try:
            served(prepared, 8, over=over)
            got[name] = "compiled"
        except ValueError as e:
            got[name] = "raised: " + str(e)[:80]
    knobs = dict(STANDALONE_KNOBS)
    for name, extra in (("k_hfull_standalone", {"hfull_bar": True}),
                        ("k_served_weak", dict(SERVED_EXTRA, strong=False)),
                        ("k_served_nohfull", dict(SERVED_EXTRA, hfull_bar=False))):
        try:
            W8.W4A8ExpertsKernel(16, 8, 896, w8_grid(), False, None, **dict(knobs, **extra))
            got[name] = "built"
        except ValueError as e:
            got[name] = "raised: " + str(e)[:80]
    H.record({"kind": "check", "item": "w8i.failclosed", **got})
    print("CHECK failclosed", got, flush=True)
    # the env cases must fail at PersistentExperts construction with the env message (the boot-time contract), not later in
    # the kernel's own served check (w8i_failopen was MISSED when any raise counted)
    assert all(got[k].startswith("raised: K31MK_EXP_W8 needs") for k in ("strong0", "hfull0")), got
    assert all(got[k].startswith("raised: W4A8ExpertsKernel") for k in ("k_hfull_standalone", "k_served_weak", "k_served_nohfull")), got


@needs_sm107
@pytest.mark.parametrize("case", ("random", "distinct"))
def test_served_w8(case):
    """Served W8 (Run form + K31MK_EXP_W8) vs the standalone W8 (bitwise: the integration adds sync forms and the
    bitwise SiTU-lean only), vs FP64 (T1) and vs the served Run form without the knob (ACC lines, the REORDER class),
    B8, 2 eager + 2 graph calls with new inputs. Bug caught: a served-path wiring that changes values (maps, plan,
    workspace, scalars) or loses accuracy."""
    H.load_plugin(MUTATIONS)
    batch = 8
    logical, prepared = T16.weights16()
    sv, out_v, _ = served(prepared, batch)
    w8, out_w = w8_engine(prepared, batch, w8_grid())
    rd, out_r, _ = served(prepared, batch, w8=False)
    g = H.generator(1600 + (case == "distinct"))
    x = torch.empty((batch, R.H_LAT), device="cuda")
    ids = torch.empty((batch, 16), dtype=torch.int32, device="cuda")

    def call():
        sv.run(x, ids)
        w8.run(x, ids)
        rd.run(x, ids)

    graph, fros, rows_max, diff = None, {"v": [], "r": []}, {"v": 0.0, "r": 0.0}, 0
    for n in range(4):
        xv = T16.latent(batch, g)
        i, _ = R.random_route(batch, g, distinct=case == "distinct")
        x.copy_(xv)
        ids.copy_(i)
        H.poison(out_v, out_w, out_r)
        if n < 2:
            call()
        else:
            graph = graph or T16.graph_of(call)
            H.poison(out_v, out_w, out_r)
            graph.replay()
        torch.cuda.synchronize()
        ref = M.expert_rows(logical, xv, i)
        T16.check_rows(f"w8i.b8.{case}.call{n}", out_v.view(batch, 16, -1), ref)
        diff += int((out_v.view(torch.int16) != out_w.view(torch.int16)).sum())
        for k, o in (("v", out_v), ("r", out_r)):
            f, r = stats(o.view(batch, 16, -1), ref)
            fros[k].append(f)
            rows_max[k] = max(rows_max[k], r)
    rec = {"kind": "check", "item": f"w8i.b8.{case}", "cells_vs_standalone": diff,
           "mean_fro_served": statistics.mean(fros["v"]), "mean_fro_runform": statistics.mean(fros["r"]),
           "row_max_served": rows_max["v"], "row_max_runform": rows_max["r"]}
    H.record(rec)
    print("CHECK", rec, flush=True)
    assert diff == 0, rec
    assert rec["mean_fro_served"] <= ACC_MEAN * rec["mean_fro_runform"] and rows_max["v"] <= ACC_ROW * rows_max["r"], rec
