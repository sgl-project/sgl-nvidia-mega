#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Token-split indexer (TSI) microbenchmark, K3.1 shapes, synthetic tensors.

Modes (one JSON line per item on stdout):
  one   1 GPU, per L and rows: DCP1 chain = DeepGEMM logits over L + topk_v2 (deterministic False, row page table), as
        kimi_k31/dsa/verify.py runs it; TSI rank-0 chain = DeepGEMM over rank 0's local pages + topk_v2 raw (stable
        ties) + tsi_merge (W = 16; the 15 peers' records pre-filled for the current tag by their own push halves, so
        the poll returns at once: compute only). Components too (dg, topk with both plan settings, merge). Arms in
        ABBA order on this GPU. Each item: a CUDA graph over NC distinct input copies (identical data at distinct
        addresses, >= 4 x L2 bytes read per graph, capped), replayed --reps times per measurement, median of 5.
  live  torchrun on 4 GPUs: world 16 with ranks 0..3 live (one per GPU) and 4..15 virtual (their candidates computed
        locally at setup and re-tagged per layer by the fill kernel): the per-layer TSI chain with a real NVLink
        exchange among the live ranks = (graph A: F x (prepare + layers x (fill + select)) - graph B: F x (prepare +
        layers x fill)) / (F x layers).
Sanity: the TSI output of copy 0 == the exact top-2048 (torch, full DCP1 logits) in both modes; with --ref-times, DCP1
numbers checked against those reference times (+-15%) at the end: VALID or VOID (NO_REF without the file).
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "../../..")))

from sgl_nvidia_mega.kimi_k31.dsa import tsi as T  # noqa: E402

PAGE, IDXB, TOPK, W = 64, 132, 2048, 16
L2_BYTES = torch.cuda.get_device_properties(0).L2_cache_size if torch.cuda.is_available() else 0  # no CUDA: no timing
# Reference times for the VALID/VOID line (+-15%) come from --ref-times: a JSON file
# {"chain_dcp1": {L: us}, "dg_dcp1": {L: us}, "topk_dcp1": {L: us}} measured on the GPU under test.
REF_ITEMS = ("chain_dcp1", "dg_dcp1", "topk_dcp1")


def load_refs(path):
    """{item: {L: us}} from the --ref-times JSON file; {} without one."""
    if not path:
        return {}
    with open(path) as f:
        raw = json.load(f)
    return {name: {int(L): float(us) for L, us in raw.get(name, {}).items()} for name in REF_ITEMS}


def emit(**kw):
    print(json.dumps(kw), flush=True)


def sms():
    return torch.cuda.get_device_properties(torch.cuda.current_device()).multi_processor_count


def local_len(n, r):
    if n <= 0:
        return 0
    pages, rem = divmod(n, PAGE)
    return (pages + W - 1 - r) // W * PAGE + (rem if pages % W == r else 0)


def hold_fp8_clock():
    a = torch.randn(256, 256, device="cuda").to(torch.float8_e4m3fn)
    one = torch.ones((), device="cuda")
    torch._scaled_mm(a, a.t(), scale_a=one, scale_b=one, out_dtype=torch.bfloat16)
    torch.cuda.synchronize()


def time_graph(fns, reps):
    """One CUDA graph running every closure in order; us per closure, (median, min, max) over 5 measurements."""
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for f in fns:
            f()
    torch.cuda.current_stream().wait_stream(s)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        for f in fns:
            f()
    torch.cuda.synchronize()
    graph.replay()
    torch.cuda.synchronize()
    out = []
    for _ in range(5):
        a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        a.record()
        for _ in range(reps):
            graph.replay()
        b.record()
        torch.cuda.synchronize()
        out.append(a.elapsed_time(b) * 1000.0 / (reps * len(fns)))
    out.sort()
    del graph
    return out[2], out[0], out[-1]


class Request:
    """``rows`` / 8 requests of context L each (disjoint random pages), FP8 index-K pool, query and gates."""

    def __init__(self, L, rows, g):
        self.L, self.rows, self.b = L, rows, rows // 8
        self.pages = -(-L // PAGE)
        pool = self.b * self.pages + 1
        perm = torch.randperm(pool, generator=g, device="cuda").to(torch.int32)
        self.table = perm[: self.b * self.pages].view(self.b, self.pages).contiguous()
        vals = (torch.randn(pool, PAGE * 128, generator=g, device="cuda") * 0.5).to(torch.float8_e4m3fn)
        scales = torch.rand(pool, PAGE, generator=g, device="cuda") * 0.02 + 0.01
        self.cache = torch.cat([vals.view(torch.uint8), scales.view(torch.uint8).view(pool, PAGE * 4)], 1).reshape(-1)
        self.contexts = torch.full((self.b, 8), L, dtype=torch.int32, device="cuda")
        self.causal = (L - 8 + 1 + torch.arange(8, device="cuda")).to(torch.int32).repeat(self.b)
        self.query = torch.randn(rows, 32, 128, generator=g, device="cuda").to(torch.float8_e4m3fn)
        self.gates = torch.rand(rows, 32, generator=g, device="cuda") + 0.05
        self.row_table = self.table.expand(8, -1) if self.b == 1 else self.table.repeat_interleave(8, 0)

    def full_logits(self, cache):
        import deep_gemm
        from sglang.kernels.ops.attention.dsa.paged_mqa_logits import _restore_row_stride

        sched = deep_gemm.get_paged_mqa_logits_metadata(self.contexts, PAGE, sms())
        return _restore_row_stride(deep_gemm.fp8_paged_mqa_logits(
            self.query.view(self.b, 8, 32, 128), cache.view(-1, PAGE, 1, IDXB), self.gates, self.contexts,
            self.table, sched, self.pages * PAGE, clean_logits=False))

    def reference(self, logits):
        ref = torch.full((self.rows, TOPK), -1, dtype=torch.int32, device="cuda")
        for row in range(self.rows):
            n, b = int(self.causal[row]), row // 8
            top = torch.sort(logits[row, :n].float(), descending=True, stable=True).indices[: min(n, TOPK)]
            pos = torch.sort(top).values
            ref[row, : pos.numel()] = (self.table[b, pos // PAGE].long() * PAGE + pos % PAGE).int()
        return ref


def copies(req, nbytes_read):
    return max(2, math.ceil(4 * L2_BYTES / max(nbytes_read, 1)))


def stage_ranks(req, cache0, live_from=1):
    """16 emulated ranks on this GPU (all prepared, epoch 1); ranks >= live_from run their push half on cache0."""
    records = T.region_records(W, 32, T.CAND)
    regions = [torch.zeros(records, dtype=torch.int64, device="cuda") for _ in range(W)]
    comms = [T.TsiComm("cuda", W, r, cap=32, peers=regions) for r in range(W)]
    states = [T.TsiState(c, req.rows, req.pages) for c in comms]
    out = torch.empty(req.rows, TOPK, dtype=torch.int32, device="cuda")
    for st in states:
        st.prepare(req.table, req.contexts, req.causal)
    for st in states[live_from:]:
        st.select(0, req.query, req.gates, cache0, out, phases=T.PUSH)
    torch.cuda.synchronize()
    return comms, states, regions


_REF = {}


def load_ref(path):
    """Another tsi.cuh (e.g. the previous commit's) as its own JIT module: same-GPU A/B of the merge kernel."""
    if path not in _REF:
        import sglang
        from sglang.kernels.jit.utils import load_jit

        _REF[path] = load_jit(
            "k31_tsi_ref", f"w{W}_c{T.CAND}",
            cuda_files=[path], cuda_wrappers=[("merge", f"k31_tsi::Binding<{W}, {T.CAND}>::merge")],
            extra_include_paths=[os.path.join(os.path.dirname(sglang.__file__), "kernels/jit/csrc")],
        )
    return _REF[path]


def run_one(L, rows, reps, g, max_copies, ref_cuh=None, variants=()):
    import deep_gemm
    from sglang.kernels.ops.attention.dsa.paged_mqa_logits import _restore_row_stride
    from sglang.kernels.ops.attention.dsv4.topk import plan_topk_v2, topk_transform_paged_v2, topk_v2_plan_is_written

    req = Request(L, rows, g)
    ll0 = local_len(L, 0)
    nc1 = min(max_copies, copies(req, req.b * L * IDXB))
    nc0 = min(max_copies, copies(req, req.b * ll0 * IDXB))
    caches = [req.cache] + [req.cache.clone() for _ in range(max(nc0, nc1) - 1)]
    emit(kind="setup", L=L, rows=rows, copies_dcp1=nc1, copies_tsi=nc0, local_len0=ll0,
         bytes_read_dcp1=nc1 * req.b * L * IDXB, bytes_read_tsi=nc0 * req.b * ll0 * IDXB)

    # ---- DCP1 (verify.py) ----
    sched1 = deep_gemm.get_paged_mqa_logits_metadata(req.contexts, PAGE, sms())
    plan1 = plan_topk_v2(req.causal)
    sel = torch.empty(rows, TOPK, dtype=torch.int32, device="cuda")

    def dg1(c):
        return _restore_row_stride(deep_gemm.fp8_paged_mqa_logits(
            req.query.view(req.b, 8, 32, 128), caches[c].view(-1, PAGE, 1, IDXB), req.gates, req.contexts,
            req.table, sched1, req.pages * PAGE, clean_logits=False))

    def tk1(s):
        topk_transform_paged_v2(s, req.causal, req.row_table, sel, PAGE, plan1, deterministic=False)

    fixed1 = [dg1(c) for c in range(nc1)]
    torch.cuda.synchronize()
    ref = req.reference(fixed1[0])

    # ---- TSI rank 0 (peers 1..15 pushed their records for epoch 1 / ordinal 0) ----
    comms, states, regions = stage_ranks(req, caches[0])
    st0 = states[0]
    out0 = torch.empty(rows, TOPK, dtype=torch.int32, device="cuda")
    plan_default = plan_topk_v2(st0.lcausal, -1)
    plan_written = {"knob": topk_v2_plan_is_written(st0.lcausal), "rows": rows}

    def dg0(c):
        return _restore_row_stride(deep_gemm.fp8_paged_mqa_logits(
            req.query.view(req.b, 8, 32, 128), caches[c].view(-1, PAGE, 1, IDXB), req.gates, st0.lctx, st0.ltab,
            st0.schedule, st0.width * PAGE, clean_logits=False))

    fixed0 = [dg0(c) for c in range(nc0)]
    torch.cuda.synchronize()

    def tk0(s, plan):
        topk_transform_paged_v2(s, st0.lcausal, None, st0.raw, PAGE, plan, deterministic=True)

    def mg0(s):
        comms[0].merge(0, st0.raw, s, req.causal, req.table, out0, st0.fallback, st0.scratch, 8, st0.end_bit)

    tk0(fixed0[0], st0.plan)
    torch.cuda.synchronize()
    # T1L variants (tsi.cuh K31_TSI_T1L) read the local top-C instead of the top-2048 (same plan, stable ties)
    raw_c = torch.full((rows, T.CAND), -1, dtype=torch.int32, device="cuda")

    def tk0_c(s):
        topk_transform_paged_v2(s, st0.lcausal, None, raw_c, PAGE, st0.plan, deterministic=True)

    tk0_c(fixed0[0])
    torch.cuda.synchronize()

    # sanity: the compute-only chain's output on copy 0 is the exact selection
    out0.fill_(-7)
    st0.select(0, req.query, req.gates, caches[0], out0)
    torch.cuda.synchronize()
    bad = int((out0 != ref).any(dim=1).sum())
    emit(kind="sanity_tsi_exact", L=L, rows=rows, rows_differ=bad, fallback=int(st0.fallback.sum()),
         ok=bad == 0 and int(st0.fallback.sum()) == 0, plan_written=plan_written)

    items = {}

    def measure(name, fns):
        m, lo, hi = time_graph(fns, reps)
        items.setdefault(name, []).append(m)
        emit(kind=name, L=L, rows=rows, copies=len(fns), us=round(m, 3), us_min=round(lo, 3), us_max=round(hi, 3))

    chain1 = [lambda c=c: tk1(dg1(c)) for c in range(nc1)]
    chain0 = [lambda c=c: st0.select(0, req.query, req.gates, caches[c], out0) for c in range(nc0)]
    for name, fns in (("chain_dcp1", chain1), ("chain_tsi", chain0), ("chain_tsi", chain0), ("chain_dcp1", chain1)):
        measure(name, fns)
    measure("dg_dcp1", [lambda c=c: dg1(c) for c in range(nc1)])
    measure("topk_dcp1", [lambda c=c: tk1(fixed1[c]) for c in range(nc1)])
    measure("dg_local", [lambda c=c: dg0(c) for c in range(nc0)])
    measure("topk_local_knob", [lambda c=c: tk0(fixed0[c], st0.plan) for c in range(nc0)])
    measure("topk_local_default", [lambda c=c: tk0(fixed0[c], plan_default) for c in range(nc0)])
    measure("merge", [lambda c=c: mg0(fixed0[c]) for c in range(nc0)])
    # merge arms on the same inputs, this GPU: the shipped build ("ab"), the reference build (--ref-cuh) and every
    # --variants build; order arms + reversed(arms) (each arm measured twice, mirrored)
    arms = [("ab", (), lambda s: mg0(s))]
    if ref_cuh:
        refmod = load_ref(ref_cuh)
        arms.append(("ref", None, lambda s: refmod.merge(comms[0].meta(0), comms[0].epoch, st0.raw, s, req.causal,
                                                         req.table, out0, st0.fallback, st0.scratch, 8, st0.end_bit)))
    t1l_comm = None
    for vname, vdefs in variants:
        vc = T.TsiComm("cuda", W, 0, cap=32, peers=regions, epoch=comms[0].epoch, defines=vdefs)
        rawv = raw_c if vc.t1l else st0.raw
        t1l_comm = vc if vc.t1l else t1l_comm
        arms.append((vname, vdefs, lambda s, vc=vc, rawv=rawv: vc.merge(0, rawv, s, req.causal, req.table, out0,
                                                                        st0.fallback, st0.scratch, 8, st0.end_bit)))
    tk0(fixed0[0], st0.plan)
    for name, _, fn in arms:
        out0.fill_(-7)
        fn(fixed0[0])
        torch.cuda.synchronize()
        emit(kind="sanity_arm_exact", L=L, rows=rows, arm=name, ok=bool(torch.equal(out0, ref)))
    for name, _, fn in arms + arms[::-1]:
        measure(f"merge_{name}", [lambda c=c, fn=fn: fn(fixed0[c]) for c in range(nc0)])
    emit(kind="merge_arms", L=L, rows=rows,
         arms={name: [round(x, 3) for x in items[f"merge_{name}"]] for name, _, _ in arms})
    if t1l_comm is not None:  # rank-0 chain with the T1L variant: DG local logits -> top-C -> T1L merge, vs chain_tsi
        def t1l_chain(c):
            s = dg0(c)
            tk0_c(s)
            t1l_comm.merge(0, raw_c, s, req.causal, req.table, out0, st0.fallback, st0.scratch, 8, st0.end_bit)

        chain_t = [lambda c=c: t1l_chain(c) for c in range(nc0)]
        for name, fns in (("chain_tsi", chain0), ("chain_tsi_t1l", chain_t), ("chain_tsi_t1l", chain_t), ("chain_tsi", chain0)):
            measure(name, fns)
        measure("topk_local_c", [lambda c=c: tk0_c(fixed0[c]) for c in range(nc0)])
    for name in ("chain_dcp1", "chain_tsi") + (("chain_tsi_t1l",) if t1l_comm is not None else ()):
        v = items[name]
        emit(kind=f"{name}_abba", L=L, rows=rows, blocks=[round(x, 3) for x in v], mean=round(sum(v) / len(v), 3),
             spread=round(max(v) - min(v), 3))
    del caches, fixed0, fixed1, states, comms
    torch.cuda.empty_cache()
    return items


def run_live(Ls, rows, layers, reps, max_copies):
    import torch.distributed as dist

    dist.init_process_group("nccl")
    rank, live = dist.get_rank(), dist.get_world_size()
    torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", rank)))
    hold_fp8_clock()
    comm = T.TsiComm.from_group(dist.group.WORLD, torch.device("cuda"), W, rank, live=live, cap=32)
    for L in Ls:
        g = torch.Generator(device="cuda")
        g.manual_seed(1234 + L)  # identical data on every live rank (replicated index-K)
        req = Request(L, rows, g)
        st = T.TsiState(comm, rows, req.pages)
        st.prepare(req.table, req.contexts, req.causal)
        ll = local_len(L, rank)
        ncc = min(max_copies, max(layers, copies(req, req.b * ll * IDXB)))
        forwards = -(-ncc // layers)
        caches = [req.cache] + [req.cache.clone() for _ in range(ncc - 1)]
        # virtual sources: their push halves into local scratch; this rank's slots -> payload [W - live, rows, C, 2]
        _, vstates, vregions = stage_ranks(req, caches[0], live_from=live)
        pay = torch.empty((W - live, rows, T.CAND, 2), dtype=torch.int32, device="cuda")
        for q in range(live, W):
            for row in range(rows):
                a = T.cand_index(W, T.CAND, 32, 0, q, row, 0)
                recs = vregions[rank][a : a + 2 * T.CAND].view(torch.int32).view(T.CAND, 2, 2)
                pay[q - live, row] = recs[:, :, 0]  # payload = low word of each record (little endian)
        del vstates, vregions
        out = torch.empty(rows, TOPK, dtype=torch.int32, device="cuda")
        ref = req.reference(req.full_logits(caches[0]))
        own = comm.pointers[rank]

        def forward(f, with_select):
            st.prepare(req.table, req.contexts, req.causal)
            for o in range(layers):
                comm.fill(o, pay, own, T.KIND_CAND, live)
                if with_select:
                    st.select(o, req.query, req.gates, caches[(f * layers + o) % ncc], out)

        # eager check: one forward, every live rank's output == the exact selection
        dist.barrier()
        out.fill_(-7)
        forward(0, True)
        torch.cuda.synchronize()
        bad = int((out != ref).any(dim=1).sum())
        emit(kind="sanity_live_exact", L=L, rank=rank, rows_differ=bad, fallback=int(st.fallback.sum()), ok=bad == 0)
        res = {}
        for name, sel in (("A", True), ("B", False)):
            s = torch.cuda.Stream()
            s.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(s):
                forward(0, sel)
            torch.cuda.current_stream().wait_stream(s)
            torch.cuda.synchronize()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                for f in range(forwards):
                    forward(f, sel)
            torch.cuda.synchronize()
            dist.barrier()
            graph.replay()
            torch.cuda.synchronize()
            ts = []
            for _ in range(5):
                dist.barrier()
                a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                a.record()
                for _ in range(reps):
                    graph.replay()
                b.record()
                torch.cuda.synchronize()
                ts.append(a.elapsed_time(b) * 1000.0 / (reps * forwards * layers))
            ts.sort()
            res[name] = ts
            del graph
        emit(kind="live_chain", L=L, rows=rows, rank=rank, live=live, layers=layers, forwards=forwards, copies=ncc,
             us_A=round(res["A"][2], 3), us_B_fill=round(res["B"][2], 3), us_chain=round(res["A"][2] - res["B"][2], 3),
             A=[round(x, 3) for x in res["A"]], B=[round(x, 3) for x in res["B"]])
        del caches, st
        torch.cuda.empty_cache()
    dist.barrier()
    dist.destroy_process_group()


def verdict(all_items, ref_times):
    if not ref_times:
        emit(kind="sanity_dcp1", valid=None, checks=[], verdict="NO_REF")
        return
    lines, ok = [], True
    for (L, rows), items in sorted(all_items.items()):
        if rows != 8:
            continue
        for name in REF_ITEMS:
            refs = ref_times.get(name, {})
            if L in refs and name in items:
                got = sum(items[name]) / len(items[name])
                within = abs(got - refs[L]) <= 0.15 * refs[L]
                ok &= within
                lines.append(f"{name}@{L}: {got:.2f} vs {refs[L]:.2f} {'ok' if within else 'OUT'}")
    emit(kind="sanity_dcp1", valid=ok, checks=lines, verdict="VALID" if ok else "VOID")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="one", choices=("one", "live"))
    ap.add_argument("--L", default="65536,131072,262144,393216,524288,655360,786432")
    ap.add_argument("--rows", default="8,16")
    ap.add_argument("--reps", type=int, default=20)
    ap.add_argument("--layers", type=int, default=24)
    ap.add_argument("--max-copies", type=int, default=1024)
    ap.add_argument("--ref-cuh", default="", help="tsi.cuh of a reference build: merge A/B (ABBA, same inputs)")
    ap.add_argument("--variants", default="", help="merge variant builds: name=DEF1+DEF2;name2=DEF3 (tsi.cuh -D)")
    ap.add_argument("--ref-times", default="", help="JSON reference times (us) for the DCP1 VALID/VOID line")
    a = ap.parse_args()
    ref_times = load_refs(a.ref_times)
    Ls = [int(x) for x in a.L.split(",") if x]
    p = torch.cuda.get_device_properties(0)
    emit(kind="device", name=p.name, sms=p.multi_processor_count, cc=f"{p.major}.{p.minor}", cand=T.CAND,
         cluster_min=T.TOPK_CLUSTER_MIN, t=time.strftime("%Y-%m-%dT%H:%M:%S"))
    if a.mode == "live":
        run_live(Ls, int(a.rows.split(",")[0]), a.layers, max(1, a.reps // 4), min(a.max_copies, 512))
        return 0
    hold_fp8_clock()
    g = torch.Generator(device="cuda")
    g.manual_seed(1234)
    all_items = {}
    for rows in [int(x) for x in a.rows.split(",") if x]:
        for L in Ls:
            try:
                variants = [(v.split("=", 1)[0], tuple(d for d in v.split("=", 1)[1].split("+") if d))
                            for v in a.variants.split(";") if v]
                all_items[(L, rows)] = run_one(L, rows, a.reps, g, a.max_copies, a.ref_cuh or None, variants)
            except Exception as e:  # report and continue: one failing shape must not hide the others
                emit(kind="error", L=L, rows=rows, err=repr(e)[:600])
            torch.cuda.empty_cache()
    verdict(all_items, ref_times)
    return 0


if __name__ == "__main__":
    sys.exit(main())
