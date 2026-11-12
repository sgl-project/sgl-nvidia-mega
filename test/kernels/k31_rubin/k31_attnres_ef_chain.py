# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Probe: the H4 landing -> ClusterAttnRes ring ->
KDA in-projection boundary at TP16 per-rank shapes (packed projection 3216 x 7168, 8 rows), one GPU, L layers
with distinct projection weights, PDL-chained as the verify executor launches them (the UB-A chain form):

  tail stand-in (2 x SMs CTAs): every CTA spins and releases its dependents; CTAs 8.. count themselves done and
  exit; CTAs 0-7 (the hop) wait for the others, spin H us, publish the layer's 8 delta rows into the ring
  [1, 64, 7168] with st.relaxed.sys (the MoE tail's H4 publish, unicast form), exit.
     -> ClusterAttnRes ring (arm variant) -> SplitKGemm (64, 8, 2, 12)

Arms (same inputs; every arm's x, prefix and blocks compared after an eager pass):
  A / A2  shipped (two graph instances: the instance control)
  E       K31MK_ATTNRES_LATE_REARM (re-arm + fence after the output stores; must be bitwise == A)
  N       REARM_FENCE "0" (no re-arm fence; a COST CONTROL, unsafe, never served; bitwise == A here)
  Q       K31MK_ATTNRES_ONE_ROUND (REORDER: within the reference tolerance, word diffs vs A counted)
  EQ      both (bitwise == Q)
Mutant (eager only, must FAIL): ME = E with the late re-arm dropped (ring left published).

Metrics per (arm, nb), median over replays x layers (CUPTI kernel records via torch.profiler):
  P = projection END - stand-in END (decides), B = AttnRes END - stand-in END, proj = projection END - AttnRes END.

Usage (one GPU): python3 k31_attnres_ef_chain.py --out DIR [--layers 8] [--reps 25] [--hop 2.0] [--nbs 1,8] [--tag gpu0]
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

os.environ.setdefault("CUDA_MODULE_LOADING", "EAGER")

import torch  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _harness as H  # noqa: E402
import _ref_attnkda as R  # noqa: E402

N, K, ROWS, CAP = 3216, 7168, 8, 64
TACTIC = (64, 8, 2, 12)  # TP16 B8 KDA projection as served
EPS = 1e-6
SENTINEL = -0x80000000  # int32 view of 0x80000000
ARMS = ("A", "E", "N", "Q", "EQ", "A2")
FAMILIES = (("standin", "EfTailStandin"), ("attnres", "ClusterAttnRes"), ("proj", "SplitKGemm"))


def kernels(sms, smem, spin_ns):
    import cutlass
    import cutlass.cute as cute
    from cutlass import Int32, Int64
    from sgl_nvidia_mega.kimi_k31.k3.cutedsl.common.primitives import (
        asm, at, fail_stop, global_timer_ns, launch, load_words, store_words)

    @asm()
    def add_release(pointer):
        return "red.release.gpu.global.add.u32 [$0],1;", "l,~{memory}", (pointer,)

    @asm(Int32)
    def load_acquire(pointer):
        return "ld.acquire.gpu.global.u32 $0,[$1];", "=r,l,~{memory}", (pointer,)

    nctas = 2 * sms

    class EfTailStandin:
        """The GaTailStandin form: CTAs >= ROWS spin, release dependents, count themselves done; CTAs < ROWS (the hop)
        spin, release dependents, wait for every other CTA, spin ``hop_ns``, publish row = CTA index, exit."""

        @cute.jit
        def __call__(self, counter: cute.Tensor, slot: Int32, hop_ns: Int32, src: cute.Tensor, dst: cute.Tensor,
                     stream):
            launch(self.kernel, (counter, slot, hop_ns, src, dst, stream), nctas, 128, smem=cutlass.Int64(smem))

        @cute.kernel
        def kernel(self, args):
            counter, slot, hop_ns, src, dst = args
            block, _, _ = cute.arch.block_idx()
            tid, _, _ = cute.arch.thread_idx()
            cute.arch.griddepcontrol_wait()
            start = global_timer_ns()
            now = global_timer_ns()
            while now - start < Int64(spin_ns):
                now = global_timer_ns()
            cute.arch.griddepcontrol_launch_dependents()
            if block >= ROWS:
                cute.arch.sync_threads()
                if tid == 0:
                    add_release(at(counter, slot))
            else:
                seen = load_acquire(at(counter, slot))
                began = global_timer_ns()
                while seen < Int32(nctas - ROWS):
                    seen = load_acquire(at(counter, slot))
                    if global_timer_ns() - began > Int64(2_000_000_000):
                        fail_stop()
                start = global_timer_ns()
                now = global_timer_ns()
                while now - start < Int64(hop_ns):
                    now = global_timer_ns()
                for k in cutlass.range_constexpr(7):
                    col = (k * 128 + tid) * 8
                    words = load_words(at(src, block, col), count=4, mutable=True)
                    store_words(at(dst, 0, block, col), words, scope="sys")

    return EfTailStandin(), nctas


def canonical(x):
    """BF16 rows without -0.0 (the ring's sentinel word is 0x80000000 = two -0.0 halves only as a pair)."""
    x = x.clone()
    x[x == 0] = 0.0
    return x


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--layers", type=int, default=8)
    ap.add_argument("--reps", type=int, default=25)
    ap.add_argument("--hop", type=float, default=2.0)
    ap.add_argument("--spin-us", type=float, default=8.0)
    ap.add_argument("--nbs", default="1,8")
    ap.add_argument("--tag", default="gpu0")
    ap.add_argument("--seed", type=int, default=5)
    ap.add_argument("--watchdog", type=int, default=600)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    import faulthandler

    faulthandler.dump_traceback_later(args.watchdog, exit=False)
    res = {"tag": args.tag, "args": vars(args), "ok": False}
    t0 = time.time()
    try:
        H.load_plugin()
        import cutlass
        import cutlass.cute as cute
        from cutlass.cute.runtime import make_fake_stream
        import sgl_nvidia_mega.kimi_k31.k3.cutedsl.attn_res.kernel as KM
        import sgl_nvidia_mega.kimi_k31.k3.cutedsl.common.mailbox as MB
        from sgl_nvidia_mega.kimi_k31.k3.cutedsl.attn_res.prepared import PreparedAttnRes, compile_consumer
        from sgl_nvidia_mega.kimi_k31.k3.cutedsl.common.host import architecture, stream, to_cute
        from sgl_nvidia_mega.kimi_k31.k3.cutedsl.gemm.projection import PreparedProjection

        dev = torch.device("cuda", torch.cuda.current_device())
        sms = torch.cuda.get_device_properties(dev).multi_processor_count
        res.update(sms=sms, device=torch.cuda.get_device_name(dev), rearm_fence_mode=MB.REARM_FENCE)
        if MB.REARM_FENCE != "gpu":
            raise RuntimeError(f"served REARM_FENCE is 'gpu'; this process has {MB.REARM_FENCE!r}")
        nbs = [int(x) for x in args.nbs.split(",")]
        g = H.generator(args.seed)
        L = args.layers
        standin, nctas = kernels(sms, 48 * 1024, int(args.spin_us * 1000))
        counter = torch.zeros(L, dtype=torch.int32, device=dev)
        mailbox = torch.empty((1, CAP, K), dtype=torch.bfloat16, device=dev)
        mailbox.view(torch.int32).fill_(SENTINEL)
        mview = to_cute(mailbox)
        arch = architecture(dev)
        norm = lambda: NS(weight=(1.0 + H.randn((K,), 0.2, g)).to(torch.bfloat16), variance_epsilon=EPS)  # noqa: E731
        patched = {}

        def variant(target, arm):
            """A PreparedAttnRes whose ring code is the arm's variant (knobs read at bind time, as served)."""
            env = {"K31MK_ATTNRES_LATE_REARM": "1" if arm in ("E", "EQ", "ME") else "0",
                   "K31MK_ATTNRES_ONE_ROUND": "1" if arm in ("Q", "EQ") else "0"}
            old = {k: os.environ.get(k) for k in env}
            os.environ.update(env)
            try:
                prep = PreparedAttnRes.for_target(target, deadline_ns=4_000_000_000, final=False, has_delta=True)
                if arm in ("N", "ME"):
                    # Compiles that bypass compile_consumer's cache, with a patched module global / function
                    # (never served): N = no re-arm fence (cost control); ME = E with its late re-arm dropped.
                    ckey = (prep.key, arm)
                    if ckey not in patched:
                        saved = (MB.REARM_FENCE, KM.rearm)
                        if arm == "N":
                            MB.REARM_FENCE = ""
                        else:
                            @cute.jit
                            def _drop(pointer):
                                pass

                            KM.rearm = _drop
                        try:
                            patched[ckey] = compile_consumer.__wrapped__(
                                *prep.key, 1, CAP, arch, prep.deadline_ns, False, False, False, False, arm == "ME",
                                False)
                        finally:
                            MB.REARM_FENCE, KM.rearm = saved
                    prep.ring = patched[ckey], to_cute(mailbox), (1, CAP)
                else:
                    prep.bind_ring(mailbox)
            finally:
                for k, v in old.items():
                    if v is None:
                        os.environ.pop(k, None)
                    else:
                        os.environ[k] = v
            return prep

        weights = [H.randn((N, K), 0.02, g) for _ in range(L)]
        projs = [PreparedProjection(w, {ROWS: TACTIC}).prepare() for w in weights]
        deltas = [canonical(H.randn((ROWS, K), 1.0, g)) for _ in range(L)]
        dviews = [to_cute(d) for d in deltas]
        sview = to_cute(counter)
        st_code = cute.compile(standin, sview, cutlass.Int32(0), cutlass.Int32(0), dviews[0], mview,
                               make_fake_stream())
        sets = {}
        for nb in nbs:
            targets = [NS(self_attention_res_norm=norm(), self_attention_res_proj=NS(weight=H.randn((1, K), 0.02, g)),
                          input_layernorm=norm(), prev_valid_blocks=nb, is_block_write_layer=False, block_write_idx=0)
                       for _ in range(L)]
            prefix0, blocks0 = H.randn((ROWS, K), 2.0, g), H.randn((ROWS, 9, K), 2.0, g)
            sets[nb] = NS(targets=targets, prefix0=prefix0, blocks0=blocks0, prefix=prefix0.clone(),
                          blocks=blocks0.clone(), x=[torch.empty((ROWS, K), dtype=torch.bfloat16, device=dev)
                                                     for _ in range(L)],
                          y=[torch.empty((ROWS, N), dtype=torch.bfloat16, device=dev) for _ in range(L)],
                          arms={arm: [variant(t, arm) for t in targets] for arm in ARMS + ("ME",)})

        def run(nb, arm):
            s = sets[nb]
            for i in range(L):
                st_code(sview, cutlass.Int32(i), cutlass.Int32(int(args.hop * 1000)), dviews[i], mview, stream(dev))
                s.arms[arm][i].run_ring(s.prefix, s.blocks, s.x[i])
                projs[i].run(s.x[i], s.y[i])

        def run_graph(nb, arm):
            counter.zero_()
            run(nb, arm)

        def reset(nb):
            s = sets[nb]
            s.prefix.copy_(s.prefix0)
            s.blocks.copy_(s.blocks0)
            mailbox.view(torch.int32).fill_(SENTINEL)

        print("STAGE eager", flush=True)
        checks, info = {}, {}
        for nb in nbs:
            s = sets[nb]
            outs = {}
            for arm in ARMS + ("ME",):
                reset(nb)
                run_graph(nb, arm)
                torch.cuda.synchronize()
                words = mailbox[0, :ROWS].contiguous().view(torch.int32)
                rearmed = int((words != SENTINEL).sum())
                outs[arm] = (torch.stack([x.clone() for x in s.x]), s.prefix.clone(), s.blocks.clone(),
                             torch.stack([y.clone() for y in s.y]))
                if arm != "ME":
                    checks[f"nb{nb}.{arm}.rearmed"] = rearmed
                else:
                    info[f"nb{nb}.ME.rearmed"] = rearmed
            base = outs["A"]

            def ndiff(a, b):
                return sum(int((p.view(torch.int16) != q.view(torch.int16)).sum()) for p, q in zip(a, b))

            for arm in ("A2", "E", "N"):
                checks[f"nb{nb}.{arm}.diff_vs_A"] = ndiff(outs[arm], base)
            checks[f"nb{nb}.EQ.diff_vs_Q"] = ndiff(outs["EQ"], outs["Q"])
            me = ndiff(outs["ME"], outs["E"]) + info[f"nb{nb}.ME.rearmed"]
            info[f"nb{nb}.ME.diff_plus_rearmed"] = me
            checks[f"nb{nb}.ME.caught"] = 0 if me > 0 else 1  # 0 = the mutant was caught
            # FP32 reference chain (prefix evolves layer by layer); A and Q against it, every layer.
            p_ref, viol = s.prefix0.clone(), {"A": 0, "Q": 0}
            for i in range(L):
                t = s.targets[i]
                out_ref, p_ref, _ = R.attnres(deltas[i], p_ref, s.blocks0, nb=nb, has_delta=True, write=-1,
                                              norm_w=t.self_attention_res_norm.weight,
                                              query_w=t.self_attention_res_proj.weight,
                                              out_w=t.input_layernorm.weight, eps=EPS, out_eps=EPS)
                for arm in viol:
                    st = H.check(f"nb{nb}.{arm}.x{i}", outs[arm][0][i], out_ref, 1e-3, 2**-7, frac=1.0)
                    viol[arm] += st["violations"]
            checks[f"nb{nb}.A.x_violations"] = viol["A"]
            checks[f"nb{nb}.Q.x_violations"] = viol["Q"]
            info[f"nb{nb}.Q.x_words_diff_vs_A"] = ndiff((outs["Q"][0],), (base[0],))
            info[f"nb{nb}.Q.prefix_diff_vs_A"] = ndiff((outs["Q"][1],), (base[1],))
            info[f"nb{nb}.x_words"] = base[0].numel()
            reset(nb)
        res["checks"], res["info"] = checks, info
        res["correct"] = all(v == 0 for v in checks.values())
        print("STAGE capture", json.dumps(checks), json.dumps(info), flush=True)
        graphs = {}
        cap = torch.cuda.Stream(device=dev)
        cap.wait_stream(torch.cuda.current_stream(dev))
        for nb in nbs:
            for arm in ARMS:
                torch.cuda.synchronize()
                gr = torch.cuda.CUDAGraph()
                with torch.cuda.graph(gr, stream=cap):
                    run_graph(nb, arm)
                graphs[nb, arm] = gr
        torch.cuda.synchronize()
        H.hold_fp8_clock()
        for gr in graphs.values():
            for _ in range(3):
                gr.replay()
        torch.cuda.synchronize()
        schedule = []
        for nb in nbs:
            order = [(nb, a) for a in ARMS]
            schedule += (order + order[::-1]) * args.reps
        from torch.profiler import ProfilerActivity, profile

        print("STAGE timing", flush=True)
        with profile(activities=[ProfilerActivity.CUDA]) as prof:
            for key in schedule:
                graphs[key].replay()
            torch.cuda.synchronize()
        trace = os.path.join(args.out, f"trace_{args.tag}.json.gz")
        raw = trace[:-3]
        prof.export_chrome_trace(raw)
        with open(raw, "rb") as f, gzip.open(trace, "wb") as z:
            z.write(f.read())
        os.remove(raw)
        words = mailbox[0, :ROWS].contiguous().view(torch.int32)
        res["rearmed_after_timing"] = int((words != SENTINEL).sum())
        res["arms"] = parse(trace, schedule, L)
        res["ok"] = res["correct"] and res["rearmed_after_timing"] == 0
    except BaseException as e:  # noqa: BLE001
        res["error"] = repr(e)[:2000]
        res["trace"] = traceback.format_exc()[-6000:]
        print(f"ERROR {e!r}\n{traceback.format_exc()}", flush=True)
    res["total_s"] = round(time.time() - t0, 1)
    with open(os.path.join(args.out, f"chain_{args.tag}.json"), "w") as f:
        json.dump(res, f, indent=1, default=str)
    brief = {k: res.get(k) for k in ("tag", "ok", "checks", "info", "rearmed_after_timing", "error")}
    brief["arms"] = res.get("arms")
    print("EF_RESULT", json.dumps(brief), flush=True)
    sys.exit(0 if res["ok"] else 1)


def med(xs):
    return round(statistics.median(xs), 4) if xs else None


def parse(trace_path, schedule, layers):
    d = json.load(gzip.open(trace_path))
    ks = sorted((e for e in d["traceEvents"] if e.get("cat") == "kernel"), key=lambda e: e["ts"])
    fam = [(next((n for n, p in FAMILIES if p in e["name"]), None), e) for e in ks]
    fam = [(f, e) for f, e in fam if f is not None]
    out, pos = {}, 0
    for nb, arm in schedule:
        per = layers * 3
        chunk, pos = fam[pos:pos + per], pos + per
        by = {}
        for f, e in chunk:
            by.setdefault(f, []).append(e)
        row = out.setdefault(f"{arm}@nb{nb}", {"P": [], "B": [], "proj": []})
        st, a, p = by.get("standin", []), by.get("attnres", []), by.get("proj", [])
        if not (len(st) == len(a) == len(p) == layers):
            raise RuntimeError(f"{arm}@nb{nb}: {len(st)} stand-ins, {len(a)} attnres, {len(p)} proj (want {layers})")
        for i in range(layers):
            st_end = st[i]["ts"] + st[i]["dur"]
            a_end = a[i]["ts"] + a[i]["dur"]
            p_end = p[i]["ts"] + p[i]["dur"]
            row["P"].append(p_end - st_end)
            row["B"].append(a_end - st_end)
            row["proj"].append(p_end - a_end)
    return {k: {m: med(v) for m, v in r.items()} | {"n": len(r["P"])} for k, r in out.items()}


if __name__ == "__main__":
    main()
