# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""ABBA microbenchmark + gates of opt-in KDA ReplaySSM verify variants vs the shipped schedule.

Usage (inside the image, plugin on PYTHONPATH): python3 _w4kda_bench.py OUT.jsonl
Env: W4_REQS="1,2,4", W4_Q=8, W4_ARMS="stock,<opt>,..." (each arm = a K31MK_KDA_VERIFY_OPT value; "stock" = unset),
W4_TIME=1 (ABBA timing per arm vs stock), W4_REPS=20.

Gates (fixed before any run; the same as test_k31_kda.test_kda_verify): output |o-ref| <= .02 + .02|ref| against the
independent FP32 torch reference; rings raw K/V 1e-3 + 2^-7 rel, log-g and beta 1e-4 + 1e-4 rel; conv windows
bit-exact; checkpoint state and persistent conv bit-identical; padded request output exactly 0. Bit-equality with the
stock arm is reported, not gated (reduction order may change).

W4_CHAIN=1 (in-chain timing, pre-registered before any chain run): the back-to-back timing above makes each
verify the PDL predecessor of the next verify, which is not the model's chain (an early/lt arm's overlap depends on its
neighbours). Each timed element is instead [packed KDA projection [R.PITCH (TP8 6288, TP16 3216), 7168] (the verify's real predecessor, the
providers' tactic) -> verify -> split-K [7168, 1536] GEMM] where the last GEMM stands in for the TP o-proj (the real
PreparedOproj needs the TP mailbox; it also streams its weights before its PDL wait, but prefetches all of them, not
two stages: the proxy under-weights how much a late trigger (lt) delays it). Weight copies defeat L2 as in
test_k31_projection. Metric: median us per element over W4_REPS graph replays; ABBA per arm vs stock; a "none" element
(projection + proxy, no verify) is timed before and after the arms, so in-chain verify time = element - none. Decision
(fixed now): arms with early or lt are reported at their in-chain gain (stock element - arm element); the back-to-back
gain is called an overstatement if it exceeds the in-chain gain by more than 0.2 us.
"""

import json
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _harness as H  # noqa: E402
import _ref_attnkda as R  # noqa: E402
import test_k31_kda as T  # noqa: E402

OUT = sys.argv[1]
Q = int(os.environ.get("W4_Q", "8"))
HEADS, D = T.HEADS, T.D


def emit(row):
    row = dict(row)
    with open(OUT, "a") as f:
        f.write(json.dumps(row) + "\n")
    print("W4", json.dumps(row), flush=True)


FENCE_ENV = ("K31MK_KDA_PROJ_FENCE",)


def with_opt(opt, fn):
    """Run ``fn`` with K31MK_KDA_VERIFY_OPT set to ``opt`` (unset for stock): kernels are chosen at prepare.
    ``opt@l.o.c`` also forces K31MK_KDA_VERIFY_SCHEDULE=l.o.c (lanes, occupancy, cluster) for that arm.
    A trailing ``!F<m>`` forces fix A mode m (1-3) on the chain's KDA projection
    (K31MK_KDA_PROJ_FENCE=m), ``!U`` keeps the shipped trigger under early (the test-only
    kda.verify.projection_fence patched to 0, measurement-only); otherwise kda.verify.projection_fence() decides (EARLY_FENCE_MODE with early)."""
    opt, _, flag = opt.partition("!")
    opt, _, sched = opt.partition("@")
    if opt in ("", "stock"):
        os.environ.pop("K31MK_KDA_VERIFY_OPT", None)
    else:
        os.environ["K31MK_KDA_VERIFY_OPT"] = opt
    if sched:
        os.environ["K31MK_KDA_VERIFY_SCHEDULE"] = sched
    from sgl_nvidia_mega.kimi_k31.k3.cutedsl.kda import verify as KV

    _SHIPPED_FENCE = KV.projection_fence
    if flag == "U":
        KV.projection_fence = lambda: 0
    elif flag:
        assert flag[0] == "F" and flag[1:] in ("1", "2", "3"), flag
        os.environ[FENCE_ENV[0]] = flag[1:]
    try:
        return fn()
    finally:
        KV.projection_fence = _SHIPPED_FENCE
        for key in ("K31MK_KDA_VERIFY_OPT", "K31MK_KDA_VERIFY_SCHEDULE", *FENCE_ENV):
            os.environ.pop(key, None)


def arm_fenced(arm):
    """The arm's KDA projection fix A mode (0 = shipped; the rule providers.build_resources applies)."""
    from sgl_nvidia_mega.kimi_k31.k3.cutedsl.kda.verify import projection_fence

    return with_opt(arm, projection_fence)


def make_core(requests, pools, layer, opt):
    from sgl_nvidia_mega.kimi_k31.k3.cutedsl.kda.prepared import decode_conv_weight
    from sgl_nvidia_mega.kimi_k31.verify_kda import PreparedKdaVerify

    weights = T._NS(
        conv_weight=decode_conv_weight(layer.conv1d.weight), has_bias=False, f_b=layer.f_b_proj.weight,
        a_log=layer.A_log, dt_bias=layer.dt_bias, norm_weight=layer.o_norm.weight.float(),
        lower_bound=T.LOWER, scale=128**-0.5, norm_eps=T.EPS,
    )
    core = PreparedKdaVerify(weights, ((requests, Q, R.PITCH, 0),), torch.cuda.current_device())
    return with_opt(opt, lambda: core.prepare_caches(pools))


def setup(requests):
    g = H.generator(4000 + requests)
    layer = T._layer(g)
    cap, ring_len, scratch_rows = 2 * requests + 3, 16, 2 * requests + 1
    conv_phys, state = T._pools(g, cap)
    proj = T._proj(g, requests * Q)
    slots = (torch.randperm(cap, generator=torch.Generator().manual_seed(requests))[:requests]).to(torch.int32).cuda()
    scratch = torch.randperm(scratch_rows, generator=torch.Generator().manual_seed(7 + requests))[:requests]
    scratch = scratch.to(torch.int32).cuda()
    if requests == 4:
        slots[2] = -1
    return dict(g=g, layer=layer, cap=cap, ring_len=ring_len, scratch_rows=scratch_rows, conv=conv_phys,
                state=state, proj=proj, slots=slots, scratch=scratch)


def fresh_pools(s):
    cap, L = s["cap"], s["ring_len"]
    rawv = torch.zeros((cap, HEADS, L, 128), dtype=torch.bfloat16, device="cuda")
    rawk = torch.zeros_like(rawv)
    log_g = torch.zeros((cap, HEADS, L, 128), dtype=torch.float32, device="cuda")
    beta = torch.zeros((cap, HEADS, L), dtype=torch.float32, device="cuda")
    windows = torch.zeros((s["scratch_rows"], Q, 3, 3 * D), dtype=torch.bfloat16, device="cuda")
    for t in (rawv, rawk, log_g, beta, windows):
        H.poison(t)
    conv, state = s["conv"].clone(), s["state"].clone()
    return T._NS(conv=conv.transpose(-1, -2), temporal=state, rawv=rawv, rawk=rawk, log_g=log_g, beta=beta,
                 conv_windows=windows.transpose(-1, -2)), conv, state, windows


def run_checked(requests, s, opt, ref):
    pools, conv_phys, state, windows = fresh_pools(s)
    core = make_core(requests, pools, s["layer"], opt)
    out = torch.empty((requests * Q, D), dtype=torch.bfloat16, device="cuda")
    H.poison(out)
    core.run(s["proj"], out, pools=pools, step=T._NS(state_slots=s["slots"], scratch_rows=s["scratch"], query_width=Q))
    torch.cuda.synchronize()
    ref_out, ref_rings, ref_windows = ref
    item = f"w4.breq{requests}.q{Q}.{opt or 'stock'}"
    res = dict(item=item, ok=True)
    try:
        H.check(item + ".out", out, ref_out, 0.02, 0.02)
        assert torch.equal(state.view(torch.int32), s["state"].view(torch.int32)), "checkpoint state modified"
        assert torch.equal(conv_phys.view(torch.int16), s["conv"].view(torch.int16)), "persistent conv modified"
        for r in range(requests):
            if int(s["slots"][r]) < 0:
                assert int(out[r * Q:(r + 1) * Q].view(torch.int16).abs().sum()) == 0, "padded request output"
        got = [torch.stack([buf[sl, :, t] for (sl, t) in ref_rings]) for buf in
               (pools.rawv, pools.rawk, pools.log_g, pools.beta)]
        refs = [torch.stack([v[i] for v in ref_rings.values()]) for i in range(4)]
        H.check(item + ".rawv", got[0], refs[0], 1e-3, 2**-7)
        H.check(item + ".rawk", got[1], refs[1], 1e-3, 2**-7)
        H.check(item + ".log_g", got[2], refs[2], 1e-4, 1e-4)
        H.check(item + ".beta", got[3], refs[3], 1e-4, 1e-4)
        for row, win in ref_windows.items():
            assert torch.equal(windows[row].view(torch.int16), win.view(torch.int16)), f"conv window row {row}"
    except AssertionError as e:
        res.update(ok=False, error=str(e)[:500])
    res["resources"] = H.function_resources(core.provider.kernels[requests, Q])
    return res, core, out, pools


def chain_parts(requests, fenced=0, base=None):
    """W4_CHAIN=1: per-copy KDA projections (before the verify) and o-proj proxies (after it); see the module doc.
    ``fenced``: the KDA projections use fix A; ``base``: an unfenced chain_parts whose weights and
    o-proj proxies are reused (same bytes, same proxies, only the projection's trigger differs)."""
    from sgl_nvidia_mega.kimi_k31.k3.cutedsl.gemm.projection import PreparedProjection, projection_tactic

    rows = requests * Q
    tactic = projection_tactic("kda", rows, torch.device("cuda", torch.cuda.current_device()))
    if base is not None:
        ins = [PreparedProjection(p.weight, {rows: tactic}, fenced_trigger=fenced).prepare() for p in base["ins"]]
        return dict(base, ins=ins, fenced=fenced)
    g = H.generator(77)
    wbytes = (R.PITCH * 7168 + 7168 * D) * 2
    n = H.copies_for(wbytes)
    ins = [PreparedProjection(H.randn((R.PITCH, 7168), 7168**-0.5, g), {rows: tactic}, fenced_trigger=fenced).prepare()
           for _ in range(n)]
    outs = [PreparedProjection(H.randn((7168, D), D**-0.5, g), {rows: tactic}).prepare() for _ in range(n)]
    x = H.randn((rows, 7168), 1.0, g)
    return dict(ins=ins, outs=outs, x=x, wbytes=wbytes, tactic=tactic, fenced=fenced)


def chain_wrap(requests, verify_calls, chain, count=None):
    """[projection -> verify -> o-proj proxy] per element; ``verify_calls`` None = the "none" reference."""
    rows, n = requests * Q, len(chain["ins"])
    count = len(verify_calls) if verify_calls else count
    projected = [torch.empty((rows, R.PITCH), dtype=torch.bfloat16, device="cuda") for _ in range(count)]
    local = [torch.zeros((rows, D), dtype=torch.bfloat16, device="cuda") for _ in range(count)]
    hidden = [torch.empty((rows, 7168), dtype=torch.bfloat16, device="cuda") for _ in range(count)]

    def element(i):
        chain["ins"][i % n].run(chain["x"], projected[i])
        if verify_calls:
            verify_calls[i](projected[i], local[i])
        chain["outs"][i % n].run(local[i], hidden[i])

    return [lambda i=i: element(i) for i in range(count)]


def timed_calls(requests, s, core, opt):
    state_bytes = HEADS * 128 * 128 * 4
    nbytes = requests * (state_bytes + Q * (R.PITCH * 2 + D * 2 + 4 * HEADS * 128 * 2 + 3 * 3 * D * 2))
    nbytes += D * 128 * 2 + 3 * D * 4 * 4
    copies = H.copies_for(nbytes, cap=48)
    g = H.generator(99)
    tconv, tstate = T._pools(g, copies * requests)
    L = s["ring_len"]
    pools = T._NS(conv=tconv.transpose(-1, -2), temporal=tstate,
                  rawv=torch.zeros((copies * requests, HEADS, L, 128), dtype=torch.bfloat16, device="cuda"),
                  rawk=torch.zeros((copies * requests, HEADS, L, 128), dtype=torch.bfloat16, device="cuda"),
                  log_g=torch.zeros((copies * requests, HEADS, L, 128), dtype=torch.float32, device="cuda"),
                  beta=torch.zeros((copies * requests, HEADS, L), dtype=torch.float32, device="cuda"),
                  conv_windows=torch.zeros((s["scratch_rows"], Q, 3, 3 * D), dtype=torch.bfloat16,
                                           device="cuda").transpose(-1, -2))
    with_opt(opt, lambda: core.prepare_caches(pools))
    outs = [torch.empty((requests * Q, D), dtype=torch.bfloat16, device="cuda") for _ in range(copies)]
    steps = [T._NS(state_slots=torch.arange(i * requests, (i + 1) * requests, dtype=torch.int32, device="cuda"),
                   scratch_rows=s["scratch"], query_width=Q) for i in range(copies)]
    if os.environ.get("W4_CHAIN") == "1":  # the caller supplies the projected input and the output
        return [lambda p, o, st=st: core.run(p, o, pools=pools, step=st) for st in steps], nbytes
    return [lambda o=o, st=st: core.run(s["proj"], o, pools=pools, step=st) for o, st in zip(outs, steps)], nbytes


def main():
    H.load_plugin(T.MUTATIONS)
    reqs = [int(x) for x in os.environ.get("W4_REQS", "1").split(",")]
    arms = os.environ.get("W4_ARMS", "stock").split(",")
    reps = int(os.environ.get("W4_REPS", "20"))
    for requests in reqs:
        s = setup(requests)
        ref = R.kda_verify(s["proj"], s["conv"], s["state"], s["slots"], s["scratch"], Q, ring_len=s["ring_len"],
                           **T._ref_kwargs(s["layer"]))
        cores, outs, rings = {}, {}, {}
        for arm in arms:
            try:
                res, core, out, pools = run_checked(requests, s, arm, ref)
            except Exception as e:  # compile / launch failure of one arm: record, keep going
                import traceback
                emit(dict(kind="check", requests=requests, arm=arm, ok=False,
                          error=f"{type(e).__name__}: {str(e)[:800]}", trace=traceback.format_exc()[-2500:]))
                torch.cuda.synchronize()
                continue
            cores[arm], outs[arm] = core, out
            # Every tensor the verify writes: raw V / K, log-g, beta rings and the conv windows.
            rings[arm] = [t.view(torch.int16 if t.element_size() == 2 else torch.int32) for t in
                          (pools.rawv, pools.rawk, pools.log_g, pools.beta, pools.conv_windows)]
            if arm != arms[0] and arms[0] in outs:
                same = bool(torch.equal(out.view(torch.int16), outs[arms[0]].view(torch.int16)))
                res["bit_equal_out_vs_" + arms[0]] = same
                written = all(torch.equal(a, b) for a, b in zip(rings[arm], rings[arms[0]]))
                res["bit_equal_rings_vs_" + arms[0]] = written
                # Before any chunk run: "chunk" changes only the outputs' FP32 order.
                if "chunk" in arm.partition("!")[0].partition("@")[0].split("+") and not written:
                    res.update(ok=False, error="chunk changed a ring / window vs the base arm")
                # Pre-registered (before any run): these knobs issue the stock arithmetic in the
                # stock order (wide/quad only with wl8, i.e. stock lanes), so their output must be
                # bit-identical to stock; a 0.02-tolerance pass could hide an index bug.
                knobs = set(arm.partition("!")[0].partition("@")[0].split("+"))
                lanes_kept = not (knobs & {"wide", "quad"}) or "wl8" in knobs
                # "pw" added before any pw run (L2 prefetches only).
                if knobs <= {"early", "dmc", "rs", "wide", "quad", "wl8", "pw"} and lanes_kept:
                    res["expect_bit_equal"] = True
                    if not same or not written:
                        res.update(ok=False, error="expected bit-identical outputs and rings vs stock")
            emit(dict(kind="check", requests=requests, arm=arm, **res))
        if os.environ.get("W4_TIME", "1") != "1":
            continue
        chain = chain_parts(requests) if os.environ.get("W4_CHAIN") == "1" else None
        chains = {0: chain}
        fenced = {arm: arm_fenced(arm) for arm in arms if arm in cores}
        for m in sorted(set(fenced.values()) - {0}):
            if chain:
                chains[m] = chain_parts(requests, m, base=chain)
        calls = {}
        for arm in arms:
            if arm in cores:
                calls[arm], nbytes = timed_calls(requests, s, cores[arm], arm)
                if chain:
                    calls[arm] = chain_wrap(requests, calls[arm], chains[fenced[arm]])
        base = arms[0]
        if base not in calls:
            emit(dict(kind="error", requests=requests, error=f"base arm {base} unavailable; no timing"))
            continue
        kind = "chain" if chain else "time"
        extra = dict(tactic=chain["tactic"]) if chain else {}
        cycles = int(os.environ.get("W4_CYCLES", "1"))  # ABBA repeated per contrast
        nones = {}
        if chain:
            for f, c in chains.items():
                nones["none" + (f"_m{f}" if f else "")] = chain_wrap(requests, None, c, count=len(calls[base]))
            for name, none in nones.items():
                emit(dict(kind=kind, requests=requests, arm=name, contrast="none", **H.cold_time_us(none, reps=reps), **extra))
        for arm in [a for a in arms[1:] if a in calls]:
            for cycle in range(cycles):
                for a in (base, arm, arm, base):
                    t = H.cold_time_us(calls[a], reps=reps)
                    emit(dict(kind=kind, requests=requests, arm=a, contrast=arm, cycle=cycle,
                              fenced=int(fenced.get(a, 0)), **t, **H.floor_us(nbytes), **extra))
        for name, none in nones.items():
            emit(dict(kind=kind, requests=requests, arm=name, contrast="none", **H.cold_time_us(none, reps=reps), **extra))
    print("W4_DONE", flush=True)


if __name__ == "__main__":
    main()
