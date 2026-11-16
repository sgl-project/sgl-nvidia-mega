# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""K3.1 routed experts on one GPU at the TP8 per-rank shapes (896 experts, I = 384 per
rank, MXFP4 weights in the host's FlashInfer layout x MXFP8 activations, SiTU 4 / 25):
``ClusterExpertsTP8`` (TP8 B1 small-batch chain) and ``PersistentExperts`` (B1-B8,
non-publish: per-(token, slot) BF16 rows in top-16 order).

Reference (``_ref_moe``): FP64 torch of the stock method from the logical (unshuffled)
weights, independent of the kernels; plus the stock FlashInfer
``trtllm_fp4_block_scale_routed_moe`` on the same MXFP8 inputs as the yardstick.

Tolerances (stated before any run; never loosened after):
  T1 per-expert rows (BF16) vs the FP64 reference: Frobenius rel-L2 over all checked
     rows <= 1e-2, every row's rel-L2 <= 3e-2, every cell finite (outputs are NaN-poisoned
     before each call, so an unwritten cell fails).
     Basis: the method matches exactly except FP32 accumulation order (~1e-6), BF16 output
     rounding (2^-9 rel) and the kernels' approximate tanh/ex2/rcp in SiTU (~5e-4 rel),
     which flip a few hidden E4M3 roundings (~1/16 of one hidden element each, about
     0.3% rel-L2 per flip on the output). A dropped K chunk (256 of 3584), a wrong
     scale byte, swapped gate/up or a wrong expert moves the row by >= ~10%.
  T2 weighted sum over the 16 slots vs the FP64 reference, against FlashInfer on the
     same inputs: rel-L2(MK) <= 1.25 rel-L2(FI) + 1e-3 and max|MK - ref| <=
     2 max|FI - ref| + 2^-8 rms(ref). FlashInfer is the shipped stock method, so this
     says "no less accurate than stock".
Every call is checked eagerly and once more replayed from a CUDA graph with new inputs.

Mutations (each must make the suite fail; the bug class each stands for):
  cl_situ_swap     ClusterExpertsTP8 passes (up, gate) to SiTU      (operand order)
  cl_xscale        ClusterExpertsTP8 x block scale byte + 1          (activation scale)
  cl_fc2_tile      ClusterExpertsTP8 drops each CTA's last FC2 tile  (missing output rows)
  cl_hidden_half   ClusterExpertsTP8 hidden block amax over 16 of 32 (scale-block mismatch)
  pe_situ_swap     PersistentExperts SiTU operand swap
  pe_drop_partial  PersistentExperts drops split-K FC1 peer partials (lost K part)
  pe_hidden_half   PersistentExperts hidden amax over one warp's half (scale-block mismatch)
  pe_xscale        PersistentExperts x block scale byte + 1

Timing (``K31_TIME=1``, unmutated): cold weights (each call routes to experts no earlier
call in the cycle touched, so they stream from HBM), CUDA-graph replay, median of 20;
MK and FlashInfer (with whatever FlashInfer build is installed; the run script installs
the DynBlock sm107 module first) in ABBA order on the same GPU.
"""

import os

import pytest
import torch

import _harness as H
import _ref_moe as R

SB = "kimi_k31/k3/cutedsl/moe/small_batch/experts.py"
PE = "kimi_k31/k3/cutedsl/moe/experts.py"
MUTATIONS = {
    "cl_situ_swap": (SB, "hidden = situ(other, value, *situ_args) if up else Float32(0)\n                amax = abs(hidden)",
                     "hidden = situ(value, other, *situ_args) if up else Float32(0)\n                amax = abs(hidden)"),
    "cl_xscale": (SB, "sm[self.off_xsf + (kb // 4) * 128 + kb % 4] = Uint8(sb)",
                  "sm[self.off_xsf + (kb // 4) * 128 + kb % 4] = Uint8(sb + 1)"),
    "cl_fc2_tile": (SB, "t2lo, t2hi = 28 * rank // 6, 28 * (rank + 1) // 6",
                    "t2lo, t2hi = 28 * rank // 6, 28 * (rank + 1) // 6 - 1"),
    "cl_hidden_half": (SB, "sb = hidden_exponent(max(amax, maxima[ew ^ 1]))", "sb = hidden_exponent(amax)"),
    "pe_situ_swap": (PE, "h[t] = situ(gate, values[t], *scalars) if up else Float32(0)",
                     "h[t] = situ(values[t], gate, *scalars) if up else Float32(0)"),
    "pe_drop_partial": (PE, "values[t] += bits_float(words[t])", "values[t] += Float32(0) * bits_float(words[t])"),
    "pe_hidden_half": (PE, "cute.arch.fmax(maxima[t], peer), floor=self.activation_floor",
                       "maxima[t], floor=self.activation_floor"),
    "pe_xscale": (PE, "sm[self.off_xsf + kb // 4 * 128 + token * 16 + kb % 4] = Uint8(scale)",
                  "sm[self.off_xsf + kb // 4 * 128 + token * 16 + kb % 4] = Uint8(scale + 1)"),
}

needs_gpu = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 10,
    reason="needs an SM100-family GPU",
)

T1_FRO, T1_ROW = 1e-2, 3e-2
T2_REL, T2_ADD, T2_MAX = 1.25, 1e-3, 2.0

_STATE = {}


def weights():
    return R.shared_weights()


def latent(batch, g):
    # FP32 latent rows of K3.1-like magnitude (front output ~N(0, 1)), a few outliers.
    x = torch.randn((batch, R.H_LAT), generator=g, device="cuda") * 1.0
    x[:, :: 97] *= 8.0
    return x.contiguous()


def check_rows(name, rows, ref):
    """T1 on BF16 rows [B, 16, 3584] vs FP64 reference."""
    assert bool(torch.isfinite(rows.float()).all()), f"{name}: non-finite rows"
    fro = R.rel_l2(rows, ref)
    per_row = ((rows.double() - ref).norm(dim=-1) / ref.norm(dim=-1)).flatten()
    worst = float(per_row.max())
    stats = {"kind": "check", "item": name, "t1_fro": fro, "t1_row_max": worst, "rows": per_row.numel()}
    print("CHECK", stats, flush=True)
    H.record(stats)
    assert fro <= T1_FRO and worst <= T1_ROW, f"{name}: T1 fro {fro:.3e} (<= {T1_FRO}) row max {worst:.3e} (<= {T1_ROW})"


def check_sum(name, rows, w, ref_rows, fi_out):
    """T2: weighted sum vs reference, against FlashInfer."""
    ref = R.weighted(ref_rows, w)
    mk = R.weighted(rows.float(), w)
    rel_mk, rel_fi = R.rel_l2(mk, ref), R.rel_l2(fi_out, ref)
    max_mk = float((mk - ref).abs().max())
    max_fi = float((fi_out.double() - ref).abs().max())
    rms = float(ref.pow(2).mean().sqrt())
    stats = {"kind": "check", "item": name, "t2_rel_mk": rel_mk, "t2_rel_fi": rel_fi,
             "t2_max_mk": max_mk, "t2_max_fi": max_fi, "ref_rms": rms}
    print("CHECK", stats, flush=True)
    H.record(stats)
    assert torch.isfinite(fi_out.float()).all(), f"{name}: FlashInfer output non-finite"
    assert rel_mk <= T2_REL * rel_fi + T2_ADD, f"{name}: T2 rel {rel_mk:.3e} vs FI {rel_fi:.3e}"
    assert max_mk <= T2_MAX * max_fi + rms / 256, f"{name}: T2 max {max_mk:.3e} vs FI {max_fi:.3e}"


def fi_call(prepared, x, ids, w):
    xq, xs = R.fi_inputs(x)
    out = torch.empty((x.shape[0], R.H_LAT), dtype=torch.bfloat16, device="cuda")
    return R.fi_routed(prepared, xq, xs, ids, w, out), out


class Cluster:
    """ClusterExpertsTP8 launched exactly as ``SmallBatchChain`` does (``coarse``: the
    opt-in ClusterExpertsTP8Coarse with its own two maps)."""

    def __init__(self, experts, coarse=False):
        from sgl_nvidia_mega.kimi_k31.k3.cutedsl.common.host import stream
        from sgl_nvidia_mega.kimi_k31.k3.cutedsl.moe.small_batch import chain as CH
        from sgl_nvidia_mega.kimi_k31.k3.cutedsl.moe.small_batch import experts as SBE

        self.CH, self.stream, self.experts = CH, stream, experts
        self.scales = tuple(experts.weights[i].view(torch.uint8).reshape(-1) for i in (1, 3))
        self.kind = "experts_tp8_coarse" if coarse else "experts_tp8"
        self.maps = SBE.coarse_maps(experts).view(2, 128) if coarse else experts.maps[256:640].view(3, 128)
        self.counter = torch.zeros(1, dtype=torch.int32, device="cuda")

    def bind(self, x, ids, out):
        tensors = (x.reshape(-1), ids.reshape(-1), self.maps, *self.scales, out, self.counter)
        code, args = self.CH._launch(self.kind, tensors)
        self.code = code
        args = args + self.experts.scalars
        return lambda: code(*args, self.stream(x.device))


def persistent(prepared, batches):
    from sgl_nvidia_mega.kimi_k31.k3.cutedsl.moe import experts_runtime as ER

    ex = ER.PersistentExperts(*prepared, beta=R.BETA, linear_beta=R.LIN, activation_floor=None)
    buffers = ER.scratch(8, torch.device("cuda"))
    outs = {}
    for b in batches:
        outs[b] = torch.empty((b * 16, R.H_LAT), dtype=torch.bfloat16, device="cuda")
        ex.prepare(b, buffers, outs[b])
    return ex, outs


def graph_of(fn):
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        fn()
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g, stream=s):
        fn()
    torch.cuda.synchronize()
    return g


@needs_gpu
def test_cluster_tp8_b1():
    plugin = H.load_plugin(MUTATIONS)  # noqa: F841
    from sgl_nvidia_mega.kimi_k31.k3.cutedsl.moe import experts_runtime as ER

    logical, prepared = weights()
    ex = ER.PersistentExperts(*prepared, beta=R.BETA, linear_beta=R.LIN)
    cl = Cluster(ex)
    g = H.generator(7)
    x = torch.empty((1, R.H_LAT), device="cuda")
    ids = torch.empty((1, 16), dtype=torch.int32, device="cuda")
    out = torch.empty((16, R.H_LAT), dtype=torch.bfloat16, device="cuda")
    run = cl.bind(x, ids, out)
    graph = None
    for call in range(4):  # 0-1 eager, 2-3 graph replay with new inputs
        x.copy_(latent(1, g))
        i, w = R.random_route(1, g, distinct=True)
        ids.copy_(i)
        H.poison(out)
        if call < 2:
            run()
        else:
            graph = graph or graph_of(run)
            H.poison(out)
            graph.replay()
        torch.cuda.synchronize()
        ref = R.expert_rows(logical, x, ids)
        check_rows(f"cluster_tp8.b1.call{call}", out.view(1, 16, -1), ref)
        fi, fo = fi_call(prepared, x, ids, w)
        fi()
        torch.cuda.synchronize()
        check_sum(f"cluster_tp8.b1.call{call}.sum", out.view(1, 16, -1), w, ref, fo)
    res = H.function_resources(cl.code)
    H.record({"kind": "resources", "item": "cluster_tp8.b1", "grid": 96, "threads": 256, **res, **H.device_facts()})

    if os.environ.get("K31_TIME") == "1" and not os.environ.get("K31_MUTATION"):
        time_b1(prepared, cl)


def time_b1(prepared, cl):
    """B1: 56 calls, each on 16 experts no other call in the cycle uses (896 = 56 x 16)."""
    g = H.generator(70)
    perm = torch.randperm(R.E_ALL, generator=g, device="cuda").int().view(56, 16)
    xs = [latent(1, g) for _ in range(56)]
    w = [R.random_route(1, g)[1] for _ in range(56)]
    outs = [torch.empty((16, R.H_LAT), dtype=torch.bfloat16, device="cuda") for _ in range(56)]
    ids = [perm[i].view(1, 16).contiguous() for i in range(56)]
    mk = [cl.bind(xs[i], ids[i], outs[i]) for i in range(56)]
    fi = [fi_call(prepared, xs[i], ids[i], w[i])[0] for i in range(56)]
    abba(f"experts.b1", {"mk_cluster": mk, "fi": fi}, R.floor_us(16))


def abba(item, arms, floor):
    names = list(arms)
    order = names + names[::-1]
    got = {n: [] for n in names}
    for n in order:
        got[n].append(H.cold_time_us(arms[n]))
    for n in names:
        meds = [t["us_median"] for t in got[n]]
        row = {"kind": "time", "item": f"{item}.{n}", "us_median": sum(meds) / len(meds),
               "us_runs": meds, "us_min": min(t["us_min"] for t in got[n]), **floor}
        row["tbps_eff"] = floor["bytes"] / (row["us_median"] * 1e-6) / 1e12
        H.record(row)
        print("TIME", row, flush=True)


@needs_gpu
@pytest.mark.parametrize("batch", (1, 2, 3, 4, 5, 8), ids=lambda b: f"b{b}")
def test_persistent(batch):
    H.load_plugin(MUTATIONS)
    logical, prepared = weights()
    ex, outs = persistent(prepared, (batch,))
    out = outs[batch]
    g = H.generator(100 + batch)
    x = torch.empty((batch, R.H_LAT), device="cuda")
    ids = torch.empty((batch, 16), dtype=torch.int32, device="cuda")
    run = lambda: ex.run(x, ids)  # noqa: E731
    graph = None
    for call in range(4):
        x.copy_(latent(batch, g))
        i, w = R.random_route(batch, g)
        ids.copy_(i)
        H.poison(out)
        if call < 2:
            run()
        else:
            graph = graph or graph_of(run)
            H.poison(out)
            graph.replay()
        torch.cuda.synchronize()
        ref = R.expert_rows(logical, x, ids)
        rows = out.view(batch, 16, -1)
        check_rows(f"persistent.b{batch}.call{call}", rows, ref)
        fi, fo = fi_call(prepared, x, ids, w)
        fi()
        torch.cuda.synchronize()
        check_sum(f"persistent.b{batch}.call{call}.sum", rows, w, ref, fo)
    H.record({"kind": "resources", "item": f"persistent.b{batch}", "grid": ex.grid_for(batch), "threads": 384,
              "distinct": int(torch.unique(ids).numel()), **H.device_facts()})

    if os.environ.get("K31_TIME") == "1" and not os.environ.get("K31_MUTATION") and batch in (1, 8):
        n = 56 if batch == 1 else 16
        gt = H.generator(800 + batch)
        xs = [latent(batch, gt) for _ in range(n)]
        if batch == 1:
            perm = torch.randperm(R.E_ALL, generator=gt, device="cuda").int().view(56, 1, 16)
            routes = [(perm[i].contiguous(), R.random_route(1, gt)[1]) for i in range(n)]
        else:
            # Calls cycle over 4 disjoint pools of 224 experts (no L2 reuse between
            # consecutive calls); 8 tokens x 16 of 224 touch ~100 distinct experts.
            pools = torch.randperm(R.E_ALL, generator=gt, device="cuda").view(4, 224)
            routes = [R.random_route(batch, gt, pool=pools[i % 4]) for i in range(n)]
        distinct = sum(int(torch.unique(r[0]).numel()) for r in routes) / n
        # One prepared instance, distinct inputs per call (outputs alias; the timing does not read them).
        mk = [lambda x=xs[i], r=routes[i]: ex.run(x, r[0]) for i in range(n)]
        fi = [fi_call(prepared, xs[i], routes[i][0], routes[i][1])[0] for i in range(n)]
        fl = R.floor_us(round(distinct))
        fl["distinct_mean"] = distinct
        abba(f"experts.b{batch}", {"mk_persistent": mk, "fi": fi}, fl)


GB300_SMS = 152  # GB300's SM count (public); the first arm of test_persistent_grid


@needs_gpu
@pytest.mark.parametrize("batch", (1, 8), ids=lambda b: f"b{b}")
def test_persistent_grid(batch):
    """Grid sweep (timing experiment, unmutated runs only): PersistentExperts at the grids
    derived for GB300's SM count (GB300_SMS), for this device's and for 16 more SMs,
    via K31MK_EXPERTS_GRID_SMS. Each grid is first checked with T1 on
    one call (a grid change must not change correctness)."""
    if os.environ.get("K31_TIME") != "1" or os.environ.get("K31_MUTATION"):
        pytest.skip("timing experiment")
    H.load_plugin(MUTATIONS)
    logical, prepared = weights()
    gt = H.generator(900 + batch)
    n = 56 if batch == 1 else 16
    xs = [latent(batch, gt) for _ in range(n)]
    if batch == 1:
        perm = torch.randperm(R.E_ALL, generator=gt, device="cuda").int().view(56, 1, 16)
        routes = [(perm[i].contiguous(), R.random_route(1, gt)[1]) for i in range(n)]
    else:
        pools = torch.randperm(R.E_ALL, generator=gt, device="cuda").view(4, 224)
        routes = [R.random_route(batch, gt, pool=pools[i % 4]) for i in range(n)]
    distinct = sum(int(torch.unique(r[0]).numel()) for r in routes) / n
    arms = {}
    old = os.environ.get("K31MK_EXPERTS_GRID_SMS")
    try:
        device_sms = torch.cuda.get_device_properties(0).multi_processor_count
        for sms in (GB300_SMS, device_sms, device_sms + 16):
            os.environ["K31MK_EXPERTS_GRID_SMS"] = str(sms)
            ex, outs = persistent(prepared, (batch,))
            grid = ex.grid_for(batch)
            H.poison(outs[batch])
            ex.run(xs[0], routes[0][0])
            torch.cuda.synchronize()
            ref = R.expert_rows(logical, xs[0], routes[0][0])
            check_rows(f"persistent_grid{grid}.b{batch}", outs[batch].view(batch, 16, -1), ref)
            arms[f"grid{grid}"] = [lambda ex=ex, x=xs[i], r=routes[i]: ex.run(x, r[0]) for i in range(n)]
    finally:
        if old is None:
            os.environ.pop("K31MK_EXPERTS_GRID_SMS", None)
        else:
            os.environ["K31MK_EXPERTS_GRID_SMS"] = old
    fl = R.floor_us(round(distinct))
    fl["distinct_mean"] = distinct
    abba(f"experts_grid.b{batch}", arms, fl)


@needs_gpu
def test_cluster_coarse_b1():
    """Opt-in ClusterExpertsTP8Coarse (port): bit-identical rows to the shipped
    kernel on the same inputs (same MMA order) and T1; timing ABBA vs shipped."""
    H.load_plugin(MUTATIONS)
    from sgl_nvidia_mega.kimi_k31.k3.cutedsl.moe.small_batch import experts as SBE

    if not hasattr(SBE, "coarse_maps"):
        pytest.skip("opt-in ClusterExpertsTP8Coarse is not in this build")
    from sgl_nvidia_mega.kimi_k31.k3.cutedsl.moe import experts_runtime as ER

    logical, prepared = weights()
    ex = ER.PersistentExperts(*prepared, beta=R.BETA, linear_beta=R.LIN)
    ship, coarse = Cluster(ex), Cluster(ex, coarse=True)
    g = H.generator(77)
    x = torch.empty((1, R.H_LAT), device="cuda")
    ids = torch.empty((1, 16), dtype=torch.int32, device="cuda")
    outs = [torch.empty((16, R.H_LAT), dtype=torch.bfloat16, device="cuda") for _ in range(2)]
    runs = [ship.bind(x, ids, outs[0]), coarse.bind(x, ids, outs[1])]
    graph = None
    for call in range(4):
        x.copy_(latent(1, g))
        ids.copy_(R.random_route(1, g, distinct=True)[0])
        for o in outs:
            H.poison(o)
        if call < 2:
            runs[0]()
            runs[1]()
        else:
            graph = graph or graph_of(lambda: (runs[0](), runs[1]()))
            for o in outs:
                H.poison(o)
            graph.replay()
        torch.cuda.synchronize()
        ref = R.expert_rows(logical, x, ids)
        check_rows(f"cluster_coarse.b1.call{call}", outs[1].view(1, 16, -1), ref)
        same = bool(torch.equal(outs[0].view(torch.int16), outs[1].view(torch.int16)))
        H.record({"kind": "check", "item": f"cluster_coarse.b1.call{call}.bitwise", "equal": same})
        assert same, f"coarse rows differ from the shipped kernel (call {call})"
    H.record({"kind": "resources", "item": "cluster_coarse.b1", **H.function_resources(coarse.code)})
    if os.environ.get("K31_TIME") == "1" and not os.environ.get("K31_MUTATION"):
        gt = H.generator(71)
        perm = torch.randperm(R.E_ALL, generator=gt, device="cuda").int().view(56, 16)
        xs = [latent(1, gt) for _ in range(56)]
        oo = [torch.empty((16, R.H_LAT), dtype=torch.bfloat16, device="cuda") for _ in range(56)]
        idl = [perm[i].view(1, 16).contiguous() for i in range(56)]
        arms = {"shipped": [ship.bind(xs[i], idl[i], oo[i]) for i in range(56)],
                "coarse": [coarse.bind(xs[i], idl[i], oo[i]) for i in range(56)]}
        abba("experts_coarse.b1", arms, R.floor_us(16))
