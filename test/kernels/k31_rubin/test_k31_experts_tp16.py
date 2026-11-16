# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""K3.1 routed experts on one GPU at the TP16 per-rank shapes on the FP32 latent:
``ClusterExpertsTP16`` (TP16 B1/B2 small-batch chain) and ``PersistentExperts`` at TP16 (the generic path, B1-B16;
B16 runs the grouped plan written by ``ExpertsPlanKernel`` from FP32). 896 experts, I = 192 per rank padded to 256
in the host's FlashInfer layout (``_ref_moe16``), MXFP4 weights x MXFP8 activations, SiTU 4 / 25.

Independent references: ``_ref_moe16`` (FP64 torch of the stock method from the logical, unpadded weights:
FP32 latent -> MXFP8 -> FC1 -> SiTU -> MXFP8 hidden -> FC2), and stock FlashInfer
``trtllm_fp4_block_scale_routed_moe`` on the padded weights and the reference MXFP8 of the same FP32 latent.

Tolerances (stated before any run; never loosened after):
  T1  expert rows (BF16) vs the FP64 reference: as test_k31_experts.py (Frobenius rel-L2 <= 1e-2, every row
      <= 3e-2, every cell finite; outputs NaN-poisoned before each call). Same basis.
  T2  weighted sum vs the reference, against FlashInfer on the same inputs: as test_k31_experts.py
      (rel <= 1.25 rel_FI + 1e-3; max <= 2 max_FI + rms/256).
  B   bitwise, quantized activations: the kernel's rows for x and for x' = the reference MXFP8 of x (FP32 values,
      exactly representable, -0 made +0) are bit-identical, on every call (eager and CUDA-graph replay).
      Basis: re-quantizing x' with the stock rule gives the same values (the same codes and scale, or codes x2 at
      scale / 2 when a block maximum rounds down to 224 x 2^s), so identical rows <=> the kernel's MXFP8 of x has
      the reference's values. Quantizing BF16-rounded x instead flips ~1/32 of the E4M3 codes (a BF16 neighbour
      lands on an E4M3 rounding midpoint), each moving its FC1 terms by one E4M3 step: the rows change bits.
  P   bitwise, plan: ``ExpertsPlanKernel`` on FP32 x writes E4M3 bytes and UE8M0 scale bytes equal to
      ``_ref_moe16.mx8_bytes(x)`` (the stock rule) byte for byte.
  X   TP8 cross-check: one TP8 rank's expert rows (``ClusterExpertsTP8`` B1, ``PersistentExperts`` TP8 B8) vs
      the FP64 sum of the two TP16 ranks that own its intermediate halves (``_ref_moe16.tp8_halves``), same FP32 x
      and ids: Frobenius rel-L2 <= 5e-3 and every row <= 2e-2.
      Basis: the same FP32 x and MXFP8 rule, the same SiTU code, hidden MXFP8 blocks aligned with the halves
      (192 = 6 x 32); only FC1 accumulation order (~1e-7 rel) and one more BF16 output rounding (2^-9 rel per
      half) differ, plus rare hidden E4M3 flips (~0.3% on that row).

Mutations (each must make the suite fail; the bug each stands for):
  c16_bf16_x       ClusterExpertsTP16 rounds the FP32 latent to BF16 before MXFP8 (the pre-A4 precision)  [B, X]
  c16_xscale_drop  ClusterExpertsTP16 writes scale 2^0 instead of the block scale (dropped x scale)       [T1]
  c16_block_off    ClusterExpertsTP16 stores block b's scale in block b+1's slot (off-by-one block)      [T1]
  pe_bf16_x        PersistentExperts rounds FP32 x to BF16 before MXFP8 (B<=8 path)                      [B]
  plan_bf16        the FP32 plan writer rounds x to BF16 before MXFP8 (B>8 path)                          [P, B]
Run each mutation on the tests that cover its code (``run_a4.sh`` passes -k). B runs before T1 and X before T1, so a
mutation's failure names the first of them that catches it. Rows more than 5e-3 from the FP64 reference get a
recorded control (stock FlashInfer's row of the same expert and token; ``diagnose``) that never decides pass/fail.
Base passes; every mutation is caught by the checks in brackets.
"""

import pytest
import torch

import _harness as H
import _ref_moe as R
import _ref_moe16 as M

SB = "kimi_k31/k3/cutedsl/moe/small_batch/experts.py"
PE = "kimi_k31/k3/cutedsl/moe/experts.py"
_TO_BF16 = "Uint32(float_bits(Float32(cutlass.BFloat16(bits_float(packed[t, q])))))"
MUTATIONS = {
    "c16_bf16_x": (SB, "                sb, words = quantize_mx8(packed, t, fp32=True)\n",
                   "                for q in cutlass.range_constexpr(8):\n"
                   f"                    packed[t, q] = {_TO_BF16}\n"
                   "                sb, words = quantize_mx8(packed, t, fp32=True)\n"),
    "c16_xscale_drop": (SB, "sm[self.off_xsf + kb // 4 * 128 + 16 * t + kb % 4] = Uint8(sb)",
                        "sm[self.off_xsf + kb // 4 * 128 + 16 * t + kb % 4] = Uint8(127)"),
    "c16_block_off": (SB, "kc, kb = 16 * n + 2 * block + k // 2, 8 * n + block",
                      "kc, kb = 16 * n + 2 * block + k // 2, (8 * n + block + 1) % 56"),
    "pe_bf16_x": (PE, "values[i] = bits_float(raw[index, i // 4, i % 4])",
                  "values[i] = Float32(cutlass.BFloat16(bits_float(raw[index, i // 4, i % 4])))"),
    "plan_bf16": (PE, "values[i] = bits_float(words[i])", "values[i] = Float32(cutlass.BFloat16(bits_float(words[i])))"),
}

needs_gpu = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 10,
    reason="needs an SM100-family GPU",
)

T1_FRO, T1_ROW = 1e-2, 3e-2
T2_REL, T2_ADD, T2_MAX = 1.25, 1e-3, 2.0
X_FRO, X_ROW = 5e-3, 2e-2

_W = {}


def weights16():
    """(logical, host-layout) TP16 weights of one rank, built once per process."""
    if "w16" not in _W:
        logical = M.make_logical16(R.E_ALL, H.generator(1631))
        _W["w16"] = logical, M.host_layout16(*logical)
    return _W["w16"]


def weights_x():
    """TP8 rank weights and the two TP16 ranks that own its intermediate halves (logical, host layout)."""
    if "x" not in _W:
        logical8, prepared8 = R.shared_weights()
        halves = [(h, M.host_layout16(*h)) for h in M.tp8_halves(logical8)]
        _W["x"] = (logical8, prepared8), halves
    return _W["x"]


def latent(batch, g):
    """FP32 latent rows of K3.1-like magnitude with a few outliers, -0 made +0 (the mailbox sentinel); blocks
    whose quantized maximum would land on 224 x 2^s get a 240 x 2^s maximum (``_ref_moe16.keep_scale``)."""
    x = torch.randn((batch, R.H_LAT), generator=g, device="cuda")
    x[:, ::97] *= 8.0
    return M.canon(M.keep_scale(x))


def grid_of(x):
    xg = M.on_grid(x)
    M.assert_on_grid(x, xg)
    return xg


def experts_of(prepared):
    """The prepared experts (maps, scales, scalars) without compiling a persistent launch."""
    from sgl_nvidia_mega.kimi_k31.k3.cutedsl.moe import experts_runtime as ER

    return ER.PersistentExperts(*prepared, beta=R.BETA, linear_beta=R.LIN, activation_floor=None)


def persistent(prepared, batch):
    from sgl_nvidia_mega.kimi_k31.k3.cutedsl.moe import experts_runtime as ER

    ex = ER.PersistentExperts(*prepared, beta=R.BETA, linear_beta=R.LIN, activation_floor=None)
    out = torch.empty((batch * 16, R.H_LAT), dtype=torch.bfloat16, device="cuda")
    ex.prepare(batch, ER.scratch(ex.tp, torch.device("cuda")), out)
    return ex, out


class Cluster16:
    """ClusterExpertsTP16 launched exactly as ``SmallBatchChain`` does (FP32 latent rows, maps 0-1, scales)."""

    def __init__(self, ex):
        from sgl_nvidia_mega.kimi_k31.k3.cutedsl.common.host import stream
        from sgl_nvidia_mega.kimi_k31.k3.cutedsl.moe.small_batch import chain as CH

        assert ex.tp == 16
        self.CH, self.stream, self.ex = CH, stream, ex
        self.maps = ex.maps[:256].view(2, 128)
        self.scales = tuple(ex.weights[i].view(torch.uint8).reshape(-1) for i in (1, 3))

    def bind(self, x, ids, out):
        batch = x.shape[0]
        assert x.dtype == torch.float32
        tensors = (x.reshape(-1), ids.reshape(-1), self.maps, *self.scales, out)
        code, args = self.CH._launch("experts_tp16", tensors, batch)
        self.code = code
        return lambda: code(*args, *self.ex.scalars, self.stream(x.device))


class Cluster8:
    """ClusterExpertsTP8 as ``SmallBatchChain`` launches it (FP32 latent row)."""

    def __init__(self, ex):
        from sgl_nvidia_mega.kimi_k31.k3.cutedsl.common.host import stream
        from sgl_nvidia_mega.kimi_k31.k3.cutedsl.moe.small_batch import chain as CH

        assert ex.tp == 8
        self.CH, self.stream, self.ex = CH, stream, ex
        self.maps = ex.maps[256:640].view(3, 128)
        self.scales = tuple(ex.weights[i].view(torch.uint8).reshape(-1) for i in (1, 3))
        self.counter = torch.zeros(1, dtype=torch.int32, device="cuda")

    def bind(self, x, ids, out):
        tensors = (x.reshape(-1), ids.reshape(-1), self.maps, *self.scales, out, self.counter)
        code, args = self.CH._launch("experts_tp8", tensors)
        return lambda: code(*args, *self.ex.scalars, self.stream(x.device))


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


def check_rows(name, rows, ref, fro_max=T1_FRO, row_max=T1_ROW, kind="t1"):
    assert bool(torch.isfinite(rows.float()).all()), f"{name}: non-finite rows"
    fro = R.rel_l2(rows, ref)
    per_row = ((rows.double() - ref.double()).norm(dim=-1) / ref.double().norm(dim=-1)).flatten()
    worst = float(per_row.max())
    stats = {"kind": "check", "item": name, f"{kind}_fro": fro, f"{kind}_row_max": worst, "rows": per_row.numel()}
    print("CHECK", stats, flush=True)
    H.record(stats)
    assert fro <= fro_max and worst <= row_max, f"{name}: {kind} fro {fro:.3e} (<= {fro_max}) row {worst:.3e} (<= {row_max})"


def check_sum(name, rows, w, ref_rows, fi_out):
    ref = R.weighted(ref_rows, w)
    mk = R.weighted(rows.float(), w)
    rel_mk, rel_fi = R.rel_l2(mk, ref), R.rel_l2(fi_out, ref)
    max_mk = float((mk - ref).abs().max())
    max_fi = float((fi_out.double() - ref).abs().max())
    rms = float(ref.pow(2).mean().sqrt())
    stats = {"kind": "check", "item": name, "t2_rel_mk": rel_mk, "t2_rel_fi": rel_fi, "t2_max_mk": max_mk,
             "t2_max_fi": max_fi, "ref_rms": rms}
    print("CHECK", stats, flush=True)
    H.record(stats)
    assert torch.isfinite(fi_out.float()).all(), f"{name}: FlashInfer output non-finite"
    assert rel_mk <= T2_REL * rel_fi + T2_ADD, f"{name}: T2 rel {rel_mk:.3e} vs FI {rel_fi:.3e}"
    assert max_mk <= T2_MAX * max_fi + rms / 256, f"{name}: T2 max {max_mk:.3e} vs FI {max_fi:.3e}"


def fi_row(prepared, x, expert):
    """Stock FlashInfer's row of one expert for one token [3584] (BF16): weight 1 on ``expert``, 0 on 15 others."""
    others = [e for e in range(R.E_ALL) if e != expert][:15]
    ids = torch.tensor([[expert] + others], dtype=torch.int32, device="cuda")
    w = torch.zeros((1, 16), dtype=torch.float32, device="cuda")
    w[0, 0] = 1.0
    xq, xs = R.fi_inputs(x)
    out = torch.empty((1, R.H_LAT), dtype=torch.bfloat16, device="cuda")
    M.fi_routed16(prepared, xq, xs, ids, w, out)()
    torch.cuda.synchronize()
    return out[0].double()


def diagnose(name, logical, prepared, x, ids, rows, ref, limit=5e-3):
    """Control for rows whose rel-L2 to the FP64 reference exceeds ``limit`` (recorded only, no pass/fail): the
    same row from stock FlashInfer (one-hot weights), its error against the FP32-SiTU variant of the reference,
    the smallest hidden-block scale margin and the hidden participation ratio (``_ref_moe16.row_control``)."""
    per_row = ((rows.double() - ref.double()).norm(dim=-1) / ref.double().norm(dim=-1))
    for t, j in (per_row > limit).nonzero().tolist():
        expert = int(ids[t, j])
        variants, margin = M.row_control(logical, x, t, expert)
        got, fi = rows[t, j].double(), fi_row(prepared, x[t: t + 1], expert)
        rel = lambda a, b: float((a - b).norm() / b.norm())  # noqa: E731
        rec = {"kind": "control", "item": f"{name}.t{t}.j{j}", "expert": expert,
               "rel_mk_fp64_ref": rel(got, variants["fp64"]), "rel_mk_fp32_situ_ref": rel(got, variants["fp32"]),
               "rel_fi_fp64_ref": rel(fi, variants["fp64"]), "rel_mk_fi": rel(got, fi),
               "row_rms_over_call_rms": float(variants["fp64"].pow(2).mean().sqrt() / ref.double().pow(2).mean().sqrt()),
               "min_hidden_scale_margin_log2": margin}
        H.record(rec)
        print("CONTROL", rec, flush=True)


def check_bitwise(name, a, b):
    same = bool(torch.equal(a.view(torch.int16), b.view(torch.int16)))
    differ = int((a.view(torch.int16) != b.view(torch.int16)).sum())
    H.record({"kind": "check", "item": name, "bitwise_x_vs_grid": same, "cells_differ": differ})
    print("CHECK", name, "bitwise_x_vs_grid", same, differ, flush=True)
    assert same, f"{name}: rows on x and on its reference MXFP8 grid differ in {differ} cells"


def fi_check(name, prepared, x, ids, w, rows, ref):
    xq, xs = R.fi_inputs(x)
    out = torch.empty((x.shape[0], R.H_LAT), dtype=torch.bfloat16, device="cuda")
    M.fi_routed16(prepared, xq, xs, ids, w, out)()
    torch.cuda.synchronize()
    check_sum(name, rows, w, ref, out)


@needs_gpu
@pytest.mark.parametrize("batch", (1, 2), ids=lambda b: f"b{b}")
def test_cluster16(batch):
    """T1, T2 and B on ClusterExpertsTP16 (B1/B2): two eager calls, then two CUDA-graph replays with new inputs."""
    H.load_plugin(MUTATIONS)
    logical, prepared = weights16()
    cl = Cluster16(experts_of(prepared))
    g = H.generator(160 + batch)
    src = {k: torch.empty((batch, R.H_LAT), device="cuda") for k in ("x", "grid")}
    ids_src = torch.empty((batch, 16), dtype=torch.int32, device="cuda")
    xs = {k: torch.empty((batch, R.H_LAT), device="cuda") for k in ("x", "grid")}
    ids = {k: torch.empty((batch, 16), dtype=torch.int32, device="cuda") for k in ("x", "grid")}
    outs = {k: torch.empty((batch * 16, R.H_LAT), dtype=torch.bfloat16, device="cuda") for k in ("x", "grid")}
    runs = {k: cl.bind(xs[k], ids[k], outs[k]) for k in ("x", "grid")}

    def call():  # the kernel re-arms its ids to -1: refill before every launch
        for k in ("x", "grid"):
            xs[k].copy_(src[k])
            ids[k].copy_(ids_src)
            runs[k]()

    graph = None
    for n in range(4):
        x = latent(batch, g)
        i, w = R.random_route(batch, g, distinct=batch == 1)
        src["x"].copy_(x)
        src["grid"].copy_(grid_of(x))
        ids_src.copy_(i)
        for o in outs.values():
            H.poison(o)
        if n < 2:
            call()
        else:
            graph = graph or graph_of(call)
            for o in outs.values():
                H.poison(o)
            graph.replay()
        torch.cuda.synchronize()
        assert bool((ids["x"] == -1).all()), "ClusterExpertsTP16 did not consume (re-arm) its ids"
        ref = M.expert_rows(logical, x, i)
        rows = outs["x"].view(batch, 16, -1)
        diagnose(f"cluster16.b{batch}.call{n}", logical, prepared, x, i, rows, ref)
        check_bitwise(f"cluster16.b{batch}.call{n}.grid", outs["x"], outs["grid"])
        check_rows(f"cluster16.b{batch}.call{n}", rows, ref)
        fi_check(f"cluster16.b{batch}.call{n}.sum", prepared, x, i, w, rows, ref)
    H.record({"kind": "resources", "item": f"cluster16.b{batch}", **H.function_resources(cl.code),
              **H.device_facts()})


@needs_gpu
@pytest.mark.parametrize("batch", (1, 2, 4, 8, 16), ids=lambda b: f"b{b}")
def test_persistent16(batch):
    """T1, T2 and B on PersistentExperts at TP16 (FP32 x; B16: the FP32 plan writer first)."""
    H.load_plugin(MUTATIONS)
    logical, prepared = weights16()
    ex, out = persistent(prepared, batch)
    g = H.generator(260 + batch)
    xs = {k: torch.empty((batch, R.H_LAT), device="cuda") for k in ("x", "grid")}
    ids = torch.empty((batch, 16), dtype=torch.int32, device="cuda")
    saved = torch.empty_like(out)

    def call():
        ex.run(xs["x"], ids)
        saved.copy_(out)
        ex.run(xs["grid"], ids)

    graph = None
    for n in range(4):
        x = latent(batch, g)
        i, w = R.random_route(batch, g)
        xs["x"].copy_(x)
        xs["grid"].copy_(grid_of(x))
        ids.copy_(i)
        H.poison(out, saved)
        if n < 2:
            call()
        else:
            graph = graph or graph_of(call)
            H.poison(out, saved)
            graph.replay()
        torch.cuda.synchronize()
        ref = M.expert_rows(logical, x, i)
        rows = saved.view(batch, 16, -1)
        diagnose(f"persistent16.b{batch}.call{n}", logical, prepared, x, i, rows, ref)
        check_bitwise(f"persistent16.b{batch}.call{n}.grid", saved, out)
        check_rows(f"persistent16.b{batch}.call{n}", rows, ref)
        fi_check(f"persistent16.b{batch}.call{n}.sum", prepared, x, i, w, rows, ref)
    H.record({"kind": "resources", "item": f"persistent16.b{batch}", "grid": ex.grid_for(batch),
              "distinct": int(torch.unique(ids).numel()), **H.device_facts()})


@needs_gpu
@pytest.mark.parametrize("batch", (1, 16), ids=lambda b: f"b{b}")
def test_plan16(batch):
    """P: the FP32 plan writer's MXFP8 bytes equal the stock rule's, byte for byte."""
    H.load_plugin(MUTATIONS)
    from sgl_nvidia_mega.kimi_k31.k3.cutedsl.common.host import stream, to_cute
    from sgl_nvidia_mega.kimi_k31.k3.cutedsl.moe import experts as EX
    from sgl_nvidia_mega.kimi_k31.k3.cutedsl.moe import experts_runtime as ER

    code = ER.compile_plan(batch, torch.cuda.current_device(), True)
    plan = torch.empty(EX.PLAN_BYTES, dtype=torch.uint8, device="cuda")
    g = H.generator(360 + batch)
    for n in range(3):
        x = latent(batch, g)
        if n == 2:  # small and all-zero blocks: the 1e-10 clamp and zero codes
            x[:, :32] = 0.0
            x[:, 32:64] *= 1e-30
        plan.fill_(0xA5)
        code(to_cute(x, 16), to_cute(plan, 16), stream(x.device))
        torch.cuda.synchronize()
        q_ref, s_ref = M.mx8_bytes(x)
        q = plan[: batch * R.H_LAT].view(batch, R.H_LAT)
        scales = plan[EX.PLAN_XSF: EX.PLAN_XSF + batch * 448].view(batch, 28, 16)[:, :, :4].reshape(batch, 112)
        bad_q, bad_s = int((q != q_ref).sum()), int((scales != s_ref).sum())
        H.record({"kind": "check", "item": f"plan16.b{batch}.call{n}", "bad_codes": bad_q, "bad_scales": bad_s})
        print("CHECK plan16", batch, n, bad_q, bad_s, flush=True)
        assert bad_q == 0 and bad_s == 0, f"plan16.b{batch}.call{n}: {bad_q} E4M3 bytes, {bad_s} scale bytes differ"


@needs_gpu
@pytest.mark.parametrize("kind", ("cluster_b1", "persistent_b8"))
def test_crosscheck_tp8(kind):
    """X: one TP8 rank's rows vs the sum of its two TP16 halves (same FP32 x, ids); T1 on every arm too."""
    H.load_plugin(MUTATIONS)
    (logical8, prepared8), halves = weights_x()
    batch = 1 if kind == "cluster_b1" else 8
    g = H.generator(460 + batch)
    x = torch.empty((batch, R.H_LAT), device="cuda")
    ids = torch.empty((batch, 16), dtype=torch.int32, device="cuda")
    ids_src = torch.empty_like(ids)
    if kind == "cluster_b1":
        out8 = torch.empty((16, R.H_LAT), dtype=torch.bfloat16, device="cuda")
        run8 = Cluster8(experts_of(prepared8)).bind(x, ids, out8)
        outs16, runs16, ids16 = [], [], []
        for _, prep in halves:
            ex16 = experts_of(prep)
            outs16.append(torch.empty((16, R.H_LAT), dtype=torch.bfloat16, device="cuda"))
            ids16.append(torch.empty_like(ids))
            runs16.append(Cluster16(ex16).bind(x, ids16[-1], outs16[-1]))

        def call():
            ids.copy_(ids_src)
            run8()
            for j in range(2):
                ids16[j].copy_(ids_src)
                runs16[j]()
    else:
        ex8, out8 = persistent(prepared8, batch)
        pairs = [persistent(prep, batch) for _, prep in halves]
        outs16 = [o for _, o in pairs]

        def call():
            ids.copy_(ids_src)
            ex8.run(x, ids)
            for ex16, _ in pairs:
                ex16.run(x, ids)
    for n in range(3):
        xv = latent(batch, g)
        i, _ = R.random_route(batch, g, distinct=batch == 1)
        x.copy_(xv)
        ids_src.copy_(i)
        H.poison(out8, *outs16)
        call()
        torch.cuda.synchronize()
        rows8 = out8.view(batch, 16, -1)
        both = outs16[0].double() + outs16[1].double()  # X first, so a mutation shows whether X alone catches it
        check_rows(f"x.{kind}.call{n}.tp8_vs_tp16", both.view(batch, 16, -1), rows8.double(), X_FRO, X_ROW, "x")
        check_rows(f"x.{kind}.call{n}.tp8", rows8, R.expert_rows(logical8, xv, i))
        for j, (half, _) in enumerate(halves):
            check_rows(f"x.{kind}.call{n}.tp16h{j}", outs16[j].view(batch, 16, -1), M.expert_rows(half, xv, i))
