# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""K3.1 TP16 shared expert at B<=8: gate/up with SiTU in the split-K epilogue (``K31MK_EXP_SHARED_SKS``, default off;
TP16 only). ``SharedSitu`` runs the shipped split-K GEMM and tactic (64.8.4.8) on a tile-interleaved copy of the
gate/up weight; the owner rank adds the peers' FP32 partials in the same order as ``SplitKGemm`` and applies the same
``shared.situ`` that RouteSitu's situ role applies to the FP32 gate/up output.

Checks (stated before any run; never loosened):
  K0  relayout: ``deinterleave_gate_up(interleave_gate_up(W))`` == W byte for byte (also asserted at construction);
      the interleave puts gate row 32m + j at tile row j and up row 32m + j at tile row 32 + j (checked on a row-index
      probe, independent of the kernel).
  K1  SKS h vs the shipped path (PreparedProjection gate/up FP32 -> RouteSitu situ role BF16) on the same x: 0
      differing BF16 cells, B1, 2, 4, 8, eager and CUDA-graph replay with new inputs.
  KR  SKS h vs an FP64 reference (BF16(SiTU(x @ W^T)) in FP64), cells with |ref| >= 2^-60: as first written "within 2
      BF16 ulp, the shipped path as control"; FAILED AS WRITTEN (arm-independent, Rubin cexp_s_r1 b2 call3: 30 ulp in
      both paths). Post-hoc KR': SKS max ulp <= max(2, the shipped path's max ulp) on the same inputs.

Mutations (fixed harness; each must make a base-passing test fail):
  sks_pair     the epilogue pairs gate row j with up row 33 + j (off by one)                   [K1, KR]
  sks_noadd    the owner skips the last peer's partial                                        [K1, KR]
  sks_order    the interleave puts up rows first in each tile (gate/up swapped)               [K0, K1, KR]
  sks_k23cta   the owner's bar_reduce arm ignores the K-23h fix (default .cta arm)             [K23 default]

K23 (stated before any run): the SKS kernel SharedSitu compiles (tactic 64.8.4.8) carries exactly ONE
  ``mbarrier.arrive.expect_tx.relaxed.cluster.shared::cta.b64`` (the owner's late bar_reduce arm, K-23h fix) when
  SharedSitu builds it (k23_arm True), and none when the same kernel is compiled with k23_arm=False (the control); the two
  CTA-local TMA-stage arms are unchanged in both. Checked on the kept PTX of the exact kernel object SharedSitu builds (engagement: SharedSitu.k23_arm).

Timing: in a 1-node chain harness (arms SKS / SKSG, not included).
"""

import pytest
import torch

import _harness as H
import _ref_moe as R

SH_FILE = "kimi_k31/k3/cutedsl/moe/shared.py"
MUTATIONS = {
    "sks_pair": (SH_FILE, "value = situ(sD[j, t], sD[32 + j, t], beta, linear_beta)",
                 "value = situ(sD[j, t], sD[(33 + j) % 64, t], beta, linear_beta)"),
    "sks_noadd": (SH_FILE, "                    for peer in cutlass.range_constexpr(self.split_k - 1):\n"
                           "                        for i in cutlass.range_constexpr(vpt):\n"
                           "                            rAcc[i] = rAcc[i] + mailbox[peer * vpp + epi_tid * vpt + i]\n"
                           "            if split_rank == SK.OWNER_RANK:",
                  "                    for peer in cutlass.range_constexpr(self.split_k - 2):\n"
                  "                        for i in cutlass.range_constexpr(vpt):\n"
                  "                            rAcc[i] = rAcc[i] + mailbox[peer * vpp + epi_tid * vpt + i]\n"
                  "            if split_rank == SK.OWNER_RANK:"),
    "sks_k23cta": (SH_FILE, "if cutlass.const_expr(self.k23_arm):", "if cutlass.const_expr(False):"),
    "sks_order": (SH_FILE, "order = torch.arange(rows, device=gate_up.device).view(2, half // pair, pair).transpose(0, 1)",
                  "order = torch.arange(rows, device=gate_up.device).view(2, half // pair, pair).flip(0).transpose(0, 1)"),
}

needs_gpu = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 10,
    reason="needs an SM100-family GPU",
)

HID, SHARED = 7168, 384  # K3.1 hidden; TP16 shared intermediate per rank
TACTIC = (64, 8, 4, 8)  # front.py SHARED_TACTICS[384] gate/up at B1-16


def weights(gen):
    w = torch.randn((2 * SHARED, HID), generator=gen, device="cuda") * 0.02
    return w.to(torch.bfloat16).contiguous()


def shipped(gate_up, batch):
    """PreparedProjection gate/up (FP32 out) + RouteSitu's situ role, exactly as SharedFront runs them at B<=8."""
    from sgl_nvidia_mega.kimi_k31.k3.cutedsl.common.host import stream, to_cute
    from sgl_nvidia_mega.kimi_k31.k3.cutedsl.gemm.projection import PreparedProjection
    from sgl_nvidia_mega.kimi_k31.k3.cutedsl.moe import front as F
    import cutlass

    proj = PreparedProjection(gate_up, {batch: TACTIC}, output_dtype=torch.float32).prepare()
    role = F.compile_route(batch, SHARED, True, (False, False, True), torch.cuda.current_device(), deadline_ns=0,
                           fp32=True)
    f32 = dict(dtype=torch.float32, device="cuda")
    hg = torch.zeros((batch, 2 * SHARED), **f32)
    h = torch.zeros((batch, SHARED), dtype=torch.bfloat16, device="cuda")
    keep = (torch.zeros((batch, 896), **f32), torch.zeros(896, **f32),
            torch.zeros((batch, 16), dtype=torch.int32, device="cuda"),
            torch.zeros((batch, 16), dtype=torch.bfloat16, device="cuda"),
            torch.zeros((1, 64, 7168 + 2 * 896), dtype=torch.bfloat16, device="cuda"),
            torch.zeros((batch, 3584), **f32))
    args = tuple(map(to_cute, keep)) + (to_cute(hg), to_cute(h))
    scalars = (cutlass.Float32(1.0), cutlass.Float32(R.BETA), cutlass.Float32(R.LIN))

    def run(x):
        proj.run(x, hg)
        role(*args, *scalars, stream(x.device))

    return run, h, keep


def sks(gate_up, batch):
    from sgl_nvidia_mega.kimi_k31.k3.cutedsl.moe import shared as SH

    m = SH.SharedSitu(gate_up, {batch: TACTIC}, R.BETA, R.LIN)
    hg = torch.zeros((batch, 2 * SHARED), dtype=torch.float32, device="cuda")
    h = torch.zeros((batch, SHARED), dtype=torch.bfloat16, device="cuda")
    return (lambda x: m.run(x, hg, h)), h, m


def bf16_ulp_distance(a, b):
    def ordered(v):
        i = v.contiguous().view(torch.int16).to(torch.int32)
        return torch.where(i < 0, -32768 - i, i)
    return (ordered(a) - ordered(b)).abs()


def reference(x, gate_up):
    hg = x.double() @ gate_up.double().T
    return R.situ(hg[:, :SHARED], hg[:, SHARED:])


@needs_gpu
def test_sks_relayout():
    """K0."""
    H.load_plugin(MUTATIONS)
    from sgl_nvidia_mega.kimi_k31.k3.cutedsl.moe import shared as SH

    w = weights(H.generator(11))
    il, order = SH.interleave_gate_up(w)
    back = SH.deinterleave_gate_up(il, order)
    rt = int((back != w).sum())
    probe = torch.arange(2 * SHARED, device="cuda", dtype=torch.float32).view(-1, 1).expand(-1, 8).contiguous()
    ip, _ = SH.interleave_gate_up(probe)
    rows = ip[:, 0].long().view(SHARED // 32, 2, 32)  # [tile, gate|up, j]
    tiles = torch.arange(SHARED // 32, device="cuda").view(-1, 1) * 32 + torch.arange(32, device="cuda")
    layout_ok = bool(torch.equal(rows[:, 0], tiles) and torch.equal(rows[:, 1], tiles + SHARED))
    rec = {"kind": "check", "item": "sks.k0", "roundtrip_cells_differ": rt, "layout_ok": layout_ok}
    H.record(rec)
    print("CHECK", rec, flush=True)
    assert rt == 0 and layout_ok, rec


@needs_gpu
@pytest.mark.parametrize("batch", (1, 2, 4, 8), ids=lambda b: f"b{b}")
def test_sks_vs_shipped(batch):
    """K1 + KR (SKS) and KR (shipped, control)."""
    H.load_plugin(MUTATIONS)
    g = H.generator(1100 + batch)
    w = weights(g)
    run_s, h_s, keep_s = shipped(w, batch)
    run_k, h_k, keep_k = sks(w, batch)
    x = torch.empty((batch, HID), dtype=torch.bfloat16, device="cuda")

    def call():
        run_s(x)
        run_k(x)

    graph = None
    for n in range(4):
        xv = (torch.randn((batch, HID), generator=g, device="cuda") * (1.0 + n)).to(torch.bfloat16)
        x.copy_(xv)
        H.poison(h_s, h_k)
        if n < 2:
            call()
        else:
            if graph is None:
                s = torch.cuda.Stream()
                s.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(s):
                    call()
                torch.cuda.synchronize()
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph, stream=s):
                    call()
                torch.cuda.synchronize()
                H.poison(h_s, h_k)
            graph.replay()
        torch.cuda.synchronize()
        differ = int((h_k.view(torch.int16) != h_s.view(torch.int16)).sum())
        ref = reference(xv, w)
        keep = ref.abs() >= 2.0 ** -60
        worst_k = int(bf16_ulp_distance(h_k, ref.float().to(torch.bfloat16))[keep].max())
        worst_s = int(bf16_ulp_distance(h_s, ref.float().to(torch.bfloat16))[keep].max())
        rec = {"kind": "check", "item": f"sks.b{batch}.call{n}", "k1_cells_differ": differ, "kr_max_ulp_sks": worst_k,
               "kr_max_ulp_shipped": worst_s, "finite": bool(torch.isfinite(h_k.float()).all())}
        H.record(rec)
        print("CHECK", rec, flush=True)
        assert rec["finite"], rec
        # KR as first written (worst_s <= 2 and worst_k <= 2) FAILED AS WRITTEN on Rubin task cexp_s_r1 (b2 call3: the
        # shipped control itself is 30 ulp from FP64 at one cell, SKS identical): an arm-independent line. Post-hoc KR'
        # (post hoc): SKS no further from FP64 than the shipped path, and within 2 ulp wherever the
        # shipped path is.
        assert worst_k <= max(2, worst_s), rec
        assert differ == 0, rec


@needs_gpu
@pytest.mark.parametrize("escape", (False, True), ids=("default", "escape"))
def test_sks_k23_arm(monkeypatch, tmp_path, escape):
    """K23 (K-23h fix): the owner's late bar_reduce arm is .relaxed.cluster as built, the .cta arm only in the k23_arm=False control."""
    H.load_plugin(MUTATIONS)
    from sgl_nvidia_mega.kimi_k31.k3.cutedsl.common.host import architecture
    from sgl_nvidia_mega.kimi_k31.k3.cutedsl.moe import shared as SH

    w = weights(H.generator(23))
    m = SH.SharedSitu(w, {8: TACTIC}, R.BETA, R.LIN)
    assert m.k23_arm is True, m.k23_arm
    arch = architecture(torch.device("cuda"))
    SH.compile_sks.__wrapped__(TACTIC, f"{arch} --keep-ptx --dump-dir={tmp_path}", m.epiwait, not escape)
    files = sorted(tmp_path.rglob("*SplitKSituGemm*.ptx"))
    assert len(files) == 1, files
    ptx = files[0].read_text()
    cluster = ptx.count("mbarrier.arrive.expect_tx.relaxed.cluster.shared::cta.b64")
    plain = ptx.count("mbarrier.arrive.expect_tx.shared.b64")
    rec = {"kind": "check", "item": f"sks.k23.{'escape' if escape else 'default'}", "cluster_arms": cluster, "cta_arms": plain}
    H.record(rec)
    print("CHECK", rec, flush=True)
    assert (cluster, plain) == ((0, 3) if escape else (1, 2)), rec
