# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""RS (``SplitKRsGemm`` projections (``gemm/splitk_rs.py``, 6-field tactics
128.n.split.stages.owners.cta_k) at the K3.1 TP16 KDA per-rank shape [3216, 7168] and MID rows 65-128, both trigger
modes used today (2 = warp 3 after its wait, fenced; 0 = the shipped weight-warp trigger).

Reference: ``x.double() @ W.double().T`` (float64, independent of every kernel). Tolerances (stated before any run;
TM64's test_k31_projection_tm64.py form): (1) every cell |out - ref| <= 1e-3 + 2^-7 |ref| (one BF16 rounding of an
FP32 sum); (2) outcome envelope vs the served path: RS's max and mean |out - ref| at most the served SplitKGemm
tactic's (64, 64, 2, 6) (same inputs) plus one BF16 ulp of max |ref| (max) and x 1.25 + 1e-6 (mean); (3) guard
regions: rows past ``rows`` and columns past N of the output buffer keep their poison; (4) graph replay == eager,
bitwise; (5) 20 eager repeats bitwise equal (a missing wait in the reduce-scatter shows as run-to-run differences);
(6) a graph of 8 PDL-chained launches on 8 inputs == each input's eager result, bitwise.

Mutations (each must make test_rs_projection fail): ``rs_skip_last`` owners skip the last rank's slot;
``rs_even_k`` drops the uneven K remainder; ``rs_row_shift`` owners read their share rows shifted by one;
``rs_store_row`` the finish stores two 8-row words at one row offset. ``rs_no_wait`` (owners do not wait for the
peers' partials) and ``rs_alias_nobar`` (v4: the tile written into the mailbox without the barrier after the sums) are race
mutants, reported but not required.
"""

import pytest
import torch

import _harness as H

_RS = "kimi_k31/k3/cutedsl/gemm/splitk_rs.py"
MUTATIONS = {
    "rs_skip_last": (_RS, "self.mutant = int(mutant)", "self.mutant = 1"),
    "rs_even_k": (_RS, "self.mutant = int(mutant)", "self.mutant = 2"),
    "rs_row_shift": (_RS, "self.mutant = int(mutant)", "self.mutant = 3"),
    "rs_store_row": (
        _RS,
        "store_words(at(mC, row0 + 8 * c, col0 + token, l_idx), packet)",
        "store_words(at(mC, row0 + 8 * (c // 2), col0 + token, l_idx), packet)",
    ),
    "rs_no_wait": (_RS, "            _wait_cluster(bar_reduce, Int32(0))\n", "\n"),
    # v4 race mutant (reported, not required): the output tile inside the mailbox without the barrier that waits for every
    # thread's sums first.
    "rs_alias_nobar": (_RS, "            if cutlass.const_expr(self.tile_alias):\n                # v4: every epilogue thread",
                       "            if cutlass.const_expr(False):\n                # v4: every epilogue thread"),
}

N, K = 3216, 7168
SERVED = (64, 64, 2, 6)  # MID served KDA in-projection at 65-128 rows (projection.py SM107_TACTICS kda 128)
RS_T1 = (128, 128, 6, 3, 4, 128)
# v4 deep rings: A1 / A2 / A3 put the output tile inside the mailbox (rs_layout), N7 = 112-column tile with its own tile
RS_A1, RS_N7, RS_A2, RS_A3 = (128, 128, 6, 7, 4, 64), (128, 112, 6, 7, 4, 64), (128, 112, 6, 8, 4, 64), (128, 96, 6, 9, 4, 64)
CASES = ([(RS_A1, r) for r in (128, 120, 104)] + [(RS_N7, r) for r in (112, 104)] + [(RS_A2, r) for r in (112, 104)]
         + [(RS_A3, r) for r in (96, 72)])

needs_rubin = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() != (10, 7), reason="RS is an SM107 kernel"
)


def _run(proj, x, rows, n, pad_rows=8, pad_cols=64):
    buf = torch.empty((rows + pad_rows, n + pad_cols), dtype=torch.bfloat16, device="cuda")
    H.poison(buf)
    keep = buf.clone()
    out = buf[:rows, :n]
    proj.run(x, out)
    torch.cuda.synchronize()
    guard_rows = int((buf[rows:].view(torch.int16) != keep[rows:].view(torch.int16)).sum())
    guard_cols = int((buf[:rows, n:].view(torch.int16) != keep[:rows, n:].view(torch.int16)).sum())
    return out.clone(), guard_rows + guard_cols


def _err(out, ref):
    d = (out.double() - ref).abs()
    return float(d.max()), float(d.mean())


def _bits(t):
    return t.contiguous().view(torch.int16)


@needs_rubin
@pytest.mark.parametrize("fenced", [2, 0])
@pytest.mark.parametrize("tactic,rows", CASES)
def test_rs_projection(tactic, rows, fenced):
    H.load_plugin(MUTATIONS)
    from sgl_nvidia_mega.kimi_k31.k3.cutedsl.gemm.projection import PreparedProjection

    g = H.generator(2250 + rows + tactic[1] + 7 * fenced)
    weight = H.randn((N, K), 0.02, g)
    x = H.randn((rows, K), 1.0, g)
    ref = x.double() @ weight.double().T
    rs = PreparedProjection(weight, {rows: tactic}, fenced_trigger=fenced).prepare()
    served = PreparedProjection(weight, {rows: SERVED}, fenced_trigger=fenced).prepare()
    out, guard = _run(rs, x, rows, N)
    base, _ = _run(served, x, rows, N)
    tag = f"rs.{'.'.join(map(str, tactic))}.r{rows}.f{fenced}"
    (tmax, tmean), (smax, smean) = _err(out, ref), _err(base, ref)
    ulp = float(ref.abs().max()) * 2**-7  # one BF16 ulp at the largest |ref| (upper bound)
    print(f"RS_ERR {tag} max {tmax:.3e} mean {tmean:.3e} | served {SERVED} max {smax:.3e} mean {smean:.3e} "
          f"| differs_from_served {int((_bits(out) != _bits(base)).sum())}/{out.numel()}", flush=True)
    assert guard == 0, f"{tag}: RS wrote {guard} cells outside its output"
    H.check(tag, out, ref.float(), 1e-3, 2**-7)
    assert tmax <= smax + ulp, (tag, tmax, smax, ulp)
    assert tmean <= 1.25 * smean + 1e-6, (tag, tmean, smean)
    # (5) repeats bitwise
    rep = torch.empty_like(out)
    diff = 0
    for _ in range(20):
        H.poison(rep)
        rs.run(x, rep)
        torch.cuda.synchronize()
        diff += int((_bits(rep) != _bits(out)).sum())
    assert diff == 0, f"{tag}: {diff} cells differ over 20 eager repeats"
    # (4) graph replay == eager (bitwise)
    outs = torch.empty_like(out)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        rs.run(x, outs)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        rs.run(x, outs)
    H.poison(outs)
    graph.replay()
    torch.cuda.synchronize()
    assert torch.equal(_bits(outs), _bits(out)), f"{tag}: graph replay differs from eager"
    print(f"RS_CASE_OK {tag}", flush=True)


@needs_rubin
@pytest.mark.parametrize("fenced", [2, 0])
def test_rs_chain(fenced):
    """(6) 8 PDL-chained RS launches (v4: A1, output tile inside the mailbox) (8 inputs -> 8 outputs) in one graph == each input's eager result, bitwise."""
    H.load_plugin(MUTATIONS)
    from sgl_nvidia_mega.kimi_k31.k3.cutedsl.gemm.projection import PreparedProjection

    rows = 128
    g = H.generator(2260 + fenced)
    weight = H.randn((N, K), 0.02, g)
    xs = [H.randn((rows, K), 1.0, g) for _ in range(8)]
    rs = PreparedProjection(weight, {rows: RS_A1}, fenced_trigger=fenced).prepare()  # v4: the tile-in-mailbox form
    eager = []
    for x in xs:
        o = torch.empty((rows, N), dtype=torch.bfloat16, device="cuda")
        rs.run(x, o)
        torch.cuda.synchronize()
        eager.append(o.clone())
    outs = [torch.empty((rows, N), dtype=torch.bfloat16, device="cuda") for _ in xs]
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for x, o in zip(xs, outs):
            rs.run(x, o)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        for x, o in zip(xs, outs):
            rs.run(x, o)
    bad = 0
    for _ in range(5):
        for o in outs:
            H.poison(o)
        graph.replay()
        torch.cuda.synchronize()
        bad += sum(int((_bits(o) != _bits(e)).sum()) for o, e in zip(outs, eager))
    assert bad == 0, f"chain f{fenced}: {bad} cells differ from eager over 5 replays of 8 launches"
    print(f"RS_CHAIN_OK f{fenced}", flush=True)


@needs_rubin
def test_rs_routing(monkeypatch):
    """Default off: without a 6-field K31MK_PROJ_TACTICS entry the served tables are unchanged at 65-128 rows; with
    kda:128=128.128.6.3.4.128 the KDA projection at 128 rows takes it (exact width only)."""
    H.load_plugin(MUTATIONS)
    from sgl_nvidia_mega.kimi_k31.k3.cutedsl.gemm import projection as P

    dev = torch.device("cuda", torch.cuda.current_device())
    monkeypatch.delenv("K31MK_PROJ_TACTICS", raising=False)
    P.tactic_overrides.cache_clear()
    off = {r: P.projection_tactic("kda", r, dev) for r in (72, 96, 104, 128)}
    assert all(t == SERVED for t in off.values()), off
    monkeypatch.setenv("K31MK_PROJ_TACTICS", "kda:128=" + ".".join(map(str, RS_T1)))
    assert P.projection_tactic("kda", 128, dev) == RS_T1
    assert P.projection_tactic("kda", 104, dev) == SERVED
    assert P.projection_tactic("mla", 128, dev) != RS_T1
    print("RS_ROUTING_OK", flush=True)
