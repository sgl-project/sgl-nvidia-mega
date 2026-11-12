# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""``K31MK_DSA_QF_G18`` (mla/head_gate.py): the plugin reorders the QF-packed MLA projection [q_a | kv_a | gate | wk |
weights_proj] into a new weight [q_a | kv_a | wk | weights_proj | gate] at load. At 8 rows the projection runs as TM8 over
rows 0:2272 (18 clusters) and a SplitKGemm (128, 8, 7, 5) over the head-gate rows on a side stream; every gate reader takes
the gate through ``head_gate()``, which joins that stream first.

Run with K31MK_DSA_QF=1 K31MK_DSA_QF_G18=1 K31MK_PROJ_TMEM8=1 K31_TEST_HEADS=6 (TP16: 6 heads, pitch 3040).

Gates (stated before any run):
- relayout: every block of a weight packed in the host's QF order (host kimi_k31_verify_binding.py: q_a + kv_a 2112 rows,
  gate 128 * heads, wk 128, weights_proj 32) lands at the design's rows (q_a + kv_a 0, wk 2112, weights_proj 2240, gate 2272
  at 6 heads), bitwise; restore(relayout(W)) == W bitwise; a weight of another row count is refused.
- columns: gate_col() == 2272 under the knob; dsa/verify.qf_columns(6) == (2112, 2240, 3040) under the knob.
- lint: no plugin source slices the gate at a literal 2112 (every reader goes through head_gate), and each reader file
  calls head_gate.
- GPU, 8 rows: G18Projection's q_a + kv_a, wk, weights_proj and head_gate() columns BITWISE equal to the same blocks of the
  served QF path (PreparedProjection over the QF-order weight with the TM8 tactic (128, 8, 7, 12, 15, 16, 4)); the QF path
  within 1e-3 + 2^-7 |ref| of an FP32 PyTorch reference x @ W.T (independent of the kernels). The same inside a captured CUDA
  graph (fork and join in the capture) on two replays with fresh inputs.
- GPU, 16 rows: one projection over the reordered weight, column-mapped, BITWISE equal to the QF path's at 16 rows.

Mutations (each must make a test fail): order (relayout keeps the QF order), lint_site (one reader back at literal 2112),
gate_col (gate_col ignores the knob), split4 (gate GEMM at split K 4: another sum order), nojoin (head_gate does not join
the side stream; the capture then ends with an unjoined fork).
"""

import os
import re
from pathlib import Path

import pytest
import torch

import _harness as H

MUTATIONS = {
    "order": ("kimi_k31/k3/cutedsl/mla/head_gate.py",
              "return torch.cat((weight[:q], weight[g:], weight[q:g]), dim=0).contiguous()",
              "return torch.cat((weight[:q], weight[q:g], weight[g:]), dim=0).contiguous()"),
    "lint_site": ("kimi_k31/dsa/verify.py",
                  "        value(to_cute(self.mla.output[:, 0, :h]), uv, to_cute(head_gate(projected, h)),",
                  "        value(to_cute(self.mla.output[:, 0, :h]), uv, to_cute(projected[:, 2112 : 2112 + 128 * h]),"),
    "gate_col": ("kimi_k31/k3/cutedsl/mla/head_gate.py",
                 "return QUERY_ROWS + INDEX_ROWS if G18 else QUERY_ROWS",
                 "return QUERY_ROWS"),
    "split4": ("kimi_k31/k3/cutedsl/mla/head_gate.py",
               "GATE_TACTIC = (128, 8, 7, 5)",
               "GATE_TACTIC = (128, 8, 4, 5)"),
    "nojoin": ("kimi_k31/k3/cutedsl/mla/head_gate.py",
               "        torch.cuda.current_stream(projected.device).wait_stream(pending)\n",
               "        pass\n"),
}

HEADS = int(os.environ.get("K31_TEST_HEADS", "6"))
GATE = 128 * HEADS
PITCH = 2112 + GATE + 160
TM8 = (128, 8, 7, 12, 15, 16, 4)  # projection.SM107_TMEM8["mla"]
# Expected rows of each block after the relayout (the design, independent of head_gate.py's constants).
NEW_ROWS = {"qkv": (0, 2112), "wk": (2112, 2240), "wp": (2240, 2272), "gate": (2272, 2272 + GATE)}
QF_ROWS = {"qkv": (0, 2112), "gate": (2112, 2112 + GATE), "wk": (2112 + GATE, 2240 + GATE), "wp": (2240 + GATE, PITCH)}

needs_gpu = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() != (10, 7),
    reason="needs an SM107 GPU (TM8)",
)


def plugin():
    return H.load_plugin(MUTATIONS)


def head_gate_module():
    plugin()
    from sgl_nvidia_mega.kimi_k31.k3.cutedsl.mla import head_gate

    return head_gate


def test_knob_engaged():
    assert os.environ.get("K31MK_DSA_QF_G18") == "1", "run with K31MK_DSA_QF_G18=1"
    hg = head_gate_module()
    assert hg.G18
    assert hg.gate_col() == NEW_ROWS["gate"][0]


def test_relayout_blocks():
    hg = head_gate_module()
    w = torch.randn((PITCH, 64), dtype=torch.float32).to(torch.bfloat16)
    r = hg.relayout(w, HEADS)
    assert r.shape == w.shape and r.is_contiguous()
    for name, (a, b) in NEW_ROWS.items():
        qa, qb = QF_ROWS[name]
        assert torch.equal(r[a:b], w[qa:qb]), name
    assert torch.equal(hg.restore(r, HEADS), w)
    with pytest.raises(ValueError):
        hg.relayout(w[:-16], HEADS)


def test_lint_gate_readers():
    root = Path(plugin().__file__).resolve().parent
    literal = re.compile(r"projected\[:,\s*2112\s*:\s*2112")
    hits = [f"{p.relative_to(root)}:{i + 1}" for p in root.rglob("*.py")
            for i, line in enumerate(p.read_text().splitlines()) if literal.search(line)]
    assert not hits, f"gate read at a literal column (use head_gate): {hits}"
    for rel in ("kimi_k31/dsa/verify.py", "kimi_k31/dsa/prepared.py", "kimi_k31/dcp/attention.py",
                "kimi_k31/k3/cutedsl/mla/prepared.py"):
        assert "head_gate(projected, " in (root / rel).read_text(), rel


@needs_gpu
def test_qf_columns_under_knob():
    plugin()
    from sgl_nvidia_mega.kimi_k31.dsa.verify import qf_columns

    assert qf_columns(HEADS) == (NEW_ROWS["wk"][0], NEW_ROWS["wp"][0], PITCH)


def _projections(seed):
    plugin()
    from sgl_nvidia_mega.kimi_k31.k3.cutedsl.gemm import projection as pm
    from sgl_nvidia_mega.kimi_k31.k3.cutedsl.gemm.projection import PreparedProjection

    hg = head_gate_module()
    g = H.generator(seed)
    w = H.randn((PITCH, 7168), 0.02, g)
    sixteen = pm.projection_tactic("mla", 16, torch.device("cuda", torch.cuda.current_device()), width=PITCH)
    served = PreparedProjection(w, {8: TM8, 16: sixteen}).prepare()
    g18 = hg.G18Projection(hg.relayout(w, HEADS), {8: TM8, 16: sixteen}, HEADS, pm).prepare()
    return hg, g, w, served, g18


def _same_blocks(name, out_new, gate_new, out_qf):
    for block in ("qkv", "wk", "wp"):
        a, b = NEW_ROWS[block]
        qa, qb = QF_ROWS[block]
        assert torch.equal(out_new[:, a:b], out_qf[:, qa:qb]), f"{name}: {block} not bitwise"
    qa, qb = QF_ROWS["gate"]
    assert torch.equal(gate_new, out_qf[:, qa:qb]), f"{name}: head gate not bitwise"


@needs_gpu
def test_g18_eight_rows_bitwise():
    hg, g, w, served, g18 = _projections(1801)
    x = H.randn((8, 7168), 1.0, g)
    out_qf = torch.empty((8, PITCH), dtype=torch.bfloat16, device="cuda")
    out_new = torch.empty_like(out_qf)
    H.poison(out_qf, out_new)
    served.run(x, out_qf)
    g18.run(x, out_new)
    gate_new = hg.head_gate(out_new, HEADS).clone()
    torch.cuda.synchronize()
    H.check("qf_g18.served_vs_fp32", out_qf, (x.float() @ w.float().T).to(torch.bfloat16), 1e-3, 2**-7)
    _same_blocks("eager", out_new, gate_new, out_qf)
    assert g18.inproj.policies[8] == TM8 and g18.gate.policies[8] == hg.GATE_TACTIC
    assert g18.inproj.weight.shape[0] == 2272 and g18.gate.weight.shape[0] == GATE


@needs_gpu
def test_g18_eight_rows_graph():
    hg, g, w, served, g18 = _projections(1802)
    x = torch.empty((8, 7168), dtype=torch.bfloat16, device="cuda")
    out_new = torch.empty((8, PITCH), dtype=torch.bfloat16, device="cuda")
    gate_new = torch.empty((8, GATE), dtype=torch.bfloat16, device="cuda")
    out_qf = torch.empty_like(out_new)
    x.copy_(H.randn((8, 7168), 1.0, g))
    g18.run(x, out_new)  # eager warm-up (compiled code loaded before the capture)
    hg.head_gate(out_new, HEADS)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        g18.run(x, out_new)
        gate_new.copy_(hg.head_gate(out_new, HEADS))
    for rep in range(2):
        x.copy_(H.randn((8, 7168), 1.0, g))
        H.poison(out_new, gate_new, out_qf)
        graph.replay()
        served.run(x, out_qf)
        torch.cuda.synchronize()
        _same_blocks(f"graph.replay{rep}", out_new, gate_new, out_qf)


@needs_gpu
def test_g18_sixteen_rows_bitwise():
    hg, g, w, served, g18 = _projections(1803)
    x = H.randn((16, 7168), 1.0, g)
    out_qf = torch.empty((16, PITCH), dtype=torch.bfloat16, device="cuda")
    out_new = torch.empty_like(out_qf)
    H.poison(out_qf, out_new)
    served.run(x, out_qf)
    g18.run(x, out_new)  # one projection: no fork at 16 rows
    gate_new = hg.head_gate(out_new, HEADS)
    torch.cuda.synchronize()
    _same_blocks("rows16", out_new, gate_new, out_qf)
