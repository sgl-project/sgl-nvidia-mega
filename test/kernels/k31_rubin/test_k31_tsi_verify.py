# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""K31MK_TSI wired into the sparse verify layer (kimi_k31/dsa/verify.py) and its context gate.

One GPU, SM100 family, K31_TEST_HEADS=6. The layer is test_k31_verify_streams.Layer (Breq 1 x Q8, context 12000,
random weights; the real SparseVerifyIndexKey writer, fused index query and trtllm-gen sparse core) with its page
table SORTED, so slot order == position order. Arm D = DCP1 (DeepGEMM logits + stock topk_v2) with each row's slots
sorted right after top-k (on top-k's stream, D only); arm T = the same layer with step.sparse == 2 and a TsiState.
T's 15 peer ranks are emulated on this GPU: when the layer calls select, ranks 1..15 first run their scan + push on
the SAME index query, gates and index-K cache (TP replicates them), then rank 0 runs the layer's own select.

References and gates (exact, no tolerance; fixed before any run):
- selection: torch stable sort of the FULL-context DeepGEMM logits over each row's causal prefix (score desc,
  position asc), first 2048, sorted by position, mapped through the table (test_k31_tsi.reference; independent of
  TSI). T's selected == it exactly; D's selected == it exactly (requires no tie at the cut: asserted).
- downstream: T's sparse-core latent and layer output == D's, bitwise (same slots in the same order, same query).
- K31MK_TSI_S1R=1 (s1r merge): the reference orders each row source-major (test_k31_tsi.emission_order) and D's
  slots are put in that order right after top-k, so the bitwise downstream check still compares the same slot
  order; the test asserts its data has no certificate-fallback row (those stay position-sorted in s1r).
- engagement: rank 0's fallback flags (poisoned 7) end in {0, 1} (the TSI merge wrote them) and the peers ran.
- CUDA graph: each arm captured once; two replays on new inputs equal eager runs of the same inputs, bitwise
  (the per-forward prepare is captured, so every replay advances the epoch and tags).
- fail closed: a step labelled tsi on a build without TSI raises before any launch (layer output, latent and MLA
  query scratch stay poisoned).
- gate (CPU): tsi_graph_label at the measured crossover (two live deltas, linearly interpolated
  and page-rounded), rows 8 / 16 only, every request's context, knob off -> never.

Mutations (``K31_MUTATION=<name>`` through _harness / the fixed run_suite.sh; each must FAIL this file):
  tsi_no_prepare   the per-forward prepare is skipped        -> test_tsi_layer_exact, test_tsi_layer_graph
  tsi_dcp1_path    a tsi-labelled step runs DCP1's selection -> test_tsi_layer_exact (engagement)
  tsi_gate_max     the gate reads the longest request        -> test_gate_label
  tsi_gate_rows32  the gate admits rows 32 (unmeasured)      -> test_gate_label
  tsi_fail_open    no early refusal of an unbuilt tsi step   -> test_tsi_label_without_build
"""

from __future__ import annotations

import math
from types import SimpleNamespace

import pytest
import torch

import _harness as H

V_FILE = "kimi_k31/dsa/verify.py"
MUTATIONS = {
    "tsi_no_prepare": (V_FILE, "                if ordinal == 0:  # once per forward",
                       "                if ordinal < 0:  # once per forward"),
    "tsi_dcp1_path": (V_FILE, "            if step.sparse == 2:  # TSI", "            if step.sparse == 3:  # TSI"),
    "tsi_gate_max": (V_FILE, "int(seq_lens.min()) >= TSI_MIN_CONTEXT", "int(seq_lens.max()) >= TSI_MIN_CONTEXT"),
    "tsi_gate_rows32": (V_FILE, "TSI_GATE_ROWS = (8, 16)\n", "TSI_GATE_ROWS = (8, 16, 32)\n"),
    "tsi_fail_open": (
        V_FILE,
        "        if step.sparse == 2 and self.tsi is None:\n            raise ValueError(",
        "        if False:\n            raise ValueError(",
    ),
}
H.load_plugin(MUTATIONS)

import test_k31_tsi as TT  # noqa: E402
from test_k31_mla import HEADS  # noqa: E402
from test_k31_verify_streams import CTX, Q, ROWS, Layer, num_sms  # noqa: E402

needs_gpu = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 10,
    reason="needs an SM100-family GPU",
)


def _verify():
    from sgl_nvidia_mega.kimi_k31.dsa import verify as V

    return V


# ------------------------------------------------------------------------------------------------ gate (CPU)
def test_gate_label(monkeypatch):
    V = _verify()
    lo, hi, d_lo, d_hi = 393216, 524288, 3.5, -2.5  # measured crossover inputs (rows 8, live minus DCP1)
    crossing = lo + d_lo / (d_lo - d_hi) * (hi - lo)
    assert V.TSI_MIN_CONTEXT == 64 * math.ceil(crossing / 64), (V.TSI_MIN_CONTEXT, crossing)
    t = lambda *ctx: torch.tensor(ctx, dtype=torch.int64)  # noqa: E731
    monkeypatch.setattr(V, "TSI", True)
    g = V.TSI_MIN_CONTEXT
    assert V.tsi_graph_label(t(g), 8) and not V.tsi_graph_label(t(g - 1), 8), "rows 8 boundary"
    assert V.tsi_graph_label(t(g, 700000), 8), "rows 16, both requests above the gate"
    assert not V.tsi_graph_label(t(700000, 10000), 8), "rows 16 with a short request must stay DCP1"
    assert not V.tsi_graph_label(t(700000, 700000, 700000, 700000), 8), "rows 32 is unmeasured: never TSI"
    assert not V.tsi_graph_label(None, 8) and not V.tsi_graph_label(t(), 8)
    monkeypatch.setattr(V, "TSI", False)
    assert not V.tsi_graph_label(t(900000), 8), "knob off: never TSI"


# ------------------------------------------------------------------------------------------------ layer (GPU)
class Peers:
    """Rank 0's TsiState as the layer sees it; ranks 1..15 (same GPU) prepare with it and push before its select."""

    def __init__(self, max_pages):
        _, self.states, _ = TT.ranks(ROWS, max_pages)
        self.outs = [torch.empty(ROWS, TT.T.TOPK, dtype=torch.int32, device="cuda") for _ in self.states]
        self.inputs = None
        self.selects = 0

    def prepare(self, table, contexts, causal):
        for st in self.states:
            st.prepare(table, contexts, causal)

    def select(self, ordinal, query, gates, cache, out, *, phases=None):
        self.inputs, self.selects = (query, gates), self.selects + 1
        for st, o in zip(self.states[1:], self.outs[1:]):
            st.select(ordinal, query, gates, cache, o, phases=TT.T.PUSH)
        return self.states[0].select(ordinal, query, gates, cache, out, phases=TT.T.BOTH if phases is None else phases)


def build(seed):
    layer = Layer(seed=seed)
    layer.table = layer.table.sort(dim=-1).values.contiguous()  # slot order == position order
    pos = torch.arange(CTX - Q, CTX, device="cuda")
    layer.slots = layer.table[0, pos // 64].long() * 64 + pos % 64
    D, Tarm = layer.arm("D", 0), layer.arm("T", 0)
    peers = Peers(layer.table.shape[1])
    Tarm.core.tsi = (peers, 0)
    Tarm.core.step = SimpleNamespace(**{**vars(layer.step), "sparse": 2})
    return layer, D, Tarm, peers


@pytest.fixture
def sort_dcp1(monkeypatch):
    """Sort each row's slots right after the stock top-k, on its stream, for the registered (D) buffers only (the
    table is sorted, so slot order == position order); under TT.T.S1R then source-major (page index % 16 ascending,
    position ascending inside a source: the s1r order of a row whose certificate held)."""
    import sglang.kernels.ops.attention.dsv4.topk as K

    class Targets(list):
        table = None  # the (sorted, strictly increasing) page-table row of the registered layer

    orig, targets = K.topk_transform_paged_v2, Targets()

    def wrapped(*args, **kwargs):
        out = orig(*args, **kwargs)
        sel = args[3]
        if any(sel.data_ptr() == t.data_ptr() for t in targets):
            ordered = torch.sort(sel, dim=-1).values
            if TT.T.S1R:
                page = torch.searchsorted(targets.table, (ordered // 64).contiguous())  # strictly increasing row
                key = (page.long() % TT.W) * (1 << 40) + ordered.long()
                key = torch.where(ordered < 0, torch.full_like(key, 1 << 62), key)  # -1 slots stay last
                ordered = torch.gather(ordered, 1, torch.sort(key, dim=-1).indices)
            sel.copy_(ordered)
        return out

    monkeypatch.setattr(K, "topk_transform_paged_v2", wrapped)
    return targets


def run(layer, a):
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        layer.chain(a)
    torch.cuda.current_stream().wait_stream(s)
    return snap(layer, a)


def snap(layer, a):
    torch.cuda.synchronize()
    return {"selected": a.core.selection.selected[:ROWS].clone(), "latent": a.core.mla.output[:, 0, :HEADS].clone(),
            "out": a.out.clone()}


def reference(layer, a, peers):
    """Exact top-2048 slots from the full-context DeepGEMM logits of the index query the layer computed."""
    import deep_gemm
    from sglang.kernels.ops.attention.dsa.paged_mqa_logits import _restore_row_stride

    query, gates = peers.inputs
    ctx = layer.step.score_contexts
    logits = _restore_row_stride(deep_gemm.fp8_paged_mqa_logits(
        query.view(1, Q, 32, 128), a.idx.view(-1, 64, 1, TT.T.INDEX_BYTES), gates.reshape(ROWS, 32), ctx,
        layer.table, deep_gemm.get_paged_mqa_logits_metadata(ctx, 64, num_sms()), layer.table.shape[1] * 64,
        clean_logits=False,
    ))
    ref, flags, ties = TT.reference(SimpleNamespace(rows=ROWS, causal=layer.step.causal_lengths, table=layer.table),
                                    logits)
    assert max(ties) == 1, f"test construction: ties at the top-2048 cut {ties} (stock top-k is not exact there)"
    if TT.T.S1R:
        assert int(flags.sum()) == 0, f"test construction: certificate-fallback rows {flags.tolist()} under S1R"
    return ref


def check(tag, layer, D, Tarm, peers, d, t):
    ref = reference(layer, Tarm, peers)
    for name, got in (("T", t["selected"]), ("D", d["selected"])):
        bad = (got != ref).any(dim=1)
        assert not bool(bad.any()), f"{tag}: arm {name} selection != exact top-2048 in {int(bad.sum())} rows"
    for key in ("latent", "out"):
        diff = int((t[key].view(torch.uint8) != d[key].view(torch.uint8)).sum())
        H.record({"kind": "bitwise", "item": f"tsi_vs_dcp1.{tag}.{key}", "differ_bytes": diff})
        assert diff == 0, f"{tag}: {key} T vs D differs in {diff} bytes"
    assert torch.isfinite(t["out"].float()).all(), f"{tag}: non-finite output"


@needs_gpu
def test_tsi_layer_exact(sort_dcp1):
    layer, D, Tarm, peers = build(41)
    sort_dcp1.append(D.core.selection.selected)
    sort_dcp1.table = layer.table[0].contiguous()
    for it in range(2):
        layer.new_inputs()
        H.poison(D.out, Tarm.out)
        peers.states[0]._fallback.fill_(7)
        selects = peers.selects
        d, t = run(layer, D), run(layer, Tarm)
        assert peers.selects == selects + 1, "the tsi-labelled step did not run the TSI selection"
        flags = peers.states[0]._fallback[:ROWS]
        assert bool(((flags == 0) | (flags == 1)).all()), f"TSI merge did not write its fallback flags: {flags.tolist()}"
        check(f"eager.it{it}", layer, D, Tarm, peers, d, t)
        print("TSI_LAYER eager", it, "fallback", flags.tolist(), flush=True)


@needs_gpu
def test_tsi_layer_graph(sort_dcp1):
    layer, D, Tarm, peers = build(43)
    sort_dcp1.append(D.core.selection.selected)
    sort_dcp1.table = layer.table[0].contiguous()
    graphs = {}
    for a in (D, Tarm):
        run(layer, a)  # warm-up (lazy init) before capture
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            layer.chain(a)
        graphs[a.name] = g
    for it in range(2):
        layer.new_inputs()
        snaps = {}
        for a in (D, Tarm):
            H.poison(a.out)
            graphs[a.name].replay()
            snaps[a.name] = snap(layer, a)
        check(f"graph.it{it}", layer, D, Tarm, peers, snaps["D"], snaps["T"])
        eager = run(layer, Tarm)
        for key in eager:
            assert torch.equal(eager[key].view(torch.uint8), snaps["T"][key].view(torch.uint8)), f"graph != eager: {key}"


@needs_gpu
def test_tsi_label_without_build():
    layer = Layer(seed=47)
    D = layer.arm("D", 0)
    assert D.core.tsi is None
    D.core.step = SimpleNamespace(**{**vars(layer.step), "sparse": 2})
    H.poison(D.out, D.core.mla.output, layer.base.query_scratch)
    with pytest.raises(ValueError, match="no TSI state"):
        run(layer, D)
    torch.cuda.synchronize()
    for name, buf in (("out", D.out), ("latent", D.core.mla.output), ("mla query", layer.base.query_scratch)):
        assert bool((buf.view(torch.uint8) == 0xFF).all()), f"refused tsi step still wrote {name}"
