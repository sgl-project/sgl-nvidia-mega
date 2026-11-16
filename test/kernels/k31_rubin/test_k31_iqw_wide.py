# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""K3.1 MK sparse MLA verify layer above 8 rows (Breq 2 / 4 / 8 x Q8 = 16 / 32 / 64 rows), the stock index-query chain
(no fused index query above 8 rows), per-rank heads from K31_TEST_HEADS, random weights, the real plugin objects in
K31VerifyExecutor's order: SparseVerifyIndexKey (unpacked: stock index-K writer on the index stream, gate GEMM on the gate
stream), a projection stand-in on main, PreparedSparseVerifyMla.run (MLA query, stock index query, DeepGEMM logits, stock
topk_v2 with the per-row page table, trtllm-gen sparse MLA, MlaValue).

Arm A: shipped scheduling (index_q_mode 0 above 8 rows: the stock index query queued on the index stream behind the
index-K writer). Arm B: K31MK_INDEX_Q_OFF_WRITER_WIDE=1 (the index query on the query stream after the input projection
and the gate GEMM; the scoring joins the writer). Same kernels, same inputs.

Gates (written before any run; adapted from test_k31_verify_streams.py):
- B vs A BITWISE (logits over the context columns, top-k slots, sparse-core latent, output, index-K rows, MLA KV rows),
  eager and CUDA-graph replay, two input sets, every Breq; control A vs a rerun of A bitwise. The stock top-k emits its
  slots in a run-dependent order, so these tests sort each row's slots right after top-k on top-k's own stream (both arms).
- B against independent references: each row's top-k overlaps the fp64 top-2048 (reference index query over the cache as
  written) by >= 0.99; output vs the stock core fed B's own query codes within test_k31_mla's sparse tolerances.
- Ordering: a ~5 ms spin kernel before one producer (projection on main, gate GEMM on the gate stream, index query, index-K
  store on the index stream, top-k) with every consumer input poisoned first: both arms bitwise equal to undelayed A.
- Soak (randomized timing): 24 runs of B with random spins (0-300 us) before a random subset of the producers, inputs
  poisoned each time: every run bitwise equal to undelayed A.
- Engagement: with the store delayed, B's index query ends >= 1 ms before the store (A: after it).
Mutations (each must FAIL the suite): MUTATIONS below.
"""

from __future__ import annotations

import math
import os
import random
from types import SimpleNamespace

import pytest
import torch

import _harness as H
import _ref_mla as R
from test_k31_mla import HEADS, MLA_ATOL, MLA_RTOL, fakes, fi_mla, index_pool, kv_pool, make_weights, num_sms

V_FILE = "kimi_k31/dsa/verify.py"
MUTATIONS = {
    # the knob is read but above 8 rows the index query stays behind the writer
    "iqw_inert": (V_FILE, "            self.index_q_mode = INDEX_Q_OFF_WRITER_WIDE if rows > 8 else 0\n",
                  "            self.index_q_mode = 0\n"),
    # mode 0/1: the index query no longer waits for the gate GEMM
    "iqw_no_gate_join": (
        V_FILE,
        "        else:\n            side.wait_stream(main)\n            if not self.packed_index_gate:\n"
        "                side.wait_stream(self.gate_stream)\n",
        "        else:\n            side.wait_stream(main)\n",
    ),
    # mode 1: the index query no longer waits for the input projection (arm A keeps its wait)
    "iqw_no_proj_wait": (
        V_FILE,
        "        else:\n            side.wait_stream(main)\n",
        "        else:\n            if side is self.index_stream:\n                side.wait_stream(main)\n",
    ),
    # mode 1: the scoring no longer waits for the index-K store
    "iqw_no_store_join": (V_FILE, "            side.wait_stream(self.index_stream)\n", "            pass\n"),
    # main joins the writer stream instead of the stream that ran top-k (trtllm core path, the one this test runs)
    "iqw_join_writer_core": (V_FILE, "        main.wait_stream(side)\n        self.mla.run_core(",
                             "        main.wait_stream(self.index_stream)\n        self.mla.run_core("),
}

Q, CTX = 8, 12000
GATE_END = 2112 + 128 * HEADS
SPIN = 10_000_000  # cycles of torch.cuda._sleep: several ms
BREQS = tuple(int(x) for x in os.environ.get("K31_IQW_BREQS", "2,4,8").split(","))

needs_gpu = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 10,
    reason="needs an SM100-family GPU",
)


class Layer:
    """Shared weights, pools and inputs for Breq requests x Q rows; one (core, writer, caches) per arm."""

    def __init__(self, breq, seed=31):
        H.load_plugin(MUTATIONS)
        import deep_gemm
        from sglang.kernels.ops.attention.dsv4.topk import plan_topk_v2
        from sgl_nvidia_mega.kimi_k31.k3.cutedsl.mla.prepared import PreparedMla

        g = H.generator(seed)
        self.g, self.breq, self.rows = g, breq, breq * Q
        self.w = w = make_weights(seed)
        self.wk = (torch.randn((128, 7168), generator=g, device="cuda") / math.sqrt(7168)).to(torch.bfloat16)
        npg = math.ceil(CTX / 64)
        self.pages = breq * npg + 2
        self.kv0, self.idx0 = kv_pool(self.pages, g), index_pool(self.pages, g)
        perm = torch.randperm(self.pages - 1, generator=g, device="cuda")[: breq * npg] + 1
        self.table = perm.view(breq, npg).to(torch.int32).contiguous()
        pos = torch.arange(CTX - Q, CTX, device="cuda")
        self.slots = torch.cat([self.table[r, pos // 64].long() * 64 + pos % 64 for r in range(breq)])
        self.lengths = torch.full((breq,), CTX, dtype=torch.int32, device="cuda")
        causal = (CTX - Q + 1 + torch.arange(Q, device="cuda")).to(torch.int32).repeat(breq)
        contexts = torch.full((breq, Q), CTX, dtype=torch.int32, device="cuda")
        self.step = SimpleNamespace(
            sparse=1, batch_requests=breq, query_width=Q, causal_lengths=causal, score_contexts=contexts,
            score_schedule=deep_gemm.get_paged_mqa_logits_metadata(contexts, 64, num_sms()),
            topk_plan=plan_topk_v2(causal),
        )
        scratch = torch.empty((self.rows, HEADS, 576), dtype=torch.uint8, device="cuda")
        attention, _ = fakes(w)
        self.base = PreparedMla(attention, {self.rows: GATE_END}, scratch, max_context=1 << 17, core="trtllm",
                                queries=(Q,))
        self.first = torch.empty((self.rows, 7168), dtype=torch.bfloat16, device="cuda")
        self.projected = torch.empty((self.rows, GATE_END), dtype=torch.bfloat16, device="cuda")
        self.hidden_src = torch.empty_like(self.first)
        self.proj_src = torch.empty_like(self.projected)
        self.logits = {}
        self.arm_name = None
        self.new_inputs()

    def new_inputs(self):
        self.hidden_src.copy_(torch.randn(self.hidden_src.shape, generator=self.g, device="cuda").to(torch.bfloat16))
        self.proj_src.copy_(torch.randn(self.proj_src.shape, generator=self.g, device="cuda").to(torch.bfloat16))

    def arm(self, name, wide):
        from sgl_nvidia_mega.kimi_k31.dsa import verify as V
        from sgl_nvidia_mega.kimi_k31.dsa.indexer import IndexWeights
        from sgl_nvidia_mega.kimi_k31.verify_indexer import IndexKeyWeights, PreparedVerifyIndexKey

        w = self.w
        default = V.INDEX_Q_OFF_WRITER_WIDE
        V.INDEX_Q_OFF_WRITER_WIDE = wide
        try:
            core = V.PreparedSparseVerifyMla(self.base, IndexWeights(w.wq_b, w.w_proj, w.w_qa, 1e-5), self.rows, 1 << 17)
        finally:
            V.INDEX_Q_OFF_WRITER_WIDE = default
        assert core.fused_index_q is None, "above 8 rows the stock index query must serve"
        assert (core.query_stream is core.index_stream) == (wide == 0), "arm does not run the path it names"
        kv, idx = self.kv0.clone(), self.idx0.clone()
        core.prepare_cache(kv, prepare_base=not hasattr(self, "_based"))
        self._based = True
        writer = V.SparseVerifyIndexKey(
            PreparedVerifyIndexKey(IndexKeyWeights(self.wk, w.gamma, w.beta, w.k_eps)), core
        )
        assert writer.packed_weight is None and not core.packed_index_gate, "above 8 rows the writer must not pack"
        writer.bind(idx, 64, rows=(self.rows,))
        core.step = self.step
        return SimpleNamespace(name=name, core=core, writer=writer, kv=kv, idx=idx,
                               out=torch.empty((self.rows, HEADS * 128), dtype=torch.bfloat16, device="cuda"))

    def poison(self, a):
        """Reset every buffer a consumer could read before its producer wrote it."""
        H.poison(a.out, a.core.mla.output)
        self.projected.zero_()
        a.core.index_projection.query.zero_()
        a.core.index_projection.gate.zero_()
        a.core.selection.selected.fill_(-1)
        page, off = self.slots // 64, self.slots % 64
        for p, o in zip(page.tolist(), off.tolist()):
            a.idx[p, o * 128 : o * 128 + 128] = 0
            a.idx[p, 64 * 128 + o * 4 : 64 * 128 + o * 4 + 4] = 0

    def chain(self, a, delay_projection=0):
        """One layer as K31VerifyExecutor runs it, on the current stream."""
        self.arm_name = a.name
        self.first.copy_(self.hidden_src)  # the layer input producer (AttnRes in the model)
        a.writer.run(self.first, self.slots)
        if delay_projection:
            torch.cuda._sleep(delay_projection)
        self.projected.copy_(self.proj_src)  # the input projection (MK split-K in the model)
        a.core.run(self.projected, a.out, cache=a.kv, slots=self.slots, block_table=self.table,
                   lengths=self.lengths, q=Q)

    def snapshot(self, a):
        torch.cuda.synchronize()
        h = HEADS
        page, off = self.slots // 64, self.slots % 64
        idx_rows = torch.stack([a.idx[p, o * 128 : o * 128 + 128] for p, o in zip(page.tolist(), off.tolist())])
        idx_scales = torch.stack(
            [a.idx[p, 64 * 128 + o * 4 : 64 * 128 + o * 4 + 4] for p, o in zip(page.tolist(), off.tolist())]
        )
        return {
            "logits": self.logits[a.name].clone(),
            "selected": a.core.selection.selected[: self.rows].clone(),
            "latent": a.core.mla.output[:, 0, :h].clone(),
            "out": a.out.clone(),
            "index_k": torch.cat([idx_rows, idx_scales], 1),
            "kv_rows": a.kv.view(-1, 576)[self.slots].clone(),
        }


def bitwise(name, got, ref):
    bad = []
    for key in ref:
        g, r = got[key].view(torch.uint8), ref[key].view(torch.uint8)
        diff = int((g != r).sum())
        H.record({"kind": "bitwise", "item": f"{name}.{key}", "differ_bytes": diff, "bytes": r.numel()})
        if diff:
            bad.append(f"{key}: {diff}/{r.numel()} bytes differ")
    print("BITWISE", name, "ok" if not bad else bad, flush=True)
    assert not bad, f"{name}: {bad}"


@pytest.fixture
def canonical_topk(monkeypatch):
    """Sort each row's selected slots right after top-k, on the same stream (both arms alike)."""
    import sglang.kernels.ops.attention.dsv4.topk as T

    orig = T.topk_transform_paged_v2

    def wrapped(*args, **kwargs):
        out = orig(*args, **kwargs)
        sel = args[3]
        sel.copy_(torch.sort(sel, dim=-1).values)
        return out

    monkeypatch.setattr(T, "topk_transform_paged_v2", wrapped)


@pytest.fixture
def capture_logits(monkeypatch):
    """Copy DeepGEMM's logits (context columns) into a per-arm buffer, on the stream that computed them."""
    import deep_gemm

    orig = deep_gemm.fp8_paged_mqa_logits
    layers = []

    def wrapped(*args, **kwargs):
        out = orig(*args, **kwargs)
        layer = layers[0]
        view = out.reshape(-1, out.shape[-1])[:, :CTX]
        buf = layer.logits.get(layer.arm_name)
        if buf is None:
            buf = layer.logits[layer.arm_name] = torch.empty(view.shape, dtype=view.dtype, device=view.device)
        buf.copy_(view)
        return out

    monkeypatch.setattr(deep_gemm, "fp8_paged_mqa_logits", wrapped)
    return layers


def run_eager(layer, a, **kw):
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        layer.chain(a, **kw)
    torch.cuda.current_stream().wait_stream(s)
    return layer.snapshot(a)


def references(layer, a, snap):
    w = layer.w
    projected, hidden = layer.proj_src, layer.hidden_src
    name = f"h{HEADS}.b{layer.breq}"
    q_codes, w_ref = R.index_q(projected, hidden, w.w_qa, 1e-5, w.wq_b, w.w_proj)
    causal = layer.step.causal_lengths
    for j in range(layer.rows):
        r = j // Q
        n = int(causal[j])
        pos_of_slot = R.slot_positions(layer.table[r], layer.pages)
        scores = R.index_scores(a.idx, layer.table[r], q_codes[j], w_ref[j], n)
        want = torch.topk(scores, R.TOPK).indices
        got = pos_of_slot[snap["selected"][j].long()]
        assert bool((got >= 0).all()) and bool((got < n).all()), f"{name} row {j}: slot outside the causal prefix"
        overlap = torch.isin(got, want).float().mean().item()
        H.record({"kind": "check", "item": f"wide_sel_overlap.{name}.row{j}", "overlap": overlap})
        assert overlap >= 0.99, f"{name} row {j}: overlap {overlap:.4f} < 0.99 with the fp64 top-2048"
    mk_q = a.core.mla.query[:, 0, :HEADS].view(torch.uint8).contiguous()
    lat_mk = fi_mla(mk_q, a.kv, snap["selected"], causal, sparse=True)
    gate = projected[:, 2112:GATE_END]
    H.check(f"wide_out.{name}", snap["out"], R.value(lat_mk, w.w_uv, gate, causal), MLA_ATOL, MLA_RTOL)


# ----------------------------------------------------------------------------------------------- correctness
@needs_gpu
@pytest.mark.parametrize("breq", BREQS)
def test_bitwise_eager(breq, capture_logits, canonical_topk):
    layer = Layer(breq)
    capture_logits.append(layer)
    A, B = layer.arm("A", 0), layer.arm("B", 1)
    for it in range(2):
        layer.new_inputs()
        ra, rb, ra2 = (run_eager(layer, a) for a in (A, B, A))
        bitwise(f"b{breq}.control_A_rerun.it{it}", ra2, ra)
        bitwise(f"b{breq}.eager_B_vs_A.it{it}", rb, ra)
    references(layer, B, rb)


@needs_gpu
@pytest.mark.parametrize("breq", BREQS)
def test_bitwise_graph(breq, capture_logits, canonical_topk):
    layer = Layer(breq, seed=37)
    capture_logits.append(layer)
    A, B = layer.arm("A", 0), layer.arm("B", 1)
    graphs = {}
    for a in (A, B):
        run_eager(layer, a)  # warm-up before capture
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            layer.chain(a)
        graphs[a.name] = g
    for it in range(2):
        layer.new_inputs()
        snaps = {}
        for a in (A, B):
            layer.arm_name = a.name
            graphs[a.name].replay()
            snaps[a.name] = layer.snapshot(a)
        bitwise(f"b{breq}.graph_B_vs_A.it{it}", snaps["B"], snaps["A"])
        bitwise(f"b{breq}.graph_A_vs_eager_A.it{it}", snaps["A"], run_eager(layer, A))


# ----------------------------------------------------------------------------------------------- ordering
DELAYS = ("d1_projection", "d2g_gate", "d2_index_query", "d3_store", "d4_topk")


class Spins:
    """Producer wrappers that spin (torch.cuda._sleep) on the producer's own stream before it runs, by a per-producer
    cycle count set in ``cycles`` (0 = none)."""

    def __init__(self, monkeypatch, arms):
        import sglang.kernels.ops.attention.dsv4.topk as T
        import sglang.kernels.ops.kvcache.dsa_fused_store_index_cache as M

        self.cycles = {d: 0 for d in DELAYS}
        sp = self.spin
        store, topk = M.fused_store_index_k_cache, T.topk_transform_paged_v2
        monkeypatch.setattr(M, "fused_store_index_k_cache", lambda *x, **k: (sp("d3_store"), store(*x, **k))[1])
        monkeypatch.setattr(T, "topk_transform_paged_v2", lambda *x, **k: (sp("d4_topk"), topk(*x, **k))[1])
        for a in arms:
            ip = a.core.index_projection
            pq, pg = ip.project_query, ip.prepare_gate
            monkeypatch.setattr(ip, "project_query", lambda *x, _o=pq, **k: (sp("d2_index_query"), _o(*x, **k))[1])
            monkeypatch.setattr(ip, "prepare_gate", lambda *x, _o=pg, **k: (sp("d2g_gate"), _o(*x, **k))[1])

    def spin(self, key):
        if self.cycles[key]:
            torch.cuda._sleep(self.cycles[key])


@needs_gpu
@pytest.mark.parametrize("delay", DELAYS)
@pytest.mark.parametrize("breq", BREQS)
def test_ordering(breq, delay, capture_logits, canonical_topk, monkeypatch):
    layer = Layer(breq, seed=41)
    capture_logits.append(layer)
    A, B = layer.arm("A", 0), layer.arm("B", 1)
    layer.new_inputs()
    ref = run_eager(layer, A)
    spins = Spins(monkeypatch, (A, B))
    if delay != "d1_projection":
        spins.cycles[delay] = SPIN
    for a in (A, B):
        layer.poison(a)
        got = run_eager(layer, a, delay_projection=SPIN if delay == "d1_projection" else 0)
        bitwise(f"b{breq}.{delay}.{a.name}_vs_undelayed_A", got, ref)


@needs_gpu
@pytest.mark.parametrize("breq", BREQS)
def test_soak(breq, capture_logits, canonical_topk, monkeypatch):
    """Randomized timing: 24 runs of B with random spins (0-300 us) before a random subset of producers."""
    layer = Layer(breq, seed=53)
    capture_logits.append(layer)
    A, B = layer.arm("A", 0), layer.arm("B", 1)
    layer.new_inputs()
    ref = run_eager(layer, A)
    spins = Spins(monkeypatch, (A, B))
    rnd = random.Random(1103 + breq)
    for it in range(24):
        for d in DELAYS:
            spins.cycles[d] = rnd.choice((0, 0, rnd.randint(1, 600_000)))  # a few hundred us at most
        proj, spins.cycles["d1_projection"] = spins.cycles["d1_projection"], 0  # the projection spin runs on main
        layer.poison(B)
        bitwise(f"b{breq}.soak.it{it}", run_eager(layer, B, delay_projection=proj), ref)


@needs_gpu
@pytest.mark.parametrize("breq", BREQS)
def test_engaged(breq, capture_logits, monkeypatch):
    """With the store delayed, B's index query completes before the store (A: after it)."""
    import sglang.kernels.ops.kvcache.dsa_fused_store_index_cache as M

    layer = Layer(breq, seed=43)
    capture_logits.append(layer)
    A, B = layer.arm("A", 0), layer.arm("B", 1)
    layer.new_inputs()
    events = {}

    def stamp(key):
        e = torch.cuda.Event(enable_timing=True)
        e.record()
        events[key] = e

    store = M.fused_store_index_k_cache
    monkeypatch.setattr(M, "fused_store_index_k_cache",
                        lambda *x, **k: (torch.cuda._sleep(SPIN), store(*x, **k), stamp("store"))[1])
    for a in (A, B):
        ip = a.core.index_projection
        monkeypatch.setattr(ip, "project_query", lambda *x, _o=ip.project_query, **k: (_o(*x, **k), stamp("index_q"))[0])
    result = {}
    for a in (A, B):
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            stamp("start")
            layer.chain(a)
        torch.cuda.synchronize()
        t_store = events["start"].elapsed_time(events["store"])
        t_iq = events["start"].elapsed_time(events["index_q"])
        result[a.name] = (t_store, t_iq)
        H.record({"kind": "engaged", "item": f"h{HEADS}.b{breq}.{a.name}", "store_end_ms": t_store, "index_q_end_ms": t_iq})
        print("ENGAGED", breq, a.name, t_store, t_iq, flush=True)
    sa, qa = result["A"]
    assert qa >= sa, f"arm A: index query ended {sa - qa:.3f} ms before the store (shipped order broken)"
    sb, qb = result["B"]
    assert sb - qb >= 1.0, f"arm B: index query ended only {sb - qb:.3f} ms before the delayed store (not engaged)"
