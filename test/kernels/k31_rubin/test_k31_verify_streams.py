# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""K3.1 MK sparse MLA verify layer, Breq 1 x Q8 (8 rows), per-rank heads from K31_TEST_HEADS (6 TP16, 12 TP8), random
weights, the real plugin objects in K31VerifyExecutor's order: SparseVerifyIndexKey (packed wk mm, LayerNorm, stock
index-K store on the index stream), a projection stand-in on main, PreparedSparseVerifyMla.run (MLA query, fused index
query, DeepGEMM logits, stock topk_v2, trtllm-gen sparse MLA, MlaValue).

Arm A: shipped scheduling (the index query queued behind the index-K writer). Arms B / C: K31MK_INDEX_Q_OFF_WRITER=1 / 2
(the index query after the projection only, on the scoring stream / on main ahead of the MLA query; the scoring
joins the writer). Same kernels, same inputs. "B" in the gates below means each of B and C.

Gates (written before any run, never loosened):
- B vs A BITWISE: DeepGEMM logits over the context columns, top-k slots, sparse-core latent (real heads), output,
  index-K rows and MLA KV rows, eager and CUDA-graph replay (two input sets). Control: A vs a rerun of A, bitwise.
  The stock top-k (topk_transform_paged_v2, deterministic=False) emits its 2048 slots in a run-dependent order (first
  GB300 run: A vs A rerun differs in ~46% of the slot bytes, logits identical), and the sparse core's
  sum follows that order. So these tests sort each row's slots right after top-k, on top-k's own stream, in both
  arms (``canonical_topk``): FMHA still has to wait for that stream. test_production_topk keeps the shipped order and
  checks logits bitwise, slots as sorted sets bitwise, and records the pre-registered fallback for order-dependent
  tensors (max |B-A| <= max |A-A'| over 3 reruns).
- B against independent references: MLA query codes and KV rows within 1 FP8 step in <= 1% of cells (_ref_mla); each
  row's top-k overlaps the fp64 top-2048 (reference index query over the cache as written) by >= 0.99; output vs the
  stock core fed B's own query codes, atol 0.004 / rtol 0.03 per cell (test_k31_mla's sparse tolerances).
- Ordering: a ~5 ms spin kernel before one producer (d1 projection, d2 index query, d3 index-K store, d4 top-k) with
  the consumers' inputs reset first (zeros / -1 / NaN): both arms stay bitwise equal to the undelayed arm A.
- Engagement: with d3, B's index query finishes before the store (A: after it), so the flag changes the schedule.

Mutations (each must FAIL the suite): see MUTATIONS.
"""

from __future__ import annotations

import math
import os
from types import SimpleNamespace

import pytest
import torch

import _harness as H
import _ref_mla as R
from test_k31_mla import HEADS, MLA_ATOL, MLA_RTOL, fakes, fi_mla, index_pool, kv_pool, make_weights, num_sms

V_FILE = "kimi_k31/dsa/verify.py"
MUTATIONS = {
    # mode 1: the index query no longer waits for the input projection (arm A keeps its wait)
    "l4_no_proj_wait": (
        V_FILE,
        "        else:\n            side.wait_stream(main)\n",
        "        else:\n            if side is self.index_stream:\n                side.wait_stream(main)\n",
    ),
    # mode 2: the scoring stream forks before the index query is queued (scoring no longer waits for it)
    "l4m2_fork_early": (
        V_FILE,
        "            iq, gates = self.index_query(projected)\n            side.wait_stream(main)\n",
        "            side.wait_stream(main)\n            iq, gates = self.index_query(projected)\n",
    ),
    # modes 1/2: the scoring no longer waits for the index-K store
    "l4_no_store_join": (
        V_FILE,
        "            side.wait_stream(self.index_stream)  # the scoring",
        "            pass  # the scoring",
    ),
    # scoring on the writer stream: after the store, not after the index query, and not joined by main
    "l4_logits_on_writer": (
        V_FILE,
        "        with torch.cuda.stream(side):\n            selected = self.selection.run(",
        "        with torch.cuda.stream(self.index_stream):\n            selected = self.selection.run(",
    ),
    # main joins the writer stream instead of the stream that ran top-k
    "l4_join_writer": (V_FILE, "        main.wait_stream(side)\n", "        main.wait_stream(self.index_stream)\n"),
    # the flag is read but the B arms keep the shipped schedule
    "l4_inert": (V_FILE, "self.index_q_mode = INDEX_Q_OFF_WRITER if", "self.index_q_mode = 0 if"),
    # mode 1 keeps its own stream but still waits for the store before the index query (values stay correct)
    "l4_iq_waits_store_m1": (
        V_FILE,
        "            with torch.cuda.stream(side):\n                iq, gates = self.index_query(projected)\n",
        "            side.wait_stream(self.index_stream)\n"
        "            with torch.cuda.stream(side):\n                iq, gates = self.index_query(projected)\n",
    ),
    # mode 2 waits for the store on main before the index query (values stay correct)
    "l4_iq_waits_store_m2": (
        V_FILE,
        "        if self.index_q_mode == 2:\n            iq, gates",
        "        if self.index_q_mode == 2:\n            main.wait_stream(self.index_stream)\n            iq, gates",
    ),
}

ROWS, Q, CTX = 8, 8, 12000
GATE_END = 2112 + 128 * HEADS  # verify packs [q_a | kv_a | g_proj]; index-K has its own projection
SPIN = 10_000_000  # cycles of torch.cuda._sleep: several ms, >> the launch time of the rest of the chain

needs_gpu = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 10,
    reason="needs an SM100-family GPU",
)


# ----------------------------------------------------------------------------------------------- the layer
class Layer:
    """Shared weights, pools and inputs; one (core, writer, caches) per arm."""

    def __init__(self, seed=31):
        H.load_plugin(MUTATIONS)
        import deep_gemm
        from sglang.kernels.ops.attention.dsv4.topk import plan_topk_v2
        from sgl_nvidia_mega.kimi_k31.k3.cutedsl.mla.prepared import PreparedMla

        g = H.generator(seed)
        self.g = g
        self.w = w = make_weights(seed)
        self.wk = (torch.randn((128, 7168), generator=g, device="cuda") / math.sqrt(7168)).to(torch.bfloat16)
        self.pages = math.ceil(CTX / 64) + 2
        self.kv0, self.idx0 = kv_pool(self.pages, g), index_pool(self.pages, g)
        npg = math.ceil(CTX / 64)
        self.table = (torch.randperm(self.pages - 1, generator=g, device="cuda")[:npg] + 1).to(torch.int32)[None]
        pos = torch.arange(CTX - Q, CTX, device="cuda")
        self.slots = self.table[0, pos // 64].long() * 64 + pos % 64
        self.lengths = torch.full((1,), CTX, dtype=torch.int32, device="cuda")
        causal = (CTX - Q + 1 + torch.arange(Q, device="cuda")).to(torch.int32)
        contexts = torch.full((1, Q), CTX, dtype=torch.int32, device="cuda")
        self.step = SimpleNamespace(
            sparse=1, batch_requests=1, query_width=Q, causal_lengths=causal, score_contexts=contexts,
            score_schedule=deep_gemm.get_paged_mqa_logits_metadata(contexts, 64, num_sms()),
            topk_plan=plan_topk_v2(causal),
        )
        scratch = torch.empty((ROWS, HEADS, 576), dtype=torch.uint8, device="cuda")
        attention, _ = fakes(w)
        self.base = PreparedMla(attention, {ROWS: GATE_END}, scratch, max_context=1 << 17, core="trtllm", queries=(Q,))
        self.first = torch.empty((ROWS, 7168), dtype=torch.bfloat16, device="cuda")
        self.projected = torch.empty((ROWS, GATE_END), dtype=torch.bfloat16, device="cuda")
        self.hidden_src = torch.empty_like(self.first)
        self.proj_src = torch.empty_like(self.projected)
        self.logits = {}
        self.arm_name = None
        self.new_inputs()

    def new_inputs(self):
        self.hidden_src.copy_(torch.randn(self.hidden_src.shape, generator=self.g, device="cuda").to(torch.bfloat16))
        self.proj_src.copy_(torch.randn(self.proj_src.shape, generator=self.g, device="cuda").to(torch.bfloat16))

    def arm(self, name, mode):
        from sgl_nvidia_mega.kimi_k31.dsa import verify as V
        from sgl_nvidia_mega.kimi_k31.dsa.indexer import IndexWeights
        from sgl_nvidia_mega.kimi_k31.verify_indexer import IndexKeyWeights, PreparedVerifyIndexKey

        w = self.w
        default = V.INDEX_Q_OFF_WRITER
        V.INDEX_Q_OFF_WRITER = mode
        try:
            core = V.PreparedSparseVerifyMla(self.base, IndexWeights(w.wq_b, w.w_proj, w.w_qa, 1e-5), ROWS, 1 << 17)
        finally:
            V.INDEX_Q_OFF_WRITER = default
        assert core.fused_index_q is not None, "the fused index query must serve 8 rows"
        assert (core.query_stream is core.index_stream) == (mode == 0), "arm does not run the path it names"
        kv, idx = self.kv0.clone(), self.idx0.clone()
        core.prepare_cache(kv, prepare_base=not hasattr(self, "_based"))
        self._based = True
        writer = V.SparseVerifyIndexKey(
            PreparedVerifyIndexKey(IndexKeyWeights(self.wk, w.gamma, w.beta, w.k_eps)), core
        )
        assert writer.packed_weight is not None, "8-row writer packs the index-K projection"
        writer.bind(idx, 64, rows=(ROWS,))
        core.step = self.step
        return SimpleNamespace(name=name, core=core, writer=writer, kv=kv, idx=idx,
                               out=torch.empty((ROWS, HEADS * 128), dtype=torch.bfloat16, device="cuda"))

    def poison(self, a):
        """Reset every buffer a consumer could read before its producer wrote it: zeros for the scoring inputs
        (finite, so a stale read gives wrong logits instead of NaN slots), -1 (no slot) for the top-k, NaN outputs."""
        H.poison(a.out, a.core.mla.output)
        self.projected.zero_()
        a.core.fused_index_q.query.zero_()
        a.core.fused_index_q.gates.zero_()
        a.core.selection.selected.fill_(-1)
        page, off = self.slots // 64, self.slots % 64
        for p, o in zip(page.tolist(), off.tolist()):
            a.idx[p, o * 128 : o * 128 + 128] = 0
            a.idx[p, 64 * 128 + o * 4 : 64 * 128 + o * 4 + 4] = 0

    def chain(self, a, delay_projection=False):
        """One layer as K31VerifyExecutor runs it, on the current stream."""
        self.arm_name = a.name
        self.first.copy_(self.hidden_src)  # the layer input producer (AttnRes in the model)
        a.writer.run(self.first, self.slots)
        if delay_projection:
            torch.cuda._sleep(SPIN)
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
            "selected": a.core.selection.selected[:ROWS].clone(),
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


# ----------------------------------------------------------------------------------------------- correctness
@needs_gpu
def test_bitwise_eager(capture_logits, canonical_topk):
    """B vs A bitwise (two input sets), A vs a rerun of A (control), B against the independent references."""
    layer = Layer()
    capture_logits.append(layer)
    A, B, C = layer.arm("A", 0), layer.arm("B", 1), layer.arm("C", 2)
    for it in range(2):
        layer.new_inputs()
        ra, rb, rc, ra2 = (run_eager(layer, a) for a in (A, B, C, A))
        bitwise(f"control_A_rerun.it{it}", ra2, ra)
        bitwise(f"eager_B_vs_A.it{it}", rb, ra)
        bitwise(f"eager_C_vs_A.it{it}", rc, ra)
    references(layer, B, rb)
    references(layer, C, rc)


def references(layer, a, snap):
    w, h = layer.w, HEADS
    projected, hidden = layer.proj_src, layer.hidden_src
    name = f"h{h}"
    q_ref = R.mla_query(projected, w.w_qa, w.w_qb, w.w_uk, 1e-5)
    mk_q = a.core.mla.query[:, 0, :h].view(torch.uint8).contiguous()
    H.record({"kind": "check", **R.fp8_ulp_check(f"verify_query.{name}", mk_q, q_ref)})
    H.record({"kind": "check", **R.fp8_ulp_check(f"verify_kv_row.{name}", snap["kv_rows"],
                                                 R.kv_row(projected, w.w_kva, 1e-5))})
    q_codes, w_ref = R.index_q(projected, hidden, w.w_qa, 1e-5, w.wq_b, w.w_proj)
    pos_of_slot = R.slot_positions(layer.table[0], layer.pages)
    causal = layer.step.causal_lengths
    for j in range(ROWS):
        n = int(causal[j])
        scores = R.index_scores(a.idx, layer.table[0], q_codes[j], w_ref[j], n)
        want = torch.topk(scores, R.TOPK).indices
        got = pos_of_slot[snap["selected"][j].long()]
        assert bool((got >= 0).all()) and bool((got < n).all()), f"{name} row {j}: slot outside the causal prefix"
        overlap = torch.isin(got, want).float().mean().item()
        H.record({"kind": "check", "item": f"verify_sel_overlap.{name}.row{j}", "overlap": overlap})
        assert overlap >= 0.99, f"{name} row {j}: overlap {overlap:.4f} < 0.99 with the fp64 top-2048"
    lat_mk = fi_mla(mk_q, a.kv, snap["selected"], causal, sparse=True)
    gate = projected[:, 2112:GATE_END]
    H.check(f"verify_out.{name}", snap["out"], R.value(lat_mk, w.w_uv, gate, causal), MLA_ATOL, MLA_RTOL)


@needs_gpu
def test_bitwise_graph(capture_logits, canonical_topk):
    """Each arm captured in its own CUDA graph; two replays on new inputs: B vs A bitwise, A vs eager A bitwise."""
    layer = Layer(seed=37)
    capture_logits.append(layer)
    A, B, C = layer.arm("A", 0), layer.arm("B", 1), layer.arm("C", 2)
    graphs = {}
    for a in (A, B, C):
        run_eager(layer, a)  # warm-up (lazy init, logits buffer) before capture
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            layer.chain(a)
        graphs[a.name] = g
    for it in range(2):
        layer.new_inputs()
        snaps = {}
        for a in (A, B, C):
            layer.arm_name = a.name
            graphs[a.name].replay()
            snaps[a.name] = layer.snapshot(a)
        bitwise(f"graph_B_vs_A.it{it}", snaps["B"], snaps["A"])
        bitwise(f"graph_C_vs_A.it{it}", snaps["C"], snaps["A"])
        bitwise(f"graph_A_vs_eager_A.it{it}", snaps["A"], run_eager(layer, A))


@needs_gpu
def test_production_topk(capture_logits):
    """Shipped top-k order (no sorting): logits bitwise and slots as sorted sets bitwise (each after its A-vs-A
    control), B against the references; the pre-registered fallback for order-dependent tensors is recorded."""
    layer = Layer(seed=47)
    capture_logits.append(layer)
    A, B, C = layer.arm("A", 0), layer.arm("B", 1), layer.arm("C", 2)
    layer.new_inputs()
    ra = run_eager(layer, A)
    reruns = [run_eager(layer, A) for _ in range(3)]
    rb, rc = run_eager(layer, B), run_eager(layer, C)
    sets = lambda r: {"selected_sorted": torch.sort(r["selected"], dim=-1).values}  # noqa: E731
    exact = ("logits", "index_k", "kv_rows")
    bitwise("prod_control_A_rerun", {k: reruns[0][k] for k in exact}, {k: ra[k] for k in exact})
    bitwise("prod_control_A_rerun_sets", sets(reruns[0]), sets(ra))
    for name, r in (("B", rb), ("C", rc)):
        bitwise(f"prod_{name}_vs_A", {k: r[k] for k in exact}, {k: ra[k] for k in exact})
        bitwise(f"prod_{name}_vs_A_sets", sets(r), sets(ra))
        for key in ("selected", "latent", "out"):
            f = (lambda t: t.long()) if key == "selected" else (lambda t: t.float())
            env = max(float((f(x[key]) - f(ra[key])).abs().max()) for x in reruns)
            got = float((f(r[key]) - f(ra[key])).abs().max())
            row = {"kind": "fallback", "item": f"prod.h{HEADS}.{name}.{key}", "max_abs_arm_minus_A": got,
                   "max_abs_A_minus_rerun_over_3": env, "pass_as_written": got <= env}
            H.record(row)
            print("FALLBACK", row, flush=True)
    references(layer, B, rb)
    references(layer, C, rc)


# ----------------------------------------------------------------------------------------------- ordering
DELAYS = ("d1_projection", "d2_index_query", "d3_store", "d4_topk")


def install_delay(monkeypatch, delay, arms):
    if delay == "d2_index_query":
        for a in arms:
            fq = a.core.fused_index_q
            orig = fq.run
            monkeypatch.setattr(fq, "run", lambda *x, _o=orig, **k: (torch.cuda._sleep(SPIN), _o(*x, **k))[1])
    elif delay == "d3_store":
        import sglang.kernels.ops.kvcache.dsa_fused_store_index_cache as M

        orig = M.fused_store_index_k_cache
        monkeypatch.setattr(M, "fused_store_index_k_cache",
                            lambda *x, **k: (torch.cuda._sleep(SPIN), orig(*x, **k))[1])
    elif delay == "d4_topk":
        import sglang.kernels.ops.attention.dsv4.topk as T

        orig = T.topk_transform_paged_v2
        monkeypatch.setattr(T, "topk_transform_paged_v2", lambda *x, **k: (torch.cuda._sleep(SPIN), orig(*x, **k))[1])


@needs_gpu
@pytest.mark.parametrize("delay", DELAYS)
def test_ordering(delay, capture_logits, canonical_topk, monkeypatch):
    """A spin kernel before one producer, consumers' inputs poisoned: both arms equal the undelayed arm A bitwise."""
    layer = Layer(seed=41)
    capture_logits.append(layer)
    A, B, C = layer.arm("A", 0), layer.arm("B", 1), layer.arm("C", 2)
    layer.new_inputs()
    ref = run_eager(layer, A)
    install_delay(monkeypatch, delay, (A, B, C))
    for a in (A, B, C):
        layer.poison(a)
        bitwise(f"{delay}.{a.name}_vs_undelayed_A", run_eager(layer, a, delay_projection=delay == "d1_projection"), ref)


@needs_gpu
def test_engaged(capture_logits, monkeypatch):
    """With the store delayed, B's index query completes before the store (A: after it)."""
    import sglang.kernels.ops.kvcache.dsa_fused_store_index_cache as M

    layer = Layer(seed=43)
    capture_logits.append(layer)
    A, B, C = layer.arm("A", 0), layer.arm("B", 1), layer.arm("C", 2)
    layer.new_inputs()
    events = {}

    def stamp(key):
        e = torch.cuda.Event(enable_timing=True)
        e.record()
        events[key] = e

    store = M.fused_store_index_k_cache
    monkeypatch.setattr(M, "fused_store_index_k_cache",
                        lambda *x, **k: (torch.cuda._sleep(SPIN), store(*x, **k), stamp("store"))[1])
    for a in (A, B, C):
        fq = a.core.fused_index_q
        monkeypatch.setattr(fq, "run", lambda *x, _o=fq.run, **k: (_o(*x, **k), stamp("index_q"))[0])
    result = {}
    for a in (A, B, C):
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            stamp("start")
            layer.chain(a)
        torch.cuda.synchronize()
        t_store = events["start"].elapsed_time(events["store"])
        t_iq = events["start"].elapsed_time(events["index_q"])
        result[a.name] = (t_store, t_iq)
        H.record({"kind": "engaged", "item": f"h{HEADS}.{a.name}", "store_end_ms": t_store, "index_q_end_ms": t_iq})
        print("ENGAGED", a.name, t_store, t_iq, flush=True)
    sa, qa = result["A"]
    assert qa >= sa, f"arm A: index query ended {sa - qa:.3f} ms before the store (shipped order broken)"
    for name in ("B", "C"):
        sb, qb = result[name]
        assert sb - qb >= 1.0, f"arm {name}: index query ended only {sb - qb:.3f} ms before the delayed store (not engaged)"
