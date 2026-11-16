# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""L38 (K31MK_SPARSE_MLA6 core) and TSI (K31MK_TSI selection) together on one sparse verify layer: the merged
kimi_k31/dsa/verify.py (TSI swaps the selection, L38 swaps the core).

One GPU, SM100 family, K31_TEST_HEADS=6. The layer of test_k31_tsi_verify (Breq 1 x Q8, context 12000, SORTED page
table, 15 TSI peer ranks emulated on this GPU) with K31MK_SPARSE_MLA6=1 set before the plugin is imported, so BOTH arms
run L38's core (sparse_mla6 + MlaValueFold):
  arm D = DCP1 selection (slots sorted on top-k's stream) + L38 core  = L38 alone
  arm T = TSI selection (step.sparse == 2)               + L38 core  = both changes, in the merged order

References and gates (exact, no tolerance; fixed before any run):
- selection: T's and D's selected == the exact top-2048 (test_k31_tsi_verify.reference: torch stable sort of the
  full-context DeepGEMM logits; independent of TSI and of L38).
- downstream: T's latent and layer output == D's, bitwise. L38's core is a function of (slots in order, query, KV),
  and the merge hands it the same selection tensor on either path, so "both on" == "TSI's selection, then L38's
  core" == L38 alone on the same slots.
- engagement: per step, SparseMla6.run ran once per arm, the stock trtllm-gen core (SparseMla.run_core) never ran,
  and TSI's select ran on the T step only.
- K-18 (L38's one-time fail-closed stream check): each arm's graph capture adds exactly one new (device, capture id)
  key to sparse_mla6._STREAM_CHECKED, i.e. the first launch inside EVERY capture went through run_checked (the "tsi"
  host graph is a separate capture). Two replays == eager runs of the same inputs, bitwise.

Mutations (``K31_MUTATION=<name>`` through _harness / the fixed run_suite.sh; each must FAIL this file):
  l38tsi_tsi_off_under_l38  the merged run() drops TSI's selection when L38 is on   -> both tests (engagement / selection)
  l38tsi_l38_off_on_tsi     the merged run() drops L38's core on a tsi step         -> both tests
  l38tsi_capture_key        the K-18 key ignores the capture (one check per device) -> test_l38_tsi_layer_graph
"""

from __future__ import annotations

import os

os.environ["K31MK_SPARSE_MLA6"] = "1"  # before the plugin (and verify.py's knob block) is imported

import pytest  # noqa: E402
import torch  # noqa: E402

import _harness as H  # noqa: E402

V_FILE = "kimi_k31/dsa/verify.py"
M_FILE = "kimi_k31/dsa/sparse_mla6.py"
MUTATIONS = {
    "l38tsi_tsi_off_under_l38": (V_FILE, "            if step.sparse == 2:  # TSI",
                                 "            if step.sparse == 2 and self.mla6 is None:  # TSI"),
    "l38tsi_l38_off_on_tsi": (V_FILE, "        if self.mla6 is not None:  # L38",
                              "        if self.mla6 is not None and step.sparse != 2:  # L38"),
    "l38tsi_capture_key": (M_FILE, "        capture = int(info[2])\n", "        capture = \"eager\"\n"),
}
H.load_plugin(MUTATIONS)  # first load wins: test_k31_tsi_verify's own load_plugin returns this module

import test_k31_tsi_verify as W  # noqa: E402
from test_k31_tsi_verify import needs_gpu, sort_dcp1  # noqa: E402,F401  (fixture)


def _count(monkeypatch, cls, name, box):
    orig = getattr(cls, name)

    def wrapped(self, *args, **kwargs):
        box[name] = box.get(name, 0) + 1
        return orig(self, *args, **kwargs)

    monkeypatch.setattr(cls, name, wrapped)


def _arms(seed, sort_dcp1):
    layer, D, Tarm, peers = W.build(seed)
    for a in (D, Tarm):
        assert a.core.mla6 is not None, f"arm {a.name}: L38 core not built (K31MK_SPARSE_MLA6=1, heads {W.HEADS})"
    assert D.core.tsi is None and Tarm.core.tsi is not None
    sort_dcp1.append(D.core.selection.selected)
    return layer, D, Tarm, peers


@needs_gpu
def test_l38_tsi_layer_exact(sort_dcp1, monkeypatch):
    layer, D, Tarm, peers = _arms(53, sort_dcp1)
    calls = {}
    _count(monkeypatch, type(D.core.mla6), "run", calls)
    _count(monkeypatch, type(D.core.mla), "run_core", calls)
    for it in range(2):
        layer.new_inputs()
        H.poison(D.out, Tarm.out, D.core.mla6.partial, Tarm.core.mla6.partial)
        peers.states[0]._fallback.fill_(7)
        selects, core = peers.selects, calls.get("run", 0)
        d = W.run(layer, D)
        assert peers.selects == selects, "the DCP1 step ran the TSI selection"
        t = W.run(layer, Tarm)
        assert peers.selects == selects + 1, "the tsi-labelled step did not run the TSI selection"
        assert calls.get("run", 0) == core + 2, "L38's core did not run once per arm"
        assert calls.get("run_core", 0) == 0, "the stock trtllm-gen core ran with K31MK_SPARSE_MLA6=1"
        flags = peers.states[0]._fallback[:W.ROWS]
        assert bool(((flags == 0) | (flags == 1)).all()), f"TSI merge did not write its fallback flags: {flags.tolist()}"
        W.check(f"l38tsi.eager.it{it}", layer, D, Tarm, peers, d, t)
        print("L38TSI_LAYER eager", it, "fallback", flags.tolist(), flush=True)


@needs_gpu
def test_l38_tsi_layer_graph(sort_dcp1):
    from sgl_nvidia_mega.kimi_k31.dsa import sparse_mla6 as M6

    layer, D, Tarm, peers = _arms(59, sort_dcp1)
    graphs = {}
    for a in (D, Tarm):
        W.run(layer, a)  # warm-up (lazy init, eager key) before capture
        before = set(M6._STREAM_CHECKED)
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            layer.chain(a)
        new = set(M6._STREAM_CHECKED) - before
        assert len(new) == 1 and all(k[1] != "eager" for k in new), \
            f"arm {a.name}: the K-18 stream check did not run in this capture (new keys {new})"
        graphs[a.name] = g
    for it in range(2):
        layer.new_inputs()
        snaps = {}
        for a in (D, Tarm):
            H.poison(a.out)
            graphs[a.name].replay()
            snaps[a.name] = W.snap(layer, a)
        W.check(f"l38tsi.graph.it{it}", layer, D, Tarm, peers, snaps["D"], snaps["T"])
        eager = W.run(layer, Tarm)
        for key in eager:
            assert torch.equal(eager[key].view(torch.uint8), snaps["T"][key].view(torch.uint8)), f"graph != eager: {key}"
