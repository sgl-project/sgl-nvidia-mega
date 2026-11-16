# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""K3.1 MK sparse MLA verify layer (Breq 1 x Q8, K31_TEST_HEADS heads): the index-K writer's LayerNorm launched with
PDL (``K31MK_IDXK_LN_PDL=1``). The layer, arms and references come from test_k31_verify_streams.py.

Arm C: the development base H schedule (K31MK_INDEX_Q_OFF_WRITER=2), LayerNorm launched as stock launches it (no PDL).
Arm D: C + K31MK_IDXK_LN_PDL=1 (FlashInfer's CuTe LayerNorm, enable_pdl=True, into a fixed output buffer).

Gates (written before any run, never loosened):
- D vs C BITWISE: DeepGEMM logits over the context columns, top-k slots (each row sorted on top-k's stream, both arms,
  as in test_k31_verify_streams), sparse-core latent, output, index-K rows and MLA KV rows; eager (two input sets, with
  the C-vs-C-rerun control) and CUDA-graph replay (two input sets, plus graph D vs eager D). D also passes the
  independent references of test_k31_verify_streams (query codes, KV rows, top-k overlap >= 0.99 with the fp64
  top-2048, output vs the stock core).
- Ordering: an early-trigger producer right before the LayerNorm (it releases its dependents at entry, waits ~5 ms on
  %globaltimer, then writes the LayerNorm input, which was set to NaN before it): C and D stay bitwise equal to the
  undelayed C.
- Engagement: with that producer, D's LayerNorm starts >= 1 ms before the producer ends (it launched early and waited
  inside), C's starts after it ends (torch profiler / CUPTI kernel stamps).

Mutations (each must FAIL the suite; fixed harness run_suite.sh): see MUTATIONS.
"""

from __future__ import annotations

import json
import os
import tempfile

import pytest
import torch

import _harness as H
from test_k31_verify_streams import (  # noqa: F401  (fixtures are used by name)
    Layer,
    bitwise,
    canonical_topk,
    capture_logits,
    needs_gpu,
    references,
    run_eager,
)

V_FILE = "kimi_k31/dsa/verify.py"
LN_CALL = "    layernorm_cute(out, x, w.norm, w.bias, w.eps, enable_pdl=True)\n"
MUTATIONS = {
    # the knob changes the buffer but not the launch: no PDL (values stay right; only the engagement test sees it)
    "lnpdl_nopdl": (V_FILE, LN_CALL, LN_CALL.replace("enable_pdl=True", "enable_pdl=False")),
    # gamma and beta swapped in the knob path (a wrong argument order)
    "lnpdl_gb_swap": (V_FILE, LN_CALL, LN_CALL.replace("w.norm, w.bias", "w.bias, w.norm")),
    # the LayerNorm on an unordered side stream (a lost stream dependency)
    "lnpdl_stream": (
        V_FILE,
        LN_CALL,
        "    with torch.cuda.stream(torch.cuda.Stream(device=x.device)):\n    " + LN_CALL,
    ),
}

DELAY_NS = 5_000_000  # 5 ms: >> the launch time of the rest of the chain


class LnLayer(Layer):
    def __init__(self, seed=31):
        H.load_plugin(MUTATIONS)  # first: Layer's own load_plugin call then returns this (possibly mutated) copy
        super().__init__(seed)

    def ln_arm(self, name, ln_pdl):
        from sgl_nvidia_mega.kimi_k31.dsa import verify as V

        old = V.IDXK_LN_PDL
        V.IDXK_LN_PDL = ln_pdl
        try:
            a = self.arm(name, 2)
        finally:
            V.IDXK_LN_PDL = old
        assert (a.writer.key is not None) == ln_pdl, "arm does not run the LayerNorm path it names"
        return a


# ----------------------------------------------------------------------------------------------- early producer
_EARLY = {}


def _early_copy_code(src, dst):
    """A kernel that releases its PDL dependents at entry, waits for its own predecessor, spins DELAY_NS on
    %globaltimer, then copies src -> dst (one CTA per row, one thread per column)."""
    import cutlass.cute as cute
    from cutlass import Int64
    from cutlass.cute.runtime import make_fake_stream
    from sgl_nvidia_mega.kimi_k31.k3.cutedsl.common.host import stream, to_cute
    from sgl_nvidia_mega.kimi_k31.k3.cutedsl.common.primitives import global_timer_ns, launch

    class EarlyCopy:
        @cute.jit
        def __call__(self, src: cute.Tensor, dst: cute.Tensor, stream):
            # Not a PDL secondary itself (it waits for the poison fill); its dependents still launch at its entry.
            launch(self.kernel, (src, dst, stream), src.shape[0], 128, pdl=False)

        @cute.kernel
        def kernel(self, args):
            src, dst = args
            row, _, _ = cute.arch.block_idx()
            tid, _, _ = cute.arch.thread_idx()
            cute.arch.griddepcontrol_launch_dependents()
            cute.arch.griddepcontrol_wait()
            start = global_timer_ns()
            now = global_timer_ns()
            while now - start < Int64(DELAY_NS):
                now = global_timer_ns()
            dst[row, tid] = src[row, tid]

    views = (to_cute(src), to_cute(dst))
    key = (tuple(src.shape), tuple(dst.stride()))
    if key not in _EARLY:
        _EARLY[key] = cute.compile(EarlyCopy(), *views, make_fake_stream())
    code = _EARLY[key]
    return lambda: code(*views, stream(src.device))


def install_early_producer(monkeypatch):
    """Before each LayerNorm (both launch paths): stash its input, set it to NaN, then the early-trigger copy restores
    it ~5 ms later. A LayerNorm that reads its input before waiting for that copy reads NaN."""
    import flashinfer.norm as N
    from sgl_nvidia_mega.kimi_k31.dsa import verify as V

    stash = {}

    def prepare(x):
        buf = stash.get(x.data_ptr())
        if buf is None:
            buf = stash[x.data_ptr()] = torch.empty(x.shape, dtype=x.dtype, device=x.device)
        buf.copy_(x)
        x.fill_(float("nan"))
        _early_copy_code(buf, x)()

    default_ln, pdl_ln = N.layernorm, V._layernorm_pdl
    monkeypatch.setattr(N, "layernorm", lambda x, *a, **k: (prepare(x), default_ln(x, *a, **k))[1])
    monkeypatch.setattr(V, "_layernorm_pdl", lambda out, x, w: (prepare(x), pdl_ln(out, x, w))[1])


# ----------------------------------------------------------------------------------------------- correctness
@needs_gpu
def test_lnpdl_bitwise_eager(capture_logits, canonical_topk):
    """D vs C bitwise (two input sets), C vs a rerun of C (control), D against the independent references."""
    layer = LnLayer()
    capture_logits.append(layer)
    C, D = layer.ln_arm("C", False), layer.ln_arm("D", True)
    for it in range(2):
        layer.new_inputs()
        rc, rd, rc2 = (run_eager(layer, a) for a in (C, D, C))
        bitwise(f"lnpdl_control_C_rerun.it{it}", rc2, rc)
        bitwise(f"lnpdl_eager_D_vs_C.it{it}", rd, rc)
    references(layer, D, rd)


@needs_gpu
def test_lnpdl_bitwise_graph(capture_logits, canonical_topk):
    """Each arm in its own CUDA graph; two replays on new inputs: D vs C bitwise, graph D vs eager D bitwise."""
    layer = LnLayer(seed=37)
    capture_logits.append(layer)
    C, D = layer.ln_arm("C", False), layer.ln_arm("D", True)
    graphs = {}
    for a in (C, D):
        run_eager(layer, a)  # warm-up before capture
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            layer.chain(a)
        graphs[a.name] = g
    for it in range(2):
        layer.new_inputs()
        snaps = {}
        for a in (C, D):
            layer.arm_name = a.name
            graphs[a.name].replay()
            snaps[a.name] = layer.snapshot(a)
        bitwise(f"lnpdl_graph_D_vs_C.it{it}", snaps["D"], snaps["C"])
        bitwise(f"lnpdl_graph_D_vs_eager_D.it{it}", snaps["D"], run_eager(layer, D))


@needs_gpu
def test_lnpdl_ordering(capture_logits, canonical_topk, monkeypatch):
    """Early-trigger producer before the LayerNorm, its input NaN until the producer writes it: both arms equal the
    undelayed C bitwise."""
    layer = LnLayer(seed=41)
    capture_logits.append(layer)
    C, D = layer.ln_arm("C", False), layer.ln_arm("D", True)
    layer.new_inputs()
    ref = run_eager(layer, C)
    install_early_producer(monkeypatch)
    for a in (C, D):
        layer.poison(a)
        bitwise(f"lnpdl_early_producer.{a.name}_vs_undelayed_C", run_eager(layer, a), ref)


def _kernel_stamps(fn):
    from torch.profiler import ProfilerActivity, profile

    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        fn()
        torch.cuda.synchronize()
    path = os.path.join(tempfile.mkdtemp(prefix="lnpdl_"), "trace.json")
    prof.export_chrome_trace(path)
    events = [e for e in json.load(open(path))["traceEvents"] if e.get("cat") == "kernel"]
    return sorted(events, key=lambda e: e["ts"])


@needs_gpu
def test_lnpdl_engaged(capture_logits, monkeypatch):
    """With the early-trigger producer: D's LayerNorm starts >= 1 ms before the producer ends, C's after it ends."""
    layer = LnLayer(seed=43)
    capture_logits.append(layer)
    C, D = layer.ln_arm("C", False), layer.ln_arm("D", True)
    layer.new_inputs()
    install_early_producer(monkeypatch)
    for a in (C, D):
        run_eager(layer, a)  # compile and warm outside the profile
    result = {}
    for a in (C, D):
        ev = _kernel_stamps(lambda: run_eager(layer, a))
        ln = [e for e in ev if "layernorm" in e["name"].lower()]
        pub = [e for e in ev if "earlycopy" in e["name"].lower()]
        assert len(ln) == 1 and len(pub) == 1, f"arm {a.name}: {len(ln)} LayerNorm / {len(pub)} producer kernels"
        lead_us = (pub[0]["ts"] + pub[0]["dur"]) - ln[0]["ts"]  # > 0: the LayerNorm started before the producer ended
        result[a.name] = lead_us
        H.record({"kind": "engaged", "item": f"lnpdl.h{os.environ.get('K31_TEST_HEADS', '')}.{a.name}",
                  "ln_start_before_producer_end_us": lead_us})
        print("ENGAGED", a.name, lead_us, flush=True)
    assert result["C"] <= 1.0, f"arm C: LayerNorm started {result['C']:.1f} us before its producer ended"
    assert result["D"] >= 1000.0, f"arm D: LayerNorm started only {result['D']:.1f} us before the producer ended"
