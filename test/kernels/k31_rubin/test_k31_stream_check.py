# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""K-18: the one-time fail-closed stream check on the verify path's tvm-ffi launches
(kimi_k31/dsa/stream_check.py + .cuh, called from dsa/verify.py).

The selection stream's event joins (verify.py: side.wait_stream(main) ... main.wait_stream(side)) and the PDL edges
inside it (DeepGEMM logits -> topk_v2; TSI prep -> scan -> merge) order the DCP1 / TSI tvm-ffi launches only if
tvm-ffi launches them on that stream. stream_check.ensure resolves tvm-ffi's launch stream for a tensor of the
launch's device (as LaunchKernel does) and raises before the context launches anything when it is not the expected
stream; once per (device, capture id | "eager", torch's current stream, expected stream).

One GPU, SM100 family, K31_TEST_HEADS=6 (the layer tests reuse test_k31_tsi_verify's DCP1 / TSI arms).
- test_contexts_resolve_torch_stream: in the default, a side and a third stream context, tvm-ffi resolves torch's
  current stream (the K-18 assumption, measured) and each context records its key.
- test_wrong_expected_trips: a context whose launches would go to another stream than the expected one raises, twice
  (a refused check marks nothing); the error names both handles. Bugs: no comparison; key marked before the check.
- test_tvmffi_divergence_trips (K-18's literal case): the probe gets a tvm_ffi tensor, so tvm-ffi finds no torch
  stream and resolves its own env stream -> raises; control: the reported launch stream differs from the expected.
- test_eager_key_per_stream: after a pass on stream A, a divergent context on stream B is checked again (the
  eager key carries the stream). Bug: a device-only eager key.
- test_capture_rechecks: after an eager pass on stream X, a divergent graph capture on X is checked again and raises;
  a clean capture records a key with its capture id. Bug: the key ignores the capture.
- test_layer_checks_contexts: the real verify layer (DCP1 arm, TSI arm), eager and captured, records keys for the
  selection stream and main; outputs as test_k31_tsi_verify (selection exact, T == D bitwise). Bugs: verify.py not
  calling the check; the selection launched outside the selection stream.
- test_layer_fails_closed: with the probe diverging, the TSI layer raises before the selection: the layer output
  stays poison, the TSI merge never ran (peers' select count unchanged, fallback flags still 7).

Mutations (``K31_MUTATION=<name>`` via _harness / the fixed run_suite.sh; each must FAIL this file):
  sc_cpp_nocheck         the probe compares nothing                       -> wrong_expected, tvmffi_divergence
  sc_mark_first          the key is marked before the check              -> wrong_expected (second call)
  sc_eager_device_only   the key is (device, capture) only               -> eager_key_per_stream
  sc_capture_ignored     the key ignores the capture                      -> capture_rechecks
  sc_verify_unchecked    verify.py's check is a no-op                     -> layer_checks_contexts, layer_fails_closed
  sc_selection_on_main   the DCP1 / TSI selection launches on main        -> layer_checks_contexts (raises)
"""

from __future__ import annotations

import pytest
import torch

import _harness as H

S_FILE = "kimi_k31/dsa/stream_check.py"
V_FILE = "kimi_k31/dsa/verify.py"
MUTATIONS = {
    "sc_cpp_nocheck": (
        "kimi_k31/dsa/stream_check.cuh",
        "RuntimeCheck(launch_stream == expected_stream,",
        "RuntimeCheck(launch_stream == launch_stream,",
    ),
    "sc_mark_first": (
        S_FILE,
        "    _load().check(like, expected)\n    _CHECKED.add(k)\n",
        "    _CHECKED.add(k)\n    _load().check(like, expected)\n",
    ),
    "sc_eager_device_only": (
        S_FILE,
        "return (device_index, capture, int(current), int(expected))",
        "return (device_index, capture)",
    ),
    "sc_capture_ignored": (S_FILE, "        capture = int(info[2])\n", "        capture = \"eager\"\n"),
    "sc_verify_unchecked": (
        V_FILE,
        "from .stream_check import ensure as ensure_stream\n",
        "ensure_stream = lambda *args, **kwargs: None  # noqa: E731\n",
    ),
    "sc_selection_on_main": (
        V_FILE,
        "        with torch.cuda.stream(side):\n            ensure_stream(projected, side)  # DCP1",
        "        with torch.cuda.stream(main):\n            ensure_stream(projected, side)  # DCP1",
    ),
}
H.load_plugin(MUTATIONS)  # first load wins: test_k31_tsi_verify's own load_plugin returns this module

import test_k31_tsi_verify as W  # noqa: E402

needs_gpu = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 10,
    reason="needs an SM100-family GPU",
)


def _sc():
    from sgl_nvidia_mega.kimi_k31.dsa import stream_check as SC

    return SC


class _Foreign:
    """K-18's literal failure: the probe gets a tvm_ffi tensor, so tvm-ffi sees no torch tensor and keeps its own env
    stream instead of torch's current stream."""

    def __init__(self, module):
        self.module = module

    def check(self, like, expected):
        import tvm_ffi

        return self.module.check(tvm_ffi.from_dlpack(like), expected)


@pytest.fixture
def sc():
    SC = _sc()
    SC._CHECKED.clear()
    yield SC
    SC._CHECKED.clear()


@pytest.fixture
def foreign(monkeypatch, sc):
    real = sc._load()
    monkeypatch.setattr(sc, "_load", lambda: _Foreign(real))
    return sc


def _handles(text):
    launch = int(text.split("launch stream ")[1].split()[0])
    expected = int(text.split("expected stream ")[1].split(";")[0])
    return launch, expected


@needs_gpu
def test_contexts_resolve_torch_stream(sc):
    x = torch.zeros(16, device="cuda")
    a, b = torch.cuda.Stream(), torch.cuda.Stream()
    dev = x.device.index
    sc.ensure(x, torch.cuda.current_stream())
    for s in (a, b):
        with torch.cuda.stream(s):
            sc.ensure(x, s)
    want = {(dev, "eager", h, h) for h in (torch.cuda.current_stream().cuda_stream, a.cuda_stream, b.cuda_stream)}
    assert sc._CHECKED == want, f"recorded keys {sc._CHECKED} != {want}"


@needs_gpu
def test_wrong_expected_trips(sc):
    x = torch.zeros(16, device="cuda")
    side, other = torch.cuda.Stream(), torch.cuda.Stream()
    for attempt in range(2):  # fail closed every time: a refused check must not mark its key
        with torch.cuda.stream(other), pytest.raises(Exception, match="K-18 stream check") as err:
            sc.ensure(x, side)
        launch, expected = _handles(str(err.value))
        assert (launch, expected) == (other.cuda_stream, side.cuda_stream), f"attempt {attempt}: {err.value}"
    assert not sc._CHECKED, f"a refused check marked its key: {sc._CHECKED}"


@needs_gpu
def test_tvmffi_divergence_trips(foreign):
    x = torch.zeros(16, device="cuda")
    side = torch.cuda.Stream()
    with torch.cuda.stream(side), pytest.raises(Exception, match="K-18 stream check") as err:
        foreign.ensure(x, side)
    launch, expected = _handles(str(err.value))
    assert expected == side.cuda_stream and launch != expected, f"the forced divergence did not diverge: {err.value}"
    assert not foreign._CHECKED


@needs_gpu
def test_eager_key_per_stream(sc, monkeypatch):
    x = torch.zeros(16, device="cuda")
    a, b = torch.cuda.Stream(), torch.cuda.Stream()
    with torch.cuda.stream(a):
        sc.ensure(x, a)  # real probe: passes, records stream A's key
    real = sc._load()
    monkeypatch.setattr(sc, "_load", lambda: _Foreign(real))
    with torch.cuda.stream(b), pytest.raises(Exception, match="K-18 stream check"):
        sc.ensure(x, b)  # a new eager stream must be checked again (here it diverges)


@needs_gpu
def test_capture_rechecks(sc, monkeypatch):
    x = torch.zeros(16, device="cuda")
    s = torch.cuda.Stream()
    with torch.cuda.stream(s):
        sc.ensure(x, s)  # eager pass on s
    torch.cuda.synchronize()
    real = sc._load()
    monkeypatch.setattr(sc, "_load", lambda: _Foreign(real))
    bad = torch.cuda.CUDAGraph()
    with pytest.raises(Exception, match="K-18 stream check"):
        with torch.cuda.graph(bad, stream=s):
            sc.ensure(x, s)  # same streams as the eager pass, but a capture: checked again
    monkeypatch.setattr(sc, "_load", lambda: real)
    before = set(sc._CHECKED)
    good = torch.cuda.CUDAGraph()
    with torch.cuda.graph(good, stream=s):
        sc.ensure(x, s)
        y = x + 1  # the capture needs one op
    new = set(sc._CHECKED) - before
    assert len(new) == 1 and all(k[1] != "eager" and k[2] == k[3] == s.cuda_stream for k in new), new
    good.replay()
    torch.cuda.synchronize()
    assert bool((y == 1).all())


@needs_gpu
def test_layer_checks_contexts(sc, sort_dcp1):
    layer, D, Tarm, peers = W.build(61)
    sort_dcp1.append(D.core.selection.selected)
    dev = layer.slots.device.index
    for a in (D, Tarm):
        sc._CHECKED.clear()
        layer.new_inputs()
        H.poison(a.out)
        main = torch.cuda.Stream()
        main.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(main):
            layer.chain(a)
        torch.cuda.current_stream().wait_stream(main)
        torch.cuda.synchronize()
        side, m = a.core.query_stream.cuda_stream, main.cuda_stream
        keys = {k for k in sc._CHECKED if k[0] == dev and k[1] == "eager"}
        assert (dev, "eager", side, side) in keys, f"arm {a.name}: selection stream not checked ({keys})"
        assert (dev, "eager", m, m) in keys, f"arm {a.name}: main context not checked ({keys})"
    d, t = W.run(layer, D), W.run(layer, Tarm)
    W.check("stream_check.eager", layer, D, Tarm, peers, d, t)
    for a in (D, Tarm):
        before = set(sc._CHECKED)
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            layer.chain(a)
        side = a.core.query_stream.cuda_stream
        new = {k for k in set(sc._CHECKED) - before if k[1] != "eager"}
        assert any(k[2] == k[3] == side for k in new), f"arm {a.name}: capture did not check the selection stream"
        H.poison(a.out)
        g.replay()
        torch.cuda.synchronize()
        assert not bool((a.out.view(torch.uint8) == 0xFF).all()), f"arm {a.name}: replay wrote no output"


@needs_gpu
def test_layer_fails_closed(foreign, sort_dcp1):
    layer, D, Tarm, peers = W.build(67)
    sort_dcp1.append(D.core.selection.selected)
    layer.new_inputs()
    H.poison(Tarm.out)
    peers.states[0]._fallback.fill_(7)
    selects = peers.selects
    with pytest.raises(Exception, match="K-18 stream check"):
        W.run(layer, Tarm)
    torch.cuda.synchronize()
    assert peers.selects == selects, "the TSI selection ran although the stream check refused"
    assert bool((peers.states[0]._fallback == 7).all()), "the TSI merge wrote its flags although the check refused"
    assert bool((Tarm.out.view(torch.uint8) == 0xFF).all()), "the layer wrote its output although the check refused"


sort_dcp1 = W.sort_dcp1
