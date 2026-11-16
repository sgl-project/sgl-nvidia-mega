# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""KNOWN K-13 for L38 (K31MK_SPARSE_MLA6): the sparse MLA6 core -> MlaValueFold PDL edge orders the two
kernels only if both run on ONE stream. The core is a tvm-ffi JIT kernel (launch stream = tvm-ffi's env stream);
the fold is a CuTe kernel launched on an explicit stream (verify.py: torch's current stream). SparseMla6.run's first
launch per device and per graph capture goes through Binding::run_checked, which compares tvm-ffi's launch stream with
the fold's and launches nothing when they differ (fail closed).

Each test poisons (NaN) the core's partial / stats (and, in the layer test, the layer output) before the call, so
"nothing launched" is read from memory, not assumed. One GPU, SM100 family; K31_TEST_HEADS=6.

- test_checked_launch_bitwise: the checked first launch writes exactly what an unchecked launch writes
  (bitwise). Bug caught: run_checked launching nothing or other arguments.
- test_other_stream_trips (M2): the core under another torch stream than the fold's -> raises "K-13", partial / stats
  still poison; a second call raises again. Bugs caught: no comparison; the key marked before the check passes.
- test_tvmffi_stream_trips (M1, K-13's literal case): tvm-ffi resolves a stream other than torch's current one
  (the binding gets tvm_ffi tensors, so tvm-ffi finds no torch stream) -> raises, nothing launched; the error names a
  launch stream different from the expected one (control: the forced divergence really diverged).
- test_capture_rechecks: an eager launch passes; then the first launch of a graph capture that names another
  dependent stream raises (the check is per capture, not only per device); a capture with the right stream records,
  and its replay writes the eager outputs bitwise. Bug caught: a device-only key.
- test_layer_fails_closed: PreparedSparseVerifyMla.run (knob on, Breq 1 x Q8, context 8704) with M1 forced raises;
  the layer output and the partials stay poison (no core output, the fold never ran).

Mutations (``K31_MUTATION=<name>`` through _harness / the fixed run_suite.sh; each must FAIL this file):
  k13_cpp_nocheck  run_checked compares nothing       -> other_stream, tvmffi_stream, layer_fails_closed
  k13_device_only  the check key ignores the capture  -> capture_rechecks
  k13_mark_first   the key is marked before the check -> other_stream (second call)
Not covered here: verify.py passing a stream other than the fold's (``stream=main.cuda_stream``, main == s by
construction in the same frame); a code read covers that line.
"""

from __future__ import annotations

import os

import pytest
import torch

import _harness as H

MUTATIONS = {
    "k13_cpp_nocheck": (
        "kimi_k31/dsa/sparse_mla6.cuh",
        "RuntimeCheck(launch_stream == expected_stream,",
        "RuntimeCheck(launch_stream == launch_stream,",
    ),
    "k13_device_only": ("kimi_k31/dsa/sparse_mla6.py", "key = (device_index, capture)", "key = (device_index, 0)"),
    "k13_mark_first": (
        "kimi_k31/dsa/sparse_mla6.py",
        "        self.module.run_checked(*args, int(stream))\n        _STREAM_CHECKED.add(key)\n",
        "        _STREAM_CHECKED.add(key)\n        self.module.run_checked(*args, int(stream))\n",
    ),
}
H.load_plugin(MUTATIONS)

from test_k31_verify_streams_l38 import BMM1, HEADS, build_layer, case_inputs  # noqa: E402

needs_gpu = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 10,
    reason="needs an SM100-family GPU",
)
ROWS = 8


def _mod():
    from sgl_nvidia_mega.kimi_k31.dsa import sparse_mla6 as M

    return M


def _reset():
    _mod()._STREAM_CHECKED.clear()


class _ForeignStream:
    """M1: hand the binding tvm_ffi tensors, so tvm-ffi sees no torch tensor and keeps its own env stream (NULL)
    instead of torch's current stream."""

    def __init__(self, module):
        self.module = module

    @staticmethod
    def _conv(args):
        import tvm_ffi

        return [tvm_ffi.from_dlpack(a) if isinstance(a, torch.Tensor) else a for a in args]

    def run(self, *args):
        return self.module.run(*self._conv(args))

    def run_checked(self, *args):
        return self.module.run_checked(*self._conv(args))


def _core_and_inputs(seed):
    core = _mod().SparseMla6(torch.device("cuda"), ROWS, HEADS, 128)
    query, kv, selected, lengths, _ = case_inputs(ROWS, "full", seed)
    return core, (query, kv, selected, lengths, (BMM1, 1.0))


def _assert_poison(core, what):
    torch.cuda.synchronize()
    assert bool(torch.isnan(core.partial).all()) and bool(torch.isnan(core.stats).all()), f"{what}: core wrote outputs"


def _assert_written(core, what):
    assert not bool(torch.isnan(core.partial).any()), f"{what}: partial has unwritten (NaN) cells"
    assert not bool(torch.isnan(core.stats).any()), f"{what}: stats have unwritten (NaN) cells"


@needs_gpu
def test_checked_launch_bitwise():
    core, inp = _core_and_inputs(1301)
    side = torch.cuda.Stream()
    _reset()
    H.poison(core.partial, core.stats)
    with torch.cuda.stream(side):
        core.run(*inp)  # first launch on this device: run_checked
    torch.cuda.synchronize()
    assert len(_mod()._STREAM_CHECKED) == 1, "the first launch did not record a passed check"
    _assert_written(core, "checked")
    got = (core.partial.clone(), core.stats.clone())
    H.poison(core.partial, core.stats)
    with torch.cuda.stream(side):
        query, kv, selected, lengths, scales = inp
        q = query.view(torch.uint8).reshape(query.shape[0], -1)[:ROWS]
        core.module.run(q, kv.view(torch.uint8).view(-1), selected[:ROWS], lengths[:ROWS], core.partial, core.stats,
                        float(scales[0]) * _mod().LOG2E, core.tokens)
    torch.cuda.synchronize()
    _assert_written(core, "unchecked")
    assert torch.equal(got[0], core.partial) and torch.equal(got[1], core.stats), "checked launch differs from run()"


@needs_gpu
def test_other_stream_trips():
    core, inp = _core_and_inputs(1302)
    fold, other = torch.cuda.Stream(), torch.cuda.Stream()
    _reset()
    for attempt in range(2):  # fail closed every time: a refused check must not mark the key
        H.poison(core.partial, core.stats)
        with torch.cuda.stream(other), pytest.raises(Exception, match="K-13 stream check"):
            core.run(*inp, stream=fold.cuda_stream)
        _assert_poison(core, f"attempt {attempt}")
    assert not _mod()._STREAM_CHECKED, "a refused check marked its key"


@needs_gpu
def test_tvmffi_stream_trips():
    core, inp = _core_and_inputs(1303)
    core.module = _ForeignStream(core.module)
    side = torch.cuda.Stream()
    _reset()
    H.poison(core.partial, core.stats)
    with torch.cuda.stream(side), pytest.raises(Exception, match="K-13 stream check") as err:
        core.run(*inp)  # expected = torch's current stream (side); tvm-ffi resolves its own
    _assert_poison(core, "M1")
    text = str(err.value)
    launch = int(text.split("launch stream ")[1].split()[0])
    expected = int(text.split("dependent stream ")[1].split(";")[0])
    assert expected == side.cuda_stream and launch != expected, f"M1 did not diverge as built: {text}"


@needs_gpu
def test_capture_rechecks():
    core, inp = _core_and_inputs(1304)
    side, other = torch.cuda.Stream(), torch.cuda.Stream()
    _reset()
    H.poison(core.partial, core.stats)
    with torch.cuda.stream(side):
        core.run(*inp)  # eager: passes and marks this device
    torch.cuda.synchronize()
    _assert_written(core, "eager")
    ref = (core.partial.clone(), core.stats.clone())
    H.poison(core.partial, core.stats)
    bad = torch.cuda.CUDAGraph()
    with pytest.raises(Exception, match="K-13 stream check"):
        with torch.cuda.graph(bad):
            core.run(*inp, stream=other.cuda_stream)  # this capture's first launch names another dependent stream
    _assert_poison(core, "refused capture")
    good = torch.cuda.CUDAGraph()
    with torch.cuda.graph(good):
        core.run(*inp)  # checked again in this capture (dependent = the capture stream): passes, recorded
    H.poison(core.partial, core.stats)
    good.replay()
    torch.cuda.synchronize()
    _assert_written(core, "replay")
    assert torch.equal(ref[0], core.partial) and torch.equal(ref[1], core.stats), "replay differs from eager"


@needs_gpu
def test_layer_fails_closed():
    assert int(os.environ.get("K31_TEST_HEADS", "12")) == HEADS, "run with K31_TEST_HEADS=6 (TP16 shapes)"
    L = build_layer(8, 8704, 4313)
    a = L.arms["mla6"]
    a.core.mla6.module = _ForeignStream(a.core.mla6.module)
    _reset()
    H.poison(a.out, a.core.mla6.partial, a.core.mla6.stats)
    # A non-default stream, as serving runs: on the legacy default stream (handle 0) tvm-ffi's NULL env stream IS
    # torch's current stream, so M1 does not diverge there (MEASURED: k13_e36032f base, this test without it).
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    assert side.cuda_stream != 0
    with torch.cuda.stream(side):
        a.writer.run(L.hidden, L.slots)
        with pytest.raises(Exception, match="K-13 stream check"):
            a.core.run(L.projected, a.out, cache=a.kv, slots=L.slots, block_table=L.table, lengths=L.lengths, q=8)
    torch.cuda.synchronize()
    assert bool(torch.isnan(a.out.float()).all()), "layer output written: a dependent ran after the refused core"
    _assert_poison(a.core.mla6, "layer core")
    # control: the same layer without the forced divergence runs and writes every output cell
    a.core.mla6.module = a.core.mla6.module.module
    _reset()
    with torch.cuda.stream(side):
        a.core.run(L.projected, a.out, cache=a.kv, slots=L.slots, block_table=L.table, lengths=L.lengths, q=8)
    torch.cuda.synchronize()
    assert bool(torch.isfinite(a.out.float()).all()), "control layer run left unwritten output cells"
