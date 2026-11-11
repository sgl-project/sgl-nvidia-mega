# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""The benchmark-only switches of ``runtime/bench.py`` on one SM100-family GPU.

- Fixed routing: a synthesized table has the requested distinct experts and split
  between the expert-parallel halves at 6 rows; the logits built from it make the real
  gate top-k kernel (target 384 / top-6 and draft 128 / top-3) choose exactly the
  table's experts, for any correction bias; refused without simulated acceptance.
- The step stamp: captured in a CUDA graph on a forked stream, every replay writes one
  event (increasing time, the position, the slot) and, per noted id tensor, the
  previous replay's distinct-expert count and its first-half count.
"""

import ctypes

import numpy as np
import pytest
import torch

from sgl_nvidia_mega.deepseek_v41.runtime import bench
from sgl_nvidia_mega.deepseek_v41.sm103.moe import routed_expert_gemm as reg


def _ready() -> bool:
    if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 10:
        return False
    try:
        from cuda.bindings import nvrtc  # noqa: F401
    except ImportError:
        return False
    return True


needs_gpu = pytest.mark.skipif(not _ready(), reason="needs an SM100-family GPU")


@pytest.mark.parametrize("d,half0", [(25, 13), (24, 12), (12, 0), (36, 30), (6, 3)])
def test_synth_counts(d, half0):
    spec = bench._parse_fixed(f"synth:d={d},half0={half0},seed=3")
    for layer in range(5):
        ids = bench._synth_ids(spec, "target", layer, 384, 6)
        six = ids[:6]
        assert all(len(set(r)) == 6 for r in ids)  # distinct within each row
        u = np.unique(six)
        assert len(u) == d
        assert int((u < 192).sum()) == half0


def test_refused_without_simulated_acceptance(monkeypatch):
    monkeypatch.setenv(bench.FIXED_ENV, "synth")
    monkeypatch.delenv("SGLANG_SIMULATE_ACC_LEN", raising=False)
    bench.fixed_spec.cache_clear()
    with pytest.raises(RuntimeError, match="benchmark only"):
        bench.fixed_spec()
    monkeypatch.setenv("SGLANG_SIMULATE_ACC_LEN", "3.51")
    bench.fixed_spec.cache_clear()
    assert bench.fixed_spec()["mode"] == "synth"
    bench.fixed_spec.cache_clear()


def _gate(logits, bias, experts, top_k):
    rows = logits.size(0)
    w = torch.empty((rows, top_k), dtype=torch.float32, device="cuda")
    ids = torch.empty((rows, top_k), dtype=torch.int32, device="cuda")
    reg._launch(
        reg.load_gate_topk(0, experts, top_k),
        (rows, 1, 1),
        [
            reg._pointer(logits),
            reg._pointer(bias),
            ctypes.c_void_p(0),
            ctypes.c_void_p(0),
            ctypes.c_int32(0),
            ctypes.c_int64(-1),
            ctypes.c_void_p(0),
            reg._pointer(w),
            reg._pointer(ids),
            ctypes.c_float(1.5),
            ctypes.c_float(0.0),
        ],
        torch.cuda.current_stream().cuda_stream,
    )
    return w, ids


@needs_gpu
@pytest.mark.parametrize("kind,experts,top_k", [("target", 384, 6), ("draft", 128, 3)])
@pytest.mark.parametrize("bias_scale", [0.0, 0.5, 8.0])
def test_fixed_logits_select_table(kind, experts, top_k, bias_scale):
    spec = bench._parse_fixed("synth:d=20,half0=9,draft_d=9,draft_half0=4,seed=1")
    gen = torch.Generator().manual_seed(7)
    bias = (torch.randn(experts, generator=gen) * bias_scale).cuda()
    for layer in range(3):
        ids = bench._synth_ids(spec, kind, layer, experts, top_k)
        fixed = bench.FixedLayer(kind, layer, ids, bias, 1.5, torch.device("cuda"))
        for rows in (1, 6, 48):
            real = torch.randn(rows, experts, device="cuda")
            w, got = _gate(fixed.router_logits(real), bias, experts, top_k)
            want = torch.from_numpy(ids[:rows]).cuda().to(torch.int32)
            assert torch.equal(got.sort(-1).values, want.sort(-1).values)
            assert torch.allclose(w.sum(-1), torch.full((rows,), 1.5, device="cuda"))
        t_ids, t_w = fixed.topk(
            torch.zeros(6, top_k, dtype=torch.int64, device="cuda"),
            torch.zeros(6, top_k, device="cuda"),
        )
        assert t_ids.dtype == torch.int64 and torch.equal(
            t_ids.cpu(), torch.from_numpy(ids[:6])
        )


@needs_gpu
def test_stamp_in_graph(tmp_path):
    timer = bench.StepTimer(str(tmp_path), torch.device("cuda", 0))
    positions = torch.tensor([8192, 8193, 8194, 8195, 8196, 8197], device="cuda")
    ids_a = torch.zeros(6, 6, dtype=torch.int32, device="cuda")
    ids_b = torch.zeros(6, 3, dtype=torch.int64, device="cuda")
    graph = torch.cuda.CUDAGraph()
    work = torch.zeros(1 << 20, device="cuda")
    s = torch.cuda.Stream()
    with torch.cuda.stream(s):
        with torch.cuda.graph(graph, stream=s):
            with timer.forward("verify", positions):
                work.mul_(1.0001)
                timer.note(ids_a, 0, 384)
                timer.note(ids_b, 1, 128)
    choices = [
        (np.arange(36).reshape(6, 6) % 10, np.arange(18).reshape(6, 3) % 4 + 70),
        (np.arange(36).reshape(6, 6) % 25 + 180, np.arange(18).reshape(6, 3) % 9),
    ]
    for step in range(6):
        a, b = choices[step % 2]
        ids_a.copy_(torch.from_numpy(a))
        ids_b.copy_(torch.from_numpy(b))
        positions.add_(4)
        graph.replay()
        torch.cuda.synchronize()
    import time

    time.sleep(1.0)  # the drain thread
    ev = np.fromfile(tmp_path / "events-r0.bin", dtype=np.int64).reshape(-1, 4)
    assert len(ev) == 6 and list(ev[:, 3]) == list(range(1, 7))
    assert (np.diff(ev[:, 0]) > 0).all()
    assert list(ev[:, 2]) == [8196 + 4 * i for i in range(6)]
    raw = np.fromfile(tmp_path / "counts-r0.bin", dtype=np.int16).reshape(
        -1, 4 + 2 * bench.MAX_ENTRIES
    )
    counts = raw[:, 4:]
    # replay n reads the ids replay n - 1 left (the copies run before the replay here,
    # so the ids are this step's)
    for n in range(6):
        a, b = choices[n % 2]
        assert (
            counts[n, 0] == len(np.unique(a))
            and counts[n, 1] == (np.unique(a) < 192).sum()
        )
        assert (
            counts[n, 2] == len(np.unique(b))
            and counts[n, 3] == (np.unique(b) < 64).sum()
        )
