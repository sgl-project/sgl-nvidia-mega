# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Shared helpers for the Kimi K3.1 megakernel GPU tests on VR200 (SM107) and GB300.

- ``load_plugin()``: imports ``sgl_nvidia_mega``; with ``K31_MUTATION=<name>`` set it
  first copies the package to a temp dir and applies the named source mutation from the
  calling test module's ``MUTATIONS`` table, so a test can be shown to fail on a
  deliberate break of the code under test.
- ``check()``: one numeric gate (stated atol/rtol, no NaN/Inf, all cells checked).
- ``poison()``: fill outputs with NaN bytes before a kernel call.
- ``cold_time_us()``: per-call device time over a CUDA graph that cycles through
  enough distinct weight copies to defeat the L2 (weights stream from HBM, as
  in a 93-layer decode step); the process runs an FP8 MMA kernel first (``hold_fp8_clock``).
- ``record()``: appends one JSON line per measured or checked item to ``$K31_RESULTS``.
"""

from __future__ import annotations

import json
import math
import os
import shutil
import sys
import tempfile
import time
from pathlib import Path

import torch

PLUGIN_ROOT = Path(__file__).resolve().parents[3]
# Bandwidth floors of the timing helpers: test parameters (bytes / s), NaN floors when unset.
HBM_BPS = float(os.environ.get("K31_TEST_HBM_BPS", "nan"))
HBM_LOCAL_BPS = float(os.environ.get("K31_TEST_HBM_LOCAL_BPS", "nan"))
L2_BYTES = torch.cuda.get_device_properties(0).L2_cache_size if torch.cuda.is_available() else 0  # no CUDA: no timing

_loaded = None


def load_plugin(mutations: dict | None = None):
    """Import sgl_nvidia_mega (mutated copy when K31_MUTATION names an entry)."""
    global _loaded
    if _loaded is not None:
        return _loaded
    name = os.environ.get("K31_MUTATION", "")
    root = PLUGIN_ROOT
    if name:
        if not mutations or name not in mutations:
            raise KeyError(f"unknown mutation {name!r} for this test module")
        tmp = Path(tempfile.mkdtemp(prefix="k31mut_"))
        shutil.copytree(PLUGIN_ROOT / "sgl_nvidia_mega", tmp / "sgl_nvidia_mega")
        edits = mutations[name]
        edits = [edits] if isinstance(edits[0], str) else edits
        for rel, old, new in edits:
            path = tmp / "sgl_nvidia_mega" / rel
            text = path.read_text()
            count = text.count(old)
            if count != 1:
                raise RuntimeError(f"mutation {name}: {old!r} occurs {count}x in {rel}")
            path.write_text(text.replace(old, new))
        root = tmp
        # The variant's JIT builds go to a private cache, so no build of the shipped source can be reused.
        os.environ["SGLANG_JIT_CACHE_DIR"] = str(tmp / "jit_cache")
        print(f"K31_MUTATION {name} applied in {tmp}", flush=True)
    sys.path.insert(0, str(root))
    for key in [k for k in sys.modules if k.startswith("sgl_nvidia_mega")]:
        del sys.modules[key]
    import sgl_nvidia_mega

    got = Path(sgl_nvidia_mega.__file__).resolve().parents[1]
    if got != root.resolve():
        raise RuntimeError(f"imported sgl_nvidia_mega from {got}, expected {root}")
    _loaded = sgl_nvidia_mega
    return sgl_nvidia_mega


def device_facts() -> dict:
    p = torch.cuda.get_device_properties(torch.cuda.current_device())
    return {
        "name": p.name,
        "cc": f"{p.major}.{p.minor}",
        "sms": p.multi_processor_count,
        "smem_per_block_optin": getattr(p, "shared_memory_per_block_optin", None),
        "smem_per_sm": getattr(p, "shared_memory_per_multiprocessor", None),
        "max_threads_per_sm": getattr(p, "max_threads_per_multi_processor", None),
        "l2": getattr(p, "L2_cache_size", None),
    }


def generator(seed: int) -> torch.Generator:
    g = torch.Generator(device="cuda")
    g.manual_seed(seed)
    return g


def randn(shape, scale=1.0, g=None, dtype=torch.bfloat16):
    return (torch.randn(shape, generator=g, device="cuda", dtype=torch.float32) * scale).to(dtype)


def bf16_head_gate_q_scale(gate, q_scale, softmax_scale):
    """Reference DSA indexer head weights [tokens, heads, 1] FP32: float(bf16(gate * heads**-0.5)) * q_scale * softmax_scale,
    the head-gate arithmetic the host's indexer_q_quant_head_gate documents (gate [tokens, heads], any row stride; q_scale
    one FP32 value per (token, head) in row-major order)."""
    tokens, heads = gate.shape
    g = (gate.float() * heads**-0.5).to(torch.bfloat16).float()
    qs = q_scale.contiguous().reshape(-1)[: tokens * heads].reshape(tokens, heads).float()
    return (g * qs * softmax_scale).unsqueeze(-1)


def bundle_grid_sms() -> int:
    """The SM count the TP16 bundle derives its experts grids for (``K31MK_EXPERTS_GRID_SMS`` of the served decode env):
    this device's SM count less the SMs the bundle leaves to the shared expert."""
    return torch.cuda.get_device_properties(0).multi_processor_count - 16


def poison(*tensors):
    """NaN in every float lane (0xFF bytes); integer outputs become -1."""
    for t in tensors:
        t.view(torch.uint8).fill_(0xFF)


def check(name, out, ref, atol, rtol, *, frac=0.0):
    """|out - ref| <= atol + rtol * |ref| on every cell (``frac``: allowed fraction
    of violating cells, stated up front by the caller). Raises AssertionError."""
    o, r = out.float(), ref.float()
    assert o.shape == r.shape, (name, tuple(o.shape), tuple(r.shape))
    finite = torch.isfinite(o)
    if not bool(finite.all()):
        bad = int((~finite).sum())
        raise AssertionError(f"{name}: {bad}/{o.numel()} non-finite outputs")
    if not bool(torch.isfinite(r).all()):
        raise AssertionError(f"{name}: reference has non-finite values")
    err = (o - r).abs()
    bound = atol + rtol * r.abs()
    viol = err > bound
    nviol = int(viol.sum())
    stats = {
        "item": name,
        "cells": o.numel(),
        "max_abs": float(err.max()) if err.numel() else 0.0,
        "mean_abs": float(err.mean()) if err.numel() else 0.0,
        "ref_rms": float(r.pow(2).mean().sqrt()) if r.numel() else 0.0,
        "violations": nviol,
        "atol": atol,
        "rtol": rtol,
    }
    print("CHECK", json.dumps(stats), flush=True)
    record({"kind": "check", **stats})
    digest(name, out)
    if nviol > frac * o.numel():
        raise AssertionError(f"{name}: {nviol}/{o.numel()} cells beyond atol={atol} rtol={rtol}; {stats}")
    return stats


def digest(name, tensor):
    """With ``K31_DIGEST=1``, record the SHA-256 of ``tensor``'s bytes (bitwise comparisons
    across runs, e.g. CUDA_MODULE_LOADING=EAGER vs LAZY + K31MK_PRELOAD=1)."""
    if os.environ.get("K31_DIGEST") != "1":
        return
    import hashlib

    data = tensor.detach().contiguous().view(torch.uint8).cpu().numpy().tobytes()
    record({"kind": "digest", "item": name, "sha256": hashlib.sha256(data).hexdigest(), "bytes": len(data)})


def loading_mode() -> str:
    """The driver's module loading mode for this process (CUDA initialized first)."""
    import cuda.bindings.driver as cu

    torch.cuda.init()
    status, mode = cu.cuModuleGetLoadingMode()
    return "LAZY" if mode == cu.CUmoduleLoadingMode.CU_MODULE_LAZY_LOADING else "EAGER"


def maybe_preload(where):
    """The MK prepare hook's preload (``K31MK_PRELOAD=1``), run where a test's kernels are built."""
    from sgl_nvidia_mega.kimi_k31.k3 import preload

    if not preload.ENABLED:
        return None
    stats = preload.preload()
    record({"kind": "preload", "where": where, "loading": loading_mode(), **stats})
    print("PRELOAD", where, stats, flush=True)
    return stats


def preload_audit(where):
    """CuTe kernels not loaded at ``where`` (nothing is loaded by the audit)."""
    from sgl_nvidia_mega.kimi_k31.k3 import preload

    stats = preload.audit()
    record({"kind": "preload_audit", "where": where, "loading": loading_mode(), **stats})
    print("PRELOAD_AUDIT", where, stats, flush=True)
    return stats


def record(row: dict):
    path = os.environ.get("K31_RESULTS")
    row = {
        "t": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "mutation": os.environ.get("K31_MUTATION", ""),
        "gpu": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
        "test": os.environ.get("PYTEST_CURRENT_TEST", "").split(" ")[0],
        **row,
    }
    if path:
        with open(path, "a") as f:
            f.write(json.dumps(row) + "\n")


_fp8_held = False


def hold_fp8_clock():
    """Run an FP8 MMA kernel once before timing, so timed runs match serving conditions
    (serving runs FP8 / FP4 MMAs)."""
    global _fp8_held
    if _fp8_held:
        return
    a = torch.randn(256, 256, device="cuda").to(torch.float8_e4m3fn)
    one = torch.ones((), device="cuda")
    torch._scaled_mm(a, a.t(), scale_a=one, scale_b=one, out_dtype=torch.bfloat16)
    torch.cuda.synchronize()
    _fp8_held = True


def copies_for(bytes_per_call: int, minimum: int = 2, cap: int = 64) -> int:
    """Distinct weight copies so a cycle streams >= 3x L2 (no L2 reuse across calls)."""
    if bytes_per_call <= 0:
        return minimum
    return max(minimum, min(cap, math.ceil(3 * L2_BYTES / bytes_per_call)))


def cold_time_us(calls, *, reps: int = 20, warmup: int = 3) -> dict:
    """``calls``: list of zero-argument callables (one per weight copy). Captures one
    CUDA graph running them in order, replays it ``reps`` times, returns per-call us
    (median over replays)."""
    hold_fp8_clock()
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(warmup):
            for c in calls:
                c()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        for c in calls:
            c()
    torch.cuda.synchronize()
    for _ in range(warmup):
        graph.replay()
    torch.cuda.synchronize()
    times = []
    for _ in range(reps):
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record()
        graph.replay()
        end.record()
        end.synchronize()
        times.append(start.elapsed_time(end) * 1e3 / len(calls))
    times.sort()
    return {
        "us_median": times[len(times) // 2],
        "us_min": times[0],
        "us_max": times[-1],
        "calls_per_graph": len(calls),
        "reps": reps,
    }


def floor_us(nbytes: int) -> dict:
    return {
        "bytes": int(nbytes),
        "floor_us": nbytes / HBM_BPS * 1e6,
        "floor_local_us": nbytes / HBM_LOCAL_BPS * 1e6,
    }


def function_resources(compiled, device_index=None) -> dict:
    """Registers, static/dynamic SMEM, local (spill) bytes and the binary's SM version
    of a CuTe-compiled one-kernel module (``common.host.resolve_kernel``)."""
    import cuda.bindings.driver as cuda

    plugin = load_plugin()
    from sgl_nvidia_mega.kimi_k31.k3.cutedsl.common.host import resolve_kernel

    device_index = torch.cuda.current_device() if device_index is None else device_index
    _, function = resolve_kernel(compiled, device_index)
    attr = cuda.CUfunction_attribute
    out = {}
    for key, a in (
        ("regs", attr.CU_FUNC_ATTRIBUTE_NUM_REGS),
        ("static_smem", attr.CU_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES),
        ("max_dynamic_smem", attr.CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES),
        ("local_bytes", attr.CU_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES),
        ("max_threads", attr.CU_FUNC_ATTRIBUTE_MAX_THREADS_PER_BLOCK),
        ("binary_version", attr.CU_FUNC_ATTRIBUTE_BINARY_VERSION),
        ("ptx_version", attr.CU_FUNC_ATTRIBUTE_PTX_VERSION),
    ):
        status, value = cuda.cuFuncGetAttribute(a, function)
        out[key] = int(value) if status == cuda.CUresult.CUDA_SUCCESS else f"err {status}"
    del plugin
    return out


def is_sm107() -> bool:
    return torch.cuda.get_device_capability() == (10, 7)
