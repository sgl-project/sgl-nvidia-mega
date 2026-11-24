# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Component ``rs_samem``: leave out the GEMM + reduce-scatter's pre-launch barrier when another full barrier already
separates the call from the previous GEMM + reduce-scatter call.

Under ``--enable-layernorm-sp`` SGLang's ``matmul_reduce_scatter`` (one CUTLASS GEMM with a multicast reduce-scatter
epilogue, ``sglang.kernels.ops.communication.sp_collective_gemm``) issues a symmetric-memory barrier before every launch,
so that no rank starts writing the shared workspace or its flags while a peer is still finishing the previous call on
it. On a sequence-parallel prefill pass that runs ``fp8_prefill``'s MXFP8 qkv all-gather, every layer's all-gather issues
a barrier of the same kind ("every rank's shard is staged") between two consecutive reduce-scatter calls: each rank
arrives at it only after its previous reduce-scatter kernel has completed (stream order), and each rank starts its next
reduce-scatter only after it has seen every rank arrive. That is the ordering the pre-launch barrier provides.

This component records each such all-gather barrier with its forward and decoder layer and, for the reduce-scatter call
of the same layer in the same forward (the layer's o_proj: the i-th reduce-scatter call of a forward belongs to layer i),
runs SGLang's call with the pre-launch barrier left out (the workspace handle's ``barrier`` is a no-op for that one
call). Any other call keeps the stock barrier (counted ``keep``; a recorded fence that does not match the call's
forward and layer is counted ``keep:mismatch`` and logged as an error). The decision is a function of the forward's call
sequence only, so every rank makes the same decision for every call. It is made on the host when the call is issued, so
a captured CUDA graph keeps the same sequence of barriers and calls on replay.

Ordering basis: torch's symmetric-memory barrier arrives with a release at system scope and waits with acquire loads at
system scope (PTX ISA, memory consistency model: release and acquire patterns); kernels in one stream run in issue order
(CUDA Programming Guide, streams).

Numerics: unchanged. The same kernel runs on the same inputs; only a synchronization step is left out.

Opt in with ``MUSE_VR_RS_SAMEM=1`` (needs ``fp8_prefill`` engaged: its all-gather is the barrier this component relies
on). ``SGLANG_MIRRORS`` holds the digests of the SGLang functions whose call structure it relies on (one workspace lookup
before the pre-launch barrier; the call site imports ``matmul_reduce_scatter`` at call time; the model's layer and call
structure), checked by ``test/unit/test_contract.py``; a call that looks the workspace up more than once raises.
Counters: ``skip``, ``keep``, ``keep:mismatch``.
"""

from __future__ import annotations

import functools
import logging
import threading

from sgl_nvidia_mega.muse_spark.components import bed

# The SGLang names this module reads (test/unit/test_contract.py pins them against the supported SGLang)
SGLANG_READS = {
    "srt/models/muse_spark.py": ("MuseSpark1xModel", "forward"),
    "kernels/ops/communication/sp_collective_gemm.py": (
        "matmul_reduce_scatter",
        "matmul_reduce_scatter(workspace)",
    ),
}

# The SGLang functions whose code this module relies on, by the digest of their code (test/unit/test_contract.py
# recomputes them on the installed SGLang). One digest: the same in SGLang cf6133a and 306ccf6.
SGLANG_MIRRORS = {
    # one workspace(...) lookup, then ws.handle.barrier(channel=0), then the launch
    "kernels/ops/communication/sp_collective_gemm.py:matmul_reduce_scatter": "eee82831fa289a66",
    # imports matmul_reduce_scatter from its module at call time (so the wrapped function is the one called)
    "srt/layers/layernorm_sp.py:row_parallel_gbar_matmul": "8bfbed80df1b9e08",
    # one call per model forward; every decoder layer's only reduce-scatter call is its o_proj, after its qkv all-gather
    "srt/models/muse_spark.py:MuseSpark1xModel.forward": "82b3cba268aa1fdc",
}


logger = logging.getLogger("muse_vr_plugin")

NAME = "rs_samem"

_lock = threading.Lock()
# "fence": (forward epoch, layer) of the all-gather barrier issued since the last reduce-scatter call, else None;
# "epoch": forwards seen; "rs_idx": reduce-scatter calls so far in this forward (one per decoder layer: its o_proj)
_state: dict = {
    "engaged": None,
    "orig_rs": None,
    "orig_ag": None,
    "orig_fwd": None,
    "fence": None,
    "epoch": 0,
    "rs_idx": 0,
    "logged": set(),
}


class _NoBarrierHandle:
    """The workspace's symmetric-memory handle with ``barrier`` as a no-op (every other attribute forwarded)."""

    def __init__(self, handle):
        self._h = handle

    def barrier(self, *args, **kwargs):
        return None

    def __getattr__(self, name):
        return getattr(self._h, name)


class _NoBarrierWorkspace:
    """A workspace whose ``handle.barrier`` is a no-op (every other attribute forwarded)."""

    def __init__(self, ws):
        self._ws = ws
        self.handle = _NoBarrierHandle(ws.handle)

    def __getattr__(self, name):
        return getattr(self._ws, name)


def new_forward() -> None:
    """A model forward starts: no fence carries over, the reduce-scatter index restarts."""
    with _lock:
        _state["epoch"] += 1
        _state["rs_idx"] = 0
        _state["fence"] = None


def note_fence(layer: int) -> None:
    """Record that decoder ``layer``'s all-gather just issued a full barrier on the reduce-scatter's group, on the current
    stream."""
    with _lock:
        _state["fence"] = (_state["epoch"], int(layer))


def _take_fence() -> tuple:
    """(skip, reason) for the reduce-scatter call being issued; consumes the fence. Skips only when the fence was recorded in
    this forward by the all-gather of the same decoder layer as this call (the layer's o_proj is its i-th reduce-scatter
    call), so every rank, running the same forward, makes the same decision for every call."""
    with _lock:
        f, idx = _state["fence"], _state["rs_idx"]
        _state["fence"] = None
        _state["rs_idx"] = idx + 1
        if f is None:
            return False, "keep"
        if f != (_state["epoch"], idx):
            return False, "keep:mismatch"
        return True, "skip"


def _wrap_rs(orig):
    @functools.wraps(orig)
    def matmul_reduce_scatter(x, weight, group, *, workspace):
        skip, why = _take_fence()
        if not skip:
            bed.count(NAME, why)
            if why != "keep" and why not in _state["logged"]:
                _state["logged"].add(why)
                logger.error(
                    "muse_vr_plugin: %s: fence / call mismatch (forward %d, call %d); stock barrier kept",
                    NAME,
                    _state["epoch"],
                    _state["rs_idx"] - 1,
                )
            return orig(x, weight, group, workspace=workspace)
        calls = []

        def ws_no_barrier(*args, **kwargs):
            calls.append(1)
            return _NoBarrierWorkspace(workspace(*args, **kwargs))

        out = orig(x, weight, group, workspace=ws_no_barrier)
        if (
            len(calls) != 1
        ):  # the call structure in SGLANG_MIRRORS looks the workspace up exactly once
            raise RuntimeError(
                f"{NAME}: workspace looked up {len(calls)} times; the call structure in SGLANG_MIRRORS changed"
            )
        bed.count(NAME, "skip")
        return out

    matmul_reduce_scatter.__wrapped_by_rs_samem__ = True
    return matmul_reduce_scatter


def _wrap_ag(orig):
    @functools.wraps(orig)
    def qkv_ag_gemm(self, x, layer, *args, **kwargs):
        out = orig(self, x, layer, *args, **kwargs)
        note_fence(
            layer
        )  # it returned: its "every rank's shard is staged" barrier was issued on this stream
        return out

    qkv_ag_gemm.__wrapped_by_rs_samem__ = True
    return qkv_ag_gemm


def _wrap_forward(orig):
    @functools.wraps(orig)
    def forward(*args, **kwargs):
        new_forward()
        return orig(*args, **kwargs)

    forward.__wrapped_by_rs_samem__ = True
    return forward


def register() -> bool:
    if not bed.enabled(NAME):
        return False
    if not bed.enabled("fp8_prefill"):
        bed.report(
            NAME,
            False,
            reason="needs fp8_prefill (its all-gather barrier is the fence this relies on)",
        )
        return False
    try:
        import sglang.kernels.ops.communication.sp_collective_gemm as scg
        import sglang.srt.models.muse_spark as muse_spark

        from . import fp8_prefill

        with _lock:
            if not getattr(scg.matmul_reduce_scatter, "__wrapped_by_rs_samem__", False):
                _state["orig_rs"] = scg.matmul_reduce_scatter
                scg.matmul_reduce_scatter = _wrap_rs(scg.matmul_reduce_scatter)
            cls = fp8_prefill.Fp8Ctx
            if not getattr(cls.qkv_ag_gemm, "__wrapped_by_rs_samem__", False):
                _state["orig_ag"] = cls.qkv_ag_gemm
                cls.qkv_ag_gemm = _wrap_ag(cls.qkv_ag_gemm)
            mcls = muse_spark.MuseSpark1xModel
            if not getattr(mcls.forward, "__wrapped_by_rs_samem__", False):
                _state["orig_fwd"] = mcls.forward
                mcls.forward = _wrap_forward(mcls.forward)
            _state["fence"] = None
    except Exception as e:  # noqa: BLE001
        bed.report(NAME, False, reason=f"install failed: {type(e).__name__}: {e}")
        return False
    _state["engaged"] = True
    bed.report(NAME, True)
    return True
