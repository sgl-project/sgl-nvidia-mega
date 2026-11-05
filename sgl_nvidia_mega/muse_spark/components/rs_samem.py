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
on). Engagement is guarded by sha256 pins on the SGLang files whose call structure it relies on (one workspace lookup
before the pre-launch barrier; the call site imports ``matmul_reduce_scatter`` at call time; the model's layer and call
structure). Counters: ``skip``, ``keep``, ``keep:mismatch``.
"""

from __future__ import annotations

import functools
import logging
import os
import threading

from . import bed

# The SGLang names this module reads (test/unit/test_contract.py pins them against the supported SGLang)
SGLANG_READS = {
    "srt/models/muse_spark.py": ("MuseSpark1xModel",),
}


logger = logging.getLogger("muse_vr_plugin")

NAME = "rs_samem"
SGLANG_PINS = {
    # matmul_reduce_scatter: one workspace(...) lookup, then ws.handle.barrier(channel=0), then the launch
    "sglang/kernels/ops/communication/sp_collective_gemm.py": "f1cc99967104416a31fe7c99a710a9ccf8024ac1579c08579f11ac68a476a836",
    # row_parallel_gbar_matmul imports matmul_reduce_scatter from the module at call time
    "sglang/srt/layers/layernorm_sp.py": "bc814d983b945e5bdd41db436966eb4b3fee4c410bb1dcccfc98857d4b026bda",
    "sglang/srt/layers/communication/sp_collective_gemm_workspace.py": "099026321d22ee2956e7d1314bd4e9a26c14279aadf54cabc3c81f0748225a3c",
    # MuseSpark1xModel.forward: one call per model forward; every decoder layer is a sparse (MoE) layer whose only
    # reduce-scatter call is its o_proj, after its own qkv all-gather
    "sglang/srt/models/muse_spark.py": (
        "8bafbb8802fb0d9f81c190e87b9012fca546182be08bef6c8b3db814d27bfe19",
        "c0db42467a960c509b658c3c40570ee889f73d41b52cdbd80b9c291c1792a33f",
    ),
}

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


def _aid_dump() -> None:
    """Test aid ``MUSE_VR_RS_SAMEM_DUMP=1`` (never on a served configuration): rewrite the engagement record now (it is
    otherwise written only when a new counter key appears)."""
    if _state.get("dump"):
        with bed._lock:  # bed's own writer, under its lock (as bed.count calls it)
            bed._dump()


def new_forward() -> None:
    """A model forward starts: no fence carries over, the reduce-scatter index restarts. With the test aid the engagement
    record is rewritten when a forward follows one that issued reduce-scatter calls, and after every reduce-scatter call (a
    forward replayed from a CUDA graph does not run this hook), so the last record holds every call issued so far."""
    with _lock:
        had_calls = _state["rs_idx"] > 0
        _state["epoch"] += 1
        _state["rs_idx"] = 0
        _state["fence"] = None
    if had_calls:
        _aid_dump()


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
            _aid_dump()
            return orig(x, weight, group, workspace=workspace)
        calls = []

        def ws_no_barrier(*args, **kwargs):
            calls.append(1)
            return _NoBarrierWorkspace(workspace(*args, **kwargs))

        out = orig(x, weight, group, workspace=ws_no_barrier)
        if (
            len(calls) != 1
        ):  # the pinned call structure looks the workspace up exactly once
            raise RuntimeError(
                f"{NAME}: workspace looked up {len(calls)} times; pinned call structure violated"
            )
        bed.count(NAME, "skip")
        _aid_dump()
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
    for rel, sha in SGLANG_PINS.items():
        if not bed.pinned(rel, sha):
            bed.report(NAME, False, reason=f"pin mismatch: {rel}")
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
            _state["dump"] = os.environ.get("MUSE_VR_RS_SAMEM_DUMP", "").strip() == "1"
    except Exception as e:  # noqa: BLE001
        bed.report(NAME, False, reason=f"install failed: {type(e).__name__}: {e}")
        return False
    _state["engaged"] = True
    bed.report(NAME, True, dump=_state["dump"])
    return True
