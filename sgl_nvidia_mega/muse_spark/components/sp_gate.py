# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Component ``sp_gate``: run LayerNorm sequence parallelism only for prefill passes above a token threshold.

With ``--enable-layernorm-sp`` SGLang runs every EXTEND forward of Muse Spark sequence-parallel (``layernorm_sp.runs_sp``
is true for any extend): the token rows are scattered over the TP ranks, the attention projections become
all-gather + GEMM and GEMM + reduce-scatter, and the routed experts exchange tokens with an all-to-all. Those
collectives have a fixed cost per layer, so small extends (short follow-up turns, small chunks, prefix-cache hits) run
slower sequence-parallel than on the replicated path that decode and the base recipe use, while large chunks run
faster.

This component keeps SGLang's own two code paths and only chooses between them per pass: a forward runs
sequence-parallel when it is an extend AND its token count is above ``MUSE_VR_SP_GATE_MIN_TOKENS``; otherwise it
runs the replicated path exactly as a server without the flag would. The decision is a pure function of the pass's
token count, so it is identical on every TP rank. Prefill CUDA graphs are captured per token bucket and replayed
with the batch padded up to its bucket; the threshold must be one of the captured bucket sizes, so that a pass and
the bucket it replays always fall on the same side of the threshold (a pass with ``n`` tokens replays bucket
``b >= n``; with the threshold ``t`` a captured size, ``n > t`` exactly when ``b > t``). The component checks this
against the server's prefill capture list on first use and stays off (stock behaviour) if it does not hold.

Every pass computes exactly what one of SGLang's two paths computes for it (sequence-parallel or replicated), so the
numerics of a pass are those of the path it takes.

Opt in with ``MUSE_VR_SP_GATE=1`` (only meaningful with ``--enable-layernorm-sp``). ``MUSE_VR_SP_GATE_MIN_TOKENS``
(default 1280): passes with more tokens than this run sequence-parallel. The best value is the pass size at which the
two paths cost the same, which depends on the deployment's collectives (for example the all-reduce sizes the
replicated path's custom all-reduce covers): measure extend passes of a few sizes on both paths and pick the captured
bucket nearest the crossover. Engagement is guarded by sha256 pins on the
SGLang files whose SP entry/exit logic and prefill-graph replay contract the gate relies on.
"""

from __future__ import annotations

import bisect
import functools
import logging
import os
import threading

from . import bed

# The SGLang names this module reads (test/unit/test_contract.py pins them against the supported SGLang)
SGLANG_READS = {
    "srt/models/muse_spark.py": (
        "MuseSpark1xForCausalLM|MuseSparkTextForCausalLM",
        "MuseSpark1xModel",
    ),
    "srt/model_executor/cuda_graph_config.py": ("Backend",),
    "srt/runtime_context.py": ("get_exec",),
}


logger = logging.getLogger("muse_vr_plugin")

NAME = "sp_gate"
DEFAULT_MIN_TOKENS = 1280

# sha256 of files relative to the directory that holds the ``sglang`` package.
SGLANG_PINS = {
    # runs_sp / maybe_exit_gather / entry scatter + exit gather
    "sglang/srt/layers/layernorm_sp.py": (
        "ffdde97e6282103614d7cec20cc29fb3cf0157d051778e315b03a99bb8e81090",
        "bc814d983b945e5bdd41db436966eb4b3fee4c410bb1dcccfc98857d4b026bda",
    ),
    # decoder SP entry (MuseSpark1xModel._forward), aux-tap gather and omni exit (MuseSpark1xForCausalLM._forward)
    "sglang/srt/models/muse_spark.py": (
        "7e5018b3cb2ef41ee6cf09407e91e28dde069e5a33719ea58beb0d7137087355",
        "8bafbb8802fb0d9f81c190e87b9012fca546182be08bef6c8b3db814d27bfe19",
        "c0db42467a960c509b658c3c40570ee889f73d41b52cdbd80b9c291c1792a33f",
    ),
    # LM-head exit gather (maybe_exit_gather with the forward mode)
    "sglang/srt/layers/logits_processor.py": (
        "ecc0318937eee486414f6ae7165f2d1de17c01794a25aae83a99af61cbb0e3d0",
        "3558330bccf3ea0e46aabdaa0dee6ae7daf294c8e15b9a4e9744dab6883c4398",
    ),
    # layer-boundary SP entry (sp_active from runs_sp)
    "sglang/srt/layers/layer_boundary/stage.py": (
        "c496ad878963c724c0a46db4489798a139c34e5c0b22a273e4af71f3ae45e276",
        "782bbc56750c38b8e9cd1fd6adbb315e33289a1fa135a0720b1c830c274aec7f",
    ),
    # participant linears read sp_active
    "sglang/srt/layers/linear.py": "3ce31df0a8aab09ed8afa0ada8c0ce30ca207a748c02c3073d7f53648b9b7c3c",
}

_lock = threading.Lock()
_state: dict = {
    "engaged": None,
    "min_tokens": DEFAULT_MIN_TOKENS,
    "orig_runs_sp": None,
    "checked": None,  # None: capture list not checked yet; True/False: gate usable
    "tokens": None,  # token count of the forward in flight (set at the model entry points)
}


def decide(num_tokens: int | None, min_tokens: int) -> bool:
    """Whether a pass with ``num_tokens`` tokens may run sequence-parallel."""
    return num_tokens is None or num_tokens > min_tokens


def threshold_ok(min_tokens: int, capture_sizes) -> bool:
    """The threshold must be a captured prefill bucket (or at/above the largest one, or graphs off)."""
    sizes = sorted(int(s) for s in (capture_sizes or ()))
    if not sizes:
        return True
    if min_tokens >= sizes[-1]:
        return True
    i = bisect.bisect_left(sizes, min_tokens)
    return i < len(sizes) and sizes[i] == min_tokens


def _prefill_capture_sizes():
    """The server's prefill CUDA-graph token buckets ([] when prefill graphs are disabled)."""
    from sglang.srt.model_executor.cuda_graph_config import Backend
    from sglang.srt.runtime_context import get_exec

    cfg = get_exec().graph.cuda_graph_config.prefill
    if cfg.backend == Backend.DISABLED:
        return []
    return list(cfg.bs or [])


def _gate_usable() -> bool:
    ok = _state["checked"]
    if ok is not None:
        return ok
    with _lock:
        if _state["checked"] is None:
            try:
                sizes = _prefill_capture_sizes()
                ok = threshold_ok(_state["min_tokens"], sizes)
                if not ok:
                    logger.error(
                        "muse_vr_plugin: sp_gate threshold %d is not a prefill CUDA-graph bucket; gate OFF "
                        "(stock sequence parallelism for every extend)",
                        _state["min_tokens"],
                    )
                bed.report(
                    NAME + "_check",
                    ok,
                    min_tokens=_state["min_tokens"],
                    buckets=len(sizes),
                )
            except Exception as e:  # noqa: BLE001 - unknown capture config: keep stock behaviour
                logger.error(
                    "muse_vr_plugin: sp_gate cannot read the prefill capture list (%s); gate OFF",
                    e,
                )
                ok = False
            _state["checked"] = ok
    return _state["checked"]


def _runs_sp(forward_mode) -> bool:
    if not _state["orig_runs_sp"](forward_mode):
        return False
    if not _gate_usable():
        return True
    sp = decide(_state["tokens"], _state["min_tokens"])
    bed.count(NAME, "sp" if sp else "replicated")
    return sp


def _num_tokens(input_ids, input_embeds) -> int | None:
    if input_ids is not None:
        return int(input_ids.shape[0])
    if input_embeds is not None:
        return int(input_embeds.shape[0])
    return None


def _wrap_entry(fn):
    @functools.wraps(fn)
    def forward(
        self, input_ids, positions, forward_batch, input_embeds=None, *args, **kwargs
    ):
        _state["tokens"] = _num_tokens(input_ids, input_embeds)
        return fn(
            self, input_ids, positions, forward_batch, input_embeds, *args, **kwargs
        )

    forward.__wrapped_by_sp_gate__ = True
    return forward


def install(min_tokens: int) -> None:
    """Patch ``layernorm_sp.runs_sp`` and the two Muse Spark forward entry points (also used by the tests)."""
    import sglang.srt.layers.layernorm_sp as layernorm_sp
    import sglang.srt.models.muse_spark as muse_spark

    with _lock:
        _state["min_tokens"] = int(min_tokens)
        _state["checked"] = None
        if _state["orig_runs_sp"] is None:
            _state["orig_runs_sp"] = layernorm_sp.runs_sp
            layernorm_sp.runs_sp = _runs_sp
            # The outer causal LM runs eagerly on every pass, also when the decoder replays a prefill graph; the
            # decoder entry runs at graph capture (bucket-sized dummy batch) and on eager passes.
            for cls in (muse_spark.MuseSpark1xForCausalLM, muse_spark.MuseSpark1xModel):
                if not getattr(cls.forward, "__wrapped_by_sp_gate__", False):
                    cls.forward = _wrap_entry(cls.forward)


def engage() -> bool:
    if _state["engaged"] is not None:
        return _state["engaged"]
    if not bed.enabled(NAME):
        _state["engaged"] = False
        return False
    for rel, sha in SGLANG_PINS.items():
        if not bed.pinned(rel, sha):
            bed.report(NAME, False, reason=f"pin mismatch: {rel}")
            _state["engaged"] = False
            return False
    try:
        min_tokens = int(
            os.environ.get("MUSE_VR_SP_GATE_MIN_TOKENS", str(DEFAULT_MIN_TOKENS))
        )
        if min_tokens < 0:
            raise ValueError("negative")
    except ValueError as e:
        bed.report(NAME, False, reason=f"bad MUSE_VR_SP_GATE_MIN_TOKENS ({e})")
        _state["engaged"] = False
        return False
    try:
        install(min_tokens)
    except Exception as e:  # noqa: BLE001
        bed.report(NAME, False, reason=f"install failed: {type(e).__name__}: {e}")
        _state["engaged"] = False
        return False
    bed.report(NAME, True, min_tokens=min_tokens)
    _state["engaged"] = True
    return True
