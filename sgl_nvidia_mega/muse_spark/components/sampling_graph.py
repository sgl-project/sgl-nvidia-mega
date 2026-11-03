# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Component ``sampling_graph`` (env ``MUSE_VR_SAMPLING_GRAPH=1``, default off).

After the target verify, SGLang's DFLASH worker samples the accept lengths and bonus tokens eagerly
(``DFlashWorkerV2._selector_sampling_accept``: scatter of the selector's sparse draft probabilities into a dense buffer,
softmax with per-row temperature, two uniform draws, chain sampling, accept/bonus gather, reset of the dense buffer).
That is a few dozen small launches whose host cost can exceed their GPU time at small batches. This component captures
the same call, per batch size, into a CUDA graph and replays it: each step copies the call's inputs (verify logits,
draft tokens, selector candidates and probabilities, temperatures) into the graph's own buffers, replays, and returns
copies of the two outputs. The captured body is SGLang's method itself, with ``softmax_split`` and
``spec_sample_split`` inside when those are enabled; the uniform draws come from the default CUDA generator through
PyTorch's graph-safe RNG, so a replay draws what the eager call would draw from the same generator state
(``tests/test_sampling_graph.py`` checks replay against the eager method, bit for bit, from equal RNG states).

The first call for a batch size runs the method eagerly (that is the step's result, with the step's uniform draws) and
then captures it (thread-local capture mode; the generator state is restored after capture, so replays continue the
same random stream). The graph reads the draft tokens and the selector's candidates and probabilities from the
caller's persistent tensors; when the caller passes other tensors the batch size is captured again.
Engaged only for batches of at most ``MUSE_VR_SAMPLING_GRAPH_MAX_BS`` requests (default 8; tunable), without top-k or
top-p (whose target-probability path synchronizes with the host), without block verification and outside any other
capture; everything else calls the method as stock.
"""

from __future__ import annotations

import logging
import os
import threading
import types

import torch

from . import bed

logger = logging.getLogger(__name__)

NAME = "sampling_graph"  # env MUSE_VR_SAMPLING_GRAPH
TARGET = (
    "sglang.srt.speculative.dflash_worker_v2.DFlashWorkerV2._selector_sampling_accept"
)
ENV_MAX_BS = "MUSE_VR_SAMPLING_GRAPH_MAX_BS"
DEFAULT_MAX_BS = 8

# sha256 of the SGLang sources (relative to the directory holding the ``sglang`` package) whose code is captured: the
# worker method, the accept/softmax functions it calls and the chain sampling kernel.
PINS = {
    "sglang/srt/speculative/dflash_worker_v2.py": (
        "7d462f5f175828bcc3b4d47fcdfa33cb6af4b14669ef016eeb79b1a5191eec8c",
        "8f82de8d666f81f3c1aa43dc651006f4182ff58593fdfe4eea28c5ee2a14bfc0",
    ),
    "sglang/kernels/ops/speculative/dspark/dspark_accept.py": (
        "6202a51f6d2f3a79fef3cbef2aee04296640087be2402d8b11a7f1b394015a9c",
        "4d87d25d0510a309e7f3484c0d2c6d806da39bd5557096c49b9ac2a92e5c3ddf",
    ),
    "sglang/kernels/ops/speculative/reject_sampling.py": "62caac17755082e2f89b1c81a743f55c81c31b7ff0485b7533c0267e72e9262c",
    "sglang/srt/speculative/dflash_utils.py": (
        "6e0423578bade4a8c58b921430df678840c6fa2de6ea411fe58eb77c98d5a820",
        "35e885f930e95ed82a1f47a0dfa2dab583225d170c6cabe54f6c7bb4e3abbf79",
    ),
}

_tls = threading.local()
_graphs: dict = {}
_state: dict = {"pool": None, "max_bs": None}


def enabled() -> bool:
    return bed.enabled(NAME)


def pins_match() -> tuple[bool, str]:
    for rel, want in PINS.items():
        if not bed.pinned(rel, want):
            return False, f"{rel} differs from the pinned revision"
    return True, "ok"


def capturing() -> bool:
    """True while this component captures the sampling call (other components may then run under capture)."""
    return getattr(_tls, "capturing", False)


def _max_bs() -> int:
    v = _state["max_bs"]
    if v is None:
        try:
            v = int(os.environ.get(ENV_MAX_BS, str(DEFAULT_MAX_BS)))
        except ValueError:
            v = DEFAULT_MAX_BS
        _state["max_bs"] = v
    return v


def _eligible(
    worker, candidates, logits, candidate_ids, q_rows, sampling_info
) -> str | None:
    if torch.cuda.is_current_stream_capturing():
        return "graph capture"
    if candidates.dim() != 2 or candidates.shape[0] > _max_bs():
        return "batch above cap"
    if getattr(worker, "_block_verification", True):
        return "block verification"
    if bool(getattr(sampling_info, "need_top_k_sampling", True)) or bool(
        getattr(sampling_info, "need_top_p_sampling", True)
    ):
        return "top-k/top-p"
    temps = getattr(sampling_info, "temperatures", None)
    ts = (candidates, logits, candidate_ids, q_rows, temps)
    if not all(isinstance(t, torch.Tensor) and t.is_cuda for t in ts):
        return "inputs"
    if logits.dtype != torch.float32 or logits.dim() != 2:
        return "logits dtype/layout"
    buf = getattr(worker, "_draft_probs_buf", None)
    bs, block = candidates.shape
    if (
        buf is None
        or buf.shape[0] < bs
        or tuple(buf.shape[1:]) != (block - 1, logits.shape[-1])
    ):
        return "dense draft buffer not sized yet"
    return None


def _key(worker, candidates, logits, candidate_ids, q_rows, temps):
    buf = worker._draft_probs_buf
    return (
        id(worker),
        torch.cuda.current_device(),
        candidates.shape[0],
        tuple(logits.shape),
        tuple(candidate_ids.shape),
        candidate_ids.dtype,
        q_rows.dtype,
        candidates.dtype,
        temps.dtype,
        buf.data_ptr(),
        tuple(buf.shape),
    )


def _pool():
    if _state["pool"] is None:
        _state["pool"] = torch.cuda.graph_pool_handle()
    return _state["pool"]


def _same(a: torch.Tensor, b: torch.Tensor) -> bool:
    return (
        a.data_ptr() == b.data_ptr()
        and a.shape == b.shape
        and a.stride() == b.stride()
        and a.dtype == b.dtype
    )


def _capture(
    original,
    worker,
    candidates,
    logits,
    candidate_ids,
    q_rows,
    sampling_info,
    draft_input,
):
    """Run the call eagerly (its result is this step's result, consuming the step's uniform draws as stock does),
    then capture it. The graph reads the draft tokens and selector outputs from the caller's own (persistent) tensors,
    which it keeps referenced, and the logits and temperatures from buffers of its own."""
    ins = {
        "candidates": candidates,
        "candidate_ids": candidate_ids,
        "q_rows": q_rows,
        "logits": torch.empty_like(logits, memory_format=torch.contiguous_format),
        "temps": torch.empty_like(sampling_info.temperatures),
    }
    info = types.SimpleNamespace(
        need_top_k_sampling=False, need_top_p_sampling=False, temperatures=ins["temps"]
    )
    eager = original(
        worker,
        candidates=candidates,
        next_token_logits=logits,
        candidate_ids=candidate_ids,
        q_rows=q_rows,
        sampling_info=sampling_info,
        draft_input=draft_input,
    )
    ins["logits"].copy_(logits)
    ins["temps"].copy_(sampling_info.temperatures)

    def body():
        return original(
            worker,
            candidates=ins["candidates"],
            next_token_logits=ins["logits"],
            candidate_ids=ins["candidate_ids"],
            q_rows=ins["q_rows"],
            sampling_info=info,
            draft_input=draft_input,
        )

    graph = torch.cuda.CUDAGraph()
    rng = torch.cuda.get_rng_state()
    _tls.capturing = True
    try:
        with torch.cuda.graph(graph, pool=_pool(), capture_error_mode="thread_local"):
            out_len, out_bonus = body()
    finally:
        _tls.capturing = False
        torch.cuda.set_rng_state(rng)
    return {"graph": graph, "ins": ins, "out": (out_len, out_bonus)}, eager


def selector_sampling_accept(original, worker, **kw):
    """``DFlashWorkerV2._selector_sampling_accept`` replayed from a per-batch-size CUDA graph when eligible."""
    if len(kw) != 6 or "next_token_logits" not in kw:
        return original(worker, **kw)
    cand, logits, ids, q, info = (
        kw["candidates"],
        kw["next_token_logits"],
        kw["candidate_ids"],
        kw["q_rows"],
        kw["sampling_info"],
    )
    reason = _eligible(worker, cand, logits, ids, q, info)
    if reason is not None:
        bed.count(NAME, "stock:" + reason)
        return original(worker, **kw)
    key = _key(worker, cand, logits, ids, q, info.temperatures)
    entry = _graphs.get(key)
    if entry is not None and entry["graph"] is not None:
        ins = entry["ins"]
        if not (
            _same(ins["candidates"], cand)
            and _same(ins["candidate_ids"], ids)
            and _same(ins["q_rows"], q)
        ):
            entry = None  # the caller's tensors moved: capture again for them
    if entry is None:
        try:
            entry, eager = _capture(
                original, worker, cand, logits, ids, q, info, kw["draft_input"]
            )
        except Exception as e:  # noqa: BLE001 - capture failure leaves this batch size on the stock path
            logger.warning(
                "muse_vr_plugin: %s capture failed for bs=%d (%s: %s)",
                NAME,
                cand.shape[0],
                type(e).__name__,
                e,
            )
            _graphs[key] = {"graph": None}
            bed.count(NAME, "capture failed")
            return original(worker, **kw)
        _graphs[key] = entry
        bed.count(NAME, f"captured:bs={cand.shape[0]}")
        return eager
    if entry["graph"] is None:
        return original(worker, **kw)
    ins = entry["ins"]
    ins["logits"].copy_(logits)
    ins["temps"].copy_(info.temperatures)
    entry["graph"].replay()
    bed.count(NAME, "replay")
    out_len, out_bonus = entry["out"]
    return out_len.clone(), out_bonus.clone()


def register() -> bool:
    if not enabled():
        return False
    ok, why = pins_match()
    if not ok:
        bed.report(NAME, False, reason=why)
        return False
    from sglang.srt.plugins.hook_registry import HookRegistry, HookType

    def hook(original, *args, **kwargs):
        if len(args) != 1:
            return original(*args, **kwargs)
        return selector_sampling_accept(original, args[0], **kwargs)

    HookRegistry.register(TARGET, hook, HookType.AROUND)
    bed.report(NAME, True)
    return True
