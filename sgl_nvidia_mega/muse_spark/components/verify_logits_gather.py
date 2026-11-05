# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Component ``verify_logits_gather`` (env ``MUSE_VR_VERIFY_LOGITS_GATHER=1``, default off).

With speculative verification under non-greedy sampling, SGLang's DFLASH worker leaves the target-verify logits as
vocab-parallel shards and gathers them after the verify step (``LogitsProcessor.finish_deferred_gather``): an NCCL
``all_gather_into_tensor`` into a ``[tp, rows, vocab / tp]`` stack, then ``gathered_softcap_logits`` writes the
softcapped ``[rows, vocab]`` row block. This component keeps both steps and only moves the gather onto SGLang's own
NVLink collectives of the tensor-parallel group's custom all-reduce (``NVLinkVocabGather``): the pull kernel into a
symmetric-memory stack that is allocated once and reused. It produces the same rank-major stack as the NCCL call, so
the softcap kernel reads bit-identical input and the logits are bit-identical (``tests/dist_verify_logits_gather.py``).

Row blocks above ``MUSE_VR_VERIFY_LOGITS_GATHER_MAX_ROWS`` (default 64; tunable, same script with ``--bench``) keep
the NCCL gather, as does any other case: no multicast plane, more rows than the stack holds, CUDA-graph capture, other
dtypes or layouts. ``MUSE_VR_PULL_BOUNDED=1`` (module ``pull_bounded``) runs the pull kernel's copy whose barrier
waits end in a trap after a time limit instead of polling forever. The push kernel
(``MUSE_VR_VERIFY_LOGITS_GATHER_PUSH=1``, default off) is used instead of the pull kernel when this rank's shard fits
one push slot; it is exact except that it delivers +0.0 as -0.0 (it reserves +0.0 as its empty-slot marker).
``MUSE_VR_VERIFY_LOGITS_GATHER_SHADOW=1`` (debug) also runs the stock gather on every call and tallies bitwise matches
with ``bed.shadow_result``.

The stack holds ``MUSE_VR_VERIFY_LOGITS_GATHER_ROWS`` rows per rank (default: the largest verify batch, max running
requests x speculative draft tokens, capped at ``MUSE_VR_VERIFY_LOGITS_GATHER_MAX_ROWS`` because larger row blocks
never use it). It is allocated on the first eligible call, which every rank of the group makes in the same order (the
call is collective), never inside CUDA-graph capture.

Not combined with ``verify_topk_sample``: with that component enabled, sampled verify steps take their top-k from
per-rank records and skip the full-row gather, so the full-row gathers that remain on those steps feed only SGLang's
synthetic-acceptance benchmark mode. A speedup there would not reach served traffic, so when
``MUSE_VR_VERIFY_TOPK_SAMPLE`` is enabled this component refuses to install (logged as an error, engagement record
``engaged=False``) and every gather stays stock.
"""

from __future__ import annotations

import logging
import os

import torch

from . import bed, pull_bounded

# The SGLang names this module reads (test/unit/test_contract.py pins them against the supported SGLang)
SGLANG_READS = {
    "srt/layers/logits_processor.py": ("gathered_softcap_logits", "get_parallel"),
    "srt/runtime_context.py": ("get_schedule", "get_spec"),
    "srt/distributed/device_communicators/vocab_gather.py": (
        "NVLinkVocabGather",
        "make_vocab_gather",
    ),
    "srt/plugins/hook_registry.py": ("HookRegistry", "HookType"),
}


logger = logging.getLogger(__name__)

NAME = "verify_logits_gather"  # env MUSE_VR_VERIFY_LOGITS_GATHER
ENV_ROWS = "MUSE_VR_VERIFY_LOGITS_GATHER_ROWS"
ENV_PUSH = "MUSE_VR_VERIFY_LOGITS_GATHER_PUSH"
ENV_MAX_ROWS = "MUSE_VR_VERIFY_LOGITS_GATHER_MAX_ROWS"
ENV_SHADOW = "MUSE_VR_VERIFY_LOGITS_GATHER_SHADOW"  # debug: also run the stock gather and compare bits (host sync)
EXCLUSIVE_WITH = (
    "verify_topk_sample"  # env MUSE_VR_VERIFY_TOPK_SAMPLE; see the module docstring
)
DEFAULT_MAX_ROWS = 64  # tunable (tests/dist_verify_logits_gather.py --bench; MUSE_VR_VERIFY_LOGITS_GATHER_MAX_ROWS)
TARGET = "sglang.srt.layers.logits_processor.LogitsProcessor.finish_deferred_gather"

# sha256 of the SGLang sources (relative to the directory holding the ``sglang`` package) this component relies on:
# the deferred-gather method it mirrors, the NVLink gather class and wrappers it calls, their kernel, and the
# communicator that owns the push and pull planes.
PINS = {
    "sglang/srt/layers/logits_processor.py": (
        "ecc0318937eee486414f6ae7165f2d1de17c01794a25aae83a99af61cbb0e3d0",
        "3558330bccf3ea0e46aabdaa0dee6ae7daf294c8e15b9a4e9744dab6883c4398",
    ),
    "sglang/srt/distributed/device_communicators/vocab_gather.py": "7363debb7d54a2873971c33c14bcb46487db2b9038105ada14fa614a282aff55",
    "sglang/kernels/ops/communication/nvlink_comm.py": (
        "6ba171bd054fd70b1248be37214419887f6731eea9c927c6e0aa9ee0997bc3fd",
        "0388df5a897cc9c437e7fb0a2e6e21231a7a8d0ce0033798846331c0698bdaaa",
    ),
    "sglang/kernels/jit/csrc/distributed/nvlink_comm.cuh": (
        "8fb785518279cf8944a6a438036cfe39a2c00ecc6b0b5b86a6a2a0b46ad01e37",
        "e3c8f3c93a61c8788d4b97b9fbfa9cc0b5b1c9ee1b2154c9e7e04ab642431367",
    ),
    "sglang/srt/distributed/device_communicators/custom_all_reduce_v2.py": (
        "277da88e2837e585f3ebdf4122906d29ea7a96cc30b981f7a028f6f63b6cf001",
        "f7b23a55019734be7b2f1958e20cc1f78a31251a035f2d75885911aa4a46e6b5",
    ),
    "sglang/srt/speculative/dflash_worker_v2.py": (
        "7d462f5f175828bcc3b4d47fcdfa33cb6af4b14669ef016eeb79b1a5191eec8c",
        "8f82de8d666f81f3c1aa43dc651006f4182ff58593fdfe4eea28c5ee2a14bfc0",
    ),
}

_state: dict = {"gather": None, "key": None}


def enabled() -> bool:
    return bed.enabled(NAME)


def pins_match() -> tuple[bool, str]:
    for rel, want in PINS.items():
        if not bed.pinned(rel, want):
            return False, f"{rel} differs from the pinned revision"
    return True, "ok"


def _rows_capacity() -> int:
    v = os.environ.get(ENV_ROWS, "").strip()
    if v:
        return max(int(v), 0)
    try:
        from sglang.srt.runtime_context import get_schedule, get_spec

        running = int(get_schedule().max_running_requests or 0)
        ndraft = int(get_spec().speculative_num_draft_tokens or 1)
        return max(
            min(running * ndraft, _max_rows()), 0
        )  # row blocks above the cap take the NCCL gather
    except Exception:  # noqa: BLE001 - no server context: push path only
        return 0


def _max_rows() -> int:
    return bed.env_int(ENV_MAX_ROWS, DEFAULT_MAX_ROWS)


def _push_allowed() -> bool:
    """The push kernel marks empty slots with +0.0 and sends +0.0 as -0.0, so it is exact up to the sign of zeros
    only; off by default (``MUSE_VR_VERIFY_LOGITS_GATHER_PUSH=1`` allows it). The pull kernel copies bits."""
    return bed.flag(NAME + "_push")


def _nvlink_gather(group, width: int):
    """SGLang's NVLinkVocabGather for ``group`` and this shard width, created once (collective), or None."""
    from sglang.srt.distributed.device_communicators.vocab_gather import (
        NVLinkVocabGather,
        make_vocab_gather,
    )

    key = (id(group), width)
    if _state["key"] != key:
        pull_bounded.prepare(
            torch.float32
        )  # before the plane's collective set-up (MUSE_VR_PULL_BOUNDED)
        vg = make_vocab_gather(
            group, local_width=width, dtype=torch.float32, symm_rows=_rows_capacity()
        )
        _state["gather"] = vg if isinstance(vg, NVLinkVocabGather) else None
        _state["key"] = key
        pull_rows = (
            0
            if _state["gather"] is None or _state["gather"].pull_out is None
            else (_state["gather"].pull_out.shape[0] // group.world_size)
        )
        bed.report(
            NAME + ":plane",
            _state["gather"] is not None,
            world=group.world_size,
            width=width,
            pull_rows=pull_rows,
        )
    return _state["gather"]


def gather_stacked(group, local: torch.Tensor):
    """``[world, rows, width]`` stack of every rank's ``local`` (rank-major, as ``all_gather_into_tensor``), or None
    when the NVLink planes cannot take it."""
    rows, width = local.shape
    world = group.world_size
    if rows > _max_rows():
        return None, "rows above MUSE_VR_VERIFY_LOGITS_GATHER_MAX_ROWS"
    vg = _nvlink_gather(group, width)
    if vg is None:
        return None, "no NVLink gather"
    if (
        _push_allowed()
        and local.nbytes <= vg.slot_bytes
        and width * local.element_size() % 16 == 0
    ):
        return vg.gather_stacked(local).view(world, rows, width), "push"
    if vg.pull_out is not None and world * rows <= vg.pull_out.shape[0]:
        out = vg.pull_out[: world * rows]
        # SGLang's pull kernel, or its copy with bounded barrier waits (MUSE_VR_PULL_BOUNDED=1): same bytes
        pull_bounded.all_gather_pull(
            vg.comm, local, out, out_mc_ptr=vg.pull_mc_ptr, site=NAME
        )
        # The stack is read by the softcap kernel below on this stream; the pull kernel's entry barrier keeps every
        # other rank from writing it again until this rank has issued its next gather.
        return out.view(world, rows, width), "pull"
    return None, "rows beyond capacity"


def _eligible(lp, logits: torch.Tensor) -> str | None:
    if not (
        logits.is_cuda
        and logits.dtype == torch.float32
        and logits.dim() == 2
        and logits.is_contiguous()
    ):
        return "dtype/layout"
    if not lp.final_logit_softcapping or not lp.is_deferred_shard(logits):
        return "not a deferred softcapped shard"
    if torch.cuda.is_current_stream_capturing():
        return "graph capture"
    return None


def finish_deferred_gather(original, lp, logits: torch.Tensor) -> torch.Tensor:
    """``LogitsProcessor.finish_deferred_gather`` with the NCCL gather replaced (pinned revision of the body:
    ``_gather_softcap_fused(logits, None, use_buffer=False)``)."""
    import sglang.srt.layers.logits_processor as lpmod

    reason = _eligible(lp, logits)
    # the tensor-parallel group as the stock body names it on each layout
    group = (
        lpmod.get_tp_group()
        if hasattr(lpmod, "get_tp_group")
        else lpmod.get_parallel().tp_group
    )
    if reason is None and group.world_size < 2:
        reason = "single rank"
    gathered = None
    if reason is None:
        gathered, path = gather_stacked(group, logits)
        if gathered is None:
            reason = path
    if gathered is None:
        bed.count(NAME, "stock:" + reason)
        return original(lp, logits)
    num_tokens, shard_width = logits.shape
    bed.count(NAME, f"{path}:rows={num_tokens}")
    width = min(lp.vocab_size, group.world_size * shard_width)
    output = torch.empty((num_tokens, width), dtype=torch.float32, device=logits.device)
    out = lpmod.gathered_softcap_logits(gathered, lp.final_logit_softcapping, output)
    if bed.flag(NAME + "_shadow"):
        ref = original(lp, logits)
        bed.shadow_result(
            NAME, bool(torch.equal(ref.view(torch.int32), out.view(torch.int32)))
        )
    return out


def refused() -> str | None:
    """Why this component must not install in this process (None: it may)."""
    if bed.enabled(EXCLUSIVE_WITH):
        return f"refused: {EXCLUSIVE_WITH} is enabled (its sampled verify steps skip the full-row gather)"
    return None


def register() -> bool:
    if not enabled():
        return False
    why = refused()
    if why is not None:
        logger.error("muse_vr_plugin: %s %s; the stock gather is kept", NAME, why)
        bed.report(NAME, False, reason=why)
        return False
    ok, why = pins_match()
    if not ok:
        bed.report(NAME, False, reason=why)
        return False
    from sglang.srt.plugins.hook_registry import HookRegistry, HookType

    def hook(original, *args, **kwargs):
        if kwargs or len(args) != 2:
            return original(*args, **kwargs)
        return finish_deferred_gather(original, *args)

    HookRegistry.register(TARGET, hook, HookType.AROUND)
    bed.report(NAME, True)
    return True
