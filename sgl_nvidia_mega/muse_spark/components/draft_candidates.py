# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Component ``draft_candidates`` (env ``MUSE_VR_DRAFT_CANDIDATES=1``, default off).

The DFlash2 draft proposes its block through a candidate selector that needs, for every drafted position, the global
top-k (k = 16) tokens of the target head. SGLang computes them in ``candidate_topk`` (``models/dflash.py``): each
tensor-parallel rank projects onto its vocabulary shard and takes the shard's top-k, then two NCCL all-gathers
collect every rank's values and global ids, ``torch.topk`` picks the global top-k from the gathered ``tp * k``
entries and ``torch.gather`` maps them to ids; ``DFlash2DraftModel._transform_unary_logits`` then applies the output
multiplier and the soft cap in four elementwise launches. All of it runs inside the draft CUDA graph.

This component keeps the projection and the shard top-k and replaces the rest with one kernel
(``csrc/draft_candidates.cuh``): it pushes each rank's records through the tensor-parallel group's custom all-reduce
push plane (the same plane and phase protocol as SGLang's one-shot push all-reduce), and every rank merges the gathered
entries into the global top-k with the selection and ordering of ``torch.topk`` (ties included), maps them to global ids
and applies the same transform. Outputs are bit-identical to the stock path (``tests/test_draft_candidates.py``
on one GPU for the merge and transform; ``tests/dist_draft_candidates.py`` across ranks for the whole call).

Engaged only for: tensor-parallel size 2, 4 or 8 with a custom all-reduce push plane, a dense (unquantized) head,
the FlashInfer shard top-k (bf16 values, int64 positions), k = 16 and row blocks that fit one push slot; the CUDA
kernels are built and loaded on the first eager call (the warm-up before graph capture), and a capture that finds them
unloaded keeps the stock path. Every rank decides the same way for a call, so the exchange is issued on all ranks or
none. The tie order of ``torch.topk`` on the gathered entries depends on the PyTorch build (``TIE_RULES``, measured per
build with ``tests/test_draft_candidates.py``): the kernel orders the selected entries either as the sorting network
of the older builds does or stably (value descending, entry ascending), and the host picks the mode per call from the
build's rule, the row count and the number of gathered entries. Before the first use, a self-check compares the kernel
in the predicted mode with ``torch.topk`` on tie-heavy rows at the row counts where the rule switches; if it differs,
the component keeps the stock path (``stock:tie order self-check``). Builds without a rule keep the stock path.

``MUSE_VR_DRAFT_CANDIDATES_TOPK=1`` (default off; only with this component) also replaces the shard top-k (FlashInfer
``top_k(sorted=True, deterministic=True)``) by ``ops/shard_topk.topk16`` for at most
``MUSE_VR_DRAFT_CANDIDATES_TOPK_MAX_ROWS`` rows (default 56; tunable, tests/test_shard_topk.py --bench): the same
values in the same order, but among equal values the columns are taken in ascending order, while FlashInfer's choice
among equal values follows its own scan order. Candidate ids can therefore differ from stock at exact ties (equal bf16
logits); the outputs are no longer bit-identical in that case.
"""

from __future__ import annotations

import logging
from pathlib import Path

import torch

from . import bed

# The SGLang names this module reads (test/unit/test_contract.py pins them against the supported SGLang)
SGLANG_READS = {
    "srt/distributed/communication_op.py": ("get_tp_group",),
    "srt/models/dflash.py": (
        "_flashinfer_top_k",
        "get_parallel",
        "is_dense_head_weight",
        "should_apply_lm_head_quant_method",
        "tensor_model_parallel_all_gather",
        "_project_candidate_logits",
        "_radix_topk",
    ),
    "kernels/jit/utils/__init__.py": (
        "is_arch_support_pdl",
        "load_jit",
        "make_cpp_args",
    ),
    "srt/distributed/device_communicators/custom_all_reduce_v2.py": (
        "CustomAllReduceV2",
    ),
    "srt/plugins/hook_registry.py": ("HookRegistry", "HookType"),
}


logger = logging.getLogger(__name__)

NAME = "draft_candidates"  # env MUSE_VR_DRAFT_CANDIDATES
TARGET = "sglang.srt.models.dflash.DFlash2DraftModel.compute_candidates"
TOP_K = 16
_CUH = Path(__file__).resolve().parent / "csrc" / "draft_candidates.cuh"

# sha256 of the SGLang sources (relative to the directory holding the ``sglang`` package) this component relies on: the
# draft model (candidate_topk, its projection and shard top-k, the unary transform), the push plane and its
# communicator header, and the gather kernel whose protocol the exchange follows.
PINS = {
    "sglang/srt/models/dflash.py": (
        "59ae9eab87fd3416828604018c36ad6a57be07c4ff6d17f699bb9c0be8c719ff",
        "b7b541368cbbb072a679d08ba5bf0f952f4a43f02b6c078562b2ea7a9ddfd01b",
    ),
    "sglang/srt/distributed/device_communicators/custom_all_reduce_v2.py": (
        "277da88e2837e585f3ebdf4122906d29ea7a96cc30b981f7a028f6f63b6cf001",
        "f7b23a55019734be7b2f1958e20cc1f78a31251a035f2d75885911aa4a46e6b5",
    ),
    "sglang/kernels/jit/include/sgl_kernel/distributed/communicator.cuh": "30d5c75e86e86da7ce99cd6eab85212fdbc71f83a8e3c715c4e9f2c6c1d8986f",
}
# PyTorch builds whose torch.topk tie order on the gathered entries the bitwise test matched (torch.version.git_version),
# with the order each one uses (measured with tests/test_draft_candidates.py at every row count up to 8192):
#   "network": the sorting network's order at every row count;
#   "rows": the network's order for 32 gathered entries and below a row threshold, the stable order (value descending,
#           entry ascending) for more than 32 entries at or above it; the threshold is the device's multiprocessor
#           count (runtime attribute).
TIE_RULES = {
    "e0474491d782a73b73cf132f732a8513b3c9f49d": "network",
    "7bf52b3a9782d3440a17eb3f00983795f3b12af0": "rows",
}
VALIDATED_TORCH = tuple(TIE_RULES)

_state: dict = {"loaded": set(), "check": {}, "sms": {}}


def _sm_count(device) -> int:
    idx = (
        device.index
        if device is not None and device.index is not None
        else torch.cuda.current_device()
    )
    n = _state["sms"].get(idx)
    if n is None:
        n = _state["sms"][idx] = int(
            torch.cuda.get_device_properties(idx).multi_processor_count
        )
    return n


def stable_ties(world: int, rows: int, device=None) -> bool:
    """True when torch.topk on this build orders the ``world * k`` gathered entries of ``rows`` rows stably."""
    if TIE_RULES.get(torch.version.git_version) != "rows":
        return False
    return world * TOP_K > 32 and rows >= _sm_count(device)


def enabled() -> bool:
    return bed.enabled(NAME)


def pins_match() -> tuple[bool, str]:
    for rel, want in PINS.items():
        if not bed.pinned(rel, want):
            return False, f"{rel} differs from the pinned revision"
    if torch.version.git_version not in VALIDATED_TORCH:
        return (
            False,
            f"PyTorch build {torch.version.git_version[:12]} not in the validated set",
        )
    return True, "ok"


def _use_pdl() -> bool:
    from sglang.kernels.jit.utils import is_arch_support_pdl

    return bool(is_arch_support_pdl())


def module(world: int, pdl: bool):
    return _module(int(world), bool(pdl))


def _cache(fn):
    from functools import lru_cache

    return lru_cache(maxsize=None)(fn)


@_cache
def _module(world: int, pdl: bool):
    from sglang.kernels.jit.utils import load_jit, make_cpp_args

    args = make_cpp_args(world, pdl)
    return load_jit(
        "muse_vr_draft_candidates",
        *args,
        cuda_files=[str(_CUH)],
        cuda_wrappers=[
            ("run", f"MuseVrDraftCandidates<{args}>::run"),
            ("merge_only", f"MuseVrDraftCandidates<{args}>::merge_only"),
        ],
    )


def transform_params(model) -> tuple[float, float, bool, bool]:
    """(multiplier, soft cap, has multiplier, has soft cap) as ``_transform_unary_logits`` applies them."""
    cfg = model.draft_config
    mult = float(cfg.output_multiplier)
    cap = cfg.final_logit_softcapping
    return mult, float(cap) if cap is not None else 1.0, mult != 1.0, cap is not None


def merge_gathered(
    gathered_words: torch.Tensor,
    world: int,
    mult: float,
    cap: float,
    has_mult: bool,
    has_cap: bool,
    stable: bool | None = None,
):
    """Test entry: merge ``[N, world * k, 2]`` int32 records (as pushed) on one GPU. Returns (ids int64, vals fp32).
    ``stable``: the order mode; None takes the one this build's rule predicts for ``N`` rows."""
    n = gathered_words.shape[0]
    dev = gathered_words.device
    if stable is None:
        stable = stable_ties(world, n, dev)
    out_ids = torch.empty((n, TOP_K), dtype=torch.int64, device=dev)
    out_vals = torch.empty((n, TOP_K), dtype=torch.float32, device=dev)
    vals_like = torch.empty((n, TOP_K), dtype=torch.bfloat16, device=dev)
    ids_like = torch.empty((n, TOP_K), dtype=torch.int64, device=dev)
    module(world, False).merge_only(
        gathered_words,
        out_ids,
        out_vals,
        vals_like,
        ids_like,
        mult,
        cap,
        has_mult,
        has_cap,
        bool(stable),
    )
    return out_ids, out_vals


def _tie_rows(world: int, rows: int, device):
    """Deterministic tie-heavy gathered entries (bf16 values from a few levels, signed zeros included; distinct ids)."""
    e = world * TOP_K
    levels = torch.tensor([3.0, 2.5, 2.5, 2.0, 1.0, -0.0, 0.0, 7.0], device=device)
    r = torch.arange(rows, device=device).unsqueeze(1)
    c = torch.arange(e, device=device).unsqueeze(0)
    vals = (
        levels[(r * 5 + c * 3 + (r * c) % 7) % levels.numel()]
        .to(torch.bfloat16)
        .contiguous()
    )
    gids = (c * 977 + 13).expand(rows, e).contiguous()
    return vals, gids


def _words(vals_bf16: torch.Tensor, gids: torch.Tensor) -> torch.Tensor:
    bits = vals_bf16.view(torch.int16).to(torch.int32) & 0xFFFF
    return torch.stack(
        [(bits << 16) | 1, (gids + 1).to(torch.int32)], dim=-1
    ).contiguous()


def self_check(world: int, device) -> str | None:
    """None when the merge kernel, in the mode the rule predicts, equals torch.topk + gather on tie-heavy rows at the
    row counts around the rule's switch (1, T - 1, T, 2T); otherwise the first mismatch. At T and 2T two torch.topk
    calls must also agree (a stock order that varies from call to call keeps the stock path). Inputs are deterministic
    and the shapes deterministic, so every rank reaches the same answer."""
    t = _sm_count(device)
    for rows in sorted({1, max(1, t - 1), t, 2 * t}):
        lab = (
            "T"
            if rows == t
            else ("2T" if rows == 2 * t else ("T-1" if rows == t - 1 else str(rows)))
        )
        vals, gids = _tie_rows(world, rows, device)
        ref_vals, sel = torch.topk(vals.float(), TOP_K, dim=-1)
        if rows >= t:
            again_vals, again_sel = torch.topk(vals.float(), TOP_K, dim=-1)
            if not (
                torch.equal(sel, again_sel)
                and torch.equal(
                    again_vals.view(torch.int32), ref_vals.view(torch.int32)
                )
            ):
                return f"world {world} rows {lab}: two torch.topk calls differ"
        ref_ids = torch.gather(gids, -1, sel).long()
        ids, out = merge_gathered(_words(vals, gids), world, 1.0, 1.0, False, False)
        if not (
            torch.equal(ids, ref_ids)
            and torch.equal(out.view(torch.int32), ref_vals.view(torch.int32))
        ):
            return f"world {world} rows {lab}"
    return None


def _preload(world: int, device) -> None:
    """Build and load the module and launch its merge kernel once (on zeros, local only) outside graph capture."""
    key = (world, _use_pdl(), device.index)
    if key in _state["loaded"]:
        return
    words = torch.zeros((1, world * TOP_K, 2), dtype=torch.int32, device=device)
    merge_gathered(words, world, 1.0, 1.0, False, False)
    bad = self_check(world, device)
    _state["check"][key] = bad
    if bad is not None:
        logger.warning(
            "%s: tie order self-check failed (%s); the stock path is kept", NAME, bad
        )
    _module(world, _use_pdl())  # same build for the plane path
    if bed.flag(NAME + "_topk"):
        from .ops import shard_topk

        shard_topk.module(shard_topk.use_pdl())
    torch.cuda.current_stream(device).synchronize()
    _state["loaded"].add(key)


def _eligible(model, hidden: torch.Tensor):
    """(reason, context): reason is None when the fused path applies; context carries what it needs."""
    import sglang.srt.distributed.communication_op as co
    import sglang.srt.models.dflash as d
    from sglang.srt.distributed.device_communicators.custom_all_reduce_v2 import (
        CustomAllReduceV2,
    )

    lm_head = getattr(model, "lm_head", None)
    selector = getattr(model, "candidate_selector", None)
    if lm_head is None or selector is None or int(selector.top_k) != TOP_K:
        return "head/selector", None
    if d._flashinfer_top_k is None:
        return "no shard top-k kernel", None
    if not (hidden.is_cuda and hidden.dim() == 2):
        return "hidden layout", None
    weight = getattr(lm_head, "weight", None)
    if d.should_apply_lm_head_quant_method(
        lm_head, getattr(lm_head, "quant_method", None)
    ):
        return "quantized head", None
    if not d.is_dense_head_weight(weight):
        return "head weight", None
    # the same lookups the stock path makes: tensor-parallel size as candidate_topk reads it, group as its gather does
    tp = int(d.get_parallel().tp_size)
    group = co.get_tp_group()
    if tp not in (2, 4, 8) or group.world_size != tp:
        return "tensor-parallel size", None
    comm = getattr(group, "ca_comm", None)
    if (
        not isinstance(comm, CustomAllReduceV2)
        or comm.disabled
        or getattr(comm, "obj", None) is None
    ):
        return "no custom all-reduce", None
    n = hidden.shape[0]
    if n == 0:
        return "no rows", None
    if n * TOP_K * 8 > int(comm.max_push_size):
        return "rows above push slot", None
    key = (tp, _use_pdl(), hidden.device.index)
    if torch.cuda.is_current_stream_capturing():
        if key not in _state["loaded"]:
            return "kernel not loaded before graph capture", None
    else:
        _preload(tp, hidden.device)
    if _state["check"].get(key) is not None:
        return "tie order self-check", None
    return None, (lm_head, comm, tp)


def _stock_tail(model, lm_head, vals, ids):
    """The rest of ``candidate_topk`` (tensor-parallel branch, pinned revision) and ``_transform_unary_logits``."""
    import sglang.srt.models.dflash as d

    global_ids = ids.long() + int(lm_head.shard_indices.org_vocab_start_index)
    gathered_vals = d.tensor_model_parallel_all_gather(vals.float(), dim=-1)
    gathered_ids = d.tensor_model_parallel_all_gather(global_ids, dim=-1)
    top_vals, sel = torch.topk(gathered_vals, TOP_K, dim=-1)
    return torch.gather(gathered_ids, -1, sel).long(), model._transform_unary_logits(
        top_vals
    )


def n_rows_ok(logits) -> bool:
    return logits.shape[0] <= bed.env_int("MUSE_VR_DRAFT_CANDIDATES_TOPK_MAX_ROWS", 56)


def compute_candidates(original, model, hidden: torch.Tensor):
    """``DFlash2DraftModel.compute_candidates`` with the gather, global top-k and transform fused (pinned body:
    ``candidate_topk`` then ``_transform_unary_logits``)."""
    reason, ctx = _eligible(model, hidden)
    if reason is not None:
        bed.count(NAME, "stock:" + reason)
        return original(model, hidden)
    import sglang.srt.models.dflash as d

    lm_head, comm, tp = ctx
    shard = lm_head.shard_indices
    logits = d._project_candidate_logits(
        hidden, lm_head, num_org=int(shard.num_org_elements), use_quant_head=False
    )
    vals = ids = None
    if bed.flag(NAME + "_topk") and n_rows_ok(logits):
        from .ops import shard_topk

        if shard_topk.supported(logits):
            vals, ids = shard_topk.topk16(logits)
    if vals is None:
        vals, ids = d._radix_topk(logits, TOP_K)
    if (
        vals.dtype != torch.bfloat16
        or ids.dtype != torch.int64
        or not (vals.is_contiguous() and ids.is_contiguous())
    ):
        # nothing collective has happened yet on any rank: every rank sees the same dtypes and takes this branch; the
        # rest of candidate_topk and the transform run as SGLang runs them, on the shard top-k already computed
        bed.count(NAME, "stock:shard top-k dtype")
        return _stock_tail(model, lm_head, vals, ids)
    n = hidden.shape[0]
    out_ids = torch.empty((n, TOP_K), dtype=torch.int64, device=hidden.device)
    out_vals = torch.empty((n, TOP_K), dtype=torch.float32, device=hidden.device)
    mult, cap, has_mult, has_cap = transform_params(model)
    stable = stable_ties(tp, n, hidden.device)
    _module(tp, _use_pdl()).run(
        comm.obj,
        vals,
        ids,
        out_ids,
        out_vals,
        int(shard.org_vocab_start_index),
        mult,
        cap,
        has_mult,
        has_cap,
        stable,
    )
    bed.count(NAME, f"fused:rows={n}:tp={tp}:{'stable' if stable else 'network'}")
    return out_ids, out_vals


def register() -> bool:
    if not enabled():
        return False
    ok, why = pins_match()
    if not ok:
        bed.report(NAME, False, reason=why)
        return False
    from sglang.srt.plugins.hook_registry import HookRegistry, HookType

    def hook(original, *args, **kwargs):
        if kwargs or len(args) != 2:
            return original(*args, **kwargs)
        return compute_candidates(original, *args)

    HookRegistry.register(TARGET, hook, HookType.AROUND)
    bed.report(NAME, True)
    return True
