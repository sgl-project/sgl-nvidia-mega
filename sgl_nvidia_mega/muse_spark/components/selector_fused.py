# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Component ``selector_fused`` (env ``MUSE_VR_SELECTOR_FUSED=1``, default off).

The DFLASH2 candidate selector builds the whole lattice of edge scores (``CandidateSelector.build_lattice``: the
projected draft hidden state, codebook gathers, an elementwise product and a batched GEMM, several small kernels) and
then walks one path through it (``sample_path``, one program per request). The walk reads a single row per slot, the row
of the candidate chosen at the slot before. This component makes ``build_lattice`` hand over its inputs instead of the
lattice (a placeholder tensor of the lattice's shape, registered with its inputs), and ``sample_path`` on that
placeholder runs ``csrc/selector_fused.cuh``: one kernel that builds the lattice in shared memory, decides every row's
pick and walks them (bf16, at most 8 slots, K 8 or 16: one cluster per request with a CTA per slot; otherwise one block
per request).
The uniform draws stay where the stock code takes them (same generator calls, same order).

Numerics: draft-numerics. The score of a candidate is the same expression with operands in the same dtypes as stock
(the stock einsum multiplies rows of the two codebook parameters and the projection output in their own dtypes, so
bf16 parameters give bf16 operands with fp32 accumulation there and here, fp32 parameters fp32 operands; the bed count
records the codebook dtype served), but its dot product is summed in another order (and the walk's exponential and
sums are this kernel's), so proposals and q rows can differ in the last places from the stock path; the verification
and the committed tokens follow the target as before (tests/test_selector_fused.py reports the agreement with the
stock path). The uniform draws are the stock ones, consumed one per slot as the stock walk does.

Engaged when the selector's top_k <= 32, at most 16 slots, rank <= 1024, codebooks bf16 or fp32, the projected hidden
state bf16 or fp32, and the staged rows fit the block's shared memory; otherwise both calls run stock. Hooks:
``CandidateSelector.build_lattice`` and ``CandidateSelector.sample_path`` (AROUND), and ``_SelectorDraftSampler.__call__``
(AROUND, calls the original): inside the draft-graph sampler the walk writes its tokens and q rows straight into the
sampler's ``out`` / ``q_out`` buffers, which makes the sampler's two copies of them exact aliases (no copy launched; the
values are the same either way). A placeholder that reaches anything but
``sample_path`` would be read uninitialised, so the only callers of ``build_lattice`` (``_selector_lattice``, whose result
goes to ``sample_path`` and nowhere else, at both call sites of the pinned revision) are pinned.
"""

from __future__ import annotations

import functools
import logging
import threading
from pathlib import Path

import torch

from . import bed

# The SGLang names this module reads (test/unit/test_contract.py pins them against the supported SGLang)
SGLANG_READS = {
    "kernels/jit/utils/__init__.py": ("load_jit",),
    "srt/plugins/hook_registry.py": ("HookRegistry", "HookType"),
}


logger = logging.getLogger(__name__)

NAME = "selector_fused"  # env MUSE_VR_SELECTOR_FUSED
T_LATTICE = "sglang.srt.models.dflash.CandidateSelector.build_lattice"
T_SAMPLE = "sglang.srt.models.dflash.CandidateSelector.sample_path"
T_CALL = "sglang.srt.speculative.dflash_worker_v2._SelectorDraftSampler.__call__"
_CUH = Path(__file__).resolve().parent / "csrc" / "selector_fused.cuh"

PINS = {
    "sglang/srt/models/dflash.py": (
        "59ae9eab87fd3416828604018c36ad6a57be07c4ff6d17f699bb9c0be8c719ff",
        "b7b541368cbbb072a679d08ba5bf0f952f4a43f02b6c078562b2ea7a9ddfd01b",
    ),
    "sglang/srt/speculative/dflash_worker_v2.py": (
        "7d462f5f175828bcc3b4d47fcdfa33cb6af4b14669ef016eeb79b1a5191eec8c",
        "8f82de8d666f81f3c1aa43dc651006f4182ff58593fdfe4eea28c5ee2a14bfc0",
    ),
}
MAX_K, MAX_SLOTS, MAX_RANK, SMEM_CAP = 32, 16, 1024, 200 * 1024
SINGLE_BLOCK = (
    False  # tests: force the one-block kernel where the cluster kernel would run
)

_tls = threading.local()


def enabled() -> bool:
    return bed.enabled(NAME)


def pins_match() -> tuple[bool, str]:
    for rel, want in PINS.items():
        if not bed.pinned(rel, want):
            return False, f"{rel} differs from the pinned revision"
    return True, "ok"


@functools.lru_cache(maxsize=None)
def module():
    from sglang.kernels.jit.utils import load_jit

    return load_jit(
        "muse_vr_selector_fused",
        cuda_files=[str(_CUH)],
        cuda_wrappers=[("run", "muse_vr::selector_fused::SelectorFused::run")],
    )


def _code(dtype):
    return {torch.bfloat16: 0, torch.float32: 1}.get(dtype)


def _smem(slots, k, r, book_dtype) -> int:  # csrc/selector_fused.cuh smem_bytes
    n, e = 1 + (slots - 1) * k, torch.finfo(book_dtype).bits // 8
    return 4 * ((slots * r + 3) // 4 * 4 + (n * (k + 1) + 3) // 4 * 4) + e * (
        n + slots * k
    ) * (r + 16 // e)


def _cluster_ok(
    slots, k, r, book_dtype, hid_dtype
) -> bool:  # csrc/selector_fused.cuh SelectorFused::run
    return (
        book_dtype == hid_dtype == torch.bfloat16
        and slots <= 8
        and k in (8, 16)
        and r % 16 == 0
    )


def _eligible(sel, candidate_ids, unary_logits, hidden):
    a, b = sel.predecessor_codebook, sel.successor_codebook
    if not (
        candidate_ids.is_cuda
        and candidate_ids.dim() == 3
        and candidate_ids.dtype == torch.int64
    ):
        return None
    bs, slots, k = candidate_ids.shape
    r = int(a.shape[1])
    if not (1 <= k <= MAX_K and 1 <= slots <= MAX_SLOTS and 1 <= r <= MAX_RANK):
        return None
    if (
        a.dtype != b.dtype
        or _code(a.dtype) is None
        or _code(hidden.dtype) is None
        or (a.dtype, hidden.dtype) == (torch.bfloat16, torch.float32)
    ):
        return None
    if unary_logits.dtype != torch.float32 or tuple(unary_logits.shape) != (
        bs,
        slots,
        k,
    ):
        return None
    if tuple(hidden.shape) != (bs, slots, r):
        return None
    if _smem(slots, k, r, a.dtype) > SMEM_CAP and (
        SINGLE_BLOCK or not _cluster_ok(slots, k, r, a.dtype, hidden.dtype)
    ):
        return None
    if (
        (r * a.element_size()) % 16
        or a.data_ptr() % 16
        or b.data_ptr() % 16
        or not (a.is_contiguous() and b.is_contiguous())
    ):
        return None
    if (
        a.dtype == torch.bfloat16 and r % 16
    ):  # the bf16 lattice runs on 16-deep tensor-core steps
        return None
    return bs, slots, k, r


def around_build_lattice(original, sel, *args, **kwargs):
    if args:
        return original(sel, *args, **kwargs)
    cand, unary, hs, anchor = (
        kwargs.get("candidate_ids"),
        kwargs.get("unary_logits"),
        kwargs.get("hidden_states"),
        kwargs.get("anchor_token_ids"),
    )
    if cand is None or unary is None or hs is None or anchor is None:
        return original(sel, *args, **kwargs)
    hidden = sel.hidden_projection(
        hs
    )  # as the stock build_lattice does (selector_proj_overlap may stand in for it)
    shape = _eligible(sel, cand, unary, hidden)
    if shape is None:
        bed.count(NAME, "stock:shape / dtype")
        return original(sel, *args, **kwargs)
    bs, slots, k, _ = shape
    placeholder = torch.empty(
        (bs, slots, k, k), dtype=torch.float32, device=cand.device
    )  # never written or read
    pend = getattr(_tls, "pending", None)
    if pend is None:
        pend = _tls.pending = {}
    pend[id(placeholder)] = (
        placeholder,
        cand.contiguous(),
        unary.contiguous(),
        hidden.contiguous(),
        anchor.reshape(-1).to(torch.int64).contiguous(),
    )
    return placeholder


def around_call(original, sampler, *args, **kwargs):
    """The draft-graph sampler copies sample_path's tokens and q rows into its own buffers (``out``, ``q_out``). While
    it runs, the fused walk writes them there directly, so those two copies alias their sources exactly and launch
    nothing (Tensor.copy_ returns early on an exact alias)."""
    prev = getattr(_tls, "sink", None)
    _tls.sink = sampler
    try:
        return original(sampler, *args, **kwargs)
    finally:
        _tls.sink = prev


def _sink_views(bs, slots, k, device):
    sampler = getattr(_tls, "sink", None)
    out, q_out = getattr(sampler, "out", None), getattr(sampler, "q_out", None)
    if not (isinstance(out, torch.Tensor) and isinstance(q_out, torch.Tensor)):
        return None
    if (
        out.dtype != torch.int64
        or out.dim() != 1
        or not out.is_contiguous()
        or out.numel() < bs * slots
        or out.device != device
    ):
        return None
    if (
        q_out.dtype != torch.float32
        or q_out.dim() != 3
        or tuple(q_out.shape[1:]) != (slots, k)
        or q_out.shape[0] < bs
        or not q_out.is_contiguous()
        or q_out.device != device
    ):
        return None
    return out[: bs * slots].view(bs, slots), q_out[:bs]


def around_sample_path(original, sel, *args, **kwargs):
    scores = kwargs.get("scores")
    pend = getattr(_tls, "pending", None) or {}
    entry = pend.pop(id(scores), None) if scores is not None else None
    if entry is None or entry[0] is not scores or args:
        return original(sel, *args, **kwargs)
    _, cand, unary, hidden, anchor = entry
    if (
        kwargs.get("candidate_ids") is not None
        and kwargs["candidate_ids"].data_ptr() != cand.data_ptr()
    ):
        raise RuntimeError(
            "selector_fused: sample_path got other candidates than its lattice"
        )
    bs, slots, k = cand.shape
    sink = _sink_views(bs, slots, k, cand.device)
    if sink is not None:
        tokens, q_rows = sink
    else:
        tokens = torch.empty((bs, slots), dtype=torch.int64, device=cand.device)
        q_rows = torch.empty((bs, slots, k), dtype=torch.float32, device=cand.device)
    module().run(
        cand,
        unary,
        hidden,
        anchor,
        sel.predecessor_codebook,
        sel.successor_codebook,
        kwargs["uniforms"].contiguous(),
        kwargs["temperatures"].reshape(-1).to(torch.float32).contiguous(),
        kwargs["greedy_mask"].reshape(-1).contiguous().view(torch.uint8),
        tokens,
        q_rows,
        _code(sel.predecessor_codebook.dtype),
        _code(hidden.dtype),
        int(SINGLE_BLOCK),
    )
    bed.count(
        NAME,
        f"fused:bs={bs}:book={str(sel.predecessor_codebook.dtype).split('.')[-1]}"
        f"{':sink' if sink is not None else ''}",
    )  # capture time too
    return tokens, q_rows


def register() -> bool:
    if not enabled():
        return False
    ok, why = pins_match()
    if not ok:
        bed.report(NAME, False, reason=why)
        return False
    from sglang.srt.plugins.hook_registry import HookRegistry, HookType

    HookRegistry.register(T_LATTICE, around_build_lattice, HookType.AROUND)
    HookRegistry.register(T_SAMPLE, around_sample_path, HookType.AROUND)
    HookRegistry.register(T_CALL, around_call, HookType.AROUND)
    bed.report(NAME, True)
    return True
