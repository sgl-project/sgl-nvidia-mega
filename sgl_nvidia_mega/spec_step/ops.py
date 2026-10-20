# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Kernels of the speculative step fusions: the one-launch relay scatter, the topk-1
draft-extend tail, and the chain verify sampling without a draft_probs buffer.

The sampling kernel is SGLang's JIT kernel with a target-only chain form and a
two-pass chunked form (``csrc/``); the other two are Triton.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Optional

import torch
import triton
import triton.language as tl

# SGLang's split-row argmax, whose partial pass and pair reduction the tail reuses
from sglang.kernels.ops.speculative.row_argmax import (
    _argmax_pair,
    _medium_argmax_partial_kernel,
)

# The SGLang names this module reads (test/unit/test_contract.py pins them)
SGLANG_READS = {
    "kernels/ops/speculative/row_argmax.py": (
        "_argmax_pair",
        "_medium_argmax_partial_kernel",
    ),
    "kernels/jit/utils/compile/loader.py": ("load_jit", "extra_include_paths"),
}

CSRC = Path(__file__).resolve().parent / "csrc"


@triton.jit
def _scatter_spec_extras_kernel(
    indices,
    sources,
    destinations,
    INDEX_STRIDE: tl.constexpr,
    SOURCE_STRIDES: tl.constexpr,
    WIDTHS: tl.constexpr,
    SLOTS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    dst = tl.load(indices + row * INDEX_STRIDE).to(tl.int64)
    dst = tl.where(dst < 0, dst + SLOTS, dst)
    col = tl.program_id(1).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    for i in tl.static_range(len(WIDTHS)):
        value = tl.load(sources[i] + row * SOURCE_STRIDES[i] + col, col < WIDTHS[i])
        tl.store(destinations[i] + dst * WIDTHS[i] + col, value, col < WIDTHS[i])


def scatter_spec_extras(indices, pairs):
    """``dst[indices] = src`` for every (dst, src) pair in one launch; the
    destinations are row-major relay buffers with as many rows as each other."""
    if indices.numel() == 0 or not pairs:
        return
    sources = tuple(src.reshape(src.shape[0], -1) for _, src in pairs)
    sources = tuple(src if src.stride(1) == 1 else src.contiguous() for src in sources)
    widths = tuple(src.shape[1] for src in sources)
    block = min(1024, triton.next_power_of_2(max(widths)))
    _scatter_spec_extras_kernel[(indices.numel(), triton.cdiv(max(widths), block))](
        indices,
        sources,
        tuple(dst for dst, _ in pairs),
        indices.stride(0),
        tuple(src.stride(0) for src in sources),
        widths,
        pairs[0][0].shape[0],
        BLOCK=block,
    )


@triton.jit
def _draft_tail_final_kernel(
    PV,
    PI,
    OUT,
    PROBS,
    SEED_SRC,
    SEED_INDEX,
    SEED_OUT,
    SPLITS: tl.constexpr,
    BLOCK: tl.constexpr,
    SEED_WIDTH: tl.constexpr,
    SEED_SRC_STRIDE: tl.constexpr,
    SEED_INDEX_STRIDE: tl.constexpr,
    SEED_BLOCK: tl.constexpr,
    HAS_SEED: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    part = tl.arange(0, BLOCK)
    v = tl.load(PV + row * SPLITS + part, part < SPLITS, float("-inf"))
    i = tl.load(PI + row * SPLITS + part, part < SPLITS, 0x7FFFFFFF)
    _, index = tl.reduce((v, i), 0, _argmax_pair)
    tl.store(OUT + row, index.to(tl.int64))
    tl.store(PROBS + row, 1.0)
    if HAS_SEED:
        # The row's seed: SEED_SRC[SEED_INDEX[row]], as an indexed gather.
        src = tl.load(SEED_INDEX + row * SEED_INDEX_STRIDE).to(tl.int64)
        for start in tl.static_range(0, SEED_WIDTH, SEED_BLOCK):
            cols = start + tl.arange(0, SEED_BLOCK)
            m = cols < SEED_WIDTH
            seed = tl.load(SEED_SRC + src * SEED_SRC_STRIDE + cols, m)
            tl.store(SEED_OUT + row * SEED_WIDTH + cols, seed, m)


def draft_extend_tail(
    logits: torch.Tensor,
    seed_src: Optional[torch.Tensor] = None,
    seed_index: Optional[torch.Tensor] = None,
):
    """The topk-1 draft-extend tail in two launches.

    Returns ``(topk_p, topk_index, seed)`` equal to
    ``torch.ones_like(i, dtype=torch.float32)``, ``i = logits.argmax(-1,
    keepdim=True)`` (NaN first, lowest index on ties) and ``seed_src[seed_index]``
    (None without ``seed_src``). ``logits``: FP32 rows with a unit column
    stride; ``seed_src``: 2-D with a unit column stride; ``seed_index``: 1-D
    non-negative row indices, one per logits row.
    """
    assert logits.ndim == 2 and logits.dtype == torch.float32
    assert logits.stride(1) == 1
    rows, n = logits.shape
    assert n > 0
    has_seed = seed_src is not None
    index = torch.empty((rows, 1), dtype=torch.int64, device=logits.device)
    probs = torch.empty((rows, 1), dtype=torch.float32, device=logits.device)
    seed = None
    if has_seed:
        assert seed_src.ndim == 2 and seed_src.stride(1) == 1
        assert seed_index is not None and seed_index.shape == (rows,)
        seed = torch.empty(
            (rows, seed_src.shape[1]), dtype=seed_src.dtype, device=seed_src.device
        )
    if rows:
        block = 2048 if rows <= 64 else 4096 if rows <= 256 else 8192
        splits = triton.cdiv(n, block)
        pv = torch.empty((rows, splits), dtype=torch.float32, device=logits.device)
        pi = torch.empty((rows, splits), dtype=torch.int32, device=logits.device)
        _medium_argmax_partial_kernel[(rows, splits)](
            logits, pv, pi, n, logits.stride(0), splits, block, num_warps=4
        )
        seed_width = seed_src.shape[1] if has_seed else 1
        _draft_tail_final_kernel[(rows,)](
            pv,
            pi,
            index,
            probs,
            seed_src if has_seed else pv,
            seed_index if has_seed else pi,
            seed if has_seed else pv,
            splits,
            triton.next_power_of_2(splits),
            seed_width,
            seed_src.stride(0) if has_seed else 0,
            seed_index.stride(0) if has_seed else 0,
            min(1024, triton.next_power_of_2(seed_width)),
            has_seed,
            num_warps=4,
        )
    return probs, index, seed


def _load(name: str, wrapper: str, *marker):
    from sglang.kernels.jit.utils import load_jit

    return load_jit(
        "sgl_nvidia_mega_spec_sampling",
        name,
        *marker,
        cuda_files=[str(CSRC / "sampling.cuh")],
        cuda_wrappers=[("sample", wrapper)],
        extra_include_paths=[str(CSRC / "include")],
        extra_dependencies=["flashinfer"],
    )


_modules: dict = {}


def _module(vec_size: int, deterministic: bool, *, no_draft_probs=False, chunked=False):
    key = (vec_size, deterministic, no_draft_probs, chunked)
    if key not in _modules:
        det = "true" if deterministic else "false"
        if chunked:
            wrapper = f"chain_speculative_sampling_target_only_two_pass<{vec_size}, {det}>"
            _modules[key] = _load("chunked", wrapper, vec_size, deterministic)
        else:
            ndp = "true" if no_draft_probs else "false"
            wrapper = f"tree_speculative_sampling_target_only<{vec_size}, {det}, {ndp}>"
            _modules[key] = _load("tree", wrapper, vec_size, deterministic, ndp)
    return _modules[key]


# Vocabulary chunk of the sampling kernels: 1024 threads x vec_size elements.
_CHUNK_THREADS = 1024


def tree_speculative_sampling_target_only(
    predicts: torch.Tensor,  # mutable
    accept_index: torch.Tensor,  # mutable
    accept_token_num: torch.Tensor,  # mutable
    candidates: torch.Tensor,
    retrive_index: torch.Tensor,
    retrive_next_token: torch.Tensor,
    retrive_next_sibling: torch.Tensor,
    uniform_samples: torch.Tensor,
    uniform_samples_for_final_sampling: torch.Tensor,
    target_probs: torch.Tensor,
    draft_probs: Optional[torch.Tensor],
    threshold_single: float = 1.0,
    threshold_acc: float = 1.0,
    deterministic: bool = True,
    chunked: bool = False,
) -> None:
    """SGLang's ``tree_speculative_sampling_target_only``, plus ``draft_probs=None``
    (chains only: one candidate per tree level), which samples as a zero-filled
    ``draft_probs`` would without the buffer, and ``chunked`` (with
    ``draft_probs=None`` only): the same sampling as two launches, the vocabulary
    passes split over one CTA per chunk. The outputs are the same bits."""
    vec_size = math.gcd(4, target_probs.shape[-1])
    no_draft_probs = draft_probs is None
    if chunked and no_draft_probs:
        bs = target_probs.shape[0]
        chunk = _CHUNK_THREADS * vec_size
        num_chunks = (target_probs.shape[-1] + chunk - 1) // chunk
        chunk_sums = target_probs.new_empty((2, bs, num_chunks))
        chunk_last_valid = accept_token_num.new_empty((bs, num_chunks))
        _module(vec_size, deterministic, chunked=True).sample(
            predicts,
            accept_index,
            accept_token_num,
            candidates,
            retrive_index,
            retrive_next_token,
            retrive_next_sibling,
            uniform_samples,
            uniform_samples_for_final_sampling,
            target_probs,
            chunk_sums[0],
            chunk_sums[1],
            chunk_last_valid,
            threshold_single,
            threshold_acc,
        )
        return
    _module(vec_size, deterministic, no_draft_probs=no_draft_probs).sample(
        predicts,
        accept_index,
        accept_token_num,
        candidates,
        retrive_index,
        retrive_next_token,
        retrive_next_sibling,
        uniform_samples,
        uniform_samples_for_final_sampling,
        target_probs,
        target_probs if no_draft_probs else draft_probs,
        threshold_single,
        threshold_acc,
    )
