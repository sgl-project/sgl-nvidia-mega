# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Attention for the DFlash draft block (``csrc/draft_attn.cuh``): kQ query tokens per request at the end of the
sequence, no causal mask, optional window, grouped query heads, head_dim 128, paged bf16 KV cache."""

from __future__ import annotations

import functools
import os
from pathlib import Path

import torch

# The SGLang names this module reads (test/unit/test_contract.py pins them against the supported SGLang)
SGLANG_READS = {
    "kernels/jit/utils/__init__.py": (
        "is_arch_support_pdl",
        "load_jit",
        "make_cpp_args",
    ),
}


_CSRC = Path(__file__).resolve().parents[1] / "csrc" / "draft_attn.cuh"

SPLITS = 16  # CTAs per (request, KV head); tunable (tests/test_draft_attn.py --bench)


@functools.lru_cache(maxsize=None)
def use_pdl() -> bool:
    from sglang.kernels.jit.utils import is_arch_support_pdl

    return bool(is_arch_support_pdl())


@functools.lru_cache(maxsize=None)
def early_trigger() -> bool:
    """``MUSE_VR_DRAFT_ATTN_EARLY_TRIGGER=1`` (default off): signal the dependent launch right after the kernel's own grid
    dependency wait instead of at its end (same outputs, bit for bit)."""
    return os.environ.get("MUSE_VR_DRAFT_ATTN_EARLY_TRIGGER", "0").strip() not in (
        "",
        "0",
    )


@functools.lru_cache(maxsize=None)
def pre_kv() -> bool:
    """``MUSE_VR_DRAFT_ATTN_PRE_KV=1`` (default off): read the sequence lengths and the page table and issue the first load
    round's copies of the keys before the block before the kernel's grid dependency wait (same outputs, bit for bit)."""
    return os.environ.get("MUSE_VR_DRAFT_ATTN_PRE_KV", "0").strip() not in ("", "0")


@functools.lru_cache(maxsize=None)
def row_split() -> int:
    """``MUSE_VR_DRAFT_ATTN_ROW_SPLIT=2`` (default 1): split each (request, KV head)'s query rows over two clusters of
    4-warp CTAs (same key partition and combine order per row: same outputs, bit for bit)."""
    v = int(os.environ.get("MUSE_VR_DRAFT_ATTN_ROW_SPLIT", "1").strip() or "1")
    if v not in (1, 2):
        raise ValueError(f"MUSE_VR_DRAFT_ATTN_ROW_SPLIT={v}: expected 1 or 2")
    return v


@functools.lru_cache(maxsize=None)
def key_parts() -> int:
    """``MUSE_VR_DRAFT_ATTN_KEY_PARTS=4`` (default 2): four warps per row slice share a load round's keys (shorter per-warp
    chains; the partials are combined in part order - a different softmax / output summation order)."""
    v = int(os.environ.get("MUSE_VR_DRAFT_ATTN_KEY_PARTS", "2").strip() or "2")
    if v not in (2, 4):
        raise ValueError(f"MUSE_VR_DRAFT_ATTN_KEY_PARTS={v}: expected 2 or 4")
    return v


@functools.lru_cache(maxsize=None)
def module(
    q_tokens: int,
    group: int,
    splits: int,
    pdl: bool,
    early: bool = False,
    pre: bool = False,
    rows: int = 1,
    parts: int = 2,
):
    from sglang.kernels.jit.utils import load_jit, make_cpp_args

    if parts != 2:
        args = make_cpp_args(
            q_tokens, group, splits, pdl, bool(early), bool(pre), int(rows), int(parts)
        )
    elif rows != 1:
        args = make_cpp_args(
            q_tokens, group, splits, pdl, bool(early), bool(pre), int(rows)
        )
    elif pre:
        args = make_cpp_args(q_tokens, group, splits, pdl, bool(early), True)
    elif early:
        args = make_cpp_args(q_tokens, group, splits, pdl, True)
    else:
        args = make_cpp_args(q_tokens, group, splits, pdl)
    return load_jit(
        "muse_vr_draft_attn",
        *args,
        cuda_files=[str(_CSRC)],
        cuda_wrappers=[("run", f"muse_vr::draft_attn::DraftAttn<{args}>::run")],
    )


def attention(
    q,
    k_cache,
    v_cache,
    page_table,
    seqlens,
    scale,
    window=(-1, -1),
    q_tokens=8,
    out=None,
    splits=None,
    pdl=None,
    early=None,
    pre=None,
    rows=None,
    parts=None,
):
    """q [T, Hq, 128] (T = requests * q_tokens), caches [pages, page, Hkv, 128] -> out [T, Hq, 128]."""
    if out is None:
        out = torch.empty(q.shape, dtype=q.dtype, device=q.device)
    group = q.shape[1] // k_cache.shape[2]
    mod = module(
        q_tokens,
        group,
        SPLITS if splits is None else splits,
        use_pdl() if pdl is None else pdl,
        early_trigger() if early is None else bool(early),
        pre_kv() if pre is None else bool(pre),
        row_split() if rows is None else int(rows),
        key_parts() if parts is None else int(parts),
    )
    mod.run(
        q,
        k_cache,
        v_cache,
        page_table,
        seqlens,
        out,
        int(window[0]),
        int(window[1]),
        float(scale),
    )
    return out
