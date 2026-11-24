# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Attention for the DFlash draft block (``csrc/draft_attn.cuh``): kQ query tokens per request at the end of the
sequence, no causal mask, optional window, grouped query heads, head_dim 128, paged bf16 KV cache."""

from __future__ import annotations

import functools
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
def module(q_tokens: int, group: int, splits: int, pdl: bool):
    from sglang.kernels.jit.utils import load_jit, make_cpp_args

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
