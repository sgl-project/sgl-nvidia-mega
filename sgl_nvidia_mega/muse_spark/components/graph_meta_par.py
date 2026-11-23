# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Component ``graph_meta_par`` (env ``MUSE_VR_GRAPH_META_PAR=1``, default off).

Before every replay of a decode / target-verify / draft-extend CUDA graph, SGLang's TRT-LLM MHA backend rebuilds the
graph's attention metadata in one Triton launch (``update_trtllm_mha_graph_metadata``): sequence lengths, their prefix
sums, the page table row of every request (``req_to_token[req, p * page_size] // page_size`` for every live page ``p``)
and, with a sliding-window pool, the same row translated through the full-to-SWA mapping. One program walks a whole
request's row, so with few requests in flight the launch takes time proportional to the context length.

This component launches the same kernel body with the page loop spread over a second grid dimension: program
``(row, block)`` writes pages ``[block * PAGE_BLOCK, (block + 1) * PAGE_BLOCK)`` of its row that are live (bounded by the
same on-device ``cdiv(cache_seqlen, page_size)``); the sequence-length stores, the prefix sums and the SWA output
locations run in the ``block == 0`` programs exactly as before. Every output element is computed by the same integer
expression from the same inputs and written once, so every buffer the original writes holds the same bits afterwards,
and the buffers it leaves stale (page-table tail) stay untouched (tests/test_graph_meta_par.py).

Mechanism: an AROUND hook on the backend module's imported name
``sglang.srt.layers.attention.trtllm_mha_backend.update_trtllm_mha_graph_metadata``; the hook takes the same keyword
arguments and launches this module's kernel instead. Calls with arguments this form does not reproduce (anything other
than the keyword set below) run the original. ``MUSE_VR_GRAPH_META_PAR_BLOCK`` (default 128; power of two, 32..512) sets
the pages per program.
"""

# The Triton kernel below is derived from SGLang (https://github.com/sgl-project/sglang), Apache License 2.0; see
# include/LICENSE.sglang. Source: SGLang python/sglang/kernels/ops/attention/trtllm_mha_graph_metadata.py
# (update_trtllm_mha_graph_metadata_kernel, update_trtllm_mha_graph_metadata) at the image revision pinned below.
# Modified by NVIDIA: the per-row page loop is spread over a second grid dimension (one page block per program); the
# per-element arithmetic is unchanged.

from __future__ import annotations

import logging
import os

import triton
import triton.language as tl

from . import bed

# The SGLang names this module reads (test/unit/test_contract.py pins them against the supported SGLang)
SGLANG_READS = {
    "srt/plugins/hook_registry.py": ("HookRegistry", "HookType"),
}


logger = logging.getLogger(__name__)

NAME = "graph_meta_par"  # env MUSE_VR_GRAPH_META_PAR
TARGET = (
    "sglang.srt.layers.attention.trtllm_mha_backend.update_trtllm_mha_graph_metadata"
)
ENV_BLOCK = "MUSE_VR_GRAPH_META_PAR_BLOCK"

PINS = {
    "sglang/kernels/ops/attention/trtllm_mha_graph_metadata.py": "6a5014672e47157e315f3a99b5a34cac37b784f8cf901a47ad99c4e70a9f2321",
    "sglang/srt/layers/attention/trtllm_mha_backend.py": "07223cbf82ab5ac14fcbaa9a47a8351132991016bae66c122bf09762ed8ed646",
}

_KEYS = frozenset(
    (
        "req_pool_indices",
        "seq_lens",
        "req_to_token",
        "cache_seqlens",
        "cu_seqlens_k",
        "page_table",
        "bs",
        "seqlen_offset",
        "max_seq_pages",
        "page_size",
        "swa_mapping",
        "swa_page_table",
        "out_cache_loc",
        "swa_out_cache_loc",
        "cu_seqlens_q",
        "qlens",
        "q_stride",
        "q_mode",
        "skip_page_table",
    )
)


def enabled() -> bool:
    return bed.enabled(NAME)


def pins_match() -> tuple[bool, str]:
    for rel, want in PINS.items():
        if not bed.pinned(rel, want):
            return False, f"{rel} differs from the pinned revision"
    return True, "ok"


def page_block() -> int:
    raw = os.environ.get(ENV_BLOCK, "128").strip() or "128"
    v = int(raw)
    if v < 32 or v > 512 or v & (v - 1):
        raise ValueError(f"{ENV_BLOCK}={raw}: expected a power of two in 32..512")
    return v


@triton.jit
def _meta_par_kernel(
    req_pool_indices_ptr,
    seq_lens_ptr,
    req_to_token_ptr,
    swa_mapping_ptr,
    out_cache_loc_ptr,
    qlens_ptr,
    cache_seqlens_ptr,
    cu_seqlens_k_ptr,
    cu_seqlens_q_ptr,
    page_table_ptr,
    swa_page_table_ptr,
    swa_out_cache_loc_ptr,
    bs,
    seqlen_offset,
    max_seq_pages,
    q_stride,
    num_out_tokens,
    swa_out_len,
    req_to_token_stride,
    page_table_stride,
    swa_page_table_stride,
    PAGE_SIZE: tl.constexpr,
    HAS_SWA: tl.constexpr,
    HAS_SWA_OUT: tl.constexpr,
    Q_MODE: tl.constexpr,
    PAGE_BLOCK: tl.constexpr,
    OUT_BLOCK: tl.constexpr,
    BS_BLOCK: tl.constexpr,
    SKIP_PAGE_TABLE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    blk = tl.program_id(axis=1)

    if pid < bs:
        seqlen = (tl.load(seq_lens_ptr + pid) + seqlen_offset).to(tl.int32)
        if blk == 0:
            tl.store(cache_seqlens_ptr + pid, seqlen)
        if not SKIP_PAGE_TABLE:
            num_live_pages = tl.minimum(tl.cdiv(seqlen, PAGE_SIZE), max_seq_pages)
            if blk * PAGE_BLOCK < num_live_pages:
                req_pool_index = tl.load(req_pool_indices_ptr + pid).to(tl.int64)
                row_in = req_to_token_ptr + req_pool_index * req_to_token_stride
                row_out = page_table_ptr + pid.to(tl.int64) * page_table_stride
                page_idx = blk * PAGE_BLOCK + tl.arange(0, PAGE_BLOCK)
                mask = page_idx < num_live_pages
                token = tl.load(
                    row_in + page_idx.to(tl.int64) * PAGE_SIZE, mask=mask, other=0
                )
                tl.store(row_out + page_idx, token // PAGE_SIZE, mask=mask)
                if HAS_SWA:
                    swa_row_out = (
                        swa_page_table_ptr + pid.to(tl.int64) * swa_page_table_stride
                    )
                    token64 = token.to(tl.int64)
                    swa_token = tl.load(
                        swa_mapping_ptr + token64, mask=mask & (token64 >= 0), other=-1
                    )
                    swa_page = tl.where(swa_token < 0, -1, swa_token // PAGE_SIZE)
                    tl.store(swa_row_out + page_idx, swa_page.to(tl.int32), mask=mask)
    elif pid == bs:
        if blk == 0:
            offs = tl.arange(0, BS_BLOCK)
            mask = offs < bs
            seqlens = (tl.load(seq_lens_ptr + offs, mask=mask, other=0)).to(tl.int32)
            seqlens = tl.where(mask, seqlens + seqlen_offset, 0)
            tl.store(cu_seqlens_k_ptr + 1 + offs, tl.cumsum(seqlens, axis=0), mask=mask)
            if Q_MODE == 1:
                qlens = tl.load(qlens_ptr + offs, mask=mask, other=0).to(tl.int32)
                qlens = tl.where(mask, qlens, 0)
                tl.store(
                    cu_seqlens_q_ptr + 1 + offs, tl.cumsum(qlens, axis=0), mask=mask
                )
            if Q_MODE == 2:
                tl.store(
                    cu_seqlens_q_ptr + 1 + offs,
                    ((offs + 1) * q_stride).to(tl.int32),
                    mask=mask,
                )
    else:
        if HAS_SWA_OUT:
            if blk == 0:
                out_idx = (pid - bs - 1) * OUT_BLOCK + tl.arange(0, OUT_BLOCK)
                in_range = out_idx < swa_out_len
                is_real = in_range & (out_idx < num_out_tokens)
                loc = tl.load(out_cache_loc_ptr + out_idx, mask=is_real, other=0)
                translated = tl.load(
                    swa_mapping_ptr + loc, mask=is_real & (loc >= 0), other=0
                )
                translated = tl.where(is_real & (loc < 0), -1, translated)
                tl.store(swa_out_cache_loc_ptr + out_idx, translated, mask=in_range)


def launch(
    *,
    req_pool_indices,
    seq_lens,
    req_to_token,
    cache_seqlens,
    cu_seqlens_k,
    page_table,
    bs: int,
    seqlen_offset: int,
    max_seq_pages: int,
    page_size: int,
    swa_mapping=None,
    swa_page_table=None,
    out_cache_loc=None,
    swa_out_cache_loc=None,
    cu_seqlens_q=None,
    qlens=None,
    q_stride: int = 0,
    q_mode: int = 0,
    skip_page_table: bool = False,
    block: int | None = None,
):
    """The same contract as SGLang's ``update_trtllm_mha_graph_metadata`` (keyword arguments identical); the page-table
    walk runs over ``(bs, cdiv(max_seq_pages, block))`` programs."""
    if bs == 0:
        return
    out_block = (
        512  # the original's tile width for the SWA output locations (its PAGE_BLOCK)
    )
    pb = block or page_block()
    if skip_page_table:
        page_table = cache_seqlens
        swa_page_table = None
    has_swa = swa_page_table is not None
    has_swa_out = swa_out_cache_loc is not None
    swa_out_len = swa_out_cache_loc.shape[0] if has_swa_out else 0
    if has_swa_out and out_cache_loc is not None:
        num_out_tokens = min(swa_out_len, out_cache_loc.shape[0])
    else:
        num_out_tokens = 0
    if num_out_tokens == 0:
        out_cache_loc = swa_out_cache_loc
    grid_extra = triton.cdiv(swa_out_len, out_block) if has_swa_out else 0
    nblk = 1 if skip_page_table else max(1, triton.cdiv(max_seq_pages, pb))
    grid = (bs + 1 + grid_extra, nblk)
    _meta_par_kernel[grid](
        req_pool_indices,
        seq_lens,
        req_to_token,
        swa_mapping,
        out_cache_loc,
        qlens,
        cache_seqlens,
        cu_seqlens_k,
        cu_seqlens_q,
        page_table,
        swa_page_table,
        swa_out_cache_loc,
        bs,
        seqlen_offset,
        max_seq_pages,
        q_stride,
        num_out_tokens,
        swa_out_len,
        req_to_token.stride(0),
        page_table.stride(0),
        swa_page_table.stride(0) if has_swa else 0,
        PAGE_SIZE=page_size,
        HAS_SWA=has_swa,
        HAS_SWA_OUT=has_swa_out,
        Q_MODE=q_mode,
        PAGE_BLOCK=pb,
        OUT_BLOCK=out_block,
        BS_BLOCK=triton.next_power_of_2(max(bs, 1)),
        SKIP_PAGE_TABLE=1 if skip_page_table else 0,
    )


def around_update(original, *args, **kwargs):
    if args or not set(kwargs) <= _KEYS:
        bed.count(NAME, "stock:call form")
        return original(*args, **kwargs)
    launch(**kwargs)
    bed.count(NAME, "par:skip_pt" if kwargs.get("skip_page_table") else "par")
    return None


def register() -> bool:
    if not enabled():
        return False
    ok, why = pins_match()
    if not ok:
        bed.report(NAME, False, reason=why)
        return False
    try:
        block = page_block()
    except ValueError as e:
        bed.report(NAME, False, reason=str(e))
        return False
    from sglang.srt.plugins.hook_registry import HookRegistry, HookType

    HookRegistry.register(TARGET, around_update, HookType.AROUND)
    bed.report(NAME, True, block=block)
    return True
