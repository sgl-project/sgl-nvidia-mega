# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

from __future__ import annotations

import logging
from types import SimpleNamespace

import torch

logger = logging.getLogger(__name__)

SGLANG_READS = {
    "kernels/plugin_kernels/nvidia/spec.py": (
        "DraftAttentionView",
        "layer",
        "q",
        "k_pool",
        "v_pool",
        "k_scale",
        "v_scale",
        "page_size",
        "page_table",
        "cache_seqlens",
        "q_len",
        "out_dtype",
        "step",
    ),
    "srt/layers/radix_attention.py": (
        "layer_id",
        "tp_q_head_num",
        "tp_k_head_num",
        "head_dim",
        "scaling",
        "sliding_window_size",
    ),
}

SLOT = "draft_cp"
PRECISE = True
PRECISE_MAX_REQUESTS = 64

_LOGGED: set = set()


def stock_cutover(requests: int, q_len: int) -> bool:
    if requests <= PRECISE_MAX_REQUESTS:
        return False
    if "cutover" not in _LOGGED:
        _LOGGED.add("cutover")
        logger.warning(
            "NVIDIA megakernel: Qwen3.5 draft attention above %d requests runs the stock "
            "kernel (first: %d requests x %d rows); CP-lite keeps the calls of %d requests "
            "or fewer",
            PRECISE_MAX_REQUESTS,
            requests,
            q_len,
            PRECISE_MAX_REQUESTS,
        )
    return True


def reasons(layer, k_pool, rows: int, requests: int, q_len: int, page_size: int) -> str:
    from sgl_nvidia_mega.qwen3_5.attention import split_kv
    from sgl_nvidia_mega.qwen3_5.sm103.attention.q35_cp_lite import _MAX_ROWS, HEAD_DIM

    cp = split_kv.cp_lite()
    if cp is None:
        return "the plugin did not build CP-lite"
    if q_len < 1 or requests < 1:
        return f"{requests} requests of {q_len} rows"
    if requests < cp.min_bs:
        return f"{requests} request(s) (CP-lite serves {cp.min_bs} or more)"
    if stock_cutover(requests, q_len):
        return (
            f"{requests} requests (the draft's CP-lite serves {PRECISE_MAX_REQUESTS} or "
            "fewer; stock attention above)"
        )
    if rows < requests * q_len:
        return f"{rows} rows for {requests} requests of {q_len}"
    if requests * q_len > _MAX_ROWS:
        return f"{requests * q_len} rows (CP-lite serves {_MAX_ROWS})"
    if page_size != split_kv.PAGE_SIZE:
        return f"pages of {page_size} slots (CP-lite: {split_kv.PAGE_SIZE})"
    if (
        layer.tp_k_head_num != 1
        or layer.tp_q_head_num != cp.hq
        or layer.head_dim != HEAD_DIM
    ):
        return (
            f"{layer.tp_q_head_num} query / {layer.tp_k_head_num} KV heads of "
            f"{layer.head_dim} a rank (CP-lite: {cp.hq} / 1 of {HEAD_DIM})"
        )
    if layer.sliding_window_size not in (None, -1):
        return f"a sliding window of {layer.sliding_window_size}"
    if (
        k_pool.dtype != torch.float8_e4m3fn
        or k_pool.dim() != 3
        or tuple(k_pool.shape[1:]) != (1, HEAD_DIM)
        or not k_pool.is_contiguous()
    ):
        return (
            f"a KV pool of {k_pool.dtype} {tuple(k_pool.shape)} "
            "(NHD FP8 rows of one head)"
        )
    return ""


def attend(view) -> torch.Tensor | None:
    from sgl_nvidia_mega.qwen3_5.attention import split_kv

    layer = view.layer
    rows = int(view.q.shape[0])
    requests = int(view.cache_seqlens.shape[0])
    why = (
        f"{view.out_dtype} output (CP-lite writes bf16)"
        if view.out_dtype != torch.bfloat16
        else reasons(layer, view.k_pool, rows, requests, view.q_len, view.page_size)
    )
    o = None
    if not why:
        kv = SimpleNamespace(
            layer_ids=[layer.layer_id],
            k=[view.k_pool],
            v=[view.v_pool],
            k_scale=[float(view.k_scale)],
            v_scale=[float(view.v_scale)],
        )
        meta = SimpleNamespace(
            page_table=view.page_table, cache_seqlens=view.cache_seqlens
        )
        o = split_kv.cp_core(
            layer,
            view.q.reshape(rows, -1),
            kv,
            meta,
            view.step,
            view.q_len,
            min_bs=None,
            pad_rows=True,
            precise=PRECISE,
        )
        if o is None:
            why = "CP-lite declined the call"
    key = (requests, view.q_len, rows, why)
    if key not in _LOGGED:
        _LOGGED.add(key)
        logger.info(
            "NVIDIA megakernel: Qwen3.5 slot %s: draft attention of %d requests x %d rows "
            "(%d rows) %s",
            SLOT,
            requests,
            view.q_len,
            rows,
            f"runs stock ({why})" if why else "runs CP-lite's precise call",
        )
    return o
