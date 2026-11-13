# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""MXFP8 operands and GEMM for the prefill projections (helper module, no component of its own).

MXFP8 (OCP MX): E4M3 values with one power-of-two (E8M0) scale per block of 32 consecutive elements along K, for weights
and activations. Quantization and GEMM are FlashInfer's (``flashinfer.mxfp8_quantize`` / ``flashinfer.mm_mxfp8``), the
functions SGLang's own MXFP8 dense linear method calls; scales are kept in the 128 x 4 swizzled layout those GEMMs read.

Layout facts used by the collectives (FlashInfer's swizzled layout ``SfLayout.layout_128x4``, per its scale index
function; checked by tests/dist_fp8_prefill.py LAYOUT): the scales of an [rows, K] operand are stored as tiles of 128
rows x 4 scale columns, 512 bytes each, row tiles outer and column tiles inner. So the scales of any range of whole
128-row tiles are one contiguous byte range, and the scales of the columns of one K slice (a multiple of 4 x 32
elements wide) are, per row tile, one contiguous run.
"""

from __future__ import annotations

import functools

import torch

FP8 = torch.float8_e4m3fn
SF_VEC = 32
SF_TILE_ROWS = 128
SF_TILE_COLS = 4
SF_TILE_BYTES = SF_TILE_ROWS * SF_TILE_COLS


def sf_bytes(rows: int, k: int) -> int:
    """Bytes of the swizzled scales of an [rows, k] operand (rows padded to 128, scale columns to 4)."""
    rt = -(-rows // SF_TILE_ROWS)
    ct = -(-(k // SF_VEC) // SF_TILE_COLS)
    return rt * ct * SF_TILE_BYTES


@functools.lru_cache(maxsize=None)
def _fi():
    import flashinfer

    return flashinfer


@functools.lru_cache(maxsize=None)
def fq_mode() -> int:
    """``MUSE_VR_FP8_PREFILL_FQ``: 0 (default) = FlashInfer's quantizer; 1 = ``mxfp8_fq`` (same bytes) for every operand it
    takes; 2 = 1 plus one grouped launch per layer for the just-in-time weights (``fp8_prefill``)."""
    import os

    v = os.environ.get("MUSE_VR_FP8_PREFILL_FQ", "").strip()
    return int(v) if v else 0


def quantize(x: torch.Tensor):
    """[rows, K] bf16 -> (E4M3 [rows, K], swizzled E8M0 scales 1-D uint8)."""
    if fq_mode():
        from . import mxfp8_fq

        if mxfp8_fq.takes(x):
            return mxfp8_fq.quantize(x)
    q, s = _fi().mxfp8_quantize(x, is_sf_swizzled_layout=True)
    return q, s.view(torch.uint8).reshape(-1)


def quantize_into(x: torch.Tensor, q_out: torch.Tensor, s_out: torch.Tensor) -> None:
    """``quantize`` written into caller-owned buffers (the same values; a copy until a fused producer writes them)."""
    q, s = quantize(x)
    q_out.copy_(q)
    s_out[: s.numel()].copy_(s)


_PINNED: dict = {"table": None, "runner": None, "factory": None, "bands": False}


def set_runner_factory(spec: str) -> None:
    """``spec`` = ``module:function`` of the CuTe-DSL MXFP8 runner factory that takes a configuration per call (the
    target-specific runner FlashInfer selects for this GPU), from the deployment's private configuration."""
    import importlib

    mod, fn = spec.split(":", 1)
    _PINNED["factory"] = getattr(importlib.import_module(mod), fn)


def load_pinned(path: str) -> int:
    """Pinned configurations for the CuTe-DSL backend: JSON {"<N>x<K>": {"<M key>": "<tactic repr>"}} (chosen per shape and
    row key by a sweep in which every listed configuration gives the default configuration's bits).

    Two lookup forms. Default: the keys are power-of-two row buckets and a call with M rows uses the key equal to the next
    power of two >= M (capped at the largest key; a missing bucket runs the default configuration). With a top-level entry
    ``"__lookup__": "bands"`` the keys are inclusive upper row bounds and a call uses the smallest key >= M (the largest key
    above all keys). A table without that entry is read exactly as before. Returns the number of entries."""
    import ast
    import json

    raw = json.load(open(path))
    mode = raw.pop("__lookup__", "pow2")
    if mode not in ("pow2", "bands"):
        raise ValueError(f"pinned table: unknown __lookup__ {mode!r}")
    table = {}
    for shape, cells in raw.items():
        n, k = (int(x) for x in shape.split("x"))
        table[(n, k)] = {int(m): ast.literal_eval(t) for m, t in cells.items()}
        if not table[(n, k)] or min(table[(n, k)]) < 1:
            raise ValueError(f"pinned table: shape {shape} needs positive row keys")
    _PINNED["table"] = table
    _PINNED["bands"] = mode == "bands"
    return sum(len(v) for v in table.values())


def pinned_lookup() -> str:
    """The loaded table's lookup form ("pow2" or "bands"); "none" without a table."""
    if not _PINNED["table"]:
        return "none"
    return "bands" if _PINNED["bands"] else "pow2"


def pinned_row_keys(n: int, k: int) -> list:
    """Sorted row keys of the loaded table for shape (n, k) (empty without one)."""
    table = _PINNED["table"]
    return sorted(table[(n, k)]) if table and (n, k) in table else []


def _pinned_tactic(m: int, n: int, k: int):
    table = _PINNED["table"]
    if not table or (n, k) not in table:
        return None
    cells = table[(n, k)]
    if _PINNED["bands"]:
        keys = [key for key in cells if key >= m]
        return cells[min(keys) if keys else max(cells)]
    b = 1
    while b < m:
        b <<= 1
    return cells.get(min(b, max(cells)))


def gemm(
    q_a: torch.Tensor,
    s_a: torch.Tensor,
    q_w: torch.Tensor,
    s_w: torch.Tensor,
    backend: str,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    """BF16 [rows, N] = MXFP8 [rows, K] x MXFP8 [N, K]^T (``q_w`` row-major [N, K]; scales swizzled). With a pinned table
    and a runner factory set (``set_runner_factory``) and the CuTe-DSL backend, the call runs that runner at the table's
    configuration for its shape and row key (same bits as the default configuration, by the table's construction)."""
    from flashinfer.autotuner import autotune

    m, n, k = q_a.shape[0], q_w.shape[0], q_w.shape[1]
    tac = (
        _pinned_tactic(m, n, k)
        if backend == "cute-dsl" and _PINNED["factory"] is not None
        else None
    )
    if tac is not None:
        if _PINNED["runner"] is None:
            _PINNED["runner"] = _PINNED["factory"](torch.bfloat16)
        o = (
            out
            if out is not None
            else torch.empty((m, n), dtype=torch.bfloat16, device=q_a.device)
        )
        _PINNED["runner"].forward(
            [q_a, q_w.t(), s_a, s_w, torch.bfloat16, o, None], tactic=tac
        )
        return o
    with autotune(False):
        return _fi().mm_mxfp8(
            q_a, q_w.t(), s_a, s_w, out=out, out_dtype=torch.bfloat16, backend=backend
        )
