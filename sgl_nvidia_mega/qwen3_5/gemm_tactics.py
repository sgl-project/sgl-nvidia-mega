# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""The cuBLASLt algorithm of the TP8 shard's FP8 projection GEMM per verify width."""

from __future__ import annotations

import ctypes
import logging
from functools import cache
from typing import NamedTuple, Optional

import torch

from sgl_nvidia_mega.qwen3_5 import topology

logger = logging.getLogger(__name__)

SGLANG_READS = {
    "srt/layers/quantization/fp8_utils.py": ("flashinfer_bmm_fp8",),
}
FLASHINFER_READS = {
    "flashinfer/gemm/gemm_base.py": (
        "get_gemm_module",
        "cublas_fp8_gemm_runner",
        "_get_algos",
        "forward(tactic)",
        "DEFAULT_WORKSPACE_SIZE",
    ),
    "flashinfer/utils.py": ("_get_cache_buf",),
}
# What a FlashInfer without one of those names raises on the call that reads it
_API_ERRORS = (AttributeError, ImportError, TypeError)

OPTION = "gemm_tactics"
SHAPE_NK = (2560, 4096)  # (N, K) of the pinned GEMM
CUBLASLT_VERSION = 130101  # the cublasLtGetVersion() the table's configurations are for

_CONFIG_ATTRS = tuple(range(9))
_ALGO_BYTES = 64  # sizeof(cublasLtMatmulAlgo_t)


class Config(NamedTuple):
    """A cuBLASLt matmul algorithm's configuration, in attribute order."""

    algo: int
    tile: int
    splitk: int
    reduction: int
    swizzle: int
    custom: int
    stages: int
    inner: int
    cluster: int


class Entry(NamedTuple):
    """Rows ``lo..hi`` (inclusive) run ``config``; None runs FlashInfer's heuristic."""

    lo: int
    hi: int
    config: Optional[Config]
    name: str


# The table's algorithms, named by CTA tile and cluster shape.
_64x8_4x1 = Config(66, 10, 1, 0, 0, 1, 36, 0, 4)
_64x24_4x2 = Config(66, 313, 1, 0, 0, 3, 36, 0, 7)
_96x64_4x2 = Config(66, 27, 1, 0, 0, 2, 36, 0, 7)
_64x48_2x2 = Config(66, 315, 1, 0, 0, 3, 36, 0, 6)
_64x64_2x2 = Config(66, 15, 1, 0, 0, 3, 36, 0, 6)
_64x72_4x2 = Config(66, 317, 1, 0, 0, 3, 36, 0, 7)
_64x80_2x2 = Config(66, 318, 1, 0, 0, 3, 36, 0, 6)
_128x40_2x4 = Config(66, 396, 1, 0, 0, 3, 36, 0, 9)
_64x56_4x1 = Config(66, 316, 1, 0, 0, 1, 36, 0, 4)
_64x64_2x1 = Config(66, 15, 1, 0, 0, 1, 36, 0, 3)
_128x48_2x4 = Config(66, 397, 1, 0, 0, 3, 36, 0, 9)
_128x56_2x4 = Config(66, 398, 1, 0, 0, 3, 36, 0, 9)
_80x128_4x1 = Config(66, 135, 1, 0, 0, 0, 36, 0, 4)
_112x128_4x2 = Config(66, 165, 1, 0, 0, 2, 36, 0, 7)

ENTRIES: tuple[Entry, ...] = (
    Entry(7, 7, _64x8_4x1, "64x8 4x1"),
    Entry(14, 14, None, "heuristic"),
    Entry(21, 21, None, "heuristic"),
    Entry(28, 28, None, "heuristic"),
    Entry(33, 48, _64x24_4x2, "64x24 4x2"),
    Entry(49, 69, None, "heuristic"),
    Entry(70, 83, _96x64_4x2, "96x64 4x2 2-CTA"),
    Entry(84, 97, _64x48_2x2, "64x48 2x2 2-CTA"),
    Entry(98, 128, _64x64_2x2, "64x64 2x2 2-CTA"),
    Entry(129, 140, _64x72_4x2, "64x72 4x2"),
    Entry(141, 147, _64x80_2x2, "64x80 2x2 2-CTA"),
    Entry(148, 154, _128x40_2x4, "128x40 2x4"),
    Entry(155, 161, _64x56_4x1, "64x56 4x1"),
    Entry(162, 168, _64x64_2x1, "64x64 2x1 2-CTA"),
    Entry(169, 175, _128x48_2x4, "128x48 2x4 2-CTA"),
    Entry(176, 182, _64x64_2x1, "64x64 2x1 2-CTA"),
    Entry(183, 189, _128x48_2x4, "128x48 2x4 2-CTA"),
    Entry(190, 203, None, "heuristic"),
    Entry(204, 217, _128x56_2x4, "128x56 2x4"),
    Entry(218, 287, None, "heuristic"),
    Entry(288, 314, _80x128_4x1, "80x128 4x1"),
    Entry(315, 315, None, "heuristic"),
    Entry(316, 336, _80x128_4x1, "80x128 4x1"),
    Entry(337, 441, None, "heuristic"),
    Entry(442, 448, _112x128_4x2, "112x128 4x2 2-CTA"),
)
# The tables per (topology name, topology.Sku, cuBLASLt version).
TABLES: dict[tuple[str, topology.Sku, int], tuple[Entry, ...]] = {
    (name, topology.GB300, CUBLASLT_VERSION): ENTRIES for name in ("tp8", "tep8")
}

_TACTIC: dict = {}
_WARNED: set = set()


@cache
def _cublaslt() -> Optional[ctypes.CDLL]:
    for name in ("libcublasLt.so.13", "libcublasLt.so.12", "libcublasLt.so"):
        try:
            return ctypes.CDLL(name)
        except OSError:
            continue
    return None


@cache
def cublaslt_version() -> Optional[int]:
    lib = _cublaslt()
    if lib is None:
        return None
    lib.cublasLtGetVersion.restype = ctypes.c_size_t
    return int(lib.cublasLtGetVersion())


def _warn_once(key, message: str, *args) -> None:
    if key not in _WARNED:
        _WARNED.add(key)
        logger.warning(
            "NVIDIA megakernel: Qwen3.5 option gemm_tactics FALLBACK: " + message,
            *args,
        )


@cache
def _table(key) -> tuple[Entry, ...]:
    if key is None:
        return ()
    version = cublaslt_version()
    table = TABLES.get((*key, version), ())
    if not table and any(k[:2] == key for k in TABLES):
        _warn_once(
            "version",
            "the table is for cuBLASLt %d, this process runs %s; "
            "FlashInfer's BMM runs unchanged",
            CUBLASLT_VERSION,
            version,
        )
    return table


def enabled() -> bool:
    from sgl_nvidia_mega.qwen3_5 import runtime

    return runtime.measured_option(OPTION) and bool(_table(runtime.MEASURED_KEY))


def entry(rows: int, n: int, k: int) -> Optional[Entry]:
    """The table entry of ``rows`` for the pinned shape where the option holds, else None."""
    from sgl_nvidia_mega.qwen3_5 import runtime

    if (n, k) != SHAPE_NK or not runtime.measured_option(OPTION):
        return None
    return next((e for e in _table(runtime.MEASURED_KEY) if e.lo <= rows <= e.hi), None)


def config(algo_buf: torch.Tensor, index: int) -> Optional[Config]:
    """The configuration of the ``index``-th algorithm in a FlashInfer algorithm buffer."""
    lib = _cublaslt()
    if lib is None:
        return None
    fn = lib.cublasLtMatmulAlgoConfigGetAttribute
    fn.argtypes = [
        ctypes.c_void_p,
        ctypes.c_int,
        ctypes.c_void_p,
        ctypes.c_size_t,
        ctypes.POINTER(ctypes.c_size_t),
    ]
    fn.restype = ctypes.c_int
    ptr = algo_buf.data_ptr() + index * _ALGO_BYTES
    values = []
    for attr in _CONFIG_ATTRS:
        size = ctypes.c_size_t(0)
        if fn(ptr, attr, None, 0, ctypes.byref(size)) != 0 or not 0 < size.value <= 8:
            return None
        value, written = ctypes.c_uint64(0), ctypes.c_size_t(0)
        if fn(ptr, attr, ctypes.byref(value), size.value, ctypes.byref(written)) != 0:
            return None
        values.append(value.value)
    return Config(*values)


@cache
def _runner():
    from flashinfer.gemm.gemm_base import get_gemm_module

    return get_gemm_module().cublas_fp8_gemm_runner()


def _inputs(x8, weight, x_scale, w_scale, out) -> list:
    from flashinfer.gemm.gemm_base import DEFAULT_WORKSPACE_SIZE
    from flashinfer.utils import _get_cache_buf

    workspace = _get_cache_buf("bmm_fp8_workspace", DEFAULT_WORKSPACE_SIZE, x8.device)
    return [
        x8.unsqueeze(0),
        weight.unsqueeze(0),
        x_scale.reshape(1),
        w_scale.reshape(1),
        out.unsqueeze(0),
        workspace,
    ]


def algorithms(x8, weight, x_scale, w_scale, out) -> list[Optional[Config]]:
    """cuBLASLt's algorithm list for FlashInfer's BMM at this shape (index = tactic)."""
    algo_buf, count = _runner()._get_algos(_inputs(x8, weight, x_scale, w_scale, out))
    return [config(algo_buf, i) for i in range(count)]


def run_tactic(x8, weight, x_scale, w_scale, out, tactic: int) -> torch.Tensor:
    """FlashInfer's BMM on its ``tactic``-th cuBLASLt algorithm (-1: its heuristic)."""
    _runner().forward(_inputs(x8, weight, x_scale, w_scale, out), tactic=tactic)
    return out


def _bitwise(x8, weight, x_scale, w_scale, dtype, tactic: int) -> bool:
    """Whether ``tactic`` gives the heuristic's bits on a check GEMM of these weights."""
    rows, k = x8.shape
    g = torch.Generator(device=x8.device).manual_seed(rows)
    probe = (torch.randn((rows, k), device=x8.device, generator=g) * 64.0).to(
        torch.float8_e4m3fn
    )
    out = torch.empty((rows, weight.shape[1]), dtype=dtype, device=x8.device)
    ref = run_tactic(probe, weight, x_scale, w_scale, out.clone(), -1)
    got = run_tactic(probe, weight, x_scale, w_scale, out, tactic)
    return torch.equal(got.view(torch.int16), ref.view(torch.int16))


def _api_mismatch(exc: Exception) -> None:
    _warn_once(
        "flashinfer",
        "FlashInfer's cuBLAS FP8 runner does not match FLASHINFER_READS (%s: %s); "
        "FlashInfer's BMM runs unchanged",
        type(exc).__name__,
        exc,
    )


def _resolve(x8, weight, x_scale, w_scale, dtype, e: Entry) -> Optional[int]:
    rows = x8.shape[0]
    if e.config is None:
        logger.info(
            "NVIDIA megakernel: Qwen3.5 option gemm_tactics: FlashInfer's heuristic at "
            "%d rows",
            rows,
        )
        return -1
    out = torch.empty((rows, weight.shape[1]), dtype=dtype, device=x8.device)
    try:
        listed = algorithms(x8, weight, x_scale, w_scale, out)
        found = [i for i, c in enumerate(listed) if c == e.config]
        same = bool(found) and _bitwise(x8, weight, x_scale, w_scale, dtype, found[0])
    except _API_ERRORS as exc:
        _api_mismatch(exc)
        return None
    if not found:
        _warn_once((rows, "list"), "cuBLASLt lists no %s at %d rows", e.name, rows)
        return -1
    if not same:
        _warn_once(
            (rows, "check"), "%s differs from the heuristic at %d rows", e.name, rows
        )
        return -1
    logger.info(
        "NVIDIA megakernel: Qwen3.5 option gemm_tactics: cuBLASLt %s (tactic %d) at %d "
        "rows",
        e.name,
        found[0],
        rows,
    )
    return found[0]


def pinned(x8, weight, x_scale, w_scale, dtype) -> Optional[torch.Tensor]:
    rows, k = x8.shape
    e = entry(rows, weight.shape[1], k)
    if e is None:
        return None
    key = (x8.device.index, rows, dtype)
    if key not in _TACTIC:
        if e.config is not None and torch.cuda.is_current_stream_capturing():
            _warn_once(
                (rows, "capture"), "first call at %d rows inside a capture", rows
            )
            _TACTIC[key] = -1
        else:
            _TACTIC[key] = _resolve(x8, weight, x_scale, w_scale, dtype, e)
    tactic = _TACTIC[key]
    if tactic is None:
        return None
    out = torch.empty((rows, weight.shape[1]), dtype=dtype, device=x8.device)
    try:
        # a missing name raises before any launch
        return run_tactic(x8, weight, x_scale, w_scale, out, tactic)
    except _API_ERRORS as exc:
        _api_mismatch(exc)
        _TACTIC[key] = None
        return None


def bmm(x8, weight, x_scale, w_scale, dtype) -> torch.Tensor:
    """The FP8 BMM of a covered call site: the table's algorithm, else FlashInfer's."""
    out = pinned(x8, weight, x_scale, w_scale, dtype)
    if out is not None:
        return out
    from sglang.srt.layers.quantization.fp8_utils import flashinfer_bmm_fp8

    return flashinfer_bmm_fp8(x8, weight, x_scale, w_scale, dtype)
