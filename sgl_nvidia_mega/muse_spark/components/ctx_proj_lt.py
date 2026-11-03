# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Pinned cuBLASLt column slices for ``ctx_proj_overlap`` shard mode (``MUSE_VR_CTX_PROJ_OVERLAP_SHARD_PIN=1``, default
off; used only by that component).

A column slice ``x @ W[r * n:(r + 1) * n]^T`` equals the same columns of the full ``x @ W^T`` bit for bit when both run
the same summation: the same split of the reduction dimension and the same reduction of the split parts.
``F.linear`` lets the library choose a configuration per shape, and for the slice shape it may choose a different
split than for the full shape, so the slice and the full columns can differ in the last bits.

This module asks the library's heuristic (``cublasLtMatmulAlgoGetHeuristic``) for the configuration of the FULL shape
- bf16 operands and output, fp32 compute and scale, the transposes ``F.linear`` uses, and the preference fields the
full GEMM is planned with (workspace size, 16-byte operand alignment) - and keeps that configuration's split count and
reduction scheme for the slice. ``slice_candidates`` lists the configurations to try for the slice shape, in order: the
heuristic's first picks for the slice shape with the full configuration's split count and reduction scheme put in (a
tile and schedule chosen for the slice shape, the full GEMM's split of the reduction and reduction of the parts), then
the full configuration itself. ``run_slice`` runs one (``cublasLtMatmul``, alpha 1, beta 0) on the caller's stream,
with a workspace owned by that stream's caller. The equality is not assumed anywhere: ``ctx_proj_overlap`` checks each
candidate at engage time, per row count, against PyTorch's own full GEMM, on every rank, keeps one that is
bit-identical on every rank (the fastest by its slowest rank), and shards only the row counts that have one.

Workspace size: ``MUSE_VR_CTX_PROJ_PIN_WS_BYTES`` if set, else the size PyTorch gives its cuBLAS handles
(``CUBLAS_WORKSPACE_CONFIG`` parsed the same way, else its default for the device).
"""

from __future__ import annotations

import ctypes
import os
import threading

import torch

LIBS = ("libcublasLt.so.13", "libcublasLt.so.12")

# cuBLASLt enum values (cublasLt.h / library_types.h)
_COMPUTE_32F = 68
_R_32F, _R_16BF = 0, 14
_DESC_TRANSA, _DESC_TRANSB = 3, 4
_OP_N, _OP_T = 0, 1
_PREF_MAX_WORKSPACE_BYTES = 1
_PREF_MIN_ALIGNMENT = (5, 6, 7, 8)  # A, B, C, D bytes
_CFG_ATTRS = tuple(
    range(9)
)  # CUBLASLT_ALGO_CONFIG_ID .. CLUSTER_SHAPE_ID; 7 and 8 are uint16
_CFG_U16 = (7, 8)
_SUCCESS = 0


class _Algo(ctypes.Structure):
    _fields_ = [("data", ctypes.c_uint64 * 8)]


class _Heur(ctypes.Structure):
    _fields_ = [
        ("algo", _Algo),
        ("ws", ctypes.c_size_t),
        ("state", ctypes.c_int),
        ("waves", ctypes.c_float),
        ("res", ctypes.c_int * 4),
    ]


_lock = threading.Lock()
_state: dict = {"lib": None}
_dev: dict = {}  # device index -> {"handle", "desc", "pref", "layouts", "ws"}
_ONE = ctypes.c_float(1.0)
_ZERO = ctypes.c_float(0.0)


def _lib():
    if _state["lib"] is None:
        err = None
        for name in LIBS:
            try:
                _state["lib"] = ctypes.CDLL(name)
                break
            except OSError as e:  # try the next soname
                err = e
        if _state["lib"] is None:
            raise OSError(f"no cuBLASLt library loadable ({err})")
    return _state["lib"]


def library_version() -> int:
    lt = _lib()
    lt.cublasLtGetVersion.restype = ctypes.c_size_t
    return int(lt.cublasLtGetVersion())


def _check(rc: int, what: str) -> None:
    if rc != _SUCCESS:
        raise RuntimeError(f"{what}: cuBLASLt status {rc}")


def torch_workspace_bytes(device: torch.device) -> int:
    """The workspace size PyTorch gives its cuBLAS handles (``CUBLAS_WORKSPACE_CONFIG`` = ``:SIZE_KiB:COUNT`` pairs, else
    its per-device default), unless ``MUSE_VR_CTX_PROJ_PIN_WS_BYTES`` overrides it."""
    override = os.environ.get("MUSE_VR_CTX_PROJ_PIN_WS_BYTES")
    if override:
        return int(override)
    cfg = os.environ.get("CUBLAS_WORKSPACE_CONFIG")
    if cfg:
        vals = [v for v in cfg.split(":") if v]
        try:
            return sum(
                int(vals[i]) * 1024 * int(vals[i + 1])
                for i in range(0, len(vals) - 1, 2)
            )
        except ValueError:
            pass
    major = torch.cuda.get_device_properties(device).major
    return 32 * 1024 * 1024 if major >= 9 else 4096 * 1024 * 2 + 16 * 1024 * 8


def _device_state(device: torch.device) -> dict:
    idx = device.index if device.index is not None else torch.cuda.current_device()
    st = _dev.get(idx)
    if st is not None:
        return st
    with _lock:
        st = _dev.get(idx)
        if st is not None:
            return st
        lt = _lib()
        handle = ctypes.c_void_p()
        _check(lt.cublasLtCreate(ctypes.byref(handle)), "create")
        desc = ctypes.c_void_p()
        _check(
            lt.cublasLtMatmulDescCreate(
                ctypes.byref(desc), ctypes.c_int(_COMPUTE_32F), ctypes.c_int(_R_32F)
            ),
            "matmul desc",
        )
        for attr, op in ((_DESC_TRANSA, _OP_T), (_DESC_TRANSB, _OP_N)):
            v = ctypes.c_int32(op)
            _check(
                lt.cublasLtMatmulDescSetAttribute(
                    desc, ctypes.c_int(attr), ctypes.byref(v), ctypes.c_size_t(4)
                ),
                "transpose",
            )
        pref = ctypes.c_void_p()
        _check(lt.cublasLtMatmulPreferenceCreate(ctypes.byref(pref)), "preference")
        ws = ctypes.c_uint64(torch_workspace_bytes(torch.device("cuda", idx)))
        _check(
            lt.cublasLtMatmulPreferenceSetAttribute(
                pref,
                ctypes.c_int(_PREF_MAX_WORKSPACE_BYTES),
                ctypes.byref(ws),
                ctypes.c_size_t(8),
            ),
            "preference workspace",
        )
        for attr in _PREF_MIN_ALIGNMENT:
            v = ctypes.c_uint32(16)
            _check(
                lt.cublasLtMatmulPreferenceSetAttribute(
                    pref, ctypes.c_int(attr), ctypes.byref(v), ctypes.c_size_t(4)
                ),
                "preference alignment",
            )
        st = {
            "handle": handle,
            "desc": desc,
            "pref": pref,
            "layouts": {},
            "ws": None,
            "ws_pref": int(ws.value),
        }
        _dev[idx] = st
    return st


def _layout(st: dict, rows: int, cols: int, ld: int):
    key = (rows, cols, ld)
    lay = st["layouts"].get(key)
    if lay is None:
        lay = ctypes.c_void_p()
        _check(
            _lib().cublasLtMatrixLayoutCreate(
                ctypes.byref(lay),
                ctypes.c_int(_R_16BF),
                ctypes.c_uint64(rows),
                ctypes.c_uint64(cols),
                ctypes.c_int64(ld),
            ),
            "layout",
        )
        st["layouts"][key] = lay
    return lay


def _layouts(st: dict, m: int, n: int, k: int, ldw: int):
    """A, B, C/D for ``out[m, n] = x[m, k] @ w[n, k]^T`` (rows of ``w`` ``ldw`` elements apart): column-major views
    C[n, m] = op_T(A[k, n]) x B[k, m]."""
    return _layout(st, k, n, ldw), _layout(st, k, m, k), _layout(st, n, m, n)


def config(algo: _Algo) -> tuple:
    """The nine ``CUBLASLT_ALGO_CONFIG_*`` values of ``algo`` (for reports)."""
    lt = _lib()
    out = []
    for attr in _CFG_ATTRS:
        v = ctypes.c_uint16(0) if attr in _CFG_U16 else ctypes.c_uint32(0)
        w = ctypes.c_size_t(0)
        lt.cublasLtMatmulAlgoConfigGetAttribute(
            ctypes.byref(algo),
            ctypes.c_int(attr),
            ctypes.byref(v),
            ctypes.c_size_t(ctypes.sizeof(v)),
            ctypes.byref(w),
        )
        out.append(int(v.value))
    return tuple(out)


def full_shape_algo(device: torch.device, m: int, n_full: int, k: int):
    """The heuristic's first configuration for the full GEMM ``[m, k] x [n_full, k]^T`` (None if it returns none)."""
    st = _device_state(device)
    a, b, c = _layouts(st, m, n_full, k, k)
    res = _Heur()
    got = ctypes.c_int(0)
    rc = _lib().cublasLtMatmulAlgoGetHeuristic(
        st["handle"],
        st["desc"],
        a,
        b,
        c,
        c,
        st["pref"],
        ctypes.c_int(1),
        ctypes.byref(res),
        ctypes.byref(got),
    )
    if rc != _SUCCESS or got.value < 1 or res.state != _SUCCESS:
        return None
    return res.algo


def _build(handle, cfg: tuple):
    """An algorithm descriptor with the nine ``CUBLASLT_ALGO_CONFIG_*`` values ``cfg`` (None if the library refuses)."""
    lt = _lib()
    algo = _Algo()
    if (
        lt.cublasLtMatmulAlgoInit(
            handle,
            ctypes.c_int(_COMPUTE_32F),
            ctypes.c_int(_R_32F),
            ctypes.c_int(_R_16BF),
            ctypes.c_int(_R_16BF),
            ctypes.c_int(_R_16BF),
            ctypes.c_int(_R_16BF),
            ctypes.c_int(cfg[0]),
            ctypes.byref(algo),
        )
        != _SUCCESS
    ):
        return None
    for attr, val in zip(_CFG_ATTRS[1:], cfg[1:]):
        v = ctypes.c_uint16(val) if attr in _CFG_U16 else ctypes.c_uint32(val)
        if (
            lt.cublasLtMatmulAlgoConfigSetAttribute(
                ctypes.byref(algo),
                ctypes.c_int(attr),
                ctypes.byref(v),
                ctypes.c_size_t(ctypes.sizeof(v)),
            )
            != _SUCCESS
        ):
            return None
    return algo


def slice_candidates(
    device: torch.device, full_algo: _Algo, m: int, n: int, k: int, count: int = 3
) -> list:
    """``[(algo, workspace bytes)]`` to try for the slice ``[m, k] x [n, k]^T``, in order: the heuristic's first ``count``
    picks for the slice shape with the full configuration's split count and reduction scheme put in, then the full
    configuration itself; only configurations the library accepts for the slice shape, without repeats."""
    st = _device_state(device)
    full_cfg = config(full_algo)
    a, b, c = _layouts(st, m, n, k, k)
    res = (_Heur * count)()
    got = ctypes.c_int(0)
    picks = []
    if (
        _lib().cublasLtMatmulAlgoGetHeuristic(
            st["handle"],
            st["desc"],
            a,
            b,
            c,
            c,
            st["pref"],
            ctypes.c_int(count),
            res,
            ctypes.byref(got),
        )
        == _SUCCESS
    ):
        for i in range(got.value):
            if res[i].state == _SUCCESS:
                cfg = list(config(res[i].algo))
                cfg[2], cfg[3] = (
                    full_cfg[2],
                    full_cfg[3],
                )  # SPLITK_NUM, REDUCTION_SCHEME of the full configuration
                picks.append(tuple(cfg))
    out, seen = [], set()
    for cfg in picks + [full_cfg]:
        if cfg in seen:
            continue
        seen.add(cfg)
        algo = full_algo if cfg == full_cfg else _build(st["handle"], cfg)
        ws = None if algo is None else slice_workspace(device, algo, m, n, k)
        if ws is not None:
            out.append((algo, ws))
    return out


def slice_workspace(device: torch.device, algo: _Algo, m: int, n: int, k: int):
    """Workspace bytes ``algo`` needs for the slice ``[m, k] x [n, k]^T`` (rows of the weight ``k`` apart), or None if
    the library rejects the configuration for that shape (``cublasLtMatmulAlgoCheck``)."""
    st = _device_state(device)
    a, b, c = _layouts(st, m, n, k, k)
    res = _Heur()
    rc = _lib().cublasLtMatmulAlgoCheck(
        st["handle"], st["desc"], a, b, c, c, ctypes.byref(algo), ctypes.byref(res)
    )
    if rc != _SUCCESS or res.state != _SUCCESS:
        return None
    return int(res.ws)


def reserve_workspace(device: torch.device, nbytes: int) -> None:
    """Allocate (or grow) this device's slice workspace. Call it on the stream that runs the slices, outside graph
    capture; every slice of this module runs on that one stream, so the workspace is never used concurrently."""
    st = _device_state(device)
    if st["ws"] is None or st["ws"].numel() < nbytes:
        st["ws"] = torch.empty(max(nbytes, 256), dtype=torch.uint8, device=device)


def workspace_bytes(device: torch.device) -> int:
    """Size of this device's slice workspace (0 before ``reserve_workspace``)."""
    st = _device_state(device)
    return 0 if st["ws"] is None else int(st["ws"].numel())


def run_slice(
    algo: _Algo,
    x: torch.Tensor,
    w_slice: torch.Tensor,
    stream: torch.cuda.Stream | None = None,
    workspace: torch.Tensor | None = None,
) -> torch.Tensor:
    """``F.linear(x, w_slice)`` (2-D bf16, contiguous rows) through ``algo`` on ``stream`` (default: the current
    stream), into a new tensor. The caller has checked the configuration for this shape and reserved a workspace at
    least as large as the configuration needs for it: this module's (``reserve_workspace``), or ``workspace`` (a uint8
    buffer the caller owns and uses on that stream only)."""
    st = _device_state(x.device)
    stream = torch.cuda.current_stream(x.device) if stream is None else stream
    m, k = x.shape
    n = w_slice.shape[0]
    a, b, c = _layouts(st, m, n, k, w_slice.stride(0))
    out = torch.empty((m, n), dtype=torch.bfloat16, device=x.device)
    ws = st["ws"] if workspace is None else workspace
    rc = _lib().cublasLtMatmul(
        st["handle"],
        st["desc"],
        ctypes.byref(_ONE),
        ctypes.c_void_p(w_slice.data_ptr()),
        a,
        ctypes.c_void_p(x.data_ptr()),
        b,
        ctypes.byref(_ZERO),
        ctypes.c_void_p(out.data_ptr()),
        c,
        ctypes.c_void_p(out.data_ptr()),
        c,
        ctypes.byref(algo),
        ctypes.c_void_p(ws.data_ptr()),
        ctypes.c_size_t(ws.numel()),
        ctypes.c_void_p(stream.cuda_stream),
    )
    _check(rc, "matmul")
    return out
