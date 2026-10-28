# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# Adapted from SGLang branch nvidia/qwen35-agentx-gb300:
# python/sglang/kernels/ops/attention/mka_fused.py and mka_fused.cu (its host side).
"""MKA (``src/mka.cu``): the fused full-attention prologue and epilogue of one request's
verify rows (at most 8) at hidden 4096 and head_dim 256, the kernels of SGLang branch
nvidia/qwen35-agentx-gb300 built with NVRTC.

``prologue``: the normed hidden rows' static FP8 quantization, the qkv (+ gate) GEMM, the
q / k Gemma RMSNorm, the partial NeoX M-RoPE, the FP8 query rows, the FP8 K / V rows
written into the paged cache (one KV head a rank) and the bf16 gate. ``epilogue``: the
attention output times sigmoid(gate), its static FP8 quantization and the ``o_proj``
GEMM to bf16, optionally pushed into the decode hop's mailbox. Each launches clusters of
8 CTAs with the dependent launch; the weights come packed by ``pack_qkv`` / ``pack_o``.
"""

from __future__ import annotations

import ctypes
from functools import cache
from pathlib import Path
from typing import Optional

import torch

from sgl_nvidia_mega.common import jit
from sgl_nvidia_mega.qwen3_5.sm103 import hop
from sgl_nvidia_mega.qwen3_5.sm103.hop import Site

SOURCE = Path(__file__).parent / "src" / "mka.cu"
THREADS, CLUSTER, HEAD_DIM, HIDDEN, MAX_ROWS = 256, 8, 256, 4096, 8
ENTRIES = (
    "q35mka_prologue",
    "q35mka_prologue_late",
    "q35mka_gate",
    "q35mka_epilogue_q4",
    "q35mka_epilogue_q4_late",
    "q35mka_epilogue_q8",
    "q35mka_epilogue_q8_late",
)


class ProParams(ctypes.Structure):
    _fields_ = [
        (name, ctypes.c_void_p)
        for name in (
            "x",
            "w",
            "sx",
            "sw",
            "qnw",
            "knw",
            "cos_sin",
            "pos",
            "axis",
            "loc",
        )
    ] + [
        ("loc_i32", ctypes.c_int),
        ("k_cache", ctypes.c_void_p),
        ("v_cache", ctypes.c_void_p),
        ("k_scale", ctypes.c_void_p),
        ("v_scale", ctypes.c_void_p),
        ("q_out", ctypes.c_void_p),
        ("gate_out", ctypes.c_void_p),
        ("m", ctypes.c_int),
        ("nq", ctypes.c_int),
        ("pos_stride", ctypes.c_int),
        ("cs_stride", ctypes.c_int),
        ("eps", ctypes.c_float),
        ("split", ctypes.c_int),
        ("wg", ctypes.c_void_p),
    ]


class EpiParams(ctypes.Structure):
    _fields_ = [
        (name, ctypes.c_void_p) for name in ("attn", "gate", "w", "sx", "sw", "out")
    ] + [("m", ctypes.c_int), ("push", ctypes.c_int), ("site", Site)]


@cache
def _kernels(device_index: int) -> dict[str, jit.Kernel]:
    major, minor = torch.cuda.get_device_capability(device_index)
    cubin = jit.compile_source(
        SOURCE.read_text(),
        "q35_mka",
        f"sm_{major}{minor}a",
        ("-std=c++17", "-default-device"),
        headers=hop.headers(),
    )
    return {n: jit.load(cubin, n, (THREADS, 1, 1), 0) for n in ENTRIES}


def build(device) -> None:
    """Compile (or load) the kernels now, before any capture."""
    _kernels(torch.device(device).index)


def _launch(name: str, grid: int, params, device, cluster: bool = True) -> None:
    jit.launch(
        _kernels(device.index)[name],
        (grid, 1, 1),
        [params],
        torch.cuda.current_stream(device).cuda_stream,
        (CLUSTER, 1, 1) if cluster else None,
    )


def _pack_fragments(w_nk: torch.Tensor, units: int, steps: int) -> torch.Tensor:
    # [units*256, K] e4m3 -> [unit][rank 8][warp 8][step][m16 block 2][lane 32][16 B] mma.m16n8k32 A
    # fragments; rank r covers K [r*steps*32, (r+1)*steps*32), warp w rows [32w, 32w + 32) of the unit.
    w = w_nk.contiguous().view(torch.uint8)
    assert w.shape[0] == units * 256 and w.shape[1] == 8 * steps * 32, w.shape
    v = w.view(units, 8, 2, 2, 8, 8, steps, 2, 4, 4)
    return v.permute(0, 5, 1, 6, 2, 4, 8, 7, 3, 9).contiguous().view(-1)


def _pack_rows32(w_nk: torch.Tensor) -> torch.Tensor:
    # [R, 4096] e4m3 -> [cta = R/32][warp 8][step 16][m16 block 2][lane 32][16 B]; warp w covers K
    # [512w, 512w + 512) of the CTA's 32 rows (mka_gate_kernel).
    w = w_nk.contiguous().view(torch.uint8)
    v = w.view(w.shape[0] // 32, 2, 2, 8, 8, 16, 2, 4, 4)
    return v.permute(0, 4, 5, 1, 3, 7, 6, 2, 8).contiguous().view(-1)


def split_gate(nq: int) -> bool:
    """Whether the prologue hands the gate rows to mka_gate_kernel.

    2*nq + 2 clusters of 8 CTAs exceed the 15 that co-reside on GB300 once nq = 8 (TP4).
    """
    return nq >= 8


def pack_qkv(w_nk: torch.Tensor, nq: int) -> tuple[torch.Tensor, torch.Tensor]:
    """qkv(+gate) weight [N, 4096] e4m3 -> (prologue layout, gate-kernel layout or empty)."""
    main = _pack_fragments(w_nk, w_nk.shape[0] // 256, 16)
    if not split_gate(nq):
        return main, main.new_empty(0)
    gate_rows = (
        w_nk[: nq * 512]
        .reshape(nq, 2, 256, w_nk.shape[1])[:, 1]
        .reshape(nq * 256, w_nk.shape[1])
    )
    return main, _pack_rows32(gate_rows)


def pack_o(w_nk: torch.Tensor) -> torch.Tensor:
    """o_proj weight [4096, nq*256] e4m3 -> epilogue layout (K chunk nq*32 per cluster rank)."""
    return _pack_fragments(w_nk, 16, w_nk.shape[1] // 256)


def prologue(
    x,
    w_packed,
    sx,
    sw,
    q_norm_w,
    k_norm_w,
    cos_sin,
    positions,
    axis_map,
    cache_loc,
    k_cache,
    v_cache,
    k_scale,
    v_scale,
    q_out,
    gate_out,
    nq: int,
    eps: float,
    early_trigger: bool = True,
) -> None:
    """The branch's ``prologue`` (its host function's checks and launches)."""
    w_main, w_gate = w_packed
    if (
        not (x.dim() == 2 and x.shape[0] <= MAX_ROWS and x.shape[1] == HIDDEN)
        or not x.is_contiguous()
    ):
        raise ValueError(
            f"MKA prologue: x {tuple(x.shape)} is not [<= 8, 4096] contiguous"
        )
    if positions.dim() != 2 or positions.shape[0] != 3 or positions.stride(1) != 1:
        raise ValueError(
            "MKA prologue: positions must be M-RoPE [3, rows], rows contiguous"
        )
    if cos_sin.dtype != torch.float32 or axis_map.dtype != torch.int64:
        raise ValueError("MKA prologue: cos_sin must be float32 and axis_map int64")
    if positions.dtype != torch.int64:
        raise ValueError("MKA prologue: positions must be int64")
    if k_cache.stride(0) != HEAD_DIM or v_cache.stride(0) != HEAD_DIM:
        raise ValueError("MKA prologue: one KV head of 256 per slot")
    if cache_loc.dtype not in (torch.int64, torch.int32):
        raise ValueError(f"MKA prologue: cache_loc of {cache_loc.dtype}")
    split = w_gate.numel() > 0
    p = ProParams(
        x=x.data_ptr(),
        w=w_main.data_ptr(),
        sx=sx.data_ptr(),
        sw=sw.data_ptr(),
        qnw=q_norm_w.data_ptr(),
        knw=k_norm_w.data_ptr(),
        cos_sin=cos_sin.data_ptr(),
        pos=positions.data_ptr(),
        axis=axis_map.data_ptr(),
        loc=cache_loc.data_ptr(),
        loc_i32=int(cache_loc.dtype == torch.int32),
        k_cache=k_cache.data_ptr(),
        v_cache=v_cache.data_ptr(),
        k_scale=k_scale.data_ptr(),
        v_scale=v_scale.data_ptr(),
        q_out=q_out.data_ptr(),
        gate_out=gate_out.data_ptr(),
        m=int(x.shape[0]),
        nq=int(nq),
        pos_stride=int(positions.stride(0)),
        cs_stride=int(cos_sin.stride(0)),
        eps=float(eps),
        split=int(split),
        wg=w_gate.data_ptr() if split else None,
    )
    if not split:
        name = "q35mka_prologue" if early_trigger else "q35mka_prologue_late"
        _launch(name, (2 * nq + 2) * CLUSTER, p, x.device)
        return
    # The gate launch must start while the prologue runs, so the prologue triggers right after its wait.
    _launch("q35mka_prologue", (nq + 2) * CLUSTER, p, x.device)
    _launch("q35mka_gate", nq * HEAD_DIM // 32, p, x.device, cluster=False)


def epilogue(
    attn,
    gate,
    w_packed,
    sx,
    sw,
    out,
    nq: int,
    early_trigger: bool = True,
    site: Optional[Site] = None,
) -> None:
    """The branch's ``epilogue``; ``site``: the hop mailbox to push the rows into."""
    if not (attn.is_contiguous() and gate.is_contiguous() and out.is_contiguous()):
        raise ValueError("MKA epilogue: attn, gate and out must be contiguous")
    if attn.shape[0] > MAX_ROWS or attn.shape[1] != nq * HEAD_DIM:
        raise ValueError(f"MKA epilogue: attn {tuple(attn.shape)} for {nq} heads")
    if nq not in (4, 8):
        raise ValueError("MKA epilogue: nq must be 4 (TP8) or 8 (TP4)")
    m = int(attn.shape[0])
    if site is not None and m > site.m_cap:
        raise ValueError(f"MKA epilogue: {m} rows past the hop's {site.m_cap}")
    p = EpiParams(
        attn=attn.data_ptr(),
        gate=gate.data_ptr(),
        w=w_packed.data_ptr(),
        sx=sx.data_ptr(),
        sw=sw.data_ptr(),
        out=out.data_ptr(),
        m=m,
        push=int(site is not None),
        site=site if site is not None else Site(),
    )
    name = f"q35mka_epilogue_q{nq}" + ("" if early_trigger else "_late")
    _launch(name, 16 * CLUSTER, p, attn.device)
