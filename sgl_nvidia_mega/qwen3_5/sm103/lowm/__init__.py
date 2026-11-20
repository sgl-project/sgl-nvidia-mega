# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Low-M projections (``src/lowm.cu``): ``C = A @ W^T`` for up to 64 rows on warp MMA, the weights loaded
into registers before the programmatic dependency wait and the activations after it.

A launch has an FP8 segment (E4M3 weights with a per-tensor scale; E4M3 activations, or BF16 ones quantized
in registers with the static input scale exactly as SGLang's static quantize kernel) and a BF16 segment
(BF16 weights and activations), each a whole number of CTAs. The FP8 segment's epilogue is a BF16 output
(``acc * s_in * s_w``), a SwiGLU on (8 gate rows, 8 up rows) pairs written as E4M3 with a static scale, or a
BF16 output times ``sigmoid(gate)`` per row. Numeric against SGLang's quantize + cuBLAS path: the same E4M3
operands, FP32 sums in another order.
"""

from __future__ import annotations

import ctypes
from functools import cache
from pathlib import Path
from typing import NamedTuple, Optional

import torch

from sgl_nvidia_mega.common import jit

SOURCE = Path(__file__).parent / "src" / "lowm.cu"
MAX_ROWS = 64
CHUNK = 64


class Params(ctypes.Structure):
    """Params of src/lowm.cu."""

    _fields_ = [
        ("x8", ctypes.c_void_p),
        ("ld8", ctypes.c_int64),
        ("x16", ctypes.c_void_p),
        ("ld16", ctypes.c_int64),
        ("w8", ctypes.c_void_p),
        ("s_w", ctypes.c_void_p),
        ("s_in", ctypes.c_void_p),
        ("s_out", ctypes.c_void_p),
        ("w16", ctypes.c_void_p),
        ("out", ctypes.c_void_p),
        ("ldo", ctypes.c_int64),
        ("out2", ctypes.c_void_p),
        ("ldo2", ctypes.c_int64),
        ("gate", ctypes.c_void_p),
        ("ldg", ctypes.c_int64),
        ("hop_mc", ctypes.c_void_p),
        ("hop_uc", ctypes.c_void_p),
        ("hop_ctrl", ctypes.c_void_p),
        ("hop_rank", ctypes.c_int),
        ("hop_world", ctypes.c_int),
        ("hop_m_cap", ctypes.c_int),
    ]


class Tactic(NamedTuple):
    """``kw`` warps split K (one 64-column chunk at a time); ``nt8`` / ``nt16`` groups of 8 weight rows a
    CTA of the FP8 / BF16 segment; ``min_blocks`` CTAs a SM for the launch bounds; ``ksplit`` CTAs of a
    cluster split K further and sum their partials through distributed shared memory in rank order."""

    kw: int
    nt8: int
    nt16: int = 1
    min_blocks: int = 1
    ksplit: int = 1


# the compiled launches' register and local-memory use (checks and tuning)
KERNEL_INFO: dict = {}


@cache
def _kernel(
    device_index: int,
    m: int,
    k: int,
    tactic: Tactic,
    ctas8: int,
    ctas16: int,
    act: int,
    epi8: int,
    out2_f32: bool,
    trigger: bool,
    push: bool = False,
) -> jit.Kernel:
    major, minor = torch.cuda.get_device_capability(device_index)
    defines = dict(
        M_ROWS=m,
        K_DIM=k,
        KW=tactic.kw,
        NT8=tactic.nt8,
        NT16=tactic.nt16,
        CTAS8=ctas8,
        CTAS16=ctas16,
        ACT=act,
        EPI8=epi8,
        OUT2_F32=int(out2_f32),
        TRIGGER=int(trigger),
        MIN_BLOCKS=tactic.min_blocks,
        KSPLIT=tactic.ksplit,
        PUSH=int(push),
    )
    source = "".join(f"#define {key} {value}\n" for key, value in defines.items())
    headers = None
    if push:
        from sgl_nvidia_mega.qwen3_5.sm103 import hop

        headers = hop.headers()
    cubin = jit.compile_source(
        source + SOURCE.read_text(),
        "q35_lowm",
        f"sm_{major}{minor}a",
        ("-std=c++17", "-default-device"),
        headers,
    )
    nc = max(tactic.nt8 if ctas8 else 0, tactic.nt16 if ctas16 else 0) * 8
    shared = tactic.kw * ((m + 15) // 16) * 16 * nc * 4
    if tactic.ksplit > 1:
        shared += m * nc * 4
    kernel = jit.load(cubin, "q35_lowm", (32 * tactic.kw, 1, 1), shared)
    cu = jit.driver()
    attr = cu.CUfunction_attribute
    regs = jit.check(
        cu.cuFuncGetAttribute(attr.CU_FUNC_ATTRIBUTE_NUM_REGS, kernel.function), "regs"
    )[0]
    local = jit.check(
        cu.cuFuncGetAttribute(attr.CU_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES, kernel.function),
        "lmem",
    )[0]
    KERNEL_INFO[(m, k, tactic, ctas8, ctas16, act, epi8, push)] = (
        int(regs),
        int(local),
        shared,
    )
    return kernel


def default_tactic(m: int, n8: int, n16: int, k: int, swiglu: bool = False) -> Tactic:
    """A tactic for ``m`` rows of an ``[n8 | n16, k]`` launch: each lane prefetches at most 256 bytes of
    weights, warps take at least one chunk, and the grid fits two CTAs a SM where it can."""
    chunks = k // CHUNK
    best = None
    for nt in (1, 2, 4, 8):
        if swiglu and nt % 2:
            continue
        if (n8 and n8 % (8 * nt)) or (n16 and n16 % (8 * min(nt, n16 // 8))):
            continue
        nt16 = min(nt, n16 // 8) if n16 else 1
        ctas = (n8 // (8 * nt) if n8 else 0) + (n16 // (8 * nt16) if n16 else 0)
        for kw in (1, 2, 4, 8, 16):
            if chunks % kw:
                continue
            per_lane = (chunks // kw) * max(
                nt * 16 if n8 else 0, nt16 * 32 if n16 else 0
            )
            if per_lane > 256:
                continue
            waves = -(-ctas // (2 * 148))
            key = (waves, chunks // kw > 8, -min(ctas, 2 * 148), chunks // kw, -kw)
            if best is None or key < best[0]:
                best = (key, Tactic(kw, nt, nt16, 1))
    if best is None:
        raise ValueError(f"no low-M tactic for N={n8}+{n16}, K={k}")
    return best[1]


def check_launch(
    m: int, n8: int, n16: int, k: int, tactic: Tactic, swiglu: bool = False
) -> str:
    """Why ``tactic`` cannot run this launch ("" when it can)."""
    if not 0 < m <= MAX_ROWS:
        return f"{m} rows"
    if k % CHUNK or (k // CHUNK) % (tactic.kw * tactic.ksplit):
        return f"K={k} over {tactic.kw} warps x {tactic.ksplit} CTAs"
    if tactic.ksplit not in (1, 2, 4, 8):
        return f"a cluster of {tactic.ksplit}"
    if n8 and n8 % (8 * tactic.nt8):
        return f"N8={n8} over {tactic.nt8} tiles"
    if swiglu and tactic.nt8 % 2:
        return "a SwiGLU segment takes an even number of tiles a CTA"
    if n16 and n16 % (8 * tactic.nt16):
        return f"N16={n16} over {tactic.nt16} tiles"
    return ""


def prepare(
    device_index: int,
    m: int,
    k: int,
    n8: int,
    n16: int,
    act: int,
    epi8: int = 0,
    out2_f32: bool = False,
    tactic: Optional[Tactic] = None,
    trigger: bool = True,
    push: bool = False,
) -> Tactic:
    """Compile a launch's kernel ahead of its first call (no compile inside CUDA-graph capture);
    returns its tactic."""
    if tactic is None:
        tactic = default_tactic(m, n8, n16, k, epi8 == 1)
    reason = check_launch(m, n8, n16, k, tactic, epi8 == 1)
    if reason:
        raise ValueError(f"low-M launch: {reason}")
    ctas8 = n8 // (8 * tactic.nt8) * tactic.ksplit if n8 else 0
    ctas16 = n16 // (8 * tactic.nt16) * tactic.ksplit if n16 else 0
    _kernel(
        device_index, m, k, tactic, ctas8, ctas16, act, epi8, out2_f32, trigger, push
    )
    return tactic


def gemm(
    *,
    x8: Optional[torch.Tensor] = None,
    x16: Optional[torch.Tensor] = None,
    w8: Optional[torch.Tensor] = None,
    w_scale: Optional[torch.Tensor] = None,
    in_scale: Optional[torch.Tensor] = None,
    w16: Optional[torch.Tensor] = None,
    out: Optional[torch.Tensor] = None,
    out2: Optional[torch.Tensor] = None,
    swiglu_scale: Optional[torch.Tensor] = None,
    gate: Optional[torch.Tensor] = None,
    tactic: Optional[Tactic] = None,
    trigger: bool = True,
    push_site=None,
):
    """``[x] @ [w8; w16]^T`` into ``out`` (FP8 segment) and ``out2`` (BF16 segment); returns (out, out2).

    x8 [M, K] E4M3 or x16 [M, K] BF16 (unit column stride, 16-byte aligned rows); w8 [N8, K] E4M3 row-major
    with the per-tensor ``w_scale`` and the static ``in_scale``; w16 [N16, K] BF16 row-major (takes x16).
    out: BF16 [M, N8], or E4M3 [M, N8 / 2] with ``swiglu_scale`` (w8 rows paired 8 gate / 8 up, see
    ``swiglu_pairs``); ``gate`` [M] FP32 logits multiply the BF16 output by their sigmoid. out2: BF16 or
    FP32 [M, N16]. ``push_site`` (a sm103/hop ``Site``; BF16 output of hidden width only): the output rows
    are also pushed into the decode hop's mailbox, whose all-reduce must come next."""
    xa = x16 if x16 is not None else x8
    m, k = xa.shape
    n8 = 0 if w8 is None else w8.shape[0]
    n16 = 0 if w16 is None else w16.shape[0]
    act = 0 if (n8 and x8 is not None) else 1
    epi8 = 1 if swiglu_scale is not None else (2 if gate is not None else 0)
    if tactic is None:
        tactic = default_tactic(m, n8, n16, k, epi8 == 1)
    reason = check_launch(m, n8, n16, k, tactic, epi8 == 1)
    if reason:
        raise ValueError(f"low-M launch: {reason}")
    ctas8 = n8 // (8 * tactic.nt8) * tactic.ksplit if n8 else 0
    ctas16 = n16 // (8 * tactic.nt16) * tactic.ksplit if n16 else 0
    out2_f32 = out2 is not None and out2.dtype == torch.float32
    push = push_site is not None
    if push and (epi8 != 0 or n16 or out is None or out.stride(0) % 8):
        raise ValueError("the hop push takes a plain BF16 FP8-segment output")
    kernel = _kernel(
        xa.device.index, m, k, tactic, ctas8, ctas16, act, epi8, out2_f32, trigger, push
    )
    p = Params(
        x8=0 if x8 is None else x8.data_ptr(),
        ld8=0 if x8 is None else x8.stride(0),
        x16=0 if x16 is None else x16.data_ptr(),
        ld16=0 if x16 is None else x16.stride(0),
        w8=0 if w8 is None else w8.data_ptr(),
        s_w=0 if w_scale is None else w_scale.data_ptr(),
        s_in=0 if in_scale is None else in_scale.data_ptr(),
        s_out=0 if swiglu_scale is None else swiglu_scale.data_ptr(),
        w16=0 if w16 is None else w16.data_ptr(),
        out=0 if out is None else out.data_ptr(),
        ldo=0 if out is None else out.stride(0),
        out2=0 if out2 is None else out2.data_ptr(),
        ldo2=0 if out2 is None else out2.stride(0),
        gate=0 if gate is None else gate.data_ptr(),
        ldg=0 if gate is None else gate.stride(0),
    )
    if push:
        p.hop_mc, p.hop_uc, p.hop_ctrl = push_site.mc, push_site.uc, push_site.ctrl
        p.hop_rank, p.hop_world, p.hop_m_cap = (
            push_site.rank,
            push_site.world,
            push_site.m_cap,
        )
    jit.launch(
        kernel,
        (ctas8 + ctas16, 1, 1),
        [p],
        torch.cuda.current_stream(xa.device).cuda_stream,
        cluster=(tactic.ksplit, 1, 1) if tactic.ksplit > 1 else None,
    )
    return out, out2


def swiglu_pairs(gate_up: torch.Tensor) -> torch.Tensor:
    """Rows ``[gate (I); up (I)]`` -> ``[gate 0..7; up 0..7; gate 8..15; up 8..15; ...]``."""
    rows, k = gate_up.shape
    inter = rows // 2
    gate = gate_up[:inter].reshape(inter // 8, 8, k)
    up = gate_up[inter:].reshape(inter // 8, 8, k)
    return torch.stack((gate, up), dim=1).reshape(rows, k).contiguous()
