# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Weight-streaming BF16 GEMV for the DFlash draft layers, with fused epilogues (``csrc/draft_gemv.cuh``).

``conv_in(x, w, base0, tactic)``: the block convolution's coefficient projection (``w`` [4 G, H]) with the input-side
convolution fused: returns (convolved input [M, H], coefficients [M, 4 G] of which the output-side half is written).
``swiglu(x, w, tactic)``: the merged gate/up projection (``w`` [2 I, H]) with SiLU(gate) * up fused: returns [M, I].
``plain(x, w, tactic)``: ``x @ w.T``.
Rounding points match the stock path (bf16 operands, fp32 accumulation, bf16 outputs); the fp32 summation order of the
projection differs.
"""

from __future__ import annotations

import functools
import re
from dataclasses import dataclass
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


_CSRC = Path(__file__).resolve().parents[1] / "csrc" / "draft_gemv.cuh"

TOKEN_TILES = (8, 16, 32, 64)
MAX_M = TOKEN_TILES[-1]
EPI_PLAIN, EPI_CONV, EPI_SWIGLU = 0, 1, 2


@dataclass(frozen=True)
class Tactic:
    rows: int  # weight rows per tile (all blocks together)
    step_k: int  # K elements per pipeline stage
    stages: int  # shared-memory ring depth
    warps: int  # consumer warps
    cluster: int  # CTAs per cluster splitting K

    def key(self) -> str:
        return f"r{self.rows}k{self.step_k}s{self.stages}w{self.warps}c{self.cluster}"

    @staticmethod
    def parse(s: str) -> Tactic:
        m = re.fullmatch(r"r(\d+)k(\d+)s(\d+)w(\d+)c(\d+)", s)
        if not m:
            raise ValueError(f"bad tactic {s!r}")
        return Tactic(*(int(v) for v in m.groups()))


# defaults per (epilogue, token tile); tunable (tests/test_draft_gemv.py --bench)
DEFAULTS = {
    EPI_CONV: "r32k256s6w2c2",
    EPI_SWIGLU: "r144k128s5w9c2",
    EPI_PLAIN: "r80k128s8w5c2",
}


def token_tile(m: int) -> int:
    for t in TOKEN_TILES:
        if m <= t:
            return t
    raise ValueError(f"draft_gemv supports M <= {MAX_M}, got {m}")


def smem_bytes(tok_tile: int, t: Tactic) -> int:
    stage = (t.step_k // 64) * (t.rows + tok_tile) * 128
    part = tok_tile * t.rows * 4 if t.cluster > 1 else 0
    return 1024 + t.stages * stage + part + 1024


@functools.lru_cache(maxsize=None)
def _smem_optin(device_index: int) -> int:
    return int(
        torch.cuda.get_device_properties(device_index).shared_memory_per_block_optin
    )


def fits(tok_tile: int, epi: int, device_index: int, tactic=None) -> bool:
    """Whether the tactic's shared-memory ring fits one block on the device (227 KiB on sm_103; the default SwiGLU
    tactic exceeds it at token tiles 32 and 64)."""
    return smem_bytes(tok_tile, _tactic(epi, tactic)) <= _smem_optin(device_index)


@functools.lru_cache(maxsize=None)
def use_pdl() -> bool:
    from sglang.kernels.jit.utils import is_arch_support_pdl

    return bool(is_arch_support_pdl())


@functools.lru_cache(maxsize=None)
def module(tok_tile: int, t: Tactic, boxes: int, epi: int, pdl: bool):
    from sglang.kernels.jit.utils import load_jit, make_cpp_args

    args = make_cpp_args(
        tok_tile, t.rows, t.step_k, t.stages, t.warps, t.cluster, boxes, epi, pdl
    )
    name = {EPI_PLAIN: "run", EPI_CONV: "run_conv", EPI_SWIGLU: "run_swiglu"}[epi]
    return load_jit(
        "muse_vr_draft_gemv",
        *args,
        cuda_files=[str(_CSRC)],
        cuda_wrappers=[(name, f"muse_vr::draft_gemv::DraftGemv<{args}>::{name}")],
    )


def _tactic(epi, tactic):
    if tactic is None:
        tactic = DEFAULTS[epi]
    return Tactic.parse(tactic) if isinstance(tactic, str) else tactic


def conv_in(x, w, base0, tactic=None, conv_out=None, coef_out=None, pdl=None):
    t = _tactic(EPI_CONV, tactic)
    m, h = x.shape
    if conv_out is None:
        conv_out = torch.empty((m, h), dtype=x.dtype, device=x.device)
    if coef_out is None:
        coef_out = torch.empty((m, w.shape[0]), dtype=x.dtype, device=x.device)
    module(token_tile(m), t, 4, EPI_CONV, use_pdl() if pdl is None else pdl).run_conv(
        x, w, base0, conv_out, coef_out
    )
    return conv_out, coef_out


def swiglu(x, w, tactic=None, out=None, pdl=None):
    t = _tactic(EPI_SWIGLU, tactic)
    m = x.shape[0]
    if out is None:
        out = torch.empty((m, w.shape[0] // 2), dtype=x.dtype, device=x.device)
    module(
        token_tile(m), t, 2, EPI_SWIGLU, use_pdl() if pdl is None else pdl
    ).run_swiglu(x, w, out)
    return out


def plain(x, w, tactic=None, out=None, pdl=None):
    t = _tactic(EPI_PLAIN, tactic)
    m = x.shape[0]
    if out is None:
        out = torch.empty((m, w.shape[0]), dtype=x.dtype, device=x.device)
    module(token_tile(m), t, 1, EPI_PLAIN, use_pdl() if pdl is None else pdl).run(
        x, w, out
    )
    return out
