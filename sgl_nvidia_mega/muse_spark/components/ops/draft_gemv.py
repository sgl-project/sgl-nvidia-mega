# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Weight-streaming BF16 GEMV for the DFlash draft layers, with fused epilogues (``csrc/draft_gemv.cuh``).

``conv_in(x, w, base0, tactic)``: the block convolution's coefficient projection (``w`` [4 G, H]) with the input-side
convolution fused: returns (convolved input [M, H], coefficients [M, 4 G] of which the output-side half is written).
``swiglu(x, w, tactic)``: the merged gate/up projection (``w`` [2 I, H]) with SiLU(gate) * up fused: returns [M, I].
``plain(x, w, tactic)``: ``x @ w.T``.
Rounding points match the stock path (bf16 operands, fp32 accumulation, bf16 outputs); the fp32 summation order of the
projection differs. ``MUSE_VR_DRAFT_GEMV_EARLY_TRIGGER=1`` (default off) signals the dependent launch right after the first
stages' weight copies (same outputs). ``MUSE_VR_DRAFT_GEMV_L2PF_{CONV,SWIGLU,PLAIN,PUSH}=<steps>`` (default 0 = off; tunable,
tests/test_draft_gemv.py --bench): before waiting on its predecessor, each CTA also requests the weight boxes of its next <steps>
pipeline steps beyond the ring into L2 (same outputs, bit for bit).
"""

from __future__ import annotations

import functools
import os
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
EPI_PUSH = 3  # plain epilogue + push (module_push)
EPI_QK = (
    4  # QKV projection + QK norm + RoPE + KV cache write (one 128-row head per tile)
)


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
    EPI_QK: "r128k128s4w8c8",
}


# per-epilogue tactic overrides (tunable; tests/test_draft_gemv.py --bench); unset = DEFAULTS. ``<VAR>_T<tile>`` (for example
# MUSE_VR_DRAFT_GEMV_TACTIC_CONV_T8) applies to one token tile only and takes precedence over ``<VAR>``.
ENV_TACTIC = {
    EPI_CONV: "MUSE_VR_DRAFT_GEMV_TACTIC_CONV",
    EPI_SWIGLU: "MUSE_VR_DRAFT_GEMV_TACTIC_SWIGLU",
    EPI_PLAIN: "MUSE_VR_DRAFT_GEMV_TACTIC_PLAIN",
    EPI_QK: "MUSE_VR_DRAFT_GEMV_TACTIC_QK",
}


# per-epilogue L2 lookahead steps (0 = off; tunable, tests/test_draft_gemv.py --bench)
ENV_L2PF = {
    EPI_CONV: "MUSE_VR_DRAFT_GEMV_L2PF_CONV",
    EPI_SWIGLU: "MUSE_VR_DRAFT_GEMV_L2PF_SWIGLU",
    EPI_PLAIN: "MUSE_VR_DRAFT_GEMV_L2PF_PLAIN",
    EPI_PUSH: "MUSE_VR_DRAFT_GEMV_L2PF_PUSH",
    EPI_QK: "MUSE_VR_DRAFT_GEMV_L2PF_QK",
}


@functools.lru_cache(maxsize=None)
def l2pf(epi: int) -> int:
    raw = os.environ.get(ENV_L2PF[epi], "0").strip() or "0"
    v = int(raw)
    if not 0 <= v <= 256:
        raise ValueError(f"{ENV_L2PF[epi]}={raw}: expected 0..256")
    return v


def _cpp_args(base, early, pf, epi_pf=False):
    from sglang.kernels.jit.utils import make_cpp_args

    if epi_pf:
        return make_cpp_args(*base, bool(early), int(pf), True)
    if pf:
        return make_cpp_args(*base, bool(early), int(pf))
    if early:
        return make_cpp_args(*base, True)
    return make_cpp_args(*base)


def default_tactic(epi: int, tok_tile: int | None = None) -> str:
    if tok_tile is not None:
        per_tile = os.environ.get(f"{ENV_TACTIC[epi]}_T{tok_tile}")
        if per_tile:
            return per_tile
    return os.environ.get(ENV_TACTIC[epi]) or DEFAULTS[epi]


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
def conv_prefetch() -> bool:
    """``MUSE_VR_DRAFT_GEMV_CONV_PF=1`` (default off): the conv GEMV's epilogue CTA reads its activation vectors right after
    the grid dependency wait, before the MMA loop, instead of at the epilogue (same values, same outputs bit for bit)."""
    return os.environ.get("MUSE_VR_DRAFT_GEMV_CONV_PF", "0").strip() not in ("", "0")


@functools.lru_cache(maxsize=None)
def qk_prefetch() -> bool:
    """``MUSE_VR_DRAFT_GEMV_QK_PF=1`` (default off): the QKV-epilogue GEMV reads each token's RoPE table entries (and key
    cache row) right after the grid dependency wait, before the MMA loop, instead of at the epilogue (same values, same
    outputs bit for bit)."""
    return os.environ.get("MUSE_VR_DRAFT_GEMV_QK_PF", "0").strip() not in ("", "0")


@functools.lru_cache(maxsize=None)
def early_trigger() -> bool:
    """``MUSE_VR_DRAFT_GEMV_EARLY_TRIGGER=1`` (default off): signal the dependent launch right after the first stages' weight
    copies instead of after the last copy (same outputs, bit for bit)."""
    return os.environ.get("MUSE_VR_DRAFT_GEMV_EARLY_TRIGGER", "0").strip() not in (
        "",
        "0",
    )


@functools.lru_cache(maxsize=None)
def module(
    tok_tile: int,
    t: Tactic,
    boxes: int,
    epi: int,
    pdl: bool,
    early: bool = False,
    pf: int = 0,
    epi_pf: bool = False,
):
    from sglang.kernels.jit.utils import load_jit

    args = _cpp_args(
        (tok_tile, t.rows, t.step_k, t.stages, t.warps, t.cluster, boxes, epi, pdl),
        early,
        pf,
        epi_pf and epi in (EPI_CONV, EPI_QK),
    )
    name = {
        EPI_PLAIN: "run",
        EPI_CONV: "run_conv",
        EPI_SWIGLU: "run_swiglu",
        EPI_QK: "run_qk",
    }[epi]
    unfused = epi == EPI_QK and _qk_unfused()
    return load_jit(
        "muse_vr_draft_gemv" + ("_qku" if unfused else ""),
        *args,
        cuda_files=[str(_CSRC)],
        cuda_wrappers=[(name, f"muse_vr::draft_gemv::DraftGemv<{args}>::{name}")],
        extra_cuda_cflags=["-DMUSE_VR_QK_UNFUSED=1"] if unfused else None,
    )


def _qk_unfused() -> bool:
    """The QK epilogue's unfused arithmetic form (ops/swa_verify_attn.py, qk_arith)."""
    from .swa_verify_attn import qk_arith

    return qk_arith() == "unfused"


def _tactic(epi, tactic, m=None):
    if tactic is None:
        tactic = default_tactic(epi, None if m is None else token_tile(m))
    return Tactic.parse(tactic) if isinstance(tactic, str) else tactic


_CSRC_PUSH = Path(__file__).resolve().parent.parent / "csrc" / "draft_gemv_push.cuh"


@functools.lru_cache(maxsize=None)
def push_mc() -> bool:
    """``MUSE_VR_DRAFT_PUSH_MC=1`` (default off; with the push epilogue): the push GEMV stores every output vector once
    through the push plane's multicast address instead of once per peer, and the consumer-only boundary polls its own slot
    too (csrc/draft_gemv_push.cuh kMcPush). Same vectors in the same slots: outputs bit for bit."""
    return os.environ.get("MUSE_VR_DRAFT_PUSH_MC", "0").strip() not in ("", "0")


@functools.lru_cache(maxsize=None)
def module_push(
    tok_tile: int,
    t: Tactic,
    world: int,
    pdl: bool,
    early: bool = False,
    pf: int = 0,
    mc: bool = False,
):
    from sglang.kernels.jit.utils import load_jit, make_cpp_args

    base = (tok_tile, t.rows, t.step_k, t.stages, t.warps, t.cluster, world, pdl)
    args = (
        make_cpp_args(*base, bool(early), int(pf), True)
        if mc
        else _cpp_args(base, early, pf)
    )
    return load_jit(
        "muse_vr_draft_gemv_push",
        *args,
        cuda_files=[str(_CSRC_PUSH)],
        cuda_wrappers=[("run", f"sglang::MuseVrDraftGemvPush<{args}>::run")],
    )


def push(
    x, w, comm, consumer_cluster, tactic=None, out=None, pdl=None, pf=None, mc=None
):
    """``x @ w.T`` (plain epilogue) whose output vectors are also pushed into the peers' one-shot push slots, where the draft
    layer boundary (ops.draft_ar_norm with consumer_only=True) reads them. ``comm``: the group's CustomAllReduceV2.
    ``mc`` (None = ``push_mc()``): one store per vector through the multicast address; the boundary must then be called
    with ``producer_mc=True``."""
    t = _tactic(EPI_PLAIN, tactic)
    m = x.shape[0]
    if out is None:
        out = torch.empty((m, w.shape[0]), dtype=x.dtype, device=x.device)
    pf = l2pf(EPI_PUSH) if pf is None else pf
    mc = push_mc() if mc is None else bool(mc)
    module_push(
        token_tile(m),
        t,
        int(comm.world_size),
        use_pdl() if pdl is None else pdl,
        early_trigger(),
        pf,
        mc,
    ).run(comm.obj, x, w, out, int(consumer_cluster))
    return out


def conv_in(
    x,
    w,
    base0,
    tactic=None,
    conv_out=None,
    coef_out=None,
    pdl=None,
    pf=None,
    epi_pf=None,
):
    m, h = x.shape
    t = _tactic(EPI_CONV, tactic, m)
    if conv_out is None:
        conv_out = torch.empty((m, h), dtype=x.dtype, device=x.device)
    if coef_out is None:
        coef_out = torch.empty((m, w.shape[0]), dtype=x.dtype, device=x.device)
    pf = l2pf(EPI_CONV) if pf is None else pf
    epi_pf = conv_prefetch() if epi_pf is None else epi_pf
    mod = module(
        token_tile(m),
        t,
        4,
        EPI_CONV,
        use_pdl() if pdl is None else pdl,
        early_trigger(),
        pf,
        bool(epi_pf),
    )
    mod.run_conv(x, w, base0, conv_out, coef_out)
    return conv_out, coef_out


def swiglu(x, w, tactic=None, out=None, pdl=None, pf=None):
    m = x.shape[0]
    t = _tactic(EPI_SWIGLU, tactic, m)
    if out is None:
        out = torch.empty((m, w.shape[0] // 2), dtype=x.dtype, device=x.device)
    pf = l2pf(EPI_SWIGLU) if pf is None else pf
    mod = module(
        token_tile(m),
        t,
        2,
        EPI_SWIGLU,
        use_pdl() if pdl is None else pdl,
        early_trigger(),
        pf,
    )
    mod.run_swiglu(x, w, out)
    return out


def plain(x, w, tactic=None, out=None, pdl=None, pf=None):
    m = x.shape[0]
    t = _tactic(EPI_PLAIN, tactic, m)
    if out is None:
        out = torch.empty((m, w.shape[0]), dtype=x.dtype, device=x.device)
    pf = l2pf(EPI_PLAIN) if pf is None else pf
    module(
        token_tile(m),
        t,
        1,
        EPI_PLAIN,
        use_pdl() if pdl is None else pdl,
        early_trigger(),
        pf,
    ).run(x, w, out)
    return out


# Which half-warp's partial sum of squares the QK norm + RoPE kernel's reduction keeps (tests/test_draft_gemv.py --qk checks
# both against that kernel and reports the matching one).
QK_HALF = 1


def qk_half() -> int:
    return bed_env_int("MUSE_VR_DRAFT_GEMV_QK_HALF", QK_HALF)


def bed_env_int(var: str, default: int) -> int:
    raw = os.environ.get(var, "").strip()
    return int(raw) if raw else default


def qkv_qk(
    x,
    w,
    q_out,
    k_cache,
    v_cache,
    loc,
    pos,
    cos_sin,
    q_w,
    k_w,
    eps,
    tactic=None,
    pdl=None,
    pf=None,
    half=None,
    epi_pf=None,
):
    """The attention QKV projection (``w`` [(nhq + 2 nhk) * 128, K]: query, key, value heads) with the draft's QK norm + RoPE
    (``ops/draft_qk``'s arithmetic) and the KV cache write fused: rotated query heads to ``q_out`` [M, nhq, 128], rotated key
    heads and value heads to the cache rows ``loc`` of ``k_cache`` / ``v_cache`` [slots, nhk, 128]. The QKV tensor is not
    materialized. The projection's fp32 summation order is the GEMV's (tactic with 128-row tiles)."""
    t = _tactic(EPI_QK, tactic, x.shape[0])
    if t.rows != 128:
        raise ValueError(
            f"qkv_qk needs 128-row tiles (one head per tile), got {t.key()}"
        )
    pf = l2pf(EPI_QK) if pf is None else pf
    epi_pf = qk_prefetch() if epi_pf is None else epi_pf
    mod = module(
        token_tile(x.shape[0]),
        t,
        1,
        EPI_QK,
        use_pdl() if pdl is None else pdl,
        early_trigger(),
        pf,
        bool(epi_pf),
    )
    mod.run_qk(
        x,
        w,
        q_out,
        k_cache,
        v_cache,
        loc,
        pos,
        cos_sin,
        q_w,
        k_w,
        float(eps),
        int(qk_half() if half is None else half),
    )
    return q_out
