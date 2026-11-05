# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""FR: the split (RS) form's front launch (csrc/wfc_c4_fr.cuh): development form, not wired into the server.

From the library router logits and the library pooled rows of the owned rows, one launch routes each row (ids published
ids-first as tag << 32 | packed words) and norm + NVFP4-quantizes it into a 2320-byte token row published as 48-bit payload
words with a 16-bit tag. This module builds the launch with a local (one-GPU) transport for tests.
"""

from __future__ import annotations

import functools
from pathlib import Path

import torch

# The SGLang names this module reads (test/unit/test_contract.py pins them against the supported SGLang)
SGLANG_READS = {
    "kernels/jit/utils/__init__.py": ("load_jit", "make_cpp_args"),
    "kernels/jit/utils/compile/paths.py": ("KERNEL_PATH",),
}


_CSRC = Path(__file__).resolve().parents[1] / "csrc"
_CUH = _CSRC / "wfc_c4_fr.cuh"
ROW_BYTES, ROW_WORDS = 2320, 387


def route_form() -> int:
    """Routing form of every FR build (default-off knob MUSE_VR_C4FR_ROUTE_SEL): 0 = route_row (default); 1 / 2 =
    route_row_sel, 3 = route_row_sel_sm (wfc_c4_route.cuh; selection rounds without the winner broadcast on their
    dependency chain; outputs equal route_row's for every input). Read when a module is first built."""
    import os

    v = os.environ.get("MUSE_VR_C4FR_ROUTE_SEL", "0").strip() or "0"
    if v not in ("0", "1", "2", "3"):
        raise ValueError(f"MUSE_VR_C4FR_ROUTE_SEL must be 0, 1, 2 or 3 (got {v!r})")
    return int(v)


def push_form() -> int:
    """Token-word assembly of every FR build (default-off knob MUSE_VR_C4FR_PUSH_VEC): 0 = byte loads (default), 1 = two
    aligned 8-byte shared loads per word (same words)."""
    import os

    v = os.environ.get("MUSE_VR_C4FR_PUSH_VEC", "0").strip() or "0"
    if v not in ("0", "1"):
        raise ValueError(f"MUSE_VR_C4FR_PUSH_VEC must be 0 or 1 (got {v!r})")
    return int(v)


def quant_prew() -> int:
    """Quant group of every FR build (default-off knob MUSE_VR_C4FR_QUANT_PREW): 1 = the norm weight's loads issued with the
    row's loads (same bytes)."""
    import os

    v = os.environ.get("MUSE_VR_C4FR_QUANT_PREW", "0").strip() or "0"
    if v not in ("0", "1"):
        raise ValueError(f"MUSE_VR_C4FR_QUANT_PREW must be 0 or 1 (got {v!r})")
    return int(v)


def _route_flags():
    f, pv, qw = route_form(), push_form(), quant_prew()
    flags = ([f"-DMUSE_VR_C4FR_ROUTE_SEL={f}"] if f else []) + (
        ["-DMUSE_VR_C4FR_PUSH_VEC=1"] if pv else []
    )
    flags += ["-DMUSE_VR_C4FR_QUANT_PREW=1"] if qw else []
    return flags, (f"_rs{f}" if f else "") + ("_pv" if pv else "") + (
        "_qw" if qw else ""
    )


@functools.lru_cache(maxsize=None)
def module():
    import os

    from sglang.kernels.jit.utils import load_jit
    from sglang.kernels.jit.utils.compile.paths import KERNEL_PATH

    from sgl_nvidia_mega.muse_spark.components.sgl_compat import csrc_include_paths

    plant = (
        os.environ.get("MUSE_VR_WFC_C4FR_TEST_PLANT", "") == "1"
    )  # tests only: the planted-race control build
    nofb = (
        os.environ.get("MUSE_VR_WFC_C4FR_TEST_NOFB", "") == "1"
    )  # tests only: route_row_sel without its recompute
    rf, rn = _route_flags()
    flags = (["-DMUSE_VR_C4FR_TEST_PLANT_RACE"] if plant else []) + rf
    flags += ["-DMUSE_VR_C4_ROUTE_TEST_NO_FALLBACK"] if nofb else []
    extra = {"extra_cuda_cflags": flags} if flags else {}
    return load_jit(
        "muse_vr_wfc_c4_fr"
        + ("_plant" if plant else "")
        + ("_nofb" if nofb else "")
        + rn,
        cuda_files=[str(_CUH)],
        cuda_wrappers=[("run", "WfcC4Fr::run"), ("ready", "WfcC4Fr::ready")],
        extra_include_paths=csrc_include_paths(KERNEL_PATH / "csrc") + [str(_CSRC)],
        **extra,
    )


class State:
    def __init__(self, R: int, device):
        d = device
        self.R = R
        self.topk_w = torch.zeros(R, 8, dtype=torch.float32, device=d)
        self.topk_ids = torch.zeros(R, 8, dtype=torch.int32, device=d)
        self.packed = torch.zeros(R, 8, dtype=torch.int32, device=d)
        self.tag_word = torch.zeros(1, dtype=torch.int64, device=d)
        self.ids_board = torch.zeros(R, 8, dtype=torch.int64, device=d)
        self.tok_board = torch.zeros(R, ROW_WORDS + 1, dtype=torch.int64, device=d)

    def run(
        self,
        logits,
        bias,
        pooled,
        qweight=None,
        *,
        mult=1.0,
        route_scale=1.0,
        div_eps=0.0,
        qeps=1e-6,
        gsi=1.0 / (448.0 * 6.0),
        pdl=False,
        lready=None,
        err=None,
        lready_max=0,
    ):
        """pdl: launch with programmatic stream serialization (every producer-written read after griddepcontrol.wait).
        lready: an int64 [1] logits ready word (written by ready() after the router's producer); the routing warp waits
        for (*tag_word) + 1 there, at most lready_max polls (0 = default), then sets bit 2 of err ([1] int32) and
        publishes no ids words."""
        module().run(
            logits.contiguous(),
            bias.contiguous(),
            pooled.contiguous(),
            qweight if qweight is not None else pooled[0],
            int(qweight is not None),
            self.topk_w,
            self.topk_ids,
            self.packed,
            self.tag_word,
            self.ids_board,
            self.tok_board,
            self.R,
            float(mult),
            float(route_scale),
            float(div_eps),
            float(qeps),
            float(gsi),
            int(bool(pdl)),
            lready,
            err,
            int(lready_max),
        )

    def ready(self, word):
        """The ready word's producer side: one thread stores (*tag_word) + 1 into word with release semantics."""
        module().ready(word, self.tag_word)


# ------------------------------------------------------------------------------------------------ balance CTA (development)
NUM_STAMPS = 8 + 5 * 1024


def _stamps_dir() -> str:
    """Timing builds only: a directory holding the (private) stamp header; unset = production build, no stamps."""
    import os

    return os.environ.get("MUSE_VR_C4FR_STAMPS_DIR", "").strip()


@functools.lru_cache(maxsize=None)
def module_bal():
    from sglang.kernels.jit.utils import load_jit
    from sglang.kernels.jit.utils.compile.paths import KERNEL_PATH

    from sgl_nvidia_mega.muse_spark.components.sgl_compat import csrc_include_paths

    sd = _stamps_dir()
    rf, rn = _route_flags()
    flags = (["-DMUSE_VR_C4FR_STAMPS"] if sd else []) + rf
    extra = {"extra_cuda_cflags": flags} if flags else {}
    return load_jit(
        "muse_vr_wfc_c4_fr_bal" + ("_stamps" if sd else "") + rn,
        cuda_files=[str(_CUH)],
        cuda_wrappers=[("run", "WfcC4FrBal::run")],
        extra_include_paths=csrc_include_paths(KERNEL_PATH / "csrc")
        + [str(_CSRC)]
        + ([sd] if sd else []),
        **extra,
    )


class StateBal:
    """FR + the balance CTA with the local transport: the ids board holds all M rows of the step (this launch's rows at
    [own0, own0 + R)); the balance CTA publishes 33 assignment words into `board`."""

    def __init__(self, R: int, M: int, own0: int, world: int, placement, device):
        from sgl_nvidia_mega.muse_spark.components import bal

        d = device
        self.R, self.M, self.own0, self.world = R, M, own0, world
        self.topk_w = torch.zeros(R, 8, dtype=torch.float32, device=d)
        self.topk_ids = torch.zeros(R, 8, dtype=torch.int32, device=d)
        self.packed = torch.zeros(R, 8, dtype=torch.int32, device=d)
        self.tag_word = torch.zeros(1, dtype=torch.int64, device=d)
        self.ids_board = torch.zeros(M, 8, dtype=torch.int64, device=d)
        self.tok_board = torch.zeros(R, ROW_WORDS + 1, dtype=torch.int64, device=d)
        self.tbl = bal.pw_table(placement, d)
        self.maxm = bal.pw_max_members(placement)
        self.board = torch.zeros(bal.PW_WORDS, dtype=torch.int64, device=d)
        self.err = torch.zeros(1, dtype=torch.int32, device=d)
        self.stamps = (
            torch.zeros(NUM_STAMPS, dtype=torch.int64, device=d)
            if _stamps_dir()
            else None
        )

    def run(
        self,
        logits,
        bias,
        pooled,
        qweight=None,
        *,
        pass_rows=48,
        sweeps=2,
        mult=1.0,
        route_scale=1.0,
        div_eps=0.0,
        qeps=1e-6,
        gsi=1.0 / (448.0 * 6.0),
        spin_rounds=64,
        max_rounds=1 << 22,
        backoff_ns=32,
        bal_cta=True,
        variant=2,
        coop=True,
    ):
        module_bal().run(
            logits.contiguous(),
            bias.contiguous(),
            pooled.contiguous(),
            qweight if qweight is not None else pooled[0],
            int(qweight is not None),
            self.topk_w,
            self.topk_ids,
            self.packed,
            self.tag_word,
            self.ids_board,
            self.tok_board,
            self.R,
            self.own0,
            float(mult),
            float(route_scale),
            float(div_eps),
            float(qeps),
            float(gsi),
            self.tbl,
            self.board,
            self.err,
            self.stamps,
            self.M,
            self.world,
            int(sweeps),
            int(pass_rows),
            int(spin_rounds),
            int(max_rounds),
            int(backoff_ns),
            int(bool(bal_cta)),
            int(variant),
            int(bool(coop)),
            self.maxm,
        )


# ------------------------------------------------------------------------------------------------ the hop roles' transport
IDS_WIDTH = 24  # hop-site elements per ids row: 8 words
TOK_WIDTH = 1164  # hop-site elements per token row: 388 words >= ROW_WORDS (pitch 400)


def hop_available() -> bool:
    """True when this build carries the hop roles' headers (c3_hops.cuh, lb_hop_ops.cuh) the hop transport needs."""
    return (_CSRC / "c3_hops.cuh").exists() and (_CSRC / "lb_hop_ops.cuh").exists()


def site_specs(
    rows_max: int,
    ids_name: str = "fr_ids",
    tok_name: str = "fr_tok",
    parity: bool = False,
):
    """HopSpecs of FR's ids site and token site (unique; rows are global, so rows_max >= the step's M)."""
    from .lb_hop import HopSpec

    return (
        HopSpec(
            ids_name,
            rows_max=int(rows_max),
            width=IDS_WIDTH,
            policy="unique",
            parity=bool(parity),
        ),
        HopSpec(
            tok_name,
            rows_max=int(rows_max),
            width=TOK_WIDTH,
            policy="unique",
            parity=bool(parity),
        ),
    )


@functools.lru_cache(maxsize=None)
def module_c3(world: int):
    from sglang.kernels.jit.utils import load_jit, make_cpp_args
    from sglang.kernels.jit.utils.compile.paths import KERNEL_PATH

    from sgl_nvidia_mega.muse_spark.components.sgl_compat import csrc_include_paths

    if not hop_available():
        raise RuntimeError(
            "wfc_c4_fr: the hop transport needs c3_hops.cuh and lb_hop_ops.cuh in this build"
        )
    args = make_cpp_args(int(world))
    rf, rn = _route_flags()
    extra = {"extra_cuda_cflags": rf} if rf else {}
    return load_jit(
        f"muse_vr_wfc_c4_fr_c3_w{int(world)}" + rn,
        *args,
        cuda_files=[str(_CUH)],
        cuda_wrappers=[("run", f"WfcC4FrC3<{args}>::run")],
        extra_include_paths=csrc_include_paths(KERNEL_PATH / "csrc") + [str(_CSRC)],
        **extra,
    )


def run_c3(
    arena,
    ids_name: str,
    tok_name: str,
    M: int,
    logits,
    bias,
    pooled,
    topk_w,
    topk_ids,
    packed,
    qweight=None,
    *,
    acquire: bool = False,
    mult=1.0,
    route_scale=1.0,
    div_eps=0.0,
    qeps=1e-6,
    gsi=1.0 / (448.0 * 6.0),
    pdl=False,
    lready=None,
    err=None,
    lready_max=0,
):
    """FR for this rank's owned rows of a step of M rows, publishing through the hop arena's ids and token sites (rows by
    GLOBAL index). logits [R, 256] fp32 and pooled [R, 4096] bf16 hold the owned rows (R = own1 - own0); topk_w / topk_ids
    / packed [R, 8] receive the local routing outputs. acquire: every CTA acquires the forward's step tokens itself (else
    the caller orders FR after a kernel of the same forward that did)."""
    rank, i_rows, i_width, i_pitch, i_policy, i_parity, _, i_site, _ = arena.site_args(
        ids_name
    )
    _, t_rows, t_width, t_pitch, t_policy, t_parity, _, t_site, _ = arena.site_args(
        tok_name
    )
    module_c3(arena.world).run(
        logits.contiguous(),
        bias.contiguous(),
        pooled.contiguous(),
        qweight if qweight is not None else pooled[0],
        int(qweight is not None),
        topk_w,
        topk_ids,
        packed,
        arena.push_table(ids_name),
        arena.push_table(tok_name),
        arena.tok_table(),
        arena.ctl_table(),
        rank,
        i_rows,
        i_width,
        i_pitch,
        i_policy,
        i_parity,
        i_site,
        t_rows,
        t_width,
        t_pitch,
        t_policy,
        t_parity,
        t_site,
        *arena.ctl_args(),
        int(bool(acquire)),
        int(M),
        float(mult),
        float(route_scale),
        float(div_eps),
        float(qeps),
        float(gsi),
        int(bool(pdl)),
        lready,
        err,
        int(lready_max),
    )


@functools.lru_cache(maxsize=None)
def module_c3_bal(world: int):
    from sglang.kernels.jit.utils import load_jit, make_cpp_args
    from sglang.kernels.jit.utils.compile.paths import KERNEL_PATH

    from sgl_nvidia_mega.muse_spark.components.sgl_compat import csrc_include_paths

    if not hop_available():
        raise RuntimeError(
            "wfc_c4_fr: the hop transport needs c3_hops.cuh and lb_hop_ops.cuh in this build"
        )
    args = make_cpp_args(int(world))
    rf, rn = _route_flags()
    extra = {"extra_cuda_cflags": rf} if rf else {}
    return load_jit(
        f"muse_vr_wfc_c4_fr_c3bal_w{int(world)}" + rn,
        *args,
        cuda_files=[str(_CUH)],
        cuda_wrappers=[("run", f"WfcC4FrC3Bal<{args}>::run")],
        extra_include_paths=csrc_include_paths(KERNEL_PATH / "csrc") + [str(_CSRC)],
        **extra,
    )


def run_c3_bal(
    arena,
    ids_name: str,
    tok_name: str,
    M: int,
    logits,
    bias,
    pooled,
    topk_w,
    topk_ids,
    packed,
    bal_tbl,
    bal_board,
    bal_err,
    qweight=None,
    *,
    acquire: bool = False,
    pass_rows: int = 48,
    sweeps: int = 2,
    bal_spin_rounds: int = 64,
    bal_max_rounds: int = 1 << 22,
    bal_backoff_ns: int = 32,
    bal_world=None,
    maxm=16,
    mult=1.0,
    route_scale=1.0,
    div_eps=0.0,
    qeps=1e-6,
    gsi=1.0 / (448.0 * 6.0),
):
    """run_c3 plus the balance CTA: this GPU's assignment words (bal_pw.cuh) into bal_board [33] int64. bal_world: the
    assignment's world (= the arena's world in served use; a world-1 arena may emulate 4 or 8 in tests)."""
    rank, i_rows, i_width, i_pitch, i_policy, i_parity, _, i_site, _ = arena.site_args(
        ids_name
    )
    _, t_rows, t_width, t_pitch, t_policy, t_parity, _, t_site, _ = arena.site_args(
        tok_name
    )
    module_c3_bal(arena.world).run(
        logits.contiguous(),
        bias.contiguous(),
        pooled.contiguous(),
        qweight if qweight is not None else pooled[0],
        int(qweight is not None),
        topk_w,
        topk_ids,
        packed,
        arena.push_table(ids_name),
        arena.push_table(tok_name),
        arena.tok_table(),
        arena.ctl_table(),
        rank,
        i_rows,
        i_width,
        i_pitch,
        i_policy,
        i_parity,
        i_site,
        t_rows,
        t_width,
        t_pitch,
        t_policy,
        t_parity,
        t_site,
        *arena.ctl_args(),
        int(bool(acquire)),
        int(M),
        float(mult),
        float(route_scale),
        float(div_eps),
        float(qeps),
        float(gsi),
        bal_tbl,
        bal_board,
        bal_err,
        int(sweeps),
        int(pass_rows),
        int(bal_spin_rounds),
        int(bal_max_rounds),
        int(bal_backoff_ns),
        int(bal_world or arena.world),
        int(maxm),
    )
