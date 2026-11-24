# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Hop roles on the hop device API v2 (lb_hop): the owner-row reduce-scatter as one persistent launch.

Default off: nothing in the plugin calls this module unless a caller selects it with its own default-off knob.

Rows: rank r owns the contiguous rows [floor(M r / W), floor(M (r + 1) / W)). ``rs_rows`` reduces every rank's bf16 partial
[M, H] to the owned rows: out[i] = bf16(fp32 fold over source ranks 0..W-1 in order) (first source's value, then one rounded
add per source). With ``mask`` (int32 [M], bit r = rank r holds a partial for the row) the fold adds +0.0 at the position of
every absent rank, and ranks skip rows they hold no partial for (``skip``, uint8 [M]).

The site is a unique lb_hop site whose width is H rounded up to a multiple of 6 and whose unit count is the launch's grid
(one missed-advance word per CTA); size the grid to one resident wave (every CTA pushes before any CTA waits on another
rank).
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Optional

import torch

from .lb_hop import HopArena, HopSpec

# The SGLang names this module reads (test/unit/test_contract.py pins them against the supported SGLang)
SGLANG_READS = {
    "kernels/jit/utils/__init__.py": (
        "is_arch_support_pdl",
        "load_jit",
        "make_cpp_args",
    ),
}


_CSRC = Path(__file__).resolve().parent.parent / "csrc"
_CUH = _CSRC / "c3_hops_ops.cuh"

STEP_RELEASE = 1
STEP_ACQUIRE = 2
COLLECT_ONLY = (
    4  # the partials were pushed by another kernel (e.g. an expert epilogue): fold only
)
STOCK_ZERO_SIGN = (
    8  # zero-sign parity with the engine's push all-reduce (see stock_zero_sign_flag)
)
RS_FLAT = 16  # launch the flat-grid rs_rows form (every CTA pushes and folds; same protocol and arithmetic)
EARLY_TRIGGER = 32  # trigger PDL dependents at entry + max shared-memory carveout (rs_rows step flag; ag_rows early_trigger)
SPLIT_ROLES = 64  # RS_FLAT rs_rows / ag_rows_tagged: half of each CTA's warps push while the other half folds / gathers
PUSH_BEFORE_GUARD = 128  # RS_FLAT rs_rows / ag_rows_tagged without STEP_ACQUIRE: push before the guard barrier
ROW_FOLD = 256  # RS_FLAT rs_rows: fold each owned row inside one CTA (the form a fused per-row epilogue needs)


def stock_zero_sign_flag(comm, nbytes: int, graph: bool = True) -> int:
    """STOCK_ZERO_SIGN when the engine's custom all-reduce ``comm`` would reduce a tensor of ``nbytes`` with a push algorithm
    (``graph`` selects its CUDA-graph or its eager size policy; served decode steps run captured).

    Those algorithms send an all-zero 32-bit word (a bf16 pair (+0.0, +0.0)) as (-0.0, +0.0); with this flag rs_rows applies the
    same rewrite (and uses (-0.0, +0.0) for an absent source in the masked fold), so a column that sums to zero keeps the sign the
    engine's all-reduce gives it. Returns 0 when ``comm`` is absent / disabled or picks another algorithm at this size."""
    nbytes = int(nbytes)
    if comm is None or getattr(comm, "disabled", True) or nbytes <= 0 or nbytes % 16:
        return 0
    override = getattr(comm, "override_algo", None)
    pick = getattr(comm, "_pick_config", None)
    if override is not None:
        algo = override
    elif pick is not None:
        cfg = pick(nbytes, bool(graph))
        algo = None if cfg is None else cfg.algo
    else:  # older communicator: its own query (current capture state decides graph vs eager)
        push = any(
            getattr(comm, n, lambda _b: False)(nbytes)
            for n in ("uses_one_shot_push", "uses_two_shot_push")
        )
        return STOCK_ZERO_SIGN if push else 0
    name = getattr(algo, "name", str(algo)).upper()
    return STOCK_ZERO_SIGN if (algo is not None and "PUSH" in name) else 0


def rs_width(hidden: int) -> int:
    """Site width for rows of ``hidden`` bf16 elements (a multiple of 6)."""
    return ((int(hidden) + 5) // 6) * 6


def rs_spec(
    name: str, rows_max: int, hidden: int, ctas: int, parity: bool = False
) -> HopSpec:
    """A unique site for ``rs_rows`` with ``ctas`` consumer units (= the launch grid)."""
    return HopSpec(
        name,
        rows_max=int(rows_max),
        width=rs_width(hidden),
        policy="unique",
        parity=bool(parity),
        units=int(ctas),
    )


@lru_cache(maxsize=None)
def _module(world: int, use_pdl: bool):
    from sglang.kernels.jit.utils import is_arch_support_pdl, load_jit, make_cpp_args

    pdl = bool(use_pdl) and is_arch_support_pdl()
    args = make_cpp_args(world, pdl)
    return load_jit(
        f"muse_vr_c3_hops_w{world}",
        *args,
        cuda_files=[str(_CUH)],
        cuda_wrappers=[
            ("rows", f"MuseVrC3Rs<{args}>::rows"),
            ("agt", f"MuseVrC3Rs<{args}>::agt"),
            ("fr_unpack", f"MuseVrC3Rs<{args}>::fr_unpack"),
        ],
        extra_include_paths=[str(_CSRC)],
    )


_PLACEHOLDERS: dict = {}


def _placeholders(dev: torch.device):
    """One-element tensors standing in for absent optional arguments (allocated once per device, never per call)."""
    key = str(dev)
    if key not in _PLACEHOLDERS:
        _PLACEHOLDERS[key] = (
            torch.zeros(1, dtype=torch.uint8, device=dev),
            torch.zeros(1, dtype=torch.int32, device=dev),
        )
    return _PLACEHOLDERS[key]


def rs_rows(
    arena: HopArena,
    name: str,
    part: torch.Tensor,
    out: torch.Tensor,
    *,
    mask: Optional[torch.Tensor] = None,
    skip: Optional[torch.Tensor] = None,
    step_flags: int = 0,
    use_pdl: bool = True,
    rows: Optional[int] = None,
) -> None:
    """Owner-row reduce-scatter of ``part`` [M, H] (bf16) into ``out`` [own_n, H] (bf16) on site ``name``. With
    COLLECT_ONLY, ``part`` may be any [>= 1, H] bf16 tensor and ``rows`` gives M."""
    M, H = (int(rows) if rows is not None else int(part.shape[0])), int(part.shape[1])
    if not (step_flags & COLLECT_ONLY) and M != int(part.shape[0]):
        raise ValueError("rs_rows: rows must equal part.shape[0] unless COLLECT_ONLY")
    if part.dtype != torch.bfloat16 or out.dtype != torch.bfloat16:
        raise ValueError("rs_rows: bf16 partial and output")
    own0, own1 = (M * arena.rank) // arena.world, (M * (arena.rank + 1)) // arena.world
    if out.shape[0] < own1 - own0 or out.shape[1] != H:
        raise ValueError("rs_rows: out must hold the owned rows [own_n, H]")
    none_u8, none_i32 = _placeholders(part.device)
    _module(arena.world, use_pdl).rows(
        arena.table(name),
        arena.ctl_table(),
        arena.tok_table(),
        *arena.site_args(name),
        *arena.ctl_args(),
        int(step_flags),
        part,
        out,
        M,
        skip if skip is not None else none_u8,
        1 if skip is not None else 0,
        mask if mask is not None else none_i32,
        1 if mask is not None else 0,
    )


# ------------------------------------------------------------------------------------------------------------ AG_x
@lru_cache(maxsize=None)
def _ag_module(world: int, use_pdl: bool):
    from sglang.kernels.jit.utils import is_arch_support_pdl, load_jit, make_cpp_args

    pdl = bool(use_pdl) and is_arch_support_pdl()
    args = make_cpp_args(world, pdl)
    return load_jit(
        f"muse_vr_c3_ag_w{world}",
        *args,
        cuda_files=[str(_CUH)],
        cuda_wrappers=[("rows", f"MuseVrC3Ag<{args}>::rows")],
        extra_include_paths=[str(_CSRC)],
    )


class AgxLanding:
    """Plugin-owned landing of the owner-row all-gather: ``copies`` images of [rows_max, hidden] bf16 (two selected by layer
    parity for the layer input; one per tap layer for captured residuals) plus one 64-bit flag per (source rank, CTA), in one
    symmetric allocation with a multicast address. ``view(copy)`` is this rank's unicast view of a copy (what a library
    kernel or a capture reads)."""

    def __init__(
        self,
        group,
        device: torch.device,
        rank: int,
        world: int,
        rows_max: int,
        hidden: int,
        ctas_max: int,
        copies: int = 2,
        allocator=None,
    ):
        from .lb_hop import allocate_symmetric

        if (hidden * 2) % 16:
            raise ValueError("AgxLanding: row bytes must be a multiple of 16")
        self.rank, self.world, self.rows_max, self.hidden, self.ctas_max = (
            rank,
            world,
            rows_max,
            hidden,
            ctas_max,
        )
        self.copies = int(copies)
        data = self.copies * rows_max * hidden * 2
        self._flag_off = (data + 4095) // 4096 * 4096
        total = self._flag_off + world * ctas_max * 8
        self._tensor, symm = (allocator or allocate_symmetric)(total, device, group)
        mc = int(getattr(symm, "multicast_ptr", 0) or 0)
        if mc == 0:
            raise RuntimeError("AgxLanding: the allocation has no multicast address")
        uc = symm.get_buffer(rank, [total], torch.uint8)
        uc.zero_()
        torch.cuda.synchronize(device)
        self._uc = uc
        self._last = torch.zeros(
            ctas_max, dtype=torch.int64, device=device
        )  # missed-advance guard, this rank only
        self._table = torch.tensor(
            [
                mc,
                uc.data_ptr(),
                mc + self._flag_off,
                uc.data_ptr() + self._flag_off,
                self._last.data_ptr(),
            ],
            dtype=torch.int64,
            device="cpu",
        )

    def view(self, copy: int) -> torch.Tensor:
        n = self.rows_max * self.hidden * 2
        return (
            self._uc[copy * n : (copy + 1) * n]
            .view(torch.bfloat16)
            .view(self.rows_max, self.hidden)
        )

    def table(self) -> torch.Tensor:
        return self._table


def ag_rows(
    landing: AgxLanding,
    arena: HopArena,
    x_owned: torch.Tensor,
    M: int,
    copy: int,
    layer: int,
    ctas: int,
    use_pdl: bool = True,
    early_trigger: bool = False,
) -> None:
    """Owner-row all-gather of ``x_owned`` [own_n, H] into copy ``copy`` of every rank's landing. The flag stamp is derived on
    the device from ``arena``'s forward sequence word (x 64 + ``layer``, layer < 64), so CUDA-graph replays stay correct; at most
    one call per (forward, layer) and landing. Errors go to ``arena``'s sticky error word."""
    if not 0 <= int(copy) < landing.copies:
        raise ValueError("ag_rows: copy outside the landing")
    _ag_module(landing.world, use_pdl).rows(
        landing.table(),
        arena.ctl_table(),
        landing.rank,
        landing.rows_max,
        landing.ctas_max,
        *arena.ctl_args(),
        x_owned,
        int(M),
        int(copy),
        arena.seq64,
        int(layer),
        int(ctas),
        EARLY_TRIGGER if early_trigger else 0,
    )


# ------------------------------------------------------------------------------------------------------------ AG_x tagged
def agx_spec(name: str, rows_max: int, hidden: int, ctas: int) -> HopSpec:
    """A unique site for ``ag_rows_tagged`` (one per (forward, layer) use) with ``ctas`` consumer units (= the launch grid)."""
    return HopSpec(
        name,
        rows_max=int(rows_max),
        width=rs_width(hidden),
        policy="unique",
        parity=False,
        units=int(ctas),
    )


def ag_rows_tagged(
    arena: HopArena,
    name: str,
    x_owned: torch.Tensor,
    out: torch.Tensor,
    M: int,
    *,
    step_flags: int = 0,
    use_pdl: bool = True,
) -> None:
    """Owner-row all-gather without flags: this rank's ``x_owned`` [own_n, H] (bf16) to every rank as tagged words on site
    ``name`` (multicast when the arena has multicast pushes on), and every rank's rows of the pass into ``out`` [>= M, H]
    (plain bf16, what the next library kernel reads). step_flags: STEP_ACQUIRE / STEP_RELEASE / EARLY_TRIGGER."""
    M, H = int(M), int(out.shape[1])
    own0, own1 = (M * arena.rank) // arena.world, (M * (arena.rank + 1)) // arena.world
    if x_owned.dtype != torch.bfloat16 or out.dtype != torch.bfloat16:
        raise ValueError("ag_rows_tagged: bf16 rows")
    if x_owned.shape[0] < own1 - own0 or x_owned.shape[1] != H or out.shape[0] < M:
        raise ValueError("ag_rows_tagged: x_owned [own_n, H], out [>= M, H]")
    _module(arena.world, use_pdl).agt(
        arena.push_table(name),
        arena.ctl_table(),
        arena.tok_table(),
        *arena.site_args(name),
        *arena.ctl_args(),
        int(step_flags),
        x_owned,
        out,
        M,
    )


FR_ROW_WORDS = 387  # the front role's token row: 2320 bytes in kB48 words
FR_IDS_WORDS = 8


def fr_unpack(
    arena: HopArena,
    ids_name: str,
    tok_name: str,
    M: int,
    hs_fp4: torch.Tensor,
    hs_sf: torch.Tensor,
    pts: torch.Tensor,
    packed: torch.Tensor,
    *,
    use_pdl: bool = True,
) -> None:
    """Front-role landing unpack (default off): every step row's token words and ids words, as the front role published them
    on the unique sites ``tok_name`` / ``ids_name`` (rows by GLOBAL index), read from this rank's receive regions (each word
    accepted on its own matching load; bounded waits set the sticky error word) and written as plain tensors for the stock
    expert stage: hs_fp4 [M, 2048] uint8, hs_sf [M, 256] uint8, pts [M] float32, packed [M, 8] int32. One launch after the
    front role's launch of the same forward."""
    M = int(M)
    if (
        hs_fp4.dtype != torch.uint8
        or hs_sf.dtype != torch.uint8
        or pts.dtype != torch.float32
        or packed.dtype != torch.int32
    ):
        raise ValueError("fr_unpack: hs_fp4 / hs_sf uint8, pts float32, packed int32")
    rank, i_rows, i_width, i_pitch, i_policy, i_parity, _, i_site, _ = arena.site_args(
        ids_name
    )
    _, t_rows, t_width, t_pitch, t_policy, t_parity, _, t_site, _ = arena.site_args(
        tok_name
    )
    _module(arena.world, use_pdl).fr_unpack(
        arena.table(ids_name),
        arena.table(tok_name),
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
        M,
        hs_fp4,
        hs_sf,
        pts,
        packed,
    )
