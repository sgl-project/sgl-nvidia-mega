# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Expert-output hop, route all-gather form: every rank receives every routed slot row of every token and computes the
engine's fused finalize + one-shot push all-reduce result itself.

Default off: nothing in the plugin calls this module unless a caller selects it with its own default-off knob.

``push`` (stand-in producer, launched after the expert kernel) stores each local slot row (``g2[e2p[t * 8 + k]]``) into every
other rank's receive region at linear row t * 8 + k. ``consume`` stages the 8 slot rows of each token (local ones from
``g2``, the others from the receive region, tag-checked), and writes ``out[t]`` = the engine's result for more than 64 tokens:
per slot the post-expert RMS norm with the engine's reduction tree, per owning rank in rank order the slot-order fma
accumulation rounded once to bf16, the push all-reduce's zero-word rewrite, and the fp32 rank-order fold rounded once to bf16.

The site is a ring site of the hop arena (``spec``): width 4608 (tile-aligned rows: each 128-column tile = 48 words, 44
tagged + 4 pads, three whole 128-byte lines), rows_max = ceil(M_max * 8 / W) (the region is addressed by
linear row), ``units`` = the largest consumer grid (one missed-advance word per consumer CTA). Slot reuse: with a sync-credit
arena (``async_credit=False``, the intended form) the consumer grid is one CTA per token (up to ``units``) and the CTA that
finishes last releases the slot once per rank (a per-rank count word per ring slot, allocated on the first call, which must
run outside graph capture); with step flag ``DEFER_RELEASE`` (sync-credit arena) the consume of transfer t instead releases
transfer t - 1 from one extra CTA while its token CTAs work (consecutive consumes of a site are consecutive transfers); with an
async-credit arena every one of the ``units`` consumer CTAs releases its own credit word.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

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
_CUH = _CSRC / "c3_hb0_ops.cuh"

H = 4096
TOPK = 8
WIDTH = 4608  # tile-aligned rows: 32 tiles x 48 words (44 tagged + 4 pads; three 128-byte lines per tile)
MIN_M = 65  # the engine's finalize uses this reduction tree above 64 tokens

EW_F32 = 1
NO_STOCK_ZERO = 2
EARLY_TRIGGER = 4
CLUSTER2 = 8  # removed (the two-CTA cluster form); refused by the launcher
DEFER_RELEASE = 16  # sync-credit arena: one extra CTA releases the site's PREVIOUS transfer (t - 1); no count (use on every call of a site)
PRE_WAIT_TABLE = 32  # far only: read the expanded-index map + route weights before the dependency wait; the immediate predecessor
# must publish them before its trigger (moe_ldw built with MUSE_VR_LDW_EARLY_SLOTTAB); re-checked after the wait
ROWS_READY = 64  # far only (with PRE_WAIT_TABLE): no dependency wait before the rows; each token's local rows are read once the
# expert kernel's flags of their passes carry its launch epoch (moe_ldw MUSE_VR_LDW_ROWS_READY); waits before exit


FAR_WIDTH = 3072  # far form: 1024 words per (source rank, token) row: 16 bytes (the bf16 partial, never an all-zero word) per thread


def far_spec(name: str, m_max: int, world: int, units: int, calls: int = 1) -> HopSpec:
    """Ring site of the far form for M <= m_max tokens (one row per token in each source rank's block); ``units`` = the largest
    grid (one missed-advance word per CTA); ``calls`` = call sites per step. Use with a sync-credit arena."""
    return HopSpec(
        name,
        rows_max=int(m_max),
        width=FAR_WIDTH,
        policy="ring",
        units=int(units),
        calls=int(calls),
    )


def spec(name: str, m_max: int, world: int, units: int, calls: int = 1) -> HopSpec:
    """Ring site for M <= m_max tokens; ``units`` = the largest consumer grid; ``calls`` = call sites per step."""
    rows = -(-int(m_max) * TOPK // int(world))
    return HopSpec(
        name,
        rows_max=rows,
        width=WIDTH,
        policy="ring",
        units=int(units),
        calls=int(calls),
    )


@lru_cache(maxsize=None)
def _module(
    world: int,
    use_pdl: bool,
    plant: bool = False,
    stamps: bool = False,
    plant_rel: int = 0,
):
    from sglang.kernels.jit.utils import is_arch_support_pdl, load_jit, make_cpp_args

    pdl = bool(use_pdl) and is_arch_support_pdl()
    args = make_cpp_args(world, pdl)
    flags = []
    if (
        plant
    ):  # tests only: the staging barrier dropped (a race the race checker must flag)
        flags.append("-DMUSE_VR_HB0_PLANT_NOBAR=1")
    if plant_rel:  # tests only: 1 = CTA 0 reads late; 2 / 3 = the same + a planted early count / deferred release (must be caught)
        flags.append(f"-DMUSE_VR_HB0_PLANT_REL={int(plant_rel)}")
    inc = [str(_CSRC)]
    if stamps:  # timing builds only: the harness supplies the stamp header's directory
        import os

        sd = os.environ.get("MUSE_VR_HB0_STAMPS_DIR", "").strip()
        if not sd:
            raise ValueError(
                "hb0: a timing build needs MUSE_VR_HB0_STAMPS_DIR (the stamp header's directory)"
            )
        flags.append("-DMUSE_VR_HB0_STAMPS=1")
        inc.append(sd)
    kw = {"extra_cuda_cflags": flags} if flags else {}
    return load_jit(
        f"muse_vr_c3_hb0_w{world}"
        + ("_plant" if plant else "")
        + ("_stamps" if stamps else "")
        + (f"_rel{int(plant_rel)}" if plant_rel else ""),
        *args,
        cuda_files=[str(_CUH)],
        cuda_wrappers=[
            ("consume", f"MuseVrHb0<{args}>::consume"),
            ("push", f"MuseVrHb0<{args}>::push"),
            ("far", f"MuseVrHb0<{args}>::far"),
        ],
        extra_include_paths=inc,
        **kw,
    )


def _check_arena(arena: HopArena, name: str, width: int = WIDTH) -> None:
    s = arena.spec(name)
    if s.policy != "ring" or s.width != width:
        raise ValueError("hb0: site must come from c3_hb0.spec (far: c3_hb0.far_spec)")


def push(
    arena: HopArena,
    name: str,
    e2p: torch.Tensor,
    g2: torch.Tensor,
    M: int,
    *,
    index: int = 0,
    grid: int = 0,
    use_pdl: bool = True,
    _tiles: int = 0,
) -> None:
    """Stand-in producer: every local slot row of the M tokens to every other rank (after the expert kernel). ``_tiles`` (tests
    only): store only the first ``_tiles`` 128-column tiles of every row (the rest never lands)."""
    _check_arena(arena, name)
    if g2.dtype != torch.bfloat16 or e2p.dtype != torch.int32 or g2.shape[1] != H:
        raise ValueError("hb0 push: g2 bf16 [*, 4096], e2p int32 [M * 8]")
    if grid <= 0:
        grid = torch.cuda.get_device_properties(g2.device).multi_processor_count
    _module(arena.world, use_pdl).push(
        arena.table(name, index),
        arena.ctl_table(),
        *arena.site_args(name, index),
        *arena.ctl_args(),
        e2p,
        g2,
        int(M),
        int(grid),
        int(_tiles) * 22,
    )


def consume(
    arena: HopArena,
    name: str,
    packed: torch.Tensor,
    e2p: torch.Tensor,
    ew: torch.Tensor,
    g2: torch.Tensor,
    norm_w: torch.Tensor,
    out: torch.Tensor,
    eps: float,
    M: int,
    n_local: int,
    *,
    placement: str,
    index: int = 0,
    step_flags: int = 0,
    use_pdl: bool = True,
    allow_small_m: bool = False,
    _plant: bool = False,
    _stamps: torch.Tensor | None = None,
    _plant_rel: int = 0,
) -> None:
    """out[:M] = the engine's fused finalize + push all-reduce result, from this rank's slots (g2) and the received ones.

    ``placement`` must be "static" (experts placed contiguously, ``n_local`` per rank: owner = expert / n_local); any other
    placement (balancing, replicas) is refused."""
    if placement != "static":
        raise ValueError(
            f"hb0 consume: expert placement {placement!r} is not supported (static placement only)"
        )
    _check_arena(arena, name)
    if M < MIN_M and not allow_small_m:
        raise ValueError(
            f"hb0 consume: M >= {MIN_M} (the engine uses other reduction trees below)"
        )
    if (
        out.dtype != torch.bfloat16
        or g2.dtype != torch.bfloat16
        or norm_w.dtype != torch.bfloat16
    ):
        raise ValueError("hb0 consume: bf16 out / g2 / norm weight")
    if (
        packed.dtype != torch.int32
        or e2p.dtype != torch.int32
        or ew.dtype not in (torch.bfloat16, torch.float32)
    ):
        raise ValueError("hb0 consume: int32 packed / e2p, bf16 or fp32 route weights")
    if int(step_flags) & CLUSTER2:
        raise ValueError("hb0 consume: the two-CTA cluster form was removed")
    fl = (int(step_flags) & ~EW_F32) | (EW_F32 if ew.dtype == torch.float32 else 0)
    st = (
        _stamps if _stamps is not None else out
    )  # placeholder tensor when no stamps are taken
    _module(
        arena.world, use_pdl, bool(_plant), _stamps is not None, int(_plant_rel)
    ).consume(
        arena.table(name, index),
        arena.ctl_table(),
        *arena.site_args(name, index),
        *arena.ctl_args(),
        fl,
        packed,
        e2p,
        ew,
        g2,
        norm_w,
        out,
        float(eps),
        int(M),
        int(n_local),
        done_words(arena, name),
        st,
        1 if _stamps is not None else 0,
    )


def far(
    arena: HopArena,
    name: str,
    packed: torch.Tensor,
    e2p: torch.Tensor,
    ew: torch.Tensor,
    g2: torch.Tensor,
    norm_w: torch.Tensor,
    out: torch.Tensor,
    eps: float,
    M: int,
    n_local: int,
    *,
    placement: str,
    index: int = 0,
    step_flags: int = 0,
    use_pdl: bool = True,
    allow_small_m: bool = False,
    _stamps: torch.Tensor | None = None,
    rows_ready: tuple | None = None,
    _plant_rel: int = 0,
) -> None:
    """out[:M] = the engine's fused finalize + one-shot push all-reduce result, as ONE kernel after the expert kernel (no producer
    in the expert kernel): this rank's partial from its local slot rows (g2), pushed to the other ranks as tagged words, their
    partials polled (a word is ready when non-zero: the zero-word rewrite never pushes an all-zero word; consumers clear what they
    read), folded in rank order. Site from ``far_spec`` on a sync-credit arena; same numerics as ``consume`` with the rewrite on."""
    if placement != "static":
        raise ValueError(
            f"hb0 far: expert placement {placement!r} is not supported (static placement only)"
        )
    _check_arena(arena, name, FAR_WIDTH)
    if arena.async_credit:
        raise ValueError("hb0 far: the arena must use sync credits")
    if M < MIN_M and not allow_small_m:
        raise ValueError(
            f"hb0 far: M >= {MIN_M} (the engine uses other reduction trees below)"
        )
    if (
        out.dtype != torch.bfloat16
        or g2.dtype != torch.bfloat16
        or norm_w.dtype != torch.bfloat16
    ):
        raise ValueError("hb0 far: bf16 out / g2 / norm weight")
    if (
        packed.dtype != torch.int32
        or e2p.dtype != torch.int32
        or ew.dtype not in (torch.bfloat16, torch.float32)
    ):
        raise ValueError("hb0 far: int32 packed / e2p, bf16 or fp32 route weights")
    if int(step_flags) & CLUSTER2:
        raise ValueError("hb0 far: no cluster form")
    if int(step_flags) & NO_STOCK_ZERO:
        raise ValueError(
            "hb0 far: needs the zero-word rewrite (a received word is ready when it is not all-zero bits)"
        )
    rr = bool(int(step_flags) & ROWS_READY)
    if rr:
        # (published epoch [1], per-CTA last epochs, flags, e2pass int16 [>= M * 8]) from moe_ldw.rows_ready_views(scratch)
        if (
            not int(step_flags) & PRE_WAIT_TABLE
            or rows_ready is None
            or len(rows_ready) != 4
        ):
            raise ValueError(
                "hb0 far: ROWS_READY needs PRE_WAIT_TABLE and the expert kernel's (epoch, last, flags, e2pass) views"
            )
        pub, last, flg, e2pass = rows_ready
        if (
            pub.dtype != torch.int64
            or last.dtype != torch.int64
            or flg.dtype != torch.int64
            or e2pass.dtype != torch.int16
            or e2pass.numel() < M * 8
        ):
            raise ValueError(
                "hb0 far: ROWS_READY views: int64 epoch / last / flags, int16 e2pass [>= M * 8]"
            )
    else:
        pub = last = flg = e2pass = out  # placeholders (not read)
    fl = (int(step_flags) & ~EW_F32) | (EW_F32 if ew.dtype == torch.float32 else 0)
    if _plant_rel not in (0, 4):
        raise ValueError(
            "hb0 far: _plant_rel 0, or 4 (tests only: ROWS_READY rows read without their flags)"
        )
    _module(arena.world, use_pdl, False, _stamps is not None, int(_plant_rel)).far(
        arena.table(name, index),
        arena.ctl_table(),
        *arena.site_args(name, index),
        *arena.ctl_args(),
        fl,
        packed,
        e2p,
        ew,
        g2,
        norm_w,
        out,
        float(eps),
        int(M),
        int(n_local),
        done_words(arena, name),
        _stamps if _stamps is not None else out,
        1 if _stamps is not None else 0,
        pub,
        last,
        flg,
        e2pass,
    )


def done_words(arena: HopArena, name: str) -> torch.Tensor:
    """The site's per-rank count words (one per ring slot; count form). Allocated zeroed on the first call, which must run
    outside graph capture; kept on the arena object."""
    d = arena.__dict__.setdefault("_hb0_done", {})
    t = d.get(name)
    if t is None:
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError(
                "hb0 consume: the first call on a site must run outside graph capture (it allocates the count words)"
            )
        t = torch.zeros(3, dtype=torch.int64, device=arena.device)
        d[name] = t
    return t
