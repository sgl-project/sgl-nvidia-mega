# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""MoE-LD-W (``csrc/moe_ldw.cuh``): routed NVFP4 experts of one expert-parallel rank for 1..512 token rows in one launch of
one CTA per SM streaming each active local expert's weights once per pass of up to 128 routed rows.

``moe_ldw(...)`` takes the arguments of ``moe_ld_ops.moe_ld`` and returns the same ``(gemm2_out, expert_weights,
expanded_idx_to_permuted_idx)`` triple (FC2 rows in a compact expert-major order, the map points at them), every routed
row bitwise equal to the stock call's.

``Scratch`` (per device; the kernel keeps it consistent across launches that are ordered by completion): permuted token
rows, token-scale images, activation rows, FC2 operand rows, two counter sets (alternating launches), start tickets,
sticky error word. Expert
weights are best placed with ``ld_layout.split_rows_copy`` (each expert's row halves on the two locality domains).
"""

from __future__ import annotations

import os
import re
from functools import lru_cache
from pathlib import Path

import torch

# The SGLang names this module reads (test/unit/test_contract.py pins them against the supported SGLang)
SGLANG_READS = {
    "kernels/jit/utils/__init__.py": ("load_jit",),
}


_CSRC = Path(__file__).resolve().parents[1] / "csrc"
_CUH = _CSRC / "moe_ldw.cuh"
HIDDEN = 4096
INTER = 4096
TOP_K = 8
MAX_TOKENS = 512
MAX_SLOTS = MAX_TOKENS * TOP_K
MAX_POOL = 128
PASS_ROWS = 128
MAX_PASSES = MAX_POOL + MAX_SLOTS // PASS_ROWS
ACT_WORDS = INTER // 4
IMG_BYTES = (HIDDEN // 64) * 512
CNT_SET = 2 * MAX_PASSES + 32
FLAG_EVICT_FIRST = 2
FLAG_TAIL_TRIGGER = 32
STAMP_WORDS = 24
DEFAULT_FLAGS = FLAG_EVICT_FIRST | FLAG_TAIL_TRIGGER
# the RING4 build: 32-row passes and a 4-stage ring (more weight bytes in flight per CTA); same rows, its own Scratch
RING4_PASS_ROWS = 32
RING4 = (
    "-DMUSE_VR_LDW_GATHER=0",
    f"-DMUSE_VR_LDW_PASS_ROWS={RING4_PASS_ROWS}",
    "-DMUSE_VR_LDW_STAGES=4",
)
# RING4 with 48-row passes (experts with 33-48 rows in one pass; fewer passes for hot experts); its own scratch
RING48_PASS_ROWS = 48
RING48 = (
    "-DMUSE_VR_LDW_GATHER=0",
    "-DMUSE_VR_LDW_PASS_ROWS=48",
    "-DMUSE_VR_LDW_STAGES=4",
)
# the landed build (development, added to RING4 / RING48 / the base build): routed ids and token rows straight from the hop
# landing buffers (moe_ldw_landed below); default off, never selected by the serving path
LANDED = ("-DMUSE_VR_LDW_LANDED=1",)
# the finalize build (development, with LANDED): per-route norm moments in the FC2 epilogue + per-token weighted sum (tok_out)
EPI = ("-DMUSE_VR_LDW_EPI=1",)
EPI_PARTS = (HIDDEN // 128) * 4  # partials per routed row (FC2 tiles x epilogue warps)
TOK_ROW_BYTES = HIDDEN // 2 + HIDDEN // 16 + 4  # FP4 row, block scales, per-token scale
TOK_WORDS = (TOK_ROW_BYTES + 5) // 6  # 6 payload bytes per 64-bit landing word
# the wide build (added to any build): up to 1024 rows per call; it needs a Scratch made with max_tokens=MAX_TOKENS_WIDE
MAX_TOKENS_WIDE = 1024
WIDE = ("-DMUSE_VR_LDW_MAX_TOKENS=1024",)


def max_tokens_of(extra) -> int:
    """Rows per call of the build ``extra`` describes (512, or 1024 for the wide build)."""
    return MAX_TOKENS_WIDE if WIDE[0] in tuple(extra) else MAX_TOKENS


def scratch_passes(pass_rows: int, max_tokens: int = MAX_TOKENS) -> int:
    """The kernel's kMaxPasses for that pass height and row ceiling (the wide build caps it at 256)."""
    passes = MAX_POOL + max_tokens * TOP_K // pass_rows
    return min(passes, 256) if max_tokens > MAX_TOKENS else passes


GLOBAL_SCALE_INV = float(
    torch.tensor(1.0, dtype=torch.float32) / (torch.tensor(448.0) * torch.tensor(6.0))
)


def _flashinfer_include_paths() -> list[str]:
    import flashinfer

    data = Path(flashinfer.__file__).resolve().parent / "data"
    paths = [
        data / "csrc" / "nv_internal",
        data / "csrc" / "nv_internal" / "include",
        data / "include",
        data / "cutlass" / "include",
    ]
    return [str(p) for p in paths if p.exists()]


@lru_cache(maxsize=None)
def _module(extra: tuple = ()):
    import hashlib

    _key = tuple(extra)

    from sglang.kernels.jit.utils import load_jit

    from .. import mma_k

    extra = tuple(extra) + mma_k.build_flags(
        "moe_ldw"
    )  # MMA-K form (MUSE_VR_MOE_MMA_K; no define for the default 64)
    extra = (
        tuple(extra) + epoch_flags()
    )  # launch-counter slot claim (MUSE_VR_MOE_LDW_SLOT_EPOCH=1; default off)
    extra = tuple(extra) + compact_flags(
        extra
    )  # compact prologue (MUSE_VR_MOE_LDW_COMPACT=1; default off)
    extra = tuple(extra) + split_rel_flags(
        extra
    )  # copy phase's split release (MUSE_VR_MOE_LDW_SPLIT_REL=1; default off)
    extra = tuple(extra) + thead_flags(
        extra
    )  # item-0 token rows by tensor-copy gathers (MUSE_VR_MOE_LDW_THEAD=1; default off)
    extra = tuple(extra) + copy_diet_flags(
        extra
    )  # copy phase with operands by value (MUSE_VR_MOE_LDW_COPY_DIET=1; default off)
    extra = tuple(extra) + early_slottab_flags(
        extra
    )  # slot table published before the trigger (MUSE_VR_W_EARLY_SLOTTAB=1)
    extra = tuple(extra) + rows_ready_flags(
        extra
    )  # rows-ready flags (MUSE_VR_W_ROWS_READY=1, with the early slot table)
    extra = tuple(extra) + early_copy_flags(
        extra
    )  # copy phase before the work list (MUSE_VR_MOE_LDW_EARLY_COPY=1); last
    _RR_BUILDS[_key] = ROWS_READY[0] in extra
    # one symbol namespace per build variant (two variants may be loaded in one process)
    abi = "v" + hashlib.sha256(" ".join(extra).encode()).hexdigest()[:12]
    from .. import slot_claim

    extra = tuple(extra) + slot_claim.build_flags(
        "moe_ldw"
    )  # exact slot claim (default) or the UNSAFE form
    parts = [re.sub(r"[^A-Za-z0-9_]", "_", e) for e in extra]
    if (
        len("_".join(parts)) > 200
    ):  # the build's cache directory name must stay below the file-name limit
        parts = ["h" + hashlib.sha256(" ".join(extra).encode()).hexdigest()[:16]]
    return load_jit(
        "muse_vr_moe_ldw",
        *parts,
        cuda_files=[str(_CUH)],
        cuda_wrappers=[
            ("run", "muse_vr_ldw::MuseVrLdw::run"),
            ("preload", "muse_vr_ldw::MuseVrLdw::preload"),
        ]
        + (
            [("run_landed", "muse_vr_ldw::MuseVrLdw::run_landed")]
            if LANDED[0] in extra
            else []
        ),
        extra_include_paths=[str(_CSRC)] + _flashinfer_include_paths(),
        extra_cuda_cflags=[
            "-O3",
            "-use_fast_math",
            "-DENABLE_BF16",
            "-lineinfo",
            f"-DMUSE_VR_LDW_ABI={abi}",
            *extra,
        ],
    )


def preload(extra: tuple = ()) -> None:
    _module(tuple(extra)).preload()


CTL_CLAIM_WORD = 21  # first word of the slot claim area in the control tensor (the kernel's kClaimWord)
EPOCH_ENV = "MUSE_VR_MOE_LDW_SLOT_EPOCH"
EPOCH_WORDS = 256 * 8  # per-SM launch counters after the 32 control words
_epoch_recorded: list = []


def epoch() -> bool:
    """Launch-counter slot claim (slot_claim.cuh, -DMUSE_VR_SLOT_EPOCH=1): the exact claim with no bitmap and no
    last-CTA check (default off; the UNSAFE form takes precedence)."""
    from .. import slot_claim

    return os.environ.get(EPOCH_ENV, "0").strip() == "1" and not slot_claim.unsafe()


def epoch_flags() -> tuple[str, ...]:
    if not epoch():
        return ()
    if not _epoch_recorded:
        _epoch_recorded.append(True)
        from .. import bed

        bed.report("slot_epoch_moe_ldw", True, env=f"{EPOCH_ENV}=1")
    return ("-DMUSE_VR_SLOT_EPOCH=1",)


THEAD_ENV = "MUSE_VR_MOE_LDW_THEAD"
_thead_recorded: list = []


def thead_flags(extra: tuple = ()) -> tuple[str, ...]:
    """The first FC1 item's token rows by tensor-copy four-row gathers and its scale images by warp 3
    (-DMUSE_VR_LDW_THEAD=1, head rows = the build's pass rows) on the served ring builds (-DMUSE_VR_LDW_STAGES, no head copies),
    with MUSE_VR_MOE_LDW_THEAD=1 (default off); never on landed or board builds. Routed rows are bitwise the same."""
    if os.environ.get(THEAD_ENV, "0").strip() != "1":
        return ()
    if not any(e.startswith("-DMUSE_VR_LDW_STAGES=") for e in extra):
        return ()
    if any(
        e.startswith(
            ("-DMUSE_VR_LDW_LANDED=1", "-DMUSE_VR_LDW_BALB", "-DMUSE_VR_LDW_THEAD")
        )
        for e in extra
    ):
        return ()
    rows = next(
        (e.split("=", 1)[1] for e in extra if e.startswith("-DMUSE_VR_LDW_PASS_ROWS=")),
        "32",
    )
    if not _thead_recorded:
        _thead_recorded.append(True)
        from .. import bed

        bed.report("moe_ldw_thead", True, env=f"{THEAD_ENV}=1")
    return ("-DMUSE_VR_LDW_THEAD=1", f"-DMUSE_VR_LDW_HEAD_ROWS={rows}")


SPLIT_REL_ENV = "MUSE_VR_MOE_LDW_SPLIT_REL"
_split_rel_recorded: list = []


def split_rel_flags(extra: tuple = ()) -> tuple[str, ...]:
    """The copy phase's split release (-DMUSE_VR_LDW_SPLIT_REL=1: the rows-ready count after the token rows and scale images,
    a second count after the per-slot metadata) on the served builds, with MUSE_VR_MOE_LDW_SPLIT_REL=1 (default off); never
    on landed or board builds. Routed rows are bitwise the same."""
    if os.environ.get(SPLIT_REL_ENV, "0").strip() != "1":
        return ()
    if any(
        e.startswith(
            ("-DMUSE_VR_LDW_LANDED=1", "-DMUSE_VR_LDW_BALB", "-DMUSE_VR_LDW_SPLIT_REL")
        )
        for e in extra
    ):
        return ()
    if not _split_rel_recorded:
        _split_rel_recorded.append(True)
        from .. import bed

        bed.report("moe_ldw_split_rel", True, env=f"{SPLIT_REL_ENV}=1")
    return ("-DMUSE_VR_LDW_SPLIT_REL=1",)


COPY_DIET_ENV = "MUSE_VR_MOE_LDW_COPY_DIET"
_copy_diet_recorded: list = []


def copy_diet_flags(extra: tuple = ()) -> tuple[str, ...]:
    """The copy phase with its operands by value (-DMUSE_VR_LDW_COPY_DIET=1: kernel pointers from the calling roles, the
    CTA's slot tables read with shared loads) on the served builds, with MUSE_VR_MOE_LDW_COPY_DIET=1 (default off); never on
    landed or board builds. Same stores and releases; routed rows are bitwise the same."""
    if os.environ.get(COPY_DIET_ENV, "0").strip() != "1":
        return ()
    if any(
        e.startswith(
            ("-DMUSE_VR_LDW_LANDED=1", "-DMUSE_VR_LDW_BALB", "-DMUSE_VR_LDW_COPY_DIET")
        )
        for e in extra
    ):
        return ()
    if not _copy_diet_recorded:
        _copy_diet_recorded.append(True)
        from .. import bed

        bed.report("moe_ldw_copy_diet", True, env=f"{COPY_DIET_ENV}=1")
    return ("-DMUSE_VR_LDW_COPY_DIET=1",)


COMPACT_ENV = "MUSE_VR_MOE_LDW_COMPACT"
_compact_recorded: list = []


def compact_flags(extra: tuple = ()) -> tuple[str, ...]:
    """The compact prologue (-DMUSE_VR_LDW_PRO_COMPACT=1: the routed ids read in a rolled loop, prefixes over the used chunks
    only, the FC2 pass-row table off the work list's path) on every W build except the board build, with
    MUSE_VR_MOE_LDW_COMPACT=1 (default off). Routed rows are bitwise the same (tests/test_moe_ldw.py)."""
    if os.environ.get(COMPACT_ENV, "0").strip() != "1":
        return ()
    if any(
        e.startswith(
            ("-DMUSE_VR_LDW_BALB", "-DMUSE_VR_LDW_PRO_COMPACT", "-DMUSE_VR_LDW_START")
        )
        for e in extra
    ):
        return ()
    if not _compact_recorded:
        _compact_recorded.append(True)
        from .. import bed

        bed.report("moe_ldw_compact", True, env=f"{COMPACT_ENV}=1")
    return ("-DMUSE_VR_LDW_PRO_COMPACT=1",)


EARLY_SLOTTAB_ENV = "MUSE_VR_W_EARLY_SLOTTAB"
EARLY_SLOTTAB = ("-DMUSE_VR_LDW_EARLY_SLOTTAB=1",)


def early_slottab_flags(extra: tuple = ()) -> tuple[str, ...]:
    """The early slot-table publish (-DMUSE_VR_LDW_EARLY_SLOTTAB=1: after the copy phase each CTA fences its share of the
    expanded-index map and top-k weights and triggers its dependents, instead of the tail trigger, so the dependent may read them
    before its dependency wait) with MUSE_VR_W_EARLY_SLOTTAB=1 (default off). Outputs are unchanged (tests/test_moe_ldw.py).
    Landed and balanced builds step aside (logged)."""
    if os.environ.get(EARLY_SLOTTAB_ENV, "0").strip() != "1":
        return ()
    if any(e.startswith("-DMUSE_VR_LDW_EARLY_SLOTTAB") for e in extra):
        return ()
    from .. import bed

    if any(e.startswith(("-DMUSE_VR_LDW_BALB", "-DMUSE_VR_LDW_LANDED")) for e in extra):
        bed.report(
            "moe_ldw_early_slottab",
            False,
            env=f"{EARLY_SLOTTAB_ENV}=1",
            reason="landed / balanced build: step aside",
        )
        return ()
    bed.report("moe_ldw_early_slottab", True, env=f"{EARLY_SLOTTAB_ENV}=1")
    return EARLY_SLOTTAB


ROWS_READY_ENV = "MUSE_VR_W_ROWS_READY"
ROWS_READY = ("-DMUSE_VR_LDW_ROWS_READY=1",)
RR_PER_PASS = (
    HIDDEN // 128
) * 4  # flags per pass: FC2 tiles x epilogue warps (the kernel's kRrPerPass)
RR_LAST = 1024  # the consumer's per-CTA last epochs (the kernel's kRrLast; written by the consumer only)
_RR_BUILDS: dict = {}  # caller's extra -> the build has the rows-ready flags (filled by _module)


def rows_ready_flags(extra: tuple = ()) -> tuple[str, ...]:
    """Rows-ready flags (-DMUSE_VR_LDW_ROWS_READY=1: per launch an epoch from a 64-bit ticket; the copy phase also publishes
    each routed slot's pass index and the epoch before the trigger; every FC2 epilogue warp releases flag [pass][tile][warp] =
    epoch after its last row stores of an item) with MUSE_VR_W_ROWS_READY=1 (default off). Needs the early slot table; other
    builds step aside (logged). Outputs are unchanged (tests/test_moe_ldw.py)."""
    if os.environ.get(ROWS_READY_ENV, "0").strip() != "1":
        return ()
    if any(e.startswith("-DMUSE_VR_LDW_ROWS_READY") for e in extra):
        return ()
    from .. import bed

    if EARLY_SLOTTAB[0] not in tuple(extra):
        bed.report(
            "moe_ldw_rows_ready",
            False,
            env=f"{ROWS_READY_ENV}=1",
            reason="needs the early slot table: step aside",
        )
        return ()
    bed.report("moe_ldw_rows_ready", True, env=f"{ROWS_READY_ENV}=1")
    return ROWS_READY


EARLY_COPY_ENV = "MUSE_VR_MOE_LDW_EARLY_COPY"
EARLY_COPY = ("-DMUSE_VR_LDW_EARLY_COPY=1",)
# defines of the builds the early copy does not cover (the kernel refuses each with #error); a define set to 0 does not count
_EARLY_COPY_REFUSED = {
    "-DMUSE_VR_LDW_LANDED": "landed build",
    "-DMUSE_VR_LDW_BALB": "board build",
    "-DMUSE_VR_LDW_EPI": "finalize build",
    "-DMUSE_VR_LDW_START": "start-order build",
    "-DMUSE_VR_LDW_HEAD_GO": "start-order build",
    "-DMUSE_VR_LDW_HEAD_PRIO": "start-order build",
    "-DMUSE_VR_LDW_THEAD": "tensor-copy head",
    "-DMUSE_VR_LDW_EARLY_SLOTTAB": "early slot table",
    "-DMUSE_VR_LDW_ROWS_READY": "rows-ready flags",
    "-DMUSE_VR_LDW_HB0_PUSH": "expert-output hop push",
}
_early_copy_recorded: list = []
_early_copy_aside: set = set()


def early_copy_flags(extra: tuple = ()) -> tuple[str, ...]:
    """The copy phase started before the work list (-DMUSE_VR_LDW_EARLY_COPY=1: warp 0 hands the expert bases and the counter
    parity to the copy warps by a named barrier right after its expert loop; the copy warps run the unchanged copy phase while
    warp 0 builds the work list) on the builds without head copies (-DMUSE_VR_LDW_GATHER=0: the RING48 / RING4 builds and the
    no-head build), with MUSE_VR_MOE_LDW_EARLY_COPY=1 (default off). Every other build steps aside (logged once per reason):
    the head build, and the landed / board / finalize / start-order / tensor-copy head / early slot table / rows-ready /
    hop-push builds. Applied after every other build flag. Routed rows and the copy phase's bytes are bitwise the same
    (tests/test_moe_ldw_early_copy.py, tests/test_moe_ldw.py)."""
    if os.environ.get(EARLY_COPY_ENV, "0").strip() != "1":
        return ()
    if any(e.startswith("-DMUSE_VR_LDW_EARLY_COPY") for e in extra):
        return ()
    reason = None
    for e in extra:
        name, _, value = e.partition("=")
        if name in _EARLY_COPY_REFUSED and value.strip() != "0":
            reason = _EARLY_COPY_REFUSED[name]
            break
    if reason is None and "-DMUSE_VR_LDW_GATHER=0" not in tuple(extra):
        reason = "head build (head copies)"
    from .. import bed

    if reason is not None:
        if reason not in _early_copy_aside:
            _early_copy_aside.add(reason)
            tag = re.sub(r"[^a-z0-9]+", "_", reason).strip("_")
            bed.report(
                f"moe_ldw_early_copy_aside_{tag}",
                False,
                env=f"{EARLY_COPY_ENV}=1",
                reason=f"{reason}: step aside",
            )
        return ()
    if not _early_copy_recorded:
        _early_copy_recorded.append(True)
        bed.report("moe_ldw_early_copy", True, env=f"{EARLY_COPY_ENV}=1")
    return EARLY_COPY


def rows_ready_words(pass_rows: int, max_tokens: int = MAX_TOKENS) -> int:
    """int64 words of the rows-ready buffer (the kernel's kRrWords): ticket, published epoch, the consumer's per-CTA last
    epochs, flags, e2pass (int16)."""
    return (
        2
        + RR_LAST
        + scratch_passes(pass_rows, max_tokens) * RR_PER_PASS
        + max_tokens * TOP_K // 4
    )


def rows_ready_built(extra: tuple = ()) -> bool:
    """The build the caller's ``extra`` selects has the rows-ready flags (loads the module)."""
    _module(tuple(extra))
    return bool(_RR_BUILDS.get(tuple(extra), False))


def rows_ready_views(scratch: Scratch):
    """(published epoch [1], the consumer's per-CTA last epochs, flags, e2pass int16) views of the scratch's rows-ready
    buffer, for the dependent grid."""
    rr = scratch.rr
    nf = scratch_passes(scratch.pass_rows, scratch.max_tokens) * RR_PER_PASS
    b = 2 + RR_LAST
    return rr[1:2], rr[2:b], rr[b : b + nf], rr[b + nf :].view(torch.int16)


class Scratch:
    """Per-device buffers of MoE-LD-W (launches that share it must be ordered by completion). ``pass_rows`` is the build's
    pass height (``PASS_ROWS``, or ``RING4_PASS_ROWS`` for the ``RING4`` build): the per-pass buffers follow it."""

    def __init__(
        self,
        device: torch.device,
        pass_rows: int = PASS_ROWS,
        max_tokens: int = MAX_TOKENS,
    ):
        z = dict(device=device)
        if max_tokens not in (MAX_TOKENS, MAX_TOKENS_WIDE):
            raise ValueError(
                f"max_tokens {max_tokens}: {MAX_TOKENS} or {MAX_TOKENS_WIDE}"
            )
        slots = max_tokens * TOP_K
        passes = scratch_passes(pass_rows, max_tokens)
        self.pass_rows = pass_rows
        self.max_tokens = max_tokens
        self.hs_perm = torch.zeros(slots, HIDDEN // 2, dtype=torch.uint8, **z)
        self.pts_perm = torch.zeros(slots, dtype=torch.float32, **z)
        self.img1 = torch.zeros(passes * IMG_BYTES, dtype=torch.uint8, **z)
        self.img2 = torch.zeros(passes * IMG_BYTES, dtype=torch.uint8, **z)
        self.act = torch.zeros(slots * ACT_WORDS, dtype=torch.int64, **z)
        # FC2 token rows at bucket-padded per-pass bases (the kernel's kHs2Rows = 2 slots + 16 passes rows); zero-filled once
        # here, never per launch: padding rows are written by no launch, so FC2 operand boxes never cover another pass's rows
        self.hs2 = torch.zeros(
            2 * slots + 16 * passes, HIDDEN // 2, dtype=torch.uint8, **z
        )
        self.ramax = torch.zeros(slots, dtype=torch.float32, **z)
        self.cnt = torch.zeros(2 * (2 * passes + 32), dtype=torch.int32, **z)
        self.ctl = torch.zeros(
            32 + (EPOCH_WORDS if epoch() else 0), dtype=torch.int32, **z
        )
        self.err = torch.zeros(1, dtype=torch.int32, **z)
        # rows-ready builds only (never reset: flags carry the launch epoch); allocated for every scratch (small)
        self.rr = torch.zeros(
            rows_ready_words(pass_rows, max_tokens), dtype=torch.int64, **z
        )

    def error(self) -> int:
        return int(self.err.item())

    def slot_tally(self) -> dict:
        """Slot claim counters (verified launches, hinted-slot misses; exact form only)."""
        from .. import slot_claim

        return slot_claim.tally(self.ctl, CTL_CLAIM_WORD)


HB0_PREFIX = "-DMUSE_VR_LDW_HB0_PUSH="


def _hb0_args(extra: tuple, hb, scratch: Scratch) -> tuple:
    """The expert-output hop push build's arguments (empty for other builds): ``hb`` = (site table, ctl table, site args,
    ctl args) as the hop arena gives them (arena.table(name, index), arena.ctl_table(), arena.site_args(name, index),
    arena.ctl_args()); the per-row slot map lives in the scratch (allocated on first use)."""
    if not any(f.startswith(HB0_PREFIX) and not f.endswith("=0") for f in extra):
        if hb is not None:
            raise ValueError("hb given for a build without the expert-output hop push")
        return ()
    if hb is None:
        raise ValueError(
            "the expert-output hop push build needs hb = (site table, ctl table, site args, ctl args)"
        )
    table, ctl_table, site_args, ctl_args = hb
    site_args, ctl_args = tuple(site_args), tuple(ctl_args)
    if len(site_args) != 9 or len(ctl_args) != 4:
        raise ValueError(
            "hb: 9 site args (rank, rows_max, width, pitch, policy, parity, index, site, units) and 4 ctl args"
        )
    if getattr(scratch, "hb_slot", None) is None:
        scratch.hb_slot = torch.full(
            (scratch.max_tokens * TOP_K,),
            -1,
            dtype=torch.int32,
            device=scratch.err.device,
        )
    return (
        table,
        ctl_table,
        *(int(v) for v in site_args),
        *(int(v) for v in ctl_args),
        scratch.hb_slot,
    )


def moe_ldw(
    packed,
    hs,
    hs_sf,
    pts,
    w13,
    sf13,
    w2,
    sf2,
    gate_alpha,
    up_alpha,
    out_alpha,
    local_offset: int,
    scratch: Scratch,
    flags: int | None = None,
    dbg: int = 0,
    pdl: bool = False,
    extra: tuple = (),
    stamps=None,
    hb=None,
):
    from ..moe_ld_ops import domain_slots

    if flags is None:
        flags = DEFAULT_FLAGS
    m = packed.shape[0]
    limit = max_tokens_of(extra)
    if not 1 <= m <= limit:
        raise ValueError(f"num_tokens {m} outside 1..{limit}")
    dev = packed.device
    table, n0, n1 = domain_slots(
        dev.index if dev.index is not None else torch.cuda.current_device()
    )
    out = torch.empty(m * TOP_K, HIDDEN, dtype=torch.bfloat16, device=dev)
    ew = torch.empty(m, TOP_K, dtype=torch.bfloat16, device=dev)
    e2p = torch.empty(m * TOP_K, dtype=torch.int32, device=dev)
    s = scratch
    want = PASS_ROWS
    for f in extra:
        if f.startswith("-DMUSE_VR_LDW_PASS_ROWS="):
            want = int(f.split("=", 1)[1])
    if s.max_tokens != limit:
        raise ValueError(f"scratch for {s.max_tokens} rows per call, build for {limit}")
    if s.pass_rows != want:
        raise ValueError(
            f"scratch for {s.pass_rows}-row passes, build with {want}-row passes"
        )
    rr = rows_ready_built(tuple(extra))
    if rr:  # the epoch (ticket / grid) is consistent only if every launch on this scratch has the same grid
        if getattr(s, "rr_grid", None) is None:
            s.rr_grid = int(n0 + n1)
        elif s.rr_grid != int(n0 + n1):
            raise ValueError(
                f"moe_ldw rows-ready: grid {n0 + n1} != this scratch's {s.rr_grid} (one grid per scratch)"
            )
    _module(tuple(extra)).run(
        packed,
        hs.view(torch.uint8),
        hs_sf.view(torch.uint8).reshape(m, -1),
        pts.reshape(m).float(),
        w13.view(torch.uint8),
        sf13.view(torch.uint8),
        w2.view(torch.uint8),
        sf2.view(torch.uint8),
        gate_alpha,
        up_alpha,
        out_alpha,
        table,
        out,
        ew.view(torch.int16),
        e2p,
        s.hs_perm,
        s.pts_perm,
        s.img1,
        s.img2,
        s.act,
        s.hs2,
        s.ramax,
        s.cnt,
        s.ctl,
        s.err,
        stamps,
        int(local_offset),
        n0,
        n1,
        GLOBAL_SCALE_INV,
        int(flags),
        int(dbg),
        int(bool(pdl)),
        *_hb0_args(tuple(extra), hb, s),
        *((s.rr,) if rr else ()),
    )
    return out, ew, e2p


def moe_ldw_landed(
    m: int,
    ids_land,
    tok_land,
    seq64,
    last_seq,
    ids_pitch: int,
    tok_pitch: int,
    rows_max: int,
    world: int,
    ids_parity: bool,
    tok_parity: bool,
    w13,
    sf13,
    w2,
    sf2,
    gate_alpha,
    up_alpha,
    out_alpha,
    local_offset: int,
    scratch: Scratch,
    flags: int | None = None,
    dbg: int = 0,
    pdl: bool = False,
    extra: tuple = LANDED,
    stamps=None,
    bal_table=None,
    board=None,
    bal_rank: int = 0,
    norm_w=None,
    norm_eps: float = 1e-8,
    test_delay_ns: int = 0,
):
    """Landed build (development): the routed ids and token rows of ``m`` rows come from this rank's landing buffers
    (int64 words, [copy][source rank][rows_max][pitch], copy 0 first): ids as 8 words per row (32-bit tag << 32 | packed
    id), token rows as TOK_WORDS words of 6 payload bytes + a 16-bit tag in bits 63..48. ``seq64`` is the forward
    sequence word (int64 [1]) the tags and copies derive from; ``last_seq`` one int64 guard word per SM. Board build
    (-DMUSE_VR_LDW_BALB=<world>): ``bal_table`` int32 [256] placement words, ``board`` int64 [33] the step's board,
    ``bal_rank`` this rank; local experts are pool slots (``local_offset`` unused)."""
    from ..moe_ld_ops import domain_slots

    if LANDED[0] not in tuple(extra):
        raise ValueError("moe_ldw_landed needs the landed build flag")
    if flags is None:
        flags = DEFAULT_FLAGS
    limit = max_tokens_of(extra)
    if not 1 <= m <= limit:
        raise ValueError(f"num_tokens {m} outside 1..{limit}")
    dev = tok_land.device
    table, n0, n1 = domain_slots(
        dev.index if dev.index is not None else torch.cuda.current_device()
    )
    out = torch.empty(m * TOP_K, HIDDEN, dtype=torch.bfloat16, device=dev)
    ew = torch.empty(m, TOP_K, dtype=torch.bfloat16, device=dev)
    e2p = torch.empty(m * TOP_K, dtype=torch.int32, device=dev)
    s = scratch
    want = PASS_ROWS
    for f in extra:
        if f.startswith("-DMUSE_VR_LDW_PASS_ROWS="):
            want = int(f.split("=", 1)[1])
    if s.max_tokens != limit:
        raise ValueError(f"scratch for {s.max_tokens} rows per call, build for {limit}")
    if s.pass_rows != want:
        raise ValueError(
            f"scratch for {s.pass_rows}-row passes, build with {want}-row passes"
        )
    epi = EPI[0] in tuple(extra)
    epi_args = ()
    tok_out = None
    if epi:
        if norm_w is None:
            raise ValueError("the finalize build needs the post-expert norm weight")
        if getattr(s, "epi_moments", None) is None:
            s.epi_moments = torch.zeros(
                s.hs_perm.shape[0] * EPI_PARTS, dtype=torch.float32, device=dev
            )
            s.epi_pos_slot = torch.zeros(
                s.hs_perm.shape[0], dtype=torch.int32, device=dev
            )
            s.epi_tok_cnt = torch.zeros(2 * s.max_tokens, dtype=torch.int32, device=dev)
        tok_out = torch.empty(m, HIDDEN, dtype=torch.bfloat16, device=dev)
        epi_args = (
            norm_w.float().contiguous(),
            float(norm_eps),
            s.epi_moments,
            s.epi_pos_slot,
            s.epi_tok_cnt,
            tok_out,
        )
    _module(tuple(extra)).run_landed(
        int(m),
        ids_land,
        tok_land,
        seq64,
        last_seq,
        int(ids_pitch),
        int(tok_pitch),
        int(rows_max),
        int(world),
        int(bool(ids_parity)),
        int(bool(tok_parity)),
        w13.view(torch.uint8),
        sf13.view(torch.uint8),
        w2.view(torch.uint8),
        sf2.view(torch.uint8),
        gate_alpha,
        up_alpha,
        out_alpha,
        table,
        out,
        ew.view(torch.int16),
        e2p,
        s.hs_perm,
        s.pts_perm,
        s.img1,
        s.img2,
        s.act,
        s.hs2,
        s.ramax,
        s.cnt,
        s.ctl,
        s.err,
        stamps,
        int(local_offset),
        n0,
        n1,
        GLOBAL_SCALE_INV,
        int(flags),
        int(dbg),
        int(bool(pdl)),
        *epi_args,
        *(
            (bal_table, board, int(bal_rank))
            if any(f.startswith("-DMUSE_VR_LDW_BALB=") for f in extra)
            else ()
        ),
        *(
            (int(test_delay_ns),)
            if any(
                f.startswith("-DMUSE_VR_LDW_TEST_BOARD_DELAY_NS=")
                and not f.endswith("=0")
                for f in extra
            )
            else ()
        ),
    )
    return (out, ew, e2p, tok_out) if epi else (out, ew, e2p)
