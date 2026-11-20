# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Arena and launchers for the cross-rank hop device API v2 (``csrc/lb_hop.cuh``; test kernels ``csrc/lb_hop_ops.cuh``).

A hop moves one block of rows from every rank of a tensor-parallel group to every rank. This module owns the memory:
ONE symmetric allocation per group (separate from every buffer SGLang's own collectives use) holding, per site, its
receive region(s), per ring hop the credit words, and one step-token line per source rank; plus per-rank device state
(the per-forward sequence word ``seq64`` unless the caller supplies the device's one, ring transfer counters, ring
high-water marks, one missed-advance word per consumer unit and site, a sticky error word and its host-visible mirror).

Site policies (``HopSpec.policy``):
  ``unique``  the site owns its region; no slot is reused within a forward. With ``parity=True`` it owns two copies
              selected by the forward parity (sites whose words are stored before the forward's step-acquire point).
  ``ring``    three slots shared by the call sites of one hop (``index`` = call position within the step), slot reuse
              ordered by credits; transfer counters advance per step (:meth:`HopArena.end_step`).

Per forward, the caller raises ``seq64`` once before the forward's first kernel (:meth:`HopArena.forward_head`, or the
device's own sequence word passed as ``seq64=``), and calls :meth:`HopArena.before_forward` on the host before every
forward / replay: when the 16-bit tag space is about to repeat it zeroes the arena on every rank (collective; device
synchronization plus two barriers on ``group``). Kernels receive a site as ``muse_vr::lb_hop::Site<W>`` built from
:meth:`HopArena.table` and the arena's ``Ctl`` (:meth:`HopArena.ctl_table`, :meth:`HopArena.ctl_args`).

Environment: ``MUSE_VR_LB_HOP_MAX_ROUNDS`` (bound of every wait in polling rounds, default 64_000_000),
``MUSE_VR_LB_HOP_TRAP=1`` (tests: trap on an expired wait instead of poisoning outputs and setting the error word),
``MUSE_VR_LB_HOP_ASYNC_CREDIT=1`` (ring sites: every consumer CTA releases its credit from the consuming kernel itself,
default off; the separate release kernel is then skipped).
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Optional, Sequence

import torch

# The SGLang names this module reads (test/unit/test_contract.py pins them against the supported SGLang)
SGLANG_READS = {
    "srt/distributed/device_communicators/custom_all_reduce_v2.py": (
        "_allocate_symmetric_memory",
    ),
    "kernels/jit/utils/__init__.py": (
        "is_arch_support_pdl",
        "load_jit",
        "make_cpp_args",
    ),
}


K_MAX_WORLD = 16
K_RING_SLOTS = 3
K_LINE_WORDS = 16  # 64-bit words per 128-byte line
TAG_PERIOD = 65535
FLAG_TRAP = 1
FLAG_SKIP_CREDIT_UNSAFE = 2
FLAG_ASYNC_CREDIT = 4
LOOPBACK_DEV = 0x100  # policy-word bit: development no-flight twin (every push lands in this rank's own copy)
POLICY = {"unique": 0, "ring": 1}
_ALIGN = 4096
_CREDIT_BYTES = 256
_CSRC = Path(__file__).resolve().parent.parent / "csrc"
_CUH = _CSRC / "lb_hop_ops.cuh"

# sha256 of the SGLang source whose symmetric-memory helper allocates the arena
PINS = {
    "sglang/srt/distributed/device_communicators/custom_all_reduce_v2.py": (
        "277da88e2837e585f3ebdf4122906d29ea7a96cc30b981f7a028f6f63b6cf001",
        "f7b23a55019734be7b2f1958e20cc1f78a31251a035f2d75885911aa4a46e6b5",
    ),
}


def pitch_words(width: int) -> int:
    """64-bit words per row: the row's words (three elements each) padded to whole 128-byte lines."""
    words = (width + 2) // 3
    return (words + K_LINE_WORDS - 1) // K_LINE_WORDS * K_LINE_WORDS


@dataclass(frozen=True)
class CarEntry:
    """car_table v2 row for one world size and row bucket."""

    format: str  # "tag64_1shot" (every rank pushes its rows to every rank) or "s32_2shot" (not implemented here)
    slots: str  # "unique" (per-layer-unique sites, M <= 16) or "ring" (3 slots + credits, M >= 32)
    collect_n: (
        int  # items per lane per polling round (N x world <= 32 vectors per lane)
    )
    engage: bool  # the bucket's measured win vs the stock collective (False until measured: the component steps aside)


def car_table(world: int, m: int) -> CarEntry:
    """Defaults to measure: TAG64 1-shot up to M32, S32 2-shot from M64; per-layer-unique slots up to
    M16, ring-3 with credits above; collect batch N as the library's row collectors use it (4 / 2 / 1 at world <= 4 / 8 /
    16, within N x world <= 32 vectors per lane). ``engage`` stays False
    until the bucket's win is measured in-chain; callers step aside to the stock collective then."""
    if world not in (1, 2, 4, 8, 16):
        raise ValueError("world size must be 1, 2, 4, 8 or 16")
    fmt = "tag64_1shot" if m <= 32 else "s32_2shot"
    slots = "unique" if m <= 16 else "ring"
    return CarEntry(
        format=fmt,
        slots=slots,
        collect_n=1 if world >= 16 else (2 if world >= 8 else 4),
        engage=False,
    )


@dataclass(frozen=True)
class HopSpec:
    name: str
    rows_max: int
    width: int  # elements per row, multiple of 6
    policy: str = "unique"
    parity: bool = False  # unique sites only
    units: int = 1  # consumer units of the missed-advance guard (CTAs or role slots)
    calls: int = (
        1  # ring: call sites per step (one missed-advance word per call site and unit)
    )

    def copies(self) -> int:
        return K_RING_SLOTS if self.policy == "ring" else (2 if self.parity else 1)

    def pitch(self) -> int:
        return pitch_words(self.width)

    def region_bytes(self, world: int) -> int:
        return self.copies() * world * self.rows_max * self.pitch() * 8


def _env_flag(name: str) -> bool:
    return os.environ.get(name, "0").strip().lower() in ("1", "true", "yes", "on")


def _align(n: int) -> int:
    return (n + _ALIGN - 1) // _ALIGN * _ALIGN


def allocate_symmetric(nbytes: int, device: torch.device, group):
    """Symmetric allocation on ``group`` (SGLang's helper; collective)."""
    from sglang.srt.distributed.device_communicators.custom_all_reduce_v2 import (
        _allocate_symmetric_memory,
    )

    return _allocate_symmetric_memory(nbytes, device=device, group=group)


class _LocalSymm:
    """World-1 stand-in for the symmetric allocation (loopback tests): one local buffer."""

    def __init__(self, t: torch.Tensor):
        self.t = t

    def get_buffer(self, r, shape, dtype):
        assert r == 0
        return self.t.view(dtype)[: shape[0]]


class HopArena:
    """Receive regions, credits, step tokens and per-rank state for a set of hop sites on one tensor-parallel group.

    Every rank of ``group`` constructs it with the same specs at the same point (collective), before any CUDA graph
    that uses it is captured.
    """

    def __init__(
        self,
        group,
        device: torch.device,
        rank: int,
        world: int,
        specs: Sequence[HopSpec],
        *,
        seq64: Optional[torch.Tensor] = None,
        max_rounds: Optional[int] = None,
        spin_rounds: int = 4096,
        backoff_ns: int = 256,
        trap: Optional[bool] = None,
        async_credit: Optional[bool] = None,
        loopback_dev: bool = False,
        multicast: Optional[bool] = None,
        allocator=None,
    ):
        import torch.distributed as dist

        if world not in (1, 2, 4, 8, 16):
            raise ValueError("world size must be 1, 2, 4, 8 or 16")
        names = [s.name for s in specs]
        if not specs or len(set(names)) != len(names):
            raise ValueError("hop specs: at least one, unique names")
        for s in specs:
            if (
                s.width <= 0
                or s.width % 6
                or s.rows_max <= 0
                or s.units <= 0
                or s.policy not in POLICY
            ):
                raise ValueError(
                    f"hop {s.name}: width a positive multiple of 6, rows_max / units positive, policy unique|ring"
                )
            if s.parity and s.policy != "unique":
                raise ValueError(f"hop {s.name}: parity applies to unique sites only")
            if s.calls < 1 or (s.calls > 1 and s.policy != "ring"):
                raise ValueError(
                    f"hop {s.name}: calls >= 1, and > 1 only for ring hops"
                )
        self.rank, self.world, self.device, self.group = rank, world, device, group
        self.specs = {s.name: s for s in specs}
        self.order = names
        self.max_rounds = int(
            max_rounds
            if max_rounds is not None
            else int(os.environ.get("MUSE_VR_LB_HOP_MAX_ROUNDS", "64000000"))
        )
        self.spin_rounds, self.backoff_ns = int(spin_rounds), int(backoff_ns)
        self.trap = _env_flag("MUSE_VR_LB_HOP_TRAP") if trap is None else bool(trap)
        self.async_credit = (
            _env_flag("MUSE_VR_LB_HOP_ASYNC_CREDIT")
            if async_credit is None
            else bool(async_credit)
        )
        # development twin only (measurement of a hop's exposed time): pushes land in this rank's own copy at every
        # source position; consumers then never wait on another GPU. Results computed under it are not valid outputs.
        self.loopback_dev = bool(loopback_dev)

        # layout of the symmetric allocation (identical on every rank)
        self._off = {}
        off = 0
        for s in specs:
            reg = _align(s.region_bytes(world))
            # ring credit area: the sync form's word per consumer rank, or the async form's word per (consumer rank,
            # consumer CTA)
            cred = (
                _align(max(_CREDIT_BYTES, 4 * K_MAX_WORLD * s.units))
                if s.policy == "ring"
                else 0
            )
            self._off[s.name] = (off, off + reg if cred else -1)
            off += reg + cred
        self._tok_off = off
        off += _align(K_MAX_WORLD * K_LINE_WORDS * 8)
        self.total_bytes = off
        if world == 1 and allocator is None:
            t = torch.zeros(self.total_bytes, dtype=torch.uint8, device=device)
            self._tensor, symm = t, _LocalSymm(t)
        else:
            self._tensor, symm = (allocator or allocate_symmetric)(
                self.total_bytes, device, group
            )
        self._views = [
            symm.get_buffer(r, [self.total_bytes], torch.uint8) for r in range(world)
        ]
        self._views[rank].zero_()
        # multicast pushes (default off): each pushed word is stored once to every rank's copy through the allocation's
        # multicast address; without one (or on the loopback twin, or world 1) pushes stay one store per rank, loudly.
        want_mc = (
            _env_flag("MUSE_VR_LB_HOP_MULTICAST")
            if multicast is None
            else bool(multicast)
        )
        self._mc_base = int(getattr(symm, "multicast_ptr", 0) or 0) if world > 1 else 0
        self.multicast = want_mc and self._mc_base != 0 and not self.loopback_dev
        if want_mc and not self.multicast:
            import logging

            logging.getLogger(__name__).warning(
                "lb_hop: multicast pushes requested but %s; using one store per rank",
                "the arena is the loopback twin"
                if self.loopback_dev
                else "the allocation has no multicast address",
            )

        # per-rank device state
        rings = [s.name for s in specs if s.policy == "ring"]
        self._ring_idx = {n: i for i, n in enumerate(rings)}
        nr = max(1, len(rings))
        self.seq64 = (
            seq64
            if seq64 is not None
            else torch.ones(1, dtype=torch.int64, device=device)
        )
        self.own_seq = seq64 is None
        self.ring_base = torch.zeros(nr, dtype=torch.int64, device=device)
        self.hwm = torch.zeros(
            nr * 2 * K_RING_SLOTS, dtype=torch.int32, device=device
        )  # two alternating copies per slot
        self._unit_off = {}
        u = 0
        for s in specs:
            self._unit_off[s.name] = u
            u += s.units * s.calls
        self.last_seq = torch.zeros(u, dtype=torch.int64, device=device)
        self.err = torch.zeros(1, dtype=torch.int32, device=device)
        self.err_host = torch.zeros(1, dtype=torch.int32, pin_memory=True)
        torch.cuda.synchronize(device)
        if world > 1:
            dist.barrier(group=group)

        self._tables = {}
        for s in specs:
            ro, co = self._off[s.name]
            recv = [v.data_ptr() + ro for v in self._views]
            cred = [v.data_ptr() + co for v in self._views] if co >= 0 else [0] * world
            ri = self._ring_idx.get(s.name, 0)
            ptrs = (
                recv
                + cred
                + [
                    self.seq64.data_ptr(),
                    self.ring_base.data_ptr() + 8 * ri,
                    self.hwm.data_ptr() + 4 * 2 * K_RING_SLOTS * ri,
                    self.last_seq.data_ptr() + 8 * self._unit_off[s.name],
                ]
            )
            self._tables[s.name] = torch.tensor(ptrs, dtype=torch.int64, device="cpu")
        self._tok_table = torch.tensor(
            [v.data_ptr() + self._tok_off for v in self._views]
            + [self.seq64.data_ptr()],
            dtype=torch.int64,
            device="cpu",
        )
        self._ctl = torch.tensor(
            [self.err.data_ptr(), self.err_host.data_ptr()],
            dtype=torch.int64,
            device="cpu",
        )
        self._counts = [0] * nr
        self._ring_total = (
            [0] * nr
        )  # host mirror of every ring counter since the last scrub (wrap bookkeeping)
        self.forwards = 0  # host count of before_forward() calls since the last scrub (wrap bookkeeping)
        self.scrubs = 0

    # ------------------------------------------------------------------------------------------- kernel arguments
    def push_table(self, name: str, index: int = 0) -> torch.Tensor:
        """table() for producer kernels: with multicast pushes on, one more entry = the multicast address of the site's
        receive region (the kernels then store each word once to every rank); else table() itself."""
        t = self.table(name, index)
        if not self.multicast:
            return t
        key = ("mc", name, index)
        if key not in self._tables:
            self._tables[key] = torch.cat(
                [
                    t,
                    torch.tensor(
                        [self._mc_base + self._off[name][0]], dtype=torch.int64
                    ),
                ]
            )
        return self._tables[key]

    def table(self, name: str, index: int = 0) -> torch.Tensor:
        """Host int64 tensor: recv of every rank, credit of every rank, seq64, ring counter, high-water, last_seq (the
        missed-advance words of call site ``index``)."""
        if index == 0:
            return self._tables[name]
        s = self.specs[name]
        if not 0 <= index < s.calls:
            raise ValueError(f"hop {name}: call index {index} outside 0..{s.calls - 1}")
        key = (name, index)
        if key not in self._tables:
            t = self._tables[name].clone()
            t[-1] += 8 * s.units * index
            self._tables[key] = t
        return self._tables[key]

    def tok_table(self) -> torch.Tensor:
        return self._tok_table

    def ctl_table(self) -> torch.Tensor:
        return self._ctl

    def flags(self) -> int:
        return (FLAG_TRAP if self.trap else 0) | (
            FLAG_ASYNC_CREDIT if self.async_credit else 0
        )

    def ctl_args(self, flags: Optional[int] = None) -> tuple:
        """(max_rounds, spin_rounds, flags, backoff_ns) as the launchers take them."""
        return (
            self.max_rounds,
            self.spin_rounds,
            self.flags() if flags is None else int(flags),
            self.backoff_ns,
        )

    def site_args(
        self, name: str, index: int = 0, site_no: Optional[int] = None
    ) -> tuple:
        """(rank, rows_max, width, pitch, policy, parity, index, site, units) as the launchers take them (after the
        site table and, where present, the Ctl table)."""
        s = self.specs[name]
        return (
            self.rank,
            s.rows_max,
            s.width,
            s.pitch(),
            POLICY[s.policy] | (LOOPBACK_DEV if self.loopback_dev else 0),
            int(s.parity),
            int(index),
            self.order.index(name) if site_no is None else int(site_no),
            s.units,
        )

    def spec(self, name: str) -> HopSpec:
        return self.specs[name]

    # ------------------------------------------------------------------------------------------- per forward / step
    def forward_head(self, release: bool = False) -> None:
        """Raises seq64 by one on the current stream (captured into a graph when capturing). The first chain kernel
        after it must be launched without programmatic stream serialization. ``release``: the same kernel also runs
        this rank's step_release for the forward (then no hop kernel of the forward releases; every one that stores hop
        words after the head acquires)."""
        if release:
            _module("seqrel", self.world, False).bump_release(
                self.seq64, self._tok_table, self.rank
            )
        else:
            _module("seq", 1, False).bump(self.seq64)

    def before_forward(
        self, n: int = 1, ring_transfers: Optional[Sequence[int]] = None
    ) -> bool:
        """Host side, before every forward / replay on every rank (collective when it scrubs): zeroes the arena and
        resets ring state when the next ``n`` forwards would reach a forward whose unique-site tags repeat those of an
        earlier one, or when any ring counter would advance by 65535 or more transfers since the last scrub (the ring
        tag is the ring counter, several transfers per forward). ``ring_transfers``: transfers per ring the next
        forwards make (default: the last step's counts, times n). Returns True when it scrubbed."""
        add = (
            list(ring_transfers)
            if ring_transfers is not None
            else [c * n for c in self._counts]
        )
        wrap_ring = any(t + a >= TAG_PERIOD - 1 for t, a in zip(self._ring_total, add))
        if self.forwards + n < TAG_PERIOD - 1 and not wrap_ring:
            self.forwards += n
            self._ring_total = [t + a for t, a in zip(self._ring_total, add)]
            return False
        self.scrub_all()
        self.forwards = n
        self._ring_total = list(add)
        return True

    def scrub_all(self) -> None:
        """Collective: every rank zeroes its own receive regions, credits, tokens and ring state, between two barriers
        (no rank's next words can land before every rank's zeroing completed)."""
        import torch.distributed as dist

        torch.cuda.synchronize(self.device)
        if self.world > 1:
            dist.barrier(group=self.group)
        self._views[self.rank].zero_()
        self.ring_base.zero_()
        self.hwm.zero_()
        torch.cuda.synchronize(self.device)
        if self.world > 1:
            dist.barrier(group=self.group)
        self.scrubs += 1

    def begin_step(self) -> None:
        self._counts = [0] * len(self._counts)

    def next_index(self, name: str) -> int:
        i = self._ring_idx[name]
        idx = self._counts[i]
        self._counts[i] += 1
        return idx

    def end_step(self) -> None:
        """Appends the ring counter advance to the current stream (captured into a graph when capturing)."""
        if not self._ring_idx or not any(self._counts):
            return
        _module("advance", 1, True).run(
            self.ring_base, torch.tensor(self._counts, dtype=torch.int64, device="cpu")
        )

    # ------------------------------------------------------------------------------------------- errors / tests
    def error_word(self) -> int:
        return int(self.err_host.item()) & 0xFFFFFFFF

    def check(self) -> None:
        v = self.error_word()
        if v:
            raise RuntimeError(
                f"muse_vr lb_hop: a bounded wait or guard failed (error word 0x{v:08x}: code {(v >> 16) & 0x7FFF}, "
                f"rank {(v >> 8) & 0xFF}, site {v & 0xFF})"
            )

    def reset(self) -> None:
        """Test helper: zero every region, credit, token, counter, guard word and error word on every rank."""
        import torch.distributed as dist

        torch.cuda.synchronize(self.device)
        if self.world > 1:
            dist.barrier(group=self.group)
        self._views[self.rank].zero_()
        self.ring_base.zero_()
        self.hwm.zero_()
        self.last_seq.zero_()
        self.err.zero_()
        self.err_host.zero_()
        if self.own_seq:
            self.seq64.fill_(1)
        torch.cuda.synchronize(self.device)
        if self.world > 1:
            dist.barrier(group=self.group)
        self.forwards = 0

    def set_seq(self, value: int) -> None:
        """Test helper: set this rank's seq64 (tag-wrap tests)."""
        self.seq64.fill_(int(value))

    def set_ring_counter(self, name: str, value: int) -> None:
        self.ring_base[self._ring_idx[name]] = int(value)

    def units_slice(self, name: str) -> torch.Tensor:
        o = self._unit_off[name]
        s = self.specs[name]
        return self.last_seq[o : o + s.units * s.calls]


# ---------------------------------------------------------------------------------------------- JIT modules
@lru_cache(maxsize=None)
def _module(kind: str, world: int, use_pdl: bool):
    from sglang.kernels.jit.utils import is_arch_support_pdl, load_jit, make_cpp_args

    pdl = bool(use_pdl) and is_arch_support_pdl()
    if kind == "seq":
        args, wrappers = (), [("bump", "MuseVrLbHopSeq::bump")]
    elif kind == "seqrel":
        args = make_cpp_args(world)
        wrappers = [("bump_release", f"MuseVrLbHopSeqRel<{args}>::bump_release")]
    elif kind == "advance":
        args = make_cpp_args(pdl)
        wrappers = [("run", f"MuseVrLbHopAdvance<{args}>::run")]
    elif kind == "rows":
        args = make_cpp_args(world, pdl)
        wrappers = [
            ("push", f"MuseVrLbHopRows<{args}>::push"),
            ("collect", f"MuseVrLbHopRows<{args}>::collect"),
        ]
    else:
        args = make_cpp_args(world, pdl)
        wrappers = [
            ("push", f"MuseVrLbHopTest<{args}>::push"),
            ("collect", f"MuseVrLbHopTest<{args}>::collect"),
            ("release", f"MuseVrLbHopTest<{args}>::release"),
            ("late_write", f"MuseVrLbHopTest<{args}>::late_write"),
        ]
    return load_jit(
        f"muse_vr_lb_hop_{kind}"
        + (f"_w{world}" if kind in ("rows", "test", "seqrel") else ""),
        *args,
        cuda_files=[str(_CUH)],
        cuda_wrappers=wrappers,
        extra_include_paths=[str(_CSRC)],
    )


def push_rows(
    arena: HopArena, name: str, x: torch.Tensor, *, index: int = 0, blocks: int = 64
) -> None:
    """Producer of one transfer: this rank's x [rows, width] (bf16) into every rank's copy (ring: credit acquire first)."""
    _module("rows", arena.world, True).push(
        arena.push_table(name, index),
        arena.ctl_table(),
        *arena.site_args(name, index),
        *arena.ctl_args(),
        x,
        int(blocks),
    )


def collect_rows(
    arena: HopArena,
    name: str,
    out: torch.Tensor,
    *,
    index: int = 0,
    gather: bool = False,
) -> None:
    """Consumer: out [rows, width] (bf16) = bf16(fp32 rank-order sum), or with gather=True out [world, rows, width] =
    every source's block. One consumer unit per CTA (grid = the site's units). Ring credits are released by a later
    kernel (not here)."""
    _module("rows", arena.world, True).collect(
        arena.table(name, index),
        arena.ctl_table(),
        *arena.site_args(name, index),
        *arena.ctl_args(),
        out,
        int(gather),
    )


class TestOps:
    """Self-test kernels bound to one site of an arena (tests/dist_lb_hop.py)."""

    (
        MODE_SKIP_PUSH,
        MODE_EARLY_STORE,
        MODE_BENCH,
        MODE_TILE,
        MODE_NO_GUARD,
        MODE_STEP_TOK,
        MODE_RING_SCRUB,
    ) = (1, 2, 4, 8, 16, 32, 64)
    MODE_NO_RELEASE, MODE_FUZZ_READS, MODE_EARLY_CREDIT, MODE_REPLICATE = (
        128,
        256,
        512,
        1024,
    )

    def __init__(self, arena: HopArena, name: str):
        self.arena, self.name = arena, name
        self.spec = arena.spec(name)
        self._anchor = torch.zeros(1, dtype=torch.int32, device=arena.device)
        self._empty = torch.zeros(0, dtype=torch.int32, device=arena.device)

    def _m(self):
        return _module("test", self.arena.world, True)

    def push(
        self,
        rows: int,
        *,
        index: int = 0,
        fuzz: int = 0,
        mode: int = 0,
        tile_cols: int = 0,
        audit=None,
        bad=None,
        blocks: int = 64,
        pdl: bool = True,
        flags: Optional[int] = None,
    ) -> None:
        a = self.arena
        self._m().push(
            a.push_table(self.name, index),
            a.ctl_table(),
            a.tok_table(),
            *a.site_args(self.name, index),
            *a.ctl_args(flags),
            int(rows),
            int(fuzz),
            int(mode),
            int(tile_cols),
            self._empty if audit is None else audit,
            self._anchor if bad is None else bad,
            int(blocks),
            int(pdl),
        )

    def collect(
        self,
        rows: int,
        out: torch.Tensor,
        mismatch: torch.Tensor,
        *,
        index: int = 0,
        fuzz: int = 0,
        mode: int = 0,
        flags: Optional[int] = None,
    ) -> None:
        a = self.arena
        self._m().collect(
            a.table(self.name, index),
            a.ctl_table(),
            *a.site_args(self.name, index),
            *a.ctl_args(flags),
            int(rows),
            int(fuzz),
            int(mode),
            out,
            mismatch,
        )

    def release(self, rows: int, *, index: int = 0) -> None:
        if self.arena.async_credit:  # released by the consuming kernel's CTAs
            return
        self._m().release(
            self.arena.table(self.name, index),
            *self.arena.site_args(self.name, index),
            int(rows),
            self._anchor,
        )

    def late_write(self, out: torch.Tensor, delay_units: int, value: float) -> None:
        self._m().late_write(out, int(delay_units), float(value))

    def audit_buffer(self) -> torch.Tensor:
        s = self.spec
        return torch.zeros(
            self.arena.world * s.rows_max * (s.pitch() // K_LINE_WORDS),
            dtype=torch.int32,
            device=self.arena.device,
        )
