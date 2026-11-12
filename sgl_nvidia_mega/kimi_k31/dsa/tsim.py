# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""TSIM: the token-split indexer's local selection, exchange and exact merge in ONE kernel per DSA layer (tsim.cuh).

Replaces TSI's stock local top-k + v6 / s1r merge (tsi.py) under ``K31MK_TSI_MERGE=tsim`` (serving integration: next
step). Every rank ends with the same exact top-2048 KV slots in the s1r order (source-major, position-
ascending inside a source); ``fallback[row]`` holds the flag bits of rows that took the exact fallback (0 = fast path).

This module: the JIT loader, the region layout and tags (tsim.cuh), and ``Emulator`` (tests: all 16 ranks on one GPU,
cooperative launch, unicast pushes into 16 regions).
"""

from __future__ import annotations

import os
from functools import cache
from pathlib import Path

import torch

SGLANG_READS = {
    "kernels/jit/utils/__init__.py": ("load_jit",),
}

WORLD = 16
TOPK = 2048
PAGE = 64
CAP_SLOTS = 386  # tsim.cuh kSlots (384 records + 2 header elements)
KINDS = 2
KIND_CAND, KIND_FULL = 0, 1
META = 8
_SOURCE = Path(__file__).with_name("tsim.cuh")
# TSIM-LEAN: the same selection, emission order and flags as tsim.cuh, with five
# receiver / sender levers as compile-time switches. K31MK_TSIM_LEAN unset / 0 = the served tsim.cuh (default).
_LEAN_SOURCE = Path(__file__).with_name("tsim_lean.cuh")
_LEAN_KNOBS = {"hoist": "K31_TSIML_HOIST", "tbl": "K31_TSIML_TBL", "bpoll": "K31_TSIML_BPOLL",
               "nocoarse": "K31_TSIML_NOCOARSE", "widetie": "K31_TSIML_WIDETIE", "pushc": "K31_TSIML_PUSHC"}
# TSIM-PUSHC: with K31MK_TSIM_MC=1 the sender
# stages its records in SMEM (as the unicast push) and stores them lane-contiguously through the multicast VA; same tagged elements, same
# addresses, same E-R / E-T protocol. Only by name in the list: "1" / "all" keep the five original levers (_LEAN_ALL).
_LEAN_ALL = ("bpoll", "hoist", "nocoarse", "tbl", "widetie")
# TSIM-PUSHC row gate: with "pushc" in K31MK_TSIM_LEAN, a launch takes the PUSHC push only when it has
# >= K31MK_TSIM_PUSHC_MIN_ROWS rows (default 24; tsim_lean.cuh K31_TSIML_PUSHC_MIN_ROWS); smaller launches keep the LEAN push. Ignored without "pushc".


def pushc_min_rows(value: str | None = None) -> int:
    v = (os.environ.get("K31MK_TSIM_PUSHC_MIN_ROWS", "") if value is None else value).strip()
    n = int(v) if v else 24
    if n < 1:
        raise ValueError(f"K31MK_TSIM_PUSHC_MIN_ROWS={v!r}: must be >= 1")
    return n


def lean_levers(value: str | None = None) -> tuple[str, ...]:
    """K31MK_TSIM_LEAN: unset / 0 = served tsim.cuh; 1 / all = every lever; else a comma list of hoist, tbl, bpoll, nocoarse,
    widetie, pushc. Read once, at the first load() (boot); unknown names raise."""
    v = (os.environ.get("K31MK_TSIM_LEAN", "") if value is None else value).strip().lower()
    if v in ("", "0", "off", "none"):
        return ()
    if v in ("1", "all", "on"):
        return _LEAN_ALL
    on = tuple(sorted({x.strip() for x in v.split(",") if x.strip()}))
    bad = [x for x in on if x not in _LEAN_KNOBS]
    if bad or not on:
        raise ValueError(f"K31MK_TSIM_LEAN={v!r}: unknown knob(s) {bad}; known {sorted(_LEAN_KNOBS)}")
    return on


@cache
def load():
    import sglang
    from sglang.kernels.jit.utils import load_jit

    levers = lean_levers()
    pcr = pushc_min_rows() if "pushc" in levers else 0
    tag = ("k31_tsim", "w16") + (("lean_" + "_".join(levers),) if levers else ())
    tag = tag + ((f"pcr{pcr}",) if pcr else ())
    lean_flags = [f"-D{_LEAN_KNOBS[k]}={int(k in levers)}" for k in sorted(_LEAN_KNOBS)] if levers else []
    lean_flags += [f"-DK31_TSIML_PUSHC_MIN_ROWS={pcr}"] if pcr else []
    return load_jit(
        *tag,
        cuda_files=[str(_LEAN_SOURCE if levers else _SOURCE)],
        cuda_wrappers=[
            ("run", "k31_tsim::Binding::run"),
            ("bench", "k31_tsim::Binding::bench"),
            ("emulate", "k31_tsim::Binding::emulate"),
            ("max_coresident", "k31_tsim::Binding::max_coresident"),
        ],
        extra_cuda_cflags=lean_flags,
        extra_include_paths=[str(Path(sglang.__file__).parent / "kernels/jit/csrc")],
    )


def tag(epoch: int, ordinal: int, kind: int, layers: int = 24) -> int:
    """= tsim.cuh make_tag (= tsi.py tag)."""
    value = ((epoch * layers + ordinal) * KINDS + kind + 1) & 0xFFFFFFFF
    return value or 1


def region_records(cap: int) -> int:
    """int64 records of one rank's region: CAND [2][cap][16][386][2] + FULL [2][cap][2048][2]."""
    return 2 * cap * WORLD * CAP_SLOTS * 2 + 2 * cap * TOPK * 2


def groups(ncols: int) -> int:
    return -(-(ncols // 4) // 1024)


class Emulator:
    """Tests: the 16 ranks of one TP16 group on ONE GPU (tsim.cuh Binding::emulate). Each rank owns a region; the
    kernel pushes every record into all 16 regions with unicast stores; a cooperative launch keeps every CTA resident
    (the ranks poll each other). A poll deadline sets flag 8 instead of trapping (shared GPUs)."""

    def __init__(self, device="cuda", cap: int = 32, layers: int = 24, deadline_ns: int = 2_000_000_000):
        self.device = torch.device(device)
        self.cap, self.layers, self.deadline_ns = cap, layers, int(deadline_ns)
        self.module = load()
        self.regions = [torch.zeros(region_records(cap), dtype=torch.int64, device=self.device) for _ in range(WORLD)]
        self.epoch = torch.zeros(4, dtype=torch.int32, device=self.device)
        info = torch.zeros(2, dtype=torch.int32)
        self.module.max_coresident(info, 1)
        self.sms, self.per_sm_big = int(info[0]), int(info[1])
        self.module.max_coresident(info, 0)
        self.per_sm = int(info[1])

    def meta(self, ordinal: int) -> torch.Tensor:
        values = [WORLD, 0, ordinal, self.layers, self.cap, self.deadline_ns, 0, 2]  # test_mode 2: no trap (flag 8)
        return torch.tensor(values + [int(t.data_ptr()) for t in self.regions], dtype=torch.int64)

    def advance(self) -> None:
        """A new forward (prep_kernel's epoch += 1)."""
        self.epoch[0] += 1

    def select(self, ordinal, scores, lcausal, causal, table, rows_per_request, out, fallback, scratch, ncols):
        """scores [16, rows, >= ncols] float32 (rank r's local rows), lcausal [16, rows], causal [rows], table
        [rows / rows_per_request, pages], out [16, rows, 2048], fallback [16, rows], scratch [16, rows, 16 * 2048]."""
        rows = out.shape[1]
        per = self.per_sm_big if groups(ncols) > 8 else self.per_sm
        chunk = max(1, min(rows, (self.sms * per) // WORLD))
        meta = self.meta(ordinal)
        for r0 in range(0, rows, chunk):
            n = min(chunk, rows - r0)
            self.module.emulate(meta, self.epoch, scores, lcausal, causal, table, out, fallback, scratch,
                                rows_per_request, ncols, r0, n)
