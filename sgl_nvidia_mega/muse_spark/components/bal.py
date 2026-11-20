# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Per-step expert-parallel assignment over a fixed replica pool (component ``bal``; development form).

Placement (fixed arithmetic, independent of any traffic; the ep_balance rules): with ``W`` expert-parallel ranks and 256
routed experts, expert ``e`` lives on its home rank ``e // (256 // W)``; every expert with ``e % k == 0`` has a second copy
on rank ``(home + 1 + e % 3) % 4`` (``W == 4``) or ``(home + 1 + (e % 32) % 7) % 8`` (``W == 8``). Every rank holds a pool of
``home experts + the largest replica count any rank receives`` slots: home experts keep their stock slot, replicas follow
in ascending expert id.

Assignment (``csrc/bal_assign.cuh``, one warp, integer arithmetic, identical on every rank): per step, every routed
expert gets a cost from its routed row count (first pass + further passes of ``pass_rows`` rows); single-holder experts go
home; replicated experts in (row passes descending, key ascending) order, key(e) = (e % per) * W + e // per (home ranks
interleaved), go to the holder with the lower load (ties: home), the loads read once per group of 4 consecutive entries;
then at most 2 single moves off the max-load rank while they lower that rank's pair maximum. ``assign_reference`` is the
CPU reference the tests compare the kernel with.
"""

from __future__ import annotations

import functools
from pathlib import Path
from typing import Optional

import torch

# The SGLang names this module reads (test/unit/test_contract.py pins them against the supported SGLang)
SGLANG_READS = {
    "kernels/jit/utils/__init__.py": ("load_jit", "make_cpp_args"),
}


NUM_EXPERTS = 256
TOP_K = 8
MAX_ROWS = 512
COST = {
    4: (184, 197, 21),
    8: (147, 440, 0),
}  # per world: a, b (/256 per row), p (per further pass)
REPAIR_MAX = 2
GROUP = 4
UNROUTED = 0xFF
PAD_ID = 0xFFFF
NUM_STAMPS = 6
_CUH = Path(__file__).resolve().parent / "csrc" / "bal_assign.cuh"


def _secondary(e: int, world: int) -> int:
    per = NUM_EXPERTS // world
    p = e // per
    if world == 4:
        return (p + 1 + (e % 3)) % 4
    if world == 8:
        return (p + 1 + ((e % 32) % 7)) % 8
    raise ValueError(f"no replica rule for world {world}")


class Placement:
    """Fixed-rule replica placement: ``world`` ranks, every expert with ``e % k == 0`` replicated (k = 0: none)."""

    def __init__(self, world: int, k: int):
        if world not in (4, 8) or k < 0:
            raise ValueError(f"bad placement world {world} k {k}")
        self.world, self.k = world, k
        per = NUM_EXPERTS // world
        self.per = per
        self.home = [e // per for e in range(NUM_EXPERTS)]
        self.sec = [
            (_secondary(e, world) if k > 0 and e % k == 0 else -1)
            for e in range(NUM_EXPERTS)
        ]
        received = [0] * world
        for e in range(NUM_EXPERTS):
            if self.sec[e] >= 0:
                received[self.sec[e]] += 1
        self.pool = per + max(received)
        self.home_slot = [e - self.home[e] * per for e in range(NUM_EXPERTS)]
        self.sec_slot = [-1] * NUM_EXPERTS
        nxt = [per] * world
        for e in range(NUM_EXPERTS):
            if self.sec[e] >= 0:
                r = self.sec[e]
                self.sec_slot[e] = nxt[r]
                nxt[r] += 1
        self.slot_expert = [[-1] * self.pool for _ in range(world)]
        for e in range(NUM_EXPERTS):
            self.slot_expert[self.home[e]][self.home_slot[e]] = e
            if self.sec[e] >= 0:
                self.slot_expert[self.sec[e]][self.sec_slot[e]] = e

    @classmethod
    def from_rule(cls, rule: str) -> Placement:
        """ep_balance rule names: f1/4, f1/8 (world 4), f1/2, rf2 (world 8); plus f1/4w8, f1/8w8 (world 8), none4, none8."""
        table = {
            "f1/4": (4, 4),
            "f1/8": (4, 8),
            "f1/2": (8, 2),
            "rf2": (8, 1),
            "f1/4w8": (8, 4),
            "f1/8w8": (8, 8),
            "none4": (4, 0),
            "none8": (8, 0),
        }
        if rule not in table:
            raise ValueError(f"unknown rule {rule}")
        return cls(*table[rule])

    def words(self) -> list:
        out = []
        for e in range(NUM_EXPERTS):
            sr = self.sec[e] if self.sec[e] >= 0 else 0xF
            ss = self.sec_slot[e] if self.sec[e] >= 0 else 0
            out.append(
                (self.home[e] & 0xF)
                | ((sr & 0xF) << 4)
                | ((self.home_slot[e] & 0xFF) << 8)
                | ((ss & 0xFF) << 16)
            )
        return out

    def table(self, device) -> torch.Tensor:
        return (
            torch.tensor(self.words(), dtype=torch.int64)
            .to(torch.int32)
            .to(device)
            .contiguous()
        )


def costs(rows_per_expert, pass_rows: int, world: int) -> list:
    a, b, pp = COST[world]
    out = []
    for n in rows_per_expert:
        n = int(n)
        p = (n + pass_rows - 1) // pass_rows
        out.append(a + ((b * n) >> 8) + pp * (p - 1) if n else 0)
    return out


def assign_reference(packed: torch.Tensor, pl: Placement, pass_rows: int) -> list:
    """CPU reference of assign_warp: per expert the assigned rank (UNROUTED if not routed)."""
    ids = ((packed.cpu().to(torch.int64) >> 16) & 0xFFFF).reshape(-1).tolist()
    cnt = [0] * NUM_EXPERTS
    for e in ids:
        if e < NUM_EXPERTS:
            cnt[e] += 1
    c = costs(cnt, pass_rows, pl.world)
    W = pl.world
    asg = [UNROUTED] * NUM_EXPERTS
    load = [0] * W
    for e in range(NUM_EXPERTS):
        if cnt[e]:
            asg[e] = pl.home[e]
            if pl.sec[e] < 0:
                load[pl.home[e]] += c[e]
    per = NUM_EXPERTS // W
    passes = [(n + pass_rows - 1) // pass_rows for n in cnt]
    mov = sorted(
        (e for e in range(NUM_EXPERTS) if cnt[e] and pl.sec[e] >= 0),
        key=lambda e: (-passes[e], (e % per) * W + e // per),
    )
    for g0 in range(0, len(mov), GROUP):
        snap = list(load)
        for e in mov[g0 : g0 + GROUP]:
            h, s = pl.home[e], pl.sec[e]
            t = s if snap[s] < snap[h] else h
            asg[e] = t
            load[t] += c[e]
    for _ in range(REPAIR_MAX):
        mx = max(load)
        rm = load.index(mx)
        if mx - min(load) <= COST[W][0]:
            break
        best = None
        for e in mov:
            if asg[e] != rm:
                continue
            ot = pl.sec[e] if rm == pl.home[e] else pl.home[e]
            nm = max(mx - c[e], load[ot] + c[e])
            if nm < mx:
                key = (nm << 9) | e
                if best is None or key < best:
                    best = key
        if best is None:
            break
        e = best & 0x1FF
        ot = pl.sec[e] if rm == pl.home[e] else pl.home[e]
        load[rm] -= c[e]
        load[ot] += c[e]
        asg[e] = ot
    return asg


def digest_reference(asg: list) -> int:
    """digest_warp of csrc/bal_assign.cuh on the host."""
    h = 0
    for e in range(NUM_EXPERTS):
        y = (((e << 8) | (asg[e] & 0xFF)) * 0x9E3779B1) & 0xFFFFFFFF
        h = (h + (y ^ (y >> 16))) & 0xFFFFFFFF
    h ^= h >> 16
    h = (h * 0x85EBCA6B) & 0xFFFFFFFF
    h ^= h >> 13
    return h


def virtual_reference(
    packed: torch.Tensor, pl: Placement, asg: list, rank: Optional[int] = None
) -> torch.Tensor:
    p = packed.cpu().to(torch.int64)
    e = (p >> 16) & 0xFFFF
    out = p.clone()
    for i in range(p.shape[0]):
        for j in range(p.shape[1]):
            ee = int(e[i, j])
            if ee >= NUM_EXPERTS:
                continue
            r = asg[ee]
            slot = pl.home_slot[ee] if r == pl.home[ee] else pl.sec_slot[ee]
            v = r * pl.pool + slot
            if rank is not None:
                v = slot if r == rank else PAD_ID
            out[i, j] = (v << 16) | (int(p[i, j]) & 0xFFFF)
    return out.to(torch.int32)


@functools.lru_cache(maxsize=None)
def _jit_module(use_pdl: bool):
    from sglang.kernels.jit.utils import load_jit, make_cpp_args

    args = make_cpp_args(use_pdl)
    return load_jit(
        "muse_vr_bal_assign",
        *args,
        cuda_files=[str(_CUH)],
        cuda_wrappers=[("run", f"muse_vr::bal::BalIds<{args}>::run")],
    )


def preload(use_pdl: bool = False) -> None:
    _jit_module(bool(use_pdl))


def bal_ids(
    packed: torch.Tensor,
    table: torch.Tensor,
    out: torch.Tensor,
    err: torch.Tensor,
    world: int,
    pool: int,
    pass_rows: int,
    rank: int = 0,
    local_view: bool = False,
    assign_out: Optional[torch.Tensor] = None,
    stamps: Optional[torch.Tensor] = None,
    grid: int = 1,
    use_pdl: bool = False,
    digest_out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """packed [M, 8] int32 (real expert ids) -> out [M, 8] int32 (virtual ids rank * pool + slot, or the local view);
    one launch of ``grid`` one-warp CTAs (every CTA computes the same assignment; CTA 0 writes ``out``)."""
    _jit_module(bool(use_pdl)).run(
        packed,
        table,
        out,
        err,
        assign_out,
        stamps,
        digest_out,
        int(world),
        int(pool),
        int(pass_rows),
        int(rank),
        int(bool(local_view)),
        int(grid),
    )
    return out


# ---------------------------------------------------------------------------------------------- pairwise form (csrc/bal_pw.cuh)
PW_MAX_PAIRS = 28
PW_MAX_MEMBERS = 32
PW_MAX_ROUNDS = 7
PW_MAX_SLOTS = 4
PW_OFF_PAIR = 256
PW_OFF_AB = 512
PW_OFF_MEM = PW_OFF_AB + PW_MAX_PAIRS
PW_OFF_SCH = PW_OFF_MEM + PW_MAX_PAIRS * PW_MAX_MEMBERS
PW_TABLE_WORDS = PW_OFF_SCH + PW_MAX_ROUNDS * PW_MAX_SLOTS
PW_WORDS = 33
NO_HOLDER = 0xF
NO_PAIR = 0xFF
_PW_CUH = Path(__file__).resolve().parent / "csrc" / "bal_pw.cuh"


def pw_schedule(world: int) -> list:
    """Perfect matchings of the complete rank graph (circle method): world - 1 rounds of world / 2 pairs (a < b)."""
    ranks = list(range(world))
    out = []
    for _ in range(world - 1):
        out.append(
            [
                (
                    min(ranks[i], ranks[world - 1 - i]),
                    max(ranks[i], ranks[world - 1 - i]),
                )
                for i in range(world // 2)
            ]
        )
        ranks = [ranks[0]] + [ranks[-1]] + ranks[1:-1]
    return out


def pw_pairs(pl: Placement) -> list:
    """Holder pairs (a < b) in (a, b) order, each with its replicated experts in id order."""
    W = pl.world
    pairs = [(a, b) for a in range(W) for b in range(a + 1, W)]
    mem = {ab: [] for ab in pairs}
    for e in range(NUM_EXPERTS):
        if pl.sec[e] >= 0:
            mem[(min(pl.home[e], pl.sec[e]), max(pl.home[e], pl.sec[e]))].append(e)
    return [(ab, mem[ab]) for ab in pairs]


def pw_table_words(pl: Placement) -> list:
    """The static pair table of bal_pw.cuh (no traffic data: the placement's id arithmetic and a fixed schedule)."""
    pairs = pw_pairs(pl)
    if len(pairs) > PW_MAX_PAIRS or any(len(m) > PW_MAX_MEMBERS for _, m in pairs):
        raise ValueError("bal pw: too many pairs or members for this placement")
    w = [0] * PW_TABLE_WORDS
    w[:NUM_EXPERTS] = pl.words()
    for e in range(NUM_EXPERTS):
        w[PW_OFF_PAIR + e] = NO_PAIR
    pid = {}
    for p, ((a, b), mem) in enumerate(pairs):
        pid[(a, b)] = p
        w[PW_OFF_AB + p] = a | (b << 4) | (len(mem) << 8)
        for j, e in enumerate(mem):
            w[PW_OFF_MEM + p * PW_MAX_MEMBERS + j] = e
            w[PW_OFF_PAIR + e] = p | (j << 8)
    for p in range(len(pairs), PW_MAX_PAIRS):
        w[PW_OFF_AB + p] = 0
    for r in range(PW_MAX_ROUNDS):
        for i in range(PW_MAX_SLOTS):
            w[PW_OFF_SCH + r * PW_MAX_SLOTS + i] = NO_PAIR
    for r, m in enumerate(pw_schedule(pl.world)):
        for i, ab in enumerate(m):
            w[PW_OFF_SCH + r * PW_MAX_SLOTS + i] = pid[ab]
    return w


def pw_max_members(pl: Placement) -> int:
    """The largest holder pair's static member count (the register form needs it <= 16)."""
    return max((len(m) for _, m in pw_pairs(pl)), default=1) or 1


def pw_table(pl: Placement, device) -> torch.Tensor:
    return (
        torch.tensor(pw_table_words(pl), dtype=torch.int64)
        .to(torch.int32)
        .to(device)
        .contiguous()
    )


def pw_reference(experts, pl: Placement, pass_rows: int, sweeps: int = 2) -> list:
    """CPU reference of the pairwise form: experts = the step's routed expert ids (PAD_ID / >= 256 skipped) -> per expert the
    4-bit holder code (NO_HOLDER if not routed)."""
    W = pl.world
    cnt = [0] * NUM_EXPERTS
    for e in experts:
        e = int(e)
        if e < NUM_EXPERTS:
            cnt[e] += 1
    c = costs(cnt, pass_rows, W)
    load = [0] * W
    code = [NO_HOLDER] * NUM_EXPERTS
    for e in range(NUM_EXPERTS):
        if cnt[e]:
            load[pl.home[e]] += c[e]
            code[e] = pl.home[e]
    pairs = pw_pairs(pl)
    order, dec = {}, {}
    for (a, b), mem in pairs:
        act = sorted((e for e in mem if cnt[e]), key=lambda e: (-c[e], e))
        order[(a, b)] = act
        dec[(a, b)] = [1 if pl.home[e] == b else 0 for e in act]
    for _ in range(sweeps):
        for m in pw_schedule(W):
            for ab in m:
                act = order[ab]
                if not act:
                    continue
                a, b = ab
                la, lb = load[a], load[b]
                for e, d in zip(act, dec[ab]):
                    if d:
                        lb -= c[e]
                    else:
                        la -= c[e]
                nd = []
                for e in act:
                    if la <= lb:
                        la += c[e]
                        nd.append(0)
                    else:
                        lb += c[e]
                        nd.append(1)
                load[a], load[b] = la, lb
                dec[ab] = nd
    for (a, b), act in order.items():
        for e, d in zip(act, dec[(a, b)]):
            code[e] = b if d else a
    return code


def pw_digest(code: list) -> int:
    h = 0
    for e in range(NUM_EXPERTS):
        y = (((e << 8) | (code[e] & 0xF)) * 0x9E3779B1) & 0xFFFFFFFF
        h = (h + (y ^ (y >> 16))) & 0xFFFFFFFF
    h ^= h >> 16
    h = (h * 0x85EBCA6B) & 0xFFFFFFFF
    h ^= h >> 13
    return h


def pw_words(code: list, tag32: int) -> list:
    """The 33 published words (unsigned 64-bit values) for a code list and a step tag."""
    out = []
    for l in range(32):
        payload = 0
        for j in range(8):
            payload |= (code[8 * l + j] & 0xF) << (4 * j)
        out.append((tag32 << 32) | payload)
    out.append((tag32 << 32) | pw_digest(code))
    return out


def pw_loads(code: list, experts, pl: Placement, pass_rows: int) -> list:
    """Per-rank cost load of an assignment (diagnostics)."""
    cnt = [0] * NUM_EXPERTS
    for e in experts:
        e = int(e)
        if e < NUM_EXPERTS:
            cnt[e] += 1
    c = costs(cnt, pass_rows, pl.world)
    load = [0] * pl.world
    for e in range(NUM_EXPERTS):
        if cnt[e]:
            load[code[e]] += c[e]
    return load


@functools.lru_cache(maxsize=None)
def _pw_module():
    import os

    from sglang.kernels.jit.utils import load_jit

    plant = (
        os.environ.get("MUSE_VR_BAL_PW_TEST_PLANT", "") == "1"
    )  # tests only: the planted-race control build
    sd = os.environ.get(
        "MUSE_VR_BAL_PW_STAMPS_DIR", ""
    ).strip()  # timing builds only: the (private) stamp header's dir
    flags = (["-DMUSE_VR_BAL_PW_TEST_PLANT_RACE"] if plant else []) + (
        ["-DMUSE_VR_BAL_PW_STAMPS"] if sd else []
    )
    extra = {"extra_cuda_cflags": flags} if flags else {}
    if sd:
        extra["extra_include_paths"] = [sd]
    return load_jit(
        "muse_vr_bal_pw" + ("_plant" if plant else "") + ("_stamps" if sd else ""),
        cuda_files=[str(_PW_CUH)],
        cuda_wrappers=[
            ("run", "muse_vr::bal::BalPw::run"),
            ("consumer", "muse_vr::bal::BalPwConsumer::run"),
        ],
        **extra,
    )


def pw_run(
    ids_words: torch.Tensor,
    seq: torch.Tensor,
    tbl: torch.Tensor,
    out: torch.Tensor,
    err: torch.Tensor,
    world: int,
    M: int,
    pass_rows: int,
    sweeps: int = 2,
    grid: int = 1,
    threads: int = 256,
    stamps: Optional[torch.Tensor] = None,
    spin_rounds: int = 64,
    max_rounds: int = 1 << 22,
    backoff_ns: int = 32,
    variant: int = 2,
    maxm: Optional[int] = None,
) -> None:
    """Development kernel: every CTA computes the assignment from ids_words [M, 8] int64 (tagged) and publishes into its row of
    out [grid, 33] int64. variant 2: the read form; 1: register rounds; 0: shared-memory rounds (the same function).
    maxm: the placement's largest pair (pw_max_members); required for variant 2."""
    _pw_module().run(
        ids_words,
        seq,
        tbl,
        out,
        err,
        stamps,
        int(world),
        int(M),
        int(pass_rows),
        int(sweeps),
        int(grid),
        int(threads),
        int(spin_rounds),
        int(max_rounds),
        int(backoff_ns),
        int(variant),
        int(maxm if maxm is not None else 16),
    )


def pw_consumer(
    ids_words: torch.Tensor,
    seq: torch.Tensor,
    tab: torch.Tensor,
    asg_words: torch.Tensor,
    bases: torch.Tensor,
    err: torch.Tensor,
    M: int,
    rank: int,
    pool: int,
    mode: int,
    grid: int = 1,
    stamps: Optional[torch.Tensor] = None,
    tok: Optional[torch.Tensor] = None,
    tok_rows: int = 0,
    tok_words: int = 0,
    spin_rounds: int = 64,
    max_rounds: int = 1 << 22,
    backoff_ns: int = 32,
) -> None:
    """Development consumer stand-in (the expert stage's prologue shape): bases [pool + 1] int32 from CTA 0; with tok
    [rows, pitch] int64 it also waits for the first and last word of tok_rows token rows (16-bit tags)."""
    _pw_module().consumer(
        ids_words,
        seq,
        tab,
        asg_words,
        bases,
        err,
        stamps,
        tok,
        int(tok_rows),
        int(tok_words),
        int(M),
        int(rank),
        int(pool),
        int(mode),
        int(grid),
        int(spin_rounds),
        int(max_rounds),
        int(backoff_ns),
    )


def pw_consumer_reference(
    experts, pl: Placement, code: Optional[list], rank: int, mode: int
) -> list:
    """bases [pool + 1] of the consumer stand-in (mode 0: home only; 1 / 2: experts whose code is this rank)."""
    cnt = [0] * pl.pool
    for e in experts:
        e = int(e)
        if e >= NUM_EXPERTS:
            continue
        if mode == 0:
            if pl.home[e] == rank:
                cnt[pl.home_slot[e]] += 1
        elif code[e] == rank:
            cnt[pl.home_slot[e] if pl.home[e] == rank else pl.sec_slot[e]] += 1
    out, acc = [], 0
    for s in range(pl.pool):
        out.append(acc)
        acc += cnt[s]
    out.append(acc)
    return out
