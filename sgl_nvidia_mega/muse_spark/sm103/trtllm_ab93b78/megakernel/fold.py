# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# verify_mk (sgl-nvidia-mega): a new file, not from TensorRT-LLM.
"""The 212-CTA plans on a build with fewer CTAs (GB300, sm_103a: 152 SMs). Pure Python (no GPU).

TensorRT-LLM's planners lay every launch out for 212 CTAs: one per SM of its Rubin GPU and one QKV run per CTA. A build
with fewer SMs keeps the 212 QKV runs (``MSO_MK_QKV_RUNS``, ``Library.qkv_runs``; one CTA cannot hold a 152nd of the QKV
weight), so the planners still run at 212 "virtual CTAs" (``plan_ctas``) and ``fit`` folds their tables onto the
build's CTAs before the shared-memory placement (``schedule.build_tables``, with the planner's own unit sizes):

* virtual CTA ``v < ctas`` stays on CTA ``v``; each of the other ``212 - ctas`` lists is merged into the list of one
  host CTA (``choose_hosts``), at most one per host;
* every unit keeps its stage and id, except two stages that run once per CTA: a folded list's ``MoeExperts`` unit is
  dropped (the stage cuts its work by ``blockIdx.x`` over ``kCtas``; a folded list's unit is always the plain unit 0),
  and the folded lists' ``FaAttn`` units are dropped and the others renumbered in their slot order to
  ``0 .. ctas - 1`` (the layer's ``FaUnits`` is ``kCtas``; the device splits the context over the units it is given);
* the merged order: the two lists' units by stage rank (``RANK``; within a list a unit takes at least its
  predecessor's rank, so each list keeps its order), the host's first on a tie.

``check_plan`` runs a plan under a conservative model of the units' waits (``_deps``: every unit waits for at least
what the device makes it wait for; ``_gangs``: units that wait for each other, so they must sit on distinct CTAs) and
checks the clear hazards and the role-order rules the device relies on. ``fit`` checks every folded plan with it,
with the router units as the rl build runs them (before ``MhH1``) and as the stock router does (after it); the tests
check TensorRT-LLM's 212 plans with it as well.

Dev switch ``MUSE_VR_VERIFY_MK_FOLD_SPREAD=1`` (``spread_oproj``, read when ``fit`` folds a plan): on GB300 the fold puts
two ``AhOproj`` units on every host of a sliding-window launch (the hosts cannot be attention CTAs, and every other
list carries one), and they run back to back in one shared-memory region, so the launch's last o_proj ends ~2 us after
the others (stamps of the served build) and ``MhH1`` waits for it. The switch
moves one unit per such CTA to an attention CTA, after its ``AhSwaAttn`` unit and below the attention region, so no
CTA runs two. Every unit keeps its id: the same tiles are computed by the same code.
"""

from __future__ import annotations

import logging
import os
from typing import Dict, List, Optional, Sequence, Tuple

from . import schedule

logger = logging.getLogger(__name__)
SPREAD_ENV = "MUSE_VR_VERIFY_MK_FOLD_SPREAD"

#: Stage rank of the merged order: a unit never waits for a unit of a higher rank (``_deps``). ``MhRouter`` is ranked
#: before ``MhH1`` (the rl build: ``MhH1`` sums the rl units' partials) and is never folded onto a CTA with an ``MhH1``
#: unit (the stock router waits for ``MhH1``).
RANK: Dict[str, int] = {
    "XSsq": 0,
    "QkvRun": 0,
    "AhQkvFin": 1,
    "QkvSum": 1,
    "AhSwaAttn": 2,
    "FaAttn": 2,
    "Clear": 2,
    "FaMerge": 3,
    "AhOproj": 4,
    "MhRouter": 5,
    "MhH1": 6,
    "MhPool": 7,
    "IClearX": 7,
    "MhClear": 7,
    "MoeExperts": 8,
    "MoeExpertsBf16": 8,
    "MhFinal": 9,
    "I2bGhost": 9,
    "I2bWarm": 9,
    "MhUnpool": 10,
    "MhF1": 11,
}
#: Stages that run exactly once on every CTA of a launch that has them.
ONE_PER_CTA = ("MoeExperts", "MoeExpertsBf16")
#: The units of one MhH1 / MhF1 row (``schedule_moe_hops.DEFAULT_ROW_UNITS``) and of one SwaAttn KV head.
ROW_UNITS = 4
ATTN_BLOCKS = 33
#: Host choice: a rough cost of each unit before the experts (only the order of the hosts matters).
_COST = {
    "QkvRun": 3.0,
    "XSsq": 0.5,
    "AhQkvFin": 1.5,
    "QkvSum": 1.0,
    "AhSwaAttn": 2.5,
    "FaAttn": 3.0,
    "FaMerge": 1.0,
    "Clear": 0.3,
    "AhOproj": 0.7,  # per tile
    "MhRouter": 0.5,
    "MhH1": 1.5,
    "MhPool": 1.0,
    "IClearX": 0.2,
    "MhClear": 0.2,
    "MhFinal": 0.7,
    "MhUnpool": 1.0,
    "MhF1": 1.5,
}


def plan_ctas(lib) -> int:
    """The CTAs the planners lay a launch out for: the build's QKV runs (212; = its CTAs on sm_107a)."""
    return int(getattr(lib, "qkv_runs", lib.ctas))


def plan_units(lib, units: int) -> int:
    """The FaAttn units to plan with: one per virtual CTA when the plan is folded and the layer has one unit per CTA."""
    v = plan_ctas(lib)
    return v if v != lib.ctas and int(units) == lib.ctas else int(units)


def fit(tables: schedule.Tables, lib) -> schedule.Tables:
    """``tables`` (planned at ``plan_ctas(lib)`` CTAs) on ``lib``'s CTAs: unchanged when they agree, else the folded
    plan placed again with the planner's unit sizes and dynamic regions, after ``check_plan``."""
    if tables.ctas == lib.ctas:
        return tables
    assert tables.ctas > lib.ctas and tables.unit_bytes is not None, (tables.ctas, lib.ctas)
    plan = fold_plan(tables.plan, lib.ctas)
    if os.environ.get(SPREAD_ENV, "").strip() == "1":
        plan = spread_oproj(plan, tables.unit_bytes, tables.dynamic)
    for rl in (True, False):
        check_plan(plan, rl=rl, runs=tables.ctas)
    return schedule.build_tables(plan, tables.unit_bytes, tables.smem_main, tables.dynamic)


# ---------------------------------------------------------------------------------------------------------------------
# Folding
# ---------------------------------------------------------------------------------------------------------------------
def _ranks(units: Sequence[Tuple[str, int]]) -> List[int]:
    out, r = [], 0
    for st, _ in units:
        r = max(r, RANK[st])
        out.append(r)
    return out


def merge(host: Sequence[Tuple[str, int]], other: Sequence[Tuple[str, int]]) -> List[Tuple[str, int]]:
    """The two unit lists in one: by rank, each list in its own order, the host's unit first on a tie."""
    ra, rb = _ranks(host), _ranks(other)
    out: List[Tuple[str, int]] = []
    i = j = 0
    while i < len(host) or j < len(other):
        if j >= len(other) or (i < len(host) and ra[i] <= rb[j]):
            out.append(tuple(host[i]))
            i += 1
        else:
            out.append(tuple(other[j]))
            j += 1
    return out


def _gang_keys(units: Sequence[Tuple[str, int]]) -> set:
    out = set()
    for st, u in units:
        if st == "AhSwaAttn":
            out.add(("AhSwaAttn", u // ATTN_BLOCKS))
        elif st in ("MhH1", "MhF1"):
            out.add((st, u // ROW_UNITS))
    return out


def _folded_part(units: Sequence[Tuple[str, int]]) -> List[Tuple[str, int]]:
    """A folded list without its once-per-CTA units (``FaAttn``, ``MoeExperts``)."""
    out = []
    for st, u in units:
        if st in ONE_PER_CTA:
            assert u == 0, f"a folded list carries an experts final unit ({st} {u})"
            continue
        if st == "FaAttn":
            continue
        out.append((st, u))
    return out


def _cost(units: Sequence[Tuple[str, int]]) -> float:
    from . import schedule_attn

    total = 0.0
    for st, u in units:
        if st == "AhOproj":
            total += _COST["AhOproj"] * len(schedule_attn.oproj_tiles(u))
        else:
            total += _COST.get(st, 0.0)
    return total


def _fa_prefix_ok(units: Sequence[Tuple[str, int]]) -> bool:
    """The device's rule for a FaAttn unit: first on its CTA (prologue at entry), or after the CTA's QKV runs and then
    region-less QkvSum units only (``faAfterQkvRun`` runs only when role 0 is a QkvRun)."""
    names = [st for st, _ in units]
    if "FaAttn" not in names:
        return True
    fa = names.index("FaAttn")
    if fa == 0:
        return True
    k = 0
    while k < fa and names[k] == "QkvRun":
        k += 1
    return k >= 1 and all(n == "QkvSum" for n in names[k:fa])


def _feasible(host: Sequence[Tuple[str, int]], other: Sequence[Tuple[str, int]]) -> bool:
    hn, on = {st for st, _ in host}, {st for st, _ in other}
    if ("MhRouter" in hn and "MhH1" in on) or ("MhH1" in hn and "MhRouter" in on):
        return False  # the router's order against MhH1 depends on the build (rl or not)
    if _gang_keys(host) & _gang_keys(other):
        return False  # two units that wait for each other on one CTA
    merged = merge(host, other)
    return len(merged) <= schedule.MAX_ROLES and _fa_prefix_ok(merged)


#: Units on the attention critical path before o_proj: a folded QKV run (rank 0) would run ahead of them.
_EARLY_CRITICAL = ("XSsq", "AhQkvFin", "QkvSum", "AhSwaAttn")


def _host_score(host: Sequence[Tuple[str, int]], part: Sequence[Tuple[str, int]]) -> float:
    """Lower is better: the host's work, far more for a host with an early critical unit, a little more for a host
    whose unit after the experts would delay the folded list's MhFinal / MhF1."""
    hn, pn = {st for st, _ in host}, {st for st, _ in part}
    score = _cost(host)
    if hn & set(_EARLY_CRITICAL):
        score += 100.0
    if "MhUnpool" in hn and pn & {"MhFinal", "MhF1"}:
        score += 2.0
    return score


def choose_hosts(plan: schedule.Plan, ctas: int) -> Dict[int, int]:
    """Folded list ``v >= ctas`` -> its host CTA (< ctas, at most one folded list each). The heaviest folded lists first,
    each onto the feasible host with the lowest ``_host_score``; never the experts' publisher CTA (``ctas - 1``, it
    runs the routing for every CTA)."""
    folded = sorted(range(ctas, len(plan)), key=lambda v: (-_cost(_folded_part(plan[v])), v))
    free = [c for c in range(ctas - 1)]
    out: Dict[int, int] = {}
    for v in folded:
        part = _folded_part(plan[v])
        cands = [c for c in free if _feasible(plan[c], part)]
        assert cands, f"no host for virtual CTA {v}: {plan[v]}"
        best = min(cands, key=lambda c: (_host_score(plan[c], part), c))
        out[v] = best
        free.remove(best)
    return out


def fold_plan(plan: schedule.Plan, ctas: int, hosts: Optional[Dict[int, int]] = None) -> schedule.Plan:
    """``plan`` (one unit list per virtual CTA) on ``ctas`` CTAs (module docstring)."""
    v_ctas = len(plan)
    assert v_ctas >= ctas
    if v_ctas == ctas:
        return [list(map(tuple, units)) for units in plan]
    hosts = choose_hosts(plan, ctas) if hosts is None else dict(hosts)
    assert sorted(hosts) == list(range(ctas, v_ctas)), "one host per folded list"
    assert len(set(hosts.values())) == len(hosts) and all(0 <= h < ctas for h in hosts.values())
    # FaAttn: the kept units renumbered in their slot order
    kept = sorted(u for c in range(ctas) for st, u in plan[c] if st == "FaAttn")
    slot = {u: s for s, u in enumerate(kept)}
    out: schedule.Plan = []
    folded_onto = {h: v for v, h in hosts.items()}
    for c in range(ctas):
        own = [("FaAttn", slot[u]) if st == "FaAttn" else (st, u) for st, u in plan[c]]
        if c in folded_onto:
            own = merge(own, _folded_part(plan[folded_onto[c]]))
        out.append(own)
    return out


def _oproj(units: Sequence[Tuple[str, int]]) -> List[Tuple[int, int]]:
    return [(i, u) for i, (st, u) in enumerate(units) if st == "AhOproj"]


def _has(units: Sequence[Tuple[str, int]], stage: str) -> bool:
    return any(st == stage for st, _ in units)


def spread_oproj(plan: schedule.Plan, unit_bytes, dynamic) -> schedule.Plan:
    """A folded plan with no CTA running two ``AhOproj`` units (module docstring, ``MUSE_VR_VERIFY_MK_FOLD_SPREAD``).

    Only launches with an attention region (``dynamic["AhSwaAttn"]``) change; every other plan is returned as is. A unit
    fits on an attention CTA when its weight region fits below the attention region (``unit_bytes <= offset``); it goes
    right after the CTA's last ``AhSwaAttn`` unit (its rank is between the attention's and the hops'). Pass 1: each CTA
    with two units hands its later fitting unit to an attention CTA without one. Pass 2: for a CTA whose units are both
    too large, a non-attention CTA with one fitting unit takes the large unit in its place and its own unit goes to an
    attention CTA. The result is checked by ``fit`` like every folded plan."""
    region = (dynamic or {}).get("AhSwaAttn")
    if region is None or not any(len(_oproj(l)) >= 2 for l in plan):
        return plan
    gap = int(region[0])
    fits = lambda u: 0 < unit_bytes("AhOproj", u) <= gap  # noqa: E731
    out = [list(map(tuple, units)) for units in plan]
    targets = [c for c, l in enumerate(out) if _has(l, "AhSwaAttn") and not _oproj(l)]

    def put(c: int, u: int) -> None:
        k = max(j for j, (st, _) in enumerate(out[c]) if st == "AhSwaAttn") + 1
        out[c].insert(k, ("AhOproj", u))

    moved = swapped = 0
    for c in [c for c, l in enumerate(out) if len(_oproj(l)) >= 2]:  # pass 1
        small = [(i, u) for i, u in _oproj(out[c]) if fits(u)]
        if small and targets:
            i, u = small[-1]
            del out[c][i]
            put(targets.pop(0), u)
            moved += 1
    for c in [c for c, l in enumerate(out) if len(_oproj(l)) >= 2]:  # pass 2
        donors = [
            x
            for x, l in enumerate(out)
            if x != c and not _has(l, "AhSwaAttn") and len(_oproj(l)) == 1 and fits(_oproj(l)[0][1])
        ]
        if not donors or not targets:
            continue
        x = donors[0]
        (ix, ux), = _oproj(out[x])
        i, u = _oproj(out[c])[-1]
        del out[c][i]
        out[x][ix] = ("AhOproj", u)
        put(targets.pop(0), ux)
        swapped += 1
    doubled = sum(1 for l in out if len(_oproj(l)) >= 2)
    logger.warning(
        "[verify_mk fold] %s=1: %d o_proj units moved onto attention CTAs, %d swapped; CTAs with two left: %d",
        SPREAD_ENV,
        moved,
        swapped,
        doubled,
    )
    return out


# ---------------------------------------------------------------------------------------------------------------------
# The plan check: waits (conservative), gangs, hazards, role-order rules
# ---------------------------------------------------------------------------------------------------------------------
def _block_runs(block: int, runs: int, slices: int) -> frozenset:
    """The QKV runs a 64-row block's sum waits for (its 4 tiles in every K slice)."""
    out = set()
    for t in range(block * schedule.QKV_TILES_PER_BLOCK, (block + 1) * schedule.QKV_TILES_PER_BLOCK):
        out.update(r for r, _ in schedule.qkv_tile_runs(t, runs, slices))
    return frozenset(out)


def _deps(stage: str, unit: int, rl: bool, runs: int, slices: int) -> List[Tuple[str, object]]:
    """What a unit waits for (a superset of the device's waits): ``("all", S)`` every unit of stage S in the plan,
    ``("runs", ids)`` those QKV runs."""
    a = lambda *ss: [("all", s) for s in ss]  # noqa: E731
    if stage in ("QkvRun", "XSsq", "Clear", "MhClear", "IClearX", "I2bGhost", "I2bWarm"):
        return []
    if stage in ("AhQkvFin", "QkvSum"):
        return [("runs", _block_runs(unit, runs, slices))] + a("XSsq")
    if stage == "AhSwaAttn":
        return a("QkvRun", "AhQkvFin")
    if stage == "FaAttn":
        return a("QkvRun", "QkvSum")
    if stage == "FaMerge":
        return a("QkvRun", "QkvSum", "FaAttn")
    if stage == "AhOproj":
        return a("AhSwaAttn", "FaMerge")
    if stage == "MhRouter":
        return a("AhSwaAttn", "FaMerge", "AhOproj") if rl else a("MhH1")
    if stage == "MhH1":
        return a("AhOproj") + (a("MhRouter") if rl else [])
    if stage == "MhPool":
        return a("MhH1")
    if stage in ONE_PER_CTA:
        return a("MhRouter", "MhPool", "MhH1")
    if stage == "MhFinal":
        return a(*ONE_PER_CTA)
    if stage == "MhUnpool":
        return a("MhFinal")
    if stage == "MhF1":
        return a("MhUnpool", "MhH1")
    raise AssertionError(f"no wait model for stage {stage}")


def _gang(stage: str, unit: int) -> Optional[Tuple[str, int]]:
    if stage == "AhSwaAttn":
        return ("AhSwaAttn", unit // ATTN_BLOCKS)
    if stage in ("MhH1", "MhF1"):
        return (stage, unit // ROW_UNITS)
    if stage in ONE_PER_CTA:
        return (stage, 0)
    return None


def check_plan(plan: schedule.Plan, rl: bool = True, runs: Optional[int] = None, slices: int = 4) -> Dict[str, int]:
    """Raises AssertionError unless ``plan`` completes under the wait model (``_deps``, ``_gangs``: a gang member
    completes once every member has started) with every CTA running its list in order, the clear hazards hold
    (``IClearX`` after every QKV run, ``Clear`` before every ``MhF1``), and the role rules hold (at most
    ``MAX_ROLES`` units, the once-per-CTA stages on every CTA once, ``FaAttn`` slots ``0 .. n - 1`` once each, each
    ``FaAttn`` first or after the CTA's runs and sums only). Returns a census."""
    ctas = len(plan)
    keys: List[Tuple[int, int]] = []
    index: Dict[Tuple[int, int], int] = {}
    for c, units in enumerate(plan):
        assert len(units) <= schedule.MAX_ROLES, (c, units)
        assert _fa_prefix_ok(units), f"CTA {c}: FaAttn after other units {units}"
        for i in range(len(units)):
            index[(c, i)] = len(keys)
            keys.append((c, i))
    stage_of = [plan[c][i][0] for c, i in keys]
    unit_of = [int(plan[c][i][1]) for c, i in keys]
    by_stage: Dict[str, List[int]] = {}
    for k, st in enumerate(stage_of):
        by_stage.setdefault(st, []).append(k)
    for st in ONE_PER_CTA:
        if st in by_stage:
            per = [sum(1 for s, _ in units if s == st) for units in plan]
            assert per == [1] * ctas, f"{st}: {per} units per CTA"
    if "FaAttn" in by_stage:
        fa = sorted(unit_of[k] for k in by_stage["FaAttn"])
        assert fa == list(range(len(fa))), "FaAttn slots"
        assert all(sum(1 for s, _ in units if s == "FaAttn") <= 1 for units in plan), "one FaAttn per CTA"
    run_key: Dict[int, int] = {}
    for k in by_stage.get("QkvRun", []):
        assert unit_of[k] not in run_key, f"QKV run {unit_of[k]} twice"
        run_key[unit_of[k]] = k
    runs = len(run_key) if runs is None else runs
    # each unit's waits: whole stages, and QKV runs by key
    total = {st: len(ks) for st, ks in by_stage.items()}
    need_all: List[Tuple[str, ...]] = []
    need_runs: List[Tuple[int, ...]] = []
    for k, st in enumerate(stage_of):
        whole, rk = [], []
        for kind, what in _deps(st, unit_of[k], rl, runs, slices):
            if kind == "all":
                if what in total and what != st:
                    whole.append(what)
            else:
                for r in what:
                    assert r in run_key, f"{st} {unit_of[k]} waits for QKV run {r}, which no CTA runs"
                    rk.append(run_key[r])
        need_all.append(tuple(whole))
        need_runs.append(tuple(rk))
    gangs: Dict[Tuple[str, int], List[int]] = {}
    for k, st in enumerate(stage_of):
        g = _gang(st, unit_of[k])
        if g is not None:
            gangs.setdefault(g, []).append(k)
    for g, ks in gangs.items():
        cs = [keys[k][0] for k in ks]
        assert len(set(cs)) == len(cs), f"gang {g} has two units on one CTA"
    gang_of = {k: g for g, ks in gangs.items() for k in ks}
    # run: S[k] = the units complete whenever unit k starts, K[k] = whenever it is complete (key bitmasks)
    pos = [0] * ctas
    is_started = [False] * len(keys)
    is_done = [False] * len(keys)
    done_n = {st: 0 for st in total}
    stage_k = {st: 0 for st in total}  # OR of K over the stage's completed units
    S: Dict[int, int] = {}
    K: Dict[int, int] = {}
    prev_k = [0] * ctas  # K of the CTA's last completed unit
    progress = True
    while progress:
        progress = False
        for c in range(ctas):
            while pos[c] < len(plan[c]):
                k = index[(c, pos[c])]
                if not is_started[k]:
                    if any(done_n[w] < total[w] for w in need_all[k]) or any(not is_done[r] for r in need_runs[k]):
                        break
                    s = prev_k[c]
                    for w in need_all[k]:
                        s |= stage_k[w]
                    for r in need_runs[k]:
                        s |= K[r]
                    S[k] = s
                    is_started[k] = True
                    progress = True
                g = gang_of.get(k)
                kk = S[k] | (1 << k)
                if g is not None:
                    if not all(is_started[j] for j in gangs[g]):
                        break
                    for j in gangs[g]:
                        kk |= S[j]  # every member has started: what was complete then is complete now
                K[k] = kk
                is_done[k] = True
                done_n[stage_of[k]] += 1
                stage_k[stage_of[k]] |= kk
                prev_k[c] = kk
                pos[c] += 1
                progress = True
    stuck = []
    for c in range(ctas):
        if pos[c] < len(plan[c]):
            k = index[(c, pos[c])]
            waits = [w for w in need_all[k] if done_n[w] < total[w]]
            waits += [f"QkvRun {unit_of[r]}" for r in need_runs[k] if not is_done[r]]
            if is_started[k]:
                waits.append(f"gang {gang_of.get(k)}")
            stuck.append((c, plan[c][pos[c]], waits[:4]))
    assert not stuck, f"deadlock (rl={rl}): {len(stuck)} CTAs blocked, e.g. {stuck[:4]}"
    all_mask = {st: sum(1 << k for k in ks) for st, ks in by_stage.items()}
    # hazards
    runs_mask = all_mask.get("QkvRun", 0)
    for k in by_stage.get("IClearX", []):
        assert S[k] & runs_mask == runs_mask, f"IClearX {unit_of[k]} on CTA {keys[k][0]} may start before a QKV run ends"
    for k in by_stage.get("Clear", []):
        for f in by_stage.get("MhF1", []):
            assert (S[f] >> k) & 1, f"Clear {unit_of[k]} on CTA {keys[k][0]} may end after MhF1 {unit_of[f]} starts"
    return {st: len(ks) for st, ks in sorted(by_stage.items())}
