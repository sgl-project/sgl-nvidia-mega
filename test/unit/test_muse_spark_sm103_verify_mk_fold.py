# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""verify_mk on GB300: TensorRT-LLM's 212-CTA plans folded onto 152 CTAs (sm103/trtllm_ab93b78/megakernel/fold.py), on
the CPU. The served build's tables, planned at 212 virtual CTAs as the sm_107a build plans them, must pass the plan
check before and after the fold, keep every unit and its list order, and place in shared memory."""

from __future__ import annotations

import types
from collections import Counter

import pytest

from sgl_nvidia_mega.muse_spark.sm103.trtllm_ab93b78.megakernel import (
    fold,
    layer_mk,
    library,
    qkv,
    schedule,
    schedule_attn,
    schedule_experts,
    schedule_full,
    serving,
)

V = 212  # the QKV runs = TensorRT-LLM's CTAs
GB300 = types.SimpleNamespace(ctas=152, qkv_runs=V)
RUBIN = types.SimpleNamespace(ctas=V, qkv_runs=V)
FA_REGION = 166912  # fa::kRegionBytes of the served build (a 16-page ring, no FA_XQ)
MODE = serving.DEFAULT_MODE[: -len("_s4")] + "_sg_s4"  # verify_mk.runtime.MODE


def _rt():
    return dict(
        router_split=True,
        router_local=False,
        ft_gate=library.ft_gate(MODE),
        lm_ticket=bool(library.lm_level(MODE) & 1),
    )


def _late(**kw):
    return dict(
        late_ctas=schedule_attn.LATE_CTAS,
        late_first=True,
        late_sums_after=serving.FA_LATE_FIRST == 3,
        **kw,
    )


#: Every table set LayerKernels (and the stage objects it holds) builds for the served build, at 212 CTAs.
SERVED = {
    "swa": lambda: layer_mk.swa_layer_tables(
        V, router_split=True, oproj_off_merge=True, ah=library.lever_level(MODE, "ah")
    ),
    "full_two_launch": lambda: layer_mk.full_layer_tables(
        V, V, FA_REGION, router_split=True
    ),
    "full1": lambda: layer_mk.full_one_layer_tables(V, V, FA_REGION, **_rt()),
    "full2": lambda: layer_mk.full_one_layer_tables(
        V, V, FA_REGION, late_ctas=schedule_attn.LATE_CTAS, **_rt()
    ),
    "full3": lambda: layer_mk.full_one_layer_tables(
        V, V, FA_REGION, late_ctas=schedule_attn.LATE_CTAS, gates_after_fa=True, **_rt()
    ),
    "full4": lambda: layer_mk.full_one_layer_tables(
        V, V, FA_REGION, late_ctas=schedule_attn.LATE_CTAS, gates_early=True, **_rt()
    ),
    "full5": lambda: layer_mk.full_one_layer_tables(
        V, V, FA_REGION, **_late(), **_rt()
    ),
    "full5fh": lambda: layer_mk.full_one_layer_tables(
        V,
        V,
        FA_REGION,
        **_late(
            fh_slots=bool(serving.FH_PLAN & layer_mk.FH_PLAN_SLOTS),
            fh_merges=bool(serving.FH_PLAN & layer_mk.FH_PLAN_MERGES),
        ),
        **_rt(),
    ),
    "attention_half": lambda: schedule_attn.attention_tables(V),
    "full_attention_half": lambda: schedule_full.full_attention_tables(V, V, FA_REGION),
    "qkv": lambda: qkv.qkv_tables(V, False, 4),
}


@pytest.fixture(scope="module")
def served():
    return {name: make() for name, make in SERVED.items()}


@pytest.fixture(scope="module")
def folded(served):
    return {name: fold.fit(t, GB300) for name, t in served.items()}


@pytest.mark.parametrize("rl", [True, False])
@pytest.mark.parametrize("name", list(SERVED))
def test_served_212_plans_pass_the_check(served, name, rl):
    census = fold.check_plan(served[name].plan, rl=rl, runs=V)
    assert census.get("QkvRun", V) == V


def test_fit_is_the_identity_on_rubin(served):
    for t in served.values():
        assert fold.fit(t, RUBIN) is t
    assert fold.plan_ctas(RUBIN) == V and fold.plan_units(RUBIN, V) == V


def test_plan_ctas_and_units():
    assert fold.plan_ctas(GB300) == V
    assert (
        fold.plan_units(GB300, 152) == V
    )  # one FaAttn unit per CTA: planned per virtual CTA, then renumbered
    assert fold.plan_units(GB300, 64) == 64
    assert (
        fold.plan_ctas(types.SimpleNamespace(ctas=212)) == 212
    )  # a library without the attribute


def _census(plan, drop=("FaAttn", "MoeExperts")):
    return Counter((st, u) for units in plan for st, u in units if st not in drop)


@pytest.mark.parametrize("name", list(SERVED))
def test_fold_onto_152(served, folded, name):
    t, f = served[name], folded[name]
    assert f.ctas == 152 and len(f.plan) == 152
    for rl in (True, False):
        fold.check_plan(f.plan, rl=rl, runs=V)
    assert _census(f.plan) == _census(t.plan), (
        "every unit but FaAttn / MoeExperts kept once"
    )
    assert all(len(units) <= schedule.MAX_ROLES for units in f.plan)
    runs = sorted(u for units in f.plan for st, u in units if st == "QkvRun")
    had_runs = any(st == "QkvRun" for units in t.plan for st, _ in units)
    assert runs == (list(range(V)) if had_runs else [])
    fa = sorted(u for units in f.plan for st, u in units if st == "FaAttn")
    if any(st == "FaAttn" for units in t.plan for st, _ in units):
        assert fa == list(range(152)) and all(
            sum(st == "FaAttn" for st, _ in units) == 1 for units in f.plan
        )
    moe = [[u for st, u in units if st == "MoeExperts"] for units in f.plan]
    if any(moe):
        assert all(len(m) == 1 for m in moe)
        finals = sorted(m[0] for m in moe if m[0])
        assert finals == list(range(1, schedule_experts.FINAL_UNITS + 1))
    # placed with the planner's own unit sizes: the folded tables pass the same validation
    schedule.validate(f, f.unit_bytes, f.smem_main, f.dynamic)
    assert f.unit_bytes is t.unit_bytes and f.dynamic == t.dynamic


@pytest.mark.parametrize("name", list(SERVED))
def test_fold_keeps_each_list_order(served, name):
    plan = served[name].plan
    hosts = fold.choose_hosts(plan, 152)
    out = fold.fold_plan(plan, 152, hosts)
    folded_onto = {h: v for v, h in hosts.items()}
    for c in range(152):
        units = [(st, u) for st, u in out[c] if st != "FaAttn"]
        own = [(st, u) for st, u in plan[c] if st != "FaAttn"]
        part = fold._folded_part(plan[folded_onto[c]]) if c in folded_onto else []
        it = iter(units)
        assert all(x in it for x in own), c  # a subsequence, in order
        it = iter(units)
        assert all(x in it for x in part), c
        assert len(units) == len(own) + len(part), c


def test_swa_hosts_avoid_the_critical_path(served):
    plan = served["swa"].plan
    hosts = fold.choose_hosts(plan, 152)
    assert len(hosts) == V - 152 and 151 not in hosts.values()
    for v, h in hosts.items():
        names = {st for st, _ in plan[h]}
        assert not names & {"AhQkvFin", "AhSwaAttn", "QkvSum"}, (v, h, plan[h])
        if "MhH1" in {st for st, _ in plan[v]}:
            assert "MhRouter" not in names, (v, h)


def test_fold_renumbers_fa_slots_in_order(served):
    plan = served["full5fh"].plan
    out = fold.fold_plan(plan, 152)
    old = {c: u for c in range(152) for st, u in plan[c] if st == "FaAttn"}
    new = {c: u for c in range(152) for st, u in out[c] if st == "FaAttn"}
    assert sorted(old, key=old.get) == sorted(
        new, key=new.get
    )  # the same slot order on the kept CTAs


def test_tables_record_their_placement_inputs():
    t = qkv.qkv_tables(V, False, 4)
    assert (
        t.unit_bytes is not None
        and t.dynamic == {}
        and t.smem_main == schedule.SMEM_MAIN
    )
    e = schedule_experts.experts_tables(152)  # one unit per CTA: no fold needed
    assert fold.fit(e, GB300) is e


# ---- the plan check finds what it is for ----
def _plan(*ctas):
    return [list(units) for units in ctas]


def test_check_plan_finds_a_deadlock_in_order():
    # MhH1 waits for every AhOproj; an AhOproj after an MhH1 on one CTA never runs
    plan = _plan(
        [("AhOproj", schedule_attn.oproj_unit(0, 2)), ("MhH1", 0)],
        [("MhH1", 1), ("AhOproj", 2 | 2 << 10)],
    )
    with pytest.raises(AssertionError, match="deadlock"):
        fold.check_plan(plan, rl=False)


def test_check_plan_finds_a_router_order_that_fits_one_build_only():
    plan = _plan(
        [("AhOproj", schedule_attn.oproj_unit(0, 2)), ("MhH1", 0), ("MhRouter", 0)],
        [("MhH1", 1)],
        [("MhH1", 2)],
        [("MhH1", 3)],
    )
    fold.check_plan(plan, rl=False)  # the stock router after MhH1
    with pytest.raises(AssertionError, match="deadlock"):
        fold.check_plan(plan, rl=True)  # the rl units feed MhH1


def test_check_plan_finds_a_gang_on_one_cta():
    plan = _plan([("MhH1", 0), ("MhH1", 1)], [("MhH1", 2)], [("MhH1", 3)])
    with pytest.raises(AssertionError, match="gang"):
        fold.check_plan(plan)


def test_check_plan_finds_the_clear_hazards():
    with pytest.raises(AssertionError, match="IClearX"):
        fold.check_plan(
            _plan([("IClearX", 0), ("QkvRun", 0)], [("QkvRun", 1)]), runs=2, slices=1
        )
    fold.check_plan(_plan([("QkvRun", 0)], [("QkvRun", 1)]), runs=2, slices=1)


def test_check_plan_needs_the_once_per_cta_stages_everywhere():
    plan = _plan([("MoeExperts", 1)], [("QkvRun", 0)])
    with pytest.raises(AssertionError, match="MoeExperts"):
        fold.check_plan(plan, runs=1, slices=1)


def test_check_plan_needs_fa_attn_first_or_after_the_runs():
    with pytest.raises(AssertionError, match="FaAttn after"):
        fold.check_plan(_plan([("Clear", 0), ("FaAttn", 0)]))
    fold.check_plan(_plan([("FaAttn", 0), ("Clear", 0)]))


def _plan_levels(mode):
    return (
        library.lever_level(mode, "ah"),
        library.ft_gate(mode),
        bool(library.lm_level(mode) & 1),
    )


#: The lever sweep's builds whose host plans differ from the served build's (ah: the SWA front; lm: the full layers'
#: merge ticket and gate spread); the other sweep builds share the served tables.
SWEEP_PLANS = [
    m for m in library.SG_SWEEP_MODES if _plan_levels(m) != _plan_levels(MODE)
]


def test_sweep_builds_are_registered():
    assert (
        len(library.SG_SWEEP_MODES)
        == len(library.SG_SWEEP) + len(library.SG_SWEEP_PAIRS)
        == 19
    )
    for m in library.SG_SWEEP_MODES:
        assert m.endswith("_sg_s4") and m in library.MODES and m != MODE
        assert set(library.MODES[m]) ^ set(library.MODES[MODE]), m
    assert {"ah", "lm"} >= {
        lever
        for m in SWEEP_PLANS
        for lever, i in (("ah", 0), ("lm", 1))
        if _plan_levels(m)[i] != _plan_levels(MODE)[i]
    } and len(SWEEP_PLANS) >= 3


@pytest.mark.parametrize("mode", SWEEP_PLANS)
def test_sweep_plans_fold_onto_152(mode):
    ah, ft_gate, lm_ticket = _plan_levels(mode)
    rt = dict(
        router_split=True, router_local=False, ft_gate=ft_gate, lm_ticket=lm_ticket
    )
    tables = {
        "swa": layer_mk.swa_layer_tables(
            V, router_split=True, oproj_off_merge=True, ah=ah
        ),
        "full5fh": layer_mk.full_one_layer_tables(
            V,
            V,
            FA_REGION,
            **_late(
                fh_slots=bool(serving.FH_PLAN & layer_mk.FH_PLAN_SLOTS),
                fh_merges=bool(serving.FH_PLAN & layer_mk.FH_PLAN_MERGES),
            ),
            **rt,
        ),
    }
    for name, t in tables.items():
        f = fold.fit(t, GB300)
        assert f.ctas == 152, (mode, name)
        for rl in (True, False):
            fold.check_plan(t.plan, rl=rl, runs=V)
            fold.check_plan(f.plan, rl=rl, runs=V)
        assert _census(f.plan) == _census(t.plan), (mode, name)
        assert all(len(units) <= schedule.MAX_ROLES for units in f.plan), (mode, name)
        schedule.validate(f, f.unit_bytes, f.smem_main, f.dynamic)


def test_xpf_builds():
    """The expert-prefetch prototype's builds: the served build plus MSO_MK_XPF only; 0 is the served build itself."""
    assert library.xpf_mode(0) == MODE and tuple(
        library.MODES[MODE]
    ) == library.xpf_defines(0)
    assert library.XPF_MODES == [library.xpf_mode(m) for m in library.XPF_BUDGETS]
    for mib, m in zip(library.XPF_BUDGETS, library.XPF_MODES):
        assert m.endswith("_sg_s4") and m not in library.SG_SWEEP_MODES
        assert set(library.MODES[m]) - set(library.MODES[MODE]) == {
            f"-DMSO_MK_XPF={mib}"
        }
        assert library.xpf_level(m) == mib and library.xpf_level(MODE) == 0
        abi = library.xpf_abi(mib, 152)
        assert (
            abi[0] == mib and abi[4] == 152 - library.XPF_LATE_CTAS == 88
        )  # the prefetching CTAs on a GB300
        assert (
            abi[2] == (mib << 20) // library.XPF_TILE_BYTES
            and abi[6] == library.PARAM["MoeXpf"]
        )


def test_combo_builds():
    """A combination build sets each lever's define as its sweep build does, plus the prefetch budget."""
    name, defs = library.sg_combo(("ue14", "xp104", "xpf64"))
    assert name == MODE.replace("_ue10_", "_ue14_").replace(
        "_xp40_", "_xp104_"
    ).replace("_lm2_sg_s4", "_lm2_xpf64_sg_s4")
    served = set(library.MODES[MODE])
    assert set(defs) - served == {
        "-DMSO_MK_UE=14",
        "-DMSO_MK_XP=104",
        "-DMSO_MK_XPF=64",
    }
    assert served - set(defs) == {"-DMSO_MK_UE=10", "-DMSO_MK_XP=40"}
    one = dict((n, d) for n, d in [library.sg_combo(("lm3",))])
    assert one == {
        library.SG_SWEEP_MODES[0]: tuple(library.MODES[library.SG_SWEEP_MODES[0]])
    }  # a one-lever combo = the sweep build
    with pytest.raises(AssertionError):
        library.sg_combo(("ue14", "ue9"))  # two levels of one lever
    assert all(m in library.MODES for m in library.SG_COMBO_MODES)


def test_stamped_twin_and_stamp_words():
    """The stamped twin differs from the served build in its stamp defines only, and verify_mk's stamp dumps keep the
    kernel's per-unit block (kMaxRoles x 4) and the eight words from ``kStampEntry`` (msoMegakernel.cu)."""
    from pathlib import Path

    served, stamped = (
        set(library.MODES[MODE]),
        set(library.MODES[library.SG_STAMPED_MODE]),
    )
    assert served - stamped == {"-DMSO_MK_STAMPS_MODE=1"}
    assert stamped - served == {
        "-DMSO_MK_STAMPS_MODE=0",
        "-DMSO_MK_SY_LITE=1",
        "-DMSO_MK_SY_PROG=0",
    }
    src = (Path(library.__file__).parent / "msoMegakernel.cu").read_text()
    for line in (
        "constexpr int kMaxRoles = 16;",
        "constexpr int kStampEntry = kMaxRoles * 4;",
        "constexpr int kStampPro = kStampEntry + 7;",
        "constexpr int kStampSub = kStampEntry + 8;",
    ):
        assert line in src, line
    assert schedule.MAX_ROLES == 16  # verify_mk.runtime.STAMP_ROLES


def _oproj_units(units):
    return [u for st, u in units if st == "AhOproj"]


def test_spread_oproj_on_the_swa_launch(served, folded, monkeypatch):
    """MUSE_VR_VERIFY_MK_FOLD_SPREAD=1: no CTA of the sliding-window launch runs two o_proj units; the same units, each
    moved one after an attention unit and below the attention region, its weights issued before that unit ends."""
    monkeypatch.setenv(fold.SPREAD_ENV, "1")
    t, plain = served["swa"], folded["swa"]
    spread = fold.fit(t, GB300)
    assert (
        max(len(_oproj_units(l)) for l in plain.plan) == 2
    )  # what the plain fold does
    assert max(len(_oproj_units(l)) for l in spread.plan) == 1
    assert _census(spread.plan) == _census(plain.plan)
    assert sorted(u for l in spread.plan for u in _oproj_units(l)) == sorted(
        u for l in plain.plan for u in _oproj_units(l)
    )
    gap = t.dynamic["AhSwaAttn"][0]
    on_attn = 0
    for c, units in enumerate(spread.plan):
        names = [st for st, _ in units]
        if "AhSwaAttn" in names and "AhOproj" in names:
            i = names.index("AhOproj")
            assert (
                names[i - 1] == "AhSwaAttn"
                and t.unit_bytes("AhOproj", units[i][1]) <= gap
            )
            j = int(spread.issue[c, i])
            assert j < 0 or names[j] != "AhSwaAttn", (c, units, j)
            on_attn += 1
    assert on_attn >= 60


def test_spread_oproj_leaves_other_launches(served, folded, monkeypatch):
    monkeypatch.setenv(fold.SPREAD_ENV, "1")
    for name in ("full1", "full5", "full5fh", "full_attention_half", "qkv"):
        assert fold.fit(served[name], GB300).plan == folded[name].plan, name
    assert fold.fit(served["swa"], RUBIN) is served["swa"]


def test_spread_oproj_is_off_by_default(served, folded, monkeypatch):
    monkeypatch.delenv(fold.SPREAD_ENV, raising=False)
    assert fold.fit(served["swa"], GB300).plan == folded["swa"].plan
