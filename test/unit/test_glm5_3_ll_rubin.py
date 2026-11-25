# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""The TP4 kernel units on Rubin (VR200: SM107, 212 SMs), on CPU.

- ``targets.select`` picks the units' plans from the architecture, once per profile:
  GB300's (``ll.target.GB300``, measured) on SM103, Rubin's (``sm107.tp4.RUBIN``,
  derived from GB300's) on SM107, none elsewhere (every ``ll.*`` unit declines). The
  units read the plans from ``Profile.target`` and never select again, and no file of
  ``glm5_3/ll`` names SM107 or imports ``glm5_3/sm107``.
- At every TP4 profile and width, Rubin composes the verify step and the NextN draft
  exactly as GB300 does, and another GPU takes no ``ll.*`` unit, so a TP4 server there
  still stops.
- ``ll.gemm``: Rubin's plan engages at 212 SMs only, has no router split-K, every
  split-K tactic fits one wave of 212 SMs with 16 GPCs assumed, and the split-K source
  on SM107 is the vendored copy (FlashInfer's explicit tactics serve SM100 / SM103).
- ``ll.sparse_mla_fd``: on Rubin the device's co-resident clusters cut the GB300 caps;
  a width it cannot launch in one wave declines (``Decline``) instead of failing the
  build, while GB300 keeps failing (fail-stop) on the same facts.
"""

import pathlib
import re
import sys
from types import ModuleType, SimpleNamespace

import pytest

from sgl_nvidia_mega.glm5_3 import registry, targets, units
from sgl_nvidia_mega.glm5_3.ll import sparse_mla as fd
from sgl_nvidia_mega.glm5_3.ll import target as ll_target
from sgl_nvidia_mega.glm5_3.ll.target import GB300
from sgl_nvidia_mega.glm5_3.profile import MAX_ROWS, UNITS, Profile, clip_widths
from sgl_nvidia_mega.glm5_3.sm107.tp4 import RUBIN

GB300_SM, GB300_SMS = (10, 3), 152
RUBIN_SM, RUBIN_SMS = (10, 7), 212


@pytest.fixture(autouse=True)
def _no_static_reasons(monkeypatch):
    monkeypatch.setattr(fd, "static_reasons", lambda: [])


def _profile(tp, tokens, sm=GB300_SM, sms=GB300_SMS):
    return Profile(
        tp=tp,
        heads=64 // tp,
        tokens_per_request=tokens,
        widths=clip_widths(tuple(range(tokens, MAX_ROWS + 1, tokens)), tp, tokens),
        sm=sm,
        sms=sms,
        target=targets.select(sm),
    )


def _owners(profile, rows, mode, groups, modules):
    cands, _ = registry.collect(modules)
    comp = registry.select(
        profile,
        rows,
        mode,
        candidates=cands,
        preference=units.PREFERENCE,
        groups=groups,
        allowed=units.allowed,
    )
    return {f"{g}.{s}": c.name for (g, s), c in comp.owners.items()}


def _steps(tp, tokens):
    """(mode, rows) of every target verify width and every draft graph."""
    from sgl_nvidia_mega.glm5_3 import draft_units

    widths = _profile(tp, tokens).widths
    out = [("target_verify", w) for w in widths]
    out += list(draft_units.draft_steps(tokens, widths))
    return out


def _groups_modules(mode):
    from sgl_nvidia_mega.glm5_3 import draft_units

    if mode == "target_verify":
        return registry.TARGET_GROUPS, registry.target_modules()
    return registry.DRAFT_GROUPS, draft_units.draft_modules()


# ------------------------------------------------------------------ the targets


def test_targets_select_by_architecture():
    assert targets.select(GB300_SM) is GB300 and GB300.measured
    assert targets.select(RUBIN_SM) is RUBIN and not RUBIN.measured
    assert (GB300.sm, GB300.sms, GB300.gpcs) == (GB300_SM, 152, 8)
    assert (RUBIN.sm, RUBIN.sms, RUBIN.gpcs) == (RUBIN_SM, 212, 16)
    assert GB300.flashinfer_splitk and not RUBIN.flashinfer_splitk
    assert not GB300.notes and set(RUBIN.notes) == {
        "ll.attention",
        "ll.gemm",
        "ll.hops",
        "ll.moe",
    }
    for sm in ((10, 0), (10, 1), (12, 0), (9, 0)):
        assert targets.select(sm) is None
        p = SimpleNamespace(sm=sm, target=None)
        assert ll_target.reason(p).startswith(f"sm_{sm[0]}{sm[1]}: no plans")
    assert ll_target.reason(SimpleNamespace(sm=RUBIN_SM, target=RUBIN)) is None


def _resolve(monkeypatch, sm, sms):
    from sgl_nvidia_mega.glm5_3 import profile as profile_mod

    monkeypatch.setattr(profile_mod, "_device", lambda capability: (sm, sms))
    args = SimpleNamespace(
        tp_size=4, speculative_algorithm="EAGLE", speculative_num_draft_tokens=6
    )
    return Profile.from_args(args, SimpleNamespace(num_attention_heads=64))


@pytest.mark.parametrize(
    "sm, sms, want",
    [(GB300_SM, 152, GB300), (RUBIN_SM, 212, RUBIN), ((10, 0), 148, None)],
)
def test_the_profile_selects_once_and_the_units_never_again(monkeypatch, sm, sms, want):
    """The profile picks the target once from the architecture; composing every TP4
    width and draft graph reads it from the profile and never selects again."""
    calls = []
    select = targets.select
    monkeypatch.setattr(
        targets, "select", lambda arch: calls.append(arch) or select(arch)
    )
    profile = _resolve(monkeypatch, sm, sms)
    assert profile.target is want and calls == [sm]
    hash(profile)  # a profile stays hashable with its target
    steps = _steps(4, 6)

    def no_select(arch):
        raise AssertionError(f"targets.select({arch}) after the profile resolved")

    monkeypatch.setattr(targets, "select", no_select)
    for mode, rows in steps:
        groups, modules = _groups_modules(mode)
        owners = _owners(profile, rows, mode, groups, modules)
        # another GPU's draft graphs keep their draft units: check_draft_config refuses
        # that GPU at the draft's build
        if want is not None or mode == "target_verify":
            assert bool(owners) == (want is not None), (mode, rows, owners)


_LL = pathlib.Path(fd.__file__).resolve().parent
_SM107_NAMES = re.compile(
    r"sm_?107|\(10,\s*7\)|\brubin\b|vr200|glm5_3\.sm107|glm5_3 import sm107"
    r"|glm5_3\.targets|glm5_3 import targets",
    re.IGNORECASE,
)


@pytest.mark.parametrize(
    "path",
    sorted(_LL.rglob("*.py")),
    ids=lambda p: str(p.relative_to(_LL)),
)
def test_ll_has_no_sm107_branch(path):
    """glm5_3/ll is GB300's code: no file names SM107, imports glm5_3/sm107 or picks a
    target (glm5_3/targets.py does, once per profile)."""
    hits = [
        f"{n}: {line.strip()}"
        for n, line in enumerate(path.read_text().splitlines(), 1)
        if _SM107_NAMES.search(line)
    ]
    assert not hits, hits


def test_rubins_plans_live_in_sm107():
    from sgl_nvidia_mega.glm5_3.sm107 import tp4

    assert pathlib.Path(tp4.__file__).resolve().parent.name == "sm107"
    assert not any(
        isinstance(v, ll_target.Target) and v is not GB300
        for v in vars(ll_target).values()
    )


# ------------------------------------------------------------------ compositions


@pytest.mark.parametrize("tp, tokens", sorted(UNITS))
def test_rubin_composes_as_gb300(tp, tokens):
    gb300 = _profile(tp, tokens)
    rubin = _profile(tp, tokens, RUBIN_SM, RUBIN_SMS)
    for mode, rows in _steps(tp, tokens):
        groups, modules = _groups_modules(mode)
        want = _owners(gb300, rows, mode, groups, modules)
        got = _owners(rubin, rows, mode, groups, modules)
        assert want, (tp, tokens, mode, rows)
        assert got == want, (tp, tokens, mode, rows)


@pytest.mark.parametrize("tokens", sorted(t for _, t in UNITS))
def test_rubin_tp4_serves_every_width(tokens):
    rubin = _profile(4, tokens, RUBIN_SM, RUBIN_SMS)
    groups, modules = _groups_modules("target_verify")
    for rows in rubin.widths:
        owners = _owners(rubin, rows, "target_verify", groups, modules)
        assert owners["attention.qkv_a"] == "ll.attention", rows
        assert owners["hops.post_attention"] == "ll.hops", rows
        assert owners["mlp.forward"] == "ll.moe.nvfp4", rows
        core = owners.get("attention.sparse_mla")
        assert core == ("ll.sparse_mla_fd" if rows <= fd.ONE_TILE_MAX_ROWS else None)


@pytest.mark.parametrize("sm", [(10, 0), (12, 0)])
def test_other_gpus_take_no_ll_unit_at_tp4(sm):
    other = _profile(4, 6, sm, 148)
    groups, modules = _groups_modules("target_verify")
    for rows in other.widths:
        owners = _owners(other, rows, "target_verify", groups, modules)
        assert not owners, (sm, rows, owners)


def test_the_tp4_draft_admits_sm107(monkeypatch):
    """check_draft_config accepts GB300 and Rubin, and still names another GPU."""
    import torch

    from sgl_nvidia_mega.glm5_3 import draft_units

    rt = ModuleType("sglang.srt.runtime_context")
    rt.get_spec = lambda: SimpleNamespace(
        speculative_eagle_topk=1, speculative_use_rejection_sampling=False
    )
    rt.get_parallel = lambda: SimpleNamespace(
        tp_size=4, pp_size=1, enable_dp_attention=False, enable_dp_lm_head=False
    )
    monkeypatch.setitem(sys.modules, "sglang.srt.runtime_context", rt)
    from sglang.srt.environ import envs

    monkeypatch.setattr(envs.SGLANG_ENABLE_EMBED_REPLICATION, "get", lambda: True)
    model = SimpleNamespace(
        config=SimpleNamespace(hidden_size=draft_units.HIDDEN),
        model=SimpleNamespace(rot_weight=None),
    )
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    for sm in (GB300_SM, RUBIN_SM):
        monkeypatch.setattr(torch.cuda, "get_device_capability", lambda sm=sm: sm)
        draft_units.check_draft_config(model)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda: (10, 0))
    with pytest.raises(ValueError, match="SM107"):
        draft_units.check_draft_config(model)


# ------------------------------------------------------------------ ll.gemm


def _weight_module(shape):
    import torch

    w = SimpleNamespace(
        shape=torch.Size(shape),
        dtype=torch.bfloat16,
        dim=lambda: 2,
        is_contiguous=lambda: True,
        data_ptr=lambda: 0,
    )
    return SimpleNamespace(weight=w, bias=None)


@pytest.mark.parametrize(
    "target, sms, engages",
    [
        (RUBIN, RUBIN_SMS, True),
        (GB300, GB300_SMS, True),
        (RUBIN, 208, False),
        (RUBIN, GB300_SMS, False),
        (GB300, 148, False),
        (None, 148, False),
    ],
)
def test_gemm_plan_per_target(target, sms, engages):
    from sgl_nvidia_mega.glm5_3.ll import gemm

    for role, shape in (
        ("o_proj", (6144, 4096)),
        ("wk_weights_proj", (160, 6144)),
        ("lm_head", (38720, 6144)),
    ):
        e, why = gemm._planned(role, _weight_module(shape), target=target, sms=sms)
        assert (e is not None) == engages, (role, why)
        if not engages and target is not None:
            assert f"with {target.sms} SMs" in why and f"has {sms} SMs" in why
        elif not engages:
            assert "no target" in why


def test_router_splitk_stays_stock_on_rubin():
    from sgl_nvidia_mega.glm5_3.ll import gemm

    module = _weight_module((256, 6144))
    e, _ = gemm._planned("router", module, target=GB300, sms=GB300_SMS)
    assert e is not None
    e, why = gemm._planned("router", module, target=RUBIN, sms=RUBIN_SMS)
    assert e is None and "no measured plan" in why


def test_rubin_gemm_splitk_tactics_fit_one_wave():
    from sgl_nvidia_mega.glm5_3.ll import gemm_plan as plan

    entries = RUBIN.gemm_plan
    assert GB300.gemm_plan is plan.PLAN and RUBIN.gpcs == 16
    # Rubin's plan is GB300's without the router, entry for entry
    assert entries == tuple(e for e in plan.PLAN if e.role != "router")
    checked = 0
    for e in entries:
        n, _ = e.shape
        for r in e.ranges:
            if r.kernel != plan.SPLITK:
                continue
            for m in range(r.m_lo, r.m_hi + 1):
                assert plan.splitk_fits_one_wave(r.tactic, m, n, RUBIN_SMS, 16), (e, m)
                checked += 1
    assert checked == 64  # wk | weights_proj at 1-64 rows


def test_splitk_source_on_sm107_is_the_vendored_copy(monkeypatch):
    pytest.importorskip("cutlass")
    from sgl_nvidia_mega.glm5_3.ll import gemm

    fi, fi_gemm = ModuleType("flashinfer"), ModuleType("flashinfer.gemm")

    def mm_bf16(
        a,
        b,
        bias=None,
        pdl=False,
        out=None,
        out_dtype=None,
        backend="cudnn",
        *,
        tactic=None,
    ):
        return out

    fi_gemm.mm_bf16 = mm_bf16
    fi_gemm.SplitKTactic = type("SplitKTactic", (), {gemm.FI1_SYNC_MARKER: True})
    fi.gemm = fi_gemm
    monkeypatch.setitem(sys.modules, "flashinfer", fi)
    monkeypatch.setitem(sys.modules, "flashinfer.gemm", fi_gemm)
    import sgl_nvidia_mega.glm5_3.sm103 as cute_gate

    monkeypatch.setattr(cute_gate, "cutlass_supported", lambda: (True, ""))
    monkeypatch.delenv(gemm.SPLITK_ENV, raising=False)
    try:
        gemm.reset()
        assert gemm.splitk_source(GB300).name == "flashinfer"
        assert gemm.splitk_source(None).name == "flashinfer"
        assert gemm.splitk_source(RUBIN).name == "vendored"
        # forcing FlashInfer's kernel stays the operator's choice
        gemm.reset()
        monkeypatch.setenv(gemm.SPLITK_ENV, "flashinfer")
        assert gemm.splitk_source(RUBIN).name == "flashinfer"
    finally:
        gemm.reset()


# ------------------------------------------------------------------ ll.sparse_mla_fd


def _ns_profile(sm, sms):
    return SimpleNamespace(tp=4, heads=16, sm=sm, sms=sms, target=targets.select(sm))


def test_sparse_mla_covers_rubin_at_any_sm_count():
    for sms in (RUBIN_SMS, 208):
        p = _ns_profile(RUBIN_SM, sms)
        for rows in range(1, fd.ONE_TILE_MAX_ROWS + 1):
            assert fd.covers(p, rows, fd.VERIFY) is None
        assert fd.covers(p, 8, fd.VERIFY) is not None
    gb300_148 = _ns_profile(GB300_SM, 148)
    assert "SMs" in fd.covers(gb300_148, 5, fd.VERIFY)
    other = _ns_profile((10, 0), 148)
    assert "no plans" in fd.covers(other, 5, fd.VERIFY)


def test_agree_caps_allow_zero_alone():
    zero = fd.Caps(134984, 134984, 0, 0)
    got = fd.agree_caps(
        zero, None, None, with_mt=False, with_fold=False, allow_zero=True
    )
    assert got == zero
    with pytest.raises(RuntimeError, match="occupancy"):
        fd.agree_caps(zero, None, None, with_mt=False, with_fold=False)
    with pytest.raises(RuntimeError, match="build failed"):
        fd.agree_caps(
            None,
            RuntimeError("x"),
            None,
            with_mt=False,
            with_fold=False,
            allow_zero=True,
        )


def _family(sm, sms, caps, monkeypatch):
    monkeypatch.setattr(fd, "device_caps", lambda heads, with_mt, with_fold: caps)
    monkeypatch.setattr(fd, "_tp_group", lambda: None)
    family = fd.SparseMlaFdFamily.__new__(fd.SparseMlaFdFamily)
    family.profile = _ns_profile(sm, sms)
    family.heads, family.layers, family.draft = 16, [], False
    family.plan, family.declined = None, {}
    return family


@pytest.mark.parametrize(
    "caps, served, declined",
    [
        # a 16-CTA cluster at one CTA per SM fits no GPC: every width declines
        (fd.Caps(134984, 134984, 0, 0), (), (1, 5, 6, 7)),
        # five co-resident clusters: 1-5 rows served, 6-7 decline
        (fd.Caps(134984, 134984, 5, 5), (1, 5), (6, 7)),
        # more than GB300's: the measured cap of 7 rows stays the bound
        (fd.Caps(134984, 134984, 12, 12), (1, 5, 6, 7), ()),
    ],
)
def test_sparse_mla_on_rubin_declines_what_the_device_cannot_launch(
    caps, served, declined, monkeypatch
):
    family = _family(RUBIN_SM, RUBIN_SMS, caps, monkeypatch)
    family.compile([1, 5, 6, 7])
    for rows in served:
        assert family.plan.splits(rows) == fd.ONE_TILE_SPLITS
        assert family.provider(None, rows, fd.VERIFY).rows == rows
    for rows in declined:
        with pytest.raises(registry.Decline, match="one wave"):
            family.provider(None, rows, fd.VERIFY)


def test_sparse_mla_on_gb300_still_fails_stop(monkeypatch):
    family = _family(GB300_SM, GB300_SMS, fd.Caps(134984, 134984, 5, 5), monkeypatch)
    with pytest.raises(RuntimeError, match="more than one wave"):
        family.compile([5, 6])
    family = _family(GB300_SM, GB300_SMS, fd.Caps(134984, 134984, 0, 0), monkeypatch)
    with pytest.raises(RuntimeError, match="occupancy"):
        family.compile([5])
