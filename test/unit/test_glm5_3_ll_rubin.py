# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""The TP4 kernel units on Rubin (VR200: SM107, 212 SMs), on CPU.

- ``ll.device`` names the GPUs the units serve: GB300 (measured) and Rubin (plans
  derived from GB300's); any other GPU declines every ``ll.*`` unit.
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

import sys
from types import ModuleType, SimpleNamespace

import pytest

from sgl_nvidia_mega.glm5_3 import registry, units
from sgl_nvidia_mega.glm5_3.ll import device
from sgl_nvidia_mega.glm5_3.ll import sparse_mla as fd
from sgl_nvidia_mega.glm5_3.profile import MAX_ROWS, UNITS, Profile, clip_widths

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


# ------------------------------------------------------------------ the device table


def test_device_table():
    assert device.lookup(GB300_SM) is device.GB300 and device.GB300.measured
    assert device.lookup(RUBIN_SM) is device.RUBIN and not device.RUBIN.measured
    assert (device.GB300.sms, device.RUBIN.sms) == (152, 212)
    assert not device.derived(GB300_SM) and device.derived(RUBIN_SM)
    for sm in (GB300_SM, RUBIN_SM):
        assert device.reason(sm) is None
    for sm in ((10, 0), (10, 1), (12, 0), (9, 0)):
        why = device.reason(sm)
        assert device.lookup(sm) is None and why.startswith(f"sm_{sm[0]}{sm[1]}")
        assert "sm_103" in why and "sm_107" in why


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
    "sm, sms, engages",
    [
        (RUBIN_SM, RUBIN_SMS, True),
        (GB300_SM, GB300_SMS, True),
        (RUBIN_SM, 208, False),
        (RUBIN_SM, GB300_SMS, False),
        ((10, 0), 148, False),
    ],
)
def test_gemm_plan_per_device(sm, sms, engages):
    from sgl_nvidia_mega.glm5_3.ll import gemm

    for role, shape in (
        ("o_proj", (6144, 4096)),
        ("wk_weights_proj", (160, 6144)),
        ("lm_head", (38720, 6144)),
    ):
        e, why = gemm._planned(role, _weight_module(shape), sm=sm, sms=sms)
        assert (e is not None) == engages, (role, why)
        if not engages:
            assert "SM103 with 152 SMs" in why and f"SM{sm[0]}{sm[1]}" in why


def test_router_splitk_stays_stock_on_rubin():
    from sgl_nvidia_mega.glm5_3.ll import gemm

    module = _weight_module((256, 6144))
    e, _ = gemm._planned("router", module, sm=GB300_SM, sms=GB300_SMS)
    assert e is not None
    e, why = gemm._planned("router", module, sm=RUBIN_SM, sms=RUBIN_SMS)
    assert e is None and "no measured plan" in why


def test_rubin_gemm_splitk_tactics_fit_one_wave():
    from sgl_nvidia_mega.glm5_3.ll import gemm_plan as plan

    entries = plan.device_plan(RUBIN_SM, RUBIN_SMS)
    assert entries is not None and plan.device_gpcs(RUBIN_SM) == 16
    assert plan.device_plan(GB300_SM, GB300_SMS) is plan.PLAN
    assert {e.role for e in plan.PLAN} - {e.role for e in entries} == {"router"}
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
        monkeypatch.setattr(gemm, "_device_capability", lambda: GB300_SM)
        assert gemm.splitk_source().name == "flashinfer"
        gemm.reset()
        monkeypatch.setattr(gemm, "_device_capability", lambda: RUBIN_SM)
        assert gemm.splitk_source().name == "vendored"
        # forcing FlashInfer's kernel stays the operator's choice
        gemm.reset()
        monkeypatch.setenv(gemm.SPLITK_ENV, "flashinfer")
        assert gemm.splitk_source().name == "flashinfer"
    finally:
        gemm.reset()
    assert gemm.flashinfer_serves_tactics(None)
    assert not gemm.flashinfer_serves_tactics(RUBIN_SM)


# ------------------------------------------------------------------ ll.sparse_mla_fd


def test_sparse_mla_covers_rubin_at_any_sm_count():
    for sms in (RUBIN_SMS, 208):
        p = SimpleNamespace(tp=4, heads=16, sm=RUBIN_SM, sms=sms)
        for rows in range(1, fd.ONE_TILE_MAX_ROWS + 1):
            assert fd.covers(p, rows, fd.VERIFY) is None
        assert fd.covers(p, 8, fd.VERIFY) is not None
    gb300_148 = SimpleNamespace(tp=4, heads=16, sm=GB300_SM, sms=148)
    assert "SMs" in fd.covers(gb300_148, 5, fd.VERIFY)
    other = SimpleNamespace(tp=4, heads=16, sm=(10, 0), sms=148)
    assert "sm_107" in fd.covers(other, 5, fd.VERIFY)


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
    family.profile = SimpleNamespace(tp=4, heads=16, sm=sm, sms=sms)
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


# ------------------------------------------------------------------ the split-K kernel


def test_splitk_smem_capacity_falls_back_without_a_gpu(monkeypatch):
    splitk = pytest.importorskip("sgl_nvidia_mega.glm5_3.ll.kernels.cute.splitk")
    import torch

    def no_props(index):
        raise RuntimeError("no GPU")

    splitk.smem_capacity_bytes.cache_clear()
    monkeypatch.setattr(torch.cuda, "get_device_properties", no_props)
    try:
        assert splitk.smem_capacity_bytes(0) == splitk._SMEM_CAPACITY_BYTES
        splitk.smem_capacity_bytes.cache_clear()
        monkeypatch.setattr(
            torch.cuda,
            "get_device_properties",
            lambda index: SimpleNamespace(shared_memory_per_block_optin=232448),
        )
        assert splitk.smem_capacity_bytes(0) == 232448
    finally:
        splitk.smem_capacity_bytes.cache_clear()
