# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""GLM-5.3 / GLM-5.2 MoE fold hop: the opt-in fold forms (SGLANG_NVIDIA_MEGAKERNEL_GLM_FOLD_FORM)
leave every served unit (object, constexprs, compile key) unchanged when unset, change
only their own unit when set, and the cluster-free form is only offered for the
deferred-norm fold."""

import pytest

from sgl_nvidia_mega.glm5_3.sm103.hop import api
from sgl_nvidia_mega.glm5_3.sm103.hop.moe_fold import MoeFoldArNorm, MoeFoldArNormDn


def _keys(monkeypatch, value):
    if value is None:
        monkeypatch.delenv(api.FOLD_FORM_ENV, raising=False)
    else:
        monkeypatch.setenv(api.FOLD_FORM_ENV, value)
    return {s.name: s.key for s in api.unit_specs()}


def test_unset_and_empty_keep_the_served_units(monkeypatch):
    for value in ("", " ", ","):
        specs = {s.name: s for s in api.unit_specs(form=api.fold_form(value))}
        for name in ("moe_fold_split", "moe_fold_plain"):
            assert type(specs[name].obj) is MoeFoldArNorm
        for name in ("moe_fold_split_dn", "moe_fold_plain_dn"):
            assert type(specs[name].obj) is MoeFoldArNormDn
        for name in ("moe_fold_split", "moe_fold_split_dn"):
            assert specs[name].constexprs == {"max_rows": 8, "split": True}
        for name in ("moe_fold_plain", "moe_fold_plain_dn"):
            assert specs[name].constexprs == {"max_rows": 8, "split": False}
    assert _keys(monkeypatch, None) == _keys(monkeypatch, "")


def test_late8_changes_only_the_plain_fold(monkeypatch):
    base, opt = _keys(monkeypatch, None), _keys(monkeypatch, "late8")
    assert base.keys() == opt.keys()
    assert {n for n in base if base[n] != opt[n]} == {"moe_fold_plain"}
    assert "('trigger_late', True)" in opt["moe_fold_plain"]


def test_split8dn_adds_one_unit_and_keeps_the_served_ones(monkeypatch):
    base, opt = _keys(monkeypatch, None), _keys(monkeypatch, "split8dn")
    assert set(opt) - set(base) == {"moe_fold_split_dn_nc"}
    assert all(base[n] == opt[n] for n in base)
    assert "('nocluster', True)" in opt["moe_fold_split_dn_nc"]


def test_both_forms(monkeypatch):
    base, both = _keys(monkeypatch, None), _keys(monkeypatch, " Late8 , split8dn ")
    assert {n for n in base if base[n] != both[n]} == {"moe_fold_plain"}
    assert set(both) - set(base) == {"moe_fold_split_dn_nc"}


def test_unknown_forms_are_refused():
    with pytest.raises(ValueError):
        api.fold_form("split8dn,tma")


def test_nocluster_is_only_offered_for_the_deferred_norm_fold():
    with pytest.raises(ValueError):
        MoeFoldArNorm(split=True, nocluster=True)
    assert MoeFoldArNormDn(split=True, nocluster=True).nocluster
    assert not MoeFoldArNormDn(split=True).nocluster
    assert "nocluster" not in vars(MoeFoldArNormDn(split=True))
    assert "trigger_late" not in vars(MoeFoldArNorm(split=False))


def test_dn_fold_unit_selection():
    k = api.HopKernels({}, None, None, None, None, None, 1e-5, 1.0, split_max_rows=6)
    assert k.fold_limits() == 6
    served = {r: k.dn_fold_unit(r) for r in range(1, 9)}
    assert served == {**{r: "moe_fold_split_dn" for r in range(1, 7)},
                      7: "moe_fold_plain_dn", 8: "moe_fold_plain_dn"}
    k.form, k.split8dn_rows = frozenset({"split8dn"}), 8
    assert k.fold_limits() == (6, 8, ("split8dn",))
    opt = {r: k.dn_fold_unit(r) for r in range(1, 9)}
    assert opt == {**{r: "moe_fold_split_dn" for r in range(1, api.SPLIT8DN_MIN_ROWS)},
                   **{r: "moe_fold_split_dn_nc" for r in range(api.SPLIT8DN_MIN_ROWS, 9)}}
