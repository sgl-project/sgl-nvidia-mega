# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""SGLANG_NVIDIA_MEGAKERNEL_GLM_HOP16_ROUTER_ROWS selects the row-contiguous router-partial
stores for the 9..16-row routing o_proj producer only: every other unit (instance A
included) keeps its class and Constexpr key, so its compiled code and cache entry are the
served ones."""

import inspect

import pytest

pytest.importorskip("cutlass")

from sgl_nvidia_mega.glm5_3.sm103.hop import api, o_proj, o_proj_rc  # noqa: E402

UNIT = "o_proj_produce_router16"


def test_parse(monkeypatch):
    assert api.hop16_router_rows("") is False
    assert api.hop16_router_rows("0") is False
    assert api.hop16_router_rows("1") is True
    with pytest.raises(ValueError):
        api.hop16_router_rows("yes")
    monkeypatch.delenv(api.HOP16_ROUTER_ROWS_ENV, raising=False)
    assert api.hop16_router_rows() is False
    monkeypatch.setenv(api.HOP16_ROUTER_ROWS_ENV, "1")
    assert api.hop16_router_rows() is True


def test_only_the_16_row_producer_changes(monkeypatch):
    monkeypatch.delenv(api.HOP16_ROUTER_ROWS_ENV, raising=False)
    base = {s.name: s for s in api.unit_specs(plain_allreduce=True)}
    off = api.unit_specs(plain_allreduce=True, router_rows16=False)
    assert [s.key for s in off] == [s.key for s in base.values()]
    on = {s.name: s for s in api.unit_specs(plain_allreduce=True, router_rows16=True)}
    assert list(on) == list(base)
    for name, spec in on.items():
        if name == UNIT:
            assert type(spec.obj) is o_proj_rc.OProjArProduceRc
            assert spec.constexprs.get("router_rows") is True
            assert spec.key != base[name].key
            assert len(spec.args) == len(base[name].args)
        else:
            assert type(spec.obj) is type(base[name].obj), name
            assert spec.key == base[name].key, name
    monkeypatch.setenv(api.HOP16_ROUTER_ROWS_ENV, "1")
    env_on = {s.name: s for s in api.unit_specs()}
    assert type(env_on[UNIT].obj) is o_proj_rc.OProjArProduceRc
    assert type(env_on["o_proj_produce_router"].obj) is o_proj.OProjArProduce


def test_router_cta_differs_only_in_the_store_mapping():
    """The copied 16-row TP8 router CTA uses the row-contiguous (expert, row) map
    while keeping epoch_begin's epoch_overlap in step with OProjArProduce."""

    def body(module):
        src = inspect.getsource(module)
        src = src[src.index("    def router_cta(") :]
        end = min(
            (
                i
                for i in (src.find(s, 1) for s in ("\n    @", "\n    def ", "\nclass "))
                if i > 0
            ),
            default=len(src),
        )
        lines = src[:end].split('"""')[2].splitlines()  # after the docstring
        return [ln.strip() for ln in lines if ln.strip() and not ln.strip().startswith("#")]

    served = body(o_proj)
    rc = body(o_proj_rc)
    assert "ex = t // rows" in served and "m = t % rows" in served
    assert "ex = t % 16" in rc and "m = t // 16" in rc
    assert "self.epoch_begin(mb, es, True, 0, self.epoch_overlap)" in served
    assert "self.epoch_begin(mb, es, True, 0, self.epoch_overlap)" in rc
