# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""g1 / g2 schedule shared-quarter weight (CPU): the default tables are the served ones, the
knob only moves units between clusters (every unit placed once, caps respected)."""

import pytest
import torch

from sgl_nvidia_mega.glm5_3.sm103.moe import api
from sgl_nvidia_mega.glm5_3.sm103.moe import schedule as S


def _served_g2(nc, shared):
    # the served rule, written out (weight 3)
    tab = torch.zeros(nc, S.G2_D, S.G2_ROW, dtype=torch.int16)
    for d in range(S.G2_D):
        units = 4 * d
        lists = [[] for _ in range(nc)]
        rc = nc - S.SH_CLUSTERS
        cap = min((units + 3 * S.SH_CLUSTERS + nc - 1) // nc, S.G2_MAX_UNITS)
        lim = min(units, rc * cap)
        for cl in range(S.SH_CLUSTERS):
            lists[cl].append(-1)
        for u in range(lim):
            lists[S.SH_CLUSTERS + u % rc].append(u)
        for u in range(lim, units):
            lists[(u - lim) % S.SH_CLUSTERS].append(u)
        for cl in range(nc):
            tab[cl, d, 0] = len(lists[cl])
            for k, u in enumerate(lists[cl]):
                tab[cl, d, 1 + k] = u
    return tab.view(nc, -1)


@pytest.mark.parametrize("nc", [32, 36, 52])
def test_default_is_served(nc):
    assert torch.equal(S.g2_table(nc, True), _served_g2(nc, True))
    assert torch.equal(S.g2_table(nc, True, shared_weight=3), S.g2_table(nc, True))
    assert torch.equal(S.g1_table(nc, True, shared_weight=3), S.g1_table(nc, True))


@pytest.mark.parametrize("nc", [32, 36, 52])
@pytest.mark.parametrize("w", [1, 2, 4])
def test_cover(nc, w):
    for build, dmax, row in ((S.g1_table, 32, S.G1_ROW), (S.g2_table, 64, S.G2_ROW)):
        flat = build(nc, True, shared_weight=w)
        tab = flat[:, : (dmax + 1) * row].view(nc, dmax + 1, row)
        for d in range(dmax + 1):
            seen = []
            for cl in range(nc):
                n = int(tab[cl, d, 0])
                seen += [int(x) for x in tab[cl, d, 1 : 1 + n]]
            assert sorted(u for u in seen if u >= 0) == list(range(4 * d))
            assert seen.count(-1) == S.SH_CLUSTERS


def test_knob(monkeypatch):
    for m in ("G1", "G2"):
        monkeypatch.delenv(api.G12_SHARED_WEIGHT_ENV + m[1:], raising=False)
    assert api.g12_schedule_knobs("g2") == {"shared_weight": S.G12_SHARED_WEIGHT}
    monkeypatch.setenv(api.G12_SHARED_WEIGHT_ENV + "2", "2")
    assert api.g12_schedule_knobs("g2") == {"shared_weight": 2}
    assert api.g12_schedule_knobs("g1") == {"shared_weight": 3}
    monkeypatch.setenv(api.G12_SHARED_WEIGHT_ENV + "1", "9")
    with pytest.raises(ValueError):
        api.g12_schedule_knobs("g1")
