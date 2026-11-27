# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""The 8-row MLA prologue's early QN arm (``MlaPrologue(8, overlap8=True, push=True,
early_arm8=True)``), on CPU.

- Only the overlapped pushed 8-row form takes it (it arms the pushed hand-off's barrier).
- ``compile_specs(prologue8_early_arm=True)`` builds it (both trigger forms, distinct
  compile key) only together with ``prologue8_overlap`` and ``prologue_push``; the
  16-row forms never take it, and without the flag every key is unchanged.
"""

import pytest

from sgl_nvidia_mega.glm5_3.sm103.attention import mla_prologue as mp


def _numerics():
    from sgl_nvidia_mega.glm5_3.attention import fused

    return fused._prologue_numerics()


def test_early_arm8_needs_overlap8_and_push():
    num = _numerics()
    for kw in ({}, {"overlap8": True}, {"push": True, "overlap8": False}):
        with pytest.raises(ValueError):
            mp.MlaPrologue(8, num, early_arm8=True, **kw)
    with pytest.raises(ValueError):
        mp.MlaPrologue(16, num, overlap16=True, push=True, early_arm8=True)
    k = mp.MlaPrologue(8, num, overlap8=True, push=True, early_arm8=True)
    assert k.early_arm8
    # same shared-memory layout as the form without the early arm
    assert k.off == mp.MlaPrologue(8, num, overlap8=True, push=True).off


def _specs(**kw):
    from sgl_nvidia_mega.glm5_3.attention import fused
    from sgl_nvidia_mega.glm5_3.sm103.attention import api

    return api.compile_specs(
        (8, 16),
        numerics=_numerics(),
        index_q_rope_form=fused.INDEX_Q_ROPE_FORM,
        warps=(16,),
        prologue_end=True,
        **kw,
    )


@pytest.mark.parametrize(
    "flags,armed",
    [
        ({}, False),
        ({"prologue8_early_arm": True}, False),
        ({"prologue8_early_arm": True, "prologue8_overlap": True}, False),
        ({"prologue8_overlap": True, "prologue_push": True}, False),
        (
            {"prologue8_early_arm": True, "prologue8_overlap": True, "prologue_push": True},
            True,
        ),
        (
            {
                "prologue8_early_arm": True,
                "prologue8_overlap": True,
                "prologue_push": True,
                "prologue_f32x2": True,
                "prologue16_overlap": True,
            },
            True,
        ),
    ],
)
def test_compile_specs_early_arm8(flags, armed):
    specs = _specs(**flags)
    plain = _specs(**{k: v for k, v in flags.items() if k != "prologue8_early_arm"})
    for name in ("prologue.8", "prologue_end.8"):
        k, key, _ = specs[name]
        assert k.early_arm8 is armed
        assert key.get("early_arm8", False) is armed
        if not armed:
            assert key == plain[name][1]
    for name in ("prologue.16", "prologue_end.16"):
        k, key, _ = specs[name]
        assert not getattr(k, "early_arm8", False) and "early_arm8" not in key
        assert key == plain[name][1]
