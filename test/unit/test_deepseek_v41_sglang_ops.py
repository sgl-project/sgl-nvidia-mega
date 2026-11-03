# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""The SGLang operators read by location (sglang_ops) resolve on the SGLang under test."""

import pytest

from sgl_nvidia_mega.deepseek_v41 import sglang_ops


@pytest.mark.parametrize("name", sorted(sglang_ops.CANDIDATES))
def test_every_operator_resolves_on_this_sglang(name):
    assert callable(sglang_ops.resolve(name))
