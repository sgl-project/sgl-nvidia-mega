# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""The TEP8 routed body's default FP4 MoE tactic table on CPU: a hit, the autotuner
on any mismatch, and its switch."""

import pytest

from sgl_nvidia_mega.qwen3_5.moe import tactics

KEY = (tactics.FLASHINFER_VERSION, tactics.ARCH, tactics.SIGNATURE)


@pytest.fixture(autouse=True)
def switch(monkeypatch):
    monkeypatch.delenv(tactics.ENV, raising=False)
    monkeypatch.setenv(tactics.PARENT_ENV, "1")


def test_a_bucket_in_the_table_takes_its_tactic():
    assert tactics.lookup(100, 128, *KEY) == [16, 309]
    assert tactics.lookup(1000, 1024, *KEY) == [64, 6]
    # the launcher's default stays an int, as the autotuner returns it
    assert tactics.lookup(8, 8, *KEY) == -1


@pytest.mark.parametrize(
    "version, arch, signature, bucket",
    [
        ("0.7.2", tactics.ARCH, tactics.SIGNATURE, 128),
        (tactics.FLASHINFER_VERSION, "sm100", tactics.SIGNATURE, 128),
        (
            tactics.FLASHINFER_VERSION,
            tactics.ARCH,
            ("Module", 8) + tactics.SIGNATURE[2:],
            128,
        ),
        (*KEY, 2048),
    ],
    ids=["version", "arch", "signature", "bucket"],
)
def test_anything_else_asks_the_autotuner(version, arch, signature, bucket):
    assert tactics.lookup(bucket, bucket, version, arch, signature) is None


def test_the_switch_follows_the_runtime_optimizations_unless_set(monkeypatch):
    assert tactics.enabled()
    monkeypatch.setenv(tactics.ENV, "0")
    assert not tactics.enabled()
    assert tactics.lookup(100, 128, *KEY) is None
    monkeypatch.setenv(tactics.PARENT_ENV, "0")
    monkeypatch.setenv(tactics.ENV, "1")
    assert tactics.lookup(100, 128, *KEY) == [16, 309]
    monkeypatch.delenv(tactics.ENV)
    assert not tactics.enabled()
    monkeypatch.delenv(tactics.PARENT_ENV)
    assert not tactics.enabled()
