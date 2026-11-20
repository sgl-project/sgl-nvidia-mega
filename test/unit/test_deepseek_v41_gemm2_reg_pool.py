# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Slot gemm2_reg_pool on CPU: needs gemm2_pipe, and asks the MoE build (target and draft)
for the pipelined GEMM2 at the 96-register cap. The kernels are checked bit for bit in
test/kernels/test_deepseek_v41_moe_gemm2_pipe.py (build "pool")."""

from sgl_nvidia_mega.deepseek_v41 import moe, runtime
from sgl_nvidia_mega.deepseek_v41.sm103.moe import routed_expert_gemm as reg


def test_the_slot_needs_gemm2_pipe(monkeypatch):
    monkeypatch.setattr(runtime, "SLOTS", frozenset({"moe", "gemm2_reg_pool"}))
    assert any("needs slot 'gemm2_pipe'" in r for r in runtime.slot_reasons())
    monkeypatch.setattr(
        runtime, "SLOTS", frozenset({"moe", "gemm2_pipe", "gemm2_reg_pool"})
    )
    assert runtime.slot_reasons() == []


def test_gemm2_options():
    assert moe.gemm2_options(set()) == dict(gemm2_pipe=False, gemm2_registers=0)
    assert moe.gemm2_options({"gemm2_pipe"}) == dict(gemm2_pipe=True, gemm2_registers=0)
    assert moe.gemm2_options({"gemm2_reg_pool"}) == dict(
        gemm2_pipe=False, gemm2_registers=0
    )
    assert moe.gemm2_options({"gemm2_pipe", "gemm2_reg_pool"}) == dict(
        gemm2_pipe=True, gemm2_registers=96
    )


def test_the_cap_replaces_the_launch_bound():
    source = 'extern "C" __global__ __launch_bounds__(512, 1) void k() {}'
    assert reg.register_pool(source, 96) == (
        'extern "C" __global__ __maxnreg__(96) void k() {}'
    )
