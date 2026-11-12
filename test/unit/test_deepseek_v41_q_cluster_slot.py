# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Slot attn_q_cluster on CPU: opt-in, needs attn, taken only in layers without an indexer
on the low-latency wq_b at 1-8 rows, ahead of attn_q_norm_gemm, and declined (logged) where
wq_b's scales are not uniform over 32-row blocks."""

from types import SimpleNamespace
from unittest import mock

import pytest
import torch

from sgl_nvidia_mega.deepseek_v41.attention import chain
from sgl_nvidia_mega.deepseek_v41.sm103.attention import q_cluster as qc


def test_the_slot_runs_by_default_and_needs_attn(monkeypatch):
    from sgl_nvidia_mega.deepseek_v41 import runtime

    assert "attn_q_cluster" in runtime.DEFAULT_ON_SLOTS
    assert "attn_q_cluster" in runtime.NUMERICS_CHANGE_SLOTS
    assert "attn_q_cluster" not in runtime.OPT_IN_SLOTS
    monkeypatch.setattr(runtime, "SLOTS", frozenset({"attn_q_cluster"}))
    assert any("needs slot 'attn'" in r for r in runtime.slot_reasons())
    monkeypatch.setattr(runtime, "SLOTS", frozenset({"attn", "attn_q_cluster"}))
    assert runtime.slot_reasons() == []


def test_scale_layouts_round_trip():
    n, k = 256, qc.K
    rows = torch.randint(0, 255, (n, k // 32), dtype=torch.uint8)
    assert torch.equal(qc.unswizzle_scales(qc.swizzle_scales(rows), n, k), rows)
    assert qc.block_scales(rows) is None
    blocks = torch.randint(100, 130, (n // 32, k // 32), dtype=torch.uint8)
    assert torch.equal(qc.block_scales(blocks.repeat_interleave(32, 0)), blocks)


def _attn(indexer=None, uniform=True, n=4096):
    w = torch.zeros(n, qc.K, dtype=torch.float8_e4m3fn)
    e = torch.full((n // 32, qc.K // 32), 120, dtype=torch.uint8)
    rows = e.repeat_interleave(32, 0)
    if not uniform:
        rows[1, 0] = 121
    return SimpleNamespace(
        layer_id=3,
        n_local_heads=8,
        head_dim=512,
        indexer=indexer,
        wq_b=SimpleNamespace(
            weight=w, weight_scale_inv_swizzled=qc.swizzle_scales(rows)
        ),
        q_norm=SimpleNamespace(
            weight=torch.ones(qc.K).bfloat16(), variance_epsilon=1e-6
        ),
    )


def _ops(attn, q_cluster=True, q_norm_gemm=False):
    with (
        mock.patch.object(chain, "wq_b_ll_applies", return_value=True),
        mock.patch.object(chain, "index_q_mxfp8_applies", return_value=False),
        mock.patch.object(chain, "qkv_a_ll_applies", return_value=False),
    ):
        return chain.ChainOps(
            attn,
            rope_in_attention=True,
            q_cluster=q_cluster,
            q_norm_gemm=q_norm_gemm,
        )


@pytest.mark.parametrize(
    "indexer, uniform, slot, want",
    [
        (None, True, True, True),
        (object(), True, True, False),  # an indexer reads q_lora
        (None, False, True, False),  # per-row scales: declined
        (None, True, False, False),  # slot off
    ],
)
def test_which_layers_take_it(indexer, uniform, slot, want):
    ops = _ops(_attn(indexer, uniform), q_cluster=slot)
    assert ops.q_cluster is want
    if want:
        assert ops.q_cluster_wq_b.weight.shape == (4096 // 16, qc.K // 64, 1024)
        assert ops.q_cluster_wq_b.scales.shape == (4096 // 32, qc.K // 32)


def test_rows_and_precedence_over_q_norm_gemm():
    attn = _attn()
    ops = _ops(attn, q_cluster=True, q_norm_gemm=True)
    with mock.patch.object(chain.q_norm_gemm, "reason", return_value="wide"):
        for rows in range(1, 9):
            assert ops.defers_norm(attn, rows)
        assert not ops.defers_norm(attn, 9)
    q = chain.DeferredNorm(torch.zeros(6, qc.K).bfloat16())
    with (
        mock.patch.object(chain.q_cluster_kernel, "q_cluster") as launch,
        mock.patch.object(chain.q_norm_gemm, "q_norm_linear") as fused,
    ):
        launch.return_value = torch.zeros(6, 4096).bfloat16()
        out = ops.compute_q_b(attn, q, None, None)
    assert launch.call_count == 1 and fused.call_count == 0
    assert launch.call_args.args[3] is ops.q_cluster_wq_b
    assert out.shape == (6, 8, 512)


@pytest.mark.parametrize(
    "listed, heads, takes",
    [(False, 8, True), (False, 16, False), (True, 16, True)],
    ids=["default_tp8", "default_tp4_left_off", "listed_tp4"],
)
def test_a_default_start_takes_it_at_attention_tp8_only(
    monkeypatch, listed, heads, takes
):
    from sgl_nvidia_mega.deepseek_v41 import attention, runtime

    if listed:
        monkeypatch.setenv(runtime.SLOTS_ENV, "attn,attn_q_cluster")
    else:
        monkeypatch.delenv(runtime.SLOTS_ENV, raising=False)
    attn = SimpleNamespace(n_local_heads=heads)
    assert attention._q_cluster_default_takes(attn) is takes
