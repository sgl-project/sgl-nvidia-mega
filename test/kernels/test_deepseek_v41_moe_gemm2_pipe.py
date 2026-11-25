# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Slot gemm2_pipe (and its register-pool build, slot gemm2_reg_pool): the routed GEMM2
on muLLM's reference-pipeline + prelude build (gemm2_pipe.cu) against the default GEMM2
(gemm2.cu), bit for bit, on one SM100-family GPU (GB300).

Both pairs run the same route kernel and GEMM1 on the same weights and rows, so the
intermediate is the same; every routed slot's GEMM2 row must be equal bit for bit:
- the served batch-1 widths: the target verify's 6 rows (384 local experts at TP4, 192
  of 384 at EP2) and the DSpark draft's 5 rows (64 of 128);
- a late route record: its producer releases PDL dependents at entry and writes the
  record only after a delay, on a second stream joined before GEMM1 in a captured graph
  (the programmatic edge a join leaves a PDL launch). GEMM2 reads the record before its
  grid-dependency wait (its weight loader, and the prelude at entry); a GEMM2 that read
  it early would see the poisoned (empty) record and leave its rows NaN.
The register-pool build also checks the built kernel's register count.
"""

import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import test_deepseek_v41_moe as single  # noqa: E402
from deepseek_v41_kernel_helpers import late_copy  # noqa: E402
from test_deepseek_v41_moe import draft_layer, target_layer  # noqa: E402, F401

from sgl_nvidia_mega.deepseek_v41.sm103.moe import (  # noqa: E402
    routed_expert_gemm as reg,
)

needs_moe = single.needs_moe
BUILDS = {
    "pipe": {"gemm2_pipe": True},
    "pool": {"gemm2_pipe": True, "gemm2_registers": reg.GEMM2_POOL_REGISTERS},
}


def _pairs(layer, top_k, build, ep_rank=None):
    """(default pair, slot pair) on ``layer`` (or its EP2 window)."""
    router = layer[0].size(0)
    weights, offset = (layer, 0) if ep_rank is None else single._ep_half(layer, ep_rank)
    common = dict(
        top_k=top_k,
        max_rows=reg.SERVED_MAX_ROWS,
        router_experts=router,
        expert_offset=offset,
    )
    return (
        reg.RoutedExpertGemm(*weights, **common),
        reg.RoutedExpertGemm(*weights, **common, **BUILDS[build]),
    )


def _run(gemm, x_q, x_s, ids):
    """Route + GEMM1 + GEMM2 into a NaN-filled output; the routed slots' rows."""
    rows, top_k = ids.shape
    ctas = reg.max_ctas(rows, top_k, gemm.experts)
    permuted, record = gemm._route(ids, rows, ctas)
    out = torch.full(
        (rows * top_k * reg.TILE, reg.HIDDEN), float("nan"), dtype=torch.bfloat16,
        device=ids.device,
    )  # fmt: skip
    gemm._gemms(x_q, x_s, record, ctas, gemm2_rows=out)
    return out, permuted


def _gathered(out, permuted):
    kept = permuted >= 0
    return out[permuted.clamp(min=0).long()][kept].view(torch.int16), kept


def _assert_bitwise(ref, got, where):
    (a, kept_a), (b, kept_b) = ref, got
    assert torch.equal(kept_a, kept_b), where
    assert not torch.isnan(a.view(torch.bfloat16)).any(), f"{where}: reference NaN"
    mismatched = (a != b).any(-1).sum().item()
    assert mismatched == 0, f"{where}: {mismatched} of {a.size(0)} rows differ"


@needs_moe
def test_register_count(target_layer):
    pair = _pairs(target_layer, 6, "pool")
    regs, spill = reg.kernel_registers(pair[1].gemm2_pipe)
    assert regs <= reg.GEMM2_POOL_REGISTERS, (regs, spill)


@needs_moe
@pytest.mark.parametrize("build", list(BUILDS))
@pytest.mark.parametrize(
    "which, rows, top_k, window",
    [("target", 6, 6, None), ("target", 6, 6, 1), ("draft", 5, 3, 1)],
)
@pytest.mark.parametrize("distinct", ["few", "all"])
def test_batch1_widths(request, build, which, rows, top_k, window, distinct):
    layer = request.getfixturevalue(f"{which}_layer")
    experts = layer[0].size(0)
    gen = torch.Generator().manual_seed(rows * 13 + top_k + (window or 0))
    d = rows * top_k if distinct == "all" else top_k + 2
    ids = single._routing(rows, d, gen, experts, top_k).cuda()
    default, slot = _pairs(layer, top_k, build, window)
    x_q, x_s = single._quantized_rows(rows, gen)
    ref = _gathered(*_run(default, x_q, x_s, ids))
    got = _gathered(*_run(slot, x_q, x_s, ids))
    torch.cuda.synchronize()
    _assert_bitwise(ref, got, f"{which} {window} {distinct}")


@needs_moe
@pytest.mark.parametrize("build", list(BUILDS))
def test_late_route_record(target_layer, build):
    rows, top_k = 6, 6
    gen = torch.Generator().manual_seed(77)
    default, slot = _pairs(target_layer, top_k, build)
    x_q, x_s = single._quantized_rows(rows, gen)
    ids = single._routing(rows, 24, gen, 384, top_k).cuda()
    ctas = reg.max_ctas(rows, top_k, 384)
    ref = _gathered(*_run(default, x_q, x_s, ids))
    permuted, good = slot._route(ids, rows, ctas)
    torch.cuda.synchronize()
    record = torch.zeros_like(good)  # poisoned: zero tiles
    out = torch.empty((rows * top_k * reg.TILE, reg.HIDDEN), dtype=torch.bfloat16,
                      device="cuda")  # fmt: skip
    alt = torch.cuda.Stream()

    def body():
        main = torch.cuda.current_stream()  # under capture: the capture stream
        alt.wait_stream(main)
        with torch.cuda.stream(alt):
            late_copy(good, record)
        main.wait_stream(alt)  # GEMM1 is a PDL launch right after the join
        slot._gemms(x_q, x_s, record, ctas, gemm2_rows=out)

    body()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        body()
    for _ in range(5):
        record.zero_()
        out.fill_(float("nan"))
        torch.cuda.synchronize()
        graph.replay()
        torch.cuda.synchronize()
        _assert_bitwise(ref, _gathered(out, permuted), "graph late record")
