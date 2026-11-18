# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Slot moe_balanced on one SM100-family GPU: the route kernels' balanced dealing.

Both expert-parallel groups hold all 384 target experts (SGLang's balanced expert
storage); group g's pair keeps the step's distinct experts of even (g = 0) or odd (g = 1)
rank in id order. Against the undealt pair over the same 384 experts, for the route
kernel and the fused top-k + route (gate_route) at the served 6 rows, and route_wide at
48 rows:

- every valid slot is kept by exactly one group, and group g keeps exactly the distinct
  experts sorted[g::2] (so the groups serve ceil(D / 2) and floor(D / 2));
- each kept slot's GEMM2 row equals the undealt pair's row for that slot (same expert
  tile, another CTA); bit for bit is reported, one BF16 rounding is the bound;
- the dealt record of the fused launch equals route.cu's for the same ids.
"""

import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import test_deepseek_v41_moe as single  # noqa: E402

from sgl_nvidia_mega.deepseek_v41.sm103.moe import (  # noqa: E402
    routed_expert_gemm as reg,
)

EXPERTS, TOP_K = 384, 6


@pytest.fixture(scope="module")
def layer():
    return single._shuffled_layer(torch.Generator().manual_seed(5), EXPERTS)


def _pair(layer, deal_group, **kw):
    return reg.RoutedExpertGemm(
        *layer,
        top_k=TOP_K,
        max_rows=reg.SERVED_MAX_ROWS,
        router_experts=EXPERTS,
        deal_group=deal_group,
        **single.GEMM2_OPTIONS,
        **kw,
    )


def _check_split(ids, outs, full, where):
    """outs: {group: (rows_out, permuted)}; full: the undealt pair's (rows_out, permuted)."""
    valid = ids >= 0
    kept = {g: p >= 0 for g, (_, p) in outs.items()}
    assert torch.equal(kept[0] | kept[1], valid), where
    assert not bool((kept[0] & kept[1]).any()), where
    distinct = torch.unique(ids[valid]).tolist()  # sorted
    bitwise = True
    for g, (rows_out, permuted) in outs.items():
        mine = sorted(set(ids[kept[g]].tolist()))
        assert mine == distinct[g::2], f"{where}: group {g} kept {mine}"
        got = rows_out[permuted.clamp(min=0).long()][kept[g]]
        want = full[0][full[1].clamp(min=0).long()][kept[g]]
        bitwise &= torch.equal(got.view(torch.int16), want.view(torch.int16))
        torch.testing.assert_close(got.float(), want.float(), rtol=2**-7, atol=1e-6)
    return len(distinct), bitwise


@single.needs_moe
@pytest.mark.parametrize("rows, distinct", [(6, 19), (6, 36), (48, 384)])
def test_route_kernels_deal_the_experts(layer, rows, distinct):
    from sglang.srt.layers.quantization.fp8_utils import flashinfer_mxfp8_quantize

    gen = torch.Generator().manual_seed(300 + rows * 7 + distinct)
    x = torch.randn(rows, reg.HIDDEN, generator=gen).to(torch.bfloat16).cuda()
    x_q, x_s = flashinfer_mxfp8_quantize(x, False, alignment=reg.HIDDEN)
    x_s = x_s.view(torch.float8_e4m3fn).reshape(rows, -1)
    ids = single._routing(rows, distinct, gen, EXPERTS, TOP_K).cuda()
    weights = torch.softmax(torch.randn(rows, TOP_K, generator=gen), -1).cuda()
    full = _pair(layer, -1)(x_q, x_s, ids, weights)
    outs = {}
    for g in (0, 1):
        rows_out, _, permuted = _pair(layer, g)(x_q, x_s, ids, weights)
        outs[g] = (rows_out, permuted)
    torch.cuda.synchronize()
    d, bitwise = _check_split(
        ids, outs, (full[0], full[2]), f"rows={rows} d={distinct}"
    )
    print(f"moe_balanced route rows={rows} distinct={d}: split ok, bitwise={bitwise}")


@single.needs_moe
def test_gate_route_deals_the_experts(layer):
    rows = 6
    gen = torch.Generator(device="cuda").manual_seed(41 + rows)
    logits = torch.randn(rows, EXPERTS, device="cuda", generator=gen)
    bias = torch.randn(EXPERTS, device="cuda", generator=gen) * 0.05
    num_valid = torch.tensor([rows], dtype=torch.int32, device="cuda")
    x_q, x_s = single._quantized_rows(rows, torch.Generator().manual_seed(rows))
    full = _pair(layer, -1, gate_topk=True, gate_route=True)
    rows_full, _, perm_full, ids = full.gate_topk_and_gemms(
        x_q, x_s, logits, bias, single.SCALE, num_valid
    )
    outs = {}
    for g in (0, 1):
        pair = _pair(layer, g, gate_topk=True, gate_route=True)
        assert pair.serves_gate_route(rows)
        records = {}
        real = pair._gemms

        def capture(x_q, x_s, record, ctas, records=records, real=real):
            records[len(records)] = record.clone()
            return real(x_q, x_s, record, ctas)

        pair._gemms = capture
        rows_out, _, permuted, ids_g = pair.gate_topk_and_gemms(
            x_q, x_s, logits, bias, single.SCALE, num_valid
        )
        # the same ids through route.cu (gate_topk, then the route kernel)
        kernel = pair.gate_route_kernel
        pair.gate_route_kernel = None
        _, _, permuted_route, _ = pair.gate_topk_and_gemms(
            x_q, x_s, logits, bias, single.SCALE, num_valid
        )
        pair.gate_route_kernel = kernel
        torch.cuda.synchronize()
        assert torch.equal(ids_g, ids), (rows, g)
        assert torch.equal(records[0], records[1]), (rows, g)
        assert torch.equal(permuted, permuted_route), (rows, g)
        outs[g] = (rows_out, permuted)
    d, bitwise = _check_split(
        ids, outs, (rows_full, perm_full), f"gate_route rows={rows}"
    )
    print(
        f"moe_balanced gate_route rows={rows} distinct={d}: split ok, bitwise={bitwise}"
    )


@single.needs_moe
def test_dealing_needs_every_expert(layer):
    half = tuple(t[: EXPERTS // 2] for t in layer)
    with pytest.raises(ValueError, match="balanced dealing"):
        reg.RoutedExpertGemm(
            *half,
            top_k=TOP_K,
            max_rows=reg.SERVED_MAX_ROWS,
            router_experts=EXPERTS,
            expert_offset=EXPERTS // 2,
            deal_group=1,
        )
