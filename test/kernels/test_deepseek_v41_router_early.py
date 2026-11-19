# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Slot gate_route_early: the router GEMM rebuilt from SGLang's tiny_mma kernel
(sm103/moe/src/router_mma.cu), releasing its programmatic dependents once its weight loads are issued, on one
SM100-family GPU (GB300).

- Logits against SGLang's ``tiny_gemm_bf16`` bit for bit, on SGLang's own schedule for
  the target router (384 x 5120) and the DSpark draft router (128 x 5120), at every width
  SGLang serves on its MMA kernel up to 8 rows: random weights and rows over a wide
  exponent range, zeros, subnormals, infinities and NaNs; strided rows.
- A late router input and a late router output: the router's producer releases its
  dependents at entry and writes the router's rows only after a delay, so the router and
  the one-launch top-k + route + routed-input quant (gate_route_quant) behind it both
  launch early. The router's output buffer and input start poisoned (another input's
  rows and logits); the top-k's ids, weights, slot -> row map, the routed input's MXFP8
  bytes and the GEMM2 rows must equal the chain on SGLang's router bit for bit, replayed
  from a CUDA graph (re-poisoned before each replay), at the served 6 (target) and 5
  (draft) rows, with the top-k releasing GEMM1 after its wait (1) and after the row's
  top-k (3, the default).
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
from sgl_nvidia_mega.deepseek_v41.sm103.moe import router_mma  # noqa: E402

needs_moe = single.needs_moe
HIDDEN = reg.HIDDEN
ROUTERS = {"target": (384, 6), "draft": (128, 3)}


class _Gate(torch.nn.Module):
    """The two MoEGate fields the early router reads."""

    def __init__(self, weight: torch.Tensor):
        super().__init__()
        from sglang.srt.models.deepseek_v2 import tiny_router_gemm_max_tokens

        self.weight = torch.nn.Parameter(weight, requires_grad=False)
        self.tiny_router_gemm_max_tokens = tiny_router_gemm_max_tokens(
            num_experts=weight.size(0),
            hidden_size=weight.size(1),
            weight_dtype=weight.dtype,
        )


def _router(weight):
    schedule = router_mma.sglang_schedule(*weight.shape)
    assert schedule is not None, "SGLang no longer serves this router on its MMA kernel"
    return router_mma.EarlyRouter(_Gate(weight), *schedule)


def _weight(kind, experts, seed):
    gen = torch.Generator(device="cuda").manual_seed(seed)
    w = torch.randn(experts, HIDDEN, device="cuda", generator=gen) * 0.02
    if kind == "wide":
        e = torch.randint(
            -40, 20, (experts, HIDDEN // 64), device="cuda", generator=gen
        )
        w = w * torch.exp2(e.float()).repeat_interleave(64, dim=1)
    elif kind == "special":
        w[3] = 0.0
        w[5, ::7] = 1e-40
        w[9, 100] = float("inf")
        w[11, 2000] = float("-inf")
        w[13, 4000] = float("nan")
        w[15] = 3.0e38
    return w.bfloat16().contiguous()


def _rows(kind, rows, seed, stride=HIDDEN):
    gen = torch.Generator(device="cuda").manual_seed(seed)
    x = torch.randn(rows, stride, device="cuda", generator=gen)
    if kind == "wide":
        e = torch.randint(-60, 60, (rows, stride // 32), device="cuda", generator=gen)
        x = x * torch.exp2(e.float()).repeat_interleave(32, dim=1)
    elif kind == "special":
        x[0, :64] = 0.0
        x[-1, 64:128] = 1e-40
        x[rows // 2, 4999] = float("inf")
    return x.bfloat16()[:, :HIDDEN]


def _equal(a, b):
    return torch.equal(
        a.contiguous().view(torch.int32), b.contiguous().view(torch.int32)
    )


@needs_moe
@pytest.mark.parametrize("which", list(ROUTERS))
@pytest.mark.parametrize("kind", ["wide", "special"])
@pytest.mark.parametrize("strided", [False, True])
def test_logits_match_tiny_gemm(which, kind, strided):
    from sglang.kernels.ops.gemm.tiny_gemm import tiny_gemm_bf16

    experts, _ = ROUTERS[which]
    router = _router(_weight(kind, experts, 11))
    assert router.min_rows <= 6 <= router.max_rows
    for rows in range(router.min_rows, router.max_rows + 1):
        x = _rows(kind, rows, 100 + rows, HIDDEN + 256 if strided else HIDDEN)
        assert router.serves(x), (rows, x.stride())
        want = tiny_gemm_bf16(x, router.weight, out_dtype=torch.float32)
        got = router(x)
        torch.cuda.synchronize()
        assert _equal(got, want), f"{which} {kind} rows={rows} strided={strided}"


@needs_moe
def test_check_and_serves():
    router = _router(_weight("normal", 384, 1))
    assert router.check() == ""
    x = _rows("normal", 6, 1)
    assert router.serves(x)
    assert not router.serves(_rows("normal", 9, 1))
    assert not router.serves(x.float())
    assert not router.serves(_rows("normal", router.min_rows, 1)[:, :4096])
    if router.min_rows > 1:
        assert not router.serves(_rows("normal", 1, 1))  # SGLang runs SIMT there


# ------------------------------------------------- late router input and output


def _pair(layer, top_k, experts, offset):
    return reg.RoutedExpertGemm(
        *layer,
        top_k=top_k,
        max_rows=reg.SERVED_MAX_ROWS,
        gate_topk=True,
        router_experts=experts,
        expert_offset=offset or 0,
        gate_route=True,
        gate_route_quant=True,
    )


@needs_moe
@pytest.mark.parametrize("which, rows", [("target", 6), ("draft", 5)])
@pytest.mark.parametrize("release", [1, 3])
def test_late_router_input_and_output(request, which, rows, release):
    """``release``: where the one-launch top-k releases GEMM1 (bind_early_release)."""
    experts, top_k = ROUTERS[which]
    layer = request.getfixturevalue(f"{which}_layer")
    half, offset = single._ep_half(layer, 1)
    gemm = _pair(half, top_k, experts, offset)
    gemm.bind_early_release(release)
    router = _router(_weight("normal", experts, 21))
    assert router.serves(_rows("normal", rows, 0)) and gemm.serves_gate_route_quant(
        rows
    )
    gen = torch.Generator(device="cuda").manual_seed(rows)
    bias = torch.randn(experts, device="cuda", generator=gen) * 0.05
    num_valid = torch.tensor([rows], dtype=torch.int32, device="cuda")
    good = _rows("normal", rows, 300 + rows).contiguous()
    poison = _rows("normal", rows, 900 + rows).contiguous()
    from sglang.kernels.ops.gemm.tiny_gemm import tiny_gemm_bf16

    poison_logits = tiny_gemm_bf16(poison, router.weight, out_dtype=torch.float32)
    seen = {}
    real = gemm._gemms

    def capture(x_quant, x_scale, record, ctas):
        seen["q"], seen["s"] = x_quant, x_scale
        return real(x_quant, x_scale, record, ctas)

    gemm._gemms = capture

    # the chain on SGLang's router, on the good rows
    ref_logits = tiny_gemm_bf16(good, router.weight, out_dtype=torch.float32)
    ref = gemm.gate_topk_and_gemms(
        None, None, ref_logits, bias, single.SCALE, num_valid, x=good
    )
    ref_q, ref_s = seen["q"].clone(), seen["s"].clone()
    torch.cuda.synchronize()

    x = poison.clone()
    logits = poison_logits.clone()

    def chain():
        late_copy(good, x)
        router(x, out=logits)
        return gemm.gate_topk_and_gemms(
            None, None, logits, bias, single.SCALE, num_valid, x=x, early=True
        )

    def repoison():
        x.copy_(poison)
        logits.copy_(poison_logits)
        torch.cuda.synchronize()

    repoison()
    chain()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        out = chain()
    q = (seen["q"], seen["s"])
    outs, quants = [], []
    for _ in range(3):
        repoison()
        for t in out:
            t.zero_()
        graph.replay()
        torch.cuda.synchronize()
        outs.append(tuple(t.clone() for t in out))
        quants.append(tuple(t.clone() for t in q))
    torch.cuda.synchronize()
    (r0, w0, p0, i0) = ref
    for (r1, w1, p1, i1), (q1, s1) in zip(outs, quants):
        where = f"{which} rows={rows} release={release}"
        assert _equal(logits, ref_logits), where
        assert torch.equal(i1, i0), where
        assert _equal(w1, w0), where
        assert torch.equal(p1, p0), where
        assert torch.equal(q1.view(torch.uint8), ref_q.view(torch.uint8)), where
        assert torch.equal(s1.view(torch.uint8), ref_s.view(torch.uint8)), where
        kept = p0 >= 0
        a = r1[p1.clamp(min=0).long()][kept]
        b = r0[p0.clamp(min=0).long()][kept]
        assert torch.equal(a.view(torch.int16), b.view(torch.int16)), where
