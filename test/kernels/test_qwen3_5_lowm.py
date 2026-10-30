# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""The low-M projection launches (slots lowm_in / lowm_out / lowm_se, sm103/lowm) on one SM100-family GPU
(GB300), numeric against SGLang's static quantize + FlashInfer FP8 BMM, eager and replayed from a CUDA graph:
- the in-register quantization is SGLang's static quantize bit for bit: a BF16-input launch equals, byte for
  byte, the same launch on SGLang's E4M3 quantization of the rows;
- each TP8 projection (in_proj_qkvz with in_proj_ba's BF16 rows, qkv_proj, out_proj, o_proj) at the tactics
  the slots use and at the default one, against fp64 on the same E4M3 operands: at most 1.05 x the max
  relative error of FlashInfer's BMM plus one BF16 ulp of the output scale;
- the cluster K split equals the same tactic without it up to the FP32 sum order;
- the shared expert's two launches against SGLang's chain (quantize, gate_up BMM, SiluAndMul, quantize,
  down BMM, fused_gate_sigmoid_mul)."""

import pytest
import torch

from sgl_nvidia_mega.qwen3_5.sm103 import lowm

FP8 = torch.float8_e4m3fn


def _gpu_ready() -> bool:
    if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 10:
        return False
    try:
        import flashinfer  # noqa: F401
        from cuda.bindings import nvrtc  # noqa: F401
        from sglang.kernels.ops.quantization.fp8_kernel import (  # noqa: F401
            static_quant_fp8,
        )
    except ImportError:
        return False
    return True


pytestmark = pytest.mark.skipif(not _gpu_ready(), reason="needs an SM100-family GPU")


def _weight(g, n, k, std=0.05):
    w = torch.randn(n, k, device="cuda", generator=g) * std
    s = (w.abs().amax() / 448.0).float().reshape(1)
    return (w / s).to(FP8).contiguous(), s


def _rows(g, m, k):
    x = (torch.randn(m, k, device="cuda", generator=g) * 2).to(torch.bfloat16)
    s = (x.float().abs().amax() / 448.0 * 1.1).reshape(1).float()
    return x, s


def _bmm(x8, w8, s, sw):
    from sglang.srt.layers.quantization.fp8_utils import flashinfer_bmm_fp8

    return flashinfer_bmm_fp8(x8, w8.t(), s, sw, torch.bfloat16)


def _rel(a, ref):
    return float((a.float() - ref).abs().max() / ref.abs().max())


def _tactics():
    from sgl_nvidia_mega.qwen3_5.lowm import TACTICS

    return TACTICS


@pytest.mark.parametrize(
    "n, n16, k, rows",
    [
        (2560, 16, 4096, 14),
        (2560, 0, 4096, 14),
        (4096, 0, 1024, 14),
        (4096, 0, 1024, 28),
        (2560, 0, 4096, 28),
        (4096, 0, 1024, 56),
    ],
)
def test_projection_is_bmm_numerically(n, n16, k, rows):
    from sglang.kernels.ops.quantization.fp8_kernel import static_quant_fp8

    g = torch.Generator(device="cuda").manual_seed(rows * 7 + n16)
    w8, sw = _weight(g, n, k)
    w16 = (
        (torch.randn(n16, k, device="cuda", generator=g) * 0.02).to(torch.bfloat16)
        if n16
        else None
    )
    x, s = _rows(g, rows, k)
    x8, _ = static_quant_fp8(x, s)
    exact = (x8.double() @ w8.double().t()) * float(s) * float(sw)
    ref = _bmm(x8, w8, s, sw)
    bound = 1.05 * _rel(ref, exact) + 2.0**-8
    used = _tactics().get((n, n16, k, rows, False))
    for tactic in ([used] if used is not None else []) + [
        lowm.default_tactic(rows, n, n16, k),
        None,
    ]:
        out_q = torch.empty(rows, n, dtype=torch.bfloat16, device="cuda")
        out_8 = torch.empty_like(out_q)
        out2 = (
            torch.empty(rows, n16, dtype=torch.bfloat16, device="cuda") if n16 else None
        )
        lowm.gemm(
            x16=x,
            w8=w8,
            w_scale=sw,
            in_scale=s,
            w16=w16,
            out=out_q,
            out2=out2,
            tactic=tactic,
        )
        torch.cuda.synchronize()
        assert _rel(out_q, exact) <= bound, (tactic, _rel(out_q, exact), bound)
        if tactic is not None:
            # the same FP8 CTAs on SGLang's E4M3 rows: the quantization is SGLang's, byte for byte
            lowm.gemm(x8=x8, w8=w8, w_scale=sw, in_scale=s, out=out_8, tactic=tactic)
            torch.cuda.synchronize()
            assert torch.equal(out_q.view(torch.int16), out_8.view(torch.int16)), tactic
        if n16:
            e2 = x.double() @ w16.double().t()
            r2 = torch.nn.functional.linear(x, w16)
            assert _rel(out2, e2) <= 1.05 * _rel(r2, e2) + 2.0**-8


@pytest.mark.parametrize("rows", [14, 28])
def test_cluster_split_is_the_same_sums(rows):
    g = torch.Generator(device="cuda").manual_seed(rows)
    w8, sw = _weight(g, 4096, 1024)
    x, s = _rows(g, rows, 1024)
    a = torch.empty(rows, 4096, dtype=torch.bfloat16, device="cuda")
    b = torch.empty_like(a)
    lowm.gemm(
        x16=x, w8=w8, w_scale=sw, in_scale=s, out=a, tactic=lowm.Tactic(4, 4, 1, 1, 1)
    )
    lowm.gemm(
        x16=x, w8=w8, w_scale=sw, in_scale=s, out=b, tactic=lowm.Tactic(4, 4, 1, 1, 2)
    )
    torch.cuda.synchronize()
    d = (a.float() - b.float()).abs()
    assert float(d.max()) <= float(a.float().abs().max()) * 2.0**-7


@pytest.mark.parametrize("rows", [14, 28, 56])
def test_shared_expert_is_sglang_numerically(rows):
    from sgl_kernel import silu_and_mul
    from sglang.kernels.ops.elementwise.elementwise import fused_gate_sigmoid_mul
    from sglang.kernels.ops.quantization.fp8_kernel import static_quant_fp8

    g = torch.Generator(device="cuda").manual_seed(100 + rows)
    gu, sgu = _weight(g, 256, 4096)
    dn, sdn = _weight(g, 4096, 128)
    gate = (torch.randn(1, 4096, device="cuda", generator=g) * 0.02).to(torch.bfloat16)
    x, s = _rows(g, rows, 4096)
    s_h = torch.tensor([0.02], device="cuda")
    # SGLang: quantize, gate_up BMM, SiluAndMul, quantize, down BMM, gate
    q, _ = static_quant_fp8(x, s)
    h = torch.empty(rows, 128, dtype=torch.bfloat16, device="cuda")
    silu_and_mul(_bmm(q, gu, s, sgu), h)
    q2, _ = static_quant_fp8(h, s_h)
    ref = fused_gate_sigmoid_mul(x, gate.squeeze(0), _bmm(q2, dn, s_h, sdn))
    gate8 = torch.zeros(8, 4096, dtype=torch.bfloat16, device="cuda")
    gate8[:1].copy_(gate)
    t = _tactics()
    h8 = torch.empty(rows, 128, dtype=FP8, device="cuda")
    logits = torch.empty(rows, 8, dtype=torch.float32, device="cuda")
    out = torch.empty(rows, 4096, dtype=torch.bfloat16, device="cuda")

    def run():
        lowm.gemm(
            x16=x,
            w8=lowm.swiglu_pairs(gu),
            w_scale=sgu,
            in_scale=s,
            w16=gate8,
            out=h8,
            out2=logits,
            swiglu_scale=s_h,
            tactic=t.get((256, 8, 4096, rows, False)),
        )
        lowm.gemm(
            x8=h8,
            w8=dn,
            w_scale=sdn,
            in_scale=s_h,
            out=out,
            gate=logits[:, 0],
            tactic=t.get((4096, 0, 128, rows, True)),
        )

    run()
    torch.cuda.synchronize()
    # the hidden FP8 rows differ from SGLang's only where the FP32 sum order moves a BF16 rounding
    assert float((h8.float() != q2.float()).float().mean()) < 0.02
    rel = float((out.float() - ref.float()).norm() / ref.float().norm())
    assert rel < 2e-2, rel
    first = out.clone()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        run()
    out.zero_()
    graph.replay()
    torch.cuda.synchronize()
    assert torch.equal(out, first)
