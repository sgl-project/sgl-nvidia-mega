# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Slot attn_wo_a_fp8: the fused wo_a streaming wo_a as E4M3 with 32x32 UE8M0 scales is bit
for bit the BF16 launch when the BF16 wo_a is the exact dequant of that form (as SGLang
loads DeepSeek-V4.1's FP8 wo_a): codes and scales at the served 6 rows and 1 group
(attention TP8), int32 / int64 positions, on poisoned memory and in a graph replay. ``fp8_weight`` recovers the checkpoint's form exactly and refuses a BF16 weight
that is not one."""

import os
import sys

import pytest
import torch

from sgl_nvidia_mega.deepseek_v41.sm103.attention import wo_a

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from deepseek_v41_kernel_helpers import poison  # noqa: E402


def _ready() -> bool:
    return torch.cuda.is_available() and torch.cuda.get_device_capability()[0] == 10


needs_gpu = pytest.mark.skipif(not _ready(), reason="needs an SM100-family GPU")


def _checkpoint_wo_a(groups, seed):
    """BF16 wo_a as SGLang loads it: E4M3 times 32x32 power-of-two scales (exponents
    -13..-10, the checkpoint's range), and those codes and exponents."""
    g = torch.Generator(device="cuda").manual_seed(seed)
    n = groups * wo_a.RANK
    codes = (torch.randn(n, wo_a.GROUP_K, device="cuda", generator=g) * 100).clamp(
        -448, 448
    )
    codes = codes.to(torch.float8_e4m3fn)
    exp = torch.randint(
        -13, -9, (n // 32, wo_a.GROUP_K // 32), device="cuda", generator=g
    )
    scale = torch.exp2(exp.float()).repeat_interleave(32, 0).repeat_interleave(32, 1)
    weight = (codes.float() * scale).to(torch.bfloat16)
    return weight.view(groups, wo_a.RANK, wo_a.GROUP_K).contiguous(), codes, exp


def _inputs(rows, groups, int64):
    angles = torch.rand(4096, 32, device="cuda") * 6.3
    freqs = torch.view_as_real(torch.polar(torch.ones_like(angles), angles)).flatten(1)
    pos = torch.randint(0, 4096, (rows,), device="cuda")
    pos = pos if int64 else pos.int()
    o = torch.randn(rows, 8 * groups, 512, device="cuda").bfloat16()
    return o.view(rows, groups, -1), freqs.contiguous(), pos


@needs_gpu
def test_fp8_weight_recovers_the_checkpoint_form():
    groups = 1
    weight, codes, exp = _checkpoint_wo_a(groups, groups)
    fp8 = wo_a.fp8_weight(weight)
    assert fp8.codes.shape == (groups * wo_a.RANK, wo_a.GROUP_K)
    assert fp8.scales.shape == (groups * 32, wo_a.GROUP_K // 32)
    # codes may be renormalized per block (a block whose max is below 448 * 2^e takes a
    # smaller e), but the value of every element is the same
    back = fp8.codes.view(torch.float8_e4m3fn).float() * torch.exp2(
        fp8.scales.float() - 127
    ).repeat_interleave(32, 0).repeat_interleave(32, 1)
    assert torch.equal(back.to(torch.bfloat16).view_as(weight), weight)
    bad = weight.clone()
    bad.view(-1)[12345] = 0.1234  # not an E4M3 value times a power of two
    with pytest.raises(ValueError):
        wo_a.fp8_weight(bad)


@needs_gpu
def test_fp8_launch_is_the_bf16_launch_bit_for_bit():
    weight, _, _ = _checkpoint_wo_a(1, 11)
    fp8 = wo_a.fp8_weight(weight)
    for int64 in (False, True):
        x, freqs, pos = _inputs(6, 1, int64)
        kw = dict(early_trigger=True)  # as the chain launches it
        poison()
        ref_c, ref_s = wo_a.fused_rope_wo_a(x, weight, freqs, pos, **kw)
        poison()
        got_c, got_s = wo_a.fused_rope_wo_a(x, weight, freqs, pos, fp8=fp8, **kw)
        torch.cuda.synchronize()
        assert torch.equal(got_c.view(torch.uint8), ref_c.view(torch.uint8)), int64
        assert torch.equal(got_s, ref_s), int64


@needs_gpu
def test_graph_replay():
    weight, _, _ = _checkpoint_wo_a(1, 99)
    fp8 = wo_a.fp8_weight(weight)
    x, freqs, pos = _inputs(6, 1, False)
    src = torch.empty_like(x)
    outs = []

    def run():
        x.copy_(src)
        return wo_a.fused_rope_wo_a(x, weight, freqs, pos, fp8=fp8)

    src.copy_(torch.randn_like(x, dtype=torch.float32))
    run()
    torch.cuda.synchronize()
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g, stream=s):
        outs = run()
    for seed in range(4):
        gen = torch.Generator(device="cuda").manual_seed(seed)
        src.copy_(torch.randn(src.shape, device="cuda", generator=gen))
        g.replay()
        torch.cuda.synchronize()
        ref_c, ref_s = wo_a.fused_rope_wo_a(src.clone(), weight, freqs, pos)
        assert torch.equal(outs[0].view(torch.uint8), ref_c.view(torch.uint8)), seed
        assert torch.equal(outs[1], ref_s), seed
