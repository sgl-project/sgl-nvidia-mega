# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Slot attn_q_cluster: q_norm + MXFP8 quant + wq_b in one cluster launch
(``sm103/attention/q_cluster.py``, src/q_cluster.cu). A numerics change against the chain's
pair (``rmsnorm_mxfp8`` + ``mxfp8_ll.linear``), so the checks are tolerances:

- the 128x4 scale swizzle and the fragment-major repack invert / match the layouts;
- at 1-8 rows, N = 4096 (the fit entry, attention TP8) and 8192 (the plain cluster-8
  entry, TP4), both trigger builds, on poisoned memory: the error against an FP64
  evaluation of the same quantized weights on the exact normed q_lora is no larger
  than the pair's (within 5 %), and q differs from the pair's by at most a few BF16 ulps;
- against the FP64 product of the MXFP8-quantized normed input (the kernel's quant point)
  the largest error is within 1.5x the pair's (or 2^-7 of the output RMS);
- in a captured graph behind a late writer (releases its dependents before it writes)
  and in front of a PDL reader, every replay reads the new input and the reader sees
  the new q, with the early trigger and without it;
- rows outside 1-8 and non-uniform 32-row scale blocks are refused.
"""

import pytest
import torch
import triton
import triton.language as tl


def _ready() -> bool:
    if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 10:
        return False
    try:
        import flashinfer  # noqa: F401
    except ImportError:
        return False
    return True


needs_gpu = pytest.mark.skipif(not _ready(), reason="needs an SM100-family GPU")

K = 1280
EPS = 1e-6
QKV_N = 1792
SPIN_NS = 50_000


def _weight(n, seed):
    """An e4m3 [n, K] weight with 32x32-uniform UE8M0 scales, as SGLang loads the
    checkpoint's block-scaled wq_b: (weight, swizzled scales, [n/32, K/32] exponents)."""
    from sgl_nvidia_mega.deepseek_v41.sm103.attention import q_cluster as qc

    g = torch.Generator(device="cuda").manual_seed(seed)
    w = (torch.randn(n, K, device="cuda", generator=g) * 96).clamp(-448, 448)
    w = w.to(torch.float8_e4m3fn)
    e = torch.randint(
        116, 122, (n // 32, K // 32), device="cuda", dtype=torch.uint8, generator=g
    )
    return w, qc.swizzle_scales(e.repeat_interleave(32, 0)), e


def _dequant(w, e):
    scale = (
        torch.exp2(e.double() - 127).repeat_interleave(32, 0).repeat_interleave(32, 1)
    )
    return w.double() * scale


def _poison():
    junk = torch.full((1 << 24,), 0x7F, dtype=torch.uint8, device="cuda")
    del junk


def _pair(x, nw, w, ws, n):
    from sgl_nvidia_mega.deepseek_v41.sm103 import mxfp8_ll
    from sgl_nvidia_mega.deepseek_v41.sm103.attention.q_norm_quant import rmsnorm_mxfp8

    consumer = mxfp8_ll.WQ_B_TP8 if n == 4096 else None
    _, q8, s8 = rmsnorm_mxfp8(x, nw, EPS, pdl=True)
    return mxfp8_ll.linear(q8, s8, w, ws, consumer=consumer)


def _normed64(x, nw):
    x64 = x.double()
    inv = torch.rsqrt((x64 * x64).mean(-1, keepdim=True) + EPS)
    return x64 * inv * nw.double()


def _mx_quant64(y):
    """MXFP8 of BF16-rounded y (the kernel's quant point): e4m3 values times their scales,
    exponents ceil(log2(max(amax, 1e-4) / 448))."""
    y = y.float().bfloat16().float()
    blocks = y.view(y.shape[0], -1, 32)
    amax = blocks.abs().amax(-1, keepdim=True).clamp_min(1e-4)
    ratio = amax * torch.tensor(1 / 448, dtype=torch.float32)
    exp = torch.ceil(torch.log2(ratio.double())).float()
    scaled = (blocks * torch.exp2(-exp)).clamp(-448, 448).to(torch.float8_e4m3fn)
    return (scaled.double() * torch.exp2(exp.double())).view_as(y)


@needs_gpu
def test_layouts():
    from sgl_nvidia_mega.deepseek_v41.sm103 import mxfp8_ll
    from sgl_nvidia_mega.deepseek_v41.sm103.attention import q_cluster as qc

    w = torch.randn(4096, K, device="cuda").bfloat16()
    q8, sf = mxfp8_ll.quantize(w)
    rows = qc.unswizzle_scales(sf, 4096, K)
    assert torch.equal(
        qc.swizzle_scales(rows), sf.view(torch.uint8).reshape(-1)[: rows.numel()]
    )
    # FlashInfer's quantizer computes per-row scales: the repack refuses them
    assert qc.block_scales(rows) is None
    with pytest.raises(ValueError):
        qc.pack(q8, sf)
    # the fragment-major order (muLLM's layout contract): word 4c + 2h + q of lane 4g + t
    # in span s of tile i holds values 64s + 32c + 8t + 4q .. + 3 of row 16i + 8h + g
    raw = torch.arange(32 * 128, device="cuda", dtype=torch.int64)
    wb = (raw % 251).to(torch.uint8).view(32, 128)
    packed = qc.pack_fp8_weight(wb.view(torch.float8_e4m3fn))
    for i, s, lane, word in ((0, 0, 0, 0), (1, 1, 13, 7), (0, 1, 31, 5), (1, 0, 6, 2)):
        g, t = lane // 4, lane % 4
        c, h, q = word // 4, (word // 2) % 2, word % 2
        row, col = 16 * i + 8 * h + g, 64 * s + 32 * c + 8 * t + 4 * q
        got = packed[i, s, lane * 32 + word * 4 : lane * 32 + word * 4 + 4]
        assert torch.equal(got, wb[row, col : col + 4]), (i, s, lane, word)


@needs_gpu
@pytest.mark.parametrize("n", (4096, 8192))
@pytest.mark.parametrize("early_trigger", (True, False))
def test_tolerance_vs_pair_and_fp64(n, early_trigger):
    from sgl_nvidia_mega.deepseek_v41.sm103.attention import q_cluster as qc

    w, ws, e = _weight(n, n)
    packed = qc.pack(w, ws)
    w64 = _dequant(w, e)
    nw = (1 + 0.1 * torch.randn(K, device="cuda")).bfloat16()
    stats = []
    for rows in range(1, 9):
        assert not qc.reason(w, rows)
        for trial in range(3):
            qkv = torch.randn(rows, QKV_N, device="cuda").bfloat16() * (0.25 + trial)
            x = qkv[:, :K]
            pair = _pair(x, nw, w, ws, n)
            _poison()
            out = qc.q_cluster(x, nw, EPS, packed, early_trigger=early_trigger)
            torch.cuda.synchronize()
            assert out.shape == (rows, n) and torch.isfinite(out.float()).all()
            # the same weights on the exact normed input, and on the MXFP8 quant of it
            exact = _normed64(x, nw) @ w64.t()
            quant = _mx_quant64(_normed64(x, nw)) @ w64.t()
            rms = exact.pow(2).mean().sqrt()
            err_qc = (out.double() - exact).pow(2).mean().sqrt() / rms
            err_pair = (pair.double() - exact).pow(2).mean().sqrt() / rms
            assert err_qc <= 1.05 * err_pair + 1e-6, (rows, trial, err_qc, err_pair)
            # against the quantized-input product: BF16 output rounding plus FP32 order
            worst = (out.double() - quant).abs().max() / rms
            worst_pair = (pair.double() - quant).abs().max() / rms
            assert worst < max(1.5 * worst_pair, 2**-7), (rows, trial, float(worst))
            ulp = (out.view(torch.int16).int() - pair.view(torch.int16).int()).abs()
            stats.append(
                (rows, float(err_qc), float(err_pair), float((ulp > 0).float().mean()))
            )
            assert ulp.max() <= 64, (rows, trial, int(ulp.max()))
    print(f"n={n} et={early_trigger}: (rows, rel err qc, rel err pair, frac != pair)")
    for s in stats:
        print("  ", s)


@needs_gpu
def test_refusals():
    from sgl_nvidia_mega.deepseek_v41.sm103.attention import q_cluster as qc

    w, ws, _ = _weight(4096, 1)
    packed = qc.pack(w, ws)
    nw = torch.ones(K, device="cuda").bfloat16()
    assert qc.reason(w, 9) and qc.reason(w, 0)
    with pytest.raises(ValueError):
        qc.q_cluster(torch.zeros(9, K, device="cuda").bfloat16(), nw, EPS, packed)
    assert qc.reason(torch.zeros(4096, 1024, device="cuda").to(torch.float8_e4m3fn))


SPIN = None


def _late_copy_kernel():
    global SPIN
    if SPIN is None:

        @triton.jit
        def late_copy(
            SRC, DST, N: tl.constexpr, BLOCK: tl.constexpr, spin_ns: tl.constexpr
        ):
            # release the PDL dependents, then write late
            tl.extra.cuda.gdc_launch_dependents()
            t0 = tl.inline_asm_elementwise(
                "mov.u64 $0, %globaltimer;",
                "=l",
                [],
                dtype=tl.int64,
                is_pure=False,
                pack=1,
            )
            t = t0
            while t - t0 < spin_ns:
                t = tl.inline_asm_elementwise(
                    "mov.u64 $0, %globaltimer;",
                    "=l",
                    [],
                    dtype=tl.int64,
                    is_pure=False,
                    pack=1,
                )
            i = tl.arange(0, BLOCK)
            tl.store(DST + i, tl.load(SRC + i, i < N), i < N)

        SPIN = late_copy
    return SPIN


@triton.jit
def _pdl_reader(SRC, DST, N: tl.constexpr, BLOCK: tl.constexpr):
    # a dependent that reads its input only after its PDL wait, as swap-AB reads q
    tl.extra.cuda.gdc_wait()
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    tl.store(DST + i, tl.load(SRC + i, i < N), i < N)


@needs_gpu
@pytest.mark.parametrize("early_trigger", (True, False))
@pytest.mark.parametrize("rows", (1, 6, 8))
def test_graph_replay_reads_late_writer_and_feeds_pdl_reader(early_trigger, rows):
    from sgl_nvidia_mega.deepseek_v41.sm103.attention import q_cluster as qc

    w, ws, _ = _weight(4096, 7)
    packed = qc.pack(w, ws)
    nw = (1 + 0.1 * torch.randn(K, device="cuda")).bfloat16()
    src = torch.empty(rows, QKV_N, dtype=torch.bfloat16, device="cuda")
    qkv = torch.zeros_like(src)
    seen = torch.empty(rows, 4096, dtype=torch.bfloat16, device="cuda")

    def chain():
        s, d = src.view(torch.int16).reshape(-1), qkv.view(torch.int16).reshape(-1)
        _late_copy_kernel()[(1,)](
            s,
            d,
            N=s.numel(),
            BLOCK=triton.next_power_of_2(s.numel()),
            spin_ns=SPIN_NS,
            launch_pdl=True,
        )
        q = qc.q_cluster(qkv[:, :K], nw, EPS, packed, early_trigger=early_trigger)
        n = q.numel()
        _pdl_reader[(triton.cdiv(n, 4096),)](
            q.view(torch.int16).reshape(-1),
            seen.view(torch.int16).reshape(-1),
            N=n,
            BLOCK=4096,
            launch_pdl=True,
        )
        return q

    src.copy_(torch.randn_like(src, dtype=torch.float32))
    chain()
    torch.cuda.synchronize()
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g, stream=stream):
        out = chain()
    torch.cuda.synchronize()
    for seed in range(1, 9):
        gen = torch.Generator(device="cuda").manual_seed(seed)
        src.copy_(torch.randn(src.shape, device="cuda", generator=gen))
        qkv.zero_()
        want = qc.q_cluster(src[:, :K].contiguous(), nw, EPS, packed).clone()
        torch.cuda.synchronize()
        qkv.zero_()
        seen.zero_()
        g.replay()
        torch.cuda.synchronize()
        assert torch.equal(out.view(torch.int16), want.view(torch.int16)), seed
        assert torch.equal(seen.view(torch.int16), want.view(torch.int16)), seed
