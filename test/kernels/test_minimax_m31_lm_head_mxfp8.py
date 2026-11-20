# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""MiniMax M3.1's MXFP8 LM-head stream (sm103/logits/lm_head_mxfp8.py, slot
lm_head_mxfp8) on SM100-family GPUs; needs nvcc and the CUDA Python bindings.

On Gaussian weights of the TP4 shard's shape (N 50016, 96 rows past a multiple of 128;
K 6144) and one request's rows:

- bit for bit: the MXFP8 kernel equals the same kernel's BF16 build on the dequantized
  weight (same fragments, MMAs and order) at 1-8 rows, on a ragged N too, and on codes
  that cover every E4M3 value (subnormals, +-0, 448) under scales 2^-30 .. 2^7; the
  kernel's own sum against torch's on the dequantized weight (summation order only);
- the shard's N as it is: the output is [rows, N] and nothing past it (a wider buffer
  keeps its sentinel), the scale bytes of the rows the swizzle pads to 128 never reach
  it, and neither do NaN codes placed right after the last row;
- against SGLang's BF16 projection (``x @ W.T`` in torch) on the BF16 weight: the
  logits within the declared bound (spec.LM_HEAD_TOLERANCES), and the same argmax on
  rows with a clear margin (more than twice the bound of the peak);
- the slot as it serves: the stand-in built by ``logits.build`` through SGLang's own
  ``LogitsProcessor._compute_lm_head`` against SGLang's head, eagerly and from CUDA
  graph replay on new rows;
- a dependent (PDL) launch inside a CUDA graph behind the kernel that writes its rows,
  replayed on new rows, a strided input and a caller's output buffer; a launch without
  its build raises;
- the quantizer on the GPU equals the CPU one bit for bit;
- microseconds per launch beside the BF16 projection's (printed with -s).

Run with ``python -m pytest test/kernels/test_minimax_m31_lm_head_mxfp8.py -s``.
"""

from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from sgl_nvidia_mega.minimax_m31.sm103 import nvcc
from sgl_nvidia_mega.minimax_m31.sm103.logits import lm_head_mxfp8 as lh

N, K = 50_016, 6144  # one TP4 rank's shard of the 200,064-token vocabulary
ROWS = 8  # one request's verify rows
# fewer rows on the shard's variant (two warps a tile on 148 and 152 SMs)
N_SMALL = 12_000


def _not_ready() -> str:
    if not torch.cuda.is_available():
        return "needs an SM100/SM103 GPU"
    if torch.cuda.get_device_capability() not in lh.SUPPORTED_CAPABILITIES:
        return "needs an SM100/SM103 GPU"
    try:
        import cuda.bindings  # noqa: F401
    except ImportError:
        return "needs the CUDA Python bindings"
    return nvcc.why_not()


pytestmark = pytest.mark.skipif(bool(_not_ready()), reason=_not_ready() or "ready")


def _bound() -> float:
    from sgl_nvidia_mega.minimax_m31 import spec

    return spec.LM_HEAD_TOLERANCES["lm_head_mxfp8.logits.rel"]


def _weight(n, k=K, seed=0):
    g = torch.Generator(device="cuda").manual_seed(seed)
    return (torch.randn(n, k, generator=g, device="cuda") * 0.02).to(torch.bfloat16)


def _rows(rows, k=K, seed=1):
    g = torch.Generator(device="cuda").manual_seed(seed)
    return torch.randn(rows, k, generator=g, device="cuda").to(torch.bfloat16)


def _build(n, k=K):
    device = torch.device("cuda", torch.cuda.current_device())
    lh.precompile(n, k, device)
    lh.precompile(n, k, device, fp8=False)


def _rel(ours, theirs) -> float:
    return (
        (ours.float() - theirs.float()).abs().max() / theirs.float().abs().max()
    ).item()


@pytest.fixture(scope="module")
def shard():
    w = _weight(N)
    q = lh.quantize(w)
    _build(N)
    return w, q, lh.dequantize(q)


# ------------------------------------------------------------------- bit for bit


@pytest.mark.parametrize("rows", range(1, lh.MAX_ROWS + 1))
def test_mxfp8_equals_the_bf16_build_on_the_dequantized_weight(shard, rows):
    _, q, dq = shard
    x = _rows(rows)
    ours = lh.matmul(x, q)
    assert ours.shape == (rows, N) and ours.dtype == torch.bfloat16
    torch.testing.assert_close(ours, lh.matmul_bf16(x, dq), rtol=0, atol=0)
    # the kernel's sum against torch's on the same weight: summation order only
    torch.testing.assert_close(
        ours.float(), x.float() @ dq.float().T, rtol=2**-7, atol=2e-2
    )


@pytest.mark.parametrize("log2_scale", (-30, -9, 0, 7))
def test_every_code_dequantizes_exactly(log2_scale):
    """All 254 finite E4M3 codes (no NaN) in every lane position, one scale, on the
    shard's variant."""
    n = N_SMALL
    sms = torch.cuda.get_device_properties(0).multi_processor_count
    assert lh.variant(n, K, fp8=True, sms=sms) == lh.variant(N, K, fp8=True, sms=sms)
    _build(n)
    finite = torch.tensor(
        [c for c in range(256) if c & 0x7F != 0x7F], dtype=torch.uint8, device="cuda"
    )
    idx = torch.arange(n * K, device="cuda") * 7919 % finite.numel()
    codes = finite[idx].view(n, K).view(torch.float8_e4m3fn)
    scales = torch.full(
        (lh._pad128(n), K // lh.BLOCK),
        127 + log2_scale,
        dtype=torch.uint8,
        device="cuda",
    )
    q = lh.Mxfp8Weight(codes, lh.swizzle(scales))
    dq = lh.dequantize(q)
    assert torch.equal(dq.float(), codes.float() * 2.0**log2_scale), (
        "BF16 holds every code x scale"
    )
    for rows in (1, 6, 8):
        x = _rows(rows)
        torch.testing.assert_close(
            lh.matmul(x, q), lh.matmul_bf16(x, dq), rtol=0, atol=0
        )


def test_the_quantizer_on_gpu_equals_cpu_and_never_saturates():
    w = _weight(608)  # 96 rows past a multiple of 128, as the shard
    w[3, :32] = 0
    w[5, 64:96] *= 1e4
    gpu = lh.quantize(w)
    cpu = lh.quantize(w.cpu())
    assert torch.equal(gpu.codes.cpu().view(torch.uint8), cpu.codes.view(torch.uint8))
    assert torch.equal(gpu.scales.cpu(), cpu.scales)
    assert gpu.codes.float().abs().max().item() <= lh.E4M3_MAX
    assert gpu.scales.numel() == lh._pad128(608) * K // lh.BLOCK


# ------------------------------------------------------------ the shard's N as it is


def test_the_output_is_the_shard_and_no_padding_reaches_it(shard):
    """N 50016 is 3126 whole tiles and 96 rows past a multiple of 128: the codes are
    not padded and the output is [rows, N]."""
    _, q, _ = shard
    assert q.codes.shape == (N, K) and N % lh.TILE == 0 and N % 128 == 96
    x = _rows(ROWS)
    want = lh.matmul(x, q)
    assert want.shape == (ROWS, N) and not want.float().isnan().any()
    # a wider buffer of the caller's: the columns past N keep their sentinel
    wide = torch.full(
        (ROWS, N + 128), float("nan"), dtype=torch.bfloat16, device="cuda"
    )
    out = lh.matmul(x, q, out=wide[:, :N])
    assert out.data_ptr() == wide.data_ptr()
    assert torch.equal(wide[:, :N], want) and wide[:, N:].float().isnan().all()
    # the scales of the 32 rows the swizzle pads: whatever they hold
    padded = torch.zeros(
        (lh._pad128(N), K // lh.BLOCK), dtype=torch.bool, device="cuda"
    )
    padded[N:] = True
    where = lh.swizzle(padded)
    assert int(where.sum()) == 32 * K // lh.BLOCK
    poisoned = q.scales.clone()
    poisoned[where] = 0xFF
    torch.testing.assert_close(
        lh.matmul(x, lh.Mxfp8Weight(q.codes, poisoned)), want, rtol=0, atol=0
    )
    # NaN codes right after the last row: no tile reads past the shard
    codes = torch.full((N + lh.TILE, K), 0x7F, dtype=torch.uint8, device="cuda")
    codes[:N] = q.codes.view(torch.uint8)
    walled = lh.Mxfp8Weight(codes[:N].view(torch.float8_e4m3fn), q.scales)
    torch.testing.assert_close(lh.matmul(x, walled), want, rtol=0, atol=0)


def test_a_ragged_shard_masks_its_last_tile():
    """Any N: a last tile of 11 rows reads row N - 1 for the missing ones and stores
    none of them."""
    n = N_SMALL + 11
    w = _weight(n, seed=3)
    q = lh.quantize(w)
    dq = lh.dequantize(q)
    _build(n)
    x = _rows(ROWS, seed=4)
    wide = torch.full((ROWS, n + 64), float("nan"), dtype=torch.bfloat16, device="cuda")
    out = lh.matmul(x, q, out=wide[:, :n])
    torch.testing.assert_close(out, lh.matmul_bf16(x, dq), rtol=0, atol=0)
    assert not out.float().isnan().any() and wide[:, n:].float().isnan().all()
    assert _rel(out, torch.matmul(x, w.T)) <= _bound()


# ---------------------------------------------------- against the BF16 projection


@pytest.mark.parametrize("seed", (21, 22, 23))
def test_the_logits_are_within_the_declared_bound_of_the_bf16_projection(shard, seed):
    """One request's rows of Gaussian values (no row has a clear margin: the argmax
    flips are printed, not asserted)."""
    w, q, _ = shard
    x = _rows(ROWS, seed=seed)
    ours, theirs = lh.matmul(x, q), torch.matmul(x, w.T)
    rel = _rel(ours, theirs)
    flips = (ours.argmax(-1) != theirs.argmax(-1)).sum().item()
    print(
        f"LM head {ROWS} rows, seed {seed}: rel {rel:.3e} (bound {_bound():.1e}), "
        f"argmax flips {flips}"
    )
    assert rel <= _bound(), rel


def test_top1_agrees_on_rows_with_a_clear_margin(shard):
    """Rows that point at one token each (a scaled weight row plus noise), the shard's
    first and last rows among them: the BF16 projection's best logit leads the second
    by more than twice the bound of the peak, so the stream picks the same token."""
    w, q, _ = shard
    g = torch.Generator(device="cuda").manual_seed(7)
    tokens = torch.randint(0, N, (ROWS,), generator=g, device="cuda")
    tokens[0], tokens[-1] = 0, N - 1
    noise = torch.randn(ROWS, K, generator=g, device="cuda")
    x = (w[tokens].float() * 40 + noise).to(torch.bfloat16)
    theirs = torch.matmul(x, w.T).float()
    ours = lh.matmul(x, q).float()
    best = theirs.topk(2, dim=-1).values
    clear = (best[:, 0] - best[:, 1]) > 2 * _bound() * theirs.abs().max()
    assert clear.all(), "the test's rows must have a clear margin"
    assert torch.equal(theirs.argmax(-1), tokens)
    assert torch.equal(ours.argmax(-1), tokens)
    assert _rel(ours, theirs) <= _bound()


# ----------------------------------------------------------------- as it serves


def test_the_slot_through_sglangs_projection_eager_and_from_graph_replay(
    shard, monkeypatch
):
    """``logits.build``'s stand-in through SGLang's ``_compute_lm_head`` against
    SGLang's own head (its BF16 GEMM on the shard): the same dtype and shape, the
    logits within the declared bound, eagerly and from a CUDA graph replayed on new
    rows."""
    pytest.importorskip("sglang")
    from sglang.srt.layers.logits_processor import LogitsProcessor
    from sglang.srt.layers.quantization.unquant import UnquantizedEmbeddingMethod

    from sgl_nvidia_mega.minimax_m31 import logits

    w, _, _ = shard
    proc = LogitsProcessor.__new__(LogitsProcessor)
    torch.nn.Module.__init__(proc)
    proc.use_fp32_lm_head = proc.use_fp32_lm_head_output = False
    proc.rl_on_policy_target = None
    lm_head = SimpleNamespace(weight=w, quant_method=UnquantizedEmbeddingMethod())
    model = SimpleNamespace(lm_head=lm_head, logits_processor=proc)
    monkeypatch.setattr(logits, "_cpu_group", lambda: None)  # one rank
    head = logits.build(model, (8, 16), frozenset({"mega", "lm_head_mxfp8"}))
    assert isinstance(head, logits.Mxfp8LmHead)
    assert 0 < head.rel <= _bound()
    assert head.summary().endswith(f"[{N}, {K}] shard, 302.2 MiB of device memory"), (
        head.summary()
    )

    x = _rows(ROWS, seed=11)
    theirs = proc._compute_lm_head(x, lm_head)
    ours = proc._compute_lm_head(x, head)
    assert ours.dtype == theirs.dtype == torch.bfloat16
    assert ours.shape == theirs.shape == (ROWS, N)
    eager = _rel(ours, theirs)
    assert eager <= _bound(), eager

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = proc._compute_lm_head(x, head)
    assert head.quant_method.captured
    for seed in (12, 13):
        x.copy_(_rows(ROWS, seed=seed))
        captured.zero_()
        graph.replay()
        torch.testing.assert_close(
            captured, proc._compute_lm_head(x, head), rtol=0, atol=0
        )
        replayed = _rel(captured, proc._compute_lm_head(x, lm_head))
        assert replayed <= _bound(), replayed
    print(
        f"slot lm_head_mxfp8 through SGLang's projection: build check {head.rel:.3e}, "
        f"eager {eager:.3e}, graph replay {replayed:.3e} (bound {_bound():.1e})"
    )


def test_a_dependent_launch_in_a_graph_waits_for_the_kernel_that_writes_its_rows(shard):
    """The stream prefetches weights before its PDL wait and reads its rows after it:
    behind a kernel that writes them in the same graph, every replay projects that
    replay's rows; a strided input and a caller's output buffer."""
    _, q, _ = shard
    base = _rows(ROWS, k=K + 64)
    x = base[:, :K]  # row stride K + 64
    out = torch.empty(ROWS, N, dtype=torch.bfloat16, device="cuda")
    torch.testing.assert_close(
        lh.matmul(x, q, out=out), lh.matmul(x.contiguous(), q), rtol=0, atol=0
    )
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        for _ in range(3):
            rows = x * 2  # this replay's rows, written on the stream
            lh.matmul(rows, q, out=out)
    for seed in (30, 31, 32):
        base.copy_(_rows(ROWS, k=K + 64, seed=seed))
        out.zero_()
        graph.replay()
        torch.testing.assert_close(out, lh.matmul(x * 2, q), rtol=0, atol=0)


def test_a_launch_without_its_build_raises(shard):
    _, q, _ = shard
    x = _rows(ROWS)
    with patch.dict(lh._KERNELS, {}, clear=True):
        with pytest.raises(RuntimeError, match="a launch does not compile"):
            lh.matmul(x, q)
    lh.matmul(x, q)  # the built kernel is still there


# ---------------------------------------------------------------------- timing


def _us(fn, iters=20) -> float:
    """Per call from CUDA graph replay (the weight read from HBM: one copy > L2)."""
    for _ in range(3):
        fn()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        for _ in range(iters):
            fn()
    graph.replay()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(5):
        graph.replay()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) * 1e3 / (5 * iters)


def test_microseconds_per_launch(shard):
    w, q, _ = shard
    x = _rows(ROWS)
    out = torch.empty(ROWS, N, dtype=torch.bfloat16, device="cuda")
    print(
        f"LM head {ROWS} rows, N {N}, K {K} on {torch.cuda.get_device_name()}: MXFP8 "
        f"stream {_us(lambda: lh.matmul(x, q, out=out)):.1f} us, BF16 torch.matmul "
        f"{_us(lambda: torch.matmul(x, w.T)):.1f} us"
    )
