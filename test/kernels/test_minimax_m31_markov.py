# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""MiniMax M3.1's sharded DSpark Markov chain (sm103/draft/markov.py, slots markov and
draft_lm_head) on SM100-family GPUs; needs nvcc, Triton and, for the MXFP8 LM head,
FlashInfer.

At M3.1's shapes (vocabulary 200,064, Markov rank 256, 7 draft positions, BF16 base and
step logits) the chain draws SGLang's MarkovSampleStep tokens and keeps its step logits
bit for bit when the bias products sum exactly (ternary weights, |bias| <= 256), as
SGLang's draft runs the chain on the full vocabulary:

- one rank: every batch size SGLang captures a draft graph for (1-8, 10, 12, 14, 16) and
  9, the wide kernel's first, greedy, sampled and mixed temperatures, local-shard and
  full-vocab base logits, a greedy draft's chain without logit staging, three launches
  each; CUDA graphs of two batch sizes replayed alternately at new noise offsets with
  the noise forked onto its side stream and the counter advanced in the graph, for a
  pair of kernels with their own launch counters (direct + fold) and pairs that share
  them (fold + fold, wide + wide); rows at temperature 0 leave their step logits
  untouched;
- the selection keys of sampled rows, element by element: SGLang's partial kernel (the
  launch MarkovSampleStep makes for kept logits) against the chain's division and add
  over the plugin's noise kernel, bit for bit;
- Gaussian weights (bias sums that are not exact): every kept step logit within the
  declared bound (runtime.DRAFT_TOLERANCES: two BF16 ulps of |kept logit| + |base
  logit| past one ulp of the step's largest |bias|), and a token differs only at a tie
  that close;
- the MXFP8 LM-head copy (slot draft_lm_head): filled in row chunks, byte for byte the
  whole shard's quantize, and within its bound of SGLang's BF16 projection;
- four ranks (one process per GPU, the TP4 shard): every rank draws those tokens and
  holds the full-vocab step logits, at every batch size and across the graph pairs;
  then the chain's microseconds per launch from graph replay after an L2 flush, each
  timed graph behind a launch that aligns the ranks (printed with -s);
- fail-stop: when one rank never launches, the others fault within the deadline.

Run with ``python -m pytest test/kernels/test_minimax_m31_markov.py -s``.
"""

import os
import sys
import tempfile
import time
import traceback

import pytest
import torch
import torch.nn.functional as F

# the chain's noise kernel is Triton; a host without it (or a GPU) skips the module
pytest.importorskip("triton")

from sgl_nvidia_mega.minimax_m31.sm103.draft import markov  # noqa: E402

VOCAB = 200_064  # MiniMax M3.1-Flash
RANK = markov.RANK_DIM
STEPS = 7  # DSpark block 7
KEY = 0x1234_5678_9ABC_DEF
TEMPERATURE_SETS = (
    (0.0,) * 16,
    (1.0,) * 16,
    (0.7, 1.3, 1.0, 0.0, 0.5, 2.0, 0.9, 1.1, 0.0, 1.0, 0.6, 1.4, 0.0, 0.8, 1.2, 1.0),
)
# Far above any launch, so only a missing peer reaches it.
DEADLINE_S = 20.0
FAULT_DEADLINE_S = 2.0
WORLD = 4
TIMEOUT_S = 900


def _sm100(devices: int = 1) -> bool:
    return (
        torch.cuda.is_available()
        and torch.cuda.get_device_capability() in markov.SUPPORTED_CAPABILITIES
        and torch.cuda.device_count() >= devices
    )


def _ternary(g: torch.Generator, device, *shape: int) -> torch.Tensor:
    """Sparse {-1, 0, 1} BF16: |bias| <= 256, exact in BF16 in any summation order."""
    values = torch.randint(-1, 2, shape, generator=g)
    values *= torch.rand(shape, generator=g) < 0.25
    return values.to(device, torch.bfloat16)


def _reference(w1, w2, base, anchor, temperatures, offset, cache):
    """SGLang's MarkovSampleStep per step over the full vocabulary."""
    from sglang.kernels.ops.speculative.dspark.markov_sample import MarkovSampleStep

    rows = base.shape[0]
    prev, tokens = anchor.long(), []
    for step in range(base.shape[1]):
        bias = F.linear(w1[prev], w2)
        prev = MarkovSampleStep.execute(
            base_logits=base[:, step],
            bias=bias,
            temperatures=temperatures,
            key=KEY,
            offset=offset,
            step=step,
            step_dtype=torch.bfloat16,
            logits_out=None if cache is None else cache[:rows, step],
            has_sampled_rows=None if cache is not None else False,
        )
        tokens.append(prev)
    return torch.stack(tokens, 1)


def _assert_cache(cache, want_cache, temperatures, rows: int, what) -> None:
    """Rows that keep logits match; temperature-0 rows stay zero."""
    keeps = temperatures[:rows] != 0
    assert torch.equal(cache[:rows][keeps], want_cache[:rows][keeps]), what
    assert not cache[:rows][~keeps].any(), what


def _make(group, rank, world, w1, w2, max_bs, keeps_logits=True, deadline=DEADLINE_S):
    return markov.ShardedMarkov(
        embed_weight=w1,
        head_weight=w2,
        vocab_size=VOCAB,
        num_steps=STEPS,
        cache_dtype=torch.bfloat16,
        group=group,
        rank=rank,
        world=world,
        max_bs=max_bs,
        deadline_s=deadline,
        keeps_logits=keeps_logits,
    )


def _anchors(g, device, rows: int) -> torch.Tensor:
    """Row r's anchor at [r * STEPS] of int64 input ids, as the draft graph holds them."""
    ids = torch.randint(0, VOCAB, (rows, STEPS), generator=g).to(device)
    return ids.view(-1).view(rows, STEPS)[:, 0]


def _check_rows(device, chain, greedy_chain, w1, w2, rank, world, rows, g, what):
    """Every temperature set, kept and unkept logits, local and full base logits."""
    shard = slice(rank * VOCAB // world, (rank + 1) * VOCAB // world)
    offset = torch.tensor([3], dtype=torch.int64, device=device)
    base = (torch.randn(rows, STEPS, VOCAB, generator=g) * 3).to(device, torch.bfloat16)
    anchor = _anchors(g, device, rows)
    max_bs = chain.max_rows
    for temps in TEMPERATURE_SETS:
        temperatures = torch.tensor(temps, device=device)
        for keep in (True, False):  # sampled draft vs greedy draft (no logits)
            want_cache = (
                torch.zeros(max_bs, STEPS, VOCAB, device=device, dtype=torch.bfloat16)
                if keep
                else None
            )
            want = _reference(
                w1, w2, base, anchor, temperatures[:rows], offset, want_cache
            )
            for sampler in (chain,) if keep else (chain, greedy_chain):
                for local in (True, False):
                    inputs = base[:, :, shard].contiguous() if local else base
                    for launch in range(3):
                        cache = (
                            torch.zeros(
                                max_bs,
                                STEPS,
                                VOCAB,
                                device=device,
                                dtype=torch.bfloat16,
                            )
                            if keep
                            else None
                        )
                        tokens = torch.full(
                            (max_bs, STEPS), -7, dtype=torch.int64, device=device
                        )
                        if keep:
                            sampler.launch_noise(
                                rows=rows,
                                key=KEY,
                                offset=offset,
                                temperatures=temperatures,
                            )
                        sampler.sample(
                            base_logits=inputs,
                            anchor_tokens=anchor,
                            temperatures=temperatures,
                            logits_cache=cache,
                            draft_tokens=tokens[:rows],
                        )
                        torch.cuda.synchronize()
                        case = (what, rows, temps[:rows], keep, local, launch)
                        assert torch.equal(tokens[:rows], want), case
                        assert (tokens[rows:] == -7).all(), case
                        if keep:
                            _assert_cache(cache, want_cache, temperatures, rows, case)


def _check_forked_graph(device, chain, w1, w2, rank, world, rows_list, g):
    """CUDA graphs of row counts ``rows_list``, each with its noise forked onto the side
    stream and SGLang's noise counter advanced by the chain kernel (advance_offset, as
    served), replayed in that order, repeatedly, from new offsets: a list of row counts
    one kernel serves (2-8 the fold, 9-16 the wide kernel) replays graphs that share its
    launch counters and records with fewer rows between more (8, 2, 8, ...)."""
    shard = slice(rank * VOCAB // world, (rank + 1) * VOCAB // world)
    temperatures = torch.tensor(TEMPERATURE_SETS[2], device=device)
    offset = torch.tensor([3], dtype=torch.int64, device=device)
    max_bs = chain.max_rows
    cases = []
    for rows in rows_list:
        base = (torch.randn(rows, STEPS, VOCAB, generator=g) * 3).to(
            device, torch.bfloat16
        )
        anchor = _anchors(g, device, rows)
        cache = torch.zeros(max_bs, STEPS, VOCAB, device=device, dtype=torch.bfloat16)
        tokens = torch.full((max_bs, STEPS), -7, dtype=torch.int64, device=device)
        local = base[:, :, shard].contiguous()

        def forked_sample(rows=rows, local=local, anchor=anchor, cache=cache, t=tokens):
            chain.fork_noise(
                rows=rows, key=KEY, offset=offset, temperatures=temperatures
            )
            chain.join_noise()
            chain.sample(
                base_logits=local,
                anchor_tokens=anchor,
                temperatures=temperatures,
                logits_cache=cache,
                draft_tokens=t[:rows],
                advance_offset=offset,
            )

        side = torch.cuda.Stream(device)
        side.wait_stream(torch.cuda.current_stream(device))
        with torch.cuda.stream(side):
            forked_sample()
        torch.cuda.current_stream(device).wait_stream(side)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            forked_sample()
        # the graph reads these buffers: they must outlive it (``local`` is the
        # rebound closure's otherwise, and its memory is reused once that is gone)
        cases.append((rows, base, anchor, cache, tokens, local, graph))
    for step_offset in (5, 6, 1 << 33, 9):
        offset.fill_(step_offset)
        for rows, base, anchor, cache, tokens, local, graph in cases:
            # the offset this replay draws its noise at, before the kernel advances it
            drawn = offset.clone()
            graph.replay()
            want_cache = torch.zeros_like(cache)
            want = _reference(
                w1, w2, base, anchor, temperatures[:rows], drawn, want_cache
            )
            torch.cuda.synchronize()
            # the kernel advanced SGLang's counter by one block
            advanced = int(offset) == int(drawn) + 1
            ok = torch.equal(tokens[:rows], want)
            if not _all_ranks(ok and advanced):
                # every rank re-runs the replay's inputs eagerly, then explains
                why = _explain(
                    chain, base, local, anchor, temperatures, drawn,
                    tokens, cache, want, want_cache, rows, device,
                )  # fmt: skip
                state = "matched" if ok else "mismatched"
                raise AssertionError(
                    f"rows {rows} offset {int(drawn)} -> {int(offset)} (this rank "
                    f"{state}): {why}"
                )
            _assert_cache(cache, want_cache, temperatures, rows, (rows, int(drawn)))


def _all_ranks(ok: bool) -> bool:
    import torch.distributed as dist

    if not dist.is_initialized():
        return ok
    flag = torch.tensor([int(ok)])
    dist.all_reduce(flag, op=dist.ReduceOp.MIN)
    return bool(flag.item())


def _scores(values, row: int, step: int, offset: int, temperature: float):
    """SGLang's select keys of one (row, step) over the full vocab (markov_sample.py's
    torch path): value / T - log(-log1p(-u)) from its Philox draw."""
    from sglang.kernels.ops.speculative.dspark import markov_sample as ms

    values = values.float()
    width = values.shape[-1]
    quads = torch.arange((width + 3) // 4, device=values.device, dtype=torch.int64)
    words = ms._philox(
        KEY,
        quads[None, :],
        torch.full((1, 1), row, device=values.device, dtype=torch.int64),
        torch.full((1, 1), step, device=values.device, dtype=torch.int64),
        torch.full(
            (1, 1), offset & 0xFFFFFFFF, device=values.device, dtype=torch.int64
        ),
    )
    words = torch.broadcast_tensors(*words)
    bits = torch.stack(words, dim=-1).reshape(-1)[:width]
    uniform = (bits.float() + 0.5) * ms._UNIFORM_SCALE.value
    return values / temperature - torch.log(-torch.log1p(-uniform))


def _explain(chain, base, local, anchor, temperatures, offset, tokens, cache, want,
             want_cache, rows, device) -> str:  # fmt: skip
    eager = torch.full_like(tokens, -7)
    eager_cache = torch.zeros_like(cache)
    chain.launch_noise(rows=rows, key=KEY, offset=offset, temperatures=temperatures)
    chain.sample(
        base_logits=local,
        anchor_tokens=anchor,
        temperatures=temperatures,
        logits_cache=eager_cache,
        draft_tokens=eager[:rows],
    )
    torch.cuda.synchronize()
    got = tokens[:rows]
    lines = [
        f"graph {got.tolist()}",
        f"eager {eager[:rows].tolist()}",
        f"reference {want.tolist()}",
    ]
    bad = (got != want).nonzero().tolist()
    if bad:
        r, step = bad[0]
        same_logits = torch.equal(cache[r, step], want_cache[r, step])
        lines.append(
            f"first mismatch row {r} step {step}: step logits equal {same_logits}"
        )
        t = float(temperatures[r])
        if t != 0.0:
            scores = _scores(want_cache[r, step], r, step, int(offset), t)
            for name, token in (
                ("graph", int(got[r, step])),
                ("reference", int(want[r, step])),
            ):
                lines.append(
                    f"  {name} token {token}: logit {float(want_cache[r, step, token])!r} "
                    f"score {float(scores[token])!r}"
                )
            top = scores.topk(3)
            lines.append(
                f"  top-3 scores {top.values.tolist()} at {top.indices.tolist()}"
            )
    return "; ".join(lines)


# The batch sizes SGLang's default decode graphs take up to 16 requests (1-8, 10, 12,
# 14, 16) and 9, the wide kernel's first
ROW_COUNTS = (1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 14, 16)
# Graph pairs: the direct exchange and the fold (their own counters), two fold row
# counts and two wide ones (one kernel's counters and records, fewer rows between more)
GRAPH_PAIRS = ((2, 1), (8, 2), (16, 10))
# the row counts timed
TIMED_ROWS = (1, 2, 4, 8, 12, 16)


def _row_counts(chain, counts=ROW_COUNTS) -> list[int]:
    return [r for r in counts if r <= chain.max_rows]


def _pairs(chain) -> list[list[int]]:
    return [list(p) for p in GRAPH_PAIRS if max(p) <= chain.max_rows]


# ------------------------------------------------------------------------------ 1 GPU


@pytest.mark.skipif(not _sm100(), reason="needs an SM100/SM103 GPU")
@pytest.mark.parametrize("rows", [7, 56, 112])
def test_the_mxfp8_draft_lm_head_is_the_bf16_projection_within_its_bound(rows):
    """Slot draft_lm_head: the MXFP8 copy of the TP4 shard (N 50016, K 6144) against
    SGLang's BF16 projection, at 1, 8 and 16 requests' rows."""
    from types import SimpleNamespace

    from sgl_nvidia_mega.minimax_m31 import runtime, sampler

    n, k = VOCAB // WORLD, 6144
    g = torch.Generator(device="cuda").manual_seed(rows)
    weight = (torch.randn(n, k, device="cuda", generator=g) * 0.02).bfloat16()
    head = sampler.DraftLmHead(rows=n, cols=k, device="cuda", mxfp8=True)
    head.attach(SimpleNamespace(weight=weight))
    x = torch.randn(rows, k, device="cuda", generator=g).bfloat16()
    want = torch.matmul(x, weight.T).float()
    got = head(x)
    assert got.dtype == torch.bfloat16 and got.shape == (rows, n)
    rel = ((got.float() - want).abs().max() / want.abs().max()).item()
    bound = runtime.DRAFT_TOLERANCES["draft_lm_head"]["draft_lm_head.logits.rel"]
    assert rel <= bound, rel
    flips = (got.argmax(-1) != want.argmax(-1)).sum().item()
    print(
        f"draft LM head rows {rows}: rel {rel:.2e}, argmax flips {flips}; MXFP8 "
        f"{_us(lambda: head(x)):.1f} us, BF16 "
        f"{_us(lambda: torch.matmul(x, weight.T)):.1f} us"
    )


@pytest.mark.skipif(not _sm100(), reason="needs an SM100/SM103 GPU")
def test_the_mxfp8_copy_fills_in_chunks_as_one_quantize():
    """The draft spec's build allocates the copy; attach fills it from the attached
    shard in row chunks: byte for byte one quantize of the whole shard (MXFP8 scales
    are per 32 of K within a row) and its 128x4 scale layout."""
    from types import SimpleNamespace

    from flashinfer import block_scale_interleave
    from sglang.srt.layers.quantization.fp8_utils import flashinfer_mxfp8_quantize

    from sgl_nvidia_mega.minimax_m31 import sampler

    n, k = VOCAB // WORLD, 6144
    g = torch.Generator(device="cuda").manual_seed(3)
    weight = (torch.randn(n, k, device="cuda", generator=g) * 0.02).bfloat16()
    head = sampler.DraftLmHead(rows=n, cols=k, device="cuda", mxfp8=True)
    head.attach(SimpleNamespace(weight=weight))
    quantized, scale = flashinfer_mxfp8_quantize(weight, is_sf_swizzled_layout=False)
    want_scale = block_scale_interleave(
        scale.view(torch.uint8).reshape(n, k // 32).contiguous()
    ).contiguous()
    assert torch.equal(
        head.weight.view(torch.uint8), quantized.view(n, k).view(torch.uint8)
    )
    assert torch.equal(
        head.weight_scale.view(torch.uint8).view(-1),
        want_scale.view(torch.uint8).view(-1),
    )
    print(
        f"MXFP8 copy: {head.device_bytes() / 2**20:.1f} MiB, filled in "
        f"{-(-n // head.FILL_ROWS)} chunks, bytes equal one quantize"
    )


def _keys_kernel():
    """A test kernel compiled into markov.cu's own translation unit: the chain's
    selection key of a sampled element, AddRn(DivFull(value, T), noise)."""
    from sgl_nvidia_mega.common import jit
    from sgl_nvidia_mega.minimax_m31.sm103 import nvcc

    source = (
        markov.SOURCE.read_text()
        + r"""
extern "C" __global__ void m31_markov_test_keys(const float* value, const float* noise,
    const float* temperature, float* out, int rows, int vocab, int steps, int step) {
  const long long i = static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= static_cast<long long>(rows) * vocab) return;
  const int r = static_cast<int>(i / vocab);
  const long long col = i % vocab;
  out[i] = AddRn(DivFull(value[i], temperature[r]), noise[(static_cast<long long>(r) * steps + step) * vocab + col]);
}
"""
    )
    options = markov.NVCC_OPTIONS + markov.defines(
        steps=STEPS,
        base_dtype=torch.bfloat16,
        cache_dtype=torch.bfloat16,
        direct=False,
    )
    cubin = nvcc.compile_cubin(
        "minimax_m31_markov_test_keys",
        source,
        (),
        nvcc.device_arch(torch.device("cuda", torch.cuda.current_device())),
        options,
    )
    return jit.load(cubin, "m31_markov_test_keys", (256, 1, 1), 0)


@pytest.mark.skipif(not _sm100(), reason="needs an SM100/SM103 GPU")
def test_the_sampled_selection_keys_are_sglangs_bit_for_bit():
    """Token tests cannot see a last-ulp key difference (it decides only an exact tie):
    here every element's key. SGLang's: its partial kernel as MarkovSampleStep launches
    it for kept logits (no bias, store, select, sample; 1024-element blocks), one
    unmasked element per block at a time (the rest -inf), so each block's maximum is
    that element's key. The chain's: value / T then + noise in its own division and add,
    on the noise the plugin's kernel draws (SGLang's Philox words, -log of
    -log1p(-uniform))."""
    import ctypes

    from sglang.kernels.ops.speculative.dspark.markov_sample import _launch_partial

    from sgl_nvidia_mega.common import jit

    device = torch.device("cuda")
    g = torch.Generator().manual_seed(7)
    rows = 4
    temperatures = torch.tensor([0.7, 1.0, 1.3, 2.0], device=device)
    w1, w2 = _ternary(g, device, VOCAB, RANK), _ternary(g, device, VOCAB, RANK)
    chain = _make(None, 0, 1, w1, w2, max_bs=rows)
    values = (torch.randn(rows, VOCAB, generator=g) * 3).to(device, torch.bfloat16)
    offset = torch.tensor([12345], dtype=torch.int64, device=device)
    chain.launch_noise(rows=rows, key=KEY, offset=offset, temperatures=temperatures)
    kernel = _keys_kernel()
    value_f32 = values.float().contiguous()  # alive until the launches have run
    cols = torch.arange(VOCAB, device=device)
    block = 1024
    scratch = torch.empty(rows, VOCAB, dtype=torch.bfloat16, device=device)
    stream = torch.cuda.current_stream().cuda_stream
    differ = 0
    for step in (0, 3, STEPS - 1):
        mine = torch.empty(rows, VOCAB, dtype=torch.float32, device=device)
        jit.launch(
            kernel,
            (-(-rows * VOCAB // 256), 1, 1),
            [
                jit.pointer(value_f32),
                jit.pointer(chain.noise),
                jit.pointer(temperatures),
                jit.pointer(mine),
                ctypes.c_int32(rows),
                ctypes.c_int32(VOCAB),
                ctypes.c_int32(STEPS),
                ctypes.c_int32(step),
            ],
            stream,
        )
        theirs = torch.empty(rows, VOCAB, dtype=torch.float32, device=device)
        for lane in range(block):
            masked = torch.where(cols % block == lane, values, -torch.inf)
            partial = _launch_partial(
                masked,
                None,
                width=VOCAB,
                store=scratch,
                temperatures=temperatures,
                key=KEY,
                offset=offset,
                step=step,
                step_dtype=torch.bfloat16,
                has_sampled_rows=None,
            ).view(rows, -1, 2)
            at = torch.arange(lane, VOCAB, block, device=device)
            parts = at // block
            theirs[:, at] = partial[:, parts, 0]
            index = partial[:, parts, 1].contiguous().view(torch.int32)
            assert torch.equal(index, at.to(torch.int32).expand(rows, -1)), (step, lane)
        torch.cuda.synchronize()
        wrong = mine.view(torch.int32) != theirs.view(torch.int32)
        bad = int(wrong.sum())
        differ += bad
        apart = (mine.view(torch.int32).long() - theirs.view(torch.int32).long()).abs()
        print(
            f"selection keys at step {step}: {bad} of {rows * VOCAB} differ"
            + (f", at most {int(apart[wrong].max())} FP32 ulps apart" if bad else "")
        )
        for r, col in wrong.nonzero()[:4].tolist():
            print(
                f"  row {r} col {col}: value {float(value_f32[r, col])!r} T "
                f"{float(temperatures[r])!r} noise "
                f"{float(chain.noise[r * STEPS + step, col])!r}: chain "
                f"{float(mine[r, col])!r}, SGLang {float(theirs[r, col])!r}"
            )
    assert differ == 0, f"{differ} selection keys differ from SGLang's"


def _first_mismatch(tokens, want) -> torch.Tensor:
    """Per row, the first step whose token differs (STEPS where none does)."""
    differs = tokens != want
    steps = torch.arange(STEPS, device=tokens.device).expand_as(differs)
    return torch.where(differs, steps, STEPS).amin(dim=1)


@pytest.mark.skipif(not _sm100(), reason="needs an SM100/SM103 GPU")
def test_gaussian_weights_keep_logits_within_the_declared_bound():
    """Bias sums that are not exact (Gaussian weights, the served case): each kept step
    logit within the declared bound of SGLang's (runtime.DRAFT_TOLERANCES,
    sampler.bound_ulps), and a draft token differs only at a tie that close: the
    two candidates' reference keys at most their bounds (over T) apart. Steps after a
    row's first differing token are not compared (they condition on another token)."""
    from sgl_nvidia_mega.minimax_m31 import runtime, sampler

    bound = runtime.DRAFT_TOLERANCES["markov"]["draft_markov.step_logits.bf16_ulps"]
    device = torch.device("cuda")
    g = torch.Generator().manual_seed(17)
    w1 = torch.randn(VOCAB, RANK, generator=g).to(device, torch.bfloat16)
    w2 = (torch.randn(VOCAB, RANK, generator=g) * 0.1).to(device, torch.bfloat16)
    chain = _make(None, 0, 1, w1, w2, max_bs=16)
    worst, differing, compared, ties = 0.0, 0, 0, 0
    for rows in (1, 2, 8, 9, 16):
        for temps in TEMPERATURE_SETS[1:]:
            temperatures = torch.tensor(temps, device=device)
            offset = torch.tensor([rows], dtype=torch.int64, device=device)
            base = (torch.randn(rows, STEPS, VOCAB, generator=g) * 3).to(
                device, torch.bfloat16
            )
            anchor = _anchors(g, device, rows)
            want_cache = torch.zeros(
                16, STEPS, VOCAB, device=device, dtype=torch.bfloat16
            )
            want = _reference(
                w1, w2, base, anchor, temperatures[:rows], offset, want_cache
            )
            cache = torch.zeros_like(want_cache)
            tokens = torch.full((16, STEPS), -7, dtype=torch.int64, device=device)
            chain.launch_noise(
                rows=rows, key=KEY, offset=offset, temperatures=temperatures
            )
            chain.sample(
                base_logits=base,
                anchor_tokens=anchor,
                temperatures=temperatures,
                logits_cache=cache,
                draft_tokens=tokens[:rows],
            )
            torch.cuda.synchronize()
            first = _first_mismatch(tokens[:rows], want)
            for r in range(rows):
                t = float(temperatures[r])
                last = int(first[r])  # compare steps 0..last (inclusive when < STEPS)
                upto = min(last + 1, STEPS)
                ref = want_cache[r, :upto].float()
                below = base[r, :upto].float()
                magnitude = ref.abs() + below.abs()
                # one ulp of each step's largest |bias|, which the bound allows besides
                floor = sampler.bf16_ulp((ref - below).abs().amax(dim=-1))
                if t != 0:  # rows at temperature 0 keep no logits on the chain
                    got = cache[r, :upto].float()
                    ulps = sampler.bound_ulps((got - ref).abs(), ref, below)
                    worst = max(worst, ulps.max().item())
                    differing += int((got != ref).sum())
                    compared += ref.numel()
                if last < STEPS:
                    # a differing token: a tie within both candidates' bounds, in
                    # SGLang's keys (the logits at temperature 0)
                    a, b = int(tokens[r, last]), int(want[r, last])
                    scores = (
                        _scores(want_cache[r, last], r, last, int(offset), t)
                        if t != 0
                        else want_cache[r, last].float()
                    )
                    room = (
                        bound
                        * float(
                            sampler.bf16_ulp(magnitude[last, a])
                            + sampler.bf16_ulp(magnitude[last, b])
                        )
                        + 2 * float(floor[last])
                    ) / (t or 1.0)
                    gap = abs(float(scores[b] - scores[a]))
                    assert gap <= room * (1 + 1e-6), (rows, r, t, last, a, b, gap, room)
                    ties += 1
    print(
        f"Gaussian weights: {differing} of {compared} kept logits differ from SGLang's, "
        f"max {worst:.3f} BF16 ulps of |logit| + |base| past one ulp of the step's "
        f"largest |bias| (bound {bound}); {ties} tokens differ, each at a tie within the "
        "bound"
    )
    assert worst <= bound, worst


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


@pytest.mark.skipif(not _sm100(), reason="needs an SM100/SM103 GPU")
def test_one_rank_matches_markov_sample_step():
    device = torch.device("cuda")
    g = torch.Generator().manual_seed(1)
    w1, w2 = _ternary(g, device, VOCAB, RANK), _ternary(g, device, VOCAB, RANK)
    chain = _make(None, 0, 1, w1, w2, max_bs=16)
    # A greedy draft's chain has no logit staging; it serves greedy blocks only.
    greedy_chain = _make(None, 0, 1, w1, w2, max_bs=16, keeps_logits=False)
    print(
        f"one rank: ctas {chain.ctas}, narrow rows {chain.narrow_rows}, wide rows "
        f"{chain.wide_rows}"
    )
    for rows in _row_counts(chain):
        _check_rows(device, chain, greedy_chain, w1, w2, 0, 1, rows, g, "one rank")
    for pair in _pairs(chain):
        _check_forked_graph(device, chain, w1, w2, 0, 1, pair, g)
    # SGLang's noise counter advances once per block, at the kernel's end
    for rows in (1, 12):
        offset = torch.tensor([41], dtype=torch.int64, device=device)
        base = (torch.randn(rows, STEPS, VOCAB, generator=g) * 3).to(
            device, torch.bfloat16
        )
        anchor = _anchors(g, device, rows)
        temperatures = torch.tensor(TEMPERATURE_SETS[2], device=device)
        cache = torch.zeros(
            chain.max_rows, STEPS, VOCAB, device=device, dtype=torch.bfloat16
        )
        tokens = torch.full(
            (chain.max_rows, STEPS), -7, dtype=torch.int64, device=device
        )
        want_cache = torch.zeros_like(cache)
        want = _reference(w1, w2, base, anchor, temperatures[:rows], offset, want_cache)
        chain.launch_noise(rows=rows, key=KEY, offset=offset, temperatures=temperatures)
        chain.sample(
            base_logits=base,
            anchor_tokens=anchor,
            temperatures=temperatures,
            logits_cache=cache,
            draft_tokens=tokens[:rows],
            advance_offset=offset,
        )
        torch.cuda.synchronize()
        assert int(offset) == 42, int(offset)
        assert torch.equal(tokens[:rows], want), rows
        _assert_cache(cache, want_cache, temperatures, rows, rows)


# ----------------------------------------------------------------------------- 4 GPUs


def _time(chain, rows: int, sampled: bool, device, flush, inner=10, replays=100):
    """Microseconds per launch: a graph of ``inner`` launches, each after an L2 flush,
    minus a graph of the flushes alone (rank 0's CUDA events). The ranks barrier, then
    replay one untimed chain launch before each timed graph: it ends when the last rank
    has launched it, so the timed launches start aligned (behind a gloo barrier alone
    the ranks' first launches start up to hundreds of microseconds apart, which the
    first timed launch absorbs: its median then sits far above its 10th percentile)."""
    import torch.distributed as dist

    vocab_local, vocab = chain.vocab_local, chain.vocab_size
    temperatures = torch.full((16,), 1.0 if sampled else 0.0, device=device)
    offset = torch.tensor([7], dtype=torch.int64, device=device)
    cache = torch.zeros(16, STEPS, vocab, device=device, dtype=torch.bfloat16)
    base = torch.randn(rows, STEPS, vocab_local, device=device).to(torch.bfloat16) * 3
    ids = torch.randint(0, vocab, (rows * STEPS,), device=device)
    anchor = ids.view(rows, STEPS)[:, 0]
    tokens = torch.zeros(16, STEPS, dtype=torch.int64, device=device)
    if sampled:
        chain.launch_noise(rows=rows, key=KEY, offset=offset, temperatures=temperatures)

    def body(flush_only: bool, launches: int = inner):
        for _ in range(launches):
            flush.zero_()
            if not flush_only:
                chain.sample(
                    base_logits=base,
                    anchor_tokens=anchor,
                    temperatures=temperatures,
                    logits_cache=cache if sampled else None,
                    draft_tokens=tokens[:rows],
                )

    graphs = {}
    for flush_only in (False, True):
        body(flush_only)
        torch.cuda.synchronize(device)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            body(flush_only)
        graphs[flush_only] = graph
    align = torch.cuda.CUDAGraph()
    with torch.cuda.graph(align):
        body(False, 1)
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    times = {False: [], True: []}
    for replay in range(10 + replays):
        for flush_only, graph in graphs.items():
            torch.cuda.synchronize(device)
            dist.barrier()
            align.replay()
            start.record()
            graph.replay()
            end.record()
            torch.cuda.synchronize(device)
            if replay >= 10:
                times[flush_only].append(start.elapsed_time(end) * 1e3 / inner)
    per_launch = sorted(t - f for t, f in zip(times[False], times[True]))
    return per_launch[len(per_launch) // 2], per_launch[len(per_launch) // 10]


def _noise_time(chain, rows: int, device, inner=10, replays=50):
    offset = torch.tensor([7], dtype=torch.int64, device=device)
    temperatures = torch.ones(16, device=device)  # every row sampled: all the noise

    def body():
        for _ in range(inner):
            chain.launch_noise(
                rows=rows, key=KEY, offset=offset, temperatures=temperatures
            )

    body()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        body()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    graph.replay()
    start.record()
    for _ in range(replays):
        graph.replay()
    end.record()
    torch.cuda.synchronize(device)
    return start.elapsed_time(end) * 1e3 / (inner * replays)


def _init(rank: int, init_method: str):
    import torch.distributed as dist

    torch.cuda.set_device(rank)
    dist.init_process_group(
        backend="gloo", init_method=init_method, rank=rank, world_size=WORLD
    )
    return torch.device("cuda", rank), dist.group.WORLD


def _worker(rank: int, init_method: str, out_dir: str) -> None:
    status = "FAILED"
    try:
        device, group = _init(rank, init_method)
        g = torch.Generator().manual_seed(
            11
        )  # the same weights and inputs on every rank
        w1, w2 = _ternary(g, device, VOCAB, RANK), _ternary(g, device, VOCAB, RANK)
        chain = _make(group, rank, WORLD, w1, w2, max_bs=16)
        greedy_chain = _make(group, rank, WORLD, w1, w2, max_bs=16, keeps_logits=False)
        lines = [
            f"{torch.cuda.get_device_name(device)}: {torch.cuda.get_device_properties(device).multi_processor_count}"
            f" SMs; ctas {chain.ctas}, narrow rows {chain.narrow_rows}, wide rows "
            f"{chain.wide_rows}, max rows {chain.max_rows}"
        ]
        for rows in _row_counts(chain):
            _check_rows(
                device, chain, greedy_chain, w1, w2, rank, WORLD, rows, g, "TP4"
            )
        for pair in _pairs(chain) + [[chain.max_rows, 1]]:
            _check_forked_graph(device, chain, w1, w2, rank, WORLD, pair, g)
        lines.append(
            f"TP4 tokens and step logits at rows {_row_counts(chain)} and graph pairs "
            f"{_pairs(chain) + [[chain.max_rows, 1]]}: bitwise SGLang's MarkovSampleStep"
        )
        flush = torch.empty(
            2 * torch.cuda.get_device_properties(device).L2_cache_size,
            dtype=torch.uint8,
            device=device,
        )
        for rows in _row_counts(chain, TIMED_ROWS):
            for sampled in (False, True):
                median, p10 = _time(chain, rows, sampled, device, flush)
                lines.append(
                    f"chain rows={rows} {'sampled' if sampled else 'greedy'}: "
                    f"median {median:.2f} us, p10 {p10:.2f} us (cold L2)"
                )
            lines.append(
                f"noise rows={rows}: {_noise_time(chain, rows, device):.2f} us"
            )
        # the cost of the tiles beyond one per warp: a shard of exactly one tile per
        # warp (152 x 16 tiles: vocabulary 155,648), the same tables cut short
        one = 16 * 16 * chain.ctas * WORLD
        short = markov.ShardedMarkov(
            embed_weight=w1[:one],
            head_weight=w2[:one],
            vocab_size=one,
            num_steps=STEPS,
            cache_dtype=torch.bfloat16,
            group=group,
            rank=rank,
            world=WORLD,
            max_bs=16,
            deadline_s=DEADLINE_S,
            keeps_logits=True,
        )
        for rows in (1, 2, 8, 16):
            median, p10 = _time(short, rows, False, device, flush)
            lines.append(
                f"one tile per warp (vocabulary {one}): chain rows={rows} greedy: "
                f"median {median:.2f} us, p10 {p10:.2f} us (cold L2)"
            )
        status = "OK\n" + "\n".join(lines)
    except Exception:
        status = "FAILED\n" + traceback.format_exc()
    finally:
        with open(os.path.join(out_dir, f"rank{rank}.txt"), "w") as f:
            f.write(status)


def _spawn(worker, args_tail, timeout_s=TIMEOUT_S):
    import torch.multiprocessing as mp

    with tempfile.TemporaryDirectory() as tmp:
        context = mp.start_processes(
            worker,
            args=(f"file://{tmp}/store", tmp, *args_tail),
            nprocs=WORLD,
            join=False,
            start_method="spawn",
        )
        deadline = time.monotonic() + timeout_s
        while not context.join(timeout=10):
            if time.monotonic() > deadline:
                for process in context.processes:
                    if process.is_alive():
                        process.terminate()
                pytest.fail(f"workers did not finish in {timeout_s} s")
        results = {}
        for rank in range(WORLD):
            path = os.path.join(tmp, f"rank{rank}.txt")
            results[rank] = open(path).read() if os.path.exists(path) else "MISSING"
    return results


@pytest.mark.skipif(not _sm100(WORLD), reason="needs 4 SM100/SM103 GPUs")
def test_four_ranks_match_markov_sample_step():
    results = _spawn(_worker, ())
    for rank, text in results.items():
        assert text.startswith("OK"), f"rank {rank}:\n{text}"
    print("\n" + results[0].split("\n", 1)[1])


def _fault_worker(rank: int, init_method: str, out_dir: str) -> None:
    status = "FAILED"
    try:
        import torch.distributed as dist

        device, group = _init(rank, init_method)
        g = torch.Generator().manual_seed(5)
        w1, w2 = _ternary(g, device, VOCAB, RANK), _ternary(g, device, VOCAB, RANK)
        chain = _make(group, rank, WORLD, w1, w2, max_bs=8, deadline=FAULT_DEADLINE_S)
        base = torch.randn(2, STEPS, chain.vocab_local, device=device).bfloat16()
        anchor = torch.zeros(2 * STEPS, dtype=torch.int64, device=device)[::STEPS]
        temperatures = torch.zeros(8, device=device)
        tokens = torch.zeros(2, STEPS, dtype=torch.int64, device=device)

        def run(rows):
            chain.sample(
                base_logits=base[:rows],
                anchor_tokens=anchor[:rows],
                temperatures=temperatures,
                logits_cache=None,
                draft_tokens=tokens[:rows],
            )

        for rows in (1, 2):  # the direct exchange and the fold: healthy on every rank
            run(rows)
        torch.cuda.synchronize()
        dist.barrier()
        if rank == WORLD - 1:
            status = "SKIPPED"
            time.sleep(4 * FAULT_DEADLINE_S + 20)  # never joins the next launch
        else:
            t0 = time.time()
            try:
                run(1)
                torch.cuda.synchronize()
                status = "NO-FAULT"
            except Exception as e:
                status = f"FAULTED after {time.time() - t0:.1f} s: {type(e).__name__}"
    except Exception:
        status = "FAILED\n" + traceback.format_exc()
    finally:
        with open(os.path.join(out_dir, f"rank{rank}.txt"), "w") as f:
            f.write(status)
        os._exit(0)  # a faulted CUDA context must not be torn down


@pytest.mark.skipif(not _sm100(WORLD), reason="needs 4 SM100/SM103 GPUs")
def test_a_missing_rank_faults_its_peers_in_time():
    results = _spawn(_fault_worker, (), timeout_s=300)
    assert results[WORLD - 1] == "SKIPPED", results
    for rank in range(WORLD - 1):
        text = results[rank]
        assert text.startswith("FAULTED after"), f"rank {rank}: {text}"
        assert float(text.split("after ")[1].split(" s")[0]) < 4 * FAULT_DEADLINE_S + 10
    print(results)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q", "-s", "-p", "no:cacheprovider"]))
