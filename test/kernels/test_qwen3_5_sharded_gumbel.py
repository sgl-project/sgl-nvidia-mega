# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Direct tests of the Qwen3.5 sharded-Gumbel numerical slot."""

from __future__ import annotations

from contextlib import nullcontext
from types import SimpleNamespace

import pytest
import torch
import triton
import triton.language as tl
from sglang.kernels.plugin_kernels.nvidia import spec

from sgl_nvidia_mega.qwen3_5.runtime import hooks
from sgl_nvidia_mega.qwen3_5.sm103.sharded_gumbel import (
    build,
    gumbel_shard_keys,
    key_splits,
)

_KEY_FLOOR = -2139095041 << 32


@triton.jit
def _per_token_key_oracle(
    logits,
    stride_row,
    temperatures,
    seeds,
    keys,
    columns,
    vocab_start,
    valid,
    vocab_size,
    KEY_FLOOR: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """Independent elementwise oracle: no producer-sized blocks or device reduction."""
    row = tl.program_id(0).to(tl.int64)
    column = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    token_id = vocab_start + column
    in_tensor = column < columns
    in_vocab = (column < valid) & (token_id < vocab_size)
    value = tl.load(
        logits + row * stride_row + column,
        mask=in_tensor & in_vocab,
        other=float("-inf"),
    ).to(tl.float32)
    temperature = tl.load(temperatures + row)
    if temperature > 0.0:
        uniform = tl.rand(tl.load(seeds + row), token_id)
        value = value * (1.0 / tl.maximum(temperature, 1e-20)) - tl.log(
            -tl.log(uniform)
        )
    value = (
        tl.where(
            value == value,  # noqa: PLR0124 -- IEEE NaN test
            value,
            float("-inf"),
        )
        + 0.0
    )
    bits = value.to(tl.int32, bitcast=True)
    ordered = tl.where(bits < 0, bits ^ 0x7FFFFFFF, bits).to(tl.int64)
    key = (ordered << 32) | (4294967295 - token_id.to(tl.int64))
    key = tl.where(in_vocab, key, KEY_FLOOR)
    tl.store(keys + row * columns + column, key, mask=in_tensor)


def _oracle_keys(
    logits,
    temperatures,
    seeds,
    *,
    vocab_start,
    valid,
    vocab_size,
    splits,
):
    rows, columns = logits.shape
    per_token = torch.empty((rows, columns), dtype=torch.int64, device=logits.device)
    block = 256
    _per_token_key_oracle[(rows, triton.cdiv(columns, block))](
        logits,
        logits.stride(0),
        temperatures,
        seeds,
        per_token,
        columns,
        vocab_start,
        valid,
        vocab_size,
        KEY_FLOOR=_KEY_FLOOR,
        BLOCK=block,
        num_warps=4,
    )
    if splits == 1:
        return per_token.amax(dim=1)
    span = -(-valid // splits)
    parts = []
    for part in range(splits):
        lo, hi = part * span, min((part + 1) * span, valid)
        parts.append(
            per_token[:, lo:hi].amax(dim=1)
            if lo < hi
            else torch.full(
                (rows,), _KEY_FLOOR, dtype=torch.int64, device=logits.device
            )
        )
    return torch.stack(parts, dim=1)


def _tokens(keys: torch.Tensor) -> torch.Tensor:
    if keys.dim() == 2:
        keys = keys.amax(dim=1)
    return 0xFFFFFFFF - (keys & 0xFFFFFFFF)


def test_key_splits_keeps_multimem_rows_even():
    assert [key_splits(n) for n in (1, 2048, 2049, 4096, 4097, 8192)] == [
        2,
        2,
        2,
        2,
        4,
        4,
    ]
    with pytest.raises(ValueError, match="positive"):
        key_splits(0)


def test_build_prewarms_bf16_and_fp32(monkeypatch):
    launches = []
    synchronized = []

    def fake_tensor(*args, dtype, device, **kwargs):
        assert device == "cuda"
        return SimpleNamespace(dtype=dtype)

    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)
    monkeypatch.setattr(torch.cuda, "device", lambda device: nullcontext())
    monkeypatch.setattr(torch.cuda, "synchronize", synchronized.append)
    monkeypatch.setattr(torch, "empty", fake_tensor)
    monkeypatch.setattr(torch, "ones", fake_tensor)
    monkeypatch.setattr(torch, "arange", fake_tensor)
    monkeypatch.setattr(
        "sgl_nvidia_mega.qwen3_5.sm103.sharded_gumbel._launch",
        lambda logits, *args, **kwargs: launches.append(logits.dtype),
    )

    build(
        3,
        columns=31040,
        vocab_start=31040,
        valid=31040,
        vocab_size=248320,
        splits=16,
    )

    assert launches == [torch.bfloat16, torch.float32]
    assert synchronized == [3]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_padding_nan_and_ties_follow_runtime_v2_argmax_semantics():
    # gid 15 and columns 6: are padding.  Their large values must never win.
    logits = torch.tensor(
        [
            [float("nan"), 5.0, -0.0, 5.0, -2.0, 1000.0, 2000.0, 2000.0],
            [float("nan"), 1.0, 3.0, 2.0, 0.0, 1000.0, 2000.0, 2000.0],
            [
                float("nan"),
                float("nan"),
                float("nan"),
                float("nan"),
                float("nan"),
                1000.0,
                2000.0,
                2000.0,
            ],
        ],
        dtype=torch.bfloat16,
        device="cuda",
    )
    temperatures = torch.tensor([0.0, -1.0, 0.0], device="cuda")
    seeds = torch.tensor([7, 11, 13], dtype=torch.int64, device="cuda")
    keys = gumbel_shard_keys(
        logits,
        temperatures,
        seeds,
        vocab_start=10,
        valid=6,
        vocab_size=15,
    )
    # Row 0 has a tie at global ids 11 and 13.  Row 2 has no finite real
    # column, so sanitized -inf values still tie toward its first valid id.
    assert torch.equal(
        _tokens(keys), torch.tensor([11, 12, 10], device="cuda", dtype=torch.int64)
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_row_metadata_requires_exact_vectors():
    logits = torch.zeros((3, 8), dtype=torch.bfloat16, device="cuda")
    temperatures = torch.ones(3, dtype=torch.float32, device="cuda")
    seeds = torch.arange(3, dtype=torch.int64, device="cuda")
    kwargs = {"vocab_start": 0, "valid": 8, "vocab_size": 8}

    with pytest.raises(ValueError, match=r"temperatures.*\[3\]"):
        gumbel_shard_keys(logits, temperatures.view(3, 1), seeds, **kwargs)
    with pytest.raises(ValueError, match=r"seeds.*\[3\]"):
        gumbel_shard_keys(logits, temperatures, seeds.view(3, 1), **kwargs)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_seeded_keys_are_repeatable_and_independent_of_partitioning():
    generator = torch.Generator(device="cuda").manual_seed(17)
    rows, vocabulary = 9, 5003
    logits = torch.randn(
        rows,
        vocabulary,
        dtype=torch.bfloat16,
        device="cuda",
        generator=generator,
    )
    temperatures = torch.rand(rows, device="cuda", generator=generator) + 0.2
    temperatures[::4] = 0.0
    seeds = torch.arange(rows, device="cuda", dtype=torch.int64) * 7919 + 1

    whole = gumbel_shard_keys(
        logits,
        temperatures,
        seeds,
        vocab_start=0,
        valid=vocabulary,
        vocab_size=vocabulary,
    )
    repeated = gumbel_shard_keys(
        logits,
        temperatures,
        seeds,
        vocab_start=0,
        valid=vocabulary,
        vocab_size=vocabulary,
    )
    split_programs = gumbel_shard_keys(
        logits,
        temperatures,
        seeds,
        vocab_start=0,
        valid=vocabulary,
        vocab_size=vocabulary,
        splits=key_splits(vocabulary),
    )

    cut = 2501
    left = gumbel_shard_keys(
        logits[:, :cut],
        temperatures,
        seeds,
        vocab_start=0,
        valid=cut,
        vocab_size=vocabulary,
    )
    right = gumbel_shard_keys(
        logits[:, cut:].contiguous(),
        temperatures,
        seeds,
        vocab_start=cut,
        valid=vocabulary - cut,
        vocab_size=vocabulary,
    )
    sharded = torch.maximum(left, right)

    assert torch.equal(repeated, whole)
    assert torch.equal(split_programs.amax(dim=1), whole)
    assert torch.equal(sharded, whole)

    other_seeds = seeds + 104729
    changed = gumbel_shard_keys(
        logits,
        temperatures,
        other_seeds,
        vocab_start=0,
        valid=vocabulary,
        vocab_size=vocabulary,
    )
    # Greedy rows ignore the seed; every positive-temperature row should consume it.
    assert torch.equal(changed[::4], whole[::4])
    assert torch.any(changed[temperatures > 0] != whole[temperatures > 0])


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
def test_keys_match_an_independent_per_token_oracle(dtype):
    generator = torch.Generator(device="cuda").manual_seed(23)
    rows, columns = 11, 4097
    logits = torch.randn(
        rows,
        columns,
        dtype=dtype,
        device="cuda",
        generator=generator,
    )
    logits[0, 0] = float("nan")
    # Greedy row 5 has an exact tie; columns outside either validity bound hold
    # tempting values and must still reduce to the key floor.
    logits[5, 4] = logits[5, 3011]
    temperatures = torch.rand(rows, device="cuda", generator=generator) + 0.1
    temperatures[::5] = 0.0
    seeds = torch.randint(
        0,
        2**31 - 1,
        (rows,),
        dtype=torch.int64,
        device="cuda",
        generator=generator,
    )
    vocab_start, valid, vocab_size = 31040, 4093, 35129
    logits[:, vocab_size - vocab_start :] = 1000
    for splits in (1, key_splits(valid)):
        got = gumbel_shard_keys(
            logits,
            temperatures,
            seeds,
            vocab_start=vocab_start,
            valid=valid,
            vocab_size=vocab_size,
            splits=splits,
        )
        want = _oracle_keys(
            logits,
            temperatures,
            seeds,
            vocab_start=vocab_start,
            valid=valid,
            vocab_size=vocab_size,
            splits=splits,
        )
        assert torch.equal(got, want)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
@pytest.mark.parametrize("temperature", [1.0, 0.6])
def test_gumbel_max_matches_the_softmax_distribution(temperature):
    from scipy.stats import chi2

    samples, vocabulary = 200_000, 4096
    generator = torch.Generator(device="cuda").manual_seed(1)
    logits = (
        torch.randn(vocabulary, device="cuda", generator=generator) * 3
    ).bfloat16()
    # A stride-0 expansion gives every sample the same categorical distribution;
    # its seed supplies the independent Gumbel stream.
    expanded = logits.view(1, -1).expand(samples, -1)
    temperatures = torch.full((samples,), temperature, device="cuda")
    seeds = torch.arange(samples, device="cuda", dtype=torch.int64) * 7919 + 1
    tokens = _tokens(
        gumbel_shard_keys(
            expanded,
            temperatures,
            seeds,
            vocab_start=0,
            valid=vocabulary,
            vocab_size=vocabulary,
        )
    )
    counts = torch.bincount(tokens, minlength=vocabulary).double()
    expected = torch.softmax(logits.double() / temperature, dim=-1) * samples
    # Pool the tail so every Pearson bin has an expected count of at least five.
    large = expected >= 5
    observed_bins = torch.cat([counts[large], counts[~large].sum().view(1)])
    expected_bins = torch.cat([expected[large], expected[~large].sum().view(1)])
    statistic = (
        (observed_bins - expected_bins).square() / expected_bins.clamp_min(1e-12)
    ).sum()
    p_value = chi2.sf(statistic.item(), observed_bins.numel() - 1)
    assert p_value > 1e-4, f"chi-square p-value {p_value:.3g}"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
def test_prebuilt_kernel_captures_with_a_preallocated_output(dtype):
    device = torch.cuda.current_device()
    columns, vocab_start, valid, vocab_size, splits = (
        31040,
        31040,
        31040,
        248320,
        16,
    )
    build(
        device,
        columns=columns,
        vocab_start=vocab_start,
        valid=valid,
        vocab_size=vocab_size,
        splits=splits,
    )
    logits = torch.randn(4, columns, dtype=dtype, device="cuda")
    temperatures = torch.ones(4, dtype=torch.float32, device="cuda")
    seeds = torch.arange(4, dtype=torch.int64, device="cuda") + 1
    captured = torch.empty((4, splits), dtype=torch.int64, device="cuda")

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        returned = gumbel_shard_keys(
            logits,
            temperatures,
            seeds,
            vocab_start=vocab_start,
            valid=valid,
            vocab_size=vocab_size,
            splits=splits,
            out=captured,
        )
    assert returned.data_ptr() == captured.data_ptr()

    # Exercise the captured mixed-temperature branch with data written after capture.
    temperatures.zero_()
    logits.zero_()
    logits[:, 31] = 4.0
    graph.replay()
    expected = gumbel_shard_keys(
        logits,
        temperatures,
        seeds,
        vocab_start=vocab_start,
        valid=valid,
        vocab_size=vocab_size,
        splits=splits,
    )
    assert torch.equal(captured, expected)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
def test_provider_factory_draws_seeds_and_replays_from_a_cuda_graph(dtype):
    rows, columns = 8, 4097
    splits = key_splits(columns)
    weight = torch.empty((columns, 1), dtype=torch.bfloat16, device="cuda")
    model = SimpleNamespace(
        lm_head=SimpleNamespace(
            weight=weight,
            shard_indices=SimpleNamespace(
                org_vocab_start_index=0,
                num_org_elements=columns,
                num_added_elements_padded=0,
            ),
        ),
        logits_processor=SimpleNamespace(vocab_size=columns),
    )
    binding = SimpleNamespace(tp_size=1, tp_rank=0)
    # hooks.build invokes the real two-dtype precompile before this graph exists.
    factory = hooks.build(model, binding)
    assert callable(factory)

    def stock_run(io, local_keys):
        return io, local_keys

    def stock_keys(view):
        raise AssertionError("sampled provider delegated to greedy stock keys")

    defaults = spec.SpeculativeRuntimeHooks(
        post_verify=spec.PostVerifyStrategy(run=stock_run, local_keys=stock_keys)
    )
    selected = factory(defaults)
    assert selected.post_verify.run is defaults.post_verify.run
    local_keys = selected.post_verify.local_keys

    # Equal logits make the internally generated per-row seed observable in the
    # winning key.  A graph replay advances PyTorch's captured RNG state.
    logits = torch.zeros((rows, columns), dtype=dtype, device="cuda")
    temperatures = torch.ones(rows, dtype=torch.float32, device="cuda")
    view = spec.ShardedLogitsView(
        values=logits,
        shard=spec.VocabShardView(start=0, valid=columns, vocab_size=columns),
        temperatures=temperatures,
        seeds=None,
        splits=splits,
    )

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        keys = local_keys(view)
    assert keys is not None
    assert keys.dtype == torch.int64
    assert keys.device == logits.device
    assert keys.is_contiguous()
    assert tuple(keys.shape) == (rows, splits)

    first = keys.clone()
    graph.replay()
    second = keys.clone()
    graph.replay()
    third = keys.clone()
    assert not torch.equal(first, second)
    assert not torch.equal(second, third)
