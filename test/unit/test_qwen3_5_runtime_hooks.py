# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""CPU contracts for Qwen3.5's post-verify local-key provider."""

from __future__ import annotations

import logging
from types import SimpleNamespace

import pytest
import torch
from sglang.kernels.plugin_kernels.nvidia import spec

from sgl_nvidia_mega.qwen3_5.runtime import hooks


class _Tensor:
    """CUDA tensor metadata for CPU-only provider policy tests."""

    def __init__(
        self,
        shape,
        dtype,
        *,
        device="cuda:0",
        is_cuda=True,
        stride=None,
        contiguous=True,
    ):
        self.shape = tuple(shape)
        self.dtype = dtype
        self.device = torch.device(device)
        self.is_cuda = is_cuda
        self._stride = stride or (self.shape[-1], 1)
        self._contiguous = contiguous

    def dim(self):
        return len(self.shape)

    def stride(self, dimension):
        return self._stride[dimension]

    def is_contiguous(self):
        return self._contiguous


class _Weight:
    def __init__(self, columns=4, *, is_cuda=True, dimensions=2):
        self.shape = (columns, 8) if dimensions == 2 else (columns,)
        self.is_cuda = is_cuda
        self.device = torch.device("cuda:0")
        self._dimensions = dimensions

    def dim(self):
        return self._dimensions


def _descriptor(**changes):
    values = {
        "columns": 4,
        "start": 4,
        "valid": 4,
        "vocab_size": 8,
        "splits": 2,
        "device_index": 0,
    }
    values.update(changes)
    return hooks._Descriptor(**values)


def _model(
    *,
    columns=4,
    start=4,
    valid=4,
    vocab_size=8,
    added=0,
    is_cuda=True,
    dimensions=2,
):
    return SimpleNamespace(
        lm_head=SimpleNamespace(
            weight=_Weight(columns, is_cuda=is_cuda, dimensions=dimensions),
            shard_indices=SimpleNamespace(
                org_vocab_start_index=start,
                num_org_elements=valid,
                num_added_elements_padded=added,
            ),
        ),
        logits_processor=SimpleNamespace(vocab_size=vocab_size),
    )


def _defaults(default_local=None):
    if default_local is None:
        default_local = lambda view: None

    def run(io, local_keys):
        return io, local_keys

    return spec.SpeculativeRuntimeHooks(
        post_verify=spec.PostVerifyStrategy(run=run, local_keys=default_local)
    )


def _view(
    values,
    temperatures,
    *,
    seeds=None,
    start=4,
    valid=4,
    vocab_size=8,
    splits=2,
):
    return spec.ShardedLogitsView(
        values=values,
        shard=spec.VocabShardView(
            start=start,
            valid=valid,
            vocab_size=vocab_size,
        ),
        temperatures=temperatures,
        seeds=seeds,
        splits=splits,
    )


def _local(default_local=None):
    defaults = _defaults(default_local)
    factory = lambda stock: spec.SpeculativeRuntimeHooks(
        post_verify=spec.PostVerifyStrategy(
            run=stock.post_verify.run,
            local_keys=hooks._LocalKeys(_descriptor(), stock.post_verify.local_keys),
        )
    )
    return factory(defaults).post_verify.local_keys, defaults


def test_build_collects_descriptors_then_prebuilds_exact_local_shard(monkeypatch):
    model = _model()
    group = object()
    binding = SimpleNamespace(tp_size=2, tp_rank=1, tp_cpu_group=group)
    prebuilt = []

    def gather(output, local, *, group):
        assert group is binding.tp_cpu_group
        output[:] = [(_descriptor(start=0), ""), local]

    monkeypatch.setattr(hooks.dist, "all_gather_object", gather)
    monkeypatch.setattr(
        hooks.sharded_gumbel,
        "build",
        lambda *args, **kwargs: prebuilt.append((args, kwargs)),
    )

    factory = hooks.build(model, binding)

    assert callable(factory)
    assert prebuilt == [
        (
            (0,),
            {
                "columns": 4,
                "vocab_start": 4,
                "valid": 4,
                "vocab_size": 8,
                "splits": 2,
            },
        )
    ]
    defaults = _defaults()
    selected = factory(defaults)
    assert selected.post_verify.run is defaults.post_verify.run
    assert selected.post_verify.local_keys is not defaults.post_verify.local_keys


def test_padded_columns_keep_splits_uniform_across_an_uneven_valid_tail(monkeypatch):
    columns, tail, vocab_size = 4104, 4041, 8145
    model = _model(
        columns=columns,
        start=columns,
        valid=tail,
        vocab_size=vocab_size,
    )
    binding = SimpleNamespace(tp_size=2, tp_rank=1, tp_cpu_group=object())
    prebuilt = []

    def gather(output, local, *, group):
        output[:] = [
            (
                hooks._Descriptor(
                    columns=columns,
                    start=0,
                    valid=columns,
                    vocab_size=vocab_size,
                    splits=4,
                    device_index=0,
                ),
                "",
            ),
            local,
        ]

    monkeypatch.setattr(hooks.dist, "all_gather_object", gather)
    monkeypatch.setattr(
        hooks.sharded_gumbel,
        "build",
        lambda *args, **kwargs: prebuilt.append((args, kwargs)),
    )
    factory = hooks.build(model, binding)

    assert callable(factory)
    assert prebuilt == [
        (
            (0,),
            {
                "columns": columns,
                "vocab_start": columns,
                "valid": tail,
                "vocab_size": vocab_size,
                "splits": 4,
            },
        )
    ]

    monkeypatch.setattr(hooks.torch, "Tensor", _Tensor)
    generated = _Tensor((2,), torch.int64)
    launches = []
    monkeypatch.setattr(hooks.torch, "randint", lambda *args, **kwargs: generated)
    monkeypatch.setattr(
        hooks.sharded_gumbel,
        "gumbel_shard_keys",
        lambda *args, **kwargs: launches.append((args, kwargs)) or object(),
    )
    local_keys = factory(_defaults()).post_verify.local_keys
    values = _Tensor((2, columns), torch.bfloat16)
    temperatures = _Tensor((2,), torch.float32)
    local_keys(
        _view(
            values,
            temperatures,
            start=columns,
            valid=tail,
            vocab_size=vocab_size,
            splits=4,
        )
    )
    assert launches == [
        (
            (values, temperatures, generated),
            {
                "vocab_start": columns,
                "valid": tail,
                "vocab_size": vocab_size,
                "splits": 4,
            },
        )
    ]


@pytest.mark.parametrize(
    "gathered, match",
    [
        (
            [(_descriptor(start=0), ""), (None, "missing LM head")],
            "rank 1: missing LM head",
        ),
        (
            [(_descriptor(start=0), ""), (_descriptor(start=5), "")],
            "not contiguous",
        ),
        (
            [(_descriptor(start=0), ""), (_descriptor(start=4, splits=4), "")],
            "disagree",
        ),
    ],
)
def test_unsupported_topology_returns_none_uniformly_without_prebuild(
    monkeypatch, caplog, gathered, match
):
    binding = SimpleNamespace(tp_size=2, tp_rank=0, tp_cpu_group=object())

    def gather(output, local, *, group):
        output[:] = gathered

    monkeypatch.setattr(hooks.dist, "all_gather_object", gather)
    monkeypatch.setattr(
        hooks.sharded_gumbel,
        "build",
        lambda *args, **kwargs: pytest.fail("unsupported provider was prebuilt"),
    )

    with caplog.at_level(logging.INFO, logger=hooks.__name__):
        assert hooks.build(_model(start=0), binding) is None
    assert match in caplog.text


@pytest.mark.parametrize(
    "model, match",
    [
        (SimpleNamespace(), "metadata is unavailable"),
        (_model(is_cuda=False), "not a CUDA matrix"),
        (_model(dimensions=1), "not a CUDA matrix"),
        (_model(added=1), "added-vocabulary"),
        (_model(valid=0), "valid columns"),
        (_model(valid=5), "valid columns"),
        (_model(start=7, valid=4), "outside"),
    ],
)
def test_local_metadata_refusal_is_a_build_result_not_an_exception(
    monkeypatch, caplog, model, match
):
    binding = SimpleNamespace(tp_size=1, tp_rank=0)
    monkeypatch.setattr(
        hooks.sharded_gumbel,
        "build",
        lambda *args, **kwargs: pytest.fail("unsupported provider was prebuilt"),
    )
    with caplog.at_level(logging.INFO, logger=hooks.__name__):
        assert hooks.build(model, binding) is None
    assert match in caplog.text


def test_greedy_always_delegates_by_identity_even_with_seed_material():
    sentinel = object()
    calls = []

    def default(view):
        calls.append(view)
        return sentinel

    local, defaults = _local(default)
    view = _view(object(), None, seeds=object())

    assert local(view) is sentinel
    assert calls == [view]
    assert local.default is defaults.post_verify.local_keys


def test_sampled_request_seeds_decline_before_checks_or_rng(monkeypatch):
    local, _ = _local()
    monkeypatch.setattr(
        hooks.torch,
        "randint",
        lambda *args, **kwargs: pytest.fail("seeded fallback consumed RNG"),
    )
    # Neither object implements tensor metadata; an early capability read fails.
    view = _view(object(), object(), seeds=object())
    assert local(view) is None


@pytest.mark.parametrize(
    "change",
    [
        {"values": object()},
        {"values.is_cuda": False},
        {"values.device": "cuda:1"},
        {"values.shape": (2, 4, 1)},
        {"values.dtype": torch.float16},
        {"values.stride": (8, 2)},
        {"values.shape": (2, 5)},
        {"temperatures.device": "cuda:1"},
        {"temperatures.dtype": torch.bfloat16},
        {"temperatures.shape": (2, 1)},
        {"temperatures.contiguous": False},
        {"valid": 3},
        {"vocab_size": 9},
        {"splits": 4},
    ],
)
def test_every_sampled_capability_refusal_precedes_rng(monkeypatch, change):
    monkeypatch.setattr(hooks.torch, "Tensor", _Tensor)
    values = _Tensor((2, 4), torch.bfloat16)
    temperatures = _Tensor((2,), torch.float32)
    view_kwargs = {}
    if "values" in change:
        values = change["values"]
    for name, value in change.items():
        if name.startswith("values."):
            attribute = name.split(".", 1)[1]
            if attribute == "shape":
                values.shape = value
            elif attribute == "stride":
                values._stride = value
            elif attribute == "device":
                values.device = torch.device(value)
            else:
                setattr(values, attribute, value)
        elif name.startswith("temperatures."):
            attribute = name.split(".", 1)[1]
            if attribute == "contiguous":
                temperatures._contiguous = value
            elif attribute == "device":
                temperatures.device = torch.device(value)
            else:
                setattr(temperatures, attribute, value)
        elif name != "values":
            view_kwargs[name] = value
    local, _ = _local()
    monkeypatch.setattr(
        hooks.torch,
        "randint",
        lambda *args, **kwargs: pytest.fail("unsupported view consumed RNG"),
    )

    assert local(_view(values, temperatures, **view_kwargs)) is None


def test_supported_sample_draws_exact_seed_then_launches_without_collective(
    monkeypatch, caplog
):
    monkeypatch.setattr(hooks.torch, "Tensor", _Tensor)
    values = _Tensor((3, 4), torch.float32)
    temperatures = _Tensor((3,), torch.float32)
    generated = _Tensor((3,), torch.int64)
    keys = object()
    events = []

    def randint(*args, **kwargs):
        events.append(("rng", args, kwargs))
        return generated

    def launch(*args, **kwargs):
        events.append(("launch", args, kwargs))
        return keys

    monkeypatch.setattr(hooks.torch, "randint", randint)
    monkeypatch.setattr(hooks.dist, "all_gather_object", pytest.fail)
    monkeypatch.setattr(hooks.sharded_gumbel, "gumbel_shard_keys", launch)
    local, _ = _local()
    # The runtime descriptor's rank-varying start is not a call-time applicability
    # branch.  The build-captured start remains the kernel's authoritative value.
    view = _view(values, temperatures, start=777)

    with caplog.at_level(logging.INFO, logger=hooks.__name__):
        assert local(view) is keys
        assert local(view) is keys

    assert events[0] == (
        "rng",
        (0, 2**31 - 1, (3,)),
        {"dtype": torch.int64, "device": torch.device("cuda:0")},
    )
    assert events[1] == (
        "launch",
        (values, temperatures, generated),
        {
            "vocab_start": 4,
            "valid": 4,
            "vocab_size": 8,
            "splits": 2,
        },
    )
    marker = "NVIDIA megakernel: Qwen3.5 slot sharded_gumbel runs at 3 rows"
    assert caplog.text.count(marker) == 1


def test_failure_after_rng_is_hard_and_does_not_set_engagement(monkeypatch, caplog):
    monkeypatch.setattr(hooks.torch, "Tensor", _Tensor)
    values = _Tensor((2, 4), torch.bfloat16)
    temperatures = _Tensor((2,), torch.float32)
    generated = _Tensor((2,), torch.int64)
    monkeypatch.setattr(hooks.torch, "randint", lambda *args, **kwargs: generated)
    attempts = []

    def launch(*args, **kwargs):
        attempts.append(None)
        if len(attempts) == 1:
            raise RuntimeError("launch failed after RNG")
        return object()

    monkeypatch.setattr(hooks.sharded_gumbel, "gumbel_shard_keys", launch)
    local, _ = _local()
    view = _view(values, temperatures)

    with caplog.at_level(logging.INFO, logger=hooks.__name__):
        with pytest.raises(RuntimeError, match="launch failed after RNG"):
            local(view)
        assert "slot sharded_gumbel runs" not in caplog.text
        local(view)
    assert caplog.text.count("slot sharded_gumbel runs at 2 rows") == 1
