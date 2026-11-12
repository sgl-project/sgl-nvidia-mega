# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""CPU contracts for checkpoint-selected auxiliary capture widths."""
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from sgl_nvidia_mega.kimi_k31.verify_binding import KdaVerifyStep, VerifyStepView
from sgl_nvidia_mega.kimi_k31.verify_executor import K31VerifyExecutor


def step(aux):
    kda = KdaVerifyStep(torch.zeros(1, dtype=torch.int32),
                        torch.zeros(1, dtype=torch.int32),
                        torch.ones(1, dtype=torch.int32), 1, 8, 8)
    return VerifyStepView(torch.zeros(1, 32, dtype=torch.int32),
                          torch.ones(1, dtype=torch.int32),
                          torch.arange(8), kda, aux, 1, 8, 8)


@pytest.mark.parametrize('count', [6, 7])
def test_capture_width_and_slices_follow_metadata(count, monkeypatch):
    monkeypatch.setattr(
        "sgl_nvidia_mega.kimi_k31.verify_executor.VerifyMlaIndexOverlap", Mock()
    )
    ids = tuple(range(0, 2 * count, 2))
    consumers = {i: SimpleNamespace(bind_aux=Mock()) for i in range(93)}
    resources = SimpleNamespace(layers=(), before=consumers, final=consumers[92],
                                feed_forward={},
                                scratch={"hidden.0": torch.empty(8, 7168)})
    executor = K31VerifyExecutor(resources, {}, ids)
    view = step(torch.zeros(8, count * 7168, dtype=torch.bfloat16))
    view.validate(auxiliary_width=count * 7168)
    for i, layer in enumerate(ids):
        value = executor._aux(view, layer)
        assert value.shape == (8, 7168)
        value.fill_(i + 1)
        assert torch.all(view.aux_hidden[:, i * 7168:(i + 1) * 7168] == i + 1)
        consumers[layer + 1].bind_aux.assert_called_once_with(mixture=True)
    assert executor._aux(view, 91) is None
    assert not torch.cuda.is_initialized()


def test_stale_capture_count_is_rejected():
    with pytest.raises(ValueError, match='configured capture layers'):
        step(torch.empty(8, 7 * 7168, dtype=torch.bfloat16)).validate(
            auxiliary_width=6 * 7168)


@pytest.mark.parametrize('aux', [
    torch.empty(8, 6 * 7168),
    torch.empty(8, 12 * 7168, dtype=torch.bfloat16)[:, ::2],
])
def test_aux_feature_contract_is_bf16_and_dense(aux):
    with pytest.raises(ValueError, match='contiguous BF16'):
        step(aux).validate(auxiliary_width=6 * 7168)


def test_host_allocates_runtime_capture_width_and_freezes_order():
    from sglang.srt.megakernels.kimi_k31_verify_binding import K31VerifyHostBinding

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.empty(0))
            self.ids = tuple(range(0, 12, 2))
        def get_aux_hidden_states_width(self):
            return len(self.ids) * 7168

    model = Model()
    built = SimpleNamespace(widths=(8,), configure_aux=Mock())
    binding = K31VerifyHostBinding(model)
    binding.configure_aux(built, model.ids)
    assert binding.aux.shape == (8, 6 * 7168)
    assert binding.aux.dtype == torch.bfloat16
    built.configure_aux.assert_called_once_with(model.ids)
    with pytest.raises(RuntimeError, match='changed after preparation'):
        binding.configure_aux(built, tuple(reversed(model.ids)))
    model.ids = model.ids[:-1]
    with pytest.raises(ValueError, match='target capture layout'):
        binding.configure_aux(built, tuple(range(0, 12, 2)))
    assert not torch.cuda.is_initialized()
