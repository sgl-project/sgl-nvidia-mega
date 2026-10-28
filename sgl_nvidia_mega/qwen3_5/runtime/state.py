# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""``StepState`` (``st``): what one covered step carries between its step functions.

The runtime creates one per forward and passes it to every step function. Fields fixed
for the step are set at creation. A package keeps values of its own in ``scratch``
under ``"<package>.<name>"``, and its build result in ``bound["<package>"]``.
"""

from __future__ import annotations

from typing import Any

import torch


class StepState:
    def __init__(
        self,
        *,
        lm,
        forward_batch,
        positions: torch.Tensor,
        input_ids: torch.Tensor,
        slots,
        bound: dict,
        binding: Any = None,
        view: Any = None,
    ):
        self.lm = lm  # Qwen3_5MoeForConditionalGeneration
        self.model = lm.model  # Qwen3_5MoeForCausalLM: layers, embedding, final norm
        self.fb = forward_batch
        # what the layers take: the M-RoPE positions [3, rows] when the model uses them
        self.positions = positions
        self.input_ids = input_ids
        self.slots = slots  # plan.Slots: which kernels fill this width's step
        self.bound: dict[str, Any] = bound  # package -> its build() result
        self.binding = binding  # SGLang's EagleBinding: configuration of the TP group
        # Step.view, the forward's EagleForwardView: the attention and GDN metadata and
        # the KV and mamba pools, built at capture for a graph step
        self.view = view
        self.scratch: dict[str, Any] = {}
