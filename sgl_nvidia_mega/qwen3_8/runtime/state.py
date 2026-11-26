# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""``StepState`` (``st``): what one covered step carries between its step functions.

The executor creates one per forward. A package keeps values of its own in
``scratch`` under ``"<package>.<name>"``."""

from __future__ import annotations

from typing import Any, Optional

import torch


class StepState:
    def __init__(
        self,
        *,
        lm,
        step,
        forward_batch,
        positions: torch.Tensor,
        input_ids: Optional[torch.Tensor],
        slots: frozenset,
        binding: Any,
        engagement,
    ):
        self.lm = lm  # Qwen4ExpForConditionalGeneration or Qwen4ExpForCausalLMMTP
        self.model = lm.model  # Qwen4ExpModel (Qwen4ExpVLModel for the target)
        self.step = step  # spec.Step: mode, rows, view
        self.view = step.view
        self.fb = forward_batch
        # what the layers take: the M-RoPE positions [3, rows] when the model uses them
        self.positions = positions
        self.input_ids = input_ids
        self.slots = slots
        self.binding = binding
        self.engagement = engagement  # per-op implementation record of this step
        self.ple_batch = None
        # a prefill body inside a breakable prefill graph's capture (or its warmup):
        # what reads the batch's host state runs as a graph break (prefill_graph.py)
        self.graph = False
        self.scratch: dict[str, Any] = {}

    def ran(self, op: str, impl: str) -> None:
        """Record which implementation ran an op in this step (stock or a kernel)."""
        self.engagement.note(self.step.mode, self.step.rows, op, impl)
