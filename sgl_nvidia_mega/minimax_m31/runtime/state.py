# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""``StepState`` (``st``): what one covered step carries between its step functions.

The executor creates one per forward of a width the stock executor runs; a step
package finds its build result in ``bound["<package>"]``.
"""

from __future__ import annotations

from typing import Any


class StepState:
    def __init__(
        self,
        *,
        lm: Any,
        forward_batch: Any,
        positions: Any,
        input_ids: Any,
        slots: Any,
        bound: dict,
    ):
        self.lm = lm  # MiniMaxM3SparseForConditionalGeneration
        self.model = lm.model  # MiniMaxM3Model: embedding, layers, final norm
        self.fb = forward_batch
        self.positions = positions
        self.input_ids = input_ids
        self.slots = slots  # plan.Slots: the kernels that fill this width's steps
        self.bound = bound  # package -> its build result
