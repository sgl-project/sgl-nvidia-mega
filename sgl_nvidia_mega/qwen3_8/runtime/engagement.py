# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Which implementation ran each op of a covered step: ``stock`` (SGLang's own module
or kernel, called from the executor) or a plugin kernel's name.

A step's step functions record their choice through ``StepState.ran``. The record is
logged whenever a (mode, row bucket) runs a different set than before; a captured
graph's record is written while it is captured, which is the set every replay of that
graph runs. The log line is ``NVIDIA megakernel engagement: <model> <mode> rows=<n>
[graph]: op=impl; ...``."""

from __future__ import annotations

import logging

import torch

logger = logging.getLogger(__name__)


def _bucket(rows: int) -> int:
    return 1 << max(0, int(rows) - 1).bit_length()


class Engagement:
    def __init__(self, label: str):
        self.label = label
        self._step: dict[str, str] = {}
        self._seen: dict[tuple, tuple] = {}

    def begin(self) -> None:
        self._step = {}

    def note(self, mode: str, rows: int, op: str, impl: str) -> None:
        self._step[op] = impl

    def end(self, mode: str, rows: int) -> None:
        captured = (
            torch.cuda.is_available() and torch.cuda.is_current_stream_capturing()
        )
        key = (mode, rows if captured else _bucket(rows), captured)
        ops = tuple(sorted(self._step.items()))
        if self._seen.get(key) == ops:
            return
        self._seen[key] = ops
        logger.info(
            "NVIDIA megakernel engagement: %s %s rows=%d%s: %s",
            self.label,
            mode,
            rows,
            " graph" if captured else "",
            "; ".join(f"{op}={impl}" for op, impl in ops) or "none",
        )
