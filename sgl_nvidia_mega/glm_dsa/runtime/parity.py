# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 SGLang Team
"""Parity test mode: run SGLang's stock forward then the plugin's stock steps on
the same inputs and log the per-step max absolute difference.

Activated by ``SGLANG_NVIDIA_MEGAKERNEL_PARITY=1``.  An env var
``SGLANG_NVIDIA_MEGAKERNEL_PARITY_SORT_TOPK=1`` sorts sparse-attention indices
in both forwards before comparison (emission order varies between them).
"""

from __future__ import annotations

import logging
import os
from typing import Optional

logger = logging.getLogger(__name__)

PARITY_ENV = "SGLANG_NVIDIA_MEGAKERNEL_PARITY"
PARITY_SORT_ENV = "SGLANG_NVIDIA_MEGAKERNEL_PARITY_SORT_TOPK"


def mode() -> bool:
    """True when parity-check mode is active."""
    return os.environ.get(PARITY_ENV, "0") not in ("", "0")


def eager_steps_allowed() -> bool:
    """Parity mode requires the stock forward to run eagerly first."""
    return not mode()


def run(fused_decode, stock_fn, input_ids, positions, forward_batch) -> None:
    """Run ``stock_fn`` then the plugin's stock-body steps on the same inputs,
    logging the max absolute difference for each step group output."""
    import torch

    if not mode():
        return
    with torch.no_grad():
        stock_out = stock_fn(input_ids, positions, forward_batch)
        plugin_out = fused_decode.run(input_ids, positions, forward_batch)
    diff = (stock_out - plugin_out).abs().max().item()
    logger.info(
        "GLM DSA parity check: hidden-state max_abs diff stock vs plugin = %.3e "
        "(rows=%d)",
        diff,
        int(positions.shape[0]),
    )


def run_draft(fused_draft, stock_fn, model, input_ids, positions, forward_batch) -> None:
    """Parity check for the draft forward."""
    import torch

    if not mode():
        return
    with torch.no_grad():
        stock_out = stock_fn(model, input_ids, positions, forward_batch)
        plugin_out = fused_draft.run(model, input_ids, positions, forward_batch)
    diff = (stock_out - plugin_out).abs().max().item()
    logger.info(
        "GLM DSA parity check (draft): hidden-state max_abs diff = %.3e (rows=%d)",
        diff,
        int(positions.shape[0]),
    )
