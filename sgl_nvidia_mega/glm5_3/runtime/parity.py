# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Test only: the composition check of the stock step (``SGLANG_NVIDIA_MEGAKERNEL_PARITY``).

With ``1``, every eager verify step of a covered width runs SGLang's forward, then the
plugin's stock step (every step group on its stock body: the kernels run only inside
captured graphs) on the same inputs, and logs the largest difference of the logits;
``control`` runs SGLang's forward twice (the run-to-run floor). The plugin's step is
returned. The spec accepts eager decode only under this switch. With slot ``draft``,
every eager draft-decode and draft-extend step does the same with the NextN draft's
stock step, comparing the logits and the hidden states it hands to the next step.
"""

from __future__ import annotations

import logging
import os

logger = logging.getLogger(__name__)

PARITY_ENV = "SGLANG_NVIDIA_MEGAKERNEL_PARITY"


def mode() -> str:
    return os.environ.get(PARITY_ENV, "0")


def eager_steps_allowed() -> bool:
    """Whether eager verify steps of a covered width run the plugin's stock step."""
    return mode() in ("1", "control")


def run(built, step, original, input_ids, positions, forward_batch, input_embeds):
    """SGLang's forward, then the plugin's stock step (or SGLang's again under
    ``control``), on the same eager step; logs the difference of the logits."""
    want = original(input_ids, positions, forward_batch, input_embeds)
    want = _clone(want.next_token_logits)
    if mode() == "control":
        got = original(input_ids, positions, forward_batch, input_embeds)
    else:
        got = built.run_stock(step, input_ids, positions, forward_batch)
    logger.info(
        "NVIDIA megakernel parity%s %s rows=%d logits_max_abs=%s",
        " control" if mode() == "control" else "",
        step.mode,
        step.rows,
        _max_abs(want, got.next_token_logits),
    )
    return got


def _clone(t):
    return None if t is None else t.clone()


def _max_abs(a, b):
    """max |a - b|, that over max |a|, and the fraction of elements that differ."""
    if a is None or b is None:
        return "n/a" if a is None and b is None else "missing"
    if a.shape != b.shape:
        return f"shape {tuple(a.shape)} vs {tuple(b.shape)}"
    d = (a.float() - b.float()).abs()
    peak = a.float().abs().max().clamp_min(1e-30)
    return (
        f"{d.max().item():.3g} (rel {(d.max() / peak).item():.2e}, "
        f"{(d > 0).float().mean().item():.2%} differ)"
    )


def run_draft(built, step, original, input_ids, positions, forward_batch):
    """SGLang's draft forward, then the draft's stock step (or SGLang's again under
    ``control``), on the same eager draft step; logs the differences of the logits
    and of the hidden states. Both passes read the step's own top-k carry (the first
    publishes its output there for the next step; restored before the second)."""
    spec_info = forward_batch.spec_info
    has_carry = hasattr(spec_info, "dsa_topk_indices")
    carry = spec_info.dsa_topk_indices if has_carry else None
    want = original(input_ids, positions, forward_batch)
    want_logits = _clone(want.next_token_logits)
    want_hidden = _clone(want.hidden_states)
    if has_carry:
        spec_info.dsa_topk_indices = carry
    if mode() == "control":
        got = original(input_ids, positions, forward_batch)
    else:
        got = built.run_stock(step, input_ids, positions, forward_batch)
    logger.info(
        "NVIDIA megakernel parity%s %s rows=%d logits_max_abs=%s hidden_max_abs=%s",
        " control" if mode() == "control" else "",
        step.mode,
        step.rows,
        _max_abs(want_logits, got.next_token_logits),
        _max_abs(want_hidden, got.hidden_states),
    )
    return got
