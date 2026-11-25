# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""The composition check (test only): ``SGLANG_NVIDIA_MEGAKERNEL_PARITY=1`` runs every
covered step eagerly, after SGLang's own forward on the same step, and logs how far
apart they are; ``control`` runs SGLang's forward twice (the run-to-run floor).

Two log lines, both starting ``NVIDIA megakernel parity``:

- a step slot mega serves (``log_parity``, also the draft step's): the logits' (or the
  draft's rows') largest difference, its ratio to the largest value, the smallest row
  cosine, the fraction of rows whose argmax agrees and the largest row KL, against the
  tolerances ``spec.NUMERICS`` (``draft.NUMERICS``) states;
- any other covered step (``log_bitwise``), whose slots are bitwise: the largest
  difference of the logits and of the DSpark hidden states, that over the largest
  value, and the fraction of elements that differ (``0 (rel 0.00e+00, 0.00% differ)``
  when the step is SGLang's bit for bit).
"""

from __future__ import annotations

import logging
import os

logger = logging.getLogger(__name__)

PARITY_ENV = "SGLANG_NVIDIA_MEGAKERNEL_PARITY"


def mode() -> str:
    """The switch's value: "0" (off), "1" or "control"."""
    return os.environ.get(PARITY_ENV, "") or "0"


def log_parity(what: str, want, got) -> None:
    """SGLang's rows against the megakernel's on one eager step (the parity switch)."""
    import torch

    want, got = want.float(), got.float()
    diff = (want - got).abs().max()
    cosine = torch.nn.functional.cosine_similarity(want, got, dim=-1).min()
    top1 = (want.argmax(-1) == got.argmax(-1)).float().mean()
    log_p, log_q = want.log_softmax(-1), got.log_softmax(-1)
    kl = (log_p.exp() * (log_p - log_q)).sum(-1).max()
    logger.info(
        "NVIDIA megakernel parity %s rows=%d max_abs=%.4g rel=%.3g min_cosine=%.6f "
        "top1=%.3f max_kl=%.4g",
        what,
        want.shape[0],
        diff.item(),
        (diff / want.abs().max().clamp_min(1e-30)).item(),
        cosine.item(),
        top1.item(),
        kl.item(),
    )


def log_bitwise(step, want, got, *, control: bool) -> None:
    """SGLang's ``LogitsProcessorOutput`` fields (``want``: the logits and the DSpark
    hidden states, cloned) against the executor's ``got`` on one eager step."""
    diffs = [
        max_abs(a, b) for a, b in zip(want, (got.next_token_logits, got.hidden_states))
    ]
    if all(a is None for a in want):
        # SGLang's forward produced neither: name what the output does hold
        logger.info(
            "NVIDIA megakernel parity %s rows=%d: SGLang's output %s holds %s",
            step.mode,
            step.rows,
            type(got).__name__,
            _fields(got),
        )
    logger.info(
        "NVIDIA megakernel parity%s %s rows=%d logits_max_abs=%s hidden_max_abs=%s",
        " control" if control else "",
        step.mode,
        step.rows,
        *diffs,
    )


def clone(t):
    return None if t is None else t.clone()


def max_abs(a, b) -> str:
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


def _fields(out) -> str:
    """The non-None tensor fields of a forward output, with their shapes."""
    import dataclasses

    import torch

    items = (
        ((f.name, getattr(out, f.name)) for f in dataclasses.fields(out))
        if dataclasses.is_dataclass(out)
        else vars(out).items()
        if hasattr(out, "__dict__")
        else ()
    )
    return (
        ", ".join(
            f"{k}={tuple(v.shape) if isinstance(v, torch.Tensor) else type(v).__name__}"
            for k, v in items
            if v is not None
        )
        or "nothing"
    )
