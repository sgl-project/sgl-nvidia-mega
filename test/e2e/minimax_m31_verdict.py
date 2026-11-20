# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""The rules the MiniMax-M3.1-Flash end-to-end pair judges by, and the reading of a
server log that says which path an arm ran. No SGLang import and no GPU: test/unit
checks them on hand-made inputs (test_minimax_m31_e2e_verdicts.py).

The baseline is native SGLang (the launch without --enable-nvidia-megakernel), the
candidate the same launch with the flag. A verdict is ``pass``, ``fail`` or
``unavailable`` (the case holds no evidence); a test accepts ``pass`` only.

Accuracy (``accuracy_verdict``), paired on the same questions in the same order. With
b the questions only the baseline answers correctly and c those only the candidate
does, the candidate passes when

    b - c <= max(2, ceil(2 * sqrt(b + c)))

Questions both arms answer the same way say nothing about a difference; if the arms
are equally accurate each of the other b + c is as likely to fall on either side, so
b - c has mean 0 and standard deviation sqrt(b + c), and the rule is a one-sided sign
test at about two standard deviations of the discordant count. The floor of 2 is for
the counts too small for that reading. A baseline with no correct answer is
``unavailable``: the evaluation did not run, and 0 against 0 is not a comparison. No
absolute score enters the rule.

Performance (``speed_verdict``), per concurrency, on each arm's passes (tokens per
second per user, one number a pass, at least 3 passes an arm). An arm's spread is half
its range over its mean, (max - min) / (2 * mean), so +-spread holds every pass.
- The fused path (one request): the candidate's slowest pass must beat the baseline's
  fastest pass, a speed claim outside the observed noise.
- A fallback width: the candidate's mean must not be below the baseline's mean by more
  than the larger of 3% and the sum of the two arms' spreads, so no width's regression
  hides in an aggregate.

Engagement (``candidate_reasons``, ``baseline_reasons``): SGLang logs one
``NVIDIA megakernel: active (<spec>: ...`` line per rank and spec at startup and one
``NVIDIA megakernel: <spec> runs the decode graphs of batch sizes [...]`` line after
graph capture; a rank's lines start ``[<time> TP<rank> PID<pid>]``. The candidate's log
must hold both for every spec, the first on every rank; the baseline's no megakernel
line at all, in a log that does hold every rank's lines."""

import math
import re
import statistics
from dataclasses import dataclass

PASS, FAIL, UNAVAILABLE = "pass", "fail", "unavailable"
# A fallback width's allowed drop: at least this, more if the passes spread wider
FALLBACK_TOLERANCE = 0.03
MIN_PASSES = 3


@dataclass(frozen=True)
class Accuracy:
    """Both arms' answers to the same questions."""

    questions: int
    baseline: int  # correct answers
    candidate: int
    b: int  # only the baseline is right
    c: int  # only the candidate is right

    @property
    def bound(self) -> int:
        return max(2, math.ceil(2 * math.sqrt(self.b + self.c)))

    @property
    def verdict(self) -> str:
        if not self.baseline:
            return UNAVAILABLE
        return PASS if self.b - self.c <= self.bound else FAIL


def accuracy_verdict(baseline, candidate) -> Accuracy:
    """baseline, candidate: per question, whether the arm answered it correctly."""
    baseline, candidate = list(baseline), list(candidate)
    if not baseline or len(baseline) != len(candidate):
        raise ValueError(
            f"the arms answered {len(baseline)} and {len(candidate)} questions: "
            "not the same non-empty set"
        )
    return Accuracy(
        questions=len(baseline),
        baseline=sum(map(bool, baseline)),
        candidate=sum(map(bool, candidate)),
        b=sum(1 for x, y in zip(baseline, candidate) if x and not y),
        c=sum(1 for x, y in zip(baseline, candidate) if y and not x),
    )


def _spread(passes) -> float:
    return (max(passes) - min(passes)) / (2 * statistics.mean(passes))


@dataclass(frozen=True)
class Speed:
    """Both arms' passes at one concurrency, tokens per second per user."""

    baseline: tuple
    candidate: tuple
    fused: bool  # the fused path's rule, else a fallback width's

    @property
    def baseline_mean(self) -> float:
        return statistics.mean(self.baseline)

    @property
    def candidate_mean(self) -> float:
        return statistics.mean(self.candidate)

    @property
    def change(self) -> float:
        """The candidate's mean over the baseline's, minus one."""
        return self.candidate_mean / self.baseline_mean - 1

    @property
    def baseline_spread(self) -> float:
        return _spread(self.baseline)

    @property
    def candidate_spread(self) -> float:
        return _spread(self.candidate)

    @property
    def tolerance(self) -> float:
        """A fallback width's allowed relative drop of the mean."""
        return max(FALLBACK_TOLERANCE, self.baseline_spread + self.candidate_spread)

    @property
    def rule(self) -> str:
        if self.fused:
            return (
                f"slowest pass {min(self.candidate):.1f} > "
                f"native's fastest {max(self.baseline):.1f}"
            )
        return f"change >= -{100 * self.tolerance:.1f}%"

    @property
    def verdict(self) -> str:
        if self.fused:
            return PASS if min(self.candidate) > max(self.baseline) else FAIL
        return PASS if self.change >= -self.tolerance else FAIL


def speed_verdict(baseline, candidate, fused: bool) -> Speed:
    """baseline, candidate: one arm's tokens per second per user, a number a pass."""
    baseline, candidate = tuple(baseline), tuple(candidate)
    for name, passes in (("baseline", baseline), ("candidate", candidate)):
        if len(passes) < MIN_PASSES:
            raise ValueError(
                f"the {name} has {len(passes)} passes; the rule needs {MIN_PASSES}"
            )
        if not all(isinstance(x, (int, float)) and x > 0 for x in passes):
            raise ValueError(f"the {name}'s passes are not all measured: {passes}")
    return Speed(baseline=baseline, candidate=candidate, fused=fused)


MARK = "NVIDIA megakernel:"
_RANK = re.compile(r"^\[[^\]\n]*\bTP(\d+)\b[^\]\n]*\]", re.M)
_ACTIVE = re.compile(
    r"^\[[^\]\n]*\bTP(\d+)\b[^\]\n]*\] NVIDIA megakernel: active \(([^:\n]+): ", re.M
)
_GRAPHS = re.compile(
    r"NVIDIA megakernel: ([^\n]+?) runs the decode graphs of batch sizes \[([\d, ]*)\]"
)


def active_ranks(log: str) -> dict:
    """Spec name -> the TP ranks whose startup line says its megakernel is active."""
    ranks = {}
    for rank, name in _ACTIVE.findall(log):
        ranks.setdefault(name, set()).add(int(rank))
    return ranks


def graph_batch_sizes(log: str) -> dict:
    """Spec name -> the batch sizes of the decode graphs its covered step runs."""
    return {
        name: [int(size) for size in sizes.split(",") if size.strip()]
        for name, sizes in _GRAPHS.findall(log)
    }


def candidate_reasons(log: str, ranks: int, specs: dict) -> list:
    """Why the log does not show the megakernel serving this arm ([] when it does).
    specs: spec name -> the batch sizes its decode graphs must cover, the largest of
    them being the largest graph it may run."""
    reasons, active, graphs = [], active_ranks(log), graph_batch_sizes(log)
    for name, sizes in specs.items():
        missing = sorted(set(range(ranks)) - active.get(name, set()))
        if missing:
            reasons.append(f"{name}: no 'active' line on ranks {missing}")
        found = graphs.get(name)
        if found is None:
            reasons.append(f"{name}: no decode-graph coverage line")
        elif not set(sizes) <= set(found) or max(found) != max(sizes):
            reasons.append(
                f"{name}: runs the decode graphs of batch sizes {found}, "
                f"not {sorted(sizes)} up to {max(sizes)}"
            )
    return reasons


def baseline_reasons(log: str, ranks: int) -> list:
    """Why the log does not show native SGLang serving this arm ([] when it does)."""
    reasons = []
    missing = sorted(set(range(ranks)) - {int(rank) for rank in _RANK.findall(log)})
    if missing:
        reasons.append(f"the log holds no line of ranks {missing}: nothing to judge")
    lines = [line for line in log.splitlines() if MARK in line]
    if lines:
        reasons.append(f"{len(lines)} megakernel lines, the first: {lines[0]!r}")
    return reasons


def table(header, rows) -> str:
    """The rows under their header, columns aligned, for the test log."""
    cells = [[str(x) for x in row] for row in (header, *rows)]
    widths = [max(len(row[i]) for row in cells) for i in range(len(header))]
    lines = [
        " | ".join(x.ljust(w) for x, w in zip(row, widths)).rstrip() for row in cells
    ]
    return "\n".join([lines[0], "-+-".join("-" * w for w in widths), *lines[1:]])


def accuracy_row(case: str, v: Accuracy) -> tuple:
    return (
        case,
        f"{v.baseline}/{v.questions} = {v.baseline / v.questions:.3f}",
        f"{v.candidate}/{v.questions} = {v.candidate / v.questions:.3f}",
        v.b,
        v.c,
        f"b - c = {v.b - v.c} <= {v.bound}",
        v.verdict,
    )


ACCURACY_HEADER = (
    "GSM8K",
    "native SGLang",
    "megakernel",
    "b (native only)",
    "c (megakernel only)",
    "rule",
    "verdict",
)


def speed_row(concurrency: int, v: Speed) -> tuple:
    return (
        concurrency,
        f"{v.baseline_mean:.1f}",
        f"{v.candidate_mean:.1f}",
        f"{100 * v.change:+.1f}%",
        f"+-{100 * v.baseline_spread:.1f}%",
        f"+-{100 * v.candidate_spread:.1f}%",
        v.rule,
        v.verdict,
    )


SPEED_HEADER = (
    "requests",
    "native SGLang tok/s/user",
    "megakernel tok/s/user",
    "change",
    "native passes",
    "megakernel passes",
    "rule",
    "verdict",
)
