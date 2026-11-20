# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Decode speed of MiniMax-M3.1-Flash with --enable-nvidia-megakernel against native
SGLang: 4 x GB300, TP4, NVFP4 checkpoint, DSpark block 7 (8-row verify steps), one
server for 1-16 running requests. The accuracy half of the pair is
test_minimax_m31_acc.py; correctness is judged there, not here.

The arms (minimax_m31_server.py): the baseline is native SGLang, the launch without the
flag; the candidate is the same launch with it. Same SGLang, same checkpoint, arguments
and environment. One measurement-only override, on both arms:
SGLANG_SIMULATE_ACC_LEN=3.95 fixes the accepted length, so both arms run the same
number of verify steps per token (its text is meaningless: never an accuracy setting).
The baseline launches, is measured and torn down; then the candidate, on the same GPUs.

The workload is the static decode table, benchmark/static_table.py run as a subprocess
(the tool behind a PR's speed numbers, not a copy of it): unique 8K-token prompts, 1K
new tokens, streamed, greedy, closed loop; per point 2 x C warm-up requests, then 5 x C
timed ones, and the point's number is the median tokens per second per user. The points:
concurrency 1, the fused path (slot mega, the draft's Markov chain), and the fallback
widths 2 and 16, the two ends of what the bitwise executor serves (slots moe_nofill and
moe_swap) and the boundary of one server. Each arm, once it is up (launch, weight copies
and kernel builds are over), runs one untimed pass over the points and then R timed
passes, each on fresh prompt seeds (--salt, the same seeds on both arms), alternate
passes in reverse point order.

The rules, chosen before the numbers (minimax_m31_verdict.py states why), per point on
the passes' numbers; an arm's spread is half its range over its mean:
- concurrency 1: the candidate's slowest pass must beat the baseline's fastest pass, a
  speed claim outside the observed noise;
- a fallback width: the candidate's mean must not be below the baseline's mean by more
  than the larger of 3% and the sum of the two arms' spreads; each width is judged on
  its own, so no regression hides in an aggregate;
- engagement: the candidate's log holds the megakernel's ``active`` line of both specs
  on every rank and their decode-graph coverage lines; the baseline's log holds no
  megakernel line.
The log gets, per point, both arms' mean tokens per second per user, the relative change
and each arm's spread. A launch or a pass that fails fails the file, and so does a
point list without concurrency 1 or without a fallback width (unavailable coverage).
One launch per arm; the 3% floor is about the launch-to-launch difference of this
configuration.

SGLANG_NVIDIA_MEGAKERNEL_E2E_PERF_CONCURRENCY sets the points (default 1,2,16; 1 to 16),
SGLANG_NVIDIA_MEGAKERNEL_E2E_PERF_PASSES sets R (default 3, the least the rules take)
and SGLANG_NVIDIA_MEGAKERNEL_E2E_PERF_ACC_LEN the accepted length; the other variables
are minimax_m31_server.py's. Two launches (about 6 and 12 minutes) and four passes an
arm (about 1.5 minutes each) take about 30 minutes."""

import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import minimax_m31_verdict as verdict
from minimax_m31_server import (
    ARMS,
    BASELINE,
    CANDIDATE,
    CASE,
    engagement_failures,
    serve,
)
from sglang.test.test_utils import CustomTestCase

from sgl_nvidia_mega.minimax_m31 import MAX_REQUESTS

STATIC_TABLE = Path(__file__).resolve().parents[2] / "benchmark" / "static_table.py"
ISL, OSL = 8192, 1024
CONCURRENCY = [
    int(c)
    for c in os.environ.get(
        "SGLANG_NVIDIA_MEGAKERNEL_E2E_PERF_CONCURRENCY", "1,2,16"
    ).split(",")
]
PASSES = int(os.environ.get("SGLANG_NVIDIA_MEGAKERNEL_E2E_PERF_PASSES", "3"))
# Measurement only: both arms accept the same length, so they run the same steps
MEASUREMENT_ENV = {
    "SGLANG_SIMULATE_ACC_LEN": os.environ.get(
        "SGLANG_NVIDIA_MEGAKERNEL_E2E_PERF_ACC_LEN", "3.95"
    )
}
FUSED = 1  # one request: slot mega's step
PASS_TIMEOUT = 1800


def static_table(url: str, out: Path, salt: str, reverse: bool = False) -> dict:
    """One pass of benchmark/static_table.py over the points: concurrency -> the
    median tokens per second per user."""
    command = [
        sys.executable, str(STATIC_TABLE), "--base-url", url,
        "--isl", str(ISL), "--osl", str(OSL),
        "--conc", ",".join(map(str, CONCURRENCY)),
        "--salt", salt, "--label", f"{out.parent.name}-{out.name}", "--out", str(out),
    ]  # fmt: skip
    if reverse:
        command.append("--reverse")
    subprocess.run(command, check=True, timeout=PASS_TIMEOUT)
    rows = map(json.loads, (out / "summary.jsonl").read_text().splitlines())
    points = {row["conc"]: row["tps_user_p50"] for row in rows}
    if sorted(points) != sorted(CONCURRENCY):
        raise RuntimeError(f"{out.name} measured {sorted(points)}, not {CONCURRENCY}")
    return points


def measure(url: str, out: Path) -> dict:
    """Concurrency -> the timed passes' tokens per second per user."""
    static_table(url, out / "warm-up", salt="warm-up")
    passes = [
        static_table(url, out / f"r{i}", salt=f"r{i}", reverse=i % 2 == 0)
        for i in range(1, PASSES + 1)
    ]
    return {c: [points[c] for points in passes] for c in CONCURRENCY}


class TestPerformanceAgainstNativeSGLang(CustomTestCase):
    """Both arms over the same points; each test reads one rule's verdicts."""

    @classmethod
    def setUpClass(cls):
        if len(set(CONCURRENCY)) != len(CONCURRENCY) or not all(
            1 <= c <= MAX_REQUESTS for c in CONCURRENCY
        ):
            raise ValueError(f"points {CONCURRENCY}: distinct, 1 to {MAX_REQUESTS}")
        if PASSES < verdict.MIN_PASSES:
            raise ValueError(f"{PASSES} passes: the rules take {verdict.MIN_PASSES}")
        cls.servers, cls.passes = {}, {}
        with tempfile.TemporaryDirectory() as out:
            for arm, megakernel in ARMS.items():
                with serve(megakernel, extra_env=MEASUREMENT_ENV) as server:
                    cls.servers[arm] = server
                    cls.passes[arm] = measure(
                        server.url, Path(out) / arm.replace(" ", "-")
                    )
        cls.verdicts = {
            c: verdict.speed_verdict(
                cls.passes[BASELINE][c], cls.passes[CANDIDATE][c], fused=c == FUSED
            )
            for c in CONCURRENCY
        }
        rows = [verdict.speed_row(c, v) for c, v in sorted(cls.verdicts.items())]
        print(
            f"\n{CASE}; static decode table, {ISL} in / {OSL} out, streamed; "
            f"{MEASUREMENT_ENV} on both arms; {PASSES} passes an arm, +- is half "
            "their range over their mean\n"
            + verdict.table(verdict.SPEED_HEADER, rows)
            + f"\ntokens per second per user of each pass: {cls.passes}",
            flush=True,
        )

    def _check(self, points):
        failed = [
            verdict.speed_row(c, self.verdicts[c])
            for c in points
            if self.verdicts[c].verdict != verdict.PASS
        ]
        self.assertEqual(failed, [])

    def test_one_request_is_faster_than_native(self):
        self.assertIn(FUSED, CONCURRENCY, "unavailable: the fused path has no point")
        self._check([FUSED])

    def test_the_fallback_widths_do_not_regress(self):
        widths = [c for c in CONCURRENCY if c != FUSED]
        self.assertTrue(widths, "unavailable: no fallback width among the points")
        self._check(widths)

    def test_only_the_candidate_runs_the_megakernel(self):
        self.assertEqual(engagement_failures(self.servers), {})


if __name__ == "__main__":
    unittest.main()
