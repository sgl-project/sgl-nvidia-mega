# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Accuracy of MiniMax-M3.1-Flash with --enable-nvidia-megakernel against native SGLang:
4 x GB300, TP4, NVFP4 checkpoint, DSpark block 7 (8-row verify steps), one server for
1-16 running requests. The performance half of the pair is test_minimax_m31_perf.py.

The arms (minimax_m31_server.py): the baseline is native SGLang, the launch without the
flag; the candidate is the same launch with it. Same SGLang, same checkpoint, arguments
and environment, real speculative acceptance (no simulated acceptance, no other
measurement override). The baseline launches, runs the workload and is torn down; then
the candidate launches on the same GPUs and runs the same workload.

The workload, greedy, through the chat API:
- GSM8K, the first N test questions one at a time: every decode step is a one-request
  step, the fused path (slot mega's 8-row verify step, the draft's Markov chain);
- GSM8K, the first N16 questions 16 at a time: the 16-128-row verify steps (slots
  moe_nofill and moe_swap);
- three pass keys, each retrieved from about 13K tokens, where the sparse attention
  selects 16 pages of 103.

The rules, chosen before the numbers (minimax_m31_verdict.py states why):
- GSM8K, per width, paired on the same questions: with b the questions only the
  baseline answers correctly and c those only the candidate does, the candidate passes
  when b - c <= max(2, ceil(2 * sqrt(b + c))), a one-sided sign test at about two
  standard deviations of the discordant count. No absolute score is a bound. A baseline
  with no correct answer is unavailable coverage and fails the case.
- Pass keys: the candidate retrieves every key the baseline retrieves; a baseline that
  retrieves none is unavailable coverage and fails the case.
- Engagement: the candidate's log holds the megakernel's ``active`` line of both specs
  on every rank and their decode-graph coverage lines; the baseline's log holds no
  megakernel line.
A launch or a request that fails fails the file. The eager composition check against
SGLang's forward supplements this test and has its own file, test_minimax_m31_parity.py.

SGLANG_NVIDIA_MEGAKERNEL_E2E_GSM8K_QUESTIONS sets N (default 200) and
SGLANG_NVIDIA_MEGAKERNEL_E2E_GSM8K_QUESTIONS_16 sets N16 (default N); the other
variables are minimax_m31_server.py's. Two launches (about 6 and 12 minutes) and the
default workload take about 30 minutes."""

import os
import unittest

import minimax_m31_verdict as verdict
from minimax_m31_server import (
    ARMS,
    BASELINE,
    CANDIDATE,
    CASE,
    engagement_failures,
    gsm8k,
    passkeys,
    serve,
)
from sglang.test.test_utils import CustomTestCase

QUESTIONS = int(os.environ.get("SGLANG_NVIDIA_MEGAKERNEL_E2E_GSM8K_QUESTIONS", "200"))
QUESTIONS_16 = int(
    os.environ.get("SGLANG_NVIDIA_MEGAKERNEL_E2E_GSM8K_QUESTIONS_16", QUESTIONS)
)
# case -> (questions, how many at a time)
GSM8K = {"1 at a time": (QUESTIONS, 1), "16 at a time": (QUESTIONS_16, 16)}


class TestAccuracyAgainstNativeSGLang(CustomTestCase):
    """Both arms on the same questions; each test reads one comparison's verdict."""

    @classmethod
    def setUpClass(cls):
        cls.servers, cls.answers, cls.keys = {}, {}, {}
        for arm, megakernel in ARMS.items():
            with serve(megakernel) as server:
                cls.servers[arm] = server
                cls.answers[arm] = {
                    case: gsm8k(server.url, n, threads)
                    for case, (n, threads) in GSM8K.items()
                }
                cls.keys[arm] = passkeys(server.url)
        cls.verdicts = {
            case: verdict.accuracy_verdict(
                cls.answers[BASELINE][case], cls.answers[CANDIDATE][case]
            )
            for case in GSM8K
        }
        rows = [verdict.accuracy_row(case, v) for case, v in cls.verdicts.items()]
        print(
            f"\n{CASE}; chat GSM8K, greedy, real speculative acceptance\n"
            + verdict.table(verdict.ACCURACY_HEADER, rows),
            flush=True,
        )

    def _gsm8k(self, case):
        v = self.verdicts[case]
        self.assertEqual(v.verdict, verdict.PASS, verdict.accuracy_row(case, v))

    def test_gsm8k_one_request_at_a_time(self):
        self._gsm8k("1 at a time")

    def test_gsm8k_16_requests_at_a_time(self):
        self._gsm8k("16 at a time")

    def test_passkey_at_13k_tokens(self):
        baseline, candidate = self.keys[BASELINE], self.keys[CANDIDATE]
        print(f"pass keys (prompt tokens, retrieved): {baseline=} {candidate=}")
        for tokens, _ in baseline + candidate:
            self.assertGreater(tokens, 12000)
        self.assertTrue(
            any(found for _, found in baseline),
            "unavailable: the baseline retrieves no pass key",
        )
        lost = [
            i
            for i, ((_, b), (_, c)) in enumerate(zip(baseline, candidate))
            if b and not c
        ]
        self.assertEqual(lost, [], "keys only the baseline retrieves")

    def test_only_the_candidate_runs_the_megakernel(self):
        self.assertEqual(engagement_failures(self.servers), {})


if __name__ == "__main__":
    unittest.main()
