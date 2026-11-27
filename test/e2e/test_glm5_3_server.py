# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""GLM-5.3 / GLM-5.2 (NVFP4) on GB300, TP4, MTP 5 steps / 6 draft tokens: the
megakernel against SGLang's own path on the same GSM8K questions.

``TestAccuracyAgainstBaseline`` launches the baseline (no
``--enable-nvidia-megakernel``), scores it, stops it, then launches the candidate
with the flag and scores it the same way: few-shot GSM8K (5 shots, all 1319 test
questions, greedy, 512 new tokens), real speculative acceptance. Two client
concurrencies: 8 requests keep every verify step within the TP4 kernel units' widths
(up to 48 rows); 16 move the batch across that boundary (9-16 requests run SGLang's
path) and back. Each comparison prints a verdict under the rules below, which were
fixed before any candidate was measured. The candidate's log must show the TP4 kernel
units serving and the NextN draft armed; the baseline's no megakernel line.

``SGLANG_NVIDIA_MEGAKERNEL_E2E_URL`` scores a server that is already running instead
(``TestRunningServerGsm8k``): ``SGLANG_NVIDIA_MEGAKERNEL_E2E_GSM8K_OUT`` saves its
per-question answers, and ``SGLANG_NVIDIA_MEGAKERNEL_E2E_GSM8K_BASELINE`` (the
baseline arm's saved file) applies the same rules to it. Its log is not read there.
"""

import concurrent.futures
import json
import math
import os
import re
import tempfile
import unittest
from pathlib import Path

import requests
from glm5_3_server import ARGS, COVERED_REQUESTS, ENV, MODEL, URL
from sglang.test.few_shot_gsm8k import (
    INVALID,
    get_answer_value,
    get_few_shot_examples,
    get_one_example,
)
from sglang.test.test_utils import (
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
    terminate_and_kill_process_tree,
)
from sglang.utils import download_and_cache_file, read_jsonl

from sgl_nvidia_mega.glm5_3.runtime import SLOTS_ENV
from sgl_nvidia_mega.glm5_3.runtime.parity import PARITY_ENV
from sgl_nvidia_mega.plugin import FLAG_CLI

LAUNCH_TIMEOUT = 3600
GSM8K_URL = "https://raw.githubusercontent.com/openai/grade-school-math/master/grade_school_math/data/test.jsonl"
NUM_SHOTS = 5
MAX_NEW_TOKENS = 512
STOP = ["Question", "Assistant:", "<|separator|>"]
# within the kernel units' widths; across their boundary and back
PARALLEL = (COVERED_REQUESTS, 2 * COVERED_REQUESTS)

# Accuracy rule: candidate accuracy >= baseline accuracy - ACCURACY_MARGIN, on the same
# 1319 questions. Both arms decode greedily, so they differ only on the questions the
# candidate's declared-tolerance numerics (spec.NUMERICS) flip, in either direction.
# The paired difference then has a standard deviation of sqrt(lost + gained) / 1319:
# about 0.6 pp when 5% of the questions flip. 1.5 pp (20 questions net) is about 2.5 of
# those, so a matched run seldom fails by noise, and a net loss of 20 or more
# questions fails.
ACCURACY_MARGIN = 0.015
# Acceptance rule: the candidate's mean acceptance length (output tokens per verify
# step over all requests) >= ACCEPT_RATIO x the baseline's. The draft step's kernels
# propose the same greedy tokens up to their declared tolerances, so matched arms
# differ by far less than 10%; a draft step that proposes wrong tokens keeps the
# accuracy (verify rejects them) but loses much more acceptance than that.
ACCEPT_RATIO = 0.9

# the candidate's log: the TP4 kernel units serve (the active line names them), the
# verify widths they capture, and the NextN draft composed from its units
ACTIVE = re.compile(
    r"NVIDIA megakernel: active \(GLM-5\.3 / GLM-5\.2: TP4 kernel units \["
)
CAPTURE = re.compile(
    r"NVIDIA megakernel: GLM-5\.3 / GLM-5\.2: capturing the verify step, (\d+) rows"
)
DRAFT_ARMED = "NVIDIA megakernel: GLM-5.3 / GLM-5.2 draft armed (TP4)"
# SGLang's line for a built megakernel, or any line of this package's GLM-5.3 code
ANY_MEGAKERNEL = re.compile(r"NVIDIA megakernel: active|GLM-5\.3 / GLM-5\.2")
# measurement-only settings that must not reach an accuracy run, and test hooks that
# change what the candidate runs (it must run every shipped slot)
REFUSED_ENV = ("SGLANG_SIMULATE_ACC_LEN", SLOTS_ENV, PARITY_ENV)


def _questions():
    """The few-shot prompts and labels of SGLang's few_shot_gsm8k (all test
    questions)."""
    path = os.environ.get("SGLANG_NVIDIA_MEGAKERNEL_E2E_GSM8K") or (
        download_and_cache_file(GSM8K_URL)
    )
    lines = list(read_jsonl(path))
    shots = get_few_shot_examples(lines, NUM_SHOTS)
    prompts = [shots + get_one_example(lines, i, False) for i in range(len(lines))]
    labels = [get_answer_value(line["answer"]) for line in lines]
    assert INVALID not in labels
    return prompts, labels


def _gsm8k(url: str, parallel: int) -> dict:
    """One arm's GSM8K at ``parallel`` concurrent requests: each question's answer and
    whether it is right, and the output tokens and verify steps of all requests."""
    prompts, labels = _questions()

    def one(prompt):
        r = requests.post(
            url + "/generate",
            json={
                "text": prompt,
                "sampling_params": {
                    "temperature": 0,
                    "max_new_tokens": MAX_NEW_TOKENS,
                    "stop": STOP,
                },
            },
            timeout=1800,
        )
        r.raise_for_status()
        out = r.json()
        meta = out["meta_info"]
        return (
            get_answer_value(out["text"]),
            meta["completion_tokens"],
            meta.get("spec_verify_ct", 0),
        )

    with concurrent.futures.ThreadPoolExecutor(parallel) as pool:
        outs = list(pool.map(one, prompts))
    correct = [pred == label for (pred, _, _), label in zip(outs, labels)]
    tokens = sum(t for _, t, _ in outs)
    verify = sum(v for _, _, v in outs)
    result = {
        "parallel": parallel,
        "correct": correct,
        "accuracy": sum(correct) / len(correct),
        "invalid": sum(pred == INVALID for pred, _, _ in outs) / len(outs),
        "accept_length": tokens / verify if verify else None,
    }
    print(
        f"glm5_3 gsm8k parallel={parallel}: accuracy {result['accuracy']:.4f} "
        f"invalid {result['invalid']:.4f} accept_length {result['accept_length']}"
    )
    return result


def _accuracy_verdict(base: dict, cand: dict):
    """(passed, the verdict line) of the accuracy rule on paired results."""
    n = len(base["correct"])
    assert n == len(cand["correct"]), "the arms scored different question sets"
    lost = sum(b and not c for b, c in zip(base["correct"], cand["correct"]))
    gained = sum(c and not b for b, c in zip(base["correct"], cand["correct"]))
    diff = cand["accuracy"] - base["accuracy"]
    passed = diff >= -ACCURACY_MARGIN
    noise = math.sqrt(lost + gained) / n
    line = (
        f"glm5_3 gsm8k parallel={base['parallel']} verdict "
        f"{'PASS' if passed else 'FAIL'}: baseline {base['accuracy']:.4f}, candidate "
        f"{cand['accuracy']:.4f}, difference {diff:+.4f} (paired sd {noise:.4f}; "
        f"{lost} lost, {gained} gained of {n}); rule: candidate >= baseline - "
        f"{ACCURACY_MARGIN}"
    )
    print(line)
    return passed, line


def _accept_verdict(base: dict, cand: dict):
    """(passed, the verdict line) of the acceptance rule; passed is None when the
    server reports no verify steps."""
    b, c = base["accept_length"], cand["accept_length"]
    if b is None or c is None:
        line = f"glm5_3 accept length parallel={base['parallel']}: not reported"
        print(line)
        return None, line
    passed = c >= ACCEPT_RATIO * b
    line = (
        f"glm5_3 accept length parallel={base['parallel']} verdict "
        f"{'PASS' if passed else 'FAIL'}: baseline {b:.3f}, candidate {c:.3f} "
        f"({c / b - 1:+.2%}); rule: candidate >= {ACCEPT_RATIO} x baseline"
    )
    print(line)
    return passed, line


def _refused_env():
    return [name for name in REFUSED_ENV if os.environ.get(name) is not None]


@unittest.skipIf(URL, "launches the baseline and the candidate itself")
class TestAccuracyAgainstBaseline(CustomTestCase):
    """The baseline, then the candidate, on the same questions and settings."""

    @classmethod
    def setUpClass(cls):
        # popen_launch_server adds this process's environment to the launch's
        refused = _refused_env()
        if refused:
            raise RuntimeError(
                f"unset {refused}: the comparison runs the shipped configuration with "
                "real speculative acceptance"
            )
        cls.results, cls.logs = {}, {}
        for arm, flag in (("baseline", []), ("candidate", [FLAG_CLI])):
            files = tuple(
                tempfile.NamedTemporaryFile("w+", suffix=s) for s in (".out", ".err")
            )
            process = popen_launch_server(
                MODEL,
                DEFAULT_URL_FOR_TEST,
                timeout=LAUNCH_TIMEOUT,
                other_args=ARGS + flag,
                env={**os.environ, **ENV},
                return_stdout_stderr=files,
            )
            try:
                cls.results[arm] = {
                    p: _gsm8k(DEFAULT_URL_FOR_TEST, p) for p in PARALLEL
                }
            finally:
                terminate_and_kill_process_tree(process)
                cls.logs[arm] = "".join(
                    Path(f.name).read_text(errors="replace") for f in files
                )
                for f in files:
                    f.close()

    def test_candidate_runs_the_kernel_units_and_the_baseline_does_not(self):
        cand, base = self.logs["candidate"], self.logs["baseline"]
        self.assertRegex(cand, ACTIVE)
        # batch 1 and batch 8 (6 and 48 rows) bound the kernel units' widths
        captured = {int(rows) for rows in CAPTURE.findall(cand)}
        self.assertLessEqual({6, 6 * COVERED_REQUESTS}, captured)
        self.assertIn(DRAFT_ARMED, cand)
        self.assertIsNone(ANY_MEGAKERNEL.search(base), "the baseline ran the plugin")

    def test_gsm8k_within_the_kernel_units(self):
        self._accuracy(COVERED_REQUESTS)

    def test_gsm8k_across_the_kernel_units_boundary(self):
        self._accuracy(2 * COVERED_REQUESTS)

    def test_acceptance_length(self):
        for p in PARALLEL:
            passed, line = _accept_verdict(
                self.results["baseline"][p], self.results["candidate"][p]
            )
            self.assertIsNotNone(passed, line)
            self.assertTrue(passed, line)

    def _accuracy(self, parallel):
        passed, line = _accuracy_verdict(
            self.results["baseline"][parallel], self.results["candidate"][parallel]
        )
        self.assertTrue(passed, line)


@unittest.skipUnless(URL, "scores a server that is already running")
class TestRunningServerGsm8k(CustomTestCase):
    """One arm launched elsewhere; the comparison when the baseline arm's saved
    answers are given."""

    def test_gsm8k(self):
        results = {str(p): _gsm8k(URL, p) for p in PARALLEL}
        out = os.environ.get("SGLANG_NVIDIA_MEGAKERNEL_E2E_GSM8K_OUT")
        if out:
            Path(out).write_text(json.dumps(results))
        baseline = os.environ.get("SGLANG_NVIDIA_MEGAKERNEL_E2E_GSM8K_BASELINE")
        if not baseline:
            self.skipTest(
                "scored one arm; to compare, save the baseline's answers "
                "(SGLANG_NVIDIA_MEGAKERNEL_E2E_GSM8K_OUT) and pass that file as "
                "SGLANG_NVIDIA_MEGAKERNEL_E2E_GSM8K_BASELINE when scoring the candidate"
            )
        base = json.loads(Path(baseline).read_text())
        verdicts = [_accuracy_verdict(base[k], results[k]) for k in results]
        verdicts += [_accept_verdict(base[k], results[k]) for k in results]
        failed = [line for passed, line in verdicts if passed is False]
        self.assertEqual(failed, [])


if __name__ == "__main__":
    unittest.main()
