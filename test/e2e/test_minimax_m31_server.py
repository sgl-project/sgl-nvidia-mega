# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""MiniMax-M3.1-Flash with --enable-nvidia-megakernel on GB300, TP4, DSpark. Requests run
one at a time, so every verify step (8 rows) and draft block (7 rows) runs the plugin:
- GSM8K through the chat API (100 questions, greedy), for gross failures;
- passkey retrieval from 13K tokens, where the sparse attention selects 16 pages of 103;
- with SGLANG_NVIDIA_MEGAKERNEL_PARITY=1 and eager decode steps, each covered step's
  logits and each draft block's rows against SGLang's forward on the same step, within
  the tolerances the target and draft specs declare.

SGLANG_NVIDIA_MEGAKERNEL_E2E_URL runs the serving tests against a server that is already
up, so the same checks also score the switch-off arm."""

import os
import random
import re
import statistics
import tempfile
import unittest
from pathlib import Path

import requests
from minimax_m31_server import ARGS, ENV, MODEL, URL
from sglang.test.test_utils import (
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
    terminate_and_kill_process_tree,
)
from sglang.utils import download_and_cache_file, read_jsonl

from sgl_nvidia_mega.minimax_m31 import draft, spec
from sgl_nvidia_mega.plugin import FLAG_CLI

LAUNCH_TIMEOUT = 3600
GSM8K_URL = (
    "https://raw.githubusercontent.com/openai/grade-school-math/master/"
    "grade_school_math/data/test.jsonl"
)
# Chat GSM8K-100, greedy, for gross failures (both arms measured 0.97)
GSM8K_BOUND = float(os.environ.get("SGLANG_NVIDIA_MEGAKERNEL_E2E_GSM8K_BOUND", "0.9"))
NUMBER = re.compile(r"-?\d[\d,]*\.?\d*")
PARITY_LINE = re.compile(
    r"NVIDIA megakernel parity (target logits|draft hidden) rows=\d+ max_abs=\S+ "
    r"rel=(\S+) min_cosine=(\S+) top1=(\S+) max_kl=(\S+)"
)


def _launch(url, extra_env=None, extra_args=(), log=None):
    return popen_launch_server(
        MODEL,
        url,
        timeout=LAUNCH_TIMEOUT,
        other_args=ARGS + [FLAG_CLI, *extra_args],
        env={**os.environ, **ENV, **(extra_env or {})},
        return_stdout_stderr=log,
    )


def _questions(n):
    path = os.environ.get(
        "SGLANG_NVIDIA_MEGAKERNEL_E2E_GSM8K"
    ) or download_and_cache_file(GSM8K_URL)
    return list(read_jsonl(path))[:n]


def _chat(url, text, max_tokens):
    r = requests.post(
        url + "/v1/chat/completions",
        json={
            "model": "default",
            "messages": [{"role": "user", "content": text}],
            "temperature": 0,
            "max_tokens": max_tokens,
        },
        timeout=900,
    )
    r.raise_for_status()
    return r.json()["choices"][0]["message"]["content"] or ""


def _ask(url, row):
    return _chat(
        url, row["question"] + "\nGive the final answer as a number at the end.", 4096
    )


def _gsm8k_accuracy(url, n):
    correct = 0
    for row in _questions(n):
        reply = _ask(url, row)
        want = float(row["answer"].split("####")[-1].strip().replace(",", ""))
        found = NUMBER.findall(reply)
        correct += bool(found) and float(found[-1].replace(",", "").rstrip(".")) == want
    return correct / n


def _quantile(values, q):
    values = sorted(values)
    return values[min(len(values) - 1, int(q * len(values)))]


class _Server(CustomTestCase):
    EXTRA_ENV: dict = {}
    EXTRA_ARGS: list = []

    @classmethod
    def setUpClass(cls):
        cls.process, cls.url, cls.logs = None, URL or DEFAULT_URL_FOR_TEST, ()
        if URL:
            return
        cls.logs = tuple(
            tempfile.NamedTemporaryFile("w+", suffix=s) for s in (".out", ".err")
        )
        cls.process = _launch(
            cls.url,
            extra_env=cls.EXTRA_ENV,
            extra_args=["--watchdog-timeout", "600", *cls.EXTRA_ARGS],
            log=cls.logs,
        )

    @classmethod
    def tearDownClass(cls):
        if cls.process is not None:
            terminate_and_kill_process_tree(cls.process)
        for log in cls.logs:
            log.close()

    def log_text(self):
        return "".join(Path(f.name).read_text(errors="replace") for f in self.logs)


class TestMegakernelVerify(_Server):
    """The covered steps inside the decode CUDA graphs."""

    def test_gsm8k(self):
        self.assertGreater(_gsm8k_accuracy(self.url, 100), GSM8K_BOUND)

    def test_passkey_at_13k_tokens(self):
        rng = random.Random(1)
        filler = "The grass is green. The sky is blue. The sun is yellow. Here we go. "
        for _ in range(3):
            key = rng.randint(10000, 99999)
            body = filler * 700
            cut = rng.randint(len(body) // 4, 3 * len(body) // 4)
            prompt = (
                f"{body[:cut]} The pass key is {key}. Remember it. {body[cut:]}\n"
                "What is the pass key? The pass key is"
            )
            r = requests.post(
                self.url + "/v1/completions",
                json={"model": "default", "prompt": prompt, "max_tokens": 64,
                      "temperature": 0},
                timeout=900,
            )  # fmt: skip
            r.raise_for_status()
            self.assertGreater(r.json()["usage"]["prompt_tokens"], 12000)
            self.assertIn(str(key), r.json()["choices"][0]["text"])

    @unittest.skipIf(URL, "reads the server's own log")
    def test_both_specs_run_the_batch_1_graphs(self):
        text = self.log_text()
        for name in (spec.SPEC.name, draft.SPEC.name):
            self.assertRegex(
                text, rf"NVIDIA megakernel: {re.escape(name)} runs the decode graphs "
                r"of batch sizes \[1\]",
            )  # fmt: skip


@unittest.skipIf(URL, "launches its own server with the parity switch")
class TestParity(_Server):
    """Eager covered steps, each after SGLang's forward on the same step."""

    EXTRA_ENV = {"SGLANG_NVIDIA_MEGAKERNEL_PARITY": "1"}
    EXTRA_ARGS = ["--disable-decode-cuda-graph", "--disable-prefill-cuda-graph"]

    def test_the_covered_steps_stay_within_the_declared_tolerances(self):
        # the workload the tolerances were measured on
        for row in _questions(20):
            _ask(self.url, row)
        lines = {"target logits": [], "draft hidden": []}
        for what, *values in PARITY_LINE.findall(self.log_text()):
            lines[what].append([float(v) for v in values])
        target, block = lines["target logits"], lines["draft hidden"]
        self.assertGreater(len(target), 100)
        self.assertGreater(len(block), 100)
        rel, _, top1, kl = zip(*target)
        draft_rel, draft_cosine, _, _ = zip(*block)
        measured = {
            "verify.logits.rel.p50": _quantile(rel, 0.5),
            "verify.logits.rel.p99": _quantile(rel, 0.99),
            "verify.logits.top1": statistics.mean(top1),
            "verify.logits.kl.p50": _quantile(kl, 0.5),
            "verify.logits.kl.p99": _quantile(kl, 0.99),
            "draft.hidden.rel.p50": _quantile(draft_rel, 0.5),
            "draft.hidden.rel.p99": _quantile(draft_rel, 0.99),
            "draft.hidden.cosine.p1": _quantile(draft_cosine, 0.01),
        }
        bound = {**spec.NUMERICS.tolerances, **draft.NUMERICS.tolerances}
        # top1 and cosine are lower bounds, the others upper bounds
        lower = ("verify.logits.top1", "draft.hidden.cosine.p1")
        failed = {
            k: (v, bound[k])
            for k, v in measured.items()
            if (v < bound[k] if k in lower else v > bound[k])
        }
        self.assertEqual(failed, {}, f"measured {measured}")


if __name__ == "__main__":
    unittest.main()
