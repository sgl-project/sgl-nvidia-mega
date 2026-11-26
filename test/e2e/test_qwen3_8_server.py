# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Qwen3.8-Flash-Next with --enable-nvidia-megakernel against SGLang's own path (the
switch off, every other setting the same; qwen3_8_server.py) on one Rubin GPU (SM107),
TP1, MTP:

- accuracy: GSM8K through the chat template (thinking on, greedy, real acceptance) on
  the same questions. The megakernel must not be worse: no exact one-sided McNemar
  rejection at 0.05, and an accuracy drop of at most
  SGLANG_NVIDIA_MEGAKERNEL_E2E_GSM8K_MARGIN. Few-shot GSM8K gets an immediate end of
  turn from this model on either path.
- performance: a fixed load (PERF_CONCURRENCY requests of PERF_OUTPUT_TOKENS output
  tokens, the same requests every round) on servers with a fixed simulated acceptance
  length (PERF_ENV, measurement only: every verify still runs in full), so both arms
  run the same number of verify steps. Warm-up rounds run until two in a row agree
  within 1%, then PERF_ROUNDS measured rounds. The megakernel's median output
  throughput must not fall below SGLang's by more than
  SGLANG_NVIDIA_MEGAKERNEL_E2E_PERF_TOLERANCE; an arm whose middle three rounds spread
  more than PERF_NOISE fails as too noisy to judge, not as a regression.

One more server checks the stock step: with no slot on, each layer decomposed by the
executor (S1), every covered step against SGLang's forward, bit for bit (SGLANG_NVIDIA_MEGAKERNEL_PARITY=1), in SGLang's
deterministic inference: target and draft prefill, single- and multi-request verify,
draft decode and draft extend."""

import concurrent.futures as cf
import json
import math
import os
import re
import statistics
import tempfile
import time
import unittest
from pathlib import Path

import requests
import torch
from qwen3_8_server import ARGS, MODEL
from sglang.test.test_utils import (
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
    terminate_and_kill_process_tree,
)
from sglang.utils import download_and_cache_file

from sgl_nvidia_mega.plugin import FLAG_CLI
from sgl_nvidia_mega.qwen3_8.runtime import PARITY_ENV, SLOTS_ENV

LAUNCH_TIMEOUT = 3600
URL = DEFAULT_URL_FOR_TEST
GSM8K_URL = "https://raw.githubusercontent.com/openai/grade-school-math/master/grade_school_math/data/test.jsonl"
GSM8K_QUESTIONS = int(
    os.environ.get("SGLANG_NVIDIA_MEGAKERNEL_E2E_GSM8K_QUESTIONS", "200")
)
# Stock runs of these 200 questions have scored 195 to 197; slot sets within 2.5
# points of stock
GSM8K_MARGIN = float(
    os.environ.get("SGLANG_NVIDIA_MEGAKERNEL_E2E_GSM8K_MARGIN", "0.03")
)
GSM8K_BOUND = float(os.environ.get("SGLANG_NVIDIA_MEGAKERNEL_E2E_GSM8K_BOUND", "0.85"))
PERF_CONCURRENCY = 32
PERF_OUTPUT_TOKENS = 1024
PERF_ROUNDS = 5
PERF_WARMUP_ROUNDS = 8
# Measured rounds on one server spread up to 2.6% (SGLang) and 0.4% (megakernel), with
# an occasional SGLang round 6% slow: the middle three of five set the spread
PERF_NOISE = 0.05
PERF_TOLERANCE = float(
    os.environ.get("SGLANG_NVIDIA_MEGAKERNEL_E2E_PERF_TOLERANCE", "0.03")
)
SM107 = torch.cuda.is_available() and torch.cuda.get_device_capability() == (10, 7)
NEEDS = "Qwen3.8's megakernel is qualified on a Rubin GPU (SM107) only"


def _launch(extra_env, args, logs):
    # the product configuration: every slot the package ships, not a caller's override
    env = {
        **{k: v for k, v in os.environ.items() if k not in (SLOTS_ENV, PARITY_ENV)},
        **extra_env,
    }
    return popen_launch_server(
        MODEL,
        URL,
        timeout=LAUNCH_TIMEOUT,
        other_args=args,
        env=env,
        return_stdout_stderr=logs,
    )


def _gsm8k_rows():
    path = os.environ.get(
        "SGLANG_NVIDIA_MEGAKERNEL_E2E_GSM8K"
    ) or download_and_cache_file(GSM8K_URL)
    with open(path) as f:
        return [json.loads(line) for line in f]


def _number(text):
    # the last number after the reasoning
    found = re.findall(r"-?\d[\d,]*\.?\d*", text.rsplit("</think>", 1)[-1])
    try:
        return float(found[-1].replace(",", "").rstrip(".")) if found else None
    except ValueError:
        return None


def _gsm8k(rows):
    def ask(row):
        r = requests.post(
            URL + "/v1/chat/completions",
            json={
                "model": MODEL,
                "messages": [
                    {
                        "role": "user",
                        "content": row["question"]
                        + "\nSolve step by step, then give the final answer as a "
                        "number on the last line in the form '#### <number>'.",
                    }
                ],
                "temperature": 0,
                "max_tokens": 12288,
            },
            timeout=3600,
        )
        r.raise_for_status()
        pred = _number(r.json()["choices"][0]["message"].get("content") or "")
        gold = float(row["answer"].split("####")[-1].strip().replace(",", ""))
        return pred is not None and abs(pred - gold) < 1e-6

    with cf.ThreadPoolExecutor(PERF_CONCURRENCY) as pool:
        return list(pool.map(ask, rows))


def _throughput(rows):
    """Output tokens per second of one round: every request at once, fixed length."""

    def generate(i):
        r = requests.post(
            URL + "/generate",
            json={
                "text": rows[i]["question"],
                "sampling_params": {
                    "temperature": 0,
                    "max_new_tokens": PERF_OUTPUT_TOKENS,
                    "ignore_eos": True,
                },
            },
            timeout=1800,
        )
        r.raise_for_status()
        return r.json()["meta_info"]["completion_tokens"]

    start = time.perf_counter()
    with cf.ThreadPoolExecutor(PERF_CONCURRENCY) as pool:
        tokens = sum(pool.map(generate, range(PERF_CONCURRENCY)))
    return tokens / (time.perf_counter() - start)


# Measurement only: a fixed acceptance length with the draft's own tokens
PERF_ENV = {
    "SGLANG_SIMULATE_ACC_LEN": "2.32",
    "SGLANG_SIMULATE_ACC_TOKEN_MODE": "real-draft-token",
}
_SERVERS = {}


def _server(name, workload):
    """Launch one arm's server for one workload once; keep its result and log."""
    key = (name, workload)
    if key in _SERVERS:
        return _SERVERS[key]
    mega = name == "megakernel"
    perf = workload == "performance"
    rows = _gsm8k_rows()
    logs = tuple(tempfile.NamedTemporaryFile("w+", suffix=s) for s in (".out", ".err"))
    process = _launch(
        PERF_ENV if perf else {},
        ARGS + ([FLAG_CLI] if mega else []),
        logs,
    )
    try:
        if not perf:
            result = _gsm8k(rows[:GSM8K_QUESTIONS])
        else:
            last = _throughput(rows)
            for _ in range(PERF_WARMUP_ROUNDS):
                now = _throughput(rows)
                if abs(now - last) <= 0.01 * last:
                    break
                last = now
            else:
                raise AssertionError(
                    f"{name}: no two warm-up rounds within 1% in "
                    f"{PERF_WARMUP_ROUNDS}: too noisy to judge, not a regression"
                )
            result = [_throughput(rows) for _ in range(PERF_ROUNDS)]
    finally:
        terminate_and_kill_process_tree(process)
        log = "".join(Path(f.name).read_text(errors="replace") for f in logs)
        for f in logs:
            f.close()
    print(f"{name} {workload}: {sum(result) if not perf else result}", flush=True)
    _SERVERS[key] = result, log
    return _SERVERS[key]


@unittest.skipUnless(SM107, NEEDS)
class TestMegakernelAgainstSGLang(CustomTestCase):
    def _arms(self, workload):
        (stock, stock_log), (mega, mega_log) = (
            _server("sglang", workload),
            _server("megakernel", workload),
        )
        # the candidate ran the megakernel, the baseline did not
        self.assertNotIn("NVIDIA megakernel", stock_log)
        self.assertIn("NVIDIA megakernel: active", mega_log)
        engaged = re.findall(
            r"NVIDIA megakernel engagement: \S+ target_verify rows=\d+ graph: (.*)",
            mega_log,
        )
        self.assertTrue(
            any(re.search(r"=(?!stock\b)\w", ops) for ops in engaged),
            "no captured target verify ran a plugin kernel",
        )
        return stock, mega

    def test_accuracy_against_sglang(self):
        stock, mega = self._arms("accuracy")
        n = len(stock)
        worse = sum(s and not m for s, m in zip(stock, mega))
        better = sum(m and not s for s, m in zip(stock, mega))
        # exact one-sided McNemar: P(at least `worse` of the discordant pairs)
        p = sum(math.comb(worse + better, k) for k in range(worse, worse + better + 1))
        p = p / 2 ** (worse + better) if worse + better else 1.0
        acc_stock, acc_mega = sum(stock) / n, sum(mega) / n
        print(
            f"gsm8k: sglang {acc_stock:.4f}, megakernel {acc_mega:.4f} "
            f"({acc_mega - acc_stock:+.4f}); discordant {worse} worse / {better} "
            f"better, McNemar p={p:.3f}"
        )
        self.assertGreater(acc_stock, GSM8K_BOUND)
        self.assertGreater(acc_mega, GSM8K_BOUND)
        self.assertGreaterEqual(p, 0.05, "the megakernel answers worse than SGLang")
        self.assertGreaterEqual(acc_mega, acc_stock - GSM8K_MARGIN)

    def test_performance_against_sglang(self):
        rounds = dict(zip(("sglang", "megakernel"), self._arms("performance")))
        median = {name: statistics.median(v) for name, v in rounds.items()}
        middle = {name: sorted(v)[1:-1] for name, v in rounds.items()}
        spread = {name: (v[-1] - v[0]) / median[name] for name, v in middle.items()}
        change = median["megakernel"] / median["sglang"] - 1
        print(
            f"output tok/s at {PERF_CONCURRENCY} x {PERF_OUTPUT_TOKENS}, median of "
            f"{PERF_ROUNDS}: sglang {median['sglang']:.1f}, megakernel "
            f"{median['megakernel']:.1f} ({change:+.2%}); spread "
            + ", ".join(f"{name} {s:.2%}" for name, s in spread.items())
        )
        noisy = [f"{name} {s:.2%}" for name, s in spread.items() if s > PERF_NOISE]
        if noisy:
            self.fail(
                f"rounds spread above {PERF_NOISE:.0%} ({', '.join(noisy)}): "
                "too noisy to judge, not a regression"
            )
        self.assertGreaterEqual(
            change,
            -PERF_TOLERANCE,
            f"the megakernel is {-change:.2%} slower than SGLang",
        )


# SGLang's deterministic inference, which makes its own QSA top-k ties reproducible
DETERMINISTIC_ARGS = [
    "--enable-deterministic-inference",
    "--attention-backend",
    "triton",
    "--linear-attn-prefill-backend",
    "triton",
    "--linear-attn-decode-backend",
    "triton",
]
PARITY = re.compile(
    r"NVIDIA megakernel parity (target|draft) (eager|graph) (\w+) rows=(\d+) "
    r"steps=(\d+) logits_differ=(\d+) hidden_differ=(\d+) logits_compared=(\d+) "
    r"hidden_compared=(\d+)"
)


@unittest.skipUnless(SM107, NEEDS)
class TestStockStepIsSGLangs(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        cls.logs = tuple(
            tempfile.NamedTemporaryFile("w+", suffix=s) for s in (".out", ".err")
        )
        cls.process = _launch(
            {SLOTS_ENV: "", PARITY_ENV: "1"},
            ARGS + DETERMINISTIC_ARGS + [FLAG_CLI],
            cls.logs,
        )

    @classmethod
    def tearDownClass(cls):
        terminate_and_kill_process_tree(cls.process)
        for log in cls.logs:
            log.close()

    def _generate(self, i):
        r = requests.post(
            URL + "/generate",
            json={
                "text": f"Question {i}: explain step by step whether {i} * 17 + 3 is odd.",
                "sampling_params": {"temperature": 0, "max_new_tokens": 64},
            },
            timeout=600,
        )
        r.raise_for_status()

    def test_every_covered_step_is_bitwise(self):
        for i in range(4):
            self._generate(i)
        with cf.ThreadPoolExecutor(8) as pool:
            list(pool.map(self._generate, range(4, 12)))
        # the last request's prefill logs what the earlier replays added
        self._generate(12)
        text = "".join(Path(f.name).read_text(errors="replace") for f in self.logs)
        lines = PARITY.findall(text)
        # the server's start-up request: two stock forwards of it differ as well
        first = [line[:3] for line in lines].index(("target", "eager", "prefill"))
        print(f"start-up prefill, not compared: {lines[first]}")
        checked = lines[:first] + lines[first + 1 :]
        for model, kind, mode in sorted({line[:3] for line in lines}):
            same = [line for line in lines if line[:3] == (model, kind, mode)]
            print(
                f"parity {model} {kind} {mode}: {sum(int(x[4]) for x in same)} steps, "
                f"{sum(int(x[5]) for x in same)} logits differ"
            )
        verify_rows = {
            int(line[3])
            for line in checked
            if line[:3] == ("target", "graph", "target_verify")
        }
        self.assertTrue(verify_rows, "no captured target verify was compared")
        self.assertGreater(max(verify_rows), 4, "no multi-request verify was compared")
        self.assertIn(("target", "eager", "prefill"), [line[:3] for line in checked])
        self.assertTrue(
            any(line[0] == "draft" for line in checked), "no draft step was compared"
        )
        differ = [line for line in checked if int(line[5]) or int(line[6])]
        self.assertEqual(differ, [], f"{len(differ)} parity lines differ")
        self.assertGreater(sum(int(line[7]) for line in checked), 0)


if __name__ == "__main__":
    unittest.main()
