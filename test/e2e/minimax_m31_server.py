# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""The MiniMax-M3.1-Flash servers the end-to-end tests launch, and what the tests
share: GB300, TP4, NVFP4 checkpoint, DSpark (verify steps of 8 rows a request, draft
blocks of 7), 1-16 running requests in one server.

The accuracy and the performance test (test_minimax_m31_acc.py, test_minimax_m31_perf.py)
compare two arms on the same GPUs, one after the other: the baseline, native SGLang
(``serve(megakernel=False)``: the launch below without the flag), and the candidate,
the same launch with --enable-nvidia-megakernel. Both arms get the same arguments and
the same environment, so they differ only by the flag; an arm is torn down, and its GPU
memory released, before the next one launches. ``Server.reasons()`` reads an arm's log
for the path it ran (minimax_m31_verdict.py).

The environment of both arms:
- SGLANG_PLUGINS=nvidia_megakernel, SGLang's general-plugin whitelist. SGLang's cu134
  image also installs MSA's own SGLang plugin, and SGLang gives --enable-nvidia-megakernel
  to it for this architecture when it is loaded; the whitelist keeps it unloaded, so this
  package's specs serve the flag and the baseline is SGLang's own forward;
- SGLANG_FORCE_COARSE_WAR_BARRIER=1: MegaAttention keeps all its CTAs resident, so no
  kernel of the next batch may run beside it;
- no SGLANG_SIMULATE_ACC_* from the caller, which ``serve`` refuses: accuracy runs on
  real speculative acceptance, and the performance test names its own measurement-only
  override.

What the caller sets:
- SGLANG_NVIDIA_MEGAKERNEL_E2E_MODEL_PATH: the checkpoint (default: the Hub id);
- SGLANG_NVIDIA_MEGAKERNEL_E2E_EXTRA_ARGS: more server arguments, for both arms;
- SGLANG_NVIDIA_MEGAKERNEL_E2E_GSM8K: a local GSM8K test.jsonl (default: downloaded).
SGLANG_NVIDIA_MEGAKERNEL_E2E_URL is not read: a server that is already up is one arm,
and the two arms cannot share the GPUs, so these tests launch both themselves."""

import contextlib
import os
import random
import re
import tempfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import minimax_m31_verdict as verdict
import requests
from sglang.test.test_utils import (
    DEFAULT_URL_FOR_TEST,
    popen_launch_server,
    terminate_and_kill_process_tree,
)
from sglang.utils import download_and_cache_file, read_jsonl

from sgl_nvidia_mega.minimax_m31 import MAX_REQUESTS, draft, spec
from sgl_nvidia_mega.plugin import FLAG_CLI

MODEL = os.environ.get(
    "SGLANG_NVIDIA_MEGAKERNEL_E2E_MODEL_PATH", "MiniMaxAI/MiniMax-M3.1-Flash"
)
TP = 4
ENV = {
    "SGLANG_PLUGINS": "nvidia_megakernel",
    "SGLANG_FORCE_COARSE_WAR_BARRIER": "1",
}
# Measurement-only overrides a launch must not inherit from the caller's environment
SIMULATED_ACCEPTANCE = "SGLANG_SIMULATE_ACC_"
ARGS = [
    "--trust-remote-code",
    "--tp",
    str(TP),
    "--moe-runner-backend",
    "flashinfer_trtllm_routed",
    "--page-size",
    "128",
    "--speculative-algorithm",
    "DSPARK",
    "--speculative-use-block-verification",
    "--speculative-num-steps",
    "1",
    "--speculative-eagle-topk",
    "1",
    "--speculative-num-draft-tokens",
    "8",
    "--speculative-dspark-block-size",
    "7",
    "--disable-attn-tp-gather",
    "--max-running-requests",
    str(MAX_REQUESTS),
    "--cuda-graph-max-bs-decode",
    str(MAX_REQUESTS),
    "--mem-fraction-static",
    "0.8",
    "--reasoning-parser",
    "minimax-m3",
    "--watchdog-timeout",
    "600",
] + os.environ.get("SGLANG_NVIDIA_MEGAKERNEL_E2E_EXTRA_ARGS", "").split()
LAUNCH_TIMEOUT = 3600
# The arms in launch order: name -> whether the launch has the flag
BASELINE, CANDIDATE = "native SGLang", "megakernel"
ARMS = {BASELINE: False, CANDIDATE: True}
# What the candidate's log must show (verdict.candidate_reasons): the verify spec runs
# the decode graphs of every captured batch size up to 16, the draft spec batch 1
SPECS = {spec.SPEC.name: (1, 2, MAX_REQUESTS), draft.SPEC.name: (1,)}
CASE = "MiniMax-M3.1-Flash, 4 x GB300, TP4, NVFP4, DSpark block 7"


class Server:
    """One launched arm: its URL, its process and its log (kept after teardown)."""

    def __init__(self, megakernel: bool, logs):
        self.url, self.megakernel, self.process = DEFAULT_URL_FOR_TEST, megakernel, None
        self._logs, self._log = logs, None

    def log(self) -> str:
        if self._log is not None:
            return self._log
        return "".join(Path(f.name).read_text(errors="replace") for f in self._logs)

    def reasons(self) -> list:
        """Why the log does not show the path this arm claims ([] when it does)."""
        if self.megakernel:
            return verdict.candidate_reasons(self.log(), TP, SPECS)
        return verdict.baseline_reasons(self.log(), TP)

    def close(self) -> None:
        if self.process is not None:
            terminate_and_kill_process_tree(self.process)
        self._log = self.log()
        for log in self._logs:
            log.close()


@contextlib.contextmanager
def serve(megakernel: bool, extra_env=None, extra_args=()):
    """Launch one arm and tear it down on exit; a launch that fails raises."""
    # popen_launch_server hands the server this process's environment too
    inherited = sorted(k for k in os.environ if k.startswith(SIMULATED_ACCEPTANCE))
    if inherited:
        raise RuntimeError(
            f"unset {inherited}: accuracy runs on real acceptance, and the "
            "performance test sets its own measurement-only override on both arms"
        )
    env = {**os.environ, **ENV, **(extra_env or {})}
    server = Server(
        megakernel,
        tuple(tempfile.NamedTemporaryFile("w+", suffix=s) for s in (".out", ".err")),
    )
    try:
        server.process = popen_launch_server(
            MODEL,
            server.url,
            timeout=LAUNCH_TIMEOUT,
            other_args=ARGS + ([FLAG_CLI] if megakernel else []) + list(extra_args),
            env=env,
            return_stdout_stderr=server._logs,
        )
        yield server
    finally:
        server.close()


def engagement_failures(servers: dict) -> dict:
    """Arm name -> why its log does not show its path; {} when every arm's does."""
    return {name: why for name, s in servers.items() if (why := s.reasons())}


GSM8K_URL = (
    "https://raw.githubusercontent.com/openai/grade-school-math/master/"
    "grade_school_math/data/test.jsonl"
)
NUMBER = re.compile(r"-?\d[\d,]*\.?\d*")


def questions(n: int) -> list:
    """The first n GSM8K test questions, the same for every arm."""
    path = os.environ.get(
        "SGLANG_NVIDIA_MEGAKERNEL_E2E_GSM8K"
    ) or download_and_cache_file(GSM8K_URL)
    rows = list(read_jsonl(path))[:n]
    if n < 1 or len(rows) < n:
        raise ValueError(f"{path} holds {len(rows)} of the {n} questions asked for")
    return rows


def chat(url: str, text: str, max_tokens: int) -> str:
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


def ask(url: str, row: dict) -> str:
    return chat(
        url, row["question"] + "\nGive the final answer as a number at the end.", 4096
    )


def correct(url: str, row: dict) -> bool:
    want = float(row["answer"].split("####")[-1].strip().replace(",", ""))
    found = NUMBER.findall(ask(url, row))
    return bool(found) and float(found[-1].replace(",", "").rstrip(".")) == want


def gsm8k(url: str, n: int, threads: int) -> list:
    """Chat GSM8K, greedy: per question, in order, whether the reply's last number is
    the answer; ``threads`` questions at a time."""
    with ThreadPoolExecutor(threads) as pool:
        return list(pool.map(lambda row: correct(url, row), questions(n)))


def passkeys(url: str) -> list:
    """Three pass keys, each inside about 13K tokens of filler, where the sparse
    attention selects 16 pages of 103: per key, (prompt tokens, whether the reply
    holds it). The keys and their places are seeded, so every arm gets the same."""
    rng, found = random.Random(1), []
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
            url + "/v1/completions",
            json={"model": "default", "prompt": prompt, "max_tokens": 64,
                  "temperature": 0},
            timeout=900,
        )  # fmt: skip
        r.raise_for_status()
        out = r.json()
        found.append(
            (out["usage"]["prompt_tokens"], str(key) in out["choices"][0]["text"])
        )
    return found
