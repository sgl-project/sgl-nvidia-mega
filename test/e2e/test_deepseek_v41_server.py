# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""DeepSeek-V4.1-Flash with --enable-nvidia-megakernel on GB300, TP4, DSpark, across
the switch point of the post hops: up to 8 requests (48 rows) the routed-expert pair
and SGLang's fused posts, from 9 requests (54 rows) the two-shot hops; 12 and 16
requests step across it and back: GSM8K runs across the switch point (gross
failures).

SGLANG_NVIDIA_MEGAKERNEL_E2E_URL runs the serving tests against a server that is already up, so
the same checks also score the switch-off arm."""

import os
import re
import tempfile
import unittest
from types import SimpleNamespace

from deepseek_v41_server import ARGS, ENV, MODEL, URL
from sglang.test.few_shot_gsm8k import run_eval as run_eval_few_shot_gsm8k
from sglang.test.test_utils import (
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
    terminate_and_kill_process_tree,
)

from sgl_nvidia_mega.deepseek_v41.runtime import SLOTS_ENV
from sgl_nvidia_mega.plugin import FLAG_CLI

LAUNCH_TIMEOUT = 3600
# Few-shot GSM8K-200 (8 shots, 512 new tokens), for gross failures only; compare with
# the switch-off arm's own band for anything finer
GSM8K_BOUND = float(os.environ.get("SGLANG_NVIDIA_MEGAKERNEL_E2E_GSM8K_BOUND", "0.85"))
URL_ = URL or DEFAULT_URL_FOR_TEST


def _launch(args=ARGS, extra_env=None, extra_args=(), log=None, backend=True):
    # the product configuration: every slot the package ships, not a caller's override
    env = {
        **{k: v for k, v in os.environ.items() if k != SLOTS_ENV},
        **ENV,
        **(extra_env or {}),
    }
    return popen_launch_server(
        MODEL,
        DEFAULT_URL_FOR_TEST,
        timeout=LAUNCH_TIMEOUT,
        other_args=args + ([FLAG_CLI] if backend else []) + list(extra_args),
        env=env,
        return_stdout_stderr=log,
    )


def _gsm8k(parallel):
    host, port = re.match(r"https?://([^:/]+):(\d+)", URL_).groups()
    args = SimpleNamespace(
        num_shots=8,
        data_path=os.environ.get("SGLANG_NVIDIA_MEGAKERNEL_E2E_GSM8K"),
        num_questions=200,
        parallel=parallel,
        max_new_tokens=512,
        host=host,
        port=int(port),
    )
    metrics = run_eval_few_shot_gsm8k(args)
    print(f"gsm8k parallel={parallel} {metrics=}")
    return metrics


class TestMegakernelVerify(CustomTestCase):
    """DSpark verify steps of 1-64 requests (6-384 rows) run the plugin."""

    @classmethod
    def setUpClass(cls):
        cls.process, cls.logs = None, ()
        if URL:
            return
        cls.logs = tuple(
            tempfile.NamedTemporaryFile("w+", suffix=s) for s in (".out", ".err")
        )
        cls.process = _launch(extra_args=["--watchdog-timeout", "600"], log=cls.logs)

    @classmethod
    def tearDownClass(cls):
        if cls.process is not None:
            terminate_and_kill_process_tree(cls.process)
        for log in cls.logs:
            log.close()

    def test_gsm8k_across_the_two_shot_switch(self):
        # 12 and 16 requests: batches move between <= 8 (routed pair, SGLang's fused
        # posts) and >= 9 (two-shot hops)
        scores = {p: _gsm8k(p)["accuracy"] for p in (12, 16)}
        for parallel, accuracy in scores.items():
            self.assertGreater(accuracy, GSM8K_BOUND, f"parallel {parallel}")


if __name__ == "__main__":
    unittest.main()
