# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""The megakernel's stock step against SGLang's, bit for bit, on GB300 TP8 with MTP, in
eager verify steps and inside the decode CUDA graphs.

With every slot off, the covered step runs each package's stock code, which must
reproduce SGLang's verify step exactly; a bitwise slot must too, so
``SGLANG_NVIDIA_MEGAKERNEL_E2E_SLOTS`` (e.g. ``route``) runs the same check with those
slots on. ``SGLANG_NVIDIA_MEGAKERNEL_PARITY=1`` (test
only) runs SGLang's forward and then the megakernel's on every covered step and
compares the logits and the hidden rows bit for bit: an eager step logs it at once; a
captured step records both forwards and the comparison, and the next eager forward (a
request's prefill) logs what the replays added. Requests run one at a time, so every
verify step is the covered width (7 rows).

A server launched elsewhere (TP8 spans two GB300 trays) is read through
``SGLANG_NVIDIA_MEGAKERNEL_E2E_URL`` and its rank logs
(``SGLANG_NVIDIA_MEGAKERNEL_E2E_LOGS``); it must run with the parity switch and the
slots under test in ``SGLANG_NVIDIA_MEGAKERNEL_SLOTS``.
"""

import os
import re
import tempfile
import unittest
from pathlib import Path

import requests
from qwen3_5_server import ARGS, ENV, LOGS, MODEL, URL
from sglang.test.test_utils import (
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
    terminate_and_kill_process_tree,
)

from sgl_nvidia_mega.plugin import FLAG_CLI

LAUNCH_TIMEOUT = 3600
PARITY_ENV = {
    "SGLANG_NVIDIA_MEGAKERNEL_SLOTS": os.environ.get(
        "SGLANG_NVIDIA_MEGAKERNEL_E2E_SLOTS", ""
    ),
    "SGLANG_NVIDIA_MEGAKERNEL_PARITY": "1",
}
# an eager step on padding only (valid_rows=0) writes no output rows, so it is not
# compared; SGLang's graph-capture warmups log as capture-warmup (runtime/parity.py)
EAGER = re.compile(
    r"NVIDIA megakernel parity eager target_verify rows=(\d+) "
    r"logits_differ=(\d+) hidden_differ=(\d+) "
    r"logits_compared=(\d+) hidden_compared=(\d+) .*valid_rows=[1-9]"
)
GRAPH = re.compile(
    r"NVIDIA megakernel parity graph target_verify rows=(\d+) steps=(\d+) "
    r"logits_differ=(\d+) hidden_differ=(\d+) "
    r"logits_compared=(\d+) hidden_compared=(\d+)"
)
REQUESTS = 6


class _StockParity(CustomTestCase):
    """One server: its covered steps compared bit for bit."""

    EXTRA_ARGS: list = []

    @classmethod
    def setUpClass(cls):
        cls.process, cls.url = None, URL or DEFAULT_URL_FOR_TEST
        if URL:
            cls.log_paths = LOGS
            return
        cls.logs = tuple(
            tempfile.NamedTemporaryFile("w+", suffix=s) for s in (".out", ".err")
        )
        cls.log_paths = [f.name for f in cls.logs]
        cls.process = popen_launch_server(
            MODEL,
            cls.url,
            timeout=LAUNCH_TIMEOUT,
            other_args=ARGS + [FLAG_CLI, *cls.EXTRA_ARGS],
            env={**os.environ, **ENV, **PARITY_ENV},
            return_stdout_stderr=cls.logs,
        )

    @classmethod
    def tearDownClass(cls):
        if cls.process is not None:
            terminate_and_kill_process_tree(cls.process)
            for log in cls.logs:
                log.close()

    def _generate(self, i: int) -> str:
        r = requests.post(
            self.url + "/generate",
            json={
                "text": f"Question {i}: explain step by step whether {i} * 17 + 3 is odd.",
                "sampling_params": {"temperature": 0, "max_new_tokens": 64},
            },
            timeout=600,
        )
        r.raise_for_status()
        return r.json()["text"]

    def _parity_lines(self):
        # the last request's prefill logs what the earlier requests' replays added
        for i in range(REQUESTS + 1):
            self._generate(i)
        text = "".join(Path(p).read_text(errors="replace") for p in self.log_paths)
        return EAGER.findall(text), GRAPH.findall(text)

    def check(self, graph):
        """graph: the lines of captured steps (True), of eager ones (False), or
        either (None, for a server launched elsewhere)."""
        eager, graphs = self._parity_lines()
        found = {True: graphs, False: eager, None: eager + graphs}[graph]
        self.assertTrue(found, "no parity line: the covered steps did not run")
        differ = [line for line in eager + graphs if any(int(n) for n in line[-4:-2])]
        self.assertEqual(differ, [], f"{len(differ)} parity lines differ")
        self.assertEqual({int(line[0]) for line in found}, {7})
        # an autotune dummy run returns no logits: compare real ones too
        self.assertGreater(sum(int(line[-2]) for line in found), 0)
        if graphs:
            self.assertGreater(sum(int(line[1]) for line in graphs), 0)


@unittest.skipIf(URL, "launches its own server with the parity switch")
class TestEagerStockStepIsSGLangs(_StockParity):
    EXTRA_ARGS = ["--disable-decode-cuda-graph"]

    def test_every_eager_covered_step_is_bitwise(self):
        self.check(graph=False)


@unittest.skipIf(URL, "launches its own server with the parity switch")
class TestGraphStockStepIsSGLangs(_StockParity):
    def test_every_captured_covered_step_is_bitwise(self):
        self.check(graph=True)


@unittest.skipUnless(URL, "reads a server launched with the parity switch")
class TestRunningServerStockStepIsSGLangs(_StockParity):
    def test_every_covered_step_is_bitwise(self):
        self.check(graph=None)


if __name__ == "__main__":
    unittest.main()
