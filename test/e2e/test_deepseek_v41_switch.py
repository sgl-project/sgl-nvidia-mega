# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""The switch on GB300 TP4 with DSpark: its stock step against SGLang's.

``TestStockStepIsSGLangs``: with every slot off, the covered step runs each package's
stock code, which must reproduce SGLang's decode step exactly.
``SGLANG_NVIDIA_MEGAKERNEL_PARITY`` (test only) runs SGLang's forward and then the
megakernel's on every eager verify step and logs the largest difference of the logits
and the hidden states; the sparse indices are sorted in both (their atomic emission
order is not deterministic). Batches of 1, 2, 4 and 8 requests cover 6, 12, 24 and 48
rows.
"""

import concurrent.futures
import os
import re
import tempfile
import unittest
from pathlib import Path

import requests
from deepseek_v41_server import ARGS, ENV, MODEL, URL
from sglang.test.test_utils import (
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
    terminate_and_kill_process_tree,
)

from sgl_nvidia_mega.plugin import FLAG_CLI

LAUNCH_TIMEOUT = 3600
PARITY_ENV = {
    "SGLANG_NVIDIA_MEGAKERNEL_SLOTS": "",
    "SGLANG_NVIDIA_MEGAKERNEL_PARITY": "1",
    "SGLANG_NVIDIA_MEGAKERNEL_PARITY_SORT_TOPK": "1",
}
LINE = re.compile(
    r"NVIDIA megakernel parity \S+ rows=(\d+) logits_max_abs=(\S+) .*hidden_max_abs=(\S+)"
)
# a difference renders as "<max> (rel ...)"; equal tensors as "0 (rel ...)", absent as "n/a"
EQUAL = ("0", "n/a")


@unittest.skipIf(URL, "launches its own server with the parity switch")
class TestStockStepIsSGLangs(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        cls.logs = tuple(
            tempfile.NamedTemporaryFile("w+", suffix=s) for s in (".out", ".err")
        )
        cls.process = popen_launch_server(
            MODEL,
            DEFAULT_URL_FOR_TEST,
            timeout=LAUNCH_TIMEOUT,
            # eager verify steps: the parity check runs outside CUDA graphs only
            other_args=ARGS + [FLAG_CLI, "--disable-decode-cuda-graph"],
            env={**os.environ, **ENV, **PARITY_ENV},
            return_stdout_stderr=cls.logs,
        )

    @classmethod
    def tearDownClass(cls):
        terminate_and_kill_process_tree(cls.process)
        for log in cls.logs:
            log.close()

    def _generate(self, i: int) -> str:
        r = requests.post(
            DEFAULT_URL_FOR_TEST + "/generate",
            json={
                "text": f"Question {i}: explain step by step whether {i} * 17 + 3 is odd.",
                "sampling_params": {"temperature": 0, "max_new_tokens": 48},
            },
            timeout=600,
        )
        r.raise_for_status()
        return r.json()["text"]

    def test_every_covered_width_is_bitwise(self):
        for batch in (1, 2, 4, 8):
            with concurrent.futures.ThreadPoolExecutor(batch) as pool:
                list(pool.map(self._generate, range(batch)))
        text = "".join(Path(f.name).read_text() for f in self.logs)
        found = LINE.findall(text)
        self.assertTrue(found, "no parity line: the covered steps did not run")
        diffs = [line for line in found if line[1] not in EQUAL or line[2] not in EQUAL]
        self.assertEqual(diffs, [], f"{len(diffs)} of {len(found)} steps differ")
        self.assertLessEqual({6, 12, 24, 48}, {int(rows) for rows, *_ in found})


if __name__ == "__main__":
    unittest.main()
