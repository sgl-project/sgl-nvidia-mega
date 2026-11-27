# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""The switch on GB300 TP4 with MTP (6 rows a request): its stock step against SGLang's.

``TestStockStepIsSGLangs``: with every slot off (``SGLANG_NVIDIA_MEGAKERNEL_SLOTS=""``)
every verify width the TP4 profile accepts runs the executor's layer loop on the
stock step bodies (``runtime/composed.py``), which must reproduce SGLang's verify
step exactly. ``SGLANG_NVIDIA_MEGAKERNEL_PARITY=1`` (test only) runs SGLang's forward
and then the stock step on every eager verify step of a covered width and logs the
largest difference of the logits (``runtime/parity.py``); the spec admits eager
decode steps only under that switch. Batches of 1, 2, 4 and 8 requests cover 6, 12,
24 and 48 rows. The NextN draft stays SGLang's here (slot ``draft`` is off).
"""

import concurrent.futures
import os
import re
import tempfile
import unittest
from pathlib import Path

import requests
from glm5_3_server import ARGS, COVERED_REQUESTS, ENV, MODEL, TOKENS, URL
from sglang.test.test_utils import (
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
    terminate_and_kill_process_tree,
)

from sgl_nvidia_mega.glm5_3.runtime import SLOTS_ENV
from sgl_nvidia_mega.glm5_3.runtime.parity import PARITY_ENV
from sgl_nvidia_mega.plugin import FLAG_CLI

LAUNCH_TIMEOUT = 3600
PARITY = {SLOTS_ENV: "", PARITY_ENV: "1"}
# runtime/parity.py run(): "<max> (rel ..., ... differ)" for a difference, "0 (rel
# ...)" for equal logits, "n/a" when neither forward returned logits
LINE = re.compile(
    r"NVIDIA megakernel parity target_verify rows=(\d+) logits_max_abs=(\S+)"
)
EQUAL = ("0", "n/a")
BATCHES = (1, 2, 4, COVERED_REQUESTS)


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
            env={**os.environ, **ENV, **PARITY},
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
        for batch in BATCHES:
            with concurrent.futures.ThreadPoolExecutor(batch) as pool:
                list(pool.map(self._generate, range(batch)))
        text = "".join(Path(f.name).read_text(errors="replace") for f in self.logs)
        found = LINE.findall(text)
        self.assertTrue(found, "no parity line: the covered steps did not run")
        diffs = [line for line in found if line[1] not in EQUAL]
        self.assertEqual(diffs, [], f"{len(diffs)} of {len(found)} steps differ")
        # at least one step compared real logits (not only n/a)
        self.assertIn("0", {line[1] for line in found})
        self.assertLessEqual(
            {TOKENS * b for b in BATCHES}, {int(rows) for rows, _ in found}
        )


if __name__ == "__main__":
    unittest.main()
