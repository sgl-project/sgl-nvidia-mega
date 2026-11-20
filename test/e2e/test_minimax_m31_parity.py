# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""The composition check of MiniMax-M3.1-Flash with --enable-nvidia-megakernel on
GB300, TP4, DSpark: each covered step against SGLang's forward on the same step.

With SGLANG_NVIDIA_MEGAKERNEL_PARITY=1 (test only) and eager decode steps, the server
runs SGLang's forward and then the megakernel's on every covered step and logs their
difference: the 8-row steps (slot mega) and the draft blocks within the tolerances the
specs declare, the 16-128-row steps (slots moe_nofill and moe_swap) bitwise.

It supplements the task accuracy test, test_minimax_m31_acc.py, and does not replace
it: this file compares single steps on one launch, with the parity switch and without
CUDA graphs, where that one compares the answers of the product launch with native
SGLang's. It has its own file because it needs that third, different launch, and so
that a workflow can select the accuracy / performance pair without it."""

import contextlib
import re
import statistics
import unittest
from concurrent.futures import ThreadPoolExecutor

from minimax_m31_server import ask, questions, serve
from sglang.test.test_utils import CustomTestCase

from sgl_nvidia_mega.minimax_m31 import draft, spec

PARITY_LINE = re.compile(
    r"NVIDIA megakernel parity (target logits|draft hidden) rows=\d+ max_abs=\S+ "
    r"rel=(\S+) min_cosine=(\S+) top1=(\S+) max_kl=(\S+)"
)
# the bitwise widths (16-128 rows): logits and hidden states against SGLang's forward
BITWISE_LINE = re.compile(
    r"NVIDIA megakernel parity target_verify rows=(\d+) logits_max_abs=(\S+)"
    r".*? hidden_max_abs=(\S+)"
)


def _quantile(values, q):
    values = sorted(values)
    return values[min(len(values) - 1, int(q * len(values)))]


class TestParity(CustomTestCase):
    """Eager covered steps, each after SGLang's forward on the same step."""

    @classmethod
    def setUpClass(cls):
        cls.stack = contextlib.ExitStack()
        cls.server = cls.stack.enter_context(
            serve(
                megakernel=True,
                extra_env={spec.PARITY_ENV: "1"},
                extra_args=[
                    "--disable-decode-cuda-graph",
                    "--disable-prefill-cuda-graph",
                ],
            )
        )
        cls.process, cls.url = cls.server.process, cls.server.url

    @classmethod
    def tearDownClass(cls):
        cls.stack.close()

    def test_the_covered_steps_stay_within_the_declared_tolerances(self):
        # the workload the tolerances were measured on, one request at a time; its
        # lines are read before the 16-128-row workload below adds 8-row steps of its own
        for row in questions(20):
            ask(self.url, row)
        sequential = self.server.log()
        with ThreadPoolExecutor(16) as pool:
            list(pool.map(lambda row: ask(self.url, row), questions(32)))
        lines = {"target logits": [], "draft hidden": []}
        for what, *values in PARITY_LINE.findall(sequential):
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
        # SGLang's own output holds no logits on an occasional step (n/a): not compared
        wide = [
            (int(rows), logits, hidden)
            for rows, logits, hidden in BITWISE_LINE.findall(self.server.log())
            if int(rows) > 8 and logits != "n/a"
        ]
        self.assertGreater(len({rows for rows, _, _ in wide}), 1)
        self.assertEqual(
            [line for line in wide if line[1:] != ("0", "0")], [], "not bitwise"
        )


if __name__ == "__main__":
    unittest.main()
