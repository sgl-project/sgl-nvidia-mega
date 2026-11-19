# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the SGLang project
"""Kimi K3.1 split DSA selection on CPU: the merge storage comes from SGLang's
``DcpTopKMerge.create`` (which may decline) and the replicated split launch gets
that storage."""

import ast
import sys
import types
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

ROOT = Path(__file__).resolve().parents[2] / "sgl_nvidia_mega/kimi_k31/dsa"
# split.py imports its kernel module as ..sm103.dsa.split_merge
KERNEL = "split_test.sm103.dsa.split_merge"


class SplitMergeCompatibility(unittest.TestCase):
    def load_split(self):
        node = next(
            n
            for n in ast.parse((ROOT / "split.py").read_text()).body
            if isinstance(n, ast.ClassDef) and n.name == "SplitMerge"
        )
        scope = {"TOPK": 2048, "PAGE": 64, "__package__": "split_test.dsa"}
        exec(
            compile(ast.Module(body=[node], type_ignores=[]), "split.py", "exec"), scope
        )
        return scope["SplitMerge"]

    def test_factory_storage_and_replicated_kernel(self):
        for cand_k in (256, 384):
            factory_calls, launch_calls = [], []
            storage = types.SimpleNamespace(
                buffer=torch.zeros(2 * 8 * (16 * cand_k + 2048), dtype=torch.int64),
                multicast_ptr=123,
                peer_ptrs=torch.arange(16),
                counters=torch.zeros(8),
                union_scratch=torch.empty(8, 16 * 2048),
                fallback=torch.zeros(8),
                dcp_rank=5,
                usable=lambda rows: rows <= 8,
            )

            def create(*args):
                factory_calls.append(args)
                return storage

            host = types.ModuleType("sglang.srt.layers.attention.dsa.dsa_dcp_sharded")
            host.DcpTopKMerge = types.SimpleNamespace(create=create)
            kernel = types.ModuleType(KERNEL)
            kernel.dcp_topk_merge = lambda *a, **k: launch_calls.append((a, k))
            with patch.dict(
                sys.modules, {host.__name__: host, kernel.__name__: kernel}
            ):
                group = types.SimpleNamespace(rank_in_group=5, world_size=16)
                merge = self.load_split()(group, 8)
                self.assertEqual(factory_calls, [(group, 8, 2048, 5)])
                self.assertTrue(merge.usable(8))
                self.assertFalse(merge.usable(9))
                out = torch.empty(4, 2048, dtype=torch.int32)
                self.assertIs(merge.run("candidates", "lengths", "pages", out), out)
                args, kwargs = launch_calls[0]
                self.assertEqual(args, ("candidates", "lengths", "pages"))
                self.assertTrue(kwargs["split"])
                self.assertEqual(
                    (kwargs["rank"], kwargs["span"], kwargs["cand_k"]), (5, 64, cand_k)
                )
                self.assertIs(kwargs["buffer"], storage.buffer)
                self.assertEqual(kwargs["out_fallback"].shape, (4,))

    def test_declined_factory_remains_unusable(self):
        host = types.ModuleType("sglang.srt.layers.attention.dsa.dsa_dcp_sharded")
        host.DcpTopKMerge = types.SimpleNamespace(create=lambda *a: None)
        kernel = types.ModuleType(KERNEL)
        kernel.dcp_topk_merge = lambda *a, **k: self.fail("declined merge launched")
        with patch.dict(sys.modules, {host.__name__: host, kernel.__name__: kernel}):
            merge = self.load_split()(types.SimpleNamespace(rank_in_group=0), 8)
            self.assertFalse(merge.usable(1))
            with self.assertRaisesRegex(RuntimeError, "unavailable"):
                merge.run(None, None, None, None)


if __name__ == "__main__":
    unittest.main()
