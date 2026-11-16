# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Mutation check of test/unit/test_kimi_k31_tp16_dcp.py: every mutation must make the CPU tests fail.

Usage: K31_NVCC=<nvcc 13> python tools/rubin_k31/tp16_cpu_mutations.py   (from the plugin worktree root)
Copies sgl_nvidia_mega/ and the test into a temp tree, applies one mutation at a time, runs the test file.
"""

import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TEST = "test/unit/test_kimi_k31_tp16_dcp.py"
MUTATIONS = {
    "owner_rule": ("sgl_nvidia_mega/kimi_k31/dcp/layout.py", "if slot < 0 or slot % dcp != rank:",
                   "if slot < 0 or (slot // 64) % dcp != rank:"),
    "local_row": ("sgl_nvidia_mega/kimi_k31/dcp/layout.py", "    return slot // dcp\n", "    return slot // 64\n"),
    "local_len": ("sgl_nvidia_mega/kimi_k31/dcp/layout.py", "return (length + dcp - 1 - rank) // dcp",
                  "return (length + dcp - rank) // dcp"),
    "global_pos": ("sgl_nvidia_mega/kimi_k31/dcp/layout.py", "return local * dcp + rank",
                   "return local * dcp + (dcp - 1 - rank)"),
    "tag_zero": ("sgl_nvidia_mega/kimi_k31/dcp/layout.py", "* EXCHANGES + kind + 1) & 0xFFFFFFFF",
                 "* EXCHANGES + kind) & 0xFFFFFFFF"),
    "cuda_tag": ("sgl_nvidia_mega/kimi_k31/dcp/comm.cuh", "static_cast<uint32_t>(kind) + 1u;",
                 "static_cast<uint32_t>(kind) + 2u;"),
    "cuda_len": ("sgl_nvidia_mega/kimi_k31/dcp/comm.cuh", "return length <= 0 ? 0 : (length + dcp - 1 - rank) / dcp;",
                 "return length <= 0 ? 0 : (length + dcp - rank) / dcp;"),
    "kda_pad": ("sgl_nvidia_mega/kimi_k31/k3/shapes.py", "return 4 * kda_dim(tp), 128, heads(tp), -raw % ALIGN",
                "return 4 * kda_dim(tp), 128, heads(tp), raw % ALIGN"),
    "index_col": ("sgl_nvidia_mega/kimi_k31/k3/shapes.py", "    return QKV_A + mla_gate(tp)\n\n\ndef k31_pitch",
                  "    return QKV_A + 1536\n\n\ndef k31_pitch"),
    "part_align": ("sgl_nvidia_mega/kimi_k31/dcp/layout.py", "return LATENT // 2 + 2", "return LATENT // 2 + 1"),
    "tp16_gate": ("sgl_nvidia_mega/kimi_k31/k3/shapes.py", 'os.environ.get("K31MK_TP16", "0") == "1"',
                  'os.environ.get("K31MK_TP16", "1") == "1"'),
    "comm_fence": ("sgl_nvidia_mega/kimi_k31/dcp/comm.cuh", "  pdl_trigger();\n  if (!(c.phases & 2)) return;\n  const uint64_t start = gtime();\n  for (int item",
                   "  pdl_trigger();\n  asm volatile(\"fence.acq_rel.sys;\" ::: \"memory\");\n  if (!(c.phases & 2)) return;\n  const uint64_t start = gtime();\n  for (int item"),
    "fused_deadline": ("sgl_nvidia_mega/kimi_k31/k3/cutedsl/oproj/consumer.py",
                       "self.consumer = OprojConsumer(nb, has_delta=has_delta, deadline_ns=deadline_ns)",
                       "self.consumer = OprojConsumer(nb, has_delta=has_delta)"),
    "b2_grid": ("sgl_nvidia_mega/kimi_k31/k3/cutedsl/moe/small_batch/chain.py", "min(box.sms - 7 * b, 160)",
                "box.sms - 7 * b"),
    "pad_heads": ("sgl_nvidia_mega/kimi_k31/dcp/layout.py", "return PADDED_HEADS if 64 < total < 128 else total",
                  "return total"),
}


def run(tree):
    env = dict(os.environ, PYTHONPATH=str(tree), PYTHONDONTWRITEBYTECODE="1")
    return subprocess.run([sys.executable, str(tree / TEST)], cwd=tree, env=env, capture_output=True, text=True)


def main():
    caught = 0
    with tempfile.TemporaryDirectory() as tmp:
        base = Path(tmp) / "base"
        shutil.copytree(ROOT / "sgl_nvidia_mega", base / "sgl_nvidia_mega")
        (base / "test/unit").mkdir(parents=True)
        shutil.copy(ROOT / TEST, base / TEST)
        result = run(base)
        print(f"base: {'PASS' if result.returncode == 0 else 'FAIL'}")
        if result.returncode:
            print(result.stderr[-2000:])
            sys.exit(2)
        for name, (path, old, new) in MUTATIONS.items():
            tree = Path(tmp) / name
            shutil.copytree(base, tree)
            source = (tree / path).read_text()
            if source.count(old) != 1:
                print(f"{name}: mutation text not found once ({source.count(old)})")
                sys.exit(3)
            (tree / path).write_text(source.replace(old, new))
            failed = run(tree).returncode != 0
            caught += failed
            print(f"{name}: {'CAUGHT' if failed else 'MISSED'}")
            shutil.rmtree(tree)
    print(f"caught {caught}/{len(MUTATIONS)}")
    sys.exit(0 if caught == len(MUTATIONS) else 1)


if __name__ == "__main__":
    main()
