# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""CPU checks for the K3.1 MK TP16 / DCP16 plumbing. No GPU, no SGLang import.

Independent references:
- TP8 widths: the shipped constants (6288 / 3648 / 3776 / 12 heads / 4 padding rows).
- TP16 widths: K3 GB200 TP16 runs (KDA in_proj N = 3216) and a CPU audit
  (K3.1 MLA pitch 3008, index-K column 2880).
- DCP index math: brute force over global positions, from stock's stated rules (dsa_dcp_sharded.py: "position p
  lives on rank p % N at local position p // N"; widened slot page_table[p // span] * span + p % span; owner
  slot % N, row slot // N), checked against the local block-table addressing the kernels use
  (table[l // 64] * 64 + l % 64 for local position l).
- Merge math: FP64 full attention over all keys vs the stock DCP algorithm (per-rank partial softmax, LSE (log2)
  combine with exp2 weights, W_UV before the combine).
- comm.cuh's tag / local length: the CUDA host functions compiled with nvcc and run on the CPU (skipped without
  nvcc >= 13; K31_NVCC overrides the path).

Each test names the bug it catches.
"""

from __future__ import annotations

import math
import os
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

import torch

from sgl_nvidia_mega.kimi_k31.k3 import projection_storage as storage
from sgl_nvidia_mega.kimi_k31.k3 import shapes
from sgl_nvidia_mega.kimi_k31.dcp import layout as L


class Tp16Shapes(unittest.TestCase):
    def test_tp8_widths_unchanged(self):
        # Catches: any TP-parametrization that moves a shipped TP8 width.
        self.assertEqual(shapes.heads(8), 12)
        self.assertEqual(shapes.kda_rows(8), (6144, 128, 12, 4))
        self.assertEqual(shapes.kda_pitch(8), 6288)
        self.assertEqual(shapes.mla_pitch(8), 3648)
        self.assertEqual(shapes.k31_index_col(8), 3648)
        self.assertEqual(shapes.k31_pitch(8), 3776)
        self.assertEqual(storage._MLA_LAYOUT.rows, 3648)
        self.assertEqual(storage._KDA_LAYOUT.rows, 6288)
        self.assertEqual(storage._KDA_LAYOUT.padding_rows, 4)

    def test_tp16_widths(self):
        # Catches: a wrong KDA padding (e.g. raw % 16 instead of -raw % 16 gives 3212), the gate at 12 heads, the
        # index-K column left at 3648.
        self.assertEqual(shapes.heads(16), 6)
        self.assertEqual(shapes.kda_pitch(16), 3216)  # K3 TP16 in_proj N
        self.assertEqual(storage.kda_layout(6).rows, 3216)
        self.assertEqual(storage.kda_layout(6).padding_rows, 10)
        self.assertEqual(storage.mla_layout(6).rows, 2880)
        self.assertEqual(shapes.k31_index_col(16), 2880)
        self.assertEqual(shapes.k31_pitch(16), 3008)
        from sgl_nvidia_mega.kimi_k31 import projection_storage as k31_storage

        self.assertEqual(k31_storage.pitch(6), 3008)
        self.assertEqual(k31_storage.layout(6).rows, 3008)
        self.assertEqual(k31_storage.pitch(12), k31_storage.PITCH)

    def test_tp16_gate(self):
        # Catches: TP16 admitted without the flag (the shipped TP8 build must not change behaviour).
        old = os.environ.pop("K31MK_TP16", None)
        try:
            self.assertEqual(shapes.enabled_tp_sizes(), frozenset({8}))
            os.environ["K31MK_TP16"] = "1"
            self.assertEqual(shapes.enabled_tp_sizes(), frozenset({8, 16}))
        finally:
            os.environ.pop("K31MK_TP16", None)
            if old is not None:
                os.environ["K31MK_TP16"] = old
        with self.assertRaises(ValueError):
            shapes.heads(32)
        self.assertEqual(shapes.tp_of_heads(6), 16)
        self.assertEqual(shapes.tp_of_heads(12), 8)


def _table(pages_per_request, pages_total, seed):
    g = torch.Generator().manual_seed(seed)
    return (torch.randperm(pages_total - 1, generator=g)[:pages_per_request] + 1).tolist()


class DcpLayout(unittest.TestCase):
    def test_widened_slots_match_local_block_table(self):
        # Catches: owner rule w // 64 % N (instead of w % N), local row w // 64 (instead of w // N), and any layout
        # in which local position l is not at table[l // 64] * 64 + l % 64 (the address DeepGEMM and the sparse
        # core use for the local shard).
        for dcp in (8, 16):
            span = L.PAGE * dcp
            length = 3 * span + 77
            table = _table(-(-length // span), 1000, dcp)
            for p in range(length):
                w = L.widened_slot(table, p, dcp)
                owner, local = p % dcp, p // dcp
                self.assertEqual(L.owner(w, dcp), owner)
                self.assertEqual(L.global_position(local, dcp, owner), p)
                expected_row = table[local // 64] * 64 + local % 64
                for rank in range(dcp):
                    got = L.local_write_slot(w, dcp, rank)
                    self.assertEqual(got, expected_row if rank == owner else -1, (dcp, p, rank))

    def test_local_length_brute_force(self):
        # Catches: an off-by-one in the per-rank length (e.g. (L + N - r) // N), which drops or adds a key.
        for dcp in (8, 16):
            for length in list(range(0, 70)) + [2047, 2048, 2049, 131072, 131073]:
                for rank in range(dcp):
                    brute = sum(1 for p in range(length) if p % dcp == rank) if length < 5000 else None
                    got = L.local_length(length, dcp, rank)
                    if brute is not None:
                        self.assertEqual(got, brute, (dcp, length, rank))
                    self.assertEqual(sum(L.local_length(length, dcp, r) for r in range(dcp)), max(length, 0))

    def test_localize_compact(self):
        # Catches: owned slots out of order, a missing -1 tail, or the wrong local row (stock localize semantics).
        dcp, rank = 16, 5
        slots = [37, 5, -1, 21, 4096 + 5, 3, 16 * 9 + 5]
        out, count = L.localize_compact(slots, dcp, rank)  # owned: 37, 5, 21, 4101, 149 (x % 16 == 5)
        self.assertEqual(count, 5)
        self.assertEqual(out, [2, 0, 1, 256, 9, -1, -1])

    def test_tag(self):
        # Catches: a tag of 0 (matches zeroed regions), or two exchange sites of one forward (or of consecutive
        # forwards) sharing a tag, which would let a reader take another site's records.
        seen = set()
        for epoch in range(4):
            for ordinal in range(24):
                for kind in range(L.EXCHANGES):
                    t = L.tag(epoch, ordinal, kind)
                    self.assertNotEqual(t, 0)
                    self.assertNotIn(t, seen)
                    seen.add(t)
        # Wrap: the value is taken mod 2^32 and 0 maps to 1.
        epoch = (1 << 32) // (24 * 3)
        values = {L.tag(epoch + e, o, k) for e in range(3) for o in range(24) for k in range(3)}
        self.assertNotIn(0, values)

    def test_geometry_alignment(self):
        # Catches: a record block that is not 16-byte aligned (the kernels store and load v2.u64 records).
        for dcp, heads in ((8, 12), (16, 6)):
            g = L.Geometry(dcp, heads, cap=8)
            self.assertEqual(g.q_words % 2, 0)
            self.assertEqual(g.part_words % 2, 0)
            for parity, src, row, head in ((0, 0, 0, 0), (1, dcp - 1, 7, heads - 1), (0, 3, 2, 1)):
                self.assertEqual(g.q_index(parity, src, row, 0) % 2, 0)
                self.assertEqual(g.cand_index(parity, src, row, 0, 0) % 2, 0)
                self.assertEqual(g.part_index(parity, src, row, head, 0) % 2, 0)
            self.assertLess(g.bytes(), 64 << 20)  # receive regions per rank at cap 8

    def test_padded_heads(self):
        # Catches: 96 gathered heads not padded to 128 (trtllm-gen rejects 64 < heads < 128).
        self.assertEqual(L.padded_heads(16, 6), 128)
        self.assertEqual(L.padded_heads(8, 12), 128)
        self.assertEqual(L.padded_heads(16, 4), 64)


def _stable_topk(scores, positions, k):
    """Score descending, then position ascending (stock deterministic=True)."""
    order = sorted(range(len(scores)), key=lambda i: (-scores[i], positions[i]))
    return [positions[i] for i in order[:k]]


class DcpMergeMath(unittest.TestCase):
    def test_sharded_topk_is_exact(self):
        # Catches: a local candidate set smaller than min(local length, k), a wrong global position of a local key,
        # or a merge that is not exact (stock's argument: a key in the global top-k is in its shard's top-k).
        g = torch.Generator().manual_seed(7)
        for dcp, length, k, ties in ((16, 3000, 256, False), (16, 3000, 256, True), (8, 900, 64, True)):
            if ties:  # tie-heavy rows: few distinct values
                full = (torch.randint(0, 9, (length,), generator=g).double() / 8).tolist()
            else:
                full = torch.randn(length, generator=g, dtype=torch.float64).tolist()
            union_scores, union_pos = [], []
            for rank in range(dcp):
                n = L.local_length(length, dcp, rank)
                local_pos = [L.global_position(l, dcp, rank) for l in range(n)]
                local_scores = [full[p] for p in local_pos]
                chosen = _stable_topk(local_scores, local_pos, min(n, k))
                union_pos += chosen
                union_scores += [full[p] for p in chosen]
            merged = _stable_topk(union_scores, union_pos, k)
            direct = _stable_topk(full, list(range(length)), k)
            self.assertEqual(merged, direct, (dcp, length, k, ties))

    def test_lse_merge_equals_full_attention(self):
        # Catches: base-e weights on log2 LSEs, a missing max subtraction, empty shards not neutral (stock fixup:
        # zero partial, LSE -inf), and W_UV not commuting with the combine (stock dcp_v_preproject).
        g = torch.Generator().manual_seed(11)
        dcp, keys, latent, out_dim, empty = 16, 600, 64, 16, 3
        scores = torch.randn(keys, generator=g, dtype=torch.float64) * 3
        values = torch.randn(keys, latent, generator=g, dtype=torch.float64)
        w_uv = torch.randn(latent, out_dim, generator=g, dtype=torch.float64)
        selected = [p for p in torch.randperm(keys, generator=g)[:300].tolist() if p % dcp != empty]
        full = torch.softmax(scores[selected], 0) @ values[selected] @ w_uv
        partials, lses = [], []
        for rank in range(dcp):
            mine = [p for p in selected if p % dcp == rank]
            if not mine:  # stock fixup_zero_kv_rows
                partials.append(torch.zeros(latent, dtype=torch.float64))
                lses.append(-math.inf)
                continue
            partials.append(torch.softmax(scores[mine], 0) @ values[mine])
            lses.append(float(torch.logsumexp(scores[mine], 0)) / math.log(2))  # trtllm-gen LSE: log2
        self.assertEqual(lses[empty], -math.inf)
        top = max(lses)
        weights = [0.0 if l == -math.inf else 2.0 ** (l - top) for l in lses]
        projected = [p @ w_uv for p in partials]  # stock: W_UV before the exchange
        combined = sum(w * y for w, y in zip(weights, projected)) / max(sum(weights), 1.0)
        self.assertTrue(torch.allclose(combined, full, rtol=1e-12, atol=1e-12))


PLUGIN = Path(__file__).resolve().parents[2] / "sgl_nvidia_mega"


class StaticRules(unittest.TestCase):
    def test_comm_mailboxes_are_epoch_records(self):
        # Catches: a fence, a multicast store, a sentinel re-arm or an unbounded poll added to the DCP exchanges
        # (design rule: new mailboxes are epoch-tagged, no re-arm, no fences; every remote wait fails stop).
        source = (PLUGIN / "kimi_k31/dcp/comm.cuh").read_text()
        code = "\n".join(line.split("//")[0] for line in source.splitlines())
        for banned in ("fence.", "multimem", "membar", "__threadfence", "ld.volatile", "ld.global.nc", "__ldg"):
            self.assertNotIn(banned, code)
        self.assertEqual(code.count("ld_rec2("), 3)  # its definition and the two loads inside poll2
        self.assertEqual(code.count("poll2("), 4)  # its definition and the three exchange polls (cand, q, part)
        self.assertIn("gtime() - start > deadline_ns) __trap();", code)
        self.assertEqual(code.count("st.relaxed.sys.global.v2.u64"), 1)
        self.assertEqual(code.count("ld.relaxed.sys.global.v2.u64"), 1)

    def test_tp16_oproj_fused_has_a_deadline(self):
        # Catches: the TP16 fused o-projection poll left unbounded (MK_TP16.md row 4).
        import ast

        tree = ast.parse((PLUGIN / "kimi_k31/k3/cutedsl/oproj/consumer.py").read_text())
        fused = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "OprojFused")
        init = next(n for n in fused.body if isinstance(n, ast.FunctionDef) and n.name == "__init__")
        self.assertIn("deadline_ns", [a.arg for a in init.args.kwonlyargs])
        calls = [n for n in ast.walk(init) if isinstance(n, ast.Call) and getattr(n.func, "id", "") == "OprojConsumer"]
        self.assertEqual(len(calls), 1)
        self.assertIn("deadline_ns", [k.arg for k in calls[0].keywords])
        prepared = (PLUGIN / "kimi_k31/k3/cutedsl/oproj/prepared.py").read_text()
        self.assertIn("OprojFused(nb, has_delta=has_delta, deadline_ns=deadline_ns, **options)", prepared)

    def test_tp16_b2_front_grid_fits_role_mask(self):
        # Catches: the B2 front launched with more CTAs than its 5-word (160-CTA) role bitmask.
        chain = (PLUGIN / "kimi_k31/k3/cutedsl/moe/small_batch/chain.py").read_text()
        self.assertIn("grid = box.sms - 7 * b if b == 1 else min(box.sms - 7 * b, 160)", chain)
        front = (PLUGIN / "kimi_k31/k3/cutedsl/moe/small_batch/front_tp16.py").read_text()
        self.assertIn("for word in cutlass.range_constexpr(5):", front)  # the mask the cap is sized for


_HOST_PROGRAM = r"""
#define K31_DCP_DEVICE_ONLY
#include "COMM"
#include <cstdio>
int main() {
  for (unsigned e : {0u, 1u, 7u, 59652323u, 59652324u, 4294967295u})
    for (int o : {0, 1, 23})
      for (int k = 0; k < 3; ++k) std::printf("T %u %d %d %u\n", e, o, k, k31_dcp::make_tag(e, o, k, 24));
  for (int n : {8, 16})
    for (int len : {0, 1, 15, 16, 17, 2048, 131073})
      for (int r = 0; r < n; ++r) std::printf("L %d %d %d %d\n", len, n, r, k31_dcp::local_len(len, n, r));
}
"""


def _nvcc():
    path = os.environ.get("K31_NVCC") or shutil.which("nvcc")
    if not path:
        return None
    version = subprocess.run([path, "--version"], capture_output=True, text=True).stdout
    return path if "release 13" in version else None


class CommHostFunctions(unittest.TestCase):
    @unittest.skipUnless(_nvcc(), "needs nvcc >= 13 (K31_NVCC)")
    def test_cuda_tag_and_length_match_python(self):
        # Catches: comm.cuh and layout.py disagreeing on the tag or the local length (a reader would wait for a
        # tag no writer stores: a deadline trap in serving).
        comm = Path(__file__).resolve().parents[2] / "sgl_nvidia_mega/kimi_k31/dcp/comm.cuh"
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / "host.cu"
            source.write_text(_HOST_PROGRAM.replace("COMM", str(comm)))
            binary = Path(tmp) / "host"
            subprocess.run([_nvcc(), "-std=c++20", "-arch=sm_100a", "-o", str(binary), str(source)], check=True)
            lines = subprocess.run([str(binary)], capture_output=True, text=True, check=True).stdout.splitlines()
        self.assertGreater(len(lines), 50)
        for line in lines:
            kind, *values = line.split()
            values = [int(v) for v in values]
            if kind == "T":
                epoch, ordinal, k, got = values
                self.assertEqual(got, L.tag(epoch, ordinal, k), line)
            else:
                length, n, rank, got = values
                self.assertEqual(got, L.local_length(length, n, rank), line)


if __name__ == "__main__":
    unittest.main()
