# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Slot draft_stage0_table's kernels on one GPU: the gather returns the table's rows bit
for bit (row ``vocab`` for ids outside the vocabulary), eager, with and without its PDL
wait, and replayed from a CUDA graph after the ids change; the fill writes every id's
rows of a live chain one width a launch; the check passes a table filled from the chain
and finds one flipped element. The real chain's check runs in the served draft at its
first forward (its startup line reports the mismatches per width on every rank).

``python -m pytest test/kernels/test_deepseek_v41_draft_stage0_table.py -s``
"""

import pytest
import torch

if not torch.cuda.is_available():
    pytest.skip("needs a GPU", allow_module_level=True)

from sgl_nvidia_mega.deepseek_v41 import draft_stage0  # noqa: E402

VOCAB, HEADS, DIM = 1000, 8, 512


def _table(seed=0):
    t = draft_stage0.Stage0Table(VOCAB, HEADS, DIM, "cuda")
    g = torch.Generator(device="cuda").manual_seed(seed)
    t.q.copy_(torch.randn(t.q.shape, generator=g, device="cuda").bfloat16())
    t.kv.copy_(torch.randn(t.kv.shape, generator=g, device="cuda").bfloat16())
    t.filled = True
    return t


def _want(t, ids):
    rows = torch.where((ids >= 0) & (ids < VOCAB), ids, torch.full_like(ids, VOCAB))
    return t.q[rows], t.kv[rows]


def _same(a, b):
    return torch.equal(a.view(torch.int16), b.view(torch.int16))


@pytest.mark.parametrize("use_pdl", [False, True])
@pytest.mark.parametrize("rows", [1, 5, 10, 24])
def test_gather_returns_the_tables_rows(use_pdl, rows):
    t = _table()
    ids = torch.randint(0, VOCAB, (rows,), device="cuda")
    ids[0] = VOCAB - 1
    if rows > 2:
        ids[1] = VOCAB + 7  # outside: the zero row
        ids[2] = -3
    q, kv = draft_stage0.gather(t, ids, use_pdl=use_pdl)
    wq, wkv = _want(t, ids)
    assert q.shape == (rows, HEADS, DIM) and kv.shape == (rows, DIM)
    assert _same(q, wq) and _same(kv, wkv)


def test_gather_replays_from_a_graph():
    t = _table(1)
    ids = torch.randint(0, VOCAB, (5,), device="cuda")
    draft_stage0.gather(t, ids)  # compile outside the capture
    graph = torch.cuda.CUDAGraph()
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s), torch.cuda.graph(graph):
        q, kv = draft_stage0.gather(t, ids)
    torch.cuda.current_stream().wait_stream(s)
    for _ in range(3):
        ids.copy_(torch.randint(0, VOCAB, (5,), device="cuda"))
        graph.replay()
        torch.cuda.synchronize()
        wq, wkv = _want(t, ids)
        assert _same(q, wq) and _same(kv, wkv)


def _chain():
    """A deterministic stand-in for the live chain: an embedding, a BF16 GEMM per row."""
    g = torch.Generator(device="cuda").manual_seed(3)
    embed = torch.randn((VOCAB, 256), generator=g, device="cuda").bfloat16()
    wq = torch.randn((256, HEADS * DIM), generator=g, device="cuda").bfloat16() / 16
    wkv = torch.randn((256, DIM), generator=g, device="cuda").bfloat16() / 16
    calls = []

    def rows_fn(ids):
        calls.append(ids.shape[0])
        valid = ids < VOCAB
        x = embed[torch.where(valid, ids, torch.zeros_like(ids))]
        x = torch.where(valid.view(-1, 1), x, torch.zeros_like(x))
        return (x @ wq).view(-1, HEADS, DIM), x @ wkv

    return rows_fn, calls


def test_fill_writes_every_ids_rows_one_width_a_launch():
    rows_fn, calls = _chain()
    t = draft_stage0.Stage0Table(VOCAB, HEADS, DIM, "cuda")
    draft_stage0.fill(t, rows_fn, 5)
    assert t.filled and set(calls) == {5} and len(calls) == -(-(VOCAB + 1) // 5)
    ids = torch.arange(VOCAB + 1, device="cuda")
    want_q = torch.cat([rows_fn(ids[i : i + 5])[0] for i in range(0, VOCAB + 1, 5)])
    assert _same(t.q, want_q)
    assert not t.q[VOCAB].float().abs().sum() and not t.kv[VOCAB].float().abs().sum()


def test_check_passes_the_filled_table_and_finds_a_flip():
    rows_fn, _ = _chain()
    t = draft_stage0.Stage0Table(VOCAB, HEADS, DIM, "cuda")
    draft_stage0.fill(t, rows_fn, 5)
    extra = torch.tensor([17, 999, 999, 999, 999], device="cuda")
    assert draft_stage0.check(t, rows_fn, (5, 10), extra=extra) == {5: 0, 10: 0}
    # flip one element of an id the sample holds (the step's own bonus id)
    t.kv[17, 3] = -t.kv[17, 3] if t.kv[17, 3] != 0 else 1.0
    bad = draft_stage0.check(t, rows_fn, (5, 10), extra=extra)
    assert bad[5] > 0 and bad[10] > 0


def test_nbytes_is_the_q_and_kv_rows():
    t = draft_stage0.Stage0Table(VOCAB, HEADS, DIM, "cuda")
    assert t.nbytes == (VOCAB + 1) * (HEADS * DIM + DIM) * 2
