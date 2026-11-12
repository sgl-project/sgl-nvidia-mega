# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""K31MK_DSA_ROW_TABLE_ONCE: the DSA layers of one verify step all read the per-row page table that the step's first layer
copied (RowTableShare), and every new step copies it again.

Expected tables come from torch.repeat_interleave of the request table (row r reads request r // q), not from
VerifySelection. The steps below reuse ONE table tensor whose contents change between steps, as the served graphs do, so a
layer that kept an earlier step's copy reads stale pages and fails."""

import pytest
import torch

from sgl_nvidia_mega.kimi_k31.dsa import verify as V

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")

Q = 8
LAYERS = 4


def expected(table):
    return table.repeat_interleave(Q, dim=0)


def selections(rows, share):
    out = [V.VerifySelection(torch.device("cuda"), rows) for _ in range(LAYERS)]
    for s in out:
        s.share = share
    return out


def fill(table, seed):
    g = torch.Generator().manual_seed(seed)
    table.copy_(torch.randint(0, 1 << 20, tuple(table.shape), generator=g, dtype=torch.int32))


def same(a, b):
    torch.testing.assert_close(a, b, rtol=0, atol=0)


@pytest.mark.parametrize("b", [2, 4, 8])
def test_each_layer_reads_its_steps_table(b):
    share = V.RowTableShare()
    layers = selections(b * Q, share)
    table = torch.empty(b, 300, dtype=torch.int32, device="cuda")
    for step in range(3):
        fill(table, step)
        share.step += 1
        for layer in layers:
            same(layer.row_table(table, Q), expected(table))


def test_a_layer_with_another_table_copies_its_own():
    share = V.RowTableShare()
    layers = selections(2 * Q, share)
    first = torch.empty(2, 129, dtype=torch.int32, device="cuda")
    other = torch.empty(2, 129, dtype=torch.int32, device="cuda")
    fill(first, 10)
    fill(other, 11)
    share.step += 1
    a = layers[0].row_table(first, Q).clone()
    b = layers[1].row_table(other, Q).clone()
    c = layers[2].row_table(first, Q).clone()
    same(a, expected(first))
    same(b, expected(other))
    same(c, expected(first))


def test_one_request_keeps_the_stride_zero_view():
    share = V.RowTableShare()
    layer = selections(Q, share)[0]
    table = torch.empty(1, 77, dtype=torch.int32, device="cuda")
    fill(table, 3)
    share.step += 1
    rows = layer.row_table(table, Q)
    same(rows, expected(table))
    assert rows.stride(0) == 0 and rows.data_ptr() == table.data_ptr()  # still a view of the request table


def test_graph_replay_reads_the_replayed_copy():
    b = 4
    share = V.RowTableShare()
    layers = selections(b * Q, share)
    table = torch.empty(b, 257, dtype=torch.int32, device="cuda")
    outs = [torch.empty(b * Q, 257, dtype=torch.int32, device="cuda") for _ in range(LAYERS)]

    def step():
        share.step += 1
        for layer, out in zip(layers, outs):
            out.copy_(layer.row_table(table, Q))

    fill(table, 0)
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        step()  # eager warm-up: allocates the copy buffer outside the capture
    torch.cuda.current_stream().wait_stream(side)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        step()
    for seed in (1, 2):
        fill(table, seed)
        graph.replay()
        torch.cuda.synchronize()
        for out in outs:
            same(out, expected(table))


class _View(V.SparseVerifyStep):
    def validate(self, *, auxiliary_width=None):
        pass


def test_run_verify_starts_a_new_step(monkeypatch):
    """BuiltSparseVerify.run_verify (the per-step entry) must make the next step copy again."""
    share = V.RowTableShare()
    layers = selections(2 * Q, share)
    table = torch.empty(2, 200, dtype=torch.int32, device="cuda")
    seen = []

    def forward(self, embeddings, view):
        seen.append([layer.row_table(table, Q).clone() for layer in layers])

    monkeypatch.setattr(V.BuiltK31Verify, "run_verify", forward)
    built = object.__new__(V.BuiltSparseVerify)
    built.sparse_cores, built.tsi, built.row_share = {}, None, share
    view = object.__new__(_View)
    object.__setattr__(view, "sparse", 1)  # a frozen dataclass: only the field run_verify reads
    for step in range(2):
        fill(table, 20 + step)
        built.run_verify(None, view)
        for rows in seen[-1]:
            same(rows, expected(table))
