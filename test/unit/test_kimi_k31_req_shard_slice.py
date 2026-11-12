# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""CPU tests of K31MK_DSA_REQ_SHARD_SLICE (dsa/req_shard.py owner_slice + ReqShardState.select): the owner block's index
query and head gates are row slices of the step's tensors instead of two index_select gathers. Expected values are
hand-derived from the owner rule (rank r owns requests min(r * per + j, b - 1), j < per = ceil(b / world); rows are
request-major, q per request): the owner rows are one contiguous range iff per == 1 or the block is not clamped
(r * per + per - 1 <= b - 1), starting at min(r * per, b - 1) * q. The module file is loaded directly (no package import),
so the tests run without a GPU; select() is driven up to its first GPU call by a selection stub that records its inputs.
K31_TEST_REQ_SHARD_PY points them at another copy (mutation runs)."""

import importlib.util
import os
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

ROOT = Path(__file__).resolve().parents[2] / "sgl_nvidia_mega" / "kimi_k31" / "dsa"
REQ_SHARD_PY = Path(os.environ.get("K31_TEST_REQ_SHARD_PY", ROOT / "req_shard.py"))


def _mod(slice_on):
    old = os.environ.get("K31MK_DSA_REQ_SHARD_SLICE")
    os.environ["K31MK_DSA_REQ_SHARD_SLICE"] = "1" if slice_on else "0"
    try:
        spec = importlib.util.spec_from_file_location("k31_req_shard_slice_under_test", REQ_SHARD_PY)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
    finally:
        if old is None:
            del os.environ["K31MK_DSA_REQ_SHARD_SLICE"]
        else:
            os.environ["K31MK_DSA_REQ_SHARD_SLICE"] = old
    return mod


def _want_lo(b, q, world, rank):
    per = -(-b // world)
    if per == 1 or rank * per + per - 1 <= b - 1:
        return min(rank * per, b - 1) * q
    return None


def test_owner_slice_matches_the_owner_rule():
    m = _mod(True)
    seen = {True: 0, False: 0}
    for world in (2, 4, 8, 16):
        for b in range(1, 41):
            for q in (4, 5, 6, 7, 8):
                for rank in range(world):
                    want = _want_lo(b, q, world, rank)
                    assert m.owner_slice(b, q, world, rank) == want, (b, q, world, rank)
                    seen[want is not None] += 1
    assert seen[True] > 0 and seen[False] > 0  # both branches exercised (clamped multi-request blocks exist, e.g. b 17 world 16)


class _Stop(Exception):
    pass


def _state(m, b, q, world, rank, own_lo):
    per = -(-b // world)
    rec = {}

    def run(q_own, g_own, cache, table, step):
        rec["q"], rec["g"] = q_own, g_own
        raise _Stop

    st = m.ReqShardState.__new__(m.ReqShardState)
    st.q = q
    st.active = SimpleNamespace(per=per, own_lo=own_lo, table=None, step=None, tkp=None,  # tkp: K31MK_DSA_REQ_SHARD_TKPUSH (off)
                                rows_idx=torch.tensor(m.owner_rows(b, q, world, rank), dtype=torch.long),
                                selection=SimpleNamespace(run=run))
    return st, rec


def _inputs(b, q, contiguous=True):
    g = torch.Generator().manual_seed(1107 + b * 31 + q)
    rows = b * q
    query = torch.randint(0, 256, (rows, 32, 128), dtype=torch.uint8, generator=g).view(torch.float8_e4m3fn)
    gates = torch.randn((rows, 32, 1), generator=g)
    if not contiguous:  # same values, row stride doubled
        qq = torch.zeros((rows, 2, 32, 128), dtype=torch.uint8)
        qq[:, 0] = query.view(torch.uint8)
        query = qq[:, 0].view(torch.float8_e4m3fn)
        gg = torch.zeros((rows, 2, 32, 1))
        gg[:, 0] = gates
        gates = gg[:, 0]
    return query, gates


@pytest.mark.parametrize("b,q,world", [(1, 8, 16), (4, 8, 16), (12, 8, 16), (13, 6, 16), (16, 8, 16), (8, 8, 4), (17, 8, 16), (40, 5, 16)])
def test_select_passes_the_owner_rows_bitwise(b, q, world):
    m = _mod(True)
    for rank in range(world):
        query, gates = _inputs(b, q)
        rows = m.owner_rows(b, q, world, rank)
        want_q = query.view(torch.uint8)[rows]  # advanced indexing: an independent gather of the owner rows
        want_g = gates[rows]
        lo = m.owner_slice(b, q, world, rank)
        st, rec = _state(m, b, q, world, rank, lo)
        with pytest.raises(_Stop):
            st.select(0, query, gates, None)
        assert rec["q"].is_contiguous() and rec["g"].is_contiguous()
        assert torch.equal(rec["q"].view(torch.uint8), want_q) and torch.equal(rec["g"], want_g), (b, q, world, rank)
        if lo is not None:  # SLICE engaged: the owner block is a view of the step's tensors, no gather kernel
            assert rec["q"].data_ptr() == query.data_ptr() + lo * 32 * 128
            assert rec["g"].data_ptr() == gates.data_ptr() + lo * 32 * 4


def test_noncontiguous_inputs_fall_back_to_the_gathers():
    m = _mod(True)
    b, q, world, rank = 12, 8, 16, 5
    query, gates = _inputs(b, q, contiguous=False)
    rows = m.owner_rows(b, q, world, rank)
    st, rec = _state(m, b, q, world, rank, m.owner_slice(b, q, world, rank))
    with pytest.raises(_Stop):
        st.select(0, query, gates, None)
    assert rec["q"].is_contiguous() and rec["g"].is_contiguous()
    assert torch.equal(rec["q"].view(torch.uint8), query.view(torch.uint8)[rows]) and torch.equal(rec["g"], gates[rows])


def test_default_off_keeps_the_gathers():
    m = _mod(False)
    assert m.SLICE is False
    b, q, world, rank = 12, 8, 16, 3
    query, gates = _inputs(b, q)
    st, rec = _state(m, b, q, world, rank, None)  # what __init__ stores when SLICE is off
    with pytest.raises(_Stop):
        st.select(0, query, gates, None)
    assert rec["q"].data_ptr() != query.data_ptr() + 3 * q * 32 * 128  # a gathered copy, not a view
    assert torch.equal(rec["q"].view(torch.uint8), query.view(torch.uint8)[m.owner_rows(b, q, world, rank)])
