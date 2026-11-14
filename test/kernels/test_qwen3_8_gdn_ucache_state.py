# SPDX-License-Identifier: Apache-2.0
"""The GDN u-cache cursor commit and resets (kernels/ops/attention/linear/
gdn_ucache_state.py, also through runtime/gdn_ucache_state.UcacheState) on a GPU.

Exact (int32 cursors, copied conv rows) against a torch reference of the kernels.
Also: UcacheState.commit captured in a
CUDA graph and replayed on new inputs, which allocates nothing persistent and syncs
nothing."""

from types import SimpleNamespace
from unittest import mock

import pytest
import torch

from sgl_nvidia_mega.qwen3_8.kernels.ops.attention.linear import gdn_ucache_state as K
from sgl_nvidia_mega.qwen3_8.runtime import gdn_ucache_state as R

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="the u-cache cursor kernels need a CUDA GPU"
)

T = 4  # verify rows a request
SLOTS, ROWS, BS = 300, 130, 96
L, H, HV, KD, VD = 3, 2, 6, 16, 16  # small rings: the cursor kernels never read them
CONV_DIM, KM1 = 24, 3


def _buffers(n, by_row, seed):
    g = torch.Generator(device="cuda").manual_seed(seed)
    return K.GdnUcacheBuffers(
        k_ring=torch.empty(0, device="cuda"),
        u_ring=torch.empty(0, device="cuda"),
        g_ring=torch.empty(0, device="cuda"),
        hist_len=torch.randint(
            0, 17, (n,), generator=g, device="cuda", dtype=torch.int32
        ),
        cache_base=torch.randint(
            0, 32, (n,), generator=g, device="cuda", dtype=torch.int32
        ),
        flush_min=K.gdn_ucache_flush_min(T),
        by_request_row=by_row,
    )


def _batch(seed):
    """BS rows: distinct slots and request rows, some padded (slot -1 / 0), 1..T accepted."""
    g = torch.Generator(device="cpu").manual_seed(seed)
    slots = (torch.randperm(SLOTS - 1, generator=g)[:BS] + 1).to(torch.int32)
    slots[::7] = -1
    slots[3] = 0  # one row on the padding slot: the conv scatter still writes it
    rows = (torch.randperm(ROWS - 1, generator=g)[:BS] + 1).to(torch.int64)
    accepted = torch.randint(1, T + 1, (BS,), generator=g, dtype=torch.int32)
    return slots.cuda(), rows.cuda(), accepted.cuda()


def _ref_commit(hist, base, slots, ids, accepted, flush_min):
    """``ids``: the cursor index of each row (its slot, or its request row)."""
    hist, base = hist.clone().cpu(), base.clone().cpu()
    for slot, i, a in zip(slots.tolist(), ids.tolist(), accepted.tolist()):
        if slot <= 0:
            continue
        h, b = int(hist[i]), int(base[i])
        if h >= flush_min:  # the verify folded the window: restart at its end
            hist[i], base[i] = a, (b + h) & (K.RING_SLOTS - 1)
        else:
            hist[i] = h + a
    return hist.cuda(), base.cuda()


def _ref_reset(hist, base, ids):
    hist, base = hist.clone(), base.clone()
    live = ids[ids > 0].long()
    hist[live] = 0
    base[live] = 0
    return hist, base


@pytest.mark.parametrize("by_row", [False, True], ids=["by_slot", "by_row"])
def test_commit_cursors(by_row):
    b = _buffers(ROWS if by_row else SLOTS, by_row, seed=1)
    slots, rows, accepted = _batch(seed=2)
    want = _ref_commit(
        b.hist_len,
        b.cache_base,
        slots,
        rows if by_row else slots,
        accepted,
        b.flush_min,
    )
    K.commit_gdn_ucache_cursors(
        buffers=b,
        slots=slots,
        num_accept_tokens=accepted,
        request_rows=rows if by_row else None,
    )
    torch.cuda.synchronize()
    assert torch.equal(b.hist_len, want[0]) and torch.equal(b.cache_base, want[1])


@pytest.mark.parametrize("by_row", [False, True], ids=["by_slot", "by_row"])
def test_reset(by_row):
    b = _buffers(ROWS if by_row else SLOTS, by_row, seed=3)
    slots, rows, _ = _batch(seed=4)
    ids = rows if by_row else slots
    want = _ref_reset(b.hist_len, b.cache_base, ids)
    if by_row:
        K.reset_gdn_ucache_request_rows(buffers=b, request_rows=rows)
    else:
        K.reset_gdn_ucache_cursors(buffers=b, slots=slots)
    torch.cuda.synchronize()
    assert torch.equal(b.hist_len, want[0]) and torch.equal(b.cache_base, want[1])


def test_key_ring_dtype():
    assert K.gdn_ucache_key_ring_dtype("cuda", fp16=False) == torch.bfloat16
    if torch.cuda.get_device_capability() == (10, 7):
        assert K.gdn_ucache_key_ring_dtype("cuda", fp16=True) == torch.float16
    else:
        with pytest.raises(ValueError):
            K.gdn_ucache_key_ring_dtype("cuda", fp16=True)


def _state(by_row):
    with mock.patch.object(R, "_gdn_dims", return_value=(L, H, HV, KD, VD)):
        return R.UcacheState(
            None, tokens_per_request=T, key_ring_fp16=False, by_request_row=by_row
        )


def _view():
    conv = torch.randn(L, SLOTS, CONV_DIM, KM1, device="cuda")
    window = torch.randn(L, ROWS, T, CONV_DIM, KM1, device="cuda")
    mamba = SimpleNamespace(
        temporal=torch.empty(L, SLOTS, 1, 1, 1, device="cuda"),
        conv=(conv,),
        intermediate_conv_window=(window,),
        layer_ids=tuple(range(L)),
    )
    return SimpleNamespace(mamba=mamba, parts=None, num_request_rows=ROWS)


def _io(slots, rows, accepted):
    return SimpleNamespace(
        state_slots=slots,
        track_slots=None,
        request_rows=rows,
        accept_lens=accepted,
        last_step=accepted - 1,
        track_steps=None,
        draft_token_num=T,
        src_indices_raw=None,
    )


def _ref_conv(conv, window, slots, steps):
    conv = conv.clone()
    for i, (slot, step) in enumerate(zip(slots.tolist(), steps.tolist())):
        if step >= 0 and slot >= 0:
            conv[:, slot] = window[:, i, step]
    return conv


@pytest.mark.parametrize("by_row", [False, True], ids=["by_slot", "by_row"])
def test_state_reset_then_commit(by_row):
    """The protocol path: reset_state_slots (a target extend), then a verify commit."""
    state = _state(by_row)
    view = _view()
    state.prepare(view)
    b = state.buffers
    assert b.hist_len.shape[0] == (ROWS if by_row else SLOTS)
    fresh = _buffers(b.hist_len.shape[0], by_row, seed=5)
    b.hist_len.copy_(fresh.hist_len)
    b.cache_base.copy_(fresh.cache_base)
    slots, rows, accepted = _batch(seed=6)
    ids = rows if by_row else slots
    hist, base = _ref_reset(b.hist_len, b.cache_base, ids)
    state.reset_state_slots(slots, rows)
    hist, base = _ref_commit(hist, base, slots, ids, accepted, b.flush_min)
    conv0, window = view.mamba.conv[0].clone(), view.mamba.intermediate_conv_window[0]
    assert state.commit(_io(slots, rows, accepted)) is True
    torch.cuda.synchronize()
    assert torch.equal(b.hist_len, hist) and torch.equal(b.cache_base, base)
    assert torch.equal(
        view.mamba.conv[0], _ref_conv(conv0, window, slots, accepted - 1)
    )


def test_commit_in_a_cuda_graph():
    """Captured once on static inputs, replayed on new ones (a commit graph's use)."""
    state = _state(False)
    view = _view()
    state.prepare(view)
    b = state.buffers
    slots, rows, accepted = _batch(seed=7)
    io = _io(slots.clone(), rows.clone(), accepted.clone())
    static = (io.state_slots, io.request_rows, io.accept_lens, io.last_step)
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):  # compiles the kernels
        state.commit(io)
    torch.cuda.current_stream().wait_stream(side)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        assert state.commit(io) is True
    for seed in (8, 9):
        slots, rows, accepted = _batch(seed=seed)
        for dst, src in zip(static, (slots, rows, accepted, accepted - 1)):
            dst.copy_(src)
        fresh = _buffers(SLOTS, False, seed=seed)
        b.hist_len.copy_(fresh.hist_len)
        b.cache_base.copy_(fresh.cache_base)
        want = _ref_commit(
            b.hist_len, b.cache_base, slots, slots, accepted, b.flush_min
        )
        conv0, window = (
            view.mamba.conv[0].clone(),
            view.mamba.intermediate_conv_window[0],
        )
        graph.replay()
        torch.cuda.synchronize()
        assert torch.equal(b.hist_len, want[0]) and torch.equal(b.cache_base, want[1])
        assert torch.equal(
            view.mamba.conv[0], _ref_conv(conv0, window, slots, accepted - 1)
        )
