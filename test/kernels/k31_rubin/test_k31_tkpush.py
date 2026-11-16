# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""K31MK_DSA_REQ_SHARD_TKPUSH (kimi_k31/dsa/tkpush.py, dsa/x47960/topk_v2_tkpush.cuh, dsa/tag_push.cuh): the owner top-k
stores its output rows as tagged words into every rank's mailbox (tag route of dsa/tag_gather.py without _push_kernel).

All GPU, under torchrun (``K31_TKPUSH_GPU=1``, one process per GPU, world >= 2). References are independent of the code under
test: the rows every rank decodes from its mailbox must equal an NCCL all_gather of every rank's top-k output of the SAME call
(bitwise), and the raw mailbox words must equal tag_gather.encode (the host form of the word format) of those rows with the
epoch counted on the host. The top-k output itself must equal the unmodified x47960 module's (X47960Select) as a slot set per
row (the selection may emit another order of the same set on every call above 2048 keys).
- words: score widths above the small-batch cluster floor (in-kernel push expected) with per-row lengths that hit all four
  kernel paths (trivial <= 2048, register <= 16384, streaming <= floor, cluster above), 8 and 16 owner rows; and a width
  below the floor (no in-kernel push: the push_ext fallback).
- late rank: rank 1 waits ~20 ms before every poll (after its own top-k pushed), rank 2 ~10 ms before every other top-k, over
  4 forwards of 2 and 3 layers; fresh scores every call. A writer that reused a copy a slow peer still reads overwrites its
  words with the next use's tag: that peer's poll never sees its tag and traps (the K31_MUTATION one_copy control).
- graph: one 3-layer forward captured in a CUDA graph (rank 1's ~5 ms wait inside), replayed 4 times: the epochs advance on
  the device, so every replay must decode its own rows.
Mutants (K31_MUTATION): each must make a test here fail (trap or mismatch)."""

import os

import pytest
import torch

import _harness as H

TP = "kimi_k31/dsa/tag_push.cuh"
EG = "kimi_k31/dsa/epoch_gather.py"
TK = "kimi_k31/dsa/tkpush.py"
MUTATIONS = {
    # tag one epoch old (the poll waits for counter + 1): every poll times out and traps
    "tk_tag_old": (TP, "return ((e + 1u) & 0x7Fu) << 25;", "return (e & 0x7Fu) << 25;"),
    # the last rank never receives the words: its poll traps
    "tk_skip_peer": (TP, "    if (j < p.world) {\n", "    if (j + 1 < p.world) {\n"),
    # every row lands on row 0's words: rows collide, the other rows' words keep the previous tag (trap) or mismatch
    "tk_row": (TP, "const int64_t off = p.base + static_cast<int64_t>(row) * p.row_words + col;",
               "const int64_t off = p.base + col;"),
    # v instead of v + 1 in the word: decoded ids are off by one (bitwise mismatch)
    "tk_enc": (TP, "return tagv | (static_cast<uint32_t>(v + 1) & 0x1FFFFFFu);", "return tagv | (static_cast<uint32_t>(v) & 0x1FFFFFFu);"),
    # this rank's block lands at rank 0's offset
    "tk_base": (TK, "self.target = (int(c), epoch, (int(c) * self.world + self.rank) * self.n_own)",
                "self.target = (int(c), epoch, (int(c) * self.world) * self.n_own)"),
    # race control: every call uses copy 0, so a fast writer overwrites a copy a late peer has not read yet
    "one_copy": (EG, "    return ordinal % 2\n", "    return 0\n"),
}
GPU = pytest.mark.skipif(os.environ.get("K31_TKPUSH_GPU") != "1", reason="multi-GPU test: torchrun with K31_TKPUSH_GPU=1")
DEADLINE_NS = 2_000_000_000
# K31_TKPUSH_NOFENCE=1: the same tests on the TAGNF form (K31MK_DSA_REQ_SHARD_TAG_NOFENCE: tag exchange and top-k push without their
# fence.acq_rel.sys;  T1-T3). Under it the one_copy race control must still end in a TRAP (never wrong rows): the runner
# classifies that mutant's failure by its log (deadline trap vs row mismatch).
NOFENCE = os.environ.get("K31_TKPUSH_NOFENCE") == "1"
PAGE_IDS = (1 << 19) - 1  # page ids below this: slot ids (page << 6 | offset) stay below tag_gather.VALUE_LIMIT


def _dist():
    import torch.distributed as dist

    if not dist.is_initialized():
        torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
        import datetime

        # short collective timeout: a rank that traps (a mutant) must not leave its peers in an all_gather for 10 min
        dist.init_process_group("nccl", device_id=torch.device("cuda", int(os.environ["LOCAL_RANK"])),
                                timeout=datetime.timedelta(seconds=90))
    return dist, dist.get_world_size(), dist.get_rank()


def _mods():
    H.load_plugin(MUTATIONS)
    from sgl_nvidia_mega.kimi_k31.dsa import epoch_gather as E
    from sgl_nvidia_mega.kimi_k31.dsa import tag_gather as T
    from sgl_nvidia_mega.kimi_k31.dsa import tkpush as K
    from sgl_nvidia_mega.kimi_k31.dsa.x47960 import X47960Select

    return E, T, K, X47960Select


def _inputs(rows, width, lens, seed):
    g = H.generator(seed)
    scores = torch.randn((rows, width), generator=g, device="cuda", dtype=torch.float32)
    causal = torch.tensor(lens, dtype=torch.int32, device="cuda")
    table = torch.randint(0, PAGE_IDS, (rows, -(-width // 64)), generator=g, device="cuda", dtype=torch.int32)
    table[:, -1] = PAGE_IDS - 1  # the largest page id among the slots
    return scores, causal, table


def _setup(rows, layers, world, rank, dist):
    E, T, K, X = _mods()
    n_own = rows * 2048
    K.load(1, 0, NOFENCE)  # every rank builds before the TagGather barrier (as ReqShardState does)
    tg = T.TagGather(dist.group.WORLD, rank, world, n_own, E.copies_for(layers), torch.device("cuda"), DEADLINE_NS, ext=True,
                     fence=not NOFENCE)
    tk = K.TkPushSelect(torch.device("cuda"), rows, tg.peers(), world, rank, n_own, nofence=NOFENCE)
    tg.self_check(layers)
    uses = [0] * E.copies_for(layers)
    uses[E.copy_for(layers - 1, layers)] = 1  # the self-check's gather
    return E, T, K, X, tg, tk, uses


def _push(tg, tk, c, scores, causal, table, sel):
    """One owner top-k call with the push target armed, plus the fallback push when the kernel did not push."""
    tk.arm(c, tg.tag_epoch(c))
    tk.run(scores, causal, table, sel)
    pushed = tk.take()
    if not pushed:
        tg.push_ext(c, sel.view(-1), tg.tag_epoch(c))
    return pushed


def _gathered(dist, world, sel):
    out = torch.empty((world,) + tuple(sel.shape), dtype=sel.dtype, device=sel.device)
    dist.all_gather_into_tensor(out, sel.contiguous())
    return out.view(-1, sel.shape[1])


# (rows, width, per-row causal lengths, in-kernel push expected): 2048 / 16384 / 24576 / 30720 are the kernel's path bounds
CELLS = {
    "r8_131k_paths": (8, 131072, [1000, 2048, 9000, 16384, 20000, 50000, 131072, 131065], True),
    "r16_131k": (16, 131072, [131072 - 7 + j % 8 for j in range(16)], True),
    "r8_16k_fallback": (8, 16384, [3000, 2048, 16384, 16000, 9000, 16380, 12000, 1], False),
}


@GPU
@pytest.mark.parametrize("cell", list(CELLS))
def test_tkpush_words_equal_gathered_rows(cell):
    dist, world, rank = _dist()
    rows, width, lens, expect_push = CELLS[cell]
    E, T, K, X, tg, tk, uses = _setup(rows, 2, world, rank, dist)
    ref = X(torch.device("cuda"), rows)
    scores, causal, table = _inputs(rows, width, lens, 7000 + 31 * rank + rows)
    sel = torch.full((rows, 2048), -7, dtype=torch.int32, device="cuda")
    sel_ref = torch.full((rows, 2048), -7, dtype=torch.int32, device="cuda")
    sel_off = torch.full((rows, 2048), -7, dtype=torch.int32, device="cuda")
    fails, same_order = [], []
    for k in range(3):  # copies c0, c1, c0: epochs 1, 1, 2 (the self-check used c1 once: 2 there)
        c = E.copy_for(k % 2, 2)
        pushed = _push(tg, tk, c, scores, causal, table, sel)
        got = tg.poll(c).clone()
        words = tg.words(c).clone()
        uses[c] += 1
        ref.run(scores, causal, table, sel_ref)
        tk.run(scores, causal, table, sel_off)  # not armed: the fork's flag-off kernel (no push)
        torch.cuda.synchronize()
        want = _gathered(dist, world, sel)
        if pushed != expect_push:
            fails.append(f"call {k}: pushed {pushed}, expected {expect_push}")
        if not torch.equal(got.view(-1, 2048), want):
            fails.append(f"call {k}: decoded rows differ from the gathered top-k rows in {int((got.view(-1, 2048) != want).any(1).sum())} rows")
        if not torch.equal(words.view(-1, 2048), T.encode(want, uses[c])):
            fails.append(f"call {k}: mailbox words differ from encode(rows, epoch {uses[c]})")
        a = [frozenset(r.tolist()) for r in sel.cpu()]
        b = [frozenset(r.tolist()) for r in sel_ref.cpu()]
        o = [frozenset(r.tolist()) for r in sel_off.cpu()]
        if a != b:
            fails.append(f"call {k}: top-k slot sets differ from X47960Select in rows {[i for i in range(rows) if a[i] != b[i]][:4]}")
        if o != b:
            fails.append(f"call {k}: flag-off fork slot sets differ from X47960Select in rows {[i for i in range(rows) if o[i] != b[i]][:4]}")
        same_order.append(int((sel == sel_ref).all(1).sum()))  # INFO: rows in the same order as X47960Select (order may vary per call)
    H.record(dict(test="tkpush_words", nofence=NOFENCE, cell=cell, world=world, rank=rank, rows=rows, width=width, expect_push=expect_push,
                  calls=tk.calls, fallbacks=tk.fallbacks, rows_same_order=same_order, fails=fails[:6]))
    assert not fails, f"rank {rank} {cell}: {fails[:6]}"


@GPU
@pytest.mark.parametrize("layers", [2, 3])
def test_tkpush_late_rank(layers):
    dist, world, rank = _dist()
    rows, width = 8, 65536
    lens = [width - 7 + j for j in range(rows)]
    E, T, K, X, tg, tk, uses = _setup(rows, layers, world, rank, dist)
    comm = torch.cuda.Stream()
    sel = torch.full((rows, 2048), -7, dtype=torch.int32, device="cuda")
    saved, decoded = [], []
    for k in range(4 * layers):
        c = E.copy_for(k % layers, layers)
        scores, causal, table = _inputs(rows, width, lens, 9000 + 101 * k + rank)
        cur = torch.cuda.current_stream()
        if rank == 2 and k % 2:
            torch.cuda._sleep(20_000_000)  # ~10 ms late top-k (and push)
        _push(tg, tk, c, scores, causal, table, sel)
        saved.append(sel.clone())
        comm.wait_stream(cur)
        with torch.cuda.stream(comm):
            if rank == 1:
                torch.cuda._sleep(40_000_000)  # ~20 ms late poll: its peers push the next calls meanwhile
            decoded.append(tg.poll(c).clone())
        cur.wait_stream(comm)
    torch.cuda.synchronize()
    bad = []
    for k in range(4 * layers):
        want = _gathered(dist, world, saved[k])
        if not torch.equal(decoded[k].view(-1, 2048), want):
            bad.append(k)
    H.record(dict(test="tkpush_late_rank", nofence=NOFENCE, layers=layers, world=world, rank=rank, calls=tk.calls, fallbacks=tk.fallbacks,
                  bad_calls=bad))
    assert tk.fallbacks == 0, f"rank {rank}: {tk.fallbacks} calls fell back to push_ext at width {width}"
    assert not bad, f"rank {rank}: calls whose decoded rows differ from the gathered top-k rows: {bad}"


@GPU
def test_tkpush_graph_replays():
    dist, world, rank = _dist()
    rows, width, layers = 8, 65536, 3
    lens = [width - 7 + j for j in range(rows)]
    E, T, K, X, tg, tk, uses = _setup(rows, layers, world, rank, dist)
    comm = torch.cuda.Stream()
    ins = [_inputs(rows, width, lens, 11000 + 13 * o + rank) for o in range(layers)]
    sel = torch.full((rows, 2048), -7, dtype=torch.int32, device="cuda")
    saved = [torch.empty_like(sel) for _ in range(layers)]
    decoded = [torch.empty((world * rows, 2048), dtype=torch.int32, device="cuda") for _ in range(layers)]

    def forward():
        for o in range(layers):
            c = E.copy_for(o, layers)
            cur = torch.cuda.current_stream()
            _push(tg, tk, c, *ins[o], sel)
            saved[o].copy_(sel)
            comm.wait_stream(cur)
            with torch.cuda.stream(comm):
                if rank == 1:
                    torch.cuda._sleep(10_000_000)
                decoded[o].copy_(tg.poll(c).view(-1, 2048))
            cur.wait_stream(comm)

    forward()  # eager forward (also warms every launch path before capture)
    torch.cuda.synchronize()
    dist.barrier()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        forward()
    torch.cuda.synchronize()
    dist.barrier()
    bad = []
    for rep in range(4):
        graph.replay()
        torch.cuda.synchronize()
        for o in range(layers):
            if not torch.equal(decoded[o], _gathered(dist, world, saved[o])):
                bad.append((rep, o))
    H.record(dict(test="tkpush_graph", nofence=NOFENCE, world=world, rank=rank, calls=tk.calls, fallbacks=tk.fallbacks, bad=bad))
    assert not bad, f"rank {rank}: (replay, layer) with decoded rows != gathered top-k rows: {bad}"
