# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""K31MK_DSA_REQ_SHARD (kimi_k31/dsa/req_shard.py): request-sharded DSA selection.

CPU tests (no GPU): the owner map. Reference = the full selection's row order: every row r of a B x q verify step
must come back at position r after each rank selects its owner rows and the ranks' blocks are concatenated in rank
order (the all-gather), whatever B, q and the world size; every rank's block has the same static length.
GPU test (``K31_REQ_SHARD_GPU=1`` under torchrun, one process per GPU): ReqShardState.select vs VerifySelection.run
over all rows on the same replicated inputs (P5's generator): every row's selection equal as a slot set, for every
all-gather route (K31MK_DSA_REQ_SHARD_AG nccl / mm / epoch / tag; ``K31_REQ_SHARD_AG`` limits the run to one).
Epoch and tag gathers (dsa/epoch_gather.py, dsa/tag_gather.py): CPU, consecutive calls never share a copy (layer wrap
included); GPU, every rank's rows arrive intact over 3 forwards while rank 1 pushes ~25 ms late (a reader that does not wait
reads stale rows), and no gather call compiles a new kernel variant after construction.
Fused route (K31MK_DSA_REQ_SHARD_AG=fused: push only, the sparse core polls the tagged words): GPU, the tagged core equals the
served core (SparseMla6.run) bitwise on the same rows while rank 1 pushes late, and select()'s mailbox words decode to the full
selection (slot sets) with the core bitwise on them.
Mutants (K31_MUTATION, harness.load_plugin): each must make a test here fail."""

import importlib.util
import os

import pytest
import torch

import _harness as H

RS = "kimi_k31/dsa/req_shard.py"
EG = "kimi_k31/dsa/epoch_gather.py"
TG = "kimi_k31/dsa/tag_gather.py"
SM = "kimi_k31/dsa/sparse_mla6.cuh"
MUTATIONS = {
    "floor_per": (RS, "    per = -(-b // world)\n    return [min(rank * per + j, b - 1)", "    per = b // world\n    return [min(rank * per + j, b - 1)"),
    "strided_owner": (RS, "return [min(rank * per + j, b - 1) for j in range(per)]", "return [min(j * world + rank, b - 1) for j in range(per)]"),
    "row_stride": (RS, "return [r * q + j for r in owner_requests(b, world, rank) for j in range(q)]",
                   "return [r * 8 + j for r in owner_requests(b, world, rank) for j in range(q)]"),
    "gate_max": (RS, "int(seq_lens.min()) >= min_context", "int(seq_lens.max()) >= min_context"),
    "gather_tail": (RS, "return w.gathered[: w.b * self.q]", "return w.gathered[-w.b * self.q :]"),
    # K31MK_DSA_REQ_SHARD_SLICE=1: the owner slice starts one request late / ignores the clamp (a padded owner block sliced as if real)
    "sl_offset": (RS, "query[w.own_lo : w.own_lo + w.per * self.q]", "query[w.own_lo + self.q : w.own_lo + self.q + w.per * self.q]"),
    "sl_noclamp": (RS, "    return rows[0] if rows == list(range(rows[0], rows[0] + len(rows))) else None", "    return min(rows[0], (b - 1) * q)"),
    # K31MK_DSA_REQ_SHARD_AG=mm: no store (the buffer keeps whatever it held) / half of each block stored at the wrong offset
    "mm_stale": (RS, "                self.mm.all_gather_inner(w.mm, own.view(torch.bfloat16).view(1, -1), tp_hidden_dim=w.mm_width,\n"
                     "                                         skip_entry_sync=False, safe=False)",
                 "                pass"),
    "mm_half": (RS, "own.view(torch.bfloat16).view(1, -1), tp_hidden_dim=w.mm_width,",
                "own.view(torch.bfloat16).view(1, -1)[:, : w.mm_width // (2 * self.world)], tp_hidden_dim=w.mm_width // 2,"),
    # epoch route: odd layer counts reuse a copy across the forward boundary / the reader counts the epoch but does not
    # wait for the peers' flags / every rank's rows land on rank 0's block
    "eg_copy_parity": (EG, "    if layers % 2 and ordinal == layers - 1:\n        return 2\n", ""),
    "eg_no_wait": (EG, "        _acquire_eq(fl_ptr + (c * W + rr) * NB + b, want, deadline_ns)\n", ""),
    "eg_offset": (EG, "(RANK * n_own_chunks + offs) * 2", "(offs) * 2"),
    # epoch route: per-call ints specialized again (c / p values compile new variants inside a step)
    "eg_specialize": (EG, '@triton.jit(do_not_specialize=["src_ptr", "mc_ptr", "c", "p", "n_own_chunks", "deadline_ns"])',
                      "@triton.jit"),
    # tag route: the poll accepts words without checking their tag / every rank's block lands at rank 0's offset /
    # per-call ints specialized again
    "tg_no_check": (TG, "    dirty = mask & _stale(x, y, z, w, tg)\n    t0", "    dirty = mask & (offs < 0)\n    t0"),
    # (the target names _push_kernel's line by its trailing barrier comment: tkpush 03e6a5b added _push_ext_kernel with the same dst line)
    "tg_offset": (TG, "RANK * n_own_chunks + offs) * 2\n    _st4(dst, _enc(x, tagv), _enc(y, tagv), _enc(z, tagv), _enc(w, tagv), mask)\n    _bar()  # every thread",
                  "offs) * 2\n    _st4(dst, _enc(x, tagv), _enc(y, tagv), _enc(z, tagv), _enc(w, tagv), mask)\n    _bar()  # every thread"),
    "tg_specialize": (TG, '@triton.jit(do_not_specialize=["src_ptr", "c", "n_own_chunks"])', "@triton.jit"),
    # fused route (the sparse core polls the tagged words): the core accepts words without checking their tag / reads every
    # row from rank 0's block / never advances its use counter (the next use of a copy waits for a stale tag: trap) /
    # select hands the core the next copy instead of the layer's copy
    "fz_no_check": (SM, "  while ((v >> 25) != tg) {", "  while (false && (v >> 25) != tg) {"),
    "fz_row_offset": (SM, "const uint32_t* mb = p.tagged + static_cast<int64_t>(row) * kTopK + split * T;",
                      "const uint32_t* mb = p.tagged + static_cast<int64_t>(row % p.heart_rows) * kTopK + split * T;"),
    "fz_no_advance": (SM, "    if (tid == 0) p.epochs[static_cast<int64_t>(row) * kSplits + split] = static_cast<int32_t>(epoch);\n", ""),
    "fz_copy": (RS, "return TaggedRows(w.tg.words(c).view(", "return TaggedRows(w.tg.words((c + 1) % self.copies).view("),
    # tag / fused routes: the pool check refuses one slot too early / counts slots from bytes with the FP8 row size (the
    # pre-pool_slots count: under-counts a 68-byte MXFP4 index pool) / from the last dim (over-counts a 4-D view)
    "pool_ge": (RS, "    if slots > VALUE_LIMIT:\n", "    if slots >= VALUE_LIMIT:\n"),
    "pool_numel": (RS, "    return int(cache.shape[0]) * PAGE\n", "    return cache.numel() // 132\n"),
    "pool_lastdim": (RS, "    return int(cache.shape[0]) * PAGE\n", "    return cache.numel() // (cache.shape[-1] // PAGE)\n"),
    # TKPUSH wiring as 979e3d4 had it: the push top-k is installed only under TAGNF (with TKPUSH=1 and fences on it never runs)
    "x47_nofence_only": (RS, "                    w.selection.x47 = w.tkp  # with TKPUSH only (979e3d4 had this line under ``if self.nofence``)\n"
                             "                if self.nofence:\n",
                         "                if self.nofence:\n                    w.selection.x47 = w.tkp\n"),
    # race control (not a code mutant): every layer uses mailbox copy 0, so a peer can overwrite a copy a late reader has not read
    "one_copy": (EG, "    return ordinal % 2\n", "    return 0\n"),
    # the T3 classifier's positive control: the same race with a core that accepts any tag must show WRONG rows (not a trap)
    "one_copy_no_check": [(EG, "    return ordinal % 2\n", "    return 0\n"),
                          (SM, "  while ((v >> 25) != tg) {", "  while (false && (v >> 25) != tg) {")],
}
# K31_FUSED_NOFENCE=1: the fused-route tests run the TAGNF form (TagGather fence=False and TaggedRows fence=False -> the sparse core's
# kFence = false; ReqShardState under K31MK_DSA_REQ_SHARD_TAG_NOFENCE). K31_FUSED_LATE_READER=1: rank 0 also sleeps ~50 ms between its
# push and its core (rank 1 pushes ~25 ms late, so ranks 2 / 3 finish use k and push use k + 1 first); under the one_copy race control its peers then overwrite rank 0's copy with the next use before its core reads
# it: condition T3 - the core must trap at its deadline or read correct rows, never wrong rows (the runner classifies
# the failure by its log: deadline trap vs 'differs from the served core').
FUSED_NOFENCE = os.environ.get("K31_FUSED_NOFENCE") == "1"
FUSED_LATE_READER = os.environ.get("K31_FUSED_LATE_READER") == "1"


def _rs():
    H.load_plugin(MUTATIONS)
    from sgl_nvidia_mega.kimi_k31.dsa import req_shard

    return req_shard


CASES = [(b, q, w) for w in (2, 4, 16) for q in (1, 8) for b in (1, 2, 3, 4, 5, 7, 8, 16, 17, 33)]


@pytest.mark.parametrize("b,q,world", CASES, ids=[f"b{b}_q{q}_w{w}" for b, q, w in CASES])
def test_owner_map_restores_row_order(b, q, world):
    rs = _rs()
    blocks = [rs.owner_rows(b, q, world, r) for r in range(world)]
    per = -(-b // world)
    assert all(len(x) == per * q for x in blocks), "every rank's owner block must have the same static length"
    assert all(0 <= i < b * q for x in blocks for i in x), "owner rows must be real rows"
    gathered = [i for x in blocks for i in x]  # rank-major, as all_gather_into_tensor lays the blocks out
    assert gathered[: b * q] == list(range(b * q)), "the first B x q gathered rows must be the rows in order"


def test_epoch_copies_alternate():
    """Reference = the reuse rule the epoch gather's WAR ordering needs: over consecutive forwards, two consecutive calls
    (the last layer and the next forward's first included) never use the same copy, and every copy index is allocated."""
    H.load_plugin(MUTATIONS)
    from sgl_nvidia_mega.kimi_k31.dsa import epoch_gather as E

    for layers in (0, 1):
        with pytest.raises(ValueError):
            E.copy_for(0, layers)
    for layers in range(2, 34):
        seq = [E.copy_for(o, layers) for _ in range(3) for o in range(layers)]
        assert all(a != b for a, b in zip(seq, seq[1:])), f"layers {layers}: consecutive calls share a copy: {seq[:layers + 2]}"
        assert 0 <= min(seq) and max(seq) < E.copies_for(layers), f"layers {layers}: copy outside the {E.copies_for(layers)} allocated"


@pytest.mark.parametrize("slots,ok", [(1 << 20, True), ((1 << 25) - 1, True), (1 << 25, False), (3 << 25, False)])
def test_slot_pool_refusal(slots, ok):
    """Reference = the word format (tag_gather.py): a slot id v travels as v + 1 in 25 bits, so an index-K pool of ``slots``
    token slots (ids 0 .. slots - 1) fits iff slots - 1 + 1 <= 2**25 - 1. The largest pool that fits must pass (the boundary),
    every larger one must raise before any gather."""
    rs = _rs()
    if ok:
        rs.check_slot_pool(slots, "tag")
    else:
        with pytest.raises(ValueError):
            rs.check_slot_pool(slots, "fused")


@pytest.mark.parametrize("row", [132, 68])
@pytest.mark.parametrize("four_d", [False, True])
def test_pool_slots_counts_pages(row, four_d):
    """Reference = the pool's definition: ``pages`` pages of 64 token slots, whatever the row bytes (132: FP8 + scale; 68:
    MXFP4 + scale) and whether the buffer is [pages, 64 x row] or viewed [pages, 64, 1, row]. Shapes only (meta tensors)."""
    rs = _rs()
    for pages in (3, 1000, 524287):
        shape = (pages, 64, 1, row) if four_d else (pages, 64 * row)
        got = rs.pool_slots(torch.empty(shape, dtype=torch.uint8, device="meta"))
        assert got == pages * 64, f"row {row} 4-D {four_d}: {pages} pages -> {got} slots, want {pages * 64}"


@pytest.mark.parametrize("lens,expect", [
    ([131072] * 4, True), ([131072, 131072, 131072, 8192], False), ([131072] * 2, False), ([65536] * 8, True),
    ([65535] + [300000] * 7, False), ([], False),
])
def test_gate_shortest_request_decides(lens, expect):
    rs = _rs()
    assert rs.req_shard_label(torch.tensor(lens, dtype=torch.int64), 4, 65536) is expect


@pytest.mark.skipif(importlib.util.find_spec("sglang") is None, reason="imports kimi_k31/dsa/verify.py (needs the SGLang host)")
def test_label_routing(monkeypatch):
    """tsi_graph_label: R1's gate when K31MK_DSA_REQ_SHARD is on (TSI's rows / context gate then unused); TSI's own policy
    unchanged when it is off; never with both off."""
    H.load_plugin(MUTATIONS)
    from sgl_nvidia_mega.kimi_k31.dsa import verify as V

    t = lambda *ctx: torch.tensor(ctx, dtype=torch.int64)  # noqa: E731
    monkeypatch.setattr(V, "REQ_SHARD", True)
    monkeypatch.setattr(V, "TSI", True)
    g = V.REQ_SHARD_MIN_CONTEXT
    assert V.tsi_graph_label(t(g, g, g, g), 8) and not V.tsi_graph_label(t(g, g, g, g - 1), 8), "R1 context boundary"
    assert not V.tsi_graph_label(t(900000, 900000), 8), "R1 needs >= K31MK_DSA_REQ_SHARD_MIN_REQ requests (TSI would take rows 16)"
    monkeypatch.setattr(V, "REQ_SHARD", False)
    assert V.tsi_graph_label(t(900000), 8), "knob off: TSI's rows-8 policy unchanged"
    assert not V.tsi_graph_label(t(900000, 900000, 900000, 900000), 8), "knob off: TSI never takes rows 32"
    monkeypatch.setattr(V, "TSI", False)
    assert not V.tsi_graph_label(t(900000, 900000, 900000, 900000), 8), "both off: never the third graph"


class _TorchGroup:
    """GroupCoordinator's all_gather_into_tensor on a torch.distributed process group (test only)."""

    def all_gather_into_tensor(self, output, input):
        torch.distributed.all_gather_into_tensor(output, input)


@pytest.mark.skipif(os.environ.get("K31_REQ_SHARD_GPU") != "1", reason="multi-GPU test: torchrun with K31_REQ_SHARD_GPU=1")
# b 5 / 6 at world 4 pad the owner blocks (2 requests per owner, the last repeated): the gathered tail then holds
# rows the selection must drop, which the other cells (b a multiple of world) never exercise.
@pytest.mark.parametrize("ag", [a for a in ("nccl", "mm", "epoch", "tag") if (os.environ.get("K31_REQ_SHARD_AG") or a) == a])
@pytest.mark.parametrize("ctx,b", [(32768, 4), (131072, 4), (131072, 8), (300000, 4), (32768, 5), (131072, 6)])
def test_sharded_equals_full_selection(ctx, b, ag):
    import math

    import deep_gemm
    import torch.distributed as dist
    from sglang.kernels.ops.attention.dsv4.topk import plan_topk_v2
    from types import SimpleNamespace

    rs = _rs()
    from sgl_nvidia_mega.kimi_k31.dsa import verify as V
    import _ref_mla as R
    import test_k31_mla as M  # index_pool / requests (P5's generator)

    if not dist.is_initialized():
        torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
        dist.init_process_group("nccl", device_id=torch.device("cuda", int(os.environ["LOCAL_RANK"])))
    world, rank, q = dist.get_world_size(), dist.get_rank(), 8
    rows = q * b
    g = H.generator(5000 + ctx % 997 + 17 * b)
    npg = math.ceil(ctx / 64)
    pages = b * npg + 1
    cache = M.index_pool(pages, g)
    table = M.requests(b, ctx, pages, g)[0]
    query = R.fp8_bytes(torch.randn((rows, 32, 128), generator=g, device="cuda") * 100).view(R.FP8)  # as BIX-R1 / P5
    gates = (torch.randn((rows, 32), generator=g, device="cuda") * 1e-4).contiguous().unsqueeze(-1)
    for t in (cache, query.view(torch.uint8), gates, table):  # replicated inputs: rank 0's bytes everywhere
        dist.broadcast(t, 0)
    contexts = torch.full((b, q), ctx, dtype=torch.int32, device="cuda")
    causal = (ctx - q + 1 + torch.arange(q, device="cuda")).repeat(b).to(torch.int32)
    step = SimpleNamespace(batch_requests=b, query_width=q, score_contexts=contexts, causal_lengths=causal,
                           score_schedule=deep_gemm.get_paged_mqa_logits_metadata(contexts, 64, M.num_sms()),
                           topk_plan=plan_topk_v2(causal))
    full = V.VerifySelection(torch.device("cuda"), rows).run(query, gates, cache, table, step).clone()
    state = rs.ReqShardState(_TorchGroup(), world, rank, torch.device("cuda"), (rows,), q, ag=ag, layers=2)
    state.prepare(table, contexts, causal)
    shard = state.select(0, query, gates, cache).clone()
    torch.cuda.synchronize()
    a = [frozenset(x for x in r.tolist() if x >= 0) for r in full.cpu()]
    s = [frozenset(x for x in r.tolist() if x >= 0) for r in shard.cpu()]
    bad = [i for i in range(rows) if a[i] != s[i]]
    H.record(dict(test="req_shard_gpu", ag=ag, ctx=ctx, b=b, world=world, rank=rank, rows=rows, rows_differ=len(bad),
                  first_bad=bad[:4], mutation=os.environ.get("K31_MUTATION", "")))
    assert not bad, f"rank {rank}: {len(bad)} of {rows} rows differ from the full selection (first {bad[:4]})"


@pytest.mark.skipif(os.environ.get("K31_REQ_SHARD_GPU") != "1", reason="multi-GPU test: torchrun with K31_REQ_SHARD_GPU=1")
@pytest.mark.parametrize("layers", [2, 3])
def test_epoch_gather_waits_for_late_rank(layers):
    """Reference = the payload generator (every rank computes every rank's rows). Rank 1 sleeps ~25 ms on its stream before each
    push, so a reader that does not wait for rank 1's epoch flag reads rank 1's previous rows. 3 forwards of ``layers`` calls,
    copies by copy_for, rows include -1 words (an f32 NaN bit pattern on the multimem store)."""
    import torch.distributed as dist

    H.load_plugin(MUTATIONS)
    from sgl_nvidia_mega.kimi_k31.dsa import epoch_gather as E

    if not dist.is_initialized():
        torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
        dist.init_process_group("nccl", device_id=torch.device("cuda", int(os.environ["LOCAL_RANK"])))
    world, rank = dist.get_world_size(), dist.get_rank()
    n_own = 2 * 8 * 2048  # 2 requests x 8 verify rows x 2048 slots
    eg = E.EpochGather(dist.group.WORLD, rank, world, n_own, E.copies_for(layers), torch.device("cuda"), 2_000_000_000)
    eg.self_check(layers)
    i = torch.arange(n_own, dtype=torch.int64, device="cuda")

    def rows(r, k):
        v = (r * 7919 + k * 104729 + i * 2654435761) % (1 << 22)
        return torch.where(i % 97 == 5, torch.full_like(v, -1), v).to(torch.int32)

    bad = []
    for k in range(3 * layers):
        o = k % layers
        if rank == 1:
            torch.cuda._sleep(50_000_000)
        got = eg.gather(E.copy_for(o, layers), E.copy_for((o - 1) % layers, layers), rows(rank, k)).clone()
        want = torch.cat([rows(r, k) for r in range(world)])
        bad.append(int((got.view(world, -1) != want.view(world, -1)).any(1).sum()))
    torch.cuda.synchronize()
    H.record(dict(test="epoch_gather_late_rank", layers=layers, world=world, rank=rank, bad_ranks_per_call=bad,
                  mutation=os.environ.get("K31_MUTATION", "")))
    assert not any(bad), f"rank {rank}: calls with stale or wrong rank blocks: {bad}"


def _variants(fn):
    """Compiled variants of a Triton JITFunction on every device (Triton 3.x keeps them in device_caches, older in cache)."""
    caches = getattr(fn, "device_caches", None)
    if caches is not None:
        return sum(len(v[0]) for v in caches.values())
    return sum(len(v) for v in fn.cache.values())


@pytest.mark.skipif(os.environ.get("K31_REQ_SHARD_GPU") != "1", reason="multi-GPU test: torchrun with K31_REQ_SHARD_GPU=1")
def test_epoch_gather_compiles_before_first_call():
    """Reference = the protocol's own requirement: no rank may JIT-compile inside a gather while its peers spin (a compile of
    hundreds of ms can outlast the peers' deadline and trap them). After construction (which compiles), a
    forward of 3 layers (copies 0, 1, 2 and every previous-copy value) plus the self-check must compile nothing new."""
    import torch.distributed as dist

    H.load_plugin(MUTATIONS)
    from sgl_nvidia_mega.kimi_k31.dsa import epoch_gather as E

    if not dist.is_initialized():
        torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
        dist.init_process_group("nccl", device_id=torch.device("cuda", int(os.environ["LOCAL_RANK"])))
    world, rank = dist.get_world_size(), dist.get_rank()
    n_own = 8 * 2048
    eg = E.EpochGather(dist.group.WORLD, rank, world, n_own, E.copies_for(3), torch.device("cuda"), 2_000_000_000)
    n0 = (_variants(E._push_kernel), _variants(E._wait_kernel))
    eg.self_check(3)
    own = torch.arange(n_own, dtype=torch.int32, device="cuda") + rank
    for o in range(3):
        eg.gather(E.copy_for(o, 3), E.copy_for((o - 1) % 3, 3), own)
    torch.cuda.synchronize()
    n1 = (_variants(E._push_kernel), _variants(E._wait_kernel))
    H.record(dict(test="epoch_gather_compiles_once", world=world, rank=rank, variants_after_init=n0, variants_after_calls=n1,
                  mutation=os.environ.get("K31_MUTATION", "")))
    assert n0[0] >= 1 and n0[1] >= 1, f"rank {rank}: construction compiled nothing: {n0}"
    assert n1 == n0, f"rank {rank}: gather calls compiled new variants: {n0} -> {n1}"


def _dist():
    import torch.distributed as dist

    if not dist.is_initialized():
        torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
        dist.init_process_group("nccl", device_id=torch.device("cuda", int(os.environ["LOCAL_RANK"])))
    return dist, dist.get_world_size(), dist.get_rank()


@pytest.mark.skipif(os.environ.get("K31_REQ_SHARD_GPU") != "1", reason="multi-GPU test: torchrun with K31_REQ_SHARD_GPU=1")
@pytest.mark.parametrize("layers", [2, 3])
def test_tag_gather_waits_for_late_rank(layers):
    """Reference = the payload generator (every rank computes every rank's rows), with -1 and the largest carried slot id
    (VALUE_LIMIT - 1) among the words. Rank 1 sleeps ~25 ms on its stream before each push, so a poll that does not check
    the tags reads rank 1's previous rows. 3 forwards of ``layers`` calls, copies by copy_for."""
    H.load_plugin(MUTATIONS)
    from sgl_nvidia_mega.kimi_k31.dsa import epoch_gather as E
    from sgl_nvidia_mega.kimi_k31.dsa import tag_gather as T

    dist, world, rank = _dist()
    n_own = 2 * 8 * 2048
    tg = T.TagGather(dist.group.WORLD, rank, world, n_own, E.copies_for(layers), torch.device("cuda"), 2_000_000_000)
    tg.self_check(layers)
    i = torch.arange(n_own, dtype=torch.int64, device="cuda")

    def rows(r, k):
        v = (r * 7919 + k * 104729 + i * 2654435761) % (1 << 22)
        v = torch.where(i % 97 == 5, torch.full_like(v, -1), v)
        return torch.where(i % 89 == 7, torch.full_like(v, T.VALUE_LIMIT - 1), v).to(torch.int32)

    bad = []
    for k in range(3 * layers):
        if rank == 1:
            torch.cuda._sleep(50_000_000)
        got = tg.gather(E.copy_for(k % layers, layers), rows(rank, k)).clone()
        want = torch.cat([rows(r, k) for r in range(world)])
        bad.append(int((got.view(world, -1) != want.view(world, -1)).any(1).sum()))
    torch.cuda.synchronize()
    H.record(dict(test="tag_gather_late_rank", layers=layers, world=world, rank=rank, bad_ranks_per_call=bad,
                  mutation=os.environ.get("K31_MUTATION", "")))
    assert not any(bad), f"rank {rank}: calls with stale or wrong rank blocks: {bad}"


@pytest.mark.skipif(os.environ.get("K31_REQ_SHARD_GPU") != "1", reason="multi-GPU test: torchrun with K31_REQ_SHARD_GPU=1")
def test_tag_gather_compiles_before_first_call():
    """Reference = the protocol's requirement that no rank JIT-compiles inside a gather while its peers poll: after
    construction, the self-check plus a 3-layer forward (copies 0, 1, 2) compile no new variant of either kernel."""
    H.load_plugin(MUTATIONS)
    from sgl_nvidia_mega.kimi_k31.dsa import epoch_gather as E
    from sgl_nvidia_mega.kimi_k31.dsa import tag_gather as T

    dist, world, rank = _dist()
    n_own = 8 * 2048
    tg = T.TagGather(dist.group.WORLD, rank, world, n_own, E.copies_for(3), torch.device("cuda"), 2_000_000_000)
    n0 = (_variants(T._push_kernel), _variants(T._poll_kernel))
    tg.self_check(3)
    own = torch.arange(n_own, dtype=torch.int32, device="cuda") + rank
    for o in range(3):
        tg.gather(E.copy_for(o, 3), own)
    torch.cuda.synchronize()
    n1 = (_variants(T._push_kernel), _variants(T._poll_kernel))
    H.record(dict(test="tag_gather_compiles_once", world=world, rank=rank, variants_after_init=n0, variants_after_calls=n1,
                  mutation=os.environ.get("K31_MUTATION", "")))
    assert n0[0] >= 1 and n0[1] >= 1, f"rank {rank}: construction compiled nothing: {n0}"
    assert n1 == n0, f"rank {rank}: gather calls compiled new variants: {n0} -> {n1}"


def _core_inputs(rows, slots, g):
    """Sparse-core inputs with no NaN codes: query [rows, 6, 576] and a KV pool of ``slots`` rows (FP8 E4M3 codes)."""
    fp8 = lambda *s: (torch.randn(s, generator=g, device="cuda") * 2).to(torch.float8_e4m3fn).view(torch.uint8)  # noqa: E731
    return fp8(rows, 6, 576), fp8(slots * 576)


def _core_equal(core, ref, query, kv, tagged, rows_ref, lengths, scales, delay=0):
    """Reference = the served core (SparseMla6.run) on the decoded rows; the tagged core must match it bitwise. ``delay``:
    GPU sleep cycles before the tagged core (a late reader)."""
    rp, rs_ = (t.clone() for t in ref.run(query, kv, rows_ref, lengths, scales))
    H.poison(core.partial, core.stats)
    if delay:
        torch.cuda._sleep(delay)
    p, s = core.run_tagged(query, kv, tagged, lengths, scales)
    torch.cuda.synchronize()
    return bool(torch.equal(p, rp)) and bool(torch.equal(s, rs_))


@pytest.mark.skipif(os.environ.get("K31_REQ_SHARD_GPU") != "1", reason="multi-GPU test: torchrun with K31_REQ_SHARD_GPU=1")
@pytest.mark.parametrize("layers", [2, 3])
def test_fused_core_waits_for_late_rank(layers):
    """K31MK_DSA_REQ_SHARD_AG=fused at the core: TagGather.push + SparseMla6.run_tagged (the core polls the tagged words
    itself) vs the served core on the payload generator's rows (every rank computes every rank's rows; -1 among them), bitwise
    on partials and stats. Rank 1 sleeps ~25 ms before each push, so a core that does not wait for the tag reads rank 1's
    previous rows; 3 forwards of ``layers`` calls, copies by copy_for, counters continued from the self-check."""
    H.load_plugin(MUTATIONS)
    from sgl_nvidia_mega.kimi_k31.dsa import epoch_gather as E
    from sgl_nvidia_mega.kimi_k31.dsa import req_shard as RS_
    from sgl_nvidia_mega.kimi_k31.dsa import tag_gather as T
    from sgl_nvidia_mega.kimi_k31.dsa.sparse_mla6 import SparseMla6

    dist, world, rank = _dist()
    q, slots = 8, 1 << 16
    rows = world * q  # one request per rank
    copies = E.copies_for(layers)
    tg = T.TagGather(dist.group.WORLD, rank, world, q * 2048, copies, torch.device("cuda"), 2_000_000_000, fence=not FUSED_NOFENCE)
    tg.self_check(layers)
    delay = 100_000_000 if FUSED_LATE_READER and rank == 0 else 0  # ~2x rank 1's push delay: peers finish use k first
    core, ref = SparseMla6(torch.device("cuda"), rows), SparseMla6(torch.device("cuda"), rows)
    epochs = torch.zeros((copies, rows, core.splits), dtype=torch.int32, device="cuda")
    epochs[E.copy_for(layers - 1, layers)] = 1  # the self-check used the last layer's copy once
    g = H.generator(7100 + layers)
    query, kv = _core_inputs(rows, slots, g)
    lengths = torch.full((rows,), 4096, dtype=torch.int32, device="cuda")
    i = torch.arange(q * 2048, dtype=torch.int64, device="cuda")

    def own(r, k):
        v = (r * 7919 + k * 104729 + i * 2654435761) % slots
        return torch.where(i % 97 == 5, torch.full_like(v, -1), v).to(torch.int32)

    bad = []
    for k in range(3 * layers):
        c = E.copy_for(k % layers, layers)
        if rank == 1:
            torch.cuda._sleep(50_000_000)
        tg.push(c, own(rank, k))
        tagged = RS_.TaggedRows(tg.words(c).view(world * q, 2048), epochs[c], 2_000_000_000, q, world, fence=not FUSED_NOFENCE)
        want = torch.cat([own(r, k) for r in range(world)]).view(rows, 2048)
        bad.append(0 if _core_equal(core, ref, query, kv, tagged, want, lengths, (0.05, 1.0), delay) else 1)
    H.record(dict(test="fused_core_late_rank", layers=layers, world=world, rank=rank, bad_calls=bad, nofence=FUSED_NOFENCE,
                  late_reader=FUSED_LATE_READER, mutation=os.environ.get("K31_MUTATION", "")))
    assert not any(bad), f"rank {rank}: tagged core differs from the served core on calls {bad}"


@pytest.mark.skipif(os.environ.get("K31_REQ_SHARD_GPU") != "1", reason="multi-GPU test: torchrun with K31_REQ_SHARD_GPU=1")
@pytest.mark.parametrize("ctx,b", [(131072, 4), (32768, 5), (131072, 6)])
def test_fused_route_equals_full_selection(ctx, b):
    """K31MK_DSA_REQ_SHARD_AG=fused end to end on one DSA layer: (1) the slot words select() leaves in this rank's mailbox,
    decoded, equal the full selection (VerifySelection.run over all rows on replicated inputs) row by row as slot sets;
    (2) the core on the TaggedRows equals the served core on those decoded rows, bitwise. b 5 / 6 at world 4 pad the owner
    blocks with repeat rows, whose blocks no core row reads (the heartbeat poll covers them)."""
    import math

    import deep_gemm
    from sglang.kernels.ops.attention.dsv4.topk import plan_topk_v2
    from types import SimpleNamespace

    rs = _rs()
    from sgl_nvidia_mega.kimi_k31.dsa import tag_gather as T
    from sgl_nvidia_mega.kimi_k31.dsa import verify as V
    from sgl_nvidia_mega.kimi_k31.dsa.sparse_mla6 import SparseMla6
    import _ref_mla as R
    import test_k31_mla as M

    dist, world, rank = _dist()
    q = 8
    rows = q * b
    g = H.generator(6000 + ctx % 997 + 17 * b)
    npg = math.ceil(ctx / 64)
    pages = b * npg + 1
    cache = M.index_pool(pages, g)
    table = M.requests(b, ctx, pages, g)[0]
    query = R.fp8_bytes(torch.randn((rows, 32, 128), generator=g, device="cuda") * 100).view(R.FP8)
    gates = (torch.randn((rows, 32), generator=g, device="cuda") * 1e-4).contiguous().unsqueeze(-1)
    cq, kv = _core_inputs(rows, pages * 64, g)
    for t in (cache, query.view(torch.uint8), gates, table, cq, kv):
        dist.broadcast(t, 0)
    contexts = torch.full((b, q), ctx, dtype=torch.int32, device="cuda")
    causal = (ctx - q + 1 + torch.arange(q, device="cuda")).repeat(b).to(torch.int32)
    step = SimpleNamespace(batch_requests=b, query_width=q, score_contexts=contexts, causal_lengths=causal,
                           score_schedule=deep_gemm.get_paged_mqa_logits_metadata(contexts, 64, M.num_sms()),
                           topk_plan=plan_topk_v2(causal))
    full = V.VerifySelection(torch.device("cuda"), rows).run(query, gates, cache, table, step).clone()
    rs.TAG_NOFENCE = FUSED_NOFENCE  # the module reads K31MK_DSA_REQ_SHARD_TAG_NOFENCE at import; the state reads this global
    state = rs.ReqShardState(_TorchGroup(), world, rank, torch.device("cuda"), (rows,), q, ag="fused", layers=2)
    w = state.widths[rows]
    # the push top-k replaces the owner selection's x47 top-k only when it exists (979e3d4 tied that to TAGNF instead of TKPUSH)
    assert w.tkp is None or w.selection.x47 is w.tkp, f"rank {rank}: TKPUSH built but not installed as the owner top-k"
    assert (w.selection.x47 is None) == (V.VerifySelection(torch.device("cuda"), w.per * q).x47 is None) or w.tkp is not None, \
        f"rank {rank}: the owner selection lost its x47 top-k without TKPUSH"
    state.prepare(table, contexts, causal)
    tagged = state.select(0, query, gates, cache)
    assert tagged.fence == (not FUSED_NOFENCE), f"rank {rank}: TaggedRows.fence {tagged.fence} under nofence {FUSED_NOFENCE}"
    core, ref = SparseMla6(torch.device("cuda"), rows), SparseMla6(torch.device("cuda"), rows)
    H.poison(core.partial, core.stats)
    p, s = core.run_tagged(cq, kv, tagged, causal, (0.05, 1.0))
    p, s = p.clone(), s.clone()
    torch.cuda.synchronize()
    decoded = T.decode(tagged.words)[:rows]
    a = [frozenset(x for x in r.tolist() if x >= 0) for r in full.cpu()]
    d = [frozenset(x for x in r.tolist() if x >= 0) for r in decoded.cpu()]
    bad = [i for i in range(rows) if a[i] != d[i]]
    rp, rs_ = ref.run(cq, kv, decoded.contiguous(), causal, (0.05, 1.0))
    torch.cuda.synchronize()
    same = bool(torch.equal(p, rp)) and bool(torch.equal(s, rs_))
    H.record(dict(test="fused_route", ctx=ctx, b=b, world=world, rank=rank, rows=rows, rows_differ=len(bad), first_bad=bad[:4],
                  core_bitwise=same, nofence=FUSED_NOFENCE, mutation=os.environ.get("K31_MUTATION", "")))
    assert not bad, f"rank {rank}: {len(bad)} of {rows} decoded rows differ from the full selection (first {bad[:4]})"
    assert same, f"rank {rank}: the tagged core differs from the served core on the decoded rows"
