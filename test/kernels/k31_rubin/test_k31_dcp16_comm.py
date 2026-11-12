# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""K3.1 MK DCP exchanges (kimi_k31/dcp/comm.cuh) on ONE GPU with all N ranks emulated.

Every rank's receive regions live on this GPU; each exchange is launched as its push half for every rank, then its
poll half for every rank (``phases``), so the full N-rank protocol (tags, slot geometry, head blocks, scatter,
combine) runs with no cross-GPU dependency and no concurrency assumption. The same kernels run push+poll in one
launch in serving. Any SM100-family GPU works (B200/GB200/B300/GB300/VR200).

References (independent; tolerances fixed before any run):
- prologue: layout.local_write_slot (CPU unit-tested against brute force); epoch + 1 per call.
- q exchange: the concatenation of every rank's FP8 query bytes, bitwise; pad heads stay zero.
- candidates: torch scatter of every rank's (score, position) at global positions over a -inf row of the global
  length; bytes past the length untouched (sentinel 7.0); bitwise.
- merge, identity W_UV (projected partial = partial exactly): the stock dcp_push_combine arithmetic emulated
  step by step (weights torch.exp2 in FP32 on the GPU, FMA as one FP32 rounding of an exact FP64 product-sum,
  IEEE division, BF16 roundings, sigmoid 1/(1+exp(-g)) in FP32): bitwise on >= 99.99% of cells, every cell within
  1 BF16 ulp. Empty shards (count 0), NaN / +inf LSEs and padding rows (total 0, NaN gate) included.
- merge, random W_UV: projected partials from an FP64 GEMM rounded once to BF16, then the same emulation:
  >= 97% of cells bitwise, every cell within 2 BF16 ulps (FP32 MMA accumulation order differs from FP64).
- stale tags: a source whose push never happens must stop the reader at its deadline (subprocess traps), even
  when its slots hold well-formed records of the previous forward; a positive control completes.

Mutations (``K31_MUTATION=<name>`` through _harness / run_suite.sh; each must FAIL this file):
  stale_tag   stale tags accepted          -> test_stale_tag_rejected
  q_block     q head block off by one      -> test_q_exchange
  exp_base    exp instead of exp2          -> test_merge_combine_bitwise
  y_round     no BF16 rounding of y_s      -> test_merge_projection
  scatter_pos scatter position + 1         -> test_cand_exchange
  owner_rule  owner rule (w / 64) % N      -> test_prologue
  head_major  head-major push mapping      -> test_merge_combine_bitwise, test_merge_projection
Run: bash test/kernels/k31_rubin/run_suite.sh OUT test/kernels/k31_rubin/test_k31_dcp16_comm.py stale_tag q_block \
       exp_base y_round scatter_pos owner_rule head_major
"""

from __future__ import annotations

import math
import os
import subprocess
import sys
import textwrap

import pytest
import torch

import _harness as H
import _hook_variants as V

MUTATIONS = {
    name: V.edits("kimi_k31/dcp/comm.cuh", n)
    for n, name in enumerate(
        ("stale_tag", "q_block", "exp_base", "y_round", "scatter_pos", "owner_rule", "head_major"), start=1
    )
}
H.load_plugin(MUTATIONS)

from sgl_nvidia_mega.kimi_k31.dcp import layout as L  # noqa: E402
from sgl_nvidia_mega.kimi_k31.dcp.comm import POLL, PUSH, DcpComm  # noqa: E402

needs_gpu = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 10,
    reason="needs an SM100-family GPU",
)
DEV = "cuda"
BF16 = torch.bfloat16


def ranks(dcp, heads, cap, *, deadline_s=5.0):
    g = L.Geometry(dcp, heads, cap)
    regions = {
        L.KIND_Q: [torch.zeros(g.q_records(), dtype=torch.int64, device=DEV) for _ in range(dcp)],
        L.KIND_CAND: [torch.zeros(g.cand_records(), dtype=torch.int64, device=DEV) for _ in range(dcp)],
        L.KIND_PART: [torch.zeros(g.part_records(), dtype=torch.int64, device=DEV) for _ in range(dcp)],
    }
    epoch = torch.zeros(4, dtype=torch.int32, device=DEV)
    comms = [
        DcpComm(DEV, dcp, r, heads, cap, peers=regions, epoch=epoch, blocks=16,
                deadline_ns=int(deadline_s * 1e9))
        for r in range(dcp)
    ]
    return comms, regions, epoch


def gen(seed):
    g = torch.Generator(device=DEV)
    g.manual_seed(seed)
    return g


# ------------------------------------------------------------------------------------------------ prologue
@needs_gpu
@pytest.mark.parametrize("dcp", [8, 16])
def test_prologue(dcp):
    comms, _, epoch = ranks(dcp, 96 // dcp, 8)
    g = gen(1)
    widened = torch.randint(0, 1 << 20, (8,), generator=g, device=DEV, dtype=torch.int64)
    widened[3] = -1
    for r, comm in enumerate(comms):
        before = int(epoch[0])
        local = torch.full((8,), -7, dtype=torch.int64, device=DEV)
        comm.prologue(widened, local)
        torch.cuda.synchronize()
        assert int(epoch[0]) == before + 1
        expected = [L.local_write_slot(int(w), dcp, r) for w in widened.tolist()]
        assert local.tolist() == expected, (r, local.tolist(), expected)


# ------------------------------------------------------------------------------------------------ q exchange
@needs_gpu
@pytest.mark.parametrize("rows", [1, 8])
def test_q_exchange(rows):
    dcp, heads = 16, 6
    comms, _, epoch = ranks(dcp, heads, 8)
    g = gen(2 + rows)
    for round_, ordinal in ((0, 5), (1, 5), (2, 6)):  # reuse of one parity, then the other parity
        epoch[0] = 10 + round_
        qs = [torch.randint(0, 256, (rows, heads, 576), generator=g, device=DEV, dtype=torch.uint8) for _ in range(dcp)]
        glens = torch.full((rows,), 4000, dtype=torch.int32, device=DEV)
        outs = [torch.zeros((rows, 128, 576), dtype=torch.uint8, device=DEV) for _ in range(dcp)]
        for r in range(dcp):
            comms[r].qpush(ordinal, qs[r], glens, phases=PUSH)
        for r in range(dcp):
            comms[r].cand(ordinal, None, glens, None, qs[r], outs[r], phases=POLL)
        torch.cuda.synchronize()
        expected = torch.cat(qs, dim=1)
        for r in range(dcp):
            assert torch.equal(outs[r][:, : dcp * heads], expected), (round_, r)
            assert not outs[r][:, dcp * heads :].any(), "pad heads must stay zero"


# ------------------------------------------------------------------------------------------------ candidates
def local_candidates(glen, dcp, rank, rows, topk, g):
    """[rows, topk, 2] int32: min(local_len, topk) entries (random scores at distinct local positions), a few
    position -1 entries inside the prefix (stock skips them), (0, -1) after the prefix."""
    n = min(L.local_length(glen, dcp, rank), topk)
    out = torch.zeros((rows, topk, 2), dtype=torch.int32, device=DEV)
    out[..., 1] = -1
    for t in range(rows):
        local = torch.randperm(L.local_length(glen, dcp, rank), generator=g, device=DEV)[:n]
        scores = torch.randn(n, generator=g, device=DEV)
        out[t, :n, 0] = scores.view(torch.int32)
        out[t, :n, 1] = (local * dcp + rank).to(torch.int32)
        if n > 4:
            out[t, 1, 1] = -1
    return out


@needs_gpu
@pytest.mark.parametrize("rows,glen", [(1, 100), (1, 40000), (4, 5000), (8, 70000)])
def test_cand_exchange(rows, glen):
    dcp, heads, topk, ordinal = 16, 6, 2048, 9
    comms, _, epoch = ranks(dcp, heads, 8)
    epoch[0] = 3
    g = gen(rows * 7 + glen)
    span = L.PAGE * dcp
    width = -(-glen // span) * span + span
    glens = torch.full((rows,), glen, dtype=torch.int32, device=DEV)
    cands = [local_candidates(glen, dcp, r, rows, topk, g) for r in range(dcp)]
    dense = [torch.full((rows, width), 7.0, device=DEV) for _ in range(dcp)]
    q = [torch.zeros((rows, heads, 576), dtype=torch.uint8, device=DEV) for _ in range(dcp)]
    q_out = [torch.zeros((rows, 128, 576), dtype=torch.uint8, device=DEV) for _ in range(dcp)]
    for r in range(dcp):
        comms[r].qpush(ordinal, q[r], glens, dense=dense[r], phases=PUSH)  # q push + -inf fill
    for r in range(dcp):
        comms[r].cand(ordinal, cands[r], glens, dense[r], q[r], q_out[r], phases=PUSH)
    for r in range(dcp):
        comms[r].cand(ordinal, cands[r], glens, dense[r], q[r], q_out[r], phases=POLL)
    torch.cuda.synchronize()
    expected = torch.full((rows, width), 7.0, device=DEV)
    expected[:, :glen] = -math.inf
    for src, c in enumerate(cands):
        for t in range(rows):
            n = min(L.local_length(glen, dcp, src), topk)
            pos = c[t, :n, 1].long()
            keep = pos >= 0
            expected[t, pos[keep]] = c[t, :n, 0][keep].view(torch.float32)
    for r in range(dcp):
        assert torch.equal(dense[r].view(torch.int32), expected.view(torch.int32)), r


# ------------------------------------------------------------------------------------------------ stale tags
_STALE = textwrap.dedent(
    """
    import sys, torch
    sys.path.insert(0, {here!r})
    import test_k31_dcp16_comm as T
    from sgl_nvidia_mega.kimi_k31.dcp import layout as L
    comms, regions, epoch = T.ranks(16, 6, 8, deadline_s=0.3)
    rows, glen, ordinal = 1, 3000, 4
    g = T.gen(5)
    glens = torch.full((rows,), glen, dtype=torch.int32, device="cuda")
    cands = [T.local_candidates(glen, 16, r, rows, 2048, g) for r in range(16)]
    dense = torch.empty((rows, 4096), device="cuda")
    q = torch.zeros((rows, 6, 576), dtype=torch.uint8, device="cuda")
    q_out = torch.zeros((rows, 128, 576), dtype=torch.uint8, device="cuda")
    # The previous forward's records of source 1 in rank 0's region: well formed, stale tag.
    epoch[0] = 6
    geo = L.Geometry(16, 6, 8)
    stale = L.tag(5, ordinal, L.KIND_CAND)
    box = regions[L.KIND_CAND][0]
    for i in range(min(L.local_length(glen, 16, 1), 2048)):
        for half in range(2):
            box[geo.cand_index(ordinal % 2, 1, 0, i, half)] = (stale << 32) | 123
    for r in range(16):  # every source's q lands, so only source 1's candidates can be missing
        comms[r].qpush(ordinal, q, glens, phases=T.PUSH)
    for r in range(16):
        if r != 1 or {push_source_1}:
            comms[r].cand(ordinal, cands[r], glens, dense, q, q_out, phases=T.PUSH)
    comms[0].cand(ordinal, cands[0], glens, dense, q, q_out, phases=T.POLL)
    torch.cuda.synchronize()
    print("COMPLETED")
    """
)


@needs_gpu
def test_stale_tag_rejected():
    here = os.path.dirname(os.path.abspath(__file__))
    env = dict(os.environ)  # K31_MUTATION reaches the subprocess's load_plugin
    run = lambda push: subprocess.run(  # noqa: E731
        [sys.executable, "-c", _STALE.format(here=here, push_source_1=push)],
        env=env, capture_output=True, text=True, timeout=600,
    )
    control = run(True)
    assert control.returncode == 0 and "COMPLETED" in control.stdout, control.stderr[-2000:]
    stale = run(False)
    assert stale.returncode != 0 and "COMPLETED" not in stale.stdout, "a stale record was accepted"


# ------------------------------------------------------------------------------------------------ merge
def bf16_ulp(x):
    e = torch.floor(torch.log2(x.abs().clamp_min(1e-30)))
    return torch.exp2(e - 7)


def emulate_combine(ys, lses, counts, gate, totals):
    """Stock dcp_push_combine + gate for one owner: ys [N, rows, H, 128] FP32 (BF16 values), lses [N, rows, H],
    counts [N, rows], gate [rows, H*128] BF16, totals [rows] -> [rows, H*128] BF16. FP32 steps on the GPU."""
    n, rows, heads, _ = ys.shape
    empty = (counts <= 0)[:, :, None]  # [N, rows, 1]
    lse = torch.where(empty, torch.full_like(lses, -math.inf), lses)
    lse = torch.where(torch.isnan(lse) | (lse == math.inf), torch.full_like(lse, -math.inf), lse)
    ys = torch.where(empty[..., None], torch.zeros_like(ys), ys)
    top = lse.max(dim=0).values
    top = torch.where(top == -math.inf, torch.zeros_like(top), top)
    w = torch.exp2((lse - top).float())  # FP32 on the GPU: the same exp2f as the kernel
    acc = torch.zeros((rows, heads, 128), dtype=torch.float32, device=DEV)
    wsum = torch.zeros((rows, heads), dtype=torch.float32, device=DEV)
    for s in range(n):
        wsum = (wsum + w[s]).float()
        acc = (acc.double() + ys[s].double() * w[s, ..., None].double()).float()  # one rounding: FMA
    inv = (1.0 / torch.clamp(wsum, min=1.0).double()).float()
    x = (acc.double() * inv[..., None].double()).float().to(BF16).float()
    g = gate.view(rows, heads, 128).float()
    sig = (1.0 / (1.0 + torch.exp(-g))).to(BF16).float()
    out = (x.double() * sig.double()).float().to(BF16)
    out[totals <= 0] = 0
    return out.view(rows, heads * 128)


def run_merge(dcp, heads, rows, w_uv, parts, lses, counts, gates, totals, ordinal=3):
    comms, _, epoch = ranks(dcp, heads, 8)
    epoch[0] = 21
    outs = [torch.full((rows, heads * 128), float("nan"), dtype=BF16, device=DEV) for _ in range(dcp)]
    for phases in (PUSH, POLL):
        for r in range(dcp):
            comms[r].merge(ordinal, parts[r], lses[r], counts[r], totals, w_uv[r], gates[r], outs[r], phases=phases)
    torch.cuda.synchronize()
    return outs


def merge_inputs(dcp, heads, rows, seed, *, active=None):
    g = gen(seed)
    padded = 128
    parts = [
        (torch.randn((rows, padded, 512), generator=g, device=DEV) * torch.exp2(
            torch.randint(-3, 4, (rows, padded, 1), generator=g, device=DEV).float())).to(BF16)
        for _ in range(dcp)
    ]
    lses = [torch.randn((rows, padded), generator=g, device=DEV) * 4 for _ in range(dcp)]
    counts = [torch.randint(1, 200, (rows,), generator=g, device=DEV, dtype=torch.int32) for _ in range(dcp)]
    if active is not None:  # only these sources hold keys
        for r in range(dcp):
            if r not in active:
                counts[r].zero_()
    else:
        counts[2][0] = 0  # an empty shard
        lses[5][0, 7] = float("nan")
        lses[6][0, 9] = float("inf")
    gates = [torch.randn((rows, heads * 128), generator=g, device=DEV).to(BF16) for _ in range(dcp)]
    totals = torch.full((rows,), 5000, dtype=torch.int32, device=DEV)
    if rows > 1:
        totals[rows - 1] = 0  # a padding row with a NaN gate
        for gate in gates:
            gate[rows - 1] = float("nan")
    return parts, lses, counts, gates, totals


def owner_view(dcp, heads, r, rows, ys_of_source, lses, counts):
    """The owner r's sources: ys [N, rows, H, 128], lses [N, rows, H], counts [N, rows]."""
    hs = slice(r * heads, (r + 1) * heads)
    ys = torch.stack([ys_of_source(s)[:, hs] for s in range(dcp)])
    ls = torch.stack([lses[s][:, hs] for s in range(dcp)])
    cs = torch.stack(counts)
    return ys, ls, cs


@needs_gpu
@pytest.mark.parametrize("rows", [1, 8])
def test_merge_combine_bitwise(rows):
    dcp, heads = 16, 6
    parts, lses, counts, gates, totals = merge_inputs(dcp, heads, rows, 30 + rows)
    eye = torch.zeros((heads, 128, 512), device=DEV)
    eye[:, :, :128] = torch.eye(128, device=DEV)
    w_uv = [eye.to(BF16).transpose(1, 2) for _ in range(dcp)]  # [H, 512, 128] stride (65536, 1, 512)
    assert w_uv[0].stride() == (65536, 1, 512)
    outs = run_merge(dcp, heads, rows, w_uv, parts, lses, counts, gates, totals)
    for r in range(dcp):
        ys, ls, cs = owner_view(dcp, heads, r, rows, lambda s: parts[s][..., :128].float(), lses, counts)
        ref = emulate_combine(ys, ls, cs, gates[r], totals)
        got = outs[r]
        assert not torch.isnan(got.float()).any(), r
        equal = (got.view(torch.int16) == ref.view(torch.int16)).float().mean().item()
        err = ((got.float() - ref.float()).abs() / bf16_ulp(ref.float())).max().item()
        assert equal >= 0.9999 and err <= 1.0, (r, equal, err)


@needs_gpu
@pytest.mark.parametrize("rows,active", [(1, (0, 9)), (8, (3, 4)), (8, tuple(range(16)))])
def test_merge_projection(rows, active):
    dcp, heads = 16, 6
    parts, lses, counts, gates, totals = merge_inputs(dcp, heads, rows, 40 + rows, active=active)
    for r in range(dcp):  # equal LSEs: both sources weigh 1 (rounding of y_s matters most)
        lses[r].zero_()
    g = gen(99)
    w_uv = [(torch.randn((heads, 128, 512), generator=g, device=DEV) / math.sqrt(512)).to(BF16).transpose(1, 2)
            for _ in range(dcp)]
    outs = run_merge(dcp, heads, rows, w_uv, parts, lses, counts, gates, totals)
    for r in range(dcp):
        def projected(s, r=r):
            hs = slice(r * heads, (r + 1) * heads)
            y = torch.einsum("thk,hkn->thn", parts[s][:, hs].double(), w_uv[r].double())
            full = torch.zeros((rows, 128, 128), dtype=torch.float64, device=DEV)
            full[:, hs] = y
            return full.to(BF16).float()
        ys, ls, cs = owner_view(dcp, heads, r, rows, projected, lses, counts)
        ref = emulate_combine(ys, ls, cs, gates[r], totals)
        got = outs[r]
        live = totals > 0
        equal = (got[live].view(torch.int16) == ref[live].view(torch.int16)).float().mean().item()
        err = ((got[live].float() - ref[live].float()).abs() / bf16_ulp(ref[live].float())).max().item()
        assert equal >= 0.97 and err <= 2.0, (r, equal, err)
