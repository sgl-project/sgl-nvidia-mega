"""Slot qsa_pool_fp8's union kernels (runtime/attention.UnionKernels) against a dense
FP64 softmax; run on SM10x before GSM8K. Not bitwise: the gate is relative L2 <= 0.01
and rtol 0.02 / atol 2e-4, a qualification ceiling (record the measured errors).

Each row selects every block of a 2048-token context with block 0 marked invalid (four
counted entries of score 0 and V 0) plus its own uncompressed tail, and pool slot 0
holds NaN that no row attends. The new K/V go in as BF16 and are cast to E4M3 in the
pool write; a second test checks that cast is torch's, byte for byte."""

from unittest import mock

import pytest
import torch

from sgl_nvidia_mega.qwen3_8.qsa import pool_decode, pool_union_ws
from sgl_nvidia_mega.qwen3_8.runtime.attention import UnionKernels

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 10,
    reason="explicit SM10x qualification",
)

LENGTH, SCALE = 2051, 0.0625


def _case(rows, seed):
    g = torch.Generator(device="cuda").manual_seed(seed)
    requests = rows // 4
    q = torch.randn(rows, 24, 256, generator=g, device="cuda").bfloat16() * 0.25
    pool = lambda: (
        torch.randn(requests * LENGTH + 1, 2, 256, generator=g, device="cuda") * 0.25
    ).to(torch.float8_e4m3fn)
    k, v = pool(), pool()
    k[0], v[0] = float("nan"), float("nan")
    table = torch.arange(1, requests * LENGTH + 1, device="cuda", dtype=torch.int32)
    table = table.reshape(requests, LENGTH)
    req = torch.arange(requests, device="cuda", dtype=torch.int32).repeat_interleave(4)
    positions = 2047 + torch.arange(rows, device="cuda", dtype=torch.int32) % 4
    loc = table[req.long(), positions.long()].long()
    idx = torch.arange(512, device="cuda", dtype=torch.int32).expand(rows, -1)
    idx = idx.contiguous()
    idx[:, 0] = -1
    return q, k, v, table, req, positions, loc, idx


def _run(q, k, v, nk, nv, table, req, positions, loc, idx):
    return UnionKernels().run(
        q=q,
        key_cache=k,
        value_cache=v,
        new_k=nk,
        new_v=nv,
        locations=loc,
        slot_map=table,
        row_requests=req,
        indices=idx,
        sequence_lengths=positions + 1,
        query_positions=positions,
        scale=SCALE,
    )


def _reference(q, k, v, table, req, positions, idx):
    out = []
    for row in range(q.shape[0]):
        p = int(positions[row])
        n = (p + 1) // 4 * 4
        blocks = idx[row].repeat_interleave(4)
        pos = (blocks * 4 + torch.arange(4, device="cuda").repeat(512))[:n]
        valid = blocks[:n] >= 0
        tail = torch.arange(n, p + 1, device="cuda")
        pos = torch.cat((pos, tail))
        valid = torch.cat((valid, torch.ones_like(tail, dtype=torch.bool)))
        slots = table[int(req[row]), pos.clamp_min(0).long()]
        kk, vv = k[slots].double(), v[slots].double()
        kk[~valid], vv[~valid] = 0, 0
        kk, vv = kk.repeat_interleave(12, 1), vv.repeat_interleave(12, 1)
        scores = torch.einsum("hd,nhd->hn", q[row].double(), kk) * SCALE
        probs = torch.softmax(scores, -1)
        out.append(torch.einsum("hn,nhd->hd", probs, vv))
    return torch.stack(out)


@pytest.mark.parametrize("rows", [36, 64, 288, 512, 576])
@pytest.mark.parametrize("ws", [True, False])  # False: the Triton kernel 2
def test_union_tolerance(rows, ws, record_property):
    q, k, v, table, req, positions, loc, idx = _case(rows, 942)
    assert pool_decode.write_new_kv_e4m3_ok(k, v)
    # the current step's K/V as BF16: E4M3 widens exactly, so the cast write keeps them
    nk, nv = k[loc].bfloat16(), v[loc].bfloat16()
    supported = pool_union_ws.supported if ws else (lambda *args: False)
    with mock.patch.object(pool_union_ws, "supported", supported):
        out, impl = _run(q, k, v, nk, nv, table, req, positions, loc, idx)
    torch.cuda.synchronize()
    assert impl == ("union_ws_fp8" if ws else "union_fp8")
    got = out.view(rows, 24, 256).double()
    ref = _reference(q, k, v, table, req, positions, idx)
    assert torch.isfinite(got).all()
    difference = got - ref
    rel_l2 = (difference.norm() / ref.norm()).item()
    record_property("max_abs", difference.abs().max().item())
    record_property("rel_l2", rel_l2)
    assert rel_l2 <= 0.01, rel_l2
    torch.testing.assert_close(got, ref, rtol=0.02, atol=0.0002)


def test_cast_write_is_the_torch_cast():
    rows = 64
    q, k, v, table, req, positions, loc, idx = _case(rows, 5)
    g = torch.Generator(device="cuda").manual_seed(6)
    new = torch.randn(rows, 2, 2, 256, generator=g, device="cuda").bfloat16() * 8
    results = []
    fp8 = new.to(k.dtype)
    for nk, nv in ((new[:, 0], new[:, 1]), (fp8[:, 0], fp8[:, 1])):
        kc, vc = k.clone(), v.clone()
        out, _ = _run(q, kc, vc, nk, nv, table, req, positions, loc, idx)
        results.append((kc, vc, out))
    torch.cuda.synchronize()
    for a, b in zip(*results):
        assert torch.equal(a.view(torch.uint8), b.view(torch.uint8))
