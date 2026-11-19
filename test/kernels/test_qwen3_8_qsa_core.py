"""Slot qsa_pool's tcgen05 core (runtime/attention.CoreKernels) on a
one-request verify (4 rows, BF16 pool) against a dense FP64 softmax and against the
split-K pool kernels; run on SM10x. Not bitwise (the source's core reorders the softmax
and splits P into BF16 hi/lo): relative L2 <= 0.01 and rtol 0.02 / atol 2e-4, as the
union qualification (record the measured errors). The core also stores the new K/V.

The request's 4 rows sit at positions P..P+3 of a 4 K context; each row selects 512
random blocks of the first 1024 and reads its own uncompressed tail (1, 2, 3, 0
tokens), the verify's own rows from the new K/V."""

import pytest
import torch

from sgl_nvidia_mega.qwen3_8.runtime.attention import CoreKernels, PoolKernels

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 10,
    reason="explicit SM10x qualification",
)

P, CONTEXT, SCALE = 4096, 4100, 0.0625


def _case(seed):
    g = torch.Generator(device="cuda").manual_seed(seed)
    rows = 4
    q = torch.randn(rows, 24, 256, generator=g, device="cuda").bfloat16() * 0.25
    pool = lambda: torch.randn(
        CONTEXT + 1, 2, 256, generator=g, device="cuda"
    ).bfloat16()
    k, v = pool() * 0.25, pool()
    k[0], v[0] = float("nan"), float("nan")  # slot 0: no row reads it
    table = torch.arange(1, CONTEXT + 1, device="cuda", dtype=torch.int32)[None]
    req = torch.zeros(rows, device="cuda", dtype=torch.int32)
    positions = P + torch.arange(rows, device="cuda", dtype=torch.int32)
    loc = table[0, positions.long()].long()
    new = torch.randn(2, rows, 2, 256, generator=g, device="cuda").bfloat16() * 0.25
    idx = torch.stack(
        [torch.randperm(1024, generator=g, device="cuda")[:512] for _ in range(rows)]
    ).to(torch.int32)
    return q, k, v, new[0], new[1], table, req, positions, loc, idx


def _args(q, k, v, nk, nv, table, req, positions, loc, idx):
    return dict(
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


def _reference(q, k, v, nk, nv, table, positions, loc, idx):
    k, v = k.clone(), v.clone()
    k[loc], v[loc] = nk, nv  # the verify's own rows
    out = []
    for row in range(q.shape[0]):
        p = int(positions[row])
        blocks = idx[row].long().repeat_interleave(4) * 4 + torch.arange(
            4, device="cuda"
        ).repeat(512)
        tail = torch.arange((p + 1) // 4 * 4, p + 1, device="cuda")
        slots = table[0, torch.cat((blocks, tail))].long()
        kk = k[slots].double().repeat_interleave(12, 1)
        vv = v[slots].double().repeat_interleave(12, 1)
        scores = torch.einsum("hd,nhd->hn", q[row].double(), kk) * SCALE
        out.append(torch.einsum("hn,nhd->hd", torch.softmax(scores, -1), vv))
    return torch.stack(out)


def _check(got, ref, record_property, name):
    got = got.view(4, 24, 256).double()
    assert torch.isfinite(got).all()
    difference = got - ref
    rel_l2 = (difference.norm() / ref.norm()).item()
    record_property(f"{name}_max_abs", difference.abs().max().item())
    record_property(f"{name}_rel_l2", rel_l2)
    assert rel_l2 <= 0.01, (name, rel_l2)
    torch.testing.assert_close(got, ref, rtol=0.02, atol=0.0002)


@pytest.mark.parametrize("seed", [1, 2])
def test_core_tolerance_and_store(seed, record_property):
    q, k, v, nk, nv, table, req, positions, loc, idx = _case(seed)
    ref = _reference(q, k, v, nk, nv, table, positions, loc, idx)
    kc, vc = k.clone(), v.clone()
    out = CoreKernels().run(**_args(q, kc, vc, nk, nv, table, req, positions, loc, idx))
    torch.cuda.synchronize()
    assert tuple(out.shape) == (4, 24, 256) and out.dtype == torch.bfloat16
    _check(out, ref, record_property, "core")
    # the new K/V stored at their slots, nothing else written
    want_k, want_v = k.clone(), v.clone()
    want_k[loc], want_v[loc] = nk, nv
    assert torch.equal(kc[1:], want_k[1:]) and torch.equal(vc[1:], want_v[1:])
    # the split-K pool kernels on the same step, for the record
    gate = torch.zeros(4, 24 * 256, device="cuda", dtype=torch.bfloat16)
    pool = PoolKernels().run(
        **_args(
            q.reshape(4, -1),
            k.clone(),
            v.clone(),
            nk,
            nv,
            table,
            req,
            positions,
            loc,
            idx,
        ),
        gate=gate,
    )
    torch.cuda.synchronize()
    _check(pool.float() * 2, ref, record_property, "pool")  # gate 0: sigmoid 1/2


@pytest.mark.parametrize("seed", [1, 2])
def test_gate_epilogue_gates_the_bf16_core_output(seed):
    """The gate merge: the core's BF16 output times sigmoid(gate) in FP32,
    within one BF16 rounding of torch's (the sigmoid is Triton's)."""
    q, k, v, nk, nv, table, req, positions, loc, idx = _case(seed)
    g = torch.Generator(device="cuda").manual_seed(seed + 10)
    gate = torch.randn(4, 24 * 256, generator=g, device="cuda").bfloat16()
    case = (q, nk, nv, table, req, positions, loc, idx)
    run = lambda **kw: CoreKernels().run(
        **_args(case[0], k.clone(), v.clone(), *case[1:]), **kw
    )
    plain, gated = run(), run(gate=gate)
    torch.cuda.synchronize()
    want = plain.float() * torch.sigmoid(gate.view(4, 24, 256).float())
    torch.testing.assert_close(gated, want.bfloat16())  # BF16 tolerances


def test_gate_epilogue_wait_keeps_the_values():
    """With the QSA PDL fix the gate merge waits for the partials in-kernel; the
    output is the same bits."""
    q, k, v, nk, nv, table, req, positions, loc, idx = _case(3)
    gate = torch.randn(4, 24 * 256, device="cuda").bfloat16()
    from sgl_nvidia_mega.qwen3_8.kernels.ops.attention import qsa_gate_epilogue_v2

    outs = []
    for wait in (False, True):
        args = _args(q, k.clone(), v.clone(), nk, nv, table, req, positions, loc, idx)
        outs.append(
            qsa_gate_epilogue_v2.run_gated(*args.values(), gate, wait_primary=wait)
        )
    torch.cuda.synchronize()
    assert torch.equal(outs[0], outs[1])
