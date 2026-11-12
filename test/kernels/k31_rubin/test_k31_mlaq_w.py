# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""K31MK_MLAQ_W: dsa/mlaq_w.py MlaQueryWide (and verify.py _enable_mlaq_w / _mla_query) against the cluster MlaQuery it replaces.

References, independent of the code under test: an FP64 exact query (rstd, Q_B and W_UK in FP64, no intermediate rounding, times
q_scale_inv), the shipped MlaQuery kernel on the same inputs (PreparedMla.kernels[rows, 8]), and the stock method in FP32 with its BF16
roundings written here from MlaQuery's docstring (validity line only).

Gate lines (fixed before any run):
  G0 validity: MlaQuery vs the FP32 stock-method reference: no NaN code, <= 3 E4M3 steps, <= 1 % of cells differ.
  G1 FP64 error no worse: rel RMS(dequant(wide) - q64) <= 1.002 x rel RMS(dequant(MlaQuery) - q64), per case.
  G2 query codes vs MlaQuery: no NaN code, every cell within 2 E4M3 steps, at most 0.5 % of cells differ.
  G3 KV rows: rows with slot >= 0 within 1 step of MlaQuery's and at most 0.1 % of their cells differ; every other byte of the KV pool
     (rows with slot < 0 and all other slots) unchanged.
  R  routing: profiles of >= K31MK_MLAQ_W_MIN_ROWS rows engage, narrower ones keep MlaQuery; non-plain bases are refused; the eps the
     wide query uses equals the PreparedMla's.
Mutations (each must FAIL this file): see MUTATIONS.
"""

from __future__ import annotations

import math
from types import SimpleNamespace

import pytest
import torch

import _harness as H

MUTATIONS = {
    "mqw_norm": ("kimi_k31/dsa/mlaq_w.py", "rr = tl.rsqrt(mean + EPS)", "rr = tl.rsqrt(mean * 0.0 + 1.0)"),
    "mqw_qsi": ("kimi_k31/dsa/mlaq_w.py", "v = acc.to(tl.bfloat16).to(tl.float32) * qsi", "v = acc.to(tl.bfloat16).to(tl.float32)"),
    "mqw_ukhead": ("kimi_k31/dsa/mlaq_w.py", "b = tl.load(WUK + h * (128 * 512) + kk", "b = tl.load(WUK + kk"),
    "mqw_rope": ("kimi_k31/dsa/mlaq_w.py", "h * 192 + 128 + ro", "h * 192 + ro"),
    "mqw_kveps": ("kimi_k31/dsa/mlaq_w.py", "* (1.0 / 512.0) + EPS)", "* (1.0 / 512.0) + EPS * 10000.0)"),
    "mqw_kpe": ("kimi_k31/dsa/mlaq_w.py", "tl.store(CACHE + base + 512 + po, pe * ksi)", "tl.store(CACHE + base + 512 + po, pe)"),
    "mqw_route": ("kimi_k31/dsa/verify.py", "        if rows < MLAQ_W_MIN_ROWS:\n            continue\n        b = profile.base",
                  "        if rows <= MLAQ_W_MIN_ROWS:\n            continue\n        b = profile.base"),
}

Q = 8
FP8 = torch.float8_e4m3fn
CASES = [  # (heads, rows, scales (q_inv, k_inv), eps, q_a input scale)
    (6, 72, (1.0, 1.0), 1e-5, 1.0),
    (6, 80, (0.75, 1.25), 1e-5, 1.0),
    (6, 96, (1.0, 1.0), 1e-5, 1.0),
    (6, 104, (0.75, 1.25), 1e-5, 1.0),
    (6, 128, (1.0, 1.0), 1e-5, 1.0),
    (6, 128, (0.75, 1.25), 0.25, 0.05),  # eps dominates the q_a / kv_a norms: catches a wrong eps
    (6, 40, (0.75, 1.25), 1e-5, 1.0),
    (12, 128, (0.75, 1.25), 1e-5, 1.0),
]

needs_gpu = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 10, reason="needs an SM100-family GPU"
)


def steps(a, b):
    def order(c):
        c = c.view(torch.uint8).to(torch.int32)
        m = c & 0x7F
        return torch.where(c >= 0x80, -m, m)

    return (order(a) - order(b)).abs()


def nan_codes(c):
    return int(((c.view(torch.uint8) & 0x7F) == 0x7F).sum())


def bf16(x):
    return x.to(torch.bfloat16).float()


def e4m3(x):
    return x.float().clamp(-448.0, 448.0).to(FP8).view(torch.uint8)


def weights(heads, seed, scales):
    g = H.generator(seed)
    rn = lambda *s: torch.randn(s, generator=g, device="cuda", dtype=torch.float32)  # noqa: E731
    bf = torch.bfloat16
    w = SimpleNamespace(
        w_qa=(1 + 0.1 * rn(1536)).to(bf),
        w_kva=(1 + 0.1 * rn(512)).to(bf),
        w_qb=(rn(heads * 192, 1536) / math.sqrt(1536)).to(bf),
        w_uk=(rn(heads, 128, 512) / math.sqrt(128)).to(bf).contiguous(),
        w_uv=(rn(heads, 128, 512) * (5.0 / math.sqrt(512))).to(bf).transpose(1, 2),
        q_inv=torch.tensor([scales[0]], dtype=torch.float32, device="cuda"),
        k_inv=torch.tensor([scales[1]], dtype=torch.float32, device="cuda"),
    )
    return w


def attention(w, eps):
    return SimpleNamespace(
        q_a_layernorm=SimpleNamespace(weight=w.w_qa),
        kv_a_layernorm=SimpleNamespace(weight=w.w_kva),
        q_b_proj=SimpleNamespace(weight=w.w_qb),
        W_UK_T=w.w_uk,
        W_UV=w.w_uv,
        _q_scale_inv=w.q_inv,
        _k_scale_inv=w.k_inv,
        _q_scale_float=1.0,
        _k_scale_float=1.0,
        scale=192.0**-0.5,
        rms_norm_eps=eps,
    )


def q64_ref(p, w, eps, heads):
    x = p[:, :1536].double()
    xn = x * torch.rsqrt((x * x).mean(-1, keepdim=True) + eps) * w.w_qa.double()[None, :]
    qf = (xn @ w.w_qb.double().t()).view(x.shape[0], heads, 192)
    q = torch.cat([torch.einsum("bhp,hpl->bhl", qf[..., :128], w.w_uk.double()), qf[..., 128:]], -1)
    return q * float(w.q_inv)


def stock_method(p, w, eps, heads):
    """MlaQuery's roundings in FP32 (torch order): bf16 xq, bf16 q, bf16 absorbed, e4m3 of value x q_scale_inv."""
    x = p[:, :1536].float()
    xq = bf16(x * torch.rsqrt((x * x).sum(-1, keepdim=True) / 1536 + eps) * w.w_qa.float()[None, :])
    q = bf16(xq @ w.w_qb.float().t()).view(x.shape[0], heads, 192)
    ab = bf16(torch.einsum("bhp,hpl->bhl", q[..., :128], w.w_uk.float()))
    return torch.cat([e4m3(ab * float(w.q_inv)), e4m3(q[..., 128:] * float(w.q_inv))], -1)


def build_case(heads, rows, scales, eps, xscale, seed):
    from sgl_nvidia_mega.kimi_k31.k3.cutedsl.mla.prepared import PreparedMla
    from sgl_nvidia_mega.kimi_k31.dsa.verify import qf_columns

    g = H.generator(seed + 1000)
    pitch = qf_columns(heads)[2]
    w = weights(heads, seed, scales)
    pages = rows + 4
    kv = torch.randint(0, 120, (pages, 64, 576), generator=g, device="cuda", dtype=torch.uint8)
    scratch = torch.empty((rows, heads, 576), dtype=torch.uint8, device="cuda")
    base = PreparedMla(attention(w, eps), {rows: pitch}, scratch, max_context=1 << 17, core="trtllm", queries=(Q,))
    base.prepare_cache(kv)
    p = torch.randn((rows, pitch), generator=g, device="cuda")
    p[:, :2048] *= xscale
    p = p.to(torch.bfloat16)
    perm = torch.randperm(pages * 64, generator=g, device="cuda")[:rows]
    slots = perm.to(torch.int64)
    slots[3] = -1
    slots[rows - 2] = -1
    return SimpleNamespace(w=w, base=base, kv=kv, p=p, slots=slots, rows=rows, heads=heads, eps=eps)


def run_stock(c, kv):
    from sgl_nvidia_mega.kimi_k31.k3.cutedsl.common.host import stream, to_cute

    b = c.base
    mq, _, query, _ = b.kernels[c.rows, Q]
    qnorm, kvnorm, qb, uk, uv, qinv, kinv = b.views
    args = (to_cute(c.p[:, :2112]), qnorm, kvnorm, qb, uk, to_cute(kv.view(torch.uint8)), to_cute(c.slots, 8), qinv, kinv, b.eps, query)
    H.poison(b.query_scratch)
    mq(*args, *b.query_views, stream(b.device))
    torch.cuda.synchronize()
    return b.query_scratch[: c.rows].clone()


def run_wide(c, kv):
    """Through verify.py's routing: _enable_mlaq_w on a one-profile stand-in core, then PreparedSparseVerifyMla._mla_query."""
    from sgl_nvidia_mega.kimi_k31.dsa import verify as V

    prof = SimpleNamespace(base=c.base, mlaq_w=None)
    V._enable_mlaq_w(SimpleNamespace(profiles={c.rows: prof}))
    assert prof.mlaq_w is not None, "MLAQ-W did not engage"
    H.poison(c.base.query_scratch)
    V.PreparedSparseVerifyMla._mla_query(prof, None, None, c.p, kv, c.slots, None)
    torch.cuda.synchronize()
    return c.base.query_scratch[: c.rows].clone(), prof.mlaq_w


@needs_gpu
@pytest.mark.parametrize("case", CASES, ids=[f"h{c[0]}_r{c[1]}_s{c[2][0]}_e{c[3]}" for c in CASES])
def test_wide_vs_mlaquery(monkeypatch, case):
    H.load_plugin(MUTATIONS)
    from sgl_nvidia_mega.kimi_k31.dsa import verify as V

    monkeypatch.setattr(V, "MLAQ_W_MIN_ROWS", 2)
    heads, rows, scales, eps, xscale = case
    c = build_case(heads, rows, scales, eps, xscale, seed=7 + rows + heads)
    kv_s, kv_w = c.kv.clone(), c.kv.clone()
    q_s = run_stock(c, kv_s)
    q_w, wide = run_wide(c, kv_w)
    tag = f"h{heads}_r{rows}_s{scales[0]}_e{eps}"
    assert abs(wide.eps - eps) <= eps * 2.0**-20, f"{tag}: wide eps {wide.eps} vs {eps}"

    ref = stock_method(c.p, c.w, eps, heads)
    d0 = steps(q_s, ref)
    g0 = {"nan": nan_codes(q_s), "max": int(d0.max()), "frac": float((d0 > 0).float().mean())}
    q64 = q64_ref(c.p, c.w, eps, heads)
    rms = lambda q: float(((q.view(FP8).double() - q64) ** 2).sum().sqrt() / (q64**2).sum().sqrt())  # noqa: E731
    g1 = {"wide": rms(q_w), "mlaq": rms(q_s)}
    d2 = steps(q_w, q_s)
    g2 = {"nan": nan_codes(q_w), "max": int(d2.max()), "frac": float((d2 > 0).float().mean())}
    valid = c.slots >= 0
    flat_s, flat_w = kv_s.view(-1, 576), kv_w.view(-1, 576)
    vs = c.slots[valid]
    d3 = steps(flat_w[vs], flat_s[vs])
    rest = torch.ones(flat_s.shape[0], dtype=torch.bool, device="cuda")
    rest[vs] = False
    g3 = {"max": int(d3.max()), "frac": float((d3 > 0).float().mean()), "rest_equal": bool(torch.equal(flat_w[rest], c.kv.view(-1, 576)[rest])),
          "stock_rest_equal": bool(torch.equal(flat_s[rest], c.kv.view(-1, 576)[rest]))}
    row = {"case": tag, "G0": g0, "G1": g1, "G2": g2, "G3": g3}
    H.record({"kind": "mlaq_w", **row})
    print("MLAQW_CASE", row, flush=True)
    assert g0["nan"] == 0 and g0["max"] <= 3 and g0["frac"] <= 0.01, f"{tag}: G0 validity (MlaQuery vs stock method) {g0}"
    assert g3["stock_rest_equal"], f"{tag}: G0 validity (MlaQuery wrote outside the valid slots)"
    assert g1["wide"] <= 1.002 * g1["mlaq"], f"{tag}: G1 FP64 error worse {g1}"
    assert g2["nan"] == 0 and g2["max"] <= 2 and g2["frac"] <= 0.005, f"{tag}: G2 query codes vs MlaQuery {g2}"
    assert g3["max"] <= 1 and g3["frac"] <= 0.001 and g3["rest_equal"], f"{tag}: G3 KV rows {g3}"


@needs_gpu
def test_route(monkeypatch):
    """Spec: a profile engages iff rows >= K31MK_MLAQ_W_MIN_ROWS and its base is the plain DCP1 MlaQuery; the eps is the base's."""
    H.load_plugin(MUTATIONS)
    import cutlass

    from sgl_nvidia_mega.kimi_k31.k3.cutedsl.mla.query import MlaQuery
    from sgl_nvidia_mega.kimi_k31.dsa import verify as V

    w = weights(6, 3, (1.0, 1.0))
    wt = (w.w_qa, w.w_kva, w.w_qb, w.w_uk, w.w_uv, w.q_inv, w.k_inv)
    base = SimpleNamespace(query_type=MlaQuery, dcp=None, query_buffers=(), weights=wt, eps=cutlass.Float32(1e-5), heads=6)
    for limit in (72, 96, 104):
        monkeypatch.setattr(V, "MLAQ_W_MIN_ROWS", limit)
        profiles = {r: SimpleNamespace(base=base, mlaq_w=None) for r in (40, 64, 72, 88, 96, 104, 112, 128)}
        V._enable_mlaq_w(SimpleNamespace(profiles=profiles))
        got = {r for r, p in profiles.items() if p.mlaq_w is not None}
        want = {r for r in profiles if r >= limit}
        assert got == want, f"MIN_ROWS {limit}: engaged {sorted(got)}, want {sorted(want)}"
        for r in got:
            assert abs(profiles[r].mlaq_w.eps - 1e-5) <= 1e-5 * 2.0**-20, profiles[r].mlaq_w.eps

    class Sub(MlaQuery):
        pass

    monkeypatch.setattr(V, "MLAQ_W_MIN_ROWS", 72)
    for bad in (dict(query_type=Sub), dict(dcp=object()), dict(query_buffers=(torch.empty(1),))):
        b2 = SimpleNamespace(**{**vars(base), **bad})
        with pytest.raises(ValueError):
            V._enable_mlaq_w(SimpleNamespace(profiles={128: SimpleNamespace(base=b2, mlaq_w=None)}))
