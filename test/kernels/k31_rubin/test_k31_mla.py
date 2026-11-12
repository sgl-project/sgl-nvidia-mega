# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""K3.1 MK MLA + DSA decode path on SM107 (port check), at TP8 per-rank shapes with random weights:
IndexMlaQuery (folded index-K write) -> FlashInfer trtllm-gen dense MLA (12 heads) -> MlaValue; the DSA
indexer chain (IndexProjection, DeepGEMM fp8_paged_mqa_logits + topk_transform_paged_v2 / k31 rank_topk);
trtllm-gen sparse MLA (12 heads padded to 16); the whole sparse decode path. References: _ref_mla.py
(independent PyTorch of the stock method).

Tolerances (fixed before any run, never loosened):
- FP8 codes (MLA query, current KV row, index-K, index-Q): every cell within 1 E4M3 code step of the
  reference and at most 1% of cells differing (exact-match fraction reported). Index-K is checked against both
  the torch reference and the stock writer (flashinfer layernorm + sglang fused_store_index_k_cache); its FP32
  scales within 2^-20 relative. Every byte the kernel must not write stays identical (poisoned snapshot).
- MlaValue alone: |out - ref| <= 1e-3 + 2^-7 |ref| on every cell; rows with length 0 are zero.
- MLA outputs end to end (dense, sparse): atol 0.004, rtol 0.03 per cell (multireq README) against the stock
  method with the stock attention core: reference FP8 query/KV -> a direct FlashInfer trtllm-gen MLA call written
  in this test (12 heads dense; 16 heads zero-padded sparse) -> reference value. The trtllm-gen FP8 core quantizes
  P to E4M3 (an estimate: ~3% relative RMS per latent element), so against an FP32 attention the per-cell .004/.03
  gate is not meaningful at useful magnitudes; the core itself is gated against FP32 attention by relative norm:
  ||got - ref|| / ||ref|| <= 0.05 per (row, head) latent and per output row (the kprobe lane's criterion), with the
  per-cell FP32 stats recorded (not gated). Sparse core (SparseMla glue: pad 12->16, copy back) uses the same
  FP32 relative-norm gate per (row, head) plus NaN-poisoned pad heads.
- Index-Q head weights: within 2^-7 relative (one BF16 rounding of the gate).
- Top-k selection: physical slots page_table[p>>6]*64+(p&63), min(len, 2048) distinct valid entries (then -1);
  every element of the symmetric difference with the fp64 exact top-2048 within 2^-14 * max|ref| of the fp64
  2048th score (near ties); symmetric-difference size reported. Sparse end to end: >= 99% overlap with the fp64
  top-2048 computed from the reference index-Q/K.

Mutations (each must FAIL the suite): see MUTATIONS.
Timing (K31_TIME=1, base run only): cold-L2 CUDA-graph replay through distinct weight / cache copies.
"""

from __future__ import annotations

import math
import os
from types import SimpleNamespace

import pytest
import torch

import _harness as H
import _hook_variants as V
import _ref_mla as R

MUTATIONS = {
    "ln_mean": ("kimi_k31/folded_index.py", "values[i] -= mean", "values[i] -= mean * 0.5"),
    "idx_scale": (
        "kimi_k31/folded_index.py",
        "scale = max(maximum, Float32(1e-4)) / Float32(448)",
        "scale = max(maximum, Float32(1e-4)) / Float32(440)",
    ),
    "kv_norm": (
        "kimi_k31/k3/cutedsl/mla/query.py",
        "mean = fma(ss[0, 0] + ss[1, 0], Float32(1 / 512), eps)",
        "mean = fma(ss[0, 0] + ss[1, 0], Float32(1 / 256), eps)",
    ),
    "value_gate": (
        "kimi_k31/k3/cutedsl/mla/value.py",
        "* Float32(BFloat16(sigmoid(gates[r][e])))",
        "* Float32(1.0)",
    ),
    "iq_gate": (
        "kimi_k31/dsa/indexer.py",
        "return indexer_q_quant_head_gate(self.query, self.gate, 128**-0.5, SCALE_FMT)",
        "return indexer_q_quant_head_gate(self.query, self.gate, 64**-0.5, SCALE_FMT)",
    ),
    "sparse_heads": (
        "kimi_k31/dsa/mla.py",
        "query.view(torch.float8_e4m3fn).view(self.rows, 1, self.heads, 576)\n        )",
        "query.view(torch.float8_e4m3fn).view(self.rows, 1, self.heads, 576)\n        )\n"
        "        self.query[:, :, self.heads - 1].zero_()",
    ),
    "verify_causal": (
        "kimi_k31/dsa/verify.py",
        "scores, step.causal_lengths, self.row_table(table, step.query_width)",
        "scores, step.score_contexts.view(-1), self.row_table(table, step.query_width)",
    ),
    "verify_rowmap": (
        "kimi_k31/dsa/verify.py",
        "rows.view(b, q, pages).copy_(table.unsqueeze(1).expand(b, q, pages))",
        "rows.view(q, b, pages).copy_(table.unsqueeze(0).expand(q, b, pages))",
    ),
    "verify_iq_hidden": (
        "kimi_k31/dsa/verify.py",
        "return self.fused_index_q.run(self.hidden, projected)",
        "return self.fused_index_q.run(self.hidden.roll(1, 0), projected)",
    ),
    "fused_value_heads": (
        "kimi_k31/dsa/prepared.py",
        "to_cute(self.sparse_mla.output[:, 0, : self.heads]),",
        "to_cute(self.sparse_mla.output[:, 0, 1 : self.heads + 1]),",
    ),
    # Index-q FP8 scale (stock: act_quant(query, 128, "ue8m0")); the anchors are the ones test_k31_index_q.py uses.
    "iq_exact": V.edits("kimi_k31/dsa/index_q.cuh", 5),
    "iq_round": V.edits("kimi_k31/dsa/index_q.cuh", 3),
    "mk_chain_exact": ("kimi_k31/dsa/indexer.py", 'SCALE_FMT = "ue8m0"', "SCALE_FMT = None"),
}

HEADS = int(os.environ.get("K31_TEST_HEADS", "12"))  # 12: TP8, 6: TP16 per-rank shapes
GATE_END = 2112 + 128 * HEADS  # end of the output gate = first index-K column (TP8 3648, TP16 2880)
PITCH, PAGE = GATE_END + 128, 64  # TP8 3776, TP16 3008
MLA_ATOL, MLA_RTOL = 0.004, 0.03
TIMING = os.environ.get("K31_TIME") == "1" and not os.environ.get("K31_MUTATION")

needs_gpu = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 10,
    reason="needs an SM100-family GPU",
)


# ----------------------------------------------------------------------------------------------- builders
def make_weights(seed):
    g = H.generator(seed)
    rn = lambda *s: torch.randn(s, generator=g, device="cuda", dtype=torch.float32)  # noqa: E731
    bf = torch.bfloat16
    w_uv_base = (rn(HEADS, 128, 512) * (5.0 / math.sqrt(512))).to(bf)
    return SimpleNamespace(
        w_qa=(1 + 0.1 * rn(1536)).to(bf),
        w_kva=(1 + 0.1 * rn(512)).to(bf),
        w_qb=(rn(HEADS * 192, 1536) / math.sqrt(1536)).to(bf),
        w_uk=(rn(HEADS, 128, 512) / math.sqrt(128)).to(bf).contiguous(),
        w_uv=w_uv_base.transpose(1, 2),  # [12, 512, 128] stride (65536, 1, 512) like the absorbed kv_b view
        gamma=1 + 0.1 * rn(128),
        beta=0.1 * rn(128),
        k_eps=1e-6,
        wq_b=(rn(4096, 1536) / math.sqrt(1536)).to(bf),
        w_proj=(rn(32, 7168) / math.sqrt(7168)).to(bf),
    )


def fakes(w):
    one = lambda: torch.ones(1, dtype=torch.float32, device="cuda")  # noqa: E731
    attention = SimpleNamespace(
        q_a_layernorm=SimpleNamespace(weight=w.w_qa),
        kv_a_layernorm=SimpleNamespace(weight=w.w_kva),
        q_b_proj=SimpleNamespace(weight=w.w_qb),
        W_UK_T=w.w_uk,
        W_UV=w.w_uv,
        _q_scale_inv=one(),
        _k_scale_inv=one(),
        _q_scale_float=1.0,
        _k_scale_float=1.0,
        scale=192.0**-0.5,
        rms_norm_eps=1e-5,
    )
    indexer = SimpleNamespace(
        skip_rope=True,
        apply_hadamard=False,
        k_norm=SimpleNamespace(weight=w.gamma, bias=w.beta, variance_epsilon=w.k_eps),
        wq_b=SimpleNamespace(weight=w.wq_b),
        weights_proj=SimpleNamespace(weight=w.w_proj),
        num_init_tokens=0,
        num_local_tokens=0,
    )
    assert attention.W_UV.stride() == (65536, 1, 512)
    return attention, indexer


def prepared(w, rows, kv, index_cache, sparse=False, max_context=1 << 17):
    from sgl_nvidia_mega.kimi_k31.dsa.prepared import PreparedSparseIndexMla
    from sgl_nvidia_mega.kimi_k31.folded_index import PreparedIndexMla

    cls = PreparedSparseIndexMla if sparse else PreparedIndexMla
    scratch = torch.empty((max(rows), HEADS, 576), dtype=torch.uint8, device="cuda")
    obj = cls(fakes(w), {r: PITCH for r in rows}, scratch, max_context=max_context, core="trtllm", dcp=None, queries=(1,))
    obj.bind_index_cache(index_cache, PAGE)
    obj.prepare_cache(kv)
    return obj


def kv_pool(pages, g):
    return R.fp8_bytes(torch.randn((pages, PAGE, 576), generator=g, device="cuda"))


def index_pool(pages, g):
    raw = torch.empty((pages, PAGE * 132), dtype=torch.uint8, device="cuda")
    raw[:, : PAGE * 128] = R.fp8_bytes(torch.randn((pages, PAGE * 128), generator=g, device="cuda") * 112)
    scale = torch.rand((pages, PAGE), generator=g, device="cuda") * 0.02 + 0.005
    raw[:, PAGE * 128 :] = scale.view(torch.uint8)
    return raw


def requests(batch, length, pages, g, first_page=1):
    npg = math.ceil(length / PAGE)
    perm = torch.randperm(pages - first_page, generator=g, device="cuda")[: batch * npg] + first_page
    table = perm.view(batch, npg).to(torch.int32).contiguous()
    pos = length - 1
    slots = table[:, pos // PAGE].long() * PAGE + pos % PAGE
    lengths = torch.full((batch,), length, dtype=torch.int32, device="cuda")
    return table, slots, lengths


def activations(batch, g):
    projected = torch.randn((batch, PITCH), generator=g, device="cuda").to(torch.bfloat16)
    hidden = torch.randn((batch, 7168), generator=g, device="cuda").to(torch.bfloat16)
    return projected, hidden


def stock_index_writer(projected, w, cache, slots):
    from flashinfer.norm import layernorm
    from sglang.kernels.ops.kvcache.dsa_fused_store_index_cache import fused_store_index_k_cache

    key = layernorm(projected[:, GATE_END:PITCH].contiguous(), w.gamma, w.beta, w.k_eps)
    fused_store_index_k_cache(key, cache, slots, PAGE)


def index_row(cache, slots):
    page, off = slots // PAGE, slots % PAGE
    codes = torch.stack([cache[p, o * 128 : o * 128 + 128] for p, o in zip(page.tolist(), off.tolist())])
    scales = torch.stack(
        [cache[p, PAGE * 128 + o * 4 : PAGE * 128 + o * 4 + 4] for p, o in zip(page.tolist(), off.tolist())]
    )
    return codes, scales.contiguous().view(torch.float32).flatten()


def num_sms():
    return torch.cuda.get_device_properties(torch.cuda.current_device()).multi_processor_count


_FI = {}


def fi_mla(q_codes, kv, table, lengths, sparse=False):
    """Stock trtllm-gen MLA core called directly (FlashInfer), BF16 latent [B, 12, 512]."""
    from flashinfer.decode import trtllm_batch_decode_with_kv_cache_mla
    from flashinfer.utils import get_trtllm_gen_multi_ctas_kv_counter_bytes as counter_bytes

    B, heads = q_codes.shape[0], 16 if sparse else HEADS
    if heads not in _FI:
        _FI[heads] = (
            torch.zeros(128 << 20, dtype=torch.int8, device="cuda"),
            torch.zeros(counter_bytes(64, heads, num_sms()), dtype=torch.uint8, device="cuda"),
        )
    ws, counter = _FI[heads]
    q = torch.zeros((B, 1, heads, 576), dtype=torch.uint8, device="cuda")
    q[:, 0, :HEADS] = q_codes
    out = torch.empty((B, 1, heads, 512), dtype=torch.bfloat16, device="cuda")
    kw = dict(sparse_mla_top_k=R.TOPK) if sparse else {}
    trtllm_batch_decode_with_kv_cache_mla(
        query=q.view(R.FP8), kv_cache=kv.view(R.FP8).unsqueeze(1), workspace_buffer=ws, qk_nope_head_dim=128,
        kv_lora_rank=512, qk_rope_head_dim=64, block_tables=table.unsqueeze(1) if sparse else table,
        seq_lens=lengths, max_seq_len=1 << 17, out=out, bmm1_scale=192.0**-0.5, bmm2_scale=1.0,
        backend="trtllm-gen", multi_ctas_kv_counter_buffer=counter, **kw,
    )
    return out[:, 0, :HEADS]


def fp32_latent(q_codes, kv_rows_per_b):
    """FP32/fp64 attention per row: kv_rows_per_b[b] = FP8 codes [n, 576] of the attended keys."""
    return torch.stack(
        [R.attention(R.fp8_value(q_codes[b]), R.fp8_value(k), 192.0**-0.5) for b, k in enumerate(kv_rows_per_b)]
    )


def relnorm(name, got, ref, tol=0.05):
    """||got-ref||/||ref|| over the last dim, every vector <= tol; per-cell .004/.03 stats recorded only."""
    g, r = got.float(), ref.float()
    assert bool(torch.isfinite(g).all()), f"{name}: non-finite output"
    rel = (g - r).norm(dim=-1) / r.norm(dim=-1).clamp_min(1e-30)
    cell = ((g - r).abs() > MLA_ATOL + MLA_RTOL * r.abs()).float().mean().item()
    st = {"item": name, "max_relnorm": float(rel.max()), "tol": tol, "fp32_cell_viol_frac": cell,
          "max_abs": float((g - r).abs().max())}
    H.record({"kind": "relnorm", **st})
    print("RELNORM", st, flush=True)
    assert st["max_relnorm"] <= tol, f"{name}: relative-norm error {st}"
    return st


def dense_references(w, projected, kv_ref, table, lengths, q_codes):
    """(stock-core reference output, FP32-attention reference output, stock-core latent, FP32 latent)."""
    gate = projected[:, 2112:GATE_END]
    lat_fi = fi_mla(q_codes, kv_ref, table, lengths)
    rows = [R.gather_tokens(kv_ref, table[b], int(lengths[b]), 576) for b in range(projected.shape[0])]
    lat_32 = fp32_latent(q_codes, rows)
    return R.value(lat_fi, w.w_uv, gate, lengths), R.value(R.bf16(lat_32), w.w_uv, gate, lengths), lat_fi, lat_32


# ----------------------------------------------------------------------------------------------- 1 + 3
DENSE_CASES = [(r, 200) for r in (1, 2, 4, 8, 16, 64)] + [(r, c) for r in (1, 8) for c in (1000, 2048)]


@needs_gpu
@pytest.mark.parametrize("rows,ctx", DENSE_CASES, ids=[f"b{r}_ctx{c}" for r, c in DENSE_CASES])
def test_dense(rows, ctx):
    """IndexMlaQuery (query bytes, KV row, folded index-K) and the dense path end to end."""
    H.load_plugin(MUTATIONS)
    g = H.generator(100 + rows + ctx)
    w = make_weights(7)
    pages = rows * math.ceil(ctx / PAGE) + 2
    kv = kv_pool(pages, g)
    idx = torch.randint(0, 120, (pages, PAGE * 132), generator=g, device="cuda", dtype=torch.uint8)
    obj = prepared(w, (rows,), kv, idx)
    table, slots, lengths = requests(rows, ctx, pages, g)
    projected, _ = activations(rows, g)
    kv0, idx0 = kv.clone(), idx.clone()
    out = torch.empty((rows, HEADS * 128), dtype=torch.bfloat16, device="cuda")
    H.poison(out, obj.query_scratch)
    obj.run(projected, out, cache=kv, slots=slots, block_table=table, lengths=lengths)
    torch.cuda.synchronize()
    tag = f"b{rows}_ctx{ctx}"

    q_ref = R.mla_query(projected, w.w_qa, w.w_qb, w.w_uk, 1e-5)
    H.record({"kind": "check", **R.fp8_ulp_check(f"query.{tag}", obj.query_scratch[:rows], q_ref)})
    row_ref = R.kv_row(projected, w.w_kva, 1e-5)
    flat = kv.view(-1, 576)
    H.record({"kind": "check", **R.fp8_ulp_check(f"kv_row.{tag}", flat[slots], row_ref)})
    rest = flat.clone()
    rest[slots] = kv0.view(-1, 576)[slots]
    assert torch.equal(rest, kv0.view(-1, 576)), f"{tag}: KV bytes outside the current slots changed"

    codes, scales = index_row(idx, slots)
    k_ref, s_ref = R.index_k(projected, w.gamma, w.beta, w.k_eps)
    H.record({"kind": "check", **R.fp8_ulp_check(f"index_k.ref.{tag}", codes, k_ref)})
    H.check(f"index_k.scale.ref.{tag}", scales, s_ref, 0.0, 2.0**-20)
    stock = idx0.clone()
    stock_index_writer(projected, w, stock, slots)
    s_codes, s_scales = index_row(stock, slots)
    H.record({"kind": "check", **R.fp8_ulp_check(f"index_k.stock.{tag}", codes, s_codes)})
    H.check(f"index_k.scale.stock.{tag}", scales, s_scales, 0.0, 2.0**-20)
    H.record({"kind": "bitwise", "item": f"index_k.stock.{tag}", "equal": bool(torch.equal(idx, stock))})
    touched = torch.zeros_like(idx, dtype=torch.bool)
    for p, o in zip((slots // PAGE).tolist(), (slots % PAGE).tolist()):
        touched[p, o * 128 : o * 128 + 128] = True
        touched[p, PAGE * 128 + o * 4 : PAGE * 128 + o * 4 + 4] = True
    assert torch.equal(idx[~touched], idx0[~touched]), f"{tag}: index-K bytes outside the current slots changed"

    kv_ref = kv0.clone()
    kv_ref.view(-1, 576)[slots] = row_ref
    ref_fi, ref_32, lat_fi, lat_32 = dense_references(w, projected, kv_ref, table, lengths, q_ref)
    relnorm(f"trtllm_dense_core.{tag}", lat_fi, lat_32)
    relnorm(f"dense_out_vs_fp32.{tag}", out, ref_32)
    # Control, added after port1 (b8_ctx1000, b16, b64 failed the per-cell gate below): the stock core on the
    # kernel's OWN query bytes and cache isolates the MK core call + MlaValue from the query's legitimate
    # 1-code FP8 rounding flips; the attribution says whether every violating head block has a query flip.
    gate = projected[:, 2112:GATE_END]
    ctl = R.value(fi_mla(obj.query_scratch[:rows].clone(), kv, table, lengths), w.w_uv, gate, lengths)
    qdiff = (obj.query_scratch[:rows] != q_ref).any(-1)
    viol = (out.float() - ref_fi.float()).abs() > MLA_ATOL + MLA_RTOL * ref_fi.float().abs()
    viol = viol.view(rows, HEADS, 128).any(-1)
    H.record({"kind": "attribution", "item": f"dense_out.{tag}", "viol_blocks": int(viol.sum()),
              "viol_blocks_without_query_flip": int((viol & ~qdiff).sum()), "query_flip_blocks": int(qdiff.sum())})
    H.check(f"dense_out.control_kernel_query.{tag}", out, ctl, MLA_ATOL, MLA_RTOL)
    assert int((viol & ~qdiff).sum()) == 0, f"{tag}: violating head blocks without a query code flip"
    H.check(f"dense_out.{tag}", out, ref_fi, MLA_ATOL, MLA_RTOL)


# ----------------------------------------------------------------------------------------------- 2
@needs_gpu
@pytest.mark.parametrize("rows", (1, 8, 16, 64), ids=lambda r: f"b{r}")
def test_value(rows):
    """MlaValue alone (W_UV + sigmoid gate), including zeroed rows of empty requests."""
    from sgl_nvidia_mega.kimi_k31.k3.cutedsl.common.host import stream
    from sgl_nvidia_mega.kimi_k31.k3.tensor_adapter import to_cute

    H.load_plugin(MUTATIONS)
    g = H.generator(300 + rows)
    w = make_weights(9)
    kv = kv_pool(4, g)
    idx = torch.zeros((4, PAGE * 132), dtype=torch.uint8, device="cuda")
    obj = prepared(w, (rows,), kv, idx)
    mq, v, _, _ = obj.kernels[rows, 1]
    for kname, kern in (("mla_query", mq), ("mla_value", v)):  # SASS evidence: binary_version 107, regs, spills
        H.record({"kind": "resources", "item": f"{kname}.b{rows}", **H.function_resources(getattr(kern, "compiled", kern)), **H.device_facts()})
    uv = obj.views[4]
    projected, _ = activations(rows, g)
    for case in range(2):
        latent = (torch.randn((rows, HEADS, 512), generator=g, device="cuda") * 0.1).to(torch.bfloat16)
        lengths = torch.full((rows,), 100, dtype=torch.int32, device="cuda")
        lengths[case % rows] = 0 if case == 1 else 100
        out = torch.empty((rows, HEADS * 128), dtype=torch.bfloat16, device="cuda")
        H.poison(out)
        v(to_cute(latent), uv, to_cute(projected[:, 2112:GATE_END]), to_cute(lengths, 4), to_cute(out), stream(out.device))
        torch.cuda.synchronize()
        ref = R.value(latent, w.w_uv, projected[:, 2112:GATE_END], lengths)
        H.check(f"value.b{rows}.case{case}", out, ref, 1e-3, 2.0**-7)
        assert bool((out.float()[lengths == 0] == 0).all()), "rows of empty requests must be zero"


# ----------------------------------------------------------------------------------------------- 4
@needs_gpu
@pytest.mark.parametrize("rows", (1, 8), ids=lambda r: f"b{r}")
def test_sparse_core(rows):
    """trtllm-gen sparse MLA with HEADS (12, or 6 at TP16) padded to 16 over a given set of 2048 physical slots."""
    H.load_plugin(MUTATIONS)
    from sgl_nvidia_mega.kimi_k31.dsa.mla import SparseMla

    g = H.generator(400 + rows)
    pages = 4096
    kv = kv_pool(pages, g)
    sm = SparseMla(torch.device("cuda"), 1 << 17, rows, HEADS)
    query = R.fp8_bytes(torch.randn((rows, HEADS, 576), generator=g, device="cuda"))
    selected = torch.stack(
        [torch.randperm(pages * PAGE, generator=g, device="cuda")[: R.TOPK] for _ in range(rows)]
    ).to(torch.int32)
    lengths = torch.full((rows,), 65536, dtype=torch.int32, device="cuda")
    sm.query[:, :, HEADS:].view(torch.uint8).fill_(0x7F)  # NaN pad heads: they must never reach the output
    out = torch.empty((rows, HEADS, 512), dtype=torch.bfloat16, device="cuda")
    H.poison(out)
    sm.run(query, kv, selected, lengths, (192.0**-0.5, 1.0), out)
    torch.cuda.synchronize()
    flat = kv.view(-1, 576)
    ref = fp32_latent(query, [flat[selected[r].long()] for r in range(rows)])
    relnorm(f"sparse_core.b{rows}", out, ref)


# ----------------------------------------------------------------------------------------------- 5
@needs_gpu
@pytest.mark.parametrize("rows", (1, 8), ids=lambda r: f"b{r}")
def test_index_projection(rows):
    """IndexProjection.run: q_a RMSNorm, wq_b GEMM, act_quant per 128, BF16 head gate."""
    H.load_plugin(MUTATIONS)
    from sgl_nvidia_mega.kimi_k31.dsa.indexer import IndexProjection, IndexWeights

    g = H.generator(500 + rows)
    w = make_weights(11)
    proj = IndexProjection(IndexWeights(w.wq_b, w.w_proj, w.w_qa, 1e-5), rows)
    projected, hidden = activations(rows, g)
    H.poison(proj.query, proj.gate)
    q8, weights = proj.run(hidden, projected)
    torch.cuda.synchronize()
    codes_ref, w_ref = R.index_q(projected, hidden, w.w_qa, 1e-5, w.wq_b, w.w_proj)
    H.record({"kind": "check", **R.fp8_ulp_check(f"index_q.b{rows}", q8.reshape(rows, 32, 128), codes_ref)})
    H.check(f"index_w.b{rows}", weights.reshape(rows, 32), w_ref, 0.0, 2.0**-7)


def paged_index_case(length, rows_q, g, small):
    pages = math.ceil(length / PAGE) + 1
    cache = index_pool(pages, g)
    table, _, _ = requests(1, length, pages, g)
    q8 = R.fp8_bytes(torch.randn((rows_q, 32, 128), generator=g, device="cuda") * 100).view(R.FP8)
    weights = torch.randn((rows_q, 32), generator=g, device="cuda") * (1e-4 / (64 if small else 1))
    return pages, cache, table, q8, weights


SEL_CASES = [30000, 65536, 131072]


@needs_gpu
@pytest.mark.parametrize("ctx", SEL_CASES, ids=lambda c: f"ctx{c}")
def test_paged_selection(ctx):
    """Decode (1 row): DeepGEMM fp8_paged_mqa_logits + topk_transform_paged_v2(deterministic=False)."""
    H.load_plugin(MUTATIONS)
    import deep_gemm
    from sglang.kernels.ops.attention.dsv4.topk import plan_topk_v2
    from sgl_nvidia_mega.kimi_k31.dsa.indexer import PagedSelection

    g = H.generator(600 + ctx % 997)
    pages, cache, table, q8, weights = paged_index_case(ctx, 1, g, small=False)
    lengths = torch.full((1,), ctx, dtype=torch.int32, device="cuda")
    schedule = deep_gemm.get_paged_mqa_logits_metadata(lengths.view(-1, 1), PAGE, num_sms())
    plan = plan_topk_v2(lengths)
    sel = PagedSelection(torch.device("cuda"), 1)
    sel.selected.fill_(-7)
    out = sel.run(q8, weights.unsqueeze(-1), cache, table, lengths, schedule, plan)
    torch.cuda.synchronize()
    ref = R.index_scores(cache, table[0], q8[0].view(torch.uint8), weights[0], ctx)
    st = R.check_selection(f"paged_sel.ctx{ctx}", out[0], ref, ctx, R.slot_positions(table[0], pages))
    H.record({"kind": "selection", **st})
    print("SELECT", st, flush=True)


VER_CASES = [(12000, True), (12000, False), (32768, False), (131072, False)]


@needs_gpu
@pytest.mark.parametrize("ctx,small", VER_CASES, ids=[f"ctx{c}_{'small' if s else 'large'}" for c, s in VER_CASES])
def test_verify_selection(ctx, small):
    """MTP verify (1 request x 8 causal rows): DeepGEMM native [1,8] logits + stock topk_transform_paged_v2 over the
    shared request row (stride-0 per-row table)."""
    H.load_plugin(MUTATIONS)
    import deep_gemm
    from sglang.kernels.ops.attention.dsv4.topk import plan_topk_v2
    from sgl_nvidia_mega.kimi_k31.dsa.verify import VerifySelection

    g = H.generator(700 + ctx % 991 + small)
    pages, cache, table, q8, weights = paged_index_case(ctx, 8, g, small)
    contexts = torch.full((1, 8), ctx, dtype=torch.int32, device="cuda")
    causal = (ctx - 7 + torch.arange(8, device="cuda")).to(torch.int32)
    step = SimpleNamespace(
        batch_requests=1,
        query_width=8,
        score_contexts=contexts,
        score_schedule=deep_gemm.get_paged_mqa_logits_metadata(contexts, PAGE, num_sms()),
        causal_lengths=causal,
        topk_plan=plan_topk_v2(causal),
    )
    sel = VerifySelection(torch.device("cuda"), 8)
    sel.selected.fill_(-7)
    out = sel.run(q8, weights.unsqueeze(-1), cache, table, step)
    torch.cuda.synchronize()
    pos = R.slot_positions(table[0], pages)
    for j in range(8):
        n = int(causal[j])
        ref = R.index_scores(cache, table[0], q8[j].view(torch.uint8), weights[j], n)
        st = R.check_selection(f"verify_sel.ctx{ctx}.{'small' if small else 'large'}.row{j}", out[j], ref, n, pos)
        H.record({"kind": "selection", **st})
    print("SELECT", st, flush=True)


@needs_gpu
@pytest.mark.parametrize("breq", (2, 4, 8, 16), ids=lambda b: f"breq{b}")
def test_verify_selection_multi(breq, ctx=12000):
    """MTP verify (breq requests x 8 causal rows): per-row page table copied from the de-expanded request table
    (row r reads request r // 8); each row checked against its own request's fp64 top-k."""
    H.load_plugin(MUTATIONS)
    import deep_gemm
    from sglang.kernels.ops.attention.dsv4.topk import plan_topk_v2
    from sgl_nvidia_mega.kimi_k31.dsa.verify import VerifySelection

    g = H.generator(750 + breq)
    rows = breq * 8
    pages = breq * math.ceil(ctx / PAGE) + 1
    cache = index_pool(pages, g)
    table, _, _ = requests(breq, ctx, pages, g)
    q8 = R.fp8_bytes(torch.randn((rows, 32, 128), generator=g, device="cuda") * 100).view(R.FP8)
    weights = torch.randn((rows, 32), generator=g, device="cuda") * 1e-4
    contexts = torch.full((breq, 8), ctx, dtype=torch.int32, device="cuda")
    causal = (ctx - 7 + torch.arange(8, device="cuda")).repeat(breq).to(torch.int32)
    step = SimpleNamespace(
        batch_requests=breq,
        query_width=8,
        score_contexts=contexts,
        score_schedule=deep_gemm.get_paged_mqa_logits_metadata(contexts, PAGE, num_sms()),
        causal_lengths=causal,
        topk_plan=plan_topk_v2(causal),
    )
    sel = VerifySelection(torch.device("cuda"), rows)
    sel.selected.fill_(-7)
    out = sel.run(q8, weights.unsqueeze(-1), cache, table, step)
    torch.cuda.synchronize()
    for b in range(breq):
        pos = R.slot_positions(table[b], pages)
        for j in range(8):
            r, n = b * 8 + j, int(causal[b * 8 + j])
            ref = R.index_scores(cache, table[b], q8[r].view(torch.uint8), weights[r], n)
            st = R.check_selection(f"verify_sel_multi.breq{breq}.req{b}.row{j}", out[r], ref, n, pos)
            H.record({"kind": "selection", **st})
    print("SELECT", st, flush=True)


@needs_gpu
@pytest.mark.parametrize("path", ("fused", "stock"))
def test_verify_index_query(path, rows=8):
    """PreparedSparseVerifyMla.index_query at Breq 1 x Q8 = 8 rows: the fused index query (default) or the stock chain
    (K31MK_FUSED_INDEX_Q_VERIFY=0), each against the independent reference index query and against stock's index-q
    functions from the host tree with stock's arguments (act_quant(query, 128, "ue8m0"), bf16_head_gate_q_scale) on
    the same tensors (codes within 1 FP8 step in <= 1% of cells; head weights within 2^-7 relative). The hidden rows
    reach the kernel through prepare_gate, as in the verify path (SparseVerifyIndexKey stores them for the fused
    query)."""
    H.load_plugin(MUTATIONS)
    from sgl_nvidia_mega.kimi_k31.dsa import verify as V
    from sgl_nvidia_mega.kimi_k31.dsa.indexer import IndexWeights

    g = H.generator(900)
    w = make_weights(17)
    pages = 4
    base = prepared(w, (rows,), kv_pool(pages, g), index_pool(pages, g), sparse=False)
    default = V.FUSED_INDEX_Q_VERIFY
    V.FUSED_INDEX_Q_VERIFY = path == "fused"
    try:
        core = V.PreparedSparseVerifyMla(base, IndexWeights(w.wq_b, w.w_proj, w.w_qa, 1e-5), rows, 1 << 14)
    finally:
        V.FUSED_INDEX_Q_VERIFY = default
    assert (core.fused_index_q is not None) == (path == "fused"), "the case does not run the path it names"
    projected, hidden = activations(rows, g)
    core.step = SimpleNamespace(sparse=1)
    core.prepare_gate(hidden)
    q8, weights = core.index_query(projected)
    torch.cuda.synchronize()
    codes_ref, w_ref = R.index_q(projected, hidden, w.w_qa, 1e-5, w.wq_b, w.w_proj)
    scodes, sweights = stock_index_q(projected, hidden, w)
    codes, weights = q8.view(torch.uint8).reshape(rows, 32, 128), weights.reshape(rows, 32)
    H.record({"kind": "bitwise_vs_stock", "item": f"verify_index_q.{path}.b{rows}",
              "codes_exact": float((codes == scodes).float().mean()),
              "weights_exact": float((weights == sweights).float().mean())})
    H.record({"kind": "check", **R.fp8_ulp_check(f"verify_index_q.{path}.b{rows}", codes, codes_ref)})
    H.record({"kind": "check", **R.fp8_ulp_check(f"verify_index_q_vs_stock.{path}.b{rows}", codes, scodes)})
    H.check(f"verify_index_w.{path}.b{rows}", weights, w_ref, 0.0, 2.0**-7)
    H.check(f"verify_index_w_vs_stock.{path}.b{rows}", weights, sweights, 0.0, 2.0**-7)


@needs_gpu
@pytest.mark.parametrize("rows", (64, 128), ids=lambda r: f"b{r}")
def test_verify_index_query_batch(rows):
    """MK verify at Breq 8 / 16: above 8 rows the verify index query is the stock chain (IndexProjection;
    PreparedSparseVerifyMla builds the fused kernel only at <= 8 rows). Same inputs and gates as test_verify_index_query.
    128 rows (Breq 16) needs the trtllm-gen multi-CTA KV counter sized for 128 rows (compat.TRTLLM_KV_COUNTER_ROWS); with
    the old 64-row sizing FlashInfer refuses the call ("got 1536 bytes, need 3072")."""
    test_verify_index_query("stock", rows=rows)


def stock_index_q(projected, hidden, w):
    """Stock K3.1's index-q chain on these tensors (host tree): q_a RMSNorm, wq_b GEMM, act_quant(query, 128, "ue8m0")
    (dsa_indexer.py; deepseek_v2.py builds the Indexer with scale_fmt="ue8m0"), BF16 head gate GEMM and
    bf16_head_gate_q_scale(gate, q_scale, 128**-0.5). Returns (FP8 codes as uint8 [B,32,128], weights [B,32])."""
    from sgl_kernel import rmsnorm
    bf16_head_gate_q_scale = H.bf16_head_gate_q_scale  # the release host has no standalone head-gate function
    from sglang.kernels.ops.quantization.act_quant_triton import act_quant

    rows = projected.shape[0]
    query = torch.mm(rmsnorm(projected[:, :1536], w.w_qa, 1e-5), w.wq_b.T).view(rows, 32, 128)
    codes, scales = act_quant(query, 128, "ue8m0")
    weights = bf16_head_gate_q_scale(torch.mm(hidden, w.w_proj.T), scales, 128**-0.5)
    torch.cuda.synchronize()
    return codes.view(torch.uint8).reshape(rows, 32, 128), weights.reshape(rows, 32)


# ----------------------------------------------------------------------------------------------- 6
SPARSE_CASES = [(1, "fused"), (8, "fused"), (1, "stock"), (8, "stock")]


@needs_gpu
@pytest.mark.parametrize("rows,path", SPARSE_CASES, ids=[f"b{r}_{p}" for r, p in SPARSE_CASES])
def test_sparse_path(rows, path):
    """PreparedSparseIndexMla.run (64K context per row): query + folded index-K, index query (fused: the
    FusedIndexQuery kernel with the padded-query copy and MlaValue reading the core's 16-head output; stock: the
    IndexProjection chain, K31MK_FUSED_INDEX_Q=0), DeepGEMM + topk_v2, trtllm-gen sparse MLA, MlaValue."""
    H.load_plugin(MUTATIONS)
    import deep_gemm
    from sglang.kernels.ops.attention.dsv4.topk import plan_topk_v2
    from sgl_nvidia_mega.kimi_k31.dsa import prepared as P
    from sgl_nvidia_mega.kimi_k31.dsa_binding import IndexDecodeStep

    ctx = 65536
    g = H.generator(800 + rows)
    w = make_weights(13)
    pages = rows * math.ceil(ctx / PAGE) + 2
    kv = kv_pool(pages, g)
    idx = index_pool(pages, g)
    default = P.FUSED_INDEX_Q
    P.FUSED_INDEX_Q = path == "fused"
    try:
        obj = prepared(w, (rows,), kv, idx, sparse=True)
    finally:
        P.FUSED_INDEX_Q = default
    table, slots, lengths = requests(rows, ctx, pages, g)
    projected, hidden = activations(rows, g)
    schedule = deep_gemm.get_paged_mqa_logits_metadata(lengths.view(-1, 1), PAGE, num_sms())
    obj.set_step(1, IndexDecodeStep(schedule, plan_topk_v2(lengths)))
    kv0, idx0 = kv.clone(), idx.clone()
    out = torch.empty((rows, HEADS * 128), dtype=torch.bfloat16, device="cuda")
    H.poison(out, obj.query_scratch)
    obj.prepare_gate(hidden)
    # Prove which index-query implementation this case runs.
    want_kind = "FusedIndexQuery" if path == "fused" else "IndexProjection"
    assert type(obj.index_projection).__name__ == want_kind, type(obj.index_projection)
    obj.run(projected, out, cache=kv, slots=slots, block_table=table, lengths=lengths)
    torch.cuda.synchronize()
    selected = obj.selection.selected[:rows].clone()
    name = f"b{rows}_{path}_ctx{ctx}"

    q_ref = R.mla_query(projected, w.w_qa, w.w_qb, w.w_uk, 1e-5)
    kv_ref = kv0.clone()
    kv_ref.view(-1, 576)[slots] = R.kv_row(projected, w.w_kva, 1e-5)
    gate = projected[:, 2112:GATE_END]
    lat_fi = fi_mla(q_ref, kv_ref, selected, lengths, sparse=True)
    lat_32 = fp32_latent(q_ref, [kv_ref.view(-1, 576)[selected[b].long()] for b in range(rows)])
    # The sparse core's actual inputs: the MK's FP8 query codes (the core's 16-head buffer) and the KV rows it wrote.
    # Gated like every FP8-code check here (the b8 per-cell gate against a reference built from the
    # reference's own codes measured 0.2% query-code flips, not the MK's core path; the old comparison is recorded).
    mk_q = obj.sparse_mla.query[:, 0, :HEADS].view(torch.uint8).contiguous()
    H.record({"kind": "check", **R.fp8_ulp_check(f"sparse_query.{name}", mk_q, q_ref)})
    H.record({"kind": "check", **R.fp8_ulp_check(f"sparse_kv_row.{name}", kv.view(-1, 576)[slots],
                                                  kv_ref.view(-1, 576)[slots])})
    lat_mk = fi_mla(mk_q, kv, selected, lengths, sparse=True)
    lat_fi2 = fi_mla(q_ref, kv_ref, selected, lengths, sparse=True)
    ref_codes_out = R.value(lat_fi, w.w_uv, gate, lengths).float()
    viol = ((out.float() - ref_codes_out).abs() > MLA_ATOL + MLA_RTOL * ref_codes_out.abs()).sum().item()
    H.record({"kind": "record", "item": f"sparse_out_vs_reference_codes.{name}", "violations": int(viol),
              "cells": out.numel(), "core_rerun_max_abs": float((lat_fi.float() - lat_fi2.float()).abs().max())})
    relnorm(f"trtllm_sparse_core.{name}", lat_fi, lat_32)
    relnorm(f"sparse_out_vs_fp32.{name}", out, R.value(R.bf16(lat_32), w.w_uv, gate, lengths))
    H.check(f"sparse_out.{name}", out, R.value(lat_mk, w.w_uv, gate, lengths), MLA_ATOL, MLA_RTOL)

    idx_ref = idx0.clone()
    k_codes, k_scale = R.index_k(projected, w.gamma, w.beta, w.k_eps)
    for b in range(rows):
        p, o = int(slots[b] // PAGE), int(slots[b] % PAGE)
        idx_ref[p, o * 128 : o * 128 + 128] = k_codes[b]
        idx_ref[p, PAGE * 128 + o * 4 : PAGE * 128 + o * 4 + 4] = k_scale[b : b + 1].view(torch.uint8)
    q_codes, w_ref = R.index_q(projected, hidden, w.w_qa, 1e-5, w.wq_b, w.w_proj)
    for b in range(rows):
        scores = R.index_scores(idx_ref, table[b], q_codes[b], w_ref[b], ctx)
        want = torch.topk(scores, R.TOPK).indices
        got = R.slot_positions(table[b], pages)[selected[b].long()]
        assert bool((got >= 0).all()) and torch.unique(got).numel() == R.TOPK, f"{name}: invalid selection row {b}"
        overlap = torch.isin(got, want).float().mean().item()
        H.record({"kind": "check", "item": f"sparse_sel_overlap.{name}.row{b}", "overlap": overlap})
        print("OVERLAP", name, b, overlap, flush=True)
        assert overlap >= 0.99, f"{name} row {b}: selection overlaps the fp64 top-2048 by {overlap:.4f} < 0.99"


# ----------------------------------------------------------------------------------------------- timing
def _timed(name, calls, nbytes, **extra):
    t = H.cold_time_us(calls)
    fl = H.floor_us(nbytes)
    H.record({"kind": "time", "item": name, **t, **fl, **extra})
    print("TIME", name, t, fl, flush=True)
    return t


def _need_timing():
    if not TIMING:
        pytest.skip("timing runs only in the base run (K31_TIME=1, no mutation)")


@needs_gpu
@pytest.mark.parametrize("rows", (1, 8), ids=lambda r: f"b{r}")
def test_time_mla(rows):
    """MlaQuery alone, MlaValue alone, trtllm-gen dense core alone (ctx 2048), and the dense run."""
    _need_timing()
    from sgl_nvidia_mega.kimi_k31.k3.cutedsl.common.host import stream
    from sgl_nvidia_mega.kimi_k31.k3.tensor_adapter import to_cute

    H.load_plugin(MUTATIONS)
    g = H.generator(900 + rows)
    ctx = 2048
    per = HEADS * 192 * 1536 * 2 + HEADS * 128 * 512 * 2 * 2 + rows * ctx * 576
    n = H.copies_for(per, cap=48)
    pages = n * rows * (ctx // PAGE) + 2
    kv = kv_pool(pages, g)
    idx = torch.zeros((pages, PAGE * 132), dtype=torch.uint8, device="cuda")
    projected, _ = activations(rows, g)
    objs, tabs = [], []
    for i in range(n):
        objs.append(prepared(make_weights(1000 + i), (rows,), kv, idx))
        tabs.append(requests(rows, ctx, pages, g))
    out = torch.empty((rows, HEADS * 128), dtype=torch.bfloat16, device="cuda")

    def query_call(o, t):  # the stream is resolved at call time (the capture stream), as run() does
        mq = o.kernels[rows, 1][0]
        qnorm, kvnorm, qb, uk, uv, qinv, kinv = o.views
        args = (to_cute(projected[:, :2112]), qnorm, kvnorm, qb, uk, to_cute(kv.view(torch.uint8)), to_cute(t[1], 8))
        args += (qinv, kinv, o.eps, o.kernels[rows, 1][2])
        return lambda: mq(*args, *o.query_views, stream(projected.device))

    def value_call(o, t):
        v, latent = o.kernels[rows, 1][1], o.kernels[rows, 1][3]
        a = (latent, o.views[4], to_cute(projected[:, 2112:GATE_END]), to_cute(t[2], 4), to_cute(out))
        return lambda: v(*a, stream(projected.device))

    tag = f"b{rows}"
    _timed(f"mla_query.{tag}", [query_call(o, t) for o, t in zip(objs, tabs)], HEADS * 192 * 1536 * 2 + HEADS * 128 * 512 * 2)
    _timed(f"mla_value.{tag}", [value_call(o, t) for o, t in zip(objs, tabs)], HEADS * 512 * 128 * 2)
    _timed(
        f"trtllm_dense.{tag}.ctx{ctx}",
        [lambda o=o, t=t: o.stock_core(kv, t[0], t[2], rows) for o, t in zip(objs, tabs)],
        rows * ctx * 576,
    )
    _timed(
        f"dense_run.{tag}.ctx{ctx}",
        [
            lambda o=o, t=t: o.run(projected, out, cache=kv, slots=t[1], block_table=t[0], lengths=t[2])
            for o, t in zip(objs, tabs)
        ],
        per,
    )


@needs_gpu
@pytest.mark.parametrize("rows", (1, 8), ids=lambda r: f"b{r}")
def test_time_sparse_core(rows):
    _need_timing()
    H.load_plugin(MUTATIONS)
    from sgl_nvidia_mega.kimi_k31.dsa.mla import SparseMla

    g = H.generator(950 + rows)
    pages = 8192
    kv = kv_pool(pages, g)
    sm = SparseMla(torch.device("cuda"), 1 << 17, rows, HEADS)
    query = R.fp8_bytes(torch.randn((rows, HEADS, 576), generator=g, device="cuda"))
    lengths = torch.full((rows,), 65536, dtype=torch.int32, device="cuda")
    n = H.copies_for(rows * R.TOPK * 576, cap=128)
    sels = [
        torch.randint(0, pages * PAGE, (rows, R.TOPK), generator=g, device="cuda", dtype=torch.int32) for _ in range(n)
    ]
    out = torch.empty((rows, HEADS, 512), dtype=torch.bfloat16, device="cuda")
    sm.prepare_query(query)
    _timed(
        f"trtllm_sparse_core.b{rows}",
        [lambda sel=sel: sm.run_core(kv, sel, lengths, (192.0**-0.5, 1.0)) for sel in sels],
        rows * R.TOPK * 576,
    )
    _timed(
        f"trtllm_sparse_run.b{rows}",
        [lambda sel=sel: sm.run(query, kv, sel, lengths, (192.0**-0.5, 1.0), out) for sel in sels],
        rows * R.TOPK * 576,
    )


@needs_gpu
@pytest.mark.parametrize("rows", (1, 8), ids=lambda r: f"b{r}")
def test_time_index_projection(rows):
    _need_timing()
    H.load_plugin(MUTATIONS)
    from sgl_nvidia_mega.kimi_k31.dsa.indexer import IndexProjection, IndexWeights

    g = H.generator(970 + rows)
    nbytes = (4096 * 1536 + 32 * 7168) * 2
    n = H.copies_for(nbytes)
    projected, hidden = activations(rows, g)
    projs = []
    for i in range(n):
        w = make_weights(2000 + i)
        projs.append(IndexProjection(IndexWeights(w.wq_b, w.w_proj, w.w_qa, 1e-5), rows))
    _timed(f"index_q_chain.b{rows}", [lambda p=p: p.run(hidden, projected) for p in projs], nbytes)


DSA_CTX = (32768, 65536, 131072, 262144)


@needs_gpu
@pytest.mark.parametrize("ctx", DSA_CTX, ids=lambda c: f"ctx{c}")
@pytest.mark.parametrize("kind", ("decode", "verify"))
def test_time_dsa(kind, ctx):
    """Stock DSA selection baseline: DeepGEMM logits alone, top-k alone (scores L2-resident, as in the chain),
    and both back to back. decode = 1 row + topk_transform_paged_v2; verify = 1x8 native + topk_transform_paged_v2 over the
    per-row table (the default verify top-k)."""
    _need_timing()
    H.load_plugin(MUTATIONS)
    import deep_gemm
    from sglang.kernels.ops.attention.dsa.paged_mqa_logits import _restore_row_stride
    from sglang.kernels.ops.attention.dsv4.topk import plan_topk_v2, topk_transform_paged_v2

    rows = 1 if kind == "decode" else 8
    g = H.generator(990 + ctx % 977 + rows)
    npg = math.ceil(ctx / PAGE)
    n = H.copies_for(ctx * 132, cap=64)
    pages = n * npg + 1
    cache = index_pool(pages, g)
    tables = [requests(1, ctx, pages, g)[0] for _ in range(n)]
    q8 = R.fp8_bytes(torch.randn((rows, 32, 128), generator=g, device="cuda") * 100).view(R.FP8)
    weights = (torch.randn((rows, 32), generator=g, device="cuda") * 1e-4).contiguous()
    sel = torch.empty((rows, R.TOPK), dtype=torch.int32, device="cuda")
    raw = torch.empty_like(sel)  # PagedSelection runs the dual-output (page + raw) mode
    kvv = cache.view(-1, PAGE, 1, 132)
    if kind == "decode":
        lens = torch.full((1,), ctx, dtype=torch.int32, device="cuda")
        ctx2d = lens.view(-1, 1)
        q4 = q8.unsqueeze(1)
        topk_lens, plan = lens, plan_topk_v2(lens)
        top = lambda s, t: topk_transform_paged_v2(s, topk_lens, t, sel, PAGE, plan, raw, deterministic=False)  # noqa: E731
    else:
        ctx2d = torch.full((1, 8), ctx, dtype=torch.int32, device="cuda")
        q4 = q8.view(1, 8, 32, 128)
        topk_lens = (ctx - 7 + torch.arange(8, device="cuda")).to(torch.int32)
        plan = plan_topk_v2(topk_lens)
        top = lambda s, t: topk_transform_paged_v2(  # noqa: E731
            s, topk_lens, t.expand(8, -1), sel, PAGE, plan, raw, deterministic=False)
    schedule = deep_gemm.get_paged_mqa_logits_metadata(ctx2d, PAGE, num_sms())
    logits = lambda t: _restore_row_stride(  # noqa: E731
        deep_gemm.fp8_paged_mqa_logits(q4, kvv, weights, ctx2d, t, schedule, t.shape[1] * PAGE, clean_logits=False)
    )
    scores = [logits(t) for t in tables]
    torch.cuda.synchronize()
    tag = f"{kind}.r{rows}.ctx{ctx}"
    a = _timed(f"dg_logits.{tag}", [lambda t=t: logits(t) for t in tables], ctx * 132 + rows * ctx * 4)
    b = _timed(f"topk.{tag}", [lambda s=s, t=t: top(s, t) for s, t in zip(scores, tables)], rows * ctx * 4)
    c = _timed(f"logits+topk.{tag}", [lambda t=t: top(logits(t), t) for t in tables], ctx * 132 + 2 * rows * ctx * 4)
    row = {"kind": "dsa_baseline", "item": tag, "logits": a["us_median"], "topk": b["us_median"], "both": c["us_median"]}
    H.record(row)
