# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""L38 (K31MK_SPARSE_MLA6): the K3.1 TP16 sparse MLA core (kimi_k31/dsa/sparse_mla6.cuh: 6 heads per rank, no
padding, split-KV partials) and MlaValueFold (kimi_k31/k3/cutedsl/mla/value.py: the split merge folded into W_UV/gate).

Independent reference (_ref_mla.py, written from the method): per query row, fp64 softmax attention over the row's
valid selected keys (j < min(len, 2048) and slot >= 0), scale 192^-0.5 (bmm1), V = the first 512 key bytes, times
bmm2; the stock value (R.value) on its BF16 rounding.

Tolerances (fixed before any run, never loosened):
- Merged latent (the fold's BF16 latent view) vs the fp64 latent: every (row, head) vector ||got - ref|| / ||ref||
  <= 0.01, and every cell |got - ref| <= 2^-7 |ref| + 2^-8 max|ref of that vector| (BF16 output rounding is 2^-9
  relative; P is FP16 (2^-11), FP32 partials). Rows with no valid key: latent and output exactly 0.
- Accuracy evidence (gated): the mean over (row, head) of our relative L2 error vs fp64 <= the stock trtllm-gen core's
  (16-head padded, the call verify.py makes today) on the same inputs; both recorded per case.
- Output: bitwise equal to MlaValue (the shipped W_UV/gate kernel) run on the fold's own BF16 latent; and every row
  ||out - R.value(bf16(fp64 latent))|| / ||.|| <= 0.01.
- Coverage: partial / stats / latent / output poisoned (NaN) before every call; afterwards no NaN anywhere, partial /
  latent / output finite, and a split's max is -inf exactly when its sum is 0 (a split without valid keys).
- Verify layer (PreparedSparseVerifyMla, knob on vs off, Breq 1 and 2 x Q8): identical KV pools and MLA query in both
  arms (the index path is unchanged); selected slots compared as per-row sorted sets (stock top-k emission order is not
  deterministic, KNOWN K-G3-02a; set equality recorded); the knob arm never writes the stock core's padded query
  (NaN-poisoned, still NaN after: the arm ran the new path); each arm's output vs the fp64 layer reference built from
  that arm's own selected slots: knob arm every row relative L2 <= 0.01; stock arm's error recorded.
  (Test-construction fix after a GB300 run: v1 required finite stats and order-equal selections; both checks
  were wrong for every arm alike.)

Mutations (each must FAIL this suite through a base-passing test): MUTATIONS below.
"""

from __future__ import annotations

import math
import os
from types import SimpleNamespace

import pytest
import torch

import _harness as H
import _ref_mla as R

MUTATIONS = {
    # the length mask keeps one token fewer
    "smla6_len": (
        "kimi_k31/dsa/sparse_mla6.cuh",
        "const int len = min(ld_coherent(p.lengths + row), kTopK);",
        "const int len = min(ld_coherent(p.lengths + row), kTopK) - 1;",
    ),
    # softmax scale without log2(e)
    "smla6_log2e": ("kimi_k31/dsa/sparse_mla6.py", "float(scales[0]) * LOG2E", "float(scales[0])"),
    # FP8 V byte pairing for the FP16 A fragments (token / dim mix-up)
    "smla6_vpair": ("kimi_k31/dsa/sparse_mla6.cuh", '"prmt.b32 t, %2, 0, 0x3120;\\n"', '"prmt.b32 t, %2, 0, 0x3210;\\n"'),
    # the split max ignores warps 1-3 (P and the published max disagree)
    "smla6_max": (
        "kimi_k31/dsa/sparse_mla6.cuh",
        "  for (int w = 1; w < W; ++w) {\n    m0 = fmaxf(m0, s.max[w][2 * t]);",
        "  for (int w = W; w < W; ++w) {\n    m0 = fmaxf(m0, s.max[w][2 * t]);",
    ),
    # the published split sum drops warps 1-3
    "smla6_sum": ("kimi_k31/dsa/sparse_mla6.cuh", "      l += s.sum[w][tid];", "      l += 0.f * s.sum[w][tid];"),
    # split offset into the selected slots
    "smla6_split": (
        "kimi_k31/dsa/sparse_mla6.cuh",
        "p.selected + row * p.sel_stride + split * T;",
        "p.selected + row * p.sel_stride + (split / 2) * T;",
    ),
    # fold: unweighted denominator
    "fold_total": ("kimi_k31/k3/cutedsl/mla/value.py", "total = total + w * ls[c]", "total = total + ls[c]"),
    # fold: the last split is dropped
    "fold_last": (
        "kimi_k31/k3/cutedsl/mla/value.py",
        "        for c in cutlass.range_constexpr(S):\n            w = _ex2(ms[c] - top)",
        "        for c in cutlass.range_constexpr(S - 1):\n            w = _ex2(ms[c] - top)",
    ),
    # K31MK_FOLD_SPLIT_ROWS (MlaFoldMerge + MlaValue); each must fail test_fold_split or test_verify_layer_split
    # merge: latent stored to the neighbouring row
    "fsplit_row": (
        "kimi_k31/k3/cutedsl/mla/value.py",
        "store_words(at(latent_out, row, head, 4 * tid),",
        "store_words(at(latent_out, row ^ 1, head, 4 * tid),",
    ),
    # merge: the last split is dropped
    "fsplit_last": (
        "kimi_k31/k3/cutedsl/mla/value.py",
        "        for s in cutlass.range_constexpr(n_split):\n            wt = _ex2(maxes[s] - peak)",
        "        for s in cutlass.range_constexpr(n_split - 1):\n            wt = _ex2(maxes[s] - peak)",
    ),
    # merge: each split weighted by its neighbour's (m, l)
    "fsplit_stats": (
        "kimi_k31/k3/cutedsl/mla/value.py",
        "at(stats, row, s, head, 0)",
        "at(stats, row, (s + 1) % n_split, head, 0)",
    ),
    # merge: the output scale (bmm2) is dropped
    "fsplit_scale": ("kimi_k31/k3/cutedsl/mla/value.py", "div_rn(num[k], den) * out_scale", "div_rn(num[k], den)"),
    # wiring: the latent view compiled with a 6-head row pitch (the core's view has 16 heads)
    "fsplit_stride": (
        "kimi_k31/dsa/verify.py",
        "view = fake((rows, heads, 512), (16 * 512, 512, 1))",
        "view = fake((rows, heads, 512), (heads * 512, 512, 1))",
    ),
    # wiring: MlaValue built per row instead of per request (q = 1): request lengths read at the wrong index
    "fsplit_q": (
        "kimi_k31/dsa/verify.py",
        "_padded_value(rows, pitch, weight_layout, device, q, heads, early",
        "_padded_value(rows, pitch, weight_layout, device, 1, heads, early",
    ),
    # wiring, K31MK_MLAVALUE_FOLD_EARLY=f only: the fence flag lands in MlaValue's q slot (request lengths read per row)
    "fsplit_fence_q": (
        "kimi_k31/dsa/verify.py",
        "_padded_value(rows, pitch, weight_layout, device, q, heads, early == \"fence\")",
        "_padded_value(rows, pitch, weight_layout, device, 1 if early == \"fence\" else q, heads, early == \"fence\")",
    ),
    # wiring: the knob never selects the split form
    "fsplit_gate": (
        "kimi_k31/dsa/verify.py",
        "split = FOLD_SPLIT_ROWS > 0 and rows >= FOLD_SPLIT_ROWS",
        "split = FOLD_SPLIT_ROWS > 0 and rows < 0",
    ),
}

HEADS, TOPK, PAGE = 6, 2048, 64
BMM1 = 192.0**-0.5
needs_gpu = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 10,
    reason="needs an SM100-family GPU",
)


def kv_pool(pages, g):
    return R.fp8_bytes(torch.randn((pages, PAGE, 576), generator=g, device="cuda"))


def w_uv(g):
    base = (torch.randn((HEADS, 128, 512), generator=g, device="cuda") * (5.0 / math.sqrt(512))).to(torch.bfloat16)
    return base.transpose(1, 2)  # [6, 512, 128] stride (65536, 1, 512): the absorbed kv_b view


def case_inputs(rows, kind, seed, pages=4096):
    """(query codes [rows, 6, 576], kv pool, selected [rows, 2048] int32, lengths [rows] int32, valid counts)."""
    g = H.generator(seed)
    kv = kv_pool(pages, g)
    qscale = 4.0 if kind == "sharp" else 1.0
    query = R.fp8_bytes(torch.randn((rows, HEADS, 576), generator=g, device="cuda") * qscale)
    selected = torch.stack(
        [torch.randperm(pages * PAGE, generator=g, device="cuda")[:TOPK] for _ in range(rows)]
    ).to(torch.int32)
    if kind in ("full", "sharp"):
        lengths = (8192 + torch.arange(rows, device="cuda")).to(torch.int32)
    elif kind == "short":  # stock top-k contract: min(len, 2048) valid slots, then -1
        pattern = [1, 3, 100, 1000, 2047, 2048, 640, 1500]
        lengths = torch.tensor([pattern[r % len(pattern)] for r in range(rows)], dtype=torch.int32, device="cuda")
    elif kind == "empty_req":  # request 0 (rows 0-7) has length 0 (padded request): no valid slot, zero output
        lengths = torch.full((rows,), 9000, dtype=torch.int32, device="cuda")
        lengths[:8] = 0
    else:
        raise ValueError(kind)
    valid = torch.clamp(lengths, max=TOPK)
    for r in range(rows):
        selected[r, int(valid[r]) :] = -1
    return query, kv, selected, lengths, valid


def fp64_latent(query, kv, selected, valid, bmm2=1.0):
    flat = kv.view(-1, 576)
    out = torch.zeros((query.shape[0], HEADS, 512), dtype=torch.float32, device="cuda")
    for r in range(query.shape[0]):
        n = int(valid[r])
        if n > 0:
            keys = R.fp8_value(flat[selected[r, :n].long()])
            out[r] = R.attention(R.fp8_value(query[r]), keys, BMM1) * bmm2
    return out


def rel_err(got, ref):
    """Per (row, head) relative L2 error over the last dim (rows whose ref is 0 are skipped)."""
    g, r = got.float(), ref.float()
    norm = r.norm(dim=-1)
    keep = norm > 0
    return ((g - r).norm(dim=-1)[keep] / norm[keep]), keep


def stock_core(query, kv, selected, lengths):
    """The stock call verify.py makes today: trtllm-gen sparse MLA with the 6 heads padded to 16."""
    from flashinfer.decode import trtllm_batch_decode_with_kv_cache_mla
    from flashinfer.utils import get_trtllm_gen_multi_ctas_kv_counter_bytes as counter_bytes

    rows = query.shape[0]
    sms = torch.cuda.get_device_properties(torch.cuda.current_device()).multi_processor_count
    ws = torch.zeros(128 << 20, dtype=torch.int8, device="cuda")
    counter = torch.zeros(counter_bytes(64, 16, sms), dtype=torch.uint8, device="cuda")
    q = torch.zeros((rows, 1, 16, 576), dtype=torch.uint8, device="cuda")
    q[:, 0, :HEADS] = query
    # zeros: stock writes no row of a length-0 request (KNOWN K-G1-09: 24,576 uninitialized reads otherwise)
    out = torch.zeros((rows, 1, 16, 512), dtype=torch.bfloat16, device="cuda")
    trtllm_batch_decode_with_kv_cache_mla(
        query=q.view(R.FP8), kv_cache=kv.view(R.FP8).unsqueeze(1), workspace_buffer=ws, qk_nope_head_dim=128,
        kv_lora_rank=512, qk_rope_head_dim=64, block_tables=selected.unsqueeze(1), seq_lens=lengths,
        max_seq_len=1 << 17, sparse_mla_top_k=TOPK, out=out, bmm1_scale=BMM1, bmm2_scale=1.0, enable_pdl=True,
        backend="trtllm-gen", multi_ctas_kv_counter_buffer=counter,
    )
    return out[:, 0, :HEADS]


def run_mla6(query, kv, selected, lengths, gate, uv, req_lengths, q, tokens, bmm2=1.0):
    """SparseMla6 + MlaValueFold as verify.py launches them; returns (latent [rows,6,512] BF16, out, core)."""
    import cutlass
    from sgl_nvidia_mega.kimi_k31.k3.cutedsl.common.host import layout, stream, to_cute
    from sgl_nvidia_mega.kimi_k31.dsa import verify as V
    from sgl_nvidia_mega.kimi_k31.dsa.sparse_mla6 import SparseMla6

    rows = query.shape[0]
    core = SparseMla6(torch.device("cuda"), rows, HEADS, tokens)
    pitch = gate.stride(0)
    fold = V._fold_value(rows, pitch, layout(uv), torch.cuda.current_device(), q, HEADS, core.splits)
    latent_buf = torch.empty((rows, 1, 16, 512), dtype=torch.bfloat16, device="cuda")
    out = torch.empty((rows, HEADS * 128), dtype=torch.bfloat16, device="cuda")
    H.poison(core.partial, core.stats, latent_buf, out)
    partial, stats = core.run(query, kv, selected, lengths, (BMM1, bmm2))
    fold(to_cute(partial, 16), to_cute(stats, 8), to_cute(uv), to_cute(gate), to_cute(req_lengths, 4),
         cutlass.Float32(bmm2), to_cute(out), to_cute(latent_buf[:, 0, :HEADS]), stream(torch.device("cuda")))
    torch.cuda.synchronize()
    for name, t in (("partial", core.partial), ("out", out)):
        assert bool(torch.isfinite(t.float()).all()), f"{name}: non-finite (unwritten) cells"
    m, l = core.stats[..., 0], core.stats[..., 1]
    assert not bool(torch.isnan(core.stats).any()), "stats: NaN (unwritten) cells"
    assert bool(torch.isfinite(l).all()) and bool(((m == -math.inf) == (l == 0)).all()), "stats: max -inf iff sum 0"
    latent = latent_buf[:, 0, :HEADS]
    assert bool(torch.isfinite(latent.float()).all()), "latent: non-finite (unwritten) cells"
    return latent, out, core


def stock_value(latent, uv, gate, req_lengths, q):
    """The shipped MlaValue kernel on a given BF16 latent [rows, 6, 512] (contiguous copy, padded view)."""
    from sgl_nvidia_mega.kimi_k31.k3.cutedsl.common.host import layout, stream, to_cute
    from sgl_nvidia_mega.kimi_k31.dsa import verify as V

    rows = latent.shape[0]
    padded = torch.zeros((rows, 16, 512), dtype=torch.bfloat16, device="cuda")
    padded[:, :HEADS] = latent
    value = V._padded_value(rows, gate.stride(0), layout(uv), torch.cuda.current_device(), q, HEADS)
    out = torch.empty((rows, HEADS * 128), dtype=torch.bfloat16, device="cuda")
    H.poison(out)
    value(to_cute(padded[:, :HEADS]), to_cute(uv), to_cute(gate), to_cute(req_lengths, 4), to_cute(out),
          stream(torch.device("cuda")))
    torch.cuda.synchronize()
    return out


KERNEL_CASES = [
    (8, "full", 128), (16, "full", 128), (8, "sharp", 128), (8, "short", 128), (16, "empty_req", 128),
    (8, "full", 64), (8, "short", 64), (8, "full", 256), (16, "short", 256), (32, "full", 128),
    (8, "full", 80), (8, "short", 80), (16, "full", 96), (8, "sharp", 96),
]


@needs_gpu
@pytest.mark.parametrize("rows,kind,tokens", KERNEL_CASES, ids=[f"r{r}_{k}_t{t}" for r, k, t in KERNEL_CASES])
def test_core_fold(rows, kind, tokens):
    H.load_plugin(MUTATIONS)
    q = 8
    seed = 3800 + 17 * rows + {"full": 0, "sharp": 1, "short": 2, "empty_req": 3}[kind] + tokens
    query, kv, selected, lengths, valid = case_inputs(rows, kind, seed)
    g = H.generator(seed + 1)
    uv = w_uv(g)
    pitch = 2112 + 128 * HEADS + 128
    projected = torch.randn((rows, pitch), generator=g, device="cuda").to(torch.bfloat16)
    gate = projected[:, 2112 : 2112 + 128 * HEADS]
    req_lengths = lengths.view(-1, q)[:, 0].contiguous() if kind == "empty_req" else torch.full(
        (rows // q,), 9000, dtype=torch.int32, device="cuda")
    latent, out, _ = run_mla6(query, kv, selected, lengths, gate, uv, req_lengths, q, tokens)
    ref = fp64_latent(query, kv, selected, valid)
    name = f"r{rows}_{kind}_t{tokens}"

    empty = valid == 0
    if bool(empty.any()):
        assert bool((latent[empty].float() == 0).all()), f"{name}: rows without keys must give a zero latent"
        assert bool((out[empty].float() == 0).all()), f"{name}: rows without keys must give a zero output"
    err, keep = rel_err(latent, ref)
    H.record({"kind": "check", "item": f"l38_latent_relerr.{name}", "max": float(err.max()),
              "mean": float(err.mean()), "vectors": int(keep.sum())})
    print("L38_LATENT", name, float(err.max()), float(err.mean()), flush=True)
    assert float(err.max()) <= 0.01, f"{name}: latent relative L2 error {float(err.max()):.4g} > 0.01"
    vmax = ref.abs().amax(-1, keepdim=True)
    cell = (latent.float() - ref).abs() <= 2.0**-7 * ref.abs() + 2.0**-8 * vmax
    assert bool(cell.all()), f"{name}: {int((~cell).sum())} latent cells beyond 2^-7|ref| + 2^-8 max|ref|"

    # L1 evidence: the stock core on the same inputs.
    stock = stock_core(query, kv, selected, lengths)
    torch.cuda.synchronize()
    serr, _ = rel_err(stock, ref)
    H.record({"kind": "g3_l1", "item": f"l38_vs_stock_err.{name}", "ours_mean": float(err.mean()),
              "ours_max": float(err.max()), "stock_mean": float(serr.mean()), "stock_max": float(serr.max())})
    print("L38_G3L1", name, float(err.mean()), float(serr.mean()), flush=True)
    assert float(err.mean()) <= float(serr.mean()), f"{name}: mean error {float(err.mean()):.4g} > stock's"

    # Output: the fold's W_UV/gate equals the shipped MlaValue on the same latent, bitwise; and vs the reference.
    same = stock_value(latent, uv, gate, req_lengths, q)
    assert torch.equal(out.view(torch.int16), same.view(torch.int16)), f"{name}: output != MlaValue(latent)"
    ref_out = R.value(R.bf16(ref), uv, gate, req_lengths, q)
    oerr = (out.float() - ref_out).norm(dim=-1) / ref_out.norm(dim=-1).clamp_min(1e-30)
    live = ref_out.norm(dim=-1) > 0
    H.record({"kind": "check", "item": f"l38_out_relerr.{name}", "max": float(oerr[live].max()) if live.any() else 0})
    assert not bool(live.any()) or float(oerr[live].max()) <= 0.01, f"{name}: output relative L2 error > 0.01"


# K31MK_FOLD_SPLIT_ROWS: rows 8 = c=1, 64 / 128 = Breq 8 / 16; bmm2 0.5 exercises out_scale.
SPLIT_CASES = [
    (8, "full", 1.0), (16, "short", 1.0), (32, "empty_req", 1.0), (64, "full", 0.5), (64, "short", 1.0),
    (64, "empty_req", 0.5), (128, "sharp", 1.0), (128, "full", 0.5),
]


@needs_gpu
@pytest.mark.parametrize("rows,kind,bmm2", SPLIT_CASES, ids=[f"r{r}_{k}_s{b}" for r, k, b in SPLIT_CASES])
def test_fold_split(rows, kind, bmm2):
    """MlaFoldMerge + MlaValue (dsa/verify.py _split_fold_value) on the served core's own partials.

    Tolerances fixed before any run: latent and output BITWISE equal to the served MlaValueFold on the same partials
    (the split form changes no arithmetic); the core's padded heads 6-15 of the latent view stay unwritten (NaN);
    and, independently of the served kernel, test_core_fold's fp64 lines: per (row, head) latent relative L2
    <= 0.01, every cell |got - ref| <= 2^-7 |ref| + 2^-8 max|ref|, rows without keys exactly 0, every output row
    relative L2 <= 0.01 vs R.value(bf16(fp64 latent))."""
    _fold_split_case(rows, kind, bmm2)


# K31MK_MLAVALUE_FOLD_EARLY=f with the split form: MlaValue carries a fence before its PDL trigger. The fence changes no
# arithmetic, so the lines are test_fold_split's; empty_req makes a wrongly wired MlaValue (request lengths per row) show.
FENCE_CASES = [(64, "empty_req", 0.5), (128, "full", 1.0)]


@needs_gpu
@pytest.mark.parametrize("rows,kind,bmm2", FENCE_CASES, ids=[f"r{r}_{k}_s{b}" for r, k, b in FENCE_CASES])
def test_fold_split_fence(rows, kind, bmm2):
    """_split_fold_value(early="fence"): bitwise equal to the served MlaValueFold and within test_fold_split's fp64
    lines. Whether the fence is in the PTX, and before the trigger, is not visible in outputs; it is checked on the
    compiled PTX instead."""
    _fold_split_case(rows, kind, bmm2, early="fence")


def _fold_split_case(rows, kind, bmm2, early=False):
    H.load_plugin(MUTATIONS)
    import cutlass
    from sgl_nvidia_mega.kimi_k31.k3.cutedsl.common.host import layout, stream, to_cute
    from sgl_nvidia_mega.kimi_k31.dsa import verify as V

    q = 8
    seed = 5100 + 13 * rows + {"full": 0, "sharp": 1, "short": 2, "empty_req": 3}[kind] + int(bmm2 * 10)
    query, kv, selected, lengths, valid = case_inputs(rows, kind, seed)
    g = H.generator(seed + 1)
    uv = w_uv(g)
    pitch = 2112 + 128 * HEADS + 128
    projected = torch.randn((rows, pitch), generator=g, device="cuda").to(torch.bfloat16)
    gate = projected[:, 2112 : 2112 + 128 * HEADS]
    req_lengths = lengths.view(-1, q)[:, 0].contiguous() if kind == "empty_req" else torch.full(
        (rows // q,), 9000, dtype=torch.int32, device="cuda")
    fold_latent, fold_out, core = run_mla6(query, kv, selected, lengths, gate, uv, req_lengths, q, 128, bmm2)
    name = f"r{rows}_{kind}_s{bmm2}" + ("_fence" if early else "")

    split = V._split_fold_value(rows, pitch, layout(uv), torch.cuda.current_device(), q, HEADS, core.splits, early)
    assert getattr(split, "kind", None) == "split", f"{name}: _split_fold_value did not build the split form"
    latent_buf = torch.empty((rows, 1, 16, 512), dtype=torch.bfloat16, device="cuda")
    out = torch.empty((rows, HEADS * 128), dtype=torch.bfloat16, device="cuda")
    H.poison(latent_buf, out)
    split(to_cute(core.partial, 16), to_cute(core.stats, 8), to_cute(uv), to_cute(gate), to_cute(req_lengths, 4),
          cutlass.Float32(bmm2), to_cute(out), to_cute(latent_buf[:, 0, :HEADS]), stream(torch.device("cuda")))
    torch.cuda.synchronize()
    latent = latent_buf[:, 0, :HEADS]
    assert bool(torch.isnan(latent_buf[:, 0, HEADS:].float()).all()), f"{name}: wrote the padded heads 6-15"
    assert bool(torch.isfinite(latent.float()).all()), f"{name}: latent has non-finite (unwritten) cells"
    assert bool(torch.isfinite(out.float()).all()), f"{name}: output has non-finite (unwritten) cells"
    lat_same = torch.equal(latent.view(torch.int16), fold_latent.view(torch.int16))
    out_same = torch.equal(out.view(torch.int16), fold_out.view(torch.int16))
    H.record({"kind": "check", "item": f"fold_split_bitwise.{name}", "latent": lat_same, "out": out_same,
              "latent_cells_differ": int((latent.view(torch.int16) != fold_latent.view(torch.int16)).sum()),
              "out_cells_differ": int((out.view(torch.int16) != fold_out.view(torch.int16)).sum())})
    print("FOLD_SPLIT_BITWISE", name, lat_same, out_same, flush=True)

    ref = fp64_latent(query, kv, selected, valid, bmm2)
    empty = valid == 0
    if bool(empty.any()):
        assert bool((latent[empty].float() == 0).all()), f"{name}: rows without keys must give a zero latent"
        assert bool((out[empty].float() == 0).all()), f"{name}: rows without keys must give a zero output"
    err, _ = rel_err(latent, ref)
    vmax = ref.abs().amax(-1, keepdim=True)
    cell = (latent.float() - ref).abs() <= 2.0**-7 * ref.abs() + 2.0**-8 * vmax
    ref_out = R.value(R.bf16(ref), uv, gate, req_lengths, q)
    oerr = (out.float() - ref_out).norm(dim=-1) / ref_out.norm(dim=-1).clamp_min(1e-30)
    live = ref_out.norm(dim=-1) > 0
    omax = float(oerr[live].max()) if bool(live.any()) else 0.0
    H.record({"kind": "check", "item": f"fold_split_fp64.{name}", "latent_max": float(err.max()),
              "latent_mean": float(err.mean()), "cells_out": int((~cell).sum()), "out_max": omax})
    print("FOLD_SPLIT_FP64", name, float(err.max()), int((~cell).sum()), omax, flush=True)
    assert float(err.max()) <= 0.01, f"{name}: latent relative L2 error {float(err.max()):.4g} > 0.01"
    assert bool(cell.all()), f"{name}: {int((~cell).sum())} latent cells beyond 2^-7|ref| + 2^-8 max|ref|"
    assert omax <= 0.01, f"{name}: output relative L2 error {omax:.4g} > 0.01"
    # rows of a zero-length request: R.value zeroes them, so oerr skips them; check them directly
    dead = ~(req_lengths.repeat_interleave(q) > 0)
    assert bool((out[dead].float() == 0).all()), f"{name}: rows of a zero-length request must be zero"
    assert lat_same, f"{name}: latent differs from the served MlaValueFold's"
    assert out_same, f"{name}: output differs from the served MlaValueFold's"


# ----------------------------------------------------------------------------------------------- verify layer
def build_layer(rows, ctx, seed, split_rows=0):
    """One sparse verify MLA layer (Breq = rows / 8, Q8) as bench_verify_streams_l38 builds it, both arms (and a
    third, "split", with K31MK_FOLD_SPLIT_ROWS = ``split_rows``, when it is > 0)."""
    import deep_gemm
    from sglang.kernels.ops.attention.dsv4.topk import plan_topk_v2
    from sgl_nvidia_mega.kimi_k31.k3.cutedsl.mla.prepared import PreparedMla
    from sgl_nvidia_mega.kimi_k31.dsa import verify as V
    from sgl_nvidia_mega.kimi_k31.dsa.indexer import IndexWeights
    from sgl_nvidia_mega.kimi_k31.verify_indexer import IndexKeyWeights, PreparedVerifyIndexKey
    from test_k31_mla import fakes, index_pool, make_weights, num_sms

    assert int(os.environ.get("K31_TEST_HEADS", "12")) == HEADS, "run with K31_TEST_HEADS=6 (TP16 shapes)"
    Q = 8
    breq = rows // Q
    g = H.generator(seed)
    w = make_weights(seed + 1)
    npg = math.ceil(ctx / PAGE)
    pages = breq * npg + 2
    table = (torch.randperm(pages - 1, generator=g, device="cuda")[: breq * npg] + 1).to(torch.int32).view(breq, npg)
    pos = torch.arange(ctx - Q, ctx, device="cuda")
    slots = (table[:, pos // PAGE].long() * PAGE + pos % PAGE).reshape(-1)
    causal = (ctx - Q + 1 + torch.arange(Q, device="cuda")).repeat(breq).to(torch.int32)
    contexts = torch.full((breq, Q), ctx, dtype=torch.int32, device="cuda")
    step = SimpleNamespace(
        sparse=1, batch_requests=breq, query_width=Q, causal_lengths=causal, score_contexts=contexts,
        score_schedule=deep_gemm.get_paged_mqa_logits_metadata(contexts, PAGE, num_sms()), topk_plan=plan_topk_v2(causal),
    )
    kv0, idx0 = kv_pool(pages, g), index_pool(pages, g)
    wk = H.randn((128, 7168), 1 / math.sqrt(7168), g)
    pitch = 2112 + 128 * HEADS + 128
    projected = torch.randn((rows, pitch), generator=g, device="cuda").to(torch.bfloat16)
    hidden = torch.randn((rows, 7168), generator=g, device="cuda").to(torch.bfloat16)
    lengths = torch.full((breq,), ctx, dtype=torch.int32, device="cuda")
    attention, _ = fakes(w)
    scratch = torch.empty((rows, HEADS, 576), dtype=torch.uint8, device="cuda")
    base = PreparedMla(attention, {rows: pitch}, scratch, max_context=1 << 20, core="trtllm", queries=(Q,))
    arms = {}
    plan = [("stock", False, 0), ("mla6", True, 0)] + ([("split", True, split_rows)] if split_rows else [])
    for name, on, split in plan:
        default = V.SPARSE_MLA6, V.FOLD_SPLIT_ROWS
        V.SPARSE_MLA6, V.FOLD_SPLIT_ROWS = on, split
        try:
            core = V.PreparedSparseVerifyMla(base, IndexWeights(w.wq_b, w.w_proj, w.w_qa, 1e-5), rows, 1 << 20)
            kv, idx = kv0.clone(), idx0.clone()
            core.prepare_cache(kv, prepare_base=not arms)
            writer = V.SparseVerifyIndexKey(PreparedVerifyIndexKey(IndexKeyWeights(wk, w.gamma, w.beta, w.k_eps)), core)
        finally:
            V.SPARSE_MLA6, V.FOLD_SPLIT_ROWS = default
        assert (core.mla6 is not None) == on, "arm does not run the path it names"
        writer.bind(idx, PAGE, rows=(rows,))
        core.step = step
        out = torch.empty((rows, HEADS * 128), dtype=torch.bfloat16, device="cuda")
        arms[name] = SimpleNamespace(core=core, writer=writer, kv=kv, idx=idx, out=out)
    return SimpleNamespace(arms=arms, projected=projected, hidden=hidden, slots=slots, table=table, lengths=lengths,
                           causal=causal, w=w, rows=rows, base=base)


@needs_gpu
@pytest.mark.parametrize("rows,ctx", [(8, 8704), (16, 8704), (8, 131072)], ids=["b1_8k", "b2_8k", "b1_128k"])
def test_verify_layer(rows, ctx):
    H.load_plugin(MUTATIONS)
    L = build_layer(rows, ctx, 4200 + rows + ctx % 997)
    for name, a in L.arms.items():
        H.poison(a.out)
        if name == "mla6":
            H.poison(a.core.mla.query)  # the stock path's padded query copy: must stay unwritten
        a.writer.run(L.hidden, L.slots)
        a.core.run(L.projected, a.out, cache=a.kv, slots=L.slots, block_table=L.table, lengths=L.lengths, q=8)
        torch.cuda.synchronize()
        a.selected = a.core.selection.selected[:rows].clone()
        a.query = L.base.query_scratch[:rows].clone()
    s, m = L.arms["stock"], L.arms["mla6"]
    assert bool((m.core.mla.query.view(torch.uint8) == 0xFF).all()), "mla6 arm wrote the stock padded query"
    assert torch.equal(s.kv, m.kv) and torch.equal(s.query, m.query), "KV pool / MLA query differ between arms"
    same_sets = torch.equal(s.selected.sort(-1).values, m.selected.sort(-1).values)
    valid = torch.clamp(L.causal, max=TOPK)
    gate = L.projected[:, 2112 : 2112 + 128 * HEADS]
    name = f"b{rows // 8}_ctx{ctx}"
    errs = {}
    for arm, a in L.arms.items():
        lat = fp64_latent(a.query.view(rows, HEADS, 576), a.kv, a.selected, valid)
        ref_out = R.value(R.bf16(lat), L.w.w_uv, gate, L.lengths, 8)
        assert bool(torch.isfinite(a.out.float()).all()), f"{name}.{arm}: non-finite output"
        e = (a.out.float() - ref_out).norm(dim=-1) / ref_out.norm(dim=-1).clamp_min(1e-30)
        errs[arm] = (float(e.max()), float(e.mean()))
    H.record({"kind": "check", "item": f"l38_layer.{name}", "mla6_max": errs["mla6"][0], "mla6_mean": errs["mla6"][1],
              "stock_max": errs["stock"][0], "stock_mean": errs["stock"][1], "selected_sets_equal": same_sets})
    print("L38_LAYER", name, errs, flush=True)
    assert errs["mla6"][0] <= 0.01, f"{name}: mla6 layer output relative L2 error {errs['mla6'][0]:.4g} > 0.01"


@needs_gpu
@pytest.mark.parametrize("rows", [8, 64], ids=["b1", "b8"])
def test_verify_layer_split(rows):
    """K31MK_FOLD_SPLIT_ROWS=16 through PreparedSparseVerifyMla (8.7K context): the knob builds the split form at
    64 rows and leaves the served MlaValueFold at 8 rows (c=1); the split arm never writes the stock padded query;
    its layer output is within 0.01 relative L2 of the fp64 layer reference built from its own selected slots; and
    when its selection equals the mla6 arm's slot for slot, its output equals the mla6 arm's bitwise (stock top-k
    emission order is not deterministic, KNOWN K-G3-02a: the order equality is recorded; test_fold_split checks
    bitwise equality on shared partials unconditionally)."""
    H.load_plugin(MUTATIONS)
    ctx = 8704
    L = build_layer(rows, ctx, 4600 + rows, split_rows=16)
    want = "split" if rows >= 16 else None
    kinds = {n: getattr(a.core.sparse_values[rows, 8], "kind", None) for n, a in L.arms.items() if n != "stock"}
    assert kinds["mla6"] is None, "knob off must keep MlaValueFold"
    assert kinds["split"] == want, f"rows {rows}: split arm built {kinds['split']!r}, expected {want!r}"
    for name, a in L.arms.items():
        H.poison(a.out)
        if name != "stock":
            H.poison(a.core.mla.query)
        a.writer.run(L.hidden, L.slots)
        a.core.run(L.projected, a.out, cache=a.kv, slots=L.slots, block_table=L.table, lengths=L.lengths, q=8)
        torch.cuda.synchronize()
        a.selected = a.core.selection.selected[:rows].clone()
        a.query = L.base.query_scratch[:rows].clone()
    m, s = L.arms["mla6"], L.arms["split"]
    assert bool((s.core.mla.query.view(torch.uint8) == 0xFF).all()), "split arm wrote the stock padded query"
    assert torch.equal(m.kv, s.kv) and torch.equal(m.query, s.query), "KV pool / MLA query differ between arms"
    valid = torch.clamp(L.causal, max=TOPK)
    gate = L.projected[:, 2112 : 2112 + 128 * HEADS]
    lat = fp64_latent(s.query.view(rows, HEADS, 576), s.kv, s.selected, valid)
    ref_out = R.value(R.bf16(lat), L.w.w_uv, gate, L.lengths, 8)
    assert bool(torch.isfinite(s.out.float()).all()), "split arm: non-finite output"
    e = (s.out.float() - ref_out).norm(dim=-1) / ref_out.norm(dim=-1).clamp_min(1e-30)
    order_equal = torch.equal(m.selected, s.selected)
    bitwise = torch.equal(m.out.view(torch.int16), s.out.view(torch.int16))
    H.record({"kind": "check", "item": f"fold_split_layer.b{rows // 8}", "split_max": float(e.max()),
              "split_mean": float(e.mean()), "kind_built": kinds["split"], "selected_order_equal": order_equal,
              "out_bitwise_vs_mla6": bitwise})
    print("FOLD_SPLIT_LAYER", rows, float(e.max()), kinds["split"], order_equal, bitwise, flush=True)
    assert float(e.max()) <= 0.01, f"rows {rows}: split layer output relative L2 error {float(e.max()):.4g} > 0.01"
    if order_equal:
        assert bitwise, f"rows {rows}: same selection, but the split arm's output differs from the mla6 arm's"
