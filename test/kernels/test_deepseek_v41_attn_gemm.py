# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""The DeepSeek-V4.1 attention kernels on one SM100-family GPU (GB300).

- The TP4 attention projections wqkv_a (N 1792, K 5120), wq_b (N 8192, K 1280), wo_b
  (N 5120, K 2048) and the indexer's wq_b (N 4096, K 1280) on FlashInfer's low-latency
  MXFP8 GEMM launched directly (``mxfp8_ll.linear``): against SGLang's MXFP8 path for
  block-FP8-as-MXFP8 linears (FlashInfer ``mm_mxfp8`` ``cute-dsl`` with tuning off) on
  the same MXFP8 operands at 1-8 rows (plain swizzled scales) and 9-48 rows (row-tile
  scales), past 8 rows also bit for bit against the same tactic one 8-row tile at a
  time; the KV bytes that change (both wqkv_a outputs' kv columns through SGLang's KV
  store kernel into two caches, per byte and dequantized; synthetic weights, and real
  layers when ``SGL_NVIDIA_MEGA_DSV41_CKPT`` names the checkpoint); eager against a
  CUDA graph replay; every pinned tactic is one FlashInfer offers. The measured
  tolerances print as ``GEMM_REL`` / ``KV_TOLERANCE`` lines (``pytest -s``).
- Swap-AB sparse attention at 1-24 rows as ``attention._sparse_decode`` runs it (PDL,
  narrow combine, combine entry trigger; the query RoPE and KV-before-wait on the
  low-ratio layers): bit for bit against the same kernel without PDL, against SGLang's
  own 8-row kernel at 1-8 rows, within a BF16 bound of a float64 reference, eager and
  replayed with fresh metadata; the query RoPE inside attention against SGLang's q RoPE
  kernel followed by the stock call.
- The q-LoRA RMSNorm + MXFP8 quant with PDL at 1-48 rows against the launch without PDL
  and SGLang's own kernel at 1-8 rows (BF16 row, codes and swizzled scales, padding
  included) bit for bit, and FlashInfer's rmsnorm (1 ulp) + mxfp8_quantize (bitwise);
  feeding wq_b on memory that holds garbage (``fill``: every ``torch.empty`` filled with
  the poison byte; ``pool``: the allocator's blocks filled and freed first), the output
  bit for bit the same as on zeroed memory, eager and replayed (``POISON`` lines).
- The fused inverse RoPE + wo_a + MXFP8 kernel (src/wo_a_fused.cu) against SGLang's JIT
  build of the same device code, ``fused_rope_wo_a_bf16(..., out_mxfp8=True)``, bit for
  bit at every row count SGLang serves, int32 and int64 positions, the 64-head row
  stride and a packed one, eager and replayed; at 33-48 rows, where SGLang may have no
  48-row tile, against a torch reference within one FP8 step.
- The FlashMLA schedules the step head builds ahead (slot "sched_ahead"): bit for bit
  SGLang's lazy per-layer schedule (``_maybe_precompute_flashmla_sched_meta``) for every
  ratio, at verify widths with and without padded lengths, eager and replayed.
"""

import inspect
import json
import math
import os
from contextlib import contextmanager
from types import SimpleNamespace
from unittest import mock

import pytest
import torch

from sgl_nvidia_mega.deepseek_v41.attention import sched_ahead
from sgl_nvidia_mega.deepseek_v41.sm103 import mxfp8_ll
from sgl_nvidia_mega.deepseek_v41.sm103.attention import swapab, wo_a
from sgl_nvidia_mega.deepseek_v41.sm103.attention.q_norm_quant import rmsnorm_mxfp8

try:
    from sglang.srt.layers.attention import deepseek_v4_backend as backend
except ImportError:
    backend = None


# -------------------------------------------------------- low-latency MXFP8 projections

SHAPES = {
    "wqkv_a": (1792, 5120),
    "wq_b": (8192, 1280),
    "wo_b": (5120, 2048),
    # the indexer's wq_b (slot index_q_mxfp8_in), fed the same q-LoRA MXFP8 rows as wq_b
    "index_wq_b": (4096, 1280),
}
A03_ROWS = (1, 2, 3, 4, 5, 6, 7, 8)
A26_ROWS = (9, 12, 16, 17, 24, 32, 40, 48)
Q_LORA, HEAD_DIM, ROPE_DIM = 1280, 512, 64
EPS = 1e-20  # text_config.rms_norm_eps (kv_norm)
PAGE = 256  # SGLang's SWA page on SM100
MAX_POS = 1 << 16
KV_ROWS = (1, 5, 6, 8, 12, 24, 48)
SEEDS = 24
CKPT_ENV = "SGL_NVIDIA_MEGA_DSV41_CKPT"
CKPT_LAYERS = (0, 13, 26, 39)
# The BF16 outputs differ only by FP32 summation order (split-K, K tile)
GEMM_NORM_REL = 5e-4


def _ll_gemm_ready() -> bool:
    if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 10:
        return False
    try:
        import cutlass  # noqa: F401
        import flashinfer  # noqa: F401
    except ImportError:
        return False
    return mxfp8_ll.supported()


needs_ll_gemm = pytest.mark.skipif(
    not _ll_gemm_ready(), reason="needs an SM100-family GPU"
)


def _sm() -> int:
    major, minor = torch.cuda.get_device_capability()
    return major * 10 + minor


def _synthetic_weight(n, k, gen):
    w = torch.randn(n, k, device="cuda", generator=gen) / math.sqrt(k)
    w_q, w_sf = mxfp8_ll.quantize(w.bfloat16())
    return w_q.contiguous(), w_sf.reshape(-1).contiguous()


def _activations(rows, k, gen, norm_weight=None):
    """Rows as a pre-norm feeds them to a projection: heavy-tailed channels (a few
    massive ones), RMS-normalized, times the norm weight; BF16."""
    gain = torch.exp(0.75 * torch.randn(k, device="cuda", generator=gen))
    gain[torch.randperm(k, device="cuda", generator=gen)[:8]] *= 30.0
    h = torch.randn(rows, k, device="cuda", generator=gen) * gain
    y = h * torch.rsqrt(h.pow(2).mean(-1, keepdim=True) + 1e-6)
    if norm_weight is not None:
        y = y * norm_weight.float()
    return y.bfloat16()


def _cute_dsl(a, a_sf, w, w_sf):
    """SGLang's MXFP8 linear for block-FP8-as-MXFP8 weights (fp8_utils
    ``flashinfer_mxfp8_blockscaled_linear(backend="cute-dsl", pin_tactic=True)``)."""
    from flashinfer import mm_mxfp8
    from flashinfer.autotuner import autotune

    with autotune(False, skip_ops={"mxfp8_gemm"}):
        return mm_mxfp8(
            a,
            w.t(),
            a_sf.reshape(-1),
            w_sf,
            out_dtype=torch.bfloat16,
            use_8x4_sf_layout=False,
            backend="cute-dsl",
        )


def quantize_row_tiles(x):
    """FlashInfer's swizzled ``mxfp8_quantize`` of each 8-row tile on its own: codes
    [M, K] and the row-tile scale layout (one 128-row atom per tile, rows at its top)."""
    tiles = [mxfp8_ll.quantize(x[t : t + 8]) for t in range(0, x.shape[0], 8)]
    codes = torch.cat([q for q, _ in tiles])
    return codes, torch.cat([s.reshape(-1) for _, s in tiles])


def _ll(x, w, w_sf, tactic=None):
    """The plugin's arm: MXFP8 rows as the layer input arrives (plain swizzled scales
    up to 8 rows, row-tile scales past 8)."""
    if x.shape[0] <= mxfp8_ll.MAX_ROWS:
        a, a_sf = mxfp8_ll.quantize(x)
    else:
        a, a_sf = quantize_row_tiles(x)
    return mxfp8_ll.linear(a, a_sf.reshape(-1), w, w_sf, tactic=tactic), a


def _rel(out, ref):
    d = (out.float() - ref.float()).abs()
    return (d.norm() / ref.float().norm()).item(), d.max().item()


@pytest.fixture(scope="module")
def weights():
    gen = torch.Generator(device="cuda").manual_seed(7)
    return {name: _synthetic_weight(n, k, gen) for name, (n, k) in SHAPES.items()}


@needs_ll_gemm
def test_pinned_tactics_are_offered():
    sm = _sm()
    for (pin_sm, n, k, bucket), tactic in mxfp8_ll.BUCKET_TACTICS.items():
        if pin_sm != sm:
            continue
        offered = mxfp8_ll.offered_tactics(n, k, min(bucket, mxfp8_ll.MAX_ROWS))
        assert tactic in offered, (n, k, bucket, tactic)
        assert mxfp8_ll.select_tactic(sm, n, k, bucket) == tactic
    for n, k in SHAPES.values():
        assert (
            mxfp8_ll.select_tactic(sm, n, k, 8) == mxfp8_ll.PINNED_TACTICS[(sm, n, k)]
        )


@needs_ll_gemm
@pytest.mark.parametrize("name", list(SHAPES))
def test_low_latency_matches_cute_dsl(weights, name):
    n, k = SHAPES[name]
    w, w_sf = weights[name]
    gen = torch.Generator(device="cuda").manual_seed(11)
    worst = {}
    for rows in A03_ROWS + A26_ROWS:
        x = _activations(rows, k, gen)
        a, a_sf = mxfp8_ll.quantize(x)
        ref = _cute_dsl(a, a_sf, w, w_sf)
        out, a_ll = _ll(x, w, w_sf)
        torch.testing.assert_close(
            a_ll.view(torch.uint8), a.view(torch.uint8), rtol=0, atol=0
        )
        rel, max_abs = _rel(out, ref)
        print(f"GEMM_REL {name} rows={rows} norm_rel={rel:.3e} max_abs={max_abs:.3e}")
        if name == "wqkv_a":
            # the q-LoRA columns feed wq_b and the indexer query, not only the KV store
            q_rel, q_max = _rel(out[:, :Q_LORA], ref[:, :Q_LORA])
            print(
                f"GEMM_REL wqkv_a.q rows={rows} norm_rel={q_rel:.3e} max_abs={q_max:.3e}"
            )
            worst["q " + ("1-8" if rows <= 8 else "9-48")] = max(
                worst.get("q " + ("1-8" if rows <= 8 else "9-48"), 0.0), q_rel
            )
        assert rel <= GEMM_NORM_REL, (rows, rel)
        key = "1-8" if rows <= 8 else "9-48"
        worst[key] = max(worst.get(key, 0.0), rel)
        if rows > mxfp8_ll.MAX_ROWS:
            # one CTA per 8-row tile: the same tactic on each tile alone gives the same bits
            tactic = mxfp8_ll.select_tactic(_sm(), n, k, mxfp8_ll.row_bucket(rows))
            for t in range(0, rows, 8):
                tile, _ = _ll(x[t : t + 8], w, w_sf, tactic=tactic)
                assert torch.equal(tile, out[t : t + 8]), (rows, t)
    print(f"GEMM_REL_WORST {name} {json.dumps(worst)}")


@needs_ll_gemm
def test_wqkv_a_reads_the_boundarys_mxfp8_rows(weights):
    """Slot attn_qkv_a feeds the low-latency GEMM the MXFP8 rows SGLang's fused RMSNorm
    epilogue writes at the attention boundary, not FlashInfer's quantize: the same
    operands through SGLang's MXFP8 linear must agree to the GEMM bound at 1-8 rows."""
    from sglang.kernels.ops.layernorm.rmsnorm_mxfp8 import rmsnorm_mxfp8

    n, k = SHAPES["wqkv_a"]
    w, w_sf = weights["wqkv_a"]
    gen = torch.Generator(device="cuda").manual_seed(23)
    norm = torch.exp(0.3 * torch.randn(k, device="cuda", generator=gen)).bfloat16()
    for rows in (1, 2, 6, 8):
        h = (torch.randn(rows, k, device="cuda", generator=gen) * 3).bfloat16()
        _, q, sf = rmsnorm_mxfp8(h, norm, 1e-6)
        out = mxfp8_ll.linear(q, sf.reshape(-1), w, w_sf)
        ref = _cute_dsl(q, sf, w, w_sf)
        rel, max_abs = _rel(out, ref)
        print(
            f"GEMM_REL wqkv_a.boundary rows={rows} norm_rel={rel:.3e} max_abs={max_abs:.3e}"
        )
        assert rel <= GEMM_NORM_REL, (rows, rel)


@needs_ll_gemm
def test_draft_wo_b_reads_the_fused_wo_a_rows():
    """Slot draft_wo_b_ll: the low-latency wo_b (N 5120, K 2048) on the MXFP8 rows the
    fused inverse RoPE + wo_a writes, against SGLang's MXFP8 linear on the same rows,
    at the draft's 1-8 rows; eager equals a CUDA graph replay."""
    from sgl_nvidia_mega.deepseek_v41.sm103.attention import wo_a as wo_a_kernel

    gen = torch.Generator(device="cuda").manual_seed(29)
    w, w_sf = _synthetic_weight(5120, 2048, gen)
    wo_a = (torch.randn(2, 1024, 4096, device="cuda", generator=gen) / 64).bfloat16()
    freqs = torch.view_as_real(_freqs_cis()).flatten(1).contiguous()
    worst = 0.0
    for rows in (1, 2, 5, 6, 8):
        o = torch.randn(rows, 2, 4096, device="cuda", generator=gen).bfloat16()
        positions = torch.randint(0, MAX_POS, (rows,), device="cuda", generator=gen)
        codes, scales = wo_a_kernel.fused_rope_wo_a(o, wo_a, freqs, positions)
        out = mxfp8_ll.linear(codes, scales.reshape(-1), w, w_sf)
        ref = _cute_dsl(codes, scales, w, w_sf)
        rel, max_abs = _rel(out, ref)
        worst = max(worst, rel)
        print(
            f"GEMM_REL draft_wo_b.fused_wo_a rows={rows} norm_rel={rel:.3e} max_abs={max_abs:.3e}"
        )
        assert rel <= GEMM_NORM_REL, (rows, rel)
    print(f"GEMM_REL_WORST draft_wo_b {worst:.3e}")


def _padding_scale_mask(rows, k, numel):
    """Bytes of the swizzled UE8M0 scales a quantize of ``rows`` rows may leave
    unwritten: the row slots past the real rows of each 128-row atom (one atom at 1-8
    rows, one per 8-row tile past 8), at (r % 32) * 16 + (r // 32) * 4 + kb % 4 within
    each group of 4 K-blocks."""
    tiles = (
        [rows]
        if rows <= mxfp8_ll.MAX_ROWS
        else [min(8, rows - t) for t in range(0, rows, 8)]
    )
    per_tile = numel // len(tiles)
    index = [
        tile * per_tile + kb // 4 * 512 + r % 32 * 16 + r // 32 * 4 + kb % 4
        for tile, real in enumerate(tiles)
        for r in range(real, 128)
        for kb in range(k // 32)
    ]
    mask = torch.zeros(numel, dtype=torch.bool)
    mask[index] = True
    return mask.cuda()


@needs_ll_gemm
@pytest.mark.parametrize("name", list(SHAPES))
def test_low_latency_ignores_padding_row_scales(weights, name):
    """The low-latency GEMM reads whole 128-row scale atoms whose row slots past the
    real rows a producer may leave unwritten (graph-pool garbage in serving): with every
    such byte poisoned to 0xFF (a NaN UE8M0 scale), at 1-8 rows (plain) and 9-48 rows
    (row tiles), the real rows must not move; nor may a NaN output buffer leak in."""
    n, k = SHAPES[name]
    w, w_sf = weights[name]
    gen = torch.Generator(device="cuda").manual_seed(31)
    for rows in range(1, 49):
        x = _activations(rows, k, gen)
        if rows <= mxfp8_ll.MAX_ROWS:
            a, a_sf = mxfp8_ll.quantize(x)
        else:
            a, a_sf = quantize_row_tiles(x)
        a_sf = a_sf.reshape(-1).contiguous()
        clean = mxfp8_ll.linear(a, a_sf, w, w_sf)
        mask = _padding_scale_mask(rows, k, a_sf.numel())
        poisoned_sf = a_sf.clone()
        poisoned_sf[mask] = 0xFF
        assert torch.equal(poisoned_sf[~mask], a_sf[~mask])
        poisoned = mxfp8_ll.linear(a, poisoned_sf, w, w_sf)
        assert torch.equal(clean, poisoned), (name, rows)
        assert torch.isfinite(poisoned).all(), (name, rows)
        # split-K reduces in the cluster's shared memory and stores each output once
        out = torch.full((rows, n), float("nan"), dtype=torch.bfloat16, device="cuda")
        mxfp8_ll.linear(a, a_sf, w, w_sf, out=out)
        assert torch.equal(clean, out), ("out", name, rows)
    print(f"PADDING_SCALES {name} rows 1-48: real rows unchanged with 0xFF padding")


@needs_ll_gemm
@pytest.mark.parametrize("name", list(SHAPES))
def test_cuda_graph_matches_eager(weights, name):
    n, k = SHAPES[name]
    w, w_sf = weights[name]
    gen = torch.Generator(device="cuda").manual_seed(13)
    rows_list = (1, 2, 4, 6, 8, 12, 24, 48)
    mxfp8_ll.precompile(w, list(rows_list))
    for rows in rows_list:
        x = _activations(rows, k, gen)
        if rows <= mxfp8_ll.MAX_ROWS:
            a, a_sf = mxfp8_ll.quantize(x)
        else:
            a, a_sf = quantize_row_tiles(x)
        a_sf = a_sf.reshape(-1)
        eager = mxfp8_ll.linear(a, a_sf, w, w_sf)
        out = torch.zeros_like(eager)
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            mxfp8_ll.linear(a, a_sf, w, w_sf, out=out)  # warm-up off the capture
        torch.cuda.current_stream().wait_stream(stream)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            mxfp8_ll.linear(a, a_sf, w, w_sf, out=out)
        out.zero_()
        graph.replay()
        torch.cuda.synchronize()
        assert torch.equal(out, eager), rows


# --- KV tolerance --------------------------------------------------------------------


def _ckpt_tensor(root, index, name):
    from safetensors import safe_open

    with safe_open(os.path.join(root, index[name]), framework="pt", device="cpu") as f:
        return f.get_tensor(name)


def _ue8m0_bytes(scale):
    if scale.dtype == torch.float32:
        return ((scale.view(torch.int32) >> 23) & 255).to(torch.uint8)
    return scale.view(torch.uint8)


def _kv_layers(source):
    """(label, wqkv_a weight, swizzled scales, attn_norm weight, kv_norm weight) per layer:
    wq_a and wkv stacked as SGLang fuses them, 32x32 UE8M0 blocks expanded to rows."""
    from flashinfer import block_scale_interleave

    if source == "synthetic":
        gen = torch.Generator(device="cuda").manual_seed(17)
        w, w_sf = _synthetic_weight(*SHAPES["wqkv_a"], gen)
        attn_norm = torch.exp(0.3 * torch.randn(5120, device="cuda", generator=gen))
        kv_norm = torch.exp(0.3 * torch.randn(HEAD_DIM, device="cuda", generator=gen))
        return [("synthetic", w, w_sf, attn_norm.bfloat16(), kv_norm.bfloat16())]
    root = os.environ.get(CKPT_ENV)
    if not root:
        pytest.skip(f"{CKPT_ENV} is not set")
    with open(os.path.join(root, "model.safetensors.index.json")) as f:
        index = json.load(f)["weight_map"]
    layers = []
    for layer in CKPT_LAYERS:
        p = f"layers.{layer}."
        get = lambda name: _ckpt_tensor(root, index, p + name)  # noqa: E731
        w = torch.cat(
            [get(f"attn.{m}.weight").view(torch.uint8) for m in ("wq_a", "wkv")]
        )
        scale = torch.cat(
            [_ue8m0_bytes(get("attn.wq_a.scale")), _ue8m0_bytes(get("attn.wkv.scale"))]
        )
        scale = scale.repeat_interleave(32, dim=0)
        assert w.shape == SHAPES["wqkv_a"] and scale.shape == (1792, 5120 // 32)
        w_sf = block_scale_interleave(scale.cuda().contiguous()).reshape(-1)
        layers.append(
            (
                f"layer{layer}",
                w.cuda().view(torch.float8_e4m3fn).contiguous(),
                w_sf.contiguous(),
                get("attn_norm.weight").cuda().bfloat16(),
                get("attn.kv_norm.weight").cuda().bfloat16(),
            )
        )
    return layers


def _freqs_cis():
    inv = 10000.0 ** (-torch.arange(0, ROPE_DIM, 2, device="cuda").float() / ROPE_DIM)
    angles = torch.arange(MAX_POS, device="cuda").float()[:, None] * inv[None, :]
    return torch.polar(torch.ones_like(angles), angles)


def _store(kv, kv_norm, freqs_cis, positions, loc, pages, layout):
    """SGLang's KV store kernel, as set_swa_key_buffer_radix_fused_norm_rope calls it."""
    from sglang.kernels.ops.attention.dsv4 import fused_k_norm_rope_flashmla

    cache = torch.zeros(
        pages, layout.page_bytes(PAGE), dtype=torch.uint8, device="cuda"
    )
    fused_k_norm_rope_flashmla(
        kv=kv,
        kv_weight=kv_norm,
        eps=EPS,
        freqs_cis=freqs_cis,
        positions=positions,
        out_loc=loc,
        kvcache=cache,
        page_size=PAGE,
        layout=layout.value,
    )
    return cache


def _regions(cache, loc, layout):
    """The tokens at ``loc`` by region: FP8 codes [T, n], the UE8M0 scale of each
    code's block [T, n], BF16 RoPE values [T, 64] or None, and the scale bytes."""
    from sglang.kernels.ops.attention.dsv4.kv_layout import KVLayout

    page, slot = (loc // PAGE).long(), (loc % PAGE).long()
    width, sb = layout.data_bytes, layout.scale_bytes
    data = cache[:, : PAGE * width].view(-1, PAGE, width)[page, slot]
    off = layout.scale_offset(PAGE)
    scales = cache[:, off : off + PAGE * sb].view(-1, PAGE, sb)[page, slot]
    # V4 (SGLang's default window cache): 448 FP8 nope in 64-value blocks, 64 BF16
    # RoPE values, 7 scales + a pad; V4.1: 512 FP8 in 32-value blocks, 16 scales
    if layout is KVLayout.V4:
        fp8, rope, block, nscales = data[:, :448], data[:, 448:576], 64, 7
        rope = rope.contiguous().view(torch.bfloat16)
    else:
        fp8, rope, block, nscales = data, None, 32, 16
    code_scale = scales[:, :nscales].repeat_interleave(block, dim=1)
    return fp8, code_scale, rope, scales[:, :nscales]


def _ordered(bits, sign_bit):
    """Sign-magnitude codes as integers whose difference counts representable steps."""
    bits = bits.long()
    mag = bits & (sign_bit - 1)
    return torch.where(bits & sign_bit != 0, -mag, mag)


@needs_ll_gemm
@pytest.mark.parametrize("layout_name", ["v4", "v41"])
@pytest.mark.parametrize("source", ["synthetic", "checkpoint"])
def test_kv_bytes_tolerance(source, layout_name):
    """wqkv_a on the low-latency GEMM against SGLang's, through SGLang's KV store in
    each window-cache layout: FP8 codes in e4m3 steps (where both arms share the block
    scale), BF16 RoPE values in ulps, scale bytes exact or not."""
    from sglang.kernels.ops.attention.dsv4.kv_layout import KVLayout

    layout = KVLayout(layout_name)
    freqs_cis = _freqs_cis()
    pages = -(-max(KV_ROWS) * 4 // PAGE) + 1
    st = dict(tokens=0, cases=0, fp8=0, fp8_moved=0, rope=0, rope_moved=0)
    st.update(scales=0, scales_moved=0, kv_bf16_moved=0)
    max_steps, max_ulp, worst_rel = 0, 0, 0.0
    # peak-relative: a tiny change near zero spans many fine codes / ulps, so each region
    # is also measured against its coarsest quantum (FP8: the block's top-binade step,
    # 32 x the block scale; BF16 rope: the row's peak magnitude)
    max_top_steps, max_rope_rel = 0.0, 0.0
    for label, w, w_sf, attn_norm, kv_norm in _kv_layers(source):
        mxfp8_ll.precompile(w, list(KV_ROWS))
        for seed in range(SEEDS):
            gen = torch.Generator(device="cuda").manual_seed(1000 + seed)
            for rows in KV_ROWS:
                x = _activations(rows, 5120, gen, attn_norm)
                a, a_sf = mxfp8_ll.quantize(x)
                ref = _cute_dsl(a, a_sf, w, w_sf)
                out, _ = _ll(x, w, w_sf)
                # each arm is bitwise across reruns (fixed tactics)
                assert torch.equal(_cute_dsl(a, a_sf, w, w_sf), ref), (label, rows)
                assert torch.equal(_ll(x, w, w_sf)[0], out), (label, seed, rows)
                worst_rel = max(worst_rel, _rel(out, ref)[0])
                positions = torch.randint(
                    0, MAX_POS, (rows,), device="cuda", generator=gen
                )
                loc = torch.randperm(pages * PAGE, device="cuda", generator=gen)[:rows]
                loc = loc.int()
                kv_ref, kv_ll = ref[:, Q_LORA:], out[:, Q_LORA:]
                args = (kv_norm, freqs_cis, positions, loc, pages, layout)
                c_ref, c_ll = _store(kv_ref, *args), _store(kv_ll, *args)
                assert torch.equal(_store(kv_ref, *args), c_ref)
                f_ref, cs_ref, r_ref, s_ref = _regions(c_ref, loc, layout)
                f_ll, cs_ll, r_ll, s_ll = _regions(c_ll, loc, layout)
                same_scale = cs_ref == cs_ll
                steps = (_ordered(f_ref, 0x80) - _ordered(f_ll, 0x80)).abs()
                st["fp8"] += f_ref.numel()
                st["fp8_moved"] += int((f_ref != f_ll).sum())
                shared = steps[same_scale]
                if shared.numel():
                    max_steps = max(max_steps, int(shared.max()))
                unit_ref = torch.exp2(cs_ref.float() - 127.0)
                unit_ll = torch.exp2(cs_ll.float() - 127.0)
                deq_ref = f_ref.view(torch.float8_e4m3fn).float() * unit_ref
                deq_ll = f_ll.view(torch.float8_e4m3fn).float() * unit_ll
                top_step = 32.0 * torch.maximum(unit_ref, unit_ll)
                max_top_steps = max(
                    max_top_steps, ((deq_ll - deq_ref).abs() / top_step).max().item()
                )
                if r_ref is not None:
                    ulp = (
                        _ordered(r_ref.view(torch.int16).long() & 0xFFFF, 0x8000)
                        - _ordered(r_ll.view(torch.int16).long() & 0xFFFF, 0x8000)
                    ).abs()
                    st["rope"] += r_ref.numel()
                    st["rope_moved"] += int((r_ref != r_ll).sum())
                    max_ulp = max(max_ulp, int(ulp.max()))
                    peak = r_ref.float().abs().amax(-1, keepdim=True).clamp_min(1e-30)
                    rel = ((r_ll.float() - r_ref.float()).abs() / peak).max().item()
                    max_rope_rel = max(max_rope_rel, rel)
                st["scales"] += s_ref.numel()
                st["scales_moved"] += int((s_ref != s_ll).sum())
                st["kv_bf16_moved"] += int((kv_ref != kv_ll).sum())
                st["tokens"] += rows
                st["cases"] += 1
    print(
        "KV_TOLERANCE "
        + json.dumps(
            dict(
                layout=layout_name,
                weights=source if source == "synthetic" else list(CKPT_LAYERS),
                seeds=SEEDS,
                rows=KV_ROWS,
                cases=st["cases"],
                tokens=st["tokens"],
                kv_bf16_elements_moved=st["kv_bf16_moved"] / (st["tokens"] * HEAD_DIM),
                fp8_codes_moved=st["fp8_moved"] / st["fp8"],
                fp8_max_e4m3_steps=max_steps,
                fp8_max_diff_in_block_top_steps=max_top_steps,
                rope_bf16_values_moved=(st["rope_moved"] / st["rope"])
                if st["rope"]
                else None,
                rope_max_bf16_ulp=max_ulp if st["rope"] else None,
                rope_max_diff_over_row_peak=max_rope_rel if st["rope"] else None,
                scale_bytes_moved=st["scales_moved"] / st["scales"],
                wqkv_a_worst_norm_rel=worst_rel,
                reruns_bitwise=True,
            )
        )
    )
    # at most one step of the block's coarsest quantum (codes near zero move by more
    # fine steps for the same absolute change)
    assert max_top_steps <= 1.0, max_top_steps


# ---------------------------------------- swap-AB attention and the q-LoRA norm + quant

SWAPAB_ROWS = (1, 2, 3, 4, 5, 6, 7, 8, 9, 12, 16, 20, 24)
# The plugin's launch (attention._sparse_decode) without and with the low-ratio extras
PLUGIN = dict(max_rows=24, pdl=True, narrow_combine=True, combine_entry_trigger=True)
# The same arithmetic launched as a plain grid: PDL and the triggers only move waits
PLAIN = dict(max_rows=24, narrow_combine=True)


def _attention_ready() -> bool:
    if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 10:
        return False
    try:
        import flashinfer  # noqa: F401
        from triton.experimental import gluon  # noqa: F401
    except ImportError:
        return False
    return True


needs_attention = pytest.mark.skipif(
    not _attention_ready(), reason="needs an SM100-family GPU"
)


def _make_cache(page, pages=11):
    """A padded V4 cache with poisoned unused bytes, and its dequantized values."""
    stride = page * 584 + 256
    raw = torch.full((pages, stride), 0x7F, dtype=torch.uint8, device="cuda")
    fp8 = torch.randn(pages, page, 448, device="cuda").to(torch.float8_e4m3fn)
    exps = torch.randint(124, 130, (pages, page, 7), device="cuda", dtype=torch.uint8)
    rope = torch.randn(pages, page, 64, device="cuda").to(torch.bfloat16)
    tokens = raw[:, : page * 576].view(pages, page, 576)
    tokens[..., :448].copy_(fp8.view(torch.uint8))
    tokens[..., 448:].copy_(rope.view(torch.uint8))
    raw[:, page * 576 : page * 584].view(pages, page, 8)[..., :7].copy_(exps)
    nope = (
        fp8.float().view(pages, page, 7, 64) * torch.exp2(exps.float() - 127)[..., None]
    )
    values = torch.cat((nope.flatten(-2), rope.float()), -1).reshape(-1, 512)
    return raw.as_strided((pages, page, 1, 584), (stride, 584, 584, 1)), values


def _swapab_inputs(b, nk=192, ne=512, page=64, seed=0):
    """BEST's swap-AB case: strided query and index rows, out-of-range and negative
    slots, per-row edge lengths, and -inf / +inf / large sinks."""
    torch.manual_seed(20260915 + b + nk + ne + seed)
    storage = torch.full(
        (b, 1, 64, 512), float("nan"), device="cuda", dtype=torch.bfloat16
    )
    q = storage[:, :, :16]
    q.copy_(torch.randn_like(q) * 0.5)
    kv, values = _make_cache(page)
    ekv, evalues = _make_cache(16)
    ids = torch.randint(
        -3, values.shape[0] + 4, (b, 1, nk + 19), device="cuda", dtype=torch.int32
    )[..., :nk]
    ei = torch.randint(
        -3, evalues.shape[0] + 4, (b, 1, ne + 17), device="cuda", dtype=torch.int32
    )[..., :ne]
    lengths = torch.tensor(
        [[0, 1, 31, 63, 64, nk - 1, nk, nk + 3][i % 8] for i in range(b)],
        device="cuda",
        dtype=torch.int32,
    )
    el = torch.tensor(
        [[ne, ne // 2, 1, 0, 63, ne - 1, 65, ne][i % 8] for i in range(b)],
        device="cuda",
        dtype=torch.int32,
    )
    sink = torch.randn(64, device="cuda")
    sink[0], sink[1], sink[2] = -torch.inf, torch.inf, 20.0
    extra = (ekv, ei, el) if ne else (None, None, None)
    return (q, kv, ids, lengths, sink, *extra), (values, evalues)


def _swapab_reference(args, values):
    q, _, ids, lengths, sink, _, ei, el = args

    def gather(vals, idx, lens):
        idx = idx[:, 0]
        valid = (idx >= 0) & (idx < vals.shape[0])
        valid &= torch.arange(idx.shape[-1], device=idx.device)[None, :] < lens[:, None]
        data = vals[idx.clamp(0, vals.shape[0] - 1).long()].double()
        return torch.where(valid[..., None], data, 0), valid

    kv, valid = gather(values[0], ids, lengths)
    if ei is not None:
        ev, em = gather(values[1], ei, el)
        kv, valid = torch.cat((kv, ev), 1), torch.cat((valid, em), 1)
    scores = q[:, 0].double() @ kv.transpose(1, 2) * 512**-0.5
    scores.masked_fill_(~valid[:, None, :], -torch.inf)
    sinks = sink[None, :16, None].double().expand(q.shape[0], -1, -1)
    prob = torch.nan_to_num(torch.softmax(torch.cat((scores, sinks), -1), -1)[..., :-1])
    return prob @ kv


def _check_reference(out, args, values):
    gold = _swapab_reference(args, values)
    floor = (gold.bfloat16().double() - gold).square().mean().sqrt()
    error = (out.double() - gold).square().mean().sqrt()
    assert error.item() <= floor.item() * 1.01 + 1e-6
    torch.testing.assert_close(out.float(), gold.float(), atol=2e-5, rtol=0.004)


def _bitwise(a, b):
    assert torch.equal(a.view(torch.int16), b.view(torch.int16)), (
        f"{(a != b).sum().item()} of {a.numel()} differ, "
        f"max |diff| {(a.float() - b.float()).abs().max().item():.3e}"
    )


def _main_swapab():
    try:
        from sglang.kernels.ops.attention.dsv4.decode_attention_sm100 import (
            swapab_attention,
        )
    except ImportError:
        return None
    return swapab_attention


@needs_attention
@pytest.mark.parametrize("ne", [512, 0])
@pytest.mark.parametrize("b", SWAPAB_ROWS)
def test_swapab_plugin_launch(b, ne):
    args, values = _swapab_inputs(b, ne=ne)
    originals = [x.clone() for x in args if x is not None]
    out = swapab.swapab_attention(*args, **PLUGIN)
    _bitwise(out, swapab.swapab_attention(*args, **PLAIN))
    _check_reference(out, args, values)
    wide = swapab.swapab_attention(*args, max_rows=24)  # main's combine shape
    _check_reference(wide, args, values)
    for before, after in zip(originals, (x for x in args if x is not None)):
        torch.testing.assert_close(before, after, rtol=0, atol=0, equal_nan=True)
    main = _main_swapab()
    if b <= 8 and main is not None:
        _bitwise(wide, main(*args))
        if not ne:  # no extra keys: the narrow combine is not taken
            _bitwise(out, main(*args))


@needs_attention
@pytest.mark.parametrize("b", [1, 2, 6, 8, 12, 24])
def test_swapab_graph_replay(b):
    args, values = _swapab_inputs(b)
    q, _, ids, lengths, _, _, ei, el = args
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        swapab.swapab_attention(*args, **PLUGIN)
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        replayed = swapab.swapab_attention(*args, **PLUGIN)
    for i in range(8):
        # Device metadata and the query change under the graph: nothing is host-baked
        q.copy_(torch.randn_like(q) * (0.2 + i * 0.05))
        ids.copy_(
            torch.randint(
                -1, values[0].shape[0], ids.shape, device="cuda", dtype=torch.int32
            )
        )
        lengths.copy_(
            torch.randint(
                0, ids.shape[-1] + 1, lengths.shape, device="cuda", dtype=torch.int32
            )
        )
        ei.copy_(
            torch.randint(
                -1, values[1].shape[0], ei.shape, device="cuda", dtype=torch.int32
            )
        )
        el.copy_(
            torch.randint(
                0, ei.shape[-1] + 1, el.shape, device="cuda", dtype=torch.int32
            )
        )
        replayed.zero_()
        graph.replay()
        _bitwise(replayed, swapab.swapab_attention(*args, **PLAIN))
        _check_reference(replayed, args, values)


# 64 rows (65K rotated values) is what catches a reordered RoPE expression: rounding
# order flips about 1 in 30K values; the plugin serves 1-24.
@needs_attention
@pytest.mark.parametrize("b", [1, 2, 4, 5, 6, 8, 12, 16, 24, 64])
def test_swapab_q_rope_and_kv_before_wait(b):
    try:
        from sglang.kernels.ops.attention.dsv4 import fused_q_norm_rope
    except ImportError:
        pytest.skip("this SGLang has no fused_q_norm_rope")
    kv, _ = _make_cache(64)
    extra, _ = _make_cache(16)
    angles = torch.rand(70000, 32, device="cuda") * 6.3
    freqs_cis = torch.polar(torch.ones_like(angles), angles)
    freqs = torch.view_as_real(freqs_cis).flatten(-2)
    torch.manual_seed(77 + b)
    q_raw = (torch.randn(b, 16, 512, device="cuda") * 3).to(torch.bfloat16)
    positions = torch.randint(70000, (b,), device="cuda", dtype=torch.int64)
    ids = torch.randint(0, 64 * 11, (b, 1, 128), device="cuda", dtype=torch.int32)
    lengths = torch.full((b,), 128, device="cuda", dtype=torch.int32)
    ei = torch.randint(0, 16 * 11, (b, 1, 512), device="cuda", dtype=torch.int32)
    el = torch.full((b,), 500, device="cuda", dtype=torch.int32)
    sink = torch.randn(64, device="cuda")
    q_pad = torch.zeros(b, 64, 512, device="cuda", dtype=torch.bfloat16)
    fused_q_norm_rope(q_raw, q_pad[:, :16], None, freqs_cis, positions)
    tail = (kv, ids, lengths, sink, extra, ei, el)
    rows = dict(max_rows=64) if b > 24 else {}
    expected = swapab.swapab_attention(q_pad[:, :16], *tail, **{**PLAIN, **rows})
    _bitwise(
        swapab.swapab_attention(q_pad[:, :16], *tail, **{**PLUGIN, **rows}), expected
    )

    def fused(**kwargs):
        return swapab.swapab_attention(
            q_raw, *tail, **{**PLUGIN, **rows}, q_rope=(freqs, positions), **kwargs
        )

    for kv_before_wait in (False, True):
        _bitwise(fused(kv_before_wait=kv_before_wait), expected)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        fused(kv_before_wait=True)
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        out = fused(kv_before_wait=True)
    out.zero_()
    graph.replay()
    torch.cuda.synchronize()
    _bitwise(out, expected)


def _check_flashinfer(x, w, got):
    """FlashInfer's rmsnorm within 1 BF16 ulp (it reduces in another order: 1-3 of
    10-60K elements differ on GB300), and its swizzled mxfp8_quantize of our row bit
    for bit, padding scales zero."""
    import flashinfer

    y, q, sf = got
    ulps = y.view(torch.int16).int() - flashinfer.norm.rmsnorm(x, w, 1e-6).view(
        torch.int16
    )
    assert ulps.abs().max().item() <= 1
    eq, esf = flashinfer.mxfp8_quantize(y, is_sf_swizzled_layout=True)
    assert torch.equal(q.view(torch.uint8), eq.view(torch.uint8))
    g = torch.arange(Q_LORA // 32, device=x.device)
    row = torch.arange(x.shape[0], device=x.device)[:, None]
    offsets = (g // 4) * 512 + ((row % 32) * 4 + row // 32) * 4 + g % 4
    assert torch.equal(sf[offsets], esf.flatten()[offsets])
    pad = torch.ones_like(sf, dtype=torch.bool)
    pad[offsets] = False
    assert torch.count_nonzero(sf[pad]) == 0


def _same(got, want):
    for a, b in zip(got, want):
        assert a.shape == b.shape
        assert torch.equal(a.view(torch.uint8), b.view(torch.uint8))


@needs_attention
@pytest.mark.parametrize("stride", [Q_LORA, 1792])
@pytest.mark.parametrize("m", range(1, 49))
def test_rmsnorm_mxfp8_pdl(m, stride):
    torch.manual_seed(941 + m)
    x = torch.randn((m, stride), device="cuda", dtype=torch.bfloat16)[:, :Q_LORA]
    w = torch.randn(Q_LORA, device="cuda", dtype=torch.bfloat16)
    got = rmsnorm_mxfp8(x, w, 1e-6, pdl=True)
    _same(got, rmsnorm_mxfp8(x, w, 1e-6))
    try:
        from sglang.kernels.ops.layernorm.rmsnorm_mxfp8 import (
            rmsnorm_mxfp8 as main_rmsnorm_mxfp8,
        )
    except ImportError:
        main_rmsnorm_mxfp8 = None
    if m <= 8 and main_rmsnorm_mxfp8 is not None:
        _same(got, main_rmsnorm_mxfp8(x, w, 1e-6))
    _check_flashinfer(x, w, got)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        replayed = rmsnorm_mxfp8(x, w, 1e-6, pdl=True)
    for scale in (0, 1e-4, 1, 100):
        x.copy_(torch.randn_like(x) * scale)
        w.copy_(torch.randn_like(w))
        for t in replayed:
            t.view(torch.uint8).fill_(0xFF)
        graph.replay()
        _same(replayed, rmsnorm_mxfp8(x, w, 1e-6))
        _check_flashinfer(x, w, replayed)


@needs_attention
@pytest.mark.parametrize("m", range(9, 49))
def test_rmsnorm_mxfp8_row_tiles(m):
    """FlashInfer has no row-tile quantize, so the reference is its swizzled
    mxfp8_quantize of each 8-row tile of our BF16 row alone: one 128-row atom per tile
    with the tile's rows at its top, which is the row-tile layout. Codes and the live
    scale bytes are bit for bit; rows past m in the last tile carry zero scales."""
    import flashinfer

    torch.manual_seed(977 + m)
    x = torch.randn((m, Q_LORA), device="cuda", dtype=torch.bfloat16)
    w = torch.randn(Q_LORA, device="cuda", dtype=torch.bfloat16)
    y, q, sf = rmsnorm_mxfp8(x, w, 1e-6, pdl=True, row_tiles=True)
    plain = rmsnorm_mxfp8(x, w, 1e-6)
    assert torch.equal(y, plain[0])
    assert torch.equal(q.view(torch.uint8), plain[1].view(torch.uint8))
    atom = (Q_LORA // 128) * 512
    assert sf.numel() == -(-m // 8) * atom
    eq, _ = flashinfer.mxfp8_quantize(y, is_sf_swizzled_layout=True)
    assert torch.equal(q.view(torch.uint8), eq.view(torch.uint8))
    g = torch.arange(Q_LORA // 32, device=x.device)
    within = (g // 4) * 512 + torch.arange(8, device=x.device)[:, None] * 16 + g % 4
    for t in range(0, m, 8):
        rows = min(8, m - t)
        _, tile_sf = flashinfer.mxfp8_quantize(y[t : t + 8], is_sf_swizzled_layout=True)
        got = sf[(t // 8) * atom + within]
        assert torch.equal(got[:rows], tile_sf.flatten()[within[:rows]]), t
        assert torch.count_nonzero(got[rows:]) == 0, t


# --------------------------------------------------------------------------- fused wo_a

WO_A_SEEDS = (0, 1, 2)
MAX_POSITION = 4096
HEADS = 64  # the attention output x views: [T, 64, 512]


def _wo_a_ready() -> bool:
    if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 10:
        return False
    try:
        from cuda.bindings import nvrtc  # noqa: F401
    except ImportError:
        return False
    return True


needs_wo_a = pytest.mark.skipif(not _wo_a_ready(), reason="needs an SM100-family GPU")


def _wo_a_reference():
    """SGLang's wrapper and the most rows it serves; skips without SGLang."""
    module = pytest.importorskip("sglang.kernels.ops.attention.dsv4.wo_a")
    # 48 rows exist only where SGLang has the shared-inbox tile.
    return module, getattr(module, "WIDE_MAX_M", module.MAX_M)


def _wo_a_inputs(rows: int, seed: int, int64_positions: bool, padded_heads: bool):
    generator = torch.Generator(device="cuda").manual_seed(seed)
    if padded_heads:
        attention = torch.randn(
            (rows, HEADS, HEAD_DIM),
            generator=generator,
            device="cuda",
            dtype=torch.bfloat16,
        )
        x = attention.view(rows, HEADS * HEAD_DIM // wo_a.GROUP_K, wo_a.GROUP_K)[
            :, : wo_a.GROUPS
        ]
    else:
        x = torch.randn(
            (rows, wo_a.GROUPS, wo_a.GROUP_K),
            generator=generator,
            device="cuda",
            dtype=torch.bfloat16,
        )
    if seed == 1:
        x[0].zero_()  # an all-zero row takes the zero-scale branch
    weight = (
        torch.randn(
            (wo_a.GROUPS, wo_a.RANK, wo_a.GROUP_K),
            generator=generator,
            device="cuda",
            dtype=torch.float32,
        )
        * 0.02
    ).to(torch.bfloat16)
    angles = (
        torch.rand(
            (MAX_POSITION, wo_a.ROPE_DIM // 2),
            generator=generator,
            device="cuda",
        )
        * 6.2831855
    )
    freqs = torch.stack((angles.cos(), angles.sin()), dim=-1).reshape(
        MAX_POSITION, wo_a.ROPE_DIM
    )
    positions = torch.randint(
        0, MAX_POSITION, (rows,), generator=generator, device="cuda"
    ).to(torch.int64 if int64_positions else torch.int32)
    return x, weight, freqs, positions


def _assert_bitwise(actual, expected) -> None:
    codes, scales = actual
    expected_codes, expected_scales = expected
    assert codes.shape == expected_codes.shape and scales.shape == expected_scales.shape
    assert torch.equal(codes.view(torch.uint8), expected_codes.view(torch.uint8))
    assert torch.equal(scales, expected_scales)


@needs_wo_a
@pytest.mark.parametrize("padded_heads", [True, False])
@pytest.mark.parametrize("int64_positions", [False, True])
@pytest.mark.parametrize("rows", range(1, wo_a.MAX_ROWS + 1))
def test_wo_a_fused_matches_sglang(rows, int64_positions, padded_heads):
    reference, reference_max_rows = _wo_a_reference()
    if rows > reference_max_rows:
        pytest.skip(f"SGLang serves at most {reference_max_rows} rows")
    parameters = inspect.signature(reference.fused_rope_wo_a_bf16).parameters
    for seed in WO_A_SEEDS:
        x, weight, freqs, positions = _wo_a_inputs(
            rows, seed, int64_positions, padded_heads
        )
        expected = reference.fused_rope_wo_a_bf16(x, weight, freqs, positions)
        for early_trigger in (True, False):
            _assert_bitwise(
                wo_a.fused_rope_wo_a(
                    x, weight, freqs, positions, early_trigger=early_trigger
                ),
                expected,
            )
        if "row_tile_scales" in parameters:
            _assert_bitwise(
                wo_a.fused_rope_wo_a(x, weight, freqs, positions, row_tile_scales=True),
                reference.fused_rope_wo_a_bf16(
                    x, weight, freqs, positions, row_tile_scales=True
                ),
            )
    torch.cuda.synchronize()


@needs_wo_a
@pytest.mark.parametrize("rows", [1, 5, 8, 16, 17, 32, 33, 48])
def test_wo_a_fused_graph_replay_matches_eager(rows):
    wo_a.precompile(rows)
    x, weight, freqs, positions = _wo_a_inputs(
        rows, seed=3, int64_positions=False, padded_heads=True
    )
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.stream(stream), torch.cuda.graph(graph, stream=stream):
        captured = wo_a.fused_rope_wo_a(x, weight, freqs, positions)
    torch.cuda.current_stream().wait_stream(stream)
    # New inputs in the captured buffers: the replay must read them, not the capture's.
    for seed in (4, 5):
        fresh = _wo_a_inputs(rows, seed=seed, int64_positions=False, padded_heads=True)
        for buffer, values in zip((x, weight, freqs, positions), fresh):
            buffer.copy_(values)
        graph.replay()
        expected = wo_a.fused_rope_wo_a(x, weight, freqs, positions)
        _assert_bitwise(captured, expected)
    torch.cuda.synchronize()


def _torch_reference(x, weight, freqs, positions):
    """Inverse RoPE on each head's last 64 lanes (interleaved (real, imag) pairs times
    conj(cos + i sin), rounded to BF16 as the kernel stores them), then the per-group
    product in float64, rounded to BF16: [T, 2048]."""
    rows = x.shape[0]
    heads = x.float().reshape(rows, wo_a.GROUPS, -1, 512)
    pair = heads[..., 512 - wo_a.ROPE_DIM :].unflatten(-1, (-1, 2))
    f = freqs[positions.long()].view(rows, 1, 1, -1, 2)
    cos, sin = f[..., 0], f[..., 1]
    real, imag = pair[..., 0], pair[..., 1]
    rotated = torch.stack((real * cos + imag * sin, imag * cos - real * sin), -1)
    heads = heads.clone()
    heads[..., 512 - wo_a.ROPE_DIM :] = rotated.flatten(-2).bfloat16().float()
    x64 = heads.reshape(rows, wo_a.GROUPS, wo_a.GROUP_K).double()
    out = torch.einsum("tgk,gnk->tgn", x64, weight.double())
    return out.reshape(rows, wo_a.N_OUT).bfloat16()


def _dequantize(codes, scales, rows):
    """Values [T, 2048] and the per-element block exponent from the 128x4 swizzle."""
    g = torch.arange(wo_a.N_OUT // 32, device=codes.device)
    r = torch.arange(rows, device=codes.device)[:, None]
    off = (g // 4) * 512 + ((r % 32) * 4 + (r // 32) % 4) * 4 + g % 4
    exponent = scales[off].int().repeat_interleave(32, dim=1)
    values = codes.float() * torch.exp2(exponent.float() - 127.0)
    return values, exponent


@needs_wo_a
@pytest.mark.parametrize("rows", range(33, wo_a.MAX_ROWS + 1))
def test_wo_a_fused_matches_torch_wide(rows):
    """The kernel rounds an fp32 split-K sum to BF16 and quantizes it; the reference
    rounds the float64 product. The two BF16 rows differ by at most 1 ulp where the
    summation order matters, which moves an FP8 code by at most one step (or a block
    scale by one exponent when its amax crosses a power of two). So the bound is one
    FP8 step at the larger of the two block scales: |v| / 8 for normal codes, 2^-9 in
    code units for subnormal ones; and at least 99.9 % of codes and scale bytes match
    FlashInfer's mxfp8_quantize of the reference bit for bit."""
    flashinfer = pytest.importorskip("flashinfer")
    for seed in (0, 1):
        x, weight, freqs, positions = _wo_a_inputs(rows, seed, False, padded_heads=True)
        codes, scales = wo_a.fused_rope_wo_a(x, weight, freqs, positions)
        ref = _torch_reference(x, weight, freqs, positions)
        ref_codes, ref_scales = flashinfer.mxfp8_quantize(
            ref, is_sf_swizzled_layout=True
        )
        ref_scales = ref_scales.reshape(-1)
        got, got_exp = _dequantize(codes, scales, rows)
        want, want_exp = _dequantize(ref_codes, ref_scales, rows)
        block = torch.exp2(torch.maximum(got_exp, want_exp).float() - 127.0)
        step = torch.maximum(torch.maximum(got.abs(), want.abs()) / 8, block * 2**-9)
        excess = ((got - want).abs() - step).max().item()
        assert excess <= 0, (rows, seed, excess)
        same_codes = (codes.view(torch.uint8) == ref_codes.view(torch.uint8)).float()
        assert same_codes.mean().item() >= 0.999, (rows, seed)
        live = torch.unique(_live_scale_offsets(rows, scales.device))
        same_scales = (scales[live] == ref_scales[live]).float().mean().item()
        assert same_scales >= 0.999, (rows, seed)
        print(
            f"WO_A_TORCH rows={rows} seed={seed} codes_equal={same_codes.mean():.6f} "
            f"scales_equal={same_scales:.6f}"
        )
    torch.cuda.synchronize()


def _live_scale_offsets(rows, device):
    g = torch.arange(wo_a.N_OUT // 32, device=device)
    r = torch.arange(rows, device=device)[:, None]
    return ((g // 4) * 512 + ((r % 32) * 4 + (r // 32) % 4) * 4 + g % 4).flatten()


@needs_wo_a
@pytest.mark.parametrize("rows", [9, 12, 13, 16, 24, 47, 48])
def test_wo_a_row_tile_scales_relay_the_swizzled_ones(rows, monkeypatch):
    """Hop A's publishing wo_b input past 8 rows: the same codes, and each row's scales
    at its 8-row tile's own 128-row atom; the last tile's rows past ``rows`` read zero,
    though every allocation starts as 0xFF (a byte the GEMM reads but nobody writes)."""
    x, weight, freqs, positions = _wo_a_inputs(rows, 1, False, padded_heads=True)
    codes, scales = wo_a.fused_rope_wo_a(x, weight, freqs, positions)
    empty = torch.empty

    def poisoned(*args, **kwargs):
        out = empty(*args, **kwargs)
        return out.fill_(0xFF) if out.dtype == torch.uint8 else out

    monkeypatch.setattr(torch, "empty", poisoned)
    tiled_codes, tiled = wo_a.fused_rope_wo_a(
        x, weight, freqs, positions, row_tile_scales=True
    )
    monkeypatch.undo()
    assert torch.equal(tiled_codes.view(torch.uint8), codes.view(torch.uint8))
    groups = wo_a.N_OUT // 32
    g = torch.arange(groups, device=scales.device)
    padded = (rows + 7) // 8 * 8
    r = torch.arange(padded, device=scales.device)[:, None]
    at = ((r // 8) * (wo_a.N_OUT // 128) + g // 4) * 512 + (r % 8) * 16 + g % 4
    live = _live_scale_offsets(rows, scales.device).view(rows, groups)
    assert torch.equal(tiled[at[:rows]], scales[live])
    assert not tiled[at[rows:]].any()
    torch.cuda.synchronize()


# -------------------------------------------------------------- wq_b on poisoned memory

N, K = 8192, 1280  # wq_b at TP4: target and draft
POISON_EPS = 1e-6
# 1-8: plain swizzle; 9-48: row tiles (the target's 9-48, the draft's 10-45)
ROWS = (1, 6, 8, 9, 10, 12, 18, 30, 45, 48)
# 0xFF: NaN as an E4M3 code and as a UE8M0 scale; 0x7F: NaN code, scale 1.0
POISON = (0xFF, 0x7F)

_empty = torch.empty


@contextmanager
def _filled_empty(byte: int):
    """Every ``torch.empty`` in the block comes back filled with ``byte``."""

    def empty(*args, **kwargs):
        t = _empty(*args, **kwargs)
        t.view(-1).view(torch.uint8).fill_(byte)
        return t

    with mock.patch.object(torch, "empty", empty):
        yield


def _poison_pool(byte: int) -> None:
    """Fill and free small-pool blocks (up to 1 MB each) so the next allocations
    reuse garbage."""
    blocks = [_empty(1 << 19, dtype=torch.uint8, device="cuda") for _ in range(64)]
    for b in blocks:
        b.fill_(byte)
    del blocks


def _poison_inputs(rows):
    g = torch.Generator(device="cuda").manual_seed(rows)
    x = torch.randn(rows, K, device="cuda", generator=g).bfloat16()
    norm = (1 + 0.1 * torch.randn(K, device="cuda", generator=g)).bfloat16()
    w = torch.randn(N, K, device="cuda", generator=g) / K**0.5
    w_q, w_sf = mxfp8_ll.quantize(w.bfloat16())
    return x, norm, w_q.contiguous(), w_sf.reshape(-1).contiguous()


def _step(x, norm, w_q, w_sf, mode, byte):
    """The chain's normalize_q_lora + compute_q_b on poisoned memory."""

    def run():
        _, q, s = rmsnorm_mxfp8(
            x, norm, POISON_EPS, pdl=True, row_tiles=x.shape[0] > mxfp8_ll.MAX_ROWS
        )
        return s, mxfp8_ll.linear(q, s, w_q, w_sf)

    if mode == "pool":
        _poison_pool(byte)
        return run()
    with _filled_empty(byte):
        return run()


@needs_ll_gemm
@pytest.mark.parametrize("graph", [False, True], ids=["eager", "graph"])
@pytest.mark.parametrize("mode", ["fill", "pool"])
@pytest.mark.parametrize("rows", ROWS)
def test_wq_b_output_ignores_unwritten_bytes(rows, mode, graph):
    x, norm, w_q, w_sf = _poison_inputs(rows)
    mxfp8_ll.precompile(w_q, [rows])
    # compile the norm outside any capture
    _step(x, norm, w_q, w_sf, "fill", 0)
    torch.cuda.synchronize()
    want_s, want = (t.clone() for t in _step(x, norm, w_q, w_sf, "fill", 0))
    for byte in POISON:
        if graph:
            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g):
                s, out = _step(x, norm, w_q, w_sf, mode, byte)
            g.replay()
        else:
            s, out = _step(x, norm, w_q, w_sf, mode, byte)
        torch.cuda.synchronize()
        unwritten = int((s != want_s).sum())
        nans = int(out.isnan().sum())
        same = torch.equal(out, want)
        print(
            f"POISON rows={rows} mode={mode} graph={graph} byte=0x{byte:02X} "
            f"scale_bytes={s.numel()} unwritten={unwritten} nan_out={nans} "
            f"bitwise={'yes' if same else 'NO'}"
        )
        assert same, f"wq_b output changed with poisoned memory 0x{byte:02X}"
        # a pass proves something only if the poison reached the bytes left unwritten
        if rows > mxfp8_ll.MAX_ROWS and byte == 0xFF:
            assert unwritten > 0, "poison did not land in the unwritten scale bytes"


# ------------------------------------------------------------- FlashMLA schedules ahead

RATIOS = (0, 1, 2, 4, 128)
SWA_WIDTH = 128
EXTRA_WIDTH = {1: 512, 2: 512, 4: 512, 128: 128}


def _sched_ready() -> bool:
    return torch.cuda.is_available() and torch.cuda.get_device_capability()[0] == 10


needs_sched_ahead = pytest.mark.skipif(
    not _sched_ready() or backend is None,
    reason="needs an SM100-family GPU and SGLang's DeepSeek-V4 backend",
)


class _Core:
    """The step's attention metadata as the schedule reads it."""

    def __init__(self, rows: int, length_rows: int, seed: int):
        g = torch.Generator(device="cuda").manual_seed(seed)

        def lengths(width):
            return torch.randint(
                1, width + 1, (length_rows,), generator=g, device="cuda"
            ).to(torch.int32)

        self.low_ratios, self.present_ratios = (1, 2), (1, 2, 4, 128)
        self.swa_page_indices = torch.zeros(
            rows, SWA_WIDTH, dtype=torch.int32, device="cuda"
        )
        self.swa_topk_lengths = lengths(SWA_WIDTH)
        self._extra = {
            r: torch.zeros(rows, w, dtype=torch.int32, device="cuda")
            for r, w in EXTRA_WIDTH.items()
        }
        self._extra_lengths = {r: lengths(w) for r, w in EXTRA_WIDTH.items()}
        self._fm = {r: backend._create_flashmla_metadata() for r in RATIOS}

    def get_flashmla_metadata(self, ratio):
        return self._fm[ratio]

    def sparse_page_indices(self, ratio):
        return self._extra[ratio]

    def sparse_topk_lengths(self, ratio):
        return self._extra_lengths[ratio]


def _lazy(core: _Core, rows: int) -> dict:
    """SGLang's per-layer build, with the layer's own query shape and matching."""
    q = torch.empty(rows, 1, 64, 512, dtype=torch.bfloat16, device="cuda")

    def match(x, value):
        if x is None or x.shape[0] == rows:
            return x
        return (
            x[:rows]
            if x.shape[0] > rows
            else backend._pad_tensor_to_size(x, rows, value=value)
        )

    for r in RATIOS:
        extra = None if r == 0 else match(core.sparse_page_indices(r), -1)
        backend._maybe_precompute_flashmla_sched_meta(
            core.get_flashmla_metadata(r),
            q=q,
            indices=match(core.swa_page_indices, 0).unsqueeze(1),
            topk_length=match(core.swa_topk_lengths, 1),
            extra_indices=None if extra is None else extra.unsqueeze(1),
            extra_topk_length=None if r == 0 else match(core.sparse_topk_lengths(r), 1),
        )
    return {r: core.get_flashmla_metadata(r) for r in RATIOS}


def _view(core: _Core) -> SimpleNamespace:
    """The running forward's view (SGLang's ForwardView) over ``core``."""
    return SimpleNamespace(
        metadata=SimpleNamespace(core_attn_metadata=core),
        trtllm_attn=False,
        kv_pool=SimpleNamespace(request_window=None),
    )


def _ahead(core: _Core, rows: int, graph: bool) -> dict:
    view = _view(core)
    fb = SimpleNamespace(
        forward_mode=SimpleNamespace(is_target_verify=lambda: True),
        positions=torch.zeros(rows, dtype=torch.int64, device="cuda"),
    )
    with mock.patch.object(backend, "DSV4AttnMetadata", _Core):
        if not graph:
            sched_ahead.join(sched_ahead.start(view, fb, rows, 0))
        else:
            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g):
                sched_ahead.join(sched_ahead.start(view, fb, rows, 0))
            # replay into the captured buffers after scribbling over them
            for r in RATIOS:
                core.get_flashmla_metadata(r).tile_scheduler_metadata.fill_(-7)
            g.replay()
            torch.cuda.synchronize()
    return {r: core.get_flashmla_metadata(r) for r in RATIOS}


@needs_sched_ahead
@pytest.mark.parametrize("graph", [False, True], ids=["eager", "graph"])
@pytest.mark.parametrize(
    "rows, length_rows", [(30, 30), (36, 36), (42, 40), (48, 48), (48, 52), (12, 12)]
)
def test_schedule_ahead_is_sglangs_lazy_schedule(rows, length_rows, graph):
    want = _lazy(_Core(rows, length_rows, seed=rows), rows)
    got = _ahead(_Core(rows, length_rows, seed=rows), rows, graph)
    for r in RATIOS:
        assert torch.equal(
            got[r].tile_scheduler_metadata, want[r].tile_scheduler_metadata
        ), r
        assert torch.equal(got[r].num_splits, want[r].num_splits), r


@needs_sched_ahead
def test_rows_below_flashmla_fork_nothing():
    core = _Core(24, 24, seed=0)
    view = _view(core)
    fb = SimpleNamespace(
        forward_mode=SimpleNamespace(is_target_verify=lambda: True),
        positions=torch.zeros(24, dtype=torch.int64, device="cuda"),
    )
    with mock.patch.object(backend, "DSV4AttnMetadata", _Core):
        assert (
            sched_ahead.start(view, fb, 24, sched_ahead.first_flashmla_rows(24)) == []
        )
    assert all(
        core.get_flashmla_metadata(r).tile_scheduler_metadata is None for r in RATIOS
    )
