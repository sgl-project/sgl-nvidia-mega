# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the SGLang project
"""Draft MLA sublayer on megakernel blocks (stage 1, M1b): fused_a (replicated 7168 ->
2112 split-K) -> DraftMlaQuery (q_a norm, Q_B, W_UK, RoPE, FP8 query; kv_a norm + rotated
k_pe into the FP8 cache slot of each block row) -> the attention core (the caller's
``core``: the SWA-window core over the compact cache) -> DraftMlaValue (W_UV, no gate)
-> the [rows, heads * 128] value the o-projection hop reads."""

from functools import cache

import cutlass
import cutlass.cute as cute
import torch
from cutlass.cute.runtime import make_fake_stream

from ..k3.common.host import fake, layout, prepare_shared_carveout, stream, to_cute
from ..k3.gemm.projection import PreparedProjection
from .mla import DraftMlaQuery, DraftMlaValue

PROJECTED = 2112  # q_a 1536 | kv_a 512 | k_pe 64


@cache
def compile_query(tokens, heads, specs, device_index):
    """``specs``: layouts of (q_a norm, kv_a norm, q_b, w_uk, cache, slots, q_scale_inv,
    k_scale_inv, positions, cos_sin)."""
    qnorm, kvnorm, qb, uk, kv, slots, qinv, kinv, positions, cos_sin = (
        fake(shape, stride, dtype, 4 if dtype in (torch.float32, torch.int32) else 16)
        if dtype != torch.int64
        else fake(shape, stride, dtype, 8)
        for shape, stride, dtype in specs
    )
    projected = fake((tokens, PROJECTED), (PROJECTED, 1))
    query = fake((tokens, heads, 576), (heads * 576, 576, 1), torch.uint8)
    args = (projected, qnorm, kvnorm, qb, uk, kv, slots, qinv, kinv)
    args += (cutlass.Float32(1e-6), query, positions, cos_sin)
    with torch.accelerator.device_index(device_index):
        code = cute.compile(DraftMlaQuery(heads, tokens), *args, make_fake_stream())
        return prepare_shared_carveout(code, device_index, 100)


@cache
def compile_value(tokens, heads, uv_spec, q):
    shape, stride, dtype = uv_spec
    latent = fake((tokens, heads, 512))
    uv = fake(shape, stride, dtype)
    lengths = fake((tokens // q,), (1,), torch.int32, 4)
    output = fake((tokens, heads * 128), (heads * 128, 1))
    # Early trigger: the o-projection hop's weight preload overlaps the core. Safe for
    # its residual prefetch: ConvPrepare (or the folded conv projection) upstream
    # triggers only after its own PDL wait, so the previous hop has completed.
    value = DraftMlaValue(heads, tokens, q, early_trigger=True)
    return cute.compile(value, latent, uv, lengths, output, make_fake_stream())


class DraftAttention:
    """One draft layer's MLA sublayer for ``shapes.tokens`` rows (``shapes.rows`` per
    request). ``run(x, positions, cos_sin, cache, slots, core, out)``: ``cache`` is this
    layer's FP8 pool [pages, page, 576], ``slots`` the block rows' physical slots
    [tokens] int64, ``core(query_fp8 [tokens, heads, 576], latent [tokens, heads, 512])``
    the attention core."""

    def __init__(self, w, shapes, tactic, device, *, q_scale=1.0, k_scale=1.0):
        heads, self.shapes, self.device = shapes.heads, shapes, device
        bf16 = lambda t: t.detach().to(torch.bfloat16).contiguous()
        self.q_norm, self.kv_norm, self.q_b = (
            bf16(w.q_a_norm),
            bf16(w.kv_a_norm),
            bf16(w.q_b),
        )
        if self.q_b.shape != (heads * 192, 1536):
            raise ValueError(f"q_b {tuple(self.q_b.shape)}")
        kv_b = bf16(w.kv_b).view(heads, 256, 512)
        self.w_uk = kv_b[:, :128].contiguous()  # [heads, nope 128, latent 512]
        self.w_uv = (
            kv_b[:, 128:].contiguous().transpose(1, 2)
        )  # [heads, 512, v], latent-contiguous
        self.fused_a = PreparedProjection(
            bf16(w.fused_a), {shapes.tokens: tactic}
        ).prepare()
        f32 = dict(dtype=torch.float32, device=device)
        self.q_scale_inv = torch.tensor([1.0 / q_scale], **f32)
        self.k_scale_inv = torch.tensor([1.0 / k_scale], **f32)
        tokens = shapes.tokens
        self.projected = torch.empty(
            tokens, PROJECTED, dtype=torch.bfloat16, device=device
        )
        self.query = torch.empty(tokens, heads, 576, dtype=torch.uint8, device=device)
        self.latent = torch.empty(
            tokens, heads, 512, dtype=torch.bfloat16, device=device
        )
        self.lengths = torch.ones(shapes.requests, dtype=torch.int32, device=device)
        self.value = compile_value(tokens, heads, layout(self.w_uv), shapes.rows)
        self.views = tuple(
            to_cute(t, 4 if t.dtype == torch.float32 else 16)
            for t in (self.q_norm, self.kv_norm, self.q_b, self.w_uk)
        )
        self.scales = tuple(to_cute(t, 4) for t in (self.q_scale_inv, self.k_scale_inv))
        self.eps = cutlass.Float32(shapes.eps)
        self.codes = {}

    def _query_code(self, kv, slots, positions, cos_sin):
        tensors = (self.q_norm, self.kv_norm, self.q_b, self.w_uk, kv, slots)
        tensors += (self.q_scale_inv, self.k_scale_inv, positions, cos_sin)
        specs = tuple(layout(t) for t in tensors)
        if specs not in self.codes:
            index = self.device.index if self.device.index is not None else 0
            self.codes[specs] = compile_query(
                self.shapes.tokens, self.shapes.heads, specs, index
            )
        return self.codes[specs]

    def run(self, x, positions, cos_sin, cache, slots, core, out, mark=None):
        """``mark(tag)``: optional stamps after fused_a / query / core / value."""
        mark = mark or (lambda tag: None)
        s = stream(self.device)
        self.fused_a.run(x, self.projected)
        mark("fused_a")
        kv = cache.view(torch.uint8)
        code = self._query_code(kv, slots, positions, cos_sin)
        qnorm, kvnorm, qb, uk = self.views
        args = (to_cute(self.projected), qnorm, kvnorm, qb, uk, to_cute(kv))
        args += (to_cute(slots, 8), *self.scales, self.eps, to_cute(self.query))
        args += (to_cute(positions, 8), to_cute(cos_sin, 4))
        code(*args, s)
        mark("query")
        core(self.query, self.latent)
        mark("core")
        uv = to_cute(self.w_uv)
        self.value(to_cute(self.latent), uv, to_cute(self.lengths, 4), to_cute(out), s)
        mark("value")
        return out
