# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Slot attn_core_lean (muLLM's one-launch sparse attention core on SGLang's v41_new pool)
on one SM100-family GPU (GB300), 8 local heads, at the served batch-1 widths (the target
verify's 6 rows; the DSpark draft's 5 rows on a 192-slot window): against an fp64
reference on the bytes it reads and on the true latents, with SGLang's padded (-1)
indices and short lengths, the window-only entry, the pre-wait window indices (bitwise
with the post-wait read), the output's inverse RoPE, and graph replay (bitwise with
eager). ``Case`` also feeds benchmark/bench_deepseek_v41_core_lean.py.

Numerics change: the kernel is not bitwise with swap-AB (different KV encodings, P in
BF16, muLLM's split and merge order); the test bounds its error against fp64.
"""

import pytest
import torch

from sgl_nvidia_mega.deepseek_v41.sm103.attention import core_lean


def _ready() -> bool:
    if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 10:
        return False
    try:
        from triton.experimental import gluon  # noqa: F401
    except ImportError:
        return False
    return True


pytestmark = pytest.mark.skipif(not _ready(), reason="needs an SM100-family GPU")

HEADS = 8
E2M1 = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0])


def _pow2_ceil_scale(amax, fmax, floor):
    ratio = torch.clamp(amax / fmax, min=floor)
    return torch.exp2(torch.ceil(torch.log2(ratio)))


def _encode_v41(k):
    """[n, 512] float -> [n, 528] uint8 (E4M3 + UE8M0 per 32) and its dequantized values."""
    n = k.shape[0]
    x = k.float().view(n, 16, 32)
    scale = _pow2_ceil_scale(x.abs().amax(-1), 448.0, 1e-4)
    codes = (x / scale[..., None]).to(torch.float8_e4m3fn)
    deq = (codes.float() * scale[..., None]).view(n, 512)
    exps = (torch.log2(scale).round() + 127).to(torch.uint8)
    rows = torch.cat([codes.view(torch.uint8).view(n, 512), exps], dim=1)
    return rows, deq


def _encode_fp4(k):
    """[n, 512] float -> [n, 288] uint8 (E2M1 + E4M3 per 16) and its dequantized values."""
    n = k.shape[0]
    x = k.float().view(n, 32, 16)
    scale = torch.clamp(x.abs().amax(-1) / 6.0, 2.0**-9, 448.0).to(torch.float8_e4m3fn)
    s = scale.float()
    y = (x / s[..., None]).clamp(-6, 6)
    mags = E2M1.to(k.device)
    code = (y.abs()[..., None] - mags).abs().argmin(-1).to(torch.uint8)
    code = code | (torch.signbit(y).to(torch.uint8) << 3)
    deq = mags[(code & 7).long()] * torch.where(code >= 8, -1.0, 1.0) * s[..., None]
    code = code.view(n, 512)
    packed = code[:, 0::2] | (code[:, 1::2] << 4)
    rows = torch.cat([packed, scale.view(torch.uint8)], dim=1)
    return rows, deq.view(n, 512)


def _encode_v4(k):
    """[n, 512] float -> 576 data bytes + 8 scale bytes per token (swap-AB's V4 layout)."""
    n = k.shape[0]
    nope = k[:, :448].float().view(n, 7, 64)
    scale = _pow2_ceil_scale(nope.abs().amax(-1), 448.0, 1e-30)
    codes = (nope / scale[..., None]).to(torch.float8_e4m3fn)
    rope = k[:, 448:].to(torch.bfloat16)
    data = torch.cat([codes.view(torch.uint8).view(n, 448), rope.view(torch.uint8)], 1)
    exps = torch.zeros(n, 8, dtype=torch.uint8, device=k.device)
    exps[:, :7] = (torch.log2(scale).round() + 127).to(torch.uint8)
    deq = torch.cat([(codes.float() * scale[..., None]).view(n, 448), rope.float()], 1)
    return data, exps, deq


def _pages(rows, page, pad=256):
    """Token rows [pages * page, w] -> a paged pool view [pages, page, 1, w] whose pages
    sit ``pad`` bytes apart past their tokens (SGLang's views keep a row stride)."""
    n, w = rows.shape
    pages = n // page
    stride = page * w + pad
    raw = torch.full((pages, stride), 0xAB, dtype=torch.uint8, device=rows.device)
    raw[:, : page * w].copy_(rows.view(pages, page * w))
    return raw.as_strided((pages, page, 1, w), (stride, w, w, 1))


def _pages_v4(data, exps, page):
    n = data.shape[0]
    pages = n // page
    stride = page * 584 + 256
    raw = torch.full((pages, stride), 0x7F, dtype=torch.uint8, device=data.device)
    raw[:, : page * 576].copy_(data.view(pages, page * 576))
    raw[:, page * 576 : page * 584].copy_(exps.view(pages, page * 8))
    return raw.as_strided((pages, page, 1, 584), (stride, 584, 584, 1))


class Case:
    def __init__(
        self, b, seed, *, window_only=False, extra_tokens=16 * 64, holes=True, width=128
    ):
        g = torch.Generator(device="cuda").manual_seed(seed)
        dev = "cuda"
        self.b = b
        self.window_only = window_only
        self.width = width
        wt, et = 64 * 11, -(-extra_tokens // 16) * 16
        self.k_win = torch.randn(wt, 512, device=dev, generator=g) * 2
        self.k_ext = torch.randn(et, 512, device=dev, generator=g) * 2
        win41, self.win41 = _encode_v41(self.k_win)
        self.win41_rows = win41
        ext4, self.ext4 = _encode_fp4(self.k_ext)
        wd, we, self.win_v4 = _encode_v4(self.k_win)
        ed, ee, self.ext_v4 = _encode_v4(self.k_ext)
        self.swa41, self.ext41 = _pages(win41, 64), _pages(ext4, 16)
        self.swa_v4, self.extra_v4 = _pages_v4(wd, we, 64), _pages_v4(ed, ee, 16)
        self.q = (torch.randn(b, HEADS, 512, device=dev, generator=g) * 0.5).to(
            torch.bfloat16
        )
        self.ids = torch.randint(
            0, wt, (b, 1, width), device=dev, dtype=torch.int32, generator=g
        )
        self.lengths = torch.randint(
            1, width + 1, (b,), device=dev, dtype=torch.int32, generator=g
        )
        self.lengths[0] = width
        self.ei = torch.randint(
            0, et, (b, 1, 512), device=dev, dtype=torch.int32, generator=g
        )
        if holes:
            self.ei[:, :, -7:] = -1
            self.ei[:, :, 100:140] = -1  # a split with masked keys inside the length
        self.el = torch.randint(
            1, 513, (b,), device=dev, dtype=torch.int32, generator=g
        )
        self.el[-1] = 512
        self.sink = torch.randn(HEADS, device=dev, generator=g)
        angles = torch.rand(70000, 32, device=dev, generator=g) * 6.3
        self.freqs = (
            torch.view_as_real(torch.polar(torch.ones_like(angles), angles))
            .flatten(-2)
            .contiguous()
        )
        self.positions = torch.randint(
            70000, (b,), device=dev, dtype=torch.int64, generator=g
        )

    def caches(self):
        if self.window_only:
            return core_lean.Caches(
                self.swa41, self.ids, self.lengths, None, None, None
            )
        return core_lean.Caches(
            self.swa41, self.ids, self.lengths, self.ext41, self.ei, self.el
        )

    def lean(self, early=False, positions=None, caches=None):
        pos = self.positions if positions is None else positions
        return core_lean.core_lean(
            self.q,
            self.freqs,
            pos,
            self.sink,
            self.caches() if caches is None else caches,
            512**-0.5,
            early=early,
        )

    def reference(self, keys_win, keys_ext):
        """fp64 attention of the BF16 query (roped in fp64) over the given key values."""
        out = torch.empty(self.b, HEADS, 512, dtype=torch.float64, device="cuda")
        cs = self.freqs[self.positions].double().view(self.b, 32, 2)
        for r in range(self.b):
            q = self.q[r].double().clone()
            c, s = cs[r, :, 0], cs[r, :, 1]
            x0, x1 = q[:, 448::2].clone(), q[:, 449::2].clone()
            q[:, 448::2], q[:, 449::2] = x0 * c - x1 * s, x1 * c + x0 * s
            n = min(self.width, int(self.lengths[r]))
            keys = [keys_win[self.ids[r, 0, :n].long()].double()]
            if not self.window_only:
                m = min(512, int(self.el[r]))
                e = self.ei[r, 0, :m].long()
                keys.append(keys_ext[e[e >= 0]].double())
            k = torch.cat(keys)
            logits = q @ k.T * 512**-0.5
            top = torch.maximum(logits.max(-1).values, self.sink.double())
            p = torch.exp(logits - top[:, None])
            den = p.sum(-1) + torch.exp(self.sink.double() - top)
            o = (p @ k) / den[:, None]
            y0, y1 = o[:, 448::2].clone(), o[:, 449::2].clone()
            o[:, 448::2], o[:, 449::2] = y0 * c + y1 * s, y1 * c - y0 * s
            out[r] = o
        return out


def _rel_rms(a, b):
    a, b = a.double(), b.double()
    return float(((a - b).pow(2).mean() / b.pow(2).mean()).sqrt())


def _bitwise(a, b):
    assert torch.equal(a.view(torch.int16), b.view(torch.int16)), (
        f"{(a.view(torch.int16) != b.view(torch.int16)).sum().item()} of {a.numel()} differ"
    )


@pytest.mark.parametrize(
    "b, window_only, width", [(6, False, 128), (6, True, 128), (5, True, 192)]
)
def test_core_lean_against_fp64(b, window_only, width):
    for seed in range(3):
        c = Case(b, 100 * b + seed, window_only=window_only, width=width)
        got = c.lean().float()
        assert torch.isfinite(got).all()
        # the kernel's own arithmetic (BF16 P, BF16 output) on the bytes it reads
        own = _rel_rms(got, c.reference(c.win41, c.ext4))
        assert own < 4e-3, own
        if width == 128:
            # against the true latents the KV encoding dominates: on these Gaussian
            # latents FP4 compressed rows carry ~10 % (the model's compressed latent is
            # FP4-rounded already, so v41_fp4_new stores it exactly); the window's E4M3
            # rows ~3.5 %, like swap-AB's V4
            true_lean = _rel_rms(got, c.reference(c.k_win, c.k_ext))
            assert true_lean < (0.06 if window_only else 0.15), true_lean
        # the pre-wait window indices: bitwise with the post-wait read
        _bitwise(c.lean(early=True), c.lean())


def test_core_lean_graph_replay():
    cases = [Case(6, 50 + i, window_only=i == 2) for i in range(3)]
    want = [x.lean(early=True) for x in cases]
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for x in cases:
            x.lean(early=True)
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        outs = [x.lean(early=True) for x in cases]
    for _ in range(5):
        for o in outs:
            o.zero_()
        graph.replay()
        torch.cuda.synchronize()
        for o, w in zip(outs, want):
            _bitwise(o, w)


def test_core_lean_reason():
    c = Case(6, 1)
    assert core_lean.reason(6, 8, c.caches()) == ""
    assert core_lean.reason(9, 8, c.caches())
    assert core_lean.reason(6, 16, c.caches())
    v4 = core_lean.Caches(c.swa_v4, c.ids, c.lengths, c.extra_v4, c.ei, c.el)
    assert "528" in core_lean.reason(6, 8, v4)
    wide = core_lean.Caches(
        c.swa41,
        c.ids,
        c.lengths,
        c.ext41,
        torch.zeros(6, 1, 1024, dtype=torch.int32, device="cuda"),
        c.el,
    )
    assert "1024" in core_lean.reason(6, 8, wide)


def test_identity_rope_is_exact():
    """The fused wo_a's rotation by the identity leaves its input as it is."""
    freqs, positions = core_lean.identity_rope(6, torch.device("cuda"))
    assert positions.shape == (6,) and int(positions.abs().sum()) == 0
    assert torch.equal(freqs[0, 0::2], torch.ones(32, device="cuda"))
    assert torch.equal(freqs[0, 1::2], torch.zeros(32, device="cuda"))
