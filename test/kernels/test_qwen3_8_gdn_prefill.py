# SPDX-License-Identifier: Apache-2.0
"""The GDN prefill step (runtime/gdn_prefill.py) against SGLang main's stock GDN
extend on random weights at Qwen3.8's TP1 geometry (16 key heads, 48 value heads, 128
wide), on an SM10x GPU with FlashInfer's GDN prefill.

The stock side is main's ``Qwen3_5GatedDeltaNet.forward`` through
``RadixLinearAttention`` and ``GDNAttnBackend.forward_extend`` with
``FlashInferGDNKernel`` as the prefill kernel (``--linear-attn-prefill-backend
flashinfer``); the backend runs on hand-built forward metadata and the test's pools.

Bitwise: the output and every conv-state and SSM-state slot (live and track slots), for
batches that take the strip conv (>= 2048 tokens), the Triton 2D split tile, the
block-table split conv (a ragged mixed batch), the plain conv (one token), the 2D
gating (>= 512 rows), the l2norm and norm tiles, the fused state rows and the fused
track copies (aligned and unaligned rows), with the options on (the recipe) and off,
at FP16 (the recipe's ``--mamba-ssm-dtype``) and BF16 SSM states. The last test
installs FlashInfer #8854 (FlashInfer 0.7.3 only) and checks it against the stock
kernels run before the install.
"""

import dataclasses
import itertools
from types import SimpleNamespace

import pytest
import torch

from sgl_nvidia_mega.qwen3_8.runtime import gdn_prefill


def _available() -> bool:
    if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 10:
        return False
    from sglang.srt.layers.attention.linear.kernels.gdn_flashinfer import (
        is_flashinfer_gdn_prefill_available,
    )

    return is_flashinfer_gdn_prefill_available()


pytestmark = pytest.mark.skipif(
    not _available(), reason="the GDN prefill kernels need SM10x and FlashInfer"
)

H, HV, D = 16, 48, 128
Q_DIM, V_DIM = H * D, HV * D
QKV = 2 * Q_DIM + V_DIM
HIDDEN = 256
CHUNK = 64  # FlashInfer's checkpoint interval (mamba_cache_chunk_size)
LAYER = 5
BF16 = torch.bfloat16
ALL_OFF = gdn_prefill.Options(
    split_conv=False,
    conv_strip=None,
    conv_tile=None,
    gating_2d=False,
    l2norm_retile=False,
    qk_l2norm_one_launch=False,
    norm_retile=False,
    fused_state_copy=False,
    fused_track=False,
    tmem_dealloc=False,
)
# name: (extend lengths, prefix lengths, {request: tracked length in this extend})
CASES = {
    # 1494 tokens: Triton 2D split tile, 2D gating, l2norm and norm tiles; one aligned
    # (final-state) and one unaligned (checkpoint) tracked row
    "mixed": ([300, 129, 64, 1, 1000], [0, 37, 0, 5, 0], {0: 256, 4: 900}),
    # one long sequence: the strip conv
    "long": ([4096], [128], {0: 4000}),
    # a ragged mixed batch: the block-table conv
    "ragged": ([700] + [1] * 40, [0] + [11] * 40, {0: 640}),
    # one token: the plain conv on the recipe tile
    "one_token": ([1], [9], {}),
}
IMPL = {
    "mixed": "conv_split+fi+rows_fused+track_fused",
    "long": "conv_split+fi+rows_fused+track_fused",
    "ragged": "conv_split+fi+rows_fused+track_fused",
    "one_token": "conv+fi+rows_fused",
}


class _Gdn:
    """What ``Qwen3_5GatedDeltaNet.forward`` reads, with random weights."""

    def __init__(self, seed):
        from sglang.kernels.ops.attention.fla.layernorm_gated import RMSNorm
        from sglang.srt.layers.radix_linear_attention import RadixLinearAttention

        g = torch.Generator(device="cuda").manual_seed(seed)

        def randn(*shape, scale=1.0, dtype=BF16):
            return (scale * torch.randn(*shape, generator=g, device="cuda")).to(dtype)

        self.num_k_heads, self.num_v_heads = H, HV
        self.head_k_dim = self.head_v_dim = D
        self.attn_tp_size = 1
        self.w_qkvz = randn(QKV + V_DIM, HIDDEN, scale=HIDDEN**-0.5)
        self.w_ba = randn(2 * HV, HIDDEN, scale=HIDDEN**-0.5)
        self.w_out = randn(HIDDEN, V_DIM, scale=V_DIM**-0.5)
        a_log = torch.log(torch.empty(HV, device="cuda").uniform_(1, 16, generator=g))
        self.attn = RadixLinearAttention(
            layer_id=LAYER,
            num_q_heads=H,
            num_k_heads=H,
            num_v_heads=HV,
            head_q_dim=D,
            head_k_dim=D,
            head_v_dim=D,
            conv_weights=randn(QKV, 4, scale=0.5),
            bias=None,
            activation="silu",
            A_log=a_log,
            dt_bias=randn(HV, scale=0.5),
        )
        self.norm = RMSNorm(
            D, eps=1e-6, device="cuda", dtype=BF16, activation="sigmoid"
        )
        with torch.no_grad():
            self.norm.weight.copy_(1 + randn(D, scale=0.1))

    def _forward_input_proj(self, x):
        return x @ self.w_qkvz.T, x @ self.w_ba.T

    def out_proj(self, x):
        return x @ self.w_out.T, None


class _Case:
    """One batch: the pools, the metadata and the input."""

    def __init__(self, name, ssm_dtype, seed=0):
        from sglang.srt.layers.attention.linear.gdn_backend import (
            GDNAttnBackend,
            GDNKernelDispatcher,
        )
        from sglang.srt.layers.attention.linear.kernels.gdn_flashinfer import (
            FlashInferGDNKernel,
        )
        from sglang.srt.layers.attention.mamba.mamba2_metadata import (
            ForwardMetadata,
        )
        from sglang.srt.model_executor.forward_mode import ForwardMode

        lens, prefix, tracked = CASES[name]
        g = torch.Generator(device="cuda").manual_seed(seed)
        n = len(lens)
        slots = 2 * n + 3
        perm = torch.randperm(slots - 1, generator=g, device="cuda") + 1
        live, track_slots = perm[:n].int(), perm[n : 2 * n].int()
        self.conv = (0.5 * torch.randn(slots, QKV, 3, generator=g, device="cuda")).to(
            BF16
        )
        self.ssm = (0.1 * torch.randn(slots, HV, D, D, generator=g, device="cuda")).to(
            ssm_dtype
        )
        self.x = torch.randn(sum(lens), HIDDEN, generator=g, device="cuda").to(BF16)
        starts = [0, *itertools.accumulate(lens)]
        md = ForwardMetadata(
            query_start_loc=torch.tensor(starts, dtype=torch.int32, device="cuda"),
            mamba_cache_indices=live,
        )
        if tracked:
            self._track(md, lens, starts, live, track_slots, tracked)
        mode = ForwardMode.MIXED if name == "ragged" else ForwardMode.EXTEND
        self.fb = SimpleNamespace(
            forward_mode=mode,
            logical_forward_mode=mode,
            extend_seq_lens_cpu=list(lens),
            extend_prefix_lens=torch.tensor(prefix, dtype=torch.int32, device="cuda"),
            global_num_token_non_padded_cpu=None,
            tbo_parent_token_range=None,
        )
        backend = object.__new__(GDNAttnBackend)
        backend.device = "cuda"
        backend.mis_metadata = None
        backend.prefill_graph_metadata = None
        backend._use_strided_target_verify_qkv = False
        dispatcher = object.__new__(GDNKernelDispatcher)
        dispatcher.extend_kernel = FlashInferGDNKernel()
        backend.kernel_dispatcher = dispatcher
        backend.forward_metadata = md
        GDNAttnBackend._init_prefill_conv_block_table(backend, self.fb)
        assert (md.conv_block_table is not None) == (name == "ragged")
        self.backend = backend
        self.gdn = _Gdn(seed + 1)

    @staticmethod
    def _track(md, lens, starts, live, track_slots, tracked):
        """The tracked rows' metadata as main's backend builds it for FlashInfer."""
        counts = [length // CHUNK for length in lens]
        cu = [0, *itertools.accumulate(counts)]
        rows = sorted(tracked)
        windows, h_src, h_dst, f_src, f_dst = [], [], [], [], []
        for i in rows:
            rel = tracked[i]
            aligned = rel // CHUNK * CHUNK
            windows.append([max(starts[i] + aligned - 3 + j, 0) for j in range(3)])
            if rel % CHUNK == 0:
                f_src.append(int(live[i]))
                f_dst.append(int(track_slots[i]))
            else:
                h_src.append(cu[i] + rel // CHUNK - 1)
                h_dst.append(int(track_slots[i]))

        def t(v):
            return torch.tensor(v, dtype=torch.int64, device="cuda")

        md.has_mamba_track_mask = True
        md.track_conv_indices = t(windows)
        md.conv_states_mask_indices = track_slots[rows].long()
        md.track_ssm_h_src, md.track_ssm_h_dst = t(h_src), t(h_dst)
        md.track_ssm_final_src, md.track_ssm_final_dst = t(f_src), t(f_dst)
        if h_src:  # maybe_build_flashinfer_checkpoint_plan plans only for h rows
            md.state_checkpoint_cu_starts = t(cu)
            md.num_state_checkpoints = cu[-1]
            md.state_checkpoint_every_n_tokens = CHUNK

    def run(self, monkeypatch, step):
        """step(conv, ssm) on fresh copies of the pools; (out, conv, ssm)."""
        conv, ssm = self.conv.clone(), self.ssm.clone()
        cache = SimpleNamespace(conv=[conv], temporal=ssm)
        self.backend.state_layer_cache = lambda layer_id: cache
        backend = self.backend

        def forward(layer, forward_batch, mixed_qkv, a, b, **kwargs):
            return backend.forward_extend(
                layer, forward_batch, mixed_qkv, a, b, **kwargs
            )

        shim = SimpleNamespace(linear_attn_backend=backend, forward=forward)
        monkeypatch.setattr(
            "sglang.srt.layers.radix_linear_attention.get_attn_backend", lambda: shim
        )
        monkeypatch.setattr(
            "sglang.srt.model_executor.forward_context.get_attn_backend", lambda: shim
        )
        monkeypatch.setattr(
            "sglang.srt.models.qwen4_exp._shards_rows_over_cp", lambda fb: False
        )
        with torch.no_grad():
            out = step()
        torch.cuda.synchronize()
        return out, conv, ssm

    def stock(self, monkeypatch):
        from sglang.srt.models.qwen3_5 import Qwen3_5GatedDeltaNet

        return self.run(
            monkeypatch, lambda: Qwen3_5GatedDeltaNet.forward(self.gdn, self.x, self.fb)
        )

    def plugin(self, monkeypatch, options):
        ran = []
        st = SimpleNamespace(fb=self.fb, ran=lambda op, impl: ran.append((op, impl)))
        executor = SimpleNamespace(gdn_prefill=options)
        layer = SimpleNamespace(linear_attn=self.gdn)
        out = self.run(
            monkeypatch, lambda: gdn_prefill.extend(executor, layer, self.x, st)
        )
        return out, ran


@pytest.fixture(scope="module", autouse=True)
def strip_built():
    from sgl_nvidia_mega.qwen3_8.kernels.ops.mamba.gdn_prefill_conv import (
        warmup_prefill_conv_strip,
    )

    warmup_prefill_conv_strip(gdn_prefill.DEFAULT.strip_cfg())


def _assert_same(expected, actual):
    (out, conv, ssm), (out2, conv2, ssm2) = expected, actual
    assert out2 is not None
    assert bool(torch.isfinite(out).all())
    assert torch.equal(out2, out)
    assert torch.equal(conv2, conv)
    assert torch.equal(ssm2, ssm)


@pytest.mark.parametrize("ssm_dtype", [torch.float16, BF16], ids=["fp16", "bf16"])
@pytest.mark.parametrize("name", list(CASES))
def test_matches_the_stock_extend(monkeypatch, name, ssm_dtype):
    assert not gdn_prefill._tmem_fixed  # the last test installs it
    case = _Case(name, ssm_dtype)
    expected = case.stock(monkeypatch)
    actual, ran = case.plugin(monkeypatch, gdn_prefill.DEFAULT)
    _assert_same(expected, actual)
    assert ran == [("gdn", IMPL[name])]


@pytest.mark.parametrize("name", ["mixed", "one_token"])
def test_stock_launches_match_too(monkeypatch, name):
    case = _Case(name, torch.float16)
    expected = case.stock(monkeypatch)
    actual, ran = case.plugin(monkeypatch, ALL_OFF)
    _assert_same(expected, actual)
    track = "+track" if CASES[name][2] else ""
    assert ran == [("gdn", f"conv+fi+rows{track}")]


def test_split_conv_without_strip(monkeypatch):
    case = _Case("long", torch.float16)
    expected = case.stock(monkeypatch)
    options = dataclasses.replace(gdn_prefill.DEFAULT, conv_strip=None)
    actual, _ = case.plugin(monkeypatch, options)
    _assert_same(expected, actual)


def test_zz_tmem_dealloc_fix(monkeypatch):
    """FlashInfer #8854 (installed process-wide, so last) against the stock kernels."""
    try:
        from sgl_nvidia_mega.qwen3_8.kernels.ops.mamba.gdn_prefill_tmem_dealloc import (
            install,
        )
    except ImportError as e:
        pytest.skip(f"FlashInfer is not v0.7.3: {e}")
    cases = [_Case(name, torch.float16) for name in ("mixed", "long")]
    expected = [case.stock(monkeypatch) for case in cases]
    monkeypatch.setattr(gdn_prefill, "_tmem_fixed", install())
    for case, before in zip(cases, expected):
        actual, ran = case.plugin(monkeypatch, gdn_prefill.DEFAULT)
        _assert_same(before, actual)
        assert ran == [("gdn", "conv_split+fi_tmem+rows_fused+track_fused")]
