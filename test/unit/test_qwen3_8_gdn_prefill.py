# SPDX-License-Identifier: Apache-2.0
"""The Qwen3.8 executor's GDN prefill step on CPU: the kernels each step runs and in
which order, the stock fallbacks, the options and the build refusals. SGLang's
backend and model modules are replaced by fakes and the kernel launches by spies, so
no GPU, no kernel build and no model weights are needed."""

import enum
import logging
import sys
import types
import unittest
from types import SimpleNamespace
from unittest import mock

import torch

from sgl_nvidia_mega.qwen3_8.kernels.ops.attention.linear import (
    gdn_prefill_flashinfer as fi_mod,
)
from sgl_nvidia_mega.qwen3_8.kernels.ops.attention.linear import (
    gdn_prefill_gating as gating_mod,
)
from sgl_nvidia_mega.qwen3_8.kernels.ops.attention.linear import (
    gdn_prefill_norm as norm_mod,
)
from sgl_nvidia_mega.qwen3_8.kernels.ops.mamba import gdn_prefill_conv as conv_mod
from sgl_nvidia_mega.qwen3_8.kernels.ops.mamba import gdn_prefill_cp_tail as tail_mod
from sgl_nvidia_mega.qwen3_8.kernels.ops.mamba import gdn_prefill_state_glue as glue_mod
from sgl_nvidia_mega.qwen3_8.runtime import gdn_prefill as gp
from sgl_nvidia_mega.qwen3_8.runtime import prefill_graph as pg

# A small GDN layer of Qwen3.8's shape: 2 key heads, 6 value heads (ratio 3), 128 wide
H, HV, D = 2, 6, 128
Q_DIM, V_DIM = H * D, HV * D
QKV = 2 * Q_DIM + V_DIM
HIDDEN = 64
SLOTS = 9


class _Mode(enum.Enum):
    EXTEND = 1
    MIXED = 2
    DECODE = 3
    TARGET_VERIFY = 4
    DRAFT_EXTEND_V2 = 5

    def is_extend_or_draft_extend_or_mixed(self):
        return self in (_Mode.EXTEND, _Mode.MIXED)

    def is_extend_without_speculative(self):
        return self in (_Mode.EXTEND, _Mode.MIXED)


class _FlashInferKernel:  # the fake FlashInferGDNKernel
    use_state_pool = True
    _prefill_needs_fp32_state = False


class _TritonKernel:
    pass


class _Linear:  # the fake Qwen4ExpLinearDecoderLayer
    def __init__(self, layer_id, gdn):
        self.layer_id, self.linear_attn = layer_id, gdn


_STATE = {}


def _install(test):
    """SGLang's modules the step imports, as fakes, restored after the test."""
    _STATE.update(breakable=False, full_graph=False, cp=False, backend=None, fi=True)
    fakes = {
        "sglang.srt.model_executor.forward_context": dict(
            get_attn_backend=lambda: _STATE["backend"]
        ),
        "sglang.srt.model_executor.runner_backend_utils.breakable_cuda_graph": dict(
            is_in_breakable_cuda_graph=lambda: _STATE["breakable"]
        ),
        "sglang.srt.model_executor.runner_utils.prefill_graph": dict(
            is_in_full_prefill_graph=lambda: _STATE["full_graph"]
        ),
        "sglang.srt.models.qwen4_exp": dict(
            _shards_rows_over_cp=lambda fb: _STATE["cp"],
            Qwen4ExpLinearDecoderLayer=_Linear,
        ),
        "sglang.srt.models.qwen3_5": dict(_GDN_FUSED_QKVZBA_RATIOS=(1, 2, 3, 4)),
        "sglang.srt.layers.attention.linear.kernels.gdn_flashinfer": dict(
            FlashInferGDNKernel=_FlashInferKernel,
            is_flashinfer_gdn_prefill_available=lambda: _STATE["fi"],
        ),
    }
    for name, attrs in fakes.items():
        module = types.ModuleType(name)
        module.__dict__.update(attrs)
        old = sys.modules.get(name)
        sys.modules[name] = module
        test.addCleanup(_restore, name, old)
    tmem = gp._tmem_fixed
    test.addCleanup(setattr, gp, "_tmem_fixed", tmem)
    gp._tmem_fixed = False


def _restore(name, old):
    if old is None:
        sys.modules.pop(name, None)
    else:
        sys.modules[name] = old


class _Spies:
    """The kernel launches the step makes, in order, with what they were given."""

    def __init__(self, test, *, snapshot=True, state_rows=True):
        self.calls = []
        self.snapshot, self.state_rows = snapshot, state_rows
        for module, name, fn in (
            (conv_mod, "causal_conv1d_fn_split_qkv", self.split_conv),
            (conv_mod, "causal_conv1d_fn", self.conv),
            (gating_mod, "fused_gdn_gating", self.gating),
            (fi_mod, "extend", self.flashinfer),
            (norm_mod, "_layer_norm_fwd", self.norm),
            (glue_mod, "gdn_track_conv_snapshot", self.track_conv),
            (glue_mod, "gdn_track_state_rows", self.track_states),
        ):
            patcher = mock.patch.object(module, name, fn)
            patcher.start()
            test.addCleanup(patcher.stop)

    def names(self):
        return [c[0] for c in self.calls]

    def kwargs(self, name):
        return next(c[1] for c in self.calls if c[0] == name)

    def split_conv(self, x, weight, bias, **kw):
        self.calls.append(("split_conv", dict(kw, x=x, weight=weight, bias=bias)))
        dim, total = x.shape
        q_dim, k_dim = kw["q_dim"], kw["k_dim"]
        out = torch.arange(dim * total, dtype=torch.float32)
        return (
            out[: total * q_dim].view(total, q_dim),
            out[total * q_dim : total * (q_dim + k_dim)].view(total, k_dim),
            out[total * (q_dim + k_dim) :].view(total, dim - q_dim - k_dim),
        )

    def conv(self, x, weight, bias, **kw):
        self.calls.append(("conv", dict(kw, x=x)))
        return torch.zeros(x.shape[0], x.shape[1])

    def gating(self, A_log, a, b, dt_bias, **kw):
        self.calls.append(("gating", dict(kw, a=a, b=b)))
        return torch.zeros(1, a.shape[0], HV), torch.zeros(1, a.shape[0], HV)

    def flashinfer(self, kernel, q, k, v, g, beta, **kw):
        self.calls.append(("flashinfer", dict(kw, kernel=kernel, q=q, k=k, v=v)))
        h = (
            torch.zeros(1, kw["num_state_checkpoints"], HV, D, D)
            if kw["num_state_checkpoints"]
            else None
        )
        return torch.ones(1, q.shape[1], HV, D), None, h

    def norm(self, x, weight, bias, eps, **kw):
        self.calls.append(("norm", dict(kw, x=x, weight=weight, bias=bias, eps=eps)))
        return x * 2, None, None

    def track_conv(self, **kw):
        self.calls.append(("track_conv", kw))
        return self.snapshot

    def track_states(self, **kw):
        self.calls.append(("track_states", kw))
        return self.state_rows


def _gdn(tp=1, head_dim=D, num_v_heads=HV, norm=None, out_proj_calls=None):
    from sglang.kernels.ops.attention.fla.layernorm_gated import RMSNorm

    attn = SimpleNamespace(
        layer_id=3,
        q_dim=Q_DIM,
        k_dim=Q_DIM,
        v_dim=V_DIM,
        num_q_heads=H,
        num_k_heads=H,
        num_v_heads=HV,
        head_q_dim=D,
        head_k_dim=D,
        head_v_dim=D,
        conv_weights=torch.zeros(QKV, 4),
        bias=None,
        activation="silu",
        A_log=torch.zeros(HV),
        dt_bias=torch.zeros(HV),
    )
    calls = out_proj_calls if out_proj_calls is not None else []

    def input_proj(x):
        rows = x.shape[0]
        qkvz = torch.arange(rows * (QKV + V_DIM), dtype=torch.float32)
        return qkvz.view(rows, QKV + V_DIM), torch.zeros(rows, 2 * HV)

    def out_proj(x):
        calls.append(x)
        return x.sum(dim=-1, keepdim=True), None

    return SimpleNamespace(
        attn=attn,
        attn_tp_size=tp,
        num_k_heads=H,
        num_v_heads=num_v_heads,
        head_k_dim=head_dim,
        head_v_dim=head_dim,
        _forward_input_proj=input_proj,
        norm=norm if norm is not None else RMSNorm(D, eps=1e-6, activation="sigmoid"),
        out_proj=out_proj,
    )


def _metadata(lens, *, track=False):
    starts = torch.tensor(
        [0, *torch.tensor(lens).cumsum(0).tolist()], dtype=torch.int32
    )
    md = SimpleNamespace(
        query_start_loc=starts,
        mamba_cache_indices=torch.arange(1, len(lens) + 1, dtype=torch.int32),
        conv_block_table=None,
        has_mamba_track_mask=track,
        track_conv_indices=None,
        conv_states_mask_indices=None,
        track_ssm_h_src=None,
        track_ssm_h_dst=None,
        track_ssm_final_src=None,
        track_ssm_final_dst=None,
        track_ssm_recompute_dst=None,
        state_checkpoint_cu_starts=None,
        num_state_checkpoints=0,
        state_checkpoint_every_n_tokens=0,
    )
    if track:
        # one tracked row: the conv window of tokens 1..3 into track slot 7, the
        # checkpoint row 0 into slot 8
        md.track_conv_indices = torch.tensor([[1, 2, 3]])
        md.conv_states_mask_indices = torch.tensor([7])
        md.track_ssm_h_src = torch.tensor([0])
        md.track_ssm_h_dst = torch.tensor([8])
        md.track_ssm_final_src = torch.zeros(0, dtype=torch.int64)
        md.track_ssm_final_dst = torch.zeros(0, dtype=torch.int64)
        md.state_checkpoint_cu_starts = torch.tensor([0, 1])
        md.num_state_checkpoints = 1
        md.state_checkpoint_every_n_tokens = 64
    return md


class _Backend:
    def __init__(self, md, kernel=None):
        self.forward_metadata = md
        self.kernel_dispatcher = SimpleNamespace(
            extend_kernel=kernel if kernel is not None else _FlashInferKernel()
        )
        self.mis_metadata = None
        self.prefill_graph_metadata = None
        self.conv = torch.zeros(SLOTS, QKV, 3)
        self.temporal = torch.zeros(SLOTS, HV, D, D)
        self.cache_calls = []
        self.split_calls = []
        self.track_calls = []

    def state_layer_cache(self, layer_id):
        self.cache_calls.append(layer_id)
        return SimpleNamespace(conv=[self.conv], temporal=self.temporal)

    def _split_prefill_qkv(self, layer, mixed_qkv):
        self.split_calls.append(tuple(mixed_qkv.shape))
        rows = mixed_qkv.shape[0]
        q, k, v = torch.split(mixed_qkv, [Q_DIM, Q_DIM, V_DIM], dim=-1)
        return q.view(1, rows, H, D), k.view(1, rows, H, D), v.view(1, rows, HV, D)

    def _track_mamba_state_extend(self, fb, h, ssm_states, md, h_track_buf=None):
        self.track_calls.append((h, ssm_states, md, h_track_buf))


class _St:
    def __init__(self, fb):
        self.fb = fb
        self.ran_ops = []

    def ran(self, op, impl):
        self.ran_ops.append((op, impl))


def _batch(lens, mode=_Mode.EXTEND, logical=None, padded=None):
    return SimpleNamespace(
        forward_mode=mode,
        logical_forward_mode=logical if logical is not None else mode,
        extend_seq_lens_cpu=list(lens),
        extend_prefix_lens=torch.tensor([0, 5][: len(lens)]),
        global_num_token_non_padded_cpu=padded,
    )


class _Case(unittest.TestCase):
    def setUp(self):
        _install(self)

    def run_step(
        self,
        lens=(5, 3),
        options=gp.DEFAULT,
        *,
        track=False,
        spies=None,
        backend=None,
        gdn=None,
        fb=None,
        x=None,
    ):
        self.spies = spies or _Spies(self)
        self.backend = backend or _Backend(_metadata(lens, track=track))
        _STATE["backend"] = self.backend
        self.out_proj = []
        self.gdn = gdn or _gdn(out_proj_calls=self.out_proj)
        rows = sum(lens)
        self.x = x if x is not None else torch.zeros(rows, HIDDEN, dtype=torch.bfloat16)
        self.st = _St(fb or _batch(lens))
        ex = SimpleNamespace(gdn_prefill=options)
        return gp.extend(ex, SimpleNamespace(linear_attn=self.gdn), self.x, self.st)


class TestDispatch(_Case):
    def test_slot_off_runs_stock(self):
        spies = _Spies(self)
        st = _St(_batch((4,)))
        ex = SimpleNamespace(gdn_prefill=None)
        out = gp.extend(ex, SimpleNamespace(linear_attn=_gdn()), torch.zeros(4, 8), st)
        self.assertIsNone(out)
        self.assertEqual(spies.calls, [])
        self.assertEqual(st.ran_ops, [])

    def test_prefill_runs_the_kernels_in_the_stock_order(self):
        out = self.run_step()
        s = self.spies
        self.assertEqual(s.names(), ["split_conv", "gating", "flashinfer", "norm"])
        self.assertEqual(self.st.ran_ops, [("gdn", "conv_split+fi+rows_fused")])
        self.assertEqual(self.backend.cache_calls, [3])  # the layer's state, once
        # the conv: the prefill view of q|k|v, channel-last, in the stock call
        conv = s.kwargs("split_conv")
        self.assertEqual(tuple(conv["x"].shape), (QKV, 8))
        self.assertEqual(conv["x"].stride(), (1, QKV + V_DIM))
        self.assertIs(conv["conv_states"], self.backend.conv)
        self.assertEqual((conv["q_dim"], conv["k_dim"]), (Q_DIM, Q_DIM))
        self.assertEqual(conv["strip_cfg"], (32, 128, 4, 2048))
        self.assertEqual(conv["activation"], "silu")
        self.assertEqual(conv["seq_lens_cpu"], [5, 3])
        self.assertEqual(conv["has_initial_state"].tolist(), [False, True])
        self.assertIs(
            conv["cache_indices"], self.backend.forward_metadata.mamba_cache_indices
        )
        self.assertNotIn("block_table", conv)
        # the gates read b / a in place from the ba projection
        self.assertEqual(s.kwargs("gating")["gating_2d"], True)
        self.assertEqual(s.kwargs("gating")["a"].stride(), (2 * HV, 1))
        # the core: q / k / v views of the split planes, the stock kernel, the plan
        core = s.kwargs("flashinfer")
        self.assertIsInstance(core["kernel"], _FlashInferKernel)
        self.assertEqual(tuple(core["q"].shape), (1, 8, H, D))
        self.assertEqual(tuple(core["v"].shape), (1, 8, HV, D))
        self.assertTrue(core["q"].is_contiguous() and core["k"].is_contiguous())
        self.assertIs(core["ssm_states"], self.backend.temporal)
        self.assertEqual(
            (core["fused_rows"], core["l2norm_retile"], core["qk_one_launch"]),
            (True, True, True),
        )
        self.assertEqual(core["num_state_checkpoints"], 0)
        # the norm: the stock gated RMSNorm call with the strided 3D z
        norm = s.kwargs("norm")
        self.assertEqual(tuple(norm["x"].shape), (8 * HV, D))
        self.assertEqual(tuple(norm["z"].shape), (8, HV, D))
        self.assertFalse(norm["z"].is_contiguous())
        self.assertIsNone(norm["bias"])
        self.assertEqual(
            (norm["is_rms_norm"], norm["activation"], norm["norm_before_gate"]),
            (True, "sigmoid", True),
        )
        self.assertEqual((norm["group_size"], norm["retile"]), (None, True))
        # the output projection takes [T, HV * V]
        self.assertEqual(tuple(self.out_proj[0].shape), (8, V_DIM))
        self.assertEqual(tuple(out.shape), (8, 1))
        self.assertTrue(torch.equal(out, torch.full((8, 1), 2.0 * V_DIM)))

    def test_track_copies_fused(self):
        self.run_step(track=True)
        s = self.spies
        self.assertEqual(
            s.names(),
            [
                "track_conv",
                "split_conv",
                "gating",
                "flashinfer",
                "track_states",
                "norm",
            ],
        )
        md = self.backend.forward_metadata
        self.assertIs(s.kwargs("track_conv")["token_rows"], md.track_conv_indices)
        self.assertIs(s.kwargs("track_conv")["dst_slots"], md.conv_states_mask_indices)
        self.assertEqual(tuple(s.kwargs("track_states")["h"].shape), (1, HV, D, D))
        self.assertEqual(s.kwargs("flashinfer")["num_state_checkpoints"], 1)
        self.assertEqual(self.backend.track_calls, [])
        self.assertTrue(
            torch.equal(self.backend.conv, torch.zeros_like(self.backend.conv))
        )
        self.assertEqual(
            self.st.ran_ops, [("gdn", "conv_split+fi+rows_fused+track_fused")]
        )

    def test_track_copies_stock_where_the_fused_ones_decline(self):
        spies = _Spies(self, snapshot=False, state_rows=False)
        self.run_step(track=True, spies=spies)
        # the conv window by torch indexing, before the conv
        qkvz, _ = self.gdn._forward_input_proj(self.x)
        expected = qkvz[:, :QKV][torch.tensor([[1, 2, 3]])].transpose(1, 2)
        self.assertTrue(torch.equal(self.backend.conv[7], expected[0]))
        ((h, ssm, md, buf),) = self.backend.track_calls
        self.assertEqual(tuple(h.shape), (1, 1, HV, D, D))
        self.assertIs(ssm, self.backend.temporal)
        self.assertIsNone(buf)
        self.assertEqual(self.st.ran_ops, [("gdn", "conv_split+fi+rows_fused+track")])

    def test_recompute_rows_take_the_stock_track_copy(self):
        backend = _Backend(_metadata((5, 3), track=True))
        backend.forward_metadata.track_ssm_recompute_dst = torch.tensor([6])
        self.run_step(track=True, backend=backend)
        self.assertNotIn("track_states", self.spies.names())
        self.assertEqual(len(backend.track_calls), 1)

    def test_plain_conv_where_the_split_conv_does_not_apply(self):
        # one token: the split conv takes two or more
        self.run_step(lens=(1,), fb=_batch((1,)))
        conv = self.spies.kwargs("conv")
        self.assertEqual(
            conv["fwd_launch"],
            {"BLOCK_M": 8, "BLOCK_N": 256, "num_warps": 2, "num_stages": 2},
        )
        self.assertEqual(self.backend.split_calls, [(1, QKV)])
        self.assertEqual(self.st.ran_ops, [("gdn", "conv+fi+rows_fused")])

    def test_block_table_reaches_the_conv(self):
        backend = _Backend(_metadata((5, 3)))
        backend.forward_metadata.conv_block_table = torch.zeros(2, 2, dtype=torch.int32)
        self.run_step(backend=backend)
        table = self.spies.kwargs("split_conv")["block_table"]
        self.assertIs(table, backend.forward_metadata.conv_block_table)

    def test_options_off_run_the_stock_launches(self):
        off = gp.Options(
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
        self.run_step(options=off, track=True)
        s = self.spies
        self.assertEqual(s.names(), ["conv", "gating", "flashinfer", "norm"])
        self.assertIsNone(s.kwargs("conv")["fwd_launch"])
        self.assertFalse(s.kwargs("gating")["gating_2d"])
        core = s.kwargs("flashinfer")
        self.assertEqual(
            (core["fused_rows"], core["l2norm_retile"], core["qk_one_launch"]),
            (False, False, False),
        )
        self.assertFalse(s.kwargs("norm")["retile"])
        self.assertEqual(len(self.backend.track_calls), 1)
        self.assertEqual(self.st.ran_ops, [("gdn", "conv+fi+rows+track")])

    def test_strided_slots_take_the_stock_row_copies(self):
        backend = _Backend(_metadata((5, 3)))
        backend.forward_metadata.mamba_cache_indices = torch.tensor(
            [1, 0, 2, 0], dtype=torch.int32
        )[::2]
        self.run_step(backend=backend)
        self.assertFalse(self.spies.kwargs("flashinfer")["fused_rows"])
        self.assertEqual(self.st.ran_ops, [("gdn", "conv_split+fi+rows")])

    def test_installed_tmem_fix_is_recorded(self):
        gp._tmem_fixed = True
        self.run_step()
        self.assertEqual(self.st.ran_ops, [("gdn", "conv_split+fi_tmem+rows_fused")])


class TestFallbacks(_Case):
    """Every step the kernels do not serve returns None before anything is written."""

    def assert_stock(self, reason_part, **kw):
        out = self.run_step(**kw)
        self.assertIsNone(out)
        self.assertEqual(self.spies.calls, [])
        self.assertEqual(self.st.ran_ops, [])
        self.assertEqual(self.out_proj, [])
        self.assertTrue(
            torch.equal(self.backend.conv, torch.zeros_like(self.backend.conv))
        )
        reason, step = gp.prepare(self.gdn, self.x, self.st.fb)
        self.assertIsNone(step)
        self.assertIn(reason_part, reason)

    def test_modes(self):
        self.assert_stock("extend", fb=_batch((5, 3), mode=_Mode.TARGET_VERIFY))
        self.assert_stock("extend", fb=_batch((5, 3), mode=_Mode.DRAFT_EXTEND_V2))
        self.assert_stock(
            "extend", fb=_batch((5, 3), mode=_Mode.MIXED, logical=_Mode.DECODE)
        )

    def test_tp(self):
        self.assert_stock("TP2", gdn=_gdn(tp=2))

    def test_dtype(self):
        self.assert_stock("BF16", x=torch.zeros(8, HIDDEN, dtype=torch.float16))

    def test_prefill_graphs(self):
        _STATE["full_graph"] = True
        self.assert_stock("graph")
        # a breakable graph's capture takes the graph path (TestPrefillGraph); the
        # eager step still declines it
        _STATE.update(breakable=True, full_graph=False)
        x = torch.zeros(8, HIDDEN, dtype=torch.bfloat16)
        reason, step = gp.prepare(_gdn(), x, _batch((5, 3)))
        self.assertIsNone(step)
        self.assertIn("graph", reason)

    def test_cp(self):
        _STATE["cp"] = True
        self.assert_stock("CP")

    def test_dp_padding(self):
        self.assert_stock("DP padding", fb=_batch((5, 3), padded=6))

    def test_other_prefill_kernel(self):
        self.assert_stock(
            "_TritonKernel", backend=_Backend(_metadata((5, 3)), kernel=_TritonKernel())
        )

    def test_mis(self):
        backend = _Backend(_metadata((5, 3)))
        backend.mis_metadata = object()
        self.assert_stock("MIS", backend=backend)

    def test_captured_bucket(self):
        backend = _Backend(_metadata((5, 3)))
        backend.prefill_graph_metadata = object()
        self.assert_stock("bucket", backend=backend)

    def test_page_major_pool(self):
        backend = _Backend(_metadata((5, 3)))
        backend.conv = torch.zeros(QKV, SLOTS, 3).transpose(0, 1)
        self.assert_stock("strided", backend=backend)
        backend = _Backend(_metadata((5, 3)))
        backend.temporal = torch.zeros(SLOTS, HV, D, 2 * D)[..., :D]
        self.assert_stock("strided", backend=backend)


def _graph_meta(track=False, flashinfer=True):
    """A bucket's static tables (GDNPrefillGraphMetadata), with FlashInfer's."""
    tables = SimpleNamespace(
        cp_single=True,
        pre_track_mode="pre_mode",
        track_mode="mode",
        track_h_row="h_row",
    )
    return SimpleNamespace(
        track=track,
        flashinfer=tables if flashinfer else None,
        cu_seqlens="cu",
        cache_indices="slots",
        has_initial_state="init",
        conv_block_table="blocks",
        track_mode="conv_mode",
        track_conv_start="conv_start",
        track_dst="dst",
        num_logical_tokens="tokens",
    )


class TestPrefillGraph(_Case):
    """Inside a breakable prefill graph's capture the core runs on the bucket's
    FlashInfer tables in the segment, or without tables as a graph break."""

    def setUp(self):
        super().setUp()
        _STATE["breakable"] = True
        track = types.ModuleType("sglang.kernels.ops.mamba.gdn_prefill_track")
        track.gdn_track_conv_window = lambda **kw: self.spies.calls.append(
            ("track_conv_window", kw)
        )
        track.gdn_track_ssm_state = lambda **kw: self.spies.calls.append(
            ("track_ssm", kw)
        )
        tables = types.ModuleType(
            "sglang.srt.layers.attention.linear.gdn_prefill_graph"
        )
        tables.GDN_CHUNK_SIZE = 64
        patcher = mock.patch.dict(sys.modules, {m.__name__: m for m in (track, tables)})
        patcher.start()
        self.addCleanup(patcher.stop)
        patcher = mock.patch.object(fi_mod, "extend_in_graph", self.fi_graph)
        patcher.start()
        self.addCleanup(patcher.stop)
        self.ex = SimpleNamespace(gdn_prefill=gp.DEFAULT, gdn_graph_outputs=[])

    def fi_graph(self, kernel, q, k, v, g, beta, **kw):
        self.spies.calls.append(("extend_in_graph", dict(kw, kernel=kernel, q=q)))
        kw["output"].fill_(1.0)
        rows = kw["checkpoint_rows"]
        return torch.zeros(1, rows, HV, D, D) if rows else None

    def graph_step(self, meta, rows=8, lens=(5,)):
        self.spies = _Spies(self)
        self.backend = _Backend(_metadata(lens))
        _STATE["backend"] = self.backend
        self.out_proj = []
        self.gdn = _gdn(out_proj_calls=self.out_proj)
        self.x = torch.zeros(rows, HIDDEN, dtype=torch.bfloat16)
        # bucket rows past the batch's tokens: the eager step's DP padding decline
        self.st = _St(_batch(lens, padded=sum(lens)))
        self.st.view = SimpleNamespace(prefill_graph_metadata=meta)
        return gp.extend(
            self.ex, SimpleNamespace(linear_attn=self.gdn), self.x, self.st
        )

    def test_core_in_the_segment_on_the_tables(self):
        meta = _graph_meta()
        out = self.graph_step(meta)
        s = self.spies
        self.assertEqual(s.names(), ["split_conv", "gating", "extend_in_graph", "norm"])
        conv = s.kwargs("split_conv")
        self.assertEqual(tuple(conv["x"].shape), (QKV, 8))
        self.assertEqual(
            (conv["query_start_loc"], conv["cache_indices"], conv["block_table"]),
            ("cu", "slots", "blocks"),
        )
        self.assertEqual(conv["has_initial_state"], "init")
        self.assertIsNone(conv["seq_lens_cpu"])  # no host lengths in the segment
        self.assertNotIn("strip_cfg", conv)
        core = s.kwargs("extend_in_graph")
        self.assertIs(core["tables"], meta.flashinfer)
        self.assertIs(core["ssm_states"], self.backend.temporal)
        self.assertIsInstance(core["kernel"], _FlashInferKernel)
        self.assertEqual(
            (core["checkpoint_rows"], core["checkpoint_every_n_tokens"]), (0, 64)
        )
        self.assertEqual((core["l2norm_retile"], core["qk_one_launch"]), (True, True))
        # the shared output outside the graph pool, every bucket row
        (shared,) = self.ex.gdn_graph_outputs
        self.assertEqual(tuple(shared.shape), (8, HV, D))
        self.assertEqual(core["output"].data_ptr(), shared.data_ptr())
        self.assertTrue(torch.equal(s.kwargs("norm")["x"], torch.ones(8 * HV, D)))
        self.assertEqual(self.st.ran_ops, [("gdn", "graph+conv_split+fi")])
        self.assertEqual(tuple(out.shape), (8, 1))

    def test_graph_outputs_grow_and_keep_the_smaller_ones(self):
        self.graph_step(_graph_meta(), rows=8)
        first = self.ex.gdn_graph_outputs[0]
        self.graph_step(_graph_meta(), rows=4, lens=(3,))
        self.assertEqual(self.ex.gdn_graph_outputs, [first])
        self.graph_step(_graph_meta(), rows=16, lens=(9,))
        self.assertEqual(len(self.ex.gdn_graph_outputs), 2)
        self.assertEqual(self.ex.gdn_graph_outputs[0].shape[0], 16)
        self.assertIs(self.ex.gdn_graph_outputs[1], first)

    def test_track_snapshots_from_the_tables(self):
        meta = _graph_meta(track=True)
        self.graph_step(meta, rows=128, lens=(100,))
        s = self.spies
        self.assertEqual(
            s.names(),
            [
                "track_conv_window",
                "split_conv",
                "gating",
                "track_ssm",
                "extend_in_graph",
                "track_ssm",
                "norm",
            ],
        )
        window = s.kwargs("track_conv_window")
        self.assertEqual(
            (window["track_mode"], window["track_dst"], window["num_tokens"]),
            ("conv_mode", "dst", "tokens"),
        )
        pre, post = [c[1] for c in s.calls if c[0] == "track_ssm"]
        # before the kernel: chunk-0 rows snapshot the initial state
        self.assertEqual(pre["track_mode"], "pre_mode")
        self.assertEqual(tuple(pre["h"].shape), (1, *self.backend.temporal.shape))
        # after: the 64-token checkpoints of the 128-row bucket
        self.assertEqual(s.kwargs("extend_in_graph")["checkpoint_rows"], 2)
        self.assertEqual((post["track_mode"], post["track_h_row"]), ("mode", "h_row"))
        self.assertEqual(tuple(post["h"].shape), (1, 2, HV, D, D))
        self.assertEqual(self.st.ran_ops, [("gdn", "graph+conv_split+fi+track")])

    def test_no_tables_runs_the_core_as_a_break(self):
        seen = []

        def core(executor, gdn, step, qkv, a, b):
            seen.append((executor, gdn, step.kernel, tuple(qkv.shape), a.shape[0]))
            return torch.full((1, qkv.shape[0], HV, D), 3.0), "break"

        with mock.patch.object(pg, "gdn_core", core):
            out = self.graph_step(None)
        ((executor, gdn, kernel, qkv, a_rows),) = seen
        self.assertEqual((executor, gdn), (self.ex, self.gdn))
        self.assertIsInstance(kernel, _FlashInferKernel)
        self.assertEqual((qkv, a_rows), ((8, QKV), 8))
        # the projections and the norm stay in the segment
        self.assertEqual(self.spies.names(), ["norm"])
        norm = self.spies.kwargs("norm")["x"]
        self.assertTrue(torch.equal(norm, torch.full((8 * HV, D), 3.0)))
        self.assertEqual(self.st.ran_ops, [("gdn", "break")])
        self.assertEqual(tuple(out.shape), (8, 1))

    def test_triton_tables_keep_sglang_chain(self):
        self.assertIsNone(self.graph_step(_graph_meta(flashinfer=False)))
        self.assertEqual(self.spies.calls, [])
        self.assertEqual(self.st.ran_ops, [])

    def test_break_body_runs_the_eager_core_on_the_real_rows(self):
        live = _St(_batch((5,), padded=5))
        seen = {}

        def core_rows(options, step, gdn, fb, qkv, a, b, output):
            seen.update(fb=fb, rows=(qkv.shape[0], a.shape[0], b.shape[0]))
            seen["output"] = tuple(output.shape)
            output.fill_(1.0)
            return "conv_split+fi+rows_fused"

        output = torch.full((1, 8, HV, D), 7.0)
        args = (torch.zeros(8, QKV), torch.zeros(8, HV), torch.zeros(8, HV), output)
        with (
            mock.patch.object(pg, "_live", lambda executor: live),
            mock.patch.object(gp, "core_rows", core_rows),
        ):
            pg._gdn_core(self.ex, "gdn", "step", *args)
        self.assertIs(seen["fb"], live.fb)
        self.assertEqual((seen["rows"], seen["output"]), ((5, 5, 5), (1, 5, HV, D)))
        self.assertTrue(torch.equal(output[:, :5], torch.ones(1, 5, HV, D)))
        self.assertTrue(torch.equal(output[:, 5:], torch.zeros(1, 3, HV, D)))
        self.assertEqual(live.ran_ops, [("gdn.core", "conv_split+fi+rows_fused")])

    def test_core_rows_writes_the_output(self):
        _STATE["breakable"] = False
        lens = (5,)
        self.spies = _Spies(self)
        backend = _Backend(_metadata(lens))
        _STATE["backend"] = backend
        gdn = _gdn()
        fb = _batch(lens)
        _, step = gp.prepare(gdn, torch.zeros(5, HIDDEN, dtype=torch.bfloat16), fb)
        output = torch.zeros(1, 5, HV, D)
        qkvz, ba = gdn._forward_input_proj(torch.zeros(5, HIDDEN))
        qkv = qkvz[:, :QKV]
        impl = gp.core_rows(
            gp.DEFAULT, step, gdn, fb, qkv, ba[:, :HV], ba[:, HV:], output
        )
        self.assertEqual(impl, "conv_split+fi+rows_fused")
        self.assertIs(self.spies.kwargs("flashinfer")["output"], output)
        # an MIS batch keeps SGLang's extend
        backend.mis_metadata = object()
        backend.forward_extend = lambda layer, fb_, *a: torch.ones(1, 5, HV, D)
        impl = gp.core_rows(
            gp.DEFAULT, step, gdn, fb, qkv, ba[:, :HV], ba[:, HV:], output
        )
        self.assertEqual(impl, "mis_stock")
        self.assertTrue(torch.equal(output, torch.ones(1, 5, HV, D)))


class TestExtendInGraph(unittest.TestCase):
    """FlashInfer's calls on a bucket's static tables."""

    def run_core(self, cp_single, checkpoint_rows=0):
        calls, tails = [], []
        kernel = SimpleNamespace(_prefill_fn=lambda **kw: calls.append(kw))
        tables = SimpleNamespace(
            cp_single=cp_single,
            cp_cu_seqlens=torch.tensor([0, 5]),
            ncp_cu_seqlens=torch.tensor([0, 0, 0]),
            cp_state_indices="cp_slots",
            ncp_state_indices="ncp_slots",
            cp_checkpoint_cu="cp_ck",
            ncp_checkpoint_cu="ncp_ck",
        )
        prep = lambda q, k, v, retile, qk_one_launch: (q, k, v)
        q, v = torch.zeros(1, 8, H, D), torch.zeros(1, 8, HV, D)
        g, beta = torch.zeros(1, 8, HV), torch.ones(1, 8, HV)
        self.ssm, self.output = torch.zeros(SLOTS, HV, D, D), torch.zeros(8, HV, D)
        with (
            mock.patch.object(fi_mod, "gdn_prefill_qkv_prepare_fwd", prep),
            mock.patch.object(
                tail_mod, "gdn_cp_tail_neutralize", lambda *a: tails.append(a[-1])
            ),
        ):
            out = fi_mod.extend_in_graph(
                kernel,
                q,
                q,
                v,
                g,
                beta,
                ssm_states=self.ssm,
                tables=tables,
                output=self.output,
                checkpoint_rows=checkpoint_rows,
                checkpoint_every_n_tokens=64,
            )
        return calls, tails, tables, out

    def test_one_sequence_on_the_cp_kernel(self):
        calls, tails, tables, out = self.run_core(cp_single=True)
        self.assertIsNone(out)
        cp, ncp = calls
        self.assertEqual((cp["use_cp"], cp["max_seqlen"]), (True, 8))
        self.assertIs(cp["cu_seqlens"], tables.cp_cu_seqlens)
        slots = (cp["state_indices"], ncp["state_indices"])
        self.assertEqual(slots, ("cp_slots", "ncp_slots"))
        self.assertIs(ncp["cu_seqlens"], tables.ncp_cu_seqlens)
        self.assertEqual(ncp["use_cp"], False)
        self.assertNotIn("max_seqlen", ncp)
        for call in calls:  # in place by slot, FlashInfer's own l2norm off
            self.assertIs(call["initial_state"], self.ssm)
            self.assertIs(call["output_state"], self.ssm)
            self.assertIs(call["output"], self.output)
            self.assertFalse(call["use_qk_l2norm_in_kernel"])
            self.assertTrue(torch.equal(call["g"], torch.ones(8, HV)))
            self.assertNotIn("state_checkpoints", call)
        # the rows past each call's batch neutralized: CP [0, T], non-CP its last pair
        self.assertIs(tails[0], tables.cp_cu_seqlens)
        self.assertEqual(tails[1].tolist(), [0, 0])

    def test_other_batches_on_the_non_cp_kernel_with_checkpoints(self):
        calls, tails, tables, out = self.run_core(cp_single=False, checkpoint_rows=2)
        (ncp,) = calls
        self.assertEqual(ncp["use_cp"], False)
        self.assertEqual(len(tails), 1)
        self.assertEqual(tuple(ncp["state_checkpoints"].shape), (2, HV, D, D))
        self.assertEqual(ncp["checkpoint_cu_starts"], "ncp_ck")
        self.assertEqual(ncp["checkpoint_every_n_tokens"], 64)
        self.assertEqual(tuple(out.shape), (1, 2, HV, D, D))


class TestOptions(unittest.TestCase):
    def test_defaults_are_the_recipe_values(self):
        o = gp.DEFAULT
        self.assertEqual(o.strip_cfg(), (32, 128, 4, 2048))
        self.assertEqual(
            o.fwd_launch(),
            {"BLOCK_M": 8, "BLOCK_N": 256, "num_warps": 2, "num_stages": 2},
        )
        for name in (
            "split_conv",
            "gating_2d",
            "l2norm_retile",
            "qk_l2norm_one_launch",
            "norm_retile",
            "fused_state_copy",
            "fused_track",
            "tmem_dealloc",
        ):
            self.assertTrue(getattr(o, name), name)

    def test_tiles_are_validated(self):
        self.assertEqual(
            gp.Options(conv_strip=(64, 256, 8)).strip_cfg().min_tokens, 2048
        )
        for bad in (
            dict(conv_strip=(24, 128, 4, 2048)),
            dict(conv_strip=(32, 128, 3, 2048)),
            dict(conv_strip=(0,)),
            dict(conv_tile=(4, 256, 2, 2)),
            dict(conv_tile=(8, 32, 2, 2)),
            dict(conv_tile=(0,)),
        ):
            with self.subTest(**bad), self.assertRaises(ValueError):
                gp.Options(**bad)


class TestBuild(unittest.TestCase):
    def setUp(self):
        _install(self)

    def test_layer_reasons(self):
        self.assertEqual(gp.layer_reasons(_gdn()), "")
        self.assertIn("TP2", gp.layer_reasons(_gdn(tp=2)))
        self.assertIn("head dims", gp.layer_reasons(_gdn(head_dim=64)))
        self.assertIn("value heads", gp.layer_reasons(_gdn(num_v_heads=5 * H)))
        self.assertIn(
            "RMSNorm", gp.layer_reasons(_gdn(norm=SimpleNamespace(bias=None)))
        )
        norm = _gdn().norm
        norm.bias = torch.nn.Parameter(torch.zeros(D))
        self.assertIn("RMSNorm", gp.layer_reasons(_gdn(norm=norm)))

    def test_needs_an_sm10x_gpu(self):
        with mock.patch.object(torch.cuda, "is_available", return_value=False):
            self.assertIn("SM10x", gp.build(self.model()))
        with (
            mock.patch.object(torch.cuda, "is_available", return_value=True),
            mock.patch.object(torch.cuda, "get_device_capability", return_value=(9, 0)),
        ):
            self.assertIn("SM10x", gp.build(self.model()))

    def model(self, gdns=None):
        gdns = gdns if gdns is not None else [_gdn(), _gdn()]
        layers = [_Linear(i, g) for i, g in enumerate(gdns)] + [object()]
        m = SimpleNamespace(layers=layers, start_layer=0, end_layer=len(layers))
        return SimpleNamespace(model=m)

    def build(self, model, options=gp.DEFAULT, *, strip_error=None, tmem=None):
        """build on a faked SM100 with the strip warmup and the #8854 install spied."""
        self.warmups, self.installs = [], []

        def warmup(cfg):
            self.warmups.append(cfg)
            if strip_error is not None:
                raise strip_error

        def install():
            self.installs.append(1)
            if isinstance(tmem, Exception):
                raise tmem
            return True

        fake = types.ModuleType(
            "sgl_nvidia_mega.qwen3_8.kernels.ops.mamba.gdn_prefill_tmem_dealloc"
        )
        fake.install = install
        with (
            mock.patch.object(torch.cuda, "is_available", return_value=True),
            mock.patch.object(
                torch.cuda, "get_device_capability", return_value=(10, 0)
            ),
            mock.patch.object(conv_mod, "warmup_prefill_conv_strip", warmup),
            mock.patch.dict(sys.modules, {fake.__name__: fake}),
        ):
            return gp.build(model, options)

    def test_builds_the_strip_and_installs_the_tmem_fix(self):
        self.assertEqual(self.build(self.model()), "")
        self.assertEqual(self.warmups, [(32, 128, 4, 2048)])
        self.assertEqual(self.installs, [1])
        self.assertTrue(gp._tmem_fixed)
        self.assertEqual(self.build(self.model()), "")  # installed once per process
        self.assertEqual(self.installs, [])

    def test_refusals(self):
        self.assertEqual(self.build(self.model(gdns=[])), "no GDN layer")
        reason = self.build(self.model(gdns=[_gdn(), _gdn(tp=2)]))
        self.assertTrue(reason.startswith("layer 1: ") and "TP2" in reason, reason)
        _STATE["fi"] = False
        self.assertIn("FlashInfer", self.build(self.model()))
        _STATE["fi"] = True
        reason = self.build(self.model(), strip_error=RuntimeError("nvcc failed"))
        self.assertIn("strip does not build: RuntimeError: nvcc failed", reason)

    def test_strip_off_builds_nothing(self):
        options = gp.Options(conv_strip=None, tmem_dealloc=False)
        self.assertEqual(self.build(self.model(), options), "")
        self.assertEqual((self.warmups, self.installs), ([], []))
        self.assertFalse(gp._tmem_fixed)

    def test_tmem_fix_on_another_flashinfer_warns_and_builds(self):
        with self.assertLogs(gp.logger, logging.WARNING) as logs:
            reason = self.build(
                self.model(), tmem=ImportError("not FlashInfer v0.7.3's")
            )
        self.assertEqual(reason, "")
        self.assertFalse(gp._tmem_fixed)
        self.assertIn("not FlashInfer v0.7.3's", logs.output[0])


if __name__ == "__main__":
    unittest.main()
