# SPDX-License-Identifier: Apache-2.0
"""The GDN u-cache verify (runtime/gdn_ucache.py) on CPU: the per-forward prep and its
order, the flush launch's arguments under each option, the conv / norm dispatch of
``verify`` and the startup and pool refusals. The flush kernel, SGLang's ops and the
Triton launchers are spies or CPU references; the ported Triton kernels themselves run
once under Triton's interpreter, in a subprocess."""

import dataclasses
import json
import os
import subprocess
import sys
import types
import unittest
from types import SimpleNamespace
from typing import NamedTuple
from unittest import mock

import torch

from sgl_nvidia_mega.qwen3_8.kernels.ops.attention.gdn_front import conv as front_conv
from sgl_nvidia_mega.qwen3_8.kernels.ops.attention.linear import (
    gdn_prefill_norm as norm_mod,
)
from sgl_nvidia_mega.qwen3_8.kernels.ops.attention.linear import gdn_ucache_verify as K
from sgl_nvidia_mega.qwen3_8.runtime import gdn_ucache as U

RUBIN = (
    "sgl_nvidia_mega.qwen3_8.kernels.ops.attention.linear.gdn_ucache_rubin."
    "gdn_replay_mtp_umma_rubin"
)
FI = "flashinfer.gdn_kernels.gdn_replay_mtp_umma"
H, HV, KD, VD = 2, 6, 128, 128
D = (2 * H + HV) * KD
T = 4
LAYER_IDS = (3, 7)  # the pool's GDN layers; layer 7 is index 1
SLOTS = 8
CACHE_INDICES = [3, 0, 5, -1, 1, 6]  # slot 0 and -1 are padding
BF16 = torch.bfloat16


class FusedConv(NamedTuple):
    x: torch.Tensor
    weight: torch.Tensor
    conv_state: torch.Tensor
    window: torch.Tensor
    window_indices: torch.Tensor


class FusedNorm(NamedTuple):
    z: torch.Tensor
    weight: torch.Tensor
    eps: float


class _Launch:
    """A Triton kernel's ``kernel[grid](...)``: records the call, then runs ``fn``."""

    def __init__(self, name, calls, fn=None):
        self.name, self.calls, self.fn = name, calls, fn

    def __getitem__(self, grid):
        def launch(*args, **kwargs):
            self.calls.append((self.name, grid, args, kwargs))
            if self.fn is not None:
                self.fn(*args, **kwargs)

        return launch


def _gather_reference(calls):
    def gather(*, slots, buffers, request_rows=None):
        calls.append(("gather", slots, request_rows))
        live = slots > 0
        index = (request_rows if buffers.by_request_row else slots).long()
        index = torch.where(live, index, torch.zeros_like(index))
        zero = torch.zeros_like(slots)
        return K.GdnUcacheRows(
            slots=torch.where(live, slots, torch.full_like(slots, -1)),
            hist_len=torch.where(live, buffers.hist_len[index], zero),
            cache_base=torch.where(live, buffers.cache_base[index], zero),
            ring_rows=(
                torch.where(live, index.int(), zero) if buffers.by_request_row else None
            ),
        )

    return gather


def _order_reference(calls):
    def order(*, hist_len, flush_min):
        calls.append(("order", flush_min))
        folds = hist_len >= flush_min
        return torch.cat([torch.nonzero(folds), torch.nonzero(~folds)]).flatten().int()

    return order


def _zero_reference(slots, output, row_numel, BLOCK):
    output[slots < 0] = 0


class _Env:
    """Fakes for one test: the device capability, the flush kernel (SM107's vendored
    one or FlashInfer's), SGLang's ops and the Triton launchers."""

    def __init__(
        self,
        capability=(10, 7),
        *,
        conv_reason=None,
        norm_reason=None,
        rows_per_block=2,
        flashinfer=True,
    ):
        self.calls = []
        self.conv_reason = conv_reason
        self.norm_reason = norm_reason
        self.rows_per_block = rows_per_block
        calls = self.calls

        def flush(**kwargs):
            calls.append(("flush", kwargs))
            return kwargs["output"]

        rubin = types.ModuleType(RUBIN)
        rubin.GdnFusedConv, rubin.GdnFusedNorm = FusedConv, FusedNorm
        rubin.fused_conv_unsupported_reason = lambda fc, **kw: self.conv_reason
        rubin.fused_norm_unsupported_reason = lambda fn, **kw: self.norm_reason
        rubin.strided_inputs_layout = lambda *args: {}
        rubin.gated_delta_rule_mtp_ucache_flush_umma_rubin = flush
        modules = {RUBIN: rubin}
        if flashinfer:
            fi = types.ModuleType(FI)
            fi.gated_delta_rule_mtp_ucache_flush_umma = flush
            modules[FI] = fi
        proj = types.ModuleType("sglang.kernels.ops.attention.triton_gdn_fused_proj")

        def views(qkvz, ba, nk, nv, hk, hv):
            calls.append(("views", qkvz, ba))
            qkv_dim = nk * hk * 2 + nv * hv
            return (
                qkvz[:, :qkv_dim],
                qkvz[:, qkv_dim:].view(qkvz.shape[0], nv, hv),
                ba[:, :nv],
                ba[:, nv : 2 * nv],
            )

        def fused_split(mixed, nq, nk, nv, dq, dk, dv):
            calls.append(("fused_split", mixed))
            q, k, v = torch.split(mixed, [nq * dq, nk * dk, nv * dv], dim=-1)
            rows = mixed.shape[0]
            return (
                q.reshape(1, rows, nq, dq).contiguous(),
                k.reshape(1, rows, nk, dk).contiguous(),
                v.reshape(1, rows, nv, dv).contiguous(),
            )

        proj.qwen3_5_gdn_prefill_projection_views = views
        proj.fused_qkv_split_gdn_prefill = fused_split
        conv = types.ModuleType("sglang.kernels.ops.mamba.causal_conv1d_triton")

        def stock_conv(x, conv_state, weight, bias, activation, **kw):
            calls.append(("stock_conv", x, conv_state, kw))
            return torch.empty_like(x).copy_(x)

        conv.causal_conv1d_update = stock_conv
        norm = types.ModuleType("sglang.kernels.ops.attention.fla.layernorm_gated")
        norm.calc_rows_per_block = lambda rows, device: self.rows_per_block
        backend = types.ModuleType("sglang.srt.layers.attention.linear.gdn_backend")
        backend.MAX_FUSED_QKV_SPLIT_DIM = 8192
        qwen4 = types.ModuleType("sglang.srt.models.qwen4_exp")
        qwen4.Qwen4ExpLinearDecoderLayer = _LinearLayer
        for m in (proj, conv, norm, backend, qwen4):
            modules[m.__name__] = m

        def verify_norm(x, weight, bias, eps, **kw):
            calls.append(("verify_norm", x, weight, bias, eps, kw))
            return x, None, None

        def front(x, conv_state, weight, bias, activation, slots, window, rows, **kw):
            calls.append(("front_conv", x, conv_state, slots, window, rows, kw))
            out = torch.empty(
                (x.shape[0], x.shape[2], x.shape[1]), dtype=x.dtype
            ).transpose(1, 2)
            return out.copy_(x)

        self.patches = [
            mock.patch.dict(sys.modules, modules),
            mock.patch.object(
                torch.cuda, "get_device_capability", lambda *a: capability
            ),
            mock.patch.object(K, "_assert_not_capturing", lambda what: None),
            mock.patch.object(K, "gather_gdn_ucache_rows", _gather_reference(calls)),
            mock.patch.object(K, "_flush_first_order", _order_reference(calls)),
            mock.patch.object(
                K, "_zero_padded_rows_kernel", _Launch("zero", calls, _zero_reference)
            ),
            mock.patch.object(
                K, "_refresh_gate_params_kernel", _Launch("refresh", calls)
            ),
            mock.patch.object(K, "_split_qkv_kernel", _Launch("split", calls)),
            mock.patch.object(front_conv, "conv_update", front),
            mock.patch.object(
                norm_mod, "calc_rows_per_block", lambda rows, device: rows_per_block
            ),
            mock.patch.object(norm_mod, "_layer_norm_fwd", verify_norm),
        ]

    def __enter__(self):
        for p in self.patches:
            p.start()
        return self

    def __exit__(self, *exc):
        for p in reversed(self.patches):
            p.stop()

    def names(self):
        return [c[0] for c in self.calls]

    def flushes(self):
        return [c[1] for c in self.calls if c[0] == "flush"]


class _LinearLayer:  # the fake GDN decoder layer type
    def __init__(self, gdn):
        self.linear_attn = gdn


def _buffers(*, by_request_row=False, k_dtype=BF16, rows=SLOTS):
    return SimpleNamespace(
        k_ring=torch.zeros(len(LAYER_IDS), rows, H, 32, KD, dtype=k_dtype),
        u_ring=torch.zeros(len(LAYER_IDS), rows, HV, 32, VD, dtype=torch.float16),
        g_ring=torch.zeros(len(LAYER_IDS), rows, HV, 32),
        hist_len=torch.tensor([0, 5, 13, 2, 9, 7, 1, 4], dtype=torch.int32)[:rows],
        cache_base=torch.tensor([0, 3, 31, 6, 2, 1, 0, 9], dtype=torch.int32)[:rows],
        flush_min=13,
        by_request_row=by_request_row,
    )


def _view(buffers, *, dedup_window=True, batch=len(CACHE_INDICES)):
    layers = len(LAYER_IDS)
    if dedup_window:
        phys = torch.zeros(layers, batch, D, T + 2, dtype=BF16)
        window = phys.as_strided(
            (layers, batch, T, D, 3),
            (batch * D * (T + 2), D * (T + 2), 1, T + 2, 1),
        )
    else:
        window = torch.zeros(layers, batch, T, D, 3, dtype=BF16)
    mamba = SimpleNamespace(
        layer_ids=LAYER_IDS,
        conv=(torch.zeros(layers, SLOTS, D, 3, dtype=BF16),),
        temporal=torch.zeros(layers, SLOTS, HV, VD, KD, dtype=torch.float16),
        intermediate_conv_window=(window,),
    )
    indices = (
        CACHE_INDICES * ((batch + len(CACHE_INDICES) - 1) // len(CACHE_INDICES))
    )[:batch]
    meta = SimpleNamespace(
        state_indices=torch.tensor(indices, dtype=torch.int32),
        intermediate_indices=torch.arange(batch, dtype=torch.int32),
    )
    return SimpleNamespace(gdn=meta, mamba=mamba)


def _norm_init(case):
    """_Norm.__init__ with the case's attributes (``norm``) set after it."""
    init = _Norm.__init__

    def patched(self, *args, **kw):
        init(self, *args, **kw)
        for name, value in case.get("norm", {}).items():
            setattr(self, name, value)

    return patched


class _Norm:
    def __init__(self, calls, activation="sigmoid"):
        self.calls = calls
        self.group_size = None
        self.bias = None
        self.norm_before_gate = True
        self.activation = activation
        self.eps = 1e-6
        self.weight = torch.ones(VD, dtype=BF16)

    def __call__(self, x, z):
        self.calls.append(("norm", x, z))
        return x


def _gdn(env, *, layer_id=7, batch=len(CACHE_INDICES), bias=None, activation="sigmoid"):
    rows = batch * T
    qkvz = torch.randn(rows, D + HV * VD).to(BF16)
    ba = torch.randn(rows, 2 * HV).to(BF16)
    attn = SimpleNamespace(
        layer_id=layer_id,
        conv_weights=torch.zeros(D, 4, dtype=BF16),
        bias=bias,
        activation="silu",
        A_log=torch.zeros(HV),
        dt_bias=torch.zeros(HV, dtype=BF16),
        q_dim=H * KD,
        k_dim=H * KD,
        v_dim=HV * VD,
        num_q_heads=H,
        num_k_heads=H,
        num_v_heads=HV,
        head_q_dim=KD,
        head_k_dim=KD,
        head_v_dim=VD,
    )

    def out_proj(x):
        env.calls.append(("out_proj", x))
        return x[:, :16], None

    return SimpleNamespace(
        num_k_heads=H,
        num_v_heads=HV,
        attn_tp_size=1,
        head_k_dim=KD,
        head_v_dim=VD,
        attn=attn,
        norm=_Norm(env.calls, activation),
        out_proj=out_proj,
        _forward_input_proj=lambda x: (qkvz, ba),
        qkvz=qkvz,
        ba=ba,
    )


class _St:
    def __init__(self, batch=len(CACHE_INDICES), tokens=T):
        self.fb = SimpleNamespace(
            spec_info=SimpleNamespace(draft_token_num=tokens),
            req_pool_indices=torch.arange(batch, dtype=torch.int64) + 1,
        )
        self.ran_ops = []

    def ran(self, op, impl):
        self.ran_ops.append((op, impl))


def _verifier(by_request_row=False, options=None, first_layer_id=LAYER_IDS[0]):
    return U.UcacheVerify(
        device=torch.device("cpu"),
        first_layer_id=first_layer_id,
        by_request_row=by_request_row,
        options=options,
    )


def _core(uv, buffers, *, layer_id, packed=True, request_rows=None, fused=None):
    rows = len(CACHE_INDICES) * T
    if packed:
        qkv = torch.randn(rows, D).to(BF16)
        q, k, v = torch.split(qkv, [H * KD, H * KD, HV * VD], dim=-1)
    else:
        q, k, v = (torch.randn(rows, n * KD).to(BF16) for n in (H, H, HV))
    gates = torch.randn(rows, 2 * HV).to(BF16)
    li = LAYER_IDS.index(layer_id)
    A_log, dt_bias = torch.zeros(HV), torch.zeros(HV, dtype=BF16)
    out = uv.core(
        layer_id,
        q.view(1, rows, H, KD),
        k.view(1, rows, H, KD),
        v.view(1, rows, HV, VD),
        gates[:, HV:],
        gates[:, :HV],
        A_log,
        dt_bias,
        torch.zeros(SLOTS, HV, VD, KD, dtype=torch.float16),
        buffers.k_ring[li],
        buffers.u_ring[li],
        buffers.g_ring[li],
        buffers,
        torch.tensor(CACHE_INDICES, dtype=torch.int32),
        T,
        request_rows,
        **(fused or {}),
    )
    return out, (q, k, v, A_log, dt_bias)


class TestOptions(unittest.TestCase):
    def test_presets(self):
        self.assertEqual(
            (
                U.HIGH_CONCURRENCY.strided_qkv,
                U.HIGH_CONCURRENCY.pdl,
                U.HIGH_CONCURRENCY.flush_first,
                U.HIGH_CONCURRENCY.fused_norm,
            ),
            (True, True, True, True),
        )
        self.assertFalse(U.HIGH_CONCURRENCY.fused_conv)
        self.assertTrue(U.HIGH_CONCURRENCY_FUSED_CONV.fused_conv)
        self.assertEqual(U.HIGH_CONCURRENCY.fused_norm_max_batch, 48)
        self.assertIs(U.DEFAULT, U.HIGH_CONCURRENCY)
        flush = U.PLAIN.flush_options()
        self.assertEqual(flush, K.GdnUcacheFlushOptions())
        self.assertEqual(flush.kernel_kwargs(), {})
        self.assertEqual(
            U.DEFAULT.flush_options().kernel_kwargs(),
            {"strided_inputs": True, "pdl": True},
        )

    def test_sm107_runs_the_vendored_kernel_with_the_recipe_options(self):
        with _Env((10, 7)):
            uv = _verifier()
            self.assertIs(uv.options, U.DEFAULT)
            self.assertEqual(uv._flush_kernel.__name__, "flush")
            self.assertTrue(uv.fused_conv_capable)
            self.assertFalse(uv.fused_conv_enabled)
            self.assertTrue(uv.fused_norm_enabled)
            self.assertTrue(
                _verifier(options=U.HIGH_CONCURRENCY_FUSED_CONV).fused_conv_enabled
            )
            # the fused prologue / epilogue ride on the strided, dependent launch
            uv = _verifier(options=U.Options(pdl=False, fused_conv=True))
            self.assertFalse(uv.fused_conv_enabled or uv.fused_norm_enabled)

    def test_sm100_takes_flashinfer_without_options(self):
        for capability in ((10, 0), (10, 3)):
            with _Env(capability) as env:
                uv = _verifier()
                self.assertIs(uv.options, U.PLAIN)
                self.assertFalse(uv.fused_conv_capable)
                _core(uv, _buffers(), layer_id=LAYER_IDS[0])
                (kw,) = env.flushes()
                for name in ("strided_inputs", "pdl", "row_order", "ring_indices"):
                    self.assertNotIn(name, kw)
                with self.assertRaises(ValueError):
                    _verifier(options=U.DEFAULT)
                with self.assertRaises(ValueError):
                    _verifier(by_request_row=True, options=U.PLAIN)
        with _Env((10, 0), flashinfer=False):
            with self.assertRaisesRegex(RuntimeError, "FlashInfer"):
                _verifier()
        with _Env((9, 0)):
            with self.assertRaisesRegex(RuntimeError, "SM100, SM103 or SM107"):
                _verifier(options=U.PLAIN)


class TestCore(unittest.TestCase):
    def test_first_layer_prep_then_flush(self):
        with _Env() as env:
            uv = _verifier()
            buffers = _buffers()
            out, (q, k, v, A_log, dt_bias) = _core(uv, buffers, layer_id=LAYER_IDS[0])
            # no gate pair is registered before the first flush; the cursors, the
            # row order and the padded rows come before it, the zeroing last
            self.assertEqual(env.names(), ["gather", "order", "zero", "flush"])
            _core(uv, buffers, layer_id=LAYER_IDS[1])
            self.assertEqual(env.names()[4:], ["flush"])
            del env.calls[:]
            _core(uv, buffers, layer_id=LAYER_IDS[0])
            self.assertEqual(
                env.names(), ["gather", "refresh", "order", "zero", "flush"]
            )
            (_, grid, (src_ptrs, dst_ptrs, numel), kw) = env.calls[1]
            self.assertEqual((grid, numel, kw), ((2,), HV, {"BLOCK": 8}))
            self.assertEqual(src_ptrs.dtype, torch.int64)
            gather = env.calls[0]
            self.assertEqual(gather[1].tolist(), CACHE_INDICES)
            self.assertIsNone(gather[2])
            self.assertEqual(out.shape, (1, len(CACHE_INDICES) * T, HV, VD))

    def test_flush_arguments(self):
        with _Env() as env:
            uv = _verifier()
            buffers = _buffers()
            _, (q, k, v, A_log, dt_bias) = _core(uv, buffers, layer_id=LAYER_IDS[0])
            (kw,) = env.flushes()
            B = len(CACHE_INDICES)
            self.assertEqual(kw["q"].shape, (B, T, H, KD))
            self.assertEqual(kw["v"].shape, (B, T, HV, VD))
            self.assertEqual(kw["a"].shape, (B, T, HV))
            # strided_qkv: the packed rows are read in place
            self.assertEqual(kw["q"].data_ptr(), q.data_ptr())
            self.assertEqual(kw["v"].stride(1), D)
            self.assertTrue(kw["strided_inputs"] and kw["pdl"])
            self.assertFalse(kw["restart_hist_on_flush"])
            self.assertEqual(kw["flush_min"], 13)
            self.assertAlmostEqual(kw["scale"], KD**-0.5)
            self.assertEqual(kw["initial_state_indices"].tolist(), [3, -1, 5, -1, 1, 6])
            self.assertEqual(kw["hist_len"].tolist(), [2, 0, 7, 0, 5, 1])
            self.assertEqual(kw["cache_base"].tolist(), [6, 0, 1, 0, 3, 0])
            self.assertEqual(kw["row_order"].tolist(), [0, 1, 2, 3, 4, 5])
            self.assertNotIn("ring_indices", kw)
            self.assertNotIn("fused_conv", kw)
            self.assertNotIn("fused_norm", kw)
            # A_log is fp32 already (aliased); dt_bias gets an fp32 copy
            self.assertEqual(kw["A_log"].data_ptr(), A_log.data_ptr())
            self.assertEqual(kw["dt_bias"].dtype, torch.float32)
            self.assertNotEqual(kw["dt_bias"].data_ptr(), dt_bias.data_ptr())
            self.assertEqual(kw["output"].shape, (B, T, HV, VD))
            self.assertIs(kw["k_cache"]._base, buffers.k_ring)

    def test_padded_rows_read_zero_and_the_buffer_is_shared(self):
        with _Env():
            uv = _verifier()
            out, _ = _core(uv, _buffers(), layer_id=LAYER_IDS[0])
            uv._outputs[(T, HV, VD)].fill_(1)
            out, _ = _core(uv, _buffers(), layer_id=LAYER_IDS[0])
            out = out.view(len(CACHE_INDICES), -1)
            self.assertEqual(out[1].abs().sum().item(), 0)
            self.assertEqual(out[3].abs().sum().item(), 0)
            self.assertEqual(out[0].abs().sum().item(), T * HV * VD)
            out2, _ = _core(uv, _buffers(), layer_id=LAYER_IDS[1])
            self.assertEqual(out2.data_ptr(), out.data_ptr())

    def test_rings_by_request_row(self):
        with _Env() as env:
            uv = _verifier(by_request_row=True)
            buffers = _buffers(by_request_row=True)
            rows = torch.tensor([2, 9, 4, 0, 1, 7, 6], dtype=torch.int64)
            _core(uv, buffers, layer_id=LAYER_IDS[0], request_rows=rows)
            self.assertEqual(
                env.calls[0][2].tolist(), rows[: len(CACHE_INDICES)].tolist()
            )
            (kw,) = env.flushes()
            self.assertEqual(kw["ring_indices"].tolist(), [2, 0, 4, 0, 1, 7])
            self.assertEqual(kw["hist_len"].tolist(), [13, 0, 9, 0, 5, 4])
            with self.assertRaises(AssertionError):
                _core(_verifier(), buffers, layer_id=LAYER_IDS[0], request_rows=rows)

    def test_contiguous_qkv_without_strided_inputs(self):
        with _Env() as env:
            options = U.Options(strided_qkv=False, pdl=False, fused_norm=False)
            uv = _verifier(options=options)
            _, (q, k, v, *_) = _core(uv, _buffers(), layer_id=LAYER_IDS[0])
            split = [c for c in env.calls if c[0] == "split"]
            (_, grid, args, kw) = split[0]
            self.assertEqual(grid, (len(CACHE_INDICES) * T, (D + 1023) // 1024))
            self.assertEqual(args[4], D)  # the packed rows' stride
            self.assertEqual(
                kw,
                {"Q_COLS": H * KD, "K_COLS": H * KD, "V_COLS": HV * VD, "BLOCK": 1024},
            )
            (fkw,) = env.flushes()
            self.assertNotIn("strided_inputs", fkw)
            self.assertNotIn("pdl", fkw)
            self.assertTrue(fkw["q"].is_contiguous())
            self.assertEqual(fkw["row_order"].tolist(), [0, 1, 2, 3, 4, 5])
            # q / k / v already contiguous: no split launch
            del env.calls[:]
            _core(uv, _buffers(), layer_id=LAYER_IDS[1], packed=False)
            self.assertEqual(env.names(), ["flush"])
            # without flush_first: no row order
            del env.calls[:]
            uv = _verifier(options=U.Options(flush_first=False))
            _core(uv, _buffers(), layer_id=LAYER_IDS[0])
            self.assertEqual(env.names(), ["gather", "zero", "flush"])
            self.assertNotIn("row_order", env.flushes()[0])


class TestVerify(unittest.TestCase):
    def run_verify(
        self,
        env,
        *,
        options=None,
        buffers=None,
        batch=len(CACHE_INDICES),
        tokens=T,
        **kw,
    ):
        buffers = buffers or _buffers()
        view = _view(buffers, batch=batch, dedup_window=kw.pop("dedup_window", True))
        uv = _verifier(
            by_request_row=buffers.by_request_row, options=options, first_layer_id=7
        )
        gdn = _gdn(env, batch=batch, **kw)
        st = _St(batch, tokens)
        executor = SimpleNamespace(
            gdn_ucache=uv,
            ucache_state=SimpleNamespace(buffers=buffers),
            smallm_gemm=None,
        )
        out = U.verify(executor, gdn, torch.zeros(batch * T, 16), st, view)
        view.rings = buffers  # for the assertions on the rings
        return out, gdn, st, view

    def test_separate_conv_then_fused_norm(self):
        with _Env() as env:
            out, gdn, st, view = self.run_verify(env)
            names = env.names()
            self.assertEqual(
                names,
                [
                    "views",
                    "front_conv",
                    "gather",
                    "order",
                    "zero",
                    "flush",
                    "out_proj",
                ],
            )
            B, rows = len(CACHE_INDICES), len(CACHE_INDICES) * T
            (_, x, conv_state, slots, window, window_rows, ckw) = env.calls[1]
            self.assertEqual(x.shape, (B, D, T))
            self.assertEqual(x.data_ptr(), gdn.qkvz.data_ptr())  # the projection rows
            self.assertIs(conv_state._base, view.mamba.conv[0])
            self.assertEqual(
                conv_state.storage_offset(), view.mamba.conv[0][1].storage_offset()
            )
            self.assertEqual(slots.tolist(), CACHE_INDICES)
            self.assertEqual(window.shape, (B, T, D, 3))
            self.assertEqual(window_rows.tolist(), list(range(B)))
            # the source's call: the retiled kernel, else SGLang's on 128-channel tiles
            self.assertEqual(
                ckw,
                {
                    "block_n": 128,
                    "token_major_out": True,
                    "num_warps": 2,
                    "retile": front_conv.RETILE_DEFAULT,
                },
            )
            (kw,) = env.flushes()
            li = 1
            self.assertEqual(
                kw["initial_state_source"].storage_offset(),
                view.mamba.temporal[li].storage_offset(),
            )
            for name, ring in (
                ("k_cache", "k_ring"),
                ("u_cache", "u_ring"),
                ("g_cache", "g_ring"),
            ):
                self.assertEqual(
                    kw[name].storage_offset(),
                    getattr(view.rings, ring)[li].storage_offset(),
                )
            # b / a: the projection's [b | a] columns, read in place
            self.assertEqual(kw["b"].data_ptr(), gdn.ba.data_ptr())
            self.assertEqual(kw["a"].data_ptr(), gdn.ba[:, HV:].data_ptr())
            fn = kw["fused_norm"]
            self.assertEqual(fn.z.shape, (rows, HV, VD))
            self.assertEqual(fn.z.data_ptr(), gdn.qkvz[:, D:].data_ptr())
            self.assertIs(fn.weight, gdn.norm.weight)
            self.assertEqual(env.calls[-1][1].shape, (rows, HV * VD))
            self.assertEqual(st.ran_ops, [("gdn", "ucache+norm")])

    def test_separate_norm_where_the_fused_one_does_not_apply(self):
        stock = U.Options(verify_norm_retile=False)
        cases = (
            dict(env=dict(rows_per_block=1)),  # B < 3: one row per program
            dict(env=dict(norm_reason="z layout")),
            dict(kw=dict(batch=49)),  # above fused_norm_max_batch
            dict(kw=dict(activation="swish")),
            dict(kw=dict(options=U.Options(fused_norm=False))),
        )
        for case in cases:
            with self.subTest(case=case), _Env(**case.get("env", {})) as env:
                kw = dict(case.get("kw", {}))
                kw["options"] = dataclasses.replace(
                    kw.get("options", stock), verify_norm_retile=False
                )
                out, gdn, st, view = self.run_verify(env, **kw)
                self.assertNotIn("verify_norm", env.names())
                self.assertNotIn("fused_norm", env.flushes()[0])
                ((_, x, z),) = [c for c in env.calls if c[0] == "norm"]
                rows = kw.get("batch", len(CACHE_INDICES)) * T
                self.assertEqual(x.shape, (rows * HV, VD))
                self.assertEqual(z.shape, (rows, HV, VD))  # the strided z view
                self.assertEqual(z.data_ptr(), gdn.qkvz[:, D:].data_ptr())
                self.assertEqual(env.calls[-1][1].shape, (rows, HV * VD))
                self.assertEqual(st.ran_ops, [("gdn", "ucache")])

    def test_verify_norm_launch_where_the_fused_one_does_not_apply(self):
        """SGLANG_ENABLE_GDN_VERIFY_NORM_RETILE: the separate norm is SGLang's kernel on
        16 rows x 8 warps a program wherever SGLang's launch holds >= 2 rows a program."""
        cases = (
            dict(env=dict(norm_reason="z layout")),
            dict(kw=dict(batch=49)),  # above fused_norm_max_batch
            dict(kw=dict(activation="swish")),
            dict(kw=dict(options=U.Options(fused_norm=False))),
            dict(env=dict(rows_per_block=4), kw=dict(batch=64)),
        )
        for case in cases:
            with self.subTest(case=case), _Env(**case.get("env", {})) as env:
                kw = case.get("kw", {})
                out, gdn, st, view = self.run_verify(env, **kw)
                self.assertNotIn("fused_norm", env.flushes()[0])
                self.assertNotIn("norm", env.names())
                ((_, x, weight, bias, eps, nkw),) = [
                    c for c in env.calls if c[0] == "verify_norm"
                ]
                rows = kw.get("batch", len(CACHE_INDICES)) * T
                self.assertEqual(x.shape, (rows * HV, VD))
                self.assertIs(weight, gdn.norm.weight)
                self.assertIsNone(bias)
                self.assertEqual(eps, gdn.norm.eps)
                z = nkw.pop("z")
                self.assertEqual(z.shape, (rows, HV, VD))  # the strided z view
                self.assertEqual(z.data_ptr(), gdn.qkvz[:, D:].data_ptr())
                self.assertEqual(
                    nkw,
                    dict(
                        group_size=None,
                        norm_before_gate=True,
                        is_rms_norm=True,
                        activation=gdn.norm.activation,
                        launch_cfg=(16, 8),
                    ),
                )
                self.assertEqual(env.calls[-1][1].shape, (rows, HV * VD))
                self.assertEqual(st.ran_ops, [("gdn", "ucache")])

    def test_stock_norm_where_the_verify_launch_does_not_apply(self):
        cases = (
            dict(env=dict(rows_per_block=1)),  # one row a program: 4 x 32 lanes a row
            dict(norm=dict(bias=torch.zeros(VD, dtype=BF16))),
            dict(norm=dict(group_size=64)),
            dict(
                kw=dict(options=U.Options(verify_norm_retile=False, fused_norm=False))
            ),
        )
        for case in cases:
            with self.subTest(case=case), _Env(**case.get("env", {})) as env:
                with mock.patch.object(_Norm, "__init__", _norm_init(case)):
                    out, gdn, st, view = self.run_verify(env, **case.get("kw", {}))
                self.assertNotIn("verify_norm", env.names())
                ((_, x, z),) = [c for c in env.calls if c[0] == "norm"]
                self.assertEqual(x.shape, (len(CACHE_INDICES) * T * HV, VD))

    def test_verify_norm_applies(self):
        x = torch.zeros(24, VD, dtype=BF16)
        for rows_per_block, width, applies in (
            (2, VD, True),
            (4, VD, True),
            (1, VD, False),
            (4, 64, False),
        ):
            with mock.patch.object(
                norm_mod, "calc_rows_per_block", lambda rows, device: rows_per_block
            ):
                self.assertIs(
                    norm_mod.gdn_verify_norm_retile_applies(x[:, :width]), applies
                )
        self.assertEqual(norm_mod.GDN_VERIFY_NORM_DEFAULT, (16, 8))

    def test_retiled_conv_on_either_window_layout(self):
        for dedup_window in (True, False):
            with self.subTest(dedup_window=dedup_window), _Env() as env:
                out, gdn, st, view = self.run_verify(env, dedup_window=dedup_window)
                ((_, x, conv_state, slots, window, rows, ckw),) = [
                    c for c in env.calls if c[0] == "front_conv"
                ]
                self.assertEqual(x.data_ptr(), gdn.qkvz.data_ptr())
                self.assertEqual(slots.tolist(), CACHE_INDICES)
                self.assertEqual(window.shape, (len(CACHE_INDICES), T, D, 3))
                self.assertIs(ckw["retile"], front_conv.RETILE_DEFAULT)
                self.assertEqual((ckw["block_n"], ckw["num_warps"]), (128, 2))
                self.assertNotIn("stock_conv", env.names())

    def test_hoisted_conv_without_the_retile(self):
        with _Env() as env:
            self.run_verify(env, options=U.Options(conv_retile=False))
            ((*_, ckw),) = [c for c in env.calls if c[0] == "front_conv"]
            self.assertEqual(
                ckw,
                {
                    "block_n": 256,
                    "token_major_out": True,
                    "hoist": True,
                    "window_mode": 1,
                },
            )

    def test_stock_conv_without_the_hoisted_layout(self):
        for kw in (
            dict(dedup_window=False, options=U.Options(conv_retile=False)),
            dict(options=U.Options(hoisted_conv=False, conv_retile=False)),
        ):
            with self.subTest(kw=kw), _Env() as env:
                out, gdn, st, view = self.run_verify(env, **kw)
                self.assertNotIn("front_conv", env.names())
                ((_, x, conv_state, ckw),) = [
                    c for c in env.calls if c[0] == "stock_conv"
                ]
                self.assertEqual(x.data_ptr(), gdn.qkvz.data_ptr())
                self.assertEqual(ckw["conv_state_indices"].tolist(), CACHE_INDICES)
                self.assertEqual(
                    ckw["intermediate_state_indices"].tolist(),
                    list(range(len(CACHE_INDICES))),
                )
                self.assertEqual(ckw["intermediate_conv_window"].shape, (6, T, D, 3))

    def test_fused_conv(self):
        with _Env() as env:
            out, gdn, st, view = self.run_verify(
                env, options=U.HIGH_CONCURRENCY_FUSED_CONV
            )
            self.assertNotIn("front_conv", env.names())
            self.assertNotIn("stock_conv", env.names())
            (kw,) = env.flushes()
            fc = kw["fused_conv"]
            B = len(CACHE_INDICES)
            self.assertEqual(fc.x.shape, (B, T, D))
            self.assertEqual(fc.x.data_ptr(), gdn.qkvz.data_ptr())
            self.assertIs(fc.weight, gdn.attn.conv_weights)
            self.assertEqual(fc.window.shape, (B, T, D, 3))
            self.assertEqual(fc.window_indices.tolist(), list(range(B)))
            # q / k / v: views of the rows the flush writes the conv into
            self.assertNotEqual(kw["q"].data_ptr(), gdn.qkvz.data_ptr())
            self.assertEqual(kw["k"].data_ptr(), kw["q"].data_ptr() + H * KD * 2)
            self.assertEqual(kw["q"].stride()[:2], (T * D, D))
            self.assertEqual(st.ran_ops, [("gdn", "ucache+conv+norm")])

    def test_fused_conv_kept(self):
        for case in (
            dict(env=dict(conv_reason="2 of 3 value heads per CTA")),
            dict(kw=dict(bias=torch.zeros(D, dtype=BF16))),
        ):
            with self.subTest(case=case), _Env(**case.get("env", {})) as env:
                self.run_verify(
                    env, options=U.HIGH_CONCURRENCY_FUSED_CONV, **case.get("kw", {})
                )
                self.assertIn("front_conv", env.names())
                self.assertNotIn("fused_conv", env.flushes()[0])

    def test_request_rows_and_padding(self):
        with _Env() as env:
            buffers = _buffers(by_request_row=True)
            out, gdn, st, view = self.run_verify(env, buffers=buffers)
            gather = env.calls[2]
            self.assertEqual(gather[0], "gather")
            self.assertEqual(gather[2].tolist(), st.fb.req_pool_indices.tolist())
            self.assertIn("ring_indices", env.flushes()[0])

    def test_rings_passed_by_the_caller(self):
        """Rings passed explicitly (``buffers=``) in place of the executor's."""
        with _Env() as env:
            buffers = _buffers()
            view = _view(buffers)
            uv = _verifier(first_layer_id=7)
            gdn, st = _gdn(env), _St()
            x = torch.zeros(len(CACHE_INDICES) * T, 16)
            executor = SimpleNamespace(
                gdn_ucache=uv, ucache_state=None, smallm_gemm=None
            )
            U.verify(executor, gdn, x, st, view, buffers)
            (kw,) = env.flushes()
            self.assertIs(kw["k_cache"]._base, buffers.k_ring)
            self.assertEqual(kw["flush_min"], buffers.flush_min)

    def test_needs_the_executor_verifier(self):
        with _Env() as env:
            gdn = _gdn(env)
            with self.assertRaisesRegex(RuntimeError, "UcacheVerify"):
                U.verify(
                    SimpleNamespace(gdn_ucache=None, ucache_state=None),
                    gdn,
                    None,
                    _St(),
                    _view(_buffers()),
                )


def _front_conv_inputs(*, tokens=T, dedup_window=True, width=4, dtype=BF16, batch=6):
    x = torch.zeros(batch * tokens, D + HV * VD, dtype=dtype)[:, :D]
    x = x.view(batch, tokens, D).transpose(1, 2)
    if dedup_window:
        phys = torch.zeros(batch, D, tokens + 2, dtype=dtype)
        window = phys.as_strided(
            (batch, tokens, D, 3), ((tokens + 2) * D, 1, tokens + 2, 1)
        )
    else:
        window = torch.zeros(batch, tokens, D, 3, dtype=dtype)
    return dict(
        x=x,
        conv_state=torch.zeros(SLOTS, D, width - 1, dtype=BF16),
        weight=torch.zeros(D, width, dtype=BF16),
        bias=None,
        activation="silu",
        conv_state_indices=torch.tensor(CACHE_INDICES[:batch], dtype=torch.int32),
        intermediate_conv_window=window,
        intermediate_state_indices=torch.arange(batch, dtype=torch.int32),
    )


class TestRetiledConv(unittest.TestCase):
    """gdn_front/conv.py's conv_update with ``retile``: the source's retiled kernel where
    retile_unsupported_reason admits the call, else SGLang's kernel at the caller's tile."""

    def launch(self, inputs, pdl=True, **kw):
        from sglang.kernels.ops.mamba import causal_conv1d_triton as stock

        from sgl_nvidia_mega.qwen3_8 import _jit

        calls = []
        with (
            mock.patch.object(front_conv, "_RETILED", [_Launch("retiled", calls)]),
            mock.patch.object(
                stock, "_causal_conv1d_update_kernel", _Launch("stock", calls)
            ),
            mock.patch.object(_jit, "is_arch_support_pdl", lambda: pdl),
        ):
            args = dict(inputs)
            out = front_conv.conv_update(
                args.pop("x"),
                args.pop("conv_state"),
                args.pop("weight"),
                args.pop("bias"),
                args.pop("activation"),
                args.pop("conv_state_indices"),
                args.pop("intermediate_conv_window"),
                args.pop("intermediate_state_indices"),
                **kw,
            )
        return out, calls

    def test_retiled_launch(self):
        for dedup_window, pdl in ((True, True), (False, True), (True, False)):
            with self.subTest(dedup_window=dedup_window, pdl=pdl):
                inputs = _front_conv_inputs(dedup_window=dedup_window)
                out, calls = self.launch(
                    inputs,
                    pdl=pdl,
                    block_n=128,
                    token_major_out=True,
                    num_warps=2,
                    retile=front_conv.RETILE_DEFAULT,
                )
                ((name, grid, args, kw),) = calls
                self.assertEqual(name, "retiled")
                # one program per (request, 256 channels)
                self.assertEqual(grid, (len(CACHE_INDICES), D // 256))
                self.assertIs(args[0], inputs["x"])
                self.assertIs(args[7], out)
                self.assertEqual(out.transpose(1, 2).stride(), (T * D, D, 1))
                self.assertEqual(args[8:11], (len(CACHE_INDICES), D, T))
                self.assertEqual(
                    args[20:24], inputs["intermediate_conv_window"].stride()
                )
                self.assertEqual(args[-1], -1)  # PAD_SLOT_ID
                self.assertEqual(
                    kw,
                    dict(
                        HAS_BIAS=False,
                        SILU_ACTIVATION=True,
                        BLOCK_N=256,
                        ROWS=1,
                        PREFETCH=2,
                        WINDOW_TILE=dedup_window,
                        NP2_WIN=8,
                        USE_GDC=pdl,
                        num_warps=4,
                        **({"launch_pdl": True} if pdl else {}),
                    ),
                )

    def test_stock_kernel_where_the_retile_does_not_apply(self):
        cases = (
            (dict(tokens=3), "3 verify tokens"),
            (dict(tokens=9), "9 verify tokens"),
            (dict(width=3), "conv width 3"),
            (dict(dtype=torch.float32), "dtypes"),
        )
        for case, reason in cases:
            with self.subTest(case=case):
                inputs = _front_conv_inputs(**case)
                got = front_conv.retile_unsupported_reason(
                    x=inputs["x"],
                    conv_state=inputs["conv_state"],
                    weight=inputs["weight"],
                    intermediate_conv_window=inputs["intermediate_conv_window"],
                )
                self.assertIn(reason, got)
                out, calls = self.launch(
                    inputs,
                    block_n=128,
                    token_major_out=True,
                    num_warps=2,
                    retile=front_conv.RETILE_DEFAULT,
                )
                ((name, grid, args, kw),) = calls
                self.assertEqual(name, "stock")
                self.assertEqual(
                    (kw["BLOCK_N"], kw["num_warps"], kw["SAVE_INTERMEDIATE"]),
                    (128, 2, True),
                )
        window5 = _front_conv_inputs(dedup_window=False)
        window5["intermediate_conv_window"] = window5["intermediate_conv_window"][None]
        self.assertIn(
            "window of 5 dims",
            front_conv.retile_unsupported_reason(
                x=window5["x"],
                conv_state=window5["conv_state"],
                weight=window5["weight"],
                intermediate_conv_window=window5["intermediate_conv_window"],
            ),
        )

    def test_no_retile_without_the_config_or_token_major_out(self):
        inputs = _front_conv_inputs()
        for kw in (
            dict(block_n=128, token_major_out=True, num_warps=2),
            dict(block_n=128, num_warps=2, retile=front_conv.RETILE_DEFAULT),
        ):
            with self.subTest(kw=kw):
                _, calls = self.launch(inputs, **kw)
                self.assertEqual([c[0] for c in calls], ["stock"])

    def test_default_config(self):
        self.assertEqual(
            front_conv.RETILE_DEFAULT,
            front_conv.RetileConfig(block_n=256, num_warps=4, rows=1, prefetch=2),
        )


class TestReasons(unittest.TestCase):
    def test_pool_reasons(self):
        with _Env((10, 7)):
            self.assertEqual(U.pool_reasons(_buffers(), T), "")
            self.assertEqual(U.pool_reasons(_buffers(by_request_row=True), T), "")
            self.assertEqual(U.pool_reasons(_buffers(k_dtype=torch.float16), T), "")
            self.assertIn("no u-cache rings", U.pool_reasons(None, T))
            self.assertIn("verify rows", U.pool_reasons(_buffers(), 9))
            self.assertIn(
                "fold threshold", U.pool_reasons(_buffers(), 5)
            )  # 13 > 16 - 5 + 1
            state = torch.zeros(1, dtype=BF16)
            self.assertIn("fp16", U.pool_reasons(_buffers(), T, state=state))
            bad = _buffers()
            bad.k_ring = torch.zeros(2, 8, H, 16, KD, dtype=BF16)
            self.assertIn("ring shapes", U.pool_reasons(bad, T))
        with _Env((10, 0)):
            self.assertEqual(U.pool_reasons(_buffers(), T), "")
            self.assertIn("SM107", U.pool_reasons(_buffers(k_dtype=torch.float16), T))
            self.assertIn("SM107", U.pool_reasons(_buffers(by_request_row=True), T))

    def test_build(self):
        def model(*gdns):
            layers = [_LinearLayer(g) for g in gdns] + [SimpleNamespace()]
            return SimpleNamespace(
                model=SimpleNamespace(
                    layers=layers, start_layer=0, end_layer=len(layers)
                )
            )

        with _Env((10, 7)) as env:
            good = _gdn(env)
            self.assertEqual(U.build(model(good, good), "cpu"), "")
            narrow = _gdn(env)
            narrow.attn = SimpleNamespace(**{**vars(narrow.attn), "head_k_dim": 64})
            self.assertIn("head dims", U.build(model(good, narrow), "cpu"))
            wide = _gdn(env)
            wide.attn = SimpleNamespace(**{**vars(wide.attn), "num_v_heads": 10})
            self.assertIn("value heads per key head", U.build(model(wide), "cpu"))
            ratio = _gdn(env)
            ratio.num_v_heads = 5 * ratio.num_k_heads
            self.assertIn("contiguous", U.build(model(ratio), "cpu"))
        with _Env((9, 0)):
            self.assertIn("SM100, SM103 or SM107", U.build(model(), "cpu"))
        with _Env((10, 3), flashinfer=False):
            self.assertIn("does not load", U.build(model(), "cpu"))


_VENDORED = {
    "__init__.py": "e8f5a8d3c7f78304ea5c4224f625179afec98f3c08d2a89ff17fef1e74abd691",
    "_umma_helpers_rubin.py": "ad741aeef225c9995fbee9dbbefec48a346646a0a8c0af9f2c1e9551e9c5856a",
    "device_target.py": "7e8390150de1d6e2348161145110dec292c3a8aec68e43f016287c7cddda8b17",
    "gdn_replay_mtp_umma_rubin.py": "a675d78a43dbbb464b3424ea12f8ea01b236dc67626e546f6e12cc93bffb8564",
}


_INTERPRETED = r"""
import json
from types import SimpleNamespace
import torch
from sgl_nvidia_mega.qwen3_8.kernels.ops.attention.linear import gdn_ucache_verify as K

out = {}
hist = torch.tensor([0, 5, 13, 2, 9, 7, 1, 4], dtype=torch.int32)
base = torch.tensor([0, 3, 31, 6, 2, 1, 0, 9], dtype=torch.int32)
slots = torch.tensor([3, 0, 5, -1, 1, 6], dtype=torch.int32)
rows = K.gather_gdn_ucache_rows(
    slots=slots, buffers=SimpleNamespace(hist_len=hist, cache_base=base, by_request_row=False)
)
out["slot"] = [rows.slots.tolist(), rows.hist_len.tolist(), rows.cache_base.tolist()]
request = torch.tensor([2, 0, 4, 7, 1, 3], dtype=torch.int64)
rows = K.gather_gdn_ucache_rows(
    slots=slots,
    buffers=SimpleNamespace(hist_len=hist, cache_base=base, by_request_row=True),
    request_rows=request,
)
out["request"] = [
    rows.slots.tolist(), rows.hist_len.tolist(), rows.cache_base.tolist(), rows.ring_rows.tolist()
]
order = K._flush_first_order(
    hist_len=torch.tensor([3, 13, 0, 14, 2, 13], dtype=torch.int32), flush_min=13
)
out["order"] = order.tolist()
output = torch.ones(6, 4, 3, 5, dtype=torch.bfloat16)
K._zero_padded_rows_kernel[(6, 1)](rows.slots, output, 60, BLOCK=4096)
out["zero"] = [float(output[i].abs().sum()) for i in range(6)]
src = torch.randn(3, 16).to(torch.bfloat16)
q, k, v = (torch.empty(3, n, dtype=torch.bfloat16) for n in (2, 3, 4))
K._split_qkv_kernel[(3, 1)](src[:, :9], q, k, v, 16, Q_COLS=2, K_COLS=3, V_COLS=4, BLOCK=1024)
out["split"] = [torch.equal(q, src[:, :2]), torch.equal(k, src[:, 2:5]), torch.equal(v, src[:, 5:9])]
srcs = [torch.randn(48).to(torch.bfloat16) for _ in range(2)]
dsts = [torch.zeros(48) for _ in range(2)]
K._refresh_gate_params_kernel[(2,)](
    torch.tensor([t.data_ptr() for t in srcs], dtype=torch.int64),
    torch.tensor([t.data_ptr() for t in dsts], dtype=torch.int64),
    48,
    BLOCK=64,
)
out["refresh"] = [torch.equal(d, s.float()) for s, d in zip(srcs, dsts)]
print(json.dumps(out))
"""


_INTERPRETED_CONV = r"""
import json
import torch
from sgl_nvidia_mega.qwen3_8.kernels.ops.attention.gdn_front import conv as C

# fp32 (the interpreter's bf16 arithmetic is not the GPU's); slot -1 is padding
batch, D = 6, 1280
slots = torch.tensor([3, -1, 5, -1, 1, 6], dtype=torch.int32)
rows = torch.arange(batch, dtype=torch.int32)
out = {}
for T in (4, 6, 8):
    for dedup in (True, False):
        g = torch.Generator().manual_seed(T)
        proj = torch.randn(batch * T, D + 768, generator=g)
        x = proj[:, :D].view(batch, T, D).transpose(1, 2)
        w = torch.randn(D, 4, generator=g) * 0.5
        cs = torch.randn(8 + batch, D, 3, generator=g)
        if dedup:
            phys = torch.zeros(batch, D, T + 2)
            win = phys.as_strided((batch, T, D, 3), ((T + 2) * D, 1, T + 2, 1))
        else:
            phys = win = torch.zeros(batch, T, D, 3)
        cs0 = cs.clone()
        y = C.conv_update(
            x, cs, w, None, "silu", slots, win, rows, block_n=128,
            token_major_out=True, num_warps=2, retile=C.RETILE_DEFAULT,
        )
        ok = []
        for r, s in enumerate(slots.tolist()):
            if s < 0:
                ok.append(phys[r].abs().sum().item() == 0)  # padded rows write nothing
                continue
            seq = torch.cat([cs0[s], x[r]], dim=1)  # c0 c1 c2 x_0 .. x_{T-1}
            acc = sum(seq[:, k : k + T] * w[:, k : k + 1] for k in range(4))
            ok.append(torch.allclose(y[r], acc / (1 + torch.exp(-acc)), rtol=1e-6, atol=1e-6))
            ok.append(torch.equal(cs[s], x[r][:, T - 3 :]))  # the new conv state
            want = seq[:, 1:] if dedup else torch.stack([seq[:, t + 1 : t + 4] for t in range(T)])
            ok.append(torch.equal(phys[r], want))  # the step windows
        live = [s for s in slots.tolist() if s >= 0]
        other = [i for i in range(cs.shape[0]) if i not in live]
        ok.append(torch.equal(cs[other], cs0[other]))
        out[f"{T} {dedup}"] = all(ok)
print(json.dumps(out))
"""


class TestInterpretedKernels(unittest.TestCase):
    """The ported Triton kernels (byte-identical to the source) on CPU."""

    def test_cursor_gather_order_zero_split_refresh(self):
        env = dict(os.environ, TRITON_INTERPRET="1", PYTHONDONTWRITEBYTECODE="1")
        done = subprocess.run(
            [sys.executable, "-c", _INTERPRETED],
            env=env,
            capture_output=True,
            text=True,
            timeout=600,
        )
        self.assertEqual(done.returncode, 0, done.stderr[-2000:])
        got = json.loads(done.stdout.strip().splitlines()[-1])
        # slot <= 0 is padding: slot -1, zero cursors
        self.assertEqual(
            got["slot"], [[3, -1, 5, -1, 1, 6], [2, 0, 7, 0, 5, 1], [6, 0, 1, 0, 3, 0]]
        )
        # by request row: liveness from the slot, cursors and ring row from the request row
        self.assertEqual(
            got["request"],
            [
                [3, -1, 5, -1, 1, 6],
                [13, 0, 9, 0, 5, 2],
                [31, 0, 2, 0, 3, 6],
                [2, 0, 4, 0, 1, 3],
            ],
        )
        # a stable partition: folding rows (hist_len >= flush_min) first
        self.assertEqual(got["order"], [1, 3, 5, 0, 2, 4])
        self.assertEqual(
            [x == 0 for x in got["zero"]], [False, True, False, True, False, False]
        )
        self.assertEqual(got["split"], [True, True, True])
        self.assertEqual(got["refresh"], [True, True])

    def test_retiled_conv(self):
        """The ported retiled verify conv against a direct reference: outputs, the new
        conv state, both window layouts, padded rows, 4 / 6 / 8 verify tokens."""
        env = dict(os.environ, TRITON_INTERPRET="1", PYTHONDONTWRITEBYTECODE="1")
        done = subprocess.run(
            [sys.executable, "-c", _INTERPRETED_CONV],
            env=env,
            capture_output=True,
            text=True,
            timeout=600,
        )
        self.assertEqual(done.returncode, 0, done.stderr[-2000:])
        got = json.loads(done.stdout.strip().splitlines()[-1])
        self.assertEqual(len(got), 6)
        self.assertTrue(all(got.values()), got)


if __name__ == "__main__":
    unittest.main()
