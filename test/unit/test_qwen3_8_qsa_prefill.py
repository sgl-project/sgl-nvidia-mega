# SPDX-License-Identifier: Apache-2.0
"""The Qwen3.8 executor's QSA prefill attention (runtime/qsa_prefill.py) on CPU: when
it declines a step, the order of its pieces, which kernel each piece routes to and the
stock fallbacks, the MTP seed rows from host lengths, and the recipe options. SGLang's
modules are replaced by fakes and the kernel launches by spies, so no GPU, no kernel
build and no model weights are needed."""

import contextlib
import enum
import sys
import types
import unittest
from types import SimpleNamespace
from unittest import mock

import torch

# The plugin modules that import SGLang at load, before the fakes replace it
from sgl_nvidia_mega.qwen3_8.qsa import (  # noqa: F401
    pool_decode,
    prefill_pack,
    prefill_scale_fold,
    prefill_select,
    tma_prefill,
    union_prefill,
)
from sgl_nvidia_mega.qwen3_8.runtime import prefill_graph as pg
from sgl_nvidia_mega.qwen3_8.runtime import qsa_prefill as qp


class _Mode(enum.Enum):
    EXTEND = 1
    MIXED = 2
    DECODE = 3
    TARGET_VERIFY = 4
    DRAFT_EXTEND_V2 = 5

    def is_extend_or_draft_extend_or_mixed(self):
        return self in (_Mode.EXTEND, _Mode.MIXED)

    def is_draft_extend_v2(self):
        return self is _Mode.DRAFT_EXTEND_V2


class _CudaLike(torch.Tensor):
    """A CPU tensor that reports CUDA, for the host-side routing on CPU."""

    @property
    def is_cuda(self):
        return True


def _cuda(t):
    return t.as_subclass(_CudaLike)


class _Backend:  # the fake QwenSparseAttnBackend
    def __init__(self, rows=8, heads=1, dim=4, slots=64):
        self.calls = []
        self.max_context_len = 32768
        self.compress_ratio = 4
        self.capture = False
        kv = SimpleNamespace(
            key=lambda: torch.randn(slots, heads, dim),
            value=lambda: torch.randn(slots, heads, dim),
        )
        self.token_to_kv_pool = SimpleNamespace(
            layer_router=SimpleNamespace(read=lambda layer_id: kv)
        )
        self.req_to_token_pool = SimpleNamespace(
            req_to_token=torch.arange(4 * slots, dtype=torch.int32).view(4, slots)
        )
        backend = self

        class Layer:
            def store(self, layer, loc, k, v):
                backend.calls.append(("store", loc, tuple(k.shape), tuple(v.shape)))

        self.kv_layers = SimpleNamespace(layer=lambda layer_id: Layer())
        self._mtp_shared_sparse_indices = SimpleNamespace(
            capture=lambda *a: self.calls.append(("state.capture",) + a)
        )

    def should_capture_mtp_sparse_indices(self, fb):
        return self.capture

    def capture_mtp_sparse_indices(self, topk, fb, layer_id, metadata=None):
        self.calls.append(("capture", layer_id))

    @staticmethod
    def _pad_extend_output(output, num_rows):
        # QwenSparseAttnBackend._pad_extend_output
        output = output.reshape(output.shape[0], -1)
        if output.shape[0] == num_rows:
            return output
        padded = output.new_zeros((num_rows, output.shape[1]))
        padded[: output.shape[0]].copy_(output)
        return padded


_STATE = {}


def _chunk_rows(rows, keys, heads):  # qsa_indexer._qsa_prefill_row_chunk_size
    if rows <= 0 or keys <= 0:
        return max(rows, 1)
    block_q = max(1, 128 // heads)
    max_padded = max(block_q, 128 * 1024 * 1024 // (keys * 4))
    max_padded = max(block_q, max_padded // block_q * block_q)
    return min(rows, max_padded)


def _install(test):
    """SGLang's modules qsa_prefill imports, as fakes, restored after the test."""
    _STATE.update(
        breakable=False,
        cp=False,
        cp_active=False,
        invariant=False,
        padded=None,
        backend=_Backend(),
        md=None,
        sm100=True,
        tile=(16, 1, 2),
    )
    calls = _STATE["calls"] = []

    def fused_qk(q_gate, k, qw, kw_, cache, positions, eps, qh, kh, dim, rot, **kw):
        calls.append(("fused_qk", kw["has_gate"], kw["mrope_axis_map"]))
        tokens = q_gate.shape[0]
        return (
            torch.zeros(tokens, qh * dim),
            torch.zeros(tokens, kh * dim),
            torch.zeros(tokens, qh, dim),
        )

    def sigmoid_mul(x, gate, inplace=True):
        calls.append(("sigmoid_mul", gate is None))
        return x

    def cold(q, k, v, max_k, indices, cu, scale):
        calls.append(("cold", q.shape[0], max_k))
        return q + 1

    def chunk(q, k, v, indices, cu_q, cu_k, kv_lens, scale, max_q=None):
        calls.append(("chunk", q.shape[0], max_q))
        return q + 2

    def pack(**kw):
        calls.append(("stock_pack", kw["total_k"], kw["max_k"]))
        return kw["k"][: kw["total_k"]], kw["v"][: kw["total_k"]]

    def fast_topk(logits, starts, lengths, topk):
        calls.append(("fast_topk", logits.shape[0], topk))
        return torch.zeros(logits.shape[0], topk, dtype=torch.int32)

    def expand(blocks, positions, lengths, compress_ratio, token_topk):
        return torch.full(
            (blocks.shape[0], token_topk + compress_ratio - 1), 3, dtype=torch.int32
        )

    def h2d(values, dtype, device):
        calls.append(("async_h2d", list(values)))
        return torch.tensor(values, dtype=dtype, device=device)

    fakes = {
        "sglang.srt.layers.attention.qsa.glue": dict(
            resolve_qsa_sparse_backend=lambda backend: backend,
            get_qsa_indexer_metadata=lambda backend, layer_id, fb: _STATE["md"],
        ),
        "sglang.srt.model_executor.forward_context": dict(
            get_attn_backend=lambda: _STATE["backend"]
        ),
        "sglang.srt.batch_invariant_ops": dict(
            is_batch_invariant_mode_enabled=lambda: _STATE["invariant"]
        ),
        "sglang.srt.layers.attention.graph_utils": dict(
            padded_extend_real_tokens=lambda rows, fb: _STATE["padded"]
        ),
        "sglang.srt.layers.attention.qwen_sparse_attn_backend": dict(
            QwenSparseAttnBackend=_Backend
        ),
        "sglang.srt.layers.cp.utils": dict(is_cp_active=lambda fb: _STATE["cp_active"]),
        "sglang.srt.model_executor.runner_backend_utils.breakable_cuda_graph": dict(
            is_in_breakable_cuda_graph=lambda: _STATE["breakable"]
        ),
        "sglang.srt.models.qwen4_exp": dict(
            _shards_rows_over_cp=lambda fb: _STATE["cp"],
            Qwen4ExpAttentionDecoderLayer=SimpleNamespace,
        ),
        "sglang.kernels.ops.attention.fused_qk_rmsnorm_rope_gate": dict(
            fused_qk_gemma_rmsnorm_rope_gate=fused_qk
        ),
        "sglang.kernels.ops.elementwise.elementwise": dict(
            fused_sigmoid_mul=sigmoid_mul
        ),
        "sglang.srt.layers.attention.qsa.mqa": dict(HAS_TILELANG=True),
        "sglang.srt.layers.attention.qsa.kernel": dict(
            qsa_fast_topk=fast_topk, expand_qsa_block_indices=expand
        ),
        "sglang.srt.layers.attention.qsa.qsa_indexer": dict(
            _qsa_prefill_row_chunk_size=_chunk_rows
        ),
        "sglang.srt.layers.attention.qsa.sparse_attn": dict(
            sparse_gqa_fwd_interface_triton=cold,
            sparse_gqa_fwd_interface_triton_ck=chunk,
            pack_qsa_prefill_kv=pack,
            _get_best_config=lambda rows: _STATE["tile"],
        ),
    }
    import sglang.srt.utils as sglang_utils  # the real package; the slot imports these from it

    for name, value in (
        ("async_h2d", h2d),
        ("is_sm100_supported", lambda: _STATE["sm100"]),
    ):
        patcher = mock.patch.object(sglang_utils, name, value)
        patcher.start()
        test.addCleanup(patcher.stop)
    for name, attrs in fakes.items():
        module = types.ModuleType(name)
        module.__dict__.update(attrs)
        old = sys.modules.get(name)
        sys.modules[name] = module
        test.addCleanup(_restore, name, old)


def _restore(name, old):
    if old is None:
        sys.modules.pop(name, None)
    else:
        sys.modules[name] = old


class _St:
    def __init__(self, fb, positions=None):
        self.fb = fb
        self.positions = positions
        self.ran_ops = []

    def ran(self, op, impl):
        self.ran_ops.append((op, impl))


def _fb(mode=_Mode.EXTEND, extend=(5, 3), prefix=(0, 0)):
    seq = [e + p for e, p in zip(extend, prefix)]
    rows = sum(extend)
    return SimpleNamespace(
        forward_mode=mode,
        attn_cp_metadata=None,
        extend_seq_lens=torch.tensor(extend, dtype=torch.int32),
        extend_seq_lens_cpu=list(extend),
        seq_lens_cpu=seq,
        req_pool_indices=torch.arange(1, len(extend) + 1, dtype=torch.int64),
        req_pool_indices_cpu=torch.arange(1, len(extend) + 1, dtype=torch.int64),
        out_cache_loc=torch.arange(rows, dtype=torch.int64) + 100,
        positions=torch.cat([torch.arange(p, p + e) for e, p in zip(extend, prefix)]),
        _attn_output=None,
    )


def _attn(qh=2, kh=1, dim=4):
    return SimpleNamespace(
        layer_id=3,
        tp_q_head_num=qh,
        tp_k_head_num=kh,
        tp_v_head_num=kh,
        qk_head_dim=dim,
        v_head_dim=dim,
        head_dim=dim,
        scaling=0.5,
    )


class TestOptions(unittest.TestCase):
    def test_recipe_values(self):
        o = qp.DEFAULT
        self.assertIs(o, qp.HIGH_CONCURRENCY)
        self.assertEqual(
            (o.union, o.union_out_alias, o.pack_rows, o.host_path, o.scale_fold),
            (True, True, True, True, True),
        )
        self.assertEqual(
            (o.qk_rows, o.fast_select, o.fast_select_topk), (True, True, "onepass")
        )
        self.assertEqual(
            (o.scorer_tc, o.wide_logits_mib, o.wide_min_keys), (True, 256, 24576)
        )
        self.assertEqual((o.tma_prefill, o.tma_stages), (True, 3))
        mid = qp.MID_CONCURRENCY
        self.assertEqual(
            (mid.pack_rows, mid.qk_rows, mid.scorer_tc, mid.wide_logits_mib),
            (False, False, False, 0),
        )
        self.assertTrue(
            mid.union and mid.host_path and mid.scale_fold and mid.fast_select
        )

    def test_invalid(self):
        for bad in (
            dict(fast_select_topk="half"),
            dict(scale_fold=False),  # fast_select needs it
            dict(wide_logits_mib=4096),
            dict(wide_min_keys=1000),
            dict(tma_stages=0),
        ):
            with self.assertRaises(ValueError, msg=bad):
                qp.Options(**bad)
        qp.Options(scale_fold=False, fast_select=False)


class TestDecline(unittest.TestCase):
    def setUp(self):
        _install(self)
        self.layer = SimpleNamespace(is_qsa=True)
        self.x = SimpleNamespace(is_cuda=True, shape=(8, 16))

    def reason(self, fb=None, layer=None, x=None, backend=None):
        return qp.decline_reason(
            layer or self.layer, x or self.x, fb or _fb(), backend or _STATE["backend"]
        )

    def test_served(self):
        self.assertEqual(self.reason(), "")
        self.assertEqual(self.reason(_fb(_Mode.MIXED)), "")

    def test_reasons(self):
        self.assertIn("QSA", self.reason(layer=SimpleNamespace(is_qsa=False)))
        for mode in (_Mode.DECODE, _Mode.TARGET_VERIFY, _Mode.DRAFT_EXTEND_V2):
            self.assertIn("extend", self.reason(_fb(mode)))
        _STATE["breakable"] = True
        self.assertIn("capture", self.reason())
        _STATE.update(breakable=False, cp=True)
        self.assertIn("context", self.reason())
        _STATE.update(cp=False, cp_active=True)
        fb = _fb()
        fb.attn_cp_metadata = object()
        self.assertIn("context", self.reason(fb))
        _STATE.update(cp_active=False, invariant=True)
        self.assertIn("invariant", self.reason())
        _STATE["invariant"] = False
        self.assertIn("CUDA", self.reason(x=SimpleNamespace(is_cuda=False, shape=(8,))))
        fb = _fb()
        fb.seq_lens_cpu = None
        self.assertIn("host", self.reason(fb))
        _STATE["padded"] = 7
        self.assertIn("padded", self.reason())
        _STATE["padded"] = None
        self.assertIn("backend", self.reason(backend=object()))

    def test_graph_capture_leaves_the_batch_checks_to_its_break(self):
        _STATE.update(breakable=True, padded=7)
        fb = _fb()
        fb.seq_lens_cpu = None
        reason = lambda f: qp.decline_reason(
            self.layer, self.x, f, _STATE["backend"], graph=True
        )
        self.assertEqual(reason(fb), "")
        _STATE["cp"] = True
        self.assertIn("context", reason(fb))
        _STATE["cp"] = False
        self.assertIn("extend", reason(_fb(_Mode.DECODE)))


class TestExtend(unittest.TestCase):
    def setUp(self):
        _install(self)

    def test_breakable_graph_runs_indexer_and_attention_as_a_break(self):
        _STATE.update(breakable=True, padded=7)  # bucket rows past the batch's
        layer = SimpleNamespace(is_qsa=True, attn=_attn())
        st = _St(_fb(), positions="pos")
        ex = SimpleNamespace(qsa_prefill=qp.DEFAULT)
        x = SimpleNamespace(is_cuda=True, shape=(8, 16))
        seen = []
        with (
            mock.patch.object(qp, "qkv_gate", lambda *a: ("q", "k", "v", "gate")),
            mock.patch.object(qp, "attend", lambda *a: self.fail("eager attend")),
            mock.patch.object(pg, "qsa", lambda *a: seen.append(a) or "attn_rows"),
            mock.patch.object(qp, "finish", lambda layer_, out, gate: (out, gate)),
        ):
            self.assertEqual(qp.extend(ex, layer, x, st), ("attn_rows", "gate"))
        self.assertEqual(seen, [(ex, layer, x, "q", "k", "v", st)])

    def test_break_runs_on_the_live_batch_rows(self):
        fb = _fb(extend=(3, 2))  # 5 tokens of an 8-row bucket
        fb.global_num_token_non_padded_cpu = 5
        fb.out_cache_loc = torch.arange(8) + 100
        live, scopes = _St(fb), []

        @contextlib.contextmanager
        def scope(fb_, output, query_tokens, kv_tokens, kwargs):
            fb_._attn_output = output[:query_tokens]
            fb_.out_cache_loc = fb_.out_cache_loc[:query_tokens]
            scopes.append((query_tokens, kv_tokens))
            yield kwargs
            fb_._attn_output, fb_.out_cache_loc = None, torch.arange(8) + 100

        def attend(options, layer, x, q, k, v, backend, st):
            self.assertEqual((options, backend), (qp.DEFAULT, _STATE["backend"]))
            self.assertEqual(tuple(st.positions.shape), (5,))
            self.assertEqual(tuple(fb.out_cache_loc.shape), (5,))
            self.assertEqual([t.shape[0] for t in (x, q, k, v)], [5] * 4)
            return torch.ones(5, 8)

        graph_utils = sys.modules["sglang.srt.layers.attention.graph_utils"]
        graph_utils.attention_input_scope = scope
        output = torch.full((8, 8), 7.0)
        rows = [torch.zeros(8, 4) for _ in range(4)]
        ex = SimpleNamespace(qsa_prefill=qp.DEFAULT)
        with (
            mock.patch.object(pg, "_live", lambda executor: live),
            mock.patch.object(qp, "attend", attend),
        ):
            pg._qsa(ex, SimpleNamespace(), rows[0], torch.arange(8), *rows[1:], output)
        self.assertEqual(scopes, [(5, 5)])
        self.assertTrue(torch.equal(output[:5], torch.ones(5, 8)))
        self.assertTrue(torch.equal(output[5:], torch.zeros(3, 8)))
        # the union kernel writing the output rows in place: no copy
        output.fill_(7.0)
        with (
            mock.patch.object(pg, "_live", lambda executor: live),
            mock.patch.object(qp, "attend", lambda *a: fb._attn_output.fill_(2.0)),
        ):
            pg._qsa(ex, SimpleNamespace(), rows[0], torch.arange(8), *rows[1:], output)
        self.assertTrue(torch.equal(output[:5], torch.full((5, 8), 2.0)))
        self.assertTrue(torch.equal(output[5:], torch.zeros(3, 8)))

    def test_off_or_declined_launches_nothing(self):
        layer = SimpleNamespace(is_qsa=True)
        st = _St(_fb())
        spy = mock.Mock()
        with mock.patch.object(qp, "qkv_gate", spy):
            self.assertIsNone(
                qp.extend(SimpleNamespace(qsa_prefill=None), layer, None, st)
            )
            ex = SimpleNamespace(qsa_prefill=qp.DEFAULT)
            x = torch.zeros(8, 16)  # a CPU input: declined
            self.assertIsNone(qp.extend(ex, layer, x, st))
        spy.assert_not_called()
        self.assertEqual(st.ran_ops, [])

    def test_pieces_in_self_attention_order(self):
        order = []
        layer = SimpleNamespace(is_qsa=True, attn=_attn())
        st = _St(_fb(), positions="pos")
        ex = SimpleNamespace(qsa_prefill=qp.DEFAULT)

        def qkv(options, layer_, x, positions, st_):
            order.append(("qkv", positions))
            return "q", "k", "v", "gate"

        def index(options, layer_, x, positions, backend, st_):
            order.append(("indexer", backend is _STATE["backend"]))
            return "topk"

        def attend(options, attn, backend, q, k, v, topk, st_):
            order.append(("attention", attn is layer.attn, (q, k, v, topk)))
            return "attn"

        def finish(layer_, out, gate):
            order.append(("finish", out, gate))
            return "out"

        with (
            mock.patch.object(qp, "decline_reason", lambda *a, **k: ""),
            mock.patch.object(qp, "qkv_gate", qkv),
            mock.patch.object(qp, "indexer", index),
            mock.patch.object(qp, "attention", attend),
            mock.patch.object(qp, "finish", finish),
        ):
            self.assertEqual(qp.extend(ex, layer, "x", st), "out")
        self.assertEqual(
            order,
            [
                ("qkv", "pos"),
                ("indexer", True),
                ("attention", True, ("q", "k", "v", "topk")),
                ("finish", "attn", "gate"),
            ],
        )


def _qkv_layer(gate=True, heads=2, kv_heads=1, dim=256):
    calls = []
    q_size, kv_size = heads * dim, kv_heads * dim

    def proj(x):
        calls.append("qkv_proj")
        return torch.zeros(x.shape[0], 2 * q_size + 2 * kv_size), None

    layer = SimpleNamespace(
        attn_output_gate=gate,
        qkv_proj=proj,
        q_size=q_size,
        kv_size=kv_size,
        num_heads=heads,
        num_kv_heads=kv_heads,
        head_dim=dim,
        rotary_emb=SimpleNamespace(
            axis_map=torch.zeros(32, dtype=torch.int32),
            cos_sin_cache=torch.zeros(16, 64),
            rotary_dim=64,
        ),
        q_norm=SimpleNamespace(weight=torch.zeros(dim), variance_epsilon=1e-6),
        k_norm=SimpleNamespace(weight=torch.zeros(dim)),
        calls=calls,
    )
    layer._prepare_qkv_gate = lambda **kw: calls.append("prepare") or "stock"
    return layer


class TestQkvGate(unittest.TestCase):
    def setUp(self):
        _install(self)

    def test_off_or_ungated_runs_prepare(self):
        for options, gate in ((qp.MID_CONCURRENCY, True), (qp.DEFAULT, False)):
            layer = _qkv_layer(gate=gate)
            st = _St(_fb())
            out = qp.qkv_gate(options, layer, torch.zeros(4, 8), torch.arange(4), st)
            self.assertEqual(out, "stock")
            self.assertEqual(layer.calls, ["prepare"])
            self.assertEqual(st.ran_ops, [("qsa.qkv", "stock")])

    def test_stock_fused_call_below_the_rows_kernel(self):
        layer = _qkv_layer()
        st = _St(_fb())
        q, k, v, gate = qp.qkv_gate(
            qp.DEFAULT, layer, torch.zeros(4, 8), torch.arange(4), st
        )
        self.assertEqual(_STATE["calls"], [("fused_qk", True, None)])
        self.assertEqual(
            (q.shape, k.shape, v.shape, gate.shape),
            ((4, 512), (4, 256), (4, 256), (4, 512)),
        )
        self.assertEqual(st.ran_ops, [("qsa.qkv", "stock")])

    def test_rows_kernel_where_it_applies(self):
        from sgl_nvidia_mega.qwen3_8.kernels.ops.attention import (
            fused_qk_rmsnorm_rope_gate_rows as rows,
        )

        launched = []
        layer = _qkv_layer()
        st = _St(_fb())
        with (
            mock.patch.object(rows, "applies", lambda *a: True),
            mock.patch.object(rows, "launch", lambda **kw: launched.append(kw)),
        ):
            q, k, v, gate = qp.qkv_gate(
                qp.DEFAULT, layer, torch.zeros(4, 8), torch.arange(4), st
            )
            # M-RoPE positions that are not [3, T] contiguous keep the stock function
            mrope = torch.zeros(4, 3, dtype=torch.int64).T
            qp.qkv_gate(qp.DEFAULT, layer, torch.zeros(4, 8), mrope, st)
        self.assertEqual(len(launched), 1)
        kw = launched[0]
        self.assertEqual(kw["q_out"].shape, (4, 512))
        self.assertEqual(kw["k_out"].shape, (4, 256))
        self.assertEqual(kw["gate_out"].shape, (4, 2, 256))
        self.assertIsNone(kw["mrope_axis_map"])
        self.assertEqual((kw["num_q_heads"], kw["num_kv_heads"]), (2, 1))
        self.assertEqual(gate.shape, (4, 512))
        self.assertEqual(st.ran_ops, [("qsa.qkv", "qk_rows"), ("qsa.qkv", "stock")])
        (call,) = _STATE["calls"]
        self.assertEqual(call[:2], ("fused_qk", True))
        self.assertIs(call[2], layer.rotary_emb.axis_map)


def _metadata(rows, keys=64, max_row_keys=40, page=16):
    md = SimpleNamespace(
        pending_ring_slots=None,
        compress_member_rows=torch.zeros(1),
        token_to_kv_pool=SimpleNamespace(qsa_compressed_page_size=page),
        is_cuda_graph=False,
        req_pool_indices=torch.tensor([7, 9], dtype=torch.int64),
        calls=[],
    )
    md.get_token_to_batch_idx = lambda: torch.tensor([0] * 5 + [1] * (rows - 5))
    md.get_prefill_mqa_inputs = lambda layer_id, logical: (
        torch.zeros(keys, 1, 128, dtype=torch.bfloat16),
        torch.zeros(rows, dtype=torch.int32),
        torch.full((rows,), 30, dtype=torch.int32),
        torch.tensor([20, 40], dtype=torch.int32),
    )
    md.prefill_max_row_keys = lambda host: md.calls.append(host) or max_row_keys
    md.get_seqlens_expanded = lambda: torch.arange(rows) + 1
    return md


def _indexer(heads=4):
    ix = SimpleNamespace(
        layer_id=3,
        index_n_heads=heads,
        index_kv_heads=1,
        index_head_dim=128,
        compress_ratio=4,
        block_topk=512,
        token_topk=2048,
        calls=[],
    )

    def project(hidden, positions, pool=None, cache_loc=None, q_heads_padded=None):
        ix.calls.append(("project_qk", hidden.shape[0], q_heads_padded, cache_loc))
        q = torch.zeros(hidden.shape[0], heads, 128, dtype=torch.bfloat16)
        return _cuda(q), "token_k", True

    def update(token_k, logical, positions, md, state_slots=None, state_stored=False):
        ix.calls.append(("update", logical.tolist(), state_stored))

    def stock(q, keys, starts, ends, positions, lengths, max_row_keys=None):
        ix.calls.append(("stock_select", max_row_keys))
        return torch.full((q.shape[0], 2051), -5, dtype=torch.int32)

    ix.project_qk = project
    ix.update_key_state_and_compress = update
    ix._pending_ring_slots = lambda md, logical, is_extend: torch.arange(
        logical.numel()
    )
    ix.select_prefill_tokens = stock
    return ix


class TestIndexer(unittest.TestCase):
    def setUp(self):
        _install(self)
        self.fb = _fb(extend=(5, 3), prefix=(10, 20))
        _STATE["md"] = _metadata(8)
        self.ix = _indexer()
        self.layer = SimpleNamespace(layer_id=3, indexer=self.ix, stock=[])
        self.layer._compute_qsa_topk_indices = lambda x, positions, fb: (
            self.layer.stock.append(x.shape[0]) or "stock"
        )

    def run_indexer(self, options, rows=8):
        st = _St(self.fb)
        out = qp.indexer(
            options,
            self.layer,
            torch.zeros(rows, 16),
            torch.arange(rows),
            _STATE["backend"],
            st,
        )
        return out, st.ran_ops

    def test_both_off_is_stock(self):
        options = qp.Options(fast_select=False, scale_fold=False)
        self.assertEqual(
            self.run_indexer(options), ("stock", [("qsa.indexer", "stock")])
        )
        self.assertEqual(self.layer.stock, [8])

    def test_prologue_then_fast_select(self):
        selected = []
        with mock.patch.object(
            qp, "fast_select", lambda *a: selected.append(a) or a[7].fill_(4)
        ):
            out, ran = self.run_indexer(qp.DEFAULT)
        self.assertEqual(ran, [("qsa.indexer", "fast_select")])
        self.assertTrue(bool((out == 4).all()))
        self.assertEqual(tuple(out.shape), (8, 2051))
        # SGLang's prologue: logical positions from fb.positions, no head padding
        logical = self.fb.positions.tolist()
        self.assertEqual(self.ix.calls[0][:3], ("project_qk", 8, None))
        self.assertEqual(self.ix.calls[1], ("update", logical, True))
        self.assertEqual(_STATE["md"].calls, [self.fb.seq_lens_cpu])
        self.assertEqual(selected[0][-1], 40)  # the host width

    def test_scale_fold_where_fast_select_declines(self):
        scored = []

        def relative(q, keys, starts, ends, width, score_scale=None, *, scale_fold):
            scored.append((q.shape[0], width, scale_fold))
            return torch.zeros(q.shape[0], width)

        # wider than the 8,192 compressed keys of a 32K context: no fast select
        _STATE["md"] = _metadata(8, max_row_keys=9000)
        with mock.patch.object(
            prefill_scale_fold, "tilelang_qsa_mqa_prefill_relative", relative
        ):
            out, ran = self.run_indexer(qp.DEFAULT)
        self.assertEqual(ran, [("qsa.indexer", "scale_fold")])
        self.assertEqual(scored, [(8, 9000, True)])
        self.assertTrue(bool((out == 3).all()))
        self.assertEqual([c[0] for c in _STATE["calls"]], ["fast_topk"] * len(scored))

    def test_absolute_scoring_is_stock(self):
        _STATE["md"] = _metadata(8, max_row_keys=None)
        out, ran = self.run_indexer(qp.DEFAULT)
        self.assertEqual(ran, [("qsa.indexer", "stock")])
        self.assertEqual(self.ix.calls[-1], ("stock_select", None))

    def test_mtp_seed_capture(self):
        backend = _STATE["backend"]
        backend.capture = True
        with mock.patch.object(qp, "fast_select", lambda *a: None):
            self.run_indexer(qp.DEFAULT)
            self.run_indexer(qp.Options(host_path=False))
        host = [c for c in backend.calls if c[0] == "state.capture"]
        self.assertEqual(len(host), 1)
        _, topk, req_rows, lens, layer_id = host[0]
        # rows 0..4 belong to pool id 7, rows 5..7 to pool id 9: rows 4 and 7 are last
        self.assertEqual(req_rows.tolist(), [7, 9])
        self.assertEqual(lens.tolist(), [5, 8])
        self.assertEqual(layer_id, 3)
        self.assertIn(("async_h2d", [4, 7]), _STATE["calls"])
        self.assertIn(("capture", 3), backend.calls)


class TestHostAnchorRows(unittest.TestCase):
    """The host seed rows equal capture_mtp_sparse_indices' device ``nonzero``."""

    @staticmethod
    def device_rows(extend_lens, pool_ids, num_rows):
        row_to_req = torch.repeat_interleave(
            torch.arange(len(extend_lens)), torch.tensor(extend_lens)
        )
        row_pool = torch.tensor(pool_ids)[row_to_req[:num_rows]]
        is_last = torch.ones_like(row_pool, dtype=torch.bool)
        if row_pool.numel() > 1:
            is_last[:-1] = row_pool[:-1] != row_pool[1:]
        return is_last.nonzero().flatten().tolist()

    def test_matches_the_device_rule(self):
        gen = torch.Generator().manual_seed(5)
        for _ in range(200):
            n = int(torch.randint(1, 6, (1,), generator=gen))
            lens = torch.randint(0, 5, (n,), generator=gen).tolist()
            ids = torch.randint(1, 4, (n,), generator=gen).tolist()
            total = sum(lens)
            if total == 0:
                continue
            rows = int(torch.randint(1, total + 1, (1,), generator=gen))
            self.assertEqual(
                qp._last_rows_of_runs(extend_lens=lens, pool_ids=ids, num_rows=rows),
                self.device_rows(lens, ids, rows),
                (lens, ids, rows),
            )

    def test_misaligned_host_lists_use_the_device(self):
        fb = _fb(extend=(5, 3))
        md = _metadata(8)
        self.assertEqual(qp.host_anchor_rows(fb, md, 8), [4, 7])
        fb.extend_seq_lens_cpu = [5]
        self.assertIsNone(qp.host_anchor_rows(fb, md, 8))
        fb = _fb(extend=(5, 3))
        fb.req_pool_indices_cpu = None
        self.assertIsNone(qp.host_anchor_rows(fb, md, 8))


class TestFastSelect(unittest.TestCase):
    def setUp(self):
        _install(self)

    def test_capacity(self):
        self.assertEqual(qp.compressed_capacity(32768, 16), 8192)
        self.assertEqual(qp.compressed_capacity(1001, 16), 256)
        self.assertEqual(qp.compressed_capacity(1001, 64), 256)
        self.assertEqual(qp.compressed_capacity(0, 16), 0)
        self.assertEqual(qp.compressed_capacity(1024, 0), 0)

    def test_admission(self):
        ix = _indexer()
        q = torch.zeros(8, 4, 128, dtype=torch.bfloat16)
        keys = torch.zeros(64, 1, 128, dtype=torch.bfloat16)
        out = torch.empty(8, 2051, dtype=torch.int32)
        self.assertEqual(qp.fast_select_reason(ix, q, keys, out, 40, 8192), "")
        self.assertIn("capacity", qp.fast_select_reason(ix, q, keys, out, 40, 0))
        self.assertIn("capacity", qp.fast_select_reason(ix, q, keys, out, 40, 131088))
        self.assertIn("width", qp.fast_select_reason(ix, q, keys, out, 8193, 8192))
        self.assertIn("width", qp.fast_select_reason(ix, q, keys, out, 40.0, 8192))
        self.assertIn(
            "geometry", qp.fast_select_reason(_indexer(8), q, keys, out, 40, 8192)
        )
        self.assertIn(
            "query", qp.fast_select_reason(ix, q.float(), keys, out, 40, 8192)
        )
        self.assertIn("key", qp.fast_select_reason(ix, q, keys[:0], out, 40, 8192))
        self.assertIn("output", qp.fast_select_reason(ix, q, keys, out[:4], 40, 8192))

    def test_budget(self):
        self.assertEqual(qp.logits_budget(qp.DEFAULT, 24575), 128 << 20)
        self.assertEqual(qp.logits_budget(qp.DEFAULT, 24576), 256 << 20)
        self.assertEqual(qp.logits_budget(qp.MID_CONCURRENCY, 10**6), 128 << 20)

    def test_io_follows_the_source_plan(self):
        raw = prefill_select
        seen = []
        q = torch.zeros(1000, 4, 128, dtype=torch.bfloat16)
        keys = torch.zeros(64, 1, 128, dtype=torch.bfloat16)
        out = torch.empty(1000, 2051, dtype=torch.int32)
        i64 = torch.zeros(1000, dtype=torch.int64)
        with (
            mock.patch.object(pool_decode, "_num_sms", lambda device: 148),
            mock.patch.object(
                raw, "select_into", lambda io, width, **kw: seen.append((io, width, kw))
            ),
        ):
            qp.fast_select(qp.DEFAULT, q, keys, i64, i64 + 5, i64, i64, out, 30000)
        io, width, kw = seen[0]
        self.assertEqual(width, 30000)
        self.assertEqual(kw, dict(scorer_tc=True, topk_impl="onepass"))
        self.assertEqual(
            io.chunk_rows, raw.plan_chunk_rows(1000, 30000, 296, 256 << 20)
        )
        self.assertEqual(io.scorer_sms, 148)
        padded = (io.chunk_rows + 31) // 32 * 32
        self.assertEqual(tuple(io.logits_out.shape), (padded, 30000))
        self.assertEqual(tuple(io.query_scratch_out.shape), (padded, 4, 128))
        self.assertEqual(tuple(io.block_indices_out.shape), (1000, 512))
        self.assertEqual(io.row_starts.dtype, torch.int32)
        self.assertTrue(bool((io.row_lengths == 5).all()))
        self.assertIs(io.selected_indices_out, out)


class TestAttention(unittest.TestCase):
    def setUp(self):
        _install(self)
        self.union = union_prefill
        self.pack = prefill_pack

    def run_attention(self, options, fb, valid_rows=None, union=False, pack=None):
        rows = sum(fb.extend_seq_lens_cpu)
        valid_rows = rows if valid_rows is None else valid_rows
        q = torch.randn(rows, 2 * 4)
        k = torch.randn(rows, 4)
        v = torch.randn(rows, 4)
        topk = torch.zeros(valid_rows, 7, dtype=torch.int64)
        st = _St(fb)
        unions = []

        def run_union(*args, **kw):
            unions.append((args[0].shape[0], kw["extend_lens"], kw["kv_max"]))
            return args[0] * 0 + 9

        with (
            mock.patch.object(self.union, "applies", lambda *a, **kw: union),
            mock.patch.object(self.union, "kv_ok", lambda k, v: True),
            mock.patch.object(self.union, "log_once", lambda *a: None),
            mock.patch.object(self.union, "union_prefill", run_union),
            mock.patch.object(self.pack, "pack_rows", lambda **kw: pack),
        ):
            out = qp.attention(options, _attn(), _STATE["backend"], q, k, v, topk, st)
        return out, q, st.ran_ops, unions

    def test_cold_pass_stock_and_dp_padding(self):
        fb = _fb(extend=(5, 3))
        out, q, ran, unions = self.run_attention(qp.DEFAULT, fb, valid_rows=6)
        self.assertEqual(_STATE["backend"].calls[0][0], "store")
        self.assertTrue(torch.equal(_STATE["backend"].calls[0][1], fb.out_cache_loc))
        self.assertEqual(_STATE["calls"], [("cold", 6, 5)])
        self.assertEqual(out.shape, (8, 8))
        self.assertTrue(torch.equal(out[:6], q[:6] + 1))
        self.assertTrue(bool((out[6:] == 0).all()))
        self.assertEqual(ran, [("qsa", "stock")])
        self.assertEqual(unions, [])

    def test_cold_pass_union(self):
        fb = _fb(extend=(5, 3))
        out, q, ran, unions = self.run_attention(qp.DEFAULT, fb, union=True)
        self.assertEqual(unions, [(8, [5, 3], 5)])
        self.assertTrue(bool((out == 9).all()))
        self.assertEqual(ran, [("qsa", "union")])

    def test_prefix_pass_union_after_the_pack(self):
        fb = _fb(extend=(5, 3), prefix=(10, 20))
        packed = (torch.randn(38, 1, 4), torch.randn(38, 1, 4))
        out, q, ran, unions = self.run_attention(
            qp.DEFAULT, fb, union=True, pack=packed
        )
        self.assertEqual(unions, [(8, [5, 3], 23)])
        self.assertEqual(ran, [("qsa.pack", "pack_rows"), ("qsa", "union")])
        self.assertIn(("async_h2d", [15, 23]), _STATE["calls"])

    def test_prefix_pass_tma_then_stock(self):
        fb = _fb(extend=(5, 3), prefix=(10, 20))
        launched = []
        with (
            mock.patch.object(qp, "tma_reason", lambda *a: ""),
            mock.patch.object(
                tma_prefill,
                "launch",
                lambda io, max_q, stages: (
                    launched.append((max_q, stages)) or io.output.fill_(6)
                ),
            ),
        ):
            out, q, ran, _ = self.run_attention(qp.DEFAULT, fb)
        self.assertEqual(launched, [(5, 3)])
        self.assertTrue(bool((out == 6).all()))
        self.assertEqual(ran, [("qsa.pack", "stock"), ("qsa", "tma_prefill")])
        self.assertEqual(_STATE["calls"][-1], ("stock_pack", 38, 23))
        # without TMA: the stock chunk kernel with the host max query rows
        _STATE["calls"].clear()
        out, q, ran, _ = self.run_attention(qp.Options(tma_prefill=False), fb)
        self.assertEqual(_STATE["calls"][-1], ("chunk", 8, 5))
        self.assertTrue(torch.equal(out, q + 2))
        # without the host path: no pinned staging and the kernel reads max_q itself
        _STATE["calls"].clear()
        options = qp.Options(tma_prefill=False, host_path=False, union=False)
        self.run_attention(options, fb)
        self.assertEqual(_STATE["calls"][-1], ("chunk", 8, None))
        self.assertNotIn("async_h2d", [c[0] for c in _STATE["calls"]])


class TestTmaReason(unittest.TestCase):
    def setUp(self):
        _install(self)

    def args(self, rows=8, heads=2, kv_heads=1, selected=2051):
        q = torch.zeros(rows, heads, 256, dtype=torch.bfloat16)
        k = torch.zeros(40, kv_heads, 256, dtype=torch.bfloat16)
        indices = torch.zeros(rows, selected, dtype=torch.int32)
        cu = torch.tensor([0, 5, 8], dtype=torch.int32)
        lens = torch.tensor([15, 23], dtype=torch.int32)
        return q, k, k.clone(), indices, cu, cu.clone(), lens, [5, 3], _fb(), 4

    def test_admits_the_stock_tile_only(self):
        self.assertEqual(qp.tma_reason(*self.args()), "")
        _STATE["tile"] = (32, 4, 2)
        self.assertIn("tile", qp.tma_reason(*self.args()))
        _STATE.update(tile=(16, 1, 2), sm100=False)
        self.assertIn("SM100", qp.tma_reason(*self.args()))
        _STATE["sm100"] = True
        self.assertIn("indices", qp.tma_reason(*self.args(selected=2048)))
        self.assertIn("heads", qp.tma_reason(*self.args(heads=34, kv_heads=2)))


class TestUnionOutput(unittest.TestCase):
    def test_takes_only_an_exact_buffer(self):
        q = torch.zeros(6, 2, 4, dtype=torch.bfloat16)
        fb = SimpleNamespace(_attn_output=None)
        self.assertIsNone(qp.union_output(q, fb, 6))
        fb._attn_output = torch.empty(6, 8, dtype=torch.bfloat16)
        out = qp.union_output(q, fb, 6)
        self.assertEqual(out.shape, q.shape)
        self.assertEqual(out.data_ptr(), fb._attn_output.data_ptr())
        self.assertIsNone(qp.union_output(q, fb, 7))
        fb._attn_output = torch.empty(6, 8)
        self.assertIsNone(qp.union_output(q, fb, 6))


class TestBuild(unittest.TestCase):
    def test_refusals(self):
        self.assertEqual(qp.build(None, None), "no options")
        if not torch.cuda.is_available():
            self.assertIn("SM10x", qp.build(None))


if __name__ == "__main__":
    unittest.main()
