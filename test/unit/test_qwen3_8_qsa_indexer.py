# SPDX-License-Identifier: Apache-2.0
"""The Qwen3.8 executor's QSA indexer on CPU: which kernel each mode and width runs,
the stock fallbacks, the MTP hooks' order and the build refusals. SGLang's modules
are replaced by fakes and the kernel launches by spies, so no GPU, no kernel build
and no model weights are needed."""

import enum
import os
import sys
import types
import unittest
from types import SimpleNamespace
from unittest import mock

import torch

from sgl_nvidia_mega.qwen3_8.runtime import qsa_indexer as qi

STOCK_OUT = torch.full((1, 1), -7, dtype=torch.int32)
WIDTH = 8192  # compressed keys of the graph page table (a 32K context)


class _Mode(enum.Enum):
    EXTEND = 1
    DECODE = 2
    TARGET_VERIFY = 3
    DRAFT_EXTEND_V2 = 4

    def is_decode(self):
        return self is _Mode.DECODE

    def is_target_verify(self):
        return self is _Mode.TARGET_VERIFY

    def is_draft_extend_v2(self):
        return self is _Mode.DRAFT_EXTEND_V2


class _Attention:  # the fake Qwen4ExpAttentionDecoderLayer
    pass


class _Linear:
    pass


_STATE = {}


def _install(test):
    """SGLang's modules this module imports, as fakes, restored after the test."""
    _STATE.update(breakable=False, cp=False, invariant=False, tp=1, backend=None)
    calls = _STATE["calls"] = []

    def stock_score(q, cache, table, lengths, width, min_scored_len=0):
        calls.append(("qsa_mqa_decode", q.shape[0], width, min_scored_len))
        return torch.zeros(q.shape[0], width)

    def stock_select(scores, lengths, topk, row_starts=None):
        calls.append(("fast_topk", scores.shape[0], topk, row_starts))
        return torch.zeros(scores.shape[0], topk, dtype=torch.int32)

    def expand(blocks, positions, lengths, compress_ratio, token_topk):
        calls.append(("expand", blocks, positions, lengths, compress_ratio, token_topk))
        return torch.zeros(blocks.shape[0], token_topk + compress_ratio - 1)

    def metadata(backend, layer_id, fb):
        backend.calls.append(("metadata", layer_id))
        return backend.md

    fakes = {
        "sglang.srt.batch_invariant_ops": dict(
            is_batch_invariant_mode_enabled=lambda: _STATE["invariant"]
        ),
        "sglang.srt.layers.attention.qsa.glue": dict(
            get_qsa_indexer_metadata=metadata,
            resolve_qsa_sparse_backend=lambda backend: backend,
        ),
        "sglang.srt.model_executor.forward_context": dict(
            get_attn_backend=lambda: _STATE["backend"]
        ),
        "sglang.srt.model_executor.runner_backend_utils.breakable_cuda_graph": dict(
            is_in_breakable_cuda_graph=lambda: _STATE["breakable"]
        ),
        "sglang.srt.models.qwen4_exp": dict(
            _shards_rows_over_cp=lambda fb: _STATE["cp"],
            Qwen4ExpAttentionDecoderLayer=_Attention,
            Qwen4ExpLinearDecoderLayer=_Linear,
        ),
        "sglang.srt.layers.attention.qsa.mqa": dict(qsa_mqa_decode=stock_score),
        "sglang.kernels.ops.attention.fast_topk": dict(fast_topk=stock_select),
        "sglang.srt.layers.attention.qsa.kernel": dict(expand_qsa_block_indices=expand),
        "sglang.srt.runtime_context": dict(
            get_parallel=lambda: SimpleNamespace(tp_size=_STATE["tp"])
        ),
    }
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


class _Backend:
    """The resolved QSA backend's MTP hooks, recording their order."""

    def __init__(self, md, reuse=False, capture=False):
        self.md, self.reuse, self.capture = md, reuse, capture
        self.calls = []

    def set_mtp_shared_sparse_indices(self, state):
        pass

    def should_reuse_mtp_sparse_indices(self, fb):
        self.calls.append("should_reuse")
        return self.reuse

    def should_capture_mtp_sparse_indices(self, fb):
        self.calls.append("should_capture")
        return self.capture

    def capture_mtp_sparse_indices(self, topk, fb, layer_id, metadata=None):
        self.calls.append(("capture", tuple(topk.shape), layer_id, metadata is self.md))


def _metadata(rows, *, graph=True, defer=True, cache_dtype=torch.bfloat16, page=16):
    pool = SimpleNamespace(
        index_state_dtype=torch.bfloat16, qsa_compressed_dtype=cache_dtype
    )
    md = SimpleNamespace(
        is_cuda_graph=graph,
        token_to_kv_pool=pool,
        req_pool_indices=torch.arange(1, rows + 1, dtype=torch.int32),
        decode_logical_positions=torch.arange(rows, dtype=torch.int32) + 5000,
        pending_ring_slots=torch.arange(rows, dtype=torch.int64) + 8,
        graph_ring_group_locs=torch.zeros((rows, 4), dtype=torch.int32),
        graph_write_locs=torch.zeros(rows, dtype=torch.int32),
        defer_block_expansion=defer,
        seqlens=torch.arange(rows, dtype=torch.int32) + 5001,
    )
    inputs = (
        torch.zeros((2, page, 1, 128), dtype=cache_dtype),
        torch.zeros((rows, WIDTH // page), dtype=torch.int32),
        torch.full((rows,), 1300, dtype=torch.int32),
        WIDTH,
    )
    md.get_token_to_batch_idx = lambda: torch.arange(rows, dtype=torch.int32)
    md.get_seqlens_int32 = lambda: md.seqlens
    md.get_decode_mqa_inputs = lambda layer_id: inputs
    return md


def _indexer(heads=4, fused=True):
    ix = SimpleNamespace(
        layer_id=7,
        index_n_heads=heads,
        index_kv_heads=1,
        index_head_dim=128,
        compress_ratio=4,
        block_topk=512,
        token_topk=2048,
        rotary_emb=SimpleNamespace(is_neox_style=True, rotary_dim=64),
        calls=[],
    )
    ix._use_fused_prep = lambda tensor: fused
    ix._use_fused_compress = lambda pool: fused

    def project_qk(hidden, positions, pool=None, cache_loc=None, q_heads_padded=None):
        ix.calls.append(("project_qk", hidden.shape[0], q_heads_padded, cache_loc))
        rows = hidden.shape[0]
        return torch.zeros(rows, q_heads_padded, 128), torch.zeros(rows, 1, 128), True

    def update(token_k, logical, positions, md, state_slots=None, state_stored=False):
        ix.calls.append(("update", logical.shape[0], state_slots, state_stored))

    ix.project_qk = project_qk
    ix.update_key_state_and_compress = update
    return ix


def _layer(ix, is_qsa=True):
    layer = SimpleNamespace(is_qsa=is_qsa, layer_id=7, indexer=ix, stock=[])

    def stock(hidden, positions, fb):
        layer.stock.append(hidden.shape[0])
        return STOCK_OUT

    layer._compute_qsa_topk_indices = stock
    return layer


class _St:
    def __init__(self, fb):
        self.fb = fb
        self.record = []

    def ran(self, op, impl):
        self.record.append((op, impl))


def _kernels(**options):
    """IndexerKernels with spies for every launch."""
    k = qi.IndexerKernels(**options)
    k.launches = []

    def fold(indexer, md, hidden, positions, slots, qk=None):
        k.launches.append(("fold", hidden.shape[0], positions.shape, slots))
        k.projected = qk
        return torch.zeros(hidden.shape[0], 8, 128, dtype=torch.bfloat16)

    def score_kernel(variant, rows, heads, device, request_dtype):
        def launch(*args):
            k.launches.append(("score", variant, heads, request_dtype, args))

        return launch

    k._fold = fold
    k._score_kernel = score_kernel
    k._split_select = lambda s, l, o: k.launches.append(("select", "v1", s.shape))
    k._verify_select = lambda s, l, o: k.launches.append(("select", "v2", s.shape))
    return k


class _Case(unittest.TestCase):
    def setUp(self):
        _install(self)

    def run_select(
        self,
        mode,
        rows,
        *,
        requests=None,
        tokens=4,
        kernels="default",
        md=None,
        ix=None,
        reuse=False,
        capture=False,
        hidden_rows=None,
        glue=None,
        ahead=None,
    ):
        md = md or _metadata(rows)
        backend = _STATE["backend"] = _Backend(md, reuse=reuse, capture=capture)
        if requests is None:
            requests = rows if mode is _Mode.DECODE else max(rows // 4, 1)
        fb = SimpleNamespace(
            forward_mode=mode,
            req_pool_indices=torch.arange(requests),
            spec_info=SimpleNamespace(draft_token_num=tokens),
        )
        layer = _layer(ix or _indexer())
        k = _kernels() if kernels == "default" else kernels
        st = _St(fb)
        hidden = torch.zeros(hidden_rows or rows, 640, dtype=torch.bfloat16)
        positions = torch.zeros(3, hidden_rows or rows, dtype=torch.int64)
        ex = SimpleNamespace(qsa_indexer=k, qsa_draft=glue)
        step = qk = None
        if ahead is not None:  # PDL index: prepared and projected by the caller
            step = k.prepare(layer, hidden, positions, fb)
            qk = ahead(step)
        out = qi.select(ex, layer, hidden, positions, st, step, qk)
        return SimpleNamespace(
            out=out, st=st, layer=layer, backend=backend, k=k, md=md, qk=qk
        )

    def assert_stock(self, r):
        self.assertIs(r.out, STOCK_OUT)
        self.assertEqual(r.st.record, [(qi.OP, qi.STOCK)])
        self.assertEqual(len(r.layer.stock), 1)
        if r.k is not None:
            self.assertEqual(r.k.launches, [])


def _choose(mode, rows, requests=None, tokens=4, heads=4, sm107=True, **kw):
    options = qi.Options(**kw.pop("options", {}))
    args = dict(
        mode=mode,
        rows=rows,
        requests=requests if requests is not None else max(rows // 4, 1),
        tokens_per_request=tokens,
        heads=heads,
        scored=True,
        capacity=WIDTH,
        row_requests=True,
        sm107=sm107,
    )
    args.update(kw)
    return qi.choose(options, **args)


class TestDispatch(unittest.TestCase):
    """The per-mode and per-width table (``choose``)."""

    def test_target_verify(self):
        for rows in range(4, 65, 4):
            self.assertEqual(_choose("target_verify", rows), ("v4", "v1"), rows)
        for rows in range(68, 577, 4):
            self.assertEqual(_choose("target_verify", rows), ("v7", "v2"), rows)
        for rows in (580, 640, 1024):
            self.assertEqual(_choose("target_verify", rows), ("stock", "v2"), rows)
        # grouped4 is qualified on SM107 only, and up to its request count
        self.assertEqual(_choose("target_verify", 128, sm107=False), ("stock", "v2"))
        capped = dict(grouped_max_requests=16)
        self.assertEqual(_choose("target_verify", 68, options=capped), ("stock", "v2"))
        stages0 = dict(grouped4_stages=0)
        self.assertEqual(_choose("target_verify", 128, options=stages0), ("v6", "v2"))
        # grouped4 scores four logical heads only
        self.assertEqual(_choose("target_verify", 128, heads=8), ("stock", "v2"))
        self.assertEqual(_choose("target_verify", 16, heads=16), ("v4", "v1"))

    def test_target_verify_without_four_row_requests(self):
        self.assertEqual(_choose("target_verify", 12, tokens=3), ("v5", "v1"))
        self.assertEqual(_choose("target_verify", 96, tokens=3), ("stock", "v2"))
        self.assertEqual(_choose("target_verify", 96, tokens=1), ("stock", "stock"))

    def test_draft_extend(self):
        # one 4-row request: the MK scorer (v3 with the short-row variant)
        self.assertEqual(_choose("draft_extend", 4, requests=1), ("v0", "v1"))
        short = dict(short32=True)
        self.assertEqual(
            _choose("draft_extend", 4, requests=1, options=short), ("v3", "v1")
        )
        no_mk = dict(mk_score=False)
        self.assertEqual(
            _choose("draft_extend", 4, requests=1, options=no_mk), ("v5", "v1")
        )
        self.assertEqual(
            _choose("draft_extend", 4, requests=1, row_requests=False), ("v5", "v1")
        )
        self.assertEqual(_choose("draft_extend", 4, requests=1, heads=8), ("v5", "v1"))
        self.assertEqual(_choose("draft_extend", 32, requests=8), ("v5", "v1"))
        self.assertEqual(_choose("draft_extend", 128, requests=32), ("v7", "v2"))
        self.assertEqual(_choose("draft_extend", 1024, requests=256), ("stock", "v2"))
        # rows that are not four a request do not group
        self.assertEqual(_choose("draft_extend", 128, requests=40), ("stock", "stock"))
        no_union = dict(draft_extend_union=False)
        self.assertEqual(
            _choose("draft_extend", 128, requests=32, options=no_union), ("v7", "stock")
        )
        ungrouped = dict(grouped_draft_extend=False)
        self.assertEqual(
            _choose("draft_extend", 128, requests=32, options=ungrouped),
            ("stock", "stock"),
        )

    def test_decode(self):
        for rows in range(1, 65):
            self.assertEqual(_choose("decode", rows, requests=rows), ("v5", "v1"))
        for rows in (65, 68, 128, 1024):
            self.assertEqual(
                _choose("decode", rows, requests=rows), ("stock", "stock"), rows
            )

    def test_switches_and_layout(self):
        self.assertEqual(
            _choose("target_verify", 16, options=dict(score=False)), ("stock", "v1")
        )
        self.assertEqual(
            _choose("target_verify", 16, options=dict(select=False)), ("v4", "stock")
        )
        # an FP8 or non-16-key-page cache scores stock; selection still runs
        self.assertEqual(_choose("target_verify", 16, scored=False), ("stock", "v1"))
        self.assertEqual(_choose("target_verify", 128, scored=False), ("stock", "v2"))
        for capacity in (0, 131072 + 16):
            self.assertEqual(
                _choose("target_verify", 16, capacity=capacity), ("stock", "stock")
            )
        self.assertEqual(_choose("target_verify", 16, capacity=131072), ("v4", "v1"))

    def test_options_validate(self):
        with self.assertRaises(ValueError):
            qi.Options(grouped4_stages=1)
        with self.assertRaises(ValueError):
            qi.Options(grouped4_ctas=0)

    def test_environment_is_not_read(self):
        knobs = {
            "SGLANG_QSA_TOPK_THREADS": "512",
            "SGLANG_QSA_TOPK_CLUSTER": "2",
            "SGLANG_QSA_SCORER_V2": "0",
            "SGLANG_QSA_SCORER_V2_ROWS": "1",
            "SGLANG_QSA_SCORER_V2_TARGET": "64",
            "SGLANG_QSA_SPLIT_TOPK": "0",
            "SGLANG_QSA_QPF": "0",
            "SGLANG_NVIDIA_MEGAKERNEL_QWEN3_8_SCORE_SHORT32": "1",
            "SGLANG_QSA_GROUPED4_VERIFY_CHAIN_CTAS": "512",
        }
        cases = [("target_verify", r) for r in (16, 128, 1024)] + [
            ("decode", 8),
            ("draft_extend", 4),
        ]
        before = [_choose(m, r, requests=1 if r == 4 else None) for m, r in cases]
        configs = [qi.split_config(r, w) for r in (1, 16, 64) for w in (512, 65536)]
        with mock.patch.dict(os.environ, knobs):
            after = [_choose(m, r, requests=1 if r == 4 else None) for m, r in cases]
            self.assertEqual(after, before)
            self.assertEqual(
                [qi.split_config(r, w) for r in (1, 16, 64) for w in (512, 65536)],
                configs,
            )
            self.assertEqual(qi.Options(), qi.Options(short32=False))
            self.assertEqual(qi.grouped_ctas(16), 1024)

    def test_split_config_is_the_sources_default(self):
        from sgl_nvidia_mega.qwen3_8.qsa import split_topk

        unset = {"SGLANG_QSA_TOPK_THREADS": "", "SGLANG_QSA_TOPK_CLUSTER": ""}
        with mock.patch.dict(os.environ, unset):
            for rows in range(1, 65):
                for width in (16, 512, 4096, 8192, 32768, 65536, 131072):
                    self.assertEqual(
                        qi.split_config(rows, width),
                        split_topk._config(rows, width),
                        (rows, width),
                    )
                    _, cluster, iters = qi.split_config(rows, width)
                    self.assertTrue(1 <= cluster <= 16 and 1 <= iters <= 8)

    def test_grouped_ctas(self):
        # 4096 CTAs over rows / 4 groups, on the levels 16 .. 1024
        for rows, ctas in (
            (4, 1024),
            (16, 1024),
            (20, 512),
            (32, 512),
            (36, 256),
            (64, 256),
        ):
            self.assertEqual(qi.grouped_ctas(rows), ctas, rows)

    def test_compressed_capacity(self):
        self.assertEqual(qi.compressed_capacity(1), 16)
        self.assertEqual(qi.compressed_capacity(32768), 8192)
        self.assertEqual(qi.compressed_capacity(65536 + 1), 16384 + 16)


class TestSelect(_Case):
    """``select`` end to end with spies: launches, the result layout, engagement."""

    def test_verify_runs_the_kernels_and_keeps_block_indices(self):
        r = self.run_select(_Mode.TARGET_VERIFY, 16)
        self.assertEqual(r.st.record, [(qi.OP, "fold+score_v4+select_v1")])
        self.assertEqual(r.layer.stock, [])
        fold, score, select = r.k.launches
        self.assertEqual(fold[:3], ("fold", 16, (3, 16)))
        self.assertTrue(torch.equal(fold[3], r.md.pending_ring_slots[:16]))
        self.assertEqual(score[:4], ("score", "v4", 8, None))
        q, cache, table, lengths, scores, scale = score[4]
        self.assertEqual(tuple(q.shape), (16, 1, 8, 128))
        self.assertEqual(tuple(scores.shape), (16, WIDTH))
        self.assertAlmostEqual(scale, 128**0.5)
        self.assertEqual(select, ("select", "v1", (16, WIDTH)))
        # the backend defers expansion: [rows, 512] block indices
        self.assertEqual((tuple(r.out.shape), r.out.dtype), ((16, 512), torch.int32))
        self.assertEqual(_STATE["calls"], [])
        self.assertEqual(
            r.backend.calls, ["should_reuse", ("metadata", 7), "should_capture"]
        )
        self.assertEqual(r.k._warm, {("v4", 16, WIDTH), ("v1", 16, WIDTH)})

    def test_projected_qk_from_the_caller(self):
        """The PDL-index branch: the caller's step is not prepared again and its projection
        feeds the fold (or SGLang's q_prep without it); nothing projects here."""
        ahead = lambda step: torch.ones(step.rows, 640, dtype=torch.bfloat16)
        r = self.run_select(_Mode.TARGET_VERIFY, 8, ahead=ahead)
        self.assertEqual(r.st.record, [(qi.OP, "fold+score_v4+select_v1")])
        self.assertIs(r.k.projected, r.qk)
        # one prepare (the caller's): one metadata read
        self.assertEqual(
            r.backend.calls, ["should_reuse", ("metadata", 7), "should_capture"]
        )
        projected = []
        ix = _indexer()
        with mock.patch.object(
            qi,
            "_project_qk",
            lambda *a, **kw: (
                projected.append(kw["projected_qk"])
                or (torch.zeros(8, 8, 128), torch.zeros(8, 1, 128), True)
            ),
        ):
            r = self.run_select(
                _Mode.TARGET_VERIFY,
                8,
                kernels=_kernels(prologue_fold=False),
                ix=ix,
                ahead=ahead,
            )
        self.assertEqual(r.st.record, [(qi.OP, "stock+score_v4+select_v1")])
        self.assertEqual(len(projected), 1)
        self.assertIs(projected[0], r.qk)
        # without a caller's projection the fold projects itself
        r = self.run_select(_Mode.TARGET_VERIFY, 8)
        self.assertIsNone(r.k.projected)

    def test_draft_extend_seed_capture_on_slot_qsa_draft(self):
        from unittest import mock

        from sgl_nvidia_mega.qwen3_8.runtime import qsa_draft

        k = _kernels()
        k.sm107 = True
        md = _metadata(128, defer=False)
        seen = []
        glue = SimpleNamespace(capture=lambda *a: seen.append(a[1:]) or True)
        r = self.run_select(
            _Mode.DRAFT_EXTEND_V2, 128, kernels=k, md=md, capture=True, glue=glue
        )
        self.assertEqual(
            r.st.record,
            [("qsa.mtp_capture", "batched"), (qi.OP, "fold+score_v7+select_v2")],
        )
        ((topk, fb, metadata, layer_id),) = seen
        self.assertIs(topk, r.out)
        self.assertIs(metadata, md)
        self.assertNotIn(
            "capture", [c[0] for c in r.backend.calls if isinstance(c, tuple)]
        )
        # where the plugin's indexer declines, SGLang's indexer then the glue's capture
        with mock.patch.object(qsa_draft, "indices", return_value="glue") as indices:
            r = self.run_select(_Mode.DRAFT_EXTEND_V2, 128, kernels=None, glue=glue)
        self.assertEqual(r.out, "glue")
        indices.assert_called_once()
        self.assertEqual(r.st.record, [(qi.OP, qi.STOCK)])

    def test_draft_extend_expands_then_captures(self):
        k = _kernels()
        k.sm107 = True
        md = _metadata(128, defer=False)
        r = self.run_select(_Mode.DRAFT_EXTEND_V2, 128, kernels=k, md=md, capture=True)
        self.assertEqual(
            r.st.record,
            [("qsa.mtp_capture", "stock"), (qi.OP, "fold+score_v7+select_v2")],
        )
        self.assertEqual(
            [x[:2] for x in k.launches[1:]], [("score", "v7"), ("select", "v2")]
        )
        (call,) = _STATE["calls"]
        self.assertEqual(call[0], "expand")
        self.assertEqual(tuple(call[1].shape), (128, 512))
        self.assertTrue(torch.equal(call[2], md.decode_logical_positions[:128]))
        self.assertIs(call[3], md.seqlens)
        self.assertEqual(call[4:], (4, 2048))
        self.assertEqual(tuple(r.out.shape), (128, 2051))
        self.assertEqual(
            r.backend.calls,
            [
                "should_reuse",
                ("metadata", 7),
                "should_capture",
                ("capture", (128, 2051), 7, True),
            ],
        )

    def test_single_request_draft_extend_scores_mk(self):
        md = _metadata(4, defer=False)
        r = self.run_select(_Mode.DRAFT_EXTEND_V2, 4, requests=1, md=md)
        self.assertEqual(r.st.record, [(qi.OP, "fold+score_v0+select_v1")])
        score = r.k.launches[1]
        self.assertEqual(score[:4], ("score", "v0", 0, torch.int32))
        q, cache, table, lengths, requests, scores, scale = score[4]
        self.assertEqual(tuple(q.shape), (4, 8, 128))
        self.assertIs(requests, md.req_pool_indices)

    def test_mtp_reuse_runs_sglangs_lookup(self):
        r = self.run_select(_Mode.DECODE, 8, reuse=True)
        self.assert_stock(r)
        self.assertEqual(r.backend.calls, ["should_reuse"])

    def test_stock_fallbacks(self):
        cases = {
            "prefill": dict(mode=_Mode.EXTEND),
            "no kernels": dict(kernels=None),
            "no op": dict(
                kernels=_kernels(prologue_fold=False, score=False, select=False)
            ),
            "eager metadata": dict(md=_metadata(16, graph=False)),
        }
        for name, kw in cases.items():
            with self.subTest(name):
                mode = kw.pop("mode", _Mode.TARGET_VERIFY)
                self.assert_stock(self.run_select(mode, 16, **kw))
        for flag in ("breakable", "cp", "invariant"):
            with self.subTest(flag):
                _STATE[flag] = True
                r = self.run_select(_Mode.TARGET_VERIFY, 16)
                self.assert_stock(r)
                self.assertEqual(r.backend.calls, [])
                _STATE[flag] = False

    def test_non_qsa_layer_is_stock(self):
        md = _metadata(16)
        _STATE["backend"] = _Backend(md)
        fb = SimpleNamespace(forward_mode=_Mode.TARGET_VERIFY)
        layer = _layer(_indexer(), is_qsa=False)
        st = _St(fb)
        x = torch.zeros(16, 640)
        ex = SimpleNamespace(qsa_indexer=_kernels(), qsa_draft=None)
        out = qi.select(ex, layer, x, x[:1], st)
        self.assertIs(out, STOCK_OUT)

    def test_wide_decode_folds_and_scores_stock(self):
        r = self.run_select(_Mode.DECODE, 128)
        self.assertEqual(r.st.record, [(qi.OP, "fold+score_stock+select_stock")])
        self.assertEqual([x[0] for x in r.k.launches], ["fold"])
        self.assertEqual(
            _STATE["calls"],
            [("qsa_mqa_decode", 128, WIDTH, 512), ("fast_topk", 128, 512, None)],
        )

    def test_padded_rows_return_the_mapped_rows(self):
        # DP padding: more hidden rows than the request mapping; stock trims them
        r = self.run_select(_Mode.TARGET_VERIFY, 16, hidden_rows=20)
        self.assertEqual(tuple(r.out.shape), (16, 512))
        self.assertEqual(r.k.launches[0][1], 16)

    def test_prologue_without_the_fold_is_sglangs(self):
        for name, kernels, ix in (
            ("switch", _kernels(prologue_fold=False), _indexer()),
            ("unfused", _kernels(), _indexer(fused=False)),
        ):
            with self.subTest(name):
                r = self.run_select(_Mode.TARGET_VERIFY, 16, kernels=kernels, ix=ix)
                self.assertEqual(r.st.record, [(qi.OP, "stock+score_v4+select_v1")])
                project, update = ix.calls
                self.assertEqual(project[:3], ("project_qk", 16, 8))
                self.assertIs(project[3], r.md.pending_ring_slots)
                self.assertEqual(update[1], 16)
                self.assertIs(update[2], r.md.pending_ring_slots)
                self.assertTrue(update[3])
                self.assertNotIn("fold", [x[0] for x in kernels.launches])

    def test_fp8_cache_scores_stock(self):
        md = _metadata(16, cache_dtype=torch.float8_e4m3fn)
        r = self.run_select(_Mode.TARGET_VERIFY, 16, md=md)
        self.assertEqual(r.st.record, [(qi.OP, "fold+score_stock+select_v1")])
        self.assertEqual(_STATE["calls"], [("qsa_mqa_decode", 16, WIDTH, 512)])

    def test_cold_layouts_run_sglangs_ops_in_a_capture(self):
        k = _kernels()
        with mock.patch.object(qi, "_capturing", lambda: True):
            # nothing ran eagerly yet: every op is SGLang's
            self.assert_stock(self.run_select(_Mode.TARGET_VERIFY, 16, kernels=k))
            fold_key = ("fold", torch.bfloat16, torch.bfloat16, 128, True)
            k._modules[fold_key] = object()
            k._warm.add(("v4", 16, WIDTH))
            r = self.run_select(_Mode.TARGET_VERIFY, 16, kernels=k)
            self.assertEqual(r.st.record, [(qi.OP, "fold+score_v4+select_stock")])
            # a capture does not mark a layout warm
            self.assertNotIn(("v1", 16, WIDTH), k._warm)
        # an eager run of the layout makes it warm
        r = self.run_select(_Mode.TARGET_VERIFY, 16, kernels=k)
        self.assertIn(("v1", 16, WIDTH), k._warm)


class TestBuild(_Case):
    def _model(self, *layers):
        m = SimpleNamespace(layers=list(layers), start_layer=0, end_layer=len(layers))
        return SimpleNamespace(model=m)

    def _attention(self, **kw):
        layer = _Attention()
        layer.is_qsa, layer.layer_id, layer.indexer = True, 3, _indexer()
        layer.indexer.__dict__.update(kw)
        return layer

    def test_layer_reasons(self):
        self.assertEqual(qi.layer_reasons(self._attention()), "")
        self.assertEqual(qi.layer_reasons(self._attention(index_n_heads=16)), "")
        layer = self._attention()
        layer.is_qsa = False
        self.assertIn("not a QSA layer", qi.layer_reasons(layer))
        for kw in (
            dict(index_head_dim=64),
            dict(compress_ratio=8),
            dict(block_topk=2048),
            dict(index_kv_heads=2),
            dict(index_n_heads=32),
        ):
            self.assertTrue(qi.layer_reasons(self._attention(**kw)), kw)

    def test_build_refusals(self):
        k = qi.IndexerKernels()
        model = self._model(_Linear(), self._attention())
        with mock.patch.object(torch.cuda, "is_available", lambda: False):
            self.assertIn("SM10x", qi.build(model, k, source_dir="/tmp/x"))
        with (
            mock.patch.object(torch.cuda, "is_available", lambda: True),
            mock.patch.object(torch.cuda, "get_device_capability", lambda *a: (10, 0)),
        ):
            _STATE["tp"] = 4
            self.assertIn("TP4", qi.build(model, k, source_dir="/tmp/x"))
            _STATE["tp"] = 1
            self.assertEqual(
                qi.build(self._model(_Linear()), k, source_dir="/x"), "no QSA layer"
            )
            bad = self._model(self._attention(index_head_dim=64))
            self.assertIn("layer 3", qi.build(bad, k, source_dir="/tmp/x"))
            self.assertIn("absolute", qi.build(model, k, source_dir="relative/dir"))
            compiled = []
            k.compile = lambda indexer, context_len, dtype: compiled.append(
                (indexer, context_len, dtype)
            )
            self.assertEqual(
                qi.build(model, k, source_dir="/tmp/x", context_len=32768), ""
            )
            self.assertEqual(str(k.source_dir), "/tmp/x")
            self.assertFalse(k.sm107)
            self.assertEqual(compiled, [(model.model.layers[1].indexer, 32768, None)])

            def fail(*args):
                raise RuntimeError("nvcc")

            k.compile = fail
            self.assertIn("nvcc", qi.build(model, k, source_dir="/tmp/x"))


if __name__ == "__main__":
    unittest.main()
