# SPDX-License-Identifier: Apache-2.0
"""The Qwen3.8 executor's PLE pieces (runtime/ple.py) on CPU: which kernel each piece
runs per mode and option, SGLang's function wherever none serves, the verify layout's
batch, the chunked conv against SGLang's padded conv, and the recipe options. SGLang's
modules are replaced by fakes and the GPU kernels by spies, so no GPU, no kernel build
and no model weights are needed."""

import dataclasses
import enum
import sys
import types
import unittest
from types import SimpleNamespace
from typing import Any
from unittest import mock

import torch
import torch.nn.functional as F

# The plugin modules that import SGLang at load, before the fakes replace it
from sgl_nvidia_mega.qwen3_8.kernels.ops.elementwise import qwen4_gate_post
from sgl_nvidia_mega.qwen3_8.kernels.ops.embeddings import (
    qwen4_ple_gather,
    qwen4_ple_verify_prepare,
)
from sgl_nvidia_mega.qwen3_8.kernels.ops.mamba import (
    ple_short_conv,
    qwen4_ple_fused,
    qwen4_verify_conv,
)
from sgl_nvidia_mega.qwen3_8.runtime import ple as P


class _Mode(enum.Enum):
    EXTEND = 1
    MIXED = 2
    DECODE = 3
    TARGET_VERIFY = 4
    DRAFT_EXTEND_V2 = 5
    IDLE = 6

    def is_target_verify(self):
        return self is _Mode.TARGET_VERIFY

    def is_extend(self):
        return self in (_Mode.EXTEND, _Mode.MIXED, _Mode.TARGET_VERIFY)

    def is_extend_without_speculative(self):
        return self in (_Mode.EXTEND, _Mode.MIXED)

    def is_idle(self):
        return self is _Mode.IDLE


@dataclasses.dataclass(frozen=True)
class _PLEBatch:  # every field of SGLang main's _PLEBatch, none defaulted
    mode: Any
    use_decode_fast_path: bool
    physical_tokens: int
    processed_tokens: int
    lengths: Any
    row_width: int
    req_indices: Any
    token_offsets: Any
    valid_tokens: Any
    state_indices: Any
    query_start_loc: Any
    ngram_context: Any
    ngram_eos_token_id: Any


class _HostTable:  # the fake Qwen4ExpPinnedHostEmbedding
    pass


class _Embedding(SimpleNamespace):  # the fake Qwen4ExpNGramEmbedding
    def __call__(self, batch, fb):
        return self.embeddings


class _RequestTable:
    pass


class _HybridPool:
    pass


_STATE = {}


def _pad_token_rows(x, total):  # qwen4_exp._pad_token_rows
    if x.shape[0] == total:
        return x
    out = x.new_zeros((total, *x.shape[1:]))
    out[: x.shape[0]] = x
    return out


def _install(test):
    """SGLang's modules ple.py imports, as fakes, restored after the test."""
    calls = []
    router = SimpleNamespace(parts={}, waits=[])
    router.part_view = lambda part: router.parts.get(part)
    router.wait_for_part = lambda part: calls.append(("wait", part))
    router.read = lambda layer_id, part: router.parts[part]
    _STATE.update(
        calls=calls,
        router=router,
        pool=_HybridPool(),
        space=SimpleNamespace(write=None, read_v2p=None),
        breakable=False,
        capture=False,
        cp=False,
        cp_active=False,
        pp=1,
        track=None,
        varlen=False,
        reduce=False,
        verify_fuse=False,
    )
    backend = SimpleNamespace(
        runner_ids=SimpleNamespace(state_ids=lambda sub_pool: _STATE["space"])
    )

    def stock_prepare(**kw):
        calls.append(("stock_prepare", kw["ngram_size"], kw["ngram_eos_token_id"]))
        return "stock-batch"

    fakes = {
        "sglang.srt.models.qwen4_exp": dict(
            _PLEBatch=_PLEBatch,
            _prepare_ple_batch=stock_prepare,
            _commit_ple_batch=lambda **kw: calls.append(("commit", kw)),
            _get_ple_forward_mode=lambda fb: fb.ple_mode,
            _get_processed_token_count=lambda fb, n: n,
            _ple_track_targets=lambda fb, batch: _STATE["track"],
            _pad_token_rows=_pad_token_rows,
            _use_qwen4_varlen_prefill=lambda *a: _STATE["varlen"],
            _shards_rows_over_cp=lambda fb: _STATE["cp"],
            Qwen4ExpPinnedHostEmbedding=_HostTable,
        ),
        "sglang.srt.mem_cache.ple_state_pool": dict(
            NGRAM_PART="ngram", SHORT_CONV_PART="short_conv"
        ),
        "sglang.srt.model_executor.forward_context": dict(
            get_attn_backend=lambda: backend,
            get_req_to_token_pool=lambda: _STATE["pool"],
            get_token_to_kv_pool=lambda: SimpleNamespace(layer_router=router),
        ),
        "sglang.srt.model_executor.runner": dict(
            get_is_capture_mode=lambda: _STATE["capture"]
        ),
        "sglang.srt.model_executor.runner_backend_utils.breakable_cuda_graph": dict(
            is_in_breakable_cuda_graph=lambda: _STATE["breakable"]
        ),
        "sglang.srt.layers.cp.utils": dict(is_cp_active=lambda fb: _STATE["cp_active"]),
        "sglang.srt.mem_cache.kv.requests": dict(RequestTable=_RequestTable),
        "sglang.srt.mem_cache.memory_pool": dict(HybridReqToTokenPool=_HybridPool),
        "sglang.srt.runtime_context": dict(
            get_parallel=lambda: SimpleNamespace(pp_size=_STATE["pp"])
        ),
        "sglang.kernels.ops.elementwise.qwen4_gate": dict(
            can_fuse_qwen4_gate_reduce=lambda *a: _STATE["reduce"],
            fused_qwen4_gate_reduce=lambda k, q, v: (
                calls.append("gate_reduce")
                or _eager_gate((k * q).sum(-1, keepdim=True), v)
            ),
            can_fuse_qwen4_gate_value=lambda *a: False,
            fused_qwen4_gate_value=None,
            _QWEN4_HC_COUNT=4,
            _QWEN4_HIDDEN_SIZE=2560,
        ),
        "sglang.kernels.ops.mamba.qwen4_short_conv": dict(
            can_fuse_qwen4_verify_conv=lambda *a: _STATE["verify_fuse"],
            fused_qwen4_verify_conv=lambda *a: (
                calls.append("stock_verify_conv") or a[0] + a[1]
            ),
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


class _St:
    def __init__(self, fb, model=None):
        self.fb = fb
        self.model = model
        self.ran_ops = []

    def ran(self, op, impl):
        self.ran_ops.append((op, impl))


def _eager_gate(gate, value, hidden=8):
    # Qwen4ExpPLELayer.forward's chain after the reduction
    gate = gate / (hidden**0.5)
    gate = gate.abs().clamp_min(1e-6).sqrt() * gate.sign()
    gate = torch.sigmoid(gate)
    return gate * value.unsqueeze(-2)


class TestOptions(unittest.TestCase):
    def fields(self, o):
        return (
            o.verify_prepare,
            o.verify_conv,
            o.fused_gate,
            o.extend_conv,
            o.extend_conv_cudnn,
            o.conv_chunk_size,
            o.gather_rows,
            o.gather_programs,
        )

    def test_recipe_values(self):
        self.assertIs(P.DEFAULT, P.HIGH_CONCURRENCY)
        self.assertEqual(
            self.fields(P.HIGH_CONCURRENCY), (True, True, True, True, True, 0, 32, 64)
        )
        self.assertEqual(
            self.fields(P.HIGH_CONCURRENCY_PLAIN_VERIFY),
            (True, False, True, True, False, 0, 0, 0),
        )
        self.assertEqual(
            self.fields(P.MID_CONCURRENCY), (True, True, True, False, False, 0, 32, 64)
        )

    def test_invalid(self):
        for bad in (
            dict(gather_rows=3),
            dict(gather_rows=256),
            dict(gather_rows=1),
            dict(extend_conv=False),  # the cuDNN shapes act only with the extend conv
            dict(conv_chunk_size=-1),
            dict(gather_programs=-1),
        ):
            with self.assertRaises(ValueError, msg=bad):
                P.Options(**bad)


def _verify_fb(requests=3, width=4, mode=_Mode.TARGET_VERIFY):
    rows = requests * width
    return SimpleNamespace(
        forward_mode=mode,
        ple_mode=mode,
        spec_algorithm=None,
        spec_info=SimpleNamespace(draft_token_num=width, topk=1),
        tbo_parent_token_range=None,
        req_pool_indices=torch.tensor([5, 6, 7, 0][:requests] + [0] * 2),
        extend_seq_lens=torch.full((requests + 2,), width, dtype=torch.int32),
        out_cache_loc=torch.arange(rows + 3, dtype=torch.int64),
    )


class TestPrepare(unittest.TestCase):
    def setUp(self):
        _install(self)
        self.view = SimpleNamespace(context=torch.zeros(9, 2, dtype=torch.int64))
        _STATE["router"].parts["ngram"] = self.view
        self.pool = _STATE["pool"]
        self.pool.state_sub_pool = "mamba"
        self.pool.req_index_to_mamba_index_mapping = torch.arange(9)
        self.pool.ple_window_cache = "untouched"
        self.model = SimpleNamespace(ple_ngram_size=3, ple_ngram_eos_token_id=151645)
        self.layouts = []
        self.checked = []

    def layout(self, *args):
        tokens, lengths, requests, locations, mapping, v2p, history, width, eos = args
        self.layouts.append(args)
        _STATE["calls"].append("layout")
        n, rows = requests.numel(), tokens.numel()
        return qwen4_ple_verify_prepare.VerifyPLELayout(
            torch.full((n,), width),
            torch.arange(rows) // width,
            torch.arange(rows) % width,
            torch.ones(rows, dtype=torch.bool),
            torch.zeros(n, dtype=torch.int64),
            torch.zeros(n, 2 + width, dtype=torch.int64),
            torch.zeros(rows, 3, dtype=torch.int64),
        )

    def prepare(self, fb, options=P.DEFAULT, can=True, ids=None):
        def can_prepare(*args):
            self.checked.append(args)
            return can

        ids = torch.arange(12) if ids is None else ids
        st = _St(fb, self.model)
        with (
            mock.patch.object(qwen4_ple_verify_prepare, "can_prepare", can_prepare),
            mock.patch.object(
                qwen4_ple_verify_prepare, "prepare_verify_layout", self.layout
            ),
        ):
            return P.prepare(st, ids, options=options), st.ran_ops

    def test_options_off_is_stock(self):
        batch, ran = self.prepare(_verify_fb(), options=None)
        self.assertEqual(batch, "stock-batch")
        self.assertEqual(ran, [("ple.prepare", "stock")])
        self.assertEqual(_STATE["calls"], [("stock_prepare", 3, 151645)])

    def test_verify_layout(self):
        fb = _verify_fb()
        batch, ran = self.prepare(fb)
        self.assertEqual(ran, [("ple.prepare", "verify_layout")])
        self.assertIsInstance(batch, _PLEBatch)
        self.assertIsNone(batch.query_start_loc)
        self.assertEqual((batch.row_width, batch.physical_tokens), (4, 12))
        self.assertFalse(batch.use_decode_fast_path)
        self.assertEqual(batch.ngram_eos_token_id, 151645)
        self.assertIs(self.pool.ple_window_cache[0], batch)
        self.assertIsNone(self.pool.ple_window_cache[2])
        # the history read waits for the N-gram part, as the torch branch does
        self.assertEqual(_STATE["calls"], [("wait", "ngram"), "layout"])
        tokens, lengths, requests, locations, mapping, v2p, history, width, ngram = (
            self.checked[0]
        )
        self.assertEqual(
            (tokens.numel(), lengths.numel(), requests.numel()), (12, 3, 3)
        )
        self.assertEqual(locations.numel(), 12)
        self.assertIsNone(v2p)
        self.assertIs(history, self.view.context)
        self.assertEqual((width, ngram), (4, 3))

    def test_virtual_state_ids_pass_their_table(self):
        table = torch.arange(9, dtype=torch.int32)
        _STATE["space"] = SimpleNamespace(write=lambda ids: ids, read_v2p=table)
        self.prepare(_verify_fb())
        self.assertIs(self.checked[0][5], table)

    def test_declines_write_nothing(self):
        def tbo(fb):
            fb.tbo_parent_token_range = (0, 1)

        def ngram(fb):
            fb.spec_algorithm = SimpleNamespace(is_ngram=lambda: True)

        def topk(fb):
            fb.spec_info.topk = 2

        cases = [
            ("extend", lambda fb: setattr(fb, "ple_mode", _Mode.EXTEND)),
            ("tbo", tbo),
            ("ngram", ngram),
            ("topk", topk),
            (
                "dp verify as extend",
                lambda fb: setattr(fb, "forward_mode", _Mode.EXTEND),
            ),
            ("breakable", lambda fb: _STATE.update(breakable=True)),
            ("pp", lambda fb: _STATE.update(pp=2)),
            ("cp", lambda fb: _STATE.update(cp_active=True)),
            ("width", lambda fb: setattr(fb.spec_info, "draft_token_num", 5)),
            (
                "pool",
                lambda fb: _STATE.update(pool=SimpleNamespace(ple_window_cache=1)),
            ),
            ("view", lambda fb: _STATE["router"].parts.pop("ngram")),
            (
                "v2p",
                lambda fb: _STATE.update(
                    space=SimpleNamespace(write=lambda i: i, read_v2p=None)
                ),
            ),
        ]
        for name, change in cases:
            self.setUp()
            fb = _verify_fb()
            change(fb)
            batch, ran = self.prepare(fb)
            self.assertEqual(batch, "stock-batch", name)
            self.assertEqual(ran, [("ple.prepare", "stock")], name)
            self.assertEqual(self.layouts, [], name)
            self.assertNotIn(("wait", "ngram"), _STATE["calls"], name)
            self.assertEqual(self.pool.ple_window_cache, "untouched", name)
        self.setUp()
        batch, _ = self.prepare(_verify_fb(), can=False)
        self.assertEqual(batch, "stock-batch")
        self.assertEqual(self.pool.ple_window_cache, "untouched")


class TestCommit(unittest.TestCase):
    def test_is_sglangs(self):
        _install(self)
        fb = SimpleNamespace()
        st = _St(fb)
        P.commit("batch", st)
        self.assertEqual(
            _STATE["calls"], [("commit", dict(batch="batch", forward_batch=fb))]
        )
        self.assertEqual(st.ran_ops, [("ple.commit", "stock")])


def _ple(fusion=True, offloaded=False, hidden=8, channels=32, k=4, dilation=3):
    calls = []
    table = _HostTable() if offloaded else SimpleNamespace()
    table.weight_scale = torch.tensor(2.0)
    table.reduce = lambda x: calls.append("reduce") or x
    embedding = _Embedding(enable_ple_fusion=fusion, ngram_embedding=table)
    ple = SimpleNamespace(
        layer_id=11,
        hidden_size=hidden,
        hc_count=4,
        conv_channels=channels,
        short_conv_dilation=dilation,
        short_conv_state_len=(k - 1) * dilation,
        conv1d=SimpleNamespace(weight=torch.randn(channels, 1, k).bfloat16()),
        _prefetch_state=None,
        _prefetch_stream=None,
        ple_embedding=embedding,
        calls=calls,
    )
    gen = torch.Generator().manual_seed(3)
    wk = torch.randn(6, 4 * hidden, generator=gen)
    wv = torch.randn(6, hidden, generator=gen)
    ple.key_proj = lambda e: (e @ wk, None)
    ple.value_proj = lambda e: (e @ wv, None)
    norms = {"key": 0.5, "query": 0.25, "conv": 2.0}
    ple.norm_key, ple.norm_query, ple.norm_conv = "key", "query", "conv"
    ple._apply_ple_norm = lambda norm, x: x * norms[norm] + 0.125
    ple._short_conv = lambda x, fb, batch: calls.append("stock_short_conv") or x * 3
    return ple


def _batch(mode=_Mode.EXTEND, lengths=(5, 2), width=None, physical=None, fast=False):
    lengths = list(lengths)
    width = width or max(lengths)
    rows = sum(lengths)
    req = torch.cat([torch.full((n,), i) for i, n in enumerate(lengths)])
    off = torch.cat([torch.arange(n) for n in lengths])
    return SimpleNamespace(
        mode=mode,
        use_decode_fast_path=fast,
        physical_tokens=physical or rows,
        processed_tokens=rows,
        lengths=torch.tensor(lengths),
        row_width=width,
        req_indices=req,
        token_offsets=off,
        valid_tokens=torch.tensor([True] * (rows - 1) + [False]),
        state_indices=torch.tensor([3, 1][: len(lengths)] + [0] * (len(lengths) - 2)),
        query_start_loc=None,
    )


def _main_forward(ple, hidden_states, batch, embeddings):
    """Qwen4ExpPLELayer.forward's eager chain (SGLang main), on the fakes."""
    hidden_states = hidden_states[: batch.processed_tokens]
    key, _ = ple.key_proj(embeddings)
    value, _ = ple.value_proj(embeddings)
    token_count = hidden_states.shape[0]
    key = key.reshape(token_count, ple.hc_count, ple.hidden_size)
    query = hidden_states.reshape(token_count, ple.hc_count, ple.hidden_size)
    key_normed = ple._apply_ple_norm(ple.norm_key, key)
    query_normed = ple._apply_ple_norm(ple.norm_query, query)
    gate = (key_normed * query_normed).sum(dim=-1, keepdim=True)
    gated_value = _eager_gate(gate, value, ple.hidden_size)
    gated_value_normed = ple._apply_ple_norm(ple.norm_conv, gated_value)
    gated_value = gated_value.flatten(-2)
    output = gated_value + gated_value_normed.flatten(-2) * 3
    output = torch.where(
        batch.valid_tokens.unsqueeze(-1), output, torch.zeros_like(output)
    )
    return _pad_token_rows(output, batch.physical_tokens)


class TestServedAndAdd(unittest.TestCase):
    def setUp(self):
        _install(self)
        self.fb = SimpleNamespace(forward_mode=_Mode.EXTEND)

    def served(self, options, batch, ple=None):
        layer = SimpleNamespace(ple=ple or _ple())
        return P.served(options, layer, batch, self.fb)

    def test_pieces(self):
        o = P.DEFAULT
        self.assertEqual(self.served(None, _batch()), "")
        self.assertEqual(self.served(o, None), "")
        self.assertEqual(self.served(o, _batch()), "gate+conv")
        self.assertEqual(self.served(o, _batch(_Mode.MIXED)), "gate+conv")
        self.assertEqual(self.served(P.MID_CONCURRENCY, _batch()), "gate")
        self.assertEqual(
            self.served(o, _batch(_Mode.TARGET_VERIFY)), "gate+verify_conv"
        )
        self.assertEqual(
            self.served(
                P.HIGH_CONCURRENCY_PLAIN_VERIFY,
                _batch(_Mode.TARGET_VERIFY),
                _ple(fusion=False),
            ),
            "gate",
        )
        self.assertEqual(self.served(o, _batch(_Mode.DECODE, fast=True)), "")
        self.assertEqual(self.served(o, _batch(_Mode.DECODE)), "gate")
        self.assertEqual(self.served(o, _batch(_Mode.DRAFT_EXTEND_V2)), "gate")
        self.assertEqual(
            self.served(o, _batch(_Mode.DECODE, fast=True), _ple(offloaded=True)),
            "gather",
        )
        _STATE["cp"] = True
        self.assertEqual(self.served(o, _batch()), "")
        _STATE.update(cp=False, breakable=True)
        self.assertEqual(self.served(o, _batch()), "")

    def test_add_falls_back_to_the_layers_own(self):
        seen = []
        layer = SimpleNamespace(ple=_ple(), _ple_forward_batch=None, _ple_batch=None)
        layer._add_ple_embedding = lambda r: (
            seen.append((layer._ple_forward_batch, layer._ple_batch)) or r + 1
        )
        st = _St(self.fb)
        batch = _batch(_Mode.DECODE, fast=True)
        out = P.add(layer, torch.zeros(2), batch, st, options=P.DEFAULT)
        self.assertTrue(torch.equal(out, torch.ones(2)))
        self.assertEqual(seen, [(self.fb, batch)])
        self.assertIsNone(layer._ple_forward_batch)
        self.assertIsNone(layer._ple_batch)
        self.assertEqual(st.ran_ops, [("ple.add", "stock")])

    def test_add_runs_the_mirror_where_served(self):
        layer = SimpleNamespace(ple=_ple())
        st = _St(self.fb)
        with mock.patch.object(P, "forward", lambda *a: torch.full((2,), 5.0)):
            out = P.add(layer, torch.ones(2), _batch(), st, options=P.DEFAULT)
        self.assertTrue(torch.equal(out, torch.full((2,), 6.0)))


class TestForward(unittest.TestCase):
    def setUp(self):
        _install(self)
        self.fb = SimpleNamespace(forward_mode=_Mode.EXTEND)
        self.gen = torch.Generator().manual_seed(11)

    def run_forward(self, options, ple, batch):
        hidden = torch.randn(
            batch.physical_tokens, 4 * ple.hidden_size, generator=self.gen
        )
        embeddings = torch.randn(batch.processed_tokens, 6, generator=self.gen)
        ple.ple_embedding.embeddings = embeddings
        st = _St(self.fb)
        with mock.patch.object(
            P,
            "short_conv",
            lambda o, p, x, fb, b, st_: st_.ran("ple.conv", "spy") or x * 3,
        ):
            out = P.forward(options, ple, hidden, self.fb, batch, st)
        return out, _main_forward(ple, hidden, batch, embeddings), st.ran_ops

    def test_eager_chain_matches_main(self):
        options = P.Options(fused_gate=False)
        out, ref, ran = self.run_forward(options, _ple(), _batch(physical=9))
        self.assertTrue(torch.equal(out, ref))
        self.assertEqual(out.shape[0], 9)
        self.assertEqual(ran, [("ple.gate", "stock"), ("ple.conv", "spy")])

    def test_gate_post_after_the_reduction(self):
        posted = []

        def fused(gate, value):
            posted.append(tuple(gate.shape))
            return _eager_gate(gate, value)

        with (
            mock.patch.object(
                qwen4_gate_post, "can_fuse_qwen4_gate_post", lambda g, v: True
            ),
            mock.patch.object(
                qwen4_gate_post, "is_qwen4_gate_post_ready", lambda d: True
            ),
            mock.patch.object(qwen4_gate_post, "fused_qwen4_gate_post", fused),
        ):
            out, ref, ran = self.run_forward(P.DEFAULT, _ple(), _batch())
        self.assertTrue(torch.equal(out, ref))
        self.assertEqual(posted, [(7, 4, 1)])
        self.assertEqual(ran[0], ("ple.gate", "gate_post"))
        # a failed self-test keeps the chain
        with mock.patch.object(
            qwen4_gate_post, "is_qwen4_gate_post_ready", lambda d: False
        ):
            out, ref, ran = self.run_forward(P.DEFAULT, _ple(), _batch())
        self.assertTrue(torch.equal(out, ref))
        self.assertEqual(ran[0], ("ple.gate", "stock"))

    def test_verify_conv_one_kernel(self):
        seen = []
        _STATE.update(verify_fuse=True, reduce=True)
        part = SimpleNamespace(
            view_layer_id=2,
            view=SimpleNamespace(
                layer_cache=lambda i: "state", layer_intermediate_cache=lambda i: None
            ),
        )
        _STATE["router"].parts["short_conv"] = part

        def verify(*args, one_kernel):
            seen.append((args[3], args[6], args[7], one_kernel))
            return args[0] + args[1]

        batch = _batch(_Mode.TARGET_VERIFY, lengths=(4, 4), physical=10)
        for options, impl in (
            (P.DEFAULT, "verify_conv"),
            (P.HIGH_CONCURRENCY_PLAIN_VERIFY, "stock"),
        ):
            with mock.patch.object(
                qwen4_verify_conv, "fused_qwen4_verify_conv", verify
            ):
                out, _, ran = self.run_forward(options, _ple(), batch)
            self.assertEqual(out.shape[0], 10)
            self.assertTrue(bool((out[8:] == 0).all()))
            self.assertEqual(ran[-1], ("ple.conv", impl))
        self.assertEqual(seen, [("state", 4, 3, True), ("state", 4, 3, False)])
        self.assertEqual(_STATE["calls"].count("gate_reduce"), 2)

    def test_offloaded_table_gathers_here(self):
        ple = _ple(offloaded=True)
        flow = []
        ngram = ple.ple_embedding
        ngram.compute_ngram_ids = lambda b: flow.append("ids") or torch.zeros(7, 2)
        ngram._prepare_embedding_lookup = lambda ids, fb, n: (
            flow.append(("lookup", n))
            or (
                ids.long(),
                7,
            )
        )
        ngram._finish_embedding_lookup = lambda e, sem, fb, n: (
            flow.append(("finish", sem)) or e
        )
        with mock.patch.object(
            P,
            "gather",
            lambda table, ids, o, out=None: (
                flow.append(("gather", o.gather_rows)) or torch.ones(7, 2, 3)
            ),
        ):
            embeddings = P._embed(ple, _batch(), self.fb, P.DEFAULT)
        self.assertEqual(flow, ["ids", ("lookup", 7), ("gather", 32), ("finish", 7)])
        self.assertEqual(ple.calls, ["reduce"])
        self.assertTrue(torch.equal(embeddings, torch.full((7, 6), 2.0)))


def _conv_part(slots=6, channels=32, state_len=9):
    cache = torch.randn(slots, channels, state_len).bfloat16()
    return cache, SimpleNamespace(
        view_layer_id=0, view=SimpleNamespace(layer_cache=lambda i: cache)
    )


def _main_padded_conv(ple, x, batch, conv_state, track):
    """Qwen4ExpPLELayer._short_conv's padded path (not a target verify), SGLang main."""
    state = conv_state.index_select(0, batch.state_indices).to(dtype=x.dtype)
    padded_seq = x.new_zeros(
        (batch.lengths.shape[0], batch.row_width, ple.conv_channels)
    )
    padded_seq[batch.req_indices, batch.token_offsets] = x
    conv_input = torch.cat([state, padded_seq.transpose(1, 2)], dim=-1)
    conv_output = F.conv1d(
        conv_input,
        ple.conv1d.weight.to(dtype=x.dtype),
        bias=None,
        dilation=ple.short_conv_dilation,
        groups=ple.conv_channels,
    ).transpose(1, 2)
    state_cols = torch.arange(ple.short_conv_state_len, dtype=torch.long)

    def _gather_at(offsets):
        return conv_input.gather(
            2,
            (offsets.unsqueeze(1) + state_cols.unsqueeze(0))
            .unsqueeze(1)
            .expand(-1, ple.conv_channels, -1),
        )

    conv_state[batch.state_indices] = _gather_at(batch.lengths).to(conv_state.dtype)
    if track is not None:
        conv_state[track[0]] = _gather_at(track[1]).to(conv_state.dtype)
    return F.silu(conv_output[batch.req_indices, batch.token_offsets])


class TestShortConv(unittest.TestCase):
    def setUp(self):
        _install(self)
        self.cache, part = _conv_part()
        _STATE["router"].parts["short_conv"] = part

    def run_conv(self, options, batch, x=None, fb=None):
        ple = _ple()
        x = torch.randn(batch.processed_tokens, 32).bfloat16() if x is None else x
        fb = fb or SimpleNamespace(extend_seq_lens_cpu=batch.lengths.tolist())
        st = _St(fb)
        return P.short_conv(options, ple, x, fb, batch, st), ple, x, st.ran_ops

    def test_sglang_kernels_first(self):
        for batch, state in (
            (_batch(_Mode.DECODE, fast=True), {}),
            (_batch(_Mode.TARGET_VERIFY, lengths=(4, 4)), {}),
            (_batch(), {"varlen": True}),
        ):
            _STATE.update(state)
            out, ple, x, ran = self.run_conv(P.DEFAULT, batch)
            self.assertEqual(ple.calls, ["stock_short_conv"])
            self.assertTrue(torch.equal(out, x * 3))
            self.assertEqual(ran, [("ple.conv", "stock")])
            _STATE["varlen"] = False

    def test_extend_conv_over_the_token_rows(self):
        engaged = []
        _STATE["track"] = (torch.tensor([5, 4]), torch.tensor([3, 1]))
        snaps = (torch.ones(2, 32, 9), torch.full((2, 32, 9), 2.0))

        def engages(
            x, batch, lens, *, weight, conv_state, channels, state_len, cudnn_shapes
        ):
            engaged.append((lens, channels, state_len, cudnn_shapes, weight.dtype))
            return True

        def fused(x, state, weight, batch, *, dilation, state_offsets):
            engaged.append(("fused", dilation, len(state_offsets)))
            return x * 2, snaps

        with (
            mock.patch.object(qwen4_ple_fused, "conv_engages", engages),
            mock.patch.object(qwen4_ple_fused, "fused_short_conv_extend", fused),
        ):
            out, ple, x, ran = self.run_conv(P.DEFAULT, _batch())
        self.assertEqual(
            engaged, [([5, 2], 32, 9, True, torch.bfloat16), ("fused", 3, 2)]
        )
        self.assertTrue(torch.equal(out, F.silu(x * 2)))
        self.assertTrue(bool((self.cache[[3, 1]] == 1).all()))
        self.assertTrue(bool((self.cache[[5, 4]] == 2).all()))
        self.assertEqual(ran, [("ple.conv", "extend_conv")])

    def test_extend_conv_ahead_of_the_varlen_kernel(self):
        _STATE["varlen"] = True
        try:
            with (
                mock.patch.object(
                    qwen4_ple_fused, "conv_engages", lambda *a, **k: True
                ),
                mock.patch.object(
                    qwen4_ple_fused,
                    "fused_short_conv_extend",
                    lambda x, *a, **k: (x * 2, (torch.ones(2, 32, 9),)),
                ),
            ):
                out, ple, x, ran = self.run_conv(P.DEFAULT, _batch())
            self.assertEqual(ple.calls, [])
            self.assertTrue(torch.equal(out, F.silu(x * 2)))
            self.assertEqual(ran, [("ple.conv", "extend_conv")])
            out, ple, x, ran = self.run_conv(
                P.Options(extend_conv=False, extend_conv_cudnn=False), _batch()
            )
            self.assertEqual(ple.calls, ["stock_short_conv"])
            self.assertEqual(ran, [("ple.conv", "stock")])
        finally:
            _STATE["varlen"] = False

    def test_chunked_conv_matches_the_padded_conv(self):
        torch.manual_seed(0)
        options = P.Options(
            extend_conv=False, extend_conv_cudnn=False, conv_chunk_size=2
        )
        for track in (None, (torch.tensor([5, 4]), torch.tensor([3, 1]))):
            ple = _ple()
            batch = _batch(lengths=(5, 2))
            x = torch.randn(7, 32).bfloat16()
            _STATE["track"] = track
            reference_cache = self.cache.clone()
            ref = _main_padded_conv(ple, x, batch, reference_cache, track)
            st = _St(SimpleNamespace(extend_seq_lens_cpu=[5, 2]))
            out = P.short_conv(options, ple, x, st.fb, batch, st)
            self.assertEqual(st.ran_ops, [("ple.conv", "chunked")])
            self.assertEqual(ple.calls, [])
            self.assertTrue(torch.equal(out, ref))
            self.assertTrue(torch.equal(self.cache, reference_cache))

    def test_chunked_declines(self):
        options = P.Options(
            extend_conv=False, extend_conv_cudnn=False, conv_chunk_size=5
        )
        out, ple, x, ran = self.run_conv(options, _batch(lengths=(5, 2)))
        self.assertEqual(ran, [("ple.conv", "stock")])
        _STATE["capture"] = True
        options = P.Options(
            extend_conv=False, extend_conv_cudnn=False, conv_chunk_size=2
        )
        out, ple, x, ran = self.run_conv(options, _batch(lengths=(5, 2)))
        self.assertEqual(ran, [("ple.conv", "stock")])
        _STATE["capture"] = False
        fb = SimpleNamespace(extend_seq_lens_cpu=[5, 1])  # not the pass's rows
        out, ple, x, ran = self.run_conv(options, _batch(lengths=(5, 2)), fb=fb)
        self.assertEqual(ran, [("ple.conv", "stock")])


class TestGatherAndPrefetch(unittest.TestCase):
    def setUp(self):
        _install(self)

    def table(self):
        table = _HostTable()
        table.embedding_dim = 3
        table.weight = torch.zeros(10, 3, dtype=torch.bfloat16)
        table.shard_indices = SimpleNamespace(
            org_vocab_start_index=2, org_vocab_end_index=9
        )
        table._block_d = 4
        table.enqueued = []
        table._file_prefetcher = SimpleNamespace(
            enqueue=lambda ids, vocab_start, vocab_end: table.enqueued.append(
                (ids.numel(), vocab_start, vocab_end)
            )
        )
        table.allocate_output = lambda shape, device: torch.empty(
            shape, dtype=torch.bfloat16
        )
        return table

    def test_gather_passes_the_rows_and_programs(self):
        seen = []
        table = self.table()
        ids = torch.arange(6).view(2, 3)
        with mock.patch.object(
            qwen4_ple_gather, "gather_rows", lambda **kw: seen.append(kw)
        ):
            out = P.gather(table, ids, P.DEFAULT)
        self.assertEqual(out.shape, (2, 3, 3))
        (kw,) = seen
        self.assertEqual((kw["rows"], kw["programs"]), (32, 64))
        self.assertEqual((kw["tp_vocab_start"], kw["tp_vocab_end"]), (2, 9))
        self.assertFalse(kw["is_fp8"])
        self.assertEqual(kw["flat_ids"].tolist(), list(range(6)))
        self.assertEqual(table.enqueued, [(6, 2, 9)])
        with self.assertRaises(ValueError):
            P.gather(
                table, ids, P.DEFAULT, out=torch.empty(2, 3, 4, dtype=torch.bfloat16)
            )

    def test_prefetch_falls_back(self):
        fb = SimpleNamespace()
        for options, offloaded, stream in (
            (None, True, object()),
            (P.HIGH_CONCURRENCY_PLAIN_VERIFY, True, object()),
            (P.DEFAULT, False, object()),
            (P.DEFAULT, True, None),
        ):
            ple = _ple(offloaded=offloaded)
            ple._prefetch_stream = stream
            seen = []
            ple.start_prefetch = lambda batch, forward_batch: seen.append(
                (batch, forward_batch)
            )
            st = _St(fb)
            P.prefetch(SimpleNamespace(ple=ple), "batch", st, options=options)
            self.assertEqual(seen, [("batch", fb)])
            self.assertEqual(st.ran_ops, [("ple.prefetch", "stock")])

    def test_prefetch_gathers_on_the_side_stream(self):
        events = []

        class Stream:
            def wait_stream(self, other):
                events.append("wait")

        class Ids:
            shape = (7, 2)

            def record_stream(self, stream):
                events.append("record")

        class Module:
            @staticmethod
            def current_stream():
                return "main"

            class stream:
                def __init__(self, s):
                    pass

                def __enter__(self):
                    events.append("enter")

                def __exit__(self, *exc):
                    events.append("exit")

        ple = _ple(offloaded=True)
        ple._prefetch_stream = Stream()
        ple.ple_embedding.ngram_heads = 2
        ple.ple_embedding.compute_ngram_ids = lambda batch: "ids"
        ple.ple_embedding._prepare_embedding_lookup = lambda ids, fb, n: (Ids(), 7)
        buffer = torch.zeros(7, 6)
        ple._get_prefetch_buffer = lambda n, ids: buffer
        st = _St(SimpleNamespace())
        with (
            mock.patch.object(torch, "get_device_module", lambda: Module),
            mock.patch.object(
                P,
                "gather",
                lambda t, ids, o, out=None: events.append(("gather", out.shape)),
            ),
        ):
            P.prefetch(SimpleNamespace(ple=ple), _batch(), st, options=P.DEFAULT)
            with self.assertRaises(RuntimeError):
                P.prefetch(SimpleNamespace(ple=ple), _batch(), st, options=P.DEFAULT)
        self.assertEqual(
            events, ["wait", "record", "enter", ("gather", (7, 2, 3)), "exit"]
        )
        self.assertIs(ple._prefetch_state[0], buffer)
        self.assertEqual(ple._prefetch_state[1:], (7, 7))
        self.assertEqual(st.ran_ops, [("ple.prefetch", "gather_rows")])


class TestKernelCopies(unittest.TestCase):
    """The copied kernels' host-side pieces that run on CPU."""

    def test_gather_rows_check(self):
        for rows in (0, 2, 32, 128):
            self.assertEqual(qwen4_ple_gather.check_rows(rows), rows)
        for rows in (1, 3, 256):
            with self.assertRaises(ValueError):
                qwen4_ple_gather.check_rows(rows)

    def test_chunked_conv_equals_one_conv(self):
        torch.manual_seed(1)
        conv_input = torch.randn(2, 16, 9 + 11).bfloat16()
        weight = torch.randn(16, 1, 4).bfloat16()
        full = F.conv1d(conv_input, weight, dilation=3, groups=16).transpose(1, 2)
        lengths = [11, 6]
        out = ple_short_conv.chunked_packed_short_conv(
            conv_input, weight, lengths, dilation=3, chunk_size=4
        )
        ref = torch.cat([full[0, :11], full[1, :6]])
        self.assertTrue(torch.equal(out, ref))

    def test_extend_conv_declines_off_cuda(self):
        self.assertTrue(qwen4_ple_fused.native_depthwise(2, 10240, 100))
        self.assertFalse(qwen4_ple_fused.native_depthwise(16, 10240, 2**17))
        batch = _batch()
        batch.mode = SimpleNamespace(name="EXTEND")
        x = torch.zeros(7, 32, dtype=torch.bfloat16)
        args = dict(
            weight=torch.zeros(32, 1, 4, dtype=torch.bfloat16),
            conv_state=torch.zeros(4, 32, 9, dtype=torch.bfloat16),
            channels=32,
            state_len=9,
        )
        self.assertFalse(qwen4_ple_fused.conv_engages(x, batch, [5, 2], **args))
        batch.mode = SimpleNamespace(name="DECODE")
        self.assertFalse(qwen4_ple_fused.conv_engages(x, batch, [5, 2], **args))


class TestBuild(unittest.TestCase):
    def test_refusals(self):
        self.assertEqual(P.build(None, None), "no options")
        if not torch.cuda.is_available():
            self.assertIn("CUDA", P.build(None))


if __name__ == "__main__":
    unittest.main()
