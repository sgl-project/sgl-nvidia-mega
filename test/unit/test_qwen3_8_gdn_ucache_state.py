# SPDX-License-Identifier: Apache-2.0
"""The Qwen3.8 executor's GDN u-cache state (runtime/gdn_ucache_state.py) on CPU: the
ring bytes it declares, prepare's allocation and its repeat, the reset dispatch, the
commit order and the scheduler capabilities. The kernels and the track materializer are
replaced by spies, and SGLang's model module by a fake, so no GPU is needed."""

import sys
import types
import unittest
from types import SimpleNamespace
from unittest import mock

import torch

_QWEN4 = "sglang.srt.models.qwen4_exp"


def _install_fakes():
    # test_qwen3_8_executor.py installs the same module; either fake serves.
    if _QWEN4 in sys.modules:
        return
    qwen4 = types.ModuleType(_QWEN4)

    class _Linear:
        ple = None

    class _Attention:
        ple = None

    qwen4.Qwen4ExpLinearDecoderLayer = _Linear
    qwen4.Qwen4ExpAttentionDecoderLayer = _Attention
    sys.modules[_QWEN4] = qwen4


_install_fakes()

from sgl_nvidia_mega.qwen3_8.kernels.ops.attention.linear import (  # noqa: E402
    gdn_ucache_state as K,
)
from sgl_nvidia_mega.qwen3_8.runtime import gdn_ucache_state as S  # noqa: E402
from sgl_nvidia_mega.qwen3_8.runtime.gdn_ucache_state import UcacheState  # noqa: E402

_SCATTER = (
    "sglang.kernels.ops.mamba.mamba_state_scatter_triton."
    "fused_conv_window_scatter_with_mask"
)
# A CPU-only torch has no capture query; prepare asks it before allocating.
_CAPTURING = {"now": False}
_capture_patch = mock.patch(
    "torch.cuda.is_current_stream_capturing", lambda: _CAPTURING["now"]
)


def setUpModule():
    _capture_patch.start()


def tearDownModule():
    _capture_patch.stop()


def _source_ring_bytes():
    """The source's gdn_ucache_ring_bytes_per_slot, its body copied verbatim."""

    def gdn_ucache_ring_bytes_per_slot(
        *, num_layers, num_k_heads, num_v_heads, head_k_dim, head_v_dim
    ):
        k_bytes = num_k_heads * 32 * head_k_dim * 2
        u_bytes = num_v_heads * 32 * head_v_dim * 2
        g_bytes = num_v_heads * 32 * 4
        return (k_bytes + u_bytes + g_bytes) * num_layers

    return gdn_ucache_ring_bytes_per_slot


def _model(kinds="GGGA", heads=(16, 48), dims=(128, 128), start=0, end=None):
    """G: a GDN layer, A: an attention layer."""
    linear = sys.modules[_QWEN4].Qwen4ExpLinearDecoderLayer
    layers = []
    for kind in kinds:
        if kind == "G":
            layer = linear()
            layer.linear_attn = SimpleNamespace(
                attn=SimpleNamespace(
                    num_k_heads=heads[0],
                    num_v_heads=heads[1],
                    head_k_dim=dims[0],
                    head_v_dim=dims[1],
                )
            )
        else:
            layer = SimpleNamespace()
        layers.append(layer)
    m = SimpleNamespace(
        layers=layers, start_layer=start, end_layer=len(kinds) if end is None else end
    )
    return SimpleNamespace(model=m)


def _state(model=None, **kw):
    kw.setdefault("tokens_per_request", 4)
    kw.setdefault("key_ring_fp16", False)  # the fp16 key ring asks the GPU for SM107
    return UcacheState(model or _model(heads=(2, 6), dims=(8, 8)), **kw)


def _view(layers=3, slots=10, rows=6, conv_numels=(1, 0)):
    """A preparation view: [layers, slots, ...] state and conv pools (an empty conv
    type when its numel is 0) and ``rows`` request rows."""
    conv = tuple(torch.zeros(layers, slots, n, 3) for n in conv_numels)
    windows = tuple(torch.zeros(layers, rows, 4, n, 3) for n in conv_numels)
    mamba = SimpleNamespace(
        temporal=torch.zeros(layers, slots, 2, 2, 2),
        conv=conv,
        intermediate_conv_window=windows,
        layer_ids=tuple(range(layers)),
    )
    return SimpleNamespace(mamba=mamba, parts=None, num_request_rows=rows)


def _ids(*values):
    return torch.tensor(values, dtype=torch.int32)


def _io(track=False):
    return SimpleNamespace(
        state_slots=_ids(3, 0, 5),
        track_slots=_ids(7, 0, 8) if track else None,
        request_rows=_ids(1, 2, 3),
        accept_lens=_ids(2, 1, 4),
        last_step=_ids(1, 0, 3),
        track_steps=_ids(0, -1, 2) if track else None,
        draft_token_num=4,
        src_indices_raw=None,
    )


class _FakeTrack:
    """GdnUcacheTrackMaterializer without FlashInfer: records its calls."""

    calls = None

    def __init__(self, *, state, buffers):
        self.state, self.buffers = state, buffers
        _FakeTrack.calls.append(("track_init", state, buffers))

    def warm_up(self, *, max_batch_size, max_eager_batch_size=0):
        _FakeTrack.calls.append(("warm_up", max_batch_size))

    def materialize(self, **kw):
        _FakeTrack.calls.append(("materialize", kw))


class TestBytes(unittest.TestCase):
    def test_per_slot_bytes_match_the_source_formula(self):
        source = _source_ring_bytes()
        for heads, dims, kinds in (
            ((16, 48), (128, 128), "GGGA" * 3),  # TP1 heads per rank
            ((4, 12), (128, 128), "GGGA"),  # TP4
            ((2, 6), (8, 16), "GAG"),
        ):
            state = _state(_model(kinds, heads, dims))
            want = source(
                num_layers=kinds.count("G"),
                num_k_heads=heads[0],
                num_v_heads=heads[1],
                head_k_dim=dims[0],
                head_v_dim=dims[1],
            )
            self.assertEqual(state.ring_bytes_per_slot, want)
        # (16*32*128*2 + 48*32*128*2 + 48*32*4) * 3
        self.assertEqual(_state(_model()).ring_bytes_per_slot, 530432 * 3)

    def test_rank_layers_only(self):
        # PP-style slice: layers 2..4 hold one GDN layer
        state = _state(_model("GGGA", start=2, end=4))
        self.assertEqual(state.num_layers, 1)

    def test_charge_per_slot_or_request_row(self):
        per_row = _state(_model()).ring_bytes_per_slot
        rows = _state(_model(), by_request_row=True)
        self.assertEqual(
            (rows.state_bytes_per_slot, rows.state_bytes_per_request_row), (0, per_row)
        )
        slots = _state(_model(), by_request_row=False)
        self.assertEqual(
            (slots.state_bytes_per_slot, slots.state_bytes_per_request_row),
            (per_row, 0),
        )

    def test_head_shapes_must_agree(self):
        model = _model("GG")
        model.model.layers[1].linear_attn.attn.num_v_heads = 32
        with self.assertRaises(ValueError):
            UcacheState(model, tokens_per_request=4)
        with self.assertRaises(ValueError):
            UcacheState(_model("AA"), tokens_per_request=4)


class TestCapabilities(unittest.TestCase):
    def test_tracks_decode_state_gating(self):
        for track in (False, True):
            for by_row in (False, True):
                for fp16 in (False, True):
                    state = _state(
                        track_snapshots=track, by_request_row=by_row, key_ring_fp16=fp16
                    )
                    self.assertEqual(
                        state.tracks_decode_state, track and not by_row and not fp16
                    )
                    self.assertFalse(state.keeps_draft_ssm)
                    self.assertFalse(state.backs_up_state)

    def test_recipe_defaults(self):
        state = UcacheState(_model(), tokens_per_request=4)
        self.assertTrue(state.key_ring_fp16)
        self.assertTrue(state.by_request_row)
        self.assertFalse(state.tracks_decode_state)
        self.assertIsNone(state.buffers)


class TestPrepare(unittest.TestCase):
    def setUp(self):
        _FakeTrack.calls = []

    def test_slot_rings_and_repeat(self):
        state = _state(by_request_row=False)
        view = _view(slots=10, rows=6)
        with mock.patch.object(
            K, "allocate_gdn_ucache", wraps=K.allocate_gdn_ucache
        ) as alloc:
            state.prepare(view)
            b = state.buffers
            self.assertEqual(tuple(b.k_ring.shape), (3, 10, 2, 32, 8))
            self.assertEqual(tuple(b.u_ring.shape), (3, 10, 6, 32, 8))
            self.assertEqual(tuple(b.g_ring.shape), (3, 10, 6, 32))
            self.assertEqual(tuple(b.hist_len.shape), (10,))
            self.assertEqual(
                (b.k_ring.dtype, b.u_ring.dtype, b.g_ring.dtype),
                (torch.bfloat16, torch.float16, torch.float32),
            )
            self.assertEqual(b.hist_len.dtype, torch.int32)
            self.assertEqual(b.cache_base.dtype, torch.int32)
            self.assertEqual(int(b.hist_len.abs().sum() + b.k_ring.abs().sum()), 0)
            self.assertEqual(b.flush_min, 16 - 4 + 1)
            self.assertFalse(b.by_request_row)
            self.assertTrue(b.k_ring.is_contiguous() and b.u_ring.is_contiguous())
            # a repeat (recapture) on the same pool keeps the rings
            state.prepare(view)
            state.prepare(_view(slots=10, rows=6))
            self.assertIs(state.buffers, b)
            self.assertEqual(alloc.call_count, 1)
            # a pool of another size gets new ones
            state.prepare(_view(slots=12, rows=6))
            self.assertEqual(alloc.call_count, 2)
            self.assertEqual(tuple(state.buffers.hist_len.shape), (12,))

    def test_request_rings_follow_the_request_rows(self):
        state = _state(by_request_row=True)
        state.prepare(_view(slots=10, rows=6))
        b = state.buffers
        self.assertTrue(b.by_request_row)
        self.assertEqual(tuple(b.k_ring.shape[:2]), (3, 6))
        state.prepare(_view(slots=20, rows=6))  # slot count does not size them
        self.assertIs(state.buffers, b)

    def test_layer_count_must_match_the_pool(self):
        with self.assertRaises(RuntimeError):
            _state().prepare(_view(layers=2))

    def test_no_allocation_during_capture(self):
        state = _state(by_request_row=False)
        view = _view(slots=10)
        _CAPTURING["now"] = True
        try:
            with self.assertRaisesRegex(RuntimeError, "before CUDA graph capture"):
                state.prepare(view)
        finally:
            _CAPTURING["now"] = False
        state.prepare(view)
        _CAPTURING["now"] = True
        try:
            state.prepare(view)  # a recapture on the same pool allocates nothing
        finally:
            _CAPTURING["now"] = False

    def test_track_materializer_built_and_warmed_once(self):
        state = _state(track_snapshots=True, by_request_row=False)
        view = _view(slots=10, rows=6)
        with mock.patch.object(K, "GdnUcacheTrackMaterializer", _FakeTrack):
            state.prepare(view)
            state.prepare(view)
            b = state.buffers
            self.assertEqual(b.u_ring.dtype, torch.bfloat16)
            self.assertEqual(b.k_ring.dtype, torch.bfloat16)
            self.assertEqual(
                [c[0] for c in _FakeTrack.calls], ["track_init", "warm_up"]
            )
            self.assertIs(_FakeTrack.calls[0][1], view.mamba.temporal)
            self.assertIs(_FakeTrack.calls[0][2], b)
            self.assertEqual(_FakeTrack.calls[1], ("warm_up", 6))
            # a new state pool of the same size: same rings, a new materializer
            view2 = _view(slots=10, rows=6)
            state.prepare(view2)
            self.assertIs(state.buffers, b)
            self.assertEqual(len(_FakeTrack.calls), 4)
            self.assertIs(_FakeTrack.calls[2][1], view2.mamba.temporal)

    def test_no_materializer_without_tracking(self):
        with mock.patch.object(K, "GdnUcacheTrackMaterializer", _FakeTrack):
            _state(track_snapshots=True, by_request_row=True).prepare(_view())
            _state(track_snapshots=False, by_request_row=False).prepare(_view())
        self.assertEqual(_FakeTrack.calls, [])


class TestReset(unittest.TestCase):
    def test_dispatch_by_slot_or_row(self):
        slots, rows = _ids(4, 0, 9), _ids(2, 3, 5)
        for by_row in (False, True):
            state = _state(by_request_row=by_row)
            with (
                mock.patch.object(K, "reset_gdn_ucache_cursors") as by_slot,
                mock.patch.object(K, "reset_gdn_ucache_request_rows") as by_rows,
            ):
                state.reset_state_slots(slots, rows)  # before prepare: no rings
                self.assertEqual(by_slot.call_count + by_rows.call_count, 0)
                state.prepare(_view())
                state.reset_state_slots(slots, rows)
                if by_row:
                    by_slot.assert_not_called()
                    by_rows.assert_called_once()
                    kw = by_rows.call_args.kwargs
                    self.assertIs(kw["request_rows"], rows)
                else:
                    by_rows.assert_not_called()
                    by_slot.assert_called_once()
                    kw = by_slot.call_args.kwargs
                    self.assertIs(kw["slots"], slots)
                self.assertIs(kw["buffers"], state.buffers)

    def test_host_ids_go_to_the_rings_device(self):
        state = _state(by_request_row=True)
        state.prepare(_view())
        with mock.patch.object(K, "reset_gdn_ucache_request_rows") as by_rows:
            state.reset_state_slots(_ids(4), [2, 5])
        got = by_rows.call_args.kwargs["request_rows"]
        self.assertEqual(got.tolist(), [2, 5])
        self.assertEqual(got.device, state.buffers.hist_len.device)

    def test_by_row_needs_rows(self):
        state = _state(by_request_row=True)
        state.prepare(_view())
        with self.assertRaises(RuntimeError):
            state.reset_state_slots(_ids(4), None)


class TestCommit(unittest.TestCase):
    def setUp(self):
        _FakeTrack.calls = []

    def _commit(self, state, io, view):
        calls = _FakeTrack.calls
        cursors = lambda **kw: calls.append(("cursors", kw))
        scatter = lambda *a: calls.append(("conv", a))
        with (
            mock.patch.object(K, "GdnUcacheTrackMaterializer", _FakeTrack),
            mock.patch.object(K, "commit_gdn_ucache_cursors", cursors),
            mock.patch(_SCATTER, scatter),
        ):
            state.prepare(view)
            del calls[:]
            return state.commit(io)

    def test_order_with_track_snapshots(self):
        state = _state(track_snapshots=True, by_request_row=False)
        io, view = _io(track=True), _view(conv_numels=(1, 0))
        self.assertIs(self._commit(state, io, view), True)
        calls = _FakeTrack.calls
        self.assertEqual(
            [c[0] for c in calls], ["cursors", "materialize", "conv", "conv"]
        )
        cursors = calls[0][1]
        self.assertIs(cursors["buffers"], state.buffers)
        self.assertIs(cursors["slots"], io.state_slots)
        self.assertIs(cursors["num_accept_tokens"], io.accept_lens)
        self.assertIsNone(cursors["request_rows"])
        track = calls[1][1]
        self.assertIs(track["slots"], io.state_slots)
        self.assertIs(track["num_accept_tokens"], io.accept_lens)
        self.assertIs(track["track_slots"], io.track_slots)
        self.assertIs(track["steps_to_track"], io.track_steps)
        conv, window = view.mamba.conv[0], view.mamba.intermediate_conv_window[0]
        live, tracked = calls[2][1], calls[3][1]
        for got, dst, steps in (
            (live, io.state_slots, io.last_step),
            (tracked, io.track_slots, io.track_steps),
        ):
            self.assertIs(got[0], conv)
            self.assertIs(got[1], window)
            self.assertIs(got[2], dst)
            self.assertIs(got[3], steps)
            self.assertIs(got[4], io.src_indices_raw)

    def test_request_rings_without_tracking(self):
        state = _state(by_request_row=True)
        io, view = _io(), _view(conv_numels=(1, 1))
        self.assertIs(self._commit(state, io, view), True)
        calls = _FakeTrack.calls
        self.assertEqual([c[0] for c in calls], ["cursors", "conv", "conv"])
        self.assertIs(calls[0][1]["request_rows"], io.request_rows)
        for i, (_, args) in enumerate(calls[1:]):
            self.assertIs(args[0], view.mamba.conv[i])
            self.assertIs(args[2], io.state_slots)

    def test_track_slots_without_tracking_refused(self):
        with self.assertRaises(RuntimeError):
            self._commit(_state(by_request_row=False), _io(track=True), _view())

    def test_commit_before_prepare_refused(self):
        with self.assertRaises(RuntimeError):
            _state().commit(_io())


def _cfg(**kw):
    base = dict(
        mamba_ssm_dtype="float16",
        disaggregation_mode="null",
        disable_radix_cache=False,
        mamba_radix_cache_strategy="extra_buffer_lazy",
        enable_session_radix_cache=False,
        speculative_algorithm="EAGLE",
        speculative_eagle_topk=1,
        speculative_num_draft_tokens=4,
        enable_mixed_chunk=False,
        enable_streaming_session=False,
        enable_unified_memory=False,
        enable_page_major_kv_layout=False,
        enable_int8_mamba_checkpoint=False,
        enable_linear_replayssm=False,
        enable_linear_replayssm_spec=False,
    )
    base.update(kw)
    return SimpleNamespace(**base)


class TestConfigReasons(unittest.TestCase):
    def test_recipe_passes(self):
        self.assertEqual(UcacheState.config_reasons(_cfg()), [])
        self.assertEqual(
            UcacheState.config_reasons(_cfg(speculative_algorithm="NEXTN")), []
        )
        self.assertTrue(S.extra_buffer_radix(_cfg()))
        self.assertFalse(S.extra_buffer_radix(_cfg(disable_radix_cache=True)))

    def test_each_guard(self):
        for kw in (
            dict(mamba_ssm_dtype="bfloat16"),
            dict(mamba_ssm_dtype=None),
            dict(mamba_radix_cache_strategy="no_buffer"),
            dict(enable_session_radix_cache=True),
            dict(speculative_algorithm=None),
            dict(speculative_algorithm="NGRAM"),
            dict(speculative_eagle_topk=4),
            dict(speculative_num_draft_tokens=9),
            dict(speculative_num_draft_tokens=None),
            dict(enable_mixed_chunk=True),
            dict(enable_streaming_session=True),
            dict(enable_unified_memory=True),
            dict(enable_page_major_kv_layout=True),
            dict(enable_int8_mamba_checkpoint=True),
            dict(enable_linear_replayssm_spec=True),
        ):
            with self.subTest(**kw):
                self.assertEqual(len(UcacheState.config_reasons(_cfg(**kw))), 1)

    def test_pd_roles(self):
        # a decode role without the radix cache is served
        decode = _cfg(disaggregation_mode="decode", disable_radix_cache=True)
        self.assertEqual(UcacheState.config_reasons(decode), [])
        # ... with an fp16 state, and the prefill is named
        decode.mamba_ssm_dtype = "bfloat16"
        reasons = UcacheState.config_reasons(decode)
        self.assertEqual(len(reasons), 1)
        self.assertIn("PD prefill", reasons[0])
        # the radix cache is aggregated-only
        self.assertEqual(
            len(UcacheState.config_reasons(_cfg(disaggregation_mode="decode"))), 1
        )
        # a prefill role runs no verify
        prefill = _cfg(disaggregation_mode="prefill", disable_radix_cache=True)
        self.assertEqual(len(UcacheState.config_reasons(prefill)), 1)

    def test_environment_adaptive_and_device(self):
        for env in (
            {"SGLANG_RAGGED_VERIFY_MODE": "compact"},
            {"SGLANG_ENABLE_PP_SPEC": "1"},
        ):
            with self.subTest(**env), mock.patch.dict("os.environ", env):
                self.assertEqual(len(UcacheState.config_reasons(_cfg())), 1)
        adaptive = _cfg(speculative_adaptive=True, speculative_adaptive_config="x")
        target = "sglang.srt.speculative.adaptive_spec_params.resolve_candidate_steps_from_config"
        with mock.patch(target, return_value=[1, 3, 7]):
            self.assertEqual(S.max_draft_tokens(adaptive), 8)
            self.assertEqual(UcacheState.config_reasons(adaptive), [])
        with mock.patch(target, return_value=[1, 3, 8]):
            self.assertEqual(len(UcacheState.config_reasons(adaptive)), 1)
        with (
            mock.patch("torch.cuda.is_available", return_value=True),
            mock.patch("torch.cuda.get_device_capability", return_value=(10, 0)),
        ):
            self.assertEqual(len(UcacheState.config_reasons(_cfg())), 1)

    def test_missing_fields_read_defensively(self):
        reasons = UcacheState.config_reasons(SimpleNamespace())
        self.assertTrue(reasons)  # no dtype, no spec: refused, no AttributeError


if __name__ == "__main__":
    unittest.main()
