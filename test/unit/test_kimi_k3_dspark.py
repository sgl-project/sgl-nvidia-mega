# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the SGLang project
"""Focused q6 handoff, borrowed-state and replay-metadata regressions (CPU)."""

from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest
import torch
from msgspec.structs import replace
from test_kimi_k3_contract import (
    HybridPoolView,
    HybridStepView,
    KdaPoolView,
    KdaStateView,
    KdaStepView,
    MlaPoolView,
    MlaStepView,
    PreparationView,
    registry,
)

from sgl_nvidia_mega.kimi_k3 import providers, spec
from sgl_nvidia_mega.kimi_k3.executor import (
    KimiK3MegakernelExecutor,
    _validate_step_inputs,
    _verify_mla_metadata,
)
from sgl_nvidia_mega.kimi_k3.shape_policy import token_widths

Step = registry.Step


def _resources(**overrides):
    fields = {
        name: {}
        for name in (
            "attention",
            "feed_forward",
            "moe_front_weights",
            "before",
            "fused_before",
            "boxes",
            "scratch",
            "workspaces",
            "runtime_handles",
            "payload_modules",
            "memory_before",
            "memory_after",
        )
    }
    fields.update(
        model=SimpleNamespace(),
        layers=(),
        final=None,
        compilation_inventory=[],
        deadline_ns=30_000_000_000,
        tokens_per_request=6,
    )
    return providers.ResourceSet(**(fields | overrides))


def _batch(mode, requests):
    return SimpleNamespace(
        batch_size=requests,
        forward_mode=SimpleNamespace(
            is_decode=lambda: mode == "decode",
            is_target_verify=lambda: mode == "target_verify",
        ),
    )


class _MetadataTensor(torch.Tensor):
    """CUDA metadata without allocating CUDA storage."""

    @staticmethod
    def __new__(cls, shape, dtype):
        tensor = torch.Tensor._make_subclass(cls, torch.empty(0), False)
        tensor.metadata = torch.empty(shape, dtype=dtype, device="meta")
        return tensor

    @property
    def shape(self):
        return self.metadata.shape

    @property
    def dtype(self):
        return self.metadata.dtype

    @property
    def device(self):
        return torch.device("cuda:0")

    def dim(self):
        return self.metadata.dim()

    def stride(self, dim=None):
        return self.metadata.stride() if dim is None else self.metadata.stride(dim)


class _Consumer:
    def __init__(self, tap):
        self.tap = tap
        self.routes = []

    def run(self, delta, prefix, blocks, out, aux=None):
        self._record("ordinary", aux)
        return out

    def run_ring(self, prefix, blocks, out, aux=None):
        self._record("ring", aux)
        return out

    def _record(self, route, aux):
        if aux is not None:
            self.routes.append((route, aux.stride()))
            aux.fill_(self.tap)


@pytest.mark.parametrize("q", (6,))
def test_verify_metadata_separates_token_rows_from_request_rows(q):
    rows, requests, width = 2 * q, 2, 35840
    ids = _MetadataTensor((rows,), torch.int64)
    view = HybridStepView(
        mla=MlaStepView(
            _MetadataTensor((requests, 128), torch.int32),
            _MetadataTensor((requests,), torch.int32),
            _MetadataTensor((rows,), torch.int64),
            tokens_per_request=q,
        ),
        kda=KdaStepView(
            state_indices=_MetadataTensor((requests,), torch.int32),
            track_decode_state=Mock(),
            verify_scratch_indices=_MetadataTensor((requests,), torch.int32),
        ),
        aux_hidden_states=_MetadataTensor((rows, width), torch.bfloat16),
    )
    kwargs = {
        "batch": rows,
        "device": torch.device("cuda:0"),
        "tokens_per_request": q,
        "aux_width": width,
    }
    _validate_step_inputs(ids, None, view, **kwargs)
    for bad, message in (
        (replace(view, mla=replace(view.mla, tokens_per_request=1)), "query width"),
        (
            replace(
                view,
                mla=replace(
                    view.mla, sequence_lengths=_MetadataTensor((rows,), torch.int32)
                ),
            ),
            "sequence lengths has shape",
        ),
        (
            replace(
                view,
                kda=KdaStepView(
                    state_indices=view.kda.state_indices,
                    track_decode_state=Mock(),
                ),
            ),
            "scratch indices is not a tensor",
        ),
        (
            replace(view, aux_hidden_states=None),
            "auxiliary hidden states is not a tensor",
        ),
        (
            replace(
                view, aux_hidden_states=_MetadataTensor((rows, 7168), torch.bfloat16)
            ),
            "auxiliary hidden states has shape",
        ),
    ):
        with pytest.raises(RuntimeError, match=message):
            _validate_step_inputs(ids, None, bad, **kwargs)


def test_verify_binding_borrows_and_seals_both_intermediate_pools():
    q = 6
    conv, temporal = torch.zeros(1), torch.zeros(1)
    windows, checkpoints = torch.zeros(2, q + 1, 3, 4), torch.zeros(2, q + 1, 1)
    state = KdaStateView((conv,), temporal, (windows,), checkpoints)
    hybrid = HybridPoolView(
        mla=MlaPoolView(
            reserved_capacity=128,
            serving_capacity=lambda: 128,
            post_capture_active=False,
            page_size=64,
            logical_dtype=torch.float8_e4m3fn,
            storage_dtype=torch.uint8,
            layer_mapping={0: 0},
            caches={0: torch.zeros(1)},
        ),
        kda=KdaPoolView(capacity=4, layer_mapping={1: 0}, states={1: state}),
    )
    mla, kda = Mock(), Mock()
    resources = _resources(
        tokens_per_request=q,
        model=SimpleNamespace(
            config=SimpleNamespace(
                full_attention_layer_ids=(0,), linear_layer_ids=(1,)
            ),
            parameters=lambda: (),
        ),
        attention={
            0: providers.PreparedAttention(0, "mla", None, None, mla, None),
            1: providers.PreparedAttention(1, "kda", None, None, kda, None),
        },
        aux_capture_layers=(7,),
    )
    with (
        patch.object(providers, "_validate_pool_tensor"),
        patch.object(providers, "_cuda_memory", return_value={}),
        patch.object(torch.cuda, "synchronize"),
    ):
        resources.bind(hybrid)
        resources.bind(hybrid)
        kda.prepare_verify_caches.assert_called_once_with(
            (conv, temporal), (windows, checkpoints)
        )
        kda.prepare_caches.assert_not_called()
        assert resources.attention[1].intermediates[0] is windows
        assert "kda_intermediate_conv.1" in resources.binding["local_identities"]
        changed = HybridPoolView(
            mla=hybrid.mla,
            kda=KdaPoolView(
                capacity=4,
                layer_mapping={1: 0},
                states={1: replace(state, intermediate_ssm=checkpoints.clone())},
            ),
        )
        with pytest.raises(RuntimeError, match="binding changed"):
            resources.bind(changed)


def test_aux_preparation_uses_next_consumers_and_is_idempotent():
    taps = (0, 7, 23, 51, 67, 83, 92)
    model = SimpleNamespace(model=SimpleNamespace(dspark_layers_to_capture=list(taps)))
    before, final = {index: Mock() for index in range(93)}, Mock()
    resources = _resources(
        model=model, layers=tuple(range(93)), before=before, final=final
    )
    resources.prepare_aux_capture()
    resources.prepare_aux_capture()
    assert resources.aux_capture_layers == taps
    for layer_id, consumer in before.items():
        if layer_id - 1 in taps:
            consumer.bind_aux.assert_called_once_with(mixture=True)
        else:
            consumer.bind_aux.assert_not_called()
    final.bind_aux.assert_called_once_with(mixture=True)
    assert len(resources.compilation_inventory) == len(taps)
    model.model.dspark_layers_to_capture = [7]
    with pytest.raises(RuntimeError, match="layers changed"):
        resources.prepare_aux_capture()


@pytest.mark.parametrize("q", (6,))
def test_verify_plan_and_logits_use_packed_auxiliary_features(q):
    logits = Mock(return_value="logits")
    model = SimpleNamespace(
        language_model=SimpleNamespace(logits_processor=logits, lm_head=object())
    )
    built = spec._Built(model, _resources(tokens_per_request=q), token_widths(q))
    for requests in range(1, 64 // q + 1):
        step = built.plan(torch.empty(requests * q), _batch("target_verify", requests))
        assert (step.mode, step.rows) == ("target_verify", requests * q)
    for mode, requests, rows in (
        ("decode", q, q),
        ("target_verify", 2, q),
        ("extend", 1, q),
        ("target_verify", 64 // q + 1, (64 // q + 1) * q),
        ("target_verify", 2, 2 * q + 1),
    ):
        assert built.plan(torch.empty(rows), _batch(mode, requests)) is None
    ids, hidden, auxiliary = torch.zeros(2 * q, dtype=torch.int64), object(), object()
    view = SimpleNamespace(hybrid=SimpleNamespace(aux_hidden_states=auxiliary))
    built.executor = Mock(run=Mock(return_value=hidden))
    batch = _batch("target_verify", 2)
    assert (
        built.forward(Step("target_verify", 2 * q, view), None, ids, None, batch)
        == "logits"
    )
    assert logits.call_args.args == (
        ids,
        hidden,
        model.language_model.lm_head,
        batch,
        auxiliary,
    )


@pytest.mark.parametrize("q", (1, 6))
def test_padding_reuses_owned_storage_across_retirement_and_restoration(q):
    scratch = {
        "mla.verify_inactive": torch.empty(2, dtype=torch.bool),
        "mla.verify_lengths": torch.empty(2, dtype=torch.int32),
        "mla.verify_slots": torch.empty(2 * q, dtype=torch.int64),
    }
    lengths = torch.tensor([262144, 17], dtype=torch.int32)
    slots = torch.arange(2 * q, dtype=torch.int64) + 2**40
    original = lengths.clone(), slots.clone()
    pointers = tuple(x.data_ptr() for x in scratch.values())
    for physical in ([0, 0], [3, 0], [-1, 7], [3, 7]):
        out_slots, out_lengths = _verify_mla_metadata(
            torch.tensor(physical, dtype=torch.int32), lengths, slots, scratch, q
        )
        assert out_lengths.tolist() == [
            int(lengths[i]) if s > 0 else 0 for i, s in enumerate(physical)
        ]
        assert out_slots.tolist() == [
            int(slots[i]) if physical[i // q] > 0 else -1 for i in range(2 * q)
        ]
        assert torch.equal(lengths, original[0]) and torch.equal(slots, original[1])
        assert pointers == tuple(x.data_ptr() for x in scratch.values())


@pytest.mark.parametrize("q", (1, 6))
def test_decode_tracks_state_but_verify_only_writes_checkpoints(q):
    # Four representative layers retain both attention kinds and fused/unfused
    # auxiliary consumers; arithmetic providers are explicitly mocked.
    rows, taps = 2 * q, (0, 1, 92) if q > 1 else ()
    scratch = {
        name: torch.zeros(rows, 7168)
        for name in (
            "prefix",
            "blocks",
            "hidden.0",
            "hidden.1",
            "hidden.out",
            "mla.projected",
            "kda.projected",
            "attention.local",
        )
    }
    scratch.update(
        {
            "mla.verify_inactive": torch.empty(2, dtype=torch.bool),
            "mla.verify_lengths": torch.empty(2, dtype=torch.int32),
            "mla.verify_slots": torch.empty(rows, dtype=torch.int64),
        }
    )
    before, final = {i: _Consumer(i - 1) for i in (0, 1, 2, 92)}, _Consumer(92)
    attention = {
        i: SimpleNamespace(
            kind="mla" if i in (0, 92) else "kda",
            projection=Mock(),
            output=Mock(),
            core=Mock(),
            cache=object(),
            caches=(object(), object()),
            intermediates=(object(), object()),
        )
        for i in (0, 1, 2, 92)
    }
    resources = _resources(
        layers=tuple(SimpleNamespace(layer_idx=i) for i in attention),
        attention=attention,
        scratch=scratch,
        before=before,
        final=final,
        feed_forward={
            i: SimpleNamespace(
                run=lambda x, out, consume=None: (
                    out if consume is None else consume(out)
                )
            )
            for i in attention
        },
        fused_before={1: before[2], 92: final},
        aux_capture_layers=taps,
        binding_fingerprint="bound",
        tokens_per_request=q,
        widths=(rows,),
        prefetch=Mock(),
    )
    track = Mock()
    view = HybridStepView(
        mla=MlaStepView(
            torch.zeros(2, 1, dtype=torch.int32),
            torch.tensor([13, 14], dtype=torch.int32),
            torch.arange(rows, dtype=torch.int64) + 2**40,
            tokens_per_request=q,
        ),
        kda=KdaStepView(
            state_indices=torch.tensor([3, 0], dtype=torch.int32),
            track_decode_state=track,
            verify_scratch_indices=torch.tensor([0, 7], dtype=torch.int32)
            if q > 1
            else None,
        ),
        aux_hidden_states=torch.full((rows, len(taps) * 7168), -1.0) if q > 1 else None,
    )
    with patch("sgl_nvidia_mega.kimi_k3.executor._validate_step_inputs"):
        output = KimiK3MegakernelExecutor(resources).run(
            torch.zeros(rows, dtype=torch.int64), torch.zeros(rows, 7168), view
        )
    assert output.data_ptr() == scratch["hidden.out"].data_ptr()
    for i in (1, 2):
        core = attention[i].core
        call = core.run_verify.call_args if q > 1 else core.run.call_args
        assert call.kwargs["slots"] is view.kda.state_indices
        assert call.kwargs["caches"] is attention[i].caches
        if q > 1:
            core.run.assert_not_called()
            assert set(call.kwargs) == {
                "slots",
                "caches",
                "intermediates",
                "scratch_indices",
            }
            assert call.kwargs["intermediates"] is attention[i].intermediates
            assert call.kwargs["scratch_indices"] is view.kda.verify_scratch_indices
        else:
            core.run_verify.assert_not_called()
    assert [c.args[0] for c in track.call_args_list] == ([] if q > 1 else [1, 2])
    for i in (0, 92):
        kwargs = attention[i].core.run.call_args.kwargs
        assert kwargs["q"] == q and kwargs["block_table"] is view.mla.block_table
        assert kwargs["lengths"].tolist() == ([13, 0] if q > 1 else [13, 14])
    for index, tap in enumerate(taps):
        assert torch.all(
            view.aux_hidden_states[:, index * 7168 : (index + 1) * 7168] == tap
        )
    if q > 1:
        assert before[1].routes == [("ordinary", (len(taps) * 7168, 1))]
        assert before[2].routes == final.routes == [("ring", (len(taps) * 7168, 1))]
    assert view.mla.sequence_lengths.tolist() == [13, 14]
    assert view.mla.write_slots.tolist() == list(range(2**40, 2**40 + rows))
    resources.prefetch.join.assert_called_once_with()


def test_prepare_configures_aux_before_binding():
    resources = Mock()
    spec._prepare(
        SimpleNamespace(resources=resources), PreparationView(hybrid=object())
    )
    assert [call[0] for call in resources.mock_calls] == ["prepare_aux_capture", "bind"]
