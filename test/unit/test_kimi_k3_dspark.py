# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the SGLang project
"""Fixed-q4-8 routing, borrowed-state, padding and auxiliary-output contracts."""

from __future__ import annotations

import ast
import importlib.util
import inspect
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest
import torch
from msgspec.structs import replace

try:
    from sglang.kernels.plugin_kernels.nvidia.spec import BuildContext, Feature, Step
except ModuleNotFoundError as error:
    if error.name not in (
        "sglang.kernels",
        "sglang.kernels.plugin_kernels",
        "sglang.kernels.plugin_kernels.nvidia",
        "sglang.kernels.plugin_kernels.nvidia.spec",
    ):
        raise
    from sglang.srt.megakernels.spec import BuildContext, Feature, Step

try:
    from sglang.srt.kernel_plugins.nvidia.binding import (
        HybridPoolView,
        HybridStepView,
        KdaPoolView,
        KdaStateView,
        KdaStepView,
        MlaPoolView,
        MlaStepView,
        PreparationView,
    )
except ModuleNotFoundError as error:
    if error.name not in (
        "sglang.srt.kernel_plugins",
        "sglang.srt.kernel_plugins.nvidia",
        "sglang.srt.kernel_plugins.nvidia.binding",
    ):
        raise
    from sglang.srt.megakernels.binding import (
        HybridPoolView,
        HybridStepView,
        KdaPoolView,
        KdaStateView,
        KdaStepView,
        MlaPoolView,
        MlaStepView,
        PreparationView,
    )

from sgl_nvidia_mega.kimi_k3 import providers, spec
from sgl_nvidia_mega.kimi_k3.executor import (
    KimiK3MegakernelExecutor,
    _validate_step_inputs,
    _verify_mla_metadata,
)
from sgl_nvidia_mega.kimi_k3.shape_policy import (
    MAX_CONTEXT,
    VERIFY_QUERIES,
    mla_max_seq_len,
    token_widths,
)


def _resources(**overrides):
    values = {
        "model": SimpleNamespace(),
        "layers": (),
        "attention": {},
        "feed_forward": {},
        "moe_front_weights": {},
        "before": {},
        "final": None,
        "fused_before": {},
        "boxes": {},
        "scratch": {},
        "workspaces": {},
        "runtime_handles": {},
        "payload_modules": {},
        "compilation_inventory": [],
        "memory_before": {},
        "memory_after": {},
        "deadline_ns": 30_000_000_000,
        "tokens_per_request": 4,
    }
    values.update(overrides)
    return providers.ResourceSet(**values)


def _batch(mode, requests):
    return SimpleNamespace(
        batch_size=requests,
        forward_mode=SimpleNamespace(
            is_decode=lambda: mode == "decode",
            is_target_verify=lambda: mode == "target_verify",
        ),
    )


@pytest.mark.parametrize("q", VERIFY_QUERIES)
def test_fixed_dspark_only_configuration(q):
    cfg = SimpleNamespace(
        cuda_graph_config=SimpleNamespace(decode=SimpleNamespace(backend="full")),
        context_length=8192,
        speculative_algorithm="DSPARK",
        speculative_num_draft_tokens=q,
        speculative_adaptive=False,
    )
    hf_config = SimpleNamespace(max_position_embeddings=1048576)
    assert spec._config_disable_reasons(cfg, hf_config) == []
    assert Feature.RAGGED_VERIFY not in spec.SPEC.features
    assert Feature.CONTEXT_PARALLEL not in spec.SPEC.features
    assert spec.SPEC.requires.server_args["decode_attention_backend"] == {"trtllm_mla"}
    for field, value, message in (
        ("speculative_algorithm", "EAGLE", "requires DSPARK"),
        ("speculative_num_draft_tokens", 3, "fixed q4-8"),
        ("speculative_num_draft_tokens", 9, "fixed q4-8"),
        ("speculative_num_draft_tokens", 6.0, "fixed q4-8"),
        ("speculative_adaptive", True, "adaptive"),
        ("context_length", 262145, "1..262144"),
    ):
        changed = SimpleNamespace(**(vars(cfg) | {field: value}))
        assert any(
            message in reason
            for reason in spec._config_disable_reasons(changed, hf_config)
        )


@pytest.mark.parametrize("q", (1, *VERIFY_QUERIES))
@pytest.mark.parametrize(
    "length", (1, 4096, 8192, 32768, 131072, 200000, 256000, 262144)
)
def test_context_range_and_verify_scheduling_hint(length, q):
    cfg = SimpleNamespace(
        cuda_graph_config=SimpleNamespace(decode=SimpleNamespace(backend="full")),
        context_length=length,
        speculative_algorithm=None if q == 1 else "DSPARK",
        speculative_num_draft_tokens=q,
        speculative_adaptive=False,
    )
    text = SimpleNamespace(max_position_embeddings=1048576)
    # Both the native multimodal K3 config and its text config are accepted.
    for hf_config in (text, SimpleNamespace(text_config=text)):
        assert spec._config_disable_reasons(cfg, hf_config) == []
    assert mla_max_seq_len(length, q) == length + (q if q > 1 else 0)
    assert text.max_position_embeddings == 1048576


@pytest.mark.parametrize(
    "length", (None, True, False, 0, -1, 8192.0, "8192", 262145, 2**31)
)
def test_invalid_context_refuses_before_provider_construction(length):
    cfg = SimpleNamespace(
        cuda_graph_config=SimpleNamespace(decode=SimpleNamespace(backend="full")),
        context_length=length,
        speculative_algorithm=None,
    )
    assert any(
        "1..262144" in reason
        for reason in spec._config_disable_reasons(
            cfg, SimpleNamespace(max_position_embeddings=1048576)
        )
    )
    with (
        patch.object(providers, "_module") as load_payload,
        pytest.raises(ValueError, match="1..262144"),
    ):
        providers.build_resources(
            None, context_length=length, memory_before={}, deadline_ns=1
        )
    load_payload.assert_not_called()


def test_context_respects_native_model_cap_and_rope_policy_without_override():
    cfg = SimpleNamespace(
        cuda_graph_config=SimpleNamespace(decode=SimpleNamespace(backend="full")),
        context_length=32768,
        speculative_algorithm=None,
    )
    # Native context derivation, including scaling and original-window handling,
    # is reused; the plugin never changes model metadata to admit a larger limit.
    cases = (
        (SimpleNamespace(max_position_embeddings=32768), True),
        (SimpleNamespace(max_position_embeddings=32767), False),
        (
            SimpleNamespace(max_position_embeddings=8192, rope_scaling={"factor": 4}),
            True,
        ),
        (
            SimpleNamespace(
                max_position_embeddings=8192,
                rope_scaling={"factor": 4, "original_max_position_embeddings": 8192},
            ),
            False,
        ),
        (SimpleNamespace(), False),  # Native unspecified-model fallback is 2048.
    )
    for text, admitted in cases:
        original = repr(vars(text))
        reasons = spec._config_disable_reasons(cfg, SimpleNamespace(text_config=text))
        assert (not reasons) is admitted
        if not admitted:
            assert any("exceeds native model limit" in reason for reason in reasons)
        assert repr(vars(text)) == original
    for q in (0, 2, 3, 9, True, 6.0):
        with pytest.raises(ValueError, match="q1 decode or fixed q4-8"):
            mla_max_seq_len(MAX_CONTEXT, q)


def test_prepared_mla_uses_scheduling_hint_without_changing_backend_or_lengths():
    tree = ast.parse(inspect.getsource(providers.build_resources))
    hint = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Name) and target.id == "max_seq_len"
            for target in node.targets
        )
    )
    assert (
        ast.unparse(hint.value) == "mla_max_seq_len(context_length, tokens_per_request)"
    )
    calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "PreparedMla"
    ]
    assert len(calls) == 1
    kwargs = {item.arg: ast.unparse(item.value) for item in calls[0].keywords}
    assert kwargs == {
        "max_context": "max_seq_len",
        "core": "'trtllm'",
        "dcp": "None",
        "queries": "(tokens_per_request,)",
    }


def test_kda_verify_policy_uses_requests_and_preserves_decode_policy():
    for q in VERIFY_QUERIES:
        assert providers._kda_policy(q) == {
            (requests, q): (6288, 1) for requests in range(1, 64 // q + 1)
        }
    assert providers._kda_policy(1) == {
        rows: (6288, 0 if rows in (16, 32, 64) else 1 if rows == 1 else 2)
        for rows in spec.WIDTHS
    }


def test_common_token_rows_and_projection_bins_include_non_power_of_two_widths():
    assert token_widths(1) == spec.WIDTHS == (1, 2, 3, 4, 5, 6, 7, 8, 16, 32, 64)
    for q, maximum in zip(VERIFY_QUERIES, (64, 60, 60, 63, 64), strict=True):
        widths = token_widths(q)
        assert widths == tuple(range(q, maximum + 1, q))
        for pitch in (3648, 6288):
            policies = providers._projection_policies(pitch, widths)
            for rows in widths:
                tile = next(n for n in (8, 16, 32, 64) if rows <= n)
                assert policies[rows][1] == tile
                assert (
                    policies[rows]
                    == providers._projection_policies(pitch, (tile,))[tile]
                )
    for q in (0, 2, 3, 9, True, 4.0):
        with pytest.raises(ValueError, match="q1 decode or fixed q4-8"):
            token_widths(q)
    for rows in (0, 65):
        with pytest.raises(ValueError, match="projection rows"):
            providers._projection_policies(3648, (rows,))


@pytest.mark.skipif(
    importlib.util.find_spec("cutlass") is None,
    reason="prepared KDA host contract requires the pinned CuTe package",
)
@pytest.mark.parametrize("q", VERIFY_QUERIES)
def test_prepared_kda_verify_keys_and_checkpoint_capacity(q):
    from sgl_nvidia_mega.kimi_k3.cutedsl.kda import prepared

    attention = SimpleNamespace(
        conv1d=SimpleNamespace(weight=torch.zeros(4608, 1, 4), bias=None),
        dt_bias=torch.zeros(1536),
        A_log=torch.zeros(12),
        f_b_proj=SimpleNamespace(weight=torch.zeros(1536, 128)),
        o_norm=SimpleNamespace(weight=torch.ones(128), eps=1e-6),
        gate_lower_bound=-5.0,
    )
    caches = (
        torch.zeros(3, 3, 4608, dtype=torch.bfloat16),
        torch.zeros(3, 12, 128, 128),
    )

    def intermediate(capacity):
        return (
            torch.zeros(2, capacity, 3, 4608, dtype=torch.bfloat16),
            torch.zeros(2, capacity, 12, 128, 128),
        )

    core = prepared.PreparedKda(attention, providers._kda_policy(q), False)
    checkpoints = intermediate(q)
    # Compile and device conversion are mocked, not the prepared policy, operand
    # layout validation or dispatch. Real fused math is the component probe's gate.
    with patch.object(
        prepared, "executable", side_effect=lambda *args: Mock()
    ) as compile_kernel:
        core.prepare_verify_caches(caches, checkpoints)
    expected = {(requests, q) for requests in range(1, 64 // q + 1)}
    assert set(core.verify_kernels) == set(core.verify_layouts) == expected
    assert compile_kernel.call_count == len(expected)
    assert all(call.args[-2:] == (q, True) for call in compile_kernel.call_args_list)
    projected = torch.zeros(2 * q, 6288, dtype=torch.bfloat16)
    out = torch.empty(2 * q, 1536, dtype=torch.bfloat16)
    slots, scratch = (
        torch.tensor([1, 0], dtype=torch.int32),
        torch.tensor([0, 1], dtype=torch.int32),
    )
    kwargs = {
        "caches": caches,
        "slots": slots,
        "intermediates": checkpoints,
        "scratch_indices": scratch,
    }
    with (
        patch.object(prepared, "to_cute", side_effect=lambda value, align: value),
        patch.object(prepared, "stream", return_value=None),
    ):
        assert core.run_verify(projected, out, **kwargs) is out
    core.verify_kernels[(2, q)].assert_called_once()
    for key, kernel in core.verify_kernels.items():
        if key != (2, q):
            kernel.assert_not_called()
    with pytest.raises(ValueError, match="uniform rows per request"):
        core.run_verify(projected[:-1], out, **kwargs)
    with pytest.raises(ValueError, match="unprepared split-cache"):
        core.run_verify(torch.zeros(18, 6288, dtype=torch.bfloat16), out, **kwargs)
    with pytest.raises(ValueError, match="checkpoints"):
        core.prepare_verify_caches(caches, intermediate(q - 1))
    for bad in ((64 // q + 1, q), (1, 3), (1, 9)):
        invalid = prepared.PreparedKda(attention, {bad: (6288, 1)}, False)
        with pytest.raises(ValueError, match="unsupported split-cache"):
            invalid.prepare_verify_caches(caches, checkpoints)


def test_shared_front_selects_the_deployment_verify_policy():
    tree = ast.parse(inspect.getsource(providers.build_resources))
    constructors = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "SharedFront"
    ]
    assert len(constructors) == 1
    query_width = next(
        keyword.value for keyword in constructors[0].keywords if keyword.arg == "q"
    )
    assert isinstance(query_width, ast.Name) and query_width.id == "tokens_per_request"
    assert (
        inspect.signature(providers.build_resources)
        .parameters["tokens_per_request"]
        .default
        == 1
    )


@pytest.mark.parametrize("q", VERIFY_QUERIES)
def test_build_filters_token_rows_and_freezes_query_width(q):
    from sgl_nvidia_mega.kimi_k3 import compat

    model = SimpleNamespace(language_model=SimpleNamespace())
    resources = _resources(tokens_per_request=q)
    context = BuildContext(tuple(range(1, 70)), q, 30.0, object())
    with (
        patch.object(compat, "validate_runtime_compatibility"),
        patch.object(
            spec, "get_model", return_value=SimpleNamespace(context_length=8192)
        ),
        patch.object(providers, "_cuda_memory", return_value={}),
        patch.object(providers, "build_resources", return_value=resources) as build,
    ):
        built = spec._build(model, None, context)
    assert built.widths == token_widths(q)
    assert built.tokens_per_request == q
    assert built.draft_sampler is None
    assert build.call_args.kwargs["tokens_per_request"] == q


@pytest.mark.parametrize("q", VERIFY_QUERIES)
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


@pytest.mark.parametrize("taps", [None, [], [7, 7], [23, 7], [-1], [93]])
def test_aux_preparation_rejects_missing_or_invalid_taps(taps):
    resources = _resources(
        model=SimpleNamespace(model=SimpleNamespace(dspark_layers_to_capture=taps)),
        layers=tuple(range(93)),
    )
    with pytest.raises(ValueError):
        resources.prepare_aux_capture()


@pytest.mark.parametrize("q", VERIFY_QUERIES)
def test_verify_binding_borrows_and_seals_both_intermediate_pools(q):
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


@pytest.mark.skipif(
    not torch.cuda.is_available(), reason="native pool layout requires CUDA"
)
@pytest.mark.parametrize("q", VERIFY_QUERIES)
def test_native_speculative_state_layer_view_binds_without_layout_adaptation(q):
    from sglang.srt.mem_cache.memory_pool import MambaPool

    device = torch.device("cuda", torch.cuda.current_device())
    state = MambaPool.SpeculativeState(
        conv=[torch.zeros((1, 5, 3, 4608), dtype=torch.bfloat16, device=device)],
        temporal=torch.zeros((1, 5, 12, 128, 128), dtype=torch.float32, device=device),
        intermediate_conv_window=[
            torch.zeros((1, 3, q, 3, 4608), dtype=torch.bfloat16, device=device)
        ],
        intermediate_ssm=torch.zeros(
            (1, 3, q, 12, 128, 128), dtype=torch.float32, device=device
        ),
    )
    layer = state.at_layer_idx(0)
    assert isinstance(layer, MambaPool.SpeculativeState)
    assert layer.replayssm_d is None and layer.replayssm_rawv is None
    mla_cache = torch.zeros((3, 64, 576), dtype=torch.uint8, device=device)
    hybrid = HybridPoolView(
        mla=MlaPoolView(
            reserved_capacity=128,
            serving_capacity=lambda: 128,
            post_capture_active=False,
            page_size=64,
            logical_dtype=torch.float8_e4m3fn,
            storage_dtype=torch.uint8,
            layer_mapping={0: 0},
            caches={0: mla_cache},
        ),
        kda=KdaPoolView(
            capacity=4,
            layer_mapping={1: 0},
            states={
                1: KdaStateView(
                    conv=tuple(layer.conv),
                    temporal=layer.temporal,
                    intermediate_conv_window=tuple(layer.intermediate_conv_window),
                    intermediate_ssm=layer.intermediate_ssm,
                )
            },
        ),
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
    # Only kernel compilation is mocked: the actual provider checks every native
    # tensor's CUDA device, dtype, shape, stride, contiguity and alignment.
    resources.bind(hybrid)
    resources.bind(hybrid)
    mla.prepare_cache.assert_called_once_with(mla_cache)
    kda.prepare_verify_caches.assert_called_once_with(
        (layer.conv[0], layer.temporal),
        (layer.intermediate_conv_window[0], layer.intermediate_ssm),
    )
    kda.prepare_caches.assert_not_called()
    for name, tensor in (
        ("kda_conv.1", layer.conv[0]),
        ("kda_temporal.1", layer.temporal),
        ("kda_intermediate_conv.1", layer.intermediate_conv_window[0]),
        ("kda_intermediate_ssm.1", layer.intermediate_ssm),
    ):
        assert resources.binding["local_identities"][name] == tensor.data_ptr()


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


@pytest.mark.parametrize("q", VERIFY_QUERIES)
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


class _FeedForward:
    def run(self, source, out, consume=None):
        return out if consume is None else consume(out)


def _verify_metadata_scratch(device="cpu"):
    return {
        "mla.verify_inactive": torch.ones(16, dtype=torch.bool, device=device),
        "mla.verify_lengths": torch.full((16,), -77, dtype=torch.int32, device=device),
        "mla.verify_slots": torch.full((64,), -99, dtype=torch.int64, device=device),
    }


@pytest.mark.parametrize("q", VERIFY_QUERIES)
@pytest.mark.parametrize("context_length", (8192, 32768, 131072, 262144))
def test_verify_mla_padding_uses_physical_slots_without_mutating_borrowed_metadata(
    q, context_length
):
    """Native slot-zero padding must not race writes; scratch index zero is live."""
    scratch = _verify_metadata_scratch()
    pointers = {name: value.data_ptr() for name, value in scratch.items()}
    lengths = torch.tensor(
        [context_length - q, context_length, context_length + q, 5], dtype=torch.int32
    )
    slots = torch.arange(4 * q, dtype=torch.int64) + 2**40
    inputs = (lengths.clone(), slots.clone())
    for physical, expected_inactive in (
        ([7, 2, 3, 4], [False, False, False, False]),
        ([7, 0, -1, 4], [False, True, True, False]),
        ([0, 0, 0, 0], [True, True, True, True]),
        ([-1, -2, 0, -1], [True, True, True, True]),
        ([7, 2, 3, 4], [False, False, False, False]),
    ):
        state_indices = torch.tensor(physical, dtype=torch.int32)
        normalized_slots, normalized_lengths = _verify_mla_metadata(
            state_indices, lengths, slots, scratch, q
        )
        assert state_indices.tolist() == physical
        assert torch.equal(lengths, inputs[0]) and torch.equal(slots, inputs[1])
        assert scratch["mla.verify_inactive"][:4].tolist() == expected_inactive
        for request, inactive in enumerate(expected_inactive):
            assert normalized_lengths[request].item() == (
                0 if inactive else lengths[request].item()
            )
            assert normalized_slots[request * q : (request + 1) * q].tolist() == (
                [-1] * q
                if inactive
                else slots[request * q : (request + 1) * q].tolist()
            )
        assert normalized_slots.dtype == torch.int64
        assert normalized_lengths.dtype == torch.int32
        assert (
            normalized_slots.data_ptr()
            == pointers["mla.verify_slots"]
            != slots.data_ptr()
        )
        assert (
            normalized_lengths.data_ptr()
            == pointers["mla.verify_lengths"]
            != lengths.data_ptr()
        )
        assert pointers == {name: value.data_ptr() for name, value in scratch.items()}
        assert scratch["mla.verify_inactive"][4:].all()
        assert (scratch["mla.verify_lengths"][4:] == -77).all()
        assert (scratch["mla.verify_slots"][4 * q :] == -99).all()


@pytest.mark.skipif(
    not torch.cuda.is_available(), reason="metadata graph replay requires CUDA"
)
@pytest.mark.parametrize("q", VERIFY_QUERIES)
def test_verify_mla_metadata_graph_replays_startup_retirement_and_restoration(q):
    """One all-inactive capture must refresh owned metadata on every replay."""
    device = torch.device("cuda", torch.cuda.current_device())
    scratch = _verify_metadata_scratch(device)
    physical = torch.zeros(4, dtype=torch.int32, device=device)
    borrowed_lengths = [MAX_CONTEXT - q, MAX_CONTEXT, MAX_CONTEXT + q, 5]
    lengths = torch.tensor(borrowed_lengths, dtype=torch.int32, device=device)
    slots = torch.arange(4 * q, dtype=torch.int64, device=device) + 2**40
    resources = _resources(
        model=torch.nn.Module(), scratch=scratch, tokens_per_request=q
    )
    inventory = providers._owned_resource_inventory(resources, resources.model)
    assert {row["name"] for row in inventory} == {
        "resources.scratch." + name for name in scratch
    }
    stream = torch.cuda.Stream(device=device)
    stream.wait_stream(torch.cuda.current_stream(device))
    with torch.cuda.stream(stream):
        _verify_mla_metadata(physical, lengths, slots, scratch, q)
    torch.cuda.current_stream(device).wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        normalized_slots, normalized_lengths = _verify_mla_metadata(
            physical, lengths, slots, scratch, q
        )
    torch.cuda.current_stream(device).wait_stream(stream)
    pointers = (normalized_slots.data_ptr(), normalized_lengths.data_ptr())
    for indices, expected_lengths, inactive in (
        ([0, 0, 0, 0], [0, 0, 0, 0], (0, 1, 2, 3)),
        ([7, 2, 3, 4], borrowed_lengths, ()),
        ([7, 2, 3, -1], borrowed_lengths[:3] + [0], (3,)),
        ([7, 2, 3, 4], borrowed_lengths, ()),
    ):
        physical.copy_(torch.tensor(indices, dtype=torch.int32))
        graph.replay()
        torch.cuda.synchronize(device)
        expected_slots = list(range(2**40, 2**40 + 4 * q))
        for request in inactive:
            expected_slots[request * q : (request + 1) * q] = [-1] * q
        assert normalized_slots.cpu().tolist() == expected_slots
        assert normalized_lengths.cpu().tolist() == expected_lengths
        assert physical.cpu().tolist() == indices
        assert lengths.cpu().tolist() == borrowed_lengths
        assert slots.cpu().tolist() == list(range(2**40, 2**40 + 4 * q))
        assert pointers == (normalized_slots.data_ptr(), normalized_lengths.data_ptr())
        assert (
            providers._owned_resource_inventory(resources, resources.model) == inventory
        )


@pytest.mark.parametrize("q", [1, *VERIFY_QUERIES])
def test_executor_preserves_decode_and_never_commits_verify_state(q):
    rows, requests = (8, 8) if q == 1 else (2 * q, 2)
    taps = (0, 1, 7, 92) if q > 1 else ()
    scratch = {
        key: torch.zeros(rows, 7168)
        for key in (
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
    if q > 1:
        scratch.update(_verify_metadata_scratch())
    before = {index: _Consumer(index - 1) for index in range(93)}
    final, mla, kda = _Consumer(92), Mock(), Mock()
    caches, intermediates = (object(), object()), (object(), object())
    attention = {
        index: SimpleNamespace(
            kind="mla" if index == 0 else "kda",
            projection=Mock(),
            core=mla if index == 0 else kda,
            output=Mock(),
            cache=object() if index == 0 else None,
            caches=None if index == 0 else caches,
            intermediates=None if index == 0 else intermediates,
        )
        for index in range(93)
    }
    resources = _resources(
        layers=tuple(SimpleNamespace(layer_idx=index) for index in range(93)),
        attention=attention,
        scratch=scratch,
        before=before,
        final=final,
        feed_forward={index: _FeedForward() for index in range(93)},
        fused_before={1: before[2], 7: before[8], 92: final},
        aux_capture_layers=taps,
        binding_fingerprint="bound",
        tokens_per_request=q,
    )
    track = Mock(
        side_effect=AssertionError("verify must not commit decode state")
        if q > 1
        else None
    )
    view = HybridStepView(
        mla=MlaStepView(
            torch.zeros(requests, 1, dtype=torch.int32),
            torch.arange(requests, dtype=torch.int32) + 13,
            torch.arange(rows, dtype=torch.int64) + 2**40,
            tokens_per_request=q,
        ),
        kda=KdaStepView(
            state_indices=torch.tensor([3] + [0] * (requests - 1), dtype=torch.int32),
            track_decode_state=track,
            verify_scratch_indices=torch.tensor([0, 7], dtype=torch.int32)
            if q > 1
            else None,
        ),
        aux_hidden_states=(
            torch.full((rows, len(taps) * 7168), -1, dtype=torch.bfloat16)
            if q > 1
            else None
        ),
    )
    with patch("sgl_nvidia_mega.kimi_k3.executor._validate_step_inputs"):
        result = KimiK3MegakernelExecutor(resources).run(
            torch.zeros(rows, dtype=torch.int64), torch.zeros(rows, 7168), view
        )
    assert result.data_ptr() == scratch["hidden.out"].data_ptr()
    calls = kda.run_verify.call_args_list if q > 1 else kda.run.call_args_list
    assert len(calls) == 92
    for call in calls:
        assert call.kwargs["caches"] is caches
        assert call.kwargs["slots"] is view.kda.state_indices
        assert "accepted" not in call.kwargs
        if q > 1:
            assert call.kwargs["intermediates"] is intermediates
            assert call.kwargs["scratch_indices"] is view.kda.verify_scratch_indices
    assert mla.run.call_args.kwargs["q"] == q
    if q == 1:
        kda.run_verify.assert_not_called()
        assert track.call_count == 92
        assert mla.run.call_args.kwargs["lengths"] is view.mla.sequence_lengths
        assert mla.run.call_args.kwargs["slots"] is view.mla.write_slots
        return
    track.assert_not_called()
    kda.run.assert_not_called()
    assert mla.run.call_args.kwargs["lengths"].tolist() == [13, 0]
    assert (
        mla.run.call_args.kwargs["slots"].tolist()
        == list(range(2**40, 2**40 + q)) + [-1] * q
    )
    assert (
        mla.run.call_args.kwargs["lengths"].data_ptr()
        == scratch["mla.verify_lengths"].data_ptr()
    )
    assert (
        mla.run.call_args.kwargs["slots"].data_ptr()
        == scratch["mla.verify_slots"].data_ptr()
    )
    assert view.mla.sequence_lengths.tolist() == [13, 14]
    assert view.mla.write_slots.tolist() == list(range(2**40, 2**40 + rows))
    for index, tap in enumerate(taps):
        assert torch.all(
            view.aux_hidden_states[:, index * 7168 : (index + 1) * 7168] == tap
        )
    assert before[1].routes == [("ordinary", (len(taps) * 7168, 1))]
    assert (
        before[2].routes
        == before[8].routes
        == final.routes
        == [("ring", (len(taps) * 7168, 1))]
    )


def test_prepare_configures_aux_before_binding():
    resources = Mock()
    built = SimpleNamespace(resources=resources)
    spec._prepare(built, PreparationView(hybrid=object()))
    assert [call[0] for call in resources.mock_calls] == ["prepare_aux_capture", "bind"]
