# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""CPU contracts for K3.1's dense and paged sparse DSA routes."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest
import torch

from sgl_nvidia_mega.kimi_k31.spec import SPEC, _config_disable_reasons


def _config(**dsa_overrides):
    dsa = dict(
        index_topk=2048,
        index_num_heads=32,
        index_head_dim=128,
        index_rope_dim=0,
        index_k_norm_type="layernorm",
        index_topk_freq=1,
        indexer_use_fp8=False,
        dsa_layers=[4, 8],
    )
    dsa.update(dsa_overrides)
    return SimpleNamespace(
        text_config=SimpleNamespace(
            dsa_config=dsa, linear_attn_config={"full_attn_layers": [4, 8]}
        )
    )


def _cfg(length):
    return SimpleNamespace(
        context_length=length,
        attention_backend="dsa",
        prefill_attention_backend=None,
        decode_attention_backend=None,
        cuda_graph_config=SimpleNamespace(decode=SimpleNamespace(backend="full")),
    )


@pytest.mark.parametrize("length", [1024, 2047, 2048, 2049, 4096, 8192, 9216, 10240, 16384])
def test_accepts_dense_and_sparse_context(length):
    assert _config_disable_reasons(_cfg(length), _config()) == []


def test_long_context_keeps_decode_and_verify_graph_boundary():
    from sglang.srt.megakernels.kimi_k31_decode_binding import K31DecodeGraphVariants
    from sglang.srt.megakernels.kimi_k31_verify_dsa_binding import (
        K31SparseVerifyGraphVariants,
    )

    for policy in (K31DecodeGraphVariants(), K31SparseVerifyGraphVariants(8)):
        for length in (2048, 2049, 8192, 9216, 10240, 16384):
            batch = SimpleNamespace(seq_lens_cpu=torch.tensor([length]))
            assert policy.select(batch) == ("dense" if length <= 2048 else "sparse")


@pytest.mark.parametrize("length", [0, -1])
def test_refuses_nonpositive_context_before_model_build(length):
    reasons = _config_disable_reasons(_cfg(length), _config())
    assert any("positive context_length" in reason for reason in reasons)


@pytest.mark.parametrize(
    "overrides",
    [
        dict(index_topk=1024),
        dict(index_rope_dim=32),
        dict(index_num_heads=64),
        dict(index_topk_freq=2),
        dict(dsa_layers=[4]),
        dict(indexer_use_fp8=True),
    ],
)
def test_refuses_different_dsa_geometry(overrides):
    assert _config_disable_reasons(_cfg(1024), _config(**overrides))


def test_has_its_own_architecture_and_requires_raw_fp8_dsa():
    from sgl_nvidia_mega.plugin import SPECS

    assert (
        "KimiK31ForConditionalGeneration",
        "sgl_nvidia_mega.kimi_k31.spec:SPEC",
    ) in SPECS
    assert "prefill_attention_backend" not in SPEC.requires.server_args
    assert not _config_disable_reasons(_cfg(2048), _config())
    assert SPEC.requires.server_args["dsa_decode_backend"] == frozenset({"trtllm"})
    assert SPEC.requires.server_args["kv_cache_dtype"] == frozenset({"fp8_e4m3"})
    assert SPEC.prepare is not None
    assert SPEC.requires.server_args["enable_linear_replayssm"] == frozenset({False})
    assert SPEC.requires.server_args["enable_linear_replayssm_spec"] == frozenset(
        {False, True}
    )


def test_indexer_writes_key_from_attention_input_to_supplied_slots():
    import sys
    import types

    from sgl_nvidia_mega.kimi_k31.indexer import PreparedIndexKey

    path = "sglang.kernels.ops.kvcache.dsa_fused_store_index_cache"
    fake_store = types.ModuleType(path)
    fake_store.fused_store_index_k_cache = Mock()
    hidden, slots = torch.randn(2, 7168, dtype=torch.bfloat16), torch.tensor([70, 191])
    key = torch.randn(2, 128, dtype=torch.bfloat16)
    normalized = key + 1
    indexer = SimpleNamespace(
        skip_rope=True,
        apply_hadamard=False,
        wk=Mock(return_value=(key, None)),
        k_norm=Mock(return_value=normalized),
    )
    writer = PreparedIndexKey(indexer)
    writer.cache = torch.empty(4, 64 * 132, dtype=torch.uint8)
    with patch.dict(sys.modules, {path: fake_store}):
        writer.run(hidden, slots)
    indexer.wk.assert_called_once_with(hidden)
    indexer.k_norm.assert_called_once_with(key)
    fake_store.fused_store_index_k_cache.assert_called_once_with(
        normalized, writer.cache, slots, 64
    )


def test_typed_step_retains_no_backend_or_runner():
    import dataclasses
    import sys
    import types

    from sglang.srt.megakernels.binding import MegakernelBinding

    full = SimpleNamespace(
        real_page_table=torch.zeros(2, 32, dtype=torch.int32),
        cache_seqlens_int32=torch.tensor([1024, 2047], dtype=torch.int32),
    )
    indices = torch.tensor([1, 2], dtype=torch.int32)
    backend = SimpleNamespace(
        token_to_kv_pool=object(),
        full_attn_backend=SimpleNamespace(forward_metadata=full),
        linear_attn_backend=SimpleNamespace(
            forward_metadata=SimpleNamespace(mamba_cache_indices=indices)
        ),
    )
    context = types.ModuleType("sglang.srt.model_executor.forward_context")
    context.get_attn_backend = lambda: backend
    slots = torch.tensor([70, 191], dtype=torch.int64)
    with patch.dict(sys.modules, {context.__name__: context}):
        view = MegakernelBinding(model=torch.nn.Identity()).forward_view(
            SimpleNamespace(out_cache_loc=slots)
        )

    def leaves(value):
        if dataclasses.is_dataclass(value):
            for field in dataclasses.fields(value):
                yield from leaves(getattr(value, field.name))
        else:
            yield value

    assert all(isinstance(value, (int, float, torch.Tensor)) for value in leaves(view))
    assert view.hybrid.mla.write_slots is slots
    assert view.hybrid.kda.state_indices is indices


def test_pool_preparation_hands_out_only_metadata_and_buffer_references():
    import dataclasses

    from sglang.srt.megakernels.binding import MegakernelBinding

    kv, index_k = (
        torch.zeros(192, 1, 576, dtype=torch.uint8),
        torch.zeros(3, 64 * 132, dtype=torch.uint8),
    )
    conv, temporal = (
        torch.zeros(4, 3, 4608, dtype=torch.bfloat16),
        torch.zeros(4, 12, 128, 128),
    )
    state = SimpleNamespace(conv=[conv], temporal=temporal)
    full = SimpleNamespace(
        kv_buffer=[kv],
        kv_cache_dim=576,
        size=128,
        dtype=torch.float8_e4m3fn,
        store_dtype=torch.uint8,
        post_capture_active=False,
        index_page_size=64,
        index_k_with_scale_buffer=[index_k],
        get_index_k_with_scale_buffer=lambda layer: index_k,
    )
    pool = SimpleNamespace(
        use_mla=True,
        use_dsa=True,
        page_size=64,
        mamba_pool=object(),
        full_kv_pool=full,
        full_attention_layer_id_mapping={3: 0},
    )
    req = SimpleNamespace(
        mamba_map={0: 0},
        mamba_pool=SimpleNamespace(size=3),
        mamba2_layer_cache=lambda layer: state,
    )
    binding = MegakernelBinding(model=torch.nn.Identity())
    view = binding.preparation_view(pool, req)

    def leaves(value):
        if dataclasses.is_dataclass(value):
            for field in dataclasses.fields(value):
                yield from leaves(getattr(value, field.name))
        elif isinstance(value, dict):
            for key, item in value.items():
                yield from leaves(key)
                yield from leaves(item)
        elif isinstance(value, tuple):
            for item in value:
                yield from leaves(item)
        else:
            yield value

    assert all(isinstance(value, (int, float, torch.Tensor)) for value in leaves(view))
    assert view.hybrid.mla.cache(3).data_ptr() == kv.data_ptr()
    assert view.hybrid.indexer.caches[3] is index_k
    assert view.hybrid.kda.state(0).conv[0] is conv
    assert view.hybrid.mla.logical_dtype == torch.float8_e4m3fn
    assert view.hybrid.mla.storage_dtype == torch.uint8
    assert binding._pool_identity(pool, req) == binding._hybrid_identity
    full.index_k_with_scale_buffer = [index_k.clone()]
    with pytest.raises(RuntimeError, match="buffers changed"):
        binding.assert_live()


@pytest.mark.parametrize(
    "base,prefill,decode,ok",
    [
        ("dsa", None, None, True),
        (None, "dsa", "dsa", True),
        ("trtllm_mla", "dsa", "dsa", True),
        ("dsa", "trtllm_mla", None, False),
        ("dsa", None, "trtllm_mla", False),
        ("trtllm_mla", None, None, False),
    ],
)
def test_checks_effective_attention_backends(base, prefill, decode, ok):
    cfg = _cfg(2048)
    cfg.attention_backend = base
    cfg.prefill_attention_backend = prefill
    cfg.decode_attention_backend = decode
    assert (not _config_disable_reasons(cfg, _config())) is ok


@pytest.mark.parametrize("rows", (1, 2, 4, 8, 16, 32, 64))
def test_sparse_decode_keeps_each_requests_page_and_state_rows(rows):
    from sglang.srt.megakernels.kimi_k31_decode_binding import K31DecodeHostBinding
    from sglang.srt.model_executor.forward_batch_info import ForwardMode

    table = torch.arange(rows * 9, dtype=torch.int32).reshape(rows, 9)
    lengths = torch.arange(8192, 8192 + rows, dtype=torch.int32)
    slots = torch.arange(200, 200 + rows, dtype=torch.int64).flip(0)
    states = torch.arange(rows, dtype=torch.int32).flip(0)
    ids = torch.zeros(rows, dtype=torch.int64)
    observed = []
    completed = []

    def decode(input_ids, embeddings, view):
        view.validate()
        observed.append(view)
        return embeddings

    model = SimpleNamespace(language_model=SimpleNamespace(
        model=SimpleNamespace(embed_tokens=lambda _: torch.zeros(rows, 7168)),
        lm_head=None,
        logits_processor=lambda ids, hidden, *args: hidden,
    ))
    host = K31DecodeHostBinding.__new__(K31DecodeHostBinding)
    host.model = model
    host.pools = SimpleNamespace(finish_step=lambda fb: completed.append(fb))
    built = SimpleNamespace(widths=(1, 2, 4, 8, 16, 32, 64), run_decode=decode)
    fb = SimpleNamespace(
        forward_mode=ForwardMode.DECODE, batch_size=rows,
        seq_lens_cpu=lengths, out_cache_loc=slots,
    )
    backend = SimpleNamespace(
        full_attn_backend=SimpleNamespace(forward_metadata=SimpleNamespace(
            real_page_table=table, cache_seqlens_int32=lengths,
            paged_mqa_schedule_metadata=torch.zeros(8, dtype=torch.int32),
            topk_v2_plan=torch.zeros(8, dtype=torch.int32),
        )),
        linear_attn_backend=SimpleNamespace(forward_metadata=SimpleNamespace(
            mamba_cache_indices=states,
        )),
    )
    step = host.plan(built, ids, fb)
    assert step.rows == rows
    with patch("sglang.srt.model_executor.forward_context.get_attn_backend", return_value=backend):
        result = host.forward(built, step, ids, ids, fb)
    assert result.shape == (rows, 7168)
    view, = observed
    assert torch.equal(view.mla.block_table, table)
    assert torch.equal(view.mla.sequence_lengths, lengths)
    assert torch.equal(view.mla.write_slots, slots)
    assert torch.equal(view.kda.state_indices, states)
    assert completed == [fb]
    # A live B3 can replay the padded B4 graph; eager B3 is not misplanned.
    fb.batch_size = 3
    assert host.can_run_graph(built, fb)
    assert host.plan(built, torch.zeros(3, dtype=torch.int64), fb) is None


def test_sparse_decode_rejects_missing_request_metadata():
    from sgl_nvidia_mega.kimi_k31.dsa_binding import (
        IndexDecodeStep, KdaDecodeStep, MlaDecodeStep, SparseDecodeStep,
    )

    step = SparseDecodeStep(
        MlaDecodeStep(torch.zeros(4, 32), torch.ones(4), torch.zeros(3)),
        KdaDecodeStep(torch.zeros(4)), IndexDecodeStep(torch.zeros(1), torch.zeros(1)), 1,
    )
    with pytest.raises(ValueError, match="one entry per request"):
        step.validate()
