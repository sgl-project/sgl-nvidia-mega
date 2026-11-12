# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""CPU-only request/query contracts; device math is checked by the GPU gate."""
from dataclasses import replace
from types import SimpleNamespace as NS
from unittest.mock import Mock, patch

import pytest
import torch

from sgl_nvidia_mega.kimi_k31.verify_contract import request_capacity, verify_keys
from sgl_nvidia_mega.kimi_k31.verify_binding import KdaVerifyStep, VerifyStepView


def view(b=4, q=8):
    slots = torch.tensor([0, 7, 3, -1][:b], dtype=torch.int32)
    scratch = torch.tensor([6, 1, 4, -1][:b], dtype=torch.int32)
    kda = KdaVerifyStep(slots, scratch, (slots >= 0).int(), b, q, b*q)
    return VerifyStepView(
        torch.arange(b*160, dtype=torch.int32).reshape(b, 160),
        torch.tensor([64, 8192, 2051, 0][:b], dtype=torch.int32),
        torch.arange(b*q, dtype=torch.int64), kda,
        torch.empty(b*q, 6*7168, dtype=torch.bfloat16), b, q, b*q,
    )


def test_query_width_is_not_flattened_into_request_count():
    assert verify_keys(8, (1, 2, 4)) == ((1, 8, 8), (2, 8, 16), (4, 8, 32))
    assert verify_keys(16, (1,)) == ((1, 16, 16),)
    assert [request_capacity(b, (1, 2, 4)) for b in range(6)] == [None, 1, 2, 4, 4, None]
    for q, batches in [(8, (1,17)), (16, (1,2)), (1, (1,)), (8, (2,4)), (8, (1,4,2))]:
        with pytest.raises(ValueError):
            verify_keys(q, batches)


def test_breq8_and_breq16_are_q8_keys_and_breq17_is_not():
    # MKB-0: Breq 8 = 64 rows; Breq 5-7 pad to the 8 graph (request_capacity), 9+ has no capacity without a wider graph.
    # Wide tier (port of ad05010): Breq 16 = 128 rows is a key; 17+ never is. Expected keys hand-derived (b, 8, 8 b).
    assert verify_keys(8, (1, 2, 4, 8)) == ((1, 8, 8), (2, 8, 16), (4, 8, 32), (8, 8, 64))
    assert verify_keys(8, (1, 2, 4, 8, 16)) == ((1, 8, 8), (2, 8, 16), (4, 8, 32), (8, 8, 64), (16, 8, 128))
    assert verify_keys(8, (1, 16)) == ((1, 8, 8), (16, 8, 128))
    assert [request_capacity(b, (1, 2, 4, 8)) for b in range(10)] == [None, 1, 2, 4, 4, 8, 8, 8, 8, None]
    assert [request_capacity(b, (1, 2, 4, 8, 16)) for b in range(18)] == [None, 1, 2, 4, 4] + [8] * 4 + [16] * 8 + [None]
    for q, batches in [(8, (1, 2, 4, 8, 16, 32)), (8, (1, 17)), (8, (1, 32)), (8, (1, 8, 4)), (16, (1, 8)), (16, (1, 16))]:
        with pytest.raises(ValueError):
            verify_keys(q, batches)


def test_served_request_batches_default_and_knob():
    # Default = the keys the host built (kimi_k31_verify_binding.py passed (1, 2, 4));
    # MK batch boots ask for Breq 8; anything the contract refuses fails at build time.
    from sgl_nvidia_mega.kimi_k31.verify_contract import served_request_batches

    assert served_request_batches({}) == (1, 2, 4)
    assert served_request_batches({"K31MK_VERIFY_BATCHES": "1,2,4,8"}) == (1, 2, 4, 8)
    assert served_request_batches({"K31MK_VERIFY_BATCHES": "1"}) == (1,)
    assert served_request_batches({"K31MK_VERIFY_BATCHES": "1,2,4,8,16"}) == (1, 2, 4, 8, 16)
    for bad in ("1,2,4,8,16,32", "1,17", "2,4", "1,4,2", "1,9,9", "1,,2", "eight", ""):
        with pytest.raises(ValueError):
            served_request_batches({"K31MK_VERIFY_BATCHES": bad})


def test_l3_own_keys_for_breq_3_5_6_7():
    # L3: Breq 3 / 5 / 6 / 7 are Q8 keys of their own (24 / 40 / 48 / 56 rows), so a boot that names them
    # captures its own graph instead of padding to 4 / 8. Expected keys hand-derived (b, 8, 8 b); 9+ still has no key.
    from sgl_nvidia_mega.kimi_k31.verify_contract import served_request_batches

    every = tuple(range(1, 9))
    assert verify_keys(8, every) == tuple((b, 8, 8 * b) for b in every)
    assert [w for _, _, w in verify_keys(8, (1, 3, 5, 6, 7))] == [8, 24, 40, 48, 56]
    assert [request_capacity(b, every) for b in range(10)] == [None, 1, 2, 3, 4, 5, 6, 7, 8, None]
    assert [request_capacity(b, (1, 2, 3, 4, 8)) for b in range(10)] == [None, 1, 2, 3, 4, 8, 8, 8, 8, None]
    assert served_request_batches({"K31MK_VERIFY_BATCHES": "1,2,3,4,5,6,7,8"}) == every
    assert served_request_batches({}) == (1, 2, 4)  # the c=1 default is unchanged
    for q, batches in [(8, tuple(range(1, 18))), (16, (1, 3)), (8, (1, 3, 3))]:
        with pytest.raises(ValueError):
            verify_keys(q, batches)


def test_wide_tier_exact_keys_for_breq_9_to_16():
    # Wide decode tier M0 gate: a boot that names 1..16 gets an exact key (and graph) for every Breq 9-15 (72-120 rows),
    # never a silent pad to the 16 graph. Expected values hand-derived: key (b, 8, 8 b); capacity = b itself.
    # Mutants this must catch: the contract admitting only (1..8, 16) (9-15 would pad), or a capacity that rounds up.
    from sgl_nvidia_mega.kimi_k31.verify_contract import served_request_batches

    every = tuple(range(1, 17))
    assert verify_keys(8, every) == tuple((b, 8, 8 * b) for b in every)
    assert [w for _, _, w in verify_keys(8, (1, 9, 10, 11, 12, 13, 14, 15, 16))] == [8, 72, 80, 88, 96, 104, 112, 120, 128]
    assert [request_capacity(b, every) for b in range(18)] == [None] + list(every) + [None]
    assert served_request_batches({"K31MK_VERIFY_BATCHES": ",".join(map(str, every))}) == every
    assert served_request_batches({}) == (1, 2, 4)  # the c=1 default is unchanged


def test_sparse_step_validates_up_to_breq16_and_refuses_breq17():
    # The sparse DSA step re-checks its key against the contract (dsa/verify.py SparseVerifyStep.validate).
    from sgl_nvidia_mega.kimi_k31.dsa import verify as module

    def step(b, q=8):
        slots = torch.arange(b, dtype=torch.int32)
        kda = KdaVerifyStep(slots, slots.clone(), torch.ones(b, dtype=torch.int32), b, q, b*q)
        lengths = torch.full((b,), 8192, dtype=torch.int32)
        return module.SparseVerifyStep(
            torch.arange(b*160, dtype=torch.int32).reshape(b, 160), lengths,
            torch.arange(b*q, dtype=torch.int64), kda, torch.empty(b*q, 6*7168, dtype=torch.bfloat16), b, q, b*q,
            causal_lengths=torch.arange(b*q, dtype=torch.int32),
            score_contexts=lengths[:, None].expand(b, q).contiguous(),
            score_schedule=torch.zeros(8, dtype=torch.int32),
            topk_plan=torch.zeros(b*q + 1, 4, dtype=torch.int32), sparse=1,
        )

    step(8).validate(auxiliary_width=6*7168)
    for b in (3, 5, 6, 7):  # L3: own keys
        step(b).validate(auxiliary_width=6*7168)
    for b in range(9, 17):  # wide tier: own keys 72-128 rows
        step(b).validate(auxiliary_width=6*7168)
    for b in (17, 32):
        with pytest.raises(ValueError, match='supported request/query key'):
            step(b).validate(auxiliary_width=6*7168)


@pytest.mark.parametrize('b', [1, 2, 4])
def test_fixed_axes_slot_zero_and_padding(b):
    v = view(b)
    v.validate(auxiliary_width=6*7168)
    assert v.kda.state_slots[0] == 0  # Slot zero is committed state, not padding.
    assert v.aux_hidden.shape[0] == b*8
    assert not torch.cuda.is_initialized()


def test_equal_flat_rows_do_not_hide_mismatched_kda_axes():
    v = view(2)
    wrong = replace(v.kda, batch_requests=1, query_width=16,
                    state_slots=v.kda.state_slots[:1], scratch_rows=v.kda.scratch_rows[:1],
                    valid_requests=v.kda.valid_requests[:1])
    with pytest.raises(ValueError, match='axes must match'):
        replace(v, kda=wrong).validate()
    with pytest.raises(ValueError, match=r'int32\[Breq\]'):
        replace(v, kda=replace(v.kda, scratch_rows=torch.zeros(16, dtype=torch.int32))).validate()
    with pytest.raises(ValueError, match='int32'):
        replace(v, sequence_lengths=v.sequence_lengths.long()).validate()


def test_sparse_native_score_keeps_distinct_request_tables():
    from sgl_nvidia_mega.kimi_k31.dsa import verify as module
    from sglang.kernels.ops.attention.dsa import paged_mqa_logits
    from sglang.kernels.ops.attention.dsv4 import topk as stock_topk

    v = view(4)
    sparse = module.SparseVerifyStep(
        **v.__dict__, causal_lengths=torch.arange(32, dtype=torch.int32),
        score_contexts=v.sequence_lengths[:,None].expand(4,8).contiguous(),
        score_schedule=torch.zeros(8, dtype=torch.int32),
        topk_plan=torch.zeros(33, 4, dtype=torch.int32), sparse=1,
    )
    sparse.validate(auxiliary_width=6*7168)
    query=torch.empty(32,32,128, dtype=torch.float8_e4m3fn)
    gate=torch.empty(32,32)
    cache=torch.empty(4,64*132,dtype=torch.uint8)
    selector=module.VerifySelection('cpu',32)
    scores=torch.empty(32,10240)
    native=Mock(return_value=scores)
    topk=Mock()
    with patch.dict('sys.modules', {'deep_gemm':NS(fp8_paged_mqa_logits=native)}), \
         patch.object(paged_mqa_logits,'_restore_row_stride',side_effect=lambda x:x), \
         patch.object(stock_topk,'topk_transform_paged_v2',topk):
        selector.run(query,gate,cache,v.block_table,sparse)
    args=native.call_args.args
    assert args[0].shape == (4,8,32,128)
    assert args[3] is sparse.score_contexts and args[4] is v.block_table
    # the per-row table: row r*8+j reads request r's pages, not one request broadcast
    assert torch.equal(topk.call_args.args[2], v.block_table.repeat_interleave(8, 0))
    assert topk.call_args.kwargs == {'deterministic':False}
    with pytest.raises(ValueError,match=r'\[Breq,Q\]'):
        replace(sparse,score_contexts=sparse.score_contexts[:1]).validate()


def test_profile_scratch_is_fixed_and_request_capacity_selects_it():
    from sgl_nvidia_mega.kimi_k31.dsa import verify as module
    base=NS(prepare_cache=Mock())
    def profile(_base,_weights,rows,_context,_tsi=None):  # PreparedSparseVerifyMla(base, weights, rows, context, tsi)
        return NS(selection=NS(rows=rows),prepare_cache=Mock(),run=Mock(return_value=rows),step=None)
    with patch.object(module,'PreparedSparseVerifyMla',side_effect=profile):
        core=module.MultiSparseVerifyMla(base,None,(8,16,32),10240)
    identities={r:id(p) for r,p in core.profiles.items()}
    core.prepare_cache(None)
    base.prepare_cache.assert_called_once()
    for b in [1,4,2,4,1]:
        v=view(b);core.step=v
        assert core.run(torch.empty(b*8,3776)) == b*8
        assert core.profiles[b*8].step is v
    assert identities == {r:id(p) for r,p in core.profiles.items()}


def test_host_graph_capacity_uses_padding_and_rejects_uncovered_batch():
    K31SparseVerifyHostBinding = pytest.importorskip("sglang.srt.megakernels.kimi_k31_verify_dsa_binding", reason="needs the Kimi K3.1 SGLang host changes").K31SparseVerifyHostBinding
    host=K31SparseVerifyHostBinding(None)
    built=NS(request_batches=(1,2,4),supports=lambda b,q,r:(b,q,r) in verify_keys(8,(1,2,4)))
    mode=NS(is_target_verify=lambda:True)
    for b in range(1,6):
        batch=NS(batch_size=b,forward_mode=mode,spec_info=NS(ragged_verify_layout=None))
        assert host.can_run_graph(built,batch) == (b<=4)
        plan=host.plan(built,torch.empty(b*8,dtype=torch.int64),batch)
        assert (plan is not None) == (b in (1,2,4))
    batch=NS(batch_size=4,forward_mode=mode,spec_info=NS(ragged_verify_layout=object()))
    assert host.plan(built,torch.empty(32,dtype=torch.int64),batch) is None


@pytest.mark.parametrize('b', [2, 4])
def test_host_unrepeats_query_page_table_before_tensor_entry(b):
    K31SparseVerifyHostBinding = pytest.importorskip("sglang.srt.megakernels.kimi_k31_verify_dsa_binding", reason="needs the Kimi K3.1 SGLang host changes").K31SparseVerifyHostBinding
    from sglang.srt.model_executor import forward_context
    from sglang.srt.layers.attention.linear import utils
    from sglang.srt.model_executor.runner_utils import capture_mode

    q, r = 8, b * 8
    table = torch.arange(b * 160, dtype=torch.int32).view(b, 160)
    full = NS(real_page_table=table.repeat_interleave(q, 0),
              cache_seqlens_int32=torch.full((b,), 8192, dtype=torch.int32),
              dsa_seqlens_expanded=torch.arange(r, dtype=torch.int32),
              paged_mqa_schedule_metadata=torch.empty(8, dtype=torch.int32),
              topk_v2_plan=torch.empty(r + 1, 4, dtype=torch.int32))
    linear = NS(forward_metadata=NS(mamba_cache_indices=torch.arange(b, dtype=torch.int32)),
                verify_intermediate_state_indices=torch.arange(b, dtype=torch.int32),
                req_to_token_pool=NS(size=16))
    backend = NS(full_attn_backend=NS(forward_metadata=full), linear_attn_backend=linear)
    lm = NS(model=NS(embed_tokens=lambda ids: torch.empty(r, 7168, dtype=torch.bfloat16)),
            lm_head=None, logits_processor=lambda *args: args[-1])
    host = K31SparseVerifyHostBinding(NS(language_model=lm))
    host.aux = torch.empty(r, 6 * 7168, dtype=torch.bfloat16)
    host.graph_policy = NS(capture_labels=('dense', 'sparse'))
    host.request_page_table = torch.empty(b * 160, dtype=torch.int32)
    built = NS(run_verify=Mock(return_value=(torch.empty(r, 7168), host.aux)))
    batch = NS(batch_size=b, req_pool_indices=torch.arange(b),
               out_cache_loc=torch.arange(r, dtype=torch.int64))
    with patch.object(forward_context, 'get_attn_backend', return_value=backend), \
         patch.object(utils, 'select_verify_intermediate_state_indices', return_value=torch.arange(b, dtype=torch.int32)), \
         patch.object(capture_mode, 'get_capture_attention_variant', return_value='sparse'):
        host.forward(built, NS(rows=r), torch.arange(r), torch.arange(r), batch)
    actual = built.run_verify.call_args.args[1].block_table
    assert torch.equal(actual, table)
    assert actual.is_contiguous()
    assert actual.data_ptr() == host.request_page_table.data_ptr()


def test_batch_scratch_rows_cover_the_largest_batch():
    # CAP-128 #1: the row-wise scratch every batch of a boot shares must hold the boot's largest batch (Breq 16 at Q8 =
    # 128 rows); a boot of today's sizes keeps the 64-row scratch it always had. Expected shapes are the literal shapes
    # build_resources allocated before this change, with the row count replaced.
    from sgl_nvidia_mega.kimi_k31.k3.providers import _batch_scratch

    heads, mla_pitch, kda_width = 6, 3, 5  # small widths: rows are under test, widths only checked for pass-through

    def shapes(rows):
        return {
            "mla.query": (rows, heads, 576),
            "mla.projected": (rows, mla_pitch),
            "kda.projected": (rows, kda_width),
            "attention.local": (rows, heads * 128),
            "hidden.0": (rows, 7168),
            "hidden.1": (rows, 7168),
            "hidden.out": (rows, 7168),
            "prefix": (rows, 7168),
            "blocks": (rows, 8, 7168),
        }

    cases = [((8,), 64), ((8, 16, 32, 64), 64), (tuple(range(1, 9)) + (16, 32, 64), 64), ((8, 16, 32, 64, 128), 128)]
    for batches, rows in cases:
        got = _batch_scratch("cpu", heads, mla_pitch, kda_width, batches)
        assert {k: tuple(v.shape) for k, v in got.items()} == shapes(rows), batches
        assert got["mla.query"].dtype == torch.uint8 and not got["mla.query"].any()
        assert all(v.dtype == torch.bfloat16 for k, v in got.items() if k != "mla.query")
