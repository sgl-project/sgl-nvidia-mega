# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""The attention step on CPU: the prepare path SGLang would pick, the chain's gates
and engagement, the step head's FlashMLA schedules, and hop A's wo_b publish at
batch 1 and 2."""

import importlib
from types import SimpleNamespace
from unittest import mock

import pytest
import torch

backend = pytest.importorskip("sglang.srt.layers.attention.deepseek_v4_backend")

from sglang.srt.layers.quantization.fp8_utils import (  # noqa: E402
    Mxfp8DenseGemmBackend,
)
from sglang.srt.models import deepseek_v4_mhc as mhc  # noqa: E402

from sgl_nvidia_mega.deepseek_v41 import attention  # noqa: E402
from sgl_nvidia_mega.deepseek_v41 import plan  # noqa: E402
from sgl_nvidia_mega.deepseek_v41 import runtime  # noqa: E402
from sgl_nvidia_mega.deepseek_v41.attention import chain  # noqa: E402
from sgl_nvidia_mega.deepseek_v41.attention import sched_ahead  # noqa: E402
from sgl_nvidia_mega.deepseek_v41.attention import ops, prepare  # noqa: E402
from sgl_nvidia_mega.deepseek_v41.attention.step import AttentionStep  # noqa: E402
from sgl_nvidia_mega.deepseek_v41.sm103 import mxfp8_ll, wo_b_publish  # noqa: E402

# ------------------------------------------------------------------------------------
# The stock attention step on CPU: the prepare path it picks per layer and step, and
# wo_b's hop-A publish.


def _mode(*, decode=False, verify=False):
    return SimpleNamespace(
        forward_mode=SimpleNamespace(
            is_decode=lambda: decode, is_target_verify=lambda: verify
        )
    )


def _attn(
    ratio, *, streams=True, wq_b_backend=Mxfp8DenseGemmBackend.FLASHINFER_CUTEDSL
):
    return SimpleNamespace(
        compress_ratio=ratio,
        alt_streams=[object()] * 3 if streams else None,
        _multi_stream_bs_limit=128,
        wq_b=SimpleNamespace(
            quant_method=SimpleNamespace(mxfp8_dense_backend=wq_b_backend)
        ),
    )


@pytest.mark.parametrize(
    "ratio, streams, capture, mode, backend_, want",
    [
        (2, True, True, _mode(verify=True), None, prepare.low_ratio),
        (1, True, False, _mode(decode=True), None, prepare.low_ratio),
        (1, True, False, _mode(verify=True), "other", prepare.eager),
        (0, True, True, _mode(verify=True), None, prepare.multi_stream),
        (0, True, False, _mode(verify=True), None, prepare.eager),
        (2, False, True, _mode(verify=True), None, prepare.eager),
    ],
)
def test_prepare_path_is_sglangs_choice(ratio, streams, capture, mode, backend_, want):
    """The step must take the prepare SGLang's MQALayer takes, or it stops being bitwise."""
    kwargs = {} if backend_ is None else {"wq_b_backend": backend_}
    step = AttentionStep.__new__(AttentionStep)
    step.attn, step.low_ratio = _attn(ratio, streams=streams, **kwargs), ratio in (1, 2)
    with mock.patch(
        "sglang.srt.model_executor.runner.get_is_capture_mode", return_value=capture
    ):
        assert step._prepare_path(6, mode) is want


def test_wo_b_publishes_at_hop_a_widths_and_falls_back_otherwise():
    """A published partial skips SGLang's wo_b; a GEMM that declines leaves it to SGLang."""
    o = torch.randn(6, 2, 4096)
    stock = mock.Mock(return_value=(torch.ones(6, 8), None))
    attn = SimpleNamespace(wo_b=stock, attn_tp_size=4)
    published = torch.zeros(6, 8)
    publish = mock.Mock(return_value=published)
    with mock.patch(
        "sglang.srt.layers.moe.utils.should_skip_mlp_all_reduce", return_value=False
    ):
        out = ops.wo_b(attn, o, publish)
        assert isinstance(out, mhc.AttnOutput) and out[0] is published
        stock.assert_not_called()
        assert publish.call_args.args[1].shape == (6, 2 * 4096)
        publish.return_value = None
        out = ops.wo_b(attn, o, publish)
    stock.assert_called_once()
    assert stock.call_args.kwargs == {"skip_all_reduce": True}


# ------------------------------------------------------------------------------------
# The attention chain on CPU: its swap-AB gates, the query RoPE applied exactly once,
# and its slot reported as built.

# the embedding and Engram builds, both built
OTHERS = {"embedding": object(), "engram": SimpleNamespace(prefetch=object())}


def _pool(*, window=None, width=584):
    return SimpleNamespace(
        request_window=window, get_swa_key_bytes_per_token=lambda: width
    )


def test_swapab_takes_only_covered_steps():
    """Swap-AB only at 1-24 rows of decode / verify on the V4-layout window cache."""
    meta = backend.DSV4AttnMetadata.__new__(backend.DSV4AttnMetadata)

    def takes(rows, mode=None, pool=None):
        # SGLang's ForwardView of the step
        view = SimpleNamespace(
            metadata=SimpleNamespace(core_attn_metadata=meta), kv_pool=pool or _pool()
        )
        return chain.swapab_takes(rows, mode or _mode(verify=True), view)

    assert takes(1) and takes(24) and takes(6, _mode(decode=True))
    assert not takes(25)
    assert not takes(6, _mode())
    assert not takes(6, pool=_pool(window=object()))
    assert not takes(6, pool=_pool(width=528))


def test_chain_ropes_the_query_once():
    """With the RoPE in attention, wq_b's BF16 output must reach attention unroped."""
    q = torch.randn(6, 1280, dtype=torch.bfloat16)
    wq_b_out = torch.randn(6, 16 * 512, dtype=torch.bfloat16)
    attn = SimpleNamespace(
        wq_b=mock.Mock(return_value=(wq_b_out, None)), n_local_heads=16, head_dim=512
    )
    ops = chain.ChainOps.__new__(chain.ChainOps)
    ops.rope_in_attention, ops.wq_b_ll = True, False
    with mock.patch("sglang.kernels.ops.attention.dsv4.fused_q_norm_rope") as rope:
        out = ops.compute_q_b(attn, q, torch.arange(6), None)
    rope.assert_not_called()
    assert torch.equal(out, wq_b_out.view(6, 16, 512))


def test_every_shipped_slot_reports_its_build():
    """The build checks each requested slot against this table; a slot missing
    from it stops the server at build."""
    slots = {6: SimpleNamespace(hop_a=True, fast_gate=True)}
    moe = SimpleNamespace(slots={0: SimpleNamespace(gate=True, shared=object())})
    layers = {0: SimpleNamespace(chain=None), 1: SimpleNamespace(chain=object())}
    facts = SimpleNamespace(routed=True)
    built = runtime.built_slots(
        slots, {**OTHERS, "moe": moe, "attention": layers}, facts
    )
    # the slots this table reports; the boundary build reports two_shot and the
    # hops, the DSpark step its own slots
    target = {"hop_a", "moe", "gate", "attn", "embed", "engram", "shared_ll"}
    assert set(built) == target and all(built.values())
    stock = {0: SimpleNamespace(chain=None)}
    assert not runtime.built_slots(
        slots, {**OTHERS, "moe": moe, "attention": stock}, facts
    )["attn"]


def _x_quant(rows):
    from sglang.srt.layers.quantization.mxfp8_input import Mxfp8SwizzledInput

    return Mxfp8SwizzledInput(
        torch.zeros(rows, 5120, dtype=torch.float8_e4m3fn),
        torch.zeros(20480, dtype=torch.uint8),
    )


def test_wqkv_a_takes_the_low_latency_gemm_only_on_mxfp8_rows_up_to_8():
    """The low-latency kernel reads FlashInfer's plain 128x4 scales, which the boundary
    writes for its MXFP8 input; BF16 input or 9+ rows (row-tile scales) stay SGLang's."""
    attn = SimpleNamespace(wqkv_a=mock.Mock(return_value=("sglang", None)), layer_id=0)
    attn.wqkv_a.weight, attn.wqkv_a.weight_scale_inv_swizzled = "w", "w_sf"
    ops = chain.ChainOps.__new__(chain.ChainOps)
    ops.qkv_a_ll, ops.qkv_a_bf16 = True, False
    with mock.patch.object(chain.mxfp8_ll, "linear", return_value="ll") as ll:
        assert ops.qkv_a(attn, _x_quant(8)) == "ll"
        assert ll.call_args.args[2:] == ("w", "w_sf")
        assert ops.qkv_a(attn, _x_quant(9)) == "sglang"
        assert ops.qkv_a(attn, torch.zeros(4, 5120, dtype=torch.bfloat16)) == "sglang"
        ops.qkv_a_ll = False
        assert ops.qkv_a(attn, _x_quant(4)) == "sglang"
    assert ll.call_count == 1


def test_attn_qkv_a_reports_built_only_with_a_width_it_serves():
    """A requested slot built at no width stops the server; one that cannot run at any
    covered width must report so instead of measuring as on."""
    on = {0: SimpleNamespace(chain=(SimpleNamespace(qkv_a_ll=True),) * 2)}
    off = {0: SimpleNamespace(chain=(SimpleNamespace(qkv_a_ll=False),) * 2)}
    assert runtime.attn_qkv_a_built(on, (6, 12))
    assert not runtime.attn_qkv_a_built(on, (12, 24))
    assert not runtime.attn_qkv_a_built(off, (6,))
    assert not runtime.attn_qkv_a_built({0: SimpleNamespace(chain=None)}, (6,))


def test_qkv_a_engagement_counts_the_layers_that_fell_back():
    """The startup evidence must count layers whose boundary gave no MXFP8 input, not
    just that the slot built: logged once per row count, when the first layer repeats."""
    tracker = chain._Engagement()
    with mock.patch.object(chain.logger, "info") as info:
        for step in range(2):
            for layer in range(3):
                tracker.note(layer, 6, ll=layer != 0)
            if step == 0:
                info.assert_not_called()
    info.assert_called_once()
    assert info.call_args.args[1:] == (6, 2, 3, [0])


def test_attn_qkv_a_bf16_quantizes_bf16_rows_for_the_low_latency_gemm():
    """Layers whose boundary hands BF16 rows (no MXFP8) take the low-latency GEMM only
    with slot attn_qkv_a_bf16, quantized by FlashInfer's MXFP8 quantize; 9+ rows and
    non-BF16 input keep SGLang's call."""
    attn = SimpleNamespace(wqkv_a=mock.Mock(return_value=("sglang", None)), layer_id=0)
    attn.wqkv_a.weight, attn.wqkv_a.weight_scale_inv_swizzled = "w", "w_sf"
    ops = chain.ChainOps.__new__(chain.ChainOps)
    ops.qkv_a_ll, ops.qkv_a_bf16 = True, True
    bf16 = lambda n: torch.zeros(n, 5120, dtype=torch.bfloat16)  # noqa: E731
    with (
        mock.patch.object(
            chain.mxfp8_ll, "quantize", return_value=("q", torch.zeros(4))
        ) as quantize,
        mock.patch.object(chain.mxfp8_ll, "linear", return_value="ll") as ll,
    ):
        assert ops.qkv_a(attn, bf16(6)) == "ll"
        assert ll.call_args.args[0] == "q" and ll.call_args.args[2:] == ("w", "w_sf")
        assert ops.qkv_a(attn, bf16(9)) == "sglang"
        assert ops.qkv_a(attn, torch.zeros(6, 5120)) == "sglang"
    assert quantize.call_count == 1


def _chain_ops(index_q_mxfp8_in):
    ops_ = chain.ChainOps.__new__(chain.ChainOps)
    ops_.index_q_mxfp8_in = index_q_mxfp8_in
    return ops_


@pytest.mark.parametrize("rows", [1, 6, 8, 9, 48])
@pytest.mark.parametrize("on", [True, False])
def test_the_indexer_wq_b_reads_the_chain_quant_only_at_1_to_8_rows(rows, on):
    """Slot index_q_mxfp8_in: SGLang's MXFP8 linear takes the chain's 128x4 q_lora
    rows as they are; past 8 rows the chain writes row-tile scales it does not read."""
    from sglang.srt.layers.quantization.mxfp8_input import Mxfp8SwizzledInput

    q_lora = torch.zeros(rows, 1280, dtype=torch.bfloat16)
    rows_ = chain.Mxfp8Rows(torch.zeros(rows, 1280, dtype=torch.uint8), torch.ones(8))
    got = _chain_ops(on).index_q_input(None, q_lora, rows_)
    if on and rows <= 8:
        assert isinstance(got, Mxfp8SwizzledInput)
        assert got.data is rows_.quant and got.scales is rows_.scale
    else:
        assert got is q_lora
    # SGLang's own q_lora quant (not the chain's rows) is never handed over
    assert _chain_ops(on).index_q_input(None, q_lora, q_lora) is q_lora


def test_index_q_mxfp8_applies_mirrors_sglangs_mxfp8_view():
    """The indexer wq_b takes a pre-quantized input only with an MXFP8 view on
    FlashInfer's CUTLASS / CuTe DSL backend (SGLang raises otherwise)."""
    from sglang.srt.layers.quantization.fp8 import Fp8LinearMethod

    def attn(use_mxfp8=True, cutedsl=True, k=1280):
        method = Fp8LinearMethod.__new__(Fp8LinearMethod)
        method.use_mxfp8, method.block_fp8_as_mxfp8 = use_mxfp8, False
        method.mxfp8_dense_backend = SimpleNamespace(
            is_flashinfer_cutlass=lambda: False, is_flashinfer_cutedsl=lambda: cutedsl
        )
        wq_b = SimpleNamespace(quant_method=method, weight=torch.zeros(4096, k))
        return SimpleNamespace(indexer=SimpleNamespace(wq_b=wq_b), q_lora_rank=1280)

    assert chain.index_q_mxfp8_applies(attn())
    assert not chain.index_q_mxfp8_applies(attn(use_mxfp8=False))
    assert not chain.index_q_mxfp8_applies(attn(cutedsl=False))
    assert not chain.index_q_mxfp8_applies(attn(k=1536))
    assert not chain.index_q_mxfp8_applies(SimpleNamespace(indexer=None))


@pytest.mark.parametrize("on", [True, False])
def test_kv_is_gathered_before_the_wait_only_after_the_low_ratio_joins(on):
    """Slot kv_before_wait: only the low-ratio prepare joins every KV / index stream
    before the query, so only its swap-AB call may gather the KV before the PDL wait."""
    import importlib
    from contextlib import nullcontext

    from sgl_nvidia_mega.deepseek_v41.attention import prepare

    # the package's step() function shadows the submodule's name
    step_mod = importlib.import_module("sgl_nvidia_mega.deepseek_v41.attention.step")

    st = step_mod.AttentionStep.__new__(step_mod.AttentionStep)
    st.attn = SimpleNamespace(maybe_use_decode_attn_tp=lambda fb: nullcontext())
    st.chain, st.swapab, st.kv_before_wait, st.freqs = (None, None), True, on, None
    fakes = {
        name: mock.Mock(return_value="q")
        for name in ("low_ratio", "multi_stream", "eager")
    }
    view = SimpleNamespace(kv_pool=SimpleNamespace(uniform_fp8=False))
    seen = []
    with (
        mock.patch.multiple(prepare, **fakes),
        mock.patch.object(step_mod.chain, "swapab_takes", return_value=True),
        mock.patch.object(
            step_mod.chain,
            "sparse_decode",
            side_effect=lambda *a, kv_before_wait, **kw: seen.append(kv_before_wait),
        ),
        mock.patch.object(step_mod.chain, "wo_a"),
        mock.patch.object(step_mod.ops, "wo_b"),
    ):
        for name in ("low_ratio", "multi_stream", "eager"):
            st._prepare_path = lambda rows, fb, n=name: getattr(prepare, n)
            st.run(torch.zeros(6, 4), None, None, view=view)
    assert seen == [on, False, False]


def test_kv_before_wait_engagement_counts_graph_captures_only(caplog):
    """The slot's engagement line: once per (model, rows) in a capture; eager calls
    (warm-ups) are not engagement."""
    from sgl_nvidia_mega.deepseek_v41.attention import chain as chain_mod

    chain_mod.kv_before_wait_captures.clear()
    with mock.patch("torch.cuda.is_current_stream_capturing", return_value=False):
        chain_mod._note_kv_before_wait("target", 6, 2)
    assert chain_mod.kv_before_wait_captures == {}
    with (
        mock.patch("torch.cuda.is_current_stream_capturing", return_value=True),
        caplog.at_level("INFO", logger=chain_mod.logger.name),
    ):
        for layer in (2, 3, 4):
            chain_mod._note_kv_before_wait("target", 6, layer)
        chain_mod._note_kv_before_wait("draft", 5, 40)
    assert chain_mod.kv_before_wait_captures == {("target", 6): 3, ("draft", 5): 1}
    lines = [
        r.getMessage() for r in caplog.records if "before its PDL wait" in r.message
    ]
    assert lines == [
        "NVIDIA megakernel: swap-AB gathers the KV before its PDL wait: target at 6 "
        "rows (first layer 2)",
        "NVIDIA megakernel: swap-AB gathers the KV before its PDL wait: draft at 5 "
        "rows (first layer 40)",
    ]
    chain_mod.kv_before_wait_captures.clear()


# ------------------------------------------------------------------------------------
# The attention step head on CPU: which rows fork FlashMLA schedules, the inputs they
# are built from, and the (request, position) rows built once per verify step.


def _verify_mode(verify: bool):
    return SimpleNamespace(is_target_verify=lambda: verify)


def test_schedules_fork_only_where_flashmla_runs():
    """At or below the swap-AB rows (the chain's, else SGLang's own) no layer runs
    FlashMLA's decode kernel, so a schedule built ahead would be work for nothing."""
    assert sched_ahead.first_flashmla_rows(24) == 25
    assert sched_ahead.first_flashmla_rows(0) == 9


def test_schedule_inputs_are_the_layers_and_skip_built_objects():
    """Lengths padded or cut to the rows with 1, as SGLang's FlashMLA layer matches
    them; widths from the same index tensors; an object already scheduled is left."""
    built = SimpleNamespace(tile_scheduler_metadata=torch.zeros(1))
    fresh = {r: SimpleNamespace(tile_scheduler_metadata=None) for r in (0, 1, 4)}
    core = SimpleNamespace(
        low_ratios=(1,),
        present_ratios=(1, 4, 128),
        swa_topk_lengths=torch.tensor([3, 4]),
        swa_page_indices=torch.zeros(2, 128),
        get_flashmla_metadata=lambda r: built if r == 128 else fresh[r],
        sparse_page_indices=lambda r: torch.zeros(2, 64 * (r + 1)),
        sparse_topk_lengths=lambda r: torch.tensor([5, 6, 7, 8]),
    )
    out = sched_ahead._inputs(core, rows=3)
    assert [x[0] for x in out] == [fresh[0], fresh[1], fresh[4]]
    fm, lengths, width, extra_lengths, extra_width = out[1]
    assert lengths.tolist() == [3, 4, 1] and width == 128
    assert extra_lengths.tolist() == [5, 6, 7] and extra_width == 128
    assert out[0][3] is None and out[0][4] == 0


@pytest.mark.parametrize(
    "verify, low, wrote",
    [(True, (1, 2), True), (False, (1, 2), False), (True, (), False)],
)
def test_req_pos_rows_are_built_once_for_verify_steps(verify, low, wrote):
    """Only a target-verify step with low-ratio layers gets the hoisted rows, the ones
    SGLang's decode step builds the same way."""
    meta = SimpleNamespace(
        core_attn_metadata=SimpleNamespace(low_ratios=low),
        low_ratio_req_indices=None,
        low_ratio_pos_i64=None,
    )
    fb = SimpleNamespace(
        forward_mode=_verify_mode(verify),
        positions=torch.arange(12, dtype=torch.int32),
        req_pool_indices=torch.tensor([3, 5]),
        spec_info=SimpleNamespace(draft_token_num=6),
    )
    with mock.patch(
        "sglang.srt.layers.attention.dsv4.dsv41_sparse.token_req_indices",
        wraps=lambda fb, num_tokens: torch.repeat_interleave(
            fb.req_pool_indices, 6, output_size=num_tokens
        ),
    ):
        attention._req_pos_once(meta, fb, rows=12)
    if not wrote:
        assert meta.low_ratio_req_indices is None and meta.low_ratio_pos_i64 is None
        return
    assert meta.low_ratio_req_indices.tolist() == [3] * 6 + [5] * 6
    assert meta.low_ratio_pos_i64.dtype == torch.int64
    assert meta.low_ratio_pos_i64.tolist() == list(range(12))


# ------------------------------------------------------------------------------------
# Hop A at batch 2 (9-16 rows, slot hop_a_batch2): the publishing wo_b takes only row-
# tile scales there, wo_a writes them only when hop A publishes, and SGLang's wo_b never
# receives them.

# the module, not attention.step (the package's step function)
attn_step = importlib.import_module("sgl_nvidia_mega.deepseek_v41.attention.step")
HIDDEN, K = 5120, 2048


@pytest.fixture
def publisher(monkeypatch):
    monkeypatch.setattr(wo_b_publish, "_is_low_latency_linear", lambda linear: True)
    all_reduce = SimpleNamespace(capacity=plan.HOP_A_MAX_ROWS, hidden_size=HIDDEN)
    return wo_b_publish.PublishingMxfp8Linear(all_reduce)


def _wo_b():
    return SimpleNamespace(weight=torch.zeros(HIDDEN, K), bias=None)


def _pair(rows, scale_bytes, row_tiles):
    q = torch.zeros(rows, K, dtype=torch.uint8)
    s = torch.zeros(scale_bytes, dtype=torch.uint8)
    return mxfp8_ll.RowTileMxfp8(q, s) if row_tiles else (q, s)


def test_hop_a_holds_batch_2_only_with_its_slot():
    assert plan.HOP_A_MAX_ROWS == wo_b_publish.ROW_TILE_MAX_ROWS == 16
    assert plan.hop_a_max_rows(batch2=True) == 16
    assert plan.hop_a_max_rows(batch2=False) == plan.HOP_A_BATCH1_ROWS == 8


@pytest.mark.parametrize("rows", [9, 12, 16])
def test_past_8_rows_only_row_tile_scales_publish(publisher, rows):
    full = mxfp8_ll.scale_bytes(rows, K)
    assert publisher.prepare(_wo_b(), _pair(rows, full, row_tiles=True)) is not None
    # a 128x4 pair or too few scale bytes stays on the linear's own path
    assert publisher.prepare(_wo_b(), _pair(rows, full, row_tiles=False)) is None
    assert publisher.prepare(_wo_b(), _pair(rows, full - 1, row_tiles=True)) is None


def test_up_to_8_rows_any_pair_publishes_and_past_16_none(publisher):
    assert publisher.prepare(_wo_b(), _pair(6, 512 * K // 128, row_tiles=False))
    assert not publisher.supports(_wo_b(), 17)
    assert publisher.prepare(_wo_b(), _pair(17, 3 * 512 * K // 128, True)) is None


def test_wo_a_writes_row_tiles_only_for_hop_a_past_8_rows():
    publish = object()
    assert attn_step._publish_row_tiles(12, publish)
    assert not attn_step._publish_row_tiles(8, publish)
    assert not attn_step._publish_row_tiles(12, None)


def test_sglangs_wo_b_never_takes_row_tile_scales():
    attn = SimpleNamespace(wo_b=_wo_b(), attn_tp_size=4)
    o = _pair(12, mxfp8_ll.scale_bytes(12, K), row_tiles=True)
    with pytest.raises(RuntimeError, match="row-tile MXFP8 scales"):
        ops.wo_b(attn, o, publish=lambda wo_b, x: None)


@pytest.mark.parametrize(
    "slots, compiled",
    [({"hop_a", "hop_a_batch2"}, True), ({"hop_a"}, False), (set(), False)],
)
def test_row_tile_wo_a_compiles_only_with_hop_a_batch2(monkeypatch, slots, compiled):
    from sgl_nvidia_mega.deepseek_v41 import runtime

    calls = []
    monkeypatch.setattr(runtime, "SLOTS", frozenset(slots))
    monkeypatch.setattr(chain, "wq_b_ll_applies", lambda attn: False)
    monkeypatch.setattr(
        chain.wo_a_kernel, "precompile", lambda rows, **kw: calls.append((rows, kw))
    )
    chain.precompile(SimpleNamespace(use_fused_wo_a=True), (6, 12, 18))
    tiled = [rows for rows, kw in calls if kw.get("row_tile_scales")]
    assert tiled == ([12] if compiled else [])
