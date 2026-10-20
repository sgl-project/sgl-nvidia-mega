# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""The DSpark draft on CPU: the blocks its forward covers in SGLang's stage order,
the draft router's fast gate top-k, and the draft sampler's chain and RNG contract."""

import contextlib
from contextlib import nullcontext
from types import SimpleNamespace
from unittest import mock
from unittest.mock import patch

import pytest
import torch

from sgl_nvidia_mega.deepseek_v41 import draft_step, sampler
from sgl_nvidia_mega.deepseek_v41.moe import routed
from sgl_nvidia_mega.deepseek_v41.sm103.moe import routed_expert_gemm as reg

# ------------------------------------------------------------------------------------
# The draft forward on the megakernel's steps, on CPU: which draft blocks it covers, and
# that it runs SGLang's draft stage order.


def _decode(log):
    stages = [SimpleNamespace(layer_id=40 + i) for i in range(3)]
    lm = SimpleNamespace(
        stages=stages,
        hc_mult=4,
        forward_embed=lambda ids: log.append("embed") or torch.zeros(len(ids), 4, 8),
    )
    d = draft_step.DraftDecode.__new__(draft_step.DraftDecode)
    d.lm, d.widths, d._tokens = lm, (5, 10, 15), 5
    d._view = draft_step._DraftLm(lm)
    d._bound = {"boundary": None, "moe": None}
    d._slots, d._tail_bf16, d._attn = {w: None for w in d.widths}, False, {}
    d._embed_table = None
    return d, stages


def _step(mode, rows):
    return SimpleNamespace(
        forward_mode=SimpleNamespace(is_target_verify=lambda: mode == "target_verify")
    ), torch.zeros(rows, dtype=torch.int64)


@pytest.mark.parametrize(
    "mode, rows, covered",
    [("target_verify", 10, True), ("target_verify", 12, False), ("idle", 10, False)],
)
def test_the_draft_covers_whole_blocks_of_its_widths(monkeypatch, mode, rows, covered):
    import sglang.srt.model_executor.runner as runner

    monkeypatch.setattr(runner, "get_is_capture_mode", lambda: True)
    d, _ = _decode([])
    fb, ids = _step(mode, rows)
    step = d.plan(ids, fb)
    assert (step is not None) is covered
    if covered:
        assert (step.mode, step.rows) == ("target_verify", rows)


def test_the_stages_run_in_order():
    log = []
    d, stages = _decode(log)
    fake = SimpleNamespace(
        start=lambda st, x: log.append("start"),
        pre=lambda layer, sub, st: log.append(("pre", layer.layer_id)) or "a_in",
        post=lambda layer, sub, out, st: (
            log.append(("post", sub, layer.layer_id)) or "m_in"
        ),
        final=lambda st: (torch.zeros(10, 4, 8), torch.zeros(10, 4)),
        clear=lambda st: log.append("clear"),
    )
    recorder = SimpleNamespace(disable_this_region=nullcontext)
    with (
        patch.object(draft_step, "boundary", fake),
        patch.object(
            draft_step,
            "attention_step",
            side_effect=lambda s, x, st, ported, view: (
                log.append(("attn", s.layer_id)) or "part"
            ),
        ),
        patch.object(
            draft_step.moe,
            "step",
            side_effect=lambda s, x, st: log.append(("moe", s.layer_id)) or "out",
        ),
        patch(
            "sglang.srt.eplb.expert_distribution."
            "get_global_expert_distribution_recorder",
            return_value=recorder,
        ),
        patch("sglang.kernels.ops.layernorm.mhc.mhc_combine", return_value="x"),
    ):
        out = d._forward(
            SimpleNamespace(rows=10),
            torch.zeros(10, dtype=torch.int64),
            None,
            None,
            None,
        )
    assert out.hidden_states == "x"
    assert log[:2] == ["embed", "start"]
    per_stage = [
        [("pre", s.layer_id), ("attn", s.layer_id), ("post", "attn", s.layer_id),
         ("moe", s.layer_id), ("post", "ffn", s.layer_id)]
        for s in stages
    ]  # fmt: skip
    assert log[2:-1] == [e for stage in per_stage for e in stage]
    assert log[-1] == "clear"


def test_a_failed_draft_step_still_clears_its_boundary_state():
    log = []
    d, _ = _decode(log)

    def start(st, x):
        raise RuntimeError("boom")

    fake = SimpleNamespace(start=start, clear=lambda st: log.append("clear"))
    with (
        patch.object(draft_step, "boundary", fake),
        pytest.raises(RuntimeError, match="boom"),
    ):
        d._forward(
            SimpleNamespace(rows=10),
            torch.zeros(10, dtype=torch.int64),
            None,
            None,
            None,
        )
    assert log == ["embed", "clear"]


def _stage(layer_id, *, fast=True):
    attn = SimpleNamespace(
        _use_fast_kernel=fast,
        q_head_norm=False,
        maybe_use_decode_attn_tp=lambda fb: nullcontext(),
    )
    return SimpleNamespace(
        layer_id=layer_id,
        self_attn=attn,
        _run_boundary_attention=lambda x, positions, fb, quantized: "sglang",
    )


def test_a_stage_runs_the_port_only_when_built_for_it():
    """Slot draft_attn ports only the stages on SGLang's fast kernels, the arms the port
    reproduces; any other stage, and every stage without the slot, keeps SGLang's call."""
    from sgl_nvidia_mega.deepseek_v41 import draft_attention

    stages = [_stage(40), _stage(41, fast=False)]
    assert draft_attention.build(stages, frozenset({"draft"})) == {}
    with (
        patch.object(draft_attention.chain, "wq_b_ll_applies", return_value=True),
        patch.object(draft_attention.chain, "swapab_applies", return_value=True),
        patch.object(draft_attention, "wo_a_applies", return_value=False),
    ):
        built = draft_attention.build(
            stages, frozenset({"draft", "draft_attn", "draft_qfront", "draft_swapab"})
        )
    assert built == {40: draft_attention.Kernels(qfront=True, swapab=True)}
    st = SimpleNamespace(fb=None, positions=None, x_quant=None)
    view = object()
    with patch.object(draft_attention, "forward", return_value="port") as port:
        assert [draft_step.attention_step(s, "x", st, built, view) for s in stages] == [
            "port",
            "sglang",
        ]
    assert port.call_args.kwargs["kernels"] == built[40]
    # the draft forward's view of the binding, not SGLang's backend or pool
    assert port.call_args.kwargs["view"] is view


def test_unified_kv_keeps_sglangs_draft_attention():
    """Unified KV writes the draft window through a ring the binding does not offer."""
    from sgl_nvidia_mega.deepseek_v41 import draft_attention

    with patch(
        "sglang.srt.layers.attention.dsv4.unified_kv_gate.is_unified_kv_triton",
        return_value=True,
    ):
        assert draft_attention.build([_stage(40)], frozenset({"draft_attn"})) == {}


@pytest.mark.parametrize("uniform_fp8, chain_takes, want", [
    (False, True, True), (True, True, False), (False, False, False),
])  # fmt: skip
def test_swapab_reads_the_draft_forwards_view(uniform_fp8, chain_takes, want):
    from sgl_nvidia_mega.deepseek_v41 import draft_attention

    view = SimpleNamespace(kv_pool=SimpleNamespace(uniform_fp8=uniform_fp8))
    with patch.object(
        draft_attention.chain, "swapab_takes", return_value=chain_takes
    ) as takes:
        assert draft_attention._swapab_takes(6, "fb", view) is want
    if not uniform_fp8:
        takes.assert_called_once_with(6, "fb", view)


def test_fused_qkv_a_builds_from_linears_whose_scales_are_parameters():
    """SGLang's MXFP8 loader registers the swizzled scales as a parameter; building the
    fused wq_a + wkv must not fail on that at worker start."""
    from sgl_nvidia_mega.deepseek_v41 import draft_attention

    def linear(rows):
        m = torch.nn.Module()
        m.weight = torch.nn.Parameter(
            torch.zeros(rows, 64, dtype=torch.float8_e4m3fn), requires_grad=False
        )
        m.weight_scale_inv_swizzled = torch.nn.Parameter(
            torch.full((rows * 2,), rows // 128, dtype=torch.uint8), requires_grad=False
        )
        return m

    wq_a, wkv = linear(256), linear(128)
    fused = draft_attention.FusedQkvA(wq_a, wkv).fused
    assert fused.weight.shape == (384, 64)
    assert fused.weight_scale_inv_swizzled.tolist() == [2] * 512 + [1] * 256
    # wq_a itself is untouched
    assert wq_a.weight_scale_inv_swizzled.shape == (512,)


def test_draft_embed_is_sglangs_forward_embed_from_the_whole_table():
    """Slot draft_embed looks the draft's tokens up in the target's gathered table; the
    result, broadcast to the hc streams, must be SGLang's forward_embed bit for bit."""
    from sglang.srt.models.deepseek_v4_dspark import DeepseekV4ForCausalLMDSpark

    from sgl_nvidia_mega.deepseek_v41.runtime import embedding

    table = torch.randn(64, 8, dtype=torch.bfloat16)
    lm = SimpleNamespace(embed_tokens=lambda ids: table[ids], hc_mult=4)
    lm.forward_embed = lambda ids: DeepseekV4ForCausalLMDSpark.forward_embed(lm, ids)
    d = draft_step.DraftDecode.__new__(draft_step.DraftDecode)
    d.lm, d._embed_table, d._embed_fell_back = lm, table, False
    ids = torch.tensor([3, 63, 0, 17, 3])
    with patch.object(embedding, "_local", side_effect=lambda e, t, i: t[i]) as local:
        got = d._embed(ids)
    local.assert_called_once()
    assert torch.equal(got, lm.forward_embed(ids))


def test_the_gathered_table_is_built_once_whichever_spec_builds_first():
    """The draft shares the target's embedding module: the second build must reuse the
    first's table (no second all-gather, no second 1 GB a rank)."""
    from sgl_nvidia_mega.deepseek_v41.runtime import embedding

    embed, table = object(), torch.zeros(4, 2)
    with patch.dict(embedding._TABLES, {id(embed): table}):
        with patch.object(embedding.torch.distributed, "all_gather_object") as gather:
            assert embedding.replicate(embed, {}, group=None) == (table, "")
        gather.assert_not_called()


def test_a_requested_draft_prenorm_that_cannot_serve_stops_the_build():
    from sgl_nvidia_mega.deepseek_v41.boundary import prenorm

    assert draft_step._prenorm(frozenset({"draft"})) is False
    with patch.object(prenorm, "reason", return_value="no DeepGEMM"):
        with pytest.raises(RuntimeError, match="draft_prenorm: no DeepGEMM"):
            draft_step._prenorm(frozenset({"draft", "draft_prenorm"}))
    with patch.object(prenorm, "reason", return_value=""):
        assert draft_step._prenorm(frozenset({"draft", "draft_prenorm"})) is True


def test_fused_qkv_a_takes_the_low_latency_gemm_only_at_1_to_8_rows():
    from sgl_nvidia_mega.deepseek_v41 import draft_attention

    qkv = draft_attention.FusedQkvA.__new__(draft_attention.FusedQkvA)
    qkv.ll = True
    qkv.fused = SimpleNamespace(
        weight="w",
        weight_scale_inv_swizzled="w_sf",
        quant_method=SimpleNamespace(apply=lambda layer, x, bias: "dense"),
    )
    mx = draft_attention.mxfp8_ll
    with (
        patch.object(mx, "quantize", return_value=("q", torch.zeros(4))),
        patch.object(mx, "linear", return_value="ll") as ll,
    ):
        assert qkv(torch.zeros(5, 16)) == "ll"
        assert ll.call_args.args[2:] == ("w", "w_sf")
        assert qkv(torch.zeros(10, 16)) == "dense"
        qkv.ll = False
        assert qkv(torch.zeros(5, 16)) == "dense"


def test_draft_embed_reuses_the_target_table_only_through_the_shared_module():
    """The table is shared by module identity: SGLang's attach_shared_modules must hand
    the draft the target's embed_tokens itself, and a draft with its own vocabulary
    (another whole table, ~1.3 GB a GPU) must not build draft_embed."""
    from sglang.srt.models.deepseek_v4_dspark import DeepseekV4ForCausalLMDSpark

    target_embed, lm_head = object(), object()
    draft = SimpleNamespace(
        uses_own_vocab_modules=False,
        markov_head=SimpleNamespace(configure_tp_shard=lambda lm_head: None),
    )
    DeepseekV4ForCausalLMDSpark.attach_shared_modules(
        draft, embed_tokens=target_embed, lm_head=lm_head
    )
    assert draft.embed_tokens is target_embed
    with pytest.raises(RuntimeError, match="own vocabulary"):
        draft_step._embed_table(
            SimpleNamespace(uses_own_vocab_modules=True), frozenset({"draft_embed"})
        )


def test_draft_wo_b_takes_the_low_latency_gemm_only_on_the_fused_wo_a_rows():
    """Slot draft_wo_b_ll feeds wo_b the fused wo_a's MXFP8 rows (FlashInfer's 128x4
    scales) at 1-8 rows; BF16 rows, 9+ rows or the slot off keep SGLang's wo_b."""
    from sglang.srt.layers.quantization.mxfp8_input import Mxfp8SwizzledInput

    from sgl_nvidia_mega.deepseek_v41 import draft_attention

    wo_b = mock.Mock(return_value=("sglang", None))
    wo_b.weight, wo_b.weight_scale_inv_swizzled = "w", "w_sf"
    attn = SimpleNamespace(wo_b=wo_b, attn_tp_size=1)
    rows = lambda n: Mxfp8SwizzledInput(torch.zeros(n, 2048), torch.zeros(8))  # noqa: E731
    with patch.object(draft_attention.mxfp8_ll, "linear", return_value="ll") as ll:
        assert draft_attention._wo_b(attn, rows(5), ll=True) == "ll"
        assert ll.call_args.args[2:] == ("w", "w_sf")
        assert draft_attention._wo_b(attn, rows(10), ll=True) == "sglang"
        assert draft_attention._wo_b(attn, torch.zeros(5, 2048), ll=True) == "sglang"
        assert draft_attention._wo_b(attn, rows(5), ll=False) == "sglang"
    assert ll.call_count == 1


def test_draft_embed_binds_the_target_table_at_its_first_forward():
    """SGLang attaches the target's embedding to the draft after the draft's build, so
    the build must not read it; the first forward takes the table the target's embed
    slot gathered for that module, and raises (every rank alike) if there is none."""
    from sglang.srt.models.deepseek_v4_dspark import DeepseekV4ForCausalLMDSpark

    from sgl_nvidia_mega.deepseek_v41.runtime import embedding

    # the draft as SGLang builds it: no embedding until attach_shared_modules
    lm = SimpleNamespace(
        uses_own_vocab_modules=False,
        embed_tokens=None,
        hc_mult=2,
        markov_head=SimpleNamespace(configure_tp_shard=lambda lm_head: None),
    )
    assert draft_step._embed_table(lm, frozenset({"draft_embed"})) is (
        draft_step._SHARED_TABLE
    )
    target_embed, table = object(), torch.randn(16, 4)
    DeepseekV4ForCausalLMDSpark.attach_shared_modules(
        lm, embed_tokens=target_embed, lm_head=object()
    )
    d = draft_step.DraftDecode.__new__(draft_step.DraftDecode)
    d.lm, d._embed_table = lm, draft_step._SHARED_TABLE
    d._embed_fell_back = False
    ids = torch.tensor([1, 15])
    with (
        patch.dict(embedding._TABLES, {id(target_embed): table}),
        patch.object(embedding, "_local", side_effect=lambda e, t, i: t[i]),
    ):
        assert torch.equal(d._embed(ids), table[ids].unsqueeze(1).repeat(1, 2, 1))
    assert d._embed_table is table
    d._embed_table = draft_step._SHARED_TABLE
    with patch.dict(embedding._TABLES, {}, clear=True):
        with pytest.raises(RuntimeError, match="gathered no table"):
            d._embed(ids)


# ------------------------------------------------------------------------------------
# The fast gate top-k for the DSpark draft router (128 experts, top-3) on CPU: the
# routing it reproduces, which kernel variant each router loads, and the widths each
# routed GEMM takes.

TARGET, DRAFT = (384, 6), (128, 3)
ROWS = [*range(1, 17), 48]


def _reference(logits, bias, top_k, scale, num_valid=None):
    """moe_fused_gate's routing: sqrtsoftplus via log1p (identity above 20), NaN ranked
    first, the lowest id on ties, renormalized in the Triton sum order, then scaled.
    ``num_valid`` None means every row is valid."""
    act = torch.where(logits > 20, logits, torch.log1p(torch.exp(logits))).sqrt()
    biased = act + bias
    biased = torch.where(biased.isnan(), torch.full_like(biased, float("inf")), biased)
    ids = biased.sort(dim=-1, descending=True, stable=True).indices[:, :top_k]
    w = act.gather(1, ids)
    if top_k == 3:  # BLOCK_K 4: (w0 + w2) + (w1 + 0)
        total = (w[:, 0] + w[:, 2]) + w[:, 1]
    else:  # BLOCK_K 8: two threads of 4 slots each
        total = ((w[:, 0] + w[:, 2]) + (w[:, 1] + w[:, 3])) + (w[:, 4] + w[:, 5])
    total = torch.where(total > 0, total, torch.ones_like(total))
    weights = w / total[:, None] * scale
    ids = ids.to(torch.int32)
    if num_valid is not None:
        weights[num_valid:], ids[num_valid:] = 0.0, -1
    return weights, ids


def test_the_draft_routing_ranks_by_the_biased_score_and_weighs_the_unbiased():
    gen = torch.Generator().manual_seed(5)
    for rows in ROWS:
        logits = torch.randn(rows, 128, generator=gen) * 2
        bias = torch.randn(128, generator=gen) * 0.05
        weights, ids = _reference(logits, bias, 3, 1.5)
        act = torch.nn.functional.softplus(logits).sqrt()
        want = torch.topk(act + bias, 3, dim=-1).indices
        assert torch.equal(ids, want.to(torch.int32))
        picked = act.gather(1, want)
        torch.testing.assert_close(weights, picked / picked.sum(-1, keepdim=True) * 1.5)


def test_the_draft_routing_breaks_ties_low_ranks_nans_first_and_pads():
    logits = torch.zeros(4, 128)
    logits[1, 77] = float("nan")
    logits[2, 100] = 25.0  # above 20 softplus passes the logit through
    weights, ids = _reference(logits, torch.zeros(128), 3, 1.0, num_valid=3)
    assert ids[0].tolist() == [0, 1, 2]
    assert ids[1].tolist() == [77, 0, 1] and weights[1, 0].isnan()
    assert ids[2].tolist() == [100, 0, 1]
    assert ids[3].tolist() == [-1, -1, -1] and weights[3].eq(0).all()


def test_no_row_count_keeps_every_row():
    gen = torch.Generator().manual_seed(9)
    logits = torch.randn(6, 128, generator=gen)
    bias = torch.randn(128, generator=gen) * 0.05
    weights, ids = _reference(logits, bias, 3, 1.5)
    full = _reference(logits, bias, 3, 1.5, num_valid=6)
    assert torch.equal(ids, full[1]) and torch.equal(weights, full[0])
    assert ids.ge(0).all() and weights.gt(0).all()


@pytest.mark.parametrize(
    "args,program,symbol,block",
    [
        ((), "gate_topk", "dsv41_gate_topk", 384),
        (DRAFT, "gate_topk_draft", "dsv41_gate_topk_e128_k3", 128),
    ],
)
def test_the_loader_compiles_the_routers_variant(monkeypatch, args, program, symbol, block):  # fmt: skip
    seen = {}
    cuda = SimpleNamespace(
        device=lambda device: contextlib.nullcontext(),
        get_device_capability=lambda device: (10, 3),
    )
    stub = SimpleNamespace(device=lambda *a: a, zeros=lambda *a, **k: None, cuda=cuda)
    monkeypatch.setattr(reg, "torch", stub)

    def compile_source(source, name, arch, options, headers=None):
        seen.update(name=name, arch=arch, options=options, headers=sorted(headers))
        return b"cubin"

    monkeypatch.setattr(reg, "compile_source", compile_source)
    monkeypatch.setattr(reg, "_load", lambda cubin, sym, blk, smem: (sym, blk))
    assert reg.load_gate_topk.__wrapped__(0, *args) == (symbol, (block, 1, 1))
    assert seen == dict(
        name=program,
        arch="sm_103a",
        options=reg.GATE_TOPK_NVRTC_OPTIONS,
        headers=["gate_topk.cuh"],
    )
    with pytest.raises(KeyError):
        reg.load_gate_topk.__wrapped__(0, 128, 6)


@pytest.fixture
def loaded(monkeypatch):
    """RoutedExpertGemm on CPU stand-ins; records the (experts, top_k) gates loaded."""
    seen = []
    monkeypatch.setattr(reg, "check_weights", lambda *args: None)
    monkeypatch.setattr(reg, "load_kernels", lambda index: {})
    monkeypatch.setattr(reg, "load_route_wide", lambda index: None)
    monkeypatch.setattr(reg, "_weight_map", lambda *args: None)
    monkeypatch.setattr(reg, "_scale_map", lambda *args: None)

    def load_gate_topk(index, experts, top_k):
        seen.append((experts, top_k))
        return object()

    monkeypatch.setattr(reg, "load_gate_topk", load_gate_topk)
    return seen


def _gemm(experts, top_k, gate_topk):
    w = torch.zeros((experts, 1), dtype=torch.uint8)
    return reg.RoutedExpertGemm(
        w, w, w, w, None, top_k=top_k, max_rows=48, gate_topk=gate_topk
    )


def test_the_gemm_loads_the_variant_of_its_router(loaded):
    assert _gemm(*DRAFT, gate_topk=True).gate_topk_kernel is not None
    assert _gemm(*TARGET, gate_topk=True).gate_topk_kernel is not None
    assert loaded == [DRAFT, TARGET]
    assert _gemm(*DRAFT, gate_topk=False).gate_topk_kernel is None
    assert _gemm(128, 6, gate_topk=True).gate_topk_kernel is None
    assert _gemm(384, 3, gate_topk=True).gate_topk_kernel is None
    assert loaded == [DRAFT, TARGET]


def test_the_gate_takes_logits_of_its_routers_width(loaded):
    draft, target = _gemm(*DRAFT, gate_topk=True), _gemm(*TARGET, gate_topk=True)
    for rows in (1, 12, 16, 48):
        assert draft.supports_gate_topk(torch.zeros(rows, 128), torch.zeros(128))
        assert target.supports_gate_topk(torch.zeros(rows, 384), torch.zeros(384))
        assert not draft.supports_gate_topk(torch.zeros(rows, 384), torch.zeros(128))
        assert not target.supports_gate_topk(torch.zeros(rows, 128), torch.zeros(384))
    bf16 = torch.zeros(6, 128, dtype=torch.bfloat16)
    assert not draft.supports_gate_topk(bf16, torch.zeros(128))
    assert not draft.supports_gate_topk(torch.zeros(49, 128), torch.zeros(128))


def _mlp(**config):
    """A DSpark draft stage's MoE as the slot reads it (the target's routing, 128 x 3)."""
    topk_config = SimpleNamespace(
        scoring_func="sqrtsoftplus",
        sqrtsoftplus_log1p=True,
        renormalize=True,
        num_fused_shared_experts=0,
        use_grouped_topk=False,
        custom_routing_function=None,
        correction_bias=torch.zeros(128),
    )
    vars(topk_config).update(config)
    return SimpleNamespace(
        is_hash=False,  # the draft stages are built with is_nextn, so never hashed
        gate=SimpleNamespace(e_score_correction_bias_vl=None),
        topk=SimpleNamespace(topk_config=topk_config),
    )


def test_the_draft_router_is_one_the_gate_reproduces():
    assert routed.gate_applies(_mlp())
    assert not routed.gate_applies(_mlp(use_grouped_topk=True))
    assert not routed.gate_applies(_mlp(renormalize=False))
    assert not routed.gate_applies(_mlp(sqrtsoftplus_log1p=False))


# ------------------------------------------------------------------------------------
# The draft sampler's step on CPU: what it hands the chain and SGLang's RNG contract.

BS, GAMMA, VOCAB, HIDDEN = 2, 5, 12, 8


class FakeChain:
    def __init__(self):
        self.calls = []

    def fork_noise(self, *, rows, key, offset):
        self.calls.append(("fork", rows, key))

    def join_noise(self):
        self.calls.append(("join",))

    def sample(
        self, *, base_logits, anchor_tokens, temperatures, logits_cache, draft_tokens
    ):
        self.calls.append(("sample", tuple(base_logits.shape), logits_cache))
        draft_tokens.fill_(7)


def _sampler_step(*, greedy_draft):
    weight = torch.randn(VOCAB, HIDDEN)
    lm = SimpleNamespace(
        stages=[SimpleNamespace(norm=lambda x: x * 2)], collapse_hc_head=lambda x: x
    )
    chain = FakeChain()
    noise = SimpleNamespace(key=9, offset=torch.zeros(1, dtype=torch.int64))
    step_logits = None if greedy_draft else torch.zeros(4, GAMMA, VOCAB)
    step = sampler._Step(
        lm=lm,
        chain=chain,
        lm_head=lambda x: x @ weight.T,
        bs=BS,
        gamma=GAMMA,
        greedy_draft=greedy_draft,
        noise=noise,
        draft_tokens=torch.zeros(4 * GAMMA, dtype=torch.int64),
        step_logits=step_logits,
        step_temperatures=torch.zeros(4),
    )
    return step, chain, noise, step_logits


@pytest.mark.parametrize("greedy_draft", [False, True])
def test_a_block_draws_noise_and_advances_the_counter_as_sglang_does(
    greedy_draft, monkeypatch
):
    """draft_markov_block advances noise.offset once per sampled block and never for a
    greedy draft; a different count would desynchronize the accept's noise."""
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)
    step, chain, noise, step_logits = _sampler_step(greedy_draft=greedy_draft)
    hidden = torch.randn(BS * GAMMA, HIDDEN)
    tokens, features = step(hidden, torch.arange(BS))
    assert tokens.shape == (BS, GAMMA) and bool(tokens.eq(7).all())
    # the confidence head reads the collapsed hidden states, as compute_base_logits returns
    assert features is hidden
    sample = ("sample", (BS, GAMMA, VOCAB), step_logits)
    if greedy_draft:
        assert chain.calls == [sample] and int(noise.offset) == 0
    else:
        assert chain.calls == [("fork", BS, 9), ("join",), sample]
        assert int(noise.offset) == 1


def test_a_rank_that_cannot_run_the_chain_refuses_on_every_rank():
    """The slot was asked for, so an unsupported setup refuses at startup instead of
    falling back to SGLang's proposal; every rank's reason is listed."""
    assert sampler.refusal_message(["", ""]) == ""
    message = sampler.refusal_message(["", "an FP32 markov_w2", "", "a TP size"])
    assert "rank 1: an FP32 markov_w2" in message and "rank 3: a TP size" in message
    assert "rank 0" not in message


def test_ranks_that_built_different_proposals_refuse():
    """Ranks serving different batch sizes would capture graphs that wait on each other."""
    assert sampler.mismatch_message([(64, True)] * 4) == ""
    message = sampler.mismatch_message([(64, True), (8, True), (64, True), (64, False)])
    assert (
        "rank 1: batch 1-8" in message
        and "rank 3: batch 1-64, low-M LM head False" in message
    )


def test_buffers_the_chain_cannot_write_into_are_refused():
    """The chain writes int64 draft tokens and reads float32 step temperatures in place;
    another SGLang layout must refuse at startup, not corrupt the proposal."""

    def reason(tokens, temperatures):
        return sampler.buffer_reason(
            max_bs=4, gamma=GAMMA, draft_tokens=tokens, step_temperatures=temperatures
        )

    assert reason(torch.zeros(4 * GAMMA, dtype=torch.int64), torch.zeros(4)) == ""
    narrow = reason(torch.zeros(4 * GAMMA, dtype=torch.int32), torch.zeros(4))
    assert "draft tokens" in narrow
    half = reason(torch.zeros(4 * GAMMA, dtype=torch.int64), torch.zeros(4).half())
    assert "step temperatures" in half


@pytest.mark.parametrize("slots", [frozenset(), frozenset({"draft_lm_head"})])
def test_without_the_markov_slot_the_draft_keeps_sglang_s_proposal(slots):
    from sgl_nvidia_mega.deepseek_v41 import draft

    built = draft.DraftProposal(SimpleNamespace(), deadline_s=30.0, slots=slots)
    assert built.draft_sampler is None
    # the draft's forward stays SGLang's at every step
    assert built.widths == () and built.plan(None, None) is None


def test_the_draft_sampler_builds_the_chain_over_sglang_s_buffers(monkeypatch):
    """SGLang calls the factory once with its own buffers; they reach the chain as
    given, with the spec's deadline and LM-head choice."""
    from sgl_nvidia_mega.deepseek_v41 import draft

    calls = []
    monkeypatch.setattr(
        sampler, "build_steps", lambda lm, **kw: calls.append((lm, kw)) or {2: "step"}
    )
    lm = SimpleNamespace()
    built = draft.DraftProposal(
        lm, deadline_s=30.0, slots=frozenset({"markov", "draft_lm_head"})
    )
    buffers = dict(
        gamma=GAMMA,
        max_bs=4,
        greedy_draft=False,
        noise="noise",
        draft_tokens="tokens",
        step_logits="logits",
        step_temperatures="temperatures",
    )
    steps = built.draft_sampler(**buffers)
    assert calls == [(lm, {"low_m": True, "deadline_s": 30.0, **buffers})]
    assert steps(2) == "step" and steps(3) is None


def test_the_draft_spec_follows_the_target_spec():
    """The draft shares the target's GPUs and contract version; a drifted copy would
    accept a setup the target refuses, or pass an API check the target fails."""
    from sgl_nvidia_mega.deepseek_v41 import draft, runtime, spec

    assert draft.SPEC.api_version == spec.API_VERSION
    for field in ("isas", "tp_sizes", "dtypes", "decode_graph_backends"):
        assert getattr(draft.SPEC.requires, field) == getattr(spec.SPEC.requires, field)
    # the draft forward's widest step with slot "draft", else SGLang's forward
    assert draft.SPEC.max_rows == (48 if "draft" in runtime.SLOTS else 0)
