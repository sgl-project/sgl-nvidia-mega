# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""CPU checks of the MiniMax-M3.1 DSpark draft's slots, markov (shipped) and
draft_lm_head (opt-in): the slot list and what each slot needs; the numerics each spec
declares with them; the servers refused at argument resolution; their toolchains; the
target's summary and the engagement check its executor runs; the draft spec's built
object, which offers SGLang's folded proposal the draft sampler only with slot markov,
builds its buffers at the spec's build and says on every rank which proposal serves;
and the sampler's refusals of buffers and proposals it does not fit."""

import logging
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest
import torch
from sglang.kernels.plugin_kernels.nvidia.spec import NumericsKind
from sglang.srt import runtime_context
from sglang.srt.environ import envs
from sglang.srt.model_executor import cuda_graph_config

from sgl_nvidia_mega.minimax_m31 import HF_CONFIG, MAX_REQUESTS, draft, draft_step
from sgl_nvidia_mega.minimax_m31 import moe as m31_moe
from sgl_nvidia_mega.minimax_m31 import plan as m31_plan
from sgl_nvidia_mega.minimax_m31 import runtime as m31_runtime
from sgl_nvidia_mega.minimax_m31 import sampler
from sgl_nvidia_mega.minimax_m31 import spec as m31_spec
from sgl_nvidia_mega.minimax_m31.moe import moe_nofill as mn
from sgl_nvidia_mega.minimax_m31.moe import moe_swap
from sgl_nvidia_mega.minimax_m31.runtime import agree as m31_agree
from sgl_nvidia_mega.minimax_m31.runtime import executor as m31_executor
from sgl_nvidia_mega.minimax_m31.sm103 import host, nvcc

TARGET_SLOTS = frozenset({"mega", "moe_nofill", "moe_swap"})
VERIFY_WIDTHS = tuple(range(8, 129, 8))

# ------------------------------------------------------------------ the slot list


def test_markov_ships_and_draft_lm_head_is_opt_in():
    """Unset, the slot list runs markov; draft_lm_head runs only when listed. Both are
    the draft spec's: no field of the target's plan."""
    env = m31_runtime.SLOTS_ENV
    assert m31_runtime.requested_slots({}) == TARGET_SLOTS | {"markov"}
    assert "markov" in m31_runtime.SHIPPED_SLOTS
    assert "draft_lm_head" not in m31_runtime.SHIPPED_SLOTS
    # lm_head_mxfp8 is the target's (test_minimax_m31_lm_head_mxfp8.py)
    assert m31_runtime.OPT_IN_SLOTS == ("draft_lm_head", "lm_head_mxfp8")
    assert m31_runtime.DRAFT_SLOTS == ("markov", "draft_lm_head")
    listed = m31_runtime.requested_slots(
        {env: "mega,moe_nofill,moe_swap,markov,draft_lm_head"}
    )
    assert listed == TARGET_SLOTS | {"markov", "draft_lm_head"}
    assert m31_runtime.target_slots(listed) == TARGET_SLOTS
    assert m31_runtime.draft_slots(listed) == {"markov", "draft_lm_head"}
    assert m31_runtime.requested_slots({env: "mega"}) == {"mega"}
    for slot in m31_runtime.DRAFT_SLOTS:
        assert slot not in m31_plan.Slots.__struct_fields__
        assert slot not in m31_plan.Facts.__struct_fields__
    assert set(m31_runtime.DRAFT_TOLERANCES) == set(m31_runtime.DRAFT_SLOTS)


def test_the_slot_list_refusals_name_the_draft_slots():
    assert m31_runtime.slot_reasons(frozenset(m31_runtime.SHIPPED_SLOTS)) == []
    assert m31_runtime.slot_reasons(frozenset({"markov", "draft_lm_head"})) == []
    (needs,) = m31_runtime.slot_reasons(frozenset({"draft_lm_head"}))
    assert "slot 'draft_lm_head' needs slot 'markov'" in needs
    (unknown,) = m31_runtime.slot_reasons(frozenset({"markov", "draft_head"}))
    assert "unknown slot 'draft_head'" in unknown
    assert (
        "(known: mega, moe_nofill, moe_swap, markov, draft_lm_head, lm_head_mxfp8)"
        in unknown
    )


def test_the_draft_slots_state_their_tolerances_in_the_draft_spec():
    """The draft spec's numerics: its backbone's tolerances and the requested draft
    slots'. The target's numerics judge its own verify steps: a draft slot does not
    change them."""
    backbone = draft.BACKBONE_TOLERANCES
    assert dict(draft._numerics(TARGET_SLOTS).tolerances) == backbone
    markov = dict(draft._numerics(TARGET_SLOTS | {"markov"}).tolerances)
    assert markov == {**backbone, **m31_runtime.DRAFT_TOLERANCES["markov"]}
    both = draft._numerics(frozenset({"markov", "draft_lm_head"}))
    assert both.kind is NumericsKind.TOLERANCE
    assert dict(both.tolerances) == {
        **backbone,
        **m31_runtime.DRAFT_TOLERANCES["markov"],
        **m31_runtime.DRAFT_TOLERANCES["draft_lm_head"],
    }
    assert m31_runtime.draft_tolerances(TARGET_SLOTS) == {}
    # the shipped list: the draft spec declares slot markov's bound
    assert dict(draft.NUMERICS.tolerances) == dict(
        draft._numerics(m31_runtime.SLOTS).tolerances
    )
    assert (
        m31_spec._numerics(frozenset({"moe_nofill", "moe_swap", "markov"})).kind
        is NumericsKind.BITWISE
    )
    shipped = m31_spec._numerics(frozenset(m31_runtime.SHIPPED_SLOTS))
    assert dict(shipped.tolerances) == m31_spec.MEGA_TOLERANCES


# ------------------------------------------------------------ argument resolution


def _cfg(lm_head_dtype=None):
    return SimpleNamespace(speculative_dspark_lm_head_dtype=lm_head_dtype)


@pytest.mark.skipif(
    not hasattr(envs, "SGLANG_DSPARK_OPT_INT8_MARKOV_WALK"),
    reason=m31_spec.NO_M31_HOST,
)
def test_a_server_whose_proposal_the_slots_cannot_replace_is_refused():
    """The draft slots replace SGLang's folded DSpark proposal; a server that turned it
    off, or asked for SGLang's own replacements of its parts, is refused."""
    markov = TARGET_SLOTS | {"markov"}
    both = markov | {"draft_lm_head"}
    assert m31_spec.draft_config_reasons(_cfg(), markov) == []
    assert m31_spec.draft_config_reasons(_cfg(), both) == []
    with envs.SGLANG_DSPARK_FOLDED_PROPOSAL.override(False):
        (reason,) = m31_spec.draft_config_reasons(_cfg(), markov)
        assert "SGLANG_DSPARK_FOLDED_PROPOSAL is off" in reason
        assert "['markov']" in reason
        # no draft slot requested: SGLang's eager proposal is the server's choice
        assert m31_spec.draft_config_reasons(_cfg(), TARGET_SLOTS) == []
    with envs.SGLANG_DSPARK_OPT_INT8_MARKOV_WALK.override(True):
        (reason,) = m31_spec.draft_config_reasons(_cfg(), markov)
        assert "SGLANG_DSPARK_OPT_INT8_MARKOV_WALK" in reason
        assert m31_spec.draft_config_reasons(_cfg(), TARGET_SLOTS) == []
    # SGLang's own quantized draft LM head: slot markov runs it, draft_lm_head doubles it
    assert m31_spec.draft_config_reasons(_cfg("mxfp8"), markov) == []
    (reason,) = m31_spec.draft_config_reasons(_cfg("mxfp8"), both)
    assert "--speculative-dspark-lm-head-dtype mxfp8" in reason
    assert "slot draft_lm_head" in reason


def test_the_draft_slots_toolchains():
    """Slot markov builds with the CUDA Python bindings and nvcc (no version floor: the
    QMUL4 one is slot mega's) and draws its noise with Triton; slot draft_lm_head
    launches SGLang's FlashInfer MXFP8 linear."""

    def reasons(slots, missing=(), nvcc_reason=""):
        with (
            patch.object(m31_spec, "_has_module", lambda name: name not in missing),
            patch.object(nvcc, "why_not", return_value=nvcc_reason) as why_not,
        ):
            return m31_spec.toolchain_reasons(frozenset(slots)), why_not

    found, why_not = reasons({"markov"})
    assert found == []
    why_not.assert_called_once_with(None)
    found, why_not = reasons({"mega", "markov"})
    why_not.assert_called_once_with(nvcc.QMUL4_VERSION)
    found, why_not = reasons({"moe_nofill", "moe_swap"})
    why_not.assert_not_called()
    assert reasons({"markov"}, nvcc_reason="no CUDA toolkit (nvcc) found")[0] == [
        "no CUDA toolkit (nvcc) found"
    ]
    assert reasons({"markov"}, missing=("cuda.bindings",))[0] == [
        "needs the CUDA Python bindings"
    ]
    assert reasons({"markov"}, missing=("triton",))[0] == ["slot markov needs Triton"]
    assert reasons({"markov", "draft_lm_head"}, missing=("flashinfer",))[0] == [
        "slot draft_lm_head needs FlashInfer"
    ]
    assert reasons({"markov"}, missing=("flashinfer",))[0] == []


# ------------------------------------------------------------------ the target's build


def _build(slots, off=""):
    """``runtime.build`` on the verify widths with ``slots`` requested and every slot's
    build serving them; ``off``: why the server has no folded proposal."""
    served = mn.Bound(widths=VERIFY_WIDTHS)
    swap = moe_swap.Bound(widths=VERIFY_WIDTHS)
    ctx = SimpleNamespace(widths=VERIFY_WIDTHS, tokens_per_request=8, binding=None)
    with (
        patch.object(m31_runtime, "SLOTS", frozenset(slots)),
        patch.object(m31_moe, "build", return_value=served),
        patch.object(moe_swap, "build", return_value=swap),
        patch.object(m31_agree, "agree_tp", lambda value, what: None),
        patch.object(draft, "proposal_off_reason", return_value=off),
    ):
        return m31_runtime.build("model", "hf", ctx)


def test_the_targets_summary_names_the_draft_slots_and_arms_the_check():
    """The draft's slots are no slot of the target's plan (a requested one serves none
    of its widths, which is no refusal); its summary names them and its executor checks
    that the draft engaged them."""
    built = _build({"moe_nofill", "moe_swap", "markov", "draft_lm_head"})
    assert built.summary().endswith(
        "slots by width 8-128: moe_nofill, moe_swap; draft slots ['draft_lm_head', "
        "'markov'], built by the DSpark draft's spec"
    )
    assert isinstance(built._draft_check, draft.EngagementCheck)
    assert built._draft_check.slots == ["draft_lm_head", "markov"]
    # the target itself offers no sampler
    assert built.draft_sampler is None
    stock = _build({"moe_nofill", "moe_swap"})
    assert "draft slots" not in stock.summary()
    assert stock._draft_check is None


def test_eager_test_servers_do_not_arm_the_check():
    off = "decode CUDA graphs are disabled"
    built = _build({"moe_nofill", "moe_swap", "markov"}, off=off)
    assert f"draft slots ['markov'] not engaged: {off}" in built.summary()
    assert built._draft_check is None


def test_the_executor_runs_the_engagement_check_until_it_passes():
    answers = [False, False, True]
    check = Mock(side_effect=lambda: answers.pop(0))
    built = m31_executor.MegakernelDecode(
        None, widths=(8,), plan={}, bound={}, draft_check=check
    )
    batch = SimpleNamespace(
        forward_mode=SimpleNamespace(is_target_verify=lambda: False)
    )
    for _ in range(5):
        assert built.plan(torch.zeros(8), batch) is None
    assert check.call_count == 3

    def stop():
        raise RuntimeError("never asked for the draft sampler")

    built = m31_executor.MegakernelDecode(
        None, widths=(8,), plan={}, bound={}, draft_check=stop
    )
    with pytest.raises(RuntimeError, match="never asked"):
        built.plan(torch.zeros(8), batch)


def test_no_folded_proposal_only_without_decode_graphs():
    def reason(disabled):
        def check(phase, backend):
            assert (phase, backend) == ("decode", "disabled")
            return disabled

        with patch.object(cuda_graph_config, "check_cuda_graph_backend", check):
            return draft.proposal_off_reason()

    assert reason(False) == ""
    assert "decode CUDA graphs are disabled" in reason(True)
    assert m31_spec.PARITY_ENV in reason(True)


# -------------------------------------------------------- the draft spec's built object


def _draft_step(slots):
    """A DraftStep without its backbone (a GPU build): what ``_build_proposal`` and
    ``_draft_sampler`` read."""
    built = draft_step.DraftStep.__new__(draft_step.DraftStep)
    built.model = SimpleNamespace(name="draft")
    built.slots = m31_runtime.draft_slots(frozenset(slots))
    built.prebuilt = None
    return built


CTX = SimpleNamespace(deadline_s=30.0, tokens_per_request=8)


def test_without_markov_the_draft_keeps_sglangs_proposal(caplog):
    built = _draft_step(TARGET_SLOTS)
    with (
        patch.object(sampler, "prebuild") as prebuild,
        caplog.at_level(logging.INFO, logger=draft_step.__name__),
    ):
        built.proposal = built._build_proposal(CTX, rank=2)
    prebuild.assert_not_called()
    assert built.draft_sampler is None
    assert built.proposal == "SGLang's proposal (slot markov off)"
    assert "proposal: SGLang's proposal (slot markov off)" in built.summary()
    (line,) = caplog.messages
    assert "rank 2" in line and "slot markov is not requested" in line


@pytest.mark.parametrize(
    "slots, lm_head",
    [
        ({"markov"}, "SGLang's LM head (draft slots ['markov'])"),
        (
            {"markov", "draft_lm_head"},
            "an MXFP8 copy of the LM head (draft slots ['draft_lm_head', 'markov'])",
        ),
    ],
)
def test_with_markov_the_draft_offers_the_sampler(slots, lm_head):
    """The chain's buffers are built with the draft spec (before SGLang sizes its KV
    pools), for the draft tokens of a block and SGLang's draft mode, with the build
    context's deadline; the folded proposal's buffers reach ``sampler.build_steps``."""
    built = _draft_step(TARGET_SLOTS | slots)
    with (
        patch.object(draft, "proposal_off_reason", return_value=""),
        patch.object(sampler, "prebuild", return_value="prebuilt") as prebuild,
    ):
        built.proposal = built._build_proposal(CTX, rank=0)
    prebuild.assert_called_once_with(
        built.model,
        deadline_s=30.0,
        slots=frozenset(slots),
        gamma=7,
        greedy_draft=envs.SGLANG_DSPARK_USE_GREEDY_DRAFT.get(),
    )
    assert built.prebuilt == "prebuilt"
    assert built.proposal == (
        f"Markov chain in one kernel at 1-{MAX_REQUESTS} requests, {lm_head}"
    )
    assert built.summary().startswith("DSpark draft block of 7 rows: 5 layers in one")
    assert built.summary().endswith(f"proposal: {built.proposal}")
    assert callable(built.draft_sampler)

    buffers = dict(
        gamma=7,
        max_bs=16,
        greedy_draft=False,
        noise="noise",
        draft_tokens="tokens",
        step_logits="logits",
        step_temperatures="temperatures",
    )
    with (
        patch.object(draft, "_COUNTS", {"built": 1, "sampler": 0}) as counts,
        patch.object(sampler, "build_steps", return_value={1: "one"}) as build_steps,
    ):
        steps = built.draft_sampler(**buffers)
    build_steps.assert_called_once_with(built.model, prebuilt="prebuilt", **buffers)
    assert counts["sampler"] == 1
    # bs -> the step, or None where SGLang's proposal stays
    assert steps(1) == "one" and steps(17) is None


def test_with_markov_and_a_greedy_draft_the_chain_keeps_no_logits():
    built = _draft_step({"markov"})
    with (
        patch.object(draft, "proposal_off_reason", return_value=""),
        patch.object(sampler, "prebuild") as prebuild,
        envs.SGLANG_DSPARK_USE_GREEDY_DRAFT.override(True),
    ):
        built._build_proposal(CTX, rank=0)
    assert prebuild.call_args.kwargs["greedy_draft"] is True


def test_an_eager_test_server_keeps_sglangs_proposal_and_says_so(caplog):
    """Disabled decode CUDA graphs (the parity switch's eager steps): SGLang folds no
    proposal, so the slots build nothing; every rank logs it."""
    built = _draft_step(TARGET_SLOTS | {"markov", "draft_lm_head"})
    off = "decode CUDA graphs are disabled"
    with (
        patch.object(draft, "proposal_off_reason", return_value=off),
        patch.object(sampler, "prebuild") as prebuild,
        caplog.at_level(logging.WARNING, logger=draft_step.__name__),
    ):
        built.proposal = built._build_proposal(CTX, rank=3)
    prebuild.assert_not_called()
    assert built.draft_sampler is None and built.prebuilt is None
    assert built.proposal == (
        "SGLang's proposal (draft slots ['draft_lm_head', 'markov'] not engaged: "
        f"{off})"
    )
    (line,) = caplog.messages
    assert "['draft_lm_head', 'markov'] are not engaged on rank 3" in line
    assert off in line


def test_a_markov_head_the_chain_cannot_serve_refuses_the_draft():
    """SGLang gathers a draft spec's model reasons over the ranks and refuses the
    server with all of them: slot markov's Markov head check is one."""
    model = SimpleNamespace()
    group = SimpleNamespace(tp_group=SimpleNamespace(world_size=4))
    assert draft_step.proposal_disable_reasons(model, TARGET_SLOTS) == []
    with (
        patch.object(runtime_context, "get_parallel", return_value=group),
        patch.object(sampler, "head_reason", return_value="") as head_reason,
    ):
        assert draft_step.proposal_disable_reasons(model, {"markov"}) == []
        # a block's draft tokens: the 8 verify rows less the anchor
        head_reason.assert_called_once_with(model, gamma=7, world=4)
        head_reason.return_value = "a GatedMarkovHead Markov head (needs VanillaMarkov)"
        assert draft_step.proposal_disable_reasons(model, {"markov"}) == [
            "slot markov: a GatedMarkovHead Markov head (needs VanillaMarkov)"
        ]


# ------------------------------------------------------------------------ the sampler


def _lm(*, vocab=1024, world=4, rank=1, **lm_head):
    shard = vocab // world
    fields = dict(
        weight=torch.empty(shard, HF_CONFIG["hidden_size"], dtype=torch.bfloat16),
        org_vocab_size=vocab,
        num_embeddings_per_partition=shard,
        shard_indices=SimpleNamespace(org_vocab_start_index=rank * shard),
    )
    fields.update(lm_head)
    return SimpleNamespace(
        markov_head=SimpleNamespace(vocab_size=vocab),
        lm_head=SimpleNamespace(**fields),
        logits_mup_width_multiplier=None,
    )


def test_the_sampler_names_why_the_chain_cannot_read_an_lm_head():
    """Checked when the folded proposal asks for the sampler, the target's LM head
    attached: the chain reads this rank's BF16 shard of the even vocab split."""

    def reason(lm):
        return sampler.lm_head_reason(lm, world=4, rank=1)

    assert reason(_lm()) == ""
    no_head = SimpleNamespace(markov_head=SimpleNamespace(vocab_size=1024))
    no_head.lm_head = None
    assert "no LM head" in reason(no_head)
    mup = _lm()
    mup.logits_mup_width_multiplier = 4.0
    assert "muP" in reason(mup)
    assert "BF16" in reason(_lm(weight=torch.empty(256, 8)))
    assert "tokens for a" in reason(_lm(org_vocab_size=1000))
    uneven = _lm(shard_indices=SimpleNamespace(org_vocab_start_index=0))
    assert "even split" in reason(uneven)
    # the shape the build sized the LM-head copy for
    assert "shape (256, 8)" in reason(
        _lm(weight=torch.empty(256, 8, dtype=torch.bfloat16))
    )
    no_markov = _lm()
    no_markov.markov_head = None  # head_reason names it
    assert reason(no_markov) == ""


def test_buffers_and_rank_agreement_messages():
    tokens = torch.empty(16 * 7, dtype=torch.int64)
    temps = torch.empty(16, dtype=torch.float32)
    ok = dict(max_bs=16, gamma=7, draft_tokens=tokens, step_temperatures=temps)
    assert sampler.buffer_reason(**ok) == ""
    assert sampler.buffer_reason(**{**ok, "draft_tokens": tokens.int()})
    assert sampler.buffer_reason(**{**ok, "step_temperatures": temps[:8]})
    # a sampling draft keeps the whole vocabulary's step logits on every rank
    kept = (64, torch.bfloat16)
    logits = torch.empty(16, 7, 64, dtype=torch.bfloat16)
    assert sampler.buffer_reason(**ok, step_logits=logits, kept=kept) == ""
    assert "without a step-logits buffer" in sampler.buffer_reason(**ok, kept=kept)
    shard = sampler.buffer_reason(**ok, step_logits=logits[:, :, :16], kept=kept)
    assert "(16, 7, 16)" in shard and "[16, 7, 64]" in shard
    assert sampler.buffer_reason(**ok, step_logits=logits.float(), kept=kept)
    assert sampler.buffer_reason(**ok, step_logits=logits[:8], kept=kept)
    assert sampler.refusal_message(["", ""]) == ""
    assert "rank 1: x" in sampler.refusal_message(["", "x"])
    assert sampler.mismatch_message([(8, "SGLang's")] * 4) == ""
    assert "rank 3: batch 1-4" in sampler.mismatch_message(
        [(8, "SGLang's")] * 3 + [(4, "SGLang's")]
    )


def test_a_proposal_the_build_does_not_fit_is_refused():
    """The sampler reuses what the draft spec's build allocated: a proposal of other
    draft tokens or another draft mode is refused on every rank, with both."""
    chain = SimpleNamespace(max_rows=16, device_bytes=lambda: {"noise": 0})
    prebuilt = sampler.Prebuilt(
        chain=chain,
        lm_head=SimpleNamespace(weight=None),
        gamma=7,
        greedy_draft=False,
        world=1,
        rank=0,
        cpu_group=None,
        slots=frozenset({"markov"}),
    )
    buffers = dict(
        max_bs=16,
        noise=None,
        draft_tokens=torch.empty(16 * 7, dtype=torch.int64),
        step_logits=None,
        step_temperatures=torch.empty(16, dtype=torch.float32),
    )
    for gamma, greedy, want in (
        (6, False, "6 and a sampling"),
        (7, True, "greedy draft"),
    ):
        with pytest.raises(ValueError, match="built the chain for 7") as raised:
            sampler.build_steps(
                SimpleNamespace(),
                prebuilt=prebuilt,
                gamma=gamma,
                greedy_draft=greedy,
                **buffers,
            )
        assert want in str(raised.value)
        assert "the markov slot cannot run: rank 0:" in str(raised.value)
    assert sampler._memory({"a": 2**20, "b": 2**19}) == (
        "1.5 MiB of device memory (a 1.0, b 0.5 MiB)"
    )


def test_a_build_step_that_fails_on_one_rank_raises_on_every_rank():
    """One rank: the step itself. Over a TP group: the package's ``host.agree``, which
    gathers every rank's error."""
    assert sampler._on_every_rank(None, lambda: 3, "allocating") == 3
    step = Mock()
    with patch.object(host, "agree", return_value="agreed") as agree:
        assert sampler._on_every_rank("group", step, "allocating") == "agreed"
    agree.assert_called_once_with("group", step, "allocating")
    step.assert_not_called()


def test_the_bound_measure_allows_one_ulp_of_the_largest_bias():
    """sampler.bound_ulps: |difference| past one BF16 ulp of the step's largest |bias|
    (kept logit less base, over the last dimension), in BF16 ulps of |kept| + |base|."""
    reference = torch.tensor([[40.0, 0.0078125, -3.0]])
    base = torch.tensor([[36.0, 0.0, -3.0]])  # biases 4, 2^-7, 0: largest 4, ulp 2^-5
    delta = torch.tensor([[0.25, 0.03125, 0.0]])
    got = sampler.bound_ulps(delta, reference, base)
    # 76 has a BF16 ulp of 0.5: (0.25 - 2^-5) / 0.5; the second is within the allowance
    want = torch.tensor([[(0.25 - 2**-5) / 0.5, 0.0, 0.0]])
    assert torch.allclose(got, want), got
    assert float(sampler.bf16_ulp(torch.tensor(1.0))) == 2**-7
    assert float(sampler.bf16_ulp(torch.tensor(1.9))) == 2**-7
    assert float(sampler.bf16_ulp(torch.tensor(2.0))) == 2**-6


# ------------------------------------------------------------- the chain's host code


def _markov():
    """sm103/draft/markov.py, which imports Triton for its noise kernel."""
    pytest.importorskip("triton")
    from sgl_nvidia_mega.minimax_m31.sm103.draft import markov

    return markov


def test_the_chain_takes_m31_and_refuses_what_its_keys_cannot_hold():
    markov = _markov()

    def reason(**kw):
        args = dict(
            world=4,
            vocab_size=200_064,
            num_steps=7,
            embed_weight=torch.empty(200_064, markov.RANK_DIM, dtype=torch.bfloat16),
            head_weight=torch.empty(200_064, markov.RANK_DIM, dtype=torch.bfloat16),
            cache_dtype=torch.bfloat16,
            capability=(10, 3),
        )
        args.update(kw)
        return markov.unsupported_reason(**args)

    assert reason() is None
    assert reason(capability=(9, 0))
    assert reason(num_steps=9)
    assert reason(world=3)
    # the 18-bit index field
    big = markov.MAX_VOCAB + 1 + 63
    assert reason(vocab_size=big - big % 64)
    assert reason(cache_dtype=torch.float16)
    assert reason(head_weight=torch.empty(200_064, 128, dtype=torch.bfloat16))


def test_the_tiles_beyond_one_per_warp_go_to_shared_memory():
    markov = _markov()
    # M3.1's TP4 shard: 3,126 tiles on 152 CTAs of 16 warps
    assert markov.smem_tiles(tiles=200_064 // 4 // 16, ctas=152) == 5
    assert markov.smem_tiles(tiles=152 * 16, ctas=152) == 0
    # one rank: one tile per warp in shared memory too, the rest from L2
    assert markov.smem_tiles(tiles=200_064 // 16, ctas=152) == markov.MAX_SMEM_TILES
    narrow = 5 * markov.smem_tile_bytes(in_rows=8, base_dtype=torch.bfloat16)
    wide = 5 * markov.smem_tile_bytes(
        in_rows=markov.wide_in_rows(16), base_dtype=torch.bfloat16
    )
    assert (narrow, wide) == (5 * 9728, 5 * 11264)
    # and every warp's register-tile inputs beside them: 71.5 / 103 KiB, within GB300's
    # 227 KiB a block for both kernels
    for in_rows, total in ((8, 73216), (16, 105472)):
        got = markov.dynamic_smem_bytes(
            tiles_in_smem=5, in_rows=in_rows, base_dtype=torch.bfloat16
        )
        assert got == total
    cap = markov.smem_tile_cap(
        opt_in_bytes=232448, wide_rows=16, base_dtype=torch.bfloat16
    )
    assert cap == 11  # the wide kernel's budget binds


def test_the_chain_builds_with_the_packages_nvcc_helper():
    """One translation unit (markov.cu, the toolkit's headers only) per exchange, with
    its host-chosen constants as defines, cached by the helper's digest."""
    markov = _markov()
    with patch.object(nvcc, "compile_cubin", return_value=b"cubin") as compile_cubin:
        cubin = markov.compile_cubin(
            steps=7,
            base_dtype=torch.bfloat16,
            cache_dtype=torch.bfloat16,
            direct=True,
            arch="sm_103a",
            smem_tiles=5,
            wide_rows=16,
        )
    assert cubin == b"cubin"
    compile_cubin.assert_called_once_with(
        "minimax_m31_markov",
        markov.SOURCE.read_text(),
        (),
        "sm_103a",
        (
            "-std=c++17",
            "-DBASE_T=__nv_bfloat16",
            "-DCACHE_T=__nv_bfloat16",
            "-DMARKOV_DIRECT=1",
            "-DSTEPS=7",
            "-DSMEM_TILES=5",
            "-DWIDE_IN_ROWS=16",
        ),
    )
    assert (markov.NARROW_SYMBOL, markov.WIDE_SYMBOL) == (
        "m31_markov_narrow",
        "m31_markov_wide",
    )


def test_every_function_of_the_chains_source_names_where_it_runs():
    """The chain builds with the package's nvcc helper, which has no default-device
    switch: every function at file scope of markov.cu is ``__device__`` (or ``__host__
    __device__``), and its two kernels are ``extern "C" __global__``, the symbols the
    host loads."""
    source = Path(draft_step.__file__).parent / "sm103" / "draft" / "markov.cu"
    lines = source.read_text().splitlines()
    code = [
        line
        for line in lines
        if line and not line[0].isspace() and not line.startswith(("//", "#", "}"))
    ]
    declarations = ("static_assert", "constexpr", "using", "namespace")
    functions = [
        line for line in code if "(" in line and not line.startswith(declarations)
    ]
    targets = ("__device__", "__host__ __device__", 'extern "C" __global__')
    assert len(functions) > 30
    assert [line for line in functions if not line.startswith(targets)] == []
    kernels = [
        line.split("(kThreads, 1) ")[1].rstrip("(")
        for line in functions
        if line.startswith('extern "C" __global__')
    ]
    assert kernels == ["m31_markov_narrow", "m31_markov_wide"]


def test_the_parity_line_names_every_count():
    """The parity probe's counts (5) and extents (5), in order, against the bound."""
    line = sampler.parity_message(
        2, [11774, 0, 19173, 247471, 0, 0.0625, 46.25, 1.5, 2.25, 3], 2.0
    )
    for part in (
        "rank 2:",
        "blocks 11774",
        "token mismatches 0",
        "19173 at 1-8 rows and 247471 at 9+ rows",
        "first token differs 0",
        "max |difference| 0.0625",
        "max |logit| 46.2",
        "max 1.5 BF16 ulps",
        "(declared 2; 2.25 without that allowance)",
        "at most 3 BF16 values apart",
    ):
        assert part in line, (part, line)
