# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""CPU checks of the MiniMax-M3.1 DSpark draft's slot markov: the numerics the draft spec declares with them; the target's
summary and the engagement check its executor runs; the draft spec's built object,
which offers SGLang's folded proposal the draft sampler only with slot markov and
builds its buffers at the spec's build; the sampler's refusals of LM heads, buffers
and proposals it does not fit; and the chain's host code. (The slot list, its refusals
at argument resolution and the toolchains: test_minimax_m31_runtime.py and
test_minimax_m31_spec.py; the Markov heads the chain refuses and the engagement check
against SGLang's graph runner: test_minimax_m31_draft_sampler.py.)"""

import logging
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
from sgl_nvidia_mega.minimax_m31 import runtime as m31_runtime
from sgl_nvidia_mega.minimax_m31 import sampler
from sgl_nvidia_mega.minimax_m31 import spec as m31_spec
from sgl_nvidia_mega.minimax_m31.moe import moe_nofill as mn
from sgl_nvidia_mega.minimax_m31.moe import moe_swap
from sgl_nvidia_mega.minimax_m31.runtime import agree as m31_agree
from sgl_nvidia_mega.minimax_m31.runtime import executor as m31_executor
from sgl_nvidia_mega.minimax_m31.sm103 import nvcc

TARGET_SLOTS = frozenset({"mega", "moe_nofill", "moe_swap"})
VERIFY_WIDTHS = tuple(range(8, 129, 8))


def test_the_draft_slots_state_their_tolerances_in_the_draft_spec():
    """The draft spec's numerics: its backbone's tolerances and the requested draft
    slots'."""
    backbone = draft.BACKBONE_TOLERANCES
    assert dict(draft._numerics(TARGET_SLOTS).tolerances) == backbone
    markov = dict(draft._numerics(TARGET_SLOTS | {"markov"}).tolerances)
    assert markov == {**backbone, **m31_runtime.DRAFT_TOLERANCES["markov"]}
    assert draft._numerics(TARGET_SLOTS | {"markov"}).kind is NumericsKind.TOLERANCE
    assert set(m31_runtime.DRAFT_TOLERANCES) == set(m31_runtime.DRAFT_SLOTS)


# ------------------------------------------------------------------ the target's build


def _build(slots, off=""):
    """``runtime.build`` on the verify widths with ``slots`` requested and every slot's
    build serving them; ``off``: why the server has no folded proposal."""
    served = mn.Bound(widths=VERIFY_WIDTHS)
    swap = moe_swap.Bound(widths=VERIFY_WIDTHS)
    ctx = SimpleNamespace(widths=VERIFY_WIDTHS)
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
    built = _build({"moe_nofill", "moe_swap", "markov"})
    assert built.summary().endswith(
        "slots by width 8-128: moe_nofill, moe_swap; draft slots ['markov'], built "
        "by the DSpark draft's spec"
    )
    assert isinstance(built._draft_check, draft.EngagementCheck)
    assert built._draft_check.slots == ["markov"]
    # the target itself offers no sampler
    assert built.draft_sampler is None
    stock = _build({"moe_nofill", "moe_swap"})
    assert "draft slots" not in stock.summary()
    assert stock._draft_check is None
    # an eager test server has no folded proposal for the slots to replace
    off = "decode CUDA graphs are disabled"
    eager = _build({"moe_nofill", "moe_swap", "markov"}, off=off)
    assert f"draft slots ['markov'] not engaged: {off}" in eager.summary()
    assert eager._draft_check is None


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
OFF = "decode CUDA graphs are disabled"


@pytest.mark.parametrize(
    "slots, off, proposal, logged",
    [
        (
            TARGET_SLOTS,
            "",
            "SGLang's proposal (slot markov off)",
            "rank 2: SGLang's (slot markov is not requested)",
        ),
        # disabled decode CUDA graphs (the parity switch's eager steps): SGLang folds
        # no proposal, so the slots build nothing
        (
            TARGET_SLOTS | {"markov"},
            OFF,
            f"SGLang's proposal (draft slots ['markov'] not engaged: {OFF})",
            f"['markov'] are not engaged on rank 2: {OFF}",
        ),
    ],
)
def test_without_an_engaged_markov_the_draft_keeps_sglangs_proposal(
    caplog, slots, off, proposal, logged
):
    built = _draft_step(slots)
    with (
        patch.object(draft, "proposal_off_reason", return_value=off),
        patch.object(sampler, "prebuild") as prebuild,
        caplog.at_level(logging.INFO, logger=draft_step.__name__),
    ):
        built.proposal = built._build_proposal(CTX, rank=2)
    prebuild.assert_not_called()
    assert built.draft_sampler is None and built.prebuilt is None
    assert built.proposal == proposal
    assert built.summary().endswith(f"proposal: {proposal}")
    (line,) = caplog.messages  # every rank says which proposal serves
    assert logged in line


@pytest.mark.parametrize(
    "slots, lm_head",
    [
        ({"markov"}, "SGLang's LM head (draft slots ['markov'])"),
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
    # with every warp's register-tile inputs beside them: 71.5 / 103 KiB, within
    # GB300's 227 KiB a block for both kernels
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
