# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""MiniMax M3.1's DSpark draft proposal: the draft LM head and the Markov chain of every
draft step in one vocab-sharded kernel per rank (slot ``markov``).

Two phases. The draft spec's build (``prebuild``, from ``draft_step.DraftStep`` when
SGLang installs the draft model, before it sizes its KV pools) checks the Markov head and
allocates the chain's buffers (``sm103/draft/markov.py``: the symmetric-memory peer
buffer, the noise). SGLang's folded proposal (``DsparkDraftSampler``, the draft CUDA
graph's tail) then asks the draft spec for its ``draft_sampler`` once, on every rank,
with its own buffers; ``build_steps`` checks the LM head SGLang has attached by then,
compiles and warms on every rank what a step launches before the chain's
first cross-rank wait, and returns the step for each batch size the chain serves, which
replaces ``compute_base_logits`` + ``draft_markov_block`` there. A step that fails to
build on one rank raises on every rank, with each rank's error; nothing compiles inside
a step. The step projects the draft hidden states through this
rank's LM-head shard (SGLang's projection, ``project_through_lm_head``, so the base
logits are SGLang's bit for bit, under ``--speculative-dspark-lm-head-dtype`` too), then
runs the chain: ranks agree on each
step's token inside the kernel, so the proposal needs no per-step exchange launches. The
draft noise is SGLang's (same Philox key, counters and offset), drawn on a side stream
while the LM head runs. The steps serve 1 to ``MAX_REQUESTS`` requests, whatever the
draft's forward ran on (the draft backbone at one request, SGLang's draft above).

Numerics (``runtime.DRAFT_TOLERANCES``): the chain's step bias comes from tensor-core
MMAs whose FP32 summation order differs from the bias GEMM SGLang launches, so a BF16
bias can differ by one ulp; a kept step logit then differs by at most two BF16 ulps of
|kept logit| + |base logit| past one ulp of the step's largest |bias| (``bound_ulps``),
and a draft token can differ where two candidates are that close. The kept logits are
the draft probabilities SGLang's sampling accept reads, so a seeded sampled request is
not token-identical to SGLang's proposal (the target's distribution is: the chain
samples from the logits it keeps).
"""

from __future__ import annotations

import logging
import os
import threading
from typing import Optional

import torch

from sgl_nvidia_mega.minimax_m31 import HF_CONFIG, MAX_REQUESTS

logger = logging.getLogger(__name__)

# The SGLang names this module reads, by file under the sglang package (``f(p)``: a
# parameter); test/unit/test_contract.py pins them.
SGLANG_READS = {
    "srt/runtime_context.py": ("get_parallel", '"tp_group"'),
    "srt/distributed/parallel_state.py": (
        "rank_in_group",
        "world_size",
        "cpu_group",
        "device_group",
    ),
    "srt/models/dspark.py": (
        "VanillaMarkov",
        "project_through_lm_head",
        "markov_head",
        "markov_w1",
        "markov_w2",
        "vocab_size",
        "step_logits_dtype",
        "lm_head",
        "logits_mup_width_multiplier",
        "compute_base_logits",
    ),
    "srt/layers/vocab_parallel_embedding.py": (
        "shard_indices",
        "org_vocab_start_index",
        "num_embeddings_per_partition",
        "org_vocab_size",
        "weight",
    ),
    "srt/speculative/dspark_components/dspark_draft.py": (
        "key",
        "offset",
        "DraftNoise",
        "draft_markov_block",
        "draft_markov_block(markov_head)",
        "draft_markov_block(base_logits)",
        "draft_markov_block(anchor_tokens)",
        "draft_markov_block(draft_hidden)",
        "draft_markov_block(step_temperatures)",
        "draft_markov_block(noise)",
        "draft_markov_block(tp_sync)",
        "draft_markov_block(logits_out)",
        "draft_markov_block(greedy_draft)",
    ),
}

# The SGLang code a step stands in for, by the digest of its code
# (test/unit/test_contract.py recomputes them on the installed SGLang): the hook's call
# with its buffers; the stock block the step replaces (base logits, then the Markov
# block, its per-step bias, sample and token sync, and the noise counter's advance); and
# the partial kernel whose Philox draw and selection keys the chain reproduces
SGLANG_MIRRORS = {
    "srt/kernel_plugins/nvidia/dspark_binding.py:bind_dspark_draft_strategy": (
        "c4c795cadf9d1e69"
    ),
    "srt/speculative/dspark_components/dspark_draft_sampler.py:"
    "DsparkDraftSampler._markov_block": "043aea5cbc7f483a",
    "srt/models/dspark.py:DSparkDraftMixin.compute_base_logits": "151cd4b4e14a868b",
    "srt/speculative/dspark_components/dspark_draft.py:draft_markov_block": (
        "c769d1e609a182fe"
    ),
    "srt/speculative/dspark_components/dspark_draft.py:_draft_step": "e1916fd04604c39a",
    "kernels/ops/speculative/dspark/markov_sample.py:_markov_sample_partial_kernel": (
        "d5819695de7861c3"
    ),
}

# Test only: inside each captured proposal, also run SGLang's own base logits + Markov
# block on the same inputs into scratch buffers and count the tokens and kept logits
# that differ, with the kept logits' differences in BF16 ulps; every rank logs the
# counts every 30 s
PARITY = os.environ.get("SGLANG_NVIDIA_MEGAKERNEL_MARKOV_PARITY", "0") == "1"

# The chain serves at most MAX_REQUESTS requests (the megakernel's coverage); wider draft
# steps keep SGLang's proposal.


def head_reason(lm, *, gamma: int, world: int) -> str:
    """Why the chain cannot serve this draft's Markov head, or ""; the same on every
    rank (checked at the draft spec's build, before the LM head is attached)."""
    from sglang.srt.models.dspark import VanillaMarkov

    from sgl_nvidia_mega.minimax_m31.sm103.draft import markov

    head = lm.markov_head
    if head is None:
        return "the draft has no Markov head"
    if type(head) is not VanillaMarkov:
        # the chain computes VanillaMarkov's bias: markov_w2 . markov_w1[previous]
        return f"a {type(head).__name__} Markov head (needs VanillaMarkov)"
    if head.markov_w2.weight.dtype != torch.bfloat16:
        # SGLang's FP32 bias path rounds differently from the kernel's BF16 table
        return "an FP32 markov_w2 (set SGLANG_DSPARK_OPT_MARKOV_W2_BF16=1)"
    return (
        markov.unsupported_reason(
            world=world,
            vocab_size=int(head.vocab_size),
            num_steps=gamma,
            embed_weight=head.markov_w1.weight,
            head_weight=head.markov_w2.weight,
            cache_dtype=head.step_logits_dtype(torch.bfloat16),
            capability=torch.cuda.get_device_capability(),
        )
        or ""
    )


def lm_head_reason(lm, *, world: int, rank: int) -> str:
    """Why the chain cannot read this draft's base logits, or ""; the same on every rank
    (checked when the folded proposal asks for the sampler, the LM head attached)."""

    head, lm_head = lm.markov_head, lm.lm_head
    if lm_head is None:
        return "the draft has no LM head attached"
    if head is None:
        return ""  # head_reason names it
    if lm.logits_mup_width_multiplier:
        return "a muP-scaled draft LM head (logits_mup_width_multiplier)"
    if lm_head.weight.dtype != torch.bfloat16:
        return f"a {lm_head.weight.dtype} draft LM head (needs BF16)"
    vocab = int(head.vocab_size)
    if int(lm_head.org_vocab_size) != vocab:
        return f"an LM head of {int(lm_head.org_vocab_size)} tokens for a {vocab}-token Markov head"
    # the chain reads this rank's base logits, the LM head's even vocab split
    shard = lm_head.shard_indices
    if int(lm_head.num_embeddings_per_partition) != vocab // world or int(
        shard.org_vocab_start_index
    ) != rank * (vocab // world):
        return "an LM-head vocab partition that is not the even split"
    want = (vocab // world, HF_CONFIG["hidden_size"])
    if tuple(lm_head.weight.shape) != want:
        # the shape the build sized the LM-head copy for
        return f"an LM-head shard of shape {tuple(lm_head.weight.shape)}, not {want}"
    return ""


def buffer_reason(
    *,
    max_bs: int,
    gamma: int,
    draft_tokens,
    step_temperatures,
    step_logits=None,
    kept=None,
) -> str:
    """Why SGLang's buffers are not the ones the chain writes into, or "". ``kept``: the
    (vocabulary, dtype) of the step logits a sampling draft keeps, None for a greedy
    draft, which keeps none."""
    rows, tokens, temperatures = max_bs, draft_tokens, step_temperatures
    if tokens.dtype != torch.int64 or tuple(tokens.shape) != (rows * gamma,):
        return f"draft tokens {tokens.dtype} {tuple(tokens.shape)}, not int64 [{rows * gamma}]"
    if temperatures.dtype != torch.float32 or tuple(temperatures.shape) != (rows,):
        return (
            f"step temperatures {temperatures.dtype} {tuple(temperatures.shape)}, "
            f"not float32 [{rows}]"
        )
    if kept is None:
        return ""
    vocab, dtype = kept
    if step_logits is None:
        return "a sampling draft without a step-logits buffer"
    shape = tuple(step_logits.shape)
    if (
        step_logits.dtype != dtype
        or len(shape) != 3
        or shape[:2] != (rows, gamma)
        or shape[2] < vocab
    ):
        # SGLang's sharded accept keeps one shard a rank; the chain stages every rank's
        return (
            f"step logits {step_logits.dtype} {shape}, not {dtype} "
            f"[{rows}, {gamma}, {vocab}] (the whole vocabulary on every rank)"
        )
    return ""


class DraftLmHead:
    """This rank's LM-head shard for the draft's rows: SGLang's projection
    (``project_through_lm_head``) of the LM head SGLang attaches to the draft after the
    draft spec's build."""

    def __init__(self, *, rows: int, cols: int, device):
        self.rows, self.cols = rows, cols
        self.lm_head = None
        self.kind = "SGLang's BF16"

    def attach(self, lm_head) -> None:
        """Read ``lm_head`` from here on (SGLang's projection)."""
        self.lm_head = lm_head

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        from sglang.srt.models.dspark import project_through_lm_head

        return project_through_lm_head(x, self.lm_head)


class Prebuilt:
    """What the draft spec's build allocates before SGLang sizes its KV pools: the chain
    (sm103/draft/markov.py ShardedMarkov, for up to MAX_REQUESTS rows) and the LM head's
    storage, with the draft settings they were built for."""

    def __init__(
        self,
        *,
        chain,
        lm_head,
        gamma,
        greedy_draft,
        world,
        rank,
        cpu_group,
        slots,
    ):
        self.chain, self.lm_head = chain, lm_head
        self.gamma, self.greedy_draft = gamma, greedy_draft
        self.world, self.rank, self.cpu_group = world, rank, cpu_group
        self.slots = frozenset(slots)

    def device_bytes(self) -> dict[str, int]:
        return self.chain.device_bytes()


def _memory(parts: dict[str, int]) -> str:
    total = sum(parts.values()) / 2**20
    detail = ", ".join(f"{name} {size / 2**20:.1f}" for name, size in parts.items())
    return f"{total:.1f} MiB of device memory ({detail} MiB)"


def prebuild(
    lm,
    *,
    deadline_s: float,
    slots,
    gamma: int,
    greedy_draft: bool,
) -> Prebuilt:
    """The chain and the LM head's storage, for the draft spec's build; collective over
    the TP group, so every rank runs it. Raises on every rank when any rank's Markov
    head cannot run the chain."""
    from sglang.srt.runtime_context import get_parallel

    from sgl_nvidia_mega.minimax_m31.sm103.draft import markov

    group = get_parallel().tp_group
    world, rank = group.world_size, group.rank_in_group
    # one rank: plain device memory and no collectives
    cpu_group = None if world == 1 else group.cpu_group
    refusal = refusal_message(
        _gather(cpu_group, head_reason(lm, gamma=gamma, world=world))
    )
    if refusal:
        raise ValueError(refusal)
    head = lm.markov_head
    chain = markov.ShardedMarkov(
        embed_weight=head.markov_w1.weight,
        head_weight=head.markov_w2.weight,
        vocab_size=int(head.vocab_size),
        num_steps=gamma,
        cache_dtype=head.step_logits_dtype(torch.bfloat16),
        group=cpu_group,
        # the peer buffer meets where the draft backbone's arena does
        symm_group=None if world == 1 else group.device_group,
        rank=rank,
        world=world,
        # every batch size the chain serves: SGLang's widest draft graph may be wider
        max_bs=MAX_REQUESTS,
        # SGLang's BuildContext.deadline_s bounds every cross-rank wait
        deadline_s=deadline_s,
        keeps_logits=not greedy_draft,
    )
    lm_head = _on_every_rank(
        cpu_group,
        lambda: DraftLmHead(
            rows=int(head.vocab_size) // world,
            cols=HF_CONFIG["hidden_size"],
            device=head.markov_w1.weight.device,
        ),
        "allocating the draft LM head",
    )
    prebuilt = Prebuilt(
        chain=chain,
        lm_head=lm_head,
        gamma=gamma,
        greedy_draft=greedy_draft,
        world=world,
        rank=rank,
        cpu_group=cpu_group,
        slots=slots,
    )
    logger.info(
        "NVIDIA megakernel: draft proposal buffers on rank %d, before SGLang sizes "
        "its KV pools: %s",
        rank,
        _memory(prebuilt.device_bytes()),
    )
    return prebuilt


def _on_every_rank(cpu_group, step, what: str):
    """``step`` on this rank; over a TP group every rank runs it, and if it raised on
    any rank every rank raises with each rank's error (``sm103/host.py`` ``agree``), so
    no rank waits in a collective the others left."""
    if cpu_group is None:
        return step()
    from sgl_nvidia_mega.minimax_m31.sm103 import host

    return host.agree(cpu_group, step, what)


def _warm(steps: dict, prebuilt: Prebuilt, *, noise, step_temperatures) -> None:
    """Compile what a step launches on this rank before the chain's first cross-rank
    wait, on SGLang's own buffers: the noise kernel (Triton) and the LM head at every
    batch size. The caller gathers every rank's
    result, so no rank enters a chain launch while another still compiles (a JIT stall
    longer than the wait deadline would fault its peers)."""
    chain, lm_head = prebuilt.chain, prebuilt.lm_head
    device = chain.device
    # reads the counter, writes the chain's own noise buffer; SGLang's counter moves not
    chain.launch_noise(
        rows=1, key=noise.key, offset=noise.offset, temperatures=step_temperatures
    )
    for bs in steps:
        lm_head(
            torch.zeros(
                bs * prebuilt.gamma, lm_head.cols, dtype=torch.bfloat16, device=device
            )
        )
    torch.cuda.synchronize(device)


def build_steps(
    lm,
    *,
    prebuilt: Prebuilt,
    gamma: int,
    max_bs: int,
    greedy_draft: bool,
    noise,
    draft_tokens: torch.Tensor,
    step_logits: Optional[torch.Tensor],
    step_temperatures: torch.Tensor,
) -> dict:
    """The per-batch-size steps over SGLang's buffers and what the draft spec's build
    allocated (``prebuilt``); collective over the TP group, so every rank runs it.
    Raises on every rank when any rank cannot run the chain, or when the ranks would
    serve different sizes (their graphs would wait on each other)."""
    chain, lm_head = prebuilt.chain, prebuilt.lm_head
    world, rank, cpu_group = prebuilt.world, prebuilt.rank, prebuilt.cpu_group
    reason = ""
    if (prebuilt.gamma, prebuilt.greedy_draft) != (gamma, greedy_draft):
        mode = {True: "greedy", False: "sampling"}
        reason = (
            f"the draft spec built the chain for {prebuilt.gamma} draft tokens and a "
            f"{mode[prebuilt.greedy_draft]} draft; the proposal has {gamma} and a "
            f"{mode[greedy_draft]} draft"
        )
    reasons = _gather(
        cpu_group,
        reason
        or lm_head_reason(lm, world=world, rank=rank)
        or buffer_reason(
            max_bs=max_bs,
            gamma=gamma,
            draft_tokens=draft_tokens,
            step_temperatures=step_temperatures,
            step_logits=step_logits,
            kept=None if greedy_draft else (chain.vocab_size, chain.cache_dtype),
        ),
    )
    refusal = refusal_message(reasons)
    if refusal:
        raise ValueError(refusal)
    rows = min(chain.max_rows, max_bs)
    mismatch = mismatch_message(_gather(cpu_group, (rows, lm_head.kind)))
    if mismatch:
        raise RuntimeError(mismatch)
    steps = {
        bs: _Step(
            lm=lm,
            chain=chain,
            lm_head=lm_head,
            bs=bs,
            gamma=gamma,
            greedy_draft=greedy_draft,
            noise=noise,
            draft_tokens=draft_tokens,
            step_logits=step_logits,
            step_temperatures=step_temperatures,
        )
        for bs in range(1, rows + 1)
    }

    def warm() -> None:
        lm_head.attach(lm.lm_head)
        _warm(steps, prebuilt, noise=noise, step_temperatures=step_temperatures)

    # every rank compiles, or every rank raises with each
    # rank's error; the gather is the barrier: none goes on to capture a chain launch
    # while another still compiles
    _on_every_rank(cpu_group, warm, "warming the draft proposal")
    from sgl_nvidia_mega.minimax_m31.runtime import draft_slots

    logger.info(
        "NVIDIA megakernel: draft proposal on rank %d: Markov chain in one kernel "
        "(%d ranks, %d CTAs, %d head tiles a CTA in shared memory, batch 1-%d, %d "
        "draft tokens), LM head %s; tolerance numerics (draft slots %s); wait deadline "
        "%.0f s; %s, allocated before SGLang sizes its KV pools; compiled and warmed on "
        "every rank",
        rank,
        world,
        chain.ctas,
        chain.smem_tiles,
        rows,
        gamma,
        lm_head.kind,
        sorted(draft_slots(prebuilt.slots)),
        chain.timeout_ns / 1e9,
        _memory(prebuilt.device_bytes()),
    )
    return steps


class _Step:
    """One batch size's proposal: LM head, then the chain."""

    def __init__(
        self,
        *,
        lm,
        chain,
        lm_head,
        bs,
        gamma,
        greedy_draft,
        noise,
        draft_tokens,
        step_logits,
        step_temperatures,
    ):
        self.lm, self.chain, self.lm_head = lm, chain, lm_head
        self.bs, self.gamma, self.greedy_draft, self.noise = (
            bs,
            gamma,
            greedy_draft,
            noise,
        )
        self.tokens = draft_tokens[: bs * gamma].view(bs, gamma)
        self.step_logits = step_logits
        self.temperatures = step_temperatures
        self.is_logged = False

    def __call__(self, hidden_states: torch.Tensor, anchor_tokens: torch.Tensor):
        if not self.is_logged and torch.cuda.is_current_stream_capturing():
            # engagement, through the model's dispatch, once per captured graph
            logger.info(
                "NVIDIA megakernel: draft proposal engaged: batch %d, %d LM-head "
                "rows (%s)",
                self.bs,
                hidden_states.shape[0],
                self.lm_head.kind,
            )
            self.is_logged = True
        if not self.greedy_draft:
            self.chain.fork_noise(
                rows=self.bs,
                key=self.noise.key,
                offset=self.noise.offset,
                temperatures=self.temperatures,
            )
        # as compute_base_logits on a TP-sharded vocab: this rank's slice, no gather
        base_logits = self.lm_head(hidden_states).view(self.bs, self.gamma, -1)
        if not self.greedy_draft:
            self.chain.join_noise()
        self.chain.sample(
            base_logits=base_logits,
            anchor_tokens=anchor_tokens,
            temperatures=self.temperatures,
            logits_cache=self.step_logits,
            draft_tokens=self.tokens,
            # one block advances the counter once, as draft_markov_block does: at the
            # chain kernel's end, not in a launch of its own
            advance_offset=None if self.greedy_draft else self.noise.offset,
        )
        if PARITY:
            _parity(self.lm).check(self, hidden_states, anchor_tokens)
        # the confidence head's features, as compute_base_logits returns them
        return self.tokens, hidden_states


def _bf16_ordinal(x: torch.Tensor) -> torch.Tensor:
    """BF16 values as int32 ordinals: neighbouring values are 1 apart (+0 and -0 alike)."""
    bits = x.view(torch.int16).to(torch.int32)
    return torch.where(bits < 0, -(bits & 0x7FFF), bits)


def bf16_ulp(magnitude: torch.Tensor) -> torch.Tensor:
    """One BF16 ulp at ``magnitude`` (>= 0, FP32): 2^(e - 8) for magnitude = m 2^e,
    m in [0.5, 1) (8 significant bits)."""
    _, exponent = torch.frexp(magnitude)
    return torch.ldexp(torch.ones_like(magnitude), exponent - 8)


def bound_ulps(
    delta: torch.Tensor, reference: torch.Tensor, base: torch.Tensor
) -> torch.Tensor:
    """The measure runtime.DRAFT_TOLERANCES bounds, per element: the kept logit's
    |difference| ``delta`` past one BF16 ulp of the step's largest |bias|, in BF16 ulps
    of |kept logit| + |base logit|. ``reference`` is SGLang's kept logits and ``base``
    the base logits, FP32 of one shape [..., vocab]; the step's |bias| is SGLang's kept
    logit less the base, to its rounding."""
    floor = bf16_ulp((reference - base).abs().amax(dim=-1, keepdim=True))
    return (delta - floor).clamp(min=0) / bf16_ulp(reference.abs() + base.abs())


class _Parity:
    """Test only (``PARITY``): SGLang's proposal beside the plugin's, counted on the GPU."""

    def __init__(self, lm, rank: int, world: int):
        self.lm, self.rank = lm, rank
        device = lm.lm_head.weight.device
        vocab_local = int(lm.markov_head.vocab_size) // world
        # this rank's columns of the full-vocab kept logits (its base logits' shard)
        self.shard = slice(rank * vocab_local, (rank + 1) * vocab_local)
        # [blocks, token mismatches, kept-logit mismatches at 1-8 rows, at 9+ rows,
        #  blocks whose first draft token differs]
        self.counts = torch.zeros(5, dtype=torch.int64, device=device)
        # [max |kept-logit difference|, max |kept logit|, on this rank's shard the
        #  declared bound's measure (bound_ulps, runtime.DRAFT_TOLERANCES) and the same
        #  without its allowance of one ulp of the step's largest |bias|, the most
        #  BF16 values two kept logits lie apart]
        self.extent = torch.zeros(5, dtype=torch.float32, device=device)
        self.offset = torch.zeros(1, dtype=torch.int64, device=device)
        self.logits = None
        from sgl_nvidia_mega.minimax_m31.runtime import DRAFT_TOLERANCES

        self.bound = DRAFT_TOLERANCES["markov"]["draft_markov.step_logits.bf16_ulps"]
        # the reporter reads host memory only: no device sync while graphs capture
        self.host_counts = torch.zeros(5, dtype=torch.int64).pin_memory()
        self.host_extent = torch.zeros(5, dtype=torch.float32).pin_memory()
        threading.Thread(target=self._report, daemon=True).start()

    def check(self, step: _Step, hidden_states, anchor_tokens) -> None:
        from sglang.srt.speculative.dspark_components.dspark_draft import (
            DraftNoise,
            draft_markov_block,
        )

        bs, gamma = step.bs, step.gamma
        # the plugin advanced the counter after drawing at offset - 1
        self.offset.copy_(step.noise.offset - (0 if step.greedy_draft else 1))
        base, _ = self.lm.compute_base_logits(hidden_states)
        logits_out = None
        if not step.greedy_draft:
            if self.logits is None or self.logits.shape[0] < bs:
                assert (
                    not torch.cuda.is_current_stream_capturing()
                    or self.logits is not None
                )
                self.logits = torch.empty_like(step.step_logits)
            logits_out = self.logits[:bs]
        tokens, _ = draft_markov_block(
            markov_head=self.lm.markov_head,
            base_logits=base.view(bs, gamma, -1),
            anchor_tokens=anchor_tokens,
            draft_hidden=hidden_states.view(bs, gamma, -1),
            step_temperatures=step.temperatures[:bs],
            noise=DraftNoise(key=step.noise.key, offset=self.offset),
            tp_sync=_NoSync(),
            logits_out=logits_out,
            greedy_draft=step.greedy_draft,
        )
        self.counts[0].add_(1)
        self.counts[1].add_((tokens != step.tokens).sum())
        self.counts[4].add_((tokens[:, 0] != step.tokens[:, 0]).sum())
        if logits_out is not None:
            # temperature-0 rows keep no logits on the chain
            kept = (step.temperatures[:bs] > 0).view(bs, 1, 1)
            got = step.step_logits[:bs]
            differ = (logits_out != got) & kept
            self.counts[2 if bs <= 8 else 3].add_(differ.sum())
            delta = (logits_out.float() - got.float()).abs() * kept
            reference = logits_out.float().abs().nan_to_num(posinf=0.0) * kept
            torch.maximum(
                self.extent[:1], delta.nan_to_num(0.0).amax(), out=self.extent[:1]
            )
            torch.maximum(self.extent[1:2], reference.amax(), out=self.extent[1:2])
            # the declared bound's measure on this rank's columns (bound_ulps)
            local = base.view(bs, gamma, -1)
            if local.shape[-1] != logits_out.shape[-1]:
                ref_local = logits_out[:, :, self.shard].float()
                delta_local = delta[:, :, self.shard]
            else:  # full-vocab base logits
                ref_local, delta_local = logits_out.float(), delta
            ulps = bound_ulps(delta_local, ref_local, local.float())
            torch.maximum(
                self.extent[2:3], ulps.nan_to_num(0.0).amax(), out=self.extent[2:3]
            )
            plain = delta_local / bf16_ulp(ref_local.abs() + local.float().abs())
            torch.maximum(
                self.extent[3:4], plain.nan_to_num(0.0).amax(), out=self.extent[3:4]
            )
            if logits_out.dtype == torch.bfloat16:
                apart = (_bf16_ordinal(logits_out) - _bf16_ordinal(got)).abs() * kept
                torch.maximum(
                    self.extent[4:5], apart.amax().float(), out=self.extent[4:5]
                )
        self.host_counts.copy_(self.counts, non_blocking=True)
        self.host_extent.copy_(self.extent, non_blocking=True)

    def _report(self) -> None:
        import time

        last = None
        while True:
            time.sleep(30)
            counts = self.host_counts.tolist() + self.host_extent.tolist()
            if counts != last:
                logger.info(parity_message(self.rank, counts, self.bound))
                last = counts


def parity_message(rank: int, counts: list, bound: float) -> str:
    """The parity probe's line from its counts (_Parity.counts, then .extent)."""
    blocks, tokens, narrow, wide, first, delta, logit, ulps, plain, apart = counts
    return (
        f"NVIDIA megakernel: Markov parity rank {rank}: blocks {int(blocks)}, token "
        f"mismatches {int(tokens)}, kept-logit mismatches {int(narrow)} at 1-8 rows and "
        f"{int(wide)} at 9+ rows, blocks whose first token differs {int(first)}, max "
        f"|difference| {delta:.3g}, max |logit| {logit:.3g}; on this rank's shard max "
        f"{ulps:.4g} BF16 ulps of |logit| + |base| past one ulp of the step's largest "
        f"|bias| (declared {bound:g}; {plain:.4g} without that allowance); kept logits "
        f"at most {int(apart)} BF16 values apart"
    )


class _NoSync:
    def sync(self, site, values):
        return values


_PARITIES: dict = {}


def _parity(lm) -> _Parity:
    if id(lm) not in _PARITIES:
        from sglang.srt.runtime_context import get_parallel

        group = get_parallel().tp_group
        _PARITIES[id(lm)] = _Parity(lm, group.rank_in_group, group.world_size)
    return _PARITIES[id(lm)]


def refusal_message(reasons: list[str]) -> str:
    """Every rank's reason the ``markov`` slot cannot run, or "" when none has one."""
    found = [f"rank {rank}: {reason}" for rank, reason in enumerate(reasons) if reason]
    if not found:
        return ""
    return "NVIDIA megakernel: the markov slot cannot run: " + "; ".join(found)


def mismatch_message(built: list[tuple[int, str]]) -> str:
    """ "" when every rank built the same (batch sizes, LM head), else the split."""
    if len(set(built)) <= 1:
        return ""
    return "NVIDIA megakernel: the ranks built different draft proposals: " + "; ".join(
        f"rank {rank}: batch 1-{rows}, LM head {kind}"
        for rank, (rows, kind) in enumerate(built)
    )


def _gather(cpu_group, value) -> list:
    if cpu_group is None:
        return [value]
    import torch.distributed as dist

    values = [None] * dist.get_world_size(cpu_group)
    dist.all_gather_object(values, value, group=cpu_group)
    return values
