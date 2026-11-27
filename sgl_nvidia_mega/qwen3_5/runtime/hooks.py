# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Qwen3.5 providers for SGLang's generic speculative-runtime hooks.

The runtime owns the verifier, its fallback and every cross-rank exchange.  This
module offers only slot ``sharded_gumbel``'s local numerical policy.  Its vocabulary
descriptor is validated collectively at build, before graphs are captured; the
per-step callback is local and either declines before RNG or returns reduction keys.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import torch
import torch.distributed as dist
from sglang.kernels.plugin_kernels.nvidia.spec import (
    HooksFactory,
    LocalKeysFn,
    PostVerifyStrategy,
    ShardedLogitsView,
    SpeculativeRuntimeHooks,
)

from sgl_nvidia_mega.qwen3_5.sm103 import gathered_cdf, sharded_gumbel
from sgl_nvidia_mega.qwen3_5.sm103.head import verify_head

logger = logging.getLogger(__name__)

# The SGLang names this module reads (test/unit/test_contract.py pins them).
SGLANG_READS = {
    "kernels/plugin_kernels/nvidia/spec.py": (
        "HooksFactory",
        "LocalKeysFn",
        "PostVerifyStrategy",
        "ShardedLogitsView",
        "SpeculativeRuntimeHooks",
        "verify_shard_head",
        "draft_attention",
        # the gathered-logits sampler and its view's fields
        "gathered_tokens",
        "GatheredLogitsView",
        "values",
        "temperatures",
    ),
    "srt/kernel_plugins/nvidia/eagle_binding.py": (
        "tp_rank",
        "tp_size",
        "tp_cpu_group",
    ),
    "srt/layers/logits_processor.py": ("vocab_size",),
    "srt/layers/vocab_parallel_embedding.py": (
        "weight",
        "shard_indices",
        "org_vocab_start_index",
        "num_org_elements",
        "num_added_elements_padded",
    ),
}


@dataclass(frozen=True)
class _Descriptor:
    columns: int
    start: int
    valid: int
    vocab_size: int
    splits: int
    device_index: int


def _descriptor(model) -> tuple[_Descriptor | None, str]:
    """This rank's loaded LM-head shard, represented without any live tensor."""
    try:
        head = model.lm_head
        weight = head.weight
        shard = head.shard_indices
        columns = int(weight.shape[0])
        start = int(shard.org_vocab_start_index)
        valid = int(shard.num_org_elements)
        vocab_size = int(model.logits_processor.vocab_size)
        added = int(shard.num_added_elements_padded)
        if not weight.is_cuda or weight.dim() != 2:
            return None, "LM head weight is not a CUDA matrix"
        if added:
            return None, f"LM head has {added} padded added-vocabulary columns"
        if not 0 < valid <= columns:
            return None, f"valid columns {valid} are outside [1, {columns}]"
        if start < 0 or start + valid > vocab_size:
            return None, (
                f"vocabulary range [{start}, {start + valid}) is outside "
                f"[0, {vocab_size})"
            )
        device_index = weight.device.index
        if device_index is None:
            device_index = torch.cuda.current_device()
        return (
            _Descriptor(
                columns=columns,
                start=start,
                valid=valid,
                vocab_size=vocab_size,
                # Runtime owns the partition count and derives it from every
                # rank's uniform padded-logit width, not an uneven valid tail.
                splits=sharded_gumbel.key_splits(columns),
                device_index=int(device_index),
            ),
            "",
        )
    except (AttributeError, IndexError, TypeError, ValueError) as error:
        return None, f"LM-head shard metadata is unavailable: {error}"


def _gather_descriptors(local, binding) -> list[tuple[_Descriptor | None, str]]:
    """Gather build metadata on the CPU group; no provider callback gathers."""
    world = int(binding.tp_size)
    if world <= 0:
        return [(None, f"invalid TP size {world}")]
    gathered: list[tuple[_Descriptor | None, str] | None] = [None] * world
    if world == 1:
        gathered[0] = local
    else:
        dist.all_gather_object(gathered, local, group=binding.tp_cpu_group)
    return [
        item if item is not None else (None, f"rank {rank} returned no descriptor")
        for rank, item in enumerate(gathered)
    ]


def _topology_reason(gathered: list[tuple[_Descriptor | None, str]], binding) -> str:
    """A deterministic decision made identically from every rank's descriptors."""
    failures = [
        f"rank {rank}: {why}"
        for rank, (desc, why) in enumerate(gathered)
        if desc is None
    ]
    if failures:
        return "; ".join(failures)
    descriptors = [desc for desc, _ in gathered if desc is not None]
    rank = int(binding.tp_rank)
    if rank < 0 or rank >= len(descriptors):
        return f"TP rank {rank} is outside [0, {len(descriptors)})"
    common = {(desc.columns, desc.vocab_size, desc.splits) for desc in descriptors}
    if len(common) != 1:
        return "ranks disagree on LM-head columns, vocabulary size or key splits"
    ranges = sorted((desc.start, desc.start + desc.valid) for desc in descriptors)
    cursor = 0
    for start, end in ranges:
        if start != cursor:
            return f"vocabulary shards are not contiguous at token {cursor}"
        cursor = end
    if cursor != descriptors[0].vocab_size:
        return (
            f"vocabulary shards end at token {cursor}, expected "
            f"{descriptors[0].vocab_size}"
        )
    return ""


class _LocalKeys:
    """The graph-capturable local callback installed into the stock strategy."""

    def __init__(
        self,
        descriptor: _Descriptor,
        default: LocalKeysFn,
        *,
        fast: bool = False,
        head: verify_head.VerifyHeadMma | None = None,
    ):
        self.descriptor = descriptor
        self.default = default
        self.fast = fast
        # the verify head whose sampled keys the fast launch may take
        self.head = head if fast and head is not None and head.keys else None
        self._engaged = False

    def __call__(self, view: ShardedLogitsView) -> torch.Tensor | None:
        # Greedy is the runtime's bitwise policy, including when seed material is
        # present.  Do not inspect any provider capability before delegating it.
        if view.temperatures is None:
            return self.default(view)
        # A request seed is not a per-row Gumbel seed.  Stock seeded sampling stays
        # authoritative, and this decline precedes every capability check and RNG.
        if view.seeds is not None:
            return None
        if not self._supports(view):
            return None

        values = view.values
        rows = int(values.shape[0])
        seeds = torch.randint(
            0,
            2**31 - 1,
            (rows,),
            dtype=torch.int64,
            device=values.device,
        )
        # There is deliberately no catch or fallback after the RNG draw: doing so
        # would perturb stock sampling's stream.  A launch failure is a hard error.
        if self.fast:
            keys = sharded_gumbel.gumbel_shard_keys_fast(
                values,
                view.temperatures,
                seeds,
                vocab_start=self.descriptor.start,
                valid=self.descriptor.valid,
                vocab_size=self.descriptor.vocab_size,
                splits=view.splits,
                pre=None if self.head is None else verify_head.lookup(values),
            )
        else:
            keys = sharded_gumbel.gumbel_shard_keys(
                values,
                view.temperatures,
                seeds,
                vocab_start=self.descriptor.start,
                valid=self.descriptor.valid,
                vocab_size=self.descriptor.vocab_size,
                splits=view.splits,
            )
        if not self._engaged:
            self._engaged = True
            logger.info(
                "NVIDIA megakernel: Qwen3.5 slot sharded_gumbel runs at %d rows%s",
                rows,
                " (gumbel_fast)" if self.fast else "",
            )
        return keys

    def _supports(self, view: ShardedLogitsView) -> bool:
        values = view.values
        temperatures = view.temperatures
        descriptor = self.descriptor
        return bool(
            isinstance(values, torch.Tensor)
            and values.is_cuda
            and values.device.index == descriptor.device_index
            and values.dim() == 2
            and values.dtype in (torch.bfloat16, torch.float32)
            and values.stride(1) == 1
            and int(values.shape[1]) == descriptor.columns
            and isinstance(temperatures, torch.Tensor)
            and temperatures.device == values.device
            and temperatures.dtype == torch.float32
            and tuple(temperatures.shape) == (int(values.shape[0]),)
            and temperatures.is_contiguous()
            and view.shard.valid == descriptor.valid
            and view.shard.vocab_size == descriptor.vocab_size
            and view.splits == descriptor.splits
        )


def _fast_reason(gathered: list[tuple[_Descriptor | None, str]]) -> str:
    for rank, (desc, _) in enumerate(gathered):
        if desc is not None and not sharded_gumbel.fast_supported(
            vocab_start=desc.start, valid=desc.valid, splits=desc.splits
        ):
            return f"rank {rank}: its key tiles do not start on multiples of 4 ids"
    return ""


def _offers_verify_head() -> bool:
    """Whether this SGLang's runtime hooks take a verify-shard head."""
    return "verify_shard_head" in getattr(
        SpeculativeRuntimeHooks, "__struct_fields__", ()
    )


def _build_head(model, binding, *, keys: bool) -> verify_head.VerifyHeadMma | None:
    """Slot verify_head's head where every rank's LM head is a plain bf16 shard."""
    from sgl_nvidia_mega.qwen3_5.runtime import slot_supported

    try:
        why = verify_head.head_refusal(model.logits_processor, model.lm_head)
    except (AttributeError, TypeError, ValueError) as error:
        why = f"LM-head shard metadata is unavailable: {error}"
    if not slot_supported("verify_head", why, cpu_group=binding.tp_cpu_group):
        return None
    return verify_head.build(model, keys=keys)


class _GatheredCdf:
    def __init__(self, device_index: int, columns: int, seeds):
        self.device_index = device_index
        self.columns = columns
        self.seeds = seeds
        self._engaged = False

    def __call__(self, view) -> torch.Tensor | None:
        if not self._supports(view):
            return None
        values = view.values
        rows = int(values.shape[0])
        seeds = self.seeds.next(rows)
        tokens = gathered_cdf.cdf_tokens(values, view.temperatures, seeds)
        if not self._engaged:
            self._engaged = True
            logger.info(
                "NVIDIA megakernel: Qwen3.5 slot gathered_cdf runs at %d rows", rows
            )
        return tokens

    def _supports(self, view) -> bool:
        values = getattr(view, "values", None)
        temperatures = getattr(view, "temperatures", None)
        return bool(
            isinstance(values, torch.Tensor)
            and values.is_cuda
            and values.device.index == self.device_index
            and values.dim() == 2
            and values.dtype in (torch.bfloat16, torch.float16, torch.float32)
            and values.stride(1) == 1
            and 0 < int(values.shape[0]) <= _GATHERED_MAX_ROWS
            and int(values.shape[1]) == self.columns
            and isinstance(temperatures, torch.Tensor)
            and temperatures.device == values.device
            and temperatures.dtype == torch.float32
            and tuple(temperatures.shape) == (int(values.shape[0]),)
            and temperatures.is_contiguous()
        )


# the rows one seed round numbers (gathered_cdf.RoundSeeds)
_GATHERED_MAX_ROWS = 4096


def _offers_draft_attention() -> bool:
    """Whether this SGLang's runtime hooks take a draft attention."""
    return "draft_attention" in getattr(
        SpeculativeRuntimeHooks, "__struct_fields__", ()
    )


def _draft_attention():
    """Slot draft_cp's provider where SGLang takes one and CP-lite is built, else None."""
    from sgl_nvidia_mega.qwen3_5.attention import draft_cp, split_kv
    from sgl_nvidia_mega.qwen3_5.runtime import unavailable_slots

    if not _offers_draft_attention():
        why = "this SGLang's hooks have no draft attention"
    else:
        why = split_kv.cp_reason()
    if why:
        unavailable_slots(draft_cp.SLOT, why)
        return None
    return draft_cp.attend


def _offers_gathered_tokens() -> bool:
    """Whether this SGLang's post-verify strategy takes a gathered-logits sampler."""
    return "gathered_tokens" in getattr(PostVerifyStrategy, "__struct_fields__", ())


def _gathered_sampler(model) -> _GatheredCdf:
    """Slot gathered_cdf's sampler over the model's vocabulary, built at load."""
    device_index = torch.cuda.current_device()
    columns = int(model.logits_processor.vocab_size)
    seeds = gathered_cdf.build(
        torch.device("cuda", device_index), columns, max_rows=_GATHERED_MAX_ROWS
    )
    return _GatheredCdf(device_index, columns, seeds)


def _sharded_keys(
    model, binding
) -> tuple[_Descriptor, bool, verify_head.VerifyHeadMma | None] | None:
    from sgl_nvidia_mega.qwen3_5.runtime import SLOTS, unavailable_slots

    local = _descriptor(model)
    gathered = _gather_descriptors(local, binding)
    reason = _topology_reason(gathered, binding)
    if reason:
        logger.info(
            "NVIDIA megakernel: Qwen3.5 slot sharded_gumbel uses stock: %s", reason
        )
        return None
    descriptor = local[0]
    assert descriptor is not None
    fast = "gumbel_fast" in SLOTS
    if fast and (why := _fast_reason(gathered)):
        unavailable_slots("gumbel_fast", why)
        fast = False
    head = None
    if "verify_head" in SLOTS and not _offers_verify_head():
        unavailable_slots(
            "verify_head", "this SGLang's hooks have no verify-shard head"
        )
    elif "verify_head" in SLOTS:
        head = _build_head(model, binding, keys=fast and "verify_head_keys" in SLOTS)
    if "verify_head_keys" in SLOTS and (head is None or not head.keys):
        unavailable_slots("verify_head_keys", "gumbel_fast or verify_head uses stock")
    sharded_gumbel.build(
        descriptor.device_index,
        columns=descriptor.columns,
        vocab_start=descriptor.start,
        valid=descriptor.valid,
        vocab_size=descriptor.vocab_size,
        splits=descriptor.splits,
        fast=fast,
        pre=head is not None and head.keys,
    )
    return descriptor, fast, head


def build(
    model,
    binding,
    *,
    sharded: bool = True,
    gathered: bool = False,
    draft: bool = False,
) -> HooksFactory | None:
    attention = _draft_attention() if draft else None
    keys = _sharded_keys(model, binding) if sharded else None
    sampler = None
    if gathered and not _offers_gathered_tokens():
        logger.info(
            "NVIDIA megakernel: Qwen3.5 slot gathered_cdf uses stock: this SGLang has "
            "no gathered-logits sampler policy"
        )
    elif gathered:
        sampler = _gathered_sampler(model)
    if keys is None and sampler is None and attention is None:
        return None
    descriptor, fast, head = keys if keys is not None else (None, False, None)
    # each field is set only when its provider is built
    extra = {} if head is None else {"verify_shard_head": head}
    if attention is not None:
        extra["draft_attention"] = attention

    def factory(defaults: SpeculativeRuntimeHooks) -> SpeculativeRuntimeHooks:
        local_keys = defaults.post_verify.local_keys
        if descriptor is not None:
            local_keys = _LocalKeys(descriptor, local_keys, fast=fast, head=head)
        post = {} if sampler is None else {"gathered_tokens": sampler}
        return SpeculativeRuntimeHooks(
            post_verify=PostVerifyStrategy(
                run=defaults.post_verify.run, local_keys=local_keys, **post
            ),
            **extra,
        )

    return factory
