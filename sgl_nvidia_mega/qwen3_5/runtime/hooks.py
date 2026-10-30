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

from sgl_nvidia_mega.qwen3_5.sm103 import sharded_gumbel

logger = logging.getLogger(__name__)

# The SGLang names this module reads (test/unit/test_contract.py pins them).
SGLANG_READS = {
    "kernels/plugin_kernels/nvidia/spec.py": (
        "HooksFactory",
        "LocalKeysFn",
        "PostVerifyStrategy",
        "ShardedLogitsView",
        "SpeculativeRuntimeHooks",
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

    def __init__(self, descriptor: _Descriptor, default: LocalKeysFn):
        self.descriptor = descriptor
        self.default = default
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
                "NVIDIA megakernel: Qwen3.5 slot sharded_gumbel runs at %d rows",
                rows,
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


def build(model, binding) -> HooksFactory | None:
    """Build the provider factory, or return ``None`` uniformly before prewarm."""
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
    sharded_gumbel.build(
        descriptor.device_index,
        columns=descriptor.columns,
        vocab_start=descriptor.start,
        valid=descriptor.valid,
        vocab_size=descriptor.vocab_size,
        splits=descriptor.splits,
    )

    def factory(defaults: SpeculativeRuntimeHooks) -> SpeculativeRuntimeHooks:
        return SpeculativeRuntimeHooks(
            post_verify=PostVerifyStrategy(
                run=defaults.post_verify.run,
                local_keys=_LocalKeys(descriptor, defaults.post_verify.local_keys),
            )
        )

    return factory
