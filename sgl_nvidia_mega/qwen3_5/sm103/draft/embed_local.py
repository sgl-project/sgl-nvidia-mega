# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

from __future__ import annotations

import logging

import torch

logger = logging.getLogger(__name__)

__all__ = ["EmbedLocal", "build", "gather_rows", "shared_for"]

SGLANG_READS = {
    "srt/layers/vocab_parallel_embedding.py": (
        "shard_indices",
        "org_vocab_start_index",
        "org_vocab_end_index",
        "added_vocab_start_index",
        "added_vocab_end_index",
    ),
    "srt/kernel_plugins/nvidia/eagle_binding.py|srt/megakernels/eagle_binding.py": (
        "tp_rank",
        "tp_size",
        "tp_device_group",
        "tp_cpu_group",
    ),
    "srt/environ.py": ("envs", "SGLANG_ENABLE_EMBED_REPLICATION"),
}

# ids the build checks against the vocabulary-parallel lookup
CHECK_IDS = 4093

# Replicated tables by the shard they were gathered from (its storage and shape).
_SHARED: dict = {}


def _shard_key(shard: torch.Tensor) -> tuple:
    return (shard.device.index, shard.data_ptr(), tuple(shard.shape), shard.dtype)


def shared_for(shard: torch.Tensor):
    """The EmbedLocal already gathered from this shard tensor, or None."""
    return _SHARED.get(_shard_key(shard))


def gather_rows(
    table: torch.Tensor, token_ids: torch.Tensor, out: torch.Tensor | None = None
) -> torch.Tensor:
    """``table[token_ids]`` [m, hidden] bf16 with -0.0 written as +0.0, in one launch."""
    import triton

    from sgl_nvidia_mega.qwen3_5.sm103.draft.rows import _BLOCK, _copy_rows_kernel

    if table.dtype != torch.bfloat16 or table.dim() != 2 or not table.is_contiguous():
        raise ValueError(f"gather_rows: table {table.dtype} {tuple(table.shape)}")
    if token_ids.dim() != 1:
        raise ValueError(f"gather_rows: token ids {tuple(token_ids.shape)}")
    m, hidden = int(token_ids.shape[0]), int(table.shape[1])
    if out is None:
        out = torch.empty(m, hidden, dtype=torch.bfloat16, device=table.device)
    if m:
        _copy_rows_kernel[(m, triton.cdiv(hidden, _BLOCK))](
            token_ids,
            token_ids.stride(0),
            table.view(torch.int16),
            out.view(torch.int16),
            hidden,
            hidden,
            int(table.shape[0]),
            BLOCK=_BLOCK,
            PLUS_ZERO=True,
            num_warps=4,
        )
    return out


class EmbedLocal:
    @staticmethod
    def allocate(
        *, shard: torch.Tensor, vocab_local: int, world: int, device: torch.device
    ) -> torch.Tensor:
        return torch.empty(
            world * vocab_local,
            int(shard.shape[1]),
            dtype=torch.bfloat16,
            device=device,
        )

    def __init__(
        self,
        *,
        table: torch.Tensor,
        shard: torch.Tensor,
        vocab_start: int,
        vocab_local: int,
        rank: int,
        group,
        device: torch.device,
    ):
        import torch.distributed as dist

        if shard.dtype != torch.bfloat16 or shard.dim() != 2:
            raise ValueError(
                f"replicated embedding: shard {shard.dtype} {tuple(shard.shape)}"
            )
        if shard.shape[0] < vocab_local or vocab_start != rank * vocab_local:
            raise ValueError(
                f"replicated embedding: rank {rank}'s shard holds ids "
                f"[{vocab_start}, {vocab_start + vocab_local}) in {shard.shape[0]} "
                f"rows; needs rank-major shards of {vocab_local} ids"
            )
        hidden = int(shard.shape[1])
        if table.shape[1] != hidden or table.shape[0] % vocab_local:
            raise ValueError(f"replicated embedding: table {tuple(table.shape)}")
        self.hidden, self.device, self.table = hidden, device, table
        dist.all_gather_into_tensor(
            self.table, shard[:vocab_local].contiguous(), group=group
        )
        torch.cuda.synchronize(device)
        own = self.table[vocab_start : vocab_start + vocab_local]
        if not torch.equal(own, shard[:vocab_local]):
            raise RuntimeError(
                "replicated embedding: the all-gather changed this rank's rows"
            )
        self.gib = self.table.numel() * 2 / 2**30

    def __call__(self, token_ids: torch.Tensor) -> torch.Tensor:
        """The embedding rows [m, hidden] bf16 of ``token_ids`` (any integer shape)."""
        if token_ids.dim() != 1:
            token_ids = token_ids.reshape(-1)
        return gather_rows(self.table, token_ids)


def _shard_reason(embed, vocab: int, binding) -> str:
    shard = getattr(embed, "shard_indices", None)
    table = embed.weight.data
    if binding.tp_size < 2:
        return f"TP{binding.tp_size}: no vocabulary-parallel all-reduce to replace"
    if shard is None:
        return f"the embedding ({type(embed).__name__}) is not vocabulary-parallel"
    if shard.added_vocab_end_index != shard.added_vocab_start_index:
        return "the embedding has added vocabulary"
    if table.dtype != torch.bfloat16 or table.dim() != 2:
        return f"embedding table {table.dtype} {tuple(table.shape)}; needs BF16 rows"
    local = int(shard.org_vocab_end_index - shard.org_vocab_start_index)
    start = int(shard.org_vocab_start_index)
    if start != binding.tp_rank * local or local * binding.tp_size != vocab:
        return (
            f"rank {binding.tp_rank}'s embedding shard [{start}, {start + local}) is "
            f"not a rank-major slice of {vocab} ids"
        )
    return ""


def _replication_on() -> bool:
    from sglang.srt.environ import envs

    return envs.SGLANG_ENABLE_EMBED_REPLICATION.get()


def build(*, name: str, embed, vocab: int, binding, what: str):
    from sgl_nvidia_mega.qwen3_5.runtime import slot_supported

    if _replication_on():
        logger.info(
            "NVIDIA megakernel: %s steps aside: SGLang replicates the embedding "
            "(SGLANG_ENABLE_EMBED_REPLICATION)",
            name,
        )
        return None
    import torch.distributed as dist

    cpu_group = getattr(binding, "tp_cpu_group", None)
    table = embed.weight.data
    reason = _shard_reason(embed, vocab, binding)
    reused = shared_for(table) if not reason else None
    buf = None
    if not reason and reused is None:
        shard = embed.shard_indices
        local = int(shard.org_vocab_end_index - shard.org_vocab_start_index)
        try:
            buf = EmbedLocal.allocate(
                shard=table,
                vocab_local=local,
                world=binding.tp_size,
                device=table.device,
            )
        except torch.cuda.OutOfMemoryError as exc:
            reason = f"no room for the replicated table: {exc}"
    if not slot_supported(name, reason, cpu_group=cpu_group):
        del buf
        return None
    shard = embed.shard_indices
    local = int(shard.org_vocab_end_index - shard.org_vocab_start_index)
    start = int(shard.org_vocab_start_index)
    local_table = reused or EmbedLocal(
        table=buf,
        shard=table,
        vocab_start=start,
        vocab_local=local,
        rank=binding.tp_rank,
        group=binding.tp_device_group,
        device=table.device,
    )
    device = table.device
    ids = torch.linspace(0, vocab - 1, CHECK_IDS, device=device).long()
    ids[: binding.tp_size] = torch.arange(binding.tp_size, device=device) * local
    own = (ids >= start) & (ids < start + local)
    want = torch.zeros(ids.numel(), table.shape[1], dtype=torch.float32, device=device)
    want[own] = table[ids[own] - start].float()
    dist.all_reduce(want, group=binding.tp_device_group)
    got = local_table(ids)
    diff = int(
        (got.view(torch.int16) != want.bfloat16().view(torch.int16)).sum().item()
    )
    if not slot_supported(
        name,
        f"{diff} lookup values differ from SGLang's embedding" if diff else "",
        cpu_group=cpu_group,
    ):
        return None
    _SHARED[_shard_key(table)] = local_table
    logger.info(
        "NVIDIA megakernel: %s: %s gathers its embedding rows from a replicated "
        "table [%d, %d] bf16 (%.2f GiB a rank%s), no all-reduce; %d checked ids "
        "equal the vocabulary-parallel lookup bit for bit",
        name,
        what,
        local_table.table.shape[0],
        local_table.table.shape[1],
        local_table.gib,
        ", shared" if reused is not None else "",
        ids.numel(),
    )
    return local_table
