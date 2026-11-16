# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""K31 verify ABI: immutable tensor/numeric records, no host execution objects."""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class KdaVerifyWeights:
    conv_weight: torch.Tensor
    conv_bias: torch.Tensor
    f_b: torch.Tensor
    a_log: torch.Tensor
    dt_bias: torch.Tensor
    norm_weight: torch.Tensor
    has_bias: int
    lower_bound: float
    scale: float
    norm_eps: float


@dataclass(frozen=True)
class KdaVerifyPools:
    temporal: torch.Tensor
    conv: torch.Tensor
    rawv: torch.Tensor
    rawk: torch.Tensor
    log_g: torch.Tensor
    beta: torch.Tensor
    conv_windows: torch.Tensor


@dataclass(frozen=True)
class KdaVerifyStep:
    state_slots: torch.Tensor
    scratch_rows: torch.Tensor
    valid_requests: torch.Tensor
    batch_requests: int
    query_width: int
    row_count: int
    abi_version: int = 1

    def validate(self):
        b, q, r = self.batch_requests, self.query_width, self.row_count
        if self.abi_version != 1 or b < 1 or q < 1 or r != b * q:
            raise ValueError("KDA verify request/query axes disagree")
        for tensor in (self.state_slots, self.scratch_rows, self.valid_requests):
            if tensor.shape != (b,) or tensor.dtype != torch.int32:
                raise ValueError("KDA verify metadata must be int32[Breq]")


@dataclass(frozen=True)
class VerifyLayerPools:
    layer_id: int
    # Empty tuple on MLA; single per-layer record on KDA.
    kda: tuple[KdaVerifyPools, ...]
    # Empty tuple on KDA; (raw FP8 MLA cache, uint8 index-K cache) on MLA.
    mla: tuple[torch.Tensor, ...]


@dataclass(frozen=True)
class VerifyPoolView:
    layers: tuple[VerifyLayerPools, ...]
    page_size: int
    context_capacity: int
    abi_version: int = 1


@dataclass(frozen=True)
class VerifyStepView:
    block_table: torch.Tensor
    sequence_lengths: torch.Tensor
    write_slots: torch.Tensor
    kda: KdaVerifyStep
    aux_hidden: torch.Tensor
    batch_requests: int
    query_width: int
    row_count: int
    abi_version: int = 1

    def validate(self, *, auxiliary_width: int | None = None) -> None:
        b, q, r = self.batch_requests, self.query_width, self.row_count
        if self.abi_version != 1 or b < 1 or q < 1 or r != b * q:
            raise ValueError("K31 verify ABI or request/query/row axes disagree")
        self.kda.validate()
        if (self.kda.batch_requests, self.kda.query_width, self.kda.row_count) != (b, q, r):
            raise ValueError("K31 and KDA verify axes must match")
        if self.block_table.dtype != torch.int32 or self.sequence_lengths.dtype != torch.int32:
            raise ValueError("verify page table and sequence lengths must be int32")
        if self.write_slots.dtype != torch.int64:
            raise ValueError("verify cache write slots must be int64[R]")
        if self.sequence_lengths.shape != (b,) or self.block_table.shape[0] != b:
            raise ValueError("K31 verify MLA metadata must have request rows")
        if (
            self.write_slots.shape != (r,)
            or self.aux_hidden.ndim != 2
            or self.aux_hidden.shape[0] != r
        ):
            raise ValueError("K31 verify cache writes/aux must have token rows")
        if auxiliary_width is not None and self.aux_hidden.shape[1] != auxiliary_width:
            raise ValueError("K31 verify aux width must match the configured capture layers")
        if self.aux_hidden.dtype != torch.bfloat16 or self.aux_hidden.stride(1) != 1:
            raise ValueError("K31 verify aux must have contiguous BF16 feature rows")


@dataclass(frozen=True)
class VerifyLayerWeights:
    layer_id: int
    attention_kind: int  # 0 MLA, 1 KDA
    is_moe: int
    attention: tuple[torch.Tensor, ...]
    attention_scalars: tuple[float, ...]
    residual: tuple[torch.Tensor, ...]
    residual_eps: tuple[float, ...]
    feed_forward: tuple[torch.Tensor, ...]
    feed_forward_scalars: tuple[float, ...]
    previous_blocks: int
    writes_block: int


@dataclass(frozen=True)
class VerifyWeights:
    layers: tuple[VerifyLayerWeights, ...]
    embedding: torch.Tensor
    final_weights: tuple[torch.Tensor, ...]
    final_eps: tuple[float, ...]
    full_attention_ids: tuple[int, ...]
    linear_attention_ids: tuple[int, ...]
    attn_res_block_size: int
    rms_eps: float
    situ_beta: float
    situ_linear_beta: float
    moe_renormalize: int
    context_length: int
    abi_version: int = 1
    query_width: int = 8
    request_batches: tuple[int, ...] = (1,)
