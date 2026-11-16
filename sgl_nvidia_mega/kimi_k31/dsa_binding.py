# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""K3.1 sparse decode's fixed tensor/numeric step contract."""

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class MlaDecodeStep:
    block_table: torch.Tensor
    sequence_lengths: torch.Tensor
    write_slots: torch.Tensor


@dataclass(frozen=True)
class KdaDecodeStep:
    state_indices: torch.Tensor

    def track_decode_state(self, layer_id):
        """The host owns post-forward snapshots; ordinary decode has no ring."""


@dataclass(frozen=True)
class IndexDecodeStep:
    schedule: torch.Tensor
    topk_plan: torch.Tensor


@dataclass(frozen=True)
class SparseDecodeStep:
    mla: MlaDecodeStep
    kda: KdaDecodeStep
    index: IndexDecodeStep
    sparse: int

    def validate(self):
        if self.sparse not in (0, 1):
            raise ValueError("decode graph mode must be dense(0) or sparse(1)")
        rows = self.mla.sequence_lengths.numel()
        if self.mla.sequence_lengths.shape != (rows,) or rows not in (1, 2, 3, 4, 5, 6, 7, 8, 16, 32, 64):
            raise ValueError("K3.1 sparse decode needs a supported request vector")
        if self.mla.write_slots.shape != (rows,) or self.kda.state_indices.shape != (rows,):
            raise ValueError("decode slots must have one entry per request")
        if self.mla.block_table.ndim != 2 or self.mla.block_table.shape[0] != rows:
            raise ValueError("decode page table must have one row per request")
