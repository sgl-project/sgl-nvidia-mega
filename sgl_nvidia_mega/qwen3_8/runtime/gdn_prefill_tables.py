# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""FlashInfer's GDN core in a breakable prefill graph: the static tables of each
captured bucket, which ``gdn_prefill._core_in_graph`` runs FlashInfer on.

SGLang captures a non-Triton GDN extend only for an external prefill body that asks
for it (``SupportsPrefillBody.linear_prefill_graph_seqs``), and after it refills its
own tables of a bucket (``GDNPrefillGraphMetadata``) before the bucket's capture and
each replay, it calls ``refill_prefill_graph`` so these follow.
"""

from __future__ import annotations

import os
from typing import Optional

import msgspec
import numpy as np
import torch

# Off by default: with it the executor's prefill body runs FlashInfer's GDN core in
# breakable prefill graphs (with SGLANG_ENABLE_GDN_PREFILL_GRAPH_EXTEND and slot
# gdn_prefill) instead of breaking at every GDN layer.
FLASHINFER_ENV = "SGLANG_OPT_GDN_PREFILL_GRAPH_FLASHINFER"
# Static sequence rows of the non-CP call; each unused row costs about 10 us per
# layer, and batches with more sequences run eagerly.
MAX_SEQS_ENV = "SGLANG_GDN_PREFILL_GRAPH_FLASHINFER_MAX_SEQS"

SGLANG_READS = {
    "srt/layers/attention/linear/gdn_backend.py": (
        "kernel_dispatcher",
        "extend_kernel",
        "_prefill_graph_track_rows",
    ),
    "srt/layers/attention/hybrid_linear_attn_backend.py": ("state_pool", "device"),
    "srt/mem_cache/memory_pool.py": ("mamba_cache", "temporal"),
    "srt/layers/attention/linear/gdn_prefill_graph.py": (
        "GDN_CHUNK_SIZE",
        "max_seqs",
        "packed",
        "cache_indices",
    ),
    "srt/layers/attention/linear/kernels/gdn_flashinfer.py": ("FlashInferGDNKernel",),
    "srt/layers/attention/linear/kda_prefill_graph.py": ("chunk_rows", "final_rows"),
    "srt/runtime_context.py": ("mamba_cache_chunk_size", "get_exec"),
    "kernels/ops/mamba/gdn_prefill_track.py": (
        "TRACK_CHUNK_STATE",
        "TRACK_FINAL_STATE",
    ),
    "srt/model_executor/forward_batch_info.py": (
        "extend_seq_lens_cpu",
        "mamba_track_mask",
    ),
}


def requested() -> bool:
    return os.environ.get(FLASHINFER_ENV, "0").lower() in ("true", "1", "yes", "y")


def unsupported_reason(backend) -> Optional[str]:
    """Why FlashInfer's core cannot run in the graph on this GDN backend, else None."""
    from sglang.srt.layers.attention.linear.gdn_prefill_graph import (
        GDN_CHUNK_SIZE,
    )
    from sglang.srt.layers.attention.linear.kernels.gdn_flashinfer import (
        FlashInferGDNKernel,
    )
    from sglang.srt.runtime_context import get_exec, mamba_cache_chunk_size

    extend_kernel = backend.kernel_dispatcher.extend_kernel
    temporal = backend.state_pool.mamba_cache.temporal
    if not isinstance(extend_kernel, FlashInferGDNKernel):
        return f"extend kernel is {type(extend_kernel).__name__}"
    if torch.cuda.get_device_capability()[0] != 10:
        return "validated on SM10x only"
    if temporal.dtype != torch.bfloat16:
        return f"SSM state dtype {temporal.dtype}, validated with bfloat16"
    # FlashInfer writes the pool in place through 32-byte aligned pointers.
    layer_bytes = temporal.stride(0) * temporal.element_size()
    if temporal.data_ptr() % 32 or layer_bytes % 32:
        return "SSM pool is not 32-byte aligned per layer"
    # Checkpoints come every 64 tokens; the eager plan uses the cache chunk.
    cache_chunk = mamba_cache_chunk_size()
    if get_exec().mamba.enable_mamba_extra_buffer and cache_chunk != GDN_CHUNK_SIZE:
        return f"mamba cache chunk {cache_chunk} != {GDN_CHUNK_SIZE}"
    return None


class GDNFlashInferGraphTables(msgspec.Struct, frozen=True):
    """Static tables of one bucket's FlashInfer GDN core. A one-sequence batch takes
    the CP kernel (``cp_*``) when ``cp_single``, any other batch the non-CP kernel
    (``ncp_*``, ``max_seqs`` rows); the idle kernel gets an empty batch. Empty rows
    read and write the dummy state slot 0. Checkpoint j of a sequence is its state
    after (j + 1) * 64 tokens."""

    max_seqs: int
    # FlashInfer's own split for one sequence; when off the CP kernel is not captured.
    cp_single: bool
    # int64, the kernels' cu_seqlens dtype; one copy per fill.
    packed64: torch.Tensor
    cp_cu_seqlens: torch.Tensor  # [2]
    ncp_cu_seqlens: torch.Tensor  # [max_seqs + 1]
    host64: np.ndarray
    # int32; one copy per fill.
    packed32: torch.Tensor
    cp_checkpoint_cu: torch.Tensor  # [2]
    ncp_checkpoint_cu: torch.Tensor  # [max_seqs + 1]
    # Per batch row, for gdn_track_ssm_state: the post-kernel snapshot, and
    # TRACK_FINAL_STATE before the kernel for chunk-0 rows (their initial state).
    track_mode: torch.Tensor  # [max_seqs]
    pre_track_mode: torch.Tensor  # [max_seqs]
    track_h_row: torch.Tensor  # [max_seqs] row of the state checkpoints
    host32: np.ndarray
    # Filled on the device from the batch's cache slots.
    cp_state_indices: torch.Tensor  # [1]
    ncp_state_indices: torch.Tensor  # [max_seqs]

    @classmethod
    def allocate(
        cls, *, max_seqs: int, cp_single: bool, device: torch.device
    ) -> GDNFlashInferGraphTables:
        sizes64 = [2, max_seqs + 1]
        sizes32 = [2, max_seqs + 1, max_seqs, max_seqs, max_seqs]
        packed64 = torch.zeros(sum(sizes64), dtype=torch.int64, device=device)
        packed32 = torch.zeros(sum(sizes32), dtype=torch.int32, device=device)
        v64 = torch.split(packed64, sizes64)
        v32 = torch.split(packed32, sizes32)
        return cls(
            max_seqs=max_seqs,
            cp_single=cp_single,
            packed64=packed64,
            cp_cu_seqlens=v64[0],
            ncp_cu_seqlens=v64[1],
            host64=np.zeros(sum(sizes64), dtype=np.int64),
            packed32=packed32,
            cp_checkpoint_cu=v32[0],
            ncp_checkpoint_cu=v32[1],
            track_mode=v32[2],
            pre_track_mode=v32[3],
            track_h_row=v32[4],
            host32=np.zeros(sum(sizes32), dtype=np.int32),
            cp_state_indices=torch.zeros(1, dtype=torch.int32, device=device),
            ncp_state_indices=torch.zeros(max_seqs, dtype=torch.int32, device=device),
        )

    def fill(
        self, *, lens: np.ndarray, cache_indices: torch.Tensor, track_rows
    ) -> None:
        """``track_rows``: SGLang's KDAPrefillTrackRows of the batch, or None."""
        from sglang.kernels.ops.mamba.gdn_prefill_track import (
            TRACK_CHUNK_STATE,
            TRACK_FINAL_STATE,
        )
        from sglang.srt.layers.attention.linear.gdn_prefill_graph import (
            GDN_CHUNK_SIZE,
        )

        bs = lens.shape[0]
        single = bs == 1 and self.cp_single
        # Checkpoint j of a sequence is its state after (j + 1) * 64 tokens.
        n_ckpt = lens // GDN_CHUNK_SIZE
        cp_cu = self.host64[:2]
        ncp_cu = self.host64[2:]
        ncp_cu[:] = 0
        cp_cu[:] = (0, int(lens[0]) if single else 0)
        cp_ck = self.host32[:2]
        ncp_ck = self.host32[2 : self.max_seqs + 3]
        ncp_ck[:] = 0
        cp_ck[:] = (0, int(n_ckpt[0]) if single else 0)
        if not single:
            ncp_cu[1 : bs + 1] = np.cumsum(lens)
            ncp_cu[bs + 1 :] = ncp_cu[bs]
            ncp_ck[1 : bs + 1] = np.cumsum(n_ckpt)
            ncp_ck[bs + 1 :] = ncp_ck[bs]
        o = self.max_seqs + 3
        track_mode = self.host32[o : o + self.max_seqs]
        pre_track_mode = self.host32[o + self.max_seqs : o + 2 * self.max_seqs]
        track_h_row = self.host32[o + 2 * self.max_seqs :]
        track_mode[:] = 0
        pre_track_mode[:] = 0
        track_h_row[:] = 0
        if track_rows is not None:
            ckpt_start = cp_ck if single else ncp_ck
            for row, chunk in track_rows.chunk_rows:
                if chunk == 0:
                    pre_track_mode[row] = TRACK_FINAL_STATE
                else:
                    track_mode[row] = TRACK_CHUNK_STATE
                    track_h_row[row] = ckpt_start[row] + chunk - 1
            for row in track_rows.final_rows:
                track_mode[row] = TRACK_FINAL_STATE
        self.packed64.copy_(
            torch.from_numpy(self.host64).pin_memory(), non_blocking=True
        )
        self.packed32.copy_(
            torch.from_numpy(self.host32).pin_memory(), non_blocking=True
        )
        slots = cache_indices[:bs].clamp(min=0)
        if single:
            self.cp_state_indices.copy_(slots)
            self.ncp_state_indices.zero_()
        else:
            self.cp_state_indices.zero_()
            self.ncp_state_indices[:bs].copy_(slots)
            self.ncp_state_indices[bs:].zero_()


class GraphTables:
    """The tables of every captured bucket of one GDN backend, keyed by SGLang's
    per-bucket metadata (``GDNPrefillGraphMetadata``)."""

    def __init__(self, backend, max_seqs: int):
        self.backend = backend
        self.max_seqs = max_seqs
        num_v_heads = backend.state_pool.mamba_cache.temporal.shape[-3]
        # FlashInfer's own split on SM10x takes CP only while sequences x heads x 4
        # < SMs, so with many heads per rank one sequence runs on the non-CP kernel.
        self.cp_single = (
            num_v_heads * 4
            < torch.cuda.get_device_properties(backend.device).multi_processor_count
        )
        self._buckets = {}

    @classmethod
    def build(cls, backend) -> Optional[GraphTables]:
        """The tables if FlashInfer's core can run in the graph here, else None."""
        import logging

        reason = unsupported_reason(backend)
        if reason is not None:
            logging.getLogger(__name__).info(
                "GDN extend graph: FlashInfer unsupported here (%s)", reason
            )
            return None
        return cls(backend, int(os.environ.get(MAX_SEQS_ENV, "8")))

    def get(self, meta) -> Optional[GDNFlashInferGraphTables]:
        entry = self._buckets.get(id(meta))
        return None if entry is None or entry[0] is not meta else entry[1]

    def refill(self, meta, forward_batch) -> None:
        """After SGLang refilled ``meta`` for this batch, outside any capture."""
        tables = self.get(meta)
        if tables is None:
            tables = GDNFlashInferGraphTables.allocate(
                max_seqs=meta.max_seqs,
                cp_single=self.cp_single,
                device=meta.packed.device,
            )
            self._buckets[id(meta)] = (meta, tables)
        track_rows = None
        if forward_batch.mamba_track_mask is not None:
            track_rows = self.backend._prefill_graph_track_rows(forward_batch)
        tables.fill(
            lens=np.asarray(forward_batch.extend_seq_lens_cpu, dtype=np.int64),
            cache_indices=meta.cache_indices,
            track_rows=track_rows,
        )
