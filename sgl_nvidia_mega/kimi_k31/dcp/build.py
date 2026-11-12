# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""K3.1 non-spec decode under DCP = TP (``K31MK_DCP=1``): build, step contract and executor.

The host provides, per decode step:
- ``mla.block_table``: [requests, pages] widened page ids (each is also this rank's physical page id);
- ``mla.write_slots``: [rows] widened slots of the written tokens (the prologue maps them to this rank's rows);
- ``mla.sequence_lengths``: [requests] global lengths;
- ``index``: a DcpIndexStep (below): the DeepGEMM schedule of the LOCAL lengths, the topk_v2 plan of the GLOBAL
  lengths, both lengths per row.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass

import torch

from ..dsa_binding import IndexDecodeStep
from ..dsa_executor import SparseDecodeExecutor

SGLANG_READS = {
    "srt/runtime_context.py": ("get_model", "get_parallel"),
}


@dataclass(frozen=True)
class DcpIndexStep(IndexDecodeStep):
    local_lengths: torch.Tensor = None   # int32 [requests, q]: keys of each row on this rank (score contexts)
    global_lengths: torch.Tensor = None  # int32 [rows]: causal global length of each row (0: padding row)

    def validate(self, rows: int, q: int = 1) -> None:
        if self.local_lengths is None or self.global_lengths is None:
            raise ValueError("DCP decode needs local and global lengths")
        if self.local_lengths.dtype != torch.int32 or self.global_lengths.dtype != torch.int32:
            raise ValueError("DCP lengths must be int32")
        if self.local_lengths.shape != (rows // q, q) or self.global_lengths.shape != (rows,):
            raise ValueError("DCP lengths: local [requests, q], global [rows]")


class DcpDecodeExecutor(SparseDecodeExecutor):
    """Advances the DCP epoch and maps the widened write slots to this rank's rows, once per forward, then runs the
    ordinary chain (MLA cores are PreparedDcpSparseMla)."""

    def __init__(self, resources, comm):
        super().__init__(resources)
        self.comm = comm
        self.local_slots = torch.full((64,), -1, dtype=torch.int64, device=comm.device)

    def run(self, input_ids, inputs_embeds, view):
        rows = int(input_ids.shape[0])
        view.index.validate(rows)
        local = self.local_slots[:rows]
        self.comm.prologue(view.mla.write_slots, local)
        mla = dataclasses.replace(view.mla, write_slots=local)
        return super().run(input_ids, inputs_embeds, dataclasses.replace(view, mla=mla))


def build_dcp_decode(model, hf_config, context):
    """``dsa_spec.build_sparse_decode`` under DCP = TP: the same resources with every MLA core wrapped."""
    from sglang.srt.runtime_context import get_model, get_parallel

    from sgl_nvidia_mega.kimi_k31.k3.compat import validate_runtime_compatibility
    from sgl_nvidia_mega.kimi_k31.k3.providers import _cuda_memory, build_resources
    from sgl_nvidia_mega.kimi_k31.k3.shapes import heads as tp_heads
    from sgl_nvidia_mega.kimi_k31.k3.spec import WIDTHS, _deadline_ns

    from ..dsa.prepared import PreparedSparseIndexMla
    from ..dsa_spec import BuiltSparseDecode
    from ..projection_storage import coalesce_index_projection, pitch
    from .attention import PreparedDcpSparseMla, dense_scores
    from .comm import DcpComm

    widths = tuple(width for width in context.widths if width in WIDTHS)
    if context.tokens_per_request != 1 or not widths:
        raise ValueError("K3.1 DCP decode requires non-spec decode at supported row widths")
    parallel = get_parallel()
    tp, rank = int(parallel.tp_size), int(parallel.tp_rank)
    dcp = int(getattr(parallel, "attn_dcp_size", 1))
    if dcp != tp:
        raise ValueError(f"the K3.1 MK DCP path needs DCP = TP, got DCP{dcp} / TP{tp}")
    heads = tp_heads(tp)
    validate_runtime_compatibility()
    length = int(get_model().context_length)
    resources = build_resources(
        model.language_model,
        context_length=length,
        memory_before=_cuda_memory(),
        deadline_ns=_deadline_ns(context.deadline_s),
        mla_pitch=pitch(heads),
        mla_storage=coalesce_index_projection,
        mla_factory=PreparedSparseIndexMla,
    )
    device = resources.scratch["hidden.0"].device
    comm = DcpComm.from_group(
        parallel.tp_group.device_group, device, dcp, rank, heads, max(widths)
    )
    dense = dense_scores(max(widths), length, dcp, device)
    ordinals = [int(i) for i in model.language_model.config.full_attention_layer_ids]
    for layer_id, attention in resources.attention.items():
        if attention.kind == "mla":
            attention.core = PreparedDcpSparseMla(
                attention.core, comm, ordinals.index(layer_id), max_context=length, dense=dense
            )
    built = BuiltSparseDecode(model, resources, widths)
    built.executor = DcpDecodeExecutor(resources, comm)
    built.dcp_comm = comm
    return built
