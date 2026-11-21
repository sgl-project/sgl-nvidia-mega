# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the SGLang project
"""The DFlash decode aux append on the megakernel. Per (rows, slots) key, two CUDA
graphs, one per mailbox parity, captured on a side stream and replayed on the main
stream: the aux hop is a collective, so it keeps program order. The draft KV pool is
the draft runner's, which the runtime pulls."""

from __future__ import annotations

import logging
from typing import List

import torch
from sglang.srt.mem_cache.memory_pool import MLATokenToKVPool
from sglang.srt.models.dflash import DFlashMLAAttention
from sglang.srt.runtime_context import get_parallel

# The SGLang names this module reads, by file under the sglang package; test/unit/test_contract.py pins them.
SGLANG_READS = {
    "srt/mem_cache/memory_pool.py": (
        "MLATokenToKVPool",
        "dsa_kv_cache_store_fp8",
        "dcp_replicated",
        "write_loc_is_dcp_resolved",
        "kv_buffer",
        "start_layer",
    ),
    "srt/models/dflash.py": ("DFlashMLAAttention",),
    "srt/models/deepseek_v2.py": ("attn_mqa",),
    "srt/runtime_context.py": ("get_parallel", '"attn_dcp_size"'),
}
# SGLang code moved here (test/unit/test_contract.py): while SGLang still defines it,
# its digest must match, so both copies run the same code.
SGLANG_MOVED = {
    "srt/speculative/dflash_worker_v2.py:DFlashWorkerV2._take_aux_parity": "2cc78475eb4157f8",
    "srt/speculative/dflash_worker_v2.py:DFlashWorkerV2._append_target_hidden_megakernel": "b63972b980ca9eeb",
    "srt/speculative/dflash_worker_v2.py:DFlashWorkerV2._draft_mla_caches": "00ddc453b7f1c65d",
}

logger = logging.getLogger(__name__)


class AuxGraphs:
    """The aux graphs of one DFlash worker: ``megakernel_aux`` is the draft chain serving
    the aux rows, ``draft_kv_pool`` the draft runner's KV pool."""

    def __init__(self, *, megakernel_aux, draft_model, draft_kv_pool) -> None:
        self._megakernel_aux = megakernel_aux
        self.draft_model = draft_model
        self._draft_kv_pool = draft_kv_pool
        self._megakernel_aux_graphs = {}
        self._megakernel_aux_caches = None
        # Megakernel mailbox slot: alternates on every call, eager or replayed, in the
        # same order on every rank.
        self._aux_parity = 0

    def _take_aux_parity(self) -> int:
        slot, self._aux_parity = self._aux_parity, self._aux_parity ^ 1
        return slot

    def _append_target_hidden_megakernel(
        self,
        *,
        target_hidden: torch.Tensor,
        positions: torch.Tensor,
        cache_loc: torch.Tensor,
        cache_loc_2d: torch.Tensor,
        commit_lens: torch.Tensor,
    ) -> None:
        key = (tuple(target_hidden.shape), tuple(cache_loc_2d.shape))
        entry = self._megakernel_aux_graphs.get(key)
        inputs = (cache_loc, positions, commit_lens)
        parity = self._take_aux_parity()
        if entry is not None:
            for dst, src in zip(entry[0], inputs):
                dst.copy_(src)
            if target_hidden.data_ptr() != entry[2]:
                raise RuntimeError(
                    "DFLASH megakernel aux: the target's capture buffer moved after capture"
                )
            entry[1][parity].replay()
            return
        static = (
            torch.empty_like(cache_loc, dtype=torch.int64),
            torch.empty_like(positions, dtype=torch.int64),
            torch.empty_like(commit_lens, dtype=torch.int32),
        )
        for dst, src in zip(static, inputs):
            dst.copy_(src)
        loc_buf, pos_buf, commit_buf = static
        caches = self._draft_mla_caches()

        def append(slot: int):
            self._megakernel_aux.draft_aux(
                target_hidden=target_hidden,
                positions=pos_buf,
                slots_2d=loc_buf.view(cache_loc_2d.shape),
                commit_lens=commit_buf,
                kv_caches=caches,
                parity=slot,
            )

        stream = torch.cuda.Stream(device=target_hidden.device)
        current = torch.cuda.current_stream(target_hidden.device)
        stream.wait_stream(current)
        with torch.cuda.stream(stream):
            append(parity)
        current.wait_stream(stream)
        graphs, pool = {}, None
        for slot in (0, 1):
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, pool=pool, stream=stream):
                append(slot)
            pool, graphs[slot] = graph.pool(), graph
        current.wait_stream(stream)
        self._megakernel_aux_graphs[key] = (static, graphs, target_hidden.data_ptr())
        logger.info(
            "DFLASH aux CUDA graph captured on the NVIDIA megakernel rows=%d",
            target_hidden.shape[0],
        )

    def _draft_mla_caches(self) -> List[torch.Tensor]:
        if self._megakernel_aux_caches is not None:
            return self._megakernel_aux_caches
        pool = self._draft_kv_pool
        # The megakernel writes whole FP8 rows at slot * 576 bytes: the plain layout.
        plain = (
            type(pool) is MLATokenToKVPool
            and pool.dtype == torch.float8_e4m3fn
            and not pool.dsa_kv_cache_store_fp8
            and (
                pool.dcp_replicated
                or pool.write_loc_is_dcp_resolved
                or get_parallel().attn_dcp_size == 1
            )
        )
        if not plain:
            raise RuntimeError(
                "DFLASH megakernel aux needs a plain FP8 MLA draft KV pool, "
                f"got {type(pool).__name__}"
            )
        caches = []
        for layer in self.draft_model.layers:
            if not isinstance(layer.self_attn, DFlashMLAAttention):
                raise RuntimeError("DFLASH megakernel aux needs MLA draft layers")
            layer_id = layer.self_attn.attn_mqa.layer_id
            caches.append(pool.kv_buffer[layer_id - pool.start_layer])
        self._megakernel_aux_caches = caches
        return caches
