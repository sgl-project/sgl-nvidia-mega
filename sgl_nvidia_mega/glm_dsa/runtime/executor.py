# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 SGLang Team
"""Fused verify executor: DerivedWeights, GlmDsaFusedDecode, ErrorReporter."""

from __future__ import annotations

import atexit
import logging
import threading
import time
from typing import Dict, List, Optional

import torch
import torch.distributed as dist

from sglang.srt import runtime_context
from sgl_nvidia_mega.glm_dsa import eligibility
from sgl_nvidia_mega.glm_dsa.attention.step import StepContext
from sgl_nvidia_mega.glm_dsa.providers import Providers
from sgl_nvidia_mega.glm_dsa.runtime.compile import compile_stats

# SGLang symbols this module reads (validated by test/unit/test_contract.py)
SGLANG_READS = {
    "kernels/plugin_kernels/nvidia/spec.py|srt/megakernels/spec.py": (
        "MegakernelBinding",
        "ForwardView",
        "KvPoolView",
        "DsaView",
    ),
    "srt/runtime_context.py": ("get_parallel", "get_forward"),
    "srt/layers/layer_boundary/__init__.py": ("get_attn_tp_context",),
    "srt/layers/layer_boundary/adapters/attention.py": ("AttentionInputs",),
    "srt/eplb/expert_distribution.py": ("get_global_expert_distribution_recorder",),
    "srt/layers/attention/dsa/utils.py": ("maybe_prefetch_next_full_attention_kv",),
    "srt/layers/attention/index_topk_share.py": ("IndexTopKShareState",),
    "srt/utils.py": ("BumpAllocator",),
    "srt/layers/moe/moe_runner/base.py": ("moe_output_buffer_ctx",),
}

logger = logging.getLogger(__name__)

TARGET: Optional["GlmDsaFusedDecode"] = None


def tp_gather(obj) -> list:
    group = runtime_context.get_parallel().tp_group
    out = [None] * group.world_size
    dist.all_gather_object(out, obj, group=group.cpu_group)
    return out


def per_rank(gathered) -> str:
    return "; ".join(f"rank {r}: {', '.join(g)}" for r, g in enumerate(gathered) if g)


class DerivedWeights:
    """Owns packed / re-laid-out copies of stock weights (checkpoint unchanged)."""

    def __init__(self) -> None:
        self.tensors: Dict[str, List[torch.Tensor]] = {}

    def add(self, kind: str, tensor: torch.Tensor) -> torch.Tensor:
        self.tensors.setdefault(kind, []).append(tensor)
        return tensor

    def log(self, rank: int) -> None:
        nbytes = {
            k: sum(t.numel() * t.element_size() for t in ts)
            for k, ts in sorted(self.tensors.items())
        }
        if not nbytes:
            logger.info("GLM DSA fused decode: no derived weight copies (rank %d)", rank)
            return
        logger.info(
            "GLM DSA fused decode: derived weight copies on rank %d: %s (total %.1f MiB)",
            rank,
            ", ".join(f"{k} {v / 2**20:.1f} MiB" for k, v in nbytes.items()),
            sum(nbytes.values()) / 2**20,
        )


class GlmDsaFusedDecode:
    """Owns the fused verify path of one target model instance."""

    def __init__(self, model) -> None:
        eligibility.check_construction(model)
        self.model = model
        self.derived = DerivedWeights()
        self.providers: Dict[int, Providers] = {}
        self.families: Dict[str, object] = {}
        self._agreed: Dict[int, bool] = {}
        self._ready = False
        self.errors = ErrorReporter()

    def prepare(self) -> None:
        """From ``precompile_kernels_after_loading``: provider sets + kernel compile."""
        global TARGET
        from sgl_nvidia_mega.glm_dsa.plan import build_providers

        eligibility.check_loaded(self.model)
        self.providers, self.families = build_providers(
            self.model, eligibility.SUPPORTED_ROWS, self.derived
        )
        rank = runtime_context.get_parallel().tp_rank
        self._agree_on_composition(rank)
        self.derived.log(rank)
        logger.info(
            "GLM DSA fused decode: kernels compiled %d, loaded from cache %d (rank %d), "
            "not stored in the cache %d",
            compile_stats["compiled"],
            compile_stats["cached"],
            rank,
            compile_stats["not_stored"],
        )
        if rank == 0:
            for rows, p in self.providers.items():
                fused = {
                    k: v
                    for k, v in p.describe().items()
                    if not v.startswith(("AttentionDecode", "HopSteps", "MlpSteps"))
                }
                logger.info(
                    "GLM DSA fused decode: %d verify rows: %s",
                    rows,
                    fused or "all steps stock",
                )
        self._ready = True
        TARGET = self
        for family in self.families.values():
            if callable(getattr(family, "error_report", None)):
                self.errors.add(family.error_report)

    def _agree_on_composition(self, rank: int) -> None:
        gathered = tp_gather({m: p.describe() for m, p in self.providers.items()})
        differing = [r for r, c in enumerate(gathered) if c != gathered[0]]
        if differing:
            if rank == 0:
                logger.warning(
                    "GLM DSA fused decode: ranks %s composed different step groups "
                    "than rank 0; every rank serves the stock forward",
                    differing,
                )
            self.providers = {rows: Providers() for rows in self.providers}
            self.families = {}

    def engages(self, rows: int, forward_batch, input_embeds=None, pp_proxy_tensors=None) -> bool:
        return (
            self._ready
            and eligibility.step_in_graph_runner(
                self.model, rows, forward_batch, input_embeds, pp_proxy_tensors
            )
            and torch.cuda.is_current_stream_capturing()
            and self._first_capture_agreement(rows, forward_batch)
        )

    def _first_capture_agreement(self, rows: int, forward_batch) -> bool:
        if rows in self._agreed:
            return self._agreed[rows]
        reasons = list(eligibility.attention_step_reasons(self.model, forward_batch))
        for name, family in sorted(self.families.items()):
            extra = getattr(family, "step_reasons", None)
            if extra is not None:
                reasons += [f"{name}: {r}" for r in extra(self.model, forward_batch)]
        gathered = tp_gather(reasons)
        self._agreed[rows] = agreed = not any(gathered)
        if runtime_context.get_parallel().tp_rank == 0:
            if agreed:
                logger.info("GLM DSA fused decode: capturing the fused verify step, %d rows", rows)
            else:
                logger.warning(
                    "GLM DSA fused decode: %d verify rows captured on the stock forward: %s",
                    rows,
                    per_rank(gathered),
                )
        return agreed

    @torch.no_grad()
    def run(self, input_ids, positions, forward_batch) -> torch.Tensor:
        from sglang.srt.layers.layer_boundary import get_attn_tp_context

        providers = self.providers[int(positions.shape[0])]
        with get_attn_tp_context().maybe_input_scattered(forward_batch):
            return self._stack(providers, input_ids, positions, forward_batch)

    def _stack(self, providers, input_ids, positions, forward_batch) -> torch.Tensor:
        from sglang.srt.eplb.expert_distribution import get_global_expert_distribution_recorder
        from sglang.srt.layers.attention.dsa.utils import maybe_prefetch_next_full_attention_kv
        from sglang.srt.layers.attention.index_topk_share import IndexTopKShareState
        from sglang.srt.utils import BumpAllocator

        mm = self.model.model
        step = StepContext(
            forward_batch=forward_batch,
            positions=positions,
            zero_allocator=BumpAllocator(
                buffer_size=(mm.end_layer - mm.start_layer) * 2,
                dtype=torch.float32,
                device=positions.device,
            ),
        )
        topk_share = IndexTopKShareState(forward_batch, None)
        recorder = get_global_expert_distribution_recorder()
        attention, hops, mlp = providers.attention, providers.hops, providers.mlp
        mlp_out, residual = mm.embed_tokens(input_ids), None
        for i in range(mm.start_layer, mm.end_layer):
            layer = mm.layers[i]
            with recorder.with_current_layer(i):
                buffer = mlp_out if isinstance(mlp_out, torch.Tensor) else None
                x, residual = hops.pre_attention(layer, mlp_out, residual, step)
                o_input, topk_indices = attention.forward(layer, x, topk_share.topk_indices, step)
                topk_share.update(topk_indices)
                maybe_prefetch_next_full_attention_kv(
                    forward_batch, mm.next_full_attention_layer_id.get(i)
                )
                x, residual = hops.post_attention(layer, o_input, residual, step)
                mlp_out = mlp.forward(layer, x, buffer, step)
        return hops.final(self.model, mlp_out, residual, step)


class ErrorReporter:
    """Daemon thread watching for cross-GPU wait timeouts."""

    def __init__(self, sources=(), period_s: float = 0.5) -> None:
        self.sources = list(sources)
        self.record = None
        self._lock = threading.Lock()
        self._period_s = period_s
        self._watching = False
        if self.sources:
            self._start()

    def add(self, source) -> None:
        with self._lock:
            self.sources.append(source)
        self._start()

    def _start(self) -> None:
        if not self._watching:
            self._watching = True
            atexit.register(self.check)
            threading.Thread(target=self._watch, name="glm-dsa-errors", daemon=True).start()

    def check(self):
        with self._lock:
            if self.record is not None:
                return self.record
            for read in list(self.sources):
                try:
                    record = read()
                except Exception as e:
                    logger.warning("GLM DSA fused decode: error record unreadable: %s", e)
                    self.sources.remove(read)
                    continue
                if record:
                    self.record = record
                    logger.error(
                        "GLM DSA fused decode: a cross-GPU wait in the fused verify "
                        "step timed out and trapped; this process's CUDA context is "
                        "lost. Error record: %s",
                        record,
                    )
                    break
            return self.record

    def _watch(self) -> None:
        while self.sources and self.check() is None:
            time.sleep(self._period_s)
