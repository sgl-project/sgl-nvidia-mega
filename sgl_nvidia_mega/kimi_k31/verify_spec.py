# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""K31 TARGET_VERIFY build and fixed tensor-entry dispatch."""

from __future__ import annotations

import logging
import os

import torch

from sgl_nvidia_mega.common import kernel_cache
from sgl_nvidia_mega.kimi_k31.k3 import preload
from sgl_nvidia_mega.kimi_k31.k3.providers import _cuda_memory, build_resources
from sgl_nvidia_mega.kimi_k31.k3.shapes import kda_pitch, tp_of_heads
from sgl_nvidia_mega.kimi_k31.k3.spec import _deadline_ns

from .verify_binding import KdaVerifyWeights
from .verify_contract import query_width_reasons, served_query_widths, verify_keys
from .verify_executor import K31VerifyExecutor
from .verify_indexer import IndexKeyWeights, PreparedVerifyIndexKey

logger = logging.getLogger(__name__)


def config_disable_reasons(cfg, hf_config):
    reasons = []
    if str(cfg.speculative_algorithm).upper() != "DFLASH":
        reasons.append("K31 verify requires DFLASH")
    reasons += query_width_reasons(cfg.speculative_num_draft_tokens, getattr(cfg, "dcp_size", 1))
    if not cfg.enable_linear_replayssm_spec:
        reasons.append("K31 verify requires KDA ReplaySSM spec rings")
    if cfg.enable_linear_replayssm:
        reasons.append("K31 verify does not use delayed replay decode")
    if int(cfg.context_length or 0) < 8:
        reasons.append("K31 verify needs space for a complete query block")
    if cfg.cuda_graph_config is None or cfg.cuda_graph_config.decode.backend != "full":
        reasons.append("K31 verify requires full CUDA graphs")
    if (
        cfg.prefill_attention_backend or cfg.attention_backend,
        cfg.decode_attention_backend or cfg.attention_backend,
    ) != ("dsa", "dsa"):
        reasons.append("K31 verify requires DSA prefill/decode")
    return reasons


class BuiltK31Verify:
    tensor_binding_kind = 31
    draft_sampler = None
    draft_loop = draft_extend = begin_draft_round = graph_variants = None
    speculative_runtime_hooks = None  # no runtime-optimization hooks (SGLang reads the member unguarded)

    def __init__(self, weights, deadline_s):
        from .verify_weights import WeightModel

        self.query_width = int(weights.query_width)
        if self.query_width not in served_query_widths():
            raise ValueError("K31 verify requires Q8 or Q16 (Q4-Q7 with K31MK_VERIFY_QPORT=1)")
        self.keys = verify_keys(self.query_width, weights.request_batches)
        self.request_batches = tuple(b for b, _, _ in self.keys)
        self.widths = tuple(r for _, _, r in self.keys)
        model = WeightModel(weights)
        mla_pitch = None
        if os.environ.get("K31MK_DSA_QF", "0") == "1":
            # The host packs [q_a | kv_a | gate | wk | weights_proj] under the same switch (dsa/verify.py DSA_QF);
            # build_resources checks every MLA projection against this pitch, dsa/verify.py the column layout.
            mla_rows = {int(r.attention[0].shape[0]) for r in weights.layers if r.attention_kind == 0}
            if len(mla_rows) != 1:
                raise ValueError(f"K31MK_DSA_QF: packed MLA projections differ in rows: {sorted(mla_rows)}")
            mla_pitch = mla_rows.pop()
        from sgl_nvidia_mega.kimi_k31.k3.cutedsl.mla import head_gate

        self.resources = build_resources(
            model,
            context_length=min(2048, weights.context_length),
            memory_before=_cuda_memory(),
            deadline_ns=_deadline_ns(deadline_s),
            query_widths=(self.query_width,),
            packed_projection_weights=tuple(model.packed),
            batches=self.widths,
            mla_pitch=mla_pitch,
            qf_g18=mla_pitch is not None and head_gate.G18,  # K31MK_DSA_QF_G18: reorder + split (mla/head_gate.py)
        )
        self.indexers = {}
        for layer in self.resources.layers:
            if not hasattr(layer.self_attn, "q_b_proj"):
                continue
            indexer = layer.self_attn.indexer
            if not indexer.skip_rope or indexer.apply_hadamard:
                raise ValueError("K31 verify index-K requires NoPE without Hadamard")
            self.indexers[int(layer.layer_idx)] = PreparedVerifyIndexKey(
                IndexKeyWeights(
                    indexer.wk.weight,
                    indexer.k_norm.weight,
                    indexer.k_norm.bias,
                    float(indexer.k_norm.variance_epsilon),
                )
            )
        self.executor = K31VerifyExecutor(
            self.resources, self.indexers, (), query_width=self.query_width,
            request_batches=self.request_batches
        )
        self.pool_view = None
        self._active_logged = set()

    def summary(self):
        tp = 96 // self.resources.heads
        return f"TARGET_VERIFY Breq={self.request_batches} Q={self.query_width}; TP{tp}/DCP1/EP1; short all-selected DSA; host commit"

    def supports(self, b, q, r):
        return (b, q, r) in self.keys

    def configure_aux(self, capture_layers):
        old = self.executor
        self.executor = K31VerifyExecutor(
            self.resources, self.indexers, capture_layers, query_width=self.query_width,
            request_batches=self.request_batches
        )
        self.executor.kda_cores = old.kda_cores
        self.executor.kda_pools = old.kda_pools

    def prepare(self, pools):
        if pools.abi_version != 1 or pools.page_size != 64:
            raise ValueError("unsupported K31 verify pool ABI/page size")
        self.pool_view = pools
        for layer in pools.layers:
            attention = self.resources.attention[layer.layer_id]
            if layer.mla:
                cache, index_cache = layer.mla
                if cache.dtype != torch.uint8 or cache.shape[-2:] != (64, 576):
                    raise ValueError("K31 verify requires raw FP8 E4M3 MLA pages")
                attention.cache = cache
                attention.core.prepare_cache(cache)
                self.indexers[layer.layer_id].bind(index_cache, 64, rows=self.widths)
            else:
                self.executor.kda_pools[layer.layer_id] = layer.kda[0]
        self.resources.binding_fingerprint = "k31-verify-abi1"
        self._prepare_front_kda()
        kernel_cache.log_summary("the verify prepare")

    def _prepare_front_kda(self):
        # front owns this module. Missing implementation must fail loudly before
        # an overall verify graph is advertised as active.
        try:
            from .verify_kda import PreparedKdaVerify
        except ImportError:
            return
        for layer in self.resources.layers:
            layer_id = int(layer.layer_idx)
            if layer_id not in self.executor.kda_pools:
                continue
            old = self.resources.attention[layer_id].core
            conv, bias, fb, a_log, dt, norm = old.weights
            weights = KdaVerifyWeights(
                conv,
                bias,
                fb,
                a_log,
                dt,
                norm,
                int(old.has_bias),
                float(layer.self_attn.attn.lower_bound),
                128**-0.5,
                float(layer.self_attn.o_norm.eps),
            )
            core = PreparedKdaVerify(
                weights, tuple((b, q, kda_pitch(tp_of_heads(old.heads)), 0) for b, q, _ in self.keys),
                old.device.index,
            )
            core.prepare_caches(self.executor.kda_pools[layer_id])
            self.executor.kda_cores[layer_id] = core

    def run_verify(self, embeddings, view):
        if len(self.executor.kda_cores) != 69:
            raise RuntimeError("K31 verify: front KDA provider is not integrated")
        key = (view.batch_requests, view.query_width)
        if preload.ENABLED and key not in self._active_logged:
            # CuTe kernels the first call of this key would still load lazily (expected 0).
            logger.info("K31MK_PRELOAD: audit before the first verify Breq=%d Q=%d: %s", *key, preload.audit())
        result = self.executor.run(embeddings, view)
        if key not in self._active_logged:
            logger.info(
                "NVIDIA megakernel: active mode=TARGET_VERIFY Breq=%d Q=%d",
                *key,
            )
            self._active_logged.add(key)
        return result

    # Legacy protocol methods are deliberately inert: the host tensor-entry
    # adapter owns planning, logits and all model-specific runtime extraction.
    def plan(self, *args, **kwargs):
        raise RuntimeError("K31 verify requires the SGLang tensor-entry patch")

    def forward(self, *args, **kwargs):
        raise RuntimeError("K31 verify requires the SGLang tensor-entry patch")


def build_from_tensors(weights, deadline_s):
    return BuiltK31Verify(weights, deadline_s)
