# SPDX-License-Identifier: Apache-2.0
"""K31 TARGET_VERIFY build and fixed tensor-entry dispatch."""

from __future__ import annotations

import logging
import os

import torch

from sgl_nvidia_mega.kimi_k31.k3.default_scope import best_default
from sgl_nvidia_mega.kimi_k31.k3.providers import _cuda_memory, build_resources
from sgl_nvidia_mega.kimi_k31.k3.spec import _deadline_ns

from .verify_binding import KdaVerifyWeights
from .verify_contract import supported_query_width, verify_keys
from .verify_executor import K31VerifyExecutor
from .verify_indexer import IndexKeyWeights, PreparedVerifyIndexKey

# Names read from SGLang. A contract test checks these before GPU use.
SGLANG_READS = {
    "srt/runtime_context.py": ("get_parallel",),
}

logger = logging.getLogger(__name__)


def config_disable_reasons(cfg, hf_config):
    reasons = []
    if str(cfg.speculative_algorithm).upper() != "DFLASH":
        reasons.append("K31 verify requires DFLASH")
    q = int(cfg.speculative_num_draft_tokens or 0)
    if not supported_query_width(q):
        reasons.append(
            "K31 verify requires explicit Q8, Q16, or opt-in K31MK_VERIFY_Q5=1"
        )
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
    dcp = int(getattr(cfg, "dcp_size", 1) or 1)
    cp = int(getattr(cfg, "attn_cp_size", 1) or 1)
    prefill_cp = bool(getattr(cfg, "enable_prefill_cp", False))
    q5_scope = q == 5 and supported_query_width(q) and int(cfg.tp_size) == dcp == 8
    if q == 5 and not (q5_scope and cp == 1 and not prefill_cp):
        reasons.append(
            "K31 Q5 is limited to TP8/DCP8 without extra context parallelism"
        )
    if dcp > 1 or cp > 1 or prefill_cp:
        # Sparse verify's context-parallel path: index-K / latent KV token-strided over the
        # whole TP group, Q8 rows (dsa/verify.py PreparedSparseVerifyMla).
        if not (
            dcp == int(cfg.tp_size)
            and dcp in DCP_SIZES
            and (q == 8 or q5_scope)
            and cp == 1
            and not prefill_cp
        ):
            reasons.append(
                "K31 verify supports DCP8/TP8 or DCP16/TP16 at Q8, or opt-in Q5 at DCP8/TP8"
            )
    reasons += _disaggregation_reasons(cfg)
    return reasons


# DCP16 is the served scope; DCP8 runs the same dcp_size-parameterized path (96 heads as 8x12).
DCP_SIZES = (8, 16)


def _disaggregation_reasons(cfg):
    mode = cfg.disaggregation_mode
    if mode == "null":
        return []
    if mode != "decode":
        # A PD prefill pool has no speculative state and serves no TARGET_VERIFY step.
        return ["K31 verify serves only the PD decode role"]
    reasons = []
    if cfg.disaggregation_transfer_backend != "mooncake":
        reasons.append(
            "K31 verify under PD decode is limited to the mooncake transfer backend"
        )
    if cfg.disaggregation_decode_host_receive_threshold < 1:
        reasons.append("K31 verify under PD decode does not support host receive")
    return reasons


class BuiltK31Verify:
    tensor_binding_kind = 31
    draft_sampler = None
    draft_loop = draft_extend = begin_draft_round = graph_variants = None
    # no speculative runtime hooks factory: SGLang keeps its own strategies
    speculative_runtime_hooks = None

    def __init__(self, weights, deadline_s):
        from .verify_weights import WeightModel

        self.query_width = int(weights.query_width)
        if not supported_query_width(self.query_width):
            raise ValueError("K31 verify requires Q8, Q16, or opt-in K31MK_VERIFY_Q5=1")
        if self.query_width == 5:
            from sglang.srt.runtime_context import get_parallel

            parallel = get_parallel()
            if int(parallel.tp_size) != 8 or int(parallel.attn_dcp_size) != 8:
                raise ValueError("K31 Q5 requires TP8/DCP8")
        self.keys = verify_keys(self.query_width, weights.request_batches)
        self.request_batches = tuple(b for b, _, _ in self.keys)
        self.widths = tuple(r for _, _, r in self.keys)
        if self.query_width == 5:
            logger.info(
                "NVIDIA megakernel: experimental Q5 enabled; Breq=%s, rows=%s",
                self.request_batches,
                self.widths,
            )
        model = WeightModel(weights)
        self.resources = build_resources(
            model,
            context_length=min(2048, weights.context_length),
            memory_before=_cuda_memory(),
            deadline_ns=_deadline_ns(deadline_s),
            query_widths=(self.query_width,),
            packed_projection_weights=tuple(model.packed),
            batches=self.widths,
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
            self.resources,
            self.indexers,
            (),
            query_width=self.query_width,
            request_batches=self.request_batches,
        )
        self.pool_view = None
        self._active_logged = set()
        self._kda_commit = best_default("K31MK_KDA_COMMIT", "1", "0") == "1"
        self._kda_commit_logged = False
        # The DFLASH sampled accept's rank-0 broadcast on its own multicast box (K31MK_RANK0_BCAST, default_scope), built here so
        # its rendezvous and compile run with the rest of the build, on every rank.
        self._bcast, self._bcast_logged = None, False
        # Use-time round counts of the two DFLASH sampled-accept hops, logged at 1, 10, 100, ... rounds per process.
        self._rounds = {"rank-0 broadcast": [0, 1], "sampled accept": [0, 1]}
        if best_default("K31MK_RANK0_BCAST", "1", "0") == "1":
            from sglang.srt.runtime_context import get_parallel

            from .sm103.draft.accept import Rank0Bcast

            parallel = get_parallel()
            if int(parallel.tp_size) > 1:
                self._bcast = Rank0Bcast(
                    parallel.tp_group.device_group,
                    torch.device("cuda", torch.cuda.current_device()),
                    int(parallel.tp_rank),
                    deadline_ns=_deadline_ns(deadline_s),
                )
        # The DFLASH T > 0 sampled accept on the vocab shards (K31MK_SAMPLED_ACCEPT, default_scope): built by
        # prepare_sampled_accept, which the worker calls on every rank once it knows the target's head.
        self._sampled, self._sampled_logged = None, False
        self._sampled_deadline_ns = _deadline_ns(deadline_s)

    def summary(self):
        return f"TARGET_VERIFY Breq={self.request_batches} Q={self.query_width}; TP8|TP16/DCP1/EP1; short all-selected DSA; host commit"

    def supports(self, b, q, r):
        return (b, q, r) in self.keys

    def configure_aux(self, capture_layers):
        old = self.executor
        self.executor = K31VerifyExecutor(
            self.resources,
            self.indexers,
            capture_layers,
            query_width=self.query_width,
            request_batches=self.request_batches,
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

    def _prepare_front_kda(self):
        # front owns this module. Missing implementation must fail loudly before
        # an overall verify graph is advertised as active.
        try:
            from .verify_kda import PreparedKdaVerify
        except ImportError:
            return
        # The executor's projected KDA rows: kda_pitch of this rank's heads.
        pitch = self.resources.scratch["kda.projected"].shape[1]
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
                weights,
                tuple((b, q, pitch, 0) for b, q, _ in self.keys),
                old.device.index,
            )
            core.prepare_caches(self.executor.kda_pools[layer_id])
            self.executor.kda_cores[layer_id] = core

    def run_verify(self, embeddings, view):
        if len(self.executor.kda_cores) != 69:
            raise RuntimeError("K31 verify: front KDA provider is not integrated")
        result = self.executor.run(embeddings, view)
        key = (view.batch_requests, view.query_width)
        if key not in self._active_logged:
            logger.info(
                "NVIDIA megakernel: active mode=TARGET_VERIFY Breq=%d Q=%d",
                *key,
            )
            self._active_logged.add(key)
        return result

    def rank0_broadcast(self, accept_len, bonus):
        """Rank 0's sampled accept (accept_len int32 [B], bonus int64 [B]) on every rank, in place, in one
        multicast hop (draft.accept.Rank0Bcast). False (launching nothing; the caller keeps SGLang's
        broadcasts) at TP 1 or with K31MK_RANK0_BCAST off (default_scope). Collective: every rank, in the same order."""
        if self._bcast is None:
            return False
        self._bcast.run(accept_len, bonus)
        if not self._bcast_logged:
            logger.info(
                "NVIDIA megakernel: rank-0 sampled-accept broadcast active (one multicast hop; "
                "the dflash-selector NCCL broadcasts skipped)"
            )
            self._bcast_logged = True
        self._count_round("rank-0 broadcast")
        return True

    def _count_round(self, path):
        count = self._rounds[path]
        count[0] += 1
        if count[0] == count[1]:
            logger.info("NVIDIA megakernel: %s rounds = %d", path, count[0])
            count[1] *= 10

    def prepare_sampled_accept(
        self, *, vocab_size, softcap, shard_width, seed, topk=16
    ):
        """Build draft.sampled.SampledAccept (its multicast boxes and kernels) for a target head of
        ``vocab_size`` over TP shards of ``shard_width`` columns and the selector's ``topk`` (16 / 32),
        at block 5 for Q5, otherwise legacy block 8 (Q8 / Q16). True when built; False (building nothing) at TP 1, with K31MK_SAMPLED_ACCEPT off (default_scope), or
        (by default) for a selector K or shard width it does not serve. Collective: every rank, in the same order."""
        if best_default("K31MK_SAMPLED_ACCEPT", "1", "0") != "1":
            return False
        from sglang.srt.runtime_context import get_parallel

        from .sm103.draft.sampled import SLICE, TOPKS, SampledAccept

        parallel = get_parallel()
        if int(parallel.tp_size) < 2:
            return False
        # Rank-uniform: the selector and the head shard are the same on every rank.
        if (
            int(topk) not in TOPKS or int(shard_width) % SLICE
        ) and "K31MK_SAMPLED_ACCEPT" not in os.environ:
            logger.info(
                "NVIDIA megakernel: sampled accept off (selector K %d, shard width %d; it serves K %s and "
                "widths divisible by %d)",
                int(topk),
                int(shard_width),
                TOPKS,
                SLICE,
            )
            return False
        self._sampled = SampledAccept(
            parallel.tp_group.device_group,
            torch.device("cuda", torch.cuda.current_device()),
            int(parallel.tp_rank),
            int(parallel.tp_size),
            width=int(shard_width),
            vocab=int(vocab_size),
            softcap=softcap,
            seed=int(seed),
            deadline_ns=self._sampled_deadline_ns,
            block=5 if self.query_width == 5 else 8,
            topk=int(topk),
        )
        return True

    def sampled_accept_fits(self, requests, logits=None):
        return self._sampled is not None and self._sampled.fits(int(requests), logits)

    def sampled_accept(
        self, *, logits, temperatures, candidates, candidate_ids, q_rows
    ):
        """(accept_len int32 [B], bonus int64 [B], target_predict int64 [B, block]) of the chain sampled
        accept from this rank's raw verify shard (draft.sampled.SampledAccept.run); rank 0's decision
        on every rank. Collective: every rank, in the same order, after prepare_sampled_accept."""
        out = self._sampled.run(logits, temperatures, candidates, candidate_ids, q_rows)
        if not self._sampled_logged:
            # launch= names the host path (draft.sampled launch_mode): "driver/raw" expected; "cute (...)" is the
            # correct but ~4x slower CuTe-call fallback, which serving would otherwise hide.
            logger.info(
                "NVIDIA megakernel: sampled accept active (shard stats + one multicast record "
                "all-gather + rank-0 decision hop; the vocab all-gather, dense q, SoftmaxTemp and "
                "the dflash-selector NCCL broadcasts skipped) launch=%s",
                self._sampled.launch_mode,
            )
            self._sampled_logged = True
        self._count_round("sampled accept")
        return out

    def kda_commit(self, **kwargs):
        """Stage 3b: the post-verify KDA commit in one launch (draft.commit.kda_commit, keyword
        arguments as there). False (the caller keeps today's path) with K31MK_KDA_COMMIT off (default_scope)."""
        if not self._kda_commit:
            return False
        from .sm103.draft.commit import kda_commit

        kda_commit(**kwargs)
        if not self._kda_commit_logged:
            logger.info("NVIDIA megakernel: KDA commit (stage 3b) active")
            self._kda_commit_logged = True
        return True

    # Legacy protocol methods are deliberately inert: the host tensor-entry
    # adapter owns planning, logits and all model-specific runtime extraction.
    def plan(self, *args, **kwargs):
        raise RuntimeError("K31 verify requires the SGLang tensor-entry patch")

    def forward(self, *args, **kwargs):
        raise RuntimeError("K31 verify requires the SGLang tensor-entry patch")


def build_from_tensors(weights, deadline_s):
    return BuiltK31Verify(weights, deadline_s)
