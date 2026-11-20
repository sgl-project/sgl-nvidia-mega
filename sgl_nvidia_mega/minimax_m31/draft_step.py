# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# Adapted from SGLang (Apache-2.0, SGLang Team):
# python/sglang/srt/models/dspark_minimax.py (MiniMaxDraftBackbone.forward).
"""The covered DSpark draft block of MiniMax-M3.1.

The draft runner's one-request block step (seven rows: the bonus token and six mask
tokens) runs the draft backbone: the target's vocab-parallel embedding, the five dense
decoder layers and the final norm in one cooperative launch per rank, every TP4 sum
inside it. SGLang's writes of the target's hidden states into the draft KV cache are
unchanged; the kernel writes the block's own K/V. It reads SGLang's step data in place:
the block's ids, positions and cache slots, and the attention metadata's KV length and
page table.

The proposal that samples the block (the draft LM head, the Markov head, the sampling)
is SGLang's, or with slot ``markov`` the plugin's (``sampler``): the built object
offers SGLang's folded proposal a ``draft_sampler`` for draft batches of 1-16 requests,
whose Markov chain is one kernel per rank; slot ``draft_lm_head`` runs its LM head on
an MXFP8 copy. Its buffers are allocated at the build, before SGLang sizes its KV
pools; the folded proposal asks for the sampler later, with its own buffers."""

from __future__ import annotations

import logging

import torch
from sglang.kernels.plugin_kernels.nvidia.spec import Step

from sgl_nvidia_mega.minimax_m31 import (
    DRAFT_HF_CONFIG,
    DRAFT_LAYERS,
    DRAFT_ROWS,
    PAGE,
    ROWS,
    TP,
    weights,
)
from sgl_nvidia_mega.minimax_m31.runtime import SLOTS, draft_slots
from sgl_nvidia_mega.minimax_m31.sm103 import host
from sgl_nvidia_mega.minimax_m31.sm103.draft import backbone
from sgl_nvidia_mega.minimax_m31.spec import parity, scaled_rope, serves_this_step

logger = logging.getLogger(__name__)

# The SGLang names this module reads (test/unit/test_contract.py pins them)
SGLANG_READS = {
    "kernels/plugin_kernels/nvidia/spec.py": (
        "Step",
        "view",
        "deadline_s",
        "tokens_per_request",
        # the entry SGLang's folded DSpark proposal asks a built draft spec for
        "draft_sampler",
    ),
    "srt/kernel_plugins/nvidia/binding.py": (
        "metadata",
        "kv_pool",
        "pool",
    ),
    "srt/runtime_context.py": ("get_parallel", '"tp_group"', '"attn_tp_size"'),
    "srt/distributed/parallel_state.py": (
        "cpu_group",
        "device_group",
        "world_size",
        "rank_in_group",
    ),
    "srt/environ.py": (
        "envs",
        "SGLANG_DSPARK_EMBED_IN_GRAPH",
        # the draft mode the chain's logit staging follows
        "SGLANG_DSPARK_USE_GREEDY_DRAFT",
    ),
    "srt/layers/dp_attention.py": (
        "get_cuda_graph_batch_size_alignment",
        "require_attn_tp_gather",
    ),
    "srt/speculative/dspark_components/dspark_config.py": (
        "get_dspark_sample_from_anchor",
    ),
    "srt/models/dspark_minimax.py": (
        "MiniMaxDraftBackbone",
        "layers",
        "decoder_layer",
        "embed_tokens",
        "norm",
        "is_nemotron_35_draft",
        "forward(input_embeds)",
        "forward(get_embedding)",
        "forward(pp_proxy_tensors)",
    ),
    "srt/models/minimax_m3.py": (
        "self_attn",
        "mlp",
        "is_layer_sparse",
        "qkv_proj",
        "o_proj",
        "gate_up_proj",
        "down_proj",
        "q_norm",
        "k_norm",
        "input_layernorm",
        "post_attention_layernorm",
        "rotary_emb",
        "attn",
        "num_heads",
        "num_kv_heads",
    ),
    "srt/layers/radix_attention.py": ("layer_id", "sliding_window_size"),
    "srt/layers/rotary_embedding/base.py": (
        "RotaryEmbedding",
        "cos_sin_cache",
        "is_neox_style",
        "rotary_dim",
    ),
    "srt/layers/vocab_parallel_embedding.py": (
        "shard_indices",
        "org_vocab_start_index",
        "num_org_elements",
        "num_added_elements",
    ),
    "srt/layers/logits_processor.py": ("LogitsProcessorOutput", "hidden_states"),
    "srt/model_executor/forward_mode.py": ("is_target_verify",),
    "srt/model_executor/forward_batch_info.py": (
        "forward_mode",
        "batch_size",
        "positions",
        "out_cache_loc",
    ),
    "srt/layers/attention/trtllm_mha_backend.py": (
        "page_table",
        "swa_page_table",
        "cache_seqlens_int32",
        "max_seq_len_q",
    ),
    "srt/mem_cache/memory_pool.py": (
        "MHATokenToKVPool",
        "get_key_buffer",
        "get_value_buffer",
        "dtype",
        "page_size",
    ),
}

HIDDEN, HEAD_DIM, ROTARY_DIM = backbone.HIDDEN, backbone.HEAD_DIM, backbone.ROTARY_DIM
# (module, name, rows, K) of every MXFP8 linear of one layer, per TP4 rank
LINEARS = (
    ("self_attn", "qkv_proj", backbone.QKV_ROWS, HIDDEN),
    ("self_attn", "o_proj", HIDDEN, backbone.Q_SIZE),
    ("mlp", "gate_up_proj", backbone.GATE_UP_ROWS, HIDDEN),
    ("mlp", "down_proj", HIDDEN, backbone.INTER),
)


def _row_alignment() -> int:
    """The row multiple a captured draft step pads to: SGLang captures only the batch
    sizes whose rows divide by it."""
    from sglang.srt.layers.dp_attention import (
        get_cuda_graph_batch_size_alignment,
        require_attn_tp_gather,
    )

    # the dense DSpark draft is DP-local, so only the attention-TP gather applies
    return get_cuda_graph_batch_size_alignment(
        gathered_buffer_required=require_attn_tp_gather()
    )


def proposal_disable_reasons(model, slots=None) -> list[str]:
    """Why slot markov's chain cannot serve the loaded draft's Markov head ([] when it
    can, or the slot is off). The target's LM head is attached after the build:
    ``sampler.build_steps`` checks it when the folded proposal asks for the sampler."""
    if "markov" not in draft_slots(SLOTS if slots is None else slots):
        return []
    from sglang.srt.runtime_context import get_parallel

    from sgl_nvidia_mega.minimax_m31 import sampler

    # the draft tokens of a block: the verify rows less the anchor (the spec's
    # requirements fix speculative_num_draft_tokens at ROWS)
    reason = sampler.head_reason(
        model, gamma=ROWS - 1, world=get_parallel().tp_group.world_size
    )
    return [f"slot markov: {reason}"] if reason else []


def model_disable_reasons(model, hf_config) -> list[str]:
    """Why the loaded draft does not fit the backbone, or the requested draft slots:
    SGLang checks a draft's spec at load only, so every check runs here."""
    from sglang.srt.environ import envs
    from sglang.srt.layers.rotary_embedding.base import RotaryEmbedding
    from sglang.srt.runtime_context import get_parallel
    from sglang.srt.speculative.dspark_components.dspark_config import (
        get_dspark_sample_from_anchor,
    )

    config = model.config
    reasons = [
        f"config.{key} {getattr(config, key, None)!r} is not {want!r}"
        for key, want in DRAFT_HF_CONFIG.items()
        if getattr(config, key, None) != want
    ]
    layer_types = (getattr(config, "dflash_config", None) or {}).get("layer_types")
    if layer_types != ["sliding_attention"] * DRAFT_LAYERS:
        reasons.append(
            f"draft layer types {layer_types}: the kernel runs sliding attention"
        )
    if scaled_rope(config):
        reasons.append(
            f"config.rope_scaling {config.rope_scaling!r}: the kernel reads plain RoPE"
        )
    if not get_dspark_sample_from_anchor(hf_config):
        reasons.append(
            f"sample_from_anchor is off: the draft block has {DRAFT_ROWS + 1} rows, the "
            f"kernel {DRAFT_ROWS}"
        )
    if not envs.SGLANG_DSPARK_EMBED_IN_GRAPH.get():
        reasons.append(
            "SGLANG_DSPARK_EMBED_IN_GRAPH=0 passes embeddings; the kernel embeds ids"
        )
    if get_parallel().attn_tp_size != TP:
        reasons.append(
            f"attention TP {get_parallel().attn_tp_size}: the kernel runs TP{TP}"
        )
    if DRAFT_ROWS % _row_alignment():
        reasons.append(
            f"draft steps pad to a multiple of {_row_alignment()} rows, so a "
            f"{DRAFT_ROWS}-row block never runs alone; pass --disable-attn-tp-gather"
        )
    if getattr(model, "is_nemotron_35_draft", False):
        reasons.append(
            "the draft does not embed with the target's vocab-parallel embedding"
        )
    reasons += proposal_disable_reasons(model)
    layers = getattr(model, "layers", None)
    if layers is None or len(layers) != DRAFT_LAYERS:
        return reasons + [
            f"the draft model does not have {DRAFT_LAYERS} decoder layers"
        ]
    for i, layer in enumerate(layers):
        decoder, what = layer.decoder_layer, f"draft layer {i}"
        attn = decoder.self_attn
        if getattr(decoder, "is_layer_sparse", False) or not hasattr(
            decoder.mlp, "gate_up_proj"
        ):
            reasons.append(f"{what} is not a dense MLP layer")
            continue
        for owner, name, rows, k in LINEARS:
            try:
                weights.mxfp8(getattr(getattr(decoder, owner), name), rows, k)
            except ValueError as error:
                reasons.append(f"{what} {name}: {error}")
        if (attn.num_heads + 2 * attn.num_kv_heads) * HEAD_DIM != backbone.QKV_ROWS:
            reasons.append(
                f"{what}: {attn.num_heads} Q / {attn.num_kv_heads} KV heads a rank"
            )
        window = DRAFT_HF_CONFIG["sliding_window"] - 1
        if attn.attn.sliding_window_size != window:
            reasons.append(
                f"{what}: window_left {attn.attn.sliding_window_size} is not {window}"
            )
        rope = attn.rotary_emb
        table = rope.cos_sin_cache
        if (
            type(rope) is not RotaryEmbedding
            or not rope.is_neox_style
            or rope.rotary_dim != ROTARY_DIM
            or table.dtype != torch.float32
            or table.dim() != 2
            or table.shape[1] != ROTARY_DIM
        ):
            reasons.append(
                f"{what}: the kernel reads one FP32 NeoX RoPE table of {ROTARY_DIM} dims"
            )
    tables = {layer.self_attn.rotary_emb.cos_sin_cache.data_ptr() for layer in layers}
    if len(tables) != 1:
        reasons.append("the draft layers do not share one RoPE table")
    return reasons


class DraftStep:
    """One rank's covered draft block: the backbone's kernel and arena, built at load;
    its operands and step bindings, made at the first covered forward (SGLang attaches
    the target's embedding and sizes the draft KV cache after the build). With slot
    markov also the draft proposal: its buffers, built at load, and the sampler SGLang's
    folded proposal asks for."""

    widths = (DRAFT_ROWS,)
    # SGLang's proposal samples the block unless slot markov offers the sampler
    # (BuiltMegakernel; __init__ sets it)
    draft_sampler = None
    # no EAGLE entries, graph variants or speculative runtime hooks (BuiltMegakernel)
    draft_loop = None
    draft_extend = None
    begin_draft_round = None
    graph_variants = None
    speculative_runtime_hooks = None

    def __init__(self, model, hf_config, ctx):
        from sglang.srt.runtime_context import get_parallel

        from sgl_nvidia_mega.minimax_m31 import draft as draft_spec

        self.model = model
        tp_group = get_parallel().tp_group
        self.cpu_group = tp_group.cpu_group
        self.device = device = torch.device("cuda", torch.cuda.current_device())
        self.deadline_ns = round(ctx.deadline_s * 1e9)
        self.eps = float(model.config.rms_norm_eps)
        backbone.load_module(device.index, self.deadline_ns)
        self.arena = backbone.create_arena(tp_group.device_group, device)
        self.out = torch.empty(
            (DRAFT_ROWS, HIDDEN), dtype=torch.bfloat16, device=device
        )
        self.backbone = None
        self._bound = None  # the identity of the step data the backbone reads
        # the requested draft slots, and what samples the block (for the summary)
        self.slots = draft_slots(SLOTS)
        self.prebuilt = None  # sampler.Prebuilt, with slot markov built
        self.proposal = self._build_proposal(ctx, tp_group.rank_in_group)
        draft_spec._COUNTS["built"] += 1

    def _build_proposal(self, ctx, rank: int) -> str:
        """What samples the block. With slot markov: the chain's buffers (and slot
        draft_lm_head's LM-head copy), allocated here, after the backbone's arena and
        before SGLang sizes its KV pools (collective: every rank builds them in this
        order), and the sampler offered to SGLang's folded proposal. Every rank logs
        which proposal it serves and why."""
        if "markov" not in self.slots:
            logger.info(
                "NVIDIA megakernel: DSpark draft proposal on rank %d: SGLang's (slot "
                "markov is not requested)",
                rank,
            )
            return "SGLang's proposal (slot markov off)"
        from sglang.srt.environ import envs

        from sgl_nvidia_mega.minimax_m31 import draft as draft_spec
        from sgl_nvidia_mega.minimax_m31 import sampler

        requested = sorted(self.slots)
        off = draft_spec.proposal_off_reason()
        if off:
            logger.warning(
                "NVIDIA megakernel: draft slots %s are not engaged on rank %d: %s",
                requested,
                rank,
                off,
            )
            return f"SGLang's proposal (draft slots {requested} not engaged: {off})"
        self.prebuilt = sampler.prebuild(
            self.model,
            deadline_s=ctx.deadline_s,
            slots=self.slots,
            gamma=ctx.tokens_per_request - 1,
            greedy_draft=envs.SGLANG_DSPARK_USE_GREEDY_DRAFT.get(),
        )
        self.draft_sampler = self._draft_sampler
        lm_head = (
            "an MXFP8 copy of the LM head"
            if "draft_lm_head" in self.slots
            else "SGLang's LM head"
        )
        return (
            f"Markov chain in one kernel at 1-{sampler.MAX_REQUESTS} requests, "
            f"{lm_head} (draft slots {requested})"
        )

    def _draft_sampler(self, **buffers):
        """SGLang's ``draft_sampler`` factory (BuiltMegakernel): the folded proposal
        calls it once, on every rank, with its buffers (gamma, max_bs, greedy_draft,
        noise, draft_tokens, step_logits, step_temperatures). Returns bs -> the step
        that replaces the base logits and the Markov block there, or None where
        SGLang's proposal stays (more requests than the chain serves)."""
        from sgl_nvidia_mega.minimax_m31 import draft as draft_spec
        from sgl_nvidia_mega.minimax_m31 import sampler

        draft_spec._COUNTS["sampler"] += 1
        steps = sampler.build_steps(self.model, prebuilt=self.prebuilt, **buffers)
        return steps.get

    def summary(self) -> str:
        return (
            f"DSpark draft block of {DRAFT_ROWS} rows: {DRAFT_LAYERS} layers in one "
            f"launch; proposal: {self.proposal}"
        )

    def plan(
        self,
        input_ids,
        forward_batch,
        input_embeds=None,
        get_embedding=False,
        pp_proxy_tensors=None,
    ) -> Step | None:
        """The step if this draft forward is covered, else None (SGLang's forward)."""
        if input_embeds is not None or get_embedding or pp_proxy_tensors is not None:
            return None
        if not forward_batch.forward_mode.is_target_verify():
            return None
        if int(input_ids.shape[0]) != DRAFT_ROWS or int(forward_batch.batch_size) != 1:
            return None
        return (
            Step(mode="target_verify", rows=DRAFT_ROWS) if serves_this_step() else None
        )

    def _bind(self, view, input_ids, forward_batch) -> None:
        """Bind the backbone to the draft's weights, KV pool and step data; host only,
        and again only when SGLang hands other tensors."""
        from sglang.srt.mem_cache.memory_pool import MHATokenToKVPool

        pool, meta = view.kv_pool.pool, view.metadata
        page_table = getattr(meta, "page_table", None)
        kv_len = getattr(meta, "cache_seqlens_int32", None)
        tensors = (
            input_ids,
            forward_batch.positions,
            forward_batch.out_cache_loc,
            page_table,
            kv_len,
        )
        key = (id(pool), *(None if t is None else t.data_ptr() for t in tensors))
        if key == self._bound:
            return
        model = self.model

        def bind():
            if type(pool) is not MHATokenToKVPool:
                raise ValueError(
                    f"the draft KV pool {type(pool).__name__} is not SGLang's plain MHA "
                    "pool (fused, quantized or sliding-window draft caches are not "
                    "supported)"
                )
            if pool.dtype != torch.bfloat16 or pool.page_size != PAGE:
                raise ValueError(
                    f"the draft KV pool is {pool.dtype} with {pool.page_size}-token "
                    f"pages, not bfloat16 with {PAGE}"
                )
            if (
                page_table is None
                or kv_len is None
                or getattr(meta, "swa_page_table", None) is not None
            ):
                raise ValueError("the draft attention metadata has no plain page table")
            # the KV length of the block's rows: the cached tokens and the block
            if getattr(meta, "max_seq_len_q", None) != DRAFT_ROWS:
                raise ValueError(
                    f"the draft metadata's {meta.max_seq_len_q} rows a request are not "
                    f"the block's {DRAFT_ROWS}"
                )
            embed = model.embed_tokens
            shard = getattr(embed, "shard_indices", None)
            if (
                shard is None
                or shard.num_added_elements
                or embed.weight.dtype != torch.bfloat16
            ):
                raise ValueError(
                    "the draft does not embed with the target's BF16 vocab-parallel "
                    "embedding without added vocabulary"
                )
            layers, k_caches, v_caches = [], [], []
            for layer in model.layers:
                decoder = layer.decoder_layer
                attn = decoder.self_attn
                linears = {}
                for (owner, name, rows, k), field in zip(
                    LINEARS, ("qkv", "o", "gate_up", "down")
                ):
                    weight, scale = weights.mxfp8(
                        getattr(getattr(decoder, owner), name), rows, k
                    )
                    linears[field], linears[f"{field}_sf"] = weight, scale
                layers.append(
                    backbone.DraftLayerWeights(
                        input_norm=decoder.input_layernorm.weight.data,
                        post_norm=decoder.post_attention_layernorm.weight.data,
                        q_norm=attn.q_norm.weight.data,
                        k_norm=attn.k_norm.weight.data,
                        **linears,
                    )
                )
                layer_id = attn.attn.layer_id
                k_caches.append(pool.get_key_buffer(layer_id).view(-1, HEAD_DIM))
                v_caches.append(pool.get_value_buffer(layer_id).view(-1, HEAD_DIM))
            draft_weights = backbone.DraftBackboneWeights(
                layers=layers,
                embed=embed.weight.data[: shard.num_org_elements],
                vocab_start=int(shard.org_vocab_start_index),
                final_norm=model.norm.weight.data,
                eps=self.eps,
            )
            inputs = backbone.DraftBackboneInputs(
                input_ids=input_ids,
                positions=forward_batch.positions,
                slot_mapping=forward_batch.out_cache_loc,
                page_table=page_table[0],
                kv_len=kv_len[:1],
                cos_sin=model.layers[0].self_attn.rotary_emb.cos_sin_cache,
                k_caches=k_caches,
                v_caches=v_caches,
            )
            return backbone.DraftBackbone(
                self.arena,
                draft_weights,
                inputs,
                self.out,
                wait_timeout_ns=self.deadline_ns,
            )

        self.backbone = host.agree(self.cpu_group, bind, "binding the draft backbone")
        self._bound = key

    def forward(
        self,
        step,
        original,
        input_ids,
        positions,
        forward_batch,
        input_embeds=None,
        get_embedding=False,
        pp_proxy_tensors=None,
    ):
        """``MiniMaxDraftBackbone.forward`` for a covered block: the final-normed hidden
        rows, which SGLang's draft head samples from (``original``: the draft's own
        forward, for the parity check)."""
        from sglang.srt.layers.logits_processor import LogitsProcessorOutput

        from sgl_nvidia_mega.minimax_m31.runtime.parity import log_parity

        want = None
        if parity() and not torch.cuda.is_current_stream_capturing():
            want = original(input_ids, positions, forward_batch).hidden_states.clone()
        self._bind(step.view, input_ids, forward_batch)
        hidden = self.backbone.run()
        if want is not None:
            log_parity("draft hidden", want, hidden)
        return LogitsProcessorOutput(next_token_logits=None, hidden_states=hidden)
