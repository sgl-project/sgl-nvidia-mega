# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# Adapted from SGLang (Apache-2.0, SGLang Team):
# python/sglang/srt/models/minimax_m3_vl.py (MiniMaxM3SparseForConditionalGeneration.forward) and
# python/sglang/srt/models/minimax_m3.py (MiniMaxM3Model.forward).
"""The covered verify step of MiniMax-M3.1: one request's eight DSpark verify rows.

The embedding and the dense layers run SGLang's modules; their residual stream is
handed to the first MoE layer. Every MoE layer is two launches: MegaAttention (the
input norm through its TP4 all-reduce, residual add and post-attention norm), then
the token-parallel MoE (router, shared and routed experts, TP4 all-reduce and residual
add), which writes the next layer's input. SGLang's final norm and logits processor
close the step.

The attention reads SGLang's step data in place: the positions and cache slots of the
forward batch, the KV length and page table of the MSA-v1 decode plan (refreshed
before every replay) and the layer's NVFP4 KV pages, which it also writes."""

from __future__ import annotations

import logging

import torch

try:
    from sglang.kernels.plugin_kernels.nvidia.spec import Step
except ModuleNotFoundError:  # SGLang before sgl-project/sglang#45389
    from sglang.srt.megakernels.spec import Step

from sgl_nvidia_mega.minimax_m31 import PAGE, ROWS, weights
from sgl_nvidia_mega.minimax_m31.sm103 import host
from sgl_nvidia_mega.minimax_m31.sm103.attention import mega_attention
from sgl_nvidia_mega.minimax_m31.sm103.moe import token_parallel as tp_moe
from sgl_nvidia_mega.minimax_m31.spec import parity, serves_this_step, text_config

logger = logging.getLogger(__name__)

# The SGLang names this module reads (test/unit/test_contract.py pins them)
SGLANG_READS = {
    "kernels/plugin_kernels/nvidia/spec.py|srt/megakernels/spec.py": (
        "Step",
        "mode",
        "rows",
        "view",
        "widths",
        "deadline_s",
        "draft_sampler",
        "summary",
        "plan",
        "forward",
    ),
    "srt/kernel_plugins/nvidia/binding.py|srt/megakernels/binding.py": (
        "metadata",
        "kv_pool",
        "pool",
    ),
    "srt/runtime_context.py": ("get_parallel", '"tp_group"'),
    "srt/models/minimax_m3_vl.py": (
        "forward(get_embedding)",
        "forward(pp_proxy_tensors)",
        "is_mrope_enabled",
        "capture_aux_hidden_states",
        "_is_layer_to_capture",
        "logits_processor",
        "lm_head",
        "model",
    ),
    "srt/models/minimax_m3.py": (
        "start_layer",
        "end_layer",
        "layers",
        "layers_to_capture",
        "is_layer_sparse",
        "embed_tokens",
        "norm",
        "forward(captured_last_layer_outputs)",
        "rotary_emb",
        "attn",
        "residual_batch",  # srt/layers/layer_boundary/residual/batch.py, as imported here
    ),
    "srt/layers/radix_attention.py": ("layer_id",),
    "srt/layers/rotary_embedding/base.py": ("cos_sin_cache", "is_neox_style"),
    "srt/layers/aux_hidden_states.py": ("AuxHiddenStateList", "capture"),
    "srt/layers/layer_boundary/residual/batch.py": ("start", "stream_of", "clear"),
    "srt/layers/layer_boundary/residual/stream.py": (
        "pending",
        "update",
        "export",
    ),
    "srt/layers/layer_boundary/residual/add_norm.py": ("is_plain_add",),
    "srt/model_executor/forward_batch_info.py": (
        "forward_mode",
        "is_target_verify",
        "batch_size",
        "positions",
        "out_cache_loc",
    ),
    "srt/layers/attention/minimax_sparse_nvfp4_backend.py": ("msa_v1_decode_plan",),
    "srt/layers/attention/minimax_sparse_ops/msa_plan.py": (
        "page_table",
        "final_seq_lens",
    ),
    "srt/mem_cache/minimax_m3_nvfp4_pool.py": (
        "MiniMaxM3Nvfp4KVPool",
        "get_main_k_scale_buffer",
        "get_main_v_scale_buffer",
        "get_index_k_scale_buffer",
    ),
    "srt/mem_cache/memory_pool.py": (
        "page_size",
        "get_key_buffer",
        "get_value_buffer",
        "get_index_k_buffer",
    ),
}


def _moe_layers(lm) -> list[int]:
    return [
        i
        for i in range(lm.start_layer, lm.end_layer)
        if getattr(lm.layers[i], "is_layer_sparse", False)
    ]


def model_disable_reasons(model, hf_config) -> list[str]:
    """What the server arguments do not show: the layer loop and the operands."""
    lm = model.model
    layers = _moe_layers(lm)
    if not layers:
        return ["the model has no MoE decoder layer"]
    reasons = []
    if layers != list(range(layers[0], lm.end_layer)):
        reasons.append(f"MoE layers {layers} are not one tail of the decoder")
    if getattr(model, "is_mrope_enabled", False):
        reasons.append("M-RoPE positions are not supported")
    for i in layers:
        reasons += weights.layer_reasons(lm.layers[i], i)
    tables = {
        lm.layers[i].self_attn.rotary_emb.cos_sin_cache.data_ptr() for i in layers
    }
    if len(tables) != 1:
        reasons.append("the MoE layers do not share one RoPE table")
    return reasons


class VerifyStep:
    """One rank's covered verify step: the operands, the kernels and their workspaces,
    built at load; the attention's cache and step bindings, made at the first covered
    forward (SGLang sizes the KV cache after the build)."""

    # the target offers no draft sampler (BuiltMegakernel)
    draft_sampler = None

    def __init__(self, model, hf_config, ctx):
        from sglang.srt.runtime_context import get_parallel

        if ROWS not in ctx.widths:
            raise RuntimeError(
                f"MiniMax-M3.1 megakernel: the widths {list(ctx.widths)} miss one "
                f"request's {ROWS} verify rows"
            )
        self.widths = (ROWS,)
        self.model, self.lm = model, model.model
        self.layer_ids = _moe_layers(self.lm)
        self.first_moe = self.layer_ids[0]
        tp_group = get_parallel().tp_group
        self.cpu_group = tp_group.cpu_group
        self.device = device = torch.device("cuda", torch.cuda.current_device())
        self.deadline_ns = round(ctx.deadline_s * 1e9)
        text = text_config(hf_config)
        scaling = float(text.routed_scaling_factor)
        eps = float(text.rms_norm_eps)
        layers = [self.lm.layers[i] for i in self.layer_ids]
        self.attention_weights = [
            mega_attention.AttentionWeights(**weights.attention_operands(layer))
            for layer in layers
        ]
        # SGLang builds every norm of the layer with the config's epsilon
        self.eps = dict(
            input_norm_eps=eps, head_norm_eps=eps, post_attention_norm_eps=eps
        )
        # the kernel's RoPE table is BF16 [positions, cos 32 | sin 32]
        rope = layers[0].self_attn.rotary_emb
        self.cos_sin = rope.cos_sin_cache.to(torch.bfloat16).contiguous()
        group = tp_group.device_group
        self.workspace = mega_attention.create_workspace(
            group, device, self.deadline_ns
        )
        self.arenas = [
            mega_attention.new_arena(device, self.deadline_ns) for _ in layers
        ]
        self.scratch = mega_attention.AttentionScratch(device)
        # the residual stream rows entering each MoE layer (the MoE writes the next
        # one), and the attention's outputs the MoE reads
        self.layer_input, self.attention_residual, self.attention_norm = (
            torch.zeros((ROWS, tp_moe.H), dtype=torch.bfloat16, device=device)
            for _ in range(3)
        )
        moe_arena = tp_moe.create_arena(group, device)
        self.moe = [
            tp_moe.TokenParallelMoe(
                moe_arena,
                wait_timeout_ns=self.deadline_ns,
                hidden=self.attention_norm,
                residual=self.attention_residual,
                output=self.layer_input,
                **weights.moe_operands(layer, scaling),
            )
            for layer in layers
        ]
        self.attention: list[mega_attention.MegaAttention] = []
        self._bound = None  # the identity of the step data the attention reads
        logger.info(
            "MiniMax-M3.1 megakernel: MoE layers %d-%d, routed experts %.1f GiB a rank",
            self.layer_ids[0],
            self.layer_ids[-1],
            sum(
                m.tensors[name].nbytes
                for m in self.moe
                for name in (
                    "routed_fc1_weight",
                    "routed_fc1_weight_sf",
                    "routed_fc2_weight",
                    "routed_fc2_weight_sf",
                )
            )
            / 2**30,
        )

    def summary(self) -> str:
        return (
            f"one request's {ROWS}-row DSpark verify: MegaAttention + token-parallel "
            f"MoE on layers {self.layer_ids[0]}-{self.layer_ids[-1]}"
        )

    def plan(
        self, input_ids, forward_batch, get_embedding=False, pp_proxy_tensors=None
    ) -> Step | None:
        """The step if this forward is covered, else None (SGLang's forward runs)."""
        if get_embedding or pp_proxy_tensors is not None:
            return None
        if not forward_batch.forward_mode.is_target_verify():
            return None
        if int(input_ids.shape[0]) != ROWS or int(forward_batch.batch_size) != 1:
            return None
        return Step(mode="target_verify", rows=ROWS) if serves_this_step() else None

    def _bind(self, view, forward_batch) -> None:
        """Bind the attention to the KV pool and the step data this forward reads;
        host only, and again only when SGLang hands other tensors."""
        from sglang.srt.mem_cache.minimax_m3_nvfp4_pool import MiniMaxM3Nvfp4KVPool

        pool = view.kv_pool.pool
        plan = getattr(view.metadata, "msa_v1_decode_plan", None)
        tensors = (
            forward_batch.positions,
            forward_batch.out_cache_loc,
            None if plan is None else plan.page_table,
            None if plan is None else plan.final_seq_lens,
        )
        key = (id(pool), *(None if t is None else t.data_ptr() for t in tensors))
        if key == self._bound:
            return

        def bind():
            if not isinstance(pool, MiniMaxM3Nvfp4KVPool) or pool.page_size != PAGE:
                raise ValueError(
                    f"the KV pool {type(pool).__name__} is not the M3.1 NVFP4 pool with "
                    f"{PAGE}-token pages"
                )
            if plan is None:
                raise ValueError("the forward has no MSA-v1 decode plan")
            positions, slots, page_table, kv_len = tensors
            step = mega_attention.AttentionStep(
                positions=positions,
                slot_mapping=slots,
                kv_len=kv_len,
                kv_indices=page_table,
                page_table=page_table.view(-1),
                cos_sin=self.cos_sin,
            )
            attention = []
            for n, i in enumerate(self.layer_ids):
                layer_id = self.lm.layers[i].self_attn.attn.layer_id
                cache = mega_attention.AttentionCache(
                    k=pool.get_key_buffer(layer_id),
                    v=pool.get_value_buffer(layer_id),
                    ki=pool.get_index_k_buffer(layer_id),
                    k_scale=pool.get_main_k_scale_buffer(layer_id),
                    v_scale=pool.get_main_v_scale_buffer(layer_id),
                    ki_scale=pool.get_index_k_scale_buffer(layer_id),
                )
                attention.append(
                    mega_attention.MegaAttention(
                        self.workspace,
                        self.attention_weights[n],
                        cache,
                        step,
                        self.scratch,
                        self.layer_input,
                        self.attention_residual,
                        self.attention_norm,
                        **self.eps,
                        wait_timeout_ns=self.deadline_ns,
                        arena=self.arenas[n],
                    )
                )
            return attention

        self.attention = host.agree(self.cpu_group, bind, "binding MegaAttention")
        self._bound = key

    def forward(
        self,
        step,
        original,
        input_ids,
        positions,
        forward_batch,
        get_embedding=False,
        pp_proxy_tensors=None,
    ):
        """``MiniMaxM3SparseForConditionalGeneration.forward`` for a covered step
        (``original``: the model's own forward, for the parity check)."""
        if parity() and not torch.cuda.is_current_stream_capturing():
            want = original(
                input_ids, positions, forward_batch
            ).next_token_logits.clone()
            got = self._forward(step, input_ids, positions, forward_batch)
            log_parity("target logits", want, got.next_token_logits)
            return got
        return self._forward(step, input_ids, positions, forward_batch)

    def _forward(self, step, input_ids, positions, forward_batch):
        from sglang.srt.layers.aux_hidden_states import AuxHiddenStateList
        from sglang.srt.layers.layer_boundary.residual import batch as residual_batch

        self._bind(step.view, forward_batch)
        model, lm = self.model, self.lm
        hidden = lm.embed_tokens(input_ids)
        aux = AuxHiddenStateList()
        residual_batch.start(forward_batch)
        try:
            for i in range(lm.start_layer, self.first_moe):
                layer = lm.layers[i]
                hidden = layer(
                    positions=positions,
                    forward_batch=forward_batch,
                    hidden_states=hidden,
                    captured_last_layer_outputs=(
                        aux if getattr(layer, "_is_layer_to_capture", False) else None
                    ),
                )
            # the residual stream entering the first MoE layer: the last dense layer's
            # owed all-reduce completed, then its plain residual add
            stream = residual_batch.stream_of(forward_batch)
            if stream.pending is not None and not stream.pending.update.is_plain_add:
                raise RuntimeError(
                    "MiniMax-M3.1 megakernel: the last dense layer's residual update "
                    "is not a plain add"
                )
            output, residual = stream.export(hidden)
        finally:
            residual_batch.clear(forward_batch)
        if residual is None:
            self.layer_input.copy_(output)
        else:
            torch.add(output, residual, out=self.layer_input)
        for n, i in enumerate(self.layer_ids):
            if getattr(lm.layers[i], "_is_layer_to_capture", False):
                # the residual stream entering layer i, as its attention boundary has it
                aux.capture(self.layer_input)
            self.attention[n].run()
            self.moe[n].run()
        if lm.end_layer in lm.layers_to_capture:
            aux.capture(self.layer_input)
        hidden = lm.norm(self.layer_input)
        aux = aux if model.capture_aux_hidden_states and len(aux) else None
        return model.logits_processor(
            input_ids, hidden, model.lm_head, forward_batch, aux
        )


def log_parity(what: str, want: torch.Tensor, got: torch.Tensor) -> None:
    """SGLang's rows against the megakernel's on one eager step (the parity switch)."""
    want, got = want.float(), got.float()
    diff = (want - got).abs().max()
    cosine = torch.nn.functional.cosine_similarity(want, got, dim=-1).min()
    top1 = (want.argmax(-1) == got.argmax(-1)).float().mean()
    log_p, log_q = want.log_softmax(-1), got.log_softmax(-1)
    kl = (log_p.exp() * (log_p - log_q)).sum(-1).max()
    logger.info(
        "NVIDIA megakernel parity %s rows=%d max_abs=%.4g rel=%.3g min_cosine=%.6f "
        "top1=%.3f max_kl=%.4g",
        what,
        want.shape[0],
        diff.item(),
        (diff / want.abs().max().clamp_min(1e-30)).item(),
        cosine.item(),
        top1.item(),
        kl.item(),
    )
