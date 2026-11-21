# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the SGLang project
"""K3.1 DFlash2 draft layers on the megakernel chain in serving (stage 1; ``SGLANG_K31_DRAFT_MK``).

Stage 1 with M1b is on by default in the GB300 BEST scope (kimi_k31/k3/default_scope.py), off elsewhere;
``SGLANG_K31_DRAFT_MK`` set to 0 / 1 turns it off / on. With it, the chain also builds the head (2a) and the aux
(2b) for the K3.1 DFlash runtime, unless ``K31MK_DFLASH_RUNTIME=0`` (no runtime runs them). A stage the setup
does not support declines with one INFO line.

The draft graph's forward of one B1 x 8-row TARGET_VERIFY block runs ``DraftChain`` (chain.py) in place of
SGLang's decoder layers; the chain's attention hook (M1a) is SGLang's own MLA sublayer up to W_UV, so the
block's KV write, RoPE and sliding window stay SGLang's. Every other draft forward (other widths, prefill)
runs SGLang's. ``SGLANG_K31_DRAFT_MK=0``: ``max_rows`` 0, so SGLang never wraps the draft's forward.

M1b: the attention sublayer runs on MK blocks as well (``DraftAttention``:
fused_a, DraftMlaQuery's FP8 query + the block's KV insert at its out_cache_loc slots, DraftMlaValue); only the
core stays SGLang's, its sliding-window MLA call on the pre-built FP8 query (``save_kv_cache=False``); a draft
KV pool that is not plain FP8 MLA keeps M1a.

Stage 2b: the DFlash worker's decode aux append, the committed rows'
target context into the draft KV, runs ``DraftAux`` (aux.py) on the chain's comm through the K3.1 DFlash
runtime for the same B1 x 8 rows, which replays it on the main stream.

Mailboxes: one captured graph per batch size replays ``parity=0`` every round; each of the 2L hops has its
own slot, so a slot's reuse distance is a whole round, as for the target's graph-baked parity. All
captured draft graphs come from one stream, so their hops replay in program order: a capture on a second
stream raises.
"""

from __future__ import annotations

import logging
from typing import Optional

import torch

try:
    from sglang.kernels.plugin_kernels.nvidia.spec import (
        Feature,
        MegakernelSpec,
        Numerics,
        NumericsKind,
        Requirements,
        Step,
    )
except ModuleNotFoundError:  # SGLang before sgl-project/sglang#45389
    from sglang.srt.megakernels.spec import (
        Feature,
        MegakernelSpec,
        Numerics,
        NumericsKind,
        Requirements,
        Step,
    )

from sgl_nvidia_mega.kimi_k31 import spec as target
from sgl_nvidia_mega.kimi_k31.dflash.protocol import DRAFT_CHAINS, RUNTIME
from sgl_nvidia_mega.kimi_k31.k3.default_scope import best_default

# The SGLang names this module reads, by file under the sglang package; test/unit/test_contract.py pins them.
SGLANG_READS = {
    "kernels/plugin_kernels/nvidia/spec.py|srt/megakernels/spec.py": (
        "Feature",
        "MegakernelSpec",
        "Numerics",
        "NumericsKind",
        "Requirements",
        "Step",
    ),
    "srt/distributed/parallel_state.py": (
        "get_tp_group",
        "rank_in_group",
        "device_group",
        "ca_comm",
    ),
    "srt/layers/communication/k3_ar_fusion.py": ("_BUFS",),
    "srt/layers/layer_boundary/__init__.py": ("AttentionInputs", "get_attn_tp_context"),
    "srt/models/deepseek_common/attention_forward_methods/__init__.py": (
        "AttnForwardMethod",
    ),
    "srt/utils/common.py": ("BumpAllocator",),
    "srt/layers/logits_processor.py": ("LogitsProcessorOutput",),
    "srt/layers/layernorm.py": ("variance_epsilon",),
    # 2a head: the target lm_head's vocab shard and the draft config's unary transform
    "srt/layers/vocab_parallel_embedding.py": (
        "shard_indices",
        "num_org_elements",
        "org_vocab_start_index",
    ),
    "srt/speculative/dflash_utils.py": ("output_multiplier", "final_logit_softcapping"),
    # M1b: the core through the draft's own attention backend, and its KV pool
    "srt/model_executor/forward_context.py": ("get_attn_backend",),
    "srt/layers/attention/flashinfer_mla_backend.py": ("token_to_kv_pool",),
    "srt/mem_cache/memory_pool.py": (
        "MLATokenToKVPool",
        "kv_buffer",
        "start_layer",
        "dsa_kv_cache_store_fp8",
        "dcp_replicated",
        "write_loc_is_dcp_resolved",
    ),
    "srt/runtime_context.py": ("get_parallel", '"attn_dcp_size"'),
    "srt/model_executor/forward_batch_info.py": ("out_cache_loc",),
    # the draft's modules and the MLA sublayer the M1a hook drives
    "srt/models/dflash.py": (
        "attention_conv",
        "mlp_conv",
        "kernel_projection",
        "base_kernel",
        "input_layernorm",
        "post_attention_layernorm",
        "gate_up_proj",
        "down_proj",
        "norm",
        "fc",
        "fc_tp_columns",
        "hidden_norm",
        "candidate_selector",
        "hidden_projection",
        "predecessor_codebook",
        "successor_codebook",
        "top_k",
        "draft_config",
    ),
    "srt/models/deepseek_v2.py": (
        "forward_prepare",
        "forward_core",
        "fused_qkv_a_proj_with_mqa",
        "q_a_layernorm",
        "q_b_proj",
        "kv_a_layernorm",
        "kv_b_proj",
        "o_proj",
        "rotary_emb",
        "q_lora_rank",
        "attn_mqa",
    ),
}

logger = logging.getLogger(__name__)

ENV = "SGLANG_K31_DRAFT_MK"
ENABLED = best_default(ENV, "1", "0") not in ("", "0")
# Stage 2b: the worker's decode aux append (context features into the draft KV) on MK blocks too, and
# stage 2a: the draft graph's proposal (candidate top-k over the target lm_head + the selector walk, greedy
# or sampled per request); both run through the K3.1 DFlash runtime, so both build with it.
AUX = HEAD = ENABLED and RUNTIME
ROWS = 8  # one DFlash block per request; stage 1 serves B1
MLA_ROW = 576  # bytes per draft KV slot: 512 FP8 latent | 64 FP8 rotated k_pe

NUMERICS = Numerics(
    NumericsKind.TOLERANCE,
    {
        # the side-1 conv applied to each rank's partial before the all-reduce (SGLang: after it), plus the
        # fused hops' BF16 rounding points: relative to the draft's final hidden
        "draft_hidden.rel": 2**-6,
    },
)


class _NoOproj(torch.nn.Module):
    """Stands in for the attention's o_proj during the hook: the chain's hop does o_proj + all-reduce."""

    def forward(self, x):
        return x, None


class DraftLayersOnChain:
    """The draft spec's built object: ``DraftChain`` over the loaded DFlash2 draft's per-rank weights."""

    widths: tuple[int, ...] = ()
    draft_sampler = None
    # no speculative runtime hooks factory: SGLang keeps its own strategies
    speculative_runtime_hooks = None
    draft_loop = draft_extend = begin_draft_round = graph_variants = (
        None  # SGLang's own paths
    )
    # what the chain offers the K3.1 DFlash runtime (kimi_k31/dflash/runtime.py)
    draft_aux_rows: tuple[int, ...] = ()
    draft_head_requests: tuple[int, ...] = ()
    draft_head_sampling = (
        True  # the chain's DraftSelector walks greedy and sampled requests
    )

    def __init__(self, model, hf_config, ctx):
        from sglang.srt.distributed.parallel_state import get_tp_group

        from sgl_nvidia_mega.kimi_k31.sm103.draft.chain import (
            DraftChain,
            DraftComm,
            DraftShapes,
        )
        from sgl_nvidia_mega.kimi_k31.verify_executor import INSTANCES

        group = get_tp_group()
        self.model = model
        device = model.norm.weight.device
        shapes = DraftShapes(
            tp=group.world_size,
            rank=group.rank_in_group,
            eps=float(hf_config.rms_norm_eps),
            # the head box; a default of 16 against a K 32 selector raises in attach_draft_head (walk K check)
            topk=int(getattr(getattr(model, "candidate_selector", None), "top_k", 16)),
        )
        weights = [self._layer_weights(layer) for layer in model.layers]
        comm = DraftComm(group.device_group, device, shapes, layers=len(weights))
        cos_sin = model.layers[0].self_attn.rotary_emb.cos_sin_cache
        if not INSTANCES:
            raise RuntimeError(
                f"{ENV}: no K3.1 verify executor to check the draft mailboxes against"
            )
        targets = (*INSTANCES, self._ar_fusion_buffers(), group.ca_comm)
        self.chain = DraftChain(
            shapes,
            weights,
            model.norm.weight,
            cos_sin,
            comm,
            device=device,
            targets=targets,
        )
        self.out = torch.empty(
            ROWS, model.norm.weight.shape[0], dtype=torch.bfloat16, device=device
        )
        self.widths = (ROWS,) if ROWS in ctx.widths else ()
        self._stream: Optional[int] = None
        self._layers = len(weights)
        logger.info(
            "NVIDIA megakernel: DFlash2 draft layers on the MK chain (%s, B1 x %d rows, %d layers, TP%d)",
            ENV,
            ROWS,
            self._layers,
            shapes.tp,
        )
        logger.info(
            "NVIDIA megakernel: draft MK: %d target buffers disjoint",
            self.chain.target_buffers,
        )
        self.device = device
        self.aux = self._build_aux(model, shapes, comm, cos_sin) if AUX else None
        self.draft_aux_rows = (ROWS,) if self.aux is not None else ()
        self._pools: Optional[torch.Tensor] = None
        self._pool_ptrs: tuple[int, ...] = ()
        self._caches: Optional[list] = None
        self.draft_head_requests = (
            (1,) if HEAD and self._head_serves(model) else ()
        )  # the chain's head serves B1
        self.m1b = ENABLED
        # The target's DFlash runtime (dflash/runtime.py) serves only a draft chain built here.
        DRAFT_CHAINS.append(self)
        if self.m1b:
            logger.info(
                "NVIDIA megakernel: draft MK M1b: attention on MK blocks, core = SGLang's sliding-window MLA"
            )

    @staticmethod
    def _head_serves(model):
        """Whether the chain's head serves the draft's selector K (16 / 32); it declines otherwise."""
        from sgl_nvidia_mega.kimi_k31.sm103.draft.head import TOPKS

        top_k = getattr(getattr(model, "candidate_selector", None), "top_k", None)
        if top_k in TOPKS:
            return True
        logger.info(
            "NVIDIA megakernel: draft MK head off: selector top-k %s, the head serves %s",
            top_k,
            TOPKS,
        )
        return False

    @staticmethod
    def _build_aux(model, shapes, comm, cos_sin):
        """``DraftAux`` (aux.py) over this rank's SGLANG_OPT_DFLASH_TP_AUX_FC slice, on the chain's comm; None for
        a draft it does not serve."""
        from sgl_nvidia_mega.kimi_k31.sm103.draft.aux import DraftAux
        from sgl_nvidia_mega.kimi_k31.sm103.draft.chain import DEADLINE_NS
        from sgl_nvidia_mega.kimi_k31.sm103.draft.tactics import draft_tactic

        width = int(model.fc.weight.shape[0])
        attns = [layer.self_attn for layer in model.layers]
        norms = [model.hidden_norm, *(attn.kv_a_layernorm for attn in attns)]
        kv_a = [
            attn.fused_qkv_a_proj_with_mqa.weight[attn.q_lora_rank :] for attn in attns
        ]
        # GatherFused publishes rank r's slice as columns [r * width, (r + 1) * width).
        if model.fc_tp_columns != (shapes.rank * width, (shapes.rank + 1) * width):
            reason = f"it needs SGLANG_OPT_DFLASH_TP_AUX_FC's rank-ordered fc slice, got {model.fc_tp_columns}"
        elif any(float(norm.variance_epsilon) != shapes.eps for norm in norms):
            reason = f"hidden_norm and kv_a norms must share eps {shapes.eps}"
        elif any(
            tuple(w.shape) != (MLA_ROW, model.fc.weight.shape[0] * shapes.tp)
            for w in kv_a
        ):
            reason = f"kv_a rows {[tuple(w.shape) for w in kv_a]}"
        else:
            reason = None
        if reason is not None:
            logger.info("NVIDIA megakernel: draft MK aux off: %s", reason)
            return None
        aux = DraftAux(
            model.fc.weight,
            model.hidden_norm.weight,
            kv_a,
            [attn.kv_a_layernorm.weight for attn in attns],
            cos_sin,
            comm,
            shapes,
            eps=shapes.eps,
            deadline_ns=DEADLINE_NS,
        )
        device = model.fc.weight.device
        logger.info(
            "NVIDIA megakernel: draft MK aux: fc tactic %s, kv tactic %s, comm.aux parity 0 (rows %d)",
            draft_tactic("fc_slice", device),
            draft_tactic("kv_stack", device),
            shapes.tokens,
        )
        return aux

    @staticmethod
    def _ar_fusion_buffers():
        """The K3 all-reduce fusion's named symmetric buffers (a target family owned outside the executor)."""
        from sglang.srt.layers.communication.k3_ar_fusion import _BUFS

        return list(_BUFS)

    @staticmethod
    def _layer_weights(layer):
        from sgl_nvidia_mega.kimi_k31.sm103.draft.chain import DraftLayerWeights

        attn = layer.self_attn
        return DraftLayerWeights(
            input_norm=layer.input_layernorm.weight,
            attn_conv_proj=layer.attention_conv.kernel_projection.weight,
            attn_conv_base=layer.attention_conv.base_kernel,
            fused_a=attn.fused_qkv_a_proj_with_mqa.weight,
            q_a_norm=attn.q_a_layernorm.weight,
            q_b=attn.q_b_proj.weight,
            kv_a_norm=attn.kv_a_layernorm.weight,
            kv_b=attn.kv_b_proj.weight,
            o_proj=attn.o_proj.weight,
            post_norm=layer.post_attention_layernorm.weight,
            mlp_conv_proj=layer.mlp_conv.kernel_projection.weight,
            mlp_conv_base=layer.mlp_conv.base_kernel,
            gate_up=layer.mlp.gate_up_proj.weight,
            down=layer.mlp.down_proj.weight,
        )

    def summary(self) -> str:
        return (
            f"DFlash2 draft: {self._layers} layers on the MK chain at B1 x {ROWS} rows"
        )

    def plan(self, input_ids, forward_batch, *args, **kwargs):
        if not forward_batch.forward_mode.is_target_verify():
            return None
        rows = int(input_ids.shape[0])
        return Step(mode="target_verify", rows=rows) if rows in self.widths else None

    def _attention(self, positions, forward_batch):
        """The chain's M1a hook: SGLang's MLA sublayer of layer i on the conv-prepared input, up to W_UV."""
        from sglang.srt.layers.layer_boundary import (
            AttentionInputs,
            get_attn_tp_context,
        )
        from sglang.srt.models.deepseek_common.attention_forward_methods import (
            AttnForwardMethod,
        )
        from sglang.srt.utils import BumpAllocator

        def attn(i, x):
            module = self.model.layers[i].self_attn
            context = get_attn_tp_context()
            context.set_attn_inputs(
                AttentionInputs(x, forward_batch, module.prepare_qkv_latent)
            )
            o_proj = module.o_proj
            try:
                zero = BumpAllocator(
                    buffer_size=2, dtype=torch.float32, device=x.device
                )
                state = module.forward_prepare(
                    positions=positions,
                    hidden_states=x,
                    forward_batch=forward_batch,
                    zero_allocator=zero,
                )
                if state[1] != AttnForwardMethod.MLA:
                    raise RuntimeError(
                        f"{ENV}: the draft attention runs {state[1]}, not MLA"
                    )
                module.o_proj = _NoOproj()
                return module.forward_core(state)
            finally:
                module.o_proj = o_proj
                context.clear_attn_inputs()

        return attn

    def _core(self, forward_batch):
        """M1b's core: SGLang's sliding-window MLA over the draft cache, given DraftMlaQuery's FP8 query."""

        def core(i, query, latent):
            attn = self.model.layers[i].self_attn.attn_mqa
            out = attn(
                query.view(torch.float8_e4m3fn),
                None,
                None,
                forward_batch,
                save_kv_cache=False,
            )
            latent.copy_(out.view(latent.shape))

        return core

    def _kv(self, forward_batch):
        """M1b's KV: the draft layers' caches, the block's rows at its out_cache_loc (draft-pool physical)."""
        from sglang.srt.model_executor.forward_context import get_attn_backend

        from sgl_nvidia_mega.kimi_k31.sm103.draft.chain import DraftKV

        if self._caches is None:
            self._caches = self._draft_caches(get_attn_backend().token_to_kv_pool)
            if self._caches is None:  # declined: M1a from here on
                self.m1b = False
                return None
        slots = forward_batch.out_cache_loc.view(-1, ROWS)
        return DraftKV(
            pool=self._caches, ctx_len=0, window_slots=slots, block_slots=slots
        )

    def _draft_caches(self, pool):
        from sglang.srt.mem_cache.memory_pool import MLATokenToKVPool
        from sglang.srt.runtime_context import get_parallel

        # DraftMlaQuery writes whole FP8 rows at slot * 576 bytes: SGLang's plain layout (3111 byte check).
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
            logger.info(
                "NVIDIA megakernel: draft MK M1b off: the draft KV pool is not plain FP8 MLA (%s); M1a",
                type(pool).__name__,
            )
            return None
        return [
            pool.kv_buffer[layer.self_attn.attn_mqa.layer_id - pool.start_layer]
            for layer in self.model.layers
        ]

    def forward(
        self,
        step,
        original,
        input_ids,
        positions,
        forward_batch,
        input_embeds=None,
        *args,
        **kwargs,
    ):
        from sglang.srt.layers.logits_processor import LogitsProcessorOutput

        capturing = torch.cuda.is_current_stream_capturing()
        # Only captured graphs can replay concurrently (pdmux stream groups); the runner serializes the eager
        # runs, e.g. the autotune dummy run on forward_stream before the warmups on the capture stream.
        if capturing:
            stream = torch.cuda.current_stream().cuda_stream
            if self._stream is None:
                self._stream = stream
            elif stream != self._stream:
                raise RuntimeError(
                    f"{ENV}: a draft hop from a second stream (all-reduce program order)"
                )
        if input_embeds is None or input_embeds.shape != self.out.shape:
            raise RuntimeError(
                f"{ENV}: input_embeds {None if input_embeds is None else tuple(input_embeds.shape)}"
            )
        x0 = input_embeds
        # The head proposes from the block's rows 1..7; row 0 is the anchor (a no-copy view at B1).
        head = (
            {"anchor": input_ids.view(-1, ROWS)[:, 0]}
            if self.chain.head is not None
            else {}
        )
        kv = self._kv(forward_batch) if self.m1b else None
        if kv is not None:
            self.chain.run(
                x0,
                positions,
                kv,
                self.out,
                0,
                core=self._core(forward_batch),
                **head,
            )
        else:
            self.chain.run(
                x0,
                positions,
                None,
                self.out,
                0,
                attn=self._attention(positions, forward_batch),
                **head,
            )
        hidden = self.out if capturing else self.out.clone()
        return LogitsProcessorOutput(next_token_logits=None, hidden_states=hidden)

    def attach_draft_head(self, *, lm_head, selector, draft_config):
        """The runtime's draft sampler: the chain's candidate head over this rank's target lm_head shard
        and the draft's selector; returns the selector's static (tokens, candidates, q)."""
        from sgl_nvidia_mega.kimi_k31.sm103.draft.chain import DraftHeadWeights

        shard = lm_head.shard_indices
        num_org = int(shard.num_org_elements)
        self.chain.attach_head(
            DraftHeadWeights(
                lm_head=lm_head.weight[:num_org],
                hidden_projection=selector.hidden_projection.weight,
                vocab_start=int(shard.org_vocab_start_index),
                predecessor=selector.predecessor_codebook,
                successor=selector.successor_codebook,
                output_multiplier=float(draft_config.output_multiplier),
                final_logit_softcapping=draft_config.final_logit_softcapping,
            )
        )
        walk = self.chain.selector
        if walk.candidates.shape[-1] != selector.top_k:
            raise RuntimeError(
                f"draft MK head: head top-k {walk.candidates.shape[-1]} vs the selector's {selector.top_k}"
            )
        return walk.tokens, walk.candidates, walk.q

    def stage_draft_head_sampling(self, *, temperatures, greedy_mask):
        """The runtime's draft sampler: the selector's host refresh before a draft forward."""
        self.chain.selector.stage_sampling(
            temperatures=temperatures, greedy_mask=greedy_mask
        )

    def draft_aux(
        self, *, target_hidden, positions, slots_2d, commit_lens, kv_caches, parity
    ):
        """The runtime's aux append: the decode aux append on MK blocks (DraftAux)."""
        if target_hidden.shape[0] != self.aux.rows:
            raise RuntimeError(
                f"draft MK aux: target_hidden {tuple(target_hidden.shape)}"
            )
        # parity is the gather's one-hop mailbox slot; SGLang alternates it over every call, eager or replayed.
        pools = self._pool_table(kv_caches, torch.cuda.is_current_stream_capturing())
        self.aux.run(
            target_hidden, positions, slots_2d, commit_lens, pools, parity=parity
        )

    def _pool_table(self, caches, capturing):
        """int64 [layers]: each draft layer's cache base address, which the captured graph reads."""
        ptrs = tuple(int(cache.data_ptr()) for cache in caches)
        if ptrs == self._pool_ptrs:
            return self._pools
        if capturing:
            raise RuntimeError(
                "draft MK aux: the draft KV caches changed after the eager warm-up"
            )
        if len(caches) != self.aux.layers:
            raise RuntimeError(
                f"draft MK aux: {len(caches)} draft KV caches for {self.aux.layers} layers"
            )
        for cache in caches:
            if (
                cache.element_size() != 1
                or tuple(cache.shape[1:]) != (1, MLA_ROW)
                or not cache.is_contiguous()
                or cache.device != self.device
            ):
                raise RuntimeError(
                    f"draft MK aux: draft KV cache {cache.dtype} {tuple(cache.shape)}"
                )
        if self._pools is None:
            self._pools = torch.empty(len(ptrs), dtype=torch.int64, device=self.device)
        self._pools.copy_(torch.tensor(ptrs, dtype=torch.int64))
        self._pool_ptrs = ptrs
        return self._pools


def _build(model, hf_config, ctx) -> DraftLayersOnChain:
    return DraftLayersOnChain(model, hf_config, ctx)


SPEC = MegakernelSpec(
    name="kimi_k31_dflash2_draft_chain_nvidia",
    # the target's devices: the draft shares its GPUs; stage 1 is built for TP16 full decode graphs
    requires=Requirements(
        isas=target.SPEC.requires.isas,
        tp_sizes=frozenset({16}),
        decode_graph_backends=frozenset({"full"}),
    ),
    features=frozenset({Feature.SPECULATIVE_DECODING}),
    numerics=NUMERICS,
    # 0 (unset flag): SGLang does not wrap the draft's forward
    max_rows=ROWS if ENABLED else 0,
    build=_build,
    api_version=target.SPEC.api_version,
)
