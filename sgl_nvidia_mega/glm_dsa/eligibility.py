# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 SGLang Team
"""When the fused GLM-5 DSA decode / verify path engages.

* Tier 1, at construction (``check_construction``): server arguments and
  checkpoint shapes; on opt-in an unsupported setting raises ``ValueError``
  naming every reason.
* Tier 2, after loading (``check_loaded``): facts of the built, processed modules.
* Tier 3, per forward (``step_in_graph_runner``): host-only facts of the batch,
  never device data, so a captured graph and its replays agree. At the first
  capture of each row count ``attention_step_reasons`` checks that the stock
  attention takes the branches the attention steps copy (agreed across ranks).
"""

from __future__ import annotations

from typing import List

import torch

# Verify rows per forward: 1-4 requests x 4 tokens (3 draft tokens + 1).
VERIFY_TOKENS_PER_REQUEST = 4
SUPPORTED_ROWS = (4, 8, 12, 16)
TP_SIZE = 8
HIDDEN = 6144
_SM_ACCEPTED = ((10, 3), (10, 7))
# what the refusals name as the switch that asked for the path
SWITCH = "--enable-nvidia-megakernel (GLM-5 DSA)"


# GLM-5 DSA checkpoint shapes the path is built for.
_EXPECTED_SHAPES = {
    "hidden_size": HIDDEN,
    "num_attention_heads": 64,
    "q_lora_rank": 2048,
    "kv_lora_rank": 512,
    "qk_rope_head_dim": 64,
    "qk_nope_head_dim": 192,
    "v_head_dim": 256,
    "index_n_heads": 32,
    "index_head_dim": 128,
    "index_topk": 2048,
    "n_routed_experts": 256,
    "num_experts_per_tok": 8,
    "n_shared_experts": 1,
    "moe_intermediate_size": 2048,
    "first_k_dense_replace": 3,
    "num_hidden_layers": 78,
}

# Server arguments the path is validated at: name -> accepted values (the last
# one is the one to pass; None = unset).
_VALUES = {
    "tp_size": (TP_SIZE,),
    **dict.fromkeys(
        ("pp_size", "dp_size", "ep_size", "dcp_size", "attn_cp_size", "moe_dp_size"),
        (None, 1),
    ),
    "moe_a2a_backend": (None, "none"),
    "kv_cache_dtype": ("fp8_e4m3",),
    "page_size": (64,),
    "attention_backend": ("dsa",),
    "decode_attention_backend": (None, "dsa"),
    "dsa_decode_backend": ("trtllm",),
    "moe_runner_backend": ("flashinfer_trtllm",),
    "init_expert_location": (None, "trivial"),
    "expert_distribution_recorder_mode": (None,),
}
_EAGLE_VALUES = {
    "speculative_eagle_topk": (1,),
    "speculative_num_draft_tokens": (VERIFY_TOKENS_PER_REQUEST,),
    "speculative_num_steps": (VERIFY_TOKENS_PER_REQUEST - 1,),
}
# Server arguments that must be off: features, layouts and communication
# variants that change the verify step's stock op sequence.
_OFF = tuple(
    """enable_dp_attention enable_prefill_cp enable_cp_decode_attn_tp
    enable_kv_cache_sharding enable_layernorm_sp enable_attn_tp_input_scattered
    enable_dense_mlp_attn_tp enable_shared_experts_attn_tp enable_quant_communications
    enable_fused_moe_sum_all_reduce enable_dp_lm_head enable_hisparse
    enable_dsa_cache_layer_split enforce_shared_experts_fusion enable_eplb
    ep_num_redundant_experts enable_return_routed_experts enable_return_indexer_topk
    enable_lora lora_paths enable_deterministic_inference enable_two_batch_overlap
    enable_single_batch_overlap enable_torch_compile disable_cuda_graph
    disable_decode_cuda_graph""".split()
)
_EAGLE_OFF = ("speculative_adaptive", "enable_multi_layer_eagle")
# Reasons the generated "requires --x <value>" / "--x is not supported" would
# misstate (no such flag, or a hint lost); said once when two flags share one.
_WHY = {
    "expert_distribution_recorder_mode": (
        "the expert distribution recorder is not supported"
    ),
    "enforce_shared_experts_fusion": (
        "requires shared-experts fusion off (--disable-shared-experts-fusion)"
    ),
    **dict.fromkeys(
        ("disable_cuda_graph", "disable_decode_cuda_graph"),
        "requires decode CUDA graphs (the path runs only inside captured full graphs)",
    ),
}


def _mismatches(server_args, values, off) -> List[str]:
    def flag(name):
        return "--" + name.replace("_", "-")

    out = []
    for name, ok in values.items():
        got = getattr(server_args, name, None)
        if got not in ok:
            out.append(_WHY.get(name, f"requires {flag(name)} {ok[-1]} (got {got!r})"))
    out += [
        _WHY.get(n, f"{flag(n)} is not supported")
        for n in off
        if getattr(server_args, n, None)
    ]
    return list(dict.fromkeys(out))


def _sm_reason(cap) -> List[str]:
    if cap is None and torch.cuda.is_available():
        cap = torch.cuda.get_device_capability()
    if tuple(cap or ()) in _SM_ACCEPTED:
        return []
    return [
        f"requires an SM103 or SM107 GPU (found compute capability {cap}); "
        "other architectures are not validated"
    ]


def _ar_fusion_ok(applies, rows: int) -> bool:
    from sgl_nvidia_mega.glm_dsa import switches

    return applies(rows) or switches.allow_no_ar_fusion()


def _refuse(switch: str, what: str, reasons: List[str]) -> None:
    if reasons:
        raise ValueError(
            f"{switch} is set but {what} is not supported:\n  - "
            + "\n  - ".join(reasons)
        )


def construction_reasons(config, server_args, *, device_capability=None) -> List[str]:
    """Every reason the configuration is not supported (empty if supported)."""
    from sglang.srt.environ import envs

    reasons = _sm_reason(device_capability)
    if getattr(config, "model_type", None) != "glm_moe_dsa":
        reasons.append(
            f"model_type {getattr(config, 'model_type', None)!r} is not glm_moe_dsa"
        )
    for key, want in _EXPECTED_SHAPES.items():
        if getattr(config, key, None) != want:
            reasons.append(
                f"config.{key} = {getattr(config, key, None)!r}, expected {want!r}"
            )
    reasons += _mismatches(server_args, _VALUES, _OFF)
    # Speculative decoding: EAGLE / MTP chain, 3 draft tokens (NEXTN resolves to EAGLE).
    algo = getattr(server_args, "speculative_algorithm", None)
    if algo is None:
        reasons.append(
            "requires speculative decoding (EAGLE / NEXTN with the MTP layer); "
            "plain decode is not supported"
        )
    elif str(algo).upper() not in ("EAGLE", "NEXTN"):
        reasons.append(
            f"speculative algorithm {algo!r} is not supported (EAGLE / NEXTN only)"
        )
    else:
        reasons += _mismatches(server_args, _EAGLE_VALUES, _EAGLE_OFF)
    # The attention steps implement the bf16 absorbed query and the fused DSA
    # top-k page table (providers.AttentionDecode).
    if envs.SGLANG_EXPERIMENTAL_LORA_OPTI.get():
        reasons.append("SGLANG_EXPERIMENTAL_LORA_OPTI is not supported")
    if envs.SGLANG_ENABLE_DSA_Q8KV8_BORN_FP8_Q.get():
        reasons.append("SGLANG_ENABLE_DSA_Q8KV8_BORN_FP8_Q is not supported")
    if not envs.SGLANG_DSA_FUSE_TOPK.get():
        reasons.append("requires SGLANG_DSA_FUSE_TOPK (the fused DSA top-k page table)")
    # The path runs only inside captured full decode graphs.
    cfg = getattr(server_args, "cuda_graph_config", None) or {}
    entry = cfg.get("decode") if isinstance(cfg, dict) else getattr(cfg, "decode", None)
    backend = (
        entry.get("backend")
        if isinstance(entry, dict)
        else getattr(entry, "backend", None)
    )
    if backend not in (None, "full"):
        reasons.append(f"requires the full decode CUDA graph backend (got {backend!r})")
    return reasons


def model_construction_reasons(model, config=None) -> List[str]:
    """Tier 1 for a constructed model: ``construction_reasons`` of its config (or
    ``config``) and the running server's arguments, and its shared-experts fusion."""
    from sglang.srt.runtime_context import get_server_args

    reasons = construction_reasons(config or model.config, get_server_args())
    if getattr(model, "num_fused_shared_experts", 0) != 0:
        reasons.append(
            "requires shared-experts fusion off (num_fused_shared_experts == 0; "
            "pass --disable-shared-experts-fusion)"
        )
    return reasons


def check_construction(model) -> None:
    """Tier 1. Raise ``ValueError`` listing every unsupported setting."""
    _refuse(SWITCH, "the configuration", model_construction_reasons(model))


def check_loaded(model) -> None:
    """Tier 2. Raise ``ValueError`` listing every unsupported fact of the loaded model."""
    from sglang.srt.layers.flashinfer_comm_fusion import uses_cutedsl_ar_fusion
    from sglang.srt.layers.layer_boundary.residual.add_norm import (
        flashinfer_ar_fusion_applies,
    )
    from sglang.srt.layers.moe.utils import post_experts_output_is_complete
    from sglang.srt.models.deepseek_v2 import DeepseekV2MLP, DeepseekV2MoE

    reasons: List[str] = []
    mm = model.model
    if getattr(model, "capture_aux_hidden_states", False):
        reasons.append(
            "auxiliary hidden-state capture (EAGLE3 / DFLASH) is not supported"
        )
    if mm.start_layer != 0 or mm.end_layer != model.config.num_hidden_layers:
        reasons.append("pipeline-parallel layer ranges are not supported")
    for i in range(mm.start_layer, mm.end_layer):
        attn, mlp = mm.layers[i].self_attn, mm.layers[i].mlp
        if getattr(attn, "use_deep_gemm_bmm", False):
            reasons.append(f"layer {i}: DeepGEMM MLA bmm is not supported")
        w_vc = getattr(attn, "w_vc", None)
        if w_vc is None or w_vc.dtype != torch.bfloat16:
            reasons.append(
                f"layer {i}: W_UV must be bf16 (got {getattr(w_vc, 'dtype', None)})"
            )
        if getattr(attn, "learnable_sink_param", None) is not None:
            reasons.append(f"layer {i}: attention sinks are not supported")
        if getattr(attn, "dcp_size", 1) != 1:
            reasons.append(f"layer {i}: decode context parallelism is not supported")
        if i < model.config.first_k_dense_replace:
            if not isinstance(mlp, DeepseekV2MLP):
                reasons.append(f"layer {i}: expected a dense MLP layer")
            continue
        if not isinstance(mlp, DeepseekV2MoE):
            reasons.append(f"layer {i}: expected a MoE layer")
        if getattr(mlp, "num_fused_shared_experts", 0) != 0:
            reasons.append(
                f"layer {i}: shared experts are fused into the routed experts"
            )
        if getattr(mlp, "shared_experts", None) is None:
            reasons.append(f"layer {i}: no shared-expert module")
        method = type(getattr(mlp.experts, "quant_method", None)).__name__
        if method != "ModelOptNvFp4FusedMoEMethod":
            reasons.append(f"layer {i}: routed experts must be NVFP4 (got {method})")
    if not isinstance(mm.layers[mm.end_layer - 1].mlp, DeepseekV2MoE):
        reasons.append("the last layer must be a MoE layer")
    sliced = [
        i
        for i in range(mm.start_layer, mm.end_layer)
        if getattr(mm.layers[i].attn_boundary, "input_on_attn_tp_slices", False)
    ]
    if sliced:
        reasons.append(
            f"layer {sliced[0]}: attention input on attention-TP slices is not supported"
        )
    if getattr(mm, "gemm_output_zero_allocator_size", 0):
        reasons.append("the GEMM output zero allocator is not supported")
    # The residual steps are the fused all-reduce + residual + RMSNorm the layer
    # boundaries use at verify sizes.
    if uses_cutedsl_ar_fusion():
        reasons.append("the CuTe DSL all-reduce fusion backend is not supported")
    elif not _ar_fusion_ok(flashinfer_ar_fusion_applies, max(SUPPORTED_ROWS)):
        reasons.append(
            "requires the FlashInfer all-reduce + RMSNorm fusion at verify sizes "
            "(--flashinfer-allreduce-fusion-backend not disabled; without NVLink "
            "multicast set SGLANG_NVIDIA_MEGAKERNEL_GLM_ALLOW_NO_AR_FUSION=1)"
        )
    if getattr(mm, "llama_4_scaling_config", None) is not None:
        reasons.append("llama-4 attention scaling is not supported")
    # The fused steps perform the MLP output's tensor-parallel sum, as the stock
    # residual step does; a MoE combine that already completes it is out.
    if post_experts_output_is_complete(is_tp_path=True):
        reasons.append(
            "the MoE combine already completes the tensor-parallel sum "
            "(DWDP, FP4 all-gather dispatch or an all-to-all backend)"
        )
    _refuse(SWITCH, "the loaded model", reasons)


def absorbed_mla(model, forward_batch) -> bool:
    """Whether stock attention runs this batch on the absorbed MLA path (the one
    the attention steps implement)."""
    from sglang.srt.models.deepseek_common.attention_forward_methods.forward_methods import (
        AttnForwardMethod,
    )

    attn = model.model.layers[model.model.start_layer].self_attn
    return attn.dispatch_attn_forward_method(forward_batch) == AttnForwardMethod.MLA


def attention_step_reasons(model, forward_batch) -> List[str]:
    """Why stock verify attention would leave the branches ``providers.AttentionDecode``
    copies (empty if it would not); read at the first capture of each row count,
    once the backend exists."""
    from sglang.srt.layers.attention.dsa_backend import DeepseekSparseAttnBackend
    from sglang.srt.model_executor.forward_context import get_attn_backend
    from sglang.srt.models.deepseek_common.attention_forward_methods.forward_methods import (
        AttnForwardMethod,
    )

    backend = get_attn_backend()
    if not isinstance(backend, DeepseekSparseAttnBackend):
        return [f"attention backend {type(backend).__name__} is not the DSA backend"]
    reasons: List[str] = []
    if backend.dsa_decode_impl != "trtllm" or backend.use_mha:
        reasons.append(
            f"verify attention is not the TRT-LLM sparse MLA kernel "
            f"(impl {backend.dsa_decode_impl!r}, use_mha {backend.use_mha})"
        )
    if backend.kv_cache_dtype != torch.float8_e4m3fn:
        reasons.append(f"KV cache dtype {backend.kv_cache_dtype} is not fp8_e4m3")
    if not backend.use_fused_topk:
        reasons.append("the fused DSA top-k page table is off")
    elif not (
        backend.dsa_topk_backend.is_sgl_kernel()
        or backend.dsa_topk_backend.is_flashinfer()
    ):
        reasons.append(f"DSA top-k backend {backend.dsa_topk_backend} is not supported")
    if backend.dcp_size != 1 or backend.dsa_index_kpool > 1:
        reasons.append(
            "decode context parallelism / index K-pool layouts are not supported"
        )
    if backend.hisparse_coordinator is not None:
        reasons.append("HiSparse is not supported")
    if forward_batch.attn_cp_metadata is not None:
        reasons.append("attention context parallelism is not supported")

    mm = model.model
    for i in range(mm.start_layer, mm.end_layer):
        attn = mm.layers[i].self_attn
        indexer = getattr(attn, "indexer", None)
        why = []
        if attn.dispatch_attn_forward_method(forward_batch) != AttnForwardMethod.MLA:
            why.append("not absorbed MLA")
        if attn.q_lora_rank is None or attn.rotary_emb is None:
            why.append("no q_lora / RoPE")
        if not attn._fuse_rope_for_trtllm_mla(forward_batch):
            why.append("RoPE is not fused into the FP8 query quantization")
        if attn._can_fuse_bmm_into_attention(forward_batch):
            why.append("the W_UK bmm is fused into attention")
        if getattr(attn, "w_kc", None) is None or attn.w_kc.dtype != torch.bfloat16:
            why.append("W_UK is not bf16")
        if getattr(attn, "_kimi_split_gguf_kv_b", False) or hasattr(
            attn, "prepare_attention_output_gate"
        ):
            why.append("unsupported attention variant")
        if attn.attn_mqa.is_cross_attention:
            why.append("cross attention")
        if attn.skip_topk is None or (not attn.skip_topk and indexer is None):
            why.append("no DSA indexer")
        elif not attn.skip_topk:
            if not indexer.use_dsa_indexer_fusion:
                why.append("the indexer projections are not fused")
            if (
                indexer.use_gfx950_fused_indexer
                or indexer.dsa_enable_prefill_cp
                or indexer._aiter_fused_fp8_active(forward_batch)
            ):
                why.append("unsupported indexer path")
        if why:
            reasons.append(f"layer {i}: " + ", ".join(why))
            if len(reasons) > 8:
                break
    return reasons


def step_in_graph_runner(
    model, rows: int, forward_batch, input_embeds=None, pp_proxy_tensors=None
) -> bool:
    """Every Tier 3 condition but the capture itself (``runtime.engages``): an
    eligible verify forward inside the decode graph runner's full-graph capture
    procedure (its eager warmups included)."""
    from sglang.srt.layers.layer_boundary.residual.add_norm import (
        flashinfer_ar_fusion_applies,
    )
    from sglang.srt.model_executor.runner_backend_utils.breakable_cuda_graph.context import (
        is_in_breakable_cuda_graph,
    )
    from sglang.srt.model_executor.runner_utils.capture_mode import get_is_capture_mode

    return (
        forward_batch.forward_mode.is_target_verify()
        and rows in SUPPORTED_ROWS
        and input_embeds is None
        and pp_proxy_tensors is None
        and not getattr(forward_batch, "can_run_tbo", False)
        and not getattr(model, "capture_aux_hidden_states", False)
        and _ar_fusion_ok(flashinfer_ar_fusion_applies, rows)
        and absorbed_mla(model, forward_batch)
        and get_is_capture_mode()
        and not is_in_breakable_cuda_graph()
    )
