# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""K3.1 packed-MXFP4 decode while DSA selects the complete causal context."""

from __future__ import annotations

import os
from typing import Any

import msgspec
from sgl_nvidia_mega.kimi_k31.k3._host import BuildContext, MegakernelSpec, Feature
from sglang.srt.runtime_context import get_model, get_parallel

from sgl_nvidia_mega.kimi_k31.k3 import preload

from .dsa.verify import build_from_tensors

from sgl_nvidia_mega.kimi_k31.k3.spec import (
    SPEC as K3_SPEC,
)
from sgl_nvidia_mega.kimi_k31.k3.spec import (
    WIDTHS,
    _Built,
    _deadline_ns,
)
from sgl_nvidia_mega.kimi_k31.k3.spec import (
    _model_disable_reasons as _k3_model_disable_reasons,
)

SGLANG_READS = {
    "kernels/plugin_kernels/nvidia/spec.py|srt/megakernels/spec.py": (
        "BuildContext",
        "MegakernelSpec",
        "prepare",
    ),
    "srt/runtime_context.py": ("get_model", "get_parallel"),
    "srt/arg_groups/fields/*.py": (
        "attn_dcp_size",
        "dcp_size",
        "tp_size",
        "attention_backend",
        "prefill_attention_backend",
        "decode_attention_backend",
        "dsa_prefill_backend",
        "dsa_decode_backend",
        "enable_linear_replayssm",
        "enable_linear_replayssm_spec",
    ),
    "srt/models/kimi_k3.py": ("KimiK31ForConditionalGeneration",),
    "srt/kernel_plugins/nvidia/binding.py|srt/megakernels/binding.py": ("hybrid", "indexer", "caches", "page_size"),
}


def _text_config(hf_config: Any) -> Any:
    return getattr(hf_config, "text_config", hf_config)


def _dsa_reasons(text: Any) -> list[str]:
    dsa = getattr(text, "dsa_config", None)
    if not isinstance(dsa, dict):
        return ["Kimi K3.1 requires its original dsa_config"]
    expected = {
        "index_topk": 2048,
        "index_num_heads": 32,
        "index_head_dim": 128,
        "index_rope_dim": 0,
        "index_k_norm_type": "layernorm",
        "index_topk_freq": 1,
        "indexer_use_fp8": False,
    }
    reasons = [
        f"Kimi K3.1 dsa_config.{key} must be {value!r}, got {dsa.get(key)!r}"
        for key, value in expected.items()
        if dsa.get(key, False if key == "indexer_use_fp8" else None) != value  # a missing indexer_use_fp8 reads as False
    ]
    full_ids = (getattr(text, "linear_attn_config", None) or {}).get(
        "full_attn_layers", ()
    )
    if tuple(dsa.get("dsa_layers", ())) != tuple(full_ids):
        reasons.append("Kimi K3.1 needs a DSA indexer on every MLA layer")
    if int(getattr(text, "index_kpool", 1)) != 1:
        reasons.append("Kimi K3.1 dense-equivalent decode requires index_kpool=1")
    return reasons


def _context_reasons(context_length: int, text: Any) -> list[str]:
    if context_length <= 0:
        return [f"Kimi K3.1 decode requires positive context_length, got {context_length}"]
    return []


def _dcp_reasons(cfg: Any, spec: bool) -> list[str]:
    """DCP > 1 serves decode (dcp/build.py) and verify (dcp/verify.py) only with K31MK_DCP=1 and DCP = TP."""
    dcp = int(getattr(cfg, "dcp_size", 1) or 1)
    if dcp == 1:
        return []
    if dcp != int(cfg.tp_size) or os.environ.get("K31MK_DCP", "0") != "1":
        return [f"K3.1 MK DCP needs K31MK_DCP=1 and DCP = TP, got DCP{dcp} / TP{cfg.tp_size}"]
    return []


def _config_disable_reasons(cfg: Any, hf_config: Any) -> list[str]:
    text = _text_config(hf_config)
    if getattr(cfg, "speculative_algorithm", None) is not None:
        from .verify_spec import config_disable_reasons

        return _dsa_reasons(text) + _dcp_reasons(cfg, True) + config_disable_reasons(cfg, hf_config)
    reasons = _dsa_reasons(text) + _context_reasons(int(cfg.context_length or 0), text)
    reasons += _dcp_reasons(cfg, False)
    if getattr(cfg, "enable_linear_replayssm", False) or getattr(
        cfg, "enable_linear_replayssm_spec", False
    ):
        reasons.append("non-spec K31 decode does not support buffered recurrent state")
    if cfg.cuda_graph_config is None or cfg.cuda_graph_config.decode.backend != "full":
        reasons.append("Kimi K3.1 needs the FULL decode CUDA-graph backend")
    # The dev host leaves split fields unset when --attention-backend is used.
    prefill = cfg.prefill_attention_backend or cfg.attention_backend
    decode = cfg.decode_attention_backend or cfg.attention_backend
    if (prefill, decode) != ("dsa", "dsa"):
        reasons.append(
            f"Kimi K3.1 requires effective DSA prefill/decode backends, got {(prefill, decode)}"
        )
    return reasons


def _model_disable_reasons(model: Any, hf_config: Any) -> list[str]:
    reasons = _k3_model_disable_reasons(model, hf_config)
    text = _text_config(hf_config)
    reasons += _dsa_reasons(text)
    language_model = getattr(model, "language_model", None)
    if language_model is not None:
        for layer in language_model.model.layers:
            if (
                hasattr(layer.self_attn, "q_b_proj")
                and getattr(layer.self_attn, "indexer", None) is None
            ):
                reasons.append(f"Kimi K3.1 layer {layer.layer_idx} has no DSA indexer")
    return reasons


class _BuiltK31(_Built):
    def __init__(self, model: Any, resources: Any, widths: tuple[int, ...]):
        super().__init__(model, resources, widths)
        from sgl_nvidia_mega.kimi_k31.k3.executor import KimiK3MegakernelExecutor

        self.indexers = {
            int(layer.layer_idx): resources.attention[int(layer.layer_idx)].core
            for layer in resources.layers
            if hasattr(layer.self_attn, "q_b_proj")
        }
        self.executor = KimiK3MegakernelExecutor(resources)
        model.__dict__["_nvidia_kimi_k31_megakernel"] = self

    def summary(self) -> str:
        tp = 96 // self.resources.heads
        return f"TP{tp} one-token decode; context <= index_topk=2048; folded index-K writes active"


def _prepare(built: Any, view: Any) -> None:
    if getattr(built, "tensor_binding_kind", 0) == 31:
        built.prepare(view)
    else:
        if view.hybrid is None or view.hybrid.indexer is None:
            raise ValueError("Kimi K3.1 needs typed MLA/KDA/index-K pool buffers")
        for layer_id, indexer in built.indexers.items():
            indexer.bind_index_cache(
                view.hybrid.indexer.caches[layer_id], view.hybrid.indexer.page_size
            )
        built.resources.bind(view.hybrid)
        built.resources.refresh_inventory()
    preload.at_prepare()


def _build(model: Any, hf_config: Any, context: BuildContext) -> _BuiltK31:
    if context.tokens_per_request != 1:
        if context.tokens_per_request not in (8, 16):
            raise ValueError("Kimi K3.1 verify serves Q8/Q16")
        raise RuntimeError("K31 verify requires the SGLang tensor-weight build patch")
    widths = tuple(width for width in context.widths if width in WIDTHS)
    if not widths or context.binding is None:
        raise ValueError("Kimi K3.1 needs supported widths and a runtime binding")
    length = int(get_model().context_length)
    reasons = _dsa_reasons(_text_config(hf_config)) + _context_reasons(
        length, _text_config(hf_config)
    )
    if reasons:
        raise ValueError("; ".join(reasons))
    from .dcp.attention import DCP_ENABLED

    if DCP_ENABLED and int(getattr(get_parallel(), "attn_dcp_size", 1)) > 1:
        # DCP = TP: sharded MLA KV / index-K, global top-k merge, LSE merge. Dense (<= 2048) and
        # sparse graphs share this build (dense graphs attend every visible key).
        from .dcp.build import build_dcp_decode

        return build_dcp_decode(model, hf_config, context)
    if length > 2048:
        from .dsa_spec import build_sparse_decode

        return build_sparse_decode(model, hf_config, context)
    from sgl_nvidia_mega.kimi_k31.k3.compat import validate_runtime_compatibility
    from sgl_nvidia_mega.kimi_k31.k3.providers import _cuda_memory, build_resources

    from sglang.srt.runtime_context import get_parallel

    from sgl_nvidia_mega.kimi_k31.k3.shapes import heads as tp_heads

    from .folded_index import PreparedIndexMla
    from .projection_storage import coalesce_index_projection, pitch

    validate_runtime_compatibility()
    resources = build_resources(
        model.language_model,
        context_length=length,
        memory_before=_cuda_memory(),
        deadline_ns=_deadline_ns(context.deadline_s),
        mla_pitch=pitch(tp_heads(int(get_parallel().tp_size))),
        mla_storage=coalesce_index_projection,
        mla_factory=PreparedIndexMla,
    )
    return _BuiltK31(model, resources, widths)


def _verify_build(weights, deadline_s):
    from .dcp.attention import DCP_ENABLED

    if DCP_ENABLED and int(getattr(get_parallel(), "attn_dcp_size", 1)) > 1:
        from .dcp.verify import build_from_tensors as build_dcp_verify

        return build_dcp_verify(weights, deadline_s)
    return build_from_tensors(weights, deadline_s)


def _features() -> frozenset:
    """DCP (``K31MK_DCP=1``, DCP = TP, non-spec decode) also admits the host's CONTEXT_PARALLEL probe.
    ``K31MK_PD=1`` (default 0 = off) admits a PD decode instance (--disaggregation-mode decode): the megakernel serves
    decode / verify only and reads SGLang's typed MLA / KDA / index-K pools, which the PD transfer fills; a prefill
    instance runs without the megakernel flag."""
    # Admission patches written by another NVIDIA team (this one and the indexer_use_fp8 default in _dsa_reasons):
    # a PD decode worker runs the same decode and TARGET_VERIFY steps on the KV its prefill worker transferred.
    features = K3_SPEC.features | frozenset({Feature.SPECULATIVE_DECODING, Feature.PD_DISAGGREGATION})
    if os.environ.get("K31MK_DCP", "0") == "1":
        features |= frozenset({Feature.CONTEXT_PARALLEL})
    if os.environ.get("K31MK_PD", "0") == "1":
        features |= frozenset({Feature.PD_DISAGGREGATION})
    return features


SPEC: MegakernelSpec = msgspec.structs.replace(
    K3_SPEC,
    name="Kimi K3.1 packed-MXFP4 decode / TARGET_VERIFY Q8/Q16",
    features=_features(),
    tensor_verify_binding_kind=31,
    tensor_verify_build=_verify_build,
    requires=msgspec.structs.replace(
        K3_SPEC.requires,
        server_args={
            **{
                key: values
                for key, values in K3_SPEC.requires.server_args.items()
                if key not in ("prefill_attention_backend", "decode_attention_backend")
            },
            "dsa_prefill_backend": frozenset({"trtllm"}),
            "dsa_decode_backend": frozenset({"trtllm"}),
            "enable_linear_replayssm": frozenset({False}),
            "enable_linear_replayssm_spec": frozenset({False, True}),
        },
    ),
    build=_build,
    prepare=_prepare,
    config_disable_reasons=_config_disable_reasons,
    model_disable_reasons=_model_disable_reasons,
)
