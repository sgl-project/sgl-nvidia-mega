# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""K3.1 decode with SGLang-selected dense/sparse graph variants."""

from sglang.srt.runtime_context import get_model

from sgl_nvidia_mega.kimi_k31.k3.spec import WIDTHS, _Built, _deadline_ns

from .dsa_executor import SparseDecodeExecutor

# SGLang names this module reads (test/unit/test_contract.py pins them)
SGLANG_READS = {"srt/runtime_context.py": ("get_model",)}


class BuiltSparseDecode(_Built):
    tensor_binding_kind = 32

    def __init__(self, model, resources, widths):
        super().__init__(model, resources, widths)
        self.indexers = {
            int(layer.layer_idx): resources.attention[int(layer.layer_idx)].core
            for layer in resources.layers
            if hasattr(layer.self_attn, "q_b_proj")
        }
        self.executor = SparseDecodeExecutor(resources)
        model.__dict__["_nvidia_kimi_k31_megakernel"] = self

    def summary(self):
        tp = 96 // self.resources.heads
        return f"TP{tp} decode rows 1-8/16/32/64; dense <=2048, FP8 paged DSA top2048 above; folded index-K"

    def run_decode(self, input_ids, embeddings, view):
        view.validate()
        return self.executor.run(input_ids, embeddings, view)


def build_sparse_decode(model, hf_config, context):
    from sgl_nvidia_mega.kimi_k31.k3.compat import validate_runtime_compatibility
    from sgl_nvidia_mega.kimi_k31.k3.providers import _cuda_memory, build_resources

    from sglang.srt.runtime_context import get_parallel

    from sgl_nvidia_mega.kimi_k31.k3.shapes import heads as tp_heads

    from .dsa.prepared import PreparedSparseIndexMla
    from .projection_storage import coalesce_index_projection, pitch

    widths = tuple(width for width in context.widths if width in WIDTHS)
    if context.tokens_per_request != 1 or not widths:
        raise ValueError("K3.1 sparse DSA requires non-spec decode at supported row widths")
    if context.binding is None:
        raise ValueError("K3.1 sparse DSA needs its SGLang tensor binding")
    validate_runtime_compatibility()
    resources = build_resources(
        model.language_model,
        context_length=int(get_model().context_length),
        memory_before=_cuda_memory(),
        deadline_ns=_deadline_ns(context.deadline_s),
        mla_pitch=pitch(tp_heads(int(get_parallel().tp_size))),
        mla_storage=coalesce_index_projection,
        mla_factory=PreparedSparseIndexMla,
    )
    return BuiltSparseDecode(model, resources, widths)
