# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the SGLang project
"""Prepare the local vocabulary gather without changing native TP semantics.

Default OFF retains native embedding without optional gather/output preparation.
Explicit opt-in uses supported BF16 vocabulary shards, retaining native fallback
for other layouts and scattered inputs. It neither replicates weights nor removes
TP communication. Bounded short q1/q6 checks do not establish a speedup or qualify
combined optional providers and long-context ON behavior.
"""

from __future__ import annotations

import torch

SGLANG_READS = {
    "srt/layers/vocab_parallel_embedding.py": (
        "VocabParallelEmbedding",
        "enable_tp",
        "tp_size",
        "tp_group",
        "use_attn_tp_group",
        "quant_method",
        "output_dtype",
        "num_added_embeddings",
        "num_embeddings",
        "org_vocab_size",
        "num_embeddings_per_partition",
        "shard_indices",
        "org_vocab_start_index",
        "org_vocab_end_index",
        "padded_org_vocab_start_index",
        "padded_org_vocab_end_index",
        "num_org_elements_padded",
    ),
    "srt/layers/quantization/unquant.py": ("UnquantizedEmbeddingMethod",),
    "srt/layers/layer_boundary/__init__.py": ("get_attn_tp_context",),
    "srt/layers/layer_boundary/adapters/attention.py": ("input_scattered",),
    "srt/utils/async_probe.py": ("maybe_detect_oob",),
    "srt/distributed/parallel_state.py": ("world_size", "all_reduce"),
}


class PreparedInputEmbedding:
    def __init__(self, embedding, core):
        from sglang.srt.layers.layer_boundary import get_attn_tp_context
        from sglang.srt.utils.async_probe import maybe_detect_oob

        self.embedding = embedding
        self.core = core
        self.get_attn_tp_context = get_attn_tp_context
        self.maybe_detect_oob = maybe_detect_oob

    def run(self, ids, out):
        # Native scattered-input execution deliberately omits the TP reduction.
        if self.get_attn_tp_context().input_scattered:
            return self.embedding(ids)
        self.maybe_detect_oob(
            ids, 0, self.embedding.num_embeddings, "VocabParallelEmbedding input id"
        )
        return self.core.run(ids, out)


def prepare_embedding(embedding, *, tp_group, load_prepared):
    """Return a prepared gather, or None to retain the native embedding call.

    No replicated weights, added vocabulary, quantized lookup or attention-TP
    policy is inferred. Padding uses the native shard's actual valid bounds,
    not rank times shard size. The model continues owning its weight storage.
    """
    from sglang.srt.layers.quantization.unquant import UnquantizedEmbeddingMethod
    from sglang.srt.layers.vocab_parallel_embedding import VocabParallelEmbedding

    if tp_group is None or type(embedding) is not VocabParallelEmbedding:
        return None
    weight = embedding.weight
    if not (
        embedding.enable_tp
        and embedding.tp_group is tp_group
        and embedding.tp_size == tp_group.world_size > 1
        and not embedding.use_attn_tp_group
        and type(embedding.quant_method) is UnquantizedEmbeddingMethod
        and embedding.output_dtype in (None, torch.bfloat16)
        and embedding.num_added_embeddings == 0
        and embedding.num_embeddings == embedding.org_vocab_size
        and weight.ndim == 2
        and weight.shape[1] == 7168
        and weight.dtype == torch.bfloat16
        and weight.is_contiguous()
    ):
        return None
    shard = embedding.shard_indices
    start, end = shard.org_vocab_start_index, shard.org_vocab_end_index
    if not (
        0 <= start <= end <= embedding.org_vocab_size
        and start == min(shard.padded_org_vocab_start_index, embedding.org_vocab_size)
        and end == min(shard.padded_org_vocab_end_index, embedding.org_vocab_size)
        and shard.num_org_elements_padded
        == embedding.num_embeddings_per_partition
        == weight.shape[0]
        and end - start <= weight.shape[0]
    ):
        return None
    core = load_prepared()(weight, start=start, end=end, group=tp_group)
    return PreparedInputEmbedding(embedding, core)
