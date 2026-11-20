# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Component ``embed_gather`` (env ``MUSE_VR_EMBED_GATHER=1``, default off).

With SGLang's ``SGLANG_ENABLE_EMBED_REPLICATION=1`` the input embedding holds the full table on every rank and looks
tokens up without an all-reduce (the DFLASH draft block uses the same module). That replicated (``tp_size == 1``) lookup
goes through ``F.embedding``. This component looks replicated modules up with the Triton row gather the sharded path uses
(``sglang.kernels.ops.embeddings.vocab_parallel_embedding``), with the module's shard indices (which cover the whole
vocabulary when ``tp_size`` is 1). Tunable choice; timing: the census A/B of the serving recipe.

Mechanism: an AROUND hook on ``VocabParallelEmbedding._embed_local_shard``. For a module with ``tp_size == 1``, the
unquantized embedding method, a 2-D unit-stride CUDA weight in a floating dtype, and contiguous CUDA int32/int64 ids, it
returns the Triton gather (cast to ``output_dtype`` when the module has one, as the stock code does); every other module
and call runs stock. The gather copies rows: in-vocabulary ids give the table rows bit for bit, as ``F.embedding`` does
(tests/test_embed_gather.py checks every id of the served table).
"""

from __future__ import annotations

import logging

import torch

from . import bed

# The SGLang names this module reads (test/unit/test_contract.py pins them against the supported SGLang)
SGLANG_READS = {
    "srt/layers/quantization/unquant.py": ("UnquantizedEmbeddingMethod",),
    "kernels/ops/embeddings/vocab_parallel_embedding.py": ("vocab_parallel_embedding",),
    "srt/plugins/hook_registry.py": ("HookRegistry", "HookType"),
}


logger = logging.getLogger(__name__)

NAME = "embed_gather"  # env MUSE_VR_EMBED_GATHER
TARGET = "sglang.srt.layers.vocab_parallel_embedding.VocabParallelEmbedding._embed_local_shard"

PINS = {
    "sglang/srt/layers/vocab_parallel_embedding.py": (
        "5c4a96a8ad291f7601876368cec75769adc152b6af6fffe317ae5fc44d10d4d1",
        "c36f5682e528fafe922b88ad0e9dd418b7f9558762058858707a6a75c25ea5cf",
    ),
    "sglang/kernels/ops/embeddings/vocab_parallel_embedding.py": "7766e50514e621317fec277fd5975adc469d70ab2951722a99d56afbb6dc16d4",
}


def enabled() -> bool:
    return bed.enabled(NAME)


def pins_match() -> tuple[bool, str]:
    for rel, want in PINS.items():
        if not bed.pinned(rel, want):
            return False, f"{rel} differs from the pinned revision"
    return True, "ok"


def _eligible(embed, input_) -> bool:
    from sglang.srt.layers.quantization.unquant import UnquantizedEmbeddingMethod

    w = getattr(embed, "weight", None)
    return (
        getattr(embed, "tp_size", None) == 1
        and isinstance(getattr(embed, "quant_method", None), UnquantizedEmbeddingMethod)
        and getattr(embed, "shard_indices", None) is not None
        and isinstance(input_, torch.Tensor)
        and input_.is_cuda
        and input_.is_contiguous()
        and input_.dtype in (torch.int32, torch.int64)
        and w is not None
        and w.is_cuda
        and w.ndim == 2
        and w.stride(1) == 1
        and w.dtype in (torch.float16, torch.bfloat16, torch.float32)
    )


def around_embed_local_shard(original, embed, *args, **kwargs):
    if kwargs or len(args) != 1 or not _eligible(embed, args[0]):
        return original(embed, *args, **kwargs)
    from sglang.kernels.ops.embeddings.vocab_parallel_embedding import (
        vocab_parallel_embedding,
    )

    si = embed.shard_indices
    out = vocab_parallel_embedding(
        args[0],
        embed.weight,
        si.org_vocab_start_index,
        si.org_vocab_end_index,
        si.num_org_vocab_padding,
        si.added_vocab_start_index,
        si.added_vocab_end_index,
    )
    if getattr(embed, "output_dtype", None) is not None:
        out = out.to(embed.output_dtype)
    if not torch.cuda.is_current_stream_capturing():
        bed.count(NAME, "gather")
    return out


def register() -> bool:
    if not enabled():
        return False
    ok, why = pins_match()
    if not ok:
        bed.report(NAME, False, reason=why)
        return False
    from sglang.srt.plugins.hook_registry import HookRegistry, HookType

    HookRegistry.register(TARGET, around_embed_local_shard, HookType.AROUND)
    bed.report(NAME, True)
    return True
