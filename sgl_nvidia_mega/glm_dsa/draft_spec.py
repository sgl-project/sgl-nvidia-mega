# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""The GLM-5 DSA NextN draft's spec: the fused draft layer (``draft.FusedDraft``) in
the draft-decode and draft-extend CUDA graphs.

The draft kernels are compiled and their mailboxes rendezvous once the target's
embedding and LM head are shared with the draft (``set_embed_and_head``, after the
draft's build). ``SGLANG_NVIDIA_MEGAKERNEL_GLM_FUSED_DRAFT=0`` keeps SGLang's draft.
"""

from __future__ import annotations

from sglang.srt.megakernels.spec import (
    Feature,
    MegakernelSpec,
    Numerics,
    NumericsKind,
    Requirements,
)

from sgl_nvidia_mega.glm_dsa import spec as target
from sgl_nvidia_mega.glm_dsa import switches

# The SGLang names this module reads (test/unit/test_contract.py pins them)
SGLANG_READS = {
    "srt/megakernels/spec.py": (
        "Feature",
        "MegakernelSpec",
        "Numerics",
        "NumericsKind",
        "Requirements",
        "Step",
    ),
    "srt/models/deepseek_v2.py": ("set_embed_and_head", "logits_processor", "lm_head"),
}

# The draft top-1 is the stock argmax of the same logits (the distributed argmax
# keeps its NaN and tie rules); the embedding front is bitwise fused_eh_norm.
NUMERICS = Numerics(
    kind=NumericsKind.TOLERANCE,
    tolerances={"draft.top1.max_abs": 0.0, "draft.embed_eh_norm.max_abs": 0.0},
)


class _BuiltDraft:
    """The installed draft forward: armed steps run the fused draft layer."""

    draft_sampler = None

    def __init__(self, model):
        from sgl_nvidia_mega.glm_dsa import draft

        self.model = model
        # draft decode: one row per request; draft extend: one per verify token
        self.widths = draft.served_eh_rows()
        self.fused = draft.create_fused_draft(model)
        model.logits_processor = draft.make_draft_logits_processor(
            model.config, self.fused
        )
        share = model.set_embed_and_head

        def set_embed_and_head(embed, head):
            share(embed, head)
            self.fused.arm(model)

        # an instance attribute: the class and every other instance keep theirs
        model.set_embed_and_head = set_embed_and_head

    def summary(self) -> str:
        return "GLM-5 NextN draft: fused embedding front, eh_proj shard, distributed top-1"

    def plan(self, input_ids, forward_batch, *args, **kwargs):
        from sglang.srt.megakernels.spec import Step

        pp_proxy = kwargs.get("pp_proxy_tensors", args[0] if args else None)
        if pp_proxy is not None or not self.fused.engaged():
            return None
        mode = "decode" if forward_batch.forward_mode.is_decode() else "draft_extend"
        return Step(mode=mode, rows=int(input_ids.shape[0]))

    def forward(self, step, original, input_ids, positions, forward_batch, *a, **kw):
        hidden_states = self.fused.run(self.model.model, input_ids, positions, forward_batch)
        return self.model.logits_processor(
            input_ids, hidden_states, self.model.lm_head, forward_batch
        )


class _StockDraft:
    """SGLANG_NVIDIA_MEGAKERNEL_GLM_FUSED_DRAFT=0: no step, SGLang's draft forward."""

    draft_sampler = None
    widths = ()

    def summary(self) -> str:
        return "GLM-5 NextN draft: SGLang's forward"

    def plan(self, *args, **kwargs):
        return None

    def forward(self, step, original, *args, **kwargs):
        raise AssertionError("the stock draft covers no step")


def _build(model, hf_config, ctx):
    return _BuiltDraft(model) if switches.fused_draft() else _StockDraft()


SPEC = MegakernelSpec(
    name="GLM-5 DSA NextN draft",
    # the target's device, TP size, dtype and decode graphs: the draft shares its GPUs
    requires=Requirements(
        isas=target.REQUIREMENTS.isas,
        tp_sizes=target.REQUIREMENTS.tp_sizes,
        dtypes=target.REQUIREMENTS.dtypes,
        decode_graph_backends=target.REQUIREMENTS.decode_graph_backends,
        env=target.REQUIREMENTS.env,
    ),
    features=frozenset({Feature.SPECULATIVE_DECODING, Feature.MULTI_NODE_TP}),
    numerics=NUMERICS,
    # 0 keeps SGLang's draft forward (no step covered, nothing installed)
    # the draft extend at the widest covered width (draft.served_eh_rows)
    max_rows=target.MAX_ROWS if switches.fused_draft() else 0,
    build=_build,
    api_version=target.API_VERSION,
)
