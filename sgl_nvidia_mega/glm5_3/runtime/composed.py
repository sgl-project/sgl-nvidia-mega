# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""The covered verify step of the kernel-unit profiles (TP4 at 3-6 tokens per request,
up to 48 rows; ``profile.UNITS``): each width's step groups are composed from the
kernel units that cover it (``registry``, ``units``), the steps no unit takes run the
stock bodies, and the layer loop, the first-capture checks and the parity switch are
the executor's (``executor.GlmMegakernelVerify``). The logits run the LM head on
``ll.gemm``'s lm_head role where it was built.

A width no unit covers runs SGLang's forward; in parity mode
(``SGLANG_NVIDIA_MEGAKERNEL_SLOTS=""``) every accepted width runs the layer loop on the
stock bodies.
"""

from __future__ import annotations

import logging

from sgl_nvidia_mega.glm5_3 import ll, registry, units
from sgl_nvidia_mega.glm5_3.attention.step import AttentionStep
from sgl_nvidia_mega.glm5_3.boundary.boundaries import HopSteps
from sgl_nvidia_mega.glm5_3.moe.step import MlpSteps
from sgl_nvidia_mega.glm5_3.plan import Providers
from sgl_nvidia_mega.glm5_3.profile import Profile
from sgl_nvidia_mega.glm5_3.runtime.executor import GlmMegakernelVerify

# The SGLang names this module reads (test/unit/test_contract.py pins them)
SGLANG_READS = {
    "srt/runtime_context.py": ("get_parallel", '"tp_rank"'),
    "srt/kernel_plugins/nvidia/binding.py|srt/megakernels/runtime.py": (
        "megakernel_agree",
        "megakernel_agree(what)",
    ),
    "srt/layers/logits_processor.py": ("LogitsProcessor", "_logits_gatherer"),
    "srt/models/deepseek_v2.py": ("logits_processor", "lm_head", "embed_tokens"),
}

logger = logging.getLogger(__name__)

MODE = "target_verify"
# The stock step group of each group (the registry composes over them; the
# executor's Providers defaults)
STOCK = {"attention": AttentionStep, "hops": HopSteps, "mlp": MlpSteps}
# the stock step groups' class names (logs: which steps a kernel serves)
_STOCK_NAMES = ("AttentionStep", "HopSteps", "MlpSteps")


class ComposedVerify(GlmMegakernelVerify):
    """``GlmMegakernelVerify`` on step groups composed from the kernel units, whose
    logits run the LM head on ``ll.gemm``'s lm_head role (``verify_logits_processor``)."""

    def __init__(self, lm, providers, families, *, logits_processor, **kwargs):
        super().__init__(lm, providers, families, **kwargs)
        self.logits_processor = logits_processor

    def forward(
        self,
        step,
        original,
        input_ids,
        positions,
        forward_batch,
        input_embeds=None,
        pp_proxy_tensors=None,
    ):
        import torch

        from sgl_nvidia_mega.glm5_3.runtime import parity

        if not torch.cuda.is_current_stream_capturing():
            # plan() covers eager steps only under the test-only parity switch
            return parity.run(
                self, step, original, input_ids, positions, forward_batch, input_embeds
            )
        hidden = self._run(
            self.providers[step.rows], input_ids, positions, forward_batch, step.view
        )
        return self.logits_processor(
            input_ids, hidden, self.lm.lm_head, forward_batch, None
        )


def verify_logits_processor(model, widths, profile: Profile):
    """The covered verify steps' logits processor: SGLang's, with the LM head on
    ``ll.gemm``'s lm_head role (unit ``ll.gemm.lm_head``) at the widths it was built
    for; the model's own processor when that role serves none of them. It shares the
    model processor's logits all-gather: SGLang builds that gatherer's multimem state
    lazily on its first eager call and never under CUDA-graph capture, and the covered
    verify step only runs under capture, so a gatherer of its own would leave the
    verify graphs on NCCL's all-gather (the decode graph runner's eager warmups build
    the model processor's)."""
    from sglang.srt.layers.logits_processor import LogitsProcessor

    gemm = ll.module("gemm")
    stock = model.logits_processor
    if gemm is None or not widths or type(stock) is not LogitsProcessor:
        return stock
    ll.build_linear("lm_head", model.lm_head, widths, sm=profile.sm, sms=profile.sms)
    if not any(gemm.built("lm_head", model.lm_head, w) for w in widths):
        return stock

    class GlmVerifyLogitsProcessor(gemm.LmHeadMixin, LogitsProcessor):
        pass

    processor = GlmVerifyLogitsProcessor(model.config)
    processor._logits_gatherer = stock._logits_gatherer
    return processor


def build(model, ctx, profile: Profile) -> ComposedVerify:
    """``runtime.build`` for a kernel-unit profile: the compositions per width, the
    families they take (built and compiled now, before the KV cache is sized), and the
    executor. A family that fails on any rank, or ranks that composed differently,
    stop every rank."""
    from sglang.srt.runtime_context import get_parallel

    try:
        from sglang.srt.kernel_plugins.nvidia.binding import megakernel_agree
    except ModuleNotFoundError:  # SGLang before sgl-project/sglang#45389
        from sglang.srt.megakernels.runtime import megakernel_agree

    from sgl_nvidia_mega.glm5_3 import runtime
    from sgl_nvidia_mega.glm5_3.runtime.compile import compile_stats

    if not profile.widths:
        raise RuntimeError(
            f"NVIDIA megakernel: GLM-5.3 / GLM-5.2: widths {list(ctx.widths)} hold "
            f"no verify width of TP{profile.tp} at {profile.tokens_per_request} rows "
            "a request"
        )
    rank = get_parallel().tp_rank
    derived = runtime.DerivedWeights()
    candidates, factories = registry.collect(registry.target_modules())
    made = registry.build(
        model,
        derived,
        profile,
        steps=[(MODE, rows) for rows in profile.widths],
        groups=STOCK,
        candidates=candidates,
        families=factories,
        gather=runtime.tp_gather,
        rank=rank,
    )
    parity = units.parity()
    providers = {
        rows: Providers(**made.groups[(MODE, rows)])
        for rows in profile.widths
        if parity or not made.compositions[(MODE, rows)].all_stock()
    }
    if not providers:
        raise RuntimeError(
            "NVIDIA megakernel: GLM-5.3 / GLM-5.2: no kernel unit serves any width "
            f"of {list(profile.widths)} (TP{profile.tp}); leave the flag off, or see "
            "the 'stock at N rows' lines above"
        )
    # The cross-GPU steps pair up across ranks: every rank serves the same groups.
    megakernel_agree(
        {m: p.describe() for m, p in providers.items()},
        what="GLM-5.3 / GLM-5.2 step groups per width",
    )
    widths = tuple(sorted(providers))
    derived.log(rank)
    logger.info(
        "NVIDIA megakernel: GLM-5.3 / GLM-5.2: kernels compiled %d, loaded from "
        "cache %d (rank %d), not stored in the cache %d",
        compile_stats["compiled"],
        compile_stats["cached"],
        rank,
        compile_stats["not_stored"],
    )
    if rank == 0:
        for rows, p in providers.items():
            fused = {
                k: v for k, v in p.describe().items() if not v.startswith(_STOCK_NAMES)
            }
            logger.info(
                "NVIDIA megakernel: GLM-5.3 / GLM-5.2: %d verify rows: %s",
                rows,
                fused or "all steps stock",
            )
    runtime.log_coverage(
        widths, profile.tokens_per_request, sum(derived.mib().values()), rank
    )
    errors = runtime.ErrorReporter()
    for family in made.families.values():
        if callable(getattr(family, "error_report", None)):
            errors.add(family.error_report)
    hops = made.families.get("ll.hops")
    if "draft" in runtime.SLOTS and getattr(hops, "buffer", None) is not None:
        # the TP4 draft sums its row-parallel eh_proj in these hops (ll.draft)
        runtime.register_target_hops(model.model.embed_tokens.weight, hops)
    taken = sorted({c.name for comp in made.compositions.values() for c in comp.chosen})
    stock = [rows for rows in profile.widths if rows not in providers]
    summary = (
        f"TP{profile.tp} kernel units {taken or 'none (stock)'}; derived weight copies "
        f"{sum(derived.mib().values()):.1f} MiB; kernels compiled "
        f"{compile_stats['compiled']}, from cache {compile_stats['cached']}"
        + (f"; stock at {', '.join(map(str, stock))} rows" if stock else "")
    )
    return ComposedVerify(
        model,
        providers,
        made.families,
        widths=widths,
        derived=derived,
        errors=errors,
        summary=summary,
        binding=getattr(ctx, "binding", None),
        logits_processor=verify_logits_processor(model, widths, profile),
    )
