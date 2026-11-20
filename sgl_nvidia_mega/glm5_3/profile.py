# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""The resolved facts a GLM-5.3 / GLM-5.2 build is decided on (``Profile``) and the
accepted (TP size, tokens per request) table.

A profile is resolved once per model at build (target or draft) from the server
arguments, the HF config and SGLang's ``BuildContext``; at argument resolution the
same facts come from the server arguments alone (``Profile.from_args``). Every
kernel candidate decides its coverage from a profile, a width (verify rows) and a
step mode (``registry.Candidate.covers``), never from the device data, so every
rank decides alike.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, List, Optional, Tuple

from sgl_nvidia_mega.glm5_3 import TP_SIZE
from sgl_nvidia_mega.glm5_3.plan import MAX_ROWS as PLAN_MAX_ROWS
from sgl_nvidia_mega.glm5_3.plan import SERVED_TOKENS_PER_REQUEST

# The SGLang names this module reads (test/unit/test_contract.py pins them)
SGLANG_READS = {
    "srt/runtime_context.py": ("get_server_args",),
    "srt/arg_groups/model_override_base.py": ("resolved_view",),
    "kernels/plugin_kernels/nvidia/spec.py|srt/megakernels/spec.py": (
        "BuildContext",
        "widths",
        "tokens_per_request",
        "deadline_s",
    ),
    "srt/arg_groups/fields/*.py": (
        "tp_size",
        "nnodes",
        "speculative_algorithm",
        "speculative_num_draft_tokens",
        "speculative_num_steps",
        "enable_hierarchical_cache",
        "disaggregation_decode_enable_offload_kvcache",
        "enable_lmcache",
        "enable_flexkv",
        "disaggregation_mode",
    ),
}

# (attention TP size, tokens per request) -> the widest verify step (rows) the plugin
# serves; wider steps run SGLang's forward. The TP8 envelope (4 or 6 tokens, up to 16
# rows: plan.py) builds its families through plan.build_providers; the TP4 profiles
# run the kernel units (registry, units.py) up to 48 rows (8 requests at 6 tokens: a
# PD decode role at --max-running-requests 24 runs 92-95% of its steps within that).
TP8 = {(TP_SIZE, t): PLAN_MAX_ROWS for t in SERVED_TOKENS_PER_REQUEST}
UNITS = {(4, t): 48 for t in (3, 4, 5, 6)}
ACCEPTED = {**TP8, **UNITS}
# The spec's max_rows: SGLang asks for the widths range(tokens, MAX_ROWS + 1, tokens)
MAX_ROWS = 48
# GLM-5 attention heads over all ranks (eligibility pins the checkpoint shapes)
ATTENTION_HEADS = 64
# SGLang's bound on every cross-GPU wait when the context gives none (guards.MAX_DEADLINE_S)
DEFAULT_DEADLINE_S = 60.0


@dataclass(frozen=True)
class Profile:
    """The facts a build is decided on, resolved once per model."""

    tp: int  # attention TP size = the hop world (no DP attention, no PP)
    heads: int  # attention heads per rank (64 // tp)
    tokens_per_request: (
        int  # speculative_num_draft_tokens (MTP chain: steps + 1, topk 1)
    )
    widths: Tuple[int, ...]  # the requested widths clipped to ACCEPTED[(tp, tokens)]
    sm: Tuple[int, int]  # compute capability, e.g. (10, 3)
    sms: int  # streaming multiprocessors of this rank's GPU
    kv_offload: bool = False  # HiCache / KV offload (Feature.KV_OFFLOAD)
    pd_decode: bool = False  # the PD-disaggregation decode role
    multi_node: bool = False  # TP over more than one node (NVL72 trays)
    # every cross-GPU wait's deadline (BuildContext.deadline_s, below the watchdog)
    deadline_s: float = DEFAULT_DEADLINE_S

    @property
    def units(self) -> bool:
        """Whether this profile runs the kernel units (TP4), not the TP8 families."""
        return (self.tp, self.tokens_per_request) in UNITS

    @property
    def max_rows(self) -> int:
        """The widest verify step served at this (tp, tokens); 0 if not accepted."""
        return accepted_rows(self.tp, self.tokens_per_request)

    @classmethod
    def from_args(
        cls, server_args: Any, hf_config: Any, *, device_capability=None
    ) -> Profile:
        """The profile of a target build on these resolved settings (argument
        resolution: ``config_disable_reasons``); widths as SGLang will request them."""
        tp = int(getattr(server_args, "tp_size", None) or 1)
        tokens = tokens_per_request(server_args)
        sm, sms = _device(device_capability)
        heads_total = int(getattr(hf_config, "num_attention_heads", ATTENTION_HEADS))
        return cls(
            tp=tp,
            heads=heads_total // tp if tp else heads_total,
            tokens_per_request=tokens,
            widths=clip_widths(tuple(range(tokens, MAX_ROWS + 1, tokens)), tp, tokens),
            sm=sm,
            sms=sms,
            kv_offload=kv_offload(server_args),
            pd_decode=getattr(server_args, "disaggregation_mode", None) == "decode",
            multi_node=int(getattr(server_args, "nnodes", None) or 1) > 1,
        )

    @classmethod
    def from_build(cls, model: Any, ctx: Any) -> Profile:
        """The profile of ``spec.build(model, hf_config, ctx)``: the requested widths
        (``ctx.widths``) clipped to the accepted envelope, and the context's deadline."""
        p = cls.from_args(resolved_server_args(), model.config)
        tokens = int(getattr(ctx, "tokens_per_request", p.tokens_per_request))
        widths = tuple(getattr(ctx, "widths", p.widths))
        return replace(
            p,
            tokens_per_request=tokens,
            widths=clip_widths(widths, p.tp, tokens),
            deadline_s=float(getattr(ctx, "deadline_s", None) or DEFAULT_DEADLINE_S),
        )


def resolved_server_args() -> Any:
    """This server's arguments as resolution decided them: a model's resolved
    defaults (e.g. GLM-5's fp8 KV cache, page 64, DSA / TRT-LLM backends) are
    declarations over the record, whose own fields keep the raw input ('auto')."""
    from sglang.srt.arg_groups.model_override_base import resolved_view
    from sglang.srt.runtime_context import get_server_args

    return resolved_view(get_server_args())


def accepted_rows(tp: int, tokens: int) -> int:
    return ACCEPTED.get((int(tp), int(tokens)), 0)


def clip_widths(widths, tp: int, tokens: int) -> Tuple[int, ...]:
    """``widths`` up to the accepted envelope of (tp, tokens), ascending."""
    top = accepted_rows(tp, tokens)
    return tuple(sorted(w for w in set(widths) if 0 < w <= top))


def tokens_per_request(server_args: Any) -> int:
    if getattr(server_args, "speculative_algorithm", None) is None:
        return 1
    return int(getattr(server_args, "speculative_num_draft_tokens", None) or 1)


def kv_offload(server_args: Any) -> bool:
    """SGLang's KV_OFFLOAD feature probe (megakernels/guards.py FEATURE_PROBES)."""
    return bool(
        getattr(server_args, "enable_hierarchical_cache", False)
        or getattr(server_args, "disaggregation_decode_enable_offload_kvcache", False)
        or getattr(server_args, "enable_lmcache", False)
        or getattr(server_args, "enable_flexkv", False)
    )


def profile_reasons(server_args: Any) -> List[str]:
    """Why the (tp, tokens per request) pairing is outside the accepted envelope
    (empty if it is inside): an MTP chain (steps = tokens - 1) at an ACCEPTED pair."""
    tp = int(getattr(server_args, "tp_size", None) or 1)
    tokens = tokens_per_request(server_args)
    steps = getattr(server_args, "speculative_num_steps", None)
    reasons = []
    if getattr(server_args, "speculative_algorithm", None) is not None and (
        steps is None or int(steps) != tokens - 1
    ):
        reasons.append(
            f"requires an MTP chain: --speculative-num-steps = "
            f"--speculative-num-draft-tokens - 1 (got {steps!r} steps, {tokens} tokens)"
        )
    if (tp, tokens) not in ACCEPTED:
        pairs = ", ".join(f"TP{t} {k} tokens" for t, k in sorted(ACCEPTED))
        reasons.append(
            f"TP{tp} with {tokens} tokens per request is not served (served: {pairs})"
        )
    return reasons


def _device(capability: Optional[Tuple[int, int]]) -> Tuple[Tuple[int, int], int]:
    """(compute capability, SM count) of this process's GPU; ``capability`` overrides
    the capability (tests, argument resolution without a GPU)."""
    import torch

    sms = 0
    if torch.cuda.is_available():
        dev = torch.cuda.current_device()
        if capability is None:
            capability = torch.cuda.get_device_capability(dev)
        sms = int(torch.cuda.get_device_properties(dev).multi_processor_count)
    return tuple(capability or (0, 0)), sms
