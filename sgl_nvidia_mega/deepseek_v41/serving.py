# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""The KV layout the DeepSeek-V4.1 megakernel picks for one-request serving.

Slot attn_core_lean reads SGLang's ``v41_new`` KV layout, which suits one request at
attention TP8 + MoE TP4 x EP2 and nothing else: under it swap-AB declines, so verify steps
past 8 rows (more than one request) and TP4 run SGLang's slower attention.

So, on a server with ``--enable-nvidia-megakernel`` serving DeepSeek-V4.1 at ``--tp 8
--ep-size 2`` with ``--max-running-requests 1`` and no PD disaggregation, this module's
resolution step sets ``SGLANG_DSV4_KV_LAYOUT=v41_new`` when the user left it unset and
slot attn_core_lean is requested, ahead of SGLang's megakernel admission
(``handle_nvidia_megakernel``), and logs it.

Slot moe_balanced needs every expert-parallel rank to hold all routed experts
(``--ep-num-redundant-experts`` equal to the routed count). That costs weight memory and
KV pool, so the user chooses it; on the same configuration without the flag this step
only logs that moe_balanced is idle and the flag that turns it on.
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

# The SGLang names this module reads (test/unit/test_contract.py pins them)
SGLANG_READS = {
    "srt/arg_groups/resolution_hooks.py": (
        "register_resolution_hook",
        '"handle_nvidia_megakernel"',
    ),
    "srt/arg_groups/model_override_base.py": ("resolving_view", "model_config_of"),
    "srt/environ.py": ("envs", "SGLANG_DSV4_KV_LAYOUT"),
    "srt/configs/model_config.py": ("hf_config",),
    "srt/arg_groups/fields/*.py": (
        "enable_nvidia_megakernel",
        "max_running_requests",
        "disaggregation_mode",
        "tp_size",
        "ep_size",
        "ep_num_redundant_experts",
        "enable_eplb",
        "init_expert_location",
    ),
}

# SGLang's resolution step this one runs ahead of
STEP = "handle_nvidia_megakernel"
# the (tp_size, ep_size) and request count the settings suit
TOPOLOGY = (8, 2)
REQUESTS = 1
# the weights every GPU gains holding all routed experts (DeepSeek-V4.1-Flash, measured)
REDUNDANT_GB = 47


_registered = False


def register() -> None:
    """Register the resolution step once (plugin.register); an SGLang without plugin
    resolution steps keeps the settings the user passes."""
    global _registered
    try:
        from sglang.srt.arg_groups.resolution_hooks import register_resolution_hook
    except ImportError:
        return
    if not _registered:
        register_resolution_hook(STEP)(_step)
        _registered = True


def _step(server_args, previous) -> None:
    choose(server_args)
    previous(server_args)


def one_request_tp8(cfg) -> bool:
    """The server serves one request at a time at attention TP8 + MoE TP4 x EP2."""
    return (
        (cfg.tp_size, cfg.ep_size or 1) == TOPOLOGY
        and cfg.max_running_requests == REQUESTS
        and (cfg.disaggregation_mode or "null") == "null"
    )


def choose(server_args) -> list[str]:
    """Set the KV layout one-request serving takes and the user left unset; return
    what was set."""
    from sglang.srt.arg_groups.model_override_base import (
        model_config_of,
        resolving_view,
    )

    from sgl_nvidia_mega.deepseek_v41.spec import ARCHITECTURE

    cfg = resolving_view(server_args)
    if not cfg.enable_nvidia_megakernel:
        return []
    hf_config = model_config_of(server_args).hf_config
    if (
        ARCHITECTURE not in (hf_config.architectures or ())
        or getattr(hf_config, "model_type", None) != "deepseek_v41"
        or not one_request_tp8(cfg)
    ):
        return []
    chosen = _layout()
    for what in chosen:
        logger.info(
            "NVIDIA megakernel: one request at attention TP8 + MoE TP4 x EP2: %s", what
        )
    _redundant_note(cfg, hf_config)
    return chosen


def _layout() -> list[str]:
    from sglang.srt.environ import envs

    from sgl_nvidia_mega.deepseek_v41 import runtime

    if "attn_core_lean" not in runtime.SLOTS or envs.SGLANG_DSV4_KV_LAYOUT.is_set():
        return []
    envs.SGLANG_DSV4_KV_LAYOUT.set(runtime.CORE_LEAN_KV_LAYOUT)
    return [
        f"SGLANG_DSV4_KV_LAYOUT={runtime.CORE_LEAN_KV_LAYOUT} (slot attn_core_lean)"
    ]


def _redundant_note(cfg, hf_config) -> None:
    """Log that slot moe_balanced is idle without the redundant experts, and the flag."""
    from sgl_nvidia_mega.deepseek_v41 import runtime

    if (
        "moe_balanced" not in runtime.SLOTS
        or cfg.ep_num_redundant_experts
        or cfg.enable_eplb
        or cfg.init_expert_location != "trivial"
    ):
        return
    routed = hf_config.n_routed_experts
    logger.info(
        "NVIDIA megakernel: slot moe_balanced is idle: every expert-parallel rank "
        "holding all %d routed experts (--ep-num-redundant-experts %d) turns it on, at "
        "about %d GB more weights per GPU and a smaller KV pool",
        routed,
        routed,
        REDUNDANT_GB,
    )
