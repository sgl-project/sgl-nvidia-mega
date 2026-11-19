# SPDX-License-Identifier: Apache-2.0
"""Where the GB300 BEST levers are on by default: the configuration the GB300 BESTs ran (AgentX and 8K),
TP16, DCP1, DFLASH Q8 and one running request, so every verify step is one 8-row request. Elsewhere an unset lever
keeps its earlier default; a set env always wins. Read from SGLang's published config, which the plugin's modules see
when the worker imports them at the megakernel build; without it (tools, tests) out of scope."""

from __future__ import annotations

import os

# Names read from SGLang. A contract test checks these before GPU use.
SGLANG_READS = {
    "srt/runtime_context.py": (
        "get_context",
        "get_parallel",
        "get_schedule",
        "get_spec",
    ),
}

_SCOPE: list = []  # [in scope], once the config is published


def in_scope() -> bool:
    if _SCOPE:
        return _SCOPE[0]
    try:
        from sglang.srt.runtime_context import (
            get_context,
            get_parallel,
            get_schedule,
            get_spec,
        )

        if not all(
            get_context().is_config_namespace_published(n) for n in ("schedule", "spec")
        ):
            return False
        parallel = get_parallel()
        scope = (
            int(parallel.tp_size) == 16
            and int(parallel.attn_dcp_size or 1) == 1
            and int(get_spec().speculative_num_draft_tokens or 0) == 8
            and int(get_schedule().max_running_requests or 0) == 1
        )
    except Exception:  # noqa: BLE001  no SGLang context in this process
        return False
    _SCOPE.append(scope)
    return scope


def best_default(name: str, best: str, default: str) -> str:
    """``os.environ[name]`` when set, else ``best`` in scope, else the earlier ``default``."""
    value = os.environ.get(name)
    if value is not None:
        return value
    return best if in_scope() else default
