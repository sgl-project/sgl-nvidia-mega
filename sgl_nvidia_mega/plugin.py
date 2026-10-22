# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""The SGLang plugin entry point: it registers the megakernel specs with SGLang's
``sglang.srt.megakernels``, which runs them on a server started with
``--enable-nvidia-megakernel``."""

from __future__ import annotations

import logging
import os

logger = logging.getLogger(__name__)

# SGLang's server flag that turns the megakernel on
FLAG_CLI = "--enable-nvidia-megakernel"

# HF architecture -> its spec, by "module:attribute" (imported only with the flag)
SPECS = (
    ("DeepseekV4ForCausalLM", "sgl_nvidia_mega.deepseek_v41.spec:SPEC"),
    ("GlmMoeDsaForCausalLM", "sgl_nvidia_mega.glm_5.spec:SPEC"),
    # the DSpark draft: SGLang's forward, the megakernel's folded proposal
    ("DeepseekV4ForCausalLMDSpark", "sgl_nvidia_mega.deepseek_v41.draft:SPEC"),
)

# Test only: with SGLANG_NVIDIA_MEGAKERNEL_PARITY_SORT_TOPK=1, the composition check
# sorts the sparse attention indices in both forwards (their emission order varies)
PARITY_SORT_ENV = "SGLANG_NVIDIA_MEGAKERNEL_PARITY_SORT_TOPK"
TEST_HOOKS = (
    (
        "sglang.srt.layers.attention.deepseek_v4_backend.DeepseekV4AttnBackend.forward",
        "AROUND",
        "sgl_nvidia_mega.deepseek_v41.runtime.parity:around_attention_forward",
    ),
)

# The SGLang names this module reads (test/unit/test_contract.py pins them)
SGLANG_READS = {
    # the entry point name pyproject.toml declares (test/unit/test_plugin.py)
    "srt/megakernels/spec.py": (
        '"nvidia_megakernel"',
        "register",
        "register(architecture)",
        "register(path)",
    ),
    "srt/plugins/hook_registry.py": ("HookRegistry", "HookType", "register", "AROUND"),
}


def register() -> None:
    """SGLang's plugin loader calls this in every process, before any model is built."""
    try:
        import sglang.srt.megakernels.spec as megakernels
    except ImportError:
        logger.warning("sgl-nvidia-mega: this SGLang has no sglang.srt.megakernels")
        return
    for architecture, path in SPECS:
        megakernels.register(architecture=architecture, path=path)
    if os.environ.get(PARITY_SORT_ENV) == "1":
        import pkgutil

        from sglang.srt.plugins.hook_registry import HookRegistry, HookType

        for target, kind, hook in TEST_HOOKS:
            HookRegistry.register(target, pkgutil.resolve_name(hook), HookType[kind])
