# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""The GLM-5 DSA megakernel's switches, read from the environment.

With ``--enable-nvidia-megakernel`` the fused verify step always runs; these pick
the optional parts. Each is "1" / "0"; unset takes the default.
"""

from __future__ import annotations

import os

# name -> default
DEFAULTS = {
    # the layers without their own top-k also run qkv_a on the small-M kernel
    "SGLANG_NVIDIA_MEGAKERNEL_GLM_SHARED_QKV_A_SMALL_M": True,
    # with the above: the input RMSNorm deferred into the small-M qkv_a kernel
    "SGLANG_NVIDIA_MEGAKERNEL_GLM_DNORM": True,
    # the NextN draft layer's fused kernels (draft decode and draft extend graphs)
    "SGLANG_NVIDIA_MEGAKERNEL_GLM_FUSED_DRAFT": True,
    # Serve the fused verify step though SGLang's FlashInfer all-reduce + RMSNorm
    # fusion is off at verify sizes (no NVLink multicast, e.g. multi-node NVL72 trays):
    # the fused hops do not need multicast, but the stock step they replace then runs
    # NCCL + RMSNorm, so its numerics are not the ones the hops reproduce.
    "SGLANG_NVIDIA_MEGAKERNEL_GLM_ALLOW_NO_AR_FUSION": False,
    # The hop kernels prefer the maximum shared-memory carveout, so an SM a hop CTA
    # lingers on stays in the configuration the next layer's large-shared-memory
    # kernels (launched early under PDL) can share. A scheduling hint: outputs are
    # unchanged.
    "SGLANG_NVIDIA_MEGAKERNEL_GLM_HOP_MAX_SHARED_CARVEOUT": False,
}

# Not a "1" / "0" switch: N in {1, 2} caps the o_proj hop's consumer CTAs at N per SM
# (padded launch shared memory, outputs unchanged); unset or 0 leaves them unpadded.
HOP_CTAS_PER_SM = "SGLANG_NVIDIA_MEGAKERNEL_GLM_HOP_CTAS_PER_SM"


def enabled(name: str) -> bool:
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return DEFAULTS[name]
    return raw not in ("0", "false", "False")


def shared_qkv_a_small_m() -> bool:
    return enabled("SGLANG_NVIDIA_MEGAKERNEL_GLM_SHARED_QKV_A_SMALL_M")


def dnorm() -> bool:
    return shared_qkv_a_small_m() and enabled("SGLANG_NVIDIA_MEGAKERNEL_GLM_DNORM")


def fused_draft() -> bool:
    return enabled("SGLANG_NVIDIA_MEGAKERNEL_GLM_FUSED_DRAFT")


def allow_no_ar_fusion() -> bool:
    return enabled("SGLANG_NVIDIA_MEGAKERNEL_GLM_ALLOW_NO_AR_FUSION")


def hop_max_shared_carveout() -> bool:
    return enabled("SGLANG_NVIDIA_MEGAKERNEL_GLM_HOP_MAX_SHARED_CARVEOUT")


def hop_ctas_per_sm() -> int:
    raw = os.environ.get(HOP_CTAS_PER_SM, "").strip()
    if raw in ("", "0"):
        return 0
    if raw not in ("1", "2"):
        raise ValueError(f"{HOP_CTAS_PER_SM}={raw!r}: expected 1 or 2 (or 0 / unset)")
    return int(raw)
