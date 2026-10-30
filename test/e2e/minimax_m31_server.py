# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""The MiniMax-M3.1-Flash servers the serving tests launch: GB300, TP4, DSpark (verify
steps of 8 rows a request, draft blocks of 7). The megakernel switch is set per launch,
so a test can launch the normal path too."""

import os

MODEL = os.environ.get(
    "SGLANG_NVIDIA_MEGAKERNEL_E2E_MODEL_PATH", "MiniMaxAI/MiniMax-M3.1-Flash"
)
# An already running server, so the same checks can score the switch-off arm
URL = os.environ.get("SGLANG_NVIDIA_MEGAKERNEL_E2E_URL")
# MegaAttention keeps all its CTAs resident: no kernel of the next batch beside it
ENV = {"SGLANG_FORCE_COARSE_WAR_BARRIER": "1"}
ARGS = [
    "--trust-remote-code",
    "--tp",
    "4",
    "--moe-runner-backend",
    "flashinfer_trtllm_routed",
    "--page-size",
    "128",
    "--speculative-algorithm",
    "DSPARK",
    "--speculative-use-block-verification",
    "--speculative-num-steps",
    "1",
    "--speculative-eagle-topk",
    "1",
    "--speculative-num-draft-tokens",
    "8",
    "--speculative-dspark-block-size",
    "7",
    "--disable-attn-tp-gather",
    "--max-running-requests",
    "8",
    "--mem-fraction-static",
    "0.8",
    "--reasoning-parser",
    "minimax-m3",
] + os.environ.get("SGLANG_NVIDIA_MEGAKERNEL_E2E_EXTRA_ARGS", "").split()
