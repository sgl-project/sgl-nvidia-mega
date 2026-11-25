# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""The DeepSeek-V4.1-Flash servers the serving tests launch: GB300, TP4, DSpark
(verify steps of 6 rows a request). The megakernel switch is set per launch, so a
test can launch the normal path too."""

import os

MODEL = os.environ.get(
    "SGLANG_NVIDIA_MEGAKERNEL_E2E_MODEL_PATH", "deepseek-ai/DeepSeek-V4.1-Flash"
)
# An already running server, so the same checks can score the switch-off arm
URL = os.environ.get("SGLANG_NVIDIA_MEGAKERNEL_E2E_URL")
# The measured launch's environment: TP sync as measured
ENV = {
    "SGLANG_SPEC_TP_SYNC": "rng,-dspark-accept-sample",
}
COMMON = [
    "--trust-remote-code",
    "--tp",
    "4",
    # the routed-expert pair needs every expert local
    "--ep-size",
    "1",
    "--kv-cache-dtype",
    "fp8_e4m3",
    "--dsv4-attn-backend",
    "flashmla",
    "--mem-fraction-static",
    "0.8",
    # the Engram prefetch needs the per-rank host table (the spec refuses without it)
    "--engram-config",
    '{"placement": "cpu"}',
] + os.environ.get("SGLANG_NVIDIA_MEGAKERNEL_E2E_EXTRA_ARGS", "").split()
# DSpark: up to 8 requests of 6 rows on the routed pair (48 rows), 9-16 requests on
# the two-shot hops, so the same requests alternate between the two
ARGS = COMMON + [
    "--speculative-algorithm",
    "DSPARK",
    "--speculative-dspark-block-size",
    "5",
    "--cuda-graph-max-bs-decode",
    "16",
    "--max-running-requests",
    "16",
]
