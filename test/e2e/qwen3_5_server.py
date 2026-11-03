# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""The Qwen3.5-397B-A17B NVFP4 servers the serving tests launch: GB300, TP8, MTP (verify
steps of 7 rows a request), batch 1. TP8 spans two 4-GPU GB300 trays; a test launches
its own server only where one host has 8 GPUs, otherwise it reads a running server
(``SGLANG_NVIDIA_MEGAKERNEL_E2E_URL``) and the files its ranks log to
(``SGLANG_NVIDIA_MEGAKERNEL_E2E_LOGS``, comma-separated)."""

import os

MODEL = os.environ.get(
    "SGLANG_NVIDIA_MEGAKERNEL_E2E_MODEL_PATH", "nvidia/Qwen3.5-397B-A17B-NVFP4"
)
URL = os.environ.get("SGLANG_NVIDIA_MEGAKERNEL_E2E_URL")
LOGS = [
    p for p in os.environ.get("SGLANG_NVIDIA_MEGAKERNEL_E2E_LOGS", "").split(",") if p
]
ENV = {}
ARGS = [
    "--trust-remote-code",
    "--tp",
    "8",
    "--quantization",
    "modelopt_fp4",
    "--kv-cache-dtype",
    "fp8_e4m3",
    "--attention-backend",
    "trtllm_mha",
    "--moe-runner-backend",
    "flashinfer_trtllm",
    "--flashinfer-allreduce-fusion-backend",
    "cutedsl",
    "--linear-attn-decode-backend",
    "triton",
    "--enable-linear-replayssm-spec",
    "--mamba-ssm-dtype",
    "bfloat16",
    "--speculative-algorithm",
    "NEXTN",
    "--speculative-num-steps",
    "6",
    "--speculative-eagle-topk",
    "1",
    "--speculative-num-draft-tokens",
    "7",
    # batch 1: every verify step is the covered width of 7 rows
    "--max-running-requests",
    "1",
    "--cuda-graph-bs-decode",
    "1",
    "--mem-fraction-static",
    "0.85",
] + os.environ.get("SGLANG_NVIDIA_MEGAKERNEL_E2E_EXTRA_ARGS", "").split()
