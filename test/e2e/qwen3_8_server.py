# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""The Qwen3.8-Flash-Next NVFP4 servers the serving tests launch: one Rubin GPU (SM107),
TP1, MTP (NEXTN, 3 draft steps: verify steps of 4 rows a request), up to 32 requests,
BF16 KV. The megakernel switch is set per launch, so a test can launch SGLang's own
path with the same settings."""

import os

MODEL = os.environ.get(
    "SGLANG_NVIDIA_MEGAKERNEL_E2E_MODEL_PATH", "nvidia/Qwen3.8-Flash-Next-NVFP4"
)
# The kernel slots of the measured configuration (qsa_pool_fp8 needs an FP8 KV pool)
SLOTS = os.environ.get(
    "SGLANG_NVIDIA_MEGAKERNEL_E2E_SLOTS",
    "hc,hc_wide,gdn_native_pdl,gdn_verify_glue,gdn_front,gdn_prefill,qsa_pool,"
    "qsa_indexer,qsa_prefill,ple,moe_finalize,moe_routing,smallm_gemm,qsa_draft,"
    "draft_fold",
)
# The S1 layers, which hc, hc_wide, moe_finalize and the PLE add need
ENV = {"SGLANG_NVIDIA_MEGAKERNEL_QWEN3_8_DECOMPOSE": "1"}
ARGS = [
    "--trust-remote-code",
    "--quantization",
    "modelopt_mixed",
    "--tp",
    "1",
    "--context-length",
    "32768",
    "--attention-backend",
    "trtllm_mha",
    "--linear-attn-prefill-backend",
    "flashinfer",
    "--linear-attn-decode-backend",
    "flashinfer",
    "--linear-attn-verify-backend",
    "triton",
    "--mamba-ssm-dtype",
    "bfloat16",
    "--mamba-radix-cache-strategy",
    "extra_buffer_lazy",
    "--mamba-track-interval",
    "256",
    # 4 state slots a request: 32 running requests
    "--max-mamba-cache-size",
    "128",
    "--kv-cache-dtype",
    "bfloat16",
    "--speculative-draft-kv-cache-dtype",
    "bfloat16",
    "--page-size",
    "64",
    "--chunked-prefill-size",
    "8192",
    "--max-prefill-tokens",
    "8192",
    "--moe-runner-backend",
    "flashinfer_trtllm",
    "--disable-shared-experts-fusion",
    "--speculative-algorithm",
    "NEXTN",
    "--speculative-num-steps",
    "3",
    "--speculative-eagle-topk",
    "1",
    "--speculative-num-draft-tokens",
    "4",
    "--speculative-moe-runner-backend",
    "flashinfer_trtllm",
    "--enable-linear-replayssm-spec",
    "--ple-offload-embedding",
    "--ple-offload-backend",
    "pinned",
    "--cuda-graph-backend-prefill",
    "disabled",
    "--cuda-graph-backend-decode",
    "full",
    "--cuda-graph-max-bs-decode",
    "32",
    "--max-running-requests",
    "32",
    "--mem-fraction-static",
    "0.8",
    "--watchdog-timeout",
    "1800",
] + os.environ.get("SGLANG_NVIDIA_MEGAKERNEL_E2E_EXTRA_ARGS", "").split()
