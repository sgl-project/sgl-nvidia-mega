# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""The GLM-5.3 / GLM-5.2 (NVFP4) servers the serving tests launch: GB300, TP4, MTP
(EAGLE 5 steps / topk 1 / 6 draft tokens: verify steps of 6 rows a request). The
megakernel switch is set per launch, so a test launches the baseline (SGLang's own
path) and the candidate with the same arguments."""

import os

# An NVFP4 GLM-5.3 or GLM-5.2 checkpoint whose config has glm5_3.HF_CONFIG's shapes
MODEL = os.environ.get(
    "SGLANG_NVIDIA_MEGAKERNEL_E2E_MODEL_PATH", "nvidia/GLM-5.2-NVFP4"
)
# An already running server, so the same checks can score one arm launched elsewhere
URL = os.environ.get("SGLANG_NVIDIA_MEGAKERNEL_E2E_URL")
# Both arms' environment: the candidate's NextN draft step (slot draft) reads the
# replicated embedding table. The baseline runs without the flag, so nothing would set
# the variable for it; setting it here for both arms keeps their embedding layout the
# same, so the arms differ only in the flag. SGLang's defaults give the rest of
# spec.ENV.
ENV = {"SGLANG_ENABLE_EMBED_REPLICATION": "1"}
# Tokens per request of the MTP chain (--speculative-num-draft-tokens)
TOKENS = 6
# The widest verify step the TP4 kernel units serve (profile.UNITS): 8 requests
COVERED_REQUESTS = 8
# The launch the spec admits at TP4 (spec.SERVER_ARGS), as docs/models/glm5_3.md
# launches it
ARGS = [
    "--trust-remote-code",
    "--tp",
    "4",
    "--kv-cache-dtype",
    "fp8_e4m3",
    "--page-size",
    "64",
    "--attention-backend",
    "dsa",
    "--dsa-decode-backend",
    "trtllm",
    "--moe-runner-backend",
    "flashinfer_trtllm",
    "--disable-shared-experts-fusion",
    "--speculative-algorithm",
    "EAGLE",
    "--speculative-num-steps",
    str(TOKENS - 1),
    "--speculative-eagle-topk",
    "1",
    "--speculative-num-draft-tokens",
    str(TOKENS),
    # up to 8 requests (48 rows) the kernel units, 9-16 SGLang's path: a client at 16
    # requests moves the same server across that boundary and back
    "--cuda-graph-max-bs-decode",
    "16",
    "--max-running-requests",
    "16",
] + os.environ.get("SGLANG_NVIDIA_MEGAKERNEL_E2E_EXTRA_ARGS", "").split()
