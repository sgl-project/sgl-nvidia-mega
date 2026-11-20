# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""The Muse Spark Open megakernel: fused kernels for its target-verify step."""

from __future__ import annotations

ARCHITECTURE = "MuseSparkForConditionalGeneration"

# The HF config the kernels are built for; SGLang refuses a model that differs.
HF_CONFIG = {
    "model_type": "muse_spark",
}
