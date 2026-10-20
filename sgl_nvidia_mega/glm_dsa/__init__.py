# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""GLM-5 DSA (``GlmMoeDsaForCausalLM``, NVFP4, TP8) megakernel on SM103 / SM107: the
fused MTP target-verify step of 1-4 requests and the fused NextN draft layer.

Read ``spec`` -> ``eligibility`` -> ``runtime`` -> ``providers``; the draft is
``draft_spec`` -> ``draft``. The CuTe DSL kernels are under ``kernels/cute`` and import
no inference engine.
"""
