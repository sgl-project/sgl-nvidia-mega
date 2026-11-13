# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Muse Spark kernels and host runner for SM103 (GB300): ``verify_mk``, TensorRT-LLM's decode megakernel for the
one-request verify step (``trtllm_ab93b78``) ported from sm_107 to sm_103a. The component loader imports
``sm103.verify_mk`` in place of ``components.verify_mk`` on compute capability 10.3 (``components.ARCH_MODULES``)."""
