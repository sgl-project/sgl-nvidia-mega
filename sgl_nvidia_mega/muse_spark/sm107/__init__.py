# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Muse Spark kernels and host runner for SM107 (Rubin): ``verify_mk``, TensorRT-LLM's decode megakernel for the
one-request verify step (``trtllm_ab93b78``). The component loader imports ``sm107.verify_mk`` for the ``verify_mk``
component (``components.MODULES``)."""
