# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the SGLang project
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Kimi K3 megakernel integration."""

import os

if os.environ.get("K31MK_KERNEL_CACHE"):
    from sgl_nvidia_mega.common import kernel_cache

    kernel_cache.install()
