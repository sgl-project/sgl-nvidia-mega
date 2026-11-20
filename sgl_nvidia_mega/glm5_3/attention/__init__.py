# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""The verify attention of a GLM-5.3 layer: the stock step (``step``), which
reproduces SGLang's attention bit for bit, with the host-only facts the steps
need, and the kernels of slot ``attention`` (``fused``)."""
