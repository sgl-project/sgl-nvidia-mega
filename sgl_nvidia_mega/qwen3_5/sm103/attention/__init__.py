# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""The full-attention kernels: SKV2 (``skv2``), the paged FP8 split-KV attention of
one request's 7 verify rows (a CuTe DSL producer over 148 page splits and a Triton
merge)."""
