# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Fused all-reduce hops of the decode / verify step: o_proj + all-reduce + residual +
RMSNorm (optionally with the router), the MLP-output all-reduce + residual + RMSNorm,
and the MoE finalize fold. Engines use ``hop.api``."""
