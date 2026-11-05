# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 SGLang Team
# ---- glm_dsa_cute shared body below ----
"""Fused all-reduce hops of the decode / verify step: o_proj + all-reduce + residual +
RMSNorm (optionally with the router), the MLP-output all-reduce + residual + RMSNorm,
and the MoE finalize fold. Engines use ``hop.api``."""
