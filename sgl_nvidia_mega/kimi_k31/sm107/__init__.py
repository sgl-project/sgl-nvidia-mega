# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the SGLang project
"""Rubin (SM107) plans and kernels of the Kimi K3.1 megakernel. Rubin runs the kernels
of ``kimi_k31/sm103`` except the attention residual, which is ``sm107/k3/attn_res``'s;
``kimi_k31/targets.py`` picks both on SM107."""
