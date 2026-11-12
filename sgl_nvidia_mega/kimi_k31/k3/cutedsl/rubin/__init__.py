# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Rubin (cc 10.7) TP16 routed-experts engines for verify steps of 9-512 token rows (see the file headers for vLLM-derived portions):
``moe/experts_rubin64.py``
(``RubinGroupedExpertsKernel``, 9-64 rows), ``moe/experts_wide.py`` (65-512 rows), their bases and ``common/`` helpers.
``common/host.py`` takes ``to_cute`` from ``latent_primitives`` and ``moe/experts_runtime.py`` ``init_logger`` from
``_compat``.
Our ``moe.experts_runtime.PersistentExperts`` delegates to
this package only under ``K31MK_MOE_RUBIN64`` (default unset = off)."""
