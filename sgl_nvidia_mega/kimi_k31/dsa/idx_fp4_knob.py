# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""The K31MK_DSA_IDX_FP4 switch alone (no kernel imports), so the default-off path imports nothing new; dsa/idx_fp4.py holds the
MXFP4 index-K / index-query code it selects."""

import os

ENV = "K31MK_DSA_IDX_FP4"
ENABLED = os.environ.get(ENV, "0") == "1"
if ENABLED and os.environ.get("K31MK_DCP", "0") == "1":
    raise ValueError(f"{ENV}=1 serves TP verify only (the DCP index-K reads stay FP8): unset K31MK_DCP")
# The indexer-query toggle, with K31MK_DSA_IDX_FP4=1 (the MXFP4 index-K pages, both settings):
#   K31MK_DSA_IDX_Q=fp8 (the DEFAULT): the served FP8 index query scored against the MXFP4 pages (dsa/dg_logits_fp4k.py); label
#     "MXFP4 indexer cache (FP8 query)" = the benchmark precision list's "indexer cache FP4";
#   K31MK_DSA_IDX_Q=fp4: the MXFP4 index query, quantized inside the kernel that produces it (dsa/idx_fp4_q.py: index_q.cuh kQ4 for the
#     fused index query / query front, the MXFP4 QFN kernel, idx_fp4.quantize_query for the stock chain) x the MXFP4 pages; label
#     "MXFP4 indexer q".
# Legacy word K31MK_DSA_IDX_FP4_FP8Q: 1 = fp8, 0 = fp4; it must agree with K31MK_DSA_IDX_Q when both are set.
Q_ENV = "K31MK_DSA_IDX_Q"
FP8Q_ENV = "K31MK_DSA_IDX_FP4_FP8Q"
_q = os.environ.get(Q_ENV, "").strip().lower()
_legacy = os.environ.get(FP8Q_ENV, "").strip()
if _q not in ("", "fp8", "fp4"):
    raise ValueError(f"{Q_ENV} must be fp8 or fp4, got {_q!r}")
if _legacy not in ("", "0", "1"):
    raise ValueError(f"{FP8Q_ENV} must be 0 or 1, got {_legacy!r}")
if (_q or _legacy == "1") and not ENABLED:
    raise ValueError(f"{Q_ENV} / {FP8Q_ENV}=1 needs {ENV}=1 (both settings score against the MXFP4 index-K pages)")
if _q and _legacy and (_q == "fp8") != (_legacy == "1"):
    raise ValueError(f"{Q_ENV}={_q} contradicts {FP8Q_ENV}={_legacy}")
QUERY = _q or ("fp4" if _legacy == "0" else "fp8")
FP8Q = ENABLED and QUERY == "fp8"
FP4Q = ENABLED and QUERY == "fp4"


def refuse(path: str) -> None:
    """Fail closed under K31MK_DSA_IDX_FP4=1: ``path`` reads or writes FP8 index-K bytes and is not ported to the MXFP4 pages."""
    if ENABLED:
        raise ValueError(f"{ENV}=1: {path} reads or writes FP8 index-K bytes and is not ported to the MXFP4 pages")
