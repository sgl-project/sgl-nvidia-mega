# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# Vendored FlashInfer gvr_2 top-k (flashinfer/topk_varlen/kernels/gvr2_topk_{host,decode}.py, Apache-2.0): the host file is
# unmodified; the decode file carries one NVIDIA change marked in the file. THIRD_PARTY_NOTICES.txt names the exact source
# revision. Used only under K31MK_VERIFY_TOPK=gvr (dsa/verify.py), default off.
# The modules import standalone (relative imports of FlashInfer's JIT cache / availability helpers fall back on ImportError:
# kernels compile in-process, no persistent cache). Do not edit here; re-vendor from the source.
