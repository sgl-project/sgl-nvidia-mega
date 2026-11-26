# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Copied unchanged from TensorRT-LLM (https://github.com/NVIDIA/TensorRT-LLM, Apache-2.0, see ../LICENSE.tensorrt_llm):
# tensorrt_llm/_torch/_experimental/modeling_v2/models/musespark_open/muse_spark_open_816b_a42b_nvfp4__sm_107__tep/kernels/megakernel/__init__.py
# at commit ab93b785a476f01d2f1e1907c930780fc6d7832e.
"""megakernel: the persistent per-layer decode kernel for Rubin (``msoMegakernel.cu``) and its host code.

* ``schedule``: the host planner (unit lists, shared-memory placement, weight issue points), CPU only.
* ``library``: build / load / launch, the shared workspace, parameter blocks, stamps and timeline dumps.
* ``qkv``: the QKV stage of a decoder layer (weight repack, parameter fields, a stand-alone stage object).
* ``check``: float64 references and error statistics of check mode.
"""
