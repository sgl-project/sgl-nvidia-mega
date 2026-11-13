# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Copied unchanged from TensorRT-LLM (https://github.com/NVIDIA/TensorRT-LLM, Apache-2.0, see ../LICENSE.tensorrt_llm):
# tensorrt_llm/_torch/_experimental/modeling_v2/models/musespark_open/muse_spark_open_816b_a42b_nvfp4__sm_107__tep/kernels/replicated_experts/__init__.py
# at commit ab93b785a476f01d2f1e1907c930780fc6d7832e.
"""replicated_experts: the host code of

* ``replicated_experts``: ring-buddy expert replication, balanced.
* ``three_window_placement``: a third ring window for the replicated experts.
* ``hot_replicas``: copies of each NVFP4 layer's hot experts, placed per step (tep4).
"""
