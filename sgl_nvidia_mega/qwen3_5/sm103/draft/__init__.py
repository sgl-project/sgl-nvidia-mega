# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""The MTP draft's SM103 (GB300) kernels: the single-launch draft MoE (``moe``), the
q1 split-KV decode attention (``q1``), the draft step boundary's exchange, fc and head
(``xchg``), and the draft hop consume and o_proj GEMV (``hop``)."""
