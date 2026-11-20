# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Fixed geometry of the GLM-5.3 routed-expert layer at tensor parallel 8."""

HIDDEN = 6144  # model hidden size (K of FC1, N of FC2)
NUM_EXPERTS = 256  # routed experts
TOP_K = 8  # experts per token
INTER = 256  # per-rank intermediate size of one expert (gate or up rows)
QUARTERS = 4  # an expert's FC2 output is reduced from four quarter partials
SHARED_PARTS = 4  # the shared expert's output is reduced from four partials
MAX_ROWS = 16  # verify rows of up to four requests
ROUTED_SCALE = 2.5  # routed scaling factor of the checkpoint
