# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""The environment switches of the speculative step graph and its fusions.

Each defaults to off. ``hooks.register`` installs only the hooks of the switches
that are on, so with all of them off SGLang runs unchanged. The names are the ones
of SGLang's tree with these changes, so a launch script serves both.
"""

from __future__ import annotations

import os

# One CUDA graph per EAGLE v2 decode step (SGLang's tree:
# --enable-speculative-step-cuda-graph), reusing the per-stage CUDA graphs.
STEP_GRAPH = "SGLANG_NVIDIA_MEGAKERNEL_SPEC_STEP_GRAPH"
# Copy the step's live inputs into the graph's static buffers in one launch.
STEP_GRAPH_FUSED_INPUT_COPY = "SGLANG_SPEC_STEP_GRAPH_FUSED_INPUT_COPY"
# predict / accept_index / num_correct_drafts as views of one buffer: one rank-0
# TP broadcast of the verify result instead of three.
PACKED_VERIFY_BROADCAST = "SGLANG_SPEC_PACKED_VERIFY_BROADCAST"
# Chain sampled verify: temperatures not expanded, no zero-filled draft_probs.
LEAN_SAMPLED_VERIFY = "SGLANG_SPEC_LEAN_SAMPLED_VERIFY"
# With LEAN_SAMPLED_VERIFY: the sampling passes split over vocabulary chunks.
CHUNKED_VERIFY_SAMPLING = "SGLANG_SPEC_CHUNKED_VERIFY_SAMPLING"
# The topk-1 draft-extend argmax, unit probs and DSA seed gather in two launches.
FUSED_DRAFT_EXTEND_TAIL = "SGLANG_SPEC_FUSED_DRAFT_EXTEND_TAIL"
# The relay rows (FutureMap): one scatter per stash, row views for one request.
FUSED_RELAY = "SGLANG_SPEC_FUSED_RELAY"

ALL = (
    STEP_GRAPH,
    STEP_GRAPH_FUSED_INPUT_COPY,
    PACKED_VERIFY_BROADCAST,
    LEAN_SAMPLED_VERIFY,
    CHUNKED_VERIFY_SAMPLING,
    FUSED_DRAFT_EXTEND_TAIL,
    FUSED_RELAY,
)


def enabled(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in ("1", "true", "yes", "on")


def any_enabled() -> bool:
    return any(enabled(name) for name in ALL)
