# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""The EAGLE v2 speculative step graph and its fusions, for any model.

* ``SGLANG_NVIDIA_MEGAKERNEL_SPEC_STEP_GRAPH=1``: each topk-1 decode step (draft,
  target verify, draft extend and the work between them) replays as one CUDA graph,
  with the per-stage graphs spliced in as child graphs (``runner``).
* ``SGLANG_SPEC_*`` (``switches``): fusions of the step's small launches; each has
  bitwise-identical outputs.

All default to off; ``hooks.register`` installs the hooks of the ones that are on.
"""
