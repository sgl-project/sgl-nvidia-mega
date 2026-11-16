# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Short Q8 verify: overlap the independent index-K writer with MLA work.

This helper stays inside the prepared plugin. Its writer is the existing prepared
index-K provider, with the same projection, LayerNorm and cache-store operations.
The executor starts it after producing the normalized input, launches the packed
MLA projection/query/core/value/output projection on the current stream, then
joins before reusing that input or its slot metadata. The sparse verify provider owns its own index
streams; its long-context path must continue to use that provider directly.
"""

import torch


class VerifyMlaIndexOverlap:
    """One stream and two reusable events for a serialized Q8 verify executor.

    Construct during preparation and warm ``run``/``join`` before graph capture.
    ``run`` and ``join`` are paired on the current stream; CUDA graph capture keeps
    the same fork/join edges on replay. No model runner or host callback is stored.
    """

    def __init__(self, device):
        self.stream = torch.cuda.Stream(device=device)
        self.input_ready = torch.cuda.Event()
        self.complete = torch.cuda.Event()

    def run(self, writer, hidden, slots):
        current = torch.cuda.current_stream(hidden.device)
        self.input_ready.record(current)
        self.stream.wait_event(self.input_ready)
        with torch.cuda.stream(self.stream):
            writer.run(hidden, slots)
            self.complete.record(self.stream)

    def join(self):
        torch.cuda.current_stream(self.stream.device).wait_event(self.complete)
