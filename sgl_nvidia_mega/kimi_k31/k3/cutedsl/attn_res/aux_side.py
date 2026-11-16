# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the SGLang project
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""K31MK_AUX_FOLD=3: the drafter's aux mixtures on a side stream, off the verify chain.

Only the drafter reads the aux mixtures, after verify returns, so they need not run between
two layers. A capture consumer stores its updated prefix row into ``rows[index]`` (the live
prefix buffer changes again in the next o-projection); ``fork`` then launches the stock
AuxMixture passes on the side stream over that stored row and the saved blocks. Rows 0..nb-1
of the saved blocks keep their values for the rest of the step: block writes go to row nb or
later (``PreparedAttnRes.bind_aux`` keeps the in-line passes otherwise). ``join`` makes the
current stream wait for every mixture forked so far, once, before verify returns.

The fork and join are event edges, the same form as ``mla.verify_overlap``: CUDA graph capture
keeps them on replay. Construct during preparation (outside graph capture).
"""

import torch


class AuxSideMixture:
    def __init__(self, count, rows, device):
        self.stream = torch.cuda.Stream(device=device)
        self.ready = torch.cuda.Event()
        self.done = torch.cuda.Event()
        self.rows = torch.empty((count, rows, 7168), dtype=torch.bfloat16, device=device)
        self.forked = False

    def row(self, index, rows):
        """Where capture ``index``'s consumer stores its updated prefix row."""
        return self.rows[index, :rows]

    def fork(self, consumer, index, blocks, aux):
        """After ``consumer`` was launched on the current stream: mix its stored row into ``aux``."""
        self.ready.record(torch.cuda.current_stream(aux.device))
        self.stream.wait_event(self.ready)
        with torch.cuda.stream(self.stream):
            consumer.run_aux_mixture(self.row(index, aux.shape[0]), blocks, aux)
            self.done.record(self.stream)
        self.forked = True

    def join(self):
        if self.forked:
            torch.cuda.current_stream(self.stream.device).wait_event(self.done)
            self.forked = False
