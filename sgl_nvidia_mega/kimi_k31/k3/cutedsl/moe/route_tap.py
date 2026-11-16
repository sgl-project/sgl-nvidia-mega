# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the SGLang project
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""K31MK_MOE_ROUTE_TAP (default unset: off, nothing is built or launched): a census of the routed expert ids
every MoE layer hands to its experts engine, per forward step, for offline per-layer analysis (distinct experts
per step, repeats). Census only: the copy sits between the routing and the experts kernel on the main stream,
so it delays the experts' early launch; never time a run with it on.

Device ring ``ids [steps, layers, 64, 16]`` int16 (-1 = unwritten) and ``rows [steps, layers]`` (the batch rows
written). The step slot advances at the first tapped layer of each forward (a device counter, so it works under
CUDA-graph replay). ``K31MK_MOE_ROUTE_TAP_STEPS`` sets the ring depth (default 512: the last 512 steps are kept).
Dump: when ``K31MK_MOE_ROUTE_TAP_OUT`` is set, a daemon thread on the dumping rank waits for the file
``<OUT>.req`` and then writes ``<OUT>`` (npz: ids, rows, steps = forwards recorded since start, layers), removes
the request and writes ``<OUT>.done``. Request it while the server is idle: the copy is not ordered against the
serving streams."""

import os
import threading
import time

import numpy as np
import torch

ROWS = 64


def enabled():
    return os.environ.get("K31MK_MOE_ROUTE_TAP", "") not in ("", "0")


class RouteTap:
    def __init__(self, device, layers, *, dump=True, steps=None):
        self.layers = [int(layer) for layer in layers]
        self.index = {layer: i for i, layer in enumerate(self.layers)}
        self.steps = int(steps or os.environ.get("K31MK_MOE_ROUTE_TAP_STEPS", "512"))
        n = len(self.layers)
        self.ids = torch.full((self.steps, n, ROWS, 16), -1, dtype=torch.int16, device=device)
        self.rows = torch.zeros((self.steps, n), dtype=torch.int16, device=device)
        self.count = torch.zeros(1, dtype=torch.int64, device=device)
        self.slot = torch.zeros(1, dtype=torch.int64, device=device)
        self.cols = torch.arange(ROWS * 16, dtype=torch.int64, device=device)
        self.per_step, self.per_layer = n * ROWS * 16, ROWS * 16
        self.out = os.environ.get("K31MK_MOE_ROUTE_TAP_OUT", "")
        if dump and self.out:
            threading.Thread(target=self._serve, name="k31-route-tap", daemon=True).start()

    def record(self, layer_id, ids):
        """Copy this layer's routed ids ``[b, 16]`` (b <= 64) into the current step's slot (device-side index)."""
        i, b = self.index[layer_id], ids.shape[0]
        if i == 0:
            self.count.add_(1)
            torch.remainder(self.count - 1, self.steps, out=self.slot)
        pos = self.slot * self.per_step + i * self.per_layer + self.cols[: b * 16]
        self.ids.view(-1).index_put_((pos,), ids.reshape(-1).to(torch.int16))
        self.rows.view(-1).index_put_((self.slot * len(self.layers) + i,), self.rows.new_full((1,), b))

    def dump(self, path):
        stream = torch.cuda.Stream(self.ids.device)
        with torch.cuda.stream(stream):
            ids, rows, count = self.ids.cpu(), self.rows.cpu(), self.count.cpu()
        tmp = path + ".tmp.npz"
        np.savez(tmp, ids=ids.numpy(), rows=rows.numpy(), steps=count.numpy(), layers=np.array(self.layers))
        os.replace(tmp, path)

    def _serve(self):
        req = self.out + ".req"
        while True:
            if os.path.exists(req):
                try:
                    self.dump(self.out)
                    with open(self.out + ".done", "w") as f:
                        f.write(time.strftime("%Y-%m-%dT%H:%M:%S") + "\n")
                finally:
                    os.remove(req)
            time.sleep(1.0)
