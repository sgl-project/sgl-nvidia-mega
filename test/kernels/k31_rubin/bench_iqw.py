# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# Rig timing of one sparse verify layer above 8 rows, arm A (served:
# stock index query behind the index-K writer) vs arm B (K31MK_INDEX_Q_OFF_WRITER_WIDE=1), using test_k31_iqw_wide.Layer (projection =
# a copy here, so the rig shows the overlap ceiling, not the in-situ delta). CUDA graph of 10 layer calls, 20 replays per block, blocks
# A B B A x 4 in one process; us per layer call, median of blocks.
import json, statistics, sys
import torch
import test_k31_iqw_wide as T

out = {}
for breq in [int(x) for x in sys.argv[1].split(",")]:
    layer = T.Layer(breq, seed=61)
    arms = {n: layer.arm(n, w) for n, w in (("A", 0), ("B", 1))}
    gs = {}
    for n, a in arms.items():
        s = torch.cuda.Stream(); s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(3):
                layer.chain(a)
        torch.cuda.current_stream().wait_stream(s); torch.cuda.synchronize()
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            for _ in range(10):
                layer.chain(a)
        torch.cuda.synchronize(); gs[n] = g
    t = {n: [] for n in gs}
    for n in ["A", "B", "B", "A"] * 4:
        e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        gs[n].replay(); torch.cuda.synchronize(); e0.record()
        for _ in range(20):
            gs[n].replay()
        e1.record(); torch.cuda.synchronize()
        t[n].append(e0.elapsed_time(e1) * 1000.0 / 200)
    out[breq] = {n: {"median_us": round(statistics.median(v), 3), "all": [round(x, 3) for x in v]} for n, v in t.items()}
    out[breq]["B_minus_A_us"] = round(out[breq]["B"]["median_us"] - out[breq]["A"]["median_us"], 3)
    print(json.dumps({breq: out[breq]}), flush=True)
    del gs, arms, layer
    torch.cuda.empty_cache()
json.dump(out, open(sys.argv[2], "w"), indent=1)
