# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Per-call time of MlaQuery.tiled under the K-22a modes (0 as built, C own qn SMEM, A fence + release arrive), one
GPU, every arm on the same GPU and the same inputs.

Each arm: the serving object (test_k31_mla_query_k22._build, query type pinned), its query kernel captured N times
back to back in one CUDA graph (PDL-chained like consecutive layers' query kernels; weights stay L2-warm). Blocks in
the order 0 C A A C 0 0 C A A C 0; each block = WARM untimed + REPS timed replays (CUDA events); per-call time =
replay time / N. Per arm: the median of its block medians; spread = max - min of its block medians.
delta(X) = med(X) - med(0); RESOLVED iff |delta| > 2 x max(spread(0), spread(X)).
Usage: K31_TEST_HEADS=6 python bench_mla_query_k22.py [rows=8] [kind=index|mla] [N=32] [REPS=20]"""

import json
import os
import statistics
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import test_k31_mla_query_k22 as K  # noqa: E402

ROWS = int(sys.argv[1]) if len(sys.argv) > 1 else 8
KIND = sys.argv[2] if len(sys.argv) > 2 else "index"
N = int(sys.argv[3]) if len(sys.argv) > 3 else 32
REPS = int(sys.argv[4]) if len(sys.argv) > 4 else 20
WARM = 3
ORDER = "0CAAC00CAAC0"


def arm(mode, case):
    from sgl_nvidia_mega.kimi_k31.k3.cutedsl.common.host import stream, to_cute

    w, kv0, idx0, table, slots, lengths, projected = case
    kv, idx = kv0.clone(), idx0.clone()
    obj = K._build(KIND, w, ROWS, kv, idx, mode)
    mq = obj.kernels[ROWS, 1][0]
    qnorm, kvnorm, qb, uk, _, qinv, kinv = obj.views
    query = obj.kernels[ROWS, 1][2]
    args = (to_cute(projected[:, :2112]), qnorm, kvnorm, qb, uk, to_cute(kv.view(torch.uint8)), to_cute(slots, 8),
            qinv, kinv, obj.eps, query, *obj.query_views)
    capture = torch.cuda.Stream()
    capture.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(capture):
        for _ in range(2):  # eager warm-up (module load) outside the capture
            mq(*args, stream(obj.device))
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g, stream=capture):
        for _ in range(N):
            mq(*args, stream(obj.device))
    form = K._engaged(obj, ROWS, K._applied(ROWS, "0")[1]) if K._ptx(obj, ROWS) else None
    return {"graph": g, "obj": obj, "form": form}


def block(g):
    for _ in range(WARM):
        g.replay()
    torch.cuda.synchronize()
    times = []
    for _ in range(REPS):
        s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        s.record()
        g.replay()
        e.record()
        e.synchronize()
        times.append(s.elapsed_time(e) * 1000.0 / N)  # us per call
    return statistics.median(times)


def main():
    case = K._case(9500 + ROWS, ROWS)
    arms = {m: arm(m, case) for m in "0CA"}
    blocks = {m: [] for m in "0CA"}
    for m in ORDER:
        blocks[m].append(block(arms[m]["graph"]))
    med = {m: statistics.median(v) for m, v in blocks.items()}
    spread = {m: max(v) - min(v) for m, v in blocks.items()}
    out = {"kind": KIND, "rows": ROWS, "heads": K.HEADS, "N": N, "reps": REPS, "order": ORDER,
           "gpu": torch.cuda.get_device_name(), "form": {m: arms[m]["form"] for m in "0CA"},
           "block_medians_us": blocks, "median_us": med, "spread_us": spread}
    for m in "CA":
        d = med[m] - med["0"]
        thr = 2 * max(spread["0"], spread[m])
        out[f"delta_{m}_us"] = d
        out[f"threshold_{m}_us"] = thr
        out[f"verdict_{m}"] = ("RESOLVED_SLOWER" if d > 0 else "RESOLVED_FASTER") if abs(d) > thr else "NOT_RESOLVED"
    print("K22_TIME " + json.dumps(out), flush=True)


if __name__ == "__main__":
    main()
