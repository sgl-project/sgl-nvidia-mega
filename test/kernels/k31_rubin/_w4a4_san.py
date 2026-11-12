# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""compute-sanitizer driver : the W4A4 experts (x plan kernel + experts kernel) at ``tp`` / ``batch``,
two calls on random routes (image parities 1 and 0). Usage: python3 _w4a4_san.py TP BATCH [KNOBS]."""

import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import _harness as H  # noqa: E402
import _ref_moe as R  # noqa: E402
import test_k31_experts as TE  # noqa: E402
import test_k31_experts_w4a4 as KW  # noqa: E402


def main():
    tp, batch = int(sys.argv[1]), int(sys.argv[2])
    knobs = tuple(k for k in (sys.argv[3] if len(sys.argv) > 3 else "xone").split(",") if k)
    H.load_plugin(KW.MUTATIONS)
    ex, out = KW.engine4(tp, batch, knobs)
    g = H.generator(5)
    for _ in range(2):
        x = TE.latent(batch, g)
        ids, _ = R.random_route(batch, g)
        ex.run(x, ids)
        torch.cuda.synchronize()
    print("SAN_DRIVER_OK", tp, batch, knobs, bool(torch.isfinite(out.float()).all()), flush=True)


if __name__ == "__main__":
    main()
