# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""K31MK_MOE_RUBIN64_W2_DEPAD together with K31MK_MOE_WIDE (wide decode M1): the grouped 9-64-row engine reads
the de-padded 96-byte FC2 copy while the wide engine (65-128 rows) addresses FC2 rows with the padded 128-byte stride, so
under DEPAD the wide rows get their own vendored object over the padded w2 (``PersistentExperts.rubin_wide``). Before this
change the two knobs together were refused.

One GPU, TP16 host-layout weights of one rank (``test_k31_experts_tp16.weights16``, all 896 experts, die-oblivious).
Arms: P = K31MK_MOE_RUBIN64=9-64:1 + K31MK_MOE_WIDE=65-128:1 (DEPAD unset, the served wide form); D = the same + DEPAD=1.
Batches 72, 128 (wide engine, publish ("local", partial): this rank's routing-weighted partial latent rows) and 32 (grouped
engine, the 16 expert rows per token into ``out``). Per batch: 2 eager calls and 1 CUDA-graph replay, new inputs each.
Checks (written before any run, MOE-wideM1b.1; not loosened after data):
  engaged  D: the wide engine's w2 base is the padded w2 (data_ptr equal), the grouped engine's is not (the copy), and the
           two are different objects; P: one object, padded w2.
  bitwise  D's outputs == P's, every BF16 cell, every call (wide: the same engine over the same bytes; 32 rows: the
           MOEB54 DEPAD claim, same SMEM bytes).
  fp64     wide rows: partial vs sum_j w_tj * expert_rows (``_ref_moe16.expert_rows``, FP64, logical weights): Frobenius
           rel-L2 <= 1e-2 and every row <= 3e-2 (test_k31_experts_tp16's T1 bounds); 32 rows: the 16 expert rows per token
           vs expert_rows under the same bounds. Validity: arm P must pass this line too; if P fails it, the fp64 line is
           VOID for that batch (reference / tolerance problem, not evidence about DEPAD) and bitwise stands alone.
Outputs are NaN-poisoned before every call (unwritten cells fail every line).
Mutation (``MUTATIONS``, must fail): depad_wide_shared (the wide rows run on the grouped DEPAD object, the pre-fix
wiring without the refusal): D's wide engine then reads 128-byte strides over the 96-byte copy.

Usage (one GPU): python3 k31_wide_depad.py --out DIR. Exit 0 only when every check passed (``RESULT failures=0``).
"""

import argparse
import json
import os
import sys
import time
import traceback

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _harness as H  # noqa: E402
import _ref_moe as R  # noqa: E402
import _ref_moe16 as M  # noqa: E402

ER_FILE = "kimi_k31/k3/cutedsl/moe/experts_runtime.py"
MUTATIONS = {
    "depad_wide_shared": (ER_FILE, "self.rubin_wide = vendored(w2.view(u8), {}) if rubin_kw and self.wide_rows else self.rubin",
                          "self.rubin_wide = self.rubin"),
}
FRO, ROW = 1e-2, 3e-2
ENV = {"K31MK_MOE_RUBIN64": "9-64:1", "K31MK_MOE_WIDE": "65-128:1"}


def rel(rows, ref):
    rows, ref = rows.double(), ref.double()
    fro = float((rows - ref).norm() / ref.norm())
    per_row = ((rows - ref).norm(dim=-1) / ref.norm(dim=-1)).flatten()
    return fro, float(per_row.max())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--batches", default="72,128,32")
    args = ap.parse_args()
    tag = os.environ.get("K31_MUTATION") or "base"
    os.makedirs(args.out, exist_ok=True)
    os.environ["K31_RESULTS"] = os.path.join(args.out, f"wide_depad_{tag}.jsonl")
    os.environ.update(ENV)
    H.load_plugin(MUTATIONS)
    import test_k31_experts_tp16 as T16

    from sgl_nvidia_mega.kimi_k31.k3.cutedsl.moe import experts_runtime as ER

    device = torch.device("cuda", 0)
    torch.cuda.set_device(device)
    fails = []
    t0 = time.time()

    def fail(msg):
        fails.append(msg)
        print("FAIL", msg, flush=True)

    try:
        logical, prepared = T16.weights16()
        w2 = prepared[2]
        arms = {}
        for arm in ("P", "D"):
            os.environ.pop("K31MK_MOE_RUBIN64_W2_DEPAD", None)
            if arm == "D":
                os.environ["K31MK_MOE_RUBIN64_W2_DEPAD"] = "1"
            try:
                ex = ER.PersistentExperts(*prepared, beta=R.BETA, linear_beta=R.LIN, activation_floor=False)
            finally:
                os.environ.pop("K31MK_MOE_RUBIN64_W2_DEPAD", None)
            arms[arm] = ex
        p, d = arms["P"], arms["D"]
        eng = {
            "P_one_object": p.rubin_wide is p.rubin,
            "P_wide_padded": p.rubin_wide.weights[2].data_ptr() == w2.data_ptr(),
            "D_two_objects": d.rubin_wide is not d.rubin,
            "D_wide_padded": d.rubin_wide.weights[2].data_ptr() == w2.data_ptr(),
            "D_grouped_copy": d.rubin.weights[2].data_ptr() != w2.data_ptr() and d.rubin.w2_row_bytes == 96,
        }
        H.record({"kind": "check", "item": "engaged", **eng})
        print("CHECK engaged", json.dumps(eng), flush=True)
        for key, ok in eng.items():
            if not ok:
                fail(f"engaged.{key}")
        for b in map(int, args.batches.split(",")):
            wide = b > 64
            g = H.generator(7700 + b)
            x = torch.empty((b, R.H_LAT), device="cuda")
            ids = torch.empty((b, 16), dtype=torch.int32, device="cuda")
            wts = torch.empty((b, 16), dtype=torch.bfloat16, device="cuda")
            outs, partials = {}, {}
            for arm, ex in arms.items():
                outs[arm] = torch.empty((b * 16, R.H_LAT), dtype=torch.bfloat16, device="cuda")
                partials[arm] = torch.empty((b, R.H_LAT), dtype=torch.bfloat16, device="cuda")
                publish = ("local", partials[arm]) if wide else ()
                ex.prepare(b, ER.scratch(16, device), outs[arm], publish, False)

            def call():
                for arm, ex in arms.items():
                    ex.run(x, ids, wts if wide else None)

            graph = None
            for n in range(3):
                xv = T16.latent(b, g)
                i, w = R.random_route(b, g)
                x.copy_(xv)
                ids.copy_(i)
                wts.copy_(w.to(torch.bfloat16))
                for t in (*outs.values(), *partials.values()):
                    t.fill_(float("nan"))
                if n < 2:
                    call()
                else:
                    graph = graph or T16.graph_of(call)
                    for t in (*outs.values(), *partials.values()):
                        t.fill_(float("nan"))
                    graph.replay()
                torch.cuda.synchronize()
                name = f"b{b}.call{n}"
                got = {a: (partials[a] if wide else outs[a]) for a in arms}
                differ = int((got["D"].view(torch.int16) != got["P"].view(torch.int16)).sum())
                rows = M.expert_rows(logical, xv, i)
                ref = R.weighted(rows, wts) if wide else rows
                res = {"kind": "check", "item": name, "bitwise_cells_differ": differ}
                for a in arms:
                    view = got[a] if wide else got[a].view(b, 16, -1)
                    finite = bool(torch.isfinite(view.float()).all())
                    fro, worst = rel(view, ref) if finite else (float("inf"), float("inf"))
                    res[f"{a}_fro"], res[f"{a}_row_max"], res[f"{a}_finite"] = fro, worst, finite
                H.record(res)
                print("CHECK", json.dumps(res), flush=True)
                if differ:
                    fail(f"{name}.bitwise: {differ} cells of D differ from P")
                p_ok = res["P_fro"] <= FRO and res["P_row_max"] <= ROW
                d_ok = res["D_fro"] <= FRO and res["D_row_max"] <= ROW
                if not p_ok:
                    print(f"VOID {name}.fp64: arm P misses the bound (fro {res['P_fro']:.3e} row {res['P_row_max']:.3e})", flush=True)
                    H.record({"kind": "void", "item": f"{name}.fp64"})
                elif not d_ok:
                    fail(f"{name}.fp64: D fro {res['D_fro']:.3e} row {res['D_row_max']:.3e}")
    except Exception as e:  # noqa: BLE001
        fail(f"{type(e).__name__}: {e}")
        traceback.print_exc()
    with open(os.path.join(args.out, f"wide_depad_{tag}.failures.json"), "w") as fh:
        json.dump(fails, fh, indent=1)
    print(f"RESULT failures={len(fails)} ({time.time() - t0:.0f}s)", flush=True)
    sys.exit(1 if fails else 0)


if __name__ == "__main__":
    main()
