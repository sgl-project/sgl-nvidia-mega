# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""K31MK_MOE_W2_ONE96 (wide decode H-MKPREP): ONE resident FC2 copy per TP16 MoE layer. Every MK FC2 reader
takes the de-padded 96-byte rows (this engine's map 1 and packed K64 map, the grouped M64 engine, the wide engine, the
RUBIN8 B8 engine) and ``release_padded_w2`` frees the loaded padded tensor.

One GPU (VR200), TP16 host-layout weights of one rank (``test_k31_experts_tp16.weights16``, all 896 experts,
die-oblivious). Env of both arms: K31MK_MOE_RUBIN64=9-64:1, K31MK_MOE_RUBIN64_W2_DEPAD=1, K31MK_MOE_WIDE=65-128:1,
K31MK_MOE_RUBIN8=1, K31MK_EXPERTS_PACKED_K64=1 (the served c=8 F reader set plus the wide tier).
Arms: T = today's two-copy form (padded w2 for RUBIN8 / wide / our maps, a de-padded copy for the grouped engine);
O = the same + K31MK_MOE_W2_ONE96=1, built over a private clone of the padded w2, whose bytes are then POISONED (0x5A)
and released before O runs a single call (a reader left on the padded storage reads garbage or faults).
T0 / O0 = the same pair with packed=0 (our map 1, the 16U4 FC2 map).
Batches (rows): 2 (T0 / O0, map 1), 4 (packed K64 map), 8 (RUBIN8 B8, image written by ``compile_image_plan``, no
publish: the 16 expert rows per token into ``out``), 16 / 32 / 64 (grouped engine), 72 / 128 (wide engine, publish
("local", partial)). Per batch: 2 eager calls and 1 CUDA-graph replay, new inputs each, the same inputs in both arms.
Checks (written before any run, MOE-wideM1c.2; never loosened after data):
  engaged   O: one vendored object for grouped + wide; every FC2 tensor O holds (``w2_storages``) is ONE storage
            (O0 likewise), not the padded one; O.rubin / O.rubin8 w2_row_bytes 96; O.weights[2] is [896, 3584, 96].
  release   ``release_padded_w2`` returns the padded storage bytes (896 x 3584 x 128 = 411,041,792) and
            torch.cuda.memory_allocated drops by >= 0.99 x that; afterwards the layer's stock forward raises.
  bitwise   O's outputs == T's (O0 == T0 at B2), every BF16 cell, every call.
  fp64      vs ``_ref_moe16.expert_rows`` (FP64, logical weights; wide: routing-weighted sum): Frobenius rel-L2 <= 1e-2
            and every row <= 3e-2. Validity: arm T must pass it; if T fails, the line is VOID for that cell (reference
            problem, not evidence about ONE96) and bitwise stands alone. A batch where arm T raises is VOID as a whole
            (reported, never counted as a pass).
Outputs are NaN-poisoned before every call (unwritten cells fail every line).
Mutations (``MUTATIONS``, each must fail on the named lines): one96_packed_pitch (our packed w2 map keeps the 128-byte
pitch on the 96-byte copy) [b4 bitwise / fp64]; r8_pitch (RubinExpertsKernel's packed w2 map keeps 128) [b8];
wide_kw_dropped (the wide compile loses w2_row_bytes) [b72 / b128]; release_keeps_padded (the release skips the
storage swap) [release.memory].

Usage (one VR200 GPU): python3 k31_one96.py --out DIR. Exit 0 only when every check passed (``RESULT failures=0``).
"""

import argparse
import json
import os
import sys
import time
import traceback
from types import SimpleNamespace

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _harness as H  # noqa: E402
import _ref_moe as R  # noqa: E402
import _ref_moe16 as M  # noqa: E402

ER_FILE = "kimi_k31/k3/cutedsl/moe/experts_runtime.py"
RR_FILE = "kimi_k31/k3/cutedsl/rubin/moe/experts_runtime.py"
RK_FILE = "kimi_k31/k3/cutedsl/rubin/moe/experts_rubin.py"
MUTATIONS = {
    "one96_packed_pitch": (ER_FILE, "pitch = W2_LIVE_BYTES if self.one96 else W2_ROW_BYTES", "pitch = W2_ROW_BYTES"),
    "r8_pitch": (RK_FILE, "pitch = self.w2_row_bytes", "pitch = 128"),
    "wide_kw_dropped": (RR_FILE, '**({"w2_row_bytes": self.w2_row_bytes} if self.w2_row_bytes != 128 else {}),', ""),
    "release_keeps_padded": (ER_FILE, "    w2.data = w2.data.new_empty((0,))\n", "    pass\n"),
}
FRO, ROW = 1e-2, 3e-2
ENV = {"K31MK_MOE_RUBIN64": "9-64:1", "K31MK_MOE_RUBIN64_W2_DEPAD": "1", "K31MK_MOE_WIDE": "65-128:1",
       "K31MK_MOE_RUBIN8": "1", "K31MK_EXPERTS_PACKED_K64": "1"}
PADDED_BYTES = 896 * 3584 * 128


def rel(rows, ref):
    rows, ref = rows.double(), ref.double()
    fro = float((rows - ref).norm() / ref.norm())
    per_row = ((rows - ref).norm(dim=-1) / ref.norm(dim=-1)).flatten()
    return fro, float(per_row.max())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--batches", default="2,4,8,16,32,64,72,128")
    args = ap.parse_args()
    tag = os.environ.get("K31_MUTATION") or "base"
    os.makedirs(args.out, exist_ok=True)
    os.environ["K31_RESULTS"] = os.path.join(args.out, f"one96_{tag}.jsonl")
    os.environ.pop("K31MK_MOE_W2_ONE96", None)
    os.environ.update(ENV)
    H.load_plugin(MUTATIONS)
    import test_k31_experts_tp16 as T16

    from sgl_nvidia_mega.kimi_k31.k3.cutedsl.moe import experts_runtime as ER
    from sgl_nvidia_mega.kimi_k31.k3.cutedsl.rubin.moe import experts_runtime as RE

    device = torch.device("cuda", 0)
    torch.cuda.set_device(device)
    fails, voids = [], []
    t0 = time.time()
    batches = list(map(int, args.batches.split(",")))

    def fail(msg):
        fails.append(msg)
        print("FAIL", msg, flush=True)

    def void(msg):
        voids.append(msg)
        H.record({"kind": "void", "item": msg})
        print("VOID", msg, flush=True)

    def build(prepared, packed, one96):
        if one96:
            os.environ["K31MK_MOE_W2_ONE96"] = "1"
        try:
            kw = {} if packed else {"packed": 0}
            return ER.PersistentExperts(*prepared, beta=R.BETA, linear_beta=R.LIN, activation_floor=False, **kw)
        finally:
            os.environ.pop("K31MK_MOE_W2_ONE96", None)

    def inputs(b, n):
        g = H.generator(9600 + 16 * b + n)
        xv = T16.latent(b, g)
        i, w = R.random_route(b, g)
        return xv, i, w.to(torch.bfloat16)

    def run_arm(ex, b):
        """3 calls (2 eager + 1 graph replay) on one arm; returns [output clone per call]."""
        wide, b8 = b > 64, b == 8
        x = torch.empty((b, R.H_LAT), device="cuda")
        ids = torch.empty((b, 16), dtype=torch.int32, device="cuda")
        wts = torch.empty((b, 16), dtype=torch.bfloat16, device="cuda")
        out = torch.empty((b * 16, R.H_LAT), dtype=torch.bfloat16, device="cuda")
        partial = torch.empty((b, R.H_LAT), dtype=torch.bfloat16, device="cuda")
        buf = ER.scratch(16, device)
        ex.prepare(b, buf, out, ("local", partial) if wide else (), b8)
        image = RE.compile_image_plan(8, device.index) if b8 else None

        def call():
            if image is not None:  # the routing kernel's image plan, written from x (RUBIN8 reads only the image)
                image(RE.to_cute(x, 16), RE.tensor(buf["plan"]), RE.stream(device))
            ex.run(x, ids, wts if wide else None)

        got, graph = [], None
        for n in range(3):
            xv, i, w = inputs(b, n)
            x.copy_(xv)
            ids.copy_(i)
            wts.copy_(w)
            out.fill_(float("nan"))
            partial.fill_(float("nan"))
            if n < 2:
                call()
            else:
                graph = T16.graph_of(call)
                out.fill_(float("nan"))
                partial.fill_(float("nan"))
                graph.replay()
            torch.cuda.synchronize()
            got.append((partial if wide else out).clone())
        return got

    try:
        logical, prepared = T16.weights16()
        # Phase 1: arm T / T0 over the shared padded w2.
        t_out = {}
        arms_t = {"packed": build(prepared, True, False), "map1": build(prepared, False, False)}
        for b in batches:
            ex = arms_t["map1" if b == 2 else "packed"]
            try:
                t_out[b] = run_arm(ex, b)
            except Exception as e:  # noqa: BLE001
                void(f"b{b}: arm T raised {type(e).__name__}: {str(e)[:300]}")
                traceback.print_exc()
        del arms_t
        torch.cuda.synchronize()
        torch.accelerator.empty_cache()
        # Phase 2: O / O0 over a private clone of the padded w2; poison it; release it through O; then run.
        padded = prepared[2].clone()
        o_prepared = (prepared[0], prepared[1], padded, prepared[3])
        arms_o = {"packed": build(o_prepared, True, True), "map1": build(o_prepared, False, True)}
        o, o0 = arms_o["packed"], arms_o["map1"]
        padded_id = padded.untyped_storage().data_ptr()
        st, st0 = o.w2_storages(), o0.w2_storages()
        eng = {
            "O_one96": bool(o.one96) and bool(o0.one96),
            "O_one_object": o.rubin is o.rubin_wide,
            "O_one_storage": len(st) == 1 and len(st0) == 1,
            "O_not_padded": padded_id not in st and padded_id not in st0,
            "O_rows96": o.rubin.w2_row_bytes == 96 and o.rubin8.w2_row_bytes == 96,
            "O_shape": tuple(o.weights[2].shape) == (896, 3584, 96) and o.weights[2].dtype == torch.uint8,
        }
        H.record({"kind": "check", "item": "engaged", **eng})
        print("CHECK engaged", json.dumps(eng), flush=True)
        for key, ok in eng.items():
            if not ok:
                fail(f"engaged.{key}")
        padded.view(torch.uint8).fill_(0x5A)  # poison: a reader still on these bytes reads garbage
        torch.cuda.synchronize()
        routed = SimpleNamespace(w2_weight=padded, forward=lambda *a, **k: "stock")
        before = torch.cuda.memory_allocated(device)
        freed = ER.release_padded_w2(routed, o)
        torch.cuda.synchronize()
        dropped = before - torch.cuda.memory_allocated(device)
        try:
            routed.forward()
            refused = False
        except RuntimeError:
            refused = True
        rel_ok = {"freed_bytes": freed, "dropped_bytes": dropped, "stock_refused": refused}
        H.record({"kind": "check", "item": "release", **rel_ok})
        print("CHECK release", json.dumps(rel_ok), flush=True)
        if freed != PADDED_BYTES:
            fail(f"release.freed: {freed} != {PADDED_BYTES}")
        if dropped < 0.99 * PADDED_BYTES:
            fail(f"release.memory: memory_allocated dropped {dropped} < 0.99 x {PADDED_BYTES}")
        if not refused:
            fail("release.stock_refused: the stock forward still runs")
        del padded, o_prepared
        for b in batches:
            if b not in t_out:
                continue
            ex = o0 if b == 2 else o
            try:
                o_out = run_arm(ex, b)
            except Exception as e:  # noqa: BLE001
                fail(f"b{b}: arm O raised {type(e).__name__}: {str(e)[:300]}")
                traceback.print_exc()
                continue
            wide = b > 64
            for n in range(3):
                xv, i, w = inputs(b, n)
                got = {"T": t_out[b][n], "O": o_out[n]}
                name = f"b{b}.call{n}"
                differ = int((got["O"].view(torch.int16) != got["T"].view(torch.int16)).sum())
                rows = M.expert_rows(logical, xv, i)
                ref = R.weighted(rows, w) if wide else rows
                res = {"kind": "check", "item": name, "bitwise_cells_differ": differ}
                for a, v in got.items():
                    view = v if wide else v.view(b, 16, -1)
                    finite = bool(torch.isfinite(view.float()).all())
                    fro, worst = rel(view, ref) if finite else (float("inf"), float("inf"))
                    res[f"{a}_fro"], res[f"{a}_row_max"], res[f"{a}_finite"] = fro, worst, finite
                H.record(res)
                print("CHECK", json.dumps(res), flush=True)
                if differ:
                    fail(f"{name}.bitwise: {differ} cells of O differ from T")
                if not (res["T_fro"] <= FRO and res["T_row_max"] <= ROW):
                    void(f"{name}.fp64: arm T misses the bound (fro {res['T_fro']:.3e} row {res['T_row_max']:.3e})")
                elif not (res["O_fro"] <= FRO and res["O_row_max"] <= ROW):
                    fail(f"{name}.fp64: O fro {res['O_fro']:.3e} row {res['O_row_max']:.3e}")
    except Exception as e:  # noqa: BLE001
        fail(f"{type(e).__name__}: {e}")
        traceback.print_exc()
    with open(os.path.join(args.out, f"one96_{tag}.failures.json"), "w") as fh:
        json.dump({"fails": fails, "voids": voids}, fh, indent=1)
    print(f"RESULT failures={len(fails)} voids={len(voids)} ({time.time() - t0:.0f}s)", flush=True)
    sys.exit(1 if fails else 0)


if __name__ == "__main__":
    main()
