# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""K31MK_MOE_WIDE_FOLD (wide decode M1(a) gate): above 64 rows the routing kernel's copy role (``front.RouteSitu``,
polled, FP32 latent words) writes the wide experts' MXFP8 plan as 64-row blocks (token t at row t % 64 of block t // 64),
so the wide engine skips ``compile_wide_plan`` (``wide_plan.WidePlanKernel``). One GPU, no peers: the copy role reads the
front box rows of this GPU, which the test fills with FP32 latent words (the publish's layout: a box row's first 7168 BF16
elements hold the 3584 FP32 latent words; sentinel 0x80000000 words are never published, -0.0 becomes +0.0).

Checks (written before any run; all bitwise, no tolerance):
  plan     the copy role's plan bytes [blocks x PLAN_BYTES] == WidePlanKernel's bytes from the same FP32 latent, at 65 / 72 /
           96 / 100 / 127 / 128 rows (both buffers prefilled with 0xA5, so rows past the batch must stay untouched in both);
  canary   one PLAN_BYTES block after the plan blocks stays 0xA5 (no store past block ceil(b / 64) - 1);
  latent   the copied FP32 latent == the box words, bit for bit;
  rearm    every box word the copy role consumed is the sentinel again afterwards;
  repeat   a second call on fresh box words (another seed) matches its own WidePlanKernel reference (no state carried over).
The reference is an independent kernel (``WidePlanKernel``: one CTA per token, the 64-row ``plan_chunk`` quantization from the
FP32 latent); the 64-row plan quantization itself is covered by the served path's tests. Expected values never come from
the code under test.
Mutations (``MUTATIONS``, each must fail): fold_block_mod (block index token % 2 instead of token // 64), fold_row_nomod
(row token instead of token % 64), fold_blocks_dropped (the blocks branch removed: every token into block 0 at row token).

Usage (one GPU): python3 k31_wide_plan_route.py --out DIR [--batches 65,72,96,100,127,128]
Exit status 0 only when every check passed (``RESULT failures=0``).
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

F = "kimi_k31/k3/cutedsl/moe/front.py"
BLOCK = "                block = view(plan.iterator, token // 64 * PLAN_BYTES, c.Uint8, PLAN_BYTES)\n"
ROW = "                plan_chunk(block, token % 64, chunk0 + tid + 256 * i, words, fp32=self.fp32)\n"
MUTATIONS = {
    "fold_block_mod": (F, BLOCK, BLOCK.replace("token // 64 * PLAN_BYTES", "token % 2 * PLAN_BYTES")),
    "fold_row_nomod": (F, ROW, ROW.replace("plan_chunk(block, token % 64,", "plan_chunk(block, token,")),
    "fold_blocks_dropped": (F, "            elif c.const_expr(plan is not None and self.blocks > 1):",
                            "            elif c.const_expr(plan is not None and self.blocks > 1 and False):"),
}
SENTINEL = 0x80000000
WIDTH = 7168 + 2 * 896  # BF16 elements per box row: FP32 latent words, then FP32 logit words


def latent_words(b, seed, device):
    g = torch.Generator(device=device).manual_seed(seed)
    x = torch.randn((b, 3584), generator=g, device=device, dtype=torch.float32)
    # spread magnitudes per 32-element block (exercises the shared exponents), plus exact zeros and tiny values
    x = x * torch.exp2(torch.randint(-20, 12, (b, 112, 1), generator=g, device=device).float()).repeat_interleave(32, 1).view(b, 3584)
    x[:, 5::97] = 0.0
    x[:, 7::131] = 1e-30
    bits = x.view(torch.int32)
    bits[bits == -(2 ** 31)] = 0  # -0.0 -> +0.0 (the publish's positive_zero32; never the sentinel)
    return x


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--batches", default="65,72,96,100,127,128")
    ap.add_argument("--deadline-s", type=float, default=2.0)
    args = ap.parse_args()
    tag = os.environ.get("K31_MUTATION") or "base"
    os.makedirs(args.out, exist_ok=True)
    os.environ["K31_RESULTS"] = os.path.join(args.out, f"wide_plan_route_{tag}.jsonl")
    H.load_plugin(MUTATIONS)
    import cutlass as c

    from sgl_nvidia_mega.kimi_k31.k3.cutedsl.common.host import stream, to_cute
    from sgl_nvidia_mega.kimi_k31.k3.cutedsl.moe import experts_runtime as ER
    from sgl_nvidia_mega.kimi_k31.k3.cutedsl.moe import front as FR
    from sgl_nvidia_mega.kimi_k31.k3.cutedsl.moe.experts import PLAN_BYTES

    device = torch.device("cuda", 0)
    torch.cuda.set_device(device)
    fails = []
    t0 = time.time()
    rows, shared = 128, 384
    try:
        for b in map(int, args.batches.split(",")):
            blocks = -(-b // 64)
            route = FR.compile_route(b, shared, True, (False, True, False), device.index, True, True, int(args.deadline_s * 1e9),
                                     fp32=True, rows=rows, blocks=blocks)
            ref_writer = ER.compile_wide_plan(b, device.index)
            f32, bf16, i32 = torch.float32, torch.bfloat16, torch.int32
            logits, bias = torch.zeros((b, 896), dtype=f32, device=device), torch.zeros(896, dtype=f32, device=device)
            ids, weights = torch.zeros((b, 16), dtype=i32, device=device), torch.zeros((b, 16), dtype=bf16, device=device)
            gate_up, hidden = torch.zeros((b, 2 * shared), dtype=f32, device=device), torch.zeros((b, shared), dtype=bf16, device=device)
            box = torch.empty((1, rows, WIDTH), dtype=bf16, device=device)
            for call, seed in enumerate((1000 + b, 2000 + b)):
                x = latent_words(b, seed, device)
                box.view(torch.int32).fill_(SENTINEL - 2 ** 32)  # every word armed (int32 view of 0x80000000)
                box[0, :b, :7168].view(torch.int32).copy_(x.view(torch.int32))
                latent = torch.full((b, 3584), float("nan"), dtype=f32, device=device)
                plan = torch.full(((blocks + 1) * PLAN_BYTES,), 0xA5, dtype=torch.uint8, device=device)  # + one canary block
                ref = torch.full((blocks * PLAN_BYTES,), 0xA5, dtype=torch.uint8, device=device)
                torch.cuda.synchronize()
                s = stream(device)
                route(to_cute(logits), to_cute(bias), to_cute(ids), to_cute(weights), to_cute(box), to_cute(latent),
                      to_cute(gate_up), to_cute(hidden), *(c.Float32(v) for v in (1.0, 4.0, 25.0)),
                      to_cute(plan[: blocks * PLAN_BYTES]), s)
                ref_writer(to_cute(x, 16), to_cute(ref), s)
                torch.cuda.synchronize()
                name = f"b{b}.call{call}"
                diff = int((plan[: blocks * PLAN_BYTES] != ref).sum())
                canary = int((plan[blocks * PLAN_BYTES:] != 0xA5).sum())
                lat = bool(torch.equal(latent.view(torch.int32), x.view(torch.int32)))
                armed = bool((box[0, :b, :7168].view(torch.int32) == SENTINEL - 2 ** 32).all())
                row = {"kind": "check", "item": name, "plan_bytes_differ": diff, "canary_bytes_changed": canary,
                       "latent_bitwise": lat, "rearmed": armed, "plan_bytes": blocks * PLAN_BYTES}
                H.record(row)
                print("CHECK", json.dumps(row), flush=True)
                if diff:
                    fails.append(f"{name}.plan: {diff} bytes differ from WidePlanKernel")
                if canary:
                    fails.append(f"{name}.canary: {canary} bytes written past block {blocks - 1}")
                if not lat:
                    fails.append(f"{name}.latent: copy differs from the box words")
                if not armed:
                    fails.append(f"{name}.rearm: consumed box words not re-armed")
    except Exception as e:  # noqa: BLE001
        fails.append(f"{type(e).__name__}: {e}")
        traceback.print_exc()
    for f in fails[:20]:
        print("FAIL", f, flush=True)
    with open(os.path.join(args.out, f"wide_plan_route_{tag}.failures.json"), "w") as fh:
        json.dump(fails, fh, indent=1)
    print(f"RESULT failures={len(fails)} ({time.time() - t0:.0f}s)", flush=True)
    sys.exit(1 if fails else 0)


if __name__ == "__main__":
    main()
