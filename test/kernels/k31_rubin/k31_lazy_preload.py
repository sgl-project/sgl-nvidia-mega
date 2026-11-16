# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Lazy module loading vs a spinning MK consumer, and the ``K31MK_PRELOAD=1`` fix.

CUDA PG 4.8.5.1: under lazy loading a kernel loads at its first launch, and the load can wait for
the kernels already running. Here the MK's AttnRes ring consumer (``PreparedAttnRes.run_ring``,
deadline 3 s) is launched first and spins on its armed mailbox; then a publisher P that writes the
rows is launched on a second stream. If P is not loaded, its first launch waits for the consumer,
the consumer never sees the rows, and it fail-stops at its deadline: HANG.

Arms (the runner sets CUDA_MODULE_LOADING, K31MK_PRELOAD and K31_MUTATION; one process per trial,
since a kernel loads once per process):
  lazy                LAZY, P = a CuTe kernel compiled but never launched           expect HANG
  lazy_preload        LAZY + K31MK_PRELOAD=1: the MK prepare hook's preload()      expect PASS
                      runs after compile, before the consumer
  lazy_warm           LAZY, P launched once before (scratch mailbox)               expect PASS (control: P's first load)
  eager               EAGER, CuTe P (its library still loads at P's first call)    informative
  lazy_torch          LAZY, P = torch._sleep + copy (the Rubin ringdiag pattern)   expect HANG (calibration)
  eager_torch         EAGER, torch P                                               expect PASS (calibration)
  lazy_torch_preload  LAZY + preload, torch P                                      expect HANG (preload covers CuTe only)
Mutation ``preload_empty`` (the preload finds no kernel to load) with lazy_preload: expect HANG.

PASS: synchronize returns in < 1.5 s; output within 1e-3 + 2^-7 |ref| of the FP32 reference
(``_ref_attnkda.attnres``), new prefix and bank bit-exact, the read mailbox rows re-armed to the
sentinel. HANG: a CUDA error after >= 2.7 s (the consumer's fail-stop), or the runner's watchdog
killed the process. Anything else is ERROR (or WRONG for a numeric failure).

Usage: python3 k31_lazy_preload.py ARM --out DIR [--trial N]   (exit 0 PASS, 2 HANG, 1 otherwise)
"""

import argparse
import json
import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _harness as H  # noqa: E402

MUTATIONS = {
    # The preload's enumeration yields no kernel (e.g. the DSL class or library layout changed).
    "preload_empty": (
        "kimi_k31/k3/preload.py",
        "functions = kernel_functions(compiled, materialize=True)",
        "functions = []",
    ),
}

ARMS = {  # arm -> publisher kind
    "lazy": "cute",
    "lazy_preload": "cute",
    "lazy_warm": "cute",
    "eager": "cute",
    "lazy_torch": "torch",
    "eager_torch": "torch",
    "lazy_torch_preload": "torch",
}
DEADLINE_NS = 3_000_000_000
ROWS, NB = 4, 4


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("arm", choices=sorted(ARMS))
    ap.add_argument("--out", required=True)
    ap.add_argument("--trial", type=int, default=0)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    tag = os.environ.get("K31_MUTATION") or "base"
    os.environ["K31_RESULTS"] = os.path.join(args.out, f"lazy_{args.arm}_{tag}_t{args.trial}.jsonl")
    torch.cuda.set_device(0)
    row = {"kind": "lazy_preload", "arm": args.arm, "mutation": tag, "trial": args.trial, "loading": H.loading_mode()}
    H.load_plugin(MUTATIONS)
    import _ref_attnkda as R
    import test_k31_attnres as T
    from sgl_nvidia_mega.kimi_k31.k3 import preload
    from sgl_nvidia_mega.kimi_k31.k3.cutedsl.attn_res.prepared import PreparedAttnRes

    row["preload_enabled"] = preload.ENABLED
    want = ("EAGER" if args.arm.startswith("eager") else "LAZY", "preload" in args.arm)
    if (row["loading"], preload.ENABLED) != want:
        row["outcome"] = "MISCONFIGURED"
        H.record(row)
        print("RESULT", json.dumps(row), flush=True)
        sys.exit(1)

    g = H.generator(4242)
    target = T._target(g, NB, -1, False)
    prep = PreparedAttnRes.for_target(target, deadline_ns=DEADLINE_NS, has_delta=True)
    norm, query, out_norm = T._weights(target, False)
    delta, prefix0, blocks0 = T._inputs(g, ROWS, NB)
    ref_out, ref_prefix, ref_blocks = R.attnres(
        delta, prefix0, blocks0, nb=NB, has_delta=True, write=-1,
        norm_w=norm.weight, query_w=query.weight, out_w=out_norm.weight, eps=T.EPS, out_eps=T.EPS,
    )
    prefix, blocks = prefix0.clone(), blocks0.clone()
    out = torch.empty((ROWS, 7168), dtype=torch.bfloat16, device="cuda")
    H.poison(out)
    mailbox = torch.empty((1, 64, 7168), dtype=torch.bfloat16, device="cuda")
    mailbox.view(torch.int32).fill_(T.SENTINEL)
    prep.bind_ring(mailbox)
    code = None
    if ARMS[args.arm] == "cute":
        publish = T._publish_fn(0, delta, mailbox)  # compiled here, first launched after the consumer
        code = T._PUBLISH_CODE[(0, tuple(delta.shape))]
        if args.arm == "lazy_warm":
            scratch = torch.empty_like(mailbox)
            T._publish_fn(0, delta, scratch)()
    else:

        def publish():
            torch.cuda._sleep(2_000_000)
            mailbox[0, :ROWS].copy_(delta)

    torch.cuda.synchronize()
    row["preload"] = H.maybe_preload("reproducer")
    row["audit_before_launch"] = preload.audit()
    if code is not None:
        functions = preload.kernel_functions(code, materialize=False)
        row["publisher_loaded_before_launch"] = (
            None if functions is None else all(preload.is_loaded(f) for f in functions)
        )
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    torch.cuda.synchronize()
    t0 = time.time()
    try:
        prep.run_ring(prefix, blocks, out)  # spins until the rows arrive or the deadline traps
        with torch.cuda.stream(side):
            publish()
        torch.cuda.synchronize()
        row["elapsed_s"] = round(time.time() - t0, 3)
        words = mailbox[0, :ROWS].view(torch.int32)
        checks = {
            "rearmed": bool((words == T.SENTINEL).all()),
            "prefix_exact": torch.equal(prefix.view(torch.int16), ref_prefix.view(torch.int16)),
            "bank_exact": torch.equal(blocks.view(torch.int16), ref_blocks.view(torch.int16)),
        }
        try:
            stats = H.check(f"lazy.{args.arm}.out", out, ref_out, 1e-3, 2**-7)
            checks["out_max_abs"] = stats["max_abs"]
            checks["out_ok"] = True
        except AssertionError as e:
            checks["out_ok"] = False
            checks["out_error"] = str(e)[:200]
        row["checks"] = checks
        ok = checks["out_ok"] and checks["rearmed"] and checks["prefix_exact"] and checks["bank_exact"]
        row["outcome"] = ("PASS" if row["elapsed_s"] < 0.5 * DEADLINE_NS / 1e9 else "SLOW") if ok else "WRONG"
    except Exception as e:  # noqa: BLE001 (the consumer's fail-stop surfaces as a CUDA error)
        row["elapsed_s"] = round(time.time() - t0, 3)
        row["error"] = f"{type(e).__name__}: {str(e)[:300]}"
        row["outcome"] = "HANG" if row["elapsed_s"] >= 0.9 * DEADLINE_NS / 1e9 else "ERROR"
    H.record(row)
    print("RESULT", json.dumps(row), flush=True)
    sys.exit({"PASS": 0, "HANG": 2}.get(row["outcome"], 1))


if __name__ == "__main__":
    main()
