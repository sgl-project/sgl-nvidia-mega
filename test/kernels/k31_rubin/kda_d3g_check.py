# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# Bitwise check of the default-off KdaSplit opts
# "tsp2" (D3: tsp's token split at 2-CTA clusters) and "g4" (four gate warps of 16 rows) against the served ON verify string (BASE).
# Per Breq: BASE and BASE + each of ARMS run through test_k31_kda.test_kda_verify (its FP32-reference gates: output, rings, log-g, beta,
# windows, untouched state / conv); the arm must then be BIT-IDENTICAL to BASE in the output and every ring / window (expected value =
# the served kernel's own output: the claim is bitwise identity, the reference gates come from test_kda_verify), and kda.verify.D3G_BUILT
# must show the arm's paths compiled and BASE's not. K31_KDA_DECAY = std / wd / md regime. K31_MUTATION = one of MUT: the script reports
# CAUGHT / MISSED instead (a must-fail mutant MISSED = the check cannot see that slip).
# Usage (cwd = plugin tree, PYTHONPATH = plugin tree): python3 d3g_check.py REQUESTS [REQUESTS ...]
# d3g_check2 (sanitizer): = d3g_check + D3G_ARM_ONLY=1 (only the arm(s) through test_kda_verify, for sanitizer runs) + the test
# module's MUTATIONS (sanitizer controls such as ctl_early_cw_first) + san_pos_tsp2_nobar (positive control: prologue_tsp2's section
# barrier dropped, so warps read other warps' reductions slots before they are written: an intra-CTA shared RAW racecheck must flag).
import json
import os
import sys

sys.path.insert(0, os.path.join(os.getcwd(), "test", "kernels", "k31_rubin"))
os.environ.setdefault("K31_TEST_HEADS", "6")
import test_k31_kda as T  # noqa: E402

KP = "kimi_k31/k3/cutedsl/kda/kernel.py"
# One plausible slip each (must FAIL the bitwise check), plus g4_klayout (informational: the naive K layout of a 16-row warp).
MUT = {
    "tsp2_tap_cut": (KP, "        for i in cutlass.range_constexpr(TP + 3):\n            if first + i >= 3:\n",
                     "        for i in cutlass.range_constexpr(TP + 3):\n            if first + i >= 4:\n"),
    "tsp2_factor_scale": (KP, "                factor[0] = factor[0] * scale\n            normalized[u] = qk[u] * factor[0]",
                          "                factor[0] = factor[0] * Float32(1.0)\n            normalized[u] = qk[u] * factor[0]"),
    "tsp2_packet_rot": (KP, "got = cute.arch.shuffle_sync(offer[0], (lane & 28) | ((lane - k) & 3))",
                        "got = cute.arch.shuffle_sync(offer[0], (lane & 28) | ((lane + k) & 3))"),
    "tsp2_v_pair": (KP, "            t_v = 2 * v_pair + j\n", "            t_v = 2 * v_pair + 1 - j\n"),
    "tsp2_beta_writer": (KP, "            if (thread - 128) // TP == partition:\n", "            if (thread - 128) % TP == partition:\n"),
    "g4_row_offset": (KP, "value_base + gate_rows_n * ((thread - MAIN) // 32), lane)",
                      "value_base + gate_rows_n * (((thread - MAIN) // 32) & 1), lane)"),
    "g4_klayout": (KP, "            wide: cutlass.Constexpr = max(1, self.rows // 32) >= 2\n", "            wide: cutlass.Constexpr = per >= 2\n"),
}
MUT["san_pos_tsp2_nobar"] = (KP, "            reductions[warp % 4 + 4 * section + 8 * (first + index)] = total  # base slots (q 0-3, k 4-7)\n"
                             "        subset_barrier(2 + section, 128)  # the section's four warps\n",
                             "            reductions[warp % 4 + 4 * section + 8 * (first + index)] = total  # base slots (q 0-3, k 4-7)\n")
T.H.load_plugin({**T.MUTATIONS, **MUT})
import torch  # noqa: E402
from sgl_nvidia_mega.kimi_k31.k3.cutedsl.kda import verify as V  # noqa: E402

MUTANT = os.environ.get("K31_MUTATION", "")
REGIME = os.environ.get("K31_KDA_DECAY", "std")
BASE = os.environ.get("D3G_BASE", "early+wide+rs+lt+tsp+dmc+chunk+cg+cs+wlf")
ARMS = os.environ.get("D3G_ARMS", "g4,tsp2,tsp2+g4").split(",")
EXPECT = {"g4": (False, 4), "tsp2": (True, 1), "tsp2+g4": (True, 4)}


def run(requests, opt):
    os.environ["K31MK_KDA_VERIFY_OPT"] = opt
    os.environ.pop("K31MK_KDA_VERIFY_SCHEDULE", None)
    gate = "pass"
    try:
        T.test_kda_verify(requests)
    except AssertionError as e:
        gate = "FAIL: " + str(e).replace("\n", " ")[:240]
    except Exception as e:  # build / launch errors count as failures (mutants may not build)
        gate = "ERROR: " + (type(e).__name__ + " " + str(e)).replace("\n", " ")[:240]
    return gate, V.D3G_BUILT.get((requests, T.Q, frozenset(opt.split("+")))), dict(T._LAST)


def compare(b, a):
    lines = []
    for key, dt in (("out", torch.int16), ("rawv", torch.int16), ("rawk", torch.int16), ("log_g", torch.int32), ("beta", torch.int32),
                    ("windows", torch.int16)):
        if key not in b or key not in a:
            lines.append(f"{key} missing")
        elif not torch.equal(b[key].view(dt), a[key].view(dt)):
            n = int((b[key].view(dt) != a[key].view(dt)).sum())
            lines.append(f"{key} differs in {n} cells")
    return lines


bad_total = 0
if os.environ.get("D3G_ARM_ONLY") == "1":  # sanitizer runs: each arm alone through test_kda_verify's gates
    for requests in map(int, sys.argv[1:]):
        for arm in ARMS:
            ga, built_a, _ = run(requests, BASE + "+" + arm)
            print("D3G_ARM " + json.dumps({"breq": requests, "arm": arm, "built_arm": built_a, "gate": ga}), flush=True)
            bad_total += ga != "pass" or built_a != EXPECT[arm]
    print(f"D3G_ARM_ONLY {'PASS' if not bad_total else 'FAIL'} ({bad_total})", flush=True)
    sys.exit(0)
for requests in map(int, sys.argv[1:]):
    gb, built_b, b = run(requests, BASE)
    for arm in ARMS:
        ga, built_a, a = run(requests, BASE + "+" + arm)
        problems = compare(b, a) if gb == "pass" or MUTANT else ["BASE gate " + gb]
        if ga != "pass":
            problems.append("arm gate " + ga)
        if built_b != (False, 1) or built_a != EXPECT[arm]:
            problems.append(f"engagement base {built_b} arm {built_a} (expected {EXPECT[arm]})")
        rec = {"breq": requests, "regime": REGIME, "mutant": MUTANT or None, "arm": arm, "built_base": built_b, "built_arm": built_a,
               "base_gate": gb, "problems": problems}
        print("D3G_CHECK " + json.dumps(rec), flush=True)
        bad_total += bool(problems)
if MUTANT:
    print(f"D3G_MUTANT {MUTANT} {REGIME} {'CAUGHT' if bad_total else 'MISSED'}", flush=True)
else:
    print(f"D3G_CLEAN {REGIME} {'PASS' if not bad_total else 'FAIL'} ({bad_total} arm-Breq cells with problems)", flush=True)
