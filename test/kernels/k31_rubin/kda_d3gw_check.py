# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# Bitwise check of the KdaSplit opts "tsp2" /
# "g4" when composed with "wy": kda_d3g_check.py with BASE = the served ON string + "wy" (env D3G_BASE)
# and one more engagement requirement: kda.verify.WY_BUILT must show wy compiled (True, 2 CTAs, 4 rows per thread, 16 lanes) for BASE and
# every arm, so a silent fall back to the token loop cannot pass. Per Breq: BASE and BASE + each of ARMS run through
# test_k31_kda.test_kda_verify (its FP32-reference gates: output, rings, log-g, beta, windows, untouched state / conv); the arm must then
# be BIT-IDENTICAL to BASE in the output and every ring / window (the claim: tsp2 / g4 leave wy's inputs bit-identical), and
# D3G_BUILT must show the arm's paths compiled and BASE's not. K31_KDA_DECAY = std / wd / md. K31_MUTATION = one of MUT: CAUGHT / MISSED.
# D3G_ARM_ONLY=1: each arm alone through test_kda_verify's gates (sanitizer runs).
# Usage (cwd = plugin tree, PYTHONPATH = plugin tree): python3 d3gw_check.py REQUESTS [REQUESTS ...]
import json
import os
import sys

sys.path.insert(0, os.path.join(os.getcwd(), "test", "kernels", "k31_rubin"))
os.environ.setdefault("K31_TEST_HEADS", "6")
import test_k31_kda as T  # noqa: E402

KP = "kimi_k31/k3/cutedsl/kda/kernel.py"
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
    # Engagement slip: wy silently not built at 2-CTA clusters (the token loop runs instead; outputs stay within the
    # reference gates and tsp2 / g4 stay bitwise to that base), so only the WY_BUILT requirement can catch it.
    "wy_engage_drop": (KP, "        self.wy = \"wy\" in self.opt and not self.chunk and self.cluster in (1, 2, 4)\n",
                       "        self.wy = \"wy\" in self.opt and not self.chunk and self.cluster in (1, 4)\n"),
}
MUT["san_pos_tsp2_nobar"] = (KP, "            reductions[warp % 4 + 4 * section + 8 * (first + index)] = total  # base slots (q 0-3, k 4-7)\n"
                             "        subset_barrier(2 + section, 128)  # the section's four warps\n",
                             "            reductions[warp % 4 + 4 * section + 8 * (first + index)] = total  # base slots (q 0-3, k 4-7)\n")
T.H.load_plugin({**T.MUTATIONS, **MUT})
import torch  # noqa: E402
from sgl_nvidia_mega.kimi_k31.k3.cutedsl.kda import verify as V  # noqa: E402

MUTANT = os.environ.get("K31_MUTATION", "")
REGIME = os.environ.get("K31_KDA_DECAY", "std")
BASE = os.environ.get("D3G_BASE", "early+wide+rs+lt+tsp+dmc+chunk+cg+cs+wlf+wy")
ARMS = os.environ.get("D3G_ARMS", "g4,tsp2,tsp2+g4").split(",")
EXPECT = {"g4": (False, 4), "tsp2": (True, 1), "tsp2+g4": (True, 4)}
WY_EXPECT = (True, 2, 4, 16) if "wy" in BASE.split("+") else None


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
    key = (requests, T.Q, frozenset(opt.split("+")))
    return gate, V.D3G_BUILT.get(key), V.WY_BUILT.get(key), dict(T._LAST)


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


def wy_ok(built):
    return WY_EXPECT is None or built == WY_EXPECT


bad_total = 0
if os.environ.get("D3G_ARM_ONLY") == "1":  # sanitizer runs: each arm alone through test_kda_verify's gates
    for requests in map(int, sys.argv[1:]):
        for arm in ARMS:
            ga, built_a, wy_a, _ = run(requests, BASE + "+" + arm)
            print("D3G_ARM " + json.dumps({"breq": requests, "arm": arm, "built_arm": built_a, "wy_arm": wy_a, "gate": ga}), flush=True)
            bad_total += ga != "pass" or built_a != EXPECT[arm] or not wy_ok(wy_a)
    print(f"D3G_ARM_ONLY {'PASS' if not bad_total else 'FAIL'} ({bad_total})", flush=True)
    sys.exit(0)
for requests in map(int, sys.argv[1:]):
    gb, built_b, wy_b, b = run(requests, BASE)
    for arm in ARMS:
        ga, built_a, wy_a, a = run(requests, BASE + "+" + arm)
        problems = compare(b, a) if gb == "pass" or MUTANT else ["BASE gate " + gb]
        if ga != "pass":
            problems.append("arm gate " + ga)
        if built_b != (False, 1) or built_a != EXPECT[arm]:
            problems.append(f"engagement base {built_b} arm {built_a} (expected {EXPECT[arm]})")
        if not (wy_ok(wy_b) and wy_ok(wy_a)):
            problems.append(f"wy engagement base {wy_b} arm {wy_a} (expected {WY_EXPECT})")
        rec = {"breq": requests, "regime": REGIME, "mutant": MUTANT or None, "arm": arm, "built_base": built_b, "built_arm": built_a,
               "wy_base": wy_b, "wy_arm": wy_a, "base_gate": gb, "problems": problems}
        print("D3G_CHECK " + json.dumps(rec), flush=True)
        bad_total += bool(problems)
if MUTANT:
    print(f"D3G_MUTANT {MUTANT} {REGIME} {'CAUGHT' if bad_total else 'MISSED'}", flush=True)
else:
    print(f"D3G_CLEAN {REGIME} {'PASS' if not bad_total else 'FAIL'} ({bad_total} arm-Breq cells with problems)", flush=True)
