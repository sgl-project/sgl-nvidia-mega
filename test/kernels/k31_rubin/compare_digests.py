# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Bitwise comparison of the output digests (``_harness.digest``, K31_DIGEST=1) of test runs against a reference run.

Usage: python3 compare_digests.py REF_DIR OTHER_DIR [OTHER_DIR...]
Each DIR holds the tests' ``*.jsonl`` result files. A comparison passes when both runs recorded the same non-empty set
of (file, item) digests and every digest is equal. Also prints each run's preload and end-of-run audit records.
Exit 0 only when every comparison passes.
"""

import glob
import json
import os
import sys


def load(directory):
    digests, preload, audits = {}, [], []
    for path in sorted(glob.glob(os.path.join(directory, "*.jsonl"))):
        name = os.path.basename(path)
        for line in open(path):
            row = json.loads(line)
            if row.get("kind") == "digest":
                key = (name, row["item"])
                if key in digests and digests[key] != row["sha256"]:
                    digests[key] = "conflict:" + row["sha256"]
                else:
                    digests[key] = row["sha256"]
            elif row.get("kind") == "preload":
                preload.append(row)
            elif row.get("kind") == "preload_audit":
                audits.append({k: row[k] for k in ("where", "loading", "compiled", "unmaterialized", "kernels", "unloaded")})
    return digests, preload, audits


def main():
    ref_dir, others = sys.argv[1], sys.argv[2:]
    ref, ref_pre, ref_audits = load(ref_dir)
    ok = bool(ref)
    print(f"DIGESTS ref={ref_dir}: {len(ref)} items")
    for directory in [ref_dir] + others:
        _, pre, audits = load(directory)
        modes = sorted({r["loading"] for r in pre + audits})
        loaded = sum(r["loaded_now"] for r in pre)
        print(f"  {os.path.basename(directory)}: loading={modes} preload_calls={len(pre)} loaded_by_preload={loaded} "
              f"end_audits={audits}")
    for directory in others:
        got, _, _ = load(directory)
        missing = sorted(set(ref) - set(got))
        extra = sorted(set(got) - set(ref))
        diff = sorted(k for k in set(ref) & set(got) if ref[k] != got[k])
        passed = bool(got) and not missing and not extra and not diff
        ok &= passed
        print(f"BITWISE {os.path.basename(directory)} vs {os.path.basename(ref_dir)}: {'PASS' if passed else 'FAIL'} "
              f"items={len(got)} missing={len(missing)} extra={len(extra)} differ={len(diff)} {diff[:5]}")
    print(f"CHAIN_BITWISE {'PASS' if ok else 'FAIL'}")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
