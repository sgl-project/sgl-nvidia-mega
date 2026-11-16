# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Per-step GPU time of the Kimi K3.1 megakernel's kernel families in a torch-profiler trace
(SGLang /start_profile output of one rank). PDL overlaps neighbouring kernels, so the family
sums exceed the step: the report gives both the summed kernel time and the union (wall) time
of each family, the step wall time (graph span / steps) and the GPU-idle share.

Usage: python3 mk_trace_families.py TRACE.json[.gz] [--steps N] [--layers 93]
``--steps``: decode steps in the window (default: occurrences of the final AttnRes / 1, or
of KdaSplit / 69 when no final marker is recognized).
"""

import argparse
import collections
import gzip
import json
import re

FAMILIES = [  # (family, regex on the kernel name), first match wins
    ("kda_core", r"KdaSplit"),
    ("attn_res", r"ClusterAttnRes|AuxMixture|attn_res"),
    ("projection_splitk", r"SplitKGemm|splitk"),
    ("oproj_producer", r"OprojProducer|OprojUmma|OprojFused"),
    ("oproj_consumer", r"OprojConsumer"),
    ("twoshot", r"twoshot|TwoShot"),
    ("moe_front_b1", r"MoeFrontTP8|MoeFrontTP16"),
    ("moe_shared_b1", r"MoeSharedTP8"),
    ("moe_experts_b1", r"ClusterExperts"),
    ("moe_experts", r"ExpertsKernel|ExpertsGroup|ExpertsPlan"),
    ("moe_front", r"WsGemm|Fp32FrontGemm|Publish|RouteSitu|route"),
    ("moe_shared", r"SharedUmma|SharedDown|Shared"),
    ("moe_tail", r"WaveMoeTail|MoeTail|push"),
    ("moe_dense", r"DenseSitu"),
    ("mla_query", r"IndexMlaQuery|MlaQuery"),
    ("mla_value", r"MlaValue|DcpValue"),
    ("mla_trtllm_fmha", r"fmha|Fmha|trtllm"),
    ("dsa_indexer_logits", r"paged_mqa|mqa_logits|fp8_paged"),
    ("dsa_topk", r"topk|rank_topk|radix"),
    ("dsa_other", r"act_quant|head_gate|rmsnorm|layernorm|index"),
    ("lm_head_gemm", r"nvjet|gemm|Gemm|cutlass"),
    ("nccl_allreduce", r"nccl|all_reduce|allreduce|cross_device"),
    ("embedding", r"embedding|Embedding"),
    ("sampling", r"sampl|argmax|softmax|topp|top_k"),
    ("memcpy", r"Memcpy|memcpy|Memset|memset"),
]


def family(name):
    for fam, pattern in FAMILIES:
        if re.search(pattern, name):
            return fam
    return "other"


def union(intervals):
    total, end = 0.0, -1.0
    for s, e in sorted(intervals):
        if e <= end:
            continue
        total += e - max(s, end)
        end = e
    return total


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("trace")
    ap.add_argument("--steps", type=int, default=0)
    args = ap.parse_args()
    opener = gzip.open if args.trace.endswith(".gz") else open
    events = json.load(opener(args.trace))["traceEvents"]
    kernels = [e for e in events if e.get("ph") == "X" and e.get("cat") in ("kernel", "gpu_memcpy", "gpu_memset")]
    if not kernels:
        print("no GPU kernels")
        return
    by = collections.defaultdict(list)
    names = collections.defaultdict(collections.Counter)
    for e in kernels:
        f = family(e["name"])
        by[f].append((e["ts"], e["ts"] + e["dur"]))
        names[f][re.sub(r"<.*|_object_at_.*", "", e["name"])[:80]] += 1
    t0 = min(e["ts"] for e in kernels)
    t1 = max(e["ts"] + e["dur"] for e in kernels)
    kda = len(by.get("kda_core", ()))
    steps = args.steps or (max(1, round(kda / 69)) if kda else 1)
    busy = union([iv for ivs in by.values() for iv in ivs])
    print(f"{args.trace}: {len(kernels)} kernels, {steps} steps, span {(t1 - t0) / 1e3:.2f} ms, "
          f"per step {(t1 - t0) / steps:.1f} us, GPU busy {busy / (t1 - t0) * 100:.1f}%")
    print(f"{'family':22s} {'sum us/step':>11s} {'wall us/step':>12s} {'launch/step':>11s}  top kernel")
    rows = sorted(((f, v) for f, v in by.items() if v), key=lambda kv: -sum(e - s for s, e in kv[1]))
    for fam, ivs in rows:
        tot = sum(e - s for s, e in ivs)
        top = names[fam].most_common(1)[0][0]
        print(f"{fam:22s} {tot / steps:11.1f} {union(ivs) / steps:12.1f} {len(ivs) / steps:11.1f}  {top}")


if __name__ == "__main__":
    main()
