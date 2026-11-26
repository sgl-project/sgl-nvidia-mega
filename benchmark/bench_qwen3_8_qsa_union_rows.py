# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Microbenchmark: slot qsa_pool_fp8's lower row bound. One QSA layer's target-verify
sparse attention on an FP8 E4M3 KV pool, microseconds per call, on one SM10x GPU.

  union  what the slot runs (runtime/attention.UnionKernels): the BF16 -> E4M3 pool write
         of the new K/V, the token union, _qsa_pool_union_ws (PDL) and the merge;
  stock  what the executor falls back to: SGLang's QwenSparseAttnBackend
         ._try_fused_kv_attention, the target verify's path on such a pool (the torch casts
         of the new K/V, _qsa_fused_kv_prepare, TRT-LLM decode), called on a backend
         object carrying only the attributes that path reads.

Neither includes the output gate or o_proj, which follow both the same way. The shapes
are the TP1 layer's (24 query heads, 2 KV heads, dim 256, 512 selected blocks at compress
ratio 4) for 4-row chain requests, every request at the same context. Token slots sit in
shuffled 64-token pages. Each request's rows share a random 512-block selection, every
row swapping --overlap's complement of it for other blocks: at the default 0.9 a
request's 4 rows cover about 710 distinct blocks, at 1.0 512 (contexts of at most 2048
tokens select every block). Seeds are fixed.

Each (rows, context) case captures CUDA graphs of --inner calls and replays them --reps
times (median per replay, CUDA events). "cold": every call follows a write of 2x the L2
size (the rest of the layer runs between two attentions), and a graph of the writes alone
is subtracted; "warm": calls back to back. rel_l2 compares the two outputs once.

  python benchmark/bench_qwen3_8_qsa_union_rows.py [--rows 36,40,...] [--contexts 2048,...]
"""

from __future__ import annotations

import argparse
import csv
import statistics
from types import SimpleNamespace

import torch

HEADS, KV_HEADS, DIM, BLOCKS, RATIO, WIDTH, PAGE = 24, 2, 256, 512, 4, 4, 64
SCALE = DIM**-0.5


def selection(requests, blocks, overlap, g):
    """[requests * 4, 512] int32 block indices (CPU): one random selection a request,
    each row swapping round(512 * (1 - overlap)) of it for other blocks, in random
    order; -1 pads a context with fewer than 512 complete blocks."""
    if blocks <= BLOCKS:
        row = torch.arange(BLOCKS, dtype=torch.int32)
        row[blocks:] = -1
        return row.expand(requests * WIDTH, BLOCKS).contiguous()
    out = torch.empty(requests * WIDTH, BLOCKS, dtype=torch.int32)
    for r in range(requests):
        order = torch.randperm(blocks, generator=g)
        base, spare = order[:BLOCKS], order[BLOCKS:]
        swap = min(round(BLOCKS * (1.0 - overlap)), spare.numel())
        for j in range(WIDTH):
            row = base.clone()
            if swap:
                at = torch.randperm(BLOCKS, generator=g)[:swap]
                row[at] = spare[torch.randperm(spare.numel(), generator=g)[:swap]]
            out[r * WIDTH + j] = row[torch.randperm(BLOCKS, generator=g)]
    return out


def case(rows, context, overlap, seed):
    """The verify's tensors: requests at positions context - 4 .. context - 1."""
    g = torch.Generator().manual_seed(seed)
    requests = rows // WIDTH
    tokens = requests * context
    pages = torch.randperm(tokens // PAGE, generator=g)
    t = torch.arange(tokens)
    table = torch.zeros(requests + 1, context, dtype=torch.int32)
    table[1:] = (1 + pages[t // PAGE] * PAGE + t % PAGE).view(requests, context)
    req = torch.arange(1, requests + 1, dtype=torch.int32).repeat_interleave(WIDTH)
    positions = (context - WIDTH + torch.arange(rows) % WIDTH).to(torch.int32)
    # complete compressed blocks of every row (the first row's)
    idx = selection(requests, (context - WIDTH + 1) // RATIO, overlap, g)
    cpu = dict(
        table=table,
        req=req,
        positions=positions,
        lengths=positions + 1,
        loc=table[req.long(), positions.long()].long(),
        idx=idx,
        req_pool=torch.arange(1, requests + 1, dtype=torch.int64),
    )
    c = {k: v.cuda() for k, v in cpu.items()}
    gc = torch.Generator(device="cuda").manual_seed(seed)
    randn = lambda *shape: (
        torch.randn(*shape, generator=gc, device="cuda", dtype=torch.bfloat16) * 0.25
    )
    c["k_pool"] = randn(1 + tokens, KV_HEADS, DIM).to(torch.float8_e4m3fn)
    c["v_pool"] = randn(1 + tokens, KV_HEADS, DIM).to(torch.float8_e4m3fn)
    c["q"] = randn(rows, HEADS * DIM)
    c["k"] = randn(rows, KV_HEADS * DIM)
    c["v"] = randn(rows, KV_HEADS * DIM)
    return c


def stock_call(c):
    """SGLang's target-verify attention on the FP8 pool, as its backend runs it under a
    CUDA graph (preallocated valid counts)."""
    from sglang.srt.layers.attention import qwen_sparse_attn_backend as qsa
    from sglang.srt.model_executor.forward_mode import ForwardMode

    rows = c["q"].shape[0]
    kv = SimpleNamespace(key=lambda: c["k_pool"], value=lambda: c["v_pool"])
    backend = object.__new__(qsa.QwenSparseAttnBackend)
    for name, value in dict(
        _fused_kv_pool_eligible=True,
        token_to_kv_pool=SimpleNamespace(
            layer_router=SimpleNamespace(read=lambda layer_id: kv)
        ),
        compress_ratio=RATIO,
        qsa_profile=SimpleNamespace(block_topk=BLOCKS),
        forward_metadata=SimpleNamespace(
            sequence_lengths=c["lengths"],
            row_req_pool_indices=c["req"],
            indexer_metadata=SimpleNamespace(decode_logical_positions=c["positions"]),
            is_cuda_graph=True,
            fa2_valid_counts=torch.zeros(rows, dtype=torch.int32, device="cuda"),
        ),
        req_to_token_pool=SimpleNamespace(req_to_token=c["table"]),
        _fa2_scratch={},
        _trtllm_sparse_tables={},
        _cuda_graph_max_tokens=rows,
        _trtllm_workspace=None,
        _trtllm_counters={},
    ).items():
        setattr(backend, name, value)
    layer = SimpleNamespace(
        layer_id=0, tp_q_head_num=HEADS, head_dim=DIM, scaling=SCALE
    )
    fb = SimpleNamespace(
        forward_mode=ForwardMode.TARGET_VERIFY,
        spec_info=SimpleNamespace(draft_token_num=WIDTH),
        out_cache_loc=c["loc"],
        req_pool_indices=c["req_pool"],
    )

    def call():
        out = backend._try_fused_kv_attention(
            c["q"], c["k"], c["v"], layer, fb, c["idx"]
        )
        if out is None:
            raise RuntimeError("SGLang's fused K/V path declined this verify")
        return out

    return call


def union_call(c, kernels, impls):
    rows = c["q"].shape[0]

    def call():
        out, impl = kernels.run(
            q=c["q"].view(rows, HEADS, DIM),
            key_cache=c["k_pool"],
            value_cache=c["v_pool"],
            new_k=c["k"].view(rows, KV_HEADS, DIM),
            new_v=c["v"].view(rows, KV_HEADS, DIM),
            locations=c["loc"],
            slot_map=c["table"],
            row_requests=c["req"],
            indices=c["idx"],
            sequence_lengths=c["lengths"],
            query_positions=c["positions"],
            scale=SCALE,
        )
        impls.add(impl)
        return out

    return call


def timed(fn, flush, inner, reps):
    """(cold, warm) microseconds per call."""

    def body(call, flushed):
        for _ in range(inner):
            if flushed:
                flush.zero_()
            if call:
                fn()

    medians = []
    for call, flushed in ((True, True), (False, True), (True, False)):
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            body(call, flushed)
        graph.replay()
        torch.cuda.synchronize()
        times = []
        for _ in range(reps):
            start, end = (torch.cuda.Event(enable_timing=True) for _ in range(2))
            start.record()
            graph.replay()
            end.record()
            end.synchronize()
            times.append(start.elapsed_time(end) * 1000.0 / inner)
        medians.append(statistics.median(times))
        del graph
    return medians[0] - medians[1], medians[2]


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--rows", default=",".join(str(r) for r in range(36, 65, 4)))
    p.add_argument("--contexts", default="2048,16384,65536")
    p.add_argument("--overlap", type=float, default=0.9)
    p.add_argument(
        "--splits", type=int, default=None, help="union splits (slot default)"
    )
    p.add_argument("--inner", type=int, default=10)
    p.add_argument("--reps", type=int, default=30)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--csv", default=None)
    args = p.parse_args()

    import sglang
    import triton

    from sgl_nvidia_mega.qwen3_8.qsa import pool_decode
    from sgl_nvidia_mega.qwen3_8.runtime import attention

    if args.splits is not None:
        attention.UNION_SPLITS = args.splits
    reason = attention.union_build()
    if reason:
        raise SystemExit(f"slot qsa_pool_fp8 would not build here: {reason}")
    device = torch.device("cuda")
    props = torch.cuda.get_device_properties(device)
    flush = torch.empty(2 * props.L2_cache_size, dtype=torch.uint8, device=device)
    kernels = attention.UnionKernels()
    print(
        f"# {props.name} sm={props.multi_processor_count} L2={props.L2_cache_size >> 20}MB "
        f"torch={torch.__version__} triton={triton.__version__} sglang={sglang.__file__}\n"
        f"# union splits={attention.UNION_SPLITS} union_warps={attention.UNION_WARPS} "
        f"overlap={args.overlap} inner={args.inner} reps={args.reps} seed={args.seed}",
        flush=True,
    )
    head = (
        f"{'rows':>4} {'reqs':>4} {'context':>7} | {'union':>8} {'stock':>8} {'ratio':>6}"
        f" | {'union_w':>8} {'stock_w':>8} {'ratio_w':>7} | {'rel_l2':>7}"
    )
    print("# cold / _w warm: us per call; ratio = union / stock (< 1: union faster)")
    print(head, flush=True)
    results = []
    for context in [int(x) for x in args.contexts.split(",")]:
        for rows in [int(x) for x in args.rows.split(",")]:
            c = case(
                rows,
                context,
                args.overlap,
                args.seed * 1_000_003 + rows * 997 + context,
            )
            if not pool_decode.write_new_kv_e4m3_ok(c["k_pool"], c["v_pool"]):
                raise SystemExit("the E4M3 cast write differs from torch's cast here")
            impls = set()
            calls = {"union": union_call(c, kernels, impls), "stock": stock_call(c)}
            outs = {}
            for name, fn in calls.items():
                for _ in range(2):  # compile, caches, scratch: before any capture
                    outs[name] = fn().float()
            torch.cuda.synchronize()
            rel = float((outs["union"] - outs["stock"]).norm() / outs["stock"].norm())
            t = {
                name: timed(fn, flush, args.inner, args.reps)
                for name, fn in calls.items()
            }
            row = dict(
                rows=rows,
                requests=rows // WIDTH,
                context=context,
                union_us=t["union"][0],
                stock_us=t["stock"][0],
                ratio=t["union"][0] / t["stock"][0],
                union_warm_us=t["union"][1],
                stock_warm_us=t["stock"][1],
                ratio_warm=t["union"][1] / t["stock"][1],
                rel_l2=rel,
                union_impl="/".join(sorted(impls)),
            )
            results.append(row)
            print(
                f"{rows:>4} {rows // WIDTH:>4} {context:>7} | {row['union_us']:>8.1f} "
                f"{row['stock_us']:>8.1f} {row['ratio']:>6.3f} | {row['union_warm_us']:>8.1f} "
                f"{row['stock_warm_us']:>8.1f} {row['ratio_warm']:>7.3f} | {rel:>7.1e}"
                + ("" if impls == {"union_ws_fp8"} else f"  [{row['union_impl']}]")
                + ("  [rel_l2 > 0.01]" if rel > 0.01 else ""),
                flush=True,
            )
            del c, calls, outs
            torch.cuda.empty_cache()
    for context in sorted({r["context"] for r in results}):
        rows = [r for r in results if r["context"] == context]
        losing = [r["rows"] for r in rows if r["ratio"] >= 1.0]
        print(
            f"# context {context}: union faster (cold) at rows "
            f"{[r['rows'] for r in rows if r['ratio'] < 1.0]}, not at {losing}"
        )
    if args.csv:
        with open(args.csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(results[0]))
            writer.writeheader()
            writer.writerows(results)


if __name__ == "__main__":
    main()
