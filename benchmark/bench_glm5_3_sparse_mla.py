# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Sparse MLA decode microbenchmark (E1): ``ll.sparse_mla_fd`` against PR #19's
``SparseMlaDecode`` (+ ``MlaUvProj``) and the stock TRTLLM-Gen + ``torch.bmm``, on one
SM100-family GPU (GB300).

One CUDA graph holds ``--layers`` (78) layers of the per-layer chain around the
attention, each layer with its own KV region, index list, appended rows, W_UV and o_proj
weight: RoPE + FP8 quant (FlashInfer, PDL) -> KV append (``set_mla_kv_buffer``, PDL) ->
sparse MLA -> v_b -> o_proj (``F.linear``). L2 is flushed before every replay; the
times are per layer (us). PDL is on for every kernel that supports it, so each variant
meets its real producer and consumer. ``bare_*`` columns time 78 attention calls back
to back (no neighbours).

Variants:
  stock      TRTLLM-Gen (as SGLang's captured graphs call it) + torch.bmm
  pr19       SparseMlaDecode (16 warps up to 4 rows, 8 above) + MlaUvProj (8 heads)
  pr19+bmm   SparseMlaDecode + torch.bmm (8 heads)
  ours+pr19uv  ll.sparse_mla_fd one-tile + MlaUvProj (8 heads; a step-level mix)
  ours       ll.sparse_mla_fd one-tile (spread) + torch.bmm
  ours_fold  ll.sparse_mla_fd with the v_b fold (8 heads)

Defaults: 8 heads at 1, 2, 4, 6 rows (TP8: draft decode 1-2, verify 6) and 16 heads at
1, 2, 5, 6, 7 rows (TP4 verify 5 / 6), contexts 1k (every token, -1 tail), 8k and 64k.
``ours`` is what the launch plan runs at that row count: above the one-tile wave the
multi-tile mode (8 heads, <= 35 rows), else the one-tile kernel in several waves, so
``--rows8 12,18,24 --rows16 8,12`` measures the multi-tile crossover (bench_fd.py's).
Writes ``bench_glm5_3_sparse_mla.csv`` (and prints a table) to ``--out``.

Run: ``python benchmark/bench_glm5_3_sparse_mla.py --out <dir>``.
"""

from __future__ import annotations

import argparse
import csv
import math
import os

import torch
import torch.nn.functional as F

from sgl_nvidia_mega.glm5_3.ll import sparse_mla as fd

TOP_K, D_QK, D_V, V_HEAD, QK_NOPE, PAGE = 2048, 576, 512, 256, 192, 64
SM_SCALE = 1.0 / math.sqrt(256)
LOG2E = math.log2(math.e)
HIDDEN = 6144
GRAPH_MAX_SEQ_LEN = 16384
CTX_TOKENS = {"1k": 1100, "8k": 8192, "64k": 65536}


def fp8(shape, g, std=1.4):
    return (
        (torch.randn(shape, generator=g, device="cuda") * std)
        .clamp(-448, 448)
        .to(torch.float8_e4m3fn)
    )


class Pr19:
    """PR #19's sparse MLA decode and value up-projection, compiled as its glue does."""

    def __init__(self):
        import cutlass.cute as cute

        from sgl_nvidia_mega.glm5_3.attention.fused import _allow_cluster16
        from sgl_nvidia_mega.glm5_3.sm103.attention.api import example_args
        from sgl_nvidia_mega.glm5_3.sm103.attention.mla_uv_proj import MlaUvProj
        from sgl_nvidia_mega.glm5_3.sm103.attention.sparse_mla_decode import (
            SparseMlaDecode,
        )

        args = example_args()
        self.sparse = {
            w: _allow_cluster16(
                cute.compile(SparseMlaDecode(w, False, True), *args["sparse"])
            )
            for w in (8, 16)
        }
        self.uv = {cap: cute.compile(MlaUvProj(cap), *args["uv"]) for cap in (8, 16)}

    def attention(self, q, pool, idx, scale_log2, out):
        from cutlass import Float32, Int32, Int64

        from sgl_nvidia_mega.glm5_3.runtime.compile import cu_stream

        rows = q.shape[0]
        z = Int64(0)
        self.sparse[16 if rows <= 4 else 8](
            Int64(q.data_ptr()),
            Int64(pool.data_ptr()),
            Int64(idx.data_ptr()),
            Int64(out.data_ptr()),
            Int64(pool.shape[0]),
            Int32(rows),
            Float32(scale_log2),
            Float32(1.0),
            z,
            z,
            z,
            Int32(0),
            Int32(0),
            z,
            z,
            cu_stream(q.device),
        )
        return out

    def uv_proj(self, w_packed, attn, out):
        from cutlass import Int32, Int64

        from sgl_nvidia_mega.glm5_3.runtime.compile import cu_stream

        rows = attn.shape[0]
        no_slot = (Int64(0), Int64(0), Int64(0), Int64(0), Int32(0))
        self.uv[8 if rows <= 8 else 16](
            Int64(w_packed.data_ptr()),
            Int64(attn.data_ptr()),
            Int64(attn.stride(0)),
            Int64(attn.stride(1)),
            Int64(out.data_ptr()),
            Int64(out.stride(0)),
            Int32(rows),
            *no_slot,
            *no_slot,
            cu_stream(attn.device),
        )
        return out


class TrtllmGen:
    def __init__(self, heads):
        from sglang.srt.layers.attention.trtllm_mla_backend import (
            make_persistent_multi_ctas_kv_counter_buffer,
        )

        self.workspace = torch.zeros(384 << 20, dtype=torch.uint8, device="cuda")
        self.counter = make_persistent_multi_ctas_kv_counter_buffer(
            device=torch.device("cuda"), num_q_heads=heads, max_batch_size=64
        )

    def __call__(self, q, pool, idx, lens):
        import flashinfer.decode

        rows, heads, _ = q.shape
        out = flashinfer.decode.trtllm_batch_decode_with_kv_cache_mla(
            query=q.view(rows, 1, heads, D_QK),
            kv_cache=pool.view(-1, 1, PAGE, D_QK),
            workspace_buffer=self.workspace,
            qk_nope_head_dim=QK_NOPE,
            kv_lora_rank=D_V,
            qk_rope_head_dim=D_QK - D_V,
            block_tables=idx.unsqueeze(1),
            seq_lens=lens,
            max_seq_len=GRAPH_MAX_SEQ_LEN,
            sparse_mla_top_k=TOP_K,
            bmm1_scale=SM_SCALE,
            backend="trtllm-gen",
            skip_softmax_threshold_scale_factor=None,
            sparse_mla_top_k_lens=None,
            multi_ctas_kv_counter_buffer=self.counter,
        )
        return out.view(rows, heads, D_V)


class Point:
    """Static inputs of one (heads, rows, ctx) point: per-layer KV regions, index lists,
    appended slots, W_UV and o_proj weights; RoPE inputs shared."""

    def __init__(self, heads, rows, ctx, layers, g):
        self.heads, self.rows, self.layers = heads, rows, layers
        tokens = CTX_TOKENS[ctx]
        self.region = max(1 << 14, 1 << math.ceil(math.log2(tokens + rows + 64)))
        self.pool = fp8((layers * self.region, D_QK), g)
        cpu = torch.Generator().manual_seed(
            int(torch.randint(0, 2**30, (1,), generator=g, device="cuda"))
        )
        idx = torch.full((layers, rows, TOP_K), -1, dtype=torch.int32)
        lens = torch.empty(layers, rows, dtype=torch.int32)
        locs = torch.empty(layers, rows, dtype=torch.int64)
        for layer in range(layers):
            base = layer * self.region
            slots = (
                torch.randperm(self.region - 1, generator=cpu)[: tokens + rows]
                + 1
                + base
            )
            for r in range(rows):
                length = tokens + r
                pos = torch.randperm(length, generator=cpu)[:TOP_K]
                chosen = slots[pos]
                idx[layer, r, : chosen.numel()] = chosen.to(torch.int32)
                lens[layer, r] = length
            # this step's rows are the last positions of each row and read by every row
            locs[layer] = slots[tokens : tokens + rows]
            idx[layer, :, :rows] = locs[layer].to(torch.int32)[None, :]
        self.idx, self.lens, self.locs = idx.cuda(), lens.cuda(), locs.cuda()
        self.positions = torch.randint(0, 16384, (rows,), device="cuda")
        self.cos_sin = torch.randn(16384, 64, device="cuda").float()
        self.q_nope = torch.randn(rows, heads, D_V, device="cuda").to(torch.bfloat16)
        self.q_pe = torch.randn(rows, heads, D_QK - D_V, device="cuda").to(
            torch.bfloat16
        )
        self.k_nope = torch.randn(rows, D_V, device="cuda").to(torch.bfloat16)
        self.k_pe = torch.randn(rows, D_QK - D_V, device="cuda").to(torch.bfloat16)
        self.kv_b = [
            (
                torch.randn(heads * (QK_NOPE + V_HEAD), D_V, generator=g, device="cuda")
                * 0.05
            ).to(torch.bfloat16)
            for _ in range(layers)
        ]
        # W_UV [H, 256, 512] contiguous (the loader's layout), w_vc = its transpose
        self.w_uv = [
            w.view(heads, QK_NOPE + V_HEAD, D_V)[:, QK_NOPE:, :].contiguous()
            for w in self.kv_b
        ]
        self.w_o = [
            (torch.randn(HIDDEN, heads * V_HEAD, generator=g, device="cuda") * 0.02).to(
                torch.bfloat16
            )
            for _ in range(layers)
        ]
        self.packed = None
        if heads == 8:
            from sgl_nvidia_mega.glm5_3.sm103.packing import pack_uv_proj

            self.packed = [pack_uv_proj(w) for w in self.kv_b]

    def chain(self, attention, v_b, *, bare=False):
        """The graph body: ``attention(layer, q) -> o`` and ``v_b(layer, o) -> vb``
        (``v_b`` None: ``attention`` returns the o_proj input)."""
        from sglang.kernels.ops.attention.utils import mla_quantize_and_rope_for_fp8
        from sglang.kernels.ops.kvcache.set_mla_kv_buffer import set_mla_kv_buffer

        q, k, k_rope = mla_quantize_and_rope_for_fp8(
            self.q_nope,
            self.q_pe,
            self.k_nope,
            self.k_pe,
            self.positions,
            self.cos_sin,
            False,
            D_V,
            D_QK - D_V,
        )
        for layer in range(self.layers):
            if bare:
                attention(layer, q)
                continue
            q, k, k_rope = mla_quantize_and_rope_for_fp8(
                self.q_nope,
                self.q_pe,
                self.k_nope,
                self.k_pe,
                self.positions,
                self.cos_sin,
                False,
                D_V,
                D_QK - D_V,
            )
            set_mla_kv_buffer(self.pool, self.locs[layer], k, k_rope)
            o = attention(layer, q)
            vb = o if v_b is None else v_b(layer, o)
            F.linear(vb, self.w_o[layer])


def time_graph(fn, flush, replays) -> float:
    fn()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        fn()
    for _ in range(3):
        graph.replay()
    torch.cuda.synchronize()
    total = 0.0
    for _ in range(replays):
        flush.zero_()
        start, end = (
            torch.cuda.Event(enable_timing=True),
            torch.cuda.Event(enable_timing=True),
        )
        start.record()
        graph.replay()
        end.record()
        end.synchronize()
        total += start.elapsed_time(end)
    del graph
    return total * 1000.0 / replays


def bmm(point):
    def v_b(layer, o):
        rows, heads = o.shape[0], o.shape[1]
        out = torch.empty(rows, heads * V_HEAD, dtype=torch.bfloat16, device="cuda")
        torch.bmm(
            o.transpose(0, 1),
            point.w_uv[layer].transpose(1, 2),
            out=out.view(rows, heads, V_HEAD).transpose(0, 1),
        )
        return out

    return v_b


def variants(point, trt, pr19):
    heads, rows = point.heads, point.rows
    pool = point.pool
    # what the plan launches: one-tile (spread) in its wave, the multi-tile mode above it
    # (8 heads, up to MT_MAX_ROWS), else the one-tile kernel packed in several waves
    plan = fd.make_plan(heads, fd.device_caps(heads, with_mt=True, with_fold=False))
    splits = plan.splits(rows) or fd.ONE_TILE_SPLITS
    smem = plan.smem_bytes(rows) if splits == fd.ONE_TILE_SPLITS else None

    def stock(layer, q):
        return trt(q, pool, point.idx[layer], point.lens[layer])

    def ours(layer, q):
        return fd.flash_decode(
            q,
            pool,
            point.idx[layer],
            point.lens[layer],
            SM_SCALE * LOG2E,
            smem_bytes=smem,
            splits=splits,
        )

    out = {"stock": (stock, bmm(point)), "ours": (ours, bmm(point))}
    if heads == 8 and rows <= 16:

        def ours_fold(layer, q):
            return fd.flash_decode(
                q,
                pool,
                point.idx[layer],
                point.lens[layer],
                SM_SCALE * LOG2E,
                vb_proj=point.w_uv[layer],
            )

        def pr19_attn(layer, q):
            o = torch.empty(rows, heads, D_V, dtype=torch.bfloat16, device="cuda")
            return pr19.attention(q, pool, point.idx[layer], SM_SCALE * LOG2E, o)

        def pr19_uv(layer, o):
            out_ = torch.empty(
                rows, heads * V_HEAD, dtype=torch.bfloat16, device="cuda"
            )
            return pr19.uv_proj(point.packed[layer], o, out_)

        if rows <= plan.fold_rows:
            out["ours_fold"] = (ours_fold, None)
        if pr19 is not None:
            out["pr19"] = (pr19_attn, pr19_uv)
            out["pr19+bmm"] = (pr19_attn, bmm(point))
            out["ours+pr19uv"] = (ours, pr19_uv)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.environ.get("TEST_OUT", "."))
    ap.add_argument("--rows8", default="1,2,4,6")
    ap.add_argument("--rows16", default="1,2,5,6,7")
    ap.add_argument("--ctx", default="1k,8k,64k")
    ap.add_argument("--layers", type=int, default=78)
    ap.add_argument("--replays", type=int, default=20)
    ap.add_argument("--no-pr19", action="store_true")
    args = ap.parse_args()
    torch.cuda.set_device(0)
    g = torch.Generator(device="cuda").manual_seed(2026)
    flush = torch.empty(512 << 20, dtype=torch.uint8, device="cuda")
    pr19 = None if args.no_pr19 else Pr19()
    rows_by_heads = {
        8: [int(r) for r in args.rows8.split(",") if r],
        16: [int(r) for r in args.rows16.split(",") if r],
    }
    results = []
    for heads, rows_list in rows_by_heads.items():
        trt = TrtllmGen(heads)
        for ctx in args.ctx.split(","):
            for rows in rows_list:
                point = Point(heads, rows, ctx, args.layers, g)
                row = {"heads": heads, "rows": rows, "ctx": ctx}
                for name, (attention, v_b) in variants(point, trt, pr19).items():
                    row[name] = (
                        time_graph(
                            lambda: point.chain(attention, v_b), flush, args.replays
                        )
                        / args.layers
                    )
                    if v_b is not None or name == "ours_fold":
                        row[f"bare_{name}"] = (
                            time_graph(
                                lambda: point.chain(attention, v_b, bare=True),
                                flush,
                                args.replays,
                            )
                            / args.layers
                        )
                results.append(row)
                print(
                    f"{heads}h rows={rows} {ctx}: "
                    + "  ".join(
                        f"{k}={v:.2f}" for k, v in row.items() if isinstance(v, float)
                    ),
                    flush=True,
                )
                point = None  # free its buffers before the next point
                torch.cuda.empty_cache()
    keys = sorted({k for r in results for k in r} - {"heads", "rows", "ctx"})
    path = os.path.join(args.out, "bench_glm5_3_sparse_mla.csv")
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["heads", "rows", "ctx", *keys])
        w.writeheader()
        for r in results:
            w.writerow(
                {k: (f"{v:.3f}" if isinstance(v, float) else v) for k, v in r.items()}
            )
    print(f"wrote {path}")
    # per-layer savings of ours against the reference of each head count
    print("\n## per layer of the chain (us); saving = reference - ours")
    for r in results:
        if r["heads"] == 8:
            nan = float("nan")
            fold = r.get("ours_fold", nan)
            print(
                f"8h rows={r['rows']} {r['ctx']}: stock {r['stock']:.2f} pr19 {r.get('pr19', nan):.2f} "
                f"ours {r['ours']:.2f} ours_fold {fold:.2f}; fold vs pr19 {r.get('pr19', nan) - fold:+.2f}, "
                f"fold vs stock {r['stock'] - fold:+.2f}"
            )
        else:
            print(
                f"16h rows={r['rows']} {r['ctx']}: stock {r['stock']:.2f} ours {r['ours']:.2f}; saving {r['stock'] - r['ours']:+.2f}"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
