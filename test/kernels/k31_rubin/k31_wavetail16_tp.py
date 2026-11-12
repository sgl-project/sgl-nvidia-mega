# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""K3.1 TP16 two-shot MoE tail (``tail.WaveMoeTail`` twoshot through ``tail_runtime.PreparedTail``, plus
``WaveMoeTail.push`` of the shared-expert rows) over W = 4 GPUs with real symm_mem multicast: the served Breq 5-8
(33-64 rows) tail and, with ``--cap 128`` (CAP-128 #4), the Breq 16 (65-128 rows) tail over a 128-row box.

GPU g plays TP rank g of 16. Inputs are synthetic and seeded identically on every GPU: per rank q the routed latent
partial L_q [B, 3584] (the experts' rows) and the shared-expert output S_q [B, 7168].
- latent two-shot: rank g pushes its 8-column chunks of L_g to every owner's receive slot (parity x 16 + g) with a
  stand-in built on ``common.twoshot.push`` (the experts' push; recv layout [3 x 16, cap, 224] from the box); owners
  4-15 get local stand-in receive buffers (never read). Owner g's slots 4-15 are injected with L_4..L_15's chunks;
  owners 4-15's gather columns are injected into every GPU's gather with the stock reduce emulation (FP32 rank-order
  sum of the 16 BF16 partials, one BF16 rounding, -0 -> +0, as ``common.twoshot.reduce``).
- RS: rank g's push peer-stores S_g's 448-column shards to owners 0-3 (real) and 4-15 (local scratch); shards of
  ranks 4-15 for owner g are injected. A stand-in ring consumer per token polls, saves and re-arms the output columns
  of ranks 0-3 (0-1791).

Checks (tolerances stated before any run; the TP8 test's bounds):
  gather   owners 0-3's reduced chunks in this GPU's gather == the FP32 rank-order emulation within 1 BF16 ulp (the
           count of non-bitwise cells is recorded);
  RS       received shards from ranks 0-3 == their S columns (bitwise, -0 -> +0), and this rank's shards for 4-15;
  output   columns 0-1791 vs FP64 of the stock method (rank-order BF16 all-reduce, RMS of the sums, BF16 normalized
           latent, up projection, BF16, + BF16 shared sum, BF16): within 2 BF16 ulp of max(|out|, |up|, |shared|);
           bit-identical on every GPU; three layers (parities 0-2) eager, then replayed from a CUDA graph (bitwise).
Mutations (each must fail): cap4_scalar64 (the tail's run-time capacity left at 64 under a 128-row box; run at
--cap 128), colsum_drop_rank (one source rank dropped from the reducing CTAs' column sums).

Usage: python3 -m torch.distributed.run --nproc-per-node 4 k31_wavetail16_tp.py --out DIR [--batches 33,64] [--cap 128]
"""

import argparse
import json
import os
import sys
import time
import traceback
from types import SimpleNamespace

import torch
import torch.distributed as dist

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _bench_moe as B  # noqa: E402
import _harness as H  # noqa: E402
from k31_moe_tp import EPS, HID, LAT, SENT, bf, canon_bf16, gen, randn, ulp_bf16  # noqa: E402

TP, SHARD, OWN = 16, 7168 // 16, 3584 // 16
TL = "kimi_k31/k3/cutedsl/moe/tail.py"
MUTATIONS = {
    # WaveMoeTail's reducing CTAs sum in ``column_sums`` (``sum_sources``, the k31_wavetail_tp.py anchor, is the one-group path)
    "colsum_drop_rank": (TL, "        for source in cutlass.range_constexpr(self.tp):\n            for k in cutlass.range_constexpr(8):\n                acc[k] = fma_bf16(values[source, k // 2], one, acc[k], high=k % 2 == 1)",
                         "        for source in cutlass.range_constexpr(self.tp - 1):\n            for k in cutlass.range_constexpr(8):\n                acc[k] = fma_bf16(values[source, k // 2], one, acc[k], high=k % 2 == 1)"),
    "cap4_scalar64": ("kimi_k31/k3/cutedsl/moe/tail_runtime.py", "self.scalars = (Int32(cap), Float32(eps), rearm)",
                      "self.scalars = (Int32(CAP), Float32(eps), rearm)"),
}
NLAYERS = 6  # three rotating parities; a multiple of 3 (see k31_wavetail_tp.py)
LAYERS = tuple((1 + i, i) for i in range(3))
_PUSH = {}


def compile_push16(batch, cap, device):
    """Stand-in for the experts' two-shot latent push: every 8-column chunk of x [batch, 3584] into its owner's
    receive slot ``slot`` through ``common.twoshot.push`` (recv layout [3 x 16, cap, 224])."""
    key = (batch, cap, device)
    if key in _PUSH:
        return _PUSH[key]
    import cutlass
    import cutlass.cute as cute
    from cutlass import Int32 as I32, Int64 as I64
    from cutlass.cute.runtime import make_fake_stream

    from sgl_nvidia_mega.kimi_k31.k3.cutedsl.common import primitives as P
    from sgl_nvidia_mega.kimi_k31.k3.cutedsl.common import twoshot as TS
    from sgl_nvidia_mega.kimi_k31.k3.cutedsl.common.host import fake

    class Push16:
        def __init__(self):
            self.batch = batch

        @cute.jit
        def __call__(self, *args):
            P.launch(self.kernel, args, self.batch, 112, pdl=False)

        @cute.kernel
        def kernel(self, args):
            x, recv, peers, slot = args
            tid, _, _, token = P.thread()
            for i in cutlass.range_constexpr(LAT // 896):
                column = 8 * (tid + 112 * i)
                words = P.load_words(x.iterator + token * LAT + column, count=4)
                TS.push(recv, peers, slot, token, column, tuple(P.no_neg_zero(v) for v in words))

    args = (fake((batch, LAT)), fake((3 * TP, cap, OWN)), fake((TP,), None, I64), I32(0))
    with torch.accelerator.device_index(device):
        code = cute.compile(Push16(), *args, make_fake_stream())
    _PUSH[key] = code
    return code


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--batches", default="33,64")
    ap.add_argument("--cap", type=int, default=0, help="tail box rows (0 = tail_runtime.CAP)")
    ap.add_argument("--deadline-s", type=float, default=2.0)
    args = ap.parse_args()
    local = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local)
    device = torch.device("cuda", local)
    dist.init_process_group("nccl", device_id=device)
    g, world = dist.get_rank(), dist.get_world_size()
    tag = os.environ.get("K31_MUTATION") or "base"
    os.makedirs(args.out, exist_ok=True)
    os.environ["K31_RESULTS"] = os.path.join(args.out, f"wave16_tp{world}_{tag}_g{g}.jsonl")
    H.load_plugin(MUTATIONS)
    from cutlass import Int32, Int64

    from sgl_nvidia_mega.kimi_k31.k3.cutedsl.common import host
    from sgl_nvidia_mega.kimi_k31.k3.cutedsl.moe import tail_runtime as TR

    group = dist.group.WORLD
    fails = []
    t0 = time.time()
    deadline = int(args.deadline_s * 1e9)
    n_out = world * SHARD  # output columns of the real ranks

    def gather_all(t):
        out = [torch.empty_like(t) for _ in range(world)]
        dist.all_gather(out, t.contiguous(), group=group)
        return out

    try:
        tb = TR.MoeTailBox(group, device, TP, g, cap=args.cap or TR.CAP)
        cap = tb.cap
        ts = tb.twoshot
        # 16-entry push table: owners 0-3 = the real receive buffers (TwoShot's 4-entry table over this group),
        # owners 4-15 = local stand-ins (written, never read).
        standins = {o: torch.empty_like(ts.recv).view(torch.int32).fill_(SENT).view(torch.bfloat16) for o in range(world, TP)}
        size = ts.recv[0].numel() * 2
        bases = [[int(ts.peers[p, o]) if o < world else standins[o].data_ptr() + p * size for o in range(TP)] for p in range(3)]
        peers16 = torch.tensor(bases, dtype=torch.int64, device=device)
        tb.push_box = (host.to_cute(ts.recv.view(3 * TP, cap, -1)), host.to_cute(peers16[0], 16))
        # RS peers: real 0-3 + local scratch 4-15 (as k31_wavetail_tp.py)
        scratch = [torch.empty_like(tb.rs.buffer).view(torch.int32).fill_(SENT).view(torch.bfloat16) for _ in range(TP - world)]
        rs_peers = torch.tensor(list(tb.rs.handle.buffer_ptrs) + [x.data_ptr() for x in scratch], dtype=torch.int64, device=device)
        tb.peers = rs_peers
        tb.tail = (host.to_cute(rs_peers, 16), TR.flat(tb.out.buffer), Int64(tb.latent.mc), Int64(tb.out.mc))
        tb.twoshot_tail = (*tb.tail[:2], Int64(ts.gather_mc), tb.tail[3])
        torch.cuda.synchronize()
        dist.barrier(group=group)
        H.record({"kind": "info", "item": f"tail_box16.g{g}", "cap": cap, "recv": list(ts.recv.shape), "gather": list(ts.gather.shape),
                  "rs": list(tb.rs.buffer.shape), "out": list(tb.out.buffer.shape)})
        op = SimpleNamespace(rank=g, contract=SimpleNamespace(tp_size=TP, rms_eps=EPS))
        consumer = B.ring_consumer(n_out // 2, device.index)
        s = lambda: host.stream(device)  # noqa: E731
        for batch in map(int, args.batches.split(",")):
            layers = {}
            specs = [(1 + i, i) for i in range(NLAYERS)]
            for layer, ordinal in specs:
                up = randn((HID, LAT), LAT ** -0.5, 7000 + layer, device)
                gamma = (1.0 + 0.1 * torch.randn(LAT, generator=gen(7100 + layer, device), device=device)).to(torch.bfloat16)
                tail = TR.PreparedTail(op, gamma, up, (batch,), (tb, ordinal, len(specs)), deadline_ns=deadline)
                if batch not in tail.twoshots:
                    raise RuntimeError(f"batch {batch} is not a TP16 two-shot batch ({sorted(tail.twoshots)})")
                layers[layer] = SimpleNamespace(tail=tail, up=up, gamma=gamma, t=int(tail.tail_weights.parity))
            dummy = (torch.zeros((batch * 16, LAT), dtype=torch.bfloat16, device=device),
                     torch.zeros((batch, 16), dtype=torch.bfloat16, device=device))
            push = compile_push16(batch, cap, device.index)
            staged = {}

            def inputs(call):
                L = [canon_bf16(randn((batch, LAT), 0.5, 8000 + 100 * call + q, device)) for q in range(TP)]
                S = [randn((batch, HID), 0.5, 9000 + 100 * call + q, device) for q in range(TP)]
                total = L[0].float()
                for q in range(1, TP):
                    total = total + L[q].float()  # FP32, rank order
                red = canon_bf16(total.to(torch.bfloat16))  # one rounding (common.twoshot.reduce)
                staged[id(L)] = (
                    torch.stack([L[q][:, g * OWN:(g + 1) * OWN] for q in range(world, TP)]),  # owner g's slots 4-15
                    red[:, world * OWN:].contiguous(),  # owners 4-15's gather columns
                    torch.stack([canon_bf16(S[q][:, g * SHARD:(g + 1) * SHARD]) for q in range(world, TP)]),
                    red,
                )
                return L, S

            def stage_and_run(lay, L, S, saves):
                t = lay.t
                recv_src, gather_src, rs_src, _ = staged[id(L)]
                ts.recv[t, world:TP, :batch].copy_(recv_src)
                ts.gather[t, 0, :batch, world * OWN:].copy_(gather_src)
                push(host.to_cute(L[g]), tb.push_box[0], tb.push_box[1], Int32(t * TP + g), s())
                tb.rs.buffer[t, world:TP, :batch].copy_(rs_src)
                lay.tail.push(S[g])
                lay.tail.run(dummy[0], dummy[1], S[g], tb.out.buffer[0, :batch], consume=lambda o: o)
                for tok in range(batch):
                    ring = tb.out.buffer[0, tok].view(-1)[:n_out].view(torch.int32)
                    consumer(host.to_cute(ring, 16), host.to_cute(saves[tok], 16), s())

            def check(lay, L, S, saves, name, gather_check=True):
                t = lay.t
                red = staged[id(L)][3]
                if gather_check:
                    got = ts.gather[t, 0, :batch, :world * OWN].double()
                    want = red[:, :world * OWN].double()
                    d = (got - want).abs()
                    nbit = int((got != want).sum())
                    over = int((d > ulp_bf16(want.abs())).sum())
                    H.record({"kind": "check", "item": f"{name}.gather", "non_bitwise_cells": nbit, "over_1ulp": over})
                    if over:
                        fails.append(f"{name}.gather: {over} cells beyond 1 ulp of the rank-order emulation ({nbit} non-bitwise)")
                for q in range(world):
                    got = tb.rs.buffer[t, q, :batch]
                    want = canon_bf16(S[q][:, g * SHARD:(g + 1) * SHARD])
                    if not torch.equal(got.view(torch.int16), want.view(torch.int16)):
                        fails.append(f"{name}.rs: shard from rank {q} differs")
                for o in range(world, TP):
                    got = scratch[o - world][t, g, :batch]
                    want = canon_bf16(S[g][:, o * SHARD:(o + 1) * SHARD])
                    if not torch.equal(got.view(torch.int16), want.view(torch.int16)):
                        fails.append(f"{name}.rs_out: shard for owner {o} differs")
                worst = 0.0
                for tok in range(batch):
                    Lsum = sum(L[q][tok].double() for q in range(TP))
                    inv = 1.0 / torch.sqrt((Lsum ** 2).mean() + EPS)
                    xn = bf(bf(Lsum) * inv * lay.gamma.double())
                    y = bf(lay.up.double() @ xn)
                    sh = bf(sum(S[q][tok].double() for q in range(TP)))
                    ref = bf(y + sh)[:n_out]
                    out = saves[tok].view(torch.bfloat16).double()
                    lim = 2 * ulp_bf16(torch.maximum(torch.maximum(ref.abs(), y[:n_out].abs()), sh[:n_out].abs()))
                    err = (out - ref).abs()
                    nbad = int((err > lim).sum())
                    worst = max(worst, float((err / ulp_bf16(ref.abs())).max()))
                    if nbad or not torch.isfinite(out).all():
                        fails.append(f"{name}.tok{tok}.output: {nbad} cells beyond 2 ulp, max {float(err.max()):.3e}")
                H.digest(f"{name}.out", torch.stack(saves))  # cross-tree bitwise comparisons (same seeds, same inputs)
                allsaves = gather_all(torch.stack(saves))
                if not all(torch.equal(a, allsaves[g]) for a in allsaves):
                    fails.append(f"{name}.rank_bitwise: outputs differ across GPUs")
                row = {"kind": "check", "item": name, "max_ulps": worst, "fails_so_far": len(fails)}
                H.record(row)
                print("CHECK", json.dumps(row), flush=True)

            eager = []
            for call in range(len(LAYERS)):
                lay = layers[LAYERS[call][0]]
                L, S = inputs(call)
                saves = [torch.full((n_out // 2,), SENT, dtype=torch.int32, device=device) for _ in range(batch)]
                torch.cuda.synchronize()
                dist.barrier(group=group)
                stage_and_run(lay, L, S, saves)
                torch.cuda.synchronize()
                dist.barrier(group=group)
                check(lay, L, S, saves, f"wave16_tp{world}.cap{cap}.b{batch}.l{LAYERS[call][0]}.eager{call}.g{g}")
                eager.append((L, S, [x.clone() for x in saves]))
                torch.cuda.synchronize()
                dist.barrier(group=group)
            gsaves = [[torch.empty(n_out // 2, dtype=torch.int32, device=device) for _ in range(batch)] for _ in LAYERS]

            def both():
                for call in range(len(LAYERS)):
                    L, S, _ = eager[call]
                    stage_and_run(layers[LAYERS[call][0]], L, S, gsaves[call])

            graph, stream = torch.cuda.CUDAGraph(), torch.cuda.Stream()
            stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.graph(graph, stream=stream):
                both()
            torch.cuda.synchronize()
            for rep in range(2):
                for sv in gsaves:
                    for x in sv:
                        x.fill_(SENT)
                torch.cuda.synchronize()
                dist.barrier(group=group)
                graph.replay()
                torch.cuda.synchronize()
                dist.barrier(group=group)
                for call in range(len(LAYERS)):
                    same = all(torch.equal(a, b) for a, b in zip(gsaves[call], eager[call][2]))
                    H.record({"kind": "check", "item": f"wave16.cap{cap}.b{batch}.graph{rep}.call{call}.g{g}", "bitwise_vs_eager": same})
                    if not same:
                        fails.append(f"wave16.cap{cap}.b{batch}.graph{rep}.call{call}: differs from eager")
            del graph
            torch.cuda.synchronize()
            dist.barrier(group=group)
    except Exception as e:  # noqa: BLE001
        fails.append(f"g{g}: {type(e).__name__}: {e}")
        traceback.print_exc()
    for f in fails[:20]:
        print("FAIL", f, flush=True)
    with open(os.path.join(args.out, f"wave16_tp{world}_{tag}_g{g}.failures.json"), "w") as f:
        json.dump(fails, f, indent=1)
    n = torch.tensor([len(fails)], device=device)
    try:
        dist.all_reduce(n)
    except Exception:  # noqa: BLE001
        pass
    if g == 0:
        print(f"TP_RESULT failures={int(n)} ({time.time() - t0:.0f}s)", flush=True)
    dist.destroy_process_group()
    sys.exit(1 if int(n) else 0)


if __name__ == "__main__":
    main()
