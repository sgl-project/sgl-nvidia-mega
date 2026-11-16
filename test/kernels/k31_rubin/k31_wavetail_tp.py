# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""K3.1 TP8 published MoE tail at B5-B8 (``tail.WaveMoeTail`` through
``tail_runtime.PreparedTail``, plus ``WaveMoeTail.push`` of the shared-expert rows) over
W = 4 GPUs with real symm_mem multicast; the Q8 verify / B5-B8 decode tail.

GPU g runs TP rank g. Inputs are synthetic and seeded identically on every GPU: per
rank q the routed latent partial L_q [B, 3584] (the experts' published rows) and the
shared-expert output S_q [B, 7168]. rank g multicasts its own latent rows into every GPU's mailbox (a stand-in for the experts' publish; it also keeps
the ranks in lock step as the real chain's hops do: with locally injected rows a fast rank's next output could land
in a slow rank's ring before that rank consumed it); rows of ranks 4-7 are written locally;
rank g's push peer-stores S_g's shards to owners 0-3 (real) and 4-7 (local scratch);
shards of ranks 4-7 for owner g are injected. A stand-in ring consumer per token polls,
saves and re-arms the output columns of ranks 0-3.

Checks (tolerances stated before any run):
  RS       received shards from ranks 0-3 == their S columns (bitwise, -0 -> +0), and
           this rank's shards for owners 4-7 in scratch;
  output   rows of ranks 0-3 vs FP64 of the stock method (rank-order BF16 all-reduce,
           RMS of the FP32 sums, BF16 normalized latent, up projection, BF16, + BF16
           shared sum, BF16): within 2 BF16 ulp of max(|out|, |up|, |shared|); bit-identical
           on every GPU; two layers alternating, eager then replayed from a CUDA graph
           (bit-identical to eager).
Mutations: wave_drop_rank (one source rank dropped from the latent sum), wave_rms (RMS
of twice the sum of squares), push_shard (push writes to the wrong owner).

E-TAIL2HOP at B5-B8 (``--arms wave,rep``): ``rep`` = ``tail_rep.SharedPushFull`` (this rank's
whole shared rows multicast into the [latent | shared] all-reduce mailbox) +
``MoeTailRep(publish=True)`` (replicated up projection, local ring stores, no output
hop). Same reference over all 7168 columns, bit-identical on every GPU; max ulp distance
to the wave arm recorded. Timing: per-layer time of a graph of 16 alternating layers
(injection copies, push, tail and ring consumers; identical injection in both arms),
ABBA, max over ranks; pass >= 1.5 us per layer saved, kill < 0.5 us.

CAP-128 #4 (``--cap 128``): the tail box takes 128 rows (a Breq 16 boot's capacity) and every batch, 1-128, runs over
it (the tail's run-time capacity, the push's RS stride and the stand-in publish all take ``tb.cap``); same checks.
Mutations at --cap 128: cap4_scalar64 (the tail kernels' run-time capacity left at 64), cap4_push64 (the shared-row
push's RS stride left at 64).

Usage: python3 -m torch.distributed.run --nproc-per-node 4 k31_wavetail_tp.py --out DIR [--batches 5,8] [--arms wave,rep] [--cap 128]
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
from k31_moe_tp import EPS, HID, LAT, SENT, SHARD, TP, bf, canon_bf16, gen, randn, ulp_bf16  # noqa: E402

TL = "kimi_k31/k3/cutedsl/moe/tail.py"
MUTATIONS = {
    "wave_drop_rank": (TL, "        for source in cutlass.range_constexpr(self.tp):\n            for k in cutlass.range_constexpr(8):\n                acc[k] = acc[k] + bf16(values[source + start, k // 2], k % 2)",
                       "        for source in cutlass.range_constexpr(self.tp - 1):\n            for k in cutlass.range_constexpr(8):\n                acc[k] = acc[k] + bf16(values[source + start, k // 2], k % 2)"),
    "wave_rms": (TL, "self.scale(words, gx, inv_rms(total, eps, m8=True), xs, ids)",
                 "self.scale(words, gx, inv_rms(total + total, eps, m8=True), xs, ids)"),
    "push_shard": (TL, "store_relaxed(peers[column // self.shard] + Int64(offset) * 2, packed)",
                   "store_relaxed(peers[(column // self.shard + 1) % self.tp] + Int64(offset) * 2, packed)"),
    "cap4_scalar64": ("kimi_k31/k3/cutedsl/moe/tail_runtime.py", "self.scalars = (Int32(cap), Float32(eps), rearm)",
                      "self.scalars = (Int32(CAP), Float32(eps), rearm)"),
    "cap4_push64": ("kimi_k31/k3/cutedsl/moe/tail_runtime.py", "args = (to_cute(shared), self.peers, *self.ring, Int32(self.tail_box.cap))",
                    "args = (to_cute(shared), self.peers, *self.ring, Int32(CAP))"),
}
# Six MoE layers (ordinals 0-5, three rotating tail parities). A tail re-arms the previous layer's parity at its
# end; with a layer count that is a multiple of 3 that parity is never the next layer's, so a fast rank's next
# publish cannot land before this rank's re-arm (with 2 or 8 layers the wrap from the last layer to the first makes
# them equal: the decode loop is ordered there by its o-proj hop, which this harness does not have).
NLAYERS = 6
LAYERS = tuple((1 + i, i) for i in range(3))  # the checked layers, run in sequence


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--batches", default="5,8")
    ap.add_argument("--arms", default="wave")
    ap.add_argument("--deadline-s", type=float, default=2.0)
    ap.add_argument("--cap", type=int, default=0, help="tail box rows (0 = tail_runtime.CAP)")
    args = ap.parse_args()
    arms = tuple(args.arms.split(","))
    local = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local)
    device = torch.device("cuda", local)
    dist.init_process_group("nccl", device_id=device)
    g, world = dist.get_rank(), dist.get_world_size()
    tag = os.environ.get("K31_MUTATION") or "base"
    os.makedirs(args.out, exist_ok=True)
    os.environ["K31_RESULTS"] = os.path.join(args.out, f"wave_tp{world}_{tag}_g{g}.jsonl")
    H.load_plugin(MUTATIONS)
    from cutlass import Float32, Int32, Int64

    from sgl_nvidia_mega.kimi_k31.k3.cutedsl.common import host
    # The opt-in replicated-up tail (E-TAIL2HOP, killed) is not part of the integrated build.
    TRP = None
    if "rep" in arms:
        from sgl_nvidia_mega.kimi_k31.k3.cutedsl.moe import tail_rep as TRP
    from sgl_nvidia_mega.kimi_k31.k3.cutedsl.moe import tail_runtime as TR
    from sgl_nvidia_mega.kimi_k31.k3.cutedsl.oproj import prepared as OP

    group = dist.group.WORLD
    fails = []
    t0 = time.time()
    deadline = int(args.deadline_s * 1e9)
    timed = os.environ.get("K31_TIME") == "1" and tag == "base"

    def gather(t):
        out = [torch.empty_like(t) for _ in range(world)]
        dist.all_gather(out, t.contiguous(), group=group)
        return out

    try:
        tb = TR.MoeTailBox(group, device, TP, g, cap=args.cap or TR.CAP)
        H.record({"kind": "info", "item": f"tail_box.g{g}", "cap": tb.cap, "latent": list(tb.latent.buffer.shape),
                  "rs": list(tb.rs.buffer.shape), "out": list(tb.out.buffer.shape), "recv": list(tb.twoshot.recv.shape)})
        sar = OP.OprojMailbox(group, device, TP, g, (3, TP, TR.CAP, HID)) if "rep" in arms else None
        scratch = [torch.empty_like(tb.rs.buffer).view(torch.int32).fill_(SENT).view(torch.bfloat16)
                   for _ in range(TP - world)]
        peers = torch.tensor(list(tb.rs.handle.buffer_ptrs) + [x.data_ptr() for x in scratch],
                             dtype=torch.int64, device=device)
        tb.peers = peers
        tb.tail = (host.to_cute(peers, 16), TR.flat(tb.out.buffer), Int64(tb.latent.mc), Int64(tb.out.mc))
        op = SimpleNamespace(rank=g, contract=SimpleNamespace(tp_size=TP, rms_eps=EPS))
        consumers = {"wave": B.ring_consumer(LAT // 2, device.index), "rep": B.ring_consumer(HID // 2, device.index)}
        s = lambda: host.stream(device)  # noqa: E731
        c = lambda t: host.to_cute(t.reshape(-1), 16)  # noqa: E731
        sms = torch.cuda.get_device_properties(device).multi_processor_count
        gws = [torch.zeros(n, dtype=torch.bfloat16, device=device) for n in (24 * 16 * 3584, 24 * 16, 24 * 7168)]
        for batch in map(int, args.batches.split(",")):
            layers = {}
            specs = [(1 + i, i) for i in range(NLAYERS)]
            for layer, ordinal in specs:
                up = randn((HID, LAT), LAT ** -0.5, 7000 + layer, device)
                gamma = (1.0 + 0.1 * torch.randn(LAT, generator=gen(7100 + layer, device), device=device)).to(torch.bfloat16)
                tail = TR.PreparedTail(op, gamma, up, (batch,), (tb, ordinal, len(specs)), deadline_ns=deadline)
                layers[layer] = SimpleNamespace(tail=tail, up=up, gamma=gamma, t=int(tail.tail_weights.parity),
                                                rearm=tail.tail_weights.scalars[2])
            dummy = (torch.zeros((batch * 16, LAT), dtype=torch.bfloat16, device=device),
                     torch.zeros((batch, 16), dtype=torch.bfloat16, device=device))
            rep_code = push_full = None
            lat_push = B.compile_push_full(batch, device.index, capacity=tb.cap, width=LAT)  # stand-in (experts' latent publish)
            if "rep" in arms:
                rep_code = TRP.compile_tail_rep(batch, sms, device.index, deadline, publish=True)
                push_full = TRP.compile_push_full(batch, device.index)

            H.maybe_preload(f"wave.b{batch}")
            staged = {}

            def inputs(call):
                L = [randn((batch, LAT), 0.5, 8000 + 100 * call + q, device) for q in range(TP)]
                S = [randn((batch, HID), 0.5, 9000 + 100 * call + q, device) for q in range(TP)]
                # Injection sources laid out once, outside any graph (the graph holds copies only).
                staged[id(L)] = (
                    torch.stack([canon_bf16(L[q]) for q in range(world, TP)]),
                    torch.stack([canon_bf16(S[q][:, g * SHARD:(g + 1) * SHARD]) for q in range(world, TP)]),
                    torch.stack([canon_bf16(S[q]) for q in range(world, TP)]),
                )
                return L, S

            def stage_and_run(lay, L, S, saves, arm):
                t = lay.t
                lat_src, rs_src, sar_src = staged[id(L)]
                tb.latent.buffer[t, world:TP, :batch].copy_(lat_src)
                lat_push(host.to_cute(L[g]), Int64(tb.latent.mc), Int32(t), Int32(g), Int32(tb.cap), s())
                if arm == "wave":
                    tb.rs.buffer[t, world:TP, :batch].copy_(rs_src)
                    lay.tail.push(S[g])
                    lay.tail.run(dummy[0], dummy[1], S[g], tb.out.buffer[0, :batch], consume=lambda o: o)
                    n = LAT
                else:
                    sar.buffer[t, world:TP, :batch].copy_(sar_src)
                    push_full(host.to_cute(S[g]), Int64(sar.mc), Int32(t), Int32(g), Int32(TR.CAP), s())
                    rep_code(c(tb.latent.buffer), c(sar.buffer), c(lay.up), c(lay.gamma), c(gws[0]), c(gws[1]),
                             c(gws[2]), c(tb.out.buffer), Int64(tb.latent.mc), Int64(sar.mc), Int32(t), Int32(g),
                             Int32(TR.CAP), Float32(EPS), lay.rearm, s())
                    n = HID
                for tok in range(batch):
                    ring = tb.out.buffer[0, tok].view(-1)[:n].view(torch.int32)
                    consumers[arm](host.to_cute(ring, 16), host.to_cute(saves[tok], 16), s())

            def check(lay, L, S, saves, name, arm, ref_saves=None):
                t = lay.t
                if arm == "wave":
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
                    n = LAT
                else:
                    for q in range(world):
                        if not torch.equal(sar.buffer[t, q, :batch].view(torch.int16), canon_bf16(S[q]).view(torch.int16)):
                            fails.append(f"{name}.shared_ar: slot {q} differs")
                    n = HID
                worst, vs = 0.0, 0.0
                for tok in range(batch):
                    Lsum = sum(canon_bf16(L[q][tok]).double() for q in range(TP))
                    inv = 1.0 / torch.sqrt((Lsum ** 2).mean() + EPS)
                    xn = bf(bf(Lsum) * inv * lay.gamma.double())
                    y = bf(lay.up.double() @ xn)
                    sh = bf(sum(S[q][tok].double() for q in range(TP)))
                    ref = bf(y + sh)[:n]
                    out = saves[tok].view(torch.bfloat16).double()
                    lim = 2 * ulp_bf16(torch.maximum(torch.maximum(ref.abs(), y[:n].abs()), sh[:n].abs()))
                    err = (out - ref).abs()
                    nbad = int((err > lim).sum())
                    worst = max(worst, float((err / ulp_bf16(ref.abs())).max()))
                    if ref_saves is not None:
                        d = (out[:LAT] - ref_saves[tok].view(torch.bfloat16).double()).abs()
                        vs = max(vs, float((d / ulp_bf16(ref[:LAT].abs())).max()))
                    if nbad or not torch.isfinite(out).all():
                        fails.append(f"{name}.tok{tok}.output: {nbad} cells beyond 2 ulp, max {float(err.max()):.3e}")
                H.digest(f"{name}.out", torch.stack(saves))
                H.digest(f"{name}.rs", tb.rs.buffer[t] if arm == "wave" else sar.buffer[t])
                allsaves = gather(torch.stack(saves))
                if not all(torch.equal(a, allsaves[g]) for a in allsaves):
                    fails.append(f"{name}.rank_bitwise: outputs differ across GPUs")
                row = {"kind": "check", "item": name, "max_ulps": worst, "fails_so_far": len(fails)}
                if ref_saves is not None:
                    row["max_ulps_vs_wave"] = vs
                H.record(row)
                print("CHECK", json.dumps(row), flush=True)

            wave_eager = None
            for arm in arms:
                n = LAT if arm == "wave" else HID
                eager = []
                for call in range(len(LAYERS)):
                    lay = layers[LAYERS[call][0]]
                    L, S = inputs(call)
                    saves = [torch.full((n // 2,), SENT, dtype=torch.int32, device=device) for _ in range(batch)]
                    torch.cuda.synchronize()
                    dist.barrier(group=group)
                    stage_and_run(lay, L, S, saves, arm)
                    torch.cuda.synchronize()
                    dist.barrier(group=group)
                    refs = None if wave_eager is None or arm == "wave" else wave_eager[call][2]
                    check(lay, L, S, saves, f"{arm}_tp{world}.b{batch}.l{LAYERS[call][0]}.eager{call}.g{g}", arm, refs)
                    eager.append((L, S, [x.clone() for x in saves]))
                    torch.cuda.synchronize()
                    dist.barrier(group=group)
                if arm == "wave":
                    wave_eager = eager
                gsaves = [[torch.empty(n // 2, dtype=torch.int32, device=device) for _ in range(batch)]
                          for _ in LAYERS]

                def both(arm=arm, eager=eager, gsaves=gsaves):
                    for call in range(len(LAYERS)):
                        L, S, _ = eager[call]
                        stage_and_run(layers[LAYERS[call][0]], L, S, gsaves[call], arm)

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
                        H.record({"kind": "check", "item": f"{arm}_tp{world}.b{batch}.graph{rep}.call{call}.g{g}", "bitwise_vs_eager": same})
                        if not same:
                            fails.append(f"{arm}.b{batch}.graph{rep}.call{call}: differs from eager")
                del graph
                torch.cuda.synchronize()
                dist.barrier(group=group)
            if timed:
                H.hold_fp8_clock()
                seq = [layers[layer] for layer, _ in specs]
                data = [inputs(50 + i) for i in range(len(seq))]
                graphs = {}
                for arm in arms:
                    n = LAT if arm == "wave" else HID
                    sv = [[torch.empty(n // 2, dtype=torch.int32, device=device) for _ in range(batch)] for _ in seq]

                    def chain(arm=arm, sv=sv):
                        for i, lay in enumerate(seq):
                            stage_and_run(lay, *data[i], sv[i], arm)

                    for _ in range(2):
                        chain()
                    torch.cuda.synchronize()
                    dist.barrier(group=group)
                    graph, stream = torch.cuda.CUDAGraph(), torch.cuda.Stream()
                    stream.wait_stream(torch.cuda.current_stream())
                    with torch.cuda.graph(graph, stream=stream):
                        chain()
                        chain()
                    torch.cuda.synchronize()
                    graphs[arm] = (graph, chain)
                per = {a: [] for a in arms}
                for arm in list(arms) + list(arms)[::-1]:
                    graph = graphs[arm][0]
                    for _ in range(2):
                        dist.barrier(group=group)
                        graph.replay()
                        torch.cuda.synchronize()
                    times = []
                    for _ in range(10):
                        dist.barrier(group=group)
                        a, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                        a.record()
                        graph.replay()
                        e.record()
                        e.synchronize()
                        times.append(a.elapsed_time(e) * 1e3 / (2 * len(seq)))
                    times.sort()
                    allv = gather(torch.tensor([times[len(times) // 2]], device=device))
                    per[arm].append(max(float(v) for v in allv))
                row = {"kind": "time", "item": f"tail_tp{world}.b{batch}.layer_chain_with_standins",
                       **{f"{a}_us_max_over_ranks": sum(v) / len(v) for a, v in per.items()}, "runs": per}
                if "wave" in per and "rep" in per:
                    row["saved_us_per_layer"] = row["wave_us_max_over_ranks"] - row["rep_us_max_over_ranks"]
                H.record(row)
                if g == 0:
                    print("TIME", json.dumps(row), flush=True)
                from torch.profiler import ProfilerActivity, profile

                for arm in arms:
                    torch.cuda.synchronize()
                    dist.barrier(group=group)
                    with profile(activities=[ProfilerActivity.CUDA]) as prof:
                        graphs[arm][0].replay()
                        torch.cuda.synchronize()
                    durs = {}
                    for evt in prof.events():
                        if evt.device_type.name == "CUDA":
                            durs.setdefault(B.kname(evt.name), []).append(
                                evt.device_time if hasattr(evt, "device_time") else evt.cuda_time)
                    summary = {k: sorted(v)[len(v) // 2] for k, v in durs.items()}
                    H.record({"kind": "time", "item": f"{arm}_tp{world}.b{batch}.kernels.g{g}", "median_us": summary})
                    print("KERNELS", arm, batch, g, json.dumps(summary), flush=True)
                del graphs
                torch.cuda.synchronize()
                dist.barrier(group=group)
        H.preload_audit("wave.end")
    except Exception as e:  # noqa: BLE001
        fails.append(f"g{g}: {type(e).__name__}: {e}")
        traceback.print_exc()
    for f in fails[:20]:
        print("FAIL", f, flush=True)
    with open(os.path.join(args.out, f"wave_tp{world}_{tag}_g{g}.failures.json"), "w") as f:
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
