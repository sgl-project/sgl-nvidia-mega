# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""K3.1 o-projection + TP8 all-reduce + post-attention AttnRes (``oproj.prepared``)
over W GPUs (W in 2, 4, 8) with real symm_mem multicast. Process g plays TP ranks
{g, g + W, ...}: one producer each (all multicast into slot[rank] of every GPU) and the
consumer of TP rank g. With W = 4 the one-shot path runs 8 real producers and no
injected data; W = 8 (one rank per GPU, two trays) is the serving topology. Two-shot
(B64): owners >= W push into local stand-ins and their gather columns come from the
reference reduce (``_ref_comm.TwoShotTP8``).

Each case is a pair of layers with parities 0 and 1, called alternately like the decode
loop: two consecutive calls never reuse a parity, which is what makes the re-arm
race-free across GPUs (the remote producer of call k + 2 needs this GPU's producer of
call k + 1, which runs after this GPU's consumer k). Layer ids fix (nb, has_delta):
0 -> (1, no delta), 5 -> (1, delta), 12 -> (2, no delta), 41 -> (4, delta),
84 -> (8, no delta), 92 -> (8, delta).

Checks per call (reference, tolerances and mutations: ``_ref_comm``): outputs, re-armed
mailbox / gather / receive slots, the two-shot stand-ins' pushed chunks, and at B1 on
MoE layers the x_moe publish (bit-equal to the canonicalized output) and the epoch hint
(+28 per token, what ``MoeFrontTP8`` waits for); calls 0-1 eager, 2-3 replayed from one
CUDA graph with new inputs. Bugs caught: a lost or misindexed peer partial, a dropped
producer K part, a wrong residual add or AttnRes block, no re-arm, a two-shot owner
reduce or push error, state baked into the graph.

Timing (``K31_TIME=1``, first case of each one-shot width): per-layer us of a captured
chain of 32 alternating layers (each = this GPU's producers + its consumer), max over
ranks; and from torch.profiler over eager layers: producer and consumer kernel durations
and consumer end - last local producer end.

Usage (inside the image, one node): python3 -m torch.distributed.run --nproc-per-node 4
k31_oproj_tp.py --out DIR [--batches 1,2,...]
Exit status 0 only when every check on every rank passed.
"""

import argparse
import json
import os
import sys
import time
import traceback

import torch
import torch.distributed as dist

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _harness as H  # noqa: E402
import _ref_comm as C  # noqa: E402

# batch -> list of ((layer A, mla A), (layer B, mla B), front at B1)
CASES = {
    1: [((92, True), (5, False), False), ((0, False), (41, True), True)],
    2: [((84, False), (41, True), False)],
    3: [((12, True), (5, False), False)],
    4: [((92, True), (5, False), False)],
    5: [((84, True), (41, False), False)],
    8: [((92, False), (5, True), False)],
    16: [((92, True), (5, False), False), ((12, False), (41, True), False)],
    32: [((84, False), (41, True), False)],
    64: [((92, True), (5, False), False), ((12, False), (41, True), False)],
    # Breq 16 verify (CAP-128 #3): TP8 stays one-shot above 64 rows, over a 128-row mailbox (oproj.prepared.capacity).
    128: [((92, True), (5, False), False), ((12, False), (41, True), False)],
}


class Case:
    def __init__(self, P, group, g, world, device, batch, spec, deadline_ns):
        (la, ma), (lb, mb), front = spec
        assert la % 2 == 0 and lb % 2 == 1
        self.P, self.group, self.g, self.world, self.device, self.batch = P, group, g, world, device, batch
        self.layers, self.front = (la, lb), front
        self.kinds = "." + "-".join("mla" if m else "kda" for m in (ma, mb))
        self.name = f"oproj_tp{world}.b{batch}.l{la}-{lb}{'.front' if front else ''}"
        cap = P.capacity((batch,))  # as build_resources sizes the mailbox and the two-shot buffers
        self.mailbox = P.OprojMailbox(group, device, C.TP, g, (2, C.TP, cap, C.HIDDEN))
        self.twoshot = None
        if batch in P.twoshot_batches(C.TP):
            self.twoshot = C.TwoShotTP8(group, device, [o if o < world else None for o in range(C.TP)], cap=cap)
        self.xmoe = self.epoch = None
        if front:
            self.xmoe = torch.empty((2, 1, 4, C.HIDDEN), dtype=torch.bfloat16, device=device)
            self.epoch = torch.zeros(256, dtype=torch.int32, device=device)
            self.reset_front()
        ranks = tuple(range(g, C.TP, world))
        self.sims, self.weights = {}, {}
        for layer, mla in ((la, ma), (lb, mb)):
            p = layer % 2
            front_args = None
            if front:
                front_args = (self.xmoe[p], self.epoch[p * 32 : p * 32 + 224], (1,))
            decoder = C.make_decoder(layer, None, device)
            self.sims[layer] = C.Simulated(P, decoder, self.mailbox, (batch,), mla, ranks, g, device,
                                           front_args, self.twoshot, deadline_ns)
            self.weights[layer] = {q: C.rank_weight(layer, q, device) for q in range(C.TP)}
            for i, kind in enumerate(("producer", "consumer")):
                H.record({"kind": "resources", "item": f"{self.name}.l{layer}.{kind}",
                          **H.function_resources(self.sims[layer].prep.codes[batch][i])})

    def reset_front(self):
        self.xmoe.view(torch.int32).fill_(C.SENTINEL_I32)  # MoeFrontTP8 re-arms it
        self.epoch.zero_()

    def inputs(self, layer, call):
        b, d = self.batch, self.device
        xs = {q: C.rank_input(layer, call, q, b, d) for q in range(C.TP)}
        prefix_old, bank = C.state_inputs(layer, call, b, d)
        partials = [C.partial_ref(xs[q], self.weights[layer][q]) for q in range(C.TP)]
        ref = C.reference(self.sims[layer].decoder, partials, prefix_old, bank)
        return xs, prefix_old, bank, partials, ref

    @staticmethod
    def residual(layer, prefix_old):
        """What a two-shot owner adds before its rounding: the prefix on a has_delta layer, only with
        K31MK_OPROJ_SINGLE_ROUND=1 (the single rounding); knob off (base D) the owner adds no prefix and
        the consumer adds it after the gather (the stand-in owners must match the kernel's owners)."""
        write, _ = C.layer_facts(layer)
        single = os.environ.get("K31MK_OPROJ_SINGLE_ROUND", "0") == "1"
        return None if write or not single else prefix_old

    def verify(self, tag, layer, prefix, out, ref, partials, failures):
        parity = layer % 2
        name = f"{self.name}.l{layer}.g{self.g}.{tag}"
        try:
            C.check_outputs(name, prefix, out, *ref)
            ts = self.twoshot
            C.check_rearmed(name, ts.gather[parity] if ts else self.mailbox.buffer[parity], self.batch)
            if ts is not None:
                C.check_rearmed(f"{name}.recv", ts.recv[parity], self.batch)
                W = C.HIDDEN // C.TP
                for o, stand in ts.standins.items():
                    for q in self.sims[layer].ranks:
                        H.check(f"{name}.push.o{o}.q{q}", stand[parity, q, : self.batch],
                                partials[q][:, o * W : (o + 1) * W], 1e-3, 2**-7)
            if self.front:
                want = C.canon(out[0])
                same = bool(torch.equal(self.xmoe[parity, 0, 0].view(torch.int16), want.view(torch.int16)))
                hint = int(self.epoch[parity * 32])
                H.record({"kind": "check", "item": f"{name}.front", "xmoe_equal": same, "hint": hint})
                assert same, "x_moe publish differs from the consumer output"
                assert hint == 28, f"epoch hint is {hint}; MoeFrontTP8 waits for 28"
        except AssertionError as e:
            failures.append(f"{name}: {e}")
            print("FAIL", name, str(e)[:400], flush=True)

    def eager(self, failures):
        b = self.batch
        out = torch.empty((b, C.HIDDEN), dtype=torch.bfloat16, device=self.device)
        for call in (0, 1):
            for layer in self.layers:
                xs, prefix_old, bank, partials, ref = self.inputs(layer, call)
                if self.twoshot is not None:
                    self.twoshot.inject(layer % 2, partials, self.residual(layer, prefix_old))
                prefix = prefix_old.clone()
                H.poison(out)
                self.sims[layer].run({q: xs[q] for q in self.sims[layer].ranks}, prefix, bank, out)
                torch.cuda.synchronize()
                self.verify(f"call{call}", layer, prefix, out, ref, partials, failures)
                if self.front:
                    self.reset_front()

    def graph(self, failures):
        b, d = self.batch, self.device
        staged = {}
        for layer in self.layers:
            staged[layer] = {
                "x": {q: torch.empty((b, C.K), dtype=torch.bfloat16, device=d) for q in self.sims[layer].ranks},
                "prefix": torch.empty((b, C.HIDDEN), dtype=torch.bfloat16, device=d),
                "bank": torch.empty((b, 8, C.HIDDEN), dtype=torch.bfloat16, device=d),
                "out": torch.empty((b, C.HIDDEN), dtype=torch.bfloat16, device=d),
            }
        graph, stream = torch.cuda.CUDAGraph(), torch.cuda.Stream()
        with torch.cuda.graph(graph, stream=stream):
            for layer in self.layers:
                st = staged[layer]
                self.sims[layer].run(st["x"], st["prefix"], st["bank"], st["out"])
        torch.cuda.synchronize()
        dist.barrier(group=self.group)
        for call in (2, 3):
            refs = {}
            for layer in self.layers:
                xs, prefix_old, bank, partials, ref = self.inputs(layer, call)
                st = staged[layer]
                for q in self.sims[layer].ranks:
                    st["x"][q].copy_(xs[q])
                st["prefix"].copy_(prefix_old)
                st["bank"].copy_(bank)
                H.poison(st["out"])
                if self.twoshot is not None:
                    self.twoshot.inject(layer % 2, partials, self.residual(layer, prefix_old))
                refs[layer] = (ref, partials)
            torch.cuda.synchronize()
            graph.replay()
            torch.cuda.synchronize()
            delay = os.environ.get("K31_TEST_VERIFY_DELAY", "")  # "g:seconds": control for the host-check race only
            if delay and int(delay.split(":")[0]) == self.g:
                time.sleep(float(delay.split(":")[1]))
            for layer in self.layers:
                st = staged[layer]
                self.verify(f"graph{call}", layer, st["prefix"], st["out"], *refs[layer], failures)
            if self.front:
                # The checks above saw both layers' x_moe: reset once per replay.
                self.reset_front()
            # One replay runs both parities, so a peer's next replay may publish its first layer into this GPU's
            # mailbox (call k + 2 needs only this GPU's call k + 1, already done) while the host still checks the
            # re-arm of call k: hold every rank until all checks of this replay are done (host-check race only).
            if os.environ.get("K31_TEST_NO_REPLAY_BARRIER", "0") != "1":
                dist.barrier(group=self.group)
        del graph

    def time(self):
        """Per-layer us over 32 chained alternating layers (graph), max over ranks;
        profiler breakdown over eager layers."""
        g, world, b, d, group = self.g, self.world, self.batch, self.device, self.group
        H.hold_fp8_clock()
        st = {}
        for layer in self.layers:
            st[layer] = {
                "x": {q: C.rank_input(layer, 11, q, b, d) for q in self.sims[layer].ranks},
                "prefix": torch.randn((b, C.HIDDEN), device=d).to(torch.bfloat16),
                "bank": torch.randn((b, 8, C.HIDDEN), device=d).to(torch.bfloat16),
                "out": torch.empty((b, C.HIDDEN), dtype=torch.bfloat16, device=d),
            }

        def layer_pair():
            for layer in self.layers:
                s = st[layer]
                self.sims[layer].run(s["x"], s["prefix"], s["bank"], s["out"])

        pairs = 16
        graph, stream = torch.cuda.CUDAGraph(), torch.cuda.Stream()
        with torch.cuda.graph(graph, stream=stream):
            for _ in range(pairs):
                layer_pair()
        torch.cuda.synchronize()
        for _ in range(3):
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
            times.append(a.elapsed_time(e) * 1e3 / (2 * pairs))
        times.sort()
        mine = torch.tensor([times[len(times) // 2]], device=d)
        allv = [torch.zeros_like(mine) for _ in range(world)]
        dist.all_gather(allv, mine, group=group)
        per_rank = [float(v) for v in allv]
        nprod = len(self.sims[self.layers[0]].ranks)
        row = {"kind": "time", "item": f"{self.name}{self.kinds}.layer_chain",
               "us_median_max_over_ranks": max(per_rank), "per_rank": per_rank,
               "producers_per_gpu": nprod, "layers": 2 * pairs,
               "floor_weight_read_us": nprod * C.HIDDEN * C.K * 2 / H.HBM_BPS * 1e6}
        H.record(row)
        if g == 0:
            print("TIME", json.dumps(row), flush=True)
        del graph

        from torch.profiler import ProfilerActivity, profile

        for _ in range(2):
            layer_pair()
        torch.cuda.synchronize()
        dist.barrier(group=group)
        with profile(activities=[ProfilerActivity.CUDA]) as prof:
            for _ in range(8):
                layer_pair()
            torch.cuda.synchronize()
        kernels = []
        for evt in prof.events():
            if evt.device_type.name != "CUDA" or "at::" in evt.name or "emcpy" in evt.name or "emset" in evt.name:
                continue
            start = evt.time_range.start
            dur = evt.device_time if hasattr(evt, "device_time") else evt.cuda_time
            kernels.append((start, start + dur, evt.name))
        kernels.sort()
        per = nprod + 1
        tails, cons_d, prod_d = [], [], []
        for i in range(0, len(kernels) - per + 1, per):
            ks = kernels[i : i + per]
            prods, cons = ks[:nprod], ks[nprod]
            tails.append(cons[1] - max(p[1] for p in prods))
            cons_d.append(cons[1] - cons[0])
            prod_d.extend(p[1] - p[0] for p in prods)
        med = lambda v: sorted(v)[len(v) // 2] if v else None  # noqa: E731
        row = {"kind": "time", "item": f"{self.name}{self.kinds}.profile.g{g}",
               "consumer_us": med(cons_d), "producer_us": med(prod_d),
               "consumer_end_after_last_local_producer_us": med(tails),
               "kernels": len(kernels), "names": sorted({k[2][:60] for k in kernels})}
        H.record(row)
        print("PROFILE", json.dumps(row), flush=True)


def diag(P, group, g, world, device, batch, mla, deadline_ns):
    """Timing breakdown of one width and producer kind, both layers with the same
    config (90 and 91: nb 8 + delta, parities 0 / 1):
    - producer_only: chained per-call us of this GPU's producer over 18 distinct weight
      copies (cold L2), no consumer (slots just get overwritten), max over ranks;
    - consumer_only: torch.profiler duration of the consumer launched right after a
      local copy that publishes all 8 slots (no waiting, no remote traffic);
    - layer_chain: Case.time (both GPUs' real producers + consumer, multicast)."""
    import cutlass

    from sgl_nvidia_mega.kimi_k31.k3.cutedsl.common.host import stream, to_cute

    case = Case(P, group, g, world, device, batch, ((90, mla), (91, mla), False), deadline_ns)
    sim = case.sims[90]
    tag = f"oproj_tp{world}.b{batch}.{'mla' if mla else 'kda'}"
    H.hold_fp8_clock()
    code = sim.prep.codes[batch][0]
    copies = H.copies_for(C.HIDDEN * C.K * 2)
    weights = [to_cute(C.rank_weight(2000 + i, g, device)) for i in range(copies)]
    x = to_cute(C.rank_input(90, 21, g, batch, device))
    s = lambda: stream(device)  # noqa: E731
    if batch in sim.prep.twoshots:
        recv, peers, _ = sim.prep.twoshot[0]
        calls = [lambda w=w: code(x, w, recv, peers, cutlass.Int32(g), s()) for w in weights]
    else:
        box, mc, _ = sim.prep.head
        calls = [lambda w=w: code(x, w, box, mc, cutlass.Int32(g), s()) for w in weights]
    dist.barrier(group=group)
    t = H.cold_time_us(calls)
    vals = [torch.zeros(1, device=device) for _ in range(world)]
    dist.all_gather(vals, torch.tensor([t["us_median"]], device=device), group=group)
    row = {"kind": "time", "item": f"{tag}.producer_only", **t, "max_over_ranks": max(float(v) for v in vals),
           **H.floor_us(C.HIDDEN * C.K * 2 + batch * (C.K + C.TP * C.HIDDEN) * 2)}
    H.record(row)
    if g == 0:
        print("DIAG", json.dumps(row), flush=True)
    torch.cuda.synchronize()
    dist.barrier(group=group)
    # Producer-only runs left slots published on every GPU: re-arm everything.
    case.mailbox.buffer.view(torch.int32).fill_(C.SENTINEL_I32)
    if case.twoshot is not None:
        for tns in (case.twoshot.recv, case.twoshot.gather, *case.twoshot.standins.values()):
            tns.view(torch.int32).fill_(C.SENTINEL_I32)
    torch.cuda.synchronize()
    dist.barrier(group=group)
    if case.twoshot is None:
        xs, prefix_old, bank, partials, _ = case.inputs(90, 22)
        staged = torch.stack(partials)  # [8, B, H]
        prefix, out = prefix_old.clone(), torch.empty_like(prefix_old)
        slots = case.mailbox.buffer[0, :, :batch]

        def one():
            slots.copy_(staged)
            sim.consumer(batch, prefix, bank, out, s())

        for _ in range(3):
            one()
        torch.cuda.synchronize()
        from torch.profiler import ProfilerActivity, profile

        with profile(activities=[ProfilerActivity.CUDA]) as prof:
            for _ in range(20):
                one()
            torch.cuda.synchronize()
        durs = sorted(
            (evt.device_time if hasattr(evt, "device_time") else evt.cuda_time)
            for evt in prof.events()
            if evt.device_type.name == "CUDA" and "oprojcons" in evt.name
        )
        row = {"kind": "time", "item": f"{tag}.consumer_only.g{g}", "us_median": durs[len(durs) // 2] if durs else None,
               "us_min": durs[0] if durs else None, "n": len(durs)}
        H.record(row)
        print("DIAG", json.dumps(row), flush=True)
        torch.cuda.synchronize()
        dist.barrier(group=group)
        case.time()
    dist.barrier(group=group)


def staging(P, group, g, world, device, batches, stages, deadline_ns):
    """B<=8 producer K-block staging sweep (``oproj.prepared.producer`` stages 4 of 12 K
    blocks in SMEM at B1/MLA, else 8): for each width and staged count, the full eager
    check (both layers KDA-config: 90, 91), the producer-only chained per-call us (cold
    weights, max over ranks), and the published partial bits of this GPU's rank, which
    must equal the shipped setting's bit for bit (staging moves weights between
    registers and SMEM; the MMA and K-part order do not change)."""
    import cutlass

    from sgl_nvidia_mega.kimi_k31.k3.cutedsl.common.host import stream, to_cute
    from sgl_nvidia_mega.kimi_k31.k3.cutedsl.oproj.producer import OprojProducer

    shipped = P.producer
    H.hold_fp8_clock()
    s = lambda: stream(device)  # noqa: E731
    bad = []
    for batch in batches:
        bits = {}
        default = 4 if batch == 1 else 8  # GB300 policy for KDA layers (the bitwise reference)
        for st in stages:
            def factory(k, b, tp, mla, twoshot=False, kind=None, st=st):
                if b <= 8:
                    return OprojProducer(k, staged_blocks=st, twoshot=twoshot)
                return shipped(k, b, tp, mla, twoshot, kind)

            P.producer = factory
            P.compile_oproj.cache_clear()
            case = Case(P, group, g, world, device, batch, ((90, False), (91, False), False), deadline_ns)
            fails = []
            case.eager(fails)
            bad.extend(f"staged{st}: {f}" for f in fails)
            sim = case.sims[90]
            code = sim.prep.codes[batch][0]
            box, mc, _ = sim.prep.head
            res = H.function_resources(code)
            # Bits of this rank's partial for fixed (x, W).
            x = to_cute(C.rank_input(90, 31, g, batch, device))
            w = sim.wviews[g]
            dist.barrier(group=group)
            code(x, w, box, mc, cutlass.Int32(g), s())
            torch.cuda.synchronize()
            dist.barrier(group=group)
            bits[st] = case.mailbox.buffer[0, g, :batch].clone()
            # Producer-only chained timing over cold weight copies.
            copies = H.copies_for(C.HIDDEN * C.K * 2)
            weights = [to_cute(C.rank_weight(3000 + i, g, device)) for i in range(copies)]
            calls = [lambda w=w: code(x, w, box, mc, cutlass.Int32(g), s()) for w in weights]
            dist.barrier(group=group)
            t = H.cold_time_us(calls)
            vals = [torch.zeros(1, device=device) for _ in range(world)]
            dist.all_gather(vals, torch.tensor([t["us_median"]], device=device), group=group)
            same = bool(torch.equal(bits[st].view(torch.int16), bits[default].view(torch.int16))) if default in bits else None
            row = {"kind": "time", "item": f"oproj_tp{world}.b{batch}.staged{st}.producer_only", **t,
                   "max_over_ranks": max(float(v) for v in vals), "reference_staged": default,
                   "bits_equal_shipped": same, "checks_failed": len(fails), **res}
            H.record(row)
            if g == 0:
                print("STAGING", json.dumps({k: row[k] for k in ("item", "us_median", "max_over_ranks", "bits_equal_shipped", "checks_failed", "regs")}), flush=True)
            torch.cuda.synchronize()
            dist.barrier(group=group)
            case.mailbox.buffer.view(torch.int32).fill_(C.SENTINEL_I32)
            torch.cuda.synchronize()
            dist.barrier(group=group)
        for st, b in bits.items():
            if not torch.equal(b.view(torch.int16), bits[default].view(torch.int16)):
                bad.append(f"b{batch} staged{st}: partial bits differ from shipped staged{default}")
    P.producer = shipped
    P.compile_oproj.cache_clear()
    return bad


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--batches", default=",".join(map(str, CASES)))
    ap.add_argument("--deadline-s", type=float, default=2.0)
    ap.add_argument("--diag", default="", help="widths for the timing breakdown (skips the checks)")
    ap.add_argument("--staging", default="", help="B<=8 widths for the producer staging sweep")
    ap.add_argument("--stages", default="2,4,6,8")
    args = ap.parse_args()
    local = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local)
    device = torch.device("cuda", local)
    dist.init_process_group("nccl", device_id=device)
    g, world = dist.get_rank(), dist.get_world_size()
    assert C.TP % world == 0, world
    tag = os.environ.get("K31_MUTATION") or "base"
    os.makedirs(args.out, exist_ok=True)
    os.environ["K31_RESULTS"] = os.path.join(args.out, f"tp{world}_{tag}_g{g}.jsonl")
    H.load_plugin(C.MUTATIONS)
    from sgl_nvidia_mega.kimi_k31.k3.cutedsl.oproj import prepared as P

    H.record({"kind": "env", **H.device_facts(), "world": world, "attributes": C.device_attributes()})
    if g == 0:
        print("DEVICE", json.dumps(C.device_attributes()), flush=True)
    failures = []
    if args.staging:
        bad = staging(P, dist.group.WORLD, g, world, device, [int(b) for b in args.staging.split(",")],
                      [int(v) for v in args.stages.split(",")], int(args.deadline_s * 1e9))
        with open(os.path.join(args.out, f"tp{world}_staging_g{g}.failures.json"), "w") as f:
            json.dump(bad, f, indent=1)
        t = torch.tensor([len(bad)], device=device)
        dist.all_reduce(t)
        if g == 0:
            print(f"STAGING_RESULT failures={int(t)}", flush=True)
        dist.destroy_process_group()
        sys.exit(1 if int(t) else 0)
    if args.diag:
        for batch in map(int, args.diag.split(",")):
            for mla in (True, False):
                diag(P, dist.group.WORLD, g, world, device, batch, mla, int(args.deadline_s * 1e9))
        dist.destroy_process_group()
        sys.exit(0)
    timing = C.env_flag("K31_TIME") and tag == "base"
    group = dist.group.WORLD
    dead = False
    for batch in map(int, args.batches.split(",")):
        for i, spec in enumerate(CASES[batch]):
            t0 = time.time()
            try:
                case = Case(P, group, g, world, device, batch, spec, int(args.deadline_s * 1e9))
                H.maybe_preload(f"oproj.b{batch}.{i}")
                case.eager(failures)
                case.graph(failures)
                if timing and i == 0 and case.twoshot is None:
                    case.time()
                dist.barrier(group=group)
            except Exception as e:  # noqa: BLE001
                failures.append(f"b{batch}.{i}: {type(e).__name__}: {e}")
                traceback.print_exc()
                print(f"CASE_ERROR b{batch}.{i} g{g}", flush=True)
                dead = True  # a trapped context cannot continue
                break
            print(f"CASE_DONE b{batch}.{i} g{g} {time.time() - t0:.1f}s failures={len(failures)}", flush=True)
        if dead:
            break
    if not dead:
        H.preload_audit("oproj.end")
    with open(os.path.join(args.out, f"tp{world}_{tag}_g{g}.failures.json"), "w") as f:
        json.dump(failures, f, indent=1)
    bad = len(failures)
    if not dead:
        t = torch.tensor([bad], device=device)
        dist.all_reduce(t)
        bad = int(t)
        if g == 0:
            print(f"TP_RESULT world={world} tag={tag} failures={bad}", flush=True)
        dist.destroy_process_group()
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
