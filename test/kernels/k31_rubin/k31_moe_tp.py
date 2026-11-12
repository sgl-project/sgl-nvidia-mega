# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""K3.1 TP8 B1 small-batch MoE layer (``moe.small_batch.chain.SmallBatchChain``:
MoeFrontTP8 -> ClusterExpertsTP8 -> MoeSharedTP8 -> MoeTail front_poll, exactly as the
model launches it) over W = 4 GPUs with real symm_mem multicast.

GPU g runs TP rank g. Ranks 4-7 are injected: their latent-down shards (FP32, from the
FP64 reference) go into every GPU's front latent mailbox, and their tail inputs
(latent partials, shared-expert shards for owner g) into the tail mailboxes; this
GPU's shared shards for owners 4-7 land in local scratch (``peers`` entries 4-7) and are
checked. Injected values are seeded identically on every GPU. A stand-in ring consumer
(``_bench_moe.RingConsumer``) takes the place of the next AttnRes: it polls the output
row of ranks 0-3 (columns 0-3583), saves it and re-arms it.

Six MoE layers run in sequence (layers 1-6 / ordinals 0-5: front parities alternate, tail
parities rotate 0-1-2), as consecutive layers do in the decode loop (see LAYERS for why 6).

Checks (tolerances stated before any run):
  ids      top-16 set == FP64 reference set (sigmoid(router logits) + bias, lower id on
           ties), unless the reference 16th/17th gap is < 1e-4 (then recorded, skipped);
  weights  routing weights (BF16) vs score / sum(scores): |d| <= 2^-8 |ref| + 1e-6;
  latent   gathered FP32 latent (all 8 shards) vs FP64 W_down @ x:
           |d| <= 1e-4 rms + 1e-5 |ref| (FP32 accumulation over K = 7168);
  hg       shared gate/up FP32 vs FP64: same bound;
  G        expert rows: T1 of test_k31_experts (Frobenius rel-L2 <= 1e-2, row <= 3e-2)
           vs the FP64 reference on the MK's own latent and ids;
  S        shared-expert partial (BF16) vs FP64 down @ bf16(SiTU(hg)):
           |d| <= 2^-7 |ref| + 2^-8 rms;
  partial  this rank's published latent partial == BF16(sum_j w_j G_j) within 1 BF16 ulp
           of max(|ref|, max_j |w_j G_j|), and every rank's slot is bit-identical on
           every GPU (multicast delivery);
  RS       received shared shards == the senders' S columns (bitwise, -0 -> +0), and the
           shards for owners 4-7 in scratch;
  output   rows of ranks 0-3 vs FP64 of the stock method from the mailbox contents
           (BF16 rank-order all-reduce, RMS over the FP32 sum, BF16 normalized latent,
           up projection, BF16, + BF16 shared sum, BF16): within 2 BF16 ulp of
           max(|out|, |up|, |shared|); and bit-identical on every GPU;
  graph    calls replayed from one CUDA graph (injection copies inside it) reproduce the
           eager outputs bit for bit.
Mutations (each must fail): front_bias, front_latent_drop, shared_situ_swap,
tail_drop_rank, tail_rms (see MUTATIONS).

Timing (K31_TIME=1): per-kernel device durations (torch.profiler, eager, all ranks
started together) and the per-layer time of a graph of 16 alternating layers (each with
its injection copies and the ring consumer, so it is an upper bound on the chain).

E-TAIL2HOP (``--arms mk,rep``): the ``rep`` arm runs the same front, experts and
shared expert, then ``moe.tail_rep.MoeTailRep`` (one [latent | shared] all-reduce hop,
replicated up projection, local ring stores) in place of MoeTail. Same checks; its
output covers all 7168 columns (no output hop), must be bit-identical on every GPU
and within 2 BF16 ulp of the same reference; the max ulp distance to the MK arm's
output on the same inputs is recorded. Pass/kill (fixed before running):
pass >= 1.5 us per layer saved at B1 (per-layer chain time, 8 distinct-weight layers,
ABBA, max over ranks); kill < 0.5 us.

Usage: python3 -m torch.distributed.run --nproc-per-node 4 k31_moe_tp.py --out DIR [--arms mk,rep]
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
import _ref_moe as R  # noqa: E402

TP, HID, LAT, SHARD = 8, 7168, 3584, 896
EPS = 1e-5
SENT = -0x80000000
# (layer index, MoE ordinal). Cycles of 6 MoE layers: even (the front's x_moe / latent mailboxes alternate parity by
# layer) and a multiple of 3 (a tail re-arms the previous layer's parity at its end; with 2 or 8 layers the wrap
# makes it the next layer's, and only the decode loop's o-proj hop orders that, which this harness lacks).
LAYERS = tuple((1 + i, i) for i in range(6))  # checked, run in sequence
TLAYERS = LAYERS  # timing: the same cycle with its own (distinct) weights

FT = "kimi_k31/k3/cutedsl/moe/small_batch/front_tp8.py"
TL = "kimi_k31/k3/cutedsl/moe/tail.py"
MUTATIONS = {
    "front_bias": (FT, "record = score_record(score, routing_add(score, my_bias))",
                   "record = score_record(score, routing_add(score, Float32(0)))"),
    "front_latent_drop": (FT, "for vector in range(tid, 448, 256):", "for vector in range(tid, 440, 256):"),
    "shared_situ_swap": (FT, "hsm[column] = Float32(BFloat16(situ(bits_float(gate[0]), bits_float(up[0]), beta, linear_beta)))",
                         "hsm[column] = Float32(BFloat16(situ(bits_float(up[0]), bits_float(gate[0]), beta, linear_beta)))"),
    "tail_drop_rank": (TL, "        for source in cutlass.range_constexpr(self.tp):\n            for k in cutlass.range_constexpr(8):\n                acc[k] = acc[k] + bf16(values[source + start, k // 2], k % 2)",
                       "        for source in cutlass.range_constexpr(self.tp - 1):\n            for k in cutlass.range_constexpr(8):\n                acc[k] = acc[k] + bf16(values[source + start, k // 2], k % 2)"),
    "tail_rms": (TL, "            inverse = inv_rms(total, eps)\n            gx = smem_words(sg.iterator + pk * 8)",
                 "            inverse = inv_rms(total + total, eps)\n            gx = smem_words(sg.iterator + pk * 8)"),
}


def bf(x):
    return x.to(torch.bfloat16).double()


def canon_bf16(t):
    """BF16 tensor with -0 halves made +0 (the MK's published form)."""
    v = t.contiguous().view(torch.int16).clone()
    v[v == -0x8000] = 0
    return v.view(torch.bfloat16)


def ulp_bf16(v):
    a = v.abs().clamp(min=1e-30)
    return torch.exp2(torch.floor(torch.log2(a)) - 7)


def gen(seed, device):
    g = torch.Generator(device=device)
    g.manual_seed(seed)
    return g


def randn(shape, scale, seed, device, dtype=torch.bfloat16):
    return (torch.randn(shape, generator=gen(seed, device), device=device) * scale).to(dtype)


class Experts:
    def __init__(self, g, device):
        self.logical = R.make_logical(R.E_ALL, gen(31 + g, device))
        self.prepared = R.host_layout(*self.logical)


class Model:
    """Seeded weights of one MoE layer: common (identical on all ranks) and rank g's
    shards; the routed experts are shared by all test layers."""

    def __init__(self, g, device, layer, experts):
        k = 1000 * layer
        self.router = randn((896, HID), 0.02, k + 11, device)
        self.bias = (torch.randn(896, generator=gen(k + 12, device), device=device) * 0.02).float()
        self.down = randn((LAT, HID), HID ** -0.5, k + 13, device)
        self.up = randn((HID, LAT), LAT ** -0.5, k + 14, device)
        self.gamma = (1.0 + 0.1 * torch.randn(LAT, generator=gen(k + 15, device), device=device)).to(torch.bfloat16)
        self.gate_up = randn((1536, HID), HID ** -0.5, k + 100 + g, device)
        self.shared_down = randn((HID, 768), 768 ** -0.5, k + 200 + g, device)
        self.logical, self.prepared = experts.logical, experts.prepared


class Rig:
    def __init__(self, P, group, g, world, device, deadline_ns, layer_specs, arms):
        CH, TR, ER, host, TRP, OP = P
        self.g, self.world, self.device, self.group, self.arms = g, world, device, group, arms
        self.ex_w = Experts(g, device)
        self.box = CH.SmallBatchBox(group, device, TP, g)
        tb = self.tail_box = TR.MoeTailBox(group, device, TP, g)
        self.shared_ar = OP.OprojMailbox(group, device, TP, g, (3, TP, TR.CAP, HID)) if "rep" in arms else None
        self.scratch = [torch.empty_like(tb.rs.buffer).view(torch.int32).fill_(SENT).view(torch.bfloat16)
                        for _ in range(TP - world)]
        ptrs = list(tb.rs.handle.buffer_ptrs) + [s.data_ptr() for s in self.scratch]
        self.peers = torch.tensor(ptrs, dtype=torch.int64, device=device)
        from cutlass import Float32, Int32, Int64

        self.I32, self.I64, self.F32 = Int32, Int64, Float32
        tb.tail = (host.to_cute(self.peers, 16), TR.flat(tb.out.buffer), Int64(tb.latent.mc), Int64(tb.out.mc))
        self.experts = ER.PersistentExperts(*self.ex_w.prepared, beta=R.BETA, linear_beta=R.LIN)
        self.to_cute, self.stream = host.to_cute, host.stream
        self.out = tb.out.buffer[0, :1]
        self.consumer = {"mk": B.ring_consumer(LAT // 2, device.index), "rep": B.ring_consumer(HID // 2, device.index)}
        self.rep_code = None
        if "rep" in arms:
            sms = torch.cuda.get_device_properties(device).multi_processor_count
            self.rep_code = TRP.compile_tail_rep(1, sms, device.index, deadline_ns)
        self.layers = {}
        op = SimpleNamespace(rank=g, contract=SimpleNamespace(tp_size=TP, rms_eps=EPS))
        for key, specs in layer_specs.items():
            for layer, ordinal in specs:
                m = Model(g, device, layer if key == "check" else 100 + layer, self.ex_w)
                moe = SimpleNamespace(
                    shared_experts=SimpleNamespace(gate_up_proj=SimpleNamespace(weight=m.gate_up),
                                                   down_proj=SimpleNamespace(weight=m.shared_down)),
                    gate=SimpleNamespace(weight=m.router, e_score_correction_bias=m.bias),
                    routed_expert_down_proj=SimpleNamespace(weight=m.down),
                    routed_scaling_factor=1.0, moe_renormalize=True)
                tail = TR.PreparedTail(op, m.gamma, m.up, (1,), (tb, ordinal, len(specs)), deadline_ns=deadline_ns)
                chain = CH.SmallBatchChain(self.box, moe, self.experts, tail, layer, R.BETA, R.LIN, None,
                                           deadline_ns=deadline_ns)
                parity = (layer % 2, int(tail.tail_weights.parity))
                rearm = tail.tail_weights.scalars[2]
                self.layers[(key, layer)] = SimpleNamespace(m=m, chain=chain, parity=parity, rearm=rearm, layer=layer)

    # -------------------------------------------------------------------- inputs
    def inputs(self, call, lay):
        """Per-call source tensors (identical on every GPU)."""
        d, m = self.device, lay.m
        x = randn((HID,), 1.0, 1000 + call, d)
        lat_ref = m.down.double() @ x.double()
        inj_lat = {q: randn((LAT,), 0.5, 2000 + 10 * call + q, d) for q in range(self.world, TP)}
        inj_s = {q: randn((HID,), 0.5, 3000 + 10 * call + q, d) for q in range(self.world, TP)}
        return SimpleNamespace(x=x, lat_ref=lat_ref, inj_lat=inj_lat, inj_s=inj_s, lay=lay, call=call)

    def stage(self, src):
        """Source tensors laid out exactly as the injection writes them."""
        p, t = src.lay.parity
        lat32 = src.lat_ref.float()
        return SimpleNamespace(
            xmoe=canon_bf16(src.x), lat=lat32[self.world * 448:].contiguous(),
            tl=torch.stack([canon_bf16(src.inj_lat[q]) for q in range(self.world, TP)]),
            rs=torch.stack([canon_bf16(src.inj_s[q][self.g * SHARD:(self.g + 1) * SHARD]) for q in range(self.world, TP)]),
            sar=torch.stack([canon_bf16(src.inj_s[q]) for q in range(self.world, TP)]),
            hint=torch.tensor([28], dtype=torch.int32, device=self.device), p=p, t=t, lay=src.lay)

    def inject(self, st, arm):
        b, tb, w = self.box, self.tail_box, self.world
        b.xmoe[st.p, 0, 0].copy_(st.xmoe)
        b.epoch[st.p * 32: st.p * 32 + 1].copy_(st.hint)
        b.h2.buffer[st.p].view(-1).view(torch.float32)[w * 448: TP * 448].copy_(st.lat)
        tb.latent.buffer[st.t, w:TP, 0].copy_(st.tl)
        if arm == "mk":
            tb.rs.buffer[st.t, w:TP, 0].copy_(st.rs)
        else:
            self.shared_ar.buffer[st.t, w:TP, 0].copy_(st.sar)

    def run(self, lay, save, arm):
        s = self.stream(self.device)
        if arm == "mk":
            lay.chain.run(self.out, lambda o: o)
            ring = self.out.view(-1)[:LAT].view(torch.int32)
        else:
            ch, I32, I64 = lay.chain, self.I32, self.I64
            par, slot = I32(ch.parity), I32(ch.slot)
            ch.front[0](*ch.front[1], par, slot, *ch.scalars, s)
            ch.routed[0](*ch.routed[1], s)
            ch.shared[0](*ch.shared[1], par, slot, s)
            tb, b, c = self.tail_box, self.box, lambda t: self.to_cute(t.reshape(-1), 16)
            self.rep_code(c(tb.latent.buffer), c(self.shared_ar.buffer), c(lay.m.up), c(lay.m.gamma),
                          c(b.g), c(b.w), c(b.s), c(tb.out.buffer), I64(tb.latent.mc), I64(self.shared_ar.mc),
                          slot, I32(self.g), I32(64), self.F32(EPS), lay.rearm, s)
            ring = self.out.view(-1).view(torch.int32)
        self.consumer[arm](self.to_cute(ring, 16), self.to_cute(save, 16), s)

    # -------------------------------------------------------------------- checks
    def gather(self, t):
        out = [torch.empty_like(t) for _ in range(self.world)]
        dist.all_gather(out, t.contiguous(), group=self.group)
        return out

    def check_eager(self, src, save, fails, arm, mk_out=None):
        g, b, tb, w = self.g, self.box, self.tail_box, self.world
        m = src.lay.m
        p, t = src.lay.parity
        name = f"moe_tp{w}.{arm}.b1.l{src.lay.layer}.call{src.call}.g{g}"

        def fail(what, msg):
            fails.append(f"{name}.{what}: {msg}")
            print("FAIL", name, what, msg[:300], flush=True)

        x = src.x.double()
        # routing
        logits = m.router.double() @ x
        score = torch.sigmoid(logits)
        biased = score + m.bias.double()
        order = torch.sort(biased, descending=True, stable=True)
        top = order.indices[:16]
        gap = float(order.values[15] - order.values[16])
        ids = b.ids.long()
        if gap >= 1e-4 and set(ids.tolist()) != set(top.tolist()):
            fail("ids", f"{sorted(ids.tolist())} vs {sorted(top.tolist())}")
        wref = score[ids] / score[ids].sum()
        wmk = b.w[t, 0, :16].double()
        bad = (wmk - wref).abs() > (2 ** -8) * wref.abs() + 1e-6
        if bad.any():
            fail("weights", f"{int(bad.sum())} bad: {wmk.tolist()} vs {wref.tolist()}")
        H.record({"kind": "check", "item": f"{name}.route", "gap16": gap, "w_maxerr": float((wmk - wref).abs().max())})
        # latent and shared gate/up
        lat = b.latent.double()
        tol = 1e-4 * float(src.lat_ref.pow(2).mean().sqrt()) + 1e-5 * src.lat_ref.abs()
        nbad = int(((lat - src.lat_ref).abs() > tol).sum())
        H.record({"kind": "check", "item": f"{name}.latent", "max_abs": float((lat - src.lat_ref).abs().max()), "violations": nbad})
        if nbad:
            fail("latent", f"{nbad} cells; max {float((lat - src.lat_ref).abs().max()):.3e}")
        hg = b.hg.view(2, 2, 1536)[p, 0].view(torch.float32).double()
        hg_ref = m.gate_up.double() @ x
        tol = 1e-4 * float(hg_ref.pow(2).mean().sqrt()) + 1e-5 * hg_ref.abs()
        nbad = int(((hg - hg_ref).abs() > tol).sum())
        H.record({"kind": "check", "item": f"{name}.hg", "max_abs": float((hg - hg_ref).abs().max()), "violations": nbad})
        if nbad:
            fail("hg", f"{nbad} cells")
        # experts (on the MK's own latent and ids)
        G = b.g[t, :16].view(1, 16, LAT)
        ref_rows = R.expert_rows(m.logical, b.latent.view(1, LAT), b.ids.view(1, 16))
        fro = R.rel_l2(G, ref_rows)
        row = float(((G.double() - ref_rows).norm(dim=-1) / ref_rows.norm(dim=-1)).max())
        H.record({"kind": "check", "item": f"{name}.experts", "t1_fro": fro, "t1_row_max": row})
        if not (torch.isfinite(G.float()).all() and fro <= 1e-2 and row <= 3e-2):
            fail("experts", f"T1 {fro:.3e} {row:.3e}")
        # shared expert
        hs = R.situ(hg[:768], hg[768:])
        s_ref = m.shared_down.double() @ bf(hs.float())
        S = b.s[t, 0].double()
        tol = 2 ** -7 * s_ref.abs() + 2 ** -8 * float(s_ref.pow(2).mean().sqrt())
        nbad = int(((S - s_ref).abs() > tol).sum())
        H.record({"kind": "check", "item": f"{name}.shared", "max_abs": float((S - s_ref).abs().max()), "violations": nbad})
        if nbad:
            fail("shared", f"{nbad} cells")
        # published latent partial of this rank, and multicast delivery of every rank
        wts = b.w[t, 0, :16].double()
        terms = wts.view(16, 1) * G[0].double()
        part_ref = terms.sum(0)
        mine = tb.latent.buffer[t, g, 0].double()
        lim = ulp_bf16(torch.maximum(part_ref.abs(), terms.abs().max(0).values))
        nbad = int(((mine - part_ref).abs() > lim).sum())
        H.record({"kind": "check", "item": f"{name}.partial", "violations": nbad})
        if nbad:
            fail("partial", f"{nbad} cells beyond 1 ulp")
        slots = tb.latent.buffer[t, :TP, 0].clone()  # [8, 3584] on this GPU
        allslots = self.gather(slots[:w])
        for q in range(w):
            if not torch.equal(allslots[q][q].view(torch.int16), slots[q].view(torch.int16)):
                fail("multicast", f"latent slot {q} differs from rank {q}'s own copy")
        S_all = self.gather(b.s[t, 0].clone())
        if arm == "mk":  # shared reduce-scatter
            for q in range(w):
                got = tb.rs.buffer[t, q, 0]
                want = canon_bf16(S_all[q][g * SHARD:(g + 1) * SHARD])
                if not torch.equal(got.view(torch.int16), want.view(torch.int16)):
                    fail("rs", f"shard from rank {q} differs")
            for o in range(w, TP):
                got = self.scratch[o - w][t, g, 0]
                want = canon_bf16(b.s[t, 0, o * SHARD:(o + 1) * SHARD])
                if not torch.equal(got.view(torch.int16), want.view(torch.int16)):
                    fail("rs_out", f"shard for owner {o} differs")
        else:  # shared all-reduce slots (multicast)
            for q in range(w):
                got = self.shared_ar.buffer[t, q, 0]
                if not torch.equal(got.view(torch.int16), canon_bf16(S_all[q]).view(torch.int16)):
                    fail("shared_ar", f"slot {q} differs from rank {q}'s shared partial")
        # output rows
        Lsum = slots.double().sum(0)
        inv = 1.0 / torch.sqrt((Lsum ** 2).mean() + EPS)
        xn = bf(bf(Lsum) * inv * m.gamma.double())
        y = bf(m.up.double() @ xn)
        S_full = [S_all[q].double() for q in range(w)] + [src.inj_s[q].double() for q in range(w, TP)]
        sh = bf(sum(S_full))
        out_ref = bf(y + sh)
        n = LAT if arm == "mk" else HID
        out = save.view(torch.bfloat16).double()
        lim = 2 * ulp_bf16(torch.maximum(torch.maximum(out_ref[:n].abs(), y[:n].abs()), sh[:n].abs()))
        err = (out - out_ref[:n]).abs()
        nbad = int((err > lim).sum())
        row = {"kind": "check", "item": f"{name}.output", "max_abs": float(err.max()),
               "max_ulps": float((err / ulp_bf16(out_ref[:n].abs())).max()), "violations": nbad}
        if mk_out is not None:
            d = (out[:LAT] - mk_out.view(torch.bfloat16).double()).abs()
            row["max_ulps_vs_mk"] = float((d / ulp_bf16(out_ref[:LAT].abs())).max())
            row["bitwise_vs_mk"] = bool(torch.equal(save[: LAT // 2], mk_out))
        H.record(row)
        print("CHECK", json.dumps(row), flush=True)
        if nbad or not torch.isfinite(out).all():
            fail("output", f"{nbad} cells beyond 2 ulp; max {float(err.max()):.3e}")
        outs = self.gather(save.clone())
        if not all(torch.equal(o, save) for o in outs):
            fail("rank_bitwise", "output rows differ across GPUs")
        for what, value in (("out", save), ("latent", b.latent), ("experts", G), ("shared", b.s[t, 0])):
            H.digest(f"{name}.{what}", value)


def eager_and_graph(rig, fails, arm, mk_outs=None):
    d = rig.device
    lays = [rig.layers[("check", layer)] for layer, _ in LAYERS]
    srcs = [rig.inputs(c, lays[c]) for c in range(len(lays))]
    stages = [rig.stage(s) for s in srcs]
    n = (LAT if arm == "mk" else HID) // 2
    saves = [torch.empty(n, dtype=torch.int32, device=d) for _ in lays]
    eager_out = []
    for c in range(len(lays)):
        rig.inject(stages[c], arm)
        torch.cuda.synchronize()
        dist.barrier(group=rig.group)
        saves[c].fill_(SENT)
        rig.run(lays[c], saves[c], arm)
        torch.cuda.synchronize()
        dist.barrier(group=rig.group)
        rig.check_eager(srcs[c], saves[c], fails, arm, None if mk_outs is None else mk_outs[c])
        eager_out.append(saves[c].clone())
        torch.cuda.synchronize()
        dist.barrier(group=rig.group)
    gsaves = [torch.empty_like(s) for s in saves]

    def both():
        for c in range(len(lays)):
            rig.inject(stages[c], arm)
            rig.run(lays[c], gsaves[c], arm)

    graph, stream = torch.cuda.CUDAGraph(), torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.graph(graph, stream=stream):
        both()
    torch.cuda.synchronize()
    for rep in range(2):
        for s in gsaves:
            s.fill_(SENT)
        torch.cuda.synchronize()
        dist.barrier(group=rig.group)
        graph.replay()
        torch.cuda.synchronize()
        dist.barrier(group=rig.group)
        for c in range(len(lays)):
            same = torch.equal(gsaves[c], eager_out[c])
            H.record({"kind": "check", "item": f"moe_tp{rig.world}.{arm}.b1.graph{rep}.call{c}.g{rig.g}", "bitwise_vs_eager": same})
            if not same:
                fails.append(f"{arm}.graph{rep}.call{c}.g{rig.g}: differs from eager")
                print("FAIL graph", arm, rep, c, flush=True)
    del graph
    return eager_out


def timing(rig, arms):
    """Per-layer chain time (8 distinct-weight layers x 2, graph, injection copies and the
    ring consumer included; the same for both arms), ABBA over arms; per-kernel durations."""
    g, d = rig.g, rig.device
    H.hold_fp8_clock()
    lays = [rig.layers[("time", layer)] for layer, _ in TLAYERS]
    srcs = [rig.inputs(50 + c, lays[c]) for c in range(len(lays))]
    stages = [rig.stage(s) for s in srcs]
    graphs = {}
    for arm in arms:
        n = (LAT if arm == "mk" else HID) // 2
        saves = [torch.empty(n, dtype=torch.int32, device=d) for _ in lays]

        def chain(arm=arm, saves=saves):
            for c, lay in enumerate(lays):
                rig.inject(stages[c], arm)
                rig.run(lay, saves[c], arm)

        for _ in range(2):  # eager warm-up (also leaves the parities consistent)
            chain()
        torch.cuda.synchronize()
        dist.barrier(group=rig.group)
        graph, stream = torch.cuda.CUDAGraph(), torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.graph(graph, stream=stream):
            chain()
            chain()
        torch.cuda.synchronize()
        graphs[arm] = (graph, chain, saves)
    per = {a: [] for a in arms}
    for arm in list(arms) + list(arms)[::-1]:
        graph = graphs[arm][0]
        for _ in range(2):
            dist.barrier(group=rig.group)
            graph.replay()
            torch.cuda.synchronize()
        times = []
        for _ in range(10):
            dist.barrier(group=rig.group)
            a, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            a.record()
            graph.replay()
            e.record()
            e.synchronize()
            times.append(a.elapsed_time(e) * 1e3 / (2 * len(lays)))
        times.sort()
        allv = rig.gather(torch.tensor([times[len(times) // 2]], device=d))
        per[arm].append(max(float(v) for v in allv))
    row = {"kind": "time", "item": f"moe_tp{rig.world}.b1.layer_chain_with_standins",
           **{f"{a}_us_max_over_ranks": sum(v) / len(v) for a, v in per.items()}, "runs": per}
    if "mk" in per and "rep" in per:
        row["saved_us_per_layer"] = row["mk_us_max_over_ranks"] - row["rep_us_max_over_ranks"]
    H.record(row)
    if g == 0:
        print("TIME", json.dumps(row), flush=True)
    from torch.profiler import ProfilerActivity, profile

    for arm in arms:
        chain = graphs[arm][1]
        torch.cuda.synchronize()
        dist.barrier(group=rig.group)
        with profile(activities=[ProfilerActivity.CUDA]) as prof:
            graphs[arm][0].replay()  # all ranks start together (barrier above)
            torch.cuda.synchronize()
        durs = {}
        for evt in prof.events():
            if evt.device_type.name != "CUDA":
                continue
            dur = evt.device_time if hasattr(evt, "device_time") else evt.cuda_time
            durs.setdefault(B.kname(evt.name), []).append(dur)
        summary = {k: sorted(v)[len(v) // 2] for k, v in durs.items()}
        H.record({"kind": "time", "item": f"moe_tp{rig.world}.{arm}.b1.kernels.g{g}", "median_us": summary})
        print("KERNELS", arm, g, json.dumps(summary), flush=True)
    for arm in arms:
        del graphs[arm]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--deadline-s", type=float, default=2.0)
    ap.add_argument("--arms", default="mk")
    args = ap.parse_args()
    arms = tuple(args.arms.split(","))
    local = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local)
    device = torch.device("cuda", local)
    dist.init_process_group("nccl", device_id=device)
    g, world = dist.get_rank(), dist.get_world_size()
    tag = os.environ.get("K31_MUTATION") or "base"
    os.makedirs(args.out, exist_ok=True)
    os.environ["K31_RESULTS"] = os.path.join(args.out, f"moe_tp{world}_{tag}_g{g}.jsonl")
    H.load_plugin(MUTATIONS)
    from sgl_nvidia_mega.kimi_k31.k3.cutedsl.common import host
    from sgl_nvidia_mega.kimi_k31.k3.cutedsl.moe import experts_runtime as ER
    # The opt-in replicated-up tail (E-TAIL2HOP, killed) is not part of the integrated build.
    TRP = None
    if "rep" in args.arms.split(","):
        from sgl_nvidia_mega.kimi_k31.k3.cutedsl.moe import tail_rep as TRP
    from sgl_nvidia_mega.kimi_k31.k3.cutedsl.moe import tail_runtime as TR
    from sgl_nvidia_mega.kimi_k31.k3.cutedsl.moe.small_batch import chain as CH
    from sgl_nvidia_mega.kimi_k31.k3.cutedsl.oproj import prepared as OP

    H.record({"kind": "env", **H.device_facts(), "world": world})
    fails = []
    t0 = time.time()
    try:
        timed = os.environ.get("K31_TIME") == "1" and tag == "base"
        specs = {"check": LAYERS, **({"time": TLAYERS} if timed else {})}
        rig = Rig((CH, TR, ER, host, TRP, OP), dist.group.WORLD, g, world, device, int(args.deadline_s * 1e9),
                  specs, arms)
        H.maybe_preload("moe.rig")
        mk_outs = None
        for arm in arms:
            outs = eager_and_graph(rig, fails, arm, mk_outs if arm == "rep" else None)
            if arm == "mk":
                mk_outs = outs
        H.preload_audit("moe.end")
        if timed:
            timing(rig, arms)
    except Exception as e:  # noqa: BLE001
        fails.append(f"g{g}: {type(e).__name__}: {e}")
        traceback.print_exc()
    with open(os.path.join(args.out, f"moe_tp{world}_{tag}_g{g}.failures.json"), "w") as f:
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
