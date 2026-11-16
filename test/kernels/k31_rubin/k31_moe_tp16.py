# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""K3.1 TP16 MoE on the FP32 latent over W = 4 GPUs with real symm_mem multicast.

Arm ``small``: the TP16 B1/B2 small-batch layer (``moe.small_batch.chain.SmallBatchChain`` at TP16:
MoeFrontTP16 -> ClusterExpertsTP16 -> MoeTail front_poll), six layers in sequence per batch size (front parities
alternate, tail parities rotate 0-1-2, as in k31_moe_tp.py). Arm ``generic``: the TP16 generic front
(``moe.front.SharedFront.route`` at B8, the Q8 verify shape: Fp32FrontGemm publish -> polled routing kernel with the
FP32 latent copy).

GPU g runs TP rank g of 16. Ranks 4-15 are injected, seeded identically on every GPU: their FP32 latent shards (from
the FP64 reference) into every GPU's latent mailbox (small: h2; generic: the front mailbox, with their FP32 router
logits), and their tail inputs (BF16 latent partials; shared-expert shards for owner g) into the tail mailboxes. This
GPU's shared shards for owners 4-15 land in local scratch and are checked. A stand-in ring consumer
(``_bench_moe.RingConsumer``) takes the place of the next AttnRes: it polls the output row of ranks 0-3
(columns 0-1791) of each token, saves it and re-arms it.

Checks per token (tolerances stated before any run; the TP8 test's bounds where the quantity is the same):
  ids      top-16 set == FP64 reference set (sigmoid(router logits) + bias, lower id on ties), unless the reference
           16th/17th gap is < 1e-4 (recorded, skipped). Small arm: ClusterExpertsTP16 re-arms its ids, so the eager
           calls copy them between the front and the experts; the CUDA-graph calls run ``SmallBatchChain.run``.
  weights  BF16 routing weights vs score / sum(scores) over the MK's ids: |d| <= 2^-8 |ref| + 1e-6.
  latent   FP32 latent rows of ranks 0-3 vs FP64 W_down @ x: |d| <= 1e-4 rms + 1e-5 |ref| (FP32 accumulation over
           K = 7168; a BF16-rounded latent is off by up to 2^-9 |ref|); injected rows bit-exact; small arm: the
           3584 words bitwise identical on every GPU (multicast delivery).
  hg       shared gate/up (FP32 since A4) vs FP64: the latent bound.
  G        expert rows vs the FP64 stock reference (``_ref_moe16``) on the MK's own FP32 latent and ids: T1 of
           test_k31_experts (Frobenius rel-L2 <= 1e-2, every row <= 3e-2).
  S        shared-expert partial (BF16) vs FP64 down @ bf16(SiTU(hg)): |d| <= 2^-7 |ref| + 2^-8 rms.
  partial  this rank's published latent partial == BF16(sum_j w_j G_j) within 1 BF16 ulp of max(|ref|,
           max_j |w_j G_j|); every rank's slot bit-identical on every GPU.
  RS       received shared shards == the senders' S columns (bitwise, -0 -> +0), and the shards for owners 4-15.
  output   rows of ranks 0-3 vs FP64 of the stock method from the mailbox contents (BF16 rank-order all-reduce of
           16 partials, RMS, BF16 normalized latent, up projection, BF16, + BF16 shared sum, BF16): within 2 BF16
           ulp of max(|out|, |up|, |shared|); bit-identical on every GPU.
  graph    the chain replayed from one CUDA graph (injection copies inside) reproduces the eager outputs bit for bit.
Mutations (each must fail): f16_bf16_latent (MoeFrontTP16 publishes a BF16-rounded latent: the pre-A4 precision),
f16_bf16_hg (MoeFrontTP16 rounds the shared gate/up to BF16), gen_bf16_latent (Fp32FrontGemm publishes BF16-rounded
values; generic arm).

Generic arm above 64 rows: ``--front-rows 128 --generic-batches 33,64,128`` runs the same
route checks over a 128-row front box at every listed batch, plus
  plan     the routing kernel writes the experts' plan only at 8 < B <= 64 (one 64-row block, experts.PLAN_BYTES): the
           PLAN_BYTES bytes after the plan (a canary, 0x5A) are unchanged after every call, at every batch.
  hg / S   at B > 64 (the split-K shared expert; FUSED stops at 64): hg and S as in the small arm, from
           ``SharedFront.shared_expert`` with a stand-in tail (it must push exactly the returned rows, no side launch).
           DIAG lines (no pass / fail) split S misses between SiTU (h elements that differ from the reference's h) and
           the down GEMM (S against FP64 down @ the kernel's own h, within one BF16 rounding + FP32 accumulation).
Mutations for it (must fail at 128 rows): gen_wrap64 (the publish wraps token rows at 64), copy_wrap64 (the copy role
reads row token % 64), plan_gt64 (the routing kernel writes plans above 64 rows).

Usage: python3 -m torch.distributed.run --nproc-per-node 4 k31_moe_tp16.py --out DIR [--arms small,generic]
       [--generic-batches 8] [--front-rows 64]
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
import _ref_moe16 as M  # noqa: E402

TP, HID, LAT = 16, 7168, 3584
SHARD, LROWS, LOGITS = HID // TP, LAT // TP, 896 // TP  # 448 output columns, 224 latent rows, 56 logits per rank
EPS = 1e-5
SENT = -0x80000000
LAYERS = tuple((1 + i, i) for i in range(6))  # (layer index, MoE ordinal): see k31_moe_tp.py for why 6
BATCHES = (1, 2)

F16 = "kimi_k31/k3/cutedsl/moe/small_batch/front_tp16.py"
FP = "kimi_k31/k3/cutedsl/moe/projection.py"
FR_PATH = "kimi_k31/k3/cutedsl/moe/front.py"
MUTATIONS = {
    "f16_bf16_latent": (F16, "words = (positive_zero32(lo), positive_zero32(hi))",
                        "words = (positive_zero32(Float32(BFloat16(lo))), positive_zero32(Float32(BFloat16(hi))))"),
    "f16_bf16_hg": (F16, "mailbox_store(pointer, (positive_zero32(total),))",
                    "mailbox_store(pointer, (positive_zero32(Float32(BFloat16(total))),))"),
    "gen_bf16_latent": (FP, "words = tuple(positive_zero32(values[4 * v + i]) for i in range(4))",
                        "words = tuple(positive_zero32(c.Float32(c.BFloat16(values[4 * v + i]))) for i in range(4))"),
    # Front box above 64 rows (each must fail at 128 rows and stay silent at <= 64):
    "gen_wrap64": (FP, "offset = token * out.shape[2] + first + 8 * v",
                   "offset = (token % 64) * out.shape[2] + first + 8 * v"),
    "copy_wrap64": (FR_PATH, "base = mailbox.iterator + token * mailbox.shape[2] + tid * (16 if self.fp32 else 8)",
                    "base = mailbox.iterator + (token % 64) * mailbox.shape[2] + tid * (16 if self.fp32 else 8)"),
    "plan_gt64": (FR_PATH, "planned = plan is not None and 8 < b <= PLAN_ROWS", "planned = plan is not None and b > 8"),
}
CANARY = 0x5A


def bf(x):
    return x.to(torch.bfloat16).double()


def canon_bf16(t):
    v = t.contiguous().view(torch.int16).clone()
    v[v == -0x8000] = 0
    return v.view(torch.bfloat16)


def ulp_bf16(v):
    return torch.exp2(torch.floor(torch.log2(v.abs().clamp(min=1e-30))) - 7)


def gen(seed, device):
    g = torch.Generator(device=device)
    g.manual_seed(seed)
    return g


def randn(shape, scale, seed, device, dtype=torch.bfloat16):
    return (torch.randn(shape, generator=gen(seed, device), device=device) * scale).to(dtype)


class Model:
    """Seeded weights of one MoE layer: common (identical on all ranks) and rank g's shared-expert shards."""

    def __init__(self, g, device, layer):
        k = 1000 * layer
        self.router = randn((896, HID), 0.02, k + 11, device)
        self.bias = (torch.randn(896, generator=gen(k + 12, device), device=device) * 0.02).float()
        self.down = randn((LAT, HID), HID ** -0.5, k + 13, device)
        self.up = randn((HID, LAT), LAT ** -0.5, k + 14, device)
        self.gamma = (1.0 + 0.1 * torch.randn(LAT, generator=gen(k + 15, device), device=device)).to(torch.bfloat16)
        self.gate_up = randn((768, HID), HID ** -0.5, k + 100 + g, device)
        self.shared_down = randn((HID, 384), 384 ** -0.5, k + 200 + g, device)

    def moe(self):
        return SimpleNamespace(
            shared_experts=SimpleNamespace(gate_up_proj=SimpleNamespace(weight=self.gate_up),
                                           down_proj=SimpleNamespace(weight=self.shared_down)),
            gate=SimpleNamespace(weight=self.router, e_score_correction_bias=self.bias),
            routed_expert_down_proj=SimpleNamespace(weight=self.down),
            routed_scaling_factor=1.0, moe_renormalize=True)


def route_ref(m, x):
    """FP64 sigmoid scores, the stable top-16 by biased score and the 16th/17th gap."""
    score = torch.sigmoid(m.router.double() @ x.double())
    order = torch.sort(score + m.bias.double(), descending=True, stable=True)
    return score, order.indices[:16], float(order.values[15] - order.values[16])


class Checker:
    def __init__(self, name, fails):
        self.name, self.fails = name, fails

    def fail(self, what, msg):
        self.fails.append(f"{self.name}.{what}: {msg}")
        print("FAIL", self.name, what, msg[:300], flush=True)

    def route(self, m, x, ids, wmk):
        score, top, gap = route_ref(m, x)
        ids = ids.long()
        if gap >= 1e-4 and set(ids.tolist()) != set(top.tolist()):
            self.fail("ids", f"{sorted(ids.tolist())} vs {sorted(top.tolist())}")
        wref = score[ids] / score[ids].sum()
        bad = (wmk.double() - wref).abs() > (2 ** -8) * wref.abs() + 1e-6
        H.record({"kind": "check", "item": f"{self.name}.route", "gap16": gap,
                  "w_maxerr": float((wmk.double() - wref).abs().max())})
        if bad.any():
            self.fail("weights", f"{int(bad.sum())} bad")

    def close(self, what, got, ref, rows=None):
        """|got - ref| <= 1e-4 rms(ref) + 1e-5 |ref| over ``rows`` (FP32 accumulation over K = 7168)."""
        got, ref = got.double(), ref.double()
        sl = slice(None) if rows is None else rows
        tol = 1e-4 * float(ref.pow(2).mean().sqrt()) + 1e-5 * ref[sl].abs()
        err = (got[sl] - ref[sl]).abs()
        nbad = int((err > tol).sum())
        H.record({"kind": "check", "item": f"{self.name}.{what}", "max_abs": float(err.max()),
                  "max_rel_rms": float(err.max() / ref.pow(2).mean().sqrt()), "violations": nbad})
        if nbad or not torch.isfinite(got[sl]).all():
            self.fail(what, f"{nbad} cells; max {float(err.max()):.3e}")


class Rig:
    def __init__(self, P, group, g, world, device, deadline_ns, arms, gbatches=(8,), front_rows=64):
        CH, TR, ER, host, OP, FR, PM = P
        self.g, self.world, self.device, self.group = g, world, device, group
        self.host, self.stream, self.TR = host, host.stream, TR
        self.logical = M.make_logical16(R.E_ALL, gen(1600 + g, device))
        self.experts = ER.PersistentExperts(*M.host_layout16(*self.logical), beta=R.BETA, linear_beta=R.LIN)
        from cutlass import Int32, Int64

        self.I32 = Int32
        self.deadline_ns = deadline_ns
        if "small" in arms:
            self.box = CH.SmallBatchBox(group, device, TP, g)
            tb = self.tail_box = TR.MoeTailBox(group, device, TP, g)
            self.scratch = [torch.empty_like(tb.rs.buffer).view(torch.int32).fill_(SENT).view(torch.bfloat16)
                            for _ in range(TP - world)]
            ptrs = list(tb.rs.handle.buffer_ptrs) + [s.data_ptr() for s in self.scratch]
            self.peers = torch.tensor(ptrs, dtype=torch.int64, device=device)
            tb.tail = (host.to_cute(self.peers, 16), TR.flat(tb.out.buffer), Int64(tb.latent.mc), Int64(tb.out.mc))
            self.consumer = B.ring_consumer(world * SHARD // 2, device.index)
            self.ids_snap = {b: torch.empty(b * 16, dtype=torch.int32, device=device) for b in BATCHES}
            op = SimpleNamespace(rank=g, contract=SimpleNamespace(tp_size=TP, rms_eps=EPS))
            self.layers = []
            for layer, ordinal in LAYERS:
                m = Model(g, device, layer)
                tail = TR.PreparedTail(op, m.gamma, m.up, BATCHES, (tb, ordinal, len(LAYERS)), deadline_ns=deadline_ns)
                chain = CH.SmallBatchChain(self.box, m.moe(), self.experts, tail, layer, R.BETA, R.LIN, None,
                                           deadline_ns=deadline_ns)
                parity = (layer % 2, int(tail.tail_weights.parity))
                self.layers.append(SimpleNamespace(m=m, chain=chain, parity=parity, layer=layer))
        if "generic" in arms:
            self.FR, self.gbatches = FR, gbatches
            self.front_box = OP.OprojMailbox(group, device, TP, g, (1, front_rows, 2 * LAT + 2 * 896))
            self.state = PM.MoEScratch(TP, device, gbatches)
            # Plan canary: the plan the routing kernel may write is the first PLAN_BYTES of a buffer twice that
            # size; the rest holds CANARY and must stay so (a 128-row plan write would overrun the one 64-row block).
            pb = FR.PLAN_BYTES
            self.plan_canary = torch.full((2 * pb,), CANARY, dtype=torch.uint8, device=device)
            self.generic_layers = []
            for layer in (11, 12):
                m = Model(g, device, layer)
                router = m.router[g * LOGITS:(g + 1) * LOGITS]
                down = m.down[g * LROWS:(g + 1) * LROWS]
                packed = torch.cat((router, router.new_zeros((-LOGITS % 16, HID)), down)).contiguous()
                front = FR.SharedFront(m.moe(), packed, self.front_box, self.state.front, gbatches, R.BETA, R.LIN,
                                       plan=self.plan_canary[:pb], counters=self.state.shared, q=8,
                                       deadline_ns=deadline_ns)
                self.generic_layers.append(SimpleNamespace(m=m, front=front, packed=packed, layer=layer))

    def gather(self, t):
        out = [torch.empty_like(t) for _ in range(self.world)]
        dist.all_gather(out, t.contiguous(), group=self.group)
        return out

    # ------------------------------------------------------------------ small arm
    def inputs(self, call, lay, b):
        d, m, w = self.device, lay.m, self.world
        x = randn((b, HID), 1.0, 1000 + 10 * call + b, d)
        lat_ref = x.double() @ m.down.double().T  # [b, 3584]
        inj_lat = {q: randn((b, LAT), 0.5, 2000 + 100 * call + 10 * b + q, d) for q in range(w, TP)}
        inj_s = {q: randn((b, HID), 0.5, 3000 + 100 * call + 10 * b + q, d) for q in range(w, TP)}
        lat = M.canon(lat_ref.float()[:, w * LROWS:])
        p, t = lay.parity
        stage = SimpleNamespace(
            xmoe=canon_bf16(x), lat=lat, p=p, t=t, b=b,
            tl=torch.stack([canon_bf16(inj_lat[q]) for q in range(w, TP)]),
            rs=torch.stack([canon_bf16(inj_s[q][:, self.g * SHARD:(self.g + 1) * SHARD]) for q in range(w, TP)]),
            hint=torch.tensor([28 * b], dtype=torch.int32, device=d))
        return SimpleNamespace(x=x, lat_ref=lat_ref, inj_s=inj_s, lay=lay, call=call, b=b, stage=stage)

    def inject(self, st):
        box, tb, w, b = self.box, self.tail_box, self.world, st.b
        box.xmoe[st.p, 0, :b].copy_(st.xmoe)
        box.epoch[st.p * 32: st.p * 32 + 1].copy_(st.hint)
        box.h2.buffer[st.p, :b].view(torch.float32)[:, w * LROWS:].copy_(st.lat)
        tb.latent.buffer[st.t, w:TP, :b].copy_(st.tl)
        tb.rs.buffer[st.t, w:TP, :b].copy_(st.rs)

    def consume(self, saves, b):
        s = self.stream(self.device)
        for tok in range(b):
            ring = self.tail_box.out.buffer[0, tok, : self.world * SHARD].view(torch.int32)
            self.consumer(self.host.to_cute(ring, 16), self.host.to_cute(saves[tok], 16), s)

    def run(self, lay, saves, b, snapshot):
        ch = lay.chain
        if snapshot:  # SmallBatchChain.run with the ids copied out before the experts consume them
            (front, front_args), (routed, routed_args), tail = ch.codes[b]
            s = self.stream(self.device)
            front(*front_args, self.I32(ch.parity), self.I32(ch.slot), *ch.scalars, s)
            self.ids_snap[b].copy_(self.box.scratch[b][1])
            routed(*routed_args, *ch.experts, s)
            ch.tail_weights.run(ch.tail_box, self.box.g, self.box.w, self.box.s, tail, b)
        else:
            ch.run(self.tail_box.out.buffer[0, :b], lambda o: o)
        self.consume(saves, b)

    def check_small(self, src, saves, fails):
        g, box, tb, w, b = self.g, self.box, self.tail_box, self.world, src.b
        m, (p, t) = src.lay.m, src.lay.parity
        ids_all = self.ids_snap[b].view(b, 16)
        lat_all = box.h2.buffer[p, :b].view(torch.float32).clone()
        lat_gpus = self.gather(lat_all)
        hg_all = box.scratch[b][2].view(torch.float32).view(2, b, 768)[p]
        slots_all = tb.latent.buffer[t, :TP, :b].clone()  # [16, b, 3584]
        allslots = self.gather(slots_all[:w].contiguous())
        S_all = self.gather(box.s[t, :b].clone())
        saves_all = self.gather(torch.stack(saves))
        for tok in range(b):
            ck = Checker(f"moe_tp16.small.b{b}.l{src.lay.layer}.call{src.call}.t{tok}.g{g}", fails)
            x = src.x[tok]
            ck.route(m, x, ids_all[tok], box.w[t, tok, :16])
            lat = lat_all[tok]
            ck.close("latent", lat, src.lat_ref[tok], slice(0, w * LROWS))
            if not torch.equal(lat[w * LROWS:].view(torch.int32), src.stage.lat[tok].view(torch.int32)):
                ck.fail("latent_inj", "injected latent rows changed")
            if not all(torch.equal(o[tok].view(torch.int32), lat.view(torch.int32)) for o in lat_gpus):
                ck.fail("latent_multicast", "latent words differ across GPUs")
            hg = hg_all[tok]
            ck.close("hg", hg, m.gate_up.double() @ x.double())
            G = box.g[t, tok * 16:(tok + 1) * 16]
            ref_rows = M.expert_rows(self.logical, lat.view(1, LAT), ids_all[tok].view(1, 16))
            fro = R.rel_l2(G.view(1, 16, LAT), ref_rows)
            row = float(((G.double() - ref_rows[0]).norm(dim=-1) / ref_rows[0].norm(dim=-1)).max())
            H.record({"kind": "check", "item": f"{ck.name}.experts", "t1_fro": fro, "t1_row_max": row})
            if not (torch.isfinite(G.float()).all() and fro <= 1e-2 and row <= 3e-2):
                ck.fail("experts", f"T1 {fro:.3e} {row:.3e}")
            hs = R.situ(hg[:384].double(), hg[384:].double())
            s_ref = m.shared_down.double() @ bf(hs.float())
            S = box.s[t, tok].double()
            tol = 2 ** -7 * s_ref.abs() + 2 ** -8 * float(s_ref.pow(2).mean().sqrt())
            nbad = int(((S - s_ref).abs() > tol).sum())
            H.record({"kind": "check", "item": f"{ck.name}.shared", "max_abs": float((S - s_ref).abs().max()),
                      "violations": nbad})
            if nbad:
                ck.fail("shared", f"{nbad} cells")
            terms = box.w[t, tok, :16].double().view(16, 1) * G.double()
            part_ref = terms.sum(0)
            mine = tb.latent.buffer[t, g, tok].double()
            lim = ulp_bf16(torch.maximum(part_ref.abs(), terms.abs().max(0).values))
            nbad = int(((mine - part_ref).abs() > lim).sum())
            H.record({"kind": "check", "item": f"{ck.name}.partial", "violations": nbad})
            if nbad:
                ck.fail("partial", f"{nbad} cells beyond 1 ulp")
            for q in range(w):
                if not torch.equal(allslots[q][q, tok].view(torch.int16), slots_all[q, tok].view(torch.int16)):
                    ck.fail("multicast", f"latent partial slot {q} differs from rank {q}'s own copy")
            for q in range(w):
                want = canon_bf16(S_all[q][tok, g * SHARD:(g + 1) * SHARD])
                if not torch.equal(tb.rs.buffer[t, q, tok].view(torch.int16), want.view(torch.int16)):
                    ck.fail("rs", f"shard from rank {q} differs")
            for o in range(w, TP):
                got = self.scratch[o - w][t, g, tok]
                want = canon_bf16(box.s[t, tok, o * SHARD:(o + 1) * SHARD])
                if not torch.equal(got.view(torch.int16), want.view(torch.int16)):
                    ck.fail("rs_out", f"shard for owner {o} differs")
            Lsum = slots_all[:, tok].double().sum(0)
            inv = 1.0 / torch.sqrt((Lsum ** 2).mean() + EPS)
            xn = bf(bf(Lsum) * inv * m.gamma.double())
            y = bf(m.up.double() @ xn)
            S_full = [S_all[q][tok].double() for q in range(w)] + [src.inj_s[q][tok].double() for q in range(w, TP)]
            sh = bf(sum(S_full))
            out_ref = bf(y + sh)
            n = w * SHARD
            out = saves[tok].view(torch.bfloat16).double()
            lim = 2 * ulp_bf16(torch.maximum(torch.maximum(out_ref[:n].abs(), y[:n].abs()), sh[:n].abs()))
            err = (out - out_ref[:n]).abs()
            nbad = int((err > lim).sum())
            rec = {"kind": "check", "item": f"{ck.name}.output", "max_abs": float(err.max()),
                   "max_ulps": float((err / ulp_bf16(out_ref[:n].abs())).max()), "violations": nbad}
            H.record(rec)
            print("CHECK", json.dumps(rec), flush=True)
            if nbad or not torch.isfinite(out).all():
                ck.fail("output", f"{nbad} cells beyond 2 ulp; max {float(err.max()):.3e}")
            if not all(torch.equal(o[tok], saves[tok]) for o in saves_all):
                ck.fail("rank_bitwise", "output rows differ across GPUs")

    def small(self, fails):
        d, w = self.device, self.world
        n_words = w * SHARD // 2
        for b in BATCHES:
            srcs = [self.inputs(c, lay, b) for c, lay in enumerate(self.layers)]
            eager = []
            for src in srcs:
                saves = [torch.full((n_words,), SENT, dtype=torch.int32, device=d) for _ in range(b)]
                self.inject(src.stage)
                torch.cuda.synchronize()
                dist.barrier(group=self.group)
                self.run(src.lay, saves, b, snapshot=True)
                torch.cuda.synchronize()
                dist.barrier(group=self.group)
                self.check_small(src, saves, fails)
                eager.append([s.clone() for s in saves])
                torch.cuda.synchronize()
                dist.barrier(group=self.group)
            gsaves = [[torch.empty((n_words,), dtype=torch.int32, device=d) for _ in range(b)] for _ in srcs]

            def chain(srcs=srcs, gsaves=gsaves, b=b):
                for src, saves in zip(srcs, gsaves):
                    self.inject(src.stage)
                    self.run(src.lay, saves, b, snapshot=False)

            graph, stream = torch.cuda.CUDAGraph(), torch.cuda.Stream()
            stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.graph(graph, stream=stream):
                chain()
            torch.cuda.synchronize()
            for rep in range(2):
                for saves in gsaves:
                    for s in saves:
                        s.fill_(SENT)
                torch.cuda.synchronize()
                dist.barrier(group=self.group)
                graph.replay()
                torch.cuda.synchronize()
                dist.barrier(group=self.group)
                for c in range(len(srcs)):
                    same = all(torch.equal(a, e) for a, e in zip(gsaves[c], eager[c]))
                    H.record({"kind": "check", "item": f"moe_tp16.small.b{b}.graph{rep}.call{c}.g{self.g}",
                              "bitwise_vs_eager": same})
                    if not same:
                        fails.append(f"small.b{b}.graph{rep}.call{c}.g{self.g}: differs from eager")
                        print("FAIL graph", b, rep, c, flush=True)
            del graph

    # ------------------------------------------------------------------ generic arm
    def generic_call(self, lay, call, fails, b=8):
        d, w, m = self.device, self.world, lay.m
        x = randn((b, HID), 1.0, 5000 + 10 * call + lay.layer, d)
        logits_ref = x.double() @ m.router.double().T  # [b, 896]
        lat_ref = x.double() @ m.down.double().T  # [b, 3584]
        row = self.front_box.buffer[0, :b].view(torch.float32)  # [b, 4480]: latent 3584 | logits 896
        row[:, w * LROWS: LAT].copy_(M.canon(lat_ref.float()[:, w * LROWS:]))
        row[:, LAT + w * LOGITS:].copy_(M.canon(logits_ref.float()[:, w * LOGITS:]))
        torch.cuda.synchronize()
        dist.barrier(group=self.group)
        ids, weights, latent = lay.front.route(canon_bf16(x))
        torch.cuda.synchronize()
        dist.barrier(group=self.group)
        for tok in range(b):
            ck = Checker(f"moe_tp16.generic.b{b}.l{lay.layer}.call{call}.t{tok}.g{self.g}", fails)
            ck.route(m, x[tok], ids[tok], weights[tok])
            ck.close("latent", latent[tok], lat_ref[tok], slice(0, w * LROWS))
            inj = M.canon(lat_ref.float()[tok, w * LROWS:])
            if not torch.equal(latent[tok, w * LROWS:].view(torch.int32), inj.view(torch.int32)):
                ck.fail("latent_inj", "injected latent rows changed")
        if latent.dtype != torch.float32:
            fails.append(f"generic: latent dtype {latent.dtype}")
        # Plan canary: nothing past the one 64-row plan block may change (any batch, any call).
        pb = self.FR.PLAN_BYTES
        spill = int((self.plan_canary[pb:] != CANARY).sum())
        H.record({"kind": "check", "item": f"moe_tp16.generic.b{b}.l{lay.layer}.call{call}.g{self.g}.plan_canary",
                  "changed_bytes": spill, "planned": bool(lay.front.planned(b))})
        if spill:
            fails.append(f"generic b{b} l{lay.layer} call{call}: {spill} plan canary bytes changed (plan overran its 64-row block)")
            print("FAIL plan_canary", b, lay.layer, call, spill, flush=True)
        if b > self.FR.PLAN_ROWS:
            self.generic_shared(lay, call, fails, b, x)

    def generic_shared(self, lay, call, fails, b, x):
        """Above 64 rows the shared expert is the split-K chain (FUSED stops at 64): gate/up (FP32, the 65-128 tactic),
        SiTU, down (BF16), then ``tail.push``. A stand-in tail keeps what is pushed; checks hg and S as the small arm."""
        m, pushed = lay.m, []
        tail = SimpleNamespace(push=lambda t: pushed.append(t))
        rows_s, side = lay.front.shared_expert(canon_bf16(x), tail)
        torch.cuda.synchronize()
        if side is not None or len(pushed) != 1 or pushed[0].data_ptr() != rows_s.data_ptr():
            fails.append(f"generic b{b}: shared expert did not take the split-K chain (side={side}, pushes={len(pushed)})")
            return
        hg_all = lay.front.plans[b][0][1]  # views[1]: FP32 [gate 384 | up 384] per token
        h_all = lay.front.plans[b][0][4]  # views[4]: the BF16 h the down GEMM read
        w_abs = m.shared_down.double().abs()
        # Diagnostic only (no pass / fail): splits the registered S misses between SiTU and the down GEMM.
        #   h_flips: elements of the kernel's h that differ from the reference's bf16(SiTU(hg)) (the kernel's SiTU is the
        #            stock fast-math form, tanh.approx / ex2.approx; the reference's is FP64);
        #   k:       S cells beyond the down GEMM's own bound against FP64 down @ the kernel's h: one BF16 output rounding
        #            (2^-8 |ref|) plus FP32 accumulation over K = 384 (2 * 384 * 2^-24 * sum_j |w_ij h_j|);
        #   k_refh:  the same bound against FP64 down @ the reference's h (shows the bound sees an h-level difference).
        diag = {"registered": 0, "h_flips": 0, "k": 0, "k_refh": 0}
        for tok in range(b):
            ck = Checker(f"moe_tp16.generic.b{b}.l{lay.layer}.call{call}.t{tok}.g{self.g}", fails)
            hg = hg_all[tok]
            ck.close("hg", hg, m.gate_up.double() @ x[tok].double())
            hs = R.situ(hg[:384].double(), hg[384:].double())
            s_ref = m.shared_down.double() @ bf(hs.float())
            S = rows_s[tok].double()
            tol = 2 ** -7 * s_ref.abs() + 2 ** -8 * float(s_ref.pow(2).mean().sqrt())
            nbad = int(((S - s_ref).abs() > tol).sum())
            H.record({"kind": "check", "item": f"{ck.name}.shared", "max_abs": float((S - s_ref).abs().max()),
                      "violations": nbad})
            if nbad or not torch.isfinite(S).all():
                ck.fail("shared", f"{nbad} cells")
            hk = h_all[tok].double()
            flips = int((hk != bf(hs.float())).sum())
            ref_k = m.shared_down.double() @ hk
            lim_k = 2 ** -8 * ref_k.abs() + 2 * 384 * 2 ** -24 * (w_abs @ hk.abs())
            k = int(((S - ref_k).abs() > lim_k).sum())
            k_refh = int(((S - s_ref).abs() > 2 ** -8 * s_ref.abs() + 2 * 384 * 2 ** -24 * (w_abs @ bf(hs.float()).abs())).sum())
            for key, v in (("registered", nbad), ("h_flips", flips), ("k", k), ("k_refh", k_refh)):
                diag[key] += v
            if nbad or k:
                H.record({"kind": "diag", "item": f"{ck.name}.shared_refs", "registered": nbad, "h_flips": flips, "k": k,
                          "k_refh": k_refh, "max_abs_k": float((S - ref_k).abs().max())})
        name = f"moe_tp16.generic.b{b}.l{lay.layer}.call{call}.g{self.g}.shared_refs"
        H.record({"kind": "diag", "item": name, **diag})
        print("DIAG", name, json.dumps(diag), flush=True)

    def generic(self, fails):
        for b in self.gbatches:
            for call in range(3):
                for lay in self.generic_layers:
                    self.generic_call(lay, call, fails, b)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--deadline-s", type=float, default=2.0)
    ap.add_argument("--arms", default="small,generic")
    ap.add_argument("--generic-batches", default="8", help="generic arm batches (rows), e.g. 8,33,64,128")
    ap.add_argument("--front-rows", type=int, default=64, help="front box rows (a boot serving 128 rows: 128)")
    args = ap.parse_args()
    arms = tuple(args.arms.split(","))
    gbatches = tuple(int(v) for v in args.generic_batches.split(","))
    local = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local)
    device = torch.device("cuda", local)
    dist.init_process_group("nccl", device_id=device)
    g, world = dist.get_rank(), dist.get_world_size()
    tag = os.environ.get("K31_MUTATION") or "base"
    os.makedirs(args.out, exist_ok=True)
    os.environ["K31_RESULTS"] = os.path.join(args.out, f"moe_tp16_{tag}_g{g}.jsonl")
    H.load_plugin(MUTATIONS)
    from sgl_nvidia_mega.kimi_k31.k3.cutedsl.common import host
    from sgl_nvidia_mega.kimi_k31.k3.cutedsl.moe import experts_runtime as ER
    from sgl_nvidia_mega.kimi_k31.k3.cutedsl.moe import front as FR
    from sgl_nvidia_mega.kimi_k31.k3.cutedsl.moe import prepared as PM
    from sgl_nvidia_mega.kimi_k31.k3.cutedsl.moe import tail_runtime as TR
    from sgl_nvidia_mega.kimi_k31.k3.cutedsl.moe.small_batch import chain as CH
    from sgl_nvidia_mega.kimi_k31.k3.cutedsl.oproj import prepared as OP

    H.record({"kind": "env", **H.device_facts(), "world": world, "arms": list(arms), "generic_batches": list(gbatches),
              "front_rows": args.front_rows})
    fails = []
    t0 = time.time()
    try:
        rig = Rig((CH, TR, ER, host, OP, FR, PM), dist.group.WORLD, g, world, device, int(args.deadline_s * 1e9),
                  arms, gbatches, args.front_rows)
        if "small" in arms:
            rig.small(fails)
        if "generic" in arms:
            rig.generic(fails)
    except Exception as e:  # noqa: BLE001
        fails.append(f"g{g}: {type(e).__name__}: {e}")
        traceback.print_exc()
    with open(os.path.join(args.out, f"moe_tp16_{tag}_g{g}.failures.json"), "w") as f:
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
