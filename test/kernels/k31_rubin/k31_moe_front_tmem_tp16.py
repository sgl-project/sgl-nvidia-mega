# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""K31MK_MOE_FRONT_TMEM : the TP16 MoE front + shared gate/up as one ``MoeFrontTmem64`` grid at the verify
batches 16 / 24 / 32 rows (q = 8), against the served split form and an FP64 reference, over W = 4 GPUs with real
symm_mem multicast (the generic arm of k31_moe_tp16.py, same seeded weights, rank injection and checks).

Two ``SharedFront`` per layer (layers 11, 12), one shared front mailbox and scratch: S = served (env unset: split
batches run the Fp32FrontGemm publish; gate/up is the separate SharedUmma gate kernel, not run here), T =
K31MK_MOE_FRONT_TMEM=16-32:1 (set only around construction). GPU g is TP rank g of 16; ranks 4-15's FP32 latent and
router-logit words are injected into every GPU's box row before each call (from FP64).

Per (layer, call, batch, arm), two phases on identical inputs:
  A  publish only (S: the front GEMM; T: the MoeFrontTmem64 grid), then a barrier and a snapshot of the box rows of
     ranks 0-3 (FP32 latent + FP32 logit words, multicast from GPUs 0-3), then the routing kernel (consumes + re-arms);
  B  the integrated ``SharedFront.route`` (T: ``front_tmem`` branch), ids / weights / latent from its outputs.
Checks (tolerances written before any run):
  ids      top-16 set == FP64 reference (sigmoid(logits) + bias) unless the 16th/17th gap < 1e-4 (k31_moe_tp16 rule).
  weights  BF16 routing weights vs score / sum(scores): |d| <= 2^-8 |ref| + 1e-6.
  latent   FP32 latent of ranks 0-3 (phase A snapshot and phase B output) vs FP64: |d| <= 1e-4 rms + 1e-5 |ref|
           (k31_moe_tp16's bound; a BF16-rounded latent is off by up to 2^-9 |ref|); injected rows bit-exact.
  logits   phase A FP32 logits of ranks 0-3 vs FP64: same bound as latent.
  vs-S     pooled per batch over layers / calls / tokens: rms(T - ref) <= 1.5 rms(S - ref) and max |T - ref| <= 2.0
           max |S - ref|, for latent and for logits (T's FP32 sum order must not be materially worse than the served).
  h        T only: BF16 h vs FP64 bf16(SiTU(bf16(g), bf16(u))), g / u = x @ gate_up.T halves: |d| <= 2^-6 |ref| +
           2^-8 rms(ref) (fast-math SiTU + two BF16 roundings); h is NaN-filled before every call (unwritten = fail).
  engaged  T's phase A snapshot differs from S's in at least one latent or logit word at every batch (S and T sum in
           different FP32 orders; equal everywhere = the knob did not engage), and T's front holds every batch.
  determ   call 2 repeats call 0's inputs: T's phase A snapshot, h and phase B ids / weights / latent equal call 0's
           bit for bit (S likewise).
  graph    T's ``route`` captured in one CUDA graph per (layer, batch), replayed on call 0's inputs (injection outside):
           ids / weights / latent / h equal the eager call 0 bit for bit.
Mutations (``MUTATIONS``, each must fail): ftm_bf16_latent (latent published BF16-rounded), ftm_up_rows (the up
half loads the gate rows), ftm_half_cols (both halves of an owner's 8 h columns land on its first 4), ftm_inert (the knob is parsed
but no batch takes the TMEM front).

--wide (wide decode M1(a); checks written before any run, MOE-wideM1b.1): batches 72 / 96 / 128 over a 128-row front
box; T = K31MK_MOE_FRONT_TMEM=65-128:1 + K31MK_MOE_WIDE_FOLD=65-128:1 (K31MK_MOE_WIDE=65-128:1 for the whole process, so the
scratch holds the wide engine's plan, passed as ``wide_plan``); S = the served wide form (Fp32FrontGemm 65-128 geometry, no
routed plan). Every check above runs, plus:
  halves   T only: the 128-row grid's phase A words (ranks 0-3 latent + logits) and h of tokens 64 .. b-1 equal, bit for bit,
           a MoeFrontTmem64 call on x[64:b] alone (same K split, same owner order: a token's sums do not depend on its tile);
  plan     T only, phase B: the wide plan bytes the routing kernel wrote (both buffers prefilled 0xA5, ceil(b / 64) blocks)
           equal WidePlanKernel's (``compile_wide_plan``) from phase B's returned latent, bit for bit.
  unpublished  (added MOE-wideM1b.2, after FILL_moe_wftm_r1's ftm_one_tile mutant ended in an illegal-address crash instead of a
           check): phase A's snapshot of ranks 0-3 holds no armed sentinel word; otherwise FAIL and the call stops before
           the routing kernel (which would poll the missing rows past its deadline).
Mutations for --wide (each must fail): ftm_one_tile (the FTM grid covers one 64-token tile only), fold_one_block (the
routing kernel compiled for one plan block at fold rows), plus the four ftm_* ones above.

Usage (one node, 4 GPUs): python3 -m torch.distributed.run --nproc-per-node 4 k31_moe_front_tmem_tp16.py --out DIR [--quick] [--wide]
(--quick, for mutation runs: layer 11, call 0, no graph; every check above except determ / graph still runs.)
Exit status 0 only when every check on every rank passed (``TP_RESULT ... failures=0``).
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
import _ref_moe as R  # noqa: E402
import _ref_moe16 as M  # noqa: E402
from k31_moe_tp16 import Checker, Model, canon_bf16, randn  # noqa: E402

TP, HID, LAT = 16, 7168, 3584
LROWS, LOGITS = LAT // TP, 896 // TP
BATCHES = (16, 24, 32)
ENV = "K31MK_MOE_FRONT_TMEM"
SPEC = "16-32:1"
BOX_ROWS = 64
FOLD = None  # --wide: K31MK_MOE_WIDE_FOLD for arm T

FT = "kimi_k31/k3/cutedsl/moe/front_tmem.py"
FR_PATH = "kimi_k31/k3/cutedsl/moe/front.py"
SK = "kimi_k31/k3/cutedsl/gemm/splitk_tmem64.py"
MUTATIONS = {
    "ftm_bf16_latent": (FT, "                quad = tuple(positive_zero32(values[4 * v + i]) for i in range(4))\n"
                            "                offset = token * pitch + 2 * (base",
                        "                quad = tuple(positive_zero32(Float32(BFloat16(values[4 * v + i]))) for i in range(4))\n"
                        "                offset = token * pitch + 2 * (base"),
    "ftm_up_rows": (FT, "_tma_half(dst + 8192, barrier, self.gate_map, k0, self.hidden + gate)",
                    "_tma_half(dst + 8192, barrier, self.gate_map, k0, gate)"),
    "ftm_half_cols": (FT, "column = (m - FRONT_TILES) * 64 + 8 * owner + 4 * part",
                      "column = (m - FRONT_TILES) * 64 + 8 * owner"),
    "ftm_one_tile": (FT, "grid=(self.tiles, cute.ceil_div(x.layout.shape[0], 64) * self.split_k, 1),",
                     "grid=(self.tiles, self.split_k, 1),"),
    "fold_one_block": (FR_PATH, "blocks = -(-b // 64) if b in fold else 1", "blocks = 1"),
    "ftm_inert": (FR_PATH, "tmem = front_tmem_table(tp, n, packed.device) & set(batches)",
                  "tmem = front_tmem_table(tp, n, packed.device) & set()"),
    # FTM-P (K31MK_MOE_FRONT_TMEM_PROMOTE > 0 only; inert without it): later groups' accumulators never added /
    # an accumulator not restarted at its group's first MMA (adds onto its previous group or uninitialised TMEM)
    "ftm_promote_drop": (SK, "values = tuple(values[i] + loaded[i] for i in range(self.n))",
                         "values = tuple(values[i] for i in range(self.n))"),
    "ftm_promote_noreset": (SK, "first_k, acc = k % self.promote == 0, base + (group % 2) * acc_columns",
                            "first_k, acc = k == 0, base + (group % 2) * acc_columns"),
}


def stats(err, ref):
    return float(err.pow(2).mean().sqrt()), float(err.abs().max()), float(ref.pow(2).mean().sqrt())


class Rig:
    def __init__(self, P, group, g, world, device, deadline_ns, layers=(11, 12)):
        host, OP, FR, PM = P
        self.g, self.world, self.device, self.group, self.host = g, world, device, group, host
        self.front_box = OP.OprojMailbox(group, device, TP, g, (1, BOX_ROWS, 2 * LAT + 2 * 896))
        self.state = PM.MoEScratch(TP, device, BATCHES)
        self.layers = []
        for layer in layers:
            m = Model(g, device, layer)
            router = m.router[g * LOGITS:(g + 1) * LOGITS]
            down = m.down[g * LROWS:(g + 1) * LROWS]
            packed = torch.cat((router, router.new_zeros((-LOGITS % 16, HID)), down)).contiguous()
            fronts = {}
            for arm in ("S", "T"):
                old = os.environ.pop(ENV, None)
                if arm == "T":
                    os.environ[ENV] = SPEC
                    if FOLD:
                        os.environ["K31MK_MOE_WIDE_FOLD"] = FOLD
                wide = {"wide_plan": self.state.experts["rubin_wide"]["plan"]} if FOLD else {}
                try:
                    fronts[arm] = FR.SharedFront(m.moe(), packed, self.front_box, self.state.front, BATCHES, R.BETA,
                                                 R.LIN, plan=self.state.plan, counters=self.state.shared, q=8,
                                                 deadline_ns=deadline_ns, **wide)
                finally:
                    os.environ.pop(ENV, None)
                    os.environ.pop("K31MK_MOE_WIDE_FOLD", None)
                    if old is not None:
                        os.environ[ENV] = old
            self.layers.append(dict(m=m, fronts=fronts, packed=packed, layer=layer))
        # pooled error sums per (batch, arm, quantity): list of (err, ref) tensors (phase A, own ranks)
        self.pool = {}
        self.snaps = {}
        self.outs = {}

    def inputs(self, lay, call, b):
        seed_call = 0 if call == 2 else call  # call 2 repeats call 0 (determinism)
        x = randn((b, HID), 1.0, 7000 + 100 * seed_call + 10 * lay["layer"] + b, self.device)
        m = lay["m"]
        return x, x.double() @ m.router.double().T, x.double() @ m.down.double().T

    def inject(self, b, logits_ref, lat_ref):
        w = self.world
        row = self.front_box.buffer[0, :b].view(torch.float32)  # [b, 4480]: latent 3584 | logits 896
        row[:, w * LROWS: LAT].copy_(M.canon(lat_ref.float()[:, w * LROWS:]))
        row[:, LAT + w * LOGITS:].copy_(M.canon(logits_ref.float()[:, w * LOGITS:]))
        torch.cuda.synchronize()
        dist.barrier(group=self.group)

    def phase_a(self, front, arm, x, b):
        """Publish only, snapshot ranks 0-3's box words, then the routing kernel (consumes and re-arms)."""
        views, publish, route, _, args, extra, _ = front.plans[b]
        s = self.host.stream(self.device)
        if arm == "T":
            tmem, maps = front.front_tmem[b]
            work = (front.tmem_views, FR_matrix_view(x), args[7], *front.box)
            tmem(*work, *front.scalars[1:], *maps, s)
        else:
            publish(self.host.to_cute(x), front.packed, *front.box, args[0], s)
        torch.cuda.synchronize()
        dist.barrier(group=self.group)
        w = self.world
        row = self.front_box.buffer[0, :b].view(torch.float32)
        snap = (row[:, : w * LROWS].clone(), row[:, LAT: LAT + w * LOGITS].clone())
        # Words of ranks 0-3 still holding the armed sentinel = rows the publish never wrote (e.g. a token tile dropped):
        # the routing kernel would poll them past its deadline, so report them as a check and skip the route (MOE-wideM1b.2).
        armed = torch.tensor([int(sum(int((t.view(torch.int32) == -(2 ** 31)).sum()) for t in snap))], device=self.device)
        dist.all_reduce(armed, op=dist.ReduceOp.MAX, group=self.group)
        if int(armed):
            self.front_box.buffer[0, :b].view(torch.int32).fill_(-(2 ** 31))  # re-arm by hand (no route consumes them)
            torch.cuda.synchronize()
            dist.barrier(group=self.group)
            return snap, int(armed)
        route(*args, *extra, s)
        torch.cuda.synchronize()
        dist.barrier(group=self.group)
        return snap, 0

    def h_view(self, b):
        return self.state.front[4][:b]

    def halves(self, front, x, b, lat_a, log_a, h_a, ck):
        """--wide, T: tokens 64 .. b-1 alone through the same MoeFrontTmem64 code; their words and h must equal the
        b-token call's. The box rows it publishes are then re-armed by hand (no routing kernel consumes them)."""
        n, w = b - 64, self.world
        args = front.plans[b][4]
        tmem, maps = front.front_tmem[b]
        h_half = self.state.front[4][:n]
        h_half.fill_(float("nan"))
        s = self.host.stream(self.device)
        tmem(front.tmem_views, FR_matrix_view(x[64:b].contiguous()), self.host.to_cute(h_half), *front.box,
             *front.scalars[1:], *maps, s)
        torch.cuda.synchronize()
        dist.barrier(group=self.group)
        row = self.front_box.buffer[0, :n].view(torch.float32)
        same = (torch.equal(row[:, : w * LROWS].view(torch.int32), lat_a[64:b].view(torch.int32))
                and torch.equal(row[:, LAT: LAT + w * LOGITS].view(torch.int32), log_a[64:b].view(torch.int32)))
        same_h = torch.equal(h_half.view(torch.int16), h_a[64:b].view(torch.int16))
        H.record({"kind": "check", "item": f"{ck.name}.halves", "words_equal": same, "h_equal": same_h})
        if not same:
            ck.fail("halves", "tokens 64.. published differently by a separate tile call")
        if not same_h:
            ck.fail("halves_h", "h of tokens 64.. differs from a separate tile call")
        self.front_box.buffer[0, :n].view(torch.int32).fill_(-(2 ** 31))  # the armed sentinel word
        torch.cuda.synchronize()
        dist.barrier(group=self.group)

    def plan_check(self, front, b, latent, ck):
        """--wide, T, phase B: the routed plan blocks vs WidePlanKernel from the same latent (both prefilled 0xA5)."""
        from sgl_nvidia_mega.kimi_k31.k3.cutedsl.moe import experts_runtime as ER
        from sgl_nvidia_mega.kimi_k31.k3.cutedsl.moe.experts import PLAN_BYTES

        size = -(-b // 64) * PLAN_BYTES
        ref = torch.full((size,), 0xA5, dtype=torch.uint8, device=self.device)
        ER.compile_wide_plan(b, self.device.index)(self.host.to_cute(latent, 16), self.host.to_cute(ref),
                                                   self.host.stream(self.device))
        torch.cuda.synchronize()
        got = self.state.experts["rubin_wide"]["plan"][:size]
        differ = int((got != ref).sum())
        H.record({"kind": "check", "item": f"{ck.name}.plan", "plan_bytes_differ": differ, "plan_bytes": size})
        if differ:
            ck.fail("plan", f"{differ} routed plan bytes differ from WidePlanKernel")

    def call(self, lay, call, b, arm, fails):
        m, front = lay["m"], lay["fronts"][arm]
        x, logits_ref, lat_ref = self.inputs(lay, call, b)
        xc = canon_bf16(x)
        name = f"ftm.b{b}.l{lay['layer']}.call{call}.{arm}.g{self.g}"
        ck = Checker(name, fails)
        w = self.world
        # phase A
        self.h_view(b).fill_(float("nan"))
        self.inject(b, logits_ref, lat_ref)
        (lat_a, log_a), armed = self.phase_a(front, arm, xc, b)
        if armed:
            ck.fail("unpublished", f"{armed} box words of ranks 0-3 never published (phase A); phase B skipped")
            return
        ck.close("latentA", lat_a, lat_ref[:, : w * LROWS])
        ck.close("logitsA", log_a, logits_ref[:, : w * LOGITS])
        h_a = self.h_view(b).clone()
        if FOLD and arm == "T":
            self.halves(front, xc, b, lat_a, log_a, h_a, ck)
        if call < 2:
            for q, got, ref in (("latent", lat_a, lat_ref[:, : w * LROWS]), ("logits", log_a, logits_ref[:, : w * LOGITS])):
                self.pool.setdefault((b, arm, q), []).append((got.double() - ref).flatten())
                self.pool.setdefault((b, "ref", q), []).append(ref.flatten())
        if arm == "T":
            hg = x.double() @ m.gate_up.double().T  # [b, 768]
            g64, u64 = hg[:, :384].to(torch.bfloat16).double(), hg[:, 384:].to(torch.bfloat16).double()
            h_ref = R.situ(g64, u64).to(torch.bfloat16).double()
            got = h_a.double()
            tol = 2 ** -6 * h_ref.abs() + 2 ** -8 * float(h_ref.pow(2).mean().sqrt())
            err = (got - h_ref).abs()
            bad = int((~(err <= tol)).sum())  # NaN counts as bad
            H.record({"kind": "check", "item": f"{name}.h", "max_abs": float(err.nan_to_num(1e30).max()),
                      "violations": bad, "rms_ref": float(h_ref.pow(2).mean().sqrt())})
            if bad:
                ck.fail("h", f"{bad} cells")
        # phase B (integrated route)
        self.h_view(b).fill_(float("nan"))
        if FOLD and arm == "T":
            self.state.experts["rubin_wide"]["plan"].fill_(0xA5)
        self.inject(b, logits_ref, lat_ref)
        ids, weights, latent = front.route(xc)
        torch.cuda.synchronize()
        dist.barrier(group=self.group)
        if FOLD and arm == "T":
            self.plan_check(front, b, latent, ck)
        for tok in range(b):
            ck.route(m, x[tok], ids[tok], weights[tok])
        ck.close("latentB", latent, lat_ref, (slice(None), slice(0, w * LROWS)))
        inj = M.canon(lat_ref.float()[:, w * LROWS:])
        if not torch.equal(latent[:, w * LROWS:].contiguous().view(torch.int32), inj.view(torch.int32)):
            ck.fail("latent_inj", "injected latent rows changed")
        h_b = self.h_view(b).clone()
        if arm == "T" and not torch.equal(h_a.view(torch.int16), h_b.view(torch.int16)):
            ck.fail("h_AB", "h of phase B differs from phase A (same inputs)")
        out = dict(lat_a=lat_a, log_a=log_a, h=h_b, ids=ids.clone(), weights=weights.clone(), latent=latent.clone())
        self.outs[(lay["layer"], call, b, arm)] = out
        if call == 2:
            ref0 = self.outs[(lay["layer"], 0, b, arm)]
            for key in ("lat_a", "log_a", "ids", "weights", "latent") + (("h",) if arm == "T" else ()):
                a, c0 = out[key], ref0[key]
                if not torch.equal(a.contiguous().view(torch.uint8), c0.contiguous().view(torch.uint8)):
                    ck.fail("determ", f"{key} differs from call 0 on identical inputs")

    def graph(self, lay, b, fails):
        """T's route captured in one CUDA graph, replayed on call 0's inputs (injection outside the graph)."""
        m, front = lay["m"], lay["fronts"]["T"]
        name = f"ftm.graph.b{b}.l{lay['layer']}.g{self.g}"
        ck = Checker(name, fails)
        x, logits_ref, lat_ref = self.inputs(lay, 0, b)
        xs = canon_bf16(x).clone()
        # warm (eager) once on these inputs so the capture sees compiled code; it consumes one injection
        self.inject(b, logits_ref, lat_ref)
        front.route(xs)
        torch.cuda.synchronize()
        dist.barrier(group=self.group)
        graph = torch.cuda.CUDAGraph()
        s = torch.cuda.Stream(device=self.device)
        with torch.cuda.stream(s):
            with torch.cuda.graph(graph, stream=s):
                ids, weights, latent = front.route(xs)
        torch.cuda.synchronize()
        dist.barrier(group=self.group)
        self.h_view(b).fill_(float("nan"))
        self.inject(b, logits_ref, lat_ref)
        graph.replay()
        torch.cuda.synchronize()
        dist.barrier(group=self.group)
        ref0 = self.outs[(lay["layer"], 0, b, "T")]
        for key, got in (("ids", ids), ("weights", weights), ("latent", latent), ("h", self.h_view(b))):
            if not torch.equal(got.contiguous().view(torch.uint8), ref0[key].contiguous().view(torch.uint8)):
                ck.fail("graph", f"{key} differs from the eager call 0")
        H.record({"kind": "check", "item": f"{name}", "ok": True})

    def judge(self, fails):
        for b in BATCHES:
            for q in ("latent", "logits"):
                ref = torch.cat(self.pool[(b, "ref", q)])
                es, et = (torch.cat(self.pool[(b, a, q)]) for a in ("S", "T"))
                (rs, ms, rr), (rt, mt, _) = stats(es, ref), stats(et, ref)
                H.record({"kind": "check", "item": f"ftm.vsS.b{b}.{q}.g{self.g}", "rms_S": rs, "rms_T": rt,
                          "max_S": ms, "max_T": mt, "rms_ref": rr})
                print(f"vsS b{b} {q} g{self.g}: rms S {rs:.3e} T {rt:.3e} | max S {ms:.3e} T {mt:.3e} | ref rms {rr:.3e}",
                      flush=True)
                if not (rt <= 1.5 * rs and mt <= 2.0 * ms):
                    fails.append(f"ftm.vsS.b{b}.{q}.g{self.g}: T rms {rt:.3e} / max {mt:.3e} vs S {rs:.3e} / {ms:.3e}")
            differ = False
            for lay in self.layers:
                for call in (0, 1):
                    if (lay["layer"], call, b, "S") not in self.outs:
                        continue
                    s_out, t_out = self.outs[(lay["layer"], call, b, "S")], self.outs[(lay["layer"], call, b, "T")]
                    for key in ("lat_a", "log_a"):
                        if not torch.equal(s_out[key].view(torch.int32), t_out[key].view(torch.int32)):
                            differ = True
            H.record({"kind": "check", "item": f"ftm.engaged.b{b}.g{self.g}", "differs": differ})
            if not differ:
                fails.append(f"ftm.engaged.b{b}.g{self.g}: T's published words equal S's everywhere (knob not engaged)")
        for lay in self.layers:
            held = sorted(lay["fronts"]["T"].front_tmem)
            if held != list(BATCHES) or lay["fronts"]["S"].front_tmem:
                fails.append(f"ftm.engaged.l{lay['layer']}.g{self.g}: T front_tmem {held}, S {sorted(lay['fronts']['S'].front_tmem)}")


FR_matrix_view = None


def main():
    global FR_matrix_view, BATCHES, SPEC, BOX_ROWS, FOLD
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--deadline-s", type=float, default=2.0)
    ap.add_argument("--no-graph", action="store_true")
    ap.add_argument("--quick", action="store_true", help="mutation runs: layer 11, call 0 only, no graph")
    ap.add_argument("--wide", action="store_true", help=": 72 / 96 / 128 rows, 128-row box, fold")
    args = ap.parse_args()
    if args.wide:
        BATCHES, SPEC, BOX_ROWS, FOLD = (72, 96, 128), "65-128:1", 128, "65-128:1"
        os.environ["K31MK_MOE_WIDE"] = "65-128:1"  # before MoEScratch: the scratch then holds the wide engine's plan
    local = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local)
    device = torch.device("cuda", local)
    dist.init_process_group("nccl", device_id=device)
    g, world = dist.get_rank(), dist.get_world_size()
    tag = os.environ.get("K31_MUTATION") or "base"
    os.makedirs(args.out, exist_ok=True)
    tag += "_wide" if args.wide else ""
    os.environ["K31_RESULTS"] = os.path.join(args.out, f"ftm_{tag}_g{g}.jsonl")
    H.load_plugin(MUTATIONS)
    from sgl_nvidia_mega.kimi_k31.k3.cutedsl.common import host
    from sgl_nvidia_mega.kimi_k31.k3.cutedsl.gemm.projection import matrix_view
    from sgl_nvidia_mega.kimi_k31.k3.cutedsl.moe import front as FR
    from sgl_nvidia_mega.kimi_k31.k3.cutedsl.moe import prepared as PM
    from sgl_nvidia_mega.kimi_k31.k3.cutedsl.oproj import prepared as OP

    FR_matrix_view = matrix_view
    H.record({"kind": "env", **H.device_facts(), "world": world, "mutation": tag})
    fails = []
    t0 = time.time()
    try:
        layers = (11,) if args.quick else (11, 12)
        rig = Rig((host, OP, FR, PM), dist.group.WORLD, g, world, device, int(args.deadline_s * 1e9), layers)
        print(f"g{g} rig built {time.time() - t0:.0f}s", flush=True)
        for call in range(1 if args.quick else 3):
            for lay in rig.layers:
                for b in BATCHES:
                    for arm in ("S", "T"):
                        rig.call(lay, call, b, arm, fails)
        rig.judge(fails)
        if not (args.no_graph or args.quick):
            for lay in rig.layers:
                for b in BATCHES:
                    rig.graph(lay, b, fails)
        print(f"g{g} checks done {time.time() - t0:.0f}s", flush=True)
    except Exception as e:  # noqa: BLE001
        fails.append(f"g{g}: {type(e).__name__}: {e}")
        traceback.print_exc()
    with open(os.path.join(args.out, f"ftm_{tag}_g{g}.failures.json"), "w") as f:
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
