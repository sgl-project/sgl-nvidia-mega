# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""K3.1 MK KDA-layer sub-chain through the TP16 fused o-projection (the KDA part is
k31_kda_subchain.py): TP16 per-rank shapes (6 heads, projection 3216 x 7168, o_proj [7168, 768]), the c=1 verify rows
(Breq 1 x Q8 = 8), L layers with distinct weights and states (cold in L2), PDL-chained on one stream as the verify
executor launches them, over W = 4 GPUs (GPU g = TP rank g; real symm_mem multicast between them):

  [inject ranks 4-15's partials: memcpy] -> tail stand-in -> ClusterAttnRes -> SplitK KDA projection (shipped tactic)
  -> KdaSplit ReplaySSM verify -> OprojFused (TP16 fused, x = KdaSplit's output)
  -> successor: the shared-expert gate/up SplitKGemm (gate_up [768, 7168], SHARED_TACTICS[384] B8 = 64.8.4.8, FP32
     out, input = OprojFused's output)

Ranks 4-15's BF16 partials are written into this GPU's mailbox slots before each layer (present when the o-proj polls).
Not modelled: the 16-rank cross-tray lag and rank skew, the MoE front on the side stream.

Arms (one PreparedOproj per layer and arm; env flags set only around construction; one shared mailbox; every GPU
replays the same schedule): A shipped, X K31MK_OPROJ_X_SHARED=8:1, D K31MK_OPROJ_DRAIN=8:1, XD both.

Metrics (CUPTI kernel END times via torch.profiler, per layer; pre-registered):
  m1 = OprojFused end - KdaSplit end (the gate metric); m2 = successor end - KdaSplit end; kda = KdaSplit end -
  AttnRes end; oproj_dur; succ_after = successor end - OprojFused end. Per GPU: medians over replays x layers.
Checks: every arm's per-layer o-proj output, prefix and successor output bitwise equal to arm A's (graph replays from
the same restored state), A graph == A eager, finite outputs; profiled OprojFused shared memory per arm (path).

Usage (one node, 4 GPUs): python3 -m torch.distributed.run --nproc-per-node 4 k31_oproj_subchain.py --out DIR
  [--layers 8] [--reps 25] [--order A,X,D,XD] [--tag run1]
"""

import argparse
import gzip
import json
import os
import statistics
import sys
import time
import traceback
from types import SimpleNamespace as NS

os.environ.setdefault("K31_TEST_HEADS", "6")
os.environ.setdefault("CUDA_MODULE_LOADING", "EAGER")

import torch  # noqa: E402
import torch.distributed as dist  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _harness as H  # noqa: E402
import _ref_comm as C  # noqa: E402

HEADS = int(os.environ["K31_TEST_HEADS"])
D = HEADS * 128
PITCH = 4 * D + 128 + HEADS + (-(4 * D + 128 + HEADS) % 16)  # TP16 3216
K, Q, EPS, TP, HID = 7168, 8, 1e-6, 16, C.HIDDEN
ARMS = {
    "A": {},
    "X": {"K31MK_OPROJ_X_SHARED": "8:1"},
    "D": {"K31MK_OPROJ_DRAIN": "8:1"},
    "XD": {"K31MK_OPROJ_X_SHARED": "8:1", "K31MK_OPROJ_DRAIN": "8:1"},
}
FLAGS = ("K31MK_OPROJ_X_SHARED", "K31MK_OPROJ_DRAIN")
# KDA-like layer ids with alternating mailbox parity; (nb, has_delta) from _ref_comm.layer_facts: nb 1..8, delta.
LAYER_IDS = (4, 17, 28, 41, 52, 65, 76, 89)
FAMILIES = (("standin", "XoTailStandin"), ("attnres", "ClusterAttnRes"), ("splitk", "SplitKGemm"),
            ("kda", "KdaSplit"), ("oproj", "OprojFused"))
SHARED_GATE_UP = (64, 8, 4, 8)  # moe.front.SHARED_TACTICS[384] at B8


def standin(spin_ns, sms, smem, flag):
    """The MoE tail stand-in (2 x SMs CTAs, 48 KB SMEM): waits, spins, releases its dependents at its end."""
    import cutlass
    import cutlass.cute as cute
    from cutlass import Int32, Int64
    from cutlass.cute.runtime import make_fake_stream
    from sgl_nvidia_mega.kimi_k31.k3.cutedsl.common.host import stream, to_cute
    from sgl_nvidia_mega.kimi_k31.k3.cutedsl.common.primitives import global_timer_ns, launch

    class XoTailStandin:
        @cute.jit
        def __call__(self, flag: cute.Tensor, stream):
            launch(self.kernel, (flag, stream), 2 * sms, 128, smem=cutlass.Int64(smem))

        @cute.kernel
        def kernel(self, args):
            flag = args[0]
            block, _, _ = cute.arch.block_idx()
            tid, _, _ = cute.arch.thread_idx()
            cute.arch.griddepcontrol_wait()
            start = global_timer_ns()
            now = global_timer_ns()
            while now - start < Int64(spin_ns):
                now = global_timer_ns()
            cute.arch.griddepcontrol_launch_dependents()
            if (block == 0) & (tid == 0):
                flag[0] = Int32(1)

    view = to_cute(flag)
    code = cute.compile(XoTailStandin(), view, make_fake_stream())
    return lambda: code(view, stream(flag.device))


def build_kda(g, dev):
    """The KDA layer (AttnRes, packed projection, KdaSplit verify) at the shipped projection tactic."""
    from sgl_nvidia_mega.kimi_k31.k3.cutedsl.attn_res.prepared import PreparedAttnRes
    from sgl_nvidia_mega.kimi_k31.k3.cutedsl.gemm.projection import PreparedProjection, projection_tactic
    from sgl_nvidia_mega.kimi_k31.k3.cutedsl.kda.prepared import decode_conv_weight
    from sgl_nvidia_mega.kimi_k31.verify_kda import PreparedKdaVerify

    norm = lambda: NS(weight=(1.0 + H.randn((7168,), 0.2, g)).to(torch.bfloat16), variance_epsilon=EPS)  # noqa: E731
    target = NS(self_attention_res_norm=norm(), self_attention_res_proj=NS(weight=H.randn((1, 7168), 0.02, g)),
                input_layernorm=norm(), prev_valid_blocks=4, is_block_write_layer=False, block_write_idx=0)
    attnres = PreparedAttnRes.for_target(target, deadline_ns=4_000_000_000, final=False, has_delta=True)
    weight = H.randn((PITCH, K), 0.02, g)
    proj = PreparedProjection(weight, {Q: projection_tactic("kda", Q, dev)}).prepare()
    conv_w = (torch.randn((3 * D, 1, 4), generator=g, device=dev) * 0.5).float()
    kw = NS(conv_weight=decode_conv_weight(conv_w), has_bias=False, f_b=H.randn((D, 128), 0.1, g),
            a_log=(torch.randn((HEADS,), generator=g, device=dev) * 0.5 - 1.0).float(),
            dt_bias=(torch.randn((D,), generator=g, device=dev) * 0.5).float(),
            norm_weight=(1.0 + H.randn((128,), 0.2, g)).float(), lower_bound=-5.0, scale=128**-0.5, norm_eps=EPS)
    cap, ring_len, scratch_rows = 5, 16, 3
    conv = H.randn((cap, 3, 3 * D), 1.0, g)
    state = (torch.randn((cap, HEADS, 128, 128), generator=g, device=dev) * 0.1).float()
    rawv = torch.zeros((cap, HEADS, ring_len, 128), dtype=torch.bfloat16, device=dev)
    pools = NS(conv=conv.transpose(-1, -2), temporal=state, rawv=rawv, rawk=torch.zeros_like(rawv),
               log_g=torch.zeros((cap, HEADS, ring_len, 128), dtype=torch.float32, device=dev),
               beta=torch.zeros((cap, HEADS, ring_len), dtype=torch.float32, device=dev),
               conv_windows=torch.zeros((scratch_rows, Q, 3, 3 * D), dtype=torch.bfloat16, device=dev).transpose(-1, -2))
    core = PreparedKdaVerify(kw, ((1, Q, PITCH, 0),), dev.index).prepare_caches(pools)
    first = torch.empty((Q, 7168), dtype=torch.bfloat16, device=dev)
    projected = torch.empty((Q, PITCH), dtype=torch.bfloat16, device=dev)
    local = torch.empty((Q, D), dtype=torch.bfloat16, device=dev)
    return NS(attnres=attnres, proj=proj, core=core, pools=pools, first=first, projected=projected, local=local)


def build_oproj(P, lid, g, dev, gen, mailbox, deadline_ns):
    """Per arm: this rank's PreparedOproj of layer ``lid``; plus its prefix / bank, injected partials, successor."""
    from sgl_nvidia_mega.kimi_k31.k3.cutedsl.gemm.projection import PreparedProjection

    w = (H.randn((HID, D), D**-0.5, gen)).contiguous()
    lay = NS(lid=lid, w=w, preps={}, decs={})
    for arm, env in ARMS.items():
        saved = {f: os.environ.pop(f, None) for f in FLAGS}
        os.environ.update(env)
        try:
            dec = C.make_decoder(lid, w, dev)
            lay.decs[arm] = dec  # the prepared views do not hold the decoder's tensors
            lay.preps[arm] = P.PreparedOproj(dec, mailbox, (Q,), False, None, None, deadline_ns=deadline_ns)
            assert len(lay.preps[arm].codes[Q]) == 1, "TP16 B8 KDA must take the fused kernel"
        finally:
            for f in FLAGS:
                os.environ.pop(f, None)
                if saved[f] is not None:
                    os.environ[f] = saved[f]
    lay.prefix = H.randn((Q, HID), 1.0, gen)
    lay.bank = H.randn((Q, 8, HID), 1.0, gen)
    lay.inj = C.canon(H.randn((TP - 4, Q, HID), 1.0, gen))
    lay.out = torch.empty((Q, HID), dtype=torch.bfloat16, device=dev)
    lay.succ = PreparedProjection(H.randn((768, HID), 0.02, gen), {Q: SHARED_GATE_UP},
                                  output_dtype=torch.float32).prepare()
    lay.y = torch.empty((Q, 768), dtype=torch.float32, device=dev)
    return lay


def segments(trace_path, schedule, layers):
    d = json.load(gzip.open(trace_path))
    ks = sorted((e for e in d["traceEvents"] if e.get("cat") == "kernel"), key=lambda e: e["ts"])
    fam, unknown = [], set()
    for e in ks:
        f = next((name for name, pat in FAMILIES if pat in e["name"]), None)
        if f is None:
            unknown.add(e["name"][:100])
            continue
        fam.append((f, e))
    per = 6 * layers
    if len(fam) != per * len(schedule):
        raise RuntimeError(f"profiled {len(fam)} chain kernels, expected {per * len(schedule)}; unknown {sorted(unknown)[:5]}")
    out = {}
    end = lambda e: e["ts"] + e["dur"]  # noqa: E731
    for r, arm in enumerate(schedule):
        chunk = fam[r * per:(r + 1) * per]
        by = {name: [e for f, e in chunk if f == name] for name, _ in FAMILIES}
        assert len(by["splitk"]) == 2 * layers and all(len(by[n]) == layers for n in ("standin", "attnres", "kda", "oproj"))
        row = out.setdefault(arm, {k: [] for k in ("m1", "m2", "kda", "oproj_dur", "succ_after", "oproj_start_vs_kda_end")})
        row.setdefault("path", set())
        row.setdefault("per_layer_m1", [[] for _ in range(layers)])
        for i in range(layers):
            ar, kd, op = by["attnres"][i], by["kda"][i], by["oproj"][i]
            proj, succ = by["splitk"][2 * i], by["splitk"][2 * i + 1]
            assert proj["ts"] < kd["ts"] < op["ts"] < succ["ts"], "kernel order within a layer"
            row["m1"].append(end(op) - end(kd))
            row["m2"].append(end(succ) - end(kd))
            row["kda"].append(end(kd) - end(ar))
            row["oproj_dur"].append(op["dur"])
            row["succ_after"].append(end(succ) - end(op))
            row["oproj_start_vs_kda_end"].append(op["ts"] - end(kd))
            row["per_layer_m1"][i].append(end(op) - end(kd))
            a = op.get("args", {})
            row["path"].add((str(a.get("grid")), a.get("shared memory"), a.get("registers per thread")))
    return out


def med(xs):
    return round(statistics.median(xs), 4) if xs else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--layers", type=int, default=8)
    ap.add_argument("--reps", type=int, default=25)
    ap.add_argument("--order", default="A,X,D,XD")
    ap.add_argument("--spin-us", type=float, default=8.0)
    ap.add_argument("--tag", default="run1")
    ap.add_argument("--seed", type=int, default=5)
    ap.add_argument("--deadline-s", type=float, default=120.0)  # o-proj poll fail-stop (lazy loads skew ranks)
    args = ap.parse_args()
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
    dev = torch.device("cuda", local_rank)
    dist.init_process_group("nccl", device_id=dev)
    g, world = dist.get_rank(), dist.get_world_size()
    assert world == 4, "TP ranks 0-3 real, 4-15 injected"
    assert args.layers <= len(LAYER_IDS), args.layers
    os.makedirs(args.out, exist_ok=True)
    tag = f"{args.tag}_g{g}"
    for f in FLAGS:
        os.environ.pop(f, None)
    res = {"tag": tag, "args": vars(args), "ok": False, "heads": HEADS, "rank": g,
           "env": {k: v for k, v in os.environ.items() if k.startswith("K31")}}
    t0 = time.time()
    group = dist.group.WORLD

    def sync():
        torch.cuda.synchronize()
        dist.barrier(group=group)

    try:
        H.load_plugin()
        from sgl_nvidia_mega.kimi_k31.k3.cutedsl.oproj import prepared as P

        sms = torch.cuda.get_device_properties(dev).multi_processor_count
        res.update(sms=sms, device=torch.cuda.get_device_name(dev))
        mailbox = P.OprojMailbox(group, dev, TP, g)
        gk = H.generator(args.seed + g)  # KDA part: per GPU
        go = H.generator(1000 + args.seed + g)  # o-proj part: per GPU (each GPU is its own TP rank)
        kdas = [build_kda(gk, dev) for _ in range(args.layers)]
        ops = [build_oproj(P, LAYER_IDS[i], g, dev, go, mailbox, int(args.deadline_s * 1e9)) for i in range(args.layers)]
        flag = torch.zeros(4, dtype=torch.int32, device=dev)
        tail = standin(int(args.spin_us * 1000), sms, 48 * 1024, flag)
        delta = H.randn((Q, 7168), 1.0, gk)
        prefix, blocks = H.randn((Q, 7168), 2.0, gk), H.randn((Q, 8, 7168), 2.0, gk)
        step = NS(state_slots=torch.tensor([1], dtype=torch.int32, device=dev),
                  scratch_rows=torch.tensor([0], dtype=torch.int32, device=dev), query_width=Q)

        def run(arm):
            for kd, op in zip(kdas, ops):
                mailbox.buffer[op.lid % 2, 4:TP, :Q].copy_(op.inj)
                tail()
                kd.attnres.run(delta, prefix, blocks, kd.first)
                kd.proj.run(kd.first, kd.projected)
                kd.core.run(kd.projected, kd.local, pools=kd.pools, step=step)
                op.preps[arm].run(kd.local, op.prefix, op.bank, op.out)
                op.succ.run(op.out, op.y)

        state = [prefix, blocks] + [t for kd in kdas for t in (kd.pools.rawv, kd.pools.rawk, kd.pools.log_g,
                                                                kd.pools.beta, kd.pools.conv_windows)]
        state += [op.prefix for op in ops]
        saved = [t.clone() for t in state]

        def restore():
            for t, s in zip(state, saved):
                t.copy_(s)

        def outputs():
            return [(kd.projected.clone(), kd.local.clone(), op.out.clone(), op.prefix.clone(), op.y.clone())
                    for kd, op in zip(kdas, ops)]

        for arm in ARMS:  # load every module
            restore()
            sync()
            run(arm)
            sync()
        restore()
        sync()
        run("A")
        sync()
        eager = outputs()
        res["finite"] = all(bool(torch.isfinite(t.float()).all()) for row in eager for t in row)
        graphs, side = {}, torch.cuda.Stream(device=dev)
        side.wait_stream(torch.cuda.current_stream(dev))
        for arm in ARMS:
            restore()
            sync()
            graphs[arm] = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graphs[arm], stream=side):
                run(arm)
            sync()
        names = ("proj", "kda", "oproj_out", "oproj_prefix", "succ")
        bitwise = {}
        for arm in ARMS:
            restore()
            sync()
            graphs[arm].replay()
            sync()
            got = outputs()
            cells = {n: 0 for n in names}
            for x, y in zip(got, eager):
                for n, a, b in zip(names, x, y):
                    cells[n] += int((a.view(torch.int8) != b.view(torch.int8)).sum())
            bitwise[arm] = cells
        res["bitwise_vs_A_eager"] = bitwise
        res["ok_bitwise"] = all(v == 0 for c in bitwise.values() for v in c.values())
        order = args.order.split(",")
        assert sorted(order) == sorted(ARMS), order
        schedule = (order + order[::-1]) * args.reps
        for arm in ARMS:
            for _ in range(3):
                graphs[arm].replay()
        sync()
        events = [(torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)) for _ in schedule]
        from torch.profiler import ProfilerActivity, profile

        with profile(activities=[ProfilerActivity.CUDA]) as prof:
            for arm, (a, b) in zip(schedule, events):
                a.record()
                graphs[arm].replay()
                b.record()
            torch.cuda.synchronize()
        trace = os.path.join(args.out, f"trace_{tag}.json.gz")
        raw = trace[:-3]
        prof.export_chrome_trace(raw)
        with open(raw, "rb") as f, gzip.open(trace, "wb") as z:
            z.write(f.read())
        os.remove(raw)
        wall = {}
        for arm, (a, b) in zip(schedule, events):
            wall.setdefault(arm, []).append(a.elapsed_time(b) * 1000.0 / args.layers)
        seg = segments(trace, schedule, args.layers)
        res["arms"] = {}
        for arm in ARMS:
            s = seg[arm]
            res["arms"][arm] = {k: med(s[k]) for k in ("m1", "m2", "kda", "oproj_dur", "succ_after", "oproj_start_vs_kda_end")}
            res["arms"][arm].update(
                n=len(s["m1"]), wall_per_layer_med=med(wall[arm]), path=sorted(map(list, s["path"])),
                per_layer_m1=[med(x) for x in s["per_layer_m1"]],
                m1_p10=round(sorted(s["m1"])[len(s["m1"]) // 10], 4),
                m1_p90=round(sorted(s["m1"])[(9 * len(s["m1"])) // 10], 4))
        res["ok"] = res["finite"] and res["ok_bitwise"]
    except BaseException as e:  # noqa: BLE001
        res["error"] = repr(e)[:2000]
        res["trace"] = traceback.format_exc()[-6000:]
        print(f"ERROR {e!r}\n{traceback.format_exc()}", flush=True)
    res["total_s"] = round(time.time() - t0, 1)
    with open(os.path.join(args.out, f"subchain_{tag}.json"), "w") as f:
        json.dump(res, f, indent=1, default=str)
    brief = {k: res.get(k) for k in ("tag", "ok", "bitwise_vs_A_eager", "error")}
    brief["arms"] = {a: {k: v[k] for k in ("m1", "m2", "kda", "oproj_dur", "wall_per_layer_med", "path")}
                     for a, v in res.get("arms", {}).items()}
    print("XO_SUBCHAIN_RESULT", json.dumps(brief), flush=True)
    ok = torch.tensor([1 if res["ok"] else 0], device=dev)
    try:
        dist.all_reduce(ok, op=dist.ReduceOp.MIN, group=group)
        dist.destroy_process_group()
    except Exception:  # noqa: BLE001
        pass
    sys.exit(0 if int(ok) else 1)


if __name__ == "__main__":
    main()
