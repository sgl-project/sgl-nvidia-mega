# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""The KDA layer's o-projection -> MoE front half at TP16 per-rank shapes, Breq 1 x Q8 = 8 rows, L layers
(alternating o-proj mailbox parity), over W = 4 GPUs (GPU g = TP rank g; ranks 4-15 injected; real symm_mem
multicast for the o-proj mailbox and the front box). Built on an earlier o-proj chain test (itself an o-proj
subchain): the stand-in, the KDA layer and the o-proj are theirs; the MoE front half is ``moe.front.SharedFront``
run as ``moe.prepared.PreparedMoE.run`` runs it at TP16 B8 when the experts grid spans every SM (base H, grid = every SM):

  [inject] -> tail stand-in -> ClusterAttnRes -> SplitK KDA projection -> KdaSplit -> OprojFused (x)
     -> side: shared gate/up SplitKGemm (FP32) -> SiTU RouteSitu(8) -> shared down SplitKGemm ; event shared_done
     -> main: Fp32FrontGemm publish (H2 push) -> RouteSitu(16) route + latent copy (H2 poll) ; wait shared_done
The routed experts would launch at max(routing end, shared down end): J below.

Arms (env flags read at SharedFront construction; one SharedFront + MoEScratch per arm):
  A, A2  shipped (A2: a second instance, control for instance offsets);
  W   K31MK_FRONT_EARLY_WEIGHTS=1-64:2  (front register weights by coherent loads: above the PDL wait in SASS);
  W3  K31MK_FRONT_EARLY_WEIGHTS=1-64:3  (coherent + at kernel entry);
  S   K31MK_MOE_SITU_EARLY=1-8:1        (the SiTU kernel releases the shared down GEMM at entry);
  WS  W + S.
  Re-arm fence modes (common/mailbox.py REARM_FENCE; the dc89-14 cost question; every arm's values bitwise equal to A):
  FN  "0" (no fence; an unsafe cost control, never served);  FS  "sys" (fence.acq_rel.sys);
  K   K31MK_FRONT_TMA_FENCE=1-64:1 (K-16 consumer fence.proxy.async.global in the front GEMM's x copy warp);
  XD  K31MK_OPROJ_X_SHARED=8:1 + K31MK_OPROJ_DRAIN=8:1 (X+D; the same-GPU paired replicate).
  FA  "sys_alias" (fence.proxy.alias + fence.acq_rel.sys; the form a Dat3M litmus test forbids the lost publish
      with). They change the re-arm fence of OprojFused (H1 poll) and RouteSitu (H2 poll); A is "gpu" (shipped mc_rf).

Checks (tolerances stated before any run):
  x       the o-proj output (MoE input) bitwise equal on every GPU and to the calibration pass;
  front   this rank's router logits + latent-down columns (FP32 words) vs FP64 x @ W^T (k31_moe_tp16 latent bound
          1e-4 rms + 1e-5 |ref|); every arm bitwise equal to A;
  route   ids / weights vs the FP64 stock rule (k31_moe_tp16.Checker.route); latent rows of ranks 0-3 vs FP64 (same
          bound), injected rows bit-exact; every arm bitwise equal to A (ids, weights, latent);
  shared  hg (FP32 gate/up) vs FP64 (latent bound); h within 1 BF16 ulp of BF16(SiTU_FP64(hg)) (stock's SiTU is
          approximate); S vs FP64 shared_down @ h: |d| <= 2^-7 |ref| + 2^-8 rms; every arm bitwise equal to A;
  graph   one CUDA graph per arm reproduces the eager snapshots bit for bit;
  path    each arm's compiled front PTX holds the coherent weight load iff W bit 2 is set (and no ld.global.nc then),
          its SiTU PTX releases its dependents before its wait iff S is set; A's equal the shipped forms.
A failing check prints "FAIL <what>"; the run ends with "TP_RESULT rank=g failures=N".

Timing (--time): palindromic arm schedule x reps, CUPTI END times per layer:
  fa = front end - OprojFused end; rt = routing(16) end - OprojFused end; gu = gate/up end - OprojFused end;
  st = SiTU end - OprojFused end; sh = shared down end - OprojFused end; J = max(rt, sh); m1 = OprojFused end -
  KdaSplit end (the H1 hop: o-proj partials, poll, re-arm); rf = rt - fa (the H2 hop: routing end - front end). Per GPU: median over replays x layers; reported per arm: median of the 4 per-GPU medians.

Usage (one node, 4 GPUs): python3 -m torch.distributed.run --nproc-per-node 4 k31_com_front_chain.py --out DIR
  [--arms A,A2,W,W3,S,WS] [--time] [--layers 8] [--reps 25] [--tag run1]
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
# The DSL reads its keep / dump settings when cutlass is imported (k31_moe_tp16 imports it): set them first, so every
# compiled kernel carries its PTX text (path checks). One dump directory per process (names collide across ranks).
_out = sys.argv[sys.argv.index("--out") + 1] if "--out" in sys.argv else "."
os.environ["CUTE_DSL_KEEP"] = "ptx"
os.environ["CUTE_DSL_DUMP_DIR"] = os.path.join(_out, f"ptx_g{os.environ.get('LOCAL_RANK', '0')}")
os.makedirs(os.environ["CUTE_DSL_DUMP_DIR"], exist_ok=True)  # the DSL silently skips the dump into a missing directory

import torch  # noqa: E402
import torch.distributed as dist  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _harness as H  # noqa: E402
import _ref_comm as C  # noqa: E402
import _ref_moe as R  # noqa: E402
import _ref_moe16 as M  # noqa: E402
from k31_moe_tp16 import LAT, LOGITS, LROWS, Checker, Model, bf, ulp_bf16  # noqa: E402

WS_ = "kimi_k31/k3/cutedsl/gemm/ws_gemm.py"
FR_ = "kimi_k31/k3/cutedsl/moe/front.py"
MUTATIONS = {
    # The coherent weight loader reads 16 B past the weight row it is given (W arms only).
    "ew_offset": (WS_, "ld.global.L1::no_allocate.L2::cache_hint.v4.b32 {$0,$1,$2,$3}, [$4], pol;",
                  "ld.global.L1::no_allocate.L2::cache_hint.v4.b32 {$0,$1,$2,$3}, [$4+16], pol;"),
    # The entry-placed weight loads (W3) start one 8-column fragment too far along K.
    "ew3_offset": (WS_, "            wp0 = w.iterator + row * self.k + kofs\n",
                   "            wp0 = w.iterator + row * self.k + kofs + 8\n"),
    # The env knob is read but never reaches the kernel (only the path check can see it).
    "ew_inert": (FR_, "    kernel.early_weights = early_weights\n", "    kernel.early_weights = 0\n"),
    # The SiTU kernel releases early but stops waiting for the gate/up GEMM (reads hg before it is complete).
    "se_nowait": (FR_, "                else:\n                    cute.arch.griddepcontrol_wait()\n"
                       "                    rounds = (self.shared + 255) // 256",
                  "                else:\n                    if c.const_expr(not self.early):\n"
                  "                        cute.arch.griddepcontrol_wait()\n"
                  "                    rounds = (self.shared + 255) // 256"),
    # The SiTU release knob never reaches the kernel.
    "se_inert": (FR_, "                release_early=bool(situ_early.get(b, 0)),\n", "                release_early=False,\n"),
    # The K-16 fence knob never reaches the kernel (K arm: the path check must see no fence).
    "k16_inert": (FR_, "    kernel.tma_fence = tma_fence\n", "    kernel.tma_fence = 0\n"),
    # stage_x drops the K-16 fence although the knob reaches the kernel (kernel-side inert): the path check must fail.
    "k16_dropped": (WS_, "        if cutlass.const_expr(self.tma_fence):\n            fence_proxy_async_global()",
                  "        if cutlass.const_expr(False):\n            fence_proxy_async_global()"),
    # x_shared / drain never reach the fused o-projection (XD arm: OprojFused PTX equals A's).
    "xd_inert": ("kimi_k31/k3/cutedsl/oproj/prepared.py",
                 "    tables = (batch_table(os.environ.get(n, \"\") if tp == 16 else \"\") for n in names)\n",
                 "    tables = (batch_table(\"\") for n in names)\n"),
}

HEADS = int(os.environ["K31_TEST_HEADS"])
D = HEADS * 128
PITCH = 4 * D + 128 + HEADS + (-(4 * D + 128 + HEADS) % 16)  # TP16 3216
K, Q, EPS, TP, HID = 7168, 8, 1e-6, 16, C.HIDDEN
EW, SE = "K31MK_FRONT_EARLY_WEIGHTS", "K31MK_MOE_SITU_EARLY"
KF, XS, DR = "K31MK_FRONT_TMA_FENCE", "K31MK_OPROJ_X_SHARED", "K31MK_OPROJ_DRAIN"
FWK = "K31MK_FRONT_WIDE"  # FW (front.py front_table override)
FLAGS = (EW, SE, KF, XS, DR, FWK)
ARMS = {  # name: (env, early_weights mode at B8, situ releases early at B8[, re-arm fence mode])
    "A": ({}, 0, False),
    "A2": ({}, 0, False),
    "W": ({EW: "1-64:2"}, 2, False),
    "W3": ({EW: "1-64:3"}, 3, False),
    "S": ({SE: "1-8:1"}, 0, True),
    "WS": ({EW: "1-64:2", SE: "1-8:1"}, 2, True),
    "K": ({KF: "1-64:1"}, 0, False),
    "XD": ({XS: "8:1", DR: "8:1"}, 0, False),
    "FN": ({}, 0, False, ""),
    "FS": ({}, 0, False, "sys"),
    "FA": ({}, 0, False, "sys_alias"),
    # Measurement copy only: the served o-proj / front form (WS + K + XD, rb_common.sh
    # RB_BIT_ENV) with the shipped gpu re-arm fence (SV, SV2 = instance control) and with the re-arm fence removed (SVFN).
    "SV": ({EW: "1-64:2", SE: "1-8:1", KF: "1-64:1", XS: "8:1", DR: "8:1"}, 2, True),
    "SV2": ({EW: "1-64:2", SE: "1-8:1", KF: "1-64:1", XS: "8:1", DR: "8:1"}, 2, True),
    "SVFN": ({EW: "1-64:2", SE: "1-8:1", KF: "1-64:1", XS: "8:1", DR: "8:1"}, 2, True, ""),
    # The served form + EF1 (oproj/consumer.py ``EF1``: group 0's re-arm + fence after its output publish).
    "SVEF1": ({EW: "1-64:2", SE: "1-8:1", KF: "1-64:1", XS: "8:1", DR: "8:1"}, 2, True),
    # The served form + EF2 (moe/front.py ``EF2``: the route role's re-arm + fence after its ids / weights stores;
    # "1" same threads, "w" warp 7 re-arms warp 0's cells so warp 0 runs no fence).
    "SVEF2": ({EW: "1-64:2", SE: "1-8:1", KF: "1-64:1", XS: "8:1", DR: "8:1"}, 2, True),
    "SVEF2W": ({EW: "1-64:2", SE: "1-8:1", KF: "1-64:1", XS: "8:1", DR: "8:1"}, 2, True),
    # Measurement copy only: SVEF2W + LN8 (moe/front.py ``LN8``: the lean route path at B 3-8, the ported
    # ROUTE_LEAN_WIDE). "SVLN": LN8 "1" (no PDL wait in the route role); "SVLNW": LN8 "w" (wait after ranking, before stores).
    "SVLN": ({EW: "1-64:2", SE: "1-8:1", KF: "1-64:1", XS: "8:1", DR: "8:1"}, 2, True),
    "SVLNW": ({EW: "1-64:2", SE: "1-8:1", KF: "1-64:1", XS: "8:1", DR: "8:1"}, 2, True),
    "SVEF2W2": ({EW: "1-64:2", SE: "1-8:1", KF: "1-64:1", XS: "8:1", DR: "8:1"}, 2, True),  # instance control of SVEF2W
    # FW: SVEF2W with the ported FRONT_WIDE geometry at B8 (ks 8: 48 front CTAs instead of 96). A REORDER: its front /
    # ids / weights / latent are checked against FP64 (the same bounds as A), not bitwise to A (FW_ARMS).
    "SVFW": ({EW: "1-64:2", SE: "1-8:1", KF: "1-64:1", XS: "8:1", DR: "8:1", FWK: "8:8.7.16.3.2"}, 2, True),
}
EF1_ARMS = ("SVEF1",)
EF2_ARMS = {"SVEF2": "1", "SVEF2W": "w", "SVLN": "w", "SVLNW": "w", "SVEF2W2": "w", "SVFW": "w"}
FW_ARMS = ("SVFW",)
LN8_ARMS = {"SVLN": "1", "SVLNW": "w"}
FENCES = {"gpu": (2, 0, 0, 3, 0, 0), "": (0, 0, 0, 0, 0, 0), "sys": (0, 2, 0, 0, 3, 0), "sys_alias": (0, 2, 2, 0, 3, 3)}
# ^ expected (OprojFused gpu, sys, alias; RouteSitu(16) gpu, sys, alias) fence counts in the PTX per mode at TP16 B8


def fence_mode(arm):
    return ARMS[arm][3] if len(ARMS[arm]) > 3 else "gpu"
LAYER_IDS = (4, 17, 28, 41, 52, 65, 76, 89)  # alternating o-proj parity
FAMILIES = (("standin", "ComTailStandin"), ("attnres", "ClusterAttnRes"), ("splitk", "SplitKGemm"),
            ("kda", "KdaSplit"), ("oproj", "OprojFused"), ("front", "Fp32FrontGemm"), ("routesitu", "RouteSitu"))
PER_LAYER = {"standin": 1, "attnres": 1, "splitk": 3, "kda": 1, "oproj": 1, "front": 1, "routesitu": 2}
SENT = -0x80000000
METRICS = ("fa", "rt", "gu", "st", "sh", "J", "m1", "rf")


def standin(spin_ns, sms, smem, flag):
    """The MoE tail stand-in (2 x SMs CTAs, 48 KB SMEM): waits, spins, releases its dependents at its end."""
    import cutlass
    import cutlass.cute as cute
    from cutlass import Int32, Int64
    from cutlass.cute.runtime import make_fake_stream
    from sgl_nvidia_mega.kimi_k31.k3.cutedsl.common.host import stream, to_cute
    from sgl_nvidia_mega.kimi_k31.k3.cutedsl.common.primitives import global_timer_ns, launch

    class ComTailStandin:
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
    code = cute.compile(ComTailStandin(), view, make_fake_stream())
    return lambda: code(view, stream(flag.device))


def build_kda(g, dev):
    """The KDA layer (AttnRes, packed projection, KdaSplit verify) at the deployed projection tactic."""
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


def build_layer(P, lid, g, dev, gw, gc, mailbox, deadline_ns):
    """The o-proj of layer ``lid``: this rank's weight shard (``gw``, per GPU); prefix, bank and the injected
    partials of ranks 4-15 from ``gc`` (common to every GPU), so the o-proj output x is replicated as in serving."""
    w = (H.randn((HID, D), D**-0.5, gw)).contiguous()
    lay = NS(lid=lid, w=w, parity=lid % 2)
    lay.dec = C.make_decoder(lid, w, dev)
    lay.prefix = H.randn((Q, HID), 1.0, gc)
    lay.bank = H.randn((Q, 8, HID), 1.0, gc)
    lay.inj = C.canon(H.randn((TP - 4, Q, HID), 1.0, gc))
    lay.out = torch.empty((Q, HID), dtype=torch.bfloat16, device=dev)
    lay.m = Model(g, dev, lid)  # router, bias, latent-down common; shared gate/up and down per rank
    router, down = lay.m.router[g * LOGITS:(g + 1) * LOGITS], lay.m.down[g * LROWS:(g + 1) * LROWS]
    lay.packed = torch.cat((router, router.new_zeros((-LOGITS % 16, HID)), down)).contiguous()
    return lay


def own_cols(box, g):
    """This rank's front outputs in the box as FP32: router logits [8, 56] and latent-down [8, 224]."""
    words = box[0, :Q].contiguous().view(torch.float32)  # [8, 4480]: latent 3584 FP32, then 896 logits
    return torch.cat((words[:, LAT + g * LOGITS:LAT + (g + 1) * LOGITS], words[:, g * LROWS:(g + 1) * LROWS]), 1)


def ptx_of(code):
    """PTX text of a compiled kernel: the DSL returns the text, or the dumped file's path when it keeps it on disk."""
    code = getattr(code, "compiled", code)
    text = code.__ptx__
    if isinstance(text, str) and ".version" not in text and os.path.isfile(text):
        with open(text) as f:
            text = f.read()
    if not isinstance(text, str) or ".version" not in text:
        raise RuntimeError(f"no PTX text on the compiled kernel: {str(text)[:120]!r}")
    return text


def segments(trace_path, schedule, layers):
    d = json.load(gzip.open(trace_path))
    ks = sorted((e for e in d["traceEvents"] if e.get("cat") == "kernel"), key=lambda e: e["ts"])
    fam = [(next((n for n, p in FAMILIES if p in e["name"]), None), e) for e in ks]
    fam = [(f, e) for f, e in fam if f is not None]
    per = sum(PER_LAYER.values()) * layers
    if len(fam) != per * len(schedule):
        raise RuntimeError(f"profiled {len(fam)} chain kernels, expected {per * len(schedule)}")
    out = {}
    end = lambda e: e["ts"] + e["dur"]  # noqa: E731
    grid = lambda e: tuple(e.get("args", {}).get("grid", (0, 0, 0)))  # noqa: E731
    for r, arm in enumerate(schedule):
        chunk = fam[r * per:(r + 1) * per]
        by = {name: [e for f, e in chunk if f == name] for name, _ in FAMILIES}
        for name, n in PER_LAYER.items():
            assert len(by[name]) == n * layers, (name, len(by[name]))
        route = [e for e in by["routesitu"] if grid(e)[0] == 2 * Q]
        situ = [e for e in by["routesitu"] if grid(e)[0] == Q]
        assert len(route) == layers and len(situ) == layers, "RouteSitu grids (16 route+copy, 8 SiTU) not found"
        row = out.setdefault(arm, {k: [] for k in METRICS})
        starts = [e["ts"] for e in by["standin"]] + [float("inf")]
        for i in range(layers):
            kd, op, fr = by["kda"][i], by["oproj"][i], by["front"][i]
            mine = lambda e: starts[i] < e["ts"] < starts[i + 1]  # noqa: E731
            sk = sorted((e for e in by["splitk"] if mine(e)), key=lambda e: e["ts"])
            assert len(sk) == 3, f"layer {i}: {len(sk)} SplitKGemm"
            gu, dn = sk[1], sk[2]  # projection, then the shared gate/up and the shared down
            rt = next(e for e in route if mine(e))
            st = next(e for e in situ if mine(e))
            o = end(op)
            vals = {"fa": end(fr) - o, "rt": end(rt) - o, "gu": end(gu) - o, "st": end(st) - o, "sh": end(dn) - o,
                    "m1": o - end(kd)}
            vals["J"] = max(vals["rt"], vals["sh"])
            vals["rf"] = vals["rt"] - vals["fa"]  # routing end - front end: the H2 poll, ranking, copy, re-arm
            for k, v in vals.items():
                row[k].append(v)
    return out


def med(xs):
    return round(statistics.median(xs), 4) if xs else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--arms", default="A,A2,W,W3,S,WS")
    ap.add_argument("--layers", type=int, default=8)
    ap.add_argument("--reps", type=int, default=25)
    ap.add_argument("--time", action="store_true")
    ap.add_argument("--spin-us", type=float, default=8.0)
    ap.add_argument("--tag", default="run1")
    ap.add_argument("--seed", type=int, default=5)
    ap.add_argument("--deadline-s", type=float, default=120.0)
    args = ap.parse_args()
    arms = [a for a in args.arms.split(",") if a]
    assert arms[0] == "A" and all(a in ARMS for a in arms), arms
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
    dev = torch.device("cuda", local_rank)
    dist.init_process_group("nccl", device_id=dev)
    g, world = dist.get_rank(), dist.get_world_size()
    assert world == 4, "TP ranks 0-3 real, 4-15 injected"
    os.makedirs(args.out, exist_ok=True)
    tag = f"{args.tag}_g{g}"
    res = {"tag": tag, "args": vars(args), "ok": False, "rank": g, "env": {k: os.environ.get(k) for k in (
        "K31MK_PROJ_TACTICS", "K31MK_KDA_VERIFY_OPT", "K31MK_INDEX_Q_OFF_WRITER") + FLAGS}}
    fails = []

    def fail(what):
        fails.append(what)
        print(f"FAIL {what}", flush=True)

    t0 = time.time()
    group = dist.group.WORLD

    def sync():
        torch.cuda.synchronize()
        dist.barrier(group=group)

    def gather(t):
        out = [torch.empty_like(t) for _ in range(world)]
        dist.all_gather(out, t.contiguous(), group=group)
        return out

    try:
        H.load_plugin(MUTATIONS)
        from sgl_nvidia_mega.kimi_k31.k3.cutedsl.common import mailbox as MB
        from sgl_nvidia_mega.kimi_k31.k3.cutedsl.common.host import stream, to_cute
        from sgl_nvidia_mega.kimi_k31.k3.cutedsl.moe import front as FR
        from sgl_nvidia_mega.kimi_k31.k3.cutedsl.moe import prepared as PM
        from sgl_nvidia_mega.kimi_k31.k3.cutedsl.oproj import prepared as P
        from sgl_nvidia_mega.kimi_k31.k3.cutedsl.oproj import consumer as OC

        for f in FLAGS:
            os.environ.pop(f, None)
        sms = torch.cuda.get_device_properties(dev).multi_processor_count
        res.update(sms=sms, device=torch.cuda.get_device_name(dev))
        deadline = int(args.deadline_s * 1e9)
        mailbox = P.OprojMailbox(group, dev, TP, g)
        fbox = P.OprojMailbox(group, dev, TP, g, (1, 64, 2 * LAT + 2 * 896))
        gk = H.generator(args.seed + g)  # per-rank KDA layer and o-proj weights
        gw = H.generator(1000 + args.seed + g)
        gc = H.generator(2000 + args.seed)  # common: prefix, bank, injected o-proj partials
        kdas = [build_kda(gk, dev) for _ in range(args.layers)]
        lays = [build_layer(P, LAYER_IDS[i], g, dev, gw, gc, mailbox, deadline) for i in range(args.layers)]
        flag = torch.zeros(4, dtype=torch.int32, device=dev)
        tail = standin(int(args.spin_us * 1000), sms, 48 * 1024, flag)
        delta = H.randn((Q, 7168), 1.0, gc)
        prefix, blocks = H.randn((Q, 7168), 2.0, gc), H.randn((Q, 8, 7168), 2.0, gc)
        step = NS(state_slots=torch.tensor([1], dtype=torch.int32, device=dev),
                  scratch_rows=torch.tensor([0], dtype=torch.int32, device=dev), query_width=Q)
        no_tail = NS(push=lambda shared: None)  # B8 front-mode tail: no push kernel (tail_runtime.PreparedTail.push)

        def attention(kd, op, prep):
            mailbox.buffer[op.parity, 4:TP, :Q].copy_(op.inj)
            tail()
            kd.attnres.run(delta, prefix, blocks, kd.first)
            kd.proj.run(kd.first, kd.projected)
            kd.core.run(kd.projected, kd.local, pools=kd.pools, step=step)
            prep.run(kd.local, op.prefix, op.bank, op.out)

        def build_preps():
            preps = [P.PreparedOproj(op.dec, mailbox, (Q,), False, None, None, deadline_ns=deadline) for op in lays]
            assert all(len(pr.codes[Q]) == 1 for pr in preps), "TP16 B8 KDA must take the fused kernel"
            return preps

        def set_fence(mode):
            """Re-arm fence mode for the kernels compiled next (caches cleared: the mode is not a cache key)."""
            if MB.REARM_FENCE != mode:
                MB.REARM_FENCE = mode
                P.compile_oproj.cache_clear()
                FR.compile_route.cache_clear()
                FR.compile_publish.cache_clear()

        set_fence("gpu")
        base_preps = build_preps()

        state = [prefix, blocks] + [t for kd in kdas for t in (kd.pools.rawv, kd.pools.rawk, kd.pools.log_g,
                                                                kd.pools.beta, kd.pools.conv_windows)]
        state += [op.prefix for op in lays]
        saved = [t.clone() for t in state]

        def restore():
            for t, s in zip(state, saved):
                t.copy_(s)

        # Calibration pass: the attention half alone gives each layer's x; ranks 4-15's front words come from it.
        restore()
        sync()
        for kd, op, pr in zip(kdas, lays, base_preps):
            attention(kd, op, pr)
        sync()
        for i, op in enumerate(lays):
            op.x = op.out.clone()
            if not all(torch.equal(o.view(torch.int16), op.x.view(torch.int16)) for o in gather(op.x)):
                fail(f"x layer {i}: the o-proj output differs across GPUs (inputs not replicated)")
            xd = op.x.double()
            op.logits_ref, op.lat_ref = xd @ op.m.router.double().T, xd @ op.m.down.double().T
            op.inj_lat = M.canon(op.lat_ref.float()[:, world * LROWS:])
            op.inj_logits = M.canon(op.logits_ref.float()[:, world * LOGITS:])

        # One SharedFront + MoEScratch per arm, built under the arm's env.
        objs = {}
        for arm in arms:
            env = ARMS[arm][0]
            for f in FLAGS:
                os.environ.pop(f, None)
            os.environ.update(env)
            try:
                set_fence(fence_mode(arm))
                if OC.EF1 != (arm in EF1_ARMS):  # EF1 is not a compile-cache key: clear the cache when it changes
                    OC.EF1 = arm in EF1_ARMS
                    P.compile_oproj.cache_clear()
                if FR.EF2 != EF2_ARMS.get(arm, ""):  # EF2 is not a compile-cache key either
                    FR.EF2 = EF2_ARMS.get(arm, "")
                    FR.compile_route.cache_clear()
                if FR.LN8 != LN8_ARMS.get(arm, ""):  # nor is LN8
                    FR.LN8 = LN8_ARMS.get(arm, "")
                    FR.compile_route.cache_clear()
                FR.front_table.cache_clear()  # FW: the table reads K31MK_FRONT_WIDE when it is built
                own = fence_mode(arm) != "gpu" or XS in env or DR in env or arm in EF1_ARMS  # o-proj options per PreparedOproj
                preps = build_preps() if own else base_preps
                st = PM.MoEScratch(TP, dev, (Q,))
                fronts = [FR.SharedFront(op.m.moe(), op.packed, fbox, st.front, (Q,), R.BETA, R.LIN, plan=st.plan,
                                         counters=st.shared, q=Q, deadline_ns=deadline) for op in lays]
            finally:
                for f in FLAGS:
                    os.environ.pop(f, None)
                set_fence("gpu")
                if OC.EF1:
                    OC.EF1 = False
                    P.compile_oproj.cache_clear()
                if FR.EF2:
                    FR.EF2 = ""
                    FR.compile_route.cache_clear()
                if FR.LN8:
                    FR.LN8 = ""
                    FR.compile_route.cache_clear()
                FR.front_table.cache_clear()
            objs[arm] = NS(state=st, fronts=fronts, preps=preps)

        # Path: compiled front / SiTU PTX per arm (layer 0; every layer shares the compiled kernels).
        ptx = {}
        for arm in arms:
            _, front, route, situ, _, _, _ = objs[arm].fronts[0].plans[Q]
            ptx[arm] = (ptx_of(front), ptx_of(situ), ptx_of(objs[arm].preps[0].codes[Q][0]), ptx_of(route))
        for arm in arms:
            _, mode, early = ARMS[arm][:3]
            fp, sp, op_ptx, rp = ptx[arm]
            got = tuple(t.count(f) for t in (op_ptx, rp)
                        for f in ("fence.acq_rel.gpu", "fence.acq_rel.sys", "fence.proxy.alias"))
            want = FENCES[fence_mode(arm)]
            if DR in ARMS[arm][0]:  # drain adds 2 fence.acq_rel.gpu sites
                want = (want[0] + 2,) + want[1:]
            if got != want:
                fail(f"path {arm}: fence counts (OprojFused gpu/sys/alias, RouteSitu gpu/sys/alias) {got}, "
                     f"want {want} for mode {fence_mode(arm)!r}")
            coh = fp.count("ld.global.L1::no_allocate.L2::cache_hint.v4.b32")
            nc = fp.count("ld.global.nc.L1::no_allocate")
            if (mode & 2) and not (coh > 0 and nc == 0):
                fail(f"path {arm}: front PTX has {coh} coherent and {nc} .nc weight loads (W bit 2 set)")
            if not (mode & 2) and not (coh == 0 and nc > 0):
                fail(f"path {arm}: front PTX has {coh} coherent and {nc} .nc weight loads (shipped form expected)")
            kf = KF in ARMS[arm][0]
            pa, cb, wt = fp.find("fence.proxy.async.global"), fp.find("cp.async.bulk"), fp.find("griddepcontrol.wait")
            if fp.count("fence.proxy.async.global") != (1 if kf else 0) or (kf and not (0 <= wt < pa < cb)):
                fail(f"path {arm}: front PTX has {fp.count('fence.proxy.async.global')} fence.proxy.async.global "
                     f"(wait {wt}, fence {pa}, first bulk copy {cb}); want {'1 between the wait and the copy' if kf else 0}")
            xd = XS in ARMS[arm][0] or DR in ARMS[arm][0]
            if xd and op_ptx == ptx["A"][2]:
                fail(f"path {arm}: OprojFused PTX equals A's although x_shared/drain are set")
            if arm in EF1_ARMS and "SV" in ptx and op_ptx == ptx["SV"][2]:
                fail(f"path {arm}: OprojFused PTX equals SV's although EF1 is set (the knob did not reach the kernel)")
            if arm in FW_ARMS and "SVEF2W" in ptx and fp == ptx["SVEF2W"][0]:
                fail(f"path {arm}: front PTX equals SVEF2W's although K31MK_FRONT_WIDE is set (the geometry did not reach it)")
            if arm == "SVEF2W2" and "SVEF2W" in ptx and rp != ptx["SVEF2W"][3]:
                fail(f"path {arm}: RouteSitu PTX differs from SVEF2W's (same knobs; LN8 leaked)")
            if arm in EF2_ARMS and "SV" in ptx and rp == ptx["SV"][3]:
                fail(f"path {arm}: RouteSitu PTX equals SV's although EF2 is set (the knob did not reach the kernel)")
            if arm in EF2_ARMS and "SV" in ptx and op_ptx != ptx["SV"][2]:
                fail(f"path {arm}: OprojFused PTX differs from SV's although only EF2 is set")
            if arm in LN8_ARMS and "SVEF2W" in ptx and rp == ptx["SVEF2W"][3]:
                fail(f"path {arm}: RouteSitu PTX equals SVEF2W's although LN8 is set (the knob did not reach the kernel)")
            if arm in LN8_ARMS and "SVEF2W" in ptx:  # LN8 "1" drops the route role's PDL wait; "w" keeps one (after ranking)
                want_w = ptx["SVEF2W"][3].count("griddepcontrol.wait") - (1 if LN8_ARMS[arm] == "1" else 0)
                if rp.count("griddepcontrol.wait") != want_w:
                    fail(f"path {arm}: RouteSitu has {rp.count('griddepcontrol.wait')} griddepcontrol.wait, want {want_w}")
            if arm not in EF2_ARMS and "SV" in ptx and arm.startswith("SV") and rp != ptx["SV"][3] and fence_mode(arm) == "gpu":
                fail(f"path {arm}: RouteSitu PTX differs from SV's without EF2 (EF2 leaked into a later arm)")
            if not xd and fence_mode(arm) == "gpu" and op_ptx != ptx["A"][2]:
                fail(f"path {arm}: OprojFused PTX differs from A's without an o-proj option")
            if kf:
                pass  # front PTX differs from A's by the fence only (checked above)
            elif mode != ARMS["A"][1] and fp == ptx["A"][0]:
                fail(f"path {arm}: front PTX equals A's although early_weights={mode}")
            if mode == 0 and not kf and fp != ptx["A"][0]:
                fail(f"path {arm}: front PTX differs from A's with early_weights=0")
            wait, trig = sp.find("griddepcontrol.wait"), sp.find("griddepcontrol.launch_dependents")
            if wait < 0 or trig < 0 or (trig < wait) != early:
                fail(f"path {arm}: SiTU release {'before' if trig < wait else 'after'} its wait (early={early})")
        H.record({"kind": "path", "g": g, "arms": {a: [len(p[0]), len(p[1])] for a, p in ptx.items()}})

        def run(arm, snaps=None):
            main = torch.cuda.current_stream(dev)
            o = objs[arm]
            st = o.state
            for i, (kd, op) in enumerate(zip(kdas, lays)):
                row = fbox.buffer[0, :Q].view(torch.float32)  # [8, 4480]: latent 3584 | logits 896
                row[:, world * LROWS:LAT].copy_(op.inj_lat)
                row[:, LAT + world * LOGITS:].copy_(op.inj_logits)
                attention(kd, op, o.preps[i])
                front = o.fronts[i]
                st.begin.record(main)
                with torch.cuda.stream(st.stream):
                    st.stream.wait_event(st.begin)
                    _, beside = front.shared_expert(op.out, no_tail)
                    assert beside is None, "B8 TP16: split-K shared expert"
                    st.shared_done.record(st.stream)
                ids, weights, latent = front.route(op.out)
                main.wait_event(st.shared_done)
                if snaps is not None:
                    snaps.append(NS(x=op.out.clone(), ids=ids.clone(), weights=weights.clone(),
                                    latent=latent.clone(), hg=st.front[1][:Q].clone(), h=st.front[4][:Q].clone(),
                                    S=st.front[5][:Q].clone()))

        eager = {}
        for arm in arms:
            restore()
            sync()
            eager[arm] = []
            run(arm, eager[arm])
            sync()

        # References (independent of the kernels): FP64 from x, on arm A's snapshots; every arm bitwise to A.
        for i, op in enumerate(lays):
            a = eager["A"][i]
            if not torch.equal(a.x.view(torch.int16), op.x.view(torch.int16)):
                fail(f"x layer {i}: differs from the calibration pass")
            for tok in range(Q):
                ck = Checker(f"com.A.l{op.lid}.t{tok}.g{g}", fails)
                ck.route(op.m, op.x[tok], a.ids[tok], a.weights[tok])
                ck.close("latent", a.latent[tok], op.lat_ref[tok], slice(0, world * LROWS))
                if not torch.equal(a.latent[tok, world * LROWS:].view(torch.int32), op.inj_lat[tok].view(torch.int32)):
                    ck.fail("latent_inj", "injected latent rows changed")
                ck.close("hg", a.hg[tok], op.m.gate_up.double() @ op.x[tok].double())
                h_mk = a.h[tok].double()
                h_ref = bf(R.situ(a.hg[tok, :384].double(), a.hg[tok, 384:].double()).float())
                nh = int(((h_mk - h_ref).abs() > ulp_bf16(torch.maximum(h_mk.abs(), h_ref.abs()))).sum())
                s_ref = op.m.shared_down.double() @ h_mk
                tol = 2 ** -7 * s_ref.abs() + 2 ** -8 * float(s_ref.pow(2).mean().sqrt())
                ns = int(((a.S[tok].double() - s_ref).abs() > tol).sum())
                H.record({"kind": "check", "item": f"{ck.name}.shared", "h_violations": nh, "s_violations": ns})
                if nh or not torch.isfinite(h_mk).all():
                    ck.fail("shared_h", f"{nh} hidden cells beyond 1 ulp of BF16(SiTU_FP64)")
                if ns or not torch.isfinite(a.S[tok].float()).all():
                    ck.fail("shared", f"{ns} cells vs down @ h_mk")
            for arm in arms[1:]:
                b = eager[arm][i]
                for name in ("x", "ids", "weights", "latent", "hg", "h", "S"):
                    u, v = getattr(b, name), getattr(a, name)
                    nd = int((u.contiguous().view(torch.int8) != v.contiguous().view(torch.int8)).sum())
                    if arm in FW_ARMS and name in ("ids", "weights", "latent"):
                        H.record({"kind": "check", "item": f"com.{arm}.l{op.lid}.g{g}.{name}.bytes_vs_A", "differ": nd})
                        continue  # REORDER: checked against FP64 below
                    if nd:
                        fail(f"{arm} layer {i} {name}: {nd} bytes differ from A")
                if arm in FW_ARMS:
                    for tok in range(Q):
                        ck = Checker(f"com.{arm}.l{op.lid}.t{tok}.g{g}", fails)
                        ck.route(op.m, op.x[tok], b.ids[tok], b.weights[tok])
                        ck.close("latent", b.latent[tok], op.lat_ref[tok], slice(0, world * LROWS))
                        if not torch.equal(b.latent[tok, world * LROWS:].view(torch.int32), op.inj_lat[tok].view(torch.int32)):
                            ck.fail("latent_inj", "injected latent rows changed")
        # The front words this rank published (layer-wise snapshots of the box are re-armed by routing, so the
        # front is checked through routing's latent copy above and directly here on the last layer by a front-only
        # call): every arm's front alone vs FP64, then bitwise vs A.
        fronts_only = {}
        for arm in arms:
            op = lays[-1]
            fbox.buffer.view(torch.int32).fill_(SENT)
            sync()
            _, front, _, _, fargs, _, _ = objs[arm].fronts[-1].plans[Q]
            fr = objs[arm].fronts[-1]
            front(to_cute(op.x), fr.packed, *fr.box, fargs[0], stream(dev))
            sync()
            fronts_only[arm] = own_cols(fbox.buffer, g).clone()
        op = lays[-1]
        ref = torch.cat((op.logits_ref[:, g * LOGITS:(g + 1) * LOGITS], op.lat_ref[:, g * LROWS:(g + 1) * LROWS]), 1)
        st_ = H.check("com.A.front", fronts_only["A"], ref, 1e-4 * float(ref.pow(2).mean().sqrt()), 1e-5, frac=1.0)
        if st_["violations"]:
            fail(f"A front vs FP64: {st_['violations']} cells")
        for arm in arms[1:]:
            nd = int((fronts_only[arm].view(torch.int32) != fronts_only["A"].view(torch.int32)).sum())
            if arm in FW_ARMS:  # REORDER: the same FP64 bound as A; the differing cells are recorded
                H.record({"kind": "check", "item": f"com.{arm}.front.cells_vs_A", "differ": nd})
                st_ = H.check(f"com.{arm}.front", fronts_only[arm], ref, 1e-4 * float(ref.pow(2).mean().sqrt()), 1e-5, frac=1.0)
                if st_["violations"]:
                    fail(f"{arm} front vs FP64: {st_['violations']} cells")
                continue
            if nd:
                fail(f"{arm} front differs from A in {nd} cells")
        fbox.buffer.view(torch.int32).fill_(SENT)
        sync()

        # Graph replay per arm reproduces the eager snapshots.
        cap = torch.cuda.Stream(device=dev)
        cap.wait_stream(torch.cuda.current_stream(dev))
        graphs = {}
        for arm in arms:
            restore()
            sync()
            gsn = []
            gr = torch.cuda.CUDAGraph()
            with torch.cuda.graph(gr, stream=cap):
                run(arm, gsn)
            sync()
            restore()
            sync()
            gr.replay()
            sync()
            for i, (u, v) in enumerate(zip(gsn, eager[arm])):
                for name in ("x", "ids", "weights", "latent", "hg", "h", "S"):
                    nd = int((getattr(u, name).contiguous().view(torch.int8)
                              != getattr(v, name).contiguous().view(torch.int8)).sum())
                    if nd:
                        fail(f"graph {arm} layer {i} {name}: {nd} bytes differ from eager")
        if args.time and not fails:
            for arm in arms:
                restore()
                sync()
                graphs[arm] = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graphs[arm], stream=cap):
                    run(arm)
                sync()
            for arm in arms:
                for _ in range(3):
                    graphs[arm].replay()
            sync()
            schedule = (list(arms) + list(arms)[::-1]) * args.reps
            from torch.profiler import ProfilerActivity, profile

            with profile(activities=[ProfilerActivity.CUDA]) as prof:
                for arm in schedule:
                    graphs[arm].replay()
                torch.cuda.synchronize()
            trace = os.path.join(args.out, f"trace_{tag}.json.gz")
            raw = trace[:-3]
            prof.export_chrome_trace(raw)
            with open(raw, "rb") as f, gzip.open(trace, "wb") as z:
                z.write(f.read())
            os.remove(raw)
            seg = segments(trace, schedule, args.layers)
            res["arms"] = {arm: {k: med(v) for k, v in seg[arm].items()} | {"n": len(seg[arm]["J"])} for arm in arms}
    except BaseException as e:  # noqa: BLE001
        res["error"] = repr(e)[:2000]
        res["trace"] = traceback.format_exc()[-6000:]
        fail(f"exception {e!r}"[:300])
        print(traceback.format_exc(), flush=True)
    res["failures"] = fails
    res["ok"] = not fails
    res["total_s"] = round(time.time() - t0, 1)
    with open(os.path.join(args.out, f"com_{tag}.json"), "w") as f:
        json.dump(res, f, indent=1, default=str)
    print(f"TP_RESULT rank={g} failures={len(fails)} arms={json.dumps(res.get('arms'))}", flush=True)
    ok = torch.tensor([0 if fails else 1], device=dev)
    try:
        dist.all_reduce(ok, op=dist.ReduceOp.MIN, group=group)
        dist.destroy_process_group()
    except Exception:  # noqa: BLE001
        pass
    sys.exit(0 if int(ok) else 1)


if __name__ == "__main__":
    main()
