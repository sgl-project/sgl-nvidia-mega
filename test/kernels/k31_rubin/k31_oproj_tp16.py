# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""K3.1 TP16 fused o-projection + all-reduce + post-attention AttnRes (``oproj.prepared``, ``OprojFused``,
TP16 B<=8) and its x_shared / drain options (the ported Rubin decode forms), over
W = 4 GPUs with real symm_mem multicast. GPU g plays TP rank g of 16 (its fused kernel multicasts its partial into
slot g of every GPU's mailbox); ranks 4-15's BF16 partials are injected into this GPU's slots before each call.

Arms (each its own PreparedOproj per layer, one shared mailbox; the env flag is set only around construction):
  A   shipped (no flag)
  X   K31MK_OPROJ_X_SHARED=1-8:1
  D   K31MK_OPROJ_DRAIN=4:1,8:1
  XD  both

Checks per call and arm (tolerances fixed before any run; reference: ``_ref_comm``):
  ref      prefix / out vs the FP32 reference of the stock method (``_ref_comm.reference``; P_q = bf16(x_q @ W_q.T),
           FP32 rank-order sum), ``_ref_comm`` tolerances;
  bitwise  X / D / XD prefix and out equal A's bit for bit (the options must not change a bit);
  rearm    every word of rows [0, B) of the 16 slots is the sentinel after the call;
  graph    calls 2-3 replayed from one CUDA graph per arm (both layers, injection inside) with new inputs;
  path     the compiled fused kernel's PTX (CUTE_DSL_KEEP_PTX): 'bar.arrive 5,' appears iff drain is engaged;
           x_shared engaged -> PTX differs from the same arm without it; nothing engaged -> PTX equals A's.
Bugs caught (``MUTATIONS``): x tile rows / columns misplaced in SMEM, a dropped x load round, a missing barrier
before the x tile is read (race), the drain barrier
on the consumer's barrier id or with a wrong count, an option flag read but not passed to the kernel.

Usage (one node, 4 GPUs): python3 -m torch.distributed.run --nproc-per-node 4 k31_oproj_tp16.py --out DIR
  [--batches 1,4,5,8]. Exit status 0 only when every check on every rank passed (``TP_RESULT ... failures=0``).
"""

import argparse
import json
import os
import sys
import time
import traceback
import zlib

import torch
import torch.distributed as dist

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _harness as H  # noqa: E402
import _ref_comm as C  # noqa: E402

TP, K, HIDDEN = 16, 768, C.HIDDEN
ARMS = {
    "A": {},
    "X": {"K31MK_OPROJ_X_SHARED": "1-8:1"},
    "D": {"K31MK_OPROJ_DRAIN": "4:1,8:1"},
    "XD": {"K31MK_OPROJ_X_SHARED": "1-8:1", "K31MK_OPROJ_DRAIN": "4:1,8:1"},
}
FLAGS = ("K31MK_OPROJ_X_SHARED", "K31MK_OPROJ_DRAIN")
# batch -> layer pairs (even parity first). Layer ids fix (nb, has_delta) as _ref_comm.layer_facts: 92 (8, delta),
# 5 (1, delta), 84 (8, no delta), 41 (4, delta), 12 (2, no delta), 0 (1, no delta).
CASES = {8: [(92, 5), (84, 41)], 5: [(92, 41)], 4: [(12, 5)], 1: [(0, 41)]}

PR = "kimi_k31/k3/cutedsl/oproj/producer.py"
CO = "kimi_k31/k3/cutedsl/oproj/consumer.py"
PP = "kimi_k31/k3/cutedsl/oproj/prepared.py"
MUTATIONS = {
    # x_shared stores token t's x row at row t + 1 (mod B): every MMA reads the wrong token's x.
    "xs_token": (PR, "                    dst = xs.iterator + (token * self.x_stride + col) * 4\n"
                     "                    store_shared_words(dst, words)",
                 "                    dst = xs.iterator + (((token + 1) % M) * self.x_stride + col) * 4\n"
                 "                    store_shared_words(dst, words)"),
    # x_shared stores each 16-byte chunk one column further (wrapping): K slices misaligned with the weights.
    "xs_col": (PR, "                    token, col = chunk // (self.k // 8), chunk % (self.k // 8)\n"
                   "                    words = tuple(held[i, v] for v in range(4))",
               "                    token, col = chunk // (self.k // 8), (chunk + 1) % (self.k // 8)\n"
               "                    words = tuple(held[i, v] for v in range(4))"),
    # x_shared skips its last load round (B8: chunks 512-767; B<=5: all of x): stale registers land in SMEM.
    "xs_round": (PR, "            for i in cutlass.range_constexpr(rounds):\n"
                     "                chunk = tid + i * 512\n"
                     "                if chunk < chunks:\n"
                     "                    token, col = chunk // (self.k // 8), chunk % (self.k // 8)\n"
                     "                    words = load_vector(",
                 "            for i in cutlass.range_constexpr(rounds - 1):\n"
                 "                chunk = tid + i * 512\n"
                 "                if chunk < chunks:\n"
                 "                    token, col = chunk // (self.k // 8), chunk % (self.k // 8)\n"
                 "                    words = load_vector("),
    # x_shared without the CTA barrier between the shared stores and the MMA reads (a race; also the racecheck control).
    "xs_nosync": (PR, "                    store_shared_words(dst, words)\n"
                      "            cute.arch.sync_threads()\n"
                      "        elif cutlass.const_expr(self.shared_x):",
                  "                    store_shared_words(dst, words)\n"
                  "        elif cutlass.const_expr(self.shared_x):"),
    # The drain barrier reuses the consumer's group barrier 2 (count 512): a sync-type bug.
    "drain_id": (CO, "    DRAIN_BARRIER = 5  # free in this kernel: the consumer uses 1 and 2",
                 "    DRAIN_BARRIER = 2  # free in this kernel: the consumer uses 1 and 2"),
    # The drain barrier counts one warp too many: the drain warp waits for arrivals that never come (hang).
    "drain_count": (CO, "        drainers: cutlass.Constexpr = publishers + 32",
                    "        drainers: cutlass.Constexpr = publishers + 64"),
    # PreparedOproj reads the x_shared table but never passes it to the kernel (only the path check sees it).
    "xs_inert": (PP, "options = dict(x_shared=b in x_shared, drain=b in drain) if fused else {}",
                 "options = dict(x_shared=False, drain=b in drain) if fused else {}"),
    # Same for drain.
    "drain_inert": (PP, "options = dict(x_shared=b in x_shared, drain=b in drain) if fused else {}",
                    "options = dict(x_shared=b in x_shared, drain=False) if fused else {}"),
}


def _gen(device, *key):
    """The same CUDA stream in every process (same seed, same GPU model)."""
    g = torch.Generator(device=device)
    g.manual_seed(zlib.crc32(repr(key).encode()))
    return g


def rank_weight(layer, q, device):
    """o_proj weight [7168, 768] of TP rank q (identical in every process)."""
    g = _gen(device, "w16", layer, q)
    return (torch.randn((HIDDEN, K), generator=g, device=device) * K**-0.5).to(torch.bfloat16)


def rank_input(layer, call, q, batch, device):
    g = _gen(device, "x16", layer, call, q)
    return torch.randn((batch, K), generator=g, device=device).to(torch.bfloat16)


def engaged(arm, batch):
    """(x_shared, drain) engaged for ``arm`` at ``batch`` (the tables of ``ARMS``, written out)."""
    return "X" in arm and 1 <= batch <= 8, "D" in arm and batch in (4, 8)


def ptx_of(code):
    """PTX text of a compiled kernel (CuTe DSL keeps it in ``__ptx__`` with CUTE_DSL_KEEP_PTX and CUTE_DSL_DUMP_DIR)."""
    text = code.__ptx__
    if not isinstance(text, str) or ".version" not in text:
        raise RuntimeError(f"no PTX text on the compiled kernel: {str(text)[:120]!r}")
    return text


class Case:
    def __init__(self, P, group, g, world, device, batch, pair, deadline_ns):
        self.P, self.group, self.g, self.world, self.device, self.batch = P, group, g, world, device, batch
        self.layers = pair
        self.name = f"oproj16_tp{world}.b{batch}.l{pair[0]}-{pair[1]}"
        self.mailbox = P.OprojMailbox(group, device, TP, g)
        self.weights = {layer: [rank_weight(layer, q, device) for q in range(TP)] for layer in pair}
        self.decoders, self.preps, self.ptx = {}, {}, {}
        for arm, env in ARMS.items():
            saved = {f: os.environ.pop(f, None) for f in FLAGS}
            os.environ.update(env)
            try:
                for layer in pair:
                    # Kept per arm: the prepared views do not hold the decoder's tensors.
                    dec = C.make_decoder(layer, self.weights[layer][g], device)
                    self.decoders[arm, layer] = dec
                    prep = P.PreparedOproj(dec, self.mailbox, (batch,), False, None, None, deadline_ns=deadline_ns)
                    self.preps[arm, layer] = prep
                    code = prep.codes[batch]
                    assert len(code) == 1, "TP16 B<=8 KDA must take the fused kernel"
                    self.ptx[arm, layer] = ptx_of(code[0])
            finally:
                for f in FLAGS:
                    os.environ.pop(f, None)
                    if saved[f] is not None:
                        os.environ[f] = saved[f]

    def path_check(self, failures):
        for layer in self.layers:
            base = self.ptx["A", layer]
            for arm in ARMS:
                xs, dr = engaged(arm, self.batch)
                text = self.ptx[arm, layer]
                name = f"{self.name}.l{layer}.g{self.g}.{arm}.path"
                problems = []
                if ("bar.arrive 5," in text) != dr:
                    problems.append(f"drain marker {'missing' if dr else 'present'}")
                if xs and (("D" if dr else "A"), layer) in self.ptx:
                    plain = self.ptx[("D" if dr else "A"), layer]
                    if text == plain:
                        problems.append("x_shared engaged but the PTX equals the arm without it")
                if not xs and not dr and text != base:
                    problems.append("no option engaged but the PTX differs from A")
                H.record({"kind": "check", "item": name, "x_shared": xs, "drain": dr, "problems": problems,
                          "bar_arrive_5": text.count("bar.arrive 5,"), "st_shared": text.count("st.shared")})
                if problems:
                    failures.append(f"{name}: {problems}")
                    print("FAIL", name, problems, flush=True)

    def inputs(self, layer, call):
        b, d = self.batch, self.device
        xs = [rank_input(layer, call, q, b, d) for q in range(TP)]
        prefix_old, bank = C.state_inputs(layer, call, b, d)
        partials = [C.partial_ref(xs[q], self.weights[layer][q]) for q in range(TP)]
        ref = C.reference(self.decoders["A", layer], partials, prefix_old, bank)
        injected = torch.stack(partials[self.world:])  # [TP - W, B, H]
        return xs[self.g], prefix_old, bank, injected, ref

    def verify(self, tag, arm, layer, prefix, out, ref, a_out, failures):
        name = f"{self.name}.l{layer}.g{self.g}.{tag}.{arm}"
        try:
            C.check_outputs(name, prefix, out, *ref)
            C.check_rearmed(name, self.mailbox.buffer[layer % 2], self.batch)
            if a_out is not None:
                same = [bool(torch.equal(u.view(torch.int16), v.view(torch.int16))) for u, v in zip((prefix, out), a_out)]
                diff = [int((u.view(torch.int16) != v.view(torch.int16)).sum()) for u, v in zip((prefix, out), a_out)]
                H.record({"kind": "check", "item": f"{name}.bitwise_vs_A", "prefix_cells": diff[0], "out_cells": diff[1]})
                assert all(same), f"not bitwise equal to arm A: prefix {diff[0]} / out {diff[1]} cells differ"
        except AssertionError as e:
            failures.append(f"{name}: {e}")
            print("FAIL", name, str(e)[:400], flush=True)

    def sync(self):
        torch.cuda.synchronize()
        dist.barrier(group=self.group)

    def eager(self, failures, prefill=False):
        """``prefill`` (sanitizer runs only, no checks): every slot, this GPU's own and its peers' included, is filled
        before the call, so no CTA waits on another GPU (racecheck serializes enough that the cross-GPU poll never
        finishes); one call per layer and arm."""
        b, d = self.batch, self.device
        out = torch.empty((b, HIDDEN), dtype=torch.bfloat16, device=d)
        for call in ((0,) if prefill else (0, 1)):
            for layer in self.layers:
                x, prefix_old, bank, injected, ref = self.inputs(layer, call)
                a_out = None
                for arm in ARMS:
                    self.sync()
                    if prefill:
                        lay_partials = [C.partial_ref(rank_input(layer, call, q, b, d), self.weights[layer][q])
                                        for q in range(self.world)]
                        self.mailbox.buffer[layer % 2, :TP, :b].copy_(torch.cat([torch.stack(lay_partials), injected]))
                        prefix = prefix_old.clone()
                        self.preps[arm, layer].run(x, prefix, bank, out)
                        self.sync()
                        continue
                    self.mailbox.buffer[layer % 2, self.world:TP, :b].copy_(injected)
                    prefix = prefix_old.clone()
                    H.poison(out)
                    self.preps[arm, layer].run(x, prefix, bank, out)
                    self.sync()
                    self.verify(f"call{call}", arm, layer, prefix, out, ref, a_out, failures)
                    if arm == "A":
                        a_out = (prefix.clone(), out.clone())

    def graph(self, failures):
        b, d = self.batch, self.device
        st = {layer: {"x": torch.empty((b, K), dtype=torch.bfloat16, device=d),
                      "prefix": torch.empty((b, HIDDEN), dtype=torch.bfloat16, device=d),
                      "bank": torch.empty((b, 8, HIDDEN), dtype=torch.bfloat16, device=d),
                      "inj": torch.empty((TP - self.world, b, HIDDEN), dtype=torch.bfloat16, device=d),
                      "out": torch.empty((b, HIDDEN), dtype=torch.bfloat16, device=d)} for layer in self.layers}
        graphs = {}
        for arm in ARMS:
            self.sync()
            graphs[arm], stream = torch.cuda.CUDAGraph(), torch.cuda.Stream()
            with torch.cuda.graph(graphs[arm], stream=stream):
                for layer in self.layers:
                    s = st[layer]
                    self.mailbox.buffer[layer % 2, self.world:TP, :b].copy_(s["inj"])
                    self.preps[arm, layer].run(s["x"], s["prefix"], s["bank"], s["out"])
        self.sync()
        for call in (2, 3):
            ins = {layer: self.inputs(layer, call) for layer in self.layers}
            a_out = {}
            for arm in ARMS:
                for layer in self.layers:
                    x, prefix_old, bank, injected, _ = ins[layer]
                    s = st[layer]
                    s["x"].copy_(x)
                    s["prefix"].copy_(prefix_old)
                    s["bank"].copy_(bank)
                    s["inj"].copy_(injected)
                    H.poison(s["out"])
                self.sync()
                graphs[arm].replay()
                self.sync()
                for layer in self.layers:
                    s = st[layer]
                    self.verify(f"graph{call}", arm, layer, s["prefix"], s["out"], ins[layer][4],
                                a_out.get(layer), failures)
                    if arm == "A":
                        a_out[layer] = (s["prefix"].clone(), s["out"].clone())
        del graphs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--batches", default=",".join(map(str, CASES)))
    ap.add_argument("--deadline-s", type=float, default=2.0)
    ap.add_argument("--arms", default=",".join(ARMS), help="subset (A always runs: the bitwise reference)")
    ap.add_argument("--max-pairs", type=int, default=0, help="layer pairs per batch (0 = all)")
    ap.add_argument("--prefill", action="store_true", help="sanitizer mode: all slots pre-filled, eager call 0, no checks")
    args = ap.parse_args()
    keep = {"A", *args.arms.split(",")}
    for arm in [a for a in ARMS if a not in keep]:
        del ARMS[arm]
    local = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local)
    device = torch.device("cuda", local)
    dist.init_process_group("nccl", device_id=device)
    g, world = dist.get_rank(), dist.get_world_size()
    assert world == 4, "TP ranks 0-3 real, 4-15 injected"
    tag = os.environ.get("K31_MUTATION") or "base"
    os.makedirs(args.out, exist_ok=True)
    os.environ["K31_RESULTS"] = os.path.join(args.out, f"tp16_{tag}_g{g}.jsonl")
    # PTX of every compiled kernel, one dump directory per process (the dump names collide across ranks).
    os.environ["CUTE_DSL_KEEP_PTX"] = "1"
    os.environ["CUTE_DSL_DUMP_DIR"] = os.path.join(args.out, f"ptx_{tag}_g{g}")
    os.makedirs(os.environ["CUTE_DSL_DUMP_DIR"], exist_ok=True)
    for f in FLAGS:
        os.environ.pop(f, None)
    H.load_plugin(MUTATIONS)
    from sgl_nvidia_mega.kimi_k31.k3.cutedsl.oproj import prepared as P

    H.record({"kind": "env", **H.device_facts(), "world": world})
    failures, checks = [], 0
    group = dist.group.WORLD
    for batch in map(int, args.batches.split(",")):
        for pair in CASES[batch][: args.max_pairs or None]:
            t0 = time.time()
            try:
                case = Case(P, group, g, world, device, batch, pair, int(args.deadline_s * 1e9))
                case.path_check(failures)
                case.eager(failures, prefill=args.prefill)
                if not args.prefill:
                    case.graph(failures)
                checks += 1
                print(f"CASE {case.name} g{g} done in {time.time() - t0:.1f}s", flush=True)
            except Exception as e:  # noqa: BLE001 - report and continue with the next case
                failures.append(f"b{batch} {pair}: {e!r}")
                print("CASE_ERROR", batch, pair, repr(e)[:600], flush=True)
                traceback.print_exc()
    t = torch.tensor([len(failures)], device=device)
    dist.all_reduce(t, group=group)
    with open(os.path.join(args.out, f"tp16_{tag}_g{g}.failures.json"), "w") as f:
        json.dump(failures, f, indent=1)
    print(f"[rank{g}] TP_RESULT world={world} cases={checks} failures={int(t)}", flush=True)
    dist.destroy_process_group()
    sys.exit(1 if int(t) else 0)


if __name__ == "__main__":
    main()
