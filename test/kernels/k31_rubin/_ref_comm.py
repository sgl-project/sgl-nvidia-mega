# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""K3.1 o-projection + TP all-reduce + post-attention AttnRes (``oproj.prepared``) test
support for the multi-GPU test ``k31_oproj_tp.py``. (A one-GPU variant is impossible: a
one-rank torch symm_mem group has no multicast address, and OprojMailbox requires one.)

TP8 geometry is kept everywhere (mailbox [2, 8, 64, 7168], weight [7168, 1536] per rank,
the consumer polls 8 slots). A process simulates the TP ranks ``ranks`` (one GPU: all 8;
four GPUs: g and g + 4): it runs one producer per simulated rank, each multicasting
its partial rows into slot[rank] on every GPU of the group, then ONE consumer per GPU.
So the one-shot path runs with real producers for all 8 slots and no injected data.
Two-shot (B33-64): TP ranks without a consumer (owners that no GPU plays) get their
receive slots pushed into local stand-in buffers, and the test writes their gather
columns (the owner's reduce, from the FP32 reference) before the call.

Reference (stock method, ``sglang.srt.layers.attn_residual``; FFN-side aggregation of the
Kimi K3 decoder layer):
  P_q    = bf16(x_q @ W_q.T)               (FP32 accumulate, BF16 output per rank)
  attn   = sum_q P_q in FP32               (the TP all-reduce)
  prefix = bf16(attn + prefix_old)         (``has_delta``, ONE rounding as stock's fused
                                            all_reduce_push_res_kernel; else bf16(attn))
  nb > 0: rows = [bank[0:nb], prefix]; cw = bf16(mlp_res_norm.w * mlp_res_proj.w);
          s_j = (rows_j . cw) / sqrt(mean(rows_j^2) + eps); p = softmax(s);
          mix = sum_j p_j rows_j (FP32); out = bf16(mix * rsqrt(mean(mix^2) + eps) * w_out)
  nb = 0: out = bf16(prefix * rsqrt(mean(prefix^2) + eps) * w_out)
"""

from __future__ import annotations

import json
import os
import zlib
from types import SimpleNamespace

import torch

import _harness as H

HIDDEN, K, TP, BLOCK, CAP = 7168, 1536, 8, 12, 64
EPS = 1e-5
SENTINEL_I32 = -0x80000000
WIDTHS = (1, 2, 3, 4, 5, 6, 7, 8, 16, 32, 64)

# Source mutations (``K31_MUTATION``); each must make the tests fail. The bug each
# stands for is in the comment.
MUTATIONS = {
    # The consumer's polling groups drop the last rank they own from their sum (a lost
    # peer partial). (comm2's "skip_rank", range(1, n - 1), was a no-op at TP8: each of
    # the 4 groups owns 2 of n = 4 register slots, so it never removed a live rank.)
    "skip_rank": (
        "kimi_k31/k3/cutedsl/oproj/consumer.py",
        "                if rank < count:\n                    pair = pair_op(pair, unpack_bf16(raw[rank, j]))",
        "                if rank < count - 1:\n                    pair = pair_op(pair, unpack_bf16(raw[rank, j]))",
    ),
    # Group 0 of the consumer does not re-arm its rank slots (stale mailbox next call).
    "no_rearm": (
        "kimi_k31/k3/cutedsl/oproj/consumer.py",
        "            for r in cutlass.range_constexpr(min(S, mailbox.shape[0])):\n                rearm(at(mailbox, r, token, h))",
        "            for r in cutlass.range_constexpr(0):\n                rearm(at(mailbox, r, token, h))",
    ),
    # The B<=8 producer drops its last K part from the in-order part sum.
    "producer_kpart": (
        "kimi_k31/k3/cutedsl/oproj/producer.py",
        "for q in cutlass.range_constexpr(1, self.parts):\n                        part = red[",
        "for q in cutlass.range_constexpr(1, self.parts - 1):\n                        part = red[",
    ),
    # The residual delta add uses the old prefix twice instead of old + attn.
    "delta_add": (
        "kimi_k31/k3/cutedsl/oproj/consumer.py",
        "acc = pair_op(two(old, j), acc)",
        "acc = pair_op(two(old, j), two(old, j))",
    ),
    # Every bank block's AttnRes logit uses block 0's RMS (wrong block index).
    "block_rms": (
        "kimi_k31/k3/cutedsl/oproj/consumer.py",
        "mul_rn(blk[self.nb + b], rstd(blk[b], eps)) for b in range(self.nb)",
        "mul_rn(blk[self.nb + b], rstd(blk[0], eps)) for b in range(self.nb)",
    ),
    # The two-shot owner reduce drops the last rank's chunk.
    "twoshot_drop": (
        "kimi_k31/k3/cutedsl/common/twoshot.py",
        "for slot in cutlass.range_constexpr(1, tp):\n            a, b = unpack_bf16(raw[slot, j])",
        "for slot in cutlass.range_constexpr(1, tp - 1):\n            a, b = unpack_bf16(raw[slot, j])",
    ),
    # CAP-128 #3 (all three are equivalent at batches <= 64, where the mailbox has 64 rows):
    # the kernels are built for 64-row buffers while the mailbox holds 128 rows (wrong slot stride);
    "cap64_build": (
        "kimi_k31/k3/cutedsl/oproj/prepared.py",
        "                    publish and kind != \"producer\",\n                    cap,\n",
        "                    publish and kind != \"producer\",\n                    64,\n",
    ),
    # the parity-1 view's multicast address is offset by a 64-row parity (a leftover 64-row constant);
    "parity64": (
        "kimi_k31/k3/cutedsl/oproj/prepared.py",
        "self.mc + parity * self.buffer[0].numel() * 2",
        "self.mc + parity * self.buffer[0, :, :64].numel() * 2",
    ),
    # the sentinel fill arms only rows 0-63 (rows 64-127 start un-armed).
    "fill64": (
        "kimi_k31/k3/cutedsl/oproj/prepared.py",
        "self.buffer.view(torch.int32).fill_(-0x80000000)",
        "self.buffer[:, :, :64].view(torch.int32).fill_(-0x80000000)",
    ),
}

# Tolerances, fixed before any run. Partials are ~N(0, 1) per element (x ~ N(0, 1),
# W ~ N(0, 1/sqrt(K))). The kernel's P_q can differ from the FP32 reference's by one
# BF16 rounding (FP32 order over K = 1536), i.e. <= 2^-8 |P_q|; the FP32 rank-order sum
# then adds ~1e-7 relative. prefix: atol 0.03 (2^-8 of a 4-sigma partial, x2) + 2^-7
# rel. out goes through softmax / RMS / fast exp and rsqrt (~1e-6 rel) and its own
# BF16 rounding: atol 0.03 x rms(out_ref) + 2^-6 rel. A lost partial or wrong block
# moves cells by O(1) relative.
PREFIX_TOL = (0.03, 2**-7)
OUT_TOL_ABS_SCALE, OUT_TOL_REL = 0.03, 2**-6


def layer_facts(layer_idx):
    write = layer_idx % BLOCK == 0
    prev = -(-layer_idx // BLOCK)
    return write, prev


def _cpu_gen(*key):
    """Same stream in every process (``hash`` of str is salted per process)."""
    g = torch.Generator()
    g.manual_seed(zlib.crc32(repr(key).encode()))
    return g


def _randn(shape, scale, g, device):
    return (torch.randn(shape, generator=g) * scale).to(torch.bfloat16).to(device)


def canon(t):
    """BF16 with -0.0 made +0.0 (the producers' "latent" policy)."""
    u = t.view(torch.int16)
    return torch.where(u == -0x8000, torch.zeros_like(u), u).view(torch.bfloat16)


def rank_weight(layer_idx, q, device):
    """o_proj weight [7168, 1536] of TP rank q (identical in every process)."""
    return _randn((HIDDEN, K), K**-0.5, _cpu_gen("w", layer_idx, q), device)


def rank_input(layer_idx, call, q, batch, device):
    return _randn((batch, K), 1.0, _cpu_gen("x", layer_idx, call, q), device)


def make_decoder(layer_idx, weight, device):
    write, prev = layer_facts(layer_idx)
    g = _cpu_gen("norms", layer_idx)
    ns = lambda w: SimpleNamespace(weight=w, variance_epsilon=EPS)  # noqa: E731
    return SimpleNamespace(
        self_attn=SimpleNamespace(o_proj=SimpleNamespace(weight=weight)),
        mlp_res_norm=ns((1 + 0.1 * torch.randn(HIDDEN, generator=g)).to(torch.bfloat16).to(device)),
        mlp_res_proj=SimpleNamespace(weight=_randn((1, HIDDEN), 0.02, g, device)),
        post_attention_layernorm=ns((1 + 0.1 * torch.randn(HIDDEN, generator=g)).to(torch.bfloat16).to(device)),
        layer_idx=layer_idx,
        prev_valid_blocks=prev,
        is_block_write_layer=write,
    )


def state_inputs(layer_idx, call, batch, device):
    """(prefix_old [B, H], bank [B, 8, H]), replicated over TP ranks. Bank block j has
    RMS 0.5 + 0.4 j (the residual stream grows with depth), so a logit that uses the
    wrong block's RMS moves the softmax by O(1)."""
    g = _cpu_gen("state", layer_idx, call)
    bank = _randn((batch, 8, HIDDEN), 1.0, g, "cpu").float()
    bank = bank * (0.5 + 0.4 * torch.arange(8, dtype=torch.float32)).view(1, 8, 1)
    return _randn((batch, HIDDEN), 1.0, g, device), bank.to(torch.bfloat16).to(device)


def partial_ref(x, w):
    return canon((x.float() @ w.float().T).to(torch.bfloat16))


def reference(decoder, partials, prefix_old, bank):
    """(prefix, out) of the stock method (module docstring)."""
    write, prev = layer_facts(decoder.layer_idx)
    nb, has_delta = prev + int(write), not write
    attn = torch.zeros_like(partials[0], dtype=torch.float32)
    for p in partials:
        attn = attn + p.float()
    if has_delta:
        attn = attn + prefix_old.float()
    prefix = attn.to(torch.bfloat16)
    w_out = decoder.post_attention_layernorm.weight.float()
    if nb == 0:
        x = prefix.float()
        out = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + EPS) * w_out
        return prefix, out.to(torch.bfloat16)
    cw = (decoder.mlp_res_norm.weight.float() * decoder.mlp_res_proj.weight.squeeze(0).float())
    cw = cw.to(torch.bfloat16).float()
    rows = torch.cat([bank[:, :nb].float(), prefix.float().unsqueeze(1)], dim=1)  # [B, nb+1, H]
    scores = (rows * cw).sum(-1) * torch.rsqrt(rows.pow(2).mean(-1) + EPS)
    probs = torch.softmax(scores, dim=-1)
    mix = (probs.unsqueeze(-1) * rows).sum(1)
    out = mix * torch.rsqrt(mix.pow(2).mean(-1, keepdim=True) + EPS) * w_out
    return prefix, out.to(torch.bfloat16)


def check_outputs(name, prefix, out, ref_prefix, ref_out):
    H.check(f"{name}.prefix", prefix, ref_prefix, *PREFIX_TOL)
    rms = float(ref_out.float().pow(2).mean().sqrt())
    H.check(f"{name}.out", out, ref_out, OUT_TOL_ABS_SCALE * rms, OUT_TOL_REL)


def check_rearmed(name, buffer, rows):
    """Every word of rows [0, rows) of every slot is the sentinel after the call."""
    words = buffer[:, :rows].contiguous().view(torch.int32)
    bad = int((words != SENTINEL_I32).sum())
    H.record({"kind": "check", "item": f"{name}.rearm", "violations": bad, "cells": words.numel()})
    if bad:
        raise AssertionError(f"{name}: {bad}/{words.numel()} mailbox words not re-armed")


class Simulated:
    """Producers of the simulated TP ranks + one consumer, over a PreparedOproj."""

    def __init__(self, module, decoder, mailbox, batches, mla, ranks, consumer_rank, device,
                 front=None, twoshot=None, deadline_ns=2_000_000_000):
        import cutlass

        from sgl_nvidia_mega.kimi_k31.k3.cutedsl.common.host import to_cute

        self.cutlass, self.to_cute = cutlass, to_cute
        self.decoder, self.mailbox, self.ranks = decoder, mailbox, tuple(ranks)
        self.layer = decoder.layer_idx
        self.device = device
        self.weights = {q: rank_weight(self.layer, q, self.device) for q in self.ranks}
        decoder.self_attn.o_proj.weight = self.weights[consumer_rank]
        self.prep = module.PreparedOproj(
            decoder, mailbox, batches, mla, front, twoshot, deadline_ns=deadline_ns
        )
        self.parity = self.layer % 2
        self.wviews = {q: to_cute(w) for q, w in self.weights.items()}

    def producers(self, xs, s):
        """Every simulated rank's producer (its partial into slot / owner chunks)."""
        cutlass = self.cutlass
        batch = next(iter(xs.values())).shape[0]
        code = self.prep.codes[batch][0]
        for q in self.ranks:
            x = self.to_cute(xs[q])
            if batch in self.prep.twoshots:
                recv, peers, _ = self.prep.twoshot[0]
                code(x, self.wviews[q], recv, peers, cutlass.Int32(q), s)
            else:
                box, mc, _ = self.prep.head
                code(x, self.wviews[q], box, mc, cutlass.Int32(q), s)

    def consumer(self, batch, prefix, blocks, out, s):
        from sgl_nvidia_mega.kimi_k31.k3.cutedsl.common.host import dynamic_view

        prep = self.prep
        codes = prep.codes[batch]
        front = (None, None)
        if prep.front is not None and batch in prep.front[2]:
            front = prep.front[:2]
        rows = (dynamic_view(prefix), dynamic_view(blocks), *prep.views)
        rows += (dynamic_view(out), *front, *prep.eps, s)
        if batch in prep.twoshots:
            _, gather, tail = prep.twoshot
            codes[1](gather, *rows[:-1], *tail, s)
        else:
            codes[1](prep.box, *rows)

    def run(self, xs, prefix, blocks, out):
        from sgl_nvidia_mega.kimi_k31.k3.cutedsl.common.host import stream

        s = stream(self.device)
        batch = out.shape[0]
        self.producers(xs, s)
        self.consumer(batch, prefix, blocks, out, s)


class TwoShotTP8:
    """``common.twoshot.TwoShot`` with TP8 geometry over a smaller group: owners that
    no GPU's consumer plays push into local stand-in receive buffers (``standins``) and
    their gather columns are written by ``inject``. ``owner_gpu[o]``: group rank of
    owner o or None."""

    def __init__(self, group, device, owner_gpu, cap=CAP, parities=2):
        import torch.distributed as dist
        import torch.distributed._symmetric_memory as symm_mem

        self.tp, self.width, self.owner_gpu = TP, HIDDEN, owner_gpu
        W = HIDDEN // TP
        bf16 = dict(dtype=torch.bfloat16, device=device)
        self.recv = symm_mem.empty((parities, TP, cap, W), **bf16)
        self.gather = symm_mem.empty((parities, 1, cap, HIDDEN), **bf16)
        handles = [symm_mem.rendezvous(t, group) for t in (self.recv, self.gather)]
        size = self.recv[0].numel() * 2
        ptrs = handles[0].buffer_ptrs
        self.standins = {
            o: torch.empty((parities, TP, cap, W), **bf16) for o in range(TP) if owner_gpu[o] is None
        }
        bases = []
        for p in range(parities):
            row = []
            for o in range(TP):
                if owner_gpu[o] is None:
                    row.append(self.standins[o].data_ptr() + p * size)
                else:
                    row.append(int(ptrs[owner_gpu[o]]) + p * size)
            bases.append(row)
        self.peers = torch.tensor(bases, dtype=torch.int64, device=device)
        self.gather_mc = int(handles[1].multicast_ptr or 0)
        if not self.gather_mc:
            raise RuntimeError("two-shot stand-in needs multicast memory")
        for t in (self.recv, self.gather, *self.standins.values()):
            t.view(torch.int32).fill_(SENTINEL_I32)
        torch.cuda.synchronize(device)
        dist.barrier(group=group)

    def views(self, parity):
        gather_mc = self.gather_mc + parity * self.gather[0].numel() * 2
        return self.recv[parity], self.peers[parity], self.gather[parity], gather_mc

    def inject(self, parity, partials, residual=None):
        """Gather columns of owners without a consumer: their reduce of ``partials``
        (FP32 rank order, + ``residual`` (the prefix of a ``has_delta`` layer), one BF16
        rounding, no -0)."""
        W = HIDDEN // TP
        batch = partials[0].shape[0]
        total = torch.zeros_like(partials[0], dtype=torch.float32)
        for p in partials:
            total = total + p.float()
        if residual is not None:
            total = total + residual.float()
        total = canon(total.to(torch.bfloat16))
        for o, gpu in enumerate(self.owner_gpu):
            if gpu is None:
                self.gather[parity, 0, :batch, o * W : (o + 1) * W].copy_(total[:, o * W : (o + 1) * W])


def sms():
    return torch.cuda.get_device_properties(torch.cuda.current_device()).multi_processor_count


def write_json(path, obj):
    with open(path, "w") as f:
        json.dump(obj, f, indent=1)


def env_flag(name):
    return os.environ.get(name, "") not in ("", "0")


def device_attributes(index=None):
    """Driver attributes that decide co-residency (cuDeviceGetAttribute)."""
    import cuda.bindings.driver as cuda

    index = torch.cuda.current_device() if index is None else index
    a = cuda.CUdevice_attribute
    names = {
        "smem_per_sm": a.CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_MULTIPROCESSOR,
        "smem_per_block_optin": a.CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK_OPTIN,
        "smem_per_block": a.CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK,
        "regs_per_sm": a.CU_DEVICE_ATTRIBUTE_MAX_REGISTERS_PER_MULTIPROCESSOR,
        "threads_per_sm": a.CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_MULTIPROCESSOR,
        "blocks_per_sm": a.CU_DEVICE_ATTRIBUTE_MAX_BLOCKS_PER_MULTIPROCESSOR,
        "sms": a.CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT,
        "clock_khz": a.CU_DEVICE_ATTRIBUTE_CLOCK_RATE,
        "mem_clock_khz": a.CU_DEVICE_ATTRIBUTE_MEMORY_CLOCK_RATE,
        "l2": a.CU_DEVICE_ATTRIBUTE_L2_CACHE_SIZE,
        "multicast": a.CU_DEVICE_ATTRIBUTE_MULTICAST_SUPPORTED,
        "cluster_launch": a.CU_DEVICE_ATTRIBUTE_CLUSTER_LAUNCH,
    }
    # Large-SMEM mode limit (driver attribute, queried by name).
    for key, name in (("smem_oversized_per_block", "CU_DEVICE_ATTRIBUTE_MAX_OVERSIZED_SHARED_MEMORY_PER_BLOCK"),):
        if hasattr(a, name):
            names[key] = getattr(a, name)
    out = {}
    for key, attr in names.items():
        status, value = cuda.cuDeviceGetAttribute(attr, index)
        out[key] = int(value) if status == cuda.CUresult.CUDA_SUCCESS else f"err {status}"
    return out
