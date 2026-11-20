# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the SGLang project
"""Stage 3a: the target argmax gather + DFlash greedy accept as one megakernel kernel.

Today (dflash_worker_v2): ``softcap_row_argmax`` per vocab shard -> [rows, 2] FP32 (the
softcapped max, first argmax + col offset); two NCCL all_gathers; ``argmax`` over ranks
(the first max rank) and a gather -> target_predict; ``_dflash_accept_bonus_contig_kernel``
-> accept / commit lens, bonus, out_tokens, new_seq_lens; optionally
``apply_dflash_simulated_acceptance``. The lm_head GEMM and the per-shard Triton argmax
stay as they are (bitwise target logits). ``ArgmaxAccept`` replaces the rest:

1. threads 0-3 of request b multicast this rank's (value, id) of rows 8b.. into every
   rank's argmax box [TP, rows, 2] at [rank] (one hop; -0.0 published as +0.0);
2. threads 0-63 poll the 16 ranks' row pairs, re-arm, fence;
3. thread r < 8 picks row r's first max rank (NaN counts as the maximum, the first NaN
   wins: torch.argmax over the gathered values) -> target_predict;
4. thread 0 runs the accept of ``_dflash_accept_bonus_contig_kernel`` (leading matches of
   candidates[:, 1:] with target[:, :-1], bonus = target[accept], commit = accept + 1),
   or the simulated acceptance when ``sim_commit[0] > 0`` (one forced commit length per
   round for the batch, as ``apply_dflash_simulated_acceptance``)."""

from functools import cache

import cutlass
import cutlass.cute as cute
import torch
from cutlass import Float32, Int32, Int64, Uint32
from cutlass.cute.runtime import make_fake_stream

from ..k3.common.host import dynamic_view, fake, stream, to_cute
from ..k3.common.mailbox import poll, rearm, rearm_fence
from ..k3.common.primitives import (
    alloc,
    at,
    bits_float,
    float_bits,
    launch,
    multicast_words,
    store_words,
)

TP_MAX, BLOCK, FIXED_TOKEN = 16, 8, 100


@cute.jit
def first_max(v_new, v_best):
    """torch.argmax order over ranks: NaN beats everything (first NaN kept), then a
    strictly larger value replaces the best."""
    new_nan = v_new != v_new
    best_nan = v_best != v_best
    return (new_nan & (~best_nan)) | ((~best_nan) & (~new_nan) & (v_new > v_best))


class ArgmaxAccept:
    """Grid (B,), 128 threads. Arguments: local [rows, 2] FP32 (this rank's
    softcap_row_argmax), box [TP, rows, 2] int32 (this GPU's view), mc, rank,
    candidates [B, 8] int64, prefix_lens [B] int64, sim_commit [1] int32, then outputs
    target [B, 8] int64, accept_len [B] int32, commit_lens [B] int32, bonus [B] int64,
    out_tokens [B, 8] int64, new_seq_lens [B] int64."""

    def __init__(self, tp, *, sim_real_token=True, local=False, deadline_ns=0):
        if tp > TP_MAX:
            raise ValueError(f"ArgmaxAccept serves TP <= {TP_MAX}")
        self.tp, self.sim_real_token = tp, sim_real_token
        self.local, self.deadline_ns = local, deadline_ns

    @cute.jit
    def __call__(self, *args):
        launch(self.kernel, args, (args[4].shape[0], 1, 1), 128)

    @cute.kernel
    def kernel(self, args):
        local, box, mc, rank, candidates, prefix_lens, sim_commit = args[0:7]
        target, accept_out, commit_out, bonus_out, out_tokens, new_seq = args[7:13]
        req, _, _ = cute.arch.block_idx()
        tid, _, _ = cute.arch.thread_idx()
        smem = cutlass.utils.SmemAllocator()
        vals = alloc(smem, Float32, (TP_MAX, BLOCK))
        ids = alloc(smem, Int32, (TP_MAX, BLOCK))
        top = alloc(smem, Int32, BLOCK)
        row0 = req * BLOCK
        cute.arch.griddepcontrol_wait()
        cute.arch.griddepcontrol_launch_dependents()
        if tid < BLOCK // 2:  # 1. publish rows 2t, 2t + 1 of this rank
            v0 = Float32(local[row0 + 2 * tid, 0])
            i0 = Float32(local[row0 + 2 * tid, 1])
            v1 = Float32(local[row0 + 2 * tid + 1, 0])
            i1 = Float32(local[row0 + 2 * tid + 1, 1])
            words = [
                _canon(v0),
                Uint32(float_bits(i0)),
                _canon(v1),
                Uint32(float_bits(i1)),
            ]
            if cutlass.const_expr(self.local):
                store_words(at(box, rank, row0 + 2 * tid, 0), tuple(words))
            else:
                offset = cute.crd2idx((rank, row0 + 2 * tid, 0), box.layout)
                multicast_words(mc + Int64(offset) * 4, tuple(words))
        if tid < self.tp * (BLOCK // 2):  # 2. poll rank s, row pair p
            s, p = tid // (BLOCK // 2), tid % (BLOCK // 2)
            pointer = at(box, s, row0 + 2 * p, 0)
            got = poll(pointer, 0, 1, self.deadline_ns)
            for h in cutlass.range_constexpr(2):
                vals[s, 2 * p + h] = bits_float(got[0, 2 * h])
                ids[s, 2 * p + h] = Int32(bits_float(got[0, 2 * h + 1]))
            rearm(pointer)
            rearm_fence()
        cute.arch.sync_threads()
        if tid < BLOCK:  # 3. first max rank of row tid
            best_v, best_i = vals[0, tid], ids[0, tid]
            for s in cutlass.range_constexpr(1, self.tp):
                if first_max(vals[s, tid], best_v):
                    best_v, best_i = vals[s, tid], ids[s, tid]
            top[tid] = best_i
            target[req, tid] = Int64(best_i)
        cute.arch.sync_threads()
        if tid == 0:  # 4. accept (or the simulated acceptance)
            accept, live = Int32(0), Int32(1)
            for c in cutlass.range_constexpr(BLOCK - 1):
                match = Int32(candidates[req, c + 1] == Int64(top[c]))
                accept = accept + (live & match)
                live = live & match
            forced = sim_commit[0]
            simulated = forced > Int32(0)
            if simulated:
                accept = forced - Int32(1)
            commit = accept + Int32(1)
            bonus = Int64(top[accept])
            if cutlass.const_expr(not self.sim_real_token):
                if simulated:
                    bonus = Int64(FIXED_TOKEN)
            accept_out[req] = accept
            commit_out[req] = commit
            bonus_out[req] = bonus
            new_seq[req] = prefix_lens[req] + Int64(commit)
            for c in cutlass.range_constexpr(BLOCK):
                token = Int64(0)
                if cutlass.const_expr(c < BLOCK - 1):
                    token = candidates[req, c + 1]
                if c == accept:
                    token = bonus
                if cutlass.const_expr(c < BLOCK - 1):
                    # The simulated real-draft path zeroes everything past the bonus.
                    if simulated & (c > accept):
                        token = Int64(0)
                if cutlass.const_expr(not self.sim_real_token):
                    if simulated:
                        token = Int64(FIXED_TOKEN)
                out_tokens[req, c] = token


@cute.jit
def _canon(value):
    """FP32 bits with -0.0 as +0.0 (the mailbox sentinel is 0x80000000)."""
    word = Uint32(float_bits(Float32(value)))
    if word == Uint32(0x80000000):
        word = Uint32(0)
    return word


@cache
def compile_argmax_accept(rows, tp, requests, sim_real_token, local, deadline_ns):
    local_t = fake((rows, 2), (2, 1), torch.float32, 4)
    box = fake((tp, rows, 2), (rows * 2, 2, 1), torch.int32, 4)
    cand = fake((requests, BLOCK), (BLOCK, 1), torch.int64, 8)
    vec64 = fake((requests,), (1,), torch.int64, 8)
    vec32 = fake((requests,), (1,), torch.int32, 4)
    sim = fake((1,), (1,), torch.int32, 4)
    args = (local_t, box, cutlass.Int64(0), cutlass.Int32(0), cand, vec64, sim)
    args += (cand, vec32, vec32, vec64, cand, vec64)
    kernel = ArgmaxAccept(
        tp, sim_real_token=sim_real_token, local=local, deadline_ns=deadline_ns
    )
    return cute.compile(kernel, *args, make_fake_stream())


class DraftAccept:
    """Serving host of ``ArgmaxAccept`` on a ``DraftComm``'s argmax box (one hop per round,
    in the round's program order like the draft hops, so the round's later hops order the
    box's reuse: serving parity 0, an eager warm-up 1; back-to-back unit runs alternate).
    Compiled per request count at construction."""

    def __init__(self, comm, shapes, *, sim_real_token=True, deadline_ns=0):
        self.comm, self.shapes = comm, shapes
        self.rows, self.requests = shapes.tokens, shapes.requests
        self.code = compile_argmax_accept(
            self.rows,
            shapes.tp,
            self.requests,
            bool(sim_real_token),
            False,
            deadline_ns,
        )

    def run(self, local, candidates, prefix_lens, sim_commit, out, parity=0):
        """``local`` [rows, 2] FP32 (this rank's ``softcap_row_argmax``); ``candidates``
        [B, 8] int64 (the round's draft block, column 0 the last committed token);
        ``prefix_lens`` [B] int64; ``sim_commit`` [1] int32 (0 = native greedy accept, else the
        round's forced commit length); ``out`` = (target_predict [B, 8] int64, accept_len [B]
        int32, commit_lens [B] int32, bonus [B] int64, out_tokens [B, 8] int64, new_seq_lens
        [B] int64)."""
        target, accept_len, commit_lens, bonus, out_tokens, new_seq_lens = out
        b, rows = self.requests, self.rows
        expect = (
            (local, (rows, 2), torch.float32),
            (candidates, (b, BLOCK), torch.int64),
            (prefix_lens, (b,), torch.int64),
            (sim_commit, (1,), torch.int32),
            (target, (b, BLOCK), torch.int64),
            (accept_len, (b,), torch.int32),
            (commit_lens, (b,), torch.int32),
            (bonus, (b,), torch.int64),
            (out_tokens, (b, BLOCK), torch.int64),
            (new_seq_lens, (b,), torch.int64),
        )
        for tensor, shape, dtype in expect:
            if (
                tuple(tensor.shape) != shape
                or tensor.dtype != dtype
                or not tensor.is_contiguous()
            ):
                raise ValueError(
                    f"ArgmaxAccept argument {tuple(tensor.shape)} {tensor.dtype}, "
                    f"expected contiguous {shape} {dtype}"
                )
        box = self.comm.argmax[parity]
        mc = self.comm.argmax_mc + parity * box.numel() * 4
        args = (
            to_cute(local, 4),
            to_cute(box, 4),
            cutlass.Int64(mc),
            cutlass.Int32(self.shapes.rank),
        )
        args += (
            to_cute(candidates, 8),
            to_cute(prefix_lens, 8),
            to_cute(sim_commit, 4),
        )
        args += (to_cute(target, 8), to_cute(accept_len, 4), to_cute(commit_lens, 4))
        args += (to_cute(bonus, 8), to_cute(out_tokens, 8), to_cute(new_seq_lens, 8))
        self.code(*args, stream(local.device))


# The DFLASH sampled accept's rank-0 decision (accept_len, bonus) made every rank's in one multicast hop: the
# rank-0 NCCL broadcasts of SGLANG_SPEC_TP_SYNC's dflash-selector site, without NCCL. A dedicated box, not
# CustomAllReduceV2 (the target's comm, which the post-verify window may share with the aux side stream).
BCAST_MAGIC = 0x4F41  # word 3 of a published row (never the 0x80000000 sentinel)


class Rank0Broadcast:
    """Grid (B,), 32 threads. Arguments: accept_len [B] int32 and bonus [B] int64 (in place: rank 0's values on
    every rank afterwards), box [rows, 4] int32 (this GPU's view of the call's parity box), mc, rank. Rank 0's
    thread 0 multicasts (accept_len, bonus lo, bonus hi, magic) of request b into every rank's row b; every rank
    polls its row b, takes the values and re-arms the row."""

    def __init__(self, *, local=False, deadline_ns=0):
        self.local, self.deadline_ns = local, deadline_ns

    @cute.jit
    def __call__(self, *args):
        launch(self.kernel, args, (args[0].shape[0], 1, 1), 32)

    @cute.kernel
    def kernel(self, args):
        accept_len, bonus, box, mc, rank = args[0:5]
        req, _, _ = cute.arch.block_idx()
        tid, _, _ = cute.arch.thread_idx()
        cute.arch.griddepcontrol_wait()
        cute.arch.griddepcontrol_launch_dependents()
        if tid == 0:
            if rank == Int32(0):
                value = bonus[req]
                words = (
                    Uint32(accept_len[req]),
                    Uint32(value & Int64(0xFFFFFFFF)),
                    Uint32(value >> Int64(32)),
                    Uint32(BCAST_MAGIC),
                )
                if cutlass.const_expr(self.local):
                    store_words(at(box, req, 0), words)
                else:
                    offset = cute.crd2idx((req, 0), box.layout)
                    for w in cutlass.range_constexpr(
                        4
                    ):  # multimem.st has no vector form for b32
                        multicast_words(
                            mc + (Int64(offset) + w) * 4, (words[w],), kind="b32"
                        )
            pointer = at(box, req, 0)
            got = poll(pointer, 0, 1, self.deadline_ns)
            accept_len[req] = Int32(got[0, 0])
            bonus[req] = Int64(got[0, 1]) | (Int64(got[0, 2]) << Int64(32))
            rearm(pointer)
            rearm_fence()


@cache
def compile_rank0_broadcast(rows, local, deadline_ns):
    # Natural alignment only, so every rank accepts any contiguous accept_len / bonus (a rank-local fallback
    # would leave its peers polling).
    box = fake((rows, 4), (4, 1), torch.int32, 16)
    lens, bonus = (
        fake((cute.sym_int(),), (1,), torch.int32, 4),
        fake((cute.sym_int(),), (1,), torch.int64, 8),
    )
    args = (lens, bonus, box, cutlass.Int64(0), cutlass.Int32(0))
    return cute.compile(
        Rank0Broadcast(local=local, deadline_ns=deadline_ns), *args, make_fake_stream()
    )


class Rank0Bcast:
    """Host of ``Rank0Broadcast`` with its own symm-mem box [2 parity, rows, 4] int32 on ``group`` (collective
    construction and calls: every rank, same order). ``run`` alternates the parity unless one is given; in serving
    every round's collectives also order a slot's reuse (a rank re-arms its row before its next collective)."""

    def __init__(self, group, device, rank, *, rows=256, deadline_ns=0):
        import torch.distributed as dist
        import torch.distributed._symmetric_memory as symm_mem

        self.rows, self.rank, self.device = rows, rank, device
        self.box = symm_mem.empty((2, rows, 4), dtype=torch.int32, device=device)
        self.mc = int(symm_mem.rendezvous(self.box, group).multicast_ptr or 0)
        if not self.mc:
            raise RuntimeError("the rank-0 accept broadcast needs multicast memory")
        self.box.fill_(-0x80000000)
        torch.cuda.synchronize(device)
        dist.barrier(group=group)  # every box armed before rank 0's first publish
        self.code = compile_rank0_broadcast(rows, False, deadline_ns)
        self.parity, self.calls = 0, 0

    def run(self, accept_len, bonus, parity=None):
        """``accept_len`` [B] int32, ``bonus`` [B] int64, contiguous, B <= rows: rank 0's values on every rank."""
        b = accept_len.shape[0]
        if (
            accept_len.dtype != torch.int32
            or bonus.dtype != torch.int64
            or accept_len.dim() != 1
            or tuple(bonus.shape) != (b,)
            or not accept_len.is_contiguous()
            or not bonus.is_contiguous()
            or not 0 < b <= self.rows
        ):
            raise ValueError(
                f"rank-0 broadcast of accept_len {tuple(accept_len.shape)} {accept_len.dtype}, "
                f"bonus {tuple(bonus.shape)} {bonus.dtype} (rows {self.rows})"
            )
        if parity is None:
            parity, self.parity = self.parity, self.parity ^ 1
        box = self.box[parity]
        mc = self.mc + parity * box.numel() * 4
        self.code(
            dynamic_view(accept_len, 4),
            dynamic_view(bonus, 8),
            to_cute(box, 16),
            cutlass.Int64(mc),
            cutlass.Int32(self.rank),
            stream(accept_len.device),
        )
        self.calls += 1
