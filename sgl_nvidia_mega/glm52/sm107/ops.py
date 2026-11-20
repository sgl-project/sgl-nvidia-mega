# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

from functools import cache

import cutlass as c
import cutlass.cute as cute
import torch
from cutlass.cute.runtime import make_fake_stream, make_fake_tensor


def at(t, offset):
    return cute.make_tensor(t.iterator + offset, cute.make_layout((1,)))

def div64(v):
    return (v + ((v >> 31) & c.Int32(63))) >> 6

class FillStepAll:
    @cute.jit
    def __call__(self, r2t, req, pos, bt, kvl, stream):
        self.kernel(r2t, req, pos, bt, kvl).launch(grid=(24, 1, 1), block=(256, 1, 1), stream=stream)

    @cute.kernel
    def kernel(self, r2t, req, pos, bt, kvl):
        tid, _, _ = cute.arch.thread_idx()
        cta, _, _ = cute.arch.block_idx()
        last = c.Int64(pos[7])
        npg = (last >> 6) + 1
        rows = c.Int64(bt.shape[0])
        if npg > rows:
            npg = rows
        row = c.Int64(req[0]) * c.Int64(r2t.stride[0])
        k = c.Int64(cta) * 256 + c.Int64(tid)
        while k < npg:
            at(bt, k)[0] = div64(c.Int32(at(r2t, row + k * 64)[0]))
            k = k + 24 * 256
        if (cta == 0) & (tid == 0):
            kvl[0] = c.Int32(last + 1) if last + 1 > 8 else c.Int32(8)


class PostVerify:
    @cute.jit
    def __call__(self, acc, na, am, cand, seq_lens, sim_l, predict, ai, al, nsl, at_, bonus, s0_acc, s0_na, sel,
                 stream):
        self.kernel(acc, na, am, cand, seq_lens, sim_l, predict, ai, al, nsl, at_, bonus, s0_acc, s0_na,
                    sel).launch(grid=(1, 1, 1), block=(32, 1, 1), stream=stream)

    @cute.kernel
    def kernel(self, acc, na, am, cand, seq_lens, sim_l, predict, ai, al, nsl, at_, bonus, s0_acc, s0_na, sel):
        tid, _, _ = cute.arch.thread_idx()
        if tid == 0:
            p, m = [], []
            for i in c.range_constexpr(8):
                v = c.Int32(acc[i])
                p.append(v if v > 0 else c.Int32(0))
                m.append(c.Int32(am[i]))
            sim = sim_l > 0
            n = c.Int32(sim_l) if sim else c.Int32(na[0])
            for i in c.range_constexpr(7):
                p[i] = c.Int32(cand[i + 1]) if sim & (i < n - 1) else p[i]
            for i in c.range_constexpr(8):
                p[i] = m[i] if sim & (i == n - 1) else p[i]
            b = p[0]
            for j in c.range_constexpr(1, 8):
                b = p[j] if n - 1 == j else b
            for i in c.range_constexpr(8):
                predict[i] = p[i]
                ai[0, i] = c.Int32(i) if i < n else c.Int32(-1)
                at_[0, i] = p[i] if i < n else p[7]
                s0_acc[i] = p[i]
            al[0] = n
            bonus[0] = b
            s0_na[0] = n
            nsl[0] = c.Int64(seq_lens[0]) + c.Int64(n)
            s = c.Int64(n) - 1
            sel[0] = c.Int64(0) if s < 0 else (c.Int64(7) if s > 7 else s)


class DraftPages:
    @cute.jit
    def __call__(self, r2t, req, sl, bt, stream):
        self.kernel(r2t, req, sl, bt).launch(grid=(1, 1, 1), block=(32, 1, 1), stream=stream)

    @cute.kernel
    def kernel(self, r2t, req, sl, bt):
        tid, _, _ = cute.arch.thread_idx()
        if tid < 8:
            page = (c.Int64(sl[0]) + c.Int64(tid)) >> 6
            if (page >= 0) & (page < c.Int64(bt.shape[0])):
                row = c.Int64(req[0]) * c.Int64(r2t.stride[0])
                at(bt, page)[0] = div64(c.Int32(at(r2t, row + page * 64)[0]))


class ChainTree:
    @cute.jit
    def __call__(self, bonus, drafts, sl, mask, mode, positions, rbuf, draft_out, stream):
        self.kernel(bonus, drafts, sl, mask, mode, positions, rbuf, draft_out).launch(
            grid=(1, 1, 1), block=(32, 1, 1), stream=stream)

    @cute.kernel
    def kernel(self, bonus, drafts, sl, mask, mode, positions, rbuf, draft_out):
        tid, _, _ = cute.arch.thread_idx()
        if tid < 8:
            i = c.Int64(tid)
            s = c.Int64(sl[0])
            base = (s + 8) * i + s if mode == 0 else 8 * i
            for k in c.range_constexpr(8):
                at(mask, base + k)[0] = c.Uint8(1) if k <= i else c.Uint8(0)
            positions[tid] = s + i
            rbuf[tid] = i
            rbuf[8 + tid] = i + 1 if i < 7 else c.Int64(-1)
            rbuf[16 + tid] = c.Int64(-1)
            if tid == 0:
                draft_out[0] = c.Int64(bonus[0])
            else:
                draft_out[tid] = c.Int64(drafts[tid - 1])


@cache
def _compiled(name, dtype=None):
    t = {torch.int32: c.Int32, torch.int64: c.Int64, torch.uint8: c.Uint8}

    def fake(dtype, shape, stride=None):
        shape = tuple(cute.sym_int() if s is None else s for s in shape)
        stride = stride if stride is not None else (1,)
        return make_fake_tensor(t[dtype], shape, stride, assumed_align=dtype.itemsize)

    i32, i64 = torch.int32, torch.int64
    r2t = fake(i32, (None, None), (cute.sym_int(), 1))
    if name == "fill_step_all":
        fn, args = FillStepAll(), (r2t, fake(i64, (None,)), fake(i64, (None,)), fake(i32, (None,)), fake(i32, (1,)))
    elif name == "post_verify":
        v8, v1, m8 = fake(i32, (8,)), fake(i32, (1,)), fake(i32, (1, 8), (8, 1))
        fn, args = PostVerify(), (v8, v1, v8, fake(dtype, (None,)), fake(i64, (1,)), c.Int32(0), v8, m8, v1,
                                  fake(i64, (1,)), m8, v1, v8, v1, fake(i64, (1,)))
    elif name == "draft_pages":
        fn, args = DraftPages(), (r2t, fake(i64, (None,)), fake(i64, (None,)), fake(i32, (None,)))
    else:
        fn, args = ChainTree(), (fake(dtype, (None,)), fake(i64, (7,)), fake(i64, (None,)), fake(torch.uint8, (None,)),
                                 c.Int32(0), fake(i64, (8,)), fake(i64, (24,)), fake(i64, (8,)))
    major, minor = torch.cuda.get_device_capability()
    stream = make_fake_stream(use_tvm_ffi_env_stream=True)
    return cute.compile(fn, *args, stream, options=f"--gpu-arch=sm_{major}{minor}a --enable-tvm-ffi")


def fill_step_all(r2t, req, pos, bt, kvl):
    _compiled("fill_step_all")(r2t, req, pos, bt, kvl)


def post_verify(acc, na, am, cand, seq_lens, sim_l, predict, ai, al, nsl, at_, bonus, s0_acc, s0_na, sel):
    _compiled("post_verify", cand.dtype)(acc, na, am, cand, seq_lens, int(sim_l), predict, ai, al, nsl, at_, bonus,
                                         s0_acc, s0_na, sel)


def draft_pages(r2t, req, sl, bt):
    _compiled("draft_pages")(r2t, req, sl, bt)


def chain_tree(bonus, drafts, sl, mask, mode, positions, rbuf, draft_out):
    _compiled("chain_tree", bonus.dtype)(bonus.view(-1), drafts.view(-1), sl.view(-1), mask.view(-1).view(torch.uint8),
                                         int(mode), positions, rbuf.view(-1), draft_out)


def warm_up(device):
    """Compiles and launches every variant once before graph capture, so no module is loaded while capturing."""
    i32, i64 = dict(dtype=torch.int32, device=device), dict(dtype=torch.int64, device=device)
    r2t = torch.zeros((2, 128), **i32)
    z64, z32 = torch.zeros((8,), **i64), torch.zeros((8,), **i32)
    fill_step_all(r2t, z64[:1], z64, torch.zeros((4,), **i32), torch.zeros((1,), **i32))
    draft_pages(r2t, z64[:1], z64[:1], torch.zeros((4,), **i32))
    for dt in (torch.int32, torch.int64):
        post_verify(torch.ones((8,), **i32), torch.ones((1,), **i32), z32, torch.zeros((8,), dtype=dt, device=device),
                    z64[:1], 0, torch.empty((8,), **i32), torch.empty((1, 8), **i32), torch.empty((1,), **i32),
                    torch.empty((1,), **i64), torch.empty((1, 8), **i32), torch.empty((1,), **i32),
                    torch.empty((8,), **i32), torch.empty((1,), **i32), torch.empty((1,), **i64))
        chain_tree(torch.zeros((1,), dtype=dt, device=device), torch.zeros((7,), **i64), z64[:1],
                   torch.zeros((64,), dtype=torch.bool, device=device), 1, torch.empty((8,), **i64),
                   torch.empty((3, 1, 8), **i64), torch.empty((8,), **i64))
