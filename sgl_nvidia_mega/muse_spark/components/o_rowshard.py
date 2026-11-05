# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Component ``o_rowshard`` (env ``MUSE_VR_O_ROWSHARD=1``, default off): the sequence-parallel prefill output projection
computed per row shard.

Under ``--enable-layernorm-sp`` every Muse Spark attention layer of a sequence-parallel prefill pass ends with
``layernorm_sp.row_parallel_gbar_matmul``: each rank multiplies its head slice of the attention output (all ``M`` token
rows, ``K / tp`` input features) by its slice of ``o_proj`` and the ``[M, hidden]`` partial sums are reduce-scattered, so
each rank keeps ``M / tp`` rows of the result. Per rank that moves ``(tp - 1) / tp * M * hidden`` partial-sum elements.

This component computes the same rows the other way round. An all-to-all moves each rank's attention-output rows to
the rank that keeps them (every rank then holds all ``K`` input features of its own ``M / tp`` rows), and one GEMM with
the layer's full weight gives those rows of the output. Per rank that moves ``(tp - 1) / tp * M / tp * K`` activation
elements (for Muse Spark ``K`` = hidden, so ``1 / tp`` of the partial-sum bytes) plus the other ranks' weight slices,
which are static: the previous layer's call starts their copy on a side stream, and the next column-parallel projection
of the same pass (``column_parallel_g_matmul``, the QKV projection, in the same CUDA-graph segment as that call) makes
the main stream wait for it. Layer 0 of a pass copies its weight in its own call.

Data movement (no kernels of its own): copy-engine 2-D copies (``cuMemcpy2DAsync``) pull the peer rows and weight
slices out of symmetric-memory buffers, as SGLang's all-gather + GEMM pulls peer shards; per row chunk a stream write
(``cuStreamWriteValue32``) publishes the pulled rows and the main stream waits for it (``cuStreamWaitValue32``) before
that chunk's GEMM, so later chunks' copies overlap earlier chunks' GEMMs. The staging buffer is reused only after a
symmetric-memory barrier that follows every rank's previous pulls (each rank's main stream waits for its copy stream
before the call returns). The same ordering pattern as SGLang's ``_all_gather_matmul_cublas``.

Numerics: the products are the stock ones, but each output element is reduced over all ``K`` features in one fp32
accumulation (cuBLAS) instead of per-rank partial sums rounded to bf16 and then reduced: a reduction-order change, not
bitwise to the stock path. ``tests/dist_o_rowshard.py`` compares both against an fp64 reference.

MXFP8 form (only on passes where ``fp8_prefill`` runs the ``o`` group): each rank quantizes its attention-output rows
to MXFP8 (``mxfp8_ops``) in a staging layout padded per destination rank to whole 128-row scale tiles, and its own
weight slice of the layer (just in time, into a two-slot symmetric store, one layer ahead); the all-to-all and the
weight gather then move E4M3 bytes plus the scale tiles (2-D copies of whole tiles), and one MXFP8 GEMM per row chunk
gives the rank's rows.

Memory: at load the output projections' weight slices move into one symmetric-memory buffer per rank and the
parameters are rebound to views of it (the same values; the old storage is released before the KV cache is sized).
Extra: two full-weight slots, a staging and a receive buffer for the largest prefill pass.

Engaged per call only for a tagged Muse Spark ``o_proj`` on a sequence-parallel pass with a bias-free bf16 2-D input of
the slice width, padded rows at least ``MUSE_VR_O_ROWSHARD_MIN_ROWS`` (default 8192; must be a captured prefill bucket,
checked at first use, as ``sp_gate`` checks its threshold) and at most the buffers' capacity
(``MUSE_VR_O_ROWSHARD_CAP_ROWS``, default the largest prefill bucket), outside compilation. Every other call runs the
stock function, counted ``stock:<reason>``. ``MUSE_VR_O_ROWSHARD_CHUNKS`` (default 4) row chunks per call;
``MUSE_VR_O_ROWSHARD_PREFETCH=0`` makes every call copy its own weights (the prefetch A/B). Engagement needs the
sha256 pins of the SGLang sources whose call structure it replaces.

Pull form (``MUSE_VR_O_ROWSHARD_FORM=pull``; default ``rowshard`` = everything above): only the MXFP8 form, only on
passes where ``fp8_prefill`` runs the ``o`` group; every other call (BF16 passes included) runs the stock function,
counted ``stock:pass``, and no BF16 buffers are allocated (the weights stay where they are). Per call: the rank's
attention-output rows are quantized (one grouped launch of ``mxfp8_fq`` when ``MUSE_VR_FP8_PREFILL_FQ`` is set, else the
stock quantizer) straight into a symmetric staging buffer, one block per destination rank (whole 128-row scale tiles;
padding rows are quantized zero rows); then one kernel (``ops.a2a_pull``) copies, from every rank, the block staged for
this rank into the full-width operand (E4M3 values and scale tiles, bytes unchanged), and one MXFP8 GEMM with the full
weight gives the rank's rows. Two bounded barriers per call (``ops.a2a_pull.barrier``: release / acquire at system scope,
at most ``MUSE_VR_O_ROWSHARD_BARRIER_MAX_POLLS`` attempts per wait, then a CUDA error) order it: before the staging
writes (every rank's previous pulls from this rank's staging and weight slots are complete) and before the pull (every
rank's blocks and weight slices are staged). Weights (``MUSE_VR_O_ROWSHARD_WEIGHTS``): ``jit`` (default) quantizes the
rank's own weight slice one layer ahead into a two-slot symmetric store and gathers every rank's slice into a full-width
slot on a side stream (``MUSE_VR_O_ROWSHARD_PREFETCH=1``: started by the previous layer's call and joined by the next
column-parallel projection, as above; ``0``: in the call itself, concurrent with the activation pull);
``MUSE_VR_O_ROWSHARD_WGATHER`` = ``ce`` (default: copy-engine 2-D copies) or ``pull`` (the pull kernel on the side
stream). ``resident`` builds the full-width MXFP8 weight of every layer at load through the same quantize + pull (about
``hidden * hidden * 1.03`` bytes per layer per rank, allocated before the KV cache is sized). A call is taken only when
its GEMM runs on fp8_prefill's pinned runner (``MUSE_VR_FP8_PREFILL_TACTICS`` + ``_PIN_RUNNER``, a configuration launched
without programmatic dependent launch); otherwise it is counted ``stock:unpinned``. The GEMM numerics are those
of the MXFP8 form above (one fp32 accumulation over all K); the pulled operand bytes equal the quantizer's output.
"""

from __future__ import annotations

import bisect
import functools
import logging
import os
import threading

import torch

from . import bed

# The SGLang names this module reads (test/unit/test_contract.py pins them against the supported SGLang)
SGLANG_READS = {
    "srt/model_executor/model_runner.py": ("ModelRunner",),
    "srt/model_executor/cuda_graph_config.py": ("Backend",),
    "srt/runtime_context.py": ("get_exec", "get_parallel"),
}


logger = logging.getLogger("muse_vr_plugin")

NAME = "o_rowshard"
DEFAULT_MIN_ROWS = 8192
DEFAULT_CHUNKS = 4
FORMS = ("rowshard", "pull")
PULL_WEIGHTS = ("jit", "resident")
WGATHERS = ("ce", "pull")
DEFAULT_BARRIER_MAX_POLLS = 1 << 27
MIN_MAX_POLLS, MAX_MAX_POLLS = 1 << 10, (1 << 32) - 1

# sha256 of the SGLang sources (306ccf6) whose structure this component relies on: the SP linear entry points and the
# stock GEMM-RS output contract (padded rows, [padded / tp, hidden] result), the Muse Spark o_proj participants.
SGLANG_PINS = {
    "sglang/srt/layers/layernorm_sp.py": "bc814d983b945e5bdd41db436966eb4b3fee4c410bb1dcccfc98857d4b026bda",
    "sglang/srt/layers/linear.py": "3ce31df0a8aab09ed8afa0ada8c0ce30ca207a748c02c3073d7f53648b9b7c3c",
    "sglang/srt/models/muse_spark.py": (
        "8bafbb8802fb0d9f81c190e87b9012fca546182be08bef6c8b3db814d27bfe19",
        "c0db42467a960c509b658c3c40570ee889f73d41b52cdbd80b9c291c1792a33f",
    ),
    "sglang/kernels/ops/communication/sp_collective_gemm.py": "f1cc99967104416a31fe7c99a710a9ccf8024ac1579c08579f11ac68a476a836",
}

_lock = threading.Lock()
_state: dict = {
    "engaged": None,
    "ctx": None,
    "orig_gbar": None,
    "orig_g": None,
    "checked": None,
    "summary": None,
}


def _ceil_align(x: int, a: int) -> int:
    return (x + a - 1) // a * a


def chunk_rows(rows: int, chunks: int) -> list[tuple[int, int]]:
    """``chunks`` contiguous (start, count) row ranges covering ``rows`` (empty ranges dropped)."""
    step = max(1, -(-rows // max(1, chunks)))
    return [(r0, min(step, rows - r0)) for r0 in range(0, rows, step)]


def _copy2d(
    dst: int, dpitch: int, src: int, spitch: int, width: int, height: int, stream
) -> None:
    import cuda.bindings.driver as cuda

    c = cuda.CUDA_MEMCPY2D()
    c.srcMemoryType = cuda.CUmemorytype.CU_MEMORYTYPE_DEVICE
    c.srcDevice = cuda.CUdeviceptr(src)
    c.srcPitch = spitch
    c.dstMemoryType = cuda.CUmemorytype.CU_MEMORYTYPE_DEVICE
    c.dstDevice = cuda.CUdeviceptr(dst)
    c.dstPitch = dpitch
    c.WidthInBytes = width
    c.Height = height
    (status,) = cuda.cuMemcpy2DAsync(c, cuda.CUstream(stream.cuda_stream))
    if status != cuda.CUresult.CUDA_SUCCESS:
        raise RuntimeError(f"o_rowshard: 2-D copy failed: {status}")


# Debug-only, never served: MUSE_VR_SP_FLAG_EVENTS=1 replaces the stream-memory flag write / wait pairs (all of them signal from
# this device's copy stream to its own main stream) with CUDA events recorded / waited at the same points, for sanitizers that do
# not support stream-memory waits. Same buffers, same kernels, same ordering points.
_FLAG_EVENTS: dict = {"on": None, "ev": {}}


def flag_events_on() -> bool:
    if _FLAG_EVENTS["on"] is None:
        _FLAG_EVENTS["on"] = os.environ.get("MUSE_VR_SP_FLAG_EVENTS", "") == "1"
    return _FLAG_EVENTS["on"]


def _flag_event(ptr: int):
    ev = _FLAG_EVENTS["ev"].get(ptr)
    if ev is None:
        ev = _FLAG_EVENTS["ev"][ptr] = torch.cuda.Event()
    return ev


def _write_flag(stream, ptr: int) -> None:
    if flag_events_on():
        _flag_event(ptr).record(stream)
        return
    import cuda.bindings.driver as cuda

    # Default flags order the stream's earlier copies before the write (as SGLang's all-gather + GEMM publishes).
    (status,) = cuda.cuStreamWriteValue32(
        cuda.CUstream(stream.cuda_stream), cuda.CUdeviceptr(ptr), 1, 0
    )
    if status != cuda.CUresult.CUDA_SUCCESS:
        raise RuntimeError(f"o_rowshard: flag write failed: {status}")


def _wait_flag(stream, ptr: int) -> None:
    if flag_events_on():
        stream.wait_event(_flag_event(ptr))
        return
    import cuda.bindings.driver as cuda

    (status,) = cuda.cuStreamWaitValue32(
        cuda.CUstream(stream.cuda_stream),
        cuda.CUdeviceptr(ptr),
        1,
        cuda.CUstreamWaitValue_flags.CU_STREAM_WAIT_VALUE_EQ,
    )
    if status != cuda.CUresult.CUDA_SUCCESS:
        raise RuntimeError(f"o_rowshard: flag wait failed: {status}")


class RowShardContext:
    """Buffers and streams of one process (one TP rank). ``linears``: the model's o_proj modules in layer order."""

    def __init__(
        self, linears, group, rank: int, world: int, cap_rows: int, chunks: int
    ):
        import torch.distributed._symmetric_memory as symm_mem

        w0 = linears[0].weight
        self.n, self.ks = (int(s) for s in w0.shape)
        self.k = self.ks * world
        self.layers = len(linears)
        self.rank, self.world, self.chunks = rank, world, max(1, chunks)
        dev = w0.device
        bf16 = torch.bfloat16
        for lin in linears:
            w = lin.weight
            if (
                w.dtype != bf16
                or tuple(w.shape) != (self.n, self.ks)
                or not w.is_contiguous()
                or lin.bias is not None
            ):
                raise ValueError(
                    f"o_rowshard: o_proj weight {w.dtype} {tuple(w.shape)} is not a bias-free bf16 "
                    f"[{self.n}, {self.ks}] slice"
                )
        # Weight slices -> one symmetric buffer; the parameters become views of it (same values, no second copy).
        self.wstore = symm_mem.empty(
            (self.layers, self.n, self.ks), dtype=bf16, device=dev
        )
        for i, lin in enumerate(linears):
            self.wstore[i].copy_(lin.weight.data)
        torch.cuda.synchronize(dev)
        self.whandle = symm_mem.rendezvous(self.wstore, group)
        for i, lin in enumerate(linears):
            lin.weight.data = self.wstore[i]
            lin._o_rowshard_layer = i
        self.cap = _ceil_align(int(cap_rows), world)
        self.stage = symm_mem.empty((self.cap, self.ks), dtype=bf16, device=dev)
        self.shandle = symm_mem.rendezvous(self.stage, group)
        self.recv = torch.empty((self.cap // world, self.k), dtype=bf16, device=dev)
        self.slots = [
            torch.empty((self.n, self.k), dtype=bf16, device=dev) for _ in range(2)
        ]
        self.flags = torch.zeros(self.chunks, dtype=torch.int32, device=dev)
        self.act_stream = torch.cuda.Stream(device=dev)
        self.w_stream = torch.cuda.Stream(device=dev)
        self.pend_event = torch.cuda.Event()
        self.pending = (
            None  # layer whose weight copy was started by the previous layer's call
        )
        self.joined = False  # the main stream already waits for that copy
        self.prefetch = (
            bed.env_int("MUSE_VR_O_ROWSHARD_PREFETCH", 1) != 0
        )  # 0: every call copies its weights itself
        self.fp8 = None  # Fp8RowShard when fp8_prefill runs the o group
        self.whandle.barrier(channel=0)
        self.shandle.barrier(channel=0)

    def nbytes(self) -> dict:
        def b(t):
            return t.numel() * t.element_size()

        out = {
            "weight_store": b(self.wstore),
            "slots": 2 * b(self.slots[0]),
            "stage": b(self.stage),
            "recv": b(self.recv),
        }
        if self.fp8 is not None:
            out.update(self.fp8.nbytes())
        return out

    def gather_weight(self, layer: int, slot: int, stream) -> None:
        """Copy every rank's slice of ``layer`` into ``slots[slot]`` columns [p * ks, (p + 1) * ks) on ``stream``."""
        dst = self.slots[slot]
        row = self.ks * 2
        off = layer * self.n * self.ks * 2
        for step in range(self.world):
            p = (self.rank + step) % self.world
            _copy2d(
                dst.data_ptr() + p * row,
                self.k * 2,
                int(self.whandle.buffer_ptrs[p]) + off,
                row,
                row,
                self.n,
                stream,
            )

    def forward(self, x: torch.Tensor, layer: int) -> torch.Tensor:
        """x: this rank's [M, ks] head slice of the attention output (all rows). Returns [ceil(M / tp) rows, n]."""
        if self.fp8 is not None and _fp8_pass():
            bed.count(NAME, "fp8:calls")
            return self.fp8.forward(x, layer)
        m = x.shape[0]
        padded = _ceil_align(m, self.world)
        r = padded // self.world
        main = torch.cuda.current_stream()
        slot = layer % 2
        if self.pending == layer and self.joined:
            bed.count(NAME, "weights:prefetched")
        else:
            self.w_stream.wait_stream(main)
            self.gather_weight(layer, slot, self.w_stream)
            main.wait_stream(self.w_stream)
            bed.count(NAME, "weights:inline")
        self.pending, self.joined = None, False
        # All-to-all of the activation rows: stage, publish, pull per row chunk, GEMM per chunk as it lands.
        self.shandle.barrier(
            channel=0
        )  # every rank's previous pulls from its peers' staging have completed
        self.stage[:m].copy_(x)
        if padded > m:
            self.stage[m:padded].zero_()
        self.flags.zero_()
        self.shandle.barrier(channel=0)  # every rank's rows are staged
        self.act_stream.wait_stream(main)
        out = x.new_empty((r, self.n))
        row = self.ks * 2
        parts = chunk_rows(r, self.chunks)
        fptr = self.flags.data_ptr()
        with torch.cuda.stream(self.act_stream):
            for c, (r0, rc) in enumerate(parts):
                for step in range(self.world):
                    p = (self.rank + step) % self.world
                    src = int(self.shandle.buffer_ptrs[p]) + (self.rank * r + r0) * row
                    _copy2d(
                        self.recv.data_ptr() + (r0 * self.k + p * self.ks) * 2,
                        self.k * 2,
                        src,
                        row,
                        row,
                        rc,
                        self.act_stream,
                    )
                _write_flag(self.act_stream, fptr + 4 * c)
        w = self.slots[slot]
        for c, (r0, rc) in enumerate(parts):
            _wait_flag(main, fptr + 4 * c)
            torch.mm(self.recv[r0 : r0 + rc], w.t(), out=out[r0 : r0 + rc])
        main.wait_stream(self.act_stream)
        # Start the next layer's weight copy now; the next layer's QKV projection joins it.
        if layer + 1 < self.layers and self.prefetch:
            self.w_stream.wait_stream(main)
            self.gather_weight(layer + 1, (layer + 1) % 2, self.w_stream)
            self.pend_event.record(self.w_stream)
            self.pending, self.joined = layer + 1, False
        return out

    def join(self) -> None:
        if self.pending is not None and not self.joined:
            torch.cuda.current_stream().wait_event(self.pend_event)
            self.joined = True


class Fp8RowShard:
    """The MXFP8 form of ``RowShardContext.forward`` (buffers for the largest pass; scales in the 128 x 4 tile layout)."""

    def __init__(self, ctx: RowShardContext, group, backend: str):
        import torch.distributed._symmetric_memory as symm_mem

        from . import mxfp8_ops as mx

        self.ctx, self.mx, self.backend = ctx, mx, backend
        n, ks, k, w = ctx.n, ctx.ks, ctx.k, ctx.world
        t = mx.SF_TILE_ROWS
        if ks % (mx.SF_VEC * mx.SF_TILE_COLS) or n % t:
            raise ValueError(
                f"o_rowshard fp8: slice width {ks} / rows {n} do not fill whole scale tiles"
            )
        self.cts, self.ct = (
            ks // (mx.SF_VEC * mx.SF_TILE_COLS),
            k // (mx.SF_VEC * mx.SF_TILE_COLS),
        )
        self.rpad = (
            -(-(ctx.cap // w) // t) * t
        )  # one destination block, padded to whole tiles
        dev = ctx.wstore.device
        u8 = torch.uint8
        # staging (symmetric): w blocks of rpad rows, values then scales
        self.sq = w * self.rpad * ks
        self.stage = symm_mem.empty(
            (self.sq + mx.sf_bytes(w * self.rpad, ks),), dtype=u8, device=dev
        )
        self.shandle = symm_mem.rendezvous(self.stage, group)
        # own weight slice, quantized one layer ahead (symmetric, two slots)
        self.wq_bytes, self.ws_bytes = n * ks, mx.sf_bytes(n, ks)
        self.wslice = symm_mem.empty(
            (2, self.wq_bytes + self.ws_bytes), dtype=u8, device=dev
        )
        self.whandle = symm_mem.rendezvous(self.wslice, group)
        self.recv = torch.empty(
            (self.rpad * k + mx.sf_bytes(self.rpad, k),), dtype=u8, device=dev
        )
        self.slots = [
            torch.empty((n * k + mx.sf_bytes(n, k),), dtype=u8, device=dev)
            for _ in range(2)
        ]
        self.quantized = None  # layer whose own slice sits in wslice[layer % 2]
        self.shandle.barrier(channel=0)
        self.whandle.barrier(channel=0)

    def nbytes(self) -> dict:
        return {
            "fp8_stage": self.stage.numel(),
            "fp8_wslice": self.wslice.numel(),
            "fp8_recv": self.recv.numel(),
            "fp8_slots": 2 * self.slots[0].numel(),
        }

    def _quantize_slice(self, layer: int) -> None:
        q, s = self.mx.quantize(self.ctx.wstore[layer])
        dst = self.wslice[layer % 2]
        dst[: self.wq_bytes].copy_(q.view(torch.uint8).reshape(-1))
        dst[self.wq_bytes : self.wq_bytes + self.ws_bytes].copy_(s[: self.ws_bytes])

    def gather_weight(self, layer: int, slot: int, stream) -> None:
        ctx, mx = self.ctx, self.mx
        n, ks, k = ctx.n, ctx.ks, ctx.k
        dst = self.slots[slot].data_ptr()
        tb = mx.SF_TILE_BYTES
        base = (layer % 2) * (self.wq_bytes + self.ws_bytes)
        for step in range(ctx.world):
            p = (ctx.rank + step) % ctx.world
            src = int(self.whandle.buffer_ptrs[p]) + base
            _copy2d(dst + p * ks, k, src, ks, ks, n, stream)
            _copy2d(
                dst + n * k + p * self.cts * tb,
                self.ct * tb,
                src + self.wq_bytes,
                self.cts * tb,
                self.cts * tb,
                n // mx.SF_TILE_ROWS,
                stream,
            )

    def forward(self, x: torch.Tensor, layer: int) -> torch.Tensor:
        ctx, mx = self.ctx, self.mx
        w, ks, k, n = ctx.world, ctx.ks, ctx.k, ctx.n
        t, tb = mx.SF_TILE_ROWS, mx.SF_TILE_BYTES
        m = x.shape[0]
        rs = _ceil_align(m, w) // w  # rows each rank keeps (the stock row ownership)
        rp = -(-rs // t) * t
        main = torch.cuda.current_stream()
        slot = layer % 2
        # destination blocks padded to whole scale tiles (pad rows are zeros), then quantized
        xs = x.new_zeros((w, rp, ks))
        xpad = x if w * rs == m else torch.nn.functional.pad(x, (0, 0, 0, w * rs - m))
        xs[:, :rs].copy_(xpad.view(w, rs, ks))
        q, s = mx.quantize(xs.view(w * rp, ks))
        qb, sb = w * rp * ks, mx.sf_bytes(w * rp, ks)
        ctx.shandle.barrier(
            channel=0
        )  # previous pulls from both stagings are complete (one barrier serves both)
        self.stage[:qb].copy_(q.view(torch.uint8).reshape(-1))
        self.stage[self.sq : self.sq + sb].copy_(s[:sb])
        if self.quantized != layer:
            self._quantize_slice(layer)
        if layer + 1 < ctx.layers:
            self._quantize_slice(
                layer + 1
            )  # published by the barrier below; the next layer's gather reads it
        ctx.flags.zero_()
        ctx.shandle.barrier(channel=0)  # every rank's rows and weight slices are staged
        self.quantized = layer + 1
        if ctx.pending == layer and ctx.joined:
            bed.count(NAME, "fp8:weights:prefetched")
        else:
            ctx.w_stream.wait_stream(main)
            self.gather_weight(layer, slot, ctx.w_stream)
            main.wait_stream(ctx.w_stream)
            bed.count(NAME, "fp8:weights:inline")
        ctx.pending, ctx.joined = None, False
        ctx.act_stream.wait_stream(main)
        out = x.new_empty((rp, n))
        parts = [(r0 * t, rc * t) for r0, rc in chunk_rows(rp // t, ctx.chunks)]
        fptr = ctx.flags.data_ptr()
        rq = self.recv.data_ptr()
        rsf = rq + rp * k
        with torch.cuda.stream(ctx.act_stream):
            for c, (r0, rc) in enumerate(parts):
                for step in range(w):
                    p = (ctx.rank + step) % w
                    src = int(self.shandle.buffer_ptrs[p])
                    row0 = ctx.rank * rp + r0
                    _copy2d(
                        rq + r0 * k + p * ks,
                        k,
                        src + row0 * ks,
                        ks,
                        ks,
                        rc,
                        ctx.act_stream,
                    )
                    _copy2d(
                        rsf + ((r0 // t) * self.ct + p * self.cts) * tb,
                        self.ct * tb,
                        src + self.sq + (row0 // t) * self.cts * tb,
                        self.cts * tb,
                        self.cts * tb,
                        rc // t,
                        ctx.act_stream,
                    )
                _write_flag(ctx.act_stream, fptr + 4 * c)
        wsl = self.slots[slot]
        wq = wsl[: n * k].view(n, k).view(mx.FP8)
        ws = wsl[n * k :]
        aq = self.recv[: rp * k].view(rp, k).view(mx.FP8)
        asf = self.recv[rp * k :]
        for c, (r0, rc) in enumerate(parts):
            _wait_flag(main, fptr + 4 * c)
            s0, s1 = (r0 // t) * self.ct * tb, ((r0 + rc) // t) * self.ct * tb
            mx.gemm(
                aq[r0 : r0 + rc],
                asf[s0:s1],
                wq,
                ws,
                self.backend,
                out=out[r0 : r0 + rc],
            )
        main.wait_stream(ctx.act_stream)
        if layer + 1 < ctx.layers and ctx.prefetch:
            ctx.w_stream.wait_stream(main)
            self.gather_weight(layer + 1, (layer + 1) % 2, ctx.w_stream)
            ctx.pend_event.record(ctx.w_stream)
            ctx.pending, ctx.joined = layer + 1, False
        return out[:rs]


def _quant_into(mx, xs, outs) -> None:
    """Quantize BF16 [rows, K] tensors into caller-owned (q E4M3 [rows, K], s 1-D uint8) buffers: grouped ``mxfp8_fq``
    launches when the FQ quantizer is selected and takes every operand (the same bytes), else the stock quantizer."""
    if mx.fq_mode():
        from . import mxfp8_fq as fq

        if all(fq.takes(x) for x in xs):
            for i in range(0, len(xs), fq.MAX_GROUP):
                fq.quantize_group_into(
                    xs[i : i + fq.MAX_GROUP], outs[i : i + fq.MAX_GROUP]
                )
            return
    for x, (q, s) in zip(xs, outs):
        mx.quantize_into(x, q, s)


class PullContext:
    """``MUSE_VR_O_ROWSHARD_FORM=pull`` (see the module doc): buffers and streams of one process (one TP rank).
    ``linears``: the model's o_proj modules in layer order; their weights are left in place."""

    fp8_only = True  # BF16 passes run the stock function

    def __init__(
        self,
        linears,
        group,
        rank: int,
        world: int,
        cap_rows: int,
        backend: str,
        weights: str,
        prefetch: bool,
        wgather: str,
        max_polls: int,
    ):
        import torch.distributed._symmetric_memory as symm_mem

        from . import mxfp8_ops as mx
        from .ops import a2a_pull

        if weights not in PULL_WEIGHTS or wgather not in WGATHERS:
            raise ValueError(
                f"o_rowshard pull: weights {weights!r} / wgather {wgather!r}"
            )
        w0 = linears[0].weight
        self.n, self.ks = (int(s) for s in w0.shape)
        self.k = self.ks * world
        self.layers = len(linears)
        self.rank, self.world, self.chunks = rank, world, 1
        self.mx, self.pull_op, self.backend = mx, a2a_pull, backend
        self.weights, self.prefetch, self.wgather, self.max_polls = (
            weights,
            prefetch,
            wgather,
            int(max_polls),
        )
        bf16 = torch.bfloat16
        for lin in linears:
            w = lin.weight
            if (
                w.dtype != bf16
                or tuple(w.shape) != (self.n, self.ks)
                or not w.is_contiguous()
                or lin.bias is not None
            ):
                raise ValueError(
                    f"o_rowshard: o_proj weight {w.dtype} {tuple(w.shape)} is not a bias-free bf16 "
                    f"[{self.n}, {self.ks}] slice"
                )
        t = mx.SF_TILE_ROWS
        if self.ks % (mx.SF_VEC * mx.SF_TILE_COLS) or self.n % t:
            raise ValueError(
                f"o_rowshard pull: slice width {self.ks} / rows {self.n} do not fill whole scale tiles"
            )
        self.linears = list(linears)
        for i, lin in enumerate(linears):
            lin._o_rowshard_layer = i
        dev = w0.device
        u8 = torch.uint8
        self.cap = _ceil_align(int(cap_rows), world)
        self.rpad = (
            -(-(self.cap // world) // t) * t
        )  # rows of one destination block at capacity (whole tiles)
        # staging (symmetric): one block per destination rank, values [rows, ks] then their scale tiles
        self.stage = symm_mem.empty(
            (world * (self.rpad * self.ks + mx.sf_bytes(self.rpad, self.ks)),),
            dtype=u8,
            device=dev,
        )
        self.shandle = symm_mem.rendezvous(self.stage, group)
        self.stage_ptrs = [int(self.shandle.buffer_ptrs[p]) for p in range(world)]
        # signal pads of the bounded barrier (symmetric, zero at start)
        self.pad = symm_mem.empty((world,), dtype=torch.int32, device=dev)
        self.pad.zero_()
        torch.cuda.synchronize(dev)
        self.phandle = symm_mem.rendezvous(self.pad, group)
        self.pads = [int(self.phandle.buffer_ptrs[p]) for p in range(world)]
        self.recv = torch.empty(
            (self.rpad * self.k + mx.sf_bytes(self.rpad, self.k),), dtype=u8, device=dev
        )
        # own weight slice, quantized (symmetric, two slots: layer % 2)
        self.wq_bytes, self.ws_bytes = self.n * self.ks, mx.sf_bytes(self.n, self.ks)
        self.wsb = self.wq_bytes + self.ws_bytes
        self.wslice = symm_mem.empty((2, self.wsb), dtype=u8, device=dev)
        self.whandle = symm_mem.rendezvous(self.wslice, group)
        self.wslice_ptrs = [int(self.whandle.buffer_ptrs[p]) for p in range(world)]
        self.fb = self.n * self.k + mx.sf_bytes(
            self.n, self.k
        )  # bytes of one full-width MXFP8 weight
        self.slots = (
            [torch.empty((self.fb,), dtype=u8, device=dev) for _ in range(2)]
            if weights == "jit"
            else []
        )
        self.res = []
        self.w_stream = torch.cuda.Stream(device=dev)
        self.pend_event = torch.cuda.Event()
        self.pending = (
            None  # layer whose weight gather was started by the previous layer's call
        )
        self.joined = False  # the main stream already waits for it
        self.quantized = None  # layer whose own slice sits in wslice[layer % 2] (and the next layer's in the other)
        a2a_pull.module(
            a2a_pull.config()[0]
        )  # built on every rank before any use or capture
        self.shandle.barrier(channel=0)
        self.whandle.barrier(channel=0)
        self.phandle.barrier(channel=0)
        if weights == "resident":
            self._build_resident()
        self.fp8 = (
            self  # the MXFP8 form is this context (fp8_prefill warms its GEMM shape)
        )

    def nbytes(self) -> dict:
        return {
            "pull_stage": self.stage.numel(),
            "pull_recv": self.recv.numel(),
            "pull_wslice": self.wslice.numel(),
            "pull_slots": sum(s.numel() for s in self.slots),
            "pull_resident": sum(r.numel() for r in self.res),
        }

    def gemm_pinned(self, m: int) -> bool:
        """True if this call's GEMM runs on the pinned runner (a fixed configuration of the runner factory from the
        deployment's configuration, launched without programmatic dependent launch, so it starts only after the pull
        kernel has completed). The pull form takes only such calls; others keep the stock function."""
        mx = self.mx
        rs = _ceil_align(m, self.world) // self.world
        rp = -(-rs // mx.SF_TILE_ROWS) * mx.SF_TILE_ROWS
        return (
            self.backend == "cute-dsl"
            and mx._PINNED["factory"] is not None
            and mx._pinned_tactic(rp, self.n, self.k) is not None
        )

    def _barrier(self, like: torch.Tensor) -> None:
        self.pull_op.barrier(like, self.pads, self.rank, self.max_polls)

    def _wslice_views(self, slot: int):
        b = self.wslice[slot]
        return b[: self.wq_bytes].view(self.n, self.ks).view(self.mx.FP8), b[
            self.wq_bytes :
        ]

    def _quantize_slice(self, layer: int) -> None:
        _quant_into(
            self.mx, [self.linears[layer].weight.data], [self._wslice_views(layer % 2)]
        )

    def _gather_into(self, dst: torch.Tensor, slot: int, stream, wgather: str) -> None:
        """Every rank's quantized slice in ``wslice[slot]`` -> the full-width weight ``dst`` (values [n, k] then scale
        tiles), on ``stream``."""
        mx, n, ks, k = self.mx, self.n, self.ks, self.k
        if wgather == "pull":
            with torch.cuda.stream(stream):
                self.pull_op.pull(
                    dst[: n * k].view(n, k),
                    dst[n * k :],
                    [p + slot * self.wsb for p in self.wslice_ptrs],
                    ks,
                    n,
                )
            return
        tb, cts, ct = (
            mx.SF_TILE_BYTES,
            ks // (mx.SF_VEC * mx.SF_TILE_COLS),
            k // (mx.SF_VEC * mx.SF_TILE_COLS),
        )
        d = dst.data_ptr()
        for step in range(self.world):
            p = (self.rank + step) % self.world
            src = self.wslice_ptrs[p] + slot * self.wsb
            _copy2d(d + p * ks, k, src, ks, ks, n, stream)
            _copy2d(
                d + n * k + p * cts * tb,
                ct * tb,
                src + self.wq_bytes,
                cts * tb,
                cts * tb,
                n // mx.SF_TILE_ROWS,
                stream,
            )

    def _build_resident(self) -> None:
        """Full-width MXFP8 weights of every layer, through the per-call path's own quantize + pull (same bytes)."""
        dev = self.wslice.device
        main = torch.cuda.current_stream()
        for i in range(self.layers):
            self._barrier(
                self.pad
            )  # every rank's previous pull from wslice[0] is complete
            _quant_into(self.mx, [self.linears[i].weight.data], [self._wslice_views(0)])
            self._barrier(self.pad)  # every rank's slice of layer i is staged
            buf = torch.empty((self.fb,), dtype=torch.uint8, device=dev)
            self._gather_into(buf, 0, main, "pull")
            self.res.append(buf)
        self._barrier(self.pad)
        torch.cuda.synchronize(dev)

    def gather_weight(self, layer: int, slot: int, stream) -> None:
        self._gather_into(self.slots[slot], layer % 2, stream, self.wgather)

    def forward(self, x: torch.Tensor, layer: int) -> torch.Tensor:
        """x: this rank's [M, ks] head slice of the attention output (all rows). Returns [ceil(M / tp) rows, n]."""
        mx, w, ks, k, n = self.mx, self.world, self.ks, self.k, self.n
        t = mx.SF_TILE_ROWS
        m = x.shape[0]
        rs = _ceil_align(m, w) // w  # rows each rank keeps (the stock row ownership)
        rp = -(-rs // t) * t
        qb, sb = rp * ks, mx.sf_bytes(rp, ks)
        blk = qb + sb
        main = torch.cuda.current_stream()
        slot = layer % 2
        bed.count(NAME, "pull:calls")
        self._barrier(
            x
        )  # every rank's previous pulls from this rank's staging and weight slots are complete
        if w * rp == m:
            blocks = [x[d * rp : (d + 1) * rp] for d in range(w)]
        else:  # destination blocks padded to whole scale tiles: the padding rows are quantized zero rows
            xs = x.new_zeros((w, rp, ks))
            xpad = (
                x if w * rs == m else torch.nn.functional.pad(x, (0, 0, 0, w * rs - m))
            )
            xs[:, :rs].copy_(xpad.view(w, rs, ks))
            blocks = [xs[d] for d in range(w)]
            bed.count(NAME, "pull:padded")
        st = self.stage
        outs = [
            (
                st[d * blk : d * blk + qb].view(rp, ks).view(mx.FP8),
                st[d * blk + qb : (d + 1) * blk],
            )
            for d in range(w)
        ]
        _quant_into(mx, blocks, outs)
        if self.weights == "jit":
            if self.quantized != layer:
                self._quantize_slice(layer)
            if layer + 1 < self.layers:
                self._quantize_slice(
                    layer + 1
                )  # published by the barrier below; the next layer's gather reads it
            self.quantized = layer + 1
        self._barrier(x)  # every rank's blocks and weight slices are staged
        if self.weights == "jit":
            if self.pending == layer and self.joined:
                bed.count(NAME, "pull:weights:prefetched")
            else:
                self.w_stream.wait_stream(main)
                self.gather_weight(layer, slot, self.w_stream)
                main.wait_stream(self.w_stream)
                bed.count(NAME, "pull:weights:inline")
            self.pending, self.joined = None, False
            wbuf = self.slots[slot]
        else:
            wbuf = self.res[layer]
        aq = self.recv[: rp * k].view(rp, k)
        asf = self.recv[rp * k : rp * k + mx.sf_bytes(rp, k)]
        self.pull_op.pull(
            aq, asf, [p + self.rank * blk for p in self.stage_ptrs], ks, rp
        )
        out = x.new_empty((rp, n))
        mx.gemm(
            aq.view(mx.FP8),
            asf,
            wbuf[: n * k].view(n, k).view(mx.FP8),
            wbuf[n * k :],
            self.backend,
            out=out,
        )
        if self.weights == "jit" and layer + 1 < self.layers and self.prefetch:
            self.w_stream.wait_stream(main)
            self.gather_weight(layer + 1, (layer + 1) % 2, self.w_stream)
            self.pend_event.record(self.w_stream)
            self.pending, self.joined = layer + 1, False
        return out[:rs]

    def join(self) -> None:
        if self.pending is not None and not self.joined:
            torch.cuda.current_stream().wait_event(self.pend_event)
            self.joined = True


def _fp8_pass() -> bool:
    from . import fp8_prefill

    return fp8_prefill.pass_fp8("o")


def _form() -> str:
    return os.environ.get("MUSE_VR_O_ROWSHARD_FORM", "").strip() or "rowshard"


def _pull_weights() -> str:
    return os.environ.get("MUSE_VR_O_ROWSHARD_WEIGHTS", "").strip() or "jit"


def _wgather() -> str:
    return os.environ.get("MUSE_VR_O_ROWSHARD_WGATHER", "").strip() or "ce"


def _max_polls() -> int:
    return bed.env_int(
        "MUSE_VR_O_ROWSHARD_BARRIER_MAX_POLLS", DEFAULT_BARRIER_MAX_POLLS
    )


def _fp8_requested() -> tuple[bool, str]:
    """(fp8_prefill opted in with the o group, its GEMM backend), read from the environment at install."""
    groups = [
        g.strip()
        for g in os.environ.get(
            "MUSE_VR_FP8_PREFILL_GROUPS", "qkv,o,pool,unpool"
        ).split(",")
    ]
    return bed.enabled("fp8_prefill") and "o" in groups, os.environ.get(
        "MUSE_VR_FP8_PREFILL_BACKEND", "cute-dsl"
    ).strip()


def _prefill_capture_sizes():
    from sglang.srt.model_executor.cuda_graph_config import Backend
    from sglang.srt.runtime_context import get_exec

    cfg = get_exec().graph.cuda_graph_config.prefill
    if cfg.backend == Backend.DISABLED:
        return []
    return sorted(int(s) for s in (cfg.bs or []))


def threshold_ok(min_rows: int, sizes) -> bool:
    """The row threshold must be a captured prefill bucket (or at / above the largest one, or graphs off)."""
    sizes = sorted(int(s) for s in (sizes or ()))
    if not sizes or min_rows >= sizes[-1]:
        return True
    i = bisect.bisect_left(sizes, min_rows)
    return i < len(sizes) and sizes[i] == min_rows


def _threshold_usable(min_rows: int) -> bool:
    ok = _state["checked"]
    if ok is None:
        with _lock:
            if _state["checked"] is None:
                try:
                    sizes = _prefill_capture_sizes()
                    ok = threshold_ok(min_rows, sizes)
                except Exception as e:  # noqa: BLE001
                    logger.error(
                        "muse_vr_plugin: o_rowshard cannot read the prefill capture list (%s); stock",
                        e,
                    )
                    ok = False
                if not ok:
                    logger.error(
                        "muse_vr_plugin: o_rowshard MIN_ROWS %d is not a prefill CUDA-graph bucket; stock "
                        "reduce-scatter for every call",
                        min_rows,
                    )
                _state["checked"] = ok
                bed.report(NAME + "_check", ok, min_rows=min_rows)
    return _state["checked"]


def _wrap_gbar(orig):
    @functools.wraps(orig)
    def row_parallel_gbar_matmul(linear, input_, bias):
        ctx = _state["ctx"]
        layer = getattr(linear, "_o_rowshard_layer", None)
        if ctx is None or layer is None:
            return orig(linear, input_, bias)
        reason = None
        m = int(input_.shape[0]) if input_.dim() == 2 else -1
        padded = _ceil_align(max(m, 0), ctx.world)
        min_rows = bed.env_int("MUSE_VR_O_ROWSHARD_MIN_ROWS", DEFAULT_MIN_ROWS)
        if bias is not None:
            reason = "bias"
        elif (
            input_.dim() != 2
            or input_.dtype != torch.bfloat16
            or input_.shape[1] != ctx.ks
        ):
            reason = "input"
        elif torch.compiler.is_compiling():
            reason = "compiling"
        elif getattr(ctx, "fp8_only", False) and not _fp8_pass():
            reason = "pass"
        elif padded < min_rows:
            reason = "rows<min"
        elif padded > ctx.cap:
            reason = "rows>cap"
        elif not _threshold_usable(min_rows):
            reason = "threshold"
        elif getattr(ctx, "fp8_only", False) and not ctx.gemm_pinned(padded):
            reason = "unpinned"
        if reason is not None:
            bed.count(NAME, "stock:" + reason)
            return orig(linear, input_, bias)
        bed.count(NAME, f"own:m{padded}")
        return ctx.forward(input_.contiguous(), layer)

    row_parallel_gbar_matmul.__wrapped_by_o_rowshard__ = True
    return row_parallel_gbar_matmul


def _wrap_g(orig):
    @functools.wraps(orig)
    def column_parallel_g_matmul(linear, input_parallel, bias):
        ctx = _state["ctx"]
        if ctx is not None:
            ctx.join()
        return orig(linear, input_parallel, bias)

    column_parallel_g_matmul.__wrapped_by_o_rowshard__ = True
    return column_parallel_g_matmul


def _is_target_model(model) -> bool:
    name = type(model).__name__
    return name.startswith("MuseSpark") and "Draft" not in name


def _o_projs(model):
    out = []
    for mod in model.modules():
        if type(mod).__name__ == "MuseSpark1xDecoderLayer":
            out.append(mod.self_attn.o_proj)
    return out


def install_into(
    model, group, rank: int, world: int, cap_rows: int, chunks: int
) -> RowShardContext:
    """Build the context for ``model`` (collective over ``group``) and hook the SP entry points (also used by tests)."""
    import sglang.srt.layers.layernorm_sp as layernorm_sp

    linears = _o_projs(model) if not isinstance(model, (list, tuple)) else list(model)
    if not linears:
        raise ValueError("o_rowshard: no Muse Spark decoder layers")
    if not all(getattr(lin, "_sp_use_cutlass", False) for lin in linears):
        raise ValueError(
            "o_rowshard: o_proj is not a sequence-parallel participant (layernorm SP off?)"
        )
    want_fp8, backend = _fp8_requested()
    form = _form()
    if form == "pull":
        if not want_fp8:
            raise ValueError(
                "o_rowshard: MUSE_VR_O_ROWSHARD_FORM=pull needs fp8_prefill with the o group"
            )
        ctx = PullContext(
            linears,
            group,
            rank,
            world,
            cap_rows,
            backend,
            _pull_weights(),
            bed.env_int("MUSE_VR_O_ROWSHARD_PREFETCH", 1) != 0,
            _wgather(),
            _max_polls(),
        )
    else:
        ctx = RowShardContext(linears, group, rank, world, cap_rows, chunks)
        if want_fp8:
            ctx.fp8 = Fp8RowShard(ctx, group, backend)
    with _lock:
        if _state["orig_gbar"] is None:
            _state["orig_gbar"] = layernorm_sp.row_parallel_gbar_matmul
            layernorm_sp.row_parallel_gbar_matmul = _wrap_gbar(_state["orig_gbar"])
            _state["orig_g"] = layernorm_sp.column_parallel_g_matmul
            layernorm_sp.column_parallel_g_matmul = _wrap_g(_state["orig_g"])
        _state["ctx"] = ctx
    return ctx


def _wrap_load_model(fn):
    @functools.wraps(fn)
    def load_model(self, *args, **kwargs):
        out = fn(self, *args, **kwargs)
        model = getattr(self, "model", None)
        if (
            getattr(self, "is_draft_worker", False)
            or not _is_target_model(model)
            or _state["ctx"] is not None
        ):
            return out
        try:
            from sglang.srt.runtime_context import get_parallel

            par = get_parallel()
            tp = par.tp_group
            world, rank = int(tp.world_size), int(tp.rank_in_group)
            if world not in (2, 4, 8):
                raise ValueError(f"tp {world} not in (2, 4, 8)")
            sizes = _prefill_capture_sizes()
            cap = bed.env_int(
                "MUSE_VR_O_ROWSHARD_CAP_ROWS", max(sizes) if sizes else 16384
            )
            chunks = bed.env_int("MUSE_VR_O_ROWSHARD_CHUNKS", DEFAULT_CHUNKS)
            ctx = install_into(model, tp.device_group, rank, world, cap, chunks)
            torch.cuda.empty_cache()  # return the moved weight slices' old storage before the KV cache is sized
            summary = dict(
                layers=ctx.layers,
                n=ctx.n,
                ks=ctx.ks,
                tp=world,
                cap_rows=ctx.cap,
                chunks=ctx.chunks,
                min_rows=bed.env_int("MUSE_VR_O_ROWSHARD_MIN_ROWS", DEFAULT_MIN_ROWS),
                form=_form(),
                **{k + "_bytes": v for k, v in ctx.nbytes().items()},
            )
            if isinstance(ctx, PullContext):
                from .ops import a2a_pull

                u, c = a2a_pull.config()
                summary.update(
                    weights=ctx.weights,
                    prefetch=ctx.prefetch,
                    wgather=ctx.wgather,
                    barrier_max_polls=ctx.max_polls,
                    pull_unroll=u,
                    pull_ctas_per_sm=c,
                )
            _state["summary"] = summary
            bed.report(NAME, True, **summary)
        except Exception as e:  # noqa: BLE001 - a half-installed context must not serve: record, then re-raise
            bed.report(NAME, False, reason=f"install failed: {type(e).__name__}: {e}")
            raise
        return out

    load_model.__wrapped_by_o_rowshard__ = True
    return load_model


def register() -> bool:
    if not bed.enabled(NAME):
        return False
    for rel, sha in SGLANG_PINS.items():
        if not bed.pinned(rel, sha):
            bed.report(NAME, False, reason=f"pin mismatch: {rel}")
            return False
    min_rows = bed.env_int("MUSE_VR_O_ROWSHARD_MIN_ROWS", DEFAULT_MIN_ROWS)
    if min_rows < 1:
        bed.report(NAME, False, reason=f"bad MUSE_VR_O_ROWSHARD_MIN_ROWS {min_rows}")
        return False
    form, weights, wgather, polls = _form(), _pull_weights(), _wgather(), _max_polls()
    if (
        form not in FORMS
        or weights not in PULL_WEIGHTS
        or wgather not in WGATHERS
        or not MIN_MAX_POLLS <= polls <= MAX_MAX_POLLS
    ):
        bed.report(
            NAME,
            False,
            reason=f"bad config form={form} weights={weights} wgather={wgather} max_polls={polls}",
        )
        return False
    if form == "pull":
        try:
            from .ops import a2a_pull

            a2a_pull.config()
        except ValueError as e:
            bed.report(NAME, False, reason=str(e))
            return False
    try:
        import sglang.srt.model_executor.model_runner as mr

        runner = mr.ModelRunner
        if not getattr(runner.load_model, "__wrapped_by_o_rowshard__", False):
            runner.load_model = _wrap_load_model(runner.load_model)
    except Exception as e:  # noqa: BLE001
        bed.report(NAME, False, reason=f"install failed: {type(e).__name__}: {e}")
        return False
    _state["engaged"] = True
    return True
