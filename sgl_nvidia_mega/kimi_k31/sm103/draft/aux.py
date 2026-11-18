# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the SGLang project
"""DFlash2 draft aux stage on megakernel blocks (stage 2b): the target's context features
into the draft KV cache (``dflash_worker_v2._append_target_hidden_decode``).

``DraftAux.run(target_hidden, positions, slots_2d, commit_lens)``:
1. fc over this rank's TP output-column slice (SGLANG_OPT_DFLASH_TP_AUX_FC's split:
   [7168 / TP, 6 * 7168] per rank), one split-K projection;
2. ``GatherFused``: every rank multicasts its columns zero-padded to full rows (one hop;
   -0.0 published as +0.0) and ``OprojConsumer(0, has_delta=False)`` sums the TP slots,
   which gathers exactly like today's zero-padded all-reduce, then hidden_norm -> ctx;
3. one stacked projection of the draft layers' kv_a rows (fused_a rows 1536:2112 each,
   replicated) over ctx -> [rows, layers * 576];
4. ``ContextInsert``: per layer and committed row (r < commit_lens), kv_a RMSNorm of the
   512 latent (BF16), interleaved RoPE of k_pe at the row's position (BF16), FP8 into the
   layer's cache slot: the format DraftMlaQuery.insert writes."""

from functools import cache

import cutlass
import cutlass.cute as cute
import torch
from cutlass import Float32, Int32, Int64, Uint32
from cutlass.cute.runtime import make_fake_stream

from ..k3.common.host import (
    dynamic_fake,
    dynamic_view,
    fake,
    stream,
    to_cute,
)
from ..k3.common.primitives import (
    alloc,
    at,
    fp8_bits,
    launch,
    load_bf16x8,
    load_vector,
    multicast_words,
    no_neg_zero,
    warp_sum,
    zeros,
)
from ..k3.gemm.projection import PreparedProjection
from ..k3.oproj.consumer import OprojConsumer
from .kernels import HIDDEN
from .tactics import draft_tactic

LATENT, ROPE, ROW = 512, 64, 576
COLUMNS = 64  # output columns per GatherFused CTA (112 CTAs x 64 = 7168)


class GatherFused:
    """One hop that gathers TP column slices [rows, 7168 / TP] into full rows and RMSNorms
    them. Arguments: x (this rank's slice), box, mc, slot (TP rank), then the consumer rows
    (prefix = the gathered BF16 rows, blocks, norm, query, out_norm, output, xmoe, hints,
    eps, out_eps). CTA c owns columns [64 c, 64 c + 64), all in one rank's slice."""

    def __init__(self, width, deadline_ns=0):
        if width % COLUMNS:
            raise ValueError(f"slice width {width} is not a multiple of {COLUMNS}")
        self.width = width
        self.consumer = OprojConsumer(0, has_delta=False, deadline_ns=deadline_ns)

    @cute.jit
    def __call__(self, *args):
        if cutlass.const_expr(args[0].shape[0] > 8):
            raise ValueError("GatherFused serves up to 8 rows")
        launch(self.kernel, args, (8, 14, 1), 512, 8, min_blocks_per_mp=1)

    @cute.kernel
    def kernel(self, args):
        x, box, mc, slot = args[0:4]
        rows = (box, *args[4:])
        rank, cluster, _ = cute.arch.block_idx()
        tid, _, _ = cute.arch.thread_idx()
        cta = cluster * 8 + rank
        active = (cluster < x.shape[0]) & (rank < 7)
        state = self.consumer.allocate()
        self.consumer.setup(state[5])
        cute.arch.griddepcontrol_launch_dependents()
        group0 = active & (tid < 128)
        if group0:
            self.consumer.prefetch(rows, state[7], rank, cluster)
        mix = zeros(8, Float32)
        maximum, exponent_sum, qpart = Float32(0), Float32(0), Float32(0)
        if group0:
            maximum, exponent_sum, qpart = self.consumer.pre(state, rank, rows[9], mix)
        cute.arch.griddepcontrol_wait()
        if (cta < 112) & (tid < 8 * x.shape[0]):
            token, part = tid // 8, tid % 8
            col = cta * COLUMNS + part * 8
            # Columns of other ranks' slices publish zeros (a masked in-range load).
            mask = Uint32(0)
            if col // self.width == slot:
                mask = Uint32(0xFFFFFFFF)
            loaded = load_vector(at(x, token, col % self.width))
            words = [no_neg_zero(w) & mask for w in loaded]
            offset = cute.crd2idx((slot, token, col), box.layout)
            # v4 multimem.st takes .f32 (stored as raw bits), not .b32.
            multicast_words(mc + Int64(offset) * 2, tuple(words))
        if active:
            block_mix = (mix, maximum, exponent_sum, qpart)
            self.consumer.post(rows, state, block_mix, rank, cluster)


class ContextInsert:
    """Grid (layers, rows), 96 threads: threads 0-63 own 8 latent values, 64-71 8 k_pe
    values. Arguments: kv [rows, layers * 576] BF16, norms [layers, 512] BF16, positions
    [rows] int64, cos_sin [max_pos, 64] FP32, slots [rows] int64, commit [B] int32, pools
    [layers] int64 (each layer's FP8 pool base, [slots, 576] bytes), k_scale_inv, eps."""

    def __init__(self, block: int = 8):
        self.block = block

    @cute.jit
    def __call__(self, *args):
        launch(self.kernel, args, (args[1].shape[0], args[0].shape[0], 1), 96)

    @cute.kernel
    def kernel(self, args):
        kv, norms, positions, cos_sin, slots, commit, pools, ksi, eps = args[0:9]
        layer, row, _ = cute.arch.block_idx()
        tid, _, _ = cute.arch.thread_idx()
        lane, warp = tid % 32, tid // 32
        smem = cutlass.utils.SmemAllocator()
        sums = alloc(smem, Float32, 2)
        weights = load_bf16x8(at(norms, layer, (tid % 64) * 8))
        cute.arch.griddepcontrol_wait()
        cute.arch.griddepcontrol_launch_dependents()
        live = (row % self.block) < commit[row // self.block]
        column = layer * ROW
        values = load_bf16x8(at(kv, row, column + (tid % 64) * 8), mutable=True)
        square = Float32(0)
        for i in cutlass.range_constexpr(8):
            square = square + values[i] * values[i]
        square = warp_sum(square)
        if (lane == 0) & (warp < 2):
            sums[warp] = square
        cute.arch.sync_threads()
        if live:
            base = cute.make_ptr(cutlass.Uint8, pools[layer], cute.AddressSpace.gmem)
            cache = cute.make_tensor(base + slots[row] * ROW, cute.make_layout(ROW))
            if tid < 64:
                scale = rstd_512(sums[0] + sums[1], eps)
                for i in cutlass.range_constexpr(8):
                    normed = Float32(cutlass.BFloat16(values[i] * scale * weights[i]))
                    cache[tid * 8 + i] = fp8_bits(normed * ksi)
            elif tid < 72:
                c = tid - 64
                pe = load_bf16x8(at(kv, row, column + LATENT + c * 8), mutable=True)
                pos = Int32(positions[row])
                for m in cutlass.range_constexpr(4):
                    i = c * 4 + m
                    cos, sin = cos_sin[pos, i], cos_sin[pos, ROPE // 2 + i]
                    x1, x2 = pe[2 * m], pe[2 * m + 1]
                    r1 = Float32(cutlass.BFloat16(x1 * cos - x2 * sin))
                    r2 = Float32(cutlass.BFloat16(x2 * cos + x1 * sin))
                    cache[LATENT + c * 8 + 2 * m] = fp8_bits(r1 * ksi)
                    cache[LATENT + c * 8 + 2 * m + 1] = fp8_bits(r2 * ksi)


@cute.jit
def rstd_512(sum_sq, eps):
    return cute.math.rsqrt(sum_sq * Float32(1 / LATENT) + eps, fastmath=True)


@cache
def compile_gather(rows, width, tp, deadline_ns):
    x = fake((rows, width), (width, 1))
    box = fake((tp, rows, HIDDEN), (rows * HIDDEN, HIDDEN, 1))
    eps, row, weight = cutlass.Float32(1e-6), dynamic_fake(2), fake((HIDDEN,), (1,))
    tail = (row, dynamic_fake(3), weight, weight, weight, row, None, None, eps, eps)
    args = (x, box, cutlass.Int64(0), cutlass.Int32(0), *tail)
    return cute.compile(GatherFused(width, deadline_ns), *args, make_fake_stream())


@cache
def compile_insert(rows, layers, max_pos, block):
    kv = fake((rows, layers * ROW), (layers * ROW, 1))
    norms = fake((layers, LATENT), (LATENT, 1))
    positions = fake((rows,), (1,), torch.int64, 8)
    cos_sin = fake((max_pos, ROPE), (ROPE, 1), torch.float32, 4)
    slots = fake((rows,), (1,), torch.int64, 8)
    commit = fake((rows // block,), (1,), torch.int32, 4)
    pools = fake((layers,), (1,), torch.int64, 8)
    args = (kv, norms, positions, cos_sin, slots, commit, pools)
    args += (cutlass.Float32(1.0), cutlass.Float32(1e-6))
    return cute.compile(ContextInsert(block), *args, make_fake_stream())


class DraftAux:
    """``fc_slice`` [7168 / TP, 6 * 7168] (this rank's output columns), ``hidden_norm``
    [7168], ``kv_a`` [layers x [576, 7168]] (fused_a rows 1536:2112), ``kv_norms`` [layers
    x [512]], ``cos_sin`` [max_pos, 64] FP32; ``comm.aux`` is the gather's mailbox."""

    def __init__(
        self,
        fc_slice,
        hidden_norm,
        kv_a,
        kv_norms,
        cos_sin,
        comm,
        shapes,
        *,
        eps,
        k_scale=1.0,
        deadline_ns=0,
    ):
        rows, tp = shapes.tokens, shapes.tp
        bf16 = lambda t: t.detach().to(torch.bfloat16).contiguous()  # noqa: E731
        self.device, self.rows = fc_slice.device, rows
        self.comm, self.rank = comm, shapes.rank
        self.width = fc_slice.shape[0]
        if self.width * tp != HIDDEN:
            raise ValueError(f"fc slice {tuple(fc_slice.shape)} at TP {tp}")
        fc_tactic = draft_tactic("fc_slice", self.device)
        self.fc = PreparedProjection(bf16(fc_slice), {rows: fc_tactic}).prepare()
        stacked = torch.cat([bf16(w) for w in kv_a])
        self.kv = PreparedProjection(
            stacked, {rows: draft_tactic("kv_stack", self.device)}
        )
        self.kv.prepare()
        self.layers = len(kv_a)
        self.hidden_norm = to_cute(bf16(hidden_norm))
        self.norms = bf16(torch.stack([bf16(n) for n in kv_norms]))
        self.cos_sin = cos_sin.float().contiguous()
        new = lambda width, dtype=torch.bfloat16: torch.empty(  # noqa: E731
            rows, width, dtype=dtype, device=self.device
        )
        self.local, self.gathered, self.ctx = new(self.width), new(HIDDEN), new(HIDDEN)
        self.kv_rows = new(self.layers * ROW)
        self.blocks = self.gathered.view(rows, 1, HIDDEN)  # unused at nb = 0
        self.gather = compile_gather(rows, self.width, tp, deadline_ns)
        self.insert = compile_insert(
            rows, self.layers, self.cos_sin.shape[0], shapes.rows
        )
        self.eps, self.ksi = cutlass.Float32(eps), cutlass.Float32(1.0 / k_scale)

    def run(self, target_hidden, positions, slots_2d, commit_lens, pools, parity=0):
        """``pools``: int64 [layers] base addresses of the draft layers' FP8 pools.
        ``parity``: the gather slot (serving: 0; back-to-back runs must alternate)."""
        s = stream(self.device)
        self.fc.run(target_hidden, self.local)
        box = to_cute(self.comm.aux[parity])
        mc = self.comm.aux_mc + parity * self.comm.aux[0].numel() * 2
        head = (to_cute(self.local), box, cutlass.Int64(mc))
        head += (cutlass.Int32(self.rank),)
        tail = (
            dynamic_view(self.gathered),
            dynamic_view(self.blocks),
            self.hidden_norm,
        )
        tail += (self.hidden_norm, self.hidden_norm, dynamic_view(self.ctx), None, None)
        tail += (self.eps, self.eps)
        self.gather(*head, *tail, s)
        self.kv.run(self.ctx, self.kv_rows)
        args = (to_cute(self.kv_rows), to_cute(self.norms), to_cute(positions, 8))
        args += (to_cute(self.cos_sin, 4), to_cute(slots_2d.reshape(-1), 8))
        args += (to_cute(commit_lens, 4), to_cute(pools, 8), self.ksi, self.eps)
        self.insert(*args, s)
        return self.ctx
