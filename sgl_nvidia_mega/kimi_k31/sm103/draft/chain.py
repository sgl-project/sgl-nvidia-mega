# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the SGLang project
"""DFlash2 draft layers as one PDL chain of megakernel stages (stage 1, M1a).

Per layer (rows = one 8-row block per request, TP16 per-rank shards):

  x_norm -> attn conv projection (replicated 7168 -> 1792, split-K) -> ConvPrepare
         -> attention: M1a the caller's ``attn`` hook (x -> W_UV output [rows, 768]);
            M1b ``DraftAttention`` (fused_a, DraftMlaQuery with RoPE + KV insert, the
            caller's ``core`` over the SWA window, DraftMlaValue without gate)
         -> DraftFused(o_proj, side-1 conv on the partial, mailbox all-reduce, residual
            add, post-attention RMSNorm) -> x_post
         -> mlp conv projection -> ConvPrepare -> gate_up (split-K)
         -> DraftFused(silu * up -> down, side-1 conv, all-reduce, residual add, next
            input RMSNorm, or the final norm after the last layer) -> x_norm / out

Every all-reduce hop of a round has its own mailbox slot (``DraftComm``: [hop, parity,
TP, rows, 7168], hop 2L = layer L's o_proj, 2L + 1 its down), so callers alternate
``parity`` between consecutive rounds and a test can pre-publish injected peers' rows for
every hop before a replay."""

import logging
import os
import types
from dataclasses import dataclass
from functools import cache
from typing import Callable, Optional

import cutlass
import cutlass.cute as cute
import torch
import torch.distributed as dist
import torch.distributed._symmetric_memory as symm_mem
from cutlass.cute.runtime import make_fake_stream

from ..k3.common.host import (
    architecture,
    dynamic_fake,
    dynamic_view,
    fake,
    stream,
    to_cute,
)
from ..k3.gemm.projection import PreparedProjection
from .attention import DraftAttention
from .convgemm import PreparedConvProjection
from .head import BITONIC, DraftHead
from .kernels import DELTA, HIDDEN, TAPS, ConvPrepare, DraftFused, EmbedNorm, RowNorm
from .selector import DraftSelector
from .tactics import draft_tactic

logger = logging.getLogger(__name__)

# (M tile, N tile, split K, stages) of the 7168 -> 1792 projections (conv, gate_up) at
# one 8-row block: the dense-MLP gate/up tactic. ``K31DRAFT_PROJ_TACTIC`` overrides it.
PROJ_TACTIC = (64, 8, 4, 8)


# Poll deadline of the draft all-reduce consumers (0: none; units set it so a lost peer
# row fail-stops instead of spinning).
DEADLINE_NS = int(os.environ.get("K31DRAFT_DEADLINE_NS", "0"))
# The conv prepare side folded into the conv projections' split-K epilogue
# (convgemm.ConvSplitKGemm; δ columns permuted). Race tests: K31DRAFT_RACE_DELAY_NS holds
# each hop's residual write back; K31DRAFT_FOLD_EARLY_TRIGGER=1 drops the folded
# projection's residual fence (negative control only).
# Adopted 2026-11-02 (fence by construction).
CONV_FOLD = os.environ.get("K31DRAFT_CONV_FOLD", "1") == "1"
RACE_DELAY_NS = int(os.environ.get("K31DRAFT_RACE_DELAY_NS", "0"))
FOLD_EARLY_TRIGGER = os.environ.get("K31DRAFT_FOLD_EARLY_TRIGGER", "0") == "1"


def proj_tactic():
    override = os.environ.get("K31DRAFT_PROJ_TACTIC")
    return tuple(map(int, override.split(","))) if override else PROJ_TACTIC


@dataclass(frozen=True)
class DraftShapes:
    tp: int
    rank: int
    requests: int = 1
    rows: int = 8  # DFlash block size
    hidden: int = HIDDEN
    inter: int = 896  # per rank
    heads: int = 6  # per rank
    window: int = 1024
    eps: float = 1e-6
    topk: int = (
        16  # the selector top_k (draft config), 16 or 32: the head box and the walk
    )

    @property
    def tokens(self):
        return self.requests * self.rows


@dataclass
class DraftLayerWeights:
    """One draft layer's per-rank tensors in checkpoint layout (``models/dflash.py``)."""

    input_norm: torch.Tensor  # [7168]
    attn_conv_proj: torch.Tensor  # [1792, 7168] replicated
    attn_conv_base: torch.Tensor  # [2, 2, 7168] (side, tap, channel)
    fused_a: torch.Tensor  # [2112, 7168] replicated
    q_a_norm: torch.Tensor  # [1536]
    q_b: torch.Tensor  # [heads * 192, 1536]
    kv_a_norm: torch.Tensor  # [512]
    kv_b: torch.Tensor  # [heads * 256, 512]
    o_proj: torch.Tensor  # [7168, heads * 128]
    post_norm: torch.Tensor  # [7168]
    mlp_conv_proj: torch.Tensor  # [1792, 7168] replicated
    mlp_conv_base: torch.Tensor  # [2, 2, 7168]
    gate_up: torch.Tensor  # [2 * inter, 7168]: gate rows, then up rows
    down: torch.Tensor  # [7168, inter]


@dataclass
class DraftHeadWeights:
    """Stage 2a: the target lm_head's rank shard [num_org, 7168] (vocab_start = its first
    global id), the selector's hidden_projection [256, 7168] and codebooks [vocab, 256]
    (BF16), and the draft config's unary transform."""

    lm_head: torch.Tensor
    hidden_projection: torch.Tensor
    vocab_start: int
    predecessor: torch.Tensor
    successor: torch.Tensor
    output_multiplier: float = 1.0
    final_logit_softcapping: Optional[float] = None


@dataclass
class DraftEmbedWeights:
    """Stage 2c: the target's vocab-parallel embedding shard [num_org, 7168] (vocab_start =
    its first global id) and the mask token's full embedding row [7168] (replicated; e.g.
    ``embed_tokens(mask_id)`` at build)."""

    shard: torch.Tensor
    vocab_start: int
    mask_row: torch.Tensor


@dataclass
class DraftKV:
    pool: torch.Tensor  # [layers, slots, 576] FP8 E4M3 (512 latent | 64 rotated k_pe)
    ctx_len: int
    window_slots: torch.Tensor  # [B, ctx_len + rows] int32: context, then the block
    block_slots: torch.Tensor  # [B, rows] int32: where the block's own rows go


def device_ranges(*roots, depth=10, skip=()):
    """[(start, end)) byte ranges of the CUDA storages reachable from ``roots`` (objects,
    modules, containers; attributes walked to ``depth``), deduplicated; instances of the
    ``skip`` types are not entered."""
    seen, ranges = set(), set()

    def walk(obj, level):
        if level < 0 or id(obj) in seen or obj is None or isinstance(obj, skip):
            return
        seen.add(id(obj))
        if isinstance(obj, torch.Tensor):
            if obj.is_cuda and obj.numel():
                storage = obj.untyped_storage()
                start = storage.data_ptr()
                ranges.add((start, start + storage.nbytes()))
            return
        if isinstance(obj, (str, bytes, int, float, bool, type, types.ModuleType)):
            return
        if isinstance(obj, dict):
            values = list(obj.values())
        elif isinstance(obj, (list, tuple, set, frozenset)):
            values = list(obj)
        else:
            values = list(getattr(obj, "__dict__", {}).values())
        for value in values:
            walk(value, level - 1)

    for root in roots:
        walk(root, depth)
    return sorted(ranges)


class DraftComm:
    """Lamport mailboxes of the draft all-reduces over ``group`` (multicast symm_mem):
    [hops, parity, TP, tokens, 7168] BF16. TP rank r's producer multicasts its partial into
    slot r of every member; with fewer members than TP the other slots are the caller's."""

    def __init__(self, group, device, shapes, layers=5):
        self.tp, self.rank, self.hops = shapes.tp, shapes.rank, 2 * layers
        shape = (self.hops, 2, self.tp, shapes.tokens, HIDDEN)
        self.buffer = symm_mem.empty(shape, dtype=torch.bfloat16, device=device)
        self.handle = handle = symm_mem.rendezvous(self.buffer, group)
        self.mc = int(handle.multicast_ptr or 0)
        if not self.mc:
            raise RuntimeError("the draft chain's all-reduce needs multicast memory")
        self.buffer.view(torch.int32).fill_(-0x80000000)
        # The draft head's rank top-K box (stage 2a): [parity, TP, 7 per request, K x
        # (val, id)], K = shapes.topk; parity as for ``aux`` (serving 0, back-to-back unit
        # runs alternate).
        rows = (shapes.rows - 1) * shapes.requests
        self.head = symm_mem.empty(
            (2, self.tp, rows, 2 * shapes.topk), dtype=torch.int32, device=device
        )
        self.head_mc = int(symm_mem.rendezvous(self.head, group).multicast_ptr or 0)
        if not self.head_mc:
            raise RuntimeError(
                "the draft head's candidate exchange needs multicast memory"
            )
        self.head.fill_(-0x80000000)
        # The aux stage's gather (stage 2b): [parity, TP, tokens, 7168] BF16. One hop per
        # round in serving (the draft hops between rounds order its reuse, parity 0);
        # back-to-back unit runs alternate parity like the target's layers.
        self.aux = symm_mem.empty(
            (2, self.tp, shapes.tokens, HIDDEN), dtype=torch.bfloat16, device=device
        )
        self.aux_mc = int(symm_mem.rendezvous(self.aux, group).multicast_ptr or 0)
        if not self.aux_mc:
            raise RuntimeError("the draft aux gather needs multicast memory")
        self.aux.view(torch.int32).fill_(-0x80000000)
        # The block embedding's anchor-row hop (stage 2c): [parity, tokens, 7168] BF16, one
        # writer (the anchor token's owner rank) per row; parity as for ``aux``.
        self.embed = symm_mem.empty(
            (2, shapes.tokens, HIDDEN), dtype=torch.bfloat16, device=device
        )
        self.embed_mc = int(symm_mem.rendezvous(self.embed, group).multicast_ptr or 0)
        if not self.embed_mc:
            raise RuntimeError("the draft block embedding needs multicast memory")
        self.embed.view(torch.int32).fill_(-0x80000000)
        # The target argmax gather of ArgmaxAccept (stage 3a): [parity, TP, tokens, (value,
        # id)] int32; parity as for ``aux`` (serving 0, an eager warm-up 1).
        self.argmax = symm_mem.empty(
            (2, self.tp, shapes.tokens, 2), dtype=torch.int32, device=device
        )
        self.argmax_mc = int(symm_mem.rendezvous(self.argmax, group).multicast_ptr or 0)
        if not self.argmax_mc:
            raise RuntimeError("the target argmax gather needs multicast memory")
        self.argmax.fill_(-0x80000000)
        torch.accelerator.synchronize(device)
        dist.barrier(group=group)

    def slot(self, hop, parity):
        """This GPU's [TP, tokens, 7168] receive box of ``hop`` at ``parity``."""
        return self.buffer[hop, parity]

    def mc_slot(self, hop, parity):
        return self.mc + (hop * 2 + parity) * self.buffer[0, 0].numel() * 2

    def peer_slot(self, hop, parity, src):
        return self.buffer[hop, parity, src]

    def assert_disjoint(self, *targets):
        """Fail unless no CUDA storage reachable from ``targets`` (the target megakernel's
        executor / mailbox owners) overlaps the draft mailboxes: the draft hop family must
        never alias a target family's Lamport slots."""
        own = device_ranges(self.buffer, self.head, self.aux, self.embed, self.argmax)
        others = device_ranges(
            *targets, skip=(DraftComm,)
        )  # a runner may hold the draft
        if not others:
            raise RuntimeError(
                "draft mailbox check found no CUDA buffers in the targets"
            )
        clashes = [
            (hex(a), hex(b), hex(c), hex(d))
            for a, b in own
            for c, d in others
            if a < d and c < b
        ]
        if clashes:
            raise RuntimeError(f"draft mailboxes alias target buffers: {clashes[:4]}")
        return len(others)


@cache
def compile_fused(
    k, silu, tokens, tp, deadline_ns=0, has_delta=True, permuted=False, delay_ns=0
):
    width = 2 * k if silu else k
    box = fake((tp, tokens, HIDDEN), (tokens * HIDDEN, HIDDEN, 1))
    x, w = fake((tokens, width), (width, 1)), fake((HIDDEN, k), (k, 1))
    delta = fake((tokens, DELTA), (DELTA, 1))
    base = fake((2 * TAPS, HIDDEN), (HIDDEN, 1), torch.float32)
    mc, slot, eps = cutlass.Int64(0), cutlass.Int32(0), cutlass.Float32(1e-6)
    row, weight = dynamic_fake(2), fake((HIDDEN,), (1,))
    rows = (row, dynamic_fake(3), weight, weight, weight, row, None, None, eps, eps)
    kernel = DraftFused(
        k,
        silu=silu,
        deadline_ns=deadline_ns,
        has_delta=has_delta,
        permuted=permuted,
        delay_ns=delay_ns,
    )
    return cute.compile(
        kernel, x, w, box, mc, slot, delta, base, *rows, make_fake_stream()
    )


@cache
def compile_conv(arch):
    base = fake((2 * TAPS, HIDDEN), (HIDDEN, 1), torch.float32)
    args = (dynamic_fake(2), dynamic_fake(2), base, dynamic_fake(2))
    return cute.compile(
        ConvPrepare(), *args, make_fake_stream(), options=f"--gpu-arch={arch}"
    )


@cache
def compile_embed(rows, num_org, local, deadline_ns):
    args = (fake((rows,), (1,), torch.int64, 8), fake((num_org, HIDDEN), (HIDDEN, 1)))
    args += (
        fake((HIDDEN,), (1,)),
        fake((HIDDEN,), (1,)),
        fake((rows, HIDDEN), (HIDDEN, 1)),
    )
    args += (cutlass.Int64(0), cutlass.Int64(0), dynamic_fake(2), dynamic_fake(2))
    args += (cutlass.Float32(1e-6),)
    kernel = EmbedNorm(num_org, local, deadline_ns)
    return cute.compile(kernel, *args, make_fake_stream())


@cache
def compile_norm(arch):
    args = (dynamic_fake(2), fake((HIDDEN,), (1,)), dynamic_fake(2), dynamic_fake(2))
    args += (cutlass.Float32(1e-6),)
    return cute.compile(
        RowNorm(), *args, make_fake_stream(), options=f"--gpu-arch={arch}"
    )


def _bf16(tensor):
    return tensor.detach().to(torch.bfloat16).contiguous()


class _Layer:
    def __init__(self, w: DraftLayerWeights, shapes, device):
        tokens = shapes.tokens
        conv = {tokens: draft_tactic("conv", device)}
        self.input_norm, self.post_norm = _bf16(w.input_norm), _bf16(w.post_norm)
        self.o_proj, self.down = _bf16(w.o_proj), _bf16(w.down)
        if self.o_proj.shape != (HIDDEN, shapes.heads * 128):
            raise ValueError(f"o_proj {tuple(self.o_proj.shape)}")
        if self.down.shape != (HIDDEN, shapes.inter):
            raise ValueError(f"down {tuple(self.down.shape)}")
        bases = (w.attn_conv_base, w.mlp_conv_base)
        self.attn_base, self.mlp_base = (
            b.detach().float().reshape(2 * TAPS, HIDDEN).contiguous() for b in bases
        )
        if CONV_FOLD:
            fold = dict(early_trigger=FOLD_EARLY_TRIGGER)
            self.attn_conv, self.mlp_conv = (
                PreparedConvProjection(p, tokens, conv[tokens], **fold)
                for p in (w.attn_conv_proj, w.mlp_conv_proj)
            )
        else:
            self.attn_conv, self.mlp_conv = (
                PreparedProjection(_bf16(p), conv).prepare()
                for p in (w.attn_conv_proj, w.mlp_conv_proj)
            )
        gate_up = {tokens: draft_tactic("gate_up", device)}
        self.gate_up = PreparedProjection(_bf16(w.gate_up), gate_up).prepare()
        self.views = {
            name: to_cute(getattr(self, name))
            for name in ("o_proj", "down", "attn_base", "mlp_base")
        }
        self.norm_views = {
            name: to_cute(getattr(self, name)) for name in ("input_norm", "post_norm")
        }
        self.weights = w
        self.attention = None  # M1b, built on first use (DraftChain.run with ``core``)


class DraftChain:
    """The draft layers' PDL chain for one shape (``shapes.tokens`` rows)."""

    def __init__(
        self,
        shapes,
        layers,
        final_norm,
        cos_sin,
        comm,
        *,
        device=None,
        targets=(),
        head: Optional[DraftHeadWeights] = None,
        embed: Optional[DraftEmbedWeights] = None,
    ):
        """``targets``: the target megakernel's buffer owners; when given, the build
        fails if a draft mailbox aliases any of their CUDA storages. ``head``: also run
        the candidate head (RankTopK + SelectorWalk) on the block's predicted rows."""
        if shapes.tokens != 8:
            raise ValueError("stage 1 M1a runs one 8-row block (B1)")
        if len(layers) * 2 > comm.hops:
            raise ValueError(f"{len(layers)} layers need {2 * len(layers)} hops")
        self.shapes, self.comm = shapes, comm
        self.target_buffers = comm.assert_disjoint(*targets) if targets else 0
        self.device = device or final_norm.device
        arch = architecture(self.device)
        self.layers = [_Layer(w, shapes, self.device) for w in layers]
        self.final_norm = _bf16(final_norm)
        self.final_view = to_cute(self.final_norm)
        self.cos_sin = cos_sin
        self.eps = cutlass.Float32(shapes.eps)
        tokens, tp = shapes.tokens, shapes.tp
        self.hop_args = dict(tokens=tokens, tp=tp, deadline_ns=DEADLINE_NS)
        self.hop_args.update(permuted=CONV_FOLD, delay_ns=RACE_DELAY_NS)
        self.attn_hop = compile_fused(shapes.heads * 128, False, **self.hop_args)
        self.mlp_hop = compile_fused(shapes.inter, True, **self.hop_args)
        self.conv, self.norm = compile_conv(arch), compile_norm(arch)
        new = lambda width: torch.empty(
            tokens, width, dtype=torch.bfloat16, device=self.device
        )
        self.x_norm, self.x_attn, self.x_post, self.x_mlp = (
            new(HIDDEN) for _ in "1234"
        )
        self.residual = new(HIDDEN)
        self.delta_attn, self.delta_mlp, self.gate_up = (new(DELTA) for _ in "123")
        self.blocks = self.residual.view(tokens, 1, HIDDEN)  # unused at nb = 0
        self.value = new(shapes.heads * 128)  # M1b attention output
        self.sums, self.scratch = new(HIDDEN), new(HIDDEN)  # debug taps
        self.rank = cutlass.Int32(shapes.rank)
        self.embed = None
        if embed is not None:
            self.attach_embed(embed)
        logger.info(
            "draft MK chain: layers=%d tokens=%d tp=%d rank=%d fold=%s (K31DRAFT_CONV_FOLD) "
            "race_delay_ns=%d fold_early_trigger=%s tactics conv=%s gate_up=%s fused_a=%s "
            "embed=%s head=%s target_buffers_disjoint=%d",
            len(self.layers),
            tokens,
            tp,
            shapes.rank,
            "on" if CONV_FOLD else "off",
            RACE_DELAY_NS,
            FOLD_EARLY_TRIGGER,
            draft_tactic("conv", self.device),
            draft_tactic("gate_up", self.device),
            draft_tactic("fused_a", self.device),
            embed is not None,
            head is not None,
            self.target_buffers,
        )
        self.head = self.selector = None
        self.accept = None
        if head is not None:
            self.attach_head(head)

    def attach_head(self, head: DraftHeadWeights):
        """Add the candidate head (RankTopK + SelectorWalk) after construction, e.g. once
        SGLang hands the target lm_head to the draft. Compiles here: attach before any
        eager run with ``anchor`` and before graph capture, never during a capture."""
        if self.head is not None:
            raise RuntimeError("the draft chain already has a candidate head")
        shapes, comm = self.shapes, self.comm
        rows = (shapes.rows - 1) * shapes.requests
        self.head = DraftHead(
            head.lm_head, head.hidden_projection, vocab_start=head.vocab_start,
            rows=rows, tp=shapes.tp, rank=shapes.rank, box=comm.head[0],
            mc=comm.head_mc, device=self.device, topk=shapes.topk,
        )  # fmt: skip
        self.selector = DraftSelector(
            head.predecessor.to(torch.bfloat16).contiguous(),
            head.successor.to(torch.bfloat16).contiguous(),
            box=comm.head[0], head_out=self.head.head_out, num_org=self.head.num_org,
            output_multiplier=head.output_multiplier,
            final_logit_softcapping=head.final_logit_softcapping,
            deadline_ns=DEADLINE_NS, topk=shapes.topk,
        )  # fmt: skip
        logger.info(
            "draft MK head: num_org=%d vocab_start=%d head tactic=%s rank top-k=8-CTA cluster "
            "top_k=%d merges=%s",
            self.head.num_org,
            head.vocab_start,
            draft_tactic("head", self.device),
            shapes.topk,
            "bitonic" if BITONIC else "serial",
        )

    def attach_embed(self, embed: DraftEmbedWeights):
        """Add the on-device block embedding (stage 2c, ``EmbedNorm``) after construction, e.g.
        once SGLang hands the target's embedding to the draft. Compiles here: attach on every
        rank before any run with ``ids`` and before graph capture, never during a capture."""
        if self.embed is not None:
            raise RuntimeError("the draft chain already has a block embedding")
        shard, mask_row = embed.shard, embed.mask_row
        if (
            shard.dim() != 2
            or shard.shape[1] != HIDDEN
            or tuple(mask_row.shape) != (HIDDEN,)
        ):
            raise ValueError(
                f"embedding shard {tuple(shard.shape)} / mask row {tuple(mask_row.shape)}"
            )
        if int(embed.vocab_start) < 0:
            raise ValueError(f"vocab_start {embed.vocab_start}")
        self.embed = (_bf16(shard), int(embed.vocab_start), _bf16(mask_row))
        self.embed_code = compile_embed(
            self.shapes.tokens, shard.shape[0], False, DEADLINE_NS
        )
        logger.info(
            "draft MK embed: block embedding on device (num_org=%d vocab_start=%d)",
            shard.shape[0],
            int(embed.vocab_start),
        )

    def attach_accept(self, sim_real_token=True):
        """Add the target argmax gather + DFlash accept (stage 3a, ``ArgmaxAccept``) on this
        chain's comm. Compiles here (before any capture); returns the ``DraftAccept``."""
        from .accept import DraftAccept

        if self.accept is not None:
            raise RuntimeError("the draft chain already has an accept stage")
        self.accept = DraftAccept(
            self.comm,
            self.shapes,
            sim_real_token=sim_real_token,
            deadline_ns=DEADLINE_NS,
        )
        logger.info(
            "draft MK accept: ArgmaxAccept over TP%d, B%d, sim token %s",
            self.shapes.tp,
            self.shapes.requests,
            "real" if sim_real_token else "fixed",
        )
        return self.accept

    def _conv(self, x, delta, base, out):
        views = (dynamic_view(x), dynamic_view(delta), base, dynamic_view(out))
        self.conv(*views, stream(self.device))

    def _conv_prepare(self, projection, x, delta, layer, side, out):
        """δ = projection(x), then x' = conv_side0(x): two launches, or one folded."""
        if CONV_FOLD:
            projection.run(x, delta, getattr(layer, f"{side}_base"), out)
        else:
            projection.run(x, delta)
            self._conv(x, delta, layer.views[f"{side}_base"], out)

    def _hop(self, code, x, weight, hop, parity, delta, base, norm, out, prefix=None):
        box = to_cute(self.comm.slot(hop, parity))
        mc = cutlass.Int64(self.comm.mc_slot(hop, parity))
        prefix = self.residual if prefix is None else prefix
        rows = (dynamic_view(prefix), dynamic_view(self.blocks))
        rows += (norm, norm, norm, dynamic_view(out), None, None, self.eps, self.eps)
        head = (to_cute(x), weight, box, mc, self.rank, to_cute(delta), base)
        code(*head, *rows, stream(self.device))

    def _tapped_hop(self, silu, x, weight, hop, parity, delta, base, norm, out):
        """Debug taps: the same hop with the has_delta=False consumer, so ``self.sums``
        receives the exact BF16 all-reduce sum; then the residual add (BF16, as the
        consumer's) and RMSNorm (RowNorm) of the fused hop."""
        k = self.shapes.inter if silu else self.shapes.heads * 128
        code = compile_fused(k, silu, **self.hop_args, has_delta=False)
        self._hop(
            code, x, weight, hop, parity, delta, base, norm, self.scratch, self.sums
        )
        self.residual.copy_(
            (self.residual.float() + self.sums.float()).to(torch.bfloat16)
        )
        views = (dynamic_view(self.residual), norm, dynamic_view(self.scratch))
        views += (dynamic_view(out), self.eps)
        self.norm(*views, stream(self.device))
        return self.sums

    def _attention(self, i, layer, positions, kv, slots, core, mark):
        if layer.attention is None:
            layer.attention = DraftAttention(
                layer.weights,
                self.shapes,
                draft_tactic("fused_a", self.device),
                self.device,
            )
        pool = kv.pool[i]
        cache = pool.view(-1, 1, pool.shape[-1]) if pool.dim() == 2 else pool
        return layer.attention.run(
            self.x_attn,
            positions,
            self.cos_sin,
            cache,
            slots,
            lambda query, latent: core(i, query, latent),
            self.value,
            mark,
        )

    def run(
        self,
        x0: torch.Tensor,
        positions: torch.Tensor,
        kv: Optional[DraftKV],
        out: torch.Tensor,
        parity: int,
        *,
        attn: Optional[Callable] = None,
        core: Optional[Callable] = None,
        taps: Optional[dict] = None,
        stamp: Optional[Callable] = None,
        fine_stamps: bool = False,
        anchor: Optional[torch.Tensor] = None,
        ids: Optional[torch.Tensor] = None,
        embed_parity: int = 0,
    ):
        """x0 [tokens, 7168] BF16 block embeddings -> out [tokens, 7168] BF16 (final
        RMSNorm of the last layer's hidden + residual). M1a: ``attn(layer, x)`` maps the
        conv-prepared attention input [tokens, 7168] to the W_UV output [tokens, 768].
        M1b: ``core(layer, query_fp8 [tokens, heads, 576], latent [tokens, heads, 512])``
        is the attention core; the block rows' KV goes to ``kv.pool[layer]`` at
        ``kv.block_slots``."""
        if (attn is None) == (core is None):
            raise ValueError("pass exactly one of attn (M1a) and core (M1b)")
        if core is not None:
            slots = kv.block_slots.reshape(-1).to(torch.int64)
        tap = (lambda name, i, t: taps[name][i].copy_(t)) if taps else None
        # Exact all-reduce sums (eager taps only): "attn_ar" / "mlp_ar" run each hop with
        # the has_delta=False consumer; "attn_sum" / "mlp_sum" stay residual differences.
        exact = bool(taps) and "attn_ar" in taps and "mlp_ar" in taps
        mark = stamp or (lambda tag: None)
        # fine_stamps: also L{i}:conv / :fused_a / :query / :core / :value (M1b) /
        # :mlp_conv / :gate_up, between the coarse L{i}:in / :attn_sum / :out tags.
        fine = lambda i: (lambda tag: mark(f"L{i}:{tag}")) if fine_stamps else None  # noqa: E731
        fmark = lambda i, tag: fine_stamps and mark(f"L{i}:{tag}")  # noqa: E731
        s = stream(self.device)
        mark("start")
        if ids is not None:  # stage 2c: embed the block on device (x0 unused)
            if self.embed is None:
                raise RuntimeError("run(ids=...) needs attach_embed() first")
            shard, vocab_start, mask_row = self.embed
            box = self.comm.embed[embed_parity]
            mc = self.comm.embed_mc + embed_parity * box.numel() * 2
            args = (to_cute(ids, 8), to_cute(shard), to_cute(mask_row))
            args += (self.layers[0].norm_views["input_norm"], to_cute(box))
            args += (cutlass.Int64(mc), cutlass.Int64(vocab_start))
            args += (dynamic_view(self.residual), dynamic_view(self.x_norm), self.eps)
            self.embed_code(*args, s)
        else:
            views = (dynamic_view(x0), self.layers[0].norm_views["input_norm"])
            views += (dynamic_view(self.residual), dynamic_view(self.x_norm), self.eps)
            self.norm(*views, s)
        for i, layer in enumerate(self.layers):
            mark(f"L{i}:in")
            if tap:
                tap("x_norm", i, self.x_norm)
            self._conv_prepare(
                layer.attn_conv,
                self.x_norm,
                self.delta_attn,
                layer,
                "attn",
                self.x_attn,
            )
            fmark(i, "conv")
            if tap:
                tap("x_attn_in", i, self.x_attn)
                before = self.residual.float()
            if core is None:
                value = attn(i, self.x_attn)
            else:
                value = self._attention(i, layer, positions, kv, slots, core, fine(i))
            if value.shape != (self.shapes.tokens, self.shapes.heads * 128):
                raise ValueError(f"attn output {tuple(value.shape)}")
            if not value.is_contiguous():
                raise ValueError("attn output must be contiguous")
            post = layer.norm_views["post_norm"]
            hop = (value, layer.views["o_proj"], 2 * i, parity, self.delta_attn)
            hop += (layer.views["attn_base"], post, self.x_post)
            if exact:
                tap("attn_ar", i, self._tapped_hop(False, *hop))
            else:
                self._hop(self.attn_hop, *hop)
            mark(f"L{i}:attn_sum")
            if tap:
                tap("attn_sum", i, self.residual.float() - before)
                tap("residual", i, self.residual)
                tap("x_post", i, self.x_post)
            self._conv_prepare(
                layer.mlp_conv, self.x_post, self.delta_mlp, layer, "mlp", self.x_mlp
            )
            fmark(i, "mlp_conv")
            if tap:
                tap("x_mlp_in", i, self.x_mlp)
                before = self.residual.float()
            layer.gate_up.run(self.x_mlp, self.gate_up)
            fmark(i, "gate_up")
            last = i == len(self.layers) - 1
            norm = (
                self.final_view if last else self.layers[i + 1].norm_views["input_norm"]
            )
            hop = (self.gate_up, layer.views["down"], 2 * i + 1, parity, self.delta_mlp)
            hop += (layer.views["mlp_base"], norm, out if last else self.x_norm)
            if exact:
                tap("mlp_ar", i, self._tapped_hop(True, *hop))
            else:
                self._hop(self.mlp_hop, *hop)
            mark(f"L{i}:out")
            if tap:
                tap("mlp_sum", i, self.residual.float() - before)
                tap("residual_out", i, self.residual)
        mark("end")
        if self.head is not None:
            # B1: the block's predicted rows 1..7 (row 0 is the anchor).
            self.head.run(out[1 : self.shapes.rows])
            mark("head")
            if anchor is None:
                raise ValueError(
                    "the candidate head needs the block's anchor token ids"
                )
            self.selector.run(anchor)
            mark("select")
            return out, (
                self.selector.tokens,
                self.selector.candidates,
                self.selector.q,
            )
        return out
