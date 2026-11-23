# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Component ``fp8_prefill`` (env ``MUSE_VR_FP8_PREFILL=1``, default off): MXFP8 dense projections on large prefill
passes of the Muse Spark target model.

Below the shipped checkpoint's precision (the dense projections ship in BF16), opt-in: on EXTEND passes of the target
model with more than ``MUSE_VR_FP8_PREFILL_MIN_TOKENS`` tokens (default 1280; must be a captured prefill bucket, the
rule ``sp_gate`` uses, so a replayed bucket and the pass it serves take the same form) the projections named in
``MUSE_VR_FP8_PREFILL_GROUPS`` (default ``qkv,o,pool,unpool``) run on MXFP8 operands: E4M3 values with one power-of-two
scale per block of 32 elements along K, for the weight and the activation (``mxfp8_ops``: FlashInfer's quantizer and
GEMM, as SGLang's MXFP8 linear method). Decode, target-verify and mixed passes, the draft model, the LM head, the
embeddings and the experts are never touched; every projection keeps its BF16 weight for those.

Per group:
- ``qkv`` (sequence-parallel all-gather + GEMM): each rank quantizes its row shard before the all-gather, so the peers
  pull E4M3 bytes plus scales instead of BF16 (copy-engine pulls from symmetric storage with stream-memory flags, the
  ordering of SGLang's ``_all_gather_matmul_cublas``), and each shard's GEMM runs as its rows land.
- ``o``: through ``o_rowshard`` (its MXFP8 form: the all-to-all moves E4M3 rows and their scales; one full-width GEMM);
  needs ``o_rowshard`` engaged.
- ``pool`` / ``unpool`` (replicated weights, sequence-parallel row shards): the BF16 dispatch call for those weights runs
  the MXFP8 GEMM instead.
Weights: ``MUSE_VR_FP8_PREFILL_WEIGHTS=jit`` (default) quantizes each weight on every FP8 pass (no resident copy);
``resident`` keeps MXFP8 copies from load (about 1.03 bytes per weight element on top of BF16); with
``MUSE_VR_FP8_PREFILL_RESIDENT_LAYERS=N`` only layers 0 .. N - 1 keep them and the other layers quantize just in time
(install record ``resident_layers`` / ``resident_bytes``; a memory budget between the two forms). Both use the same
quantizer, so the operands are the same bytes. With ``jit``, ``MUSE_VR_FP8_PREFILL_WQ_PREFETCH=1`` quantizes the weights
one step ahead on a side stream (at a layer's o_proj: that layer's pool / unpool and the next layer's qkv, all used
inside the same CUDA-graph segment, each use waiting for its own event); otherwise each is quantized right before use.
Quantizer: ``MUSE_VR_FP8_PREFILL_FQ=1`` runs every quantization through ``mxfp8_fq`` (the same bytes as FlashInfer's
quantizer, several loads in flight per thread); ``=2`` additionally quantizes a layer's just-in-time qkv / pool / unpool
weights in one launch at the layer's first weight use, into one reused slot per group (``jit`` without prefetch only;
about one layer's MXFP8 weights of extra memory). Default 0: FlashInfer's quantizer.
Fused producers: ``MUSE_VR_FP8_PREFILL_B3`` (default empty = off) lists the boundary norms that also write the MXFP8 of
their output (``ops.b3_norm_mx``: SGLang's boundary kernel with an MXFP8 epilogue; the BF16 output, the residual stream
and the router copy are still written, byte-equal to SGLang's): ``U`` = the FFN boundary that produces the next layer's
qkv input (its MXFP8 goes straight into the qkv all-gather staging, so the qkv entry skips its quantize and staging
copies; the "previous pulls are complete" barrier moves before the boundary norm), ``A`` = the attention boundary that
produces the pool GEMM's input (into a rank-local buffer the pool call uses). The bytes the GEMMs read are the ones the
unfused path quantizes. A boundary call takes this path only when the projection it feeds will take its MXFP8 path on
this pass; every other call is SGLang's, counted ``b3:<pos>:stock:<reason>``. ``MUSE_VR_FP8_PREFILL_B3_CHECK=1`` (a
test aid, never set on a served configuration) compares every eager fused call with the unfused path, byte for byte.
Two-deep staging: ``MUSE_VR_FP8_PREFILL_AG2DEEP=1`` (default off) gives the qkv all-gather two staging slots, picked by the
layer index's parity, and leaves out the "every peer's previous pulls from this staging are complete" barrier (at the qkv
entry, or before the ``U`` boundary norm). A slot is rewritten two layers after its last use; in between, the next layer's
all-gather issues its "every rank's shard is staged" barrier, which every peer reaches only after the pulls of the
previous use have completed in its stream order (the copies, then the flag write on the copy stream, then the flag wait
on the main stream). The decoder stack must have an even number of layers (so the last layer's barrier also separates a
forward's last use of a slot from the next forward's first one); otherwise the option stays off. One more staging slot
(the shard capacity's MXFP8 bytes) is allocated. Numerics unchanged (synchronization only).
Push all-gather: ``MUSE_VR_FP8_PREFILL_AGF=1`` (default off) replaces the qkv all-gather's staged pulls with a push. Every
rank stores its MXFP8 shard (the ``U`` boundary's output, else the quantizer's) through the multicast address of a
symmetric gathered operand (``ops.mc_push``: one launch, every rank's copy receives the bytes), then one bounded barrier
(``ops.a2a_pull.barrier``: release / acquire at system scope, with ``fence.proxy.alias`` on both sides), then ONE GEMM
over the ``world x rows`` gathered rows (the per-rank row blocks are whole 128-row scale tiles, so their swizzled scales
concatenate); padding rows between the blocks are dropped from the output. No staging barrier, copy-engine pull, flag
write or flag wait. The gathered operand has two slots picked by the layer's parity (an even layer count is required),
so a slot is rewritten only after every rank has passed the next layer's barrier, which follows its GEMM on this slot in
stream order. ``MUSE_VR_FP8_PREFILL_AGF_CAP_ROWS`` (rows per rank the slots hold, whole 128-row tiles; default the shard
capacity) sizes the slots; a pass whose shard is larger takes the staged form for ALL its layers (counted
``agf:stock:rows>cap``), AG2DEEP included when set, so one pass never mixes the two forms. With the default cap every pass
takes AGF: the local staging then has one slot (only this rank reads it) and no staged-form buffer is allocated, so AGF
subsumes AG2DEEP (no ``ag2deep:*`` count is written). With ``rs_samem`` (MUSE_VR_RS_SAMEM), the full all-ranks barrier
it relies on between one layer's GEMM + reduce-scatter and the next is, on an AGF pass, this bounded barrier (issued on
every AGF call before it returns); on a staged-form pass it is the staged barrier, as before. Counted ``agf:slot<s>``,
``agf:push:b3`` /
``agf:push:quant``; the barrier's polls are bounded (``MUSE_VR_FP8_PREFILL_AGF_MAX_POLLS``, then a trap). Needs a
multicast mapping (else the option stays off, reported). Numerics: the GEMM's rows are the same products as the per-rank
calls (one call over all rows instead of four); the bytes moved are unchanged.
Dense push: ``MUSE_VR_FP8_PREFILL_AGF_DENSE=1`` (default off; needs ``MUSE_VR_FP8_PREFILL_AGF=1``, else the component
refuses the configuration) lands each rank's rows at rank x rows instead of whole 128-row tiles (``ops.mc_push_dense``: the
values densely, the scales scattered as whole 4-byte words of the gathered operand's own swizzled layout), so the gathered
operand is the dense [world x rows, K] operand and the ONE GEMM over its world x rows rows writes the final output with no
padding rows to drop. Same slots (sizes unchanged), barrier, slot parity, row cap and producers; the gathered buffer is
zeroed once at allocation (the GEMM reads the scale words of the last tile's padding rows for output rows it does not
store). Counted ``agf:dense``. Numerics as AGF (the same row products; one call over world x rows rows).
Unicast dense push: ``MUSE_VR_FP8_PREFILL_AGF_UC=1`` (default off; needs ``MUSE_VR_FP8_PREFILL_AGF_DENSE=1``) stores the same
bytes at the same places once per rank through each rank's mapping of its slot (``ops.uc_push_dense``) instead of once
through the multicast address; no multicast mapping is needed then. Every storing thread ends with an alias fence;
``MUSE_VR_FP8_PREFILL_AGF_UC_NOFENCE_UNSAFE=1`` leaves it out (the barrier's alias fence is then the only one; not guaranteed
by the PTX memory model). Counted ``agf:uc`` next to ``agf:dense``; numerics as the dense push (the same bytes).
GEMM configuration:``MUSE_VR_FP8_PREFILL_TACTICS`` (a table path) with ``MUSE_VR_FP8_PREFILL_PIN_RUNNER``
(``module:function`` of the CuTe-DSL runner factory for this GPU, from the deployment's private configuration) pins the
GEMM's configuration per shape and power-of-two row bucket (a table in which every entry gives the default configuration's
bits), or per row band when the table declares ``"__lookup__": "bands"`` (``mxfp8_ops.load_pinned``; the install record's
``pinned_lookup``); without them the GEMM's own default selection runs. No autotuning at start-up.

Every call it does not take runs SGLang's code, counted ``stock:<reason>``; taken calls are counted per group and row
bucket. Engagement needs the sha256 pins of the SGLang sources whose call structure it relies on.
"""

from __future__ import annotations

import functools
import logging
import os
import threading

import torch

from . import bed

# The SGLang names this module reads (test/unit/test_contract.py pins them against the supported SGLang)
SGLANG_READS = {
    "srt/layers/layernorm_sp.py": ("sp_num_tokens",),
    "srt/model_executor/model_runner.py": ("ModelRunner",),
    "srt/models/muse_spark.py": ("MuseSpark1xModel",),
    "kernels/ops/layernorm/muse_spark_norm_residual.py": (
        "FusedPostNormResidualPreNorm",
    ),
    "srt/model_executor/forward_batch_info.py": ("ForwardMode",),
    "srt/runtime_context.py": ("get_parallel",),
}


logger = logging.getLogger("muse_vr_plugin")

NAME = "fp8_prefill"
GROUPS_ALL = ("qkv", "o", "pool", "unpool")
DEFAULT_MIN_TOKENS = 1280
SGLANG_PINS = {
    "sglang/srt/layers/layernorm_sp.py": "bc814d983b945e5bdd41db436966eb4b3fee4c410bb1dcccfc98857d4b026bda",
    "sglang/srt/layers/linear.py": "3ce31df0a8aab09ed8afa0ada8c0ce30ca207a748c02c3073d7f53648b9b7c3c",
    "sglang/srt/models/muse_spark.py": (
        "8bafbb8802fb0d9f81c190e87b9012fca546182be08bef6c8b3db814d27bfe19",
        "c0db42467a960c509b658c3c40570ee889f73d41b52cdbd80b9c291c1792a33f",
    ),
    "sglang/srt/layers/quantization/unquant.py": "7285d3e9e9fa881c5ecd97d093a8f59e11e69f26d5e3e7006925149f624a2f9c",
    "sglang/kernels/ops/communication/sp_collective_gemm.py": "f1cc99967104416a31fe7c99a710a9ccf8024ac1579c08579f11ac68a476a836",
}

# the fused producers (MUSE_VR_FP8_PREFILL_B3) mirror SGLang's boundary op, its launch policy and its kernel
B3_PINS = {
    "sglang/kernels/ops/layernorm/muse_spark_norm_residual.py": "454f084d20e5f2353a70d260ea4a70c7dfdff3bf45a58619ef72e1eec2646fe3",
    "sglang/kernels/ops/layernorm/muse_spark_norm_residual_jit.py": "76bd0c26ed4d3cf87126619c72a56de908dea0ea1b2c69f6e5ada47318035cbe",
    "sglang/kernels/jit/csrc/muse_spark/norm_residual.cuh": "c227348c4d705183d377ac623a62b1ed6957eb8832fb0508f4ac99f1aa3cb0c8",
}
B3_POSITIONS = ("U", "A")

_lock = threading.Lock()
_state: dict = {
    "engaged": None,
    "pass": False,
    "ctx": None,
    "groups": (),
    "orig_g": None,
    "orig_dispatch": None,
    "orig_forward": None,
    "orig_gbar": None,
    "min_tokens": DEFAULT_MIN_TOKENS,
    "backend": "cute-dsl",
    "weights": "jit",
    "b3": (),
    "orig_boundary": None,
    "b3_check": False,
    "logged": set(),
}


def pass_fp8(group: str) -> bool:
    """True inside a target-model forward that runs MXFP8 for ``group``."""
    return _state["pass"] and group in _state["groups"]


def _bucket(m: int) -> str:
    b = 1
    while b < m:
        b <<= 1
    return f"m<={b}"


def _copy(dst: int, src: int, nbytes: int, stream) -> None:
    import cuda.bindings.driver as cuda

    (status,) = cuda.cuMemcpyDtoDAsync(
        dst, src, nbytes, cuda.CUstream(stream.cuda_stream)
    )
    if status != cuda.CUresult.CUDA_SUCCESS:
        raise RuntimeError(f"fp8_prefill: peer copy failed: {status}")


class Fp8Ctx:
    """Per-process state: registered weights, resident copies, the qkv all-gather workspace."""

    def __init__(
        self,
        layers,
        group,
        rank: int,
        world: int,
        cap_rows: int,
        weights: str,
        backend: str,
    ):
        import torch.distributed._symmetric_memory as symm_mem

        from . import mxfp8_ops as mx

        self.mx, self.backend, self.weights = mx, backend, weights
        self.rank, self.world = rank, world
        self.qkv = {}  # id(linear) -> layer
        self.reg = {}  # (data_ptr, shape) -> (group, layer)
        self.res = {}  # (group, layer) -> (q, s) resident copies
        self.bf16 = {}  # (group, layer) -> BF16 weight tensor
        for li, (attn, moe) in enumerate(layers):
            self.qkv[id(attn.qkv_proj)] = li
            self.bf16[("qkv", li)] = attn.qkv_proj.weight
            for g, attr in (
                ("pool", "pre_expert_proj"),
                ("unpool", "post_expert_proj"),
            ):
                lin = getattr(moe, attr, None) if moe is not None else None
                if lin is not None:
                    w = lin.weight
                    self.reg[(w.data_ptr(), tuple(w.shape))] = (g, li)
                    self.bf16[(g, li)] = w
        # MUSE_VR_FP8_PREFILL_RESIDENT_LAYERS=N (with weights=resident; default: every layer): only layers 0 .. N - 1 keep
        # resident copies; the other layers' weights are quantized just in time (the same quantizer, the same bytes)
        self.resident_layers = 0
        if weights == "resident":
            n_res = bed.env_int("MUSE_VR_FP8_PREFILL_RESIDENT_LAYERS", len(layers))
            if not 0 <= n_res <= len(layers):
                raise ValueError(
                    f"MUSE_VR_FP8_PREFILL_RESIDENT_LAYERS={n_res} outside 0 .. {len(layers)}"
                )
            self.resident_layers = n_res
            for key, w in self.bf16.items():
                if key[1] < n_res:
                    self.res[key] = mx.quantize(w.data.contiguous())
        w0 = layers[0][0].qkv_proj.weight
        self.k = int(w0.shape[1])
        self.cap = -(-int(cap_rows) // world)  # rows of one rank's shard
        self.cap = -(-self.cap // mx.SF_TILE_ROWS) * mx.SF_TILE_ROWS
        dev = w0.device
        self.sfb = mx.sf_bytes(self.cap, self.k)
        # push all-gather (MUSE_VR_FP8_PREFILL_AGF): two symmetric gathered slots with a multicast mapping (layer parity),
        # each [world x cap x K] values then [world x sf_bytes(cap)] scales, and the bounded barrier's signal pads
        self.agf, self.agf_why, self.agf_cap, self.agf_dense = False, "off", 0, False
        self.agf_uc, self.agf_uc_fence, self.agf_ptrs = False, True, []
        if _state.get("agf"):
            self.agf_why = "odd layer count or one rank"
            if len(layers) % 2 == 0 and world > 1:
                from . import o_rowshard

                # rows per rank the slots hold (MUSE_VR_FP8_PREFILL_AGF_CAP_ROWS, default the shard capacity, whole 128-row
                # tiles); a pass whose shard is larger takes the staged form for ALL its layers (agf:stock:rows>cap)
                acap = bed.env_int("MUSE_VR_FP8_PREFILL_AGF_CAP_ROWS", self.cap)
                self.agf_cap = min(
                    self.cap, -(-max(acap, 1) // mx.SF_TILE_ROWS) * mx.SF_TILE_ROWS
                )
                self.agf_vbytes = world * self.agf_cap * self.k
                self.agf_slot = self.agf_vbytes + world * mx.sf_bytes(
                    self.agf_cap, self.k
                )
                self.agf_buf = symm_mem.empty(
                    (2 * self.agf_slot,), dtype=torch.uint8, device=dev
                )
                self.agf_dense = bool(_state.get("agf_dense"))
                if self.agf_dense:
                    # MUSE_VR_FP8_PREFILL_AGF_DENSE: zeroed once, before the rendezvous and the init barrier (no peer stores
                    # into this copy before both); the scale words of the last tile's padding rows are never stored
                    self.agf_buf.zero_()
                self.agf_handle = symm_mem.rendezvous(self.agf_buf, group)
                self.agf_mc = int(getattr(self.agf_handle, "multicast_ptr", 0) or 0)
                self.agf_pad = symm_mem.empty((world,), dtype=torch.int32, device=dev)
                self.agf_pad.zero_()
                torch.cuda.synchronize(dev)
                self.agf_phandle = symm_mem.rendezvous(self.agf_pad, group)
                self.agf_pads = [
                    int(self.agf_phandle.buffer_ptrs[p]) for p in range(world)
                ]
                self.agf_polls = bed.env_int(
                    "MUSE_VR_FP8_PREFILL_AGF_MAX_POLLS",
                    o_rowshard.DEFAULT_BARRIER_MAX_POLLS,
                )
                # MUSE_VR_FP8_PREFILL_AGF_UC: the dense push stores through every rank's mapping of its slot (no multicast)
                self.agf_uc = bool(_state.get("agf_uc")) and self.agf_dense
                self.agf_uc_fence = bool(_state.get("agf_uc_fence", True))
                self.agf_ptrs = (
                    [int(self.agf_handle.buffer_ptrs[p]) for p in range(world)]
                    if self.agf_uc
                    else []
                )
                self.agf = self.agf_mc != 0 or self.agf_uc
                self.agf_dense = self.agf_dense and self.agf
                self.agf_why = "" if self.agf else "no multicast mapping"
        # staging slots: 2 with MUSE_VR_FP8_PREFILL_AG2DEEP (slot = layer parity; even layer count required), else 1. With
        # AGF covering every pass (agf_cap == cap) the staging has 1 slot (only this rank reads it) and the staged form's
        # gathered buffer is not allocated; with a row cap the declined passes keep the staged form, L2 included.
        agf_all = self.agf and self.agf_cap >= self.cap
        self.depth = (
            2 if (_state.get("ag2deep") and len(layers) % 2 == 0 and not agf_all) else 1
        )
        self.half = self.cap * self.k + self.sfb
        self.stage = symm_mem.empty(
            (self.depth * self.half,), dtype=torch.uint8, device=dev
        )
        self.handle = symm_mem.rendezvous(self.stage, group)
        self.gathered = (
            None
            if agf_all
            else torch.empty(
                (world, self.cap * self.k + self.sfb), dtype=torch.uint8, device=dev
            )
        )
        self.flags = torch.zeros(world, dtype=torch.int32, device=dev)
        self.copy_stream = torch.cuda.Stream(device=dev)
        # just-in-time weights quantized one segment ahead on a side stream (MUSE_VR_FP8_PREFILL_WQ_PREFETCH=1)
        self.wq_prefetch = bed.env_int("MUSE_VR_FP8_PREFILL_WQ_PREFETCH", 0) != 0
        self.wq_stream = torch.cuda.Stream(device=dev)
        self.slots = {}  # group -> (q, s) slot the side stream writes
        self.pre = {}  # (group, layer) -> (event, q, s) not yet used
        self.o_layer = {id(attn.o_proj): li for li, (attn, _m) in enumerate(layers)}
        self.n_layers = len(layers)
        for li, (attn, _m) in enumerate(layers):  # slots sized now, outside any capture
            for g in ("qkv", "pool", "unpool"):
                if (
                    (g, li) in self.bf16
                    and self.wq_prefetch
                    and weights == "jit"
                    and g not in self.slots
                ):
                    q, s = mx.quantize(self.bf16[(g, li)].data.contiguous())
                    self.slots[g] = (torch.empty_like(q), torch.empty_like(s))
        # MUSE_VR_FP8_PREFILL_FQ=2 with just-in-time weights (no prefetch): one grouped launch per layer quantizes that
        # layer's qkv / pool / unpool weights into one slot per group at the layer's first weight use
        # fused producers (MUSE_VR_FP8_PREFILL_B3): boundary op id -> (position, layer of the projection it feeds)
        self.b3_map = {}
        self.b3u = None  # (layer, output data_ptr, rows) staged by the U boundary for the next qkv entry
        self.b3a = None  # (layer, output data_ptr, rows, q, s) quantized by the A boundary for the next pool call
        self.b3a_q = self.b3a_s = None
        self.fq_group = mx.fq_mode() >= 2 and weights == "jit" and not self.wq_prefetch
        self.gslots = {}  # group -> (q, s)
        self.gpre = {}  # (group, layer) -> (q, s) quantized by the grouped launch, not yet used
        if self.fq_group:
            from . import mxfp8_fq

            for g in ("qkv", "pool", "unpool"):
                w = next((w for (gg, _li), w in self.bf16.items() if gg == g), None)
                if w is not None and mxfp8_fq.takes(w.data):
                    self.gslots[g] = mxfp8_fq.alloc(w.data)
        self.handle.barrier(channel=0)

    def warmup(self, shapes, max_rows: int) -> int:
        """Run the quantizer and the GEMM once per (N, K) in ``shapes`` and per power-of-two row count up to ``max_rows``
        (the GEMM picks its configuration per power-of-two row bucket), so every kernel a captured prefill bucket uses is
        built before any CUDA-graph capture. With a pinned table in the "bands" lookup form, the GEMM also runs once at
        every row key of that shape up to ``max_rows`` (each key may name its own configuration). Returns the number of
        calls."""
        mx, dev, n_calls = self.mx, self.stage.device, 0
        bands = mx.pinned_lookup() == "bands"
        for n, k in sorted(set(shapes)):
            wq, ws = mx.quantize(torch.zeros((n, k), dtype=torch.bfloat16, device=dev))
            rows = []
            m = 128
            while m <= max_rows:
                rows.append(m)
                m *= 2
            if bands:
                rows += [
                    r
                    for r in mx.pinned_row_keys(n, k)
                    if r <= max_rows and r not in rows
                ]
            for m in rows:
                q, s = mx.quantize(
                    torch.zeros((m, k), dtype=torch.bfloat16, device=dev)
                )
                mx.gemm(q, s, wq, ws, self.backend)
                n_calls += 1
        torch.cuda.synchronize(dev)
        return n_calls

    def weight(self, group: str, layer: int):
        key = (group, layer)
        if key in self.res:
            return self.res[key]
        pf = self.pre.pop(key, None)
        if (
            pf is not None
        ):  # quantized ahead on the side stream: wait for it, then use the slot
            ev, q, s = pf
            torch.cuda.current_stream().wait_event(ev)
            bed.count(NAME, f"wq:prefetched:{group}")
            return q, s
        if self.fq_group and group in self.gslots:
            hit = self.gpre.pop(key, None)
            if hit is None:
                self._group_quant(layer)
                hit = self.gpre.pop(key, None)
            if hit is not None:
                bed.count(NAME, f"wq:grouped:{group}")
                return hit
        bed.count(NAME, f"wq:inline:{group}")
        return self.mx.quantize(self.bf16[key].data.contiguous())

    def _group_quant(self, layer: int) -> None:
        """One launch for this layer's slotted, enabled weights (stream-ordered after every earlier use of the slots on
        this stream). Entries of other layers are dropped, so a slot is only handed out for the layer it holds."""
        from . import mxfp8_fq

        keys = [
            (g, layer)
            for g in ("qkv", "pool", "unpool")
            if g in self.gslots
            and g in _state["groups"]
            and (g, layer) in self.bf16
            and self.in_range(layer)
            and mxfp8_fq.takes(self.bf16[(g, layer)].data)
        ]
        self.gpre = {}
        if not keys:
            return
        outs = [self.gslots[g] for g, _li in keys]
        mxfp8_fq.quantize_group_into([self.bf16[k].data for k in keys], outs)
        bed.count(NAME, f"fq:group:{len(keys)}")
        self.gpre = dict(zip(keys, outs))

    def prefetch_weights(self, layer: int, n_layers: int) -> None:
        """Quantize, on the side stream, the weights this pass needs next within the same CUDA-graph segment: pool / unpool
        of ``layer`` and qkv of ``layer + 1`` (called at ``layer``'s o_proj, the first SP projection after its attention).
        Each lands in its own slot; its use waits for its event. The side stream first waits for the main stream, which has
        already enqueued every earlier use of those slots."""
        if self.weights != "jit" or not self.wq_prefetch:
            return
        main = torch.cuda.current_stream()
        self.wq_stream.wait_stream(main)
        keys = [("pool", layer), ("unpool", layer)] + (
            [("qkv", layer + 1)] if layer + 1 < n_layers else []
        )
        with torch.cuda.stream(self.wq_stream):
            for key in keys:
                if key not in self.bf16:
                    continue
                q, s = self.mx.quantize(self.bf16[key].data.contiguous())
                slot = self.slots.get(key[0])
                if (
                    slot is None
                    or slot[0].shape != q.shape
                    or slot[1].numel() < s.numel()
                ):
                    slot = (torch.empty_like(q), torch.empty_like(s))
                    self.slots[key[0]] = slot
                slot[0].copy_(q)
                slot[1][: s.numel()].copy_(s)
                ev = torch.cuda.Event()
                ev.record(self.wq_stream)
                self.pre[key] = (ev, slot[0], slot[1][: s.numel()])

    def agf_pass(self, rows: int) -> bool:
        """True if a pass whose shard has ``rows`` rows takes the push all-gather (AGF on and the shard fits its slots). The
        row count is the same for every layer of a pass, so a pass takes one form for all its layers."""
        return (
            self.agf
            and -(-rows // self.mx.SF_TILE_ROWS) * self.mx.SF_TILE_ROWS <= self.agf_cap
        )

    def slot_base(self, layer: int) -> int:
        """Byte offset of the staging slot the qkv all-gather of decoder ``layer`` uses (its parity with two-deep staging)."""
        return (layer % self.depth) * self.half

    def set_b3(self, decoders, positions) -> None:
        """Map the boundary ops whose output feeds an MXFP8 projection: decoder i's FFN boundary -> the qkv of decoder
        i + 1 (``U``), decoder i's attention boundary -> its pool (``A``, MoE layers)."""
        n = len(decoders)
        for i, d in enumerate(decoders):
            ub, ab = (
                getattr(d, "_fused_ffn_boundary", None),
                getattr(d, "_fused_attn_boundary", None),
            )
            if (
                "U" in positions
                and ub is not None
                and i + 1 < n
                and ("qkv", i + 1) in self.bf16
                and getattr(decoders[i + 1].self_attn.qkv_proj, "bias", None) is None
            ):
                self.b3_map[id(ub)] = ("U", i + 1)
            if "A" in positions and ab is not None and ("pool", i) in self.bf16:
                self.b3_map[id(ab)] = ("A", i)
        if "A" in positions:
            dev = self.stage.device
            self.b3a_q = torch.empty((self.cap, self.k), dtype=torch.uint8, device=dev)
            self.b3a_s = torch.empty(
                (self.mx.sf_bytes(self.cap, self.k),), dtype=torch.uint8, device=dev
            )

    def b3_boundary(
        self,
        op,
        pos: str,
        layer: int,
        stream,
        branch,
        branch_dtype,
        router_dtype,
        pre_norm_dtype,
    ):
        """The boundary op's call with the MXFP8 epilogue (``U``: into the qkv staging; ``A``: into the pool buffer)."""
        from .ops import b3_norm_mx as nm

        rows, k = int(branch.shape[0]), self.k
        common = dict(
            op.jit_kernel_kwargs(branch_dtype),
            want_router=router_dtype is not None,
            pre_norm_bf16=pre_norm_dtype is torch.bfloat16,
        )
        if pos == "U":
            rp = nm.rows_pad(rows)
            qb, sb = rp * k, self.mx.sf_bytes(rp, k)
            base = self.slot_base(layer)
            if self.depth == 1 and not self.agf_pass(
                rows
            ):  # (an AGF pass: no peer reads this staging)
                self.handle.barrier(
                    channel=0
                )  # every peer's previous pulls from this staging are complete
            out = nm.norm_residual_mx(
                branch,
                stream,
                q_out=self.stage[base : base + qb].view(rp, k),
                sf_out=self.stage[base + qb : base + qb + sb],
                pad_quant=True,
                **common,
            )
            self.b3u = (layer, out[0].data_ptr(), rows)
        else:
            q, s = self.b3a_q[:rows], self.b3a_s[: self.mx.sf_bytes(rows, k)]
            out = nm.norm_residual_mx(
                branch, stream, q_out=q, sf_out=s, pad_quant=False, **common
            )
            self.b3a = (layer, out[0].data_ptr(), rows, q.view(self.mx.FP8), s)
        bed.count(NAME, f"b3:{pos}:{_bucket(rows)}")
        if _state["b3_check"] and not torch.cuda.is_current_stream_capturing():
            self._b3_check(
                op,
                pos,
                layer,
                stream,
                branch,
                branch_dtype,
                router_dtype,
                pre_norm_dtype,
                out,
                rows,
            )
        return out

    def _b3_check(
        self,
        op,
        pos,
        layer,
        stream,
        branch,
        branch_dtype,
        router_dtype,
        pre_norm_dtype,
        out,
        rows,
    ) -> None:
        """Test aid (``MUSE_VR_FP8_PREFILL_B3_CHECK=1``, eager calls only; never set on a served configuration): runs the
        unfused path on the same inputs (SGLang's op, then the quantizer as configured) and compares every byte the fused
        call produced: the kept outputs, and the MXFP8 bytes (U: the staged shard incl. its padding rows; A: the pool
        operand). Counted ``b3:check:ok:<pos>`` / ``b3:check:diff:<pos>``; the first difference per position is logged."""
        from .ops import b3_norm_mx as nm

        def same(a, b):
            if a is None or b is None:
                return a is None and b is None
            return a.shape == b.shape and torch.equal(
                a.contiguous().view(torch.uint8), b.contiguous().view(torch.uint8)
            )

        ref = _state["orig_boundary"](
            op, stream, branch, branch_dtype, router_dtype, pre_norm_dtype
        )
        what = [
            n
            for n, a, b in (
                ("output", out[0], ref[0]),
                ("new_stream", out[1], ref[1]),
                ("router", out[2], ref[2]),
            )
            if not same(a, b)
        ]
        k, mx = self.k, self.mx
        if pos == "U":
            rp = nm.rows_pad(rows)
            qb, sb = rp * k, mx.sf_bytes(rp, k)
            xp = (
                ref[0]
                if rp == rows
                else torch.nn.functional.pad(ref[0], (0, 0, 0, rp - rows))
            )
            q, s = mx.quantize(xp.contiguous())
            base = self.slot_base(layer)
            if not torch.equal(
                self.stage[base : base + qb], q.view(torch.uint8).reshape(-1)
            ):
                what.append("staged values")
            if not torch.equal(self.stage[base + qb : base + qb + sb], s[:sb]):
                what.append("staged scales")
        else:
            q, s = mx.quantize(ref[0].contiguous())
            sb = mx.sf_bytes(rows, k)
            if not torch.equal(self.b3a_q[:rows], q.view(torch.uint8)):
                what.append("values")
            if not torch.equal(self.b3a_s[:sb], s[:sb]):
                what.append("scales")
        if what:
            bed.count(NAME, f"b3:check:diff:{pos}")
            if ("check", pos) not in _state["logged"]:
                _state["logged"].add(("check", pos))
                logger.error(
                    "muse_vr_plugin: fp8_prefill B3 check: %s layer %d rows %d differs from the unfused path: %s",
                    pos,
                    layer,
                    rows,
                    ", ".join(what),
                )
        else:
            bed.count(NAME, f"b3:check:ok:{pos}")
            n = bed.counts().get(NAME, {}).get(f"b3:check:ok:{pos}", 0)
            if (
                n & (n - 1) == 0
            ):  # progress keys (the engagement record is written on new keys only)
                bed.count(NAME, f"b3:check:ok:{pos}>={n}")

    def qkv_ag_gemm(self, x: torch.Tensor, layer: int, num_tokens: int) -> torch.Tensor:
        """x: this rank's [rows, K] bf16 shard. Returns [num_tokens, N] like SGLang's all_gather_matmul."""
        import cuda.bindings.driver as cuda

        from . import o_rowshard

        mx = self.mx
        rows, k = x.shape
        rp = -(-rows // mx.SF_TILE_ROWS) * mx.SF_TILE_ROWS
        qb, sb = rp * k, mx.sf_bytes(rp, k)
        wq, ws = self.weight("qkv", layer)
        n = int(wq.shape[0])
        if self.agf_pass(rows):
            return self._qkv_agf(x, layer, num_tokens, rp, qb, sb, wq, ws, n)
        if self.agf:
            bed.count(NAME, "agf:stock:rows>cap")
        main = torch.cuda.current_stream()
        base = self.slot_base(layer)  # 0 unless two-deep staging
        stage = self.stage[base : base + self.half]
        pre, self.b3u = self.b3u, None
        if pre is not None and pre == (layer, x.data_ptr(), rows):
            # the U boundary already wrote this shard's MXFP8 into the staging, after the "previous pulls" barrier
            bed.count(NAME, "b3:U:used")
        else:
            xp = x if rp == rows else torch.nn.functional.pad(x, (0, 0, 0, rp - rows))
            q, s = mx.quantize(xp)
            if self.depth == 1:
                self.handle.barrier(
                    channel=0
                )  # every peer's previous pulls from this staging are complete
            stage[:qb].copy_(q.view(torch.uint8).reshape(-1))
            stage[qb : qb + sb].copy_(s[:sb])
        if self.depth == 2:
            bed.count(NAME, f"ag2deep:slot{layer % 2}")
        self.flags.zero_()
        self.handle.barrier(channel=0)  # every rank's shard is staged
        self.copy_stream.wait_stream(main)
        stream = cuda.CUstream(self.copy_stream.cuda_stream)
        fbase, fbytes = self.flags.data_ptr(), self.flags.element_size()
        from . import o_rowshard

        events = (
            o_rowshard.flag_events_on()
        )  # debug-only (sanitizers): events at the same points, never served
        with torch.cuda.stream(self.copy_stream):
            for step in range(1, self.world):
                p = (self.rank + step) % self.world
                _copy(
                    self.gathered[p].data_ptr(),
                    int(self.handle.buffer_ptrs[p]) + base,
                    qb + sb,
                    self.copy_stream,
                )
                if events:
                    o_rowshard._write_flag(self.copy_stream, fbase + p * fbytes)
                    continue
                (st,) = cuda.cuStreamWriteValue32(
                    stream, cuda.CUdeviceptr(fbase + p * fbytes), 1, 0
                )
                if st != cuda.CUresult.CUDA_SUCCESS:
                    raise RuntimeError(f"fp8_prefill: flag write failed: {st}")
        out = x.new_empty((rows * self.world, n))
        mstream = cuda.CUstream(main.cuda_stream)
        for step in range(self.world):
            p = (self.rank + step) % self.world
            src = stage if step == 0 else self.gathered[p]
            if step and events:
                o_rowshard._wait_flag(main, fbase + p * fbytes)
            elif step:
                (st,) = cuda.cuStreamWaitValue32(
                    mstream,
                    cuda.CUdeviceptr(fbase + p * fbytes),
                    1,
                    cuda.CUstreamWaitValue_flags.CU_STREAM_WAIT_VALUE_EQ,
                )
                if st != cuda.CUresult.CUDA_SUCCESS:
                    raise RuntimeError(f"fp8_prefill: flag wait failed: {st}")
            a = src[:qb].view(rp, k)[:rows].view(mx.FP8)
            mx.gemm(
                a,
                src[qb : qb + sb],
                wq,
                ws,
                self.backend,
                out=out[p * rows : (p + 1) * rows],
            )
        main.wait_stream(self.copy_stream)
        return out[:num_tokens]

    def _qkv_agf(
        self, x, layer: int, num_tokens: int, rp: int, qb: int, sb: int, wq, ws, n: int
    ) -> torch.Tensor:
        """MUSE_VR_FP8_PREFILL_AGF: push this rank's MXFP8 shard into every rank's gathered slot (layer parity), one bounded
        barrier, then ONE GEMM over the world x rp gathered rows; the padding rows of each rank's block are dropped."""
        from .ops import a2a_pull, mc_push

        mx, k, world = self.mx, self.k, self.world
        rows = int(x.shape[0])
        pre, self.b3u = self.b3u, None
        if pre is not None and pre == (layer, x.data_ptr(), rows):
            # the U boundary already wrote this shard's MXFP8 (padded to rp rows) into the local staging
            bed.count(NAME, "b3:U:used")
            bed.count(NAME, "agf:push:b3")
            base = self.slot_base(
                layer
            )  # 0 unless the staging kept two slots for declined passes (row cap + L2)
            src_q, src_s = (
                self.stage[base : base + qb],
                self.stage[base + qb : base + qb + sb],
            )
        else:
            xp = x if rp == rows else torch.nn.functional.pad(x, (0, 0, 0, rp - rows))
            q, s = mx.quantize(xp)
            bed.count(NAME, "agf:push:quant")
            src_q, src_s = q.view(torch.uint8).reshape(-1), s[:sb]
        slot = layer % 2
        vbase = slot * self.agf_slot
        sbase = vbase + self.agf_vbytes
        if self.agf_dense:
            return self._qkv_agf_dense(
                x, src_q, src_s, slot, vbase, sbase, num_tokens, wq, ws
            )
        mc_push.push(
            src_q,
            src_s,
            self.agf_mc + vbase + self.rank * qb,
            self.agf_mc + sbase + self.rank * sb,
            qb,
            sb,
        )
        bed.count(NAME, f"agf:slot{slot}")
        a2a_pull.barrier(
            x, self.agf_pads, self.rank, self.agf_polls
        )  # every rank's shard is in every gathered slot
        a = self.agf_buf[vbase : vbase + world * qb].view(world * rp, k).view(mx.FP8)
        sa = self.agf_buf[sbase : sbase + world * sb]
        if rp == rows:
            return mx.gemm(a, sa, wq, ws, self.backend)[:num_tokens]
        full = mx.gemm(a, sa, wq, ws, self.backend)
        return full.view(world, rp, n)[:, :rows].reshape(world * rows, n)[:num_tokens]

    def _qkv_agf_dense(
        self,
        x,
        src_q,
        src_s,
        slot: int,
        vbase: int,
        sbase: int,
        num_tokens: int,
        wq,
        ws,
    ) -> torch.Tensor:
        """MUSE_VR_FP8_PREFILL_AGF_DENSE: push this rank's rows densely (rank x rows) with their scales scattered into the
        gathered operand's swizzled layout, one bounded barrier, then ONE GEMM over the world x rows rows (the final output)."""
        from .ops import a2a_pull, mc_push_dense

        mx, k, world = self.mx, self.k, self.world
        rows = int(x.shape[0])
        m = world * rows
        if self.agf_uc:
            from .ops import uc_push_dense

            uc_push_dense.push(
                src_q,
                src_s,
                [ptr + vbase for ptr in self.agf_ptrs],
                [ptr + sbase for ptr in self.agf_ptrs],
                rows,
                self.rank,
                world,
                k,
                self.agf_vbytes,
                self.agf_slot - self.agf_vbytes,
                fence=self.agf_uc_fence,
            )
            bed.count(NAME, "agf:uc")
        else:
            mc_push_dense.push(
                src_q,
                src_s,
                self.agf_mc + vbase,
                self.agf_mc + sbase,
                rows,
                self.rank,
                world,
                k,
                self.agf_vbytes,
                self.agf_slot - self.agf_vbytes,
            )
        bed.count(NAME, f"agf:slot{slot}")
        bed.count(NAME, "agf:dense")
        a2a_pull.barrier(
            x, self.agf_pads, self.rank, self.agf_polls
        )  # every rank's rows are in every gathered slot
        a = self.agf_buf[vbase : vbase + m * k].view(m, k).view(mx.FP8)
        sa = self.agf_buf[sbase : sbase + mx.sf_bytes(m, k)]
        return mx.gemm(a, sa, wq, ws, self.backend)[:num_tokens]


def _wrap_forward(orig):
    @functools.wraps(orig)
    def forward(
        self, input_ids, positions, forward_batch, input_embeds=None, *args, **kwargs
    ):
        from sglang.srt.model_executor.forward_batch_info import ForwardMode

        n = (
            int(input_ids.shape[0])
            if input_ids is not None
            else (int(input_embeds.shape[0]) if input_embeds is not None else 0)
        )
        on = (
            _state["ctx"] is not None
            and forward_batch.forward_mode == ForwardMode.EXTEND
            and n > _state["min_tokens"]
        )
        _state["pass"] = on
        if _state["ctx"] is not None:
            _state[
                "ctx"
            ].pre.clear()  # nothing quantized ahead outlives its pass (its event belongs to that pass)
            _state["ctx"].b3u = _state["ctx"].b3a = (
                None  # nor anything a fused producer wrote
            )
        if on:
            bed.count(NAME, f"pass:{_bucket(n)}")
        try:
            return orig(
                self, input_ids, positions, forward_batch, input_embeds, *args, **kwargs
            )
        finally:
            _state["pass"] = False

    forward.__wrapped_by_fp8_prefill__ = True
    return forward


def _wrap_g(orig):
    @functools.wraps(orig)
    def column_parallel_g_matmul(linear, input_parallel, bias):
        if not _state["pass"]:  # not-engaged fast path: one flag read before any lookup
            return orig(linear, input_parallel, bias)
        ctx = _state["ctx"]
        layer = ctx.qkv.get(id(linear)) if ctx is not None else None
        if layer is None or not pass_fp8("qkv"):
            return orig(linear, input_parallel, bias)
        reason = None
        if bias is not None:
            reason = "bias"
        elif (
            input_parallel.dim() != 2
            or input_parallel.dtype != torch.bfloat16
            or input_parallel.shape[1] != ctx.k
        ):
            reason = "input"
        elif -(-input_parallel.shape[0] // 128) * 128 > ctx.cap:
            reason = "rows>cap"
        elif torch.compiler.is_compiling():
            reason = "compiling"
        if reason is not None:
            bed.count(NAME, "qkv:stock:" + reason)
            return orig(linear, input_parallel, bias)
        import sglang.srt.layers.layernorm_sp as layernorm_sp

        from . import o_rowshard

        octx = o_rowshard._state.get("ctx")
        if octx is not None:
            octx.join()  # the o_proj weight copy started by the previous layer (o_rowshard's join point)
        bed.count(NAME, f"qkv:{_bucket(input_parallel.shape[0])}")
        return ctx.qkv_ag_gemm(
            input_parallel.contiguous(), layer, layernorm_sp.sp_num_tokens()
        )

    column_parallel_g_matmul.__wrapped_by_fp8_prefill__ = True
    return column_parallel_g_matmul


def _wrap_gbar(orig):
    @functools.wraps(orig)
    def row_parallel_gbar_matmul(linear, input_, bias):
        ctx = _state["ctx"]
        if ctx is not None and _state["pass"] and ctx.wq_prefetch:
            layer = ctx.o_layer.get(id(linear))
            if layer is not None:
                ctx.prefetch_weights(layer, ctx.n_layers)
        return orig(linear, input_, bias)

    row_parallel_gbar_matmul.__wrapped_by_fp8_prefill__ = True
    return row_parallel_gbar_matmul


def _wrap_dispatch(orig):
    @functools.wraps(orig)
    def _bf16_gemm_dispatch_impl(x, weight, bias, addend=None):
        ctx = _state["ctx"]
        if ctx is None or not _state["pass"]:
            return orig(x, weight, bias, addend)
        e = ctx.reg.get((weight.data_ptr(), tuple(weight.shape)))
        if e is None or e[0] not in _state["groups"]:
            return orig(x, weight, bias, addend)
        if bias is not None or addend is not None or x.dtype != torch.bfloat16:
            bed.count(NAME, f"{e[0]}:stock:operands")
            return orig(x, weight, bias, addend)
        x2 = x.reshape(-1, x.shape[-1])
        bed.count(NAME, f"{e[0]}:{_bucket(x2.shape[0])}")
        wq, ws = ctx.weight(*e)
        pre = ctx.b3a if e[0] == "pool" else None
        if pre is not None:
            ctx.b3a = None
        if (
            pre is not None
            and pre[:3] == (e[1], x2.data_ptr(), int(x2.shape[0]))
            and x2.is_contiguous()
        ):
            q, s = pre[3], pre[4]  # quantized by the A boundary that produced x
            bed.count(NAME, "b3:A:used")
        else:
            q, s = ctx.mx.quantize(x2.contiguous())
        return ctx.mx.gemm(q, s, wq, ws, ctx.backend).view(*x.shape[:-1], wq.shape[0])

    _bf16_gemm_dispatch_impl.__wrapped_by_fp8_prefill__ = True
    return _bf16_gemm_dispatch_impl


def _b3_reason(
    ctx, pos: str, op, stream, branch, branch_dtype, router_dtype, pre_norm_dtype
):
    """Why this boundary call keeps SGLang's op (None: it takes the fused producer)."""
    from .ops import b3_norm_mx as nm

    if not pass_fp8("qkv" if pos == "U" else "pool"):
        return "group"
    if torch.compiler.is_compiling():
        return "compiling"
    if not (
        isinstance(branch, torch.Tensor)
        and isinstance(stream, torch.Tensor)
        and branch.dim() == 2
        and branch.dtype == torch.bfloat16
        and branch_dtype == torch.bfloat16
        and stream.dtype == torch.float32
        and stream.shape == branch.shape
        and branch.is_contiguous()
        and stream.is_contiguous()
        and branch.is_cuda
    ):
        return "operands"
    if int(branch.shape[1]) != ctx.k or not op._triton_supported(
        stream, branch, branch_dtype, router_dtype, pre_norm_dtype
    ):
        return "op form"
    m = int(branch.shape[0])
    if nm.launch_config(int(branch.shape[1]), m) is None:
        return "rows"
    if nm.rows_pad(m) > ctx.cap:
        return "rows>cap"
    return None


def _wrap_boundary(orig):
    @functools.wraps(orig)
    def forward_jit(
        self,
        stream,
        branch,
        branch_dtype=torch.bfloat16,
        router_dtype=None,
        pre_norm_dtype=None,
    ):
        if not _state["pass"]:  # not-engaged fast path: one flag read before any lookup
            return orig(
                self, stream, branch, branch_dtype, router_dtype, pre_norm_dtype
            )
        ctx = _state["ctx"]
        e = ctx.b3_map.get(id(self)) if ctx is not None else None
        if e is None:
            return orig(
                self, stream, branch, branch_dtype, router_dtype, pre_norm_dtype
            )
        pos, layer = e
        reason = _b3_reason(
            ctx, pos, self, stream, branch, branch_dtype, router_dtype, pre_norm_dtype
        )
        if reason is not None:
            bed.count(NAME, f"b3:{pos}:stock:{reason}")
            return orig(
                self, stream, branch, branch_dtype, router_dtype, pre_norm_dtype
            )
        return ctx.b3_boundary(
            self, pos, layer, stream, branch, branch_dtype, router_dtype, pre_norm_dtype
        )

    forward_jit.__wrapped_by_fp8_prefill__ = True
    return forward_jit


def _target_layers(model):
    out = []
    for mod in model.modules():
        if type(mod).__name__ == "MuseSpark1xDecoderLayer":
            out.append(
                (mod.self_attn, mod.mlp if getattr(mod, "is_moe_layer", True) else None)
            )
    return out


def _wrap_load_model(fn):
    @functools.wraps(fn)
    def load_model(self, *args, **kwargs):
        out = fn(self, *args, **kwargs)
        model = getattr(self, "model", None)
        name = type(model).__name__ if model is not None else ""
        if (
            getattr(self, "is_draft_worker", False)
            or not name.startswith("MuseSpark")
            or "Draft" in name
        ):
            return out
        if _state["ctx"] is not None:
            return out
        try:
            from sglang.srt.runtime_context import get_parallel

            from . import o_rowshard

            tp = get_parallel().tp_group
            world, rank = int(tp.world_size), int(tp.rank_in_group)
            layers = _target_layers(model)
            if not layers:
                raise ValueError("no Muse Spark decoder layers")
            if "o" in _state["groups"] and o_rowshard._state.get("ctx") is None:
                raise ValueError("group 'o' needs o_rowshard engaged")
            cap = bed.env_int("MUSE_VR_FP8_PREFILL_CAP_ROWS", 16384)
            pinned = 0
            tac = os.environ.get("MUSE_VR_FP8_PREFILL_TACTICS", "").strip()
            runner = os.environ.get("MUSE_VR_FP8_PREFILL_PIN_RUNNER", "").strip()
            if tac and runner and _state["backend"] == "cute-dsl":
                from . import mxfp8_ops

                mxfp8_ops.set_runner_factory(runner)
                pinned = mxfp8_ops.load_pinned(tac)
            ctx = Fp8Ctx(
                layers,
                tp.device_group,
                rank,
                world,
                cap,
                _state["weights"],
                _state["backend"],
            )
            shapes = [tuple(w.shape) for w in ctx.bf16.values()]
            octx = o_rowshard._state.get("ctx")
            if octx is not None and octx.fp8 is not None:
                shapes.append((octx.n, octx.k))
            warm = ctx.warmup(shapes, cap)
            if _state["b3"]:
                from .ops import b3_norm_mx

                decoders = [
                    m
                    for m in model.modules()
                    if type(m).__name__ == "MuseSpark1xDecoderLayer"
                ]
                ctx.set_b3(decoders, _state["b3"])
                pdl = b3_norm_mx._use_pdl()
                for rows in (
                    128,
                    cap,
                ):  # build the kernels (both launch shapes) before any CUDA-graph capture
                    cfg = b3_norm_mx.launch_config(ctx.k, rows)
                    if cfg is not None:
                        b3_norm_mx.module(ctx.k, cfg[0], cfg[1], pdl)
            if (
                ctx.agf
            ):  # build the push and barrier kernels before any CUDA-graph capture
                from .ops import a2a_pull, mc_push

                mc_push.module(mc_push.config()[0])
                a2a_pull.module(a2a_pull.config()[0])
                if ctx.agf_dense:
                    from .ops import mc_push_dense

                    mc_push_dense.module(mc_push.config()[0])
                if ctx.agf_uc:
                    from .ops import uc_push_dense

                    uc_push_dense.module(mc_push.config()[0], ctx.agf_uc_fence)
            if ctx.wq_prefetch and _state.get("orig_gbar") is None:
                # outermost on the o_proj SP entry (o_rowshard installed its own wrapper during the load above)
                import sglang.srt.layers.layernorm_sp as layernorm_sp

                _state["orig_gbar"] = layernorm_sp.row_parallel_gbar_matmul
                layernorm_sp.row_parallel_gbar_matmul = _wrap_gbar(_state["orig_gbar"])
            _state["ctx"] = ctx
            torch.cuda.empty_cache()
            res_bytes = sum(q.numel() + s.numel() for q, s in ctx.res.values())
            bed.report(
                NAME,
                True,
                groups=",".join(_state["groups"]),
                min_tokens=_state["min_tokens"],
                backend=_state["backend"],
                weights=_state["weights"],
                resident_bytes=res_bytes,
                resident_layers=int(ctx.resident_layers),
                layers=len(layers),
                tp=world,
                cap_rows=cap,
                warmup_calls=warm,
                pinned_configs=pinned,
                pinned_lookup=ctx.mx.pinned_lookup(),
                stage_depth=ctx.depth,
                agf=bool(ctx.agf),
                agf_dense=bool(ctx.agf_dense),
                agf_why=ctx.agf_why,
                agf_uc=bool(ctx.agf_uc),
                agf_uc_fence=bool(ctx.agf_uc and ctx.agf_uc_fence),
                agf_slot_bytes=int(getattr(ctx, "agf_slot", 0)) if ctx.agf else 0,
                agf_bytes=2 * int(getattr(ctx, "agf_slot", 0)) if ctx.agf else 0,
                agf_cap=int(ctx.agf_cap),
                agf_max_polls=int(getattr(ctx, "agf_polls", 0)) if ctx.agf else 0,
                fq=ctx.mx.fq_mode(),
                fq_group=bool(ctx.fq_group),
                fq_group_slots=",".join(sorted(ctx.gslots)),
                b3=",".join(_state["b3"]),
                b3_check=bool(_state["b3_check"]),
                b3_u_sites=sum(1 for p, _li in ctx.b3_map.values() if p == "U"),
                b3_a_sites=sum(1 for p, _li in ctx.b3_map.values() if p == "A"),
            )
        except Exception as e:  # noqa: BLE001 - a half-installed context must not serve: record, then re-raise
            bed.report(NAME, False, reason=f"install failed: {type(e).__name__}: {e}")
            raise
        return out

    load_model.__wrapped_by_fp8_prefill__ = True
    return load_model


def register() -> bool:
    if not bed.enabled(NAME):
        return False
    groups = tuple(
        g.strip()
        for g in os.environ.get(
            "MUSE_VR_FP8_PREFILL_GROUPS", ",".join(GROUPS_ALL)
        ).split(",")
        if g.strip()
    )
    weights = os.environ.get("MUSE_VR_FP8_PREFILL_WEIGHTS", "jit").strip()
    backend = os.environ.get("MUSE_VR_FP8_PREFILL_BACKEND", "cute-dsl").strip()
    min_tokens = bed.env_int("MUSE_VR_FP8_PREFILL_MIN_TOKENS", DEFAULT_MIN_TOKENS)
    if (
        not groups
        or any(g not in GROUPS_ALL for g in groups)
        or weights not in ("jit", "resident")
        or min_tokens < 1
    ):
        bed.report(
            NAME,
            False,
            reason=f"bad config groups={groups} weights={weights} min_tokens={min_tokens}",
        )
        return False
    if (
        os.environ.get("MUSE_VR_FP8_PREFILL_RESIDENT_LAYERS", "").strip()
        and weights != "resident"
    ):
        bed.report(
            NAME,
            False,
            reason="bad config: MUSE_VR_FP8_PREFILL_RESIDENT_LAYERS needs MUSE_VR_FP8_PREFILL_WEIGHTS=resident",
        )
        return False
    if bed.enabled("smx"):
        # smx's mixed passes replay the prefill graphs with the running batch's verify rows inside: those rows would take
        # this component's MXFP8 projections, and decode / verify must stay BF16. Refused until a row-split form exists.
        bed.report(
            NAME,
            False,
            reason="refused next to smx (mixed passes carry verify rows through the prefill graphs)",
        )
        return False
    b3 = tuple(
        p.strip()
        for p in os.environ.get("MUSE_VR_FP8_PREFILL_B3", "").split(",")
        if p.strip()
    )
    if (
        any(p not in B3_POSITIONS for p in b3)
        or len(set(b3)) != len(b3)
        or ("U" in b3 and "qkv" not in groups)
        or ("A" in b3 and "pool" not in groups)
    ):
        bed.report(
            NAME,
            False,
            reason=f"bad config b3={b3} (positions U / A, each needing its group: U qkv, A pool)",
        )
        return False
    agf = os.environ.get("MUSE_VR_FP8_PREFILL_AGF", "").strip() == "1"
    agf_dense = os.environ.get("MUSE_VR_FP8_PREFILL_AGF_DENSE", "").strip() == "1"
    if agf_dense and not agf:
        bed.report(
            NAME,
            False,
            reason="bad config: MUSE_VR_FP8_PREFILL_AGF_DENSE=1 needs MUSE_VR_FP8_PREFILL_AGF=1",
        )
        return False
    agf_uc = os.environ.get("MUSE_VR_FP8_PREFILL_AGF_UC", "").strip() == "1"
    agf_uc_nofence = (
        os.environ.get("MUSE_VR_FP8_PREFILL_AGF_UC_NOFENCE_UNSAFE", "").strip() == "1"
    )
    if (agf_uc and not agf_dense) or (agf_uc_nofence and not agf_uc):
        bed.report(
            NAME,
            False,
            reason="bad config: MUSE_VR_FP8_PREFILL_AGF_UC=1 needs MUSE_VR_FP8_PREFILL_AGF_DENSE=1, and "
            "MUSE_VR_FP8_PREFILL_AGF_UC_NOFENCE_UNSAFE=1 needs MUSE_VR_FP8_PREFILL_AGF_UC=1",
        )
        return False
    for rel, sha in list(SGLANG_PINS.items()) + (list(B3_PINS.items()) if b3 else []):
        if not bed.pinned(rel, sha):
            bed.report(NAME, False, reason=f"pin mismatch: {rel}")
            return False
    try:
        import sglang.srt.layers.layernorm_sp as layernorm_sp
        import sglang.srt.layers.quantization.unquant as U
        import sglang.srt.model_executor.model_runner as mr
        import sglang.srt.models.muse_spark as muse_spark

        with _lock:
            _state.update(
                groups=groups,
                weights=weights,
                backend=backend,
                min_tokens=min_tokens,
                b3=b3,
                b3_check=bool(b3)
                and os.environ.get("MUSE_VR_FP8_PREFILL_B3_CHECK", "").strip() == "1",
                ag2deep=os.environ.get("MUSE_VR_FP8_PREFILL_AG2DEEP", "").strip()
                == "1",
                agf=agf,
                agf_dense=agf_dense,
                agf_uc=agf_uc,
                agf_uc_fence=not agf_uc_nofence,
            )
            if b3 and _state["orig_boundary"] is None:
                import sglang.kernels.ops.layernorm.muse_spark_norm_residual as nr

                bcls = nr.FusedPostNormResidualPreNorm
                if not getattr(bcls.forward_jit, "__wrapped_by_fp8_prefill__", False):
                    _state["orig_boundary"] = bcls.forward_jit
                    bcls.forward_jit = _wrap_boundary(bcls.forward_jit)
            if _state["orig_g"] is None:
                _state["orig_g"] = layernorm_sp.column_parallel_g_matmul
                layernorm_sp.column_parallel_g_matmul = _wrap_g(_state["orig_g"])
                _state["orig_dispatch"] = U._bf16_gemm_dispatch_impl
                U._bf16_gemm_dispatch_impl = _wrap_dispatch(_state["orig_dispatch"])
                cls = muse_spark.MuseSpark1xModel
                if not getattr(cls.forward, "__wrapped_by_fp8_prefill__", False):
                    _state["orig_forward"] = cls.forward
                    cls.forward = _wrap_forward(cls.forward)
            runner = mr.ModelRunner
            if not getattr(runner.load_model, "__wrapped_by_fp8_prefill__", False):
                runner.load_model = _wrap_load_model(runner.load_model)
    except Exception as e:  # noqa: BLE001
        bed.report(NAME, False, reason=f"install failed: {type(e).__name__}: {e}")
        return False
    _state["engaged"] = True
    return True
