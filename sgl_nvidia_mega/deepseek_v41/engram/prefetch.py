# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Engram prefetch (slot ``engram``): every Engram layer's lookup and WKV GEMM off
the critical path, with no collective on the side stream.

An Engram layer's table lookup, MXFP8 quantize and WKV GEMM read only the token
hashes, yet SGLang runs them on that layer's critical path. For a routed step of at
most ``MAX_ROWS`` rows:

- right after the hashes, a side stream looks up this rank's rows of every Engram
  layer's table (rank-local; the per-rank host table zeroes the rows it does not own);
- at ``reduce_at`` (the layer before the first Engram layer, before its MoE) the main
  stream reduces all layers' rows in one all-reduce, SGLang's ``_reduce_owned_rows``;
  the side stream then multiplies each layer's rows by this rank's quarter of the WKV
  output columns (views of the weight and of its 128-row scale tiles) into zeros;
- at ``gather_at`` (the first Engram layer) the main stream sums the zero-padded
  quarters in one all-reduce on SGLang's TP group.

Every cross-GPU wait is on the main stream, in the same order on every rank, as in
SGLang's inline path. Both all-reduces sum one owner's value with zeros and the WKV
runs the same CuTe DSL GEMM, so the rows equal SGLang's inline ones bit for bit;
build checks that at every width it serves, with the new buffers poisoned first.

With slot ``engram_publish`` (exchange.py; muLLM's Engram publish / gate collect) no
zero-padded rows and no second all-reduce: right after ``reduce`` the side stream runs
the first Engram layer's WKV columns and publishes them into every rank's mailbox; a
later Engram layer's columns follow when the main stream reaches the layer
``LATE_LEAD`` before it (muLLM's late-layer schedule), off the first layer's path and away
from layer 0's MoE; at each Engram layer the main stream collects the layer's rows
from its own mailbox (the only waiting kernel). The rows are SGLang's whole WKV GEMM
bit for bit (no sum with zeros turns a -0.0 into +0.0); build checks every width.
"""

from __future__ import annotations

import torch

from sgl_nvidia_mega.deepseek_v41 import plan

# The SGLang names this module reads, by file under the sglang package (``f(p)``: a
# parameter, ``"x"``: a string literal); test/unit/test_contract.py pins them.
SGLANG_READS = {
    "srt/layers/engram.py": (
        "engram_gate",
        "embed",
        "wkv",
        "q_weight",
        "k_weight",
        "eps",
        "clamp_value",
        "layer_hash_index",
        "row_start",
        "_owned_rows",
        "_reduce_owned_rows",
        "replication|layout",
        "rows",
        "dim",
    ),
    "srt/kernel_plugins/nvidia/binding.py|srt/megakernels/binding.py": (
        "engram_host_table",
    ),
    "srt/models/deepseek_v4.py": (
        "model",
        "layers",
        "start_layer",
        "end_layer",
        "engram",
    ),
    "srt/layers/quantization/fp8.py": (
        "Fp8LinearMethod",
        "use_mxfp8",
        "block_fp8_as_mxfp8",
        "block_fp8_mxfp8_ready",
        "mxfp8_dense_backend",
        "weight_scale_inv_swizzled",
    ),
    "srt/layers/quantization/fp8_utils.py": (
        "flashinfer_mxfp8_blockscaled_linear",
        "flashinfer_mxfp8_blockscaled_linear(backend)",
        "flashinfer_mxfp8_blockscaled_linear(pin_tactic)",
        '"cute-dsl"',
        "is_flashinfer_cutlass",
        "is_flashinfer_cutedsl",
    ),
    "srt/distributed/parallel_state.py": (
        "all_reduce",
        "world_size",
        "rank_in_group",
        "cpu_group",
    ),
}

MAX_ROWS = plan.ENGRAM_PREFETCH_MAX_ROWS
# slot engram_publish: a later Engram layer's WKV columns start at the start of the layer
# this many before it (muLLM's late-layer lead: its projection waits for layer L - 4)
LATE_LEAD = 4
# The WKV slice of each rank is whole 128-row scale tiles
SCALE_TILE_ROWS = 128


def project_points(layer_ids, first_layer: int) -> dict:
    """Slot engram_publish: each Engram layer's start point for its WKV columns, the
    main-stream layer at whose start they start, or None for right after ``reduce``
    (the first layer, and any layer within ``LATE_LEAD`` of it)."""
    return {
        i: (i - LATE_LEAD if i - LATE_LEAD > first_layer else None)
        for i in sorted(layer_ids)
    }


def _per_rank_host_table(table) -> bool:
    """Accept current replication or legacy layout, rejecting unknown tables."""
    if table is None:
        return False
    try:
        replication = table.replication
    except AttributeError:
        try:
            return table.layout == "per_rank"
        except AttributeError:
            return False
    return replication == "none"


def reason(engram, tp_size: int, table) -> str:
    """Why this Engram layer cannot prefetch, or "": the per-rank host ``table`` (the
    binding's; each rank looks up only its own rows) and an MXFP8 WKV with swizzled
    scales."""
    from sglang.srt.layers.quantization.fp8 import Fp8LinearMethod

    if not _per_rank_host_table(table):
        return (
            "needs the per-rank host Engram table (SGLANG_ENABLE_DSV41_ENGRAM_HOST_TABLE=1"
            ", SGLANG_DSV41_ENGRAM_HOST_TABLE_LAYOUT=per_rank)"
        )
    wkv = engram.wkv
    method = wkv.quant_method
    if not isinstance(method, Fp8LinearMethod):
        return f"the WKV is on {type(method).__name__}"
    backend = method.mxfp8_dense_backend
    mxfp8 = method.use_mxfp8 or (
        method.block_fp8_as_mxfp8 and getattr(wkv, "block_fp8_mxfp8_ready", False)
    )
    if not (
        mxfp8
        and backend is not None
        and (backend.is_flashinfer_cutlass() or backend.is_flashinfer_cutedsl())
    ):
        return "the WKV is not a FlashInfer MXFP8 GEMM"
    if wkv.weight.size(0) % (tp_size * SCALE_TILE_ROWS):
        return f"{wkv.weight.size(0)} WKV columns do not split into scale tiles"
    return ""


class _Layer:
    """One Engram layer's WKV quarter: this rank's output columns."""

    def __init__(self, engram, rank: int, tp_size: int):
        wkv = engram.wkv
        # detached: CuTe DSL's DLPack import refuses tensors that require grad
        weight = wkv.weight.detach()
        n = weight.size(0)
        cols = n // tp_size
        start = rank * cols
        tiles = wkv.weight_scale_inv_swizzled.reshape(n // SCALE_TILE_ROWS, -1)
        self.engram = engram
        self.weight = weight[start : start + cols]
        self.scale = tiles[
            start // SCALE_TILE_ROWS : (start + cols) // SCALE_TILE_ROWS
        ].reshape(-1)
        self.cols = slice(start, start + cols)
        self.n = n

    def quarter(self, emb: torch.Tensor) -> torch.Tensor:
        """This rank's columns of SGLang's ``wkv(emb)`` [rows, n / tp]."""
        from sglang.srt.layers.quantization.fp8_utils import (
            flashinfer_mxfp8_blockscaled_linear,
        )

        # CUTLASS's MXFP8 GEMM keeps one process-wide workspace; CuTe DSL keeps none
        return flashinfer_mxfp8_blockscaled_linear(
            input=emb.flatten(-2),
            weight=self.weight,
            weight_scale=self.scale,
            backend="cute-dsl",
            pin_tactic=True,
        )


class EngramPrefetch:
    """The side stream, its persistent buffers and the two main-stream points."""

    exchange = None  # slot engram_publish: the mailbox exchange

    def __init__(self, engrams: dict, group, first_layer: int, deadline_s: float = 0.0):
        """``deadline_s`` > 0: slot engram_publish, the exchange with that wait deadline."""
        rank, tp_size = group.rank_in_group, group.world_size
        self.group = group
        self.stream = torch.cuda.Stream()
        self.layers = {i: _Layer(e, rank, tp_size) for i, e in sorted(engrams.items())}
        self.index = {i: j for j, i in enumerate(self.layers)}
        self.reduce_at, self.gather_at = first_layer - 1, first_layer
        first = next(iter(self.layers.values()))
        self.dim = first.engram.embed.dim
        self.hash_cols = first.engram.wkv.weight.size(1) // self.dim
        device = first.weight.device
        n = len(self.layers)
        # persistent and flat: every width's views start at one address
        self.looked = torch.empty(
            n * MAX_ROWS * self.hash_cols * self.dim,
            dtype=torch.bfloat16,
            device=device,
        )
        self.parts = torch.empty(
            n * MAX_ROWS * first.n, dtype=torch.bfloat16, device=device
        )
        self.looked_up = torch.cuda.Event()
        self.reduced = torch.cuda.Event()
        self.wkv_done = torch.cuda.Event()
        self.rows = 0
        self.kv = None  # this step's rows [layers, rows, n], from ``gather``
        self._emb = None  # ``reduce``'s all-reduced rows, read by the side stream
        if deadline_s > 0:
            from sgl_nvidia_mega.deepseek_v41.engram.exchange import EngramExchange

            self.exchange = EngramExchange(
                group,
                list(self.layers),
                n=first.n,
                max_rows=MAX_ROWS,
                deadline_s=deadline_s,
                device=device,
            )
            # each layer's collected rows, persistent: [layers, MAX_ROWS, n]
            self.rows_out = torch.empty(
                n * MAX_ROWS * first.n, dtype=torch.bfloat16, device=device
            )
            self.published = {i: torch.cuda.Event() for i in self.layers}
            # the main-stream layer at whose start a layer's columns start (None: reduce)
            self.project_at = project_points(self.layers, first_layer)
            self.last_layer = max(self.layers)
            self._columns = {}  # this step's published columns, held until the join

    def _rows_of(self, layer_id: int, rows: int) -> torch.Tensor:
        """Slot engram_publish: layer ``layer_id``'s collected rows [rows, n]."""
        width = next(iter(self.layers.values())).n
        start = self.index[layer_id] * MAX_ROWS * width
        return self.rows_out[start : start + rows * width].view(rows, width)

    def _looked(self, rows: int) -> torch.Tensor:
        n = len(self.layers)
        return self.looked[: n * rows * self.hash_cols * self.dim].view(
            n, rows, self.hash_cols, self.dim
        )

    def _parts(self, rows: int) -> torch.Tensor:
        n, width = len(self.layers), next(iter(self.layers.values())).n
        return self.parts[: n * rows * width].view(n, rows, width)

    def issue(self, hash_ids: torch.Tensor) -> None:
        """Every Engram layer's own rows for ``hash_ids`` [rows, layers, hash columns],
        on the side stream (rank-local)."""
        if not 0 < hash_ids.size(0) <= MAX_ROWS:
            raise ValueError(f"the Engram prefetch takes 1-{MAX_ROWS} rows")
        self.rows = hash_ids.size(0)
        looked = self._looked(self.rows)
        self.stream.wait_stream(torch.cuda.current_stream())
        hash_ids.record_stream(self.stream)
        with torch.cuda.stream(self.stream):
            for j, layer in enumerate(self.layers.values()):
                e = layer.engram
                looked[j].copy_(e.embed._owned_rows(hash_ids[:, e.layer_hash_index]))
            self.looked_up.record(self.stream)

    def reduce(self) -> None:
        """Main stream at ``reduce_at``: SGLang's reduction of every layer's rows in one
        all-reduce; then the WKV quarters on the side stream."""
        main = torch.cuda.current_stream()
        main.wait_event(self.looked_up)
        first = next(iter(self.layers.values())).engram
        emb = first.embed._reduce_owned_rows(self._looked(self.rows))
        # An out-of-place all-reduce returns a DLPack tensor, whose record_stream the
        # caching allocator ignores: hold it until ``gather`` joins the side stream.
        emb.record_stream(self.stream)
        self._emb = emb
        self.reduced.record(main)
        if self.exchange is not None:
            with torch.cuda.stream(self.stream):
                self.stream.wait_event(self.reduced)
                for i, at in self.project_at.items():
                    if at is None:
                        self._project(i)
            return
        parts = self._parts(self.rows)
        with torch.cuda.stream(self.stream):
            self.stream.wait_event(self.reduced)
            parts.zero_()
            for j, layer in enumerate(self.layers.values()):
                parts[j, :, layer.cols] = layer.quarter(emb[j])
            self.wkv_done.record(self.stream)

    def _project(self, layer_id: int) -> None:
        """Slot engram_publish, on the side stream: this rank's WKV columns of
        ``layer_id`` into every rank's mailbox."""
        layer = self.layers[layer_id]
        columns = layer.quarter(self._emb[self.index[layer_id]]).contiguous()
        self._columns[layer_id] = columns  # held until the main stream joins
        self.exchange.publish(layer_id, columns)
        self.published[layer_id].record(self.stream)

    def at_layer(self, i: int) -> None:
        """Slot engram_publish, main stream at the start of layer ``i``: a later Engram
        layer's columns start on the side stream; at an Engram layer its rows are
        collected (the last one also joins the side stream)."""
        main = torch.cuda.current_stream()
        for layer_id, at in self.project_at.items():
            if at == i:
                self.stream.wait_stream(main)
                with torch.cuda.stream(self.stream):
                    self._project(layer_id)
        if i in self.layers:
            main.wait_event(self.published[i])
            self.exchange.collect(i, self._rows_of(i, self.rows))
            if i == self.last_layer:
                # every side-stream read of these is now ordered before main
                self._emb = None
                self._columns.clear()

    def gather(self) -> None:
        """Main stream at ``gather_at``: the zero-padded quarters summed into every
        layer's rows (this also joins the side stream)."""
        torch.cuda.current_stream().wait_event(self.wkv_done)
        self._emb = None  # the side stream's reads of it are now ordered before main
        self.kv = self.group.all_reduce(self._parts(self.rows))

    def gate(self, layer_id: int, x: torch.Tensor) -> torch.Tensor:
        """``Engram.forward(x, ...)`` of layer ``layer_id`` from its gathered rows."""
        from sglang.srt.layers.engram import engram_gate

        e = self.layers[layer_id].engram
        if self.exchange is not None:
            kv = self._rows_of(layer_id, self.rows)[: x.size(0)]
        else:
            kv = self.kv[self.index[layer_id]][: x.size(0)]
        return engram_gate(x, kv, e.q_weight, e.k_weight, e.eps, e.clamp_value)

    def poison(self, byte: int) -> None:
        """Fill the persistent buffers with ``byte`` (build's check)."""
        self.looked.view(torch.uint8).fill_(byte)
        self.parts.view(torch.uint8).fill_(byte)
        if self.exchange is not None:
            self.rows_out.view(torch.uint8).fill_(byte)

    def run(self, hash_ids: torch.Tensor) -> dict:
        """A whole step's prefetch in layer order (build's check, tests): every Engram
        layer's rows {layer: [rows, n]}."""
        self.issue(hash_ids)
        self.reduce()
        if self.exchange is None:
            self.gather()
            return {i: self.kv[j] for i, j in self.index.items()}
        for i in range(self.gather_at, self.last_layer + 1):
            self.at_layer(i)
        return {i: self._rows_of(i, self.rows) for i in self.layers}


def _any_error(group, error: str) -> str:
    """Every rank's error, joined ("" when none failed): the same on every rank."""
    errors = [None] * group.world_size
    torch.distributed.all_gather_object(errors, error, group=group.cpu_group)
    return "; ".join(f"rank {r}: {e}" for r, e in enumerate(errors) if e)


def _hash_ids(w: int, cols: int, ranges: list) -> torch.Tensor:
    """[w, hash tables, cols] hash ids; element k of table h falls in rank k % tp's
    rows of h (``ranges[rank][h]`` = (row_start, rows))."""
    tp, tables = len(ranges), max(ranges[0]) + 1
    k = torch.arange(w * cols).view(w, cols)
    owner = k % tp
    out = torch.zeros((w, tables, cols), dtype=torch.int64)
    for h in ranges[0]:
        # a rank that holds no rows of table h lends its turn to the first that does
        first = next(r for r in range(tp) if ranges[r][h][1])
        held = [r if ranges[r][h][1] else first for r in range(tp)]
        start = torch.tensor([ranges[held[r]][h][0] for r in range(tp)])
        rows = torch.tensor([ranges[held[r]][h][1] for r in range(tp)])
        out[:, h] = start[owner] + (k * 7919) % rows[owner]
    return out


def build(model, ctx, group, publish: bool = False):
    """The Engram prefetch, or None, and the reason it is off (``publish``: slot
    engram_publish, the mailbox exchange). Build runs it at every
    width it takes, with its buffers poisoned (0xFF: a NaN in BF16), and compares each
    layer's rows with SGLang's inline lookup + WKV bit for bit (which also compiles its
    GEMMs before capture)."""
    m = model.model
    engrams = {
        i: m.layers[i].engram
        for i in range(m.start_layer, m.end_layer)
        if m.layers[i].engram is not None
    }
    widths = [w for w in ctx.widths if w <= MAX_ROWS]
    if not engrams or not widths:
        return None, "no Engram layer" if not engrams else "no width it takes"
    first_layer = min(engrams)
    if first_layer == m.start_layer:
        return None, f"layer {first_layer} is the first layer: no layer to reduce in"
    binding = getattr(ctx, "binding", None)
    if binding is None:
        return None, "SGLang offers no megakernel binding (the Engram host tables)"
    reasons = (
        reason(e, group.world_size, binding.engram_host_table(e))
        for e in engrams.values()
    )
    error = _any_error(group, next((r for r in reasons if r), ""))
    if error:
        return None, error
    # the same hash ids on every rank, inside every rank's own rows of each table
    # (SGLang shards the rows on hash-column boundaries, so the ranges differ)
    ranges = [None] * group.world_size
    torch.distributed.all_gather_object(
        ranges,
        {
            e.layer_hash_index: (e.embed.row_start, e.embed.rows)
            for e in engrams.values()
        },
        group=group.cpu_group,
    )
    # rank-local setup only: a failure here is gathered before any collective runs
    try:
        prefetch = EngramPrefetch(engrams, group, first_layer)
        device = prefetch.looked.device
        ids = {w: _hash_ids(w, prefetch.hash_cols, ranges).to(device) for w in widths}
        error = ""
    except Exception as e:  # gathered: the slot is then off on every rank
        error = f"{type(e).__name__}: {e}"
    error = _any_error(group, error)
    if error:
        return None, error
    if publish:
        # collective (the mailboxes' rendezvous): every rank reaches it or none does
        prefetch = EngramPrefetch(engrams, group, first_layer, ctx.deadline_s)
    # every collective below runs on the main stream, in the same order on every
    # rank; a mismatch is recorded, not raised, so no rank leaves them early
    mismatch = ""
    for w in widths:
        prefetch.poison(0xFF)
        rows = prefetch.run(ids[w])
        for i, layer in prefetch.layers.items():
            e = layer.engram
            want, _ = e.wkv(e.embed(ids[w][:, e.layer_hash_index]).flatten(-2))
            got = rows[i]
            # the exchange copies bits: compare them (the all-reduce's sum with zeros
            # turns a -0.0 into +0.0, so its rows compare by value)
            same = (
                torch.equal(got.view(torch.int16), want.view(torch.int16))
                if prefetch.exchange is not None
                else torch.equal(got, want)
            )
            if not mismatch and not same:
                mismatch = f"layer {i} at {w} rows: not SGLang's rows"
    prefetch.kv = None
    error = _any_error(group, mismatch)
    return (None, error) if error else (prefetch, "")
