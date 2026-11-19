# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Test only: covered steps against SGLang's own forward (``SGLANG_NVIDIA_MEGAKERNEL_PARITY``).

``1`` runs SGLang's forward on the step, puts back the recurrent state it changed,
runs the executor, and compares the logits and the HC streams (``hidden_states``) bit
for bit. ``layer`` also compares every decoder layer's output streams (SGLang's
through forward hooks on ``model.layers[i]``, the executor's from its layer loop) and
reports the first differing layer. ``control`` runs SGLang's forward twice: the
run-to-run floor of the same comparison.

An eager step logs its line at once. A captured step records both forwards, the state
restore and the comparison in the graph; the comparison adds to device counters on
every replay, and the next eager forward on the model logs what the replays added
(``graph`` lines).

State a forward changes in place and later reads, put back between the two forwards
(rows of this step's state slots only): the GDN conv states (a verify's conv update
overwrites them; SGLang's commit restores them later), on a prefill also the GDN
temporal state, and the PLE n-gram context and short-conv state. Everything else a
covered step writes is written with the same values by both forwards (K/V and QSA
rings at this step's slots, verify intermediates, ring entries past the cursors)."""

from __future__ import annotations

import logging
import os

import torch

logger = logging.getLogger(__name__)

# The SGLang names the parity check reads (test/unit/test_contract.py)
SGLANG_READS = {
    "srt/mem_cache/kv/decl.py": ("STATE_PART",),
    "srt/mem_cache/ple_state_pool.py": ("NGRAM_PART", "SHORT_CONV_PART"),
    "srt/model_executor/forward_context.py": (
        "get_attn_backend",
        "get_token_to_kv_pool",
    ),
    "srt/model_executor/runner/__init__.py": ("get_is_capture_mode",),
}

# Reads and mirrors that need an SGLang change newer than the unit job's pin: the
# contract test skips each, naming the change, only where the installed SGLang lacks it.
SGLANG_REQUIRES = {
    "srt/mem_cache/kv/decl.py:STATE_PART": "sglang#48007",
}

MODE = os.environ.get("SGLANG_NVIDIA_MEGAKERNEL_PARITY", "0")
ENABLED = MODE in ("1", "layer", "control")

_BITS = {1: torch.uint8, 2: torch.int16, 4: torch.int32, 8: torch.int64}
# steps, logits differ, hidden differ, logits compared, hidden compared, verify/decode
# rows whose top-1 token differs
_FIELDS = 6
# max |logit difference|, max |hidden difference|
_DIST = 2


def _scalar(value: int, device) -> torch.Tensor:
    # a fill kernel, not a host-to-device copy, so it can be captured
    return torch.full((), int(value), dtype=torch.int64, device=device)


def _differing(a, b) -> torch.Tensor:
    if a is None or b is None or a.shape != b.shape or a.dtype != b.dtype:
        return _scalar(-1, _device(a, b))
    bits = _BITS[a.element_size()]
    return (a.contiguous().view(bits) != b.contiguous().view(bits)).sum()


def _top1_differing(a, b, device) -> torch.Tensor:
    if a is None or b is None or a.shape != b.shape or a.ndim != 2:
        return _scalar(0, device)
    return (a.argmax(dim=-1) != b.argmax(dim=-1)).sum()


def _max_abs(a, b, device) -> torch.Tensor:
    if a is None or b is None or a.shape != b.shape or a.numel() == 0:
        return torch.zeros((), dtype=torch.float32, device=device)
    return (a.float() - b.float()).abs().max()


def _device(*ts):
    for t in ts:
        if t is not None:
            return t.device
    return torch.device("cuda", torch.cuda.current_device())


def _numel(t) -> int:
    return 0 if t is None else t.numel()


def _outputs(out):
    logits = getattr(out, "next_token_logits", None)
    hidden = getattr(out, "hidden_states", None)
    return (
        None if logits is None else logits.clone(),
        None if hidden is None else hidden.clone(),
    )


def _state_rows(step, forward_batch):
    """(tensor, slot dim, slots) of the recurrent state this step's forward changes."""
    from sglang.srt.mem_cache.kv.decl import STATE_PART
    from sglang.srt.mem_cache.ple_state_pool import NGRAM_PART, SHORT_CONV_PART
    from sglang.srt.model_executor.forward_context import (
        get_attn_backend,
        get_token_to_kv_pool,
    )

    backend = get_attn_backend()
    if not hasattr(backend, "forward_state_slots"):  # the draft: no recurrent state
        return []
    router = get_token_to_kv_pool().layer_router
    mamba = router.part_view(STATE_PART)
    if mamba is None:
        return []
    slots = (
        backend.forward_state_slots(int(forward_batch.batch_size), None)
        .working.long()
        .clamp(min=0)
    )
    tensors = [(conv, 1) for conv in mamba.mamba_cache.conv]
    if step.mode == "prefill":
        tensors.append((mamba.mamba_cache.temporal, 1))
    ngram = router.part_view(NGRAM_PART)
    if ngram is not None and getattr(ngram, "context", None) is not None:
        tensors.append((ngram.context, 0))
    short_conv = router.part_view(SHORT_CONV_PART)
    if short_conv is not None and getattr(short_conv, "conv_state", None) is not None:
        tensors.append((short_conv.conv_state, 1))
    return [(t, dim, slots) for t, dim in tensors]


# Captured (mode, rows) keys a server can hold counters for; preallocated eagerly so
# a capture records additions to them, never their allocation or zeroing.
_MAX_KEYS = 4096


class Parity:
    def __init__(self, model, *, draft: bool):
        self.model = model
        self.label = ("draft" if draft else "target") + (
            " control" if MODE == "control" else ""
        )
        m = model.model
        device = torch.device("cuda", torch.cuda.current_device())
        self.table = torch.zeros(_MAX_KEYS, _FIELDS, dtype=torch.int64, device=device)
        self.dist = torch.zeros(_MAX_KEYS, _DIST, dtype=torch.float32, device=device)
        self.layer_table = torch.zeros(
            _MAX_KEYS, m.end_layer - m.start_layer, dtype=torch.int64, device=device
        )
        self.rows: dict[tuple, int] = {}  # captured key -> its table row
        self.logged: dict[tuple, list] = {}
        # set while a forward runs: the executor's per-layer outputs
        self.layers_out: list | None = None

    # --- per-layer capture ---------------------------------------------------
    def record_layer(self, i: int, hidden: torch.Tensor) -> None:
        if self.layers_out is not None:
            self.layers_out.append((i, hidden.clone()))

    def _stock_layers(self, stock):
        outs, handles = [], []
        m = self.model.model
        for i in range(m.start_layer, m.end_layer):

            def hook(module, args, out, i=i):
                outs.append((i, (out[0] if isinstance(out, tuple) else out).clone()))

            handles.append(m.layers[i].register_forward_hook(hook))
        try:
            result = stock()
        finally:
            for h in handles:
                h.remove()
        return result, outs

    # --- the comparison --------------------------------------------------------
    def run(self, step, forward_batch, stock, executor):
        saved = [
            (t, dim, slots, t.index_select(dim, slots))
            for t, dim, slots in _state_rows(step, forward_batch)
        ]
        if MODE == "layer":
            want, want_layers = self._stock_layers(stock)
        else:
            want, want_layers = stock(), []
        want_logits, want_hidden = _outputs(want)
        for t, dim, slots, rows in saved:
            t.index_copy_(dim, slots, rows)
        if MODE == "control":
            got = stock()
            got_layers = []
        else:
            self.layers_out = [] if MODE == "layer" else None
            try:
                got = executor()
            finally:
                got_layers, self.layers_out = self.layers_out or [], None
        got_logits, got_hidden = _outputs(got)
        device = _device(got_logits, got_hidden)
        values = torch.stack(
            [
                _scalar(1, device),
                _differing(want_logits, got_logits),
                _differing(want_hidden, got_hidden),
                _scalar(_numel(got_logits), device),
                _scalar(_numel(got_hidden), device),
                _top1_differing(want_logits, got_logits, device),
            ]
        )
        dist = torch.stack(
            [
                _max_abs(want_logits, got_logits, device),
                _max_abs(want_hidden, got_hidden, device),
            ]
        )
        per_layer = None
        if want_layers and got_layers:
            got_by = dict(got_layers)
            per_layer = torch.stack(
                [_differing(w, got_by.get(i)) for i, w in want_layers]
            )
        key = (step.mode, step.rows)
        if torch.cuda.is_current_stream_capturing():
            row = self.rows.setdefault(key, len(self.rows))
            if row >= _MAX_KEYS:
                raise RuntimeError("parity: more captured steps than counter rows")
            self.table[row].add_(values)
            torch.maximum(self.dist[row], dist, out=self.dist[row])
            if per_layer is not None and per_layer.numel() == self.layer_table.shape[1]:
                self.layer_table[row].add_(per_layer)
        else:
            from sglang.srt.model_executor.runner import get_is_capture_mode

            kind = "capture-warmup" if get_is_capture_mode() else "eager"
            self._log(kind, key, values.tolist(), per_layer, dist.tolist())
        return got

    def flush(self) -> None:
        """Log what the captured steps' replays added since the last flush."""
        if torch.cuda.is_current_stream_capturing():
            return
        if not self.rows:
            return
        table = self.table.tolist()
        dist = self.dist.tolist()
        for key, row in self.rows.items():
            now = table[row]
            before = self.logged.setdefault(key, [0] * _FIELDS)
            delta = [a - b for a, b in zip(now, before)]
            if delta[0] <= 0:
                continue
            self.logged[key] = now
            layers = self.layer_table[row] if MODE == "layer" else None
            # the max is since capture, not since the last flush
            self._log("graph", key, delta, layers, dist[row])

    def _log(self, kind, key, values, per_layer, dist) -> None:
        steps, logits_differ, hidden_differ, logits_n, hidden_n, top1 = values
        first = ""
        if per_layer is not None:
            differ = [(i, int(n)) for i, n in enumerate(per_layer.tolist()) if n]
            first = (
                f" first_layer_differ={differ[0][0]} layers_differ={len(differ)}"
                if differ
                else " layers_differ=0"
            )
        logger.info(
            "NVIDIA megakernel parity %s %s %s rows=%d steps=%d logits_differ=%d "
            "hidden_differ=%d logits_compared=%d hidden_compared=%d%s top1_differ=%d "
            "max_abs_logit=%.4g max_abs_hidden=%.4g",
            self.label,
            kind,
            key[0],
            key[1],
            steps,
            logits_differ,
            hidden_differ,
            logits_n,
            hidden_n,
            first,
            top1,
            dist[0],
            dist[1],
        )
