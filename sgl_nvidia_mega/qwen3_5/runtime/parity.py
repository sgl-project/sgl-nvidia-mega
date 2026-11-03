# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Test only: the covered step against SGLang's own forward, bit for bit, in eager
steps and inside the decode CUDA graphs (``SGLANG_NVIDIA_MEGAKERNEL_PARITY``).

``1`` runs SGLang's forward on the step, then the megakernel's, and compares the logits
and the hidden rows they return bit for bit. An eager step logs the comparison at once.
A captured step records both forwards and the comparison, which adds to device counters
on every replay; the next eager forward on the model (a new request's prefill) logs
what the replays added. ``control`` runs SGLang's forward twice instead: the run-to-run
floor of the same comparison. ``trace`` also hooks every module both forwards call and
logs the first module call whose input or output differs (eager steps only).

Graph lines also count the differing elements of the requests' own rows apart
(valid_*): a captured step of more requests replays for fewer, and SGLang's ops leave a
padded request's rows unwritten (its mamba slot is the pad slot), so both forwards read
whatever those rows held. Every line also reports how far the two forwards' outputs are
apart, which is what a numeric slot (not bitwise) is read by: per verify row, whether the two top-1 tokens
differ and the |difference| of the two log-softmaxes at SGLang's top-1, and the max abs
difference of the hidden rows.

A verify forward rewrites its requests' conv windows in place (SGLang's commit later
rolls them back to the accepted token), so the second forward runs on the windows as
they were before the first: their rows are saved and put back between the two. The
verify only reads the SSM checkpoints and only writes this step's ReplaySSM window.
An eager line's valid_rows counts the step's requests with a mamba slot. The forwards
SGLang runs before capturing a graph log as ``capture-warmup``: the graph runner's
placeholder inputs give every row the same KV slot, and there two runs of SGLang's own
forward already differ (``control``), first at the first full-attention layer's
attention, so those lines say nothing about the step.
"""

from __future__ import annotations

import logging
import os

import torch

logger = logging.getLogger(__name__)

# The SGLang names this module reads (test/unit/test_contract.py pins them)
SGLANG_READS = {
    "srt/layers/logits_processor.py": ("next_token_logits", "hidden_states"),
    "srt/model_executor/forward_context.py": ("get_attn_backend",),
    "srt/layers/attention/hybrid_linear_attn_backend.py": (
        "linear_attn_backend",
        "req_to_token_pool",
        "forward_metadata",
    ),
    "srt/layers/attention/mamba/mamba2_metadata.py": ("mamba_cache_indices",),
    "srt/mem_cache/memory_pool.py": ("mamba_pool", "mamba_cache", "conv"),
    "srt/model_executor/runner/__init__.py": ("get_is_capture_mode",),
}

MODE = os.environ.get("SGLANG_NVIDIA_MEGAKERNEL_PARITY", "0")
ENABLED = MODE in ("1", "control", "trace")

# element size -> the integer type its bits are compared as
_BITS = {1: torch.uint8, 2: torch.int16, 4: torch.int32, 8: torch.int64}


class Parity:
    """One rank's comparison counters per width: covered steps, the elements of the
    logits and of the hidden rows whose bits differ, the elements compared (zero when a
    forward returned no such output, e.g. SGLang's autotune dummy run), and the
    differing elements of the logits and hidden rows of requests with a mamba slot."""

    def __init__(self, widths: tuple[int, ...], device, model=None):
        self.counters = {
            w: torch.zeros(7, dtype=torch.int64, device=device) for w in widths
        }
        self.logged = {w: [0] * 7 for w in widths}
        # rows, top-1 differs, sum and max of |dlp|, hidden max abs; read at each flush
        self.distance = {
            w: torch.zeros(5, dtype=torch.float64, device=device) for w in widths
        }
        self.label = " control" if MODE == "control" else ""
        self.model = model  # whose modules trace mode hooks

    def run(self, step, original, megakernel):
        """``original()`` then ``megakernel()`` (``original()`` again for control) on the
        same step; returns the second's output."""
        traced = MODE == "trace" and not torch.cuda.is_current_stream_capturing()
        slots, convs = conv_rows()
        # padded rows hold -1; their slot 0 stand-in is saved and put back unchanged
        rows = None if slots is None else slots.clamp(min=0)
        saved = [conv[:, rows].clone() for conv in convs]
        want, calls = self._traced(original) if traced else (original(), None)
        want = (_clone(want.next_token_logits), _clone(want.hidden_states))
        for conv, rows_before in zip(convs, saved):
            conv[:, rows] = rows_before
        if traced:
            got, got_calls = self._traced(megakernel)
            logger.info(
                "NVIDIA megakernel parity trace %s rows=%d: %s",
                step.mode,
                step.rows,
                first_divergence(calls, got_calls),
            )
        else:
            got = original() if MODE == "control" else megakernel()
        pairs = list(zip(want, (got.next_token_logits, got.hidden_states)))
        apart = distance(*pairs[0], *pairs[1])
        if torch.cuda.is_current_stream_capturing():
            counts = self.counters[step.rows]
            counts[0:1].add_(1)
            valid = (
                None
                if slots is None or step.rows % slots.numel()
                else valid_rows(slots, step.rows)
            )
            for i, (a, b) in enumerate(pairs, start=1):
                counts[i : i + 1].add_(differing(a, b))
                counts[i + 2 : i + 3].add_(0 if a is None else a.numel())
                counts[i + 4 : i + 5].add_(differing(a, b, valid))
            if apart is not None:
                d = self.distance[step.rows]
                d[0:4:2].add_(apart[0:4:2])
                d[1:2].add_(apart[1:2])
                d[3:5].copy_(torch.maximum(d[3:5], apart[3:5]))
            return got
        from sglang.srt.model_executor.runner import get_is_capture_mode

        logger.info(
            "NVIDIA megakernel parity%s %s %s rows=%d logits_differ=%d "
            "hidden_differ=%d logits_compared=%d hidden_compared=%d "
            "logits_max_abs=%s hidden_max_abs=%s valid_rows=%s%s",
            self.label,
            "capture-warmup" if get_is_capture_mode() else "eager",
            step.mode,
            step.rows,
            *(int(differing(a, b)) for a, b in pairs),
            *(0 if a is None else a.numel() for a, _ in pairs),
            *(_max_abs(a, b) for a, b in pairs),
            "n/a" if slots is None else int((slots >= 0).sum()),
            _distance_text(apart),
        )
        return got

    def flush(self) -> None:
        """Log what the replays of captured steps added since the last flush; a host
        sync, so only outside capture."""
        if torch.cuda.is_current_stream_capturing():
            return
        for rows, counts in self.counters.items():
            now = counts.tolist()
            if now == self.logged[rows]:
                continue
            added = [n - o for n, o in zip(now, self.logged[rows])]
            self.logged[rows] = now
            apart = self.distance[rows].clone()
            self.distance[rows].zero_()
            logger.info(
                "NVIDIA megakernel parity%s graph target_verify rows=%d steps=%d "
                "logits_differ=%d hidden_differ=%d logits_compared=%d "
                "hidden_compared=%d valid_logits_differ=%d valid_hidden_differ=%d%s",
                self.label,
                rows,
                *added,
                _distance_text(apart),
            )

    def _traced(self, fn):
        """fn()'s output and every module call it made: (name, input, output)."""
        names = {m: n or "model" for n, m in self.model.named_modules()}
        calls = []

        def hook(module, args, out):
            calls.append((names[module], _first_tensor(args), _first_tensor(out)))

        handles = [m.register_forward_hook(hook) for m in names]
        try:
            return fn(), calls
        finally:
            for h in handles:
                h.remove()


def conv_rows():
    """This step's mamba slots and the pool's conv-window tensors ([layers, slots,
    ...]); (None, []) without a linear-attention backend."""
    from sglang.srt.model_executor.forward_context import get_attn_backend

    backend = get_attn_backend()
    linear = getattr(backend, "linear_attn_backend", None)
    if linear is None:
        return None, []
    cache = linear.req_to_token_pool.mamba_pool.mamba_cache
    return linear.forward_metadata.mamba_cache_indices, list(cache.conv)


def first_divergence(want, got) -> str:
    """The first call in got whose input or output bits differ from the same module's
    call of the same index in want."""
    index, seen = {}, {}
    for name, a_in, a_out in want:
        seen[name] = seen.get(name, -1) + 1
        index[(name, seen[name])] = (a_in, a_out)
    seen, common = {}, 0
    for name, b_in, b_out in got:
        seen[name] = seen.get(name, -1) + 1
        pair = index.get((name, seen[name]))
        if pair is None:
            continue
        common += 1
        for what, a, b in (("input", pair[0], b_in), ("output", pair[1], b_out)):
            if a is None or b is None:
                continue
            if a.shape != b.shape or a.dtype != b.dtype:
                return f"{name}#{seen[name]} {what}: {a.dtype}{tuple(a.shape)} against {b.dtype}{tuple(b.shape)}"
            n = int(differing(a, b))
            if n:
                return f"{name}#{seen[name]} {what}: {n} of {a.numel()} elements differ"
    return f"no diverging call among {common} common module calls"


def _first_tensor(value):
    """A clone of the first tensor in value (a tensor, a tuple / list of them, or an
    output with next_token_logits), or None."""
    if isinstance(value, torch.Tensor):
        return value.clone()
    if isinstance(value, (tuple, list)):
        return next((v.clone() for v in value if isinstance(v, torch.Tensor)), None)
    logits = getattr(value, "next_token_logits", None)
    return logits.clone() if isinstance(logits, torch.Tensor) else None


def distance(logits_a, logits_b, hidden_a, hidden_b):
    """[rows, rows whose top-1 differs, sum and max over rows of |dlp|, hidden max abs]
    on the device, dlp being the log-softmax difference at a's top-1; None without
    logits."""
    if logits_a is None or logits_b is None:
        return None
    top = logits_a.argmax(-1, keepdim=True)
    dlp = (
        torch.log_softmax(logits_b.float(), -1).gather(-1, top)
        - torch.log_softmax(logits_a.float(), -1).gather(-1, top)
    ).abs()
    hidden = (
        torch.zeros((), device=logits_a.device)
        if hidden_a is None
        else (hidden_a.float() - hidden_b.float()).abs().max()
    )
    # no host-to-device copy: this also runs inside graph capture
    return torch.stack(
        [
            torch.full((), float(top.shape[0]), device=logits_a.device),
            (logits_b.argmax(-1, keepdim=True) != top).sum().float(),
            dlp.sum(),
            dlp.max(),
            hidden,
        ]
    ).double()


def _distance_text(apart) -> str:
    if apart is None:
        return ""
    rows, top1, total, worst, hidden = apart.tolist()
    mean = total / rows if rows else 0.0
    return (
        f" rows_compared={int(rows)} top1_differ={int(top1)} dlp_mean={mean:.3g} "
        f"dlp_max={worst:.3g} hidden_dist_max={hidden:.3g}"
    )


def valid_rows(slots: torch.Tensor, rows: int) -> torch.Tensor:
    """[rows] bool: the rows of requests with a mamba slot (not the pad slot), for a step
    of ``slots.numel()`` requests of equal rows."""
    return (slots >= 0).repeat_interleave(rows // slots.numel())


def differing(a, b, valid=None):
    """The number of elements of a and b whose bits differ (a device scalar); with
    ``valid`` ([rows] bool), only in those rows."""
    if a is None or b is None:
        if a is None and b is None:
            return 0
        raise RuntimeError("parity: only one forward returned this output")
    if a.shape != b.shape or a.dtype != b.dtype:
        raise RuntimeError(
            f"parity: {a.dtype}{tuple(a.shape)} against {b.dtype}{tuple(b.shape)}"
        )
    bits = _BITS[a.element_size()]
    differ = a.contiguous().view(bits) != b.contiguous().view(bits)
    if valid is not None:
        differ = differ & valid.view(-1, *([1] * (differ.dim() - 1)))
    return differ.sum()


def _clone(t):
    return None if t is None else t.clone()


def _max_abs(a, b) -> str:
    if a is None:
        return "n/a"
    return f"{(a.float() - b.float()).abs().max().item():.3g}"
