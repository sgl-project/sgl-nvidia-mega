# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# Portions of this file are copied from SGLang (Apache-2.0, see include/LICENSE.sglang):
# python/sglang/srt/speculative/dflash_worker_v2.py (DFlashWorkerV2.forward_batch_generation, extend branch) at
# revision 306ccf6.
# Modified by NVIDIA: the per-request extend and prefix lengths are copied to the device from one pinned host buffer
# with a non-blocking copy instead of two pageable copies; every other statement is unchanged.
"""Component ``dflash_prefill_async`` (env ``MUSE_VR_DFLASH_PREFILL_ASYNC=1``, default off).

After the target prefill forward, SGLang's DFLASH worker (``DFlashWorkerV2.forward_batch_generation``, extend branch)
builds two small int32 device tensors, the extend and prefix lengths of each request, with
``torch.tensor(<python list>, device=<cuda device>)`` before it writes the prompt tokens' context K/V into the draft KV
cache. That form copies from pageable host memory and then waits for the current stream (PyTorch's blocking
host-to-device copy synchronizes the stream). The scheduler runs the forward on its forward stream, so the host waits
there until the whole target prefill has run on the GPU, and the scheduler's next iteration (result processing, batch
selection, the next launch) then runs while the GPU is idle.

This component runs the same extend branch with one change: both length lists go into one pinned host tensor, which is
copied to the device with one non-blocking copy on the same stream (PyTorch's pinned-memory caching allocator keeps
the host buffer until the copy has run). The two device tensors are views of that copy and hold the same int32
values in the same order; every other statement, call, kernel and argument is the stock one, in the stock order. The
host returns from the worker as soon as the branch's work is enqueued, so the scheduler's next iteration overlaps the
prefill on the GPU. The branch runs before any verification of the new requests, so acceptance (real or synthetic) is
not involved.

Engaged for extend batches (``forward_mode.is_extend()`` or ``is_extend_in_batch``) that are not idle, with both length
lists given as Python lists or tuples of the same length, on a CUDA device, outside CUDA-graph capture, called with the
scheduler's keyword arguments. Everything else, including every decode / verify batch, calls the stock method.

``MUSE_VR_DFLASH_PREFILL_GAP=1`` (instrument, default off, independent of the opt-in above) records, for every extend
call, the GPU time between the end of the work the extend call enqueued and the point where the next
``forward_batch_generation`` call starts enqueueing (CUDA events on the current stream, read without waiting), and
writes running totals to ``$MUSE_VR_ENGAGE_DIR/prefill_gap_<pid>.jsonl`` (every 64 values or 5 s, at the next call). It measures the GPU idle that follows an
extend pass with or without the component.

Checks: tests/test_dflash_prefill_async.py (values equal to the stock form; the stock form waits for the stream and
this one does not; the branch equals the pinned SGLang source statement by statement apart from the replaced lines).
"""

from __future__ import annotations

import json
import logging
import os
import threading
import time

import torch

from . import bed

logger = logging.getLogger(__name__)

NAME = "dflash_prefill_async"  # env MUSE_VR_DFLASH_PREFILL_ASYNC
GAP_NAME = "dflash_prefill_gap"  # env MUSE_VR_DFLASH_PREFILL_GAP (instrument)
TARGET = (
    "sglang.srt.speculative.dflash_worker_v2.DFlashWorkerV2.forward_batch_generation"
)

# sha256 of the SGLang sources whose code the extend branch below copies or calls: the worker (the branch itself) and
# the position helper it calls.
PINS = {
    "sglang/srt/speculative/dflash_worker_v2.py": "8f82de8d666f81f3c1aa43dc651006f4182ff58593fdfe4eea28c5ee2a14bfc0",
    "sglang/srt/model_executor/forward_batch_info.py": "d15f12fa976acae9ba0bd4950b9366d1873abd46a8de66fd8c6ce45e3f0894f9",
}

_SCHED_KWARGS = frozenset({"on_publish", "grammar_barrier", "pp_proxy_tensors"})
_dw = None  # the stock module, bound at register()


def enabled() -> bool:
    return bed.enabled(NAME)


def gap_enabled() -> bool:
    return bed.enabled(GAP_NAME)


def pins_match() -> tuple[bool, str]:
    for rel, want in PINS.items():
        if not bed.pinned(rel, want):
            return False, f"{rel} differs from the pinned revision"
    return True, "ok"


def lens_to_device(extend_lens, prefix_lens, device):
    """The two int32 device tensors of the stock branch, from one pinned host buffer and one non-blocking copy on the
    current stream. Values, dtype and lengths equal ``torch.tensor(<list>, dtype=torch.int32, device=device)``."""
    n = len(extend_lens)
    host = torch.tensor(
        list(extend_lens) + list(prefix_lens), dtype=torch.int32, pin_memory=True
    )
    dev = host.to(device, non_blocking=True)
    return dev[:n], dev[n:]


_tally: dict = {}


def _count(key: str) -> None:
    """``bed.count`` plus a ``<key>>=<n>`` record each time the tally reaches a power of two (the engagement file is
    rewritten only on new keys)."""
    bed.count(NAME, key)
    n = _tally[key] = _tally.get(key, 0) + 1
    if n & (n - 1) == 0:
        bed.count(NAME, f"{key}>={n}")


def _eligible(batch, args, kwargs) -> str | None:
    """None when the extend branch below may run, else the reason it calls the stock method."""
    if args or not set(kwargs) <= _SCHED_KWARGS:
        return "call form"
    fm = getattr(batch, "forward_mode", None)
    if fm is None or not (
        fm.is_extend() or getattr(batch, "is_extend_in_batch", False)
    ):
        return "not extend"
    if fm.is_idle():
        return "idle"
    el, pl = getattr(batch, "extend_lens", None), getattr(batch, "prefix_lens", None)
    if (
        not isinstance(el, (list, tuple))
        or not isinstance(pl, (list, tuple))
        or len(el) != len(pl)
    ):
        return "length lists"
    if not torch.cuda.is_available() or torch.cuda.is_current_stream_capturing():
        return "device or capture"
    return None


def _extend_branch(
    self, batch, on_publish=None, grammar_barrier=None, pp_proxy_tensors=None
):
    # Copied from DFlashWorkerV2.forward_batch_generation (extend branch); see the file header.
    self._validate_phase1_sampling_support(batch)

    # Target prefill: capture DFlash aux hidden states for prompt tokens.
    batch_output = self.target_worker.forward_batch_generation(
        batch,
        pp_proxy_tensors=pp_proxy_tensors,
        capture_hidden_mode=_dw.CaptureHiddenMode.FULL,
        return_kv_loc_plan=True,
    )
    kv_loc_plan, batch_output.kv_loc_plan = batch_output.kv_loc_plan, None

    logits_output, next_token_ids = (
        batch_output.logits_output,
        batch_output.next_token_ids,
    )
    self._tp_sync.sync(_dw.SpecTpSyncSite.DFLASH_TARGET, next_token_ids)
    new_seq_lens = batch.seq_lens
    batch_output.new_seq_lens = new_seq_lens
    if on_publish is not None:
        on_publish(batch_output.new_seq_lens)

    # An idle DP rank runs the empty target prefill above to stay in
    # the DP collective, but must skip the draft KV materialization,
    # which needs per-request extend info.
    if batch.forward_mode.is_idle():
        batch_output.next_draft_input = _dw.DFlashDraftInputV2.create_idle_input(
            device=self.device
        )
        return batch_output

    if logits_output.hidden_states is None:
        raise RuntimeError(
            "DFLASH requires target aux hidden capture for prefill, but got None. "
            "Make sure the target model has DFlash layers-to-capture configured."
        )

    if batch.extend_lens is None or batch.prefix_lens is None:
        raise RuntimeError(
            "DFLASH expected extend_lens / prefix_lens to be populated in extend mode, "
            "but got None."
        )

    # Materialize prompt tokens into the draft KV cache immediately. This is required
    # for radix cache safety (the scheduler may update radix after prefill returns).
    device = next_token_ids.device
    # Modified: one pinned host buffer, one non-blocking copy (the stock form waits for the stream).
    ctx_lens, draft_seq_lens = lens_to_device(
        batch.extend_lens, batch.prefix_lens, device
    )

    if batch.out_cache_loc is None:
        raise RuntimeError("DFLASH prefill expected out_cache_loc, but got None.")
    positions, _ = _dw.compute_position(
        self.model_runner.prefill_attention_backend_str,
        draft_seq_lens,
        ctx_lens,
        int(sum(batch.extend_lens)),
    )
    # The draft KV goes to the slots the target prefill just wrote.
    self._append_target_hidden_to_draft_kv_by_loc(
        target_hidden=logits_output.hidden_states,
        cache_loc=self._draft_write_ids(kv_loc_plan),
        positions=positions,
        extend_lens=ctx_lens,
    )

    # Avoid copying large hidden-state buffers to CPU in overlap scheduling.
    logits_output.hidden_states = None

    batch_output.next_draft_input = self._make_next_draft_input_prefill(
        bonus_tokens=next_token_ids,
        seq_lens=new_seq_lens,
    )
    return batch_output


class _Gap:
    """GPU idle after each extend call: event at the end of the extend call's enqueued work, event at the start of the
    next call, both on the current stream; harvested with ``query()`` (never waits)."""

    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.open = (
            None  # end event of the last extend call, waiting for the next call's start
        )
        self.pending = []  # (end event, start event, tokens, requests)
        self.n = 0
        self.sum_ms = 0.0
        self.hist = {}  # 0.1 ms buckets (capped at 20 ms)
        self.vals = []  # most recent values (bounded)
        self.t_last = 0.0
        d = os.environ.get("MUSE_VR_ENGAGE_DIR")
        self.path = os.path.join(d, f"prefill_gap_{os.getpid()}.jsonl") if d else None

    def on_call_start(self) -> None:
        with self.lock:
            if self.open is not None:
                start = torch.cuda.Event(enable_timing=True)
                start.record()
                end, tok, nreq = self.open
                self.pending.append((end, start, tok, nreq))
                self.open = None
            self._harvest()

    def on_extend_end(self, tok: int, nreq: int) -> None:
        end = torch.cuda.Event(enable_timing=True)
        end.record()
        with self.lock:
            self.open = (end, tok, nreq)

    def _harvest(self) -> None:
        keep = []
        for end, start, tok, nreq in self.pending:
            if end.query() and start.query():
                ms = max(0.0, end.elapsed_time(start))
                self.n += 1
                self.sum_ms += ms
                b = min(int(ms * 10), 200)
                self.hist[b] = self.hist.get(b, 0) + 1
                self.vals.append((round(ms, 4), tok, nreq))
            else:
                keep.append((end, start, tok, nreq))
        self.pending = keep
        if (
            self.path
            and self.vals
            and (len(self.vals) >= 64 or time.time() - self.t_last > 5.0)
        ):
            rec = {
                "pid": os.getpid(),
                "t": time.time(),
                "n": self.n,
                "sum_ms": round(self.sum_ms, 4),
                "hist_0p1ms": {str(k): v for k, v in sorted(self.hist.items())},
                "vals": self.vals,
            }
            try:
                with open(self.path, "a") as f:
                    f.write(json.dumps(rec) + "\n")
            except OSError:
                pass
            self.vals = []
            self.t_last = time.time()


_gap: _Gap | None = None


def _is_extend(batch) -> bool:
    fm = getattr(batch, "forward_mode", None)
    return (
        fm is not None
        and (fm.is_extend() or bool(getattr(batch, "is_extend_in_batch", False)))
        and not fm.is_idle()
    )


def _around(original, worker, batch, *args, **kwargs):
    gap = _gap
    timed = gap is not None and not torch.cuda.is_current_stream_capturing()
    if timed:
        gap.on_call_start()
        # decided before the call: the worker may change the batch's forward mode while it runs
        ext = _is_extend(batch)
        el = (getattr(batch, "extend_lens", None) or ()) if ext else ()
        tok, nreq = int(sum(el)), len(el)
    why = _eligible(batch, args, kwargs) if _on else "off"
    if why is None:
        out = _extend_branch(worker, batch, **kwargs)
        _count("extend:async")
    else:
        out = original(worker, batch, *args, **kwargs)
        if _on and why != "not extend":
            _count(f"stock:{why}")
    if timed and ext:
        gap.on_extend_end(
            tok, nreq
        )  # every non-idle extend call, whichever path ran it
    return out


_on = False


def register() -> bool:
    global _dw, _gap, _on
    want, want_gap = enabled(), gap_enabled()
    if not (want or want_gap):
        return False
    ok, why = pins_match()
    if not ok:
        bed.report(NAME, False, reason=why, gap=False)
        logger.warning(
            "muse_vr_plugin: %s requested but NOT engaged (%s): stock DFLASH prefill branch runs",
            NAME,
            why,
        )
        return False
    from sglang.srt.plugins.hook_registry import HookRegistry, HookType
    from sglang.srt.speculative import dflash_worker_v2 as dw

    _dw = dw
    _on = want
    _gap = _Gap() if want_gap else None
    HookRegistry.register(TARGET, _around, HookType.AROUND)
    bed.report(NAME, want, gap=bool(want_gap))
    return want
