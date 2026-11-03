# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Component ``sb_lite``: age-aware prefill/decode interval for aggregated serving (scheduling only).

SGLang's ``--prefill-decode-interval k`` makes the scheduler run at least ``k`` decode rounds after every extend
(prefill) pass before it admits the next one. A larger ``k`` spreads prefill work thinner over the decode stream, so
the requests that are decoding see fewer stalls per output token, and it delays queued prefills (time to first token).
A fixed ``k`` has to be small enough for the worst bursts: when a long uncached prompt is being prefilled chunk by
chunk, every short follow-up turn that arrives meanwhile waits behind it for ``chunks x (pass + k rounds)``.

This component keeps SGLang's own interval mechanism (``_arm_prefill_decode_interval`` / ``_should_defer_prefill``)
and only chooses the interval per extend pass:
  * normally the server's ``--prefill-decode-interval`` (``k_hi``);
  * ``MUSE_VR_SB_LITE_K_LO`` when some request that has not started decoding yet (waiting queue or the request being
    chunked) has waited ``MUSE_VR_SB_LITE_AGE_MS`` or longer, and an interval already armed is cut to ``k_lo`` as soon
    as that holds.
So the deferral is long while nobody has waited long, and short while someone has; queued prefills are bounded near
the age threshold while decode keeps a long interval in calm stretches.

Waiting time is measured on a LOGICAL clock, never the wall clock: every TP rank's scheduler advances it by the same
estimated cost for every batch it runs (``run_batch``), and stamps each request with the clock when it enters the
waiting queue. All inputs (batches, queue contents and order, the chunked request) are identical on every TP rank, so
every rank takes the same decision and the ranks never enter different collectives. Estimated costs:
decode ``a_d + b_d x running requests`` ms, extend ``a_e + b_e x extend tokens`` ms (``MUSE_VR_SB_LITE_COST``,
four comma-separated numbers; defaults below).

Nothing else changes: which requests are admitted into a pass, their order and chunk sizes are SGLang's; outputs of a
request are those of SGLang for the batches it runs in. Opt in with ``MUSE_VR_SB_LITE=1``; it needs a non-zero
``--prefill-decode-interval`` and stays off (stock) otherwise. ``MUSE_VR_SB_LITE_TRACE=<dir>`` (testing only) writes
one line per armed or cut interval per rank, for cross-rank comparison.
"""

from __future__ import annotations

import functools
import logging
import os
import threading

from . import bed

logger = logging.getLogger("muse_vr_plugin")

NAME = "sb_lite"
DEFAULT_K_LO = 3
DEFAULT_AGE_MS = 2000.0
# a_d, b_d (decode: fixed ms + ms per running request), a_e, b_e (extend: fixed ms + ms per extend token)
DEFAULT_COST = (12.0, 0.15, 30.0, 0.0135)

# sha256 of files relative to the directory that holds the ``sglang`` package.
SGLANG_PINS = {
    # run_batch, _add_request_to_queue, _arm_prefill_decode_interval, _should_defer_prefill, get_next_batch_to_run
    "sglang/srt/managers/scheduler.py": (
        "c1edcfe6f85e49bf85faf1f79129f4452445585de17cf1cdcc821de5008b1491",
        "fc96ed4b680fdf55621a5397f844cfd4cbd30213f44eec8a888b7cce75cafe60",
    ),
    # ScheduleBatch.forward_mode / is_extend_in_batch / extend_num_tokens / batch_size
    "sglang/srt/managers/schedule_batch.py": (
        "fdce33f09997f667431cf18b7f0397c64f3e707f98c7ec9cc775414a9ad837f6",
        "628d01c3fa14a669230d5413ca17144ec9c6ae33008f95f021ec1c3b882b633f",
    ),
}

_lock = threading.Lock()
_state: dict = {"engaged": None, "cls": None, "params": None, "trace": None}


class Params:
    __slots__ = ("k_lo", "age_ms", "a_d", "b_d", "a_e", "b_e")

    def __init__(self, k_lo=DEFAULT_K_LO, age_ms=DEFAULT_AGE_MS, cost=DEFAULT_COST):
        self.k_lo = int(k_lo)
        self.age_ms = float(age_ms)
        self.a_d, self.b_d, self.a_e, self.b_e = (float(x) for x in cost)
        if (
            self.k_lo < 0
            or self.age_ms < 0
            or min(self.a_d, self.b_d, self.a_e, self.b_e) < 0
        ):
            raise ValueError("negative parameter")


def batch_cost_ms(
    p: Params, is_extend: bool, batch_size: int, extend_tokens: int
) -> float:
    """Logical duration of one batch (a pure function of the batch, identical on every TP rank)."""
    if is_extend:
        return p.a_e + p.b_e * max(0, int(extend_tokens or 0))
    return p.a_d + p.b_d * max(0, int(batch_size or 0))


def oldest_age_ms(clock_ms: float, reqs) -> float:
    """Largest logical wait among ``reqs`` (requests without a stamp count as 0)."""
    age = 0.0
    for r in reqs:
        if r is None:
            continue
        t0 = getattr(r, "_sbl_t0", None)
        if t0 is not None and clock_ms - t0 > age:
            age = clock_ms - t0
    return age


def interval_for(k_hi: int, p: Params, age_ms: float) -> int:
    """Interval armed after an extend pass."""
    return p.k_lo if (age_ms >= p.age_ms and p.k_lo < k_hi) else k_hi


def _pending(sched):
    q = list(getattr(sched, "waiting_queue", None) or ())
    c = getattr(sched, "chunked_req", None)
    if c is not None:
        q.append(c)
    return q


def _trace(sched, kind: str, value: int, age: float) -> None:
    d = _state["trace"]
    if not d:
        return
    try:
        rank = getattr(sched, "tp_rank", None)
        if rank is None:
            from sglang.srt.runtime_context import get_parallel

            rank = get_parallel().tp_rank
        # SGLang arms the interval while selecting a batch, before that batch's run_batch, so the clock may not be
        # set yet on the first extend.
        line = "%d %s %d %.3f %.3f %d\n" % (
            getattr(sched, "forward_ct", -1),
            kind,
            value,
            getattr(sched, "_sbl_clk", 0.0),
            age,
            len(getattr(sched, "waiting_queue", ()) or ()),
        )
        with open(os.path.join(d, f"sb_lite_tp{rank}_{os.getpid()}.log"), "a") as f:
            f.write(line)
    except Exception:  # noqa: BLE001 - the trace is a test aid and must never stop the scheduler
        pass


def _is_extend(sched, batch) -> bool:
    # Same test as stock _arm_prefill_decode_interval.
    return (
        batch.is_extend_in_batch
        if getattr(sched, "require_mlp_sync", False)
        else batch.forward_mode.is_extend()
    )


def _wrap_run_batch(fn):
    @functools.wraps(fn)
    def run_batch(self, batch, *args, **kwargs):
        p = _state["params"]
        try:
            ext = bool(batch.forward_mode.is_extend())
            n_ext = batch.extend_num_tokens if ext else 0
            bs = batch.batch_size()
            self._sbl_clk = getattr(self, "_sbl_clk", 0.0) + batch_cost_ms(
                p, ext, bs, n_ext
            )
        except Exception:  # noqa: BLE001 - an unknown batch kind costs nothing; every rank sees the same batch
            self._sbl_clk = getattr(self, "_sbl_clk", 0.0)
        return fn(self, batch, *args, **kwargs)

    run_batch.__wrapped_by_sb_lite__ = True
    return run_batch


def _wrap_add_request(fn):
    @functools.wraps(fn)
    def _add_request_to_queue(self, req, *args, **kwargs):
        out = fn(self, req, *args, **kwargs)
        if (
            getattr(req, "_sbl_t0", None) is None
        ):  # a retracted request keeps its first stamp
            try:
                req._sbl_t0 = getattr(self, "_sbl_clk", 0.0)
            except Exception:  # noqa: BLE001 - slotted request objects: no stamp (age 0)
                pass
        return out

    _add_request_to_queue.__wrapped_by_sb_lite__ = True
    return _add_request_to_queue


def _wrap_arm(fn):
    @functools.wraps(fn)
    def _arm_prefill_decode_interval(self, batch):
        k_hi = getattr(self, "prefill_decode_interval", 0) or 0
        if k_hi == 0 or batch is None:
            return fn(self, batch)
        if _is_extend(self, batch):
            age = oldest_age_ms(getattr(self, "_sbl_clk", 0.0), _pending(self))
            k = interval_for(k_hi, _state["params"], age)
            self._prefill_decode_interval_remaining = k
            bed.count(NAME, "arm_lo" if k < k_hi else "arm_hi")
            _trace(self, "arm", k, age)
        return None

    _arm_prefill_decode_interval.__wrapped_by_sb_lite__ = True
    return _arm_prefill_decode_interval


def _wrap_defer(fn):
    @functools.wraps(fn)
    def _should_defer_prefill(self):
        rem = getattr(self, "_prefill_decode_interval_remaining", 0)
        if rem > 0:
            p = _state["params"]
            if rem > p.k_lo:
                age = oldest_age_ms(getattr(self, "_sbl_clk", 0.0), _pending(self))
                if age >= p.age_ms:
                    self._prefill_decode_interval_remaining = p.k_lo
                    bed.count(NAME, "cut")
                    _trace(self, "cut", p.k_lo, age)
        return fn(self)

    _should_defer_prefill.__wrapped_by_sb_lite__ = True
    return _should_defer_prefill


def install(params: Params, cls=None, trace_dir: str | None = None) -> None:
    """Patch the scheduler class (``cls`` for tests; default SGLang's ``Scheduler``)."""
    if cls is None:
        from sglang.srt.managers.scheduler import Scheduler as cls  # noqa: N813
    with _lock:
        _state["params"] = params
        _state["trace"] = trace_dir
        if _state["cls"] is cls:
            return
        for attr, wrap in (
            ("run_batch", _wrap_run_batch),
            ("_add_request_to_queue", _wrap_add_request),
            ("_arm_prefill_decode_interval", _wrap_arm),
            ("_should_defer_prefill", _wrap_defer),
        ):
            f = getattr(cls, attr)
            if not getattr(f, "__wrapped_by_sb_lite__", False):
                setattr(cls, attr, wrap(f))
        _state["cls"] = cls


def _params_from_env() -> Params:
    cost = os.environ.get("MUSE_VR_SB_LITE_COST", "").strip()
    cost_t = tuple(float(x) for x in cost.split(",")) if cost else DEFAULT_COST
    if len(cost_t) != 4:
        raise ValueError("MUSE_VR_SB_LITE_COST needs four numbers")
    return Params(
        k_lo=int(os.environ.get("MUSE_VR_SB_LITE_K_LO", str(DEFAULT_K_LO))),
        age_ms=float(os.environ.get("MUSE_VR_SB_LITE_AGE_MS", str(DEFAULT_AGE_MS))),
        cost=cost_t,
    )


def engage() -> bool:
    if _state["engaged"] is not None:
        return _state["engaged"]
    if not bed.enabled(NAME):
        _state["engaged"] = False
        return False
    for rel, sha in SGLANG_PINS.items():
        if not bed.pinned(rel, sha):
            bed.report(NAME, False, reason=f"pin mismatch: {rel}")
            _state["engaged"] = False
            return False
    try:
        params = _params_from_env()
    except ValueError as e:
        bed.report(NAME, False, reason=f"bad parameter ({e})")
        _state["engaged"] = False
        return False
    trace = os.environ.get("MUSE_VR_SB_LITE_TRACE") or None
    try:
        if trace:
            os.makedirs(trace, exist_ok=True)
        install(params, trace_dir=trace)
    except Exception as e:  # noqa: BLE001
        bed.report(NAME, False, reason=f"install failed: {type(e).__name__}: {e}")
        _state["engaged"] = False
        return False
    bed.report(
        NAME,
        True,
        k_lo=params.k_lo,
        age_ms=params.age_ms,
        cost=f"{params.a_d},{params.b_d},{params.a_e},{params.b_e}",
        trace=bool(trace),
    )
    _state["engaged"] = True
    return True
