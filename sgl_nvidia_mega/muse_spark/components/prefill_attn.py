# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Component ``prefill_attn``: the plugin's FP8 context-attention kernel (``ops.prefill_attn``) inside a measured window.

For prefill (context) forwards of full-attention layers over an FP8 KV cache, SGLang's TRT-LLM MHA backend calls
``flashinfer.prefill.trtllm_batch_context_with_kv_cache``. This component runs the plugin's own kernel instead for the
calls whose request lengths fall inside an engage window, and leaves every other call to the stock kernel. The window
is a measured property of the two kernels on the deployment's GPU (a table of own / stock time per one-request shape and
per small batch), so it is read at startup from the JSON file named by ``MUSE_VR_PREFILL_ATTN_WINDOW``; without a valid
file the component stays off.

Window file (JSON): ``grid_new`` and ``grid_prefix`` (ascending integer lists), ``in_window_cells`` (``[new, prefix]``
pairs of grid points where the own kernel is used) and ``r_max`` (largest request count per call). A request with ``n``
new tokens over a cached prefix of ``p`` tokens maps to the grid cell (smallest grid new >= n, largest grid prefix <= p);
a call engages when it has at most ``r_max`` requests and every request's cell is in the window.

How a call is vetted: an AROUND hook on the backend's ``forward_extend`` records the request lengths from the host-side
batch fields (``extend_prefix_lens_cpu``, ``extend_seq_lens_cpu``) for plain EXTEND forwards of full-attention layers
(no sliding window, no attention sinks, no context parallelism, FP8 KV, not under CUDA-graph capture) and only when the
window admits them; the hook on the FlashInfer call then checks the call's own options (causal, no window / sinks / LSE /
block scales, e4m3 query and KV, BF16 output, float scales, 8 query heads per KV head, head dim 64, page 128) and runs
the own kernel with those host lengths, so planning never reads lengths back from the device. Plans and the split
workspace are kept per stream, so a call runs the own kernel on whichever stream the forward uses (server warm-up and
graph-capture warm-up passes run on other streams than the serving passes).

Numerics: unsplit work units do the stock kernel's arithmetic steps and roundings in a different summation order; work
units whose key range is split (planned for long ranges at small query counts) quantize P against each chunk's own
running maximum and merge exactly, a reduction-order change. ``MUSE_VR_PREFILL_ATTN_SPLIT=0`` restricts the component to
calls the planner leaves unsplit.

Batch invariance (``MUSE_VR_PREFILL_ATTN_BI=1``, default off): by default a request's output bits can depend on the other
requests of its call, because the split chunk size comes from the call's total work and the own / stock choice is made
per call. With BI on, each request is planned from its own lengths only, and the choice is made per request: a call
whose requests are all in the window runs the own kernel; a mixed call runs the stock kernel on its out-of-window
requests only (a sub-call; the stock kernel's per-request output does not depend on the other requests of a call) and
the own kernel on its in-window requests, each writing its own rows. The request-count cap of the window does not
apply under BI. With split units off, a request whose own plan would split is treated as out of the window. Extra
counter: ``own_mixed``.

Exp arithmetic (``MUSE_VR_PREFILL_ATTN_EXP``, default ``f32``): ``f32`` is the stock kernel's per-score arithmetic;
``half_tc`` rounds the exp argument to half, takes a half-precision exp and accumulates the row sum from the e4m3 P values
on the tensor core (an approximation class of numerics, judged by the accuracy ladder before any use). Any other value
leaves the component off.

Opt in with ``MUSE_VR_PREFILL_ATTN=1`` and ``MUSE_VR_PREFILL_ATTN_WINDOW=<file>``. Engagement is guarded by sha256 pins on
the SGLang backend and the FlashInfer function whose call contract the hooks rely on. Counters: ``own``, ``own_split``,
``stock_window``, ``stock_nolens`` (no host-side lengths on the batch), ``stock_option``, ``stock_error``, one
``cell:<new>@<prefix>`` tally per engaged window cell, and ``<counter>>=<2^k>`` progress keys.
"""

from __future__ import annotations

import bisect
import contextvars
import json
import logging
import os

from . import bed

# The SGLang names this module reads (test/unit/test_contract.py pins them against the supported SGLang)
SGLANG_READS = {
    "srt/layers/attention/trtllm_mha_backend.py": ("is_cp_active",),
    "srt/plugins/hook_registry.py": ("HookRegistry", "HookType"),
}


logger = logging.getLogger("muse_vr_plugin")

NAME = "prefill_attn"
TARGET_EXTEND = (
    "sglang.srt.layers.attention.trtllm_mha_backend.TRTLLMHAAttnBackend.forward_extend"
)
TARGET_CALL = "flashinfer.prefill.trtllm_batch_context_with_kv_cache"
SGLANG_PINS = {
    # forward_extend: plain EXTEND -> _trtllm_context_attn -> flashinfer.prefill.trtllm_batch_context_with_kv_cache(...)
    "sglang/srt/layers/attention/trtllm_mha_backend.py": (
        "566d1031d644f71a8adb84d1e2a1581d3a6d1c3cac1e507451ac4cf034c39da4",
        "07223cbf82ab5ac14fcbaa9a47a8351132991016bae66c122bf09762ed8ed646",
    ),
}
FLASHINFER_PINS = {
    "flashinfer/prefill.py": (
        "0f1f9e03a7aa59ab9b95f270b85b240041717f3e08ee20c30d1332d5136d19a0",  # FlashInfer 0.7.1
        "531e220624ebc543fe865b1f6f078c5646233abc287b8407ac955e26d44f73b8",  # FlashInfer 0.7.2
    ),
}

_lens: contextvars.ContextVar = contextvars.ContextVar(
    "muse_vr_prefill_attn_lens", default=None
)
_state: dict = {"window": None, "split": True, "bi": False, "poly": 0}
_EXP_MODES = {"": 0, "f32": 0, "half_tc": 17}


def _tally(key: str) -> None:
    """Count ``key``; each time its tally reaches a power of two (>= 2) also leave a ``<key>>=<n>`` key. The engagement
    record is written only when a new key appears (no file I/O on hot eager paths), so the progress keys let it show the
    order of magnitude of every counter at the end of a run."""
    bed.count(NAME, key)
    n = bed.counts().get(NAME, {}).get(key, 0)
    if n >= 2 and n & (n - 1) == 0:
        bed.count(NAME, f"{key}>={n}")


def enabled() -> bool:
    return bed.enabled(NAME)


def _flashinfer_root():
    import importlib.util
    from pathlib import Path

    spec = importlib.util.find_spec("flashinfer")
    if spec is None or not spec.submodule_search_locations:
        return None
    return Path(list(spec.submodule_search_locations)[0]).resolve().parent


def pins_match() -> tuple[bool, str]:
    import hashlib

    for rel, want in SGLANG_PINS.items():
        if not bed.pinned(rel, want):
            return False, f"{rel} differs from the pinned revision"
    root = _flashinfer_root()
    for rel, want in FLASHINFER_PINS.items():
        try:
            got = (
                hashlib.sha256((root / rel).read_bytes()).hexdigest() if root else None
            )
        except OSError:
            got = None
        if not bed.sha_ok(got, want):
            return False, f"{rel} differs from the pinned revision"
    return True, "ok"


class Window:
    """Engage window over (new tokens, cached prefix) grid cells plus a request-count cap."""

    def in_window(self, new: int, prefix: int) -> bool:
        c = self.cell(int(new), int(prefix))
        return c is not None and c in self.cells

    def __init__(self, grid_new, grid_prefix, cells, r_max):
        self.grid_new = sorted(int(v) for v in grid_new)
        self.grid_prefix = sorted(int(v) for v in grid_prefix)
        self.cells = {(int(n), int(p)) for n, p in cells}
        self.r_max = int(r_max)
        if not self.grid_new or not self.grid_prefix or self.r_max < 1:
            raise ValueError("empty window")

    @classmethod
    def load(cls, path: str) -> Window:
        with open(path) as f:
            d = json.load(f)
        return cls(d["grid_new"], d["grid_prefix"], d["in_window_cells"], d["r_max"])

    def cell(self, new: int, prefix: int):
        i = bisect.bisect_left(self.grid_new, new)  # smallest grid new >= new
        j = (
            bisect.bisect_right(self.grid_prefix, prefix) - 1
        )  # largest grid prefix <= prefix
        if new <= 0 or i >= len(self.grid_new) or j < 0:
            return None
        return self.grid_new[i], self.grid_prefix[j]

    def admits(self, news, prefixes) -> bool:
        if not news or len(news) != len(prefixes) or len(news) > self.r_max:
            return False
        for n, p in zip(news, prefixes):
            c = self.cell(int(n), int(p))
            if c is None or c not in self.cells:
                return False
        return True


def _host_lens(forward_batch):
    """(news, prefixes) from the batch's host-side fields, or None."""
    news = getattr(forward_batch, "extend_seq_lens_cpu", None)
    pres = getattr(forward_batch, "extend_prefix_lens_cpu", None)
    if news is None or pres is None or len(news) != len(pres) or len(news) == 0:
        return None
    return [int(v) for v in news], [int(v) for v in pres]


def _around_extend(original, *args, **kwargs):
    if len(args) < 6:
        return original(*args, **kwargs)
    self, layer, forward_batch = args[0], args[4], args[5]
    win = _state["window"]
    token = None
    try:
        import torch

        if (
            win is not None
            and not _state.get("preloaded")
            and not torch.cuda.is_current_stream_capturing()
        ):
            _state["preloaded"] = (
                True  # build / load the kernel once, at the first extend (server warm-up)
            )
            from .ops import prefill_attn as op

            op.preload(
                torch.device("cuda", torch.cuda.current_device()),
                poly_pairs=_state["poly"],
            )

        ok = (
            win is not None
            and getattr(forward_batch.forward_mode, "name", "") in ("EXTEND", "MIXED")
            and kwargs.get("sinks") is None
            and getattr(layer, "sliding_window_size", -1) in (-1, None)
            and getattr(self, "data_type", None) == torch.float8_e4m3fn
            and not getattr(self, "use_fmha_v2", False)
            and not torch.cuda.is_current_stream_capturing()
        )
        if ok:
            from sglang.srt.layers.attention.trtllm_mha_backend import is_cp_active

            ok = not is_cp_active(forward_batch)
        lens = _host_lens(forward_batch) if ok else None
        if lens is not None and _state["bi"]:
            flags = [win.in_window(n, p) for n, p in zip(*lens)]
            if any(flags):
                token = _lens.set((lens[0], lens[1], flags))
                for n, p, f in zip(lens[0], lens[1], flags):
                    if f:
                        _tally("cell:%d@%d" % win.cell(int(n), int(p)))
            else:
                _tally("stock_window")
        elif lens is not None and win.admits(*lens):
            token = _lens.set(lens)
            for n, p in zip(*lens):
                _tally("cell:%d@%d" % win.cell(int(n), int(p)))
        elif ok:
            _tally("stock_window" if lens is not None else "stock_nolens")
    except Exception as e:  # noqa: BLE001 - any doubt keeps the stock path
        logger.warning("muse_vr_plugin: %s vetting failed (%s); stock kernel", NAME, e)
        token = None
    try:
        return original(*args, **kwargs)
    finally:
        if token is not None:
            _lens.reset(token)


def _call_ok(kw) -> bool:
    import torch

    q, kv = kw.get("query"), kw.get("kv_cache")
    out, out_dtype = kw.get("out"), kw.get("out_dtype")
    if q is None or not isinstance(kv, tuple) or len(kv) != 2:
        return False
    from .ops import prefill_attn as op

    k_cache, v_cache = kv
    return (
        kw.get("window_left", -1) == -1
        and kw.get("sinks") is None
        and kw.get("kv_cache_sf") is None
        and not kw.get("return_lse", False)
        and kw.get("lse") is None
        and kw.get("causal", True)
        and not kw.get("skip_softmax_threshold_scale_factor")
        and isinstance(kw.get("bmm1_scale"), float)
        and isinstance(kw.get("bmm2_scale"), float)
        and q.dim() == 3
        and k_cache.dim() == 4
        and k_cache.shape == v_cache.shape
        and op.supported(
            q.shape[1],
            k_cache.shape[1],
            q.shape[2],
            k_cache.shape[2],
            q_dtype=q.dtype,
            kv_dtype=k_cache.dtype,
            out_dtype=(out.dtype if out is not None else (out_dtype or torch.bfloat16)),
        )
        and (out is None or out.shape == q.shape)
    )


def _around_call(original, *args, **kwargs):
    lens = _lens.get()
    if lens is None or args:
        return original(*args, **kwargs)
    try:
        if not _call_ok(kwargs):
            _tally("stock_option")
            return original(*args, **kwargs)
        from .ops import prefill_attn as op

        news, pres = lens[0], lens[1]
        seq_cpu = [n + p for n, p in zip(news, pres)]
        cu_cpu = [0]
        for n in news:
            cu_cpu.append(cu_cpu[-1] + n)
        q = kwargs["query"]
        if cu_cpu[-1] != q.shape[0]:
            _tally("stock_option")
            return original(*args, **kwargs)
        if len(lens) == 3:
            return _bi_call(
                original, args, kwargs, op, q, seq_cpu, cu_cpu, list(lens[2])
            )
        units, munits, slots = op._plan(
            seq_cpu, cu_cpu, kwargs["kv_cache"][0].shape[1], q.device, None
        )
        if slots and not _state["split"]:
            _tally("stock_window")
            return original(*args, **kwargs)
        k_cache, v_cache = kwargs["kv_cache"]
        out = kwargs.get("out")
        o = op.prefill_attn(
            q,
            k_cache,
            v_cache,
            kwargs["block_tables"],
            kwargs["seq_lens"],
            kwargs["cum_seq_lens_q"],
            int(kwargs["max_q_len"]),
            kwargs["bmm1_scale"],
            kwargs["bmm2_scale"],
            out=out,
            seq_lens_cpu=seq_cpu,
            cu_seqlens_q_cpu=cu_cpu,
            poly_pairs=_state["poly"],
        )
        _tally("own_split" if slots else "own")
        return o
    except Exception as e:  # noqa: BLE001
        logger.warning(
            "muse_vr_plugin: %s own-kernel call failed (%s); stock kernel", NAME, e
        )
        _tally("stock_error")
        return original(*args, **kwargs)


def _bi_call(original, args, kwargs, op, q, seq_cpu, cu_cpu, flags):
    """Batch-invariant dispatch (MUSE_VR_PREFILL_ATTN_BI): per-request plans, per-request own / stock choice."""
    hkv = kwargs["kv_cache"][0].shape[1]
    if not _state[
        "split"
    ]:  # a request whose own plan splits is out of the window (decided from its lengths only)
        for r, f in enumerate(flags):
            if (
                f
                and op._plan(
                    [seq_cpu[r]],
                    [0, cu_cpu[r + 1] - cu_cpu[r]],
                    hkv,
                    q.device,
                    None,
                    True,
                )[2]
            ):
                flags[r] = False
        if not any(flags):
            _tally("stock_window")
            return original(*args, **kwargs)
    k_cache, v_cache = kwargs["kv_cache"]
    split = op._plan(seq_cpu, cu_cpu, hkv, q.device, None, True, flags)[2] > 0
    common = (
        q,
        k_cache,
        v_cache,
        kwargs["block_tables"],
        kwargs["seq_lens"],
        kwargs["cum_seq_lens_q"],
        int(kwargs["max_q_len"]),
        kwargs["bmm1_scale"],
        kwargs["bmm2_scale"],
    )
    if all(flags):
        o = op.prefill_attn(
            *common,
            out=kwargs.get("out"),
            seq_lens_cpu=seq_cpu,
            cu_seqlens_q_cpu=cu_cpu,
            per_request=True,
            poly_pairs=_state["poly"],
        )
        _tally("own_split" if split else "own")
        return o
    import torch

    target = kwargs.get("out")
    if target is None:
        target = torch.empty(q.shape, dtype=torch.bfloat16, device=q.device)
    _stock_subset(original, kwargs, q, seq_cpu, cu_cpu, [not f for f in flags], target)
    op.prefill_attn(
        *common,
        out=target,
        seq_lens_cpu=seq_cpu,
        cu_seqlens_q_cpu=cu_cpu,
        per_request=True,
        include=flags,
        poly_pairs=_state["poly"],
    )
    _tally("own_mixed")
    if split:
        _tally("own_split")
    return target


_SUB_CACHE: dict = {}


def _stock_subset(original, kwargs, q, seq_cpu, cu_cpu, keep, target):
    """The stock kernel on the requests flagged in ``keep`` only, written into their rows of ``target``. The stock
    kernel's output for a request does not depend on the other requests of its call, so these rows equal the rows a
    call over every request would write. Index tensors are built once per batch (the call repeats for every layer)."""
    import torch

    dev = q.device
    key = (tuple(seq_cpu), tuple(cu_cpu), tuple(keep), str(dev))
    hit = _SUB_CACHE.get(key)
    if hit is None:
        idx = [r for r, k in enumerate(keep) if k]
        rows, cu_q, cu_k = [], [0], [0]
        for r in idx:
            rows.extend(range(cu_cpu[r], cu_cpu[r + 1]))
            cu_q.append(cu_q[-1] + cu_cpu[r + 1] - cu_cpu[r])
            cu_k.append(cu_k[-1] + seq_cpu[r])
        hit = dict(
            idx=torch.tensor(idx, dtype=torch.long, device=dev),
            rows=torch.tensor(rows, dtype=torch.long, device=dev),
            cu_q=torch.tensor(cu_q, dtype=torch.int32, device=dev),
            cu_k=torch.tensor(cu_k, dtype=torch.int32, device=dev),
            n=len(idx),
            max_q=max(cu_cpu[r + 1] - cu_cpu[r] for r in idx),
            tokens=len(rows),
        )
        if len(_SUB_CACHE) > 64:
            _SUB_CACHE.clear()
        _SUB_CACHE[key] = hit
    kw = dict(kwargs)
    kw["query"] = q.index_select(0, hit["rows"])
    kw["block_tables"] = kwargs["block_tables"].index_select(0, hit["idx"])
    kw["seq_lens"] = kwargs["seq_lens"].index_select(0, hit["idx"])
    kw["cum_seq_lens_q"], kw["cum_seq_lens_kv"] = hit["cu_q"], hit["cu_k"]
    kw["batch_size"], kw["max_q_len"] = hit["n"], hit["max_q"]
    kw["out"] = torch.empty(
        (hit["tokens"],) + tuple(target.shape[1:]), dtype=target.dtype, device=dev
    )
    original(**kw)
    target.index_copy_(0, hit["rows"], kw["out"])


def register() -> bool:
    if not enabled():
        return False
    ok, why = pins_match()
    if not ok:
        bed.report(NAME, False, reason=why)
        return False
    path = os.environ.get("MUSE_VR_PREFILL_ATTN_WINDOW", "")
    try:
        _state["window"] = Window.load(path)
    except Exception as e:  # noqa: BLE001
        bed.report(NAME, False, reason=f"no usable window file ({type(e).__name__})")
        return False
    _state["split"] = os.environ.get("MUSE_VR_PREFILL_ATTN_SPLIT", "1").strip() not in (
        "0",
        "false",
        "no",
        "off",
    )
    _state["bi"] = os.environ.get("MUSE_VR_PREFILL_ATTN_BI", "0").strip() in (
        "1",
        "true",
        "yes",
        "on",
    )
    exp = os.environ.get("MUSE_VR_PREFILL_ATTN_EXP", "f32").strip().lower()
    if exp not in _EXP_MODES:
        bed.report(
            NAME,
            False,
            reason=f"unknown MUSE_VR_PREFILL_ATTN_EXP={exp!r} (f32 | half_tc)",
        )
        return False
    _state["poly"] = _EXP_MODES[exp]
    from sglang.srt.plugins.hook_registry import HookRegistry, HookType

    HookRegistry.register(TARGET_EXTEND, _around_extend, HookType.AROUND)
    HookRegistry.register(TARGET_CALL, _around_call, HookType.AROUND)
    w = _state["window"]
    bed.report(
        NAME,
        True,
        cells=len(w.cells),
        r_max=w.r_max,
        split=_state["split"],
        bi=_state["bi"],
        exp=exp or "f32",
    )
    return True
