# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Own extend-attention op: causal attention of a prefill extend's new tokens over long cached FP8 prefixes, computed by
the plugin's own context kernel with a balanced work split (plan.split), for calls inside a measured window; every
other call is declined (returns None) and the caller runs its stock path.

One boundary, shaped to move unchanged into a kernel package:
  extend_attention(call, *, <the stock context call's keywords>) -> out | None
  prepare(device_index, ...)   build every kernel, grow the scratch once to its cap, run each kernel once; idempotent
  stats()                      engagement counters (own / own_split / decline:<reason>), scratch size
The op reads only its arguments (host lengths come in ``call``); it imports torch and cuda.bindings only, never calls
another attention kernel, and refuses to build or allocate under CUDA-graph capture (it declines there).

Numerics: per query row the stock kernel's operations (e4m3 Q / K / V, fp32 scores, fp32 ex2 against the running maximum,
P = e4m3(448 p), fp32 row sum of the unquantized p, fp32 accumulation, bf16 output) in a different order; query blocks
cut by the split are merged exactly in fp32 in key order. Deterministic and batch-invariant per request.
"""

from __future__ import annotations

import functools
import threading

import torch

from . import launch as _launch
from .plan import (  # noqa: F401  (ExtendCall is part of the boundary)
    ExtendCall,
    decline_reason,
    split,
)

_lock = threading.Lock()
_state: dict = {
    "kernels": {},
    "ws": {},
    "cap_slots": {},
    "plans": {},
    "pin": {},
    "stream": {},
    "stats": {},
}
PLAN_CACHE = 8
PIN_INTS = 1 << 16


def _count(key: str, n: int = 1) -> None:
    s = _state["stats"]
    s[key] = s.get(key, 0) + n


def last_decline() -> str | None:
    """The reason of the most recent extend_attention call's decline (None when it ran)."""
    return _state.get("last")


def stats() -> dict:
    out = dict(_state["stats"])
    for dev, cap in _state["cap_slots"].items():
        out[f"scratch_mb@{dev}"] = round(cap * 256 * (64 * 4 + 8) / 2**20, 1)
    for dev, k in _state["kernels"].items():
        out[f"pfa_opt@{dev}"] = getattr(
            k, "opt", 0
        )  # the attention kernel's synchronization option (launch.OPT_ENV)
    return out


@functools.lru_cache(maxsize=None)
def _num_sms(device_index: int) -> int:
    return torch.cuda.get_device_properties(device_index).multi_processor_count


def prepare(
    device_index: int,
    *,
    max_requests: int = 4,
    max_new_tokens: int = 16384,
    num_kv_heads: int = 4,
    num_q_heads: int = 32,
) -> None:
    """Build the kernels, grow the split scratch once to the cap for ``max_requests`` requests of up to
    ``max_new_tokens`` new tokens, launch each kernel once on small dummies, and synchronize. Call on every rank before
    serving (never inside a served forward); safe to call again."""
    if torch.cuda.is_current_stream_capturing():
        raise RuntimeError("extend_attn_b.prepare under CUDA-graph capture")
    with _lock:
        dev = torch.device("cuda", device_index)
        if device_index not in _state["kernels"]:
            _state["kernels"][device_index] = _launch.Kernels(device_index)
        sms = _num_sms(device_index)
        blocks = -(-max_new_tokens // 32) * num_kv_heads
        cap = (
            max_requests * 2 * (blocks + sms)
        )  # slots: at most two pieces per cut block, one cut per range boundary
        if _state["cap_slots"].get(device_index, 0) < cap:
            _state["ws"][device_index] = (
                torch.empty(cap * 256 * 64, dtype=torch.float32, device=dev),
                torch.empty(cap * 256 * 2, dtype=torch.float32, device=dev),
            )
            _state["cap_slots"][device_index] = cap
        # one launch of each kernel on a dummy request that splits (prefix long enough for several blocks per range)
        hkv, d, page = num_kv_heads, 64, 128
        t, c = 64, 64 * page
        pages = (t + c + page - 1) // page
        kc = torch.zeros(pages, hkv, page, d, dtype=torch.float8_e4m3fn, device=dev)
        q = torch.zeros(t, num_q_heads, d, dtype=torch.float8_e4m3fn, device=dev)
        out = torch.empty(t, num_q_heads, d, dtype=torch.bfloat16, device=dev)
        table = torch.arange(pages, dtype=torch.int32, device=dev).view(1, -1)
        _run(
            device_index,
            [t],
            [c],
            hkv,
            q,
            kc,
            kc,
            table,
            torch.tensor([t + c], dtype=torch.int32, device=dev),
            torch.tensor([0, t], dtype=torch.int32, device=dev),
            out,
            0.125,
            1.0,
            min_tiles=1,
        )
        torch.cuda.synchronize(dev)


def _stage(device_index: int, vals: list) -> torch.Tensor:
    """A device int32 tensor holding ``vals``, copied from a pinned slot in stream order (never a blocking copy)."""
    dev = torch.device("cuda", device_index)
    st = _state["pin"].get(device_index)
    if st is None:
        buf = torch.zeros((4, PIN_INTS), dtype=torch.int32).pin_memory()
        st = {"buf": buf, "np": buf.numpy(), "ev": [None] * 4, "i": 0}
        _state["pin"][device_index] = st
    n = len(vals)
    if n > PIN_INTS:
        raise ValueError("plan table larger than the staging slot")
    i = st["i"]
    st["i"] = (i + 1) % 4
    if st["ev"][i] is not None:
        st["ev"][i].synchronize()  # the copy that last read this slot has run
    st["np"][i, :n] = vals
    out = torch.empty(n, dtype=torch.int32, device=dev)
    out.copy_(st["buf"][i, :n], non_blocking=True)
    e = torch.cuda.Event()
    e.record(torch.cuda.current_stream(dev))
    st["ev"][i] = e
    return out


def _plan(device_index: int, new_lens, prefix_lens, hkv: int, min_tiles=None):
    key = (device_index, tuple(new_lens), tuple(prefix_lens), hkv, min_tiles)
    hit = _state["plans"].get(key)
    if hit is not None:
        return hit
    kw = {} if min_tiles is None else {"min_tiles": min_tiles}
    units, offs, merges, slots = split(
        new_lens, prefix_lens, hkv, _num_sms(device_index), **kw
    )
    import numpy as np

    vals = np.concatenate(
        [
            np.asarray(units, dtype=np.int32).ravel(),
            np.asarray(offs, dtype=np.int32),
            np.asarray(merges, dtype=np.int32).ravel(),
        ]
    )
    dev_all = _stage(device_index, vals)
    nu, no = int(np.asarray(units).size), len(offs)
    hit = (
        dev_all[:nu].view(-1, 8),
        dev_all[nu : nu + no],
        dev_all[nu + no :].view(-1, 8),
        slots,
    )
    if len(_state["plans"]) >= PLAN_CACHE:
        _state["plans"].clear()
    _state["plans"][key] = hit
    return hit


def _run(
    device_index,
    new_lens,
    prefix_lens,
    hkv,
    q,
    k_cache,
    v_cache,
    table,
    seq_lens,
    cu_q,
    out,
    bmm1,
    bmm2,
    min_tiles=None,
) -> bool:
    units, cta, munits, slots = _plan(
        device_index, new_lens, prefix_lens, hkv, min_tiles
    )
    if slots > _state["cap_slots"].get(device_index, 0):
        _count("decline:scratch")
        _state["last"] = "scratch"
        return False
    ws_o, ws_ml = _state["ws"][device_index]
    _launch.launch(
        _state["kernels"][device_index],
        units=units,
        cta_units=cta,
        munits=munits,
        ws_o=ws_o,
        ws_ml=ws_ml,
        q=q,
        k_cache=k_cache,
        v_cache=v_cache,
        page_table=table,
        seq_lens=seq_lens,
        cu_q=cu_q,
        out=out,
        bmm1_scale=bmm1,
        bmm2_scale=bmm2,
    )
    _count("own_split" if munits.shape[0] else "own")
    return True


def extend_attention(
    call: ExtendCall,
    *,
    query,
    kv_cache,
    block_tables,
    seq_lens,
    cum_seq_lens_q,
    max_q_len,
    bmm1_scale,
    bmm2_scale,
    out=None,
    out_dtype=None,
    window_left=-1,
    sinks=None,
    kv_cache_sf=None,
    causal=True,
    skip_softmax_threshold_scale_factor=None,
    serve=None,
    **_unused,
):
    """Run the call with the own kernel and return ``out`` (allocated when None), or return None (declined; counted):
    the caller then runs its stock path. Host checks only before any launch.

    ``serve`` (optional, one bool per request): run only those requests (the window is judged on them alone) and leave the
    other requests' rows untouched for the caller to fill; None = every request."""
    reason = None
    dev = query.device
    di = dev.index if dev.index is not None else torch.cuda.current_device()
    if di not in _state["kernels"]:
        reason = "not_prepared"
    elif torch.cuda.is_current_stream_capturing():
        reason = "capturing"
    elif (
        _state["stream"].get(di, torch.cuda.current_stream(dev).cuda_stream)
        != torch.cuda.current_stream(dev).cuda_stream
    ):
        reason = "stream"  # one serving stream: the first LAUNCHED call's (persistent buffers are used in its order)
    elif (
        window_left not in (-1, None)
        or sinks is not None
        or kv_cache_sf is not None
        or not causal
        or skip_softmax_threshold_scale_factor
        or not isinstance(bmm1_scale, float)
        or not isinstance(bmm2_scale, float)
    ):
        reason = "option"
    elif not (isinstance(kv_cache, tuple) and len(kv_cache) == 2):
        reason = "kv_layout"
    else:
        k_cache, v_cache = kv_cache
        hq, hkv = query.shape[1], k_cache.shape[1]
        if (
            query.dtype != torch.float8_e4m3fn
            or k_cache.dtype != torch.float8_e4m3fn
            or query.dim() != 3
            or query.shape[2] != 64
            or k_cache.dim() != 4
            or k_cache.shape[2] != 128
            or k_cache.shape[3] != 64
            or hq != 8 * hkv
            or v_cache.shape != k_cache.shape
            or query.stride(2) != 1
            or query.stride(1) != 64
            or k_cache.stride(3) != 1
            or v_cache.stride(3) != 1
            or (out is not None and (out.stride(2) != 1 or out.stride(1) != 64))
            or (
                out is not None
                and (out.dtype != torch.bfloat16 or out.shape != query.shape)
            )
            or (out is None and out_dtype not in (None, torch.bfloat16))
        ):
            reason = "contract"
        elif (
            sum(call.new_lens) > query.shape[0]
            or len(call.new_lens) > block_tables.shape[0]
            or cum_seq_lens_q.shape[0] < len(call.new_lens) + 1
            or seq_lens.shape[0] < len(call.new_lens)
        ):
            reason = "rows"  # trailing padded rows / request slots (a graph replay's bucket) are allowed and left untouched
        elif serve is not None and len(serve) != len(call.new_lens):
            reason = "serve"
        else:
            sub = (
                call
                if serve is None
                else ExtendCall(
                    tuple(t for t, s in zip(call.new_lens, serve) if s),
                    tuple(c for c, s in zip(call.prefix_lens, serve) if s),
                    call.layer_id,
                    call.mode,
                    call.capturing,
                )
            )
            reason = decline_reason(sub, hq, _num_sms(di))
    _state["last"] = reason
    if reason is not None:
        _count("decline:" + reason)
        return None
    _state["stream"].setdefault(
        di, torch.cuda.current_stream(dev).cuda_stream
    )  # bound by the first launch only
    if out is None:
        out = torch.empty(query.shape, dtype=torch.bfloat16, device=dev)
    news, pres = call.new_lens, call.prefix_lens
    if (
        serve is not None
    ):  # unserved requests get no units (plan skips zero-length requests); their rows stay untouched
        news = tuple(t if s else 0 for t, s in zip(news, serve))
        pres = tuple(c if s else 0 for c, s in zip(pres, serve))
    ok = _run(
        di,
        news,
        pres,
        kv_cache[0].shape[1],
        query,
        kv_cache[0],
        kv_cache[1],
        block_tables,
        seq_lens,
        cum_seq_lens_q,
        out,
        bmm1_scale,
        bmm2_scale,
    )
    return out if ok else None
