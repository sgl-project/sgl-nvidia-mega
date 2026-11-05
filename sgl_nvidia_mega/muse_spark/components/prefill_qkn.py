# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Component ``prefill_qkn`` (opt-in, default off): the prefill QK norm of ``MuseSpark1xAttention.gated_attention`` on
the plugin's kernel.

In a plain EXTEND forward SGLang runs, per attention layer, the fused QK-norm Triton kernel without the KV store
(per-head RMS norm of q and k, RoPE in the layers that have it; bf16 q / k out); the TRT-LLM MHA backend then converts q
and stores K / V as usual. This component runs ``ops.prefill_qkn.qk_norm`` instead of that Triton launch: one warp per
token, the same arithmetic, the same bf16 outputs (bitwise: ``tests/test_prefill_qkn.py``), so everything after it
(the attention call with its KV store, the output norm and gate) is SGLang's code unchanged. It is safe inside the
prefill CUDA-graph capture (no host synchronization; it allocates its outputs as the stock call does), which is where
every prefill pass of the breakable prefill graphs runs it.

A call takes this path only when the stock call would run that Triton launch: plain EXTEND, the fused QK norm and
output gate configured, SGLang's Triton backend for the QK norm (no torch.compile, no fused-op override, eligible
inputs), not inside a torch.compile trace, a row count for which the Triton launch reduces 8 columns per thread (``ops.prefill_qkn.covers``), head_dim 64, int64 contiguous positions and an aligned QKV / RoPE table. Every
other call runs SGLang's method unchanged; step-asides are counted per reason and logged once per reason (WARNING).

Engagement needs the sha256 of the SGLang sources whose behaviour the hook mirrors to match the pinned revision, and
the Triton version whose code generation the arithmetic reproduces. Opt in with ``MUSE_VR_PREFILL_QKN=1``.
Counters: ``own:rope`` / ``own:norope`` (with ``capture`` / ``eager``) and ``stock:<reason>`` (with ``>=2^k`` progress keys).
q8 mode (``MUSE_VR_PREFILL_QKN_Q8=1``, default off; see ``Q8_PINS``): ``own:q8:rope`` / ``own:q8:norope`` and
``q8:stock:<reason>``.
"""

from __future__ import annotations

import contextlib
import logging
import os

import torch

from . import bed

# The SGLang names this module reads (test/unit/test_contract.py pins them against the supported SGLang)
SGLANG_READS = {
    "kernels/fused_op.py": ("_oot_dispatch_key", "get_fused_op_backend"),
    "srt/layers/attention/trtllm_mha_backend.py": (
        "TRTLLMHAAttnBackend",
        "is_cp_active",
    ),
    "kernels/spec.py": ("KernelBackend",),
    "srt/model_executor/forward_context.py": ("get_attn_backend",),
    "srt/plugins/hook_registry.py": ("HookRegistry", "HookType"),
}


logger = logging.getLogger("muse_vr_plugin")

NAME = "prefill_qkn"
TARGET = "sglang.srt.models.muse_spark.MuseSpark1xAttention.gated_attention"
PINS = {
    "sglang/srt/models/muse_spark.py": (
        "7e5018b3cb2ef41ee6cf09407e91e28dde069e5a33719ea58beb0d7137087355",
        "8bafbb8802fb0d9f81c190e87b9012fca546182be08bef6c8b3db814d27bfe19",
        "c0db42467a960c509b658c3c40570ee889f73d41b52cdbd80b9c291c1792a33f",
    ),
    "sglang/kernels/ops/muse_spark_v12/fused_qk_norm.py": (
        "99ed08f1f35bbd4174c4a40becfc7978b7a5ba0a055818e79470254e5c8a2b67",
        "af4a5bb8de205a99761e6959447748e2e00af0831e58e37fa0f862ce9e964a25",
    ),
    "sglang/srt/layers/radix_attention.py": (
        "df6e94b67b61feb4f8e3a7b1fdb0c0444cc56e4bf9d90a622e028bddff31619e",
        "56a6acb0aaeb78c7c9bb8d1d78e5a23ac51fa35436b993aeb5a630cc9f285ff0",
    ),
}
# The arithmetic reproduces the fused QK-norm Triton kernel as each validated Triton release compiles it
# (ops/swa_verify_attn.py, QK_ARITH / qk_arith); other releases keep the stock call.

# Store mode (``MUSE_VR_PREFILL_QKN_STORE=1``, on top of ``MUSE_VR_PREFILL_QKN=1``): the QK norm moves into the eager
# attention region of the prefill graph and runs fused with the e4m3 q conversion and the e4m3 K / V cache store
# (``ops.prefill_qkn.prefill_qkn``), replacing the backend's own FP8 q / KV store for that call. The gated attention
# hook passes the raw q / k / v and a spec keyword to the attention layer; SGLang's eager attention path narrows the
# call to the real tokens and forwards keywords to the backend, where ``_extend_hook`` consumes the spec. Every call
# the backend's fused FP8 path would not take falls back to the norm-only kernel followed by the stock call, so the
# backend never receives an un-normalized q. On a mixed NVFP4 / FP8 hybrid pool (``nvfp4_kv_mixed``) the decision and the
# fused store of a sliding-window layer run under the FP8-KV view that component applies to that layer's backend call, so its
# FP8 sub-pool is written exactly as an FP8 pool is; full-attention (NVFP4) layers keep the stock store.
TARGET_EXTEND = (
    "sglang.srt.layers.attention.trtllm_mha_backend.TRTLLMHAAttnBackend.forward_extend"
)
STORE_KW = "muse_vr_pqk_store"
STORE_PINS = {
    "sglang/srt/layers/attention/trtllm_mha_backend.py": (
        "07223cbf82ab5ac14fcbaa9a47a8351132991016bae66c122bf09762ed8ed646",
    ),
    "sglang/kernels/ops/kvcache/fused_fp8_qkv_kv_cache.py": (
        "88755d730cec527a7135aabff1c57779d11752ea7b286343c47c285a0be39fa3",
    ),
    "sglang/kernels/jit/csrc/attention/fused_fp8_qkv_kv_cache.cuh": (
        "1e28bbf79d380275353a7275f2d4ab7b77d15c102d8eb8677537637bafcea130",
    ),
    "sglang/srt/layers/attention/graph_utils.py": (
        "01b178e4c1248d19ce782aa7d8f8c4f292b87fbb7a5a20e250427a3eb1edf764",
    ),
    "sglang/srt/layers/attention/base_attn_backend.py": (
        "3ed449219c73b146bc3e6472390c933cdd1da18c88edcd6bfbb7d530cf4dafe0",
    ),
}
_store_state = {"engaged": False}

# q8 mode (``MUSE_VR_PREFILL_QKN_Q8=1``, on top of ``MUSE_VR_PREFILL_QKN=1``): where the norm-only form runs and the
# attention call goes to SGLang's TRT-LLM MHA backend on its native-NVFP4 prefill path (which converts q with
# ``q.to(torch.float8_e4m3fn)``, an elementwise kernel), the norm kernel writes q as e4m3 directly
# (``ops.prefill_qkn.qk_norm_q8``: the bytes of that conversion of the bf16 q) and the attention layer receives it; the
# backend's conversion then returns it unchanged. k and v are the same tensors as before. Not for FP8 layers of a mixed
# NVFP4 / FP8 pool (their backend call takes the FP8 path), context parallelism, or other forward modes; those calls
# keep the bf16 q (counted ``q8:stock:<reason>``). Engagement needs the pins of the SGLang sources whose behaviour it
# relies on (the backend's q conversion, the eager attention region's output dtype following v).
Q8_PINS = {
    rel: STORE_PINS[rel]
    for rel in (
        "sglang/srt/layers/attention/trtllm_mha_backend.py",
        "sglang/srt/layers/attention/graph_utils.py",
        "sglang/srt/layers/attention/base_attn_backend.py",
    )
}
_q8_state = {"engaged": False}

_logged: set = set()


def enabled() -> bool:
    return os.environ.get("MUSE_VR_PREFILL_QKN", "0").strip() in (
        "1",
        "true",
        "yes",
        "on",
    )


def store_enabled() -> bool:
    return os.environ.get("MUSE_VR_PREFILL_QKN_STORE", "0").strip() in (
        "1",
        "true",
        "yes",
        "on",
    )


def q8_enabled() -> bool:
    return os.environ.get("MUSE_VR_PREFILL_QKN_Q8", "0").strip() in (
        "1",
        "true",
        "yes",
        "on",
    )


class _StoreSpec:
    """What the fused store needs besides the call's own operands (static for the layer)."""

    __slots__ = ("cos_sin", "num_q_heads", "num_kv_heads", "eps", "rows")

    def __init__(
        self, cos_sin, num_q_heads: int, num_kv_heads: int, eps: float, rows=None
    ):
        self.cos_sin = cos_sin
        self.num_q_heads = int(num_q_heads)
        self.num_kv_heads = int(num_kv_heads)
        self.eps = float(eps)
        # The QKV rows the q / k / v views were split from. Under prefill-graph capture this is a weak reference (the form
        # SGLang gives eager-region inputs): the eager region receives q / k / v rebuilt as weak references whose storage
        # covers each view only, so the row span cannot be recovered from q alone.
        self.rows = rows


def _store_view(backend, layer):
    """The backend attributes the stock call of ``layer`` runs with: for a layer whose KV lives in the FP8 sliding-window
    sub-pool of a mixed NVFP4 / FP8 hybrid pool (``nvfp4_kv_mixed``), the FP8-KV view that ``nvfp4_kv_mixed`` applies
    around ``forward_extend``; otherwise the backend as it is. The store decision and the fused store run under it, so an
    FP8 sub-pool layer is judged and written exactly as on an FP8 pool."""
    from . import nvfp4_kv_mixed as mixed

    sub = (
        mixed._fp8_sub_pool_of(
            getattr(backend, "token_to_kv_pool", None), layer.layer_id
        )
        if mixed.requested()
        else None
    )
    return (
        mixed._fp8_view(backend, sub) if sub is not None else contextlib.nullcontext()
    )


def _store_reason(backend, layer, forward_batch, save_kv_cache) -> str | None:
    """Reason the backend call must keep its own FP8 store, or None (the backend's fused FP8 q + KV path would run
    with unit KV scales: the case the fused kernel reproduces)."""
    from sglang.srt.layers.attention.trtllm_mha_backend import (
        TRTLLMHAAttnBackend,
        is_cp_active,
    )

    if type(backend) is not TRTLLMHAAttnBackend:
        return "backend"
    if not save_kv_cache:
        return "no kv save"
    if backend.data_type != torch.float8_e4m3fn or backend.is_xqa_impl:
        return "kv dtype / kernel"
    if layer.k_scale is not None or layer.v_scale is not None:
        return "kv scales"
    if is_cp_active(forward_batch):
        return "context parallel"
    if getattr(forward_batch.forward_mode, "name", "") != "EXTEND":
        return "not plain extend"
    return None


def _q8_reason(backend, layer, forward_batch) -> str | None:
    """Reason the attention call must receive the bf16 q (q8 mode), or None: the backend's extend call converts q with
    ``q.to(torch.float8_e4m3fn)`` on its native-NVFP4 path (not the fused FP8 q / KV path)."""
    from sglang.srt.layers.attention.trtllm_mha_backend import (
        TRTLLMHAAttnBackend,
        is_cp_active,
    )

    from . import nvfp4_kv_mixed as mixed

    if backend is None or type(backend) is not TRTLLMHAAttnBackend:
        return "backend"
    if getattr(forward_batch.forward_mode, "name", "") != "EXTEND":
        return "not plain extend"
    if is_cp_active(forward_batch):
        return "context parallel"
    if mixed.requested() and mixed.layer_is_fp8(backend, layer):
        return "fp8 layer"
    if (
        backend.data_type == torch.float8_e4m3fn
        or not backend._forward_extend_uses_native_fp4(forward_batch)
    ):
        return "not native fp4"
    return None


def _q8_stock(reason: str) -> None:
    _tally(f"q8:stock:{reason}")
    if "q8:" + reason not in _logged:
        _logged.add("q8:" + reason)
        logger.warning("muse_vr_plugin: %s q8 steps aside (%s); bf16 q", NAME, reason)


def _qkv_rows(q: torch.Tensor, n: int) -> torch.Tensor:
    """The ``[n, row stride]`` QKV rows that the column view ``q`` (the first columns of each row) was split from."""
    return torch.as_strided(q, (n, q.stride(0)), (q.stride(0), 1))


def _tally(key: str) -> None:
    """Count ``key``; at each power-of-two tally (>= 2) also leave a ``<key>>=<n>`` key (the engagement record is written
    on new keys only, so these keys show each counter's order of magnitude at the end of a run)."""
    bed.count(NAME, key)
    n = bed.counts().get(NAME, {}).get(key, 0)
    if n >= 2 and n & (n - 1) == 0:
        bed.count(NAME, f"{key}>={n}")


def _stock(original, args, kwargs, reason: str):
    _tally(f"stock:{reason}")
    if reason not in _logged:
        _logged.add(reason)
        logger.warning("muse_vr_plugin: %s steps aside (%s); stock path", NAME, reason)
    return original(*args, **kwargs)


def _triton_version() -> str | None:
    try:
        import triton

        return str(triton.__version__)
    except Exception:  # noqa: BLE001
        return None


def _reason(attn, positions, qkv, forward_batch):
    """Reason the call must run stock, or None (then also returns the RoPE table or None)."""
    import sglang.kernels.fused_op as fused_op
    from sglang.kernels.spec import KernelBackend

    from .ops import prefill_qkn as op

    if getattr(forward_batch.forward_mode, "name", "") != "EXTEND":
        return "not plain extend", None
    layer = attn.attn
    m = qkv.shape[0]
    if m == 0 or not op.covers(m, layer.tp_q_head_num, layer.tp_k_head_num):
        return "row count", None
    if layer.head_dim != op.HEAD_DIM:
        return "head dim", None
    if torch.compiler.is_compiling():
        return "torch.compile trace", None
    qkn = attn._fused_qk_norm
    if (
        qkn is None
        or not attn.use_o_gate
        or qkn.is_torch_compile
        or qkn._head_dim != op.HEAD_DIM
    ):
        return "model configuration", None
    if (
        fused_op.get_fused_op_backend() is not None
        or fused_op._oot_dispatch_key() is not None
    ):
        return "fused-op override", None
    if attn._fused_kv_store(forward_batch) is not None:
        return "fused kv store", None
    rope = attn.rotary_emb
    cos_sin = rope.cos_sin_cache if rope is not None else None
    cands = qkn._auto_backend_candidates()
    if not cands or cands[0] is not KernelBackend.TRITON:
        return "qk norm backend", None
    if not qkn.backend_eligible(
        KernelBackend.TRITON,
        qkv,
        positions if rope is not None else None,
        cos_sin,
        None,
    ):
        return "qk norm backend", None
    if not (
        qkv.dtype == torch.bfloat16
        and qkv.stride(1) == 1
        and qkv.stride(0) % 8 == 0
        and qkv.data_ptr() % 16 == 0
    ):
        return "qkv layout", None
    if rope is not None and not (
        positions.dtype == torch.int64
        and positions.is_contiguous()
        and cos_sin.dtype == torch.float32
        and cos_sin.is_contiguous()
        and cos_sin.shape[-1] == op.HEAD_DIM
        and cos_sin.data_ptr() % 16 == 0
    ):
        return "rope layout", None
    return None, cos_sin


def _prefill_path(attn, positions, qkv, forward_batch, cos_sin):
    """MuseSpark1xAttention.gated_attention (plain EXTEND, no fused KV store) with the QK norm on the plugin's kernel."""
    from .ops import prefill_qkn as op

    split_sizes = [attn.q_size, attn.kv_size, attn.kv_size]
    q_raw, k_raw, v, o_gate = qkv.split(split_sizes + [attn.q_size], dim=-1)
    layer = attn.attn
    eps = float(attn._fused_qk_norm._eps)
    mode = ":capture" if torch.cuda.is_current_stream_capturing() else ":eager"
    if _store_state["engaged"] and qkv.stride(1) == 1:
        backend = _backend()
        rows_ref = _rows_ref(qkv)
        ok = False
        if backend is not None and rows_ref is not None:
            with _store_view(backend, layer):
                ok = _store_reason(backend, layer, forward_batch, True) is None
        if ok:
            spec = _StoreSpec(
                cos_sin, layer.tp_q_head_num, layer.tp_k_head_num, eps, rows_ref
            )
            _tally(
                ("own:store:rope" if cos_sin is not None else "own:store:norope") + mode
            )
            output = attn.attn(
                q_raw, k_raw, v, forward_batch, save_kv_cache=True, **{STORE_KW: spec}
            )
            return attn._fused_o_norm_gate(output, o_gate)
    if _q8_state["engaged"]:
        try:
            q8_reason = _q8_reason(_backend(), layer, forward_batch)
        except Exception as e:  # noqa: BLE001 - this call keeps the bf16 q
            q8_reason = f"error {type(e).__name__}"
        if q8_reason is None:
            q8, k = op.qk_norm_q8(
                qkv, positions, cos_sin, layer.tp_q_head_num, layer.tp_k_head_num, eps
            )
            _tally(("own:q8:rope" if cos_sin is not None else "own:q8:norope") + mode)
            output = attn.attn(q8, k, v, forward_batch, save_kv_cache=True)
            return attn._fused_o_norm_gate(output, o_gate)
        _q8_stock(q8_reason)
    q, k = op.qk_norm(
        qkv, positions, cos_sin, layer.tp_q_head_num, layer.tp_k_head_num, eps
    )
    _tally(("own:rope" if cos_sin is not None else "own:norope") + mode)
    output = attn.attn(q, k, v, forward_batch, save_kv_cache=True)
    return attn._fused_o_norm_gate(output, o_gate)


def _rows_ref(qkv):
    """The QKV rows to keep in the spec: a weak reference while a CUDA graph is being captured (the captured segment's
    memory pool owns the buffer), else the tensor itself. None when a weak reference is needed but unavailable."""
    if not torch.cuda.is_current_stream_capturing():
        return qkv
    try:
        from sgl_kernel import weak_ref_tensor

        return weak_ref_tensor(qkv)
    except Exception:  # noqa: BLE001
        return None


def _backend():
    try:
        from sglang.srt.model_executor.forward_context import get_attn_backend

        return get_attn_backend()
    except Exception:  # noqa: BLE001
        return None


def _extend_hook(original, *args, **kwargs):
    """``TRTLLMHAAttnBackend.forward_extend``: consume the store spec (calls without it pass through unchanged)."""
    spec = kwargs.pop(STORE_KW, None)
    if spec is None:
        return original(*args, **kwargs)
    from .ops import prefill_qkn as op

    backend, q, k, v, layer, forward_batch = args[:6]
    save_kv_cache = kwargs.pop("save_kv_cache", args[6] if len(args) > 6 else True)
    n = q.shape[0]
    width = (spec.num_q_heads + 2 * spec.num_kv_heads) * op.HEAD_DIM
    src = spec.rows
    if (
        src is not None
        and src.dim() == 2
        and src.shape[0] >= n
        and src.data_ptr() == q.data_ptr()
        and src.stride(0) == q.stride(0)
        and src.stride(1) == 1
        and src.shape[1] >= width
    ):
        rows = src[:n]
    elif (
        q.dim() == 2
        and q.stride(1) == 1
        and q.stride(0) >= width
        and q.shape[1] == spec.num_q_heads * op.HEAD_DIM
        and q.untyped_storage().nbytes()
        >= (q.storage_offset() + (n - 1) * q.stride(0) + width) * q.element_size()
    ):
        rows = _qkv_rows(q, n)
    else:  # no row span to read in place: rebuild the rows the kernels read (q, k, v heads first)
        _tally("store:rebuilt_rows")
        rows = torch.cat([q.reshape(n, -1), k.reshape(n, -1), v.reshape(n, -1)], dim=1)
    positions = forward_batch.positions[:n]
    with _store_view(backend, layer):
        reason = _store_reason(backend, layer, forward_batch, save_kv_cache)
        if reason is None:
            cache_loc = backend._get_layer_cache_loc(layer, forward_batch)
            k_cache, v_cache = backend.token_to_kv_pool.get_kv_buffer(layer.layer_id)
            kv_dim = spec.num_kv_heads * op.HEAD_DIM
            q8 = torch.empty(
                (n, spec.num_q_heads * op.HEAD_DIM),
                dtype=torch.float8_e4m3fn,
                device=q.device,
            )
            op.prefill_qkn(
                rows,
                positions,
                spec.cos_sin,
                k_cache.view(-1, kv_dim),
                v_cache.view(-1, kv_dim),
                cache_loc,
                q8,
                spec.num_q_heads,
                spec.num_kv_heads,
                spec.eps,
            )
            _tally("store:fused")
    if reason is None:
        return original(
            backend, q8, None, None, layer, forward_batch, save_kv_cache=False, **kwargs
        )
    # The served norm-only form, then the stock call (the backend keeps its own store / q conversion).
    if reason not in _logged:
        _logged.add(reason)
        logger.warning(
            "muse_vr_plugin: %s store steps aside (%s); norm-only + stock store",
            NAME,
            reason,
        )
    _tally(f"store:stock:{reason}")
    qn, kn = op.qk_norm(
        rows, positions, spec.cos_sin, spec.num_q_heads, spec.num_kv_heads, spec.eps
    )
    return original(
        backend,
        qn,
        kn.view(n, spec.num_kv_heads, op.HEAD_DIM),
        v,
        layer,
        forward_batch,
        save_kv_cache=save_kv_cache,
        **kwargs,
    )


def hook(original, *args, **kwargs):
    if len(args) != 4 or kwargs:
        return original(*args, **kwargs)
    attn, positions, qkv, forward_batch = args
    try:
        reason, cos_sin = _reason(attn, positions, qkv, forward_batch)
    except Exception as e:  # noqa: BLE001 - leave the server on the stock path for this call
        reason, cos_sin = f"error {type(e).__name__}", None
    if reason is not None:
        return _stock(original, args, kwargs, reason)
    return _prefill_path(attn, positions, qkv, forward_batch, cos_sin)


def register() -> bool:
    if not enabled():
        return False
    pins_ok = all(bed.pinned(rel, sha) for rel, sha in PINS.items())
    from .ops.swa_verify_attn import qk_arith

    triton_ok = qk_arith() is not None
    if not (pins_ok and triton_ok):
        bed.report(
            NAME, False, pins=pins_ok, triton=_triton_version(), arith=qk_arith()
        )
        return False
    from sglang.srt.plugins.hook_registry import HookRegistry, HookType

    HookRegistry.register(TARGET, hook, HookType.AROUND)
    mode = "qk_norm"
    store = None
    if store_enabled():
        store_pins = all(bed.pinned(rel, sha) for rel, sha in STORE_PINS.items())
        if store_pins:
            HookRegistry.register(TARGET_EXTEND, _extend_hook, HookType.AROUND)
            _store_state["engaged"] = True
            mode = "qk_norm+store"
        else:
            store = "store pins mismatch: norm-only"
            logger.warning(
                "muse_vr_plugin: %s store mode not engaged (pin mismatch); norm-only",
                NAME,
            )
    q8 = None
    if q8_enabled():
        if all(bed.pinned(rel, sha) for rel, sha in Q8_PINS.items()):
            _q8_state["engaged"] = True
            mode += "+q8"
        else:
            q8 = "q8 pins mismatch: bf16 q"
            logger.warning(
                "muse_vr_plugin: %s q8 mode not engaged (pin mismatch); bf16 q", NAME
            )
    bed.report(
        NAME,
        True,
        triton=_triton_version(),
        arith=qk_arith(),
        mode=mode,
        **({"store": store} if store else {}),
        **({"q8": q8} if q8 else {}),
    )
    return True
