# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Component ``gemm_lt`` (env ``MUSE_VR_GEMM_LT=1``, default off): chosen bf16 GEMM shapes and row ranges through one
fixed cuBLASLt algorithm configuration each.

SGLang's bf16 GEMM dispatch (``_bf16_gemm_dispatch_impl``) sends a GEMM ``[M, K] x [N, K]^T`` that neither its split-K
table nor its TGV shape rule takes to cuBLAS (``torch.nn.functional.linear``), where the library picks its kernel by
shape. For each entry of ``MUSE_VR_GEMM_LT_RULES`` this component runs such a GEMM - bf16 operands and output, fp32
compute and scale, the same transposes - through the public cuBLASLt API (``cublasLtMatmul``) with the entry's fixed
algorithm configuration instead of the library's pick. Entries: ``N x K : lo-hi : c0/c1/.../c8``, separated by ``;``; the
nine values are the ``CUBLASLT_ALGO_CONFIG_*`` ID, TILE_ID, SPLITK_NUM, REDUCTION_SCHEME, CTA_SWIZZLING, CUSTOM_OPTION,
STAGES_ID, INNER_SHAPE_ID and CLUSTER_SHAPE_ID. Ranges of one shape must not overlap.

The default entries are tunables for the cuBLASLt version in ``LIB_VERSION`` (chosen with a per-configuration timing
sweep). Each one gives the same output bits as ``F.linear`` at every row count of its range (``tests/test_gemm_lt.py``
checks this on several input sets); an entry whose summation order differed from the library's own pick would fail it.

Every call it cannot take runs the stock path, counted with its reason: no entry for the shape and row count, the
split-K table or the TGV rule taking the call (the dispatch would not reach cuBLAS), a bias or addend, dtype / layout,
SGLang's batch-invariant mode, a configuration the library rejects for that row count or that needs a workspace
(``cublasLtMatmulAlgoCheck``; no workspace is ever passed), a (shape, row count) whose first-use check failed or that is
first met inside a CUDA graph capture, setup first needed inside a capture, or a library error. The first eager call of
each (shape, row count, configuration) also runs the stock GEMM and compares the two outputs bit for bit (counted
``shadow:match`` / ``shadow:MISMATCH``); a mismatch keeps that (shape, row count) on stock for the process, so the
equality does not rest on the library version alone (it also depends on the device and on PyTorch's BLAS settings).
Engagement needs the cuBLASLt version the entries were chosen with (``MUSE_VR_GEMM_LT_LIB_VERSION`` overrides; 0: any)
and sha256 pins on the SGLang files whose code it wraps. It stays off in a process that runs SM-partitioned streams
(``spdo_b4``).

With ``MUSE_VR_GEMM_LT_TGV_REF=1`` (default off) an entry also takes a call whose row count SGLang's TGV rule gives to the
TGV kernel (the rule as the process has it, i.e. after any gemm_pick override): there the first-use check compares against
SGLang's own TGV entry point on the same operands (the kernel the dispatch would run), and every call it does not take,
including a failed check, runs the stock dispatch, i.e. that same TGV kernel. Such entries need a configuration whose output
equals TGV's at every row of the range (tests/test_gemm_lt_tgvref.py; counted ``tgvref:<n>x<k>:m<m>`` beside ``lt:...``).

Mechanism: ``_bf16_gemm_dispatch_impl`` is replaced in its module by a wrapper (the module's linear methods and its
``bf16_gemm_dispatch`` op look it up there at call time). Per device one cuBLASLt handle and one matmul descriptor (bf16 A
/ B / C / D, ``CUBLAS_COMPUTE_32F`` compute and fp32 scale, read back at setup), created under a lock on the first eager
call. The library is the one the process already loaded for PyTorch.
"""

from __future__ import annotations

import ctypes
import inspect
import logging
import os
import threading

import torch

from . import bed

# The SGLang names this module reads (test/unit/test_contract.py pins them against the supported SGLang)
SGLANG_READS = {
    "srt/layers/logits_processor.py": ("should_apply_lm_head_quant_method",),
    "srt/batch_invariant_ops/__init__.py": ("is_batch_invariant_mode_enabled",),
    "srt/plugins/hook_registry.py": ("HookRegistry", "HookType"),
}


logger = logging.getLogger(__name__)

NAME = "gemm_lt"  # env MUSE_VR_GEMM_LT
LIB = "libcublasLt.so.13"
LIB_VERSION = 130801  # cublasLtGetVersion() the default entries were chosen with; MUSE_VR_GEMM_LT_LIB_VERSION overrides
PINS = {
    # _bf16_gemm_dispatch_impl (wrapped), the split-K table check and the TGV rule binding it consults
    "sglang/srt/layers/quantization/unquant.py": "7285d3e9e9fa881c5ecd97d093a8f59e11e69f26d5e3e7006925149f624a2f9c",
    # LogitsProcessor._compute_lm_head (hooked): the branch order that leads to its plain torch.matmul
    "sglang/srt/layers/logits_processor.py": "3558330bccf3ea0e46aabdaa0dee6ae7daf294c8e15b9a4e9744dab6883c4398",
}
LM_TARGET = "sglang.srt.layers.logits_processor.LogitsProcessor._compute_lm_head"
DEFAULT_LM_HEAD_RULES = ";".join(
    (
        "50512x8192:104-128:66/20/1/0/0/1/35/0/3",  # 4-way vocabulary shard
        "50512x8192:136-192:66/32/1/0/0/1/35/0/3",
        "25256x8192:104-160:66/32/1/0/0/1/35/0/3",  # 8-way vocabulary shard
    )
)
DEFAULT_RULES = ";".join(
    (
        "8192x4096:200-256:66/208/1/0/0/0/35/0/7",  # MoE unpooler
        "8192x4096:360-384:66/20/1/0/0/1/35/0/3",
        "8192x2048:200-224:66/208/1/0/0/0/35/0/7",  # 4-way attention output projection
        "8192x2048:360-384:66/20/1/0/0/1/35/0/3",
        "8192x1024:200-224:66/208/1/0/0/2/35/0/6",  # 8-way attention output projection
        "4608x8192:456-512:66/220/1/0/0/2/35/0/6",  # 4-way qkvg projection
        "8192x7168:296-384:66/30/1/0/0/0/35/0/5",  # 4-way draft MLP down projection
    )
)

# cuBLASLt enum values (cublasLt.h / cublas_api.h / library_types.h)
_COMPUTE_32F = 68
_R_32F, _R_16BF = 0, 14
_DESC_COMPUTE_TYPE, _DESC_SCALE_TYPE, _DESC_TRANSA, _DESC_TRANSB = 0, 1, 3, 4
_OP_N, _OP_T = 0, 1
_CFG_ATTRS = (
    0,
    1,
    2,
    3,
    4,
    5,
    6,
    7,
    8,
)  # ID .. CLUSTER_SHAPE_ID; 7 and 8 are uint16, the rest 32-bit
_CFG_U16 = (7, 8)
_SUCCESS = 0


class _Algo(ctypes.Structure):
    _fields_ = [("data", ctypes.c_uint64 * 8)]


class _Heur(ctypes.Structure):
    _fields_ = [
        ("algo", _Algo),
        ("ws", ctypes.c_size_t),
        ("state", ctypes.c_int),
        ("waves", ctypes.c_float),
        ("res", ctypes.c_int * 4),
    ]


_state: dict = {
    "engaged": False,
    "rules": (),
    "lm_rules": (),
    "lib": None,
    "version": None,
    "orig": None,
    "tgv_ref": False,
}
_dev: dict = {}  # device index -> {"handle", "desc", "algos": {cfg: algo}, "layouts": {(m, n, k, cfg): ...}}
_checked: dict = {}  # (device index, n, k, m, cfg) -> True (first-use output equal to stock) / False (not equal)
_lock = threading.Lock()
_ONE = ctypes.c_float(1.0)
_ZERO = ctypes.c_float(0.0)


def parse_rules(text: str) -> tuple:
    """``"8192x4096:200-256:66/208/1/0/0/0/35/0/7;..."`` -> ``((8192, 4096, 200, 256, (66, 208, ...)), ...)``."""
    rules = []
    for item in filter(None, (t.strip() for t in text.split(";"))):
        shape, rng, cfg = (t.strip() for t in item.split(":"))
        n, k = (int(v) for v in shape.lower().split("x"))
        lo, hi = (int(v) for v in rng.split("-"))
        vals = tuple(int(v) for v in cfg.split("/"))
        if len(vals) != len(_CFG_ATTRS) or any(v < 0 for v in vals):
            raise ValueError(
                f"configuration {cfg!r}: needs {len(_CFG_ATTRS)} non-negative values"
            )
        if lo < 1 or hi < lo or n < 1 or k < 1:
            raise ValueError(f"entry {item!r}")
        for rn, rk, rlo, rhi, _ in rules:
            if (rn, rk) == (n, k) and lo <= rhi and rlo <= hi:
                raise ValueError(
                    f"entry {item!r} overlaps another range of the same shape"
                )
        rules.append((n, k, lo, hi, vals))
    return tuple(rules)


def entry(m: int, n: int, k: int, rules=None):
    """The configuration for a GEMM of ``m`` rows and weight ``[n, k]`` (None: no entry)."""
    for rn, rk, lo, hi, cfg in _state["rules"] if rules is None else rules:
        if n == rn and k == rk and lo <= m <= hi:
            return cfg
    return None


def _lib():
    if _state["lib"] is None:
        _state["lib"] = ctypes.CDLL(LIB)
    return _state["lib"]


def _check(rc: int, what: str) -> None:
    if rc != _SUCCESS:
        raise RuntimeError(f"{what}: cuBLASLt status {rc}")


def _layout(rows: int, cols: int, ld: int):
    lay = ctypes.c_void_p()
    _check(
        _lib().cublasLtMatrixLayoutCreate(
            ctypes.byref(lay),
            ctypes.c_int(_R_16BF),
            ctypes.c_uint64(rows),
            ctypes.c_uint64(cols),
            ctypes.c_int64(ld),
        ),
        "layout",
    )
    return lay


def build_algo(handle, cfg: tuple) -> _Algo:
    lt = _lib()
    algo = _Algo()
    _check(
        lt.cublasLtMatmulAlgoInit(
            handle,
            ctypes.c_int(_COMPUTE_32F),
            ctypes.c_int(_R_32F),
            ctypes.c_int(_R_16BF),
            ctypes.c_int(_R_16BF),
            ctypes.c_int(_R_16BF),
            ctypes.c_int(_R_16BF),
            ctypes.c_int(cfg[0]),
            ctypes.byref(algo),
        ),
        "algo init",
    )
    for attr, val in zip(_CFG_ATTRS[1:], cfg[1:]):
        v = ctypes.c_uint16(val) if attr in _CFG_U16 else ctypes.c_uint32(val)
        _check(
            lt.cublasLtMatmulAlgoConfigSetAttribute(
                ctypes.byref(algo),
                ctypes.c_int(attr),
                ctypes.byref(v),
                ctypes.c_size_t(ctypes.sizeof(v)),
            ),
            f"algo attribute {attr}",
        )
    return algo


def check_precision(desc) -> None:
    """The descriptor must hold the served precision of a bf16 linear: fp32 compute type and fp32 scale type (bf16
    operands and output come from the matrix layouts); anything else is refused."""
    lt = _lib()
    for attr, want, what in (
        (_DESC_COMPUTE_TYPE, _COMPUTE_32F, "compute type"),
        (_DESC_SCALE_TYPE, _R_32F, "scale type"),
    ):
        v, n = ctypes.c_int32(-1), ctypes.c_size_t(0)
        _check(
            lt.cublasLtMatmulDescGetAttribute(
                desc,
                ctypes.c_int(attr),
                ctypes.byref(v),
                ctypes.c_size_t(4),
                ctypes.byref(n),
            ),
            what,
        )
        if v.value != want:
            raise RuntimeError(f"{what} {v.value} != {want}")


def _device_state(dev: torch.device):
    idx = dev.index if dev.index is not None else torch.cuda.current_device()
    st = _dev.get(idx)
    if st is not None:
        return st
    if torch.cuda.is_current_stream_capturing():
        raise RuntimeError("setup first needed inside a graph capture")
    with _lock, torch.cuda.device(idx):
        st = _dev.get(idx)
        if st is not None:
            return st
        lt = _lib()
        handle = ctypes.c_void_p()
        _check(lt.cublasLtCreate(ctypes.byref(handle)), "create")
        desc = ctypes.c_void_p()
        _check(
            lt.cublasLtMatmulDescCreate(
                ctypes.byref(desc), ctypes.c_int(_COMPUTE_32F), ctypes.c_int(_R_32F)
            ),
            "matmul desc",
        )
        ta, tb = ctypes.c_int(_OP_T), ctypes.c_int(_OP_N)
        _check(
            lt.cublasLtMatmulDescSetAttribute(
                desc, ctypes.c_int(_DESC_TRANSA), ctypes.byref(ta), ctypes.c_size_t(4)
            ),
            "transa",
        )
        _check(
            lt.cublasLtMatmulDescSetAttribute(
                desc, ctypes.c_int(_DESC_TRANSB), ctypes.byref(tb), ctypes.c_size_t(4)
            ),
            "transb",
        )
        check_precision(desc)
        st = {"handle": handle, "desc": desc, "algos": {}, "layouts": {}, "idx": idx}
        _dev[idx] = st
    return st


def _plan(st: dict, m: int, n: int, k: int, cfg: tuple):
    """(algo, A, B, C, ok) for ``out[m, n] = x[m, k] @ w[n, k]^T``: C[n, m] = op_T(A[k, n]) x B[k, m]
    in column-major views of the row-major weight, input and output."""
    key = (m, n, k, cfg)
    ent = st["layouts"].get(key)
    if ent is None:
        with _lock:
            ent = st["layouts"].get(key)
            if ent is None:
                algo = st["algos"].get(cfg)
                if algo is None:
                    algo = st["algos"][cfg] = build_algo(st["handle"], cfg)
                a, b, c = _layout(k, n, k), _layout(k, m, k), _layout(n, m, n)
                res = _Heur()
                rc = _lib().cublasLtMatmulAlgoCheck(
                    st["handle"],
                    st["desc"],
                    a,
                    b,
                    c,
                    c,
                    ctypes.byref(algo),
                    ctypes.byref(res),
                )
                ok = (
                    "ok"
                    if rc == _SUCCESS and res.state == _SUCCESS and res.ws == 0
                    else (
                        "needs a workspace"
                        if rc == _SUCCESS and res.state == _SUCCESS
                        else "rejected"
                    )
                )
                ent = (algo, a, b, c, ok)
                st["layouts"][key] = ent
    return ent


def lt_linear(x: torch.Tensor, weight: torch.Tensor, cfg: tuple):
    """``F.linear(x, weight)`` for a 2-D bf16 ``x`` through configuration ``cfg``: (result, None) or (None, reason)."""
    try:
        st = _device_state(x.device)
    except RuntimeError as e:
        return None, f"setup refused ({e})"
    m, k = x.shape
    n = weight.shape[0]
    algo, a, b, c, ok = _plan(st, m, n, k, cfg)
    if ok != "ok":
        return None, f"configuration {ok} for this row count"
    stream = torch.cuda.current_stream(x.device)
    out = torch.empty((m, n), dtype=torch.bfloat16, device=x.device)
    rc = _lib().cublasLtMatmul(
        st["handle"],
        st["desc"],
        ctypes.byref(_ONE),
        ctypes.c_void_p(weight.data_ptr()),
        a,
        ctypes.c_void_p(x.data_ptr()),
        b,
        ctypes.byref(_ZERO),
        ctypes.c_void_p(out.data_ptr()),
        c,
        ctypes.c_void_p(out.data_ptr()),
        c,
        ctypes.byref(algo),
        ctypes.c_void_p(None),
        ctypes.c_size_t(0),
        ctypes.c_void_p(stream.cuda_stream),
    )
    if rc != _SUCCESS:
        return None, f"library status {rc}"
    return out, None


def _batch_invariant() -> bool:
    try:
        from sglang.srt.batch_invariant_ops import is_batch_invariant_mode_enabled
    except ImportError:
        return False
    return bool(is_batch_invariant_mode_enabled())


def checked_linear(x: torch.Tensor, weight: torch.Tensor, cfg: tuple, stock):
    """``lt_linear`` behind the first-use check: (result, None) or (None, reason). ``stock()`` computes the stock output."""
    idx = x.device.index if x.device.index is not None else torch.cuda.current_device()
    key = (idx, weight.shape[0], weight.shape[1], x.shape[0], cfg)
    state = _checked.get(key)
    if state is False:
        return None, "first-use check failed"
    if state is None and torch.cuda.is_current_stream_capturing():
        return None, "not yet checked when first met inside a graph capture"
    out, why = lt_linear(x, weight, cfg)
    if out is None or state is True:
        return out, why
    ref = stock()
    ok = bool(torch.equal(out.view(torch.int16), ref.view(torch.int16)))
    _checked[key] = ok
    bed.shadow_result(NAME, ok)
    if not ok:
        logger.warning(
            "muse_vr_plugin: gemm_lt %dx%d at %d rows differs from the stock GEMM; that call stays stock",
            weight.shape[0],
            weight.shape[1],
            x.shape[0],
        )
        return None, "first-use check failed"
    return out, None


def take(U, x: torch.Tensor, weight: torch.Tensor, bias=None, addend=None):
    """The component's output for a dispatch call (None: the stock dispatch runs it). ``U`` is SGLang's module."""
    if x.dim() < 1 or weight.dim() != 2:
        return None
    m, (n, k) = x.numel() // max(1, x.shape[-1]), weight.shape
    cfg = entry(m, n, k)
    if cfg is None:
        return None
    why, tgv = None, False
    if bias is not None or addend is not None:
        why = "bias / addend"
    elif x.dtype != torch.bfloat16 or weight.dtype != torch.bfloat16:
        why = "dtype"
    elif _batch_invariant():
        why = "batch-invariant mode"
    elif getattr(U, "_enable_bf16_splitk_gemm", False) and U.use_bf16_splitk_gemm(
        m, n, k
    ):
        why = "split-K table takes it"
    elif getattr(
        U, "_use_hopper_bf16_gemv", None
    ) is not None and U._use_hopper_bf16_gemv(m, n, k):
        why = "gemv takes it"
    elif getattr(
        U, "_use_cutedsl_bf16_gemm", None
    ) is not None and U._use_cutedsl_bf16_gemm(m, n, k):
        if _state["tgv_ref"] and getattr(U, "_cutedsl_bf16_gemm", None) is not None:
            tgv = True  # first-use reference = the TGV kernel the dispatch would run (it also runs every declined call)
        else:
            why = "TGV rule takes it"
    if why is None and not (
        x.is_cuda and x.is_contiguous() and weight.is_contiguous() and x.shape[-1] == k
    ):
        why = "layout"
    if why is None:
        x2 = x.view(m, k)
        if tgv:
            stock = lambda: U._cutedsl_bf16_gemm(x2, weight, None)  # noqa: E731
        else:
            stock = lambda: torch.nn.functional.linear(x2, weight)  # noqa: E731
        out, why = checked_linear(x2, weight, cfg, stock)
        if out is not None:
            bed.count(NAME, f"lt:{n}x{k}:m{m}")
            if tgv:
                bed.count(NAME, f"tgvref:{n}x{k}:m{m}")
            return out.view(*x.shape[:-1], n)
    bed.count(NAME, f"stock:{n}x{k}:{why}" + (" (TGV reference)" if tgv else ""))
    return None


def _plain_lm_head_path(proc, lm_head, embedding_bias) -> bool:
    """Whether the stock ``_compute_lm_head`` would run its plain ``torch.matmul`` for this call."""
    import sglang.srt.layers.logits_processor as LP

    if hasattr(lm_head, "set_lora") and hasattr(lm_head, "apply_lora"):
        return False
    if LP.should_apply_lm_head_quant_method(
        lm_head, getattr(lm_head, "quant_method", None)
    ):
        return False
    if not hasattr(lm_head, "weight") or getattr(proc, "use_fp32_lm_head", False):
        return False
    if (
        getattr(lm_head, "use_intel_amx_backend", False)
        or getattr(proc, "rl_on_policy_target", None) is not None
    ):
        return False
    return embedding_bias is None


def _around_lm_head(
    original, proc, hidden_states, lm_head, embedding_bias=None, *args, **kwargs
):
    if (
        not args
        and not kwargs
        and _state["lm_rules"]
        and hidden_states.dim() == 2
        and _plain_lm_head_path(proc, lm_head, embedding_bias)
    ):
        w = lm_head.weight
        m, k = hidden_states.shape
        cfg = (
            entry(m, w.shape[0], w.shape[1], _state["lm_rules"])
            if w.dim() == 2
            else None
        )
        if cfg is not None:
            why = None
            if hidden_states.dtype != torch.bfloat16 or w.dtype != torch.bfloat16:
                why = "dtype"
            elif _batch_invariant():
                why = "batch-invariant mode"
            elif not (
                hidden_states.is_cuda
                and hidden_states.is_contiguous()
                and w.is_contiguous()
                and w.shape[1] == k
            ):
                why = "layout"
            if why is None:
                out, why = checked_linear(
                    hidden_states, w, cfg, lambda: torch.matmul(hidden_states, w.T)
                )
                if out is not None:
                    bed.count(NAME, f"lt_lm_head:{w.shape[0]}x{k}:m{m}")
                    return out
            bed.count(NAME, f"stock_lm_head:{why}")
    return original(proc, hidden_states, lm_head, embedding_bias, *args, **kwargs)


def install() -> None:
    import sglang.srt.layers.quantization.unquant as U

    orig = U._bf16_gemm_dispatch_impl
    has_addend = "addend" in inspect.signature(orig).parameters

    def _bf16_gemm_dispatch_impl(x, weight, bias, addend=None):
        out = take(U, x, weight, bias, addend)
        if out is not None:
            return out
        if has_addend:
            return orig(x, weight, bias, addend=addend)
        if addend is not None:
            raise TypeError("addend on a dispatch without one")
        return orig(x, weight, bias)

    U._bf16_gemm_dispatch_impl = _bf16_gemm_dispatch_impl
    _state["orig"] = orig


def library_version() -> int:
    lt = _lib()
    lt.cublasLtGetVersion.restype = ctypes.c_size_t
    return int(lt.cublasLtGetVersion())


def configure(rules=None, lm_rules=None, tgv_ref=None) -> None:
    """Set the entries (tests, harness); register() reads them from the environment."""
    if rules is not None:
        _state["rules"] = rules
    if lm_rules is not None:
        _state["lm_rules"] = lm_rules
    if tgv_ref is not None:
        _state["tgv_ref"] = bool(tgv_ref)


def register() -> bool:
    if not bed.enabled(NAME):
        return False
    if bed.enabled("spdo_b4"):
        # SM-partitioned (green-context) streams: the library may pick other kernels there, and the first-use check is keyed
        # by device, not by partition, so the component stays off in such a process.
        bed.report(NAME, False, reason="spdo_b4 enabled (SM-partitioned streams)")
        return False
    for rel, sha in PINS.items():
        if not bed.pinned(rel, sha):
            bed.report(NAME, False, reason=f"pin mismatch: {rel}")
            return False
    try:
        rules = parse_rules(os.environ.get("MUSE_VR_GEMM_LT_RULES", DEFAULT_RULES))
        lm_rules = parse_rules(
            os.environ.get("MUSE_VR_GEMM_LT_LM_HEAD_RULES", DEFAULT_LM_HEAD_RULES)
        )
        want = int(os.environ.get("MUSE_VR_GEMM_LT_LIB_VERSION", str(LIB_VERSION)))
    except ValueError as e:
        bed.report(NAME, False, reason=f"bad rules ({e})")
        return False
    try:
        ver = library_version()
    except OSError as e:
        bed.report(NAME, False, reason=f"library not loadable ({e})")
        return False
    if want and ver != want:
        bed.report(NAME, False, reason=f"cuBLASLt version {ver} != {want}")
        return False
    try:
        if lm_rules:
            from sglang.srt.plugins.hook_registry import HookRegistry, HookType

            HookRegistry.register(LM_TARGET, _around_lm_head, HookType.AROUND)
        install()
    except Exception as e:  # noqa: BLE001
        if _state["orig"] is not None:  # leave SGLang's dispatch as it was
            import sglang.srt.layers.quantization.unquant as U

            U._bf16_gemm_dispatch_impl = _state["orig"]
        bed.report(NAME, False, reason=f"install failed: {type(e).__name__}: {e}")
        return False
    tgv_ref = os.environ.get("MUSE_VR_GEMM_LT_TGV_REF", "0").strip().lower() in (
        "1",
        "true",
        "yes",
        "on",
    )
    _state.update(
        engaged=True, rules=rules, lm_rules=lm_rules, version=ver, tgv_ref=tgv_ref
    )
    fmt = lambda rs: ";".join(
        f"{n}x{k}:{lo}-{hi}:{'/'.join(map(str, c))}" for n, k, lo, hi, c in rs
    )  # noqa: E731
    bed.report(
        NAME,
        True,
        rules=fmt(rules),
        lm_head_rules=fmt(lm_rules),
        lib_version=ver,
        compute="CUBLAS_COMPUTE_32F",
        scale="CUDA_R_32F",
        matrix_types="CUDA_R_16BF",
        tgv_ref=tgv_ref,
    )
    return True
