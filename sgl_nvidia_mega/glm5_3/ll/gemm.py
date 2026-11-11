# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Registered small-M BF16 GEMMs (P11): the kernels behind ``ll.linear``.

``linear(role, module, x)`` returns ``x @ module.weight.T`` for a role of
``gemm_plan`` (no bias, no tensor-parallel reduction: the caller's hop reduces): on the
PDL kernel the plan names for the row count when ``build`` prepared it there, else on
SGLang's own GEMM for that module, bit for bit ("stock").

``build(role, module, rows)`` runs at family build, outside CUDA-graph capture, with
every row count the caller will run the role at. It checks every planned tactic and
launches each served (row count, kernel) once, so capture never compiles; a kernel that
fails to build raises (every rank builds the same set). The fused step is first
recorded under capture, so a call at a row count the plan serves but nobody built
raises instead of compiling there. A role the plan does not serve at a shape or row
count (another GPU, a quantized weight, rows outside its ranges) runs stock; ``build``
logs why, once per rank 0.

Split-K source: the vendored copy of FlashInfer's kernel (``ll/kernels/cute/splitk.py``),
which carries synchronization fixes FlashInfer's body lacks, or FlashInfer's
``mm_bf16(..., backend="cute-dsl", tactic=...)`` once the installed FlashInfer has that
API (fi-1) and marks its kernel as carrying the same fixes (``SplitKTactic.<FI1_SYNC_MARKER>``).
The fixes change no arithmetic: both return bit for bit the same result.
``SGLANG_NVIDIA_MEGAKERNEL_GLM_GEMM_SPLITK`` = ``auto`` (default: as above),
``flashinfer`` (FlashInfer's kernel whenever it has the API, fixed or not; raise
without it) or ``vendored``. Each rank logs the resolved source once.

Each role is the unit (slot) ``ll.gemm.<role>`` of ``SGLANG_NVIDIA_MEGAKERNEL_SLOTS``
(``units.enabled``): unset serves the plan's preferred entries; ``-ll.gemm.<role>``
keeps that role on SGLang's GEMM; "" (parity) keeps every role there; a deferred entry
(``gemm_plan.Entry.preferred`` False) is served only where its unit is listed.
"""

from __future__ import annotations

import functools
import inspect
import logging
import os
from typing import Dict, Iterable, NamedTuple, Optional, Tuple

import torch

from sgl_nvidia_mega.glm5_3 import units
from sgl_nvidia_mega.glm5_3.ll import gemm_plan as plan
from sgl_nvidia_mega.glm5_3.ll import stock_linear
from sgl_nvidia_mega.glm5_3.runtime import SLOTS_ENV

# The SGLang names this module reads (test/unit/test_contract.py pins them)
SGLANG_READS = {
    # the TGV kernel the plan's "tgv" ranges run (SGLang's tactic pick, PDL on)
    "kernels/ops/gemm/cutedsl_bf16_gemm.py": ("cutedsl_bf16_gemm",),
    # MoEGate.forward keeps its tiny GEMM up to this many rows (stock: ll.stock_linear)
    "srt/models/deepseek_v2.py": ("MoEGate", "tiny_router_gemm_max_tokens"),
    # lm_head_logits serves the plain-matmul branch of _compute_lm_head only
    "srt/layers/logits_processor.py": (
        "LogitsProcessor",
        "_compute_lm_head",
        "should_apply_lm_head_quant_method",
        "use_fp32_lm_head",
        "rl_on_policy_target",
    ),
}

logger = logging.getLogger(__name__)

SPLITK_ENV = "SGLANG_NVIDIA_MEGAKERNEL_GLM_GEMM_SPLITK"
SPLITK_SOURCES = ("auto", "flashinfer", "vendored")
# fi-1's marker that FlashInfer's split-K body carries the synchronization fixes of the
# vendored copy (``flashinfer.gemm.SplitKTactic.SYNC_FIXED``): ``auto`` takes
# FlashInfer's kernel only when it is True
FI1_SYNC_MARKER = "SYNC_FIXED"

_DTYPES = {"bfloat16": torch.bfloat16, "float32": torch.float32}
# the named tolerances of these kernels (ll.tolerances() -> the spec's NUMERICS)
TOLERANCES = dict(plan.TOLERANCES)


class _Served(NamedTuple):
    kernel: str  # plan.SPLITK or plan.TGV
    tactic: object  # the split-K source's SplitKTactic, or None
    name: str  # plan.kernel_name [split-K source]
    source: object = None  # the _SplitKSource built for (split-K only)


# (role, N, K) -> {rows: served kernel}, filled by build
_READY: Dict[Tuple[str, int, int], Dict[int, _Served]] = {}
# (role, N, K) -> why build left it stock
_DECLINED: Dict[Tuple[str, int, int], str] = {}


# ---------------------------------------------------------------- split-K sources


def flashinfer_tactic_api():
    """``flashinfer.gemm`` when it runs the split-K kernel with an explicit tactic at
    any M and FP32 output (``mm_bf16(..., tactic=)``, FlashInfer's PR fi-1), else None."""
    try:
        import flashinfer.gemm as fi_gemm
    except Exception:  # noqa: BLE001 - no FlashInfer, or it fails to import
        return None
    mm_bf16 = getattr(fi_gemm, "mm_bf16", None)
    try:
        has_tactic = mm_bf16 is not None and (
            "tactic" in inspect.signature(mm_bf16).parameters
        )
    except (TypeError, ValueError):
        has_tactic = False
    return fi_gemm if has_tactic and hasattr(fi_gemm, "SplitKTactic") else None


class _SplitKSource(NamedTuple):
    name: str  # "flashinfer" or "vendored"
    tactic_type: type
    run: object  # (x, weight, out, tactic) -> None


def flashinfer_splitk_sync_fixed(fi_gemm) -> bool:
    """Whether fi-1's split-K kernel declares the synchronization fixes."""
    return bool(getattr(getattr(fi_gemm, "SplitKTactic", None), FI1_SYNC_MARKER, False))


def _flashinfer_source(fi_gemm) -> _SplitKSource:
    def run(x, weight, out, tactic):
        fi_gemm.mm_bf16(
            x,
            weight.t(),
            pdl=True,
            out=out,
            out_dtype=out.dtype,
            backend="cute-dsl",
            tactic=tactic,
        )

    return _SplitKSource("flashinfer", fi_gemm.SplitKTactic, run)


def _vendored_source() -> _SplitKSource:
    from sgl_nvidia_mega.glm5_3.ll.kernels.cute import splitk

    def run(x, weight, out, tactic):
        splitk.run_splitk_dense(x, weight.t(), None, out, True, tactic)

    return _SplitKSource("vendored", splitk.SplitKTactic, run)


@functools.lru_cache(maxsize=None)
def _splitk_source_for(choice: str) -> _SplitKSource:
    if choice not in SPLITK_SOURCES:
        raise ValueError(f"{SPLITK_ENV}={choice!r}; expected one of {SPLITK_SOURCES}")
    from sgl_nvidia_mega.glm5_3.sm103 import cutlass_supported

    ok, why = cutlass_supported()
    if not ok:
        raise RuntimeError(f"ll.gemm split-K: {why}")
    fi_gemm = flashinfer_tactic_api() if choice != "vendored" else None
    if choice == "flashinfer" and fi_gemm is None:
        raise RuntimeError(
            f"{SPLITK_ENV}=flashinfer, but the installed FlashInfer has no "
            "mm_bf16(..., tactic=SplitKTactic) API"
        )
    fixed = fi_gemm is not None and flashinfer_splitk_sync_fixed(fi_gemm)
    if fi_gemm is not None and (choice == "flashinfer" or fixed):
        source, why = _flashinfer_source(fi_gemm), "mm_bf16(..., tactic=)"
        why += "" if fixed else " (without the synchronization fixes)"
    else:
        source = _vendored_source()
        why = (
            f"{SPLITK_ENV}=vendored"
            if choice == "vendored"
            else "FlashInfer has no mm_bf16(..., tactic=)"
            if fi_gemm is None
            else f"FlashInfer's kernel lacks the synchronization fixes ({FI1_SYNC_MARKER})"
        )
    logger.info("ll.gemm split-K source: %s (%s)", source.name, why)
    return source


def splitk_source() -> _SplitKSource:
    """The split-K source the switch selects (resolved once per choice)."""
    return _splitk_source_for(os.environ.get(SPLITK_ENV, "") or "auto")


# ---------------------------------------------------------------- plan selection


def unit(role: str) -> str:
    """The unit name of a GEMM role (units.GEMM_UNITS)."""
    return f"ll.gemm.{role}"


def _unit_listed(role: str) -> bool:
    """Whether the slots switch names the role's unit explicitly (deferred entries)."""
    return units.listed(unit(role))


@functools.lru_cache(maxsize=None)
def _device_facts(index: int) -> Tuple[Tuple[int, int], int]:
    props = torch.cuda.get_device_properties(index)
    return (props.major, props.minor), props.multi_processor_count


def _min_rows(role: str, module) -> int:
    if role == "router":
        # MoEGate keeps its tiny GEMM up to this many rows (-1: never)
        return max(1, int(getattr(module, "tiny_router_gemm_max_tokens", 0)) + 1)
    return 1


def _planned(
    role: str, module, *, sm: Tuple[int, int], sms: int
) -> Tuple[Optional[plan.Entry], str]:
    """The plan entry serving ``role`` for ``module``'s weight, or (None, why not)."""
    if role not in plan.ROLE_DTYPES:
        raise ValueError(f"ll.gemm: unknown role {role!r}; known: {plan.ROLES}")
    if not units.enabled(unit(role)):
        return None, f"unit {unit(role)} is off (not in {SLOTS_ENV})"
    if (tuple(sm), sms) != plan.MEASURED_DEVICE:
        (major, minor), want_sms = plan.MEASURED_DEVICE
        return None, (
            f"the plan is measured on SM{major}{minor} with {want_sms} SMs; this GPU is "
            f"SM{sm[0]}{sm[1]} with {sms} SMs"
        )
    weight = module.weight
    if weight.dim() != 2 or weight.dtype != torch.bfloat16:
        return None, f"weight is {weight.dtype} {tuple(weight.shape)}, not 2-D BF16"
    if getattr(module, "bias", None) is not None:
        return None, "the module has a bias"
    e = plan.entry(role, tuple(weight.shape))
    if e is None:
        return None, f"no measured plan for (N={weight.shape[0]}, K={weight.shape[1]})"
    if not e.preferred and not _unit_listed(role):
        return None, f"deferred (list {unit(role)} in {SLOTS_ENV})"
    if not weight.is_contiguous() or weight.data_ptr() % 32:
        return None, "weight is strided or not 32-byte aligned"
    return e, ""


def _check_tactic(source, r: plan.Range, m: int, n: int, k: int, sms: int):
    tactic = source.tactic_type(*r.tactic)
    tactic.validate(m, n, k)  # ValueError for a tactic that cannot serve the problem
    if not plan.splitk_fits_one_wave(r.tactic, m, n, sms):
        raise ValueError(
            f"{plan.kernel_name(r)} needs {plan.splitk_ctas(r.tactic, m, n)} CTAs at "
            f"M={m}, N={n}: more than one wave of {sms} SMs"
        )
    return tactic


# ---------------------------------------------------------------- kernels


def _tgv(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    from sglang.kernels.ops.gemm.cutedsl_bf16_gemm import cutedsl_bf16_gemm

    # TGV hands the weight to from_dlpack, which refuses tensors that require grad
    return cutedsl_bf16_gemm(x, weight.detach())


def _run(
    served: _Served, x: torch.Tensor, weight: torch.Tensor, out_dtype
) -> torch.Tensor:
    if served.kernel == plan.TGV:
        return _tgv(x, weight)
    out = torch.empty((x.shape[0], weight.shape[0]), dtype=out_dtype, device=x.device)
    served.source.run(x, weight.detach(), out, served.tactic)
    return out


# SGLang's own GEMM for the module, before any bias add or tensor-parallel reduction:
# the router's MoEGate.forward (FP32 logits), the LM head's matmul, a linear's
# quant_method.apply, else F.linear (ll.stock_linear, shared with ll.linear's path
# without this module)
stock = stock_linear


def _cast(out: torch.Tensor, out_dtype: Optional[torch.dtype]) -> torch.Tensor:
    """``ll.linear``'s ``out_dtype``: the role's own dtype unless one is given."""
    return out if out_dtype is None or out.dtype == out_dtype else out.to(out_dtype)


# ---------------------------------------------------------------- public API


def build(
    role: str,
    module,
    rows: Iterable[int],
    *,
    sm: Optional[Tuple[int, int]] = None,
    sms: Optional[int] = None,
) -> str:
    """Prepare ``role`` for ``module``'s weight at every row count in ``rows``: check the
    planned tactics and launch each served (row count, kernel) once. ``sm`` / ``sms``
    default to the current GPU's. Returns the role's description line."""
    if torch.cuda.is_available() and torch.cuda.is_current_stream_capturing():
        raise RuntimeError(f"ll.gemm: building {role} inside CUDA-graph capture")
    if sm is None or sms is None:
        sm, sms = _device_facts(torch.cuda.current_device())
    weight = module.weight
    n, k = int(weight.shape[0]), int(weight.shape[1])
    key = (role, n, k)
    e, why = _planned(role, module, sm=sm, sms=sms)
    if e is None:
        if key not in _DECLINED and key not in _READY:
            _DECLINED[key] = why
            _log_rank0(f"ll.gemm: {role} (N={n}, K={k}) stock: {why}")
        return _describe_key(key)
    min_m = _min_rows(role, module)
    ready = _READY.setdefault(key, {})
    out_dtype = _DTYPES[plan.ROLE_DTYPES[role]]
    for m in sorted(set(int(r) for r in rows)):
        r = plan.range_at(e, m)
        if r is None or m < min_m or m in ready:
            continue
        if r.kernel == plan.SPLITK:
            source = splitk_source()
            tactic = _check_tactic(source, r, m, n, k, sms)
            served = _Served(
                plan.SPLITK, tactic, f"{plan.kernel_name(r)} [{source.name}]", source
            )
        else:
            if out_dtype != torch.bfloat16:
                raise ValueError(f"ll.gemm: {role}: TGV writes BF16 only")
            served = _Served(plan.TGV, None, plan.TGV)
        x = torch.zeros((m, k), dtype=torch.bfloat16, device=weight.device)
        try:
            _run(served, x, weight, out_dtype)
            torch.cuda.synchronize()
        except Exception as err:
            raise RuntimeError(
                f"ll.gemm: {role} (N={n}, K={k}) {served.name} at {m} rows failed to "
                f"build: {err}"
            ) from err
        ready[m] = served
    _DECLINED.pop(key, None)
    line = _describe_key(key)
    _log_rank0(f"ll.gemm: {line}")
    return line


def linear(
    role: str, module, x: torch.Tensor, *, out_dtype: Optional[torch.dtype] = None
) -> torch.Tensor:
    """``x @ module.weight.T`` (``x``: [M, K]) on the kernel ``build`` prepared for M
    rows, else SGLang's GEMM for ``module``. The router returns FP32, every other role
    BF16; ``out_dtype`` casts the result."""
    if role not in plan.ROLE_DTYPES:
        raise ValueError(f"ll.gemm: unknown role {role!r}; known: {plan.ROLES}")
    weight = module.weight
    m = int(x.shape[0])
    key = (role, int(weight.shape[0]), int(weight.shape[1]))
    served = _READY.get(key, {}).get(m)
    if served is None:
        if key not in _DECLINED and _would_serve(role, module, m):
            raise RuntimeError(
                f"ll.gemm: {role} (N={key[1]}, K={key[2]}) at {m} rows is in the plan "
                "but was not built; call ll.gemm.build with this row count at family "
                "build (outside capture)"
            )
        return _cast(stock(role, module, x), out_dtype)
    if not (
        x.dim() == 2
        and x.dtype == torch.bfloat16
        and x.shape[1] == key[2]
        and x.is_contiguous()
        and x.data_ptr() % 32 == 0
    ):
        raise ValueError(
            f"ll.gemm: {role} needs a contiguous, 32-byte aligned BF16 [M, {key[2]}] "
            f"input; got {x.dtype} {tuple(x.shape)} stride {x.stride()}"
        )
    return _cast(_run(served, x, weight, _DTYPES[plan.ROLE_DTYPES[role]]), out_dtype)


def built(role: str, module, m: int) -> bool:
    """Whether ``build`` prepared ``role`` for ``module``'s weight at ``m`` rows."""
    w = module.weight
    return int(m) in _READY.get((role, int(w.shape[0]), int(w.shape[1])), {})


def lm_head_logits(
    processor, hidden_states: torch.Tensor, lm_head, embedding_bias=None
) -> Optional[torch.Tensor]:
    """This rank's LM-head logits shard on the kernel ``build`` prepared for the row
    count (role ``lm_head``), or None where SGLang's ``LogitsProcessor._compute_lm_head``
    keeps its own GEMM: rows that were not built, a bias, LoRA, a quantized head, FP32
    logits, an RL on-policy target, a strided or non-BF16 input. Where it serves, SGLang's
    body is ``torch.matmul(hidden_states, lm_head.weight.T)`` (BF16 out), the result
    this returns within ``ll.gemm.bf16.rel``."""
    from sglang.srt.layers.logits_processor import should_apply_lm_head_quant_method

    weight = getattr(lm_head, "weight", None)
    if (
        embedding_bias is not None
        or weight is None
        or hidden_states.dim() != 2
        or not built("lm_head", lm_head, hidden_states.shape[0])
        or (hasattr(lm_head, "set_lora") and hasattr(lm_head, "apply_lora"))
        or should_apply_lm_head_quant_method(
            lm_head, getattr(lm_head, "quant_method", None)
        )
        or getattr(processor, "use_fp32_lm_head", False)
        or getattr(processor, "rl_on_policy_target", None) is not None
        or hidden_states.dtype != weight.dtype
        or not hidden_states.is_contiguous()
        or hidden_states.data_ptr() % 32
    ):
        return None
    return linear("lm_head", lm_head, hidden_states)


class LmHeadMixin:
    """Mixed in before SGLang's ``LogitsProcessor``: ``_compute_lm_head`` on the
    ``lm_head`` role where ``build`` prepared the row count (``lm_head_logits``), else
    SGLang's body."""

    def _compute_lm_head(self, hidden_states, lm_head, embedding_bias=None, **kwargs):
        logits = lm_head_logits(self, hidden_states, lm_head, embedding_bias)
        if logits is None:
            logits = super()._compute_lm_head(
                hidden_states, lm_head, embedding_bias, **kwargs
            )
        return logits


def _would_serve(role: str, module, m: int) -> bool:
    if not torch.cuda.is_available():
        return False
    sm, sms = _device_facts(module.weight.device.index or 0)
    e, _ = _planned(role, module, sm=sm, sms=sms)
    return (
        e is not None
        and m >= _min_rows(role, module)
        and plan.range_at(e, m) is not None
    )


def served_rows(role: str, n: int, k: int) -> Dict[int, str]:
    """Row count -> kernel name of what ``build`` prepared for (role, N, K)."""
    return {m: s.name for m, s in sorted(_READY.get((role, n, k), {}).items())}


def _describe_key(key: Tuple[str, int, int]) -> str:
    role, n, k = key
    head = f"{role} (N={n}, K={k})"
    if key in _DECLINED:
        return f"{head}: stock ({_DECLINED[key]})"
    by_kernel: Dict[str, list] = {}
    for m, served in sorted(_READY.get(key, {}).items()):
        by_kernel.setdefault(served.name, []).append(m)
    if not by_kernel:
        return f"{head}: stock (no planned row among the built rows)"
    kernels = ", ".join(
        f"M {plan.m_ranges(ms)} {name}" for name, ms in by_kernel.items()
    )
    return f"{head}: {kernels}"


def describe() -> Dict[str, str]:
    """Every built or declined (role, N, K) -> its served rows and kernels."""
    keys = sorted(set(_READY) | set(_DECLINED))
    return {f"{r} (N={n}, K={k})": _describe_key((r, n, k)) for r, n, k in keys}


def reset() -> None:
    """Forget every build (tests)."""
    _READY.clear()
    _DECLINED.clear()
    _splitk_source_for.cache_clear()


def _log_rank0(msg: str) -> None:
    dist = torch.distributed
    if dist.is_available() and dist.is_initialized() and dist.get_rank() != 0:
        return
    logger.info(msg)
