# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# Adapted from SGLang (Apache-2.0, SGLang Team): the source's
# python/sglang/srt/layers/hyperconnection.py (GatedResidual.mix,
# combine, combine_and_normalize, combine_normalize_and_gate and their admission, and
# the HC gate stream's priority).
"""The HC read and write-back of 33..N rows, which ``runtime/hc.py`` (1..32 rows)
does not serve: verify and draft-extend steps of more than 8 requests, and prefill.

Plain functions on a layer's ``GatedResidual`` (SGLang main's class) and tensors. The
source's environment knobs are the fields of ``Limits``; ``DEFAULT`` is
its high-concurrency tuning and ``MID_CONCURRENCY`` its mid-concurrency one. They
assume TP1 (the executor's decomposed layers), streams of 4 x 2560 BF16 values with a
per-branch norm, and a successful ``build`` (an SM10x GPU).

Kernels, by rows (each cap is the smaller of the ``Limits`` field and the kernel's):

- read, ``mix``: ``hc_mix_wide`` at 33..``mix_wide_max_rows`` (<= 512): the CuTe
  split-K silu down GEMM and a Triton up GEMM with the gated mean fused. Not bitwise
  to main's mix (the torch.compile chain): the down GEMM sums K in split-K order.
  Other rows run main's ``GatedResidual.mix``.
- read, side stream: ``hc_combine_gate`` at 33..``fused_combine_max_rows`` (<= 128)
  when the read carries no partials: the write-back's [rows, 8, 4] FP32 gate partials.
- write-back, ``write``:

  - ``hc_combine_extend`` at ``extend_min_rows`` (>= 33) and above, with no partials
    or the [rows, 4] FP32 gate totals of the previous extend write-back or boundary:
    the combine, the next read's per-branch norm and its gate totals in one kernel.
    Bitwise to main's write-back (``hc_combine``), next norm (``grouped_gemma_rmsnorm``)
    and next gate dot; its source builds only for Blackwell.
  - ``hc_combine_apply_norm_wide`` at 33..``fused_combine_max_rows`` with the side
    stream's partials and a next norm (``hc_combine_apply`` without one). Bitwise to
    main's ``hc_combine`` and ``grouped_gemma_rmsnorm``.
  - otherwise main's ``GatedResidual.combine`` (the next read normalizes).

- ``boundary``: ``hc_boundary_ks`` at 33..``boundary_ks_max_rows`` (<= 320, needs the
  extend boundary and rank 320), a write-back that carries extend gate totals and the
  next read's mix in one launch. The combined streams are ``hc_combine_extend``'s
  bits; the normalized streams, next gate totals and mixed input are not bitwise to
  the separate kernels (rounding points and K order differ).

A layer's chain: ``mixed, normed, partials = mix(hc, streams, normalized, partials)``
with what the previous write-back handed over. Where ``boundary_reason`` is '' the
write-back is ``combined, normalized, next_partials, next_mixed = boundary(...)`` and
the next read takes ``next_mixed`` in place of calling ``mix`` (``normalized`` is an
empty tensor when only the next write-back's totals read it); otherwise
``combined, normalized, next_partials = write(...)``.
"""

from __future__ import annotations

import dataclasses
import importlib
import inspect
from functools import lru_cache
from typing import Optional

import torch

from sgl_nvidia_mega.qwen3_8.runtime import bf16_gemm_pins

STOCK = "stock"
# The first row count above runtime/hc.py's kernels
FLOOR_ROWS = 33


@dataclasses.dataclass(frozen=True)
class Limits:
    """The source's HC knobs as values; 0 turns a kernel off."""

    # SGLANG_HC_MIX_SPLITK_MAX_ROWS: the widest read on hc_mix_wide
    mix_wide_max_rows: int = 0
    # SGLANG_OPT_HC_EXTEND_BOUNDARY and SGLANG_X_HC_EXTEND_MIN_ROWS: the narrowest
    # write-back on hc_combine_extend (at least 33)
    extend_min_rows: int = 0
    # SGLANG_HC_BOUNDARY_KS_MAX_ROWS: the widest one-launch boundary
    boundary_ks_max_rows: int = 0
    # SGLANG_HC_BOUNDARY_KS_FLAGS: hc_boundary_ks.FLAG_* bits
    boundary_ks_flags: int = 3
    # SGLANG_HC_FUSED_COMBINE_MAX_ROWS: the widest side-stream gate + wide combine-norm
    fused_combine_max_rows: int = 0
    # SGLANG_FORWARD_HELPER_STREAM_PRIORITY: the side stream's priority (None: torch's)
    stream_priority: Optional[int] = None


# The high-concurrency tuning
HIGH_CONCURRENCY = Limits(
    mix_wide_max_rows=512,
    extend_min_rows=33,
    boundary_ks_max_rows=320,
    stream_priority=-1,
)
# The mid-concurrency tuning
MID_CONCURRENCY = Limits(extend_min_rows=33, fused_combine_max_rows=128)
DEFAULT = HIGH_CONCURRENCY

SGLANG_READS = {
    "srt/layers/hyperconnection.py": (
        "hc_norm",
        "input_mix_weight_down",
        "input_mix_weight_up",
        "block_inject_weight",
        "hc_count",
        "hidden_size",
        "params_dtype",
        "group_size",
        "variance_epsilon",
        "_mix_up_weight_padded",
        "mix(normalized_input)",
        "combine(residuals)",
    ),
    "kernels/ops/gemm/hc_mix_splitk.py": ("permute_pad_up_weight",),
    "kernels/ops/gemm/dense_bf16_gemm_sm100_splitk_epilogue.py": ("validate_tactic",),
}


def _mix_cap(limits: Limits) -> int:
    from sgl_nvidia_mega.qwen3_8.kernels.ops.gemm.hc_mix_wide import WIDE_MAX_ROWS

    return min(limits.mix_wide_max_rows, WIDE_MAX_ROWS)


def _extend_min(limits: Limits) -> int:
    return max(FLOOR_ROWS, limits.extend_min_rows) if limits.extend_min_rows else 0


def _fused_cap(limits: Limits) -> int:
    from sgl_nvidia_mega.qwen3_8.kernels.ops.elementwise.hc_combine_decode_wide import (
        WIDE_MAX_ROWS,
    )

    cap = min(limits.fused_combine_max_rows, WIDE_MAX_ROWS)
    return cap if cap >= FLOOR_ROWS else 0


def _ks_cap(limits: Limits) -> int:
    from sgl_nvidia_mega.qwen3_8.kernels.ops.gemm.hc_boundary_ks import MAX_ROWS

    if not limits.extend_min_rows:
        return 0
    cap = min(limits.boundary_ks_max_rows, MAX_ROWS)
    return cap if cap >= FLOOR_ROWS else 0


def _per_branch(hc) -> bool:
    return hc.hc_norm.group_size == hc.hidden_size


def _mix_ok(hc) -> bool:
    """hc_mix_wide's shapes: its up GEMM tiles 32 hidden and 64 low-rank columns."""
    if not hasattr(hc, "input_mix_weight_down"):
        return False
    lowrank = hc.input_mix_weight_down.weight.shape[0]
    return (
        hc.hc_count == 4
        and (hc.hc_count * hc.hidden_size) % 2048 == 0
        and hc.hidden_size % 32 == 0
        and lowrank > 0
        and lowrank % 64 == 0
    )


def _extend_ok(hc) -> bool:
    return (
        hasattr(hc, "block_inject_weight")
        and _per_branch(hc)
        and hc.hidden_size % 512 == 0
        and (hc.hc_count * hc.hidden_size) % 2048 == 0
        and hc.hc_count <= 8
        and hc.params_dtype in (torch.bfloat16, torch.float16)
    )


def _decode_ok(hc) -> bool:
    return (
        hasattr(hc, "block_inject_weight")
        and hc.hc_count == 4
        and hc.hidden_size == 2560
        and _per_branch(hc)
        and hc.params_dtype == torch.bfloat16
    )


def _ks_ok(hc) -> bool:
    """One side of the one-launch boundary: a write-back needs the extend boundary,
    the next read the mix weights (rank 320)."""
    return (
        _mix_ok(hc)
        and hc.hidden_size == 2560
        and hc.input_mix_weight_down.weight.shape[0] == 320
        and _per_branch(hc)
        and hc.params_dtype == torch.bfloat16
        and (_extend_ok(hc) or not hasattr(hc, "block_inject_weight"))
    )


def routes(hc, rows: int, *, limits: Limits = DEFAULT) -> tuple:
    """The wide kernels this module may run at ``rows``, from shapes and limits."""
    if rows < FLOOR_ROWS:
        return ()
    found = []
    if rows <= _mix_cap(limits) and _mix_ok(hc):
        found.append("hc_mix_wide")
    if _extend_min(limits) and rows >= _extend_min(limits) and _extend_ok(hc):
        found.append("hc_combine_extend")
    if rows <= _fused_cap(limits) and _decode_ok(hc):
        found.append("hc_combine_apply_norm_wide")
    if rows <= _ks_cap(limits) and _ks_ok(hc):
        found.append("hc_boundary_ks")
    return tuple(found)


def supported(hc, rows: int, *, limits: Limits = DEFAULT) -> str:
    """'' when a wide kernel serves this module's read or write-back at ``rows``."""
    if rows < FLOOR_ROWS:
        return f"{rows} rows: runtime/hc.py serves 1..{FLOOR_ROWS - 1}"
    if not _per_branch(hc):
        return "HC norm is not a per-branch norm"
    if routes(hc, rows, limits=limits):
        return ""
    return (
        f"no wide HC kernel takes HC {hc.hc_count} x {hc.hidden_size} "
        f"{hc.params_dtype} at {rows} rows ({limits})"
    )


def build(
    *,
    limits: Limits = DEFAULT,
    hc_count: int = 4,
    hidden_size: int = 2560,
    dtype: torch.dtype = torch.bfloat16,
) -> str:
    """Compile (or load) the enabled kernels and allocate their scratch and side
    stream on the current device, before capture; '' or the failures. hc_mix_wide's
    CuTe and Triton kernels compile at their first (eager, pre-capture) call."""
    if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 10:
        return "the wide HC kernels need an SM10x GPU"
    failures = []

    def attempt(name, fn):
        try:
            fn()
        except Exception as e:  # a failed kernel keeps the stock HC
            failures.append(f"{name}: {type(e).__name__}: {e}")

    def mix_kernels():
        from sglang.kernels.ops.gemm.dense_bf16_gemm_sm100_splitk_epilogue import (
            validate_tactic,
        )

        if "max_m" not in inspect.signature(validate_tactic).parameters:
            raise RuntimeError(
                "SGLang's split-K epilogue GEMM takes no max_m (sgl-project/sglang#50194)"
            )
        importlib.import_module(
            "sgl_nvidia_mega.qwen3_8.kernels.ops.gemm.hc_mix_up_gate"
        )

    def extend():
        from sgl_nvidia_mega.qwen3_8.kernels.ops.elementwise.hc_combine_extend import (
            warmup,
        )

        warmup(hc_count, hidden_size, dtype)

    def ks():
        from sgl_nvidia_mega.qwen3_8.kernels.ops.gemm.hc_boundary_ks import warmup

        warmup()

    def decode():
        from sgl_nvidia_mega.qwen3_8.kernels.ops.elementwise.hc_combine_decode_wide import (
            _jit_module,
        )

        _jit_module()
        gate_stream(torch.device("cuda"), limits.stream_priority)

    if _mix_cap(limits) >= FLOOR_ROWS:
        attempt("hc_mix_wide", mix_kernels)
    if _extend_min(limits):
        attempt("hc_combine_extend", extend)
    if _ks_cap(limits):
        attempt("hc_boundary_ks", ks)
    if _fused_cap(limits):
        attempt("hc_combine_decode_wide", decode)
    return "; ".join(failures)


@lru_cache(None)
def _stream(index: int, priority: Optional[int]) -> torch.cuda.Stream:
    kwargs = {} if priority is None else {"priority": priority}
    return torch.cuda.Stream(device=index, **kwargs)


def gate_stream(device: torch.device, priority: Optional[int]) -> torch.cuda.Stream:
    """The side stream of the gate partials: one per device and priority, created
    before capture (SGLang's HC gate stream, with the helper-stream priority)."""
    index = device.index if device.index is not None else torch.cuda.current_device()
    return _stream(index, priority)


def _beside(fn, normed: torch.Tensor, limits: Limits):
    stream = gate_stream(normed.device, limits.stream_priority)
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        out = fn()
    normed.record_stream(stream)
    return out


def _join(partials: torch.Tensor, limits: Limits) -> None:
    current = torch.cuda.current_stream()
    current.wait_stream(gate_stream(partials.device, limits.stream_priority))
    partials.record_stream(current)


def padded_up_weight(hc) -> torch.Tensor:
    """The up weight in hc_mix_splitk's permuted, padded layout, cached on the module
    as SGLang's own split-K mix caches it."""
    if hc._mix_up_weight_padded is None:
        from sglang.kernels.ops.gemm.hc_mix_splitk import permute_pad_up_weight

        hc._mix_up_weight_padded = permute_pad_up_weight(
            hc.input_mix_weight_up.weight, hc.hc_count
        )
    return hc._mix_up_weight_padded


def mix_route(hc, normed: torch.Tensor, *, limits: Limits = DEFAULT) -> str:
    rows = normed.shape[0]
    if (
        FLOOR_ROWS <= rows <= _mix_cap(limits)
        and _mix_ok(hc)
        and normed.is_cuda
        and normed.dtype in (torch.bfloat16, torch.float16)
        and normed.is_contiguous()
        and not torch.compiler.is_compiling()
    ):
        return "hc_mix_wide"
    return STOCK


def _gates_beside(hc, normed, partials, limits: Limits) -> bool:
    return (
        partials is None
        and FLOOR_ROWS <= normed.shape[0] <= _fused_cap(limits)
        and _decode_ok(hc)
        and normed.dtype == torch.bfloat16
        and not torch.compiler.is_compiling()
    )


def mix(
    hc,
    streams: torch.Tensor,
    normalized: Optional[torch.Tensor] = None,
    partials: Optional[torch.Tensor] = None,
    *,
    limits: Limits = DEFAULT,
) -> tuple:
    """The read of ``streams`` [rows, hc * hidden]: ``(mixed, normed, partials)``.

    ``normalized`` and ``partials`` are what the previous write-back handed over; the
    norm runs here without them. ``partials`` passes through; without them, at
    33..``fused_combine_max_rows`` rows the write-back's gate partials are computed
    here on the side stream (``write`` joins it)."""
    if not _per_branch(hc):
        mixed, (_, normed, own) = bf16_gemm_pins.gated_mix(hc, streams, normalized)
        return mixed, normed, partials if partials is not None else own
    normed = normalized if normalized is not None else hc.hc_norm(streams)
    if _gates_beside(hc, normed, partials, limits):
        from sgl_nvidia_mega.qwen3_8.kernels.ops.elementwise.hc_combine_decode_wide import (
            hc_combine_gate,
        )

        weight = hc.block_inject_weight.weight
        partials = _beside(lambda: hc_combine_gate(normed, weight), normed, limits)
    if mix_route(hc, normed, limits=limits) == "hc_mix_wide":
        from sgl_nvidia_mega.qwen3_8.kernels.ops.gemm.hc_mix_wide import hc_mix_wide

        mixed = hc_mix_wide(
            normed,
            hc.input_mix_weight_down.weight.data,
            padded_up_weight(hc),
            hc.hc_count,
            hc.hidden_size,
        ).to(hc.params_dtype)
        return mixed, normed, partials
    mixed, (_, _, own) = bf16_gemm_pins.gated_mix(hc, streams, normed)
    # SGLang's mix makes partials (on its own side stream) only for one row
    return mixed, normed, partials if partials is not None else own


def _extend_rows(hc, block, residual, normed, partials, limits: Limits) -> bool:
    """Whether hc_combine_extend serves this write-back. Its partials are [rows, hc]
    FP32 gate dot totals, made only by itself or the one-launch boundary."""
    rows = block.shape[0]
    first = _extend_min(limits)
    if not (first and rows >= first and _extend_ok(hc)):
        return False
    dtype = block.dtype
    if partials is None:
        partials_ok = (
            normed is not None
            and normed.dtype == dtype
            and normed.shape == residual.shape
        )
    else:
        partials_ok = partials.dtype == torch.float32 and tuple(partials.shape) == (
            rows,
            hc.hc_count,
        )
    return (
        partials_ok
        and block.dim() == 2
        and residual.dim() == 2
        and residual.dtype == dtype
        and hc.block_inject_weight.weight.dtype == dtype
        and block.is_cuda
        and not torch.compiler.is_compiling()
    )


def _side_partials(hc, rows: int, partials, limits: Limits) -> bool:
    """Partials ``mix`` computed on the side stream."""
    return (
        partials is not None
        and partials.dim() == 3
        and FLOOR_ROWS <= rows <= _fused_cap(limits)
        and _decode_ok(hc)
    )


def write_route(
    hc, block, residual, normed, partials, next_norm, *, limits: Limits = DEFAULT
) -> str:
    if _extend_rows(hc, block, residual, normed, partials, limits):
        return "hc_combine_extend"
    if _side_partials(hc, block.shape[0], partials, limits):
        if (
            next_norm is not None
            and next_norm.group_size == hc.hidden_size
            and next_norm.weight.dtype == torch.bfloat16
            and block.dtype == torch.bfloat16
        ):
            return "hc_combine_apply_norm_wide"
        return "hc_combine_apply"
    return STOCK


def _combine_extend(hc, block, residual, normed, partials, next_norm, next_inject):
    from sgl_nvidia_mega.qwen3_8.kernels.ops.elementwise.hc_combine_extend import (
        hc_combine_extend,
    )

    dtype = block.dtype
    width = hc.hc_count * hc.hidden_size
    if next_norm is not None and not (
        next_norm.group_size == hc.hidden_size
        and next_norm.weight.dtype == dtype
        and next_norm.weight.numel() == width
    ):
        next_norm = None
    if next_norm is None or (
        next_inject is not None
        and (
            next_inject.dtype != dtype
            or tuple(next_inject.shape) != (hc.hc_count, width)
        )
    ):
        next_inject = None
    return hc_combine_extend(
        block,
        residual,
        hc.hc_count,
        hc.hidden_size,
        partials=partials,
        normed_residual=normed if partials is None else None,
        inject_weight=hc.block_inject_weight.weight if partials is None else None,
        norm_weight=next_norm.weight if next_norm is not None else None,
        eps=next_norm.variance_epsilon if next_norm is not None else 0.0,
        next_inject_weight=next_inject,
    )


def write(
    hc,
    block: torch.Tensor,
    residual: torch.Tensor,
    normed: Optional[torch.Tensor],
    partials: Optional[torch.Tensor],
    next_norm=None,
    next_inject: Optional[torch.Tensor] = None,
    *,
    limits: Limits = DEFAULT,
) -> tuple:
    """The write-back of ``block`` into ``residual``: ``(combined, normalized or None,
    the next read's gate totals or None)``.

    ``normed`` and ``partials`` are the read's (``mix``'s or ``boundary``'s);
    ``next_norm`` and ``next_inject`` the next read's norm and gate weight, None
    when something (a PLE embedding) changes the streams before that read."""
    route = write_route(hc, block, residual, normed, partials, next_norm, limits=limits)
    if route == "hc_combine_extend":
        return _combine_extend(
            hc, block, residual, normed, partials, next_norm, next_inject
        )
    if route == "hc_combine_apply_norm_wide":
        from sgl_nvidia_mega.qwen3_8.kernels.ops.elementwise.hc_combine_decode_wide import (
            hc_combine_apply_norm_wide,
        )

        _join(partials, limits)
        combined, normalized = hc_combine_apply_norm_wide(
            block, residual, partials, next_norm.weight, next_norm.variance_epsilon
        )
        return combined, normalized, None
    if route == "hc_combine_apply":
        from sgl_nvidia_mega.qwen3_8.kernels.ops.elementwise.hc_combine_decode_wide import (
            hc_combine_apply,
        )

        _join(partials, limits)
        return hc_combine_apply(block, residual, partials), None, None
    # Extend totals no kernel here reads: SGLang recomputes the gate from the norm
    return hc.combine(block, (residual, normed, None)), None, None


def boundary_reason(
    hc, block, residual, partials, next_hc, next_inject=None, *, limits=DEFAULT
) -> str:
    """'' when this write-back and ``next_hc``'s read run as one launch: the extend
    boundary with its [rows, hc] gate totals, ``next_hc``'s own norm and gate weight,
    contiguous rows and a resident grid."""
    rows = block.shape[0]
    cap = _ks_cap(limits)
    if not FLOOR_ROWS <= rows <= cap:
        return f"{rows} rows: the one-launch boundary takes {FLOOR_ROWS}..{cap}"
    if not (_ks_ok(hc) and _extend_ok(hc) and _ks_ok(next_hc)):
        return "the HC modules are not 4 x 2560 BF16, rank 320, per-branch norm"
    if partials is None or partials.dim() != 2:
        return "the write-back carries no extend gate totals"
    if next_hc.hc_norm.weight.dtype != block.dtype:
        return "the next norm's dtype is not the block's"
    if next_inject is not None and not (
        hasattr(next_hc, "block_inject_weight")
        and next_inject is next_hc.block_inject_weight.weight
    ):
        return "the next gate weight is not the next read's"
    if not _extend_rows(hc, block, residual, None, partials, limits):
        return "the extend boundary does not take this write-back"
    if not (
        block.is_contiguous() and residual.is_contiguous() and partials.is_contiguous()
    ):
        return "the rows are not contiguous"
    from sgl_nvidia_mega.qwen3_8.kernels.ops.gemm.hc_boundary_ks import supported

    device = block.device
    index = device.index if device.index is not None else torch.cuda.current_device()
    if not supported(rows, index):
        return f"{rows} rows: the grid is not resident on this device"
    return ""


def boundary(
    hc,
    block: torch.Tensor,
    residual: torch.Tensor,
    partials: torch.Tensor,
    next_hc,
    next_inject: Optional[torch.Tensor] = None,
    *,
    limits: Limits = DEFAULT,
) -> tuple:
    """The write-back and ``next_hc``'s read in one launch (``boundary_reason`` ''):
    ``(combined, normalized, next gate totals or None, next_hc's mixed input)``.

    Without flag 4 the normalized streams are not written when the next write-back
    reads the totals instead (``next_inject`` given and the extend boundary on);
    ``normalized`` is then an empty tensor, which only that write-back receives."""
    from sgl_nvidia_mega.qwen3_8.kernels.ops.gemm.hc_boundary_ks import (
        FLAG_WRITE_NORMED,
        hc_boundary_ks,
    )

    flags = limits.boundary_ks_flags
    write_normed = bool(flags & FLAG_WRITE_NORMED) or not (
        next_inject is not None and _extend_min(limits) and _extend_ok(next_hc)
    )
    next_norm = next_hc.hc_norm
    combined, normalized, next_partials, mixed = hc_boundary_ks(
        block,
        residual,
        partials=partials,
        norm_weight=next_norm.weight,
        eps=next_norm.variance_epsilon,
        next_inject_weight=next_inject,
        w_down=next_hc.input_mix_weight_down.weight.data,
        w_up_permuted=padded_up_weight(next_hc),
        write_normed=write_normed,
        flags=flags,
    )
    if normalized is None:
        normalized = residual.new_empty((0,))
    return combined, normalized, next_partials, mixed
