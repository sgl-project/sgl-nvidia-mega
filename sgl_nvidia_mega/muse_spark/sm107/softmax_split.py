# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Component ``softmax_split`` (env ``MUSE_VR_SOFTMAX_SPLIT=1``, default off).

The DFLASH verify sampling path (``dspark_accept.softmax_temp_flashinfer``) turns the verify logits into
probabilities with FlashInfer's ``softmax`` (temperature per row). For up to 128 rows over a large vocabulary FlashInfer
runs a slice pass (``OnlineSoftmaxMapKernel``: one block per row and 8192-entry slice, writing per-slice max and
denominator) and then a normalization pass with one block per row, which merges the slice partials and writes the
whole row. This component keeps the slice pass (FlashInfer's own kernel, launched as FlashInfer launches it) and
splits the normalization pass along the vocabulary (``csrc/softmax_split.cuh``): each block repeats the row's merge of
the partials with the same code and block size and writes one column range with the same per-element arithmetic.
``tests/test_softmax_split.py`` checks the probabilities bit for bit against FlashInfer's ``softmax`` and times both.

Any other case (more than 128 rows, small vocabularies, devices where FlashInfer picks another softmax kernel, other
dtypes, CUDA-graph capture) calls the stock function.
"""

from __future__ import annotations

import logging
import os
from functools import lru_cache
from pathlib import Path

import torch

from sgl_nvidia_mega.muse_spark.components import bed

# The SGLang names this module reads (test/unit/test_contract.py pins them against the supported SGLang)
SGLANG_READS = {
    "kernels/jit/utils/__init__.py": ("load_jit",),
    "kernels/ops/speculative/dspark/dspark_accept.py": ("softmax_temp_flashinfer",),
    "srt/plugins/hook_registry.py": ("HookRegistry", "HookType"),
}
# The verify softmax this component serves (the hook repeats its temperature and layout steps)
SGLANG_MIRRORS = {
    "kernels/ops/speculative/dspark/dspark_accept.py:softmax_temp_flashinfer": "9a09ba5e1b5804b6",
}


logger = logging.getLogger(__name__)

NAME = "softmax_split"  # env MUSE_VR_SOFTMAX_SPLIT
TARGET = "sglang.kernels.ops.speculative.dspark.dspark_accept.softmax_temp_flashinfer"
ENV_COLS = "MUSE_VR_SOFTMAX_SPLIT_COLS"
_TARGET_BLOCKS = 256  # normalization blocks over all rows; tunable (tests/test_softmax_split.py --bench --cols ...)
_CUH = Path(__file__).resolve().parent / "csrc" / "softmax_split.cuh"
# FlashInfer OnlineSoftmax regime that this mirrors: vocabulary splitting for small batches and large vocabularies
_MAX_ROWS = 128
_MIN_VOCAB = 24576
_SLICE = 8192

_ws: dict = {"cols": None}


def enabled() -> bool:
    return bed.enabled(NAME)


def _flashinfer_root() -> Path | None:
    import importlib.util

    spec = importlib.util.find_spec("flashinfer")
    if spec is None or not spec.submodule_search_locations:
        return None
    return Path(list(spec.submodule_search_locations)[0]).resolve().parent


@lru_cache(maxsize=None)
def _jit_module():
    from sglang.kernels.jit.utils import load_jit

    root = _flashinfer_root()
    cccl = [
        root / "flashinfer" / "data" / "cccl" / d
        for d in ("cub", "libcudacxx/include", "thrust")
    ]
    return load_jit(
        "muse_vr_softmax_split",
        cuda_files=[str(_CUH)],
        cuda_wrappers=[("run", "sglang::muse_vr::SplitSoftmax::run")],
        extra_dependencies=["flashinfer"],
        extra_include_paths=[str(p) for p in cccl],
        extra_cuda_cflags=[
            "-use_fast_math"
        ],  # as FlashInfer builds its sampling module
    )


@lru_cache(maxsize=None)
def _stock_kernel_family(device_index: int) -> bool:
    """True when FlashInfer's ``softmax`` runs its OnlineSoftmax kernels on this device (not another softmax op)."""
    try:
        from flashinfer import sampling as fs

        return not fs._supports_blackwell_softmax(device_index)
    except Exception:  # noqa: BLE001
        return False


@lru_cache(maxsize=None)
def _pdl(device) -> bool:
    from flashinfer.utils import device_support_pdl

    return bool(device_support_pdl(device))


def _cols(rows: int, vocab: int) -> int:
    """Columns per normalization block: ``MUSE_VR_SOFTMAX_SPLIT_COLS`` if set, else about ``_TARGET_BLOCKS`` blocks
    over all rows, in multiples of 4096 columns and at least 8192 (tunable: tests/test_softmax_split.py --bench)."""
    v = _ws.get("cols", 0)
    if v is None:
        try:
            v = int(os.environ.get(ENV_COLS, "0"))
        except ValueError:
            v = 0
        v = max(0, v - v % 16)
        _ws["cols"] = v
    if v > 0:
        return v
    want = -(-vocab * rows // _TARGET_BLOCKS)
    return max(8192, -(-want // 4096) * 4096)


def split_softmax(logits_2d: torch.Tensor, temp_per_row: torch.Tensor) -> torch.Tensor:
    rows, vocab = logits_2d.shape
    dev = logits_2d.device
    need = rows * (-(-vocab // _SLICE)) * 8
    ws = _ws.get(dev.index)
    if ws is None or ws.numel() < need:
        ws = _ws[dev.index] = torch.empty(
            max(need, 1 << 16), dtype=torch.uint8, device=dev
        )
    out = torch.empty_like(logits_2d)
    _jit_module().run(
        ws, logits_2d, out, temp_per_row, 1.0, _cols(rows, vocab), _pdl(dev)
    )
    return out


def eligible(logits_2d: torch.Tensor) -> str | None:
    if (
        not logits_2d.is_cuda
        or logits_2d.dtype != torch.float32
        or logits_2d.dim() != 2
    ):
        return "dtype/layout"
    rows, vocab = logits_2d.shape
    if rows == 0 or rows > _MAX_ROWS or vocab < _MIN_VOCAB:
        return "shape outside the slice regime"
    if not _stock_kernel_family(logits_2d.device.index):
        return "device uses another softmax kernel"
    if torch.cuda.is_current_stream_capturing():
        # allowed only inside ``sampling_graph``'s capture (workspace from its warm-up); that component is a host-side
        # runtime one and is not part of this package, so a capture is always refused here
        return "graph capture"
    return None


def softmax_temp_flashinfer(original, *args, **kwargs):
    """``dspark_accept.softmax_temp_flashinfer`` (its mirrored body: per-row temperature, fp32 contiguous logits,
    FlashInfer ``softmax``) with the split normalization pass when eligible."""
    if args or set(kwargs) != {"logits", "temperatures", "rows_per_request"}:
        return original(*args, **kwargs)
    logits, temperatures, rows_per_request = (
        kwargs["logits"],
        kwargs["temperatures"],
        kwargs["rows_per_request"],
    )
    num_rows = logits.shape[0]
    bs = num_rows // rows_per_request
    if bs * rows_per_request != num_rows:
        return original(**kwargs)
    temp_per_row = torch.repeat_interleave(
        temperatures.reshape(bs).to(torch.float32), rows_per_request, dim=0
    ).contiguous()
    logits_2d = logits.to(torch.float32).contiguous()
    reason = eligible(logits_2d)
    if reason is not None:
        bed.count(NAME, "stock:" + reason)
        return original(**kwargs)
    bed.count(NAME, f"split:rows={num_rows}")
    return split_softmax(logits_2d, temp_per_row)


def register() -> bool:
    if not enabled():
        return False
    from sglang.srt.plugins.hook_registry import HookRegistry, HookType

    HookRegistry.register(TARGET, softmax_temp_flashinfer, HookType.AROUND)
    bed.report(NAME, True)
    return True
