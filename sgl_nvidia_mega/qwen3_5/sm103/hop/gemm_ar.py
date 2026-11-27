# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

from __future__ import annotations

import logging

import torch

logger = logging.getLogger(__name__)

SGLANG_READS = {
    "srt/layers/quantization/fp8_utils.py": ("flashinfer_bmm_fp8",),
}

_LOGGED: set = set()


def _once(key, level, msg, *args) -> None:
    if key not in _LOGGED:
        _LOGGED.add(key)
        logger.log(level, msg, *args)


class GemmAllReduce:
    def __init__(self, twoshot, rows: tuple, opts: int):
        from sgl_nvidia_mega.qwen3_5.sm103.hop import gemm_ar_kernel as k

        self.ws = twoshot
        self.lo, self.hi = rows
        if k.workspace_rows(self.hi) > twoshot.max_rows:
            raise ValueError(
                f"slot gemm_ar at {self.hi} rows needs a {k.workspace_rows(self.hi)}-row workspace"
            )
        self.opts = int(opts)
        self.sms = torch.cuda.get_device_properties(
            twoshot.device
        ).multi_processor_count
        self._pending = None

    def covers(self, rows: int) -> bool:
        return self.lo <= rows <= self.hi

    def defer(self, out: torch.Tensor, x_fp8: torch.Tensor, linear, what: str) -> bool:
        from sgl_nvidia_mega.qwen3_5.sm103.hop import gemm_ar_kernel as k

        m = int(x_fp8.shape[0])
        w = linear.weight.t()  # [H, K], the checkpoint's row-major FP8 weight
        why = ""
        if self._pending is not None:
            raise RuntimeError("slot gemm_ar: a deferred projection was never consumed")
        if (
            x_fp8.dtype != torch.float8_e4m3fn
            or not x_fp8.is_contiguous()
            or x_fp8.data_ptr() % 128
        ):
            why = f"input {x_fp8.dtype} contiguous={x_fp8.is_contiguous()} at {x_fp8.data_ptr() % 128} of 128"
        elif (
            w.dtype != torch.float8_e4m3fn
            or not w.is_contiguous()
            or w.data_ptr() % 128
        ):
            why = f"weight {w.dtype} {tuple(w.shape)} not a contiguous [H, K] E4M3"
        elif linear.input_scale.numel() != 1 or linear.weight_scale.numel() != 1:
            why = "not a per-tensor static FP8 linear"
        elif (
            tuple(out.shape) != (m, w.shape[0])
            or out.dtype != torch.bfloat16
            or w.shape[1] != x_fp8.shape[1]
        ):
            why = f"partial {out.dtype} {tuple(out.shape)} for {tuple(x_fp8.shape)} @ {tuple(w.shape)}"
        else:
            why = k.unsupported(
                m, int(w.shape[1]), int(w.shape[0]), self.ws.world, self.sms, self.opts
            )
        if why:
            _once(
                ("refuse", what, m, why),
                logging.WARNING,
                "NVIDIA megakernel: Qwen3.5 slot gemm_ar (%d-%d rows) refuses %s at %d rows: "
                "%s; cuBLAS + BT serve it",
                self.lo,
                self.hi,
                what,
                m,
                why,
            )
            return False
        self._pending = (out, x_fp8, w, linear.input_scale, linear.weight_scale, what)
        return True

    def pending_for(self, partial: torch.Tensor) -> bool:
        p = self._pending
        return (
            p is not None
            and p[0].data_ptr() == partial.data_ptr()
            and p[0].shape == partial.shape
        )

    def check_consumed(self) -> None:
        if self._pending is not None:
            raise RuntimeError(
                f"slot gemm_ar: the deferred {self._pending[5]} was never consumed"
            )

    def run(self, residual, gamma, eps: float, fp8_scale=None):
        from sgl_nvidia_mega.qwen3_5.sm103.hop import gemm_ar_kernel as k

        out, x, w, sx, sw, what = self._pending
        self._pending = None
        m = int(x.shape[0])
        _once(
            ("run", what, m),
            logging.INFO,
            "NVIDIA megakernel: Qwen3.5 slot gemm_ar runs %s + the MoE input's all-reduce + "
            "norm%s in one launch at %d rows (GEMM + two-shot, world %d, opts %d)",
            what,
            " + FP8" if fp8_scale is not None else "",
            m,
            self.ws.world,
            self.opts,
        )
        return k.gemm_ar_norm(
            x,
            w,
            sw,
            sx,
            residual.contiguous(),
            gamma.reshape(-1),
            eps,
            self.ws.table,
            self.ws.rank,
            self.ws.world,
            scale=fp8_scale,
            opts=self.opts,
        )

    def settle(self) -> None:
        if self._pending is None:
            return
        from sglang.srt.layers.quantization.fp8_utils import flashinfer_bmm_fp8

        out, x, w, sx, sw, what = self._pending
        self._pending = None
        _once(
            ("settle", what, int(x.shape[0])),
            logging.WARNING,
            "NVIDIA megakernel: Qwen3.5 slot gemm_ar deferred %s at %d rows, but its "
            "all-reduce did not run on the BT tail: the projection runs on cuBLAS after all",
            what,
            int(x.shape[0]),
        )
        out.copy_(flashinfer_bmm_fp8(x, w.t(), sx, sw, out.dtype))


def build(eps: float, group, device, widths) -> tuple:
    import torch.distributed as dist

    from sgl_nvidia_mega.qwen3_5 import plan
    from sgl_nvidia_mega.qwen3_5.runtime import SlotUnavailable
    from sgl_nvidia_mega.qwen3_5.sm103.hop import HIDDEN
    from sgl_nvidia_mega.qwen3_5.sm103.hop import gemm_ar_kernel as k
    from sgl_nvidia_mega.qwen3_5.sm103.hop.twoshot_ar import TwoShotWorkspace

    lo, hi = plan.GEMM_AR_MIN_ROWS, plan.GEMM_AR_MAX_ROWS
    covered = [w for w in widths if lo <= w <= hi]
    if not covered:
        return None, None, f"no width of {list(widths)} within {lo}-{hi} rows"
    world = dist.get_world_size(group)
    hi = min(hi, k.max_rows(HIDDEN, world, k.DEFAULT_OPTS))
    covered = [w for w in covered if w <= hi]
    if not covered:
        return (
            None,
            None,
            f"no width of {list(widths)} within {lo}-{hi} rows at world {world}",
        )
    max_rows = k.workspace_rows(max(covered))
    try:
        ws = TwoShotWorkspace(
            hidden=HIDDEN, eps=eps, group=group, device=device, max_rows=max_rows
        )
    except SlotUnavailable as exc:
        return None, None, str(exc)
    gar = GemmAllReduce(ws, (lo, min(hi, max_rows)), k.DEFAULT_OPTS)
    logger.info(
        "NVIDIA megakernel: Qwen3.5 slot gemm_ar built: two-shot workspace for %d rows "
        "(world %d), %d-%d rows",
        max_rows,
        ws.world,
        gar.lo,
        gar.hi,
    )
    return ws, gar, ""
