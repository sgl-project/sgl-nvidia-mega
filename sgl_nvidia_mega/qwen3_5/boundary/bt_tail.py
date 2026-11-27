# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-FileCopyrightText: Copyright (c) 2026 by FlashInfer team.
# This file has been modified from its original version (the BT launches on SGLang's workspace, the
# RMSNorm launch replaced by the BT tail).
# Adapted from FlashInfer (Apache-2.0, Copyright (c) 2026 by FlashInfer team), flashinfer-python 0.7.4:
# flashinfer/comm/mnnvl_cutedsl/kernel_bt/protocol.py (FinalizeAllReduceRMSNormBTKernel.__call__,
# AllReduceRMSNormBTKernel.__call__, _BTPath._launch_tail) and flashinfer/comm/mnnvl_cutedsl_ar.py
# (MNNVLCuteDSLAllReduceFusionWorkspace._finalize_all_reduce_rms_norm, _all_reduce_rms_norm);
# and from SGLang (Apache-2.0, SGLang Team): python/sglang/srt/layers/flashinfer_mnnvl_cutedsl.py
# (FlashInferMNNVLCuteDSLARFusion.moe_finalize_all_reduce_rms_norm, all_reduce_residual_rms_norm).

from __future__ import annotations

import hashlib
import importlib.metadata
import importlib.util
import logging
from pathlib import Path
from typing import Optional

import torch

logger = logging.getLogger(__name__)

SGLANG_READS = {
    "srt/models/qwen3_5.py": ("flashinfer_mnnvl_cutedsl_fusion",),
    "srt/layers/layer_boundary/fusions/cutedsl.py": (
        "CuteDSLWorkspace",
        "_workspace",
        "rms_epsilon",
        "hidden_size",
        "top_k",
    ),
    "srt/layers/flashinfer_mnnvl_cutedsl.py": (
        "FlashInferMNNVLCuteDSLARFusion",
        "workspace",
        "supports",
    ),
}

SGLANG_MIRRORS = {
    "srt/layers/flashinfer_mnnvl_cutedsl.py:FlashInferMNNVLCuteDSLARFusion.moe_finalize_all_reduce_rms_norm": "e7b9543a6155abfd",
    "srt/layers/flashinfer_mnnvl_cutedsl.py:FlashInferMNNVLCuteDSLARFusion.all_reduce_residual_rms_norm": "4ede0c4123b6a439",
}

FLASHINFER_SOURCES = {
    "comm/mnnvl_cutedsl/kernel_bt/device_kernels.py": (
        "612381d79df87da1db9b26ce8b0f5382a684b607b21cd5b01def93d6c4865318",
    ),
    "comm/mnnvl_cutedsl/kernel_bt/protocol.py": (
        "481e97613907919f6f58097e4f71c310d901e3d4cbf9f6483d49d383aaa0b2f7",
    ),
    "comm/mnnvl_cutedsl_ar.py": (
        "56c44f9a3c68733aaf37292a2cc2c4c59b2f00ed7895bb985b8f82618e75dbe2",
    ),
    "comm/mnnvl_cutedsl/runtime.py": (
        "cdaccbb64a76961e7c32d64b6dcc2119e1684a15014b652270e6167ee53b91d2",
    ),
    # the RMSNorm stage's BF16 norm store the tail mirrors
    "comm/mnnvl_cutedsl/static_fp8_epilogue.py": (
        "d1ff189195c034ac0dcf5aefa908b111fef740b6d058b7d6716eef4264e8425f",
    ),
}
# The RMSNorm thread counts whose sum-of-squares order the tail reproduces.
RMS_THREADS = range(512, 1025, 32)
HIDDEN = 4096
POINTS = ("finalize", "all_reduce")


def flashinfer_root() -> Optional[Path]:
    found = importlib.util.find_spec("flashinfer")
    if found is None or found.origin is None:
        return None
    return Path(found.origin).parent


def flashinfer_reasons() -> list[str]:
    root = flashinfer_root()
    if root is None:
        return ["needs FlashInfer"]
    try:
        version = importlib.metadata.version("flashinfer-python")
    except importlib.metadata.PackageNotFoundError:
        version = "unknown"
    changed = []
    for relative, digests in FLASHINFER_SOURCES.items():
        path = root / relative
        digest = (
            hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else None
        )
        if digest not in digests:
            changed.append(relative)
    if changed:
        return [
            f"FlashInfer {version}'s {', '.join(changed)} differ from the BT protocol "
            "the tail was checked against"
        ]
    return []


def _check(
    t: torch.Tensor, name: str, shape: tuple, dtype: torch.dtype, align: int = 16
) -> None:
    """FlashInfer's argument checks (``_check_tensor``) of a tensor the launches read."""
    if (
        t.dtype != dtype
        or t.ndim != len(shape)
        or any(want is not None and got != want for got, want in zip(t.shape, shape))
        or not t.is_contiguous()
        or t.data_ptr() % align
    ):
        raise ValueError(
            f"bt_tail: {name} {t.dtype} {tuple(t.shape)} is not a contiguous, "
            f"{align}-byte aligned {dtype} {shape}"
        )


class BTTail:
    slot = "bt_tail"

    def __init__(
        self,
        *,
        service,
        eps: float,
        device,
        min_rows: int,
        max_rows: int,
        early: frozenset = frozenset(),
    ):
        from sgl_nvidia_mega.qwen3_5.sm103.hop import bt_tail as kernels

        self.service = service
        self.eps = float(eps)
        self.device = torch.device(device)
        self.min_rows, self.max_rows = int(min_rows), int(max_rows)
        unknown = set(early) - set(POINTS)
        if unknown:
            raise ValueError(f"bt_tail: no point {sorted(unknown)}")
        # the points whose tail releases its dependents right after its wait
        self.early = frozenset(early)
        self.kernels = kernels.kernels(self.device.index)
        self._bound, self._why = False, ""
        # (SGLang's fusion, its FlashInfer workspace, the BT protocol), once bound
        self._fi = None
        self._routes: dict = {}  # rows -> (finalize kernel, all-reduce kernel) or None
        self._pushed_rl = False  # no producer pushes into this protocol
        # slot gemm_ar: its two-shot workspace and the deferred-projection all-reduce
        self.twoshot, self.gemm_ar = None, None
        self.fc2ar = None
        self.ar_reasons: dict = {}

    # ------------------------------------------------------------------ binding

    def bind(self) -> list[str]:
        if self._bound:
            return [] if self._fi is not None else [self._why]
        if getattr(self.service, "_workspace", None) is None:
            return ["SGLang has not built its FlashInfer workspace"]
        self._bound = True
        reasons = self._bind_reasons()
        if reasons:
            from sgl_nvidia_mega.qwen3_5 import runtime

            self._why = "; ".join(reasons)
            runtime.unavailable_slots("bt_tail", self._why)
        return reasons

    def _bind_reasons(self) -> list[str]:
        from flashinfer.comm.mnnvl_cutedsl import ProtocolKind
        from flashinfer.comm.mnnvl_cutedsl.kernel_bt import device_kernels as dk
        from flashinfer.comm.mnnvl_cutedsl.kernel_bt.protocol import BTProtocol

        fusion = self.service._workspace
        ws = getattr(fusion, "workspace", None)
        if ws is None:
            return ["SGLang's fusion holds no FlashInfer MNNVL CuTe DSL workspace"]
        protocol = getattr(ws, "_protocols", {}).get(ProtocolKind.BT)
        if not isinstance(protocol, BTProtocol):
            return ["the workspace runs no BT protocol"]
        reasons = []
        # the norm's output mode (dtype, BF16 companion, per-token FP8 scales)
        output = tuple(
            getattr(ws, name, None)
            for name in ("output_dtype", "write_norm_output", "dynamic_fp8")
        )
        if (
            ws.hidden_dim != HIDDEN
            or ws.dtype != torch.bfloat16
            or not ws.apply_rms_norm
            or not ws.add_residual
            or not ws.write_residual_output
            or ws.weight_bias != 0.0
            or float(ws.rms_eps) != self.eps
            or output != (torch.bfloat16, False, False)
        ):
            reasons.append(
                f"workspace hidden {ws.hidden_dim} {ws.dtype}, norm {ws.apply_rms_norm}, "
                f"residual {ws.add_residual}/{ws.write_residual_output}, weight bias "
                f"{ws.weight_bias}, eps {ws.rms_eps}, output {output} (the tail: bf16 "
                f"4096, residual add, residual output, a bf16 norm with eps {self.eps}, no "
                "bias)"
            )
        stages = (dk.LAMPORT_GENERATIONS, dk.NEXT_STAGE, dk.ACTIVE_STAGE)
        words = getattr(dk, "STAGE_STATE_WORDS", None)
        if stages != (3, 0, 1) or words != 6:
            reasons.append("the BT stages are not three generations in six words")
        state = protocol.state
        prenorm, stage = state.prenorm_mailbox.tensor, state.stage_state
        address = state.prenorm_mailbox.multicast_address
        if (
            prenorm.dtype != torch.bfloat16
            or tuple(prenorm.shape) != (3, protocol.capacity_m, HIDDEN)
            or not prenorm.is_contiguous()
            or prenorm.device != self.device
            or stage.dtype != torch.int32
            or tuple(stage.shape) != (words,)
            or not address
            or address % 16
        ):
            reasons.append(
                f"BT state: prenorm {prenorm.dtype} {tuple(prenorm.shape)}, stages "
                f"{stage.dtype} {tuple(stage.shape)}"
            )
        for point, routes in (
            ("finalize", ws.profile.finalize_routes),
            ("all_reduce", ws.profile.all_reduce_routes),
        ):
            for target in routes.targets:
                if target.protocol is not ProtocolKind.BT:
                    continue
                rms = target.preset.collective.rms_threads
                if rms not in RMS_THREADS:
                    reasons.append(
                        f"the {point} BT RMSNorm runs {rms} threads (the tail's sum "
                        f"order is that of {RMS_THREADS.start}-{RMS_THREADS.stop - 1})"
                    )
        if not reasons:
            self._fi = (fusion, ws, protocol)
        return reasons

    # ------------------------------------------------------------------ routes

    def _route(self, m: int):
        if m in self._routes:
            return self._routes[m]
        if not self._bound:
            self.bind()
            if not self._bound:
                return None
        route = None
        if self._fi is not None and self.min_rows <= m <= self.max_rows:
            from flashinfer.comm.mnnvl_cutedsl import ProtocolKind

            fusion, ws, protocol = self._fi
            fin = ws.profile.finalize_routes.select(m)
            ar = ws.profile.all_reduce_routes.select(m)
            # a fused BT all-reduce runs as one launch, with no RMSNorm launch to replace
            if (
                fusion.supports(m)
                and m <= protocol.capacity_m
                and fin.protocol is ProtocolKind.BT
                and ar.protocol is ProtocolKind.BT
                and not ar.preset.fused
            ):
                route = (
                    protocol.finalize_kernels[fin.preset],
                    protocol.all_reduce_kernels[ar.preset],
                )
        self._routes[m] = route
        return route

    def supports(self, m: int) -> bool:
        return self._route(int(m)) is not None

    # ------------------------------------------------------------------ points

    def moe_finalize_all_reduce_rms_norm(
        self,
        *,
        routed_output,
        expert_weights,
        permuted_indices,
        gated_shared_output,
        residual,
        gamma,
        fp8_scale=None,
    ):
        if self.gemm_ar is not None:
            self.gemm_ar.check_consumed()
        if self.fc2ar is not None:
            from sgl_nvidia_mega.qwen3_5.moe import fc2ar

            if fc2ar.is_lamport(routed_output):
                return tuple(
                    self.fc2ar.moe_finalize_all_reduce_rms_norm(
                        routed_output=routed_output,
                        expert_weights=expert_weights,
                        permuted_indices=permuted_indices,
                        gated_shared_output=gated_shared_output,
                        residual=residual,
                        gamma=gamma,
                        fp8_scale=fp8_scale,
                    )
                )
        from cutlass import Int32
        from flashinfer.comm.mnnvl_cutedsl.runtime import (
            current_cu_stream,
            to_cute,
            to_cute_dynamic,
        )

        m, top_k = (int(n) for n in permuted_indices.shape)
        kernel = self._kernel(m, 0)
        _check(routed_output, "routed_output", (None, HIDDEN), torch.bfloat16)
        _check(expert_weights, "expert weights", (m, kernel.top_k), torch.bfloat16, 2)
        _check(permuted_indices, "permuted rows", (m, kernel.top_k), torch.int32, 4)
        if gated_shared_output is not None:
            _check(gated_shared_output, "shared output", (m, HIDDEN), torch.bfloat16)
        _check(residual, "residual", (m, HIDDEN), torch.bfloat16)
        _check(gamma, "gamma", (HIDDEN,), torch.bfloat16)
        outs = self._outputs(m, fp8_scale)
        _, _, protocol = self._fi
        state = protocol.state
        kernel._validate_state(state, m)
        shared_arg = gated_shared_output if gated_shared_output is not None else gamma
        kernel._compiled.publish(
            to_cute_dynamic(routed_output.flatten(), 16, divisibility=HIDDEN),
            to_cute_dynamic(expert_weights.flatten(), 2, divisibility=top_k),
            to_cute_dynamic(permuted_indices.flatten(), 4, divisibility=top_k),
            to_cute_dynamic(shared_arg.flatten(), 16, divisibility=HIDDEN),
            to_cute(state.stage_state, 4),
            to_cute(state.contribution_mailbox.peer_addresses, 8),
            Int32(m),
            current_cu_stream(),
        )
        self._tail(
            kernel, residual, gamma, state, outs, m, fp8_scale, "finalize" in self.early
        )
        return outs

    def all_reduce_residual_rms_norm(
        self, *, local_contribution, residual, gamma, fp8_scale=None
    ):
        gar = self.gemm_ar
        if gar is not None and gar.pending_for(local_contribution):
            return gar.run(residual, gamma, self.eps, fp8_scale)
        if gar is not None:
            gar.check_consumed()
        from cutlass import Int32
        from flashinfer.comm.mnnvl_cutedsl.runtime import (
            current_cu_stream,
            to_cute,
            to_cute_dynamic,
        )

        m = int(local_contribution.shape[0])
        kernel = self._kernel(m, 1)
        _check(local_contribution, "partial", (m, HIDDEN), torch.bfloat16)
        _check(residual, "residual", (m, HIDDEN), torch.bfloat16)
        _check(gamma, "gamma", (HIDDEN,), torch.bfloat16)
        outs = self._outputs(m, fp8_scale)
        _, _, protocol = self._fi
        state = protocol.state
        kernel._validate_state(state, m)
        kernel._compiled.publish(
            to_cute_dynamic(local_contribution.flatten(), 16, divisibility=HIDDEN),
            to_cute(state.stage_state, 4),
            to_cute(state.contribution_mailbox.peer_addresses, 8),
            Int32(m),
            current_cu_stream(),
        )
        self._tail(
            kernel,
            residual,
            gamma,
            state,
            outs,
            m,
            fp8_scale,
            "all_reduce" in self.early,
        )
        return outs

    def _kernel(self, m: int, point: int):
        route = self._route(m)
        if route is None:
            raise ValueError(f"bt_tail: no BT route of both points at {m} rows")
        return route[point]

    def _outputs(self, m: int, fp8_scale):
        shape = (m, HIDDEN)
        outs = [
            torch.empty(shape, dtype=torch.bfloat16, device=self.device),
            torch.empty(shape, dtype=torch.bfloat16, device=self.device),
        ]
        if fp8_scale is not None:
            outs.append(
                torch.empty(shape, dtype=torch.float8_e4m3fn, device=self.device)
            )
        return tuple(outs)

    def _tail(self, kernel, residual, gamma, state, outs, m, fp8_scale, early):
        from cutlass import Int32, Int64
        from flashinfer.comm.mnnvl_cutedsl.runtime import (
            current_cu_stream,
            to_cute,
            to_cute_dynamic,
        )

        from sgl_nvidia_mega.qwen3_5.sm103.hop import bt_tail as kernels

        kernel._compiled.tail.reduce(
            to_cute(state.contribution_mailbox.tensor.flatten(), 16),
            to_cute_dynamic(residual.flatten(), 16, divisibility=HIDDEN),
            to_cute(state.stage_state, 4),
            Int64(int(state.prenorm_mailbox.multicast_address)),
            Int32(m),
            current_cu_stream(),
        )
        norm, res, *fp8 = outs
        kernels.tail(
            self.kernels,
            prenorm=state.prenorm_mailbox.tensor,
            stage_state=state.stage_state,
            gamma=gamma,
            eps=self.eps,
            m=m,
            residual_out=res,
            norm_out=norm,
            fp8_scale=fp8_scale,
            fp8_out=fp8[0] if fp8 else None,
            early=early,
        )
