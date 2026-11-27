# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# Adapted from TensorRT-LLM (Apache-2.0): tensorrt_llm/_torch/moe/fused_moe/trtllm_gen/decode_moe.py
# (the Lamport FC2 rows) and tensorrt_llm/_torch/distributed/mnnvl_fused_allreduce.py.

from __future__ import annotations

import logging
from typing import Optional

import torch

logger = logging.getLogger(__name__)

SGLANG_READS = {
    "srt/layers/layer_boundary/fusions/moe_finalize.py": (
        "MoeDeferredFinalize",
        "routed_output",
    ),
    "srt/layers/layer_boundary/residual/batch.py": ("stream_of",),
    "srt/layers/layer_boundary/residual/stream.py": ("pending", "owed"),
}

TWO_SHOT_ROWS = 64
# the engine's CTAs in an armed call; the remaining SMs hold the all-reduce's CTAs
ENGINE_GRID = 136
# nanoseconds between the all-reduce's Lamport poll rounds (0: spin)
POLL_NS = 0
HIDDEN = 4096
SENTINEL = -1

_BUFFERS: dict = {}
# the rows armed for the current target MoE call (``scope``), or 0
_ARMED = [0]
# whether an engine call of the current armed scope wrote the Lamport rows
_HIT = [False]
_LOGGED: set = set()
# the widths whose fallback to BT was announced
_FELL: set = set()


class LamportAR:
    slot = "fc2ar"

    def __init__(self, ar, lo: int, hi: int, device, poll_ns: int = POLL_NS):
        self.ar, self.lo, self.hi = ar, int(lo), int(hi)
        self.device = torch.device(device)
        self.poll_ns = int(poll_ns)
        self.sms = torch.cuda.get_device_properties(self.device).multi_processor_count
        self.fence = torch.zeros((1,), dtype=torch.int32, device=self.device)
        self.test_no_poll = False
        self.test_trtllm_numerics = False

    def supports(self, m: int) -> bool:
        return self.lo <= int(m) <= self.hi

    def ts_ctas(self, m: int) -> int:
        if m < TWO_SHOT_ROWS:
            return 0
        from sgl_nvidia_mega.qwen3_5.sm103.hop.twoshot_ar import (
            ctas_per_token,
            tokens_per_cta,
        )

        ts = ctas_per_token(m, HIDDEN, self.sms)
        if tokens_per_cta(m, ts, self.sms) != 1:
            raise ValueError(
                f"slot fc2ar: the two-shot arithmetic at {m} rows takes one token a CTA"
            )
        return ts

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
        if not is_lamport(routed_output):
            raise RuntimeError(
                "slot fc2ar: the slot all-reduce's Lamport fold got FC2 rows outside the "
                "Lamport buffers"
            )
        if gated_shared_output is None:
            raise RuntimeError(
                "slot fc2ar: the Lamport fold needs the gated shared output"
            )
        m = int(residual.shape[0])
        trt = self.test_trtllm_numerics
        if (m, "ar") not in _LOGGED:
            _LOGGED.add((m, "ar"))
            logger.info(
                "NVIDIA megakernel: Qwen3.5 slot fc2ar: the slot all-reduce polls the engine's "
                "Lamport FC2 rows at %d rows (FlashInfer BT's rounding points, poll_ns %d)",
                m,
                self.poll_ns,
            )
        return self.ar(
            residual,
            gamma,
            moe=(routed_output, expert_weights, permuted_indices, gated_shared_output),
            fp8_scale=fp8_scale,
            fc2_lamport=True,
            ts_ctas=self.ts_ctas(m) if trt else 0,
            poll_ns=self.poll_ns,
            test_no_poll=self.test_no_poll,
            bt_numerics=not trt,
        )


_ACTIVE: list = [None]


def active() -> Optional[LamportAR]:
    return _ACTIVE[0]


def build(eps: float, group, device, widths, experts) -> tuple:
    from sgl_nvidia_mega.qwen3_5 import plan
    from sgl_nvidia_mega.qwen3_5.runtime import SlotUnavailable
    from sgl_nvidia_mega.qwen3_5.sm103.hop.slot_ar import SlotAllReduce

    lo = plan.FC2AR_MIN_ROWS
    hi = plan.widest(widths, plan.FC2AR_MAX_ROWS)
    if hi < lo:
        return (
            None,
            f"no width of {list(widths)} within {lo}-{plan.FC2AR_MAX_ROWS} rows",
        )
    try:
        ar = SlotAllReduce(
            hidden=HIDDEN, eps=eps, group=group, device=device, max_rows=hi
        )
    except SlotUnavailable as exc:
        return None, str(exc)
    lam = LamportAR(ar, lo, hi, device)
    buf = _lamport_buffer(
        torch.device(device), lamport_rows(hi, experts.top_k, experts.num_local_experts)
    )
    _ACTIVE[0] = lam
    logger.info(
        "NVIDIA megakernel: Qwen3.5 slot fc2ar built (%d-%d rows): Lamport FC2 rows + the slot "
        "one-shot MoE-output all-reduce overlapped with the MoE engine (world %d, engine "
        "early_trigger 2, poll_ns %d; Lamport FC2 buffer %d x %d, %.1f MiB)",
        lo,
        hi,
        ar.world,
        lam.poll_ns,
        buf.shape[0],
        buf.shape[1],
        buf.numel() * buf.element_size() / 2**20,
    )
    return lam, ""


def _lamport_buffer(device: torch.device, rows: int) -> torch.Tensor:
    buf = _BUFFERS.get(device.index)
    if buf is None or buf.shape[0] < rows:
        buf = torch.empty((rows, HIDDEN), dtype=torch.bfloat16, device=device)
        buf.view(torch.int32).fill_(SENTINEL)
        _BUFFERS[device.index] = buf
    return buf


def _fallback(m: int, why: str) -> None:
    if m not in _FELL:
        _FELL.add(m)
        logger.warning(
            "NVIDIA megakernel: Qwen3.5 slot fc2ar at %d rows: the MoE-output all-reduce "
            "stays on BT (%s)",
            m,
            why,
        )


class scope:
    def __init__(self, rows: int, st):
        self.rows = 0
        self._prev = (0, False)
        if not st.slots.fc2ar:
            return
        lam, tail = _ACTIVE[0], st.bound["boundary"].bt
        if lam is None or tail.fc2ar is not lam or not lam.supports(rows):
            _fallback(int(rows), "slot bt_tail holds no all-reduce for these rows")
        elif not tail.supports(rows):
            _fallback(int(rows), "no BT route at these rows")
        else:
            self.rows = int(rows)

    def __enter__(self):
        self._prev = (_ARMED[0], _HIT[0])
        _ARMED[0], _HIT[0] = self.rows, False
        return self

    def __exit__(self, *exc):
        if self.rows and not _HIT[0] and exc[0] is None:
            _fallback(
                self.rows, "no MoE engine call at these rows wrote the Lamport FC2 rows"
            )
        _ARMED[0], _HIT[0] = self._prev
        return False


def armed() -> bool:
    return _ARMED[0] > 0


def armed_at(m: int) -> bool:
    return _ARMED[0] == m and _ACTIVE[0] is not None


def aux_fence() -> None:
    if _ARMED[0] > 0 and _ACTIVE[0] is not None:
        _ACTIVE[0].fence.zero_()


def engine_opts(opts: dict, m: int) -> dict:
    if not armed_at(m):
        return opts
    out = dict(opts, early_trigger=2)
    if not opts.get("grid_ctas"):
        out["grid_ctas"] = ENGINE_GRID
    return out


def lamport_rows(rows: int, top_k: int, local: int) -> int:
    from sgl_nvidia_mega.qwen3_5.moe import body

    return max(body.engine_rows(int(rows), top_k, local, t) for t in body.TILES)


def engine_out(
    out: torch.Tensor, m: int, tile: int, trigger: Optional[int] = None
) -> torch.Tensor:
    if not armed_at(m):
        return out
    _HIT[0] = True
    buf = _BUFFERS.get(out.device.index)
    need = out.shape[0]
    if buf is None or need > buf.shape[0] or out.shape[1] != buf.shape[1]:
        raise RuntimeError(
            "slot fc2ar: the Lamport FC2 buffer made at build "
            f"({None if buf is None else tuple(buf.shape)}) cannot hold {tuple(out.shape)}"
        )
    if (m, tile) not in _LOGGED:
        _LOGGED.add((m, tile))
        logger.info(
            "NVIDIA megakernel: Qwen3.5 slot fc2ar: the MoE engine writes Lamport FC2 rows at "
            "%d rows (tile_n %d, early_trigger %d)",
            m,
            tile,
            trigger if trigger is not None else 2,
        )
    return buf[:need]


def is_lamport(t: torch.Tensor) -> bool:
    """Whether ``t`` lies in a Lamport FC2 buffer."""
    p = t.data_ptr()
    for buf in _BUFFERS.values():
        lo = buf.data_ptr()
        if lo <= p < lo + buf.numel() * buf.element_size():
            return True
    return False


def pending(st) -> Optional[LamportAR]:
    if _ACTIVE[0] is None or not _BUFFERS:
        return None
    from sglang.srt.layers.layer_boundary.fusions.moe_finalize import (
        MoeDeferredFinalize,
    )
    from sglang.srt.layers.layer_boundary.residual.batch import stream_of

    p = stream_of(st.fb).pending
    if p is not None and isinstance(p.owed, MoeDeferredFinalize):
        if is_lamport(p.owed.routed_output):
            return _ACTIVE[0]
    return None


def refuse_unconsumed(st, point: str) -> None:
    """Raise where a boundary other than the Lamport all-reduce would read a Lamport handoff."""
    if pending(st) is not None:
        raise RuntimeError(
            f"slot fc2ar: the {point} holds Lamport FC2 rows, but its boundary does not take "
            "the slot all-reduce (another reader would leave the words written and the next "
            "fold would read them as ready)"
        )
