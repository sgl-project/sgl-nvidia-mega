# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

from __future__ import annotations

import logging
from typing import Optional

import torch

logger = logging.getLogger(__name__)

SGLANG_READS = {
    "srt/models/qwen3_5_mtp.py": ("lm_head", "logits_processor"),
    "srt/layers/logits_processor.py": ("logit_scale", "final_logit_softcapping"),
    "srt/kernel_plugins/nvidia/eagle_binding.py|srt/megakernels/eagle_binding.py": (
        "tp_rank",
        "tp_cpu_group",
    ),
}

SLOT = "draft_head_lowm"
MAX_ROWS = 8
RACE_MARGIN_US = 0.5
CANDIDATES = ((8, 1, 1, 0, 1), (8, 1, 1, 1, 1))
# wait-before-trigger: the dependent is released only after this grid's own wait
PDL = dict(pdl=True, early_trigger=False, release_after_wait=True)
PUSH_RELEASES = {
    "wait": PDL,
    "end": dict(pdl=True, early_trigger=False, release_after_wait=False),
}
# (label, rows) whose capture logged the engagement line
_ENGAGED: set = set()


def note(label: str, rows: int) -> None:
    if torch.cuda.is_current_stream_capturing() and (label, rows) not in _ENGAGED:
        _ENGAGED.add((label, rows))
        logger.info(
            "NVIDIA megakernel: slot %s runs at %d rows (low-M GEMM)", label, rows
        )


def _timed(fn, reps: int = 50) -> float:
    """Microseconds a call of ``fn`` replayed from a CUDA graph."""
    fn()
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        fn()
    for _ in range(5):
        g.replay()
    torch.cuda.synchronize()
    a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    a.record()
    for _ in range(reps):
        g.replay()
    b.record()
    torch.cuda.synchronize()
    return a.elapsed_time(b) * 1000 / reps


def _timed_on(fn_of_w, copies: list, reps: int = 50) -> float:
    if len(copies) == 1:
        return _timed(lambda: fn_of_w(copies[0]), reps)
    calls = max(4, reps // len(copies))
    return _timed(lambda: [fn_of_w(w) for w in copies], calls) / len(copies)


def push_spec(
    m: int,
    k: int,
    n: int,
    tactic: tuple,
    release: str,
    push: int,
    out_stride: Optional[int] = None,
):
    """The ``trt_lowm`` push spec of a ``race_push`` pick."""
    from sgl_nvidia_mega.qwen3_5.sm103 import trt_lowm

    return trt_lowm.spec(
        m=m,
        k=k,
        n_bf16=n,
        fp8_act="bf16",
        tactic_bf16=tactic,
        out_stride=out_stride,
        push=push,
        **PUSH_RELEASES[release],
    )


def race_push(
    weight: torch.Tensor,
    rows: tuple,
    *,
    candidates: tuple,
    cpu_group,
    rank: int,
    world: int,
    label: str,
    consume: dict,
    out_stride: Optional[int] = None,
    col0: int = 0,
    cold_copies: int = 1,
) -> dict:
    import types

    import torch.distributed as dist

    from sgl_nvidia_mega.qwen3_5.sm103 import hop as decode_hop
    from sgl_nvidia_mega.qwen3_5.sm103 import trt_lowm
    from sgl_nvidia_mega.qwen3_5.sm103.draft import hop as kernels
    from sgl_nvidia_mega.qwen3_5.sm103.trt_lowm.lowm_linear import default_tactic

    weight = weight.detach()
    n, k = (int(v) for v in weight.shape)
    dev = weight.device
    rows = tuple(sorted(set(rows)))
    chosen, race = {}, []
    if rows:
        row_elems = out_stride or n
        m_cap = max(rows)
        box = torch.full(
            (3, world, m_cap, row_elems), 0x0101, dtype=torch.int16, device=dev
        )
        image = box.clone()
        ctrl = torch.zeros(4, dtype=torch.int32, device=dev)
        desc = trt_lowm.push_descriptor(
            mailbox=box.data_ptr(),
            ctrl=ctrl.data_ptr(),
            rank=rank,
            world=world,
            m_cap=m_cap,
            row_elems=row_elems,
            col0=col0,
            device=dev,
        )
        site = decode_hop.Site(
            box.data_ptr(), box.data_ptr(), ctrl.data_ptr(), rank, world, m_cap
        )
        hop = types.SimpleNamespace(
            site=site, world=world, device=dev, supports=lambda m: 1 <= m <= m_cap
        )
        copies = [weight] + [weight.clone() for _ in range(cold_copies - 1)]
        for m in rows:
            x = torch.randn(m, k, device=dev).bfloat16()
            out = torch.empty(m, row_elems, dtype=torch.bfloat16, device=dev)[:, :n]
            stub = torch.zeros(m, row_elems, dtype=torch.bfloat16, device=dev)
            zero = torch.zeros(m, row_elems, dtype=torch.bfloat16, device=dev)
            tactics = [t for t in candidates if t[3] == 0 or m <= 8]
            d = tuple(default_tactic(m, n, k, fp8=False))
            if d not in tactics:
                tactics.append(d)
            best = None
            for tac in tactics:
                for release in PUSH_RELEASES:
                    s = push_spec(m, k, n, tac, release, 2, out_stride)
                    try:
                        trt_lowm.prepare(s)
                    except Exception as exc:  # a tactic the kernel refuses: not raced
                        logger.info(
                            "%s push race: m %d tactic %s refused: %s",
                            label,
                            m,
                            tac,
                            exc,
                        )
                        break

                    def chain(w, s=s, x=x, out=out, stub=stub, zero=zero):
                        box.copy_(image)
                        trt_lowm.run(s, out=out, x16=x, w_bf16=w, push=desc)
                        kernels.draft_all_reduce_rms_norm(
                            hop,
                            local_contribution=stub,
                            residual=zero,
                            pushed=True,
                            **consume,
                        )

                    t = _timed_on(chain, copies)
                    if best is None or t < best[0]:
                        best = (t, tac, release)
            if best is not None:
                chosen[m] = (best[1], best[2])
                race.append((m, round(best[0], 2), best[1], best[2]))
        del copies, box, image
    obj = [sorted(chosen.items())]
    dist.broadcast_object_list(obj, src=0, group=cpu_group)
    if rank == 0:
        logger.info(
            "NVIDIA megakernel: %s push race (m, push GEMM + consume us with a "
            "mailbox refill, tactic, release): %s",
            label,
            race,
        )
    return dict(obj[0])


def _tactic(s) -> tuple:
    return (s.k_warps, s.n_tiles_bf16, s.ksplit, int(s.swap_ab), s.min_blocks)


class HeadLowM:
    def __init__(
        self,
        weight: torch.Tensor,
        rows: tuple,
        *,
        cpu_group,
        rank: int,
        max_rows: int = MAX_ROWS,
        candidates: tuple = CANDIDATES,
        label: str = SLOT,
        cold_copies: int = 1,
        out_stride: Optional[int] = None,
    ):
        import torch.distributed as dist

        from sgl_nvidia_mega.qwen3_5.sm103 import trt_lowm
        from sgl_nvidia_mega.qwen3_5.sm103.trt_lowm.lowm_linear import default_tactic

        if (
            weight.dtype != torch.bfloat16
            or weight.dim() != 2
            or not weight.is_contiguous()
        ):
            raise ValueError(f"{label}: weight {weight.dtype} {tuple(weight.shape)}")
        weight = weight.detach()
        self.weight, self.label = weight, label
        self.out_stride = out_stride
        n, k = weight.shape
        dev = weight.device
        self.specs: dict = {}
        race = []
        copies = [weight] + [weight.clone() for _ in range(cold_copies - 1)]
        for m in sorted({r for r in rows if 1 <= r <= max_rows}):
            x = torch.randn(m, k, device=dev).bfloat16()
            if out_stride is not None:
                win = self._out(m, dev)
                t_cublas = _timed_on(lambda w: torch.mm(x, w.t(), out=win), copies)
            else:
                t_cublas = _timed_on(lambda w: torch.nn.functional.linear(x, w), copies)
            best = None
            tactics = [t for t in candidates if t[3] == 0 or m <= 8]
            default = tuple(default_tactic(m, n, k, fp8=False))
            if default not in tactics:
                tactics.append(default)
            for tac in tactics:
                s = trt_lowm.spec(
                    m=m,
                    k=k,
                    n_bf16=n,
                    fp8_act="bf16",
                    tactic_bf16=tac,
                    out_stride=out_stride,
                    **PDL,
                )
                try:
                    trt_lowm.prepare(s)
                except Exception as exc:  # a tactic the kernel refuses: not raced
                    logger.info("%s: m %d tactic %s refused: %s", label, m, tac, exc)
                    continue
                out = self._out(m, dev)
                t = _timed_on(
                    lambda w: trt_lowm.run(s, out=out, x16=x, w_bf16=w), copies
                )
                if best is None or t < best[0]:
                    best = (t, s)
            take = best is not None and best[0] + RACE_MARGIN_US < t_cublas
            race.append((m, t_cublas, best[0] if best else None, take))
            if take:
                self.specs[m] = best[1]
        # every rank runs rank 0's choice: the same widths and tactics
        chosen = [sorted((m, _tactic(sp)) for m, sp in self.specs.items())]
        dist.broadcast_object_list(chosen, src=0, group=cpu_group)
        self.specs = {}
        for m, tac in chosen[0]:
            s = trt_lowm.spec(
                m=m,
                k=k,
                n_bf16=n,
                fp8_act="bf16",
                tactic_bf16=tuple(tac),
                out_stride=out_stride,
                **PDL,
            )
            trt_lowm.prepare(s)
            self.specs[m] = s
        del copies
        if rank == 0:
            logger.info(
                "NVIDIA megakernel: slot %s race (m, cuBLAS us, low-M us, taken): %s; "
                "picks (m, tactic): %s",
                label,
                [
                    (m, round(a, 2), None if b is None else round(b, 2), t)
                    for m, a, b, t in race
                ],
                chosen[0],
            )

    def _out(self, m: int, device) -> torch.Tensor:
        """A fresh output of ``m`` rows at the launches' row stride."""
        n = self.weight.shape[0]
        if self.out_stride is None:
            return torch.empty(m, n, dtype=torch.bfloat16, device=device)
        return torch.empty(m, self.out_stride, dtype=torch.bfloat16, device=device)[
            :, :n
        ]

    def takes(self, rows: int) -> bool:
        return rows in self.specs

    def __call__(
        self, hidden: torch.Tensor, out: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        from sgl_nvidia_mega.qwen3_5.sm103 import trt_lowm

        m = int(hidden.shape[0])
        s = self.specs[m]
        x = hidden if hidden.is_contiguous() else hidden.contiguous()
        if out is None:
            out = self._out(m, hidden.device)
        trt_lowm.run(s, out=out, x16=x, w_bf16=self.weight)
        return out


def maybe_build(parts) -> Optional[HeadLowM]:
    from sgl_nvidia_mega.qwen3_5.draft import MULTI_LOOP_BS
    from sgl_nvidia_mega.qwen3_5.runtime import slot_supported

    lm, b = parts.lm, parts.binding
    w = lm.lm_head.weight
    processor = lm.logits_processor
    reason = ""
    if parts.top1x is None:
        reason = "it serves slot draft_multi's steps"
    elif w.dtype != torch.bfloat16 or w.dim() != 2 or not w.is_contiguous():
        reason = f"LM head {w.dtype} {tuple(w.shape)}; needs a contiguous BF16 shard"
    elif processor.logit_scale is not None or getattr(
        processor, "final_logit_softcapping", None
    ):
        reason = "logit scaling or softcapping"
    if not slot_supported(SLOT, reason, cpu_group=getattr(b, "tp_cpu_group", None)):
        return None
    head = HeadLowM(w, tuple(MULTI_LOOP_BS), cpu_group=b.tp_cpu_group, rank=b.tp_rank)
    return head if head.specs else None
