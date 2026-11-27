# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# Adapted from SGLang (Apache-2.0, SGLang Team):
# python/sglang/srt/models/qwen2_moe.py (Qwen2MoeSparseMoeBlock.forward,
# forward_normal_dual_stream and _forward_router_experts with defer_finalize).

from __future__ import annotations

import logging
from typing import Optional

import torch

logger = logging.getLogger(__name__)

SGLANG_READS = {
    "srt/models/qwen2_moe.py": (
        "gate",
        "topk",
        "experts",
        "shared_expert",
        "shared_expert_gate",
        "alt_stream",
        "supports_deferred_finalize",
        "_forward_shared_experts",
        "_gate_shared_output_out_of_place",
        "gate_up_proj",
        "down_proj",
        "act_fn",
    ),
    "srt/layers/moe/fused_moe_triton/layer.py": (
        "num_experts",
        "num_local_experts",
        "moe_ep_rank",
        "moe_runner_config",
        "forward_deferred_finalize",
    ),
    "srt/layers/quantization/unquant.py": ("w13_weight", "w2_weight"),
    "srt/layers/moe/moe_runner/base.py": (
        "routing_method_type",
        "routed_scaling_factor",
    ),
    "srt/layers/moe/topk.py": (
        "TopKOutputChecker",
        "topk_config",
        "top_k",
        "correction_bias",
        "num_expert_group",
        "topk_group",
    ),
    "srt/layers/moe/utils.py": ("RoutingMethodType", "sum_post_experts_output"),
    "srt/layers/moe/moe_runner/flashinfer_trtllm.py": (
        "_make_deferred_finalize_output",
    ),
    "srt/layers/layer_boundary/fusions/moe_finalize.py": (
        "MoeDeferredFinalize",
        "from_flashinfer",
    ),
    "srt/layers/linear.py": ("reduce_results",),
    "srt/model_executor/runner/__init__.py": ("get_is_capture_mode",),
    "srt/runtime_context.py": ("get_parallel", '"tp_size"', '"moe_ep_size"'),
    "srt/kernel_plugins/nvidia/eagle_binding.py|srt/megakernels/eagle_binding.py": (
        "tp_rank",
        "tp_cpu_group",
    ),
}

SGLANG_MIRRORS = {
    "srt/models/qwen2_moe.py:Qwen2MoeSparseMoeBlock.forward": (
        "633ff9a686d3ea28",
        "80c4b54b62df6e29",
    ),
    "srt/models/qwen2_moe.py:Qwen2MoeSparseMoeBlock.forward_normal_dual_stream": "afae959064cc8cd7",
    "srt/models/qwen2_moe.py:Qwen2MoeSparseMoeBlock._forward_router_experts": (
        "32d09f01b77d76b5",
        "251e3d343aec92fe",
    ),
}

ENGINE = "draft_moe_engine"
WIDE = "draft_moe_wide"
SHARED = "draft_shared_lowm"
ROUTER = "draft_router_lowm"
# rows the engine routes in its prologue (one token bit a row)
MAX_ROWS = 32
WIDE_ROWS = (MAX_ROWS + 1, 128)
# the wide build's segment rows (MMA N)
WIDE_N_MAX = 64
SHARED_ROWS = (2, 16)
SHARED_T1 = (8, 2, 4, 0, 1)
SHARED_T2 = (2, 1, 1, 0, 1)
ROUTER_RACE_ROWS = 16
ROUTER_CANDIDATES = (
    (8, 1, 1, 0, 1),
    (4, 1, 4, 0, 1),
    (8, 2, 2, 0, 1),
    (8, 1, 2, 0, 1),
)
# (option or slot, rows) whose capture logged the engagement line
_NOTED: set = set()


def _note(what: str, rows: int) -> None:
    if torch.cuda.is_current_stream_capturing() and (what, rows) not in _NOTED:
        _NOTED.add((what, rows))
        logger.info("NVIDIA megakernel: %s runs the draft MoE at %d rows", what, rows)


# ------------------------------------------------------------------- the engine


def engine_unsupported(experts, topk) -> str:
    from sglang.srt.layers.moe.utils import RoutingMethodType

    runner = experts.moe_runner_config
    cfg = topk.topk_config
    if runner.routing_method_type not in (
        RoutingMethodType.Renormalize,
        RoutingMethodType.RenormalizeNaive,
    ):
        return f"routing {runner.routing_method_type}"
    if cfg.correction_bias is not None:
        return "a routing bias"
    if cfg.num_expert_group not in (None, 0, 1) or cfg.topk_group not in (None, 0, 1):
        return "grouped routing"
    if runner.routed_scaling_factor not in (None, 1.0):
        return f"routed scaling {runner.routed_scaling_factor}"
    if (
        cfg.top_k != 10
        or experts.num_experts != 512
        or experts.num_local_experts != 128
    ):
        return (
            f"top-{cfg.top_k} of {experts.num_experts} experts, "
            f"{experts.num_local_experts} local"
        )
    w13, w2 = experts.w13_weight, experts.w2_weight
    if (
        w13.dtype != torch.bfloat16
        or w2.dtype != torch.bfloat16
        or w13.dim() != 4
        or w2.dim() != 4
    ):
        return f"expert weights {w13.dtype} {tuple(w13.shape)}; needs BlockMajorK BF16"
    e, hb, two_i, kb = w13.shape
    hidden, inter = hb * kb, two_i // 2
    if (
        kb != 64
        or e != 128
        or tuple(w2.shape) != (e, inter // 64, hidden, 64)
        or hidden % 128
        or inter % 128
    ):
        return f"expert weights {tuple(w13.shape)} / {tuple(w2.shape)}"
    return ""


class DraftMoeEngine:
    """The engine's call state and compiled variants for one draft MoE layer's experts."""

    def __init__(self, experts, *, wide: bool):
        from sgl_nvidia_mega.qwen3_5.runtime import persistent
        from sgl_nvidia_mega.qwen3_5.sm103.draft import draftmoe_engine as kernel

        self.experts = experts
        self.w13, self.w2 = experts.w13_weight, experts.w2_weight
        self.local_start = experts.moe_ep_rank * experts.num_local_experts
        self.hidden = int(self.w2.shape[2])
        self.inter = int(self.w13.shape[2]) // 2
        self.wide = wide
        device = self.w13.device
        with persistent.persistent(device):
            self.state = kernel.new_state(device)
        # every variant compiled now, on zero rows (no compile inside a capture)
        for m in (1, 17) + ((WIDE_ROWS[0],) if wide else ()):
            x = torch.zeros(m, self.hidden, dtype=torch.bfloat16, device=device)
            logits = torch.zeros(m, 512, dtype=torch.bfloat16, device=device)
            self.run(x, logits)
        torch.cuda.synchronize(device)

    def takes(self, x: torch.Tensor, logits: torch.Tensor) -> bool:
        m = int(x.shape[0])
        rows_ok = 1 <= m <= MAX_ROWS or (
            self.wide and WIDE_ROWS[0] <= m <= WIDE_ROWS[1]
        )
        return (
            rows_ok
            and x.dtype == torch.bfloat16
            and x.dim() == 2
            and x.shape[1] == self.hidden
            and x.stride(1) == 1
            and (x.stride(0) * 2) % 16 == 0
            and logits.dtype == torch.bfloat16
            and tuple(logits.shape) == (m, 512)
            and logits.is_contiguous()
        )

    def run(self, x: torch.Tensor, logits: torch.Tensor) -> list:
        from sgl_nvidia_mega.qwen3_5.sm103.draft.draftmoe_engine import (
            TOPK,
            q35_draft_moe_engine,
        )

        m = int(x.shape[0])
        dev = x.device
        wide = m > MAX_ROWS
        h = torch.empty((m * TOPK, self.inter), dtype=torch.bfloat16, device=dev)
        out = torch.empty((m * TOPK, self.hidden), dtype=torch.bfloat16, device=dev)
        wts = torch.empty((m, TOPK), dtype=torch.bfloat16, device=dev)
        e2p = torch.empty((m, TOPK), dtype=torch.int32, device=dev)
        q35_draft_moe_engine(
            x,
            logits,
            self.w13,
            self.w2,
            h,
            out,
            wts,
            e2p,
            self.state,
            local_start=self.local_start,
            n_max=WIDE_N_MAX if wide else (16 if m <= 16 else 32),
            m_max=WIDE_ROWS[1] if wide else MAX_ROWS,
        )
        # FlashInfer's shapes: the index flat [m * top_k]
        return [out, wts, e2p.view(-1)]


def shared_unsupported(mlp) -> str:
    se = getattr(mlp, "shared_expert", None)
    if se is None:
        return "no shared expert"
    gu, dn = getattr(se, "gate_up_proj", None), getattr(se, "down_proj", None)
    if gu is None or dn is None:
        return "the shared expert is not a gate_up / down MLP"
    for name, lin in (("gate_up", gu), ("down", dn)):
        method = getattr(lin, "quant_method", None)
        if method is not None and type(method).__name__ != "UnquantizedLinearMethod":
            return f"{name} runs {type(method).__name__}"
        w = getattr(lin, "weight", None)
        if (
            w is None
            or w.dtype != torch.bfloat16
            or w.dim() != 2
            or not w.is_contiguous()
        ):
            return f"{name} weight is not a contiguous BF16 matrix"
        if getattr(lin, "bias", None) is not None:
            return f"{name} has a bias"
    if getattr(dn, "reduce_results", False):
        return "down_proj all-reduces its own output"
    two_i, k = gu.weight.shape
    h, i = dn.weight.shape
    if two_i != 2 * i or h != k or k % 64 or i % 64:
        return f"shapes gate_up {tuple(gu.weight.shape)} down {tuple(dn.weight.shape)}"
    if type(getattr(se, "act_fn", None)).__name__ != "SiluAndMul":
        return "the activation is not SiluAndMul"
    return ""


def paired_gate_up(w: torch.Tensor) -> torch.Tensor:
    two_i, k = w.shape
    i = two_i // 2
    return (
        torch.stack((w[:i].reshape(i // 8, 8, k), w[i:].reshape(i // 8, 8, k)), dim=1)
        .reshape(two_i, k)
        .contiguous()
    )


class DraftSharedLowM:
    def __init__(self, mlp, rows: tuple = SHARED_ROWS):
        from sgl_nvidia_mega.qwen3_5.draft.head_lowm import PDL
        from sgl_nvidia_mega.qwen3_5.sm103 import trt_lowm

        se = mlp.shared_expert
        self.w_gu = paired_gate_up(se.gate_up_proj.weight.detach())
        self.w_dn = se.down_proj.weight.detach()
        self.hidden, self.inter = self.w_dn.shape
        self.specs: dict = {}
        for m in range(rows[0], rows[1] + 1):
            s1 = trt_lowm.spec(
                m=m,
                k=self.hidden,
                n_bf16=2 * self.inter,
                fp8_act="bf16",
                tactic_bf16=SHARED_T1,
                swiglu_bf16=True,
                **PDL,
            )
            s2 = trt_lowm.spec(
                m=m,
                k=self.inter,
                n_bf16=self.hidden,
                fp8_act="bf16",
                tactic_bf16=SHARED_T2,
                **PDL,
            )
            trt_lowm.prepare(s1)
            trt_lowm.prepare(s2)
            self.specs[m] = (s1, s2)

    def takes(self, x: torch.Tensor) -> bool:
        return (
            x.dim() == 2
            and x.dtype == torch.bfloat16
            and int(x.shape[0]) in self.specs
            and x.shape[1] == self.hidden
            and x.is_contiguous()
            and x.data_ptr() % 16 == 0
        )

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """The ungated shared output [m, hidden] bf16 of ``x`` [m, hidden] bf16."""
        from sgl_nvidia_mega.qwen3_5.sm103 import trt_lowm

        m = int(x.shape[0])
        s1, s2 = self.specs[m]
        h = torch.empty(m, self.inter, dtype=torch.bfloat16, device=x.device)
        out = torch.empty(m, self.hidden, dtype=torch.bfloat16, device=x.device)
        trt_lowm.run(s1, out=h, x16=x, w_bf16=self.w_gu)
        trt_lowm.run(s2, out=out, x16=h, w_bf16=self.w_dn)
        return out


# ------------------------------------------------------------------- the block


class DraftMoeBlock:
    """The draft MoE block's deferred-finalize forward with the pieces built for it."""

    def __init__(self, mlp, *, engine=None, shared=None, router=None):
        self.mlp = mlp
        self.engine: Optional[DraftMoeEngine] = engine
        self.shared: Optional[DraftSharedLowM] = shared
        self.router = router  # head_lowm.HeadLowM on the gate's weight

    def deferred(self, x: torch.Tensor):
        from sglang.srt.layers.layer_boundary.fusions.moe_finalize import (
            MoeDeferredFinalize,
        )
        from sglang.srt.layers.moe.utils import sum_post_experts_output
        from sglang.srt.model_executor.runner import get_is_capture_mode

        mlp = self.mlp
        rows, hidden = x.shape
        x = x.view(-1, hidden)
        if rows == 0:
            raise RuntimeError("Qwen deferred finalize does not support M=0")
        if (
            mlp.alt_stream is not None
            and get_is_capture_mode()
            and not torch.compiler.is_compiling()
        ):
            current = torch.cuda.current_stream()
            mlp.alt_stream.wait_stream(current)
            # the routed FC2 stays on the current stream, the finalize's PDL dependency
            routed = self._router_experts(x)
            with torch.cuda.stream(mlp.alt_stream):
                shared = self._shared(x)
            current.wait_stream(mlp.alt_stream)
        else:
            shared = self._shared(x)
            routed = self._router_experts(x)
        if shared is None:
            raise RuntimeError("Qwen deferred finalize requires shared output")
        return MoeDeferredFinalize.from_flashinfer(
            routed,
            gated_shared_output=shared,
            m=rows,
            reduce=sum_post_experts_output,
        )

    def _shared(self, x: torch.Tensor):
        mlp = self.mlp
        if self.shared is not None and self.shared.takes(x):
            _note(SHARED, int(x.shape[0]))
            shared = self.shared(x)
        else:
            shared = mlp._forward_shared_experts(x, apply_gate=False)
        if shared is not None:
            shared = mlp._gate_shared_output_out_of_place(x, shared)
        return shared

    def _router_experts(self, x: torch.Tensor):
        from sglang.srt.layers.moe.moe_runner.flashinfer_trtllm import (
            _make_deferred_finalize_output,
        )
        from sglang.srt.layers.moe.topk import TopKOutputChecker

        mlp = self.mlp
        rows = int(x.shape[0])
        if self.router is not None and self.router.takes(rows):
            _note(ROUTER, rows)
            router_logits = self.router(x)
        else:
            router_logits, _ = mlp.gate(x)
        if self.engine is not None and self.engine.takes(x, router_logits):
            _note(ENGINE if rows <= MAX_ROWS else WIDE, rows)
            return _make_deferred_finalize_output(
                self.engine.run(x, router_logits), top_k=10
            )
        topk_output = mlp.topk(x, router_logits)
        if not mlp.supports_deferred_finalize:
            raise RuntimeError(
                "Qwen deferred finalize requires a compatible FlashInfer TRTLLM MoE "
                "producer and a separate shared expert"
            )
        if not TopKOutputChecker.format_is_bypassed(topk_output):
            raise RuntimeError(
                "Qwen deferred finalize requires logits-based bypassed TopK"
            )
        return mlp.experts.forward_deferred_finalize(x, topk_output)


def _tep8() -> bool:
    from sglang.srt.runtime_context import get_parallel

    parallel = get_parallel()
    return parallel.tp_size == 8 and parallel.moe_ep_size > 1


def wanted(slots) -> bool:
    from sgl_nvidia_mega.qwen3_5 import runtime

    return ROUTER in slots or any(
        runtime.measured_option(name) for name in (ENGINE, SHARED)
    )


def build(parts) -> Optional[DraftMoeBlock]:
    from sgl_nvidia_mega.qwen3_5 import runtime
    from sgl_nvidia_mega.qwen3_5.draft import head_lowm

    mlp = parts.layer.mlp
    b = parts.binding
    cpu_group = getattr(b, "tp_cpu_group", None)
    deferred = getattr(mlp, "supports_deferred_finalize", False)
    engine = shared = router = None
    if runtime.measured_option(ENGINE):
        reason = "" if _tep8() else "not TEP8"
        if not reason and not deferred:
            reason = "the draft MoE does not defer its finalize"
        reason = reason or engine_unsupported(mlp.experts, mlp.topk)
        if runtime.slot_supported(ENGINE, reason, cpu_group=cpu_group):
            engine = DraftMoeEngine(mlp.experts, wide=runtime.measured_option(WIDE))
    if runtime.measured_option(SHARED):
        reason = "" if _tep8() else "not TEP8"
        reason = reason or shared_unsupported(mlp)
        if runtime.slot_supported(SHARED, reason, cpu_group=cpu_group):
            shared = DraftSharedLowM(mlp)
    if ROUTER in parts.slots:
        gate = getattr(mlp, "gate", None)
        w = getattr(gate, "weight", None)
        reason = ""
        if (
            w is None
            or w.dtype != torch.bfloat16
            or w.dim() != 2
            or not w.is_contiguous()
        ):
            reason = "the draft MoE gate is not a contiguous BF16 linear"
        elif getattr(gate, "bias", None) is not None:
            reason = "the draft MoE gate has a bias"
        elif not deferred:
            reason = "the draft MoE does not defer its finalize"
        if runtime.slot_supported(ROUTER, reason, cpu_group=cpu_group):
            race = head_lowm.HeadLowM(
                w,
                tuple(range(1, ROUTER_RACE_ROWS + 1)),
                cpu_group=b.tp_cpu_group,
                rank=b.tp_rank,
                max_rows=MAX_ROWS,
                candidates=ROUTER_CANDIDATES,
                label=ROUTER,
            )
            router = race if race.specs else None
        if router is None:
            parts.slots = parts.slots - {ROUTER}
    if engine is None and shared is None and router is None:
        return None
    logger.info(
        "NVIDIA megakernel: Qwen3.5 MTP draft MoE block built (engine %s, shared "
        "expert %s, router %s)",
        "1-%d rows%s" % (MAX_ROWS, ", wide %d-%d" % WIDE_ROWS if engine.wide else "")
        if engine is not None
        else "FlashInfer's",
        "low-M at %d-%d rows" % SHARED_ROWS if shared is not None else "SGLang's",
        f"low-M at {sorted(router.specs)} rows" if router is not None else "SGLang's",
    )
    return DraftMoeBlock(mlp, engine=engine, shared=shared, router=router)
