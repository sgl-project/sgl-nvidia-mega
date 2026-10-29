# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 SGLang Team
"""CuTe DSL small-M NVFP4 MoE in the verify step of the opt-in fused GLM-5 DSA path.

At 4 / 8 / 12 / 16 verify rows (1..4 requests of 4 rows) each MoE layer runs the stock
router GEMM (``mlp.gate``), ``moe_route_top8`` (the top-8 selection of the stock TRT-LLM
NVFP4 MoE routing: sigmoid scores, correction bias, renormalized weights x the routed
scaling factor), one engine launch for the routed experts and the BF16 shared expert, and
``moe_finalize`` (the layer's un-reduced output, which the next hop step all-reduces as
it does the stock MoE output). When the layer's ``o_proj`` hop has already routed the
rows (``step.routing``), the router GEMM and ``moe_route_top8`` are skipped, the engine
runs without the finalize, and the step returns ``MoeRecords`` (the per-expert outputs)
for the MLP-output hop, which finalizes them inside its all-reduce.

The engine reads the server's processed routed FC1 weights in place. FC2 is read from
compact per-quarter records and the shared expert from a packed copy; both are derived
at load and registered (226.5 MB + 9.4 MB per MoE layer and rank). ``build`` keeps the
MoE stock unless the engine API accepts the configuration, every MoE layer has the
expected tensors and the server's own NVFP4 weight processing passes the engine's
layout self-test.
"""

from __future__ import annotations

import logging
import types
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import torch

from sgl_nvidia_mega.glm_dsa.eligibility import HIDDEN, MAX_ROWS
from sgl_nvidia_mega.glm_dsa.runtime.compile import compile_cached, cu_stream
from sgl_nvidia_mega.glm_dsa.moe.step import (
    MlpSteps,
    moe_split,
)

logger = logging.getLogger(__name__)

EXPERTS = 256
TOP_K = 8
INTER = 256  # routed and shared intermediate size per rank (TP 8)


from sgl_nvidia_mega.glm_dsa.moe.step import MoeRecords, MoeSteps, MlpSteps  # noqa: F401


class MoeGlue:
    """Derived MoE weights, compiled engines and per-row-count steps."""

    def __init__(self, api, layers: Dict[int, object], specs) -> None:
        self.api, self.specs, self.layers = api, specs, layers
        self.device = dev = next(iter(layers.values())).w13.device
        self.ids = torch.empty(16, TOP_K, dtype=torch.int32, device=dev)
        self.weights = torch.empty(16, TOP_K, dtype=torch.float32, device=dev)
        self.fn: Dict[str, object] = {}
        self.engine = None

    def compile(self, rows: Tuple[int, ...]) -> None:
        import cutlass.cute as cute

        modes = {self.api.mode_for(moe_split(m)) for m in rows if 1 <= m <= MAX_ROWS}
        for name in sorted({"route", "finalize"} | modes):
            kernel, args = self.specs[name]
            self.fn[name] = compile_cached(
                kernel,
                {"unit": name, "shared": True},
                lambda k=kernel, a=args: cute.compile(k, *a),
            )
        nclusters = self.api.default_clusters(self.device)
        self.engine = self.api.SmallMMoe(self.fn, nclusters, self.device, shared=True)

    def records(self, m: int) -> MoeRecords:
        sc = self.engine.scratch
        return MoeRecords(
            pq=sc.partials[: 4 * m * TOP_K * HIDDEN].view(4, m, TOP_K, HIDDEN),
            shpartials=sc.shared_partials[: 4 * m * HIDDEN].view(4, m, HIDDEN),
            ids_fin=sc.slot_ids[: m * TOP_K].view(m, TOP_K),
            w_fin=sc.slot_weights[: m * TOP_K].view(m, TOP_K),
            g2s=sc.slot_scales[: m * TOP_K].view(m, TOP_K),
        )

    def route(self, logits, bias, scale, ids, weights) -> None:
        from cutlass import Float32, Int32, Int64

        m = logits.shape[0]
        if (
            logits.dtype != torch.float32
            or tuple(logits.shape) != (m, EXPERTS)
            or not logits.is_contiguous()
        ):
            raise RuntimeError(
                f"fused verify MoE: router logits {tuple(logits.shape)} {logits.dtype}"
            )
        self.fn["route"](
            Int64(logits.data_ptr()),
            Int64(bias.data_ptr()),
            Float32(float(scale)),
            Int64(ids.data_ptr()),
            Int64(weights.data_ptr()),
            Int32(m),
            cu_stream(logits.device),
        )

    def for_rows(self, m: int) -> Optional[MoeSteps]:
        if (
            self.engine is None
            or not 1 <= m <= MAX_ROWS
            or self.api.mode_for(moe_split(m)) not in self.fn
        ):
            return None
        return MoeSteps(self, m)


def build(model, derived) -> Optional[MoeGlue]:
    from sgl_nvidia_mega.glm_dsa.kernels.cute.moe import api
    from sgl_nvidia_mega.glm_dsa.kernels.cute.moe import weights as W
    from sglang.srt.models.deepseek_v2 import DeepseekV2MoE

    def stock(why) -> None:
        logger.warning("GLM DSA fused decode: MoE stays stock (%s)", why)

    moe_layers = [l for l in model.model.layers if isinstance(l.mlp, DeepseekV2MoE)]
    if not moe_layers:
        return None
    for layer in moe_layers:
        why = _layer_reason(layer.mlp)
        if why is not None:
            return stock(f"layer {layer.layer_id}: {why}")
    try:
        specs = api.compile_specs(shared=True)
    except ValueError as e:
        return stock(e)
    dev = moe_layers[0].mlp.experts.w13_weight.device
    try:
        W.layout_self_test(_process_like_server, dev)
    except W.LayoutMismatch as e:
        return stock(f"layout self-test: {e}")
    layers = {}
    for layer in moe_layers:
        ex, sh = layer.mlp.experts, layer.mlp.shared_experts
        records = derived.add(
            "moe.fc2_records",
            W.fc2_records(
                _u8(ex.w2_weight.data).reshape(EXPERTS, HIDDEN, INTER // 2),
                _u8(ex.w2_weight_scale.data).reshape(EXPERTS, -1),
            ),
        )
        pack = derived.add(
            "moe.shared_pack",
            W.shared_pack(sh.gate_up_proj.weight.data, sh.down_proj.weight.data),
        )
        lt = api.LayerTensors(
            w13=ex.w13_weight.data,
            w13_scales=_u8(ex.w13_weight_scale.data).reshape(EXPERTS, -1),
            fc2_records=records.view(EXPERTS, 4, 4, -1),
            shared_pack=pack,
            g1_alpha=ex.g1_alphas.data,
            g2_alpha=ex.g2_alphas.data,
            gs1=float(ex.w13_input_scale_quant.item()),
            gs2=float(ex.w2_input_scale_quant.item()),
        )
        lt.check()
        layers[layer.layer_id] = lt
    logger.info("GLM DSA fused decode: MoE engine on %d layers", len(layers))
    return MoeGlue(api, layers, specs)
