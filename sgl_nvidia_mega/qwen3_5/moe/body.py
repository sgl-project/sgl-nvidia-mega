# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# Adapted from SGLang (Apache-2.0, SGLang Team):
# python/sglang/srt/layers/moe/moe_runner/flashinfer_trtllm.py
# (fused_experts_none_to_flashinfer_trtllm_fp4, its routed deferred-finalize branch).
"""The trtllm-gen NVFP4 MoE body on routing tables written by TEP8's internal
route_topk or route-big front. It replaces the runner's op, whose own routing kernel
(``routingIndicesDynBlock`` from logits or ``routingIndicesCluster`` from materialized
routing) builds the same tables before the same two GEMMs. Under expert parallelism the
tables cover the rank's local
experts (local_expert_offset = moe_ep_rank * num_local_experts, as SGLang's runner) and
a slot routed elsewhere gets permuted index -1, as the stock routing gives it.

The body is FlashInfer's DA entry (``MoERunner.forward`` with ``da_routing_metadata``): the
tactic the routed op would run (the autotuner's choice for the same op, inputs and tuning
config), FC1 and FC2 from the tables, and the routed op's deferred-finalize outputs
(FC2 rows, routing weights, expanded -> permuted map). FlashInfer's own table builder
(populate_routing_metadata_multi_tile) serves top-k <= 8 only, so the tables are ours.
Each width keeps one set of tables and body buffers for every layer: a layer's FC2 rows
are finalized at the next boundary, before the next layer's body writes them. The body
buffers (FC1 / FC2 outputs and GEMM scratch, sized by the width's row bound) are views of
one set every width shares, since one step runs at a time (TEP8 serves 96 widths).
"""

from __future__ import annotations

import logging
from typing import Optional

import torch

logger = logging.getLogger(__name__)

# The SGLang names this module reads (test/unit/test_contract.py pins them)
SGLANG_READS = {
    "srt/layers/moe/moe_runner/flashinfer_trtllm.py": (
        "get_activation_type",
        "get_activation_type(is_gated)",
        "trtllm_moe_enable_pdl",
    ),
    "srt/layers/moe/fused_moe_triton/layer.py": (
        "moe_runner_config",
        "num_experts",
        "num_local_experts",
        "moe_ep_rank",
        "top_k",
        "intermediate_size_per_partition",
        # the ring fill's per-layer record (variant slot moe_ring)
        "layer_id",
    ),
    "srt/utils/triton_utils.py|srt/utils/common.py": ("next_power_of_2",),
}
# The FlashInfer names this module reads (its DA body entry, flashinfer 0.7.1)
FLASHINFER_READS = {
    "flashinfer/fused_moe/core.py": (
        "get_trtllm_moe_sm100_module",
        "MoERunner",
        "_make_tuning_config",
        "forward(da_routing_metadata)",
        "forward(da_body_workspace)",
        "forward(do_preparation)",
    ),
    "flashinfer/fused_moe/shared/inputs.py": (
        "MoeRunnerInputs",
        "alloc_trtllm_moe_output",
    ),
    # moe/tactics.py: the table's key
    "flashinfer/autotuner/autotuner.py": ("AutoTuner._find_nearest_profile",),
    "flashinfer/fused_moe/backends/trtllm/sm100_runner.py": (
        "MoERunner.get_cache_key_extras",
    ),
    "flashinfer/version.py": ("__version__",),
}
OP = "flashinfer::trtllm_fp4_block_scale_moe"  # the routed op's autotuner entry


class Width:
    """One width's runner, routing tables and body buffers."""

    def __init__(self):
        self.runner = None
        self.tables = None  # sm103.moe.router.RouteTables of the tactic's tile_n
        self.tactic: Optional[list] = None
        self.workspace: Optional[tuple] = None
        self.engine: Optional[tuple] = (
            None  # the engine's (h, hsf, out) views, first use
        )


_WIDTHS: dict[tuple, Width] = {}
# the body buffers the widths share (FlashInfer's FP4 DA body order), grown to the largest asked for
_SHARED: list[torch.Tensor] = []


def _shared(prepared) -> tuple:
    """Views of the shared body buffers in place of a width's ``prepared`` ones (same dtype
    and row size, at least as many rows); a larger buffer replaces the shared one."""
    out = []
    for i, t in enumerate(prepared):
        if i == len(_SHARED):
            _SHARED.append(t)
        s = _SHARED[i]
        if s.dtype != t.dtype or s.shape[1:] != t.shape[1:] or s.shape[0] < t.shape[0]:
            _SHARED[i] = s = t
        out.append(s[: t.shape[0]])
    return tuple(out)


def local_offset(experts) -> int:
    """The rank's first local expert (SGLang's FP4 trtllm quant info)."""
    return experts.moe_ep_rank * experts.num_local_experts


def _kwargs(experts, m: int) -> dict:
    """The routed op's runner arguments for ``experts`` at m rows (SGLang's routed call:
    unpacked top-10, deferred finalize, no per-token scale)."""
    from flashinfer.tllm_enums import RoutingInputMode
    from sglang.srt.layers.moe.moe_runner.flashinfer_trtllm import (
        get_activation_type,
        trtllm_moe_enable_pdl,
    )

    return dict(
        routing_input_mode=RoutingInputMode.UnpackedPrecomputed,
        num_experts=experts.num_experts,
        routing_bias=None,
        gemm1_weights=experts.w13_weight.data,
        gemm1_weights_scale=experts.w13_weight_scale.data.view(torch.float8_e4m3fn),
        gemm1_bias=None,
        gemm1_alpha=None,
        gemm1_beta=None,
        gemm1_clamp_limit=None,
        gemm2_weights=experts.w2_weight.data,
        gemm2_weights_scale=experts.w2_weight_scale.data.view(torch.float8_e4m3fn),
        gemm2_bias=None,
        output1_scale_scalar=experts.g1_scale_c.data,
        output1_scale_gate_scalar=experts.g1_alphas.data,
        output2_scale_scalar=experts.g2_alphas.data,
        n_group=0,
        topk_group=0,
        local_expert_offset=local_offset(experts),
        routed_scaling_factor=None,
        routing_method_type=1,  # unused for precomputed routing (SGLang passes 1)
        enable_pdl=trtllm_moe_enable_pdl(m),
        do_finalize=False,
        activation_type=get_activation_type(
            experts.moe_runner_config.activation,
            is_gated=experts.moe_runner_config.is_gated,
        ),
        num_fused_shared_experts=0,
        norm_topk_prob=True,
        routing_replay_out=None,
        valid_hidden_size=None,
        valid_intermediate_size=None,
    )


def _inputs(tables, x_fp4: torch.Tensor, x_sf: torch.Tensor):
    """The runner's inputs (MoeRunnerInputs) as the routed op builds them."""
    from flashinfer.fused_moe.shared.inputs import (
        MoeRunnerInputs,
        alloc_trtllm_moe_output,
    )

    m = x_fp4.shape[0]
    return MoeRunnerInputs(
        output=alloc_trtllm_moe_output(m, 2 * x_fp4.shape[1], False, x_fp4.device),
        routing_logits=None,
        topk_ids=tables.ids,
        expert_weights=tables.weights,
        hidden_states=x_fp4,
        hidden_states_scale=x_sf,
        gemm1_lora_delta=None,
        per_token_scale=None,
    )


# FlashInfer's FP4 tile_n candidates (trtllm_fused_moe_kernel_launcher.cu mSupportedTileNums)
TILES = (8, 16, 32, 64, 128)


def default_tile(m: int, top_k: int, local_experts: int) -> int:
    """The tile_n the routed op runs without a tuned tactic (the launcher's
    selectDefaultTileN: the smallest of the candidates around the first tile that covers
    the mean rows an expert gets)."""
    mean = m * top_k / local_experts
    i = next((i for i, t in enumerate(TILES) if t >= mean), len(TILES) - 1)
    return TILES[max(i - 1, 0)]


def width(experts, x: torch.Tensor) -> Width:
    """The width of ``x`` [m, hidden]: on first use its runner, the routed op's tactic at
    m rows (the plugin table's, else the autotuner's; without one, FlashInfer's default,
    [-1, -1]) and that tile_n's tables (the body buffers follow on the first ``run``)."""
    from flashinfer.autotuner import AutoTuner
    from flashinfer.fused_moe.core import get_trtllm_moe_sm100_module
    from flashinfer.tllm_enums import (
        Fp8QuantizationType,
        RoutingInputMode,
        WeightLayout,
        deduce_trtllm_gen_tensor_dtype,
    )
    from sglang.srt.utils.triton_utils import next_power_of_2

    from sgl_nvidia_mega.qwen3_5.moe import tactics
    from sgl_nvidia_mega.qwen3_5.sm103.moe import router

    m = x.shape[0]
    window = (local_offset(experts), experts.num_local_experts)
    key = (m, experts.intermediate_size_per_partition) + window
    if key in _WIDTHS:
        return _WIDTHS[key]
    w = Width()
    module = get_trtllm_moe_sm100_module()
    kw = _kwargs(experts, m)
    x_fp4 = torch.empty((m, x.shape[1] // 2), dtype=torch.uint8, device=x.device)
    x_sf = torch.empty(
        (m, x.shape[1] // 16), dtype=torch.float8_e4m3fn, device=x.device
    )
    w.runner = module.MoERunner(
        module.moe_op,
        top_k=experts.top_k,
        num_local_experts=experts.num_local_experts,
        dtype_act=deduce_trtllm_gen_tensor_dtype(x_fp4, x_sf),
        dtype_weights=deduce_trtllm_gen_tensor_dtype(
            kw["gemm1_weights"], kw["gemm1_weights_scale"]
        ),
        fp8_quantization_type=Fp8QuantizationType.NoneFp8,
        hidden_size=x.shape[1],
        hidden_size_output=x.shape[1],
        intermediate_size=experts.intermediate_size_per_partition,
        activation_type=kw["activation_type"],
        weight_layout=WeightLayout.MajorK,
        use_shuffled_weight=True,
        use_per_token_scaling=False,
        num_experts=experts.num_experts,
        num_fused_shared_experts=0,
    )
    inputs = _inputs(router.RouteTables(m, 8, x.device, *window), x_fp4, x_sf)
    config = w.runner._make_tuning_config(
        inputs,
        tune_max_num_tokens=next_power_of_2(m),
        routing_input_mode=RoutingInputMode.UnpackedPrecomputed,
        hidden_states_scale_layout=None,
        use_cold_l2_cache=True,
        use_cuda_graph=True,
    )
    tactic = tactics.pick(m, w.runner, config, inputs.to_list())
    if tactic is None:
        _, tactic = AutoTuner.get().choose_one(
            OP, [w.runner], config, inputs.to_list(), **kw
        )
    if isinstance(tactic, int) or tactic[0] <= 0 or tactic[1] < 0:
        # the launcher's fallback: the default tile_n and its default config
        w.tactic = [-1, -1]
        tile = default_tile(m, experts.top_k, experts.num_local_experts)
        logger.info(
            "NVIDIA megakernel: TEP8 routed body runs the default tactic "
            "(tile_n %d) at %d rows",
            tile,
            m,
        )
    else:
        w.tactic = [int(tactic[0]), int(tactic[1])]
        tile = w.tactic[0]
    w.tables = router.RouteTables(m, tile, x.device, *window)
    _WIDTHS[key] = w
    return w


def run(w: Width, experts, x_fp4: torch.Tensor, x_sf: torch.Tensor):
    """The body on ``w``'s tables (written by the routing launch for these rows): the
    routed op's deferred-finalize outputs [FC2 rows, routing weights, expanded ->
    permuted map]."""
    if engine_serves(x_fp4.shape[0]):
        return run_engine(w, experts, x_fp4, x_sf)
    kw = _kwargs(experts, x_fp4.shape[0])
    inputs = _inputs(w.tables, x_fp4, x_sf).to_list()
    metadata = w.tables.metadata()
    if w.workspace is None:
        w.workspace = _shared(
            w.runner.forward(
                inputs,
                tactic=w.tactic,
                do_preparation=True,
                da_routing_metadata=metadata,
                **kw,
            )
        )
    return w.runner.forward(
        inputs,
        tactic=w.tactic,
        da_routing_metadata=metadata,
        da_body_workspace=w.workspace,
        **kw,
    )


ENGINE_OPTIONS = {"early_unit": 1}
RING_ROWS = (64, 128)
RING_STAGES = 3
RING_LAYERS = 128  # records: layer ids 0..RING_LAYERS-1
_ENGINE: dict = {}
_ENGINE_STATE: dict = {}
_ENGINE_SHARED: list[torch.Tensor] = []


def configure(engine: bool, ring: bool = False) -> None:
    _ENGINE.clear()
    _ENGINE.update(on=engine, ring=engine and ring, logged=set())


def _engine_config() -> dict:
    if "on" not in _ENGINE:
        configure(False)
    return _ENGINE


def engine_unsupported(experts) -> str:
    if experts.moe_runner_config.activation != "silu" or not getattr(
        experts.moe_runner_config, "is_gated", True
    ):
        return "SwiGLU experts only"
    w13, w2 = experts.w13_weight, experts.w2_weight
    if w13.dim() != 3 or w2.dim() != 3 or w13.shape[0] > 128:
        return "at most 128 local experts"
    hidden, inter = w2.shape[1], w13.shape[1] // 2
    if hidden % 256 or inter % 256:
        return f"hidden {hidden} / intermediate {inter} not multiples of 256"
    if w2.shape[2] * 2 != inter or w13.shape[2] * 2 != hidden:
        return "NVFP4 weights only"
    if inter != experts.intermediate_size_per_partition:
        return "padded intermediate"
    if torch.cuda.get_device_capability(w13.device)[0] != 10:
        return "SM100-family GPUs only"
    return ""


def engine_serves(m: int) -> bool:
    """Whether the engine runs the body at m rows (slot moe_engine built)."""
    return _engine_config()["on"] and m >= 1


def engine_n_max(m: int) -> int:
    return 16 if m <= 16 else 32 if m <= 224 else 64


def engine_tile_k(n_max: int) -> int:
    return 512 if n_max <= 32 else 256


def engine_rows(m: int, top_k: int, local: int, tile: int) -> int:
    expanded = m * top_k
    filled = min(local, expanded)
    return (filled + (expanded - filled) // tile) * tile


def _engine_buffers(w: Width, experts, m: int, device) -> tuple:
    if w.engine is not None:
        return w.engine
    if torch.cuda.is_current_stream_capturing():
        raise RuntimeError(
            "slot moe_engine: the MoE engine's buffers must exist before CUDA-graph "
            "capture (run one eager forward of this width first)"
        )
    hidden = experts.w2_weight.shape[1]
    inter = experts.w13_weight.shape[1] // 2
    rows = engine_rows(m, experts.top_k, experts.num_local_experts, w.tables.tile)
    want = (
        torch.empty((rows, inter // 2), dtype=torch.uint8, device=device),
        torch.empty((rows * inter // 16,), dtype=torch.uint8, device=device),
        torch.empty((rows, hidden), dtype=torch.bfloat16, device=device),
    )
    out = []
    for i, t in enumerate(want):
        if i == len(_ENGINE_SHARED):
            _ENGINE_SHARED.append(t)
        s = _ENGINE_SHARED[i]
        if s.dtype != t.dtype or s.shape[1:] != t.shape[1:] or s.shape[0] < t.shape[0]:
            _ENGINE_SHARED[i] = s = t
        out.append(s[: t.shape[0]])
    w.engine = tuple(out)
    return w.engine


def _engine_state(device):
    key = ("state", device.index)
    s = _ENGINE_STATE.get(key)
    if s is None:
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError(
                "slot moe_engine: the MoE engine's call state must exist before "
                "CUDA-graph capture"
            )
        from sgl_nvidia_mega.qwen3_5.runtime import persistent
        from sgl_nvidia_mega.qwen3_5.sm103.moe import q35_moe_engine

        with persistent.persistent(device):
            s = _ENGINE_STATE[key] = q35_moe_engine.new_state(device)
    return s


def _ring_records(device) -> torch.Tensor:
    key = ("ring_rec", device.index)
    r = _ENGINE_STATE.get(key)
    if r is None:
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError(
                "variant slot moe_ring: the MoE engine's ring records must exist "
                "before CUDA-graph capture"
            )
        from sgl_nvidia_mega.qwen3_5.runtime import persistent
        from sgl_nvidia_mega.qwen3_5.sm103.moe import q35_moe_engine

        with persistent.persistent(device):
            r = _ENGINE_STATE[key] = q35_moe_engine.new_record(device, RING_LAYERS)
    return r


def front_releases_early(experts, m: int, device) -> bool:
    from sgl_nvidia_mega.qwen3_5.moe import route

    kw = route.route_lean(m, experts.num_local_experts, device.index)
    return bool(kw and kw.get("et"))


def ring_opts(experts, m: int, opts: dict, device) -> tuple:
    cfg = _engine_config()
    if not cfg["ring"] or not RING_ROWS[0] <= m <= RING_ROWS[1]:
        return opts, None
    layer = getattr(experts, "layer_id", None)
    why = ""
    if not isinstance(layer, int) or not 0 <= layer < RING_LAYERS:
        why = f"layer id {layer!r} outside 0-{RING_LAYERS - 1}"
    elif not front_releases_early(experts, m, device):
        why = "route-big releases the engine at its end"
    if why:
        if ("ring-off", m) not in cfg["logged"]:
            cfg["logged"].add(("ring-off", m))
            logger.info(
                "NVIDIA megakernel: Qwen3.5 variant moe_ring is off at %d rows: %s",
                m,
                why,
            )
        return opts, None
    return dict(opts, ring=RING_STAGES, q0=1), _ring_records(device)[layer]


def run_engine(w: Width, experts, x_fp4: torch.Tensor, x_sf: torch.Tensor):
    from sgl_nvidia_mega.qwen3_5.moe import fc2ar
    from sgl_nvidia_mega.qwen3_5.sm103.moe import q35_moe_engine

    cfg = _engine_config()
    m = x_fp4.shape[0]
    t = w.tables
    h, hsf, out = _engine_buffers(w, experts, m, x_fp4.device)
    out = fc2ar.engine_out(out, m, t.tile)
    n_max = engine_n_max(m)
    opts = fc2ar.engine_opts(dict(ENGINE_OPTIONS), m)
    opts, rec = ring_opts(experts, m, opts, x_fp4.device)
    if m not in cfg["logged"]:
        cfg["logged"].add(m)
        logger.info(
            "NVIDIA megakernel: Qwen3.5 slot moe_engine runs at %d rows (tile_n %d, N %d, "
            "%d local experts, options %s%s)",
            m,
            t.tile,
            n_max,
            experts.num_local_experts,
            opts,
            ", ring fill from the layer's record" if rec is not None else "",
        )
    q35_moe_engine.q35_moe_engine(
        x_fp4.view(torch.uint8),
        x_sf.view(torch.uint8).reshape(m, -1),
        t.permuted_to_token,
        t.cta_expert,
        t.cta_row_limit,
        t.live_ctas,
        experts.w13_weight.data.view(torch.uint8),
        experts.w13_weight_scale.data.view(torch.uint8),
        experts.w2_weight.data.view(torch.uint8),
        experts.w2_weight_scale.data.view(torch.uint8),
        experts.g1_scale_c.data,
        experts.g1_alphas.data,
        experts.g2_alphas.data,
        h,
        hsf,
        out,
        _engine_state(x_fp4.device),
        tile=t.tile,
        n_max=n_max,
        tile_k=engine_tile_k(n_max),
        rec=rec,
        **opts,
    )
    return [out, t.weights, t.expanded_to_permuted]


_ROUTE_P2TP: list = []


def _route_p2tp(t, device) -> torch.Tensor:
    grid = torch.cuda.get_device_properties(device).multi_processor_count
    need = grid * (t.cta_expert.numel() + 1) * t.tile
    if not _ROUTE_P2TP or _ROUTE_P2TP[-1].numel() < need:
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError(
                "slot moe_engine_route: the engine's per-CTA tables must exist before "
                "CUDA-graph capture (run one eager forward of this width first)"
            )
        _ROUTE_P2TP.append(torch.empty((need,), dtype=torch.int32, device=device))
    return _ROUTE_P2TP[-1][:need]


ROUTED_TRIGGER = 3
TEST_ROUTE_DELAY_NS = 0


def run_engine_a(
    w: Width, experts, x_fp4: torch.Tensor, x_sf: torch.Tensor, cand: torch.Tensor
):
    from sgl_nvidia_mega.qwen3_5.moe import fc2ar
    from sgl_nvidia_mega.qwen3_5.sm103.moe import q35_moe_engine

    cfg = _engine_config()
    m = x_fp4.shape[0]
    t = w.tables
    h, hsf, out = _engine_buffers(w, experts, m, x_fp4.device)
    opts = dict(ENGINE_OPTIONS)
    if fc2ar.armed_at(m):
        out = fc2ar.engine_out(out, m, t.tile, ROUTED_TRIGGER)
        opts = dict(fc2ar.engine_opts(opts, m), early_trigger=ROUTED_TRIGGER)
    if TEST_ROUTE_DELAY_NS:
        opts["test_route_delay_ns"] = TEST_ROUTE_DELAY_NS
    p2tp = _route_p2tp(t, x_fp4.device)
    n_max = engine_n_max(m)
    if ("a", m) not in cfg["logged"]:
        cfg["logged"].add(("a", m))
        logger.info(
            "NVIDIA megakernel: Qwen3.5 slot moe_engine_route runs at %d rows (tile_n %d, "
            "N %d, %d local experts, options %s)",
            m,
            t.tile,
            n_max,
            experts.num_local_experts,
            opts,
        )
    q35_moe_engine.q35_moe_engine(
        x_fp4.view(torch.uint8),
        x_sf.view(torch.uint8).reshape(m, -1),
        t.permuted_to_token,
        t.cta_expert,
        t.cta_row_limit,
        t.live_ctas,
        experts.w13_weight.data.view(torch.uint8),
        experts.w13_weight_scale.data.view(torch.uint8),
        experts.w2_weight.data.view(torch.uint8),
        experts.w2_weight_scale.data.view(torch.uint8),
        experts.g1_scale_c.data,
        experts.g1_alphas.data,
        experts.g2_alphas.data,
        h,
        hsf,
        out,
        _engine_state(x_fp4.device),
        tile=t.tile,
        n_max=n_max,
        tile_k=engine_tile_k(n_max),
        **opts,
        cand=cand,
        ids=t.ids,
        wts=t.weights,
        e2p=t.expanded_to_permuted,
        total_padded=t.total_padded,
        m=m,
        local_start=t.local_offset,
        p2tp=p2tp,
    )
    return [out, t.weights, t.expanded_to_permuted]
