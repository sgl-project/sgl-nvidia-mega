# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Component ``mid_rs`` (env ``MUSE_VR_MID_RS=1``, default off): the MoE half of target-verify passes on owned rows.

Tensor-parallel decode replicates every token row on every rank between the attention output projection and the next
layer's attention input. On a target-verify pass this component keeps the rows owned instead: rank ``r`` of ``W`` owns
rows ``[M*r/W, M*(r+1)/W)`` of the pass's ``M`` rows, and for every engaged layer runs

* the attention exactly as served, with the output projection's all-reduce skipped (its partial sums stay local);
* a reduce-scatter of the partials to the owners (``ops.c3_hops.rs_rows``) and the attention boundary (post-attention norm,
  gated residual, pre-FFN norm, router input) on the owned rows, with the launch configuration SGLang picks for ``M``
  rows, so every row is computed as the replicated boundary computes it;
* the pooler GEMM and the router GEMM on the owned rows (library calls; the router on the layer's side stream);
* the front launch (``ops.wfc_c4_fr.run_c3``: top-8 routing, pre-expert norm, NVFP4 quantization) on the owned rows, which
  publishes every routed row and its ids to every rank, and the landing unpack (``ops.c3_hops.fr_unpack``);
* the served expert stage on all ``M`` rows (``MuseSpark1xFp4Experts.forward`` with the prepared input and packed ids, so the
  routed-MoE call takes the same kernel the replicated path takes) and its finalize without the all-reduce;
* a reduce-scatter of the expert partials to the owners, the unpooler GEMM on the owned rows, the FFN boundary on the owned
  rows (``M``-row launch configuration as above), and an all-gather of the next layer's attention input
  (``ops.c3_hops.ag_rows_tagged``).

The residual stream stays owned between engaged layers. Full rows are gathered only where something outside the span reads
them: at the draft model's capture layers (``layers_to_capture``, bf16) and after the last engaged layer (fp32). The
engaged span is the contiguous run of NVFP4 MoE layers whose experts the ``moe_ld`` component prepared, without the model's
last layer; the other layers run their own (served) forward.

A pass engages only when every rank takes the same decision from host-side batch state: a target-verify pass, not sequence
parallel, ``MUSE_VR_MID_RS_MIN_M`` (default 17) <= M <= the arena's row capacity, ``M`` divisible by the tensor-parallel
size, a supported group size, and no earlier error. Every other pass runs the served path unchanged and is counted
(``bed.count(NAME, "step_aside:<reason>")``, one WARNING per reason per process).

Hop state: one ``ops.lb_hop.HopArena`` per process (unique sites per engaged layer), allocated collectively on the
tensor-parallel group right before CUDA graphs are captured (after the KV pool is sized), scrubbed after capture. The
forward's sequence word is raised by a node at the head of every engaged forward (captured with the forward); the host
counts forwards for the tag-wrap scrub before every runner forward and every eager engaged forward.

Errors: every bounded wait that expires sets the arena's sticky error word (and traps when the arena is built with trap,
``MUSE_VR_MID_RS_TRAP``, default on). The word is checked on the host after a decode step's results are synchronized and
before they are processed, and before every runner forward; a nonzero word raises (fail-stop: that step fails, and the
component refuses further engagement in eager passes; captured graphs are not re-captured, so the raise is the stop).

Development only: ``MUSE_VR_MID_RS_DEV_DECODE=1`` also engages plain decode passes (servers without a draft model), and
``MUSE_VR_MID_RS_WORLDS`` adds tensor-parallel sizes other than 4; neither belongs in a measured configuration.

Numerics: the reduce-scatters fold fp32 partials in rank order; the owner-row GEMMs are library calls on fewer rows than the
replicated path's; see the component's numerics record for the class of each step.
"""

from __future__ import annotations

import logging
import os
import threading

import torch

from . import bed

# The SGLang names this module reads (test/unit/test_contract.py pins them against the supported SGLang)
SGLANG_READS = {
    "kernels/ops/layernorm/muse_spark_norm_residual_jit.py": (
        "fused_norm_residual",
        "norm_residual_launch_config",
    ),
    "srt/layers/quantization/unquant.py": ("bf16_gemm_dispatch",),
    "srt/runtime_context.py": ("get_forward",),
    "srt/layers/moe/utils.py": ("should_skip_mlp_all_reduce",),
    "srt/state_capturer/routed_experts.py": ("get_global_experts_capturer",),
    "srt/distributed/*.py": ("get_tp_group",),
    "srt/distributed/device_communicators/custom_all_reduce_v2.py": (
        "CustomAllReduceV2",
    ),
    "srt/layers/dp_attention.py": ("is_dp_attention_enabled",),
    "srt/plugins/hook_registry.py": ("HookRegistry", "HookType"),
}


logger = logging.getLogger(__name__)

NAME = "mid_rs"  # env MUSE_VR_MID_RS
T_LAYER = "sglang.srt.models.muse_spark.MuseSpark1xDecoderLayer.forward"
T_MODEL = "sglang.srt.models.muse_spark.MuseSpark1xModel._forward"
T_RUNNER_FWD = "sglang.srt.model_executor.model_runner.ModelRunner.forward"
T_GRAPHS = "sglang.srt.model_executor.model_runner.ModelRunner.init_cuda_graphs"
T_RESULT = (
    "sglang.srt.managers.scheduler_components.batch_result_processor."
    "SchedulerBatchResultProcessor.process_batch_result_decode"
)

H, EH, TOPK, EXPERTS = 8192, 4096, 8, 256
TOK_FP4, TOK_SF = (
    EH // 2,
    256,
)  # NVFP4 row bytes of the pooled (4096-wide) row, its block scales
NVFP4_PER_TOKEN_SCALE_INV = 1.0 / (448.0 * 6.0)
ENV_MIN_M = "MUSE_VR_MID_RS_MIN_M"
ENV_CTAS = "MUSE_VR_MID_RS_CTAS"  # consumer units (launch grid) of every site (default min(SMs, 128))
ENV_RS_CTAS = "MUSE_VR_MID_RS_RS_CTAS"  # the reduce-scatter sites' grid only (default: MUSE_VR_MID_RS_CTAS)
ENV_AG_CTAS = "MUSE_VR_MID_RS_AG_CTAS"
ENV_TRAP = "MUSE_VR_MID_RS_TRAP"
ENV_WORLDS = "MUSE_VR_MID_RS_WORLDS"  # development only: extra group sizes (the served form is 4)
ENV_HEADS = "MUSE_VR_MID_RS_HEADS"  # forwards counted per runner forward (a replayed graph may hold more than one)
ENV_MEM_RESERVE_GB = "MUSE_VR_MID_RS_MEM_RESERVE_GB"  # free memory left after the hop state (default 8 GiB)
ENV_DEV_DECODE = "MUSE_VR_MID_RS_DEV_DECODE"  # development only: also engage plain decode passes (servers without a draft)
SERVED_WORLDS = (4,)

# sha256 of the SGLang sources whose behaviour the owned-row path reproduces (main layout; the device-gate copy of
# muse_spark.py is accepted as the stock-path copy both arms may load).
PINS = {
    "sglang/srt/models/muse_spark.py": (
        "8bafbb8802fb0d9f81c190e87b9012fca546182be08bef6c8b3db814d27bfe19",
        "c0db42467a960c509b658c3c40570ee889f73d41b52cdbd80b9c291c1792a33f",
    ),
    "sglang/kernels/ops/layernorm/muse_spark_norm_residual_jit.py": "454f084d20e5f2353a70d260ea4a70c7dfdff3bf45a58619ef72e1eec2646fe3",
    "sglang/kernels/ops/layernorm/muse_spark_norm_residual.py": "76bd0c26ed4d3cf87126619c72a56de908dea0ea1b2c69f6e5ada47318035cbe",
    "sglang/srt/layers/muse_spark/muse_spark_fp4_moe.py": "1d7676d6d086e6aa45139671e345ef32d28a81b7de1b50810d0e512ee95cc738",
    "sglang/srt/layers/linear.py": "3ce31df0a8aab09ed8afa0ada8c0ce30ca207a748c02c3073d7f53648b9b7c3c",
    "sglang/srt/distributed/device_communicators/custom_all_reduce_v2.py": (
        "277da88e2837e585f3ebdf4122906d29ea7a96cc30b981f7a028f6f63b6cf001",
        "f7b23a55019734be7b2f1958e20cc1f78a31251a035f2d75885911aa4a46e6b5",
    ),
    "sglang/srt/managers/scheduler_components/batch_result_processor.py": "4867c24b439c9a26718edafdfc7ab0fa682b1597a91c0033f4facfbea896ad80",
}

_tls = threading.local()
_lock = threading.Lock()
_state: dict = {
    "st": None,
    "off": None,
    "seen": set(),
    "runner": None,
    "min_m": 17,
    "worlds": SERVED_WORLDS,
    "heads": 2,
    "reason": None,
    "dev_decode": False,
}


# ------------------------------------------------------------------------------------------------ decisions
def enabled() -> bool:
    return bed.enabled(NAME)


def _aside(reason: str) -> None:
    bed.count(NAME, "step_aside:" + reason)
    with _lock:
        first = reason not in _state["seen"]
        _state["seen"].add(reason)
    if first:
        logger.warning(
            "muse_vr_plugin: mid_rs steps aside to the served path: %s (logged once per reason; counted)",
            reason,
        )


def turn_off(reason: str) -> None:
    with _lock:
        if _state["off"] is None:
            _state["off"] = reason
            logger.error(
                "muse_vr_plugin: mid_rs turned OFF for this process: %s", reason
            )


def decide(
    mode_is_target_verify: bool,
    sp: bool,
    M: int,
    world: int,
    rows_max: int,
    ready: bool,
) -> str | None:
    """None to engage, else the step-aside reason. A pure function of host-side batch state and process state, so every
    rank decides the same way and a captured graph's decision equals its replays'."""
    if _state["off"] is not None:
        return "off_after_error"
    if not ready:
        return "not_ready"
    if not mode_is_target_verify:
        return "not_target_verify"
    if sp:
        return "sequence_parallel"
    if world not in _state["worlds"]:
        return f"world_{world}"
    if M % world:
        return "rows_not_divisible"
    if M < _state["min_m"]:
        return f"rows_below_{_state['min_m']}"
    if M > rows_max:
        return "rows_above_capacity"
    return None


def owned(M: int, world: int, rank: int) -> tuple[int, int]:
    return (M * rank) // world, (M * (rank + 1)) // world


def state_bytes(
    world: int, rows_max: int, layers: int, taps: int, ctas: int = 128
) -> int:
    """Device bytes the hop state takes (sites + landings + output buffers), without allocating anything."""
    from .ops import c3_hops as c3
    from .ops import wfc_c4_fr as fm

    per_layer = [
        c3.rs_spec("a", rows_max, H, ctas),
        c3.rs_spec("b", rows_max, EH, ctas),
        c3.agx_spec("x", rows_max, H, ctas),
        *fm.site_specs(rows_max, "i", "t"),
    ]
    sites = layers * sum(sp.region_bytes(world) for sp in per_layer)
    landings = (taps * rows_max * H + rows_max * 2 * H) * 2
    buffers = 2 * rows_max * (H * 2 + TOK_FP4 + TOK_SF + 4 + TOPK * 4) + rows_max * 2
    return int(sites + landings + buffers)


def _agree(group, ok: bool) -> bool:
    """True when every rank of the group passes ok=True (a MIN all-reduce on the group's own backend)."""
    import torch.distributed as dist

    dev = (
        torch.device("cpu")
        if dist.get_backend(group) == "gloo"
        else torch.device("cuda", torch.cuda.current_device())
    )
    t = torch.tensor([1 if ok else 0], dtype=torch.int32, device=dev)
    dist.all_reduce(t, op=dist.ReduceOp.MIN, group=group)
    return bool(int(t.item()))


def _layer_shape_reason(layer) -> str | None:
    """Why this decoder layer cannot take the owned-row path (shapes, dtype, projection reduce form), or None."""
    mlp = layer.mlp
    o_proj = layer.self_attn.o_proj
    if not (
        getattr(o_proj, "reduce_results", False)
        and getattr(o_proj, "use_dp_attention_reduce", True)
    ) or getattr(o_proj, "use_decode_attn_tp", False):
        return "o_proj reduce configuration"
    if getattr(layer, "operator_dtype", None) is not torch.bfloat16:
        return "operator dtype is not bf16"
    if (
        getattr(mlp, "hidden_size", None),
        getattr(mlp, "expert_hidden_size", None),
        getattr(mlp, "top_k", None),
        getattr(mlp, "num_experts", None),
    ) != (H, EH, TOPK, EXPERTS):
        return "model sizes differ from the built ones"
    if tuple(mlp.pre_expert_proj.weight.shape) != (EH, H) or tuple(
        mlp.post_expert_proj.weight.shape
    ) != (H, EH):
        return "pooler / unpooler shapes"
    return None


# ------------------------------------------------------------------------------------------------ state
class _State:
    """Per-process hop state for the engaged span (built collectively on the tensor-parallel group)."""

    def __init__(
        self,
        group,
        device,
        rank: int,
        world: int,
        rows_max: int,
        layers: list,
        taps: tuple,
        zs_comm,
        *,
        lm=None,
        ctas: int | None = None,
        ag_ctas: int | None = None,
        trap: bool | None = None,
    ):
        """group: the process group the arena's symmetric allocation and barriers use; zs_comm: the served
        tensor-parallel communicator whose all-reduce sign-of-zero rule the reduce-scatters reproduce; lm: the language
        model whose forwards engage (None in harness use)."""
        from .ops import c3_hops as c3
        from .ops import wfc_c4_fr as fm
        from .ops.lb_hop import HopArena

        self.c3, self.fm = c3, fm
        self.lm = lm
        self.comm = zs_comm
        self.world, self.rank, self.device = int(world), int(rank), device
        self.rows_max = int(rows_max)
        self.layers = list(layers)
        self.first, self.last = self.layers[0], self.layers[-1]
        self.span = set(self.layers)
        self.taps_all = tuple(taps)
        self.taps = tuple(t for t in taps if t in self.span)
        sm = torch.cuda.get_device_properties(self.device).multi_processor_count
        self.ctas = int(ctas) if ctas else bed.env_int(ENV_CTAS, min(int(sm), 128))
        self.ag_ctas = int(ag_ctas) if ag_ctas else bed.env_int(ENV_AG_CTAS, 64)
        self.rs_ctas = bed.env_int(ENV_RS_CTAS, self.ctas)
        specs = []
        for l in self.layers:
            specs.append(c3.rs_spec(f"rsA{l}", self.rows_max, H, self.rs_ctas))
            specs.append(c3.rs_spec(f"rsB{l}", self.rows_max, EH, self.rs_ctas))
            specs.append(c3.agx_spec(f"agX{l}", self.rows_max, H, self.ctas))
            specs.extend(fm.site_specs(self.rows_max, f"frI{l}", f"frT{l}"))
        if trap is None:
            trap = os.environ.get(ENV_TRAP, "1").strip() != "0"
        self.arena = HopArena(
            group, self.device, self.rank, self.world, specs, trap=trap
        )
        dev = self.device
        rm = self.rows_max
        self.out_x = torch.empty(2, rm, H, dtype=torch.bfloat16, device=dev)
        self.fr_hs = torch.empty(2, rm, TOK_FP4, dtype=torch.uint8, device=dev)
        self.fr_sf = torch.empty(2, rm, TOK_SF, dtype=torch.uint8, device=dev)
        self.fr_pts = torch.empty(2, rm, dtype=torch.float32, device=dev)
        self.fr_pk = torch.empty(2, rm, TOPK, dtype=torch.int32, device=dev)
        # the experts' forward reads only the row count and the dtype of its hidden-state argument when the input is
        # prepared: a [rows_max, 1] bf16 placeholder sliced to M rows
        self.rows_bf16 = torch.zeros(rm, 1, dtype=torch.bfloat16, device=dev)
        self.land_tap = (
            c3.AgxLanding(
                group,
                dev,
                self.rank,
                self.world,
                rm,
                H,
                self.ag_ctas,
                copies=len(self.taps),
            )
            if self.taps
            else None
        )
        self.land_res = c3.AgxLanding(
            group, dev, self.rank, self.world, rm, 2 * H, self.ag_ctas, copies=1
        )
        self.trap = trap
        self.scrub_seq = None

    def preload(self) -> None:
        """Build every JIT module the span launches before any capture."""
        from .ops import lb_hop
        from .ops.c3_hops import _ag_module, _module

        _module(self.world, True)
        _ag_module(self.world, True)
        lb_hop._module("seqrel", self.world, False)  # the forward head's build
        self.fm.module_c3(self.world)

    def error_word(self) -> int:
        return self.arena.error_word()


# ------------------------------------------------------------------------------------------------ owned-row ops
def _owner_norm(boundary, branch, stream_own, M: int, want_router: bool):
    """The boundary on the owned rows with the launch configuration SGLang picks for the pass's M rows (so each row is
    computed as in the replicated M-row call)."""
    from sglang.kernels.ops.layernorm.muse_spark_norm_residual_jit import (
        fused_norm_residual,
        norm_residual_launch_config,
    )

    cs, thr, pf = norm_residual_launch_config(H, int(M))
    kw = (
        dict(cluster_size=cs)
        if cs != 1
        else dict(launch_config=(thr, pf), cluster_size=1)
    )
    return fused_norm_residual(
        branch,
        stream_own,
        **boundary.jit_kernel_kwargs(torch.bfloat16),
        want_router=want_router,
        pre_norm_bf16=False,
        **kw,
    )


def _zero_sign_flags(st: _State, M: int) -> tuple[int, int]:
    """The sign-of-zero rule of the all-reduce the replicated path would run here (its graph-mode policy inside capture,
    its eager policy otherwise)."""
    c3 = st.c3
    graph = torch.cuda.is_current_stream_capturing()
    return (
        c3.stock_zero_sign_flag(st.comm, int(M) * H * 2, graph=graph),
        c3.stock_zero_sign_flag(st.comm, int(M) * EH * 2, graph=graph),
    )


class _Fwd:
    """Per-forward context of an engaged pass (thread-local while the model's forward runs)."""

    def __init__(self, st: _State, M: int):
        self.st = st
        self.M = int(M)
        self.own0, self.own1 = owned(self.M, st.world, st.rank)
        self.n = self.own1 - self.own0
        self.zsA, self.zsB = _zero_sign_flags(st, self.M)
        self.res_own = None
        self.aside = None  # a reason found after the head (all ranks find it alike): the rest of the pass is served


class LayerView:
    """What the owned-row path needs from one decoder layer (built once per engaged layer from the served module; a
    harness may build one from its own tensors)."""

    __slots__ = (
        "idx",
        "attn_boundary",
        "ffn_boundary",
        "w_pool",
        "w_unpool",
        "router",
        "router_bias",
        "router_mult",
        "route_scale",
        "route_eps",
        "qweight",
        "qeps",
        "side_stream",
    )

    def __init__(self, **kw):
        for k in self.__slots__:
            setattr(self, k, kw[k])


def served_view(layer) -> LayerView:
    v = getattr(layer, "_mid_rs_view", None)
    if v is None:
        mlp = layer.mlp
        v = LayerView(
            idx=layer.layer_idx,
            attn_boundary=layer._fused_attn_boundary,
            ffn_boundary=layer._fused_ffn_boundary,
            w_pool=mlp.pre_expert_proj.weight,
            w_unpool=mlp.post_expert_proj.weight,
            router=mlp.gate,
            router_bias=mlp.gate.e_score_correction_bias,
            router_mult=mlp.router_multiplier,
            route_scale=mlp.route_scale,
            route_eps=mlp.route_eps,
            qweight=mlp.experts._pre_norm_weight(mlp.pre_expert_norm),
            qeps=mlp.pre_expert_norm.eps,
            side_stream=mlp._side_stream(),
        )
        layer._mid_rs_view = v
    return v


def ha_front(st: _State, lv: LayerView, partial, stream_full, f: _Fwd):
    """Reduce-scatter of the attention partials, the attention boundary on the owned rows, pooler + router on the owned
    rows, the front launch and the landing unpack. Returns (stream_mid [own rows, fp32], hs_fp4, hs_sf (uint8), pts,
    packed) with the last four over all M rows."""
    from sglang.srt.layers.quantization.unquant import bf16_gemm_dispatch

    c3, fm = st.c3, st.fm
    l, M, n = lv.idx, f.M, f.n
    dev = partial.device
    red = torch.empty(n, H, dtype=torch.bfloat16, device=dev)
    first = l == st.first
    flags = c3.RS_FLAT | f.zsA | (c3.STEP_ACQUIRE if first else c3.PUSH_BEFORE_GUARD)
    c3.rs_rows(st.arena, f"rsA{l}", partial, red, step_flags=flags)
    s_own = stream_full[f.own0 : f.own1] if first else f.res_own
    x, stream_mid, router_in = _owner_norm(
        lv.attn_boundary, red, s_own, M, want_router=True
    )
    main = torch.cuda.current_stream()
    side = lv.side_stream
    side.wait_stream(main)
    xp = bf16_gemm_dispatch(x, lv.w_pool, None)
    with torch.cuda.stream(side):
        logits = lv.router(router_in)
    main.wait_stream(side)
    if not torch.cuda.is_current_stream_capturing():
        logits.record_stream(main)
    tw = torch.empty(max(n, 1), TOPK, dtype=torch.float32, device=dev)[:n]
    ti = torch.empty(max(n, 1), TOPK, dtype=torch.int32, device=dev)[:n]
    pk = torch.empty(max(n, 1), TOPK, dtype=torch.int32, device=dev)[:n]
    fm.run_c3(
        st.arena,
        f"frI{l}",
        f"frT{l}",
        M,
        logits,
        lv.router_bias,
        xp,
        tw,
        ti,
        pk,
        lv.qweight,
        acquire=False,
        mult=lv.router_mult,
        route_scale=lv.route_scale,
        div_eps=lv.route_eps,
        qeps=lv.qeps,
        gsi=NVFP4_PER_TOKEN_SCALE_INV,
    )
    copy = l % 2
    hs, sf, pts, pk_all = (
        st.fr_hs[copy][:M],
        st.fr_sf[copy][:M],
        st.fr_pts[copy][:M],
        st.fr_pk[copy][:M],
    )
    c3.fr_unpack(st.arena, f"frI{l}", f"frT{l}", M, hs, sf, pts, pk_all)
    return stream_mid, hs, sf, pts, pk_all


def rs_b(st: _State, lv: LayerView, part, f: _Fwd):
    """Reduce-scatter of the expert partials [M, 4096] to the owners."""
    c3 = st.c3
    moe_own = torch.empty(f.n, EH, dtype=torch.bfloat16, device=part.device)
    c3.rs_rows(
        st.arena,
        f"rsB{lv.idx}",
        part,
        moe_own,
        step_flags=c3.RS_FLAT | f.zsB | c3.PUSH_BEFORE_GUARD,
    )
    return moe_own


def tail(st: _State, lv: LayerView, moe_own, stream_mid, f: _Fwd):
    """Unpooler + FFN boundary on the owned rows, all-gather of the next attention input; full-row gathers at capture
    layers and after the span. Returns (stream, next attention input) as the decoder layer does."""
    from sglang.srt.layers.quantization.unquant import bf16_gemm_dispatch

    c3 = st.c3
    l, M = lv.idx, f.M
    un = bf16_gemm_dispatch(moe_own, lv.w_unpool, None)
    next_own, stream_new, _ = _owner_norm(
        lv.ffn_boundary, un, stream_mid, M, want_router=False
    )
    copy = l % 2
    c3.ag_rows_tagged(
        st.arena,
        f"agX{l}",
        next_own,
        st.out_x[copy],
        M,
        step_flags=c3.PUSH_BEFORE_GUARD,
    )
    next_attn_input = st.out_x[copy][:M]
    f.res_own = stream_new
    if l == st.last:  # full fp32 rows for the layers after the span
        c3.ag_rows(
            st.land_res,
            st.arena,
            stream_new.contiguous().view(torch.bfloat16),
            M,
            0,
            l,
            st.ag_ctas,
        )
        return st.land_res.view(0)[:M].view(torch.float32), next_attn_input
    if (
        l in st.taps
    ):  # full bf16 rows for the draft model's capture (the model casts the stream to bf16 there)
        t = st.taps.index(l)
        c3.ag_rows(
            st.land_tap, st.arena, stream_new.to(torch.bfloat16), M, t, l, st.ag_ctas
        )
        return st.land_tap.view(t)[:M], next_attn_input
    return stream_new, next_attn_input


def rs_layer(layer, positions, stream, attn_input, forward_batch, f: _Fwd):
    """One engaged layer: returns (stream, next attention input) as the decoder layer does, with ``stream`` holding the
    owned rows only, except at capture layers (full rows, bf16) and the last engaged layer (full rows, fp32)."""
    st = f.st
    lv = served_view(layer)
    # attention as served, the output projection's all-reduce skipped
    attn = layer.self_attn
    qkv, _ = attn.qkv_proj(attn_input)
    partial, _ = attn.o_proj(
        attn.gated_attention(positions, qkv, forward_batch), skip_all_reduce=True
    )
    stream_mid, hs, sf, pts, pk_all = ha_front(st, lv, partial, stream, f)
    # the served expert stage on all M rows (the routed-MoE call takes the replicated path's kernel), finalize without
    # the all-reduce
    mlp = layer.mlp
    experts = mlp.experts
    part = experts(
        st.rows_bf16[: f.M],
        None,
        pk_all,
        experts.post_expert_norm,
        pre_expert_norm=mlp.pre_expert_norm,
        packed_topk_ids=pk_all,
        prepared_input=(hs, sf.view(torch.float8_e4m3fn), pts),
        ar_workspace=None,
    )
    moe_own = rs_b(st, lv, part, f)
    return tail(st, lv, moe_own, stream_mid, f)


# ------------------------------------------------------------------------------------------------ hooks
def _around_layer(original, layer, positions, stream, attn_input, forward_batch):
    f = getattr(_tls, "fwd", None)
    if f is None or f.aside is not None or layer.layer_idx not in f.st.span:
        return original(layer, positions, stream, attn_input, forward_batch)
    from sglang.srt.runtime_context import get_forward

    if getattr(get_forward(), "sp_active", False):
        if layer.layer_idx != f.st.first:
            raise RuntimeError(
                "muse_vr_plugin: mid_rs: a sequence-parallel pass reached an engaged layer after the "
                "span began"
            )
        f.aside = "sequence_parallel"
        _aside(f.aside)
        return original(layer, positions, stream, attn_input, forward_batch)
    if layer.layer_idx == f.st.first:
        bed.count(NAME, f"fused:M={f.M}:tp={f.st.world}")
    return rs_layer(layer, positions, stream, attn_input, forward_batch, f)


def _around_model(
    original, model, input_ids, positions, forward_batch, input_embeds=None
):
    st = _state["st"]
    if st is None or model is not st.lm:
        return original(model, input_ids, positions, forward_batch, input_embeds)
    M = int(input_ids.shape[0]) if input_ids is not None else int(input_embeds.shape[0])
    mode = forward_batch.forward_mode
    verify = mode.is_target_verify() or (_state["dev_decode"] and mode.is_decode())
    reason = decide(verify, False, M, st.world, st.rows_max, True)
    if reason is None:
        from sglang.srt.layers.moe.utils import should_skip_mlp_all_reduce
        from sglang.srt.state_capturer.routed_experts import get_global_experts_capturer

        if get_global_experts_capturer() is not None:
            reason = "routed_experts_capture_on"  # the served path records every layer's ids in full token order
        elif should_skip_mlp_all_reduce():
            reason = "mlp_all_reduce_override"
    if (
        reason is None
        and tuple(getattr(model, "layers_to_capture", None) or ()) != st.taps_all
    ):
        turn_off("the model's capture layers changed after the hop state was built")
        reason = "off_after_error"
    if reason is not None:
        _aside(reason)
        return original(model, input_ids, positions, forward_batch, input_embeds)
    capturing = torch.cuda.is_current_stream_capturing()
    if not capturing:  # eager forward (capture warm-up or an uncaptured size): count it for the tag-wrap scrub
        _count_forwards(st, 1)
    f = _Fwd(st, M)
    st.arena.forward_head(release=True)
    prev = getattr(_tls, "fwd", None)
    _tls.fwd = f
    try:
        return original(model, input_ids, positions, forward_batch, input_embeds)
    finally:
        _tls.fwd = prev


def _count_forwards(st: _State, n: int) -> None:
    if st.arena.before_forward(n):
        seq = int(st.arena.seq64.item())
        last = st.scrub_seq
        st.scrub_seq = seq
        if last is not None and seq - last >= 65535 - 1:
            raise RuntimeError(
                f"muse_vr_plugin: mid_rs: {seq - last} forward heads between two scrubs (the host count "
                "fell behind the device)"
            )
        bed.count(NAME, "scrub_on_wrap")


def _check(where: str) -> None:
    st = _state["st"]
    if st is None:
        return
    w = st.error_word()
    if w:
        turn_off(f"error word 0x{w:08x} at {where}")
        bed.count(NAME, "ERROR")
        st.arena.check()  # raises with the decoded word
        raise RuntimeError(
            f"muse_vr_plugin: mid_rs: error word 0x{w:08x} at {where}; this step's outputs are invalid"
        )


def _before_runner_forward(runner, *a, **kw):
    st = _state["st"]
    if st is None or runner is not _state["runner"]:
        return None
    _check("runner forward")
    if not torch.cuda.is_current_stream_capturing():
        _count_forwards(st, _state["heads"])
    return None


def _before_result_decode(processor, batch, result, *a, **kw):
    """After the step's results are synchronized and before any of its tokens is processed (B(i) host check)."""
    if _state["st"] is None:
        return None
    done = getattr(result, "copy_done", None)
    if done is not None:
        done.synchronize()
    else:
        torch.cuda.synchronize(_state["st"].device)
    _check("decode result")
    return None


def _language_model(runner):
    m = getattr(runner, "model", None)
    for path in (("model", "language_model"), ("language_model",), ("model",)):
        x = m
        for p in path:
            x = getattr(x, p, None)
        if x is not None and type(x).__name__ == "MuseSpark1xModel":
            return x
    return None


def _engaged_span(lm) -> tuple[list, str | None]:
    """The contiguous NVFP4 MoE layers whose experts moe_ld prepared (W placement + call site)."""
    from . import moe_ld

    span = []
    for layer in lm.layers:
        mlp = getattr(layer, "mlp", None)
        ex = getattr(mlp, "experts", None)
        ok = (
            getattr(layer, "is_moe_layer", False)
            and type(ex).__name__ == "MuseSpark1xFp4Experts"
            and getattr(mlp, "pre_expert_proj", None) is not None
            and getattr(mlp, "post_expert_proj", None) is not None
            and getattr(layer, "_fused_ffn_boundary", None) is not None
            and moe_ld._state["sites"].get(ex.gate_up_proj.data_ptr()) is not None
        )
        if ok:
            span.append(layer.layer_idx)
    if span and span[-1] == len(lm.layers) - 1:
        span = span[
            :-1
        ]  # the model's last layer stays on its served path (the final norm reads its full rows)
    if not span:
        return [], "no prepared NVFP4 MoE layer below the last layer"
    if span != list(range(span[0], span[-1] + 1)):
        return (
            [],
            f"prepared layers not contiguous ({span[0]}..{span[-1]}, {len(span)} layers)",
        )
    if span[-1] >= 64:
        return [], "layer index >= 64 (the gather stamp space)"
    return span, None


def _before_graphs(runner, *a, **kw):
    """ModelRunner.init_cuda_graphs (BEFORE): build the hop state collectively, after the KV pool is sized."""
    if _state["st"] is not None or _state["reason"] is not None:
        return None
    lm = _language_model(runner)
    if lm is None:
        return None  # another model's runner (the draft model)
    try:
        from sglang.srt.distributed import get_tp_group
        from sglang.srt.distributed.device_communicators.custom_all_reduce_v2 import (
            CustomAllReduceV2,
        )
        from sglang.srt.layers.dp_attention import is_dp_attention_enabled

        from . import moe_ld
        from .ops import wfc_c4_fr as fm

        tp = get_tp_group()
        comm = tp.ca_comm
        sa = runner.server_args
        nd = int(getattr(sa, "speculative_num_draft_tokens", 0) or 0)
        if _state["dev_decode"] and getattr(sa, "speculative_algorithm", None) is None:
            nd = 8  # development servers without a draft: decode passes of up to 8 x max running requests rows
        rows_max = int(sa.max_running_requests or 0) * nd
        span, why = _engaged_span(lm)
        taps = tuple(int(t) for t in getattr(lm, "layers_to_capture", []) or [])
        if not moe_ld._state.get("engaged") or not moe_ld._state.get("ldw"):
            why = (
                "needs MUSE_VR_MOE_LD=1 and MUSE_VR_MOE_LDW=1 (the served expert stage)"
            )
        elif tp.world_size not in _state["worlds"]:
            why = f"tensor-parallel size {tp.world_size}"
        elif not isinstance(comm, CustomAllReduceV2) or comm.disabled:
            why = "custom all-reduce v2 unavailable"
        elif is_dp_attention_enabled():
            why = "DP attention"
        elif nd != 8 or rows_max <= 0:
            why = f"speculative draft tokens {nd} / max running requests {sa.max_running_requests}"
        elif not fm.hop_available():
            why = "no front-launch hop build in this tree"
        elif taps and max(taps) >= len(lm.layers) - 1:
            why = "a capture layer is the model's last layer"
        if why is None:
            for l in span:
                layer = lm.layers[l]
                r = _layer_shape_reason(layer)
                if r is None and not (
                    layer._fused_attn_boundary.jit_selected()
                    and layer._fused_ffn_boundary.jit_selected()
                ):
                    r = "a boundary is not on its JIT kernel"
                if r is not None:
                    why = f"layer {l}: {r}"
                    break
        need = 0
        if why is None:
            need = state_bytes(
                tp.world_size, rows_max, len(span), len([t for t in taps if t in span])
            )
            free = torch.cuda.mem_get_info(comm.device)[0]
            reserve = bed.env_int(ENV_MEM_RESERVE_GB, 8) << 30
            if need + reserve > free:
                why = f"hop state needs {need / 2**30:.2f} GiB, free {free / 2**30:.2f} GiB, reserve {reserve >> 30} GiB"
        if not _agree(comm.group, why is None):  # every rank builds, or none does
            why = why or "a peer rank refused the build"
        if why is not None:
            _state["reason"] = why
            bed.report(NAME, False, reason=why)
            return None
        built = None
        try:
            built = _State(
                comm.group,
                comm.device,
                comm.rank,
                comm.world_size,
                rows_max,
                span,
                taps,
                comm,
                lm=lm,
            )
            built.preload()
        except Exception as e:  # noqa: BLE001 - decided together below
            why = f"state: {type(e).__name__}: {e}"
        if not _agree(comm.group, why is None):
            _state["reason"] = why or "a peer rank failed the build"
            bed.report(NAME, False, reason=_state["reason"])
            return None
        st = built
        _state["st"], _state["runner"] = st, runner
        bed.report(
            NAME,
            True,
            tp=st.world,
            rows_max=st.rows_max,
            span=f"{st.first}..{st.last}",
            taps=list(st.taps),
            min_m=_state["min_m"],
            trap=st.trap,
            ctas=st.ctas,
            rs_ctas=st.rs_ctas,
            ag_ctas=st.ag_ctas,
            arena_bytes=int(st.arena.total_bytes),
            state_bytes=need,
        )
    except Exception as e:  # noqa: BLE001 - the served path stays
        _state["reason"] = f"state: {type(e).__name__}: {e}"
        bed.report(NAME, False, reason=_state["reason"])
    return None


def _after_graphs(result, runner, *a, **kw):
    """ModelRunner.init_cuda_graphs (AFTER): one collective scrub, so the first replay starts from clean sites."""
    st = _state["st"]
    if st is not None and runner is _state["runner"]:
        st.arena.scrub_all()
        st.arena.forwards = 0
        st.scrub_seq = int(st.arena.seq64.item())
        bed.count(NAME, "scrub_after_capture")
    return None


def pins_match() -> tuple[bool, str]:
    from . import sgl_compat

    if sgl_compat.layout() != "main":
        return False, "SGLang layout is not main"
    for rel, want in PINS.items():
        if not bed.pinned(rel, want):
            return False, f"{rel} differs from the pinned revision"
    return True, "ok"


def register() -> bool:
    if not enabled():
        return False
    ok, why = pins_match()
    if not ok:
        bed.report(NAME, False, reason=why)
        return False
    _state["min_m"] = max(17, bed.env_int(ENV_MIN_M, 17))
    extra = tuple(
        int(x) for x in os.environ.get(ENV_WORLDS, "").split(",") if x.strip()
    )
    _state["worlds"] = SERVED_WORLDS + extra
    _state["heads"] = max(1, bed.env_int(ENV_HEADS, 2))
    if bed.env_int(ENV_DEV_DECODE, 0) == 1:
        logger.warning(
            "muse_vr_plugin: mid_rs %s=1: plain decode passes engage too (development only; never in a "
            "measured configuration)",
            ENV_DEV_DECODE,
        )
        _state["dev_decode"] = True
    from sglang.srt.plugins.hook_registry import HookRegistry, HookType

    def layer_hook(original, *args, **kwargs):
        if kwargs or len(args) != 5:
            return original(*args, **kwargs)
        return _around_layer(original, *args)

    def model_hook(original, *args, **kwargs):
        if len(args) < 4 or len(args) > 5:
            return original(*args, **kwargs)
        return _around_model(original, *args, **kwargs)

    HookRegistry.register(T_LAYER, layer_hook, HookType.AROUND)
    HookRegistry.register(T_MODEL, model_hook, HookType.AROUND)
    HookRegistry.register(T_GRAPHS, _before_graphs, HookType.BEFORE)
    HookRegistry.register(T_GRAPHS, _after_graphs, HookType.AFTER)
    HookRegistry.register(T_RUNNER_FWD, _before_runner_forward, HookType.BEFORE)
    HookRegistry.register(T_RESULT, _before_result_decode, HookType.BEFORE)
    logger.warning(
        "muse_vr_plugin: mid_rs hooks installed (state is built before graph capture)"
    )
    return True
