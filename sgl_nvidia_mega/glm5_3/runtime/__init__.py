# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""The GLM-5.3 megakernel runtime: the slots, the checks of the loaded model
(``model_disable_reasons``), ``build``, which SGLang runs through the spec once the
model is prepared and before the KV cache is sized, the executor of a covered verify
step (``executor``) and kernel compilation (``compile``)."""

from __future__ import annotations

import atexit
import logging
import os
import threading
import time
import weakref
from typing import Dict, List, Optional

import torch

logger = logging.getLogger(__name__)

# The kernel slots this package ships; with the megakernel on, all of them run
# (SLOTS_ENV, for tests and bisecting, can list fewer).
SHIPPED_SLOTS = (
    # the verify attention's kernels: the small-M q_a / kv_a projection on the layers
    # with their own indexer, the MLA prologue, the indexer query, the sparse MLA and
    # the value up-projection
    "attention",
    # the NVFP4 routed experts and the shared expert on the small-M engine
    "moe",
    # the all-reduce + residual + RMSNorm hops over symmetric memory (o_proj and the
    # router on MoE layers, the MoE finalize fold)
    "hops",
    # the q_a / kv_a projection of the layers without their own top-k (they reuse the
    # previous layer's) on the small-M kernel, one packed weight copy each
    "qkv_a_small_m_shared",
    # those layers' input RMSNorm deferred into that projection's epilogue, from the
    # MoE fold hop's block sums (it moves the rounding points: spec.NUMERICS' dnorm
    # keys)
    "dnorm",
    # the NextN draft step (built with the draft spec): the fused embedding + eh norms,
    # the sharded eh_proj, the distributed top-1 and the draft layer's MLA prologue
    "draft",
)
# the draft's slots, which build with the draft spec
DRAFT_SLOTS = frozenset({"draft"})
# A test and bisect override, not a setting: the slots to run in place of
# SHIPPED_SLOTS, comma-separated ("" runs every step function's stock code)
SLOTS_ENV = "SGLANG_NVIDIA_MEGAKERNEL_SLOTS"
# slot -> the slots it needs
SLOT_NEEDS = (
    ("qkv_a_small_m_shared", "attention"),
    ("dnorm", "qkv_a_small_m_shared"),
    ("dnorm", "hops"),
)


def _requested_slots() -> frozenset:
    """Every shipped slot, or the slots ``SLOTS_ENV`` lists (comma-separated; for
    tests and bisecting: "" runs every step function's stock code)."""
    raw = os.environ.get(SLOTS_ENV)
    if raw is None:
        return frozenset(SHIPPED_SLOTS)
    return frozenset(name.strip() for name in raw.split(",")) - {""}


SLOTS = _requested_slots()


def slot_reasons() -> List[str]:
    """Why the requested slots cannot be served ([] when they can)."""
    known = SHIPPED_SLOTS
    reasons = [
        f"{SLOTS_ENV}: unknown slot {name!r} (known: {', '.join(known)})"
        for name in sorted(SLOTS - set(known))
    ]
    for slot, needs in SLOT_NEEDS:
        if slot in SLOTS and needs not in SLOTS:
            reasons.append(f"{SLOTS_ENV}: slot {slot!r} needs slot {needs!r}")
    return reasons


# The SGLang names this module reads (test/unit/test_contract.py pins them)
SGLANG_READS = {
    "srt/runtime_context.py": (
        "get_parallel",
        '"tp_group"',
        '"tp_rank"',
        "get_server_args",
    ),
    # the operator's --max-running-requests (the startup warning)
    "srt/arg_groups/fields/*.py": ("max_running_requests",),
    "srt/distributed/parallel_state.py": ("cpu_group", "world_size"),
    "kernels/plugin_kernels/nvidia/spec.py|srt/megakernels/spec.py": (
        "widths",
        "tokens_per_request",
        "deadline_s",
    ),
    "srt/kernel_plugins/nvidia/binding.py|srt/megakernels/runtime.py": (
        "megakernel_agree",
        "megakernel_agree(what)",
    ),
    "srt/layers/flashinfer_comm_fusion.py": ("uses_cutedsl_ar_fusion",),
    "srt/layers/layer_boundary/residual/add_norm.py": ("flashinfer_ar_fusion_applies",),
    "srt/layers/moe/utils.py": (
        "post_experts_output_is_complete",
        "post_experts_output_is_complete(is_tp_path)",
    ),
    # the replaced model's modules, weights and quant methods
    "srt/models/deepseek_v2.py": (
        "DeepseekV2MLP",
        "DeepseekV2MoE",
        "model",
        "embed_tokens",
        "config",
        "capture_aux_hidden_states",
        "start_layer",
        "end_layer",
        "layers",
        "self_attn",
        "mlp",
        "use_deep_gemm_bmm",
        "w_vc",
        "dcp_size",
        "num_fused_shared_experts",
        "shared_experts",
        "experts",
        "attn_boundary",
        "gemm_output_zero_allocator_size",
        "llama_4_scaling_config",
    ),
    "srt/layers/moe/fused_moe_triton/layer.py": ("quant_method",),
    "srt/layers/quantization/modelopt_quant.py": ("ModelOptNvFp4FusedMoEMethod",),
    "srt/layers/layer_boundary/*.py": ("input_on_attn_tp_slices",),
}


# Slot draft: the target's hop kernels for its NextN draft's eh gather, by the embedding
# table the target shares with the draft (EAGLE's set_embed_and_head gives the draft the
# target's weight, not its module): id(weight) -> (weak references to the weight and to
# the hop kernels). The target's build registers them; the draft only reads them; the
# entry goes with the target's hop kernels.
_TARGET_HOPS: dict = {}


def register_target_hops(weight, kernels) -> None:
    key = id(weight)
    _TARGET_HOPS[key] = (weakref.ref(weight), weakref.ref(kernels))
    weakref.finalize(kernels, _TARGET_HOPS.pop, key, None)


def target_hops(weight):
    """The hop kernels the target whose embedding table is ``weight`` registered for
    its draft, or None."""
    entry = _TARGET_HOPS.get(id(weight))
    if entry is None or entry[0]() is not weight:
        return None
    return entry[1]()


def tp_gather(obj) -> list:
    """``obj`` of every tensor-parallel rank (host only, on the CPU group; legal
    inside a graph capture)."""
    import torch.distributed as dist
    from sglang.srt.runtime_context import get_parallel

    group = get_parallel().tp_group
    out = [None] * group.world_size
    dist.all_gather_object(out, obj, group=group.cpu_group)
    return out


def per_rank(gathered) -> str:
    return "; ".join(f"rank {r}: {', '.join(g)}" for r, g in enumerate(gathered) if g)


class DerivedWeights:
    """Owns the packed / re-laid-out copies of stock weights (the checkpoint is
    never modified): the kernels keep only their device addresses (e.g.
    ``attention.fused.LayerArgs``), so these tensors must live as long as the
    built megakernel. Also logged by kind at startup."""

    def __init__(self) -> None:
        self.tensors: Dict[str, List[torch.Tensor]] = {}

    def add(self, kind: str, tensor: torch.Tensor) -> torch.Tensor:
        self.tensors.setdefault(kind, []).append(tensor)
        return tensor

    def mib(self) -> Dict[str, float]:
        return {
            k: sum(t.numel() * t.element_size() for t in ts) / 2**20
            for k, ts in sorted(self.tensors.items())
        }

    def log(self, rank: int) -> None:
        mib = self.mib()
        if not mib:
            logger.info(
                "NVIDIA megakernel: GLM-5.3 / GLM-5.2: no derived weight copies "
                "(rank %d)",
                rank,
            )
            return
        logger.info(
            "NVIDIA megakernel: GLM-5.3 / GLM-5.2: derived weight copies on rank %d: "
            "%s (total %.1f MiB)",
            rank,
            ", ".join(f"{k} {v:.1f} MiB" for k, v in mib.items()),
            sum(mib.values()),
        )


class ErrorReporter:
    """Logs the first error record left by a cross-GPU wait that timed out.

    Each source returns None or a dict describing the wait (kind, layer and hop,
    waiting rank, late peer, rows) read from host memory, so it works after the
    trap has lost the CUDA context. Once a source exists a daemon thread checks
    every ``period_s`` seconds, and once more at exit.
    """

    def __init__(self, sources=(), period_s: float = 0.5) -> None:
        self.sources = list(sources)
        self.record = None
        self._lock = threading.Lock()
        self._period_s = period_s
        self._watching = False
        if self.sources:
            self._start()

    def add(self, source) -> None:
        with self._lock:
            self.sources.append(source)
        self._start()

    def _start(self) -> None:
        if not self._watching:
            self._watching = True
            atexit.register(self.check)
            threading.Thread(
                target=self._watch, name="glm5-3-errors", daemon=True
            ).start()

    def check(self):
        with self._lock:
            if self.record is not None:
                return self.record
            for read in list(self.sources):
                try:
                    record = read()
                except Exception as e:  # noqa: BLE001
                    logger.warning(
                        "NVIDIA megakernel: GLM-5.3 / GLM-5.2: error record "
                        "unreadable: %s",
                        e,
                    )
                    self.sources.remove(read)
                    continue
                if record:
                    self.record = record
                    logger.error(
                        "NVIDIA megakernel: GLM-5.3 / GLM-5.2: a cross-GPU wait in "
                        "the verify step timed out and trapped; this process's CUDA "
                        "context is lost. Error record: %s",
                        record,
                    )
                    break
            return self.record

    def _watch(self) -> None:
        while self.sources and self.check() is None:
            time.sleep(self._period_s)


def model_disable_reasons(model) -> List[str]:
    """Facts of the loaded, processed model the kernels and the stock step copies
    need (every rank's reasons are gathered by SGLang)."""
    from sglang.srt.layers.flashinfer_comm_fusion import uses_cutedsl_ar_fusion
    from sglang.srt.layers.layer_boundary.residual.add_norm import (
        flashinfer_ar_fusion_applies,
    )
    from sglang.srt.layers.moe.utils import post_experts_output_is_complete
    from sglang.srt.models.deepseek_v2 import DeepseekV2MLP, DeepseekV2MoE

    from sgl_nvidia_mega.glm5_3.plan import MAX_ROWS

    reasons: List[str] = []
    if getattr(model, "num_fused_shared_experts", 0) != 0:
        reasons.append(
            "requires shared-experts fusion off (num_fused_shared_experts == 0; "
            "pass --disable-shared-experts-fusion)"
        )
    mm = model.model
    if getattr(model, "capture_aux_hidden_states", False):
        reasons.append(
            "auxiliary hidden-state capture (EAGLE3 / DFLASH) is not supported"
        )
    if mm.start_layer != 0 or mm.end_layer != model.config.num_hidden_layers:
        reasons.append("pipeline-parallel layer ranges are not supported")
    for i in range(mm.start_layer, mm.end_layer):
        attn, mlp = mm.layers[i].self_attn, mm.layers[i].mlp
        if getattr(attn, "use_deep_gemm_bmm", False):
            reasons.append(f"layer {i}: DeepGEMM MLA bmm is not supported")
        w_vc = getattr(attn, "w_vc", None)
        if w_vc is None or w_vc.dtype != torch.bfloat16:
            reasons.append(
                f"layer {i}: W_UV must be bf16 (got {getattr(w_vc, 'dtype', None)})"
            )
        if getattr(attn, "learnable_sink_param", None) is not None:
            reasons.append(f"layer {i}: attention sinks are not supported")
        if getattr(attn, "dcp_size", 1) != 1:
            reasons.append(f"layer {i}: decode context parallelism is not supported")
        if i < model.config.first_k_dense_replace:
            if not isinstance(mlp, DeepseekV2MLP):
                reasons.append(f"layer {i}: expected a dense MLP layer")
            continue
        if not isinstance(mlp, DeepseekV2MoE):
            reasons.append(f"layer {i}: expected a MoE layer")
        if getattr(mlp, "num_fused_shared_experts", 0) != 0:
            reasons.append(
                f"layer {i}: shared experts are fused into the routed experts"
            )
        if getattr(mlp, "shared_experts", None) is None:
            reasons.append(f"layer {i}: no shared-expert module")
        method = type(getattr(mlp.experts, "quant_method", None)).__name__
        if method != "ModelOptNvFp4FusedMoEMethod":
            reasons.append(f"layer {i}: routed experts must be NVFP4 (got {method})")
    if not isinstance(mm.layers[mm.end_layer - 1].mlp, DeepseekV2MoE):
        reasons.append("the last layer must be a MoE layer")
    sliced = [
        i
        for i in range(mm.start_layer, mm.end_layer)
        if getattr(mm.layers[i].attn_boundary, "input_on_attn_tp_slices", False)
    ]
    if sliced:
        reasons.append(
            f"layer {sliced[0]}: attention input on attention-TP slices is not supported"
        )
    if getattr(mm, "gemm_output_zero_allocator_size", 0):
        reasons.append("the GEMM output zero allocator is not supported")
    # The residual steps are the fused all-reduce + residual + RMSNorm the layer
    # boundaries use at verify sizes.
    if uses_cutedsl_ar_fusion():
        reasons.append("the CuTe DSL all-reduce fusion backend is not supported")
    elif not flashinfer_ar_fusion_applies(MAX_ROWS):
        reasons.append(
            "requires the FlashInfer all-reduce + RMSNorm fusion at verify sizes "
            "(--flashinfer-allreduce-fusion-backend not disabled)"
        )
    if getattr(mm, "llama_4_scaling_config", None) is not None:
        reasons.append("llama-4 attention scaling is not supported")
    # The covered steps perform the MLP output's tensor-parallel sum, as the stock
    # residual step does; a MoE combine that already completes it is out.
    if post_experts_output_is_complete(is_tp_path=True):
        reasons.append(
            "the MoE combine already completes the tensor-parallel sum "
            "(DWDP, FP4 all-gather dispatch or an all-to-all backend)"
        )
    return reasons


# An explicit --max-running-requests above this marks a high-concurrency server, for
# which the startup warns that the megakernel helps little
HIGH_CONCURRENCY_REQUESTS = 8


def requested_max_running_requests() -> Optional[int]:
    """The ``--max-running-requests`` the operator passed (SGLang's raw server
    argument), or None when it was not passed (SGLang then resolves it, to 48 with
    EAGLE) or outside a server."""
    from sglang.srt.runtime_context import get_server_args

    try:
        return get_server_args().max_running_requests
    except ValueError:  # no server arguments (tools, unit tests)
        return None


def log_coverage(
    widths, tokens_per_request: int, derived_mib: float, rank: int
) -> None:
    """Rank 0's startup line: the verify steps the megakernel covers and the memory
    of its derived weight copies; and a warning when the operator allowed a
    high-concurrency server, whose larger steps run SGLang's path."""
    if rank != 0:
        return
    requests = sorted({w // tokens_per_request for w in widths})
    lo, hi = requests[0], requests[-1]
    span = f"{lo}-{hi}" if hi > lo else f"{lo}"
    gib = derived_mib / 2**10
    logger.info(
        "NVIDIA megakernel: GLM-5.3 / GLM-5.2: covers verify steps of %s requests "
        "(%s rows); steps with more requests run SGLang's path. The derived weight "
        "copies use about %.1f GiB per GPU.",
        span,
        " / ".join(str(w) for w in sorted(widths)),
        gib,
    )
    passed = requested_max_running_requests()
    if passed is not None and passed > HIGH_CONCURRENCY_REQUESTS:
        logger.warning(
            "NVIDIA megakernel: GLM-5.3 / GLM-5.2: --max-running-requests %d: steps "
            "above %d requests run SGLang's path while the derived weight copies "
            "(about %.1f GiB per GPU) still take KV-cache memory, so the megakernel "
            "helps little on high-concurrency servers.",
            passed,
            hi,
            gib,
        )


def build(model, ctx):
    """Each requested kernel family's build (weights derived, buffers allocated,
    kernels compiled), the step groups per width, and the executor; ``ctx`` is
    SGLang's BuildContext. A requested family that cannot serve, or ranks that
    composed different step groups, stop every rank."""
    from sglang.srt.runtime_context import get_parallel

    try:
        from sglang.srt.kernel_plugins.nvidia.binding import megakernel_agree
    except ModuleNotFoundError:  # SGLang before sgl-project/sglang#45389
        from sglang.srt.megakernels.runtime import megakernel_agree

    from sgl_nvidia_mega.glm5_3 import plan
    from sgl_nvidia_mega.glm5_3.runtime.compile import compile_stats
    from sgl_nvidia_mega.glm5_3.runtime.executor import GlmMegakernelVerify

    tokens = ctx.tokens_per_request
    if tokens not in plan.SERVED_TOKENS_PER_REQUEST:
        raise RuntimeError(
            f"NVIDIA megakernel: GLM-5.3 / GLM-5.2: {plan.draft_length_reason(tokens)}"
        )
    supported = plan.supported_rows(tokens)
    widths = tuple(w for w in ctx.widths if w in supported)
    if not widths:
        raise RuntimeError(
            f"NVIDIA megakernel: GLM-5.3 / GLM-5.2: widths {list(ctx.widths)} "
            f"hold none of {list(supported)} ({tokens} rows a request)"
        )
    derived = DerivedWeights()
    providers, families, declined = plan.build_providers(model, widths, derived, SLOTS)
    rank = get_parallel().tp_rank
    # A family declined on any rank: every rank stops, naming each rank's reason.
    gathered = tp_gather(sorted(f"{g}: {why}" for g, why in declined.items()))
    if any(gathered):
        raise ValueError(
            "NVIDIA megakernel: GLM-5.3 / GLM-5.2: requested slots cannot be built: "
            + per_rank(gathered)
        )
    # The cross-GPU steps pair up across ranks: every rank serves the same groups.
    composition = {m: p.describe() for m, p in providers.items()}
    megakernel_agree(composition, what="GLM-5.3 / GLM-5.2 step groups per width")
    built = built_slots(providers, families)
    megakernel_agree(
        sorted(s for s, on in built.items() if on), what="GLM-5.3 / GLM-5.2 slots built"
    )
    missing = sorted(s for s in SLOTS & built.keys() if not built[s])
    if missing:
        hops = families.get("hops")
        raise RuntimeError(
            f"NVIDIA megakernel: GLM-5.3 / GLM-5.2: requested slots {missing} built "
            f"at no width (hops: {getattr(hops, 'reason', '') or 'built'})"
        )
    derived.log(rank)
    logger.info(
        "NVIDIA megakernel: GLM-5.3 / GLM-5.2: kernels compiled %d, loaded from "
        "cache %d (rank %d), not stored in the cache %d",
        compile_stats["compiled"],
        compile_stats["cached"],
        rank,
        compile_stats["not_stored"],
    )
    if rank == 0:
        for rows, p in providers.items():
            fused = {
                k: v
                for k, v in p.describe().items()
                if not v.startswith(("AttentionStep", "HopSteps", "MlpSteps"))
            }
            logger.info(
                "NVIDIA megakernel: GLM-5.3 / GLM-5.2: %d verify rows: %s",
                rows,
                fused or "all steps stock",
            )
    log_coverage(widths, tokens, sum(derived.mib().values()), rank)
    errors = ErrorReporter()
    for family in families.values():
        if callable(getattr(family, "error_report", None)):
            errors.add(family.error_report)
    hops = families.get("hops")
    if "draft" in SLOTS and getattr(hops, "kernels", None) is not None:
        _hops_for_draft(model, hops)
    summary = (
        f"slots {sorted(s for s, on in built.items() if on) or 'none (stock)'}; "
        f"derived weight copies {sum(derived.mib().values()):.1f} MiB; kernels "
        f"compiled {compile_stats['compiled']}, from cache {compile_stats['cached']}"
    )
    return GlmMegakernelVerify(
        model,
        providers,
        families,
        widths=widths,
        derived=derived,
        errors=errors,
        summary=summary,
        binding=getattr(ctx, "binding", None),
    )


def _hops_for_draft(model, family) -> None:
    """Slot draft: the eh gather's unit (the hops' plain all-reduce) compiled into the
    target's hop kernels, the draft's error-record sites named, and the kernels
    registered for the draft under the shared embedding table."""
    import cutlass.cute as cute

    from sgl_nvidia_mega.glm5_3.draft_step import SITE_NAMES
    from sgl_nvidia_mega.glm5_3.runtime.compile import compile_cached
    from sgl_nvidia_mega.glm5_3.sm103.hop import api as hop_api

    unit = {s.name: s for s in hop_api.unit_specs(plain_allreduce=True)}["allreduce"]
    family.kernels.compiled.setdefault(
        "allreduce",
        compile_cached(
            unit.obj, unit.constexprs, lambda: cute.compile(unit.obj, *unit.args)
        ),
    )
    family.register_sites(SITE_NAMES)
    register_target_hops(model.model.embed_tokens.weight, family.kernels)


def built_slots(providers: dict, families: dict) -> Dict[str, bool]:
    """Each slot: whether it serves at least one width."""
    stock = ("AttentionStep", "HopSteps", "MlpSteps")

    def served(group: str) -> bool:
        return any(
            not cls.startswith(stock)
            for p in providers.values()
            for step, cls in p.describe().items()
            if step.startswith(group + ".")
        )

    return {
        "attention": served("attention"),
        "moe": served("mlp"),
        "hops": served("hops"),
        "qkv_a_small_m_shared": any(
            getattr(p.attention, "qa_s", None) is not None for p in providers.values()
        ),
        "dnorm": any(getattr(p.hops, "dnorm_layers", ()) for p in providers.values()),
    }
