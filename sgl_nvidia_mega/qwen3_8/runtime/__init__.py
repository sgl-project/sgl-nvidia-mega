# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""The Qwen3.8 forward executor: it owns the target's and the MTP draft's forward
(``executor``), with per-step state (``state``) and the per-op engagement record
(``engagement``); the test-only parity check is ``tools/parity.py``, loaded only when
its switch is set. ``build`` runs through the spec once the model is prepared, before
the KV cache is sized."""

from __future__ import annotations

import logging
import os

logger = logging.getLogger(__name__)

# The SGLang names this package's build and checks read (test/unit/test_contract.py)
SGLANG_READS = {
    "srt/runtime_context.py": ("get_exec", "get_parallel"),
    "srt/kernel_plugins/nvidia/binding.py": ("megakernel_agree",),
    "srt/models/qwen4_exp.py": (
        "Qwen4ExpAttentionDecoderLayer",
        "Qwen4ExpLinearDecoderLayer",
        "is_qsa",
        "indexer",
        "attn_hyper_connection",
        "mlp_hyper_connection",
        "hyper_connection_mixer",
    ),
    "srt/models/qwen3_vl.py": ("lm_head",),
    "srt/models/qwen2_moe.py": (
        "Qwen2MoeSparseMoeBlock",
        "gate",
        "shared_expert",
        "gate_up_proj",
        "down_proj",
    ),
    "srt/layers/hyperconnection.py": ("input_mix_weight_down", "input_mix_weight_up"),
    "srt/models/qwen3_5.py": (
        "linear_attn",
        "_fused_in_proj_weight",
        "out_proj",
        "qkv_proj",
        "o_proj",
    ),
    "srt/layers/attention/qsa/qsa_indexer.py": ("index_qk_proj",),
    "kernels/plugin_kernels/nvidia/spec.py": ("MegakernelSpec",),
    "srt/environ.py": (
        "envs",
        "SGLANG_ENABLE_GDN_PREFILL_GRAPH_EXTEND",
        "SGLANG_OPT_GDN_PREFILL_GRAPH_FLASHINFER",
    ),
}

# Reads and mirrors that need an SGLang change newer than the unit job's pin: the
# contract test skips each, naming the change, only where the installed SGLang lacks it.
SGLANG_REQUIRES = {
    "srt/environ.py:SGLANG_OPT_GDN_PREFILL_GRAPH_FLASHINFER": "sglang#50157",
}

# Comma-separated kernel slots; unset or "" runs every op on its stock step
# function. No slot ships on by default yet.
SLOTS_ENV = "SGLANG_NVIDIA_MEGAKERNEL_SLOTS"
SHIPPED_SLOTS: tuple[str, ...] = ()
# hc: the HC write-back kernel for up to 32 rows (runtime/hc.py); needs the S1 layers
# hc_wide: HC reads / write-backs of more than 32 rows (runtime/hc_wide.py)
# gdn_native_pdl: the GDN target verify's ReplaySSM core as the plugin's PDL launch
# gdn_verify_glue: a GDN target verify of 2-8 requests as one launch per layer between
# the input and the output projection (runtime/gdn.py)
# gdn_front: a TP1 GDN target verify of one request (4 rows) as the input projection
# with the causal conv in its epilogue, then the ReplaySSM core with the gated RMSNorm
# (runtime/gdn.py)
# qsa_indexer: the QSA indexer's fold, scorer and select (runtime/qsa_indexer.py)
# qsa_pool: the TP1 QSA verify's sparse attention as split-K pool kernels (<= 64 rows)
# on a BF16 KV pool; with SGLANG_NVIDIA_MEGAKERNEL_QWEN3_8_ATTN_CORE_V1=1 (the source
# branch's tcgen05-core switch), a one-request verify on its tcgen05 core
# qsa_pool_fp8: the TP1 QSA verify's sparse attention on an FP8 E4M3 KV pool as the
# union kernels, warp-specialized (56-576 rows; runtime/attention.py), and the MTP
# draft extend's (four rows a request) the same way
# moe_finalize: the TP1 MoE's finalize + gated shared expert as one ordered kernel; from
# 1024 rows the source's extend-size MoE, and on the target a FlashInfer tuning pass of
# its routed experts at prefill sizes (runtime/moe.py, runtime/moe_prefill_autotune.py)
# moe_routing: in moe_finalize's step, the routed experts' routing, NVFP4 quant and
# trtllm-gen metadata as one kernel, else the quant beside the router GEMV
# (runtime/moe_routing.py)
# gdn_prefill: the GDN step of a covered prefill / extend on the plugin's prefill
# kernels (runtime/gdn_prefill.py)
# qsa_prefill: the QSA attention of a covered prefill / extend (runtime/qsa_prefill.py)
# ple: the PLE batch layout, table gather, gate and short conv (runtime/ple.py); its
# embedding add needs the S1 layers
# smallm_gemm: SM107 small-M BF16 GEMMs (runtime/smallm_gemm.py): out_proj / o_proj
# split-K at 4-16 and 20-32 rows, the shared expert's fused act and TGV down_proj
# The tuning-pin tables are on when their files are set, as in the source, not by slot:
# SGLANG_FLASHINFER_TACTIC_PIN_FILE (runtime/flashinfer_tactic_pins.py) and
# SGLANG_BF16_GEMM_PIN_FILE (runtime/bf16_gemm_pins.py); likewise the SM107 small-M BF16
# routes with SGLANG_OPT_SM107_BF16_SMALLM_ROUTES=1 (runtime/sm107_bf16_smallm.py).
# qsa_draft: the MTP draft's QSA glue (runtime/qsa_draft.py): a draft decode step's
# lookup of the draft extend's selection inside the K/V pack, and the draft extend's
# seed capture as one launch
# draft_fold: an MTP draft step's top-1 finalize with the M-RoPE + 1 and the next
# step's input embedding folded in, on the LM head's BF16 logits (runtime/draft_fold.py,
# through SGLang's draft_step_fold entry)
KNOWN_SLOTS: tuple[str, ...] = (
    "hc",
    "hc_wide",
    "gdn_native_pdl",
    "gdn_verify_glue",
    "gdn_front",
    "qsa_indexer",
    "qsa_pool",
    "qsa_pool_fp8",
    "moe_finalize",
    "moe_routing",
    "gdn_prefill",
    "qsa_prefill",
    "ple",
    "smallm_gemm",
    "qsa_draft",
    "draft_fold",
)
# A node-local directory for the JIT sources some kernels write before compiling.
SOURCE_DIR_ENV = "SGLANG_NVIDIA_MEGAKERNEL_QWEN3_8_SOURCE_DIR"
# The widest covered step: 4 rows a request x 256 requests.
MAX_ROWS = 1024
# Test only: compare every covered step with SGLang's forward (tools/parity.py)
PARITY_ENV = "SGLANG_NVIDIA_MEGAKERNEL_PARITY"
# S1 (test switch until it is the default): run each layer's HC read / mixer /
# HC write / MoE from the executor instead of through the stage boundaries.
DECOMPOSE_ENV = "SGLANG_NVIDIA_MEGAKERNEL_QWEN3_8_DECOMPOSE"
# The GDN u-cache target verify: the target executor allocates the rings in prepare,
# resets them and commits each verify (runtime/gdn_ucache_state.py), with request
# rings and an fp16 key ring. Needs --mamba-ssm-dtype float16.
UCACHE_ENV = "SGLANG_NVIDIA_MEGAKERNEL_QWEN3_8_UCACHE"


def ucache_requested() -> bool:
    return os.environ.get(UCACHE_ENV) == "1"


def decompose_supported(parallel) -> bool:
    """The stage boundaries do no collective and keep no rows elsewhere."""
    return (
        parallel.tp_size == 1
        and getattr(parallel, "attn_dp_size", 1) == 1
        and getattr(parallel, "attn_cp_size", 1) == 1
        and getattr(parallel, "moe_ep_size", 1) == 1
    )


def requested_slots() -> frozenset:
    raw = os.environ.get(SLOTS_ENV)
    if raw is None:
        return frozenset(SHIPPED_SLOTS)
    return frozenset(name.strip() for name in raw.split(",")) - {""}


def prefill_graph_supported() -> bool:
    """Whether SGLang hands a spec's build the prefill body breakable prefill graphs
    capture (``MegakernelSpec.prefill_graph``)."""
    from sglang.kernels.plugin_kernels.nvidia.spec import MegakernelSpec

    return "prefill_graph" in getattr(MegakernelSpec, "__struct_fields__", ())


def _prefill_backend(graph_config) -> str:
    """The prefill CUDA graph backend of SGLang's cuda_graph_config ('disabled' when
    none is configured)."""
    backend = getattr(getattr(graph_config, "prefill", None), "backend", None)
    return "disabled" if backend is None else str(backend).split(".")[-1].lower()


def gdn_graph_tables() -> bool:
    """Whether breakable prefill graphs hand an external prefill body FlashInfer's GDN
    tables; SGLang then captures no GDN extend of its own, slot gdn_prefill must."""
    from sglang.srt.environ import envs

    return bool(
        envs.SGLANG_ENABLE_GDN_PREFILL_GRAPH_EXTEND.get()
        and envs.SGLANG_OPT_GDN_PREFILL_GRAPH_FLASHINFER.get()
    )


def config_reasons(cfg, hf_config) -> list[str]:
    """Settings the executor cannot serve, checked before the weights load."""
    reasons = [
        f"{SLOTS_ENV}: unknown slot {name!r}"
        for name in sorted(requested_slots() - set(KNOWN_SLOTS))
    ]
    if ucache_requested():
        from sgl_nvidia_mega.qwen3_8.runtime.gdn_ucache_state import (
            UcacheState,
            max_draft_tokens,
        )

        reasons += [f"{UCACHE_ENV}=1: {r}" for r in UcacheState.config_reasons(cfg)]
        # SGLang has no u-cache verify of its own: the executor must cover every
        # target verify, so its widest one must fit the built widths.
        tokens = int(max_draft_tokens(cfg) or 0)
        running = int(getattr(cfg, "max_running_requests", 0) or 0)
        if tokens and running and tokens * running > MAX_ROWS:
            reasons.append(
                f"the GDN u-cache: a verify of {running} requests x "
                f"{tokens} rows exceeds the executor's {MAX_ROWS} rows"
            )
    prefill = _prefill_backend(getattr(cfg, "cuda_graph_config", None))
    if prefill == "breakable" and prefill_graph_supported():
        # the executor's prefill body is captured (Qwen3_8Executor.plan_body)
        if gdn_graph_tables() and "gdn_prefill" not in requested_slots():
            reasons.append(
                "breakable prefill graphs with FlashInfer's GDN tables "
                "(SGLANG_OPT_GDN_PREFILL_GRAPH_FLASHINFER) need slot gdn_prefill, "
                "which runs the GDN core in the graph"
            )
    elif prefill != "disabled":
        # A covered prefill step runs eagerly; only SGLang's prefill body routing
        # (MegakernelSpec.prefill_graph) lets a breakable prefill graph capture it.
        reasons.append(
            f"prefill CUDA graphs {prefill!r} are not supported by the Qwen3.8 "
            "executor; use --cuda-graph-backend-prefill breakable (with an SGLang "
            "whose megakernel spec has prefill_graph) or disabled"
        )
    return reasons


def prefill_graphs_on() -> bool:
    """Whether this server captures the prefill body of the target and of the MTP
    draft in breakable prefill graphs (config_reasons refused every other prefill
    graph)."""
    from sglang.srt.runtime_context import get_exec

    return (
        prefill_graph_supported()
        and _prefill_backend(get_exec().graph.cuda_graph_config) == "breakable"
    )


def model_reasons(model, hf_config) -> list[str]:
    """This rank's reasons the loaded model cannot run the executor."""
    reasons = []
    if getattr(model, "capture_aux_hidden_states", False):
        reasons.append("auxiliary hidden-state capture (EAGLE3) is not supported")
    pp = getattr(model, "pp_group", None)
    if pp is not None and not (pp.is_first_rank and pp.is_last_rank):
        reasons.append("pipeline parallelism is not supported")
    draft = type(model).__name__.endswith("MTP")
    if not draft and gdn_graph_tables() and prefill_graphs_on():
        from sglang.srt.runtime_context import get_parallel

        # plan_body declines CP-sharded rows, which no captured chain then serves
        if getattr(get_parallel(), "attn_cp_size", 1) > 1:
            reasons.append(
                "breakable prefill graphs with FlashInfer's GDN tables under "
                "attention CP are not supported"
            )
    return reasons


def build(model, hf_config, ctx):
    """The executor of this model (target or MTP draft); ``ctx`` is SGLang's
    BuildContext."""
    from sgl_nvidia_mega.qwen3_8.runtime.executor import (
        Qwen3_8Executor,
        Qwen3_8UcacheExecutor,
    )

    draft = type(model).__name__.endswith("MTP")
    slots = requested_slots()
    if draft:
        # A draft decode step has one row a request; its draft-extend rows a
        # request are the target's verify width.
        widths = tuple(range(1, MAX_ROWS + 1))
    else:
        widths = tuple(ctx.widths)
    from sglang.srt.runtime_context import get_parallel

    checker = None
    if os.environ.get(PARITY_ENV, "0") != "0":
        from sgl_nvidia_mega.qwen3_8.tools import parity

        checker = parity.Parity(model, draft=draft) if parity.ENABLED else None
    decompose = os.environ.get(DECOMPOSE_ENV) == "1" and decompose_supported(
        get_parallel()
    )
    own_ucache = not draft and ucache_requested()
    executor = (Qwen3_8UcacheExecutor if own_ucache else Qwen3_8Executor)(
        model,
        widths=widths,
        draft=draft,
        slots=slots,
        binding=ctx.binding,
        parity=checker,
        decompose=decompose,
    )
    if not draft:
        from sglang.srt.runtime_context import get_exec

        mamba = getattr(get_exec(), "mamba", None)
        executor.verify_required = own_ucache
    executor.prefill_graph = prefill_graphs_on()
    if own_ucache:
        from sgl_nvidia_mega.qwen3_8.runtime.gdn_ucache_state import UcacheState

        executor.ucache_state = UcacheState(
            model,
            tokens_per_request=ctx.tokens_per_request,
            track_snapshots=bool(getattr(mamba, "enable_mamba_extra_buffer", False)),
        )
    if "hc" in slots:
        executor.hc_ready = _build_hc(model, decompose)
    if "hc_wide" in slots:
        executor.hc_wide_ready = _build_hc_wide(model, decompose, executor.hc_limits)
    if "gdn_native_pdl" in slots and not draft:
        executor.gdn_pdl_ready = _build_gdn_pdl(model)
    if "gdn_verify_glue" in slots and not draft:
        executor.gdn_glue_ready = _build_gdn_glue(model)
    if "gdn_front" in slots and not draft:
        executor.gdn_front_ready = _build_gdn_front(model)
    if "qsa_indexer" in slots:
        executor.qsa_indexer = _build_qsa_indexer(model)
    if "qsa_pool" in slots and not draft:
        executor.qsa_pool_ready = _build_qsa_pool(model)
        if executor.qsa_pool_ready:
            from sgl_nvidia_mega.qwen3_8.runtime.attention import PoolKernels

            executor.qsa_pool = PoolKernels()
            executor.qsa_core = _build_qsa_core()
    if "qsa_pool_fp8" in slots:
        executor.qsa_union = _build_qsa_union(model)
    if "moe_finalize" in slots:
        executor.moe_finalize_ready = _build_moe_finalize(model, decompose)
    if "moe_routing" in slots:
        from sgl_nvidia_mega.qwen3_8.runtime import moe_routing

        reason = "" if executor.moe_finalize_ready else "needs slot moe_finalize"
        executor.moe_routing = _built_options(
            "moe_routing", reason or moe_routing.build(model), moe_routing.DEFAULT
        )
    if "gdn_prefill" in slots and not draft:
        from sgl_nvidia_mega.qwen3_8.runtime import gdn_prefill

        executor.gdn_prefill = _built_options(
            "gdn_prefill", gdn_prefill.build(model), gdn_prefill.DEFAULT
        )
    if (
        executor.prefill_graph
        and not draft
        and executor.gdn_prefill is None
        and gdn_graph_tables()
    ):
        # SGLang's captured GDN extend refuses FlashInfer's tables (the MTP draft has
        # no GDN layer)
        raise RuntimeError(
            "Qwen3.8 executor: breakable prefill graphs with FlashInfer's GDN tables "
            "need slot gdn_prefill, which did not build (see its log line)"
        )
    if "qsa_prefill" in slots:
        from sgl_nvidia_mega.qwen3_8.runtime import qsa_prefill

        executor.qsa_prefill = _built_options(
            "qsa_prefill", qsa_prefill.build(model), qsa_prefill.DEFAULT
        )
    if "ple" in slots and not draft:
        from sgl_nvidia_mega.qwen3_8.runtime import ple

        executor.ple_options = _built_options("ple", ple.build(model), ple.DEFAULT)
    if "qsa_draft" in slots and draft:
        executor.qsa_draft = _build_qsa_draft(model)
    if "draft_fold" in slots and draft:
        from functools import partial

        from sgl_nvidia_mega.qwen3_8.runtime import draft_fold

        if _built_options("draft_fold", draft_fold.build(), True):
            executor.draft_step_fold = partial(draft_fold.make, executor)
    if "smallm_gemm" in slots:
        from sgl_nvidia_mega.qwen3_8.runtime import smallm_gemm

        options = smallm_gemm.recipe_options()
        executor.smallm_gemm = _built_options(
            "smallm_gemm", smallm_gemm.build(model, options), options
        )
    _build_pins(model, executor)
    if not draft and executor.moe_finalize_ready:
        from sgl_nvidia_mega.qwen3_8.runtime import moe_prefill_autotune

        reason = moe_prefill_autotune.reasons(model)
        if _built_options("moe_prefill_autotune", reason, True):
            moe_prefill_autotune.install(model, executor)
    logger.info(
        "NVIDIA megakernel: Qwen3.8 %s executor: %s%s; slots %s; every other op stock",
        "draft" if draft else ("target (own GDN u-cache)" if own_ucache else "target"),
        "layers decomposed (S1)"
        if decompose
        else "layers through the stage boundaries (S0)",
        "; prefill body in breakable prefill graphs" if executor.prefill_graph else "",
        sorted(slots) or "none",
    )
    return executor


def prepare(built, view) -> None:
    """spec.prepare: once the pools exist, before any capture (repeated on a
    recapture); ``view`` is the binding's preparation view."""
    from sgl_nvidia_mega.qwen3_8.runtime.executor import Qwen3_8UcacheExecutor

    if isinstance(built, Qwen3_8UcacheExecutor):
        built.prepare_ucache(view)


def _build_hc(model, decompose) -> bool:
    """Whether slot hc runs, agreed on every rank."""
    from sglang.srt.kernel_plugins.nvidia.binding import megakernel_agree

    from sgl_nvidia_mega.qwen3_8.runtime import hc

    m = model.model
    reason = "" if decompose else f"needs the S1 layers ({DECOMPOSE_ENV}=1 at TP1)"
    for i in range(m.start_layer, m.end_layer):
        reason = reason or hc.layer_supported(m.layers[i])
    reason = reason or hc.build()
    megakernel_agree(reason, what="Qwen3.8 slot hc")
    if reason:
        unavailable("hc", reason)
    return not reason


def _build_hc_wide(model, decompose, limits) -> bool:
    from sglang.srt.kernel_plugins.nvidia.binding import megakernel_agree

    from sgl_nvidia_mega.qwen3_8.runtime import hc, hc_wide

    m = model.model
    reason = "" if decompose else f"needs the S1 layers ({DECOMPOSE_ENV}=1 at TP1)"
    for i in range(m.start_layer, m.end_layer):
        reason = reason or hc.layer_supported(m.layers[i])
    reason = reason or hc_wide.build(limits=limits)
    megakernel_agree(reason, what="Qwen3.8 slot hc_wide")
    if reason:
        unavailable("hc_wide", reason)
    return not reason


def _build_gdn_pdl(model) -> bool:
    from sglang.srt.kernel_plugins.nvidia.binding import megakernel_agree

    from sgl_nvidia_mega.qwen3_8.runtime import gdn

    reason = gdn.verify_reasons(model)
    megakernel_agree(reason, what="Qwen3.8 slot gdn_native_pdl")
    if reason:
        unavailable("gdn_native_pdl", reason)
    return not reason


def _build_gdn_glue(model) -> bool:
    from sglang.srt.kernel_plugins.nvidia.binding import megakernel_agree

    from sgl_nvidia_mega.qwen3_8.runtime import gdn

    reason = gdn.glue_reasons(model)
    if not reason:
        try:
            from sgl_nvidia_mega.qwen3_8.kernels.ops.attention.linear import (  # noqa: F401
                gdn_fused_verify_glue,
            )
        except Exception as e:
            reason = f"the glue kernel does not import: {type(e).__name__}: {e}"
    megakernel_agree(reason, what="Qwen3.8 slot gdn_verify_glue")
    if reason:
        unavailable("gdn_verify_glue", reason)
    return not reason


def _build_gdn_front(model) -> bool:
    """Whether slot gdn_front runs, agreed on every rank. Its kernels compile at the
    first eager verify of each layout (a capture warmup), never while capturing."""
    import torch
    from sglang.srt.kernel_plugins.nvidia.binding import megakernel_agree

    from sgl_nvidia_mega.qwen3_8.runtime import gdn

    reason = gdn.front_reasons(model)
    if not reason and (
        not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 10
    ):
        reason = "the front's tcgen05 GEMM needs an SM10x GPU"
    if not reason:
        try:
            from sgl_nvidia_mega.qwen3_8.kernels.ops.attention.gdn_front import (  # noqa: F401
                mk_core,
                mk_front,
            )
        except Exception as e:
            reason = f"the front kernels do not import: {type(e).__name__}: {e}"
    megakernel_agree(reason, what="Qwen3.8 slot gdn_front")
    if reason:
        unavailable("gdn_front", reason)
    return not reason


def _build_qsa_indexer(model):
    """The indexer kernels (qsa_indexer.IndexerKernels), or None (stock indexer)."""
    import tempfile

    from sglang.srt.kernel_plugins.nvidia.binding import megakernel_agree

    from sgl_nvidia_mega.qwen3_8.runtime import qsa_indexer

    source_dir = os.environ.get(SOURCE_DIR_ENV) or os.path.join(
        tempfile.gettempdir(), "qwen3_8-megakernel-sources"
    )
    os.makedirs(source_dir, exist_ok=True)
    kernels = qsa_indexer.IndexerKernels()
    reason = qsa_indexer.build(model, kernels, source_dir=os.path.abspath(source_dir))
    megakernel_agree(reason, what="Qwen3.8 slot qsa_indexer")
    if reason:
        unavailable("qsa_indexer", reason)
        return None
    return kernels


def _build_qsa_draft(model):
    """The draft glue (qsa_draft.DraftGlue), or None (SGLang's lookup and capture)."""
    from sglang.srt.kernel_plugins.nvidia.binding import megakernel_agree

    from sgl_nvidia_mega.qwen3_8.runtime import qsa_draft

    reason = qsa_draft.build(model)
    megakernel_agree(reason, what="Qwen3.8 slot qsa_draft")
    if reason:
        unavailable("qsa_draft", reason)
        return None
    lookup = qsa_draft.lookup_reason(model)
    if lookup:
        logger.info(
            "NVIDIA megakernel: Qwen3.8 slot qsa_draft: stock lookup (%s)", lookup
        )
    return qsa_draft.DraftGlue(lookup=not lookup)


def _qsa_layers_reason(model) -> str:
    """'' when every attention layer is a QSA layer of the pool kernels' TP1 heads."""
    from sglang.srt.models.qwen4_exp import Qwen4ExpAttentionDecoderLayer

    from sgl_nvidia_mega.qwen3_8.runtime import attention

    m = model.model
    reason = ""
    for i in range(m.start_layer, m.end_layer):
        if isinstance(m.layers[i], Qwen4ExpAttentionDecoderLayer):
            reason = reason or attention.layer_reasons(m.layers[i])
    return reason


def _build_qsa_pool(model) -> bool:
    from sglang.srt.kernel_plugins.nvidia.binding import megakernel_agree

    from sgl_nvidia_mega.qwen3_8.runtime import attention

    reason = _qsa_layers_reason(model) or attention.build()
    megakernel_agree(reason, what="Qwen3.8 slot qsa_pool")
    if reason:
        unavailable("qsa_pool", reason)
    return not reason


def _build_qsa_core():
    """Slot qsa_pool's tcgen05 core (attention.CoreKernels) where
    SGLANG_NVIDIA_MEGAKERNEL_QWEN3_8_ATTN_CORE_V1=1 asks for it and it builds on every
    rank, else None."""
    from sglang.srt.kernel_plugins.nvidia.binding import megakernel_agree

    from sgl_nvidia_mega.qwen3_8.runtime import attention

    if not attention.ATTN_CORE_V1:
        return None
    reason = attention.core_build()
    megakernel_agree(reason, what="Qwen3.8 slot qsa_pool tcgen05 core")
    if reason:
        logger.info(
            "NVIDIA megakernel: Qwen3.8 slot qsa_pool: "
            "SGLANG_NVIDIA_MEGAKERNEL_QWEN3_8_ATTN_CORE_V1 is set but the tcgen05 core "
            "does not run (%s); the pool kernels serve",
            reason,
        )
        return None
    return attention.CoreKernels()


def _build_qsa_union(model):
    """The union kernels (attention.UnionKernels), or None (stock); the pool is checked
    at the first verify, the draft's at its first draft extend
    (Qwen3_8Executor._check_pools)."""
    from sglang.srt.kernel_plugins.nvidia.binding import megakernel_agree

    from sgl_nvidia_mega.qwen3_8.runtime import attention

    reason = _qsa_layers_reason(model) or attention.union_build()
    megakernel_agree(reason, what="Qwen3.8 slot qsa_pool_fp8")
    if reason:
        unavailable("qsa_pool_fp8", reason)
        return None
    return attention.UnionKernels()


def _build_moe_finalize(model, decompose) -> bool:
    from sglang.srt.kernel_plugins.nvidia.binding import megakernel_agree
    from sglang.srt.runtime_context import get_parallel

    from sgl_nvidia_mega.qwen3_8.runtime import moe

    m = model.model
    reason = "" if decompose else f"needs the S1 layers ({DECOMPOSE_ENV}=1 at TP1)"
    for i in range(m.start_layer, m.end_layer):
        reason = reason or moe.layer_reasons(m.layers[i])
    reason = reason or moe.build(get_parallel())
    megakernel_agree(reason, what="Qwen3.8 slot moe_finalize")
    if reason:
        unavailable("moe_finalize", reason)
    return not reason


def _build_pins(model, executor) -> None:
    """The tuning-pin tables whose files are set and the SM107 small-M BF16 routes when
    their switch is (each agreed on every rank)."""
    from sgl_nvidia_mega.qwen3_8.runtime import (
        bf16_gemm_pins,
        flashinfer_tactic_pins,
        sm107_bf16_smallm,
    )

    if os.environ.get(flashinfer_tactic_pins.FILE_ENV):
        executor.moe_pins = bool(
            _built_options("moe_tactic_pins", flashinfer_tactic_pins.build(model), True)
        )
    if os.environ.get(bf16_gemm_pins.FILE_ENV) and _built_options(
        "bf16_gemm_pins", bf16_gemm_pins.build(model, _gemm_call_sites(model)), True
    ):
        bf16_gemm_pins.enable_routes()
    if sm107_bf16_smallm.requested() and _built_options(
        "sm107_bf16_smallm", sm107_bf16_smallm.build(_gemm_call_sites(model)), True
    ):
        sm107_bf16_smallm.enable()


def _gemm_call_sites(model) -> dict:
    """(N, K) -> the executor's projection calls of this model with that weight."""
    from sglang.srt.models.qwen2_moe import Qwen2MoeSparseMoeBlock
    from sglang.srt.models.qwen4_exp import Qwen4ExpLinearDecoderLayer

    sites = {}

    def add(weight, name):
        if weight is not None:
            sites.setdefault(tuple(weight.shape), set()).add(name)

    def add_mix(hc):
        add(hc.input_mix_weight_down.weight, "hc.mix_down")
        add(hc.input_mix_weight_up.weight, "hc.mix_up")

    m = model.model
    for i in range(m.start_layer, m.end_layer):
        layer = m.layers[i]
        if isinstance(layer, Qwen4ExpLinearDecoderLayer):
            add(layer.linear_attn._fused_in_proj_weight, "gdn.in_proj")
            add(layer.linear_attn.out_proj.weight, "gdn.out_proj")
        else:
            add(layer.qkv_proj.weight, "qsa.qkv_proj")
            add(layer.o_proj.weight, "qsa.o_proj")
            if layer.is_qsa:
                add(layer.indexer.index_qk_proj.weight, "qsa.index_qk_proj")
        if isinstance(layer.mlp, Qwen2MoeSparseMoeBlock):
            add(layer.mlp.gate.weight, "moe.gate")
            if layer.mlp.shared_expert is not None:
                add(layer.mlp.shared_expert.gate_up_proj.weight, "moe.shared_gate_up")
                add(layer.mlp.shared_expert.down_proj.weight, "moe.shared_down")
        add_mix(layer.attn_hyper_connection)
        add_mix(layer.mlp_hyper_connection)
    add_mix(m.hyper_connection_mixer)
    add(model.lm_head.weight, "lm_head")
    return {shape: "/".join(sorted(names)) for shape, names in sites.items()}


def _built_options(slot: str, reason: str, options):
    """``options`` when the slot built on every rank ('' everywhere), else None."""
    from sglang.srt.kernel_plugins.nvidia.binding import megakernel_agree

    megakernel_agree(reason, what=f"Qwen3.8 slot {slot}")
    if reason:
        unavailable(slot, reason)
        return None
    return options


def unavailable(slot: str, reason: str) -> None:
    logger.info("NVIDIA megakernel: Qwen3.8 slot %s uses stock (%s)", slot, reason)
