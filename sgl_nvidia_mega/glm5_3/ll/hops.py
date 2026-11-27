# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""``ll.hops``: the GLM low-latency all-reduce hops of the fused verify step (TP4,
also built at TP8), on the plugin-owned Lamport buffer (``ll.transport``).

Every boundary where the stock forward sums a tensor-parallel partial is one hop:
all-reduce + residual add + RMSNorm in one kernel (P4: one-shot up to the measured
cap, two-shot above it, up to 48 rows).

* ``post_attention``: ``o_proj`` (``ll.linear("o_proj")``), then the hop into
  ``post_attention_layernorm``. On a MoE layer whose MLP takes it, the hop also
  writes the NVFP4 quant of its norm output (P10), the routed experts' FC1 input.
* ``pre_attention``: the previous MLP output into ``input_layernorm``. A
  ``DeferredFinalize`` (the MoE's unfinalized FC2 rows, ``ll.moe``) is finalized
  inside the hop (P9: the finalize + shared-expert add before the publish).
* ``final``: the last MLP output into the model's final norm.

The launch plan (one- or two-shot, cluster geometry, unicast or multicast publish,
shared-memory carveout) is a function of the world size, the row count and the SM
count only, so every rank runs the same hop sequence. The tables are GB300
measurements at H6144, chosen on the hop followed by its dependent kernel (the
router GEMM, the fused QKV-A GEMM) at the row counts the decode graphs replay.

Numerics (named tolerance ``ll.hops.max_ulp``): where the plan keeps FlashInfer's
one-shot geometry (inside its 1 MiB band) a hop is bit-identical to FlashInfer's
one-shot ``kARResidualRMSNorm`` (FlashInfer >= 0.7.3, which squares in fp32 like the
hop; 0.7.1 and 0.7.2 squared in bf16); the two-shot is bit-identical to FlashInfer's
two-shot + ``rmsNormLamport``; above the band the sum-of-squares order follows the
chosen cluster (``test_glm5_3_ll_hops.py``).
"""

from __future__ import annotations

import functools
import logging
import os
from dataclasses import dataclass, replace
from typing import Any, Dict, Optional, Tuple

import torch

from sgl_nvidia_mega.glm5_3.boundary.boundaries import HopSteps
from sgl_nvidia_mega.glm5_3.ll import transport
from sgl_nvidia_mega.glm5_3.ll.transport import Launch
from sgl_nvidia_mega.glm5_3.registry import Candidate, Decline, DeclinedFamily

SGLANG_READS = {
    "srt/distributed/parallel_state.py": ("get_tp_group", "device_group", "cpu_group"),
}

logger = logging.getLogger(__name__)

HIDDEN = 6144
# The hop bound when the profile names none (the TP8 hops' bound).
DEADLINE_S = 20.0
# Error sites: layer * 4 + {0: post_attention, 1: pre_attention (the MLP output),
# 2: final}; sites from EXTERNAL_SITES on are other families' (the draft's).
HOPS_PER_LAYER = 4
EXTERNAL_SITES = 1000

# Named tolerance (test_glm5_3_ll_hops.py): the norm output against an fp32
# rank-order reference with fp32 squares of the bf16 sum, in bf16 ulps of the
# reference. Both forms square in fp32 (the one-shot as FlashInfer's does since
# v0.7.3); the sum-of-squares order follows the cluster, and division and rsqrt
# rounding stay within 1 ulp.
TOLERANCES = {"ll.hops.max_ulp": 2.0}

_VEC = 8  # bf16 per 16-byte vector, one vector per thread
_MAX_CLUSTER = 8
# FlashInfer v0.7.1 MNNVL_ONE_SHOT_THRESHOLD (Lamport bytes per rank): inside it the
# plan keeps FlashInfer's one-shot geometry (FlashInfer's bits).
FI_ONESHOT_MAX_BYTES = 64 * 1024 * 8 * 2


# ============================================================== geometry
@functools.lru_cache(maxsize=None)
def flashinfer_oneshot_grid(
    rows: int, hidden: int, sm_count: int
) -> Tuple[int, int, int]:
    """(block, cluster, loads per thread) of FlashInfer v0.7.1
    ``adjustGridConfig(rows, hidden, 8, useCluster=true)``: the sum-of-squares order
    follows it, so bit-identity with FlashInfer needs exactly this geometry."""
    threads = -(-hidden // _VEC)
    cluster, loads = _MAX_CLUSTER, 1
    while threads % cluster != 0 and cluster > 1:
        cluster //= 2
    max_divisible = cluster
    block = -(-threads // cluster)
    while block < 128 and cluster >= 2:
        block *= 2
        cluster //= 2
    while rows * cluster > sm_count and cluster > 1 and block <= 512:
        block *= 2
        cluster //= 2
    while cluster < max_divisible:
        cand_cluster = cluster * 2
        cand_block = -(-threads // cand_cluster)
        if cand_block < 64 or rows * cand_cluster > sm_count:
            break
        cluster, block = cand_cluster, cand_block
    while block > 1024:
        if cluster < _MAX_CLUSTER:
            cluster <<= 1
        elif loads < 8:
            loads += 1
        else:
            break
        block = -(-threads // (cluster * loads))
    return block, cluster, loads


def _whole(block: int, cluster: int, loads: int, hidden: int, min_cluster: int) -> bool:
    return (
        loads == 1
        and block <= 1024
        and block % 32 == 0
        and min_cluster <= cluster <= _MAX_CLUSTER
        and block * cluster * _VEC == hidden
    )


def oneshot_geometry(
    rows: int, hidden: int, sm_count: int
) -> Optional[Tuple[int, int]]:
    """FlashInfer's one-shot geometry when it is a cluster of 2-8 CTAs of whole warps
    covering the row exactly (H6144: up to sm_count / 2 rows), else None."""
    if rows <= 0 or hidden % _VEC:
        return None
    block, cluster, loads = flashinfer_oneshot_grid(rows, hidden, sm_count)
    return (block, cluster) if _whole(block, cluster, loads, hidden, 2) else None


def twoshot_geometry(
    rows: int, hidden: int, sm_count: int
) -> Optional[Tuple[int, int]]:
    """FlashInfer's rmsNormLamport geometry (single CTAs allowed), else None."""
    if rows <= 0 or hidden % _VEC:
        return None
    block, cluster, loads = flashinfer_oneshot_grid(rows, hidden, sm_count)
    return (block, cluster) if _whole(block, cluster, loads, hidden, 1) else None


def cluster_geometry(rows: int, hidden: int, cluster: int) -> Optional[Tuple[int, int]]:
    """A cluster of ``cluster`` CTAs of whole warps covering the row, else None."""
    if rows <= 0 or hidden % _VEC or not 2 <= cluster <= _MAX_CLUSTER:
        return None
    threads = hidden // _VEC
    if threads % cluster:
        return None
    block = threads // cluster
    return (block, cluster) if block % 32 == 0 and block <= 1024 else None


# ============================================================== launch plan
@dataclass(frozen=True)
class WorldDefaults:
    oneshot_max_bytes: int  # one-shot cap (Lamport bytes per rank)
    unicast_min_bytes: Optional[int]  # unicast publish from here on (None: never)
    cluster_above_band: Dict[int, int]  # above the band: FlashInfer's cluster -> used
    cluster_above_bytes: Optional[Tuple[int, int]]  # (bytes, cluster): above, always
    carveout_rows: Tuple[Tuple[int, int], ...]  # inclusive rows with max carveout
    twoshot_max_rows: int  # the two-shot kernel above the one-shot cap, up to here
    finalize_twoshot: bool  # the finalize hop above the cap: two-shot (else one-shot)


# GB300 (152 SMs) at H6144, hop + dependent chains:
# - W=4: one-shot up to 36 rows (42 ties two-shot once the dependents run); unicast
#   publish from 8 rows (384 KiB); FlashInfer's 2-CTA clusters (from sm / 4 rows)
#   lose to 8; carveout dropped at 9-16 rows (it shrinks the tiny router GEMM's L1).
#   Two-shot 37-48 rows; the finalize hop at 42-48 rows stays one-shot on
#   FlashInfer's 2-CTA geometry (0.7-0.8 us faster than its two-shot in the chain).
# - W=8: multicast; one-shot up to 18 rows, 4-CTA clusters above 1.5 MiB (17-18
#   rows); two-shot from 19 rows, also for the finalize hop.
# - W=2: not measured (the previous 1.5 MiB, multicast, FlashInfer's geometry).
WORLD_DEFAULTS = {
    2: WorldDefaults(1536 * 1024, None, {}, None, ((1, 2048),), 0, False),
    4: WorldDefaults(
        36 * 6144 * 4 * 2, 384 * 1024, {2: 8}, None, ((1, 8), (17, 2048)), 48, False
    ),
    8: WorldDefaults(
        18 * 6144 * 8 * 2,
        None,
        {},
        (1536 * 1024, 4),
        ((1, 11), (13, 15), (17, 2048)),
        48,
        True,
    ),
}
# The finalize hop above the one-shot cap when it stays one-shot: 1 = FlashInfer's
# one-shot geometry, 0 = the plain rule, 2 / 4 / 8 = that cluster.
FINALIZE_CLUSTER = 1
# P10: rows where the standalone quantize stays (the shared expert
# loses its SM race against an earlier FC1): TP4 1-3 rows, TP8 11-16, 36.
# (TP4 21 rows removed in v6: fused P10 saves ~300 us/step at K=6 bs=3.)
NVFP4_SKIP_ROWS = {4: ((1, 3),), 8: ((11, 16), (36, 36))}


def world_defaults(world: int) -> WorldDefaults:
    return WORLD_DEFAULTS.get(world, WORLD_DEFAULTS[2])


def max_carveout(world: int, rows: int) -> bool:
    return any(lo <= rows <= hi for lo, hi in world_defaults(world).carveout_rows)


def oneshot_plan(
    world: int, rows: int, hidden: int, sm_count: int, *, force: bool = False
) -> Optional[Launch]:
    """The one-shot launch of a hop of ``rows`` rows, None above the cap (``force``:
    regardless of the cap)."""
    d = world_defaults(world)
    nbytes = transport.oneshot_bytes(world, rows, hidden)
    if not force and nbytes > d.oneshot_max_bytes:
        return None
    geometry = oneshot_geometry(rows, hidden, sm_count)
    if nbytes > FI_ONESHOT_MAX_BYTES:
        fi_cluster = flashinfer_oneshot_grid(rows, hidden, sm_count)[1]
        cluster = d.cluster_above_band.get(fi_cluster, 0)
        if d.cluster_above_bytes is not None and nbytes > d.cluster_above_bytes[0]:
            cluster = d.cluster_above_bytes[1]
        if cluster:
            geometry = cluster_geometry(rows, hidden, cluster) or geometry
    if geometry is None:
        return None
    unicast = d.unicast_min_bytes is not None and nbytes >= d.unicast_min_bytes
    return Launch(*geometry, unicast, max_carveout(world, rows))


def twoshot_plan(world: int, rows: int, hidden: int, sm_count: int) -> Optional[Launch]:
    if rows > world_defaults(world).twoshot_max_rows:
        return None
    geometry = twoshot_geometry(rows, hidden, sm_count)
    if geometry is None:
        return None
    return Launch(*geometry, False, max_carveout(world, rows), two_shot=True)


def plain_plan(world: int, rows: int, hidden: int, sm_count: int) -> Optional[Launch]:
    """The plain hop: one-shot up to the cap, else the two-shot kernel, else None."""
    return oneshot_plan(world, rows, hidden, sm_count) or twoshot_plan(
        world, rows, hidden, sm_count
    )


def finalize_plan(
    world: int, rows: int, hidden: int, sm_count: int
) -> Optional[Launch]:
    """The finalize hop: the plain hop's one-shot plan up to the cap (so it equals the
    finalize followed by the plain hop bit for bit); above it two-shot where measured
    better (W=8), else one-shot on ``FINALIZE_CLUSTER``'s geometry with the forced
    one-shot plan's publish and carveout."""
    plan = oneshot_plan(world, rows, hidden, sm_count)
    if plan is not None:
        return plan
    if world_defaults(world).finalize_twoshot:
        plan = twoshot_plan(world, rows, hidden, sm_count)
        if plan is not None:
            return plan
    if rows > max(world_defaults(world).twoshot_max_rows, 0):
        return None
    plan = oneshot_plan(world, rows, hidden, sm_count, force=True)
    if plan is None or FINALIZE_CLUSTER == 0:
        return plan
    geometry = (
        oneshot_geometry(rows, hidden, sm_count)
        if FINALIZE_CLUSTER == 1
        else cluster_geometry(rows, hidden, FINALIZE_CLUSTER)
    )
    return replace(plan, block=geometry[0], cluster=geometry[1]) if geometry else plan


def nvfp4_skipped(world: int, rows: int) -> bool:
    return any(lo <= rows <= hi for lo, hi in NVFP4_SKIP_ROWS.get(world, ()))


def nvfp4_plan(world: int, rows: int, hidden: int, sm_count: int) -> Optional[Launch]:
    """The NVFP4-emitting post-attention hop (P10): the plain plan, None at the
    skip rows (the MoE quantizes its own input there)."""
    if nvfp4_skipped(world, rows):
        return None
    return plain_plan(world, rows, hidden, sm_count)


# ============================================================== the MoE handoff
@dataclass
class DeferredFinalize:
    """The unfinalized output of a MoE layer (``ll.moe``, contract token
    ``mlp:deferred_finalize``): FlashInfer trtllm-gen ``do_finalize=False``'s FC2 rows
    ``[P, >= H]`` bf16, the routing weights ``[M, top_k]`` (routed scaling folded in),
    ``expanded_idx_to_permuted_idx`` ``[M, top_k]`` int32 (-1: empty slot), and the
    shared-expert output ``[M, H]`` bf16 (or None). The next hop finalizes it."""

    gemm2_out: torch.Tensor
    expert_weights: torch.Tensor
    permuted_idx: Optional[torch.Tensor]
    shared_output: Optional[torch.Tensor]
    top_k: int
    identity_rows: bool = False

    @property
    def rows(self) -> int:
        return int(self.expert_weights.shape[0])


@dataclass
class DeferredFp32:
    """Atomic-accumulated [M, H] fp32 MoE partial (+ optional [M, H] bf16 shared
    output) consumed and zeroed in place by the next hop (GlmLLMnnvlFp32ZeroOneshot)."""

    fp32_acc: torch.Tensor
    shared_output: Optional[torch.Tensor] = None

    @property
    def rows(self) -> int:
        return int(self.fp32_acc.shape[0])


@dataclass
class Nvfp4Input:
    """The NVFP4 quant of a MoE layer's input (P10), handed from its post-attention
    hop to its routed experts: ``fp4`` ``[M, H / 2]`` uint8, ``sf`` ``[M, H / 16]``
    e4m3 (linear layout), of the hop's ``norm`` output under ``global_scale``."""

    layer_id: int
    norm: torch.Tensor
    fp4: torch.Tensor
    sf: torch.Tensor
    global_scale: torch.Tensor


# ============================================================== the family
class LlHopFamily:
    """The owned Lamport buffer, the built hop modules and the row plans of one
    model's tensor-parallel group (collective to build)."""

    def __init__(
        self,
        *,
        device_group,
        cpu_group,
        eps: float,
        hidden: int = HIDDEN,
        deadline_s: float = DEADLINE_S,
        sm_count: Optional[int] = None,
        allocator: str = "symm_mem",
        nvfp4: bool = True,
        finalize_dtypes: Tuple[torch.dtype, ...] = (torch.float32, torch.bfloat16),
        top_k: int = 8,
    ) -> None:
        self.device_group, self.cpu_group = device_group, cpu_group
        self.world = torch.distributed.get_world_size(device_group)
        self.rank = torch.distributed.get_rank(device_group)
        self.eps, self.hidden, self.deadline_s = float(eps), hidden, deadline_s
        self.device = torch.device("cuda", torch.cuda.current_device())
        self.sm_count = (
            sm_count
            or torch.cuda.get_device_properties(self.device).multi_processor_count
        )
        self.allocator, self.nvfp4, self.top_k = allocator, nvfp4, top_k
        self.finalize_dtypes = finalize_dtypes
        self.buffer: Optional[transport.LamportBuffer] = None
        self.site_names: Dict[int, str] = {}
        self.max_rows = 0
        self.model: Any = None  # set by ``build``: the o_proj GEMMs are prepared too
        # the GEMM plans' target (``ll.target``), set by ``_family`` from the profile;
        # None: o_proj keeps SGLang's GEMM
        self.target: Any = None
        self.derived: Any = None
        cap = (
            torch.cuda.get_device_capability(self.device)
            if torch.cuda.is_available()
            else (0, 0)
        )
        self.sm: Tuple[int, int] = (int(cap[0]), int(cap[1]))
        # SM103 TP4 (world=4, rows <= 8) fused OProjAr + MoeFoldArNorm hops
        self.sm103_kernels: Any = None
        self.sm103_layers: list = []
        self.sm103_first: int = 0
        self.sm103_last: int = 0
        self.sm103_is_moe: list = []
        self.sm103_w_o: list = []
        self.sm103_router: list = []
        self.sm103_dnorm_layers: frozenset = frozenset()
        self._sm103_keep: list = []

    # ---------------------------------------------------------- build
    def reason(self, rows: int) -> Optional[str]:
        """None when the hops serve ``rows`` rows, else the decline reason."""
        if self.world not in (4, 8):
            return f"tensor-parallel size {self.world} (the LL hops serve 4 and 8)"
        if plain_plan(self.world, rows, self.hidden, self.sm_count) is None:
            return f"no hop plan at {rows} rows (W={self.world})"
        if finalize_plan(self.world, rows, self.hidden, self.sm_count) is None:
            return f"no finalize hop plan at {rows} rows (W={self.world})"
        return None

    def compile(self, widths) -> None:
        """Build every module and create the buffer for ``widths`` (collective). A
        build failure on any rank raises on every rank."""
        widths = sorted(set(int(w) for w in widths))
        self.max_rows = max(widths)
        err = ""
        try:
            transport.compile_all(
                self.world,
                finalize=tuple((self.top_k, d) for d in self.finalize_dtypes),
                nvfp4=self.nvfp4,
            )
        except Exception as e:  # noqa: BLE001 - gathered and raised on every rank
            err = f"rank {self.rank}: {type(e).__name__}: {' '.join(str(e).split())[:400]}"
        gathered = [None] * self.world
        torch.distributed.all_gather_object(gathered, err, group=self.cpu_group)
        if any(gathered):
            raise RuntimeError(
                "GLM LL hop kernels failed to build (opt out with the ll.hops unit "
                "switch): " + "; ".join(g for g in gathered if g)
            )
        if self.model is not None:
            from sgl_nvidia_mega.glm5_3 import ll

            inner = self.model.model
            for i in range(inner.start_layer, inner.end_layer):
                ll.build_linear(
                    "o_proj",
                    inner.layers[i].self_attn.o_proj,
                    widths,
                    target=self.target,
                    sms=self.sm_count,
                )
        self.buffer = transport.LamportBuffer.create(
            device_group=self.device_group,
            cpu_group=self.cpu_group,
            max_rows=self.max_rows,
            hidden=self.hidden,
            deadline_s=self.deadline_s,
            allocator=self.allocator,
            device=self.device,
        )
        self._compile_sm103(widths)
        if self.rank == 0:
            logger.info(
                "GLM LL hops: W=%d, rows %s, buffer %s, nvfp4=%s, sm103_hop=%s; plans %s",
                self.world,
                widths,
                self.buffer.describe(),
                self.nvfp4,
                self.sm103_kernels is not None,
                self.describe_plans(widths),
            )

    def _compile_sm103(self, widths) -> None:
        if (
            os.environ.get("SGLANG_NVIDIA_MEGAKERNEL_GLM_TP4_SM103_HOP_MOE", "1") != "1"
            or self.world != 4
            or self.sm != (10, 3)
            or self.model is None
            or not any(1 <= w <= 8 for w in widths)
        ):
            return
        from sgl_nvidia_mega.glm5_3.runtime import SLOTS
        from sgl_nvidia_mega.glm5_3.runtime.compile import compile_cached
        from sgl_nvidia_mega.glm5_3.sm103.hop import api

        inner, config = self.model.model, self.model.config
        span = range(inner.start_layer, inner.end_layer)
        layers = [inner.layers[i] for i in span]
        is_moe = [bool(getattr(layer, "is_layer_sparse", False)) for layer in layers]
        w_o = [layer.self_attn.o_proj.weight for layer in layers]

        reason = ""
        for w in w_o:
            if (
                w.dtype != torch.bfloat16
                or tuple(w.shape) != (HIDDEN, 4096)
                or not w.is_contiguous()
            ):
                reason = f"o_proj weight {w.dtype} {tuple(w.shape)} (need bf16 [6144, 4096])"
                break
        if not reason:
            for layer, moe in zip(layers, is_moe):
                if moe:
                    bias = getattr(layer.mlp.gate, "e_score_correction_bias", None)
                    if bias is None or bias.dtype != torch.float32:
                        reason = "router has no fp32 score-correction bias"
                        break
        if not reason:
            import torch.distributed._symmetric_memory as symm_mem

            get_backend = getattr(symm_mem, "get_backend", None)
            backend = get_backend(self.device) if get_backend is not None else None
            if backend not in (None, "CUDA"):
                reason = f"symmetric-memory backend {backend!r}"
            elif not api.fenced_trigger_supported():
                reason = "fenced_trigger_supported is False"

        dnorm_on = (
            os.environ.get("SGLANG_NVIDIA_MEGAKERNEL_GLM_TP4_SM103_ATTN_FRONT", "1")
            == "1"
            and os.environ.get("SGLANG_NVIDIA_MEGAKERNEL_GLM_TP4_SM103_DNORM", "1")
            == "1"
            and "ll.attention" in SLOTS
            and "dnorm" in SLOTS
            and "qkv_a_small_m_shared" in SLOTS
        )
        dnorm_layers = frozenset(
            layer.layer_id
            for i, layer in enumerate(layers)
            if dnorm_on
            and i > 0
            and is_moe[i - 1]
            and getattr(layer.self_attn, "skip_topk", False)
        )

        specs = list(
            api.unit_specs(
                world=4, o_proj_in=4096, quarters=8, include_instance_b=False
            )
        )
        names = {
            "allreduce_norm",
            "o_proj_produce",
            "o_proj_consume",
            "o_proj_produce_router",
            "o_proj_consume_router",
            "moe_fold_split",
            "moe_fold_plain",
        }
        n_as = (
            api.oproj_async_min_rows()
            if api.OPROJ_ASYNC_STAGE_MIN_ROWS_ENV in os.environ
            else 4
        )
        if n_as and any(n_as <= m <= 8 for m in widths):
            specs += api.oproj_async_specs(world=4, o_proj_in=4096)
            names |= {"o_proj_produce_as", "o_proj_produce_router_as"}
        raw_remap = (
            api.oproj_remap()
            if api.OPROJ_REMAP_ENV in os.environ
            else api.oproj_remap("4-8@0")
        )
        remap = {m: p for m, p in raw_remap.items() if m in widths and m <= 8}
        if remap:
            rf = api.oproj_remap_specs(
                remap.values(),
                [p for m, p in remap.items() if n_as and m >= n_as],
                world=4,
                o_proj_in=4096,
            )
            specs += rf
            names |= {sp.name for sp in rf}
        if dnorm_layers:
            names |= {"moe_fold_split_dn", "moe_fold_plain_dn"}
            if "split8dn" in api.fold_form():
                names |= {"moe_fold_split_dn_nc"}

        compiled, probe = {}, None
        if not reason:
            try:
                import cutlass.cute as cute

                for spec in (s for s in specs if s.name in names):
                    compiled[spec.name] = compile_cached(
                        spec.obj,
                        spec.constexprs,
                        lambda spec=spec: cute.compile(spec.obj, *spec.args),
                    )
                    if spec.name == "moe_fold_split" and not hasattr(
                        compiled[spec.name], "library"
                    ):
                        probe = cute.compile(spec.obj, *spec.args)
            except Exception as e:  # noqa: BLE001
                reason = f"{type(e).__name__}: {' '.join(str(e).split())[:300]}"

        gathered = [None] * self.world
        torch.distributed.all_gather_object(gathered, reason, group=self.cpu_group)
        if any(gathered):
            if self.rank == 0:
                logger.warning(
                    "GLM LL hops: SM103 TP4 fused OProjAr/MoeFold declined: %s",
                    "; ".join(g for g in gathered if g),
                )
            return

        router = []
        keep = []
        for layer, moe in zip(layers, is_moe):
            hoist = None
            if moe:
                hoist = api.router_hoist_weights(
                    layer.mlp.gate.weight,
                    layer.post_attention_layernorm.weight,
                    layer.self_attn.o_proj.weight,
                    self.rank,
                    world=4,
                )
                for t in hoist:
                    if self.derived is not None:
                        self.derived.add("ll.hops.router_hoist", t)
                    keep.append(t)
            router.append(hoist)

        rscale = float(getattr(config, "routed_scaling_factor", 1.0))
        kernels = api.HopKernels.build(
            compiled,
            self.device_group.group_name,
            self.device,
            self.eps,
            rscale,
            self.device.index,
            instance_b=False,
            router=True,
            probe=probe,
            barrier_group=self.cpu_group,
            world=4,
            async_min_rows=n_as,
            remap=raw_remap,
        )
        limits = [None] * self.world
        torch.distributed.all_gather_object(
            limits, kernels.fold_limits(), group=self.cpu_group
        )
        if len(set(limits)) != 1:
            raise RuntimeError(
                f"GLM LL hops SM103 TP4 split-fold limits disagree across ranks: {limits}"
            )
        self.sm103_kernels = kernels
        self.sm103_layers = layers
        self.sm103_first = inner.start_layer
        self.sm103_last = inner.end_layer - 1
        self.sm103_is_moe = is_moe
        self.sm103_w_o = w_o
        self.sm103_router = router
        self.sm103_dnorm_layers = dnorm_layers
        self._sm103_keep = keep
        if self.rank == 0:
            logger.info(
                "GLM LL hops: SM103 TP4 OProjAr+Router + MoeFoldArNorm (world=4, quarters=8, dnorm=%d layers, fold_limits=%s) built for rows <= 8",
                len(dnorm_layers),
                kernels.fold_limits(),
            )

    def describe_plans(self, widths) -> str:
        def label(plan: Optional[Launch]) -> str:
            if plan is None:
                return "-"
            kind = "ts" if plan.two_shot else ("uc" if plan.unicast else "mc")
            return f"{kind}{plan.cluster}x{plan.block}{'+carve' if plan.max_carveout else ''}"

        args = (self.world, 0, self.hidden, self.sm_count)
        out = []
        for m in widths:
            a = (args[0], m, *args[2:])
            out.append(
                f"{m}: {label(plain_plan(*a))} / fin {label(finalize_plan(*a))}"
                + (f" / fp4 {label(nvfp4_plan(*a))}" if self.nvfp4 else "")
            )
        return "; ".join(out)

    # ---------------------------------------------------------- errors
    def register_sites(self, names: Dict[int, str]) -> None:
        """Name other families' sites (>= ``EXTERNAL_SITES``) in this error record."""
        for site in names:
            if int(site) < EXTERNAL_SITES:
                raise ValueError(f"error site {site} is in the hop family's range")
        self.site_names.update({int(s): str(n) for s, n in names.items()})

    def error_report(self) -> Optional[dict]:
        """The decoded error record of a timed-out hop wait, or None (host memory:
        readable after the CUDA context is lost)."""
        rec = self.buffer.err.read() if self.buffer is not None else None
        if rec is None and self.sm103_kernels is not None:
            rec = self.sm103_kernels.err.read()
        if rec is None:
            return None
        site = int(rec["site"])
        if site >= EXTERNAL_SITES:
            return {**rec, "layer": None, "hop": self.site_names.get(site, "external")}
        layer, kind = divmod(site, HOPS_PER_LAYER)
        hop = ("post_attention", "mlp_output", "final")[kind] if kind < 3 else str(kind)
        return {**rec, "layer": layer, "hop": hop}

    # ---------------------------------------------------------- hops
    def _need(self, plan: Optional[Launch], rows: int, what: str) -> Launch:
        if plan is None:
            raise RuntimeError(f"GLM LL {what} hop has no plan at {rows} rows")
        return plan

    def allreduce_norm(self, x, residual, gamma, *, site: int):
        """``(norm, residual out)`` of ``residual`` + the group sum of ``x``.
        ``residual`` None: no residual (``residual out`` is the sum itself), e.g. a
        row-parallel projection's partial into the first norm of a layer (the
        draft's eh_proj, P6a). Callable from other families with ``site >=
        EXTERNAL_SITES``."""
        rows = x.shape[0]
        if residual is None:
            residual = torch.zeros_like(x)
        plan = self._need(
            plain_plan(self.world, rows, self.hidden, self.sm_count), rows, "plain"
        )
        return transport.ar_rmsnorm(
            self.buffer, x, residual, gamma, self.eps, plan, site=site
        )

    def allreduce_norm_nvfp4(self, x, residual, gamma, global_scale, *, site: int):
        """``(norm, residual out, fp4, sf)``, or None at a P10 skip row."""
        rows = x.shape[0]
        plan = nvfp4_plan(self.world, rows, self.hidden, self.sm_count)
        if plan is None:
            return None
        return transport.ar_rmsnorm_nvfp4(
            self.buffer,
            x,
            residual,
            gamma,
            self.eps,
            plan,
            global_scale=global_scale,
            site=site,
        )

    def finalize(self, fin: DeferredFinalize, residual, gamma, *, site: int):
        """``(norm, residual out)`` of the MoE finalize of ``fin`` (+ shared) summed
        over the group, added to ``residual`` and normed."""
        rows = residual.shape[0]
        if fin.rows != rows:
            raise RuntimeError(f"MoE handoff of {fin.rows} rows at a {rows}-row hop")
        plan = self._need(
            finalize_plan(self.world, rows, self.hidden, self.sm_count),
            rows,
            "finalize",
        )
        idx = fin.permuted_idx
        if idx is not None and idx.dim() == 1:  # FlashInfer's [M * top_k] layout
            idx = idx.view(rows, fin.top_k)
        return transport.finalize_ar_rmsnorm(
            self.buffer,
            gemm2_out=fin.gemm2_out,
            expert_weights=fin.expert_weights,
            permuted_idx=idx,
            shared_output=fin.shared_output,
            residual=residual,
            gamma=gamma,
            eps=self.eps,
            launch=plan,
            site=site,
            identity_rows=fin.identity_rows,
        )

    def fp32_zero(self, df: DeferredFp32, residual, gamma, *, site: int):
        """``(norm, residual out)`` consuming an atomic-accumulated [M, H] fp32 MoE
        partial (zeroed in place) + optional bf16 shared partial."""
        rows = residual.shape[0]
        if df.rows != rows:
            raise RuntimeError(f"MoE fp32 handoff of {df.rows} rows at a {rows}-row hop")
        plan = self._need(
            plain_plan(self.world, rows, self.hidden, self.sm_count),
            rows,
            "fp32_zero",
        )
        return transport.fp32_zero_ar_rmsnorm(
            self.buffer,
            df.fp32_acc,
            df.shared_output,
            residual,
            gamma,
            self.eps,
            plan,
            site=site,
        )

    def sm103_o_proj(self, layer, o_input, residual, step=None):
        i = layer.layer_id - self.sm103_first
        site = self.site(layer.layer_id, 0)
        out_res, out_norm = torch.empty_like(residual), torch.empty_like(residual)
        tail = (residual, layer.post_attention_layernorm.weight, out_res, out_norm)
        if step is None:
            self.sm103_kernels.o_proj(o_input, self.sm103_w_o[i], *tail, site=site)
            return out_norm, out_res
        rows, dev = residual.shape[0], residual.device
        ids = torch.empty((rows, self.top_k), dtype=torch.int32, device=dev)
        wts = torch.empty((rows, self.top_k), dtype=torch.float32, device=dev)
        bias = layer.mlp.gate.e_score_correction_bias
        self.sm103_kernels.o_proj_router(
            o_input,
            self.sm103_w_o[i],
            *self.sm103_router[i],
            bias,
            *tail,
            ids,
            wts,
            site=site,
        )
        step.routing = (ids, wts)
        return out_norm, out_res

    def sm103_allreduce_norm(self, partial, residual, gamma, *, site: int):
        out_res, out_norm = torch.empty_like(residual), torch.empty_like(residual)
        self.sm103_kernels.allreduce_norm(
            partial, residual, gamma, out_res, out_norm, site=site
        )
        return out_norm, out_res

    def sm103_fold(self, rec, residual, gamma, *, site: int, dn: bool = False):
        rows = residual.shape[0]
        out_res, out = torch.empty_like(residual), torch.empty_like(residual)
        args = (rec.pq, rec.shpartials, rec.ids_fin, rec.w_fin, rec.g2s, residual)
        if dn:
            bsum = torch.empty((rows, 8), dtype=torch.float32, device=residual.device)
            self.sm103_kernels.moe_fold_dn(
                *args, gamma, out_res, out, bsum, site=site
            )
            return out, out_res, bsum
        self.sm103_kernels.moe_fold(*args, gamma, out_res, out, site=site)
        return out, out_res

    @staticmethod
    def site(layer_id: int, kind: int) -> int:
        return layer_id * HOPS_PER_LAYER + kind

    # ---------------------------------------------------------- registry
    def provider(self, candidate: Candidate, rows: int, mode: str) -> LlHops:
        """The family protocol (``registry``): the hop steps at ``rows``."""
        if self.buffer is None or rows > self.max_rows:
            raise Decline(f"the LL hops are not built for {rows} rows")
        return LlHops(self, rows)


# ============================================================== the step group
def _o_proj(layer, o_input: torch.Tensor) -> torch.Tensor:
    """This rank's ``o_proj`` partial (``ll.linear("o_proj")``: WP3's registered GEMM
    when present, else the module's own local GEMM)."""
    from sgl_nvidia_mega.glm5_3 import ll

    return ll.linear("o_proj", layer.self_attn.o_proj, o_input)


class LlHops(HopSteps):
    """The hop steps of one verify width on ``ll.hops``. ``nvfp4`` (wired by the
    registry when a taken MLP unit consumes ``mlp_input:nvfp4``): the post-attention
    hop of a MoE layer also writes the NVFP4 quant of its norm output for that MLP
    (``step.nvfp4_input``), except at the P10 skip rows."""

    def __init__(self, family: LlHopFamily, rows: int, *, nvfp4: bool = False) -> None:
        self.f, self.rows, self.nvfp4 = family, rows, nvfp4
        self.use_sm103 = rows <= 8 and family.sm103_kernels is not None

    def wire(self, token: str) -> None:
        """A taken unit consumes ``token`` at this width (``registry``)."""
        if token == "mlp_input:nvfp4":
            if not self.f.nvfp4:
                raise RuntimeError("the NVFP4 hop variant is not built")
            self.nvfp4 = True

    def _hop(self, mlp_out, residual, gamma, site: int, layer_id: Optional[int] = None, step=None):
        if self.use_sm103:
            if not isinstance(mlp_out, (torch.Tensor, DeferredFinalize, DeferredFp32)):
                if (
                    step is not None
                    and layer_id is not None
                    and layer_id in self.f.sm103_dnorm_layers
                ):
                    u, out_res, bsum = self.f.sm103_fold(
                        mlp_out, residual, gamma, site=site, dn=True
                    )
                    step.dnorm = (layer_id, bsum)
                    return u, out_res
                return self.f.sm103_fold(mlp_out, residual, gamma, site=site)
            if isinstance(mlp_out, torch.Tensor):
                return self.f.sm103_allreduce_norm(
                    mlp_out, residual, gamma, site=site
                )
        if isinstance(mlp_out, DeferredFp32):
            return self.f.fp32_zero(mlp_out, residual, gamma, site=site)
        if isinstance(mlp_out, DeferredFinalize):
            return self.f.finalize(mlp_out, residual, gamma, site=site)
        if isinstance(mlp_out, torch.Tensor):
            return self.f.allreduce_norm(mlp_out, residual, gamma, site=site)
        raise TypeError(f"GLM LL hop: unexpected MLP output {type(mlp_out).__name__}")

    def pre_attention(self, layer, mlp_out, residual, step):
        if residual is None:  # the first layer: the embedding, no all-reduce
            return super().pre_attention(layer, mlp_out, residual, step)
        norm = layer.input_layernorm
        return self._hop(
            mlp_out,
            residual,
            norm.weight,
            self.f.site(layer.layer_id, 1),
            layer_id=layer.layer_id,
            step=step,
        )

    def post_attention(self, layer, o_input, residual, step):
        if self.use_sm103:
            i = layer.layer_id - self.f.sm103_first
            if self.f.sm103_is_moe[i]:
                return self.f.sm103_o_proj(layer, o_input, residual, step)
            return self.f.sm103_o_proj(layer, o_input, residual, None)
        partial = _o_proj(layer, o_input)
        gamma = layer.post_attention_layernorm.weight
        site = self.f.site(layer.layer_id, 0)
        scale = self._nvfp4_scale(layer) if self.nvfp4 else None
        if scale is not None:
            out = self.f.allreduce_norm_nvfp4(
                partial, residual, gamma, scale, site=site
            )
            if out is not None:
                norm, res, fp4, sf = out
                step.nvfp4_input = Nvfp4Input(layer.layer_id, norm, fp4, sf, scale)
                return norm, res
        return self.f.allreduce_norm(partial, residual, gamma, site=site)

    def final(self, model, mlp_out, residual, step) -> torch.Tensor:
        last = model.model.end_layer - 1
        return self._hop(
            mlp_out, residual, model.model.norm.weight, self.f.site(last, 2)
        )[0]

    @staticmethod
    def _nvfp4_scale(layer) -> Optional[torch.Tensor]:
        """The routed experts' FC1 input global scale of a MoE layer, else None."""
        experts = getattr(getattr(layer, "mlp", None), "experts", None)
        scale = getattr(experts, "w13_input_scale_quant", None)
        if not isinstance(scale, torch.Tensor) or scale.numel() != 1:
            return None
        return scale if scale.dtype == torch.float32 else None


# ============================================================== model build
def model_reason(model) -> Optional[str]:
    """Why this model's hops cannot run on ``ll.hops`` (None if they can)."""
    inner, config = model.model, model.config
    eps = float(config.rms_norm_eps)
    if int(config.hidden_size) != HIDDEN:
        return f"hidden size {config.hidden_size} (the plans are measured at {HIDDEN})"
    for i in range(inner.start_layer, inner.end_layer):
        layer = inner.layers[i]
        for norm in (layer.input_layernorm, layer.post_attention_layernorm):
            if (
                float(norm.variance_epsilon) != eps
                or norm.weight.dtype != torch.bfloat16
            ):
                return f"layer {i}: an RMSNorm with another epsilon or dtype"
    if float(inner.norm.variance_epsilon) != eps:
        return "the final norm's epsilon differs"
    return None


def build(model, *, deadline_s: float = DEADLINE_S, sm_count: Optional[int] = None,
          nvfp4: bool = True) -> LlHopFamily:  # fmt: skip
    """The hop family of ``model`` on its tensor-parallel group (not yet compiled:
    ``compile(widths)`` is collective). Raises with the reason if the model does
    not fit."""
    from sglang.srt.distributed.parallel_state import get_tp_group

    why = model_reason(model)
    if why is not None:
        raise ValueError(why)
    tp = get_tp_group()
    return LlHopFamily(
        device_group=tp.device_group,
        cpu_group=tp.cpu_group,
        eps=float(model.config.rms_norm_eps),
        hidden=HIDDEN,
        deadline_s=min(DEADLINE_S, deadline_s),
        sm_count=sm_count,
        nvfp4=nvfp4,
    )


# ============================================================== registry
def covers(profile, rows: int, mode: str) -> Optional[str]:
    """``ll.hops`` coverage: the TP4 target verify step up to its planned rows."""
    if mode != "target_verify":
        return f"{mode} steps are the draft's"
    if profile.tp != 4:
        return f"TP{profile.tp}: the LL hops serve TP4 (the symmetric-memory hops serve TP8)"
    from sgl_nvidia_mega.glm5_3.ll import target

    why = target.reason(profile)
    if why is not None:
        return why
    a = (profile.tp, rows, HIDDEN, profile.sms)
    if plain_plan(*a) is None or finalize_plan(*a) is None:
        return f"no hop plan at {rows} rows"
    return None


def _family(model: Any, derived: Any, profile: Any):
    """``FAMILIES["ll.hops"]``: the hop family of ``model`` on its TP group."""
    why = model_reason(model)
    if why is not None:
        return DeclinedFamily(why)
    family = build(model, deadline_s=profile.deadline_s, sm_count=profile.sms)
    family.model, family.target = model, profile.target
    family.sm, family.derived = tuple(profile.sm), derived
    if profile.target is not None:
        profile.target.note("ll.hops")
    return family


HOP_STEPS = ("pre_attention", "post_attention", "final")
CANDIDATES = (
    # P4 one- / two-shot hops on every boundary; P9: finalizes an ll.moe handoff
    # ("hops:finalize"); P10: the MoE layers' FC1 input as NVFP4, written when a taken
    # MLP unit consumes "mlp_input:nvfp4" (wire)
    Candidate(
        "ll.hops",
        "hops",
        HOP_STEPS,
        "ll.hops",
        covers,
        provides=("hops:finalize", "mlp_input:nvfp4"),
    ),
)
FAMILIES = {"ll.hops": _family}
