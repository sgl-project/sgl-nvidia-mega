# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 SGLang Team
"""CuTe DSL all-reduce hops of the GLM-5.3 fused verify forward.

Every boundary at which the stock forward sums a tensor-parallel partial (the
``o_proj`` output, the MLP / MoE output) is served by one hop: a kernel
publishes this rank's partial to every rank's symmetric-memory mailbox and
reduces the eight partials in rank order, adds the residual and applies the
RMSNorm. The ``o_proj`` hop also computes the ``o_proj`` GEMV; on MoE layers it
computes the router logits during the all-reduce and routes the rows (top-8),
and the MLP-output hop of a MoE layer folds the MoE engine's per-expert outputs
(finalize) into the all-reduce.

Forms per verify row count (a function of the row count and the layer index
only, so every rank runs the same hop sequence on its mailboxes):

* MoE family served, rows <= 8: the routing ``o_proj`` hop and the fold hop on
  MoE layers; dense layers use the plain hops at rows <= 6, stock above.
* MoE family served, rows 9..16: the 16-row routing hop and fold on their own
  mailbox; dense layers stock.
* MoE family stock at this row count (declined at startup), rows <= 8 ("far
  form"): the plain hops on every boundary. Otherwise stock.

The kernels run only inside captured verify graphs; the mailboxes are created
and initialized before the first capture.
"""

from __future__ import annotations

import logging
from typing import Any, Optional

import torch
import torch.distributed as dist

from sgl_nvidia_mega.glm_dsa import switches
from sgl_nvidia_mega.glm_dsa.eligibility import HIDDEN, TP_SIZE
from sgl_nvidia_mega.glm_dsa.providers import (
    HopSteps,
    StepContext,
    compile_cached,
)

logger = logging.getLogger(__name__)

O_PROJ_IN = 2048
TOP_K = 8  # routed experts per row
SMALL_ROWS = 8  # mailbox instance A (instance B: 9..16 rows)
DENSE_MAX_ROWS = 6  # plain hops on the dense layers beside the fold; stock above
# Error sites: layer * 4 + {0: o_proj, 1: MLP output, 2: final}; sites from
# EXTERNAL_SITES on are other kernels' (named with ``HopFamily.register_sites``).
HOPS_PER_LAYER = 4
EXTERNAL_SITES = 1000


class HopFamily:
    """Derived weights, compiled units and mailboxes of the hop kernels."""

    def __init__(self, model: Any, derived: Any, api: Any) -> None:
        from sglang.srt.distributed.parallel_state import get_tp_group

        inner, config = model.model, model.config
        span = range(inner.start_layer, inner.end_layer)
        self.api, self.derived = api, derived
        self.layers = [inner.layers[i] for i in span]
        self.first, self.last = inner.start_layer, inner.end_layer - 1
        self.eps = float(config.rms_norm_eps)
        self.is_moe = [bool(layer.is_layer_sparse) for layer in self.layers]
        self.rscale = float(getattr(config, "routed_scaling_factor", 1.0))
        self.device = torch.device("cuda", torch.cuda.current_device())
        self.group = get_tp_group()
        self.rank = self.group.rank_in_group
        self.w_o = [layer.self_attn.o_proj.weight for layer in self.layers]
        self.router: list = []
        self.kernels: Any = None
        self.site_names: dict = {}
        # DNORM (both switches): the layers without their own top-k whose input
        # hop is the MoE fold may defer their attention input norm.
        dnorm = switches.dnorm()
        self.dnorm_layers = frozenset(
            layer.layer_id
            for i, layer in enumerate(self.layers)
            if dnorm
            and i > 0
            and self.is_moe[i - 1]
            and getattr(layer.self_attn, "skip_topk", False)
        )

    def check(self) -> str:
        """Empty if this model and environment fit the kernels, else the reason."""
        if self.group.world_size != TP_SIZE:
            size = self.group.world_size
            return f"tensor-parallel size {size} (the hops need {TP_SIZE})"
        for w in self.w_o:
            if w.dtype != torch.bfloat16 or tuple(w.shape) != (HIDDEN, O_PROJ_IN):
                return "o_proj weight is not a bf16 [6144, 2048] shard"
            if not w.is_contiguous():
                return "o_proj weight is not contiguous"
        for layer in self.layers:
            for norm in (layer.input_layernorm, layer.post_attention_layernorm):
                if float(norm.variance_epsilon) != self.eps:
                    return "the RMSNorm epsilons differ"
        for layer, moe in zip(self.layers, self.is_moe):
            bias = layer.mlp.gate.e_score_correction_bias if moe else None
            if moe and (bias is None or bias.dtype != torch.float32):
                return "the router has no fp32 score-correction bias"
        import torch.distributed._symmetric_memory as symm_mem

        get_backend = getattr(symm_mem, "get_backend", None)
        backend = get_backend(self.device) if get_backend is not None else None
        if backend not in (None, "CUDA"):
            return f"symmetric-memory backend {backend!r} (the hops need 'CUDA')"
        if not self.api.fenced_trigger_supported():
            return "the CuTe DSL toolchain does not provide the o_proj hop's start edge"
        return ""

    def compile(self, rows) -> None:
        """Compile the units served at ``rows`` and create the mailboxes.

        Collective over the tensor-parallel group: every rank agrees (CPU
        all-reduce) on whether the hops are served at all and on the split
        fold's row limit, so the hop sequence is the same on every rank.
        """
        reason = self.check()
        small = any(m <= SMALL_ROWS for m in rows)
        big = any(m > SMALL_ROWS for m in rows)
        names = set()
        if small:
            names |= {"allreduce_norm", "o_proj_produce", "o_proj_consume"}
            names |= {"o_proj_produce_router", "o_proj_consume_router"}
            names |= {"moe_fold_split", "moe_fold_plain"}
            if self.dnorm_layers:
                names |= {"moe_fold_split_dn", "moe_fold_plain_dn"}
        if big:
            names |= {"o_proj_produce_router16", "o_proj_consume_router16"}
            names |= {"moe_fold_block_sums16", "rmsnorm_from_block_sums16"}
        compiled, probe = {}, None
        if not reason:
            import cutlass.cute as cute

            from sgl_nvidia_mega.glm_dsa.kernels.cute import launch

            idx = self.device.index
            per_sm = switches.hop_ctas_per_sm()
            pad = launch.spread_smem(idx, per_sm) if per_sm else 0
            carveout = switches.hop_max_shared_carveout()
            specs = self.api.unit_specs(consume_pad_smem=pad)
            for spec in (s for s in specs if s.name in names):
                compiled[spec.name] = compile_cached(
                    spec.obj,
                    spec.constexprs,
                    lambda spec=spec: cute.compile(spec.obj, *spec.args),
                )
                padded = getattr(spec.obj, "pad_smem", 0)
                if carveout or padded:
                    launch.set_shared_memory_attributes(
                        compiled[spec.name],
                        idx,
                        max_carveout=carveout,
                        max_dynamic_smem=padded,
                    )
                if spec.name == "moe_fold_split" and not hasattr(
                    compiled[spec.name], "library"
                ):
                    # The occupancy query needs the driver library, which a
                    # cache-loaded unit may not keep: query a fresh compile.
                    probe = cute.compile(spec.obj, *spec.args)
            if carveout or pad:
                logger.info(
                    "GLM DSA fused decode: hops prefer the max shared carveout: %s; "
                    "o_proj consumer CTAs per SM: %s",
                    carveout,
                    f"<= {per_sm} ({pad} B launch shared memory)" if pad else "unpadded",
                )
        ok = torch.tensor([0 if reason else 1], dtype=torch.int32)
        dist.all_reduce(ok, op=dist.ReduceOp.MIN, group=self.group.cpu_group)
        if int(ok.item()) == 0:
            logger.warning(
                "GLM decode hops fall back to the stock all-reduce: %s",
                reason or "another rank declined",
            )
            return
        # Per MoE layer, the router folded into the o_proj hop: W_router scaled by
        # the post-attention norm weight, times this rank's o_proj shard, plus this
        # rank's slice of the scaled router, as bf16 hi / lo (derived copies).
        for layer, moe in zip(self.layers, self.is_moe):
            hoist = None
            if moe:
                hoist = self.api.router_hoist_weights(
                    layer.mlp.gate.weight,
                    layer.post_attention_layernorm.weight,
                    layer.self_attn.o_proj.weight,
                    self.rank,
                )
                for t in hoist:
                    self.derived.add("hops.router_hoist", t)
            self.router.append(hoist)
        kernels = self.api.HopKernels.build(
            compiled,
            self.group.device_group.group_name,
            self.device,
            self.eps,
            self.rscale,
            self.device.index,
            instance_b=big,
            router=True,
            probe=probe,
        )
        split = torch.tensor([kernels.split_max_rows], dtype=torch.int32)
        dist.all_reduce(split, op=dist.ReduceOp.MIN, group=self.group.cpu_group)
        kernels.split_max_rows = int(split.item())
        self.kernels = kernels

    def register_sites(self, names: dict) -> None:
        """Name other kernels' sites (>= ``EXTERNAL_SITES``) in this error record."""
        for site in names:
            if int(site) < EXTERNAL_SITES:
                raise ValueError(f"error site {site} is in the hop family's range")
        self.site_names.update({int(s): str(n) for s, n in names.items()})

    def error_report(self) -> Optional[dict]:
        """The decoded host-mapped error record of a timed-out hop wait, or None
        (host memory only: readable after the CUDA context is lost)."""
        rec = self.kernels.err.read() if self.kernels is not None else None
        if rec is None:
            return None
        site = int(rec["site"])
        if site >= EXTERNAL_SITES:
            return {**rec, "layer": None, "hop": self.site_names.get(site, "external")}
        layer, kind = divmod(site, HOPS_PER_LAYER)
        hop = ("o_proj", "mlp_output", "final")[kind] if kind < 3 else str(kind)
        return {**rec, "layer": layer, "hop": hop}

    def for_rows(self, rows: int) -> Optional[HopSteps]:
        """The hop step group at ``rows`` (None: stock): the fold form, whose
        ``fallback`` (the far form up to ``SMALL_ROWS`` rows, else stock) the
        composition serves where the MoE family is stock at ``rows``."""
        if self.kernels is None:
            return None
        far = _FarHops(self) if rows <= SMALL_ROWS else None
        return _FoldHops(self, rows, fallback=far)

    def site(self, layer_id: int, kind: int) -> int:
        return layer_id * HOPS_PER_LAYER + kind

    def o_proj(self, layer, o_input, residual, step: Optional[StepContext] = None):
        """The ``o_proj`` hop; with ``step`` (MoE layers) the routing variant, whose
        expert ids / weights ([rows, 8] int32 / fp32) go to ``step.routing``."""
        i, site = layer.layer_id - self.first, self.site(layer.layer_id, 0)
        out_res, out_norm = torch.empty_like(residual), torch.empty_like(residual)
        tail = (residual, layer.post_attention_layernorm.weight, out_res, out_norm)
        if step is None:
            self.kernels.o_proj(o_input, self.w_o[i], *tail, site=site)
            return out_norm, out_res
        rows, dev = residual.shape[0], residual.device
        ids = torch.empty((rows, TOP_K), dtype=torch.int32, device=dev)
        wts = torch.empty((rows, TOP_K), dtype=torch.float32, device=dev)
        bias = layer.mlp.gate.e_score_correction_bias
        self.kernels.o_proj_router(
            o_input, self.w_o[i], *self.router[i], bias, *tail, ids, wts, site=site
        )
        step.routing = (ids, wts)
        return out_norm, out_res

    def allreduce_norm(self, partial, residual, gamma, site: int):
        out_res, out_norm = torch.empty_like(residual), torch.empty_like(residual)
        self.kernels.allreduce_norm(
            partial, residual, gamma, out_res, out_norm, site=site
        )
        return out_norm, out_res

    def fold(self, rec, residual, gamma, site: int, dn: bool = False):
        """The fold hop of the MoE records: ``(normed, residual out)``; with ``dn``
        (<= 8 rows) the norm is deferred: ``(u = bf16(x * gamma), residual out,
        block sums [rows, 8] fp32)``."""
        rows = residual.shape[0]
        out_res, out = torch.empty_like(residual), torch.empty_like(residual)
        bsum = None
        if dn or rows > SMALL_ROWS:
            bsum = torch.empty((rows, 8), dtype=torch.float32, device=residual.device)
        args = (rec.pq, rec.shpartials, rec.ids_fin, rec.w_fin, rec.g2s, residual)
        if dn:
            self.kernels.moe_fold_dn(*args, gamma, out_res, out, bsum, site=site)
            return out, out_res, bsum
        self.kernels.moe_fold(*args, gamma, out_res, out, bsum=bsum, site=site)
        return out, out_res


class _FarHops(HopSteps):
    """Rows <= 8 with the stock MLP / MoE: every boundary is a hop."""

    def __init__(self, family: HopFamily) -> None:
        self.f = family

    def pre_attention(self, layer, mlp_out, residual, step: StepContext):
        if residual is None:  # first layer: the embedding
            return super().pre_attention(layer, mlp_out, residual, step)
        site = self.f.site(layer.layer_id, 1)
        gamma = layer.input_layernorm.weight
        return self.f.allreduce_norm(mlp_out, residual, gamma, site)

    def post_attention(self, layer, o_input, residual, step: StepContext):
        return self.f.o_proj(layer, o_input, residual)

    def final(self, model, mlp_out, residual, step: StepContext) -> torch.Tensor:
        site = self.f.site(self.f.last, 2)
        gamma = model.model.norm.weight
        return self.f.allreduce_norm(mlp_out, residual, gamma, site)[0]


class _FoldHops(HopSteps):
    """The MoE family is served: routing ``o_proj`` hops and folds on MoE
    layers, plain hops on dense layers up to ``DENSE_MAX_ROWS`` rows. Requires
    the MoE family at the same row count; ``fallback`` is served where it is
    stock."""

    requires = ("mlp",)

    def __init__(
        self, family: HopFamily, rows: int, fallback: Optional[HopSteps]
    ) -> None:
        self.f, self.rows, self.fallback = family, rows, fallback
        self.dense = rows <= DENSE_MAX_ROWS
        self.dnorm_layers = frozenset()

    def defer_input_norm(self) -> None:
        """Serve DNORM on the family's DNORM layers; called by the composition
        when this row count's attention consumes it."""
        if self.rows <= SMALL_ROWS:
            self.dnorm_layers = self.f.dnorm_layers
            logger.info(
                "GLM DSA fused decode: deferred input norm served on %d layers at %d "
                "rows",
                len(self.dnorm_layers),
                self.rows,
            )

    def pre_attention(self, layer, mlp_out, residual, step: StepContext):
        if residual is None:
            return super().pre_attention(layer, mlp_out, residual, step)
        site = self.f.site(layer.layer_id, 1)
        gamma = layer.input_layernorm.weight
        if not isinstance(mlp_out, torch.Tensor):
            if layer.layer_id in self.dnorm_layers:
                u, out_res, bsum = self.f.fold(mlp_out, residual, gamma, site, dn=True)
                step.dnorm = (layer.layer_id, bsum)
                return u, out_res
            return self.f.fold(mlp_out, residual, gamma, site)
        if self.dense:
            return self.f.allreduce_norm(mlp_out, residual, gamma, site)
        return super().pre_attention(layer, mlp_out, residual, step)

    def post_attention(self, layer, o_input, residual, step: StepContext):
        if self.f.is_moe[layer.layer_id - self.f.first]:
            return self.f.o_proj(layer, o_input, residual, step)
        if self.dense:
            return self.f.o_proj(layer, o_input, residual)
        return super().post_attention(layer, o_input, residual, step)

    def final(self, model, mlp_out, residual, step: StepContext) -> torch.Tensor:
        site = self.f.site(self.f.last, 2)
        gamma = model.model.norm.weight
        if not isinstance(mlp_out, torch.Tensor):
            return self.f.fold(mlp_out, residual, gamma, site)[0]
        if self.dense:
            return self.f.allreduce_norm(mlp_out, residual, gamma, site)[0]
        return super().final(model, mlp_out, residual, step)


def build(model: Any, derived: Any) -> Optional[HopFamily]:
    """The hop family of ``model``, or None to keep every boundary stock."""
    # Defensive: hop kernel modules that fail to import keep every boundary on the
    # stock all-reduce (the attention / MoE kernels do not need them).
    try:
        from sgl_nvidia_mega.glm_dsa.kernels.cute.hop import api
    except ImportError as e:
        logger.warning(
            "GLM decode hops unavailable (%s); using the stock all-reduce.", e
        )
        return None
    return HopFamily(model, derived, api)
