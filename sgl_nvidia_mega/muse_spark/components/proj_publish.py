# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Component ``proj_publish`` (env ``MUSE_VR_PROJ_PUBLISH=1``, default off): the MoE pooler / unpooler as published slices.

At tensor parallelism, SGLang's Muse Spark MoE block computes the pooler (``pre_expert_proj``) and the unpooler
(``post_expert_proj``) with the full replicated weight on every rank whenever its sliced paths are not admitted, which on
this device is every decode / verify call above the small-projection row cap. With this component each rank multiplies
its contiguous column block only, and the GEMM epilogue publishes the block into every rank's copy of the push-plane
gather slot (``ops/proj_publish.py``):

- site ``P`` (pooler; ``MUSE_VR_PROJ_PUBLISH_SITES`` default ``P``), rows ``MUSE_VR_PROJ_PUBLISH_ROWS`` (default ``56-336``):
  on the tensor-parallel decode / verify path of NVFP4 expert layers (per-token prepare path, no sequence parallelism),
  the pooler's ``bf16_gemm_dispatch`` call becomes ``publish_gemm`` and the experts' ``prepare_input`` of that output becomes
  ``pub_norm_fp4_quant`` (SGLang's RMS norm + per-token NVFP4 quantization of the gathered row);
- site ``U`` (unpooler; ``MUSE_VR_PROJ_PUBLISH_SITES=P,U``), rows ``MUSE_VR_PROJ_PUBLISH_U_ROWS`` (default ``192-288``):
  SGLang's deferred path is taken (``gemm_ag_supported`` admits this device for the unpooler only); its ``gemm_local`` call
  becomes ``publish_gemm`` and the FFN boundary's ``all_gather_norm_residual`` call becomes the consumer-only
  ``pub_ag_norm`` (SGLang's boundary arithmetic without its push of a local slice). Rows in
  ``MUSE_VR_PROJ_PUBLISH_UAG_ROWS`` (default none) take SGLang's own gathered unpooler path unchanged (sliced GEMM + its
  all-gather boundary), the form that measured faster than the publish at some row counts.

Each publish is consumed by its own consumer as the next use of the push plane (``ops/proj_publish.py`` refuses an
unpaired call, and a pooler publish not consumed inside its MoE call raises). Every decision depends only on rank-uniform
inputs (row count, shapes, layer kind, communicator), so every rank takes the same path. Everything else runs SGLang's
code unchanged; the reason is counted.

Numerics: the unpooler's TGV column-slice GEMM is bitwise equal to the replicated dispatch at the tested row counts (units),
and the published values reach the boundary through SGLang's all-gather marker rule (an exact +0.0 half arrives as -0.0),
as SGLang's own gathered path does. The pooler's slice GEMM differs from the replicated cuBLAS GEMM in summation order below
136 rows on this image (a reduction-order class) and is bitwise above (units).
"""

from __future__ import annotations

import logging
import os
from typing import Optional

import torch

from . import bed

# The SGLang names this module reads (test/unit/test_contract.py pins them against the supported SGLang)
SGLANG_READS = {
    "srt/models/muse_spark.py": ("MuseSpark1xDecoderLayer", "MuseSpark1xMoE"),
    "srt/layers/muse_spark/muse_spark_fp4_moe.py": ("MuseSpark1xFp4Experts",),
    "srt/runtime_context.py": ("get_parallel", "get_forward"),
}


logger = logging.getLogger(__name__)

NAME = "proj_publish"
ROWS_ENV = "MUSE_VR_PROJ_PUBLISH_ROWS"  # pooler (site P) rows
U_ROWS_ENV = "MUSE_VR_PROJ_PUBLISH_U_ROWS"  # unpooler (site U) rows
UAG_ROWS_ENV = "MUSE_VR_PROJ_PUBLISH_UAG_ROWS"  # unpooler rows on SGLang's own gathered path (no publish); default none
SITES_ENV = "MUSE_VR_PROJ_PUBLISH_SITES"
DEFAULT_ROWS = (56, 336)
DEFAULT_U_ROWS = (192, 288)
H, EH = 8192, 4096
# sha256 of the SGLang files whose names this component rebinds (306ccf6 layout)
PINS = {
    "sglang/srt/models/muse_spark.py": "8bafbb8802fb0d9f81c190e87b9012fca546182be08bef6c8b3db814d27bfe19",
}

_state = {
    "engaged": None,
    "rows": DEFAULT_ROWS,
    "u_rows": DEFAULT_U_ROWS,
    "uag_rows": None,
    "sites": ("P",),
    "markers": {},
    "allow_p": None,
}


def _range(env: str, default):
    v = os.environ.get(env, "").strip()
    if not v:
        return default
    lo, hi = (int(t) for t in v.split("-"))
    if not 1 <= lo <= hi <= 512:
        raise ValueError(f"{env}={v}: needs 1 <= lo <= hi <= 512")
    return lo, hi


def _sites():
    v = os.environ.get(SITES_ENV, "P").strip().upper()
    sites = tuple(s.strip() for s in v.split(",") if s.strip())
    if not sites or any(s not in ("U", "P") for s in sites):
        raise ValueError(f"{SITES_ENV}={v}: use P, U or P,U")
    return sites


def _in(rng, m: int) -> bool:
    return rng is not None and rng[0] <= m <= rng[1]


def _comm():
    from sglang.srt.runtime_context import get_parallel

    try:
        return get_parallel().tp_group.ca_comm
    except Exception:  # noqa: BLE001
        return None


def _marker(m: int, n: int, device, site: str) -> torch.Tensor:
    """A shape carrier for SGLang's code between a publish and its consumer (never read)."""
    t = torch.empty(m, n, dtype=torch.bfloat16, device=device)
    _state["markers"][t.data_ptr()] = site
    return t


def _take_marker(t: torch.Tensor, site: str) -> bool:
    return (
        _state["markers"].pop(t.data_ptr(), None) == site
        if isinstance(t, torch.Tensor)
        else False
    )


def install() -> None:
    import sglang.srt.models.muse_spark as model

    from .ops import proj_publish as pp

    sites = _state["sites"]
    orig_supported = model.gemm_ag_supported
    orig_local = model.gemm_local
    orig_agn = model.all_gather_norm_residual
    orig_dispatch = model.bf16_gemm_dispatch

    def publishable(comm, x, weight, shape, rows) -> Optional[str]:
        if (
            comm is None
            or x.dim() != 2
            or tuple(weight.shape) != shape
            or x.dtype != torch.bfloat16
        ):
            return "shape"
        if not _in(rows, x.shape[0]):
            return "rows"
        return pp.step_aside_reason(comm, x.shape[0], shape[0], shape[1])

    def gemm_ag_supported(comm, x, weight, *a, **k):
        if orig_supported(comm, x, weight, *a, **k):
            return True
        # the unpooler only (the pooler call site checks the device capability itself): published rows, and rows that
        # take SGLang's own gathered path unchanged
        if "U" not in sites:
            return False
        return (
            publishable(comm, x, weight, (H, EH), _state["u_rows"]) is None
            or publishable(comm, x, weight, (H, EH), _state["uag_rows"]) is None
        )

    def gemm_local(comm, x, weight):
        r = (
            publishable(comm, x, weight, (H, EH), _state["u_rows"])
            if "U" in sites
            else "site"
        )
        if r is not None:
            if (
                "U" in sites
                and publishable(comm, x, weight, (H, EH), _state["uag_rows"]) is None
            ):
                bed.count(NAME, f"U gathered M{x.shape[0]}")
            else:
                bed.count(NAME, f"U stock: {r}")
            return orig_local(comm, x, weight)
        pp.publish_gemm(comm, x.contiguous(), weight, "ag_norm")
        bed.count(NAME, f"U publish M{x.shape[0]}")
        return _marker(x.shape[0], H // comm.world_size, x.device, "U")

    def all_gather_norm_residual(comm, local, stream, *a, **k):
        if _take_marker(local, "U"):
            return pp.pub_ag_norm(comm, stream, *a, **k)
        return orig_agn(comm, local, stream, *a, **k)

    def bf16_gemm_dispatch(x, weight, bias, *a, **k):
        allow = _state["allow_p"]
        if allow is not None and bias is None and not a and not k and weight is allow:
            _state["allow_p"] = None  # one publish per admitted MoE call
            comm = _comm()
            r = publishable(comm, x, weight, (EH, H), _state["rows"])
            if r is None:
                pp.publish_gemm(comm, x.contiguous(), weight, "norm_fp4_quant")
                bed.count(NAME, f"P publish M{x.shape[0]}")
                bed.count(
                    NAME, f"P layer {_state.get('layer', -1)}"
                )  # engagement record names the target layers
                return _marker(x.shape[0], EH, x.device, "P")
            bed.count(NAME, f"P stock: {r}")
        return orig_dispatch(x, weight, bias, *a, **k)

    model.gemm_ag_supported = gemm_ag_supported
    model.gemm_local = gemm_local
    model.all_gather_norm_residual = all_gather_norm_residual
    if "U" in sites:
        orig_layer_forward = model.MuseSpark1xDecoderLayer.forward

        def layer_forward(self, *a, **k):
            out = orig_layer_forward(self, *a, **k)
            # fail-stop inside the same forward: an unpooler publish must be consumed by this layer's FFN boundary
            if pp.pending(_comm()) is not None:
                raise RuntimeError(
                    f"proj_publish: a publish was not consumed inside decoder layer "
                    f"{getattr(self, 'layer_idx', '?')}: {pp.pending(_comm())}"
                )
            return out

        model.MuseSpark1xDecoderLayer.forward = layer_forward
    if "P" in sites:
        model.bf16_gemm_dispatch = bf16_gemm_dispatch
        import sglang.srt.layers.muse_spark.muse_spark_fp4_moe as fp4

        orig_prepare = fp4.MuseSpark1xFp4Experts.prepare_input
        orig_experts = model.MuseSpark1xMoE._forward_experts

        def _forward_experts(
            self, hidden_states, router_input=None, *, ar_workspace=None
        ):
            # Site P is admitted only on the tensor-parallel decode / verify path (called from _forward_tp, which passes
            # router_input), NVFP4 experts with the per-token prepare path, no sequence parallelism, no marlin path.
            from sglang.srt.runtime_context import get_forward

            ok = (
                router_input is not None
                and self.pre_expert_proj is not None
                and self.nvfp4_checkpoint
                and isinstance(self.experts, fp4.MuseSpark1xFp4Experts)
                and not getattr(self.experts, "use_marlin", False)
                and getattr(
                    self.experts.quant_config, "use_per_token_activation", False
                )
                and not getattr(get_forward(), "sp_active", False)
            )
            if not ok:
                return orig_experts(
                    self, hidden_states, router_input, ar_workspace=ar_workspace
                )
            _state["allow_p"] = self.pre_expert_proj.weight
            _state["layer"] = getattr(self, "layer_idx", -1)
            try:
                return orig_experts(
                    self, hidden_states, router_input, ar_workspace=ar_workspace
                )
            finally:
                _state["allow_p"] = None
                if pp.pending(_comm()) is not None:
                    raise RuntimeError(
                        "proj_publish: a pooler publish was not consumed inside its MoE call"
                    )

        model.MuseSpark1xMoE._forward_experts = _forward_experts

        def prepare_input(self, hidden_states, pre_expert_norm=None):
            if _take_marker(hidden_states, "P"):
                if (
                    pre_expert_norm is None
                    or not self.quant_config.use_per_token_activation
                ):
                    raise RuntimeError(
                        "proj_publish: published pooler output needs the per-token NVFP4 prepare path"
                    )
                comm = _comm()
                fp4_out, sf, pts = pp.pub_norm_fp4_quant(
                    comm,
                    hidden_states.shape[0],
                    hidden_states.shape[1],
                    self._pre_norm_weight(pre_expert_norm),
                    pre_expert_norm.eps,
                    hidden_states.device,
                )
                hs_scale = sf.view(torch.float8_e4m3fn).reshape(*sf.shape[:-1], -1)
                return fp4_out, hs_scale, pts
            return orig_prepare(self, hidden_states, pre_expert_norm)

        fp4.MuseSpark1xFp4Experts.prepare_input = prepare_input


def register() -> bool:
    if _state["engaged"] is not None:
        return _state["engaged"]
    if not bed.enabled(NAME):
        _state["engaged"] = False
        return False
    for rel, sha in PINS.items():
        if not bed.pinned(rel, sha):
            bed.report(NAME, False, reason=f"pin mismatch: {rel}")
            _state["engaged"] = False
            return False
    try:
        _state["rows"] = _range(ROWS_ENV, DEFAULT_ROWS)
        _state["u_rows"] = _range(U_ROWS_ENV, DEFAULT_U_ROWS)
        _state["uag_rows"] = _range(UAG_ROWS_ENV, None)
        _state["sites"] = _sites()
        install()
    except Exception as e:  # noqa: BLE001
        bed.report(NAME, False, reason=f"install failed: {type(e).__name__}: {e}")
        _state["engaged"] = False
        return False
    uag = _state["uag_rows"]
    bed.report(
        NAME,
        True,
        p_rows="%d-%d" % _state["rows"],
        u_rows="%d-%d" % _state["u_rows"],
        uag_rows=("%d-%d" % uag) if uag else "none",
        sites=",".join(_state["sites"]),
    )
    _state["engaged"] = True
    return True
